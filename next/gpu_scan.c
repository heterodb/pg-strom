/*
 * gpu_scan.c
 *
 * Sequential scan accelerated with GPU processors
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/* static variables */
static set_rel_pathlist_hook_type set_rel_pathlist_next = NULL;
static CustomPathMethods	gpuscan_path_methods;
static CustomScanMethods	gpuscan_plan_methods;
static CustomExecMethods	gpuscan_exec_methods;
static bool					enable_gpuscan;		/* GUC */
static bool					enable_pullup_outer_scan;	/* GUC */

/*
 * form/deform private information of CustomScan(GpuScan)
 */
typedef struct
{
	const Bitmapset *gpu_cache_devs; /* device for GpuCache, if any */
	const Bitmapset *gpu_direct_devs; /* device for GPU-Direct SQL, if any */
	bytea		   *kern_quals;		/* device qualifiers */
	bytea		   *kern_projs;		/* device projection */
	uint32_t		extra_flags;
	uint32_t		extra_bufsz;
	uint32_t		kvars_nslots;
	uint32_t		nrows_per_block;	/* average # rows per block */
	const Bitmapset *outer_refs;	/* referenced columns */
	List		   *used_params;	/* Param list in use */
	List		   *dev_quals;		/* Device qualifiers */
	Oid				index_oid;		/* OID of BRIN-index, if any */
	List		   *index_conds;	/* BRIN-index key conditions */
	List		   *index_quals;	/* Original BRIN-index qualifier*/
} GpuScanInfo;

static void
form_gpuscan_info(CustomScan *cscan, GpuScanInfo *gs_info)
{
	List	   *privs = NIL;
	List	   *exprs = NIL;

	privs = lappend(privs, bms_to_pglist(gs_info->gpu_cache_devs));
	privs = lappend(privs, bms_to_pglist(gs_info->gpu_direct_devs));
	privs = lappend(privs, makeConst(BYTEAOID,
									 -1,
									 InvalidOid,
									 -1,
									 PointerGetDatum(gs_info->kern_quals),
									 (gs_info->kern_quals == NULL),
									 false));
	privs = lappend(privs, makeConst(BYTEAOID,
									 -1,
									 InvalidOid,
									 -1,
									 PointerGetDatum(gs_info->kern_projs),
									 (gs_info->kern_projs == NULL),
									 false));
	privs = lappend(privs, makeInteger(gs_info->extra_flags));
	privs = lappend(privs, makeInteger(gs_info->extra_bufsz));
	privs = lappend(privs, makeInteger(gs_info->kvars_nslots));
	privs = lappend(privs, makeInteger(gs_info->nrows_per_block));
	privs = lappend(privs, bms_to_pglist(gs_info->outer_refs));
	exprs = lappend(exprs, gs_info->used_params);
	exprs = lappend(exprs, gs_info->dev_quals);
	privs = lappend(privs, makeInteger(gs_info->index_oid));
	exprs = lappend(exprs, gs_info->index_conds);
	exprs = lappend(exprs, gs_info->index_quals);

	cscan->custom_private = privs;
	cscan->custom_exprs = exprs;
}

static void
deform_gpuscan_info(GpuScanInfo *gs_info, CustomScan *cscan)
{
	List	   *privs = cscan->custom_private;
	List	   *exprs = cscan->custom_exprs;
	int			pindex = 0;
	int			eindex = 0;
	Const	   *con;

	gs_info->gpu_cache_devs = bms_from_pglist(list_nth(privs, pindex++));
	gs_info->gpu_direct_devs = bms_from_pglist(list_nth(privs, pindex++));
	con = list_nth(privs, pindex++);
	if (!con->constisnull)
		gs_info->kern_quals	= DatumGetByteaP(con->constvalue);
	con = list_nth(privs, pindex++);
	if (!con->constisnull)
		gs_info->kern_projs = DatumGetByteaP(con->constvalue);
	gs_info->extra_flags	= intVal(list_nth(privs, pindex++));
	gs_info->extra_bufsz	= intVal(list_nth(privs, pindex++));
	gs_info->kvars_nslots   = intVal(list_nth(privs, pindex++));
	gs_info->nrows_per_block = intVal(list_nth(privs, pindex++));
	gs_info->outer_refs		= bms_from_pglist(list_nth(privs, pindex++));
	gs_info->used_params	= list_nth(exprs, eindex++);
	gs_info->dev_quals		= list_nth(exprs, eindex++);
	gs_info->index_oid		= intVal(list_nth(privs, pindex++));
	gs_info->index_conds	= list_nth(exprs, eindex++);
	gs_info->index_quals	= list_nth(exprs, eindex++);
}

/*
 * GpuScanSharedState
 */
typedef struct
{
	/* for arrow_fdw file scan */
	pg_atomic_uint32	af_rbatch_index;
	pg_atomic_uint32	af_rbatch_nload;	/* # of loaded record-batches */
	pg_atomic_uint32	af_rbatch_nskip;	/* # of skipped record-batches */
	/* for gpu_cache cache scan */
	pg_atomic_uint32	gc_fetch_count;
	/* for block-based regular table scan */
	BlockNumber			pbs_nblocks;		/* # blocks in relation at start of scan */
	slock_t				pbs_mutex;			/* lock of the fields below */
	BlockNumber			pbs_startblock;		/* starting block number */
	BlockNumber			pbs_nallocated;		/* # of blocks allocated to workers */
	/* common parallel table scan descriptor */
	ParallelTableScanDescData phscan;
} GpuScanSharedState;

/*
 * GpuScanState
 */
typedef struct
{
	pgstromTaskState	pts;
	GpuScanInfo			gs_info;
	/* for CPU fallbacks */
	ExprState		   *dev_quals;
	TupleTableSlot	   *base_slot;	/* base relation */
	ProjectionInfo	   *base_proj;	/* base --> tlist_dev projection */
} GpuScanState;








/*
 * create_gpuscan_path - constructor of CustomPath(GpuScan) node
 */
static CustomPath *
create_gpuscan_path(PlannerInfo *root,
					RelOptInfo *baserel,
					List *dev_quals,
					List *host_quals,
					bool parallel_aware,	/* for parallel-scan */
					IndexOptInfo *indexOpt,	/* for BRIN-index */
					List *indexConds,		/* for BRIN-index */
					List *indexQuals,		/* for BRIN-index */
					int64_t indexNBlocks)	/* for BRIN-index */
{
	GpuScanInfo	   *gs_info = palloc0(sizeof(GpuScanInfo));
	CustomPath	   *cpath = makeNode(CustomPath);
	ParamPathInfo  *param_info;
	int				parallel_nworkers = 0;
	double			parallel_divisor = 1.0;
	double			gpu_ratio = pgstrom_gpu_operator_cost / cpu_operator_cost;
	double			brin_ratio = 1.0;
	Cost			startup_cost = pgstrom_gpu_setup_cost;
	Cost			startup_delay = 0.0;
	Cost			run_cost = 0.0;
	Cost			disk_scan_cost = 0.0;
	double			selectivity;
	double			spc_seq_page_cost;
	double			spc_rand_page_cost;
	QualCost		qcost;
	double			ntuples;
	double			nchunks;
	int				nrows_per_block = 0;

	/* CPU Parallel parameters */
	if (parallel_aware)
	{
		double	leader_contribution;

		parallel_nworkers = compute_parallel_worker(baserel,
													baserel->pages, -1,
													max_parallel_workers_per_gather);
		if (parallel_nworkers <= 0)
			return NULL;
		parallel_divisor = (double)parallel_nworkers;
		if (parallel_leader_participation)
		{
			leader_contribution = 1.0 - (0.3 * (double)parallel_nworkers);
			if (leader_contribution > 0.0)
				parallel_divisor += leader_contribution;
		}
	}
	/* Has GpuCache? */
//	gs_info->gpu_cache_devs = baseRelHasGpuCache(root, baserel);
	/* Can use GPU-Direct SQL? */
	gs_info->gpu_direct_devs = baseRelCanUseGpuDirect(root, baserel);
	/* cost of full-disk scan */
	get_tablespace_page_costs(baserel->reltablespace,
							  &spc_rand_page_cost,
							  &spc_seq_page_cost);
	disk_scan_cost = spc_seq_page_cost * baserel->pages;

	/* consideration of BRIN-index, if any */
	if (indexOpt)
	{
		BrinStatsData	statsData;
		Relation		indexRel;
		Cost			index_scan_cost;
		double			index_nitems;
		ListCell	   *lc;

		indexRel = index_open(indexOpt->indexoid, AccessShareLock);
		brinGetStats(indexRel, &statsData);
		index_close(indexRel, AccessShareLock);

		get_tablespace_page_costs(indexOpt->reltablespace,
								  &spc_rand_page_cost,
								  &spc_seq_page_cost);
		index_scan_cost = spc_rand_page_cost * statsData.revmapNumPages;
		index_nitems = ceil(baserel->pages / (double)statsData.pagesPerRange);
		foreach (lc, indexQuals)
		{
			cost_qual_eval_node(&qcost, (Node *)lfirst(lc), root);
			index_scan_cost += (qcost.startup +
								qcost.per_tuple * index_nitems);
		}
		index_scan_cost += spc_seq_page_cost * indexNBlocks;
		if (disk_scan_cost > index_scan_cost)
		{
			disk_scan_cost = index_scan_cost;
			if (baserel->pages > 0)
				brin_ratio = (double)indexNBlocks / (double)baserel->pages;
		}
		else
			indexOpt = NULL;	/* disables BRIN-index if no benefit */
	}

	/*
	 * Cost discount for more efficient I/O with multiplexing.
	 * PG background workers can issue read request to filesystem
	 * concurrently. It enables to work I/O subsystem during blocking-
	 * time for other workers, then, it pulls up usage ratio of the
	 * storage system.
	 */
	if (parallel_nworkers > 0)
		disk_scan_cost /= Min(2.0, sqrt(parallel_divisor));

	/*
	 * Discount dist_scan_cost if we can use GPU-Direct SQL on the source
	 * table. It offers much much efficient i/o subsystem to load database
	 * blocks to GPU device.
	 */
	if (gs_info->gpu_direct_devs)
		disk_scan_cost /= 2.0;
	if (!gs_info->gpu_cache_devs)
		run_cost += disk_scan_cost;

	/*
	 * Size estimation
	 */
	selectivity = clauselist_selectivity(root,
										 dev_quals,
										 baserel->relid,
										 JOIN_INNER,
										 NULL);
	nchunks = ceil((double)baserel->pages /
				   (double)(PGSTROM_CHUNK_SIZE / BLCKSZ - 1));
	if (baserel->pages > 0)
		nrows_per_block = ceil(baserel->tuples / (double)baserel->pages);

	/* Cost for GPU qualifiers */
	ntuples = baserel->tuples * brin_ratio;
	cost_qual_eval_node(&qcost, (Node *)dev_quals, root);
	startup_cost += qcost.startup;
	run_cost += qcost.per_tuple * gpu_ratio * ntuples;
	ntuples *= selectivity;		/* rows after dev_quals */

	/* Cost for DMA receive (GPU-->Host) */
	run_cost += pgstrom_gpu_dma_cost * ntuples;

	/* cost for CPU qualifiers */
	cost_qual_eval(&qcost, host_quals, root);
	startup_cost += qcost.startup;
	run_cost += qcost.per_tuple * ntuples;

	/* PPI costs (as a part of host quals, if any) */
	param_info = get_baserel_parampathinfo(root, baserel,
										   baserel->lateral_relids);
	if (param_info)
	{
		List   *ppi_quals = param_info->ppi_clauses;

		cost_qual_eval(&qcost, ppi_quals, root);
		startup_cost += qcost.startup;
		run_cost += qcost.per_tuple * ntuples;
	}

	/* Cost for Projection */
	startup_cost += baserel->reltarget->cost.startup;
	run_cost += baserel->reltarget->cost.per_tuple * baserel->rows;

	/* Latency to get the first chunk */
	if (nchunks > 0)
		startup_delay = run_cost * (1.0 / nchunks);

	/* setup GpuScanInfo (Path phase) */
	gs_info->nrows_per_block = nrows_per_block;
	if (indexOpt)
	{
		gs_info->index_oid = indexOpt->indexoid;
		gs_info->index_conds = indexConds;
		gs_info->index_quals = indexQuals;
	}
	/* setup CustomPath */
	cpath->path.pathtype = T_CustomScan;
    cpath->path.parent = baserel;
    cpath->path.pathtarget = baserel->reltarget;
    cpath->path.param_info = param_info;
    cpath->path.parallel_aware = parallel_nworkers > 0 ? true : false;
    cpath->path.parallel_safe = baserel->consider_parallel;
    cpath->path.parallel_workers = parallel_nworkers;
	if (param_info)
		cpath->path.rows = param_info->ppi_rows / parallel_divisor;
	else
		cpath->path.rows = baserel->rows / parallel_divisor;
	cpath->path.startup_cost = startup_cost + startup_delay;
	cpath->path.total_cost = startup_cost + run_cost;
	cpath->path.pathkeys = NIL; /* unsorted results */
	cpath->flags = 0;
	cpath->custom_paths = NIL;
	cpath->custom_private = list_make1(gs_info);
	cpath->methods = &gpuscan_path_methods;

	return cpath;
}

/*
 * GpuScanAddScanPath
 */
static void
GpuScanAddScanPath(PlannerInfo *root,
				   RelOptInfo *baserel,
				   Index rtindex,
				   RangeTblEntry *rte)
{
	CustomPath *cpath;
	List	   *dev_quals = NIL;
	List	   *host_quals = NIL;
	IndexOptInfo *indexOpt;
	List	   *indexConds;
	List	   *indexQuals;
	int64_t		indexNBlocks;
	ListCell   *lc;

	/* call the secondary hook */
	if (set_rel_pathlist_next)
		set_rel_pathlist_next(root, baserel, rtindex, rte);
	/* nothing to do, if either PG-Strom or GpuScan is not enabled */
	if (!pgstrom_enabled || !enable_gpuscan)
		return;
	/* We already proved the relation empty, so nothing more to do */
	if (is_dummy_rel(baserel))
		return;
	/* It is the role of built-in Append node */
	if (rte->inh)
		return;
	/* GpuScan can run on heap relations or arrow_fdw table */
	switch (rte->relkind)
	{
		case RELKIND_RELATION:
		case RELKIND_MATVIEW:
			if (get_relation_am(rte->relid, true) != HEAP_TABLE_AM_OID)
				return;
			break;

		case RELKIND_FOREIGN_TABLE:
#if 0
			if (!baseRelIsArrowFdw(baserel))
				return;
			break;
#else
			return;
#endif
		default:
			/* not supported */
			return;
	}

	/* Check whether the qualifier can run on GPU device */
	foreach (lc, baserel->baserestrictinfo)
	{
		RestrictInfo *rinfo = lfirst(lc);
		List	   *input_rels_tlist = list_make1(makeInteger(baserel->relid));

		if (pgstrom_gpu_expression(rinfo->clause, input_rels_tlist, NULL))
			dev_quals = lappend(dev_quals, rinfo);
		else
			host_quals = lappend(host_quals, rinfo);
	}
	if (dev_quals == NIL)
		return;

	/* Check opportunity of GpuScan+BRIN-index */
	indexOpt = pgstrom_tryfind_brinindex(root, baserel,
										 &indexConds,
										 &indexQuals,
										 &indexNBlocks);
	/* add GpuScan path in single process */
	cpath = create_gpuscan_path(root, baserel,
								dev_quals,
								host_quals,
								false,
								indexOpt,
								indexConds,
								indexQuals,
								indexNBlocks);
	if (cpath && custom_path_remember(root, baserel, false, false, cpath))
		add_path(baserel, &cpath->path);
	/* If appropriate, consider parallel GpuScan */
	if (baserel->consider_parallel && baserel->lateral_relids == NULL)
	{
		cpath = create_gpuscan_path(root, baserel,
									dev_quals,
									host_quals,
									true,
									indexOpt,
									indexConds,
									indexQuals,
									indexNBlocks);
		if (cpath && custom_path_remember(root, baserel, true, false, cpath))
			add_partial_path(baserel, &cpath->path);
	}
}

/*
 * gpuscan_build_projection - make custom_scan_tlist
 */
typedef struct
{
	List	   *tlist_dev;
	List	   *input_rels_tlist;
	bool		resjunk;
} build_projection_context;

static bool
__gpuscan_build_projection_walker(Node *node, void *__priv)
{
	build_projection_context *context = __priv;
	ListCell   *lc;

	if (!node)
		return false;
	foreach (lc, context->tlist_dev)
	{
		TargetEntry	   *tle = lfirst(lc);

		if (equal(node, tle->expr))
			return false;
	}
	if (IsA(node, Var) ||
		pgstrom_gpu_expression((Expr *)node, context->input_rels_tlist, NULL))
	{
		AttrNumber		resno = list_length(context->tlist_dev) + 1;
		TargetEntry	   *tle = makeTargetEntry((Expr *)node,
											  resno,
											  NULL,
											  context->resjunk);
		context->tlist_dev = lappend(context->tlist_dev, tle);
		return false;
	}
	return expression_tree_walker(node, __gpuscan_build_projection_walker, __priv);
}

static List *
gpuscan_build_projection(RelOptInfo *baserel,
						 List *tlist,
						 List *host_quals,
						 List *dev_quals,
						 List *input_rels_tlist)
{
	build_projection_context context;
	List	   *vars_list;
	ListCell   *lc;

	memset(&context, 0, sizeof(build_projection_context));
	context.input_rels_tlist = input_rels_tlist;

	if (tlist != NIL)
	{
		foreach (lc, tlist)
		{
			TargetEntry *tle = lfirst(lc);

			if (IsA(tle->expr, Const) || IsA(tle->expr, Param))
				continue;
			__gpuscan_build_projection_walker((Node *)tle->expr, &context);
		}
	}
	else
	{
		/*
		 * When ProjectionPath is on CustomPath(GpuScan), it always assigns
		 * the result of build_path_tlist() and calls PlanCustomPath method
		 * with tlist == NIL.
		 * So, if GPU projection wants to make something valuable, we need
		 * to check path-target.
		 * Also don't forget all the Var-nodes to be added must exist at
		 * the custom_scan_tlist because setrefs.c references this list.
		 */
		foreach (lc, baserel->reltarget->exprs)
		{
			Node   *node = lfirst(lc);

			if (IsA(node, Const) || IsA(node, Param))
				continue;
			__gpuscan_build_projection_walker(node, &context);
        }
	}
	vars_list = pull_vars_of_level((Node *)host_quals, 0);
	foreach (lc, vars_list)
		__gpuscan_build_projection_walker((Node *)lfirst(lc), &context);

	context.resjunk = true;
	vars_list = pull_vars_of_level((Node *)dev_quals, 0);
	foreach (lc, vars_list)
		__gpuscan_build_projection_walker((Node *)lfirst(lc), &context);

	return context.tlist_dev;
}

/*
 * PlanGpuScanPath
 */
static Plan *
PlanGpuScanPath(PlannerInfo *root,
				RelOptInfo *baserel,
				CustomPath *best_path,
				List *tlist,
				List *clauses,
				List *custom_children)
{
	GpuScanInfo	   *gs_info = linitial(best_path->custom_private);
	CustomScan	   *cscan;
	List		   *host_quals = NIL;
	List		   *dev_quals = NIL;
	List		   *dev_costs = NIL;
	List		   *tlist_dev = NIL;
	List		   *input_rels_tlist = list_make1(makeInteger(baserel->relid));
	Bitmapset	   *outer_refs = NULL;
	uint32_t		qual_extra_bufsz = 0;
	uint32_t		proj_extra_bufsz = 0;
	uint32_t		qual_kvars_nslots = 0;
	uint32_t		proj_kvars_nslots = 0;
	int				j, k;
	ListCell	   *cell;

	/* sanity checks */
	Assert(baserel->relid > 0 &&
		   baserel->rtekind == RTE_RELATION &&
		   custom_children == NIL);
	/* check referenced columns */
	for (j=baserel->min_attr; j <= baserel->max_attr; j++)
	{
		if (!baserel->attr_needed[j - baserel->min_attr])
			continue;
		k = j - FirstLowInvalidHeapAttributeNumber;
		outer_refs = bms_add_member(outer_refs, k);
	}

	/*
	 * Distribution of clauses into device executable and others.
	 */
	foreach (cell, clauses)
	{
		RestrictInfo *rinfo = lfirst(cell);
		int		devcost;

		Assert(exprType((Node *)rinfo->clause) == BOOLOID);
		if (pgstrom_gpu_expression(rinfo->clause, input_rels_tlist, &devcost))
		{
			ListCell   *lc1, *lc2;
			int			pos = 0;

			forboth (lc1, dev_quals,
					 lc2, dev_costs)
			{
				if (devcost < lfirst_int(lc2))
				{
					dev_quals = list_insert_nth(dev_quals, pos, rinfo);
					dev_costs = list_insert_nth_int(dev_quals, pos, devcost);
					break;
				}
				pos++;
			}
			if (!lc1 && !lc2)
			{
				dev_quals = lappend(dev_quals, rinfo);
				dev_costs = lappend_int(dev_costs, devcost);
			}
		}
		else
		{
			host_quals = lappend(host_quals, rinfo);
		}
		pull_varattnos((Node *)rinfo->clause, baserel->relid, &outer_refs);
	}
	if (dev_quals == NIL)
		elog(ERROR, "GpuScan: Bug? no device executable qualifiers are given");
	dev_quals = extract_actual_clauses(dev_quals, false);
	host_quals = extract_actual_clauses(host_quals, false);

	/* code generation for WHERE-clause */
	pgstrom_build_xpucode(&gs_info->kern_quals,
						  (Expr *)dev_quals,
						  input_rels_tlist,
						  &gs_info->extra_flags,
						  &qual_extra_bufsz,
						  &qual_kvars_nslots,
						  &gs_info->used_params);
	/* code generation for the Projection */
	tlist_dev = gpuscan_build_projection(baserel,
										 tlist,
										 host_quals,
										 dev_quals,
										 input_rels_tlist);
	pgstrom_build_projection(&gs_info->kern_projs,
							 tlist_dev,
							 input_rels_tlist,
							 &gs_info->extra_flags,
							 &proj_extra_bufsz,
							 &proj_kvars_nslots,
							 &gs_info->used_params);
	gs_info->extra_bufsz = Max(qual_extra_bufsz,
							   proj_extra_bufsz);
	gs_info->kvars_nslots = Max(qual_kvars_nslots,
								proj_kvars_nslots);
	gs_info->dev_quals = dev_quals;
	gs_info->outer_refs = outer_refs;
	
	/*
	 * Build CustomScan(GpuScan) node
	 */
	cscan = makeNode(CustomScan);
	cscan->scan.plan.targetlist = tlist;
	cscan->scan.plan.qual = host_quals;
	cscan->scan.scanrelid = baserel->relid;
	cscan->flags = best_path->flags;
	cscan->methods = &gpuscan_plan_methods;
	cscan->custom_plans = NIL;
	cscan->custom_scan_tlist = tlist_dev;

	form_gpuscan_info(cscan, gs_info);

	return &cscan->scan.plan;
}

/*
 * CreateGpuScanState
 */
static Node *
CreateGpuScanState(CustomScan *cscan)
{
	GpuScanState   *gss = palloc0(sizeof(GpuScanState));

	/* Set tag and executor callbacks */
	NodeSetTag(gss, T_CustomScanState);
	gss->pts.css.flags = cscan->flags;
	Assert(cscan->methods == &gpuscan_plan_methods);
	gss->pts.css.methods = &gpuscan_exec_methods;
	deform_gpuscan_info(&gss->gs_info, cscan);

	return (Node *)gss;
}

/*
 * ExecInitGpuScan
 */
static void
ExecInitGpuScan(CustomScanState *node, EState *estate, int eflags)
{
	GpuScanState   *gss = (GpuScanState *)node;
	CustomScan	   *cscan = (CustomScan *)node->ss.ps.plan;
	Relation		rel = gss->pts.css.ss.ss_currentRelation;
	TupleDesc		tupdesc;
	List		   *dev_quals;
	List		   *tlist_dev = NIL;
	ListCell	   *lc;


	/* sanity checks */
	Assert(rel != NULL &&
		   outerPlanState(node) == NULL &&
		   innerPlanState(node) == NULL);

	/*
	 * Re-initialization of scan tuple-descriptor and projection-info,
	 * because commit 1a8a4e5cde2b7755e11bde2ea7897bd650622d3e of
	 * PostgreSQL makes to assign result of ExecTypeFromTL() instead
	 * of ExecCleanTypeFromTL; that leads incorrect projection.
	 * So, we try to remove junk attributes from the scan-descriptor.
	 *
	 * And, device projection returns a tuple in heap-format, so we
	 * prefer TTSOpsHeapTuple, instead of the TTSOpsVirtual.
	 */
	tupdesc = ExecCleanTypeFromTL(cscan->custom_scan_tlist);
	ExecInitScanTupleSlot(estate, &gss->pts.css.ss, tupdesc,
						  &TTSOpsHeapTuple);
	ExecAssignScanProjectionInfoWithVarno(&gss->pts.css.ss, INDEX_VAR);

	/*
	 * Init resources for CPU fallbacks
	 */
	dev_quals = (List *)
		fixup_varnode_to_origin((Node *)gss->gs_info.dev_quals,
								cscan->custom_scan_tlist);
	gss->dev_quals = ExecInitQual(dev_quals, &gss->pts.css.ss.ps);
	foreach (lc, cscan->custom_scan_tlist)
	{
		TargetEntry *tle = lfirst(lc);

		if (!tle->resjunk)
			tlist_dev = lappend(tlist_dev, tle);
	}
	gss->base_slot = MakeSingleTupleTableSlot(RelationGetDescr(rel),
											  &TTSOpsHeapTuple);
	gss->base_proj = ExecBuildProjectionInfo(tlist_dev,
											 gss->pts.css.ss.ps.ps_ExprContext,
											 gss->pts.css.ss.ss_ScanTupleSlot,
											 &gss->pts.css.ss.ps,
											 RelationGetDescr(rel));
	//init BRIN-Index Support
	//init Arrow_Fdw Support

}

/*
 * GpuScanGetNext
 */
static TupleTableSlot *
GpuScanGetNext(GpuScanState *gss)
{
#if 0
	if (gss->gc_state)
		pds = loadGpuCache();
#endif
#if 0
	if (gss->af_state)
		pds = loadArrow();
	if (gss->gd_state)
	{
		pds = loadBlock();
	}
	else
	{
		pds = execScanChunk();
	}

#endif	
	return NULL;
}

/*
 * GpuScanReCheckTuple
 */
static bool
GpuScanReCheckTuple(GpuScanState *gss, TupleTableSlot *epq_slot)
{
	/*
	 * NOTE: Only immutable operators/functions are executable
	 * on the GPU devices, so its decision will never changed.
	 */
	return true;
}

/*
 * ExecGpuScan
 */
static TupleTableSlot *
ExecGpuScan(CustomScanState *node)
{
	GpuScanState   *gss = (GpuScanState *)node;

	if (!gss->pts.css.ss.ss_currentScanDesc)
		gss->pts.ps_state = pgstromSharedStateCreate(&gss->pts.css, NULL);
	if (!gss->pts.conn)
	{
		const XpuCommand *session;
		const Bitmapset *gpuset = NULL;

		session = pgstromBuildSessionInfo(&gss->pts.css.ss.ps,
										  gss->gs_info.used_params,
										  gss->gs_info.extra_bufsz,
										  gss->gs_info.kvars_nslots,
										  gss->gs_info.kern_quals,
										  gss->gs_info.kern_projs);
		if (gss->pts.gc_state)
			gpuset = gss->gs_info.gpu_cache_devs;
		else if (gss->pts.gd_state)
			gpuset = gss->gs_info.gpu_direct_devs;
		gss->pts.conn = gpuClientOpenSession(gpuset, session);
	}
	return ExecScan(&node->ss,
					(ExecScanAccessMtd) GpuScanGetNext,
					(ExecScanRecheckMtd) GpuScanReCheckTuple);
}

/*
 * ExecEndGpuScan
 */
static void
ExecEndGpuScan(CustomScanState *node)
{
	GpuScanState   *gss = (GpuScanState *)node;

	if (gss->pts.conn)
		xpuClientCloseSession(gss->pts.conn);
	if (gss->base_slot)
		ExecDropSingleTupleTableSlot(gss->base_slot);
	if (gss->pts.css.ss.ss_currentScanDesc)
		table_endscan(gss->pts.css.ss.ss_currentScanDesc);
}

/*
 * ExecReScanGpuScan
 */
static void
ExecReScanGpuScan(CustomScanState *node)
{
	GpuScanState   *gss = (GpuScanState *)node;

	if (gss->pts.conn)
	{
		xpuClientCloseSession(gss->pts.conn);
		gss->pts.conn = NULL;
	}
}

/*
 * EstimateGpuScanDSM
 */
static Size
EstimateGpuScanDSM(CustomScanState *node,
				   ParallelContext *pcxt)
{
	return pgstromSharedStateEstimate(node);
}

/*
 * InitializeGpuScanDSM
 */
static void
InitializeGpuScanDSM(CustomScanState *node,
					 ParallelContext *pcxt,
					 void *coordinate)
{
	GpuScanState   *gss = (GpuScanState *)node;

	gss->pts.ps_state = pgstromSharedStateCreate(node, coordinate);
}

/*
 * ReInitializeGpuScanDSM
 */
static void
ReInitializeGpuScanDSM(CustomScanState *node,
					   ParallelContext *pcxt,
					   void *coordinate)
{
	GpuScanState   *gss = (GpuScanState *)node;

	pgstromSharedStateReset(gss->pts.ps_state);
}

/*
 * InitGpuScanWorker
 */
static void
InitGpuScanWorker(CustomScanState *node, shm_toc *toc, void *coordinate)
{
	GpuScanState   *gss = (GpuScanState *)node;

	gss->pts.ps_state = (pgstromSharedState *)coordinate;
}

/*
 * ExecShutdownGpuScan
 */
static void
ExecShutdownGpuScan(CustomScanState *node)
{
	GpuScanState   *gss = (GpuScanState *)node;

	gss->pts.ps_state = pgstromSharedStateShutdown(&gss->pts.css, gss->pts.ps_state);
}

/*
 * ExplainGpuScan
 */
static void
ExplainGpuScan(CustomScanState *node,
			   List *ancestors,
			   ExplainState *es)
{
	GpuScanState   *gss = (GpuScanState *) node;
	StringInfoData	temp;

	initStringInfo(&temp);
	pgstrom_explain_xpucode(&temp,
							gss->gs_info.kern_quals,
							&gss->pts.css,
							es, ancestors);
	ExplainPropertyText("GPU Quals", temp.data, es);

	resetStringInfo(&temp);
	pgstrom_explain_xpucode(&temp,
							gss->gs_info.kern_projs,
							&gss->pts.css,
							es, ancestors);
	ExplainPropertyText("GPU Projection", temp.data, es);
}

/*
 * pgstrom_init_gpuscan
 */
void
pgstrom_init_gpu_scan(void)
{
	/* pg_strom.enable_gpuscan */
	DefineCustomBoolVariable("pg_strom.enable_gpuscan",
							 "Enables the use of GPU accelerated full-scan",
							 NULL,
							 &enable_gpuscan,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* pg_strom.pullup_outer_scan */
	DefineCustomBoolVariable("pg_strom.pullup_outer_scan",
							 "Enables to pull up simple outer scan",
							 NULL,
							 &enable_pullup_outer_scan,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);

	/* setup path methods */
	memset(&gpuscan_path_methods, 0, sizeof(gpuscan_path_methods));
	gpuscan_path_methods.CustomName			= "GpuScan";
	gpuscan_path_methods.PlanCustomPath		= PlanGpuScanPath;

    /* setup plan methods */
    memset(&gpuscan_plan_methods, 0, sizeof(gpuscan_plan_methods));
    gpuscan_plan_methods.CustomName			= "GpuScan";
    gpuscan_plan_methods.CreateCustomScanState = CreateGpuScanState;
    RegisterCustomScanMethods(&gpuscan_plan_methods);

    /* setup exec methods */
    memset(&gpuscan_exec_methods, 0, sizeof(gpuscan_exec_methods));
    gpuscan_exec_methods.CustomName			= "GpuScan";
    gpuscan_exec_methods.BeginCustomScan	= ExecInitGpuScan;
    gpuscan_exec_methods.ExecCustomScan		= ExecGpuScan;
    gpuscan_exec_methods.EndCustomScan		= ExecEndGpuScan;
    gpuscan_exec_methods.ReScanCustomScan	= ExecReScanGpuScan;
    gpuscan_exec_methods.EstimateDSMCustomScan = EstimateGpuScanDSM;
    gpuscan_exec_methods.InitializeDSMCustomScan = InitializeGpuScanDSM;
    gpuscan_exec_methods.InitializeWorkerCustomScan = InitGpuScanWorker;
    gpuscan_exec_methods.ReInitializeDSMCustomScan = ReInitializeGpuScanDSM;
    gpuscan_exec_methods.ShutdownCustomScan	= ExecShutdownGpuScan;
    gpuscan_exec_methods.ExplainCustomScan	= ExplainGpuScan;

	/* hook registration */
	set_rel_pathlist_next = set_rel_pathlist_hook;
	set_rel_pathlist_hook = GpuScanAddScanPath;
}
