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
#include "cuda_common.h"

/* static variables */
static set_rel_pathlist_hook_type set_rel_pathlist_next = NULL;
static CustomPathMethods	gpuscan_path_methods;
static CustomScanMethods	gpuscan_plan_methods;
static CustomExecMethods	gpuscan_exec_methods;
static bool					enable_gpuscan;		/* GUC */
static bool					enable_pullup_outer_scan;	/* GUC */

/*
 * form_gpuscan_info
 *
 * GpuScanInfo --> custom_private/custom_exprs
 */
void
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
	privs = lappend(privs, bms_to_pglist(gs_info->outer_refs));
	exprs = lappend(exprs, gs_info->used_params);
	exprs = lappend(exprs, gs_info->dev_quals);
	privs = lappend(privs, makeInteger(gs_info->index_oid));
	privs = lappend(privs, gs_info->index_conds);
	exprs = lappend(exprs, gs_info->index_quals);

	cscan->custom_private = privs;
	cscan->custom_exprs = exprs;
}

/*
 * deform_gpuscan_info
 *
 * custom_private/custom_exprs -> GpuScanInfo
 */
void
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
	gs_info->outer_refs		= bms_from_pglist(list_nth(privs, pindex++));
	gs_info->used_params	= list_nth(exprs, eindex++);
	gs_info->dev_quals		= list_nth(exprs, eindex++);
	gs_info->index_oid		= intVal(list_nth(privs, pindex++));
	gs_info->index_conds	= list_nth(privs, pindex++);
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
	XpuCommand		   *xcmd_req;	/* request command buffer */
	size_t				xcmd_len;
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
	Cost			startup_cost = pgstrom_gpu_setup_cost;
	Cost			run_cost = 0.0;
	Cost			disk_cost = 0.0;
	Cost			comp_cost = 0.0;
	double			gpu_ratio;
	double			selectivity;
	double			avg_seq_page_cost;
	double			spc_seq_page_cost;
	double			spc_rand_page_cost;
	QualCost		qcost;
	double			ntuples;

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
	/*
	 * Discount disk_cost if we can use GPU-Direct SQL on the source
	 * table. It offers much much efficient i/o subsystem to load database
	 * blocks to GPU device.
	 */
	if (!gs_info->gpu_direct_devs)
		avg_seq_page_cost = spc_seq_page_cost;
	else
		avg_seq_page_cost = (spc_seq_page_cost * (1.0 - baserel->allvisfrac) +
							 pgstrom_gpu_direct_seq_page_cost * baserel->allvisfrac);
	disk_cost = avg_seq_page_cost * baserel->pages;

	/* consideration of BRIN-index, if any */
	ntuples = baserel->tuples;
	if (indexOpt)
	{
		Cost	index_disk_cost;

		index_disk_cost = cost_brin_bitmap_build(root,
												 baserel,
												 indexOpt,
												 indexQuals);
		index_disk_cost += avg_seq_page_cost * indexNBlocks;
		if (disk_cost > index_disk_cost)
		{
			disk_cost = index_disk_cost;
			if (baserel->pages > 0)
				ntuples *= (double)indexNBlocks / (double)baserel->pages;
		}
		else
			indexOpt = NULL;	/* disables BRIN-index if no benefit */
	}
	/* No need to say, GpuCache does not need disk i/o */
	if (!gs_info->gpu_cache_devs)
		run_cost += disk_cost;

	/*
	 * Cost for GPU qualifiers
	 */
	if (cpu_operator_cost > 0.0)
		gpu_ratio = pgstrom_gpu_operator_cost / cpu_operator_cost;
	else if (pgstrom_gpu_operator_cost == 0.0)
		gpu_ratio = 1.0;
	else
		gpu_ratio = disable_cost;	/* very large but still finite */
	cost_qual_eval_node(&qcost, (Node *)dev_quals, root);
	startup_cost += qcost.startup;
	comp_cost += qcost.per_tuple * gpu_ratio * ntuples;

	selectivity = clauselist_selectivity(root,
										 dev_quals,
										 baserel->relid,
										 JOIN_INNER,
										 NULL);
	ntuples *= selectivity;		/* rows after dev_quals */

	/* Cost for DMA receive (GPU-->Host) */
	run_cost += pgstrom_gpu_dma_cost * ntuples;

	/* cost for CPU qualifiers */
	cost_qual_eval(&qcost, host_quals, root);
	startup_cost += qcost.startup;
	comp_cost += qcost.per_tuple * ntuples;

	/* PPI costs (as a part of host quals, if any) */
	param_info = get_baserel_parampathinfo(root, baserel,
										   baserel->lateral_relids);
	if (param_info)
	{
		List   *ppi_quals = param_info->ppi_clauses;

		cost_qual_eval(&qcost, ppi_quals, root);
		startup_cost += qcost.startup;
		comp_cost += qcost.per_tuple * ntuples;
	}

	/* Cost for Projection */
	startup_cost += baserel->reltarget->cost.startup;
	run_cost += baserel->reltarget->cost.per_tuple * baserel->rows;

	/* add computational cost */
	run_cost += (comp_cost / parallel_divisor);

	/* setup GpuScanInfo (Path phase) */
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
	cpath->path.rows = (param_info
						? param_info->ppi_rows
						: baserel->rows) / parallel_divisor;
	cpath->path.startup_cost = startup_cost;
	cpath->path.total_cost = startup_cost + run_cost;
	cpath->path.pathkeys = NIL; /* unsorted results */
	cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
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
	indexOpt = pgstromTryFindBrinIndex(root, baserel,
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
 * PlanXpuScanPathCommon
 */
CustomScan *
PlanXpuScanPathCommon(PlannerInfo *root,
					  RelOptInfo  *baserel,
					  CustomPath  *best_path,
					  List        *tlist,
					  List        *clauses,
					  GpuScanInfo *gs_info,
					  const CustomScanMethods *xpuscan_plan_methods)
{
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
	ListCell	   *cell;

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
	/* pickup referenced attributes */
	outer_refs = pickup_outer_referenced(root, baserel, outer_refs);

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
	cscan->methods = xpuscan_plan_methods;
	cscan->custom_plans = NIL;
	cscan->custom_scan_tlist = tlist_dev;

	return cscan;
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

	/* sanity checks */
	Assert(baserel->relid > 0 &&
		   baserel->rtekind == RTE_RELATION &&
		   custom_children == NIL);

	cscan = PlanXpuScanPathCommon(root,
								  baserel,
								  best_path,
								  tlist,
								  clauses,
								  gs_info,
								  &gpuscan_plan_methods);
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

	/* sanity checks */
	Assert(node->ss.ss_currentRelation != NULL &&
		   outerPlanState(node) == NULL &&
		   innerPlanState(node) == NULL);
	pgstromBrinIndexExecBegin(&gss->pts,
							  gss->gs_info.index_oid,
							  gss->gs_info.index_conds,
							  gss->gs_info.index_quals);
	//pgstromGpuCacheExecBegin here
	pgstromGpuDirectExecBegin(&gss->pts, gss->gs_info.gpu_direct_devs);
	
	//pgstromGpuDirectExecBegin here
	pgstromExecInitTaskState(&gss->pts, gss->gs_info.dev_quals);
	gss->pts.cb_cpu_fallback = ExecFallbackCpuScan;
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

	if (!gss->pts.ps_state)
		pgstromSharedStateInitDSM(&gss->pts, NULL);
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
			gpuset = pgstromGpuDirectDevices(&gss->pts);
		gpuClientOpenSession(&gss->pts, gpuset, session);
	}
	return ExecScan(&node->ss,
					(ExecScanAccessMtd) pgstromExecTaskState,
					(ExecScanRecheckMtd) GpuScanReCheckTuple);
}

/*
 * ExecEndGpuScan
 */
static void
ExecEndGpuScan(CustomScanState *node)
{
	GpuScanState   *gss = (GpuScanState *)node;

	pgstromExecEndTaskState(&gss->pts);
}

/*
 * ExecReScanGpuScan
 */
static void
ExecReScanGpuScan(CustomScanState *node)
{
	GpuScanState   *gss = (GpuScanState *)node;

	pgstromExecResetTaskState(&gss->pts);
}

/*
 * EstimateGpuScanDSM
 */
static Size
EstimateGpuScanDSM(CustomScanState *node,
				   ParallelContext *pcxt)
{
	return pgstromSharedStateEstimateDSM((pgstromTaskState *)node);
}

/*
 * InitializeGpuScanDSM
 */
static void
InitializeGpuScanDSM(CustomScanState *node,
					 ParallelContext *pcxt,
					 void *dsm_addr)
{
	pgstromSharedStateInitDSM((pgstromTaskState *)node, dsm_addr);
}

/*
 * ReInitializeGpuScanDSM
 */
static void
ReInitializeGpuScanDSM(CustomScanState *node,
					   ParallelContext *pcxt,
					   void *dsm_addr)
{
	GpuScanState   *gss = (GpuScanState *)node;

	pgstromSharedStateReInitDSM(&gss->pts);
}

/*
 * InitGpuScanWorker
 */
static void
InitGpuScanWorker(CustomScanState *node, shm_toc *toc, void *dsm_addr)
{
	GpuScanState   *gss = (GpuScanState *)node;
		
	pgstromSharedStateAttachDSM(&gss->pts, dsm_addr);
}

/*
 * ExecShutdownGpuScan
 */
static void
ExecShutdownGpuScan(CustomScanState *node)
{
	GpuScanState   *gss = (GpuScanState *)node;

	pgstromSharedStateShutdownDSM(&gss->pts);
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
 * ExecFallbackCpuScan
 */
void
ExecFallbackCpuScan(pgstromTaskState *pts, HeapTuple tuple)
{
	TupleTableSlot *scan_slot = pts->base_slot;

	/* check WHERE-clause if any */
	ExecForceStoreHeapTuple(tuple, scan_slot, false);
	if (pts->base_quals)
	{
		ExprContext	   *econtext = pts->css.ss.ps.ps_ExprContext;

		econtext->ecxt_scantuple = scan_slot;
		ResetExprContext(econtext);
		if (!ExecQual(pts->base_quals, econtext))
			return;
	}
	/* apply Projection if any */
	if (pts->base_proj)
		scan_slot = ExecProject(pts->base_proj);

	/* build fallback_store if none */
	if (!pts->fallback_store)
	{
		EState	   *estate = pts->css.ss.ps.state;
		MemoryContext oldcxt = MemoryContextSwitchTo(estate->es_query_cxt);

		pts->fallback_store = tuplestore_begin_heap(false, false, work_mem);

		MemoryContextSwitchTo(oldcxt);
	}
	tuplestore_puttupleslot(pts->fallback_store, scan_slot);
}

/*
 * Handle GpuScan Commands
 */
void
gpuservHandleGpuScanExec(gpuClient *gclient, XpuCommand *xcmd)
{
	kern_gpuscan	*kgscan = NULL;
	const char		*kds_src_fullpath = NULL;
	strom_io_vector *kds_src_iovec = NULL;
	kern_data_store *kds_src = NULL;
	kern_data_store *kds_dst = NULL;
	kern_data_store *kds_dst_head = NULL;
	kern_data_store **kds_dst_array = NULL;
	int				kds_dst_nrooms = 0;
	int				kds_dst_nitems = 0;
	const char	   *kern_funcname;
	CUfunction		f_kern_gpuscan;
	const gpuMemChunk *chunk = NULL;
	CUdeviceptr		m_kds_src = 0UL;
	CUdeviceptr		dptr;
	CUresult		rc;
	int				grid_sz;
	int				block_sz;
	unsigned int	shmem_sz;
	size_t			sz;
	void		   *kern_args[5];

	if (xcmd->u.scan.kds_src_fullpath)
		kds_src_fullpath = (char *)xcmd + xcmd->u.scan.kds_src_fullpath;
	if (xcmd->u.scan.kds_src_iovec)
		kds_src_iovec = (strom_io_vector *)((char *)xcmd + xcmd->u.scan.kds_src_iovec);
	if (xcmd->u.scan.kds_src_offset)
		kds_src = (kern_data_store *)((char *)xcmd + xcmd->u.scan.kds_src_offset);
	if (xcmd->u.scan.kds_dst_offset)
		kds_dst_head = (kern_data_store *)((char *)xcmd + xcmd->u.scan.kds_dst_offset);

	if (!kds_src)
	{
		gpuClientELog(gclient, "KDS_FORMAT_COLUMN is not yet implemented");
		return;
	}
	else if (kds_src->format == KDS_FORMAT_ROW)
	{
		kern_funcname = "kern_gpuscan_main_row";
		m_kds_src = (CUdeviceptr)kds_src;
	}
	else if (kds_src->format == KDS_FORMAT_BLOCK)
	{
		kern_funcname = "kern_gpuscan_main_block";

		if (kds_src_fullpath && kds_src_iovec)
		{
			chunk = gpuservLoadKdsBlock(gclient,
										kds_src,
										kds_src_fullpath,
										kds_src_iovec);
			if (!chunk)
				return;
			m_kds_src = chunk->base + chunk->offset;
		}
		else
		{
			Assert(kds_src->block_nloaded == kds_src->nitems);
			m_kds_src = (CUdeviceptr)kds_src;
		}
	}
	else
	{
		gpuClientELog(gclient, "unknown GpuScan Source format (%c)",
					  kds_src->format);
		return;
	}

	rc = cuModuleGetFunction(&f_kern_gpuscan,
							 gclient->cuda_module,
							 kern_funcname);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on cuModuleGetFunction: %s",
					   cuStrError(rc));
		goto bailout;
	}

	rc = gpuOptimalBlockSize(&grid_sz,
							 &block_sz,
							 &shmem_sz,
							 f_kern_gpuscan,
							 0,
							 sizeof(kern_gpuscan_suspend_warp),
							 0);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on gpuOptimalBlockSize: %s",
					   cuStrError(rc));
		goto bailout;
	}
	
	/*
	 * Allocation of the control structure
	 */
	grid_sz = Min(grid_sz, (kds_src->nitems + block_sz - 1) / block_sz);

//	block_sz = 32;
//	grid_sz = 1;

	sz = offsetof(kern_gpuscan, suspend_context) + shmem_sz * grid_sz;
	rc = cuMemAllocManaged(&dptr, sz, CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on cuMemAllocManaged(%lu): %s",
					   sz, cuStrError(rc));
		goto bailout;
	}
	kgscan = (kern_gpuscan *)dptr;
	memset(kgscan, 0, sz);
	kgscan->grid_sz		= grid_sz;
	kgscan->block_sz	= block_sz;

	/* prefetch source KDS, if managed memory */
	if (!chunk)
	{
		rc = cuMemPrefetchAsync((CUdeviceptr)kds_src,
								kds_src->length,
								CU_DEVICE_PER_THREAD,
								CU_STREAM_PER_THREAD);
		if (rc != CUDA_SUCCESS)
		{
			gpuClientFatal(gclient, "failed on cuMemPrefetchAsync: %s",
						   cuStrError(rc));
			goto bailout;
		}
	}

	/*
	 * Allocation of the destination buffer
	 */
resume_kernel:
	sz = KDS_HEAD_LENGTH(kds_dst_head) + PGSTROM_CHUNK_SIZE;
	rc = cuMemAllocManaged(&dptr, sz, CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on cuMemAllocManaged(%lu): %s",
					   sz, cuStrError(rc));
		goto bailout;
	}
	kds_dst = (kern_data_store *)dptr;
	memcpy(kds_dst, kds_dst_head, KDS_HEAD_LENGTH(kds_dst_head));
	kds_dst->length = sz;
	if (kds_dst_nitems >= kds_dst_nrooms)
	{
		kern_data_store **kds_dst_temp;

		kds_dst_nrooms = 2 * kds_dst_nrooms + 10;
		kds_dst_temp = alloca(sizeof(kern_data_store *) * kds_dst_nrooms);
		if (kds_dst_nitems > 0)
			memcpy(kds_dst_temp,
				   kds_dst_array,
				   sizeof(kern_data_store *) * kds_dst_nitems);
		kds_dst_array = kds_dst_temp;
	}
	kds_dst_array[kds_dst_nitems++] = kds_dst;
	
	/*
	 * Launch kernel
	 */
	kern_args[0] = &gclient->session;
	kern_args[1] = &kgscan;
	kern_args[2] = &m_kds_src;
	kern_args[3] = &kds_dst;

	rc = cuLaunchKernel(f_kern_gpuscan,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						shmem_sz,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on cuLaunchKernel: %s", cuStrError(rc));
		goto bailout;
	}

	rc = cuEventRecord(CU_EVENT_PER_THREAD, CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on cuEventRecord: %s", cuStrError(rc));
		goto bailout;
	}

	/* point of synchronization */
	rc = cuEventSynchronize(CU_EVENT_PER_THREAD);
	if (rc != CUDA_SUCCESS)
	{
		gpuClientFatal(gclient, "failed on cuEventSynchronize: %s", cuStrError(rc));
		goto bailout;
	}

	/* status check */
	if (kgscan->kerror.errcode == ERRCODE_STROM_SUCCESS)
	{
		XpuCommand	resp;

		if (kgscan->suspend_count > 0)
		{
			if (gpuServiceGoingTerminate())
			{
				gpuClientFatal(gclient, "GpuService is going to terminate during GpuScan kernel suspend/resume");
				goto bailout;
			}
			/* reset */
			kgscan->suspend_count = 0;
			goto resume_kernel;
		}
		/* send back status and kds_dst */
		memset(&resp, 0, offsetof(XpuCommand, u.results));
		resp.magic = XpuCommandMagicNumber;
		resp.tag   = XpuCommandTag__Success;
		resp.u.results.chunks_nitems = kds_dst_nitems;
		resp.u.results.chunks_offset = offsetof(XpuCommand, u.results.stats.scan.data);
		resp.u.results.stats.scan.nitems_in = kgscan->nitems_in;
		resp.u.results.stats.scan.nitems_out = kgscan->nitems_out;
		gpuClientWriteBack(gclient,
						   &resp, resp.u.results.chunks_offset,
						   kds_dst_nitems, kds_dst_array);
	}
	else if (kgscan->kerror.errcode == ERRCODE_CPU_FALLBACK)
	{
		XpuCommand	resp;

		/* send back kds_src with XpuCommandTag__CPUFallback */
		memset(&resp, 0, offsetof(XpuCommand, u.results));
		resp.magic = XpuCommandMagicNumber;
		resp.tag   = XpuCommandTag__CPUFallback;
		resp.u.results.chunks_nitems = 1;
		resp.u.results.chunks_offset = offsetof(XpuCommand, u.results.stats.scan.data);
		gpuClientWriteBack(gclient,
						   &resp, resp.u.results.chunks_offset,
						   1, &kds_src);
	}
	else
	{
		/* send back error status */
		__gpuClientELogRaw(gclient, &kgscan->kerror);
	}
bailout:
	if (kgscan)
	{
		rc = cuMemFree((CUdeviceptr)kgscan);
		if (rc != CUDA_SUCCESS)
			fprintf(stderr, "warning: failed on cuMemFree: %s\n", cuStrError(rc));
	}
	if (chunk)
		gpuMemFree(chunk);
	while (kds_dst_nitems > 0)
	{
		kds_dst = kds_dst_array[--kds_dst_nitems];

		rc = cuMemFree((CUdeviceptr)kds_dst);
		if (rc != CUDA_SUCCESS)
			fprintf(stderr, "warning: failed on cuMemFree: %s\n", cuStrError(rc));
	}
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
