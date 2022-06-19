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

static GpuScanInfo *
deform_gpuscan_info(CustomScan *cscan)
{
	GpuScanInfo *gs_info = palloc0(sizeof(GpuScanInfo));
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
		gs_info->kern_projs	= DatumGetByteaP(con->constvalue);
	gs_info->extra_flags	= intVal(list_nth(privs, pindex++));
	gs_info->extra_bufsz	= intVal(list_nth(privs, pindex++));
	gs_info->nrows_per_block = intVal(list_nth(privs, pindex++));
	gs_info->outer_refs		= bms_from_pglist(list_nth(privs, pindex++));
	gs_info->used_params	= list_nth(exprs, eindex++);
	gs_info->dev_quals		= list_nth(exprs, eindex++);
	gs_info->index_oid		= intVal(list_nth(privs, pindex++));
	gs_info->index_conds	= list_nth(exprs, eindex++);
	gs_info->index_quals	= list_nth(exprs, eindex++);

	return gs_info;
}

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
	Bitmapset	   *outer_refs = NULL;
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
	int				j, nrows_per_block = 0;

	/* CPU Parallel parameters */
	if (parallel_aware)
	{
		double	leader_contribution;

		parallel_nworkers = compute_parallel_worker(baserel,
													baserel->pages, -1,
													max_parallel_workers_per_gather);
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
	pull_varattnos((Node *)dev_quals, baserel->relid, &outer_refs);

	/* Cost for DMA receive (GPU-->Host) */
	run_cost += pgstrom_gpu_dma_cost * ntuples;

	/* cost for CPU qualifiers */
	cost_qual_eval(&qcost, host_quals, root);
	startup_cost += qcost.startup;
	run_cost += qcost.per_tuple * ntuples;
	pull_varattnos((Node *)host_quals, baserel->relid, &outer_refs);

	/* PPI costs (as a part of host quals, if any) */
	param_info = get_baserel_parampathinfo(root, baserel,
										   baserel->lateral_relids);
	if (param_info)
	{
		List   *ppi_quals = param_info->ppi_clauses;

		cost_qual_eval(&qcost, ppi_quals, root);
		startup_cost += qcost.startup;
		run_cost += qcost.per_tuple * ntuples;
		pull_varattnos((Node *)ppi_quals, baserel->relid, &outer_refs);
	}

	/* Cost for Projection */
	startup_cost += baserel->reltarget->cost.startup;
	run_cost += baserel->reltarget->cost.per_tuple * baserel->rows;

	/* Latency to get the first chunk */
	if (nchunks > 0)
		startup_delay = run_cost * (1.0 / nchunks);

	/* check referenced columns */
	for (j=baserel->min_attr; j <= baserel->max_attr; j++)
	{
		if (!baserel->attr_needed[j - baserel->min_attr])
			continue;
		outer_refs = bms_add_member(outer_refs,
									j - FirstLowInvalidHeapAttributeNumber);
	}

	/* setup GpuScanInfo (Path phase) */
	gs_info->nrows_per_block = nrows_per_block;
	gs_info->outer_refs = outer_refs;
	gs_info->dev_quals = dev_quals;
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

		if (pgstrom_gpu_expression(rinfo->clause, 0, NULL))
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
	if (custom_path_remember(root, baserel, false, false, cpath))
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
		if (custom_path_remember(root, baserel, true, false, cpath))
			add_partial_path(baserel, &cpath->path);
	}
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
	//add opcode
	return NULL;
}

/*
 * CreateGpuScanState
 */
static Node *
CreateGpuScanState(CustomScan *cscan)
{
	return NULL;
}

/*
 * ExecInitGpuScan
 */
static void
ExecInitGpuScan(CustomScanState *node, EState *estate, int eflags)
{}

/*
 * ExecGpuScan
 */
static TupleTableSlot *
ExecGpuScan(CustomScanState *node)
{
	return NULL;
}

/*
 * ExecEndGpuScan
 */
static void
ExecEndGpuScan(CustomScanState *node)
{}

/*
 * ExecReScanGpuScan
 */
static void
ExecReScanGpuScan(CustomScanState *node)
{}

/*
 * EstimateGpuScanDSM
 */
static Size
EstimateGpuScanDSM(CustomScanState *node,
				   ParallelContext *pcxt)
{
	return 0;
}

/*
 * InitializeGpuScanDSM
 */
static void
InitializeGpuScanDSM(CustomScanState *node,
					 ParallelContext *pcxt,
					 void *coordinate)
{}

/*
 * ReInitializeGpuScanDSM
 */
static void
ReInitializeGpuScanDSM(CustomScanState *node,
					   ParallelContext *pcxt,
					   void *coordinate)
{}

/*
 * InitGpuScanWorker
 */
static void
InitGpuScanWorker(CustomScanState *node, shm_toc *toc, void *coordinate)
{}

/*
 * ExecShutdownGpuScan
 */
static void
ExecShutdownGpuScan(CustomScanState *node)
{}

/*
 * ExplainGpuScan
 */
static void
ExplainGpuScan(CustomScanState *node,
			   List *ancestors,
			   ExplainState *es)
{}

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
