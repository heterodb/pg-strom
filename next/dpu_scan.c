/*
 * dpu_scan.c
 *
 * Sequential scan accelerated with DPU processors
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/* static variables */
static set_rel_pathlist_hook_type set_rel_pathlist_next = NULL;
static CustomPathMethods	dpuscan_path_methods;
static CustomScanMethods	dpuscan_plan_methods;
static CustomExecMethods	dpuscan_exec_methods;
static bool					enable_dpuscan;		/* GUC */

/*
 * DpuScanSharedState
 */
typedef struct
{
	/* for arrow_fdw file scan */
	pg_atomic_uint32	af_rbatch_index;
	pg_atomic_uint32	af_rbatch_nload;	/* # of loaded record-batches */
	pg_atomic_uint32	af_rbatch_nskip;	/* # of skipped record-batches */
	/* for block-based regular table scan */
	BlockNumber			pbs_nblocks;		/* # blocks in relation at start of scan */
	slock_t				pbs_mutex;			/* lock of the fields below */
	BlockNumber			pbs_startblock;		/* starting block number */
	BlockNumber			pbs_nallocated;		/* # of blocks allocated to workers */
	/* common parallel table scan descriptor */
	ParallelTableScanDescData phscan;
} DpuScanSharedState;

/*
 * DpuScanState
 */
typedef struct
{
	pgstromTaskState	pts;
	DpuScanInfo			ds_info;
	XpuCommand		   *xcmd_req;
	size_t				xcmd_len;
} DpuScanState;

/*
 * DpuScanAddScanPath
 */
static void
DpuScanAddScanPath(PlannerInfo *root,
				   RelOptInfo *baserel,
				   Index rtindex,
				   RangeTblEntry *rte)
{
	List	   *input_rels_tlist;
	List	   *dev_quals = NIL;
	List	   *host_quals = NIL;
	ParamPathInfo *param_info;
	ListCell   *lc;

	/* call the secondary hook */
	if (set_rel_pathlist_next)
		set_rel_pathlist_next(root, baserel, rtindex, rte);
	/* nothing to do, if either PG-Strom or DpuScan is not enabled */
	if (!pgstrom_enabled || !enable_dpuscan)
		return;
	/* We already proved the relation empty, so nothing more to do */
	if (is_dummy_rel(baserel))
		return;
	/* It is the role of built-in Append node */
	if (rte->inh)
		return;

	/*
	 * check whether the qualifier can run on DPU device
	 */
	input_rels_tlist = list_make1(makeInteger(baserel->relid));
	foreach (lc, baserel->baserestrictinfo)
	{
		RestrictInfo *rinfo = lfirst(lc);

		if (pgstrom_dpu_expression(rinfo->clause,
								   input_rels_tlist,
								   NULL))
			dev_quals = lappend(dev_quals, rinfo);
		else
			host_quals = lappend(host_quals, rinfo);
	}
	/*
	 * check parametalized qualifiers
	 */
	param_info = get_baserel_parampathinfo(root, baserel,
										   baserel->lateral_relids);
	if (param_info)
	{
		foreach (lc, param_info->ppi_clauses)
		{
			RestrictInfo *rinfo = lfirst(lc);

			if (pgstrom_gpu_expression(rinfo->clause,
									   input_rels_tlist,
									   NULL))
				dev_quals = lappend(dev_quals, rinfo);
			else
				host_quals = lappend(host_quals, rinfo);
		}
	}

	 /* Creation of DpuScan path */
	for (int try_parallel=0; try_parallel < 2; try_parallel++)
	{
		DpuScanInfo		ds_data;
		DpuScanInfo	   *ds_info;
		CustomPath	   *cpath;
		ParamPathInfo  *param_info = NULL;
		int				parallel_nworkers = 0;
		Cost			startup_cost = 0.0;
		Cost			run_cost = 0.0;
		Cost			final_cost = 0.0;

		memset(&ds_data, 0, sizeof(GpuScanInfo));
		if (!considerXpuScanPathParams(root,
									   baserel,
									   DEVKIND__NVIDIA_DPU,
									   try_parallel > 0,	/* parallel_aware */
									   dev_quals,
									   host_quals,
									   &parallel_nworkers,
									   &ds_data.index_oid,
									   &ds_data.index_conds,
									   &ds_data.index_quals,
									   &startup_cost,
									   &run_cost,
									   &final_cost,
									   NULL,
									   NULL,
									   &ds_data.ds_entry))
			return;

		/* setup DpuScanInfo (Path phase) */
		ds_info = pmemdup(&ds_data, sizeof(DpuScanInfo));
		cpath = makeNode(CustomPath);
		cpath->path.pathtype = T_CustomScan;
		cpath->path.parent = baserel;
		cpath->path.pathtarget = baserel->reltarget;
		cpath->path.param_info = param_info;
		cpath->path.parallel_aware = (try_parallel > 0);
		cpath->path.parallel_safe = baserel->consider_parallel;
		cpath->path.parallel_workers = parallel_nworkers;
		cpath->path.rows = (param_info ? param_info->ppi_rows : baserel->rows);
		cpath->path.startup_cost = startup_cost;
		cpath->path.total_cost = startup_cost + run_cost + final_cost;
		cpath->path.pathkeys = NIL; /* unsorted results */
		cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
		cpath->custom_paths = NIL;
		cpath->custom_private = list_make1(ds_info);
		cpath->methods = &dpuscan_path_methods;

		if (custom_path_remember(root, baserel, (try_parallel > 0), cpath))
		{
			if (try_parallel == 0)
				add_path(baserel, &cpath->path);
			else
				add_partial_path(baserel, &cpath->path);
		}
	}
}

/*
 * PlanDpuScanPath
 */
static Plan *
PlanDpuScanPath(PlannerInfo *root,
                RelOptInfo *baserel,
                CustomPath *best_path,
                List *tlist,
                List *clauses,
                List *custom_children)
{
	DpuScanInfo	   *ds_info = linitial(best_path->custom_private);
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
								  ds_info,
								  &dpuscan_plan_methods);
	form_gpuscan_info(cscan, ds_info);

	return &cscan->scan.plan;
}

/*
 * CreateDpuScanState
 */
static Node *
CreateDpuScanState(CustomScan *cscan)
{
	DpuScanState   *dss = palloc0(sizeof(DpuScanState));

    Assert(cscan->methods == &dpuscan_plan_methods);
	NodeSetTag(dss, T_CustomScanState);
    dss->pts.css.flags = cscan->flags;
    dss->pts.css.methods = &dpuscan_exec_methods;
	dss->pts.devkind = DEVKIND__NVIDIA_DPU;
    deform_dpuscan_info(&dss->ds_info, cscan);

	return (Node *)dss;
}

/*
 * ExecInitDpuScan
 */
static void
ExecInitDpuScan(CustomScanState *node, EState *estate, int eflags)
{
	DpuScanState   *dss = (DpuScanState *)node;

	 /* sanity checks */
    Assert(relation != NULL &&
           outerPlanState(node) == NULL &&
           innerPlanState(node) == NULL);
	pgstromExecInitTaskState(&dss->pts,
							 DEVKIND__NVIDIA_DPU,
							 dss->ds_info.dev_quals,
							 dss->ds_info.outer_refs,
							 dss->ds_info.index_oid,
                             dss->ds_info.index_conds,
                             dss->ds_info.index_quals);
	dss->pts.cb_cpu_fallback = ExecFallbackCpuScan;
}

/*
 * DpuScanReCheckTuple
 */
static bool
DpuScanReCheckTuple(DpuScanState *dss, TupleTableSlot *epq_slot)
{
	/*
	 * NOTE: Only immutable operators/functions are executable
	 * on DPU devices, so its decision will never changed.
	 */
	return true;
}

/*
 * ExecDpuScan
 */
static TupleTableSlot *
ExecDpuScan(CustomScanState *node)
{
	DpuScanState   *dss = (DpuScanState *)node;

	if (!dss->pts.ps_state)
		pgstromSharedStateInitDSM(&dss->pts, NULL, NULL);
	if (!dss->pts.conn)
	{
		const XpuCommand *session;

		session = pgstromBuildSessionInfo(&dss->pts,
										  dss->ds_info.used_params,
										  dss->ds_info.extra_bufsz,
										  dss->ds_info.kvars_depth,
										  dss->ds_info.kvars_resno,
										  dss->ds_info.kexp_kvars_load,
										  dss->ds_info.kexp_scan_quals,
										  NULL,	/* join-load-vars */
										  NULL,	/* join-quals */
										  NULL,	/* hash-values */
										  NULL,	/* gist-join */
										  dss->ds_info.kexp_projection,
										  0);	/* No join_inner_handle */
		DpuClientOpenSession(&dss->pts, session);
	}
	return ExecScan(&node->ss,
					(ExecScanAccessMtd) pgstromExecTaskState,
					(ExecScanRecheckMtd) DpuScanReCheckTuple);
}

/*
 * ExecEndDpuScan
 */
static void
ExecEndDpuScan(CustomScanState *node)
{
	pgstromExecEndTaskState((pgstromTaskState *)node);
}

/*
 * ExecReScanDpuScan
 */
static void
ExecReScanDpuScan(CustomScanState *node)
{
	pgstromExecResetTaskState((pgstromTaskState *)node);
}

/*
 * EstimateDpuScanDSM
 */
static Size
EstimateDpuScanDSM(CustomScanState *node,
				   ParallelContext *pcxt)
{
	return pgstromSharedStateEstimateDSM((pgstromTaskState *)node);
}

/*
 * InitializeDpuScanDSM
 */
static void
InitializeDpuScanDSM(CustomScanState *node,
					 ParallelContext *pcxt,
					 void *dsm_addr)
{
	pgstromSharedStateInitDSM((pgstromTaskState *)node, pcxt, dsm_addr);
}

/*
 * InitDpuScanWorker
 */
static void
InitDpuScanWorker(CustomScanState *node, shm_toc *toc, void *dsm_addr)
{
	pgstromSharedStateAttachDSM((pgstromTaskState *)node, dsm_addr);
}

/*
 * ExecShutdownDpuScan
 */
static void
ExecShutdownDpuScan(CustomScanState *node)
{
	pgstromSharedStateShutdownDSM((pgstromTaskState *)node);
}

/*
 * ExplainDpuScan
 */
static void
ExplainDpuScan(CustomScanState *node,
               List *ancestors,
               ExplainState *es)
{
	DpuScanState   *dss = (DpuScanState *) node;
	DpuScanInfo	   *ds_info = &dss->ds_info;
	CustomScan	   *cscan = (CustomScan *)node->ss.ps.plan;
	List		   *dcontext;

	dcontext = set_deparse_context_plan(es->deparse_cxt,
										node->ss.ps.plan,
										ancestors);
	pgstromExplainScanState(&dss->pts, es,
							dcontext,
							cscan->custom_scan_tlist,
							ds_info->dev_quals,
							ds_info->scan_tuples,
							ds_info->scan_rows);
	/* XPU Code (if verbose) */
	pgstrom_explain_xpucode(&dss->pts.css, es, dcontext,
							"Scan Var-Loads Code",
							ds_info->kexp_kvars_load);
	pgstrom_explain_xpucode(&dss->pts.css, es, dcontext,
							"Scan Quals Code",
							ds_info->kexp_scan_quals);
	pgstrom_explain_xpucode(&dss->pts.css, es, dcontext,
							"DPU Projection Code",
							ds_info->kexp_projection);

	pgstromExplainTaskState(&dss->pts, es, dcontext);
}

/*
 * pgstrom_init_dpu_scan
 */
void
pgstrom_init_dpu_scan(void)
{
	/* pg_strom.enable_dpuscan */
	DefineCustomBoolVariable("pg_strom.enable_dpuscan",
							 "Enables the use of DPU accelerated full-scan",
							 NULL,
							 &enable_dpuscan,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* setup path methods */
	memset(&dpuscan_path_methods, 0, sizeof(dpuscan_path_methods));
	dpuscan_path_methods.CustomName			= "DpuScan";
	dpuscan_path_methods.PlanCustomPath		= PlanDpuScanPath;

	/* setup plan methods */
	memset(&dpuscan_plan_methods, 0, sizeof(dpuscan_plan_methods));
	dpuscan_plan_methods.CustomName			= "DpuScan";
	dpuscan_plan_methods.CreateCustomScanState = CreateDpuScanState;
	RegisterCustomScanMethods(&dpuscan_plan_methods);

	/* setup exec methods */
	memset(&dpuscan_exec_methods, 0, sizeof(dpuscan_exec_methods));
    dpuscan_exec_methods.CustomName			= "DpuScan";
    dpuscan_exec_methods.BeginCustomScan	= ExecInitDpuScan;
    dpuscan_exec_methods.ExecCustomScan		= ExecDpuScan;
    dpuscan_exec_methods.EndCustomScan		= ExecEndDpuScan;
    dpuscan_exec_methods.ReScanCustomScan	= ExecReScanDpuScan;
    dpuscan_exec_methods.EstimateDSMCustomScan = EstimateDpuScanDSM;
    dpuscan_exec_methods.InitializeDSMCustomScan = InitializeDpuScanDSM;
    dpuscan_exec_methods.InitializeWorkerCustomScan = InitDpuScanWorker;
    dpuscan_exec_methods.ShutdownCustomScan = ExecShutdownDpuScan;
    dpuscan_exec_methods.ExplainCustomScan	= ExplainDpuScan;

    /* hook registration */
    set_rel_pathlist_next = set_rel_pathlist_hook;
    set_rel_pathlist_hook = DpuScanAddScanPath;
}
