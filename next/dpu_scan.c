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
		pgstromPlanInfo pp_data;
		pgstromPlanInfo *pp_info;
		CustomPath	   *cpath;
		ParamPathInfo  *param_info = NULL;
		int				parallel_nworkers = 0;
		Cost			startup_cost = 0.0;
		Cost			run_cost = 0.0;
		Cost			final_cost = 0.0;

		memset(&pp_data, 0, sizeof(pgstromPlanInfo));
		if (!consider_xpuscan_path_params(root,
										  baserel,
										  TASK_KIND__DPUSCAN,
										  dev_quals,
										  host_quals,
										  try_parallel > 0,	/* parallel_aware */
										  &parallel_nworkers,
										  &startup_cost,
										  &run_cost,
										  &pp_data))
			return;

		/* setup DpuScanInfo (Path phase) */
		pp_info = pmemdup(&pp_data, sizeof(pgstromPlanInfo));
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
		cpath->custom_private = list_make1(pp_info);
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
	pgstromPlanInfo *pp_info = linitial(best_path->custom_private);
	CustomScan *cscan;

	/* sanity checks */
	Assert(baserel->relid > 0 &&
		   baserel->rtekind == RTE_RELATION &&
		   custom_children == NIL);
	cscan = PlanXpuScanPathCommon(root,
								  baserel,
								  best_path,
								  tlist,
								  clauses,
								  pp_info,
								  &dpuscan_plan_methods);
	form_pgstrom_plan_info(cscan, pp_info);

	return &cscan->scan.plan;
}

/*
 * CreateDpuScanState
 */
static Node *
CreateDpuScanState(CustomScan *cscan)
{
	pgstromTaskState *pts = palloc0(sizeof(pgstromTaskState));

	Assert(cscan->methods == &dpuscan_plan_methods);
	NodeSetTag(pts, T_CustomScanState);
	pts->css.flags = cscan->flags;
	pts->css.methods = &dpuscan_exec_methods;
	pts->task_kind = TASK_KIND__DPUSCAN;
	pts->pp_info = deform_pgstrom_plan_info(cscan);
	Assert(pts->task_kind == pts->pp_info->task_kind);

	return (Node *)pts;
}

/*
 * ExecDpuScan
 */
static TupleTableSlot *
ExecDpuScan(CustomScanState *node)
{
	pgstromTaskState *pts = (pgstromTaskState *)node;

	if (!pts->ps_state)
		pgstromSharedStateInitDSM(&pts->css, NULL, NULL);
	if (!pts->conn)
	{
		const XpuCommand *session
			= pgstromBuildSessionInfo(pts, 0);
		DpuClientOpenSession(pts, session);
	}
	return pgstromExecTaskState(pts);
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
    dpuscan_exec_methods.BeginCustomScan	= pgstromExecInitTaskState;
    dpuscan_exec_methods.ExecCustomScan		= ExecDpuScan;
    dpuscan_exec_methods.EndCustomScan		= pgstromExecEndTaskState;
    dpuscan_exec_methods.ReScanCustomScan	= pgstromExecResetTaskState;
    dpuscan_exec_methods.EstimateDSMCustomScan = pgstromSharedStateEstimateDSM;
	dpuscan_exec_methods.InitializeDSMCustomScan = pgstromSharedStateInitDSM;
    dpuscan_exec_methods.InitializeWorkerCustomScan = pgstromSharedStateAttachDSM;
    dpuscan_exec_methods.ShutdownCustomScan = pgstromSharedStateShutdownDSM;
    dpuscan_exec_methods.ExplainCustomScan	= pgstromExplainTaskState;

    /* hook registration */
    set_rel_pathlist_next = set_rel_pathlist_hook;
    set_rel_pathlist_hook = DpuScanAddScanPath;
}
