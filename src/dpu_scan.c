/*
 * dpu_scan.c
 *
 * Sequential scan accelerated with DPU processors
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/* static variables */
static set_rel_pathlist_hook_type set_rel_pathlist_next = NULL;
	   CustomPathMethods	dpuscan_path_methods;
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
	/* Creation of DpuScan path */
	for (int try_parallel=0; try_parallel < 2; try_parallel++)
	{
		CustomPath *cpath;

		cpath = buildXpuScanPath(root,
								 baserel,
								 (try_parallel > 0),
								 true,		/* allow host quals */
								 false,		/* disallow no device quals */
								 TASK_KIND__DPUSCAN);
		if (cpath && custom_path_remember(root,
										  baserel,
										  (try_parallel > 0),
										  TASK_KIND__DPUSCAN,
										  cpath))
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
	pgstromPlanInfo  *pp_info = deform_pgstrom_plan_info(cscan);

	Assert(cscan->methods == &dpuscan_plan_methods);
	NodeSetTag(pts, T_CustomScanState);
	pts->css.flags = cscan->flags;
	pts->css.methods = &dpuscan_exec_methods;
	pts->xpu_task_flags = pp_info->xpu_task_flags;
	pts->pp_info = pp_info;
	Assert((pts->xpu_task_flags & TASK_KIND__MASK) == TASK_KIND__DPUSCAN);

	return (Node *)pts;
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
    dpuscan_exec_methods.ExecCustomScan		= pgstromExecTaskState;
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
