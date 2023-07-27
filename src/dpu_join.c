/*
 * dpu_join.c
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
static set_join_pathlist_hook_type set_join_pathlist_next = NULL;
       CustomPathMethods	dpujoin_path_methods;
static CustomScanMethods	dpujoin_plan_methods;
static CustomExecMethods	dpujoin_exec_methods;
bool		pgstrom_enable_dpujoin;			/* GUC */
bool		pgstrom_enable_dpuhashjoin;		/* GUC */
bool		pgstrom_enable_dpugistindex;	/* GUC */

/*
 * dpujoin_add_custompath
 */
static void
dpujoin_add_custompath(PlannerInfo *root,
					   RelOptInfo *joinrel,
					   RelOptInfo *outerrel,
					   RelOptInfo *innerrel,
					   JoinType join_type,
					   JoinPathExtraData *extra)
{
	/* calls secondary module if exists */
	if (set_join_pathlist_next)
		set_join_pathlist_next(root,
							   joinrel,
							   outerrel,
							   innerrel,
							   join_type,
							   extra);
	/* quick bailout if PG-Strom/DpuJoin is not enabled */
	if (!pgstrom_enabled || !pgstrom_enable_dpujoin)
		return;
	/* common portion to add custom-paths for xPU-Join */
	xpujoin_add_custompath(root,
						   joinrel,
						   outerrel,
						   innerrel,
						   join_type,
						   extra,
						   TASK_KIND__DPUJOIN,
						   &dpujoin_path_methods);
}

/*
 * PlanDpuJoinPath
 */
static Plan *
PlanDpuJoinPath(PlannerInfo *root,
				RelOptInfo *joinrel,
				CustomPath *cpath,
				List *tlist,
				List *clauses,
				List *custom_plans)
{
	pgstromPlanInfo *pp_info = linitial(cpath->custom_private);
	CustomScan *cscan;

	cscan = PlanXpuJoinPathCommon(root,
								  joinrel,
								  cpath,
								  tlist,
								  custom_plans,
								  pp_info,
								  &dpujoin_plan_methods);
	form_pgstrom_plan_info(cscan, pp_info);
	return &cscan->scan.plan;
}

/* ----------------------------------------------------------------
 *
 * Executor Routines
 *
 * ----------------------------------------------------------------
 */
static Node *
CreateDpuJoinState(CustomScan *cscan)
{
	pgstromTaskState *pts;
	pgstromPlanInfo  *pp_info = deform_pgstrom_plan_info(cscan);
	int			num_rels = list_length(cscan->custom_plans);

	Assert(cscan->methods == &dpujoin_plan_methods);
	pts = palloc0(offsetof(pgstromTaskState, inners[num_rels]));
	NodeSetTag(pts, T_CustomScanState);
	pts->css.flags = cscan->flags;
	pts->css.methods = &dpujoin_exec_methods;
	pts->xpu_task_flags = pp_info->xpu_task_flags;
	pts->pp_info = pp_info;
	Assert((pts->xpu_task_flags & TASK_KIND__MASK) == TASK_KIND__DPUJOIN &&
		   pp_info->num_rels == num_rels);
	pts->num_rels = num_rels;

	return (Node *)pts;
}

/*
 * pgstrom_init_dpu_join
 */
void
pgstrom_init_dpu_join(void)
{
	/* turn on/off dpujoin */
	DefineCustomBoolVariable("pg_strom.enable_dpujoin",
							 "Enables the use of DpuJoin logic",
							 NULL,
							 &pgstrom_enable_dpujoin,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* turn on/off dpuhashjoin */
	DefineCustomBoolVariable("pg_strom.enable_dpuhashjoin",
							 "Enables the use of DpuHashJoin logic",
							 NULL,
							 &pgstrom_enable_dpuhashjoin,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* turn on/off dpugistindex */
	DefineCustomBoolVariable("pg_strom.enable_dpugistindex",
							 "Enables the use of DpuGistIndex logic",
							 NULL,
							 &pgstrom_enable_dpugistindex,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* setup path methods */
	memset(&dpujoin_path_methods, 0, sizeof(CustomPathMethods));
	dpujoin_path_methods.CustomName             = "DpuJoin";
	dpujoin_path_methods.PlanCustomPath         = PlanDpuJoinPath;

	/* setup plan methods */
	memset(&dpujoin_plan_methods, 0, sizeof(CustomScanMethods));
	dpujoin_plan_methods.CustomName             = "DpuJoin";
	dpujoin_plan_methods.CreateCustomScanState  = CreateDpuJoinState;
	RegisterCustomScanMethods(&dpujoin_plan_methods);

	/* setup exec methods */
	memset(&dpujoin_exec_methods, 0, sizeof(CustomExecMethods));
	dpujoin_exec_methods.CustomName             = "DpuJoin";
	dpujoin_exec_methods.BeginCustomScan        = pgstromExecInitTaskState;
	dpujoin_exec_methods.ExecCustomScan         = pgstromExecTaskState;
	dpujoin_exec_methods.EndCustomScan          = pgstromExecEndTaskState;
	dpujoin_exec_methods.ReScanCustomScan       = pgstromExecResetTaskState;
	dpujoin_exec_methods.EstimateDSMCustomScan  = pgstromSharedStateEstimateDSM;
	dpujoin_exec_methods.InitializeDSMCustomScan = pgstromSharedStateInitDSM;
	dpujoin_exec_methods.InitializeWorkerCustomScan = pgstromSharedStateAttachDSM;
	dpujoin_exec_methods.ShutdownCustomScan     = pgstromSharedStateShutdownDSM;
	dpujoin_exec_methods.ExplainCustomScan      = pgstromExplainTaskState;

	/* hook registration */
	set_join_pathlist_next = set_join_pathlist_hook;
	set_join_pathlist_hook = dpujoin_add_custompath;
}
