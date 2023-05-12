/*
 * dpu_preagg.c
 *
 * Aggregation and Group-By with DPU acceleration
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/* static variables */
static create_upper_paths_hook_type create_upper_paths_next;
static CustomPathMethods	dpupreagg_path_methods;
static CustomScanMethods	dpupreagg_plan_methods;
static CustomExecMethods	dpupreagg_exec_methods;
static bool		pgstrom_enable_dpupreagg;
static bool		pgstrom_enable_partitionwise_dpupreagg;

/*
 * dpupreagg_add_custompath
 */
static void
dpupreagg_add_custompath(PlannerInfo *root,
						 UpperRelationKind stage,
						 RelOptInfo *input_rel,
						 RelOptInfo *group_rel,
						 void *extra)
{
	if (create_upper_paths_next)
		create_upper_paths_next(root,
								stage,
								input_rel,
								group_rel,
								extra);
	if (stage != UPPERREL_GROUP_AGG)
		return;
	if (!pgstrom_enabled || !pgstrom_enable_dpupreagg)
		return;
	/* add custom-paths */
	xpupreagg_add_custompath(root,
							 input_rel,
							 group_rel,
							 extra,
							 TASK_KIND__DPUPREAGG,
							 &dpupreagg_path_methods);
}

/*
 * PlanDpuPreAggPath
 */
static Plan *
PlanDpuPreAggPath(PlannerInfo *root,
				  RelOptInfo *joinrel,
				  CustomPath *cpath,
				  List *tlist,
				  List *clauses,
				  List *custom_plans)
{
	pgstromPlanInfo *pp_info = linitial(cpath->custom_private);
	CustomScan	   *cscan;

	cscan = PlanXpuJoinPathCommon(root,
								  joinrel,
								  cpath,
								  tlist,
								  custom_plans,
								  pp_info,
								  &dpupreagg_plan_methods);
	form_pgstrom_plan_info(cscan, pp_info);
	return &cscan->scan.plan;
}

/*
 * CreateDpuPreAggScanState
 */
static Node *
CreateDpuPreAggScanState(CustomScan *cscan)
{
	pgstromTaskState *pts;
	pgstromPlanInfo  *pp_info = deform_pgstrom_plan_info(cscan);
	int			num_rels = list_length(cscan->custom_plans);

	Assert(cscan->methods == &dpupreagg_plan_methods);
	pts = palloc0(offsetof(pgstromTaskState, inners[num_rels]));
	NodeSetTag(pts, T_CustomScanState);
	pts->css.flags = cscan->flags;
	pts->css.methods = &dpupreagg_exec_methods;
	pts->xpu_task_flags = pp_info->xpu_task_flags;
	pts->pp_info = pp_info;
	Assert((pts->xpu_task_flags & TASK_KIND__MASK) == TASK_KIND__DPUPREAGG &&
		   pp_info->num_rels == num_rels);
	pts->num_rels = num_rels;

	return (Node *)pts;
}

/*
 * pgstrom_init_dpu_preagg
 */
void
pgstrom_init_dpu_preagg(void)
{
	/* turn on/off gpu_groupby */
	DefineCustomBoolVariable("pg_strom.enable_dpupreagg",
							 "Enables the use of DPU PreAgg",
							 NULL,
							 &pgstrom_enable_dpupreagg,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* pg_strom.enable_partitionwise_dpupreagg */
	DefineCustomBoolVariable("pg_strom.enable_partitionwise_dpupreagg",
							 "Enabled Enables partition wise DpuPreAgg",
							 NULL,
							 &pgstrom_enable_partitionwise_dpupreagg,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* initialization of path method table */
	memset(&dpupreagg_path_methods, 0, sizeof(CustomPathMethods));
	dpupreagg_path_methods.CustomName     = "DpuPreAgg";
	dpupreagg_path_methods.PlanCustomPath = PlanDpuPreAggPath;

    /* initialization of plan method table */
    memset(&dpupreagg_plan_methods, 0, sizeof(CustomScanMethods));
    dpupreagg_plan_methods.CustomName     = "DpuPreAgg";
    dpupreagg_plan_methods.CreateCustomScanState = CreateDpuPreAggScanState;
    RegisterCustomScanMethods(&dpupreagg_plan_methods);

    /* initialization of exec method table */
    memset(&dpupreagg_exec_methods, 0, sizeof(CustomExecMethods));
    dpupreagg_exec_methods.CustomName          = "GpuPreAgg";
    dpupreagg_exec_methods.BeginCustomScan     = pgstromExecInitTaskState;
    dpupreagg_exec_methods.ExecCustomScan      = pgstromExecTaskState;
    dpupreagg_exec_methods.EndCustomScan       = pgstromExecEndTaskState;
    dpupreagg_exec_methods.ReScanCustomScan    = pgstromExecResetTaskState;
    dpupreagg_exec_methods.EstimateDSMCustomScan = pgstromSharedStateEstimateDSM;
    dpupreagg_exec_methods.InitializeDSMCustomScan = pgstromSharedStateInitDSM;
    dpupreagg_exec_methods.InitializeWorkerCustomScan = pgstromSharedStateAttachDSM;
    dpupreagg_exec_methods.ShutdownCustomScan  = pgstromSharedStateShutdownDSM;
    dpupreagg_exec_methods.ExplainCustomScan   = pgstromExplainTaskState;
    /* hook registration */
    create_upper_paths_next = create_upper_paths_hook;
    create_upper_paths_hook = dpupreagg_add_custompath;
}
