/*
 * dpu_join.c
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
static set_join_pathlist_hook_type set_join_pathlist_next = NULL;
static CustomPathMethods	dpujoin_path_methods;
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
	return NULL;
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

	return NULL;
}

/*
 * ExecInitGpuJoin
 */
static void
ExecInitDpuJoin(CustomScanState *node,
				EState *estate,
				int eflags)
{}

/*
 * DpuJoinReCheckTuple
 */
static bool
DpuJoinReCheckTuple(pgstromTaskState *pts, TupleTableSlot *epq_slot)
{
	return true;
}

/*
 * ExecDpuJoin
 */
static TupleTableSlot *
ExecDpuJoin(CustomScanState *node)
{
	return ExecScan(&node->ss,
					(ExecScanAccessMtd) pgstromExecTaskState,
					(ExecScanRecheckMtd) DpuJoinReCheckTuple);
}

/*
 * ExecEndDpuJoin
 */
static void
ExecEndDpuJoin(CustomScanState *node)
{
	pgstromTaskState *pts = (pgstromTaskState *) node;

	pgstromExecEndTaskState(pts);
}

/*
 * ExecReScanDpuJoin
 */
static void
ExecReScanDpuJoin(CustomScanState *node)
{
	pgstromTaskState *pts = (pgstromTaskState *) node;

	pgstromExecResetTaskState(pts);
}

/*
 * ExecDpuJoinEstimateDSM
 */
static Size
ExecDpuJoinEstimateDSM(CustomScanState *node,
                       ParallelContext *pcxt)
{
    pgstromTaskState *pts = (pgstromTaskState *) node;

    return pgstromSharedStateEstimateDSM(pts);
}

/*
 * ExecDpuJoinInitDSM
 */
static void
ExecDpuJoinInitDSM(CustomScanState *node,
                   ParallelContext *pcxt,
                   void *dsm_addr)
{
    pgstromTaskState *pts = (pgstromTaskState *) node;

    pgstromSharedStateInitDSM(pts, pcxt, dsm_addr);
}

/*
 * ExecDpuJoinInitWorker
 */
static void
ExecDpuJoinInitWorker(CustomScanState *node,
					  shm_toc *toc,
					  void *dsm_addr)
{
	pgstromTaskState *pts = (pgstromTaskState *) node;

	pgstromSharedStateAttachDSM(pts, dsm_addr);
}

/*
 * ExecShutdownDpuJoin
 */
static void
ExecShutdownDpuJoin(CustomScanState *node)
{
	pgstromTaskState *pts = (pgstromTaskState *) node;

	pgstromSharedStateShutdownDSM(pts);
}

/*
 * ExplainDpuJoin
 */
static void
ExplainDpuJoin(CustomScanState *node,
			   List *ancestors,
			   ExplainState *es)
{
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
	dpujoin_exec_methods.BeginCustomScan        = ExecInitDpuJoin;
	dpujoin_exec_methods.ExecCustomScan         = ExecDpuJoin;
	dpujoin_exec_methods.EndCustomScan          = ExecEndDpuJoin;
	dpujoin_exec_methods.ReScanCustomScan       = ExecReScanDpuJoin;
	dpujoin_exec_methods.MarkPosCustomScan      = NULL;
	dpujoin_exec_methods.RestrPosCustomScan     = NULL;
	dpujoin_exec_methods.EstimateDSMCustomScan  = ExecDpuJoinEstimateDSM;
	dpujoin_exec_methods.InitializeDSMCustomScan = ExecDpuJoinInitDSM;
	dpujoin_exec_methods.InitializeWorkerCustomScan = ExecDpuJoinInitWorker;
	dpujoin_exec_methods.ShutdownCustomScan     = ExecShutdownDpuJoin;
	dpujoin_exec_methods.ExplainCustomScan      = ExplainDpuJoin;

	/* hook registration */
	set_join_pathlist_next = set_join_pathlist_hook;
	set_join_pathlist_hook = dpujoin_add_custompath;
}
