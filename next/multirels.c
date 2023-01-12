/*
 * multirels.c
 *
 * Multi-relations join accelerated with GPU processors
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "cuda_common.h"

/* static variables */
static CustomScanMethods	multirels_plan_methods;
static CustomExecMethods	multirels_exec_methods;



/*
 * multirels_create_plan
 */
Plan *
multirels_create_plan(GpuJoinInfo *gj_info,
					  List *custom_plans)
{
	CustomScan *cscan = makeNode(CustomScan);
	Cost		total_cost = 0.0;
	Cardinality	plan_rows = 0.0;
	bool		parallel_safe = true;
	ListCell   *lc;

	/* sanity checks */
	Assert(gj_info->num_rels == list_length(custom_plans));
	foreach (lc, custom_plans)
	{
		Plan   *plan = lfirst(lc);

		total_cost += plan->total_cost;
		plan_rows += plan->plan_rows;
		if (!plan->parallel_safe)
			parallel_safe = false;
	}
	cscan->scan.plan.startup_cost = total_cost;
	cscan->scan.plan.total_cost = total_cost;
	cscan->scan.plan.plan_rows = plan_rows;
	cscan->scan.plan.parallel_aware = false;
	cscan->scan.plan.parallel_safe = parallel_safe;
	cscan->custom_plans = custom_plans;
	cscan->methods = &multirels_plan_methods;

	return &cscan->scan.plan;
}

/*
 * CreateMultiRelsState
 */
static Node *
CreateMultiRelsState(CustomScan *cscan)
{
    return NULL;
}

/*
 * ExecInitMultiRels
 */
static void
ExecInitMultiRels(CustomScanState *node,
				  EState *estate,
				  int eflags)
{}

/*
 * ExecMultiRels
 */
static TupleTableSlot *
ExecMultiRels(CustomScanState *node)
{
	elog(ERROR, "Bug? CustomScan(MultiRels) does not support tuple one-by-one mode");
	return NULL;
}

/*
 * ExecEndMultiRels
 */
static void
ExecEndMultiRels(CustomScanState *node)
{}

/*
 * ExecReScanMultiRels
 */
static void
ExecReScanMultiRels(CustomScanState *node)
{}

/*
 * ExecMultiRelsEstimateDSM
 */
static Size
ExecMultiRelsEstimateDSM(CustomScanState *node,
						 ParallelContext *pcxt)
{
	return 0;
}

/*
 * ExecMultiRelsInitDSM
 */
static void
ExecMultiRelsInitDSM(CustomScanState *node,
					 ParallelContext *pcxt,
					 void *coordinate)
{}

/*
 * ExecMultiRelsInitWorker
 */
static void
ExecMultiRelsInitWorker(CustomScanState *node,
						shm_toc *toc,
						void *coordinate)
{}

/*
 * ExecShutdownMultiRels
 */
static void
ExecShutdownMultiRels(CustomScanState *node)
{}

/*
 * ExplainMultiRels
 */
static void
ExplainMultiRels(CustomScanState *node,
				 List *ancestors,
				 ExplainState *es)
{}

/*
 * pgstrom_init_multirels
 */
void
pgstrom_init_multirels(void)
{
	/* setup plan methods */
	memset(&multirels_plan_methods, 0, sizeof(CustomScanMethods));
	multirels_plan_methods.CustomName         = "MultiRels";
	multirels_plan_methods.CreateCustomScanState  = CreateMultiRelsState;

	/* setup exec methods */
	memset(&multirels_exec_methods, 0, sizeof(CustomExecMethods));
	multirels_exec_methods.CustomName         = "MultiRels";
	multirels_exec_methods.BeginCustomScan    = ExecInitMultiRels;
	multirels_exec_methods.ExecCustomScan     = ExecMultiRels;
	multirels_exec_methods.EndCustomScan      = ExecEndMultiRels;
	multirels_exec_methods.ReScanCustomScan   = ExecReScanMultiRels;
	multirels_exec_methods.EstimateDSMCustomScan = ExecMultiRelsEstimateDSM;
	multirels_exec_methods.InitializeDSMCustomScan = ExecMultiRelsInitDSM;
	multirels_exec_methods.InitializeWorkerCustomScan = ExecMultiRelsInitWorker;
	multirels_exec_methods.ShutdownCustomScan = ExecShutdownMultiRels;
	multirels_exec_methods.ExplainCustomScan  = ExplainMultiRels;

	
}
