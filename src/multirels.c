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

typedef struct
{
	JoinType	join_type;			/* one of JOIN_* */
	List	   *hash_inner;			/* hashing expression for inner-side */
	Oid			gist_index_oid;		/* GiST index oid */
	AttrNumber	gist_index_col;		/* GiST index column number */
	Node	   *gist_clause;		/* GiST index clause */
	Selectivity	gist_selectivity;	/* GiST selectivity */
} MultiRelsInnerInfo;

typedef struct
{
	int		num_rels;
	MultiRelsInnerInfo inners[FLEXIBLE_ARRAY_MEMBER];
} MultiRelsInfo;

/*
 * form_multirels_info
 */
static void
form_multirels_info(CustomScan *cscan, MultiRelsInfo *mr_info)
{
	List   *privs = NIL;
	List   *exprs = NIL;

	privs = lappend(privs, makeInteger(mr_info->num_rels));
	for (int i=0; i < mr_info->num_rels; i++)
	{
		MultiRelsInnerInfo *mr_inner = &mr_info->inners[i];
		List   *__privs = NIL;
		List   *__exprs = NIL;

		__privs = lappend(__privs, makeInteger(mr_inner->join_type));
		__exprs = lappend(__exprs, mr_inner->hash_inner);
		__privs = lappend(__privs, makeInteger(mr_inner->gist_index_oid));
		__privs = lappend(__privs, makeInteger(mr_inner->gist_index_col));
		__exprs = lappend(__exprs, mr_inner->gist_clause);
		__privs = lappend(__privs, __makeFloat(mr_inner->gist_selectivity));

		privs = lappend(privs, __privs);
		exprs = lappend(exprs, __exprs);
	}
	cscan->custom_exprs = exprs;
	cscan->custom_private = privs;
}

/*
 * deform_multirels_info
 */
static MultiRelsInfo *
deform_multirels_info(CustomScan *cscan)
{
	MultiRelsInfo	mr_data;
	MultiRelsInfo  *mr_info;
	List   *privs = cscan->custom_private;
	List   *exprs = cscan->custom_exprs;
	int		pindex = 0;
	int		eindex = 0;

	memset(&mr_data, 0, sizeof(MultiRelsInfo));
	mr_data.num_rels = intVal(list_nth(privs, pindex++));

	mr_info = palloc0(offsetof(MultiRelsInfo, inners[mr_data.num_rels]));
	memcpy(mr_info, &mr_data, offsetof(MultiRelsInfo, inners));
	for (int i=0; i < mr_info->num_rels; i++)
	{
		MultiRelsInnerInfo *mr_inner = &mr_info->inners[i];
		List   *__privs = list_nth(privs, pindex++);
		List   *__exprs = list_nth(exprs, eindex++);
		int		__pindex = 0;
		int		__eindex = 0;

		mr_inner->join_type = intVal(list_nth(__privs, __pindex++));
		mr_inner->hash_inner = list_nth(__exprs, __eindex++);
		mr_inner->gist_index_oid = intVal(list_nth(__privs, __pindex++));
		mr_inner->gist_index_col = intVal(list_nth(__privs, __pindex++));
		mr_inner->gist_clause = list_nth(__exprs, __eindex++);
		mr_inner->gist_selectivity = floatVal(list_nth(__privs, __pindex++));
	}
	return mr_info;
}

/*
 * multirels_create_plan
 */
Plan *
multirels_create_plan(GpuJoinInfo *gj_info,
					  List *custom_plans)
{
	CustomScan	   *cscan = makeNode(CustomScan);
	MultiRelsInfo  *mr_info;
	Cost			total_cost = 0.0;
	Cardinality		plan_rows = 0.0;
	bool			parallel_safe = true;
	List		   *temp_list;
	List		   *vars_list = NIL;
	List		   *tlist_dev = NIL;
	ListCell	   *lc1, *lc2;

	/* sanity checks */
	Assert(gj_info->num_rels == list_length(custom_plans));
	mr_info = palloc0(offsetof(MultiRelsInfo, inners[gj_info->num_rels]));
	mr_info->num_rels = gj_info->num_rels;
	for (int i=0; i < gj_info->num_rels; i++)
	{
		GpuJoinInnerInfo *gj_inner = &gj_info->inners[i];
		MultiRelsInnerInfo *mr_inner = &mr_info->inners[i];

		mr_inner->join_type        = gj_inner->join_type;
		mr_inner->hash_inner       = gj_inner->hash_inner;
		mr_inner->gist_index_oid   = gj_inner->gist_index_oid;
		mr_inner->gist_index_col   = gj_inner->gist_index_col;
		mr_inner->gist_clause      = gj_inner->gist_clause;
		mr_inner->gist_selectivity = gj_inner->gist_selectivity;

		temp_list = pull_var_clause((Node *)gj_inner->hash_inner, 0);
		vars_list = list_concat(vars_list, temp_list);
		temp_list = pull_var_clause((Node *)gj_inner->gist_clause, 0);
		vars_list = list_concat(vars_list, temp_list);
	}

	foreach (lc1, custom_plans)
	{
		Plan   *plan = lfirst(lc1);

		plan_rows += plan->plan_rows;
		total_cost += plan->total_cost;
		total_cost += cpu_tuple_cost * plan->plan_rows;
		if (!plan->parallel_safe)
			parallel_safe = false;
	}

	foreach (lc1, vars_list)
	{
		Expr   *var = lfirst(lc1);

		foreach (lc2, tlist_dev)
		{
			TargetEntry *tle = lfirst(lc2);

			if (equal(tle->expr, var))
				break;
		}
		if (!lc2)
		{
			AttrNumber	resno = list_length(tlist_dev) + 1;
			TargetEntry *tle = makeTargetEntry(var, resno, NULL, true);

			tlist_dev = lappend(tlist_dev, tle);
		}
	}
	
	cscan->scan.plan.startup_cost = total_cost;
	cscan->scan.plan.total_cost = total_cost;
	cscan->scan.plan.plan_rows = plan_rows;
	cscan->scan.plan.parallel_aware = false;
	cscan->scan.plan.parallel_safe = parallel_safe;
	cscan->custom_plans = custom_plans;
	cscan->custom_scan_tlist = tlist_dev;
	cscan->methods = &multirels_plan_methods;
	form_multirels_info(cscan, mr_info);

	return &cscan->scan.plan;
}

/* ----------------------------------------------------------------
 *
 * Executor routines of MultiRels
 *
 * ----------------------------------------------------------------
 */
typedef struct
{
	CustomScanState css;
	MultiRelsInfo  *mr_info;
} MultiRelsState;

/*
 * CreateMultiRelsState
 */
static Node *
CreateMultiRelsState(CustomScan *cscan)
{
	MultiRelsState *mrs = palloc0(sizeof(MultiRelsState));

	/* Set tag and executor callbacks */
	NodeSetTag(mrs, T_CustomScanState);
	mrs->css.flags = cscan->flags;
	Assert(cscan->methods == &multirels_plan_methods);
    mrs->css.methods = &multirels_exec_methods;
    mrs->mr_info = deform_multirels_info(cscan);

	return (Node *)mrs;
}

/*
 * ExecInitMultiRels
 */
static void
ExecInitMultiRels(CustomScanState *node,
				  EState *estate,
				  int eflags)
{
	MultiRelsState *mrs = (MultiRelsState *)node;
	CustomScan	   *cscan = (CustomScan *)node->ss.ps.plan;
	ListCell	   *lc;

	foreach (lc, cscan->custom_plans)
	{
		Plan	   *plan = lfirst(lc);
		PlanState  *ps = ExecInitNode(plan, estate, eflags);

		mrs->css.custom_ps = lappend(mrs->css.custom_ps, ps);
	}
}

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
