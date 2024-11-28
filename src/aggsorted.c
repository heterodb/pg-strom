/*
 * aggsort.c
 *
 * Final Aggregation + Partial Sorting; A lightweight pre-processing for
 * window functions.
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"



typedef struct
{
	CustomScanState		css;

} AggSortedState;


/* static variables */
static bool					pgstrom_enable_aggsorted = true;		/* GUC */
static CustomPathMethods	aggsorted_path_methods;
static CustomScanMethods	aggsorted_plan_methods;
static CustomExecMethods	aggsorted_exec_methods;



/*
 * try_add_one_final_aggsorted_path
 */
static void
try_add_one_final_aggsorted_path(PlannerInfo *root,
								 RelOptInfo *group_rel,
								 PathTarget *target_final,
								 List *having_quals,
								 List *window_pathkeys,
								 Path *part_path)
{
	
	
}

/*
 * validate_window_function_pathkeys
 */
static bool
validate_window_pathkeys(PlannerInfo *root,
						 List *window_pathkeys,
						 List *window_clauses,
						 PathTarget *target_partial)
{
	WindowClause *wc;
	ListCell   *lc1, *lc2;

	if (window_clauses == NIL)
		return false;
	wc = (WindowClause *)linitial(window_clauses);
	if (list_length(wc->partitionClause) +
		list_length(wc->orderClause) != list_length(window_pathkeys))
		return false;	/* not consistent */

	foreach (lc1, window_pathkeys)
	{
		PathKey	   *pkey = lfirst(lc1);
		EquivalenceClass *ec = pkey->pk_eclass;
		EquivalenceMember *em;

		if (list_length(ec->ec_members) != 1 ||
			ec->ec_sources != NIL ||
			ec->ec_derives != NIL)
			return false;	/* not supported */
		em = (EquivalenceMember *)linitial(ec->ec_members);
		foreach (lc2, target_partial->exprs)
		{
			Expr   *expr = lfirst(lc2);

			if (equal(expr, em->em_expr))
			{
				//XXX - check hash-function
				break;
			}
		}
		if (!lc2)
			return false;	/* not found */
	}
	return true;
}

/*
 * try_add_final_aggsorted_paths
 */
void
try_add_final_aggsorted_paths(PlannerInfo *root,
							  RelOptInfo *group_rel,
							  PathTarget *target_final,
							  List *having_quals,
							  Path *part_path,
							  double num_groups)
{
	PlannerInfo	   *parent_root = root->parent_root;
	ListCell	   *lc1, *lc2;

	if (!pgstrom_enable_aggsorted)
		return;		/* aggsorted disabled */

	if (root->window_pathkeys != NIL)
	{
		Query  *parse = root->parse;

		if (validate_window_pathkeys(root,
									 root->window_pathkeys,
									 parse->windowClause,
									 part_path->pathtarget))
			try_add_one_final_aggsorted_path(root,
											 group_rel,
											 target_final,
											 having_quals,
											 root->window_pathkeys,
											 part_path);
	}
	else if (parent_root &&
			 parent_root->window_pathkeys != NIL)
	{
		Query  *parse = parent_root->parse;

		if (validate_window_pathkeys(root,
									 root->query_pathkeys,
									 parse->windowClause,
									 part_path->pathtarget))
			try_add_one_final_aggsorted_path(root,
											 group_rel,
											 target_final,
											 having_quals,
											 root->query_pathkeys,
											 part_path);
	}
}

/*
 * PlanAggSortedPath
 */
static Plan *
PlanAggSortedPath(PlannerInfo *root,
				RelOptInfo *rel,
				CustomPath *best_path,
				List *tlist,
				List *clauses,
				List *custom_plans)
{
	return NULL;
}

/*
 * CreateAggSortedState
 */
static Node *
CreateAggSortedState(CustomScan *cscan)
{
	AggSortedState	   *ass = palloc0(sizeof(AggSortedState));

	Assert(cscan->methods == &aggsorted_plan_methods);
	NodeSetTag(ass, T_CustomScanState);
	ass->css.flags = cscan->flags;
	ass->css.methods = &aggsorted_exec_methods;

	return (Node *)ass;
}

/*
 * BeginAggSorted
 */
static void
BeginAggSorted(CustomScanState *node, EState *estate, int eflags)
{}

/*
 * ExecAggSorted
 */
static TupleTableSlot *
ExecAggSorted(CustomScanState *node)
{
	return NULL;
}

/*
 * EndAggSorted
 */
static void
EndAggSorted(CustomScanState *node)
{}

/*
 * ReScanAggSorted
 */
static void
ReScanAggSorted(CustomScanState *node)
{}

/*
 * ExplainAggSorted
 */
static void
ExplainAggSorted(CustomScanState *node,
			   List *ancestors,
			   ExplainState *es)
{}

/*
 * pgstrom_init_aggsorted
 */
void
pgstrom_init_aggsorted(void)
{
	/* pg_strom.enable_aggsort */
	DefineCustomBoolVariable("pg_strom.enable_aggsort",
							 "Enables the use of DPU accelerated full-scan",
							 NULL,
							 &pgstrom_enable_aggsorted,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* setup path methods */
	memset(&aggsorted_path_methods, 0, sizeof(aggsorted_path_methods));
	aggsorted_path_methods.CustomName			= "Agg::Sorted";
	aggsorted_path_methods.PlanCustomPath		= PlanAggSortedPath;

	/* setup plan methods */
	memset(&aggsorted_plan_methods, 0, sizeof(aggsorted_plan_methods));
	aggsorted_plan_methods.CustomName			= "Agg::Sorted";
	aggsorted_plan_methods.CreateCustomScanState = CreateAggSortedState;
	RegisterCustomScanMethods(&aggsorted_plan_methods);

	/* setup exec methods (no parallel callbacks are needed) */
	memset(&aggsorted_exec_methods, 0, sizeof(aggsorted_exec_methods));
	aggsorted_exec_methods.CustomName			= "Agg::Sorted";
	aggsorted_exec_methods.BeginCustomScan		= BeginAggSorted;
	aggsorted_exec_methods.ExecCustomScan		= ExecAggSorted;
	aggsorted_exec_methods.EndCustomScan		= EndAggSorted;
	aggsorted_exec_methods.ReScanCustomScan		= ReScanAggSorted;
	aggsorted_exec_methods.ExplainCustomScan	= ExplainAggSorted;
}
