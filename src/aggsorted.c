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
 * create_aggsorted_path
 */
static Path *
create_aggsorted_path(PlannerInfo *root,
					  RelOptInfo *group_rel,
					  Path *part_path,
					  PathTarget *target_final,
					  List *having_quals,
					  double num_groups,
					  int hash_key,
					  List *sort_keys)
{
	return NULL;
}

/*
 * __build_aggsorted_hashfunc
 */
static Expr *
__build_aggsorted_hashfunc(PlannerInfo *root,
						   List *window_pathkeys,
						   List *window_clauses,
						   Path *part_path,
						   List **p_sort_keys)
{
	WindowClause *wc;
	ListCell   *lc1, *lc2;
	FuncExpr   *hash_func = NULL;
	List	   *sort_keys = NIL;

	if (window_clauses == NIL)
		return NULL;		/* no window function */
	wc = (WindowClause *)linitial(window_clauses);
	if (list_length(wc->partitionClause) +
		list_length(wc->orderClause) != list_length(window_pathkeys))
		return NULL;		/* not consistent */

	foreach (lc1, window_pathkeys)
	{
		PathTarget	   *part_target = part_path->pathtarget;
		PathKey		   *pkey = lfirst(lc1);
		EquivalenceClass *ec = pkey->pk_eclass;
		EquivalenceMember *em;
		int				part_attnum = 1;
		Expr		   *em_expr;
		List		   *func_args;

		if (list_length(ec->ec_members) != 1 ||
			ec->ec_sources != NIL ||
			ec->ec_derives != NIL)
			return NULL;		/* not supported */
		em = (EquivalenceMember *)linitial(ec->ec_members);
		/* strip Relabel for equal() comparison */
		for (em_expr = em->em_expr;
			 IsA(em_expr, RelabelType);
			 em_expr = ((RelabelType *)em_expr)->arg);

		foreach (lc2, part_target->exprs)
		{
			Expr		   *expr = lfirst(lc2);

			if (equal(expr, em_expr))
				goto found;
			part_attnum++;
		}
		return NULL;		/* not found */
	found:
		if (list_length(sort_keys) < list_length(wc->partitionClause))
		{
			Oid		type_oid = exprType((Node *)em_expr);
			devtype_info   *dtype = pgstrom_devtype_lookup(type_oid);

			if (!dtype || !OidIsValid(dtype->type_devhash))
				return NULL;	/* not supported */
			if (!hash_func)
				hash_func = (FuncExpr *)makeConst(INT8OID,
												  -1,
												  InvalidOid,
												  sizeof(int64),
												  (Datum) 0,
												  false,	/* isnull */
												  true);	/* byval */
			func_args = list_make2(copyObject(em_expr),
								   hash_func);
			hash_func = makeFuncExpr(dtype->type_devhash,
									 INT8OID,
									 func_args,
									 InvalidOid,
									 InvalidOid,
									 COERCE_IMPLICIT_CAST);
		}
		sort_keys = lappend_int(sort_keys, part_attnum);
	}
	if (p_sort_keys)
		*p_sort_keys = sort_keys;
	return (Expr *)hash_func;
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
	PlannerInfo *parent_root = root->parent_root;
	List	   *window_pathkeys;
	List	   *window_clause;
	Expr	   *hash_expr;
	List	   *sort_keys;
	int			hash_key;
	Path	   *aggsorted_path;

	if (!pgstrom_enable_aggsorted)
		return;		/* aggsorted disabled */

	Assert(root->parse->groupClause == NIL);	/* No GROUPING SET */
	Assert(pgstrom_is_gpupreagg_path(part_path));
	if (root->window_pathkeys != NIL)
	{
		window_pathkeys = root->window_pathkeys;
		window_clause = root->parse->windowClause;
	}
	else if (parent_root &&
             parent_root->window_pathkeys != NIL)
	{
		window_pathkeys = root->query_pathkeys;
		window_clause = parent_root->parse->windowClause;
	}
	else
	{
		return;		/* unsupported */
	}
	hash_expr = __build_aggsorted_hashfunc(root,
										   window_pathkeys,
										   window_clause,
										   part_path,
										   &sort_keys);
	if (!hash_expr)
		return;		/* unsupported */

	/* duplicate part_path (no need to copy recursive) */
	part_path = pmemdup(part_path, sizeof(CustomPath));
	part_path->pathtarget = copy_pathtarget(part_path->pathtarget);
	part_path->pathtarget->exprs = lappend(part_path->pathtarget->exprs,
										   hash_expr);
	hash_key = list_length(part_path->pathtarget->exprs);

	/* creation of aggsorted path */
	aggsorted_path = create_aggsorted_path(root,
										   group_rel,
										   part_path,
										   target_final,
										   having_quals,
										   num_groups,
										   hash_key,
										   sort_keys);
	if (aggsorted_path)
		add_path(group_rel, aggsorted_path);
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
