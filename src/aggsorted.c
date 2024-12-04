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
	Expr	   *hash_expr;
	int			hash_anum;
	List	   *sort_keys;
	List	   *having_quals;
	double		num_groups;
	double		num_partitions;
	double		input_nrows;
} AggSortedPlanInfo;

typedef struct
{
	CustomScanState		css;
	AggSortedPlanInfo  *asp_info;
} AggSortedState;


/* static variables */
static bool					pgstrom_enable_aggsorted = true;		/* GUC */
static CustomPathMethods	aggsorted_path_methods;
static CustomScanMethods	aggsorted_plan_methods;
static CustomExecMethods	aggsorted_exec_methods;

/*
 * form_aggsorted_plan_info
 */
static void
form_aggsorted_plan_info(CustomScan *cscan, AggSortedPlanInfo *asp_info)
{
	List	   *privs = NIL;
	List	   *exprs = NIL;

	exprs = lappend(exprs, asp_info->hash_expr);
	privs = lappend(privs, makeInteger(asp_info->hash_anum));
	privs = lappend(privs, asp_info->sort_keys);
	exprs = lappend(exprs, asp_info->having_quals);
	privs = lappend(privs, __makeFloat(asp_info->num_groups));
	privs = lappend(privs, __makeFloat(asp_info->num_partitions));
	privs = lappend(privs, __makeFloat(asp_info->input_nrows));

	cscan->custom_exprs = exprs;
	cscan->custom_private = privs;
}

/*
 * deform_aggsorted_plan_info
 */
static AggSortedPlanInfo *
deform_aggsorted_plan_info(CustomScan *cscan)
{
	AggSortedPlanInfo *asp_info = palloc0(sizeof(AggSortedPlanInfo));
	List	   *privs = cscan->custom_private;
	List	   *exprs = cscan->custom_exprs;
	int			pindex = 0;
	int			eindex = 0;

	asp_info->hash_expr = list_nth(exprs, eindex++);
	asp_info->hash_anum = intVal(list_nth(privs, pindex++));
	asp_info->sort_keys = list_nth(privs, pindex++);
	asp_info->having_quals = list_nth(exprs, eindex++);
	asp_info->num_groups = floatVal(list_nth(privs, pindex++));
	asp_info->num_partitions = floatVal(list_nth(privs, pindex++));
	asp_info->input_nrows = floatVal(list_nth(privs, pindex++));

	return asp_info;
}

/*
 * create_aggsorted_path
 */
#define LOG2(x) 	(log(x) / 0.693147180559945)

static Path *
create_aggsorted_path(PlannerInfo *root,
					  RelOptInfo *group_rel,
					  PathTarget *target_final,
					  AggClauseCosts *agg_clause_costs,
					  List *having_quals,
					  List *window_pathkeys,
					  Path *sub_path,
					  double num_groups,
					  double num_partitions,
					  double input_nrows,
					  Expr *hash_expr,
					  int hash_anum,
					  List *sort_keys,
					  List *part_keys)
{
	AggSortedPlanInfo *asp_info;
	CustomPath *cpath;
	double		output_nrows = num_groups;
	double		partition_sz;
	Cost		startup_cost = sub_path->total_cost;
	Cost		run_cost = 0.0;
	Cost		comp_cost = 2.0 * cpu_operator_cost;
	Cost		sort_cost;

	/* cost estimation for hash-based aggreagation */
	/* (should be filly on-memory, so ignore disk cost here) */
	startup_cost += agg_clause_costs->transCost.startup;
	startup_cost += agg_clause_costs->transCost.per_tuple * num_groups;

	/* cost of computing hash value */
	startup_cost += (cpu_operator_cost * (list_length(sort_keys) -
										  list_length(part_keys))) * num_groups;
	startup_cost += agg_clause_costs->finalCost.startup;
	run_cost = cpu_tuple_cost * num_groups;

	/* cost for HAVING quals */
	if (having_quals)
	{
		QualCost	qual_cost;

		cost_qual_eval(&qual_cost, having_quals, root);
		startup_cost += qual_cost.startup;
		run_cost += output_nrows * qual_cost.per_tuple;

		output_nrows = clamp_row_est(output_nrows *
									 clauselist_selectivity(root,
															having_quals,
															0,
															JOIN_INNER,
															NULL));
	}

	/* cost for SORT portion */
	partition_sz = Max(num_groups / num_partitions, 1.0);
	sort_cost = comp_cost * num_groups * LOG2(partition_sz);
	startup_cost += sort_cost / num_partitions;
	run_cost     += (sort_cost - sort_cost / num_partitions);

	/* build Agg::Sorted CustomPath */
	asp_info = palloc0(sizeof(AggSortedPlanInfo));
	asp_info->hash_expr          = hash_expr;
	asp_info->hash_anum          = hash_anum;
	asp_info->sort_keys          = sort_keys;
	asp_info->having_quals       = having_quals;
	asp_info->num_groups         = num_groups;
	asp_info->num_partitions     = num_partitions;
	asp_info->input_nrows        = input_nrows;

	cpath = makeNode(CustomPath);
	cpath->path.pathtype         = T_CustomScan;
	cpath->path.parent           = group_rel;
	cpath->path.pathtarget       = target_final;
	cpath->path.param_info       = NULL;
	cpath->path.parallel_aware   = false;
	cpath->path.parallel_safe    = (group_rel->consider_parallel &&
									sub_path->parallel_safe);
	cpath->path.parallel_workers = sub_path->parallel_workers;
	cpath->path.rows             = num_groups;
	cpath->path.startup_cost     = startup_cost;
	cpath->path.total_cost       = startup_cost + run_cost;
	cpath->path.pathkeys         = window_pathkeys;
	cpath->custom_paths          = list_make1(sub_path);
	cpath->custom_private        = list_make1(asp_info);
	cpath->methods               = &aggsorted_path_methods;

	return &cpath->path;
}

/*
 * __build_aggsorted_hashfunc
 */
static Expr *
__build_aggsorted_hashfunc(PlannerInfo *root,
						   List *window_pathkeys,
						   List *window_clauses,
						   PathTarget *preagg_target,
						   List **p_sort_keys,
						   List **p_part_keys)
{
	WindowClause *wc;
	ListCell   *lc1, *lc2;
	FuncExpr   *hash_func = NULL;
	List	   *part_keys = NIL;
	List	   *sort_keys = NIL;

	if (window_clauses == NIL)
		return NULL;		/* no window function */
	wc = (WindowClause *)linitial(window_clauses);
	if (list_length(wc->partitionClause) +
		list_length(wc->orderClause) != list_length(window_pathkeys))
		return NULL;		/* not consistent */

	foreach (lc1, window_pathkeys)
	{
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

		foreach (lc2, preagg_target->exprs)
		{
			Expr		   *expr = lfirst(lc2);

			if (equal(expr, em_expr))
				goto found;
			part_attnum++;
		}
		return NULL;		/* not found */
	found:
		if (list_length(part_keys) < list_length(wc->partitionClause))
		{
			Oid				type_oid = exprType((Node *)em_expr);
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
			part_keys = lappend(part_keys, em_expr);
		}
		sort_keys = lappend_int(sort_keys, part_attnum);
	}
	if (p_sort_keys)
		*p_sort_keys = sort_keys;
	if (p_part_keys)
		*p_part_keys = part_keys;
	return (Expr *)hash_func;
}

/*
 * try_add_final_aggsorted_paths
 */
void
try_add_final_aggsorted_paths(PlannerInfo *root,
							  RelOptInfo *group_rel,
							  PathTarget *target_final,
							  AggClauseCosts *agg_clause_costs,
							  List *having_quals,
							  Path *preagg_path,
							  bool be_parallel,
							  double num_groups,
							  double input_nrows)
{
	PlannerInfo *parent_root = root->parent_root;
	Path	   *sub_path = preagg_path;
	List	   *window_pathkeys;
	List	   *window_clause;
	Expr	   *hash_expr;
	List	   *sort_keys;
	List	   *part_keys;
	int			hash_anum;
	double		num_partitions;
	Path	   *aggsorted_path;

	if (!pgstrom_enable_aggsorted)
		return;		/* aggsorted disabled */

	Assert(pgstrom_is_gpupreagg_path(preagg_path));
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
										   preagg_path->pathtarget,
										   &sort_keys,
										   &part_keys);
	if (!hash_expr)
		return;		/* unsupported */
	Assert(list_length(sort_keys) > list_length(part_keys));

	/* estimate number of partitions */
	num_partitions = estimate_num_groups(root, part_keys,
										 input_nrows,
										 NULL, NULL);

	/* duplicate sub_path (no need to copy recursive) */
	preagg_path = pmemdup(preagg_path, sizeof(CustomPath));
	preagg_path->pathtarget = copy_pathtarget(preagg_path->pathtarget);
	preagg_path->pathtarget->exprs = lappend(preagg_path->pathtarget->exprs,
											 hash_expr);
	hash_anum = list_length(sub_path->pathtarget->exprs);

	/* inject parallel Gather node */
	if (be_parallel)
	{
		sub_path = (Path *)
			create_gather_path(root,
							   group_rel,
							   preagg_path,
							   preagg_path->pathtarget,
							   NULL,
							   &num_groups);
	}

	/* creation of aggsorted path */
	aggsorted_path = create_aggsorted_path(root,
										   group_rel,
										   target_final,
										   agg_clause_costs,
										   having_quals,
										   window_pathkeys,
										   sub_path,
										   num_groups,
										   num_partitions,
										   input_nrows,
										   hash_expr,
										   hash_anum,
										   sort_keys,
										   part_keys);
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
	CustomScan *cscan = makeNode(CustomScan);
	AggSortedPlanInfo *asp_info = linitial(best_path->custom_private);
	Plan	   *sub_plan = linitial(custom_plans);

	cscan->scan.plan.targetlist = tlist;
	//cscan->scan.plan.qual = having;
	cscan->scan.plan.lefttree = sub_plan;
	cscan->flags = 0;
	cscan->methods = &aggsorted_plan_methods;
	cscan->custom_scan_tlist = sub_plan->targetlist;
	form_aggsorted_plan_info(cscan, asp_info);

	return &cscan->scan.plan;
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
	ass->asp_info = deform_aggsorted_plan_info(cscan);

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
