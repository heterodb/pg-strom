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
	double		num_partitions;
} HashedSortPlanInfo;

typedef struct
{
	CustomScanState		css;
	HashedSortPlanInfo  *hsp_info;
} HashedSortState;


/* static variables */
static bool					pgstrom_enable_clustered_windowagg = true;	/* GUC */
static CustomPathMethods	hashedsort_path_methods;
static CustomScanMethods	hashedsort_plan_methods;
static CustomExecMethods	hashedsort_exec_methods;

/*
 * form_hashedsort_plan_info
 */
static void
form_hashedsort_plan_info(CustomScan *cscan, HashedSortPlanInfo *hsp_info)
{
	List	   *privs = NIL;
	List	   *exprs = NIL;

	exprs = lappend(exprs, hsp_info->hash_expr);
	privs = lappend(privs, makeInteger(hsp_info->hash_anum));
	exprs = lappend(exprs, hsp_info->sort_keys);
	privs = lappend(privs, __makeFloat(hsp_info->num_partitions));

	cscan->custom_exprs = exprs;
	cscan->custom_private = privs;
}

/*
 * deform_hashedsort_plan_info
 */
static HashedSortPlanInfo *
deform_hashedsort_plan_info(CustomScan *cscan)
{
	HashedSortPlanInfo *hsp_info = palloc0(sizeof(HashedSortPlanInfo));
	List	   *privs = cscan->custom_private;
	List	   *exprs = cscan->custom_exprs;
	int			pindex = 0;
	int			eindex = 0;

	hsp_info->hash_expr = list_nth(exprs, eindex++);
	hsp_info->hash_anum = intVal(list_nth(privs, pindex++));
	hsp_info->sort_keys = list_nth(exprs, eindex++);
	hsp_info->num_partitions = floatVal(list_nth(privs, pindex++));

	return hsp_info;
}

/*
 * create_hashed_sort_path
 */
#define LOG2(x) 	(log(x) / 0.693147180559945)

static CustomPath *
create_hashed_sort_path(PlannerInfo *root,
						RelOptInfo *group_rel,
						List *window_pathkeys,
						PathTarget *target_final,
						Path *sub_path,
						int hash_anum,
						Expr *hash_expr,
						List *sort_keys,
						double num_partitions)
{
	HashedSortPlanInfo *hsp_info;
	CustomPath *cpath;
	double		num_groups = sub_path->rows;
	double		partition_sz;
	Cost		startup_cost = sub_path->total_cost;
	Cost		run_cost = 0.0;
	Cost		comp_cost = 2.0 * cpu_operator_cost;
	Cost		sort_cost;

	/* cost to store the input */
	startup_cost += cpu_tuple_cost * num_groups;
	/* cost for sort portion */
	partition_sz = Max(num_groups / num_partitions, 1.0);
	sort_cost = comp_cost * num_groups * LOG2(partition_sz);
	startup_cost += sort_cost / num_partitions;
	run_cost += sort_cost - (sort_cost / num_partitions);

	/* setup HashedSortPlanInfo */
	hsp_info = palloc0(sizeof(HashedSortPlanInfo));
	hsp_info->hash_expr          = hash_expr;
	hsp_info->hash_anum          = hash_anum;
	hsp_info->sort_keys          = sort_keys;
	hsp_info->num_partitions     = num_partitions;

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
	cpath->flags                 = CUSTOMPATH_SUPPORT_PROJECTION;
	cpath->custom_paths          = list_make1(sub_path);
	cpath->custom_private        = list_make1(hsp_info);
	cpath->methods               = &hashedsort_path_methods;

	return cpath;
}

/*
 * __build_partial_key_hashfunc
 */
static Expr *
__build_partial_key_hashfunc(PlannerInfo *root,
							 List *window_pathkeys,
							 List *window_clauses,
							 PathTarget *path_target,
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
		int				sort_attnum = 1;
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

		foreach (lc2, path_target->exprs)
		{
			Expr	   *expr = lfirst(lc2);

			if (equal(expr, em_expr))
				goto found;
			sort_attnum++;
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
									 COERCE_EXPLICIT_CALL);
			part_keys = lappend(part_keys, em_expr);
		}
		sort_keys = lappend(sort_keys, em_expr);
	}
	if (p_sort_keys)
		*p_sort_keys = sort_keys;
	if (p_part_keys)
		*p_part_keys = part_keys;
	return (Expr *)hash_func;
}

/*
 * try_add_hashed_sort_path
 */
void
try_add_hashed_sort_path(PlannerInfo *root,
						 RelOptInfo *group_rel,
						 AggPath *final_path,
						 CustomPath *preagg_path,
						 bool be_parallel,
						 double input_nrows)
{
	PlannerInfo *parent_root = root->parent_root;
	List	   *window_pathkeys;
	List	   *window_clause;
	Expr	   *hash_expr;
	List	   *sort_keys;
	List	   *part_keys;
	int			hash_anum;
	double		num_partitions;
	double		num_groups = final_path->path.rows;
	Path	   *sub_path;
	AggPath	   *agg_path;
	CustomPath *cpath;

	if (!pgstrom_enable_clustered_windowagg)
		return;

	Assert(pgstrom_is_gpupreagg_path((Path *)preagg_path));
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
	hash_expr = __build_partial_key_hashfunc(root,
											 window_pathkeys,
											 window_clause,
											 preagg_path->path.pathtarget,
											 &sort_keys,
											 &part_keys);
	if (!hash_expr)
		return;
	Assert(list_length(sort_keys) > list_length(part_keys));
	num_partitions = estimate_num_groups(root, part_keys,
										 input_nrows,
										 NULL, NULL);

	/* attach hash_expr as nokey item on the PreAggPath */
	sub_path = xpupreagg_path_attach_nokey(preagg_path, hash_expr);

	/* inject parallel Gather node */
	if (be_parallel)
	{
		sub_path = (Path *)
			create_gather_path(root,
							   group_rel,
							   sub_path,
							   sub_path->pathtarget,
							   NULL,
							   &num_groups);
	}
	/* duplicate AggPath */
	agg_path = pmemdup(final_path, sizeof(AggPath));
	agg_path->path.pathtarget = copy_pathtarget(agg_path->path.pathtarget);
	agg_path->path.pathtarget->exprs = lappend(agg_path->path.pathtarget->exprs,
											   hash_expr);
	hash_anum = list_length(agg_path->path.pathtarget->exprs);
	agg_path->subpath = sub_path;

	/* creation of hashed-sort path (no need to copy recursively) */
	cpath = create_hashed_sort_path(root,
									group_rel,
									window_pathkeys,
									final_path->path.pathtarget,
									&agg_path->path,
									hash_anum,
									hash_expr,
									sort_keys,
									num_partitions);
	if (cpath)
		add_path(group_rel, (Path *)cpath);
}

/*
 * PlanHashedSortPath
 */
static Plan *
PlanHashedSortPath(PlannerInfo *root,
				   RelOptInfo *rel,
				   CustomPath *best_path,
				   List *tlist,
				   List *clauses,
				   List *custom_plans)
{
	CustomScan *cscan = makeNode(CustomScan);
	HashedSortPlanInfo *hsp_info = linitial(best_path->custom_private);
	Plan	   *sub_plan = linitial(custom_plans);
	List	   *cs_tlist = list_copy(tlist);
	TargetEntry *tle;

	tle = makeTargetEntry(hsp_info->hash_expr,
						  list_length(cs_tlist),
						  NULL,
						  false);
	cs_tlist = lappend(cs_tlist, tle);

	cscan->scan.plan.targetlist = tlist;
	cscan->scan.plan.lefttree = sub_plan;
	cscan->flags = best_path->flags;
	cscan->methods = &hashedsort_plan_methods;
	cscan->custom_scan_tlist = cs_tlist;

	form_hashedsort_plan_info(cscan, hsp_info);

	return &cscan->scan.plan;
}

/*
 * CreateHashedSortState
 */
static Node *
CreateHashedSortState(CustomScan *cscan)
{
	HashedSortState	   *hss = palloc0(sizeof(HashedSortState));

	Assert(cscan->methods == &hashedsort_plan_methods);
	NodeSetTag(hss, T_CustomScanState);
	hss->css.flags = cscan->flags;
	hss->css.methods = &hashedsort_exec_methods;
	hss->hsp_info = deform_hashedsort_plan_info(cscan);

	return (Node *)hss;
}

/*
 * BeginHashSorted
 */
static void
BeginHashSorted(CustomScanState *node, EState *estate, int eflags)
{
	HashedSortState	   *hss = (HashedSortState *)node;
	HashedSortPlanInfo *hsp_info = hss->hsp_info;
	CustomScan		   *cscan = (CustomScan *)hss->css.ss.ps.plan;

	hss->css.ss.ps.lefttree = ExecInitNode(cscan->scan.plan.lefttree, estate, eflags);
}

/*
 * ExecHashSorted
 */
static TupleTableSlot *
ExecHashSorted(CustomScanState *node)
{
	return NULL;
}

/*
 * EndHashSorted
 */
static void
EndHashSorted(CustomScanState *node)
{
	HashedSortState	   *hss = (HashedSortState *)node;
	
	ExecEndNode(hss->css.ss.ps.lefttree);
}

/*
 * ReScanHashSorted
 */
static void
ReScanHashSorted(CustomScanState *node)
{}

/*
 * ExplainHashSorted
 */
static void
ExplainHashSorted(CustomScanState *node,
				  List *ancestors,
				  ExplainState *es)
{
	HashedSortState	   *hss = (HashedSortState *)node;
	HashedSortPlanInfo *hsp_info = hss->hsp_info;
	CustomScan		   *cscan = (CustomScan *)node->ss.ps.plan;
	List			   *dcontext;
	TargetEntry		   *tle;
	char			   *str;
	ListCell		   *lc;
	StringInfoData		buf;

	/* setup deparse context */
	dcontext = set_deparse_context_plan(es->deparse_cxt,
										node->ss.ps.plan,
										ancestors);
	Assert(hsp_info->hash_anum > 0 &&
		   hsp_info->hash_anum <= list_length(cscan->custom_scan_tlist));
	tle = list_nth(cscan->custom_scan_tlist,
				   hsp_info->hash_anum - 1);
	str = deparse_expression((Node *)tle->expr, dcontext, es->verbose, true);
	ExplainPropertyText("Hash Key", str, es);

	initStringInfo(&buf);
	foreach (lc, hsp_info->sort_keys)
	{
		Var	   *var = lfirst(lc);

		str = deparse_expression((Node *)var, dcontext, es->verbose, true);
		if (lc != list_head(hsp_info->sort_keys))
            appendStringInfo(&buf, ", ");
		appendStringInfoString(&buf, str);
#if 0
		Assert(key > 0 && key <= list_length(cscan->custom_scan_tlist));
		tle = list_nth(cscan->custom_scan_tlist, key - 1);
		if (lc != list_head(hsp_info->sort_keys))
			appendStringInfo(&buf, ", ");
		str = deparse_expression((Node *)tle->expr, dcontext, es->verbose, true);
		appendStringInfoString(&buf, str);
#endif
	}
	ExplainPropertyText("Sort Keys", buf.data, es);
	pfree(buf.data);
}

/*
 * pgstrom_init_hashed_sort
 */
void
pgstrom_init_hashed_sort(void)
{
	/* pg_strom.enable_clustered_windowagg */
	DefineCustomBoolVariable("pg_strom.enable_clustered_windowagg",
							 "Enables clustered input for window-functions",
							 NULL,
							 &pgstrom_enable_clustered_windowagg,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* setup path methods */
	memset(&hashedsort_path_methods, 0, sizeof(hashedsort_path_methods));
	hashedsort_path_methods.CustomName			= "Hashed Sort";
	hashedsort_path_methods.PlanCustomPath		= PlanHashedSortPath;

	/* setup plan methods */
	memset(&hashedsort_plan_methods, 0, sizeof(hashedsort_plan_methods));
	hashedsort_plan_methods.CustomName			= "Hashed Sort";
	hashedsort_plan_methods.CreateCustomScanState = CreateHashedSortState;
	RegisterCustomScanMethods(&hashedsort_plan_methods);

	/* setup exec methods (no parallel callbacks are needed) */
	memset(&hashedsort_exec_methods, 0, sizeof(hashedsort_exec_methods));
	hashedsort_exec_methods.CustomName			= "Hashed Sort";
	hashedsort_exec_methods.BeginCustomScan		= BeginHashSorted;
	hashedsort_exec_methods.ExecCustomScan		= ExecHashSorted;
	hashedsort_exec_methods.EndCustomScan		= EndHashSorted;
	hashedsort_exec_methods.ReScanCustomScan	= ReScanHashSorted;
	hashedsort_exec_methods.ExplainCustomScan	= ExplainHashSorted;
}
