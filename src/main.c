/*
 * main.c
 *
 * Entrypoint of PG-Strom extension
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

PG_MODULE_MAGIC;

/* misc variables */
static Oid	__pgstrom_namespace_oid = UINT_MAX;
static bool	__pgstrom_enabled_guc = true;			/* GUC */
int			pgstrom_cpu_fallback_elevel = ERROR;	/* GUC */
bool		pgstrom_regression_test_mode = false;	/* GUC */
bool		pgstrom_explain_developer_mode = false;	/* GUC */
long		PAGE_SIZE;
long		PAGE_MASK;
int			PAGE_SHIFT;
long		PHYS_PAGES;
long		PAGES_PER_BLOCK;
static bool	pgstrom_enable_select_into_direct;		/* GUC */

static planner_hook_type	planner_hook_next = NULL;
static CustomPathMethods	pgstrom_dummy_path_methods;
static CustomScanMethods	pgstrom_dummy_plan_methods;
static ExecutorStart_hook_type	executor_start_hook_next = NULL;

/* pg_strom.githash() */
PG_FUNCTION_INFO_V1(pgstrom_githash);
PUBLIC_FUNCTION(Datum)
pgstrom_githash(PG_FUNCTION_ARGS)
{
	PG_RETURN_TEXT_P(cstring_to_text(pgstrom_githash_cstring));
}

/*
 * pgstrom_enabled()
 */
static void
pgstrom_extension_checker_callback(Datum arg, int cacheid, uint32 hashvalue)
{
	Assert(cacheid == NAMESPACEOID);
	__pgstrom_namespace_oid = UINT_MAX;
}

bool
pgstrom_enabled(void)
{
	if (__pgstrom_namespace_oid == UINT_MAX)
		__pgstrom_namespace_oid = get_namespace_oid("pgstrom", true);
	if (OidIsValid(__pgstrom_namespace_oid))
		return __pgstrom_enabled_guc;
	return false;
}

/*
 * pg_kern_ereport - raise an ereport at host side
 */
void
pg_kern_ereport(kern_context *kcxt)
{
	ereport(ERROR, (errcode(kcxt->errcode),
					errmsg("%s:%u  %s",
						   kcxt->error_filename,
						   kcxt->error_lineno,
						   kcxt->error_message)));
}

/*
 * pg_hash_any - the standard hash function at device code
 */
uint32_t
pg_hash_any(const void *ptr, int sz)
{
	return (uint32_t)hash_any((const unsigned char *)ptr, sz);
}

/*
 * pgstrom_init_gucs
 */
static void
pgstrom_init_gucs(void)
{
	static struct config_enum_entry	__cpu_fallback_options[] = {
		{"notice",	NOTICE,	false},
		{"on",		DEBUG2,	false},
		{"off",		ERROR,	false},
		{"true",	DEBUG2,	true},
		{"false",	ERROR,	true},
		{"yes",		DEBUG2,	true},
		{"no",		ERROR,	true},
		{"1",		DEBUG2,	true},
		{"0",		ERROR,	true},
		{NULL, 0, false}
	};
	/* Disables PG-Strom features at all */
	DefineCustomBoolVariable("pg_strom.enabled",
							 "Enables the planner's use of PG-Strom",
							 NULL,
							 &__pgstrom_enabled_guc,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* turn on/off CPU fallback if GPU could not execute the query */
	DefineCustomEnumVariable("pg_strom.cpu_fallback",
							 "Enables CPU fallback if xPU required re-run",
							 NULL,
							 &pgstrom_cpu_fallback_elevel,
							 ERROR,
							 __cpu_fallback_options,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* disables some platform specific EXPLAIN output */
	DefineCustomBoolVariable("pg_strom.regression_test_mode",
							 "Disables some platform specific output in EXPLAIN; that can lead undesired test failed but harmless",
							 NULL,
							 &pgstrom_regression_test_mode,
							 false,
							 PGC_USERSET,
							 GUC_NO_SHOW_ALL | GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* turns on/off developer-level detailed EXPLAIN output */
	DefineCustomBoolVariable("pg_strom.explain_developer_mode",
							 "Turns on/off some detailed internal information in EXPLAIN",
							 NULL,
							 &pgstrom_explain_developer_mode,
							 false,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* turns on/off SELECT INTO direct mode */
	DefineCustomBoolVariable("pg_strom.enable_select_into_direct",
							 "Enables SELECT INTO direct mode",
							 NULL,
							 &pgstrom_enable_select_into_direct,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
}

/* --------------------------------------------------------------------------------
 *
 * add/remove dummy plan node
 *
 * -------------------------------------------------------------------------------- */
bool
pgstrom_is_dummy_path(const Path *path)
{
	if (IsA(path, CustomPath))
	{
		const CustomPath *cpath = (const CustomPath *)path;

		return (cpath->methods == &pgstrom_dummy_path_methods);
	}
	return false;
}

Path *
pgstrom_create_dummy_path(PlannerInfo *root, Path *subpath)
{
	CustomPath *cpath = makeNode(CustomPath);
	RelOptInfo *upper_rel = subpath->parent;
	PathTarget *upper_target = upper_rel->reltarget;
	PathTarget *sub_target = subpath->pathtarget;
	ListCell   *lc1, *lc2;

	/* sanity checks */
	if (list_length(upper_target->exprs) != list_length(sub_target->exprs))
		elog(ERROR, "CustomScan(dummy): incompatible tlist is supplied\n%s\n%s",
			 nodeToString(upper_target->exprs),
			 nodeToString(sub_target->exprs));
	forboth (lc1, upper_target->exprs,
			 lc2, sub_target->exprs)
	{
		Node   *node1 = lfirst(lc1);
		Node   *node2 = lfirst(lc2);

		if (exprType(node1) != exprType(node2))
			elog(ERROR, "CustomScan(dummy): incompatible tlist entry: [%s] <-> [%s]",
				 nodeToString(node1),
				 nodeToString(node2));
	}
	Assert(subpath->parent == upper_rel);
	cpath->path.pathtype		= T_CustomScan;
	cpath->path.parent			= upper_rel;
	cpath->path.pathtarget		= upper_target;
	cpath->path.param_info		= NULL;
	cpath->path.parallel_aware	= subpath->parallel_aware;
	cpath->path.parallel_safe	= subpath->parallel_safe;
	cpath->path.parallel_workers = subpath->parallel_workers;
	cpath->path.pathkeys		= subpath->pathkeys;
	cpath->path.rows			= subpath->rows;
	cpath->path.startup_cost	= subpath->startup_cost;
	cpath->path.total_cost		= subpath->total_cost;

	cpath->custom_paths			= list_make1(subpath);
	cpath->methods				= &pgstrom_dummy_path_methods;

	return &cpath->path;
}

/*
 * pgstrom_dummy_create_plan - PlanCustomPath callback
 */
static Plan *
pgstrom_dummy_create_plan(PlannerInfo *root,
						  RelOptInfo *rel,
						  CustomPath *best_path,
						  List *tlist,
						  List *clauses,
						  List *custom_plans)
{
	CustomScan *cscan = makeNode(CustomScan);

	Assert(list_length(custom_plans) == 1);
	cscan->scan.plan.parallel_aware = best_path->path.parallel_aware;
	cscan->scan.plan.targetlist = tlist;
	cscan->scan.plan.qual = NIL;
	cscan->scan.plan.lefttree = linitial(custom_plans);
	cscan->scan.scanrelid = 0;
	cscan->custom_scan_tlist = tlist;
	cscan->methods = &pgstrom_dummy_plan_methods;

	return &cscan->scan.plan;
}

/*
 * pgstrom_dummy_create_scan_state - CreateCustomScanState callback
 */
static Node *
pgstrom_dummy_create_scan_state(CustomScan *cscan)
{
	elog(ERROR, "Bug? dummy custom scan should not remain at the executor stage");
}

/*
 * pgstrom_removal_dummy_plans
 *
 * Due to the interface design of the create_upper_paths_hook, some other path
 * nodes can be stacked on the GpuPreAgg node, with the original final target-
 * list. Even if a pair of Agg + GpuPreAgg adopted its modified target-list,
 * the stacked path nodes (like sorting, window functions, ...) still consider
 * it has the original target-list.
 * It makes a problem at setrefs.c when PostgreSQL optimizer tries to replace
 * the expressions by var-node using OUTER_VAR, because Agg + GpuPreAgg pair
 * does not have the original expression, then it leads "variable not found"
 * error.
 */
static void
pgstrom_removal_dummy_plans(PlannedStmt *pstmt, Plan **p_plan)
{
	Plan	   *plan = *p_plan;
	ListCell   *lc;

	if (!plan)
		return;

	switch (nodeTag(plan))
	{
		case T_Append:
			{
				Append	   *splan = (Append *)plan;

				foreach (lc, splan->appendplans)
					pgstrom_removal_dummy_plans(pstmt, (Plan **)&lfirst(lc));
			}
			break;

		case T_MergeAppend:
			{
				MergeAppend *splan = (MergeAppend *)plan;

				foreach (lc, splan->mergeplans)
					pgstrom_removal_dummy_plans(pstmt, (Plan **)&lfirst(lc));
			}
			break;

		case T_BitmapAnd:
			{
				BitmapAnd  *splan = (BitmapAnd *)plan;

				foreach (lc, splan->bitmapplans)
					pgstrom_removal_dummy_plans(pstmt, (Plan **)&lfirst(lc));
			}
			break;

		case T_BitmapOr:
			{
				BitmapOr   *splan = (BitmapOr *)plan;

				foreach (lc, splan->bitmapplans)
					pgstrom_removal_dummy_plans(pstmt, (Plan **)&lfirst(lc));
			}
			break;

		case T_SubqueryScan:
			{
				SubqueryScan   *sscan = (SubqueryScan *)plan;

				pgstrom_removal_dummy_plans(pstmt, &sscan->subplan);
			}
			break;

		case T_CustomScan:
			{
				CustomScan	   *cscan = (CustomScan *)plan;

				if (cscan->methods == &pgstrom_dummy_plan_methods)
				{
					Plan	   *subplan = outerPlan(cscan);
					ListCell   *lc1, *lc2;

					/* sanity checks */
					Assert(innerPlan(cscan) == NULL);
					if (list_length(cscan->scan.plan.targetlist) !=
						list_length(subplan->targetlist))
						elog(ERROR, "Bug? dummy plan's targelist length mismatch");
					forboth (lc1, cscan->scan.plan.targetlist,
                             lc2, subplan->targetlist)
					{
						TargetEntry *tle1 = lfirst(lc1);
						TargetEntry *tle2 = lfirst(lc2);

						if (exprType((Node *)tle1->expr) != exprType((Node *)tle2->expr))
							elog(ERROR, "Bug? dummy TLE type mismatch [%s] [%s]",
								 nodeToString(tle1),
								 nodeToString(tle2));
						/*
						 * assign resource name and 'junk' state (grouping-keys that
						 * don't appear in the final result).
						 * See, apply_tlist_labeling()
						 */
						tle2->resname = tle1->resname;
						tle2->resjunk = tle1->resjunk;
					}
					subplan->initPlan = cscan->scan.plan.initPlan;
					*p_plan = subplan;
					pgstrom_removal_dummy_plans(pstmt, p_plan);
					return;
				}
				foreach (lc, cscan->custom_plans)
					pgstrom_removal_dummy_plans(pstmt, (Plan **)&lfirst(lc));
			}
			break;

		default:
			/* nothing special sub-plans */
			break;
	}
	if (plan->lefttree)
		pgstrom_removal_dummy_plans(pstmt, &plan->lefttree);
	if (plan->righttree)
		pgstrom_removal_dummy_plans(pstmt, &plan->righttree);
}

/*
 * pgstrom_path_tracker
 */
static HTAB	   *pgstrom_paths_htable = NULL;

typedef struct
{
	PlannerInfo *root;
	Relids		parent_relids;
	uint32_t	xpu_devkind;	/* one of DEVKIND__* */
	pgstromOuterPathLeafInfo *op_normal_single;
	pgstromOuterPathLeafInfo *op_normal_parallel;
	List	   *op_leaf_single;
	List	   *op_leaf_parallel;
	Cost		total_cost_single;
	Cost		total_cost_parallel;
	bool		identical_inners_single;
	bool		identical_inners_parallel;
} pgstromPathEntry;

static uint32
pgstrom_path_entry_hash(const void *key, Size keysize)
{
	const pgstromPathEntry *entry = key;

	return (hash_bytes((const unsigned char *)&entry->root,
					   sizeof(PlannerInfo *)) ^
			bms_hash_value(entry->parent_relids) ^
			entry->xpu_devkind);
}

static int
pgstrom_path_entry_match(const void *key1, const void *key2, Size keysize)
{
	const pgstromPathEntry *entry1 = key1;
	const pgstromPathEntry *entry2 = key2;

	Assert(keysize == offsetof(pgstromPathEntry,
							   xpu_devkind) + sizeof(uint32_t));
	return (entry1->root == entry2->root &&
			bms_equal(entry1->parent_relids,
					  entry2->parent_relids) &&
			entry1->xpu_devkind == entry2->xpu_devkind ? 0 : -1);
}

static void
__pgstrom_build_paths_htable(void)
{
	if (!pgstrom_paths_htable)
	{
		HASHCTL		hctl;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.hcxt = CurrentMemoryContext;
		hctl.keysize = offsetof(pgstromPathEntry,
								xpu_devkind) + sizeof(uint32_t);
		hctl.entrysize = sizeof(pgstromPathEntry);
		hctl.hash = pgstrom_path_entry_hash;
		hctl.match = pgstrom_path_entry_match;
		pgstrom_paths_htable = hash_create("PG-Strom Outer/Leaf Paths Tracker",
										   256L,
										   &hctl,
										   HASH_ELEM |
										   HASH_FUNCTION |
										   HASH_COMPARE |
										   HASH_CONTEXT);
	}
}

void
pgstrom_remember_op_normal(PlannerInfo *root,
						   RelOptInfo *outer_rel,
						   pgstromOuterPathLeafInfo *op_leaf,
						   bool be_parallel)
{
	pgstromPathEntry *pp_entry;
	pgstromPathEntry  pp_key;
	bool		found;

	/* sanity checks */
	Assert(list_length(op_leaf->inner_paths_list) == op_leaf->pp_info->num_rels);
	op_leaf->outer_rel = outer_rel;

	/* lookup the hash-table */
	__pgstrom_build_paths_htable();
	memset(&pp_key, 0, sizeof(pgstromPathEntry));
	pp_key.root = root;
	pp_key.parent_relids = outer_rel->relids;
	pp_key.xpu_devkind = (op_leaf->pp_info->xpu_task_flags & DEVKIND__ANY);
	pp_entry = (pgstromPathEntry *)
		hash_search(pgstrom_paths_htable,
					&pp_key,
					HASH_ENTER,
					&found);
	if (!found)
	{
		Assert(pp_key.root == pp_entry->root &&
			   bms_equal(pp_key.parent_relids,
						 pp_entry->parent_relids));
		memcpy(pp_entry, &pp_key, sizeof(pgstromPathEntry));
	}

	if (be_parallel)
	{
		if (!pp_entry->op_normal_parallel ||
			pp_entry->op_normal_parallel->leaf_cost > op_leaf->leaf_cost)
			pp_entry->op_normal_parallel = op_leaf;
	}
	else
	{
		if (!pp_entry->op_normal_single ||
			pp_entry->op_normal_single->leaf_cost > op_leaf->leaf_cost)
			pp_entry->op_normal_single = op_leaf;
	}
}

void
pgstrom_remember_op_leafs(PlannerInfo *root,
						  RelOptInfo *parent_rel,
						  List *op_leaf_list,
						  bool be_parallel)
{
	pgstromPathEntry *pp_entry;
	pgstromPathEntry  pp_key;
	ListCell   *cell;
	List	   *inner_paths_list = NIL;
	uint32_t	xpu_devkind = 0;
	int			identical_inners = -1;
	Cost		total_cost = 0.0;
	bool		found;

	__pgstrom_build_paths_htable();
	/* calculation of total cost */
	foreach (cell, op_leaf_list)
	{
		pgstromOuterPathLeafInfo *op_leaf = lfirst(cell);

		/* sanity checks */
		Assert(list_length(op_leaf->inner_paths_list) == op_leaf->pp_info->num_rels);
		op_leaf->outer_rel = parent_rel;
		total_cost += op_leaf->leaf_cost;

		/* check whether all entries have identical xPU device */
		if (xpu_devkind == 0)
			xpu_devkind = (op_leaf->pp_info->xpu_task_flags & DEVKIND__ANY);
		else if (xpu_devkind != (op_leaf->pp_info->xpu_task_flags & DEVKIND__ANY))
			elog(ERROR, "Bug? different xPU devices are mixtured.");

		if (cell == list_head(op_leaf_list))
		{
			inner_paths_list = op_leaf->inner_paths_list;
		}
		else if (identical_inners != 0)
		{
			ListCell   *lc1, *lc2;

			forboth (lc1, inner_paths_list,
					 lc2, op_leaf->inner_paths_list)
			{
				Path   *__i_path1 = lfirst(lc1);
				Path   *__i_path2 = lfirst(lc2);

				if (!bms_equal(__i_path1->parent->relids,
							   __i_path2->parent->relids))
					break;
			}
			if (lc1 == NULL && lc2 == NULL)
				identical_inners = 1;
			else
				identical_inners = 0;
		}
	}
	/* lookup the hash-table */
	memset(&pp_key, 0, sizeof(pgstromPathEntry));
	pp_key.root = root;
	pp_key.parent_relids = parent_rel->relids;
	pp_key.xpu_devkind = xpu_devkind;
	pp_entry = (pgstromPathEntry *)
		hash_search(pgstrom_paths_htable,
					&pp_key,
					HASH_ENTER,
					&found);
	if (!found)
	{
		Assert(pp_key.root == pp_entry->root &&
			   bms_equal(pp_key.parent_relids,
						 pp_entry->parent_relids));
		memcpy(pp_entry, &pp_key, sizeof(pgstromPathEntry));
	}

	if (be_parallel)
	{
		if (pp_entry->op_leaf_parallel == NIL ||
			pp_entry->total_cost_parallel > total_cost)
		{
			pp_entry->op_leaf_parallel = op_leaf_list;
			pp_entry->total_cost_parallel = total_cost;
			pp_entry->identical_inners_parallel = (identical_inners > 0);
		}
	}
	else
	{
		if (pp_entry->op_leaf_single == NIL ||
			pp_entry->total_cost_single > total_cost)
		{
			pp_entry->op_leaf_single = op_leaf_list;
			pp_entry->total_cost_single = total_cost;
			pp_entry->identical_inners_single = (identical_inners > 0);
		}
	}
}

pgstromOuterPathLeafInfo *
pgstrom_find_op_normal(PlannerInfo *root,
					   RelOptInfo *outer_rel,
					   uint32_t xpu_task_flags,
					   bool be_parallel)
{
	if (pgstrom_paths_htable)
	{
		pgstromPathEntry *pp_entry;
		pgstromPathEntry  pp_key;

		memset(&pp_key, 0, sizeof(pgstromPathEntry));
		pp_key.root = root;
		pp_key.parent_relids = outer_rel->relids;
		pp_key.xpu_devkind = (xpu_task_flags & DEVKIND__ANY);
		pp_entry = (pgstromPathEntry *)
			hash_search(pgstrom_paths_htable,
						&pp_key,
						HASH_FIND,
						NULL);
		if (pp_entry)
			return (be_parallel
					? pp_entry->op_normal_parallel
					: pp_entry->op_normal_single);
	}
	return NULL;
}

List *
pgstrom_find_op_leafs(PlannerInfo *root,
					  RelOptInfo *parent_rel,
					  uint32_t xpu_task_flags,
					  bool be_parallel,
					  bool *p_identical_inners)
{
	if (pgstrom_paths_htable)
	{
		pgstromPathEntry *pp_entry;
		pgstromPathEntry  pp_key;

		memset(&pp_key, 0, sizeof(pgstromPathEntry));
		pp_key.root = root;
		pp_key.parent_relids = parent_rel->relids;
		pp_key.xpu_devkind = (xpu_task_flags & DEVKIND__ANY);
		pp_entry = (pgstromPathEntry *)
			hash_search(pgstrom_paths_htable,
						&pp_key,
						HASH_FIND,
						NULL);
		if (pp_entry)
		{
			if (p_identical_inners)
				*p_identical_inners = (be_parallel
									   ? pp_entry->identical_inners_parallel
									   : pp_entry->identical_inners_single);
			return (be_parallel
					? pp_entry->op_leaf_parallel
					: pp_entry->op_leaf_single);
		}
	}
	return NIL;
}

/*
 * pgstrom_post_planner
 */
static PlannedStmt *
pgstrom_post_planner(Query *parse,
					 const char *query_string,
					 int cursorOptions,
					 ParamListInfo boundParams)
{
	HTAB	   *saved_paths_htable = pgstrom_paths_htable;
	PlannedStmt *pstmt;
	ListCell   *lc;

	PG_TRY();
	{
		pgstrom_paths_htable = NULL;

		pstmt = planner_hook_next(parse,
								  query_string,
								  cursorOptions,
								  boundParams);
		/* remove dummy plan */
		pgstrom_removal_dummy_plans(pstmt, &pstmt->planTree);
		foreach (lc, pstmt->subplans)
			pgstrom_removal_dummy_plans(pstmt, (Plan **)&lfirst(lc));
	}
	PG_CATCH();
	{
		hash_destroy(pgstrom_paths_htable);
		pgstrom_paths_htable = saved_paths_htable;
		PG_RE_THROW();
	}
	PG_END_TRY();
	hash_destroy(pgstrom_paths_htable);
	pgstrom_paths_htable = saved_paths_htable;
	return pstmt;
}

/*
 * pgstrom_executor_start
 */
static void
pgstrom_executor_start(QueryDesc *queryDesc, int eflags)
{
	if (executor_start_hook_next)
		executor_start_hook_next(queryDesc, eflags);
	else
		standard_ExecutorStart(queryDesc, eflags);
	/*
	 * check whether SELECT INTO direct mode is possible, or not.
	 * also, see ExecCreateTableAs() for the related initializations.
	 */
	if (!pgstrom_enable_select_into_direct)
		elog(DEBUG2, "SELECT INTO Direct disabled: because of pg_strom.enable_select_into_direct parameter");
	else if (queryDesc->dest->mydest != DestIntoRel)
		elog(DEBUG2, "SELECT INTO Direct disabled: because dest-receiver is not DestIntoRel");
	else if (!pgstrom_is_gpuscan_state(queryDesc->planstate) &&
			 !pgstrom_is_gpujoin_state(queryDesc->planstate) &&
			 !pgstrom_is_gpupreagg_state(queryDesc->planstate))
		elog(DEBUG2, "SELECT INTO Direct disabled: because top-level plan is not GPU-aware");
	else if (pgstrom_cpu_fallback_elevel < ERROR)
		elog(DEBUG2, "SELECT INTO Direct disabled: because CPU-fallback is enabled");
	else
	{
		pgstromTaskState *pts = (pgstromTaskState *)queryDesc->planstate;

		/*
		 * Even if ProjectionInfo would be assigned on the planstate,
		 * we check whether it is actually incompatible or not.
		 */
		if (pts->css.ss.ps.ps_ProjInfo)
		{
			ProjectionInfo *projInfo = pts->css.ss.ps.ps_ProjInfo;
			CustomScan *cscan = (CustomScan *)pts->css.ss.ps.plan;
			List	   *tlist_cpu = (List *)projInfo->pi_state.expr;
			ListCell   *lc;
			int			nvalids = 0;
			int			anum = 1;
			bool		meet_resjunk = false;

			Assert(IsA(cscan, CustomScan));
			foreach (lc, cscan->custom_scan_tlist)
			{
				TargetEntry *tle = lfirst(lc);

				if (tle->resjunk)
					meet_resjunk = true;
				else if (!meet_resjunk)
					nvalids++;
				else
				{
					elog(DEBUG2, "Bug? CustomScan has valid TLEs after junks: %s",
						 nodeToString(cscan->custom_scan_tlist));
					return;
				}
			}
			if (list_length(tlist_cpu) != nvalids)
			{
				elog(DEBUG2, "SELECT INTO Direct disabled: because of CPU projection (%d -> %d attributes)",
					 nvalids, list_length(tlist_cpu));
				return;
			}
			foreach (lc, tlist_cpu)
			{
				TargetEntry *tle = lfirst(lc);
				Var	   *var = (Var *)tle->expr;

				if (tle->resjunk ||
					!IsA(var, Var) ||
					var->varno != INDEX_VAR ||
					var->varattno != anum)
				{
					elog(DEBUG2, "SELECT INTO Direct disabled: because of CPU projection: %s",
						 nodeToString(tle->expr));
					return;
				}
				anum++;
			}
		}
		/*
		 * This GPU-task can potentially run SELECT INTO direct mode.
		 * After the DestReceiver::rStartup invocation at ExecutorRun(),
		 * we must have the final check of ...
		 * - whether the relation is unlogged or temporary
		 * - whether the EXCLUSIVE lock is held
		 * - whether the TAM (Table Access Method) is heap
		 * at the first invocation of execution.
		 */
		pts->select_into_dest = queryDesc->dest;
	}
}

/*
 * pgstrom_sigpoll_handler
 */
static void
pgstrom_sigpoll_handler(SIGNAL_ARGS)
{
	/* do nothing here, but invocation of this handler may wake up epoll(2) / poll(2) */
}

/*
 * _PG_init
 *
 * Main entrypoint of PG-Strom. It shall be invoked only once when postmaster
 * process is starting up, then it calls other sub-systems to initialize for
 * each ones.
 */
void
_PG_init(void)
{
	/*
	 * PG-Strom must be loaded using shared_preload_libraries
	 */
	if (!process_shared_preload_libraries_in_progress)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("PG-Strom must be loaded via shared_preload_libraries")));
	/* init misc variables */
	PAGE_SIZE = sysconf(_SC_PAGESIZE);
	PAGE_MASK = PAGE_SIZE - 1;
	PAGE_SHIFT = get_next_log2(PAGE_SIZE);
	PHYS_PAGES = sysconf(_SC_PHYS_PAGES);
	PAGES_PER_BLOCK = BLCKSZ / PAGE_SIZE;

	/* init pg-strom infrastructure */
	pgstrom_init_gucs();
	pgstrom_init_extra();
	pgstrom_init_codegen();
	pgstrom_init_relscan();
	pgstrom_init_brin();
	pgstrom_init_arrow_fdw();
	pgstrom_init_executor();
	/* dump version number */
	elog(LOG, "PG-Strom version %s built for PostgreSQL %s (githash: %s)",
		 PGSTROM_VERSION,
		 PG_MAJORVERSION,
		 pgstrom_githash_cstring);
	/* init GPU related stuff */
	if (pgstrom_init_gpu_device())
	{
		pgstrom_init_gpu_service();
		pgstrom_init_gpu_scan();
		pgstrom_init_gpu_join();
		pgstrom_init_gpu_preagg();
		pgstrom_init_gpu_cache();
	}
	/* init DPU related stuff */
	if (pgstrom_init_dpu_device())
	{
		pgstrom_init_dpu_scan();
		pgstrom_init_dpu_join();
		pgstrom_init_dpu_preagg();
	}
	/* callback for the extension checker */
	CacheRegisterSyscacheCallback(NAMESPACEOID, pgstrom_extension_checker_callback, 0);
	/* dummy custom-scan node */
	memset(&pgstrom_dummy_path_methods, 0, sizeof(CustomPathMethods));
	pgstrom_dummy_path_methods.CustomName   = "Dummy";
	pgstrom_dummy_path_methods.PlanCustomPath = pgstrom_dummy_create_plan;

	memset(&pgstrom_dummy_plan_methods, 0, sizeof(CustomScanMethods));
	pgstrom_dummy_plan_methods.CustomName   = "Dummy";
	pgstrom_dummy_plan_methods.CreateCustomScanState = pgstrom_dummy_create_scan_state;

	/* post planner hook */
	planner_hook_next = (planner_hook ? planner_hook : standard_planner);
	planner_hook = pgstrom_post_planner;
	/* executor hook */
	executor_start_hook_next = ExecutorStart_hook;
	ExecutorStart_hook = pgstrom_executor_start;

	/* signal handler for wake up */
	pqsignal(SIGPOLL, pgstrom_sigpoll_handler);
}
