/*
 * main.c
 *
 * Entrypoint of PG-Strom extension
 * ----
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
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
 * fixup_upper_expression_by_customscan
 *
 * It transforms an expression that references the outer results using OUTER_VAR
 * into a new expression that can be pushed down to the outer custom-scan.
 */
typedef struct
{
	CustomScan *cscan;
	int			status;
} __fixup_upper_expression_by_customscan_context;

static Node *
__fixup_upper_expression_by_customscan_walker(Node *node, void *__data)
{
	__fixup_upper_expression_by_customscan_context *con = __data;

	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		CustomScan *cscan = con->cscan;
		const Var  *var = (const Var *)node;

		if (var->varno == OUTER_VAR &&
			var->varattno > 0 &&
			var->varattno <= list_length(cscan->scan.plan.targetlist))
		{
			TargetEntry *tle = list_nth(cscan->scan.plan.targetlist,
										var->varattno-1);
			Assert(var->vartype   == exprType((Node *)tle->expr) &&
				   var->vartypmod == exprTypmod((Node *)tle->expr));
			return copyObject((Node *)tle->expr);
		}
		con->status = -1;
	}
	return expression_tree_mutator(node, __fixup_upper_expression_by_customscan_walker, __data);
}

static Expr *
fixup_upper_expression_by_customscan(Expr *expr_orig, CustomScan *cscan)
{
	__fixup_upper_expression_by_customscan_context con;
	Node	   *expr_new;

	con.cscan = cscan;
	con.status = 0;
	expr_new = __fixup_upper_expression_by_customscan_walker((Node *)expr_orig, &con);
	if (con.status == 0)
		return (Expr *)expr_new;
	return NULL;
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

		case T_Result:
			/*
			 * MEMO: When the GPU-PreAgg node is placed directly under the Result
			 * node, it may be possible to integrate CPU-Projection processing.
			 * This is because when converting from the GPU-PreAgg Path to a Plan,
			 * it is necessary to refer to the targetlist to create the tlist_dev,
			 * so the pushdown of the ProjectionPath by CUSTOMPATH_SUPPORT_PROJECTION
			 * cannot be used.
			 */
			if (outerPlan(plan) != NULL &&
				pgstrom_is_gpupreagg_plan(outerPlan(plan)) &&
				((Result *)plan)->resconstantqual == NULL)
			{
				CustomScan *cscan = (CustomScan *)outerPlan(plan);
				ListCell   *cell;
				List	   *tlist_new = NIL;

				foreach (cell, plan->targetlist)
				{
					TargetEntry *tle = lfirst(cell);
					Expr   *expr_new =
						fixup_upper_expression_by_customscan(tle->expr, cscan);
					if (!expr_new)
						break;
					tlist_new = lappend(tlist_new,
										makeTargetEntry(expr_new,
														tle->resno,
														tle->resname,
														tle->resjunk));
				}
				if (!cell)
				{
					cscan->scan.plan.targetlist = tlist_new;
					*p_plan = &cscan->scan.plan;
					pgstrom_removal_dummy_plans(pstmt, p_plan);
					return;
				}
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
 * pgstrom_plan_info_tracker
 */
static HTAB	   *pgstrom_ppinfo_htable = NULL;

typedef struct
{
	PlannerInfo *root;
	Relids		relids;
} pgstromPlanInfoKey;

typedef struct
{
	pgstromPlanInfoKey key;
	CustomPath		cpath_single;
	CustomPath		cpath_parallel;
} pgstromPlanInfoEntry;

static uint32
pgstrom_plan_info_key_hash(const void *__key, Size keysize)
{
	const pgstromPlanInfoKey *key = __key;

	Assert(keysize == sizeof(pgstromPlanInfoKey));
	return (hash_bytes((const unsigned char *)&key->root,
					   sizeof(PlannerInfo *)) ^
			bms_hash_value(key->relids));
}

static int
pgstrom_plan_info_key_match(const void *__key1,
							const void *__key2, Size keysize)
{
	const pgstromPlanInfoKey *key1 = __key1;
	const pgstromPlanInfoKey *key2 = __key2;

	Assert(keysize == sizeof(pgstromPlanInfoKey));
	return (key1->root == key2->root &&
			bms_equal(key1->relids, key2->relids) ? 0 : -1);
}

void
pgstrom_remember_custom_path(PlannerInfo *root,
							 RelOptInfo *outer_rel,
							 CustomPath *cpath,
							 bool be_parallel)
{
	pgstromPlanInfoKey key;
	pgstromPlanInfoEntry *entry;
	bool		found;

	Assert(IsA(cpath, CustomPath));
	if (!pgstrom_ppinfo_htable)
	{
		HASHCTL		hctl;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.hcxt = CurrentMemoryContext;
		hctl.keysize = sizeof(pgstromPlanInfoKey);
		hctl.entrysize = sizeof(pgstromPlanInfoEntry);
		hctl.hash = pgstrom_plan_info_key_hash;
		hctl.match = pgstrom_plan_info_key_match;
		pgstrom_ppinfo_htable  = hash_create("PG-Strom PlanInfo Tracker",
											 256L,
											 &hctl,
											 HASH_ELEM |
											 HASH_FUNCTION |
											 HASH_COMPARE |
											 HASH_CONTEXT);
	}
	key.root   = root;
	key.relids = outer_rel->relids;
	entry = (pgstromPlanInfoEntry *)
		hash_search(pgstrom_ppinfo_htable,
					&key,
					HASH_ENTER,
					&found);
	Assert(entry->key.root == root &&
		   bms_equal(entry->key.relids, outer_rel->relids));
	if (!found)
	{
		if (!be_parallel)
		{
			memcpy(&entry->cpath_single, cpath, sizeof(CustomPath));
			memset(&entry->cpath_parallel, 0, sizeof(CustomPath));
		}
		else
		{
			memset(&entry->cpath_single, 0, sizeof(CustomPath));
			memcpy(&entry->cpath_parallel, cpath, sizeof(CustomPath));
		}
	}
	else if (!be_parallel)
	{
		if (!IsA(&entry->cpath_single, CustomPath))
			memcpy(&entry->cpath_single, cpath, sizeof(CustomPath));
		else
		{
			pgstromPlanInfo *pp_orig = linitial(entry->cpath_single.custom_private);
			pgstromPlanInfo *pp_info = linitial(cpath->custom_private);

			if ((pp_orig->startup_cost +
				 pp_orig->inner_cost +
				 pp_orig->run_cost) > (pp_info->startup_cost +
									   pp_info->inner_cost +
									   pp_info->run_cost))
				memcpy(&entry->cpath_single, cpath, sizeof(CustomPath));
		}
	}
	else
	{
		if (!IsA(&entry->cpath_parallel, CustomPath))
			memcpy(&entry->cpath_parallel, cpath, sizeof(CustomPath));
		else
		{
			pgstromPlanInfo *pp_orig = linitial(entry->cpath_parallel.custom_private);
			pgstromPlanInfo *pp_info = linitial(cpath->custom_private);

			if ((pp_orig->startup_cost +
				 pp_orig->inner_cost +
				 pp_orig->run_cost) > (pp_info->startup_cost +
									   pp_info->inner_cost +
									   pp_info->run_cost))
				memcpy(&entry->cpath_parallel, cpath, sizeof(CustomPath));
		}
	}
}

CustomPath *
pgstrom_find_custom_path(PlannerInfo *root,
						 RelOptInfo *outer_rel,
						 bool be_parallel)
{
	if (pgstrom_ppinfo_htable)
	{
		pgstromPlanInfoKey key;
		pgstromPlanInfoEntry *entry;

		key.root = root;
		key.relids = outer_rel->relids;
		entry = (pgstromPlanInfoEntry *)
			hash_search(pgstrom_ppinfo_htable,
						&key,
						HASH_FIND,
						NULL);
		if (entry)
		{
			CustomPath *cpath = (!be_parallel
								 ? &entry->cpath_single
								 : &entry->cpath_parallel);
			return IsA(cpath, CustomPath) ? cpath : NULL;
		}
	}
	return NULL;
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
	HTAB	   *saved_ppinfo_htable = pgstrom_ppinfo_htable;
	PlannedStmt *pstmt;
	ListCell   *lc;

	PG_TRY();
	{
		pgstrom_ppinfo_htable = NULL;

		pstmt = planner_hook_next(parse,
								  query_string,
								  cursorOptions,
								  boundParams);
		/* remove dummy plan & push-down Result + GpuPreAgg */
		pgstrom_removal_dummy_plans(pstmt, &pstmt->planTree);
		foreach (lc, pstmt->subplans)
			pgstrom_removal_dummy_plans(pstmt, (Plan **)&lfirst(lc));
	}
	PG_CATCH();
	{
		hash_destroy(pgstrom_ppinfo_htable);
		pgstrom_ppinfo_htable = saved_ppinfo_htable;
		PG_RE_THROW();
	}
	PG_END_TRY();
	hash_destroy(pgstrom_ppinfo_htable);
	pgstrom_ppinfo_htable = saved_ppinfo_htable;
	return pstmt;
}

/*
 * see DR_intorel definition in commands/createas.c
 */
typedef struct
{
	DestReceiver	pub;		/* publicly-known function pointers */
	IntoClause	   *into;		/* target relation specification */
	/* These fields are filled by intorel_startup: */
	Relation		rel;		/* relation to write to */
	ObjectAddress	reladdr;	/* address of rel, for ExecCreateTableAs */
	CommandId		output_cid;	/* cmin to insert in output tuples */
	int				ti_options;	/* table_tuple_insert performance options */
	BulkInsertState	bistate;	/* bulk insert state */
	/* --- below is PG-Strom original enhancement --- */
	pgstromTaskState *pts;
	List		   *select_into_proj;
	void		  (*rStartup_orig) (DestReceiver *self,
									int operation,
									TupleDesc typeinfo);
} DR_intorel_with_pgstrom;

/*
 * select_into_direct_on_startup
 */
static void
select_into_direct_on_startup(DestReceiver *self,
							  int operation,
							  TupleDesc typeinfo)
{
	DR_intorel_with_pgstrom *dest = (DR_intorel_with_pgstrom *)self;
	pgstromTaskState   *pts = dest->pts;

	/* call the original dest-receiver startup */
	dest->rStartup_orig(self, operation, typeinfo);

	/* setup SELECT INTO direct related stuff */
	if (dest->rel)
	{
		CustomScan		*cscan = (CustomScan *)pts->css.ss.ps.plan;
		pgstromPlanInfo *pp_info = pts->pp_info;

		pp_info->xpu_task_flags |= DEVTASK__SELECT_INTO_DIRECT;
		pp_info->select_into_relid = RelationGetRelid(dest->rel);
		pp_info->select_into_proj = dest->select_into_proj;
		form_pgstrom_plan_info(cscan, pp_info);
	}
}

/*
 * pgstrom_executor_start
 */
static void
pgstrom_executor_start(QueryDesc *queryDesc, int eflags)
{
	DR_intorel_with_pgstrom *dest = (DR_intorel_with_pgstrom *)queryDesc->dest;

	if (executor_start_hook_next)
		executor_start_hook_next(queryDesc, eflags);
	else
		standard_ExecutorStart(queryDesc, eflags);
	/*
	 * check whether SELECT INTO direct mode is possible, or not.
	 * also, see ExecCreateTableAs() for the related initializations.
	 */
	if (IsParallelWorker())
		return;		/* only main process can check SELECT INTO Direct capability */
	else if (!pgstrom_enable_select_into_direct)
		elog(DEBUG2, "SELECT INTO Direct disabled: because of pg_strom.enable_select_into_direct parameter");
	else if (pgstrom_cpu_fallback_elevel < ERROR)
		elog(DEBUG2, "SELECT INTO Direct disabled: because CPU-fallback is enabled");
	else if (dest->pub.mydest != DestIntoRel)
		elog(DEBUG2, "SELECT INTO Direct disabled: because dest-receiver is not DestIntoRel");
	else if (queryDesc->plannedstmt->hasReturning)
		elog(DEBUG2, "SELECT INTO Direct disabled: because of RETURNING clause");
	else if (dest->into->rel->relpersistence != RELPERSISTENCE_UNLOGGED &&
			 dest->into->rel->relpersistence != RELPERSISTENCE_TEMP)
		elog(DEBUG2, "SELECT INTO Direct disabled, because the table is neither temporary nor unlogged");
	else
	{
		/* only heap access method supports SELECT INTO Direct */
		const char *accessMethod = (dest->into->accessMethod != NULL
									? dest->into->accessMethod
									: default_table_access_method);
		if (get_table_am_oid(accessMethod, true) != HEAP_TABLE_AM_OID)
			elog(DEBUG2, "SELECT INTO Direct disabled, because the table does not use heap AM");
		else
		{
			/* ok, it looks SELECT INTO allows direct mode */
			PlanState  *pstate = queryDesc->planstate;

			if (IsA(pstate, GatherState))
			{
				if (pstate->ps_ProjInfo)
				{
					elog(DEBUG2, "SELECT INTO Direct disabled: Gather has projection");
					return;
				}
				pstate = outerPlanState(pstate);
			}
			/* check top-level PlanState except for Gather node */
			if (pgstrom_is_gpuscan_state(pstate) ||
				pgstrom_is_gpujoin_state(pstate) ||
				pgstrom_is_gpupreagg_state(pstate))
			{
				pgstromTaskState *pts = (pgstromTaskState *)pstate;
				List	   *select_into_proj = NIL;

				if ((pts->xpu_task_flags & DEVKIND__ANY) != DEVKIND__NVIDIA_GPU)
					elog(DEBUG2, "SELECT INTO Direct disabled: because only GPU supports it");
				else if (pts->css.ss.ps.qual)
					elog(DEBUG2, "SELECT INTO Direct disabled: because of host qualifiers");
				else if (tryAddSelectIntoDirectProjection(pts, &select_into_proj))
				{
					/* OK, Let's run SELECT INTO Direct in GPU-Direct mode */
					DR_intorel_with_pgstrom *copy = palloc(sizeof(*copy));

					memcpy(copy, dest, offsetof(DR_intorel_with_pgstrom, pts));
					copy->pts = pts;
					copy->select_into_proj = select_into_proj;
					copy->rStartup_orig = dest->pub.rStartup;
					copy->pub.rStartup = select_into_direct_on_startup;
					queryDesc->dest = &copy->pub;

					pts->xpu_task_flags |= DEVTASK__SELECT_INTO_DIRECT;
				}
			}
			else
			{
				elog(DEBUG2, "SELECT INTO Direct disabled: because top-level plan is not GPU-aware");
			}
		}
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
	/* init pg-strom infrastructure */
	pgstrom_init_gucs();
	pgstrom_init_extra();
	pgstrom_init_codegen();
	pgstrom_init_relscan();
	pgstrom_init_brin();
	pgstrom_init_arrow_fdw();
	pgstrom_init_parquet_cache();
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
		pgstrom_init_select_into();
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
