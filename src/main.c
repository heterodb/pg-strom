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
bool		pgstrom_enabled;				/* GUC */
bool		pgstrom_cpu_fallback_enabled;	/* GUC */
bool		pgstrom_regression_test_mode;	/* GUC */
int			pgstrom_max_async_tasks;		/* GUC */
long		PAGE_SIZE;
long		PAGE_MASK;
int			PAGE_SHIFT;
long		PHYS_PAGES;
long		PAGES_PER_BLOCK;

static planner_hook_type	planner_hook_next = NULL;
static CustomPathMethods	pgstrom_dummy_path_methods;
static CustomScanMethods	pgstrom_dummy_plan_methods;

/* pg_strom.githash() */
PG_FUNCTION_INFO_V1(pgstrom_githash);
Datum
pgstrom_githash(PG_FUNCTION_ARGS)
{
#ifdef PGSTROM_GITHASH
	PG_RETURN_TEXT_P(cstring_to_text(PGSTROM_GITHASH));
#else
	PG_RETURN_NULL();
#endif
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
	/* Disables PG-Strom features at all */
	DefineCustomBoolVariable("pg_strom.enabled",
							 "Enables the planner's use of PG-Strom",
							 NULL,
							 &pgstrom_enabled,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* turn on/off CPU fallback if GPU could not execute the query */
	DefineCustomBoolVariable("pg_strom.cpu_fallback",
							 "Enables CPU fallback if GPU required re-run",
							 NULL,
							 &pgstrom_cpu_fallback_enabled,
							 false,
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
	DefineCustomIntVariable("pg_strom.max_async_tasks",
							"Limit of conccurent execution at the xPU devices",
							NULL,
							&pgstrom_max_async_tasks,
							7,
							1,
							255,
							PGC_SUSET,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
}

/*
 * xPU-aware path tracker
 *
 * motivation: add_path() and add_partial_path() keeps only cheapest paths.
 * Once some other dominates GpuXXX paths, it shall be wiped out, even if
 * it potentially has a chance for more optimization (e.g, GpuJoin outer
 * pull-up, GpuPreAgg + GpuJoin combined mode).
 * So, we preserve PG-Strom related Path-nodes for the later referenced.
 */
typedef struct
{
	PlannerInfo	   *root;
	Relids			relids;
	bool			parallel_path;
	uint32_t		devkind;		/* one of DEVKIND_* */
	CustomPath	   *cpath;
} custom_path_entry;

static HTAB	   *custom_path_htable = NULL;

static uint32
custom_path_entry_hashvalue(const void *key, Size keysize)
{
	custom_path_entry *cent = (custom_path_entry *)key;
	uint32      hash;

	hash = hash_bytes((unsigned char *)&cent->root, sizeof(PlannerInfo *));
	hash ^= bms_hash_value(cent->relids);
	if (cent->parallel_path)
		hash ^= 0x9e3779b9U;
	hash ^= hash_uint32(cent->devkind);

	return hash;
}

static int
custom_path_entry_compare(const void *key1, const void *key2, Size keysize)
{
	custom_path_entry *cent1 = (custom_path_entry *)key1;
	custom_path_entry *cent2 = (custom_path_entry *)key2;

	if (cent1->root == cent2->root &&
		bms_equal(cent1->relids, cent2->relids) &&
		cent1->parallel_path == cent2->parallel_path &&
		cent1->devkind == cent2->devkind)
		return 0;
	/* not equal */
	return 1;
}

CustomPath *
custom_path_find_cheapest(PlannerInfo *root,
						  RelOptInfo *rel,
						  bool parallel_path,
						  uint32_t devkind)
{
	custom_path_entry  hkey;
	custom_path_entry *cent;

	memset(&hkey, 0, sizeof(custom_path_entry));
	hkey.root = root;
	hkey.relids = rel->relids;
	hkey.parallel_path = (parallel_path ? true : false);
	hkey.devkind = (devkind & DEVKIND__ANY);

	cent = hash_search(custom_path_htable, &hkey, HASH_FIND, NULL);
	if (!cent)
		return NULL;
	return cent->cpath;
}

bool
custom_path_remember(PlannerInfo *root,
					 RelOptInfo *rel,
					 bool parallel_path,
					 uint32_t devkind,
					 const CustomPath *cpath)
{
	custom_path_entry  hkey;
	custom_path_entry *cent;
	bool		found;

	Assert((devkind & DEVKIND__ANY) == DEVKIND__NVIDIA_GPU ||
		   (devkind & DEVKIND__ANY) == DEVKIND__NVIDIA_DPU);
	memset(&hkey, 0, sizeof(custom_path_entry));
	hkey.root = root;
	hkey.relids = rel->relids;
	hkey.parallel_path = (parallel_path ? true : false);
	hkey.devkind = (devkind & DEVKIND__ANY);

	cent = hash_search(custom_path_htable, &hkey, HASH_ENTER, &found);
	if (found)
	{
		/* new path is more expensive than prior one! */
		if (cent->cpath->path.total_cost <= cpath->path.total_cost)
			return false;
	}
	cent->cpath = (CustomPath *)pgstrom_copy_pathnode(&cpath->path);

	return true;
}

/* --------------------------------------------------------------------------------
 *
 * add/remove dummy plan node
 *
 * -------------------------------------------------------------------------------- */
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
		elog(ERROR, "CustomScan(dummy): incompatible tlist is supplied");
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

	Assert(plan != NULL);
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
						/* assign resource name */
						tle2->resname = tle1->resname;
					}
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
 * pgstrom_post_planner
 */
static PlannedStmt *
pgstrom_post_planner(Query *parse,
					 const char *query_string,
					 int cursorOptions,
					 ParamListInfo boundParams)
{
	HTAB	   *custom_path_htable_saved = custom_path_htable;
	HASHCTL		hctl;
	PlannedStmt *pstmt;
	ListCell   *lc;

	PG_TRY();
	{
		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.hcxt = CurrentMemoryContext;
		hctl.keysize = offsetof(custom_path_entry, cpath);
		hctl.entrysize = sizeof(custom_path_entry);
		hctl.hash = custom_path_entry_hashvalue;
		hctl.match = custom_path_entry_compare;
		custom_path_htable = hash_create("HTable to preserve Custom-Paths",
										 512,
										 &hctl,
										 HASH_CONTEXT |
										 HASH_ELEM |
										 HASH_FUNCTION |
										 HASH_COMPARE);
		pstmt = planner_hook_next(parse,
								  query_string,
								  cursorOptions,
								  boundParams);
	}
	PG_CATCH();
	{
		hash_destroy(custom_path_htable);
		custom_path_htable = custom_path_htable_saved;
		PG_RE_THROW();
	}
	PG_END_TRY();
	hash_destroy(custom_path_htable);
	custom_path_htable = custom_path_htable_saved;

	/* remove dummy plan */
	pgstrom_removal_dummy_plans(pstmt, &pstmt->planTree);
	foreach (lc, pstmt->subplans)
		pgstrom_removal_dummy_plans(pstmt, (Plan **)&lfirst(lc));

	return pstmt;
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
	elog(LOG, "PG-Strom version %s built for PostgreSQL %s (git: %s)",
		 PGSTROM_VERSION,
		 PG_MAJORVERSION,
		 PGSTROM_GITHASH);
	/* init GPU related stuff */
	if (pgstrom_init_gpu_device())
	{
		pgstrom_init_gpu_service();
		pgstrom_init_gpu_scan();
		pgstrom_init_gpu_join();
		pgstrom_init_gpu_preagg();
	}
	/* init DPU related stuff */
	if (pgstrom_init_dpu_device())
	{
		pgstrom_init_dpu_scan();
		pgstrom_init_dpu_join();
		pgstrom_init_dpu_preagg();
	}
	pgstrom_init_pcie();
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
	/* signal handler for wake up */
	pqsignal(SIGPOLL, pgstrom_sigpoll_handler);
}
