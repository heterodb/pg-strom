/*
 * main.c
 *
 * Entrypoint of PG-Strom extension
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
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
	bool			outer_parallel;
	bool			inner_parallel;
	const char	   *custom_name;
	const CustomPath *cpath;
} custom_path_entry;

static HTAB	   *custom_path_htable = NULL;

static uint32
custom_path_entry_hashvalue(const void *key, Size keysize)
{
	custom_path_entry *cent = (custom_path_entry *)key;
	const char *custom_name = cent->custom_name;
	uint32      hash;

	hash = hash_any((unsigned char *)&cent->root, sizeof(PlannerInfo *));
	if (cent->relids != NULL)
	{
		Bitmapset  *relids = cent->relids;

		hash ^= hash_any((unsigned char *)relids,
						 offsetof(Bitmapset, words[relids->nwords]));
	}
	if (cent->outer_parallel)
		hash ^= 0x9e3779b9U;
	if (cent->inner_parallel)
		hash ^= 0x49a0f4ddU;
	hash ^= hash_any((unsigned char *)custom_name, strlen(custom_name));

	return hash;
}

static int
custom_path_entry_compare(const void *key1, const void *key2, Size keysize)
{
	custom_path_entry *cent1 = (custom_path_entry *)key1;
	custom_path_entry *cent2 = (custom_path_entry *)key2;

	if (cent1->root == cent2->root &&
		bms_equal(cent1->relids, cent2->relids) &&
		cent1->outer_parallel == cent2->outer_parallel &&
		cent1->inner_parallel == cent2->inner_parallel &&
		strcmp(cent1->custom_name, cent2->custom_name) == 0)
		return 0;
	/* not equal */
	return 1;
}

static void *
custom_path_entry_keycopy(void *dest, const void *src, Size keysize)
{
	custom_path_entry *dent = (custom_path_entry *)dest;
	const custom_path_entry *sent = (const custom_path_entry *)src;

	dent->root = sent->root;
	dent->relids = bms_copy(sent->relids);
	dent->outer_parallel = sent->outer_parallel;
	dent->inner_parallel = sent->inner_parallel;
	dent->custom_name = pstrdup(sent->custom_name);

	return dest;
}

const CustomPath *
custom_path_find_cheapest(PlannerInfo *root,
						  RelOptInfo *rel,
						  bool outer_parallel,
						  bool inner_parallel,
						  const char *custom_name)
{
	custom_path_entry  hkey;
	custom_path_entry *cent;

	memset(&hkey, 0, sizeof(custom_path_entry));
	hkey.root = root;
	hkey.relids = rel->relids;
	hkey.outer_parallel = outer_parallel;
	hkey.inner_parallel = inner_parallel;
	hkey.custom_name = custom_name;

	cent = hash_search(custom_path_htable, &hkey, HASH_FIND, NULL);
	if (!cent)
		return NULL;
	return cent->cpath;
}

bool
custom_path_remember(PlannerInfo *root,
					 RelOptInfo *rel,
					 bool outer_parallel,
					 bool inner_parallel,
					 const CustomPath *cpath)
{
	custom_path_entry  hkey;
	custom_path_entry *cent;
	bool		found;

	memset(&hkey, 0, sizeof(custom_path_entry));
	hkey.root = root;
	hkey.relids = rel->relids;
	hkey.outer_parallel = outer_parallel;
	hkey.inner_parallel = inner_parallel;
	hkey.custom_name = cpath->methods->CustomName;

	cent = hash_search(custom_path_htable, &hkey, HASH_ENTER, &found);
	if (found)
	{
		/* new path is more expensive than prior one! */
		if (cent->cpath->path.total_cost <= cpath->path.total_cost)
			return false;
	}
	Assert(cent->root == root &&
		   bms_equal(cent->relids, rel->relids) &&
		   cent->outer_parallel == outer_parallel &&
		   cent->inner_parallel == inner_parallel &&
		   strcmp(cent->custom_name, cpath->methods->CustomName) == 0);
	cent->cpath = (const CustomPath *)pgstrom_copy_pathnode(&cpath->path);

	return true;
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

	PG_TRY();
	{
		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.hcxt = CurrentMemoryContext;
		hctl.keysize = offsetof(custom_path_entry, cpath);
		hctl.entrysize = sizeof(custom_path_entry);
		hctl.hash = custom_path_entry_hashvalue;
		hctl.match = custom_path_entry_compare;
		hctl.keycopy = custom_path_entry_keycopy;
		custom_path_htable = hash_create("HTable to preserve Custom-Paths",
										 512,
										 &hctl,
										 HASH_CONTEXT |
										 HASH_ELEM |
										 HASH_FUNCTION |
										 HASH_COMPARE |
										 HASH_KEYCOPY);
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
		pgstrom_init_gpu_direct();
		pgstrom_init_gpu_scan();
		//pgstrom_init_gpu_join();
		//pgstrom_init_gpu_preagg();
	}
	/* init DPU related stuff */
	if (pgstrom_init_dpu_device())
	{
		pgstrom_init_dpu_scan();
		//pgstrom_init_dpu_join();
		//pgstrom_init_dpu_preagg();
	}
	/* post planner hook */
	planner_hook_next = (planner_hook ? planner_hook : standard_planner);
	planner_hook = pgstrom_post_planner;
	/* signal handler for wake up */
	pqsignal(SIGPOLL, pgstrom_sigpoll_handler);
}
