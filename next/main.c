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
double		pgstrom_gpu_setup_cost = 100 * DEFAULT_SEQ_PAGE_COST;	/* GUC */
double		pgstrom_gpu_dma_cost = DEFAULT_CPU_TUPLE_COST / 10.0;	/* GUC */
double		pgstrom_gpu_operator_cost = DEFAULT_CPU_OPERATOR_COST / 16.0;	/* GUC */
long		PAGE_SIZE;
long		PAGE_MASK;
int			PAGE_SHIFT;
long		PHYS_PAGES;

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
 * pgstrom_init_misc_options
 */
static void
pgstrom_init_misc_options(void)
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
}

/*
 * pgstrom_init_gpu_options - init GUC params related to GPUs
 */
static void
pgstrom_init_gpu_options(void)
{
	/* cost factor for Gpu setup */
	DefineCustomRealVariable("pg_strom.gpu_setup_cost",
							 "Cost to setup GPU device to run",
							 NULL,
							 &pgstrom_gpu_setup_cost,
							 100 * DEFAULT_SEQ_PAGE_COST,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* cost factor for each Gpu task */
	DefineCustomRealVariable("pg_strom.gpu_dma_cost",
							 "Cost to send/recv tuple via DMA",
							 NULL,
							 &pgstrom_gpu_dma_cost,
							 DEFAULT_CPU_TUPLE_COST / 10.0,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* cost factor for Gpu operator */
	DefineCustomRealVariable("pg_strom.gpu_operator_cost",
							 "Cost of processing each operators by GPU",
							 NULL,
							 &pgstrom_gpu_operator_cost,
							 DEFAULT_CPU_OPERATOR_COST / 16.0,
							 0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
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

	/* init pg-strom infrastructure */
	pgstrom_init_misc_options();
	pgstrom_init_extra();
	pgstrom_init_shmbuf();
	pgstrom_init_codegen();
	pgstrom_init_relscan();
	/* dump version number */
	elog(LOG, "PG-Strom version %s built for PostgreSQL %s (git: %s)",
		 PGSTROM_VERSION,
		 PG_MAJORVERSION,
		 PGSTROM_GITHASH);
	/* init GPU related stuff */
	if (pgstrom_init_gpu_device())
	{
		pgstrom_init_gpu_options();
		pgstrom_init_gpu_service();
		pgstrom_init_gpu_scan();
		//pgstrom_init_gpu_join();
		//pgstrom_init_gpu_preagg();
	}
}
