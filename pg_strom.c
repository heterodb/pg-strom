/*
 * pg_strom.c
 *
 * Entrypoint of the pg_strom module
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
#include "foreign/fdwapi.h"
#include "miscadmin.h"
#include "utils/guc.h"
#include "pg_strom.h"

PG_MODULE_MAGIC;

/*
 * Local declarations
 */
void	_PG_init(void);

FdwRoutine	pgstromFdwHandlerData = {
	.type				= T_FdwRoutine,
	.PlanForeignScan	= pgstrom_plan_foreign_scan,
	.ExplainForeignScan	= pgstrom_explain_foreign_scan,
	.BeginForeignScan	= pgstrom_begin_foreign_scan,
	.IterateForeignScan	= pgstrom_iterate_foreign_scan,
	.ReScanForeignScan	= pgboost_rescan_foreign_scan,
	.EndForeignScan		= pgboost_end_foreign_scan,
};

/*
 * pgstrom_fdw_handler
 *
 * FDW Handler function of pg_strom
 */
Datum
pgstrom_fdw_handler(PG_FUNCTION_ARGS)
{
	PG_RETURN_POINTER(&pgstromFdwHandlerData);
}

/****/
Datum
pgstrom_fdw_validator(PG_FUNCTION_ARGS)
{
	PG_RETURN_VOID();
}

static void
check_guc_work_group_size(int newval, void *extra)
{
	if ((PGSTROM_CHUNK_SIZE / BITS_PER_BYTE) % newval != 0)
		ereport(ERROR,
				(errcode(ERRCODE_CANT_CHANGE_RUNTIME_PARAM),
				 errmsg("chunk size (%d/8) must be multiple number of %s",
						PGSTROM_CHUNK_SIZE,
						"pg_strom.work_group_size")));
}

static void
pgstrom_guc_init(void)
{
	DefineCustomIntVariable("pg_strom.max_async_chunks",
							"max number of concurrency to exec async kernels",
							NULL,
							&pgstrom_max_async_chunks,
							32,
							1,
							1024,
							PGC_USERSET,
							0,
							NULL, NULL, NULL);

	DefineCustomIntVariable("pg_strom.work_group_size",
							"size of work group on execution of kernel code",
							NULL,
							&pgstrom_work_group_size,
							32,
							1,
							PGSTROM_CHUNK_SIZE / BITS_PER_BYTE,
							PGC_USERSET,
							0,
							NULL, check_guc_work_group_size, NULL);
}

/*
 * Entrypoint of the pg_strom module
 */
void
_PG_init(void)
{
	/*
	 * pg_strom has to be loaded using shared_preload_libraries setting.
	 */
	if (!process_shared_preload_libraries_in_progress)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
		errmsg("pg_strom must be loaded via shared_preload_libraries")));

	/* Register GUC variables */
	pgstrom_guc_init();

	/* Register Hooks of PostgreSQL */
	pgstrom_utilcmds_init();

	/* Collect properties of GPU devices */
	pgstrom_devinfo_init();
}
