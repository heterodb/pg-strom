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
	.ReScanForeignScan	= pgstrom_rescan_foreign_scan,
	.EndForeignScan		= pgstrom_end_foreign_scan,
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

	/* Register Hooks of PostgreSQL */
	pgstrom_utilcmds_init();

	/* Initialize stuff related to GPU devices */
	pgstrom_devinfo_init();

	/* Initialize stuff related to scan.c */
	pgstrom_scan_init();
}
