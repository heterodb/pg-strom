/*
 * main.c
 *
 * Entrypoint of the PG-Strom module
 * --
 * Copyright 2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "pg_strom.h"

PG_MODULE_MAGIC;

/*
 * Local declarations
 */
void		_PG_init(void);

FdwRoutine	PgStromFdwHandlerData = {
	.type				= T_FdwRoutine,
	.GetForeignRelSize	= pgstrom_get_foreign_rel_size,
	.GetForeignPaths	= pgstrom_get_foreign_paths,
	.GetForeignPlan		= pgstrom_get_foreign_plan,
	.ExplainForeignScan	= pgstrom_explain_foreign_scan,
	.BeginForeignScan	= pgstrom_begin_foreign_scan,
	.IterateForeignScan	= pgstrom_iterate_foreign_scan,
	.ReScanForeignScan	= pgstrom_rescan_foreign_scan,
	.EndForeignScan		= pgstrom_end_foreign_scan,
};

/*
 * pgstrom_fdw_handler - FDW Handler function of PG-Strom
 */
Datum
pgstrom_fdw_handler(PG_FUNCTION_ARGS)
{
	PG_RETURN_POINTER(&PgStromFdwHandlerData);
}
PG_FUNCTION_INFO_V1(pgstrom_fdw_handler);

void
_PG_init(void)
{
	/*
	 * PG-Strom has to be loaded using shared_preload_libraries option
	 */
	if (!process_shared_preload_libraries_in_progress)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
		errmsg("PG-Strom must be loaded via shared_preload_libraries")));

	/* initialize shared memory segment */
	pgstrom_shmseg_init();

	/* initialize CUDA related stuff */
	pgstrom_gpu_init();

	/* initialize executor related stuff */
	pgstrom_executor_init();

	/* register utility commands hooks */
	pgstrom_utilcmds_init();
}
