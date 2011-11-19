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


/*
 * pgstrom_fdw_handler
 *
 * FDW Handler function of pg_strom
 */
Datum
pgstrom_fdw_handler(PG_FUNCTION_ARGS)
{
	FdwRoutine *fdwroutine = makeNode(FdwRoutine);

	fdwroutine->PlanForeignScan = pgstrom_plan_foreign_scan;
	fdwroutine->ExplainForeignScan = pgstrom_explain_foreign_scan;
	fdwroutine->BeginForeignScan = pgstrom_begin_foreign_scan;
	fdwroutine->IterateForeignScan = pgstrom_iterate_foreign_scan;
	fdwroutine->ReScanForeignScan = pgboost_rescan_foreign_scan;
	fdwroutine->EndForeignScan = pgboost_end_foreign_scan;

	PG_RETURN_POINTER(fdwroutine);
}

/*
 * pgstrom_log_device_info
 *
 * Logs name and properties of installed GPU devices.
 */
static void
pgstrom_log_device_info(void)
{
	struct cudaDeviceProp prop;
	cudaError_t	rc;
	int			i, ngpus;

	rc = pgcuda_get_device_count(&ngpus);
	if (rc != cudaSuccess)
		elog(ERROR, "Failed to get number of GPUs : %s",
			 pgcuda_get_error_string(rc));

	for (i = 0; i < ngpus; i++)
	{
		rc = pgcuda_get_device_properties(&prop, i);
		if (rc != cudaSuccess)
			elog(ERROR, "Failed to get properties of GPU(%d) : %s",
				 i, pgcuda_get_error_string(rc));

		elog(LOG,
			 "GPU(%d) : %s (capability v%d.%d), "
			 "%d of MP (WarpSize: %d, %dMHz), "
			 "Device Memory %luMB (%dMHz, %dbits)",
			 i, prop.name, prop.major, prop.minor,
			 prop.multiProcessorCount, prop.warpSize, prop.clockRate / 1000,
			 prop.totalGlobalMem / (1024 *1024),
			 prop.memoryClockRate / 1000, prop.memoryBusWidth);
	}
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

	/* Logs information of GPU deviced installed */
	pgstrom_log_device_info();
}
