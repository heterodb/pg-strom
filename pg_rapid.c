/*
 * pg_rapid.c
 *
 * Entrypoint of the pg_rapid module
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
#include "miscadmin.h"
#include "pg_rapid.h"

PG_MODULE_MAGIC;

/*
 * Local declarations
 */
void	_PG_init(void);
bool	pgrapid_fdw_handler_is_called = false;



/*
 *
 *
 *
 */
Datum
pgrapid_fdw_handler(PG_FUNCTION_ARGS)
{
	FdwRoutine *fdwroutine = makeNode(FdwRoutine);

	pgrapid_fdw_handler_is_called = true;

	fdwroutine->PlanForeignScan = NULL;
	fdwroutine->ExplainForeignScan = NULL;
	fdwroutine->BeginForeignScan = NULL;
	fdwroutine->IterateForeignScan = NULL;
	fdwroutine->ReScanForeignScan = NULL;
	fdwroutine->EndForeignScan = NULL;

	PG_RETURN_POINTER(fdwroutine);
}

/*
 * pgrapid_log_device_info
 *
 * Logs name and properties of installed GPU devices.
 */
static void
pgrapid_log_device_info(void)
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
 * Entrypoint of the pg_rapid module
 */
void
_PG_init(void)
{
	/*
	 * pg_rapid has to be loaded using shared_preload_libraries setting.
	 */
	if (!process_shared_preload_libraries_in_progress)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
		errmsg("sepgsql must be loaded via shared_preload_libraries")));

	/* Register Hooks of PostgreSQL */
	pgrapid_utilcmds_init();

	/* Logs information of GPU deviced installed */
	pgrapid_log_device_info();
}
