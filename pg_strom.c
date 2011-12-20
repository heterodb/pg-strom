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
#if 0
/*
 * pgstrom_log_device_info
 *
 * Logs name and properties of installed GPU devices.
 */
static void
pgstrom_log_device_info(void)
{
	cl_platform_id	platform_ids[64];
	cl_device_id	device_ids[128];
	cl_uint			num_platforms;
	cl_uint			num_devices;
	cl_int			ret, pi, di;

	ret = clGetPlatformIDs(lengthof(platform_ids),
						   platform_ids, &num_platforms);
	if (ret != CL_SUCCESS)
		elog(ERROR, "openCL: failed to get number of platforms");

	for (pi=0; pi < num_platforms; pi++)
	{
		char	pf_name[255];
		char	pf_version[64];
		char	pf_vendor[64];
		size_t	retsize;

		ret = clGetPlatformInfo(platform_ids[pi], CL_PLATFORM_NAME,
								sizeof(pf_name), &pf_name, &retsize);
		if (ret != CL_SUCCESS)
			elog(ERROR, "openCL: failed to get platform name");

		ret = clGetPlatformInfo(platform_ids[pi], CL_PLATFORM_VERSION,
								sizeof(pf_version), &pf_version, &retsize);
		if (ret != CL_SUCCESS)
			elog(ERROR, "openCL: failed to get platform version");

		ret = clGetPlatformInfo(platform_ids[pi], CL_PLATFORM_VENDOR,
								sizeof(pf_vendor), &pf_vendor, &retsize);
		if (ret != CL_SUCCESS)
			elog(ERROR, "openCL: failed to get platform vendor");

		ret = clGetDeviceIDs(platform_ids[pi],
							 CL_DEVICE_TYPE_DEFAULT,
							 lengthof(device_ids),
							 device_ids, &num_devices);
		if (ret != CL_SUCCESS)
			elog(ERROR, "openCL: failed to get number of devices");

		for (di=0; di < num_devices; di++)
		{
			char	dev_name[128];
			char	dev_version[64];

			ret = clGetDeviceInfo(device_ids[di], CL_DEVICE_NAME,
								  sizeof(dev_name), &dev_name, &retsize);
			if (ret != CL_SUCCESS)
				elog(ERROR, "openCL: failed to get device name");

			ret = clGetDeviceInfo(device_ids[di], CL_DEVICE_VERSION,
								  sizeof(dev_version), &dev_version, &retsize);
			if (ret != CL_SUCCESS)
				elog(ERROR, "openCL: failed to get device version");

			elog(LOG, "pg_strom: %s, %s by %s (%s/%s)",
				 pf_name, pf_version, pf_vendor,
				 dev_name, dev_version);
		}
	}

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
#endif

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

	/* Collect properties of GPU devices */
	pgstrom_device_info_init();
}
