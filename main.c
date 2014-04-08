/*
 * main.c
 *
 * The entrypoint of PG-Strom extension.
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "utils/guc.h"
#include <limits.h>
#include "pg_strom.h"

PG_MODULE_MAGIC;

/*
 * miscellaneous GUC parameters
 */
int		pgstrom_max_async_chunks;
int		pgstrom_min_async_chunks;

static void
pgstrom_init_misc_guc(void)
{
	/* GUC variables according to the device information */

	DefineCustomIntVariable("pg_strom.min_async_chunks",
							"least number of chunks to be run asynchronously",
							NULL,
							&pgstrom_min_async_chunks,
							2,
							2,
							INT_MAX,
							PGC_USERSET,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.max_async_chunks",
							"max number of chunk to be run asynchronously",
							NULL,
							&pgstrom_max_async_chunks,
							32,
							pgstrom_min_async_chunks + 1,
							INT_MAX,
							PGC_USERSET,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	if (pgstrom_max_async_chunks <= pgstrom_min_async_chunks)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("\"pg_strom.max_async_chunks\" must be larger than \"pg_strom.min_async_chunks\"")));
}

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

	/* load OpenCL runtime and initialize entrypoints */
	pgstrom_init_opencl_entry();

	/* initialization of device info on postmaster stage */
	pgstrom_init_opencl_devinfo();
	pgstrom_init_opencl_devprog();

	/* initialization of message queue on postmaster stage */
	pgstrom_init_mqueue();

	/* initialization of resource tracking subsystem */
	pgstrom_init_restrack();

	/* initialize shared memory segment and memory context stuff */
	pgstrom_init_shmem();

	/* registration of OpenCL background worker process */
	pgstrom_init_opencl_server();

	/* registration of custom-plan providers */
	pgstrom_init_gpuscan();

	/* miscellaneous GUC init above */
	pgstrom_init_misc_guc();
}

/*
 * pgstrom_strerror
 *
 * translation from StromError_* to human readable form
 */
const char *
pgstrom_strerror(cl_int errcode)
{
	static char		unknown_buf[256];

	switch (errcode)
	{
		case StromError_Success:
			return "success";
		case StromError_RowFiltered:
			return "row is filtered";
		case StromError_RowReCheck:
			return "row should be rechecked";
		case StromError_ServerNotReady:
			return "OpenCL server is not ready";
		case StromError_BadRequestMessage:
			return "request message is bad";
		case StromError_ProgramCompile:
			return "program compile error";
		case StromError_OutOfMemory:
			return "out of memory";
		case StromError_OutOfSharedMemory:
			return "out of shared memory";
		case StromError_DivisionByZero:
			return "division by zero";
		default:
			snprintf(unknown_buf, sizeof(unknown_buf),
					 "undefined strom error (code: %d)", errcode);
			break;
	}
	return unknown_buf;
}

/* ------------------------------------------------------------
 *
 * Routines for debugging
 *
 * ------------------------------------------------------------
 */
Datum
pgstrom_shmem_alloc_func(PG_FUNCTION_ARGS)
{
#ifdef PGSTROM_DEBUG
	Size	size = PG_GETARG_INT64(0);
	void   *address;

	address = pgstrom_shmem_alloc(size);

	PG_RETURN_INT64((Size) address);
#else
	elog(ERROR, "%s is not implemented for production release", __FUNCTION__);

	PG_RETURN_NULL();
#endif
}
PG_FUNCTION_INFO_V1(pgstrom_shmem_alloc_func);

Datum
pgstrom_shmem_free_func(PG_FUNCTION_ARGS)
{
#ifdef PGSTROM_DEBUG
	void		   *address = (void *) PG_GETARG_INT64(0);

	pgstrom_shmem_free(address);

	PG_RETURN_BOOL(true);
#else
	elog(ERROR, "%s is not implemented for production release", __FUNCTION__);

	PG_RETURN_NULL();
#endif
}
PG_FUNCTION_INFO_V1(pgstrom_shmem_free_func);
