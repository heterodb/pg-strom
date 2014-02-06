/*
 * opencl_serv.c
 *
 * Backend server process to manage OpenCL devices
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "nodes/pg_list.h"
#include "postmaster/bgworker.h"
#include "utils/builtins.h"
#include "pg_strom.h"
#include <limits.h>
#include <signal.h>
#include <unistd.h>

/* flags set by signal handlers */
static volatile sig_atomic_t got_signal = false;

static void
pgstrom_opencl_sigterm(SIGNAL_ARGS)
{
	got_signal = true;
}

static void
pgstrom_opencl_sighup(SIGNAL_ARGS)
{
	got_signal = true;
}


/*
 * get_shmem_zone_length
 *
 * It calculate length of the shared memory zone; that should be the smallest
 * max memory allocation size in all of the OpenCL devices.
 */
static Size
get_shmem_zone_length(void)
{
	int		i, n = pgstrom_get_opencl_device_num();
	Size	length = LONG_MAX;

	for (i=0; i < n; i++)
	{
		pgstrom_device_info	*devinfo
			= pgstrom_get_opencl_device_info(i);

		if (length < devinfo->dev_max_mem_alloc_size)
			length = devinfo->dev_max_mem_alloc_size;
	}
	return length;
}

static void *
on_shmem_zone_callback(void *address, Size length)
{
	elog(LOG, "zone: address = %p length = %zu", address, length);
	return NULL;
}




static void
pgstrom_opencl_main(Datum main_arg)
{
	/* Establish signal handlers before unblocking signals. */
    pqsignal(SIGHUP, pgstrom_opencl_sighup);
    pqsignal(SIGTERM, pgstrom_opencl_sigterm);

    /* We're now ready to receive signals */
    BackgroundWorkerUnblockSignals();

	/* Gather any properties of OpenCL devices */
	pgstrom_init_opencl_device_info();

	elog(LOG, "Starting PG-Strom OpenCL Server");

	pgstrom_setup_shmem(get_shmem_zone_length(),
						on_shmem_zone_callback);

	while (!got_signal)
	{
		sleep(2);
	}
}

void
pgstrom_init_opencl_server(void)
{
	BackgroundWorker	worker;

	/* launch a background worker process */	
	memset(&worker, 0, sizeof(BackgroundWorker));
	strcpy(worker.bgw_name, "PG-Strom OpenCL Server");
	worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
	worker.bgw_start_time = BgWorkerStart_PostmasterStart;
	worker.bgw_restart_time = BGW_NEVER_RESTART;

	worker.bgw_main = pgstrom_opencl_main;
	worker.bgw_main_arg = 0;
	RegisterBackgroundWorker(&worker);
}
