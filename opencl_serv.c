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
#include "utils/memutils.h"
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
 * on_shmem_zone_callback
 *
 * It is a callback function for each zone on shared memory segment
 * initialization. It assigns a buffer object of OpenCL for each zone
 * for asynchronous memory transfer later.
 */
static void *
on_shmem_zone_callback(void *address, Size length, void *cb_private)
{
	List	   *dev_list = cb_private;
	ListCell   *cell;
	int			n_devices = list_length(dev_list);
	int			index = 0;
	cl_mem	   *host_mems;
	cl_int		rc;

	host_mems = MemoryContextAllocZero(TopMemoryContext,
								 sizeof(cl_mem) * n_devices);
	index = 0;
	foreach(cell, dev_list)
	{
		pgstrom_device_info	*devinfo = lfirst(cell);

		host_mems[index] = clCreateBuffer(devinfo->context,
										  CL_MEM_READ_WRITE |
										  CL_MEM_USE_HOST_PTR,
										  length,
										  address,
										  &rc);
		if (rc != CL_SUCCESS)
			elog(ERROR, "clCreateBuffer failed on host memory (%p-%p): %s",
				 address, (char *)address + length - 1, opencl_strerror(rc));
		index++;
	}
	elog(LOG, "PG-Strom: zone %p-%p was mapped (len: %luMB)",
		 address, (char *)address + length - 1, length >> 20);
	return host_mems;
}

static void
init_opencl_devices_and_shmem(void)
{
	Size		zone_length = LONG_MAX;
	List	   *devList;
	ListCell   *cell;
	int			index = 0;

	devList = pgstrom_collect_opencl_device_info();
	foreach (cell, devList)
	{
		pgstrom_device_info	*devinfo = lfirst(cell);

		elog(LOG, "PG-Strom: device[%d] %s (%uMHz x %uunits, %luMB)",
			 index, devinfo->dev_name,
			 devinfo->dev_max_clock_frequency,
			 devinfo->dev_max_compute_units,
			 devinfo->dev_global_mem_size >> 20);

		if (zone_length > devinfo->dev_max_mem_alloc_size)
			zone_length = devinfo->dev_max_mem_alloc_size;
		index++;
	}
	elog(LOG, "PG-Strom: setting up shared memory (zone length=%zu)",
		 zone_length);
	pgstrom_setup_shmem(zone_length, on_shmem_zone_callback, devList);

	pgstrom_register_device_info(devList);
}

static void
pgstrom_opencl_main(Datum main_arg)
{
	/* Establish signal handlers before unblocking signals. */
    pqsignal(SIGHUP, pgstrom_opencl_sighup);
    pqsignal(SIGTERM, pgstrom_opencl_sigterm);

    /* We're now ready to receive signals */
    BackgroundWorkerUnblockSignals();

	/* initialize opencl devices and shared memory segment */
	init_opencl_devices_and_shmem();

	elog(LOG, "Starting PG-Strom OpenCL Server");

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
