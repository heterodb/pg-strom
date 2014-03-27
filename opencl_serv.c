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
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "pg_strom.h"
#include <limits.h>
#include <signal.h>
#include <unistd.h>

/* flags set by signal handlers */
static volatile sig_atomic_t got_signal = false;
static int		opencl_platform_index;

/* OpenCL resources for quick reference */
#define MAX_NUM_DEVICES		128

static cl_platform_id		opencl_platform_id;
static cl_context			opencl_context;
static cl_uint				opencl_num_devices;
static cl_device_id			opencl_devices[MAX_NUM_DEVICES];
static cl_command_queue		opencl_cmdq[MAX_NUM_DEVICES];

static void
pgstrom_opencl_sigterm(SIGNAL_ARGS)
{
	InterruptPending = true;
	got_signal = true;
}

static void
pgstrom_opencl_sighup(SIGNAL_ARGS)
{
	InterruptPending = true;
	got_signal = true;
}




/*
 * pgstrom_collect_device_info
 *
 * It collects properties of all the OpenCL devices. It shall be called once
 * by the OpenCL management worker process, prior to any other backends.
 */
static List *
construct_opencl_device_info(int platform_index)
{
	cl_platform_id	platforms[32];
	cl_device_id	devices[MAX_NUM_DEVICES];
	cl_uint			n_platform;
	cl_uint			n_devices;
	cl_int			i, j, rc;
	long			score_max = -1;
	List		   *result = NIL;

	rc = clGetPlatformIDs(lengthof(platforms),
						  platforms,
						  &n_platform);
	if (rc != CL_SUCCESS)
		elog(ERROR, "clGetPlatformIDs failed (%s)", opencl_strerror(rc));

	for (i=0; i < n_platform; i++)
	{
		pgstrom_platform_info  *pl_info;
		pgstrom_device_info	   *dev_info;
		long		score = 0;
		List	   *temp = NIL;

		pl_info = collect_opencl_platform_info(platforms[i]);
		pl_info->pl_index = i;

		rc = clGetDeviceIDs(platforms[i],
							CL_DEVICE_TYPE_CPU |
							CL_DEVICE_TYPE_GPU |
							CL_DEVICE_TYPE_ACCELERATOR,
							lengthof(devices),
							devices,
							&n_devices);
		if (rc != CL_SUCCESS)
			elog(ERROR, "clGetDeviceIDs failed (%s)", opencl_strerror(rc));

		elog(LOG, "PG-Strom: [%d] OpenCL Platform: %s", i, pl_info->pl_name);

		for (j=0; j < n_devices; j++)
		{
			dev_info = collect_opencl_device_info(devices[j]);
			dev_info->pl_info = pl_info;
			dev_info->dev_index = j;

			elog(LOG, "PG-Strom:  + device %s (%uMHz x %uunits, %luMB)",
				 dev_info->dev_name,
				 dev_info->dev_max_clock_frequency,
				 dev_info->dev_max_compute_units,
				 dev_info->dev_global_mem_size >> 20);

			/* rough estimation about computing power */
			if ((dev_info->dev_type & CL_DEVICE_TYPE_GPU) != 0)
				score += 32 * (dev_info->dev_max_compute_units *
							   dev_info->dev_max_clock_frequency);
			else
				score += (dev_info->dev_max_compute_units *
						  dev_info->dev_max_clock_frequency);

			temp = lappend(temp, dev_info);
		}

		if (platform_index == i || (platform_index < 0 && score > score_max))
		{
			opencl_platform_id = platforms[i];
			opencl_num_devices = n_devices;
			for (j=0; j < n_devices; j++)
				opencl_devices[j] = devices[j];

			score_max = score;
			result = temp;
		}
	}

	/* show platform name if auto-selection */
	if (platform_index < 0 && result != NIL)
	{
		pgstrom_platform_info *pl_info
			= ((pgstrom_device_info *) linitial(result))->pl_info;
		elog(LOG, "PG-Strom: auto platform selection: %s", pl_info->pl_name);
	}

	if (result != NIL)
	{
		/*
		 * Create an OpenCL context
		 */
		opencl_context = clCreateContext(NULL,
										 opencl_num_devices,
										 opencl_devices,
										 NULL,
										 NULL,
										 &rc);
		if (rc != CL_SUCCESS)
			elog(ERROR, "clCreateContext failed: %s", opencl_strerror(rc));

		/*
		 * Create an OpenCL command queue for each device
		 */
		for (j=0; j < opencl_num_devices; j++)
		{
			opencl_cmdq[j] =
				clCreateCommandQueue(opencl_context,
									 opencl_devices[j],
									 CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
									 CL_QUEUE_PROFILING_ENABLE,
									 &rc);
			if (rc != CL_SUCCESS)
				elog(ERROR, "clCreateCommandQueue failed: %s",
					 opencl_strerror(rc));
		}
	}
	return result;
}

/*
 * on_shmem_zone_callback
 *
 * It is a callback function for each zone on shared memory segment
 * initialization. It assigns a buffer object of OpenCL for each zone
 * for asynchronous memory transfer later.
 */
static void *
on_shmem_zone_callback(void *address, Size length)
{
	cl_mem		host_mem;
	cl_int		rc;

	host_mem = clCreateBuffer(opencl_context,
							  CL_MEM_READ_WRITE |
							  CL_MEM_USE_HOST_PTR,
							  length,
							  address,
							  &rc);
	if (rc != CL_SUCCESS)
		elog(ERROR, "clCreateBuffer failed on host memory (%p-%p): %s",
			 address, (char *)address + length - 1, opencl_strerror(rc));
	elog(LOG, "PG-Strom: zone %p-%p was mapped (len: %luMB)",
		 address, (char *)address + length - 1, length >> 20);
	return host_mem;
}

static void
init_opencl_devices_and_shmem(void)
{
	Size		zone_length = LONG_MAX;
	List	   *devList;
	ListCell   *cell;

	devList = construct_opencl_device_info(opencl_platform_index);
	if (devList == NIL)
		elog(ERROR, "PG-Strom: unavailable to use any OpenCL devices");

	foreach (cell, devList)
	{
		pgstrom_device_info	*dev_info = lfirst(cell);

		if (zone_length > dev_info->dev_max_mem_alloc_size)
			zone_length = dev_info->dev_max_mem_alloc_size;
	}
	elog(LOG, "PG-Strom: setting up shared memory (zone length=%zu)",
		 zone_length);
	pgstrom_setup_shmem(zone_length, on_shmem_zone_callback);
	pgstrom_setup_opencl_devinfo(devList);
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

	/* final setup of message queue */
	pgstrom_setup_mqueue();
	pgstrom_setup_opencl_devprog();

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

	/* selection of opencl platform */
	DefineCustomIntVariable("pgstrom.opencl_platform",
							"selection of OpenCL platform to be used",
							NULL,
							&opencl_platform_index,
							-1,		/* auto selection */
							-1,
							INT_MAX,
							PGC_POSTMASTER,
                            GUC_NOT_IN_SAMPLE,
                            NULL, NULL, NULL);

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
