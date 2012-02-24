/*
 * opencl_serv.c
 *
 * Routines of computing engine component based on OpenCL. 
 *
 * --
 * Copyright 2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
//#include "storage/ipc.h"
//#include "storage/shmem.h"
#include "utils/guc.h"
#include "pg_strom.h"
#include <CL/cl.h>
#include <pthread.h>
#include <unistd.h>

typedef struct {
	SHM_QUEUE  *send_cmdq;	/* dual-linked list of GPU/CPU/Recv queue */
	SHM_QUEUE  *recv_cmdq;	/* pointer to the response queue */
	uint32	   *gpu_cmds;	/* command array handled by GPU */
	uint32	   *cpu_cmds;	/* command array handled by CPU */
	int			nattrs;		/* number of columns */
	int			nitems;		/* number of rows */
	bits8	   *cs_rowmap;	/* rowmap of CS, also base address of buffer */
	int		   *cs_isnull;	/* offset from the cs_rowmap, or 0 */
	int		   *cs_values;	/* offset from the cs_rowmap, or 0 */
} ChunkBuffer;




typedef struct {
	cl_platform_id		platform_id;
	cl_device_id		device_id;
	cl_device_exec_capabilities		dev_exec_capabilities;

	char				device_name[128];
} DeviceInfo;

/*
 * Local declarations
 */
static const char  *pgstrom_opencl_error_string(int errcode);
static int			pgstrom_num_devices;
static DeviceInfo  *pgstrom_device_info;
static cl_context	pgstrom_cl_context;
static cl_mem		pgstrom_cl_shmbuf;
static SHM_QUEUE   *pgstrom_gpu_cmdq;
static pthread_t	pgstrom_opencl_thread;


static void *
pgstrom_opencl_server(void *arg)
{
	SHM_QUEUE  *item;

	while (true)
	{
		item = pgstrom_shmqueue_dequeue(pgstrom_gpu_cmdq, true);
		Assert(item != NULL);


		// TODO: input request an appropriate device
		// TODO: need GPU scheduler?
		// TODO: an complete event shall be backed,
		// TODO: or chained to CPU computing server
	}
	elog(FATAL, "PG-Strom: GPU computing server should never exit");
	return NULL;
}

static void
pgstrom_opencl_device_init(void)
{
	cl_platform_id	platforms[8];
	cl_device_id	devices[64];
	cl_uint			num_platforms;
	cl_uint			num_devices;
	cl_int			i, ret;

	/*
	 * TODO: GUC to choose a particular platform to be used,
	 *       if we have multiple platforms at a single box.
	 *       Right now, we use the first one only.
	 */
	ret = clGetPlatformIDs(lengthof(platforms), platforms, &num_platforms);
	if (ret != CL_SUCCESS)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("OpenCL: failed to get platform handlers (%s)",
						pgstrom_opencl_error_string(ret))));

	ret = clGetDeviceIDs(platforms[0],
						 CL_DEVICE_TYPE_DEFAULT,
						 lengthof(devices), devices, &num_devices);
	if (ret != CL_SUCCESS)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("OpenCL: failed to get device handlers (%s)",
						pgstrom_opencl_error_string(ret))));

	pgstrom_device_info = malloc(sizeof(DeviceInfo) * num_devices);
	if (!pgstrom_device_info)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));

	pgstrom_num_devices = 0;
	for (i=0; i < num_devices; i++)
	{
		DeviceInfo  *devinfo
			= &pgstrom_device_info[pgstrom_num_devices++];

		devinfo->platform_id = platforms[0];
		devinfo->device_id = devices[i];

		if (clGetDeviceInfo(devinfo->device_id,
							CL_DEVICE_NAME,
							sizeof(devinfo->device_name),
							&devinfo->device_name[0],
							NULL) != CL_SUCCESS ||
			clGetDeviceInfo(devinfo->device_id,
							CL_DEVICE_EXECUTION_CAPABILITIES,
							sizeof(devinfo->dev_exec_capabilities),
							&devinfo->dev_exec_capabilities,
							NULL) != CL_SUCCESS)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("OpenCL: failed to get device info")));

		elog(LOG, "PG-Strom: %s", devinfo->device_name);
	}

	/*
	 * Create OpenCL context with above devices
	 */
	pgstrom_cl_context = clCreateContext(NULL,
										 num_devices,
										 devices,
										 NULL, NULL, &ret);
	if (ret != CL_SUCCESS)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("OpenCL: failed to create device context (%s)",
						pgstrom_opencl_error_string(ret))));
}









void
pgstrom_opencl_startup(void *shmptr, Size shmsize)
{
	cl_int	ret;

	/* Collect platform/device information */
	pgstrom_opencl_device_init();

	/* Set up shared memory segment as page-locked memory */
	pgstrom_cl_shmbuf = clCreateBuffer(pgstrom_cl_context,
									   CL_MEM_USE_HOST_PTR,
									   shmsize, shmptr,
									   &ret);
	if (ret != CL_SUCCESS)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("OpenCL: failed to create host buffer (%s)",
						pgstrom_opencl_error_string(ret))));

	/* Create a command queue of GPU device */
	pgstrom_gpu_cmdq = pgstrom_shmqueue_create();
	if (!pgstrom_gpu_cmdq)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("PG-Strom: failed to create command queue")));

	/* Launch computing server thread of GPU */
	if (pthread_create(&pgstrom_opencl_thread, NULL,
					   pgstrom_opencl_server, NULL) != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("PG-Strom: failed to launch GPU computing server")));
}

static const char *
pgstrom_opencl_error_string(int errcode)
{
	static char strbuf[128];

	switch (errcode)
	{
		case CL_SUCCESS:
			return "success";
		default:
			snprintf(strbuf, sizeof(strbuf), "error code: %d", errcode);
			break;
	}
	return strbuf;
}
