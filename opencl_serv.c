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
#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "pg_strom.h"
#include <CL/cl.h>
#include <pthread.h>
#include <unistd.h>





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
static cl_program	pgstrom_cl_program;
static ShmsegQueue *pgstrom_gpu_cmdq;
static pthread_t	pgstrom_opencl_thread;


static void *
pgstrom_opencl_server(void *arg)
{
	SHM_QUEUE  *item;

	while (true)
	{
		item = pgstrom_shmqueue_dequeue(pgstrom_gpu_cmdq);
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
	char			namebuf[MAXPGPATH];
	char		   *kernel_path;
	bytea		   *kernel_bytea;
	const char	   *kernel_source;
	size_t			kernel_length;

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

	/*
	 * Load the source of OpenCL code
	 */
	get_share_path(my_exec_path, namebuf);
	kernel_path = alloca(strlen(namebuf) + 40);
	sprintf(kernel_path, "%s/extension/opencl_kernel", namebuf);

	kernel_bytea = read_binary_file(kernel_path, 0, -1);
	kernel_source = VARDATA(kernel_bytea);
	kernel_length = VARSIZE_ANY_EXHDR(kernel_bytea);

	pgstrom_cl_program = clCreateProgramWithSource(pgstrom_cl_context,
												   1,
												   &kernel_source,
												   &kernel_length,
												   &ret);
	if (ret != CL_SUCCESS)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("OpenCL: failed to create program (%s)",
						pgstrom_opencl_error_string(ret))));

	ret = clBuildProgram(pgstrom_cl_program,
						 num_devices,
						 devices,
						 "-cl-mad-enable",	// TODO: add options...
						 NULL,
						 NULL);
	if (ret != CL_SUCCESS)
	{
		if (ret == CL_BUILD_PROGRAM_FAILURE)
		{
			char   *build_log = alloca(8192);

			if (clGetProgramBuildInfo(pgstrom_cl_program,
									  devices[0],
									  CL_PROGRAM_BUILD_LOG,
									  8192,
									  build_log,
									  NULL) == CL_SUCCESS)
				elog(LOG, "%s", build_log);
		}
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("OpenCL: failed to build program (%s)",
						pgstrom_opencl_error_string(ret))));
	}
	else
	{
		size_t *prog_binsz = alloca(sizeof(size_t) * num_devices);

		if (clGetProgramInfo(pgstrom_cl_program,
							 CL_PROGRAM_BINARY_SIZES,
							 sizeof(size_t) * num_devices,
							 prog_binsz,
							 NULL) == CL_SUCCESS)
			elog(LOG, "binary size = %lu kB", prog_binsz[0] / 1024);
	}
}

int
pgstrom_opencl_num_devices(void)
{
	return pgstrom_num_devices;
}

bool
pgstrom_opencl_fp64_supported(void)
{
	return true;	/* tentasive */
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
	pgstrom_gpu_cmdq = pgstrom_shmseg_alloc(sizeof(ShmsegQueue));
	if (!pgstrom_gpu_cmdq)
		elog(ERROR, "PG-Strom: out of shared memory");
	if (!pgstrom_shmqueue_init(pgstrom_gpu_cmdq))
		elog(ERROR, "PG-Strom: failed to init shmqueue");

	/* Launch computing server thread of GPU */
	if (pthread_create(&pgstrom_opencl_thread, NULL,
					   pgstrom_opencl_server, NULL) != 0)
		elog(ERROR, "PG-Strom: failed to launch GPU computing server");
}

static const char *
pgstrom_opencl_error_string(int errcode)
{
	static char strbuf[128];

	switch (errcode)
	{
		case CL_SUCCESS:
			return "success";
		case CL_DEVICE_NOT_FOUND:
			return "device not found";
		case CL_DEVICE_NOT_AVAILABLE:
			return "device not available";
		case CL_COMPILER_NOT_AVAILABLE:
			return "compiler not available";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			return "memory object allocation failure";
		case CL_OUT_OF_RESOURCES:
			return "out of resources";
		case CL_OUT_OF_HOST_MEMORY:
			return "out of host memory";
		case CL_PROFILING_INFO_NOT_AVAILABLE:
			return "profiling info not available";
		case CL_MEM_COPY_OVERLAP:
			return "memory copy overlap";
		case CL_IMAGE_FORMAT_MISMATCH:
			return "image format mismatch";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:
			return "image format not supported";
		case CL_BUILD_PROGRAM_FAILURE:
			return "build program failure";
		case CL_MAP_FAILURE:
			return "map failure";
		case CL_MISALIGNED_SUB_BUFFER_OFFSET:
			return "misaligned sub buffer offset";
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
			return "exec status error for events in wait list";
		case CL_INVALID_VALUE:
			return "invalid value";
		case CL_INVALID_DEVICE_TYPE:
			return "invalid device type";
		case CL_INVALID_PLATFORM:
			return "invalid platform";
		case CL_INVALID_DEVICE:
			return "invalid device";
		case CL_INVALID_CONTEXT:
			return "invalid context";
		case CL_INVALID_QUEUE_PROPERTIES:
			return "invalid queue properties";
		case CL_INVALID_COMMAND_QUEUE:
			return "invalid command queue";
		case CL_INVALID_HOST_PTR:
			return "invalid host pointer";
		case CL_INVALID_MEM_OBJECT:
			return "invalid memory object";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
			return "invalid image format descriptor";
		case CL_INVALID_IMAGE_SIZE:
			return "invalid image size";
		case CL_INVALID_SAMPLER:
			return "invalid sampler";
		case CL_INVALID_BINARY:
			return "invalid binary";
		case CL_INVALID_BUILD_OPTIONS:
			return "invalid build options";
		case CL_INVALID_PROGRAM:
			return "invalid program";
		case CL_INVALID_PROGRAM_EXECUTABLE:
			return "invalid program executable";
		case CL_INVALID_KERNEL_NAME:
			return "invalid kernel name";
		case CL_INVALID_KERNEL_DEFINITION:
			return "invalid kernel definition";
		case CL_INVALID_KERNEL:
			return "invalid kernel";
		case CL_INVALID_ARG_INDEX:
			return "invalid argument index";
		case CL_INVALID_ARG_VALUE:
			return "invalid argument value";
		case CL_INVALID_ARG_SIZE:
			return "invalid argument size";
		case CL_INVALID_KERNEL_ARGS:
			return "invalid kernel arguments";
		case CL_INVALID_WORK_DIMENSION:
			return "invalid work dimension";
		case CL_INVALID_WORK_GROUP_SIZE:
			return "invalid work group size";
		case CL_INVALID_WORK_ITEM_SIZE:
			return "invalid work item size";
		case CL_INVALID_GLOBAL_OFFSET:
			return "invalid global offset";
		case CL_INVALID_EVENT_WAIT_LIST:
			return "invalid event wait list";
		case CL_INVALID_EVENT:
			return "invalid event";
		case CL_INVALID_OPERATION:
			return "invalid operation";
		case CL_INVALID_GL_OBJECT:
			return "invalid GL object";
		case CL_INVALID_BUFFER_SIZE:
			return "invalid buffer size";
		case CL_INVALID_MIP_LEVEL:
			return "invalid MIP level";
		case CL_INVALID_GLOBAL_WORK_SIZE:
			return "invalid global work size";
		case CL_INVALID_PROPERTY:
			return "invalid property";
		default:
			snprintf(strbuf, sizeof(strbuf), "error code: %d", errcode);
			break;
	}
	return strbuf;
}
