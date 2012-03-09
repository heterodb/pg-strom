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

typedef struct {
	cl_context			context;
	cl_mem				shmbuf;
	cl_program			program;
	cl_kernel			kernel_qual;
	size_t			   *kernel_wkgrp_maxsz;
	size_t			   *kernel_wkgrp_unitsz;
	ShmsegQueue		   *host_cmdq;
	cl_int				num_cmdq;
	cl_command_queue   *gpu_cmdq;
} GpuServerState;

typedef struct {
	ChunkBuffer	   *chunk;
	cl_mem			mem_const;
	cl_mem			mem_cmds;
	cl_mem			mem_isnull;
	cl_mem			mem_values;
	cl_mem			mem_buffer;
	cl_event		ev_memcpy_HtoD;
	cl_event		ev_kernel_exec;
	cl_event		ev_memcpy_DtoH;
} GpuKernelState;

/*
 * Local declarations
 */
static const char  *pgstrom_opencl_error_string(int errcode);
static cl_uint		pgstrom_num_devices;
static DeviceInfo  *pgstrom_device_info;
static ShmsegQueue *pgstrom_host_cmdq;

static void
pgstrom_opencl_cleanup_kernel(cl_event event, cl_int status, void *arg)
{
	GpuKernelState *gpu_kern = (GpuKernelState *) arg;
	ChunkBuffer	   *chunk = gpu_kern->chunk;

	clReleaseEvent(gpu_kern->ev_memcpy_DtoH);
	clReleaseEvent(gpu_kern->ev_kernel_exec);
	clReleaseEvent(gpu_kern->ev_memcpy_HtoD);

	clReleaseMemObject(gpu_kern->mem_buffer);
	clReleaseMemObject(gpu_kern->mem_values);
	clReleaseMemObject(gpu_kern->mem_isnull);
	clReleaseMemObject(gpu_kern->mem_cmds);
	clReleaseMemObject(gpu_kern->mem_const);

	/*
	 * Enqueue the chunk-buffer to the caller
	 */
	Assert(status == CL_COMPLETE || status < 0);

	if (status == CL_COMPLETE)
		chunk->status = CHUNKBUF_STATUS_READY;
	else
		chunk->status = status;

	pgstrom_shmqueue_enqueue(chunk->recv_cmdq, &chunk->chain);

	/* cleanup */
	free(gpu_kern);
}

static int
pgstrom_opencl_schedule_kernel(GpuServerState *gpu_serv, ChunkBuffer *chunk)
{
	/* TODO: add device scheduling logic */
	return 0;
}

static cl_int
pgstrom_opencl_exec_kernel(GpuServerState *gpu_serv, ChunkBuffer *chunk)
{
	GpuKernelState	   *gpu_kern;
	cl_buffer_region	region;
	int			qindex;
	int		   *argbuf, *p;
	size_t		argsize;
	size_t		local_wkgrp_sz;
	size_t		global_wkgrp_sz;
	cl_int		ret;

	qindex = pgstrom_opencl_schedule_kernel(gpu_serv, chunk);

	gpu_kern = malloc(sizeof(GpuKernelState));
	if (!gpu_kern)
		return CL_OUT_OF_HOST_MEMORY;
	gpu_kern->chunk = chunk;

	/*
	 * Set up constant kernel arguments
	 */
	p = argbuf = alloca(sizeof(int) * 2 +
						VARSIZE_ANY_EXHDR(chunk->gpu_cmds) +
						sizeof(int) * 2 * chunk->nattrs);
	*p++ = chunk->nitems;
	*p++ = chunk->nattrs;
	memcpy(p, VARDATA(chunk->gpu_cmds), VARSIZE_ANY_EXHDR(chunk->gpu_cmds));
	p += VARSIZE_ANY_EXHDR(chunk->gpu_cmds) / sizeof(int);
	memcpy(p, chunk->cs_isnull, sizeof(int) * chunk->nattrs);
	p += chunk->nattrs;
	memcpy(p, chunk->cs_values, sizeof(int) * chunk->nattrs);
	p += chunk->nattrs;

	argsize = ((uintptr_t) p) - ((uintptr_t) argbuf);
	gpu_kern->mem_const = clCreateBuffer(gpu_serv->context,
										 CL_MEM_READ_ONLY |
										 CL_MEM_COPY_HOST_PTR,
										 argsize, argbuf, &ret);
	if (ret != CL_SUCCESS)
		goto error_0;

	/* The first argument: commands[] */
	region.origin = 0;
	region.size = 2 * sizeof(int) + VARSIZE_ANY_EXHDR(chunk->gpu_cmds);
	gpu_kern->mem_cmds = clCreateSubBuffer(gpu_kern->mem_const,
										   CL_MEM_READ_ONLY,
										   CL_BUFFER_CREATE_TYPE_REGION,
										   &region, &ret);
	if (ret != CL_SUCCESS)
		goto error_1;

	/* The second argument: cs_isnull[] */
	region.origin += region.size;
	region.size = sizeof(int) * chunk->nattrs;
	gpu_kern->mem_isnull = clCreateSubBuffer(gpu_kern->mem_const,
											 CL_MEM_READ_ONLY,
											 CL_BUFFER_CREATE_TYPE_REGION,
											 &region, &ret);
	if (ret != CL_SUCCESS)
		goto error_2;

	/* The third argument: cs_values[] */
	region.origin += region.size;
	region.size = sizeof(int) * chunk->nattrs;
	gpu_kern->mem_values = clCreateSubBuffer(gpu_kern->mem_const,
											 CL_MEM_READ_ONLY,
											 CL_BUFFER_CREATE_TYPE_REGION,
											 &region, &ret);
	if (ret != CL_SUCCESS)
		goto error_3;

	/* The forth argument: cs_rowmap */
	gpu_kern->mem_buffer = clCreateBuffer(gpu_serv->context,
										  CL_MEM_READ_WRITE,
										  chunk->dma_length,
										  chunk->cs_rowmap,
										  &ret);
	if (ret != CL_SUCCESS)
		goto error_4;

	/* Enqueue async data transfer: HtoD */
	ret = clEnqueueWriteBuffer(gpu_serv->gpu_cmdq[qindex],
							   gpu_kern->mem_buffer,
							   CL_FALSE,
							   0,
							   chunk->dma_length,
							   chunk->cs_rowmap,
							   0, NULL,
							   &gpu_kern->ev_memcpy_HtoD);
	if (ret != CL_SUCCESS)
		goto error_5;

	/* Enqueue kernel execution */
	local_wkgrp_sz = gpu_serv->kernel_wkgrp_maxsz[qindex];
	if (local_wkgrp_sz > gpu_serv->kernel_wkgrp_unitsz[qindex])
		local_wkgrp_sz -= (local_wkgrp_sz %
						   gpu_serv->kernel_wkgrp_unitsz[qindex]);
	global_wkgrp_sz = (chunk->nitems + local_wkgrp_sz - 1) / local_wkgrp_sz;

	ret = clEnqueueNDRangeKernel(gpu_serv->gpu_cmdq[qindex],
								 gpu_serv->kernel_qual,
								 1,
								 NULL,
								 &global_wkgrp_sz,
								 &local_wkgrp_sz,
								 1, &gpu_kern->ev_memcpy_HtoD,
								 &gpu_kern->ev_kernel_exec);
	if (ret != CL_SUCCESS)
		goto error_6;

	/* Enqueue async data transfer: DtoH */
	ret = clEnqueueReadBuffer(gpu_serv->gpu_cmdq[qindex],
							  gpu_kern->mem_buffer,
							  CL_FALSE,
							  0,
							  chunk->nitems / BITS_PER_BYTE,
							  chunk->cs_rowmap,
							  1, &gpu_kern->ev_kernel_exec,
							  &gpu_kern->ev_memcpy_DtoH);
	if (ret != CL_SUCCESS)
		goto error_7;

	/* Register callback function to clean up */
	ret = clSetEventCallback(gpu_kern->ev_memcpy_DtoH,
							 CL_COMPLETE,
							 pgstrom_opencl_cleanup_kernel,
							 gpu_kern);
    if (ret != CL_SUCCESS)
		goto error_8;

	return CL_SUCCESS;

error_8:
	clWaitForEvents(1, &gpu_kern->ev_memcpy_DtoH);
	clReleaseEvent(gpu_kern->ev_memcpy_DtoH);
error_7:
	clWaitForEvents(1, &gpu_kern->ev_kernel_exec);
	clReleaseEvent(gpu_kern->ev_kernel_exec);
error_6:
	clWaitForEvents(1, &gpu_kern->ev_memcpy_HtoD);
	clReleaseEvent(gpu_kern->ev_memcpy_HtoD);
error_5:
	clReleaseMemObject(gpu_kern->mem_buffer);
error_4:
	clReleaseMemObject(gpu_kern->mem_values);
error_3:
	clReleaseMemObject(gpu_kern->mem_isnull);
error_2:
	clReleaseMemObject(gpu_kern->mem_cmds);
error_1:
	clReleaseMemObject(gpu_kern->mem_const);
error_0:
	free(gpu_kern);
	return ret;
}

static void *
pgstrom_opencl_server(void *arg)
{
	GpuServerState *gpu_serv = (GpuServerState *)arg;
	ChunkBuffer	   *chunk;
	ShmsegList	   *item;
	cl_int			ret;

	while (true)
	{
		item = pgstrom_shmqueue_dequeue(gpu_serv->host_cmdq);
		chunk = container_of(item, ChunkBuffer, chain);

		ret = pgstrom_opencl_exec_kernel(gpu_serv, chunk);
		if (ret != CL_SUCCESS)
		{
			chunk->status = ret;
			pgstrom_shmqueue_enqueue(chunk->recv_cmdq, &chunk->chain);
		}
	}
	elog(FATAL, "PG-Strom: GPU computing server should never exit");
	return NULL;
}

static pthread_t
pgstrom_opencl_server_launch(cl_context context,
							 cl_program program,
							 cl_mem     shmbuf,
							 int		num_gpu_cmdq,
							 cl_command_queue *gpu_cmdq,
							 cl_device_id *devices)
{
	GpuServerState *gpu_serv;
	pthread_t		thread;
	cl_kernel		kernel;
	cl_int			i, ret;
	char		   *p;

	/* construction of GpuServerState */
	gpu_serv = malloc(sizeof(GpuServerState) +
					  sizeof(size_t) * num_gpu_cmdq +
						  sizeof(size_t) * num_gpu_cmdq +
					  sizeof(cl_command_queue) * num_gpu_cmdq);
	if (!gpu_serv)
		elog(ERROR, "out of memory");
	p = (char *)&gpu_serv[1];
	gpu_serv->kernel_wkgrp_maxsz = (size_t *) p;
	p += sizeof(size_t) * num_gpu_cmdq;
	gpu_serv->kernel_wkgrp_unitsz = (size_t *) p;
	p += sizeof(size_t) * num_gpu_cmdq;
	gpu_serv->gpu_cmdq = (cl_command_queue *) p;

	/* reference to the command queue from backend to opencl-serv */
	gpu_serv->host_cmdq = pgstrom_host_cmdq;

	/* increment refcount of context */
	ret = clRetainContext(context);
	if (ret != CL_SUCCESS)
		elog(ERROR, "OpenCL: failed to retain context (%s)",
			 pgstrom_opencl_error_string(ret));
	gpu_serv->context = context;

	/* increment refcount of program */
	ret = clRetainProgram(program);
	if (ret != CL_SUCCESS)
		elog(ERROR, "OpenCL: failed to retain program (%s)",
			 pgstrom_opencl_error_string(ret));
	gpu_serv->program = program;

	/*
	 * Kernel object should be constructed for each threads
	 * to avoid race condition around clSetKernelArg and
	 * clEnqueueNDRangeKernel
	 */
	kernel = clCreateKernel(program, "opencl_qual", &ret);
	if (ret != CL_SUCCESS)
		elog(ERROR, "OpenCL: failed to create kernel object (%s)",
			 pgstrom_opencl_error_string(ret));
	gpu_serv->kernel_qual = kernel;

	for (i=0; i < num_gpu_cmdq; i++)
	{
		ret = clGetKernelWorkGroupInfo(kernel,
									   devices[i],
									   CL_KERNEL_WORK_GROUP_SIZE,
									   sizeof(size_t),
									   &gpu_serv->kernel_wkgrp_maxsz[i],
									   NULL);
		if (ret != CL_SUCCESS)
			elog(ERROR, "OpenCL: failed to get local workgroup size (%s)",
				 pgstrom_opencl_error_string(ret));

		ret = clGetKernelWorkGroupInfo(kernel,
									   devices[i],
								CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
									   sizeof(size_t),
									   &gpu_serv->kernel_wkgrp_unitsz[i],
									   NULL);
		if (ret != CL_SUCCESS)
			elog(ERROR, "OpenCL: failed to get local workgroup unit size (%s)",
				 pgstrom_opencl_error_string(ret));
	}

	/* increment refcount of shared memory buffer */
	ret = clRetainMemObject(shmbuf);
	if (ret != CL_SUCCESS)
		elog(ERROR, "OpenCL: failed to retain shared memory buffer (%s)",
			 pgstrom_opencl_error_string(ret));
	gpu_serv->shmbuf = shmbuf;

	/* increment refcount of command queues*/
	for (i=0; i < num_gpu_cmdq; i++)
	{
		ret = clRetainCommandQueue(gpu_cmdq[i]);
		if (ret != CL_SUCCESS)
			elog(ERROR, "OpenCL: failed to retain command queue (%s)",
				 pgstrom_opencl_error_string(ret));
		gpu_serv->gpu_cmdq[i] = gpu_cmdq[i];
	}
	gpu_serv->num_cmdq = num_gpu_cmdq;

	/*
	 * Launch OpenCL Server Thread
	 */
	if (pthread_create(&thread, NULL, pgstrom_opencl_server, gpu_serv) != 0)
		elog(ERROR, "failed to launch OpenCL server thread");

	return thread;
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
pgstrom_opencl_enqueue_chunk(ChunkBuffer *chunk)
{
	pgstrom_shmqueue_enqueue(pgstrom_host_cmdq, &chunk->chain);
}

void
pgstrom_opencl_startup(void *shmptr, Size shmsize)
{
	int					pgstrom_num_opencl_serv = 2;	/* tentasive */
	cl_device_id	   *devices;
	cl_context			context;
	cl_program			program;
	cl_command_queue   *gpu_cmdq;
	cl_mem				shmbuf;
	cl_int				i, ret;
	char				namebuf[MAXPGPATH];
	char			   *kernel_path;
    bytea			   *kernel_bytea;
    const char		   *kernel_source;
    size_t				kernel_length;

	/* Create a command queue Host to GPU-serv */
	pgstrom_host_cmdq = pgstrom_shmseg_alloc(sizeof(ShmsegQueue));
	if (!pgstrom_host_cmdq)
		elog(ERROR, "PG-Strom: out of shared memory");
	if (!pgstrom_shmqueue_init(pgstrom_host_cmdq))
		elog(ERROR, "PG-Strom: failed to init shmqueue");

	/* Create a OpenCL context */
	devices = alloca(pgstrom_num_devices * sizeof(cl_device_id));
	for (i=0; i < pgstrom_num_devices; i++)
		devices[i] = pgstrom_device_info[i].device_id;

	context = clCreateContext(NULL,
							  pgstrom_num_devices,
							  devices,
							  NULL, NULL, &ret);
	if (ret != CL_SUCCESS)
		elog(ERROR, "OpenCL: failed to create device context (%s)",
			 pgstrom_opencl_error_string(ret));

	/* Load the source code of OpenCL kernel */
	get_share_path(my_exec_path, namebuf);
	kernel_path = alloca(strlen(namebuf) + 40);
	sprintf(kernel_path, "%s/extension/opencl_kernel", namebuf);

	kernel_bytea = read_binary_file(kernel_path, 0, -1);
	kernel_source = VARDATA(kernel_bytea);
	kernel_length = VARSIZE_ANY_EXHDR(kernel_bytea);

	program = clCreateProgramWithSource(context,
										1,
										&kernel_source,
										&kernel_length,
										&ret);
	if (ret != CL_SUCCESS)
		elog(ERROR, "OpenCL: failed to create program (%s)",
			 pgstrom_opencl_error_string(ret));

	/* Build the code of OpenCL kernel */
	ret = clBuildProgram(program,
						 pgstrom_num_devices,
                         devices,
                         "-cl-mad-enable",  // TODO: add options...
                         NULL,
                         NULL);
	if (ret != CL_SUCCESS)
	{
		if (ret == CL_BUILD_PROGRAM_FAILURE)
		{
			char   *build_log = alloca(65536);

			if (clGetProgramBuildInfo(program,
									  devices[0],
									  CL_PROGRAM_BUILD_LOG,
									  65536,
									  build_log,
									  NULL) == CL_SUCCESS)
				elog(LOG, "%s", build_log);
		}
		elog(ERROR, "OpenCL: failed to build program (%s)",
			 pgstrom_opencl_error_string(ret));
	}
	else
	{
		size_t *prog_binsz = alloca(sizeof(size_t) * pgstrom_num_devices);

		if (clGetProgramInfo(program,
							 CL_PROGRAM_BINARY_SIZES,
							 sizeof(size_t) * pgstrom_num_devices,
							 prog_binsz,
							 NULL) == CL_SUCCESS)
			elog(LOG, "binary size = %lu kB", prog_binsz[0] / 1024);
	}

	/* Set up shared memory segment as page-locked memory */
	shmbuf = clCreateBuffer(context,
							CL_MEM_USE_HOST_PTR,
							shmsize, shmptr,
							&ret);
	if (ret != CL_SUCCESS)
		elog(ERROR, "OpenCL: failed to create host buffer (%s)",
			 pgstrom_opencl_error_string(ret));

	/* Create command queues for each devices */
	gpu_cmdq = alloca(sizeof(cl_command_queue) * pgstrom_num_devices);
	for (i=0; i < pgstrom_num_devices; i++)
	{
		gpu_cmdq[i] =
			clCreateCommandQueue(context,
								 devices[i],
								 CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
								 CL_QUEUE_PROFILING_ENABLE,
								 &ret);
		if (ret != CL_SUCCESS)
			elog(ERROR, "OpenCL: failed to create device command queue (%s)",
				 pgstrom_opencl_error_string(ret));
	}

	/* Launch OpenCL Server threads */
	for (i=0; i < pgstrom_num_opencl_serv; i++)
	{
		pgstrom_opencl_server_launch(context,
									 program,
									 shmbuf,
									 pgstrom_num_devices,
									 gpu_cmdq,
									 devices);
	}

	/* Decrement refcnt of the resources */
	for (i=0; i < pgstrom_num_devices; i++)
	{
		ret = clReleaseCommandQueue(gpu_cmdq[i]);
		if (ret != CL_SUCCESS)
			elog(ERROR, "OpenCL: failed to release command queue (%s)",
				 pgstrom_opencl_error_string(ret));
	}

	ret = clReleaseMemObject (shmbuf);
	if (ret != CL_SUCCESS)
		elog(ERROR, "OpenCL: failed to release memory object (%s)",
			 pgstrom_opencl_error_string(ret));

	ret = clReleaseProgram(program);
	if (ret != CL_SUCCESS)
		elog(ERROR, "OpenCL: failed to release program object (%s)",
			 pgstrom_opencl_error_string(ret));

	ret = clReleaseContext(context);
	if (ret != CL_SUCCESS)
		elog(ERROR, "OpenCL: failed to release context (%s)",
			 pgstrom_opencl_error_string(ret));
}

void
pgstrom_opencl_init(void)
{
	cl_platform_id	platform;
	cl_device_id	devices[64];
	cl_int			i, ret;

	/*
	 * TODO: GUC to choose a particular platform to be used,
	 *       if we have multiple platforms at a single box.
	 *       Right now, we use the first one only.
	 */
	ret = clGetPlatformIDs(1, &platform, NULL);
	if (ret != CL_SUCCESS)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("OpenCL: failed to get platform handlers (%s)",
						pgstrom_opencl_error_string(ret))));

	ret = clGetDeviceIDs(platform,
						 CL_DEVICE_TYPE_DEFAULT,
						 lengthof(devices), devices,
						 &pgstrom_num_devices);
	if (ret != CL_SUCCESS)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("OpenCL: failed to get device handlers (%s)",
						pgstrom_opencl_error_string(ret))));

	pgstrom_device_info = malloc(sizeof(DeviceInfo) * pgstrom_num_devices);
	if (!pgstrom_device_info)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of memory")));

	for (i=0; i < pgstrom_num_devices; i++)
	{
		DeviceInfo  *devinfo = &pgstrom_device_info[i];

		devinfo->platform_id = platform;
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
