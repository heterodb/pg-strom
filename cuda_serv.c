/*
 * cuda_serv.c
 *
 * The background computing engine stick on the CUDA infrastructure.
 * In addition, it also provide catalog of supported type and functions.
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "access/hash.h"
#include "catalog/pg_namespace.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_type.h"
#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/syscache.h"
#include "pg_strom.h"
#include "cuda_cmds.h"
#include <cuda.h>
#include <unistd.h>

/*
 * Local type definitions
 */
typedef struct {
	CUdevice	device;
	CUcontext	context;
	CUmodule	module;
	CUfunction	kernel_qual;
	char		dev_name[256];
	int			dev_major;
	int			dev_minor;
	int			dev_proc_nums;
	int			dev_proc_warp_sz;
	int			dev_proc_clock;
	size_t		dev_global_mem_sz;
} GpuDevState;

typedef struct {
	ShmsegList	chain;
	ChunkBuffer	*chunk;
	CUcontext	context;		/* reference to GpuDevState */
	CUfunction	kernel_qual;	/* reference to GpuDevState */
	CUstream	stream;
	CUdeviceptr	devmem;
	CUevent		events[4];
} GpuExecState;

/*
 * Local variables
 */
#define MAX_NUM_LOAD_SERVS	32
#define MAX_NUM_POLL_SERVS	4
static const char  *pgstrom_gpu_error_string(CUresult errcode);
static int			gpu_device_nums;
static GpuDevState *gpu_device_state;
static int			gpu_num_load_servs;
static int			gpu_num_poll_servs;
static pthread_t	gpu_load_servs[MAX_NUM_LOAD_SERVS];
static pthread_t	gpu_poll_servs[MAX_NUM_POLL_SERVS];
static ShmsegQueue *gpu_load_cmdq;
static ShmsegQueue	gpu_poll_cmdq;
static List		   *gpu_type_info_slot[128];
static List		   *gpu_func_info_slot[512];

static void *
pgstrom_gpu_poll_serv(void *argv)
{
	GpuExecState   *gexec;
	ChunkBuffer	   *chunk;
	ShmsegList	   *item;
	CUcontext		dummy;
	int				index;
	float			elapsed;
	CUresult		ret;

	while ((item = pgstrom_shmqueue_dequeue(&gpu_poll_cmdq)) != NULL)
	{
		gexec = container_of(item, GpuExecState, chain);
		chunk = gexec->chunk;

		ret = cuCtxPushCurrent(gexec->context);
		Assert(ret == CUDA_SUCCESS);

		/*
		 * XXX: We have no interface to synchronize at least one stream
		 * that complete all the tasks inside of the stream, so we try
		 * to synchronize the earliest stream being enqueued.
		 * Heuristically, it almost matches with order of completion.
		 * It is an idea to increase number of threads for chunk-poller.
		 */
		if (chunk->pf_enabled)
			ret = cuEventSynchronize(gexec->events[3]);
		else
			ret = cuStreamSynchronize(gexec->stream);

		if (ret != CUDA_SUCCESS)
		{
			chunk->status = CHUNKBUF_STATUS_ERROR;
			snprintf(chunk->error_msg, sizeof(chunk->error_msg),
					 "cuda: failed on stream synchronization (%s)",
					 pgstrom_gpu_error_string(ret));
		}
		else
		{
			chunk->status = CHUNKBUF_STATUS_READY;
			if (chunk->pf_enabled)
			{
				ret = cuEventElapsedTime(&elapsed,
										 gexec->events[0],
										 gexec->events[1]);
				chunk->pf_async_memcpy = (uint64)(elapsed * 1000.0);

				ret = cuEventElapsedTime(&elapsed,
										 gexec->events[1],
										 gexec->events[2]);
				chunk->pf_async_kernel = (uint64)(elapsed * 1000.0);

				ret = cuEventElapsedTime(&elapsed,
										 gexec->events[2],
										 gexec->events[3]);
				chunk->pf_async_memcpy += (uint64)(elapsed * 1000.0);
			}
		}
		ret = cuMemFree(gexec->devmem);
		Assert(ret == CUDA_SUCCESS);

		if (chunk->pf_enabled)
		{
			for (index=0; index < lengthof(gexec->events); index++)
			{
				ret = cuEventDestroy(gexec->events[index]);
				Assert(ret == CUDA_SUCCESS);
			}
		}

		ret = cuStreamDestroy(gexec->stream);
		Assert(ret == CUDA_SUCCESS);

		cuCtxPopCurrent(&dummy);

		/*
		 * Back the chunk-buffer to its originator
		 */
		pgstrom_shmqueue_enqueue(chunk->recv_cmdq, &chunk->chain);

		free(gexec);
	}
	elog(FATAL, "%s should not exit", __FUNCTION__);
	return NULL;
}

static int
pgstrom_gpu_schedule(ChunkBuffer *chunk)
{
	static int	next_gpu_scheduled = 0;

	/*
	 * TODO: more wise scheduling policy, rather than round-robin.
	 * IDEA: based on length of pending queue, computing capabilities, ...
	 */
	return __sync_fetch_and_add(&next_gpu_scheduled, 1) % gpu_device_nums;
}

static bool
pgstrom_gpu_exec_kernel(ChunkBuffer *chunk)
{
	GpuExecState   *gexec;
	CUcontext		dummy;
	CUdeviceptr		kernel_data[5];
	void		   *kernel_args[5];
	CUresult		ret;
	int				index;
	int				n_blocks;
	int				n_threads;

	gexec = malloc(sizeof(GpuExecState));
	if (!gexec)
	{
		snprintf(chunk->error_msg, sizeof(chunk->error_msg),
				 "PG-Strom: failed on GpuExecState allocation");
		goto error_0;
	}
	memset(gexec, 0, sizeof(GpuExecState));
	gexec->chunk = chunk;

	index = pgstrom_gpu_schedule(chunk);
	gexec->context = gpu_device_state[index].context;
	gexec->kernel_qual = gpu_device_state[index].kernel_qual;

	ret = cuCtxPushCurrent(gexec->context);
	if (ret != CUDA_SUCCESS)
	{
		snprintf(chunk->error_msg, sizeof(chunk->error_msg),
				 "cuda: failed on switch context (%s)",
				 pgstrom_gpu_error_string(ret));
		goto error_1;
	}

	ret = cuStreamCreate(&gexec->stream, 0);
	if (ret != CUDA_SUCCESS)
	{
		snprintf(chunk->error_msg, sizeof(chunk->error_msg),
				 "cuda: failed on create stream (%s)",
				 pgstrom_gpu_error_string(ret));
		goto error_2;
	}

	if (chunk->pf_enabled)
	{
		for (index=0; index < lengthof(gexec->events); index++)
		{
			/*
			 * XXX - Now we're under investigation why CU_EVENT_DEFAULT
			 * lock out synchronization mechanism. Thus, it is unavailable
			 * to obtain elapsed time between asyncronous operations.
			 */
			ret = cuEventCreate(&gexec->events[index],
								CU_EVENT_DISABLE_TIMING);
			if (ret != CUDA_SUCCESS)
			{
				while (--index >= 0)
					cuEventDestroy(gexec->events[index]);
				snprintf(chunk->error_msg, sizeof(chunk->error_msg),
						 "cuda: failed on create event object (%s)",
						 pgstrom_gpu_error_string(ret));
				goto error_3;
			}
		}
	}

	/*
	 * Allocation of the device memory
	 */
	ret = cuMemAlloc(&gexec->devmem, chunk->dma_length);
	if (ret != CUDA_SUCCESS)
	{
		snprintf(chunk->error_msg, sizeof(chunk->error_msg),
				 "cuda: failed on allocate device memory (%s)",
				 pgstrom_gpu_error_string(ret));
		goto error_4;
	}

	/*
	 * Asynchronous copy of chunk buffer from host to device
	 */
	if (chunk->pf_enabled)
	{
		ret = cuEventRecord(gexec->events[0], gexec->stream);
		if (ret != CUDA_SUCCESS)
		{
			snprintf(chunk->error_msg, sizeof(chunk->error_msg),
					 "cuda: failed on enqueue prep HtoD copy event (%s)",
					 pgstrom_gpu_error_string(ret));
			goto error_5;
		}
	}

	ret = cuMemcpyHtoDAsync(gexec->devmem,
							chunk->dma_buffer,
							chunk->dma_length,
							gexec->stream);
	if (ret != CUDA_SUCCESS)
	{
		snprintf(chunk->error_msg, sizeof(chunk->error_msg),
				 "cuda: failed on enqueue cuMemcpyHtoDAsync (%s)",
				 pgstrom_gpu_error_string(ret));
		goto error_5;
	}

	if (chunk->pf_enabled)
	{
		ret = cuEventRecord(gexec->events[1], gexec->stream);
		if (ret != CUDA_SUCCESS)
		{
			snprintf(chunk->error_msg, sizeof(chunk->error_msg),
					 "cuda: failed on enqueue post HtoD copy event (%s)",
					 pgstrom_gpu_error_string(ret));
			goto error_5;
		}
	}

	/*
	 * Asynchronous kernel execution on this chunk buffer
	 */
	ret = cuFuncGetAttribute(&n_threads,
							 CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
							 gexec->kernel_qual);
	if (ret != CUDA_SUCCESS)
	{
		snprintf(chunk->error_msg, sizeof(chunk->error_msg),
				 "cuda: failed to obtain max threads / block ratio (%s)",
				 pgstrom_gpu_error_string(ret));
		goto error_6;
	}

	kernel_data[0] = chunk->nitems;			/* int nitems */
	kernel_args[0] = &kernel_data[0];
	kernel_data[1] = gexec->devmem + ((char *)chunk->gpu_cmds -
									  (char *)chunk->dma_buffer);
	kernel_args[1] = &kernel_data[1];		/* int commands[] */
	kernel_data[2] = gexec->devmem + ((char *)chunk->cs_isnull -
									  (char *)chunk->dma_buffer);
	kernel_args[2] = &kernel_data[2];		/* int cs_isnull[] */
	kernel_data[3] = gexec->devmem + ((char *)chunk->cs_values -
									  (char *)chunk->dma_buffer);
	kernel_args[3] = &kernel_data[3];		/* int cs_values[] */
	kernel_data[4] = gexec->devmem + ((char *)chunk->cs_rowmap -
									  (char *)chunk->dma_buffer);
	kernel_args[4] = &kernel_data[4];		/* uchar cs_rowmap[] */

	n_blocks = ((chunk->nitems + n_threads * BITS_PER_BYTE - 1) /
				(n_threads * BITS_PER_BYTE));
	ret = cuLaunchKernel(gexec->kernel_qual,
						 n_blocks, 1, 1,
						 n_threads, 1, 1,
						 0,
						 gexec->stream,
						 kernel_args,
						 NULL);
	if (ret != CUDA_SUCCESS)
	{
		snprintf(chunk->error_msg, sizeof(chunk->error_msg),
				 "cuda: failed on enqueue an execution of kernel "
				 "(n_block=%u, b_threads=%u) : %s", n_blocks, n_threads,
				 pgstrom_gpu_error_string(ret));
		goto error_6;
	}

	/*
	 * Write back of the result
	 */
	if (chunk->pf_enabled)
	{
		ret = cuEventRecord(gexec->events[2], gexec->stream);
		if (ret != CUDA_SUCCESS)
		{
			snprintf(chunk->error_msg, sizeof(chunk->error_msg),
					 "cuda: failed on enqueue prep DtoH copy event (%s)",
					 pgstrom_gpu_error_string(ret));
			goto error_5;
		}
	}

	ret = cuMemcpyDtoHAsync(chunk->cs_rowmap,
							gexec->devmem + (uintptr_t)(chunk->cs_rowmap -
														chunk->dma_buffer),
							chunk->nitems / BITS_PER_BYTE,
							gexec->stream);
	if (ret != CUDA_SUCCESS)
	{
		snprintf(chunk->error_msg, sizeof(chunk->error_msg),
				 "cuda: failed on enqueue cuMemcpyDtoHAsync (%s)",
				 pgstrom_gpu_error_string(ret));
		goto error_6;
	}

	if (chunk->pf_enabled)
	{
		ret = cuEventRecord(gexec->events[3], gexec->stream);
		if (ret != CUDA_SUCCESS)
		{
			snprintf(chunk->error_msg, sizeof(chunk->error_msg),
					 "cuda: failed on enqueue post DtoH copy event (%s)",
					 pgstrom_gpu_error_string(ret));
			goto error_5;
		}
	}

	/*
	 * A series of sequence were successfully enqueued, so we'll wait
	 * for completion of the commands by chunk-poller server.
	 */
	pgstrom_shmqueue_enqueue(&gpu_poll_cmdq, &gexec->chain);

	cuCtxPopCurrent(&dummy);

	return true;

error_6:
	cuStreamSynchronize(gexec->stream);	
error_5:
	cuMemFree(gexec->devmem);
error_4:
	if (chunk->pf_enabled)
	{
		for (index=0; index < lengthof(gexec->events); index++)
			cuEventDestroy(gexec->events[index]);
	}
error_3:
	cuStreamDestroy(gexec->stream);
error_2:
	cuCtxPopCurrent(&dummy);
error_1:
	free(gexec);
error_0:
	return false;
}

static void *
pgstrom_gpu_load_serv(void *argv)
{
	ChunkBuffer	   *chunk;
	ShmsegList	   *item;
	struct timeval	tv;

	while ((item = pgstrom_shmqueue_dequeue(gpu_load_cmdq)) != NULL)
	{
		chunk = container_of(item, ChunkBuffer, chain);

		Assert(chunk->status == CHUNKBUF_STATUS_EXEC);
		Assert(chunk->gpu_cmds != NULL);
		Assert(chunk->cs_isnull != NULL);
		Assert(chunk->cs_values != NULL);
		Assert(chunk->cs_rowmap != NULL);

		Assert(((char *)chunk->gpu_cmds - (char *)chunk->dma_buffer) >= 0);
		Assert(((char *)chunk->gpu_cmds -
				(char *)chunk->dma_buffer) < chunk->dma_length);
		Assert(((char *)chunk->cs_isnull - (char *)chunk->dma_buffer) >= 0);
		Assert(((char *)chunk->cs_isnull -
				(char *)chunk->dma_buffer) < chunk->dma_length);
		Assert(((char *)chunk->cs_values - (char *)chunk->dma_buffer) >= 0);
		Assert(((char *)chunk->cs_values -
				(char *)chunk->dma_buffer) < chunk->dma_length);
		Assert(((char *)chunk->cs_rowmap - (char *)chunk->dma_buffer) >= 0);
		Assert(((char *)chunk->cs_rowmap -
				(char *)chunk->dma_buffer) < chunk->dma_length);
		if (chunk->pf_enabled)
		{
			gettimeofday(&tv, NULL);
			chunk->pf_queue_wait += TIMEVAL_ELAPSED(&chunk->pf_timeval, &tv);
		}

		if (!pgstrom_gpu_exec_kernel(chunk))
		{
			chunk->status = CHUNKBUF_STATUS_ERROR;
			pgstrom_shmqueue_enqueue(chunk->recv_cmdq, &chunk->chain);
		}
	}
	elog(FATAL, "%s should not exit", __FUNCTION__);
	return NULL;
}

void pgstrom_gpu_enqueue_chunk(ChunkBuffer *chunk)
{
	pgstrom_shmqueue_enqueue(gpu_load_cmdq, &chunk->chain);
}

int
pgstrom_gpu_num_devices(void)
{
	return gpu_device_nums;
}





/*
 *
 *
 *
 *
 */
void
pgstrom_gpu_startup(void *shmptr, Size shmsize)
{
	CUresult	ret;
	char	   *shmbase;
	char		namebuf[MAXPGPATH];
	char	   *kernel_path;
	bytea	   *kernel_bytea;
	char	   *kernel_image;
	int			i;

	/*
	 * register shared memory segment as page-locked memory
	 */
	shmbase = (void *)(((uintptr_t) shmptr) & ~(getpagesize() - 1));
	shmsize = ((((uintptr_t) shmptr & (getpagesize() - 1))
				+ shmsize + getpagesize() - 1)
			   & ~(getpagesize() - 1));
	ret = cuMemHostRegister(shmbase, shmsize,
							CU_MEMHOSTREGISTER_PORTABLE);
	if (ret != CUDA_SUCCESS)
		elog(ERROR, "cuda: failed lock shared memory segment (%s)",
			 pgstrom_gpu_error_string(ret));
	elog(LOG, "cuda: 0x%p - 0x%p were locked for DMA buffer",
		 shmbase, ((char *)shmbase) + shmsize);

	/*
	 * Create a command queue from backend to chunk-loader
	 */
	gpu_load_cmdq = pgstrom_shmseg_alloc(sizeof(ShmsegQueue));
	if (!gpu_load_cmdq)
		elog(ERROR, "PG-Strom: out of shared memory");
	if (!pgstrom_shmqueue_init(gpu_load_cmdq))
		elog(ERROR, "PG-Strom: failed to init shmqueue of chunk-loader");

	/*
	 * Create a command queue from chunk-loader to chunk-poller
	 * (this command queue is in local memory)
	 */
	if (!pgstrom_shmqueue_init(&gpu_poll_cmdq))
		elog(ERROR, "PG-Strom: failed to init shmqueue of chunk-poller");

	/*
	 * Load the module for each devices
	 */
	get_share_path(my_exec_path, namebuf);
	kernel_path = alloca(strlen(namebuf) + 40);
	sprintf(kernel_path, "%s/extension/cuda_kernel.ptx", namebuf);
	kernel_bytea = read_binary_file(kernel_path, 0, -1);
	kernel_image = text_to_cstring(kernel_bytea);

	for (i=0; i < gpu_device_nums; i++)
	{
		CUcontext	dummy;
		CUmodule	module;
		CUfunction	kernel_qual;
		int			fn_max_threads;
		int			fn_num_regs;
		int			fn_ptx_version;
		int			fn_bin_version;

		ret = cuCtxPushCurrent(gpu_device_state[i].context);
		if (ret != CUDA_SUCCESS)
			elog(ERROR, "cuda: failed to switch context (%s)",
				 pgstrom_gpu_error_string(ret));

		// TODO: JIT options using cuModuleLoadDataEx
		ret = cuModuleLoadData(&module, kernel_image);
		if (ret != CUDA_SUCCESS)
			elog(ERROR, "cuda: failed to load module (%s)",
				 pgstrom_gpu_error_string(ret));

		ret =  cuModuleGetFunction(&kernel_qual, module, "kernel_qual");
		if (ret != CUDA_SUCCESS)
			elog(ERROR, "cuda: failed to resolve \"kernel_qual\" (%s)",
				 pgstrom_gpu_error_string(ret));

		ret = cuFuncSetCacheConfig(kernel_qual, CU_FUNC_CACHE_PREFER_L1);
		if (ret != CUDA_SUCCESS)
			elog(ERROR, "cuda: failed to switch L1 cache preference (%s)",
				 pgstrom_gpu_error_string(ret));

		if ((ret = cuFuncGetAttribute(&fn_max_threads,
									  CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
									  kernel_qual)) != CUDA_SUCCESS ||
			(ret = cuFuncGetAttribute(&fn_num_regs,
									  CU_FUNC_ATTRIBUTE_NUM_REGS,
									  kernel_qual)) != CUDA_SUCCESS ||
			(ret = cuFuncGetAttribute(&fn_ptx_version,
									  CU_FUNC_ATTRIBUTE_PTX_VERSION,
									  kernel_qual)) != CUDA_SUCCESS ||
			(ret = cuFuncGetAttribute(&fn_bin_version,
									  CU_FUNC_ATTRIBUTE_BINARY_VERSION,
									  kernel_qual)) != CUDA_SUCCESS)
			elog(ERROR, "cuda: failed to get attribute of the kernel (%s)",
				 pgstrom_gpu_error_string(ret));

		gpu_device_state[i].module = module;
		gpu_device_state[i].kernel_qual = kernel_qual;

		elog(LOG, "function \"kernel_qual\" on device[%d] %s; "
			 "threads/block ratio = %d, regs/thread ratio = %d, "
			 "ptx version = %d, binary version = %d",
			 i, gpu_device_state[i].dev_name,
			 fn_max_threads, fn_num_regs, fn_ptx_version, fn_bin_version);

		ret = cuCtxPopCurrent(&dummy);
		if (ret != CUDA_SUCCESS)
			elog(ERROR, "cuda: failed to switch context (%s)",
				 pgstrom_gpu_error_string(ret));
	}
	pfree(kernel_image);
	pfree(kernel_bytea);

	/*
	 * Launch the chunk-loader servers
	 */
	for (i=0; i < gpu_num_load_servs; i++)
	{
		if (pthread_create(&gpu_load_servs[i], NULL,
						   pgstrom_gpu_load_serv, NULL) != 0)
			elog(ERROR, "PG-Strom: failed to launch chunk-loader server");
	}

	/*
	 * Launch the chunk-poller servers
	 */
	for (i=0; i < gpu_num_poll_servs; i++)
	{
		if (pthread_create(&gpu_poll_servs[i], NULL,
						   pgstrom_gpu_poll_serv, NULL) != 0)
			elog(ERROR, "PG-Strom: failed to launch chunk-poller server");
	}
}

/*
 *
 *
 *
 */
void pgstrom_gpu_init(void)
{
	CUresult	ret;
	int			i, j;

	memset(gpu_type_info_slot, 0, sizeof(gpu_type_info_slot));
	memset(gpu_func_info_slot, 0, sizeof(gpu_func_info_slot));

	/*
	 * GUC Parameters
	 */
	DefineCustomIntVariable("pg_strom.num_load_servs",
							"number of servers to load chunks to devices",
							NULL,
							&gpu_num_load_servs,
							2,
							1,
							MAX_NUM_LOAD_SERVS,
							PGC_POSTMASTER,
							0,
							NULL, NULL, NULL);

	DefineCustomIntVariable("pg_strom.num_poll_servs",
							"number of servers to poll chunks to be ready",
							NULL,
							&gpu_num_poll_servs,
							1,
							1,
							MAX_NUM_POLL_SERVS,
							PGC_POSTMASTER,
							0,
							NULL, NULL, NULL);
	/*
	 * Initialize CUDA API
	 */
	ret = cuInit(0);
	if (ret != CUDA_SUCCESS)
		elog(ERROR, "CUDA: failed to initialize driver API (%s)",
			 pgstrom_gpu_error_string(ret));

	/*
	 * Collect device properties
	 */
	ret = cuDeviceGetCount(&gpu_device_nums);
	if (ret != CUDA_SUCCESS)
		elog(ERROR, "cuda: failed to get number of devices (%s)",
			 pgstrom_gpu_error_string(ret));

	gpu_device_state = malloc(sizeof(GpuDevState) * gpu_device_nums);
	if (!gpu_device_state)
		elog(ERROR, "out of memory");

	for (i=0; i < gpu_device_nums; i++)
	{
		GpuDevState *devstate = &gpu_device_state[i];
		static struct {
			size_t				offset;
			CUdevice_attribute	attribute;
		} device_attrs[] = {
			{ offsetof(GpuDevState, dev_proc_nums),
			  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT },
			{ offsetof(GpuDevState, dev_proc_warp_sz),
			  CU_DEVICE_ATTRIBUTE_WARP_SIZE },
			{ offsetof(GpuDevState, dev_proc_clock),
			  CU_DEVICE_ATTRIBUTE_CLOCK_RATE },
		};

		ret = cuDeviceGet(&devstate->device, i);
		if (ret != CUDA_SUCCESS)
			elog(ERROR, "cuda: failed to get device handler (%s)",
				 pgstrom_gpu_error_string(ret));

		ret = cuCtxCreate(&devstate->context, 0, devstate->device);
		if (ret != CUDA_SUCCESS)
			elog(ERROR, "cuda: failed to create device context (%s)",
				 pgstrom_gpu_error_string(ret));

		ret = cuDeviceGetName(devstate->dev_name,
							  sizeof(devstate->dev_name),
							  devstate->device);
		if (ret != CUDA_SUCCESS)
			elog(ERROR, "cuda: failed to get device name (%s)",
				 pgstrom_gpu_error_string(ret));

		ret = cuDeviceComputeCapability(&devstate->dev_major,
										&devstate->dev_minor,
										devstate->device);
		if (ret != CUDA_SUCCESS)
			elog(ERROR, "cuda: failed to get computing capability (%s)",
				 pgstrom_gpu_error_string(ret));

		ret = cuDeviceTotalMem(&devstate->dev_global_mem_sz, devstate->device);
		if (ret != CUDA_SUCCESS)
			elog(ERROR, "cuda: failed to get size of global memory (%s)",
				 pgstrom_gpu_error_string(ret));

		for (j=0; j < lengthof(device_attrs); j++)
		{
			ret = cuDeviceGetAttribute((int *)((uintptr_t) devstate +
											   device_attrs[j].offset),
									   device_attrs[j].attribute,
									   devstate->device);
			if (ret != CUDA_SUCCESS)
				elog(ERROR, "cuda: failed to get attribute of device (%s)",
					 pgstrom_gpu_error_string(ret));
		}

		/*
		 * Logs of the device properties
		 */
		elog(LOG, "PG-Strom: GPU device[%d] %s; capability v%d.%d, "
			 "%d of streaming processor units (%d wraps per unit, %dMHz)",
			 i, devstate->dev_name, devstate->dev_major, devstate->dev_minor,
             devstate->dev_proc_nums, devstate->dev_proc_warp_sz,
             devstate->dev_proc_clock / 1000);
	}
}

static const char *
pgstrom_gpu_error_string(CUresult errcode)
{
	static char	strbuf[256];

	switch (errcode)
	{
		case CUDA_SUCCESS:
			return "success";
		case CUDA_ERROR_INVALID_VALUE:
			return "invalid value";
		case CUDA_ERROR_OUT_OF_MEMORY:
			return "out of memory";
		case CUDA_ERROR_NOT_INITIALIZED:
			return "not initialized";
		case CUDA_ERROR_DEINITIALIZED:
			return "deinitialized";
		case CUDA_ERROR_PROFILER_DISABLED:
			return "profiler disabled";
		case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
			return "profiler not initialized";
		case CUDA_ERROR_PROFILER_ALREADY_STARTED:
			return "profiler already started";
		case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
			return "profiler already stopped";
		case CUDA_ERROR_NO_DEVICE:
			return "no device";
		case CUDA_ERROR_INVALID_DEVICE:
			return "invalid device";
		case CUDA_ERROR_INVALID_IMAGE:
			return "invalid image";
		case CUDA_ERROR_INVALID_CONTEXT:
			return "invalid context";
		case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
			return "context already current";
		case CUDA_ERROR_MAP_FAILED:
			return "map failed";
		case CUDA_ERROR_UNMAP_FAILED:
			return "unmap failed";
		case CUDA_ERROR_ARRAY_IS_MAPPED:
			return "array is mapped";
		case CUDA_ERROR_ALREADY_MAPPED:
			return "already mapped";
		case CUDA_ERROR_NO_BINARY_FOR_GPU:
			return "no binary for gpu";
		case CUDA_ERROR_ALREADY_ACQUIRED:
			return "already acquired";
		case CUDA_ERROR_NOT_MAPPED:
			return "not mapped";
		case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
			return "not mapped as array";
		case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
			return "not mapped as pointer";
		case CUDA_ERROR_ECC_UNCORRECTABLE:
			return "ecc uncorrectable";
		case CUDA_ERROR_UNSUPPORTED_LIMIT:
			return "unsupported limit";
		case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
			return "context already in use";
		case CUDA_ERROR_INVALID_SOURCE:
			return "invalid source";
		case CUDA_ERROR_FILE_NOT_FOUND:
			return "file not found";
		case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
			return "shared object symbol not found";
		case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
			return "shared object init failed";
		case CUDA_ERROR_OPERATING_SYSTEM:
			return "operating system";
		case CUDA_ERROR_INVALID_HANDLE:
			return "invalid handle";
		case CUDA_ERROR_NOT_FOUND:
			return "not found";
		case CUDA_ERROR_NOT_READY:
			return "not ready";
		case CUDA_ERROR_LAUNCH_FAILED:
			return "launch failed";
		case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
			return "launch out of resources";
		case CUDA_ERROR_LAUNCH_TIMEOUT:
			return "launch timeout";
		case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
			return "launch incompatible texturing";
		case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
			return "peer access already enabled";
		case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
			return "peer access not enabled";
		case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
			return "primary context active";
		case CUDA_ERROR_CONTEXT_IS_DESTROYED:
			return "context is destroyed";
		default:
			snprintf(strbuf, sizeof(strbuf), "cuda error = %d", errcode);
			break;
	}
	return strbuf;
}

/* ------------------------------------------------------------ *
 *
 * Catalog of GPU Types
 *
 * ------------------------------------------------------------ */
#define GPUTYPE_FLAGS_X2REGS	0x0001	/* need x2 registers */
#define GPUTYPE_FLAGS_FP64		0x0002	/* need 64bits FP support  */

static struct {
	Oid			type_oid;
	int			type_flags;
	int32		type_varref;
	int32		type_conref;
	const char *type_explain;
} gpu_type_catalog[] = {
	{
		BOOLOID,
		0,
		GPUCMD_VARREF_BOOL,
		GPUCMD_CONREF_BOOL,
		"uchar",
	},
	{
		INT2OID,
		0,
		GPUCMD_VARREF_INT2,
		GPUCMD_CONREF_INT2,
		"short",
	},
	{
		INT4OID,
		0,
		GPUCMD_VARREF_INT4,
		GPUCMD_CONREF_INT4,
		"int",
	},
	{
		INT8OID,
		GPUTYPE_FLAGS_X2REGS,
		GPUCMD_VARREF_INT8,
		GPUCMD_CONREF_INT8,
		"long",
	},
	{
		FLOAT4OID,
		0,
		GPUCMD_VARREF_FLOAT4,
		GPUCMD_CONREF_FLOAT4,
		"float",
	},
	{
		FLOAT8OID,
		GPUTYPE_FLAGS_X2REGS | GPUTYPE_FLAGS_FP64,
		GPUCMD_VARREF_FLOAT8,
		GPUCMD_CONREF_FLOAT8,
		"double",
	},
};

GpuTypeInfo *
pgstrom_gpu_type_lookup(Oid type_oid)
{
	GpuTypeInfo	   *entry;
	HeapTuple		tuple;
	Form_pg_type	typeform;
	MemoryContext	oldcxt;
	ListCell	   *cell;
	int				i, hash;

	hash = hash_uint32((uint32) type_oid) % lengthof(gpu_type_info_slot);
	foreach (cell, gpu_type_info_slot[hash])
	{
		entry = lfirst(cell);
		if (entry->type_oid == type_oid)
		{
			/* supported type has _varref and _conref command */
			if (entry->type_varref != 0 && entry->type_conref != 0)
				return entry;
			return NULL;
		}
	}

	/*
	 * Not found, insert a new entry
	 */
	tuple = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
	Assert(HeapTupleIsValid(tuple));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for type %u", type_oid);
	typeform = (Form_pg_type) GETSTRUCT(tuple);

	oldcxt = MemoryContextSwitchTo(CacheMemoryContext);

	entry = palloc0(sizeof(GpuTypeInfo));
	entry->type_oid = type_oid;
	if (typeform->typnamespace == PG_CATALOG_NAMESPACE)
	{
		for (i=0; i < lengthof(gpu_type_catalog); i++)
		{
			if (gpu_type_catalog[i].type_oid == type_oid)
			{
				if (gpu_type_catalog[i].type_flags & GPUTYPE_FLAGS_X2REGS)
					entry->type_x2regs = true;
				if (gpu_type_catalog[i].type_flags & GPUTYPE_FLAGS_FP64)
					entry->type_fp64 = true;
				entry->type_varref = gpu_type_catalog[i].type_varref;
				entry->type_conref = gpu_type_catalog[i].type_conref;
				entry->type_explain = gpu_type_catalog[i].type_explain;
				break;
			}
		}
	}
	gpu_type_info_slot[hash] = lappend(gpu_type_info_slot[hash], entry);
	MemoryContextSwitchTo(oldcxt);
	ReleaseSysCache(tuple);

	if (entry->type_varref != 0 && entry->type_conref != 0)
		return entry;
	elog(INFO, "unsupported type: %u", entry->type_oid);
	return NULL;
}

static GpuTypeInfo *
pgstrom_gpucmd_type_lookup(int gpucmd)
{
	int		i;

	for (i=0; i < lengthof(gpu_type_catalog); i++)
	{
		if (gpu_type_catalog[i].type_varref == gpucmd ||
			gpu_type_catalog[i].type_conref == gpucmd)
			return pgstrom_gpu_type_lookup(gpu_type_catalog[i].type_oid);
	}
	elog(ERROR, "cache lookup failed for gpucmd: %d", gpucmd);

	return NULL; /* for compiler quiet */
}

/* ------------------------------------------------------------ *
 *
 * Catalog of GPU Functions
 *
 * ------------------------------------------------------------ */
static struct {
	int32		func_cmd;
	const char *func_name;
	int16		func_nargs;
	Oid			func_argtypes[4];
	const char *func_explain;
} gpu_func_catalog[] = {
	/*
	 * Type Cast Functions
	 */
	{ GPUCMD_CAST_INT2_TO_INT4,		"int4",		1, {INT2OID}, "c:" },
	{ GPUCMD_CAST_INT2_TO_INT8,		"int8",		1, {INT2OID}, "c:" },
	{ GPUCMD_CAST_INT2_TO_FLOAT4,	"float4",	1, {INT2OID}, "c:" },
	{ GPUCMD_CAST_INT2_TO_FLOAT8,	"float8",	1, {INT2OID}, "c:" },

	{ GPUCMD_CAST_INT4_TO_INT2,		"int2",		1, {INT4OID}, "c:" },
	{ GPUCMD_CAST_INT4_TO_INT8,		"int8",		1, {INT4OID}, "c:" },
	{ GPUCMD_CAST_INT4_TO_FLOAT4,	"float4",	1, {INT4OID}, "c:" },
	{ GPUCMD_CAST_INT4_TO_FLOAT8,	"float8",	1, {INT4OID}, "c:" },

	{ GPUCMD_CAST_INT8_TO_INT2,		"int2",		1, {INT8OID}, "c:" },
	{ GPUCMD_CAST_INT8_TO_INT4,		"int4",		1, {INT8OID}, "c:" },
	{ GPUCMD_CAST_INT8_TO_FLOAT4,	"float4",	1, {INT8OID}, "c:" },
	{ GPUCMD_CAST_INT8_TO_FLOAT8,	"float8",	1, {INT8OID}, "c:" },

	{ GPUCMD_CAST_FLOAT4_TO_INT2,	"int2",		1, {FLOAT4OID}, "c:" },
	{ GPUCMD_CAST_FLOAT4_TO_INT4,	"int4",		1, {FLOAT4OID}, "c:" },
	{ GPUCMD_CAST_FLOAT4_TO_INT8,	"int8",		1, {FLOAT4OID}, "c:" },
	{ GPUCMD_CAST_FLOAT4_TO_FLOAT8,	"float8",	1, {FLOAT4OID}, "c:" },

	{ GPUCMD_CAST_FLOAT8_TO_INT2,	"int2",		1, {FLOAT8OID}, "c:" },
	{ GPUCMD_CAST_FLOAT8_TO_INT4,	"int4",		1, {FLOAT8OID}, "c:" },
	{ GPUCMD_CAST_FLOAT8_TO_INT8,	"int8",		1, {FLOAT8OID}, "c:" },
	{ GPUCMD_CAST_FLOAT8_TO_FLOAT4,	"float4",	1, {FLOAT8OID}, "c:" },

	/* '+' : add operators */
	{ GPUCMD_OPER_INT2_PL,		"int2pl",	1, {INT2OID, INT2OID}, "b:+" },
	{ GPUCMD_OPER_INT24_PL,		"int24pl",	2, {INT2OID, INT4OID}, "b:+" },
	{ GPUCMD_OPER_INT28_PL,		"int28pl",	2, {INT2OID, INT8OID}, "b:+" },
	{ GPUCMD_OPER_INT42_PL,		"int42pl",	2, {INT4OID, INT2OID}, "b:+" },
	{ GPUCMD_OPER_INT4_PL,		"int4pl",	2, {INT4OID, INT4OID}, "b:+" },
	{ GPUCMD_OPER_INT48_PL,		"int48pl",	2, {INT4OID, INT8OID}, "b:+" },
	{ GPUCMD_OPER_INT82_PL,		"int82pl",	2, {INT8OID, INT2OID}, "b:+" },
	{ GPUCMD_OPER_INT84_PL,		"int84pl",	2, {INT8OID, INT4OID}, "b:+" },
	{ GPUCMD_OPER_INT8_PL,		"int8pl",	2, {INT8OID, INT8OID}, "b:+" },
	{ GPUCMD_OPER_FLOAT4_PL,	"float4pl",	2, {FLOAT4OID, FLOAT4OID}, "b:+" },
	{ GPUCMD_OPER_FLOAT48_PL,	"float48pl",2, {FLOAT4OID, FLOAT8OID}, "b:+" },
	{ GPUCMD_OPER_FLOAT84_PL,	"float84pl",2, {FLOAT8OID, FLOAT4OID}, "b:+" },
	{ GPUCMD_OPER_FLOAT8_PL,	"float8pl",	2, {FLOAT8OID, FLOAT8OID}, "b:+" },

	/* '-' : subtract operators */
	{ GPUCMD_OPER_INT2_MI,		"int2mi",	1, {INT2OID, INT2OID}, "b:-" },
	{ GPUCMD_OPER_INT24_MI,		"int24mi",	2, {INT2OID, INT4OID}, "b:-" },
	{ GPUCMD_OPER_INT28_MI,		"int28mi",	2, {INT2OID, INT8OID}, "b:-" },
	{ GPUCMD_OPER_INT42_MI,		"int42mi",	2, {INT4OID, INT2OID}, "b:-" },
	{ GPUCMD_OPER_INT4_MI,		"int4mi",	2, {INT4OID, INT4OID}, "b:-" },
	{ GPUCMD_OPER_INT48_MI,		"int48mi",	2, {INT4OID, INT8OID}, "b:-" },
	{ GPUCMD_OPER_INT82_MI,		"int82mi",	2, {INT8OID, INT2OID}, "b:-" },
	{ GPUCMD_OPER_INT84_MI,		"int84mi",	2, {INT8OID, INT4OID}, "b:-" },
	{ GPUCMD_OPER_INT8_MI,		"int8mi",	2, {INT8OID, INT8OID}, "b:-" },
	{ GPUCMD_OPER_FLOAT4_MI,	"float4mi",	2, {FLOAT4OID, FLOAT4OID}, "b:-" },
	{ GPUCMD_OPER_FLOAT48_MI,	"float48mi",2, {FLOAT4OID, FLOAT8OID}, "b:-" },
	{ GPUCMD_OPER_FLOAT84_MI,	"float84mi",2, {FLOAT8OID, FLOAT4OID}, "b:-" },
	{ GPUCMD_OPER_FLOAT8_MI,	"float8mi",	2, {FLOAT8OID, FLOAT8OID}, "b:-" },

	/* '*' : mutiply operators */
	{ GPUCMD_OPER_INT2_MUL,		"int2mul",	1, {INT2OID, INT2OID}, "b:*" },
	{ GPUCMD_OPER_INT24_MUL,	"int24mul",	2, {INT2OID, INT4OID}, "b:*" },
	{ GPUCMD_OPER_INT28_MUL,	"int28mul",	2, {INT2OID, INT8OID}, "b:*" },
	{ GPUCMD_OPER_INT42_MUL,	"int42mul",	2, {INT4OID, INT2OID}, "b:*" },
	{ GPUCMD_OPER_INT4_MUL,		"int4mul",	2, {INT4OID, INT4OID}, "b:*" },
	{ GPUCMD_OPER_INT48_MUL,	"int48mul",	2, {INT4OID, INT8OID}, "b:*" },
	{ GPUCMD_OPER_INT82_MUL,	"int82mul",	2, {INT8OID, INT2OID}, "b:*" },
	{ GPUCMD_OPER_INT84_MUL,	"int84mul",	2, {INT8OID, INT4OID}, "b:*" },
	{ GPUCMD_OPER_INT8_MUL,		"int8mul",	2, {INT8OID, INT8OID}, "b:*" },
	{ GPUCMD_OPER_FLOAT4_MUL,	"float4mul",2, {FLOAT4OID, FLOAT4OID}, "b:*" },
	{ GPUCMD_OPER_FLOAT48_MUL,	"float48mul",2,{FLOAT4OID, FLOAT8OID}, "b:*" },
	{ GPUCMD_OPER_FLOAT84_MUL,	"float84mul",2,{FLOAT8OID, FLOAT4OID}, "b:*" },
	{ GPUCMD_OPER_FLOAT8_MUL,	"float8mul",2, {FLOAT8OID, FLOAT8OID}, "b:*" },

	/* '/' : divide operators */
	{ GPUCMD_OPER_INT2_DIV,		"int2div",	1, {INT2OID, INT2OID}, "b:/" },
	{ GPUCMD_OPER_INT24_DIV,	"int24div",	2, {INT2OID, INT4OID}, "b:/" },
	{ GPUCMD_OPER_INT28_DIV,	"int28div",	2, {INT2OID, INT8OID}, "b:/" },
	{ GPUCMD_OPER_INT42_DIV,	"int42div",	2, {INT4OID, INT2OID}, "b:/" },
	{ GPUCMD_OPER_INT4_DIV,		"int4div",	2, {INT4OID, INT4OID}, "b:/" },
	{ GPUCMD_OPER_INT48_DIV,	"int48div",	2, {INT4OID, INT8OID}, "b:/" },
	{ GPUCMD_OPER_INT82_DIV,	"int82div",	2, {INT8OID, INT2OID}, "b:/" },
	{ GPUCMD_OPER_INT84_DIV,	"int84div",	2, {INT8OID, INT4OID}, "b:/" },
	{ GPUCMD_OPER_INT8_DIV,		"int8div",	2, {INT8OID, INT8OID}, "b:/" },
	{ GPUCMD_OPER_FLOAT4_DIV,	"float4div",2, {FLOAT4OID, FLOAT4OID}, "b:/" },
	{ GPUCMD_OPER_FLOAT48_DIV,	"float48div",2,{FLOAT4OID, FLOAT8OID}, "b:/" },
	{ GPUCMD_OPER_FLOAT84_DIV,	"float84div",2,{FLOAT8OID, FLOAT4OID}, "b:/" },
	{ GPUCMD_OPER_FLOAT8_DIV,	"float8div",2, {FLOAT8OID, FLOAT8OID}, "b:/" },

	/* '%' : reminder operators */
	{ GPUCMD_OPER_INT2_MOD,		"int2mod",	2, {INT2OID, INT2OID}, "b:%" },
	{ GPUCMD_OPER_INT4_MOD,		"int4mod",	2, {INT4OID, INT4OID}, "b:%" },
	{ GPUCMD_OPER_INT8_MOD,		"int8mod",	2, {INT8OID, INT8OID}, "b:%" },

	/* '+' : unary plus operators */
	{ GPUCMD_OPER_INT2_UP,		"int2up",	1, {INT2OID}, "l:+" },
	{ GPUCMD_OPER_INT4_UP,		"int4up",	1, {INT4OID}, "l:+" },
	{ GPUCMD_OPER_INT8_UP,		"int8up",	1, {INT8OID}, "l:+" },
	{ GPUCMD_OPER_FLOAT4_UP,	"float4up",	1, {FLOAT4OID}, "l:+" },
	{ GPUCMD_OPER_FLOAT8_UP,	"float8up",	1, {FLOAT8OID}, "l:+" },

	/* '-' : unary minus operators */
	{ GPUCMD_OPER_INT2_UM,		"int2um",	1, {INT2OID}, "l:-" },
	{ GPUCMD_OPER_INT4_UM,		"int4um",	1, {INT4OID}, "l:-" },
	{ GPUCMD_OPER_INT8_UM,		"int8um",	1, {INT8OID}, "l:-" },
	{ GPUCMD_OPER_FLOAT4_UM,	"float4um",	1, {FLOAT4OID}, "l:-" },
	{ GPUCMD_OPER_FLOAT8_UM,	"float8um",	1, {FLOAT8OID}, "l:-" },

	/* '@' : absolute value operators */
	{ GPUCMD_OPER_INT2_ABS,		"int2abs",	1, {INT2OID}, "f:abs" },
	{ GPUCMD_OPER_INT4_ABS,		"int4abs",	1, {INT4OID}, "f:abs" },
	{ GPUCMD_OPER_INT8_ABS,		"int8abs",	1, {INT8OID}, "f:abs" },
	{ GPUCMD_OPER_FLOAT4_ABS,	"float4abs",1, {FLOAT4OID}, "f:abs" },
	{ GPUCMD_OPER_FLOAT8_ABS,	"float8abs",1, {FLOAT8OID}, "f:abs" },

	/* '=' : equal operators */
	{ GPUCMD_OPER_INT2_EQ,		"int2eq",	2, {INT2OID,INT2OID}, "b:==" },
	{ GPUCMD_OPER_INT24_EQ,		"int24eq",	2, {INT2OID,INT4OID}, "b:==" },
	{ GPUCMD_OPER_INT28_EQ,		"int28eq",	2, {INT2OID,INT8OID}, "b:==" },
	{ GPUCMD_OPER_INT42_EQ,		"int42eq",	2, {INT4OID,INT2OID}, "b:==" },
	{ GPUCMD_OPER_INT4_EQ,		"int4eq",	2, {INT4OID,INT4OID}, "b:==" },
	{ GPUCMD_OPER_INT48_EQ,		"int48eq",	2, {INT4OID,INT8OID}, "b:==" },
	{ GPUCMD_OPER_INT82_EQ,		"int82eq",	2, {INT8OID,INT2OID}, "b:==" },
	{ GPUCMD_OPER_INT84_EQ,		"int84eq",	2, {INT8OID,INT4OID}, "b:==" },
	{ GPUCMD_OPER_INT8_EQ,		"int8eq",	2, {INT8OID,INT8OID}, "b:==" },
	{ GPUCMD_OPER_FLOAT4_EQ,	"float4eq",	2, {FLOAT4OID,FLOAT4OID}, "b:==" },
	{ GPUCMD_OPER_FLOAT48_EQ,	"float48eq",2, {FLOAT4OID,FLOAT8OID}, "b:==" },
	{ GPUCMD_OPER_FLOAT84_EQ,	"float84eq",2, {FLOAT8OID,FLOAT4OID}, "b:==" },
	{ GPUCMD_OPER_FLOAT8_EQ,	"float8eq",	2, {FLOAT8OID,FLOAT8OID}, "b:==" },

	/* '<>' : not equal operators */
	{ GPUCMD_OPER_INT2_NE,		"int2ne",	2, {INT2OID,INT2OID}, "b:!=" },
	{ GPUCMD_OPER_INT24_NE,		"int24ne",	2, {INT2OID,INT4OID}, "b:!=" },
	{ GPUCMD_OPER_INT28_NE,		"int28ne",	2, {INT2OID,INT8OID}, "b:!=" },
	{ GPUCMD_OPER_INT42_NE,		"int42ne",	2, {INT4OID,INT2OID}, "b:!=" },
	{ GPUCMD_OPER_INT4_NE,		"int4ne",	2, {INT4OID,INT4OID}, "b:!=" },
	{ GPUCMD_OPER_INT48_NE,		"int48ne",	2, {INT4OID,INT8OID}, "b:!=" },
	{ GPUCMD_OPER_INT82_NE,		"int82ne",	2, {INT8OID,INT2OID}, "b:!=" },
	{ GPUCMD_OPER_INT84_NE,		"int84ne",	2, {INT8OID,INT4OID}, "b:!=" },
	{ GPUCMD_OPER_INT8_NE,		"int8ne",	2, {INT8OID,INT8OID}, "b:!=" },
	{ GPUCMD_OPER_FLOAT4_NE,	"float4ne",	2, {FLOAT4OID,FLOAT4OID}, "b:!=" },
	{ GPUCMD_OPER_FLOAT48_NE,	"float48ne",2, {FLOAT4OID,FLOAT8OID}, "b:!=" },
	{ GPUCMD_OPER_FLOAT84_NE,	"float84ne",2, {FLOAT8OID,FLOAT4OID}, "b:!=" },
	{ GPUCMD_OPER_FLOAT8_NE,	"float8ne",	2, {FLOAT8OID,FLOAT8OID}, "b:!=" },

	/* '>' : equal operators */
	{ GPUCMD_OPER_INT2_GT,		"int2gt",	2, {INT2OID,INT2OID}, "b:>" },
	{ GPUCMD_OPER_INT24_GT,		"int24gt",	2, {INT2OID,INT4OID}, "b:>" },
	{ GPUCMD_OPER_INT28_GT,		"int28gt",	2, {INT2OID,INT8OID}, "b:>" },
	{ GPUCMD_OPER_INT42_GT,		"int42gt",	2, {INT4OID,INT2OID}, "b:>" },
	{ GPUCMD_OPER_INT4_GT,		"int4gt",	2, {INT4OID,INT4OID}, "b:>" },
	{ GPUCMD_OPER_INT48_GT,		"int48gt",	2, {INT4OID,INT8OID}, "b:>" },
	{ GPUCMD_OPER_INT82_GT,		"int82gt",	2, {INT8OID,INT2OID}, "b:>" },
	{ GPUCMD_OPER_INT84_GT,		"int84gt",	2, {INT8OID,INT4OID}, "b:>" },
	{ GPUCMD_OPER_INT8_GT,		"int8gt",	2, {INT8OID,INT8OID}, "b:>" },
	{ GPUCMD_OPER_FLOAT4_GT,	"float4gt",	2, {FLOAT4OID,FLOAT4OID}, "b:>" },
	{ GPUCMD_OPER_FLOAT48_GT,	"float48gt",2, {FLOAT4OID,FLOAT8OID}, "b:>" },
	{ GPUCMD_OPER_FLOAT84_GT,	"float84gt",2, {FLOAT8OID,FLOAT4OID}, "b:>" },
	{ GPUCMD_OPER_FLOAT8_GT,	"float8gt",	2, {FLOAT8OID,FLOAT8OID}, "b:>" },

	/* '<' : equal operators */
	{ GPUCMD_OPER_INT2_LT,		"int2lt",	2, {INT2OID,INT2OID}, "b:<" },
	{ GPUCMD_OPER_INT24_LT,		"int24lt",	2, {INT2OID,INT4OID}, "b:<" },
	{ GPUCMD_OPER_INT28_LT,		"int28lt",	2, {INT2OID,INT8OID}, "b:<" },
	{ GPUCMD_OPER_INT42_LT,		"int42lt",	2, {INT4OID,INT2OID}, "b:<" },
	{ GPUCMD_OPER_INT4_LT,		"int4lt",	2, {INT4OID,INT4OID}, "b:<" },
	{ GPUCMD_OPER_INT48_LT,		"int48lt",	2, {INT4OID,INT8OID}, "b:<" },
	{ GPUCMD_OPER_INT82_LT,		"int82lt",	2, {INT8OID,INT2OID}, "b:<" },
	{ GPUCMD_OPER_INT84_LT,		"int84lt",	2, {INT8OID,INT4OID}, "b:<" },
	{ GPUCMD_OPER_INT8_LT,		"int8lt",	2, {INT8OID,INT8OID}, "b:<" },
	{ GPUCMD_OPER_FLOAT4_LT,	"float4lt",	2, {FLOAT4OID,FLOAT4OID}, "b:<" },
	{ GPUCMD_OPER_FLOAT48_LT,	"float48lt",2, {FLOAT4OID,FLOAT8OID}, "b:<" },
	{ GPUCMD_OPER_FLOAT84_LT,	"float84lt",2, {FLOAT8OID,FLOAT4OID}, "b:<" },
	{ GPUCMD_OPER_FLOAT8_LT,	"float8lt",	2, {FLOAT8OID,FLOAT8OID}, "b:<" },

	/* '>=' : relational greater-than or equal-to */
	{ GPUCMD_OPER_INT2_GE,		"int2ge",	2, {INT2OID,INT2OID}, "b:>=" },
	{ GPUCMD_OPER_INT24_GE,		"int24ge",	2, {INT2OID,INT4OID}, "b:>=" },
	{ GPUCMD_OPER_INT28_GE,		"int28ge",	2, {INT2OID,INT8OID}, "b:>=" },
	{ GPUCMD_OPER_INT42_GE,		"int42ge",	2, {INT4OID,INT2OID}, "b:>=" },
	{ GPUCMD_OPER_INT4_GE,		"int4ge",	2, {INT4OID,INT4OID}, "b:>=" },
	{ GPUCMD_OPER_INT48_GE,		"int48ge",	2, {INT4OID,INT8OID}, "b:>=" },
	{ GPUCMD_OPER_INT82_GE,		"int82ge",	2, {INT8OID,INT2OID}, "b:>=" },
	{ GPUCMD_OPER_INT84_GE,		"int84ge",	2, {INT8OID,INT4OID}, "b:>=" },
	{ GPUCMD_OPER_INT8_GE,		"int8ge",	2, {INT8OID,INT8OID}, "b:>=" },
	{ GPUCMD_OPER_FLOAT4_GE,	"float4ge",	2, {FLOAT4OID,FLOAT4OID}, "b:>=" },
	{ GPUCMD_OPER_FLOAT48_GE,	"float48ge",2, {FLOAT4OID,FLOAT8OID}, "b:>=" },
	{ GPUCMD_OPER_FLOAT84_GE,	"float84ge",2, {FLOAT8OID,FLOAT4OID}, "b:>=" },
	{ GPUCMD_OPER_FLOAT8_GE,	"float8ge",	2, {FLOAT8OID,FLOAT8OID}, "b:>=" },

	/* '<=' : relational greater-than or equal-to */
	{ GPUCMD_OPER_INT2_LE,		"int2le",	2, {INT2OID,INT2OID}, "b:<=" },
	{ GPUCMD_OPER_INT24_LE,		"int24le",	2, {INT2OID,INT4OID}, "b:<=" },
	{ GPUCMD_OPER_INT28_LE,		"int28le",	2, {INT2OID,INT8OID}, "b:<=" },
	{ GPUCMD_OPER_INT42_LE,		"int42le",	2, {INT4OID,INT2OID}, "b:<=" },
	{ GPUCMD_OPER_INT4_LE,		"int4le",	2, {INT4OID,INT4OID}, "b:<=" },
	{ GPUCMD_OPER_INT48_LE,		"int48le",	2, {INT4OID,INT8OID}, "b:<=" },
	{ GPUCMD_OPER_INT82_LE,		"int82le",	2, {INT8OID,INT2OID}, "b:<=" },
	{ GPUCMD_OPER_INT84_LE,		"int84le",	2, {INT8OID,INT4OID}, "b:<=" },
	{ GPUCMD_OPER_INT8_LE,		"int8le",	2, {INT8OID,INT8OID}, "b:<=" },
	{ GPUCMD_OPER_FLOAT4_LE,	"float4le",	2, {FLOAT4OID,FLOAT4OID}, "b:<=" },
	{ GPUCMD_OPER_FLOAT48_LE,	"float48le",2, {FLOAT4OID,FLOAT8OID}, "b:<=" },
	{ GPUCMD_OPER_FLOAT84_LE,	"float84le",2, {FLOAT8OID,FLOAT4OID}, "b:<=" },
	{ GPUCMD_OPER_FLOAT8_LE,	"float8le",	2, {FLOAT8OID,FLOAT8OID}, "b:<=" },

	/* '&' : bitwise and */
	{ GPUCMD_OPER_INT2_AND,		"int2and",	2, {INT2OID,INT2OID}, "b:&" },
	{ GPUCMD_OPER_INT4_AND,		"int4and",	2, {INT4OID,INT4OID}, "b:&" },
	{ GPUCMD_OPER_INT8_AND,		"int8and",	2, {INT8OID,INT8OID}, "b:&" },

	/* '|'  : bitwise or */
	{ GPUCMD_OPER_INT2_OR,		"int2or",	2, {INT2OID,INT2OID}, "b:|" },
	{ GPUCMD_OPER_INT4_OR,		"int4or",	2, {INT4OID,INT4OID}, "b:|" },
	{ GPUCMD_OPER_INT8_OR,		"int8or",	2, {INT8OID,INT8OID}, "b:|" },

	/* '#'  : bitwise xor */
	{ GPUCMD_OPER_INT2_XOR,		"int2xor",	2, {INT2OID,INT2OID}, "b:^" },
	{ GPUCMD_OPER_INT4_XOR,		"int4xor",	2, {INT4OID,INT4OID}, "b:^" },
	{ GPUCMD_OPER_INT8_XOR,		"int8xor",	2, {INT8OID,INT8OID}, "b:^" },

	/* '~'  : bitwise not operators */
	{ GPUCMD_OPER_INT2_NOT,		"int2not",	1, {INT2OID}, "l:!" },
	{ GPUCMD_OPER_INT4_NOT,		"int4not",	1, {INT4OID}, "l:!" },
	{ GPUCMD_OPER_INT8_NOT,		"int8not",	1, {INT8OID}, "l:!" },

	/* '>>' : right shift */
	{ GPUCMD_OPER_INT2_SHR,		"int2shr",	2, {INT2OID,INT4OID}, "b:>>" },
	{ GPUCMD_OPER_INT4_SHR,		"int4shr",	2, {INT4OID,INT4OID}, "b:>>" },
	{ GPUCMD_OPER_INT8_SHR,		"int8shr",	2, {INT8OID,INT4OID}, "b:>>" },

	/* '<<' : left shift */
	{ GPUCMD_OPER_INT2_SHL,		"int2shl",	2, {INT2OID,INT4OID}, "b:<<" },
	{ GPUCMD_OPER_INT4_SHL,		"int4shl",	2, {INT4OID,INT4OID}, "b:<<" },
	{ GPUCMD_OPER_INT8_SHL,		"int8shl",	2, {INT8OID,INT4OID}, "b:<<" },

	/*
	 * Mathmatical functions
	 */
	{ GPUCMD_FUNC_FLOAT8_CBRT,	"cbrt",		1, {FLOAT8OID}, "f:cbrt" },
	{ GPUCMD_FUNC_FLOAT8_CEIL,	"ceil",		1, {FLOAT8OID}, "f:ceil" },
	{ GPUCMD_FUNC_FLOAT8_EXP,	"exp",		1, {FLOAT8OID}, "f:exp" },
	{ GPUCMD_FUNC_FLOAT8_FLOOR,	"floor",	1, {FLOAT8OID}, "f:floor" },
	{ GPUCMD_FUNC_FLOAT8_LOG,	"ln",		1, {FLOAT8OID}, "f:log" },
	{ GPUCMD_FUNC_FLOAT8_LOG10,	"log",		1, {FLOAT8OID}, "f:log10" },
	{ GPUCMD_FUNC_FLOAT8_PI,	"pi",		0, {}, "f:pi" },
	{ GPUCMD_FUNC_FLOAT8_POWER,	"power", 2, {FLOAT8OID,FLOAT8OID}, "f:pow" },
	{ GPUCMD_FUNC_FLOAT8_POWER,	"pow",	 2, {FLOAT8OID,FLOAT8OID}, "f:pow" },
	{ GPUCMD_FUNC_FLOAT8_POWER,	"dpow",	 2, {FLOAT8OID,FLOAT8OID}, "f:pow" },
	{ GPUCMD_FUNC_FLOAT8_ROUND,	"round",	1, {FLOAT8OID}, "f:round" },
	{ GPUCMD_FUNC_FLOAT8_SIGN,	"sign",		1, {FLOAT8OID}, "f:sign" },
	{ GPUCMD_FUNC_FLOAT8_SQRT,	"sqrt",		1, {FLOAT8OID}, "f:sqrt" },
	{ GPUCMD_FUNC_FLOAT8_SQRT,	"dsqrt",	1, {FLOAT8OID}, "f:sqrt" },
	{ GPUCMD_FUNC_FLOAT8_TRUNC,	"trunc",	1, {FLOAT8OID}, "f:trunc" },
	{ GPUCMD_FUNC_FLOAT8_TRUNC,	"dtrunc",	1, {FLOAT8OID}, "f:trunc" },

	/*
     * Trigonometric function
     */
	{ GPUCMD_FUNC_FLOAT8_ACOS,	"acos",		1, {FLOAT8OID}, "f:acos" },
	{ GPUCMD_FUNC_FLOAT8_ASIN,	"asin",		1, {FLOAT8OID}, "f:asin" },
	{ GPUCMD_FUNC_FLOAT8_ATAN,	"atan",		1, {FLOAT8OID}, "f:atan" },
	{ GPUCMD_FUNC_FLOAT8_ATAN2,	"atan2", 1, {FLOAT8OID,FLOAT8OID}, "f:atan2" },
	{ GPUCMD_FUNC_FLOAT8_COS,	"cos",		1, {FLOAT8OID}, "f:cos" },
	{ GPUCMD_FUNC_FLOAT8_COT,  "cot_double",1, {FLOAT8OID}, "f:cot" },
	{ GPUCMD_FUNC_FLOAT8_SIN,	"sin",		1, {FLOAT8OID}, "f:sin" },
	{ GPUCMD_FUNC_FLOAT8_TAN,	"tan",		1, {FLOAT8OID}, "f:tan" },
};

GpuFuncInfo *
pgstrom_gpu_func_lookup(Oid func_oid)
{
	GpuFuncInfo	   *entry;
	HeapTuple		tuple;
	Form_pg_proc	procform;
	MemoryContext	oldcxt;
	ListCell	   *cell;
	int				i, hash;

	hash = hash_uint32((uint32) func_oid) % lengthof(gpu_func_info_slot);
	foreach (cell, gpu_func_info_slot[hash])
	{
		entry = lfirst(cell);
		if (entry->func_oid == func_oid)
		{
			/* supported function has func_cmd */
			if (entry->func_cmd != 0)
				return entry;
			return NULL;
		}
	}

	/*
	 * Not found, insert a new entry
	 */
	tuple = SearchSysCache1(PROCOID, ObjectIdGetDatum(func_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for function %u", func_oid);
	procform = (Form_pg_proc) GETSTRUCT(tuple);

	oldcxt = MemoryContextSwitchTo(CacheMemoryContext);

	entry = palloc0(sizeof(GpuFuncInfo) +
					sizeof(Oid) * procform->pronargs);
	entry->func_oid = func_oid;

	if (procform->pronamespace != PG_CATALOG_NAMESPACE)
		goto out;
	if (!pgstrom_gpu_type_lookup(procform->prorettype))
		goto out;
	for (i=0; i < procform->pronargs; i++)
	{
		if (!pgstrom_gpu_type_lookup(procform->proargtypes.values[i]))
			goto out;
	}

	for (i=0; i < lengthof(gpu_func_catalog); i++)
	{
		if (strcmp(NameStr(procform->proname),
				   gpu_func_catalog[i].func_name) == 0 &&
			procform->pronargs == gpu_func_catalog[i].func_nargs &&
			memcmp(procform->proargtypes.values,
				   gpu_func_catalog[i].func_argtypes,
				   sizeof(Oid) * procform->pronargs) == 0)
		{
			entry->func_cmd = gpu_func_catalog[i].func_cmd;
			entry->func_explain = gpu_func_catalog[i].func_explain;
			entry->func_nargs = procform->pronargs;
			entry->func_rettype = procform->prorettype;
			memcpy(entry->func_argtypes,
				   procform->proargtypes.values,
				   sizeof(Oid) * procform->pronargs);
			break;
		}
	}
out:
	gpu_func_info_slot[hash] = lappend(gpu_func_info_slot[hash], entry);
	MemoryContextSwitchTo(oldcxt);
	ReleaseSysCache(tuple);

	if (entry->func_cmd != 0)
		return entry;

	elog(DEBUG1, "PG-Strom: %s is not supported function",
		 format_procedure(entry->func_oid));
	return NULL;
}

static GpuFuncInfo *
pgstrom_gpucmd_func_lookup(int gpucmd)
{
	int		i;

	for (i=0; i < lengthof(gpu_func_catalog); i++)
	{
		const char *func_name;
		oidvector  *func_args;
		Oid			func_oid;

		if (gpu_func_catalog[i].func_cmd != gpucmd)
			continue;

		func_name = gpu_func_catalog[i].func_name;
		func_args = buildoidvector(gpu_func_catalog[i].func_argtypes,
								   gpu_func_catalog[i].func_nargs);
		func_oid = GetSysCacheOid3(PROCNAMEARGSNSP,
								   CStringGetDatum(func_name),
								   PointerGetDatum(func_args),
								   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
		pfree(func_args);

		return pgstrom_gpu_func_lookup(func_oid);
	}
	elog(ERROR, "catalog lookup failed for gpucmd: %d", gpucmd);

	return NULL;	/* for compiler quiet */
}

/*
 * pgstrom_gpu_command_string
 *
 * It returns text representation of the supplied GPU command series
 */
int
pgstrom_gpu_command_string(Oid ftableOid, int cmds[],
						   char *buf, size_t buflen)
{
	GpuFuncInfo	*gfunc;
	GpuTypeInfo	*gtype[5];
	int			 i;

	switch (cmds[0])
	{
		/*
		 * End of commands
		 */
		case GPUCMD_TERMINAL_COMMAND:
			snprintf(buf, buflen, "end;");
			return -1;

		/*
		 * Constraint reference
		 */
		case GPUCMD_CONREF_NULL:
			snprintf(buf, buflen, "reg%d = null;", cmds[1]);
			return 2;
		case GPUCMD_CONREF_BOOL:
			snprintf(buf, buflen, "reg%d = %u::uchar", cmds[1], cmds[2]);
			return 3;
		case GPUCMD_CONREF_INT2:
			snprintf(buf, buflen, "reg%d = %u::short", cmds[1], cmds[2]);
			return 3;
		case GPUCMD_CONREF_INT4:
			snprintf(buf, buflen, "reg%d = %u::int", cmds[1], cmds[2]);
			return 3;
		case GPUCMD_CONREF_INT8:
			snprintf(buf, buflen, "xreg%d = %lu::long",
					 cmds[1], *((int64 *)&cmds[2]));
			return 4;
		case GPUCMD_CONREF_FLOAT4:
			snprintf(buf, buflen, "reg%d = %f::float",
					 cmds[1], *((float *)&cmds[2]));
			return 3;
		case GPUCMD_CONREF_FLOAT8:
			snprintf(buf, buflen, "xreg%d = %f::double",
					 cmds[1], *((double *)&cmds[2]));
			return 4;
		/*
		 * Variable References
		 */
		case GPUCMD_VARREF_BOOL:
		case GPUCMD_VARREF_INT2:
		case GPUCMD_VARREF_INT4:
		case GPUCMD_VARREF_INT8:
		case GPUCMD_VARREF_FLOAT4:
		case GPUCMD_VARREF_FLOAT8:
			gtype[0] = pgstrom_gpucmd_type_lookup(cmds[0]);
			snprintf(buf, buflen, "%sreg%d = $(%s.%s)",
					 gtype[0]->type_x2regs ? "x" : "", cmds[1],
					 get_rel_name(ftableOid),
					 get_attname(ftableOid, cmds[2] + 1));
			return 3;
		/*
		 * Boolean operations
		 */
		case GPUCMD_BOOLOP_AND:
			snprintf(buf, buflen, "reg%d = reg%d & reg%d",
					 cmds[1], cmds[1], cmds[2]);
			return 3;

		case GPUCMD_BOOLOP_OR:
			snprintf(buf, buflen, "reg%d = reg%d | reg%d",
					 cmds[1], cmds[1], cmds[2]);
			return 3;

		case GPUCMD_BOOLOP_NOT:
			snprintf(buf, buflen, "reg%d = ! reg%d", cmds[1], cmds[1]);
			return 3;

		/*
		 * Functions or Operators
		 */
		default:
			gfunc = pgstrom_gpucmd_func_lookup(cmds[0]);
			gtype[0] = pgstrom_gpu_type_lookup(gfunc->func_rettype);
			for (i=0; i < gfunc->func_nargs; i++)
				gtype[i+1] = pgstrom_gpu_type_lookup(gfunc->func_argtypes[i]);

			if (strncmp(gfunc->func_explain, "c:", 2) == 0)
			{
				Assert(gfunc->func_nargs == 1);
				snprintf(buf, buflen, "%sreg%d = (%s)%sreg%d",
						 gtype[0]->type_x2regs ? "x" : "", cmds[1],
						 gtype[0]->type_explain,
						 gtype[1]->type_x2regs ? "x" : "", cmds[2]);
			}
			else if (strncmp(gfunc->func_explain, "l:", 2) == 0)
			{
				Assert(gfunc->func_nargs == 1);
				snprintf(buf, buflen, "%sreg%d = %s(%sreg%d)",
						 gtype[0]->type_x2regs ? "x" : "", cmds[1],
						 gfunc->func_explain + 2,
						 gtype[1]->type_x2regs ? "x" : "", cmds[2]);
			}
			else if (strncmp(gfunc->func_explain, "r:", 2) == 0)
			{
				Assert(gfunc->func_nargs == 1);
				snprintf(buf, buflen, "%sreg%d = (%sreg%d)%s",
						 gtype[0]->type_x2regs ? "x" : "", cmds[1],
						 gtype[1]->type_x2regs ? "x" : "", cmds[2],
						 gfunc->func_explain + 2);
			}
			else if (strncmp(gfunc->func_explain, "b:", 2) == 0)
			{
				Assert(gfunc->func_nargs == 2);
				snprintf(buf, buflen, "%sreg%d =  (%sreg%d %s %sreg%d)",
						 gtype[0]->type_x2regs ? "x" : "", cmds[1],
						 gtype[1]->type_x2regs ? "x" : "", cmds[2],
						 gfunc->func_explain + 2,
						 gtype[2]->type_x2regs ? "x" : "", cmds[3]);
			}
			else if (strncmp(gfunc->func_explain, "f:", 2) == 0)
			{
				size_t	ofs;

				ofs = snprintf(buf, buflen, "%sreg%d = %s(",
							   gtype[0]->type_x2regs ? "x" : "", cmds[1],
							   gfunc->func_explain + 2);
				for (i=0; i < gfunc->func_nargs; i++)
					ofs += snprintf(buf + ofs, buflen - ofs, "%s%sreg%d",
									(i > 0) ? ", " : "",
									gtype[i+1]->type_x2regs ? "x" : "",
									cmds[i+2]);
				snprintf(buf + ofs, buflen - ofs, ")");
			}
			else
				elog(ERROR, "unexpected device function identifier: %s",
					 gfunc->func_explain);

			return gfunc->func_nargs + 2;
	}
}
