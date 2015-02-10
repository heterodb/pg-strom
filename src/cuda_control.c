/*
 * cuda_control.c
 *
 * Overall logic to control cuda context and devices.
 * ----
 * Copyright 2011-2015 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2015 (C) The PG-Strom Development Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */
#include "postgres.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "lib/ilist.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/pg_crc.h"
#include "utils/resowner.h"
#include "pg_strom.h"

/* available devices set by postmaster startup */
static List		   *cuda_device_ordinals = NIL;
static size_t		cuda_max_malloc_size = INT_MAX;
static size_t		cuda_max_threads_per_block = INT_MAX;
static int			cuda_compute_capability = INT_MAX;

/* stuffs related to GpuContext */
static dlist_head	gcontext_hash[100];
static GpuContext  *gcontext_last = NULL;

/* CUDA runtime stuff per backend process */
static int			cuda_num_devices = -1;
static CUdevice	   *cuda_devices = NULL;

/*
 * pgstrom_cuda_init
 *
 * initialize CUDA runtime per backend process.
 */
static void
pgstrom_init_cuda(void)
{
	CUdevice	device;
	CUresult	rc;
	ListCell   *cell;
	int			i = 0;

	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuInit: %s", errorText(rc));

	cuda_num_devices = list_length(cuda_device_ordinals);
	cuda_devices = MemoryContextAllocZero(TopMemoryContext,
										  sizeof(CUdevice) * cuda_num_devices);
	foreach (cell, cuda_device_ordinals)
	{
		int		ordinal = lfirst_int(cell);

		rc = cuDeviceGet(&device, ordinal);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGet: %s", errorText(rc));
		cuda_devices[i++] = device;
	}
}

static inline int
gpucontext_hash_index(ResourceOwner resowner)
{
	pg_crc32	crc;

	INIT_CRC32C(crc);
	COMP_CRC32C(crc, resowner, sizeof(ResourceOwner));
	FIN_CRC32C(crc);

	return crc % lengthof(gcontext_hash);
}

static GpuContext *
pgstrom_create_gpucontext(ResourceOwner resowner)
{
	GpuContext	   *gcontext = NULL;
	MemoryContext	memcxt = NULL;
	CUcontext		cuda_context;
	Size			length_gcxt;
	Size			length_init;
	Size			length_max;
	char			namebuf[200];
	int				index;
	CUresult		rc;

	if (cuda_num_devices < 0)
		pgstrom_init_cuda();
	if (cuda_num_devices < 1)
		elog(ERROR, "No cuda device were detected");

	/* make a first CUcontext */
	rc = cuCtxCreate(&cuda_context, CU_CTX_SCHED_AUTO, cuda_devices[0]);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxCreate: %s", errorText(rc));

	PG_TRY();
	{
		/* make a new memory context */
		snprintf(namebuf, sizeof(namebuf), "GPU DMA Buffer (%p)", resowner);
		length_init = 4 * (1UL << get_next_log2(pgstrom_chunk_size()));
		length_max = 1024 * length_init;
		memcxt = HostPinMemContextCreate(NULL,
										 namebuf,
										 cuda_context,
										 0,		/* no pre-allocation */
										 length_init,
										 length_max);
		length_gcxt = offsetof(GpuContext, cuda_context[cuda_num_devices]);
		gcontext = MemoryContextAllocZero(memcxt, length_gcxt);
		gcontext->refcnt = 2;
		gcontext->resowner = resowner;
		gcontext->memcxt = memcxt;
		dlist_init(&gcontext->state_list);
		dlist_init(&gcontext->pds_list);
		gcontext->cuda_context[0] = cuda_context;
		for (index=1; index < cuda_num_devices; index++)
		{
			rc = cuCtxCreate(&gcontext->cuda_context[index],
							 CU_CTX_SCHED_AUTO,
							 cuda_devices[index]);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuCtxCreate: %s", errorText(rc));
		}
		gcontext->num_context = cuda_num_devices;
		gcontext->cur_context = 0;
		elog(INFO, "create a new gpucontext=%p", gcontext);
	}
	PG_CATCH();
	{
		if (!gcontext)
		{
			rc = cuCtxDestroy(cuda_context);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuCtxDestroy: %s", errorText(rc));
		}
		else
		{
			while (index > 0)
			{
				rc = cuCtxDestroy(gcontext->cuda_context[--index]);
				if (rc != CUDA_SUCCESS)
					elog(WARNING, "failed on cuCtxDestroy: %s", errorText(rc));
			}
		}
		if (memcxt != NULL)
			MemoryContextDelete(memcxt);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return gcontext;
}

GpuContext *
pgstrom_get_gpucontext(void)
{
	GpuContext *gcontext;
	dlist_iter	iter;
	int			hindex;

	if (gcontext_last != NULL &&
		gcontext_last->resowner == CurrentResourceOwner)
	{
		gcontext = gcontext_last;
		gcontext->refcnt++;
		elog(INFO, "get gcontext 1 = %p", gcontext);
		return gcontext;
	}
	/* not a last one, so search a hash table */
	hindex = gpucontext_hash_index(CurrentResourceOwner);
	dlist_foreach (iter, &gcontext_hash[hindex])
	{
		gcontext = dlist_container(GpuContext, chain, iter.cur);

		if (gcontext->resowner == CurrentResourceOwner)
		{
			gcontext->refcnt++;
			elog(INFO, "get gcontext 2 = %p", gcontext);
			return gcontext;
		}
	}
	/*
	 * Hmm... no gpu context is not attached this resource owner,
	 * so create a new one.
	 */
	gcontext = pgstrom_create_gpucontext(CurrentResourceOwner);
	dlist_push_tail(&gcontext_hash[hindex], &gcontext->chain);

	return gcontext;
}

void
pgstrom_sync_gpucontext(GpuContext *gcontext)
{
	CUresult	rc;
	int			i;

	/* Ensure all the concurrent tasks getting completed */
	for (i=0; i < gcontext->num_context; i++)
	{
		rc = cuCtxSetCurrent(gcontext->cuda_context[i]);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuCtxSetCurrent: %s", errorText(rc));

		rc = cuCtxSynchronize();
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuCtxSynchronize: %s", errorText(rc));
	}
}

static void
pgstrom_release_gpucontext(GpuContext *gcontext, bool is_commit)
{
	CUcontext	cuda_context;
	CUresult	rc;
	int			i;

	/* Ensure all the concurrent tasks getting completed */
	pgstrom_sync_gpucontext(gcontext);

	/* Release underlying TaskState, if any */
	while (!dlist_is_empty(&gcontext->state_list))
	{
		dlist_node	   *dnode = dlist_pop_head_node(&gcontext->state_list);
		GpuTaskState   *gts = dlist_container(GpuTaskState, chain, dnode);

		Assert(gts->gcontext == gcontext);

		/* release cuda module, if any */
		if (gts->cuda_module)
		{
			rc = cuModuleUnload(gts->cuda_module);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuModuleUnload: %s", errorText(rc));
			gts->cuda_module = NULL;
		}
		/* release task objects */
		while (!dlist_is_empty(&gts->tracked_tasks))
		{
			dlist_node *dnode = dlist_pop_head_node(&gts->tracked_tasks);
			GpuTask	   *task = dlist_container(GpuTask, tracker, dnode);

			Assert(task->gts == gts);
			if (is_commit)
				elog(WARNING, "Unreferenced GpuTask leak: %p", task);
			task->cb_release(task);
		}
		/* release task state */
		if (is_commit)
			elog(WARNING, "Unreferenced GpuTaskState leak: %p", gts);
		if (gts->cb_cleanup)
			gts->cb_cleanup(gts);
	}

	/*
	 * Release pgstrom_data_store; because KDS_FORMAT_ROW may have mmap(2)
	 * state in case of file-mapped data-store, so we have to ensure
	 * these temporary files are removed and unmapped.
	 */
	while (!dlist_is_empty(&gcontext->pds_list))
	{
		dlist_node *dnode = dlist_pop_head_node(&gcontext->pds_list);
		pgstrom_data_store *pds
			= dlist_container(pgstrom_data_store, chain, dnode);
		if (pds->kds_fname)
			pgstrom_file_unmap_data_store(pds);
	}

	/* Ensure CUDA context is empty */
	rc = cuCtxSetCurrent(NULL);
	if (rc != CUDA_SUCCESS)
		elog(WARNING, "failed on cuCtxSetCurrent(NULL): %s", errorText(rc));

	/*
	 * NOTE: Be careful to drop the primary CUDA context because it also
	 * removes/unmaps all the memory region allocated by cuMemHostAlloc()
	 * that includes the GpuContext object and MemoryContext.
	 * So, we have to drop non-primary CUDA context, memory context, then
	 * the primary CUDA context.
	 */
	cuda_context = gcontext->cuda_context[0];
	for (i = gcontext->num_context - 1; i > 0; i--)
	{
		rc = cuCtxDestroy(gcontext->cuda_context[i]);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuCtxDestroy: %s", errorText(rc));
	}
	/* release  */
	MemoryContextDelete(gcontext->memcxt);

	/* Drop the primary CUDA context */
	rc = cuCtxDestroy(cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(WARNING, "failed on cuCtxDestroy: %s", errorText(rc));
}

void
pgstrom_put_gpucontext(GpuContext *gcontext)
{
	Assert(gcontext->refcnt > 0);
	if (--gcontext->refcnt == 0)
	{
		if (gcontext_last == gcontext)
			gcontext_last = NULL;
		dlist_delete(&gcontext->chain);
		pgstrom_release_gpucontext(gcontext, true);
	}
}

/*
 * pgstrom_cleanup_gputaskstate
 *
 * cleanup any active tasks and release cuda related resource, to end or
 * rescan executor.
 */
void
pgstrom_cleanup_gputaskstate(GpuTaskState *gts)
{
	dlist_mutable_iter iter;
	GpuTask		   *gputask;

	/* synchronize all the concurrent task, if any */
	pgstrom_sync_gpucontext(gts->gcontext);

	SpinLockAcquire(&gts->lock);
	gputask = gts->curr_task;
	if (gputask)
	{
		dlist_delete(&gputask->tracker);
		memset(&gputask->tracker, 0, sizeof(dlist_node));
		gts->curr_task = NULL;
		SpinLockRelease(&gts->lock);
		gputask->cb_release(gputask);
		SpinLockAcquire(&gts->lock);
	}
	dlist_foreach_modify(iter, &gts->tracked_tasks)
	{
		gputask = dlist_container(GpuTask, tracker, iter.cur);

		dlist_delete(&gputask->tracker);
		SpinLockRelease(&gts->lock);
		gputask->cb_release(gputask);
		SpinLockAcquire(&gts->lock);
	}
	dlist_init(&gts->running_tasks);
	dlist_init(&gts->pending_tasks);
	dlist_init(&gts->completed_tasks);
	gts->num_running_tasks = 0;
    gts->num_pending_tasks = 0;
	gts->num_completed_tasks = 0;
	SpinLockRelease(&gts->lock);
}

void
pgstrom_release_gputaskstate(GpuTaskState *gts)
{
	CUresult	rc;

	/* unlink this GpuTaskState from the GpuContext */
	dlist_delete(&gts->chain);

	/* clean-up and release any concurrent tasks */
	pgstrom_cleanup_gputaskstate(gts);

	/* release cuda module, if any */
	if (gts->cuda_module)
	{
		rc = cuModuleUnload(gts->cuda_module);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuModuleUnload: %s", errorText(rc));
		gts->cuda_module = NULL;
	}
	/* put reference to the GpuContext */
	pgstrom_put_gpucontext(gts->gcontext);
}

void
pgstrom_init_gputaststate(GpuContext *gcontext, GpuTaskState *gts,
						  void (*cb_cleanup)(GpuTaskState *gts))
{
	dlist_push_tail(&gcontext->state_list, &gts->chain);
	gts->gcontext = gcontext;
	gts->kern_source = NULL;	/* to be set later */
	gts->extra_flags = 0;		/* to be set later */
	SpinLockInit(&gts->lock);
	dlist_init(&gts->tracked_tasks);
	dlist_init(&gts->running_tasks);
	dlist_init(&gts->pending_tasks);
	dlist_init(&gts->completed_tasks);
	gts->num_running_tasks = 0;
	gts->num_pending_tasks = 0;
	gts->num_completed_tasks = 0;
	gts->cb_cleanup = cb_cleanup;
	memset(&gts->pfm_accum, 0, sizeof(pgstrom_perfmon));
}

void
pgstrom_init_gputask(GpuTaskState *gts, GpuTask *task,
					 bool (*cb_process)(GpuTask *task),
					 void (*cb_release)(GpuTask *task))
{
	GpuContext *gcontext = gts->gcontext;
	CUdevice	cuda_device;
	CUcontext	cuda_context;
	CUstream	cuda_stream;
	CUresult	rc;
	int			index;

	index = (gcontext->cur_context++ % gcontext->num_context);
	cuda_context = gcontext->cuda_context[index];
	rc = cuCtxGetDevice(&cuda_device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxGetDevice: %s", errorText(rc));

	rc = cuCtxSetCurrent(cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));

	rc = cuStreamCreate(&cuda_stream, CU_STREAM_NON_BLOCKING);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuStreamCreate: %s", errorText(rc));

	memset(task, 0, sizeof(GpuTask));
	task->gts = gts;
	task->cuda_stream = cuda_stream;
	task->cuda_device = cuda_device;
	task->cuda_context = cuda_context;
	task->cb_process = cb_process;
	task->cb_release = cb_release;
	/* tracked by GpuTaskState */
	SpinLockAcquire(&gts->lock);
	dlist_push_tail(&gts->tracked_tasks, &task->tracker);
	SpinLockRelease(&gts->lock);
}

static void
gpucontext_cleanup_callback(ResourceReleasePhase phase,
							bool is_commit,
							bool is_toplevel,
							void *arg)
{
	dlist_mutable_iter iter;
	int			hindex = gpucontext_hash_index(CurrentResourceOwner);

	if (phase != RESOURCE_RELEASE_AFTER_LOCKS)
		return;

	dlist_foreach_modify(iter, &gcontext_hash[hindex])
	{
		GpuContext *gcontext = dlist_container(GpuContext, chain, iter.cur);

		if (gcontext->resowner == CurrentResourceOwner)
		{
			/* OK, GpuContext to be released */
			if (gcontext_last == gcontext)
				gcontext_last = NULL;
			dlist_delete(&gcontext->chain);

			if (is_commit && gcontext->refcnt > 1)
				elog(WARNING, "Probably, someone forgot to put GpuContext");

			pgstrom_release_gpucontext(gcontext, is_commit);
			return;
		}
	}
}





/*
 *
 *
 */
static __thread int __dynamic_shmem_per_thread;
static size_t
dynamic_shmem_size_per_block(int blockSize)
{
	return __dynamic_shmem_per_thread * (size_t)blockSize;
}

void
pgstrom_compute_workgroup_size(size_t *p_grid_size,
							   size_t *p_block_size,
							   CUfunction function,
							   CUdevice device,
							   bool maximum_blocksize,
							   size_t nitems,
							   size_t dynamic_shmem_per_thread)
{
	int			grid_size;
	int			block_size;
	int			maximum_shmem_per_block;
	int			static_shmem_per_block;
	int			warp_size;
	CUresult	rc;

	/* get statically allocated shared memory */
	rc = cuFuncGetAttribute(&static_shmem_per_block,
							CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
							function);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuFuncGetAttribute: %s", errorText(rc));

	if (maximum_blocksize)
	{
		rc = cuFuncGetAttribute(&block_size,
								CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
								function);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuFuncGetAttribute: %s",
				 errorText(rc));

		rc = cuDeviceGetAttribute(&maximum_shmem_per_block,
							CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
								  device);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGetAttribute: %s",
				 errorText(rc));

		rc = cuDeviceGetAttribute(&warp_size,
								  CU_DEVICE_ATTRIBUTE_WARP_SIZE,
								  device);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGetAttribute: %s",
				 errorText(rc));

		while (static_shmem_per_block +
			   dynamic_shmem_per_thread * block_size > maximum_shmem_per_block)
			block_size <<= 1;

		if (block_size < warp_size)
			elog(ERROR, "Expected block size is too small (%d)", block_size);

		*p_block_size = block_size;
		*p_grid_size = (nitems + block_size - 1) / block_size;
	}
	else
	{
		__dynamic_shmem_per_thread = dynamic_shmem_per_thread;
		rc = cuOccupancyMaxPotentialBlockSize(&grid_size,
											  &block_size,
											  function,
											  dynamic_shmem_size_per_block,
											  static_shmem_per_block,
											  cuda_max_threads_per_block);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuOccupancyMaxPotentialBlockSize: %s",
				 errorText(rc));

		*p_grid_size = grid_size;
		*p_block_size = block_size;
	}
}

static bool
pgstrom_check_device_capability(int ordinal, CUdevice device)
{
	bool		result = true;
	char		dev_name[256];
	size_t		dev_mem_sz;
	int			dev_mem_clk;
	int			dev_mem_width;
	int			dev_l2_sz;
	int			dev_cap_major;
	int			dev_cap_minor;
	int			dev_mpu_nums;
	int			dev_mpu_clk;
	int			dev_max_threads_per_block;
	int			num_cores;
	CUresult	rc;
	CUdevice_attribute attrib;

	rc = cuDeviceGetName(dev_name, sizeof(dev_name), device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetName: %s", errorText(rc));

	rc = cuDeviceTotalMem(&dev_mem_sz, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceTotalMem: %s", errorText(rc));

	attrib = CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
	rc = cuDeviceGetAttribute(&dev_max_threads_per_block, attrib, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", errorText(rc));

	attrib = CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE;
	rc = cuDeviceGetAttribute(&dev_mem_clk, attrib, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", errorText(rc));

	attrib = CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH;
	rc = cuDeviceGetAttribute(&dev_mem_width, attrib, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", errorText(rc));

	attrib = CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE;
	rc = cuDeviceGetAttribute(&dev_l2_sz, attrib, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", errorText(rc));

	attrib = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
	rc = cuDeviceGetAttribute(&dev_cap_major, attrib, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", errorText(rc));

	attrib = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
	rc = cuDeviceGetAttribute(&dev_cap_minor, attrib, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", errorText(rc));

	attrib = CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
	rc = cuDeviceGetAttribute(&dev_mpu_nums, attrib, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", errorText(rc));

	attrib = CU_DEVICE_ATTRIBUTE_CLOCK_RATE;
	rc = cuDeviceGetAttribute(&dev_mpu_clk, attrib, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetAttribute: %s", errorText(rc));

	/*
	 * CUDA device older than Kepler is not supported
	 */
	if (dev_cap_major < 3)
		result = false;

	/*
	 * Number of CUDA cores (just for log messages)
	 */
	if (dev_cap_major == 1)
		num_cores = 8;
	else if (dev_cap_major == 2)
	{
		if (dev_cap_minor == 0)
			num_cores = 32;
		else if (dev_cap_minor == 1)
			num_cores = 48;
		else
			num_cores = -1;
	}
	else if (dev_cap_major == 3)
		num_cores = 192;
	else if (dev_cap_major == 5)
		num_cores = 128;
	else
		num_cores = -1;		/* unknown */

	/*
	 * Track referenced device property
	 */
	cuda_max_malloc_size = Min(cuda_max_malloc_size,
							   (dev_mem_sz / 3) & ~((1UL << 20) - 1));
	cuda_max_threads_per_block = Min(cuda_max_threads_per_block,
									 dev_max_threads_per_block);
	cuda_compute_capability = Min(cuda_compute_capability,
								  100 * dev_cap_major + dev_cap_minor);

	/* Log the brief CUDA device properties */
	elog(LOG, "GPU[%d] %s (%d CUDA cores, %d SMs, %dMHz), L2 %dKB, RAM %zuMB (%dbits, %dKHz), computing capability %d.%d%s",
		 ordinal,
		 dev_name,
		 num_cores > 0 ? num_cores * dev_mpu_nums : -1,
		 dev_mpu_nums,
		 dev_mpu_clk / 1000,
		 dev_l2_sz >> 10,
		 dev_mem_sz >> 20,
		 dev_mem_width,
		 dev_mem_clk / 1000,
		 dev_cap_major,
		 dev_cap_minor,
		 !result ? ", NOT SUPPORTED" : "");

	return result;
}

void
pgstrom_init_cuda_control(void)
{
	MemoryContext oldcxt;
	CUdevice	device;
	CUresult	rc;
	int			i, count;

	/*
	 * initialization of CUDA runtime
	 */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuInit(%s)", errorText(rc));

	/*
	 * construct a list of available devices
	 */
	rc = cuDeviceGetCount(&count);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetCount(%s)", errorText(rc));

	oldcxt = MemoryContextSwitchTo(TopMemoryContext);
	for (i=0; i < count; i++)
	{
		rc = cuDeviceGet(&device, i);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGet(%s)", errorText(rc));
		if (pgstrom_check_device_capability(i, device))
			cuda_device_ordinals = lappend_int(cuda_device_ordinals, i);
	}
	MemoryContextSwitchTo(oldcxt);
	if (cuda_device_ordinals == NIL)
		elog(ERROR, "No CUDA devices, PG-Strom was disabled");

	/*
	 * initialization of GpuContext related stuff
	 */
	for (i=0; i < lengthof(gcontext_hash); i++)
		dlist_init(&gcontext_hash[i]);
	RegisterResourceReleaseCallback(gpucontext_cleanup_callback, NULL);
}

/*
 * errorText
 *
 * translation from cuda error code to text representation
 */
const char *
errorText(int errcode)
{
	static __thread char buffer[512];
	const char *error_val;
	const char *error_str;

	switch (errcode)
	{
		case StromError_CpuReCheck:
			snprintf(buffer, sizeof(buffer), "CPU ReCheck");
			break;
		case StromError_CudaInternal:
			snprintf(buffer, sizeof(buffer), "CUDA Internal Error");
			break;
		case StromError_OutOfMemory:
			snprintf(buffer, sizeof(buffer), "Out of memory");
			break;
		case StromError_OutOfSharedMemory:
			snprintf(buffer, sizeof(buffer), "Out of shared memory");
			break;
		case StromError_DataStoreCorruption:
			snprintf(buffer, sizeof(buffer), "Data store corruption");
			break;
		case StromError_DataStoreNoSpace:
			snprintf(buffer, sizeof(buffer), "Data store no space");
			break;
		case StromError_DataStoreOutOfRange:
			snprintf(buffer, sizeof(buffer), "Data store out of range");
			break;
		case StromError_SanityCheckViolation:
			snprintf(buffer, sizeof(buffer), "Sanity check violation");
			break;
		default:
			if (cuGetErrorName(errcode, &error_val) == CUDA_SUCCESS &&
				cuGetErrorString(errcode, &error_str) == CUDA_SUCCESS)
				snprintf(buffer, sizeof(buffer), "%s - %s",
						 error_val, error_str);
			else
				snprintf(buffer, sizeof(buffer), "%d - unknown", errcode);
	}
	return buffer;
}

/*
 * pgstrom_device_info
 *
 *
 *
 */
#define DEVATTR_BOOL		1
#define DEVATTR_INT			2
#define DEVATTR_KB			3
#define DEVATTR_MHZ			4
#define DEVATTR_COMP_MODE	5
#define DEVATTR_BITS		6
Datum
pgstrom_device_info(PG_FUNCTION_ARGS)
{
	static struct {
		CUdevice_attribute	attrib;
		const char		   *attname;
		int					attkind;
	} catalog[] = {
		{CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
		 "max threads per block", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
		 "Maximum block dimension X", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
		 "Maximum block dimension Y", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
		 "Maximum block dimension Z", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
		 "Maximum grid dimension X", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
		 "Maximum grid dimension Y", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
		 "Maximum grid dimension Z", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
		 "Maximum shared memory available per block", DEVATTR_KB },
		{CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
		 "Memory available on device for __constant__", DEVATTR_KB},
		{CU_DEVICE_ATTRIBUTE_WARP_SIZE,
		 "Warp size in threads", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_PITCH,
		 "Maximum pitch in bytes allowed by memory copies", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
		 "Maximum number of 32bit registers available per block", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
		 "Typical clock frequency in kilohertz", DEVATTR_MHZ},
		{CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
		 "Alignment requirement for textures", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
		 "Number of multiprocessors on device", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
		 "Has kernel execution timeout", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_INTEGRATED,
		 "Integrated with host memory", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
		 "Host memory can be mapped to CUDA address space", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
		 "Compute mode", DEVATTR_COMP_MODE},
		{CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT,
		 "Alignment requirement for surfaces", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
		 "Multiple concurrent kernel support", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
		 "Device has ECC support enabled", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
		 "PCI bus ID of the device", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
		 "PCI device ID of the device", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_TCC_DRIVER,
		 "Device is using TCC driver model", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
		 "Peak memory clock frequency", DEVATTR_MHZ},
		{CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
		 "Global memory bus width", DEVATTR_BITS},
		{CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
		 "Size of L2 cache in bytes", DEVATTR_KB},
		{CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
		 "Maximum threads per multiprocessor", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
		 "Number of asynchronous engines", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
		 "Device shares unified address space", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
		 "PCI domain ID of the device", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
		 "Major compute capability version number", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
		 "Minor compute capability version number", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED,
		 "Device supports stream priorities", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED,
		 "Device supports caching globals in L1", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED,
		 "Device supports caching locals in L1", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
		 "Maximum shared memory per multiprocessor", DEVATTR_KB},
		{CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
		 "Maximum number of 32bit registers per multiprocessor", DEVATTR_INT},
		{CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
		 "Device can allocate managed memory on this system", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD,
		 "Device is on a multi-GPU board", DEVATTR_BOOL},
		{CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID,
		 "Unique id if device is on a multi-GPU board", DEVATTR_INT},
	};
	FuncCallContext *fncxt;
	CUresult	rc;
	int			dindex;
	int			aindex;
	const char *att_name;
	char	   *att_value;
	Datum		values[3];
	bool		isnull[3];
	HeapTuple	tuple;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;

		fncxt = SRF_FIRSTCALL_INIT();
        oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(3, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "id",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "attribute",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "value",
						   TEXTOID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		fncxt->user_fctx = 0;

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	dindex = fncxt->call_cntr / (lengthof(catalog) + 2);
	aindex = fncxt->call_cntr % (lengthof(catalog) + 2);

	if (cuda_num_devices < 0)
		pgstrom_init_cuda();

	if (dindex >= cuda_num_devices)
		SRF_RETURN_DONE(fncxt);

	if (aindex == 0)
	{
		char	dev_name[256];

		rc = cuDeviceGetName(dev_name, sizeof(dev_name), cuda_devices[dindex]);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGetName: %s", errorText(rc));
		att_name = "Device name";
		att_value = pstrdup(dev_name);
	}
	else if (aindex == 1)
	{
		size_t	dev_memsz;

		rc = cuDeviceTotalMem(&dev_memsz, cuda_devices[dindex]);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceTotalMem: %s", errorText(rc));
		att_name = "Total global memory size";
		att_value = psprintf("%zu MBytes", dev_memsz >> 20);
	}
	else
	{
		int		property;

		rc = cuDeviceGetAttribute(&property,
								  catalog[aindex].attrib,
								  cuda_devices[dindex]);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGetAttribute: %s",
				 errorText(rc));

		att_name = catalog[aindex].attname;
		switch (catalog[aindex].attkind)
		{
			case DEVATTR_BOOL:
				att_value = psprintf("%s", property != 0 ? "True" : "False");
				break;
			case DEVATTR_INT:
				att_value = psprintf("%d", property);
				break;
			case DEVATTR_KB:
				att_value = psprintf("%d KBytes", property / 1024);
				break;
			case DEVATTR_MHZ:
				att_value = psprintf("%d MHz", property / 1000);
				break;
			case DEVATTR_COMP_MODE:
				break;
			case DEVATTR_BITS:
				att_value = psprintf("%d bits", property);
				break;
			default:
				elog(ERROR, "Bug? unexpected device attribute type");
		}
	}
	memset(isnull, 0, sizeof(isnull));
	values[0] = Int32GetDatum(dindex);
	values[1] = CStringGetTextDatum(att_name);
	values[2] = CStringGetTextDatum(att_value);

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_device_info);
