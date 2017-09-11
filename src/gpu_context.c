/*
 * gpu_context.c
 *
 * Routines to manage GPU context.
 * ----
 * Copyright 2011-2017 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2017 (C) The PG-Strom Development Team
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
#include "access/twophase.h"
#include "storage/ipc.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/pg_crc.h"
#include "utils/resowner.h"
#include "pg_strom.h"

/* variables */
static shmem_startup_hook_type shmem_startup_next = NULL;
static pg_atomic_uint32 *global_num_running_tasks;	/* per device */
int				global_max_async_tasks;		/* GUC */
int				local_max_async_tasks;		/* GUC */
static slock_t		activeGpuContextLock;
static dlist_head	activeGpuContextList;

/*
 * Resource tracker of GpuContext
 *
 * It enables to track various resources with GpuContext, to detect resource
 * leaks.
 */
#define RESTRACK_CLASS__GPUMEMORY		2
#define RESTRACK_CLASS__HOSTMEMORY		3
#define RESTRACK_CLASS__GPUPROGRAM		4

typedef struct ResourceTracker
{
	dlist_node	chain;
	pg_crc32	crc;
	cl_int		resclass;
	union {
		struct {
			CUdeviceptr	ptr;	/* RESTRACK_CLASS__GPUMEMORY */
			void   *extra;
		} devmem;
		struct {
			void   *ptr;		/* RESTRACK_CLASS__HOSTMEMORY */
			void   *extra;
		} hostmem;
		ProgramId	program_id;	/* RESTRACK_CLASS__GPUPROGRAM */
	} u;
} ResourceTracker;

static inline pg_crc32
resource_tracker_hashval(cl_int resclass, void *data, size_t len)
{
	pg_crc32	crc;

	INIT_LEGACY_CRC32(crc);
	COMP_LEGACY_CRC32(crc, &resclass, sizeof(cl_int));
	COMP_LEGACY_CRC32(crc, data, len);
	FIN_LEGACY_CRC32(crc);

	return crc;
}

/*
 * resource tracker for GPU program
 */
bool
trackCudaProgram(GpuContext *gcontext, ProgramId program_id)
{
	ResourceTracker *tracker = calloc(1, sizeof(ResourceTracker));
	pg_crc32	crc;

	if (!tracker)
		return false;	/* out of memory */

	crc = resource_tracker_hashval(RESTRACK_CLASS__GPUPROGRAM,
								   &program_id, sizeof(ProgramId));
	tracker->crc = crc;
	tracker->resclass = RESTRACK_CLASS__GPUPROGRAM;
	tracker->u.program_id = program_id;
	SpinLockAcquire(&gcontext->restrack_lock);
	dlist_push_tail(&gcontext->restrack[crc % RESTRACK_HASHSIZE],
					&tracker->chain);
	SpinLockRelease(&gcontext->restrack_lock);
	return true;
}

void
untrackCudaProgram(GpuContext *gcontext, ProgramId program_id)
{
	dlist_head *restrack_list;
    dlist_iter	iter;
    pg_crc32	crc;

	crc = resource_tracker_hashval(RESTRACK_CLASS__GPUPROGRAM,
								   &program_id, sizeof(ProgramId));
	SpinLockAcquire(&gcontext->restrack_lock);
	restrack_list = &gcontext->restrack[crc % RESTRACK_HASHSIZE];
	dlist_foreach(iter, restrack_list)
	{
		ResourceTracker *tracker
			= dlist_container(ResourceTracker, chain, iter.cur);

		if (tracker->crc == crc &&
			tracker->resclass == RESTRACK_CLASS__GPUPROGRAM &&
			tracker->u.program_id == program_id)
		{
			dlist_delete(&tracker->chain);
			SpinLockRelease(&gcontext->restrack_lock);
			free(tracker);
			return;
		}
	}
	SpinLockRelease(&gcontext->restrack_lock);
	wnotice("Bug? CUDA Program %lu was not tracked", program_id);
}

/*
 * resource tracker for device memory
 */
bool
trackGpuMem(GpuContext *gcontext, CUdeviceptr devptr, void *extra)
{
	ResourceTracker *tracker = calloc(1, sizeof(ResourceTracker));
	pg_crc32	crc;

	if (!tracker)
		return false;	/* out of memory */

	crc = resource_tracker_hashval(RESTRACK_CLASS__GPUMEMORY,
								   &devptr, sizeof(CUdeviceptr));
	tracker->crc = crc;
	tracker->resclass = RESTRACK_CLASS__GPUMEMORY;
	tracker->u.devmem.ptr = devptr;
	tracker->u.devmem.extra = extra;

	SpinLockAcquire(&gcontext->restrack_lock);
	dlist_push_tail(&gcontext->restrack[crc % RESTRACK_HASHSIZE],
					&tracker->chain);
	SpinLockRelease(&gcontext->restrack_lock);
	return true;
}

void *
lookupGpuMem(GpuContext *gcontext, CUdeviceptr devptr)
{
	dlist_head *restrack_list;
	dlist_iter	iter;
	pg_crc32	crc;
	void	   *extra = NULL;

	crc = resource_tracker_hashval(RESTRACK_CLASS__GPUMEMORY,
								   &devptr, sizeof(CUdeviceptr));
	restrack_list = &gcontext->restrack[crc % RESTRACK_HASHSIZE];
	SpinLockAcquire(&gcontext->restrack_lock);
	dlist_foreach (iter, restrack_list)
	{
		ResourceTracker *tracker
			= dlist_container(ResourceTracker, chain, iter.cur);

		if (tracker->crc == crc &&
			tracker->resclass == RESTRACK_CLASS__GPUMEMORY &&
			tracker->u.devmem.ptr == devptr)
		{
			extra = tracker->u.devmem.extra;
			break;
		}
	}
	SpinLockRelease(&gcontext->restrack_lock);
	return extra;
}

void *
untrackGpuMem(GpuContext *gcontext, CUdeviceptr devptr)
{
	dlist_head *restrack_list;
	dlist_iter	iter;
	pg_crc32	crc;
	void	   *extra;

	crc = resource_tracker_hashval(RESTRACK_CLASS__GPUMEMORY,
								   &devptr, sizeof(CUdeviceptr));
	restrack_list = &gcontext->restrack[crc % RESTRACK_HASHSIZE];
	SpinLockAcquire(&gcontext->restrack_lock);
	dlist_foreach (iter, restrack_list)
	{
		ResourceTracker *tracker
			= dlist_container(ResourceTracker, chain, iter.cur);

		if (tracker->crc == crc &&
			tracker->resclass == RESTRACK_CLASS__GPUMEMORY &&
			tracker->u.devmem.ptr == devptr)
		{
			dlist_delete(&tracker->chain);
			extra = tracker->u.devmem.extra;
			SpinLockRelease(&gcontext->restrack_lock);
			free(tracker);
			return extra;
		}
	}
	SpinLockRelease(&gcontext->restrack_lock);
	wnotice("Bug? GPU Device Memory %p was not tracked", (void *)devptr);
	abort();
	return NULL;
}

/*
 * resource tracker for host memory
 */
bool
trackHostMem(GpuContext *gcontext, void *hostptr, void *extra)
{
	ResourceTracker *tracker = calloc(1, sizeof(ResourceTracker));
	pg_crc32	crc;

	if (!tracker)
		return false;	/* out of memory */

	crc = resource_tracker_hashval(RESTRACK_CLASS__HOSTMEMORY,
								   &hostptr, sizeof(void *));
	tracker->crc = crc;
	tracker->resclass = RESTRACK_CLASS__HOSTMEMORY;
	tracker->u.hostmem.ptr = hostptr;
	tracker->u.hostmem.extra = extra;

	SpinLockAcquire(&gcontext->restrack_lock);
	dlist_push_tail(&gcontext->restrack[crc % RESTRACK_HASHSIZE],
					&tracker->chain);
	SpinLockRelease(&gcontext->restrack_lock);
	return true;
}

void *
lookupHostMem(GpuContext *gcontext, void *hostptr)
{
	dlist_head *restrack_list;
	dlist_iter	iter;
	pg_crc32	crc;
	void	   *extra = NULL;

	crc = resource_tracker_hashval(RESTRACK_CLASS__HOSTMEMORY,
								   &hostptr, sizeof(void *));
	restrack_list = &gcontext->restrack[crc % RESTRACK_HASHSIZE];
	SpinLockAcquire(&gcontext->restrack_lock);
	dlist_foreach (iter, restrack_list)
	{
		ResourceTracker *tracker
			= dlist_container(ResourceTracker, chain, iter.cur);

		if (tracker->crc == crc &&
			tracker->resclass == RESTRACK_CLASS__HOSTMEMORY &&
			tracker->u.hostmem.ptr == hostptr)
		{
			extra = tracker->u.hostmem.extra;
			break;
		}
	}
	SpinLockRelease(&gcontext->restrack_lock);
	return extra;
}

void *
untrackHostMem(GpuContext *gcontext, void *hostptr)
{
	dlist_head *restrack_list;
	dlist_iter	iter;
	pg_crc32	crc;
	void	   *extra;

	crc = resource_tracker_hashval(RESTRACK_CLASS__GPUMEMORY,
								   &hostptr, sizeof(void *));
	restrack_list = &gcontext->restrack[crc % RESTRACK_HASHSIZE];
	SpinLockAcquire(&gcontext->restrack_lock);
	dlist_foreach (iter, restrack_list)
	{
		ResourceTracker *tracker
			= dlist_container(ResourceTracker, chain, iter.cur);

		if (tracker->crc == crc &&
			tracker->resclass == RESTRACK_CLASS__HOSTMEMORY &&
			tracker->u.hostmem.ptr == hostptr)
		{
			dlist_delete(&tracker->chain);
			extra = tracker->u.hostmem.extra;
			SpinLockRelease(&gcontext->restrack_lock);
			free(tracker);
			return extra;
		}
	}
	SpinLockRelease(&gcontext->restrack_lock);
	wnotice("Bug? GPU Device Memory %p was not tracked", hostptr);
	return NULL;
}

/*
 * GpuContextLookupModule
 */
typedef struct
{
	dlist_node	chain;
	ProgramId	program_id;
	CUmodule	cuda_module;
} GpuContextModuleEntry;

static CUmodule
GpuContextLookupModule(GpuContext *gcontext, ProgramId program_id)
{
	GpuContextModuleEntry *entry;
	dlist_iter	iter;
	cl_int		index = program_id % CUDA_MODULES_HASHSIZE;
	CUmodule	cuda_module;

	SpinLockAcquire(&gcontext->cuda_modules_lock);
	dlist_foreach(iter, &gcontext->cuda_modules_slot[index])
	{
		entry = dlist_container(GpuContextModuleEntry, chain, iter.cur);
		if (entry->program_id == program_id)
		{
			cuda_module = entry->cuda_module;
			SpinLockRelease(&gcontext->cuda_modules_lock);

			return cuda_module;
		}
	}
	entry = calloc(1, sizeof(GpuContextModuleEntry));
	if (!entry)
	{
		SpinLockRelease(&gcontext->cuda_modules_lock);
		werror("out of memory");
	}
	cuda_module = pgstrom_load_cuda_program(program_id);

	entry->cuda_module = cuda_module;
	entry->program_id = program_id;
	dlist_push_head(&gcontext->cuda_modules_slot[index],
					&entry->chain);
	SpinLockRelease(&gcontext->cuda_modules_lock);

	return cuda_module;
}

/*
 * ReleaseLocalResources - release all the private resources tracked by
 * the resource tracker of GpuContext
 */
static void
ReleaseLocalResources(GpuContext *gcontext, bool normal_exit)
{
	ResourceTracker *tracker;
	dlist_node		*dnode;
	CUresult		rc;
	int				i;

	/* should be deactivated already */
	Assert(!gcontext->cuda_context);

	/* OK, release other resources */
	for (i=0; i < RESTRACK_HASHSIZE; i++)
	{
		while (!dlist_is_empty(&gcontext->restrack[i]))
		{
			dnode = dlist_pop_head_node(&gcontext->restrack[i]);
			tracker = dlist_container(ResourceTracker, chain, dnode);

			switch (tracker->resclass)
			{
				case RESTRACK_CLASS__GPUMEMORY:
					if (normal_exit)
						wnotice("GPU memory %p likely leaked",
								(void *)tracker->u.devmem.ptr);
					rc = gpuMemFreeExtra(gcontext,
										 tracker->u.devmem.ptr,
										 tracker->u.devmem.extra);
					if (rc != CUDA_SUCCESS)
						wnotice("failed on cuMemFree(%p): %s",
								(void *)tracker->u.devmem.ptr,
								errorText(rc));
					break;
				case RESTRACK_CLASS__GPUPROGRAM:
					if (normal_exit)
						wnotice("CUDA Program ID=%lu is likely leaked",
								tracker->u.program_id);
					pgstrom_put_cuda_program(NULL, tracker->u.program_id);
					break;
				default:
					wnotice("Bug? unknown resource tracker class: %d",
							(int)tracker->resclass);
					break;
			}
			free(tracker);
		}
	}
	/* unmap GPU device memory segment */
	pgstrom_gpu_mmgr_cleanup_gpucontext(gcontext);
	/* release Host pinned memory, if kept */
	pgstrom_datastore_cleanup_gpucontext(gcontext);

	/* NOTE: cuda_module is already released by cuCtxDestroy() */
	for (i=0; i < CUDA_MODULES_HASHSIZE; i++)
	{
		GpuContextModuleEntry *entry;
		dlist_node	   *dnode;

		while (!dlist_is_empty(&gcontext->cuda_modules_slot[i]))
		{
			dnode = dlist_pop_head_node(&gcontext->cuda_modules_slot[i]);
			entry = dlist_container(GpuContextModuleEntry, chain, dnode);
			free(entry);
		}
	}
	if (gcontext->error_message)
		free(gcontext->error_message);
	free(gcontext);
}

/*
 * GpuContextWorkerReportError
 */
__thread GpuContext	   *GpuWorkerCurrentContext = NULL;
__thread sigjmp_buf	   *GpuWorkerExceptionStack = NULL;
__thread CUevent		CU_EVENT0_PER_THREAD = NULL;
__thread CUevent		CU_EVENT1_PER_THREAD = NULL;
__thread CUevent		CU_EVENT2_PER_THREAD = NULL;
__thread CUevent		CU_EVENT3_PER_THREAD = NULL;

void
GpuContextWorkerReportError(int elevel,
							const char *filename, int lineno,
							const char *fmt, ...)
{
	GpuContext *gcontext = GpuWorkerCurrentContext;
	uint32		expected = 0;
	va_list		va_args;
	ssize_t		length;

	Assert(gcontext != NULL);
	Assert(elevel != 0);

	va_start(va_args, fmt);
	length = vfprintf(stderr, fmt, va_args);
	if (elevel < ERROR)
	{
		va_end(va_args);
		return;
	}

	if (pg_atomic_compare_exchange_u32(&gcontext->error_level,
									   &expected, (uint32)elevel))
	{
		gcontext->error_filename	= filename;
		gcontext->error_lineno		= lineno;
		gcontext->error_message		= malloc(length + 1);
		if (gcontext->error_message)
			vsnprintf(gcontext->error_message, length+1, fmt, va_args);
	}
	va_end(va_args);

	if (GpuWorkerExceptionStack)
		siglongjmp(*GpuWorkerExceptionStack, 1);
}

/*
 * GpuContextWorkerMain
 */
static void *
GpuContextWorkerMain(void *arg)
{
	GpuContext	   *gcontext = arg;
	dlist_node	   *dnode;
	GpuTask		   *gtask;
	CUresult		rc;

	rc = cuCtxSetCurrent(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
	{
		/* NOTE: GpuWorkerExceptionStack is not set, so werror() will not
		 * make a long-jump at this timing. */
		werror("failed on cuCtxSetCurrent: %s", errorText(rc));
		return NULL;
	}

	/* general purpose events objects */
	if ((rc = cuEventCreate(&CU_EVENT0_PER_THREAD,
							CU_EVENT_DISABLE_TIMING)) != CUDA_SUCCESS ||
		(rc = cuEventCreate(&CU_EVENT1_PER_THREAD,
							CU_EVENT_DISABLE_TIMING)) != CUDA_SUCCESS ||
		(rc = cuEventCreate(&CU_EVENT2_PER_THREAD,
							CU_EVENT_DISABLE_TIMING)) != CUDA_SUCCESS ||
		(rc = cuEventCreate(&CU_EVENT3_PER_THREAD,
							CU_EVENT_DISABLE_TIMING)) != CUDA_SUCCESS)
	{
		werror("failed on cuEventCreate: %s", errorText(rc));
		return NULL;
	}
	GpuWorkerCurrentContext = gcontext;

	wnotice("worker is now running");
	pthreadMutexLock(&gcontext->mutex);
	while (!gcontext->terminate_workers)
	{
		//try asyncronous program build if any
		//don't forget to signal cond

		if (dlist_is_empty(&gcontext->pending_tasks))
			pthreadCondWait(&gcontext->cond, &gcontext->mutex);
		else
		{
			dnode = dlist_pop_head_node(&gcontext->pending_tasks);
			gtask = dlist_container(GpuTask, chain, dnode);
			pthreadMutexUnlock(&gcontext->mutex);

			wnotice("fetch a GpuTask %p", gtask);

			/* Execution of the GpuTask */
			STROM_TRY();
			{
				CUmodule	cuda_module;
				cl_int		retval;

				cuda_module = GpuContextLookupModule(gcontext,
													 gtask->program_id);
				do {
					/*
					 * pgstromProcessGpuTask() returns the following status:
					 *
					 *  0 : GpuTask gets completed successfully, then task
					 *      object shall be backed to the backend.
					 *  1 : Unable to launch GpuTask due to lack of GPU's
					 *      resource. It shall be retried after a short wait.
					 * -1 : GpuTask gets completed successfully, and the
					 *      handler wants to release GpuTask immediately.
					 */
					retval = pgstromProcessGpuTask(gtask, cuda_module);
					if (retval == 0)
					{
						pthreadMutexLock(&gcontext->mutex);
						gcontext->num_running_tasks--;
						dlist_push_tail(&gcontext->completed_tasks,
										&gtask->chain);
						pthreadMutexUnlock(&gcontext->mutex);

						SetLatch(MyLatch);
					}
					else if (retval > 0)
					{
						/* Wait for 40ms */
						pg_usleep(40000L);
					}
					else
					{
						pthreadMutexLock(&gcontext->mutex);
						gcontext->num_running_tasks--;
						pthreadMutexUnlock(&gcontext->mutex);

						pgstromReleaseGpuTask(gtask);

						SetLatch(MyLatch);
					}
				} while (retval > 0);
			}
			STROM_CATCH();
			{
				wnotice("Got exception, thus, exit workers");
				/* Wake up and terminate other workers also */
				pthreadMutexLock(&gcontext->mutex);
				gcontext->terminate_workers = true;
				pthreadCondBroadcast(&gcontext->cond);
				pthreadMutexUnlock(&gcontext->mutex);
				SetLatch(MyLatch);
			}
			STROM_END_TRY();

			pthreadMutexLock(&gcontext->mutex);
		}
	}
	pthreadMutexUnlock(&gcontext->mutex);
	wnotice("worker now exit...");

	return NULL;
}

/*
 * ActivateGpuContext - kicks worker threads for workq
 */
static void
ActivateGpuContext(GpuContext *gcontext)
{
	CUdevice	cuda_device;
	CUcontext	cuda_context;
	CUresult	rc;
	cl_int		i;

	if (gcontext->cuda_context)
		return;		/* already activated */

	rc = cuDeviceGet(&cuda_device,
					 gcontext->cuda_dindex);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGet: %s",
			 errorText(rc));

	if (gcontext->never_use_mps)
		rc = CUDA_ERROR_OUT_OF_MEMORY;
	else
		rc = cuCtxCreate(&cuda_context,
						 CU_CTX_SCHED_AUTO,
						 cuda_device);
	if (rc != CUDA_SUCCESS)
	{
		char   *env_saved;

		env_saved = getenv("CUDA_MPS_PIPE_DIRECTORY");
		if (setenv("CUDA_MPS_PIPE_DIRECTORY", "/dev/null", 1) != 0)
			elog(ERROR, "failed on setenv: %m");
		rc = cuCtxCreate(&cuda_context,
                         CU_CTX_SCHED_AUTO,
                         cuda_device);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxCreate: %s",
				 errorText(rc));
		if (!env_saved)
			unsetenv("CUDA_MPS_PIPE_DIRECTORY");
		else
			setenv("CUDA_MPS_PIPE_DIRECTORY", env_saved, 1);
	}
	gcontext->cuda_device	= cuda_device;
	gcontext->cuda_context	= cuda_context;

	/* creation of worker threads */
	for (i=0; i < gcontext->num_workers; i++)
	{
		pthread_t	thread;

		if ((errno = pthread_create(&thread, NULL,
									GpuContextWorkerMain,
									gcontext)) != 0)
		{
			int		errno_saved = errno;

			pthreadMutexLock(&gcontext->mutex);
			gcontext->terminate_workers = true;
			pthreadCondBroadcast(&gcontext->cond);
			pthreadMutexUnlock(&gcontext->mutex);

			while (i > 0)
			{
				thread = gcontext->worker_threads[--i];

				if ((errno = pthread_join(thread, NULL)) != 0)
					elog(WARNING, "failed on pthread_join: %m");
			}
			rc = cuCtxDestroy(gcontext->cuda_context);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuCtxDestroy: %s", errorText(rc));
			gcontext->cuda_context	= NULL;
			gcontext->cuda_device	= 0;

			errno = errno_saved;
			elog(ERROR, "failed on pthread_create: %m");
		}
		gcontext->worker_threads[i] = thread;
	}
	elog(NOTICE, "GpuContext is activated (%p)", gcontext->cuda_context);
}

/*
 * GetGpuContext - acquire a free GpuContext
 */
GpuContext *
AllocGpuContext(int cuda_dindex,
				bool never_use_mps,
				bool with_activation)
{
	static bool		cuda_driver_initialized = false;
	GpuContext	   *gcontext = NULL;
	dlist_iter		iter;
	CUresult		rc;
	int				i;

	/* per-process driver initialization */
	if (!cuda_driver_initialized)
	{
		rc = cuInit(0);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuInit: %s", errorText(rc));
		cuda_driver_initialized = true;
		elog(NOTICE, "CUDA Driver Initialized");
	}

	/*
	 * Lookup an existing active GpuContext
	 */
	SpinLockAcquire(&activeGpuContextLock);
	dlist_foreach(iter, &activeGpuContextList)
	{
		gcontext = dlist_container(GpuContext, chain, iter.cur);

		if (gcontext->resowner == CurrentResourceOwner &&
			(!never_use_mps || gcontext->never_use_mps) &&
			(cuda_dindex < 0 || gcontext->cuda_dindex == cuda_dindex))
		{
			pg_atomic_fetch_add_u32(&gcontext->refcnt, 1);
			SpinLockRelease(&activeGpuContextLock);
			return gcontext;
		}
	}
	SpinLockRelease(&activeGpuContextLock);

	/*
	 * Not found, let's create a new GpuContext
	 */
	gcontext = calloc(1, offsetof(GpuContext,
								  worker_threads[local_max_async_tasks]));
	if (!gcontext)
		elog(ERROR, "out of memory");

	/* choose a device to use, if no preference */
	if (cuda_dindex < 0)
	{
		cuda_dindex = (IsParallelWorker()
					   ? ParallelWorkerNumber
					   : MyProc->pgprocno) % numDevAttrs;
	}

	pg_atomic_init_u32(&gcontext->refcnt, 1);
	gcontext->resowner		= CurrentResourceOwner;
	gcontext->never_use_mps	= never_use_mps;
	gcontext->cuda_dindex	= cuda_dindex;
	SpinLockInit(&gcontext->cuda_modules_lock);
	for (i=0; i < CUDA_MODULES_HASHSIZE; i++)
		dlist_init(&gcontext->cuda_modules_slot[i]);
	/* resource management */
	SpinLockInit(&gcontext->restrack_lock);
	for (i=0; i < RESTRACK_HASHSIZE; i++)
		dlist_init(&gcontext->restrack[i]);
	/* GPU device memory management */
	pgstrom_gpu_mmgr_init_gpucontext(gcontext);
	/* error information buffer */
	pg_atomic_init_u32(&gcontext->error_level, 0);
	gcontext->error_filename = NULL;
	gcontext->error_lineno	= 0;
	gcontext->error_message	= NULL;
	/* management of work-queue */
	gcontext->global_num_running_tasks
		= &global_num_running_tasks[cuda_dindex];
	pthreadMutexInit(&gcontext->mutex);
	pthreadCondInit(&gcontext->cond);
	gcontext->num_running_tasks = 0;
	dlist_init(&gcontext->pending_tasks);
	dlist_init(&gcontext->completed_tasks);
	gcontext->num_workers = local_max_async_tasks;

	SpinLockAcquire(&activeGpuContextLock);
	dlist_push_head(&activeGpuContextList, &gcontext->chain);
	SpinLockRelease(&activeGpuContextLock);

	if (with_activation)
		ActivateGpuContext(gcontext);

	return gcontext;
}

/*
 * GetGpuContext - increment reference counter
 */
GpuContext *
GetGpuContext(GpuContext *gcontext)
{
	uint32		oldcnt __attribute__((unused));

	oldcnt = pg_atomic_fetch_add_u32(&gcontext->refcnt, 1);
	Assert(oldcnt > 0);

	return gcontext;
}

/*
 * PutGpuContext - detach GpuContext; to be called by only backend
 */
void
PutGpuContext(GpuContext *gcontext)
{
	uint32		newcnt __attribute__((unused));

	newcnt = pg_atomic_sub_fetch_u32(&gcontext->refcnt, 1);
	if (newcnt == 0)
	{
		SpinLockAcquire(&activeGpuContextLock);
		dlist_delete(&gcontext->chain);
		SpinLockRelease(&activeGpuContextLock);
		/* wait for completion of worker threads */
		SynchronizeGpuContext(gcontext);
		/* cleanup local resources */
		ReleaseLocalResources(gcontext, true);
	}
}

/*
 * SynchronizeGpuContext - terminate all the worker and wait for completion
 */
void
SynchronizeGpuContext(GpuContext *gcontext)
{
	CUresult	rc;
	int			i;

	if (!gcontext->cuda_context)
		return;

	/* signal to terminate workers */
	pthreadMutexLock(&gcontext->mutex);
	gcontext->terminate_workers = true;
	pthreadCondBroadcast(&gcontext->cond);
	pthreadMutexUnlock(&gcontext->mutex);

	/* Drop CUDA context to interrupt long-running GPU kernels, if any */
	rc = cuCtxDestroy(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on cuCtxDestroy: %s", errorText(rc));

	/* Wait for completion of the worker threads */
	for (i=0; i < gcontext->num_workers; i++)
	{
		pthread_t	thread = gcontext->worker_threads[i];

		if ((errno = pthread_join(thread, NULL)) != 0)
			elog(FATAL, "failed on pthread_join: %m");
	}
	gcontext->cuda_device = 0;
	gcontext->cuda_context = NULL;
	memset(gcontext->worker_threads, 0,
		   sizeof(pthread_t) * gcontext->num_workers);
}

/*
 * gpucontext_cleanup_callback - cleanup callback when drop of ResourceOwner
 */
static void
gpucontext_cleanup_callback(ResourceReleasePhase phase,
							bool isCommit,
							bool isTopLevel,
							void *arg)
{
	dlist_mutable_iter iter;

	if (phase != RESOURCE_RELEASE_BEFORE_LOCKS)
		return;

	SpinLockAcquire(&activeGpuContextLock);
	dlist_foreach_modify(iter, &activeGpuContextList)
	{
		GpuContext  *gcontext = (GpuContext *)
			dlist_container(GpuContext, chain, iter.cur);

		if (gcontext->resowner != CurrentResourceOwner)
			continue;
		if (isCommit)
			wnotice("GpuContext reference leak (refcnt=%d)",
					pg_atomic_read_u32(&gcontext->refcnt));
		dlist_delete(&gcontext->chain);
		SynchronizeGpuContext(gcontext);
		ReleaseLocalResources(gcontext, isCommit);
	}
	SpinLockRelease(&activeGpuContextLock);
}

/*
 * gpucontext_proc_exit_cleanup - cleanup callback when process exit
 */
static void
gpucontext_proc_exit_cleanup(int code, Datum arg)
{
	//for each gpucontext
	//terminate workq
}

/*
 * pgstrom_startup_gpu_context
 */
static void
pgstrom_startup_gpu_context(void)
{
	bool	found;
	int		i;

	if (shmem_startup_next)
		(*shmem_startup_next)();

	global_num_running_tasks =
		ShmemInitStruct("Global number of running tasks counter",
						sizeof(pg_atomic_uint32) * numDevAttrs,
						&found);
	if (found)
		elog(ERROR, "Bug? Global number of running tasks counter exists");
	for (i=0; i < numDevAttrs; i++)
		pg_atomic_init_u32(&global_num_running_tasks[i], 0);
}

/*
 * pgstrom_init_gpu_context
 */
void
pgstrom_init_gpu_context(void)
{
	DefineCustomIntVariable("pg_strom.global_max_async_tasks",
			"Soft limit for the number of concurrent GpuTasks in system-wide",
							NULL,
							&global_max_async_tasks,
							160,
							8,
							INT_MAX,
							PGC_SUSET,
							GUC_NOT_IN_SAMPLE,
                            NULL, NULL, NULL);
	DefineCustomIntVariable("pg_strom.local_max_async_tasks",
			"Soft limit for the number of concurrent GpuTasks per backend",
							NULL,
							&local_max_async_tasks,
							8,
							1,
							64,
							PGC_SUSET,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	/* initialization of GpuContext List */
	SpinLockInit(&activeGpuContextLock);
	dlist_init(&activeGpuContextList);

	/* shared memory */
	RequestAddinShmemSpace(MAXALIGN(sizeof(pg_atomic_uint32) * numDevAttrs));
	shmem_startup_next = shmem_startup_hook;
    shmem_startup_hook = pgstrom_startup_gpu_context;

	/* register the callback to clean up resources */
	RegisterResourceReleaseCallback(gpucontext_cleanup_callback, NULL);
	before_shmem_exit(gpucontext_proc_exit_cleanup, 0);
}
