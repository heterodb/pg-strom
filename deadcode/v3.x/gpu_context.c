/*
 * gpu_context.c
 *
 * Routines to manage GPU context.
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "postgres.h"
#include "access/twophase.h"
#include "storage/ipc.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/pg_crc.h"
#include "utils/resowner.h"
#include "pg_strom.h"

/* Per device raw CUDA resources */
typedef struct CudaResource
{
	cl_int			refcnt;
	cl_int			cuda_dindex;
	CUdevice		cuda_device;
	CUcontext		cuda_context;
	bool			can_reuse;
} CudaResource;

/* variables */
int					pgstrom_max_async_tasks;		/* GUC */
bool				pgstrom_reuse_cuda_context;	/* GUC */
static CudaResource *cuda_resources_array = NULL;
static slock_t		activeGpuContextLock;
static dlist_head	activeGpuContextList;

/*
 * Resource tracker of GpuContext
 *
 * It enables to track various resources with GpuContext, to detect resource
 * leaks.
 */
#define RESTRACK_CLASS__GPUMEMORY		2
#define RESTRACK_CLASS__GPUPROGRAM		3
#define RESTRACK_CLASS__GPUMEMORY_IPC	4
#define RESTRACK_CLASS__FILEDESC		5
#define RESTRACK_CLASS__GPUMODULE		6

typedef struct ResourceTracker
{
	dlist_node	chain;
	pg_crc32	crc;
	cl_int		resclass;
	const char *filename;
	cl_int		lineno;
	union {
		struct {
			CUdeviceptr	ptr;	/* RESTRACK_CLASS__GPUMEMORY */
			void   *extra;
		} devmem;
		struct {				/* RESTRACK_CLASS__GPUMEMORY_IPC */
			CUdeviceptr ptr;
			CUipcMemHandle handle;
			cl_uint		mapcount;
		} ipcmem;
		struct {				/* RESTRACK_CLASS__GPUMODULE */
			ProgramId	program_id;
			CUmodule	cuda_module;
		} module;
		ProgramId	program_id;	/* RESTRACK_CLASS__GPUPROGRAM */
		GPUDirectFileDesc filedesc;	/* RESTRACK_CLASS__FILEDESC */
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
trackCudaProgram(GpuContext *gcontext, ProgramId program_id,
				 const char *filename, int lineno)
{
	ResourceTracker *tracker = calloc(1, sizeof(ResourceTracker));
	pg_crc32	crc;

	if (!tracker)
		return false;	/* out of memory */

	crc = resource_tracker_hashval(RESTRACK_CLASS__GPUPROGRAM,
								   &program_id, sizeof(ProgramId));
	tracker->crc = crc;
	tracker->resclass = RESTRACK_CLASS__GPUPROGRAM;
	tracker->filename = filename;
	tracker->lineno = lineno;
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
trackGpuMem(GpuContext *gcontext, CUdeviceptr devptr, void *extra,
			const char *filename, int lineno)
{
	ResourceTracker *tracker = calloc(1, sizeof(ResourceTracker));
	pg_crc32	crc;

	if (!tracker)
		return false;	/* out of memory */

	crc = resource_tracker_hashval(RESTRACK_CLASS__GPUMEMORY,
								   &devptr, sizeof(CUdeviceptr));
	tracker->crc = crc;
	tracker->resclass = RESTRACK_CLASS__GPUMEMORY;
	tracker->filename = filename;
	tracker->lineno = lineno;
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
	return NULL;
}

/*
 * trackRawFileDesc - tracker of raw file descriptors
 */
bool
trackRawFileDesc(GpuContext *gcontext, GPUDirectFileDesc *filedesc,
				 const char *filename, int lineno)
{
	ResourceTracker *tracker = calloc(1, sizeof(ResourceTracker));
	pg_crc32	crc;

	if (!tracker)
		return false;	/* out of memory */

	crc = resource_tracker_hashval(RESTRACK_CLASS__FILEDESC,
								   &filedesc->rawfd, sizeof(int));
	tracker->crc = crc;
	tracker->resclass = RESTRACK_CLASS__FILEDESC;
	tracker->filename = filename;
	tracker->lineno = lineno;
	memcpy(&tracker->u.filedesc, filedesc, sizeof(GPUDirectFileDesc));
	SpinLockAcquire(&gcontext->restrack_lock);
	dlist_push_tail(&gcontext->restrack[crc % RESTRACK_HASHSIZE],
					&tracker->chain);
	SpinLockRelease(&gcontext->restrack_lock);
	return true;
}

/*
 * untrackRawFileDesc - untracker of raw file descriptors
 */
void
untrackRawFileDesc(GpuContext *gcontext, GPUDirectFileDesc *filedesc)
{
	dlist_head *restrack_list;
	dlist_iter	iter;
	pg_crc32	crc;

	crc = resource_tracker_hashval(RESTRACK_CLASS__FILEDESC,
								   &filedesc->rawfd, sizeof(int));
	SpinLockAcquire(&gcontext->restrack_lock);
	restrack_list = &gcontext->restrack[crc % RESTRACK_HASHSIZE];
	dlist_foreach(iter, restrack_list)
	{
		ResourceTracker *tracker
			= dlist_container(ResourceTracker, chain, iter.cur);

		if (tracker->crc == crc &&
			tracker->resclass == RESTRACK_CLASS__FILEDESC &&
			tracker->u.filedesc.rawfd == filedesc->rawfd)
		{
			dlist_delete(&tracker->chain);
			SpinLockRelease(&gcontext->restrack_lock);
			free(tracker);
			return;
		}
	}
	SpinLockRelease(&gcontext->restrack_lock);
	wnotice("Bug? File Descriptor %d was not tracked", filedesc->rawfd);
}

/*
 * Routines to manage external device memory opened via IPC handles.
 *
 * Note that gpu_context.c has these routines instead of gpu_mmgr.c,
 * because open/close of IPC handles must be serialized by restriction
 * of CUDA API.
 */

/*
 * __gpuIpcOpenMemHandle
 */
CUresult
__gpuIpcOpenMemHandle(GpuContext *gcontext,
					  CUdeviceptr *p_deviceptr,
					  CUipcMemHandle m_handle,
					  unsigned int flags,
					  const char *filename, int lineno)
{
	ResourceTracker *tracker;
	CUdeviceptr	m_deviceptr;
	CUresult	rc = CUDA_ERROR_OUT_OF_MEMORY;
	dlist_iter	iter;
	int			i;

	SpinLockAcquire(&gcontext->restrack_lock);
	/* XXX - do we have wise way to lookup mapped IPC memory? */
	for (i=0; i < RESTRACK_HASHSIZE; i++)
	{
		dlist_foreach(iter, &gcontext->restrack[i])
		{
			tracker = dlist_container(ResourceTracker, chain, iter.cur);
			if (tracker->resclass == RESTRACK_CLASS__GPUMEMORY_IPC &&
				memcmp(&tracker->u.ipcmem.handle, &m_handle,
					   sizeof(CUipcMemHandle)) == 0)
			{
				*p_deviceptr = tracker->u.ipcmem.ptr;
				tracker->u.ipcmem.mapcount++;
				SpinLockRelease(&gcontext->restrack_lock);

				return CUDA_SUCCESS;
			}
		}
	}
	/* not open yet */
	tracker = calloc(1, sizeof(ResourceTracker));
	if (!tracker)
		goto error;

	GPUCONTEXT_PUSH(gcontext);
	rc = cuIpcOpenMemHandle(&m_deviceptr, m_handle, flags);
	GPUCONTEXT_POP(gcontext);
	if (rc != CUDA_SUCCESS)
		goto error;

	tracker->crc = resource_tracker_hashval(RESTRACK_CLASS__GPUMEMORY_IPC,
											&m_deviceptr, sizeof(CUdeviceptr));
	tracker->resclass = RESTRACK_CLASS__GPUMEMORY_IPC;
	tracker->filename = filename;
	tracker->lineno = lineno;
	tracker->u.ipcmem.ptr = m_deviceptr;
	memcpy(&tracker->u.ipcmem.handle, &m_handle, sizeof(CUipcMemHandle));
	tracker->u.ipcmem.mapcount = 1;

	dlist_push_tail(&gcontext->restrack[tracker->crc % RESTRACK_HASHSIZE],
					&tracker->chain);
	SpinLockRelease(&gcontext->restrack_lock);

	*p_deviceptr = m_deviceptr;

	return CUDA_SUCCESS;

error:
	if (tracker)
		free(tracker);
	SpinLockRelease(&gcontext->restrack_lock);
	return rc;
}

/*
 * gpuIpcCloseMemHandle
 */
CUresult
gpuIpcCloseMemHandle(GpuContext *gcontext, CUdeviceptr m_deviceptr)
{
	dlist_head *restrack_list;
	dlist_iter	iter;
	pg_crc32	crc;
	CUresult	rc;

	crc = resource_tracker_hashval(RESTRACK_CLASS__GPUMEMORY_IPC,
								   &m_deviceptr, sizeof(CUdeviceptr));
	restrack_list = &gcontext->restrack[crc % RESTRACK_HASHSIZE];
    SpinLockAcquire(&gcontext->restrack_lock);
	dlist_foreach(iter, restrack_list)
	{
		ResourceTracker *tracker = dlist_container(ResourceTracker,
												   chain, iter.cur);
		if (tracker->crc == crc &&
			tracker->resclass == RESTRACK_CLASS__GPUMEMORY_IPC &&
			tracker->u.ipcmem.ptr == m_deviceptr)
		{
			Assert(tracker->u.ipcmem.mapcount > 0);
			if (--tracker->u.ipcmem.mapcount > 0)
			{
				SpinLockRelease(&gcontext->restrack_lock);
				return CUDA_SUCCESS;
			}
			dlist_delete(&tracker->chain);

			GPUCONTEXT_PUSH(gcontext);
			rc = cuIpcCloseMemHandle(m_deviceptr);
			GPUCONTEXT_POP(gcontext);
			
			SpinLockRelease(&gcontext->restrack_lock);
			free(tracker);

			return rc;
		}
	}
	SpinLockRelease(&gcontext->restrack_lock);
	wnotice("Bug? GPU Device Memory (IPC) %p was not tracked",
			(void *)m_deviceptr);
	return CUDA_ERROR_INVALID_VALUE;
}

/*
 * __GpuContextLookupModule
 */
CUmodule
__GpuContextLookupModule(GpuContext *gcontext, ProgramId program_id,
						 const char *filename, int lineno)
{
	ResourceTracker *tracker;
	dlist_head *restrack_list;
	dlist_iter	iter;
	pg_crc32	crc;
	CUmodule	cuda_module = NULL;

	crc = resource_tracker_hashval(RESTRACK_CLASS__GPUMODULE,
								   &program_id, sizeof(ProgramId));
	restrack_list = &gcontext->restrack[crc % RESTRACK_HASHSIZE];
	PG_TRY();
	{
		SpinLockAcquire(&gcontext->restrack_lock);
		dlist_foreach(iter, restrack_list)
		{
			tracker = dlist_container(ResourceTracker, chain, iter.cur);
			if (tracker->crc == crc &&
				tracker->resclass == RESTRACK_CLASS__GPUMODULE &&
				tracker->u.module.program_id == program_id)
			{
				cuda_module = tracker->u.module.cuda_module;
				break;
			}
		}

		if (!cuda_module)
		{
			cuda_module = pgstrom_load_cuda_program(program_id);
			tracker = calloc(1, sizeof(ResourceTracker));
            if (!tracker)
			{
				cuModuleUnload(cuda_module);
				werror("out of memory");
			}
			tracker->crc = crc;
			tracker->resclass = RESTRACK_CLASS__GPUMODULE;
			tracker->filename = filename;
			tracker->lineno = lineno;
			tracker->u.module.program_id = program_id;
			tracker->u.module.cuda_module = cuda_module;

			dlist_push_tail(restrack_list, &tracker->chain);
		}
		SpinLockRelease(&gcontext->restrack_lock);
	}
	PG_CATCH();
	{
		SpinLockRelease(&gcontext->restrack_lock);
		PG_RE_THROW();
	}
	PG_END_TRY();

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
	dlist_node *dnode;
	CUresult	rc;
	int			i;

	Assert(!gcontext->worker_is_running);
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
						wnotice("GPU memory %p by (%s:%d) likely leaked",
								(void *)tracker->u.devmem.ptr,
								__basename(tracker->filename),
								tracker->lineno);
					Assert(gcontext->cuda_context != NULL);
					if (tracker->u.devmem.extra == GPUMEM_DEVICE_RAW_EXTRA)
					{
						GPUCONTEXT_PUSH(gcontext);
						rc = cuMemFree(tracker->u.devmem.ptr);
						if (rc != CUDA_SUCCESS)
							wnotice("failed on cuMemFree: %s", errorText(rc));
						GPUCONTEXT_POP(gcontext);
					}
					else if (tracker->u.devmem.extra == GPUMEM_HOST_RAW_EXTRA)
					{
						GPUCONTEXT_PUSH(gcontext);
						rc = cuMemFreeHost((void *)tracker->u.devmem.ptr);
						if (rc != CUDA_SUCCESS)
							wnotice("failed on cuMemFreeHost: %s", errorText(rc));
						GPUCONTEXT_POP(gcontext);
					}
					break;
				case RESTRACK_CLASS__GPUMEMORY_IPC:
					if (normal_exit)
						wnotice("GPU memory (IPC) %p by (%s:%d) likely leaked",
								(void *)tracker->u.devmem.ptr,
								__basename(tracker->filename),
								tracker->lineno);
					Assert(gcontext->cuda_context != NULL);
					GPUCONTEXT_PUSH(gcontext);
					rc = cuIpcCloseMemHandle(tracker->u.devmem.ptr);
					if (rc != CUDA_SUCCESS)
						wnotice("failed on cuIpcCloseMemHandle: %s",
								errorText(rc));
					GPUCONTEXT_POP(gcontext);
					/*
					 * Even though we expect cuCtxDestroy() releases all
					 * the resources relevant the CUDA context, IPC device
					 * memory that is not closed leaves some kind of state.
					 * It leads a problem when we try to open another IPC
					 * device memory in a new CUDA context, so we explicitly
					 * close the IPC handle prior of cuCtxDestroy().
					 *
					 * (2020-03-04) This is reported to NVIDIA with a code
					 * for bug reproduction, for CUDA 10.2 + 440.33.01.
					 */
					break;
				case RESTRACK_CLASS__GPUPROGRAM:
					if (normal_exit)
						wnotice("CUDA Program ID=%lu by (%s:%d) is likely leaked",
								tracker->u.program_id,
								__basename(tracker->filename),
								tracker->lineno);
					pgstrom_put_cuda_program(NULL, tracker->u.program_id);
					break;
				case RESTRACK_CLASS__FILEDESC:
					if (normal_exit)
						wnotice("File desc %d by (%s:%d) is likely leaked",
								tracker->u.filedesc.rawfd,
								__basename(tracker->filename),
								tracker->lineno);
					gpuDirectFileDescClose(&tracker->u.filedesc);
					break;
				case RESTRACK_CLASS__GPUMODULE:
					rc = cuModuleUnload(tracker->u.module.cuda_module);
					if (rc != CUDA_SUCCESS)
						wnotice("failed on cuModuleUnload: %s", errorText(rc));
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

	/* destroy the CUDA context */
	if (gcontext->cuda_context)
	{
		CudaResource *cuda_resource = &cuda_resources_array[gcontext->cuda_dindex];

		Assert(cuda_resource->refcnt > 0);
		if (!normal_exit)
			cuda_resource->can_reuse = false;
		Assert(cuda_resource->refcnt > 0);
		if (--cuda_resource->refcnt == 0 &&
			(!cuda_resource->can_reuse ||
			 !pgstrom_reuse_cuda_context))
		{
			rc = cuCtxDestroy(cuda_resource->cuda_context);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuCtxDestroy: %s", errorText(rc));
			memset(cuda_resource, 0, sizeof(CudaResource));
		}
	}

	/* print debug counter, if any */
	{
		uint64	debug_count1 = pg_atomic_read_u64(&gcontext->debug_count1);
		uint64	debug_count2 = pg_atomic_read_u64(&gcontext->debug_count2);
		uint64	debug_count3 = pg_atomic_read_u64(&gcontext->debug_count3);
		uint64	debug_count4 = pg_atomic_read_u64(&gcontext->debug_count4);

		if (debug_count1 || debug_count2 || debug_count3 || debug_count4)
		{
			elog(NOTICE, "GpuContext %p { debug1: %lu, debug2: %lu, debug3: %lu, debug4: %lu }",
				 gcontext, debug_count1, debug_count2, debug_count3, debug_count4);
		}
	}
	free(gcontext);
}

/*
 * GpuContextWorkerReportError
 */
__thread GpuContext	   *GpuWorkerCurrentContext = NULL;
__thread sigjmp_buf	   *GpuWorkerExceptionStack = NULL;
__thread cl_int			GpuWorkerIndex = -1;

void
GpuContextWorkerReportError(int elevel,
							int errcode,
							const char *filename, int lineno,
							const char *funcname,
							const char *fmt, ...)
{
	GpuContext *gcontext = GpuWorkerCurrentContext;
	uint32		expected = 0;
	va_list		va_args;

	Assert(gcontext != NULL);
	Assert(elevel >= ERROR);
	if (pg_atomic_compare_exchange_u32(&gcontext->error_level,
									   &expected, (uint32)(2 * elevel + 1)))
	{
		const char *slash;

		slash = strrchr(filename,'/');
		if (slash)
			filename = slash + 1;

		gcontext->error_code		= errcode;
		gcontext->error_filename	= filename;
		gcontext->error_lineno		= lineno;
		gcontext->error_funcname	= funcname;
		va_start(va_args, fmt);
		vsnprintf(gcontext->error_message,
				  sizeof(gcontext->error_message), fmt, va_args);
		va_end(va_args);
		/* unlock error information */
		pg_atomic_fetch_and_u32(&gcontext->error_level, 0xfffffffeU);
	}
	siglongjmp(*GpuWorkerExceptionStack, 1);
}

/*
 * GpuContextWorkerMain
 */
__thread CUevent		CU_EVENT_PER_THREAD = NULL;

static void *
GpuContextWorkerMain(void *arg)
{
	GpuContext	   *gcontext = arg;
	dlist_node	   *dnode;
	GpuTask		   *gtask;
	CUresult		rc;

	/* setup worker index */
	GpuWorkerIndex = pg_atomic_fetch_add_u32(&gcontext->worker_index, 1);
	Assert(GpuWorkerIndex < gcontext->num_workers);

	rc = cuCtxSetCurrent(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
	{
		/* NOTE: GpuWorkerExceptionStack is not set, so werror() will not
		 * make a long-jump at this timing. */
		werror("failed on cuCtxSetCurrent: %s", errorText(rc));
		return NULL;
	}
	GpuWorkerCurrentContext = gcontext;

	STROM_TRY();
	{
		/* setup CU_EVENT_PER_THREAD variable */
		rc = cuEventCreate(&CU_EVENT_PER_THREAD,
						   CU_EVENT_BLOCKING_SYNC);
		if (rc != CUDA_SUCCESS)
			werror("failed on cuEventCreate: %s", errorText(rc));

		for (;;)
		{
			GpuTaskState *gts;
			CUmodule	cuda_module;
			cl_int		retval;
			bool		is_wakeup;

			pthreadMutexLock(&gcontext->worker_mutex);
			/* workers are required to terminate, so exit */
			if (pg_atomic_read_u32(&gcontext->terminate_workers) != 0)
			{
				pthreadMutexUnlock(&gcontext->worker_mutex);
				break;
			}

			/* here is no pending GpuTask, so sleep for a while */
			if (dlist_is_empty(&gcontext->pending_tasks))
			{
				is_wakeup = pthreadCondWaitTimeout(&gcontext->worker_cond,
												   &gcontext->worker_mutex,
												   3000L);
				pthreadMutexUnlock(&gcontext->worker_mutex);
				if (!is_wakeup)
				{
					/*
					 * Once GPU related tasks get idle for a certain duration
					 * (3s), we assume GPU device memory can be released
					 * prior to query execution end. Likely, workloads are
					 * moved to CPU or I/O intensive portion.
					 */
					pthreadCondSignal(&gcontext->worker_cond);
					gpuMemReclaimSegment(gcontext);
				}
				continue;
			}

			/* ok, dispatch a GpuTask in the head of pending-queue */
			dnode = dlist_pop_head_node(&gcontext->pending_tasks);
			gtask = dlist_container(GpuTask, chain, dnode);
			pthreadMutexUnlock(&gcontext->worker_mutex);

			gts = gtask->gts;
			if (!gts->cuda_module)
				werror("No CUDA module is not loaded");
			cuda_module = gts->cuda_module;

		retry_gputask:
			/*
			 * gts->cb_process_task shall return the following status:
			 *
			 *  0 : GpuTask gets completed successfully, then task
			 *      object shall be backed to the backend.
			 * >0 : Unable to launch GpuTask due to lack of GPU's
			 *      resource. It shall be retried after a short wait.
			 * <0 : GpuTask gets completed successfully, and the
			 *      handler wants to release GpuTask immediately.
			 */
			retval = gts->cb_process_task(gtask, cuda_module);
			if (retval > 0)
			{
				pg_usleep(20000L);		/* 20ms */
				pthreadMutexLock(&gcontext->worker_mutex);
				if (pg_atomic_read_u32(&gcontext->terminate_workers) != 0)
				{
					/* urgent bailout if GpuContext is shutting down. */
					dlist_push_tail(&gcontext->pending_tasks,
									&gtask->chain);
					gts->num_running_tasks--;
					pthreadMutexUnlock(&gcontext->worker_mutex);
					break;
				}
				/* elsewhere, try again */
				pthreadMutexUnlock(&gcontext->worker_mutex);
				goto retry_gputask;
			}
			else if (gtask->kerror.errcode != ERRCODE_STROM_SUCCESS)
			{
				/* GPU kernel completed with error status */
				GpuContextWorkerReportError(
					ERROR,
					gtask->kerror.errcode & ~ERRCODE_FLAGS_CPU_FALLBACK,
					gtask->kerror.filename,
					gtask->kerror.lineno,
					gtask->kerror.funcname,
					"GPU kernel: %s",
					gtask->kerror.message);
			}
			else if (retval == 0)
			{
				/* Back GpuTask to GTS */
				pthreadMutexLock(&gcontext->worker_mutex);
				dlist_push_tail(&gts->ready_tasks,
								&gtask->chain);
				gts->num_running_tasks--;
				gts->num_ready_tasks++;
				pthreadMutexUnlock(&gcontext->worker_mutex);

				SetLatch(MyLatch);
			}
			else
			{
				/*
				 * XXX - Anyone still use this logic for cleanup?
				 *
				 * Release GpuTask immediately, expect for the case
				 * when GpuTask returned -2 and it is the last one,
				 * to give the chance to release resources.
				 */
				pthreadMutexLock(&gcontext->worker_mutex);
				if (--gts->num_running_tasks == 0 &&
					retval == -2 &&
					gts->scan_done)
				{
					dlist_push_tail(&gts->ready_tasks,
									&gtask->chain);
					gts->num_ready_tasks++;
					pthreadMutexUnlock(&gcontext->worker_mutex);
				}
				else
				{
					pthreadMutexUnlock(&gcontext->worker_mutex);

					gts->cb_release_task(gtask);
				}
				SetLatch(MyLatch);
			}
		}
	}
	STROM_CATCH();
	{
		/* Wake up and terminate other workers also */
		pthreadMutexLock(&gcontext->worker_mutex);
		pg_atomic_write_u32(&gcontext->terminate_workers, 1);
		pthreadCondBroadcast(&gcontext->worker_cond);
		pthreadMutexUnlock(&gcontext->worker_mutex);
		SetLatch(MyLatch);
	}
	STROM_END_TRY();

	return NULL;
}

/*
 * gpuInit - a thin wrapper for cuInit
 */
CUresult
gpuInit(unsigned int flags)
{
	static bool	cuda_driver_initialized = false;
	CUresult	rc = CUDA_SUCCESS;

	if (!cuda_driver_initialized)
	{
		rc = cuInit(0);
		if (rc != CUDA_SUCCESS)
			return rc;
		cuda_driver_initialized = true;
	}
	return rc;
}

/*
 * activate_cuda_context - create a CUDA context on demand
 */
static void
activate_cuda_context(GpuContext *gcontext)
{
	CudaResource *cuda_resource;
	CUdevice	cuda_device;
	CUcontext	cuda_context;
	CUresult	rc;
	cl_int		dindex = gcontext->cuda_dindex;

	if (gcontext->cuda_context)
		return;
	Assert(dindex >= 0 && dindex < numDevAttrs);
	cuda_resource = &cuda_resources_array[dindex];
	if (cuda_resource->cuda_context)
	{
		Assert(cuda_resource->cuda_dindex == dindex);
		gcontext->cuda_device  = cuda_resource->cuda_device;
		gcontext->cuda_context = cuda_resource->cuda_context;
		cuda_resource->refcnt++;
		return;
	}
	/* no valid CUDA context, so create a new one */
	rc = cuDeviceGet(&cuda_device, devAttrs[dindex].DEV_ID);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuDeviceGet: %s", errorText(rc));

	rc = cuCtxCreate(&cuda_context,
					 CU_CTX_SCHED_AUTO,
					 cuda_device);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuCtxCreate: %s", errorText(rc));
	gcontext->cuda_device  = cuda_device;
	gcontext->cuda_context = cuda_context;

	/* setup CudaResource also */
	cuda_resource->cuda_dindex = dindex;
	cuda_resource->cuda_device = cuda_device;
	cuda_resource->cuda_context = cuda_context;
	cuda_resource->refcnt = 1;
	cuda_resource->can_reuse = true;
}

/*
 * activate_cuda_workers - launch worker threads on demand
 */
static void
activate_cuda_workers(GpuContext *gcontext)
{
	cl_int		i;

	if (gcontext->worker_is_running)
		return;
	/* creation of worker threads */
	Assert(gcontext->cuda_context != NULL);

	for (i=0; i < gcontext->num_workers; i++)
	{
		pthread_t	thread;

		errno = pthread_create(&thread, NULL,
							   GpuContextWorkerMain,
							   gcontext);
		if (errno != 0)
			elog(ERROR, "failed on pthread_create: %m");
		gcontext->worker_threads[i] = thread;

		/*
		 * NOTE: Even if pthread_create() failed to launch worker threads
		 * later, SynchronizeGpuContext() may terminate the worker threads
		 * already launched if worker_is_running.
		 */
		gcontext->worker_is_running = true;
	}
}

/*
 * GetGpuContext - acquire a free GpuContext
 */
GpuContext *
AllocGpuContext(const Bitmapset *optimal_gpus,
				bool activate_context,
				bool activate_workers)
{
	GpuContext	   *gcontext = NULL;
	dlist_iter		iter;
	CUresult		rc;
	int				cuda_dindex;
	int				i, num_workers = pgstrom_max_async_tasks;

	/* per-process driver initialization */
	rc = gpuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on gpuInit: %s", errorText(rc));

	/*
	 * Lookup an existing active GpuContext
	 */
	SpinLockAcquire(&activeGpuContextLock);
	dlist_foreach(iter, &activeGpuContextList)
	{
		gcontext = dlist_container(GpuContext, chain, iter.cur);

		if (gcontext->resowner == CurrentResourceOwner &&
			(bms_is_empty(optimal_gpus) ||
			 bms_is_member(gcontext->cuda_dindex, optimal_gpus)))
		{
			pg_atomic_fetch_add_u32(&gcontext->refcnt, 1);
			SpinLockRelease(&activeGpuContextLock);
			goto activation;
		}
	}
	SpinLockRelease(&activeGpuContextLock);
	/*
	 * Not found, so allocate a new one
	 */
	gcontext = calloc(1, offsetof(GpuContext, worker_threads[num_workers]));
	if (!gcontext)
		elog(ERROR, "out of memory");

	/* choose a device to use */
	if (bms_is_empty(optimal_gpus))
	{
		cuda_dindex = (IsParallelWorker()
					   ? ParallelWorkerNumber
					   : MyProc->pgprocno) % numDevAttrs;
	}
	else
	{
		int		ndiv = bms_num_members(optimal_gpus);
		int		count;

		Assert(ndiv > 0);
		count = (IsParallelWorker()
				 ? ParallelWorkerNumber
				 : MyProc->pgprocno) % ndiv;
		for (cuda_dindex = bms_next_member(optimal_gpus, -1);
			 cuda_dindex >= 0;
			 cuda_dindex = bms_next_member(optimal_gpus, cuda_dindex))
		{
			if (--count)
				break;
		}
		Assert(cuda_dindex >= 0 && cuda_dindex < numDevAttrs);
	}
	/* setup fields */
	pg_atomic_init_u32(&gcontext->refcnt, 1);
	gcontext->resowner		= CurrentResourceOwner;
	gcontext->cuda_dindex	= cuda_dindex;
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
	memset(gcontext->error_message, 0, sizeof(gcontext->error_message));
	/* management of work-queue */
	gcontext->worker_is_running = false;
	pthreadMutexInit(&gcontext->worker_mutex, 1);
	pthreadCondInit(&gcontext->worker_cond, 1);
	pg_atomic_init_u32(&gcontext->terminate_workers, 0);
	dlist_init(&gcontext->pending_tasks);
	gcontext->num_workers = num_workers;
	pg_atomic_init_u32(&gcontext->worker_index, 0);
	for (i=0; i < num_workers; i++)
		gcontext->worker_threads[i] = pthread_self();

	SpinLockAcquire(&activeGpuContextLock);
	dlist_push_head(&activeGpuContextList, &gcontext->chain);
	SpinLockRelease(&activeGpuContextLock);

activation:
	if (activate_context)
		activate_cuda_context(gcontext);
	if (activate_workers)
		activate_cuda_workers(gcontext);
	return gcontext;
}

/*
 * ActivateGpuContext
 */
void
ActivateGpuContext(GpuContext *gcontext)
{
	if (!gcontext->cuda_context)
		activate_cuda_context(gcontext);
	if (!gcontext->worker_is_running)
		activate_cuda_workers(gcontext);
}

/*
 * ActivateGpuContextNoWorkers - activate only cuda_context
 */
void
ActivateGpuContextNoWorkers(GpuContext *gcontext)
{
	if (!gcontext->cuda_context)
		activate_cuda_context(gcontext);
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
	uint32		newcnt;

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
	int			i;

	if (!gcontext->worker_is_running)
		return;

	/* signal to terminate all workers */
	pthreadMutexLock(&gcontext->worker_mutex);
	pg_atomic_write_u32(&gcontext->terminate_workers, 1);
	pthreadCondBroadcast(&gcontext->worker_cond);
	pthreadMutexUnlock(&gcontext->worker_mutex);

	/* wait for completion of the worker threads */
	for (i=0; i < gcontext->num_workers; i++)
	{
		pthread_t	thread = gcontext->worker_threads[i];

		if (pthread_equal(pthread_self(), thread) == 0 &&
			(errno = pthread_join(thread, NULL)) != 0)
			elog(PANIC, "failed on pthread_join: %m");
		gcontext->worker_threads[i] = pthread_self();
	}
	memset(gcontext->worker_threads, 0,
		   sizeof(pthread_t) * gcontext->num_workers);
	/* reset state for next activation */
	gcontext->worker_is_running = false;
	pg_atomic_write_u32(&gcontext->terminate_workers, 0);
	pg_atomic_write_u32(&gcontext->worker_index, 0);
}

/*
 * SynchronizeGpuContextOnDSMDetach
 */
void
SynchronizeGpuContextOnDSMDetach(dsm_segment *seg, Datum arg)
{
	GpuContext *gcontext = (GpuContext *)arg;
	dlist_iter	iter;

	SpinLockAcquire(&activeGpuContextLock);
	dlist_foreach(iter, &activeGpuContextList)
	{
		GpuContext *curr = dlist_container(GpuContext, chain, iter.cur);

		if (curr == gcontext)
		{
			SpinLockRelease(&activeGpuContextLock);
			SynchronizeGpuContext(gcontext);
			return;
		}
	}
	SpinLockRelease(&activeGpuContextLock);
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
 * gpucontext_shmem_exit_cleanup
 *
 * Cleanup callback when a process is going to exit horribly, just before
 * shmem detach. We need to fixup shared resources not to have side-effect
 * for the concurrent / working processes.
 */
static void
gpucontext_shmem_exit_cleanup(int code, Datum arg)
{
	while (!dlist_is_empty(&activeGpuContextList))
	{
		dlist_node *dnode = dlist_pop_head_node(&activeGpuContextList);
		GpuContext *gcontext = dlist_container(GpuContext, chain, dnode);
		dlist_iter	iter;
		int			i;

		/*
		 * GPU device memory shall be released on termination of the local
		 * process, so only CUDA Program resource shall be detached
		 * explicitly.
		 */
		for (i=0; i < RESTRACK_HASHSIZE; i++)
		{
			dlist_foreach(iter, &gcontext->restrack[i])
			{
				ResourceTracker *tracker =
					dlist_container(ResourceTracker, chain, iter.cur);

				if (tracker->resclass != RESTRACK_CLASS__GPUPROGRAM)
					continue;

				pgstrom_put_cuda_program(NULL, tracker->u.program_id);
			}
		}
	}
}

/*
 * pgstrom_init_gpu_context
 */
void
pgstrom_init_gpu_context(void)
{
	DefineCustomIntVariable("pg_strom.max_async_tasks",
							"Soft limit for CUDA worker threads per backend",
							NULL,
							&pgstrom_max_async_tasks,
							5,
							1,
							64,
							PGC_SUSET,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_strom.reuse_cuda_context",
							 "Reuse CUDA context, if query completed successfully",
							 NULL,
							 &pgstrom_reuse_cuda_context,
							 false,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);

	/* force to disable MPS to avoid troubles */
	if (setenv("CUDA_MPS_PIPE_DIRECTORY", "/dev/null", 1) != 0)
		elog(ERROR, "failed on setenv: %m");

	/* initialization of GpuContext List */
	cuda_resources_array = calloc(numDevAttrs, sizeof(CudaResource));
	if (!cuda_resources_array)
		elog(ERROR, "out of memory");

	SpinLockInit(&activeGpuContextLock);
	dlist_init(&activeGpuContextList);

	/* register the callback to clean up resources */
	RegisterResourceReleaseCallback(gpucontext_cleanup_callback, NULL);
	before_shmem_exit(gpucontext_shmem_exit_cleanup, 0);
}
