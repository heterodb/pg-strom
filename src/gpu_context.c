/*
 * gpu_context.c
 *
 * Routines to manage GPU context.
 * ----
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
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

/* IPC stuff of GpuContext */
typedef struct
{
	dlist_node			chain;
	pthread_mutex_t		mutex;
	pthread_cond_t		cond;
	pg_atomic_uint32	command;
} GpuContextIPCEntry;

typedef struct
{
	slock_t			lock;
	dlist_head		free_list;	/* list of GpuContextIPCEntry */
	dlist_head	   *active_list;/* list of GpuContextIPCEntry; per device */
	GpuContextIPCEntry ipc_entries[FLEXIBLE_ARRAY_MEMBER];
} GpuContextIPCHead;

/* variables */
static shmem_startup_hook_type shmem_startup_next = NULL;
static pg_atomic_uint32 *global_num_running_tasks;	/* shared */
static GpuContextIPCHead *gcontext_ipc_head;	/* shared */
int					global_max_async_tasks;		/* GUC */
int					local_max_async_tasks;		/* GUC */
int					max_num_gpucontext;			/* GUC */
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
		ProgramId	program_id;	/* RESTRACK_CLASS__GPUPROGRAM */
		int			filedesc;	/* RESTRACK_CLASS__FILEDESC */
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
trackRawFileDesc(GpuContext *gcontext, int filedesc,
				 const char *filename, int lineno)
{
	ResourceTracker *tracker = calloc(1, sizeof(ResourceTracker));
	pg_crc32	crc;

	if (!tracker)
		return false;	/* out of memory */

	crc = resource_tracker_hashval(RESTRACK_CLASS__FILEDESC,
								   &filedesc, sizeof(int));
	tracker->crc = crc;
	tracker->resclass = RESTRACK_CLASS__FILEDESC;
	tracker->filename = filename;
	tracker->lineno = lineno;
	tracker->u.filedesc = filedesc;
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
untrackRawFileDesc(GpuContext *gcontext, int filedesc)
{
	dlist_head *restrack_list;
	dlist_iter	iter;
	pg_crc32	crc;

	crc = resource_tracker_hashval(RESTRACK_CLASS__FILEDESC,
								   &filedesc, sizeof(int));
	SpinLockAcquire(&gcontext->restrack_lock);
	restrack_list = &gcontext->restrack[crc % RESTRACK_HASHSIZE];
	dlist_foreach(iter, restrack_list)
	{
		ResourceTracker *tracker
			= dlist_container(ResourceTracker, chain, iter.cur);

		if (tracker->crc == crc &&
			tracker->resclass == RESTRACK_CLASS__FILEDESC &&
			tracker->u.filedesc == filedesc)
        {
			dlist_delete(&tracker->chain);
			SpinLockRelease(&gcontext->restrack_lock);
			free(tracker);
			return;
		}
	}
	SpinLockRelease(&gcontext->restrack_lock);
	wnotice("Bug? File Descriptor %d was not tracked", filedesc);
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
 * GpuContextLookupModule
 */
typedef struct
{
	dlist_node	chain;
	ProgramId	program_id;
	CUmodule	cuda_module;
} GpuContextModuleEntry;

CUmodule
GpuContextLookupModule(GpuContext *gcontext, ProgramId program_id)
{
	GpuContextModuleEntry *entry;
	dlist_iter	iter;
	cl_int		index = program_id % CUDA_MODULES_HASHSIZE;
	CUmodule	cuda_module;

	STROM_TRY();
	{
		while (!pthreadMutexLockTimeout(&gcontext->cuda_modules_lock, 500))
		{
			if (pg_atomic_read_u32(&gcontext->terminate_workers) != 0)
				werror("worker termination is required");
		}

		dlist_foreach(iter, &gcontext->cuda_modules_slot[index])
		{
			entry = dlist_container(GpuContextModuleEntry, chain, iter.cur);
			if (entry->program_id == program_id)
			{
				cuda_module = entry->cuda_module;
				goto found;
			}
		}
		entry = calloc(1, sizeof(GpuContextModuleEntry));
		if (!entry)
			werror("out of memory");
		cuda_module = pgstrom_load_cuda_program(program_id);

		entry->cuda_module = cuda_module;
		entry->program_id = program_id;
		dlist_push_head(&gcontext->cuda_modules_slot[index],
						&entry->chain);
	found:
		pthreadMutexUnlock(&gcontext->cuda_modules_lock);
	}
	STROM_CATCH();
	{
		pthreadMutexUnlock(&gcontext->cuda_modules_lock);
		STROM_RE_THROW();
	}
	STROM_END_TRY();

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
					/*
					 * NOTE: All the GPU related memory shall be wiped out
					 * at the cuCtxDestroy() below, so no need to release
					 * individual memory chunks by ourselves.
					 */
					Assert(gcontext->cuda_context != NULL);
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
								tracker->u.filedesc,
								__basename(tracker->filename),
								tracker->lineno);
					if (close(tracker->u.filedesc))
						wnotice("failed on close(2): %m");
					break;
				default:
					wnotice("Bug? unknown resource tracker class: %d",
							(int)tracker->resclass);
					break;
			}
			free(tracker);
		}
	}

	/* drop cuda context */
	if (gcontext->cuda_context)
	{
		rc = cuCtxDestroy(gcontext->cuda_context);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "Failed on cuCtxDestroy: %s", errorText(rc));
		gcontext->cuda_context = NULL;
	}

	/* unmap GPU device memory segment */
	pgstrom_gpu_mmgr_cleanup_gpucontext(gcontext);

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
	uint32			command;
	bool			is_wakeup;

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
		while (pg_atomic_read_u32(&gcontext->terminate_workers) == 0)
		{
			GpuTaskState *gts;
			CUmodule	cuda_module;
			cl_int		retval;

			pthreadMutexLock(gcontext->mutex);
			if (dlist_is_empty(&gcontext->pending_tasks))
			{
				is_wakeup = pthreadCondWaitTimeout(gcontext->cond,
												   gcontext->mutex,
												   4000);
				pthreadMutexUnlock(gcontext->mutex);
				if (is_wakeup)
					command = pg_atomic_exchange_u32(gcontext->command, 0);
				else
					command = GPUCTX_CMD__RECLAIM_MEMORY;

				if ((command & GPUCTX_CMD__RECLAIM_MEMORY) != 0)
				{
					/*
					 * XXX - Once GPU related tasks get idle, all the worker
					 * threads may reach the timeout almost simultaneously.
					 */
					pthreadCondSignal(gcontext->cond);
					gpuMemReclaimSegment(gcontext);
				}
			}
			else
			{
				dnode = dlist_pop_head_node(&gcontext->pending_tasks);
				gtask = dlist_container(GpuTask, chain, dnode);
				pthreadMutexUnlock(gcontext->mutex);

				gts = gtask->gts;
				cuda_module = GpuContextLookupModule(gcontext,
													 gtask->program_id);
			retry_gputask:
				/*
				 * pgstromProcessGpuTask() returns the following status:
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
					/* wait for 40ms */
					pg_usleep(40000L);
					if (pg_atomic_read_u32(&gcontext->terminate_workers) == 0)
						goto retry_gputask;
					else
					{
						/*
						 * urgent bailout if GpuContext is shutting down.
						 */
						pthreadMutexLock(gcontext->mutex);
						dlist_push_tail(&gcontext->pending_tasks,
										&gtask->chain);
						gts->num_running_tasks--;
						pthreadMutexUnlock(gcontext->mutex);
					}
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
					pthreadMutexLock(gcontext->mutex);
					dlist_push_tail(&gts->ready_tasks,
									&gtask->chain);
					gts->num_running_tasks--;
					gts->num_ready_tasks++;
					pthreadMutexUnlock(gcontext->mutex);

					SetLatch(MyLatch);
				}
				else
				{
					/*
					 * Release GpuTask immediately, expect for the last
					 * GpuTask when retval==-2.
					 */
					pthreadMutexLock(gcontext->mutex);
					if (--gts->num_running_tasks == 0 &&
						retval == -2 &&
						gts->scan_done)
					{
						wnotice("last one task");
						dlist_push_tail(&gts->ready_tasks,
										&gtask->chain);
						gts->num_ready_tasks++;
						pthreadMutexUnlock(gcontext->mutex);
					}
					else
					{
						pthreadMutexUnlock(gcontext->mutex);

						gts->cb_release_task(gtask);
					}
					SetLatch(MyLatch);
				}
			}
		}
	}
	STROM_CATCH();
	{
		/* Wake up and terminate other workers also */
		pg_atomic_write_u32(&gcontext->terminate_workers, 1);
		pthreadCondBroadcast(gcontext->cond);
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
		if (rc == CUDA_SUCCESS)
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
	CUdevice	cuda_device;
	CUcontext	cuda_context;
	CUresult	rc;
	cl_int		dindex = gcontext->cuda_dindex;

	if (gcontext->cuda_context)
		return;
	Assert(dindex >= 0 && dindex < numDevAttrs);
	rc = cuDeviceGet(&cuda_device, devAttrs[dindex].DEV_ID);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuDeviceGet: %s", errorText(rc));

	if (!gcontext->never_use_mps)
	{
		rc = cuCtxCreate(&cuda_context,
						 CU_CTX_SCHED_AUTO,
						 cuda_device);
	}
	else
	{
		char   *environ_saved = getenv("CUDA_MPS_PIPE_DIRECTORY");

		if (setenv("CUDA_MPS_PIPE_DIRECTORY", "/dev/null", 1) != 0)
			werror("failed on setenv: %m");
		rc = cuCtxCreate(&cuda_context,
						 CU_CTX_SCHED_AUTO,
						 cuda_device);
		if (!environ_saved)
			unsetenv("CUDA_MPS_PIPE_DIRECTORY");
		else
			setenv("CUDA_MPS_PIPE_DIRECTORY", environ_saved, 1);
	}
	if (rc != CUDA_SUCCESS)
		werror("failed on cuCtxCreate: %s", errorText(rc));
	gcontext->cuda_context = cuda_context;
}

/*
 * activate_cuda_workers - launch worker threads on demand
 */
static void
activate_cuda_workers(GpuContext *gcontext)
{
	CUresult	rc;
	cl_int		i;

	if (gcontext->worker_is_running)
		return;

	Assert(gcontext->cuda_context != NULL);
	GPUCONTEXT_PUSH(gcontext);
	for (i=0; i < gcontext->num_workers; i++)
	{
		if (!gcontext->cuda_events0[i])
		{
			rc = cuEventCreate(&gcontext->cuda_events0[i],
							   CU_EVENT_BLOCKING_SYNC);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));
		}

		if (!gcontext->cuda_events1[i])
		{
			rc = cuEventCreate(&gcontext->cuda_events1[i],
							   CU_EVENT_BLOCKING_SYNC);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));
		}
	}
	GPUCONTEXT_POP(gcontext);

	/* creation of worker threads */
	for (i=0; i < gcontext->num_workers; i++)
	{
		pthread_t	thread;

		if ((errno = pthread_create(&thread, NULL,
									GpuContextWorkerMain,
									gcontext)) != 0)
			elog(ERROR, "failed on pthread_create: %m");

		gcontext->worker_threads[i] = thread;
	}
	gcontext->worker_is_running = true;
}

/*
 * GetGpuContext - acquire a free GpuContext
 */
GpuContext *
AllocGpuContext(int cuda_dindex, bool never_use_mps,
				bool activate_context,
				bool activate_workers)
{
	GpuContextIPCEntry *ipc_entry;
	GpuContext	   *gcontext = NULL;
	dlist_node	   *dnode;
	dlist_iter		iter;
	CUresult		rc;
	int				i, num_workers = local_max_async_tasks;

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
			(!never_use_mps || gcontext->never_use_mps) &&
			(cuda_dindex < 0 || gcontext->cuda_dindex == cuda_dindex))
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
	gcontext = calloc(1, offsetof(GpuContext, worker_threads[num_workers]) +
					  2 * sizeof(CUevent) * num_workers);
	if (!gcontext)
		elog(ERROR, "out of memory");
	gcontext->cuda_events0 = (CUevent *)
		((char *)gcontext + offsetof(GpuContext, worker_threads[num_workers]));
	gcontext->cuda_events1 = gcontext->cuda_events0 + num_workers;

	/* choose a device to use, if no preference */
	if (cuda_dindex < 0)
	{
		cuda_dindex = (IsParallelWorker()
					   ? ParallelWorkerNumber
					   : MyProc->pgprocno) % numDevAttrs;
	}

	/* Pick up IPC stuff */
	SpinLockAcquire(&gcontext_ipc_head->lock);
	if (dlist_is_empty(&gcontext_ipc_head->free_list))
	{
		SpinLockRelease(&gcontext_ipc_head->lock);
		elog(ERROR, "out of GpuContext (IPC stuff)");
	}
	dnode = dlist_pop_head_node(&gcontext_ipc_head->free_list);
	ipc_entry = dlist_container(GpuContextIPCEntry, chain, dnode);
	SpinLockRelease(&gcontext_ipc_head->lock);
	pthreadMutexInit(&ipc_entry->mutex, 1);
	pthreadCondInit(&ipc_entry->cond);
	pg_atomic_init_u32(&ipc_entry->command, 0);

	/* setup fields */
	pg_atomic_init_u32(&gcontext->refcnt, 1);
	gcontext->resowner		= CurrentResourceOwner;
	gcontext->never_use_mps	= never_use_mps;
	gcontext->cuda_dindex	= cuda_dindex;
	pthreadMutexInit(&gcontext->cuda_modules_lock, 0);
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
	memset(gcontext->error_message, 0, sizeof(gcontext->error_message));
	/* management of work-queue */
	gcontext->worker_is_running = false;
	gcontext->global_num_running_tasks
		= &global_num_running_tasks[cuda_dindex];
	gcontext->mutex		= &ipc_entry->mutex;
	gcontext->cond		= &ipc_entry->cond;
	gcontext->command	= &ipc_entry->command;
	pg_atomic_init_u32(&gcontext->terminate_workers, 0);
	dlist_init(&gcontext->pending_tasks);
	gcontext->num_workers = num_workers;
	pg_atomic_init_u32(&gcontext->worker_index, 0);
	for (i=0; i < num_workers; i++)
		gcontext->worker_threads[i] = pthread_self();

	SpinLockAcquire(&activeGpuContextLock);
	dlist_push_head(&activeGpuContextList, &gcontext->chain);
	SpinLockRelease(&activeGpuContextLock);

	SpinLockAcquire(&gcontext_ipc_head->lock);
	dlist_push_tail(&gcontext_ipc_head->active_list[cuda_dindex],
					&ipc_entry->chain);
	SpinLockRelease(&gcontext_ipc_head->lock);
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
 * DetachGpuContextIPCEntry
 */
static void
DetachGpuContextIPCEntry(GpuContext *gcontext)
{
	GpuContextIPCEntry *ipc_entry = (GpuContextIPCEntry *)
		((char *)gcontext->mutex - offsetof(GpuContextIPCEntry, mutex));

	SpinLockAcquire(&gcontext_ipc_head->lock);
	/* detach from the active list */
	dlist_delete(&ipc_entry->chain);
	dlist_push_tail(&gcontext_ipc_head->free_list,
					&ipc_entry->chain);
	SpinLockRelease(&gcontext_ipc_head->lock);
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
		DetachGpuContextIPCEntry(gcontext);
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

	if (!gcontext->worker_is_running)
		return;

	/* signal to terminate all workers */
	pg_atomic_write_u32(&gcontext->terminate_workers, 1);
	pthreadCondBroadcast(gcontext->cond);
	/* interrupt cuEventSynchronize() */
	GPUCONTEXT_PUSH(gcontext);
	for (i=0; i < gcontext->num_workers; i++)
	{
		if ((rc = cuEventRecord(gcontext->cuda_events0[i],
								CU_STREAM_PER_THREAD)) != CUDA_SUCCESS)
			elog(WARNING, "failed on cuEventRecord: %s", errorText(rc));
		if ((rc = cuEventRecord(gcontext->cuda_events1[i],
								CU_STREAM_PER_THREAD)) != CUDA_SUCCESS)
			elog(WARNING, "failed on cuEventRecord: %s", errorText(rc));
	}
	GPUCONTEXT_POP(gcontext);

	/* wait for completion of the worker threads */
	for (i=0; i < gcontext->num_workers; i++)
	{
		pthread_t	thread = gcontext->worker_threads[i];

		if ((errno = pthread_join(thread, NULL)) != 0)
			elog(PANIC, "failed on pthread_join: %m");
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
		DetachGpuContextIPCEntry(gcontext);
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

	gcontext_ipc_head =
		ShmemInitStruct("IPC stuff for GpuContex",
						MAXALIGN(offsetof(GpuContextIPCHead,
										  ipc_entries[max_num_gpucontext])) +
						MAXALIGN(sizeof(dlist_head) * numDevAttrs),
						&found);
	if (found)
		elog(ERROR, "Bug? IPC stuff for GpuContex exists");
	SpinLockInit(&gcontext_ipc_head->lock);
	dlist_init(&gcontext_ipc_head->free_list);
	gcontext_ipc_head->active_list = (dlist_head *)
		((char *)gcontext_ipc_head +
		 MAXALIGN(offsetof(GpuContextIPCHead,
						   ipc_entries[max_num_gpucontext])));
	for (i=0; i < numDevAttrs; i++)
		dlist_init(&gcontext_ipc_head->active_list[i]);
	for (i=0; i < max_num_gpucontext; i++)
	{
		GpuContextIPCEntry *entry = &gcontext_ipc_head->ipc_entries[i];

		memset(entry, 0, sizeof(GpuContextIPCEntry));
		dlist_push_tail(&gcontext_ipc_head->free_list,
						&entry->chain);
	}
}

/*
 * pgstrom_init_gpu_context
 */
void
pgstrom_init_gpu_context(void)
{
	cl_int		max_nprocs;

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

	max_nprocs = MaxConnections + max_worker_processes;
	DefineCustomIntVariable("pg_strom.max_number_of_gpucontext",
							"Max number of GpuContext available at same time",
							NULL,
							&max_num_gpucontext,
							Max(3 * max_nprocs, 256),
							Max(3 * max_nprocs, 256),
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE | GUC_NO_SHOW_ALL,
							NULL, NULL, NULL);

	/* initialization of GpuContext List */
	SpinLockInit(&activeGpuContextLock);
	dlist_init(&activeGpuContextList);

	/* shared memory */
	RequestAddinShmemSpace(MAXALIGN(sizeof(pg_atomic_uint32) * numDevAttrs) +
						   MAXALIGN(offsetof(GpuContextIPCHead,
											ipc_entries[max_num_gpucontext])) +
						   MAXALIGN(sizeof(dlist_head) * numDevAttrs));
	shmem_startup_next = shmem_startup_hook;
    shmem_startup_hook = pgstrom_startup_gpu_context;

	/* register the callback to clean up resources */
	RegisterResourceReleaseCallback(gpucontext_cleanup_callback, NULL);
	before_shmem_exit(gpucontext_shmem_exit_cleanup, 0);
}
