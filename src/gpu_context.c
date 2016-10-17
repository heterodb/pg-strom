/*
 * gpu_context.c
 *
 * Routines to manage GPU context.
 * ----
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
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

typedef struct SharedGpuContextHead
{
	slock_t			lock;
	dlist_head		active_list;
	dlist_head		free_list;
	SharedGpuContext master_context;
	SharedGpuContext context_array[FLEXIBLE_ARRAY_MEMBER];
} SharedGpuContextHead;

/* static variables */
static shmem_startup_hook_type shmem_startup_hook_next = NULL;
static SharedGpuContextHead *sharedGpuContextHead = NULL;
static GpuContext_v2	masterGpuContext;
static int				numGpuContexts;		/* GUC */

static dlist_head		activeGpuContextList;
static dlist_head		inactiveGpuContextList;
static dlist_head		inactiveResourceTracker;

/*
 * Resource tracker of GpuContext
 *
 * It enables to track various resources with GpuContext, to detect resource
 * leaks.
 */
#define RESTRACK_HASHSIZE				53

#define RESTRACK_CLASS__GPUMEMORY		2
#define RESTRACK_CLASS__GPUPROGRAM		3
#define RESTRACK_CLASS__IOMAPMEMORY		4
#define RESTRACK_CLASS__SSD2GPUDMA		5

typedef struct ResourceTracker
{
	dlist_node	chain;
	pg_crc32	crc;
	cl_int		resclass;
	union {
		CUdeviceptr	devptr;		/* RESTRACK_CLASS__GPUMEMORY
								 * RESTRACK_CLASS__IOMAPMEMORY */
		ProgramId	program_id;	/* RESTRACK_CLASS__GPUPROGRAM */
		unsigned long dma_task_id; /* RESTRACK_CLASS__SSD2GPUDMA */
	} u;
} ResourceTracker;

static inline ResourceTracker *
resource_tracker_alloc(void)
{
	ResourceTracker	   *restrack;

	if (dlist_is_empty(&inactiveResourceTracker))
		return MemoryContextAllocZero(TopMemoryContext,
									  sizeof(ResourceTracker));

	restrack = dlist_container(ResourceTracker, chain,
							   dlist_pop_head_node(&inactiveResourceTracker));
	memset(restrack, 0, sizeof(ResourceTracker));

	return restrack;
}

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
 * resource tracker for device memory
 */
CUresult
gpuMemAlloc_v2(GpuContext_v2 *gcontext, CUdeviceptr *p_devptr, size_t bytesize)
{
	SharedGpuContext *shgcon = gcontext->shgcon;
	ResourceTracker *tracker;
	CUdeviceptr		devptr;
    CUresult		rc;
	pg_crc32		crc;
	struct timeval	tv1, tv2;

	Assert(IsGpuServerProcess());
	if (shgcon->pfm.enabled)
		gettimeofday(&tv1, NULL);

	rc = cuMemAlloc(&devptr, bytesize);
	if (rc != CUDA_SUCCESS)
		return rc;

	tracker = resource_tracker_alloc();
	crc = resource_tracker_hashval(RESTRACK_CLASS__GPUMEMORY,
								   &devptr, sizeof(CUdeviceptr));
	tracker->crc = crc;
	tracker->resclass = RESTRACK_CLASS__GPUMEMORY;
	tracker->u.devptr = devptr;
	dlist_push_tail(&gcontext->restrack[crc % RESTRACK_HASHSIZE],
					&tracker->chain);
	*p_devptr = devptr;

	if (shgcon->pfm.enabled)
	{
		gettimeofday(&tv2, NULL);

		SpinLockAcquire(&shgcon->lock);
		shgcon->pfm.num_gpumem_alloc++;
		shgcon->pfm.tv_gpumem_alloc += PERFMON_TIMEVAL_DIFF(tv1,tv2);
		shgcon->pfm.size_gpumem_total += bytesize;
		SpinLockRelease(&shgcon->lock);
	}
	return rc;
}

CUresult
gpuMemFree_v2(GpuContext_v2 *gcontext, CUdeviceptr devptr)
{
	SharedGpuContext *shgcon = gcontext->shgcon;
	dlist_head	   *restrack_list;
	dlist_iter		iter;
	pg_crc32		crc;
	CUresult		rc;
	struct timeval	tv1, tv2;

	if (shgcon->pfm.enabled)
		gettimeofday(&tv1, NULL);

	crc = resource_tracker_hashval(RESTRACK_CLASS__GPUMEMORY,
								   &devptr, sizeof(CUdeviceptr));
	restrack_list = &gcontext->restrack[crc % RESTRACK_HASHSIZE];

	dlist_foreach(iter, restrack_list)
	{
		ResourceTracker *tracker
			= dlist_container(ResourceTracker, chain, iter.cur);

		if (tracker->crc == crc &&
			tracker->resclass == RESTRACK_CLASS__GPUMEMORY &&
			tracker->u.devptr == devptr)
		{
			dlist_delete(&tracker->chain);
			memset(tracker, 0, sizeof(ResourceTracker));
			dlist_push_head(&inactiveResourceTracker,
							&tracker->chain);
			goto found;
		}
    }
    elog(WARNING, "Bug? device pointer %p was not tracked", (void *)devptr);
found:
	rc = cuMemFree(devptr);
	notifierGpuMemFree(shgcon->device_id);

	if (shgcon->pfm.enabled)
	{
		gettimeofday(&tv2, NULL);

		SpinLockAcquire(&shgcon->lock);
		shgcon->pfm.num_gpumem_free++;
		shgcon->pfm.tv_gpumem_free += PERFMON_TIMEVAL_DIFF(tv1,tv2);
		SpinLockRelease(&shgcon->lock);
	}
	return rc;
}

/*
 * resource tracker for GPU program
 */
void
trackCudaProgram(GpuContext_v2 *gcontext, ProgramId program_id)
{
	ResourceTracker *tracker = resource_tracker_alloc();
	pg_crc32	crc;

	crc = resource_tracker_hashval(RESTRACK_CLASS__GPUPROGRAM,
								   &program_id, sizeof(ProgramId));
	tracker->crc = crc;
	tracker->resclass = RESTRACK_CLASS__GPUPROGRAM;
	tracker->u.program_id = program_id;

	dlist_push_tail(&gcontext->restrack[crc % RESTRACK_HASHSIZE],
					&tracker->chain);
}

void
untrackCudaProgram(GpuContext_v2 *gcontext, ProgramId program_id)
{
	pg_crc32	crc;
	cl_int		index;
	dlist_iter	iter;

	crc = resource_tracker_hashval(RESTRACK_CLASS__GPUPROGRAM,
								   &program_id, sizeof(ProgramId));
	index = crc % RESTRACK_HASHSIZE;

	dlist_foreach(iter, &gcontext->restrack[index])
	{
		ResourceTracker *tracker
			= dlist_container(ResourceTracker, chain, iter.cur);

		if (tracker->crc == crc &&
			tracker->resclass == RESTRACK_CLASS__GPUPROGRAM &&
			tracker->u.program_id == program_id)
		{
			dlist_delete(&tracker->chain);
			memset(tracker, 0, sizeof(ResourceTracker));
			dlist_push_head(&inactiveResourceTracker,
							&tracker->chain);
			return;
		}
	}
	elog(WARNING, "Bug? CUDA Program %lu was not tracked", program_id);
}

/*
 * resource tracker for i/o mapped memory
 */
void
trackIOMapMem(GpuContext_v2 *gcontext, CUdeviceptr devptr)
{
	ResourceTracker *tracker = resource_tracker_alloc();
	pg_crc32	crc;

	crc = resource_tracker_hashval(RESTRACK_CLASS__IOMAPMEMORY,
								   &devptr, sizeof(CUdeviceptr));
	tracker->crc = crc;
	tracker->resclass = RESTRACK_CLASS__IOMAPMEMORY;
	tracker->u.devptr = devptr;

	dlist_push_tail(&gcontext->restrack[crc % RESTRACK_HASHSIZE],
					&tracker->chain);
}

void
untrackIOMapMem(GpuContext_v2 *gcontext, CUdeviceptr devptr)
{
	pg_crc32	crc;
	cl_int		index;
	dlist_iter	iter;

	crc = resource_tracker_hashval(RESTRACK_CLASS__IOMAPMEMORY,
								   &devptr, sizeof(CUdeviceptr));
	index = crc % RESTRACK_HASHSIZE;

	dlist_foreach (iter, &gcontext->restrack[index])
	{
		ResourceTracker *tracker
			= dlist_container(ResourceTracker, chain, iter.cur);

		if (tracker->crc == crc &&
			tracker->resclass == RESTRACK_CLASS__IOMAPMEMORY &&
			tracker->u.devptr == devptr)
		{
			dlist_delete(&tracker->chain);
			memset(tracker, 0, sizeof(ResourceTracker));
			dlist_push_head(&inactiveResourceTracker,
							&tracker->chain);
			return;
		}
	}
	elog(WARNING, "Bug? I/O Mapped Memory %p was not tracked", (void *)devptr);
}

/*
 * resource tracker for SSD-to-GPU Direct DMA task
 */
void
trackSSD2GPUDMA(GpuContext_v2 *gcontext, unsigned long dma_task_id)
{
	ResourceTracker *tracker = resource_tracker_alloc();
	pg_crc32	crc;

	crc = resource_tracker_hashval(RESTRACK_CLASS__SSD2GPUDMA,
								   &dma_task_id, sizeof(unsigned long));
	tracker->crc = crc;
	tracker->resclass = RESTRACK_CLASS__SSD2GPUDMA;
	tracker->u.dma_task_id = dma_task_id;

	dlist_push_tail(&gcontext->restrack[crc % RESTRACK_HASHSIZE],
					&tracker->chain);
}

void
untrackSSD2GPUDMA(GpuContext_v2 *gcontext, unsigned long dma_task_id)
{
	pg_crc32	crc;
	cl_int		index;
	dlist_iter	iter;

	crc = resource_tracker_hashval(RESTRACK_CLASS__SSD2GPUDMA,
								   &dma_task_id, sizeof(unsigned long));
	index = crc % RESTRACK_HASHSIZE;

	dlist_foreach (iter, &gcontext->restrack[index])
	{
		ResourceTracker *tracker
			= dlist_container(ResourceTracker, chain, iter.cur);

		if (tracker->crc == crc &&
			tracker->resclass == RESTRACK_CLASS__SSD2GPUDMA &&
			tracker->u.dma_task_id == dma_task_id)
		{
			dlist_delete(&tracker->chain);
			memset(tracker, 0, sizeof(ResourceTracker));
			dlist_push_head(&inactiveResourceTracker,
							&tracker->chain);
			return;
		}
	}
	elog(WARNING, "Bug? SSD-to-GPU Direct DMA (%p) was not tracked",
		 (void *)dma_task_id);
}

/*
 * ReleaseLocalResources - release all the private resources tracked by
 * the resource tracker of GpuContext
 */
static void
ReleaseLocalResources(GpuContext_v2 *gcontext, bool normal_exit)
{
	ResourceTracker *tracker;
	dlist_node		*dnode;
	CUresult		rc;
	int				i;

	/* close the socket if any */
	if (gcontext->sockfd != PGINVALID_SOCKET)
	{
		if (close(gcontext->sockfd) != 0)
			elog(WARNING, "failed on close(%d) socket: %m",
				 gcontext->sockfd);
		else
			elog(DEBUG2, "socket %d was closed", gcontext->sockfd);
		gcontext->sockfd = PGINVALID_SOCKET;
	}

	/*
	 * NOTE: RESTRACK_CLASS__SSD2GPUDMA (only available if NVMe-Strom is
	 * installed) must be released prior to any i/o mapped memory, because
	 * we have no way to cancel asynchronous DMA request once submitted,
	 * thus, release of i/o mapped memory prior to Async DMA will cause
	 * unexpected device memory corruption.
	 */
	for (i=0; i < RESTRACK_HASHSIZE; i++)
	{
		dlist_mutable_iter	iter;

		dlist_foreach_modify(iter, &gcontext->restrack[i])
		{
			tracker = dlist_container(ResourceTracker, chain, iter.cur);
			if (tracker->resclass != RESTRACK_CLASS__SSD2GPUDMA)
				continue;

			dlist_delete(&tracker->chain);
			// wait for completion of DMA
			elog(NOTICE, "SSD2GPU DMA %p is not completed",
				 (void *)tracker->u.dma_task_id);

			memset(tracker, 0, sizeof(ResourceTracker));
			dlist_push_head(&inactiveResourceTracker, &tracker->chain);
		}
	}

	/*
	 * OK, release other resources
	 */
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
						elog(WARNING, "GPU memory %p likely leaked",
							 (void *)tracker->u.devptr);
					/*
					 * normal device memory should be already released
					 * once CUDA context is destroyed
					 */
					if (!gpuserv_cuda_context)
						break;
					rc = cuMemFree(tracker->u.devptr);
					if (rc != CUDA_SUCCESS)
						elog(WARNING, "failed on cuMemFree(%p): %s",
							 (void *)tracker->u.devptr, errorText(rc));
					break;
				case RESTRACK_CLASS__GPUPROGRAM:
					if (normal_exit)
						elog(WARNING, "CUDA Program ID=%lu is likely leaked",
							 tracker->u.program_id);
					pgstrom_put_cuda_program(NULL, tracker->u.program_id);
					break;
				case RESTRACK_CLASS__IOMAPMEMORY:
					if (normal_exit)
						elog(WARNING, "I/O Mapped Memory %p likely leaked",
							 (void *)tracker->u.devptr);
					rc = gpuMemFreeIOMap(NULL, tracker->u.devptr);
					if (rc != CUDA_SUCCESS)
						elog(WARNING, "failed on gpuMemFreeIOMap(%p): %s",
							 (void *)tracker->u.devptr, errorText(rc));
					break;
				default:
					elog(WARNING, "Bug? unknown resource tracker class: %d",
						 (int)tracker->resclass);
					break;
			}
			memset(tracker, 0, sizeof(ResourceTracker));
			dlist_push_head(&inactiveResourceTracker, &tracker->chain);
		}
	}
}











/*
 * MasterGpuContext - acquire the persistent GpuContext; to allocate shared
 * memory segment valid until Postmaster die. No need to put.
 */
GpuContext_v2 *
MasterGpuContext(void)
{
	return &masterGpuContext;
}

/*
 * GetGpuContext - acquire a free GpuContext
 */
GpuContext_v2 *
AllocGpuContext(bool with_connection)
{
	GpuContext_v2  *gcontext = NULL;
	SharedGpuContext *shgcon;
	dlist_iter		iter;
	dlist_node	   *dnode;
	int				i;

	Assert(!IsGpuServerProcess());
	if (IsGpuServerProcess())
		elog(FATAL, "Bug? Only backend process can get a new GpuContext");

	/*
	 * Lookup an existing active GpuContext
	 */
	dlist_foreach(iter, &activeGpuContextList)
	{
		gcontext = dlist_container(GpuContext_v2, chain, iter.cur);

		if (gcontext->resowner == CurrentResourceOwner &&
			(with_connection
			 ? gcontext->sockfd != PGINVALID_SOCKET
			 : gcontext->sockfd == PGINVALID_SOCKET))
		{
			gcontext->refcnt++;
			return gcontext;
		}
	}

	/*
	 * Not found, let's create a new GpuContext
	 */
	if (dlist_is_empty(&inactiveGpuContextList))
	{
		Size	len = offsetof(GpuContext_v2, restrack[RESTRACK_HASHSIZE]);
		gcontext = MemoryContextAllocZero(TopMemoryContext, len);
	}
	else
	{
		dnode = dlist_pop_head_node(&inactiveGpuContextList);
		gcontext = (GpuContext_v2 *)
			dlist_container(GpuContext_v2, chain, dnode);
	}

	SpinLockAcquire(&sharedGpuContextHead->lock);
	if (dlist_is_empty(&sharedGpuContextHead->free_list))
	{
		SpinLockRelease(&sharedGpuContextHead->lock);
		dlist_push_head(&inactiveGpuContextList,
						&gcontext->chain);
		elog(ERROR, "No available SharedGpuContext item.");
	}
	dnode = dlist_pop_head_node(&sharedGpuContextHead->free_list);
	shgcon = (SharedGpuContext *)
		dlist_container(SharedGpuContext, chain, dnode);
	memset(&shgcon->chain, 0, sizeof(dlist_node));

	SpinLockRelease(&sharedGpuContextHead->lock);

	Assert(shgcon == &sharedGpuContextHead->context_array[shgcon->context_id]);
	shgcon->refcnt = 1;
	shgcon->device_id = -1;		/* set on GPU server attachment */
	shgcon->server = NULL;
	shgcon->backend = MyProc;
	dlist_init(&shgcon->dma_buffer_list);
	shgcon->num_async_tasks = 0;
	/* perfmon fields */
	memset(&shgcon->pfm, 0, sizeof(shgcon->pfm));
	shgcon->pfm.enabled = pgstrom_perfmon_enabled;

	/* init local GpuContext */
	gcontext->refcnt = 1;
	gcontext->sockfd = PGINVALID_SOCKET;
	gcontext->resowner = CurrentResourceOwner;
	gcontext->shgcon = shgcon;
	for (i=0; i < RESTRACK_HASHSIZE; i++)
		dlist_init(&gcontext->restrack[i]);
	dlist_push_head(&activeGpuContextList, &gcontext->chain);

	/*
	 * ------------------------------------------------------------------
	 * At this point, GpuContext can be reclaimed automatically because
	 * it is now already tracked by resource owner.
	 * ------------------------------------------------------------------
	 */

	/*
	 * Open the connection on demand, however, connection may often fails
	 * because of GPU server resources. In these cases, 'unconnected GPU
	 * context' shall be returned.
	 */
	if (with_connection)
		gpuservOpenConnection(gcontext);
	else
		gcontext->sockfd = PGINVALID_SOCKET;

	return gcontext;
}

/*
 * AttachGpuContext - attach a GPU server session on the supplied GpuContext
 * which is already acquired by a certain backend.
 */
GpuContext_v2 *
AttachGpuContext(pgsocket sockfd,
				 cl_int context_id,
				 BackendId backend_id,
				 cl_int device_id)
{
	GpuContext_v2	   *gcontext;
	SharedGpuContext   *shgcon;
	int					i;

	/* to be called by the GPU server process */
	Assert(IsGpuServerProcess());

	if (context_id >= numGpuContexts)
		elog(ERROR, "context_id (%d) is out of range", context_id);

	if (dlist_is_empty(&inactiveGpuContextList))
	{
		Size	len = offsetof(GpuContext_v2, restrack[RESTRACK_HASHSIZE]);
		gcontext = MemoryContextAllocZero(TopMemoryContext, len);
	}
	else
	{
		dlist_node *dnode = dlist_pop_head_node(&inactiveGpuContextList);
		gcontext = (GpuContext_v2 *)
			dlist_container(GpuContext_v2, chain, dnode);
	}

	shgcon = &sharedGpuContextHead->context_array[context_id];
	SpinLockAcquire(&shgcon->lock);
	/* sanity check */
	if (shgcon->refcnt == 0 ||		/* nobody own the GpuContext */
		shgcon->backend == NULL ||	/* no backend assigned yet */
		shgcon->server != NULL ||	/* a server is already assigned */
		shgcon->backend->backendId != backend_id)	/* wrong backend */
	{
		SpinLockRelease(&shgcon->lock);
		dlist_push_head(&inactiveGpuContextList, &gcontext->chain);
		elog(FATAL, "Bug? GpuContext (context_id=%d) has wrong state",
			 context_id);
	}
	shgcon->refcnt++;
	shgcon->device_id = device_id;
	shgcon->server = MyProc;
	shgcon->num_async_tasks = 0;
	SetLatch(&shgcon->backend->procLatch);
	SpinLockRelease(&shgcon->lock);

	gcontext->refcnt = 1;
	gcontext->sockfd = sockfd;
	gcontext->resowner = CurrentResourceOwner;
	gcontext->shgcon = shgcon;
	for (i=0; i < RESTRACK_HASHSIZE; i++)
		dlist_init(&gcontext->restrack[i]);
	dlist_push_head(&activeGpuContextList, &gcontext->chain);

	return gcontext;
}

/*
 * GetGpuContext - increment reference counter
 */
GpuContext_v2 *
GetGpuContext(GpuContext_v2 *gcontext)
{
	Assert(gcontext > 0);
	gcontext->refcnt++;
	return gcontext;
}

/*
 * PutSharedGpuContext - detach SharedGpuContext
 */
static void
PutSharedGpuContext(SharedGpuContext *shgcon)
{
	SpinLockAcquire(&shgcon->lock);
	Assert(shgcon->refcnt > 0);
	if (IsGpuServerProcess())
		shgcon->server = NULL;
	else
		shgcon->backend = NULL;

	if (--shgcon->refcnt > 0)
		SpinLockRelease(&shgcon->lock);
	else
	{
		Assert(!shgcon->server && !shgcon->backend);
		Assert(!shgcon->chain.prev && !shgcon->chain.next);
		SpinLockRelease(&shgcon->lock);

		/* release DMA buffer segments */
		dmaBufferFreeAll(shgcon);

		SpinLockAcquire(&sharedGpuContextHead->lock);
		dlist_push_head(&sharedGpuContextHead->free_list,
						&shgcon->chain);
		SpinLockRelease(&sharedGpuContextHead->lock);
	}
}

/*
 * PutGpuContext - detach GpuContext; to be called by only backend
 */
bool
PutGpuContext(GpuContext_v2 *gcontext)
{
	bool	is_last_one = false;

	Assert(gcontext->refcnt > 0);
	if (--gcontext->refcnt == 0)
	{
		is_last_one = true;

		dlist_delete(&gcontext->chain);
		ReleaseLocalResources(gcontext, true);
		PutSharedGpuContext(gcontext->shgcon);
		if (gcontext->sockfd != PGINVALID_SOCKET)
		{
			if (close(gcontext->sockfd) != 0)
				elog(WARNING, "failed on close socket(%d): %m",
					 gcontext->sockfd);
			else
				elog(DEBUG2, "socket %d was closed", gcontext->sockfd);
		}
		memset(gcontext, 0, sizeof(GpuContext_v2));
		dlist_push_head(&inactiveGpuContextList, &gcontext->chain);
	}
	return is_last_one;
}

/*
 * ForcePutGpuContext
 *
 * It detach GpuContext and release relevant resources regardless of
 * the reference count. Although it is fundamentally a danger operation,
 * we may need to keep the status of shared resource correct.
 * We intend this routine is called only when the final error cleanup
 * just before the process exit.
 */
bool
ForcePutGpuContext(GpuContext_v2 *gcontext)
{
	bool	is_last_one = (gcontext->refcnt < 2 ? true : false);

	dlist_delete(&gcontext->chain);
	ReleaseLocalResources(gcontext, false);
	PutSharedGpuContext(gcontext->shgcon);
	if (gcontext->sockfd != PGINVALID_SOCKET)
	{
		if (close(gcontext->sockfd) != 0)
			elog(WARNING, "failed on close socket(%d): %m",
				 gcontext->sockfd);
		else
			elog(DEBUG2, "socket %d was closed", gcontext->sockfd);
	}
	memset(gcontext, 0, sizeof(GpuContext_v2));
	dlist_push_head(&inactiveGpuContextList, &gcontext->chain);

	return is_last_one;
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
	dlist_mutable_iter	iter;

	if (phase == RESOURCE_RELEASE_BEFORE_LOCKS)
	{
		dlist_foreach_modify(iter, &activeGpuContextList)
		{
			GpuContext_v2  *gcontext = (GpuContext_v2 *)
				dlist_container(GpuContext_v2, chain, iter.cur);

			if (gcontext->resowner == CurrentResourceOwner)
			{
				if (isCommit)
					elog(WARNING, "GpuContext reference leak (refcnt=%d)",
						gcontext->refcnt);

				dlist_delete(&gcontext->chain);
				ReleaseLocalResources(gcontext, isCommit);
				PutSharedGpuContext(gcontext->shgcon);
				memset(gcontext, 0, sizeof(GpuContext_v2));
				dlist_push_head(&inactiveGpuContextList,
								&gcontext->chain);
			}
		}
	}
}

/*
 * gpucontext_proc_exit_cleanup - cleanup callback when process exit
 */
static void
gpucontext_proc_exit_cleanup(int code, Datum arg)
{
	GpuContext_v2  *gcontext;
	dlist_iter		iter;

	if (!IsUnderPostmaster)
		return;

	dlist_foreach(iter, &activeGpuContextList)
	{
		gcontext = dlist_container(GpuContext_v2, chain, iter.cur);

		elog(WARNING, "GpuContext (ID=%u) remained at pid=%u, cleanup",
			 gcontext->shgcon->context_id, MyProcPid);
		ForcePutGpuContext(gcontext);
	}
}

/*
 * pgstrom_startup_gpu_context
 */
static void
pgstrom_startup_gpu_context(void)
{
	SharedGpuContext *shgcon;
	Size		length;
	int			i;
	bool		found;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	/* sharedGpuContextHead */
	length = offsetof(SharedGpuContextHead, context_array[numGpuContexts]);
	sharedGpuContextHead = ShmemInitStruct("sharedGpuContextHead",
										   length, &found);
	Assert(!found);

	memset(sharedGpuContextHead, 0, length);
	SpinLockInit(&sharedGpuContextHead->lock);
	dlist_init(&sharedGpuContextHead->active_list);
	dlist_init(&sharedGpuContextHead->free_list);

	for (i=0; i < numGpuContexts; i++)
	{
		shgcon = &sharedGpuContextHead->context_array[i];
		shgcon->context_id = i;
		SpinLockInit(&shgcon->lock);
		shgcon->refcnt = 0;
		shgcon->backend = NULL;
		shgcon->server = NULL;
		dlist_init(&shgcon->dma_buffer_list);
		dlist_push_tail(&sharedGpuContextHead->free_list, &shgcon->chain);
	}

	/*
	 * construction of MasterGpuContext
	 */
	shgcon = &sharedGpuContextHead->master_context;
	shgcon->context_id = -1;
	SpinLockInit(&shgcon->lock);
	shgcon->refcnt = 1;
	dlist_init(&shgcon->dma_buffer_list);

	memset(&masterGpuContext, 0, sizeof(GpuContext_v2));
	masterGpuContext.refcnt = 1;
	masterGpuContext.sockfd = PGINVALID_SOCKET;
	masterGpuContext.resowner = NULL;
	masterGpuContext.shgcon = shgcon;
}

/*
 * pgstrom_init_gpu_context
 */
void
pgstrom_init_gpu_context(void)
{
	uint32		numBackends;	/* # of normal backends + background worker */

	/*
	 * Maximum number of GPU context - it is preferable to preserve
	 * enough number of SharedGpuContext items.
	 */
	numBackends = MaxConnections + max_worker_processes + 100;
	DefineCustomIntVariable("pg_strom.num_gpu_contexts",
							"maximum number of GpuContext",
							NULL,
							&numGpuContexts,
							numBackends,
							numBackends,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	/* initialization of GpuContext/ResTracker List */
	dlist_init(&activeGpuContextList);
	dlist_init(&inactiveGpuContextList);
	dlist_init(&inactiveResourceTracker);

	/* require the static shared memory */
	RequestAddinShmemSpace(MAXALIGN(offsetof(SharedGpuContextHead,
											 context_array[numGpuContexts])));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gpu_context;

	/* register the callback to clean up resources */
	RegisterResourceReleaseCallback(gpucontext_cleanup_callback, NULL);
	before_shmem_exit(gpucontext_proc_exit_cleanup, 0);
}
