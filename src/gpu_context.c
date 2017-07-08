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
#include <sys/epoll.h>

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
#define ACTIVE_GPU_CONTEXT_NSLOTS			768
static slock_t			activeGpuContextLock;
static dlist_head		activeGpuContextSlot[ACTIVE_GPU_CONTEXT_NSLOTS];
static __thread dlist_head inactiveResourceTracker;

/*
 * Resource tracker of GpuContext
 *
 * It enables to track various resources with GpuContext, to detect resource
 * leaks.
 */
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
		struct {
			CUdeviceptr	ptr;	/* RESTRACK_CLASS__GPUMEMORY */
			size_t		size;	/* RESTRACK_CLASS__IOMAPMEMORY */
		} devmem;
		ProgramId	program_id;	/* RESTRACK_CLASS__GPUPROGRAM */
		unsigned long dma_task_id; /* RESTRACK_CLASS__SSD2GPUDMA */
	} u;
} ResourceTracker;

static inline ResourceTracker *
resource_tracker_alloc(void)
{
	ResourceTracker	   *restrack;

	if (dlist_is_empty(&inactiveResourceTracker))
		restrack = calloc(1, sizeof(ResourceTracker));
	else
	{
		dlist_node *dnode = dlist_pop_head_node(&inactiveResourceTracker);
		restrack = dlist_container(ResourceTracker, chain, dnode);
		memset(restrack, 0, sizeof(ResourceTracker));
	}
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
 * gpuMemMaxAllocSize - hard limit for device memory allocation
 */
Size
gpuMemMaxAllocSize(void)
{
	/*
	 * NOTE: We don't allow to allocate more than 4GB as a single device
	 * memory chunk, because most of our GPU code uses cl_uint for offset
	 * values, so larger chunk than 4GB potentially leads a problem.
	 */
	return Min(devBaselineMemorySize / 3, 0x100000000UL);
}

/*
 * resource tracker for device memory
 */
CUresult
gpuMemAlloc(GpuContext_v2 *gcontext, CUdeviceptr *p_devptr, size_t bytesize)
{
	ResourceTracker *tracker;
	CUdeviceptr		devptr = 0UL;
    CUresult		rc;
	pg_crc32		crc;

	Assert(IsGpuServerProcess());
	if (bytesize > gpuMemMaxAllocSize())
		return CUDA_ERROR_INVALID_VALUE;

	rc = cuMemAlloc(&devptr, bytesize);
	if (rc == CUDA_SUCCESS)
	{
		tracker = resource_tracker_alloc();
		if (!tracker)
		{
			rc = cuMemFree(devptr);
			if (rc != CUDA_SUCCESS)
				wnotice("failed on cuMemFree(%p): %s",
						(void *)devptr, errorText(rc));
			werror("out of memory");
		}
		crc = resource_tracker_hashval(RESTRACK_CLASS__GPUMEMORY,
									   &devptr, sizeof(CUdeviceptr));
		tracker->crc = crc;
		tracker->resclass = RESTRACK_CLASS__GPUMEMORY;
		tracker->u.devmem.ptr = devptr;
		tracker->u.devmem.size = bytesize;
		SpinLockAcquire(&gcontext->lock);
		dlist_push_tail(&gcontext->restrack[crc % RESTRACK_HASHSIZE],
						&tracker->chain);
		SpinLockRelease(&gcontext->lock);
		*p_devptr = devptr;
	}
	return rc;
}

CUresult
gpuMemAllocManaged(GpuContext_v2 *gcontext,
				   CUdeviceptr *p_devptr, size_t bytesize, int flags)
{
	ResourceTracker *tracker;
	CUdeviceptr		devptr = 0UL;
    CUresult		rc;
	pg_crc32		crc;

	Assert(IsGpuServerProcess());
	if (bytesize > gpuMemMaxAllocSize())
		return CUDA_ERROR_INVALID_VALUE;

	rc = cuMemAllocManaged(&devptr, bytesize, flags);
	if (rc == CUDA_SUCCESS)
	{
		tracker = resource_tracker_alloc();
		if (!tracker)
		{
			rc = cuMemFree(devptr);
			if (rc != CUDA_SUCCESS)
				wnotice("failed on cuMemFree(%p): %s",
						(void *)devptr, errorText(rc));
			werror("out of memory");
		}
		crc = resource_tracker_hashval(RESTRACK_CLASS__GPUMEMORY,
									   &devptr, sizeof(CUdeviceptr));
		tracker->crc = crc;
		tracker->resclass = RESTRACK_CLASS__GPUMEMORY;
		tracker->u.devmem.ptr = devptr;
		tracker->u.devmem.size = bytesize;
		SpinLockAcquire(&gcontext->lock);
		dlist_push_tail(&gcontext->restrack[crc % RESTRACK_HASHSIZE],
						&tracker->chain);
		SpinLockRelease(&gcontext->lock);
		*p_devptr = devptr;
	}
	return rc;
}

CUresult
gpuMemFree(GpuContext_v2 *gcontext, CUdeviceptr devptr)
{
	dlist_head	   *restrack_list;
	dlist_iter		iter;
	pg_crc32		crc;
	CUresult		rc;

	crc = resource_tracker_hashval(RESTRACK_CLASS__GPUMEMORY,
								   &devptr, sizeof(CUdeviceptr));

	restrack_list = &gcontext->restrack[crc % RESTRACK_HASHSIZE];
	SpinLockAcquire(&gcontext->lock);
	dlist_foreach(iter, restrack_list)
	{
		ResourceTracker *tracker
			= dlist_container(ResourceTracker, chain, iter.cur);

		if (tracker->crc == crc &&
			tracker->resclass == RESTRACK_CLASS__GPUMEMORY &&
			tracker->u.devmem.ptr == devptr)
		{
			dlist_delete(&tracker->chain);
			SpinLockRelease(&gcontext->lock);

			memset(tracker, 0, sizeof(ResourceTracker));
			dlist_push_head(&inactiveResourceTracker,
							&tracker->chain);
			goto found;
		}
    }
	SpinLockRelease(&gcontext->lock);
	wnotice("Bug? device pointer %p was not tracked", (void *)devptr);
found:
	rc = cuMemFree(devptr);
	if (rc != CUDA_SUCCESS)
		wnotice("failed on cuMemFree(%p): %s", (void *)devptr, errorText(rc));
	return rc;
}

/*
 * resource tracker for GPU program
 */
bool
trackCudaProgram(GpuContext_v2 *gcontext, ProgramId program_id)
{
	ResourceTracker *tracker = resource_tracker_alloc();
	pg_crc32	crc;

	if (!tracker)
		return false;	/* out of memory */

	crc = resource_tracker_hashval(RESTRACK_CLASS__GPUPROGRAM,
								   &program_id, sizeof(ProgramId));
	tracker->crc = crc;
	tracker->resclass = RESTRACK_CLASS__GPUPROGRAM;
	tracker->u.program_id = program_id;
	SpinLockAcquire(&gcontext->lock);
	dlist_push_tail(&gcontext->restrack[crc % RESTRACK_HASHSIZE],
					&tracker->chain);
	SpinLockRelease(&gcontext->lock);
	return true;
}

void
untrackCudaProgram(GpuContext_v2 *gcontext, ProgramId program_id)
{
	dlist_head *restrack_list;
    dlist_iter	iter;
    pg_crc32	crc;

	crc = resource_tracker_hashval(RESTRACK_CLASS__GPUPROGRAM,
								   &program_id, sizeof(ProgramId));
	SpinLockAcquire(&gcontext->lock);
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
			SpinLockRelease(&gcontext->lock);
			memset(tracker, 0, sizeof(ResourceTracker));
			dlist_push_head(&inactiveResourceTracker,
							&tracker->chain);
			return;
		}
	}
	SpinLockRelease(&gcontext->lock);
	wnotice("Bug? CUDA Program %lu was not tracked", program_id);
}

/*
 * resource tracker for i/o mapped memory
 */
bool
trackIOMapMem(GpuContext_v2 *gcontext, CUdeviceptr devptr)
{
	ResourceTracker *tracker = resource_tracker_alloc();
	pg_crc32	crc;

	if (!tracker)
		return false;	/* out of memory */

	crc = resource_tracker_hashval(RESTRACK_CLASS__IOMAPMEMORY,
								   &devptr, sizeof(CUdeviceptr));
	tracker->crc = crc;
	tracker->resclass = RESTRACK_CLASS__IOMAPMEMORY;
	tracker->u.devmem.ptr = devptr;
	tracker->u.devmem.size = 0;

	SpinLockAcquire(&gcontext->lock);
	dlist_push_tail(&gcontext->restrack[crc % RESTRACK_HASHSIZE],
					&tracker->chain);
	SpinLockRelease(&gcontext->lock);
	return true;
}

void
untrackIOMapMem(GpuContext_v2 *gcontext, CUdeviceptr devptr)
{
	dlist_head *restrack_list;
    dlist_iter	iter;
    pg_crc32	crc;

	crc = resource_tracker_hashval(RESTRACK_CLASS__IOMAPMEMORY,
								   &devptr, sizeof(CUdeviceptr));
	restrack_list = &gcontext->restrack[crc % RESTRACK_HASHSIZE];
	SpinLockAcquire(&gcontext->lock);
	dlist_foreach (iter, restrack_list)
	{
		ResourceTracker *tracker
			= dlist_container(ResourceTracker, chain, iter.cur);

		if (tracker->crc == crc &&
			tracker->resclass == RESTRACK_CLASS__IOMAPMEMORY &&
			tracker->u.devmem.ptr == devptr)
		{
			dlist_delete(&tracker->chain);
			SpinLockRelease(&gcontext->lock);
			memset(tracker, 0, sizeof(ResourceTracker));
			dlist_push_head(&inactiveResourceTracker,
							&tracker->chain);
			return;
		}
	}
	SpinLockRelease(&gcontext->lock);
	wnotice("Bug? I/O Mapped Memory %p was not tracked", (void *)devptr);
}

/*
 * resource tracker for SSD-to-GPU Direct DMA task
 */
bool
trackSSD2GPUDMA(GpuContext_v2 *gcontext, unsigned long dma_task_id)
{
	ResourceTracker *tracker = resource_tracker_alloc();
	pg_crc32	crc;

	if (!tracker)
		return false;	/* out of memory */

	crc = resource_tracker_hashval(RESTRACK_CLASS__SSD2GPUDMA,
								   &dma_task_id, sizeof(unsigned long));
	tracker->crc = crc;
	tracker->resclass = RESTRACK_CLASS__SSD2GPUDMA;
	tracker->u.dma_task_id = dma_task_id;

	SpinLockAcquire(&gcontext->lock);
	dlist_push_tail(&gcontext->restrack[crc % RESTRACK_HASHSIZE],
					&tracker->chain);
	SpinLockRelease(&gcontext->lock);
	return true;
}

void
untrackSSD2GPUDMA(GpuContext_v2 *gcontext, unsigned long dma_task_id)
{
	dlist_head *restrack_list;
    dlist_iter	iter;
    pg_crc32	crc;

	crc = resource_tracker_hashval(RESTRACK_CLASS__SSD2GPUDMA,
								   &dma_task_id, sizeof(unsigned long));
	restrack_list = &gcontext->restrack[crc % RESTRACK_HASHSIZE];
	SpinLockAcquire(&gcontext->lock);
	dlist_foreach (iter, restrack_list)
	{
		ResourceTracker *tracker
			= dlist_container(ResourceTracker, chain, iter.cur);

		if (tracker->crc == crc &&
			tracker->resclass == RESTRACK_CLASS__SSD2GPUDMA &&
			tracker->u.dma_task_id == dma_task_id)
		{
			dlist_delete(&tracker->chain);
			SpinLockRelease(&gcontext->lock);
			memset(tracker, 0, sizeof(ResourceTracker));
			dlist_push_head(&inactiveResourceTracker,
							&tracker->chain);
			return;
		}
	}
	SpinLockRelease(&gcontext->lock);
	wnotice("Bug? SSD-to-GPU Direct DMA (%p) was not tracked",
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
			wnotice("failed on close(%d) socket: %m", gcontext->sockfd);
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
			gpuMemCopyFromSSDWaitRaw(tracker->u.dma_task_id);
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
						wnotice("GPU memory %p likely leaked",
								(void *)tracker->u.devmem.ptr);
					/*
					 * normal device memory should be already released
					 * once CUDA context is destroyed
					 */
					if (!gpuserv_cuda_context)
						break;
					rc = cuMemFree(tracker->u.devmem.ptr);
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
				case RESTRACK_CLASS__IOMAPMEMORY:
					if (normal_exit)
						wnotice("I/O Mapped Memory %p likely leaked",
								(void *)tracker->u.devmem.ptr);
					rc = gpuMemFreeIOMap(NULL, tracker->u.devmem.ptr);
					if (rc != CUDA_SUCCESS)
						wnotice("failed on gpuMemFreeIOMap(%p): %s",
								(void *)tracker->u.devmem.ptr,
								errorText(rc));
					break;
				default:
					wnotice("Bug? unknown resource tracker class: %d",
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

	if (IsGpuServerProcess())
		elog(FATAL, "Bug? Only backend process can get a new GpuContext");

	/*
	 * Lookup an existing active GpuContext
	 */
	SpinLockAcquire(&activeGpuContextLock);
	dlist_foreach(iter, &activeGpuContextSlot[0])
	{
		gcontext = dlist_container(GpuContext_v2, chain, iter.cur);

		if (gcontext->resowner == CurrentResourceOwner &&
			(with_connection
			 ? gcontext->sockfd != PGINVALID_SOCKET
			 : gcontext->sockfd == PGINVALID_SOCKET))
		{
			SpinLockRelease(&activeGpuContextLock);
			pg_atomic_fetch_add_u32(&gcontext->refcnt, 1);
			return gcontext;
		}
	}
	SpinLockRelease(&activeGpuContextLock);

	/*
	 * Not found, let's create a new GpuContext
	 */
	gcontext = calloc(1, sizeof(GpuContext_v2));
	if (!gcontext)
		elog(ERROR, "out of memory");

	SpinLockAcquire(&sharedGpuContextHead->lock);
	if (dlist_is_empty(&sharedGpuContextHead->free_list))
	{
		SpinLockRelease(&sharedGpuContextHead->lock);
		free(gcontext);
		elog(ERROR, "No available SharedGpuContext item.");
	}
	dnode = dlist_pop_head_node(&sharedGpuContextHead->free_list);
	shgcon = (SharedGpuContext *)
		dlist_container(SharedGpuContext, chain, dnode);
	memset(&shgcon->chain, 0, sizeof(dlist_node));
	SpinLockRelease(&sharedGpuContextHead->lock);

	shgcon->server = NULL;
	shgcon->backend = MyProc;
	SpinLockInit(&shgcon->lock);
	shgcon->refcnt = 1;
	dlist_init(&shgcon->dma_buffer_list);
	shgcon->num_async_tasks = 0;

	/* init local GpuContext */
	gcontext->sockfd = PGINVALID_SOCKET;
	gcontext->resowner = CurrentResourceOwner;
	gcontext->shgcon = shgcon;
	pg_atomic_init_u32(&gcontext->refcnt, 1);
	pg_atomic_init_u32(&gcontext->is_unlinked, 0);
	SpinLockInit(&gcontext->lock);
	for (i=0; i < RESTRACK_HASHSIZE; i++)
		dlist_init(&gcontext->restrack[i]);

	SpinLockAcquire(&activeGpuContextLock);
	dlist_push_head(&activeGpuContextSlot[0], &gcontext->chain);
	SpinLockRelease(&activeGpuContextLock);

	/*
	 * ------------------------------------------------------------------
	 * At this point, GpuContext can be reclaimed automatically because
	 * it is now already tracked by resource owner.
	 * ------------------------------------------------------------------
	 */
	if (with_connection)
		gpuservOpenConnection(gcontext);

	return gcontext;
}

/*
 * AttachGpuContext - attach a GPU server session on the supplied GpuContext
 * which is already acquired by a certain backend.
 */
GpuContext_v2 *
AttachGpuContext(pgsocket sockfd, SharedGpuContext *shgcon, int epoll_fd)
{
	GpuContext_v2  *gcontext;
	struct epoll_event ep_event;
	int				i;

	/* to be called by the GPU server process */
	if (!IsGpuServerProcess())
		wfatal("Bug? backend tried to attach GPU context");

	/* allocation of a local GpuContext */
	gcontext = calloc(1, sizeof(GpuContext_v2));
	if (!gcontext)
		werror("out of memory");

	gcontext->sockfd = sockfd;
	gcontext->resowner = CurrentResourceOwner;
	pg_atomic_init_u32(&gcontext->refcnt, 1);
	pg_atomic_init_u32(&gcontext->is_unlinked, 0);
	SpinLockInit(&gcontext->lock);
	gcontext->shgcon = shgcon;
	for (i=0; i < RESTRACK_HASHSIZE; i++)
		dlist_init(&gcontext->restrack[i]);

	/* add gcontext to epoll fd */
	ep_event.events = EPOLLIN | EPOLLET;
	ep_event.data.fd = sockfd;
	if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, sockfd, &ep_event) < 0)
	{
		free(gcontext);
		werror("failed on epoll_ctl(EPOLL_CTL_ADD): %m");
	}

	SpinLockAcquire(&shgcon->lock);
	Assert(shgcon->refcnt > 0 &&		/* someone must own the GpuContext */
		   shgcon->backend != NULL &&	/* backend must be assigned on */
		   shgcon->server == NULL);		/* no server should be assigned yet */
	shgcon->refcnt++;
	shgcon->server = MyProc;
	shgcon->num_async_tasks = 0;
	SpinLockRelease(&shgcon->lock);

	SpinLockAcquire(&activeGpuContextLock);
	dlist_push_head(&activeGpuContextSlot[sockfd % ACTIVE_GPU_CONTEXT_NSLOTS],
					&gcontext->chain);
	SpinLockRelease(&activeGpuContextLock);

	return gcontext;
}

/*
 * GetGpuContext - increment reference counter
 */
GpuContext_v2 *
GetGpuContext(GpuContext_v2 *gcontext)
{
	uint32		oldcnt __attribute__((unused));

	oldcnt = pg_atomic_fetch_add_u32(&gcontext->refcnt, 1);
	Assert(oldcnt > 0);

	return gcontext;
}

/*
 * GetGpuContextBySockfd - Get a GpuContext which hold the supplied sockfd
 */
GpuContext_v2 *
GetGpuContextBySockfd(pgsocket sockfd)
{
	dlist_head	   *dhead;
	dlist_iter		iter;
	GpuContext_v2  *gcontext;

	if (!IsGpuServerProcess())
		elog(FATAL, "Bug? GetGpuContextBySockfd called on backend");

	SpinLockAcquire(&activeGpuContextLock);
	dhead = &activeGpuContextSlot[sockfd % ACTIVE_GPU_CONTEXT_NSLOTS];
	dlist_foreach(iter, dhead)
	{
		gcontext = dlist_container(GpuContext_v2, chain, iter.cur);
		if (gcontext->sockfd == sockfd)
		{
			GetGpuContext(gcontext);
			SpinLockRelease(&activeGpuContextLock);

			return gcontext;
		}
	}
	SpinLockRelease(&activeGpuContextLock);

	return NULL;
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
void
PutGpuContext(GpuContext_v2 *gcontext)
{
	uint32		newcnt __attribute__((unused));

	newcnt = pg_atomic_sub_fetch_u32(&gcontext->refcnt, 1);
	if (newcnt == 0)
	{
		SpinLockAcquire(&activeGpuContextLock);
		dlist_delete(&gcontext->chain);
		SpinLockRelease(&activeGpuContextLock);

		if (IsGpuServerProcess())
			gpuservClenupGpuContext(gcontext);
		ReleaseLocalResources(gcontext, true);
		PutSharedGpuContext(gcontext->shgcon);
#ifdef PGSTROM_DEBUG
		if (gcontext->debug_tv1 > 0)
			wnotice("%u: debug1=%.2f", MyProcPid,
					((double)gcontext->debug_tv1) / 1000000.0);
		if (gcontext->debug_tv2 > 0)
			wnotice("%u: debug2=%.2f", MyProcPid,
					((double)gcontext->debug_tv2) / 1000000.0);
		if (gcontext->debug_tv3 > 0)
			wnotice("%u: debug3=%.2f", MyProcPid,
					((double)gcontext->debug_tv3) / 1000000.0);
		if (gcontext->debug_tv4 > 0)
			wnotice("%u: debug4=%.2f", MyProcPid,
					((double)gcontext->debug_tv4) / 1000000.0);
#endif
		free(gcontext);
	}
}

/*
 * ForcePutAllGpuContext
 *
 * It detach GpuContext and release relevant resources regardless of
 * the reference count. Although it is fundamentally a danger operation,
 * we may need to keep the status of shared resource correct.
 * We intend this routine is called only when the final error cleanup
 * just before the process exit.
 */
void
ForcePutAllGpuContext(void)
{
	int			i;

	if (IsGpuServerProcess() < 0)
		elog(FATAL, "Bug? ForcePutAllGpuContext is called under multi-thread process");

	for (i=0; i < ACTIVE_GPU_CONTEXT_NSLOTS; i++)
	{
		dlist_head	   *dhead = &activeGpuContextSlot[i];
		dlist_node	   *dnode;
		GpuContext_v2  *gcontext;

		SpinLockAcquire(&activeGpuContextLock);
		while (!dlist_is_empty(dhead))
		{
			dnode = dlist_pop_head_node(dhead);
			gcontext = dlist_container(GpuContext_v2, chain, dnode);
			SpinLockRelease(&activeGpuContextLock);

			ReleaseLocalResources(gcontext, false);
			PutSharedGpuContext(gcontext->shgcon);
			wnotice("GpuContext remained at pid=%u, cleanup\n", MyProcPid);
			free(gcontext);

			SpinLockAcquire(&activeGpuContextLock);
		}
		SpinLockRelease(&activeGpuContextLock);
	}
}

/*
 * GpuContextIsEstablished
 *
 * It checks whether GpuContext is established; it means both client and
 * server are still connected via local socket, and both of them look at.
 */
bool
GpuContextIsEstablished(GpuContext_v2 *gcontext)
{
	SharedGpuContext   *shgcon = gcontext->shgcon;
	bool				retval;

	SpinLockAcquire(&shgcon->lock);
	retval = (shgcon->server && shgcon->backend);
	SpinLockRelease(&shgcon->lock);

	return retval;
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
	int		i, n = (!IsGpuServerProcess() ? 1 : ACTIVE_GPU_CONTEXT_NSLOTS);

	if (phase != RESOURCE_RELEASE_BEFORE_LOCKS)
		return;

	for (i=0; i < n; i++)
	{
		dlist_head	   *dhead = &activeGpuContextSlot[i];
		dlist_mutable_iter iter;

		SpinLockAcquire(&activeGpuContextLock);
		dlist_foreach_modify(iter, dhead)
		{
			GpuContext_v2  *gcontext = (GpuContext_v2 *)
				dlist_container(GpuContext_v2, chain, iter.cur);

			if (gcontext->resowner == CurrentResourceOwner)
			{
				if (isCommit)
					wnotice("GpuContext reference leak (refcnt=%d)",
							pg_atomic_read_u32(&gcontext->refcnt));

				dlist_delete(&gcontext->chain);
				ReleaseLocalResources(gcontext, isCommit);
				PutSharedGpuContext(gcontext->shgcon);
				free(gcontext);
			}
		}
		SpinLockRelease(&activeGpuContextLock);
	}
}

/*
 * gpucontext_proc_exit_cleanup - cleanup callback when process exit
 */
static void
gpucontext_proc_exit_cleanup(int code, Datum arg)
{
	if (!IsUnderPostmaster)
		return;
	ForcePutAllGpuContext();
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
	void	   *ptr;

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
	SpinLockInit(&shgcon->lock);
	shgcon->refcnt = 1;
	dlist_init(&shgcon->dma_buffer_list);

	memset(&masterGpuContext, 0, sizeof(GpuContext_v2));
	masterGpuContext.sockfd = PGINVALID_SOCKET;
	masterGpuContext.resowner = NULL;
	masterGpuContext.shgcon = shgcon;
	pg_atomic_init_u32(&masterGpuContext.refcnt, 1);
	SpinLockInit(&masterGpuContext.lock);
	for (i=0; i < RESTRACK_HASHSIZE; i++)
		dlist_init(&masterGpuContext.restrack[i]);

	/*
	 * pre-fault of the first segment of DMA buffer
	 */
	ptr = dmaBufferAlloc(MasterGpuContext(), BLCKSZ);
	if (!ptr)
		elog(ERROR, "failed on pre-fault of DMA buffer allocation");
	dmaBufferFree(ptr);
}

/*
 * pgstrom_init_gpu_context
 */
void
pgstrom_init_gpu_context(void)
{
	uint32		numBackends;	/* # of normal backends + background worker */
	int			i;

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
	SpinLockInit(&activeGpuContextLock);
	for (i=0; i < ACTIVE_GPU_CONTEXT_NSLOTS; i++)
		dlist_init(&activeGpuContextSlot[i]);
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
