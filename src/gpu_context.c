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
#define RESTRACK_HASHSIZE		27

#define RESTRACK_CLASS__FILEDESC		1
#define RESTRACK_CLASS__GPUMEMORY		2
#define RESTRACK_CLASS__GPUPROGRAM		3
typedef struct ResourceTracker
{
	dlist_node	chain;
	pg_crc32	crc;
	cl_int		resclass;
	union {
		int			fdesc;		/* RESTRACK_CLASS__FILEDESC */
		CUdeviceptr	devptr;		/* RESTRACK_CLASS__GPUMEMORY */
		ProgramId	program_id;	/* RESTRACK_CLASS__GPUPROGRAM */
	} u;
} ResourceTracker;

static inline ResourceTracker *
resource_tracker_alloc(void)
{
	ResourceTracker	   *restrack;

	if (dlist_is_empty(&inactiveResourceTracker))
		return MemoryContextAllocZero(CacheMemoryContext,
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
 * resource tracker for file descriptor
 */
void
trackFileDesc(GpuContext_v2 *gcontext, int fdesc)
{
	ResourceTracker *tracker = resource_tracker_alloc();
	pg_crc32	crc;

	crc = resource_tracker_hashval(RESTRACK_CLASS__FILEDESC,
								   &fdesc, sizeof(int));
	tracker->crc = crc;
	tracker->resclass = RESTRACK_CLASS__FILEDESC;
	tracker->u.fdesc = fdesc;

	dlist_push_tail(&gcontext->restrack[crc % RESTRACK_HASHSIZE],
					&tracker->chain);
}

int
closeFileDesc(GpuContext_v2 *gcontext, int fdesc)
{
	dlist_head *restrack_list;
	dlist_iter	iter;
	pg_crc32	crc;

	crc = resource_tracker_hashval(RESTRACK_CLASS__FILEDESC,
								   &fdesc, sizeof(int));
	restrack_list = &gcontext->restrack[crc % RESTRACK_HASHSIZE];

	dlist_foreach(iter, restrack_list)
	{
		ResourceTracker *tracker
			= dlist_container(ResourceTracker, chain, iter.cur);

		if (tracker->crc == crc &&
			tracker->resclass == RESTRACK_CLASS__FILEDESC &&
			tracker->u.fdesc == fdesc)
		{
			dlist_delete(&tracker->chain);
			memset(tracker, 0, sizeof(ResourceTracker));
			dlist_push_head(&inactiveResourceTracker,
							&tracker->chain);
			return close(fdesc);
		}
	}
	elog(WARNING, "Bug? file-descriptor %d was not tracked", fdesc);

	return close(fdesc);
}

/*
 * resource tracker for device memory
 */
CUresult
gpuMemAlloc_v2(GpuContext_v2 *gcontext, CUdeviceptr *p_devptr, size_t bytesize)
{
	ResourceTracker *tracker;
	CUdeviceptr	devptr;
    CUresult	rc;
	pg_crc32	crc;

	Assert(IsGpuServerProcess());
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
	return rc;
}

CUresult
gpuMemFree_v2(GpuContext_v2 *gcontext, CUdeviceptr devptr)
{
	dlist_head *restrack_list;
	dlist_iter	iter;
	pg_crc32	crc;

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
			return cuMemFree(devptr);
        }
    }
    elog(WARNING, "Bug? device pointer %p was not tracked", (void *)devptr);

	return cuMemFree(devptr);
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
	pgstrom_put_cuda_program(NULL, program_id);
}

/*
 * ReleaseLocalResources - release all the private resources tracked by
 * the resource tracker of GpuContext
 */
static void
ReleaseLocalResources(GpuContext_v2 *gcontext)
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
	}

	for (i=0; i < RESTRACK_HASHSIZE; i++)
	{
		while (!dlist_is_empty(&gcontext->restrack[i]))
		{
			dnode = dlist_pop_head_node(&gcontext->restrack[i]);
			tracker = dlist_container(ResourceTracker, chain, dnode);

			switch (tracker->resclass)
			{
				case RESTRACK_CLASS__FILEDESC:
					if (close(tracker->u.fdesc) != 0)
						elog(WARNING, "failed on close(%d): %m",
							 tracker->u.fdesc);
					break;

				case RESTRACK_CLASS__GPUMEMORY:
					rc = cuMemFree(tracker->u.devptr);
					if (rc != CUDA_SUCCESS)
						elog(WARNING, "failed on cuMemFree(%p): %s",
							 (void *)tracker->u.devptr, errorText(rc));
					break;
				case RESTRACK_CLASS__GPUPROGRAM:
					untrackCudaProgram(NULL, tracker->u.program_id);
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
GetGpuContext(void)
{
	GpuContext_v2  *gcontext = NULL;
	SharedGpuContext *shgcon;
	dlist_iter		iter;
	dlist_node	   *dnode;
	int				i;

	if (!IsGpuServerProcess())
		elog(FATAL, "Bug? Only backend process can get a new GpuContext");

	/*
	 * Lookup an existing active GpuContext
	 */
	dlist_foreach(iter, &activeGpuContextList)
	{
		gcontext = dlist_container(GpuContext_v2, chain, iter.cur);

		if (gcontext->resowner == CurrentResourceOwner)
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
		gcontext = MemoryContextAllocZero(CacheMemoryContext, len);
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
	SpinLockRelease(&sharedGpuContextHead->lock);

	Assert(shgcon == &sharedGpuContextHead->context_array[shgcon->context_id]);
	shgcon->refcnt = 1;
	shgcon->server = NULL;
	shgcon->backend = MyProc;
	shgcon->error_code = 0;

	gcontext->refcnt = 1;
	gcontext->sockfd = PGINVALID_SOCKET;
	gcontext->resowner = CurrentResourceOwner;
	gcontext->shgcon = shgcon;
	for (i=0; i < RESTRACK_HASHSIZE; i++)
		dlist_init(&gcontext->restrack[i]);
	dlist_push_head(&activeGpuContextList, &gcontext->chain);

	/*
	 * At this point, GpuContext can be reclaimed automatically because
	 * it is now already tracked by resource owner.
	 */

	/* try to open the connection to GpuServer */
	if (!gpuservOpenConnection(gcontext))
	{
		PutGpuContext(gcontext);
		return NULL;
	}
	Assert(gcontext->sockfd != PGINVALID_SOCKET);
	return gcontext;
}

/*
 * AttachGpuContext - attach a GPU server session on the supplied GpuContext
 * which is already acquired by a certain backend.
 */
GpuContext_v2 *
AttachGpuContext(pgsocket sockfd, cl_int context_id, BackendId backend_id)
{
	GpuContext_v2	   *gcontext;
	SharedGpuContext   *shgcon;
	int					i;

	/* to be called by the GPU server process */
	Assert(IsGpuServerProcess());
	/* GPU server should have up to one GpuContext at a time */
	Assert(dlist_is_empty(&activeGpuContextList));

	if (context_id >= numGpuContexts)
		elog(ERROR, "context_id (%d) is out of range", context_id);

	if (dlist_is_empty(&inactiveGpuContextList))
	{
		Size	len = offsetof(GpuContext_v2, restrack[RESTRACK_HASHSIZE]);
		gcontext = MemoryContextAllocZero(CacheMemoryContext, len);
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
		shgcon->backend->backendId != backend_id)	/* wrond backend */
	{
		SpinLockRelease(&shgcon->lock);
		dlist_push_head(&inactiveGpuContextList, &gcontext->chain);
		elog(ERROR, "Bug? GpuContext (context_id=%d) has wrond state",
			 context_id);
	}
	shgcon->refcnt++;
	shgcon->server = MyProc;
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
 * PutSharedGpuContext - detach SharedGpuContext
 */
void
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
		dlist_delete(&shgcon->chain);
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
	Assert(!IsGpuServerProcess());
	Assert(gcontext->refcnt > 0);
	if (--gcontext->refcnt == 0)
	{
		dlist_delete(&gcontext->chain);
		ReleaseLocalResources(gcontext);
		PutSharedGpuContext(gcontext->shgcon);
		memset(gcontext, 0, sizeof(GpuContext_v2));
		dlist_push_head(&inactiveGpuContextList, &gcontext->chain);
	}
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
	dlist_iter		iter;

	if (phase == RESOURCE_RELEASE_BEFORE_LOCKS)
	{
		dlist_foreach(iter, &activeGpuContextList)
		{
			GpuContext_v2  *gcontext = (GpuContext_v2 *)
				dlist_container(GpuContext_v2, chain, iter.cur);

			if (gcontext->resowner == CurrentResourceOwner)
			{
				if (isCommit)
					elog(WARNING, "GpuContext reference leak (refcnt=%d)",
						gcontext->refcnt);

				dlist_delete(&gcontext->chain);
				ReleaseLocalResources(gcontext);
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
		PutSharedGpuContext(gcontext->shgcon);
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
	uint32		TotalProcs;	/* see InitProcGlobal() */

	/*
	 * Maximum number of GPU context - it is preferable to preserve
	 * enough number of SharedGpuContext items.
	 */
	TotalProcs = MaxBackends + NUM_AUXILIARY_PROCS + max_prepared_xacts;
	DefineCustomIntVariable("pg_strom.num_gpu_contexts",
							"maximum number of GpuContext",
							NULL,
							&numGpuContexts,
							2 * TotalProcs,
							MaxBackends,
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
