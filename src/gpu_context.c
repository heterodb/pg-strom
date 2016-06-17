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

static dlist_head		localGpuContextList;
static dlist_head		inactiveGpuContextList;

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
	pgsocket		sockfd;

	if (!IsGpuServerProcess())
		elog(FATAL, "Bug? Only backend process can get a new GpuContext");

	/*
	 * Lookup an existing active GpuContext
	 */
	dlist_foreach(iter, &localGpuContextList)
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
		gcontext = MemoryContextAlloc(CacheMemoryContext,
									  sizeof(GpuContext_v2));
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
	dlist_push_head(&localGpuContextList, &gcontext->chain);
	/*
	 * At this point, we can release GpuContext automatically because
	 * it is already tracked by resource owner.
	 */

#if 0
	/* try to open the connection for GpuServer */
	sockfd = gpuserv_open_connection();
	if (sockfd == PGINVALID_SOCKET)
	{
		PutGpuContext(gcontext);
		return NULL;
	}
	gcontext->sockfd = sockfd;
#endif
	return gcontext;
}

/*
 * AttachGpuContext - attach a GPU server session on the supplied GpuContext
 * which is already acquired by a certain backend.
 */
SharedGpuContext *
AttachGpuContext(cl_int context_id, BackendId backend_id)
{
	SharedGpuContext   *shgcon;

	if (context_id >= numGpuContexts)
		elog(ERROR, "supplied context_id (%d) is out of range", context_id);

	shgcon = &sharedGpuContextHead->context_array[context_id];
	SpinLockAcquire(&shgcon->lock);
	if (shgcon->refcnt == 0 ||		/* nobody own the GpuContext */
		shgcon->backend == NULL ||	/* no backend assigned yet */
		shgcon->server != NULL ||	/* server already assigned */
		shgcon->backend->backendId != backend_id)
	{
		SpinLockRelease(&shgcon->lock);
		elog(ERROR, "supplied context_id (%d) has strange state", context_id);
	}
	shgcon->refcnt++;
	shgcon->server = MyProc;
	SetLatch(&shgcon->backend->procLatch);
	SpinLockRelease(&shgcon->lock);

	return shgcon;
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
		if (gcontext->sockfd != PGINVALID_SOCKET)
			close(gcontext->sockfd);
		PutSharedGpuContext(gcontext->shgcon);

		memset(gcontext, 0, sizeof(GpuContext_v2));
		dlist_push_head(&inactiveGpuContextList,
						&gcontext->chain);
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
		dlist_foreach(iter, &localGpuContextList)
		{
			GpuContext_v2  *gcontext = (GpuContext_v2 *)
				dlist_container(GpuContext_v2, chain, iter.cur);

			if (gcontext->resowner == CurrentResourceOwner)
			{
				if (isCommit)
					elog(WARNING, "GpuContext reference leak (refcnt=%d)",
						gcontext->refcnt);

				dlist_delete(&gcontext->chain);
				if (gcontext->sockfd != PGINVALID_SOCKET)
					close(gcontext->sockfd);
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

	dlist_foreach(iter, &localGpuContextList)
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

	/* initialization of resource tracker */
	dlist_init(&localGpuContextList);
	dlist_init(&inactiveGpuContextList);

	/* require the static shared memory */
	RequestAddinShmemSpace(MAXALIGN(offsetof(SharedGpuContextHead,
											 context_array[numGpuContexts])));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gpu_context;

	/* register the callback to clean up resources */
	RegisterResourceReleaseCallback(gpucontext_cleanup_callback, NULL);
	before_shmem_exit(gpucontext_proc_exit_cleanup, 0);
}
