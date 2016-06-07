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
#include "utils/memutils.h"
#include "utils/resowner.h"
#include "pg_strom.h"

typedef struct SharedGpuContextHead
{
	slock_t			lock;
	dlist_head		active_list;
	dlist_head		free_list;
	SharedGpuContext context_array[FLEXIBLE_ARRAY_MEMBER];
} SharedGpuContextHead;

/* static variables */
static shmem_startup_hook_type shmem_startup_hook_next = NULL;
static SharedGpuContextHead *sharedGpuContextHead = NULL;
static int			numGpuContexts;	/* GUC */

static dlist_head		localGpuContextList;
static dlist_head		inactiveGpuContextList;

/*
 * GetGpuContext - acquire a free GpuContext
 */
static GpuContext_v2 *
GetGpuContext(void)
{
	GpuContext_v2  *gcontext = NULL;
	SharedGpuContext *shgcon;
	dlist_iter		iter;
	dlist_node	   *dnode;

	if (!IsGpuServerProcess())
		elog(FATAL, "Bug? Only backend process can get a new GpuContext");

	/*
	 * Find out an existing local GpuContext
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
	 * OK, let's create a new GpuContext
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
	 * After the point, we can release GpuContext automatically because
	 * it is already tracked by resource owner.
	 */

	/* try to open the connection for GpuServer */
	sockfd = gpuserv_open_connection();
	if (sockfd == PGINVALID_SOCKET)
	{
		PutGpuContext(gcontext);
		return NULL;
	}
	gcontext->sockfd = sockfd;

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

	if (cmd->context_id >= numGpuContexts)
		elog(ERROR, "supplied context_id (%d) is out of range", context_id);

	shgcon = &sharedGpuContextHead->context_array[context_id];
	SpinLockAcquire(&shgcon->lock);
	if (shgcon->refcnt == 0 ||		/* nobody own the GpuContext */
		shgcon->backend == NULL ||	/* no backend assigned yet */
		shgcon->server != NULL ||	/* server already assigned */
		shgcon->backend->backendId != cmd->open.backend_id)
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

		/*
		 * TODO: Release shared memory here
		 */
		dlist_push_head(&sharedGpuContextHead->free_list, &shgcon->chain);
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
 * pgstrom_startup_gpu_context
 */
void
pgstrom_startup_gpu_context(void)
{
	Size	length;
	int		i;

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
		SharedGpuContext   *shgcon = &sharedGpuContextHead->context_array[i];

		shgcon->context_id = i;
		shgcon->refcnt = 0;
		shgcon->backend = NULL;
		shgcon->server = NULL;
		dlist_push_tail(&sharedGpuContextHead->free_list, &shgcon->chain);
	}
}

/*
 * pgstrom_init_gpu_context
 */
void
pgstrom_init_gpu_context(void)
{
	uint32		TotalProcs;	/* see InitProcGlobal() */
	int			i;

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

	for (i=0; i < ResourceTrackerNumSlots; i++)
		dlist_init(&resource_tracker_slots[i]);
	dlist_init(&resource_tracker_free);

	/* require the static shared memory */
	RequestAddinShmemSpace(MAXALIGN(offsetof(SharedGpuContextHead,
											 context_array[numGpuContexts])));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gpu_context;

	/* register the callback */
	RegisterResourceReleaseCallback(gpucontext_cleanup_callback, NULL);
}
