/*
 * restrack.c
 *
 * Resource tracking for proper error handling 
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "utils/memutils.h"
#include "utils/resowner.h"
#include "pg_strom.h"

/*
 * Object classes extended from pgstrom_message has to be tracked.
 * Once a message is enqueued, it shall be processed by OpenCL server
 * asynchronously. It means, backend process may raise an error during
 * the server job is in-progress, and reference counter of message objects
 * needs to be controled in proper way.
 * pgstrom_restrack_callback is a resource owner callback to be invoked
 * when transaction is committed or aborted.
 */

typedef struct {
	dlist_node		chain;
	ResourceOwner	owner;
	StromTag	   *object;
} tracker_entry;

#define RESTRACK_HASHSZ		1051
#if SIZEOF_VOID_P < 8
#define tracker_hash(msg)	(((uintptr_t)msg >> 2) % RESTRACK_HASHSZ)
#else
#define tracker_hash(msg)	(((uintptr_t)msg >> 3) % RESTRACK_HASHSZ)
#endif

#define IS_TRACKABLE_OBJECT(stag)					\
	(*((StromTag *)stag) == StromTag_MsgQueue ||	\
	 *((StromTag *)stag) == StromTag_DevProgram ||	\
	 *((StromTag *)stag) == StromTag_GpuScan ||		\
	 *((StromTag *)stag) == StromTag_GpuSort ||		\
	 *((StromTag *)stag) == StromTag_HashJoin||		\
	 *((StromTag *)stag) == StromTag_TestMessage)

static dlist_head		tracker_free;
static dlist_head		tracker_slot[RESTRACK_HASHSZ];
static MemoryContext	ResTrackContext;

static void
pgstrom_restrack_callback(ResourceReleasePhase phase,
						  bool is_commit,
						  bool is_toplevel,
						  void *arg)
{
	tracker_entry  *entry;
	int		i;

	if (phase != RESOURCE_RELEASE_AFTER_LOCKS)
		return;

	if (is_commit)
	{
#ifdef PGSTROM_DEBUG
		/*
		 * If transaction is committed as usual, shared objects shall be
		 * released in the regular code path. So, we should not need to
		 * release these objects in the resource-release callback.
		 */
		for (i=0; i < RESTRACK_HASHSZ; i++)
		{
			dlist_iter	iter;

			dlist_foreach(iter, &tracker_slot[i])
			{
				entry = dlist_container(tracker_entry, chain, iter.cur);
				Assert(entry->owner != CurrentResourceOwner);
			}
		}
#endif
		return;
	}

	/*
	 * In case of transaction abort, we decrement reference counter of
	 * tracked objects. Some of them are released immediately, or some
	 * other will be released later by OpenCL server.
	 */
	for (i=0; i < RESTRACK_HASHSZ; i++)
	{
		dlist_mutable_iter	miter;

		dlist_foreach_modify(miter, &tracker_slot[i])
		{
			entry = dlist_container(tracker_entry, chain, miter.cur);

			if (entry->owner != CurrentResourceOwner)
				continue;

			dlist_delete(&entry->chain);
			if (*entry->object == StromTag_MsgQueue)
			{
				pgstrom_queue  *mqueue = (pgstrom_queue *)entry->object;
				pgstrom_close_queue(mqueue);
			}
			else if (*entry->object == StromTag_DevProgram)
			{
				Datum	dprog_key = PointerGetDatum(entry->object);

				pgstrom_put_devprog_key(dprog_key);
			}
			else
			{
				Assert(IS_TRACKABLE_OBJECT(entry->object));
				pgstrom_put_message((pgstrom_message *) entry->object);
			}
			dlist_push_head(&tracker_free, &entry->chain);
		}
	}
}

/*
 * pgstrom_track_object
 *
 * registers a shared object as one acquired by this backend.
 */
void
pgstrom_track_object(StromTag *stag)
{
	tracker_entry *entry;
	int		i;

	Assert(IS_TRACKABLE_OBJECT(stag));

	if (!dlist_is_empty(&tracker_free))
		entry = dlist_container(tracker_entry, chain,
								dlist_pop_head_node(&tracker_free));
	else
	{
		PG_TRY();
		{
			entry = MemoryContextAlloc(ResTrackContext,
									   sizeof(tracker_entry));
		}
		PG_CATCH();
		{
			if (*stag == StromTag_MsgQueue)
				pgstrom_close_queue((pgstrom_queue *)stag);
			else
				pgstrom_put_message((pgstrom_message *)stag);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	i = tracker_hash(stag);
	entry->object = stag;
	entry->owner = CurrentResourceOwner;
	dlist_push_head(&tracker_slot[i], &entry->chain);
}

/*
 * pgstrom_untrack_object
 *
 * unregister the supplied object from tracking table
 */
void
pgstrom_untrack_object(StromTag *stag)
{
	dlist_iter	iter;
	int			index;

	Assert(IS_TRACKABLE_OBJECT(stag));

	index = tracker_hash(stag);
	dlist_foreach(iter, &tracker_slot[index])
	{
		tracker_entry  *entry
			= dlist_container(tracker_entry, chain, iter.cur);

		if (entry->object == stag)
		{
			entry->object = NULL;
			dlist_delete(&entry->chain);
			dlist_push_head(&tracker_free, &entry->chain);
			return;
		}
	}
	elog(LOG, "Bug? untracked message %p (stag: %d)was not tracked",
		 stag, *stag);
}

bool
pgstrom_object_is_tracked(StromTag *stag)
{
	dlist_iter	iter;
	int			index;

	Assert(IS_TRACKABLE_OBJECT(stag));

	index = tracker_hash(stag);
	dlist_foreach(iter, &tracker_slot[index])
	{
		tracker_entry  *entry
			= dlist_container(tracker_entry, chain, iter.cur);

		if (entry->object == stag)
			return true;
	}
	return false;
}

void
pgstrom_init_restrack(void)
{
	int		i;

	ResTrackContext = AllocSetContextCreate(CacheMemoryContext,
											"PG-Strom resource tracker",
											ALLOCSET_DEFAULT_MINSIZE,
											ALLOCSET_DEFAULT_INITSIZE,
											ALLOCSET_DEFAULT_MAXSIZE);
	RegisterResourceReleaseCallback(pgstrom_restrack_callback, NULL);

	/* init hash table */
	dlist_init(&tracker_free);
	for (i=0; i < RESTRACK_HASHSZ; i++)
		dlist_init(&tracker_slot[i]);
}
