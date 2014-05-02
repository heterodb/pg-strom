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
#include "utils/pg_crc.h"
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
	ResourceReleasePhase phase;
	dlist_head		objects;
} tracker_entry;

#define RESTRACK_HASHSZ		100

#define IS_TRACKABLE_OBJECT(sobject)			\
	(StromTagIs(sobject,MsgQueue) ||			\
	 StromTagIs(sobject,DevProgram) ||			\
	 StromTagIs(sobject,TCacheHead) ||			\
	 StromTagIs(sobject,TCacheRowStore) ||		\
	 StromTagIs(sobject,TCacheColumnStore) ||	\
	 StromTagIs(sobject,GpuScan)	||			\
	 StromTagIs(sobject,GpuSort) ||				\
	 StromTagIs(sobject,HashJoin))

static dlist_head		tracker_free;
static dlist_head		tracker_slot[RESTRACK_HASHSZ];
static MemoryContext	ResTrackContext;

static inline int
restrack_hash_index(ResourceOwner resource_owner,
					ResourceReleasePhase phase)
{
	pg_crc32	crc;

	INIT_CRC32(crc);
	COMP_CRC32(crc, resource_owner, sizeof(ResourceOwner));
	COMP_CRC32(crc, &phase, sizeof(ResourceReleasePhase));
	FIN_CRC32(crc);

    return crc % RESTRACK_HASHSZ;
}

static tracker_entry *
restrack_get_entry(ResourceOwner resource_owner,
				   ResourceReleasePhase phase,
				   bool create_on_demand)
{
	tracker_entry  *entry = NULL;
	dlist_iter		iter;
	int		i;

	i = restrack_hash_index(resource_owner, phase);
	dlist_foreach(iter, &tracker_slot[i])
	{
		entry = dlist_container(tracker_entry, chain, iter.cur);

		if (entry->owner == CurrentResourceOwner &&
			entry->phase == phase)
			return entry;
	}
	if (!create_on_demand)
		return NULL;

	if (dlist_is_empty(&tracker_free))
	{
		entry = MemoryContextAllocZero(ResTrackContext,
									   sizeof(tracker_entry));
	}
	else
	{
		dlist_node *dnode = dlist_pop_head_node(&tracker_free);
		entry = dlist_container(tracker_entry, chain, dnode);
	}
	dlist_push_head(&tracker_slot[i], &entry->chain);
	entry->owner = resource_owner;
	entry->phase = phase;
	dlist_init(&entry->objects);

	return entry;
}

static void
pgstrom_restrack_callback(ResourceReleasePhase phase,
						  bool is_commit,
						  bool is_toplevel,
						  void *arg)
{
	tracker_entry  *entry
		= restrack_get_entry(CurrentResourceOwner, phase, false);

	if (entry)
	{
		dlist_mutable_iter miter;

		/*
		 * In case of normal transaction commit, all the tracked objects
		 * shall be untracked by regular code path, so we should not have
		 * any valid objects in resource tracker.
		 */
		Assert(!is_commit || dlist_is_empty(&entry->objects));

		/*
		 * In case of transaction abort, we decrement reference counter of
		 * tracked objects, then eventually they are released (even if
		 * OpenCL server still grabed it).
		 */
		dlist_foreach_modify(miter, &entry->objects)
		{
			StromObject *sobject
				= dlist_container(StromObject, tracker, miter.cur);

			dlist_delete(&sobject->tracker);
			memset(&sobject->tracker, 0, sizeof(dlist_node));

			if (StromTagIs(sobject, MsgQueue))
				pgstrom_close_queue((pgstrom_queue *)sobject);
			else if (StromTagIs(sobject, DevProgram))
				pgstrom_put_devprog_key(PointerGetDatum(sobject));
			else if (StromTagIs(sobject, TCacheHead))
				tcache_put_tchead((tcache_head *)sobject);
			else if (StromTagIs(sobject, TCacheRowStore))
				tcache_put_row_store((tcache_row_store *)sobject);
			else if (StromTagIs(sobject, TCacheColumnStore))
				tcache_put_column_store((tcache_column_store *)sobject);
			else
			{
				Assert(IS_TRACKABLE_OBJECT(sobject));
				pgstrom_put_message((pgstrom_message *) sobject);
			}
		}
		dlist_delete(&entry->chain);
		memset(entry, 0, sizeof(tracker_entry));
		dlist_push_head(&tracker_free, &entry->chain);
	}
}

/*
 * pgstrom_track_object
 *
 * registers a shared object as one acquired by this backend.
 */
void
pgstrom_track_object(StromObject *sobject)
{
	Assert(IS_TRACKABLE_OBJECT(sobject));

	PG_TRY();
	{
		tracker_entry *entry;

		/* XXX - right now all the object class uses after-locks */
		entry = restrack_get_entry(CurrentResourceOwner,
								   RESOURCE_RELEASE_AFTER_LOCKS,
								   true);
		dlist_push_head(&entry->objects, &sobject->tracker);
	}
	PG_CATCH();
	{
		if (StromTagIs(sobject, MsgQueue))
			pgstrom_close_queue((pgstrom_queue *)sobject);
		else if (StromTagIs(sobject, DevProgram))
			pgstrom_put_devprog_key((Datum)sobject);
		else if (StromTagIs(sobject, TCacheHead))
			tcache_put_tchead((tcache_head *)sobject);
		else if (StromTagIs(sobject, TCacheRowStore))
			tcache_put_row_store((tcache_row_store *)sobject);
		else if (StromTagIs(sobject, TCacheColumnStore))
			tcache_put_column_store((tcache_column_store *)sobject);
		else
			pgstrom_put_message((pgstrom_message *)sobject);
		PG_RE_THROW();
	}
	PG_END_TRY();
}

/*
 * pgstrom_untrack_object
 *
 * unregister the supplied object from tracking table
 */
void
pgstrom_untrack_object(StromObject *sobject)
{
	Assert(IS_TRACKABLE_OBJECT(sobject));
	Assert(sobject->tracker.prev != NULL && sobject->tracker.next != NULL);
	dlist_delete(&sobject->tracker);
	memset(&sobject->tracker, 0, sizeof(dlist_node));
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
