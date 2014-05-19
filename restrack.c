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
	dlist_head		sobject_list;
} tracker_entry;

typedef struct {
	dlist_node		tracker_chain;
	dlist_node		ptrmap_chain;
	ResourceOwner	owner;
	StromObject	   *sobject;
	Datum			private;
} sobject_entry;

#define IS_TRACKABLE_OBJECT(sobject)			\
	(StromTagIs(sobject,MsgQueue) ||			\
	 StromTagIs(sobject,DevProgram) ||			\
	 StromTagIs(sobject,TCacheHead) ||			\
	 StromTagIs(sobject,TCacheRowStore) ||		\
	 StromTagIs(sobject,TCacheColumnStore) ||	\
	 StromTagIs(sobject,GpuScan)	||			\
	 StromTagIs(sobject,GpuSort) ||				\
	 StromTagIs(sobject,GpuSortMulti) ||		\
	 StromTagIs(sobject,HashJoin))

#define RESTRACK_HASHSZ		100
#define PTRMAP_HASHSZ		1200

static dlist_head		sobject_free_list;
static dlist_head		tracker_free_list;
static dlist_head		tracker_slot[RESTRACK_HASHSZ];
static dlist_head		ptrmap_slot[PTRMAP_HASHSZ];
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

static inline int
ptrmap_hash_index(ResourceOwner resource_owner,
				  StromObject *sobject)
{
	pg_crc32	crc;

	INIT_CRC32(crc);
	COMP_CRC32(crc, resource_owner, sizeof(ResourceOwner));
	COMP_CRC32(crc, sobject, sizeof(StromObject *));
	FIN_CRC32(crc);

	return crc % PTRMAP_HASHSZ;
}

static tracker_entry *
restrack_get_entry(ResourceOwner resource_owner,
				   ResourceReleasePhase phase,
				   bool create_on_demand)
{
	tracker_entry  *tracker = NULL;
	dlist_iter		iter;
	int		i;

	i = restrack_hash_index(resource_owner, phase);
	dlist_foreach(iter, &tracker_slot[i])
	{
		tracker = dlist_container(tracker_entry, chain, iter.cur);

		if (tracker->owner == CurrentResourceOwner &&
			tracker->phase == phase)
			return tracker;
	}
	if (!create_on_demand)
		return NULL;

	if (dlist_is_empty(&tracker_free_list))
	{
		tracker = MemoryContextAllocZero(ResTrackContext,
										  sizeof(tracker_entry));
	}
	else
	{
		dlist_node *dnode = dlist_pop_head_node(&tracker_free_list);
		tracker = dlist_container(tracker_entry, chain, dnode);
	}
	dlist_push_head(&tracker_slot[i], &tracker->chain);
	tracker->owner = resource_owner;
	tracker->phase = phase;
	dlist_init(&tracker->sobject_list);

	return tracker;
}

static sobject_entry *
sobject_get_entry(ResourceOwner resource_owner,
				  StromObject *sobject, Datum private)
{
	sobject_entry  *so_entry;
	int		i;

	if (dlist_is_empty(&sobject_free_list))
	{
		so_entry = MemoryContextAllocZero(ResTrackContext,
										  sizeof(sobject_entry));
	}
	else
	{
		dlist_node *dnode = dlist_pop_head_node(&sobject_free_list);
		so_entry = dlist_container(sobject_entry, ptrmap_chain, dnode);
	}
	i = ptrmap_hash_index(resource_owner, sobject);
	dlist_push_head(&ptrmap_slot[i], &so_entry->ptrmap_chain);
	so_entry->owner = resource_owner;
	so_entry->sobject = sobject;
	so_entry->private = private;

	return so_entry;
}

static void
pgstrom_restrack_callback(ResourceReleasePhase phase,
						  bool is_commit,
						  bool is_toplevel,
						  void *arg)
{
	tracker_entry  *tracker
		= restrack_get_entry(CurrentResourceOwner, phase, false);

	if (tracker)
	{
		dlist_mutable_iter miter;

		/*
		 * In case of transaction abort, we decrement reference counter of
		 * tracked objects, then eventually they are released (even if
		 * OpenCL server still grabed it).
		 */
		dlist_foreach_modify(miter, &tracker->sobject_list)
		{
			sobject_entry  *so_entry;
			StromObject	   *sobject;
			Datum			private;

			so_entry = dlist_container(sobject_entry,
									   tracker_chain,
									   miter.cur);
			dlist_delete(&so_entry->tracker_chain);
			dlist_delete(&so_entry->ptrmap_chain);
			sobject = so_entry->sobject;
			private = so_entry->private;

			/*
			 * In case of normal transaction commit, all the tracked
			 * objects should be untracked by regular code path, so
			 * we should not have any valid objects in resource tracker.
			 */
			Assert(!is_commit);

			if (StromTagIs(sobject, MsgQueue))
				pgstrom_close_queue((pgstrom_queue *)sobject);
			else if (StromTagIs(sobject, DevProgram))
				pgstrom_put_devprog_key(PointerGetDatum(sobject));
			else if (StromTagIs(sobject, TCacheHead))
				tcache_abort_tchead((tcache_head *)sobject, private);
			else if (StromTagIs(sobject, TCacheRowStore))
				tcache_put_row_store((tcache_row_store *)sobject);
			else if (StromTagIs(sobject, TCacheColumnStore))
				tcache_put_column_store((tcache_column_store *)sobject);
			else
			{
				Assert(IS_TRACKABLE_OBJECT(sobject));
				pgstrom_put_message((pgstrom_message *) sobject);
			}
			memset(so_entry, 0, sizeof(sobject_entry));
			dlist_push_tail(&sobject_free_list, &so_entry->ptrmap_chain);
		}
		dlist_delete(&tracker->chain);
		memset(tracker, 0, sizeof(tracker_entry));
		dlist_push_head(&tracker_free_list, &tracker->chain);
	}
}

/*
 * pgstrom_track_object
 *
 * registers a shared object as one acquired by this backend.
 */
void
pgstrom_track_object(StromObject *sobject, Datum private)
{
	tracker_entry  *tracker = NULL;
	sobject_entry  *so_entry = NULL;

	Assert(IS_TRACKABLE_OBJECT(sobject));
	PG_TRY();
	{
		ResourceReleasePhase	phase;

		if (StromTagIs(sobject, TCacheHead))
			phase = RESOURCE_RELEASE_BEFORE_LOCKS;
		else
			phase = RESOURCE_RELEASE_AFTER_LOCKS;

		tracker = restrack_get_entry(CurrentResourceOwner, phase, true);
		so_entry = sobject_get_entry(CurrentResourceOwner, sobject, private);
		dlist_push_head(&tracker->sobject_list,
						&so_entry->tracker_chain);
	}
	PG_CATCH();
	{
		if (StromTagIs(sobject, MsgQueue))
			pgstrom_close_queue((pgstrom_queue *)sobject);
		else if (StromTagIs(sobject, DevProgram))
			pgstrom_put_devprog_key((Datum)sobject);
		else if (StromTagIs(sobject, TCacheHead))
			tcache_abort_tchead((tcache_head *)sobject, private);
		else if (StromTagIs(sobject, TCacheRowStore))
			tcache_put_row_store((tcache_row_store *)sobject);
		else if (StromTagIs(sobject, TCacheColumnStore))
			tcache_put_column_store((tcache_column_store *)sobject);
		else
			pgstrom_put_message((pgstrom_message *)sobject);

		/* also, tracker objects shall be backed to free-list */
		if (tracker)
			dlist_move_head(&tracker_free_list, &tracker->chain);
		Assert(so_entry == NULL);
		PG_RE_THROW();
	}
	PG_END_TRY();
}

/*
 * pgstrom_untrack_object
 *
 * unregister the supplied object from tracking table
 */
Datum
pgstrom_untrack_object(StromObject *sobject)
{
	dlist_mutable_iter	miter;
	sobject_entry	   *so_entry;
	Datum				private;
	int					i;

	Assert(IS_TRACKABLE_OBJECT(sobject));

	i = ptrmap_hash_index(CurrentResourceOwner, sobject);
	dlist_foreach_modify(miter, &ptrmap_slot[i])
	{
		so_entry = dlist_container(sobject_entry, ptrmap_chain, miter.cur);

		if (so_entry->owner != CurrentResourceOwner ||
			so_entry->sobject != sobject)
			continue;

		dlist_delete(&so_entry->tracker_chain);
		dlist_delete(&so_entry->ptrmap_chain);
		private = so_entry->private;
		memset(so_entry, 0, sizeof(sobject_entry));
		dlist_push_head(&sobject_free_list, &so_entry->ptrmap_chain);

		return private;
	}
	elog(ERROR, "StromObject %p (tag=%d) is not tracked",
		 sobject, (int)sobject->stag);
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
	dlist_init(&sobject_free_list);
	dlist_init(&tracker_free_list);
	for (i=0; i < RESTRACK_HASHSZ; i++)
		dlist_init(&tracker_slot[i]);
	for (i=0; i < PTRMAP_HASHSZ; i++)
		dlist_init(&ptrmap_slot[i]);
}
