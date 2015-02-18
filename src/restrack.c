/*
 * restrack.c
 *
 * Resource tracking for proper error handling 
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
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
#include "common/pg_crc.h"
#include "utils/memutils.h"
#include "utils/resowner.h"
#include "pg_strom.h"
#include "opencl_hashjoin.h"

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
	const char	   *filename;
	int				lineno;
	StromObject	   *sobject;
	Datum			private;
} sobject_entry;

#define IS_TRACKABLE_OBJECT(sobject)	\
	(StromTagIs(sobject,MsgQueue)	||	\
	 StromTagIs(sobject,DevProgram)	||	\
	 StromTagIs(sobject,DataStore)	||	\
	 StromTagIs(sobject,GpuScan)	||	\
	 StromTagIs(sobject,GpuPreAgg)	||	\
	 StromTagIs(sobject,GpuHashJoin)||	\
	 StromTagIs(sobject,GpuSort)	||	\
	 StromTagIs(sobject,HashJoinTable))

#define RESTRACK_HASHSZ		100
#define PTRMAP_HASHSZ		1200

static dlist_head		sobject_free_list;
static dlist_head		tracker_free_list;
static dlist_head		tracker_slot[RESTRACK_HASHSZ];
static dlist_head		ptrmap_slot[PTRMAP_HASHSZ];
static MemoryContext	ResTrackContext;
static bool				restrack_is_cleanup_context = false;

static inline int
restrack_hash_index(ResourceOwner resource_owner,
					ResourceReleasePhase phase)
{
	pg_crc32	crc;

	INIT_CRC32C(crc);
	COMP_CRC32C(crc, resource_owner, sizeof(ResourceOwner));
	COMP_CRC32C(crc, &phase, sizeof(ResourceReleasePhase));
	FIN_CRC32C(crc);

    return crc % RESTRACK_HASHSZ;
}

static inline int
ptrmap_hash_index(ResourceOwner resource_owner,
				  StromObject *sobject)
{
	pg_crc32	crc;

	INIT_CRC32C(crc);
	COMP_CRC32C(crc, resource_owner, sizeof(ResourceOwner));
	COMP_CRC32C(crc, sobject, sizeof(StromObject *));
	FIN_CRC32C(crc);

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
				  const char *filename, int lineno,
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
	so_entry->filename = filename;
	so_entry->lineno = lineno;
	so_entry->sobject = sobject;
	so_entry->private = private;

	return so_entry;
}

/*
 * pgstrom_restrack_cleanup_context - It informs another portions whether
 * the current context is restrack's cleanup context, or not.
 * Some class of resources (like, shared buffer) are released prior to
 * the callback of resource-owner.
 */
bool
pgstrom_restrack_cleanup_context(void)
{
	return restrack_is_cleanup_context;
}

static void
pgstrom_restrack_callback(ResourceReleasePhase phase,
						  bool is_commit,
						  bool is_toplevel,
						  void *arg)
{
	tracker_entry  *tracker;
	bool			saved_context = restrack_is_cleanup_context;

	tracker = restrack_get_entry(CurrentResourceOwner, phase, false);
	if (!tracker)
		return;

	PG_TRY();
	{
		dlist_mutable_iter miter;

		/* switch current context */
		restrack_is_cleanup_context = true;

		/*
		 * In case of transaction abort, we decrement reference counter of
		 * tracked objects, then eventually they are released (even if
		 * OpenCL server still grabed it).
		 */
		dlist_foreach_modify(miter, &tracker->sobject_list)
		{
			sobject_entry  *so_entry;
			StromObject	   *sobject;

			so_entry = dlist_container(sobject_entry,
									   tracker_chain,
									   miter.cur);
			dlist_delete(&so_entry->tracker_chain);
			dlist_delete(&so_entry->ptrmap_chain);
			sobject = so_entry->sobject;

			/*
			 * In case of normal transaction commit, all the tracked
			 * objects should be untracked by regular code path, so
			 * we should not have any valid objects in resource tracker.
			 */
			if (is_commit)
				elog(WARNING, "StromObject (%s at %s:%d) was not untracked",
					 StromTagGetLabel(sobject),
					 so_entry->filename, so_entry->lineno);

			if (StromTagIs(sobject, MsgQueue))
				pgstrom_close_queue((pgstrom_queue *)sobject);
			else if (StromTagIs(sobject, DevProgram))
				pgstrom_put_devprog_key(PointerGetDatum(sobject));
			else if (StromTagIs(sobject, HashJoinTable))
				multihash_put_tables((pgstrom_multihash_tables *) sobject);
			else if (StromTagIs(sobject, DataStore))
				pgstrom_put_data_store((pgstrom_data_store *) sobject);
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
	PG_CATCH();
	{
		restrack_is_cleanup_context = saved_context;
		PG_RE_THROW();
	}
	PG_END_TRY();
	restrack_is_cleanup_context = saved_context;
}

/*
 * pgstrom_track_object
 *
 * registers a shared object as one acquired by this backend.
 */
void
__pgstrom_track_object(const char *filename, int lineno,
					   StromObject *sobject, Datum private)
{
	tracker_entry  *tracker = NULL;
	sobject_entry  *so_entry = NULL;

	Assert(IS_TRACKABLE_OBJECT(sobject));
	PG_TRY();
	{
		ResourceReleasePhase	phase = RESOURCE_RELEASE_AFTER_LOCKS;

		tracker = restrack_get_entry(CurrentResourceOwner, phase, true);
		so_entry = sobject_get_entry(CurrentResourceOwner,
									 filename, lineno,
									 sobject, private);
		dlist_push_head(&tracker->sobject_list,
						&so_entry->tracker_chain);
	}
	PG_CATCH();
	{
		if (StromTagIs(sobject, MsgQueue))
			pgstrom_close_queue((pgstrom_queue *)sobject);
		else if (StromTagIs(sobject, DevProgram))
			pgstrom_put_devprog_key((Datum)sobject);
		else if (StromTagIs(sobject, DataStore))
			pgstrom_put_data_store((pgstrom_data_store *) sobject);
		else if (StromTagIs(sobject, HashJoinTable))
			multihash_put_tables((pgstrom_multihash_tables *) sobject);
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
	elog(INFO, "StromObject %p (%s) is not tracked",
		 sobject, StromTagGetLabel(sobject));
	Assert(false);
	return 0;
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
