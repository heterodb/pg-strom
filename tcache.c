/*
 * tcache.c
 *
 * Routines for T-tree cache
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "access/sysattr.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "storage/proc.h"
#include "storage/shmem.h"
#include "utils/guc.h"
#include "utils/pg_crc.h"
#include "utils/syscache.h"
#include "pg_strom.h"

#define TCACHE_HASH_SIZE	2048
typedef struct {
	dlist_node	chain;
	pid_t		pid;
	Oid			datoid;
	Oid			reloid;
	Latch	   *latch;
} tcache_columnizer;

typedef struct {
	slock_t		lock;
	dlist_head	lru_list;		/* LRU list of tc_head */
	dlist_head	free_list;		/* list of free tc_head objects */
	dlist_head	pending_list;	/* list of tc_head pending for columnization */
	dlist_head	block_list;		/* list of blocks allocated for tcache_head */
	dlist_head	slot[TCACHE_HASH_SIZE];

	/* properties of columnizers */
	dlist_head	inactive_list;	/* list of inactive columnizers */
	tcache_columnizer columnizers[FLEXIBLE_ARRAY_MEMBER];
} tcache_common;

/*
 * static variables
 */
static shmem_startup_hook_type shmem_startup_hook_next;
static tcache_common  *tc_common = NULL;
static int	num_columnizers;






static tcache_head *
pgstrom_create_tcache_nolock(Oid reloid, Bitmapset *required,
							 tcache_head *tcache_old)
{
	dlist_node	   *dnode;
	tcache_head	   *result;
	Bitmapset	   *tempset;
	HeapTuple		tup;
	Form_pg_attribute attr;
	pg_crc32		crc;
	int				index;
	int				i, j;

	/* calculate hash index */
	INIT_CRC32(crc);
	COMP_CRC32(crc, &MyDatabaseId, sizeof(Oid));
	COMP_CRC32(crc, &reloid, sizeof(Oid));
	FIN_CRC32(crc);
	index = crc % TCACHE_HASH_SIZE;

	/*
	 * A new block has to be allocated, if no free tcache_head
	 * objects were not prepared.
	 */
	if (dlist_is_empty(&tc_common->free_list))
	{
		tcache_head	   *temp;
		dlist_node	   *block
			= pgstrom_shmem_alloc(sizeof(dlist_node) +
								  sizeof(tcache_head) *
								  TCACHE_HEAD_PER_BLOCK);
		if (!block)
			ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
					 errmsg("out of shared memory")));

		/*
		 * TODO: add shared memory reclaiming using LRU list of
		 * cache mechanism.
		 */
		dlist_push_tail(&tc_common->block_list, block);

		temp = (tcache_head *)(block + 1);
		for (i=0; i < TCACHE_HEAD_PER_BLOCK; i++)
		{
			temp->state = TC_STATE_FREE;
			dlist_push_tail(&tc_common->free_list, &temp->chain);
		}
	}
	Assert(!dlist_is_empty(&tc_common->free_list));

	/*
	 * initialize this tcache_head according to the required columns
	 */
	dnode = dlist_pop_head_node(&tc_common->free_list);
	result = dlist_container(tcache_head, chain, dnode);
	PG_TRY();
	{
		memset(result, 0, sizeof(tcache_head));
		result->refcnt = 1;
		LWLockInitialize(&result->lock, 0);
		result->state = TC_STATE_NOT_BUILD;
		dlist_init(&result->free_list);
		dlist_init(&result->block_list);
		dlist_init(&result->trs_list);
		result->datoid = MyDatabaseId;
		result->reloid = reloid;

		tempset = bms_copy(required);
		if (tcache_old)
		{
			for (i=0; i < tcache_old->nattrs; i++)
			{
				j = (tcache_old->attrs[i].attnum -
					 FirstLowInvalidHeapAttributeNumber);
				tempset = bms_add_member(tempset, j);
			}
		}
		if (bms_num_members(tempset) > TCACHE_MAX_ATTRS)
			elog(ERROR, "too many columns being cached");

		j = 0;
		while ((i = bms_first_member(tempset)) >= 0)
		{
			i += FirstLowInvalidHeapAttributeNumber;

			if (i <= 0)
				continue;

			tup = SearchSysCache2(ATTNUM,
								  ObjectIdGetDatum(reloid),
								  Int16GetDatum(i));
			if (!HeapTupleIsValid(tup))
				elog(ERROR, "cache lookup failed for attr %d of rel %u",
					 i, reloid);
			attr = (Form_pg_attribute) GETSTRUCT(tup);

			result->attrs[j].attlen = attr->attlen;
			result->attrs[j].attnum = attr->attnum;
			switch (attr->attalign)
			{
				case 'c':
					result->attrs[j].attalign = sizeof(cl_char);
					break;
				case 's':
					result->attrs[j].attalign = sizeof(cl_short);
					break;
				case 'i':
					result->attrs[j].attalign = sizeof(cl_int);
					break;
				case 'd':
					result->attrs[j].attalign = sizeof(cl_long);
					break;
				default:
					elog(ERROR, "unexpected attribute alignment: '%c'",
						 attr->attalign);
			}
			result->attrs[j].attbyval = attr->attbyval;
			result->attrs[j].attnotnull = attr->attnotnull;
			j++;
			ReleaseSysCache(tup);
		}
		result->nattrs = j;
		dlist_push_tail(&tc_common->slot[index], &result->chain);
		dlist_push_head(&tc_common->lru_list, &result->lru_chain);
	}
	PG_CATCH();
	{
		dlist_push_head(&tc_common->free_list, &result->chain);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return result;
}

static void
pgstrom_put_tcache_nolock(tcache_head *tc_head)
{
	if (--tc_head->refcnt == 0)
	{
		Assert(!tc_head->chain.prev && !tc_head->chain.next);
		Assert(!tc_head->lru_chain.prev && !tc_head->lru_chain.next);

		/*
		 * TODO: release blocks for column-store and row-store
		 */
		tc_head->state = TC_STATE_FREE;

		dlist_push_tail(&tc_common->free_list, &tc_head->chain);
	}
}

void
pgstrom_put_tcache(tcache_head *tc_head)
{
	SpinLockAcquire(&tc_common->lock);
	pgstrom_put_tcache_nolock(tc_head);
	SpinLockRelease(&tc_common->lock);
}

tcache_head *
pgstrom_get_tcache(Oid reloid, Bitmapset *required,
					bool create_on_demand)
{
	pg_crc32		crc;
	int				index;
	dlist_iter		iter;
	tcache_head	   *tc_old = NULL;
	tcache_head    *result = NULL;

	/* calculate hash index */
	INIT_CRC32(crc);
	COMP_CRC32(crc, &MyDatabaseId, sizeof(Oid));
	COMP_CRC32(crc, &reloid, sizeof(Oid));
	FIN_CRC32(crc);
	index = crc % TCACHE_HASH_SIZE;

	SpinLockAcquire(&tc_common->lock);
	PG_TRY();
	{
		dlist_foreach(iter, &tc_common->slot[index])
		{
			tcache_head	   *temp
				= dlist_container(tcache_head, chain, iter.cur);

			if (temp->datoid == MyDatabaseId &&
				temp->reloid == reloid)
			{
				Bitmapset  *tempset = bms_copy(required);
				int			i, j = 0;

				while ((i = bms_first_member(tempset)) >= 0 &&
					   j < temp->nattrs)
				{
					i += FirstLowInvalidHeapAttributeNumber;

					/* is this column on the cache? */
					while (j < temp->nattrs &&
						   temp->attrs[j].attnum != i)
						j++;
				}
				bms_free(tempset);

				if (j < temp->nattrs)
				{
					/*
					 * Perfect! Cache of the target relation exists and all
					 * the required columns are cached.
					 */
					temp->refcnt++;
					dlist_move_head(&tc_common->lru_list, &temp->lru_chain);
					result = temp;
					break;
				}

				/*
				 * Elsewhere, cache exists but all the required columns are
				 * not cached in-memory.
				 */
				tc_old = temp;
				break;
			}
		}

		if (!result && create_on_demand)
		{
			result = pgstrom_create_tcache_nolock(reloid, required, tc_old);
			if (tc_old)
			{
				dlist_delete(&tc_old->chain);
				dlist_delete(&tc_old->lru_chain);
				memset(&tc_old->chain, 0, sizeof(dlist_node));
				memset(&tc_old->lru_chain, 0, sizeof(dlist_node));
				pgstrom_put_tcache_nolock(tc_old);
			}
		}
	}
	PG_CATCH();
	{
		SpinLockRelease(&tc_common->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return result;
}



















static void
pgstrom_tcache_main(Datum index)
{
	tcache_columnizer  *columnizer;

	Assert(tc_common != NULL);
	Assert(index < num_columnizers);

	columnizer = &tc_common->columnizers[index];
	memset(columnizer, 0, sizeof(tcache_columnizer));
	columnizer->pid = getpid();
	columnizer->latch = &MyProc->procLatch;

	SpinLockAcquire(&tc_common->lock);
	dlist_push_tail(&tc_common->inactive_list, &columnizer->chain);
	SpinLockRelease(&tc_common->lock);

	/* pending now */
	



}

static void
pgstrom_startup_tcache(void)
{
	int		i;
	Size	length;
	bool	found;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	length = offsetof(tcache_common, columnizers[num_columnizers]);
	tc_common = ShmemInitStruct("tc_common", MAXALIGN(length), &found);
	Assert(!found);
	memset(tc_common, 0, sizeof(tcache_common));
	SpinLockInit(&tc_common->lock);
	dlist_init(&tc_common->lru_list);
	dlist_init(&tc_common->free_list);
	dlist_init(&tc_common->pending_list);
	for (i=0; i < TCACHE_HASH_SIZE; i++)
		dlist_init(&tc_common->slot[i]);
	dlist_init(&tc_common->inactive_list);
}

void
pgstrom_init_tcache(void)
{
	BackgroundWorker	worker;
	Size	length;
	int		i;

	/* number of columnizer worker processes */
	DefineCustomIntVariable("pgstrom.num_columnizers",
							"number of columnizer worker processes",
							NULL,
							&num_columnizers,
							1,
							1,
							INT_MAX,
							PGC_POSTMASTER,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	/* launch background worker processes */
	for (i=0; i < num_columnizers; i++)
	{
		memset(&worker, 0, sizeof(BackgroundWorker));
		snprintf(worker.bgw_name, sizeof(worker.bgw_name),
				 "PG-Strom columnizer-%u", i);
		worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
		worker.bgw_start_time = BgWorkerStart_PostmasterStart;
		worker.bgw_restart_time = BGW_NEVER_RESTART;
		worker.bgw_main = pgstrom_tcache_main;
		worker.bgw_main_arg = i;
		RegisterBackgroundWorker(&worker);
	}

	/* aquires shared memory region */
	length = offsetof(tcache_common, columnizers[num_columnizers]);
	RequestAddinShmemSpace(MAXALIGN(length));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_tcache;
}
