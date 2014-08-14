/*
 * tcache.c
 *
 * Implementation of T-tree cache
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "access/heapam.h"
#include "access/sysattr.h"
#include "catalog/dependency.h"
#include "catalog/indexing.h"
#include "catalog/objectaccess.h"
#include "catalog/objectaddress.h"
#include "catalog/pg_class.h"
#include "catalog/pg_language.h"
#include "catalog/pg_namespace.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_trigger.h"
#include "catalog/pg_type.h"
#include "commands/trigger.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "storage/barrier.h"
#include "storage/bufmgr.h"
#include "storage/ipc.h"
#include "storage/proc.h"
#include "storage/shmem.h"
#include "utils/builtins.h"
#include "utils/bytea.h"
#include "utils/fmgroids.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/pg_crc.h"
#include "utils/rel.h"
#include "utils/syscache.h"
#include "utils/tqual.h"
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
	dlist_head	pending_list;	/* list of columnizer pending tc_head */
	dlist_head	slot[TCACHE_HASH_SIZE];

	/* properties of columnizers */
	dlist_head	inactive_list;	/* list of inactive columnizers */
	tcache_columnizer columnizers[FLEXIBLE_ARRAY_MEMBER];
} tcache_common;

/*
 * static variables
 */
static shmem_startup_hook_type shmem_startup_hook_next;
static object_access_hook_type object_access_hook_next;
static heap_page_prune_hook_type heap_page_prune_hook_next;
static tcache_common  *tc_common = NULL;
static int	num_columnizers;

/*
 * static declarations
 */
static void tcache_reset_tchead(tcache_head *tc_head);

static tcache_column_store *tcache_create_column_store(tcache_head *tc_head);
static tcache_column_store *tcache_duplicate_column_store(tcache_head *tc_head,
												  tcache_column_store *tcs_old,
												  bool duplicate_toastbuf);

static tcache_toastbuf *tcache_create_toast_buffer(Size required);
static tcache_toastbuf *tcache_duplicate_toast_buffer(tcache_toastbuf *tbuf,
													  Size required);
static tcache_toastbuf *tcache_get_toast_buffer(tcache_toastbuf *tbuf);
static void tcache_put_toast_buffer(tcache_toastbuf *tbuf);

static tcache_node *tcache_find_next_node(tcache_head *tc_head,
										  BlockNumber blkno);
static tcache_column_store *tcache_find_next_column_store(tcache_head *tc_head,
														  BlockNumber blkno);
//static tcache_node *tcache_find_prev_node(tcache_head *tc_head,
//										  BlockNumber blkno_cur);
static tcache_column_store *tcache_find_prev_column_store(tcache_head *tc_head,
														  BlockNumber blkno);

static void tcache_copy_cs_varlena(tcache_column_store *tcs_dst, int base_dst,
								   tcache_column_store *tcs_src, int base_src,
								   int attidx, int nitems);
static void tcache_rebalance_tree(tcache_head *tc_head,
								  tcache_node *tc_node,
								  tcache_node *tc_upper);

static void pgstrom_wakeup_columnizer(bool wakeup_all);

/*
 * NOTE: we usually put NULLs on 'prev' and 'next' of dlist_node to
 * mark this node is not linked.
 */
#define dnode_is_linked(dnode)		((dnode)->prev && (dnode)->next)

/*
 * NOTE: we cannot make a tcache_node compact under shared-lwlock,
 * we set a special mark (FrozenTransactionId on xmax) on vacuumed
 * tuples. These tuples are eventually released by columnizer process
 * under the exclusive-lock.
 */
#define CachedTupleIsVacuumed(htup)				\
	(HeapTupleHeaderGetRawXmax(htup) == FrozenTransactionId)

/*
 * NOTE: Default ItemPointerGetBlockNumber and ItemPointerGetOffsetNumber
 * internally have assertion checks. It may cause undesirable assertion
 * failure if these macros are called towards virtual tuples.
 */
#define ItemPointerGetBlockNumberNoAssert(ip)	\
	(((ip)->ip_blkid.bi_hi << 16) | (uint16)((ip)->ip_blkid.bi_lo))

#define ItemPointerGetOffsetNumberNoAssert(ip)	\
	((ip)->ip_posid)


/*
 * Misc utility functions
 */
static inline int
tcache_hash_index(Oid datoid, Oid reloid)
{
	pg_crc32	crc;

	INIT_CRC32(crc);
	COMP_CRC32(crc, &datoid, sizeof(Oid));
	COMP_CRC32(crc, &reloid, sizeof(Oid));
	FIN_CRC32(crc);

	return crc % TCACHE_HASH_SIZE;
}

static inline void
memswap(void *x, void *y, Size len)
{
	union {
		cl_uchar	v_uchar;
		cl_ushort	v_ushort;
		cl_uint		v_uint;
		cl_ulong	v_ulong;
		char		v_misc[32];	/* our usage is up to 32bytes right now */
	} temp;

	switch (len)
	{
		case sizeof(cl_uchar):
			temp.v_uchar = *((cl_uchar *) x);
			*((cl_uchar *) x) = *((cl_uchar *) y);
			*((cl_uchar *) y) = temp.v_uchar;
			break;
		case sizeof(cl_ushort):
			temp.v_ushort = *((cl_ushort *) x);
			*((cl_ushort *) x) = *((cl_ushort *) y);
			*((cl_ushort *) y) = temp.v_ushort;
			break;
		case sizeof(cl_uint):
			temp.v_uint = *((cl_uint *) x);
			*((cl_uint *) x) = *((cl_uint *) y);
			*((cl_uint *) y) = temp.v_uint;
			break;
		case sizeof(cl_ulong):
			temp.v_ulong = *((cl_ulong *) x);
			*((cl_ulong *) x) = *((cl_ulong *) y);
			*((cl_ulong *) y) = temp.v_ulong;
			break;
		default:
			Assert(len < sizeof(temp.v_misc));
			memcpy(temp.v_misc, x, len);
			memcpy(x, y, len);
			memcpy(y, temp.v_misc, len);
			break;
	}
}

static inline void
bitswap(uint8 *bitmap, int x, int y)
{
	bool	temp;

	temp = (bitmap[x / BITS_PER_BYTE] & (1 << (x % BITS_PER_BYTE))) != 0;

	if ((bitmap[y / BITS_PER_BYTE] &  (1 << (y % BITS_PER_BYTE))) != 0)
		bitmap[x / BITS_PER_BYTE] |=  (1 << (x % BITS_PER_BYTE));
	else
		bitmap[x / BITS_PER_BYTE] &= ~(1 << (x % BITS_PER_BYTE));

	if (temp)
		bitmap[y / BITS_PER_BYTE] |=  (1 << (y % BITS_PER_BYTE));
	else
		bitmap[y / BITS_PER_BYTE] &= ~(1 << (y % BITS_PER_BYTE));
}

/* almost same memcpy but use fast path if small data type */
static inline void
memcopy(void *dest, void *source, Size len)
{
	switch (len)
	{
		case sizeof(cl_uchar):
			*((cl_uchar *) dest) = *((cl_uchar *) source);
			break;
		case sizeof(cl_ushort):
			*((cl_ushort *) dest) = *((cl_ushort *) source);
			break;
		case sizeof(cl_uint):
			*((cl_uint *) dest) = *((cl_uint *) source);
			break;
		case sizeof(cl_ulong):
			*((cl_ulong *) dest) = *((cl_ulong *) source);
			break;
		case sizeof(ItemPointerData):
			*((ItemPointerData *) dest) = *((ItemPointerData *) source);
			break;
		default:
			memcpy(dest, source, len);
			break;
	}
}

static inline void
bitmapcopy(uint8 *dstmap, size_t dindex,
		   uint8 *srcmap, size_t sindex, size_t nbits)
{
    int     width = sizeof(Datum) * BITS_PER_BYTE;
    uint8  *temp;
    Datum  *dst;
    Datum  *src;
    int		dmod;
	int     smod;
    int     i, j;

    /* adjust alignment (destination) */
    temp = dstmap + dindex / BITS_PER_BYTE;
    dst = (Datum *)TYPEALIGN_DOWN(sizeof(Datum), temp);
    dmod = ((uintptr_t)temp -
            (uintptr_t)dst) * BITS_PER_BYTE + dindex % BITS_PER_BYTE;
    Assert(dmod < width);

    /* adjust alignment (source) */
    temp = srcmap + sindex / BITS_PER_BYTE;
    src = (Datum *)TYPEALIGN_DOWN(sizeof(Datum), temp);
    smod = ((uintptr_t)temp -
			(uintptr_t)src) * BITS_PER_BYTE + sindex % BITS_PER_BYTE;
	Assert(smod < width);

	nbits += dmod;

    /* ok, copy the bitmap */
    for (i=0, j=0; j < nbits; i++, j += width)
    {
		Datum	mask;
		Datum	bitmap;

		mask = ((i==0 ? ((1UL << dmod) - 1) : 0) |
				(j + width > nbits ? ~((1UL << (nbits - j)) - 1) : 0));
		if (dmod > smod)
		{
			bitmap = src[i] << (dmod - smod);
			if (i > 0)
				bitmap |= src[i-1] >> (width - (dmod - smod));
		}
		else
		{
			bitmap = (src[i] >> (smod - dmod));
			if (smod - dmod > 0)
				bitmap |= (src[i+1] << (width - (smod - dmod)));
		}
		dst[i] = (dst[i] & mask) | (bitmap & ~mask);
	}
}

/*
 * bms_fixup_sysattrs
 *
 * It fixes up the supplied bitmap to fix usual tcache manner.
 * - system columns: dropped from the bitmap
 * - whole row reference: expanded to all the regular columns
 */
static inline Bitmapset *
bms_fixup_sysattrs(int nattrs, Bitmapset *required)
{
	Bitmapset  *result;
	bitmapword	mask;
	int			i, j;

	if (!required)
		return NULL;

	mask = (bitmapword)(~((1UL << (1-FirstLowInvalidHeapAttributeNumber))- 1));
	result = bms_copy(required);
	if ((result->words[0] & mask) != result->words[0])
	{
		/*
		 * If whole-row-reference is required, it is equivalent to
		 * reference all the user columns. So, we expand the bitmap
		 * first, then turn of the bits of system columns.
		 */
		if (bms_is_member(-FirstLowInvalidHeapAttributeNumber, required))
		{
			for (i=1; i <= nattrs; i++)
			{
				j = i - FirstLowInvalidHeapAttributeNumber;
				result = bms_add_member(result, j);
			}
		}
		result->words[0] &= mask;
	}
	return result;
}

/*
 * lock, trylock and unlock
 */
static inline void
tcache_lock_tchead(tcache_head *tc_head, bool is_exclusive)
{
	LockAcquireResult	rc;
	LOCKMODE			lockmode
		= (is_exclusive ? AccessExclusiveLock : AccessShareLock);

	rc = LockAcquire(&tc_head->locktag, lockmode, false, false);
	Assert(rc == LOCKACQUIRE_OK || rc == LOCKACQUIRE_ALREADY_HELD);

	if (rc == LOCKACQUIRE_ALREADY_HELD)
		elog(INFO, "be careful. giant lock of tcache_head acquired twice");
}

static inline bool
tcache_trylock_tchead(tcache_head *tc_head, bool is_exclusive)
{
	LockAcquireResult	rc;
	LOCKMODE			lockmode
		= (is_exclusive ? AccessExclusiveLock : AccessShareLock);

	rc = LockAcquire(&tc_head->locktag, lockmode, false, true);
	if (rc == LOCKACQUIRE_NOT_AVAIL)
		return false;

	Assert(rc == LOCKACQUIRE_OK || rc == LOCKACQUIRE_ALREADY_HELD);
	if (rc == LOCKACQUIRE_ALREADY_HELD)
		elog(INFO, "be careful. giant lock of tcache_head acquired twice");
	return true;
}

static inline void
tcache_unlock_tchead(tcache_head *tc_head, bool is_exclusive)
{
	bool			rc;
	LOCKMODE		lockmode
		= (is_exclusive ? AccessExclusiveLock : AccessShareLock);

	rc = LockRelease(&tc_head->locktag, lockmode, false);
	Assert(rc);
}

static inline void
tcache_lock_held_by_me(tcache_head *tc_head, bool is_exclusive)
{
#ifdef USE_ASSERT_CHECKING
	LockAcquireResult	rc;

	rc = LockAcquire(&tc_head->locktag, AccessExclusiveLock, false, true);
	if (is_exclusive || rc == LOCKACQUIRE_ALREADY_HELD)
	{
		Assert(rc == LOCKACQUIRE_ALREADY_HELD);
		LockRelease(&tc_head->locktag, AccessExclusiveLock, false);
		return;
	}
    if (rc == LOCKACQUIRE_OK)
		LockRelease(&tc_head->locktag, AccessExclusiveLock, false);

	/* test shared lock */
	rc = LockAcquire(&tc_head->locktag, AccessShareLock, false, true);
	Assert(rc == LOCKACQUIRE_ALREADY_HELD);
	LockRelease(&tc_head->locktag, AccessShareLock, false);
#endif
}

/*
 * 
 *
 */
static tcache_column_store *
tcache_create_column_store(tcache_head *tc_head)
{
	tcache_column_store *tcs;
	TupleDesc	tupdesc = tc_head->tupdesc;
	Size	length;
	Size	offset;
	int		i;

	/* estimate length of column store */
	length = MAXALIGN(offsetof(tcache_column_store, cdata[tupdesc->natts]));
	length += MAXALIGN(sizeof(ItemPointerData) * NUM_ROWS_PER_COLSTORE);
	length += MAXALIGN(sizeof(HeapTupleHeaderData) * NUM_ROWS_PER_COLSTORE);

	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupdesc->attrs[i];

		/* is column cached? */
		if (!bms_is_member(attr->attnum - FirstLowInvalidHeapAttributeNumber,
						   tc_head->cached_attrs))
			continue;

		if (!attr->attnotnull)
			length += MAXALIGN(NUM_ROWS_PER_COLSTORE / BITS_PER_BYTE);
		length += MAXALIGN((attr->attlen > 0
							? attr->attlen
							: sizeof(cl_uint)) * NUM_ROWS_PER_COLSTORE);
	}

	/* OK, allocate it */
	tcs = pgstrom_shmem_alloc(length);
	if (!tcs)
		elog(ERROR, "out of shared memory");
	memset(tcs, 0, offsetof(tcache_column_store, cdata[tupdesc->natts]));

	tcs->sobj.stag = StromTag_TCacheColumnStore;
	SpinLockInit(&tcs->refcnt_lock);
	tcs->refcnt = 1;
	tcs->ncols = tupdesc->natts;
	offset = MAXALIGN(offsetof(tcache_column_store,
							   cdata[tupdesc->natts]));

	/* array of item-pointers */
	tcs->ctids = (ItemPointerData *)((char *)tcs + offset);
	offset += MAXALIGN(sizeof(ItemPointerData) *
					   NUM_ROWS_PER_COLSTORE);

	/* array of other system columns */
	tcs->theads = (HeapTupleHeaderData *)((char *)tcs + offset);
	offset += MAXALIGN(sizeof(HeapTupleHeaderData) *
					   NUM_ROWS_PER_COLSTORE);

	/* array of user defined columns */
	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupdesc->attrs[i];

		if (!bms_is_member(attr->attnum - FirstLowInvalidHeapAttributeNumber,
						   tc_head->cached_attrs))
			continue;

		if (attr->attnotnull)
			tcs->cdata[i].isnull = NULL;
		else
		{
			tcs->cdata[i].isnull = (uint8 *)((char *)tcs + offset);
			offset += MAXALIGN(NUM_ROWS_PER_COLSTORE / BITS_PER_BYTE);
		}
		tcs->cdata[i].values = ((char *)tcs + offset);
		offset += MAXALIGN((attr->attlen > 0
							? attr->attlen
							: sizeof(cl_uint)) *
						   NUM_ROWS_PER_COLSTORE);
		tcs->cdata[i].toast = NULL;	/* to be set later on demand */
	}
	Assert(offset == length);

	return tcs;
}

static tcache_column_store *
tcache_duplicate_column_store(tcache_head *tc_head,
							  tcache_column_store *tcs_old,
							  bool duplicate_toastbuf)
{
	tcache_column_store *tcs_new = tcache_create_column_store(tc_head);
	TupleDesc	tupdesc = tc_head->tupdesc;
	int			nrows = tcs_old->nrows;
	int			i;

	PG_TRY();
	{
		memcpy(tcs_new->ctids,
			   tcs_old->ctids,
			   sizeof(ItemPointerData) * nrows);
		memcpy(tcs_new->theads,
			   tcs_old->theads,
			   sizeof(HeapTupleHeaderData) * nrows);
		for (i=0; i < tupdesc->natts; i++)
		{
			Form_pg_attribute	attr = tupdesc->attrs[i];

			if (!tcs_old->cdata[i].values)
				continue;

			if (!attr->attnotnull)
			{
				Assert(tcs_new->cdata[i].isnull != NULL);
				memcpy(tcs_new->cdata[i].isnull,
					   tcs_old->cdata[i].isnull,
					   (nrows + BITS_PER_BYTE - 1) / BITS_PER_BYTE);
			}

			if (attr->attlen > 0)
			{
				memcpy(tcs_new->cdata[i].values,
					   tcs_old->cdata[i].values,
					   attr->attlen * nrows);
			}
			else if (!duplicate_toastbuf)
			{
				memcpy(tcs_new->cdata[i].values,
					   tcs_old->cdata[i].values,
					   sizeof(cl_uint) * nrows);
				tcs_new->cdata[i].toast
					= tcache_get_toast_buffer(tcs_old->cdata[i].toast);
			}
			else
			{
				Size	tbuf_length = tcs_old->cdata[i].toast->tbuf_length;

				tcs_new->cdata[i].toast
					= tcache_duplicate_toast_buffer(tcs_old->cdata[i].toast,
													tbuf_length);
			}
		}
	}
	PG_CATCH();
	{
		tcache_put_column_store(tcs_new);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return tcs_new;
}

tcache_column_store *
tcache_get_column_store(tcache_column_store *tcs)
{
	SpinLockAcquire(&tcs->refcnt_lock);
	Assert(tcs->refcnt > 0);
	tcs->refcnt++;
	SpinLockRelease(&tcs->refcnt_lock);

	return tcs;
}

void
tcache_put_column_store(tcache_column_store *tcs)
{
	bool	do_release = false;
	int		i;

	SpinLockAcquire(&tcs->refcnt_lock);
	Assert(tcs->refcnt > 0);
	if (--tcs->refcnt == 0)
		do_release = true;
	SpinLockRelease(&tcs->refcnt_lock);

	if (do_release)
	{
		for (i=0; i < tcs->ncols; i++)
		{
			if (tcs->cdata[i].toast)
				tcache_put_toast_buffer(tcs->cdata[i].toast);
		}
		pgstrom_shmem_free(tcs);
	}
}

/*
 * create, duplicate, get and put of toast_buffer
 */
static tcache_toastbuf *
tcache_create_toast_buffer(Size required)
{
	tcache_toastbuf *tbuf;
	Size		allocated;

	required = Max(required, TCACHE_TOASTBUF_INITSIZE);

	tbuf = pgstrom_shmem_alloc_alap(required, &allocated);
	if (!tbuf)
		elog(ERROR, "out of shared memory");

	SpinLockInit(&tbuf->refcnt_lock);
	tbuf->refcnt = 1;
	tbuf->tbuf_length = allocated;
	tbuf->tbuf_usage = offsetof(tcache_toastbuf, data[0]);
	tbuf->tbuf_junk = 0;

	return tbuf;
}

static tcache_toastbuf *
tcache_duplicate_toast_buffer(tcache_toastbuf *tbuf_old, Size required)
{
	tcache_toastbuf *tbuf_new;

	Assert(required >= tbuf_old->tbuf_usage);

	tbuf_new = tcache_create_toast_buffer(required);
	memcpy(tbuf_new->data,
		   tbuf_old->data,
		   tbuf_old->tbuf_usage - offsetof(tcache_toastbuf, data[0]));
	tbuf_new->tbuf_usage = tbuf_old->tbuf_usage;
	tbuf_new->tbuf_junk = tbuf_old->tbuf_junk;

	return tbuf_new;
}

static tcache_toastbuf *
tcache_get_toast_buffer(tcache_toastbuf *tbuf)
{
	SpinLockAcquire(&tbuf->refcnt_lock);
	Assert(tbuf->refcnt > 0);
	tbuf->refcnt++;
	SpinLockRelease(&tbuf->refcnt_lock);

	return tbuf;
}

static void
tcache_put_toast_buffer(tcache_toastbuf *tbuf)
{
	bool	do_release = false;

	SpinLockAcquire(&tbuf->refcnt_lock);
	Assert(tbuf->refcnt > 0);
	if (--tbuf->refcnt == 0)
		do_release = true;
	SpinLockRelease(&tbuf->refcnt_lock);

	if (do_release)
		pgstrom_shmem_free(tbuf);
}

/*
 * tcache_alloc_tcnode
 *
 * allocate a tcache_node according to the supplied tcache_head
 */
static tcache_node *
tcache_alloc_tcnode(tcache_head *tc_head)
{
	dlist_node	   *dnode;
	tcache_node	   *tc_node = NULL;

	SpinLockAcquire(&tc_head->lock);
	PG_TRY();
	{
		if (dlist_is_empty(&tc_head->free_list))
		{
			dlist_node *block;
			int			i;

			block = pgstrom_shmem_alloc(SHMEM_BLOCKSZ - SHMEM_ALLOC_COST);
			if (!block)
				elog(ERROR, "out of shared memory");
			dlist_push_tail(&tc_head->block_list, block);

			tc_node = (tcache_node *)(block + 1);
			for (i=0; i < TCACHE_NODE_PER_BLOCK_BARE; i++)
				dlist_push_tail(&tc_head->free_list, &tc_node[i].chain);
		}
		dnode = dlist_pop_head_node(&tc_head->free_list);
		tc_node = dlist_container(tcache_node, chain, dnode);
		memset(tc_node, 0, sizeof(tcache_node));

		SpinLockInit(&tc_node->lock);
		tc_node->tcs = tcache_create_column_store(tc_head);
		if (!tc_node->tcs)
			elog(ERROR, "out of shared memory");
	}
	PG_CATCH();
	{
		if (tc_node)
			dlist_push_tail(&tc_head->free_list, &tc_node->chain);
		SpinLockRelease(&tc_head->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	SpinLockRelease(&tc_head->lock);

	return tc_node;
}

/*
 * tcache_free_node
 *
 * release a tcache_node and detach column-store (not release it immediately,
 * because someone may copy the data)
 */
static void
tcache_free_node_nolock(tcache_head *tc_head, tcache_node *tc_node)
{
	SpinLockAcquire(&tc_node->lock);
	if (tc_node->tcs)
		tcache_put_column_store(tc_node->tcs);
	tc_node->tcs = NULL;
	SpinLockRelease(&tc_node->lock);
	dlist_push_head(&tc_head->free_list, &tc_node->chain);
}

static void
tcache_free_node_recurse(tcache_head *tc_head, tcache_node *tc_node)
{
	/* NOTE: caller must be responsible to hold tc_head->lock */
	if (tc_node->right)
		tcache_free_node_recurse(tc_head, tc_node->right);
	if (tc_node->left)
		tcache_free_node_recurse(tc_head, tc_node->left);
	tcache_free_node_nolock(tc_head, tc_node);
}

#if 0
static void
tcache_free_node(tcache_head *tc_head, tcache_node *tc_node)


{
	SpinLockAcquire(&tc_head->lock);
	tcache_free_node_nolock(tc_head, tc_node);
	SpinLockRelease(&tc_head->lock);
}
#endif

/*
 * tcache_find_next_record
 *
 * It finds a record with the least item-pointer greater than supplied
 * ctid, within a particular tcache_column_store.
 * If found, it return an index value [0 ... tcs->nrows - 1]. Elsewhere,
 * it returns a negative value.
 * Note that it may take linear time if tcache_column_store is not sorted.
 */
static int
tcache_find_next_record(tcache_column_store *tcs, ItemPointer ctid)
{
	BlockNumber	blkno_cur = ItemPointerGetBlockNumber(ctid);
	int			index = -1;

	if (tcs->nrows == 0)
		return -1;	/* no records are cached */
	if (blkno_cur > tcs->blkno_max)
		return -1;	/* ctid points higher block, so no candidate is here */

	if (tcs->is_sorted)
	{
		int		i_min = 0;
		int		i_max = tcs->nrows;

		while (i_min < i_max)
		{
			int	i_mid = (i_min + i_max) / 2;

			if (ItemPointerCompare(&tcs->ctids[i_mid], ctid) >= 0)
				i_max = i_mid;
			else
				i_min = i_mid + 1;
		}
		Assert(i_min == i_max);
		if (i_min >= 0 && i_min < tcs->nrows)
			index = i_min;
	}
	else
	{
		ItemPointerData	ip_cur;
		int		i;

		ItemPointerSet(&ip_cur, MaxBlockNumber, MaxOffsetNumber);
		for (i=0; i < tcs->nrows; i++)
		{
			if (ItemPointerCompare(&tcs->ctids[i], ctid) >= 0 &&
				ItemPointerCompare(&tcs->ctids[i], &ip_cur) < 0)
			{
				ItemPointerCopy(&tcs->ctids[i], &ip_cur);
				index = i;
			}
		}
	}
	return index;
}

/*
 * tcache_find_next_node
 * tcache_find_next_column_store
 *
 * It tried to find least column-store that can contain any records larger
 * than the supplied 'ctid'. Usually, this routine is aplied to forward scan.
 */
static void *
tcache_find_next_internal(tcache_node *tc_node, BlockNumber blkno_cur,
						  bool column_store)
{
	tcache_column_store *tcs = NULL;
	void	   *temp;

	SpinLockAcquire(&tc_node->lock);
	if (tc_node->tcs->nrows == 0)
	{
		SpinLockRelease(&tc_node->lock);
		return NULL;
	}

	if (blkno_cur > tc_node->tcs->blkno_max)
	{
		/*
		 * if current blockno is larger then or equal to the 'blkno_max',
		 * it is obvious that this node is not a one to be fetched on the
		 * next. So, we try to walk on the next right branch.
		 */
		SpinLockRelease(&tc_node->lock);

		if (!tc_node->right)
			return NULL;
		return tcache_find_next_internal(tc_node->right, blkno_cur,
										 column_store);
	}
	else if (!tc_node->left || blkno_cur >= tc_node->tcs->blkno_min)
	{
		/*
		 * Unlike above, this case is obvious that this chunk has records
		 * larger than required item-pointer.
		 */
		if (column_store)
			tcs = tcache_get_column_store(tc_node->tcs);
		SpinLockRelease(&tc_node->lock);

		return !tcs ? (void *)tc_node : (void *)tcs;
	}
	SpinLockRelease(&tc_node->lock);

	/*
	 * Even if ctid is less than ip_min and left-node is here, we need
	 * to pay attention on the case when ctid is larger than ip_max of
	 * left node tree. In this case, this tc_node shall still be a node
	 * to be fetched.
	 */
	if ((temp = tcache_find_next_internal(tc_node->left, blkno_cur,
										  column_store)) != NULL)
		return temp;

	/* if no left node is suitable, this node should be fetched */
	if (column_store)
	{
		SpinLockAcquire(&tc_node->lock);
		tcs = tcache_get_column_store(tc_node->tcs);
		SpinLockRelease(&tc_node->lock);
	}
	return !tcs ? (void *)tc_node : (void *)tcs;
}

static tcache_node *
tcache_find_next_node(tcache_head *tc_head, BlockNumber blkno)
{
	tcache_lock_held_by_me(tc_head, false);

	if (!tc_head->tcs_root)
		return NULL;
	return tcache_find_next_internal(tc_head->tcs_root, blkno, false);
}

static tcache_column_store *
tcache_find_next_column_store(tcache_head *tc_head, BlockNumber blkno)
{
	tcache_lock_held_by_me(tc_head, false);

	if (!tc_head->tcs_root)
		return NULL;
	return tcache_find_next_internal(tc_head->tcs_root, blkno, true);
}

/*
 * tcache_find_prev_node
 * tcache_find_prev_column_store
 *
 *
 *
 */
static void *
tcache_find_prev_internal(tcache_node *tc_node, BlockNumber blkno_cur,
						  bool column_store)
{
	tcache_column_store *tcs = NULL;
	void	   *temp;

	SpinLockAcquire(&tc_node->lock);
	if (tc_node->tcs->nrows == 0)
	{
		SpinLockRelease(&tc_node->lock);
		return NULL;
	}

	if (blkno_cur < tc_node->tcs->blkno_min)
	{
		/*
		 * it is obvious that this chunk cannot be a candidate to be
		 * fetched as previous one.
		 */
		SpinLockRelease(&tc_node->lock);

		if (!tc_node->left)
			return NULL;
		return tcache_find_prev_internal(tc_node->left, blkno_cur,
										 column_store);
	}
	else if (!tc_node->right || blkno_cur <= tc_node->tcs->blkno_max)
	{
		/*
		 * If ctid is less than ip_max but greater than or equal to ip_min,
		 * or tc_node has no left node, this node shall be fetched on the
		 * next.
		 */
		if (column_store)
			tcs = tcache_get_column_store(tc_node->tcs);
		SpinLockRelease(&tc_node->lock);

		return !tcs ? (void *)tc_node : (void *)tcs;
	}
	SpinLockRelease(&tc_node->lock);

	/*
	 * Even if ctid is less than ip_min and left-node is here, we need
	 * to pay attention on the case when ctid is larger than ip_max of
	 * left node tree. In this case, this tc_node shall still be a node
	 * to be fetched.
	 */
	if ((temp = tcache_find_prev_internal(tc_node->right, blkno_cur,
										  column_store)) != NULL)
		return temp;

	/* if no left node is suitable, this node should be fetched */
	if (column_store)
	{
		SpinLockAcquire(&tc_node->lock);
		tcs = tcache_get_column_store(tc_node->tcs);
		SpinLockRelease(&tc_node->lock);
	}
	return !tcs ? (void *)tc_node : (void *)tcs;
}

#if 0
static tcache_node *
tcache_find_prev_node(tcache_head *tc_head, BlockNumber blkno)
{
	tcache_lock_held_by_me(tc_head, false);

	if (!tc_head->tcs_root)
		return NULL;
	return tcache_find_prev_internal(tc_head->tcs_root, blkno, false);
}
#endif

static tcache_column_store *
tcache_find_prev_column_store(tcache_head *tc_head, BlockNumber blkno)
{
	tcache_lock_held_by_me(tc_head, false);

	if (!tc_head->tcs_root)
		return NULL;
	return tcache_find_prev_internal(tc_head->tcs_root, blkno, true);
}

/*
 * tcache_sort_tcnode
 *
 * It sorts contents of the column-store of a particular tcache_node
 * according to the item-pointers.
 */
static void
tcache_sort_tcnode_internal(tcache_head *tc_head, tcache_node *tc_node,
							tcache_column_store *tcs, int left, int right)
{
	int			li = left;
	int			ri = right;
	TupleDesc	tupdesc = tc_head->tupdesc;
	ItemPointerData pivot;

	if (left >= right)
		return;

	ItemPointerCopy(&tcs->ctids[(li + ri) / 2], &pivot);
	while (li < ri)
	{
		while (ItemPointerCompare(&tcs->ctids[li], &pivot) < 0)
			li++;
		while (ItemPointerCompare(&tcs->ctids[ri], &pivot) > 0)
			ri--;
		/*
		 * Swap values
		 */
		if (li < ri)
		{
			int		attlen;
			int		i;

			memswap(&tcs->ctids[li], &tcs->ctids[ri],
					sizeof(ItemPointerData));
			memswap(&tcs->theads[li], &tcs->theads[ri],
					sizeof(HeapTupleHeaderData));

			for (i=0; i < tupdesc->natts; i++)
			{
				Form_pg_attribute attr = tupdesc->attrs[i];

				if (!tcs->cdata[i].values)
					continue;

				attlen = (attr->attlen > 0
						  ? attr->attlen
						  : sizeof(cl_uint));
				/* isnull flags */
				if (!attr->attnotnull)
				{
					Assert(tcs->cdata[i].isnull != NULL);
					bitswap(tcs->cdata[i].isnull, li, ri);
				}
				memswap(tcs->cdata[i].values + attlen * li,
						tcs->cdata[i].values + attlen * ri,
						attlen);
			}
			li++;
			ri--;
		}
	}
	tcache_sort_tcnode_internal(tc_head, tc_node, tcs, left, li - 1);
	tcache_sort_tcnode_internal(tc_head, tc_node, tcs, ri + 1, right);
}

static void
tcache_sort_tcnode(tcache_head *tc_head, tcache_node *tc_node, bool is_inplace)
{
	tcache_column_store *tcs_new;

	if (is_inplace)
		tcs_new = tc_node->tcs;
	else
	{
		/*
		 * even if duplication mode, sort does not move varlena data on
		 * the toast buffer. So, we just reuse existing toast buffer,
		 */
		tcs_new = tcache_duplicate_column_store(tc_head, tc_node->tcs, false);
		tcache_put_column_store(tc_node->tcs);
		tc_node->tcs = tcs_new;
	}
	tcache_sort_tcnode_internal(tc_head, tc_node, tcs_new,
								0, tcs_new->nrows - 1);
	tcs_new->is_sorted = true;
}

/*
 * tcache_copy_cs_varlena
 *
 *
 *
 */
static void
tcache_copy_cs_varlena(tcache_column_store *tcs_dst, int base_dst,
					   tcache_column_store *tcs_src, int base_src,
					   int attidx, int nitems)
{
	tcache_toastbuf *tbuf_src = tcs_src->cdata[attidx].toast;
	tcache_toastbuf *tbuf_dst = tcs_dst->cdata[attidx].toast;
	uint8	   *src_isnull;
	uint8	   *dst_isnull;
	cl_uint	   *src_ofs;
	cl_uint	   *dst_ofs;
	cl_uint		vpos;
	cl_uint		vsize;
	char	   *vptr;
	int			i, si, di;

	Assert(tcs_src->cdata[attidx].values && tbuf_src &&
		   tcs_dst->cdata[attidx].values && tbuf_dst);
	src_isnull = tcs_src->cdata[attidx].isnull;
	dst_isnull = tcs_dst->cdata[attidx].isnull;
	src_ofs = (cl_uint *)(tcs_src->cdata[attidx].values);
	dst_ofs = (cl_uint *)(tcs_dst->cdata[attidx].values);
	for (i=0; i < nitems; i++)
	{
		si = base_src + i;
		di = base_dst + i;

		if (src_isnull && att_isnull(si, src_isnull))
		{
			Assert(dst_isnull);
			dst_isnull[di / BITS_PER_BYTE] &= ~(1 << (di % BITS_PER_BYTE));
			vpos = 0;
		}
		else
		{
			if (dst_isnull)
				dst_isnull[di / BITS_PER_BYTE] |= (1 << (di % BITS_PER_BYTE));

			vptr = (char *)tbuf_src + src_ofs[si];
			vsize = VARSIZE_ANY(vptr);
			if (tbuf_dst->tbuf_length < tbuf_dst->tbuf_usage + INTALIGN(vsize))
			{
				Size	new_len = 2 * tbuf_dst->tbuf_length;

				tcs_dst->cdata[attidx].toast
					= tcache_duplicate_toast_buffer(tbuf_dst, new_len);
				tbuf_dst = tcs_dst->cdata[attidx].toast;
			}
			memcpy((char *)tbuf_dst + tbuf_dst->tbuf_usage,
				   vptr,
				   vsize);
			vpos = tbuf_dst->tbuf_usage;
			tbuf_dst->tbuf_usage += INTALIGN(vsize);
		}
		dst_ofs[di] = vpos;
	}
}




/*
 * tcache_compaction_tcnode
 *
 *
 * NOTE: caller must hold exclusive global lock of tc_head
 */
static void
tcache_compaction_tcnode(tcache_head *tc_head, tcache_node *tc_node)
{
	tcache_column_store	*tcs_new;
	tcache_column_store *tcs_old = tc_node->tcs;

	tcache_lock_held_by_me(tc_head, true);

	tcs_new = tcache_create_column_store(tc_head);
	PG_TRY();
	{
		TupleDesc	tupdesc = tc_head->tupdesc;
		Size		required;
		int			i, j, k;

		/* assign a toast buffer first */
		for (i=0; i < tupdesc->natts; i++)
		{
			if (!tcs_old->cdata[i].toast)
				continue;

			required = tcs_old->cdata[i].toast->tbuf_length;
			tcs_new->cdata[i].toast = tcache_create_toast_buffer(required);
		}

		/*
		 * It ensures ip_max/ip_min shall be updated during the loop
		 * below, not a bug that miscopies min <-> max.
		 */
		tcs_new->blkno_min = tcs_old->blkno_max;
		tcs_new->blkno_max = tcs_old->blkno_min;

		/* OK, let's make it compacted */
		for (i=0, j=0; i < tcs_old->nrows; i++)
		{
			BlockNumber		blkno_cur;

			/*
			 * Once a record on the column-store is vacuumed, it will have
			 * FrozenTransactionId less than FirstNormalTransactionId.
			 * Nobody will never see the record, so we can skip it.
			 */
			if (CachedTupleIsVacuumed(&tcs_old->theads[i]))
				continue;

			/*
			 * Data copy
			 */
			memcopy(&tcs_new->ctids[j],
					&tcs_old->ctids[i],
					sizeof(ItemPointerData));
			memcopy(&tcs_new->theads[j],
					&tcs_old->theads[i],
					sizeof(HeapTupleHeaderData));
			blkno_cur = ItemPointerGetBlockNumber(&tcs_new->ctids[j]);
			if (blkno_cur > tcs_new->blkno_max)
				tcs_new->blkno_max = blkno_cur;
			if (blkno_cur < tcs_new->blkno_min)
				tcs_new->blkno_min = blkno_cur;

			for (k=0; k < tupdesc->natts; k++)
			{
				Form_pg_attribute attr = tupdesc->attrs[k];

				if (!tcs_old->cdata[k].values)
					continue;	/* skip, if not cached */

				/* nullmap */
				if (tcs_old->cdata[k].isnull)
					bitmapcopy(tcs_new->cdata[k].isnull, j,
							   tcs_old->cdata[k].isnull, i, 1);
				/* values */
				if (attr->attlen > 0)
					memcopy(tcs_new->cdata[k].values + attr->attlen * j,
							tcs_old->cdata[k].values + attr->attlen * i,
							attr->attlen);
				else
				{
					tcache_copy_cs_varlena(tcs_new, j,
										   tcs_old, i, k, 1);
				}
			}
			j++;
		}
		tcs_new->nrows = j;
		tcs_new->njunks = 0;
		tcs_new->is_sorted = tcs_old->is_sorted;

		Assert(tcs_old->nrows - tcs_old->njunks == tcs_new->nrows);

		/* ok, replace it */
		tc_node->tcs = tcs_new;
		tcache_put_column_store(tcs_old);

		/*
		 * TODO: how to handle the case when nrows == 0 ?
		 */
	}
	PG_CATCH();
	{
		tcache_put_column_store(tcs_new);
		PG_RE_THROW();
	}
	PG_END_TRY();
}

/*
 * tcache_try_merge_tcnode
 *
 *
 *
 */
static bool
do_try_merge_tcnode(tcache_head *tc_head,
					tcache_node *tc_parent,
					tcache_node *tc_child)	/* <- to be removed */
{
	if (tc_parent->tcs->nrows < NUM_ROWS_PER_COLSTORE / 2 &&
		tc_child->tcs->nrows  < NUM_ROWS_PER_COLSTORE / 2 &&
		(tc_parent->tcs->nrows +
		 tc_child->tcs->nrows)  < ((2 * NUM_ROWS_PER_COLSTORE) / 3))
	{
		tcache_column_store *tcs_src = tc_child->tcs;
		tcache_column_store *tcs_dst = tc_parent->tcs;
		TupleDesc	tupdesc = tc_head->tupdesc;
		int			base = tcs_dst->nrows;
		int			nmoved = tcs_src->nrows;
		int			i;

		memcpy(tcs_dst->ctids + base,
			   tcs_src->ctids,
			   sizeof(ItemPointerData) * nmoved);
		memcpy(tcs_dst->theads + base,
			   tcs_src->theads,
			   sizeof(HeapTupleHeaderData) * nmoved);
		for (i=0; i < tupdesc->natts; i++)
		{
			Form_pg_attribute	attr = tupdesc->attrs[i];

			/* skip, if uncached columns */
			if (!tcs_src->cdata[i].values)
				continue;

			/* move nullmap */
			if (!attr->attnotnull)
				bitmapcopy(tcs_dst->cdata[i].isnull, base,
						   tcs_src->cdata[i].isnull, 0,
						   nmoved);

			if (attr->attlen > 0)
			{
				memcpy(tcs_dst->cdata[i].values + attr->attlen * base,
					   tcs_src->cdata[i].values,
					   attr->attlen * nmoved);
			}
			else
			{
				tcache_copy_cs_varlena(tcs_dst, base,
									   tcs_src, 0,
									   i, nmoved);
			}
		}
		tcs_dst->nrows	+= tcs_src->nrows;
		tcs_dst->njunks	+= tcs_src->njunks;
		/* XXX - caller should set is_sorted */
		tcs_dst->blkno_max = Max(tcs_dst->blkno_max, tcs_src->blkno_max);
		tcs_dst->blkno_min = Max(tcs_dst->blkno_min, tcs_src->blkno_min);

		return true;
	}
	return false;
}

static bool
tcache_try_merge_left_recurse(tcache_head *tc_head,
							  tcache_node *tc_node,
							  tcache_node *target)
{
	if (!tc_node->left)
		return true;	/* first left-open node; that is merginable */
	else if (tcache_try_merge_left_recurse(tc_head, tc_node->left, target))
	{
		if (do_try_merge_tcnode(tc_head, target, tc_node->left))
		{
			tcache_node	*child = tc_node->left;

			Assert(!child->left);
			tc_node->left = child->right;
			tc_node->l_depth = child->r_depth + 1;
			tc_node->tcs->is_sorted = false;
			tcache_free_node_nolock(tc_head, child);
		}
	}
	return false;
}

static bool
tcache_try_merge_right_recurse(tcache_head *tc_head,
							   tcache_node *tc_node,
							   tcache_node *target)
{
	if (!tc_node->right)
		return true;	/* first right-open node; that is merginable */
	else if (tcache_try_merge_left_recurse(tc_head, tc_node->right, target))
	{
		if (do_try_merge_tcnode(tc_head, target, tc_node->right))
		{
			tcache_node	*child = tc_node->right;

			Assert(!child->right);
			tc_node->right = child->left;
			tc_node->r_depth = child->l_depth + 1;
			if (tc_node->tcs->is_sorted)
				tc_node->tcs->is_sorted = child->tcs->is_sorted;
			tcache_free_node_nolock(tc_head, child);
		}
	}
	return false;
}

static void
tcache_try_merge_recurse(tcache_head *tc_head,
						 tcache_node *tc_node,
						 tcache_node **p_upper,
						 tcache_node *l_candidate,
						 tcache_node *r_candidate,
						 tcache_node *target)
{
	if (tc_node->tcs->blkno_min > target->tcs->blkno_max)
	{
		/*
		 * NOTE: target's block-number is less than this node, so
		 * we go down the left branch. This node may be marginable
		 * if target target is right-open node, so we inform this
		 * node can be a merge candidate.
		 */
		Assert(tc_node->left != NULL);
		l_candidate = tc_node;	/* Last node that goes down left branch */
		tcache_try_merge_recurse(tc_head, tc_node->left, &tc_node->left,
								 l_candidate, r_candidate, target);
		if (!tc_node->left)
			tc_node->l_depth = 0;
		else
		{
			tc_node->l_depth = Max(tc_node->left->l_depth,
								   tc_node->left->r_depth) + 1;
			tcache_rebalance_tree(tc_head, tc_node->left, tc_node);
		}
	}
	else if (tc_node->tcs->blkno_max < target->tcs->blkno_min)
	{
		/*
		 * NOTE: target's block-number is greater than this node,
		 * so we go down the right branch. This node may be marginable
		 * if target target is left-open node, so we inform this node
		 * can be a merge candidate.
		 */
		Assert(tc_node->right != NULL);
		r_candidate = tc_node;	/* Last node that goes down right branch */
		tcache_try_merge_recurse(tc_head, tc_node->right, &tc_node->right,
								 l_candidate, r_candidate, target);
		if (!tc_node->right)
			tc_node->r_depth = 0;
		else
		{
			tc_node->r_depth = Max(tc_node->right->l_depth,
								   tc_node->right->r_depth) + 1;
			tcache_rebalance_tree(tc_head, tc_node->right, tc_node);
		}
	}
	else
	{
		Assert(tc_node == target);
		/* try to merge with the least greater node */
		if (tc_node->right)
			tcache_try_merge_left_recurse(tc_head, tc_node->right, target);
		/* try to merge with the greatest less node */
		if (tc_node->left)
			tcache_try_merge_right_recurse(tc_head, tc_node->left, target);

		if (!tc_node->right && l_candidate &&
			do_try_merge_tcnode(tc_head, l_candidate, tc_node))
		{
			/*
			 * try to merge with the last upper node that goes down left-
			 * branch, if target is right-open node.
			 */
			*p_upper = tc_node->left;
			tcache_free_node_nolock(tc_head, tc_node);			
		}
		else if (!tc_node->left && r_candidate &&
				 do_try_merge_tcnode(tc_head, r_candidate, tc_node))
		{
			/*
			 * try to merge with the last upper node that goes down right-
			 * branch, if target is left-open node.
			 */
			*p_upper = tc_node->right;
			tcache_free_node_nolock(tc_head, tc_node);
		}
	}
}

static void
tcache_try_merge_tcnode(tcache_head *tc_head, tcache_node *tc_node)
{
	tcache_lock_held_by_me(tc_head, true);

	/*
	 * NOTE: no need to walk on the tree if target contains obviously
	 * large enough number of records not to be merginable
	 */
	if (tc_node->tcs->nrows < NUM_ROWS_PER_COLSTORE / 2)
	{
		tcache_try_merge_recurse(tc_head,
								 tc_head->tcs_root,
								 &tc_head->tcs_root,
								 NULL,
								 NULL,
								 tc_node);
		tcache_rebalance_tree(tc_head, tc_head->tcs_root, NULL);
	}
}

/*
 * tcache_split_tcnode
 *
 * It creates a new tcache_node and move the largest one block of the records;
 * including varlena datum.
 *
 * NOTE: caller must hold exclusive lock of tc_head.
 */
static void
tcache_split_tcnode(tcache_head *tc_head, tcache_node *tc_node_old)
{
	tcache_node *tc_node_new;
    tcache_column_store *tcs_new;
    tcache_column_store *tcs_old = tc_node_old->tcs;

	tcache_lock_held_by_me(tc_head, true);

	tc_node_new = tcache_alloc_tcnode(tc_head);
	tcs_new = tc_node_new->tcs;
	PG_TRY();
	{
		TupleDesc tupdesc = tc_head->tupdesc;
		int		nremain;
		int		nmoved;
		int		i;

		/* assign toast buffers first */
		for (i=0; i < tupdesc->natts; i++)
		{
			Size	required;

			if (!tcs_old->cdata[i].toast)
				continue;

			required = tcs_old->cdata[i].toast->tbuf_length;
			tcs_new->cdata[i].toast = tcache_create_toast_buffer(required);
		}

		/*
		 * We have to sort this column-store first, if not yet.
		 * We assume this routine is called under the exclusive lock,
		 * so in-place sorting is safe.
		 */
		if (!tcs_old->is_sorted)
			tcache_sort_tcnode(tc_head, tc_node_old, true);

		/*
		 * Find number of records to be moved into the new one.
		 * Usually, a column-store being filled caches contents of
		 * multiple heap-pages. So, block-number of ip_min and ip_max
		 * should be different.
		 */
		Assert(tcs_old->blkno_min != tcs_old->blkno_max);

		for (nremain = tcs_old->nrows; nremain > 0; nremain--)
		{
			BlockNumber	blkno
				= ItemPointerGetBlockNumber(&tcs_old->ctids[nremain - 1]);

			if (blkno != tcs_old->blkno_max)
				break;
		}
		nmoved = tcs_old->nrows - nremain;
		Assert(nremain > 0 && nmoved > 0);

		/*
		 * copy item-pointers; also update ip_min/ip_max
		 */
		memcpy(tcs_new->ctids,
			   tcs_old->ctids + nremain,
			   sizeof(ItemPointerData) * nmoved);

		/*
		 * copy system columns
		 */
		memcpy(tcs_new->theads,
			   tcs_old->theads + nremain,
			   sizeof(HeapTupleHeaderData) * nmoved);

		/*
		 * copy regular columns
		 */
		for (i=0; i < tupdesc->natts; i++)
		{
			Form_pg_attribute attr = tupdesc->attrs[i];

			if (!tcs_old->cdata[i].values)
				continue;

			/* nullmap */
			if (!attr->attnotnull)
			{
				bitmapcopy(tcs_new->cdata[i].isnull, 0,
						   tcs_old->cdata[i].isnull, nremain,
						   nmoved);
			}
			/* regular columns */
			if (attr->attlen > 0)
			{
				memcpy(tcs_new->cdata[i].values,
					   tcs_old->cdata[i].values + attr->attlen * nremain,
					   attr->attlen * nmoved);
			}
			else
			{
				tcache_copy_cs_varlena(tcs_new, 0,
									   tcs_old, nremain,
									   i, nmoved);
			}
		}

		tcs_new->nrows = nmoved;
		tcs_new->njunks = 0;
		tcs_new->is_sorted = true;
		tcs_new->blkno_min
			= ItemPointerGetBlockNumber(&tcs_new->ctids[0]);
		tcs_new->blkno_max
			= ItemPointerGetBlockNumber(&tcs_new->ctids[nmoved - 1]);
		Assert(tcs_new->blkno_min == tcs_new->blkno_max);

		/*
		 * OK, tc_node_new is ready to chain as larger half of
		 * this column-store.
		 */
		tc_node_new->right = tc_node_old->right;
		tc_node_new->r_depth = tc_node_old->r_depth;
		tc_node_old->right = tc_node_new;
		tc_node_old->r_depth = tc_node_new->r_depth + 1;

		tcs_old->nrows = nremain;
		tcs_old->blkno_max
			= ItemPointerGetBlockNumber(&tcs_old->ctids[nremain - 1]);
	}
	PG_CATCH();
	{
		tcache_free_node_nolock(tc_head, tc_node_new);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/*
	 * At last, we try to remove garbages in the tc_node_old.
	 * tcache_compaction_tcnode() may cause an error, but larger half is
	 * already moved to the tc_node_new. In this case, the tree is still
	 * valid, even if tc_node_old has ideal format.
	 *
	 * XXX - it is an option to have compaction only toast-buffer,
	 * because tcache_compaction_tcnode kicks compaction on values-array
	 * also, not only toast-buffers. Usually, it may be expensive.
	 */
	tcache_compaction_tcnode(tc_head, tc_node_old);

	/*
	 * NOTE - Even once we split a node into two portion, we may need to
	 * merge them again because of over-compaction, if chunk has many
	 * junk records.
	 * 
	 * TODO - Is it possible to determine necessity of node split, prior
	 * to actual jobs.
	 */
	tcache_try_merge_tcnode(tc_head, tc_node_old);
}

/*
 * tcache_rebalance_tree
 *
 * It rebalances the t-tree structure if supplied 'tc_node' was not
 * a balanced tree.
 */
#define TCACHE_NODE_DEPTH(tc_node) \
	(!(tc_node) ? 0 : Max((tc_node)->l_depth, (tc_node)->r_depth) + 1)

static void
tcache_rebalance_tree(tcache_head *tc_head,
					  tcache_node *tc_node,
					  tcache_node *tc_upper)
{
	int		branch;

	tcache_lock_held_by_me(tc_head, true);
	Assert(!tc_upper || (tc_upper->left == tc_node ||
						 tc_upper->right == tc_node));
	if (!tc_upper)
		branch = 0;
	else if (tc_upper->left == tc_node)
		branch = -1;
	else
		branch = 1;

	if (tc_node->l_depth + 1 < tc_node->r_depth)
	{
		/* anticlockwise rotation */
		tcache_node *r_node = tc_node->right;

		tc_node->right = r_node->left;
		r_node->left = tc_node;

		tc_node->r_depth = TCACHE_NODE_DEPTH(tc_node->right);
		r_node->l_depth = TCACHE_NODE_DEPTH(tc_node);

		if (branch > 0)
		{
			tc_upper->right = r_node;
			tc_upper->r_depth = TCACHE_NODE_DEPTH(r_node);
		}
		else if (branch < 0)
		{
			tc_upper->left = r_node;
			tc_upper->l_depth = TCACHE_NODE_DEPTH(r_node);
		}
		else
			tc_head->tcs_root = r_node;
	}
	else if (tc_node->l_depth > tc_node->r_depth + 1)
	{
		/* clockwise rotation */
		tcache_node	*l_node = tc_node->left;

		tc_node->left = l_node->right;
		l_node->right = tc_node;

		tc_node->l_depth = TCACHE_NODE_DEPTH(tc_node->left);
		l_node->r_depth = TCACHE_NODE_DEPTH(tc_node);

		if (branch > 0)
		{
			tc_upper->right = l_node;
			tc_upper->r_depth = TCACHE_NODE_DEPTH(l_node);
		}
		else if (branch < 0)
		{
			tc_upper->left = l_node;
			tc_upper->l_depth = TCACHE_NODE_DEPTH(l_node);
		}
		else
			tc_head->tcs_root = l_node;
	}
}


/*
 * tcache_insert_tuple_row
 *
 *
 *
 *
 *
 */
bool
tcache_row_store_insert_tuple(tcache_row_store *trs, HeapTuple tuple)
{
	cl_uint	   *tupoffset;
	Size		required;
	rs_tuple   *rs_tup;

	required = MAXALIGN(sizeof(HeapTupleData)) + MAXALIGN(tuple->t_len);
	tupoffset = kern_rowstore_get_offset(&trs->kern);

	/* Do we have a space to put one more tuple? */
	if ((uintptr_t)(&tupoffset[trs->kern.nrows + 1]) >=
		(uintptr_t)((char *)&trs->kern + trs->usage) - required)
		return false;

	/* OK, this row-store still has space to hold this tuple */
	trs->usage -= required;
	rs_tup = (rs_tuple *)((char *)&trs->kern + trs->usage);

	memcpy(&rs_tup->htup, tuple, sizeof(HeapTupleData));
	rs_tup->htup.t_data = &rs_tup->data;
	memcpy(&rs_tup->data, tuple->t_data, tuple->t_len);

	tupoffset[trs->kern.nrows++] = trs->usage;
	/*
	 * NOTE: can be called to store virtual tuples.
	 */
	if (trs->blkno_max < ItemPointerGetBlockNumberNoAssert(&tuple->t_self))
		trs->blkno_max = ItemPointerGetBlockNumberNoAssert(&tuple->t_self);
	if (trs->blkno_min > ItemPointerGetBlockNumberNoAssert(&tuple->t_self))
		trs->blkno_min = ItemPointerGetBlockNumberNoAssert(&tuple->t_self);
	return true;
}

static void
tcache_insert_tuple_row(tcache_head *tc_head, HeapTuple tuple)
{
	tcache_row_store *trs = NULL;

	/* shared lwlock is sufficient to insert */
	tcache_lock_held_by_me(tc_head, false);
	SpinLockAcquire(&tc_head->lock);
	PG_TRY();
	{
	retry:
		if (tc_head->trs_curr)
			trs = pgstrom_get_row_store(tc_head->trs_curr);
		else
		{
			tc_head->trs_curr = pgstrom_create_row_store(tc_head->tupdesc);
			trs = pgstrom_get_row_store(tc_head->trs_curr);
		}

		if (!tcache_row_store_insert_tuple(trs, tuple))
		{
			/*
			 * No more space to put tuples any more. So, move this trs into
			 * columnizer pending list (if nobody do it), them retry again.
			 */
			dlist_push_head(&tc_head->trs_list, &trs->chain);
			pgstrom_put_row_store(trs);
			tc_head->trs_curr = trs = NULL;
			pgstrom_wakeup_columnizer(false);
			goto retry;
		}
	}
	PG_CATCH();
	{
		SpinLockRelease(&tc_head->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	SpinLockRelease(&tc_head->lock);
}

/*
 *
 *
 *
 *
 */
static bool
tcache_update_tuple_hints_rowstore(tcache_row_store *trs, HeapTuple tuple)
{
	int			index;

	if (ItemPointerGetBlockNumber(&tuple->t_self) > trs->blkno_max ||
		ItemPointerGetBlockNumber(&tuple->t_self) < trs->blkno_min)
		return false;

	for (index=0; index < trs->kern.nrows; index++)
	{
		rs_tuple   *rs_tup = kern_rowstore_get_tuple(&trs->kern, index);

		if (!rs_tup)
			continue;
		if (ItemPointerEquals(&rs_tup->htup.t_self, &tuple->t_self))
		{
			memcpy(&rs_tup->data, tuple->t_data, sizeof(HeapTupleHeaderData));
			return true;
		}
	}
	return false;
}

static void
tcache_update_tuple_hints(tcache_head *tc_head, HeapTuple tuple)
{
	BlockNumber		blkno = ItemPointerGetBlockNumber(&tuple->t_self);
	tcache_node	   *tc_node;
	bool			hit_on_tcs = false;

	tcache_lock_held_by_me(tc_head, false);

	tc_node = tcache_find_next_node(tc_head, blkno);
	if (tc_node)
	{
		tcache_column_store *tcs;
		int		index;

		SpinLockAcquire(&tc_node->lock);
		tcs = tc_node->tcs;
		index = tcache_find_next_record(tcs, &tuple->t_self);
		if (index >= 0)
		{
			HeapTupleHeader	htup;

			Assert(index < tcs->nrows);
			htup = &tcs->theads[index];
			Assert(HeapTupleHeaderGetRawXmax(htup) < FirstNormalTransactionId);
			memcpy(htup, tuple->t_data, sizeof(HeapTupleHeaderData));
			hit_on_tcs = true;
		}
		SpinLockRelease(&tc_node->lock);
	}

	/* if no entries in column-store, try to walk on row-store */
	if (!hit_on_tcs)
	{
		dlist_iter		iter;

		SpinLockAcquire(&tc_head->lock);
		if (tc_head->trs_curr &&
			tcache_update_tuple_hints_rowstore(tc_head->trs_curr, tuple))
			goto out;

		dlist_foreach(iter, &tc_head->trs_list)
		{
			tcache_row_store   *trs
				= dlist_container(tcache_row_store, chain, iter.cur);

			if (tcache_update_tuple_hints_rowstore(trs, tuple))
				break;
		}
	out:		
		SpinLockRelease(&tc_head->lock);
	}
}

/*
 * tcache_insert_tuple
 *
 *
 *
 *
 *
 */
static void
do_insert_tuple(tcache_head *tc_head, tcache_node *tc_node, HeapTuple tuple)
{
	tcache_column_store *tcs = tc_node->tcs;
	TupleDesc	tupdesc = tc_head->tupdesc;
	Datum	   *values = alloca(sizeof(Datum) * tupdesc->natts);
	bool	   *isnull = alloca(sizeof(bool) * tupdesc->natts);
	int			i;

	tcache_lock_held_by_me(tc_head, true);
	Assert(tcs->nrows < NUM_ROWS_PER_COLSTORE);
	Assert(tcs->ncols == tupdesc->natts);

	heap_deform_tuple(tuple, tupdesc, values, isnull);

	/* copy system columns */
	tcs->ctids[tcs->nrows] = tuple->t_self;
	memcopy(&tcs->theads[tcs->nrows], tuple->t_data,
			sizeof(HeapTupleHeaderData));
	for (i=0; i < tcs->ncols; i++)
	{
		Form_pg_attribute	attr = tupdesc->attrs[i];
		uint8	   *cs_isnull = tcs->cdata[i].isnull;
		char	   *cs_values = tcs->cdata[i].values;

		/* skip if it is not a cached column */
		if (!cs_values)
			continue;

		if (!cs_isnull)
			Assert(!isnull[i]);	/* should be always not null */
		else if (!isnull[i])
		{
			cs_isnull[tcs->nrows / BITS_PER_BYTE]
				|= (1 << (tcs->nrows % BITS_PER_BYTE));
		}
		else
		{
			cs_isnull[tcs->nrows / BITS_PER_BYTE]
				&= ~(1 << (tcs->nrows % BITS_PER_BYTE));

			if (attr->attlen > 0)
				memset(cs_values + attr->attlen * tcs->nrows,
					   0,
					   attr->attlen);
			else
				memset(cs_values + sizeof(cl_uint) * tcs->nrows,
					   0,
					   sizeof(cl_uint));
			continue;	/* no need to put values any more */
		}

		if (attr->attlen > 0)
		{
			if (attr->attbyval)
				memcopy(cs_values + attr->attlen * tcs->nrows,
						&values[i],
						attr->attlen);
			else
				memcpy(cs_values + attr->attlen * tcs->nrows,
					   DatumGetPointer(values[i]),
					   attr->attlen);
		}
		else
		{
			/*
			 * varlena datum shall be copied into toast-buffer once,
			 * and its offset (from the head of toast-buffer) shall be
			 * put on the values array.
			 */
			tcache_toastbuf	*tbuf = tcs->cdata[i].toast;
			Size		vsize = VARSIZE_ANY(values[i]);
			cl_uint	   *cs_ofs = (cl_uint *)tcs->cdata[i].values;

			if (!tbuf)
			{
				/* assign a toast buffer with default size */
				tcs->cdata[i].toast = tbuf = tcache_create_toast_buffer(0);
			}
			else if (tbuf->tbuf_length < tbuf->tbuf_usage + INTALIGN(vsize))
			{
				tcache_toastbuf	*tbuf_new;
				Size	new_len = 2 * tbuf->tbuf_length;

				/*
				 * Needs to expand toast-buffer if no more room exist
				 * to store new varlenas. Usually, twice amount of
				 * toast buffer is best choice for buddy allocator.
				 */
				tbuf_new = tcache_duplicate_toast_buffer(tbuf, new_len);

				/* replace older buffer by new (larger) one */
				tcache_put_toast_buffer(tbuf);
				tcs->cdata[i].toast = tbuf = tbuf_new;
			}
			Assert(tbuf->tbuf_usage + INTALIGN(vsize) < tbuf->tbuf_length);

			cs_ofs[tcs->nrows] = tbuf->tbuf_usage;
			memcpy((char *)tbuf + tbuf->tbuf_usage,
				   DatumGetPointer(values[i]),
				   vsize);
			tbuf->tbuf_usage += INTALIGN(vsize);
		}
	}

	/*
	 * update ip_max and ip_min, if needed
	 */
	if (tcs->nrows == 0)
	{
		tcs->blkno_min = ItemPointerGetBlockNumber(&tuple->t_self);
		tcs->blkno_max = ItemPointerGetBlockNumber(&tuple->t_self);
		tcs->is_sorted = true;	/* it is obviously sorted! */
	}
	else if (tcs->is_sorted)
	{
		if (ItemPointerCompare(&tuple->t_self, &tcs->ctids[tcs->nrows-1]) > 0)
			tcs->blkno_max = ItemPointerGetBlockNumber(&tuple->t_self);
		else
		{
			/*
			 * Oh... the new record placed on 'nrows' does not have
			 * largest item-pointer. It breaks the assumption; this
			 * column-store is sorted by item-pointer.
			 * It may take sorting again in the future.
			 */
			tcs->is_sorted = false;
			if (ItemPointerGetBlockNumber(&tuple->t_self) < tcs->blkno_min)
				tcs->blkno_min = ItemPointerGetBlockNumber(&tuple->t_self);
		}
	}
	else
	{
		if (ItemPointerGetBlockNumber(&tuple->t_self) > tcs->blkno_max)
			tcs->blkno_max = ItemPointerGetBlockNumber(&tuple->t_self);
		if (ItemPointerGetBlockNumber(&tuple->t_self) < tcs->blkno_min)
			tcs->blkno_min = ItemPointerGetBlockNumber(&tuple->t_self);
	}
	/* all valid, so increment nrows */
	pg_memory_barrier();
	tcs->nrows++;

	Assert(ItemPointerGetBlockNumber(&tuple->t_self) >= tcs->blkno_min &&
		   ItemPointerGetBlockNumber(&tuple->t_self) <= tcs->blkno_max);
}

static void
tcache_insert_tuple(tcache_head *tc_head,
					tcache_node *tc_node,
					HeapTuple tuple)
{
	tcache_column_store *tcs = tc_node->tcs;
	BlockNumber		blkno_cur = ItemPointerGetBlockNumber(&tuple->t_self);

	tcache_lock_held_by_me(tc_head, true);

	if (tcs->nrows == 0)
	{
		do_insert_tuple(tc_head, tc_node, tuple);
		/* no rebalance is needed obviously */
		return;
	}

retry:
	if (blkno_cur < tcs->blkno_min)
	{
		if (!tc_node->left && tcs->nrows < NUM_ROWS_PER_COLSTORE)
			do_insert_tuple(tc_head, tc_node, tuple);
		else
		{
			if (!tc_node->left)
			{
				tc_node->left = tcache_alloc_tcnode(tc_head);
				tc_node->l_depth = 1;
			}
			tcache_insert_tuple(tc_head, tc_node->left, tuple);
			tc_node->l_depth = TCACHE_NODE_DEPTH(tc_node->left);
			tcache_rebalance_tree(tc_head, tc_node->left, tc_node);
		}
	}
	else if (blkno_cur > tcs->blkno_max)
	{
		if (!tc_node->right && tcs->nrows < NUM_ROWS_PER_COLSTORE)
			do_insert_tuple(tc_head, tc_node, tuple);
		else
		{
			if (!tc_node->right)
			{
				tc_node->right = tcache_alloc_tcnode(tc_head);
				tc_node->r_depth = 1;
			}
			tcache_insert_tuple(tc_head, tc_node->right, tuple);
			tc_node->r_depth = TCACHE_NODE_DEPTH(tc_node->right);
			tcache_rebalance_tree(tc_head, tc_node->right, tc_node);
		}
	}
	else
	{
		if (tcs->nrows < NUM_ROWS_PER_COLSTORE)
			do_insert_tuple(tc_head, tc_node, tuple);
		else
		{
			/*
			 * No more room to store new records, so we split this chunk
			 * into two portions; the largest one block shall be pushed
			 * out into a new node.
			 */
			tcache_split_tcnode(tc_head, tc_node);
			goto retry;
		}
	}
}

/*
 * tcache_build_main
 *
 * main routine to construct columnar cache. It fully scans the heap
 * and insert the record into in-memory cache structure.
 */
static void
tcache_build_main(tcache_scandesc *tc_scan)
{
	tcache_head	   *tc_head = tc_scan->tc_head;
	HeapScanDesc	heapscan = tc_scan->heapscan;
	HeapTuple		tuple;
	struct timeval	tv1, tv2;

	tcache_lock_held_by_me(tc_head, true);
	Assert(heapscan != NULL);

	elog(INFO, "now building tcache...");
	gettimeofday(&tv1, NULL);
	while (true)
	{
		tuple = heap_getnext(heapscan, ForwardScanDirection);
		if (!HeapTupleIsValid(tuple))
			break;

		tcache_insert_tuple(tc_head, tc_head->tcs_root, tuple);
		tcache_rebalance_tree(tc_head, tc_head->tcs_root, NULL);
	}
	gettimeofday(&tv2, NULL);
	tc_scan->time_tcache_build = timeval_diff(&tv1, &tv2);
	elog(INFO, "tcache build done...");

	SpinLockAcquire(&tc_head->lock);
	tc_head->is_ready = true;
	SpinLockRelease(&tc_head->lock);

	heap_endscan(heapscan);
	tc_scan->heapscan = NULL;
}

/*
 *
 * NOTE: this operation does not increment reference counter of tcache_head,
 * so caller must be responsible to get and track it.
 */
tcache_scandesc *
tcache_begin_scan(tcache_head *tc_head, Relation heap_rel)
{
	tcache_scandesc	   *tc_scan;

	Assert(heap_rel != NULL);

	tc_scan = palloc0(sizeof(tcache_scandesc));
	tc_scan->rel = heap_rel;
	tc_scan->heapscan = NULL;
	tc_scan->has_exlock = false;
	tc_scan->tc_head = tc_head;
	tc_scan->tcs_blkno_min = InvalidBlockNumber;
	tc_scan->tcs_blkno_max = InvalidBlockNumber;
	tc_scan->trs_curr = NULL;

	PG_TRY();
	{
		if (tcache_trylock_tchead(tc_head, false))
		{
		retry:
			SpinLockAcquire(&tc_head->lock);
			if (!tc_head->is_ready)
			{
				if (!tc_scan->has_exlock)
				{
					SpinLockRelease(&tc_head->lock);
					tcache_unlock_tchead(tc_head, false);

					/*
					 * XXX - A worst scenario - if many concurrent jobs are
					 * launched towards unbuilt tcache, concurrent jobs try
					 * to acquire exclusive lock at same time, and none-
					 * builder process has to wait, although it is rare.
					 */
					tcache_lock_tchead(tc_head, true);
					tc_scan->has_exlock = true;
					goto retry;
				}
				else
				{
					/* OMG! I have to build up this tcache */
					SpinLockRelease(&tc_head->lock);
					tc_scan->heapscan = heap_beginscan(heap_rel,
													   SnapshotAny,
													   0, NULL);
				}
			}
			else if (tc_scan->has_exlock)
			{
				/*
				 * Only a process who is building up this tcache can hold
				 * exclusive lwlock of the tcache. If it was not actually
				 * needed, we should degrade the lock.
				 */
				SpinLockRelease(&tc_head->lock);
				tcache_unlock_tchead(tc_head, true);
				tcache_lock_tchead(tc_head, false);
				tc_scan->has_exlock = false;

				goto retry;
			}
			else
				SpinLockRelease(&tc_head->lock);
		}
		else
		{
			/*
			 * In case when we cannot acquire the shared-lwlock, it implies
			 * there is a concurrent writer job (either columnizer or backend
			 * building cache). Usually, columnizer's job will end in a short
			 * term, so we put a short delay then retry to get a lock.
			 */
			pg_usleep(50000L);	/* 50msec */

			if (tcache_trylock_tchead(tc_head, false))
				goto retry;

			elog(INFO, "could not acquired shared lock");

			/*
			 * If unavailable to get lwlock after the short delay above,
			 * probably, cache building is in-progress. So, give it up.
			 */
			pfree(tc_scan);
			tc_scan = NULL;
		}
	}
	PG_CATCH();
	{
		pfree(tc_scan);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return tc_scan;
}

StromObject *
tcache_scan_next(tcache_scandesc *tc_scan)
{
	tcache_head		   *tc_head = tc_scan->tc_head;
	tcache_row_store   *trs_prev;
	tcache_row_store   *trs_curr;
	dlist_node		   *dnode;

	/*
	 * NOTE: In case when tcache_head is not build yet, tc_scan will have
	 * a valid 'heapscan'. Even though it is a bit ugly design, we try to
	 * load contents of the heap once, then move to 
	 */
	if (tc_scan->heapscan)
		tcache_build_main(tc_scan);

	/* at least, we have to hold shared-lwlock on tc_head here */
	tcache_lock_held_by_me(tc_head, false);

	if (!tc_scan->trs_curr)
	{
		tcache_column_store *tcs;
		BlockNumber		blkno;

		if (tc_scan->tcs_blkno_max == InvalidBlockNumber)
			blkno = 0;
		else
			blkno = Min(tc_scan->tcs_blkno_max + 1, MaxBlockNumber);

		tcs = tcache_find_next_column_store(tc_head, blkno);
		if (tcs)
		{
			tc_scan->tcs_blkno_min = tcs->blkno_min;
			tc_scan->tcs_blkno_max = tcs->blkno_max;
			return &tcs->sobj;
		}
	}
	tc_scan->tcs_blkno_min = InvalidBlockNumber;
	tc_scan->tcs_blkno_max = InvalidBlockNumber;

	/*
	 * OK, we have no column-store any more, so we also walks on row-stores
	 * being not columnized yet.
	 * A tcache has one active (that means "writable") row-store by
	 * synchronizer trigger, and can also have multiple columnization
	 * pending row-stores. We also pay attention that concurrent job may
	 * write new tuples on the active row-store, then it can be linked
	 * to the pending list. So, we needs to check whether the previous
	 * row-store is connected to the list or not. If we already scanned
	 * a row-store that is not connected, it means we are now end of scan.
	 */
	SpinLockAcquire(&tc_head->lock);
	trs_prev = tc_scan->trs_curr;
	if (!trs_prev)
	{
		/*
		 * First choice. If we have empty pending list, let's pick up
		 * a row-store from the active one (maybe NULL).
		 */
		if (!dlist_is_empty(&tc_head->trs_list))
		{
			dnode = dlist_head_node(&tc_head->trs_list);
			trs_curr = dlist_container(tcache_row_store, chain, dnode);
		}
		else
			trs_curr = tc_head->trs_curr;

		if (trs_curr)
			tc_scan->trs_curr = pgstrom_get_row_store(trs_curr);
	}
	else if (dnode_is_linked(&trs_prev->chain))
	{
		/*
		 * In case when the previous row-store is one of pending list,
		 * we try to pick up the next one in the pending list. If no
		 * row-stores any more, we can pick up the active one.
		 */
		if (dlist_has_next(&tc_head->trs_list, &trs_prev->chain))
		{
			dnode = dlist_next_node(&tc_head->trs_list, &trs_prev->chain);
			trs_curr = dlist_container(tcache_row_store, chain, dnode);
			tc_scan->trs_curr = pgstrom_get_row_store(trs_curr);
		}
		else
		{
			if (tc_head->trs_curr)
				tc_scan->trs_curr = pgstrom_get_row_store(tc_head->trs_curr);
			else
				tc_scan->trs_curr = NULL;
		}
	}
	else
	{
		/*
		 * OK, the previous one is the active row-store, so now we are
		 * end of the scan.
		 */
		tc_scan->trs_curr = NULL;
	}
	SpinLockRelease(&tc_head->lock);

	if (tc_scan->trs_curr)
		return &tc_scan->trs_curr->sobj;
	return NULL;
}

StromObject *
tcache_scan_prev(tcache_scandesc *tc_scan)
{
	tcache_head		   *tc_head = tc_scan->tc_head;
	tcache_column_store *tcs;
	BlockNumber		blkno;
	dlist_node	   *dnode;

	/*
	 * NOTE: In case when tcache_head is not build yet, tc_scan will have
	 * a valid 'heapscan'. Even though it is a bit ugly design, we try to
	 * load contents of the heap once, then move to 
	 */
	if (tc_scan->heapscan)
		tcache_build_main(tc_scan);

	/* at least, we have to hold shared-lwlock on tc_head here */
	tcache_lock_held_by_me(tc_head, false);

	if (tc_scan->tcs_blkno_min == InvalidBlockNumber)
	{
		tcache_row_store *trs_prev = tc_scan->trs_curr;
		tcache_row_store *trs_curr;

		SpinLockAcquire(&tc_head->lock);
		if (!trs_prev)
		{
			if (tc_head->trs_curr)
			{
				trs_curr = pgstrom_get_row_store(tc_head->trs_curr);
				Assert(!dnode_is_linked(&trs_curr->chain));
			}
			else if (!dlist_is_empty(&tc_head->trs_list))
			{
				dnode = dlist_tail_node(&tc_head->trs_list);
				trs_curr = dlist_container(tcache_row_store, chain, dnode);
				trs_curr = pgstrom_get_row_store(trs_curr);
			}
			else
				trs_curr = NULL;
		}
		else if (!dnode_is_linked(&trs_prev->chain))
		{
			if (!dlist_is_empty(&tc_head->trs_list))
			{
				dnode = dlist_tail_node(&tc_head->trs_list);
				trs_curr = dlist_container(tcache_row_store, chain, dnode);
				trs_curr = pgstrom_get_row_store(trs_curr);
			}
			else
				trs_curr = NULL;
		}
		else
		{
			if (!dlist_has_prev(&tc_head->trs_list, &trs_prev->chain))
			{
				dnode = dlist_prev_node(&tc_head->trs_list,
										&trs_prev->chain);
				trs_curr = dlist_container(tcache_row_store, chain, dnode);
				trs_curr = pgstrom_get_row_store(trs_curr);
			}
			else
				trs_curr = NULL;
		}
		tc_scan->trs_curr = trs_curr;
		SpinLockRelease(&tc_head->lock);

		/* if we have a row-store, return it */
		if (tc_scan->trs_curr)
			return &tc_scan->trs_curr->sobj;
	}
	/* if we have no row-store, we also walks on column-stores */

	/* it's obvious we have no more column-store in this direction */
	if (tc_scan->tcs_blkno_min == 0)
	{
		tc_scan->tcs_blkno_min = InvalidBlockNumber;
		tc_scan->tcs_blkno_max = InvalidBlockNumber;
		return NULL;
	}
	Assert(tc_scan->tcs_blkno_min > 0);
	if (tc_scan->tcs_blkno_min == InvalidBlockNumber)
		blkno = MaxBlockNumber;
	else
		blkno = tc_scan->tcs_blkno_min - 1;

	tcs = tcache_find_prev_column_store(tc_head, blkno);
	if (tcs)
	{
		tc_scan->tcs_blkno_min = tcs->blkno_min;
		tc_scan->tcs_blkno_max = tcs->blkno_max;
		return &tcs->sobj;
	}
	return NULL;
}

void
tcache_end_scan(tcache_scandesc *tc_scan)
{
	tcache_head	   *tc_head = tc_scan->tc_head;

	if (tc_scan->heapscan)
	{
		/*
		 * If heapscan is already reached to end of the relation,
		 * tc_scan->heapscan shall be already closed. If not, it implies
		 * transaction was aborted in the middle.
		 */
		Assert(!tcache_state_is_ready(tc_head));
		tcache_reset_tchead(tc_head);
		heap_endscan(tc_scan->heapscan);
	}
	else
	{
		/* cache should be already available */
		Assert(tcache_state_is_ready(tc_head));
	}
	/* release either exclusive or shared lock */
	tcache_unlock_tchead(tc_head, tc_scan->has_exlock);

	pfree(tc_scan);
}

void
tcache_rescan(tcache_scandesc *tc_scan)
{
	tcache_head	   *tc_head = tc_scan->tc_head;

	tc_scan->tcs_blkno_min = InvalidBlockNumber;
	tc_scan->tcs_blkno_max = InvalidBlockNumber;
	tc_scan->trs_curr = NULL;

	if (tc_scan->heapscan)
	{
		/*
		 * XXX - right now, we have no way to recovery a half-built cache,
		 * so, we reset the current cache once, then rebuild it again.
		 */
		Assert(!tcache_state_is_ready(tc_head));
		tcache_reset_tchead(tc_head);
		heap_rescan(tc_scan->heapscan, NULL);
	}
}









/*
 * tcache_create_tchead
 *
 * It constructs an empty tcache_head that is capable to cache required
 * attributes. Usually, this routine is called by tcache_get_tchead with
 * on-demand creation. Caller has to acquire tc_common->lock on invocation.
 */
static tcache_head *
tcache_create_tchead(Oid reloid, Bitmapset *required,
					 tcache_head *tcache_old)
{
	tcache_head	   *tc_head;
	HeapTuple		reltup;
	HeapTuple		atttup;
	Form_pg_class	relform;
	TupleDesc		tupdesc;
	Size			length;
	Size			allocated;
	Bitmapset	   *tempset;
	bitmapword		attxor = 0;
	int				i, nwords;

	/* calculation of the length */
	reltup = SearchSysCache1(RELOID, ObjectIdGetDatum(reloid));
	if (!HeapTupleIsValid(reltup))
		elog(ERROR, "cache lookup failed for relation %u", reloid);
	relform = (Form_pg_class) GETSTRUCT(reltup);

	nwords = (relform->relnatts -
			  FirstLowInvalidHeapAttributeNumber) / BITS_PER_BITMAPWORD + 1;
	length = (MAXALIGN(offsetof(tcache_head, data[0])) +
			  MAXALIGN(sizeof(*tupdesc)) +
			  MAXALIGN(sizeof(Form_pg_attribute) * relform->relnatts) +
			  MAXALIGN(sizeof(FormData_pg_attribute)) * relform->relnatts +
			  MAXALIGN(offsetof(Bitmapset, words[nwords])));

	/* allocation of a shared memory block (larger than length) */
	tc_head = pgstrom_shmem_alloc_alap(length, &allocated);
	if (!tc_head)
		elog(ERROR, "out of shared memory");

	PG_TRY();
	{
		Size	offset = MAXALIGN(offsetof(tcache_head, data[0]));

		memset(tc_head, 0, sizeof(tcache_head));

		tc_head->sobj.stag = StromTag_TCacheHead;
		/*
		 * This refcnt shall be decremented when tc_head was put and
		 * unlinked from the hash table. Initially, tc_head is acquired
		 * by this context and linked to the hash. So, refcnt should be
		 * initialized to '2'.
		 */
		tc_head->refcnt = 2;
		SpinLockInit(&tc_head->lock);
		tc_head->is_ready = false;
		dlist_init(&tc_head->free_list);
		dlist_init(&tc_head->block_list);
		dlist_init(&tc_head->pending_list);
		dlist_init(&tc_head->trs_list);
		tc_head->datoid = MyDatabaseId;
		tc_head->reloid = reloid;

		tempset = bms_fixup_sysattrs(relform->relnatts, required);
		if (tcache_old)
			tempset = bms_union(tcache_old->cached_attrs, tempset);

		if (tempset)
		{
			for (i=0; i < tempset->nwords; i++)
				attxor ^= tempset->words[i];
		}
		tc_head->cached_attrs = (Bitmapset *)((char *)tc_head + offset);

		memset(tc_head->cached_attrs, 0, offsetof(Bitmapset, words[nwords]));
		tc_head->cached_attrs->nwords = nwords;
		if (tempset)
		{
			Assert(tempset->nwords <= nwords);
			memcpy(tc_head->cached_attrs, tempset,
				   offsetof(Bitmapset, words[tempset->nwords]));
		}
		offset += MAXALIGN(offsetof(Bitmapset, words[nwords]));

		/* setting up locktag */
		tc_head->locktag.locktag_field1 = MyDatabaseId;
		tc_head->locktag.locktag_field2 = reloid;
		tc_head->locktag.locktag_field3 = attxor;
		tc_head->locktag.locktag_field4 = 0;
		tc_head->locktag.locktag_type = LOCKTAG_RELATION_EXTEND;
		tc_head->locktag.locktag_lockmethodid = DEFAULT_LOCKMETHOD;

		/* setting up tuple-descriptor */
		tupdesc = (TupleDesc)((char *)tc_head + offset);
		memset(tupdesc, 0, sizeof(*tupdesc));
		offset += MAXALIGN(sizeof(*tupdesc));

		tupdesc->natts = relform->relnatts;
		tupdesc->attrs = (Form_pg_attribute *)((char *)tc_head + offset);
		offset += MAXALIGN(sizeof(Form_pg_attribute) * relform->relnatts);
		tupdesc->tdtypeid = relform->reltype;
		tupdesc->tdtypmod = -1;
		tupdesc->tdhasoid = relform->relhasoids;
		tupdesc->tdrefcount = -1;

		for (i=0; i < tupdesc->natts; i++)
		{
			atttup = SearchSysCache2(ATTNUM,
									 ObjectIdGetDatum(reloid),
									 Int16GetDatum(i+1));
			if (!HeapTupleIsValid(atttup))
				elog(ERROR, "cache lookup failed for attr %d of relation %u",
					 i+1, reloid);
			tupdesc->attrs[i] = (Form_pg_attribute)((char *)tc_head + offset);
			offset += MAXALIGN(sizeof(FormData_pg_attribute));
			memcpy(tupdesc->attrs[i], GETSTRUCT(atttup),
				   sizeof(FormData_pg_attribute));
			tupdesc->attrs[i]->attlen
				= pgstrom_try_varlena_inline(tupdesc->attrs[i]);
			ReleaseSysCache(atttup);
		}
		Assert(offset <= length);
		tc_head->tupdesc = tupdesc;
		bms_free(tempset);

		/* remaining area shall be used to tcache_node */
		while (offset + sizeof(tcache_node) < allocated)
		{
			tcache_node *tc_node
				= (tcache_node *)((char *)tc_head + offset);

			dlist_push_tail(&tc_head->free_list, &tc_node->chain);
			offset += MAXALIGN(sizeof(tcache_node));
		}

		/* also, allocate first empty tcache node as root */
		tc_head->tcs_root = tcache_alloc_tcnode(tc_head);
	}
	PG_CATCH();
	{
		pgstrom_shmem_free(tc_head);
		PG_RE_THROW();
	}
	PG_END_TRY();
	ReleaseSysCache(reltup);

	return tc_head;
}


static void
tcache_put_tchead_nolock(tcache_head *tc_head)
{
	if (--tc_head->refcnt == 0)
	{
		dlist_mutable_iter iter;

		if (dnode_is_linked(&tc_head->chain))
		{
			dlist_delete(&tc_head->chain);
			memset(&tc_head->chain, 0, sizeof(dlist_node));
		}
		if (dnode_is_linked(&tc_head->lru_chain))
		{
			dlist_delete(&tc_head->lru_chain);
			memset(&tc_head->lru_chain, 0, sizeof(dlist_node));
		}

		/* release tcache_node root recursively */
		tcache_free_node_recurse(tc_head, tc_head->tcs_root);

		/* release blocks allocated for tcache_node */
		dlist_foreach_modify(iter, &tc_head->block_list)
		{
#ifdef USE_ASSERT_CHECKING
			int		i;
			tcache_node	*tc_node = (tcache_node *)(iter.cur + 1);

			/*
			 * all the blocks should be already released
			 * (to be linked at tc_head->free_list)
			 */
			for (i=0; i < TCACHE_NODE_PER_BLOCK_BARE; i++)
				Assert(dnode_is_linked(&tc_node[i].chain));
#endif
			pgstrom_shmem_free(iter.cur);
		}
		/* TODO: also check tc_nodes behind of the tc_head */

		/* also, all the row-store should be released */
		Assert(dlist_is_empty(&tc_head->trs_list));
		
		pgstrom_shmem_free(tc_head);
	}
}

static tcache_head *
tcache_get_tchead_internal(Oid datoid, Oid reloid,
						   Bitmapset *required,
						   bool create_on_demand)
{
	dlist_iter		iter;
	tcache_head	   *tc_head = NULL;
	tcache_head	   *tc_old = NULL;
	int				hindex = tcache_hash_index(MyDatabaseId, reloid);

	SpinLockAcquire(&tc_common->lock);
	PG_TRY();
	{
		dlist_foreach(iter, &tc_common->slot[hindex])
		{
			tcache_head	   *temp
				= dlist_container(tcache_head, chain, iter.cur);

			if (temp->datoid == MyDatabaseId &&
				temp->reloid == reloid)
			{
				Bitmapset  *referenced
					= bms_fixup_sysattrs(temp->tupdesc->natts, required);

				if (bms_is_subset(referenced, temp->cached_attrs))
				{
					/*
					 * Perfect! Cache of the target relation exists and all
					 * the required columns are cached.
					 */
					temp->refcnt++;
					dlist_move_head(&tc_common->lru_list, &temp->lru_chain);
					tc_head = temp;
				}
				else
				{
					/*
					 * Elsewhere, cache exists towards the required relation
					 * but all the required columns are not cached in-memory.
					 */
					tc_old = temp;
				}
				bms_free(referenced);
				break;
			}
		}

		if (!tc_head && create_on_demand)
		{
			tc_head = tcache_create_tchead(reloid, required, tc_old);
			if (tc_head)
			{
				/* add this tcache_head to the hash table */
				dlist_push_head(&tc_common->slot[hindex], &tc_head->chain);
				dlist_push_head(&tc_common->lru_list, &tc_head->lru_chain);

				/* also, old tcache_head is unlinked */
				if (tc_old)
				{
					dlist_delete(&tc_old->chain);
					dlist_delete(&tc_old->lru_chain);
					memset(&tc_old->chain, 0, sizeof(dlist_node));
					memset(&tc_old->lru_chain, 0, sizeof(dlist_node));
					tcache_put_tchead_nolock(tc_old);
				}
			}
		}
	}
	PG_CATCH();
	{
		SpinLockRelease(&tc_common->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	SpinLockRelease(&tc_common->lock);

	return tc_head;
}

tcache_head *
tcache_get_tchead(Oid reloid, Bitmapset *required)
{
	return tcache_get_tchead_internal(MyDatabaseId, reloid, required, false);
}

tcache_head *
tcache_try_create_tchead(Oid reloid, Bitmapset *required)
{
	return tcache_get_tchead_internal(MyDatabaseId, reloid, required, true);
}

void
tcache_put_tchead(tcache_head *tc_head)
{
	SpinLockAcquire(&tc_common->lock);
	tcache_put_tchead_nolock(tc_head);
	SpinLockRelease(&tc_common->lock);
}

void
tcache_abort_tchead(tcache_head *tc_head, Datum private)
{
	bool	is_builder = DatumGetBool(private);

	/*
	 * In case when a builder process got aborted until tcache
	 * build does not become ready, it means this tcache has
	 * "half-constructed" state. We have, right now, no way to
	 * continue tcache building using the half-constructed cache,
	 * so we simply reset it, then someone later will re-built
	 * it again, from the scratch.
	 */
	if (is_builder && !tcache_state_is_ready(tc_head))
		tcache_reset_tchead(tc_head);

	tcache_put_tchead(tc_head);
}

/*
 * tcache_state_is_ready
 *
 * it asks tcache state; whether it is ready or not.
 */
bool
tcache_state_is_ready(tcache_head *tc_head)
{
	bool	is_ready;

	SpinLockAcquire(&tc_head->lock);
	is_ready = tc_head->is_ready;
	SpinLockRelease(&tc_head->lock);

	return is_ready;
}

static void
tcache_reset_tchead(tcache_head *tc_head)
{
	dlist_mutable_iter	iter;
	tcache_node		   *tc_node;
	tcache_column_store *tcs;
	int					i;

	tcache_lock_held_by_me(tc_head, true);

	/* no need to columnize any more! */
	SpinLockAcquire(&tc_common->lock);
	if (dnode_is_linked(&tc_head->pending_chain))
	{
		dlist_delete(&tc_head->pending_chain);
		memset(&tc_head->pending_chain, 0, sizeof(dlist_node));
	}
	SpinLockRelease(&tc_common->lock);

	/* release column store tree, except for the root */
	tc_node = tc_head->tcs_root;
	if (tc_node->right)
		tcache_free_node_recurse(tc_head, tc_node->right);
	if (tc_node->left)
		tcache_free_node_recurse(tc_head, tc_node->left);
	tc_node->right = NULL;
	tc_node->left = NULL;
	tc_node->r_depth = 0;
	tc_node->l_depth = 0;

	/* reset root node */
	tcs = tc_node->tcs;
	tcs->nrows = 0;
	tcs->njunks = 0;
	for (i=0; i < tcs->ncols; i++)
	{
		tcache_toastbuf *tbuf = tcs->cdata[i].toast;

		if (tbuf)
		{
			tbuf->tbuf_usage = offsetof(tcache_toastbuf, data[0]);
			tbuf->tbuf_junk = 0;
		}
	}

	/* row-store should be also released */
	SpinLockAcquire(&tc_head->lock);
	dlist_foreach_modify(iter, &tc_head->trs_list)
	{
		tcache_row_store   *trs
			= dlist_container(tcache_row_store, chain, iter.cur);
		dlist_delete(&trs->chain);
		pgstrom_put_row_store(trs);
	}
	Assert(dlist_is_empty(&tc_head->trs_list));
	if (tc_head->trs_curr)
		pgstrom_put_row_store(tc_head->trs_curr);
	tc_head->trs_curr = NULL;

	/* all clear, this cache become 'not ready' again */
	tc_head->is_ready = false;
	SpinLockRelease(&tc_head->lock);
}



/*
 * pgstrom_tcache_synchronizer
 *
 * trigger function to be called after INSERT, UPDATE, DELETE for each row
 * or TRUNCATE statement, to keep consistency of tcache.
 */
Datum
pgstrom_tcache_synchronizer(PG_FUNCTION_ARGS)
{
	TriggerData    *trigdata;
	Oid				tgrel_oid;
	HeapTuple		result = NULL;
	tcache_head	   *tc_head;

	if (!CALLED_AS_TRIGGER(fcinfo))
		elog(ERROR, "%s: not fired by trigger manager", __FUNCTION__);

	trigdata = (TriggerData *) fcinfo->context;
	tgrel_oid = RelationGetRelid(trigdata->tg_relation);
	tc_head = tcache_get_tchead(tgrel_oid, NULL);
	if (!tc_head)
		return PointerGetDatum(trigdata->tg_newtuple);

	PG_TRY();
	{
		TriggerEvent	tg_event = trigdata->tg_event;

		/*
		 * TODO: it may make sense if we can add this tuple into column-
		 * store directly, in case when column-store has at least one
		 * slot to store the new tuple.
		 */
		tcache_lock_tchead(tc_head, false);

		if (TRIGGER_FIRED_AFTER(tg_event) &&
			TRIGGER_FIRED_FOR_ROW(tg_event) &&
			TRIGGER_FIRED_BY_INSERT(tg_event))
		{
			/* after insert for each row */
			tcache_insert_tuple_row(tc_head, trigdata->tg_trigtuple);
			result = trigdata->tg_trigtuple;
		}
		else if (TRIGGER_FIRED_AFTER(tg_event) &&
				 TRIGGER_FIRED_FOR_ROW(tg_event) &&
				 TRIGGER_FIRED_BY_UPDATE(tg_event))
		{
			/* after update for each row */
			tcache_update_tuple_hints(tc_head, trigdata->tg_trigtuple);
			tcache_insert_tuple_row(tc_head, trigdata->tg_newtuple);
			result = trigdata->tg_newtuple;
		}
		else if (TRIGGER_FIRED_AFTER(tg_event) &&
				 TRIGGER_FIRED_FOR_ROW(tg_event) &&
				 TRIGGER_FIRED_BY_DELETE(tg_event))
        {
			/* after delete for each row */
			tcache_update_tuple_hints(tc_head, trigdata->tg_trigtuple);
			result = trigdata->tg_trigtuple;
		}
		else if (TRIGGER_FIRED_AFTER(tg_event) &&
				 TRIGGER_FIRED_FOR_STATEMENT(tg_event) &&
				 TRIGGER_FIRED_BY_TRUNCATE(tg_event))
		{
			/* after truncate for statement */
			tcache_put_tchead(tc_head);
		}
		else
			elog(ERROR, "%s: fired on unexpected context (%08x)",
				 trigdata->tg_trigger->tgname, tg_event);
	}
	PG_CATCH();
	{
		tcache_unlock_tchead(tc_head, false);
		tcache_put_tchead(tc_head);
		PG_RE_THROW();
	}
	PG_END_TRY();
	tcache_unlock_tchead(tc_head, false);
	tcache_put_tchead(tc_head);

	PG_RETURN_POINTER(result);
}
PG_FUNCTION_INFO_V1(pgstrom_tcache_synchronizer);

/*
 * pgstrom_assign_synchronizer
 *
 * It shall be called for each creation of relations, to assign trigger
 * functions to keep tcache status.
 */
static void
pgstrom_assign_synchronizer(Oid reloid)
{
	Relation	class_rel;
	Relation	tgrel;
	ScanKeyData	skey;
	SysScanDesc	sscan;
	Form_pg_class class_form;
	Datum		values[Natts_pg_trigger];
	bool		isnull[Natts_pg_trigger];
	HeapTuple	tuple;
	HeapTuple	tgtup;
	Oid			funcoid;
	Oid			tgoid;
	ObjectAddress myself;
	ObjectAddress referenced;
	const char *funcname = "pgstrom_tcache_synchronizer";
	const char *tgname_s = "pgstrom_tcache_sync_stmt";
	const char *tgname_r = "pgstrom_tcache_sync_row";

	/*
	 * Fetch a relation tuple (probably) needs to be updated
	 *
	 * TODO: add description the reason why to use inplace_update
	 *
	 *
	 */
	class_rel = heap_open(RelationRelationId, RowExclusiveLock);

	ScanKeyInit(&skey,
				ObjectIdAttributeNumber,
				BTEqualStrategyNumber, F_OIDEQ,
				ObjectIdGetDatum(reloid));

	sscan = systable_beginscan(class_rel, ClassOidIndexId, true,
							   SnapshotSelf, 1, &skey);
	tuple = systable_getnext(sscan);
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "catalog lookup failed for relation %u", reloid);
	class_form = (Form_pg_class) GETSTRUCT(tuple);

	/* only regular (none-toast) relation has synchronizer */
	if (class_form->relkind != RELKIND_RELATION)
		goto skip_make_trigger;
	/* we don't support synchronizer on system tables */
	if (class_form->relnamespace == PG_CATALOG_NAMESPACE)
		goto skip_make_trigger;

	/* OK, this relation should have tcache synchronizer */


	/* Lookup synchronizer function */
	funcoid = GetSysCacheOid3(PROCNAMEARGSNSP,
							  PointerGetDatum(funcname),
							  PointerGetDatum(buildoidvector(NULL, 0)),
							  ObjectIdGetDatum(PG_PUBLIC_NAMESPACE));
	if (!OidIsValid(funcoid))
	{
		ereport(INFO,
				(errcode(ERRCODE_UNDEFINED_FUNCTION),
				 errmsg("cache lookup failed for trigger function: %s",
						funcname),
				 errhint("Try 'CREATE EXTENSION pg_strom;'")));
		goto skip_make_trigger;
	}

	/*
	 * OK, let's construct trigger definitions
	 */
	tgrel = heap_open(TriggerRelationId, RowExclusiveLock);

	/*
	 * construct a tuple of statement level synchronizer
	 */
	memset(isnull, 0, sizeof(isnull));
	values[Anum_pg_trigger_tgrelid - 1] = ObjectIdGetDatum(reloid);
	values[Anum_pg_trigger_tgname - 1]
		= DirectFunctionCall1(namein, CStringGetDatum(tgname_s));
	values[Anum_pg_trigger_tgfoid - 1] = ObjectIdGetDatum(funcoid);
	values[Anum_pg_trigger_tgtype - 1]
		= Int16GetDatum(TRIGGER_TYPE_TRUNCATE);
	values[Anum_pg_trigger_tgenabled - 1]
		= CharGetDatum(TRIGGER_FIRES_ON_ORIGIN);
	values[Anum_pg_trigger_tgisinternal - 1] = BoolGetDatum(true);
	values[Anum_pg_trigger_tgconstrrelid - 1] = ObjectIdGetDatum(InvalidOid);
	values[Anum_pg_trigger_tgconstrindid - 1] = ObjectIdGetDatum(InvalidOid);
	values[Anum_pg_trigger_tgconstraint - 1] = ObjectIdGetDatum(InvalidOid);
	/*
	 * XXX - deferrable trigger may make sense for cache invalidation
	 * because transaction might be aborted later, In this case, it is
	 * waste of time to re-construct columnar-cache again.
	 */
	values[Anum_pg_trigger_tgdeferrable - 1] = BoolGetDatum(false);
	values[Anum_pg_trigger_tginitdeferred - 1] = BoolGetDatum(false);

	values[Anum_pg_trigger_tgnargs - 1] = Int16GetDatum(0);
	values[Anum_pg_trigger_tgargs - 1]
		= DirectFunctionCall1(byteain, CStringGetDatum(""));
	values[Anum_pg_trigger_tgattr - 1]
		= PointerGetDatum(buildint2vector(NULL, 0));
	isnull[Anum_pg_trigger_tgqual - 1] = true;

	tgtup = heap_form_tuple(tgrel->rd_att, values, isnull);
	tgoid = simple_heap_insert(tgrel, tgtup);
	CatalogUpdateIndexes(tgrel, tgtup);

	/* record dependency on the statement-level trigger */
	myself.classId = TriggerRelationId;
	myself.objectId = tgoid;
	myself.objectSubId = 0;

	referenced.classId = ProcedureRelationId;
	referenced.objectId = funcoid;
	referenced.objectSubId = 0;
	recordDependencyOn(&myself, &referenced, DEPENDENCY_NORMAL);

	referenced.classId = RelationRelationId;
	referenced.objectId = reloid;
	referenced.objectSubId = 0;
	recordDependencyOn(&myself, &referenced, DEPENDENCY_AUTO);

	heap_freetuple(tgtup);

	/*
	 * also, a tuple for row-level synchronizer
	 */
	values[Anum_pg_trigger_tgname - 1]
		= DirectFunctionCall1(namein, CStringGetDatum(tgname_r));
	values[Anum_pg_trigger_tgtype - 1]
		= Int16GetDatum(TRIGGER_TYPE_ROW |
						TRIGGER_TYPE_INSERT |
						TRIGGER_TYPE_DELETE |
						TRIGGER_TYPE_UPDATE);
	tgtup = heap_form_tuple(tgrel->rd_att, values, isnull);
	tgoid = simple_heap_insert(tgrel, tgtup);
	CatalogUpdateIndexes(tgrel, tgtup);

	/* record dependency on the row-level trigger */
	myself.classId = TriggerRelationId;
	myself.objectId = tgoid;
	myself.objectSubId = 0;

	referenced.classId = ProcedureRelationId;
	referenced.objectId = funcoid;
	referenced.objectSubId = 0;
	recordDependencyOn(&myself, &referenced, DEPENDENCY_NORMAL);

	referenced.classId = RelationRelationId;
	referenced.objectId = reloid;
	referenced.objectSubId = 0;
	recordDependencyOn(&myself, &referenced, DEPENDENCY_AUTO);

	heap_freetuple(tgtup);

	heap_close(tgrel, NoLock);

	/*
	 * We also need to put a flag of 'relhastriggers'. This is new relation
	 * uncommitted, so it is obvious that nobody touched this catalog.
	 * So, we can apply heap_inplace_update(), instead of the regular
	 * operations.
	 */
	if (!class_form->relhastriggers)
	{
		class_form->relhastriggers = true;
		heap_inplace_update(class_rel, tuple);
		//CatalogUpdateIndexes(class_rel, tuple);
	}
skip_make_trigger:
	systable_endscan(sscan);
	heap_close(class_rel, NoLock);
}

/*
 * pgstrom_relation_has_synchronizer
 *
 * A table that can have columnar-cache also needs to have trigger to
 * synchronize the in-memory cache and heap. It returns true, if supplied
 * relation has triggers that invokes pgstrom_tcache_synchronizer on
 * appropriate context.
 */
bool
pgstrom_relation_has_synchronizer(Relation rel)
{
	int		i, numtriggers;
	bool	has_on_insert_synchronizer = false;
	bool	has_on_update_synchronizer = false;
	bool	has_on_delete_synchronizer = false;
	bool	has_on_truncate_synchronizer = false;

	if (!rel->trigdesc)
		return false;

	numtriggers = rel->trigdesc->numtriggers;
	for (i=0; i < numtriggers; i++)
	{
		Trigger	   *trig = rel->trigdesc->triggers + i;
		HeapTuple	tup;

		if (!trig->tgenabled)
			continue;

		tup = SearchSysCache1(PROCOID, ObjectIdGetDatum(trig->tgfoid));
		if (!HeapTupleIsValid(tup))
			elog(ERROR, "cache lookup failed for function %u", trig->tgfoid);

		if (((Form_pg_proc) GETSTRUCT(tup))->prolang == ClanguageId)
		{
			Datum		value;
			bool		isnull;
			char	   *prosrc;
			char	   *probin;

			value = SysCacheGetAttr(PROCOID, tup,
									Anum_pg_proc_prosrc, &isnull);
			if (isnull)
				elog(ERROR, "null prosrc for C function %u", trig->tgoid);
			prosrc = TextDatumGetCString(value);

			value = SysCacheGetAttr(PROCOID, tup,
									Anum_pg_proc_probin, &isnull);
			if (isnull)
				elog(ERROR, "null probin for C function %u", trig->tgoid);
			probin = TextDatumGetCString(value);

			if (strcmp(prosrc, "pgstrom_tcache_synchronizer") == 0 &&
				strcmp(probin, "$libdir/pg_strom") == 0)
			{
				int16       tgtype = trig->tgtype;

				if (TRIGGER_TYPE_MATCHES(tgtype,
										 TRIGGER_TYPE_ROW,
										 TRIGGER_TYPE_AFTER,
										 TRIGGER_TYPE_INSERT))
					has_on_insert_synchronizer = true;
				if (TRIGGER_TYPE_MATCHES(tgtype,
										 TRIGGER_TYPE_ROW,
										 TRIGGER_TYPE_AFTER,
										 TRIGGER_TYPE_UPDATE))
					has_on_update_synchronizer = true;
				if (TRIGGER_TYPE_MATCHES(tgtype,
										 TRIGGER_TYPE_ROW,
										 TRIGGER_TYPE_AFTER,
										 TRIGGER_TYPE_DELETE))
					has_on_delete_synchronizer = true;
				if (TRIGGER_TYPE_MATCHES(tgtype,
										 TRIGGER_TYPE_STATEMENT,
										 TRIGGER_TYPE_AFTER,
										 TRIGGER_TYPE_TRUNCATE))
					has_on_truncate_synchronizer = true;
			}
			pfree(prosrc);
			pfree(probin);
		}
		ReleaseSysCache(tup);
	}

	if (has_on_insert_synchronizer &&
		has_on_update_synchronizer &&
		has_on_delete_synchronizer &&
		has_on_truncate_synchronizer)
		return true;
	return false;
}

/*
 * tcache_vacuum_heappage_column
 *
 * callback function for each heap-page. Caller should already acquires
 * shared-lock on the tcache_head, so it is prohibited to modify tree-
 * structure. All we can do is mark particular records as junk.
 */
static void
tcache_vacuum_column_store(tcache_head *tc_head, Buffer buffer)
{
	BlockNumber		blknum = BufferGetBlockNumber(buffer);
	OffsetNumber	offnum = InvalidOffsetNumber;
	Page			page = BufferGetPage(buffer);
	ItemPointerData	ctid;
	ItemId			itemid;
	tcache_node	   *tc_node;
	tcache_column_store *tcs;
	int				index;

	tc_node = tcache_find_next_node(tc_head, blknum);
	if (!tc_node)
		return;

	SpinLockAcquire(&tc_node->lock);
	if (!tc_node->tcs->is_sorted)
		tcache_sort_tcnode(tc_head, tc_node, false);
	tcs = tcache_get_column_store(tc_node->tcs);
	Assert(tcs->is_sorted);

	ItemPointerSet(&ctid, blknum, FirstOffsetNumber);
	index = tcache_find_next_record(tcs, &ctid);
	if (index < 0)
	{
		tcache_put_column_store(tcs);
		SpinLockRelease(&tc_node->lock);
		return;
	}

	while (index < tcs->nrows &&
		   ItemPointerGetBlockNumber(&tcs->ctids[index]) == blknum)
	{
		offnum = ItemPointerGetOffsetNumber(&tcs->ctids[index]);
		itemid = PageGetItemId(page, offnum);

		if (!ItemIdIsNormal(itemid))
		{
			/* find an actual item-pointer, if redirected */
			while (ItemIdIsRedirected(itemid))
				itemid = PageGetItemId(page, ItemIdGetRedirect(itemid));

			if (ItemIdIsNormal(itemid))
			{
				/* needs to update item-pointer */
				ItemPointerSetOffsetNumber(&tcs->ctids[index],
										   ItemIdGetOffset(itemid));
				/*
				 * if this offset update breaks pre-sorted array,
				 * we have to set is_sorted = false;
				 */
				if (tcs->is_sorted &&
					((index > 0 &&
					  ItemPointerCompare(&tcs->ctids[index - 1],
										 &tcs->ctids[index]) > 0) ||
					 (index < tcs->nrows &&
					  ItemPointerCompare(&tcs->ctids[index + 1],
										 &tcs->ctids[index]) < 0)))
					tcs->is_sorted = false;
			}
			else
			{
				/* remove this record from the column-store */
				HeapTupleHeaderSetXmax(&tcs->theads[index],
									   FrozenTransactionId);
			}
		}
		index++;
	}
	tcache_put_column_store(tcs);
	SpinLockRelease(&tc_node->lock);
}

/*
 * tcache_vacuum_row_store
 *
 *
 *
 */
static void
do_vacuum_row_store(tcache_row_store *trs, Buffer buffer)
{
	BlockNumber		blknum = BufferGetBlockNumber(buffer);
	OffsetNumber	offnum;
	ItemId			itemid;
	Page			page;
	cl_uint			index;

	if (blknum < trs->blkno_min || trs->blkno_max > blknum)
		return;

	page = BufferGetPage(buffer);
	for (index=0; index < trs->kern.nrows; index++)
	{
		rs_tuple   *rs_tup
			= kern_rowstore_get_tuple(&trs->kern, index);

		if (!rs_tup ||
			ItemPointerGetBlockNumber(&rs_tup->htup.t_self) != blknum)
			continue;

		offnum = ItemPointerGetOffsetNumber(&rs_tup->htup.t_self);
		itemid = PageGetItemId(page, offnum);

		if (!ItemIdIsNormal(itemid))
		{
			/* find an actual item-pointer, if redirected */
			while (ItemIdIsRedirected(itemid))
				itemid = PageGetItemId(page, ItemIdGetRedirect(itemid));

			if (ItemIdIsNormal(itemid))
			{
				/* needs to update item-pointer */
				ItemPointerSetOffsetNumber(&rs_tup->htup.t_self,
										   ItemIdGetOffset(itemid));
			}
			else
			{
				/* remove this record from the column store */
				cl_uint	   *tupoffset
					= kern_rowstore_get_offset(&trs->kern);

				tupoffset[index] = 0;
			}
		}
	}
}

static void
tcache_vacuum_row_store(tcache_head *tc_head, Buffer buffer)
{
	dlist_iter	iter;

	SpinLockAcquire(&tc_head->lock);
	if (tc_head->trs_curr)
		do_vacuum_row_store(tc_head->trs_curr, buffer);
	dlist_foreach(iter, &tc_head->trs_list)
	{
		tcache_row_store *trs
			= dlist_container(tcache_row_store, chain, iter.cur);
		do_vacuum_row_store(trs, buffer);
	}
	SpinLockRelease(&tc_head->lock);
}

/*
 * tcache_on_page_prune
 *
 * 
 *
 *
 */
static void
tcache_on_page_prune(Relation relation,
					 Buffer buffer,
					 int ndeleted,
					 TransactionId OldestXmin,
					 TransactionId latestRemovedXid)
{
	tcache_head	   *tc_head;

	if (heap_page_prune_hook_next)
		heap_page_prune_hook_next(relation, buffer, ndeleted,
								  OldestXmin, latestRemovedXid);

	tc_head = tcache_get_tchead(RelationGetRelid(relation), NULL);
	if (tc_head)
	{
		/*
		 * At least, we need to acquire shared-lock on the tcache_head,
		 * but no need for exclusive-lock because vacuum page never
		 * create or drop tcache_nodes. Per node level spinlock is
		 * sufficient to do.
		 * Note that, vacuumed records are marked as junk, then columnizer
		 * actually removes them from the cache later, under the exclusive
		 * lock.
		 */
		tcache_trylock_tchead(tc_head, false);

		tcache_vacuum_row_store(tc_head, buffer);
		tcache_vacuum_column_store(tc_head, buffer);

		tcache_unlock_tchead(tc_head, false);
	}
}

/*
 * tcache_on_object_access
 *
 * It invalidates an existing columnar-cache if cached tables were altered
 * or dropped. Also, it enforces to assign synchronizer trigger on new table
 * creation/
 */
static void
tcache_on_object_access(ObjectAccessType access,
						Oid classId,
						Oid objectId,
						int subId,
						void *arg)
{

	if (object_access_hook_next)
		object_access_hook_next(access, classId, objectId, subId, arg);

	/*
	 * Only relations, we are interested in
	 */
	if (classId == RelationRelationId)
	{
		if (access == OAT_POST_CREATE)
		{
			/*
			 * We consider to assign synchronizer trigger on statement-
			 * and row-level. It is needed to synchronize / invalidate
			 * cached object being constructed.
			 */
			pgstrom_assign_synchronizer(objectId);
		}
		else if (access == OAT_DROP || access == OAT_POST_ALTER)
		{
			/*
			 * Existing columnar-cache is no longer available across
			 * DROP or ALTER command (TODO: it's depend on the context.
			 * it may possible to have existing cache, if ALTER command
			 * does not change something related to the cached columns).
			 * So, we simply unlink the tcache_head associated with this
			 * relation, eventually someone who decrement its reference
			 * counter into zero releases the cache.
			 */
			int		hindex = tcache_hash_index(MyDatabaseId, objectId);

			SpinLockAcquire(&tc_common->lock);
			PG_TRY();
			{
				dlist_mutable_iter	iter;

				dlist_foreach_modify(iter, &tc_common->slot[hindex])
				{
					tcache_head	   *tc_head
						= dlist_container(tcache_head, chain, iter.cur);

					/* XXX - usually, only one cache per relation is linked */
					if (tc_head->datoid == MyDatabaseId &&
						tc_head->reloid == objectId)
						tcache_put_tchead_nolock(tc_head);
				}
			}
			PG_CATCH();
			{
				SpinLockRelease(&tc_common->lock);
				PG_RE_THROW();
			}
			PG_END_TRY();
			SpinLockRelease(&tc_common->lock);
		}
	}
}







static void
pgstrom_wakeup_columnizer(bool wakeup_all)
{
	dlist_iter	iter;

	SpinLockAcquire(&tc_common->lock);
	dlist_foreach(iter, &tc_common->inactive_list)
	{
		tcache_columnizer  *columnizer
			= dlist_container(tcache_columnizer, chain, iter.cur);

		SetLatch(columnizer->latch);
		if (!wakeup_all)
			break;
	}
    SpinLockRelease(&tc_common->lock);
}

static void
pgstrom_columnizer_main(Datum index)
{
	tcache_columnizer  *columnizer;
	int		rc;

	Assert(tc_common != NULL);
	Assert(index < num_columnizers);

	columnizer = &tc_common->columnizers[index];
	memset(columnizer, 0, sizeof(tcache_columnizer));
	columnizer->pid = getpid();
	columnizer->latch = &MyProc->procLatch;

	SpinLockAcquire(&tc_common->lock);
	dlist_push_tail(&tc_common->inactive_list, &columnizer->chain);
	SpinLockRelease(&tc_common->lock);

	/* We're now ready to receive signals */
	BackgroundWorkerUnblockSignals();

	while (true)
	{
		tcache_head		   *tc_head = NULL;
		tcache_node		   *tc_node;
		tcache_row_store   *trs;
		dlist_node	*dnode;

	retry:
		SpinLockAcquire(&tc_common->lock);
		if (dlist_is_empty(&tc_common->pending_list))
		{
			dlist_push_tail(&tc_common->inactive_list, &columnizer->chain);
			SpinLockRelease(&tc_common->lock);

			ResetLatch(&MyProc->procLatch);
			rc = WaitLatch(&MyProc->procLatch,
						   WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
						   15 * 1000);  /* wake up per 15s at least */
			if (rc & WL_POSTMASTER_DEATH)
				return;
			goto retry;
		}
		else
		{
			dnode = dlist_pop_head_node(&tc_common->pending_list);
			tc_head = dlist_container(tcache_head, pending_chain, dnode);
			tc_head->refcnt++;
			columnizer->datoid = tc_head->datoid;
			columnizer->reloid = tc_head->reloid;

			dlist_delete(&columnizer->chain);
		}
		SpinLockRelease(&tc_common->lock);

		if (!tc_head)
			continue;

		/*
		 * TODO: add error handler routine
		 */
		tcache_lock_tchead(tc_head, true);
		// SpinLockAcquire(&tc_head->lock);	/* probably unneeded */
		PG_TRY();
		{
			if (!dlist_is_empty(&tc_head->trs_list))
			{
				int		index;

				dnode = dlist_pop_head_node(&tc_head->trs_list);
				trs = dlist_container(tcache_row_store, chain, dnode);
				memset(&trs->chain, 0, sizeof(dlist_node));

				/*
				 * Move tuples in row-store into column-store
				 */
				for (index=0; index < trs->kern.nrows; index++)
				{
					rs_tuple *rs_tup
						= kern_rowstore_get_tuple(&trs->kern, index);
					if (rs_tup)
						tcache_insert_tuple(tc_head,
											tc_head->tcs_root,
											&rs_tup->htup);
				}
				/* row-store shall be released */
				pgstrom_put_row_store(trs);
			}
			else if (!dlist_is_empty(&tc_head->pending_list))
			{
				dnode = dlist_pop_head_node(&tc_head->pending_list);
				tc_node = dlist_container(tcache_node, chain, dnode);
				memset(&tc_node->chain, 0, sizeof(dlist_node));

				tcache_compaction_tcnode(tc_head, tc_node);
				tcache_try_merge_tcnode(tc_head, tc_node);
			}
		}
		PG_CATCH();
		{
			tcache_unlock_tchead(tc_head, true);
			PG_RE_THROW();
		}
		PG_END_TRY();
		tcache_unlock_tchead(tc_head, true);

		/* OK, release this tcache_head */
		SpinLockAcquire(&tc_common->lock);
		columnizer->datoid = InvalidOid;
		columnizer->reloid = InvalidOid;
		tcache_put_tchead_nolock(tc_head);
		SpinLockRelease(&tc_common->lock);
	}
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
	memset(tc_common, 0, length);
	SpinLockInit(&tc_common->lock);
	dlist_init(&tc_common->lru_list);
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
		worker.bgw_main = pgstrom_columnizer_main;
		worker.bgw_main_arg = i;
		RegisterBackgroundWorker(&worker);
	}

	/* callback on vacuum-pages for cache invalidation */
	heap_page_prune_hook_next = heap_page_prune_hook;
	heap_page_prune_hook = tcache_on_page_prune;

	/* callback on object-access for cache invalidation */
	object_access_hook_next = object_access_hook;
	object_access_hook = tcache_on_object_access;

	/* aquires shared memory region */
	length = offsetof(tcache_common, columnizers[num_columnizers]);
	RequestAddinShmemSpace(MAXALIGN(length));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_tcache;
}

/*
 *
 * Utility functions to dump internal status
 *
 */
typedef struct
{
	Oid		datoid;
	Oid		reloid;
	int2vector *cached;
	int		refcnt;
	bool	is_ready;
} tcache_head_info;

Datum
pgstrom_tcache_info(PG_FUNCTION_ARGS)
{
	FuncCallContext	   *fncxt;
	tcache_head_info   *tchead_info;
	HeapTuple	tuple;
	Datum		values[6];
	bool		isnull[6];

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;
		List		   *info_list = NIL;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(6, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "datoid",
						   OIDOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "reloid",
						   OIDOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "cached",
						   INT2VECTOROID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "refcnt",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 5, "state",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 6, "lwlock",
						   TEXTOID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);
		SpinLockAcquire(&tc_common->lock);
		PG_TRY();
		{
			dlist_iter	iter;
			int			i;

			for (i=0; i < TCACHE_HASH_SIZE; i++)
			{
				dlist_foreach(iter, &tc_common->slot[i])
				{
					tcache_head *tc_head
						= dlist_container(tcache_head, chain, iter.cur);
					Bitmapset	*tempset = bms_copy(tc_head->cached_attrs);
					int			 j, k, n;
					int16		*anums;

					n = bms_num_members(tempset);
					anums = palloc0(sizeof(int16) * n);
					for (k=0; (j = bms_first_member(tempset)) >= 0; k++)
					{
						anums[k] = j - FirstLowInvalidHeapAttributeNumber;
					}
					Assert(n == k);

					tchead_info = palloc(sizeof(tcache_head_info));
					tchead_info->datoid = tc_head->datoid;
					tchead_info->reloid = tc_head->reloid;
					tchead_info->cached = buildint2vector(anums, n);
					tchead_info->refcnt = tc_head->refcnt;
					SpinLockAcquire(&tc_head->lock);
					tchead_info->is_ready = tc_head->is_ready;
					SpinLockRelease(&tc_head->lock);
					info_list = lappend(info_list, tchead_info);
				}
			}
		}
		PG_CATCH();
		{
			SpinLockRelease(&tc_common->lock);
			PG_RE_THROW();
		}
		PG_END_TRY();
		SpinLockRelease(&tc_common->lock);

		fncxt->user_fctx = info_list;

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	if (fncxt->user_fctx == NIL)
		SRF_RETURN_DONE(fncxt);

	tchead_info = linitial((List *) fncxt->user_fctx);
	fncxt->user_fctx = list_delete_first((List *)fncxt->user_fctx);

	memset(isnull, 0, sizeof(isnull));
	values[0] = ObjectIdGetDatum(tchead_info->datoid);
	values[1] = ObjectIdGetDatum(tchead_info->reloid);
	values[2] = PointerGetDatum(tchead_info->cached);
	values[3] = Int32GetDatum(tchead_info->refcnt);
	if (!tchead_info->is_ready)
		values[4] = CStringGetTextDatum("not ready");
	else
		values[4] = CStringGetTextDatum("ready");
	isnull[5] = true;

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_tcache_info);

typedef struct
{
	bool		row_store;
	void	   *addr;
	void	   *r_node;
	void	   *l_node;
	int			r_depth;
	int			l_depth;
	int			refcnt;
	int			nrows;
	size_t		usage;
	size_t		length;
	BlockNumber	blkno_min;
	BlockNumber	blkno_max;
} tcache_node_info;

static List *
collect_tcache_column_node_info(tcache_head *tc_head,
								tcache_node *tc_node,
								List *info_list)
{
	tcache_node_info    *tcnode_info;
	tcache_column_store *tcs;
	TupleDesc	tupdesc = tc_head->tupdesc;
	int			i;

	tcnode_info = palloc0(sizeof(tcache_node_info));
	tcnode_info->row_store = false;
	tcnode_info->addr = tc_node;
	tcnode_info->r_node = tc_node->right;
	tcnode_info->l_node = tc_node->left;
	tcnode_info->r_depth = tc_node->r_depth;
	tcnode_info->l_depth = tc_node->l_depth;

	SpinLockAcquire(&tc_node->lock);
    tcs = tc_node->tcs;
	SpinLockAcquire(&tcs->refcnt_lock);
	tcnode_info->refcnt = tcs->refcnt;
	SpinLockRelease(&tcs->refcnt_lock);
	tcnode_info->nrows = tcs->nrows;
	tcnode_info->usage =
		(STROMALIGN(offsetof(tcache_column_store, cdata[tcs->ncols])) +
		 STROMALIGN(sizeof(ItemPointerData) * tcs->nrows) +
		 STROMALIGN(sizeof(HeapTupleHeaderData) * tcs->nrows));
	tcnode_info->length =
		(STROMALIGN(offsetof(tcache_column_store, cdata[tcs->ncols])) +
		 STROMALIGN(sizeof(ItemPointerData) * NUM_ROWS_PER_COLSTORE) +
		 STROMALIGN(sizeof(HeapTupleHeaderData) * NUM_ROWS_PER_COLSTORE));
	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupdesc->attrs[i];

		if (!tcs->cdata[i].values)
			continue;

		if (!attr->attnotnull)
		{
			Assert(tcs->cdata[i].isnull != NULL);
			tcnode_info->usage +=
				STROMALIGN((tcs->nrows + BITS_PER_BYTE - 1) / BITS_PER_BYTE);
			tcnode_info->length +=
				STROMALIGN(NUM_ROWS_PER_COLSTORE / BITS_PER_BYTE);
		}
		if (attr->attlen > 0)
		{
			tcnode_info->usage += STROMALIGN(attr->attlen * tcs->nrows);
			tcnode_info->length += STROMALIGN(attr->attlen *
											  NUM_ROWS_PER_COLSTORE);
		}
		else
		{
			tcnode_info->usage += STROMALIGN(sizeof(cl_uint) * tcs->nrows);
			tcnode_info->length += STROMALIGN(sizeof(cl_uint) *
											  NUM_ROWS_PER_COLSTORE);
			if (tcs->cdata[i].toast)
			{
				tcnode_info->usage += tcs->cdata[i].toast->tbuf_usage;
				tcnode_info->length += tcs->cdata[i].toast->tbuf_length;
			}
		}
	}
	tcnode_info->blkno_min = tcs->blkno_min;
	tcnode_info->blkno_max = tcs->blkno_max;
	SpinLockRelease(&tc_node->lock);

	info_list = lappend(info_list, tcnode_info);

	if (tc_node->right)
		info_list = collect_tcache_column_node_info(tc_head,
													tc_node->right,
													info_list);
	if (tc_node->left)
		info_list = collect_tcache_column_node_info(tc_head,
													tc_node->left,
													info_list);
	return info_list;
}

static List *
collect_tcache_row_node_info(tcache_head *tc_head,
							 tcache_row_store *trs,
							 List *info_list)
{
	tcache_node_info   *tcnode_info = palloc0(sizeof(tcache_node_info));

	tcnode_info->row_store = true;
	tcnode_info->addr = trs;
	SpinLockAcquire(&trs->refcnt_lock);
	tcnode_info->refcnt = trs->refcnt;
	SpinLockRelease(&trs->refcnt_lock);
	tcnode_info->nrows = trs->kern.nrows;
	tcnode_info->usage = ROWSTORE_DEFAULT_SIZE;
	tcnode_info->blkno_min = trs->blkno_min;
	tcnode_info->blkno_max = trs->blkno_max;

	return lappend(info_list, tcnode_info);
}

static List *
collect_tcache_node_info(tcache_head *tc_head)
{
	List   *info_list = NIL;

	if (tc_head->tcs_root)
		info_list = collect_tcache_column_node_info(tc_head,
													tc_head->tcs_root,
													info_list);
	SpinLockAcquire(&tc_head->lock);
	PG_TRY();
	{
		tcache_row_store *trs;
		dlist_iter	iter;

		dlist_foreach(iter, &tc_head->trs_list)
		{
			trs = dlist_container(tcache_row_store, chain, iter.cur);

			info_list = collect_tcache_row_node_info(tc_head, trs, info_list);
		}
		if (tc_head->trs_curr)
			info_list = collect_tcache_row_node_info(tc_head,
													 tc_head->trs_curr,
													 info_list);
	}
	PG_CATCH();
	{
		SpinLockRelease(&tc_head->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	SpinLockRelease(&tc_head->lock);

	return info_list;
}

Datum
pgstrom_tcache_node_info(PG_FUNCTION_ARGS)
{
	Oid			reloid = PG_GETARG_OID(0);
	FuncCallContext *fncxt;
	tcache_node_info *tcnode_info;
	HeapTuple	tuple;
	Datum		values[12];
	bool		isnull[12];

	if (SRF_IS_FIRSTCALL())
    {
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;
		tcache_head	   *tc_head;
		List		   *info_list = NIL;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);

		tupdesc = CreateTemplateTupleDesc(12, false);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "type",
						   TEXTOID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "addr",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "l_node",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 4, "r_node",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 5, "l_depth",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 6, "r_depth",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 7, "refcnt",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 8, "nrows",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 9, "usage",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 10, "length",
						   INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 11, "blkno_min",
						   INT4OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 12, "blkno_max",
						   INT4OID, -1, 0);
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		tc_head = tcache_get_tchead(reloid, NULL);
		if (tc_head)
		{
			tcache_lock_tchead(tc_head, false);
			PG_TRY();
			{
				info_list = collect_tcache_node_info(tc_head);
			}
			PG_CATCH();
			{
				tcache_unlock_tchead(tc_head, false);
				tcache_put_tchead(tc_head);
				PG_RE_THROW();
			}
			PG_END_TRY();
			tcache_unlock_tchead(tc_head, false);
			tcache_put_tchead(tc_head);
		}
		fncxt->user_fctx = info_list;

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();

	if (fncxt->user_fctx == NIL)
		SRF_RETURN_DONE(fncxt);

	tcnode_info = linitial((List *) fncxt->user_fctx);
	fncxt->user_fctx = list_delete_first((List *)fncxt->user_fctx);

	memset(isnull, 0, sizeof(isnull));
	if (tcnode_info->row_store)
		values[0] = CStringGetTextDatum("row");
	else
		values[0] = CStringGetTextDatum("column");
	values[1] = Int64GetDatum(tcnode_info->addr);
	if (tcnode_info->row_store)
		isnull[2] = isnull[3] = isnull[4] = isnull[5] = true;
	else
	{
		values[2] = Int64GetDatum(tcnode_info->l_node);
		values[3] = Int64GetDatum(tcnode_info->r_node);
		values[4] = Int32GetDatum(tcnode_info->l_depth);
		values[5] = Int32GetDatum(tcnode_info->r_depth);
	}
	values[6] = Int32GetDatum(tcnode_info->refcnt);
	values[7] = Int32GetDatum(tcnode_info->nrows);
	values[8] = Int64GetDatum(tcnode_info->usage);
	values[9] = Int64GetDatum(tcnode_info->length);
	values[10] = Int32GetDatum(tcnode_info->blkno_min);
	values[11] = Int32GetDatum(tcnode_info->blkno_max);

	tuple = heap_form_tuple(fncxt->tuple_desc, values, isnull);

	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(pgstrom_tcache_node_info);
