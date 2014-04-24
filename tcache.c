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
	dlist_head	pending_list;	/* list of tc_head pending for columnization */
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

/*
 * static declarations
 */
static tcache_column_store *tcache_create_column_store(tcache_head *tc_head);
static tcache_column_store *tcache_duplicate_column_store(tcache_head *tc_head,
												  tcache_column_store *tcs_old,
												  bool duplicate_toastbuf);
static tcache_column_store *tcache_get_column_store(tcache_column_store *tcs);
static void tcache_put_column_store(tcache_column_store *tcs);

static tcache_toastbuf *tcache_create_toast_buffer(Size required);
static tcache_toastbuf *tcache_duplicate_toast_buffer(tcache_toastbuf *tbuf);
static tcache_toastbuf *tcache_get_toast_buffer(tcache_toastbuf *tbuf);
static void tcache_put_toast_buffer(tcache_toastbuf *tbuf);

/*
 * Misc utility functions
 */
static inline bool
TCacheHeadLockedByMe(tcache_head *tc_head, bool be_exclusive)
{
	bool	result = true;

	if (!LWLockHeldByMe(&tc_head->lwlock))
		return false;
	if (be_exclusive)
	{
		SpinLockAcquire(&tc_head->lwlock.mutex);
		if (tc_head->lwlock.exclusive == 0)
			result = false;
		SpinLockRelease(&tc_head->lwlock.mutex);
	}
	return result;
}

static inline void
memswap(void *x, void *y, Size len)
{
	union {
		cl_uchar	v_char;
		cl_ushort	v_short;
		cl_uint		v_uint;
		cl_ulong	v_ulong;
		char		v_misc[32];	/* our usage is up to 32bytes right now */
	} temp;

	switch (len)
	{
		case sizeof(cl_uchar):
			temp.v_char = *((cl_uchar *) x);
			*((cl_uchar *) x) = *((cl_uchar *) y);
			*((cl_uchar *) y) = temp.v_char;
			break;
		case sizeof(cl_ushort):
			temp.v_short = *((cl_ushort *) x);
			*((cl_ushort *) x) = *((cl_ushort *) y);
			*((cl_ushort *) y) = temp.v_short;
			break;
		case sizeof(cl_uint):
			temp.v_uint = *((cl_uint *) x);
			*((cl_uint *) x) = *((cl_uint *) y);
			*((cl_uint *) y) = temp.v_int;
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

	temp = (bitmap[x >> 3] & (1 << (x & 7)) != 0);

	if ((bitmap[y >> 3] &  (1 << (y & 7))) != 0)
		bitmap[x >> 3] |=  (1 << (x & 7));
	else
		bitmap[x >> 3] &= ~(1 << (x & 7));

	if (temp)
		bitmap[y >> 3] |=  (1 << (y & 7));
	else
		bitmap[y >> 3] &= ~(1 << (y & 7));
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
bitcopy(uint8 *dstmap, int dindex, uint8 *srcmap, int sindex)
{
	if ((srcmap[sindex >> 3] & (1 << (sindex & 7))) != 0)
		dstmap[dindex >> 3] |=  (1 << (dindex & 7));
	else
		dstmap[dindex >> 3] &= ~(1 << (dindex & 7));
}












/*
 * 
 *
 */
static tcache_column_store *
tcache_create_column_store(tcache_head *tc_head)
{
	Form_pg_attribute attr;
	Size	length;
	Size	offset;
	int		i, j;

	/* estimate length of column store */
	length = MAXALIGN(offsetof(tcache_column_store, cdata[natts]));
	length += MAXALIGN(sizeof(ItemPointerData) * NUM_ROWS_PER_COLSTORE);
	length += MAXALIGN(sizeof(HeapTupleHeaderData) * NUM_ROWS_PER_COLSTORE);

	for (i=0; i < tc_head->ncols; i++)
	{
		j = tc_head->i_cached[i];

		Assert(j >= 0 && j < tc_head->tupdesc->natts);
		attr = tc_head->tupdesc->attrs[j];
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
	memset(tcs, 0, sizeof(tcache_column_store));

	tcs->stag = StromTag_TCacheRowNode;
	SpinLockInit(&tcs->refcnt_lock);
	tcs->refcnt = 1;
	tcs->ncols = tc_head->ncols;

	offset = MAXALIGN(offsetof(tcache_column_store,
							   cdata[tcs->ncols]));
	/* array of item-pointers */
	tcs->ctids = (ItemPointerData *)((char *)tcs + offset);
	offset += MAXALIGN(sizeof(ItemPointerData) *
					   NUM_ROWS_PER_COLSTORE);
	/* array of other system columns */
	tcs->theads = (HeapTupleHeaderData *)((char *)tcs + offset);
	offset += MAXALIGN(sizeof(HeapTupleHeaderData) *
					   NUM_ROWS_PER_COLSTORE);
	/* array of user defined columns */
	for (i=0; i < tc_head->ncols; i++)
	{
		j = tc_head->i_cached[i];

		Assert(j >= 0 && j < tc_head->tupdesc->natts);
		attr = tc_head->tupdesc->attrs[j];
		if (attr->attnotnull)
			tcs->cdata[i].isnull = NULL;
		else
		{
			tcs->cdata[i].isnull = (char *)((char *)tcs + offset);
			offset += MAXALIGN(NUM_ROWS_PER_COLSTORE / BITS_PER_BYTE);
		}
		tcs->cdata[i].values = ((char *)tcs + offset);
		offset += MAXALIGN((attr->attlen > 0
							? attr->attlen
							: sizeof(cl_uint)) * NUM_ROWS_PER_COLSTORE);
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
	int		nrows = tcs_old->nrows;
	int		attlen;
	int		i, j;

	PG_TRY();
	{
		memcpy(tcs_new->ctids,
			   tcs_old->ctids,
			   sizeof(ItemPointerData) * nrows);
		memcpy(tcs_new->theads,
			   tcs_old->theads,
			   sizeof(HeapTupleHeaderData) * nrows);
		for (i=0; i < tc_head->col_natts; i++)
		{
			j = tc_head->i_cached[i];

			attlen = (tc_head->attrs[j].attlen > 0
					  ? tc_head->attrs[j].attlen
					  : sizeof(cl_uint));
			if (tcs->cdata[i].isnull)
			{
				memcpy(tcs_new->cdata[i].isnull,
					   tcs_old->cdata[i].isnull,
					   (nrows + BITS_PER_BYTE - 1) / BITS_PER_BYTE);
			}
			memcpy(tcs_new->cdata[i].values,
				   tcs_old->cdata[i].values,
				   attlen * nrows);

			if (tcs_old->cdata[i].toast)
			{
				tcache_toastbuf	*tbuf_old = tcs_old->cdata[i].toast;

				if (duplicate_toastbuf)
					tcs_new->cdata[i].toast =
						tcache_duplicate_toast_buffer(tbuf_old);
				else
					tcs_new->cdata[i].toast
						= tcache_get_toast_buffer(tbuf_old);
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

static tcache_column_store *
tcache_get_column_store(tcache_column_store *tcs)
{
	SpinLockAcquire(&tcs->refcnt_lock);
	Assert(tcs->refcnt > 0);
	tcs->refcnt++;
	SpinLockRelease(&tcs->refcnt_lock);

	return tcs;
}

static void
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
tcache_duplicate_toast_buffer(tcache_toastbuf *tbuf_old)
{
	tcache_toastbuf *tbuf_new;

	tbuf_new = tcache_create_toast_buffer(tbuf_old->tbuf_length);
	memcpy(tbuf_new->data,
		   tbuf_old->data,
		   tbuf_old->usage - offsetof(tcache_toastbuf, data[0]));
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
		if (dlist_is_emptry(&tc_head->free_list))
		{
			dlist_node *block;
			int			i;

			block = pgstrom_shmem_alloc(SHMEM_BLOCKSZ - sizeof(cl_uint));
			if (!block)
				elog(ERROR, "out of shared memory");
			dlist_push_tail(&tc_head->block_list, block);

			tc_node = (tcache_node *)(block + 1);
			for (i=0; i < TCACHE_NODE_PER_BLOCK_BARE; i++)
				dlist_push_tail(&tc_head->free_list, &tc_node[i].chain);
		}
		dnode = dlist_pop_head_node(&tc_head->free_list);
		tc_node = dlist_container(tcache_node, chain, dnode);
		memset(&tc_node, 0, sizeof(tcache_node));

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
tcache_free_node_recursive(tcache_head *tc_head, tcache_node *tc_node)
{
	/* NOTE: caller must be responsible to hold tc_head->lock */

	if (tc_node->right)
		tcache_free_node_recursive(tc_head, tc_node->right);
	if (tc_node->left)
		tcache_free_node_recursive(tc_head, tc_node->left);
	tcache_free_node_nolock(tc_head, tc_node);
}

static void
tcache_free_node(tcache_head *tc_head, tcache_node *tc_node)


{
	SpinLockAcquire(&tc_head->lock);
	tcache_free_node_nolock(tc_head, tc_node);
	SpinLockRelease(&tc_head->lock);
}

/*
 * tcache_find_least_colstore
 *
 * It tried to find least column-store that can contain any records larger
 * than the supplied 'ctid'. Usually, this routine is aplied to forward scan.
 */
static tcache_column_store *
internal_find_least_column_store(tcache_node *tc_node, ItemPointer ctid)
{
	tcache_column_store *tcs;

	SpinLockAcquire(&tc_node->lock);
	if (tc_node->nrows == 0)
	{
		SpinLockRelease(&tc_node->lock);
		return NULL;
	}

	if (ItemPointerCompare(ctid, &tc_node->tcs.ip_max) >= 0)
	{
		/*
		 * if citd <= ip_max of this node, this node is obviously not a one
		 * to be fetched on the next. So, we try to walks on the next right
		 * node or stop walking if no more larger ones.
		 */
		SpinLockRelease(&tc_node->lock);

		if (!tc_node->right)
			return NULL;
		return internal_find_least_column_store(tc_node->right, ctid);
	}
	/*
	 * no candidate that can contain less ctid than current node
	 */
	if (!tc_node->left)
	{
		tcs = tcache_get_column_store(tc_node->tcs);
		SpinLockRelease(&tc_node->lock);
		return tcs;
	}

	/*
	 * try to walks on the left node, and get thid node if not suitable
	 */
	SpinLockRelease(&tc_node->lock);
	tcs = internal_find_least_column_store(tc_node->left, ctid);
	if (!tcs)
	{
		SpinLockAcquire(&tc_node->lock);
		tcs = tcache_get_column_store(tc_node->tcs);
		SpinLockRelease(&tc_node->lock);
	}
	return tcs;
}
static tcache_column_store *
tcache_find_least_column_store(tcache_head *tc_head, ItemPointer ctid)
{
	if (!tc_head->tcs_root)
		return NULL;
	return internal_find_least_column_store(tc_head->tcs_root, ctid);
}

/*
 * tcache_find_greatest_colstore
 *
 * It tried to find greatest column-store that can contain any records less
 * than the supplied 'ctid'. Usually, this routine is aplied to backward scan.
 */
static tcache_column_store *
internal_find_greatest_column_store(tcache_node *tc_node, ItemPointer ctid)
{
	tcache_column_store *tcs;

	SpinLockAcquire(&tc_node->lock);
	if (tc_node->nrows == 0)
	{
		SpinLockRelease(&tc_node->lock);
		return NULL;
	}

	if (ItemPointerCompare(ctid, &tc_node->tcs.ip_min) <= 0)
	{
		/*
		 * if ctid <= min ctid of this node, this node is obviously not
		 * a one to be fetched on the next. So, we try to walk on the
		 * next left node or stop walking down if no more smaller ones.
		 */
		SpinLockRelease(&tc_node->lock);

		if (!tc_node->left)
			return NULL;
		return internal_find_greatest_column_store(tc_node->left, ctid);
	}
	/*
	 * no candidate that can contain less ctid than current node
	 */
	if (!tc_node->right)
	{
		tcs = tcache_get_column_store(tc_node->tcs);
		SpinLockRelease(&tc_node->lock);
		return tcs;
	}

	/*
	 * try to walks on the left node, and get thid node if not suitable
	 */
	SpinLockRelease(&tc_node->lock);
	tcs = internal_find_greatest_column_store(tc_node->right, ctid);
	if (!tcs)
	{
		SpinLockAcquire(&tc_node->lock);
		tcs = tcache_get_column_store(tc_node->tcs);
		SpinLockRelease(&tc_node->lock);
	}
	return tcs;
}

static tcache_column_store *
tcache_find_greatest_column_store(tcache_head *tc_head, ItemPointer ctid)
{
	if (!tc_head->tcs_root)
		return NULL;
	return internal_find_greatest_column_store(tc_head->tcs_root, ctid);
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
	int		li = left;
	int		ri = right;
	ItemPointer pivot = &tcs->ctids[(li + ri) / 2];

	if (left >= right)
		return;

	while (li < ri)
	{
		while (ItemPointerCompare(&tcs->ctids[li], pivot) < 0)
			li++;
		while (ItemPointerCompare(&tcs->ctids[ri], pivot) > 0)
			ri--;
		/*
		 * Swap values
		 */
		if (li < ri)
		{
			Form_pg_attribute attr;
			int		i, j;

			memswap(&tcs->ctids[li], &tcs->ctids[ri],
					sizeof(ItemPointerData));
			memswap(&tcs->theads[li], &tcs->theads[ri],
					sizeof(HeapTupleHeaderData));

			for (i=0; i < tc_head->col_natts; i++)
			{
				j = tc_head->i_cached[i];
				attr = tc_head->tupdesc->attrs[j];

				/* isnull flags */
				if (!attr->attnotnull)
				{
					Assert(tcs->cdata[i].isnull != NULL);
					bitswap(tcs->cdata[i].isnull, li, ri);
				}
				memswap(tcs->cdata[i].values + attr->attlen * li,
						tcs->cdata[i].values + attr->attlen * ri,
						attr->attlen);
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
 * tcache_compaction_tcnode
 *
 *
 * NOTE: caller must hold exclusive lwlock on tc_head.
 */
static void
tcache_compaction_tcnode(tcache_head *tc_head, tcache_node *tc_node)
{
	tcache_column_store	*tcs_new;
	tcache_column_store *tcs_old = tc_node->tcs;

	Assert(TCacheHeadLockedByMe(tc_head, true));

	tcs_new = tcache_create_column_store(tc_head);
	PG_TRY();
	{
		Size	required;
		int		i, j, k;

		/* assign a toast buffer first */
		for (i=0; i < tcs_old->ncols; i++)
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
		ItemPointerSet(&tcs_new->ip_max, &tcs_old->ip_min);
		ItemPointerSet(&tcs_new->ip_min, &tcs_old->ip_max);

		/* OK, let's make it compacted */
		for (i=0, j=0; i < tcs_old->nrows; i++)
		{
			TransactionId		xmax;

			/*
			 * Once a record on the column-store is vacuumed, it will have
			 * FrozenTransactionId less than FirstNormalTransactionId.
			 * Nobody will never see the record, so we can skip it.
			 */
			xmax = HeapTupleHeaderGetRawXmax(&tcs_old->theads[i]);
			if (xmax < FirstNormalTransactionId)
				continue;

			/*
			 * Data copy
			 */
			memcopy(&tcs_new->ctids[j],
					&tcs_old->ctids[j],
					sizeof(ItemPointerData));
			memcopy(&tcs_new->theads[j],
					&tcs_old->theads[j],
					sizeof(HeapTupleHeaderData));

			if (ItemPointerCompare(&tcs_new->ctids[j], &tcs_new->ip_max) > 0)
				ItemPointerCopy(&tcs_new->ctids[j], &tcs_new->ip_max);
			if (ItemPointerCompare(&tcs_new->ctids[j], &tcs_new->ip_min) < 0)
				ItemPointerCopy(&tcs_new->ctids[j], &tcs_new->ip_min);

			for (k=0; k < tcs_old->ncols; k++)
			{
				int		l = tc_head->i_cached[k];
				int		attlen = tc_head->tupdesc->attrs[l]->attlen;

				/* nullmap */
				if (tcs_old->cdata[k].isnull)
					bitcopy(tcs_new->cdata[k].isnull, j,
							tcs_old->cdata[k].isnull, i);
				/* values */
				if (attlen > 0)
					memcopy(tcs_new->cdata[k].values + attlen * j,
							tcs_old->cdata[k].values + attlen * i,
							attlen);
				else
				{
					tcache_toastbuf	*tbuf_old = tcs_old->cdata[k].toast;
					tcache_toastbuf *tbuf_new = tcs_new->cdata[k].toast;
					char			*vptr;

					vptr = (char *)tbuf_old +
						((cl_uint *)tcs_old->cdata[k].values)[i];
					memcopy((char *)tbuf_new + tbuf_new->tbuf_usage,
							vptr,
							VARSIZE(vptr));
					((cl_uint *)tcs_new->cdata[k].values)[j]
						= tbuf_new->tbuf_usage;
					tbuf_new->tbuf_usage += MAXALIGN(VARSIZE(vptr));
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
 * tcache_split_tcnode
 *
 * it creates a new tcache_node and move half of the records;
 * includes varlena datum.
 *
 * NOTE: caller must hold exclusive lwlock on tc_head.
 */
static void
tcache_split_tcnode(tcache_head *tc_head, tcache_node *tc_node_old)
{
	tcache_node *tc_node_new;
    tcache_column_store *tcs_new;
    tcache_column_store *tcs_old = tc_node_old->tcs;

	Assert(TCacheHeadLockedByMe(tc_head, true));

	tc_node_new = tcache_alloc_tcnode(tc_head);
	tcs_new = tc_node_new->tcs;
	PG_TRY();
	{
		Form_pg_attribute attr;
		/* 'nremain' must be multiplexer because of nullmap alignment. */
		int		nremain = TYPEALIGN_DOWN(BITS_PER_BYTE, tcs_old->nrows / 2);
		int		nmoved = tcs_old->nrows - nremain;
		int		i, j, k;
		

		/* assign toast buffers first */
		for (i=0; i < tcs_old->ncols; i++)
		{
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
			tcache_sort_tcnode(tc_head, tc_node, true);

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
		for (i=0; i < tcs_old->ncols; i++)
		{
			j = tc_head->i_cached[i];
			attr = tc_head->tupdesc->attrs[j];

			/* nullmap */
			if (!attr->attnotnull)
			{
				Assert(nremain % BITS_PER_BYTE == 0);
				memcpy(tcs_new->cdata[i].isnull,
					   tcs_old->cdata[i].isnull + nremain / BITS_PER_BYTE,
					   (nmoved + BITS_PER_BYTE - 1) / BITS_PER_BYTE);
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
				for (k=0; k < nmoved; k++)
				{
					tcache_toastbuf *tbuf_old = tcs_old->cdata[i].toast;
					tcache_toastbuf	*tbuf_new = tcs_new->cdata[i].toast;
					char			*vptr;

					vptr = (char *)tbuf_old +
						((cl_uint *)tcs_old->cdata[i].values)[k + nremain];
					memcpy((char *)tbuf_new + tbuf_new->tbuf_usage,
						   vptr,
						   VARSIZE(vptr));
					((cl_uint *)tcs_new->cdata[i].values)[k]
						= tbuf_new->tbuf_usage;
					tbuf_new->tbuf_usage += MAXALIGN(VARSIZE(vptr));
				}
			}
		}
		tcs_new->nrows = nmoved;
		tcs_new->njunks = 0;
		tcs_new->is_sorted = true;
		tcs_new->ip_min = tcs_new->ctids[0];
		tcs_new->ip_max = tcs_new->ctids[nmoved - 1];

		/*
		 * OK, tc_node_new is ready to chain as larger half of
		 * this column-store.
		 */
		tc_node_new->right = tc_node_old->right;
		tc_node_new->r_depth = tc_node_old->r_depth;
		tc_node_old->right = tc_node_new;
		tc_node_old->r_depth = tc_node_new->r_depth + 1;

		tcs_old->nrows = nremain;
		tcs_old->ip_max = tcs_old->ctids[nremain - 1];
	}
	PG_CATCH();
	{
		tcache_free_node_nolock(tc_node_new);
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
}

static void
tcache_rebalance_tree(tcache_head *tc_head, tcache_node *tc_node,
					  tcache_node **p_tc_node)
{
	Assert(TCacheHeadLockedByMe(tc_head, true));

	if (tc_node->l_depth + 1 < tc_node->r_depth)
	{
		tcache_node	*tc_lnode;


		/* anticlockwise rotation */

	}
	else if (tc_node->l_depth > tc_node->r_depth + 1)
	{
		/* clockwise rotation */

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
do_insert_record(tcache_head *tc_head, tcache_node *tc_node, HeapTuple tuple)
{
	tcache_column_store *tcs = tc_node->tcs;
	TupleDesc	tupdesc = tc_head->tupdesc;
	Datum	   *values = alloca(sizeof(Datum) * tupdesc->natts);
	bool	   *isnull = alloca(sizeof(bool) * tupdesc->natts);
	int			i, j, k;

	Assert(TCacheHeadLockedByMe(tc_head, true));
	Assert(tcs->nrows >= 0 && tcs->nrows < NUM_ROWS_PER_COLSTORE);

	heap_deform_tuple(tuple, tupdesc, values, isnull);

	/* copy system columns */
	tcs->ctids[tcs->nrows] = tuple->t_self;
	memcopy(&tcs->theads[tcs->nrows], tuple->t_data,
			sizeof(HeapTupleHeaderData));

	for (i=0; i < tcs->ncols; i++)
	{
		Form_pg_attribute	attr;

		j = c_head->i_cached[i];
		Assert(j >= 0 && j < tupdesc->natts);
		attr = tupdesc->attrs[j];

		if (attr->attlen > 0)
		{
			/* fixed-length variable is simple to put */
			memcopy(tcs->cdata[i].values + attr->attlen * tcs->nrows,
					&values[j],
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
			Size		vsize = VARSIZE_ANY(values[j]);

			if (tbuf->tbuf_usage + MAXALIGN(vsize) < tbuf->tbuf_length)
			{
				tcache_toastbuf	*tbuf_new;

				/*
				 * Needs to expand toast-buffer if no more room exist
				 * to store new varlenas. Usually, twice amount of
				 * toast buffer is best choice for buddy allocator.
				 */
				tbuf_new = tcache_create_toast_buffer(2 * tbuf->tbuf_length);
				memcpy(tbuf_new->data,
					   tbuf->data,
					   tbuf->tbuf_usage - offsetof(tcache_toastbuf, data[0]));
				tbuf_new->tbuf_usage = tbuf->tbuf_usage;
				tbuf_new->tbuf_junk = tbuf->tbuf_junk;

				/* replace older buffer by new (larger) one */
				tcache_put_toast_buffer(tbuf);
				tcs->cdata[i].toast = tbuf = tbuf_new;
			}
			Assert(tbuf->tbuf_usage + MAXALIGN(vsize) < tbuf->tbuf_length);

			((cl_uint *)tcs->cdata[i].values)[tcs->nrows] = tbuf->tbuf_usage;
			memcpy((char *)tbuf + tbuf->tbuf_usage,
				   DatumGetPointer(values[j]),
				   vsize);
			tbuf->tbuf_usage += MAXALIGN(vsize);
		}
	}
	/* update ip_max and ip_min, if needed */
	if (tcs->nrows == 0)
	{
		tcs->ip_min = tuple->t_self;
		tcs->ip_max = tuple->t_self;
		tcs->is_sorted = true;	/* it is obviously sorted! */
	}
	else if (tcs->is_sorted)
	{
		if (ItemPointerCompare(&tuple->t_self, &tcs->ip_max) > 0)
		{
			tcs->ip_max = tuple->t_self;
		}
		else
		{
			/*
			 * Oh... the new record placed on 'nrows' does not have
			 * largest item-pointer. It breaks the assumption; this
			 * column-store is sorted by item-pointer.
			 * It may take sorting again in the future.
			 */
			tcs->is_sorted = false;
			if (ItemPointerCompare(&tuple->t_self, &tcs->ip_min) < 0)
				tcs->ip_min = tuple->t_self;
		}
	}
	else
	{
		if (ItemPointerCompare(&tuple->t_self, &tcs->ip_min) < 0)
			tcs->ip_min = tuple->t_self;
		if (ItemPointerCompare(&tuple->t_self, &tcs->ip_max) > 0)
			tcs->ip_max = tuple->t_self;
	}
	/* all valid, so increment nrows */
	pg_memory_barrier();
	tcs->nrows++;
}

static void
tcache_insert_tuple(tcache_head *tc_head,
					tcache_node *tc_node,
					tcache_node **p_this_node,
					HeapTuple tuple)
{
	bool	needs_rebalance = false;

	/* NOTE: we assume exclusive lwlock is acquired */
	Assert(TCacheHeadLockedByMe(tc_head, true));

	tcache_column_store *tcs = tc_node->tcs;

	if (tcs->nrows == 0)
	{
		do_insert_record(tc_head, tc_node, tuple);
		return;
	}

retry:
	if (ItemPointerCompare(ctid, &tcs->ip_min) < 0)
	{
		if (!tc_node->left && tcs->nrows < NUM_ROWS_PER_COLSTORE)
			do_insert_record(tc_head, tc_node, tuple);
		else
		{
			if (!tc_node->left)
				tc_node->left = tcache_alloc_tcnode(tc_head);

			tcache_insert_record(tc_head, tc_node->left,
								 ctid, thead, values, isnull);
			tc_node->l_depth = TCACHE_DEPTH(tc_node->left);
			needs_rebalance = true;
		}
	}
	else if (ItemPointerCompare(ctid, &tcs->max_ctid) > 0)
	{
		if (!tc_node->right && tcs->nrows < NUM_ROWS_PER_COLSTORE)
			do_insert_record(tc_head, tc_node, tuple);
		else
		{
			if (!tcs->right)
				tcs->right = tcache_alloc_tcnode(tc_head);

			tcache_insert_record(tc_head, tc_node->right,
								 ctid, thead, values, isnull);
			tc_node->r_depth = TCACHE_DEPTH(tc_node->right);
			needs_rebalance = true;
		}
	}
	else
	{
		if (tcs->nrows < NUM_ROWS_PER_COLSTORE)
			do_insert_record(tc_head, tc_node, tuple);
		else
		{
			/* split this chunk into two */
			tcache_split_tcnode(tc_head, tc_node);
			needs_rebalance = true;
			goto retry;
		}
	}

	/*
	 * Rebalance the t-tree cache, if needed
	 */
	if (needs_rebalance)
		tcache_rebalance_tree(tc_head, tc_node, p_this_node);
}

static bool
tcache_build_main(tcache_head *tc_head, HeapScanDesc scan)
{
	Relation	relation = scan->rs_rd;
	TupleDesc	tupdesc = RelationGetDescr(relation);
	Datum	   *values = palloc(sizeof(Datum) * tupdesc->natts);
	bool	   *isnull = palloc(sizeof(bool) * tupdesc->natts);
	HeapTuple	tuple;

	while (HeapTupleIsValid(tuple = heap_getnext(tc_head->heapscan,
												 ForwardScanDirection)))
	{
		heap_deform_tuple(tuple, tupdesc, values, isnull);

		/* insert a tuple into a column-store */
	}
	pfree(values);
	pfree(isnull);

	return true;
}






tcache_scandesc *
tcache_begin_scan(Relation rel, Bitmapset *required)
{
	tcache_scandesc	   *tc_scan;
	tcache_head		   *tc_head;
	bool		has_wrlock = false;

	tc_scan = palloc0(sizeof(tcache_scandesc));
	tc_scan->rel = rel;
	tc_head = tcache_get(RelationGetRelid(rel), required, true);
	if (!tc_head)
		elog(ERROR, "out of shared memory");
	pgstrom_track_object(&tc_head->stag);
	tc_scan->tc_head = tc_head;

	LWLockAcquire(&tc_head->lwlock, LW_SHARED);
retry:
	SpinLockAcquire(&tc_head->lock);
	if (tc_head->state == TC_STATE_NOT_BUILT)
	{
		if (!has_wrlock)
		{
			SpinLockRelease(&tc_head->lock);
			LWLockRelease(&tc_head->lock);
			LWLockAcquire(&tc_head->lwlock, LW_EXCLUSIVE);
			has_wrlock = true;
			goto retry;
		}
		tc_head->state = TC_STATE_NOW_BUILD;
		SpinLockRelease(&tc_head->lock);
		tc_scan->heapscan = heap_beginscan(rel, SnapshotAny, 0, NULL);

	}
	else if (has_wrlock)
	{
		SpinLockRelease(&tc_head->lock);
		LWLockRelease(&tc_head->lock);
		LWLockAcquire(&tc_head->lwlock, LW_SHARED);
		has_wrlock = false;
		goto retry;
	}
	else
	{
		Assert(tc_head->state == TC_STATE_READY);
		SpinLockRelease(&tc_head->lock);
	}
	return tc_scan;
}

StromTag *
tcache_next_chunk(tcache_scandesc *tc_scan, ScanDirection direction)
{
	ItemPointerData	ctid;

	Assert(direction != NoMovementScanDirection);

	if (tc_scan->scan)
	{
		/*
		 * read heap pages and construct a tcache_column_store object,
		 * then insert this store into cache structure.
		 */

		/*
		 * get this tcache_column_store object, and return it.
		 */
		return NULL;
	}

	LWLockAcquire(&tc_scan->tc_head->lock, LW_SHARED);
	/*
	 * Find the next next row/column store that may contain next
	 * item-pointer
	 */
	if (direction > 0)
	{
		if (!tc_scan->curr_trs)
		{
			if (!tc_scan->curr_tcs)
				ItemPointerSet(&ctid, 0, FirstOffsetNumber);
			else
				ItemPointerCopy(tc_scan->curr_tcs->min_ctid, &ctid);
			/*
			 * find a next columnar-store. If not, fetch first row-store
			 * and set curr_tcs as NULL.
			 */
		}
		if (tc_scan->curr_trs)
		{
			/* fetch next row-store */
		}
		/* if both of column- and row-store is null, it is end of scan */
	}
	else
	{
		if (!tc_scan->curr_trs)
		{
			if (!tc_scan->curr_tcs)
				ItemPointerSet(&ctid, MaxBlockNumber, MaxOffsetNumber);
			else
				ItemPointerCopy(tc_scan->curr_tcs->max_ctid, &ctid);
			/*
			 * find a greater column-store that can have 
			 */
		}
		if (tc_scan->curr_trs)
		{

		}
		/* if both of column- and row-store is null, it is end of scan */
	}
	LWLockRelease(&tc_scan->tc_head->lock);




}

void
tcache_end_scan(tcache_scandesc *tc_scan)
{
	/*
	 * If scan is already reached end of the relation, tc_scan->scan shall
	 * be already closed. If not, it implies scan is aborted in the middle.
	 */
	SpinLockAcquire(&tc_common->lock);
	if (tc_scan->scan)
	{
		
		SpinLockRelease(&tc_common->lock);
		/*
		 * TODO: release all the tcache_(row|column)_store being chained
		 * in this cache
		 */
		heap_endscan(tc_scan->scan);
	}
	else if (tc_scan->state == TC_STATE_BUILD_NOW)
	{
		/* OK, cache was successfully built */
		tc_scan->state = TC_STATE_READY;
		SpinLockRelease(&tc_common->lock);
	}
	else
	{
		Assert(tc_scan->state == TC_STATE_READY);
		SpinLockRelease(&tc_common->lock);
	}
	Assert(*tc_scan->curr.stag == StromTag_TCacheRowStore ||
		   *tc_scan->curr.stag == StromTag_TCacheColumnStore);
	if (*tc_scan->curr.stag == StromTag_TCacheRowStore)
		tcache_put_row_store(tc_scan->curr.trs);
	else
		tcache_put_column_store(tc_scan->curr.tcs);

	tcache_put_tchead(tc_scan->tc_head);
	pfree(tc_scan);
}

void
tcache_rescan(tcache_scandesc *tc_scan)
{
	int		state;

	tc_scan->curr.stag = NULL;
	tc_scan->curr_index = 0;
	SpinLockAcquire(&tc_common->lock);
	state = tc_scan->tc_head->state;
	SpinLockRelease(&tc_common->lock);

	/* Also, needs to drop half constructed cache nodes */
	if (tc_scan->scan)
		heap_rescan(tc_scan->scan);
	else if (state == TC_STATE_BUILD_NOW)
		tc_scan->scan = heap_beginscan(tc_scan->rel, SnapshotAny, 0, NULL);
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
	int				i, j, k;

	/* calculation of the length */
	tup = SearchSysCache1(RELOID, ObjectIdGetDatum(reloid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for relation %u", reloid);
	relform = (Form_pg_class) GETSTRUCT(tup);

	length = (MAXALIGN(offsetof(tcache_head, data[0])) +
			  MAXALIGN(sizeof(*tupdesc)) +
			  MAXALIGN(sizeof(Form_pg_attribute) * relform->relnatts) +
			  MAXALIGN(sizeof(FormData_pg_attribute)) * relform->relnatts +
			  MAXALIGN(sizeof(AttrNumber) * relform->relnatts));

	/* allocation of a shared memory block (larger than length) */
	tc_head = pgstrom_shmem_alloc_alap(length, &allocated);
	if (!tc_head)
		elog(ERROR, "out of shared memory");

	PG_TRY();
	{
		Size	offset = MAXALIGN(offsetof(tcache_head, data[0]));

		memset(tc_head, 0, sizeof(tcache_head));

		tc_head->stag = StromTag_TCacheHead;
		tc_head->refcnt = 1;

		LWLockInitialize(&tc_head->lwlock, 0);

		SpinLockInit(&tc_head->lock);
		tc_head->state = TC_STATE_NOT_BUILD;
		dlist_init(&tc_head->free_list);
		dlist_init(&tc_head->block_list);
		dlist_init(&tc_head->pending_list);
		dlist_init(&tc_head->trs_list);
		tc_head->datoid = MyDatabaseId;
		tc_head->reloid = reloid;

		tempset = bms_copy(required);
		if (tcache_old)
		{
			for (i=0; i < tcache_old->nattrs; i++)
			{
				j = (tcache_old->tupdesc->attrs[i].attnum
					 - FirstLowInvalidHeapAttributeNumber);
				tempset = bms_add_member(tempset, j);
			}
		}
		tc_head->ncols = bms_num_members(tempset);
		tc_head->i_cached = (AttrNumber *)((char *)tc_head + offset);
		offset += MAXALIGN(sizeof(AttrNumber) * relform->relnatts);

		tupdesc = (TupleDesc)((char *)tc_head + offset);
		memset(tupdesc, 0, sizeof(*tupdesc));
		offset += MAXALIGN(*tupdesc);

		tupdesc->natts = relform->relnatts;
		tupdesc->attrs = (Form_pg_attribute *)((char *)tc_head + offset);
		offset += MAXALIGN(sizeof(Form_pg_attribute) * relform->relnatts);
		tupdesc->tdtypeid = relform->reltype;
		tupdesc->tdtypmod = -1;
		tupdesc->tdhasoid = relform->relhasoids;
		tupdesc->tdrefcount = -1;

		for (i=0, j=0; i < tupdesc->natts; i++)
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

			k = ((Form_pg_attribute) GETSTRUCT(atttup))->attnum
				- FirstLowInvalidHeapAttributeNumber;
			if (bms_is_member(k, tempset))
				tc_head->i_cached[j++] = i;

			ReleaseSysCache(atttup);
		}
		Assert(offset <= length);
		Assert(tc_head->ncols == j);
		tc_head->tupdesc = tupdesc;
		bms_free(tempset);

		/* remaining area shall be used to tcache_node */
		while (offset + sizeof(tcache_node) < allocated)
		{
			tcache_node *tc_node
				= (tcache_node *)((char *)tc_head + offset);

			dlist_push_tail(&tc_head->free_list, &tc_node->chain);
			offset += MAXALIGN(tcache_node);
		}
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
	/*
	 * TODO: needs to check tc_head->state.
	 * If TC_STATE_NOW_BUILD, we have to release it and revert the status
	 *
	 * Also, it has to be done prior to release locking.
	 */


	if (--tc_head->refcnt == 0)
	{
		dlist_node	   *dnode;

		Assert(!tc_head->chain.prev && !tc_head->chain.next);
		Assert(!tc_head->lru_chain.prev && !tc_head->lru_chain.next);

		/*
		 * release blocks for column-store nodes
		 */
		dlist_foreach_modify(iter, &tc_head->block_list)
		{
#ifdef USE_ASSERT_CHECKING
			int		i;
			tcache_column_store *tcs
				= (tcache_column_store *)(iter.cur + 1);
			/* all the blocks should be already released */
			for (i=0; i < TCACHE_COLSTORE_PER_BLOCK(tc_head->nattrs); i++)
				Assert(tcs[i].chain.prev && tcs[i].chain.next);
#endif
			pgstrom_shmem_free(iter.cur);
		}
		/* also, all the row-store should be released */
		Assert(dlist_is_empty(&tcs->trs_list));
		tc_head->state = TC_STATE_FREE;
		dlist_push_tail(&tc_common->free_list, &tc_head->chain);
	}
}

void
tcache_put_tchead(tcache_head *tc_head)
{
	SpinLockAcquire(&tc_common->lock);
	tcache_put_tchead_nolock(tc_head);
	SpinLockRelease(&tc_common->lock);
}

tcache_head *
tcache_get_tchead(Oid reloid, Bitmapset *required,
				  bool create_on_demand)
{
	dlist_iter		iter;
	tcache_head	   *tc_head = NULL;
	tcache_head	   *tc_old = NULL;
	pg_crc32		crc;
	int				hindex;

	/* calculate hash index */
	INIT_CRC32(crc);
	COMP_CRC32(crc, &MyDatabaseId, sizeof(Oid));
	COMP_CRC32(crc, &reloid, sizeof(Oid));
	FIN_CRC32(crc);
	hindex = crc % TCACHE_HASH_SIZE;

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
				Bitmapset  *tempset = bms_copy(required);
				int			i, j = 0;
				int			k, l;

				while ((i = bms_first_member(tempset)) >= 0 &&
					   j < temp->tupdesc->natts)
				{
					i += FirstLowInvalidHeapAttributeNumber;

					/* all the system attributes are cached in the default */
					if (i < 0)
						continue;

					/*
					 * whole row reference is equivalent to references to
					 * all the valid (none dropped) columns.
					 * also, row reference shall apear prior to all the
					 * regular columns because of its attribute number
					 */
					if (i == InvalidAttrNumber)
					{
						for (k=0; k < temp->tupdesc->natts; k++)
						{
							if (temp->tupdesc->attrs[k]->attisdropped)
								continue;

							l = k - FirstLowInvalidHeapAttributeNumber;
							tempset = bms_add_member(l, tempset);
						}
						continue;
					}

					/*
					 * Is this regular column cached?
					 */
					while (j < temp->ncols)
					{
						k = temp->i_cached[j];
						if (temp->tupdesc->attrs[k]->attnum != i)
							break;
						j++;
					}
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
				break;
			}
		}

		if (!tc_head && create_on_demand)
		{
			tc_head = tcache_create_tchead_nolock(reloid, required, tc_old);
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
					pgstrom_put_tcache_nolock(tc_old);
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

	return tc_head;
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
	/* lock if we don't have lwlock because heap-scan may kick vacuuming */




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

}





/*
 * pgstrom_tcache_synchronizer
 *
 *
 *
 *
 */
Datum
pgstrom_tcache_synchronizer(PG_FUNCTION_ARGS)
{
	TriggerData    *trigdata;
	Relation        rel;
	HeapTuple       tuple;
	HeapTuple       newtup;

	if (!CALLED_AS_TRIGGER(fcinfo))
		elog(ERROR, "%s: not fired by trigger manager", __FUNCTION__);

	trigdata = (TriggerData *) fcinfo->context;
	rel = trigdata->tg_relation;
	tuple = trigdata->tg_trigtuple;
	newtup = trigdata->tg_newtuple;




}
PG_FUNCTION_INFO_V1(pgstrom_tcache_synchronizer);

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
				strcmp(probin, "$libdir/cache_scan") == 0)
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












static void
pgstrom_columnizer_main(Datum index)
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

	/* pick a pending tcache_head */
	/* lock this tc_head exclusively */
	/* pick a row-store from pending list */
	/* insert record */

	/* pick a column-store from pending list */
	/* do compaction if junk records is larger than threshold */

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
		worker.bgw_main = pgstrom_columnizer_main;
		worker.bgw_main_arg = i;
		RegisterBackgroundWorker(&worker);
	}

	/* aquires shared memory region */
	length = offsetof(tcache_common, columnizers[num_columnizers]);
	RequestAddinShmemSpace(MAXALIGN(length));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_tcache;
}
