/*
 * select_into.c
 *
 * Routines related to SELECT INTO Direct mode
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

void
pgstrom_init_select_into(void)
{
}

INLINE_FUNCTION(void)
__initPageHeaderData(PageHeaderData *hpage)
{
	/*
	 * NOTE: PG-Strom's should write heap blocks only to a newly created empty table
	 * in SELECT INTO Direct mode.
	 * This table is not visible to other transactions until the current transaction is
	 * committed, and the table is deleted if any error occurs before the transaction is
	 * committed.
	 * So, when writing to an empty table with SELECT INTO, you can safely consider
	 * the tuples to be ALL_VISIBLE and XMIN_FROZEN.
	 */
    hpage->pd_checksum = 0;
    hpage->pd_flags = PD_ALL_VISIBLE;   /* see the comments above */
    hpage->pd_lower = offsetof(PageHeaderData, pd_linp);
    hpage->pd_upper = BLCKSZ;
    hpage->pd_special = BLCKSZ;
    hpage->pd_pagesize_version = (BLCKSZ | PG_PAGE_LAYOUT_VERSION);
    hpage->pd_prune_xid = 0;
}

bool
__initSelectIntoState(selectIntoState *si_state, const kern_session_info *session)
{
	const char *base_name = SESSION_SELECT_INTO_PATHNAME(session);
	const kern_data_store *kds_head = SESSION_SELECT_INTO_KDS_HEAD(session);

	if (!base_name ||
		!kds_head ||
		(session->xpu_task_flags & DEVTASK__SELECT_INTO_DIRECT) == 0)
		return true;		/* nothing to do */
	assert(kds_head->block_offset >= KDS_HEAD_LENGTH(kds_head) &&
		   kds_head->block_offset == TYPEALIGN(BLCKSZ, kds_head->block_offset) &&
		   kds_head->block_offset + BLCKSZ * RELSEG_SIZE == kds_head->length);
	pthreadMutexInit(&si_state->kds_heap_lock);
	si_state->kds_heap = malloc(kds_head->length);
	if (!si_state->kds_heap)
		goto error;
	memcpy(si_state->kds_heap, kds_head, KDS_HEAD_LENGTH(kds_head));
	si_state->kds_heap_segno = 0;

	pthreadMutexInit(&si_state->dpage_lock);
	si_state->dpage = malloc(BLCKSZ);
	__initPageHeaderData(si_state->dpage);

	si_state->base_name = strdup(base_name);
	if (!si_state->base_name)
		goto error;
	return true;
error:
	__cleanupSelectIntoState(si_state);
	return false;
}

void
__cleanupSelectIntoState(selectIntoState *si_state)
{
	if (si_state->kds_heap)
		free(si_state->kds_heap);
	if (si_state->dpage)
        free(si_state->dpage);
    if (si_state->base_name)
        free((void *)si_state->base_name);
}

/*
 * __selectIntoWriteOutOneBlock
 */
static bool
__selectIntoWriteOutOneBlock(gpuClient *gclient,
							 selectIntoState *si_state,
							 PageHeaderData *hpage)
{
	kern_data_store *kds_heap;
	PageHeaderData *dpage;

	pthreadMutexLock(&si_state->kds_heap_lock);
	kds_heap = si_state->kds_heap;
	assert(kds_heap->nitems < RELSEG_SIZE);
	dpage = KDS_BLOCK_PGPAGE(kds_heap, kds_heap->nitems++);
	/* replace and write out the buffer, if last one */
	if (kds_heap->nitems < RELSEG_SIZE)
		__atomic_add_int64((int64_t *)&kds_heap->usage, 2);
	else
	{
		kern_data_store *kds_new = malloc(kds_heap->length);

		if (!kds_new)
		{
			pthreadMutexUnlock(&si_state->kds_heap_lock);
			gpuClientELog(gclient, "out of memory");
			return false;
		}
		memcpy(kds_new, kds_heap, KDS_HEAD_LENGTH(kds_heap));
		kds_new->nitems = 0;
		kds_new->usage = 0;
		si_state->kds_heap = kds_new;
		/* mark writable */
		__atomic_add_int64((int64_t *)&kds_heap->usage, 3);
	}
	pthreadMutexUnlock(&si_state->kds_heap_lock);
	memcpy(dpage, hpage, BLCKSZ);
	if (__atomic_add_int64((int64_t *)&kds_heap->usage, -2) == 3)
	{
		uint32_t	segment_no = __atomic_add_uint32(&si_state->kds_heap_segno, 1);
		char	   *fname = alloca(strlen(si_state->base_name) + 20);
		int			fdesc;
		size_t		remained = Min(kds_heap->nitems, RELSEG_SIZE) * BLCKSZ;
		const char *curr_pos = (const char *)kds_heap + kds_heap->block_offset;
		ssize_t		nbytes;

		if (segment_no == 0)
			strcpy(fname, si_state->base_name);
		else
			sprintf(fname, "%s.%u", si_state->base_name, segment_no);
		fdesc = open(fname, O_WRONLY | O_CREAT | O_TRUNC, 0600);
		if (fdesc < 0)
		{
			gpuClientELog(gclient, "failed on open('%s'): %m", fname);
			return false;
		}

		while (remained > 0)
		{
			nbytes = write(fdesc, curr_pos, remained);
			if (nbytes <= 0)
			{
				if (errno == EINTR)
					continue;
				gpuClientELog(gclient, "failed on write('%s', %lu): %m",
							  fname, remained);
				close(fdesc);
				return false;
			}
			assert(nbytes <= remained);
			curr_pos += nbytes;
			remained -= nbytes;
		}
		free(kds_heap);
	}
	return true;
}

/*
 * selectIntoWriteOutHeapNormal
 */
bool
selectIntoWriteOutHeapNormal(gpuClient *gclient,
							 selectIntoState *si_state,
							 kern_data_store *kds_dst)	/* managed memory */
{
	PageHeaderData *hpage = (PageHeaderData *)alloca(BLCKSZ);
	uint32_t	lp_index = 0;
	uint32_t	lp_offset = BLCKSZ;

	assert(kds_dst->format == KDS_FORMAT_ROW ||
		   kds_dst->format == KDS_FORMAT_HASH);
	__initPageHeaderData(hpage);
	for (uint32_t index=0; index < kds_dst->nitems; index++)
	{
		kern_tupitem   *titem = KDS_GET_TUPITEM(kds_dst, index);
		ItemIdData		lp_item;
		int32_t			tupsz;
		HeapTupleHeaderData *htup;

		if (!titem)
			continue;
		tupsz = MINIMAL_TUPLE_OFFSET + titem->t_len;
		/*
		 * move the local page to the kds_heap because it cannot load
		 * any tuples here. 
		 */
	again:
		if (offsetof(PageHeaderData,
					 pd_linp[lp_index+1]) + MAXALIGN(tupsz) > lp_offset)
		{
			/* tuple is too large to write out heap blocks without toast */
			if (lp_index == 0)
			{
				gpuClientELog(gclient, "SELECT INTO: too large HeapTuple (tupsz=%u)",
							  tupsz);
				return false;
			}
			hpage->pd_lower = offsetof(PageHeaderData, pd_linp[lp_index]);
			hpage->pd_upper = lp_offset;
			assert(hpage->pd_lower <= hpage->pd_upper);
			if (!__selectIntoWriteOutOneBlock(gclient, si_state, hpage))
				return false;
			/* reset the local buffer usage */
			lp_index = 0;
			lp_offset = BLCKSZ;
			goto again;
		}
		/* ok, the tuple still fits the local block buffer */
		lp_offset -= MAXALIGN(tupsz);
		lp_item.lp_off = lp_offset;
		lp_item.lp_flags = LP_NORMAL;
		lp_item.lp_len = tupsz;
		hpage->pd_linp[lp_index++] = lp_item;
		htup = (HeapTupleHeaderData *)((char *)hpage + lp_offset);
		memcpy(&htup->t_infomask2,
			   &titem->t_infomask2,
			   titem->t_len - offsetof(kern_tupitem, t_infomask2));
		htup->t_choice.t_heap.t_xmin = FrozenTransactionId;
		htup->t_choice.t_heap.t_xmax = InvalidTransactionId;
		htup->t_choice.t_heap.t_field3.t_cid = InvalidCommandId;
		htup->t_choice.t_datum.datum_typeid = kds_dst->tdtypeid;
		htup->t_ctid.ip_blkid.bi_hi = 0xffff;
		htup->t_ctid.ip_blkid.bi_lo = 0xffff;
		htup->t_ctid.ip_posid = 0;
		/* force tuple to all-visible */
		htup->t_infomask &= ~HEAP_XACT_MASK;
		htup->t_infomask |= (HEAP_XMIN_FROZEN | HEAP_XMAX_INVALID);
	}
	/* flush remained tuples using tuple-by-tuple (slow) mode */
	if (lp_index > 0)
	{
		HeapTupleHeaderData *htup;
		PageHeaderData *dpage;

		pthreadMutexLock(&si_state->dpage_lock);
		dpage = si_state->dpage;
		for (int index=0; index < lp_index; index++)
		{
			ItemIdData	lp_item = hpage->pd_linp[index];
			int32_t		tupsz = lp_item.lp_len;

			htup = (HeapTupleHeaderData *)((char *)hpage + lp_item.lp_off);
			if (dpage->pd_lower +
				sizeof(ItemIdData) +
				MAXALIGN(tupsz) > dpage->pd_upper)
			{
				if (!__selectIntoWriteOutOneBlock(gclient, si_state, dpage))
				{
					pthreadMutexUnlock(&si_state->dpage_lock);
					return false;
				}
				/* reset the buffer usage */
				dpage->pd_lower = offsetof(PageHeaderData, pd_linp);
				dpage->pd_upper = BLCKSZ;
			}
			assert(dpage->pd_lower +
				   sizeof(ItemIdData) +
				   MAXALIGN(tupsz) <= dpage->pd_upper);
			dpage->pd_upper -= MAXALIGN(tupsz);
			memcpy((char *)dpage + dpage->pd_upper, htup, tupsz);
			lp_item.lp_off = dpage->pd_upper;
			dpage->pd_linp[(dpage->pd_lower -
							offsetof(PageHeaderData, pd_linp)) / sizeof(ItemIdData)] = lp_item;
			dpage->pd_lower += sizeof(ItemIdData);
		}
		pthreadMutexUnlock(&si_state->dpage_lock);
	}
	return true;
}
