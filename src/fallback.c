/*
 * fallback.c
 *
 * Portion of CPU Fallback operations
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "cuda_common.h"

/*
 * Routines to store/fetch fallback tuples
 */
static void
pgstromStoreFallbackTuple(pgstromTaskState *pts, HeapTuple htuple)
{
	MemoryContext memcxt = pts->css.ss.ps.state->es_query_cxt;
	kern_tupitem *titem;
	uint32_t	rowid;
	size_t		sz;

	if (!pts->fallback_tuples)
	{
		pts->fallback_index = 0;
		pts->fallback_nitems = 0;
		pts->fallback_nrooms = 1000;
		pts->fallback_tuples =
			MemoryContextAlloc(memcxt, sizeof(off_t) * pts->fallback_nrooms);
	}
	if (!pts->fallback_buffer)
	{
		pts->fallback_usage = 0;
		pts->fallback_bufsz = 8 * BLCKSZ;
		pts->fallback_buffer =
			MemoryContextAlloc(memcxt, pts->fallback_bufsz);
	}
	sz = MAXALIGN(offsetof(kern_tupitem, htup) + htuple->t_len);
	while (pts->fallback_usage + sz > pts->fallback_bufsz)
	{
		pts->fallback_bufsz = pts->fallback_bufsz * 2 + BLCKSZ;
		pts->fallback_buffer = repalloc_huge(pts->fallback_buffer,
											 pts->fallback_bufsz);
	}
	while (pts->fallback_nitems >= pts->fallback_nrooms)
	{
		pts->fallback_nrooms = pts->fallback_nrooms * 2 + 100;
		pts->fallback_tuples = repalloc_huge(pts->fallback_tuples,
											 sizeof(off_t) * pts->fallback_nrooms);
	}
	rowid = pts->fallback_nitems++;

	titem = (kern_tupitem *)(pts->fallback_buffer +
							 pts->fallback_usage);
	titem->t_len = htuple->t_len;
	titem->rowid = rowid;
	memcpy(&titem->htup, htuple->t_data, htuple->t_len);

	pts->fallback_tuples[rowid] = pts->fallback_usage;
	pts->fallback_usage += sz;
}

TupleTableSlot *
pgstromFetchFallbackTuple(pgstromTaskState *pts)
{
	if (pts->fallback_tuples &&
		pts->fallback_buffer &&
		pts->fallback_index < pts->fallback_nitems)
	{
		TupleTableSlot *slot = pts->css.ss.ss_ScanTupleSlot;
		HeapTuple		htuple = palloc0(sizeof(HeapTupleData));
		kern_tupitem   *titem;

		titem = (kern_tupitem *)(pts->fallback_buffer +
								 pts->fallback_tuples[pts->fallback_index++]);
		htuple->t_len  = titem->t_len;
		htuple->t_data = &titem->htup;
		ExecForceStoreHeapTuple(htuple, slot, true);
		/* reset the buffer if last one */
		if (pts->fallback_index == pts->fallback_nitems)
		{
			pts->fallback_index = 0;
			pts->fallback_nitems = 0;
			pts->fallback_usage = 0;
		}
		slot_getallattrs(slot);
		return slot;
	}
	return NULL;
}

/*
 * Core routines for CPU-Fallback SCAN/JOIN/PREAGG
 */
static void
__execFallbackCpuJoinOneDepth(pgstromTaskState *pts,
							  int depth,
							  uint64_t l_state,
							  bool matched);

static void
__execFallbackLoadVarsSlot(TupleTableSlot *fallback_slot,
						   List *inner_load_src,
						   List *inner_load_dst,
						   const kern_data_store *kds,
						   const ItemPointer t_self,
						   const HeapTupleHeaderData *htup)
{
	uint32_t	h_off, ncols;
	bool		heap_hasnull;
	ListCell   *lc1, *lc2;

	/* extract system attributes, if rquired */
	forboth (lc1, inner_load_src,
			 lc2, inner_load_dst)
	{
		int		src = lfirst_int(lc1);
		int		dst = lfirst_int(lc2);
		Datum	datum;

		if (!htup)
			goto fillup_by_null;
		if (src >= 0)
			break;
		switch (src)
		{
			case SelfItemPointerAttributeNumber:
				datum = PointerGetDatum(t_self);
				break;
			case MinTransactionIdAttributeNumber:
				datum = TransactionIdGetDatum(HeapTupleHeaderGetRawXmin(htup));
				break;
			case MaxTransactionIdAttributeNumber:
				datum = TransactionIdGetDatum(HeapTupleHeaderGetRawXmax(htup));
				break;
			case MinCommandIdAttributeNumber:
			case MaxCommandIdAttributeNumber:
				datum = CommandIdGetDatum(HeapTupleHeaderGetRawCommandId(htup));
				break;
			case TableOidAttributeNumber:
				datum = ObjectIdGetDatum(kds->table_oid);
				break;
			default:
				elog(ERROR, "invalid attnum: %d", src);
		}
		fallback_slot->tts_isnull[dst] = false;
		fallback_slot->tts_values[dst] = datum;
	}
	/* extract the user data */
	h_off = htup->t_hoff;
	ncols = Min(htup->t_infomask2 & HEAP_NATTS_MASK, kds->ncols);
	heap_hasnull = ((htup->t_infomask & HEAP_HASNULL) != 0);
	for (int j=0; j < ncols && lc1 && lc2; j++)
	{
		const kern_colmeta *cmeta = &kds->colmeta[j];
		const char *addr;
		Datum		datum;

		if (heap_hasnull && att_isnull(j, htup->t_bits))
		{
			addr = NULL;
			datum = 0;
		}
		else
		{
			if (cmeta->attlen > 0)
				h_off = TYPEALIGN(cmeta->attalign, h_off);
			else if (!VARATT_NOT_PAD_BYTE((char *)htup + h_off))
				h_off = TYPEALIGN(cmeta->attalign, h_off);
			addr = ((char *)htup + h_off);
			if (cmeta->attlen > 0)
				h_off += cmeta->attlen;
			else if (cmeta->attlen == -1)
				h_off += VARSIZE_ANY(addr);
			else
				elog(ERROR, "unknown typlen (%d)", cmeta->attlen);

			if (cmeta->attbyval)
			{
				switch (cmeta->attlen)
				{
					case 1:
						datum = *((uint8_t *)addr);
						break;
					case 2:
						datum = *((uint16_t *)addr);
						break;
					case 4:
						datum = *((uint32_t *)addr);
						break;
					case 8:
						datum = *((uint64_t *)addr);
						break;
					default:
						elog(ERROR, "invalid typlen (%d) of inline type",
							 cmeta->attlen);
				}
			}
			else
			{
				datum = PointerGetDatum(addr);
			}
		}
		if (lfirst_int(lc1) == j+1)
		{
			int		dst = lfirst_int(lc2) - 1;

			fallback_slot->tts_isnull[dst] = !addr;
			fallback_slot->tts_values[dst] = datum;

			lc1 = lnext(inner_load_src, lc1);
			lc2 = lnext(inner_load_dst, lc2);
		}
	}
fillup_by_null:
	/* fill-up by NULL for the remaining fields */
	while (lc1 && lc2)
	{
		int		dst = lfirst_int(lc2) - 1;

		fallback_slot->tts_isnull[dst] = true;
		fallback_slot->tts_values[dst] = 0;

		lc1 = lnext(inner_load_src, lc1);
		lc2 = lnext(inner_load_dst, lc2);
	}
}

static void
__execFallbackCpuNestLoop(pgstromTaskState *pts,
						  kern_data_store *kds_in,
						  bool *oj_map,
						  int depth,
						  uint64_t l_state, bool matched)
{
	pgstromTaskInnerState *istate = &pts->inners[depth-1];
	ExprContext    *econtext = pts->css.ss.ps.ps_ExprContext;
	TupleTableSlot *scan_slot = pts->css.ss.ss_ScanTupleSlot;

	Assert(kds_in->format == KDS_FORMAT_ROW);
	for (uint32_t index=l_state; index < kds_in->nitems; index++)
	{
		kern_tupitem   *tupitem = KDS_GET_TUPITEM(kds_in, index);

		if (!tupitem)
			continue;
		ResetExprContext(econtext);
		/* load inner variable */
		if (istate->inner_load_src != NIL &&
			istate->inner_load_dst != NIL)
		{
			ItemPointerData	t_self;

			ItemPointerSetInvalid(&t_self);
			__execFallbackLoadVarsSlot(scan_slot,
									   istate->inner_load_src,
									   istate->inner_load_dst,
									   kds_in,
									   &t_self,
									   &tupitem->htup);
		}
		/* check JOIN-clause */
		if (istate->join_quals == NULL ||
			ExecQual(istate->join_quals, econtext))
		{
			if (istate->other_quals == NULL ||
				ExecQual(istate->other_quals, econtext))
			{
				/* Ok, go to the next depth */
				__execFallbackCpuJoinOneDepth(pts, depth+1, 0, false);
			}
			/* mark outer-join map, if any */
			if (oj_map)
				oj_map[index] = true;
			/* mark as 'matched' in this depth */
			matched = true;
		}
	}

	/* LEFT OUTER JOIN handling */
	if (!matched && (istate->join_type == JOIN_LEFT ||
					 istate->join_type == JOIN_FULL))
	{
		__execFallbackLoadVarsSlot(scan_slot,
								   istate->inner_load_src,
								   istate->inner_load_dst,
								   kds_in,
								   NULL,
								   NULL);
		__execFallbackCpuJoinOneDepth(pts, depth+1, 0, false);
	}
}

static void
__execFallbackCpuHashJoin(pgstromTaskState *pts,
						  kern_data_store *kds_in,
						  bool *oj_map,
						  int depth,
						  uint64_t l_state, bool matched)
{
	pgstromTaskInnerState *istate = &pts->inners[depth-1];
	ExprContext    *econtext = pts->css.ss.ps.ps_ExprContext;
	TupleTableSlot *scan_slot = pts->css.ss.ss_ScanTupleSlot;
	kern_hashitem  *hitem;
	uint32_t		hash;
	ListCell	   *lc1, *lc2;

	Assert(kds_in->format == KDS_FORMAT_HASH);

	/*
	 * Compute that hash-value
	 */
	if (l_state == 0)
	{
		hash = 0xffffffffU;
		forboth (lc1, istate->hash_outer_keys,
				 lc2, istate->hash_outer_funcs)
		{
			ExprState   *h_key = lfirst(lc1);
			devtype_hashfunc_f h_func = lfirst(lc2);
			Datum		datum;
			bool		isnull;

			datum = ExecEvalExprSwitchContext(h_key, econtext, &isnull);
			hash = pg_hash_merge(hash, h_func(isnull, datum));
		}
		hash ^= 0xffffffffU;
		hitem = KDS_HASH_FIRST_ITEM(kds_in, hash);
	}
	else
	{
		/* restart from the hash-chain */
		Assert(l_state <  kds_in->length &&
			   l_state >= kds_in->length - kds_in->usage);
		hitem = (kern_hashitem *)((char *)kds_in + l_state);
		hash = hitem->hash;
	}

	/*
	 * walks on the hash-join-table
	 */
	while (hitem != NULL)
	{
		if (hitem->hash == hash)
		{
			if (istate->inner_load_src != NIL &&
				istate->inner_load_dst != NIL)
			{
				ItemPointerData	t_self;

				ItemPointerSetInvalid(&t_self);
				__execFallbackLoadVarsSlot(scan_slot,
										   istate->inner_load_src,
										   istate->inner_load_dst,
										   kds_in,
										   &t_self,
										   &hitem->t.htup);
			}
			/* check JOIN-clause */
			if (istate->join_quals == NULL ||
				ExecQual(istate->join_quals, econtext))
			{
				if (istate->other_quals == NULL ||
					ExecQual(istate->other_quals, econtext))
				{
					/* Ok, go to the next depth */
					__execFallbackCpuJoinOneDepth(pts, depth+1, 0, false);
				}
				/* mark outer-join map, if any */
				if (oj_map)
					oj_map[hitem->t.rowid] = true;
				/* mark as 'matched' in this depth */
				matched = true;
			}
		}
		hitem = KDS_HASH_NEXT_ITEM(kds_in, hitem->next);
	}

	/* LEFT OUTER JOIN handling */
	if (!matched && (istate->join_type == JOIN_LEFT ||
					 istate->join_type == JOIN_FULL))
	{
		__execFallbackLoadVarsSlot(scan_slot,
								   istate->inner_load_src,
								   istate->inner_load_dst,
								   kds_in,
								   NULL,
								   NULL);
		__execFallbackCpuJoinOneDepth(pts, depth+1, 0, false);
	}
}

static void
__execFallbackCpuJoinOneDepth(pgstromTaskState *pts,
							  int depth,
							  uint64_t l_state,
							  bool matched)
{
	if (depth > pts->num_rels)
	{
		ExprContext	   *econtext = pts->css.ss.ps.ps_ExprContext;
		TupleTableSlot *scan_slot = pts->css.ss.ss_ScanTupleSlot;
		HeapTuple		tuple;

		/* apply projection if any */
		if (pts->fallback_proj)
		{
			ListCell   *lc;
			int			dst = 0;

			foreach (lc, pts->fallback_proj)
			{
				ExprState  *state = lfirst(lc);
				Datum	datum;
				bool	isnull;

				if (state)
				{
					datum = ExecEvalExpr(state, econtext, &isnull);

					if (isnull)
					{
						scan_slot->tts_isnull[dst] = true;
						scan_slot->tts_values[dst] = 0;
					}
					else
					{
						scan_slot->tts_isnull[dst] = false;
						scan_slot->tts_values[dst] = datum;
					}
				}
				dst++;
			}
		}
		tuple = heap_form_tuple(scan_slot->tts_tupleDescriptor,
								scan_slot->tts_values,
								scan_slot->tts_isnull);
		pgstromStoreFallbackTuple(pts, tuple);
		pfree(tuple);
	}
	else
	{
		kern_multirels	   *h_kmrels = pts->h_kmrels;
		kern_data_store	   *kds_in;
		bool			   *oj_map;

		kds_in = KERN_MULTIRELS_INNER_KDS(h_kmrels, depth-1);
		oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(h_kmrels, depth-1);
		if (h_kmrels->chunks[depth-1].is_nestloop)
		{
			__execFallbackCpuNestLoop(pts, kds_in, oj_map, depth, l_state, matched);
		}
		else
		{
			__execFallbackCpuHashJoin(pts, kds_in, oj_map, depth, l_state, matched);
		}
	}
}

/*
 * execCpuFallbackOneTuple
 */
static void
execCpuFallbackOneTuple(pgstromTaskState *pts,
						int depth,
						uint64_t l_state,
						bool matched)
{
	ExprContext	   *econtext = pts->css.ss.ps.ps_ExprContext;
	TupleTableSlot *scan_slot = pts->css.ss.ss_ScanTupleSlot;

	/*
	 * scan_slot must be set on the ecxt_scantuple prior to
	 * any ExecQual() because Var node is modified to reference
	 * INDEX_VAR.
	 */
	econtext->ecxt_scantuple = scan_slot;

	/* check WHERE-clause if any */
	if (depth == 0)
	{
		Assert(l_state == 0 && !matched);
		econtext->ecxt_scantuple = scan_slot;
		if (pts->base_quals)
		{
			ResetExprContext(econtext);
			if (!ExecQual(pts->base_quals, econtext))
				return;
		}
		depth++;
	}
	/* Run JOIN, if any */
	__execFallbackCpuJoinOneDepth(pts, depth, l_state, matched);
}

/*
 * execCpuFallbackBaseTuple
 */
void
execCpuFallbackBaseTuple(pgstromTaskState *pts,
						 HeapTuple base_tuple)
{
	TupleTableSlot *base_slot = pts->base_slot;
	TupleTableSlot *scan_slot = pts->css.ss.ss_ScanTupleSlot;
	ListCell	   *lc1, *lc2;

	ExecForceStoreHeapTuple(base_tuple, base_slot, false);
	slot_getallattrs(base_slot);
	ExecStoreAllNullTuple(scan_slot);
	forboth (lc1, pts->fallback_load_src,
			 lc2, pts->fallback_load_dst)
	{
		int		src = lfirst_int(lc1) - 1;
		int		dst = lfirst_int(lc2) - 1;

		scan_slot->tts_isnull[dst] = base_slot->tts_isnull[src];
		scan_slot->tts_values[dst] = base_slot->tts_values[src];
	}
	if (pts->ps_state)
	{
		pgstromSharedState *ps_state = pts->ps_state;
		pg_atomic_fetch_add_u64(&ps_state->fallback_nitems, 1);
	}
	execCpuFallbackOneTuple(pts, 0, 0, false);
}

/*
 * execCpuFallbackOneChunk
 */
void
execCpuFallbackOneChunk(pgstromTaskState *pts)
{
	kern_data_store	   *kds = pts->curr_kds;

	if (kds->format == KDS_FORMAT_FALLBACK)
	{
		Relation		scan_rel = pts->css.ss.ss_currentRelation;
		TupleTableSlot *scan_slot = pts->css.ss.ss_ScanTupleSlot;
		uint64_t	   *rowindex = KDS_GET_ROWINDEX(kds);
		HeapTupleData	htuple;

		elog(pgstrom_cpu_fallback_elevel, "%s: CPU fallback %u tuples (%s)",
			 pts->css.methods->CustomName, kds->nitems, format_bytesz(kds->usage));
		for (uint32_t i=0; i < kds->nitems; i++)
		{
			kern_fallbackitem *fb_item = (kern_fallbackitem *)
				((char *)kds + kds->length - rowindex[i]);

			ExecStoreAllNullTuple(scan_slot);
			htuple.t_len  = fb_item->t_len;
			ItemPointerSetInvalid(&htuple.t_self);
			htuple.t_tableOid = RelationGetRelid(scan_rel);
			htuple.t_data = &fb_item->htup;
			heap_deform_tuple(&htuple,
							  scan_slot->tts_tupleDescriptor,
							  scan_slot->tts_values,
							  scan_slot->tts_isnull);
			if (pts->ps_state)
			{
				pgstromSharedState *ps_state = pts->ps_state;
				int		depth = fb_item->depth;

				if (depth == 0)
					pg_atomic_fetch_add_u64(&ps_state->fallback_nitems, 1);
				else if (depth <= pts->num_rels)
					pg_atomic_fetch_add_u64(&ps_state->inners[depth-1].fallback_nitems, 1);
			}
			execCpuFallbackOneTuple(pts,
									fb_item->depth,
									fb_item->l_state,
									fb_item->matched);
		}
	}
}

/*
 * ExecFallbackCpuJoinRightOuter
 */
static void
__execFallbackCpuJoinRightOuterOneDepth(pgstromTaskState *pts, int depth)
{
	pgstromTaskInnerState *istate = &pts->inners[depth-1];
	ExprContext		   *econtext = pts->css.ss.ps.ps_ExprContext;
	TupleTableSlot	   *fallback_slot = pts->css.ss.ss_ScanTupleSlot;
	kern_multirels	   *h_kmrels = pts->h_kmrels;
	kern_data_store	   *kds_in = KERN_MULTIRELS_INNER_KDS(h_kmrels, depth-1);
	bool			   *oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(h_kmrels, depth-1);

	Assert(oj_map != NULL);

	ExecStoreAllNullTuple(fallback_slot);
	econtext->ecxt_scantuple = fallback_slot;
	for (uint32_t i=0; i < kds_in->nitems; i++)
	{
		if (oj_map[i])
			continue;
		if (istate->inner_load_src != NIL &&
			istate->inner_load_dst != NIL)
		{
			kern_tupitem   *titem = KDS_GET_TUPITEM(kds_in, i);
			ItemPointerData	t_self;

			if (!titem)
				continue;
			ItemPointerSetInvalid(&t_self);
			__execFallbackLoadVarsSlot(fallback_slot,
									   istate->inner_load_src,
									   istate->inner_load_dst,
									   kds_in,
									   &t_self,
									   &titem->htup);
		}
		if (istate->other_quals && !ExecQual(istate->other_quals, econtext))
			continue;
		__execFallbackCpuJoinOneDepth(pts, depth+1, 0, false);
	}
}

void
ExecFallbackCpuJoinRightOuter(pgstromTaskState *pts)
{
	uint32_t	count;

	count = pg_atomic_add_fetch_u32(pts->rjoin_exit_count, 1);
	//TODO: use sibling count if partitioned join
	if (count == 1)
	{
		for (int depth=1; depth <= pts->num_rels; depth++)
		{
			JoinType	join_type = pts->inners[depth-1].join_type;

			if (join_type == JOIN_RIGHT || join_type == JOIN_FULL)
				__execFallbackCpuJoinRightOuterOneDepth(pts, depth);
		}
	}
}

/*
 * ExecFallbackCpuJoinOuterJoinMap
 *
 * NOTE: GPU-Service updates the outer-join-map of the host-buffer by itself,
 *       so this routine makes sense only if DPU-mode.
 */
void
ExecFallbackCpuJoinOuterJoinMap(pgstromTaskState *pts, XpuCommand *resp)
{
	kern_multirels *h_kmrels = pts->h_kmrels;
	bool	   *ojmap_resp = (bool *)((char *)resp + resp->u.results.ojmap_offset);

	if (resp->u.results.ojmap_offset == 0)
		return;

	Assert(resp->u.results.ojmap_offset +
		   resp->u.results.ojmap_length <= resp->length);
	for (int depth=1; depth <= pts->num_rels; depth++)
	{
		kern_data_store *kds_in = KERN_MULTIRELS_INNER_KDS(h_kmrels, depth-1);
		bool   *ojmap_curr = KERN_MULTIRELS_OUTER_JOIN_MAP(h_kmrels, depth-1);

		if (!ojmap_curr)
			continue;

		for (uint32_t i=0; i < kds_in->nitems; i++)
		{
			ojmap_curr[i] |= ojmap_resp[i];
		}
		ojmap_resp += MAXALIGN(sizeof(bool) * kds_in->nitems);
	}
}
