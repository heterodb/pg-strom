/*
 * cuda_gpujoin.cu
 *
 * GPU accelerated parallel relations join based on hash-join or
 * nested-loop logic.
 * --
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "cuda_common.h"

/*
 * GPU Nested-Loop
 */
STATIC_FUNCTION(int)
execGpuJoinNestLoop(kern_context *kcxt,
					kern_warp_context *wp,
					kern_multirels *kmrels,
					int			depth,
					char	   *src_kvecs_buffer,
					char	   *dst_kvars_addr_wp, //FIXME
					uint32_t   &l_state,
					bool	   &matched)
{
	kern_data_store *kds_heap = KERN_MULTIRELS_INNER_KDS(kmrels, depth-1);
	bool	   *oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth-1);
	kern_expression *kexp;
	uint32_t	read_pos;
	uint32_t	write_pos;
	uint32_t	mask;
	bool		left_outer = kmrels->chunks[depth-1].left_outer;
	bool		tuple_is_valid = false;

	if (WARP_WRITE_POS(wp,depth) >= WARP_READ_POS(wp,depth) + warpSize)
	{
		/*
		 * The destination depth already keeps warpSize or more pending
		 * tuple. So, flush out these tuples first.
		 */
		return depth+1;
	}

	if (__all_sync(__activemask(), l_state >= kds_heap->nitems) &&
		(!left_outer || __all_sync(__activemask(), l_state == UINT_MAX)))
	{
		/*
		 * OK, all the threads in this warp reached to the end of hash-slot
		 * chain. Due to the above checks, the next depth has enough space
		 * to store the result in this depth.
		 */
		if (LaneId() == 0)
			WARP_READ_POS(wp,depth-1) = Min(WARP_READ_POS(wp,depth-1) + warpSize,
											WARP_WRITE_POS(wp,depth-1));
		__syncwarp();
		l_state = 0;
		matched = false;
		if (wp->scan_done >= depth)
		{
			assert(wp->scan_done == depth);
			if (WARP_READ_POS(wp,depth-1) >= WARP_WRITE_POS(wp,depth-1))
			{
				if (LaneId() == 0)
					wp->scan_done = depth + 1;
				return depth+1;
			}
			/*
			 * Elsewhere, remaining tuples in the combination buffer
			 * shall be wiped-out first, then, we update 'scan_done'
			 * to mark this depth will never generate results any more.
			 */
		}
		else
		{
			/* back to the previous depth to generate the source tuples. */
			if (WARP_READ_POS(wp,depth-1) + warpSize > WARP_WRITE_POS(wp,depth-1))
				return depth-1;
		}
	}

	read_pos = WARP_READ_POS(wp,depth-1) + LaneId();
	if (read_pos < WARP_WRITE_POS(wp,depth-1))
	{
		uint32_t	index = l_state++;

		kcxt->kvecs_curr_id = (read_pos % KVEC_UNITSZ);
		kcxt->kvecs_curr_buffer = src_kvecs_buffer;
		if (index < kds_heap->nitems)
		{
			kern_tupitem *tupitem;
			uint32_t	offset = KDS_GET_ROWINDEX(kds_heap)[index];
			xpu_int4_t	status;

			tupitem = (kern_tupitem *)((char *)kds_heap +
									   kds_heap->length -
									   __kds_unpack(offset));
			kexp = SESSION_KEXP_LOAD_VARS(kcxt->session, depth);
			ExecLoadVarsHeapTuple(kcxt, kexp, depth, kds_heap, &tupitem->htup);
			kexp = SESSION_KEXP_JOIN_QUALS(kcxt->session, depth);
			if (EXEC_KERN_EXPRESSION(kcxt, kexp, &status))
			{
				assert(!XPU_DATUM_ISNULL(&status));
				if (status.value > 0)
					tuple_is_valid = true;
				if (status.value != 0)
					matched = true;
			}
			if (oj_map && matched)
			{
				assert(tupitem->rowid < kds_heap->nitems);
				oj_map[tupitem->rowid] = true;
			}
		}
		else if (left_outer && index >= kds_heap->nitems && !matched)
		{
			/* fill up NULL fields, if FULL/LEFT OUTER JOIN */
			kexp = SESSION_KEXP_LOAD_VARS(kcxt->session, depth);
			ExecLoadVarsHeapTuple(kcxt, kexp, depth, kds_heap, NULL);
			tuple_is_valid = true;
			l_state = UINT_MAX;
		}
		else
		{
			l_state = UINT_MAX;
		}
	}
	else
	{
		l_state = UINT_MAX;
	}
	/* error checks */
	if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
		return -1;
	/* save the result */
	mask = __ballot_sync(__activemask(), tuple_is_valid);
	if (LaneId() == 0)
	{
		write_pos = WARP_WRITE_POS(wp,depth);
		WARP_WRITE_POS(wp,depth) += __popc(mask);
	}
	write_pos = __shfl_sync(__activemask(), write_pos, 0);
	mask &= ((1U << LaneId()) - 1);
	write_pos += __popc(mask);

	if (tuple_is_valid)
	{
		write_pos = (write_pos % UNIT_TUPLES_PER_DEPTH);
		memcpy(dst_kvars_addr_wp + write_pos * kcxt->kvars_nbytes,
			   kcxt->kvars_slot,
			   kcxt->kvars_nbytes);
	}
	__syncwarp();
	if (WARP_WRITE_POS(wp,depth) >= WARP_READ_POS(wp,depth) + warpSize)
		return depth+1;
	return depth;
}

/*
 * GPU Hash-Join
 */
STATIC_FUNCTION(int)
execGpuJoinHashJoin(kern_context *kcxt,
					kern_warp_context *wp,
					kern_multirels *kmrels,
					int			depth,
					char	   *src_kvecs_buffer,
					char	   *dst_kvars_addr_wp,
					uint32_t   &l_state,
					bool	   &matched)
{
	kern_data_store *kds_hash = KERN_MULTIRELS_INNER_KDS(kmrels, depth-1);
	bool	   *oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth-1);
	kern_expression *kexp = NULL;
	kern_hashitem *khitem = NULL;
	uint32_t	read_pos;
	uint32_t	write_pos;
	uint32_t	index;
	uint32_t	mask;
	bool		tuple_is_valid = false;

	if (WARP_WRITE_POS(wp,depth) >= WARP_READ_POS(wp,depth) + warpSize)
	{
		/*
		 * Next depth already keeps warpSize or more pending tuples,
		 * so wipe out these tuples first.
		 */
		return depth+1;
	}

	if (__all_sync(__activemask(), l_state == UINT_MAX))
	{
		/*
		 * OK, all the threads in this warp reached to the end of hash-slot
		 * chain. Due to the above checks, the next depth has enough space
		 * to store the result in this depth.
		 * So, we process this depth again (if we have enough pending tuples),
		 * back to the previsou depth (if we don't have enough pending tuples
		 * in this depth), or move to the next depth if previous depth already
		 * reached to end of the chunk.
		 */
		if (LaneId() == 0)
			WARP_READ_POS(wp,depth-1) = Min(WARP_READ_POS(wp,depth-1) + warpSize,
											WARP_WRITE_POS(wp,depth-1));
		__syncwarp();
		l_state = 0;
		matched = false;
		if (wp->scan_done < depth)
		{
			/*
			 * The previous depth still may generate the source tuple.
			 */
			if (WARP_WRITE_POS(wp,depth-1) < WARP_READ_POS(wp,depth-1) + warpSize)
				return depth-1;
		}
		else
		{
			assert(wp->scan_done == depth);
			if (WARP_READ_POS(wp,depth-1) >= WARP_WRITE_POS(wp,depth-1))
			{
				if (LaneId() == 0)
					wp->scan_done = depth+1;
				return depth+1;
			}
			/*
			 * Elsewhere, remaining tuples in the combination buffer
			 * shall be wiped-out first, then, we update 'scan_done'
			 * to mark this depth will never generate results any more.
			 */
		}
	}
	write_pos = WARP_WRITE_POS(wp,depth-1);
	read_pos = WARP_READ_POS(wp,depth-1) + LaneId();
	kcxt->kvecs_curr_id = (read_pos % KVEC_UNITSZ);
	kcxt->kvecs_curr_buffer = src_kvecs_buffer;

	if (l_state == 0)
	{
		/* pick up the first item from the hash-slot */
		if (read_pos < write_pos)
		{
			xpu_int4_t	hash;

			kexp = SESSION_KEXP_HASH_VALUE(kcxt->session, depth);
			if (EXEC_KERN_EXPRESSION(kcxt, kexp, &hash))
			{
				assert(!XPU_DATUM_ISNULL(&hash));
				for (khitem = KDS_HASH_FIRST_ITEM(kds_hash, hash.value);
					 khitem != NULL && khitem->hash != hash.value;
					 khitem = KDS_HASH_NEXT_ITEM(kds_hash, khitem->next));
			}
		}
		else
		{
			l_state = UINT_MAX;
		}
	}
	else if (l_state != UINT_MAX)
	{
		/* pick up the next one if any */
		uint32_t	hash_value;

		khitem = (kern_hashitem *)((char *)kds_hash + __kds_unpack(l_state));
		hash_value = khitem->hash;
		for (khitem = KDS_HASH_NEXT_ITEM(kds_hash, khitem->next);
			 khitem != NULL && khitem->hash != hash_value;
			 khitem = KDS_HASH_NEXT_ITEM(kds_hash, khitem->next));
	}
	/* error checks */
	if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
		return -1;

	if (khitem)
	{
		xpu_int4_t	status;

		kexp = SESSION_KEXP_LOAD_VARS(kcxt->session, depth);
		ExecLoadVarsHeapTuple(kcxt, kexp, depth, kds_hash, &khitem->t.htup);
		kexp = SESSION_KEXP_JOIN_QUALS(kcxt->session, depth);
		if (EXEC_KERN_EXPRESSION(kcxt, kexp, &status))
		{
			assert(!XPU_DATUM_ISNULL(&status));
			if (status.value > 0)
				tuple_is_valid = true;
			if (status.value != 0)
				matched = true;
		}
		if (oj_map && matched)
		{
			assert(khitem->t.rowid < kds_hash->nitems);
			oj_map[khitem->t.rowid] = true;
		}
		l_state = __kds_packed((char *)khitem - (char *)kds_hash);
	}
	else
	{
		if (kmrels->chunks[depth-1].left_outer &&
			l_state != UINT_MAX && !matched)
		{
			/* load NULL values on the inner portion */
			 kexp = SESSION_KEXP_LOAD_VARS(kcxt->session, depth);
			 ExecLoadVarsHeapTuple(kcxt, kexp, depth, kds_hash, NULL);
			 tuple_is_valid = true;
		}
		l_state = UINT_MAX;
	}
	/* error checks */
	if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
		return -1;
	/* save the result on the destination buffer */
	mask = __ballot_sync(__activemask(), tuple_is_valid);
	if (LaneId() == 0)
	{
		write_pos = WARP_WRITE_POS(wp,depth);
		WARP_WRITE_POS(wp,depth) += __popc(mask);
	}
	write_pos = __shfl_sync(__activemask(), write_pos, 0);
	mask &= ((1U << LaneId()) - 1);
	write_pos += __popc(mask);
	if (tuple_is_valid)
	{
		index = write_pos % UNIT_TUPLES_PER_DEPTH;
		memcpy(dst_kvars_addr_wp + index * kcxt->kvars_nbytes,
			   kcxt->kvars_slot,
			   kcxt->kvars_nbytes);
	}
	__syncwarp();
	if (WARP_WRITE_POS(wp,depth) >= WARP_READ_POS(wp,depth) + warpSize)
		return depth+1;
	return depth;
}

/*
 * gpujoin_prep_gistindex
 */
KERNEL_FUNCTION(void)
gpujoin_prep_gistindex(kern_multirels *kmrels, int depth)
{
	kern_data_store *kds_hash = KERN_MULTIRELS_INNER_KDS(kmrels, depth-1);
	kern_data_store *kds_gist = KERN_MULTIRELS_GIST_INDEX(kmrels, depth-1);
	BlockNumber		block_nr;
	OffsetNumber	i, maxoff;

	assert(kds_hash && kds_hash->format == KDS_FORMAT_HASH &&
		   kds_gist && kds_gist->format == KDS_FORMAT_BLOCK);
	for (block_nr = get_group_id();
		 block_nr < kds_gist->nitems;
		 block_nr += get_num_groups())
	{
		PageHeaderData *gist_page;
		ItemIdData	   *lpp;
		IndexTupleData *itup;
		kern_hashitem  *khitem;
		uint32_t		hash, t_off;

		gist_page = KDS_BLOCK_PGPAGE(kds_gist, block_nr);
		if (!GistPageIsLeaf(gist_page))
			continue;
		maxoff = PageGetMaxOffsetNumber(gist_page);
		for (i = get_local_id(); i < maxoff; i += get_local_size())
		{
			lpp = PageGetItemId(gist_page, i+1);
			if (ItemIdIsDead(lpp))
				continue;
			itup = (IndexTupleData *)PageGetItem(gist_page, lpp);

			/* lookup kds_hash */
			hash = pg_hash_any(&itup->t_tid, sizeof(ItemPointerData));
			for (khitem = KDS_HASH_FIRST_ITEM(kds_hash, hash);
				 khitem != NULL;
				 khitem = KDS_HASH_NEXT_ITEM(kds_hash, khitem->next))
			{
				if (ItemPointerEquals(&khitem->t.htup.t_ctid, &itup->t_tid))
				{
					t_off = __kds_packed((char *)&khitem->t.htup -
										 (char *)kds_hash);
					itup->t_tid.ip_blkid.bi_hi = (t_off >> 16);
					itup->t_tid.ip_blkid.bi_lo = (t_off & 0x0000ffffU);
					itup->t_tid.ip_posid = InvalidOffsetNumber;
					break;
				}
			}
			/* invalidate this leaf item, if not exist on kds_hash */
			if (!khitem)
				lpp->lp_flags = LP_DEAD;
		}
	}
}

/*
 * GiST-INDEX-JOIN
 */
STATIC_FUNCTION(int)
execGpuJoinGiSTJoin(kern_context *kcxt,
					kern_warp_context *wp,
					kern_multirels *kmrels,
					int         depth,
					char       *src_kvars_addr_wp,	//FIXME
					char       *dst_kvars_addr_wp,	//FIXME
					const kern_expression *kexp_gist,
					char	   *gist_kvars_addr_wp,
					uint32_t   &l_state,
					bool       &matched)
{
	kern_data_store *kds_hash = KERN_MULTIRELS_INNER_KDS(kmrels, depth-1);
	kern_data_store *kds_gist = KERN_MULTIRELS_GIST_INDEX(kmrels, depth-1);
	int				gist_depth = kexp_gist->u.gist.gist_depth;
	uint32_t		mask, index;
	uint32_t		read_pos;
	uint32_t		write_pos;

	assert(kds_hash && kds_hash->format == KDS_FORMAT_HASH &&
		   kds_gist && kds_gist->format == KDS_FORMAT_BLOCK);

	if (wp->scan_done > depth)
	{
		/*
		 * This depth will not generate any more tuples, so we move to
		 * the next level.
		 */
		return depth+1;
	}

	if (WARP_WRITE_POS(wp,depth) >= WARP_READ_POS(wp,depth) + warpSize)
	{
		/*
		 * Next depth already have warpSize or more pending tuples,
		 * so wipe out these tuples first.
		 */
		return depth+1;
	}

	if (WARP_WRITE_POS(wp,gist_depth)  >= WARP_READ_POS(wp,gist_depth) + warpSize ||
		(wp->scan_done >= depth &&		/* is terminal case? */
		 WARP_WRITE_POS(wp,depth-1) == WARP_READ_POS(wp,depth-1) &&
		 __all_sync(__activemask(), l_state == UINT_MAX)))
	{
		/*
		 * We already have 32 or more pending tuples; that is fetched by
		 * the GiST-index. So, try to fetch Join-Quals for these tuples.
		 */
		bool	join_is_valid = false;

		read_pos = WARP_READ_POS(wp,gist_depth) + LaneId();
		if (read_pos < WARP_WRITE_POS(wp,gist_depth))
		{
			const kern_expression *kexp_load
				= SESSION_KEXP_LOAD_VARS(kcxt->session, depth);
			const kern_expression *kexp_join
				= SESSION_KEXP_JOIN_QUALS(kcxt->session, depth);

			index = (read_pos % UNIT_TUPLES_PER_DEPTH);
			kcxt->kvars_slot = (kern_variable *)
				(gist_kvars_addr_wp + index * kcxt->kvars_nbytes);
			kcxt->kvars_class = (int *)(kcxt->kvars_slot + kcxt->kvars_nslots);
			//Run LoadVar from the GiST-depth
			join_is_valid = ExecGiSTIndexPostQuals(kcxt, depth,
												   kds_hash,
												   kexp_gist,
												   kexp_load,
												   kexp_join);
		}
		/* error checks */
		if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
			return -1;
		if (LaneId() == 0)
			WARP_READ_POS(wp,gist_depth) = Max(WARP_READ_POS(wp,gist_depth) + warpSize,
											   WARP_WRITE_POS(wp,gist_depth));

		mask = __ballot_sync(__activemask(), join_is_valid);
		if (LaneId() == 0)
		{
			write_pos = WARP_WRITE_POS(wp,depth);
			WARP_WRITE_POS(wp,depth) += __popc(mask);
		}
		write_pos = __shfl_sync(__activemask(), write_pos, 0);
		mask &= ((1U << LaneId()) - 1);
		write_pos += __popc(mask);
		if (join_is_valid)
		{
			index = write_pos % UNIT_TUPLES_PER_DEPTH;
			memcpy(dst_kvars_addr_wp + index * kcxt->kvars_nbytes,
				   kcxt->kvars_slot,
				   kcxt->kvars_nbytes);
		}
		__syncwarp();
		/* termination checks */
		if (LaneId() == 0 &&
			wp->scan_done >= depth &&
			WARP_WRITE_POS(wp,depth-1) == WARP_READ_POS(wp,depth-1) &&
			WARP_WRITE_POS(wp,gist_depth) <= WARP_READ_POS(wp,gist_depth) &&
			__all_sync(__activemask(), l_state == UINT_MAX))
		{
			assert(wp->scan_done == depth);
			wp->scan_done++;
			depth++;
		}
		return __shfl_sync(__activemask(), depth, 0);
	}

	if (__all_sync(__activemask(), l_state == UINT_MAX))
	{
		/*
		 * OK, all the threads in this warp reached to the end of the GiST
		 * index tree. Due to the above checks, the next depth has enough
		 * space to store the result in this depth.
		 */
		if (LaneId() == 0)
			WARP_READ_POS(wp,depth-1) = Min(WARP_READ_POS(wp,depth-1) + warpSize,
											WARP_WRITE_POS(wp,depth-1));
		__syncwarp();
		l_state = 0;
		matched = false;
		if (wp->scan_done < depth)
		{
			/* back to the previous depth; that still may generate source tuples */
			if (WARP_WRITE_POS(wp,depth-1) < WARP_READ_POS(wp,depth-1) + warpSize)
				return depth-1;
		}
		else
		{
			assert(wp->scan_done == depth);
			if (WARP_WRITE_POS(wp,depth-1) <= WARP_READ_POS(wp,depth-1))
			{
				/* wipe out the remaining tuples */
				return depth;
			}
			/*
			 * Elsewhere, the pending source tuples should be processed
			 * first, then, we update the 'scan_done' to mark this depth
			 * will never generate any results.
			 */
		}
	}

	/*
	 * Restart GiST-index scan from the head, or the previous position
	 */
	read_pos = WARP_READ_POS(wp,depth-1) + LaneId();
	if (read_pos < WARP_WRITE_POS(wp,depth-1))
	{
		if (l_state != UINT_MAX)
		{
			index = (read_pos % UNIT_TUPLES_PER_DEPTH);
			kcxt->kvars_slot = (kern_variable *)
				(src_kvars_addr_wp + index * kcxt->kvars_nbytes);
			kcxt->kvars_class = (int *)(kcxt->kvars_slot + kcxt->kvars_nslots);

			l_state = ExecGiSTIndexGetNext(kcxt,
										   kds_hash,
										   kds_gist,
										   kexp_gist,
										   l_state);
		}
	}
	else
	{
		l_state = UINT_MAX;
	}
	/* error checks */
	if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
		return -1;
	/* save the result on the destination buffer */
	mask = __ballot_sync(__activemask(), l_state != UINT_MAX);
	if (LaneId() == 0)
	{
		write_pos = WARP_WRITE_POS(wp,gist_depth);
		WARP_WRITE_POS(wp,gist_depth) += __popc(mask);
	}
	write_pos = __shfl_sync(__activemask(), write_pos, 0);
	mask &= ((1U << LaneId()) - 1);
	write_pos += __popc(mask);
	if (l_state != UINT_MAX)
	{
		index = write_pos % UNIT_TUPLES_PER_DEPTH;

		memcpy(gist_kvars_addr_wp + index * kcxt->kvars_nbytes,
			   kcxt->kvars_slot,
			   kcxt->kvars_nbytes);
	}
	__syncwarp();
	return depth;
}

/*
 * GPU Projection
 */
PUBLIC_FUNCTION(int)
execGpuJoinProjection(kern_context *kcxt,
					  kern_warp_context *wp,
					  int n_rels,	/* index of read/write-pos */
					  kern_data_store *kds_dst,
					  kern_expression *kexp_projection,
					  char *src_kvecs_buffer,
					  bool *p_try_suspend)
{
	uint32_t	write_pos = WARP_WRITE_POS(wp,n_rels);
	uint32_t	read_pos = WARP_READ_POS(wp,n_rels);
	uint32_t	count;
	uint32_t	mask;
	uint32_t	row_id;
	uint32_t	offset;
	int			tupsz = 0;
	int			total_sz = 0;
	bool		try_suspend = false;
	union {
		struct {
			uint32_t	nitems;
			uint32_t	usage;
		} i;
		uint64_t		v64;
	} oldval, curval, newval;

	/*
	 * The previous depth still may produce new tuples, and number of
	 * the current result tuples is not sufficient to run projection.
	 */
	if (wp->scan_done <= n_rels && read_pos + warpSize > write_pos)
		return n_rels;

	read_pos += LaneId();
	if (read_pos < write_pos)
	{
		kcxt->kvecs_curr_id = (read_pos % KVEC_UNITSZ);
		kcxt->kvecs_curr_buffer = src_kvecs_buffer;

		tupsz = kern_estimate_heaptuple(kcxt,
										kexp_projection,
										kds_dst);
		if (tupsz < 0)
			STROM_ELOG(kcxt, "unable to compute tuple size");
	}
	/* error checks */
	if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
		return -1;
	/* allocation of the destination buffer */
	assert(kds_dst->format == KDS_FORMAT_ROW);
	mask = __ballot_sync(__activemask(), tupsz > 0);
	count = __popc(mask);
	mask &= ((1U << LaneId()) - 1);
	row_id = __popc(mask);
	assert(tupsz == 0 || row_id < count);

	offset = __reduce_stair_add_sync(tupsz, &total_sz);
	if (LaneId() == 0)
	{
		curval.i.nitems = kds_dst->nitems;
		curval.i.usage  = kds_dst->usage;
		do {
			newval = oldval = curval;
			newval.i.nitems += count;
			newval.i.usage  += __kds_packed(total_sz);

			if (KDS_HEAD_LENGTH(kds_dst) +
				MAXALIGN(sizeof(uint32_t) * newval.i.nitems) +
				__kds_unpack(newval.i.usage) > kds_dst->length)
			{
				try_suspend = true;
				break;
			}
		} while ((curval.v64 = atomicCAS((unsigned long long *)&kds_dst->nitems,
										 oldval.v64,
										 newval.v64)) != oldval.v64);
	}
	oldval.v64 = __shfl_sync(__activemask(), oldval.v64, 0);
	row_id += oldval.i.nitems;
	/* data store has no space? */
	if (__any_sync(__activemask(), try_suspend))
	{
		*p_try_suspend = true;
		return -1;
	}
	/* write out the tuple */
	if (tupsz > 0)
	{
		kern_tupitem   *tupitem;

		offset += __kds_unpack(oldval.i.usage);
		KDS_GET_ROWINDEX(kds_dst)[row_id] = __kds_packed(offset);
		tupitem = (kern_tupitem *)
			((char *)kds_dst + kds_dst->length - offset);
		tupitem->rowid = row_id;
		tupitem->t_len = kern_form_heaptuple(kcxt,
											 kexp_projection,
											 kds_dst,
											 &tupitem->htup);
	}
	/* update the read position */
	if (LaneId() == 0)
	{
		WARP_READ_POS(wp,n_rels) += count;
		assert(WARP_WRITE_POS(wp,n_rels) >= WARP_READ_POS(wp,n_rels));
	}
	__syncwarp();
	if (wp->scan_done <= n_rels)
	{
		if (WARP_WRITE_POS(wp,n_rels) < WARP_READ_POS(wp,n_rels) + warpSize)
			return n_rels;	/* back to the previous depth */
	}
	else
	{
		if (WARP_READ_POS(wp,n_rels) >= WARP_WRITE_POS(wp,n_rels))
			return -1;		/* ok, end of GpuJoin */
	}
	return n_rels + 1;		/* elsewhere, try again? */
}

/*
 * kern_gpujoin_main
 */
KERNEL_FUNCTION(void)
kern_gpujoin_main(kern_session_info *session,
				  kern_gputask *kgtask,
				  kern_multirels *kmrels,
				  kern_data_store *kds_src,
				  kern_data_extra *kds_extra,
				  kern_data_store *kds_dst)
{
	kern_context	   *kcxt;
	kern_warp_context  *wp, *wp_saved;
	char			   *kvec_buffer_base;
	uint32_t			kvec_buffer_size;
	uint32_t		   *l_state;
	bool			   *matched;
	uint32_t			wp_base_sz;
	uint32_t			n_rels = (kmrels ? kmrels->num_rels : 0);
	int					depth;
	__shared__ uint32_t smx_row_count;

	assert(kgtask->kvars_nslots == session->kcxt_kvars_nslots &&
		   kgtask->kvecs_bufsz  == session->kcxt_kvecs_bufsz &&
		   kgtask->kvecs_ndims  >= n_rels &&
		   kgtask->n_rels       == n_rels);
	/* setup execution context */
	INIT_KERNEL_CONTEXT(kcxt, session);
	wp_base_sz = __KERN_WARP_CONTEXT_BASESZ(kgtask->kvars_ndims);
	wp = (kern_warp_context *)SHARED_WORKMEM(wp_base_sz, get_local_id() / warpSize);
	INIT_KERN_GPUTASK_SUBFIELDS(kgtask,
								&wp_saved,
								&l_state,
								&matched);
	kvec_buffer_base = (char *)wp_saved + wp_base_sz;
	kvec_buffer_size = TYPEALIGN(CUDA_L1_CACHELINE_SZ, kcxt->kvecs_bufsz);
#define __KVEC_BUFFER(__depth)							\
	(kvec_buffer_base + kvec_buffer_size * (__depth))

	if (kgtask->resume_context)
	{
		/* resume the warp-context from the previous execution */
		if (LaneId() == 0)
			memcpy(wp, wp_saved, wp_base_sz);
		if (get_local_id() == 0)
			smx_row_count = wp->smx_row_count;
		depth = __shfl_sync(__activemask(), wp->depth, 0);
	}
	else
	{
		/* zero clear the wp */
		if (LaneId() == 0)
			memset(wp, 0, wp_base_sz);
		if (get_local_id() == 0)
			smx_row_count = 0;
		for (depth=0; depth < kgtask->kvars_ndims; depth++)
		{
			l_state[depth * get_global_size() + get_global_id()] = 0;
			matched[depth * get_global_size() + get_global_id()] = false;
		}
	}
	__syncthreads();
#define __L_STATE(__depth)						\
	l_state[get_global_size() * ((__depth)-1) + get_global_id()]
#define __MATCHED(__depth)						\
	matched[get_global_size() * ((__depth)-1) + get_global_id()]
	
	/* main logic of GpuJoin */
	while (depth >= 0)
	{
		kcxt_reset(kcxt);
		if (depth == 0)
		{
			/* LOAD FROM THE SOURCE */
			depth = execGpuScanLoadSource(kcxt, wp,
										  kds_src,
										  kds_extra,
										  SESSION_KEXP_LOAD_VARS(session, 0),
										  SESSION_KEXP_SCAN_QUALS(session),
										  __KVEC_BUFFER(0),
										  &smx_row_count);
		}
		else if (depth > n_rels)
		{
			bool	try_suspend = false;

			assert(depth == n_rels+1);
			if (session->xpucode_projection)
			{
				/* PROJECTION */
				depth = execGpuJoinProjection(kcxt, wp,
											  n_rels,
											  kds_dst,
											  SESSION_KEXP_PROJECTION(session),
											  __KVEC_BUFFER(n_rels),
											  &try_suspend);
			}
			else
			{
				/* PRE-AGG */
				depth = execGpuPreAggGroupBy(kcxt, wp,
											 n_rels,
											 kds_dst,
											 __KVEC_BUFFER(n_rels),
											 &try_suspend);
			}
			if (__any_sync(__activemask(), try_suspend))
			{
				if (LaneId() == 0)
					atomicAdd(&kgtask->suspend_count, 1);
				assert(depth < 0);
			}
		}
		else if (kmrels->chunks[depth-1].is_nestloop)
		{
			/* NEST-LOOP */
			depth = execGpuJoinNestLoop(kcxt, wp,
										kmrels,
										depth,
										__KVEC_BUFFER(depth-1),
										__KVEC_BUFFER(depth),
										__L_STATE(depth),	/* call by reference */
										__MATCHED(depth));	/* call by reference */
		}
		else if (kmrels->chunks[depth-1].gist_offset != 0)
		{
			/* GiST-INDEX-JOIN */
			const kern_expression *kexp_gist
				= SESSION_KEXP_GIST_EVALS(kcxt->session, depth);
			uint32_t		gist_depth;

			assert(kexp_gist != NULL &&
				   kexp_gist->opcode == FuncOpCode__GiSTEval &&
				   kexp_gist->u.gist.gist_depth < kgtask->kvars_ndims);
			gist_depth = kexp_gist->u.gist.gist_depth;

			depth = execGpuJoinGiSTJoin(kcxt, wp,
										kmrels,
										depth,
										__KVEC_BUFFER(depth-1),
										__KVEC_BUFFER(depth),
										kexp_gist,
										__KVEC_BUFFER(gist_depth),
										__L_STATE(depth),	/* call by reference */
										__MATCHED(depth));	/* call by reference */
		}
		else
		{
			/* HASH-JOIN */
			depth = execGpuJoinHashJoin(kcxt, wp,
										kmrels,
										depth,
										__KVEC_BUFFER(depth-1),
										__KVEC_BUFFER(depth),
										__L_STATE(depth),	/* call by reference */
										__MATCHED(depth));	/* call by reference */
		}
		assert(__shfl_sync(__activemask(), depth, 0) == depth);
		/* bailout if any error status */
		if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
			break;
	}
	__syncthreads();
#undef __KVEC_BUFFER
#undef __L_STATE
#undef __MATCHED

	if (LaneId() == 0)
	{
		/* update the statistics */
		if (depth < 0 && WARP_READ_POS(wp,n_rels) >= WARP_WRITE_POS(wp,n_rels))
		{
			/* number of raw-tuples fetched from the heap block */
			if (kds_src->format == KDS_FORMAT_BLOCK)
				atomicAdd(&kgtask->nitems_raw, wp->lp_wr_pos);
			else if (get_global_id() == 0)
				atomicAdd(&kgtask->nitems_raw, kds_src->nitems);
			atomicAdd(&kgtask->nitems_in, WARP_WRITE_POS(wp, 0));
			for (int i=0; i < n_rels; i++)
			{
				const kern_expression *kexp_gist
					= SESSION_KEXP_GIST_EVALS(session, i+1);
				if (kexp_gist)
				{
					int		gist_depth = kexp_gist->u.gist.gist_depth;

					assert(gist_depth > n_rels &&
						   gist_depth < kgtask->kvars_ndims);
					atomicAdd(&kgtask->stats[i].nitems_gist,
							  WARP_WRITE_POS(wp, gist_depth));
				}
				atomicAdd(&kgtask->stats[i].nitems_out,
						  WARP_WRITE_POS(wp,i+1));
			}
			atomicAdd(&kgtask->nitems_out, WARP_WRITE_POS(wp, n_rels));
		}
		/* suspend the execution context */
		wp->depth = depth;
		wp->smx_row_count = smx_row_count;
		memcpy(wp_saved, wp, wp_base_sz);
	}
	STROM_WRITEBACK_ERROR_STATUS(&kgtask->kerror, kcxt);
}
