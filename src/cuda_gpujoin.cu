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
 * Simple GPU-Sort + LIMIT clause
 */
KERNEL_FUNCTION(void)
kern_buffer_simple_limit(kern_data_store *kds_final, uint64_t old_length)
{
	uint64_t   *row_index = KDS_GET_ROWINDEX(kds_final);
	uint32_t	base;
	__shared__ uint64_t base_usage;

	assert(kds_final->format == KDS_FORMAT_ROW ||
		   kds_final->format == KDS_FORMAT_HASH);
	for (base = get_global_base();
		 base < kds_final->nitems;
		 base += get_global_size())
	{
		const kern_tupitem *titem = NULL;
		uint32_t	index = base + get_local_id();
		uint32_t	tupsz = 0;
		uint64_t	offset;
		uint64_t	total_sz;

		if (index < kds_final->nitems)
		{
			// XXX - must not use KDS_GET_TUPITEM() because kds_final->length
			//       is already truncated.
			assert(row_index[index] != 0);
			titem = (const kern_tupitem *)
				((char *)kds_final + old_length - row_index[index]);
			tupsz = MAXALIGN(offsetof(kern_tupitem, htup) + titem->t_len);
		}
		/* allocation of the destination buffer */
		offset = pgstrom_stair_sum_uint64(tupsz, &total_sz);
		if (get_local_id() == 0)
			base_usage = __atomic_add_uint64(&kds_final->usage,  total_sz);
		__syncthreads();
		/* put tuples on the destination */
		offset += base_usage;
		if (tupsz > 0)
		{
			kern_tupitem   *__titem = (kern_tupitem *)
				((char *)kds_final + kds_final->length - offset);
			__titem->rowid = index;
			__titem->t_len = titem->t_len;
			memcpy(&__titem->htup, &titem->htup, titem->t_len);
			assert(offsetof(kern_tupitem, htup) + titem->t_len <= tupsz);
			__threadfence();
			row_index[index] = ((char *)kds_final
								+ kds_final->length
								- (char *)__titem);
		}
		__syncthreads();
	}
}

/*
 * GPU Buffer consolidation (ROW-format)
 */
KERNEL_FUNCTION(void)
kern_buffer_consolidation(kern_data_store *kds_dst,
						  const kern_data_store *__restrict__ kds_src)
{
	__shared__ uint32_t base_rowid;
	__shared__ uint64_t base_usage;
	uint32_t	base;

	assert(kds_dst->format == KDS_FORMAT_ROW &&
		   kds_src->format == KDS_FORMAT_ROW);
	for (base = get_global_base();
		 base < kds_src->nitems;
		 base += get_global_size())
	{
		const kern_tupitem *titem = NULL;
		uint32_t	index = base + get_local_id();
		uint32_t	tupsz = 0;
		uint32_t	row_id;
		uint32_t	count;
		uint64_t	offset;
		uint64_t	total_sz;

		if (index < kds_src->nitems)
		{
			titem = KDS_GET_TUPITEM(kds_src, index);
			tupsz = MAXALIGN(offsetof(kern_tupitem, htup) + titem->t_len);
		}
		/* allocation of the destination buffer */
		row_id = pgstrom_stair_sum_binary(tupsz > 0, &count);
		offset = pgstrom_stair_sum_uint64(tupsz, &total_sz);
		if (get_local_id() == 0)
		{
			base_rowid = __atomic_add_uint32(&kds_dst->nitems, count);
			base_usage = __atomic_add_uint64(&kds_dst->usage,  total_sz);
		}
		__syncthreads();
		/* put tuples on the destination */
		row_id += base_rowid;
		offset += base_usage;
		if (tupsz > 0)
		{
			kern_tupitem   *__titem = (kern_tupitem *)
				((char *)kds_dst + kds_dst->length - offset);
			__titem->rowid = row_id;
			__titem->t_len = titem->t_len;
			memcpy(&__titem->htup, &titem->htup, titem->t_len);
			assert(offsetof(kern_tupitem, htup) + titem->t_len <= tupsz);
			__threadfence();
			KDS_GET_ROWINDEX(kds_dst)[row_id] = ((char *)kds_dst
												 + kds_dst->length
												 - (char *)__titem);
		}
		__syncthreads();
	}
}

/*
 * GPU Buffer reconstruction (HASH-format)
 */
KERNEL_FUNCTION(void)
kern_buffer_reconstruction(kern_data_store *kds_dst,
						   const kern_data_store *__restrict__ kds_src)
{
	uint64_t   *rowindex = KDS_GET_ROWINDEX(kds_src);
	uint32_t	base;
	__shared__ uint32_t base_rowid;
	__shared__ uint64_t base_usage;

	assert(kds_dst->format == KDS_FORMAT_HASH &&
		   kds_src->format == KDS_FORMAT_HASH);
	for (base = get_global_base();
		 base < kds_src->nitems;
		 base += get_global_size())
	{
		const kern_hashitem *hitem = NULL;
		uint32_t	index = base + get_local_id();
		uint32_t	tupsz = 0;
		uint32_t	row_id;
		uint32_t	count;
		uint64_t	offset;
		uint64_t	total_sz;

		if (index < kds_src->nitems)
		{
			hitem = (kern_hashitem *)((char *)kds_src + kds_src->length
									  - rowindex[index]
									  - offsetof(kern_hashitem, t));
			tupsz = MAXALIGN(offsetof(kern_hashitem,
									  t.htup) + hitem->t.t_len);
		}
		/* allocation of the destination buffer */
		row_id = pgstrom_stair_sum_binary(tupsz > 0, &count);
		offset = pgstrom_stair_sum_uint64(tupsz, &total_sz);
		if (get_local_id() == 0)
		{
			base_rowid = __atomic_add_uint32(&kds_dst->nitems, count);
			base_usage = __atomic_add_uint64(&kds_dst->usage,  total_sz);
		}
		__syncthreads();
		/* put tuples on the destination */
		row_id += base_rowid;
		offset += base_usage;
		if (tupsz > 0)
		{
			kern_hashitem  *__hitem = (kern_hashitem *)
				((char *)kds_dst + kds_dst->length - offset);
			uint64_t	   *__hslots = KDS_GET_HASHSLOT(kds_dst, hitem->hash);

			__hitem->hash = hitem->hash;
			__hitem->next = __atomic_exchange_uint64(__hslots, offset);
			__hitem->t.rowid = row_id;
			__hitem->t.t_len = hitem->t.t_len;
			memcpy(&__hitem->t.htup, &hitem->t.htup, hitem->t.t_len);
			assert(offsetof(kern_hashitem, t.htup) + hitem->t.t_len <= tupsz);
			__threadfence();
			KDS_GET_ROWINDEX(kds_dst)[row_id] = ((char *)kds_dst
												 + kds_dst->length
												 - (char *)&__hitem->t);
		}
		__syncthreads();
	}
}

/*
 * GPU Buffer Partitioning
 */
KERNEL_FUNCTION(void)
kern_buffer_partitioning(kern_buffer_partitions *kbuf_parts,
						 const kern_data_store *__restrict__ kds_src)
{
	uint64_t   *rowindex = KDS_GET_ROWINDEX(kds_src);
	uint32_t	hash_divisor = kbuf_parts->hash_divisor;
	uint64_t   *kpart_usage = (uint64_t *)SHARED_WORKMEM(0);
	uint32_t   *kpart_nitems = (uint32_t *)(kpart_usage + hash_divisor);
	uint32_t	base;

	assert(hash_divisor >= 2);
	for (base = get_global_base();
		 base < kds_src->nitems;
		 base += get_global_size())
	{
		const kern_hashitem *hitem = NULL;
		kern_data_store *kds_in = NULL;
		uint32_t	index = base + get_local_id();
		uint32_t	tupsz = 0;
		int			part_id;
		uint32_t	row_id;
		uint64_t	offset;

		/* reset buffer */
		for (int p=get_local_id(); p < hash_divisor; p += get_local_size())
		{
			kpart_usage[p] = 0;
			kpart_nitems[p] = 0;
		}
		__syncthreads();

		/* fetch row from kds_src */
		if (index < kds_src->nitems)
		{
			hitem = (kern_hashitem *)((char *)kds_src + kds_src->length
									  - rowindex[index]
									  - offsetof(kern_hashitem, t));
			tupsz   = MAXALIGN(offsetof(kern_hashitem,
										t.htup) + hitem->t.t_len);
			part_id = hitem->hash % hash_divisor;
			row_id = __atomic_add_uint32(&kpart_nitems[part_id], 1);
			offset = __atomic_add_uint64(&kpart_usage[part_id], tupsz);
		}
		__syncthreads();

		/* allocation of the destination buffer */
		if (get_local_id() < hash_divisor)
		{
			int		p = get_local_id();

			kds_in = kbuf_parts->parts[p].kds_in;
			kpart_nitems[p] = __atomic_add_uint32(&kds_in->nitems, kpart_nitems[p]);
			kpart_usage[p]  = __atomic_add_uint64(&kds_in->usage,  kpart_usage[p]);
		}
		__syncthreads();

		/* write out to the position */
		if (hitem)
		{
			kern_hashitem  *__hitem;
			uint64_t	   *__hslots;

			assert(part_id >= 0 && part_id < hash_divisor);
			kds_in = kbuf_parts->parts[part_id].kds_in;
			row_id += kpart_nitems[part_id];
			offset += kpart_usage[part_id] + tupsz;

			__hslots = KDS_GET_HASHSLOT(kds_in, hitem->hash);
			__hitem = (kern_hashitem *)
				((char *)kds_in + kds_in->length - offset);
			__hitem->hash = hitem->hash;
			__hitem->next = __atomic_exchange_uint64(__hslots, offset);
			__hitem->t.rowid = row_id;
			__hitem->t.t_len = hitem->t.t_len;
			memcpy(&__hitem->t.htup, &hitem->t.htup, hitem->t.t_len);
			assert(offsetof(kern_hashitem, t.htup) + hitem->t.t_len <= tupsz);
			__threadfence();
			KDS_GET_ROWINDEX(kds_in)[row_id] = ((char *)kds_in
												+ kds_in->length
												- (char *)&__hitem->t);
		}
		__syncthreads();
	}
}

/*
 * GPU Nested-Loop
 */
STATIC_FUNCTION(int)
execGpuJoinNestLoop(kern_context *kcxt,
					kern_warp_context *wp,
					kern_multirels *kmrels,
					int			depth,
					char	   *src_kvecs_buffer,
					char	   *dst_kvecs_buffer,
					uint64_t   &l_state,
					bool	   &matched)
{
	const kern_expression *kexp;
	kern_data_store *kds_heap = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	bool	   *oj_map = KERN_MULTIRELS_GPU_OUTER_JOIN_MAP(kmrels, depth,
														   stromTaskProp__cuda_dindex);
	uint32_t	rd_pos;
	uint32_t	wr_pos;
	uint32_t	count;
	bool		left_outer = kmrels->chunks[depth-1].left_outer;
	bool		tuple_is_valid = false;

	if (WARP_WRITE_POS(wp,depth) >= WARP_READ_POS(wp,depth) + get_local_size())
	{
		/*
		 * The destination depth already keeps warpSize or more pending
		 * tuple. So, flush out these tuples first.
		 */
		return depth+1;
	}

	if (__syncthreads_count(l_state < kds_heap->nitems) == 0 &&
		(!left_outer || __syncthreads_count(l_state != ULONG_MAX) == 0))
	{
		/*
		 * OK, all the threads in this block reached to end of the inner
		 * heap chain. Due to the above checks, the next depth has enough
		 * space to store the result in this depth.
		 */
		if (get_local_id() == 0)
			WARP_READ_POS(wp,depth-1) = Min(WARP_READ_POS(wp,depth-1) + get_local_size(),
											WARP_WRITE_POS(wp,depth-1));
		__syncthreads();
		l_state = 0;
		matched = false;
		if (wp->scan_done >= depth)
		{
			assert(wp->scan_done == depth);
			if (WARP_READ_POS(wp,depth-1) >= WARP_WRITE_POS(wp,depth-1))
			{
				if (get_local_id() == 0)
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
			if (WARP_READ_POS(wp,depth-1) + get_local_size() > WARP_WRITE_POS(wp,depth-1))
				return depth-1;
		}
	}

	rd_pos = WARP_READ_POS(wp,depth-1) + get_local_id();
	kcxt->kvecs_curr_id = (rd_pos % KVEC_UNITSZ);
	kcxt->kvecs_curr_buffer = src_kvecs_buffer;
	if (rd_pos < WARP_WRITE_POS(wp,depth-1))
	{
		uint32_t	index = l_state++;

		if (index < kds_heap->nitems)
		{
			kern_tupitem *tupitem;
			uint64_t	offset = KDS_GET_ROWINDEX(kds_heap)[index];
			int			status;

			tupitem = (kern_tupitem *)((char *)kds_heap
									   + kds_heap->length
									   - offset);
			kexp = SESSION_KEXP_LOAD_VARS(kcxt->session, depth);
			ExecLoadVarsHeapTuple(kcxt, kexp, depth, kds_heap, &tupitem->htup);
			kexp = SESSION_KEXP_JOIN_QUALS(kcxt->session, depth);
			if (ExecGpuJoinQuals(kcxt, kexp, &status))
			{
				if (status > 0)
					tuple_is_valid = true;
				if (status != 0)
				{
					assert(tupitem->rowid < kds_heap->nitems);
					if (oj_map)
						oj_map[tupitem->rowid] = true;
					matched = true;
				}
			}
			else
			{
				HandleErrorIfCpuFallback(kcxt, depth, index, matched);
			}
		}
		else if (left_outer && index >= kds_heap->nitems && !matched)
		{
			bool		status;
			/* fill up NULL fields, if FULL/LEFT OUTER JOIN */
			kexp = SESSION_KEXP_LOAD_VARS(kcxt->session, depth);
			ExecLoadVarsHeapTuple(kcxt, kexp, depth, kds_heap, NULL);
			kexp = SESSION_KEXP_JOIN_QUALS(kcxt->session, depth);
			if (ExecGpuJoinOtherQuals(kcxt, kexp, &status))
			{
				tuple_is_valid = status;
			}
			else
			{
				HandleErrorIfCpuFallback(kcxt, depth, index, matched);
			}
			l_state = ULONG_MAX;
		}
		else
		{
			l_state = ULONG_MAX;
		}
	}
	else
	{
		l_state = ULONG_MAX;
	}
	/* error checks */
	if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
		return -1;
	/* save the result */
	wr_pos = WARP_WRITE_POS(wp,depth);
	wr_pos += pgstrom_stair_sum_binary(tuple_is_valid, &count);
	if (get_local_id() == 0)
		WARP_WRITE_POS(wp,depth) += count;

	if (tuple_is_valid)
	{
		const kern_expression  *kexp_move
			= SESSION_KEXP_MOVE_VARS(kcxt->session, depth);
		if (!ExecMoveKernelVariables(kcxt,
									 kexp_move,
									 dst_kvecs_buffer,
									 (wr_pos % KVEC_UNITSZ)))
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
		}
	}
	/* error checks */
	if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
		return -1;
	if (WARP_WRITE_POS(wp,depth) >= WARP_READ_POS(wp,depth) + get_local_size())
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
					char	   *dst_kvecs_buffer,
					uint64_t   &l_state,
					bool	   &matched)
{
	kern_data_store *kds_hash = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	bool	   *oj_map = KERN_MULTIRELS_GPU_OUTER_JOIN_MAP(kmrels, depth,
														   stromTaskProp__cuda_dindex);
	kern_expression *kexp = NULL;
	kern_hashitem *khitem = NULL;
	uint32_t	rd_pos;
	uint32_t	wr_pos;
	uint32_t	count;
	bool		tuple_is_valid = false;

	if (WARP_WRITE_POS(wp,depth) >= WARP_READ_POS(wp,depth) + get_local_size())
	{
		/*
		 * Next depth already keeps blockSize or more pending tuples,
		 * so wipe out these tuples first.
		 */
		return depth+1;
	}

	if (__syncthreads_count(l_state != ULONG_MAX) == 0)
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
		if (get_local_id() == 0)
			WARP_READ_POS(wp,depth-1) = Min(WARP_READ_POS(wp,depth-1) + get_local_size(),
											WARP_WRITE_POS(wp,depth-1));
		__syncthreads();
		l_state = 0;
		matched = false;
		if (wp->scan_done < depth)
		{
			/*
			 * The previous depth still may generate the source tuple.
			 */
			if (WARP_WRITE_POS(wp,depth-1) < WARP_READ_POS(wp,depth-1) + get_local_size())
				return depth-1;
		}
		else
		{
			assert(wp->scan_done == depth);
			if (WARP_READ_POS(wp,depth-1) >= WARP_WRITE_POS(wp,depth-1))
			{
				if (get_local_id() == 0)
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
	wr_pos = WARP_WRITE_POS(wp,depth-1);
	rd_pos = WARP_READ_POS(wp,depth-1) + get_local_id();
	kcxt->kvecs_curr_id = (rd_pos % KVEC_UNITSZ);
	kcxt->kvecs_curr_buffer = src_kvecs_buffer;

	if (l_state == 0)
	{
		/* pick up the first item from the hash-slot */
		if (rd_pos < wr_pos)
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
			else if (HandleErrorIfCpuFallback(kcxt, depth, 0, false))
			{
				l_state = ULONG_MAX;
			}
		}
		else
		{
			l_state = ULONG_MAX;
		}
	}
	else if (l_state != ULONG_MAX)
	{
		/* pick up the next one if any */
		uint32_t	hash_value;

		khitem = (kern_hashitem *)((char *)kds_hash + l_state);
		assert(__KDS_TUPITEM_CHECK_VALID(kds_hash, &khitem->t));
		hash_value = khitem->hash;
		for (khitem = KDS_HASH_NEXT_ITEM(kds_hash, khitem->next);
			 khitem != NULL && khitem->hash != hash_value;
			 khitem = KDS_HASH_NEXT_ITEM(kds_hash, khitem->next));
	}
	/* error checks */
	if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
		return -1;

	if (khitem)
	{
		int		status;

		l_state = ((char *)khitem - (char *)kds_hash);
		kexp = SESSION_KEXP_LOAD_VARS(kcxt->session, depth);
		ExecLoadVarsHeapTuple(kcxt, kexp, depth, kds_hash, &khitem->t.htup);
		kexp = SESSION_KEXP_JOIN_QUALS(kcxt->session, depth);
		if (ExecGpuJoinQuals(kcxt, kexp, &status))
		{
			if (status > 0)
				tuple_is_valid = true;
			if (status != 0)
			{
				assert(khitem->t.rowid < kds_hash->nitems);
				if (oj_map)
					oj_map[khitem->t.rowid] = true;
				matched = true;
			}
		}
		else if (HandleErrorIfCpuFallback(kcxt, depth, l_state, matched))
		{
			l_state = ULONG_MAX;
		}
	}
	else
	{
		if (kmrels->chunks[depth-1].left_outer &&
			l_state != ULONG_MAX && !matched)
		{
			bool	status;
			/* load NULL values on the inner portion */
			kexp = SESSION_KEXP_LOAD_VARS(kcxt->session, depth);
			ExecLoadVarsHeapTuple(kcxt, kexp, depth, kds_hash, NULL);
			kexp = SESSION_KEXP_JOIN_QUALS(kcxt->session, depth);
			if (ExecGpuJoinOtherQuals(kcxt, kexp, &status))
			{
				tuple_is_valid = status;
			}
			else
			{
				HandleErrorIfCpuFallback(kcxt, depth, l_state, matched);
			}
		}
		l_state = ULONG_MAX;
	}
	/* error checks */
	if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
		return -1;
	/* save the result on the destination buffer */
	wr_pos = WARP_WRITE_POS(wp,depth);
	wr_pos += pgstrom_stair_sum_binary(tuple_is_valid, &count);
	if (get_local_id() == 0)
		WARP_WRITE_POS(wp,depth) += count;

	if (tuple_is_valid)
	{
		const kern_expression  *kexp_move
			= SESSION_KEXP_MOVE_VARS(kcxt->session, depth);
		if (!ExecMoveKernelVariables(kcxt,
									 kexp_move,
									 dst_kvecs_buffer,
									 (wr_pos % KVEC_UNITSZ)))
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
		}
	}
	/* error checks */
	if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
		return -1;
	if (WARP_WRITE_POS(wp,depth) >= WARP_READ_POS(wp,depth) + get_local_size())
		return depth+1;
	return depth;
}

/*
 * gpujoin_prep_gistindex
 */
KERNEL_FUNCTION(void)
gpujoin_prep_gistindex(kern_multirels *kmrels, int depth)
{
	kern_data_store *kds_hash = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	kern_data_store *kds_gist = KERN_MULTIRELS_GIST_INDEX(kmrels, depth);
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
		uint32_t		hash;

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
					uint32_t	rowid = khitem->t.rowid;

					itup->t_tid.ip_blkid.bi_hi = (rowid >> 16);
					itup->t_tid.ip_blkid.bi_lo = (rowid & 0xffffU);
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
					char	   *src_kvecs_buffer,
					char	   *dst_kvecs_buffer,
					const kern_expression *kexp_gist,
					char	   *gist_kvecs_buffer,
					uint64_t   &l_state,
					bool       &matched)
{
	kern_data_store *kds_hash = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	kern_data_store *kds_gist = KERN_MULTIRELS_GIST_INDEX(kmrels, depth);
	int				gist_depth = kexp_gist->u.gist.gist_depth;
	uint32_t		count;
	uint32_t		rd_pos;
	uint32_t		wr_pos;

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

	if (WARP_WRITE_POS(wp,depth) >= WARP_READ_POS(wp,depth) + get_local_size())
	{
		/*
		 * Next depth already have blockSize or more pending tuples,
		 * so wipe out these tuples first.
		 */
		return depth+1;
	}

	if (WARP_WRITE_POS(wp,gist_depth) >= (WARP_READ_POS(wp,gist_depth)
										  + get_local_size()) ||
		(wp->scan_done >= depth &&		/* is terminal case? */
		 WARP_WRITE_POS(wp,depth-1) == WARP_READ_POS(wp,depth-1) &&
		 __syncthreads_count(l_state != ULONG_MAX) == 0))
	{
		/*
		 * We already have blockSize or more pending tuples; they were
		 * fetched by the GiST-index. So, we try Join-quals for them.
		 */
		bool	join_is_valid = false;

		rd_pos = WARP_READ_POS(wp,gist_depth) + get_local_id();
		if (rd_pos < WARP_WRITE_POS(wp,gist_depth))
		{
			const kern_expression *kexp_load
				= SESSION_KEXP_LOAD_VARS(kcxt->session, depth);
			const kern_expression *kexp_join
				= SESSION_KEXP_JOIN_QUALS(kcxt->session, depth);

			kcxt->kvecs_curr_id = (rd_pos % KVEC_UNITSZ);
			kcxt->kvecs_curr_buffer = gist_kvecs_buffer;
			join_is_valid = ExecGiSTIndexPostQuals(kcxt, depth,
												   kds_hash,
												   kexp_gist,
												   kexp_load,
												   kexp_join);
		}
		/* error checks */
		if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
			return -1;
		if (get_local_id() == 0)
			WARP_READ_POS(wp,gist_depth) = Min(WARP_READ_POS(wp,gist_depth) + get_local_size(),
											   WARP_WRITE_POS(wp,gist_depth));
		wr_pos = WARP_WRITE_POS(wp,depth);
		wr_pos += pgstrom_stair_sum_binary(join_is_valid, &count);
		if (get_local_id() == 0)
			WARP_WRITE_POS(wp,depth) += count;

		if (join_is_valid)
		{
			const kern_expression *kexp_move
				= SESSION_KEXP_MOVE_VARS(kcxt->session, depth);
			if (!ExecMoveKernelVariables(kcxt,
										 kexp_move,
										 dst_kvecs_buffer,
										 (wr_pos % KVEC_UNITSZ)))
			{
				assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
			}
		}
		/* error checks */
		if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
			return -1;
		/* termination checks */
		if (wp->scan_done >= depth &&
			WARP_WRITE_POS(wp,depth-1) == WARP_READ_POS(wp,depth-1) &&
			WARP_WRITE_POS(wp,gist_depth) <= WARP_READ_POS(wp,gist_depth) &&
			__syncthreads_count(l_state != ULONG_MAX) == 0)
		{
			if (get_local_id() == 0)
				wp->scan_done++;
			depth++;
		}
		return depth;
	}

	if (__syncthreads_count(l_state != ULONG_MAX) == 0)
	{
		/*
		 * OK, all the threads in this block reached to the end of the GiST
		 * index tree. Due to the above checks, the next depth has enough
		 * space to store the result in this depth.
		 */
		if (get_local_id() == 0)
			WARP_READ_POS(wp,depth-1) = Min(WARP_READ_POS(wp,depth-1) + get_local_size(),
											WARP_WRITE_POS(wp,depth-1));
		__syncthreads();
		l_state = 0;
		matched = false;
		if (wp->scan_done < depth)
		{
			/* back to the previous depth; that still may generate source tuples */
			if (WARP_WRITE_POS(wp,depth-1) < WARP_READ_POS(wp,depth-1) + get_local_size())
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
	rd_pos = WARP_READ_POS(wp,depth-1) + get_local_id();
	if (rd_pos < WARP_WRITE_POS(wp,depth-1))
	{
		if (l_state != ULONG_MAX)
		{
			kcxt->kvecs_curr_buffer = src_kvecs_buffer;
			kcxt->kvecs_curr_id = (rd_pos % KVEC_UNITSZ);
			l_state = ExecGiSTIndexGetNext(kcxt,
										   kds_hash,
										   kds_gist,
										   kexp_gist,
										   l_state);
		}
	}
	else
	{
		l_state = ULONG_MAX;
	}
	/* error checks */
	if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
		return -1;
	/* save the result on the destination buffer */
	wr_pos = WARP_WRITE_POS(wp,gist_depth);
	wr_pos += pgstrom_stair_sum_binary(l_state != ULONG_MAX, &count);
	if (get_local_id() == 0)
		WARP_WRITE_POS(wp,gist_depth) += count;

	if (l_state != ULONG_MAX)
	{
		const kern_expression  *kexp_move
			= SESSION_KEXP_MOVE_VARS(kcxt->session, gist_depth);
		if (!ExecMoveKernelVariables(kcxt,
									 kexp_move,
									 gist_kvecs_buffer,
									 (wr_pos % KVEC_UNITSZ)))
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
		}
	}
	/* error checks */
	if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
		return -1;
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
					  char *src_kvecs_buffer)
{
	uint32_t	wr_pos = WARP_WRITE_POS(wp,n_rels);
	uint32_t	rd_pos = WARP_READ_POS(wp,n_rels);
	uint32_t	nr_input = Min(wr_pos - rd_pos, get_local_size());
	uint32_t	count;
	uint32_t	row_id;
	uint64_t	offset;
	int32_t		tupsz = 0;
	uint32_t	total_sz = 0;
	int32_t		hash_value = 0;
	__shared__ uint32_t	base_rowid;
	__shared__ uint64_t	base_usage;

	/*
	 * The previous depth still may produce new tuples, and number of
	 * the current result tuples is not sufficient to run projection.
	 */
	if (wp->scan_done <= n_rels && rd_pos + get_local_size() > wr_pos)
		return n_rels;
	rd_pos += get_local_id();

	kcxt->kvecs_curr_id = (rd_pos % KVEC_UNITSZ);
	kcxt->kvecs_curr_buffer = src_kvecs_buffer;
	if (rd_pos < wr_pos)
	{
		tupsz = kern_estimate_heaptuple(kcxt,
										kexp_projection,
										kds_dst);
		if (tupsz < 0)
		{
			if (HandleErrorIfCpuFallback(kcxt, n_rels+1, 0, false))
				tupsz = 0;
			else if (kcxt->errcode != ERRCODE_SUSPEND_FALLBACK)
				STROM_ELOG(kcxt, "unable to compute tuple size");
		}
		else if (kds_dst->format == KDS_FORMAT_ROW)
		{
			tupsz = MAXALIGN(offsetof(kern_tupitem, htup) + tupsz
							 + kcxt->session->gpusort_htup_margin);
		}
		else
		{
			kern_expression *kexp_hash = (kern_expression *)
				((char *)kexp_projection + kexp_projection->u.proj.hash);
			xpu_int4_t	status;

			/*
			 * Calculation of Hash-value if destination buffer needs to
			 * set up hash-table. Usually, when GpuScan results are
			 * reused as an inner buffer of GpuJoin.
			 */
			assert(kds_dst->format == KDS_FORMAT_HASH &&
				   __KEXP_IS_VALID(kexp_projection, kexp_hash));
			if (EXEC_KERN_EXPRESSION(kcxt, kexp_hash, &status))
			{
				if (XPU_DATUM_ISNULL(&status))
					STROM_ELOG(kcxt, "unable to compute hash-value");
				else
					hash_value = status.value;
				tupsz = MAXALIGN(offsetof(kern_hashitem, t.htup) + tupsz
								 + kcxt->session->gpusort_htup_margin);
			}
			else if (HandleErrorIfCpuFallback(kcxt, n_rels+1, 0, false))
			{
				tupsz = 0;
			}
			else
			{
				assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
			}
		}
	}
	/* error checks */
	if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
		return -1;
	/* allocation of the destination buffer */
	row_id = pgstrom_stair_sum_binary(tupsz > 0, &count);
	offset = pgstrom_stair_sum_uint32(tupsz, &total_sz);
	for (;;)
	{
		uint32_t	__nitems;
		uint64_t	__usage;
		bool		try_suspend = false;
		bool		allocation_ok = false;

		if (get_local_id() == 0)
		{
			__nitems = __volatileRead(&kds_dst->nitems);
			if (__nitems != UINT_MAX &&
				__nitems == __atomic_cas_uint32(&kds_dst->nitems,
												__nitems,
												UINT_MAX))		/* LOCK */
			{
				__usage = __volatileRead(&kds_dst->usage);
				if (__KDS_CHECK_OVERFLOW(kds_dst,
										 __nitems + count,
										 __usage + total_sz))
				{
					base_rowid = __nitems;
					base_usage = __usage;
					allocation_ok = true;

					__atomic_add_uint64(&kds_dst->usage, total_sz);
					__nitems += count;
				}
				else
				{
					try_suspend = true;
				}
				__atomic_write_uint32(&kds_dst->nitems, __nitems);	/* UNLOCK */
			}
		}
		if (__syncthreads_count(try_suspend) > 0)
		{
			SUSPEND_NO_SPACE(kcxt, "GpuProjection - no space to write");
			return -1;
		}
		if (__syncthreads_count(allocation_ok) > 0)
			break;
	}
	/* write out the tuple */
	if (tupsz > 0)
	{
		row_id += base_rowid;
		offset += base_usage;

		if (kds_dst->format == KDS_FORMAT_ROW)
		{
			kern_tupitem   *tupitem = (kern_tupitem *)
				((char *)kds_dst + kds_dst->length - offset);

			tupitem->rowid = row_id;
			tupitem->t_len = kern_form_heaptuple(kcxt,
												 kexp_projection,
												 kds_dst,
												 &tupitem->htup);
			assert(offsetof(kern_tupitem, htup) + tupitem->t_len <= tupsz);
			__threadfence();
			KDS_GET_ROWINDEX(kds_dst)[row_id] = offset;
		}
		else
		{
			kern_hashitem  *khitem = (kern_hashitem *)
				((char *)kds_dst + kds_dst->length - offset);
			uint64_t	   *hslots = KDS_GET_HASHSLOT(kds_dst, hash_value);

			khitem->hash = hash_value;
			khitem->next = __atomic_exchange_uint64(hslots, offset);
			khitem->t.rowid = row_id;
			khitem->t.t_len = kern_form_heaptuple(kcxt,
												  kexp_projection,
												  kds_dst,
												  &khitem->t.htup);
			assert(offsetof(kern_hashitem, t.htup) + khitem->t.t_len <= tupsz);
			__threadfence();
			KDS_GET_ROWINDEX(kds_dst)[row_id] = ((char *)kds_dst
												 + kds_dst->length
												 - (char *)&khitem->t);
		}
	}
	/* update the read position */
	if (get_local_id() == 0)
	{
		WARP_READ_POS(wp,n_rels) += nr_input;
		assert(WARP_WRITE_POS(wp,n_rels) >= WARP_READ_POS(wp,n_rels));
	}
	__syncthreads();
	if (wp->scan_done <= n_rels)
	{
		if (WARP_WRITE_POS(wp,n_rels) < WARP_READ_POS(wp,n_rels) + get_local_size())
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
 * Load RIGHT OUTER values
 */
PUBLIC_FUNCTION(int)
loadGpuJoinRightOuter(kern_context *kcxt,
					  kern_warp_context *wp,
					  kern_multirels *kmrels,
					  int depth,
					  char *dst_kvecs_buffer)
{
	kern_data_store *kds_in = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	bool	   *oj_map = KERN_MULTIRELS_GPU_OUTER_JOIN_MAP(kmrels, depth,
														   stromTaskProp__cuda_dindex);
	uint32_t	count;
	uint32_t	index;
	uint32_t	wr_pos;
	kern_tupitem *tupitem = NULL;

	if (WARP_WRITE_POS(wp,depth) >= WARP_READ_POS(wp,depth) + get_local_size())
	{
		/*
		 * Current depth already keeps blockSize or more pending tuples,
		 * so wipe out these tuples first.
		 */
		return depth+1;
	}
	/* fetch the next row-index from the kds_in */
	index = get_global_size() * wp->smx_row_count + get_global_base();
	if (index >= kds_in->nitems)
	{
		if (get_local_id() == 0)
			wp->scan_done = depth+1;
		__syncthreads();
		return depth+1;
	}
	index += get_local_id();

	/*
	 * fetch the inner tuple that has not matched any outer tuples
	 */
	assert(oj_map != NULL);
	if (index < kds_in->nitems && !oj_map[index])
	{
		assert(kds_in->format == KDS_FORMAT_ROW ||
			   kds_in->format == KDS_FORMAT_HASH);
		tupitem = KDS_GET_TUPITEM(kds_in, index);
	}

	/*
	 * load the inner tuple and fill up other outer slots by NULLs
	 */
	wr_pos = WARP_WRITE_POS(wp,depth);
	wr_pos += pgstrom_stair_sum_binary(tupitem != NULL, &count);
	if (tupitem != NULL)
	{
		const kern_expression *kexp;

		for (int __depth=0; __depth < depth; __depth++)
		{
			kexp = SESSION_KEXP_LOAD_VARS(kcxt->session, __depth);
			ExecLoadVarsHeapTuple(kcxt, kexp, __depth, kds_in, NULL);
		}
		kexp = SESSION_KEXP_LOAD_VARS(kcxt->session, depth);
		ExecLoadVarsHeapTuple(kcxt, kexp, depth, kds_in, &tupitem->htup);

		kexp = SESSION_KEXP_MOVE_VARS(kcxt->session, depth);
		if (!ExecMoveKernelVariables(kcxt,
									 kexp,
									 dst_kvecs_buffer,
									 (wr_pos % KVEC_UNITSZ)))
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
		}
	}
	/* error checks */
	if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
		return -1;
	/* make advance the position */
	if (get_local_id() == 0)
	{
		wp->smx_row_count++;
		WARP_WRITE_POS(wp,depth) += count;
	}
	__syncthreads();
	if (WARP_WRITE_POS(wp,depth) >= WARP_READ_POS(wp,depth) + get_local_size())
		return depth+1;
	return depth;
}

/*
 * GPU-Task specific read-only properties.
 */
PUBLIC_SHARED_DATA(uint32_t, stromTaskProp__cuda_dindex);
PUBLIC_SHARED_DATA(uint32_t, stromTaskProp__cuda_stack_limit);
PUBLIC_SHARED_DATA(int32_t,  stromTaskProp__partition_divisor);
PUBLIC_SHARED_DATA(int32_t,  stromTaskProp__partition_reminder);

/*
 * kern_gpujoin_main
 */
KERNEL_FUNCTION(void)
kern_gpujoin_main(kern_session_info *session,
				  kern_gputask *kgtask,
				  kern_multirels *kmrels,
				  kern_data_store *kds_src,
				  kern_data_extra *kds_extra,
				  kern_data_store *kds_dst,
				  kern_data_store *kds_fallback)
{
	kern_context	   *kcxt;
	kern_warp_context  *wp, *wp_saved;
	char			   *kvec_buffer_base;
	uint32_t			kvec_buffer_size;
	uint64_t		   *l_state;
	bool			   *matched;
	uint32_t			wp_base_sz;
	uint32_t			n_rels = (kmrels ? kmrels->num_rels : 0);
	int					depth;

	/* sanity checks */
	assert(kgtask->kvars_nslots == session->kcxt_kvars_nslots &&
		   kgtask->kvecs_bufsz  == session->kcxt_kvecs_bufsz &&
		   kgtask->kvecs_ndims  >= n_rels &&
		   kgtask->n_rels       == n_rels &&
		   get_local_size()     <= CUDA_MAXTHREADS_PER_BLOCK);
	assert(kgtask->right_outer_depth == 0 ? kds_src != NULL : kds_src == NULL);
	/* save the GPU-Task specific read-only properties */
	if (get_local_id() == 0)
	{
		stromTaskProp__cuda_dindex        = kgtask->cuda_dindex;
		stromTaskProp__cuda_stack_limit   = kgtask->cuda_stack_limit;
		stromTaskProp__partition_divisor  = kgtask->partition_divisor;
		stromTaskProp__partition_reminder = kgtask->partition_reminder;
	}
	/* setup execution context */
	INIT_KERNEL_CONTEXT(kcxt, session, kds_fallback);
	wp_base_sz = __KERN_WARP_CONTEXT_BASESZ(kgtask->kvecs_ndims);
	wp = (kern_warp_context *)SHARED_WORKMEM(0);
	INIT_KERN_GPUTASK_SUBFIELDS(kgtask,
								&wp_saved,
								&l_state,
								&matched);
	setupGpuPreAggGroupByBuffer(kcxt, kgtask, SHARED_WORKMEM(wp_base_sz));
	kvec_buffer_base = (char *)wp_saved + wp_base_sz;
	kvec_buffer_size = TYPEALIGN(CUDA_L1_CACHELINE_SZ, kcxt->kvecs_bufsz);
#define __KVEC_BUFFER(__depth)							\
	(kvec_buffer_base + kvec_buffer_size * (__depth))

	if (kgtask->resume_context)
	{
		/* resume the warp-context from the previous execution */
		if (get_local_id() == 0)
			memcpy(wp, wp_saved, wp_base_sz);
	}
	else
	{
		/* zero clear the wp */
		if (get_local_id() == 0)
		{
			memset(wp, 0, wp_base_sz);
			/* Set RIGHT-OUTER-JOIN special starting point */
			if (kgtask->right_outer_depth > 0)
				wp->depth = wp->scan_done = kgtask->right_outer_depth;
		}
		for (int d=0; d < kgtask->n_rels; d++)
		{
			l_state[d * get_global_size() + get_global_id()] = 0;
			matched[d * get_global_size() + get_global_id()] = false;
		}
	}
	__syncthreads();
	depth = wp->depth;

#define __L_STATE(__depth)											\
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
										  SESSION_KEXP_MOVE_VARS(session, 0),
										  __KVEC_BUFFER(0));
		}
		else if (depth > n_rels)
		{
			assert(depth == n_rels+1);
			if (session->xpucode_projection)
			{
				/* PROJECTION */
				depth = execGpuJoinProjection(kcxt, wp,
											  n_rels,
											  kds_dst,
											  SESSION_KEXP_PROJECTION(session),
											  __KVEC_BUFFER(n_rels));
			}
			else
			{
				/* PRE-AGG */
				depth = execGpuPreAggGroupBy(kcxt, wp,
											 n_rels,
											 kds_dst,
											 __KVEC_BUFFER(n_rels));
			}
		}
		else if (depth == kgtask->right_outer_depth)
		{
			/* Load RIGHT-OUTER Tuples */
			depth = loadGpuJoinRightOuter(kcxt, wp,
										  kmrels,
										  depth,
										  __KVEC_BUFFER(depth));
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
				   kexp_gist->u.gist.gist_depth < kgtask->kvecs_ndims);
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
		/* bailout if any error status */
		if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
			break;
		if (get_local_id() == 0)
			wp->depth = depth;
	}
	__syncthreads();

#undef __KVEC_BUFFER
#undef __L_STATE
#undef __MATCHED
	/* merge the group-by buffer to the kds_final, if any */
	mergeGpuPreAggGroupByBuffer(kcxt, kds_dst);

	/* update the statistics */
	if (get_local_id() == 0)
	{
		if (depth < 0 && WARP_READ_POS(wp,n_rels) >= WARP_WRITE_POS(wp,n_rels))
		{
			int		start = 0;

			if (kgtask->right_outer_depth == 0)
			{
				assert(kds_src != NULL);
				/* number of raw-tuples fetched from the source KDS */
				if (kds_src->format == KDS_FORMAT_BLOCK)
					atomicAdd(&kgtask->nitems_raw, wp->lp_wr_pos);
				else if (get_global_id() == 0)
					atomicAdd(&kgtask->nitems_raw, kds_src->nitems);
				atomicAdd(&kgtask->nitems_in, WARP_WRITE_POS(wp, 0));
			}
			else
			{
				assert(kds_src == NULL);
				/* number of the generated RIGHT-OUTER-JOIN tuples */
				start = kgtask->right_outer_depth;
				atomicAdd(&kgtask->stats[start-1].nitems_roj,
						  WARP_WRITE_POS(wp,start));
			}
			for (int i=start; i < n_rels; i++)
			{
				const kern_expression *kexp_gist
					= SESSION_KEXP_GIST_EVALS(session, i+1);
				if (kexp_gist)
				{
					int		gist_depth = kexp_gist->u.gist.gist_depth;

					assert(gist_depth > n_rels &&
						   gist_depth < kgtask->kvecs_ndims);
					atomicAdd(&kgtask->stats[i].nitems_gist,
							  WARP_WRITE_POS(wp, gist_depth));
				}
				atomicAdd(&kgtask->stats[i].nitems_out,
						  WARP_WRITE_POS(wp,i+1));
			}
			atomicAdd(&kgtask->nitems_out, WARP_WRITE_POS(wp, n_rels));
		}
		/* suspend the execution context */
		memcpy(wp_saved, wp, wp_base_sz);
	}
	STROM_WRITEBACK_ERROR_STATUS(&kgtask->kerror, kcxt);
}

/*
 * kern_gpujoin_main
 */
KERNEL_FUNCTION(void)
gpujoin_merge_outer_join_map(uint32_t *dst_ojmap,
							 const uint32_t *src_ojmap,
							 uint32_t nitems)
{
	uint32_t	index;

	for (index = get_global_id();
		 index < nitems;
		 index += get_global_size())
	{
		dst_ojmap[index] |= src_ojmap[index];
	}
}
