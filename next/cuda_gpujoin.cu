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
execGpuJoinNestLoop(kern_context      *kcxt,
					kern_warp_context *wp,
					kern_multirels    *kmrels,
					int                depth,
					void             **src_kvars_addr,
					int               *src_kvars_len,
					void             **dst_kvars_addr,
					int               *dst_kvars_len,
					uint32_t          &l_state,
					bool              &matched)
{
	kern_data_store *kds_heap = KERN_MULTIRELS_INNER_KDS(kmrels, depth-1);
	kern_expression *kexp;
	uint32_t	read_pos;
	uint32_t	write_pos;
	uint32_t	mask;
	bool		tuple_is_valid = false;

	if (WARP_WRITE_POS(wp,depth) >= WARP_READ_POS(wp,depth) + warpSize)
	{
		/*
		 * The destination depth already keeps warpSize or more pending
		 * tuple. So, flush out these tuples first.
		 */
		return depth+1;
	}

	if (__all_sync(__activemask(), l_state >= kds_heap->nitems))
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
					wp->scan_done = Max(wp->scan_done, depth+1);
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

		read_pos = (read_pos % GPU_KVARS_UNITSZ);
		kcxt->kvars_addr = src_kvars_addr + read_pos * kcxt->kvars_nslots;
		kcxt->kvars_len  = src_kvars_len  + read_pos * kcxt->kvars_nslots;
		if (index < kds_heap->nitems)
		{
			kern_tupitem *tupitem;
			uint32_t	offset = KDS_GET_ROWINDEX(kds_heap)[index];
			xpu_int4_t	status;

			tupitem = (kern_tupitem *)((char *)kds_heap +
									   kds_heap->length -
									   __kds_unpack(offset));
			kexp = SESSION_KEXP_JOIN_LOAD_VARS(kcxt->session, depth);
			ExecLoadVarsHeapTuple(kcxt, kexp, depth, kds_heap, &tupitem->htup);
			kexp = SESSION_KEXP_JOIN_QUALS(kcxt->session, depth);
			if (EXEC_KERN_EXPRESSION(kcxt, kexp, &status))
			{
				assert(!status.isnull);
				if (status.value > 0)
					tuple_is_valid = true;
				if (status.value != 0)
					matched = true;
			}
		}
		else if (kmrels->chunks[depth-1].left_outer &&
				 index >= kds_heap->nitems && !matched)
		{
			/* fill up NULL fields, if FULL/LEFT OUTER JOIN */
			kexp = SESSION_KEXP_JOIN_LOAD_VARS(kcxt->session, depth);
			ExecLoadVarsHeapTuple(kcxt, kexp, depth, kds_heap, NULL);
			tuple_is_valid = true;
		}
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
		write_pos = (write_pos % GPU_KVARS_UNITSZ);
		memcpy(dst_kvars_addr + write_pos * kcxt->kvars_nslots,
			   kcxt->kvars_addr,
			   sizeof(void *) * kcxt->kvars_nslots);
		memcpy(dst_kvars_len + write_pos * kcxt->kvars_nslots,
			   kcxt->kvars_len,
			   sizeof(int) * kcxt->kvars_nslots);
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
execGpuJoinHashJoin(kern_context      *kcxt,
					kern_warp_context *wp,
					kern_multirels    *kmrels,
					int                depth,
					void             **src_kvars_addr,
					int               *src_kvars_len,
					void             **dst_kvars_addr,
					int               *dst_kvars_len,
					uint32_t          &l_state,
					bool              &matched)
{
	kern_data_store *kds_hash = KERN_MULTIRELS_INNER_KDS(kmrels, depth-1);
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
		if (wp->scan_done >= depth)
		{
			assert(wp->scan_done == depth);
			if (WARP_READ_POS(wp,depth-1) >= WARP_WRITE_POS(wp,depth-1))
			{
				if (LaneId() == 0)
					wp->scan_done = Max(wp->scan_done, depth+1);
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
			/*
			 * Back to the previous depth to generate the source tuples.
			 */
			if (WARP_READ_POS(wp,depth-1) + warpSize > WARP_WRITE_POS(wp,depth-1))
				return depth-1;
		}
	}
	write_pos = WARP_WRITE_POS(wp,depth-1);
	read_pos = WARP_READ_POS(wp,depth-1) + LaneId();
	index = (read_pos % GPU_KVARS_UNITSZ);
	kcxt->kvars_addr = src_kvars_addr + index * kcxt->kvars_nslots;
	kcxt->kvars_len  = src_kvars_len  + index * kcxt->kvars_nslots;
	if (l_state == 0)
	{
		/* pick up the first item from the hash-slot */
		if (read_pos < write_pos)
		{
			xpu_int4_t	hash;

			kexp = SESSION_KEXP_HASH_VALUE(kcxt->session, depth);
			if (EXEC_KERN_EXPRESSION(kcxt, kexp, &hash))
			{
				assert(!hash.isnull);
				for (khitem = KDS_HASH_FIRST_ITEM(kds_hash, hash.value);
					 khitem != NULL && khitem->hash != hash.value;
					 khitem = KDS_HASH_NEXT_ITEM(kds_hash, khitem));
			}
		}
	}
	else if (l_state != UINT_MAX)
	{
		/* pick up the next one if any */
		uint32_t	hash_value;

		khitem = (kern_hashitem *)((char *)kds_hash
								   + __kds_unpack(l_state)
								   - offsetof(kern_hashitem, t.htup));
		hash_value = khitem->hash;
		while (khitem != NULL && khitem->hash != hash_value)
			khitem = KDS_HASH_NEXT_ITEM(kds_hash, khitem);
	}
	/* error checks */
	if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
		return -1;
	if (khitem)
	{
		xpu_int4_t	status;

		kexp = SESSION_KEXP_JOIN_LOAD_VARS(kcxt->session, depth);
		ExecLoadVarsHeapTuple(kcxt, kexp, depth, kds_hash, &khitem->t.htup);
		kexp = SESSION_KEXP_JOIN_QUALS(kcxt->session, depth);
		if (EXEC_KERN_EXPRESSION(kcxt, kexp, &status))
		{
			assert(!status.isnull);
			if (status.value > 0)
				tuple_is_valid = true;
			if (status.value != 0)
				matched = true;
		}
		l_state = __kds_packed((char *)&khitem->t.htup - (char *)kds_hash);
	}
	else
	{
		if (kmrels->chunks[depth-1].left_outer &&
			l_state != UINT_MAX && !matched)
		{
			/* load NULL values on the inner portion */
			 kexp = SESSION_KEXP_JOIN_LOAD_VARS(kcxt->session, depth);
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
		index = write_pos % UNIT_TUPLES_PER_WARP;
		dst_kvars_addr += index * kcxt->kvars_nslots;
		dst_kvars_len  += index * kcxt->kvars_nslots;
		memcpy(dst_kvars_addr, kcxt->kvars_addr, sizeof(void *) * kcxt->kvars_nslots);
		memcpy(dst_kvars_len,  kcxt->kvars_len,  sizeof(int) * kcxt->kvars_nslots);
	}
	__syncwarp();
	if (WARP_WRITE_POS(wp,depth) >= WARP_READ_POS(wp,depth) + warpSize)
		return depth+1;
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
					  void **kvars_addr,
					  int *kvars_len)
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
	}	oldval, curval, newval;

	/*
	 * The previous depth still may produce new tuples, and number of
	 * the current result tuples is not sufficient to run projection.
	 */
	if (wp->scan_done <= n_rels && read_pos + warpSize < write_pos)
		return n_rels;
	read_pos += LaneId();
	if (read_pos < write_pos)
	{
		xpu_int4_t	__tupsz;
		int			index = (read_pos % GPU_KVARS_UNITSZ);

		kcxt->kvars_addr = kvars_addr + kcxt->kvars_nslots * index;
		kcxt->kvars_len  = kvars_len  + kcxt->kvars_nslots * index;
		if (EXEC_KERN_EXPRESSION(kcxt, kexp_projection, &__tupsz))
		{
			if (!__tupsz.isnull && __tupsz.value > 0)
				tupsz = __tupsz.value;
			else
				STROM_ELOG(kcxt, "unable to comput tuple size");
		}
		else
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
		}
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
	/* data store has space? */
	if (__any_sync(__activemask(), try_suspend))
		return -2;
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
	if (wp->scan_done >= n_rels)
	{
		if (WARP_READ_POS(wp,n_rels) >= WARP_WRITE_POS(wp,n_rels))
			return -1;	/* ok, end of GpuJoin */
	}
	else
	{
		if (WARP_WRITE_POS(wp,n_rels) < WARP_READ_POS(wp,n_rels) + warpSize)
			return n_rels;	/* back to the previous depth */
	}
	return n_rels + 1;		/* elsewhere, try again? */
}

/*
 * kern_gpujoin_main
 */
KERNEL_FUNCTION(void)
kern_gpujoin_main(kern_session_info *session,
				  kern_gpujoin *kgjoin,
				  kern_multirels *kmrels,
				  kern_data_store *kds_src,
				  kern_data_extra *kds_extra,
				  kern_data_store *kds_dst)
{
	kern_context	   *kcxt;
	kern_warp_context  *wp, *wp_saved;
	void			  **kvars_addr;
	int				   *kvars_len;
	uint32_t			kvars_width;
	uint32_t		   *l_state;
	bool			   *matched;
	uint32_t			wp_unitsz;
	uint32_t			n_rels = kmrels->num_rels;
	int					depth;
	int					status;
	__shared__ uint32_t smx_row_count;

	/* setup execution context */
	INIT_KERNEL_CONTEXT(kcxt, session, kds_src, kmrels, kds_dst);
	wp_unitsz = __KERN_WARP_CONTEXT_UNITSZ_BASE(n_rels);
	wp = (kern_warp_context *)SHARED_WORKMEM(wp_unitsz, get_local_id() / warpSize);
	wp_saved = KERN_GPUJOIN_WARP_CONTEXT(kgjoin,  n_rels, kcxt->kvars_nslots);
	l_state  = KERN_GPUJOIN_LSTATE_ARRAY(kgjoin,  n_rels, kcxt->kvars_nslots);
	matched  = KERN_GPUJOIN_MATCHED_ARRAY(kgjoin, n_rels, kcxt->kvars_nslots);
	kvars_width = GPU_KVARS_UNITSZ * kcxt->kvars_nslots;
	kvars_addr = (void **)((char *)wp_saved + wp_unitsz);
	kvars_len = (int *)(kvars_addr + kvars_width * (n_rels+1));
	assert((char *)(kvars_len + kvars_width * (n_rels+1)) <=
		   (char *)wp_saved + KERN_WARP_CONTEXT_UNITSZ(n_rels, kcxt->kvars_nslots));

	if (kgjoin->resume_context)
	{
		/* resume the warp-context from the previous execution */
		if (LaneId() == 0)
			memcpy(wp, wp_saved, wp_unitsz);
		if (get_local_id() == 0)
			smx_row_count = wp->smx_row_count;
		depth = __shfl_sync(__activemask(), wp->depth, 0);
	}
	else
	{
		/* zero clear the wp */
		if (LaneId() == 0)
			memset(wp, 0, wp_unitsz);
		if (get_local_id() == 0)
			smx_row_count = 0;
		depth = 0;
		memset(l_state, 0, sizeof(void *) * kcxt->kvars_nslots);
		memset(matched, 0, sizeof(bool)   * kcxt->kvars_nslots);
	}
	__syncthreads();

	/* main logic of GpuJoin */
	while (depth >= 0)
	{
		if (depth == 0)
		{
			/* LOAD FROM THE SOURCE */
			depth = execGpuScanLoadSource(kcxt, wp,
										  kds_src,
										  kds_extra,
										  SESSION_KEXP_SCAN_LOAD_VARS(session),
										  SESSION_KEXP_SCAN_QUALS(session),
										  kvars_addr,	/* depth=0 */
										  kvars_len,	/* depth=0 */
										  &smx_row_count);
		}
		else if (depth > n_rels)
		{
			/* PROJECTION */
			assert(depth == n_rels+1);
			status = execGpuJoinProjection(kcxt, wp,
										   n_rels,
										   kds_dst,
										   SESSION_KEXP_PROJECTION(session),
										   kvars_addr + kvars_width * n_rels,
										   kvars_len  + kvars_width * n_rels);
			if (status >= 0)
				depth = status;
			else if (status == -2)
			{
				/* no space, try suspend! */
				if (LaneId() == 0)
					atomicAdd(&kgjoin->suspend_count, 1);
				break;
			}
			else
				depth = -1;
		}
		else if (kmrels->chunks[depth-1].is_nestloop)
		{
			/* NEST-LOOP */
			depth = execGpuJoinNestLoop(kcxt, wp,
										kmrels,
										depth,
										kvars_addr + kvars_width * (depth-1),
										kvars_len  + kvars_width * (depth-1),
										kvars_addr + kvars_width * depth,
										kvars_len  + kvars_width * depth,
										l_state[depth-1],	/* call by reference */
										matched[depth-1]);	/* call by reference */
		}
#if 0
		else if (kmrels->chunks[depth-1].gist_offset != 0)
		{
			/* GiST-INDEX-JOIN */
			depth = execGpuJoinGiSTJoin(kcxt, wp, ...);
		}
#endif
		else
		{
			/* HASH-JOIN */
			depth = execGpuJoinHashJoin(kcxt, wp,
										kmrels,
										depth,
										kvars_addr + kvars_width * (depth-1),
										kvars_len  + kvars_width * (depth-1),
										kvars_addr + kvars_width * depth,
										kvars_len  + kvars_width * depth,
										l_state[depth-1],	/* call by reference */
										matched[depth-1]);	/* call by reference */
		}
		assert(__shfl_sync(__activemask(), depth, 0) == depth);
		/* bailout if any error status */
		if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
			break;
	}
	__syncthreads();
	/* suspend the execution context */
	if (LaneId() == 0)
	{
		wp->depth = depth;
		wp->smx_row_count = smx_row_count;
		memcpy(wp_saved, wp, wp_unitsz);
	}
	STROM_WRITEBACK_ERROR_STATUS(&kgjoin->kerror, kcxt);
}
