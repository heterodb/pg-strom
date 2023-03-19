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
					char	   *src_kvars_addr_wp,
					char	   *dst_kvars_addr_wp,
					uint32_t   &l_state,
					bool	   &matched)
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

		read_pos = (read_pos % UNIT_TUPLES_PER_DEPTH);
		kcxt->kvars_addr = (void **)(src_kvars_addr_wp + read_pos * kcxt->kvars_nbytes);
		kcxt->kvars_len  = (int *)(kcxt->kvars_addr + kcxt->kvars_nslots);
		if (index < kds_heap->nitems)
		{
			kern_tupitem *tupitem;
			uint32_t	offset = KDS_GET_ROWINDEX(kds_heap)[index];
			xpu_int4_t	status;

			tupitem = (kern_tupitem *)((char *)kds_heap +
									   kds_heap->length -
									   __kds_unpack(offset));
			kexp = SESSION_KEXP_JOIN_LOAD_VARS(kcxt->session, depth-1);
			ExecLoadVarsHeapTuple(kcxt, kexp, depth, kds_heap, &tupitem->htup);
			kexp = SESSION_KEXP_JOIN_QUALS(kcxt->session, depth-1);
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
			kexp = SESSION_KEXP_JOIN_LOAD_VARS(kcxt->session, depth-1);
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
		write_pos = (write_pos % UNIT_TUPLES_PER_DEPTH);
		memcpy(dst_kvars_addr_wp + write_pos * kcxt->kvars_nbytes,
			   kcxt->kvars_addr,
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
					char	   *src_kvars_addr_wp,
					char	   *dst_kvars_addr_wp,
					uint32_t   &l_state,
					bool	   &matched)
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
	index = (read_pos % UNIT_TUPLES_PER_DEPTH);
	kcxt->kvars_addr = (void **)(src_kvars_addr_wp + index * kcxt->kvars_nbytes);
	kcxt->kvars_len  = (int *)(kcxt->kvars_addr + kcxt->kvars_nslots);
	if (l_state == 0)
	{
		/* pick up the first item from the hash-slot */
		if (read_pos < write_pos)
		{
			xpu_int4_t	hash;

			kexp = SESSION_KEXP_HASH_VALUE(kcxt->session, depth-1);
			if (EXEC_KERN_EXPRESSION(kcxt, kexp, &hash))
			{
				assert(!hash.isnull);
				for (khitem = KDS_HASH_FIRST_ITEM(kds_hash, hash.value);
					 khitem != NULL && khitem->hash != hash.value;
					 khitem = KDS_HASH_NEXT_ITEM(kds_hash, khitem));
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
		for (khitem = KDS_HASH_NEXT_ITEM(kds_hash, khitem);
			 khitem != NULL && khitem->hash != hash_value;
			 khitem = KDS_HASH_NEXT_ITEM(kds_hash, khitem));
	}
	/* error checks */
	if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
		return -1;
	if (khitem)
	{
		xpu_int4_t	status;

		kexp = SESSION_KEXP_JOIN_LOAD_VARS(kcxt->session, depth-1);
		ExecLoadVarsHeapTuple(kcxt, kexp, depth, kds_hash, &khitem->t.htup);
		kexp = SESSION_KEXP_JOIN_QUALS(kcxt->session, depth-1);
		if (EXEC_KERN_EXPRESSION(kcxt, kexp, &status))
		{
			assert(!status.isnull);
			if (status.value > 0)
				tuple_is_valid = true;
			if (status.value != 0)
				matched = true;
		}
		l_state = __kds_packed((char *)khitem - (char *)kds_hash);
	}
	else
	{
		if (kmrels->chunks[depth-1].left_outer &&
			l_state != UINT_MAX && !matched)
		{
			/* load NULL values on the inner portion */
			 kexp = SESSION_KEXP_JOIN_LOAD_VARS(kcxt->session, depth-1);
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
			   kcxt->kvars_addr,
			   kcxt->kvars_nbytes);
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
					  char *kvars_addr_wp)
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
		int			index = (read_pos % UNIT_TUPLES_PER_DEPTH);

		kcxt->kvars_addr = (void **)(kvars_addr_wp + index * kcxt->kvars_nbytes);
		kcxt->kvars_len = (int *)(kcxt->kvars_addr + kcxt->kvars_nslots);
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
	char			   *kvars_addr_wp;
	uint32_t			kvars_chunksz;
//	void			  **kvars_addr;
//	int				   *kvars_len;
//	uint32_t			kvars_width;
	uint32_t		   *l_state;
	bool			   *matched;
	uint32_t			wp_base_sz;
	uint32_t			n_rels = kmrels->num_rels;
	int					depth;
	int					status;
	__shared__ uint32_t smx_row_count;

	assert(kgtask->kvars_nslots == session->kcxt_kvars_nslots &&
		   kgtask->kvars_nbytes == session->kcxt_kvars_nbytes &&
		   kgtask->n_rels == n_rels);
	/* setup execution context */
	INIT_KERNEL_CONTEXT(kcxt, session, kds_src);
	wp_base_sz = __KERN_WARP_CONTEXT_BASESZ(n_rels);
	wp = (kern_warp_context *)SHARED_WORKMEM(wp_base_sz, get_local_id() / warpSize);
	wp_saved = KERN_GPUTASK_WARP_CONTEXT(kgtask);
	l_state = KERN_GPUTASK_LSTATE_ARRAY(kgtask);
	matched = KERN_GPUTASK_MATCHED_ARRAY(kgtask);
	kvars_chunksz = kcxt->kvars_nbytes * UNIT_TUPLES_PER_DEPTH;
	kvars_addr_wp = (char *)wp_saved + wp_base_sz;

	
#if 0	
	wp_unitsz = __KERN_WARP_CONTEXT_UNITSZ_BASE(n_rels);
	wp = (kern_warp_context *)SHARED_WORKMEM(wp_unitsz, get_local_id() / warpSize);
	wp_saved = KERN_GPUTASK_WARP_CONTEXT(kgtask);
	l_state = KERN_GPUTASK_LSTATE_ARRAY(kgtask);
	matched = KERN_GPUTASK_MATCHED_ARRAY(kgtask);
	kvars_width = UNIT_TUPLES_PER_DEPTH * kcxt->kvars_nslots;
	//!!!!!
	kvars_addr = (void **)((char *)wp_saved + wp_unitsz);
	kvars_len = (int *)(kvars_addr + kvars_width * (n_rels+1));
#endif
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
		depth = 0;
		memset(l_state, 0, sizeof(void *) * kcxt->kvars_nslots);
		memset(matched, 0, sizeof(bool)   * kcxt->kvars_nslots);
	}
	__syncthreads();

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
										  SESSION_KEXP_SCAN_LOAD_VARS(session),
										  SESSION_KEXP_SCAN_QUALS(session),
										  kvars_addr_wp,	/* depth=0 */
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
										   kvars_addr_wp + kvars_chunksz * n_rels);
			if (status >= 0)
				depth = status;
			else if (status == -2)
			{
				/* no space, try suspend! */
				if (LaneId() == 0)
					atomicAdd(&kgtask->suspend_count, 1);
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
										kvars_addr_wp + kvars_chunksz * (depth-1),
										kvars_addr_wp + kvars_chunksz * depth,
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
										kvars_addr_wp + kvars_chunksz * (depth-1),
										kvars_addr_wp + kvars_chunksz * depth,
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
		memcpy(wp_saved, wp, wp_base_sz);
	}
	STROM_WRITEBACK_ERROR_STATUS(&kgtask->kerror, kcxt);
}
