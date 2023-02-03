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
 * kern_gpujoin_main
 */
INLINE_FUNCTION(kern_warp_context *)
gpujoin_resume_context(kern_gpujoin *kgjoin,
					   uint32_t **combufs,
					   uint32_t *l_state,
					   bool *matched)
{
	kern_warp_context *wp;
	uint32_t	unitsz = KERN_WARP_CONTEXT_UNITSZ(kgjoin->num_rels);
	uint32_t	nwarps = (get_local_size() / warpSize);
	uint32_t	ncombs;
	uint32_t	comb_sz;
	char	   *pos;

	assert(unitsz * nwarps <= DynamicShmemSize());
	ncombs = ((kgjoin->num_rels + 1) *
			  (kgjoin->num_rels + 2) * (UNIT_TUPLES_PER_WARP / 2));
	comb_sz = MAXALIGN(sizeof(uint32_t) * ncombs);
	if ((unitsz + comb_sz) * nwarps <= DynamicShmemSize())
	{
		/* Here is enough amount of dynamic shared memory.
		 * so, both of wp+combuf are allocated from __shared__
		 */
		wp = (kern_warp_context *)
			SHARED_WORKMEM(unitsz + comb_sz, get_local_id() / warpSize);
		pos = kgjoin->data + (unitsz + comb_sz) * (get_global_id() / warpSize);
		if (LaneId() == 0)
		{
			memcpy(wp, pos, unitsz + comb_sz);
			wp->nrels = kgjoin->num_rels;
		}
		pos = (char *)wp + unitsz;
		for (int i=0; i <= kgjoin->num_rels; i++)
		{
			combufs[i] = (uint32_t *)pos;
			pos += sizeof(uint32_t) * (i+1) * UNIT_TUPLES_PER_WARP;
		}
		assert(pos <= (char *)wp + unitsz + comb_sz);
	}
	else
	{
		/* It does not have enough amount of dynamic shared memory.
		 * So, wp is allocated from the __shared__, but combuf refers
		 * the global memory directly. It is a slower fallback, but
		 * not avoidable due to the hardware restriction.
		 */
		wp = (kern_warp_context *)
			SHARED_WORKMEM(unitsz, get_local_id() / warpSize);
		pos = kgjoin->data + (unitsz + comb_sz) * (get_global_id() / warpSize);
		if (LaneId() == 0)
		{
			memcpy(wp, pos, unitsz);
			wp->nrels = kgjoin->num_rels;
		}
		pos += unitsz;
		for (int i=0; i <= kgjoin->num_rels; i++)
		{
			combufs[i] = (uint32_t *)pos;
			pos += sizeof(uint32_t) * (i+1) * UNIT_TUPLES_PER_WARP;
		}
	}
	/* restore l_state and matched */
	pos = kgjoin->data + (unitsz + comb_sz) * (get_global_size() / warpSize);
	unitsz = sizeof(uint32_t) * kgjoin->num_rels;
	memcpy(l_state, pos + unitsz * get_global_id(), unitsz);
	pos += MAXALIGN(unitsz * get_global_size());

	unitsz = sizeof(bool) * kgjoin->num_rels;
	memcpy(matched, pos + unitsz * get_global_id(), unitsz);
	pos += MAXALIGN(unitsz * get_global_size());

	return wp;
}

INLINE_FUNCTION(void)
gpujoin_suspend_context(kern_gpujoin *kgjoin,
						kern_warp_context *wp,
						uint32_t **combufs,
						uint32_t *l_state,
						bool *matched)
{
	uint32_t	unitsz = KERN_WARP_CONTEXT_UNITSZ(kgjoin->num_rels);
	uint32_t	nwarps = (get_local_size() / warpSize);
	uint32_t	ncombs;
	uint32_t	comb_sz;
	char	   *pos;

	assert(unitsz * nwarps <= DynamicShmemSize());
	ncombs = ((kgjoin->num_rels + 1) *
			  (kgjoin->num_rels + 2) * (UNIT_TUPLES_PER_WARP / 2));
	comb_sz = MAXALIGN(sizeof(uint32_t) * ncombs);
	pos = kgjoin->data + (unitsz + comb_sz) * (get_global_id() / warpSize);
	if (LaneId() == 0)
	{
		memcpy(pos, wp, unitsz);
		pos += unitsz;
		for (int i=0; i <= kgjoin->num_rels; i++)
		{
			uint32_t	sz = sizeof(uint32_t) * (i+1) * UNIT_TUPLES_PER_WARP;

			if ((char *)combufs[i] != pos)
				memcpy(pos, combufs[i], sz);
			pos += sz;
		}
	}
	/* suspend l_state and matched */
	pos = kgjoin->data + (unitsz + comb_sz) * (get_global_size() / warpSize);
	unitsz = sizeof(uint32_t) * kgjoin->num_rels;
	memcpy(pos + unitsz * get_global_id(), l_state, unitsz);
	pos += MAXALIGN(unitsz * get_global_size());

	unitsz = sizeof(bool) * kgjoin->num_rels;
	memcpy(pos + unitsz * get_global_id(), matched, unitsz);
	pos += MAXALIGN(unitsz * get_global_size());
}

/*
 * GPU Nested-Loop
 */
INLINE_FUNCTION(int)
execGpuJoinNestLoop(kern_context *kcxt,
					kern_warp_context *wp,
					kern_gpujoin *kgjoin,
					kern_multirels *kmrels,
					kern_data_store *kds_src,
					kern_data_extra *kds_extra,
					kern_data_store **kds_inners,
					int depth,
					uint32_t *rd_combuf,
					uint32_t *wr_combuf,
					uint32_t &l_state,
					bool &matched)
{
	kern_data_store *kds_heap = kds_inners[depth-1];
	kern_expression *kexp_join_quals = SESSION_KEXP_JOIN_QUALS(kcxt->session, depth);
	uint32_t	read_pos;
	uint32_t	write_pos;
	uint32_t	inner_pos = 0;
	uint32_t	mask;
	bool		tuple_is_valid = false;

	if (WARP_WRITE_POS(wp,depth) >= WARP_READ_POS(wp,depth) + warpSize)
	{
		/* next depth already keeps warpSize or more pending tuples. */
		return depth+1;
	}
	else if (__all_sync(__activemask(), l_state >= kds_heap->nitems))
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
	rd_combuf += (read_pos % UNIT_TUPLES_PER_WARP) * depth;
	if (read_pos < WARP_WRITE_POS(wp,depth-1))
	{
		kern_expression *kexp = SESSION_KEXP_JOIN_QUALS(kcxt->session, depth);
		uint32_t	index = l_state++;

		if (index < kds_heap->nitems)
		{
			kern_tupitem *tupitem;
			uint32_t	offset = KDS_GET_ROWINDEX(kds_heap)[index];
			int			status;

			tupitem = (kern_tupitem *)((char *)kds_heap +
									   kds_heap->length -
									   __kds_unpack(offset));
			inner_pos = __kds_packed((char *)&tupitem->htup - (char *)kds_heap);
			if (ExecKernJoinQuals(kcxt,
								  kexp,
								  &status,
								  rd_combuf,
								  inner_pos,
								  kds_src,
								  kds_extra,
								  depth,
								  kds_inners))
			{
				if (status > 0)
					tuple_is_valid = true;
				else if (status < 0)
					matched = true;
			}
		}
		else if (kmrels->chunks[depth-1].left_outer &&
				 index == kds_heap->nitems && !matched)
		{
			/* no matched outer rows, but LEFT/FULL OUTER JOIN */
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
		wr_combuf += (write_pos % UNIT_TUPLES_PER_WARP) * (depth+1);
		memcpy(wr_combuf, rd_combuf, sizeof(uint32_t) * depth);
		wr_combuf[depth] = inner_pos;
	}
	__syncwarp();
	if (WARP_WRITE_POS(wp,depth) >= WARP_READ_POS(wp,depth) + warpSize)
		return depth+1;
	return depth;
}

/*
 * GPU Hash-Join
 */
INLINE_FUNCTION(int)
execGpuJoinHashJoin(kern_context *kcxt,
                    kern_warp_context *wp,
                    kern_gpujoin *kgjoin,
                    kern_multirels *kmrels,
                    kern_data_store *kds_src,
                    kern_data_extra *kds_extra,
					kern_data_store **kds_inners,
                    int depth,
					uint32_t *rd_combuf,
					uint32_t *wr_combuf,
					uint32_t &l_state,
					bool &matched)
{
	kern_data_store *kds_hash = kds_inners[depth-1];
	kern_expression *kexp_join_quals = SESSION_KEXP_JOIN_QUALS(kcxt->session, depth);
	kern_hashitem *khitem = NULL;
	uint32_t	hash;
	uint32_t	read_pos;
	uint32_t	write_pos;
	uint32_t	inner_pos;
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
	else if (__all_sync(__activemask(), l_state == UINT_MAX))
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
	read_pos = WARP_READ_POS(wp,depth-1) + LaneId();
	rd_combuf += (read_pos % UNIT_TUPLES_PER_WARP) * depth;
	if (l_state == 0)
	{
		if (read_pos < WARP_WRITE_POS(wp,depth-1))
		{
			kern_expression *kexp = SESSION_KEXP_HASH_VALUE(kcxt->session, depth);

			if (ExecKernHashValue(kcxt,
								  kexp,
								  &hash,
								  rd_combuf,
								  kds_src,
								  kds_extra,
								  depth,
								  kds_inners))
			{
				khitem = KDS_HASH_FIRST_ITEM(kds_hash, hash);
			}
		}
		else
		{
			l_state = UINT_MAX;
		}
	}
	else if (l_state != UINT_MAX)
	{
		khitem = (kern_hashitem *)((char *)kds_hash
								   + __kds_unpack(l_state)
								   - offsetof(kern_hashitem, t.htup));
		hash = khitem->hash;
		/* pick up next one if any */
		khitem = KDS_HASH_NEXT_ITEM(kds_hash, khitem);
	}
	/* error checks */
	if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
		return -1;

	while (khitem && khitem->hash != hash)
		khitem = KDS_HASH_NEXT_ITEM(kds_hash, khitem);

	if (khitem)
	{
		kern_expression *kexp = SESSION_KEXP_JOIN_QUALS(kcxt->session, depth);
		int		status;

		inner_pos = __kds_packed((char *)&khitem->t.htup - (char *)kds_hash);
		if (ExecKernJoinQuals(kcxt,
							  kexp,
							  &status,
							  rd_combuf,
							  inner_pos,
							  kds_src,
							  kds_extra,
							  depth,
							  kds_inners))
		{
			if (status > 0)
				tuple_is_valid = true;
			else if (status < 0)
				matched = true;
		}
	}
	else if (kmrels->chunks[depth-1].left_outer &&
			 l_state != UINT_MAX && !matched)
	{
		/* no matched outer rows, but LEFT/FULL OUTER JOIN */
		tuple_is_valid = true;
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
		wr_combuf += (write_pos % UNIT_TUPLES_PER_WARP) * (depth+1);
		memcpy(wr_combuf, rd_combuf, sizeof(uint32_t) * depth);
		wr_combuf[depth] = inner_pos;
	}
	__syncwarp();
	if (WARP_WRITE_POS(wp,depth) >= WARP_READ_POS(wp,depth) + warpSize)
		return depth+1;
	return depth;
}

/*
 * GPU Projection
 */
STATIC_FUNCTION(int)
execGpuJoinProjection(kern_context *kcxt,
					  kern_warp_context *wp,
					  kern_gpujoin *kgjoin,
					  kern_multirels *kmrels,
					  kern_data_store *kds_src,
					  kern_data_extra *kds_extra,
					  kern_data_store *kds_dst,
					  kern_data_store **kds_inners,
					  int n_rels,
					  uint32_t *rd_combuf)
{
	kern_expression *kexp = SESSION_KEXP_SCAN_PROJS(kcxt->session);
	uint32_t	read_pos;
	int			tupsz = 0;

	read_pos = WARP_READ_POS(wp,n_rels) + LaneId();
	if (read_pos < WARP_WRITE_POS(wp,n_rels))
	{
		rd_combuf += (read_pos % UNIT_TUPLES_PER_WARP) * (n_rels + 1);
		tupsz = ExecKernProjection(kcxt,
								   kexp,
								   kds_dst,
								   rd_combuf,
								   kds_src,
								   kds_extra,
								   n_rels,
								   kds_inners);
	}
	/* error checks */
	if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
		return -1;
	/* write out to kds_dst */
	if (execGpuProjection(kcxt, kexp, kds_dst, tupsz) == 0)
	{
		/* kds_dst is full, so try suspend/resume */
		assert(__activemask() == 0xffffffffU);
		return -2;
	}
	/* update the read position */
	if (LaneId() == 0)
		WARP_READ_POS(wp,n_rels) = Min(WARP_READ_POS(wp,n_rels) + warpSize,
									   WARP_WRITE_POS(wp,n_rels));
	__syncwarp();
	if (wp->scan_done >= n_rels)
	{
		if (WARP_WRITE_POS(wp,n_rels) <= WARP_READ_POS(wp,n_rels))
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
	kern_warp_context  *wp;
	kern_data_store	  **kds_inners;
	uint32_t		  **combufs;
	uint32_t		   *l_state;
	bool			   *matched;
	int					n_rels = kmrels->num_rels;
	int					depth;
	__shared__ uint32_t smx_row_count;

	INIT_KERNEL_CONTEXT(kcxt, session);
	kds_inners = (kern_data_store **) alloca(sizeof(kern_data_store *) * n_rels);
	for (int i=0; i < kmrels->num_rels; i++)
		kds_inners[i] = KERN_MULTIRELS_INNER_KDS(kmrels, i);
	combufs = (uint32_t **) alloca(sizeof(uint32_t *) * (n_rels+1));
	l_state = (uint32_t *) alloca(sizeof(uint32_t) * n_rels);
	matched = (bool *) alloca(sizeof(bool) * n_rels);

	/* sanity checks */
	assert(kgjoin->num_rels == kmrels->num_rels);
	/* resume the previous execution context */
	wp = gpujoin_resume_context(kgjoin, combufs, l_state, matched);
	if (get_local_id() == 0)
		smx_row_count = wp->smx_row_count;
	depth = wp->depth;
	__syncthreads();

	/* main logic of GpuJoin */
	assert(depth == __shfl_sync(__activemask(), depth, 0));
	while (depth >= 0)
	{
		if (depth == 0)
		{
			/* LOAD FROM THE SOURCE */
			depth = execGpuScanLoadSource(kcxt, wp,
										  kds_src,
										  kds_extra,
										  SESSION_KEXP_SCAN_QUALS(session),
										  combufs[0],
										  &smx_row_count);
		}
		else if (depth > n_rels)
		{
			/* PROJECTION */
			assert(depth == n_rels + 1);
			depth = execGpuJoinProjection(kcxt, wp,
										   kgjoin,
										   kmrels,
										   kds_src,
										   kds_extra,
										   kds_dst,
										   kds_inners,
										   n_rels,
										   combufs[n_rels]);
			/* special case handling if suspend/resume is needed */
			if (depth == -2)
			{
				if (LaneId() == 0)
					atomicAdd(&kgjoin->suspend_count, 1);
				depth = n_rels + 1;
				break;
			}
		}
		else if (kmrels->chunks[depth-1].is_nestloop)
		{
			/* NEST-LOOP */
			depth = execGpuJoinNestLoop(kcxt, wp,
										kgjoin,
										kmrels,
										kds_src,
										kds_extra,
										kds_inners,
										depth,
										combufs[depth-1],	/* read combuf */
										combufs[depth],		/* write combuf */
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
										kgjoin,
										kmrels,
										kds_src,
										kds_extra,
										kds_inners,
										depth,
										combufs[depth-1],	/* read combuf */
										combufs[depth],		/* write combuf */
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
	}
	gpujoin_suspend_context(kgjoin, wp, combufs, l_state, matched);
}
