/*
 * cuda_gpuscan.cu
 *
 * Device implementation of GpuScan
 * ----
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "cuda_common.h"

/*
 * pgstrom_stair_sum_xxxx
 */
static __shared__ union {
	uint32_t	u32[WARPSIZE];
	uint64_t	u64[WARPSIZE];
	int32_t		i32[WARPSIZE];
	int64_t		i64[WARPSIZE];
	float8_t	fp64[WARPSIZE];
	int128_t	i128[WARPSIZE];
} __stair_sum_buffer;

template <typename T>
INLINE_FUNCTION(T)
__stair_sum_warp_common(T my_value)
{
	T	curr = my_value;
	T	temp;

	assert(__activemask() == ~0U);
	temp = __shfl_sync(__activemask(), curr, (LaneId() & ~0x01));
	if ((LaneId() & 0x01) != 0)
		curr += temp;

	temp = __shfl_sync(__activemask(), curr, (LaneId() & ~0x03) | 0x01);
	if ((LaneId() & 0x02) != 0)
		curr += temp;

	temp = __shfl_sync(__activemask(), curr, (LaneId() & ~0x07) | 0x03);
	if ((LaneId() & 0x04) != 0)
		curr += temp;

	temp = __shfl_sync(__activemask(), curr, (LaneId() & ~0x0f) | 0x07);
	if ((LaneId() & 0x08) != 0)
		curr += temp;

	temp = __shfl_sync(__activemask(), curr, (LaneId() & ~0x1f) | 0x0f);
	if ((LaneId() & 0x10) != 0)
		curr += temp;

	return curr;
}

INLINE_FUNCTION(int128_t)
__stair_sum_warp_common(int128_t my_value)
{
	int128_packed_t		curr, temp;

	assert(__activemask() == ~0U);
	__store_int128_packed(&curr, my_value);
	temp.u64_lo = __shfl_sync(__activemask(), curr.u64_lo, (LaneId() & ~0x01));
	temp.u64_hi = __shfl_sync(__activemask(), curr.u64_hi, (LaneId() & ~0x01));
	if ((LaneId() & 0x01) != 0)
		__store_int128_packed(&curr, (__fetch_int128_packed(&curr) +
									  __fetch_int128_packed(&temp)));

	temp.u64_lo = __shfl_sync(__activemask(), curr.u64_lo, (LaneId() & ~0x03) | 0x01);
	temp.u64_hi = __shfl_sync(__activemask(), curr.u64_hi, (LaneId() & ~0x03) | 0x01);
	if ((LaneId() & 0x02) != 0)
		__store_int128_packed(&curr, (__fetch_int128_packed(&curr) +
									  __fetch_int128_packed(&temp)));

	temp.u64_lo = __shfl_sync(__activemask(), curr.u64_lo, (LaneId() & ~0x07) | 0x03);
	temp.u64_hi = __shfl_sync(__activemask(), curr.u64_hi, (LaneId() & ~0x07) | 0x03);
	if ((LaneId() & 0x04) != 0)
		__store_int128_packed(&curr, (__fetch_int128_packed(&curr) +
									  __fetch_int128_packed(&temp)));

	temp.u64_lo = __shfl_sync(__activemask(), curr.u64_lo, (LaneId() & ~0x0f) | 0x07);
	temp.u64_hi = __shfl_sync(__activemask(), curr.u64_hi, (LaneId() & ~0x0f) | 0x07);
	if ((LaneId() & 0x08) != 0)
		__store_int128_packed(&curr, (__fetch_int128_packed(&curr) +
									  __fetch_int128_packed(&temp)));

	temp.u64_lo = __shfl_sync(__activemask(), curr.u64_lo, (LaneId() & ~0x1f) | 0x0f);
	temp.u64_hi = __shfl_sync(__activemask(), curr.u64_hi, (LaneId() & ~0x1f) | 0x0f);
	if ((LaneId() & 0x10) != 0)
		__store_int128_packed(&curr, (__fetch_int128_packed(&curr) +
									  __fetch_int128_packed(&temp)));
	return __fetch_int128_packed(&curr);
}

PUBLIC_FUNCTION(uint32_t)
pgstrom_stair_sum_binary(bool predicate, uint32_t *p_total_count)
{
	uint32_t	n_warps = get_local_size() / warpSize;
	uint32_t	warp_id = get_local_id()   / warpSize;
	uint32_t	mask;
	uint32_t	sum;

	assert(get_local_size() <= WARPSIZE * WARPSIZE);
	assert(__activemask() == ~0U);
	mask = __ballot_sync(__activemask(), predicate);
	if (LaneId() == 0)
		__stair_sum_buffer.u32[warp_id] = __popc(mask);
	__syncthreads();

	if (warp_id == 0)
	{
		uint32_t	temp = (LaneId() < n_warps ? __stair_sum_buffer.u32[LaneId()] : 0);

		__stair_sum_buffer.u32[LaneId()] = __stair_sum_warp_common(temp);
	}
	__syncthreads();

	if (p_total_count)
		*p_total_count = __stair_sum_buffer.u32[warpSize-1];
	sum = (warp_id > 0 ? __stair_sum_buffer.u32[warp_id-1] : 0);
	__syncthreads();

	mask &= ((1U << LaneId()) - 1);		/* not include myself */
	return sum + __popc(mask);
}

#define PGSTROM_STAIR_SUM_TEMPLATE(SUFFIX, BASETYPE, FIELD)				\
	PUBLIC_FUNCTION(BASETYPE)											\
	pgstrom_stair_sum_##SUFFIX(BASETYPE value, BASETYPE *p_total_count)	\
	{																	\
		uint32_t	n_warps = get_local_size() / warpSize;				\
		uint32_t	warp_id = get_local_id()   / warpSize;				\
		BASETYPE	warp_sum;											\
		BASETYPE	sum;												\
																		\
		assert(get_local_size() <= WARPSIZE * WARPSIZE);				\
		assert(__activemask() == ~0U);									\
		warp_sum = __stair_sum_warp_common(value);						\
		if (LaneId() == warpSize - 1)									\
			__stair_sum_buffer.FIELD[warp_id] = warp_sum;				\
		__syncthreads();												\
																		\
		if (warp_id == 0)												\
		{																\
			BASETYPE	temp = (LaneId() < n_warps						\
								? __stair_sum_buffer.FIELD[LaneId()] : 0); \
			__stair_sum_buffer.FIELD[LaneId()] = __stair_sum_warp_common(temp);	\
		}																\
		__syncthreads();												\
																		\
		if (p_total_count)												\
			*p_total_count = __stair_sum_buffer.FIELD[warpSize-1];		\
		sum = (warp_id > 0 ? __stair_sum_buffer.FIELD[warp_id-1] : 0);	\
		__syncthreads();												\
																		\
		return sum + warp_sum;											\
	}

PGSTROM_STAIR_SUM_TEMPLATE(uint32, uint32_t, u32)
PGSTROM_STAIR_SUM_TEMPLATE(uint64, uint64_t, u64)
PGSTROM_STAIR_SUM_TEMPLATE(int64,  int64_t,  i64)
PGSTROM_STAIR_SUM_TEMPLATE(int128, int128_t, i128)
PGSTROM_STAIR_SUM_TEMPLATE(fp64,   float8_t, fp64)

#define PGSTROM_LOCAL_MINMAX_TEMPLATE(SUFFIX, BASETYPE, FIELD, OPER, INVAL)	\
	PUBLIC_FUNCTION(BASETYPE)											\
	pgstrom_local_##SUFFIX(BASETYPE my_value)							\
	{																	\
		int			warp_id = get_local_id()   / warpSize;				\
		int			n_warps = get_local_size() / warpSize;				\
		BASETYPE	curr = my_value;									\
		BASETYPE	temp;												\
																		\
		/* makes warp local min/max */									\
		assert(__activemask() == ~0U);									\
		temp = __shfl_xor_sync(__activemask(), curr, 0x0001);			\
		curr = OPER(curr, temp);										\
		temp = __shfl_xor_sync(__activemask(), curr, 0x0002);			\
		curr = OPER(curr, temp);										\
		temp = __shfl_xor_sync(__activemask(), curr, 0x0004);			\
		curr = OPER(curr, temp);										\
		temp = __shfl_xor_sync(__activemask(), curr, 0x0008);			\
		curr = OPER(curr, temp);										\
		temp = __shfl_xor_sync(__activemask(), curr, 0x0010);			\
		curr = OPER(curr, temp);										\
																		\
		if (LaneId() == 0)												\
			__stair_sum_buffer.FIELD[warp_id] = curr;					\
		__syncthreads();												\
																		\
		if (warp_id == 0)												\
		{																\
			assert(__activemask() == ~0U);								\
			curr = (LaneId() < n_warps ? __stair_sum_buffer.FIELD[LaneId()] : INVAL); \
																		\
			temp = __shfl_xor_sync(__activemask(), curr, 0x0001);		\
			curr = OPER(curr, temp);									\
			temp = __shfl_xor_sync(__activemask(), curr, 0x0002);		\
			curr = OPER(curr, temp);									\
			temp = __shfl_xor_sync(__activemask(), curr, 0x0004);		\
			curr = OPER(curr, temp);									\
			temp = __shfl_xor_sync(__activemask(), curr, 0x0008);		\
			curr = OPER(curr, temp);									\
			temp = __shfl_xor_sync(__activemask(), curr, 0x0010);		\
			curr = OPER(curr, temp);									\
																		\
			__stair_sum_buffer.FIELD[LaneId()] = curr;					\
		}																\
		__syncthreads();												\
		curr = __stair_sum_buffer.FIELD[LaneId()];						\
		__syncthreads();												\
		return curr;													\
	}

PGSTROM_LOCAL_MINMAX_TEMPLATE(min_int32, int32_t, i32,  Min,  INT_MAX)
PGSTROM_LOCAL_MINMAX_TEMPLATE(max_int32, int32_t, i32,  Max,  INT_MIN)
PGSTROM_LOCAL_MINMAX_TEMPLATE(min_int64, int64_t, i64,  Min,  LONG_MAX)
PGSTROM_LOCAL_MINMAX_TEMPLATE(max_int64, int64_t, i64,  Max,  LONG_MIN)
PGSTROM_LOCAL_MINMAX_TEMPLATE(min_fp64, float8_t, fp64, Min,  DBL_MAX)
PGSTROM_LOCAL_MINMAX_TEMPLATE(max_fp64, float8_t, fp64, Max, -DBL_MAX)
PGSTROM_LOCAL_MINMAX_TEMPLATE(or_uint32, uint32_t, u32, Or, 0)

/*
 * pgstrom_global_stair_sum_u32
 *
 * nitems <= 2^11 : 1-step
 * nitems <= 2^22 : 2-step
 * nitems <= 2^33 : 3-step
 */
template <typename T>
INLINE_FUNCTION(void)
__stair_sum_warp_common64(T &value0, T &value1)
{
	T	temp;

	value1 += value0;

	assert(__activemask() == ~0U);
	temp = __shfl_sync(__activemask(), value1, (LaneId() & ~0x01));
	if ((LaneId() & 0x01) != 0)
	{
		value0 += temp;
		value1 += temp;
	}
	temp = __shfl_sync(__activemask(), value1, (LaneId() & ~0x03) | 0x01);
	if ((LaneId() & 0x02) != 0)
	{
		value0 += temp;
		value1 += temp;
	}
	temp = __shfl_sync(__activemask(), value1, (LaneId() & ~0x07) | 0x03);
	if ((LaneId() & 0x04) != 0)
	{
		value0 += temp;
		value1 += temp;
	}
	temp = __shfl_sync(__activemask(), value1, (LaneId() & ~0x0f) | 0x07);
	if ((LaneId() & 0x08) != 0)
	{
		value0 += temp;
		value1 += temp;
	}
	temp = __shfl_sync(__activemask(), value1, (LaneId() & ~0x1f) | 0x0f);
	if ((LaneId() & 0x10) != 0)
	{
		value0 += temp;
		value1 += temp;
	}
}

KERNEL_FUNCTION(void)
kern_global_stair_sum_u32(uint32_t *values, uint32_t nitems, uint32_t step)
{
	uint32_t   *upper;
	uint32_t	upper_nitems;
	uint32_t	reverse;

	assert(get_local_size() == 512 && __activemask() == ~0U);
	if (nitems <= (1U<<11))			/* <= 2048 */
	{
		upper = values + nitems;
		upper_nitems = 1;
		if (step > 0)
			return;		/* nothing to do */
		reverse = 0;
	}
	else if (nitems <= (1U<<22))	/* <= 4194304 */
	{
		upper = values + nitems;
		upper_nitems = (nitems + 2047) >> 11;
		if (step == 1)
		{
			values = upper;
			nitems = upper_nitems;
			upper = values + nitems;
			upper_nitems = 1;
		}
		else if (step > 2)
			return;		/* nothing to do */
		reverse = 1;
	}
	else
	{
		upper = values + nitems;
		upper_nitems = (nitems + 2047) >> 11;
		if (step >= 1 && step <= 3)
		{
			values = upper;
			nitems = upper_nitems;
			upper = values + nitems;
			upper_nitems = (nitems + 2047) >> 11;
			if (step == 2)
			{
				values = upper;
				nitems = upper_nitems;
				upper = values + nitems;
				upper_nitems = 1;
			}
		}
		else if (step > 4)
			return;		/* nothing to do */
		reverse = 2;
	}
	/* run the main stair-like sum */
	if (step <= reverse)
	{
		uint32_t	group_id;

		for (group_id = get_group_id();
			 (group_id << 11) < nitems;
			 group_id += get_num_groups())
		{
			uint32_t	warp_id0 = get_local_id() / warpSize;
			uint32_t	warp_id1 = warp_id0 + (get_local_size() / warpSize);
			uint32_t	index0 = (group_id << 11) + 2 * get_local_id();
			uint32_t	index1 = index0 + 2 * get_local_size();
			uint32_t	x0, y0, x1, y1, temp;

			/* calculation of the first half */
			x0 = (index0   < nitems ? values[index0]   : 0);
			y0 = (index0+1 < nitems ? values[index0+1] : 0);
			__stair_sum_warp_common64(x0, y0);
			if (LaneId() == warpSize-1)
				__stair_sum_buffer.u32[warp_id0] = y0;
			/* calculation of the second half */
			x1 = (index1   < nitems ? values[index1]   : 0);
			y1 = (index1+1 < nitems ? values[index1+1] : 0);
			__stair_sum_warp_common64(x1, y1);
			if (LaneId() == warpSize-1)
				__stair_sum_buffer.u32[warp_id1] = y1;
			/* summarization of the 32 items */
			__syncthreads();
			if (get_local_id() < warpSize)
			{
				temp = __stair_sum_buffer.u32[LaneId()];
				__stair_sum_buffer.u32[LaneId()] = __stair_sum_warp_common(temp);
			}
			__syncthreads();
			temp = (warp_id0 > 0 ? __stair_sum_buffer.u32[warp_id0-1] : 0);
			x0 += temp;
			y0 += temp;
			temp = (warp_id1 > 0 ? __stair_sum_buffer.u32[warp_id1-1] : 0);
			x1 += temp;
			y1 += temp;
			/* write back results */
			if (index0 < nitems)
				values[index0] = x0;
			if (index0+1 < nitems)
				values[index0+1] = y0;
			if (index1 < nitems)
				values[index1] = x1;
			if (index1+1 < nitems)
				values[index1+1] = y1;
			assert(group_id < upper_nitems);
			if (get_local_id()+1 == get_local_size())
				upper[group_id] = y1;
		}
	}
	if (step >= reverse)
	{
		uint32_t	index;

		for (index = get_global_id(); index < nitems; index += get_global_size())
		{
			uint32_t	upper_id = (index >> 11);

			assert(upper_id < upper_nitems);
			if (upper_id > 0)
				values[index] += upper[upper_id-1];
		}
	}
}

/* ----------------------------------------------------------------
 *
 * execGpuScanLoadSource and related
 *
 * ----------------------------------------------------------------
 */

/*
 * __gpuscan_load_source_block
 */
STATIC_FUNCTION(int)
__gpuscan_load_source_block(kern_context *kcxt,
							kern_warp_context *wp,
							const kern_data_store *kds_src,
							const kern_expression *kexp_load_vars,
							const kern_expression *kexp_scan_quals,
							const kern_expression *kexp_move_vars,
							char *dst_kvecs_buffer)
{
	uint32_t	wr_pos = wp->lp_wr_pos;
	uint32_t	rd_pos = wp->lp_rd_pos;
	uint32_t	block_id;
	uint32_t	count;
	bool		has_next_lp_items = false;
	HeapTupleHeaderData *htup = NULL;

	assert(wr_pos >= rd_pos);
	block_id = (get_global_size() / warpSize) * wp->smx_row_count;
	if (block_id >= kds_src->nitems || wr_pos >= rd_pos + get_local_size())
	{
		uint32_t	off;

		rd_pos += get_local_id();
		if (rd_pos < wr_pos)
		{
			off = wp->lp_items[rd_pos % LP_ITEMS_PER_BLOCK];
			htup = (HeapTupleHeaderData *)((char *)kds_src + off);
			if (!ExecLoadVarsOuterHeap(kcxt,
									   kexp_load_vars,
									   kexp_scan_quals,
									   kds_src, htup))
				htup = NULL;
		}
		/* error checks */
		if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
			return -1;
		if (get_local_id() == 0)
			wp->lp_rd_pos = Min(wp->lp_wr_pos,
								wp->lp_rd_pos + get_local_size());
		/*
		 * save the private kvars on the warp-buffer
		 */
		wr_pos = WARP_WRITE_POS(wp,0);
		wr_pos += pgstrom_stair_sum_binary(htup != NULL, &count);
		if (get_local_id() == 0)
			WARP_WRITE_POS(wp,0) += count;
		if (htup != NULL)
		{
			if (!ExecMoveKernelVariables(kcxt,
										 kexp_move_vars,
										 dst_kvecs_buffer,
										 (wr_pos % KVEC_UNITSZ)))
			{
				assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
			}
		}
		/* error checks */
		if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
			return -1;
		/* end-of-scan checks */
		if (block_id >= kds_src->nitems &&	/* no more blocks to fetch */
			wp->lp_rd_pos >= wp->lp_wr_pos)	/* no more pending tuples */
		{
			if (get_local_id() == 0)
				wp->scan_done = 1;
			return 1;
		}
		/* move to the next depth if more than blockSize tuples were fetched */
		return (WARP_WRITE_POS(wp,0) >= WARP_READ_POS(wp,0) + get_local_size() ? 1 : 0);
	}

	/*
	 * Here, number of pending tuples (which is saved in the lp_items[]) is
	 * not enough to run ScanQuals checks. So, we move to the next bunch of
	 * line-items or next block.
	 * The pending tuples just passed the MVCC visivility checks, but
	 * ScanQuals check is not applied yet. We try to run ScanQuals checks
	 * with maximum number of threads simultaneously, as large as we can.
	 */
	block_id += (get_global_id() / warpSize);
	if (block_id < kds_src->nitems)
	{
		PageHeaderData *pg_page = KDS_BLOCK_PGPAGE(kds_src, block_id);
		BlockNumber		block_nr = KDS_BLOCK_BLCKNR(kds_src, block_id);
		uint32_t		nitems = PageGetMaxOffsetNumber(pg_page);
		uint32_t		index;

		index = wp->lp_count * warpSize + LaneId();
		if (index < PageGetMaxOffsetNumber(pg_page))
		{
			ItemIdData *lpp = &pg_page->pd_linp[index];

			assert((char *)lpp < (char *)pg_page + BLCKSZ);
			if (ItemIdIsNormal(lpp))
			{
				htup = (HeapTupleHeaderData *)PageGetItem(pg_page, lpp);
				/* for ctid system column reference */
				htup->t_ctid.ip_blkid.bi_hi = (uint16_t)(block_nr >> 16);
				htup->t_ctid.ip_blkid.bi_lo = (uint16_t)(block_nr & 0xffffU);
				htup->t_ctid.ip_posid = index + 1;
			}
		}
		has_next_lp_items = (index + warpSize < nitems);
	}
	/* put visible tuples on the lp_items[] array */
	wr_pos = wp->lp_wr_pos;
	wr_pos += pgstrom_stair_sum_binary(htup != NULL, &count);
	if (get_local_id() == 0)
		wp->lp_wr_pos += count;
	if (htup != NULL)
	{
		wp->lp_items[wr_pos % LP_ITEMS_PER_BLOCK]
			= (uint32_t)((char *)htup - (char *)kds_src);
	}
	/* increment the row/line pointer */
	if (__syncthreads_count(has_next_lp_items) > 0)
	{
		if (get_local_id() == 0)
			wp->lp_count++;
	}
	else
	{
		if (get_local_id() == 0)
		{
			wp->smx_row_count++;
			wp->lp_count = 0;
		}
	}
	return 0;	/* stay depth=0 */
}

/*
 * __gpuscan_load_source_arrow
 */
STATIC_FUNCTION(int)
__gpuscan_load_source_arrow(kern_context *kcxt,
							kern_warp_context *wp,
							const kern_data_store *kds_src,
							const kern_expression *kexp_load_vars,
							const kern_expression *kexp_scan_quals,
							const kern_expression *kexp_move_vars,
							char *dst_kvecs_buffer)
{
	uint32_t	count;
	uint32_t	index;
	uint32_t	wr_pos;
	bool		is_valid = false;

	/* compute the next row-index */
	index = get_global_size() * wp->smx_row_count + get_global_base();
	if (index >= kds_src->nitems)
	{
		if (get_local_id() == 0)
			wp->scan_done = 1;
		return 1;
	}
	index += get_local_id();

	/*
	 * fetch arrow tuple
	 */
	if (index < kds_src->nitems)
	{
		if (ExecLoadVarsOuterArrow(kcxt,
								   kexp_load_vars,
								   kexp_scan_quals,
								   kds_src,
								   index))
			is_valid = true;
	}
	/* error checks */
    if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
		return -1;
	/*
	 * save the private kvars slot on the combination buffer (depth=0)
	 */
	wr_pos = WARP_WRITE_POS(wp,0);
	wr_pos += pgstrom_stair_sum_binary(is_valid, &count);
	if (is_valid)
	{
		if (!ExecMoveKernelVariables(kcxt,
									 kexp_move_vars,
									 dst_kvecs_buffer,
									 (wr_pos % KVEC_UNITSZ)))
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
		}
	}
	/* error checks */
	if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
		return -1;
	/* make forward read/write pointer */
	if (get_local_id() == 0)
	{
		wp->smx_row_count++;
		WARP_WRITE_POS(wp,0) += count;
	}
	__syncthreads();
	/* move to the next depth, if more than blockSize rows were fetched. */
	return (WARP_WRITE_POS(wp,0) >= WARP_READ_POS(wp,0) + get_local_size() ? 1 : 0);
}

/*
 * __gpuscan_load_source_row (KDS_FORMAT_ROW/HASH)
 */
INLINE_FUNCTION(bool)
__gpuscan_check_row_visibility(kern_context *kcxt,
							   kern_tupitem *titem)
{
	SerializedTransactionState *xstate = SESSION_XACT_STATE(kcxt->session);
	const kern_tupitem_xact_attrs *xattrs
		= KERN_TUPITEM_GET_XACT_ATTRS(titem);
	assert(xstate != NULL);
	if (!xattrs)
		return true;
	if (xattrs->xmin == InvalidTransactionId)
		return false;
	if (xattrs->xmin != FrozenTransactionId)
	{
		for (int i=0; i < xstate->nParallelCurrentXids; i++)
		{
			if (xattrs->xmin == xstate->parallelCurrentXids[i])
				goto xmin_is_visible;
		}
		return false;
	}
xmin_is_visible:
	if (xattrs->xmax == InvalidTransactionId)
		return true;
	if (xattrs->xmax == FrozenTransactionId)
		return false;
	for (int i=0; i < xstate->nParallelCurrentXids; i++)
	{
		if (xattrs->xmax == xstate->parallelCurrentXids[i])
			return false;
	}
	return true;
}

STATIC_FUNCTION(int)
__gpuscan_load_source_row(kern_context *kcxt,
						  kern_warp_context *wp,
						  const kern_data_store *kds_src,
						  const kern_expression *kexp_load_vars,
						  const kern_expression *kexp_scan_quals,
						  const kern_expression *kexp_move_vars,
						  char *dst_kvecs_buffer)
{
	uint32_t	count;
	uint32_t	index;
	uint32_t	wr_pos;
	bool		is_valid = false;

	/* fetch next blockSize tuples */
	index = get_global_size() * wp->smx_row_count + get_global_base();
	if (index >= kds_src->nitems)
	{
		if (get_local_id() == 0)
			wp->scan_done = 1;
		return 1;
	}
	index += get_local_id();

	/*
	 * fetch the outer tuple to scan
	 */
	if (index < kds_src->nitems)
	{
		kern_tupitem *titem = KDS_GET_TUPITEM(kds_src, index);

		if (__gpuscan_check_row_visibility(kcxt, titem) &&
			ExecLoadVarsMinimalTuple(kcxt,
									 kexp_load_vars,
									 0,
									 kds_src,
									 titem))
		{
			xpu_bool_t	retval;

			if (!kexp_scan_quals)
				is_valid = true;
			else if (EXEC_KERN_EXPRESSION(kcxt, kexp_scan_quals, &retval))
			{
				if (!XPU_DATUM_ISNULL(&retval) && retval.value)
					is_valid = true;
			}
			else if (!HandleErrorIfCpuFallback(kcxt, 0, 0, false))
			{
				assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
			}
		}
	}
	/* error checks */
	if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
		return -1;
	/*
	 * save the private kvars slot on the combination buffer (depth=0)
	 */
	wr_pos = WARP_WRITE_POS(wp,0);
	wr_pos += pgstrom_stair_sum_binary(is_valid, &count);
	if (is_valid)
	{
		if (!ExecMoveKernelVariables(kcxt,
									 kexp_move_vars,
									 dst_kvecs_buffer,
									 (wr_pos % KVEC_UNITSZ)))
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
		}
	}
	/* error checks */
	if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
		return -1;
	/* make forward read/write pointer */
	if (get_local_id() == 0)
	{
		wp->smx_row_count++;
		WARP_WRITE_POS(wp,0) += count;
	}
	__syncthreads();
	/* move to the next depth if more than 32 htuples were fetched */
	return (WARP_WRITE_POS(wp,0) >= WARP_READ_POS(wp,0) + get_local_size() ? 1 : 0);
}

PUBLIC_FUNCTION(int)
execGpuScanLoadSource(kern_context *kcxt,
					  kern_warp_context *wp,
					  const kern_data_store *kds_src,
					  const kern_data_extra *kds_extra,
					  const kern_expression *kexp_load_vars,
					  const kern_expression *kexp_scan_quals,
					  const kern_expression *kexp_move_vars,
					  char *dst_kvecs_buffer)
{
	/*
	 * Move to the next depth (or projection), if combination buffer (depth=0)
	 * may overflow on the next action, or we already reached to the KDS tail.
	 */
	if (wp->scan_done > 0 ||
		WARP_WRITE_POS(wp,0) >= WARP_READ_POS(wp,0) + get_local_size())
		return 1;

	/* no source kernel-vectorized buffer for depth==0 */
	kcxt->kvecs_curr_buffer = NULL;
	kcxt->kvecs_curr_id = 0;

	switch (kds_src->format)
	{
		case KDS_FORMAT_BLOCK:
			return __gpuscan_load_source_block(kcxt, wp,
											   kds_src,
											   kexp_load_vars,
											   kexp_scan_quals,
											   kexp_move_vars,
											   dst_kvecs_buffer);
		case KDS_FORMAT_ARROW:
			return __gpuscan_load_source_arrow(kcxt, wp,
											   kds_src,
											   kexp_load_vars,
											   kexp_scan_quals,
											   kexp_move_vars,
											   dst_kvecs_buffer);
		case KDS_FORMAT_ROW:
		case KDS_FORMAT_HASH:
			return __gpuscan_load_source_row(kcxt, wp,
											 kds_src,
											 kexp_load_vars,
											 kexp_scan_quals,
											 kexp_move_vars,
											 dst_kvecs_buffer);
		default:
			STROM_ELOG(kcxt, "Bug? Unknown KDS format");
			break;
	}
	return -1;
}

/* ------------------------------------------------------------
 *
 * Routines to manage GpuCache
 *
 * ------------------------------------------------------------
 */
STATIC_FUNCTION(kern_gpucache_data_store *)
gpucache_lookup_data_store(kern_gpucache_master_state *gc_mstate,
						   uint32_t database_oid,
						   uint32_t table_oid,
						   uint32_t table_sig)
{
	kern_gpucache_data_store *curr;
	struct {
		uint32_t	database_oid;
		uint32_t	table_oid;
		uint32_t	table_sig;
	} hkey;
	uint32_t		hindex;

	hkey.database_oid = database_oid;
	hkey.table_oid    = table_oid;
	hkey.table_sig    = table_sig;
	hindex = pg_hash_any(&hkey, sizeof(hkey)) % GPUCACHE_KDS_HASH_NSLOTS;

	for (curr = gc_mstate->hslots[hindex]; curr; curr = curr->next)
	{
		if (curr->database_oid == database_oid &&
			curr->table_oid    == table_oid &&
			curr->table_sig    == table_sig)
		{
			return curr;
		}
	}
	return NULL;
}

INLINE_FUNCTION(bool)
__apply_one_insert_log(kern_context *kcxt,
					   kern_gpucache_data_store *kds_gc,
					   const kern_tupitem *src_titem)
{
	size_t		required = MAXALIGN(offsetof(kern_hashitem, t) + src_titem->t_len);
	uint64_t	__rowid = __atomic_add_uint32(&kds_gc->kds.nitems, 1);
	uint32_t	__usage = __atomic_add_uint64(&kds_gc->kds.usage, required);
	uint64_t   *hslot;
	kern_hashitem *dst_hitem;

	if (!__KDS_CHECK_OVERFLOW(&kds_gc->kds,
							  __rowid + 1,
							  __usage + required))
	{
		STROM_ELOG(kcxt, "gpucache: out of kds buffer");
		return false;	/* overflow */
	}
	dst_hitem = (kern_hashitem *)((char *)&kds_gc->kds
								  + kds_gc->kds.length
								  - __usage);
	memcpy(&dst_hitem->t, src_titem, src_titem->t_len);
	KERN_TUPITEM_SET_ROWID(&dst_hitem->t, __rowid);
	hslot = KDS_GET_HASHSLOT(&kds_gc->kds, dst_hitem->t.hash);
	dst_hitem->next = __atomic_exchange_uint64(hslot, __usage);
	__threadfence();
	KDS_GET_ROWINDEX(&kds_gc->kds)[__rowid] = (__usage - offsetof(kern_hashitem, t));
	return true;
}

STATIC_FUNCTION(void)
gpucache_apply_insert_logs(kern_context *kcxt,
						   kern_gpucache_master_state *gc_mstate,
						   GpuCacheLogInsert *log)
{
	kern_gpucache_data_store *kds_gc
		= gpucache_lookup_data_store(gc_mstate,
									 log->database_oid,
									 log->table_oid,
									 log->table_sig);
	if (kds_gc)
		__apply_one_insert_log(kcxt, kds_gc, &log->tupitem);
}

#define __GPUCACHE_APPLY_SIMPLE_LOGS_TEMPLATE(NAME,TYPE,FIELD,VALUE)	\
	STATIC_FUNCTION(void)												\
	gpucache_apply_##NAME##_logs(kern_context *kcxt,					\
								 kern_gpucache_master_state *gc_mstate,	\
								 GpuCacheLog##TYPE *log)				\
	{																	\
		kern_gpucache_data_store *kds_gc;								\
		kern_hashitem *hitem;											\
		uint32_t	hash;												\
																		\
		kds_gc = gpucache_lookup_data_store(gc_mstate,					\
											log->database_oid,			\
											log->table_oid,				\
											log->table_sig);			\
		if (!kds_gc)													\
			return;														\
		hash = pg_hash_any(&log->ctid, sizeof(ItemPointerData));		\
		for (hitem = KDS_HASH_FIRST_ITEM(&kds_gc->kds, hash);			\
			 hitem != NULL;												\
			 hitem = KDS_HASH_NEXT_ITEM(&kds_gc->kds, hitem->next))		\
		{																\
			kern_tupitem_xact_attrs *xattrs;							\
																		\
			if (hitem->t.hash != hash)									\
				continue;												\
			xattrs = KERN_TUPITEM_GET_XACT_ATTRS(&hitem->t);			\
			if (ItemPointerEquals(&xattrs->ctid, &log->ctid))			\
			{															\
				xattrs->FIELD = VALUE;									\
			}															\
		}																\
	}
__GPUCACHE_APPLY_SIMPLE_LOGS_TEMPLATE(delete,Delete,xmax,log->xid)
__GPUCACHE_APPLY_SIMPLE_LOGS_TEMPLATE(commit_ins,Xact,xmin,FrozenTransactionId)
__GPUCACHE_APPLY_SIMPLE_LOGS_TEMPLATE(commit_del,Xact,xmax,FrozenTransactionId)
__GPUCACHE_APPLY_SIMPLE_LOGS_TEMPLATE(abort_ins,Xact,xmin,InvalidTransactionId)
__GPUCACHE_APPLY_SIMPLE_LOGS_TEMPLATE(abort_del,Xact,xmax,InvalidTransactionId)

KERNEL_FUNCTION(void)
kern_gpucache_apply_logs(kern_gpucache_master_state *gc_mstate, int phase)
{
	kern_context	kcxt;	/* just for error message */
	uint32_t		index;

	/* bailout if any errors */
	if (__syncthreads_count(gc_mstate->kerror.errcode) > 0)
		return;
	memset(&kcxt, 0, offsetof(kern_context, vlbuf));
	for (index = get_global_id();
		 index < gc_mstate->nitems;
		 index += get_global_size())
	{
		GpuCacheLogCommon *log = (GpuCacheLogCommon *)
			((char *)gc_mstate + gc_mstate->log_items[index]);
		switch (phase)
		{
			case 1:		/* apply INSERT logs */
				if (log->type == GCACHE_TX_LOG__INSERT)
					gpucache_apply_insert_logs(&kcxt, gc_mstate,
											   (GpuCacheLogInsert *)log);
				break;
			case 2:		/* apply DELETE logs */
				if (log->type == GCACHE_TX_LOG__DELETE)
					gpucache_apply_delete_logs(&kcxt, gc_mstate,
											   (GpuCacheLogDelete *)log);
				break;
			case 3:		/* apply XACT logs */
				switch (log->type)
				{
					case GCACHE_TX_LOG__COMMIT_INS:
						gpucache_apply_commit_ins_logs(&kcxt, gc_mstate,
													   (GpuCacheLogXact *)log);
						break;
					case GCACHE_TX_LOG__COMMIT_DEL:
						gpucache_apply_commit_del_logs(&kcxt, gc_mstate,
													   (GpuCacheLogXact *)log);
						break;
					case GCACHE_TX_LOG__ABORT_INS:
						gpucache_apply_abort_ins_logs(&kcxt, gc_mstate,
													  (GpuCacheLogXact *)log);
						break;
					case GCACHE_TX_LOG__ABORT_DEL:
						gpucache_apply_abort_del_logs(&kcxt, gc_mstate,
													  (GpuCacheLogXact *)log);
                        break;
					default:
						break;
				}
				break;
			default:
				STROM_ELOG(&kcxt, "gpucache: unknown phase");
				break;
		}
		if (kcxt.errcode != ERRCODE_STROM_SUCCESS)
			break;
	}
	STROM_WRITEBACK_ERROR_STATUS(&gc_mstate->kerror, &kcxt);
}

KERNEL_FUNCTION(void)
kern_gpucache_compaction(kern_gpucache_data_store *comp,
						 kern_gpucache_data_store *orig)
{
	kern_data_store *kds_src = &orig->kds;
	kern_context	kcxt;	/* just for error message */
	uint32_t		index;

	memset(&kcxt, 0, offsetof(kern_context, vlbuf));
	for (index = get_global_id();
		 index < kds_src->nitems;
		 index += get_global_size())
	{
		kern_tupitem *titem = KDS_GET_TUPITEM(kds_src, index);
		if (titem)
		{
			const kern_tupitem_xact_attrs *xattrs
				= KERN_TUPITEM_GET_XACT_ATTRS(titem);
			if (xattrs &&
				xattrs->xmin != InvalidTransactionId &&
				xattrs->xmax != FrozenTransactionId)
			{
				if (!__apply_one_insert_log(&kcxt, comp, titem))
					break;		/* overflow */
			}
		}
	}
	/*
	 * The compaction process can fail only when the buffer runs out of space.
	 * Therefore, __KDS_CHECK_OVERFLOW always returns false.
	 */
	if (__syncthreads_count(kcxt.errcode != ERRCODE_STROM_SUCCESS) > 0)
	{
		if (get_local_id() == 0)
			assert(!__KDS_CHECK_OVERFLOW(&comp->kds,
										 __volatileRead(&comp->kds.nitems),
										 __volatileRead(&comp->kds.usage)));
	}
}
