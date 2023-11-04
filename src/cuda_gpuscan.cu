/*
 * cuda_gpuscan.cu
 *
 * Device implementation of GpuScan
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "cuda_common.h"

/*
 * pgstrom_stair_sum_xxxx
 */
static __shared__ union {
	uint32_t	u32[MAXTHREADS_PER_BLOCK / WARPSIZE];
	uint64_t	u64[MAXTHREADS_PER_BLOCK / WARPSIZE];
	int32_t		i32[MAXTHREADS_PER_BLOCK / WARPSIZE];
	int64_t		i64[MAXTHREADS_PER_BLOCK / WARPSIZE];
	float8_t	fp64[MAXTHREADS_PER_BLOCK / WARPSIZE];
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

PUBLIC_FUNCTION(uint32_t)
pgstrom_stair_sum_binary(bool predicate, uint32_t *p_total_count)
{
	uint32_t	n_warps = get_local_size() / warpSize;
	uint32_t	warp_id = get_local_id()   / warpSize;
	uint32_t	mask;
	uint32_t	sum;

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
		assert(__activemask() == ~0U);									\
		warp_sum = __stair_sum_warp_common(value);						\
		assert(warp_sum >= value);										\
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

/* ----------------------------------------------------------------
 *
 * execGpuScanLoadSource and related
 *
 * ----------------------------------------------------------------
 */
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
	kern_tupitem *tupitem = NULL;

	/* compute the next row-index */
	count = wp->smx_row_count;
	__syncthreads();
	if (get_local_id() == 0)
		wp->smx_row_count++;
	index = get_global_size() * count + get_global_base();
	if (index >= kds_src->nitems)
	{
		if (get_local_id() == 0)
			wp->scan_done = 1;
		__syncthreads();
		return 1;
	}
	index += get_local_id();

	/*
	 * fetch the outer tuple to scan
	 */
	if (index < kds_src->nitems)
	{
		uint32_t	offset = KDS_GET_ROWINDEX(kds_src)[index];

		assert(offset <= kds_src->usage);
		tupitem = (kern_tupitem *)((char *)kds_src +
								   kds_src->length -
								   __kds_unpack(offset));
		assert((char *)tupitem >= (char *)kds_src &&
			   (char *)tupitem <  (char *)kds_src + kds_src->length);
		if (!ExecLoadVarsOuterRow(kcxt,
								  kexp_load_vars,
								  kexp_scan_quals,
								  kds_src,
								  &tupitem->htup))
			tupitem = NULL;
	}
	/* error checks */
	if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
		return -1;

	/*
	 * save the private kvars slot on the combination buffer (depth=0)
	 */
	wr_pos = WARP_WRITE_POS(wp,0);
	wr_pos += pgstrom_stair_sum_binary(tupitem != NULL, &count);
	if (get_local_id() == 0)
		WARP_WRITE_POS(wp,0) += count;
	if (tupitem != NULL)
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
	/* move to the next depth, if more than blockSize tuples were fetched. */
	return (WARP_WRITE_POS(wp,0) >= WARP_READ_POS(wp,0) + get_local_size() ? 1 : 0);
}

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
			htup = (HeapTupleHeaderData *)((char *)kds_src + __kds_unpack(off));
			if (!ExecLoadVarsOuterRow(kcxt,
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
			= __kds_packed((char *)htup - (char *)kds_src);
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
	count = wp->smx_row_count;
	__syncthreads();
	if (get_local_id() == 0)
		wp->smx_row_count++;
	index = get_global_size() * count + get_global_base();
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
	if (get_local_id() == 0)
		WARP_WRITE_POS(wp,0) += count;
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
	/* move to the next depth, if more than blockSize rows were fetched. */
	return (WARP_WRITE_POS(wp,0) >= WARP_READ_POS(wp,0) + get_local_size() ? 1 : 0);
}

/*
 * __gpuscan_load_source_column (KDS_FORMAT_COLUMN)
 */
INLINE_FUNCTION(GpuCacheSysattr *)
kds_column_get_sysattr(const kern_data_store *kds, uint32_t rowid)
{
	const kern_colmeta *cmeta = &kds->colmeta[kds->nr_colmeta - 1];
	GpuCacheSysattr	   *base;

	assert(!cmeta->attbyval &&
		   cmeta->attalign == sizeof(uint32_t) &&
		   cmeta->attlen == sizeof(GpuCacheSysattr) &&
		   cmeta->nullmap_offset == 0);
	base = (GpuCacheSysattr *)
		((char *)kds + __kds_unpack(cmeta->values_offset));
	if (rowid < kds->column_nrooms)
		return &base[rowid];
	return NULL;
}

STATIC_FUNCTION(bool)
kds_column_check_visibility(kern_context *kcxt,
							const kern_data_store *kds,
							uint32_t rowid)
{
	SerializedTransactionState *xstate = SESSION_XACT_STATE(kcxt->session);
	GpuCacheSysattr *sysattr = kds_column_get_sysattr(kds, rowid);

	assert(xstate != NULL && sysattr != NULL);

	if (sysattr->xmin == InvalidTransactionId)
		return false;
	if (sysattr->xmin != FrozenTransactionId)
	{
		for (int i=0; i < xstate->nParallelCurrentXids; i++)
		{
			if (sysattr->xmin == xstate->parallelCurrentXids[i])
				goto xmin_is_visible;
		}
		return false;
	}
xmin_is_visible:
	if (sysattr->xmax == InvalidTransactionId)
		return true;
	if (sysattr->xmax == FrozenTransactionId)
		return false;
	for (int i=0; i < xstate->nParallelCurrentXids; i++)
	{
		if (sysattr->xmax == xstate->parallelCurrentXids[i])
			return false;
	}
	return true;
}

STATIC_FUNCTION(int)
__gpuscan_load_source_column(kern_context *kcxt,
							 kern_warp_context *wp,
							 const kern_data_store *kds_src,
							 const kern_data_extra *kds_extra,
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
	count = wp->smx_row_count;
	__syncthreads();
	if (get_local_id() == 0)
		wp->smx_row_count++;
	index = get_global_size() * count + get_global_base();
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
	if (index < kds_src->nitems &&
		kds_column_check_visibility(kcxt, kds_src, index))
	{
		if (ExecLoadVarsOuterColumn(kcxt,
									kexp_load_vars,
									kexp_scan_quals,
									kds_src,
									kds_extra,
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
	if (get_local_id() == 0)
		WARP_WRITE_POS(wp,0) += count;
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
		case KDS_FORMAT_ROW:
			return __gpuscan_load_source_row(kcxt, wp,
											 kds_src,
											 kexp_load_vars,
											 kexp_scan_quals,
											 kexp_move_vars,
											 dst_kvecs_buffer);
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
		case KDS_FORMAT_COLUMN:
			return __gpuscan_load_source_column(kcxt, wp,
												kds_src,
												kds_extra,
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
STATIC_FUNCTION(void)
gpucache_cleanup_row_owner(kern_context *kcxt,
						   kern_gpucache_redolog *redo,
						   kern_data_store *kds)
{
	uint32_t	owner_id;

	for (owner_id = get_global_id();
		 owner_id < redo->nitems;
		 owner_id += get_global_size())
	{
		GCacheTxLogCommon *tx_log;
		GpuCacheSysattr *sysattr;
		uint32_t		offset = redo->redo_items[owner_id];
		uint32_t		rowid;

		tx_log = (GCacheTxLogCommon *)
			((char *)redo + __kds_unpack(offset));
		switch (tx_log->type)
		{
			case GCACHE_TX_LOG__INSERT:
				rowid = ((GCacheTxLogInsert *)tx_log)->rowid;
				break;
			case GCACHE_TX_LOG__DELETE:
				rowid = ((GCacheTxLogDelete *)tx_log)->rowid;
				break;
			case GCACHE_TX_LOG__COMMIT_INS:
			case GCACHE_TX_LOG__COMMIT_DEL:
			case GCACHE_TX_LOG__ABORT_INS:
			case GCACHE_TX_LOG__ABORT_DEL:
				rowid = ((GCacheTxLogXact *)tx_log)->rowid;
				break;
			default:
				STROM_ELOG(kcxt, "unknown GCacheTxLog type");
				return;
		}
		assert(rowid < kds->column_nrooms);
		sysattr = kds_column_get_sysattr(kds, rowid);
		sysattr->owner = 0;
	}
}

STATIC_FUNCTION(void)
gpucache_assign_update_owner(kern_context *kcxt,
							 kern_gpucache_redolog *redo,
							 kern_data_store *kds)
{
	uint32_t	owner_id;

	for (owner_id = get_global_id();
		 owner_id < redo->nitems;
		 owner_id += get_global_size())
	{
		GCacheTxLogCommon *tx_log;
		GpuCacheSysattr *sysattr;
		uint32_t		offset = redo->redo_items[owner_id];
		uint32_t		rowid;

		tx_log = (GCacheTxLogCommon *)
			((char *)redo + __kds_unpack(offset));
		if (tx_log->type == GCACHE_TX_LOG__INSERT)
		{
			rowid = ((GCacheTxLogInsert *)tx_log)->rowid;
			sysattr = kds_column_get_sysattr(kds, rowid);
			__atomic_max_uint32(&sysattr->owner, owner_id);
		}
		else if (tx_log->type == GCACHE_TX_LOG__DELETE)
		{
			rowid = ((GCacheTxLogDelete *)tx_log)->rowid;
			sysattr = kds_column_get_sysattr(kds, rowid);
			__atomic_max_uint32(&sysattr->owner, owner_id);			
		}
	}
}

STATIC_FUNCTION(bool)
__gpucache_apply_insert_log(kern_context *kcxt,
							kern_data_store *kds,
							kern_data_extra *extra,
							GpuCacheSysattr *sysattr,
							const GCacheTxLogInsert *i_log)
{
	const HeapTupleHeaderData *htup = &i_log->htup;
	uint32_t	rowid = i_log->rowid;
	bool		heap_hasnull = ((htup->t_infomask & HEAP_HASNULL) != 0);
	uint32_t	offset = htup->t_hoff;
	int			j, ncols = Min(kds->ncols, (htup->t_infomask2 & HEAP_NATTS_MASK));

	for (j=0; j < ncols; j++)
	{
		const kern_colmeta *cmeta = &kds->colmeta[j];
		char	   *base;

		if (cmeta->nullmap_offset != 0)
		{
			uint32_t   *nullmap = (uint32_t *)
				((char *)kds + __kds_unpack(cmeta->nullmap_offset));

			if (heap_hasnull && att_isnull(j, htup->t_bits))
			{
				__atomic_and_uint32(&nullmap[rowid>>5], ~(1U<<(rowid&31)));
				continue;
			}
			else
			{
				__atomic_or_uint32(&nullmap[rowid>>5], (1U<<(rowid&31)));
			}
		}
		else
		{
			if (heap_hasnull && att_isnull(j, htup->t_bits))
			{
				STROM_ELOG(kcxt, "NULL appeared at not-null column");
				return false;
			}
		}

		assert(cmeta->values_offset != 0);
		base = (char *)kds + __kds_unpack(cmeta->values_offset);
		if (cmeta->attlen > 0)
		{
			offset = TYPEALIGN(cmeta->attalign, offset);
			memcpy(base + cmeta->attlen * rowid,
				   (char *)htup + offset,
				   cmeta->attlen);
			offset += cmeta->attlen;
		}
		else
		{
			char	   *vl_pos;
			uint32_t	vl_len;
			uint32_t	vl_off;

			assert(cmeta->attlen == -1);
			if (!VARATT_NOT_PAD_BYTE((char *)htup + offset))
				offset = TYPEALIGN(cmeta->attalign, offset);
			vl_pos = (char *)htup + offset;
			vl_len = VARSIZE_ANY(vl_pos);
			vl_off = __atomic_add_uint64(&extra->usage, MAXALIGN(vl_len));
			if (vl_off + vl_len > extra->length)
			{
				STROM_EREPORT(kcxt, ERRCODE_BUFFER_NO_SPACE,
							  "gpucache: extra buffer has no space");
				return false;
			}
			memcpy((char *)extra + vl_off,
				   (char *)htup + offset,
				   vl_len);
			((uint32_t *)base)[rowid] = __kds_packed(vl_off);
			offset += vl_len;
		}
	}
	sysattr->xmin = htup->t_choice.t_heap.t_xmin;
	sysattr->xmax = htup->t_choice.t_heap.t_xmax;
	memcpy(&sysattr->ctid, &htup->t_ctid, sizeof(ItemPointerData));
	
	return true;
}

STATIC_FUNCTION(void)
gpucache_apply_update_logs(kern_context *kcxt,
						   kern_gpucache_redolog *redo,
						   kern_data_store *kds,
						   kern_data_extra *extra)
{
	__shared__ uint32_t smx_rowid_max;
	uint32_t	rowid_max = UINT_MAX;
	uint32_t	owner_id;

	if (get_local_id() == 0)
		smx_rowid_max = 0;
	__syncthreads();

	for (owner_id = get_global_id();
		 owner_id < redo->nitems;
		 owner_id += get_global_size())
	{
		GCacheTxLogCommon *tx_log;
		GpuCacheSysattr *sysattr;
		uint32_t		offset = redo->redo_items[owner_id];

		tx_log = (GCacheTxLogCommon *)
			((char *)redo + __kds_unpack(offset));
		if (tx_log->type == GCACHE_TX_LOG__INSERT)
		{
			GCacheTxLogInsert  *i_log = (GCacheTxLogInsert *)tx_log;

			sysattr = kds_column_get_sysattr(kds, i_log->rowid);
			if (sysattr->owner == owner_id)
			{
				__gpucache_apply_insert_log(kcxt, kds, extra, sysattr, i_log);
				if (rowid_max == UINT_MAX || rowid_max < i_log->rowid)
					rowid_max = i_log->rowid;
			}
		}
		else if (tx_log->type == GCACHE_TX_LOG__DELETE)
		{
			GCacheTxLogDelete  *d_log = (GCacheTxLogDelete *)tx_log;

			sysattr = kds_column_get_sysattr(kds, d_log->rowid);
			if (sysattr->owner == owner_id)
			{
				sysattr->xmax = d_log->xid;
				if (rowid_max == UINT_MAX || rowid_max < d_log->rowid)
					rowid_max = d_log->rowid;
			}
		}
	}
	/* update kds->nitems */
	if (rowid_max != UINT_MAX)
		__atomic_max_uint32(&smx_rowid_max, rowid_max);
	if (__syncthreads_count(rowid_max != UINT_MAX) > 0 && get_local_id() == 0)
		__atomic_max_uint32(&kds->nitems, smx_rowid_max+1);
}

STATIC_FUNCTION(void)
gpucache_assign_xact_owner(kern_context *kcxt,
							 kern_gpucache_redolog *redo,
							 kern_data_store *kds)
{
	uint32_t	owner_id;

	for (owner_id = get_global_id();
		 owner_id < redo->nitems;
		 owner_id += get_global_size())
	{
		GCacheTxLogCommon *tx_log;
		GpuCacheSysattr *sysattr;
		uint32_t		offset = redo->redo_items[owner_id];
		uint32_t		rowid;

		tx_log = (GCacheTxLogCommon *)
			((char *)redo + __kds_unpack(offset));
		if (tx_log->type == GCACHE_TX_LOG__COMMIT_INS ||
			tx_log->type == GCACHE_TX_LOG__COMMIT_DEL ||
			tx_log->type == GCACHE_TX_LOG__ABORT_INS ||
			tx_log->type == GCACHE_TX_LOG__ABORT_DEL)
		{
			rowid = ((GCacheTxLogXact *)tx_log)->rowid;
			 sysattr = kds_column_get_sysattr(kds, rowid);
            __atomic_max_uint32(&sysattr->owner, owner_id);
		}
	}
}

STATIC_FUNCTION(uint64_t)
__gpucache_count_deadspace(kern_data_store *kds,
						   kern_data_extra *extra,
						   uint32_t rowid)
{
	uint64_t	retval = 0;

	if (kds->has_varlena)
	{
		assert(rowid < kds->column_nrooms);
		for (int j=0; j < kds->ncols; j++)
		{
			const kern_colmeta *cmeta = &kds->colmeta[j];

			if (cmeta->attlen > 0)
				continue;
			assert(cmeta->attlen == -1);
			if (!KDS_COLUMN_ITEM_ISNULL(kds, cmeta, rowid))
			{
				uint32_t   *base = (uint32_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				char	   *vl = (char *)extra + __kds_unpack(base[rowid]);

				retval += MAXALIGN(VARSIZE_ANY(vl));
			}
		}
	}
	return retval;
}


STATIC_FUNCTION(void)
gpucache_apply_xact_logs(kern_context *kcxt,
						 kern_gpucache_redolog *redo,
						 kern_data_store *kds,
						 kern_data_extra *extra)
{
	__shared__ uint32_t smx_rowid_max;
	__shared__ uint64_t smx_deadspace;
	uint32_t	rowid_max = UINT_MAX;
	uint32_t	owner_id;
	uint64_t	sz;

	if (get_local_id() == 0)
	{
		smx_rowid_max = 0;
		smx_deadspace = 0;
	}
	__syncthreads();

	for (owner_id = get_global_id();
		 owner_id < redo->nitems;
		 owner_id += get_global_size())
	{
		GCacheTxLogXact *tx_log;
		GpuCacheSysattr *sysattr;
		uint32_t		offset = redo->redo_items[owner_id];

		tx_log = (GCacheTxLogXact *)
			((char *)redo + __kds_unpack(offset));
		switch (tx_log->type)
		{
			case GCACHE_TX_LOG__COMMIT_INS:
				sysattr = kds_column_get_sysattr(kds, tx_log->rowid);
				if (sysattr->owner == owner_id)
				{
					sysattr->xmin = FrozenTransactionId;
					if (rowid_max == UINT_MAX || rowid_max < tx_log->rowid)
						rowid_max = tx_log->rowid;
				}
				break;
			case GCACHE_TX_LOG__COMMIT_DEL:
				sysattr = kds_column_get_sysattr(kds, tx_log->rowid);
				if (sysattr->owner == owner_id)
				{
					sysattr->xmax = FrozenTransactionId;
					if (rowid_max == UINT_MAX || rowid_max < tx_log->rowid)
						rowid_max = tx_log->rowid;
					sz = __gpucache_count_deadspace(kds, extra, tx_log->rowid);
					if (sz > 0)
						__atomic_add_uint64(&smx_deadspace, sz);
				}
				break;
			case GCACHE_TX_LOG__ABORT_INS:
				sysattr = kds_column_get_sysattr(kds, tx_log->rowid);
				if (sysattr->owner == owner_id)
				{
					sysattr->xmin = InvalidTransactionId;
					if (rowid_max == UINT_MAX || rowid_max < tx_log->rowid)
						rowid_max = tx_log->rowid;
					sz = __gpucache_count_deadspace(kds, extra, tx_log->rowid);
					if (sz > 0)
						__atomic_add_uint64(&smx_deadspace, sz);
				}
				break;
			case GCACHE_TX_LOG__ABORT_DEL:
				sysattr = kds_column_get_sysattr(kds, tx_log->rowid);
				if (sysattr->owner == owner_id)
				{
					sysattr->xmax = InvalidTransactionId;
					if (rowid_max == UINT_MAX || rowid_max < tx_log->rowid)
						rowid_max = tx_log->rowid;
				}
				break;
			default:
				break;
		}
	}
	/* update kds->nitems */
	if (rowid_max != UINT_MAX)
		__atomic_max_uint32(&smx_rowid_max, rowid_max);
	if (__syncthreads_count(rowid_max != UINT_MAX) > 0 && get_local_id() == 0)
		__atomic_max_uint32(&kds->nitems, smx_rowid_max+1);
	/* update extra->deadspace */
	if (get_local_id() == 0 && smx_deadspace > 0)
	{
		assert(extra != NULL);
		__atomic_add_uint64(&extra->deadspace, smx_deadspace);
	}
}

KERNEL_FUNCTION(void)
kern_gpucache_apply_redo(kern_gpucache_redolog *gcache_redo,
						 kern_data_store *kds,
						 kern_data_extra *extra,
						 int phase)
{
	kern_context	kcxt;	/* just for error message */

	/* bailout if any errors */
	if (__syncthreads_count(gcache_redo->kerror.errcode) > 0)
		return;

	memset(&kcxt, 0, offsetof(kern_context, vlbuf));
	switch (phase)
	{
		case 1:		/* clean up the owner_id of sysattr */
			gpucache_cleanup_row_owner(&kcxt, gcache_redo, kds);
			break;
		case 2:		/* assign the largest owner_id of INS/DEL log entries */
			gpucache_assign_update_owner(&kcxt, gcache_redo, kds);
			break;
		case 3:		/* apply INS/DEL log entries */
			gpucache_apply_update_logs(&kcxt, gcache_redo, kds, extra);
			break;
		case 4:		/* clean up the owner_id of sysattr */
			gpucache_cleanup_row_owner(&kcxt, gcache_redo, kds);
            break;
		case 5:		/* assign the largest owner_id of XACT log entries */
			gpucache_assign_xact_owner(&kcxt, gcache_redo, kds);
			break;
		case 6:		/* apply XACT log entries */
			gpucache_apply_xact_logs(&kcxt, gcache_redo, kds, extra);
			break;
		default:
			STROM_ELOG(&kcxt, "gpucache: unknown phase");
			break;
	}
	STROM_WRITEBACK_ERROR_STATUS(&gcache_redo->kerror, &kcxt);
}

KERNEL_FUNCTION(void)
kern_gpucache_compaction(kern_data_store *kds,
						 kern_data_extra *extra_src,
						 kern_data_extra *extra_dst)
{
	uint32_t	index;

	for (index = get_global_id();
		 index < kds->nitems;
		 index += get_global_size())
	{
		for (int j=0; j < kds->ncols; j++)
		{
			kern_colmeta   *cmeta = &kds->colmeta[j];
			uint32_t	   *values;
			char		   *vl_src;
			uint32_t		vl_len;
			uint64_t		offset;

			if (cmeta->attlen >= 0)
				continue;
			if (cmeta->nullmap_offset != 0)
			{
				uint8_t	   *nullmap = (uint8_t *)
					((char *)kds + __kds_unpack(cmeta->nullmap_offset));

				if (att_isnull(index, nullmap))
					continue;
			}
			values = (uint32_t *)
				((char *)kds + __kds_unpack(cmeta->values_offset));
			vl_src = ((char *)extra_src + __kds_unpack(values[index]));
			vl_len = VARSIZE_ANY(vl_src);

			offset = __atomic_add_uint64(&extra_dst->usage, MAXALIGN(vl_len));
			if (offset + vl_len <= extra_dst->length)
			{
				memcpy((char *)extra_dst + offset, vl_src, vl_len);
				values[index] = __kds_packed(offset);
			}
		}
	}
}
