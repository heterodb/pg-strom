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





INLINE_FUNCTION(kern_warp_context *)
GPUJOIN_GLOBAL_WARP_CONTEXT(kern_gpujoin *kgjoin)
{
	uint32_t	unitsz = KERN_WARP_CONTEXT_UNITSZ(kgjoin->num_rels);

	return (kern_warp_context *)
		(kgjoin->data + unitsz * (get_global_id() / warpSize));
}

INLINE_FUNCTION(kern_warp_context *)
GPUJOIN_LOCAL_WARP_CONTEXT(int nrels)
{
	uint32_t	unitsz = KERN_WARP_CONTEXT_UNITSZ(nrels);

	return (kern_warp_context *)
		(__pgstrom_dynamic_shared_workmem + unitsz * WarpId());
}









INLINE_FUNCTION(int)
__gpuscan_load_source_row(kern_context *kcxt,
						  kern_warp_context *wp,
						  kern_data_store *kds_src,
						  kern_expression *kern_scan_quals,
						  uint32_t *p_row_count)
{
	HeapTupleHeaderData *htup = NULL;
	uint32_t	count;
	uint32_t	index;
	uint32_t	mask;
	uint32_t	wr_pos;

	/* fetch next warpSize tuples */
	if (LaneId() == 0)
		count = atomicAdd(p_row_count, 1);
	count = __shfl_sync(__activemask(), count, 0);
	index = (get_num_groups() * count + get_group_id()) * warpSize;
	if (index >= kds_src->nitems)
	{
		wp->scan_done = true;
		__syncwarp(__activemask());
		return 1;
	}
	index += LaneId();

	if (index < kds_src->nitems)
	{
		uint32_t	offset = KDS_GET_ROWINDEX(kds_src)[index];
		xpu_bool_t	retval;
		kern_tupitem *tupitem;

		assert(offset <= kds_src->usage);
		kcxt_reset(kcxt);
		tupitem = (kern_tupitem *)((char *)kds_src +
								   kds_src->length -
								   __kds_unpack(offset));
		assert((char *)tupitem >= (char *)kds_src &&
			   (char *)tupitem <  (char *)kds_src + kds_src->length);
		if (kern_scan_quals == NULL ||
			ExecLoadVarsOuterRow(kcxt,
								 kern_scan_quals,
								 (xpu_datum_t *)&retval,
								 kds_src,
								 &tupitem->htup,
								 0, NULL, NULL))
		{
			if (!retval.isnull && retval.value)
				htup = &tupitem->htup;
		}
		else
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
		}
	}
	/* error checks */
	if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
		return -1;
	/*
	 * save the htuple on the local combination buffer (depth=0)
	 */
	mask = __ballot_sync(__activemask(), htup != NULL);
	if (LaneId() == 0)
	{
		wr_pos = WARP_WRITE_POS(wp,0);
		WARP_WRITE_POS(wp,0) += __popc(mask);
	}
	wr_pos = __shfl_sync(__activemask(), wr_pos, 0);
	mask &= ((1U << LaneId()) - 1);
	wr_pos += __popc(mask);
	if (htup != NULL)
	{
		WARP_COMB_BUF(wp)[wr_pos % UNIT_TUPLES_PER_WARP]
			= __kds_packed((char *)htup - (char *)kds_src);
	}
	/* move to the next depth if more than 32 htuples were fetched */
	return (WARP_WRITE_POS(wp,0) >= WARP_READ_POS(wp,0) + warpSize ? 1 : 0);
}

/*
 * __gpuscan_load_source_block
 */
STATIC_FUNCTION(int)
__gpuscan_load_source_block(kern_context *kcxt,
							kern_warp_context *wp,
							kern_data_store *kds_src,
							kern_expression *kern_scan_quals,
							uint32_t *p_row_count)
{
	uint32_t	block_id = __shfl_sync(__activemask(), wp->block_id, 0);
	uint32_t	wr_pos = __shfl_sync(__activemask(), wp->lp_wr_pos, 0);
	uint32_t	rd_pos = __shfl_sync(__activemask(), wp->lp_rd_pos, 0);
	uint32_t	count;
	uint32_t	mask;

	assert(wr_pos >= rd_pos);
	if (block_id > kds_src->nitems || wr_pos >= rd_pos + warpSize)
	{
		HeapTupleHeaderData *htup = NULL;
		xpu_bool_t	retval;
		uint32_t	off;

		kcxt_reset(kcxt);
		rd_pos += LaneId();
		htup = NULL;
		if (rd_pos < wr_pos)
		{
			off = wp->lp_items[rd_pos % UNIT_TUPLES_PER_WARP];
			htup = (HeapTupleHeaderData *)((char *)kds_src + __kds_unpack(off));
			if (kern_scan_quals == NULL ||
				ExecLoadVarsOuterRow(kcxt,
									 kern_scan_quals,
									 (xpu_datum_t *)&retval,
									 kds_src,
									 htup,
									 0, NULL, NULL))
			{
				if (retval.isnull || !retval.value)
					htup = NULL;
			}
			else
			{
				assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
			}
		}
		/* error checks */
		if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
			return -1;
		if (LaneId() == 0)
			wp->lp_rd_pos = Min(wp->lp_wr_pos,
								wp->lp_rd_pos + warpSize);
		/*
		 * save the htupls
		 */
		mask = __ballot_sync(__activemask(), htup != NULL);
		if (LaneId() == 0)
		{
			wr_pos = WARP_WRITE_POS(wp,0);
			WARP_WRITE_POS(wp,0) += __popc(mask);
		}
		wr_pos = __shfl_sync(__activemask(), wr_pos, 0);
        mask &= ((1U << LaneId()) - 1);
		wr_pos += __popc(mask);
		if (htup != NULL)
		{
			WARP_COMB_BUF(wp,0)[wr_pos % UNIT_TUPLES_PER_WARP]
				= __kds_packed((char *)htup - (char *)kds_src);
		}
		/* end-of-scan checks */
		if (block_id > kds_src->nitems &&	/* no more blocks to fetch */
			wp->lp_rd_pos >= wp->lp_wr_pos)	/* no more pending tuples  */
		{
			if (LaneId() == 0)
				wp->scan_done = true;
			return 1;
		}
		/* move to the next depth if more than 32 htuples were fetched */
		return (WARP_WRITE_POS(wp,0) >= WARP_READ_POS(wp,0) + warpSize ? 1 : 0);
	}

	/*
	 * Here, number of pending tuples (which is saved in the lp_items[]) is
	 * not enough to run ScanQuals checks. So, we move to the next bunch of
	 * line-items or next block.
	 * The pending tuples just passed the MVCC visivility checks, but
	 * ScanQuals check is not applied yet. We try to run ScanQuals checks
	 * with 32 threads simultaneously.
	 */
	if (block_id == 0)
	{
		/*
		 * block_id == 0 means this warp is not associated with particular
		 * block-page, so we try to fetch the next page.
		 */
		if (LaneId() == 0)
			count = atomicAdd(p_row_count, 1);
		count = __shfl_sync(__activemask(), count, 0);
		block_id = (get_num_groups() * count + get_group_id()) + 1;
		if (LaneId() == 0)
			wp->block_id = block_id;
	}
	if (block_id <= kds_src->nitems)
	{
		PageHeaderData *pg_page = KDS_BLOCK_PGPAGE(kds_src, block_id-1);
		HeapTupleHeaderData *htup = NULL;

		count = __shfl_sync(__activemask(), wp->lp_count, 0);
		if (count < PageGetMaxOffsetNumber(pg_page))
		{
			count += LaneId();
			if (count < PageGetMaxOffsetNumber(pg_page))
			{
				ItemIdData *lpp = &pg_page->pd_linp[count];

				assert((char *)lpp < (char *)pg_page + BLCKSZ);
				if (ItemIdIsNormal(lpp))
					htup = (HeapTupleHeaderData *)PageGetItem(pg_page, lpp);
				else
					htup = NULL;
			}
			/* put visible tuples on the lp_items[] array */
			mask = __ballot_sync(__activemask(), htup != NULL);
			if (LaneId() == 0)
			{
				wr_pos = wp->lp_wr_pos;
				wp->lp_wr_pos += __popc(mask);
			}
			wr_pos = __shfl_sync(__activemask(), wr_pos, 0);
			mask &= ((1U << LaneId()) - 1);
			wr_pos += __popc(mask);
			if (htup != NULL)
			{
				wp->lp_items[wr_pos % UNIT_TUPLES_PER_WARP]
					= __kds_packed((char *)htup - (char *)kds_src);
			}
			if (LaneId() == 0)
				wp->lp_count += warpSize;
		}
		else
		{
			/* no more tuples to fetch from the current page */
			if (LaneId() == 0)
			{
				wp->block_id = 0;
				wp->lp_count = 0;
			}
			__syncwarp(__activemask());
		}
	}
	return 0;	/* stay depth-0 */
}

/*
 * __gpuscan_load_source_arrow
 */
INLINE_FUNCTION(int)
__gpuscan_load_source_arrow(kern_context *kcxt,
							kern_warp_context *wp,
							kern_data_store *kds_src,
							kern_expression *kern_scan_quals,
							uint32_t *p_row_count)
{
	uint32_t	count;
	uint32_t	index;
	uint32_t	mask;
	uint32_t	wr_pos;
	bool		is_valid = false;

	/* fetch next warpSize tuples */
	if (LaneId() == 0)
		count = atomicAdd(p_row_count, 1);
	count = __shfl_sync(__activemask(), count, 0);
	index = (get_num_groups() * count + get_group_id()) * warpSize;
	if (index >= kds_src->nitems)
	{
		wp->scan_done = true;
		__syncwarp(__activemask());
		return 1;
	}
	index += LaneId();

	if (index < kds_src->nitems)
	{
		xpu_bool_t	retval;

		if (kern_scan_quals == NULL ||
			ExecLoadVarsOuterArrow(kcxt,
								   kern_scan_quals,
								   (xpu_datum_t *)&retval,
								   kds_src,
								   index,
								   0, NULL, NULL))
		{
			if (!retval.isnull && retval.value)
				is_valid = true;
		}
		else
		{
			assert(kcxt->errcode != 0);
		}
	}
	/* error checks */
	if (__any_sync(__activemask(), kcxt->errcode != 0))
		return -1;
	/*
	 * save the htuple on the local combination buffer (depth=0)
	 */
	mask = __ballot_sync(__activemask(), is_valid);
	if (LaneId() == 0)
	{
		wr_pos = WARP_WRITE_POS(wp,0);
		WARP_WRITE_POS(wp,0) += __popc(mask);
	}
	wr_pos = __shfl_sync(__activemask(), wr_pos, 0);
	mask &= ((1U << LaneId()) - 1);
	wr_pos += __popc(mask);
	if (is_valid)
		WARP_COMB_BUF(wp)[wr_pos % UNIT_TUPLES_PER_WARP] = index;
	/* move to the next depth if more than 32 htuples were fetched */
	return (WARP_WRITE_POS(wp,0) >= WARP_READ_POS(wp,0) + warpSize ? 1 : 0);
}

/*
 * __gpuscan_load_source_column
 */
INLINE_FUNCTION(int)
__gpuscan_load_source_column(kern_context *kcxt,
							 kern_warp_context *wp,
							 kern_data_store *kds_src,
							 kern_data_extra *kds_extra,
							 kern_expression *kern_scan_quals,
							 uint32_t *p_row_count)
{
	STROM_ELOG(kcxt, "KDS_FORMAT_COLUMN not implemented");
	return -1;
}

PUBLIC_FUNCTION(int)
gpuscan_load_source(kern_context *kcxt,
					kern_warp_context *wp,
					kern_data_store *kds_src,
					kern_data_extra *kds_extra,
					kern_expression *kern_scan_quals,
					uint32_t *p_row_count)
{
	/*
	 * Move to the next depth (or projection), if combination buffer (depth=0)
	 * may overflow on the next action, or we already reached to the KDS tail.
	 */
	assert(WARP_WRITE_POS(wp,0) >= WARP_READ_POS(wp,0));
	if (wp->scan_done || WARP_WRITE_POS(wp,0) >= WARP_READ_POS(wp,0) + warpSize)
		return 1;

	switch (kds_src->format)
	{
		case KDS_FORMAT_ROW:
			return __gpuscan_load_source_row(kcxt, wp, kds_src,
											 kern_scan_quals,
											 p_row_count);
		case KDS_FORMAT_BLOCK:
			return __gpuscan_load_source_block(kcxt, wp, kds_src,
											   kern_scan_quals,
											   p_row_count);
		case KDS_FORMAT_ARROW:
			return __gpuscan_load_source_arrow(kcxt, wp, kds_src,
											   kern_scan_quals,
											   p_row_count);
		case KDS_FORMAT_COLUMN:
			return __gpuscan_load_source_column(kcxt, wp, kds_src, kds_extra,
												kern_scan_quals,
												p_row_count);
		default:
			STROM_ELOG(kcxt, "Bug? Unknown KDS format");
			break;
	}
	return -1;
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
	kern_warp_context *wp = GPUJOIN_LOCAL_WARP_CONTEXT(kmrels->num_rels);
	kern_context   *kcxt;
	int				depth;
	__shared__ static uint32_t row_count;

	INIT_KERNEL_CONTEXT(kcxt, session);
	/* sanity checks */
	assert(kgjoin->num_rels == kmrels->num_rels);
	/* resume the previous execution context */
	if (get_local_id() == 0)
		row_count = 0;
	__syncthreads();
	if (LaneId() == 0)
	{
		uint32_t	unitsz = KERN_WARP_CONTEXT_UNITSZ(kmrels->num_rels);
		uint32_t	offset = unitsz * (get_global_id() / warpSize);

		memcpy(wp, kgjoin->data + offset, unitsz);
		wp->nrels = kmrels->num_rels;
	}
	__syncthreads();

	/* main logic of GpuJoin */
	while ((depth = __shfl_sync(__activemask(), wp->depth, 0)) >= 0)
	{
		if (depth == 0)
		{
			/* LOAD FROM THE SOURCE */
			depth = gpuscan_load_source(kcxt, wp,
										kds_src,
										kds_extra,
										SESSION_KEXP_SCAN_QUALS(kcxt->session),
										&row_count);
			if (__any_sync(__activemask(), depth < 0) != 0)
			{
				assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
				depth = -1;		/* bailout */
			}
		}
		else if (depth > kmrels->num_rels)
		{
			/* PROJECTION */

		}
		else if (kmrels->chunks[depth-1].is_nestloop)
		{
			/* NEST-LOOP */

		}
#if 0
		else if (kmrels->chunks[depth-1].gist_offset != 0)
		{
			/* GiST-INDEX-JOIN */
		}
#endif
		else
		{
			/* HASH-JOIN */


		}
	}

	/* suspend the execution context */
	if (LaneId() == 0)
	{
		uint32_t    unitsz = KERN_WARP_CONTEXT_UNITSZ(kmrels->num_rels);
		uint32_t    offset = unitsz * (get_global_id() / warpSize);

		memcpy(kgjoin->data + offset, wp, unitsz);
	}
}
