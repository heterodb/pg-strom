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

/* ----------------------------------------------------------------
 *
 * execGpuScanLoadSource and related
 *
 * ----------------------------------------------------------------
 */
STATIC_FUNCTION(int)
__gpuscan_load_source_row(kern_context *kcxt,
						  kern_warp_context *wp,
						  kern_data_store *kds_src,
						  kern_expression *kexp_load_vars,
						  kern_expression *kexp_scan_quals,
						  char *kvars_addr_wp,
						  uint32_t *p_smx_row_count)
{
	uint32_t	count;
	uint32_t	index;
	uint32_t	mask;
	uint32_t	wr_pos;
	kern_tupitem *tupitem = NULL;

	/* fetch next warpSize tuples */
	if (LaneId() == 0)
		count = atomicAdd(p_smx_row_count, 1);
	count = __shfl_sync(__activemask(), count, 0);
	index = (get_num_groups() * count + get_group_id()) * warpSize;
	if (index >= kds_src->nitems)
	{
		if (LaneId() == 0)
			wp->scan_done = 1;
		__syncwarp();
		return 1;
	}
	index += LaneId();

	if (index < kds_src->nitems)
	{
		uint32_t	offset = KDS_GET_ROWINDEX(kds_src)[index];

		assert(offset <= kds_src->usage);
		tupitem = (kern_tupitem *)((char *)kds_src +
								   kds_src->length -
								   __kds_unpack(offset));
		assert((char *)tupitem >= (char *)kds_src &&
			   (char *)tupitem <  (char *)kds_src + kds_src->length);
		kcxt->kvars_slot = (kern_variable *)alloca(kcxt->kvars_nbytes);
		kcxt->kvars_class = (int *)(kcxt->kvars_slot + kcxt->kvars_nslots);
		if (!ExecLoadVarsOuterRow(kcxt,
								  kexp_load_vars,
								  kexp_scan_quals,
								  kds_src,
								  &tupitem->htup))
			tupitem = NULL;
	}
	/* error checks */
	if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
		return -1;
	/*
	 * save the private kvars slot on the combination buffer (depth=0)
	 */
	mask = __ballot_sync(__activemask(), tupitem != NULL);
	if (LaneId() == 0)
	{
		wr_pos = WARP_WRITE_POS(wp,0);
		WARP_WRITE_POS(wp,0) += __popc(mask);
	}
	wr_pos = __shfl_sync(__activemask(), wr_pos, 0);
	mask &= ((1U << LaneId()) - 1);
	wr_pos += __popc(mask);
	if (tupitem != NULL)
	{
		index = (wr_pos % UNIT_TUPLES_PER_DEPTH);
		memcpy((char *)kvars_addr_wp + index * kcxt->kvars_nbytes,
			   kcxt->kvars_slot,
			   kcxt->kvars_nbytes);
	}
	kcxt->kvars_slot = NULL;
	kcxt->kvars_class = NULL;
	__syncwarp();
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
							kern_expression *kexp_load_vars,
							kern_expression *kexp_scan_quals,
							char *kvars_addr_wp,
							uint32_t *p_smx_row_count)
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
		uint32_t	off;
		int			index;

		rd_pos += LaneId();
		if (rd_pos < wr_pos)
		{
			off = wp->lp_items[rd_pos % UNIT_TUPLES_PER_DEPTH];
			htup = (HeapTupleHeaderData *)((char *)kds_src + __kds_unpack(off));
			kcxt->kvars_slot = (kern_variable *)alloca(kcxt->kvars_nbytes);
			kcxt->kvars_class = (int *)(kcxt->kvars_slot + kcxt->kvars_nslots);
			if (!ExecLoadVarsOuterRow(kcxt,
									  kexp_load_vars,
									  kexp_scan_quals,
									  kds_src, htup))
				htup = NULL;
		}
		/* error checks */
		if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
			return -1;
		if (LaneId() == 0)
			wp->lp_rd_pos = Min(wp->lp_wr_pos,
								wp->lp_rd_pos + warpSize);
		/*
		 * save the private kvars on the warp-buffer
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
			index = (wr_pos % UNIT_TUPLES_PER_DEPTH);
			memcpy(kvars_addr_wp + index * kcxt->kvars_nbytes,
				   kcxt->kvars_slot,
				   kcxt->kvars_nbytes);
		}
		kcxt->kvars_slot = NULL;
		kcxt->kvars_class = NULL;
		__syncwarp();
		/* end-of-scan checks */
		if (block_id > kds_src->nitems &&	/* no more blocks to fetch */
			wp->lp_rd_pos >= wp->lp_wr_pos)	/* no more pending tuples  */
		{
			if (LaneId() == 0)
				wp->scan_done = 1;
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
			count = atomicAdd(p_smx_row_count, 1);
		count = __shfl_sync(__activemask(), count, 0);
		block_id = (get_num_groups() * count + get_group_id()) + 1;
		if (LaneId() == 0)
			wp->block_id = block_id;
	}
	if (block_id <= kds_src->nitems)
	{
		HeapTupleHeaderData *htup = NULL;
		PageHeaderData *pg_page = KDS_BLOCK_PGPAGE(kds_src, block_id-1);
		BlockNumber		block_nr = KDS_BLOCK_BLCKNR(kds_src, block_id-1);

		count = __shfl_sync(__activemask(), wp->lp_count, 0);
		if (count < PageGetMaxOffsetNumber(pg_page))
		{
			count += LaneId();
			if (count < PageGetMaxOffsetNumber(pg_page))
			{
				ItemIdData *lpp = &pg_page->pd_linp[count];

				assert((char *)lpp < (char *)pg_page + BLCKSZ);
				if (ItemIdIsNormal(lpp))
				{
					htup = (HeapTupleHeaderData *)PageGetItem(pg_page, lpp);
					/* for ctid system column reference */
					htup->t_ctid.ip_blkid.bi_hi = (uint16_t)(block_nr >> 16);
					htup->t_ctid.ip_blkid.bi_lo = (uint16_t)(block_nr & 0xffffU);
					htup->t_ctid.ip_posid = count + 1;
				}
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
				wp->lp_items[wr_pos % UNIT_TUPLES_PER_DEPTH]
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
			__syncwarp();
		}
	}
	return 0;	/* stay depth-0 */
}

/*
 * __gpuscan_load_source_arrow
 */
STATIC_FUNCTION(int)
__gpuscan_load_source_arrow(kern_context *kcxt,
							kern_warp_context *wp,
							kern_data_store *kds_src,
							kern_expression *kexp_load_vars,
							kern_expression *kexp_scan_quals,
							char *kvars_addr_wp,
							uint32_t *p_smx_row_count)
{
	uint32_t	kds_index;
	uint32_t	count;
	uint32_t	mask;
	uint32_t	wr_pos;
	bool		is_valid = false;

	/* fetch next warpSize tuples */
	if (LaneId() == 0)
		count = atomicAdd(p_smx_row_count, 1);
	count = __shfl_sync(__activemask(), count, 0);
	kds_index = (get_num_groups() * count + get_group_id()) * warpSize;
	if (kds_index >= kds_src->nitems)
	{
		wp->scan_done = 1;
		__syncwarp(__activemask());
		return 1;
	}
	kds_index += LaneId();

	if (kds_index < kds_src->nitems)
	{
		kcxt->kvars_slot = (kern_variable *)alloca(kcxt->kvars_nbytes);
		kcxt->kvars_class = (int *)(kcxt->kvars_slot + kcxt->kvars_nslots);
		if (ExecLoadVarsOuterArrow(kcxt,
								   kexp_load_vars,
								   kexp_scan_quals,
								   kds_src,
								   kds_index))
			is_valid = true;
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
	{
		int		index = (wr_pos % UNIT_TUPLES_PER_DEPTH);

		memcpy(kvars_addr_wp + index * kcxt->kvars_nbytes,
			   kcxt->kvars_slot,
			   kcxt->kvars_nbytes);
	}
	kcxt->kvars_slot = NULL;
	kcxt->kvars_class = NULL;
	/* move to the next depth if more than 32 htuples were fetched */
	return (WARP_WRITE_POS(wp,0) >= WARP_READ_POS(wp,0) + warpSize ? 1 : 0);
}

/*
 * __gpuscan_load_source_column
 */
STATIC_FUNCTION(int)
__gpuscan_load_source_column(kern_context *kcxt,
							 kern_warp_context *wp,
							 kern_data_store *kds_src,
							 kern_data_extra *kds_extra,
							 kern_expression *kexp_load_vars,
							 kern_expression *kexp_scan_quals,
							 char *kvars_addr_wp,
							 uint32_t *p_smx_row_count)
{
	uint32_t	count;
	uint32_t	index;
	uint32_t	mask;
	uint32_t	wr_pos;
	bool		row_is_valid = false;

	/* fetch next warpSize tuples */
	if (LaneId() == 0)
		count = atomicAdd(p_smx_row_count, 1);
	count = __shfl_sync(__activemask(), count, 0);
	index = (get_num_groups() * count + get_group_id()) * warpSize;
	if (index >= kds_src->nitems)
	{
		if (LaneId() == 0)
			wp->scan_done = 1;
		__syncwarp();
		return 1;
	}
	index += LaneId();

	if (index < kds_src->nitems)
	{
		kcxt->kvars_slot = (kern_variable *)alloca(kcxt->kvars_nbytes);
		kcxt->kvars_class = (int *)(kcxt->kvars_slot + kcxt->kvars_nslots);
		if (!ExecLoadVarsOuterColumn(kcxt,
									 kexp_load_vars,
									 kexp_scan_quals,
									 kds_src,
									 kds_extra,
									 index))
			row_is_valid = true;
	}
	/* error checks */
	if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
		return -1;
	/*
	 * save the private kvars slot on the combination buffer (depth=0)
	 */
	mask = __ballot_sync(__activemask(), row_is_valid);
	if (LaneId() == 0)
	{
		wr_pos = WARP_WRITE_POS(wp,0);
		WARP_WRITE_POS(wp,0) += __popc(mask);
	}
	wr_pos = __shfl_sync(__activemask(), wr_pos, 0);
	mask &= ((1U << LaneId()) - 1);
	wr_pos += __popc(mask);
	if (row_is_valid)
	{
		index = (wr_pos % UNIT_TUPLES_PER_DEPTH);
		memcpy((char *)kvars_addr_wp + index * kcxt->kvars_nbytes,
			   kcxt->kvars_slot,
			   kcxt->kvars_nbytes);
	}
	kcxt->kvars_slot = NULL;
	kcxt->kvars_class = NULL;
	__syncwarp();
	/* move to the next depth if more than 32 htuples were fetched */
	return (WARP_WRITE_POS(wp,0) >= WARP_READ_POS(wp,0) + warpSize ? 1 : 0);
}

PUBLIC_FUNCTION(int)
execGpuScanLoadSource(kern_context *kcxt,
					  kern_warp_context *wp,
					  kern_data_store *kds_src,
					  kern_data_extra *kds_extra,
					  kern_expression *kexp_load_vars,
					  kern_expression *kexp_scan_quals,
					  char *kvars_addr_wp,
					  uint32_t *p_smx_row_count)
{
	/*
	 * Move to the next depth (or projection), if combination buffer (depth=0)
	 * may overflow on the next action, or we already reached to the KDS tail.
	 */
	if (wp->scan_done || WARP_WRITE_POS(wp,0) >= WARP_READ_POS(wp,0) + warpSize)
		return 1;

	switch (kds_src->format)
	{
		case KDS_FORMAT_ROW:
			return __gpuscan_load_source_row(kcxt, wp,
											 kds_src,
											 kexp_load_vars,
											 kexp_scan_quals,
											 kvars_addr_wp,
											 p_smx_row_count);
		case KDS_FORMAT_BLOCK:
			return __gpuscan_load_source_block(kcxt, wp,
											   kds_src,
											   kexp_load_vars,
											   kexp_scan_quals,
											   kvars_addr_wp,
											   p_smx_row_count);
		case KDS_FORMAT_ARROW:
			return __gpuscan_load_source_arrow(kcxt, wp,
											   kds_src,
											   kexp_load_vars,
											   kexp_scan_quals,
											   kvars_addr_wp,
											   p_smx_row_count);
		case KDS_FORMAT_COLUMN:
			return __gpuscan_load_source_column(kcxt, wp,
												kds_src,
												kds_extra,
												kexp_load_vars,
												kexp_scan_quals,
												kvars_addr_wp,
												p_smx_row_count);
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
INLINE_FUNCTION(GpuCacheSysattr *)
kds_column_get_sysattr(kern_data_store *kds, uint32_t rowid)
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
		if (tx_log->type == GCACHE_TX_LOG__MAGIC)
		{
			rowid = ((GCacheTxLogInsert *)tx_log)->rowid;
		}
		else if (tx_log->type == GCACHE_TX_LOG__DELETE)
		{
			rowid = ((GCacheTxLogDelete *)tx_log)->rowid;
		}
		else if (tx_log->type == GCACHE_TX_LOG__COMMIT_INS ||
				 tx_log->type == GCACHE_TX_LOG__COMMIT_DEL ||
				 tx_log->type == GCACHE_TX_LOG__ABORT_INS ||
				 tx_log->type == GCACHE_TX_LOG__ABORT_DEL)
		{
			rowid = ((GCacheTxLogXact *)tx_log)->rowid;
		}
		else
		{
			STROM_ELOG(kcxt, "unknown GCacheTxLog type");
			break;
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
		if (tx_log->type == GCACHE_TX_LOG__MAGIC)
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

STATIC_FUNCTION(void)
gpucache_apply_update_logs(kern_context *kcxt,
						   kern_gpucache_redolog *redo,
						   kern_data_store *kds,
						   kern_data_extra *extra)
{}

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

STATIC_FUNCTION(void)
gpucache_apply_xact_logs(kern_context *kcxt,
						 kern_gpucache_redolog *redo,
						 kern_data_store *kds,
						 kern_data_extra *extra)
{
	uint32_t	owner_id;

	for (owner_id = get_global_id();
		 owner_id < redo->nitems;
		 owner_id += get_global_size())
	{
		GCacheTxLogXact *tx_log;
		GpuCacheSysattr *sysattr;
		uint32_t		offset = redo->redo_items[owner_id];
		uint32_t		rowid;

		tx_log = (GCacheTxLogXact *)
			((char *)redo + __kds_unpack(offset));
		switch (tx_log->type)
		{
			case GCACHE_TX_LOG__COMMIT_INS:
				sysattr = kds_column_get_sysattr(kds, tx_log->rowid);
				//do commit insert
				break;
			case GCACHE_TX_LOG__COMMIT_DEL:
				sysattr = kds_column_get_sysattr(kds, tx_log->rowid);
				//do commit delete
				break;
			case GCACHE_TX_LOG__ABORT_INS:
				sysattr = kds_column_get_sysattr(kds, tx_log->rowid);
				//do abort insert
				break;
			case GCACHE_TX_LOG__ABORT_DEL:
				sysattr = kds_column_get_sysattr(kds, tx_log->rowid);
//				if (sysattr->owner_id == owner_id)
//					sysattr->xmax = tx_log->xid;
				//do abort delete
				break;
			default:
				break;
		}
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
