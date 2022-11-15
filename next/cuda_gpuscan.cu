/*
 * cuda_gpuscan.cu
 *
 * Device implementation of GpuScan
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "cuda_common.h"

#define GPUSCAN_THREADS_UNITSZ_SHIFT	11
#define GPUSCAN_THREADS_UNITSZ_MASK		(GPUSCAN_THREADS_UNITSZ - 1)
#define GPUSCAN_THREADS_UNITSZ			(1U<<GPUSCAN_THREADS_UNITSZ_SHIFT)

#define __WARP_GET_HTUPLE(rd_pos)										\
	((HeapTupleHeaderData *)											\
	 ((char *)kds_src +													\
	  __kds_unpack(warp->htuples[(rd_pos) % GPUSCAN_TUPLES_PER_WARP])))
#define __WARP_SET_HTUPLE(wr_pos,htup)					\
	warp->htuples[(wr_pos) % GPUSCAN_TUPLES_PER_WARP]	\
		= __kds_packed((char *)htup - (char *)kds_src)

#define __WARP_GET_LPITEM(rd_pos)										\
	((HeapTupleHeaderData *)											\
	 ((char *)kds_src +													\
	  __kds_unpack(warp->lpitems[(rd_pos) % GPUSCAN_TUPLES_PER_WARP])))
#define __WARP_SET_LPITEM(wr_pos,htup)					\
	warp->lpitems[(wr_pos) % GPUSCAN_TUPLES_PER_WARP]	\
		= __kds_packed((char *)htup - (char *)kds_src)

KERNEL_FUNCTION(void)
kern_gpuscan_main_row(kern_session_info *session,
					  kern_gpuscan *kgscan,
					  kern_data_store *kds_src,
					  kern_data_store *kds_dst)
{
	kern_context	   *kcxt;
	kern_expression	   *kexp_scan_quals = SESSION_KEXP_SCAN_QUALS(session);
	kern_expression	   *kexp_scan_projs = SESSION_KEXP_SCAN_PROJS(session);
	bool				scan_done = false;
	kern_gpuscan_suspend_warp *__suspend_warp;
	kern_gpuscan_suspend_warp *warp;
	__shared__ uint32_t	smx_row_count;

	assert(kds_src->format == KDS_FORMAT_ROW &&
		   kexp_scan_quals->opcode == FuncOpCode__LoadVars &&
		   kexp_scan_projs->opcode == FuncOpCode__LoadVars);
	INIT_KERNEL_CONTEXT(kcxt, session);

	/* resume the previous execution context */
	__suspend_warp = &kgscan->suspend_context[get_global_id() / warpSize];
	warp = SHARED_WORKMEM(kern_gpuscan_suspend_warp) + WarpId();
	if (get_local_id() == 0)
		smx_row_count = __suspend_warp->row_count;
	if (LaneId() == 0)
		memcpy(warp, __suspend_warp, sizeof(kern_gpuscan_suspend_warp));
	__syncthreads();
	
	for (;;)
	{
		uint32_t		write_pos;
		uint32_t		read_pos;
		uint32_t		count;
		uint32_t		index;
		uint32_t		mask;
		int				status;
		HeapTupleHeaderData *htup;

		/*
		 * Projection
		 */
		write_pos = __shfl_sync(__activemask(), warp->write_pos, 0);
		read_pos  = __shfl_sync(__activemask(), warp->read_pos, 0);
		assert(write_pos >= read_pos);
		if (write_pos >= read_pos + warpSize)
		{
			kcxt_reset(kcxt);
			read_pos += LaneId();
			htup = __WARP_GET_HTUPLE(read_pos);
			status = ExecProjectionOuterRow(kcxt,
											kexp_scan_projs,
											kds_dst,
											kds_src,
											htup,
											0, NULL, NULL);
			if (status <= 0)
			{
				assert(__activemask() == 0xffffffffU);
				if (status == 0 && LaneId() == 0)
					atomicAdd(&kgscan->suspend_count, 1);
				break;		/* error or no space */
			}
			if (LaneId() == 0)
				warp->read_pos += warpSize;
		}
		else if (scan_done)
		{
			if (write_pos > read_pos)
			{
				kcxt_reset(kcxt);
				assert(write_pos - read_pos <= warpSize);
				read_pos += LaneId();
				if (read_pos < write_pos)
					htup = __WARP_GET_HTUPLE(read_pos);
				else
					htup = NULL;
				status = ExecProjectionOuterRow(kcxt,
												kexp_scan_projs,
												kds_dst,
												kds_src,
												htup,
												0, NULL, NULL);
				if (status <= 0)
				{
					assert(__activemask() == 0xffffffffU);
					if (status == 0 && LaneId() == 0)
						atomicAdd(&kgscan->suspend_count, 1);
					break;		/* error or no space */
				}
				if (LaneId() == 0)
					warp->read_pos = warp->write_pos;
			}
			break;
		}

		/*
		 * Identify the row-index to be fetched
		 */
		if (LaneId() == 0)
			count = atomicAdd(&smx_row_count, warpSize);
		count = __shfl_sync(__activemask(), count, 0);
		index = ((count & ~GPUSCAN_THREADS_UNITSZ_MASK) * get_num_groups() +
				 (GPUSCAN_THREADS_UNITSZ * get_group_id()) +
				 (count &  GPUSCAN_THREADS_UNITSZ_MASK));
		if (index >= kds_src->nitems)
			scan_done = true;
		index += LaneId();

		/*
		 * Fetch kern_tupitem
		 */
		htup = NULL;
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
			if (ExecLoadVarsOuterRow(kcxt,
									 kexp_scan_quals,
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
			break;

		/*
		 * Save the tupitem on the local circular buffer
		 */
		mask = __ballot_sync(__activemask(), htup != NULL);
		if (LaneId() == 0)
		{
			write_pos = warp->write_pos;
			warp->write_pos += __popc(mask);
		}
		write_pos =  __shfl_sync(__activemask(), write_pos, 0);
		mask &= ((1U << LaneId()) - 1);
		write_pos += __popc(mask);
		if (htup != NULL)
			__WARP_SET_HTUPLE(write_pos, htup);
	}
	__syncthreads();

	/* save the execution context (may be resumed if needed) */
	if (LaneId() == 0)
	{
		warp->row_count = smx_row_count;
		memcpy(__suspend_warp, warp,
			   sizeof(kern_gpuscan_suspend_warp));
	}
	STROM_WRITEBACK_ERROR_STATUS(&kgscan->kerror, kcxt);
}

KERNEL_FUNCTION(void)
kern_gpuscan_main_block(kern_session_info *session,
						kern_gpuscan *kgscan,
						kern_data_store *kds_src,
						kern_data_store *kds_dst)
{
	kern_context	   *kcxt;
	kern_expression	   *kexp_scan_quals = SESSION_KEXP_SCAN_QUALS(session);
	kern_expression	   *kexp_scan_projs = SESSION_KEXP_SCAN_PROJS(session);
	bool				scan_done = false;
	kern_gpuscan_suspend_warp *__suspend_warp;
	kern_gpuscan_suspend_warp *warp;
	__shared__ uint32_t	smx_row_count;
	PageHeaderData	   *curr_page = NULL;
	uint32_t			count, index, mask;
	
	assert(kds_src->format == KDS_FORMAT_BLOCK &&
		   kexp_scan_quals->opcode == FuncOpCode__LoadVars &&
		   kexp_scan_projs->opcode == FuncOpCode__LoadVars);
	INIT_KERNEL_CONTEXT(kcxt, session);

	/* resume the previous execution state */
	__suspend_warp = &kgscan->suspend_context[get_global_id() / warpSize];
	warp = SHARED_WORKMEM(kern_gpuscan_suspend_warp) + WarpId();
	if (get_local_id() == 0)
		smx_row_count = __suspend_warp->row_count;
	if (LaneId() == 0)
		memcpy(warp, __suspend_warp, sizeof(kern_gpuscan_suspend_warp));
	__syncthreads();

	for (;;)
	{
		uint32_t		write_pos;
		uint32_t		read_pos;
		int				status;
		HeapTupleHeaderData *htup;

		/*
		 * Projection
		 */
		write_pos = __shfl_sync(__activemask(), warp->write_pos, 0);
		read_pos  = __shfl_sync(__activemask(), warp->read_pos, 0);
		assert(write_pos >= read_pos);
		if (write_pos >= read_pos + warpSize)
		{
			kcxt_reset(kcxt);
			read_pos += LaneId();
			htup = __WARP_GET_HTUPLE(read_pos);
			status = ExecProjectionOuterRow(kcxt,
											kexp_scan_projs,
											kds_dst,
											kds_src,
											htup,
											0, NULL, NULL);
			if (status <= 0)
			{
				assert(__activemask() == 0xffffffffU);
				if (status == 0 && LaneId() == 0)
					atomicAdd(&kgscan->suspend_count, 1);
				break;		/* error or no space */
			}
			if (LaneId() == 0)
				warp->read_pos += warpSize;
		}
		else if (scan_done)
		{
			if (write_pos > read_pos)
			{
				kcxt_reset(kcxt);
				assert(write_pos - read_pos <= warpSize);
				read_pos += LaneId();
				htup = (read_pos < write_pos
						? __WARP_GET_HTUPLE(read_pos)
						: NULL);
				status = ExecProjectionOuterRow(kcxt,
												kexp_scan_projs,
												kds_dst,
												kds_src,
												htup,
												0, NULL, NULL);
				if (status <= 0)
				{
					assert(__activemask() == 0xffffffffU);
					if (status == 0 && LaneId() == 0)
						atomicAdd(&kgscan->suspend_count, 1);
					break;		/* no space */
				}
				if (LaneId() == 0)
					warp->read_pos = warp->write_pos;
			}
			/* no more pending tuples? ok, terminate the loop */
			if (__shfl_sync(__activemask(), warp->write_lp_pos, 0) ==
				__shfl_sync(__activemask(), warp->read_lp_pos, 0))
				break;
		}

		/*
		 * Fetch from block
		 */
		write_pos = __shfl_sync(__activemask(), warp->write_lp_pos, 0);
		read_pos  = __shfl_sync(__activemask(), warp->read_lp_pos, 0);
		assert(write_pos >= read_pos);
		if (!scan_done && write_pos - read_pos < warpSize)
		{
			uint32_t	lp_count = __shfl_sync(__activemask(), warp->lp_count, 0);

			if (curr_page && lp_count < PageGetMaxOffsetNumber(curr_page))
			{
				lp_count += LaneId();
				if (lp_count < PageGetMaxOffsetNumber(curr_page))
				{
					ItemIdData *lpp = &curr_page->pd_linp[lp_count];

					assert((char *)lpp < (char *)curr_page + BLCKSZ);
					if (ItemIdIsNormal(lpp))
						htup = (HeapTupleHeaderData *) PageGetItem(curr_page, lpp);
					else
						htup = NULL;
				}
				else
				{
					htup = NULL;
				}
				mask = __ballot_sync(__activemask(), htup != NULL);
				if (LaneId() == 0)
					warp->write_lp_pos += __popc(mask);
				mask &= ((1U << LaneId()) - 1);
				write_pos += __popc(mask);
				if (htup != NULL)
				{
					__WARP_SET_LPITEM(write_pos, htup);
				}
				if (LaneId() == 0)
					warp->lp_count += warpSize;
			}
			else
			{
				if (LaneId() == 0)
					warp->row_count = atomicAdd(&smx_row_count, 1);
				count = __shfl_sync(__activemask(), warp->row_count, 0);
				index = count * get_num_groups() + get_group_id();
				if (index < kds_src->nitems)
				{
					curr_page = (PageHeaderData *)((char *)kds_src +
												   kds_src->block_offset +
												   index * BLCKSZ);
					if (LaneId() == 0)
						warp->lp_count = 0;
					assert((char *)curr_page >= (char *)kds_src &&
						   (char *)curr_page+BLCKSZ <= (char *)kds_src+kds_src->length);
				}
				else
				{
					curr_page = NULL;
					scan_done = true;
				}
			}
		}
		else if (!scan_done
				 ? write_pos - read_pos >= warpSize
				 : write_pos - read_pos > 0)
		{
			xpu_bool_t	retval;

			read_pos += LaneId();
			if (read_pos < write_pos)
			{
				htup = __WARP_GET_LPITEM(read_pos);
				if (ExecLoadVarsOuterRow(kcxt,
										 kexp_scan_quals,
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
			else
			{
				htup = NULL;
			}
			/* error checks */
			if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
				break;
			/*
			 * Save the HeapTuple on the local ring buffer
			 */
			mask = __ballot_sync(__activemask(), htup != NULL);
			if (LaneId() == 0)
			{
				write_pos = warp->write_pos;
				warp->write_pos += __popc(mask);
			}
			write_pos =  __shfl_sync(__activemask(), write_pos, 0);
			mask &= ((1U << LaneId()) - 1);
			write_pos += __popc(mask);
			if (htup != NULL)
				__WARP_SET_HTUPLE(write_pos, htup);
			if (LaneId() == 0)
				warp->read_lp_pos = Min(warp->read_lp_pos + warpSize,
										warp->write_lp_pos);
		}
	}
	__syncthreads();

	return;
	/* save the execution context (may be resumed if needed) */
	if (LaneId() == 0)
	{
		warp->row_count = smx_row_count;
		memcpy(__suspend_warp, warp,
			   sizeof(kern_gpuscan_suspend_warp));
	}
	STROM_WRITEBACK_ERROR_STATUS(&kgscan->kerror, kcxt);
}

KERNEL_FUNCTION(void)
kern_gpuscan_main_arrow(kern_session_info *session,
						kern_gpuscan *kgscan,
						kern_data_store *kds_src,
						kern_data_store *kds_dst)
{
	kern_context   *kcxt;

	assert(kds_src->format == KDS_FORMAT_ARROW);
	INIT_KERNEL_CONTEXT(kcxt, session);

	
}

KERNEL_FUNCTION(void)
kern_gpuscan_main_column(kern_session_info *session,
						 kern_gpuscan *kgscan,
						 kern_data_store *kds_src,
						 kern_data_extra *kds_extra,
						 kern_data_store *kds_dst)
{
	kern_context   *kcxt;

	assert(kds_src->format == KDS_FORMAT_COLUMN);
	INIT_KERNEL_CONTEXT(kcxt, session);

	
}



