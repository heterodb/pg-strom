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
	kern_gpuscan_suspend_context *__suspend_cxt;
	kern_gpuscan_suspend_warp *warp;
	__shared__ kern_gpuscan_suspend_context smx;

	assert(kds_src->format == KDS_FORMAT_ROW &&
		   kexp_scan_quals->opcode == FuncOpCode__LoadVars &&
		   kexp_scan_projs->opcode == FuncOpCode__LoadVars);
	INIT_KERNEL_CONTEXT(kcxt, session);
	
	/* resume the previous execution state */
	__suspend_cxt = &kgscan->suspend_context[get_group_id()];
	if (get_local_id() == 0)
		smx.row_count = __suspend_cxt->row_count;
	warp = &smx.warps[WarpId()];
	if (LaneId() == 0)
		memcpy(warp, &__suspend_cxt->warps[WarpId()],
			   sizeof(kern_gpuscan_suspend_warp));
	__syncthreads();
	
	for (;;)
	{
		uint32_t		write_pos;
		uint32_t		read_pos;
		uint32_t		count;
		uint32_t		index;
		uint32_t		mask;
		kern_tupitem   *tupitem = NULL;

		/*
		 * Projection
		 */
		write_pos = __shfl_sync(__activemask(), warp->write_pos, 0);
		read_pos  = __shfl_sync(__activemask(), warp->read_pos, 0);
		assert(write_pos >= read_pos);
		if (write_pos >= read_pos + warpSize)
		{
			kcxt_reset(kcxt);
			read_pos = (read_pos + LaneId()) % GPUSCAN_TUPLES_PER_WARP;
			if (!ExecProjectionOuterRow(kcxt,
										kexp_scan_projs,
										kds_dst,
										kds_src,
										warp->tupitems[read_pos],
										0, NULL, NULL))
			{
				assert(__activemask() == 0xffffffffU);
				if (LaneId() == 0)
					atomicAdd(&kgscan->suspend_count, 1);
				break;		/* no space */
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
					tupitem = warp->tupitems[read_pos % GPUSCAN_TUPLES_PER_WARP];
				if (!ExecProjectionOuterRow(kcxt,
											kexp_scan_projs,
											kds_dst,
											kds_src,
											tupitem,
											0, NULL, NULL))
				{
					assert(__activemask() == 0xffffffffU);
					if (LaneId() == 0)
						atomicAdd(&kgscan->suspend_count, 1);
					break;		/* no space */
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
			count = atomicAdd(&smx.row_count, warpSize);
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
		if (index < kds_src->nitems)
		{
			uint32_t	offset = KDS_GET_ROWINDEX(kds_src)[index];
			xpu_bool_t	retval;

			assert(offset <= kds_src->usage);
			kcxt_reset(kcxt);
			tupitem = (kern_tupitem *)((char *)kds_src +
									   kds_src->length -
									   __kds_unpack(offset));
			assert((char *)tupitem >= (char *)kds_src &&
				   (char *)tupitem <  (char *)kds_src + kds_src->length);
			if (!ExecLoadVarsOuterRow(kcxt,
									  kexp_scan_quals,
									  (xpu_datum_t *)&retval,
									  kds_src,
									  tupitem,
									  0, NULL, NULL))
			{
				assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
				tupitem = NULL;
			}
			if (retval.isnull || !retval.value)
				tupitem = NULL;
		}
		/* error checks */
		if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
			break;

		/*
		 * Save the tupitem on the local circular buffer
		 */
		mask = __ballot_sync(__activemask(), tupitem != NULL);
		if (LaneId() == 0)
		{
			write_pos = warp->write_pos;
			warp->write_pos += __popc(mask);
		}
		write_pos =  __shfl_sync(__activemask(), write_pos, 0);
		mask &= ((1U << LaneId()) - 1);
		write_pos += __popc(mask);
		if (tupitem != NULL)
			warp->tupitems[write_pos % GPUSCAN_TUPLES_PER_WARP] = tupitem;
	}
	__syncwarp();

	/* save the execution context (may be resumed if needed) */
	if (LaneId() == 0)
	{
		atomicMax(&__suspend_cxt->row_count, smx.row_count);
		memcpy(&__suspend_cxt->warps[WarpId()], warp,
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
	kern_context   *kcxt;

	assert(kds_src->format == KDS_FORMAT_BLOCK);
	INIT_KERNEL_CONTEXT(kcxt, session);
	
	
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



