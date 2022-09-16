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
					  kern_data_extra *__not_in_use__,
					  kern_data_store *kds_dst)
{
	kern_context	   *kcxt;
	kern_expression	   *kexp_scan_quals = SESSION_KEXP_SCAN_QUALS(session);
	kern_expression	   *kexp_scan_projs = SESSION_KEXP_SCAN_PROJS(session);
	bool				scan_done = false;
	kern_gpuscan_suspend_warp *warp;
	kern_gpuscan_suspend_smx *__smx;
	__shared__ kern_gpuscan_suspend_smx smx;

	assert(kds_src->format == KDS_FORMAT_ROW &&
		   kexp_scan_quals->opcode == FuncOpCode__LoadVars &&
		   kexp_scan_projs->opcode == FuncOpCode__LoadVars);
	INIT_KERNEL_CONTEXT(kcxt, session);

	/* resume the gpuscan execution context */
	__smx = &kgscan->suspend_smx[get_group_id()];
	if (get_local_id() == 0)
		smx.smx_count = __smx->smx_count;
	warp = &smx.warps[WarpId()];
	if (LaneId() == 0)
		memcpy(warp, &__smx->warps[WarpId()],
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
			read_pos = (read_pos + LaneId()) % GPUSCAN_TUPLES_PER_WARP;
			ExecProjectionOuterRow(kcxt,
								   kexp_scan_projs,
								   kds_dst,
								   kds_src,
								   warp->tupitems[read_pos],
								   0, NULL, NULL);
			if (LaneId() == 0)
				warp->read_pos += warpSize;
		}
		else if (scan_done)
		{
			if (write_pos > read_pos)
			{
				uint32_t	nvalids = write_pos - read_pos;

				read_pos = (read_pos + LaneId()) % GPUSCAN_TUPLES_PER_WARP;
				ExecProjectionOuterRow(kcxt,
									   kexp_scan_projs,
									   kds_dst,
									   kds_src,
									   LaneId() < nvalids
									   ? warp->tupitems[read_pos]
									   : NULL,
									   0, NULL, NULL);
				if (LaneId() == 0)
					warp->read_pos += nvalids;
			}
			break;
		}

		/*
		 * Identify the row-index to be fetched
		 */
		if (LaneId() == 0)
			count = atomicAdd(&smx.smx_count, warpSize);
		count = __shfl_sync(__activemask(), count, 0);
		index = ((count >> GPUSCAN_THREADS_UNITSZ_SHIFT) * get_num_groups() +
				 (count & GPUSCAN_THREADS_UNITSZ_MASK));
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
			tupitem = (kern_tupitem *)((char *)kds_src +
									   kds_src->length -
									   __kds_unpack(offset));
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
		}
		/* error checks */
		if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
			break;

		/*
		 * Save the tupitem on the local circular buffer
		 */
		mask = __ballot_sync(__activemask(), tupitem != NULL);
		if (LaneId() == 0)
			warp->write_pos += __popc(mask);
		write_pos =  __shfl_sync(__activemask(), warp->write_pos, 0);
		write_pos += __popc(((1U<<LaneId()) - 1) & mask);
		if (tupitem != NULL)
			warp->tupitems[write_pos % GPUSCAN_TUPLES_PER_WARP] = tupitem;
	}
	__syncwarp();

	/* save the execution context (may be resumed if needed) */
	if (LaneId() == 0)
	{
		atomicMax(&__smx->smx_count, smx.smx_count);
		memcpy(&__smx->warps[WarpId()], warp,
			   sizeof(kern_gpuscan_suspend_warp));
	}
	STROM_WRITEBACK_ERROR_STATUS(&kgscan->kerror, kcxt);
}

KERNEL_FUNCTION(void)
kern_gpuscan_main_block(kern_session_info *session,
						kern_gpuscan *kgscan,
						kern_data_store *kds_src,
						kern_data_extra *__not_in_use__,
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
						kern_data_extra *__not_in_use__,
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



