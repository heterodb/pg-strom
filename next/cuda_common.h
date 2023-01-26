/*
 * cuda_common.h
 *
 * Common header for CUDA device code, in addition to xPU common definitions.
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H
#include "xpu_common.h"

#define WARPSIZE				32
#define MAXTHREADS_PER_BLOCK	1024
#define MAXWARPS_PER_BLOCK		(MAXTHREADS_PER_BLOCK / WARPSIZE)

#if defined(__CUDACC__)
/*
 * Thread index at CUDA C
 */
#define get_group_id()			(blockIdx.x)
#define get_num_groups()		(gridDim.x)
#define get_local_id()			(threadIdx.x)
#define get_local_size()		(blockDim.x)
#define get_global_id()			(threadIdx.x + blockIdx.x * blockDim.x)
#define get_global_size()		(blockDim.x * gridDim.x)

/* Dynamic shared memory entrypoint */
extern __shared__ char __pgstrom_dynamic_shared_workmem[] __MAXALIGNED__;
#define SHARED_WORKMEM(TYPE)	((TYPE *) __pgstrom_dynamic_shared_workmem)

INLINE_FUNCTION(uint32_t) WarpId(void)
{
	uint32_t	rv;

	asm volatile("mov.u32 %0, %warpid;" : "=r"(rv) );

	return rv;
}

INLINE_FUNCTION(uint32_t) LaneId(void)
{
	uint32_t	rv;

	asm volatile("mov.u32 %0, %laneid;" : "=r"(rv) );

	return rv;
}

template <typename T>
INLINE_FUNCTION(T)
__reduce_stair_add_sync(T value, T *p_total_sum = NULL)
{
	uint32_t	lane_id = LaneId();
	uint32_t	mask;
	T			temp;

	assert(__activemask() == 0xffffffffU);
	for (mask = 1; mask < warpSize; mask <<= 1)
	{
		temp = __shfl_sync(__activemask(), value, (lane_id & ~mask) | (mask - 1));
		if (lane_id & mask)
			value += temp;
	}
	temp = __shfl_sync(__activemask(), value, warpSize - 1);
	if (p_total_sum)
		*p_total_sum = temp;
	return value;
}

INLINE_FUNCTION(void)
STROM_WRITEBACK_ERROR_STATUS(kern_errorbuf *ebuf, kern_context *kcxt)
{
	if (kcxt->errcode != ERRCODE_STROM_SUCCESS &&
		atomicCAS(&ebuf->errcode,
				  ERRCODE_STROM_SUCCESS,
				  kcxt->errcode) == ERRCODE_STROM_SUCCESS)
	{
		ebuf->errcode = kcxt->errcode;
		ebuf->lineno  = kcxt->error_lineno;
		__strncpy(ebuf->filename,
				  kcxt->error_filename,
				  KERN_ERRORBUF_FILENAME_LEN);
		__strncpy(ebuf->funcname,
				  kcxt->error_funcname,
				  KERN_ERRORBUF_FUNCNAME_LEN);
		__strncpy(ebuf->message,
				  kcxt->error_message,
				  KERN_ERRORBUF_MESSAGE_LEN);
	}
}
#endif	/* __CUDACC__ */

/* ----------------------------------------------------------------
 *
 * Definitions related to GpuScan
 *
 * ----------------------------------------------------------------
 */
#define GPUSCAN_TUPLES_PER_WARP		(2 * WARPSIZE)
typedef struct {
	uint32_t		row_count;
	uint32_t		lp_count;		/* only KDS_FORMAT_BLOCK */
	uint32_t		write_pos;
	uint32_t		read_pos;
	uint32_t		write_lp_pos;	/* only KDS_FORMAT_BLOCK */
	uint32_t		read_lp_pos;	/* only KDS_FORMAT_BLOCK */
	uint32_t		lpitems[GPUSCAN_TUPLES_PER_WARP];	/* only KDS_FORMAT_BLOCK */
	uint32_t		htuples[GPUSCAN_TUPLES_PER_WARP];
} kern_gpuscan_suspend_warp;

typedef struct {
	kern_errorbuf	kerror;
	uint32_t		grid_sz;
	uint32_t		block_sz;
	uint32_t		nitems_in;
	uint32_t		nitems_out;
	uint32_t		extra_sz;
	/* suspend/resume support */
	uint32_t		suspend_count;
	kern_gpuscan_suspend_warp suspend_context[1];	/* per warp */
} kern_gpuscan;

KERNEL_FUNCTION(void)
kern_gpuscan_main_row(kern_session_info *session,
					  kern_gpuscan *kgscan,
					  kern_data_store *kds_src,
					  kern_data_store *kds_dst);

KERNEL_FUNCTION(void)
kern_gpuscan_main_block(kern_session_info *session,
						kern_gpuscan *kgscan,
						kern_data_store *kds_src,
						kern_data_store *kds_dst);
KERNEL_FUNCTION(void)
kern_gpuscan_main_arrow(kern_session_info *session,
						kern_gpuscan *kgscan,
						kern_data_store *kds_src,
						kern_data_store *kds_dst);
KERNEL_FUNCTION(void)
kern_gpuscan_main_column(kern_session_info *session,
						 kern_gpuscan *kgscan,
						 kern_data_store *kds_src,
						 kern_data_extra *kds_extra,
						 kern_data_store *kds_dst);

/*
 * Definitions related to GpuJoin
 */
typedef struct
{
	kern_errorbuf	kerror;
	uint32_t		grid_sz;
	uint32_t		block_sz;
	uint32_t		nitems_in;
	uint32_t		nitems_out;
	uint32_t		num_rels;
	/* kern_warp_context array */
	char			data[1]	__MAXALIGNED__;
} kern_gpujoin;

#define UNIT_TUPLES_PER_WARP		(2 * WARPSIZE)
typedef struct
{
	uint32_t		__saved_row_count;
	bool			scan_done;	/* true, if it already reached to the KDS tail */
	int				depth;		/* current depth, if JOIN */
	uint32_t		nrels;		/* number of inner relations, if JOIN */
	/* only KDS_FORMAT_BLOCK */
	uint32_t		block_id;	/* BLOCK format needs to keep htuples on the */
	uint32_t		lp_count;	/* lp_items array once, to pull maximum GPU */
	uint32_t		lp_wr_pos;	/* utilization by simultaneous execution of */
	uint32_t		lp_rd_pos;	/* the kern_scan_quals. */
	uint32_t		lp_items[UNIT_TUPLES_PER_WARP];
	/*
	 * read/write_pos and combination buffer with the layout below
	 */
	uint32_t		regs[1];
	/*
	 * kern_warp_context    -------------------------------------------------
	 *                                                                     ^
	 * |       :                                                           |
	 * +--- values[] ------------+  ---                                    |
	 * | read_pos  (depth=0)     |   ^                                     |
	 * | write_pos (depth=0)     |   | current position to indicate the    |
	 * +-------------------------+   | GpuJoin combination buffer for      |
	 * | read_pos  (depth=1)     |   | each depth.                         |
	 * | write_pos (depth=1)     |   |                                     |
	 * +-------------------------+   | (2 * (NRELS+1)) items               |
	 * :        :                :   |                                     |
	 * :        :                :   |                                     |
	 * +-------------------------+   |                                     |
	 * | read_pos  (depth=NRELS) |   |                                     |
	 * | write_pos (depth=NRELS) |   v                                     |
	 * +-------------------------+  ---                                    |
	 * |                         |   ^                                     |
	 * | GpuJoin combination     |   | GpuJoin combination buffer for      |
	 * | buffer (depth = 0)      |   | depth=0.                            |
	 * |  (UNIT_TUPLES_PER_WARP) |   |                                     |
	 * |  items for depth=0      |   | (1+depth) * UNIT_TUPLES_PER_WARP    |
	 * |                         |   v   items for each depth              |
	 * +-------------------------+  ---                                    |
	 * :        :                :                                         |
	 * :        :                :                        KERN_WARP_CONTEXT_UNITSZ()
	 * +-------------------------+  ---                                    |
	 * |                         |   ^                                     |
	 * | GpuJoin combination     |   | GpuJoin combination buffer for      |
	 * | buffer (depth = NRELS)  |   | depth=NRELS.                        |
	 * |  (UNIT_TUPLES_PER_WARP  |   |                                     |
	 * |   * NRELS) items        |   | NRELS * UNIT_TUPLES_PER_WARP        |
	 * |                         |   v    items for the final depth        v
	 * +-------------------------+  ---                                   ---
	 */
} kern_warp_context;

INLINE_FUNCTION(uint32_t)
KERN_WARP_CONTEXT_UNITSZ(int nrels = 0)
{
	int		nitems = (2 * (nrels + 1)) +					/* read/write_pos */
		((nrels+1) * (nrels+2) * (UNIT_TUPLES_PER_WARP/2));	/* combination buffer */
	return MAXALIGN(offsetof(kern_warp_context, regs[nitems]));
}

#define WARP_READ_POS(warp,depth)		((warp)->regs[2*(depth)])
#define WARP_WRITE_POS(warp,depth)		((warp)->regs[2*(depth)+1])

INLINE_FUNCTION(uint32_t *)
WARP_COMB_BUF(kern_warp_context *warp, int depth=0)
{
	uint32_t   *base = warp->regs + (2 * (warp->nrels + 1));

	return base + warp->nrels * (warp->nrels + 1) * UNIT_TUPLES_PER_WARP;
}

#endif	/* CUDA_COMMON_H */
