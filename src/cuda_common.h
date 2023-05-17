/*
 * cuda_common.h
 *
 * Common header for CUDA device code, in addition to xPU common definitions.
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
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
#define CUDA_L1_CACHELINE_SZ	128

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
#define SHARED_WORKMEM(UNITSZ,INDEX)						\
	(__pgstrom_dynamic_shared_workmem + (UNITSZ)*(INDEX))

INLINE_FUNCTION(uint32_t) LaneId(void)
{
	uint32_t	rv;

	asm volatile("mov.u32 %0, %laneid;" : "=r"(rv) );

	return rv;
}

INLINE_FUNCTION(uint32_t) DynamicShmemSize(void)
{
	uint32_t	rv;

	asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(rv) );

	return rv;
}

INLINE_FUNCTION(uint32_t) TotalShmemSize(void)
{
	uint32_t	rv;

	asm volatile("mov.u32 %0, %total_smem_size;" : "=r"(rv) );

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
 * Definitions related to per-warp context
 *
 * ----------------------------------------------------------------
 */
#define UNIT_TUPLES_PER_DEPTH		(2 * WARPSIZE)
typedef struct
{
	uint32_t		smx_row_count;	/* just for suspend/resume */
	uint32_t		__nrels__deprecated;		/* number of inner relations, if JOIN */
	int				depth;		/* 'depth' when suspended */
	int				scan_done;	/* smallest depth that may produce more tuples */
	/* only KDS_FORMAT_BLOCK */
	uint32_t		block_id;	/* BLOCK format needs to keep htuples on the */
	uint32_t		lp_count;	/* lp_items array once, to pull maximum GPU */
	uint32_t		lp_wr_pos;	/* utilization by simultaneous execution of */
	uint32_t		lp_rd_pos;	/* the kern_scan_quals. */
	uint32_t		lp_items[UNIT_TUPLES_PER_DEPTH];
	/* read/write_pos of the combination buffer for each depth */
	struct {
		uint32_t	read;		/* read_pos of depth=X */
		uint32_t	write;		/* write_pos of depth=X */
	} pos[1];		/* variable length */
	/*
	 * <----- __KERN_WARP_CONTEXT_BASESZ ----->
	 * Above fields are always kept in the device shared memory.
	 *
	 * +-------------------------------------------------------------+------
	 * | kvars_slot[nslots] + kvars_class[nslots] + extra_sz (pos-0) |
	 * | kvars_slot[nslots] + kvars_class[nslots] + extra_sz (pos-1) |
	 * | kvars_slot[nslots] + kvars_class[nslots] + extra_sz (pos-2) | depth=0
	 * |      :                    :                     :           |
	 * | kvars_slot[nslots] + kvars_class[nslots] + extra_sz (pos-63)|
	 * +-------------------------------------------------------------+------
	 *        :                    :                     :
	 * +-------------------------------------------------------------+------
	 * | kvars_slot[nslots] + kvars_class[nslots] + extra_sz (pos-0) |
	 * | kvars_slot[nslots] + kvars_class[nslots] + extra_sz (pos-1) |
	 * | kvars_slot[nslots] + kvars_class[nslots] + extra_sz (pos-2) | depth=nrels
	 * |      :                    :                     :           |
	 * | kvars_slot[nslots] + kvars_class[nslots] + extra_sz (pos-63)|
	 * +-------------------------------------------------------------+------
	 */
} kern_warp_context;

#define __KERN_WARP_CONTEXT_BASESZ(n_rels)				\
	MAXALIGN(offsetof(kern_warp_context, pos[(n_rels)+1]))
#define KERN_WARP_CONTEXT_UNITSZ(n_rels,nbytes)			\
	(__KERN_WARP_CONTEXT_BASESZ(n_rels) +				\
	 (nbytes) * UNIT_TUPLES_PER_DEPTH * ((n_rels)+1))
#define WARP_READ_POS(warp,depth)		((warp)->pos[(depth)].read)
#define WARP_WRITE_POS(warp,depth)		((warp)->pos[(depth)].write)

/*
 * definitions related to generic device executor routines
 */
EXTERN_FUNCTION(int)
execGpuScanLoadSource(kern_context *kcxt,
					  kern_warp_context *wp,
					  kern_data_store *kds_src,
					  kern_data_extra *kds_extra,
					  kern_expression *kexp_load_vars,
					  kern_expression *kexp_scan_quals,
					  char     *kvars_addr_wp,
					  uint32_t *p_smx_row_count);
EXTERN_FUNCTION(int)
execGpuJoinProjection(kern_context *kcxt,
					  kern_warp_context *wp,
					  int n_rels,
					  kern_data_store *kds_dst,
					  kern_expression *kexp_projection,
					  char *kvars_addr_wp,
					  bool *p_try_suspend);
EXTERN_FUNCTION(int)
execGpuPreAggGroupBy(kern_context *kcxt,
					 kern_warp_context *wp,
					 int n_rels,
					 kern_data_store *kds_final,
					 char *kvars_addr_wp,
					 bool *p_try_suspend);
/*
 * Definitions related to GpuScan/GpuJoin/GpuPreAgg
 */
typedef struct {
	kern_errorbuf	kerror;
	uint32_t		grid_sz;
	uint32_t		block_sz;
	uint32_t		extra_sz;
	uint32_t		kvars_nslots;			/* width of the kvars slot */
	uint32_t		kvars_nbytes;	/* extra buffer size of kvars-slot */
	uint32_t		n_rels;			/* >0, if JOIN is involved */
	/* suspend/resume support */
	bool			resume_context;
	uint32_t		suspend_count;
	/* kernel statistics */
	uint32_t		nitems_raw;		/* nitems in the raw data chunk */
	uint32_t		nitems_in;		/* nitems after the scan_quals */
	uint32_t		nitems_out;		/* nitems of final results */
	struct {
		uint32_t	nitems_gist;	/* nitems picked up by GiST index */
		uint32_t	nitems_out;		/* nitems after this depth */
	} stats[1];
	/*
	 * variable length fields
	 * +-----------------------------------+
	 * | kern_warp_context[0] for warp-0   |
	 * | kern_warp_context[1] for warp-1   |
	 * |     :    :            :           |
	 * | kern_warp_context[nwarps-1]       |
	 * +-----------------------------------+ -----
	 * | l_state[num_rels] for each thread |  only if JOIN is involved
	 * +-----------------------------------+  (n_rels > 0)
	 * | matched[num_rels] for each thread |
	 * +-----------------------------------+ -----
	 */
} kern_gputask;

#define __KERN_GPUTASK_WARP_OFFSET(n_rels,nbytes,gid)				\
	(MAXALIGN(offsetof(kern_gputask,stats[(n_rels)])) +				\
	 KERN_WARP_CONTEXT_UNITSZ(n_rels,nbytes) * ((gid)/WARPSIZE))

#define KERN_GPUTASK_WARP_CONTEXT(kgtask)								\
	((kern_warp_context *)												\
	 ((char *)(kgtask) +												\
	  __KERN_GPUTASK_WARP_OFFSET((kgtask)->n_rels,						\
								 (kgtask)->kvars_nbytes,				\
								 get_global_id())))
#define KERN_GPUTASK_LSTATE_ARRAY(kgtask)								\
	((kgtask)->n_rels == 0 ? NULL : (uint32_t *)						\
	 ((char *)(kgtask) +												\
	  __KERN_GPUTASK_WARP_OFFSET((kgtask)->n_rels,						\
								 (kgtask)->kvars_nbytes,				\
								 get_global_size()) +					\
	  sizeof(uint32_t) * (kgtask)->n_rels * get_global_id()))
#define KERN_GPUTASK_MATCHED_ARRAY(kgtask)								\
	((kgtask)->n_rels == 0 ? NULL : (bool *)							\
	 ((char *)(kgtask) +												\
	  __KERN_GPUTASK_WARP_OFFSET((kgtask)->n_rels,						\
								 (kgtask)->kvars_nbytes,				\
								 get_global_size()) +					\
	  sizeof(uint32_t) * (kgtask)->n_rels * get_global_size() +			\
	  sizeof(bool) *     (kgtask)->n_rels * get_global_id()))

#define KERN_GPUTASK_LENGTH(n_rels,nbytes,n_threads)					\
	(__KERN_GPUTASK_WARP_OFFSET((n_rels),(nbytes),(n_threads)) +		\
	 sizeof(uint32_t) * (n_rels) * (n_threads) +						\
	 sizeof(bool)     * (n_rels) * (n_threads))

/*
 * GPU Kernel Entrypoint
 */
KERNEL_FUNCTION(void)
kern_gpuscan_main(kern_session_info *session,
				  kern_gputask *kgtask,
				  kern_multirels *__kmrels,	/* always null */
				  kern_data_store *kds_src,
				  kern_data_extra *kds_extra,
				  kern_data_store *kds_dst);
KERNEL_FUNCTION(void)
kern_gpujoin_main(kern_session_info *session,
				  kern_gputask *kgtask,
				  kern_multirels *kmrels,
				  kern_data_store *kds_src,
				  kern_data_extra *kds_extra,
				  kern_data_store *kds_dst);

#endif	/* CUDA_COMMON_H */
