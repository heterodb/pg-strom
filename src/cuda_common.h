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

#if defined(__CUDACC__)
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
	if (kcxt->errcode != ERRCODE_STROM_SUCCESS)
	{
		uint32_t	errcode_old;
		uint32_t	errcode_cur;

		errcode_cur = __volatileRead(&ebuf->errcode);
		do {
			errcode_old = errcode_cur;
			switch (errcode_cur)
			{
				case ERRCODE_SUSPEND_FALLBACK:
				case ERRCODE_SUSPEND_NO_SPACE:
					if (ERRCODE_IS_SUSPEND(kcxt->errcode))
						return;
				case ERRCODE_STROM_SUCCESS:
					break;
				default:
					return;		/* significant error code is already set */
			}
		} while ((errcode_cur = __atomic_cas_uint32(&ebuf->errcode,
													errcode_old,
													kcxt->errcode)) != errcode_old);
		ebuf->lineno  = kcxt->error_lineno;
		__strncpy(ebuf->filename,
				  __basename(kcxt->error_filename),
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
#define LP_ITEMS_PER_BLOCK			(2 * CUDA_MAXTHREADS_PER_BLOCK)
typedef struct
{
	uint32_t		smx_row_count;	/* current position of outer relation */
	int				depth;		/* depth when last kernel is suspended */
	int				scan_done;	/* smallest depth that may produce more tuples */
	/* only KDS_FORMAT_BLOCK */
	uint32_t		block_id;	/* BLOCK format needs to keep htuples on the */
	uint32_t		lp_count;	/* lp_items array once, to pull maximum GPU */
	uint32_t		lp_wr_pos;	/* utilization by simultaneous execution of */
	uint32_t		lp_rd_pos;	/* the kern_scan_quals. */
	uint32_t		lp_items[LP_ITEMS_PER_BLOCK];
	/* read/write_pos of the combination buffer for each depth */
	struct {
		uint32_t	read;		/* read_pos of depth=X */
		uint32_t	write;		/* write_pos of depth=X */
	} pos[1];		/* variable length */
	/*
	 * above fields are always kept in the device shared memory.
	 * <----- __KERN_WARP_CONTEXT_BASESZ ----->
	 * +-------------------------------------------------------+---------
	 * | kernel vectorized variable buffer (depth-0)           |     ^
	 * | kernel vectorized variable buffer (depth-1)           |     |
	 * |                  :                                    | kvec_ndims
	 * | kernel vectorized variable buffer (depth-[n_rels])    |  buffer
	 * |                  :                                    |     |
	 * | kernel vectorized variable buffer (depth-[n_dims-1])  |     V
	 * +-------------------------------------------------------+---------
	 */
} kern_warp_context;

#define WARP_READ_POS(warp,depth)		((warp)->pos[(depth)].read)
#define WARP_WRITE_POS(warp,depth)		((warp)->pos[(depth)].write)
#define __KERN_WARP_CONTEXT_BASESZ(kvecs_ndims)					\
	TYPEALIGN(CUDA_L1_CACHELINE_SZ,								\
			  offsetof(kern_warp_context, pos[(kvecs_ndims)]))
#define KERN_WARP_CONTEXT_LENGTH(kvecs_ndims,kvecs_bufsz)		\
	(__KERN_WARP_CONTEXT_BASESZ(kvecs_ndims) +					\
	 (kvecs_ndims) * TYPEALIGN(CUDA_L1_CACHELINE_SZ,(kvecs_bufsz)))

/*
 * Definitions related to GpuScan/GpuJoin/GpuPreAgg
 */
typedef struct {
	kern_errorbuf	kerror;
	uint32_t		grid_sz;
	uint32_t		block_sz;
	uint32_t		kvars_nslots;	/* width of the kvars slot (scalar values) */
	uint32_t		kvecs_bufsz;	/* length of the kvecs buffer (vectorized values) */
	uint32_t		kvecs_ndims;	/* number of kvecs buffers for each warp */
	uint32_t		extra_sz;
	uint32_t		n_rels;			/* >0, if JOIN is involved */
	uint32_t		groupby_prepfn_bufsz;
	uint32_t		groupby_prepfn_nbufs;
	/* suspend/resume support */
	bool			resume_context;
	uint32_t		suspend_count;
	uint32_t		suspend_by_nospace;		/* destination buffer has no space */
	uint32_t		suspend_by_fallback;	/* unable to continue fallback */
	/* kernel statistics */
	uint32_t		nitems_raw;		/* nitems in the raw data chunk */
	uint32_t		nitems_in;		/* nitems after the scan_quals */
	uint32_t		nitems_out;		/* nitems of final results */
	struct {
		uint32_t	nitems_gist;	/* nitems picked up by GiST index */
		uint32_t	nitems_out;		/* nitems after this depth */
	} stats[1];		/* 'n_rels' items */
	/*
	 * variable length fields
	 * +-----------------------------------+  <---  __KERN_GPUTASK_WARP_OFFSET()
	 * | kern_warp_context for each block  |
	 * +-----------------------------------+ -----
	 * | l_state[num_rels] for each thread |  only if JOIN is involved
	 * +-----------------------------------+  (n_rels > 0)
	 * | matched[num_rels] for each thread |
	 * +-----------------------------------+ -----
	 */
} kern_gputask;

#define __KERN_GPUTASK_WARP_OFFSET(kvecs_ndims,kvecs_bufsz)					\
	TYPEALIGN(CUDA_L1_CACHELINE_SZ, offsetof(kern_gputask, stats[(kvecs_ndims)]))
#define KERN_GPUTASK_LENGTH(kvecs_ndims,kvecs_bufsz,grid_sz,block_sz)		\
	(__KERN_GPUTASK_WARP_OFFSET((kvecs_ndims),(kvecs_bufsz)) +				\
	 KERN_WARP_CONTEXT_LENGTH((kvecs_ndims),(kvecs_bufsz)) * (grid_sz) +	\
	 MAXALIGN(sizeof(uint64_t) * (grid_sz) * (block_sz) * (kvecs_ndims)) +	\
	 MAXALIGN(sizeof(bool)     * (grid_sz) * (block_sz) * (kvecs_ndims)))

#if defined(__CUDACC__)
INLINE_FUNCTION(void)
INIT_KERN_GPUTASK_SUBFIELDS(kern_gputask *kgtask,
							kern_warp_context **p_wp_context,
							uint64_t **p_lstate_array,
							bool **p_matched_array)
{
	uint32_t	wp_unitsz;
	char	   *pos;

	wp_unitsz = KERN_WARP_CONTEXT_LENGTH(kgtask->kvecs_ndims,
										 kgtask->kvecs_bufsz);
	pos = ((char *)kgtask +
		   __KERN_GPUTASK_WARP_OFFSET(kgtask->kvecs_ndims,
									  kgtask->kvecs_bufsz));
	*p_wp_context = (kern_warp_context *)(pos + wp_unitsz * get_group_id());
	pos += wp_unitsz * get_num_groups();

	*p_lstate_array  = (uint64_t *)pos;
	pos += MAXALIGN(sizeof(uint64_t) * get_global_size() * kgtask->n_rels);

	*p_matched_array = (bool *)pos;
	pos += MAXALIGN(sizeof(bool)     * get_global_size() * kgtask->n_rels);
}
#endif

/*
 * Declarations related to generic device executor routines
 */
EXTERN_FUNCTION(uint32_t)
pgstrom_stair_sum_binary(bool predicate, uint32_t *p_total_count);
EXTERN_FUNCTION(uint32_t)
pgstrom_stair_sum_uint32(uint32_t value, uint32_t *p_total_count);
EXTERN_FUNCTION(uint64_t)
pgstrom_stair_sum_uint64(uint64_t value, uint64_t *p_total_count);
EXTERN_FUNCTION(int64_t)
pgstrom_stair_sum_int64(int64_t value, int64_t *p_total_count);
EXTERN_FUNCTION(float8_t)
pgstrom_stair_sum_fp64(float8_t value, float8_t *p_total_count);
EXTERN_FUNCTION(int32_t)
pgstrom_local_min_int32(int32_t my_value);
EXTERN_FUNCTION(int32_t)
pgstrom_local_max_int32(int32_t my_value);
EXTERN_FUNCTION(int64_t)
pgstrom_local_min_int64(int64_t my_value);
EXTERN_FUNCTION(int64_t)
pgstrom_local_max_int64(int64_t my_value);
EXTERN_FUNCTION(float8_t)
pgstrom_local_min_fp64(float8_t my_value);
EXTERN_FUNCTION(float8_t)
pgstrom_local_max_fp64(float8_t my_value);

EXTERN_FUNCTION(int)
execGpuScanLoadSource(kern_context *kcxt,
					  kern_warp_context *wp,
					  const kern_data_store *kds_src,
					  const kern_data_extra *kds_extra,
					  const kern_expression *kexp_load_vars,
					  const kern_expression *kexp_scan_quals,
					  const kern_expression *kexp_move_vars,
					  char     *dst_kvecs_buffer);
EXTERN_FUNCTION(int)
execGpuJoinProjection(kern_context *kcxt,
					  kern_warp_context *wp,
					  int n_rels,
					  kern_data_store *kds_dst,
					  kern_expression *kexp_projection,
					  char *kvars_addr_wp);
EXTERN_FUNCTION(int)
execGpuPreAggGroupBy(kern_context *kcxt,
					 kern_warp_context *wp,
					 int n_rels,
					 kern_data_store *kds_final,
					 char *kvars_addr_wp);
EXTERN_FUNCTION(void)
setupGpuPreAggGroupByBuffer(kern_context *kcxt,
							kern_gputask *kgtask,
							char *groupby_prepfn_buffer);
EXTERN_FUNCTION(void)
mergeGpuPreAggGroupByBuffer(kern_context *kcxt,
							kern_data_store *kds_final);

/*
 * Definitions related to GpuCache
 */
typedef struct {
	uint32_t	database_oid;
	uint32_t	table_oid;
	uint64_t	signature;
} GpuCacheIdent;

INLINE_FUNCTION(bool)
GpuCacheIdentEqual(const GpuCacheIdent *a, const GpuCacheIdent *b)
{
	return (a->database_oid == b->database_oid &&
			a->table_oid    == b->table_oid &&
			a->signature    == b->signature);
}

#define GCACHE_TX_LOG__MAGIC		0xEBAD7C00
#define GCACHE_TX_LOG__INSERT		(GCACHE_TX_LOG__MAGIC | 'I')
#define GCACHE_TX_LOG__DELETE		(GCACHE_TX_LOG__MAGIC | 'D')
#define GCACHE_TX_LOG__COMMIT_INS	(GCACHE_TX_LOG__MAGIC | 'C')
#define GCACHE_TX_LOG__COMMIT_DEL	(GCACHE_TX_LOG__MAGIC | 'c')
#define GCACHE_TX_LOG__ABORT_INS	(GCACHE_TX_LOG__MAGIC | 'A')
#define GCACHE_TX_LOG__ABORT_DEL	(GCACHE_TX_LOG__MAGIC | 'a')

typedef struct {
    uint32_t	type;
    uint32_t	length;
	char		data[1];		/* variable length */
} GCacheTxLogCommon;

typedef struct {
	uint32_t	type;
	uint32_t	length;
	uint32_t	rowid;
	uint32_t	__padding__;
	HeapTupleHeaderData htup;
} GCacheTxLogInsert;

typedef struct {
	uint32_t	type;
	uint32_t	length;
	uint32_t	xid;
	uint32_t	rowid;
	ItemPointerData ctid;
} GCacheTxLogDelete;

/*
 * COMMIT/ABORT
 */
typedef struct {
	uint32_t	type;
	uint32_t	length;
	uint32_t	rowid;
	uint32_t	__padding__;
} GCacheTxLogXact;

/*
 * REDO Log Buffer
 */
typedef struct {
	kern_errorbuf	kerror;
	size_t			length;
	uint32_t		nrooms;
	uint32_t		nitems;
	uint32_t		redo_items[1];
} kern_gpucache_redolog;

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
				  kern_data_store *kds_dst,
				  kern_data_store *kds_fallback);

#endif	/* CUDA_COMMON_H */
