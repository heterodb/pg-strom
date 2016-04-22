/*
 * cuda_gpujoin.h
 *
 * GPU accelerated parallel relations join based on hash-join or
 * nested-loop logic.
 * --
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */
#ifndef CUDA_GPUJOIN_H
#define CUDA_GPUJOIN_H

/*
 * definition of the inner relations structure. it can load multiple
 * kern_data_store or kern_hash_table.
 */
typedef struct
{
	cl_uint			pg_crc32_table[256];	/* used to hashjoin */
	cl_uint			nrels;			/* number of relations */
	cl_uint			ojmap_length;	/* length of outer-join map, if any */
	struct
	{
		cl_uint		chunk_offset;	/* offset to KDS or Hash */
		cl_uint		ojmap_offset;	/* offset to outer-join map, if any */
		cl_bool		is_nestloop;	/* true, if NestLoop. */
		cl_bool		left_outer;		/* true, if JOIN_LEFT or JOIN_FULL */
		cl_bool		right_outer;	/* true, if JOIN_RIGHT or JOIN_FULL */
		cl_char		__padding__[1];
	} chunks[FLEXIBLE_ARRAY_MEMBER];
} kern_multirels;

#define KERN_MULTIRELS_INNER_KDS(kmrels, depth)	\
	((kern_data_store *)						\
	 ((char *)(kmrels) + (kmrels)->chunks[(depth)-1].chunk_offset))

#define KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth, outer_join_map)	\
	((cl_bool *)((kmrels)->chunks[(depth)-1].right_outer				\
				 ? ((char *)(outer_join_map) +							\
					(kmrels)->chunks[(depth)-1].ojmap_offset)			\
				 : NULL))

#define KERN_MULTIRELS_LEFT_OUTER_JOIN(kmrels, depth)	\
	((kmrels)->chunks[(depth)-1].left_outer)

#define KERN_MULTIRELS_RIGHT_OUTER_JOIN(kmrels, depth)	\
	((kmrels)->chunks[(depth)-1].right_outer)

/*
 * kern_gpujoin - control object of GpuJoin
 */
typedef struct
{
	cl_uint		window_base;	/* base of the virtual partition window */
	cl_uint		window_size;	/* size of the virtual partition window */
	cl_uint		inner_nitems;	/* out: number of inner join results */
	cl_uint		total_nitems;	/* out: number of (inner+outer) join results */
	cl_float	row_dist_score;	/* out: count of non-zero histgram items on
								 * window resize. larger score means more
								 * distributed depth, thus to be target of
								 * the window split */
} kern_join_scale;

typedef struct
{
	cl_uint			kparams_offset;		/* offset to the kparams */
	cl_uint			kresults_1_offset;	/* offset to the 1st kresults buffer */
	cl_uint			kresults_2_offset;	/* offset to the 2nd kresults buffer */
	cl_uint			kresults_max_items;	/* max items kresult_buf can hold */
	/* number of inner relations */
	cl_uint			num_rels;
	/* error status to be backed (OUT) */
	kern_errorbuf	kerror;
	/*
	 * Performance statistics
	 */
	cl_uint			num_kern_outer_scan;
	cl_uint			num_kern_exec_nestloop;
	cl_uint			num_kern_exec_hashjoin;
	cl_uint			num_kern_outer_nestloop;
	cl_uint			num_kern_outer_hashjoin;
	cl_uint			num_kern_projection;
	cl_uint			num_kern_rows_dist;
	cl_float		tv_kern_outer_scan;
	cl_float		tv_kern_exec_nestloop;
	cl_float		tv_kern_exec_hashjoin;
	cl_float		tv_kern_outer_nestloop;
	cl_float		tv_kern_outer_hashjoin;
	cl_float		tv_kern_projection;
	cl_float		tv_kern_rows_dist;
	cl_uint			num_minor_retry;
	cl_uint			num_major_retry;
	/*
	 * Scale of inner virtual window for each depth
	 * (note that jscale has (num_rels + 1) elements
	 */
	kern_join_scale	jscale[FLEXIBLE_ARRAY_MEMBER];
} kern_gpujoin;

#define KERN_GPUJOIN_PARAMBUF(kgjoin)			\
	((kern_parambuf *)((char *)(kgjoin) + (kgjoin)->kparams_offset))
#define KERN_GPUJOIN_PARAMBUF_LENGTH(kgjoin)	\
	STROMALIGN(KERN_GPUJOIN_PARAMBUF(kgjoin)->length)
#define KERN_GPUJOIN_HEAD_LENGTH(kgjoin)				\
	STROMALIGN((char *)KERN_GPUJOIN_PARAMBUF(kgjoin) +	\
			   KERN_GPUJOIN_PARAMBUF_LENGTH(kgjoin) -	\
			   (char *)(kgjoin))

#define KERN_GPUJOIN_1ST_RESULTBUF(kgjoin)		\
	((kern_resultbuf *)((char *)(kgjoin) + (kgjoin)->kresults_1_offset))
#define KERN_GPUJOIN_2ND_RESULTBUF(kgjoin)		\
	((kern_resultbuf *)((char *)(kgjoin) + (kgjoin)->kresults_2_offset))

#ifdef __CUDACC__

/* utility macros for automatically generated code */
#define GPUJOIN_REF_HTUP(chunk,offset)			\
	((offset) == 0								\
	 ? NULL										\
	 : (HeapTupleHeaderData *)((char *)(chunk) + (size_t)(offset)))
/* utility macros for automatically generated code */
#define GPUJOIN_REF_DATUM(colmeta,htup,colidx)	\
	(!(htup) ? NULL : kern_get_datum_tuple((colmeta),(htup),(colidx)))

/*
 * gpujoin_join_quals
 *
 * Evaluation of join qualifier in the given depth. It shall return true
 * if supplied pair of the rows matches the join condition.
 *
 * NOTE: if x-axil (outer input) or y-axil (inner input) are out of range,
 * we expect outer_index or inner_htup are NULL. Don't skip to call this
 * function, because nested-loop internally uses __syncthread operation
 * to reduce DRAM accesses.
 */
STATIC_FUNCTION(cl_bool)
gpujoin_join_quals(kern_context *kcxt,
				   kern_data_store *kds,
				   kern_multirels *kmrels,
				   int depth,
				   cl_uint *x_buffer,
				   HeapTupleHeaderData *inner_htup,
				   cl_bool *joinquals_matched);

/*
 * gpujoin_hash_value
 *
 * Calculation of hash value if this depth uses hash-join logic.
 */
STATIC_FUNCTION(cl_uint)
gpujoin_hash_value(kern_context *kcxt,
				   cl_uint *pg_crc32_table,
				   kern_data_store *kds,
				   kern_multirels *kmrels,
				   cl_int depth,
				   cl_uint *x_buffer,
				   cl_bool *p_is_null_keys);

/*
 * gpujoin_projection
 *
 * Implementation of device projection. Extract a pair of outer/inner tuples
 * on the tup_values/tup_isnull array.
 */
STATIC_FUNCTION(void)
gpujoin_projection(kern_context *kcxt,
				   kern_data_store *kds_src,
				   kern_multirels *kmrels,
				   cl_uint *r_buffer,
				   kern_data_store *kds_dst,
				   Datum *tup_values,
				   cl_bool *tup_isnull,
				   cl_short *tup_depth,
				   cl_char *extra_buf,
				   cl_uint *extra_len);

/*
 * argument layout to launch inner/outer join functions
 */
typedef struct
{
	kern_gpujoin	   *kgjoin;
	kern_data_store	   *kds;
	kern_multirels	   *kmrels;
	kern_resultbuf	   *kresults_src;
	kern_resultbuf	   *kresults_dst;
	cl_bool			   *outer_join_map;
	cl_int				depth;
	cl_int				cuda_index;
	cl_uint				window_base;
	cl_uint				window_size;
} kern_join_args_t;

/*
 * gpujoin_exec_outerscan
 *
 * Evaluation of outer-relation's qualifier, if any. Elsewhere, it always
 * returns true.
 */
STATIC_FUNCTION(cl_bool)
gpujoin_outer_quals(kern_context *kcxt,
					kern_data_store *kds,
					size_t kds_index);

KERNEL_FUNCTION(void)
gpujoin_exec_outerscan(kern_gpujoin *kgjoin,
					   kern_data_store *kds,
					   kern_resultbuf *kresults)
{
	kern_parambuf  *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_context	kcxt;
	cl_uint			window_base = kgjoin->jscale[0].window_base;
	cl_uint			window_size = kgjoin->jscale[0].window_size;
	cl_uint			kds_index = window_base + get_global_id();
	cl_uint			count;
	cl_uint			offset;
	cl_bool			matched;
	__shared__ cl_int base;

	INIT_KERNEL_CONTEXT(&kcxt,gpujoin_exec_outerscan,kparams);
	assert(kresults->nrels == 1);	/* only happen if depth == 1 */

	if (kds_index < min(kds->nitems, window_base + window_size))
		matched = gpujoin_outer_quals(&kcxt, kds, kds_index);
	else
		matched = false;

	/* expand kresults->nitems */
	offset = arithmetic_stairlike_add(matched ? 1 : 0, &count);
	if (count > 0)
	{
		if (get_local_id() == 0)
			base = atomicAdd(&kresults->nitems, count);
		__syncthreads();

		if (base + offset >= kresults->nrooms)
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		else if (matched)
		{
			HeapTupleHeaderData *htup = kern_get_tuple_row(kds, kds_index);
			kresults->results[base + offset] = (size_t)htup - (size_t)kds;
		}
	}
	kern_writeback_error_status(&kgjoin->kerror, kcxt.e);
}

KERNEL_FUNCTION(void)
gpujoin_exec_nestloop(kern_gpujoin *kgjoin,
					  kern_data_store *kds,
					  kern_multirels *kmrels,
					  kern_resultbuf *kresults_src,
					  kern_resultbuf *kresults_dst,
					  cl_bool *outer_join_map,
					  cl_int depth,
					  cl_int cuda_index,
					  cl_uint window_base,
					  cl_uint window_size)
{
	kern_parambuf	   *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_context		kcxt;
	kern_data_store	   *kds_in;
	cl_bool			   *oj_map;
	HeapTupleHeaderData *y_htup;
	cl_uint			   *x_buffer;
	cl_uint			   *r_buffer;
	cl_uint				x_index;
	cl_uint				x_limit;
	cl_uint				y_index;
	cl_uint				y_limit;
	cl_uint				y_offset = UINT_MAX;	/* poison initial value */
	cl_bool				is_matched;
	cl_uint				offset;
	cl_uint				count;
	__shared__ cl_uint	base;

	INIT_KERNEL_CONTEXT(&kcxt,gpujoin_exec_nestloop,kparams);

	/* sanity checks */
	assert(depth > 0 && depth <= kgjoin->num_rels);
	assert(kresults_dst->nrels == depth + 1);
	assert(kresults_src->nrels == depth);
	assert(kresults_src->nitems <= kresults_src->nrooms);

	/* inner tuple is pointed by y_index */
	kds_in = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	assert(kds_in != NULL);

	/*
	 * Because of a historic reason, we call index of outer tuple 'x_index',
	 * and index of inner tuple 'y_index'.
	 */
	x_limit = kresults_src->nitems;
	x_index = get_global_id() % x_limit;
	y_limit = min(window_base + window_size, kds_in->nitems);
	y_index = window_base + (get_global_id() / x_limit);

	/* will be valid, if LEFT OUTER JOIN */
	oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth, outer_join_map);
	/* inside of the range? */
	if (x_index < x_limit && y_index < y_limit)
	{
		/* outer input */
		x_buffer = KERN_GET_RESULT(kresults_src, x_index);
		/* inner input */
		y_htup = kern_get_tuple_row(kds_in, y_index);

		/* does it satisfies join condition? */
		is_matched = gpujoin_join_quals(&kcxt,
										kds,
										kmrels,
										depth,
										x_buffer,
										y_htup,
										NULL);
		if (is_matched)
		{
			y_offset = (size_t)y_htup - (size_t)kds_in;
			if (oj_map && !oj_map[y_index])
				oj_map[y_index] = true;
		}
	}
	else
		is_matched = false;

	__syncthreads();

	/*
	 * Expand kresults_dst->nitems, and put values
	 */
	offset = arithmetic_stairlike_add(is_matched ? 1 : 0, &count);
	if (count > 0)
	{
		if (get_local_id() == 0)
			base = atomicAdd(&kresults_dst->nitems, count);
		__syncthreads();

		/* still have space to store? */
		if (base + count >= kresults_dst->nrooms)
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		else if (is_matched)
		{
			assert(x_buffer != NULL && y_htup != NULL);
			r_buffer = KERN_GET_RESULT(kresults_dst, base + offset);
			memcpy(r_buffer, x_buffer, sizeof(cl_int) * depth);
			r_buffer[depth] = y_offset;
		}
		__syncthreads();
	}
	kern_writeback_error_status(&kgjoin->kerror, kcxt.e);
}

/*
 * gpujoin_exec_hashjoin
 *
 *
 *
 */
KERNEL_FUNCTION(void)
gpujoin_exec_hashjoin(kern_gpujoin *kgjoin,
					  kern_data_store *kds,
					  kern_multirels *kmrels,
					  kern_resultbuf *kresults_src,
					  kern_resultbuf *kresults_dst,
					  cl_bool *outer_join_map,
					  cl_int depth,
					  cl_int cuda_index,
					  cl_uint window_base,
					  cl_uint window_size)
{
	kern_parambuf	   *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_data_store	   *kds_hash = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	kern_context		kcxt;
	kern_hashitem	   *khitem = NULL;
	cl_uint			   *x_buffer = NULL;
	cl_uint			   *r_buffer;
	cl_bool			   *oj_map;
	cl_uint				crc_index;
	cl_uint				hash_value;
	cl_uint				offset;
	cl_uint				count;
	cl_bool				is_matched;
	cl_bool				needs_outer_row = false;
	cl_bool				is_null_keys;
	__shared__ cl_uint	base;
	__shared__ cl_uint	pg_crc32_table[256];

	INIT_KERNEL_CONTEXT(&kcxt,gpujoin_exec_hashjoin,kparams);

	/* sanity checks */
	assert(depth > 0 && depth <= kgjoin->num_rels);
	assert(kresults_dst->nrels == depth + 1);
	assert(kresults_src->nrels == depth);
	assert(kresults_src->nitems <= kresults_src->nrooms);

	/* move crc32 table to __local memory from __global memory.
	 *
	 * NOTE: calculation of hash value (based on crc32 in GpuHashJoin) is
	 * the core of calculation workload in the GpuHashJoin implementation.
	 * If we keep the master table is global memory, it will cause massive
	 * amount of computing core stall because of RAM access latency.
	 * So, we try to move them into local shared memory at the beginning.
	 */
	for (crc_index = get_local_id();
		 crc_index < 256;
		 crc_index += get_local_size())
	{
		pg_crc32_table[crc_index] = kmrels->pg_crc32_table[crc_index];
	}
	__syncthreads();

	/* will be valid, if RIGHT OUTER JOIN */
	oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth, outer_join_map);

	/* Calculation of hash-value by the outer join keys */
	if (get_global_id() < kresults_src->nitems)
	{
		x_buffer = KERN_GET_RESULT(kresults_src, get_global_id());
		assert(((size_t)x_buffer[0] & (sizeof(cl_ulong) - 1)) == 0);
		hash_value = gpujoin_hash_value(&kcxt,
										pg_crc32_table,
										kds,
										kmrels,
										depth,
										x_buffer,
										&is_null_keys);
		if (hash_value >= kds_hash->hash_min &&
			hash_value <= kds_hash->hash_max)
		{
			/* NOTE: NULL-keys never match on inner join */
			if (!is_null_keys)
				khitem = KERN_HASH_FIRST_ITEM(kds_hash, hash_value);
			needs_outer_row = true;
		}
	}

	/*
	 * Walks on the hash entries chain from the khitem
	 */
	do {
		if (khitem && (khitem->hash  == hash_value &&
					   khitem->rowid >= window_base &&
					   khitem->rowid <  window_base + window_size))
		{
			HeapTupleHeaderData *h_htup = &khitem->t.htup;
			cl_bool			joinquals_matched;

			is_matched = gpujoin_join_quals(&kcxt,
											kds,
											kmrels,
											depth,
											x_buffer,
											h_htup,
											&joinquals_matched);
			if (joinquals_matched)
			{
				/* no need LEFT/FULL OUTER JOIN */
				needs_outer_row = false;
				/* no need RIGHT/FULL OUTER JOIN */
				assert(khitem->rowid < kds_hash->nitems);
				if (oj_map && !oj_map[khitem->rowid])
					oj_map[khitem->rowid] = true;
			}
		}
		else
			is_matched = false;

		/* Expand kresults_dst->nitems */
		offset = arithmetic_stairlike_add(is_matched ? 1 : 0, &count);
		if (count > 0)
		{
			if (get_local_id() == 0)
				base = atomicAdd(&kresults_dst->nitems, count);
			__syncthreads();

			/* kresults_dst still have enough space? */
			if (base + count >= kresults_dst->nrooms)
				STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			else if (is_matched)
			{
				r_buffer = KERN_GET_RESULT(kresults_dst, base + offset);
				memcpy(r_buffer, x_buffer, sizeof(cl_int) * depth);
				r_buffer[depth] = (size_t)&khitem->t.htup - (size_t)kds_hash;
			}
		}
		__syncthreads();

		/*
		 * Fetch next hash entry, then checks whether all the local
		 * threads still have valid hash-entry or not.
		 * (NOTE: this routine contains reduction operation)
		 */
		khitem = KERN_HASH_NEXT_ITEM(kds_hash, khitem);
		arithmetic_stairlike_add(khitem != NULL ? 1 : 0, &count);
	} while (count > 0);

	/*
	 * If no inner rows were matched on LEFT OUTER JOIN case, we fill up
	 * the inner-side of result tuple with NULL.
	 */
	if (KERN_MULTIRELS_LEFT_OUTER_JOIN(kmrels, depth))
	{
		if (needs_outer_row)
		{
			assert(x_buffer != NULL);
			is_matched = gpujoin_join_quals(&kcxt,
											kds,
											kmrels,
											depth,
											x_buffer,
											NULL,
											NULL);
		}
		else
			is_matched = false;

		offset = arithmetic_stairlike_add(is_matched ? 1 : 0, &count);
		if (count > 0)
		{
			if (get_local_id() == 0)
				base = atomicAdd(&kresults_dst->nitems, count);
			__syncthreads();

			/* kresults_dst still have enough space? */
			if (base + count >= kresults_dst->nrooms)
				STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			else if (is_matched)
			{
				assert(x_buffer != NULL);
				r_buffer = KERN_GET_RESULT(kresults_dst, base + offset);
				memcpy(r_buffer, x_buffer, sizeof(cl_int) * depth);
				r_buffer[depth] = 0;	/* inner NULL */
			}
		}
	}
	kern_writeback_error_status(&kgjoin->kerror, kcxt.e);
}

/*
 * gpujoin_collocate_outer_join_map
 *
 * it merges the result of other GPU devices and CPU fallback
 */
KERNEL_FUNCTION(void)
gpujoin_colocate_outer_join_map(kern_multirels *kmrels,
								cl_bool *outer_join_map)
{
	cl_uint		nitems = kmrels->ojmap_length / sizeof(cl_uint);
	cl_uint	   *self_map = (cl_uint *)outer_join_map;
	cl_uint	   *recv_map = (cl_uint *)((char *)outer_join_map +
									   kmrels->ojmap_length);
	if (get_global_id() < nitems)
		self_map[get_global_id()] |= recv_map[get_global_id()];
}

/*
 * gpujoin_outer_nestloop
 *
 * It injects unmatched tuples to kresuts_out buffer if RIGHT OUTER JOIN
 */
KERNEL_FUNCTION(void)
gpujoin_outer_nestloop(kern_gpujoin *kgjoin,
					   kern_data_store *kds,	/* never referenced */
					   kern_multirels *kmrels,
					   kern_resultbuf *kresults_src,	/* never referenced */
					   kern_resultbuf *kresults_dst,
					   cl_bool *outer_join_map,
					   cl_int depth,
					   cl_int cuda_index,
					   cl_uint window_base,
					   cl_uint window_size)
{
	kern_parambuf	   *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_data_store	   *kds_in = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	kern_context		kcxt;
	cl_bool				needs_outer_row;
	cl_uint				y_index = window_base + get_global_id();
	cl_uint				y_limit = min(window_base + window_size,
									  kds_in->nitems);
	cl_uint				count;
	cl_uint				offset;
	cl_uint			   *r_buffer;
	__shared__ cl_uint	base;

	INIT_KERNEL_CONTEXT(&kcxt,gpujoin_outer_nestloop,kparams);

	/* sanity checks */
	assert(depth > 0 && depth <= kgjoin->num_rels);
	assert(kresults_dst->nrels == depth + 1);

	/*
	 * check whether the relevant inner tuple has any matched outer tuples,
	 * including the jobs by other devices.
	 */
	if (y_index < y_limit)
	{
		cl_uint		nitems = kds_in->nitems;
		cl_bool	   *oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth,
														   outer_join_map);

		assert(oj_map != NULL);
		needs_outer_row = (oj_map[y_index] ? false : true);
		/* check non-join-quals again */
        if (needs_outer_row)
        {
			HeapTupleHeaderData *htup = kern_get_tuple_row(kds_in, y_index);
            needs_outer_row = gpujoin_join_quals(&kcxt,
												 kds,
												 kmrels,
												 depth,
												 NULL,	/* NULL for Left */
												 htup,
												 NULL);
		}
	}
	else
		needs_outer_row = false;

	/*
	 * Count up number of inner tuples that were not matched with outer-
	 * relations. Then, we allocates slot in kresults_dst for outer-join
	 * tuples.
	 */
	offset = arithmetic_stairlike_add(needs_outer_row ? 1 : 0, &count);
	if (count > 0)
	{
		if (get_local_id() == 0)
			base = atomicAdd(&kresults_dst->nitems, count);
		__syncthreads();

		/* Does kresults_dst still have rooms to store? */
		if (base + count >= kresults_dst->nrooms)
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		else if (needs_outer_row)
		{
			/*
			 * OK, we know which row should be materialized using left outer
			 * join manner, and result buffer was acquired. Let's put result
			 * for the next stage.
			 */
			HeapTupleHeaderData	*htup = kern_get_tuple_row(kds_in, y_index);
			r_buffer = KERN_GET_RESULT(kresults_dst, base + offset);
			memset(r_buffer, 0, sizeof(cl_int) * depth);	/* NULL */
			r_buffer[depth] = (size_t)htup - (size_t)kds_in;
		}
	}
	__syncthreads();
	kern_writeback_error_status(&kgjoin->kerror, kcxt.e);
}

/*
 * gpujoin_outer_hashjoin
 *
 * It injects unmatched tuples to kresuts_out buffer if RIGHT OUTER JOIN.
 * We expect kernel is launched with larger than nslots threads.
 */
KERNEL_FUNCTION(void)
gpujoin_outer_hashjoin(kern_gpujoin *kgjoin,
					   kern_data_store *kds,	/* never referenced */
					   kern_multirels *kmrels,
					   kern_resultbuf *kresults_src,
					   kern_resultbuf *kresults_dst,
					   cl_bool *outer_join_map,
					   cl_int depth,
					   cl_int cuda_index,
					   cl_uint window_base,
					   cl_uint window_size)
{
	kern_parambuf	   *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_data_store	   *kds_hash = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	kern_hashitem	   *khitem = NULL;
	kern_context		kcxt;
	cl_uint				kds_index;
	cl_bool				needs_outer_row;
	cl_uint				offset;
	cl_uint				count;
	cl_uint			   *r_buffer;
	__shared__ cl_uint	base;

	INIT_KERNEL_CONTEXT(&kcxt,gpujoin_outer_hashjoin,kparams);

	/* sanity checks */
	assert(depth > 0 && depth <= kgjoin->num_rels);
	assert(kresults_dst->nrels == depth + 1);
	assert(window_base + window_size <= kds_hash->nitems);
	assert(window_size > 0);

	kds_index = window_base + get_global_id();
	if (kds_index < min(window_base + window_size,
						kds_hash->nitems))
	{
		cl_bool	   *oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth,
														   outer_join_map);
		khitem = KERN_DATA_STORE_HASHITEM(kds_hash, kds_index);
		assert(khitem->rowid == kds_index);
		needs_outer_row = (oj_map[kds_index] ? false : true);

		/* check non-join-quals again */
		if (needs_outer_row)
		{
			needs_outer_row = gpujoin_join_quals(&kcxt,
												 kds,
												 kmrels,
												 depth,
												 NULL,	/* NULL for Left */
												 &khitem->t.htup,
												 NULL);
		}
	}
	else
		needs_outer_row = false;

	/* expand kresults->nitems */
	offset = arithmetic_stairlike_add(needs_outer_row ? 1 : 0, &count);
	if (count > 0)
	{
		if (get_local_id() == 0)
			base = atomicAdd(&kresults_dst->nitems, count);
		__syncthreads();

		if (base + offset >= kresults_dst->nrooms)
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		else if (needs_outer_row)
		{
			r_buffer = KERN_GET_RESULT(kresults_dst, base + offset);
			memset(r_buffer, 0, sizeof(cl_uint) * depth);	/* NULL */
			assert((size_t)&khitem->t.htup > (size_t)kds_hash);
			r_buffer[depth] = (size_t)&khitem->t.htup - (size_t)kds_hash;
		}
	}
	kern_writeback_error_status(&kgjoin->kerror, kcxt.e);
}

/*
 * gpujoin_projection_row
 *
 * It makes joined relation on kds_dst
 */

KERNEL_FUNCTION(void)
gpujoin_projection_row(kern_gpujoin *kgjoin,
					   kern_multirels *kmrels,
					   kern_data_store *kds_src,
					   kern_data_store *kds_dst,
					   kern_resultbuf *kresults)
{
	kern_parambuf  *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_context	kcxt;
	cl_uint			res_index = get_global_id();
	cl_uint			required;
	cl_uint			offset;
	cl_uint			count;
	__shared__ cl_uint base;
#if GPUJOIN_DEVICE_PROJECTION_NFIELDS > 0
	Datum			tup_values[GPUJOIN_DEVICE_PROJECTION_NFIELDS];
	cl_bool			tup_isnull[GPUJOIN_DEVICE_PROJECTION_NFIELDS];
	cl_short		tup_depth[GPUJOIN_DEVICE_PROJECTION_NFIELDS];
#else
	Datum		   *tup_values = NULL;
	cl_bool		   *tup_isnull = NULL;
	cl_short	   *tup_depth = NULL;
#endif

	/* sanity checks */
	assert(kresults->nrels == kgjoin->num_rels + 1);
	assert(kds_src == NULL || kds_src->format == KDS_FORMAT_ROW);
	assert(kds_dst->format == KDS_FORMAT_ROW && kds_dst->nslots == 0);

	INIT_KERNEL_CONTEXT(&kcxt, gpujoin_projection_row, kparams);

	assert(kds_dst->nitems == kresults->nitems &&
		   kds_dst->nitems <= kds_dst->nrooms);
	if (res_index < kds_dst->nitems)
	{
#if GPUJOIN_DEVICE_PROJECTION_EXTRA_SIZE > 0
		cl_char		extra_buf[GPUJOIN_DEVICE_PROJECTION_EXTRA_SIZE]
					__attribute__ ((aligned(MAXIMUM_ALIGNOF)));
#else
		cl_char	   *extra_buf = NULL;
#endif
		cl_uint		extra_len;

		/*
		 * result buffer
		 * -------------
		 * r_buffer[0] -> offset from the 'kds_src'
		 * r_buffer[i; i > 0] -> offset from the kern_data_store of individual
		 *   depth in the kern_multirels buffer.
		 *   (can be picked up using KERN_MULTIRELS_INNER_KDS)
		 * r_buffer[*] may be 0, if NULL-tuple was set
		 */

		/*
		 * Step.1 - compute length of the result tuple to be written
		 */
		cl_uint	   *r_buffer = KERN_GET_RESULT(kresults, res_index);

		gpujoin_projection(&kcxt,
						   kds_src,
						   kmrels,
						   r_buffer,
						   kds_dst,
						   tup_values,
						   tup_isnull,
						   tup_depth,
						   extra_buf,
						   &extra_len);
		assert(extra_len <= GPUJOIN_DEVICE_PROJECTION_EXTRA_SIZE);
		required = MAXALIGN(offsetof(kern_tupitem, htup) +
							compute_heaptuple_size(&kcxt,
												   kds_dst,
												   tup_values,
												   tup_isnull,
												   NULL));
	}
	else
		required = 0;	/* out of the range */

	/*
	 * Step.2 - increment the buffer usage of kds_dst
	 */
	offset = arithmetic_stairlike_add(required, &count);
	if (count > 0)
	{
		if (get_local_id() == 0)
			base = atomicAdd(&kds_dst->usage, count);
		__syncthreads();

		if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
			STROMALIGN(sizeof(cl_uint) * kresults->nitems) +
			base + count > kds_dst->length)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			goto out;
		}
	}
	__syncthreads();

	/*
	 * Step.3 - write out the HeapTuple on the destination buffer
	 */
	if (required > 0)
	{
		cl_uint			pos = kds_dst->length - (base + offset + required);
		cl_uint		   *tup_pos = KERN_DATA_STORE_ROWINDEX(kds_dst);
		kern_tupitem   *tupitem = (kern_tupitem *)((char *)kds_dst + pos);

		tup_pos[res_index] = pos;
		form_kern_heaptuple(&kcxt, kds_dst, tupitem,
							tup_values, tup_isnull, NULL);
	}
out:
	/* write-back execution status to host-side */
	kern_writeback_error_status(&kgjoin->kerror, kcxt.e);
}

KERNEL_FUNCTION(void)
gpujoin_projection_slot(kern_gpujoin *kgjoin,
						kern_multirels *kmrels,
						kern_data_store *kds_src,
						kern_data_store *kds_dst,
						kern_resultbuf *kresults)
{
	kern_parambuf  *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_context	kcxt;
	cl_uint		   *r_buffer;
	Datum		   *tup_values;
	cl_bool		   *tup_isnull;
#if GPUJOIN_DEVICE_PROJECTION_NFIELDS > 0
	cl_short		tup_depth[GPUJOIN_DEVICE_PROJECTION_NFIELDS];
#else
	cl_short	   *tup_depth = NULL;
#endif
#if GPUJOIN_DEVICE_PROJECTION_EXTRA_SIZE > 0
	cl_char			extra_buf[GPUJOIN_DEVICE_PROJECTION_EXTRA_SIZE]
					__attribute__ ((aligned(MAXIMUM_ALIGNOF)));
#else
	cl_char		   *extra_buf = NULL;
#endif
	cl_uint			extra_len;
	char		   *vl_buf	__attribute__ ((unused)) = NULL;
	cl_uint			offset	__attribute__ ((unused));
	cl_uint			count	__attribute__ ((unused));
	__shared__ cl_uint base	__attribute__ ((unused));
	cl_uint			res_index = get_global_id();

	/* sanity checks */
	assert(kresults->nrels == kgjoin->num_rels + 1);
	assert(kds_src == NULL || kds_src->format == KDS_FORMAT_ROW);
	assert(kds_dst->format == KDS_FORMAT_SLOT);

	INIT_KERNEL_CONTEXT(&kcxt, gpujoin_projection_slot, kparams);

	/* Do projection if thread is responsible */
	if (res_index < kresults->nitems)
	{
		r_buffer = KERN_GET_RESULT(kresults, res_index);
		tup_values = KERN_DATA_STORE_VALUES(kds_dst, res_index);
		tup_isnull = KERN_DATA_STORE_ISNULL(kds_dst, res_index);

		gpujoin_projection(&kcxt,
						   kds_src,
						   kmrels,
						   r_buffer,
						   kds_dst,
						   tup_values,
						   tup_isnull,
						   tup_depth,
						   extra_buf,
						   &extra_len);
	}
	else
		extra_len = 0;

#if GPUJOIN_DEVICE_PROJECTION_EXTRA_SIZE > 0
	/*
	 * In case when GpuJoin result contains any indirect or numeric
	 * data types, we have to allocate extra area on the kds_dst
	 * buffer to store the contents.
	 */
	assert(extra_len <= GPUJOIN_DEVICE_PROJECTION_EXTRA_SIZE);
	assert(extra_len == MAXALIGN(extra_len));
	offset = arithmetic_stairlike_add(extra_len, &count);
	if (count > 0)
	{
		if (get_local_id() == 0)
			base = atomicAdd(&kds_dst->usage, count);
		__syncthreads();

		if (KERN_DATA_STORE_SLOT_LENGTH(kds_dst, kresults->nitems) +
			base + count > kds_dst->length)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			goto out;
		}
		vl_buf = ((char *)kds_dst + kds_dst->length
				  - (base + offset + extra_len));
	}
	__syncthreads();
#else
	if (extra_len > 0)
	{
		STROM_SET_ERROR(&kcxt.e, StromError_WrongCodeGeneration);
		goto out;
	}
#endif
	/*
	 * At this point, tup_values have device pointer or internal
	 * data representation. We have to fix up these values to fit
	 * host-side representation.
	 */
	if (res_index < kresults->nitems)
	{
		cl_uint		i, ncols = kds_dst->ncols;

		for (i=0; i < ncols; i++)
		{
			kern_colmeta	cmeta = kds_dst->colmeta[i];

			if (tup_isnull[i])
				tup_values[i] = (Datum) 0;	/* clean up */
			else if (!cmeta.attbyval)
			{
				char   *addr = DatumGetPointer(tup_values[i]);

				if (tup_depth[i] == 0)
					tup_values[i] = devptr_to_host(kds_src, addr);
				else if (tup_depth[i] > 0)
				{
					kern_data_store *kds_in =
						KERN_MULTIRELS_INNER_KDS(kmrels, tup_depth[i]);
					tup_values[i] = devptr_to_host(kds_in, addr);
				}
				else if (tup_depth[i] == -1)
				{
					/* value is stored in the local array */
					cl_uint		vl_len = (cmeta.attlen > 0 ?
										  cmeta.attlen :
										  VARSIZE_ANY(addr));
					memcpy(vl_buf, addr, vl_len);
					tup_values[i] = devptr_to_host(kds_dst, vl_buf);
					vl_buf += MAXALIGN(vl_len);
				}
				else if (tup_depth[i] == -2)
				{
					/* value is simple reference to kparams */
					assert(addr >= (char *)kparams &&
						   addr <  (char *)kparams + kparams->length);
					tup_values[i] = devptr_to_host(kparams, addr);
				}
				else
					STROM_SET_ERROR(&kcxt.e, StromError_WrongCodeGeneration);
			}
		}
	}
out:
	/* write-back execution status to host-side */
	kern_writeback_error_status(&kgjoin->kerror, kcxt.e);
}

/*
 * gpujoin_count_rows_dist
 *
 * It counts a rough rows distribution degreee on buffer overflow.
 * No needs to be a strict count, so we use a histgram constructed on
 * a shared memory region as an indicator of the distribution.
 */
typedef struct
{
	kern_gpujoin   *kgjoin;
	kern_multirels *kmrels;
	kern_resultbuf *kresults;
	cl_int			thread_width;
} kern_rows_dist_args_t;

#define ROW_DIST_COUNT_MAX_THREAD_WIDTH		18
#define ROW_DIST_HIST_SIZE		(1024 * ROW_DIST_COUNT_MAX_THREAD_WIDTH)

KERNEL_FUNCTION_MAXTHREADS(void)
gpujoin_count_rows_dist(kern_gpujoin *kgjoin,
						kern_multirels *kmrels,
						kern_resultbuf *kresults,
						cl_int thread_width)
{
	cl_uint		nvalids = min(kresults->nitems, kresults->nrooms);
	cl_uint		limit;
	cl_uint		i, j;
	__shared__ cl_uint	pg_crc32_table[256];
	__shared__ cl_bool	row_dist_hist[ROW_DIST_HIST_SIZE];

	/* move crc32 table to __local memory from __global memory */
	for (i = get_local_id();
		 i < 256;
		 i += get_local_size())
	{
		pg_crc32_table[i] = kmrels->pg_crc32_table[i];
	}
	__syncthreads();

	limit = ((ROW_DIST_HIST_SIZE +
			  get_local_size() - 1) / get_local_size()) * get_local_size();

	for (i=0; i < kresults->nrels; i++)
	{
		cl_uint		count = 0;
		cl_uint		hash;

		/* clear the row distribution histgram of this depth */
		for (j = get_local_id();
			 j < ROW_DIST_HIST_SIZE;
			 j += get_local_size())
		{
			row_dist_hist[j] = 0;
		}
		__syncthreads();

		/* Makes row distribution histgram */
		for (j=0; j < thread_width; j++)
		{
			cl_uint		r_index = get_global_id() * thread_width + j;

			if (r_index < nvalids)
			{
				cl_uint	   *r_buffer = KERN_GET_RESULT(kresults, r_index);

				hash = pg_common_comp_crc32(pg_crc32_table, 0U,
											(const char *)(r_buffer + i),
											sizeof(cl_uint));
				row_dist_hist[hash % ROW_DIST_HIST_SIZE] = 1;
			}
		}
		__syncthreads();

		/* Count the row distribution histgram */
		for (j = get_local_id();
			 j < limit;
			 j += get_local_size())
		{
			count += __syncthreads_count(j < ROW_DIST_HIST_SIZE
										 ? (int)row_dist_hist[j]
										 : 0);
		}

		if (get_local_id() == 0)
			atomicAdd(&kgjoin->jscale[i].row_dist_score, (cl_float)count);
		__syncthreads();
	}
}

/*
 * returns true, if major retry, elsewhere minor retry
 *
 *
 */
STATIC_FUNCTION(cl_int)
gpujoin_resize_window(kern_gpujoin *kgjoin,
					  kern_multirels *kmrels,
					  kern_data_store *kds_src,
					  kern_resultbuf *kresults,
					  cl_int nsplits,
					  cl_int smx_count,
					  cl_int smx_clock)
{
	cudaFuncAttributes fattrs;
	kern_rows_dist_args_t *kern_args;
	dim3			grid_sz;
	dim3			block_sz;
	cl_uint			nvalids;
	cl_uint			unitsz;
	cl_uint			thread_width;
	cl_uint			depth;
	cl_uint			target_depth = 0;
	cl_float		row_dist_score_largest = FLT_MIN;
	cl_ulong		tv1, tv2;
	kern_errorbuf	kerror_save;
	cudaError_t		status = cudaSuccess;

	/*
	 * clear the error code, but may need to restore if we cannot
	 * split the target window any more
	 */
	kerror_save = kgjoin->kerror;
	memset(&kgjoin->kerror, 0, sizeof(kern_errorbuf));

	/* get max available block size */
	status = cudaFuncGetAttributes(&fattrs, (const void *)
								   gpujoin_count_rows_dist);
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
		return -1;
	}

	/* how many items to be processed per thread? */
	nvalids = min(kresults->nitems, kresults->nrooms);
	thread_width = (nvalids - 1) / (smx_count * fattrs.maxThreadsPerBlock) + 1;
	if (thread_width > ROW_DIST_COUNT_MAX_THREAD_WIDTH)
		thread_width = ROW_DIST_COUNT_MAX_THREAD_WIDTH;

	/* allocation of argument buffer */
	kern_args = (kern_rows_dist_args_t *)
		cudaGetParameterBuffer(sizeof(void *),
							   sizeof(kern_rows_dist_args_t));
	if (!kern_args)
	{
		STROM_SET_ERROR(&kgjoin->kerror, StromError_OutOfKernelArgs);
		return -1;
	}
	kern_args->kgjoin = kgjoin;
	kern_args->kmrels = kmrels;
	kern_args->kresults = kresults;
	kern_args->thread_width = thread_width;

	/* special calculation of the kernel block size */
	block_sz.x = fattrs.maxThreadsPerBlock;
	block_sz.y = 1;
	block_sz.z = 1;
	unitsz = thread_width * fattrs.maxThreadsPerBlock;
	grid_sz.x = (nvalids - 1) / unitsz + 1;
	grid_sz.y = 1;
	grid_sz.z = 1;

	/* OK, launch the histgram calculation kernel */
	status = cudaLaunchDevice((void *)gpujoin_count_rows_dist,
							  kern_args, grid_sz, block_sz,
							  0,
							  NULL);
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
		return -1;
	}

	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
		return -1;
	}
	tv2 = clock64();
	TIMEVAL_RECORD(kgjoin,kern_rows_dist,tv1,tv2,smx_clock);
	if (kgjoin->kerror.errcode != StromError_Success)
		return -1;
	/*
	 * Find out the most distributed depth
	 */
	for (depth=0; depth < kresults->nrels; depth++)
	{
		cl_uint     window_size = kgjoin->jscale[depth].window_size;
		cl_float    row_dist_score;

		if (window_size == 0)
			continue;
		/*
		 * NOTE: Adjustment of the row-distribution score
		 *
		 * Because of performance reason, we don't make a strict histgram
		 * (it is almost equivalent to GpuPreAgg!). It is a sum of per-block
		 * summary, thus, may contain duplications to be adjusted.
		 * A unit size of the histgram is (thread-width * block-size).
		 * If window size is less than the unit-size, it is obvious that rows
		 * may be distributed in any histgrams. So, simply we divide the score
		 * by grid-size.
		 * If window size is larger than unit-size, we have to pay attention
		 * for duplications across histgram. We adopted a simple approximate
		 * that assumes rows will duplicate according to the square root of
		 * number of histgrams.
		 */
		row_dist_score = kgjoin->jscale[depth].row_dist_score;
		if (window_size <= unitsz)
			row_dist_score /= (cl_float)grid_sz.x;
		else
			row_dist_score /= (1.0 + sqrt((cl_float)(grid_sz.x - 1)));

		if (row_dist_score > row_dist_score_largest)
		{
			target_depth = depth;
			row_dist_score_largest = row_dist_score;
		}
		kgjoin->jscale[depth].row_dist_score = row_dist_score;
	}

	/*
	 * Reduction of the virtual partition window. Of course, we cannot
	 * reduce the window size less than 1.
	 */
	kgjoin->jscale[target_depth].window_size
		= kgjoin->jscale[target_depth].window_size / nsplits + 1;
	if (kgjoin->jscale[target_depth].window_size <= 1)
	{
		kgjoin->kerror = kerror_save;
		return -1;
	}

	/*
	 * Inform caller the victim depth. If it is the last depth, we may
	 * not need to retry from the head, but from the last depth
	 */
	return target_depth;
}

/* NOTE: This macro assumes name of local variables in gpujoin_main() */
#define SETUP_KERN_JOIN_ARGS(argbuf)			\
	(argbuf)->kgjoin = kgjoin;					\
	(argbuf)->kds = kds_src;					\
	(argbuf)->kmrels = kmrels;					\
	(argbuf)->kresults_src = kresults_src;		\
	(argbuf)->kresults_dst = kresults_dst;		\
	(argbuf)->outer_join_map = outer_join_map;	\
	(argbuf)->depth = depth;					\
	(argbuf)->cuda_index = cuda_index;			\
	(argbuf)->window_base = window_base;		\
	(argbuf)->window_size = window_size

/*
 * gpujoin_main - controller function of GpuJoin logic
 */
KERNEL_FUNCTION(void)
gpujoin_main(kern_gpujoin *kgjoin,		/* in/out: misc stuffs */
			 kern_multirels *kmrels,	/* in: inner sources */
			 cl_bool *outer_join_map,	/* internal buffer */
			 kern_data_store *kds_src,	/* in: outer source (may be NULL) */
			 kern_data_store *kds_dst,	/* out: join results */
			 cl_int cuda_index)			/* device index on the host side */
{
	kern_parambuf	   *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_resultbuf	   *kresults_src = KERN_GPUJOIN_1ST_RESULTBUF(kgjoin);
	kern_resultbuf	   *kresults_dst = KERN_GPUJOIN_2ND_RESULTBUF(kgjoin);
	kern_resultbuf	   *kresults_tmp;
	kern_context		kcxt;
	const void		   *kernel_projection;
	void			  **kern_args;
	kern_join_args_t   *kern_join_args;
	dim3				grid_sz;
	dim3				block_sz;
	cl_uint				kresults_max_items = kgjoin->kresults_max_items;
	cl_uint				window_base;
	cl_uint				window_size;
	cl_int				device;
	cl_int				smx_clock;
	cl_int				smx_count;
	cl_int				depth;
	cl_int				nsplits;
	cl_int				victim;
	cl_ulong			tv1, tv2;
	cudaError_t			status = cudaSuccess;

	/* Init kernel context */
	INIT_KERNEL_CONTEXT(&kcxt, gpujoin_main, kparams);
	assert(get_global_size() == 1);		/* only single thread */
	assert(!kds_src || kds_src->format == KDS_FORMAT_ROW);
	assert(kds_dst->format == KDS_FORMAT_ROW ||
		   kds_dst->format == KDS_FORMAT_SLOT);

	/* Get device clock for performance monitor */
	status = cudaGetDevice(&device);
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
		goto out;
	}

	status = cudaDeviceGetAttribute(&smx_clock,
									cudaDevAttrClockRate,
									device);
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
		goto out;
	}

	status = cudaDeviceGetAttribute(&smx_count,
									cudaDevAttrMultiProcessorCount,
									device);
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
		goto out;
	}

retry_major:
	/* rewind the destination buffer */
	kds_dst->nitems = 0;
	kds_dst->usage = 0;

	for (depth = 1; depth <= kgjoin->num_rels; depth++)
	{
		cl_bool		exec_right_outer;

	retry_minor:
		/* Does this depth need to launch RIGHT OUTER JOIN? */
		exec_right_outer = (kds_src == NULL &&
							KERN_MULTIRELS_RIGHT_OUTER_JOIN(kmrels, depth));
		/*
		 * Initialization of the kresults_src buffer if start depth.
		 * Elsewhere, kresults_dst buffer of the last depth is also
		 * kresults_src buffer in this depth.
		 */
		if (depth == 1)
		{
			memset(kresults_src, 0, offsetof(kern_resultbuf, results[0]));
			kresults_src->nrels = depth;
			kresults_src->nrooms = kresults_max_items / depth;
			if (kds_src != NULL)
			{
				/* only happen if depth == 1 */
				assert(depth == 1);

				/* Launch:
				 * gpujoin_exec_outerscan(kern_gpujoin *kgjoin,
				 *                        kern_data_store *kds,
				 *                        kern_resultbuf *kresults)
				 */
				tv1 = clock64();
				kern_args = (void **)
					cudaGetParameterBuffer(sizeof(void *),
										   sizeof(void *) * 3);
				if (!kern_args)
				{
					STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
					goto out;
				}
				kern_args[0] = kgjoin;
				kern_args[1] = kds_src;
				kern_args[2] = kresults_src;

				status = pgstrom_optimal_workgroup_size(&grid_sz,
														&block_sz,
														(const void *)
														gpujoin_exec_outerscan,
														kds_src->nitems,
														sizeof(kern_errorbuf));
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
					goto out;
				}

				status = cudaLaunchDevice((void *)gpujoin_exec_outerscan,
										  kern_args, grid_sz, block_sz,
										  sizeof(kern_errorbuf) * block_sz.x,
										  NULL);
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
					goto out;
				}

				status = cudaDeviceSynchronize();
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
					goto out;
				}
				tv2 = clock64();
				TIMEVAL_RECORD(kgjoin,kern_outer_scan,tv1,tv2,smx_clock);
				if (kgjoin->kerror.errcode != StromError_Success)
					return;
				/* update run-time statistics */
				kgjoin->jscale[0].inner_nitems = kresults_src->nitems;
				kgjoin->jscale[0].total_nitems = kresults_src->nitems;
			}
			else
			{
				kgjoin->jscale[0].inner_nitems = 0;	/* no input rows */
				kgjoin->jscale[0].total_nitems = 0;	/* update later */
			}
		}
		/* make the kresults_dst buffer empty */
		memset(kresults_dst, 0, offsetof(kern_resultbuf, results[0]));
		kresults_dst->nrels = depth + 1;
		kresults_dst->nrooms = kresults_max_items / (depth + 1);

		/* inner partition window in this depth */
		window_base = kgjoin->jscale[depth].window_base;
		window_size = kgjoin->jscale[depth].window_size;

		/* Launch:
		 * KERNEL_FUNCTION_MAXTHREADS(void)
		 * gpujoin_exec_nestloop(kern_gpujoin *kgjoin,
		 *                       kern_data_store *kds,
		 *                       kern_multirels *kmrels,
		 *                       kern_resultbuf *kresults_src,
		 *                       kern_resultbuf *kresults_dst,
		 *                       cl_bool *outer_join_map,
		 *                       cl_int depth,
		 *                       cl_int cuda_index,
		 *                       cl_uint window_base,
		 *                       cl_uint window_size)
		 */
		if (kmrels->chunks[depth-1].is_nestloop)
		{
			if (kresults_src->nitems > 0)
			{
				cl_uint		shmem_size;

				tv1 = clock64();
				kern_join_args = (kern_join_args_t *)
					cudaGetParameterBuffer(sizeof(void *),
										   sizeof(kern_join_args_t));
				if (!kern_join_args)
				{
					STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
					goto out;
				}
				SETUP_KERN_JOIN_ARGS(kern_join_args);

				status = pgstrom_optimal_workgroup_size(
					&grid_sz,
					&block_sz,
					(const void *)gpujoin_exec_nestloop,
					(size_t)kresults_src->nitems * (size_t)window_size,
					sizeof(kern_errorbuf));
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
					goto out;
				}
				status = cudaLaunchDevice((void *)gpujoin_exec_nestloop,
										  kern_join_args,
										  grid_sz, block_sz,
										  sizeof(cl_uint) * block_sz.x,
										  NULL);
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
					goto out;
				}

				status = cudaDeviceSynchronize();
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
					goto out;
				}
				tv2 = clock64();
				TIMEVAL_RECORD(kgjoin,kern_exec_nestloop,tv1,tv2,smx_clock);

				if (kgjoin->kerror.errcode == StromError_DataStoreNoSpace)
				{
					nsplits = kresults_dst->nitems / kresults_dst->nrooms + 1;
					victim = gpujoin_resize_window(kgjoin,
												   kmrels,
												   kds_src,
												   kresults_dst,
												   nsplits,
												   smx_count,
												   smx_clock);
					if (victim < 0)
						return;
					else if (victim < depth)
					{
						kgjoin->num_major_retry++;
						goto retry_major;
					}
					kgjoin->num_minor_retry++;
					goto retry_minor;
				}
				else if (kgjoin->kerror.errcode != StromError_Success)
					return;
				/* update run-time statistics */
				kgjoin->jscale[depth].inner_nitems = kresults_dst->nitems;
				kgjoin->jscale[depth].total_nitems = kresults_dst->nitems;
			}
			else
			{
				/* in case when no input rows. INNER JOIN produce no rows */
				kgjoin->jscale[depth].inner_nitems = 0;
				kgjoin->jscale[depth].total_nitems = 0;
			}

			/* Launch:
			 * KERNEL_FUNCTION(void)
			 * gpujoin_outer_nestloop(kern_gpujoin *kgjoin,
			 *                        kern_data_store *kds,
			 *                        kern_multirels *kmrels,
			 *                        kern_resultbuf *kresults_src,
			 *                        kern_resultbuf *kresults_dst,
			 *                        cl_bool *outer_join_map,
			 *                        cl_int depth,
			 *                        cl_int cuda_index,
			 *                        cl_uint window_base,
			 *                        cl_uint window_size)
			 *
			 * NOTE: Host-side has to co-locate the outer join map
			 * into this device, prior to the kernel launch.
			 */
			if (exec_right_outer)
			{
				tv1 = clock64();
				kern_join_args = (kern_join_args_t *)
					cudaGetParameterBuffer(sizeof(void *),
										   sizeof(kern_join_args_t));
				if (!kern_join_args)
				{
					STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
					goto out;
				}
				SETUP_KERN_JOIN_ARGS(kern_join_args);

				status = pgstrom_optimal_workgroup_size(&grid_sz,
														&block_sz,
														(const void *)
														gpujoin_outer_nestloop,
														window_size,
														sizeof(kern_errorbuf));
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
					goto out;
				}

				status = cudaLaunchDevice((void *)gpujoin_exec_nestloop,
										  kern_join_args,
										  grid_sz, block_sz,
										  sizeof(kern_errorbuf) * block_sz.x,
										  NULL);
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
					goto out;
				}

				status = cudaDeviceSynchronize();
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
					goto out;
				}
				tv2 = clock64();
				TIMEVAL_RECORD(kgjoin,kern_outer_nestloop,tv1,tv2,smx_clock);

				if (kgjoin->kerror.errcode == StromError_DataStoreNoSpace)
				{
					nsplits = kresults_dst->nitems / kresults_dst->nrooms + 1;
					victim = gpujoin_resize_window(kgjoin,
												   kmrels,
												   kds_src,
												   kresults_dst,
												   nsplits,
												   smx_count,
												   smx_clock);
					if (victim < 0)
						return;
					else if (victim < depth)
					{
						kgjoin->num_major_retry++;
						goto retry_major;
					}
					kgjoin->num_minor_retry++;
					goto retry_minor;
				}
				else if (kgjoin->kerror.errcode != StromError_Success)
					return;
				/* update run-time statistics */
				kgjoin->jscale[depth].total_nitems = kresults_dst->nitems;
			}
		}
		else
		{
			/* Launch:
			 * KERNEL_FUNCTION(void)
			 * gpujoin_exec_hashjoin(kern_gpujoin *kgjoin,
			 *                       kern_data_store *kds,
			 *                       kern_multirels *kmrels,
			 *                       kern_resultbuf *kresults_src,
			 *                       kern_resultbuf *kresults_dst,
			 *                       cl_bool *outer_join_map,
			 *                       cl_int depth,
			 *                       cl_int cuda_index,
			 *                       cl_uint window_base,
			 *                       cl_uint window_size);
			 */
			if (kresults_src->nitems > 0)
			{
				tv1 = clock64();
				kern_join_args = (kern_join_args_t *)
					cudaGetParameterBuffer(sizeof(void *),
										   sizeof(kern_join_args_t));
				if (!kern_join_args)
				{
					STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
					goto out;
				}
				SETUP_KERN_JOIN_ARGS(kern_join_args);

				status = pgstrom_optimal_workgroup_size(&grid_sz,
														&block_sz,
														(const void *)
														gpujoin_exec_hashjoin,
														kresults_src->nitems,
														sizeof(kern_errorbuf));
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
					goto out;
				}

				status = cudaLaunchDevice((void *)gpujoin_exec_hashjoin,
										  kern_join_args,
										  grid_sz, block_sz,
										  sizeof(kern_errorbuf) * block_sz.x,
										  NULL);
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
					goto out;
				}

				status = cudaDeviceSynchronize();
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
					goto out;
				}
				tv2 = clock64();
				TIMEVAL_RECORD(kgjoin,kern_exec_hashjoin,tv1,tv2,smx_clock);

				if (kgjoin->kerror.errcode == StromError_DataStoreNoSpace)
				{
					nsplits = kresults_dst->nitems / kresults_dst->nrooms + 1;
					victim = gpujoin_resize_window(kgjoin,
												   kmrels,
												   kds_src,
												   kresults_dst,
												   nsplits,
												   smx_count,
												   smx_clock);
					if (victim < 0)
						return;
					else if (victim < depth)
					{
						kgjoin->num_major_retry++;
						goto retry_major;
					}
					kgjoin->num_minor_retry++;
					goto retry_minor;
				}
				else if (kgjoin->kerror.errcode != StromError_Success)
					return;
				/* update run-time statistics */
				kgjoin->jscale[depth].inner_nitems = kresults_dst->nitems;
				kgjoin->jscale[depth].total_nitems = kresults_dst->nitems;
			}
			else
			{
				/* no input rows, then no output rows */
				kgjoin->jscale[depth].inner_nitems = 0;
				kgjoin->jscale[depth].total_nitems = 0;
			}

			/* Launch:
			 * KERNEL_FUNCTION(void)
			 * gpujoin_outer_hashjoin(kern_gpujoin *kgjoin,
			 *                        kern_data_store *kds,
			 *                        kern_multirels *kmrels,
			 *                        cl_bool *outer_join_map,
			 *                        cl_int depth,
			 *                        cl_int cuda_index);
			 */
			if (exec_right_outer)
			{
				tv1 = clock64();
				kern_join_args = (kern_join_args_t *)
					cudaGetParameterBuffer(sizeof(void *),
										   sizeof(kern_join_args_t));
				if (!kern_join_args)
				{
					STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
					goto out;
				}
				SETUP_KERN_JOIN_ARGS(kern_join_args);

				status = pgstrom_optimal_workgroup_size(&grid_sz,
														&block_sz,
														(const void *)
														gpujoin_outer_hashjoin,
														window_size,
														sizeof(kern_errorbuf));
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
					goto out;
				}

				status = cudaLaunchDevice((void *)gpujoin_outer_hashjoin,
										  kern_join_args,
										  grid_sz, block_sz,
										  sizeof(kern_errorbuf) * block_sz.x,
										  NULL);
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
					goto out;
				}

				status = cudaDeviceSynchronize();
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
					goto out;
				}
				tv2 = clock64();
				TIMEVAL_RECORD(kgjoin,kern_outer_hashjoin,tv1,tv2,smx_clock);

				if (kgjoin->kerror.errcode == StromError_DataStoreNoSpace)
				{
					nsplits = kresults_dst->nitems / kresults_dst->nrooms + 1;
					victim = gpujoin_resize_window(kgjoin,
												   kmrels,
												   kds_src,
												   kresults_dst,
												   nsplits,
												   smx_count,
												   smx_clock);
					if (victim < 0)
						return;
					else if (victim < depth)
					{
						kgjoin->num_major_retry++;
						goto retry_major;
					}
					kgjoin->num_minor_retry++;
					goto retry_minor;
				}
				else if (kgjoin->kerror.errcode != StromError_Success)
					return;
				/* update run-time statistics */
				kgjoin->jscale[depth].total_nitems = kresults_dst->nitems;
			}
		}

		/*
		 * Swap result buffer
		 */
		kresults_tmp = kresults_src;
		kresults_src = kresults_dst;
		kresults_dst = kresults_tmp;
	}

	/*
	 * Launch:
	 * KERNEL_FUNCTION(void)
	 * gpujoin_projection_(row|slot)(kern_gpujoin *kgjoin,
	 *                               kern_multirels *kmrels,
	 *                               kern_data_store *kds_src,
	 *                               kern_data_store *kds_dst,
	 *                               kern_resultbuf *kresults)
	 */
	tv1 = clock64();
	if (kds_dst->format == KDS_FORMAT_ROW)
		kernel_projection = (const void *)gpujoin_projection_row;
	else
		kernel_projection = (const void *)gpujoin_projection_slot;

	/*
	 * Setup kds_dst according to the final kern_resultbuf
	 */
	if (kresults_src->nitems >= kds_dst->nrooms)
	{
		nsplits = kresults_src->nitems / kds_dst->nrooms + 1;
		victim = gpujoin_resize_window(kgjoin,
									   kmrels,
									   kds_src,
									   kresults_src,
									   nsplits,
									   smx_count,
									   smx_clock);
		if (victim < 0)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			goto out;
		}
		kgjoin->num_major_retry++;
		goto retry_major;
	}
	kds_dst->nitems = kresults_src->nitems;
	/* No need to launch projection kernel? */
	if (kds_dst->nitems == 0)
		goto out;

	kern_args = (void **)cudaGetParameterBuffer(sizeof(void *),
												sizeof(void *) * 5);
	if (!kern_args)
	{
		STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
		goto out;
	}
	kern_args[0] = kgjoin;
	kern_args[1] = kmrels;
	kern_args[2] = kds_src;
	kern_args[3] = kds_dst;
	kern_args[4] = kresults_src;

	status = pgstrom_optimal_workgroup_size(&grid_sz,
											&block_sz,
											kernel_projection,
											kresults_src->nitems,
											sizeof(kern_errorbuf));
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
		goto out;
	}

	status = cudaLaunchDevice((void *)kernel_projection,
							  kern_args, grid_sz, block_sz,
							  sizeof(kern_errorbuf) * block_sz.x,
							  NULL);
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
		goto out;
	}

	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
		goto out;
	}
	tv2 = clock64();
	TIMEVAL_RECORD(kgjoin,kern_projection,tv1,tv2,smx_clock);

	if (kgjoin->kerror.errcode == StromError_DataStoreNoSpace)
	{
		cl_uint		ncols = kds_dst->ncols;
		cl_uint		nitems_to_fit;
		cl_uint		width_avg;

		if (kds_dst->nitems == 0)
			return;		/* should never happen */
		if (kds_dst->format == KDS_FORMAT_SLOT)
		{
			width_avg = (LONGALIGN((sizeof(Datum) +
									sizeof(char)) * ncols) +
						 MAXALIGN(kds_dst->usage /
								  kds_dst->nitems + 1));
			nitems_to_fit = (kds_dst->length -
							 STROMALIGN(offsetof(kern_data_store,
												 colmeta[ncols]))) / width_avg;
		}
		else
		{
			width_avg = MAXALIGN(kds_dst->usage /
								 kds_dst->nitems + 1) + sizeof(cl_uint);
			nitems_to_fit = (kds_dst->length -
							 STROMALIGN(offsetof(kern_data_store,
												 colmeta[ncols]))) / width_avg;
		}
		if (gpujoin_resize_window(kgjoin,
								  kmrels,
								  kds_src,
								  kresults_src,
								  kds_dst->nitems / nitems_to_fit + 1,
								  smx_count,
								  smx_clock) < 0)
			return;
		kgjoin->num_major_retry++;
		goto retry_major;
	}

	/*
	 * TODO: If kds_dst still has space, do major_retry if we can shift
	 * inner_base
	 */
out:
	kern_writeback_error_status(&kgjoin->kerror, kcxt.e);
}

#endif	/* __CUDACC__ */
#endif	/* CUDA_GPUJOIN_H */
