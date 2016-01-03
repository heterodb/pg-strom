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
	cl_uint			ndevs;			/* number of devices installed */
	struct
	{
		cl_uint		chunk_offset;	/* offset to KDS or Hash */
		cl_uint		ojmap_offset;	/* offset to Left-Outer Map, if any */
		cl_bool		left_outer;		/* true, if JOIN_LEFT or JOIN_FULL */
		cl_bool		right_outer;	/* true, if JOIN_RIGHT or JOIN_FULL */
		cl_char		__padding__[2];
	} chunks[FLEXIBLE_ARRAY_MEMBER];
} kern_multirels;

#define KERN_MULTIRELS_INNER_KDS(kmrels, depth)	\
	((kern_data_store *)						\
	 ((char *)(kmrels) + (kmrels)->chunks[(depth)-1].chunk_offset))

#define KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth, nitems,		\
									  cuda_index, outer_join_map)	\
	((cl_bool *)													\
	 ((kmrels)->chunks[(depth)-1].right_outer						\
	  ? ((char *)(outer_join_map) +									\
		 STROMALIGN(sizeof(cl_bool) * (nitems)) *					\
		 (cuda_index))												\
	  : NULL))

#define KERN_MULTIRELS_LEFT_OUTER_JOIN(kmrels, depth)	\
	((kmrels)->chunks[(depth)-1].left_outer)

#define KERN_MULTIRELS_RIGHT_OUTER_JOIN(kmrels, depth)	\
	((kmrels)->chunks[(depth)-1].right_outer)

/*
 * kern_gpujoin - control object of GpuJoin
 */
#define GPUJOIN_MAX_DEPTH	24

typedef struct
{
	/* offset to the primary kern_resultbuf */
	size_t			kresults_1_offset;
	/* offset to the secondary kern_resultbuf */
	size_t			kresults_2_offset;
	/* max allocatable number of kern_resultbuf items */
	cl_uint			kresults_max_space;
	/* number of inner relations */
	cl_uint			num_rels;
	/* least depth in this call chain */
	cl_uint			start_depth;
	/* OUT: number of result rows actually generated for each depth */
	cl_uint			result_nitems[GPUJOIN_MAX_DEPTH + 1];
	/* OUT: maximum valid depth in the above result */
	cl_uint			result_valid_until;
	/* error status to be backed (OUT) */
	kern_errorbuf	kerror;
	/* run-time parameter buffer */
	kern_parambuf	kparams;
} kern_gpujoin;

#define KERN_GPUJOIN_PARAMBUF(kgjoin)			\
	((kern_parambuf *)(&(kgjoin)->kparams))
#define KERN_GPUJOIN_PARAMBUF_LENGTH(kgjoin)	\
	STROMALIGN(KERN_GPUJOIN_PARAMBUF(kgjoin)->length)
#define KERN_GPUJOIN_HEAD_LENGTH(kgjoin)		\
	((kgjoin)->kresults_1_offset +				\
	 STROMALIGN(offsetof(kern_resultbuf, results[0])))
#define KERN_GPUJOIN_IN_RESULTS(kgjoin,depth)			\
	((kern_resultbuf *)((char *)(kgjoin) +				\
						(((depth) & 0x01)				\
						 ? (kgjoin)->kresults_1_offset	\
						 : (kgjoin)->kresults_2_offset)))
#define KERN_GPUJOIN_OUT_RESULTS(kgjoin,depth)			\
	((kern_resultbuf *)((char *)(kgjoin) +				\
						(((depth) & 0x01)				\
						 ? (kgjoin)->kresults_2_offset  \
						 : (kgjoin)->kresults_1_offset)))



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
 * gpujoin_outer_quals
 *
 * Evaluation of outer-relation's qualifier, if any. Elsewhere, it always
 * returns true.
 */
STATIC_FUNCTION(cl_bool)
gpujoin_outer_quals(kern_context *kcxt,
					kern_data_store *kds,
					size_t kds_index);
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
				   HeapTupleHeaderData *inner_htup);

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
				   cl_uint *x_buffer);

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

KERNEL_FUNCTION(void)
gpujoin_preparation(kern_gpujoin *kgjoin,
					kern_data_store *kds,
					kern_multirels *kmrels,
					cl_int depth)
{
	kern_parambuf  *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_resultbuf *kresults_in;
	kern_resultbuf *kresults_out;
	kern_context	kcxt;
	cl_uint			start_depth = kgjoin->start_depth;

	/* sanity check */
	assert(depth > 0 && depth <= kgjoin->num_rels);
	assert(depth >= start_depth && start_depth <= kgjoin->num_rels);
	assert(!kmrels || start_depth == 1);
	assert(kgjoin->kresults_1_offset > 0);
	assert(kgjoin->kresults_2_offset > 0);
	kresults_in = KERN_GPUJOIN_IN_RESULTS(kgjoin, depth);
	kresults_out = KERN_GPUJOIN_OUT_RESULTS(kgjoin, depth);

	INIT_KERNEL_CONTEXT(&kcxt, gpujoin_preparation, kparams);
	if (depth > start_depth &&
		kresults_in->kerror.errcode != StromError_Success)
		kcxt.e = kresults_in->kerror;

	/*
	 * In case of start depth, input result buffer is not initialized
	 * yet. So, we need to put initial values first of all.
	 */
	if (depth == start_depth)
	{
		if (kds == NULL)
		{
			/*
			 * Special case if RIGHT/FULL OUTER JOIN. We have no input rows
			 * on depth == start_depth, so results_in shall be simply cleared.
			 */
			if (get_global_id() == 0)
			{
				kresults_in->nrels = depth;
				kresults_in->nrooms = 0;
				kresults_in->nitems = 0;
				memset(&kresults_in->kerror, 0,
					   sizeof(kern_errorbuf));
				memset(kgjoin->result_nitems, 0,
					   sizeof(kgjoin->result_nitems));
			}
		}
		else
		{
			cl_uint		kds_index = get_global_id();
			cl_uint		count;
			cl_uint		offset;
			cl_bool		is_matched;
			__shared__ cl_int	base;

			assert(depth == 1);
			assert(kresults_in->nrels == 1);
			assert(kresults_in->nrooms == kgjoin->kresults_max_space);
			/*
			 * Check qualifier of outer scan that was pulled-up (if any).
			 * then, it allocates result buffer on kresults_in and put
			 * get_global_id() if it match.
			 */
			if (kds_index < kds->nitems)
				is_matched = gpujoin_outer_quals(&kcxt, kds, kds_index);
			else
				is_matched = false;

			/* expand kresults_in->nitems */
			offset = arithmetic_stairlike_add(is_matched ? 1 : 0, &count);
			if (get_local_id() == 0)
			{
				if (count > 0)
					base = atomicAdd(&kresults_in->nitems, count);
				else
					base = 0;
				atomicMax(&kgjoin->result_nitems[0], base + count);
			}
			__syncthreads();

			if (base + count >= kgjoin->kresults_max_space)
				STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			else if (is_matched)
			{
				HeapTupleHeaderData	   *htup
					= kern_get_tuple_row(kds, kds_index);
				kresults_in->results[base + offset]
					= (size_t)htup - (size_t)kds;
			}
		}
	}
	/*
	 * Initialize output kresults buffer
	 */
	if (get_global_id() == 0)
	{
		assert(kresults_in->nrels == depth);
		kresults_out->nrels = depth + 1;
		kresults_out->nrooms = kgjoin->kresults_max_space / (depth + 1);
		kresults_out->nitems = 0;
		memset(&kresults_out->kerror, 0, sizeof(kern_errorbuf));
		/*
		 * Update statistics - unless we have no errors, nitems in the
		 * previous depth is reliable.
		 *
		 * NOTE: The reason why result_valid_until will have 'depth', not
		 * 'depth - 1', is that we shall put exact value on result_nitems[]
		 * even if next depth will raise NoSpaceDataStore error. So, host
		 * code can know the number of rows to be generated in this depth.
		 */
		if (kresults_in->kerror.errcode == StromError_Success)
		{
			kgjoin->result_nitems[depth - 1] = kresults_in->nitems;
			kgjoin->result_valid_until = depth;
		}
	}
	kern_writeback_error_status(&kresults_in->kerror, kcxt.e);
}

KERNEL_FUNCTION_MAXTHREADS(void)
gpujoin_exec_nestloop(kern_gpujoin *kgjoin,
					  kern_data_store *kds,
					  kern_multirels *kmrels,
					  cl_int depth,
					  cl_int cuda_index,
					  cl_bool *outer_join_map,
					  cl_uint inner_base,
					  cl_uint inner_size)
{
	kern_parambuf  *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_resultbuf *kresults_in = KERN_GPUJOIN_IN_RESULTS(kgjoin, depth);
	kern_resultbuf *kresults_out = KERN_GPUJOIN_OUT_RESULTS(kgjoin, depth);
	kern_context	kcxt;
	kern_data_store *kds_in;
	cl_bool		   *lo_map;
	size_t			nvalids;
	cl_uint			y_index;
	cl_uint			y_offset;
	cl_uint			y_limit;
	cl_uint			x_index;
	cl_uint			x_limit;

	/*
	 * immediate bailout if previous stage already have error status
	 */
	kcxt.e = kresults_in->kerror;
	if (kcxt.e.errcode != StromError_Success)
		goto out;
	INIT_KERNEL_CONTEXT(&kcxt,gpujoin_exec_nestloop,kparams);

	/* sanity checks */
	assert(depth > 0 && depth <= kgjoin->num_rels);
	assert(kresults_out->nrels == depth + 1);
	assert(kresults_in->nrels == depth);

	/*
	 * NOTE: size of Y-axis deterministric on the time of kernel launch.
	 * host-side guarantees get_global_ysize() is larger then kds->nitems
	 * of the depth. On the other hands, nobody can know correct size of
	 * X-axis unless gpujoin_exec_nestloop() of the previous stage.
	 * So, we ensure all the outer items are picked up by the loop below.
	 */
	kds_in = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	assert(kds_in != NULL);

	/* will be valid, if LEFT OUTER JOIN */
	lo_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth, kds_in->nitems,
										   cuda_index, outer_join_map);

	nvalids = min(kresults_in->nitems, kresults_in->nrooms);
	x_limit = ((nvalids + get_local_xsize() - 1) /
			   get_local_xsize()) * get_local_xsize();
	y_limit = min(inner_base + inner_size, kds_in->nitems);
	y_index = inner_base + get_global_yid();
	for (x_index = get_global_xid();
		 x_index < x_limit;
		 x_index += get_global_xsize())
	{
		HeapTupleHeaderData *y_htup;
		cl_uint		   *x_buffer;
		cl_uint		   *r_buffer;
		cl_bool			is_matched;
		cl_uint			offset;
		cl_uint			count;
		__shared__ cl_uint base;

		/* outer input */
		if (x_index < nvalids)
			x_buffer = KERN_GET_RESULT(kresults_in, x_index);
		else
			x_buffer = NULL;

		/* inner input */
		if (y_index < y_limit)
			y_htup = kern_get_tuple_row(kds_in, y_index);
		else
			y_htup = NULL;

		/* does it satisfies join condition? */
		is_matched = gpujoin_join_quals(&kcxt,
										kds,
										kmrels,
										depth,
										x_buffer,
										y_htup);
		if (is_matched)
		{
			y_offset = (size_t)y_htup - (size_t)kds_in;
			if (lo_map && !lo_map[y_index])
				lo_map[y_index] = true;
		}
		else
			y_offset = UINT_MAX;

		/*
		 * Expand kresults_out->nitems, and put values
		 */
		offset = arithmetic_stairlike_add(is_matched ? 1 : 0, &count);
		if (get_local_xid() == 0 && get_local_yid() == 0)
		{
			if (count > 0)
				base = atomicAdd(&kresults_out->nitems, count);
			else
				base = 0;
			atomicMax(&kgjoin->result_nitems[depth], base + count);
		}
		__syncthreads();

		/* still have space to store? */
		if (base + count >= kresults_out->nrooms)
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		else if (is_matched)
		{
			assert(x_buffer != NULL && y_htup != NULL);
			r_buffer = KERN_GET_RESULT(kresults_out, base + offset);
			memcpy(r_buffer, x_buffer, sizeof(cl_int) * depth);
			r_buffer[depth] = y_offset;
		}
	}
out:
	kern_writeback_error_status(&kresults_out->kerror, kcxt.e);
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
					  cl_int depth,
					  cl_int cuda_index,
					  cl_bool *outer_join_map,
					  cl_uint inner_base,
					  cl_uint inner_size)
{
	kern_parambuf	   *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_resultbuf	   *kresults_in = KERN_GPUJOIN_IN_RESULTS(kgjoin, depth);
	kern_resultbuf	   *kresults_out = KERN_GPUJOIN_OUT_RESULTS(kgjoin, depth);
	kern_data_store	   *kds_hash = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	kern_context		kcxt;
	cl_bool			   *lo_map;
	size_t				nvalids;
	cl_int				crc_index;
	cl_uint				x_limit;
	cl_uint				x_index;
	__shared__ cl_uint	base;
	__shared__ cl_uint	pg_crc32_table[256];

	/*
	 * immediate bailout if previous stage already have error status
	 */
	kcxt.e = kresults_in->kerror;
	if (kcxt.e.errcode != StromError_Success)
		goto out;
	INIT_KERNEL_CONTEXT(&kcxt,gpujoin_exec_hashjoin,kparams);

	/* sanity checks */
	assert(get_global_ysize() == 1);
	assert(depth > 0 && depth <= kgjoin->num_rels);
	assert(kresults_out->nrels == depth + 1);
	assert(kresults_in->nrels == depth);

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

	/*
     * NOTE: Nobody can know correct size of outer relation because it
	 * is determined in the previous step, so all we can do is to launch
	 * a particular number of threads according to statistics.
	 * We have to take the giant loop below to ensure all the outer items
	 * getting evaluated.
	 */
	nvalids = min(kresults_in->nitems, kresults_in->nrooms);
	x_limit = ((nvalids + get_local_xsize() - 1) /
			   get_local_xsize()) * get_local_xsize();

	/* will be valid, if LEFT OUTER JOIN */
	lo_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth, kds_hash->nitems,
										   cuda_index, outer_join_map);
	for (x_index = get_global_xid();
		 x_index < x_limit;
		 x_index += get_global_xsize())
	{
		kern_hashitem  *khitem = NULL;
		cl_uint			hash_value;
        cl_uint		   *x_buffer = NULL;
        cl_uint		   *r_buffer;
		cl_uint			offset;
		cl_uint			count;
		cl_bool			is_matched;
		cl_bool			needs_outer_row = false;

		/*
		 * Calculation of hash-value of the outer relations.
		 */
		if (x_index < nvalids)
		{
			x_buffer = KERN_GET_RESULT(kresults_in, x_index);
			assert(((size_t)x_buffer[0] & (sizeof(cl_ulong) - 1)) == 0);
			hash_value = gpujoin_hash_value(&kcxt,
											pg_crc32_table,
											kds,
											kmrels,
											depth,
											x_buffer);
			if (hash_value >= kds_hash->hash_min &&
				hash_value <= kds_hash->hash_max)
			{
				khitem = KERN_HASH_FIRST_ITEM(kds_hash, hash_value);
				needs_outer_row = true;
			}
		}

		/*
		 * walks on the hash entries
		 */
		do {
			HeapTupleHeaderData *h_htup;

			if (khitem && (khitem->hash  == hash_value &&
						   khitem->rowid >= inner_base &&
						   khitem->rowid <  inner_base + inner_size))
				h_htup = &khitem->htup;
			else
				h_htup = NULL;

			is_matched = gpujoin_join_quals(&kcxt,
											kds,
											kmrels,
											depth,
											x_buffer,	/* valid if in range */
											h_htup);	/* valid if in range */
			if (is_matched)
			{
				assert(khitem->rowid < kds_hash->nitems);
				if (lo_map && !lo_map[khitem->rowid])
					lo_map[khitem->rowid] = true;
				needs_outer_row = false;
			}

			/*
			 * Expand kresults_out->nitems
			 */
			offset = arithmetic_stairlike_add(is_matched ? 1 : 0, &count);
			if (get_local_xid() == 0)
			{
				if (count > 0)
					base = atomicAdd(&kresults_out->nitems, count);
				else
					base = 0;
				atomicMax(&kgjoin->result_nitems[depth], base + count);
			}
			__syncthreads();

			/* kresults_out still have enough space? */
			if (base + count >= kresults_out->nrooms)
				STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			else if (is_matched)
			{
				r_buffer = KERN_GET_RESULT(kresults_out, base + offset);
				memcpy(r_buffer, x_buffer, sizeof(cl_int) * depth);
				r_buffer[depth] = (size_t)&khitem->htup - (size_t)kds_hash;
			}

			/*
			 * Fetch next hash entry, then checks whether all the local
			 * threads still have valid hash-entry or not.
			 * (NOTE: this routine contains reduction operation)
			 */
			khitem = KERN_HASH_NEXT_ITEM(kds_hash, khitem);
			arithmetic_stairlike_add(khitem != NULL ? 1 : 0, &count);
		} while (count > 0);

		/*
		 * If no inner rows were matched on LEFT OUTER JOIN case, we fill
		 * up the inner-side of result tuple with NULL.
		 */
		if (KERN_MULTIRELS_LEFT_OUTER_JOIN(kmrels, depth))
		{
			offset = arithmetic_stairlike_add(needs_outer_row ? 1 : 0,
											  &count);
			if (get_local_xid() == 0)
			{
				if (count > 0)
					base = atomicAdd(&kresults_out->nitems, count);
				else
					base = 0;
				atomicMax(&kgjoin->result_nitems[depth], base + count);
			}
			__syncthreads();

			/* kresults_out still have enough space? */
			if (base + count >= kresults_out->nrooms)
				STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			else if (needs_outer_row)
			{
				assert(x_buffer != NULL);
				r_buffer = KERN_GET_RESULT(kresults_out, base + offset);
				memcpy(r_buffer, x_buffer, sizeof(cl_int) * depth);
				r_buffer[depth] = 0;	/* inner NULL */
			}
		}
	}
out:
	kern_writeback_error_status(&kresults_out->kerror, kcxt.e);
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
					   cl_int depth,
					   cl_int cuda_index,
					   cl_bool *outer_join_map,
					   cl_uint inner_base,
					   cl_uint inner_size)
{
	kern_parambuf	   *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_resultbuf	   *kresults_out = KERN_GPUJOIN_OUT_RESULTS(kgjoin, depth);
	kern_data_store	   *kds_in = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	kern_context		kcxt;
	cl_bool			   *lo_map;
	cl_bool				needs_outer_row;
	size_t				kds_index = get_global_id() + inner_base;
	cl_uint				count;
	cl_uint				offset;
	cl_uint			   *r_buffer;
	cl_int				i, ndevs = kmrels->ndevs;
	__shared__ cl_uint	base;

	/*
	 * immediate bailout if previous stage already have error status
	 */
	kcxt.e = kresults_out->kerror;
	if (kcxt.e.errcode != StromError_Success)
		goto out;
	INIT_KERNEL_CONTEXT(&kcxt,gpujoin_outer_nestloop,kparams);

	/* sanity checks */
	assert(get_global_xsize() == 1);
	assert(depth > 0 && depth <= kgjoin->num_rels);
	assert(kresults_out->nrels == depth + 1);

	/*
	 * check whether the relevant inner tuple has any matched outer tuples,
	 * including the jobs by other devices.
	 */
	if (get_global_id() < inner_size)
	{
		cl_uint		nitems = kds_in->nitems;

		needs_outer_row = true;
		for (i=0; i < ndevs; i++)
		{
			lo_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth, nitems,
												   i, outer_join_map);
			assert(lo_map != NULL);
			if (lo_map[kds_index])		// offset by inner_base?
				needs_outer_row = false;
		}
	}
	else
		needs_outer_row = false;

	/*
	 * Count up number of inner tuples that were not matched with outer-
	 * relations. Then, we allocates slot in kresults_out for outer-join
	 * tuples.
	 */
	offset = arithmetic_stairlike_add(needs_outer_row ? 1 : 0, &count);
	if (get_local_id() == 0)
	{
		if (count > 0)
			base = atomicAdd(&kresults_out->nitems, count);
		else
			base = 0;
		atomicMax(&kgjoin->result_nitems[depth], base + count);
	}
	__syncthreads();

	/* In case when (base + num_unmatched) is larger than nrooms, it means
	 * we don't have enough space to write back nested-loop results.
	 * So, we have to tell the host-side to acquire larger kern_resultbuf.
	 */
	if (base + count >= kresults_out->nrooms)
		STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
	else if (needs_outer_row)
	{
		/*
		 * OK, we know which row should be materialized using left outer
		 * join manner, and result buffer was acquired. Let's put result
		 * for the next stage.
		 */
		HeapTupleHeaderData	   *htup = kern_get_tuple_row(kds_in, kds_index);
		r_buffer = KERN_GET_RESULT(kresults_out, base + offset);
		memset(r_buffer, 0, sizeof(cl_int) * depth);	/* NULL */
		r_buffer[depth] = (size_t)htup - (size_t)kds_in;
	}
out:
	kern_writeback_error_status(&kresults_out->kerror, kcxt.e);
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
					   cl_int depth,
					   cl_int cuda_index,
					   cl_bool *outer_join_map,
					   cl_uint inner_base,
					   cl_uint inner_size)
{
	kern_parambuf	   *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_resultbuf	   *kresults_in = KERN_GPUJOIN_IN_RESULTS(kgjoin, depth);
	kern_resultbuf	   *kresults_out = KERN_GPUJOIN_OUT_RESULTS(kgjoin, depth);
	kern_data_store	   *kds_hash = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	kern_hashitem	   *khitem;
	kern_context		kcxt;
	cl_bool			   *lo_map;
	cl_bool				needs_outer_row;
//	cl_uint				offset;
//	cl_uint				count;
	cl_int				i, ndevs = kmrels->ndevs;
	cl_uint			   *r_buffer;
//	__shared__ cl_uint	base;

	/*
	 * immediate bailout if previous stage already have error status
	 */
	kcxt.e = kresults_in->kerror;
	if (kcxt.e.errcode != StromError_Success)
		goto out;
	INIT_KERNEL_CONTEXT(&kcxt,gpujoin_outer_hashjoin,kparams);

	/* sanity checks */
	assert(get_global_ysize() == 1);
	assert(depth > 0 && depth <= kgjoin->num_rels);
	assert(ndevs > 0);
	assert(kresults_out->nrels == depth + 1);
	assert(inner_base + inner_size <= kds_hash->nitems);
	assert(inner_size > 0);

#if 1
	/*
	 * A workaround implementation based on global memory operation.
	 *
	 * NOTE: we had a trouble around the code at #else .. #endif
	 * It looks to me the problem comes from shared memory and inter-
	 * thread synchronization, however, not ensured 100% at this moment.
	 * So, I put alternative implementation without reduction operation
	 * on the shared memory, but takes atomic operation on global
	 * memory for each outer tuple. It is a concern.
	 */
	if (get_global_id() < kds_hash->nslots)
		khitem = KERN_HASH_FIRST_ITEM(kds_hash, get_global_id());
	else
		khitem = NULL;

	while (khitem != NULL)
	{
		assert((khitem->hash % kds_hash->nslots) == get_global_id());
		if (khitem->rowid >= inner_base &&
			khitem->rowid <  inner_base + inner_size)
		{
			cl_uint		nitems = kds_hash->nitems;

			assert(khitem->rowid < nitems);
			needs_outer_row = true;
			for (i=0; i < ndevs; i++)
			{
				lo_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth, nitems,
													   i, outer_join_map);
				assert(lo_map != NULL);
				assert(khitem->rowid < nitems);
				if (lo_map[khitem->rowid])
					needs_outer_row = false;
			}

			if (needs_outer_row)
			{
				cl_uint		index;

				index = atomicAdd(&kresults_out->nitems, 1);
				atomicMax(&kgjoin->result_nitems[depth], index + 1);

				if (index < kresults_out->nrooms)
				{
					r_buffer = KERN_GET_RESULT(kresults_out, index);
					memset(r_buffer, 0, sizeof(cl_int) * depth);    /* NULL */
					r_buffer[depth] = (size_t)&khitem->htup - (size_t)kds_hash;
					assert((size_t)&khitem->htup - (size_t)kds_hash > 0UL);
				}
				else
					STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			}
		}
		khitem = KERN_HASH_NEXT_ITEM(kds_hash, khitem);
	}
#else
	/*
	 * Fetch a hash-entry from each hash-slot
	 */
	if (get_global_id() < kds_hash->nslots)
		khitem = KERN_HASH_FIRST_ITEM(kds_hash, get_global_id());
	else
		khitem = NULL;

	do {
		__syncthreads();
		assert(!khitem || (khitem->hash %
						   kds_hash->nslots) == get_global_id());

		if (khitem != NULL && (khitem->rowid >= inner_base &&
							   khitem->rowid <  inner_base + inner_size))
		{
			/*
			 * check whether the relevant inner tuple has any matched outer
			 * tuples, including the jobs by other devices.
			 */
			cl_uint		nitems = kds_hash->nitems;

			assert(khitem->rowid < nitems);
			needs_outer_row = true;
			for (i=0; i < ndevs; i++)
			{
				lo_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth, nitems,
													   i, outer_join_map);
				assert(lo_map != NULL);
				assert(khitem->rowid < nitems);
				if (lo_map[khitem->rowid])
					needs_outer_row = false;
			}
		}
		else
			needs_outer_row = false;

		/*
		 * Then, count up number of unmatched inner tuples
		 */
		offset = arithmetic_stairlike_add(needs_outer_row ? 1 : 0, &count);
		if (get_local_id() == 0)
		{
			if (count > 0)
				base = atomicAdd(&kresults_out->nitems, count);
			else
				base = 0;

			atomicMax(&kgjoin->result_nitems[depth], base + count);
		}
		__syncthreads();
		assert(count <= get_global_xsize());

		/* In case when (base + num_unmatched) is larger than nrooms, it means
		 * we don't have enough space to write back nested-loop results.
		 * So, we have to tell the host-side to acquire larger kern_resultbuf.
		 */
		if (base + count >= kresults_out->nrooms)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		}
		else if (needs_outer_row)
		{
			assert(khitem != NULL);
			/*
			 * OK, we know which row should be materialized using left outer
			 * join manner, and result buffer was acquired. Let's put result
			 * for the next stage.
			 */
			r_buffer = KERN_GET_RESULT(kresults_out, base + offset);
			memset(r_buffer, 0, sizeof(cl_int) * depth);	/* NULL */
			r_buffer[depth] = (size_t)&khitem->htup - (size_t)kds_hash;
			assert((size_t)&khitem->htup - (size_t)kds_hash > 0UL);
		}

		/*
		 * Walk on the hash-chain until all the local threads reaches to
		 * end of the hash-list
		 */
		khitem = KERN_HASH_NEXT_ITEM(kds_hash, khitem);
		arithmetic_stairlike_add(khitem != NULL ? 1 : 0, &count);
		//__syncthreads();
		assert(++loop < 100000);
	} while (count > 0);
	__syncthreads();
#endif
out:
	kern_writeback_error_status(&kresults_out->kerror, kcxt.e);
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
					   kern_data_store *kds_dst)
{
	kern_parambuf  *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_resultbuf *kresults = KERN_GPUJOIN_OUT_RESULTS(kgjoin,
														kgjoin->num_rels);
	kern_context	kcxt;
	size_t		res_index;
	size_t		res_limit;

	/* sanity checks */
	assert(kresults->nrels == kgjoin->num_rels + 1);
	assert(kds_src == NULL || kds_src->format == KDS_FORMAT_ROW);
	assert(kds_dst->format == KDS_FORMAT_ROW);

	/*
	 * Update nitems of kds_dst. note that get_global_id(0) is not always
	 * called earlier than other thread. So, we should not expect nitems
	 * of kds_dst is initialized.
	 */
	if (get_global_id() == 0)
	{
		kds_dst->nitems = kresults->nitems;
		kgjoin->result_nitems[kgjoin->num_rels] = kresults->nitems;
		if (kresults->kerror.errcode == StromError_Success)
			kgjoin->result_valid_until = kgjoin->num_rels;
	}

	/* Immediate bailout if previous stage raise an error status */
	kcxt.e = kresults->kerror;
	if (kcxt.e.errcode != StromError_Success)
		goto out;
	INIT_KERNEL_CONTEXT(&kcxt, gpujoin_projection_row, kparams);

	/* Case of overflow; it shall be retried or executed by CPU instead,
	 * so no projection is needed anyway. We quickly exit the kernel.
	 * No need to set an error code because kern_gpuhashjoin_main()
	 * should already set it.
	 */
	if (kresults->nitems > kresults->nrooms ||
		kresults->nitems > kds_dst->nrooms)
	{
		STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		goto out;
	}

	/* Do projection if thread is responsible */
	res_limit = ((kresults->nitems + get_local_size() - 1) /
				 get_local_size()) * get_local_size();
	for (res_index = get_global_id();
		 res_index < res_limit;
		 res_index += get_global_size())
	{
		Datum		tup_values[GPUJOIN_DEVICE_PROJECTION_NFIELDS];
		cl_bool		tup_isnull[GPUJOIN_DEVICE_PROJECTION_NFIELDS];
		cl_short	tup_depth[GPUJOIN_DEVICE_PROJECTION_NFIELDS];
#if GPUJOIN_DEVICE_PROJECTION_EXTRA_SIZE > 0
		cl_char		extra_buf[GPUJOIN_DEVICE_PROJECTION_EXTRA_SIZE]
					__attribute__ ((aligned(MAXIMUM_ALIGNOF)));
#endif
		cl_uint		extra_len;
		cl_uint		required;
		cl_uint		offset;
		cl_uint		count;
		__shared__ cl_uint base;

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
		if (res_index < kresults->nitems)
		{
			cl_uint	   *r_buffer = KERN_GET_RESULT(kresults, res_index);

			gpujoin_projection(&kcxt,
							   kds_src,
							   kmrels,
							   r_buffer,
							   kds_dst,
							   tup_values,
							   tup_isnull,
							   tup_depth,
#if GPUJOIN_DEVICE_PROJECTION_EXTRA_SIZE > 0
							   extra_buf,
#else
							   NULL,
#endif
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
			required = 0;

		/*
		 * Step.2 - increment the buffer usage of kds_dst
		 */
		offset = arithmetic_stairlike_add(required, &count);
		if (get_local_id() == 0)
		{
			if (count > 0)
				base = atomicAdd(&kds_dst->usage, count);
			else
				base = 0;
		}
		__syncthreads();

		if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
			STROMALIGN(sizeof(cl_uint) * kresults->nitems) +
			base + count > kds_dst->length)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			goto out;
		}

		/*
		 * Step.3 - write out the HeapTuple on the destination buffer
		 */
		if (required > 0)
		{
			cl_uint			pos = kds_dst->length - (base + offset + required);
			cl_uint		   *tup_pos = (cl_uint *)KERN_DATA_STORE_BODY(kds_dst);
			kern_tupitem   *tupitem = (kern_tupitem *)((char *)kds_dst + pos);

			tup_pos[res_index] = pos;
			form_kern_heaptuple(&kcxt, kds_dst, tupitem,
								tup_values, tup_isnull, NULL);
		}
	}
out:
	/* write-back execution status to host-side */
	kern_writeback_error_status(&kgjoin->kerror, kcxt.e);
}

KERNEL_FUNCTION(void)
gpujoin_projection_slot(kern_gpujoin *kgjoin,
						kern_multirels *kmrels,
						kern_data_store *kds_src,
						kern_data_store *kds_dst)
{
	kern_parambuf  *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_resultbuf *kresults = KERN_GPUJOIN_OUT_RESULTS(kgjoin,
														kgjoin->num_rels);
	kern_context	kcxt;
	cl_uint		   *r_buffer;
	Datum		   *tup_values;
	cl_bool		   *tup_isnull;
	cl_short		tup_depth[GPUJOIN_DEVICE_PROJECTION_NFIELDS];
#if GPUJOIN_DEVICE_PROJECTION_EXTRA_SIZE > 0
	cl_char			extra_buf[GPUJOIN_DEVICE_PROJECTION_EXTRA_SIZE]
					__attribute__ ((aligned(MAXIMUM_ALIGNOF)));
#endif
	cl_uint			extra_len;
	cl_uint			offset	__attribute__ ((unused));
	cl_uint			count	__attribute__ ((unused));
	__shared__ cl_uint base	__attribute__ ((unused));
	size_t			res_limit;
	size_t			res_index;

	/* sanity checks */
	assert(kresults->nrels == kgjoin->num_rels + 1);
	assert(kds_src == NULL || kds_src->format == KDS_FORMAT_ROW);
	assert(kds_dst->format == KDS_FORMAT_SLOT);

	/*
	 * Update nitems of kds_dst. note that get_global_id(0) is not always
	 * called earlier than other thread. So, we should not expect nitems
	 * of kds_dst is initialized.
	 */
	if (get_global_id() == 0)
	{
		kds_dst->nitems = kresults->nitems;
		kgjoin->result_nitems[kgjoin->num_rels] = kresults->nitems;
		if (kresults->kerror.errcode == StromError_Success)
			kgjoin->result_valid_until = kgjoin->num_rels;
	}

	/* Immediate bailout if previous stage raise an error status */
	kcxt.e = kresults->kerror;
	if (kcxt.e.errcode != StromError_Success)
		goto out;
	INIT_KERNEL_CONTEXT(&kcxt, gpujoin_projection_slot, kparams);

	/*
	 * Case of overflow; it shall be retried or executed by CPU instead,
	 * so no projection is needed anyway. We quickly exit the kernel.
	 * No need to set an error code because kern_gpuhashjoin_main()
	 * should already set it.
	 */
	if (kresults->nitems > kresults->nrooms ||
		kresults->nitems > kds_dst->nrooms)
	{
		STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		goto out;
	}

	/* Do projection if thread is responsible */
	res_limit = ((kresults->nitems + get_local_size() - 1) /
				 get_local_size()) * get_local_size();
	for (res_index = get_global_id();
		 res_index < res_limit;
		 res_index += get_global_size())
	{
		char   *vl_buf __attribute__((unused)) = NULL;

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
#if GPUJOIN_DEVICE_PROJECTION_EXTRA_SIZE > 0
							   extra_buf,
#else
							   NULL,
#endif
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
		if (get_local_id() == 0)
		{
			if (count > 0)
				base = atomicAdd(&kds_dst->usage, count);
			else
				base = 0;
		}
		__syncthreads();
		if (KERN_DATA_STORE_SLOT_LENGTH(kds_dst, kresults->nitems) +
			base + count > kds_dst->length)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			goto out;
		}
		vl_buf = ((char *)kds_dst + kds_dst->length
				  - (base + offset + extra_len));
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
#if GPUJOIN_DEVICE_PROJECTION_EXTRA_SIZE > 0
					else if (tup_depth[i] < 0)
					{
						cl_uint		vl_len = (cmeta.attlen > 0 ?
											  cmeta.attlen :
											  VARSIZE_ANY(addr));
						memcpy(vl_buf, addr, vl_len);
						tup_values[i] = devptr_to_host(kds_dst, vl_buf);
						vl_buf += MAXALIGN(vl_len);
					}
#endif
					else
						STROM_SET_ERROR(&kcxt.e,
										StromError_WrongCodeGeneration);
				}
			}
		}
	}
out:
	/* write-back execution status to host-side */
	kern_writeback_error_status(&kgjoin->kerror, kcxt.e);
}
#endif	/* __CUDACC__ */
#endif	/* CUDA_GPUJOIN_H */
