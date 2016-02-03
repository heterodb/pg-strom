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
		cl_ushort	gnl_shmem_xsize;/* dynamix shmem size of xitem if NL */
		cl_ushort	gnl_shmem_ysize;/* dynamix shmem size of yitem if NL */
		cl_bool		is_nestloop;	/* true, if NestLoop. */
		cl_bool		left_outer;		/* true, if JOIN_LEFT or JOIN_FULL */
		cl_bool		right_outer;	/* true, if JOIN_RIGHT or JOIN_FULL */
		cl_char		__padding__[1];
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
typedef struct
{
	cl_uint		kds_dst_length;	/* copy of kds_dst->length */
	cl_uint		source_nitems;	/* source nitems that is actually inputed */
	cl_uint		source_matched;	/* source nitems matched to the outer quels */
	struct {
		cl_uint	inner_base;		/* base of the inner window */
		cl_uint	inner_size;		/* size of the inner window */
		cl_uint inner_nitems;	/* out: number of inner join results */
		cl_uint total_nitems;	/* out: number of (inner+outer) join results */
	} r[FLEXIBLE_ARRAY_MEMBER];
} kern_join_scale;

typedef struct
{
	cl_uint			kparams_offset;		/* offset to the kparams */
	cl_uint			kresults_1_offset;	/* offset to the 1st kresults buffer */
	cl_uint			kresults_2_offset;	/* offset to the 2nd kresults buffer */
	cl_uint			kresults_max_items;	/* max items kresults buffer can hold */
	/* number of inner relations */
	cl_uint			num_rels;
	/* least depth in this call chain */
	cl_uint			start_depth;
	/* least depth in case of RIGHT/FULL OUTER JOIN */
	cl_uint			outer_join_start_depth;
	/* error status to be backed (OUT) */
	kern_errorbuf	kerror;
	/*
	 * Performance statistics
	 */
	cl_uint			num_kern_outer_eval;
	cl_uint			num_kern_nestloop;
	cl_uint			num_kern_hashjoin;
	cl_uint			num_kern_nestloop_outer;
	cl_uint			num_kern_hashjoin_outer;
	cl_uint			num_kern_projection_row;
	cl_uint			num_kern_projection_slot;
	cl_float		usec_kern_outer_eval;
	cl_float		usec_kern_nestloop;
	cl_float		usec_kern_hashjoin;
	cl_float		usec_kern_nestloop_outer;
	cl_float		usec_kern_hashjoin_outer;
	cl_float		usec_kern_projection;
	/*
	 * Scale of inner virtual window for each depth
	 */
	kern_join_scale	jscale;
} kern_gpujoin;

#define KERN_GPUJOIN_PARAMBUF(kgjoin)			\
	((kern_parambuf *)((kgjoin)->jscale.r + (kgjoin)->num_rels))
#define KERN_GPUJOIN_PARAMBUF_LENGTH(kgjoin)	\
	STROMALIGN(KERN_GPUJOIN_PARAMBUF(kgjoin)->length)
#define KERN_GPUJOIN_HEAD_LENGTH(kgjoin)				\
	STROMALIGN((char *)KERN_GPUJOIN_PARAMBUF(kgjoin) +	\
			   KERN_GPUJOIN_PARAMBUF_LENGTH(kgjoin) -	\
			   (char *)(kgjoin))

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

KERNEL_FUNCTION(void)
gpujoin_source_eval(kern_gpujoin *kgjoin,
					kern_data_store *kds,
					kern_resultbuf *kresults)
{
	cl_uint		kds_index = get_global_id();
	cl_uint		count;
	cl_uint		offset;
	cl_bool		matched;
	__shared__ cl_int base;

	assert(kresults->nrels == 1);	/* only happen if depth == 1 */

	if (get_global_id() < kds->nitems)
		matched = gpujoin_outer_quals(&kcxt, kds, get_global_id());
	else
		matched = false;

	/* expand kresults->nitems */
	offset = arithmetic_stairlike_add(matched ? 1 : 0, &count);
	if (count > 0)
	{
		if (get_local_id() == 0)
			base = atomicAdd(&kresults->nitems, count);
		__syncthreads();

		if (base + count >= kresults->nrooms)
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		else if (matched)
		{
			HeapTupleHeaderData	   *htup
				= kern_get_tuple_row(kds, get_global_id());
			kresults->results[base + offset]
				= (size_t)htup - (size_t)kds;
		}
	}
out:
	kern_writeback_error_status(&kgjoin->kerror, kcxt,e);
}

/*
 * argument layout to launch inner/outer join functions
 */
typedef struct
{
	kern_gpujoin	   *kgjoin;
	kern_data_store	   *kds;
	kern_multirels	   *kmrels;
	cl_bool			   *outer_join_map;
	cl_int				depth;
	cl_int				cuda_index;
} kern_join_args_t;


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
					   kern_data_store *kds_dst,
					   kern_resultbuf *kresults)
{
	kern_parambuf  *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
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
#if GPUJOIN_DEVICE_PROJECTION_NFIELDS > 0
		Datum		tup_values[GPUJOIN_DEVICE_PROJECTION_NFIELDS];
		cl_bool		tup_isnull[GPUJOIN_DEVICE_PROJECTION_NFIELDS];
		cl_short	tup_depth[GPUJOIN_DEVICE_PROJECTION_NFIELDS];
#else
		Datum	   *tup_values = NULL;
		cl_bool	   *tup_isnull = NULL;
		cl_short   *tup_depth = NULL;
#endif
#if GPUJOIN_DEVICE_PROJECTION_EXTRA_SIZE > 0
		cl_char		extra_buf[GPUJOIN_DEVICE_PROJECTION_EXTRA_SIZE]
					__attribute__ ((aligned(MAXIMUM_ALIGNOF)));
#else
		cl_char	   *extra_buf = NULL;
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
#else
		assert(extra_len == 0);
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

/*
 * returns true, if major retry, elsewhere minor retry
 *
 *
 */
STATIC_FUNCTION(cl_bool)
gpujoin_resize_inner_window(kern_gpujoin *kgjoin,
							kern_multirels *kmrels,
							kern_data_store *kds_src,
							cl_int depth)
{


}

#define TIMEVAL_RECORD(kgjoin,field,tv1,tv2,smx_clock)	\
	do {												\
		(kgjoin)->num_##field++;						\
		(kgjoin)->usec_##field += (cl_float)			\
			((1000 * ((tv2) - (tv1))) / (smx_clock));	\
	} while(0)

/*
 *
 *
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
	cl_int				device;
	cl_int				smx_clock;
	cl_int				depth;
	cl_int				errcode;
	cl_ulong			tv1, tv2;
	cudaError_t			status = cudaSuccess;

	/* Init kernel context */
	INIT_KERNEL_CONTEXT(&kcxt, gpujoin_main, kparams);
	assert(get_global_size() == 1);		/* only single thread */
	assert(kds_src != NULL
		   ? kgjoin->start_depth == 1
		   : kgjoin->start_depth > 0 && kgjoin->start_depth <= kgjoin->num_rels);
	assert(!kds_src || kds_src->format == KDS_FORMAT_ROW);
	assert(kds_dst->format == KDS_FORMAT_ROW ||
		   kds_dst->format == KDS_FORMAT_SLOT);

	/* Get device clock for performance monitor */
	status = cudaGetDevice(&device);
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
		return;
	}

	status = cudaDeviceGetAttribute(&smx_clock,
									cudaDevAttrClockRate,
									device);
	status = cudaGetDevice(&device);
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
		return;
	}

retry_major:
	for (depth = kgjoin->start_depth; depth <= kgjoin->num_rels; depth++)
	{
		/*
		 * Initialization of the kresults_src buffer if start depth.
		 * Elsewhere, kresults_dst buffer of the last depth is also
		 * kresults_src buffer in this depth.
		 */
	retry_minor:
		if (depth == kgjoin->start_depth)
		{
			memset(kresults_src, 0, offsetof(kern_resultbuf, results[0]));
			kresults_src->nrels = depth;
			kresults_src->nrooms = kresults_max_items / (depth + 1);
			if (kds_src != NULL)
			{
				/* Launch:
				 * gpujoin_source_eval(kern_gpujoin *kgjoin,
				 *                     kern_data_store *kds,
				 *                     kern_resultbuf *kresults)
				 */
				tv1 = clock64();
				kern_args = (void **)
					cudaGetParameterBuffer(sizeof(void *),
										   sizeof(void *) * 3);
				if (!kern_args)
				{
					STROM_SET_ERROR(&kgjoin->kerror,
									StromError_OutOfKernelArgs);
					return;
				}
				kern_args[0] = kgjoin;
				kern_args[1] = kds_src;
				kern_args[2] = kresults_src;

				status = pgstrom_optimal_workgroup_size(&grid_sz,
														&block_sz,
														(const void *)
														gpujoin_source_eval,
														kds_src->nitems,
														sizeof(kern_errorbuf));
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
					return;
				}
				// update source_nitems

				status = cudaLaunchDevice((void *)gpujoin_source_eval,
										  kern_args, grid_sz, block_sz,
										  sizeof(kern_errorbuf) * block_sz.x,
										  NULL);
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
					return;
				}

				status = cudaDeviceSynchronize();
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
					return;
				}
				tv2 = clock64();
				TIMEVAL_RECORD(kgjoin,kern_outer_eval,tv1,tv2,smx_clock);
				if (kgjoin->kerror.errcode != StromError_Success)
					return;
				/* update run-time statistics */
				kgjoin->jscale.source_nitems = kds_src->nitems;
				kgjoin->jscale.source_matched = kresults_src->nitems;
			}
		}
		/* make the kresults_dst buffer empty */
		memset(kresults_dst, 0, offsetof(kern_resultbuf, results[0]));
		kresults_dst->nrels = depth + 1;
		kresults_dst->nrooms = kresults_max_items / (depth + 1);

		if (kmrels->chunks[depth-1].is_nestloop)
		{
			if (kds_src != NULL || depth > kgjoin->outer_join_start_depth)
			{
				cl_ushort	gnl_shmem_xsize;
				cl_ushort	gnl_shmem_ysize;

				/* Launch:
				 * KERNEL_FUNCTION_MAXTHREADS(void)
				 * gpujoin_exec_nestloop(kern_gpujoin *kgjoin,
				 *                       kern_data_store *kds,
				 *                       kern_multirels *kmrels,
				 *                       cl_bool *outer_join_map,
				 *                       cl_int depth,
				 *                       cl_int cuda_index)
				 */
				tv1 = clock64();
				kern_join_args = (kern_join_args_t *)
					cudaGetParameterBuffer(sizeof(void *),
										   sizeof(kern_join_args_t));
				if (!kern_join_args)
				{
					STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
					goto out;
				}
				kern_join_args->kgjoin = kgjoin;
				kern_join_args->kds = kds_src;
				kern_join_args->kmrels = kmrels;
				kern_join_args->outer_join_map = outer_join_map;
				kern_join_args->depth = depth;
				kern_join_args->cuda_index = cuda_index;

				gnl_shmem_xsize = kmrels->chunks[depth-1].gnl_shmem_xsize;
				gnl_shmem_ysize = kmrels->chunks[depth-1].gnl_shmem_ysize;
				status = pgstrom_largest_workgroup_size_2d(
					&grid_sz,
					&block_sz,
					(const void *)gpujoin_exec_nestloop,
					kresults_src->nitems,
					kgjoin->jscale.r[depth-1].inner_size,
					gnl_shmem_xsize,
					gnl_shmem_ysize,
					sizeof(kern_errorbuf));
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
					return;
				}

				shmem_size = Max(sizeof(kern_errorbuf) * (block_sz.x *
														  block_sz.y),
								 gnl_shmem_xsize * block_sz.x +
								 gnl_shmem_ysize * block_sz.y);
				status = cudaLaunchDevice((void *)gpujoin_exec_nestloop,
										  kern_join_args,
										  grid_sz, block_sz,
										  shmem_size,
										  NULL);
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
					return;
				}

				status = cudaDeviceSynchronize();
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
					return;
				}
				tv2 = clock64();
				TIMEVAL_RECORD(kgjoin,kern_nestloop_outer,tv1,tv2,smx_clock);
				if (kgjoin->kerror.errcode == StromError_DataStoreNoSpace)
				{
					memset(&kgjoin->kerror, 0, sizeof(kern_errorbuf));
					if (gpujoin_resize_inner_window(kgjoin,
													kmrels,
													kds_src,
													depth))
						goto retry_major;
					goto retry_minor;
				}
				else if (kgjoin->kerror.errcode != StromError_Success)
					return;
				/* update run-time statistics */
				kgjoin->jscale.r[depth-1].inner_nitems = kresults_dst->nitems;
				kgjoin->jscale.r[depth-1].total_nitems = kresults_dst->nitems;
			}

			if (kds_src == NULL &&
				KERN_MULTIRELS_RIGHT_OUTER_JOIN(kmrels, depth))
			{
				/* Launch:
				 * KERNEL_FUNCTION(void)
				 * gpujoin_outer_nestloop(kern_gpujoin *kgjoin,
				 *                        kern_data_store *kds,
				 *                        kern_multirels *kmrels,
				 *                        cl_bool *outer_join_map,
				 *                        cl_int depth,
				 *                        cl_int cuda_index)
				 *
				 * NOTE: Host-size has to co-locate the outer join map
				 * into this device, prior to the kernel launch.
				 */
				tv1 = clock64();
				kern_join_args = (kern_join_args_t *)
					cudaGetParameterBuffer(sizeof(void *),
										   sizeof(kern_join_args_t));
				if (!kern_join_args)
				{
					STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
					goto out;
				}
				kern_join_args->kgjoin = kgjoin;
				kern_join_args->kds = kds_src;
				kern_join_args->kmrels = kmrels;
				kern_join_args->outer_join_map = outer_join_map;
				kern_join_args->depth = depth;
				kern_join_args->cuda_index = cuda_index;

				status = pgstrom_optimal_workgroup_size(&grid_sz,
														&block_sz,
														(const void *)
														gpujoin_outer_nestloop,
														inner_size,
														sizeof(kern_errorbuf));
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
					return;
				}




				shmem_size = Max(sizeof(kern_errorbuf) * (block_sz.x *
														  block_sz.y),
								 dynshmem_per_xitem * block_sz.x +
								 dynshmem_per_yitem * block_sz.y);
				status = cudaLaunchDevice((void *)gpujoin_exec_nestloop,
										  kern_join_args,
										  grid_sz, block_sz,
										  shmem_size,
										  NULL);
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
					return;
				}

				status = cudaDeviceSynchronize();
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
					return;
				}
				tv2 = clock64();
				TIMEVAL_RECORD(kgjoin,kern_nestloop,tv1,tv2,smx_clock);

				if (kgjoin->kerror.errcode == StromError_DataStoreNoSpace)
				{
					memset(kgjoin->kerror.errcode, 0, sizeof(kern_errorbuf));
					if (gpujoin_resize_inner_window(kgjoin,
													kmrels,
													kds_src,
													depth))
						goto retry_major;

					goto retry_minor;
				}
				else if (kgjoin->kerror.errcode != StromError_Success)
					return;
				/* update run-time statistics */
				kgjoin->jscale.v[depth-1].total_nitems = kresults_dst->nitems;
			}
		}
		else
		{
			if (kds_src != NULL || depth > outer_join_start_depth)
			{
				/* Launch:
				 * KERNEL_FUNCTION_MAXTHREADS(void)
				 * gpujoin_exec_hashjoin(kern_gpujoin *kgjoin,
				 *                       kern_data_store *kds,
				 *                       kern_multirels *kmrels,
				 *                       cl_bool *outer_join_map,
				 *                       cl_int depth,
				 *                       cl_int cuda_index)
				 */
				tv1 = clock64();
				kern_join_args = (kern_join_args_t *)
					cudaGetParameterBuffer(sizeof(void *),
										   sizeof(kern_join_args_t));
				if (!kern_join_args)
				{
					STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
					goto out;
				}
				kern_join_args->kgjoin = kgjoin;
				kern_join_args->kds = kds_src;
				kern_join_args->kmrels = kmrels;
				kern_join_args->outer_join_map = outer_join_map;
				kern_join_args->depth = depth;
				kern_join_args->cuda_index = cuda_index;

				status = pgstrom_optimal_workgroup_size(&grid_sz,
														&block_sz,
														(const void *)
														gpujoin_exec_hashjoin,
														kresults_src->nitems,
														sizeof(kern_errorbuf));
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
					return;
				}

				status = cudaLaunchDevice((void *)gpujoin_exec_hashjoin,
										  kern_join_args,
										  grid_sz, block_sz,
										  sizeof(kern_errorbuf) * block_sz.x,
										  NULL);
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
					return;
				}

				status = cudaDeviceSynchronize();
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
					return;
				}
				tv2 = clock64();
				TIMEVAL_RECORD(kgjoin,kern_nestloop_outer,tv1,tv2,smx_clock);

				if (kgjoin->kerror.errcode == StromError_DataStoreNoSpace)
				{
					if (gpujoin_resize_inner_window(kgjoin,
													kmrels,
													kds_src,
													depth))
						goto retry_major;

					goto retry_minor;
				}
				else if (kgjoin->kerror.errcode != StromError_Success)
					return;
				/* update run-time statistics */
				kgjoin->jscale.v[depth-1].inner_nitems = kresults_dst->nitems;
				kgjoin->jscale.v[depth-1].total_nitems = kresults_dst->nitems;
			}

			if (kds_src == NULL &&
				KERN_MULTIRELS_RIGHT_OUTER_JOIN(kmrels, depth))
			{
				/* Launch:
				 * KERNEL_FUNCTION(void)
				 * gpujoin_outer_hashjoin(kern_gpujoin *kgjoin,
				 *                        kern_data_store *kds,
				 *                        kern_multirels *kmrels,
				 *                        cl_bool *outer_join_map,
				 *                        cl_int depth,
				 *                        cl_int cuda_index);
				 */
				tv1 = clock64();
				kern_join_args = (kern_join_args_t *)
					cudaGetParameterBuffer(sizeof(void *),
										   sizeof(kern_join_args_t));
				if (!kern_join_args)
				{
					STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
					goto out;
				}
				kern_join_args->kgjoin = kgjoin;
				kern_join_args->kds = kds_src;
				kern_join_args->kmrels = kmrels;
				kern_join_args->outer_join_map = outer_join_map;
				kern_join_args->depth = depth;
				kern_join_args->cuda_index = cuda_index;

				status = pgstrom_optimal_workgroup_size(&grid_sz,
														&block_sz,
														(const void *)
														gpujoin_outer_hashjoin,
														inner_size,
														sizeof(kern_errorbuf));
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
					return;
				}

				status = cudaLaunchDevice((void *)gpujoin_outer_hashjoin,
										  kern_join_args,
										  grid_sz, block_sz,
										  sizeof(kern_errorbuf) * block_sz.x,
										  NULL);
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
					return;
				}

				status = cudaDeviceSynchronize();
				if (status != cudaSuccess)
				{
					STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
					return;
				}
				tv2 = clock64();
				TIMEVAL_RECORD(kgjoin,kern_hashjoin_outer,tv1,tv2,smx_clock);

				if (kgjoin->kerror.errcode == StromError_DataStoreNoSpace)
				{
					if (gpujoin_resize_inner_window(kgjoin,
													kmrels,
													kds_src,
													depth))
						goto retry_major;

					goto retry_minor;
				}
				else if (kgjoin->kerror.errcode != StromError_Success)
					return;
				/* update run-time statistics */
				kgjoin->jscale.v[depth-1].total_nitems = kresults_dst->nitems;
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
		STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
		return;
	}

	status = cudaLaunchDevice(gpujoin_projection,
							  kern_args, grid_sz, block_sz,
							  sizeof(kern_errorbuf) * block_sz.x,
							  NULL);
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
		return;
	}

	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kgjoin->kerror, status);
		return;
	}
	tv2 = clock64();
	TIMEVAL_RECORD(kgjoin,kern_projection,tv1,tv2,smx_clock);

	if (kgjoin->kerror.errcode == StromError_DataStoreNoSpace)
	{
		// adjust inner_size
		// if size == 0, then GpuReCheck

		// clear error
		memset(&kgjoin->kerror, 0, sizeof(kern_errorbuf));
		goto retry_major;
	}
}

#endif	/* __CUDACC__ */
#endif	/* CUDA_GPUJOIN_H */
