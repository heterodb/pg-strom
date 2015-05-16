/*
 * cuda_gpujoin.h
 *
 * GPU accelerated parallel relations join based on hash-join or
 * nested-loop logic.
 * --
 * Copyright 2011-2015 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2015 (C) The PG-Strom Development Team
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

#define KERN_MULTIRELS_INNER_HASH(kmrels, depth)			\
	((kern_hashtable *)										\
	 ((char *)(kmrels) + (kmrels)->chunks[(depth)-1].chunk_offset))

#define KERN_MULTIRELS_INNER_NITEMS(kmrels, depth)	\
	(KERN_MULTIRELS_INNER_KDS(kmrels, depth)->nitems)

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
 * Hash table and entry
 *
 *
 * +-------------------+
 * | kern_hashtable    |
 * | +-----------------+
 * | |   :             |
 * | | ncols (=M)      |
 * | | nitems (=K)     |
 * | | nslots (=N)     |
 * | |   :             |
 * | +-----------------+
 * | | colmeta[0]      |
 * | |   :             |
 * | | colmeta[M-1]    |
 * +-+-----------------+
 * | cl_uint           |
 * |   hash_slot[0]    |
 * |     :             |
 * |   hash_slot[K-2] o-----+
 * |   hash_slot[K-1]  |    |
 * +-------------------+ <--+
 * | kern_hashentry    |
 * | +-----------------|
 * | | hash            |
 * | | next          o------+
 * | | rowid           |    |
 * | | htup            |    |
 * +-+-----------------+    |
 * |     :             |    |
 * +-------------------+ <--+  <---+
 * | kern_hashentry    |           |
 * | +-----------------|           |
 * | | hash            |           |
 * | | next            |           |
 * | | rowid           |           |
 * | | htup            |           |
 * +-+-----------------+           |
 * | cl_uint           |           |
 * |   row_index[0]    |           |
 * |   row_index[1]  o-------------+
 * |     :             |
 * |   row_offset[N-1] |
 * +-------------------+
 */
typedef struct
{
	cl_uint			hash;   	/* 32-bit hash value */
	cl_uint			next;		/* offset of the next */
	cl_uint			rowid;		/* unique identifier of this hash entry */
	cl_uint			t_len;		/* length of the tupe itself */
	HeapTupleHeaderData htup;	/* variable length heap-tuple */
} kern_hashentry;

typedef struct
{
	hostptr_t		hostptr;
	cl_uint			length;		/* length of this hashtable chunk */
	cl_uint			usage;		/* usage of this hashtable chunk */
	cl_uint			ncols;		/* number of inner relation's columns */
	cl_uint			nitems;		/* number of inner relation's items */
	/* NOTE: !fields above are compatible with kern_data_store! */
	cl_uint			nslots;		/* width of hash slot */
	cl_char			__dummy1__;	/* for layout compatibility to KDS */
	cl_char			__dummy2__;	/* for layout compatibility to KDS */
	cl_uint			hash_min;	/* minimum hash value */
	cl_uint			hash_max;	/* maximum hash value */
	/*
	 * NOTE: offsetof(kern_hashtable, colmeta) should be equivalent to
	 * offsetof(kern_data_store, colmeta) for code simplification.
	 */
	kern_colmeta	colmeta[FLEXIBLE_ARRAY_MEMBER];
} kern_hashtable;

#define KERN_HASHTABLE_SLOT(khtable)					\
	((cl_uint *)((char *)(khtable) +					\
				 LONGALIGN(offsetof(kern_hashtable,		\
									colmeta[(khtable)->ncols]))))

STATIC_INLINE(kern_hashentry *)
KERN_HASH_FIRST_ENTRY(kern_hashtable *khtable, cl_uint hash)
{
	cl_uint	   *slot = KERN_HASHTABLE_SLOT(khtable);
	cl_uint		index = hash % khtable->nslots;

	if (slot[index] == 0)
		return NULL;
	return (kern_hashentry *)((char *) khtable + slot[index]);
}

STATIC_INLINE(kern_hashentry *)
KERN_HASH_NEXT_ENTRY(kern_hashtable *khtable, kern_hashentry *khentry)
{
	if (!khentry || khentry->next == 0)
		return NULL;
	return (kern_hashentry *)((char *)khtable + khentry->next);
}

/*
 * kern_gpujoin - control object of GpuJoin
 */
typedef struct
{
	size_t			kresults_1_offset;
	size_t			kresults_2_offset;
	size_t			kresults_total_items;
	size_t			kresults_max_items;
	cl_uint			max_depth;
	cl_int			errcode;
	kern_parambuf	kparams;
} kern_gpujoin;

#define KERN_GPUJOIN_PARAMBUF(kgjoin)			\
	((kern_parambuf *)(&(kgjoin)->kparams))
#define KERN_GPUJOIN_PARAMBUF_LENGTH(kgjoin)	\
	STROMALIGN(KERN_GPUJOIN_PARAMBUF(kgjoin)->length)
#define KERN_GPUJOIN_HEAD_LENGTH(kgjoin)		\
	((kgjoin)->kresults_1_offset +				\
	 STROMALIGN(offsetof(kern_resultbuf, results[0])))
#define KERN_GPUJOIN_IN_RESULTBUF(kgjoin,depth)			\
	((kern_resultbuf *)((char *)(kgjoin) +				\
						(((depth) & 0x01)				\
						 ? (kgjoin)->kresults_1_offset	\
						 : (kgjoin)->kresults_2_offset)))
#define KERN_GPUJOIN_OUT_RESULTBUF(kgjoin,depth)		\
	((kern_resultbuf *)((char *)(kgjoin) +				\
						(((depth) & 0x01)				\
						 ? (kgjoin)->kresults_2_offset  \
						 : (kgjoin)->kresults_1_offset)))



#ifdef __CUDACC__

/* utility macros for automatically generated code */
#define GPUJOIN_REF_HTUP(chunk,offset)			\
	((offset) == 0								\
	 ? NULL										\
	 : (HeapTupleHeaderData *)((char *)(chunk) + (offset)))
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
gpujoin_outer_quals(cl_int *errcode,
					kern_parambuf *kparams,
					kern_data_store *kds,
					size_t kds_index);
/*
 * gpujoin_join_quals
 *
 * Evaluation of join qualifier in the given depth. It shall return true
 * if supplied pair of the rows matches the join condition.
 */
STATIC_FUNCTION(cl_bool)
gpujoin_join_quals(cl_int *errcode,
				   kern_parambuf *kparams,
				   kern_data_store *kds,
				   kern_multirels *kmrels,
				   int depth,
				   cl_int *outer_index,
				   HeapTupleHeaderData *htup);

/*
 * gpujoin_hash_value
 *
 * Calculation of hash value if this depth uses hash-join logic.
 */
STATIC_FUNCTION(cl_uint)
gpujoin_hash_value(cl_int *errcode,
				   kern_parambuf *kparams,
				   cl_uint *pg_crc32_table,
				   kern_data_store *kds,
				   kern_multirels *kmrels,
				   cl_int depth,
				   cl_int *outer_index);

/*
 * gpujoin_projection_mapping
 *
 * Lookup source depth/colidx pair by the destination colidx.
 */
STATIC_FUNCTION(void)
gpujoin_projection_mapping(cl_int dest_colidx,
						   cl_int *src_depth,
						   cl_int *src_colidx);


KERNEL_FUNCTION(void)
gpujoin_preparation(kern_gpujoin *kgjoin,
					kern_data_store *kds,
					kern_multirels *kmrels,
					cl_int depth)
{
	kern_parambuf  *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_resultbuf *kresults_in;
	kern_resultbuf *kresults_out;
	cl_int			errcode;

	/* sanity check */
	if (get_global_id() == 0)
	{
		assert(depth > 0 && depth <= kgjoin->max_depth);
		assert(kgjoin->kresults_1_offset > 0);
		assert(kgjoin->kresults_2_offset > 0);
	}
	kresults_in = KERN_GPUJOIN_IN_RESULTBUF(kgjoin, depth);
	kresults_out = KERN_GPUJOIN_OUT_RESULTBUF(kgjoin, depth);
	errcode = (depth > 1 ? kresults_in->errcode : StromError_Success);

	/*
	 * In case of depth == 1, input result buffer is not initialized
	 * yet. So, we need to put initial values first of all.
	 *
	 * NOTE: gpunestloop_prep(depth==1) expects base portion of the
	 * kern_resultbuf structure is zero cleared, so host-side has to
	 * call cuMemsetD32() or others.
	 */
	if (depth == 1)
	{
		cl_bool		is_matched;
		cl_uint		count;
		cl_uint		offset;
		__shared__ cl_int	base;

		/*
		 * Check qualifier of outer scan that was pulled-up (if any).
		 * then, it allocates result buffer on kresults_in and put
		 * get_global_id() if it match.
		 */
		if (get_global_id() < kds->nitems)
			is_matched = gpujoin_outer_quals(&errcode, kparams, kds,
											 get_global_id());
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
		}
		__syncthreads();

		if (base + count > kgjoin->kresults_total_items)
		{
			STROM_SET_ERROR(&errcode, StromError_DataStoreNoSpace);
			goto out;
		}

		if (is_matched)
		{
			HeapTupleHeaderData	   *htup
				= kern_get_tuple_row(kds, get_global_id());
			kresults_in->results[base + offset] = (size_t)htup - (size_t)kds;
		}

		/* init other base portion */
		if (get_global_id() == 0)
		{
			kresults_in->nrels = 1;
			kresults_in->nrooms = kds->nitems;
			kresults_in->errcode = StromError_Success;
		}
	}

	/* init output kresults buffer */
	if (get_global_id() == 0)
	{
		assert(kresults_in->nrels == depth);
		kresults_out->nrels = depth + 1;
		kresults_out->nrooms = kgjoin->kresults_total_items / (depth + 1);
		kresults_out->nitems = 0;
		kresults_out->errcode = StromError_Success;
	}
out:
	/* Update required length of kresults if overflow. */
	if (depth > 1 && get_global_id() == 0)
	{
		size_t	last_total_items = kresults_in->nrels * kresults_in->nitems;
		if (kgjoin->kresults_max_items < last_total_items)
			kgjoin->kresults_max_items = last_total_items;
	}
	kern_writeback_error_status(&kresults_out->errcode, errcode);
}

KERNEL_FUNCTION_MAXTHREADS(void)
gpujoin_exec_nestloop(kern_gpujoin *kgjoin,
					  kern_data_store *kds,
					  kern_multirels *kmrels,
					  cl_int depth,
					  cl_int cuda_index,
					  cl_bool *outer_join_map)
{
	kern_parambuf  *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_resultbuf *kresults_in = KERN_GPUJOIN_IN_RESULTBUF(kgjoin, depth);
	kern_resultbuf *kresults_out = KERN_GPUJOIN_OUT_RESULTBUF(kgjoin, depth);
	kern_data_store *kds_in;
	cl_bool		   *lo_map;
	size_t			nvalids;
	cl_int			y_index;
	cl_int			y_offset;
	cl_int			x_index;
	cl_int			x_limit;
	cl_int			errcode;

	/*
	 * immediate bailout if previous stage already have error status
	 */
	errcode = kresults_in->errcode;
	if (errcode != StromError_Success)
		goto out;

	/* sanity checks */
	if (get_global_id() == 0)
	{
		assert(depth > 0 && depth <= kgjoin->max_depth);
		assert(kresults_out->nrels == depth + 1);
		assert(kresults_in->nrels == depth);
	}

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
	y_index = get_global_yid();
	nvalids = min(kresults_in->nitems, kresults_in->nrooms);
	x_limit = ((nvalids + get_global_xsize() - 1) /
			   get_global_xsize()) * get_global_xsize();

	for (x_index = get_global_xid();
		 x_index < x_limit;
		 x_index += get_global_xsize())
	{
		cl_int		   *x_buffer = KERN_GET_RESULT(kresults_in, x_index);
		cl_int		   *r_buffer;
		cl_bool			is_matched;
		cl_uint			count;
		cl_uint			offset;
		__shared__ cl_int base;

		if (y_index < kds_in->nitems && x_index < nvalids)
		{
			HeapTupleHeaderData	   *htup
				= kern_get_tuple_row(kds_in, y_index);

			is_matched = gpujoin_join_quals(&errcode,
											kparams,
											kds,
											kmrels,
											depth,
											x_buffer,
											htup);
			if (is_matched)
			{
				if (lo_map && !lo_map[y_index])
					lo_map[y_index] = true;
			}
			y_offset = (size_t)htup - (size_t)kds_in;
		}
		else
			is_matched = false;

		/*
		 * Expand kresults_out->nitems, and put values
		 */
		offset = arithmetic_stairlike_add(is_matched ? 1 : 0, &count);
		if (get_local_id() == 0)
		{
			if (count > 0)
				base = atomicAdd(&kresults_out->nitems, count);
			else
				base = 0;
		}
		__syncthreads();

		/* still have space to store? */
		if (base + count > kresults_out->nrooms)
		{
			STROM_SET_ERROR(&errcode, StromError_DataStoreNoSpace);
			goto out;
		}

		if (is_matched)
		{
			assert(x_buffer != NULL);
			r_buffer = KERN_GET_RESULT(kresults_out, base + offset);
			memcpy(r_buffer, x_buffer, sizeof(cl_int) * depth);
			r_buffer[depth] = y_offset;
		}
	}
out:
	kern_writeback_error_status(&kresults_out->errcode, errcode);
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
					  cl_bool *outer_join_map)
{
	kern_parambuf  *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_resultbuf *kresults_in = KERN_GPUJOIN_IN_RESULTBUF(kgjoin, depth);
	kern_resultbuf *kresults_out = KERN_GPUJOIN_OUT_RESULTBUF(kgjoin, depth);
	kern_hashtable *khtable = KERN_MULTIRELS_INNER_HASH(kmrels, depth);
	cl_bool		   *lo_map;
	size_t			nvalids;
	cl_int			crc_index;
	cl_int			x_index;
	cl_int			x_limit;
	cl_int			errcode;
	__shared__ cl_uint base;
	__shared__ cl_uint pg_crc32_table[256];

	/*
	 * immediate bailout if previous stage already have error status
	 */
	errcode = kresults_in->errcode;
	if (errcode != StromError_Success)
		goto out;

	/* sanity checks */
	if (get_global_id() == 0)
	{
		assert(get_global_ysize() == 1);
		assert(depth > 0 && depth <= kgjoin->max_depth);
		assert(kresults_out->nrels == depth + 1);
		assert(kresults_in->nrels == depth);
	}

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
	x_limit = ((nvalids + get_global_xsize() - 1) /
			   get_global_xsize()) * get_global_xsize();

	/* will be valid, if LEFT OUTER JOIN */
	lo_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth, khtable->nitems,
										   cuda_index, outer_join_map);

	for (x_index = get_global_xid();
		 x_index < x_limit;
		 x_index += get_global_xsize())
	{
		kern_hashentry *khentry = NULL;
		cl_uint			hash_value;
        cl_int         *x_buffer = NULL;
        cl_int         *r_buffer;
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
			hash_value = gpujoin_hash_value(&errcode,
											kparams,
											pg_crc32_table,
											kds,
											kmrels,
											depth,
											x_buffer);
			if (hash_value >= khtable->hash_min &&
				hash_value <= khtable->hash_max)
			{
				khentry = KERN_HASH_FIRST_ENTRY(khtable, hash_value);
				needs_outer_row = true;
			}
		}

		/*
		 * walks on the hash entries
		 */
		do {
			if (!khentry || khentry->hash != hash_value)
				is_matched = false;
			else
			{
				assert(x_buffer != NULL);
				is_matched = gpujoin_join_quals(&errcode,
												kparams,
												kds,
												kmrels,
												depth,
												x_buffer,
												&khentry->htup);
				if (is_matched)
				{
					if (lo_map && !lo_map[khentry->rowid])
						lo_map[khentry->rowid] = true;
					needs_outer_row = false;
				}
			}

			/*
			 * Expand kresults_out->nitems
			 */
			offset = arithmetic_stairlike_add(is_matched ? 1 : 0, &count);
			if (get_local_id() == 0)
			{
				if (count > 0)
					base = atomicAdd(&kresults_out->nitems, count);
				else
					base = 0;
			}
			__syncthreads();

			/* kresults_out still have enough space? */
			if (base + count > kresults_out->nrooms)
			{
				STROM_SET_ERROR(&errcode, StromError_DataStoreNoSpace);
				goto out;
			}

			/* put offset of the items */
			if (is_matched)
			{
				r_buffer = KERN_GET_RESULT(kresults_out, base + offset);
				memcpy(r_buffer, x_buffer, sizeof(cl_int) * depth);
				r_buffer[depth] = (size_t)&khentry->htup - (size_t)khtable;
			}

			/*
			 * Fetch next hash entry, then checks whether all the local
			 * threads still have valid hash-entry or not.
			 * (NOTE: this routine contains reduction operation)
			 */
			khentry = KERN_HASH_NEXT_ENTRY(khtable, khentry);
			arithmetic_stairlike_add(khentry != NULL ? 1 : 0, &count);
		} while (count > 0);

		/*
		 * If no inner rows were matched on LEFT OUTER JOIN case, we fill
		 * up the inner-side of result tuple with NULL.
		 */
		if (KERN_MULTIRELS_LEFT_OUTER_JOIN(kmrels, depth))
		{
			offset = arithmetic_stairlike_add(needs_outer_row ? 1 : 0,
											  &count);
			if (get_local_id() == 0)
			{
				if (count > 0)
					base = atomicAdd(&kresults_out->nitems, count);
				else
					base = 0;
			}
			__syncthreads();

			/* kresults_out still have enough space? */
			if (base + count > kresults_out->nrooms)
			{
				STROM_SET_ERROR(&errcode, StromError_DataStoreNoSpace);
				goto out;
			}

			/* put offset of the outer join tuple */
			if (needs_outer_row)
			{
				assert(x_buffer != NULL);
				r_buffer = KERN_GET_RESULT(kresults_out, base + offset);
				memcpy(r_buffer, x_buffer, sizeof(cl_int) * depth);
				r_buffer[depth] = 0;	/* inner NULL */
			}
		}
	}
out:
	kern_writeback_error_status(&kresults_out->errcode, errcode);
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
					   cl_bool *outer_join_map)
{
	kern_resultbuf *kresults_out = KERN_GPUJOIN_OUT_RESULTBUF(kgjoin, depth);
	kern_data_store *kds_in = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	cl_int			errcode;
	cl_bool		   *lo_map;
	cl_bool			needs_outer_row;
	cl_uint			count;
	cl_uint			offset;
	cl_int		   *r_buffer;
	cl_int			i, ndevs = kmrels->ndevs;
	__shared__ cl_uint base;

	/* if prior stage raised an error, we skip this kernel */
	errcode = kresults_out->errcode;
	if (errcode != StromError_Success)
		goto out;

	/* sanity checks */
	if (get_global_id() == 0)
	{
		assert(get_global_xsize() == 1);
		assert(depth > 0 && depth <= kgjoin->max_depth);
		assert(kresults_out->nrels == depth + 1);
	}

	/*
	 * check whether the relevant inner tuple has any matched outer tuples,
	 * including the jobs by other devices.
	 */
	if (get_global_id() < kds_in->nitems)
	{
		cl_uint		nitems = kds_in->nitems;

		needs_outer_row = false;
		for (i=0; i < ndevs; i++)
		{
			lo_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth, nitems,
												   i, outer_join_map);
			assert(lo_map != NULL);
			needs_outer_row |= (!lo_map[get_global_id()] ? 1 : 0);
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
	}
	__syncthreads();

	/* In case when (base + num_unmatched) is larger than nrooms, it means
	 * we don't have enough space to write back nested-loop results.
	 * So, we have to tell the host-side to acquire larger kern_resultbuf.
	 */
	if (base + count > kresults_out->nrooms)
	{
		errcode = StromError_DataStoreNoSpace;
		goto out;
	}

	/*
	 * OK, we know which row should be materialized using left outer
	 * join manner, and result buffer was acquired. Let's put result
	 * for the next stage.
	 */
	if (needs_outer_row)
	{
		HeapTupleHeaderData	   *htup
			= kern_get_tuple_row(kds_in, get_global_id());
		r_buffer = KERN_GET_RESULT(kresults_out, base + offset);
		memset(r_buffer, 0, sizeof(cl_int) * depth);	/* NULL */
		r_buffer[depth] = (size_t)htup - (size_t)kds_in;
	}
out:
	kern_writeback_error_status(&kresults_out->errcode, errcode);
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
					   cl_bool *outer_join_map)
{
	kern_resultbuf *kresults_out = KERN_GPUJOIN_OUT_RESULTBUF(kgjoin, depth);
	kern_hashtable *khtable = KERN_MULTIRELS_INNER_HASH(kmrels, depth);
	kern_hashentry *khentry;
	cl_bool		   *lo_map;
	cl_bool			needs_outer_row;
	cl_int			errcode;
	cl_uint			offset;
	cl_uint			count;
	cl_int			i, ndevs = kmrels->ndevs;
	cl_int		   *r_buffer;
	__shared__ cl_uint base;

	/*
	 * immediate bailout if previous stage already have error status
	 */
	errcode = kresults_out->errcode;
	if (errcode != StromError_Success)
		goto out;

	/* sanity checks */
	if (get_global_id() == 0)
	{
		assert(get_global_ysize() == 1);
		assert(depth > 0 && depth <= kgjoin->max_depth);
		assert(kresults_out->nrels == depth + 1);
	}

	/*
	 * Fetch a hash-entry from each hash-slot
	 */
	if (get_global_id() < khtable->nslots)
		khentry = KERN_HASH_FIRST_ENTRY(khtable, get_global_id());
	else
		khentry = NULL;

	do {
		if (khentry != NULL)
		{
			/*
			 * check whether the relevant inner tuple has any matched outer
			 * tuples, including the jobs by other devices.
			 */
			cl_uint		nitems = khtable->nitems;

			assert(khentry->rowid < nitems);
			needs_outer_row = false;
			for (i=0; i < ndevs; i++)
			{
				lo_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth, nitems,
													   i, outer_join_map);
				assert(lo_map != NULL);
				needs_outer_row |= (!lo_map[khentry->rowid] ? 1 : 0);
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
		}
		__syncthreads();

		/* In case when (base + num_unmatched) is larger than nrooms, it means
		 * we don't have enough space to write back nested-loop results.
		 * So, we have to tell the host-side to acquire larger kern_resultbuf.
		 */
		if (base + count > kresults_out->nrooms)
		{
			errcode = StromError_DataStoreNoSpace;
			goto out;
		}

		/*
		 * OK, we know which row should be materialized using left outer join
		 * manner, and result buffer was acquired. Let's put result for the
		 * next stage.
		 */
		if (needs_outer_row)
		{
			r_buffer = KERN_GET_RESULT(kresults_out, base + offset);
			memset(r_buffer, 0, sizeof(cl_int) * depth);	/* NULL */
			r_buffer[depth] = (size_t)&khentry->htup - (size_t)khtable;
		}

		/*
		 * Walk on the hash-chain until all the local threads reaches to
		 * end of the hash-list
		 */
		khentry = KERN_HASH_NEXT_ENTRY(khtable, khentry);
		arithmetic_stairlike_add(khentry != NULL ? 1 : 0, &count);
	} while (count > 0);
out:
	kern_writeback_error_status(&kresults_out->errcode, errcode);
}

/*
 * gpujoin_projection_row
 *
 * It makes joined relation on kds_dst
 */
STATIC_FUNCTION(void)
__gpujoin_projection_row(cl_int *errcode,
						 kern_gpujoin *kgjoin,
						 kern_resultbuf *kresults,
						 size_t res_index,
						 kern_multirels *kmrels,
						 kern_data_store *kds_src,
						 kern_data_store *kds_dst)
{
	cl_int			nrels = kgjoin->max_depth + 1;
	cl_int		   *r_buffer;
	size_t			required;
	size_t			t_hoff;
	size_t			data_len;
	cl_bool			heap_hasnull = false;
	cl_int			src_depth;
	cl_int			src_colidx;
	kern_colmeta   *src_colmeta;
	HeapTupleHeaderData *src_htup;
	void		   *datum;
	cl_uint			offset;
	cl_uint			total_length;
	__shared__ cl_uint usage_prev;

	/* result buffer
	 * -------------
	 * r_buffer[0] -> offset from the 'kds_src'
	 * r_buffer[i; i > 0] -> offset from the kern_data_store or
	 *   kern_hashtable that can be picked up using
	 *   KERN_MULTIRELS_INNER_KDS/HASH(kmrels, depth)
	 * r_buffer[*] may be 0, if NULL-tuple was set
	 */
	if (res_index < kresults->nitems)
		r_buffer = KERN_GET_RESULT(kresults, res_index);
	else
		r_buffer = NULL;

	/*
	 * Step.1 - compute length of the joined tuple
	 */
	if (!r_buffer)
		required = 0;
	else
	{
		cl_uint		i, ncols = kds_dst->ncols;

		/* t_len and ctid */
		required = offsetof(kern_tupitem, htup);
		data_len = 0;
		heap_hasnull = false;

		/* estimation of data length */
		for (i=0; i < ncols; i++)
		{
			kern_colmeta	cmeta = kds_dst->colmeta[i];

			/* lookup source depth and colidx */
			gpujoin_projection_mapping(i, &src_depth, &src_colidx);
			assert(src_depth <= nrels);

			/* fetch source tuple */
			if (r_buffer[src_depth] == 0)
				src_htup = NULL;
			else if (src_depth == 0)
			{
				src_colmeta = kds_src->colmeta;
                src_htup = GPUJOIN_REF_HTUP(kds_src, r_buffer[0]);
			}
			else
			{
				kern_data_store *kchunk
					= KERN_MULTIRELS_INNER_KDS(kmrels, src_depth);
				src_colmeta = kchunk->colmeta;
                src_htup = GPUJOIN_REF_HTUP(kchunk, r_buffer[src_depth]);
			}
			/* fetch datum of the source tuple */
			if (!src_htup)
				datum = NULL;
			else
				datum = kern_get_datum_tuple(src_colmeta,
											 src_htup,
											 src_colidx);
			if (!datum)
				heap_hasnull = true;
			else
			{
				/* att_align_datum */
				if (cmeta.attlen > 0 || !VARATT_IS_1B(datum))
					data_len = TYPEALIGN(cmeta.attalign, data_len);
				/* att_addlength_datum */
				if (cmeta.attlen > 0)
					data_len += cmeta.attlen;
				else
					data_len += VARSIZE_ANY(datum);
			}
		}
		t_hoff = offsetof(HeapTupleHeaderData, t_bits);
		if (heap_hasnull)
			t_hoff += bitmaplen(ncols);
		if (kds_src->tdhasoid)
			t_hoff += sizeof(cl_uint);
		required += MAXALIGN(t_hoff) + MAXALIGN(data_len);
	}

	/*
	 * Step.2 - takes advance usage counter of kds_dst->usage
	 */
	offset = arithmetic_stairlike_add(required, &total_length);
	if (get_local_id() == 0)
	{
		if (total_length > 0)
			usage_prev = atomicAdd(&kds_dst->usage, total_length);
		else
			usage_prev = 0;
	}
	/* check expected usage of the buffer */
	if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
		STROMALIGN(sizeof(cl_uint) * kresults->nitems) +
		usage_prev + total_length > kds_dst->length)
	{
		STROM_SET_ERROR(errcode, StromError_DataStoreNoSpace);
		return;
	}

	/*
	 * Step.3 - construction of a heap-tuple
	 */
	if (required > 0)
	{
		HeapTupleHeaderData *htup;
		kern_tupitem   *titem;
		cl_uint		   *htup_index;
		cl_uint			htup_offset;
		cl_uint			i, ncols = kds_dst->ncols;
		cl_uint			curr;

		/* setup kern_tupitem */
		htup_offset = kds_dst->length - (usage_prev + offset + required);
		htup_index = (cl_uint *)KERN_DATA_STORE_BODY(kds_dst);
		htup_index[res_index] = htup_offset;

		/* setup header of kern_tupitem */
		titem = (kern_tupitem *)((char *)kds_dst + htup_offset);
		titem->t_len = t_hoff + data_len;
		titem->t_self.ip_blkid.bi_hi = 0xffff;	/* InvalidBlockNumber */
		titem->t_self.ip_blkid.bi_lo = 0xffff;
		titem->t_self.ip_posid = 0;				/* InvalidOffsetNumber */
		htup = &titem->htup;

		/* setup HeapTupleHeader */
		SET_VARSIZE(&htup->t_choice.t_datum, required);
		htup->t_choice.t_datum.datum_typmod = kds_dst->tdtypmod;
		htup->t_choice.t_datum.datum_typeid = kds_dst->tdtypeid;
		htup->t_ctid.ip_blkid.bi_hi = 0xffff;
		htup->t_ctid.ip_blkid.bi_lo = 0xffff;
		htup->t_ctid.ip_posid = 0;
		htup->t_infomask2 = (ncols & HEAP_NATTS_MASK);
		htup->t_infomask = (heap_hasnull ? HEAP_HASNULL : 0);
		htup->t_hoff = t_hoff;
		curr = t_hoff;

		/* setup tuple body */
		for (i=0; i < ncols; i++)
		{
			kern_colmeta	cmeta = kds_dst->colmeta[i];

			/* lookup source depth and colidx */
			gpujoin_projection_mapping(i, &src_depth, &src_colidx);
			assert(src_depth <= nrels);

			/* fetch source tuple */
			if (r_buffer[src_depth] == 0)
				src_htup = NULL;
			else if (src_depth == 0)
			{
				src_colmeta = kds_src->colmeta;
				src_htup = GPUJOIN_REF_HTUP(kds_src, r_buffer[0]);
			}
			else
			{
				kern_data_store *kchunk
					= KERN_MULTIRELS_INNER_KDS(kmrels, src_depth);
				src_colmeta = kchunk->colmeta;
				src_htup = GPUJOIN_REF_HTUP(kchunk, r_buffer[src_depth]);
			}

			/* fetch datum of the source tuple */
			if (!src_htup)
				datum = NULL;
			else
				datum = kern_get_datum_tuple(src_colmeta,
											 src_htup,
											 src_colidx);

			/* put datum on the destination kds */
			if (!datum)
				htup->t_bits[i >> 3] &= ~(1 << (i & 0x07));
			else
			{
				if (cmeta.attbyval)
				{
					char   *dest;

					while (TYPEALIGN(cmeta.attalign, curr) != curr)
						((char *)htup)[curr++] = '\0';
					dest = (char *)htup + curr;

					switch (cmeta.attlen)
					{
						case sizeof(cl_char):
							*((cl_char *) dest) = *((cl_char *) datum);
							break;
						case sizeof(cl_short):
							*((cl_short *) dest) = *((cl_short *) datum);
							break;
						case sizeof(cl_int):
							*((cl_int *) dest) = *((cl_int *) datum);
							break;
						case sizeof(cl_long):
							*((cl_long *) dest) = *((cl_long *) datum);
							break;
						default:
							memcpy(dest, datum, cmeta.attlen);
							break;
					}
					curr += cmeta.attlen;
				}
				else if (cmeta.attlen > 0)
				{
					while (TYPEALIGN(cmeta.attalign, curr) != curr)
						((char *)htup)[curr++] = '\0';

					memcpy((char *)htup + curr, datum, cmeta.attlen);

					curr += cmeta.attlen; 
				}
				else
				{
					cl_uint		vl_len = VARSIZE_ANY(datum);

					/* put 0 and align here, if not a short varlena */
					if (!VARATT_IS_1B(datum))
					{
						while (TYPEALIGN(cmeta.attalign, curr) != curr)
							((char *)htup)[curr++] = 0;
					}
					memcpy((char *)htup + curr, datum, vl_len);
					curr += vl_len;
				}
				if (heap_hasnull)
					htup->t_bits[i >> 3] |= (1 << (i & 0x07));
			}
		}
		titem->t_len = curr;
	}
}

KERNEL_FUNCTION(void)
gpujoin_projection_row(kern_gpujoin *kgjoin,
					   kern_multirels *kmrels,
					   kern_data_store *kds_src,
					   kern_data_store *kds_dst)
{
	kern_resultbuf *kresults;
	size_t		res_index;
	size_t		res_limit;
	cl_int		errcode = StromError_Success;

	kresults = KERN_GPUJOIN_OUT_RESULTBUF(kgjoin, kgjoin->max_depth);

	/* sanity checks */
	if (get_global_id() == 0)
	{
		assert(offsetof(kern_data_store, colmeta) ==
			   offsetof(kern_hashtable, colmeta));
		assert(kresults->nrels == kgjoin->max_depth + 1);
		assert(kds_src->format == KDS_FORMAT_ROW &&
			   kds_dst->format == KDS_FORMAT_ROW);
	}

	if (get_global_id() == 0)
	{
		/*
		 * Update max number of items on kern_resultbuf, if overflow.
		 * Kernel retry will adjust length according to this information.
		 */
		size_t	last_total_items = kresults->nrels * kresults->nitems;

		if (kgjoin->kresults_max_items < last_total_items)
			kgjoin->kresults_max_items = last_total_items;
		/*
		 * Update nitems of kds_dst. note that get_global_id(0) is not always
		 * called earlier than other thread. So, we should not expect nitems
		 * of kds_dst is initialized.
		 */		
		kds_dst->nitems = kresults->nitems;
	}

	/* Case of overflow; it shall be retried or executed by CPU instead,
	 * so no projection is needed anyway. We quickly exit the kernel.
	 * No need to set an error code because kern_gpuhashjoin_main()
	 * should already set it.
	 */
	if (kresults->nitems > kresults->nrooms ||
		kresults->nitems > kds_dst->nrooms)
	{
		STROM_SET_ERROR(&errcode, StromError_DataStoreNoSpace);
		goto out;
	}

	/* Do projection if thread is responsible */
	res_limit = ((kresults->nitems + get_local_size() - 1) /
				 get_local_size()) * get_local_size();
	for (res_index = get_global_id();
		 res_index < res_limit;
		 res_index += get_global_size())
	{
		__gpujoin_projection_row(&errcode, kgjoin, kresults,
								 res_index, kmrels, kds_src, kds_dst);
	}
out:
	/* write-back execution status to host-side */
	kern_writeback_error_status(&kgjoin->errcode, errcode);
}

STATIC_FUNCTION(void)
__gpujoin_projection_slot(cl_int *errcode,
						  kern_resultbuf *kresults,
						  size_t res_index,
						  kern_multirels *kmrels,
						  kern_data_store *kds_src,
						  kern_data_store *kds_dst)
{
	cl_int	   *r_buffer = KERN_GET_RESULT(kresults, res_index);
	Datum	   *slot_values = KERN_DATA_STORE_VALUES(kds_dst, res_index);
	cl_char	   *slot_isnull = KERN_DATA_STORE_ISNULL(kds_dst, res_index);
	cl_int		nrels = kresults->nrels;
	cl_int		i, ncols = kds_dst->ncols;

	/* result buffer
	 * -------------
	 * r_buffer[0] -> offset from the 'kds_src'
	 * r_buffer[i; i > 0] -> offset from the kern_data_store or
	 *   kern_hashtable that can be picked up using
	 *   KERN_MULTIRELS_INNER_KDS/HASH(kmrels, depth)
	 * r_buffer[*] may be 0, if NULL-tuple was set
	 */
	for (i=0; i < ncols; i++)
	{
		kern_colmeta	cmeta = kds_dst->colmeta[i];
		void		   *datum;
		cl_int			src_depth;
		cl_int			src_colidx;
		hostptr_t	   *src_hostptr;
		kern_colmeta   *src_colmeta;
		HeapTupleHeaderData *src_htup;

		/* lookup source depth and colidx */
		gpujoin_projection_mapping(i, &src_depth, &src_colidx);
		assert(src_depth <= nrels);

		/* fetch source tuple */
		if (r_buffer[src_depth] == 0)
			src_htup = NULL;
		else if (src_depth == 0)
		{
			src_colmeta = kds_src->colmeta;
			src_htup = GPUJOIN_REF_HTUP(kds_src, r_buffer[0]);
			src_hostptr = &kds_src->hostptr;
		}
		else
		{
			kern_data_store *kchunk
				= KERN_MULTIRELS_INNER_KDS(kmrels, src_depth);
			src_colmeta = kchunk->colmeta;
			src_htup = GPUJOIN_REF_HTUP(kchunk, r_buffer[src_depth]);
			src_hostptr = &kchunk->hostptr;
		}

		/* fetch datum of the source tuple */
		if (!src_htup)
			datum = NULL;
		else
			datum = kern_get_datum_tuple(src_colmeta,
										 src_htup,
										 src_colidx);
		/* put datum onto the destination slot */
		if (!datum)
			slot_isnull[i] = true;
		else
		{
			slot_isnull[i] = false;

			if (cmeta.attbyval)
			{
				assert(cmeta.attlen <= sizeof(Datum));
				switch (cmeta.attlen)
				{
					case sizeof(cl_char):
						slot_values[i] = (Datum)(*((cl_char *) datum));
						break;
					case sizeof(cl_short):
						slot_values[i] = (Datum)(*((cl_short *) datum));
						break;
					case sizeof(cl_int):
						slot_values[i] = (Datum)(*((cl_int *) datum));
						break;
					case sizeof(cl_long):
						slot_values[i] = (Datum)(*((cl_long *) datum));
						break;
					default:
						memcpy(slot_values + i, datum, cmeta.attlen);
						break;
				}
			}
			else
			{
				slot_values[i] = (Datum)((hostptr_t) datum -
										 (hostptr_t) src_hostptr +
										 *src_hostptr);
			}
		}
	}
}

KERNEL_FUNCTION(void)
gpujoin_projection_slot(kern_gpujoin *kgjoin,
						kern_multirels *kmrels,
						kern_data_store *kds_src,
						kern_data_store *kds_dst)
{
	kern_resultbuf *kresults;
	cl_int		errcode;
	size_t		res_index;

	kresults = KERN_GPUJOIN_OUT_RESULTBUF(kgjoin, kgjoin->max_depth);
	errcode = kresults->errcode;

	/* Update resource consumption information */
	if (get_global_id() == 0)
	{
		/*
		 * Update max number of items on kern_resultbuf, if overflow.
		 * Kernel retry will adjust length according to this information.
		 */
		size_t	last_total_items = kresults->nrels * kresults->nitems;

		if (kgjoin->kresults_max_items < last_total_items)
			kgjoin->kresults_max_items = last_total_items;
		/*
		 * Update nitems of kds_dst. note that get_global_id(0) is not always
		 * called earlier than other thread. So, we should not expect nitems
		 * of kds_dst is initialized.
		 */		
		kds_dst->nitems = kresults->nitems;
	}

	if (errcode != StromError_Success)
		goto out;

	/* sanity checks */
	if (get_global_id() == 0)
	{
		assert(offsetof(kern_data_store, colmeta) ==
			   offsetof(kern_hashtable, colmeta));
		assert(kresults->nrels == kgjoin->max_depth + 1);
		assert(kds_src->format == KDS_FORMAT_ROW &&
			   kds_dst->format == KDS_FORMAT_SLOT);
	}

	/* Case of overflow; it shall be retried or executed by CPU instead,
	 * so no projection is needed anyway. We quickly exit the kernel.
	 * No need to set an error code because kern_gpuhashjoin_main()
	 * should already set it.
	 */
	if (kresults->nitems > kresults->nrooms ||
		kresults->nitems > kds_dst->nrooms)
	{
		STROM_SET_ERROR(&errcode, StromError_DataStoreNoSpace);
		goto out;
	}

	/* Do projection if thread is responsible */
	for (res_index = get_global_id();
		 res_index < kresults->nitems;
		 res_index += get_global_size())
	{
		__gpujoin_projection_slot(&errcode, kresults, res_index,
								  kmrels, kds_src, kds_dst);
	}
out:
	/* write-back execution status to host-side */
	kern_writeback_error_status(&kgjoin->errcode, errcode);
}

#endif	/* __CUDACC__ */
#endif	/* CUDA_GPUJOIN_H */
