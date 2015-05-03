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
	struct
	{
		cl_uint		inner_offset;	/* offset to KDS or Hash table */
		cl_uint		match_offset;	/* offset to outer match map, if any */
	} krels[FLEXIBLE_ARRAY_MEMBER];
} kern_multirels;

#define KERN_MULTIRELS_INNER_KDS(kmrels, depth)							\
	((kern_data_store *)												\
	 ((char *)(kmrels) + (kmrels)->rels[(depth) - 1].inner_offset))

#define KERN_MULTIRELS_INNER_HASH(kmrels, depth)						\
	((kern_hashtable *)													\
	 ((char *)(kmrels) + (kmrels)->rels[(depth) - 1].inner_offset))

#define KERN_MULTIRELS_MATCH_MAP(kmrels, depth, match_buffer)
	((cl_bool *)														\
	 ((kmrels)->rels[(depth) - 1].match_offset > 0						\
	  ? ((char *)(match_buffer) +										\
		 (kmrels)->rels[(depth) - 1].match_offset) : NULL))

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
	cl_uint			hash;   /* 32-bit hash value */
	cl_uint			next;   /* offset of the next */
	cl_uint			rowid;	/* identifier of this hash entry */
	cl_uint			t_len;	/* length of the tuple */
	HeapTupleHeaderData htup;	/* tuple of the inner relation */
} kern_hashentry;

typedef struct
{
	hostptr_t		hostptr;
	cl_uint			length;		/* length of this hashtable chunk */
	cl_uint			usage;		/* usage of this hashtable chunk */
	cl_uint			ncols;		/* number of inner relation's columns */
	cl_uint			nitems;		/* number of inner relation's items */
	cl_uint			nslots;		/* width of hash slot */
	cl_uint			hash_min;	/* minimum hash value */
	cl_uint			hash_max;	/* maximum hash value */
	cl_char			__padding__[4];
	kern_colmeta    colmeta[FLEXIBLE_ARRAY_MEMBER];
} kern_hashtable;

#define KERN_HASHTABLE_SLOT(khtable)					\
	((cl_uint *)((char *)(khtable) +					\
				 LONGALIGN(offsetof(kern_hashtable,		\
									colmeta[(khtable)->ncols]))))

STATIC_INLINE(kern_hashentry *)
KERN_HASH_FIRST_ENTRY(kern_hashtable *khtable, cl_uint hash)
{
	cl_uint	   *slot = KERN_HASHTABLE_SLOT(khtable);
	cl_uint		index;

	if (hash





 = hash % khtable->nslots;

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
	cl_uint			max_depth;
	cl_int			errcode;
	kern_parambuf	kparams;
} kern_gpujoin;

#define KERN_GPUJOIN_PARAMBUF(kgjoin)			\
	((kern_parambuf *)(&(kgjoin)->kparams))
#define KERN_GPUJOIN_PARAMBUF_LENGTH(kgjoin)	\
	STROMALIGN(KERN_GPUJOIN_PARAMBUF(kgjoin)->length)
#define KERN_GPUJOIN_IN_RESULTBUF(kgjoin,depth)			\
	((kern_resultbuf *)((char *)(kgjoin) +				\
						(((depth) & 0x01)				\
						 ? (kgjoin)->kresults_2_offset	\
						 : (kgjoin)->kresults_1_offset)))
#define KERN_GPUJOIN_OUT_RESULTBUF(kgjoin,depth)		\
	((kern_resultbuf *)((char *)(kgjoin) +				\
						(((depth) & 0x01)				\
						 ? (kgjoin)->kresults_2_offset  \
						 : (kgjoin)->kresults_1_offset)))



#ifdef __CUDACC__
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
				   kern_multi_relstore *kmrels,
				   int depth,
				   cl_int *outer_index,
				   HeapTupleHeaderData *inner_htup);

/*
 * gpujoin_hash_value
 *
 * Calculation of hash value if this depth uses hash-join logic.
 */
STATIC_FUNCTION(cl_uint)
gpujoin_hash_value(cl_int *errcode,
				   kern_parambuf *kparams,
				   kern_data_store *kds,
				   kern_multi_relstore *kmrels,
				   cl_int depth,
				   cl_int *outer_index);

/*
 * gpujoin_projection_*
 *
 * Lookup destination colidx by source depth/colidx pair, or its reverse.
 */
STATIC_FUNCTION(cl_int)
gpujoin_projection_forward(cl_int src_depth,
						  cl_int src_colidx);
STSTIC_FUNCTION(void)
gpujoin_projection_reverse(cl_int dest_resno,
						   cl_int *src_depth,
						   cl_int *src_colidx);








KERNEL_FUNCTION(void)
gpujoin_preparation(kern_gpujoin *kgjoin,
					kern_data_store *kds,
					kern_multi_relstore *kmrels,
					cl_int depth)
{
	kern_resultbuf *kresults_in;
	kern_resultbuf *kresults_out;
	cl_int			errcode = StromError_Success;

	/* sanity check */
	assert(depth > 0 && depth <= kgjoin->max_depth);
	assert(kgjoin->kresults_1_offset > 0);
	assert(kgjoin->kresults_2_offset > 0);

	kresults_in = KERN_GPUJOIN_IN_RESULTSBUF(kgjoin, depth);
	kresults_out = KERN_GPUJOIN_OUT_RESULTSBUF(kgjoin, depth);

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
		cl_int		num_matched;

		/*
		 * Check qualifier of outer scan that was pulled-up (if any).
		 * then, it allocates result buffer on kresults_in and put
		 * get_global_id() if it match.
		 */
		if (get_global_id() < kds_in->nitems)
			is_matched = gpujoin_outer_quals(&errcode, kparams, kds,
											 get_global_id());
		else
			is_matched = false;

		/* expand kresults_in->nitems */
		offset = arithmetic_stairlike_add(is_matched ? 1 : 0, &num_matched);
		if (get_local_id() == 0)
		{
			if (num_matched > 0)
				base = atomicAdd(&kresults_in->nitems, num_matched);
			else
				base = 0;
		}
		__synchthreads();

		if (base + num_matched > kgjoin->kresults_total_items)
		{
			errcode = StromError_DataStoreNoSpace;
			goto out;
		}
		kresults->result[base + offset] = get_global_id();

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
	kern_writeback_error_status(&kgjoin->errcode, errcode);
}

KERNEL_FUNCTION_MAXTHREADS(void)
gpujoin_exec_nestloop(kern_gpujoin *kgjoin,
					  kern_data_store *kds,
					  kern_multi_relstore *kmrels,
					  cl_int depth,
					  cl_bool *left_outer_map)
{
	kern_parambuf  *kparams = KERN_NESTLOOP_PARAMBUF(kgjoin);
	kern_resultbuf *kresults_in = KERN_GPUJOIN_IN_RESULTBUF(kgjoin, depth);
	kern_resultbuf *kresults_out = KERN_GPUJOIN_OUT_RESULTBUF(kgjoin, depth);
	kern_data_store *kds_in;
	cl_int			y_index;
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
	assert(depth > 0 && depth <= kgjoin->max_depth);
	assert(kresults_out->nrels == depth + 1);
	assert(kresults_in->nrels == depth);

	/*
	 * NOTE: size of Y-axis deterministric on the time of kernel launch.
	 * host-side guarantees get_global_ysize() is larger then kds->nitems
	 * of the depth. On the other hands, nobody can know correct size of
	 * X-axis unless gpujoin_exec_nestloop() of the previous stage.
	 * So, we ensure all the outer items are picked up by the loop below.
	 */
	kds_in = KERN_MULTI_RELSTORE_INNER_KDS(kmrels, depth);
	assert(kds_in != NULL);

	y_index = get_global_yid();
	x_limit = ((kresults_in->nitems + get_global_xsize() - 1) /
			   get_global_xsize()) * get_global_xsize();
	for (x_index = get_global_xid();
		 x_index < x_limit;
		 x_index += get_global_xsize())
	{
		cl_int		   *x_buffer = KERN_GET_RESULT(kresults_in, x_index);
		cl_int		   *r_buffer;
		cl_bool			is_matched;
		__shared__ cl_int base;

		if (y_index < kds_in->nitems &&
			x_index < kresults_in->nitems)
		{
			kern_tupitem *tupitem = KERN_DATA_STORE_TUPITEM(kds_in, y_index);

			matched = gpujoin_join_quals(&errcode,
										 kparams,
										 kds,
										 kmrels,
										 depth,
										 x_buffer,
										 &tupitem->htup);
		}
		else
			matched = false;

		/* expand kresults_out->nitems */
		offset = arithmetic_stairlike_add(matched ? 1 : 0,
										  &num_matched);
		if (get_local_id() == 0)
		{
			if (num_matched > 0)
				base = atomicAdd(&kresults_out->nitems, num_matched);
			else
				base = 0;
		}
		__synchthreads();

		if (base + num_matched < kresults_out->nrooms)
		{
			if (is_matched)
			{
				rbuffer = KERN_GET_RESULT(kresults_out, base + offset);
				for (i=0; i < depth; i++)
					rbuffer[i] = xbuffer[i];
				rbuffer[depth] = y_index;
			}
		}
		else
			STROM_SET_ERROR(&errcode, StromError_DataStoreNoSpace);
	}
out:
	kern_writeback_error_status(&kresults_out->errcode, errcode);
}

KERNEL_FUNCTION(void)
gpujoin_exec_hashjoin(kern_gpujoin *kgjoin,
					  kern_data_store *kds,
					  kern_multi_relstore *kmrels,
					  cl_int depth,
					  cl_bool *left_outer_map)
{
	kern_parambuf  *kparams = KERN_NESTLOOP_PARAMBUF(kgjoin);
	kern_resultbuf *kresults_in = KERN_GPUJOIN_IN_RESULTBUF(kgjoin, depth);
	kern_resultbuf *kresults_out = KERN_GPUJOIN_OUT_RESULTBUF(kgjoin, depth);
	kern_hashtable *khtable = KERN_MULTIRELS_INNER_HASH(kmrels, depth);
	cl_uint			hash_value;
	cl_int			x_index;
	cl_int			x_limit;
	cl_int			errcode;
	__shared__ cl_uint base;
	__shared__ cl_uint crc32_table[256];

	/*
	 * immediate bailout if previous stage already have error status
	 */
	errcode = kresults_in->errcode;
	if (errcode != StromError_Success)
		goto out;

	/* sanity checks */
	assert(get_global_ysize() == 1);
	assert(depth > 0 && depth <= kgjoin->max_depth);
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
		crc32_table[crc_index] = kmrels->pg_crc32_table[crc_index];
	}
	__syncthreads();


	/*
     * NOTE: Nobody can know correct size of outer relation because it
	 * is determined in the previous step, so all we can do is to launch
	 * a particular number of threads according to statistics.
	 * We have to take the giant loop below to ensure all the outer items
	 * getting evaluated.
	 */
	x_limit = ((kresults_in->nitems + get_global_xsize() - 1) /
			   get_global_xsize()) * get_global_xsize();
	for (x_index = get_global_xid();
		 x_index < x_limit;
		 x_index += get_global_xsize())
	{
        cl_int         *x_buffer = KERN_GET_RESULT(kresults_in, x_index);
        cl_int         *r_buffer;
        cl_bool         is_matched;

		/*
		 * Calculation of hash-value of the outer relations.
		 */
		if (x_index < kresults_in->nitems)
			hash_value = gpujoin_hash_value(&errcode,
											kparams,
											kds,
											kmrels,
											depth,
											x_buffer);
		else
			hash_value = -1;

		/*
		 * walks on the hash entries
		 *
		 */
		khentry = KERN_HASH_FIRST_ENTRY(khtable, hash_value);
		do {
			if (khentry != NULL &&
				x_index < kresults_in->nitems)
				matched = gpujoin_join_quals(&errcode,
											 kparams,
											 kds,
											 kmrels,
											 depth,
											 x_buffer,
											 &khentry->htup);
			else
				matched = false;

			/* expand kresults_out->nitems */
			offset = arithmetic_stairlike_add(matched ? 1 : 0,
											  &num_matched);
			if (get_local_id() == 0)
			{
				if (num_matched > 0)
					base = atomicAdd(&kresults_out->nitems, num_matched);
				else
					base = 0;
			}
			__synchthreads();

			if (base + num_matched < kresults_out->nrooms)
			{
				if (is_matched)
				{
					rbuffer = KERN_GET_RESULT(kresults_out, base + offset);
					for (i=0; i < depth; i++)
						rbuffer[i] = xbuffer[i];
					rbuffer[depth] = hentry->rowid;
				}
				else if (is_right_outer &&
						 hash_value >= khtable->hash_min &&
						 hash_value <= khtable->hash_max)
				{
					rbuffer = KERN_GET_RESULT(kresults_out, base + offset);
					for (i=0; i < depth; i++)
						rbuffer[i] = xbuffer[i];
					rbuffer[depth] = -1;	/* filled by NULL */
				}
			}
			else
				STROM_SET_ERROR(&errcode, StromError_DataStoreNoSpace);

			/* fetch next hash entry, if any */
			khentry = KERN_HASH_NEXT_ENTRY(khtable, khentry);
		} while (num_matched == 0);
	}
out:
	kern_writeback_error_status(&kresults_out->errcode, errcode);
}

/*
 * gpunestloop_outer_checkup
 *
 * It checks referencial map of inner relations, then if nobody didn't
 * pick up the entry, it adds an outer entry for each unreferenced one.
 */
KERNEL_FUNCTION(void)
gpujoin_outer_post_process(kern_gpujoin *kgjoin,
						   kern_multi_relstore *kmrels,
						   kern_data_store *kds,
						   cl_int depth)
{
	kern_resultbuf	   *kresults_out;
	kern_data_store	   *kds_in;
	cl_bool			   *refmap;
	cl_bool				is_outer;
	cl_int				num_outer;
	cl_int				offset;
	cl_int			   *rbuffer;
	cl_int				errcode = StromError_Success;
	__shared__ cl_uint	base;
	cl_int				i;

	kresults_out = KERN_NESTLOOP_OUT_RESULTSBUF(knestloop, depth);
	kds_in = KERN_MULTI_RELSTORE_INNER_KDS(kmrels, depth);
	refmap = KERN_MULTI_RELSTORE_REFERENCE_MAP(kmrels, depth);
	if (!refmap)
	{
		STROM_SET_ERROR(&errcode, StromError_SanityCheckViolation);
		goto out;
	}

	/*
	 * check number of outer rows to be injected, then allocate slots
	 * for outer tuples.
	 */
	if (get_global_id() < kds_in->nitems)
		is_outer = !refmap[get_global_id()];
	else
		is_outer = false;
	offset = arithmetic_stairlike_add(is_outer ? 1 : 0, &num_outer);

	if (get_local_id() == 0)
	{
		if (num_outer > 0)
			base = atomicAdd(&kresults_out->nitems, num_outer);
		else
			base = 0;
	}
	__synchthreads();

	/* In case when (base + num_outer) is larger than or equal to the nrooms,
	 * it means we don't have enough space to write back nested-loop results.
	 * So, we have to tell the host-side to acquire larger kern_resultbuf.
	 */
	if (base + num_outer > kresults_out->nrooms)
	{
		errcode = StromError_DataStoreNoSpace;
		goto out;
    }

	/*
	 * OK, we know which row should be materialized using left outer join
	 * manner, and result buffer was acquired. Let's put result for the
	 * next stage.
	 */
	rbuffer = KERN_GET_RESULT(kresults_out, base + offset);
	for (i=0; i < depth; i++)
		rbuffer[i] = -1;	/* NULL */
	rbuffer[depth] = get_global_id();

out:
	kern_writeback_error_status(&knestloop->errcode, errcode);
}

KERNEL_FUNCTION(void)
gpujoin_projection_row(kern_gpujoin *kgjoin,
					   kern_multi_relstore *kmrels,
					   kern_data_store *kds_src,
					   kern_data_store *kds_dst)
{
	kern_resultbuf *kresults;
	cl_int		   *rbuffer;
	cl_int			depth = knestloop->max_depth;
	cl_int			errcode = StromError_Success;

	/* sanity checks */
	assert(kds_src->format == KDS_FORMAT_ROW);
	assert(kds_dst->format == KDS_FORMAT_ROW);

	kresults = KERN_NESTLOOP_OUT_RESULTSBUF(knestloop, depth);
	assert(kresults->nrels == depth + 1);

	for (index = get_global_id();
		 index < kresults->nitems;
		 index += get_global_size())
	{
		rbuffer = KERN_GET_RESULT(kresults, index);
		/*
		 * rbuffer[0] -> index to 'kds_src'
		 * rbuffer[i; i > 0] -> index to kern_data_store that can be
		 *   picked up using KERN_MULTI_RELSTORE_INNER_KDS(kmrels, depth)
		 *
		 * rbuffer[*} may be -1, in this case, null shall be put.
		 */









	}
out:
	kern_writeback_error_status(&knestloop->errcode, errcode);		
}

KERNEL_FUNCTION(void)
gpujoin_projection_slot(kern_gpujoin *kgjoin,
						kern_multi_relstore *kmrels,
						kern_data_store *kds_src,
						kern_data_store *kds_dst)
{
	kern_resultbuf *kresults;
	cl_int		   *rbuffer;
	cl_int			depth = knestloop->max_depth;
	cl_int			errcode = StromError_Success;

	/* sanity checks */
	assert(kds_src->format == KDS_FORMAT_ROW);
	assert(kds_dst->format == KDS_FORMAT_SLOT);

	kresults = KERN_NESTLOOP_OUT_RESULTSBUF(knestloop, depth);
	assert(kresults->nrels == depth + 1);

	for (index = get_global_id();
		 index < kresults->nitems;
		 index += get_global_size())
	{
		rbuffer = KERN_GET_RESULT(kresults, index);
		/*
		 * rbuffer[0] -> index to 'kds_src'
		 * rbuffer[i; i > 0] -> index to kern_data_store that can be
		 *   picked up using KERN_MULTI_RELSTORE_INNER_KDS(kmrels, depth)
		 *
		 * rbuffer[*} may be -1, in this case, null shall be put.
		 */






	}
out:
	kern_writeback_error_status(&knestloop->errcode, errcode);		
}

#endif	/* __CUDACC__ */
#endif	/* CUDA_GPUJOIN_H */
