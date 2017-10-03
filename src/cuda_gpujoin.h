/*
 * cuda_gpujoin.h
 *
 * GPU accelerated parallel relations join based on hash-join or
 * nested-loop logic.
 * --
 * Copyright 2011-2017 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2017 (C) The PG-Strom Development Team
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
	cl_uint			kmrels_length;	/* length of kern_multirels */
	cl_uint			ojmaps_length;	/* length of outer-join map, if any */
	cl_uint			cuda_dindex;	/* device index of PG-Strom */
	cl_uint			nrels;			/* number of inner relations */
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

#define KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth)					\
	((cl_bool *)((kmrels)->chunks[(depth)-1].right_outer				\
				 ? ((char *)(kmrels) +									\
					(size_t)(kmrels)->kmrels_length +					\
					(size_t)(kmrels)->cuda_dindex *						\
					(size_t)(kmrels)->ojmaps_length +					\
					(size_t)(kmrels)->chunks[(depth)-1].ojmap_offset)	\
				 : NULL))

#define KERN_MULTIRELS_LEFT_OUTER_JOIN(kmrels, depth)	\
	__ldg(&((kmrels)->chunks[(depth)-1].left_outer))

#define KERN_MULTIRELS_RIGHT_OUTER_JOIN(kmrels, depth)	\
	__ldg(&((kmrels)->chunks[(depth)-1].right_outer))

/*
 * kern_gpujoin - control object of GpuJoin
 */
typedef struct
{
	cl_uint		window_orig;	/* 'window_base' value on kernel invocation */
	cl_uint		window_base;	/* base of the virtual partition window */
	cl_uint		window_size;	/* size of the virtual partition window */
	cl_uint		stat_nitems;	/* out: # of join results in this depth */
	cl_float	row_dist_score;	//deprecated
} kern_join_scale;

struct kern_gpujoin
{
	cl_uint			kparams_offset;		/* offset to the kparams */
	cl_uint			pstack_offset;		/* offset to the pseudo-stack */
	cl_uint			pstack_nrooms;		/* size of pseudo-stack */
	cl_uint			num_rels;			/* number of inner relations */
	cl_uint			src_read_pos;		/* position to read from kds_src */
	/* error status to be backed (OUT) */
	cl_uint			source_nitems;		/* out: # of source rows */
	cl_uint			outer_nitems;		/* out: # of filtered source rows */
	kern_errorbuf	kerror;
	/*
	 * Scale of inner virtual window for each depth
	 * (note that jscale has (num_rels + 1) elements
	 */
	kern_join_scale	jscale[FLEXIBLE_ARRAY_MEMBER];
	/*
	 * pseudo stack shall be added next to the jscale fields
	 */
};
typedef struct kern_gpujoin		kern_gpujoin;

#define KERN_GPUJOIN_PARAMBUF(kgjoin)			\
	((kern_parambuf *)((char *)(kgjoin) + (kgjoin)->kparams_offset))
#define KERN_GPUJOIN_PARAMBUF_LENGTH(kgjoin)	\
	STROMALIGN(KERN_GPUJOIN_PARAMBUF(kgjoin)->length)
#define KERN_GPUJOIN_HEAD_LENGTH(kgjoin)				\
	STROMALIGN((char *)KERN_GPUJOIN_PARAMBUF(kgjoin) +	\
			   KERN_GPUJOIN_PARAMBUF_LENGTH(kgjoin) -	\
			   (char *)(kgjoin))
#define KERN_GPUJOIN_PSEUDO_STACK(kgjoin)		\
	((cl_uint *)((char *)(kgjoin) + (kgjoin)->pstack_offset))

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
 * static shared variables
 */
static __shared__ cl_bool	scan_done;
static __shared__ cl_int	base_depth;
static __shared__ cl_uint	src_read_pos;
static __shared__ cl_uint	dst_base_nitems;
static __shared__ cl_uint	dst_base_usage;
static __shared__ cl_uint	read_pos[GPUJOIN_MAX_DEPTH+1];
static __shared__ cl_uint	write_pos[GPUJOIN_MAX_DEPTH+1];
static __shared__ cl_uint	stat_source_nitems;
static __shared__ cl_uint	stat_nitems[GPUJOIN_MAX_DEPTH+1];
static __shared__ cl_uint	pg_crc32_table[256];

/*
 * gpujoin_rewind_stack
 */
STATIC_INLINE(cl_int)
gpujoin_rewind_stack(cl_int depth, cl_uint *l_state, cl_bool *matched)
{
	static __shared__ cl_int	__depth;

	assert(depth > base_depth);
	if (get_local_id() == 0)
	{
		__depth = depth;
		do {
			if (__depth <= GPUJOIN_MAX_DEPTH)
			{
				read_pos[__depth] = 0;
				write_pos[__depth] = 0;
			}
			if (__depth == base_depth ||
				read_pos[__depth-1] < write_pos[__depth-1])
				break;
		} while (__depth-- > base_depth);
	}
	__syncthreads();
	depth = __depth;
	if (depth < GPUJOIN_MAX_DEPTH)
	{
		memset(l_state + depth + 1, 0,
			   sizeof(cl_uint) * (GPUJOIN_MAX_DEPTH - depth));
		memset(matched + depth + 1, 0,
			   sizeof(cl_bool) * (GPUJOIN_MAX_DEPTH - depth));
	}
	return depth;
}

/*
 * gpujoin_load_source
 */
STATIC_FUNCTION(cl_int)
gpujoin_load_source(kern_context *kcxt,
					kern_gpujoin *kgjoin,
					kern_data_store *kds_src,
					cl_uint outer_unit_sz,
					cl_uint *wr_stack,
					cl_uint *l_state)
{
	HeapTupleHeaderData *htup = NULL;
	ItemPointerData	t_self;
	cl_uint			t_offset;
	cl_uint			count;
	cl_bool			visible;
	cl_uint			wr_index;

	/* extract a HeapTupleHeader */
	if (kds_src->format == KDS_FORMAT_ROW)
	{
		kern_tupitem   *tupitem;
		cl_uint			row_index;

		/* fetch next window */
		if (get_local_id() == 0)
			src_read_pos = atomicAdd(&kgjoin->src_read_pos,
									 outer_unit_sz);
		__syncthreads();
		row_index = src_read_pos + get_local_id();

		if (row_index < kds_src->nitems)
		{
			tupitem = KERN_DATA_STORE_TUPITEM(kds_src, row_index);
			t_offset = (cl_uint)((char *)&tupitem->htup - (char *)kds_src);
			t_self = tupitem->t_self;
			htup = &tupitem->htup;
		}
	}
	else
	{
		cl_uint		part_sz = KERN_DATA_STORE_PARTSZ(kds_src);
		cl_uint		part_id;
		cl_uint		line_no;
		cl_uint		n_parts;
		cl_uint		n_lines;
		cl_uint		loops = l_state[0]++;

		/* fetch next window, if needed */
		if (loops == 0 && get_local_id() == 0)
			src_read_pos = atomicAdd(&kgjoin->src_read_pos,
									 outer_unit_sz);
		__syncthreads();
		part_id = src_read_pos + get_local_id() / part_sz;
		n_parts = get_local_size() / part_sz;
		line_no = get_local_id() % part_sz + loops * part_sz + 1;

		if (part_id < kds_src->nitems &&
			get_local_id() < part_sz * n_parts)
		{
			PageHeaderData *pg_page;
			BlockNumber	block_nr;

			pg_page = KERN_DATA_STORE_BLOCK_PGPAGE(kds_src, part_id);
			n_lines = PageGetMaxOffsetNumber(pg_page);
			block_nr = KERN_DATA_STORE_BLOCK_BLCKNR(kds_src, part_id);

			if (line_no <= n_lines)
			{
				ItemIdData *lpp = PageGetItemId(pg_page, line_no);
				if (ItemIdIsNormal(lpp))
				{
					t_offset = (cl_uint)((char *)lpp - (char *)kds_src);
					t_self.ip_blkid.bi_hi = block_nr >> 16;
					t_self.ip_blkid.bi_lo = block_nr & 0xffff;
					t_self.ip_posid = line_no;

					htup = PageGetItem(pg_page, lpp);
				}
			}
		}
	}

	count = __syncthreads_count(htup != NULL);
	if (count > 0)
	{
		if (get_local_id() == 0)
			stat_source_nitems += count;

		if (htup)
			visible = gpuscan_quals_eval(kcxt,
										 kds_src,
										 &t_self,
										 htup);
		else
			visible = false;

		/* error checks */
		if (__syncthreads_count(kcxt->e.errcode) > 0)
			return -1;

		/* store the tuple-offset if visible */
		wr_index = write_pos[0];
		wr_index += pgstromStairlikeBinaryCount(visible, &count);
		if (get_local_id() == 0)
		{
			write_pos[0] += count;
			stat_nitems[0] += count;
		}
		if (visible)
			wr_stack[wr_index] = t_offset;
		__syncthreads();

		/*
		 * An iteration can fetch up to get_local_size() tuples
		 * at once, thus, we try to dive into deeper depth prior
		 * to the next outer tuples.
		 */
		if (write_pos[0] + get_local_size() > kgjoin->pstack_nrooms)
			return 1;
		__syncthreads();
	}
	else
	{
		/* no tuples we could fetch */
		assert(write_pos[0] + get_local_size() <= kgjoin->pstack_nrooms);
		l_state[0] = 0;
		__syncthreads();
	}

	/* End of the outer relation? */
	if (src_read_pos >= kds_src->nitems)
	{
		/* don't rewind the stack any more */
		if (get_local_id() == 0)
			scan_done = true;
		__syncthreads();

		/*
		 * We may have to dive into the deeper depth if we still have
		 * pending join combinations.
		 */
		if (write_pos[0] == 0)
		{
			for (cl_int depth=1; depth <= GPUJOIN_MAX_DEPTH; depth++)
			{
				if (read_pos[depth] < write_pos[depth])
					return depth+1;
			}
			return -1;
		}
		return 1;
	}
	return 0;
}

/*
 * gpujoin_load_outer
 */
STATIC_FUNCTION(cl_int)
gpujoin_load_outer(kern_context *kcxt,
				   kern_gpujoin *kgjoin,
				   kern_multirels *kmrels,
				   cl_int outer_depth,
				   cl_uint *wr_stack,
				   cl_uint *l_state)
{
	kern_data_store *kds_in = KERN_MULTIRELS_INNER_KDS(kmrels, outer_depth);
	cl_bool		   *ojmap = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, outer_depth);
	HeapTupleHeaderData *htup = NULL;
	kern_tupitem   *tupitem;
	cl_uint			t_offset;
	cl_uint			row_index;
	cl_uint			wr_index;
	cl_uint			count;

	assert(ojmap != NULL);

	if (get_local_id() == 0)
		src_read_pos = atomicAdd(&kgjoin->src_read_pos,
								 get_local_size());
	__syncthreads();
	row_index = src_read_pos + get_local_id();

	/* pickup inner rows, if unreferenced */
	if (row_index < kds_in->nitems && !ojmap[row_index])
	{
		tupitem = KERN_DATA_STORE_TUPITEM(kds_in, row_index);
		t_offset = (cl_uint)((char *)&tupitem->htup - (char *)kds_in);
		htup = &tupitem->htup;
	}

	if (__syncthreads_count(htup != NULL) > 0)
	{
		wr_index = write_pos[outer_depth];
		wr_index += pgstromStairlikeBinaryCount(htup != NULL, &count);
		if (get_local_id() == 0)
		{
			write_pos[outer_depth] += count;
			stat_nitems[outer_depth] += count;
		}
		if (htup)
		{
			wr_stack += wr_index * (outer_depth + 1);
			memset(wr_stack, 0, sizeof(cl_uint) * outer_depth);
			wr_stack[outer_depth] = t_offset;
		}
		__syncthreads();
	}

	/* end of the inner relation? */
	if (src_read_pos >= kds_in->nitems)
	{
		/* don't rewind the stack any more */
		if (get_local_id() == 0)
			scan_done = true;
		__syncthreads();

		/*
		 * We may have to dive into the deeper depth if we still have
		 * pending join combinations.
		 */
		if (write_pos[outer_depth] == 0)
		{
			for (cl_int dep=outer_depth + 1; dep <= GPUJOIN_MAX_DEPTH; dep++)
			{
				if (read_pos[dep] < write_pos[dep])
					return dep+1;
			}
			return -1;
		}
		return outer_depth+1;
	}
	return outer_depth;
}

/*
 * gpujoin_projection_row
 */
STATIC_FUNCTION(cl_int)
gpujoin_projection_row(kern_context *kcxt,
					   kern_gpujoin *kgjoin,
					   kern_multirels *kmrels,
					   kern_data_store *kds_src,
					   kern_data_store *kds_dst,
					   cl_uint *rd_stack,
					   cl_uint *l_state,
					   cl_bool *matched)
{
	cl_uint		nrels = kgjoin->num_rels;
	cl_uint		read_index;
	cl_uint		dest_index;
	cl_uint		dest_offset;
	cl_uint		count;
	cl_uint		nvalids;
	cl_uint		required;
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

	/* sanity checks */
	assert(rd_stack != NULL);

	/* Any more result rows to be written? */
	if (read_pos[nrels] >= write_pos[nrels])
	{
		if (scan_done)
			return -1;
		return gpujoin_rewind_stack(nrels+1, l_state, matched);
	}

	/* pick up combinations from the pseudo-stack */
	nvalids = Min(write_pos[nrels] - read_pos[nrels],
				  get_local_size());
	read_index = read_pos[nrels] + get_local_id();
	__syncthreads();
	if (get_local_id() == 0)
		read_pos[nrels] += get_local_size();

	/* step.1 - compute length of the result tuple to be written */
	if (read_index < write_pos[nrels])
	{
		rd_stack += read_index * (nrels + 1);

		gpujoin_projection(kcxt,
						   kds_src,
						   kmrels,
						   rd_stack,
						   kds_dst,
						   tup_values,
						   tup_isnull,
						   tup_depth,
						   extra_buf,
						   &extra_len);
		assert(extra_len <= GPUJOIN_DEVICE_PROJECTION_EXTRA_SIZE);
		required = MAXALIGN(offsetof(kern_tupitem, htup) +
							compute_heaptuple_size(kcxt,
												   kds_dst,
												   tup_values,
												   tup_isnull,
												   NULL));
	}
	else
		required = 0;

	if (__syncthreads_count(kcxt->e.errcode) > 0)
		return -1;		/* bailout */

	/* step.2 - increments nitems/usage of the kds_dst */
	dest_offset = pgstromStairlikeSum(required, &count);
	assert(count > 0);
	if (get_local_id() == 0)
	{
		dst_base_nitems = atomicAdd(&kds_dst->nitems, nvalids);
		dst_base_usage  = atomicAdd(&kds_dst->usage, count);
	}
	__syncthreads();
	dest_index = dst_base_nitems + get_local_id();
	dest_offset += dst_base_usage + required;

	if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
		STROMALIGN(sizeof(cl_uint) * (dst_base_nitems + nvalids)) +
		dst_base_usage + count > kds_dst->length)
		STROM_SET_ERROR(&kcxt->e, StromError_DataStoreNoSpace);
	if (__syncthreads_count(kcxt->e.errcode) > 0)
		return -1;	/* bailout */

	/* step.3 - write out HeapTuple on the destination buffer */
	if (required > 0)
	{
		cl_uint	   *row_index = KERN_DATA_STORE_ROWINDEX(kds_dst);
		kern_tupitem *tupitem = (kern_tupitem *)
			((char *)kds_dst + kds_dst->length - dest_offset);

		row_index[dest_index] = kds_dst->length - dest_offset;
		form_kern_heaptuple(kcxt, kds_dst, tupitem, NULL,
							tup_values, tup_isnull, NULL);
	}
	if (__syncthreads_count(kcxt->e.errcode) > 0)
		return -1;	/* bailout */

	return nrels + 1;
}

/*
 * gpujoin_exec_nestloop
 */
STATIC_FUNCTION(cl_int)
gpujoin_exec_nestloop(kern_context *kcxt,
					  kern_gpujoin *kgjoin,
					  kern_multirels *kmrels,
					  kern_data_store *kds_src,
					  cl_int depth,
					  cl_uint *rd_stack,
					  cl_uint *wr_stack,
					  cl_uint *l_state,
					  cl_bool *matched)
{
	kern_data_store *kds_in = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	cl_bool		   *oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth);
	kern_tupitem   *tupitem = NULL;
	cl_uint			kds_index;
	cl_uint			rd_index;
	cl_uint			wr_index;
	cl_uint			count;
	cl_bool			result;

	assert(kds_in->format == KDS_FORMAT_ROW);
	assert(depth >= 1 && depth <= GPUJOIN_MAX_DEPTH);

	if (l_state[depth] >= kds_in->nitems)
	{
		/*
		 * If LEFT OUTER JOIN, we need to check whether the outer
		 * combination had any matched inner tuple, or not.
		 */
		if (KERN_MULTIRELS_LEFT_OUTER_JOIN(kmrels, depth) &&
			__syncthreads_count(!matched[depth]))
		{
			if (read_pos[depth-1] + get_local_id() < write_pos[depth-1])
				result = matched[depth];
			else
				result = false;
			matched[depth] = true;
			goto left_outer;
		}
		l_state[depth] = 0;
		matched[depth] = false;
		if (get_local_id() == 0)
			read_pos[depth-1] += get_local_size();
		return depth;
	}
	else if (read_pos[depth-1] >= write_pos[depth-1])
	{
		/*
         * when this depth has enough room to store the combinations, upper
         * depth can generate more outer tuples.
         */
		if (!scan_done &&
			write_pos[depth] + get_local_size() <= kgjoin->pstack_nrooms)
			return gpujoin_rewind_stack(depth, l_state, matched);
		/* elsewhere, dive into the deeper depth or projection */
		return depth + 1;
	}
	kds_index = l_state[depth]++;
	assert(kds_index < kds_in->nitems);
	tupitem = KERN_DATA_STORE_TUPITEM(kds_in, kds_index);

	rd_index = read_pos[depth-1] + get_local_id();
	rd_stack += (rd_index * depth);
	if (rd_index < write_pos[depth-1])
	{
		result = gpujoin_join_quals(kcxt,
									kds_src,
									kmrels,
									depth,
									rd_stack,
									&tupitem->htup,
									NULL);
		if (result)
		{
			matched[depth] = true;
			if (oj_map && !oj_map[kds_index])
				oj_map[kds_index] = true;
		}
	}
	else
		result = false;

left_outer:
	wr_index = write_pos[depth];
	wr_index += pgstromStairlikeBinaryCount(result, &count);
	if (get_local_id() == 0)
	{
		write_pos[depth] += count;
		stat_nitems[depth] += count;
	}
	wr_stack += wr_index * (depth + 1);
	if (result)
	{
		memcpy(wr_stack, rd_stack, sizeof(cl_uint) * depth);
		wr_stack[depth] = (!tupitem ? 0 : (cl_uint)((char *)&tupitem->htup -
													(char *)kds_in));
	}
	__syncthreads();
	/*
	 * If we have enough room to store the combinations more, execute this
	 * depth one more. Elsewhere, dive into a deeper level to flush results.
	 */
	if (write_pos[depth] + get_local_size() <= kgjoin->pstack_nrooms)
		return depth;
	return depth + 1;
}

/*
 * gpujoin_exec_hashjoin
 */
STATIC_FUNCTION(cl_int)
gpujoin_exec_hashjoin(kern_context *kcxt,
					  kern_gpujoin *kgjoin,
					  kern_multirels *kmrels,
					  kern_data_store *kds_src,
					  cl_int depth,
					  cl_uint *rd_stack,
					  cl_uint *wr_stack,
					  cl_uint *l_state,
					  cl_bool *matched)
{
	kern_data_store	   *kds_hash = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	cl_bool			   *oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth);
	kern_hashitem	   *khitem = NULL;
	cl_uint				hash_value;
	cl_uint				rd_index;
	cl_uint				wr_index;
	cl_uint				count;
	cl_bool				result;

	assert(kds_hash->format == KDS_FORMAT_HASH);
	assert(depth >= 1 && depth <= GPUJOIN_MAX_DEPTH);

	if (__syncthreads_or(l_state[depth] != UINT_MAX) == 0)
	{
		/*
		 * OK, all the threads reached to the end of hash-slot chain
		 * Move to the next outer window.
		 */
		if (get_local_id() == 0)
			read_pos[depth-1] += get_local_size();
		l_state[depth] = 0;
		matched[depth] = false;
		return depth;
	}
	else if (read_pos[depth-1] >= write_pos[depth-1])
	{
		/*
		 * When this depth has enough room to store the combinations, upper
		 * depth can generate more outer tuples.
		 */
		if (!scan_done &&
			write_pos[depth] + get_local_size() <= kgjoin->pstack_nrooms)
			return gpujoin_rewind_stack(depth, l_state, matched);
		/* Elsewhere, dive into the deeper depth or projection */
		return depth + 1;
	}
	rd_index = read_pos[depth-1] + get_local_id();
	rd_stack += (rd_index * depth);

	if (l_state[depth] == 0)
	{
		/* first touch to the hash-slot */
		if (rd_index < write_pos[depth-1])
		{
			cl_bool		is_null_keys;

			hash_value = gpujoin_hash_value(kcxt,
											pg_crc32_table,
											kds_src,
											kmrels,
											depth,
											rd_stack,
											&is_null_keys);
			if (hash_value >= kds_hash->hash_min &&
				hash_value <= kds_hash->hash_max)
			{
				/* MEMO: NULL-keys will never match to inner-join */
				if (!is_null_keys)
					khitem = KERN_HASH_FIRST_ITEM(kds_hash, hash_value);
			}
		}
	}
	else if (l_state[depth] != UINT_MAX)
	{
		/* walks on the hash-slot chain */
		khitem = (kern_hashitem *)((char *)kds_hash
								   + l_state[depth]
								   - offsetof(kern_hashitem, t.htup));
		hash_value = khitem->hash;

		/* pick up next one if any */
		khitem = KERN_HASH_NEXT_ITEM(kds_hash, khitem);
	}

	while (khitem && khitem->hash != hash_value)
		khitem = KERN_HASH_NEXT_ITEM(kds_hash, khitem);

	if (khitem)
	{
		cl_bool		joinquals_matched;

		assert(khitem->hash == hash_value);

		result = gpujoin_join_quals(kcxt,
									kds_src,
									kmrels,
									depth,
									rd_stack,
									&khitem->t.htup,
									&joinquals_matched);
		if (joinquals_matched)
		{
			/* No LEFT/FULL JOIN are needed */
			matched[depth] = true;
			/* No RIGHT/FULL JOIN are needed */
			assert(khitem->rowid < kds_hash->nitems);
			if (oj_map && !oj_map[khitem->rowid])
				oj_map[khitem->rowid] = true;
		}
	}
	else if (KERN_MULTIRELS_LEFT_OUTER_JOIN(kmrels, depth) &&
			 l_state[depth] != UINT_MAX &&
			 !matched[depth])
	{
		/* No matched outer rows, but LEFT/FULL OUTER */
		result = true;
	}
	else
		result = false;

	/* save the current hash item */
	l_state[depth] = (!khitem ? UINT_MAX : (cl_uint)((char *)&khitem->t.htup -
													 (char *)kds_hash));
	wr_index = write_pos[depth];
	wr_index += pgstromStairlikeBinaryCount(result, &count);
	if (get_local_id() == 0)
	{
		write_pos[depth] += count;
		stat_nitems[depth] += count;
	}
	wr_stack += wr_index * (depth + 1);
	if (result)
	{
		memcpy(wr_stack, rd_stack, sizeof(cl_uint) * depth);
		wr_stack[depth] = (!khitem ? 0U : (cl_uint)((char *)&khitem->t.htup -
													(char *)kds_hash));
	}
	__syncthreads();
	/* enough room exists on this depth? */
	if (write_pos[depth] + get_local_size() <= kgjoin->pstack_nrooms)
		return depth;
	return depth+1;
}

#define PSTACK_DEPTH(d)							\
	((d) >= 0 && (d) <= kgjoin->num_rels		\
	 ? (pstack_base + pstack_nrooms * ((d) * ((d) + 1)) / 2) : NULL)

/*
 * gpujoin_main
 */
KERNEL_FUNCTION(void)
gpujoin_main(kern_gpujoin *kgjoin,
			 kern_multirels *kmrels,
			 kern_data_store *kds_src,
			 kern_data_store *kds_dst)
{
	kern_parambuf  *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_context	kcxt;
	cl_int			depth;
	cl_int			index;
	cl_uint			outer_unit_sz;
	cl_uint			pstack_nrooms;
	cl_uint		   *pstack_base;
	cl_uint			l_state[GPUJOIN_MAX_DEPTH+1];
	cl_bool			matched[GPUJOIN_MAX_DEPTH+1];

	INIT_KERNEL_CONTEXT(&kcxt, gpujoin_main, kparams);
	assert(kds_src->format == KDS_FORMAT_ROW ||
		   kds_src->format == KDS_FORMAT_BLOCK);
	assert(kds_dst->format == KDS_FORMAT_ROW);

	/* setup private variables */
	outer_unit_sz = (kds_src->format == KDS_FORMAT_ROW
					 ? get_local_size()
					 : KERN_DATA_STORE_PARTSZ(kds_src));
	pstack_nrooms = kgjoin->pstack_nrooms;
	pstack_base = (cl_uint *)((char *)kgjoin + kgjoin->pstack_offset)
		+ get_global_index() * pstack_nrooms * ((GPUJOIN_MAX_DEPTH+1) *
												(GPUJOIN_MAX_DEPTH+2)) / 2;
	memset(l_state, 0, sizeof(l_state));
	memset(matched, 0, sizeof(matched));

	/* setup crc32 table */
	for (index = get_local_id();
		 index < lengthof(pg_crc32_table);
		 index += get_local_size())
		pg_crc32_table[index] = kmrels->pg_crc32_table[index];
	__syncthreads();

	/* setup per-depth context */
	memset(l_state, 0, sizeof(l_state));
	memset(matched, 0, sizeof(matched));
	if (get_local_id() == 0)
	{
		src_read_pos = UINT_MAX;
		stat_source_nitems = 0;
		memset(stat_nitems, 0, sizeof(stat_nitems));
		memset(read_pos, 0, sizeof(read_pos));
		memset(write_pos, 0, sizeof(write_pos));
		scan_done = false;
		base_depth = 0;
	}
	__syncthreads();

	/* main logic of GpuJoin */
	depth = 0;
	while (depth >= 0)
	{
		if (depth == 0)
		{
			/* LOAD FROM KDS_SRC (ROW/BLOCK) */
			depth = gpujoin_load_source(&kcxt,
										kgjoin,
										kds_src,
										outer_unit_sz,
										PSTACK_DEPTH(depth),
										l_state);
		}
		else if (depth > kgjoin->num_rels)
		{
			/* PROJECTION (ROW) */
			assert(depth == kmrels->nrels + 1);
			depth = gpujoin_projection_row(&kcxt,
										   kgjoin,
										   kmrels,
										   kds_src,
										   kds_dst,
										   PSTACK_DEPTH(kgjoin->num_rels),
										   l_state,
										   matched);
		}
		else if (kmrels->chunks[depth-1].is_nestloop)
		{
			/* NEST-LOOP */
			depth = gpujoin_exec_nestloop(&kcxt,
										  kgjoin,
										  kmrels,
										  kds_src,
										  depth,
										  PSTACK_DEPTH(depth-1),
										  PSTACK_DEPTH(depth),
										  l_state,
										  matched);
		}
		else
		{
			/* HASH-JOIN */
			depth = gpujoin_exec_hashjoin(&kcxt,
										  kgjoin,
										  kmrels,
										  kds_src,
										  depth,
										  PSTACK_DEPTH(depth-1),
										  PSTACK_DEPTH(depth),
										  l_state,
										  matched);
		}
		__syncthreads();
	}
	/* write out statistics */
	if (get_local_id() == 0)
	{
		atomicAdd(&kgjoin->source_nitems, stat_source_nitems);
		atomicAdd(&kgjoin->outer_nitems, stat_nitems[0]);
		for (index=0; index < GPUJOIN_MAX_DEPTH; index++)
			atomicAdd(&kgjoin->jscale[index].stat_nitems,
					  stat_nitems[index+1]);
	}
	__syncthreads();
	kern_writeback_error_status(&kgjoin->kerror, kcxt.e);
}

/*
 * gpujoin_collocate_outer_join_map
 *
 * it merges the result of other GPU devices and CPU fallback
 */
KERNEL_FUNCTION(void)
gpujoin_colocate_outer_join_map(kern_multirels *kmrels,
								cl_uint num_devices)
{
	size_t		nrooms = kmrels->ojmaps_length / sizeof(cl_uint);
	cl_uint	   *ojmaps = (cl_uint *)((char *)kmrels + kmrels->kmrels_length);
	cl_uint	   *destmap = ojmaps + kmrels->cuda_dindex * nrooms;
	cl_uint		i, map = 0;

	if (get_global_id() < nrooms)
	{
		for (i=0; i <= num_devices; i++)
		{
			map |= ojmaps[get_global_id()];
			ojmaps += nrooms;
		}
		destmap[get_global_id()] = map;
	}
}

/*
 * gpujoin_right_outer
 */
KERNEL_FUNCTION(void)
gpujoin_right_outer(kern_gpujoin *kgjoin,
					kern_multirels *kmrels,
					cl_int outer_depth,
					kern_data_store *kds_dst)
{
	kern_parambuf  *kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	kern_context	kcxt;
	cl_int			depth;
	cl_int			index;
	cl_uint			pstack_nrooms;
	cl_uint		   *pstack_base;
	cl_uint			l_state[GPUJOIN_MAX_DEPTH+1];
	cl_bool			matched[GPUJOIN_MAX_DEPTH+1];

	INIT_KERNEL_CONTEXT(&kcxt, gpujoin_right_outer, kparams);
	assert(kds_dst->format == KDS_FORMAT_ROW);
	assert(KERN_MULTIRELS_RIGHT_OUTER_JOIN(kmrels, outer_depth));

	/* setup private variables */
	pstack_nrooms = kgjoin->pstack_nrooms;
	pstack_base = (cl_uint *)((char *)kgjoin + kgjoin->pstack_offset)
		+ get_global_index() * pstack_nrooms * ((GPUJOIN_MAX_DEPTH+1) *
												(GPUJOIN_MAX_DEPTH+2)) / 2;
	/* setup crc32 table */
	for (index = get_local_id();
		 index < lengthof(pg_crc32_table);
		 index += get_local_size())
		pg_crc32_table[index] = kmrels->pg_crc32_table[index];
	__syncthreads();

	/* setup per-depth context */
	memset(l_state, 0, sizeof(l_state));
	memset(matched, 0, sizeof(matched));
	if (get_local_id() == 0)
	{
		src_read_pos = UINT_MAX;
		stat_source_nitems = 0;
		memset(stat_nitems, 0, sizeof(stat_nitems));
		memset(read_pos, 0, sizeof(read_pos));
		memset(write_pos, 0, sizeof(write_pos));
		scan_done = false;
		base_depth = outer_depth;
	}
	__syncthreads();

	/* main logic of GpuJoin */
	depth = outer_depth;
	while (depth >= outer_depth)
	{
		if (depth == outer_depth)
		{
			/* makes RIGHT OUTER combinations using OUTER JOIN map */
			depth = gpujoin_load_outer(&kcxt,
									   kgjoin,
									   kmrels,
									   outer_depth,
									   PSTACK_DEPTH(outer_depth),
									   l_state);
		}
		else if (depth > kgjoin->num_rels)
		{
			/* PROJECTION (ROW) */
			assert(depth == kmrels->nrels + 1);
			depth = gpujoin_projection_row(&kcxt,
										   kgjoin,
										   kmrels,
										   NULL,
										   kds_dst,
										   PSTACK_DEPTH(kgjoin->num_rels),
										   l_state,
										   matched);
		}
		else if (kmrels->chunks[depth-1].is_nestloop)
		{
			/* NEST-LOOP */
			depth = gpujoin_exec_nestloop(&kcxt,
										  kgjoin,
										  kmrels,
										  NULL,
										  depth,
										  PSTACK_DEPTH(depth-1),
										  PSTACK_DEPTH(depth),
										  l_state,
										  matched);
		}
		else
		{
			/* HASH-JOIN */
			depth = gpujoin_exec_hashjoin(&kcxt,
										  kgjoin,
										  kmrels,
										  NULL,
										  depth,
										  PSTACK_DEPTH(depth-1),
										  PSTACK_DEPTH(depth),
										  l_state,
										  matched);
		}
		__syncthreads();
	}
	/* write out statistics */
	if (get_local_id() == 0)
	{
		assert(stat_source_nitems == 0);
		assert(stat_nitems[0] == 0);
		for (index = outer_depth; index <= GPUJOIN_MAX_DEPTH; index++)
		{
			atomicAdd(&kgjoin->jscale[index-1].stat_nitems,
					  stat_nitems[index]);
		}
	}
	__syncthreads();
	kern_writeback_error_status(&kgjoin->kerror, kcxt.e);
}

#endif	/* __CUDACC__ */
#endif	/* CUDA_GPUJOIN_H */
