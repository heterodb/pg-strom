/*
 * cuda_gpujoin.cu
 *
 * GPU accelerated parallel relations join based on hash-join or
 * nested-loop logic.
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "cuda_common.h"
#include "cuda_gpujoin.h"

/*
 * static shared variables
 */
static __shared__ cl_bool	scan_done;
static __shared__ cl_int	base_depth;
static __shared__ cl_uint	src_read_pos;
static __shared__ cl_uint	dst_base_index;
static __shared__ size_t	dst_base_usage;
extern __shared__ cl_uint	wip_count[0];	/* [GPUJOIN_MAX_DEPTH+1] items */
extern __shared__ cl_uint	read_pos[0];	/* [GPUJOIN_MAX_DEPTH+1] items */
extern __shared__ cl_uint	write_pos[0];	/* [GPUJOIN_MAX_DEPTH+1] items */
extern __shared__ cl_uint	temp_pos[0];	/* [GPUJOIN_MAX_DEPTH+1] items */
extern __shared__ cl_uint	gist_pos[0];	/* [(GPUJOIN_MAX_DEPTH+1)*32] items */
static __shared__ cl_uint	stat_source_nitems;
extern __shared__ cl_uint	stat_nitems[0];	/* [GPUJOIN_MAX_DEPTH+1] items */
extern __shared__ cl_uint	stat_nitems2[0]; /* [GPUJOIN_MAX_DEPTH+1] items */

/*
 * gpujoin_suspend_context
 */
STATIC_FUNCTION(void)
gpujoin_suspend_context(kern_gpujoin *kgjoin,
						cl_int depth, cl_uint *l_state, cl_bool *matched)
{
	gpujoinSuspendContext *sb;
	cl_int		i, max_depth = kgjoin->num_rels;

	sb = KERN_GPUJOIN_SUSPEND_CONTEXT(kgjoin, get_group_id());
	if (get_local_id() == 0)
	{
		sb->depth = depth;
		sb->scan_done = scan_done;
		sb->src_read_pos = src_read_pos;
		sb->stat_source_nitems = stat_source_nitems;
	}

	for (i=get_local_id(); i <= max_depth; i+=get_local_size())
	{
		sb->pd[i].wip_count = wip_count[i];
		sb->pd[i].read_pos = read_pos[i];
		sb->pd[i].write_pos = write_pos[i];
		sb->pd[i].temp_pos = temp_pos[i];
		memcpy(sb->pd[i].gist_pos, gist_pos + i * MAXWARPS_PER_BLOCK,
			   sizeof(cl_uint) * MAXWARPS_PER_BLOCK);
		sb->pd[i].stat_nitems = stat_nitems[i];
		sb->pd[i].stat_nitems2 = stat_nitems2[i];
	}

	for (i=0; i <= max_depth; i++)
	{
		sb->pd[i].l_state[get_local_id()] = l_state[i];
		sb->pd[i].matched[get_local_id()] = matched[i];
	}
	/* tells host-code GPU kernel needs to be resumed */
	if (get_local_id() == 0)
		atomicAdd(&kgjoin->suspend_count, 1);
	__syncthreads();
}

/*
 * gpujoin_resume_context
 */
STATIC_FUNCTION(cl_int)
gpujoin_resume_context(kern_gpujoin *kgjoin,
					   cl_uint *l_state, cl_bool *matched)
{
	gpujoinSuspendContext *sb;
	cl_int		i, max_depth = kgjoin->num_rels;

	sb = KERN_GPUJOIN_SUSPEND_CONTEXT(kgjoin, get_group_id());
	if (get_local_id() == 0)
	{
		scan_done = sb->scan_done;
		src_read_pos = sb->src_read_pos;
		stat_source_nitems = sb->stat_source_nitems;
	}

	for (i=get_local_id(); i <= max_depth; i+=get_local_size())
	{
		wip_count[i] = sb->pd[i].wip_count;
		read_pos[i] = sb->pd[i].read_pos;
		write_pos[i] = sb->pd[i].write_pos;
		temp_pos[i] = sb->pd[i].temp_pos;
		memcpy(gist_pos + i * MAXWARPS_PER_BLOCK, sb->pd[i].gist_pos,
			   sizeof(cl_uint) * MAXWARPS_PER_BLOCK);
		stat_nitems[i] = sb->pd[i].stat_nitems;
		stat_nitems2[i] = sb->pd[i].stat_nitems2;
	}

	for (i=0; i <= max_depth; i++)
	{
		l_state[i] = sb->pd[i].l_state[get_local_id()];
		matched[i] = sb->pd[i].matched[get_local_id()];
	}
	return sb->depth;
}

/*
 * gpujoin_rewind_stack
 */
STATIC_INLINE(cl_int)
gpujoin_rewind_stack(kern_gpujoin *kgjoin, cl_int depth,
					 cl_uint *l_state, cl_bool *matched)
{
	cl_int		max_depth = kgjoin->num_rels;
	static __shared__ cl_int	__depth;

	assert(depth >= base_depth && depth <= max_depth);
	__syncthreads();
	if (get_local_id() == 0)
	{
		__depth = depth;
		for (;;)
		{
			/*
			 * At the time of rewind, all the upper tuples (outer combinations
			 * from the standpoint of deeper depth) are already processed.
			 * So, we can safely rewind the read/write index of this depth.
			 */
			read_pos[__depth] = 0;
			write_pos[__depth] = 0;

			/*
			 * If any of outer combinations are in progress to find out
			 * matching inner tuple, we have to resume the task, prior
			 * to the increment of read pointer.
			 */
			if (wip_count[__depth] > 0)
				break;
			if (__depth == base_depth ||
				read_pos[__depth-1] < write_pos[__depth-1])
				break;
			__depth--;
		}
	}
	__syncthreads();
	depth = __depth;
	if (depth < max_depth)
	{
		memset(l_state + depth + 1, 0,
			   sizeof(cl_uint) * (max_depth - depth));
		memset(matched + depth + 1, 0,
			   sizeof(cl_bool) * (max_depth - depth));
	}
	if (scan_done && depth == base_depth)
		return -1;
	return depth;
}

/*
 * gpujoin_load_source
 */
STATIC_FUNCTION(cl_int)
gpujoin_load_source(kern_context *kcxt,
					kern_gpujoin *kgjoin,
					kern_data_store *kds_src,
					kern_data_extra *kds_extra,
					cl_uint *wr_stack,
					cl_uint *l_state)
{
	cl_uint		t_offset = UINT_MAX;
	cl_bool		visible = false;
	cl_uint		count;
	cl_uint		wr_index;

	/* extract a HeapTupleHeader */
	if (kds_src->format == KDS_FORMAT_ROW)
	{
		kern_tupitem   *tupitem;
		cl_uint			row_index;

		/* fetch next window */
		if (get_local_id() == 0)
			src_read_pos = atomicAdd(&kgjoin->src_read_pos,
									 get_local_size());
		__syncthreads();
		row_index = src_read_pos + get_local_id();

		if (row_index < __ldg(&kds_src->nitems))
		{
			tupitem = KERN_DATA_STORE_TUPITEM(kds_src, row_index);
			t_offset = __kds_packed((char *)&tupitem->htup -
									(char *)kds_src);
			visible = gpujoin_quals_eval(kcxt,
										 kds_src,
										 &tupitem->htup.t_ctid,
										 &tupitem->htup);
		}
		assert(wip_count[0] == 0);
	}
	else if (kds_src->format == KDS_FORMAT_BLOCK)
	{
		cl_uint		part_sz = KERN_DATA_STORE_PARTSZ(kds_src);
		cl_uint		n_parts = get_local_size() / part_sz;
		cl_uint		part_id;
		cl_uint		line_no;
		cl_uint		n_lines;
		cl_uint		loops = l_state[0]++;

		/* fetch next window, if needed */
		if (loops == 0 && get_local_id() == 0)
			src_read_pos = atomicAdd(&kgjoin->src_read_pos, n_parts);
		__syncthreads();
		part_id = src_read_pos + get_local_id() / part_sz;
		line_no = get_local_id() % part_sz + loops * part_sz + 1;

		if (part_id < __ldg(&kds_src->nitems) &&
			get_local_id() < part_sz * n_parts)
		{
			PageHeaderData *pg_page;
			BlockNumber		block_nr;
			ItemPointerData	t_self;
			HeapTupleHeaderData *htup;

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

					visible = gpujoin_quals_eval(kcxt,
												 kds_src,
												 &t_self,
												 htup);
				}
			}
		}
	}
	else if (kds_src->format == KDS_FORMAT_ARROW)
	{
		cl_uint			row_index;

		/* fetch next window */
		if (get_local_id() == 0)
			src_read_pos = atomicAdd(&kgjoin->src_read_pos,
									 get_local_size());
		__syncthreads();
		row_index = src_read_pos + get_local_id();

		if (row_index < __ldg(&kds_src->nitems))
		{
			t_offset = row_index + 1;
			visible = gpujoin_quals_eval_arrow(kcxt,
											   kds_src,
											   row_index);
		}
		assert(wip_count[0] == 0);
	}
	else if (kds_src->format == KDS_FORMAT_COLUMN)
	{
		cl_uint			row_index;

		/* fetch next window */
		if (get_local_id() == 0)
			src_read_pos = atomicAdd(&kgjoin->src_read_pos,
									 get_local_size());
		__syncthreads();

		row_index = src_read_pos + get_local_id();
		if (row_index < kds_src->nitems &&
			kern_check_visibility_column(kcxt, kds_src, row_index))
		{
			t_offset = row_index + 1;
			visible = gpujoin_quals_eval_column(kcxt,
												kds_src,
												kds_extra,
												row_index);
		}
		assert(wip_count[0] == 0);
	}
	else
	{
		STROM_ELOG(kcxt, "unsupported KDS format");
	}	
	/* error checks */
	if (__syncthreads_count(kcxt->errcode) > 0)
		return -1;
	/* statistics */
	count = __syncthreads_count(t_offset != UINT_MAX);
	if (get_local_id() == 0)
	{
		if (__ldg(&kds_src->format) == KDS_FORMAT_BLOCK)
			wip_count[0] = count;
		stat_source_nitems += count;
	}

	/* store the source tuple if visible */
	wr_index = pgstromStairlikeBinaryCount(visible, &count);
	if (count > 0)
	{
		wr_index += write_pos[0];
		__syncthreads();
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
		if (write_pos[0] + get_local_size() > GPUJOIN_PSEUDO_STACK_NROOMS)
			return 1;
		__syncthreads();
	}
	else
	{
		/* no tuples we could fetch */
		assert(write_pos[0] + get_local_size() <= GPUJOIN_PSEUDO_STACK_NROOMS);
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
			cl_int		max_depth = kgjoin->num_rels;

			for (cl_int depth=1; depth <= max_depth; depth++)
			{
				if (temp_pos[depth] > 0)
					return depth;
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
		t_offset = __kds_packed((char *)&tupitem->htup -
								(char *)kds_in);
		htup = &tupitem->htup;
	}
	wr_index = write_pos[outer_depth];
	wr_index += pgstromStairlikeBinaryCount(htup != NULL, &count);
	__syncthreads();
	if (count > 0)
	{
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
			cl_int		max_depth = kgjoin->num_rels;

			for (cl_int depth=outer_depth + 1; depth <= max_depth; depth++)
			{
				if (read_pos[depth] < write_pos[depth])
					return depth+1;
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
					   kern_data_extra *kds_extra,
					   kern_data_store *kds_dst,
					   cl_uint *rd_stack,
					   cl_uint *l_state,
					   cl_bool *matched)
{
	cl_uint		nrels = kgjoin->num_rels;
	cl_uint		read_index;
	cl_uint		dest_index;
	size_t		dest_offset;
	cl_uint		count;
	cl_uint		nvalids;
	cl_uint		required;
	cl_char	   *tup_dclass;
	Datum	   *tup_values;
	cl_int		needs_suspend = 0;

	/* sanity checks */
	assert(rd_stack != NULL);

	/* Any more result rows to be written? */
	if (read_pos[nrels] >= write_pos[nrels])
		return gpujoin_rewind_stack(kgjoin, nrels, l_state, matched);

	/* Allocation of tup_dclass/values */
	tup_dclass = (cl_char *)
		kern_context_alloc(kcxt, sizeof(cl_char) * kds_dst->ncols);
	tup_values = (Datum *)
		kern_context_alloc(kcxt, sizeof(Datum) * kds_dst->ncols);
	if (!tup_dclass || !tup_values)
		STROM_EREPORT(kcxt, ERRCODE_OUT_OF_MEMORY, "out of memory");
	if (__syncthreads_count(kcxt->errcode) > 0)
		return -1;		/* bailout GpuJoin */

	/* pick up combinations from the pseudo-stack */
	nvalids = Min(write_pos[nrels] - read_pos[nrels],
				  get_local_size());
	read_index = read_pos[nrels] + get_local_id();
	__syncthreads();

	/* step.1 - compute length of the result tuple to be written */
	if (read_index < write_pos[nrels])
	{
		rd_stack += read_index * (nrels + 1);

		gpujoin_projection(kcxt,
						   kds_src,
						   kds_extra,
						   kmrels,
						   rd_stack,
						   kds_dst,
						   tup_dclass,
						   tup_values,
						   NULL);
		required = MAXALIGN(offsetof(kern_tupitem, htup) +
							compute_heaptuple_size(kcxt,
												   kds_dst,
												   tup_dclass,
												   tup_values));
	}
	else
		required = 0;

	if (__syncthreads_count(kcxt->errcode) > 0)
		return -1;		/* bailout */

	/* step.2 - increments nitems/usage of the kds_dst */
	dest_offset = pgstromStairlikeSum(required, &count);
	assert(count > 0);
	if (get_local_id() == 0)
	{
		union {
			struct {
				cl_uint	nitems;
				cl_uint	usage;
			} i;
			cl_ulong	v64;
		} oldval, curval, newval;

		needs_suspend = 0;
		curval.i.nitems	= kds_dst->nitems;
		curval.i.usage	= kds_dst->usage;
		do {
			newval = oldval = curval;
			newval.i.nitems	+= nvalids;
			newval.i.usage	+= __kds_packed(count);

			if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
				STROMALIGN(sizeof(cl_uint) * newval.i.nitems) +
				__kds_unpack(newval.i.usage) > kds_dst->length)
			{
				needs_suspend = 1;
				break;
			}
		} while ((curval.v64 = atomicCAS((cl_ulong *)&kds_dst->nitems,
										 oldval.v64,
										 newval.v64)) != oldval.v64);
		dst_base_index = oldval.i.nitems;
		dst_base_usage = __kds_unpack(oldval.i.usage);
	}
	if (__syncthreads_count(needs_suspend) > 0)
	{
		/* No space left on the kds_dst, suspend the GPU kernel and bailout */
		gpujoin_suspend_context(kgjoin, nrels+1, l_state, matched);
		return -2;	/* <-- not to update statistics */
	}
	dest_index = dst_base_index + get_local_id();
	dest_offset += dst_base_usage + required;

	/* step.3 - write out HeapTuple on the destination buffer */
	if (required > 0)
	{
		cl_uint	   *row_index = KERN_DATA_STORE_ROWINDEX(kds_dst);
		kern_tupitem *tupitem = (kern_tupitem *)
			((char *)kds_dst + kds_dst->length - dest_offset);
		form_kern_heaptuple(kcxt,
							tupitem,
							kds_dst,
							NULL,		/* ItemPointerData */
							tup_dclass,
							tup_values);
		tupitem->rowid = dest_index;
		row_index[dest_index] = __kds_packed(kds_dst->length - dest_offset);
	}
	if (__syncthreads_count(kcxt->errcode) > 0)
		return -1;	/* bailout */

	/* step.4 - make advance the read position */
	if (get_local_id() == 0)
		read_pos[nrels] += nvalids;
	return nrels + 1;
}

/* to be defined by gpupreagg.c */
DEVICE_FUNCTION(void)
gpupreagg_projection_slot(kern_context *kcxt_gpreagg,
						  cl_char *src_dclass,
						  Datum   *src_values,
						  cl_char *dst_dclass,
						  Datum   *dst_values);

/*
 * gpujoin_projection_slot
 */
STATIC_FUNCTION(cl_int)
gpujoin_projection_slot(kern_context *kcxt,
						kern_parambuf *kparams_gpreagg,
						kern_gpujoin *kgjoin,
						kern_multirels *kmrels,
						kern_data_store *kds_src,
						kern_data_extra *kds_extra,
						kern_data_store *kds_dst,
						cl_uint *rd_stack,
						cl_uint *l_state,
						cl_bool *matched)
{
	kern_parambuf *kparams_saved = kcxt->kparams;
	cl_uint		nrels = kgjoin->num_rels;
	cl_uint		read_index;
	cl_uint		dest_index;
	size_t		dest_offset;
	cl_uint		count;
	cl_uint		nvalids;
	cl_bool		tup_is_valid = false;
	cl_char	   *tup_dclass = NULL;
	Datum	   *tup_values = NULL;
	cl_uint	   *tup_extras = NULL;
	cl_uint		extra_sz = 0;
	cl_int		needs_suspend = 0;

	/* sanity checks */
	assert(rd_stack != NULL);

	/* Any more result rows to be written? */
	if (read_pos[nrels] >= write_pos[nrels])
		return gpujoin_rewind_stack(kgjoin, nrels, l_state, matched);

	/* Allocation of tup_dclass/values/extra */
	tup_dclass = (cl_char *)
		kern_context_alloc(kcxt, sizeof(cl_char) * kds_dst->ncols);
	tup_values = (Datum *)
		kern_context_alloc(kcxt, sizeof(Datum) * kds_dst->ncols);
	tup_extras = (cl_uint *)
		kern_context_alloc(kcxt, sizeof(cl_uint) * kds_dst->ncols);
	if (!tup_dclass || !tup_values || !tup_extras)
		STROM_EREPORT(kcxt, ERRCODE_OUT_OF_MEMORY, "out of memory");
	if (__syncthreads_count(kcxt->errcode) > 0)
		return -1;		/* bailout GpuJoin */

	/* pick up combinations from the pseudo-stack */
	nvalids = Min(write_pos[nrels] - read_pos[nrels],
				  get_local_size());
	read_index = read_pos[nrels] + get_local_id();
	__syncthreads();

	/* step.1 - projection by GpuJoin */
	if (read_index < write_pos[nrels])
	{
		rd_stack += read_index * (nrels + 1);

		extra_sz = gpujoin_projection(kcxt,
									  kds_src,
									  kds_extra,
									  kmrels,
									  rd_stack,
									  kds_dst,
									  tup_dclass,
									  tup_values,
									  tup_extras);
		tup_is_valid = true;
	}

	/* step.2 - increments nitems/usage of the kds_dst */
	dest_offset = pgstromStairlikeSum(extra_sz, &count);
	if (get_local_id() == 0)
	{
		union {
			struct {
				cl_uint nitems;
				cl_uint usage;
			} i;
			cl_ulong	v64;
		} oldval, curval, newval;

		needs_suspend = 0;
		curval.i.nitems = kds_dst->nitems;
        curval.i.usage  = kds_dst->usage;
		do {
			newval = oldval = curval;
			newval.i.nitems += nvalids;
			newval.i.usage  += __kds_packed(count);

			if (KERN_DATA_STORE_SLOT_LENGTH(kds_dst, newval.i.nitems) +
				__kds_unpack(newval.i.usage) > kds_dst->length)
			{
				needs_suspend = 1;
				break;
			}
		} while ((curval.v64 = atomicCAS((cl_ulong *)&kds_dst->nitems,
										 oldval.v64,
										 newval.v64)) != oldval.v64);
		dst_base_index = oldval.i.nitems;
		dst_base_usage = __kds_unpack(oldval.i.usage);
	}
	if (__syncthreads_count(needs_suspend) > 0)
	{
		/* No space left on the kds_dst, suspend the GPU kernel and bailout */
		gpujoin_suspend_context(kgjoin, nrels+1, l_state, matched);
		return -2;	/* <-- not to update statistics */
	}
	dest_index = dst_base_index + get_local_id();
	dest_offset += dst_base_usage + extra_sz;

	/* step.3 - projection by GpuPreAgg on the destination buffer */
	if (tup_is_valid)
	{
		cl_char	   *dst_dclass = KERN_DATA_STORE_DCLASS(kds_dst, dest_index);
		Datum	   *dst_values = KERN_DATA_STORE_VALUES(kds_dst, dest_index);

		/*
		 * Fixup pointers, if it points out of kds_src/kmrels because these
		 * variables must be visible to the next GpuPreAgg kernel.
		 */
		if (extra_sz > 0)
		{
			char   *dpos = (char *)kds_dst + kds_dst->length - dest_offset;
			char   *addr;
			cl_int	extra_sum = 0;
			cl_int	len;

			for (int j=0; j < kds_dst->ncols; j++)
			{
				len = tup_extras[j];
				if (len == 0)
					continue;
				addr = DatumGetPointer(tup_values[j]);
				memcpy(dpos, addr, len);
				tup_values[j] = PointerGetDatum(dpos);
				dpos += MAXALIGN(len);
				extra_sum += MAXALIGN(len);
			}
			assert(extra_sz == extra_sum);
		}
		/*
		 * Initial projection by GpuPreAgg
		 *
		 * This code block is generated by gpupreagg.c; that may reference
		 * const/parameters of GpuPreAgg, not GpuJoin. So, we temporarily
		 * switch kparams of the current context.
		 */
		kcxt->kparams = kparams_gpreagg;
		gpupreagg_projection_slot(kcxt,
								  tup_dclass,
								  tup_values,
								  dst_dclass,
								  dst_values);
		kcxt->kparams = kparams_saved;
	}
	if (__syncthreads_count(kcxt->errcode) > 0)
		return -1;	/* bailout */

	/* step.4 - make advance the read position */
	if (get_local_id() == 0)
		read_pos[nrels] += nvalids; //get_local_size();
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
					  kern_data_extra *kds_extra,
					  cl_int depth,
					  cl_uint *rd_stack,
					  cl_uint *wr_stack,
					  cl_uint *l_state,
					  cl_bool *matched)
{
	kern_data_store *kds_in = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	cl_bool		   *oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth);
	kern_tupitem   *tupitem = NULL;
	cl_int			max_depth = kgjoin->num_rels;
	cl_uint			x_unitsz;
	cl_uint			y_unitsz;
	cl_uint			x_index;	/* outer index */
	cl_uint			y_index;	/* inner index */
	cl_uint			wr_index;
	cl_uint			count;
	cl_bool			result = false;
	__shared__ cl_bool matched_sync[MAXTHREADS_PER_BLOCK];

	assert(kds_in->format == KDS_FORMAT_ROW);
	assert(depth >= 1 && depth <= max_depth);
	if (read_pos[depth-1] >= write_pos[depth-1])
	{
		/*
		 * When this depth has enough room (even if all the threads generate
		 * join combinations on the next try), upper depth may be able to
		 * generate more outer tuples; which shall be used to input for the
		 * next depth.
		 * It is mostly valuable to run many combinations on the next depth.
		 */
		assert(wip_count[depth] == 0);
		if (write_pos[depth] + get_local_size() <= GPUJOIN_PSEUDO_STACK_NROOMS)
		{
			cl_int	__depth = gpujoin_rewind_stack(kgjoin, depth-1,
												   l_state, matched);
			if (__depth >= base_depth)
				return __depth;
		}
		/* elsewhere, dive into the deeper depth or projection */
		return depth + 1;
	}
	__syncthreads();
	x_unitsz = Min(write_pos[depth-1], get_local_size());
	y_unitsz = get_local_size() / x_unitsz;

	x_index = get_local_id() % x_unitsz;
	y_index = get_local_id() / x_unitsz;

	if (y_unitsz * l_state[depth] >= kds_in->nitems)
	{
		/*
		 * In case of LEFT OUTER JOIN, we need to check whether the outer
		 * combination had any matched inner tuples, or not.
		 */
		if (KERN_MULTIRELS_LEFT_OUTER_JOIN(kmrels, depth))
		{
			if (get_local_id() < x_unitsz)
				matched_sync[get_local_id()] = false;
			__syncthreads();
			if (matched[depth])
				matched_sync[x_index] = true;
			if (__syncthreads_count(!matched_sync[x_index]) > 0)
			{
				if (y_index == 0 && y_index < y_unitsz)
					result = !matched_sync[x_index];
				else
					result = false;
				/* adjust x_index and rd_stack as usual */
				x_index += read_pos[depth-1];
				assert(x_index < write_pos[depth-1]);
				rd_stack += (x_index * depth);
				/* don't generate LEFT OUTER tuple any more */
				matched[depth] = true;
				goto left_outer;
			}
		}
		l_state[depth] = 0;
		matched[depth] = false;
		if (get_local_id() == 0)
		{
			wip_count[depth] = 0;
			read_pos[depth-1] += x_unitsz;
		}
		return depth;
	}
	x_index += read_pos[depth-1];
	rd_stack += (x_index * depth);
	if (x_index < write_pos[depth-1] && y_index < y_unitsz)
	{
		y_index += y_unitsz * l_state[depth];
		if (y_index < kds_in->nitems)
		{
			tupitem = KERN_DATA_STORE_TUPITEM(kds_in, y_index);

			result = gpujoin_join_quals(kcxt,
										kds_src,
										kds_extra,
										kmrels,
										depth,
										rd_stack,
										&tupitem->htup,
										NULL);
			if (result)
			{
				matched[depth] = true;
				if (oj_map && !oj_map[y_index])
					oj_map[y_index] = true;
			}
		}
	}
	l_state[depth]++;

left_outer:
	wr_index = write_pos[depth];
	wr_index += pgstromStairlikeBinaryCount(result, &count);
	if (get_local_id() == 0)
	{
		wip_count[depth] = get_local_size();
		write_pos[depth] += count;
		stat_nitems[depth] += count;
	}
	wr_stack += wr_index * (depth + 1);
	if (result)
	{
		memcpy(wr_stack, rd_stack, sizeof(cl_uint) * depth);
		wr_stack[depth] = (!tupitem ? 0 : __kds_packed((char *)&tupitem->htup -
													   (char *)kds_in));
	}
	__syncthreads();
	/*
	 * If we have enough room to store the combinations more, execute this
	 * depth one more. Elsewhere, dive into a deeper level to flush results.
	 */
	if (write_pos[depth] + get_local_size() <= GPUJOIN_PSEUDO_STACK_NROOMS)
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
					  kern_data_extra *kds_extra,
					  cl_int depth,
					  cl_uint *rd_stack,
					  cl_uint *wr_stack,
					  cl_uint *l_state,
					  cl_bool *matched)
{
	kern_data_store	   *kds_hash = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	cl_bool			   *oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth);
	kern_hashitem	   *khitem = NULL;
	cl_int				max_depth = kgjoin->num_rels;
	cl_uint				t_offset = UINT_MAX;
	cl_uint				hash_value;
	cl_uint				rd_index;
	cl_uint				wr_index;
	cl_uint				count;
	cl_bool				result;

	assert(kds_hash->format == KDS_FORMAT_HASH);
	assert(depth >= 1 && depth <= max_depth);

	if (__syncthreads_count(l_state[depth] != UINT_MAX) == 0)
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
		 * When this depth has enough room (even if all the threads generate
		 * join combinations on the next try), upper depth may be able to
		 * generate more outer tuples; which shall be used to input for the
		 * next depth.
		 * It is mostly valuable to run many combinations on the next depth.
		 */
		assert(wip_count[depth] == 0);
		if (write_pos[depth] + get_local_size() <= GPUJOIN_PSEUDO_STACK_NROOMS)
		{
			cl_int	__depth = gpujoin_rewind_stack(kgjoin, depth-1,
												   l_state, matched);
			if (__depth >= base_depth)
				return __depth;
		}
		/* elsewhere, dive into the deeper depth or projection */
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
											kds_src,
											kds_extra,
											kmrels,
											depth,
											rd_stack,
											&is_null_keys);
			/* MEMO: NULL-keys will never match to inner-join */
			if (!is_null_keys)
				khitem = KERN_HASH_FIRST_ITEM(kds_hash, hash_value);
			/* rewind the varlena buffer */
			kcxt->vlpos = kcxt->vlbuf;
		}
		else
		{
			/*
			 * MEMO: We must ensure the threads without outer tuple don't
			 * generate any LEFT OUTER results.
			 */
			l_state[depth] = UINT_MAX;
		}
	}
	else if (l_state[depth] != UINT_MAX)
	{
		/* walks on the hash-slot chain */
		khitem = (kern_hashitem *)((char *)kds_hash
								   + __kds_unpack(l_state[depth])
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
									kds_extra,
									kmrels,
									depth,
									rd_stack,
									&khitem->t.htup,
									&joinquals_matched);
		assert(result == joinquals_matched);
		if (joinquals_matched)
		{
			/* No LEFT/FULL JOIN are needed */
			matched[depth] = true;
			/* No RIGHT/FULL JOIN are needed */
			assert(khitem->t.rowid < kds_hash->nitems);
			if (oj_map && !oj_map[khitem->t.rowid])
				oj_map[khitem->t.rowid] = true;
		}
		t_offset = __kds_packed((char *)&khitem->t.htup -
								(char *)kds_hash);
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
	l_state[depth] = t_offset;
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
		wr_stack[depth] = (!khitem ? 0U : t_offset);
	}
	/* count number of threads still in-progress */
	count = __syncthreads_count(khitem != NULL);
	if (get_local_id() == 0)
		wip_count[depth] = count;
	/*
	 * (2019/05/25) We saw a strange behavior on Tesla T4 (CUDA 10.1 with
	 * driver 418.67), but never seen at Pascal/Volta devices.
	 * Even though "write_pos[depth]" is updated by the leader thread above,
	 * then __syncthreads_count() shall synchronize all the local threads,
	 * a part of threads read different value from this variable.
	 * I doubt compiler may have some optimization problem here, therefore,
	 * the code below avoid to reference "write_pos[depth]" directly.
	 * It loads this value to local variable once, then injects a barrier
	 * synchronization explicitly.
	 *
	 * We should check whether the future version of CUDA can fix the problem.
	 */
	wr_index = write_pos[depth];
	__syncthreads();
	if (wr_index + get_local_size() <= GPUJOIN_PSEUDO_STACK_NROOMS)
		return depth;
	return depth+1;
}

//#include "cuda_postgis.h"

/*
 * gpujoin_prep_gistindex
 *
 * MEMO: We must load the entire GiST-index, but part of the leaf items indicate
 * invalid items because a part of inner rows can be filtered out already.
 * So, this kernel function preliminary invalidates these items on the inner
 * preload timing.
 */
KERNEL_FUNCTION(void)
gpujoin_prep_gistindex(kern_multirels *kmrels, int depth)
{
	kern_data_store *kds_hash = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	kern_data_store *kds_gist = KERN_MULTIRELS_GIST_INDEX(kmrels, depth);
	BlockNumber		block_nr;
	OffsetNumber	i, maxoff;

	assert(kds_hash->format == KDS_FORMAT_HASH &&
		   kds_gist->format == KDS_FORMAT_BLOCK);
	assert(depth >= 1 && depth <= kmrels->nrels);

	for (block_nr = get_group_id();
		 block_nr < kds_gist->nrooms;
		 block_nr += get_num_groups())
	{
		PageHeaderData *gist_page;
		ItemIdData	   *lpp;
		IndexTupleData *itup;
		kern_hashitem  *khitem;
		cl_uint			hash, t_off;

		gist_page = KERN_DATA_STORE_BLOCK_PGPAGE(kds_gist, block_nr);
		if (!GistPageIsLeaf(gist_page))
			continue;
		maxoff = PageGetMaxOffsetNumber(gist_page);
		for (i = get_local_id(); i < maxoff; i += get_local_size())
		{
			lpp = PageGetItemId(gist_page, i+1);
			if (ItemIdIsDead(lpp))
				continue;
			itup = (IndexTupleData *)PageGetItem(gist_page, lpp);

			/* lookup kds_hash */
			hash = pg_hash_any((cl_uchar *)&itup->t_tid,
							   sizeof(ItemPointerData));
			for (khitem = KERN_HASH_FIRST_ITEM(kds_hash, hash);
				 khitem != NULL;
				 khitem = KERN_HASH_NEXT_ITEM(kds_hash, khitem))
			{
				if (ItemPointerEquals(&khitem->t.htup.t_ctid, &itup->t_tid))
				{
					t_off = __kds_packed((char *)&khitem->t.htup -
										 (char *)kds_hash);
					itup->t_tid.ip_blkid.bi_hi = (t_off >> 16);
					itup->t_tid.ip_blkid.bi_lo = (t_off & 0x0000ffffU);
					itup->t_tid.ip_posid = USHRT_MAX;
					break;
				}
			}
			/* invalidate this leaf item, if not exist on kds_hash */
			if (!khitem)
				lpp->lp_flags = LP_DEAD;
		}
	}
}

/*
 * gpujoin_gist_getnext
 */
STATIC_INLINE(ItemPointerData *)
gpujoin_gist_getnext(kern_context *kcxt,
					 kern_gpujoin *kgjoin,
					 cl_int depth,
					 kern_data_store *kds_gist,
					 void *gist_keys,
					 cl_uint *p_item_offset)
{
	PageHeaderData *gist_base = KERN_DATA_STORE_BLOCK_PGPAGE(kds_gist, 0);
	PageHeaderData *gist_page;
	cl_char		   *vlpos_saved = kcxt->vlpos;
	OffsetNumber	start;
	OffsetNumber	index;
	OffsetNumber	maxoff;
	ItemIdData	   *lpp = NULL;
	IndexTupleData *itup = NULL;
	cl_bool			rv = false;

	assert(kds_gist->format == KDS_FORMAT_BLOCK);

	/*
	 * Setup starting point of GiST-index lookup
	 */
	if (*p_item_offset == UINT_MAX)
	{
		/* this warp already reached to the end */
		return NULL;
	}
	else if (*p_item_offset == 0)
	{
		/* walk on GiST index from the root page */
		start = FirstOffsetNumber + LaneId();
		gist_page = KERN_DATA_STORE_BLOCK_PGPAGE(kds_gist, GIST_ROOT_BLKNO);
		assert(gist_page->pd_parent_blkno == InvalidBlockNumber &&
			   gist_page->pd_parent_item  == InvalidOffsetNumber);
	}
	else
	{
		/* walk on GiST index from the next item */
		PageHeaderData *gist_base = KERN_DATA_STORE_BLOCK_PGPAGE(kds_gist, 0);
		size_t		off;

		assert(*p_item_offset < kds_gist->length);
		lpp = (ItemIdData *)((char *)kds_gist + *p_item_offset);
		off = (((char *)lpp - (char *)gist_base) & (BLCKSZ - 1));
		gist_page = (PageHeaderData *)((char *)lpp - off);
		start = (lpp - gist_page->pd_linp) + 1 + warpSize;
	}
restart:
	assert((((char *)gist_page - (char *)gist_base) & (BLCKSZ - 1)) == 0);

	if (GistPageIsDeleted(gist_page))
		maxoff = InvalidOffsetNumber;	/* skip any entries */
	else
		maxoff = PageGetMaxOffsetNumber(gist_page);

	rv = false;
	for (index=start; index <= maxoff; index += warpSize)
	{
		lpp = PageGetItemId(gist_page, index);
		if (ItemIdIsDead(lpp))
			continue;
		itup = (IndexTupleData *) PageGetItem(gist_page, lpp);

		kcxt->vlpos = vlpos_saved;		/* rewind */
		rv = gpujoin_gist_index_quals(kcxt, depth,
									  kds_gist, gist_page,
									  itup, gist_keys);
		if (rv)
			break;
	}
	kcxt->vlpos = vlpos_saved;		/* rewind */

	assert(__activemask() == ~0U);
	if (__any_sync(__activemask(), rv))
	{
		/* By here, one or more threads meet the matched entry */
		if (!GistPageIsLeaf(gist_page))
		{
			/* dive into deeper tree node */
			BlockNumber		blkno_curr;
			BlockNumber		blkno_next;
			PageHeaderData *gist_next;
			OffsetNumber	least_index = (rv ? index : UINT_MAX);
			OffsetNumber	buddy_index;

			for (int mask=1; mask <= 16; mask *= 2)
			{
				buddy_index = __shfl_xor_sync(__activemask(), least_index, mask);
				least_index = Min(least_index, buddy_index);
			}
			__syncwarp(~0U);
			assert(least_index <= maxoff);

			lpp = PageGetItemId(gist_page, least_index);
			itup = (IndexTupleData *) PageGetItem(gist_page, lpp);
			blkno_curr = ((char *)gist_page - (char *)gist_base) / BLCKSZ;
			blkno_next = ((BlockNumber)itup->t_tid.ip_blkid.bi_hi << 16 |
						  (BlockNumber)itup->t_tid.ip_blkid.bi_lo);
			assert(blkno_next < kds_gist->nrooms);
			gist_next = KERN_DATA_STORE_BLOCK_PGPAGE(kds_gist, blkno_next);
			assert(gist_next->pd_parent_blkno == blkno_curr &&
				   gist_next->pd_parent_item  == least_index);
			gist_page = gist_next;
			start = FirstOffsetNumber + LaneId();
			goto restart;
		}

		/* this is matched */
		if (rv)
		{
			assert((char *)lpp >= (char *)gist_page &&
				   (char *)lpp <  (char *)gist_page + BLCKSZ);
			*p_item_offset = (cl_uint)((char *)lpp - (char *)kds_gist);

			return &itup->t_tid;
		}

		/*
		 * this is not matched - ensure the next call skips the main loop
		 * above, we set next offset of the 'maxoff' onto the p_item_offset.
		 */
		lpp = PageGetItemId(gist_page, maxoff+1);
		*p_item_offset = (cl_uint)((char *)lpp - (char *)kds_gist);

		return NULL;
	}

	/*
	 * By here, nobody meet any entries in this page
	 */
	if (gist_page != gist_base)
	{
		/* pop up to the parent */
		BlockNumber		blkno_next = gist_page->pd_parent_blkno;

		assert(blkno_next < kds_gist->nrooms);
		start = gist_page->pd_parent_item + 1 + LaneId();
		gist_page = KERN_DATA_STORE_BLOCK_PGPAGE(kds_gist, blkno_next);
		goto restart;
	}
	/* cannot pop up from the root page */
	assert(gist_page->pd_parent_blkno == InvalidBlockNumber &&
		   gist_page->pd_parent_item  == InvalidOffsetNumber);
	*p_item_offset = UINT_MAX;

	return NULL;
}

/*
 * gpujoin_exec_gistindex
 */
STATIC_FUNCTION(cl_int)
gpujoin_exec_gistindex(kern_context *kcxt,
					   kern_gpujoin *kgjoin,
					   kern_multirels *kmrels,
					   kern_data_store *kds_src,
					   kern_data_extra *kds_extra,
					   cl_int depth,
					   cl_uint *__rd_stack_base,
					   cl_uint *__wr_stack_base,
					   cl_uint *l_state,
					   cl_bool *matched)
{
	kern_data_store *kds_hash = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
	kern_data_store *kds_gist = KERN_MULTIRELS_GIST_INDEX(kmrels, depth);
	cl_bool		   *oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(kmrels, depth);
	cl_uint		   *wr_stack;
	cl_uint		   *temp_stack;
	cl_uint			rd_index;
	cl_uint			wr_index;
	cl_uint			temp_index;
	cl_uint			count;
	void		   *gist_keys;
	cl_char		   *vlpos_saved_1 = kcxt->vlpos;

	assert(kds_hash->format == KDS_FORMAT_HASH);
	assert(depth >= 1 && depth <= kgjoin->num_rels);

	if (__syncthreads_count(l_state[depth] != UINT_MAX &&
							l_state[depth] != 0) == 0 &&
		read_pos[depth-1] >= write_pos[depth-1])
	{
		if (write_pos[depth] + get_local_size() <= GPUJOIN_PSEUDO_STACK_NROOMS)
		{
			cl_int	__depth = gpujoin_rewind_stack(kgjoin, depth-1,
												   l_state, matched);
			if (__depth >= base_depth)
				return __depth;
		}
		/* flush if temporary index search results still remain */
		if (scan_done && temp_pos[depth] > 0)
			goto bailout;
		/* elsewhere, dive into the deeper depth or projection */
		return depth + 1;
	}
	__syncthreads();

reload:
	kcxt->vlpos = vlpos_saved_1;	/* rewind */
	assert(__activemask() == ~0U);
	if (__all_sync(__activemask(), l_state[depth] == UINT_MAX) ||
		__all_sync(__activemask(), l_state[depth] == 0))
	{
		/*
		 * all the threads in warp reached in the tail of GiST-index tree, so move to
		 * the next index key.
		 */
		if (LaneId() == 0)
		{
			rd_index = atomicAdd(&read_pos[depth-1], 1);
			gist_pos[depth * MAXWARPS_PER_BLOCK + get_local_id() / warpSize] = rd_index;
		}
		__syncwarp(~0U);
		rd_index = __shfl_sync(__activemask(), rd_index, 0);
		l_state[depth] = 0;
	}
	else
	{
		/* resume the index-key */
		rd_index = gist_pos[depth * MAXWARPS_PER_BLOCK + get_local_id() / warpSize];
	}
	/* threads in a warp must load exactly same index-key */
	assert(rd_index == __shfl_sync(__activemask(), rd_index, 0));

	if (rd_index < write_pos[depth-1])
	{
		cl_uint	   *rd_stack = __rd_stack_base + (rd_index * depth);
		cl_char	   *vlpos_saved_2;

		gist_keys = gpujoin_gist_load_keys(kcxt,
										   kmrels,
										   kds_src,
										   kds_extra,
										   depth,
										   rd_stack);
		assert(__activemask() == ~0U);
		if (__any_sync(__activemask(), kcxt->errcode != 0))
			goto bailout;	/* error */
		assert(gist_keys != NULL);

		/*
		 * MEMO: Cost to run gpujoin_gist_getnext highly depends on the key value.
		 * If key never matches any bounding-box, gpujoin_gist_getnext() returns
		 * immediately. If key matches some entries, thus walks down into the leaf
		 * of R-tree, it takes longer time than the above misshit cases.
		 * In case when individual warps have various execution time, in general,
		 * we should not put __syncthreads() because the warps that returned
		 * immediately from the gpujoin_gist_getnext() are blocked until completion
		 * of someone's R-tree index search.
		 * So, we don't put any __syncthreads() in the loop below. If a warp finished
		 * gpujoin_gist_getnext() very early, it can reload another index-key for
		 * the next search during the GiST-index search by the other warps/threads.
		 * If usage of temp_stack[] exceeds get_local_size(), all the warps move to
		 * the second phase to run gpujoin_join_quals(), because it means we can
		 * utilize all the core to evaluate Join quals in parallel; that is the most
		 * efficient way to run.
		 */
		vlpos_saved_2 = kcxt->vlpos;
		do {
			ItemPointerData *t_ctid;
			cl_uint		mask;
			cl_uint		t_off;
			cl_uint		l_next = l_state[depth];

			t_ctid = gpujoin_gist_getnext(kcxt,
										  kgjoin,
										  depth,
										  kds_gist,
										  gist_keys,
										  &l_next);
			assert(__activemask() == ~0U);
			if (__any_sync(__activemask(), kcxt->errcode != 0))
				goto bailout;	/* error */
			
			mask = __ballot_sync(__activemask(), t_ctid != NULL);
			count = __popc(mask);
			if (LaneId() == 0)
				temp_index = atomicAdd(&temp_pos[depth], count);
			__syncwarp(~0U);
			temp_index = __shfl_sync(__activemask(), temp_index, 0);

			if (temp_index + count > GPUJOIN_PSEUDO_STACK_NROOMS)
				goto bailout;	/* urgent flush; cannot write out all the results */
			temp_index += __popc(mask & ((1U << LaneId()) - 1));

			if (t_ctid)
			{
				assert(t_ctid->ip_posid == USHRT_MAX);
				t_off = (((cl_uint)t_ctid->ip_blkid.bi_hi << 16) |
						 ((cl_uint)t_ctid->ip_blkid.bi_lo));
				assert(temp_index < GPUJOIN_PSEUDO_STACK_NROOMS);
				temp_stack = __wr_stack_base +
					(depth+1) * (GPUJOIN_PSEUDO_STACK_NROOMS + temp_index);
				memcpy(temp_stack, rd_stack, sizeof(cl_uint) * depth);
				temp_stack[depth] = t_off;
				assert(__kds_unpack(t_off) < kds_hash->length);
			}

			if (LaneId() == 0)
				atomicAdd(&stat_nitems2[depth], count);
			__syncwarp(~0U);
			l_state[depth] = l_next;
			kcxt->vlpos = vlpos_saved_2;	/* rewind */
			assert(__activemask() == ~0U);
		} while (__any_sync(__activemask(), l_state[depth] != UINT_MAX));
		/* try to reload the next index-key, if temp_stack[] still has space. */
		assert(__activemask() == ~0U);
		if (__shfl_sync(__activemask(), temp_pos[depth], 0) < get_local_size())
			goto reload;
	}
	else
	{
		l_state[depth] = UINT_MAX;
	}
bailout:
	/* error checks */
	if (__syncthreads_count(kcxt->errcode != 0) > 0)
		return -1;

	if (temp_pos[depth] >= (scan_done ? 1 : get_local_size()))
	{
		temp_stack = NULL;
		if (get_local_id() < temp_pos[depth])
		{
			kern_tupitem   *tupitem;
			cl_bool			joinquals_matched = false;

			temp_stack = __wr_stack_base +
				(depth+1) * (GPUJOIN_PSEUDO_STACK_NROOMS + get_local_id());
			tupitem = (kern_tupitem *)((char *)kds_hash
									   + __kds_unpack(temp_stack[depth])
									   - offsetof(kern_tupitem, htup));
			assert((char *)tupitem < (char *)kds_hash + kds_hash->length);
			/* check join quals */
			if (gpujoin_join_quals(kcxt,
								   kds_src,
								   kds_extra,
								   kmrels,
								   depth,
								   temp_stack,
								   &tupitem->htup,
								   &joinquals_matched))
			{
				assert(joinquals_matched);
				/* No RIGHT JOIN are needed */
				assert(tupitem->rowid < kds_hash->nitems);
				if (oj_map && !oj_map[tupitem->rowid])
					oj_map[tupitem->rowid] = true;
			}
			else
			{
				temp_stack = NULL;
			}
		}

		/* write out the result */
		wr_index = write_pos[depth];
		wr_index += pgstromStairlikeBinaryCount(temp_stack != NULL, &count);
		if (get_local_id() == 0)
		{
			write_pos[depth] += count;
			stat_nitems[depth] += count;
		}
		wr_stack = __wr_stack_base + (depth+1) * wr_index;
		if (temp_stack)
			memcpy(wr_stack, temp_stack, sizeof(cl_uint) * (depth+1));
		__syncthreads();

		/* rewind the temp stack */
		if (get_local_id() == 0)
		{
			if (get_local_size() < temp_pos[depth])
			{
				cl_uint		remain = temp_pos[depth] - get_local_size();
				
				temp_stack = __wr_stack_base + (depth+1) * GPUJOIN_PSEUDO_STACK_NROOMS;
				memcpy(temp_stack,
					   temp_stack + (depth+1) * get_local_size(),
					   sizeof(cl_uint) * (depth+1) * remain);
				temp_pos[depth] -= get_local_size();
			}
			else
			{
				temp_pos[depth] = 0;
			}
		}
	}
	/* count number of threads still in-progress */
	count = __syncthreads_count(l_state[depth] != UINT_MAX &&
								l_state[depth] != 0);
	if (get_local_id() == 0)
		wip_count[depth] = count;

	/* see comment in gpujoin_exec_hashjoin */
	wr_index = write_pos[depth];
	__syncthreads();
	if (wr_index + get_local_size() <= GPUJOIN_PSEUDO_STACK_NROOMS)
		return depth;
	return depth+1;	
}

#define PSTACK_DEPTH(d)											\
	((d) >= 0 && (d) <= kgjoin->num_rels						\
	 ? (cl_uint *)((char *)pstack + pstack->ps_headsz +			\
				   get_group_id() * pstack->ps_unitsz +			\
				   pstack->ps_offset[(d)])						\
	 : NULL)

/*
 * gpujoin_main
 */
DEVICE_FUNCTION(void)
gpujoin_main(kern_context *kcxt,
			 kern_gpujoin *kgjoin,
			 kern_multirels *kmrels,
			 kern_data_store *kds_src,
			 kern_data_extra *kds_extra,
			 kern_data_store *kds_dst,
			 kern_parambuf *kparams_gpreagg, /* only if combined GpuJoin */
			 cl_uint *l_state,
			 cl_bool *matched)
{
	gpujoinPseudoStack *pstack = kgjoin->pstack;
	cl_int			max_depth = kgjoin->num_rels;
	cl_int			depth;
	__shared__ cl_int depth_thread0 __attribute__((unused));

	assert(kds_src->format == KDS_FORMAT_ROW ||
		   kds_src->format == KDS_FORMAT_BLOCK ||
		   kds_src->format == KDS_FORMAT_ARROW ||
		   kds_src->format == KDS_FORMAT_COLUMN);
	assert((kds_dst->format == KDS_FORMAT_ROW  && kparams_gpreagg == NULL) ||
		   (kds_dst->format == KDS_FORMAT_SLOT && kparams_gpreagg != NULL));

	/* init per-depth context */
	if (get_local_id() == 0)
	{
		src_read_pos = UINT_MAX;
		stat_source_nitems = 0;
		memset(stat_nitems, 0, sizeof(cl_uint) * (max_depth+1));
		memset(stat_nitems2, 0, sizeof(cl_uint) * (max_depth+1));
		memset(wip_count, 0, sizeof(cl_uint) * (max_depth+1));
		memset(read_pos, 0, sizeof(cl_uint) * (max_depth+1));
		memset(write_pos, 0, sizeof(cl_uint) * (max_depth+1));
		memset(temp_pos, 0, sizeof(cl_uint) * (max_depth+1));
		memset(gist_pos, 0, sizeof(cl_uint) * (max_depth+1) * MAXWARPS_PER_BLOCK);
		scan_done = false;
		base_depth = 0;
	}
	/* resume the per-depth context, if any */
	if (kgjoin->resume_context)
		depth = gpujoin_resume_context(kgjoin, l_state, matched);
	else
		depth = 0;
	__syncthreads();
	
	/* main logic of GpuJoin */
	while (depth >= 0)
	{
		/* rewind the varlena buffer */
		kcxt->vlpos = kcxt->vlbuf;
		if (depth == 0)
		{
			/* LOAD FROM KDS_SRC (ROW/BLOCK/ARROW) */
			depth = gpujoin_load_source(kcxt,
										kgjoin,
										kds_src,
										kds_extra,
										PSTACK_DEPTH(depth),
										l_state);
		}
		else if (depth > max_depth)
		{
			assert(depth == kmrels->nrels + 1);
			if (kds_dst->format == KDS_FORMAT_ROW)
			{
				/* PROJECTION (ROW) */
				depth = gpujoin_projection_row(kcxt,
											   kgjoin,
											   kmrels,
											   kds_src,
											   kds_extra,
											   kds_dst,
											   PSTACK_DEPTH(kgjoin->num_rels),
											   l_state,
											   matched);
			}
			else
			{
				/* PROJECTION (SLOT) */
				depth = gpujoin_projection_slot(kcxt,
												kparams_gpreagg,
												kgjoin,
												kmrels,
												kds_src,
												kds_extra,
												kds_dst,
												PSTACK_DEPTH(kgjoin->num_rels),
												l_state,
												matched);
			}
		}
		else if (kmrels->chunks[depth-1].is_nestloop)
		{
			/* NEST-LOOP */
			depth = gpujoin_exec_nestloop(kcxt,
										  kgjoin,
										  kmrels,
										  kds_src,
										  kds_extra,
										  depth,
										  PSTACK_DEPTH(depth-1),
										  PSTACK_DEPTH(depth),
										  l_state,
										  matched);
		}
		else if (kmrels->chunks[depth-1].gist_offset != 0)
		{
			/* GiST-INDEX */
			depth = gpujoin_exec_gistindex(kcxt,
										   kgjoin,
										   kmrels,
										   kds_src,
										   kds_extra,
										   depth,
										   PSTACK_DEPTH(depth-1),
										   PSTACK_DEPTH(depth),
										   l_state,
										   matched);
		}
		else
		{
			/* HASH-JOIN */
			depth = gpujoin_exec_hashjoin(kcxt,
										  kgjoin,
										  kmrels,
										  kds_src,
										  kds_extra,
										  depth,
										  PSTACK_DEPTH(depth-1),
										  PSTACK_DEPTH(depth),
										  l_state,
										  matched);
		}
		if (get_local_id() == 0)
			depth_thread0 = depth;
		if (__syncthreads_count(kcxt->errcode) > 0)
			return;
		assert(depth_thread0 == depth);
	}

	/* update statistics only if normal exit */
	if (depth == -1 && get_local_id() == 0)
	{
		gpujoinSuspendContext *sb
			= KERN_GPUJOIN_SUSPEND_CONTEXT(kgjoin, get_group_id());
		sb->depth = -1;		/* no more suspend/resume! */

		atomicAdd(&kgjoin->source_nitems, stat_source_nitems);
		atomicAdd(&kgjoin->outer_nitems, stat_nitems[0]);
		for (int i=0; i <= max_depth; i++)
		{
			atomicAdd(&kgjoin->stat[i].nitems, stat_nitems[i+1]);
			atomicAdd(&kgjoin->stat[i].nitems2, stat_nitems2[i+1]);
		}
	}
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
	cl_uint		i, j, map;

	for (i = get_global_id();
		 i < nrooms;
		 i += get_global_size())
	{
		map = 0;
		for (j = 0; j <= num_devices; j++)
		{
			map |= ojmaps[i];
			ojmaps += nrooms;
		}
		destmap[i] = map;
	}
}

/*
 * gpujoin_right_outer
 */
DEVICE_FUNCTION(void)
gpujoin_right_outer(kern_context *kcxt,
					kern_gpujoin *kgjoin,
					kern_multirels *kmrels,
					cl_int outer_depth,
					kern_data_store *kds_dst,
					kern_parambuf *kparams_gpreagg,
					cl_uint *l_state,
					cl_bool *matched)
{
	gpujoinPseudoStack *pstack = kgjoin->pstack;
	cl_int			max_depth = kgjoin->num_rels;
	cl_int			depth;
	__shared__ cl_int depth_thread0 __attribute__((unused));

	assert(KERN_MULTIRELS_RIGHT_OUTER_JOIN(kmrels, outer_depth));
	assert((kds_dst->format == KDS_FORMAT_ROW  && kparams_gpreagg == NULL) ||
		   (kds_dst->format == KDS_FORMAT_SLOT && kparams_gpreagg != NULL));

	/* setup per-depth context */
	memset(l_state, 0, sizeof(l_state));
	memset(matched, 0, sizeof(matched));
	if (get_local_id() == 0)
	{
		src_read_pos = UINT_MAX;
		stat_source_nitems = 0;
		memset(stat_nitems, 0, sizeof(cl_uint) * (max_depth+1));
		memset(stat_nitems2, 0, sizeof(cl_uint) * (max_depth+1));
		memset(wip_count, 0, sizeof(cl_uint) * (max_depth+1));
		memset(read_pos, 0, sizeof(cl_uint) * (max_depth+1));
		memset(write_pos, 0, sizeof(cl_uint) * (max_depth+1));
		memset(temp_pos, 0, sizeof(cl_uint) * (max_depth+1));
		memset(gist_pos, 0, sizeof(cl_uint) * (max_depth+1) * MAXWARPS_PER_BLOCK);
		scan_done = false;
		base_depth = outer_depth;
	}
	/* resume the per-depth context, if any */
	if (kgjoin->resume_context)
		depth = gpujoin_resume_context(kgjoin, l_state, matched);
	else
		depth = outer_depth;
	__syncthreads();

	/* main logic of GpuJoin */
	while (depth >= outer_depth)
	{
		/* rewind the varlena buffer */
		kcxt->vlpos = kcxt->vlbuf;
		if (depth == outer_depth)
		{
			/* makes RIGHT OUTER combinations using OUTER JOIN map */
			depth = gpujoin_load_outer(kcxt,
									   kgjoin,
									   kmrels,
									   outer_depth,
									   PSTACK_DEPTH(outer_depth),
									   l_state);
		}
		else if (depth > max_depth)
		{
			assert(depth == kmrels->nrels + 1);
			if (kds_dst->format == KDS_FORMAT_ROW)
			{
				/* PROJECTION (ROW) */
				depth = gpujoin_projection_row(kcxt,
											   kgjoin,
											   kmrels,
											   NULL,
											   NULL,
											   kds_dst,
											   PSTACK_DEPTH(kgjoin->num_rels),
											   l_state,
											   matched);
			}
			else
			{
				/* PROJECTION (SLOT) */
				depth = gpujoin_projection_slot(kcxt,
												kparams_gpreagg,
												kgjoin,
												kmrels,
												NULL,
												NULL,
												kds_dst,
												PSTACK_DEPTH(kgjoin->num_rels),
												l_state,
												matched);
			}
		}
		else if (kmrels->chunks[depth-1].is_nestloop)
		{
			/* NEST-LOOP */
			depth = gpujoin_exec_nestloop(kcxt,
										  kgjoin,
										  kmrels,
										  NULL,
										  NULL,
										  depth,
										  PSTACK_DEPTH(depth-1),
										  PSTACK_DEPTH(depth),
										  l_state,
										  matched);
		}
		else if (kmrels->chunks[depth-1].gist_offset)
		{
			/* GiST-INDEX */
			depth = gpujoin_exec_gistindex(kcxt,
										   kgjoin,
										   kmrels,
										   NULL,
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
			depth = gpujoin_exec_hashjoin(kcxt,
										  kgjoin,
										  kmrels,
										  NULL,
										  NULL,
										  depth,
										  PSTACK_DEPTH(depth-1),
										  PSTACK_DEPTH(depth),
										  l_state,
										  matched);
		}
		if (get_local_id() == 0)
			depth_thread0 = depth;
		if (__syncthreads_count(kcxt->errcode) > 0)
			return;
		assert(depth == depth_thread0);
	}

	/* write out statistics */
	if (get_local_id() == 0)
	{
		gpujoinSuspendContext *sb
			= KERN_GPUJOIN_SUSPEND_CONTEXT(kgjoin, get_group_id());
		sb->depth = -1;		/* no more suspend/resume! */

		assert(stat_source_nitems == 0);
		assert(stat_nitems[0] == 0);
		for (int i=outer_depth; i <= max_depth; i++)
		{
			atomicAdd(&kgjoin->stat[i-1].nitems,  stat_nitems[i]);
			atomicAdd(&kgjoin->stat[i-1].nitems2, stat_nitems2[i]);
		}
	}
	__syncthreads();
}
