/*
 * cuda_gpujoin.cu
 *
 * GPU accelerated parallel relations join based on hash-join or
 * nested-loop logic.
 * --
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
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
static __shared__ cl_uint	stat_source_nitems;
extern __shared__ cl_uint	stat_nitems[0];	/* [GPUJOIN_MAX_DEPTH+1] items */

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
		sb->pd[i].stat_nitems = stat_nitems[i];
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
		stat_nitems[i] = sb->pd[i].stat_nitems;
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
					cl_uint *wr_stack,
					cl_uint *l_state)
{
	cl_uint		t_offset = UINT_MAX;
	cl_bool		visible = false;
	cl_uint		count;
	cl_uint		wr_index;

	/* extract a HeapTupleHeader */
	if (__ldg(&kds_src->format) == KDS_FORMAT_ROW)
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
										 &tupitem->t_self,
										 &tupitem->htup);
		}
		assert(wip_count[0] == 0);
	}
	else if (__ldg(&kds_src->format) == KDS_FORMAT_BLOCK)
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
	else
	{
		assert(__ldg(&kds_src->format) == KDS_FORMAT_ARROW);
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
			cl_int		max_depth = kgjoin->num_rels;

			for (cl_int depth=1; depth <= max_depth; depth++)
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

		row_index[dest_index] = __kds_packed(kds_dst->length - dest_offset);
		form_kern_heaptuple(kcxt,
							tupitem,
							kds_dst,
							NULL,		/* ItemPointerData */
							NULL,		/* HeapTupleHeaderData */
							tup_dclass,
							tup_values);
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
		if (write_pos[depth] + get_local_size() <= kgjoin->pstack_nrooms)
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
		if (write_pos[depth] + get_local_size() <= kgjoin->pstack_nrooms)
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
			assert(khitem->rowid < kds_hash->nitems);
			if (oj_map && !oj_map[khitem->rowid])
				oj_map[khitem->rowid] = true;
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
	if (wr_index + get_local_size() <= kgjoin->pstack_nrooms)
		return depth;
	return depth+1;
}

#define PSTACK_DEPTH(d)							\
	((d) >= 0 && (d) <= kgjoin->num_rels		\
	 ? (pstack_base + pstack_nrooms * ((d) * ((d) + 1)) / 2) : NULL)

/*
 * gpujoin_main
 */
DEVICE_FUNCTION(void)
gpujoin_main(kern_context *kcxt,
			 kern_gpujoin *kgjoin,
			 kern_multirels *kmrels,
			 kern_data_store *kds_src,
			 kern_data_store *kds_dst,
			 kern_parambuf *kparams_gpreagg, /* only if combined GpuJoin */
			 cl_uint *l_state,
			 cl_bool *matched)
{
	cl_int			max_depth = kgjoin->num_rels;
	cl_int			depth;
	cl_int			index;
	cl_uint			pstack_nrooms;
	cl_uint		   *pstack_base;
	__shared__ cl_int depth_thread0 __attribute__((unused));

	assert(__ldg(&kds_src->format) == KDS_FORMAT_ROW ||
		   __ldg(&kds_src->format) == KDS_FORMAT_BLOCK ||
		   __ldg(&kds_src->format) == KDS_FORMAT_ARROW);
	assert((kds_dst->format == KDS_FORMAT_ROW  && kparams_gpreagg == NULL) ||
		   (kds_dst->format == KDS_FORMAT_SLOT && kparams_gpreagg != NULL));

	/* setup private variables */
	pstack_nrooms = kgjoin->pstack_nrooms;
	pstack_base = (cl_uint *)((char *)kgjoin + kgjoin->pstack_offset)
		+ get_group_id() * pstack_nrooms * ((max_depth+1) *
											(max_depth+2)) / 2;
	/* init per-depth context */
	if (get_local_id() == 0)
	{
		src_read_pos = UINT_MAX;
		stat_source_nitems = 0;
		memset(stat_nitems, 0, sizeof(cl_uint) * (max_depth+1));
		memset(wip_count, 0, sizeof(cl_uint) * (max_depth+1));
		memset(read_pos, 0, sizeof(cl_uint) * (max_depth+1));
		memset(write_pos, 0, sizeof(cl_uint) * (max_depth+1));
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
		for (index=0; index <= max_depth; index++)
			atomicAdd(&kgjoin->stat_nitems[index],
					  stat_nitems[index+1]);
	}
	__syncthreads();
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
	cl_int			max_depth = kgjoin->num_rels;
	cl_int			depth;
	cl_int			index;
	cl_uint			pstack_nrooms;
	cl_uint		   *pstack_base;
	__shared__ cl_int depth_thread0 __attribute__((unused));

	assert(KERN_MULTIRELS_RIGHT_OUTER_JOIN(kmrels, outer_depth));
	assert((kds_dst->format == KDS_FORMAT_ROW  && kparams_gpreagg == NULL) ||
		   (kds_dst->format == KDS_FORMAT_SLOT && kparams_gpreagg != NULL));
	/* setup private variables */
	pstack_nrooms = kgjoin->pstack_nrooms;
	pstack_base = (cl_uint *)((char *)kgjoin + kgjoin->pstack_offset)
		+ get_group_id() * pstack_nrooms * ((max_depth+1) *
											(max_depth+2)) / 2;
	/* setup per-depth context */
	memset(l_state, 0, sizeof(l_state));
	memset(matched, 0, sizeof(matched));
	if (get_local_id() == 0)
	{
		src_read_pos = UINT_MAX;
		stat_source_nitems = 0;
		memset(stat_nitems, 0, sizeof(cl_uint) * (max_depth+1));
		memset(wip_count, 0, sizeof(cl_uint) * (max_depth+1));
		memset(read_pos, 0, sizeof(cl_uint) * (max_depth+1));
		memset(write_pos, 0, sizeof(cl_uint) * (max_depth+1));
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
		for (index = outer_depth; index <= max_depth; index++)
		{
			atomicAdd(&kgjoin->stat_nitems[index-1],
					  stat_nitems[index]);
		}
	}
	__syncthreads();
}
