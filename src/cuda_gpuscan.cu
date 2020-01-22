/*
 * libgpuscan.cu
 *
 * GPU implementation of GpuScan
 * ----
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
#include "cuda_gpuscan.h"

/*
 * gpuscan_main_row - GpuScan logic for KDS_FORMAT_ROW
 */
DEVICE_FUNCTION(void)
gpuscan_main_row(kern_context *kcxt,
				 kern_gpuscan *kgpuscan,
				 kern_data_store *kds_src,
				 kern_data_store *kds_dst,
				 bool has_device_projection)
{
	gpuscanSuspendContext *my_suspend
		= KERN_GPUSCAN_SUSPEND_CONTEXT(kgpuscan, get_group_id());
	gpuscanResultIndex *gs_results	__attribute__((unused))
		= KERN_GPUSCAN_RESULT_INDEX(kgpuscan);
	cl_uint		dst_ncols = (kds_dst ? kds_dst->ncols : kds_src->ncols);
	cl_uint		part_index = 0;
	cl_uint		src_index;
	cl_uint		src_base;
	cl_uint		nitems_offset;
	cl_uint		usage_offset	__attribute__((unused));
	cl_uint		total_nitems_in = 0;	/* stat */
	cl_uint		total_nitems_out = 0;	/* stat */
	cl_uint		total_extra_size = 0;	/* stat */
	__shared__ cl_uint	nitems_base;
	__shared__ cl_ulong	usage_base	__attribute__((unused));

	assert(kds_src->format == KDS_FORMAT_ROW);
	assert(!kds_dst || kds_dst->format == KDS_FORMAT_ROW);
	/* quick bailout if any error happen on the prior kernel */
	if (__syncthreads_count(kgpuscan->kerror.errcode) != 0)
		return;
	/* resume kernel from the point where suspended, if any */
	if (kgpuscan->resume_context)
	{
		assert(my_suspend != NULL);
		part_index = my_suspend->part_index;
	}

	for (src_base = get_global_base() + part_index * get_global_size();
		 src_base < kds_src->nitems;
		 src_base += get_global_size(), part_index++)
	{
		kern_tupitem   *tupitem;
		cl_bool			rc;
		cl_uint			nvalids;
		cl_uint			required	__attribute__((unused));
		cl_uint			extra_sz = 0;
		cl_uint			suspend_kernel	__attribute__((unused)) = 0;
		cl_char		   *tup_dclass = NULL;
		Datum		   *tup_values = NULL;

		/* rewind the varlena buffer */
		kcxt->vlpos = kcxt->vlbuf;
		/* Evalidation of the rows by WHERE-clause */
		src_index = src_base + get_local_id();
		if (src_index < kds_src->nitems)
		{
			tupitem = KERN_DATA_STORE_TUPITEM(kds_src, src_index);
			rc = gpuscan_quals_eval(kcxt, kds_src,
									&tupitem->t_self,
									&tupitem->htup);
		}
		else
		{
			tupitem = NULL;
			rc = false;
		}
		/* bailout if any error */
		if (__syncthreads_count(kcxt->errcode) > 0)
			goto out_nostat;

		/* how many rows servived WHERE-clause evaluation? */
		nitems_offset = pgstromStairlikeBinaryCount(tupitem && rc, &nvalids);
		if (nvalids == 0)
			goto skip;

		if (has_device_projection)
		{
			/* extract the source tuple to the private slot, if any */
			if (tupitem && rc)
			{
				kcxt->vlpos = kcxt->vlbuf;	/* rewind */
				tup_dclass = (cl_char *)
					kern_context_alloc(kcxt, sizeof(cl_char) * dst_ncols);
				tup_values = (Datum *)
					kern_context_alloc(kcxt, sizeof(Datum) * dst_ncols);

				if (!tup_dclass || !tup_values)
				{
					required = 0;
					STROM_CPU_FALLBACK(kcxt, ERRCODE_OUT_OF_MEMORY,
									   "out of memory");
				}
				else
				{
					gpuscan_projection_tuple(kcxt,
											 kds_src,
											 &tupitem->htup,
											 &tupitem->t_self,
											 tup_dclass,
											 tup_values);
					required = MAXALIGN(offsetof(kern_tupitem, htup) +
										compute_heaptuple_size(kcxt,
															   kds_dst,
															   tup_dclass,
															   tup_values));
				}
			}
			else
				required = 0;
			/* bailout if any error */
			if (__syncthreads_count(kcxt->errcode) > 0)
				goto out_nostat;

			usage_offset = pgstromStairlikeSum(required, &extra_sz);
			if (get_local_id() == 0)
			{
				union {
					struct {
						cl_uint	nitems;
						cl_uint	usage;
					} i;
					cl_ulong	v64;
				} oldval, curval, newval;

				curval.i.nitems	= kds_dst->nitems;
				curval.i.usage	= kds_dst->usage;
				do {
					newval = oldval = curval;
					newval.i.nitems += nvalids;
					newval.i.usage  += __kds_packed(extra_sz);

					if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
						STROMALIGN(sizeof(cl_uint) * newval.i.nitems) +
						__kds_unpack(newval.i.usage) > kds_dst->length)
					{
						atomicAdd(&kgpuscan->suspend_count, 1);
						suspend_kernel = 1;
						break;
					}
				} while ((curval.v64 = atomicCAS((cl_ulong *)&kds_dst->nitems,
												 oldval.v64,
												 newval.v64)) != oldval.v64);
				nitems_base = oldval.i.nitems;
				usage_base  = __kds_unpack(oldval.i.usage);
			}
			if (__syncthreads_count(suspend_kernel) > 0)
				break;

			/* store the result heap-tuple on destination buffer */
			if (tupitem && rc)
			{
				cl_uint	   *tup_index = KERN_DATA_STORE_ROWINDEX(kds_dst);
				size_t		pos;

				pos = kds_dst->length - (usage_base + usage_offset + required);
				tup_index[nitems_base + nitems_offset] = __kds_packed(pos);
				form_kern_heaptuple(kcxt,
									(kern_tupitem *)((char *)kds_dst + pos),
									kds_dst,
									&tupitem->t_self,
									&tupitem->htup,
									tup_dclass,
									tup_values);
			}
		}
		else
		{
			if (get_local_id() == 0)
				nitems_base = atomicAdd(&gs_results->nitems, nvalids);
			__syncthreads();
			if (tupitem && rc)
			{
				assert(nitems_base + nitems_offset < kds_src->nrooms);
				gs_results->results[nitems_base + nitems_offset]
					= __kds_packed((char *)&tupitem->htup -
								   (char *)kds_src);
			}
		}
	skip:
		/* update statistics */
		if (get_local_id() == 0)
		{
			total_nitems_in  += Min(kds_src->nitems - src_base,
									get_local_size());
			total_nitems_out += nvalids;
			total_extra_size += extra_sz;
		}
	}
	/* write back statistics and error code */
	if (get_local_id() == 0)
	{
		atomicAdd(&kgpuscan->nitems_in,  total_nitems_in);
		atomicAdd(&kgpuscan->nitems_out, total_nitems_out);
		atomicAdd(&kgpuscan->extra_size, total_extra_size);
	}
out_nostat:
	/* suspend the current position (even if normal exit) */
	if (my_suspend && get_local_id() == 0)
	{
		my_suspend->part_index = part_index;
		my_suspend->line_index = 0;
	}
}

/*
 * gpuscan_main_block - GpuScan logic for KDS_FORMAT_BLOCK
 */
DEVICE_FUNCTION(void)
gpuscan_main_block(kern_context *kcxt,
				   kern_gpuscan *kgpuscan,
				   kern_data_store *kds_src,
				   kern_data_store *kds_dst,
				   bool has_device_projection)
{
	gpuscanSuspendContext *my_suspend
		= KERN_GPUSCAN_SUSPEND_CONTEXT(kgpuscan, get_group_id());
	cl_uint		src_nitems = kds_src->nitems;
	cl_uint		dst_ncols = kds_dst->ncols;
	cl_uint		part_sz;
	cl_uint		n_parts;
	cl_uint		nitems_offset;
	cl_uint		usage_offset;
	cl_uint		window_sz;
	cl_uint		part_base;
	cl_uint		part_index = 0;
	cl_uint		line_index = 0;
	cl_uint		total_nitems_in = 0;	/* stat */
	cl_uint		total_nitems_out = 0;	/* stat */
	cl_uint		total_extra_size = 0;	/* stat */
	cl_bool		thread_is_valid = false;
	__shared__ cl_uint	nitems_base;
	__shared__ cl_ulong	usage_base;

	assert(kds_src->format == KDS_FORMAT_BLOCK);
	assert(kds_dst->format == KDS_FORMAT_ROW);
	/* quick bailout if any error happen on the prior kernel */
	if (__syncthreads_count(kgpuscan->kerror.errcode) != 0)
		return;

	part_sz = KERN_DATA_STORE_PARTSZ(kds_src);
	n_parts = get_local_size() / part_sz;
	if (get_global_id() == 0)
		kgpuscan->part_sz = part_sz;
	if (get_local_id() < part_sz * n_parts)
		thread_is_valid = true;
	window_sz = n_parts * get_num_groups();

	/* resume kernel from the point where suspended, if any */
	if (kgpuscan->resume_context)
	{
		part_index = my_suspend->part_index;
		line_index = my_suspend->line_index;
	}
	__syncthreads();

	for (;;)
	{
		cl_uint		part_id;
		cl_uint		line_no;
		cl_uint		n_lines = 0;
		cl_uint		nvalids;
		cl_uint		nitems_real;

		part_base = part_index * window_sz + get_group_id() * n_parts;
		if (part_base >= kds_src->nitems)
			break;
		part_id = get_local_id() / part_sz + part_base;
		line_no = get_local_id() % part_sz + line_index * part_sz;

		do {
			HeapTupleHeaderData *htup = NULL;
			ItemPointerData t_self;
			PageHeaderData *pg_page;
			BlockNumber	block_nr;
			cl_ushort	t_len	__attribute__((unused));
			cl_uint		required;
			cl_uint		extra_sz = 0;
			cl_uint		suspend_kernel = 0;
			cl_bool		rc;
			cl_char	   *tup_dclass = NULL;
			Datum	   *tup_values = NULL;

			/* rewind the varlena buffer */
			kcxt->vlpos = kcxt->vlbuf;

			/* identify the block */
			if (thread_is_valid && part_id < src_nitems)
			{
				pg_page = KERN_DATA_STORE_BLOCK_PGPAGE(kds_src, part_id);
				n_lines = PageGetMaxOffsetNumber(pg_page);
				block_nr = KERN_DATA_STORE_BLOCK_BLCKNR(kds_src, part_id);
				t_self.ip_blkid.bi_hi = block_nr >> 16;
				t_self.ip_blkid.bi_lo = block_nr & 0xffff;
				t_self.ip_posid = line_no + 1;

				if (line_no < n_lines)
				{
					ItemIdData *lpp = PageGetItemId(pg_page, line_no+1);
					if (ItemIdIsNormal(lpp))
						htup = PageGetItem(pg_page, lpp);
					t_len = ItemIdGetLength(lpp);
				}
			}

			/* evaluation of the qualifiers */
			if (htup)
				rc = gpuscan_quals_eval(kcxt, kds_src,
										&t_self,
										htup);
			else
				rc = false;
			/* bailout if any error */
			if (__syncthreads_count(kcxt->errcode) > 0)
				goto out_nostat;

			/* how many rows servived WHERE-clause evaluations? */
			nitems_offset = pgstromStairlikeBinaryCount(htup && rc, &nvalids);
			if (nvalids == 0)
				goto skip;

			/* store the result heap-tuple to destination buffer */
			if (htup && rc)
			{
				if (has_device_projection)
				{
					tup_dclass = (cl_char *)
						kern_context_alloc(kcxt, sizeof(cl_char) * dst_ncols);
					tup_values = (Datum *)
						kern_context_alloc(kcxt, sizeof(Datum) * dst_ncols);

					if (!tup_dclass || !tup_values)
					{
						required = 0;
						STROM_EREPORT(kcxt, ERRCODE_OUT_OF_MEMORY,
									  "out of memory");
					}
					else
					{
						gpuscan_projection_tuple(kcxt,
												 kds_src,
												 htup,
												 &t_self,
												 tup_dclass,
												 tup_values);
						required = MAXALIGN(offsetof(kern_tupitem, htup) +
											compute_heaptuple_size(kcxt,
																   kds_dst,
																   tup_dclass,
																   tup_values));
					}
				}
				else
				{
					/* no projection; just write the source tuple as is */
					required = MAXALIGN(offsetof(kern_tupitem, htup) + t_len);
				}
			}
			else
				required = 0;
			/* bailout if any error */
			if (__syncthreads_count(kcxt->errcode) > 0)
				goto out;

			usage_offset = pgstromStairlikeSum(required, &extra_sz);
			if (get_local_id() == 0)
			{
				union {
					struct {
						cl_uint	nitems;
						cl_uint	usage;
					} i;
					cl_ulong	v64;
				} oldval, curval, newval;

				curval.i.nitems = kds_dst->nitems;
				curval.i.usage  = kds_dst->usage;
				do {
					newval = oldval = curval;
					newval.i.nitems += nvalids;
					newval.i.usage  += __kds_packed(extra_sz);

					if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
						STROMALIGN(sizeof(cl_uint) * newval.i.nitems) +
						__kds_unpack(newval.i.usage) > kds_dst->length)
					{
						atomicAdd(&kgpuscan->suspend_count, 1);
						suspend_kernel = 1;
						break;
					}
				} while ((curval.v64 = atomicCAS((cl_ulong *)&kds_dst->nitems,
												 oldval.v64,
												 newval.v64)) != oldval.v64);
				nitems_base = oldval.i.nitems;
				usage_base  = __kds_unpack(oldval.i.usage);
			}
			if (__syncthreads_count(suspend_kernel) > 0)
				goto out;

			/* store the result heap tuple */
			if (htup && rc)
			{
				cl_uint	   *tup_index = KERN_DATA_STORE_ROWINDEX(kds_dst);
				size_t		pos = (kds_dst->length
								   - (usage_base + usage_offset + required));
				if (has_device_projection)
				{
					form_kern_heaptuple(kcxt,
										(kern_tupitem *)((char *)kds_dst + pos),
										kds_dst,
										&t_self,
										htup,
										tup_dclass,
										tup_values);
				}
				else
				{
					kern_tupitem *tupitem;

					tupitem = (kern_tupitem *)((char *)kds_dst + pos);
					tupitem->t_len = t_len;
					tupitem->t_self = t_self;
					memcpy(&tupitem->htup, htup, t_len);
				}
				tup_index[nitems_base + nitems_offset] = __kds_packed(pos);
			}
		skip:
			/* update statistics */
			pgstromStairlikeBinaryCount(htup != NULL, &nitems_real);
			if (get_local_id() == 0)
			{
				total_nitems_in		+= nitems_real;
				total_nitems_out	+= nvalids;
				total_extra_size	+= extra_sz;
			}

			/*
			 * Move to the next window of the line items, if any.
			 * If no threads in CUDA block wants to continue, exit the loop.
			 */
			line_index++;
			line_no += part_sz;
		} while (__syncthreads_count(thread_is_valid &&
									 line_no < n_lines) > 0);
		/* move to the next window */
		part_index++;
		line_index = 0;
	}
out:
	/* update statistics */
	if (get_local_id() == 0)
	{
		atomicAdd(&kgpuscan->nitems_in,  total_nitems_in);
		atomicAdd(&kgpuscan->nitems_out, total_nitems_out);
		atomicAdd(&kgpuscan->extra_size, total_extra_size);
	}
out_nostat:
	if (get_local_id() == 0)
	{
		my_suspend->part_index = part_index;
		my_suspend->line_index = line_index;
	}
}

/*
 * gpuscan_main_arrow - GpuScan logic for KDS_FORMAT_ARROW
 */
DEVICE_FUNCTION(void)
gpuscan_main_arrow(kern_context *kcxt,
				   kern_gpuscan *kgpuscan,
				   kern_data_store *kds_src,
				   kern_data_store *kds_dst,
				   bool has_device_projection)
{
	gpuscanSuspendContext *my_suspend
		= KERN_GPUSCAN_SUSPEND_CONTEXT(kgpuscan, get_group_id());
	cl_uint		part_index = 0;
	cl_uint		dst_ncols = kds_dst->ncols;
	cl_uint		src_base;
	cl_uint		src_index;
	cl_uint		nitems_offset;
	cl_uint		usage_offset	__attribute__((unused));
	cl_uint		total_nitems_in = 0;	/* stat */
	cl_uint		total_nitems_out = 0;	/* stat */
	cl_uint		total_extra_size = 0;	/* stat */
	__shared__ cl_uint	nitems_base;
	__shared__ cl_ulong	usage_base	__attribute__((unused));

	assert(kds_src->format == KDS_FORMAT_ARROW);
	assert(!kds_dst || kds_dst->format == KDS_FORMAT_ROW);
	/* quick bailout if any error happen on the prior kernel */
	if (__syncthreads_count(kgpuscan->kerror.errcode) != 0)
		return;
	/* resume kernel from the point where suspended, if any */
	if (kgpuscan->resume_context)
	{
		assert(my_suspend != NULL);
		part_index = my_suspend->part_index;
	}

	for (src_base = get_global_base() + part_index * get_global_size();
		 src_base < kds_src->nitems;
		 src_base += get_global_size(), part_index++)
	{
		kern_tupitem   *tupitem		__attribute__((unused));
		cl_bool			rc;
		cl_uint			nvalids;
		cl_uint			required	__attribute__((unused));
		cl_uint			extra_sz = 0;
		cl_uint			suspend_kernel = 0;
		cl_char		   *tup_dclass = NULL;
		Datum		   *tup_values = NULL;

		/* rewind the varlena buffer */
		kcxt->vlpos = kcxt->vlbuf;

		/* Evalidation of the rows by WHERE-clause */
		src_index = src_base + get_local_id();
		if (src_index < kds_src->nitems)
			rc = gpuscan_quals_eval_arrow(kcxt, kds_src, src_index);
		else
			rc = false;
		/* bailout if any error */
		if (__syncthreads_count(kcxt->errcode) > 0)
			goto out_nostat;

		/* how many rows servived WHERE-clause evaluation? */
		nitems_offset = pgstromStairlikeBinaryCount(rc, &nvalids);
		if (nvalids == 0)
			goto skip;

		/*
		 * OK, extract the source columns to form a result row
		 */
		if (rc)
		{
			kcxt->vlpos = kcxt->vlbuf;	/* rewind */
			tup_dclass = (cl_char *)
				kern_context_alloc(kcxt, sizeof(cl_char) * dst_ncols);
			tup_values = (Datum *)
				kern_context_alloc(kcxt, sizeof(Datum) * dst_ncols);

			if (!tup_dclass || !tup_values)
			{
				required = 0;
				STROM_EREPORT(kcxt, ERRCODE_OUT_OF_MEMORY,
							  "out of memory");
			}
			else
			{
				gpuscan_projection_arrow(kcxt,
										 kds_src,
										 src_index,
										 tup_dclass,
										 tup_values);
				required = MAXALIGN(offsetof(kern_tupitem, htup) +
									compute_heaptuple_size(kcxt,
														   kds_dst,
														   tup_dclass,
														   tup_values));
			}
		}
		else
			required = 0;

		usage_offset = pgstromStairlikeSum(required, &extra_sz);
		if (get_local_id() == 0)
		{
			union {
				struct {
					cl_uint	nitems;
					cl_uint	usage;
				} i;
				cl_ulong	v64;
			} oldval, curval, newval;

			curval.i.nitems = kds_dst->nitems;
			curval.i.usage  = kds_dst->usage;
			do {
				newval = oldval = curval;
				newval.i.nitems += nvalids;
				newval.i.usage  += __kds_packed(extra_sz);

				if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
					STROMALIGN(sizeof(cl_uint) * newval.i.nitems) +
					__kds_unpack(newval.i.usage) > kds_dst->length)
				{
					atomicAdd(&kgpuscan->suspend_count, 1);
					suspend_kernel = 1;
					break;
				}
			} while ((curval.v64 = atomicCAS((cl_ulong *)&kds_dst->nitems,
											 oldval.v64,
											 newval.v64)) != oldval.v64);
			nitems_base = oldval.i.nitems;
			usage_base  = __kds_unpack(oldval.i.usage);
		}
		if (__syncthreads_count(suspend_kernel) > 0)
			break;

		/* store the result heap-tuple on destination buffer */
		if (required > 0)
		{
			cl_uint	   *tup_index = KERN_DATA_STORE_ROWINDEX(kds_dst);
			cl_uint		pos;

			pos = kds_dst->length - (usage_base + usage_offset + required);
			tup_index[nitems_base + nitems_offset] = __kds_packed(pos);
			form_kern_heaptuple(kcxt,
								(kern_tupitem *)((char *)kds_dst + pos),
								kds_dst,
								NULL,	/* ItemPointerData */
								NULL,	/* HeapTupleHeaderData */
								tup_dclass,
								tup_values);
		}
		/* bailout if any error */
		if (__syncthreads_count(kcxt->errcode) > 0)
			break;
	skip:
		/* update statistics */
		if (get_local_id() == 0)
		{
			total_nitems_in  += Min(kds_src->nitems - src_base,
									get_local_size());
			total_nitems_out += nvalids;
			total_extra_size += extra_sz;
		}
	}
	/* write back statistics and error code */
	if (get_local_id() == 0)
	{
		atomicAdd(&kgpuscan->nitems_in,  total_nitems_in);
		atomicAdd(&kgpuscan->nitems_out, total_nitems_out);
		atomicAdd(&kgpuscan->extra_size, total_extra_size);
	}
out_nostat:
	/* suspend the current position (even if normal exit) */
	if (my_suspend && get_local_id() == 0)
	{
		my_suspend->part_index = part_index;
		my_suspend->line_index = 0;
	}
}
