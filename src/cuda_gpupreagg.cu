/*
 * cuda_gpupreagg.h
 *
 * Preprocess of aggregate using GPU acceleration, to reduce number of
 * rows to be processed by CPU; including the Sort reduction.
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
#include "cuda_gpupreagg.h"
/*
 * gpupreagg_final_data_move
 *
 * It moves the value from source buffer to destination buffer. If it needs
 * to allocate variable-length buffer, it expands extra area of the final
 * buffer and returns allocated area.
 */
STATIC_FUNCTION(cl_int)
gpupreagg_final_data_move(kern_context *kcxt,
						  kern_data_store *kds_src, cl_uint rowidx_src,
						  kern_data_store *kds_dst, cl_uint rowidx_dst)
{
	Datum	   *src_values = KERN_DATA_STORE_VALUES(kds_src, rowidx_src);
	Datum	   *dst_values = KERN_DATA_STORE_VALUES(kds_dst, rowidx_dst);
	cl_char	   *src_dclass = KERN_DATA_STORE_DCLASS(kds_src, rowidx_src);
	cl_char	   *dst_dclass = KERN_DATA_STORE_DCLASS(kds_dst, rowidx_dst);
	cl_uint		i, ncols = kds_src->ncols;
	cl_int		alloc_size = 0;
	char	   *curr = NULL;
	int			len;

	/* Paranoire checks? */
	assert(kds_src->format == KDS_FORMAT_SLOT &&
		   kds_dst->format == KDS_FORMAT_SLOT);
	assert(kds_src->ncols == kds_dst->ncols);
	assert(rowidx_src < kds_src->nitems);
	assert(rowidx_dst < kds_dst->nitems);

	/* size for allocation */
	for (i=0; i < ncols; i++)
	{
		kern_colmeta   *cmeta = &kds_src->colmeta[i];
		cl_char			dclass = src_dclass[i];

		if (dclass == DATUM_CLASS__NULL)
			continue;

		if (cmeta->attbyval)
			assert(dclass == DATUM_CLASS__NORMAL);
		else if (cmeta->attlen > 0)
		{
			assert(dclass == DATUM_CLASS__NORMAL);
			alloc_size += MAXALIGN(cmeta->attlen);
		}
		else
		{
			assert(cmeta->attlen == -1);
			switch (dclass)
			{
				case DATUM_CLASS__VARLENA:
					len = pg_varlena_datum_length(kcxt, src_values[i]);
					break;
				case DATUM_CLASS__ARRAY:
					len = pg_array_datum_length(kcxt, src_values[i]);
					break;
				case DATUM_CLASS__COMPOSITE:
					len = pg_composite_datum_length(kcxt, src_values[i]);
					break;
				default:
					assert(dclass == DATUM_CLASS__NORMAL);
					len = VARSIZE_ANY(DatumGetPointer(src_values[i]));
					break;
			}
			alloc_size += MAXALIGN(len);
		}
	}
	/* allocate extra buffer by atomic operation */
	if (alloc_size > 0)
	{
		size_t		usage_prev;
		size_t		usage_slot;

		usage_slot = KERN_DATA_STORE_SLOT_LENGTH(kds_dst, rowidx_dst + 1);
		usage_prev = __kds_unpack(atomicAdd(&kds_dst->usage,
											__kds_packed(alloc_size)));
		if (usage_slot + usage_prev + alloc_size >= kds_dst->length)
		{
			STROM_EREPORT(kcxt, ERRCODE_STROM_DATASTORE_NOSPACE,
						  "kds_final has no more space");
			/*
			 * NOTE: Uninitialized dst_values[] for NULL values will lead
			 * a problem around atomic operation, because we designed 
			 * reduction operation that assumes values are correctly
			 * initialized even if it is NULL.
			 */
			for (i=0; i < ncols; i++)
			{
				dst_dclass[i] = DATUM_CLASS__NULL;
				dst_values[i] = src_values[i];
			}
			return -1;
		}
		curr = ((char *)kds_dst + kds_dst->length - (usage_prev + alloc_size));
	}
	/* move the data */
	for (i=0; i < ncols; i++)
	{
		kern_colmeta   *cmeta = &kds_src->colmeta[i];
		cl_char			dclass = src_dclass[i];

		if (dclass == DATUM_CLASS__NULL || cmeta->attbyval)
		{
			dst_dclass[i] = dclass;
			dst_values[i] = src_values[i];
		}
		else if (cmeta->attlen > 0)
		{
			assert(dclass == DATUM_CLASS__NORMAL);
			memcpy(curr, DatumGetPointer(src_values[i]), cmeta->attlen);
			dst_dclass[i] = DATUM_CLASS__NORMAL;
			dst_values[i] = PointerGetDatum(curr);
			curr += MAXALIGN(cmeta->attlen);
		}
		else
		{
			Datum		datum = src_values[i];
			assert(cmeta->attlen == -1);
			switch (dclass)
			{
				case DATUM_CLASS__VARLENA:
					len = pg_varlena_datum_write(kcxt, curr, datum);
                    break;
                case DATUM_CLASS__ARRAY:
					len = pg_array_datum_write(kcxt, curr, datum);
					break;
				case DATUM_CLASS__COMPOSITE:
					len = pg_composite_datum_write(kcxt, curr, datum);
					break;
				default:
					len = VARSIZE_ANY(datum);
					memcpy(curr, DatumGetPointer(datum), len);
					break;
			}
			dst_dclass[i] = DATUM_CLASS__NORMAL;
            dst_values[i] = PointerGetDatum(curr);
			curr += MAXALIGN(len);
		}
	}
	return alloc_size;
}

/*
 * common portion for gpupreagg_setup_*
 */
STATIC_FUNCTION(bool)
gpupreagg_setup_common(kern_context    *kcxt,
					   kern_gpupreagg  *kgpreagg,
					   kern_data_store *kds_src,
					   kern_data_store *kds_slot,
					   cl_uint			nvalids,
					   cl_uint          slot_index,
					   cl_char         *tup_dclass,
					   Datum           *tup_values,
					   cl_int		   *tup_extra)
{
	cl_uint		offset;
	cl_uint		required;
	cl_uint		extra_sz = 0;
	cl_bool		suspend_kernel = false;
	__shared__ cl_uint	nitems_base;
	__shared__ cl_uint	extra_base;

	/*
	 * calculation of the required extra buffer
	 */
	if (slot_index != UINT_MAX)
	{
		if (kds_slot->ncols > 0)
			memset(tup_extra, 0, sizeof(cl_int) * kds_slot->ncols);

		for (int j=0; j < kds_slot->ncols; j++)
		{
			kern_colmeta   *cmeta = &kds_slot->colmeta[j];
			cl_char			dclass = tup_dclass[j];
			cl_char		   *addr;

			if (dclass == DATUM_CLASS__NULL)
				continue;
			if (cmeta->attbyval)
			{
				assert(dclass == DATUM_CLASS__NORMAL);
				continue;
			}
			if (cmeta->attlen > 0)
			{
				assert(dclass == DATUM_CLASS__NORMAL);
				addr = DatumGetPointer(tup_values[j]);
				if (addr <  (char *)kds_src ||
					addr >= (char *)kds_src + kds_src->length)
				{
					tup_extra[j] = cmeta->attlen;
					extra_sz += MAXALIGN(cmeta->attlen);
				}
			}
			else
			{
				/*
				 * NOTE: DATUM_CLASS__* that is not NORMAL only happen when
				 * Var-node references the kds_src buffer which is not
				 * a normal heap-tuple (Apache Arrow). So, it is sufficient
				 * to copy only pg_varlena_t or pg_array_t according to the
				 * datum class. Unlike gpupreagg_final_data_move(), kds_src
				 * buffer shall be valid until reduction steps.
				 */
				assert(cmeta->attlen == -1);
				switch (dclass)
				{
					case DATUM_CLASS__VARLENA:
						tup_extra[j] = sizeof(pg_varlena_t);
						extra_sz += MAXALIGN(sizeof(pg_varlena_t));
						break;
					case DATUM_CLASS__ARRAY:
						tup_extra[j] = sizeof(pg_array_t);
						extra_sz += MAXALIGN(sizeof(pg_array_t));
						break;
					case DATUM_CLASS__COMPOSITE:
						tup_extra[j] = sizeof(pg_composite_t);
						extra_sz += MAXALIGN(sizeof(pg_composite_t));
						break;
					default:
						assert(dclass == DATUM_CLASS__NORMAL);
						addr = DatumGetPointer(tup_values[j]);
						if (addr <  (char *)kds_src ||
							addr >= (char *)kds_src + kds_src->length)
						{
							tup_extra[j] = VARSIZE_ANY(addr);
							extra_sz += MAXALIGN(VARSIZE_ANY(addr));
						}
						break;
				}
			}		
		}
	}

	/*
	 * allocation of extra buffer for indirect/varlena values
	 */
	offset = pgstromStairlikeSum(extra_sz, &required);
	if (get_local_id() == 0)
	{
		union {
			struct {
				cl_uint	nitems;
				cl_uint	usage;
			} i;
			cl_ulong	v64;
		} oldval, curval, newval;

		curval.i.nitems = kds_slot->nitems;
		curval.i.usage  = kds_slot->usage;
		do {
			newval = oldval = curval;
			newval.i.nitems += nvalids;
			newval.i.usage  += __kds_packed(required);
			if (KERN_DATA_STORE_SLOT_LENGTH(kds_slot, newval.i.nitems) +
				__kds_unpack(newval.i.usage) > kds_slot->length)
			{
				suspend_kernel = true;
				atomicAdd(&kgpreagg->suspend_count, 1);
				break;
			}
		} while((curval.v64 = atomicCAS((cl_ulong *)&kds_slot->nitems,
										oldval.v64,
										newval.v64)) != oldval.v64);
		nitems_base = oldval.i.nitems;
		extra_base = __kds_unpack(oldval.i.usage);
	}
	if (__syncthreads_count(suspend_kernel) > 0)
		return false;

	if (slot_index != UINT_MAX)
	{
		assert(slot_index < nvalids);
		slot_index += nitems_base;
		/*
		 * Fixup pointers if needed. Please note that any variables on
		 * kcxt->vlbuf is not visible to other threads.
		 */
		if (extra_sz > 0)
		{
			char	   *extra_pos
				= (char *)kds_slot + kds_slot->length
				- (extra_base + required) + offset;

			for (int j=0; j < kds_slot->ncols; j++)
			{
				if (tup_extra[j] == 0)
					continue;
				memcpy(extra_pos,
					   DatumGetPointer(tup_values[j]),
					   tup_extra[j]);
				tup_values[j] = PointerGetDatum(extra_pos);
				extra_pos += MAXALIGN(tup_extra[j]);
			}
		}
		memcpy(KERN_DATA_STORE_VALUES(kds_slot, slot_index),
			   tup_values, sizeof(Datum) * kds_slot->ncols);
		memcpy(KERN_DATA_STORE_DCLASS(kds_slot, slot_index),
			   tup_dclass, sizeof(cl_char) * kds_slot->ncols);
	}
	return true;
}

/*
 * gpupreagg_setup_row
 */
DEVICE_FUNCTION(void)
gpupreagg_setup_row(kern_context *kcxt,
					kern_gpupreagg *kgpreagg,
					kern_data_store *kds_src,	/* in: KDS_FORMAT_ROW */
					kern_data_store *kds_slot)	/* out: KDS_FORMAT_SLOT */
{
	cl_uint			src_nitems = __ldg(&kds_src->nitems);
	cl_uint			src_base;
	cl_uint			src_index;
	cl_uint			slot_index;
	cl_uint			count;
	cl_uint			nvalids;
	cl_char		   *vlbuf_base;
	cl_char		   *tup_dclass;
	Datum		   *tup_values;
	cl_int		   *tup_extra;
	kern_tupitem   *tupitem;
	gpupreaggSuspendContext *my_suspend;
	cl_bool			rc;

	assert(kds_src->format == KDS_FORMAT_ROW &&
		   kds_slot->format == KDS_FORMAT_SLOT);

	/* resume kernel from the point where suspended, if any */
	my_suspend = KERN_GPUPREAGG_SUSPEND_CONTEXT(kgpreagg, get_group_id());
	if (kgpreagg->resume_context)
		src_base = my_suspend->r.src_base;
	else
		src_base = get_global_base();
	__syncthreads();

	tup_dclass = (cl_char *)
		kern_context_alloc(kcxt, sizeof(cl_char) * kds_slot->ncols);
	tup_values = (Datum *)
		kern_context_alloc(kcxt, sizeof(Datum) * kds_slot->ncols);
	tup_extra  = (cl_int *)
		kern_context_alloc(kcxt, sizeof(cl_int) * kds_slot->ncols);
	if (!tup_dclass || !tup_values || !tup_extra)
		STROM_EREPORT(kcxt, ERRCODE_OUT_OF_MEMORY, "out of memory");
	/* bailout if any errors */
	if (__syncthreads_count(kcxt->errcode) > 0)
		goto skip;
	vlbuf_base = kcxt->vlpos;

	while (src_base < src_nitems)
	{
		kcxt->vlpos = vlbuf_base;		/* rewind */
		src_index = src_base + get_local_id();
		if (src_index < src_nitems)
		{
			tupitem = KERN_DATA_STORE_TUPITEM(kds_src, src_index);
			rc = gpupreagg_quals_eval(kcxt, kds_src,
									  &tupitem->t_self,
									  &tupitem->htup);
			kcxt->vlpos = vlbuf_base;	/* rewind */
		}
		else
		{
			tupitem = NULL;
			rc = false;
		}
		/* bailout if any errors on gpupreagg_quals_eval */
		if (__syncthreads_count(kcxt->errcode) > 0)
			break;
		/* allocation of kds_slot buffer, if any */
		slot_index = pgstromStairlikeBinaryCount(rc, &nvalids);
		if (nvalids > 0)
		{
			if (rc)
			{
				assert(tupitem != NULL);
				gpupreagg_projection_row(kcxt,
										 kds_src,
										 &tupitem->htup,
										 tup_dclass,
										 tup_values);
			}
			/* bailout if any errors */
			if (__syncthreads_count(kcxt->errcode) > 0)
				break;

			if (!gpupreagg_setup_common(kcxt,
										kgpreagg,
										kds_src,
										kds_slot,
										nvalids,
										rc ? slot_index : UINT_MAX,
										tup_dclass,
										tup_values,
										tup_extra))
				break;
		}
		/* update statistics */
		count = __syncthreads_count(tupitem != NULL);
		if (get_local_id() == 0)
		{
			atomicAdd(&kgpreagg->nitems_real, count);
			atomicAdd(&kgpreagg->nitems_filtered, count - nvalids);
		}
		/* move to the next window */
		src_base += get_global_size();
	}
skip:
	/* save the current execution context */
	if (get_local_id() == 0)
		my_suspend->r.src_base = src_base;
}

DEVICE_FUNCTION(void)
gpupreagg_setup_block(kern_context *kcxt,
					  kern_gpupreagg *kgpreagg,
					  kern_data_store *kds_src,
					  kern_data_store *kds_slot)
{
	cl_uint			window_sz;
	cl_uint			part_sz;
	cl_uint			n_parts;
	cl_uint			count;
	cl_uint			part_index = 0;
	cl_uint			line_index = 0;
	cl_bool			thread_is_valid = false;
	cl_char		   *vlbuf_base;
	cl_char		   *tup_dclass;
	Datum		   *tup_values;
	cl_int		   *tup_extra;
	gpupreaggSuspendContext *my_suspend;

	assert(kds_src->format == KDS_FORMAT_BLOCK &&
		   kds_slot->format == KDS_FORMAT_SLOT);

	part_sz = Min((kds_src->nrows_per_block +
				   warpSize-1) & ~(warpSize-1), get_local_size());
	n_parts = get_local_size() / part_sz;
	if (get_local_id() < part_sz * n_parts)
		thread_is_valid = true;
	window_sz = n_parts * get_num_groups();

	/* resume kernel from the point where suspended, if any */
	my_suspend = KERN_GPUPREAGG_SUSPEND_CONTEXT(kgpreagg, get_group_id());
	if (kgpreagg->resume_context)
	{
		part_index = my_suspend->b.part_index;
		line_index = my_suspend->b.line_index;
	}
	__syncthreads();

	tup_dclass = (cl_char *)
		kern_context_alloc(kcxt, sizeof(cl_char) * kds_slot->ncols);
	tup_values = (Datum *)
		kern_context_alloc(kcxt, sizeof(Datum) * kds_slot->ncols);
	tup_extra  = (cl_int *)
		kern_context_alloc(kcxt, sizeof(cl_int) * kds_slot->ncols);
	if (!tup_dclass || !tup_values || !tup_extra)
		STROM_EREPORT(kcxt, ERRCODE_OUT_OF_MEMORY, "out of memory");
	/* bailout if any errors */
	if (__syncthreads_count(kcxt->errcode) > 0)
		goto out;
	vlbuf_base = kcxt->vlpos;

	for (;;)
	{
		cl_uint		part_base;
		cl_uint		part_id;
		cl_uint		line_no;
		cl_uint		n_lines;
		cl_uint		nvalids;
		PageHeaderData *pg_page;
		ItemPointerData	t_self	__attribute__ ((unused));
		BlockNumber		block_nr;

		part_base = part_index * window_sz + get_group_id() * n_parts;
		if (part_base >= kds_src->nitems)
			break;
		part_id = get_local_id() / part_sz + part_base;
		line_no = get_local_id() % part_sz + line_index * part_sz;

		do {
			HeapTupleHeaderData *htup = NULL;
			ItemIdData *curr_lpp = NULL;
			cl_uint		slot_index;
			cl_bool		rc = false;

			kcxt->vlpos = vlbuf_base;		/* rewind */
			if (thread_is_valid && part_id < kds_src->nitems)
			{
				pg_page = KERN_DATA_STORE_BLOCK_PGPAGE(kds_src, part_id);
				n_lines = PageGetMaxOffsetNumber(pg_page);
				block_nr = KERN_DATA_STORE_BLOCK_BLCKNR(kds_src, part_id);
				t_self.ip_blkid.bi_hi = block_nr >> 16;
				t_self.ip_blkid.bi_lo = block_nr & 0xffff;
				t_self.ip_posid = line_no + 1;

				if (line_no < n_lines)
				{
					curr_lpp = PageGetItemId(pg_page, line_no + 1);
					if (ItemIdIsNormal(curr_lpp))
						htup = PageGetItem(pg_page, curr_lpp);
				}
			}
			else
			{
				pg_page = NULL;
				n_lines = 0;
			}

			/* evaluation of the qualifier */
			if (htup)
			{
				rc = gpupreagg_quals_eval(kcxt, kds_src, &t_self, htup);
				kcxt->vlpos = vlbuf_base;		/* rewind */
			}
			/* bailout if any errors on gpupreagg_quals_eval */
			if (__syncthreads_count(kcxt->errcode) > 0)
				goto out;
			/* allocation of the kds_slot buffer */
			slot_index = pgstromStairlikeBinaryCount(rc, &nvalids);
			if (nvalids > 0)
			{
				if (rc)
				{
					gpupreagg_projection_row(kcxt,
											 kds_src,
											 htup,
											 tup_dclass,
											 tup_values);
				}
				/* bailout if any errors */
				if (__syncthreads_count(kcxt->errcode) > 0)
					goto out;

				if (!gpupreagg_setup_common(kcxt,
											kgpreagg,
											kds_src,
											kds_slot,
											nvalids,
											rc ? slot_index : UINT_MAX,
											tup_dclass,
											tup_values,
											tup_extra))
					goto out;
			}
			/* update statistics */
			count = __syncthreads_count(htup != NULL);
			if (get_local_id() == 0)
			{
				atomicAdd(&kgpreagg->nitems_real, count);
				atomicAdd(&kgpreagg->nitems_filtered, count - nvalids);
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
	if (get_local_id() == 0)
	{
		my_suspend->b.part_index = part_index;
		my_suspend->b.line_index = line_index;
	}
}

/*
 * gpupreagg_setup_arrow
 */
DEVICE_FUNCTION(void)
gpupreagg_setup_arrow(kern_context *kcxt,
					  kern_gpupreagg *kgpreagg,
					  kern_data_store *kds_src,	 /* in: KDS_FORMAT_ARROW */
					  kern_data_store *kds_slot) /* out: KDS_FORMAT_SLOT */
{
	cl_uint			src_nitems = __ldg(&kds_src->nitems);
	cl_uint			src_base;
	cl_uint			src_index;
	cl_uint			slot_index;
	cl_uint			count;
	cl_uint			nvalids;
	cl_char		   *vlbuf_base;
	cl_char		   *tup_dclass;
	Datum		   *tup_values;
	cl_int		   *tup_extra;
	gpupreaggSuspendContext *my_suspend;
	cl_bool			rc;

	assert(kds_src->format == KDS_FORMAT_ARROW &&
		   kds_slot->format == KDS_FORMAT_SLOT);

	/* resume kernel from the point where suspended, if any */
	my_suspend = KERN_GPUPREAGG_SUSPEND_CONTEXT(kgpreagg, get_group_id());
	if (kgpreagg->resume_context)
		src_base = my_suspend->c.src_base;
	else
		src_base = get_global_base();
	__syncthreads();

	tup_dclass = (cl_char *)
		kern_context_alloc(kcxt, sizeof(cl_char) * kds_slot->ncols);
	tup_values = (Datum *)
		kern_context_alloc(kcxt, sizeof(Datum) * kds_slot->ncols);
	tup_extra  = (cl_int *)
		kern_context_alloc(kcxt, sizeof(cl_int) * kds_slot->ncols);
	if (!tup_dclass || !tup_values || !tup_extra)
		STROM_EREPORT(kcxt, ERRCODE_OUT_OF_MEMORY, "out of memory");
	/* bailout if any errors */
	if (__syncthreads_count(kcxt->errcode) > 0)
		goto skip;
	vlbuf_base = kcxt->vlpos;

	while (src_base < src_nitems)
	{
		kcxt->vlpos = vlbuf_base;		/* rewind */
		src_index = src_base + get_local_id();
		if (src_index < src_nitems)
		{
			rc = gpupreagg_quals_eval_arrow(kcxt, kds_src, src_index);
			kcxt->vlpos = vlbuf_base;	/* rewind */
		}
		else
		{
			rc = false;
		}
		/* Bailout if any error on gpupreagg_quals_eval_arrow */
		if (__syncthreads_count(kcxt->errcode) > 0)
			break;
		/* allocation of kds_slot buffer, if any */
		slot_index = pgstromStairlikeBinaryCount(rc ? 1 : 0, &nvalids);
		if (nvalids > 0)
		{
			if (rc)
			{
				gpupreagg_projection_arrow(kcxt,
										   kds_src,
										   src_index,
										   tup_dclass,
										   tup_values);
			}
			/* Bailout if any error */
			if (__syncthreads_count(kcxt->errcode) > 0)
				break;
			/* common portion */
			if (!gpupreagg_setup_common(kcxt,
										kgpreagg,
										kds_src,
										kds_slot,
										nvalids,
										rc ? slot_index : UINT_MAX,
										tup_dclass,
										tup_values,
										tup_extra))
				break;
		}
		/* update statistics */
		count = __syncthreads_count(rc);
		if (get_local_id() == 0)
		{
			atomicAdd(&kgpreagg->nitems_real, count);
			atomicAdd(&kgpreagg->nitems_filtered, count - nvalids);
		}
		/* move to the next window */
		src_base += get_global_size();
	}
skip:
	/* save the current execution context */
	if (get_local_id() == 0)
		my_suspend->c.src_base = src_base;
}

/*
 * gpupreagg_nogroup_reduction
 */
DEVICE_FUNCTION(void)
gpupreagg_nogroup_reduction(kern_context *kcxt,
							kern_gpupreagg *kgpreagg,		/* in/out */
							kern_errorbuf *kgjoin_errorbuf,	/* in */
							kern_data_store *kds_slot,		/* in */
							kern_data_store *kds_final,		/* shared out */
							kern_global_hashslot *f_hash)	/* shared out */
{
	varlena		   *kparam_0 = (varlena *)kparam_get_value(kcxt->kparams, 0);
	cl_char		   *attr_is_preagg = (cl_char *)VARDATA(kparam_0);
	cl_uint			nvalids;
	cl_int			i, j;
	cl_bool			is_last_reduction = false;
	cl_char		   *slot_dclass;
	Datum		   *slot_values;
	cl_char		   *row_inval_map;
#define NOGROUP_BLOCK_SZ		4096	/* 36kB of shared memory consumption */
	__shared__ cl_bool	l_dclass[NOGROUP_BLOCK_SZ];
	__shared__ Datum	l_values[NOGROUP_BLOCK_SZ];
	__shared__ cl_uint	base;

	/* skip if previous stage reported an error */
	if (kgjoin_errorbuf &&
		__syncthreads_count(kgjoin_errorbuf->errcode) != 0)
		return;
	if (__syncthreads_count(kgpreagg->kerror.errcode) != 0)
		return;

	assert(kgpreagg->num_group_keys == 0);
	assert(kds_slot->format == KDS_FORMAT_SLOT);
	assert(kds_final->format == KDS_FORMAT_SLOT);
	assert(kds_slot->ncols == kds_final->ncols);
	if (get_global_id() == 0)
		kgpreagg->setup_slot_done = true;
	row_inval_map = KERN_GPUPREAGG_ROW_INVALIDATION_MAP(kgpreagg);

	do {
		/* fetch next items from the kds_slot */
		if (get_local_id() == 0)
			base = atomicAdd(&kgpreagg->read_slot_pos, NOGROUP_BLOCK_SZ);
		__syncthreads();

		if (base + NOGROUP_BLOCK_SZ >= kds_slot->nitems)
			is_last_reduction = true;
		if (base >= kds_slot->nitems)
			break;
		nvalids = Min(kds_slot->nitems - base, NOGROUP_BLOCK_SZ);
		/* reductions for each columns */
		for (j=0; j < kds_slot->ncols; j++)
		{
			int		dist, buddy;

			/* do nothing, if attribute is not preagg-function */
			if (!attr_is_preagg[j])
				continue;
			/* load the value from kds_slot to local */
			for (i = get_local_id(); i < nvalids; i += get_local_size())
			{
				assert(base + i < kds_slot->nitems);
				l_dclass[i] = KERN_DATA_STORE_DCLASS(kds_slot, base + i)[j];
				l_values[i] = KERN_DATA_STORE_VALUES(kds_slot, base + i)[j];
			}
			__syncthreads();

			/* do reduction */
			for (dist=2, buddy=1; dist < 2 * nvalids; buddy=dist, dist *= 2)
			{
				for (i = get_local_id(); i < nvalids; i += get_local_size())
				{
					if ((i & (dist - 1)) == 0 &&
						(i + (buddy)) < nvalids)
					{
						gpupreagg_nogroup_calc(j,
											   &l_dclass[i],
											   &l_values[i],
											   l_dclass[i + buddy],
											   l_values[i + buddy]);
					}
				}
				__syncthreads();
			}

			/* store this value to kds_slot from isnull/values */
			if (get_local_id() == 0)
			{
				KERN_DATA_STORE_DCLASS(kds_slot, base)[j] = l_dclass[0];
				KERN_DATA_STORE_VALUES(kds_slot, base)[j] = l_values[0];
			}
			__syncthreads();
		}
		/* update the final reduction buffer */
		if (get_local_id() == 0)
		{
			cl_uint		old_nitems = 0;
			cl_uint		new_nitems = 0xffffffff;	/* LOCKED */
			cl_uint		cur_nitems;

		try_again:
			cur_nitems = atomicCAS(&kds_final->nitems,
								   old_nitems,
								   new_nitems);
			if (cur_nitems == 0)
			{
				/* just copy */
				memcpy(KERN_DATA_STORE_DCLASS(kds_final, 0),
					   KERN_DATA_STORE_DCLASS(kds_slot, base),
					   sizeof(cl_char) * kds_slot->ncols);
				memcpy(KERN_DATA_STORE_VALUES(kds_final, 0),
					   KERN_DATA_STORE_VALUES(kds_slot, base),
					   sizeof(Datum) * kds_slot->ncols);
				atomicAdd(&kgpreagg->num_groups, 1);
				atomicExch(&kds_final->nitems, 1);	/* UNLOCKED */
			}
			else if (cur_nitems == 1)
			{
				/*
				 * NOTE: nogroup reduction has no grouping keys, and
				 * GpuPreAgg does not support aggregate functions that
				 * have variable length fields as internal state.
				 * So, we don't care about copy of grouping keys.
				 */
				gpupreagg_global_calc(
					KERN_DATA_STORE_DCLASS(kds_final, 0),
					KERN_DATA_STORE_VALUES(kds_final, 0),
					KERN_DATA_STORE_DCLASS(kds_slot, base),
					KERN_DATA_STORE_VALUES(kds_slot, base));
			}
			else
			{
				assert(cur_nitems == 0xffffffff);
				goto try_again;
			}
		}
		/*
		 * Mark this row is invalid because its values are already
		 * accumulated to the final buffer. Once subsequent operations
		 * reported CpuReCheck error, CPU fallback routine shall ignore
		 * the rows to avoid duplication in count.
		 */
		for (i = get_local_id(); i < nvalids; i += get_local_size())
			row_inval_map[base + i] = true;
		__syncthreads();
	} while (!is_last_reduction);
}

/*
 * gpupreagg_init_final_hash
 *
 * It initializes the f_hash prior to gpupreagg_final_reduction
 */
KERNEL_FUNCTION(void)
gpupreagg_init_final_hash(kern_global_hashslot *f_hash,
						  size_t f_hashsize,
						  size_t f_hashlimit)
{
	size_t		hash_index;

	if (get_global_id() == 0)
	{
		f_hash->lock = 0;
		f_hash->hash_usage = 0;
		f_hash->hash_size = f_hashsize;
		f_hash->hash_limit = f_hashlimit;
	}

	for (hash_index = get_global_id();
		 hash_index < f_hashsize;
		 hash_index += get_global_size())
	{
		f_hash->hash_slot[hash_index].s.hash = 0;
		f_hash->hash_slot[hash_index].s.index = (cl_uint)(0xffffffff);
	}
}

/*
 * gpupreagg_expand_final_hash - expand size of the final hash slot on demand,
 * up to the f_hashlimit. It internally acquires shared lock of the final
 * hash-slot, if it returns true. So, caller MUST release it when a series of
 * operations get completed. Elsewhere, it returns false. caller MUST retry.
 */
STATIC_FUNCTION(bool)
gpupreagg_expand_final_hash(kern_context *kcxt,
							kern_global_hashslot *f_hash,
							cl_uint num_new_items)
{
	pagg_hashslot curval;
	pagg_hashslot newval;
	cl_uint		i, j, f_hashsize;
	cl_uint		old_lock;
	cl_uint		cur_lock;
	cl_uint		new_lock;
	cl_uint		count;
	cl_bool		lock_wait;
	cl_bool		has_exclusive_lock = false;
	__shared__ cl_uint	curr_usage;
	__shared__ cl_uint	curr_size;

	/*
	 * Get shared-lock on the final hash-slot
	 */
	if (get_local_id() == 0)
	{
		old_lock = f_hash->lock;
		if ((old_lock & 0x0001) != 0)
			lock_wait = true;		/* someone has exclusive lock */
		else
		{
			new_lock = old_lock + 2;
			cur_lock = atomicCAS(&f_hash->lock,
								 old_lock,
								 new_lock);
			if (cur_lock == old_lock)
				lock_wait = false;	/* OK, shared lock is acquired */
			else
				lock_wait = true;	/* Oops, conflict. Retry again */
		}
	}
	else
		lock_wait = false;
	if (__syncthreads_count(lock_wait) > 0)
		return false;

	/*
	 * Expand the final hash-slot on demand, if it may overflow.
	 * No concurrent blocks are executable during the hash-slot eapansion,
	 * we need to acquire exclusive lock here.
	 */
	if (get_local_id() == 0)
	{
		curr_usage = f_hash->hash_usage;
		curr_size  = f_hash->hash_size;
	}
	__syncthreads();
	if (curr_usage + num_new_items < GLOBAL_HASHSLOT_THRESHOLD(curr_size))
		return true;		/* no need to expand the hash-slot now */
	if (curr_size > f_hash->hash_limit)
	{
		/* no more space to expand */
		STROM_EREPORT(kcxt, ERRCODE_STROM_DATASTORE_NOSPACE,
					  "f_hash has no more space");
		goto out_unlock;
	}

	/*
	 * Hmm... it looks current hash-slot usage is close to the current size,
	 * so try to acquire the exclusive lock and expand the hash-slot.
	 */
	if (get_local_id() == 0)
	{
		old_lock = f_hash->lock;
		if ((old_lock & 0x0001) != 0)
			lock_wait = true;		/* someone already has exclusive lock */
		else
		{
			assert(old_lock >= 0x0002);
			/* release shared lock, and acquire exclusive lock */
			new_lock = (old_lock - 2) | 0x0001;
			cur_lock = atomicCAS(&f_hash->lock,
								 old_lock,
								 new_lock);
			if (cur_lock == old_lock)
				lock_wait = false;	/* OK, exclusive lock is acquired */
			else
				lock_wait = true;
		}
	}
	else
		lock_wait = false;

	/* cannot acquire the exclusive lock? */
	if (__syncthreads_count(lock_wait) > 0)
		goto out_unlock;

	/* wait for completion of other shared-lock holder */
	for (;;)
	{
		if (get_local_id() == 0)
			lock_wait = (f_hash->lock != 0x0001 ? true : false);
		else
			lock_wait = false;
		if (__syncthreads_count(lock_wait) == 0)
			break;
	}
	has_exclusive_lock = true;

	/*
	 * OK, Expand the final hash-slot
	 */
	if (get_local_id() == 0)
		curr_size = f_hash->hash_size;
	__syncthreads();
	if (curr_size >= f_hash->hash_limit)
	{
		/* no more space to expand */
		if (get_local_id() == 0)
			f_hash->hash_size = UINT_MAX;
		STROM_EREPORT(kcxt, ERRCODE_STROM_DATASTORE_NOSPACE,
					  "f_hash has no more space");
		goto out_unlock;
	}
	else if (curr_size >= f_hash->hash_limit / 2)
		f_hashsize = f_hash->hash_limit;
	else
		f_hashsize = curr_size * 2;

	for (i = curr_size + get_local_id();
		 i < f_hashsize;
		 i += get_local_size())
	{
		f_hash->hash_slot[i].s.hash = 0;
		f_hash->hash_slot[i].s.index = (cl_uint)(0xffffffff);
	}
	__syncthreads();

	/*
	 * Move the hash entries to new position
	 */
	count = 0;
	for (i = get_local_id(); i < f_hashsize; i += get_local_size())
	{
		cl_int		nloops;

		newval.s.hash = 0;
		newval.s.index = (cl_uint)(0xffffffff);
		curval.value = atomicExch(&f_hash->hash_slot[i].value, newval.value);

		for (nloops = 32;
			 nloops > 0 && curval.s.index != (cl_uint)(0xffffffff);
			 nloops--)
		{
			/* should not be under locking */
			assert(curval.s.index != (cl_uint)(0xfffffffe));
			j = curval.s.hash % f_hashsize;

			newval = curval;
			curval.value = atomicExch(&f_hash->hash_slot[j].value,
									  newval.value);
		}

		/*
		 * NOTE: If hash-key gets too deep confliction than the threshold,
		 * we give up to find out new location without shift operations.
		 * It is a little waste of space on the kds_final buffer, but harmless
		 * because partial aggregation results are already on the kds_final
		 * buffer, then, further entries with same grouping keys will be added
		 * as if it is newly injected.
		 * It takes extra area of kds_final, however, it is much better than
		 * infinite loop.
		 */
		if (curval.s.index != (cl_uint)(0xffffffff))
			count++;
	}
	/* note: pgstromStairlikeSum contains __syncthreads() */
	count = pgstromStairlikeSum(count, NULL);
	if (get_local_id() == 0)
	{
		printf("GpuPreAgg: expand final hash slot %u -> %u (%u dropped)\n",
			   f_hash->hash_size, f_hashsize, count);
		f_hash->hash_size = f_hashsize;
	}
	__syncthreads();

out_unlock:
	/* release exclusive lock and suggest caller retry */
	if (get_local_id() == 0)
	{
		if (has_exclusive_lock)
			atomicAnd(&f_hash->lock, (cl_uint)(0xfffffffe));
		else
			atomicSub(&f_hash->lock, 2);
	}
	return false;
}

/*
 * gpupreagg_final_reduction
 */
STATIC_FUNCTION(cl_bool)
gpupreagg_final_reduction(kern_context *kcxt,
						  kern_gpupreagg *kgpreagg,
						  kern_data_store *kds_slot,
						  cl_uint slot_index,
						  cl_uint hash_value,
						  kern_data_store *kds_final,
						  kern_global_hashslot *f_hash)
{
	size_t			f_hashsize = f_hash->hash_size;
	cl_uint			nconflicts = 0;
	cl_bool			is_owner = false;
	cl_bool			meet_locked = false;
	cl_uint			allocated = 0;
	cl_uint			index;
	pagg_hashslot	old_slot;
	pagg_hashslot	new_slot;
	pagg_hashslot	cur_slot;

	new_slot.s.hash	 = hash_value;
	new_slot.s.index = (cl_uint)(0xfffffffeU);	/* LOCK */
	old_slot.s.hash  = 0;
	old_slot.s.index = (cl_uint)(0xffffffffU);	/* EMPTY */
	index = hash_value % f_hashsize;

retry_fnext:
	if (f_hashsize > f_hash->hash_limit)
	{
		/* no more space to expand */
		STROM_EREPORT(kcxt, ERRCODE_STROM_DATASTORE_NOSPACE,
					  "f_hash has no more space");
		return false;
	}
	else if (f_hash->hash_usage > GLOBAL_HASHSLOT_THRESHOLD(f_hashsize))
	{
		/* try to expand the final hash-slot */
		return false;
	}
	cur_slot.value = atomicCAS(&f_hash->hash_slot[index].value,
							   old_slot.value, new_slot.value);
	if (cur_slot.value == old_slot.value)
	{
		atomicAdd(&f_hash->hash_usage, 1);

		/*
		 * This thread shall be responsible to this grouping-key
		 *
		 * MEMO: We may need to check whether the new nitems exceeds
		 * usage of the extra length of the kds_final from the tail,
		 * instead of the nrooms, however, some code assumes kds_final
		 * can store at least 'nrooms' items. So, right now, we don't
		 * allow to expand extra area across nrooms boundary.
		 */
		new_slot.s.index = atomicAdd(&kds_final->nitems, 1);
		if (new_slot.s.index < kds_final->nrooms)
		{
			cl_int	len = gpupreagg_final_data_move(kcxt,
													kds_slot,
													slot_index,
													kds_final,
													new_slot.s.index);
			if (len < 0)
				new_slot.s.index = (cl_uint)(0xffffffffU);	/* EMPTY */
			else
				allocated += len;
		}
		else
		{
			new_slot.s.index = (cl_uint)(0xffffffffU);
			STROM_EREPORT(kcxt, ERRCODE_STROM_DATASTORE_NOSPACE,
						  "kds_final has no more space");
		}
		__threadfence();
		/* UNLOCK */
		old_slot.value = atomicExch(&f_hash->hash_slot[index].value,
									new_slot.value);
		assert(old_slot.s.hash == new_slot.s.hash &&
			   old_slot.s.index == (cl_uint)(0xfffffffeU));
		/* this thread performs as owner of this slot */
		is_owner = true;
	}
	else if (cur_slot.s.hash != hash_value)
	{
		/* Hash-value conflicts by other grouping key */
		index = (index + 1) % f_hashsize;
		nconflicts++;
		goto retry_fnext;
	}
	else if (cur_slot.s.index == (cl_uint)(0xfffffffe))
	{
		/*
		 * This final hash-slot is currently locked by the concurrent thread,
		 * so we cannot check grouping-keys right now. Caller must retry.
		 */
		meet_locked = true;
		nconflicts++;
	}
	else if (!gpupreagg_keymatch(kcxt,
								 kds_slot, slot_index,
								 kds_final, cur_slot.s.index))
	{
		/*
		 * Hash-value conflicts by other grouping key; which has same hash-
		 * value, but grouping-keys itself are not identical. So, we try to
		 * use the next slot instead.
		 */
		index = (index + 1) % f_hashsize;
		goto retry_fnext;
	}
	else
	{
		/*
		 * Global reduction for each column
		 *
		 * Any threads that are NOT responsible to grouping-key calculates
		 * aggregation on the item that is responsibles.
		 * Once atomic operations got finished, isnull/values of the current
		 * thread shall be accumulated.
		 */

#if 0
		/*
		 * MEMO: Multiple concurrent threads updates @kds_final->nitems
		 * using atomicAdd(), thus, this operation works on L2-cache.
		 * On the other hands, NVIDIA says Volta architecture uses L1-cache
		 * implicitly, if compiler detects the target variable is read-only.
		 * If NVRTC compiler oversights memory update of atomicXXX() and
		 * it does not invalidate L1-cache, we may look at older version
		 * of the variable.
		 * In fact, we never reproduce the problem when @kds_final->nitems
		 * is referenced using atomic operation which has no effect.
		 *
		 * The above memo is just my hypothesis, shall be reported to
		 * NVIDIA for confirmation and further investigation later.
		 */
		assert(cur_slot.s.index < Min(kds_final->nitems,
									  kds_final->nrooms));
#endif
		gpupreagg_global_calc(
			KERN_DATA_STORE_DCLASS(kds_final, cur_slot.s.index),
			KERN_DATA_STORE_VALUES(kds_final, cur_slot.s.index),
			KERN_DATA_STORE_DCLASS(kds_slot, slot_index),
			KERN_DATA_STORE_VALUES(kds_slot, slot_index));
	}

	/*
	 * update run-time statistics
	 */
	if (is_owner)
		atomicAdd(&kgpreagg->num_groups, 1);
	if (allocated > 0)
		atomicAdd(&kgpreagg->extra_usage, allocated);
	if (nconflicts > 0)
		atomicAdd(&kgpreagg->fhash_conflicts, nconflicts);

	return !meet_locked;
}

/*
 * gpupreagg_group_reduction
 */
DEVICE_FUNCTION(void)
gpupreagg_groupby_reduction(kern_context *kcxt,
							kern_gpupreagg *kgpreagg,		/* in/out */
							kern_errorbuf *kgjoin_errorbuf,	/* in */
							kern_data_store *kds_slot,		/* in */
							kern_data_store *kds_final,		/* shared out */
							kern_global_hashslot *f_hash)	/* shared out */
{
	varlena		   *kparam_0 = (varlena *)kparam_get_value(kcxt->kparams, 0);
	cl_char		   *attr_is_preagg = (cl_char *)VARDATA(kparam_0);
	cl_uint			i, j;
	cl_uint			kds_index;
	cl_uint			buf_index;
	cl_uint			hash_index;
	cl_uint			hash_value;
	cl_bool			is_leader;
	cl_bool			is_last_reduction = false;
	cl_bool			l_hashslot_cleanup = true;
	cl_char		   *row_inval_map;
#define GROUPBY_LOCAL_HASHSIZE	2400
#define GROUPBY_LOCAL_BUFSIZE	1800
	__shared__ cl_bool	l_dclass[GROUPBY_LOCAL_BUFSIZE];
	__shared__ Datum	l_values[GROUPBY_LOCAL_BUFSIZE];
	__shared__ cl_short l_b2h_index[GROUPBY_LOCAL_BUFSIZE];	/* B->H index */
	__shared__ pagg_hashslot l_hashslot[GROUPBY_LOCAL_HASHSIZE];
	__shared__ cl_short	l_h2b_index[GROUPBY_LOCAL_HASHSIZE];/* H->B index */
	__shared__ cl_uint	l_nitems;
	__shared__ cl_uint	base;

	/* skip if previous stage reported an error */
	if (kgjoin_errorbuf &&
		__syncthreads_count(kgjoin_errorbuf->errcode) != 0)
		return;
	if (__syncthreads_count(kgpreagg->kerror.errcode) != 0)
		return;

	assert(kgpreagg->num_group_keys > 0);
	assert(kds_slot->format == KDS_FORMAT_SLOT);
	assert(kds_final->format == KDS_FORMAT_SLOT);
	row_inval_map = KERN_GPUPREAGG_ROW_INVALIDATION_MAP(kgpreagg);
	if (get_global_id() == 0)
		kgpreagg->setup_slot_done = true;

	do {
		cl_bool	   *slot_dclass = NULL;
		Datum	   *slot_values = NULL;
		cl_bool	   *dest_dclass = NULL;
		Datum	   *dest_values = NULL;

		/* cleanup hash-slot */
		if (l_hashslot_cleanup)
		{
			l_hashslot_cleanup = false;
			for (i = get_local_id();
				 i < GROUPBY_LOCAL_HASHSIZE;
				 i += get_local_size())
			{
				l_hashslot[i].s.hash = 0;
				l_hashslot[i].s.index = (cl_uint)(0xffffffff);
			}
			if (get_local_id() == 0)
				l_nitems = 0;
			__syncthreads();
		}
		/* fetch next items from the kds_slot */
		if (get_local_id() == 0)
			base = atomicAdd(&kgpreagg->read_slot_pos, get_local_size());
		__syncthreads();
		if (base + get_local_size() >= kds_slot->nitems)
			is_last_reduction = true;
		if (base >= kds_slot->nitems)
			goto skip_local_reduction;
		/*
		 * Lookup local hash-slot, or create a new one if not exists
		 */
		kds_index = base + get_local_id();
		if (kds_index < kds_slot->nitems)
		{
			slot_dclass = KERN_DATA_STORE_DCLASS(kds_slot, kds_index);
			slot_values = KERN_DATA_STORE_VALUES(kds_slot, kds_index);
			hash_value = gpupreagg_hashvalue(kcxt,
											 slot_dclass,
											 slot_values);
		}
		/* error checks */
		if (__syncthreads_count(kcxt->errcode) > 0)
			return;

		if (kds_index < kds_slot->nitems)
		{
			pagg_hashslot	old_slot;
			pagg_hashslot	new_slot;
			pagg_hashslot	cur_slot;

			new_slot.s.hash     = hash_value;
			new_slot.s.index    = kds_index;
			old_slot.s.hash     = 0;
			old_slot.s.index    = (cl_uint)(0xffffffff);
			hash_index = hash_value % GROUPBY_LOCAL_HASHSIZE;
			do {
				cur_slot.value = atomicCAS(&l_hashslot[hash_index].value,
										   old_slot.value,
										   new_slot.value);
				if (cur_slot.value == old_slot.value)
				{
					/* hash-slot was empty, so inject a new one */
					buf_index = atomicAdd(&l_nitems, 1);
					l_h2b_index[hash_index] = buf_index;
					l_b2h_index[buf_index] = hash_index;
					dest_dclass = slot_dclass;
					dest_values = slot_values;
					break;
				}
				assert(cur_slot.s.index < kds_slot->nitems &&
					   cur_slot.s.index != kds_index);
				if (cur_slot.s.hash == hash_value &&
					gpupreagg_keymatch(kcxt,
									   kds_slot, kds_index,
									   kds_slot, cur_slot.s.index))
				{
					/* grouping-keys are already on hash-slot */
					i = cur_slot.s.index;
					dest_dclass = KERN_DATA_STORE_DCLASS(kds_slot, i);
					dest_values = KERN_DATA_STORE_VALUES(kds_slot, i);
					break;
				}
				hash_index = (hash_index + 1) % GROUPBY_LOCAL_HASHSIZE;
			} while (kcxt->errcode == ERRCODE_STROM_SUCCESS);
		}
		else
		{
			hash_index = UINT_MAX;	/* shall never fit */
		}
		/* error checks */
		if (__syncthreads_count(kcxt->errcode) > 0)
			return;
		/*
		 * Note that l_h2b_index / l_b2h_index are not updated atomically,
		 * so we have to fetch values from these array after synchronization
		 * point.
		 */
		if (hash_index < GROUPBY_LOCAL_HASHSIZE)
		{
			buf_index = l_h2b_index[hash_index];
			assert(buf_index < l_nitems);
			assert(l_b2h_index[buf_index] == hash_index);
		}
		else
			buf_index = UINT_MAX;	/* shall never fit */

		/*
		 * Determines the thread that performs as "owner" of the entry.
		 * Owner loads initial value of reduction to l_dclass[]/l_values[]
		 * buffer, then stores the reduction results to the same slot on
		 * the kds_slot. These are NOT atomic operation.
		 */
		for (i = get_local_id();
			 i < GROUPBY_LOCAL_BUFSIZE;
			 i += get_local_size())
			l_values[i] = 0;
		__syncthreads();
		if (buf_index < GROUPBY_LOCAL_BUFSIZE &&
			atomicMax(&l_values[buf_index], 1) == 0)
		{
			is_leader = true;
		}
		else
		{
			is_leader = false;
		}
		__syncthreads();

		/*
		 * Local reduction for each column
		 */
		for (j=0; j < kds_slot->ncols; j++)
		{
			if (!attr_is_preagg[j])
				continue;
			/* load the value to local storage */
			if (is_leader)
			{
				assert(buf_index < GROUPBY_LOCAL_BUFSIZE);
				l_dclass[buf_index] = dest_dclass[j];
				l_values[buf_index] = dest_values[j];
			}
			__syncthreads();

			/* reduction by atomic operation */
			if (kds_index < kds_slot->nitems && slot_values != dest_values)
			{
				assert(buf_index < GROUPBY_LOCAL_BUFSIZE);
				gpupreagg_local_calc(j,
									 &l_dclass[buf_index],
									 &l_values[buf_index],
									 slot_dclass[j],
									 slot_values[j]);
			}
			__syncthreads();

			/* store the value from local storage */
			if (is_leader)
			{
				dest_dclass[j] = l_dclass[buf_index];
				dest_values[j] = l_values[buf_index];
			}
			__syncthreads();
		}
		/*
		 * NOTE: This row is not merged to the final buffer yet, however,
		 * its values are already accumulated to hash-owner's slot. So,
		 * when CPU fallback routine processes the kds_slot, no need to
		 * process this row again. (owner's row already contains its own
		 * values and accumulated ones from non-owner's row)
		 */
		if (!is_leader && kds_index < kds_slot->nitems)
			row_inval_map[kds_index] = true;

	skip_local_reduction:
		/*
		 * final reduction steps if needed
		 */
		if (is_last_reduction ||
			l_nitems + get_local_size() > GROUPBY_LOCAL_BUFSIZE)
		{
			cl_uint		nloops;
			cl_uint		loop;
			cl_uint		nvalids;

			nloops = (l_nitems + get_local_size() - 1) / get_local_size();
			for (loop=0; loop < nloops; loop++)
			{
				cl_bool		reduction_done = false;
				cl_bool		lock_wait;

				nvalids = Min(l_nitems - loop * get_local_size(),
							  get_local_size());
				do {
					/*
					 * Get shared-lock of the final hash-slot
					 * If it may have overflow, expand length of the final
					 * hash-slot on demand, under its exclusive lock.
					 */
					if (!gpupreagg_expand_final_hash(kcxt, f_hash, nvalids))
						lock_wait = true;
					else if (get_local_id() >= nvalids || reduction_done)
						lock_wait = false;
					else
					{
						buf_index  = get_local_size() * loop + get_local_id();
						assert(buf_index < GROUPBY_LOCAL_BUFSIZE);
						hash_index = l_b2h_index[buf_index];
						assert(hash_index < GROUPBY_LOCAL_HASHSIZE);
						kds_index  = l_hashslot[hash_index].s.index;
						hash_value = l_hashslot[hash_index].s.hash;
						assert(kds_index < kds_slot->nitems);

						if (gpupreagg_final_reduction(kcxt,
													  kgpreagg,
													  kds_slot,
													  kds_index,
													  hash_value,
													  kds_final,
													  f_hash))
						{
							reduction_done = true;
							lock_wait = false;
						}
						else
						{
							lock_wait = true;
						}
					}
					__syncthreads();
					/* release shared lock of the final hash-slot */
					if (get_local_id() == 0)
						atomicSub(&f_hash->lock, 2);
					/* quick bailout on error */
					if (__syncthreads_count(kcxt->errcode) > 0)
						return;
				} while (__syncthreads_count(lock_wait) > 0);
			}
			/*
			 * OK, pending items were successfully moved to the final buffer.
			 * Local hash-slots shall be reset, and restart.
			 */
			l_hashslot_cleanup = true;
		}
		__syncthreads();
	} while(!is_last_reduction);
}
