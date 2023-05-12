/*
 * cuda_gpupreagg.h
 *
 * Preprocess of aggregate using GPU acceleration, to reduce number of
 * rows to be processed by CPU; including the Sort reduction.
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "cuda_common.h"
#include "cuda_gpupreagg.h"
#include "cuda_postgis.h"

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
					case DATUM_CLASS__GEOMETRY:
						tup_extra[j] = sizeof(pg_geometry_t);
						extra_sz += MAXALIGN(sizeof(pg_geometry_t));
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
									  &tupitem->htup.t_ctid,
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
		count = __syncthreads_count(src_index < src_nitems);
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
 * gpupreagg_setup_column
 */
DEVICE_FUNCTION(void)
gpupreagg_setup_column(kern_context *kcxt,
                       kern_gpupreagg *kgpreagg,
                       kern_data_store *kds_src,    /* in: KDS_FORMAT_COLUMN */
                       kern_data_extra *kds_extra,
                       kern_data_store *kds_slot)
{
	cl_uint		src_base;
	cl_char	   *tup_dclass;
	Datum	   *tup_values;
	cl_int	   *tup_extra;	/* !!not related to extra buffer of column format!! */
	cl_char	   *vlbuf_base;
	gpupreaggSuspendContext *my_suspend;

	assert(kds_src->format == KDS_FORMAT_COLUMN &&
		   kds_slot->format == KDS_FORMAT_SLOT);
	/* resume kernel from the point where suspended, if any */
	my_suspend = KERN_GPUPREAGG_SUSPEND_CONTEXT(kgpreagg, get_group_id());
	if (kgpreagg->resume_context)
		src_base = my_suspend->c.src_base;
	else
		src_base = get_global_base();

	tup_dclass = (cl_char *)
		kern_context_alloc(kcxt, sizeof(cl_char) * kds_slot->ncols);
	tup_values = (Datum *)
		kern_context_alloc(kcxt, sizeof(Datum)   * kds_slot->ncols);
	tup_extra  = (cl_int *)
		kern_context_alloc(kcxt, sizeof(cl_int)  * kds_slot->ncols);
	if (!tup_dclass || !tup_values || !tup_extra)
		STROM_EREPORT(kcxt, ERRCODE_OUT_OF_MEMORY, "out of memory");
	/* bailout if any errors */
	if (__syncthreads_count(kcxt->errcode) > 0)
		goto skip;
	vlbuf_base = kcxt->vlpos;

	while (src_base < kds_src->nitems)
	{
		cl_uint		src_index = src_base + get_local_id();
		cl_uint		slot_index;
		cl_uint		nvalids;
		cl_uint		count;
		cl_bool		visible = false;
		cl_bool		rc = false;

		kcxt->vlpos = vlbuf_base;		/* rewind */
		if (src_index < kds_src->nitems)
		{
			visible = kern_check_visibility_column(kcxt,
												   kds_src,
												   src_index);
			if (visible)
			{
				rc = gpupreagg_quals_eval_column(kcxt,
												 kds_src,
												 kds_extra,
												 src_index);
			}
			kcxt->vlpos = vlbuf_base;	/* rewind */
		}
		/* bailout if any errors */
		if (__syncthreads_count(kcxt->errcode) > 0)
			break;
		/* allocation of kds_slot buffer, if any */
		slot_index = pgstromStairlikeBinaryCount(rc ? 1 : 0, &nvalids);
		if (nvalids > 0)
		{
			if (rc)
			{
				gpupreagg_projection_column(kcxt,
											kds_src,
											kds_extra,
											src_index,
											tup_dclass,
											tup_values);
			}
			/* bailout if any errors */
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
		/* update  statistics */
		count = __syncthreads_count(visible);
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
							kern_data_store *kds_final,		/* global out */
                            cl_char *p_dclass,		/* __private__ */
                            Datum   *p_values,		/* __private__ */
							char	*p_extras)		/* __private__ */
{
	cl_bool		is_last_reduction = false;
	cl_bool		try_final_merge = true;
	cl_uint		lane_id = (get_local_id() & warpSize - 1);

	/* init local/private buffer */
	assert(MAXWARPS_PER_BLOCK <= get_local_size() &&
		   MAXWARPS_PER_BLOCK == warpSize);
	gpupreagg_init_local_slot(p_dclass, p_values, p_extras);

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

	/* start private reduction */
	is_last_reduction = false;
	do {
		cl_uint		index;

		if (lane_id == 0)
			index = atomicAdd(&kgpreagg->read_slot_pos, warpSize);
		index = __shfl_sync(__activemask(), index, 0);
		if (index + warpSize >= kds_slot->nitems)
			is_last_reduction = true;
		index += lane_id;

		/* accumulate to the private buffer */
		if (index < kds_slot->nitems)
		{
			gpupreagg_update_normal(p_dclass,
									p_values,
									GPUPREAGG_ACCUM_MAP_LOCAL,
									KERN_DATA_STORE_DCLASS(kds_slot, index),
									KERN_DATA_STORE_VALUES(kds_slot, index),
									GPUPREAGG_ACCUM_MAP_GLOBAL);
		}
	} while (!is_last_reduction);

	__syncthreads();

	/*
	 * inter-warp reduction using shuffle operations
	 */
	for (cl_uint mask = 1; mask < warpSize; mask += mask)
	{
		cl_uint		buddy_id = ((get_local_id() ^ mask) & (warpSize-1));

		gpupreagg_merge_shuffle(p_dclass,
								p_values,
								GPUPREAGG_ACCUM_MAP_LOCAL,
								buddy_id);
	}

	/*
	 * update the final buffer
	 */
	try_final_merge = ((get_local_id() & (warpSize - 1)) == 0);
	do {
		if (try_final_merge)
		{
			union {
				struct {
					cl_uint	nitems;
					cl_uint	usage;
				} i;
				cl_ulong	v64;
			} oldval, curval, newval;

			assert((get_local_id() & (warpSize - 1)) == 0);

			oldval.i.nitems = 0;
			oldval.i.usage  = kds_final->usage;
			newval.i.nitems = 0xffffffffU;		/* LOCKED */
			newval.i.usage  = kds_final->usage
				+ __kds_packed(GPUPREAGG_ACCUM_EXTRA_BUFSZ);

			curval.v64 = atomicCAS((cl_ulong *)&kds_final->nitems,
								   oldval.v64,
								   newval.v64);
			if (curval.i.nitems <= 1)
			{
				cl_char	   *f_dclass = KERN_DATA_STORE_DCLASS(kds_final, 0);
				Datum	   *f_values = KERN_DATA_STORE_VALUES(kds_final, 0);
				char	   *f_extras;

				if (curval.i.nitems == 0)
				{
					f_extras = ((char *)kds_final +
								kds_final->length -
								__kds_unpack(curval.i.usage) -
								GPUPREAGG_ACCUM_EXTRA_BUFSZ);
					gpupreagg_init_final_slot(f_dclass, f_values, f_extras);
					atomicAdd(&kgpreagg->num_groups, 1);
					__threadfence();
					atomicExch(&kds_final->nitems, 1);	/* UNLOCK */
				}
				gpupreagg_merge_atomic(f_dclass,
									   f_values,
									   GPUPREAGG_ACCUM_MAP_GLOBAL,
									   p_dclass,
									   p_values,
									   GPUPREAGG_ACCUM_MAP_LOCAL);
				try_final_merge = false;
				kgpreagg->final_buffer_modified = true;
			}
			else
			{
				assert(curval.i.nitems == 0xffffffffU);
			}
		}
	} while (__syncthreads_count(try_final_merge) > 0);
}

#define HASHITEM_EMPTY		(0xffffffffU)
#define HASHITEM_LOCKED		(0xfffffffeU)

static __shared__ cl_bool l_final_buffer_modified;

/*
 * gpupreagg_init_final_hash
 */
KERNEL_FUNCTION(void)
gpupreagg_init_final_hash(kern_global_hashslot *f_hash,
						  size_t f_hash_nslots,
						  size_t f_hash_length)
{
	if (get_global_id() == 0)
	{
		f_hash->length = f_hash_length;
		f_hash->lock = 0;
		f_hash->usage = 0;
		f_hash->nslots = f_hash_nslots;
	}

	for (size_t i = get_global_id(); i < f_hash_nslots; i += get_global_size())
		f_hash->slots[i] = HASHITEM_EMPTY;
}

/*
 * gpupreagg_create_final_slot
 *
 *
 */
STATIC_FUNCTION(cl_uint)
gpupreagg_create_final_slot(kern_context *kcxt,
							kern_data_store *kds_final,
							kern_data_store *kds_src,
							cl_uint src_index,
							cl_char *l_dclass,
							Datum *l_values)
{
	cl_char	   *src_dclass = KERN_DATA_STORE_DCLASS(kds_src, src_index);
	Datum	   *src_values = KERN_DATA_STORE_VALUES(kds_src, src_index);
	cl_char	   *dst_dclass;
	Datum	   *dst_values;
	cl_uint		dst_index;
	cl_uint		alloc_sz;
	char	   *extra = NULL;
	union {
		struct {
			cl_uint	nitems;
			cl_uint	usage;
		} i;
		cl_ulong	v64;
	} oldval, curval, newval;

	/* sanity checks */
	assert(kds_final->format == KDS_FORMAT_SLOT &&
		   kds_src->format == KDS_FORMAT_SLOT);
	assert(kds_final->ncols == kds_src->ncols);
	assert(src_index < kds_src->nitems);

	/* size for extra allocation */
	alloc_sz = GPUPREAGG_ACCUM_EXTRA_BUFSZ;
	for (int j=0; j < kds_src->ncols; j++)
	{
		kern_colmeta   *cmeta = &kds_src->colmeta[j];
		cl_char			dclass = src_dclass[j];
		cl_uint			len;

		if (GPUPREAGG_ATTR_IS_ACCUM_VALUES[j])
			continue;
		if (dclass == DATUM_CLASS__NULL)
			continue;

		if (cmeta->attbyval)
		{
			assert(dclass == DATUM_CLASS__NORMAL);
		}
		else if (cmeta->attlen > 0)
		{
			assert(dclass == DATUM_CLASS__NORMAL);
			alloc_sz += MAXALIGN(cmeta->attlen);
		}
		else
		{
			assert(cmeta->attlen == -1);
			switch (dclass)
			{
				case DATUM_CLASS__NORMAL:
					len = VARSIZE_ANY(DatumGetPointer(src_values[j]));
					break;
				case DATUM_CLASS__VARLENA:
					len = pg_varlena_datum_length(kcxt, src_values[j]);
					break;
				case DATUM_CLASS__ARRAY:
					len = pg_array_datum_length(kcxt, src_values[j]);
					break;
				case DATUM_CLASS__COMPOSITE:
					len = pg_composite_datum_length(kcxt, src_values[j]);
					break;
				case DATUM_CLASS__GEOMETRY:
					len = pg_geometry_datum_length(kcxt, src_values[j]);
					break;
				default:
					STROM_ELOG(kcxt, "unexpected internal format code");
					return UINT_MAX;
			}
			alloc_sz += MAXALIGN(len);
		}
	}

	/*
	 * allocation of a new slot and extra buffer
	 */
	curval.i.nitems = __volatileRead(&kds_final->nitems);
	curval.i.usage  = __volatileRead(&kds_final->usage);
	do {
		newval = oldval = curval;
		newval.i.nitems += 1;
		newval.i.usage  += __kds_packed(alloc_sz);
		if (KERN_DATA_STORE_SLOT_LENGTH(kds_final, newval.i.nitems) +
			__kds_unpack(newval.i.usage) > kds_final->length)
		{
			STROM_EREPORT(kcxt, ERRCODE_OUT_OF_MEMORY,
						  "out of memory (kds_final)");
			return UINT_MAX;
		}
	} while ((curval.v64 = atomicCAS((cl_ulong *)&kds_final->nitems,
									 oldval.v64,
									 newval.v64)) != oldval.v64);
	/*
	 * Move the initial values to kds_final
	 */
	dst_index = oldval.i.nitems;
	dst_dclass = KERN_DATA_STORE_DCLASS(kds_final, dst_index);
	dst_values = KERN_DATA_STORE_VALUES(kds_final, dst_index);
	if (alloc_sz > 0)
		extra = (char *)kds_final + kds_final->length - __kds_unpack(newval.i.usage);
	l_final_buffer_modified = true;

	/* init final slot */
	gpupreagg_init_final_slot(dst_dclass, dst_values, extra);
	extra += GPUPREAGG_ACCUM_EXTRA_BUFSZ;

	/* copy the grouping keys */
	for (int j=0; j < kds_src->ncols; j++)
	{
		kern_colmeta   *cmeta = &kds_src->colmeta[j];
		cl_char			dclass = src_dclass[j];
		Datum			datum = src_values[j];
		cl_uint			len;

		if (GPUPREAGG_ATTR_IS_ACCUM_VALUES[j])
			continue;

		if (dclass == DATUM_CLASS__NULL || cmeta->attbyval)
		{
			dst_dclass[j] = dclass;
            dst_values[j] = datum;
		}
		else if (cmeta->attlen > 0)
		{
			assert(dclass == DATUM_CLASS__NORMAL);
			memcpy(extra, DatumGetPointer(datum), cmeta->attlen);
			dst_dclass[j] = DATUM_CLASS__NORMAL;
			dst_values[j] = PointerGetDatum(extra);
			extra += MAXALIGN(cmeta->attlen);
		}
		else
		{
			assert(cmeta->attlen == -1);
			switch (dclass)
			{
				case DATUM_CLASS__NORMAL:
					len = VARSIZE_ANY(datum);
					memcpy(extra, DatumGetPointer(datum), len);
					break;
				case DATUM_CLASS__VARLENA:
					len = pg_varlena_datum_write(kcxt, extra, datum);
					break;
				case DATUM_CLASS__ARRAY:
					len =  pg_array_datum_write(kcxt, extra, datum);
					break;
				case DATUM_CLASS__COMPOSITE:
					len = pg_composite_datum_write(kcxt, extra, datum);
					break;
				case DATUM_CLASS__GEOMETRY:
					len = pg_geometry_datum_write(kcxt, extra, datum);
					break;
				default:
					STROM_ELOG(kcxt, "unexpected internal format code");
					return UINT_MAX;
			}
			dst_dclass[j] = DATUM_CLASS__NORMAL;
			dst_values[j] = PointerGetDatum(extra);
			extra += MAXALIGN(len);
		}
	}
	/* copy the accum values */
	if (l_dclass && l_values)
		gpupreagg_merge_atomic(dst_dclass,
							   dst_values,
							   GPUPREAGG_ACCUM_MAP_GLOBAL,
							   l_dclass,
							   l_values,
							   GPUPREAGG_ACCUM_MAP_LOCAL);
	else
		gpupreagg_update_atomic(dst_dclass,
								dst_values,
								GPUPREAGG_ACCUM_MAP_GLOBAL,
								src_dclass,
								src_values,
								GPUPREAGG_ACCUM_MAP_GLOBAL);
	__threadfence();

	return dst_index;
}

/*
 * gpupreagg_expand_global_hash - expand size of the global hash slot on demand.
 * up to the f_hashlimit. It internally acquires shared lock of the final
 * hash-slot, if it returns true. So, caller MUST release it when a series of
 * operations get completed. Elsewhere, it returns false. caller MUST retry.
 */
STATIC_FUNCTION(cl_bool)
__expand_global_hash(kern_context *kcxt, kern_global_hashslot *f_hash)
{
	cl_bool		expanded = false;
	cl_uint		i, j;

	/*
	 * Expand the global hash-slot
	 */
	if (get_local_id() == 0)
	{
		cl_uint		__nslots = 2 * f_hash->nslots + 2000;
		cl_uint		__usage = 2 * f_hash->usage + 2000;
		size_t		consumed;

		/* expand twice and mode */
		consumed = (MAXALIGN(offsetof(kern_global_hashslot, slots[__nslots])) +
					MAXALIGN(sizeof(preagg_hash_item) * __usage));
		if (consumed <= f_hash->length)
		{
			f_hash->nslots = __nslots;
			expanded = true;
		}
		else
		{
			STROM_EREPORT(kcxt, ERRCODE_STROM_DATASTORE_NOSPACE,
						  "f_hash has no more space");
		}
	}
	if (__syncthreads_count(expanded) == 0)
		return false;		/* failed */

	/* fix up the global hash-slot */
	for (i = get_local_id(); i < f_hash->nslots; i += get_local_size())
	{
		f_hash->slots[i] = HASHITEM_EMPTY;
	}
	__syncthreads();

	for (i = 0; i < f_hash->usage; i += get_local_size())
	{
		preagg_hash_item *hitem = NULL;
		cl_uint		hindex = UINT_MAX;
		cl_uint		next;

		j = i + get_local_id();
		if (j < f_hash->usage)
		{
			hitem = GLOBAL_HASHSLOT_GETITEM(f_hash, j);
			hindex = hitem->hash % f_hash->nslots;
		}

		do {
			if (hitem)
			{
				next = __volatileRead(&f_hash->slots[hindex]);
				assert(next == HASHITEM_EMPTY || next < f_hash->usage);
				hitem->next = next;
				if (atomicCAS(&f_hash->slots[hindex], next, j) == next)
					hitem = NULL;
			}
		} while(__syncthreads_count(hitem != NULL) > 0);
	}
	return true;
}

STATIC_INLINE(cl_bool)
gpupreagg_expand_global_hash(kern_context *kcxt,
							 kern_global_hashslot *f_hash,
							 cl_uint required)
{
	cl_bool		lock_wait = false;
	cl_bool		expand_hash = false;
	cl_uint		old_lock;
	cl_uint		new_lock;
	cl_uint		curr_usage;

	/* Get shared/exclusive lock on the final hash slot */
	do {
		if (get_local_id() == 0)
		{
			curr_usage = __volatileRead(&f_hash->usage);
			expand_hash = (curr_usage + required > f_hash->nslots);

			old_lock = __volatileRead(&f_hash->lock);
			if ((old_lock & 0x0001) != 0)
				lock_wait = true;	/* someone has exclusive lock */
			else
			{
				if (expand_hash)
					new_lock = old_lock + 3;	/* shared + exclusive lock */
				else
					new_lock = old_lock + 2;	/* shared lock */

				if (atomicCAS(&f_hash->lock,
							  old_lock,
							  new_lock) == old_lock)
					lock_wait = false;	/* Ok, lock is acquired */
				else
					lock_wait = true;	/* Oops, conflict. Retry again. */
			}
		}
	} while (__syncthreads_count(lock_wait) > 0);

	if (__syncthreads_count(expand_hash) > 0)
	{
		/* wait while other threads are running in the critial section */
		lock_wait = false;
		do {
			if (get_local_id() == 0)
			{
				old_lock = __volatileRead(&f_hash->lock);
				assert((old_lock & 1) == 1);
				lock_wait = (old_lock != 3);
			}
		} while(__syncthreads_count(lock_wait) > 0);

		/*
		 * Expand the global hash table
		 */
		if (!__expand_global_hash(kcxt, f_hash))
		{
			/* Error! release exclusive lock */
			__syncthreads();
			if (get_local_id() == 0)
			{
				old_lock = atomicSub(&f_hash->lock, 3);
				assert((old_lock & 0x0001) != 0);
			}
			return false;
		}
		/* Ensure the updates of f_hash visible to others */
		__threadfence();
		/* Downgrade the lock */
		__syncthreads();
		if (get_local_id() == 0)
		{
			old_lock = atomicSub(&f_hash->lock, 1);
			assert((old_lock & 0x0001) != 0);
		}
	}
	return true;
}

/*
 * gpupreagg_global_reduction
 */
STATIC_FUNCTION(cl_bool)
gpupreagg_global_reduction(kern_context *kcxt,
						   kern_data_store *kds_slot,
						   cl_uint kds_index,
						   cl_uint hash,
						   kern_data_store *kds_final,
						   kern_global_hashslot *f_hash,
						   cl_char *l_dclass,	/* can be NULL */
						   Datum *l_values)		/* can be NULL */
{
	preagg_hash_item *hitem = NULL;
	cl_uint		hindex = hash % f_hash->nslots;
	cl_uint		next;
	cl_uint		curr;
	cl_uint		dst_index;
	cl_char	   *dst_dclass;
	Datum	   *dst_values;
	cl_bool		is_locked = false;

	/*
	 * Step-1: Lookup hash slot without locking
	 */
	curr = next = __volatileRead(&f_hash->slots[hindex]);
	__threadfence();
	if (curr == HASHITEM_LOCKED)
		return false;	/* locked, try again */
restart:
	while (curr != HASHITEM_EMPTY)
	{
		assert(curr < __volatileRead(&f_hash->usage));

		hitem = GLOBAL_HASHSLOT_GETITEM(f_hash, curr);
		if (hitem->hash == hash &&
			gpupreagg_keymatch(kcxt,
							   kds_slot, kds_index,
							   kds_final, hitem->index))
		{
			dst_dclass = KERN_DATA_STORE_DCLASS(kds_final, hitem->index);
			dst_values = KERN_DATA_STORE_VALUES(kds_final, hitem->index);

			if (l_dclass && l_values)
				gpupreagg_merge_atomic(dst_dclass,
									   dst_values,
									   GPUPREAGG_ACCUM_MAP_GLOBAL,
									   l_dclass,
									   l_values,
									   GPUPREAGG_ACCUM_MAP_LOCAL);
			else
				gpupreagg_update_atomic(dst_dclass,
										dst_values,
										GPUPREAGG_ACCUM_MAP_GLOBAL,
										KERN_DATA_STORE_DCLASS(kds_slot, kds_index),
										KERN_DATA_STORE_VALUES(kds_slot, kds_index),
										GPUPREAGG_ACCUM_MAP_GLOBAL);
			if (is_locked)
				atomicExch(&f_hash->slots[hindex], next);	//UNLOCK
			l_final_buffer_modified = true;
			return true;
		}
		curr = hitem->next;
	}

	/*
	 * Step-2: Ensure that f_hash has no entry under the lock
	 */
	if (!is_locked)
	{
		curr = next = __volatileRead(&f_hash->slots[hindex]);
		__threadfence();
		if (curr == HASHITEM_LOCKED ||
			atomicCAS(&f_hash->slots[hindex],
					  curr,
					  HASHITEM_LOCKED) != curr)
			return false;	/* already locked, try again */
		is_locked = true;
		goto restart;
	}

	/*
	 * Step-3: create a slot on kds_final
	 */
	dst_index = gpupreagg_create_final_slot(kcxt,
											kds_final,
											kds_slot,
											kds_index,
											l_dclass,
											l_values);
	if (dst_index == UINT_MAX)
	{
		/* likely, out of memory */
		atomicExch(&f_hash->slots[hindex], next);	//UNLOCK
		return false;
	}

	/*
	 * Step-4: allocation of hash entry
	 */
	curr = atomicAdd(&f_hash->usage, 1);
	if (offsetof(kern_global_hashslot, slots[f_hash->nslots]) +
		sizeof(preagg_hash_item) * (curr + 1) >= f_hash->length)
	{
		STROM_EREPORT(kcxt, ERRCODE_OUT_OF_MEMORY, "out of memory");
		atomicExch(&f_hash->slots[hindex], next);	//UNLOCK
		return false;
	}
	hitem = GLOBAL_HASHSLOT_GETITEM(f_hash, curr);
	hitem->index = dst_index;
	hitem->hash = hash;
	hitem->next = next;

	/*
	 * NOTE: Above modification to kds_final/f_hash are weakly-ordered memory
	 * writes, thus, updates on the hitem and kds_final may not be visible to
	 * other threads in the device.
	 * __threadfence() ensures any writes prior to the invocation are visible
	 * to other threads. Don't eliminate this.
	 */
	__threadfence();

	atomicExch(&f_hash->slots[hindex], curr);		//UNLOCK;

	return true;
}

/*
 * gpupreagg_local_reduction
 *
 *
 */
STATIC_INLINE(int)
gpupreagg_local_reduction(kern_context *kcxt,
						  kern_data_store *kds_slot,
						  cl_uint		index,
						  cl_uint		hash,
						  preagg_local_hashtable *l_htable,
						  preagg_hash_item *l_hitems,
						  cl_char     *l_dclass,	/* __shared__ */
						  Datum       *l_values,	/* __shared__ */
						  char        *l_extras)	/* __shared__ */
{
	cl_uint		hindex = hash % GPUPREAGG_LOCAL_HASH_NSLOTS;
	cl_uint		curr;
	cl_uint		next;
	cl_bool		is_locked = false;

	curr = next = __volatileRead(&l_htable->l_hslots[hindex]);
	__threadfence_block();
	if (curr == HASHITEM_LOCKED)
		return -1;	/* locked */
restart:
	while (curr < GPUPREAGG_LOCAL_HASH_NROOMS)
	{
		preagg_hash_item   *hitem = &l_hitems[curr];

		if (hitem->hash == hash &&
			gpupreagg_keymatch(kcxt,
							   kds_slot, index,
							   kds_slot, hitem->index))
		{
			if (is_locked)
				atomicExch(&l_htable->l_hslots[hindex], next);	//UNLOCK
			goto found;
		}
		curr = hitem->next;
	}
	assert(curr == HASHITEM_EMPTY);

	if (__volatileRead(&l_htable->nitems) >= GPUPREAGG_LOCAL_HASH_NROOMS)
	{
		/*
		 * Here we could not find out the entry on the local hash-table,
		 * but obviously no space on the local hash-table also.
		 * In this case, thread goes to the second path for the global-to-
		 * global reduction.
		 */
		if (is_locked)
			atomicExch(&l_htable->l_hslots[hindex], next);	//UNLOCK
		return 0;	/* not found */
	}
	assert(l_hitems && l_dclass && l_values);

	/*
	 * Begin critical section
	 */
	if (!is_locked)
	{
	    curr = next = __volatileRead(&l_htable->l_hslots[hindex]);
		__threadfence_block();
		if (curr == HASHITEM_LOCKED ||
			atomicCAS(&l_htable->l_hslots[hindex],
					  next,
					  HASHITEM_LOCKED) != next)
			return -1;	/* lock contension, retry again. */
		is_locked = true;
		goto restart;
	}
	curr = atomicAdd(&l_htable->nitems, 1);
	if (curr >= GPUPREAGG_LOCAL_HASH_NROOMS)
	{
		/*
		 * Oops, the local hash-table has no space to save a new
		 * entry any more. So, unlock the slot, then return to
		 * the caller to go to the second path for the global-to-
		 * global reduction.
		 */
		atomicExch(&l_htable->l_hslots[hindex], next);	//UNLOCK
		return 0;	/* not found */
	}

	/*
	 * initial allocation of the hash-item that is allocated above.
	 */
	l_hitems[curr].index = index;
	l_hitems[curr].hash  = hash;
	l_hitems[curr].next  = next;

	if (l_extras)
		l_extras += GPUPREAGG_ACCUM_EXTRA_BUFSZ * curr;
	gpupreagg_init_local_slot(l_dclass + GPUPREAGG_NUM_ACCUM_VALUES * curr,
							  l_values + GPUPREAGG_NUM_ACCUM_VALUES * curr,
							  l_extras);
	/*
	 * __threadfence_block() makes above updates visible to other concurent
	 * threads within this block.
	 */
	__threadfence_block();
	/* UNLOCK */
	atomicExch(&l_htable->l_hslots[hindex], curr);
found:
	/* Runs global-to-local reduction */
	gpupreagg_update_atomic(l_dclass + GPUPREAGG_NUM_ACCUM_VALUES * curr,
							l_values + GPUPREAGG_NUM_ACCUM_VALUES * curr,
							GPUPREAGG_ACCUM_MAP_LOCAL,
							KERN_DATA_STORE_DCLASS(kds_slot, index),
							KERN_DATA_STORE_VALUES(kds_slot, index),
							GPUPREAGG_ACCUM_MAP_GLOBAL);
	return 1;	/* ok, merged */
}

/*
 * gpupreagg_group_reduction
 */
DEVICE_FUNCTION(void)
gpupreagg_groupby_reduction(kern_context *kcxt,
							kern_gpupreagg *kgpreagg,		/* in/out */
							kern_errorbuf *kgjoin_errorbuf,	/* in */
							kern_data_store *kds_slot,		/* in */
							kern_data_store *kds_final,		/* out */
							kern_global_hashslot *f_hash,	/* out */
							preagg_hash_item *l_hitems,     /* __shared__ */
							cl_char    *l_dclass,           /* __shared__ */
							Datum      *l_values,           /* __shared__ */
							char	   *l_extras)			/* __shared__ */
{
	cl_bool		is_last_reduction = false;
	cl_uint		l_nitems;
	__shared__ preagg_local_hashtable l_htable;
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
	if (get_global_id() == 0)
		kgpreagg->setup_slot_done = true;

	/*
	 * setup local hash-table
	 */
	if (get_local_id() == 0)
	{
		l_final_buffer_modified = false;
		l_htable.nitems = 0;
	}
	for (int i = get_local_id(); i < GPUPREAGG_LOCAL_HASH_NSLOTS; i += get_local_size())
		l_htable.l_hslots[i] = HASHITEM_EMPTY;
	__syncthreads();

	/*
	 * main loop for the local/global hybrid reduction
	 */
	do {
		cl_uint		hash = UINT_MAX;
		int			status;
		int			index;
		int			count;

		/* fetch next items from the kds_slot */
		if (get_local_id() == 0)
			base = atomicAdd(&kgpreagg->read_slot_pos, get_local_size());
		__syncthreads();
		if (base >= kds_slot->nitems)
			break;
		if (base + get_local_size() >= kds_slot->nitems)
			is_last_reduction = true;

		/* calculation of the hash-value of the item */
		index = base + get_local_id();
		if (index < kds_slot->nitems)
		{
			cl_char	   *__dclass = KERN_DATA_STORE_DCLASS(kds_slot, index);
			Datum	   *__values = KERN_DATA_STORE_VALUES(kds_slot, index);

			hash = gpupreagg_hashvalue(kcxt, __dclass, __values);
		}
		if (__syncthreads_count(kcxt->errcode) > 0)
			return;		/* error */

		/*
		 * 1st path - try local reduction
		 */
		status = -1;
		do {
			if (status < 0 && index < kds_slot->nitems)
				status = gpupreagg_local_reduction(kcxt,
												   kds_slot,
												   index,
												   hash,
												   &l_htable,
												   l_hitems,
												   l_dclass,
												   l_values,
												   l_extras);
			else
				status = 1;

			if (__syncthreads_count(kcxt->errcode) > 0)
				return;		/* error */
		} while (__syncthreads_count(status < 0) > 0);

		/*
		 * 2nd path - try global reduction
		 */
		assert(status >= 0);
		while ((count = __syncthreads_count(status == 0)) > 0)
		{
			if (gpupreagg_expand_global_hash(kcxt, f_hash, count))
			{
				if (status == 0)
				{
					assert(index < kds_slot->nitems);

					if (gpupreagg_global_reduction(kcxt,
												   kds_slot,
												   index,
												   hash,
												   kds_final,
												   f_hash,
												   NULL,
												   NULL))
						status = 1;		/* successfully, merged */
				}
				/* unlock global hash slots */
				__syncthreads();
				if (get_local_id() == 0)
					atomicSub(&f_hash->lock, 2);
			}
			/* quick bailout on error */
			if (__syncthreads_count(kcxt->errcode) > 0)
				return;
		}
	} while (!is_last_reduction);

	__syncthreads();

	/*
	 * last path - flush pending local reductions
	 */
	l_nitems = Min(l_htable.nitems, GPUPREAGG_LOCAL_HASH_NROOMS);
	for (cl_uint i = 0; i < l_nitems; i += get_local_size())
	{
		cl_uint		j = i + get_local_id();
		cl_int		status = 0;
		cl_int		count;

		while ((count = __syncthreads_count(!status && j < l_nitems)) > 0)
		{
			if (gpupreagg_expand_global_hash(kcxt, f_hash, count))
			{
				if (!status && j < l_nitems)
				{
					preagg_hash_item *hitem = &l_hitems[j];
					cl_char    *my_dclass = l_dclass + GPUPREAGG_NUM_ACCUM_VALUES * j;
					Datum      *my_values = l_values + GPUPREAGG_NUM_ACCUM_VALUES * j;

					if (gpupreagg_global_reduction(kcxt,
												   kds_slot,
												   hitem->index,
												   hitem->hash,
												   kds_final,
												   f_hash,
												   my_dclass,
												   my_values))
						status = 1;		/* merged */
				}
				else
				{
					status = 1;
				}
                /* unlock global hash slots */
                __syncthreads();
                if (get_local_id() == 0)
                    atomicSub(&f_hash->lock, 2);
            }
            /* quick bailout on error */
            if (__syncthreads_count(kcxt->errcode) > 0)
                return;
		}
	}
	__syncthreads();
	if (get_local_id() == 0 && l_final_buffer_modified)
		kgpreagg->final_buffer_modified = true;
}

/*
 * aggcalc operations for hyper-log-log
 */
DEVICE_FUNCTION(void)
aggcalc_init_hll_sketch(cl_char *p_accum_dclass,
						Datum   *p_accum_datum,
						char    *extra_pos)
{
	cl_uint		sz = VARHDRSZ + (1U << GPUPREAGG_HLL_REGISTER_BITS);

	*p_accum_dclass = DATUM_CLASS__NULL;
	memset(extra_pos, 0, sz);
	SET_VARSIZE(extra_pos, sz);
	*p_accum_datum = PointerGetDatum(extra_pos);
}

DEVICE_FUNCTION(void)
aggcalc_shuffle_hll_sketch(cl_char *p_accum_dclass,
						   Datum   *p_accum_datum,
						   int      lane_id)
{
	cl_char		my_dclass;
	cl_char		buddy_dclass;
	varlena	   *hll_state = (varlena *)DatumGetPointer(*p_accum_datum);
	cl_uint	   *hll_regs = (cl_uint *)VARDATA(hll_state);
	cl_uint		nrooms = (1U << GPUPREAGG_HLL_REGISTER_BITS);
	cl_uint		index;

	assert(VARSIZE_EXHDR(hll_state) == nrooms);
	assert(__activemask() == ~0U);
	my_dclass = *p_accum_dclass;
	buddy_dclass = __shfl_sync(__activemask(), my_dclass, lane_id);

	nrooms /= sizeof(cl_uint);
	for (index=0; index < nrooms; index++)
	{
		union {
			cl_uchar	regs[4];
			cl_uint		v32;
		}	myself, buddy;

		myself.v32 = hll_regs[index];
		buddy.v32 = __shfl_sync(__activemask(), myself.v32, lane_id);
		if (my_dclass == DATUM_CLASS__NULL)
		{
			if (buddy_dclass != DATUM_CLASS__NULL)
			{
				hll_regs[index] = buddy.v32;
				*p_accum_dclass = DATUM_CLASS__NORMAL;
			}
		}
		else
		{
			assert(my_dclass == DATUM_CLASS__NORMAL);
			if (buddy_dclass != DATUM_CLASS__NULL)
			{
				assert(buddy_dclass == DATUM_CLASS__NORMAL);
				if (myself.regs[0] < buddy.regs[0])
					myself.regs[0] = buddy.regs[0];
				if (myself.regs[1] < buddy.regs[1])
					myself.regs[1] = buddy.regs[1];
				if (myself.regs[2] < buddy.regs[2])
					myself.regs[2] = buddy.regs[2];
				if (myself.regs[3] < buddy.regs[3])
					myself.regs[3] = buddy.regs[3];
				hll_regs[index] = myself.v32;
			}
		}
	}
}

DEVICE_FUNCTION(void)
aggcalc_normal_hll_sketch(cl_char *p_accum_dclass,
						  Datum   *p_accum_datum,
						  cl_char  newval_dclass,
						  Datum    newval_datum)	/* = int8 hash */
{
	cl_uint		nrooms = (1U << GPUPREAGG_HLL_REGISTER_BITS);
	cl_uint		index;
	cl_uint		count;
	cl_char	   *hll_regs;

	if (newval_dclass != DATUM_CLASS__NULL)
	{
		assert(newval_dclass == DATUM_CLASS__NORMAL);

		
		index = (newval_datum & (nrooms - 1));
		count = __clzll(__brevll(newval_datum >> GPUPREAGG_HLL_REGISTER_BITS)) + 1;
		hll_regs = VARDATA(*p_accum_datum);
		if (hll_regs[index] < count)
			hll_regs[index] = count;
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
}

DEVICE_FUNCTION(void)
aggcalc_merge_hll_sketch(cl_char *p_accum_dclass,
						 Datum   *p_accum_datum,
						 cl_char  newval_dclass,
						 Datum    newval_datum)		/* =bytea sketch */
{
	if (newval_dclass != DATUM_CLASS__NULL)
	{
		cl_uint	   *dst_regs = (cl_uint *)VARDATA(*p_accum_datum);
		cl_uint	   *new_regs = (cl_uint *)VARDATA(newval_datum);
		cl_uint		nrooms = (1U << GPUPREAGG_HLL_REGISTER_BITS);
		cl_uint		index;

		assert(newval_dclass == DATUM_CLASS__NORMAL);
		assert(VARSIZE_EXHDR(*p_accum_datum) == nrooms &&
			   VARSIZE_EXHDR(newval_datum) == nrooms);
		nrooms /= sizeof(cl_uint);
		for (index=0; index < nrooms; index++)
		{
			union {
				cl_uchar	regs[4];
				cl_uint		v32;
			}	oldval, curval, newval, tmpval;

			tmpval.v32 = __volatileRead(&new_regs[index]);
			curval.v32 = __volatileRead(&dst_regs[index]);
			do {
				newval = oldval = curval;
				if (newval.regs[0] < tmpval.regs[0])
					newval.regs[0] = tmpval.regs[0];
				if (newval.regs[1] < tmpval.regs[1])
					newval.regs[1] = tmpval.regs[1];
				if (newval.regs[2] < tmpval.regs[2])
					newval.regs[2] = tmpval.regs[2];
				if (newval.regs[3] < tmpval.regs[3])
					newval.regs[3] = tmpval.regs[3];
				if (newval.v32 == curval.v32)
					break;
			} while ((curval.v32 = atomicCAS(&dst_regs[index],
											 oldval.v32,
											 newval.v32)) != oldval.v32);
		}
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
}

DEVICE_FUNCTION(void)
aggcalc_update_hll_sketch(cl_char *p_accum_dclass,
						  Datum   *p_accum_datum,
						  cl_char  newval_dclass,
						  Datum    newval_datum)	/* =int8 hash */
{
	cl_uint		nrooms = (1U << GPUPREAGG_HLL_REGISTER_BITS);
	cl_uint		index;
	cl_uint		count;
	cl_uint	   *hll_regs;

	if (newval_dclass != DATUM_CLASS__NULL)
	{
		union {
			cl_uchar	regs[4];
			cl_uint		v32;
		}		oldval, curval, newval;

		assert(newval_dclass == DATUM_CLASS__NORMAL);

		index = (newval_datum & (nrooms - 1));
		count = __clzll(__brevll(newval_datum >> GPUPREAGG_HLL_REGISTER_BITS)) + 1;
		hll_regs = (cl_uint *)VARDATA(*p_accum_datum);
		hll_regs += (index >> 2);
		index &= 3;

		curval.v32 = __volatileRead(hll_regs);
		do {
			if (count <= curval.regs[index])
				break;
			newval = oldval = curval;
			newval.regs[index] = count;
		} while ((curval.v32 = atomicCAS(hll_regs,
										 oldval.v32,
										 newval.v32)) != oldval.v32);
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
}
