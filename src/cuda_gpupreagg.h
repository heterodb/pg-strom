/*
 * cuda_gpupreagg.h
 *
 * Preprocess of aggregate using GPU acceleration, to reduce number of
 * rows to be processed by CPU; including the Sort reduction.
 * --
 * Copyright 2011-2018 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2018 (C) The PG-Strom Development Team
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
#ifndef CUDA_GPUPREAGG_H
#define CUDA_GPUPREAGG_H

/*
 * +--------------------+
 * | kern_gpupreagg     |
 * |     :              |
 * | +------------------+
 * | | kern_parambuf    |
 * | |    :             |
 * +-+------------------+
 */
struct kern_gpupreagg
{
	kern_errorbuf	kerror;				/* kernel error information */
	cl_uint			num_group_keys;		/* nogroup reduction, if 0 */
	cl_uint			read_src_pos;		/* offset to read kds_src */
	cl_uint			read_slot_pos;		/* offset to read kds_slot */
	/* -- suspend/resume (KDS_FORMAT_BLOCK) */
	cl_uint			suspend_count;		/* number of suspended blocks */
	cl_uint			suspend_offset;		/* offset to suspend buffer, if any */
	cl_uint			lpp_array_offset;	/* offset to lpp_array, if any */

	/* -- runtime statistics -- */
	cl_uint			nitems_real;		/* out: # of outer input rows */
	cl_uint			nitems_filtered;	/* out: # of removed rows by quals */
	cl_uint			num_conflicts;		/* only used in kernel space */
	cl_uint			num_groups;			/* out: # of new groups */
	cl_uint			extra_usage;		/* out: size of new allocation */
	cl_uint			ghash_conflicts;	/* out: # of ghash conflicts */
	cl_uint			fhash_conflicts;	/* out: # of fhash conflicts */
	/* -- debug counter -- */
	cl_ulong		tv_stat_debug1;		/* out: debug counter 1 */
	cl_ulong		tv_stat_debug2;		/* out: debug counter 2 */
	cl_ulong		tv_stat_debug3;		/* out: debug counter 3 */
	cl_ulong		tv_stat_debug4;		/* out: debug counter 4 */
	/* -- other hashing parameters -- */
	cl_uint			key_dist_salt;			/* hashkey distribution salt */
	cl_uint			hash_size;				/* size of global hash-slots */
	cl_uint			pg_crc32_table[256];	/* master CRC32 table */
	kern_parambuf	kparams;
};
typedef struct kern_gpupreagg	kern_gpupreagg;

/*
 * gpupreagg_suspend_context is used to suspend gpupreagg_setup_block kernel.
 * Because KDS_FORMAT_BLOCK can have more items than estimation, so we cannot
 * avoid overflow of @kds_slot buffer preliminary. If @nitems exceeds @nrooms,
 * gpupreagg_setup_block will exit immediately, and save the current context
 * on the gpupreagg_suspend_context array to resume later.
 */
typedef struct
{
	cl_uint	part_index;
	cl_uint	line_index;
} gpupreagg_suspend_context;

/* macro definitions to reference packed values */
#define KERN_GPUPREAGG_PARAMBUF(kgpreagg)				\
	(&(kgpreagg)->kparams)
#define KERN_GPUPREAGG_PARAMBUF_LENGTH(kgpreagg)		\
	((kgpreagg)->kparams.length)
#define KERN_GPUPREAGG_LENGTH(kgpreagg)					\
	(offsetof(kern_gpupreagg, kparams) +				\
	 KERN_GPUPREAGG_PARAMBUF_LENGTH(kgpreagg))
#define KERN_GPUPREAGG_DMASEND_LENGTH(kgpreagg)			\
	(offsetof(kern_gpupreagg, kparams) +				\
	 KERN_GPUPREAGG_PARAMBUF_LENGTH(kgpreagg))
#define KERN_GPUPREAGG_DMARECV_LENGTH(kgpreagg)			\
	offsetof(kern_gpupreagg, pg_crc32_table[0])
/* suspend/resume buffer for KDS_FORMAT_BLOCK */
#define KERN_GPUPREAGG_SUSPEND_CONTEXT(kgpreagg)		\
	((gpupreagg_suspend_context *)						\
	 ((kgpreagg)->suspend_offset == 0					\
	  ? NULL											\
	  : ((char *)(kgpreagg) + (kgpreagg)->suspend_offset)))

#define KERN_GPUPREAGG_ITEMIDDATA_ARRAY(kgpreagg)		\
	((ItemIdData **)									\
	 ((kgpreagg)->lpp_array_offset == 0					\
	  ? NULL											\
	  : ((char *)(kgpreagg) + (kgpreagg)->lpp_array_offset)))

/*
 * NOTE: hashtable of gpupreagg is an array of pagg_hashslot.
 * It contains a pair of hash value and get_local_id(0) of responsible
 * thread if local reduction, or index on the kern_data_store if global
 * reduction.
 * On hashtable construction phase, it fetches an item from the hash-
 * slot using hash % get_local_size(0) or hash % hash_size.
 * Then, if it is empty, thread put a pair of its hash value and either
 * get_local_id(0) or kds_index according to the context. If not empty,
 * reduction function tries to merge the value, or makes advance the
 * slot if grouping key is not same.
 */
typedef union
{
	cl_ulong	value;	/* for 64bit atomic operation */
	struct {
		cl_uint	hash;	/* hash value of the entry */
		cl_uint	index;	/* loca/global thread-id that is responsible for */
	} s;
} pagg_hashslot;

/*
 * kern_global_hashslot
 *
 * An array of pagg_datum and its usage statistics, to be placed on
 * global memory area. Usage counter is used to break a loop to find-
 * out an empty slot if hash-slot is already filled-up.
 */
#define GLOBAL_HASHSLOT_THRESHOLD(g_hashsize)	\
	((size_t)(0.75 * (double)(g_hashsize)))

typedef struct
{
	cl_uint		lock;			/* lock when hash_size is expanded */
	cl_uint		hash_usage;		/* current number of hash_slot in use */
	cl_uint		hash_size;		/* current size of the hash_slot */
	cl_uint		hash_limit;		/* max limit of the hash_slot */
	pagg_hashslot hash_slot[FLEXIBLE_ARRAY_MEMBER];
} kern_global_hashslot;

/*
 * definition for special system parameter
 *
 * KPARAM_0 - array of the GPUPREAGG_FIELD_IS_* flags as cl_char[] array.
 * Each item informs usage of the related field.
 */
#define GPUPREAGG_FIELD_IS_GROUPKEY		1
#define GPUPREAGG_FIELD_IS_AGGFUNC		2

#ifdef __CUDACC__

/*
 * hash value calculation function - to be generated by PG-Strom on the fly
 */
STATIC_FUNCTION(cl_uint)
gpupreagg_hashvalue(kern_context *kcxt,
					cl_uint *crc32_table,	/* __shared__ memory */
					cl_uint hash_value,
					cl_bool *slot_isnull,
					Datum *slot_values);

/*
 * comparison function - to be generated by PG-Strom on the fly
 *
 * It compares two records indexed by 'x_index' and 'y_index' on the supplied
 * kern_data_store, then returns -1 if record[X] is less than record[Y],
 * 0 if record[X] is equivalent to record[Y], or 1 if record[X] is greater
 * than record[Y].
 * (auto generated function)
 */
STATIC_FUNCTION(cl_bool)
gpupreagg_keymatch(kern_context *kcxt,
				   kern_data_store *x_kds, size_t x_index,
				   kern_data_store *y_kds, size_t y_index);

/*
 * nogroup calculation function - to be generated by PG-Strom on the fly
 */
STATIC_FUNCTION(void)
gpupreagg_nogroup_calc(cl_int attnum,
					   cl_bool *p_accum_isnull,
					   Datum   *p_accum_datum,
					   cl_bool  newval_isnull,
					   Datum    newval_datum);


/*
 * local calculation function - to be generated by PG-Strom on the fly
 */
STATIC_FUNCTION(void)
gpupreagg_local_calc(cl_int attnum,
					 cl_bool *p_accum_isnull,
					 Datum   *p_accum_datum,
					 cl_bool  newval_isnull,
					 Datum    newval_datum);

/*
 * global calculation function - to be generated by PG-Strom on the fly
 */
STATIC_FUNCTION(void)
gpupreagg_global_calc(cl_bool *accum_isnull,
					  Datum   *accum_values,
					  cl_bool *newval_isnull,
					  Datum   *newval_values);
/*
 * translate a kern_data_store (input) into an output form
 * (auto generated function)
 */
STATIC_FUNCTION(void)
gpupreagg_projection_row(kern_context *kcxt,
						 kern_data_store *kds_src,	/* in */
						 HeapTupleHeaderData *htup,	/* in */
						 Datum *dst_values,			/* out */
						 cl_char *dst_isnull);		/* out */

STATIC_FUNCTION(void)
gpupreagg_projection_column(kern_context *kcxt,
							kern_data_store *kds_src,	/* in */
							cl_uint src_index,			/* out */
							Datum *dst_values,			/* out */
							cl_char *dst_isnull);		/* out */

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
	cl_char	   *src_isnull = KERN_DATA_STORE_ISNULL(kds_src, rowidx_src);
	cl_char	   *dst_isnull = KERN_DATA_STORE_ISNULL(kds_dst, rowidx_dst);
	cl_uint		i, ncols = kds_src->ncols;
	cl_int		alloc_size = 0;
	char	   *alloc_ptr = NULL;

	/* Paranoire checks? */
	assert(kds_src->format == KDS_FORMAT_SLOT &&
		   kds_dst->format == KDS_FORMAT_SLOT);
	assert(kds_src->ncols == kds_dst->ncols);
	assert(rowidx_src < kds_src->nitems);
	assert(rowidx_dst < kds_dst->nitems);

	/* size for allocation */
	for (i=0; i < ncols; i++)
	{
		kern_colmeta cmeta = kds_src->colmeta[i];

		if (src_isnull[i])
			continue;	/* no buffer needed */
		if (!cmeta.attbyval)
		{
			if (cmeta.attlen > 0)
				alloc_size += MAXALIGN(cmeta.attlen);
			else
			{
				char   *datum = DatumGetPointer(src_values[i]);

				alloc_size += MAXALIGN(VARSIZE_ANY(datum));
			}
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
			STROM_SET_ERROR(&kcxt->e, StromError_DataStoreNoSpace);
			/*
			 * NOTE: Uninitialized dst_values[] for NULL values will lead
			 * a problem around atomic operation, because we designed 
			 * reduction operation that assumes values are correctly
			 * initialized even if it is NULL.
			 */
			for (i=0; i < ncols; i++)
			{
				dst_isnull[i] = true;
				dst_values[i] = src_values[i];
			}
			return -1;
		}
		alloc_ptr = ((char *)kds_dst + kds_dst->length -
					 (usage_prev + alloc_size));
	}
	/* move the data */
	for (i=0; i < ncols; i++)
	{
		kern_colmeta cmeta = kds_src->colmeta[i];

		if (src_isnull[i])
		{
			dst_isnull[i] = true;
			dst_values[i] = src_values[i];
		}
		else if (cmeta.attbyval)
		{
			dst_isnull[i] = false;
			dst_values[i] = src_values[i];
		}
		else
		{
			char	   *datum = DatumGetPointer(src_values[i]);
			cl_uint		datum_len = (cmeta.attlen > 0 ?
									 cmeta.attlen :
									 VARSIZE_ANY(datum));
			memcpy(alloc_ptr, datum, datum_len);
			dst_isnull[i] = false;
			dst_values[i] = PointerGetDatum(alloc_ptr);

			alloc_ptr += MAXALIGN(datum_len);
		}
	}
	return alloc_size;
}

/*
 * gpupreagg_setup_row
 */
KERNEL_FUNCTION(void)
gpupreagg_setup_row(kern_gpupreagg *kgpreagg,
					kern_data_store *kds_src,	/* in: KDS_FORMAT_ROW */
					kern_data_store *kds_slot)	/* out: KDS_FORMAT_SLOT */
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_context	kcxt;
	kern_tupitem   *tupitem;
	cl_uint			src_nitems = kds_src->nitems;
	cl_uint			src_index;
	cl_uint			slot_index;
	cl_uint			offset;
	cl_uint			count;
	cl_uint			nvalids;
	Datum		   *slot_values;
	cl_bool		   *slot_isnull;
	cl_bool			try_next_window = true;
	cl_bool			rc;
	__shared__ cl_uint	base;

	INIT_KERNEL_CONTEXT(&kcxt, gpupreagg_setup_row, kparams);
	do {
		if (get_local_id() == 0)
			base = atomicAdd(&kgpreagg->read_src_pos, get_local_size());
		__syncthreads();

		if (base + get_local_size() >= src_nitems)
			try_next_window = false;
		if (base >= src_nitems)
			break;

		src_index = base + get_local_id();
		if (src_index < src_nitems)
		{
			tupitem = KERN_DATA_STORE_TUPITEM(kds_src, src_index);
#ifdef GPUPREAGG_PULLUP_OUTER_SCAN
			rc = gpuscan_quals_eval(&kcxt, kds_src,
									&tupitem->t_self,
									&tupitem->htup);
#else
			rc = true;
#endif
		}
		else
		{
			tupitem = NULL;
			rc = false;
		}

#ifdef GPUPREAGG_PULLUP_OUTER_SCAN
		/* Bailout if any error */
		if (__syncthreads_count(kcxt.e.errcode) > 0)
			break;
#endif
		/* allocation of kds_slot buffer, if any */
		offset = pgstromStairlikeBinaryCount(tupitem && rc, &nvalids);
		if (nvalids > 0)
		{
			if (get_local_id() == 0)
				base = atomicAdd(&kds_slot->nitems, nvalids);
			__syncthreads();
			if (base + nvalids > kds_slot->nrooms)
			{
				STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
				break;
			}
			slot_index = base + offset;
			if (tupitem && rc)
			{
				slot_values = KERN_DATA_STORE_VALUES(kds_slot, slot_index);
				slot_isnull = KERN_DATA_STORE_ISNULL(kds_slot, slot_index);

				gpupreagg_projection_row(&kcxt,
										 kds_src,
										 &tupitem->htup,
										 slot_values,
										 slot_isnull);
			}
		}
		/* bailout if any error */
		if (__syncthreads_count(kcxt.e.errcode) > 0)
			break;
		/* update statistics */
		pgstromStairlikeBinaryCount(tupitem ? 1 : 0, &count);
		if (get_local_id() == 0)
		{
			atomicAdd(&kgpreagg->nitems_real, count);
			atomicAdd(&kgpreagg->nitems_filtered, count - nvalids);
		}
	} while(try_next_window);

	/* write back error status if any */
	kern_writeback_error_status(&kgpreagg->kerror, &kcxt.e);
}

#ifdef GPUPREAGG_PULLUP_OUTER_SCAN
KERNEL_FUNCTION(void)
gpupreagg_setup_block(kern_gpupreagg *kgpreagg,
					  kern_data_store *kds_src,
					  kern_data_store *kds_slot)
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_context	kcxt;
	cl_uint			window_sz;
	cl_uint			part_sz;
	cl_uint			n_parts;
	cl_uint			count;
	cl_uint			offset;
	cl_uint			part_index = 0;
	cl_uint			line_index = 0;
	cl_bool			thread_is_valid = false;
	gpupreagg_suspend_context *my_suspend;
	ItemIdData	  **lpp_array;
	__shared__ cl_uint	base;

	INIT_KERNEL_CONTEXT(&kcxt, gpupreagg_setup_block, kparams);

	assert(kds_src->format == KDS_FORMAT_BLOCK &&
		   kds_slot->format == KDS_FORMAT_SLOT);

	part_sz = Min((kds_src->nrows_per_block +
				   warpSize-1) & ~(warpSize-1), get_local_size());
	n_parts = get_local_size() / part_sz;
	if (get_local_id() < part_sz * n_parts)
		thread_is_valid = true;
	window_sz = n_parts * get_num_groups();

	/* resume kernel from the point where suspended, if any */
	my_suspend = KERN_GPUPREAGG_SUSPEND_CONTEXT(kgpreagg) + get_global_index();
	lpp_array = KERN_GPUPREAGG_ITEMIDDATA_ARRAY(kgpreagg);
	if (kgpreagg->suspend_count > 0)
	{
		part_index = my_suspend->part_index;
		line_index = my_suspend->line_index;
	}
	__syncthreads();

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

		part_base = part_index * window_sz + get_global_index() * n_parts;
		if (part_base >= kds_src->nitems)
			break;
		part_id = get_local_id() / part_sz + part_base;
		line_no = get_local_id() % part_sz + line_index * part_sz;

		do {
			HeapTupleHeaderData *htup = NULL;
			ItemIdData *curr_lpp = NULL;
			cl_uint		slot_index;
			cl_bool	   *slot_isnull;
			Datum	   *slot_values;
			cl_bool		rc = false;

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
#ifdef GPUPREAGG_HAS_OUTER_QUALS
			if (htup)
				rc = gpuscan_quals_eval(&kcxt, kds_src, &t_self, htup);
			/* bailout if any errors */
			if (__syncthreads_count(kcxt.e.errcode) > 0)
				goto out;
#else
			rc = true;
#endif
			/* allocation of the kds_slot buffer */
			offset = pgstromStairlikeBinaryCount(htup && rc, &nvalids);
			if (nvalids > 0)
			{
				while (get_local_id() == 0)
				{
					cl_uint		old_nitems = kds_slot->nitems;
					cl_uint		new_nitems = old_nitems + nvalids;
					cl_uint		cur_nitems;

					if (new_nitems > kds_slot->nrooms)
					{
						base = UINT_MAX;
						break;
					}
					cur_nitems = atomicCAS(&kds_slot->nitems,
										   old_nitems, new_nitems);
					if (cur_nitems == old_nitems)
					{
						base = old_nitems;
						break;
					}
				}
				__syncthreads();

				/*
				 * No more space on the kds_slot buffer, so suspend the kernel
				 * and run reduction operations earlier.
				 */
				if (base == UINT_MAX)
				{
					if (get_local_id() == 0)
						atomicAdd(&kgpreagg->suspend_count, 1);
					goto out;
				}

				slot_index = base + offset;
				if (htup && rc)
				{
					assert(slot_index < kds_slot->nrooms);
					slot_values = KERN_DATA_STORE_VALUES(kds_slot, slot_index);
					slot_isnull = KERN_DATA_STORE_ISNULL(kds_slot, slot_index);

					gpupreagg_projection_row(&kcxt,
											 kds_src,
											 htup,
											 slot_values,
											 slot_isnull);
					/* for invalidation of ItemIdData */
					lpp_array[slot_index] = curr_lpp;
				}
				/* bailout if any errors */
				if (__syncthreads_count(kcxt.e.errcode) > 0)
					goto out;
			}
			/* update statistics */
#ifdef GPUPREAGG_HAS_OUTER_QUALS
			pgstromStairlikeBinaryCount(htup != NULL ? 1 : 0, &count);
#else
			count = nvalids;
#endif
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
		my_suspend->part_index = part_index;
		my_suspend->line_index = line_index;
	}
	/* write back error status if any */
	kern_writeback_error_status(&kgpreagg->kerror, &kcxt.e);
}
#endif	/* GPUPREAGG_PULLUP_OUTER_SCAN */

/*
 * gpupreagg_setup_column
 */
KERNEL_FUNCTION(void)
gpupreagg_setup_column(kern_gpupreagg *kgpreagg,
					   kern_data_store *kds_src,	/* in: KDS_FORMAT_COLUMN */
					   kern_data_store *kds_slot)	/* out: KDS_FORMAT_SLOT */
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_context	kcxt;
	cl_uint			src_nitems = kds_src->nitems;
	cl_uint			src_index;
	cl_uint			slot_index;
	cl_uint			offset;
	cl_uint			count;
	cl_uint			nvalids;
	Datum		   *slot_values;
	cl_bool		   *slot_isnull;
	cl_bool			try_next_window = true;
	cl_bool			rc;
	__shared__ cl_uint	base;

	INIT_KERNEL_CONTEXT(&kcxt, gpupreagg_setup_column, kparams);
	do {
		if (get_local_id() == 0)
			base = atomicAdd(&kgpreagg->read_src_pos, get_local_size());
		__syncthreads();

		if (base + get_local_size() >= src_nitems)
			try_next_window = false;
		if (base >= src_nitems)
			break;

		src_index = base + get_local_id();
		if (src_index < src_nitems)
		{
#ifdef GPUPREAGG_PULLUP_OUTER_SCAN
			rc = gpuscan_quals_eval_column(&kcxt, kds_src, src_index);
#else
			rc = true;
#endif
		}
		else
		{
			rc = false;
		}
#ifdef GPUPREAGG_PULLUP_OUTER_SCAN
		/* Bailout if any error */
		if (__syncthreads_count(kcxt.e.errcode) > 0)
			break;
#endif
		/* allocation of kds_slot buffer, if any */
		offset = pgstromStairlikeBinaryCount(rc ? 1 : 0, &nvalids);
		if (nvalids > 0)
		{
			if (get_local_id() == 0)
				base = atomicAdd(&kds_slot->nitems, nvalids);
			__syncthreads();
			if (base + nvalids > kds_slot->nrooms)
			{
				STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
				break;
			}
			slot_index = base + offset;
			if (rc)
			{
				slot_values = KERN_DATA_STORE_VALUES(kds_slot, slot_index);
				slot_isnull = KERN_DATA_STORE_ISNULL(kds_slot, slot_index);

				gpupreagg_projection_column(&kcxt,
											kds_src,
											src_index,
											slot_values,
											slot_isnull);
			}
		}
		/* bailout if any error */
		if (__syncthreads_count(kcxt.e.errcode) > 0)
			break;
		/* update statistics */
		pgstromStairlikeBinaryCount(rc, &count);
		if (get_local_id() == 0)
		{
			atomicAdd(&kgpreagg->nitems_real, count);
			atomicAdd(&kgpreagg->nitems_filtered, count - nvalids);
		}
	} while(try_next_window);

	/* write back error status if any */
	kern_writeback_error_status(&kgpreagg->kerror, &kcxt.e);
}

/*
 * gpupreagg_nogroup_reduction
 */
KERNEL_FUNCTION_MAXTHREADS(void)
gpupreagg_nogroup_reduction(kern_gpupreagg *kgpreagg,		/* in/out */
							kern_errorbuf *kgjoin_errorbuf,	/* in */
							kern_data_store *kds_slot,		/* in */
							kern_data_store *kds_final,		/* shared out */
							kern_global_hashslot *f_hash)	/* shared out */
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_context	kcxt;
	varlena		   *kparam_0 = (varlena *)kparam_get_value(kparams, 0);
	cl_char		   *attr_is_preagg = (cl_char *)VARDATA(kparam_0);
	cl_uint			slot_nitems = kds_slot->nitems;
	cl_uint			nvalids;
	cl_uint			slot_index;
	cl_int			index;
	cl_bool			is_last_reduction = false;
	cl_bool		   *slot_isnull;
	Datum		   *slot_values;
	ItemIdData	  **lpp_array = KERN_GPUPREAGG_ITEMIDDATA_ARRAY(kgpreagg);
	__shared__ cl_bool	l_isnull[MAXTHREADS_PER_BLOCK];
	__shared__ Datum	l_values[MAXTHREADS_PER_BLOCK];
	__shared__ cl_uint	base;

	/* skip if previous stage reported an error */
	if (kgjoin_errorbuf &&
		kgjoin_errorbuf->errcode != StromError_Success &&
		kgjoin_errorbuf->errcode != StromError_Suspend)
		return;
	if (kgpreagg->kerror.errcode != StromError_Success)
		return;

	assert(kgpreagg->num_group_keys == 0);
	assert(kds_slot->format == KDS_FORMAT_SLOT);
	assert(kds_final->format == KDS_FORMAT_SLOT);
	assert(kds_slot->ncols == kds_final->ncols);
	INIT_KERNEL_CONTEXT(&kcxt, gpupreagg_nogroup_reduction, kparams);

	do {
		/* fetch next items from the kds_slot */
		if (get_local_id() == 0)
			base = atomicAdd(&kgpreagg->read_slot_pos, get_local_size());
		__syncthreads();

		if (base + get_local_size() >= slot_nitems)
			is_last_reduction = true;
		if (base >= slot_nitems)
			break;
		nvalids = Min(slot_nitems - base, get_local_size());
		slot_index = base + get_local_id();
		assert(slot_index < slot_nitems || get_local_id() >= nvalids);

		/* reductions for each columns */
		slot_isnull = KERN_DATA_STORE_ISNULL(kds_slot, slot_index);
		slot_values = KERN_DATA_STORE_VALUES(kds_slot, slot_index);
		for (index=0; index < kds_slot->ncols; index++)
		{
			int		dist, buddy;

			/* do nothing, if attribute is not preagg-function */
			if (!attr_is_preagg[index])
				continue;
			/* load the value from kds_slot to local */
			if (get_local_id() < nvalids)
			{
				l_isnull[get_local_id()] = slot_isnull[index];
                l_values[get_local_id()] = slot_values[index];
			}
			__syncthreads();

			/* do reduction */
			for (dist=2, buddy=1; dist < 2 * nvalids; buddy=dist, dist *= 2)
			{
				if ((get_local_id() & (dist-1)) == 0 &&
					(get_local_id() + (buddy)) < nvalids)
				{
					gpupreagg_nogroup_calc(index,
										   &l_isnull[get_local_id()],
										   &l_values[get_local_id()],
										   l_isnull[get_local_id() + buddy],
										   l_values[get_local_id() + buddy]);
				}
				__syncthreads();
			}

			/* store this value to kds_slot from isnull/values */
			if (get_local_id() == 0)
			{
				slot_isnull[index] = l_isnull[get_local_id()];
				slot_values[index] = l_values[get_local_id()];
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
				memcpy(KERN_DATA_STORE_ISNULL(kds_final, 0),
					   KERN_DATA_STORE_ISNULL(kds_slot, slot_index),
					   sizeof(cl_char) * kds_slot->ncols);
				memcpy(KERN_DATA_STORE_VALUES(kds_final, 0),
					   KERN_DATA_STORE_VALUES(kds_slot, slot_index),
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
					KERN_DATA_STORE_ISNULL(kds_final, 0),
					KERN_DATA_STORE_VALUES(kds_final, 0),
					KERN_DATA_STORE_ISNULL(kds_slot, slot_index),
					KERN_DATA_STORE_VALUES(kds_slot, slot_index));
			}
			else
			{
				assert(cur_nitems == 0xffffffff);
				goto try_again;
			}
		}
		__syncthreads();
		/*
		 * Invalidation of ItemId - if and when suspended GPU kernel needs
		 * CPU fallback, rows already processed must be ignored.
		 */
		if (lpp_array && slot_index < slot_nitems)
			ItemIdSetUnused(lpp_array[slot_index]);
	} while (!is_last_reduction);

	/* write-back execution status into host side */
    kern_writeback_error_status(&kgpreagg->kerror, &kcxt.e);
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
	{
		cudaDeviceSynchronize();
		return false;
	}

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
		STROM_SET_ERROR(&kcxt->e, StromError_DataStoreNoSpace);
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
	{
		cudaDeviceSynchronize();
		goto out_unlock;
	}

	/* wait for completion of other shared-lock holder */
	for (;;)
	{
		if (get_local_id() == 0)
			lock_wait = (f_hash->lock != 0x0001 ? true : false);
		else
			lock_wait = false;
		if (__syncthreads_count(lock_wait) == 0)
			break;
		cudaDeviceSynchronize();
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
		STROM_SET_ERROR(&kcxt->e, StromError_DataStoreNoSpace);
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
		STROM_SET_ERROR(&kcxt->e, StromError_DataStoreNoSpace);
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
			STROM_SET_ERROR(&kcxt->e, StromError_DataStoreNoSpace);
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
			KERN_DATA_STORE_ISNULL(kds_final, cur_slot.s.index),
			KERN_DATA_STORE_VALUES(kds_final, cur_slot.s.index),
			KERN_DATA_STORE_ISNULL(kds_slot, slot_index),
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

/* keep shared memory consumption less than 32KB */
#define GPUPREAGG_LOCAL_HASHSIZE	1720

KERNEL_FUNCTION_MAXTHREADS(void)
gpupreagg_groupby_reduction(kern_gpupreagg *kgpreagg,		/* in/out */
							kern_errorbuf *kgjoin_errorbuf,	/* in */
							kern_data_store *kds_slot,		/* in */
							kern_data_store *kds_final,		/* shared out */
							kern_global_hashslot *f_hash)	/* shared out */
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	varlena		   *kparam_0 = (varlena *)kparam_get_value(kparams, 0);
	cl_char		   *attr_is_preagg = (cl_char *)VARDATA(kparam_0);
	kern_context	kcxt;
	pagg_hashslot	old_slot;
	pagg_hashslot	new_slot;
	pagg_hashslot	cur_slot;
	cl_bool		   *slot_isnull;
	Datum		   *slot_values;
	cl_uint			index;
	cl_uint			slot_nitems = kds_slot->nitems;
	cl_uint			count;
	cl_uint			kds_index;
	cl_uint			hash_value;
	cl_uint			owner_index;
	cl_bool			is_owner = false;
	cl_bool			is_last_reduction = false;
	ItemIdData	  **lpp_array = KERN_GPUPREAGG_ITEMIDDATA_ARRAY(kgpreagg);
	__shared__ cl_uint	crc32_table[256];
	__shared__ cl_bool	l_isnull[MAXTHREADS_PER_BLOCK];
	__shared__ Datum	l_values[MAXTHREADS_PER_BLOCK];
	__shared__ cl_int	l_kds_index[MAXTHREADS_PER_BLOCK];
	__shared__ pagg_hashslot l_hashslot[GPUPREAGG_LOCAL_HASHSIZE];
	__shared__ cl_uint	base;

	/* skip if previous stage reported an error */
	if (kgjoin_errorbuf &&
		kgjoin_errorbuf->errcode != StromError_Success &&
		kgjoin_errorbuf->errcode != StromError_Suspend)
		return;
	if (kgpreagg->kerror.errcode != StromError_Success)
		return;

	assert(kgpreagg->num_group_keys > 0);
	assert(kds_slot->format == KDS_FORMAT_SLOT);
	assert(kds_final->format == KDS_FORMAT_SLOT);
	INIT_KERNEL_CONTEXT(&kcxt, gpupreagg_groupby_reduction, kparams);
	/* setup crc32 table */
	for (index = get_local_id();
		 index < lengthof(crc32_table);
		 index += get_local_size())
		crc32_table[index] = kgpreagg->pg_crc32_table[index];
	__syncthreads();

clean_restart:
	/* setup local hashslot */
	for (index = get_local_id();
		 index < GPUPREAGG_LOCAL_HASHSIZE;
		 index += get_local_size())
	{
		l_hashslot[index].s.hash = 0;
		l_hashslot[index].s.index = (cl_uint)(0xffffffff);
	}
	__syncthreads();

	kds_index = INT_MAX;
	hash_value = ~0;
	is_owner = false;
	owner_index = INT_MAX;
	slot_isnull = NULL;
	slot_values = NULL;
	do {
		/* fetch next items from the kds_slot */
		index = pgstromStairlikeBinaryCount(!is_owner, &count);
		assert(count > 0);
		if (get_local_id() == 0)
			base = atomicAdd(&kgpreagg->read_slot_pos, count);
		__syncthreads();
		if (base + count >= slot_nitems)
			is_last_reduction = true;
		if (base >= slot_nitems)
			goto skip_local_reduction;
		/* Assign a kds_index if thread is not owner */
		if (!is_owner)
		{
			kds_index = base + index;
			if (kds_index < slot_nitems)
			{
				slot_isnull = KERN_DATA_STORE_ISNULL(kds_slot, kds_index);
				slot_values = KERN_DATA_STORE_VALUES(kds_slot, kds_index);
				INIT_LEGACY_CRC32(hash_value);
				hash_value = gpupreagg_hashvalue(&kcxt,
												 crc32_table,
												 hash_value,
												 slot_isnull,
												 slot_values);
				FIN_LEGACY_CRC32(hash_value);
				/*
				 * Invalidation of ItemId - if and when suspended GPU kernel
				 * needs CPU fallback, rows already processed must be ignored.
				 */
				if (lpp_array)
					ItemIdSetUnused(lpp_array[kds_index]);
			}
			else
			{
				slot_isnull = NULL;
				slot_values = NULL;
			}
			l_kds_index[get_local_id()] = kds_index;
		}
		else
			assert(l_kds_index[get_local_id()] == kds_index);
		/* error checks */
		if (__syncthreads_count(kcxt.e.errcode) > 0)
			goto bailout;

		/* Local hash-table lookup to get owner index */
		if (is_owner)
			assert(get_local_id() == owner_index);
		else if (kds_index < slot_nitems)
		{
			new_slot.s.hash		= hash_value;
			new_slot.s.index	= get_local_id();
			old_slot.s.hash		= 0;
			old_slot.s.index	= (cl_uint)(0xffffffff);
			index = hash_value % GPUPREAGG_LOCAL_HASHSIZE;

		lhash_next:
			cur_slot.value = atomicCAS(&l_hashslot[index].value,
									   old_slot.value,
									   new_slot.value);
			if (cur_slot.value == old_slot.value)
			{
				/* hashslot was empty, so thread should own this slot */
				owner_index = new_slot.s.index;
				is_owner = true;
			}
			else
			{
				cl_uint		buddy_index = l_kds_index[cur_slot.s.index];

				assert(cur_slot.s.index < get_local_size());
				assert(buddy_index < slot_nitems);
				if (cur_slot.s.hash == hash_value &&
					gpupreagg_keymatch(&kcxt,
									   kds_slot, kds_index,
									   kds_slot, buddy_index))
				{
					owner_index = cur_slot.s.index;
				}
				else
				{
					index = (index + 1) % GPUPREAGG_LOCAL_HASHSIZE;
					if (kcxt.e.errcode == StromError_Success)
						goto lhash_next;
				}
			}
		}
		else
			owner_index = INT_MAX;
		/* error checks */
		if (__syncthreads_count(kcxt.e.errcode) > 0)
			goto bailout;

		/* Local reduction for each column */
		for (index=0; index < kds_slot->ncols; index++)
		{
			if (!attr_is_preagg[index])
				continue;
			/* load the value to local storage */
			if (kds_index < slot_nitems)
			{
				l_isnull[get_local_id()] = slot_isnull[index];
				l_values[get_local_id()] = slot_values[index];
			}
			__syncthreads();

			/* reduction by atomic operation */
			if (!is_owner && kds_index < slot_nitems)
			{
				assert(owner_index < get_local_size());
				gpupreagg_local_calc(index,
									 &l_isnull[owner_index],
									 &l_values[owner_index],
									 l_isnull[get_local_id()],
									 l_values[get_local_id()]);
			}
			__syncthreads();

			/* move the aggregation value */
			if (is_owner)
			{
				assert(get_local_id() == owner_index);
				slot_isnull[index] = l_isnull[owner_index];
				slot_values[index] = l_values[owner_index];
			}
			__syncthreads();
		}
	skip_local_reduction:
		/*
		 * final reduction steps if needed
		 */
		count = __syncthreads_count(is_owner);
		if (is_last_reduction || count > get_local_size() / 8)
		{
			cl_bool		lock_wait;

			do {
				/*
				 * Get shared-lock of the final hash-slot
				 * If it may have overflow, expand length of the hash-slot
				 * on demand, under its exclusive lock.
				 */
				if (!gpupreagg_expand_final_hash(&kcxt, f_hash, count))
					lock_wait = true;
				else
				{
					lock_wait = false;
					if (is_owner)
					{
						if (gpupreagg_final_reduction(&kcxt,
													  kgpreagg,
													  kds_slot,
													  kds_index,
													  hash_value,
													  kds_final,
													  f_hash))
							is_owner = false;
						else
							lock_wait = true;
					}
					__syncthreads();
					/* release shared lock of the final hash-slot */
					if (get_local_id() == 0)
						atomicSub(&f_hash->lock, 2);
				}
				/* quick bailout on error */
				if (__syncthreads_count(kcxt.e.errcode) > 0)
					goto bailout;
				count = __syncthreads_count(is_owner);
			} while (__syncthreads_count(lock_wait) > 0);
			/* OK, successfully moved pending items */
			if (!is_last_reduction)
				goto clean_restart;
		}
	} while(!is_last_reduction);
bailout:
	/* write-back execution status into host side */
	kern_writeback_error_status(&kgpreagg->kerror, &kcxt.e);
}

/* ----------------------------------------------------------------
 *
 * A thin abstraction layer for atomic functions
 *
 * Due to hardware capability difference we have to implement
 * alternative one using atomicCAS operation. For simplification.
 * we wrap all the atomic functions using:
 *   pg_atomic_(min|max|add)_<type>(<type> *addr, <type> value) 
 *
 * ----------------------------------------------------------------
 */
STATIC_INLINE(void)
aggcalc_atomic_min_int(cl_bool *p_accum_isnull, Datum *p_accum_datum,
					   cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		cl_int	newval_int = (cl_int)(newval_datum & 0xffffffffU);

		atomicMin((cl_int *)p_accum_datum, newval_int);
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_atomic_max_int(cl_bool *p_accum_isnull, Datum *p_accum_datum,
					   cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		cl_int	newval_int = (cl_int)(newval_datum & 0xffffffffU);

		atomicMax((cl_int *)p_accum_datum, newval_int);
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_atomic_add_int(cl_bool *p_accum_isnull, Datum *p_accum_datum,
					   cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		cl_int		newval_int = (cl_int)(newval_datum & 0xffffffff);

		atomicAdd((cl_int *)p_accum_datum, newval_int);
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_atomic_min_long(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		atomicMin((cl_long *)p_accum_datum, (cl_long)newval_datum);
		*p_accum_isnull = false;
	}
}


STATIC_INLINE(void)
aggcalc_atomic_max_long(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		atomicMax((cl_long *)p_accum_datum, (cl_long)newval_datum);
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_atomic_add_long(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		atomicAdd((cl_ulong *)p_accum_datum, (cl_ulong)newval_datum);
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_atomic_min_float(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						 cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		cl_uint		curval = *((cl_uint *)p_accum_datum);
		cl_uint		newval = (newval_datum & 0xffffffff);
		cl_uint		oldval;

		do {
			oldval = curval;
			if (__int_as_float(oldval) < __int_as_float(newval))
				break;
		} while ((curval = atomicCAS((cl_uint *)p_accum_datum,
									 oldval, newval)) != oldval);
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_atomic_max_float(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						 cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		cl_uint		curval = *((cl_uint *)p_accum_datum);
		cl_uint		newval = (newval_datum & 0xffffffff);
		cl_uint		oldval;

		do {
			oldval = curval;
			if (__int_as_float(oldval) > __int_as_float(newval))
				break;
		} while ((curval = atomicCAS((cl_uint *)p_accum_datum,
									 oldval, newval)) != oldval);
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_atomic_add_float(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						 cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		atomicAdd((cl_float *)p_accum_datum,
				  __int_as_float(newval_datum & 0xffffffff));
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_atomic_min_double(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						  cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		cl_ulong	curval = *((cl_ulong *)p_accum_datum);
		cl_ulong	newval = (cl_ulong)newval_datum;
		cl_ulong	oldval;

		do {
			oldval = curval;
			if (__longlong_as_double(oldval) < __longlong_as_double(newval))
				break;
		} while ((curval = atomicCAS((cl_ulong *)p_accum_datum,
									 oldval, newval)) != oldval);
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_atomic_max_double(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						  cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		cl_ulong	curval = *((cl_ulong *)p_accum_datum);
		cl_ulong	newval = (cl_ulong)newval_datum;
		cl_ulong	oldval;

		do {
			oldval = curval;
			if (__longlong_as_double(oldval) > __longlong_as_double(newval))
				break;
		} while ((curval = atomicCAS((cl_ulong *)p_accum_datum,
									 oldval, newval)) != oldval);
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_atomic_add_double(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						  cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		atomicAdd((cl_double *)p_accum_datum,
				  __longlong_as_double(newval_datum));
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_normal_min_int(cl_bool *p_accum_isnull, Datum *p_accum_datum,
					   cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		cl_int		newval_int = (cl_int)(newval_datum & 0xffffffffU);

		*((cl_int *)p_accum_datum) = Min(*((cl_int *)p_accum_datum),
										 newval_int);
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_normal_max_int(cl_bool *p_accum_isnull, Datum *p_accum_datum,
					   cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		cl_int		newval_int = (cl_int)(newval_datum & 0xffffffffU);

		*((cl_int *)p_accum_datum) = Max(*((cl_int *)p_accum_datum),
										 newval_int);
		*p_accum_isnull = false;
	}
}


STATIC_INLINE(void)
aggcalc_normal_add_int(cl_bool *p_accum_isnull, Datum *p_accum_datum,
					   cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		*((cl_int *)p_accum_datum) += (cl_int)(newval_datum & 0xffffffff);
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_normal_min_long(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		*((cl_long *)p_accum_datum) = Min(*((cl_long *)p_accum_datum),
										  (cl_long)newval_datum);
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_normal_max_long(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		*((cl_long *)p_accum_datum) = Max(*((cl_long *)p_accum_datum),
										  (cl_long)newval_datum);
		*p_accum_isnull = false;
	}
}


STATIC_INLINE(void)
aggcalc_normal_add_long(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		*((cl_long *)p_accum_datum) += (cl_long)newval_datum;
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_normal_min_float(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						 cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		*((cl_float *)p_accum_datum)
			= Min(*((cl_float *)p_accum_datum),
				  __int_as_float(newval_datum & 0xffffffff));
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_normal_max_float(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						 cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		*((cl_float *)p_accum_datum)
			= Max(*((cl_float *)p_accum_datum),
				  __int_as_float(newval_datum & 0xffffffff));
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_normal_add_float(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						 cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		*((cl_float *)p_accum_datum)
			+= __int_as_float(newval_datum & 0xffffffff);
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_normal_min_double(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						  cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		*((cl_double *)p_accum_datum)
			= Min(*((cl_double *)p_accum_datum),
				  __longlong_as_double((cl_ulong)newval_datum));
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_normal_max_double(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						  cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		*((cl_double *)p_accum_datum)
			= Max(*((cl_double *)p_accum_datum),
				  __longlong_as_double((cl_ulong)newval_datum));
		*p_accum_isnull = false;
	}
}

STATIC_INLINE(void)
aggcalc_normal_add_double(cl_bool *p_accum_isnull, Datum *p_accum_datum,
						  cl_bool newval_isnull, Datum newval_datum)
{
	if (!newval_isnull)
	{
		*((cl_double *)p_accum_datum)
			+= __longlong_as_double((cl_ulong)newval_datum);
		*p_accum_isnull = false;
	}
}
#endif	/* __CUDACC__ */
#endif	/* CUDA_GPUPREAGG_H */
