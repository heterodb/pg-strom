/*
 * cuda_gpupreagg.h
 *
 * Preprocess of aggregate using GPU acceleration, to reduce number of
 * rows to be processed by CPU; including the Sort reduction.
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
#ifndef CUDA_GPUPREAGG_H
#define CUDA_GPUPREAGG_H

/*
 * Sequential Scan using GPU/MIC acceleration
 *
 * It packs a kern_parambuf and kern_resultbuf structure within a continuous
 * memory ares, to transfer (usually) small chunk by one DMA call.
 *
 * +----------------+
 * | status         |
 * +----------------+
 * | hash_size      |
 * +----------------+
 * | pg_crc32_table |
 * |      :         |
 * +----------------+ ---
 * | kern_parambuf  |  ^
 * | +--------------+  | kparams.length
 * | | length       |  |
 * | +--------------+  |
 * | | nparams      |  |
 * | +--------------+  |
 * | |    :         |  |
 * | |    :         |  v
 * +-+--------------+ ---
 * | kern_resultbuf |
 * | +--------------+
 * | | nrels        |
 * | +--------------+
 * | | nrooms       |
 * | +--------------+
 * | | nitems       |
 * | +--------------+
 * | | errcode      |
 * | +--------------+
 * | | results[]    |
 * | |    :         |
 * | |    :         |
 * +-+--------------+
 */
typedef struct
{
	kern_errorbuf	kerror;					/* kernel error information */
	/* -- runtime statistics -- */
	cl_uint			num_groups;				/* out: # of new groups */
	cl_uint			varlena_usage;			/* out: size of varlena usage */
	cl_uint			ghash_conflicts;		/* out: # of ghash conflicts */
	cl_uint			fhash_conflicts;		/* out: # of fhash conflicts */
	/* -- other hashing parameters -- */
	cl_uint			key_dist_salt;			/* hashkey distribution salt */
	cl_uint			hash_size;				/* size of global hash-slots */
	cl_uint			pg_crc32_table[256];	/* master CRC32 table */
	kern_parambuf	kparams;
	/*
	 * kern_resultbuf with nrels==1 shall be located next to kern_parambuf
	 */
} kern_gpupreagg;

/* macro definitions to reference packed values */
#define KERN_GPUPREAGG_PARAMBUF(kgpreagg)				\
	(&(kgpreagg)->kparams)
#define KERN_GPUPREAGG_PARAMBUF_LENGTH(kgpreagg)		\
	((kgpreagg)->kparams.length)
#define KERN_GPUPREAGG_RESULTBUF(kgpreagg)				\
	((kern_resultbuf *)									\
	 ((char *)KERN_GPUPREAGG_PARAMBUF(kgpreagg)			\
	  + KERN_GPUPREAGG_PARAMBUF_LENGTH(kgpreagg)))

#define KERN_GPUPREAGG_LENGTH(kgpreagg,nitems)			\
	((uintptr_t)(KERN_GPUPREAGG_RESULTBUF(kgpreagg)->results + (nitems)) - \
	 (uintptr_t)(kgpreagg))

#define KERN_GPUPREAGG_DMASEND_OFFSET(kgpreagg)		0
#define KERN_GPUPREAGG_DMASEND_LENGTH(kgpreagg)			\
	(offsetof(kern_gpupreagg, kparams) +				\
	 KERN_GPUPREAGG_PARAMBUF_LENGTH(kgpreagg) +			\
	 offsetof(kern_resultbuf, results[0]))
#define KERN_GPUPREAGG_DMARECV_OFFSET(kgpreagg)		0
#define KERN_GPUPREAGG_DMARECV_LENGTH(kgpreagg)			\
	offsetof(kern_gpupreagg, pg_crc32_table[0])

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
	cl_uint			hash_usage;		/* current number of hash_slot in use */
	cl_uint			hash_size;		/* total size of the hash_slot below */
	pagg_hashslot	hash_slot[FLEXIBLE_ARRAY_MEMBER];
} kern_global_hashslot;

/*
 * NOTE: pagg_datum is a set of information to calculate running total.
 * group_id indicates which group does this work-item belong to, instead
 * of gpupreagg_keymatch().
 * isnull indicates whether the current running total is NULL, or not.
 * XXX_val is a running total itself.
 */
typedef struct
{
	cl_int			isnull;
	cl_char			__padding__[4];
	union {
		cl_short	short_val;
		cl_ushort	ushort_val;
		cl_int		int_val;
		cl_uint		uint_val;
		cl_long		long_val;
		cl_ulong	ulong_val;
		cl_float	float_val;
		cl_double	double_val;
	};
} pagg_datum;

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
					kern_data_store *kds,
					size_t kds_index);

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
 * local calculation function - to be generated by PG-Strom on the fly
 *
 * It aggregates the newval to accum using atomic operation on the
 * local pagg_datum array
 */
STATIC_FUNCTION(void)
gpupreagg_local_calc(kern_context *kcxt,
					 cl_int attnum,
					 pagg_datum *accum,
					 pagg_datum *newval);

/*
 * global calculation function - to be generated by PG-Strom on the fly
 *
 * It also aggregates the newval to accum using atomic operation on
 * the global kern_data_store
 */
STATIC_FUNCTION(void)
gpupreagg_global_calc(kern_context *kcxt,
					  cl_int attnum,
					  kern_data_store *accum_kds,  size_t accum_index,
					  kern_data_store *newval_kds, size_t newval_index);

/*
 * Reduction operation with no atomic operations. It can be used if no
 * GROUP-BY clause is given, because all the rows shall be eventually
 * consolidated into one aggregated row.
 */
STATIC_FUNCTION(void)
gpupreagg_nogroup_calc(kern_context *kcxt,
					   cl_int attnum,
					   pagg_datum *accum,
					   pagg_datum *newval);

/*
 * translate a kern_data_store (input) into an output form
 * (auto generated function)
 */
STATIC_FUNCTION(void)
gpupreagg_projection(kern_context *kcxt,
					 kern_data_store *kds_in,
					 kern_data_store *kds_src,
					 size_t rowidx_in,
					 size_t rowidx_out);
/*
 * check qualifiers being pulled-up from the outer relation.
 * if not valid, this record shall not be processed.
 */
STATIC_FUNCTION(bool)
gpupreagg_qual_eval(kern_context *kcxt,
					kern_data_store *kds,
					size_t kds_index);

/*
 * load the data from kern_data_store to pagg_datum structure
 */
STATIC_FUNCTION(void)
gpupreagg_data_load(pagg_datum *pdatum,		/* __shared__ */
					kern_context *kcxt,
					kern_data_store *kds,
					cl_uint colidx, cl_uint rowidx)
{
	kern_colmeta	cmeta;
	Datum		   *values;
	cl_char		   *isnull;

	if (kds->format != KDS_FORMAT_SLOT || colidx >= kds->ncols)
	{
		STROM_SET_ERROR(&kcxt->e, StromError_DataStoreCorruption);
		return;
	}
	cmeta = kds->colmeta[colidx];
	values = KERN_DATA_STORE_VALUES(kds,rowidx);
	isnull = KERN_DATA_STORE_ISNULL(kds,rowidx);

	/*
	 * Right now, expected data length for running total of partial
	 * aggregates are 2, 4, or 8. Elasewhere, it may be a bug.
	 */
	if (cmeta.attlen == sizeof(cl_short) ||		/* also, cl_short */
		cmeta.attlen == sizeof(cl_int))			/* also, cl_float */
	{
		pdatum->isnull = isnull[colidx];
		pdatum->int_val = (cl_int)(values[colidx] & 0xffffffffUL);
	}
	else if (cmeta.attlen == sizeof(cl_long))	/* also, cl_double */
	{
		pdatum->isnull = isnull[colidx];
		pdatum->long_val = (cl_long)values[colidx];
	}
	else
	{
		STROM_SET_ERROR(&kcxt->e, StromError_DataStoreCorruption);
	}
}

/*
 * store the data from pagg_datum structure to kern_data_store
 */
STATIC_FUNCTION(void)
gpupreagg_data_store(pagg_datum *pdatum,	/* __shared__ */
					 kern_context *kcxt,
					 kern_data_store *kds,
					 cl_uint colidx, cl_uint rowidx)
{
	kern_colmeta	cmeta;
	Datum		   *values;
	cl_char		   *isnull;

	if (kds->format != KDS_FORMAT_SLOT || colidx >= kds->ncols)
	{
		STROM_SET_ERROR(&kcxt->e, StromError_DataStoreCorruption);
		return;
	}
	cmeta = kds->colmeta[colidx];
	values = KERN_DATA_STORE_VALUES(kds,rowidx);
	isnull = KERN_DATA_STORE_ISNULL(kds,rowidx);

	/*
	 * Right now, expected data length for running total of partial
	 * aggregates are 2, 4, or 8. Elasewhere, it may be a bug.
	 */
	if (cmeta.attlen == sizeof(cl_short))
	{
		isnull[colidx] = pdatum->isnull;
		values[colidx] = pdatum->int_val;
	}
	else if (cmeta.attlen == sizeof(cl_int))	/* also, cl_float */
	{
		isnull[colidx] = pdatum->isnull;
		values[colidx] = pdatum->int_val;
	}
	else if (cmeta.attlen == sizeof(cl_long))	/* also, cl_double */
	{
		isnull[colidx] = pdatum->isnull;
		values[colidx] = pdatum->long_val;
	}
	else
	{
		STROM_SET_ERROR(&kcxt->e, StromError_DataStoreCorruption);
	}
}

/* gpupreagg_data_move - it moves grouping key from the source kds to
 * the destination kds as is. We assume toast buffer is shared and
 * resource number of varlena key is not changed. So, all we need to
 * do is copying the offset value, not varlena body itself.
 */
STATIC_FUNCTION(void)
gpupreagg_data_move(kern_context *kcxt,
					cl_uint colidx,
					kern_data_store *kds_src, cl_uint rowidx_src,
					kern_data_store *kds_dst, cl_uint rowidx_dst)
{
	Datum	   *src_values;
	Datum	   *dst_values;
	cl_char	   *src_isnull;
	cl_char	   *dst_isnull;

	/*
	 * XXX - Paranoire checks?
	 */
	if (kds_src->format != KDS_FORMAT_SLOT ||
		kds_dst->format != KDS_FORMAT_SLOT)
	{
		STROM_SET_ERROR(&kcxt->e, StromError_DataStoreCorruption);
		return;
	}
	if (colidx >= kds_src->ncols || colidx >= kds_dst->ncols)
	{
		STROM_SET_ERROR(&kcxt->e, StromError_DataStoreCorruption);
		return;
	}
	src_values = KERN_DATA_STORE_VALUES(kds_src, rowidx_src);
	src_isnull = KERN_DATA_STORE_ISNULL(kds_src, rowidx_src);
	dst_values = KERN_DATA_STORE_VALUES(kds_dst, rowidx_dst);
	dst_isnull = KERN_DATA_STORE_ISNULL(kds_dst, rowidx_dst);

	dst_isnull[colidx] = src_isnull[colidx];
	dst_values[colidx] = src_values[colidx];
}

/*
 * gpupreagg_final_data_move
 *
 * It moves the value from source buffer to destination buffer. If it needs
 * to allocate variable-length buffer, it expands extra area of the final
 * buffer and returns allocated area.
 */
STATIC_FUNCTION(cl_uint)
gpupreagg_final_data_move(kern_context *kcxt,
						  cl_uint colidx,
						  kern_data_store *kds_src, cl_uint rowidx_src,
						  kern_data_store *kds_dst, cl_uint rowidx_dst)
{
	Datum	   *src_values;
	Datum	   *dst_values;
	cl_char	   *src_isnull;
	cl_char	   *dst_isnull;
	cl_uint		alloc_size = 0;

	kern_colmeta cmeta = kds_src->colmeta[colidx];

	/* Paranoire checks? */
	assert(kds_src->format == KDS_FORMAT_SLOT &&
		   kds_dst->format == KDS_FORMAT_SLOT);
	assert(colidx < kds_src->ncols && colidx < kds_dst->ncols);
	assert(rowidx_src < kds_src->nitems);
	assert(rowidx_dst < kds_dst->nitems);

	/* Get relevant slot */
	src_values = KERN_DATA_STORE_VALUES(kds_src, rowidx_src);
	src_isnull = KERN_DATA_STORE_ISNULL(kds_src, rowidx_src);
	dst_values = KERN_DATA_STORE_VALUES(kds_dst, rowidx_dst);
	dst_isnull = KERN_DATA_STORE_ISNULL(kds_dst, rowidx_dst);

	if (src_isnull[colidx])
		dst_isnull[colidx] = true;
	else if (cmeta.attbyval)
	{
		dst_isnull[colidx] = false;
		dst_values[colidx] = src_values[colidx];
	}
	else
	{
		void		   *datum = kern_get_datum(kds_src,colidx,rowidx_src);
		pg_varlena_t	vl_src = pg_varlena_datum_ref(kcxt,datum,false);
		cl_uint			vl_len;
		cl_uint			usage_prev;

		vl_len = (cmeta.attlen >= 0
				  ? cmeta.attlen
				  : VARSIZE_ANY(vl_src.value));
		alloc_size = MAXALIGN(vl_len);
		usage_prev = atomicAdd(&kds_dst->usage, alloc_size);
		if (KERN_DATA_STORE_SLOT_LENGTH(kds_dst, kds_dst->nrooms) +
			usage_prev + alloc_size >= kds_dst->length)
		{
			STROM_SET_ERROR(&kcxt->e, StromError_DataStoreNoSpace);
			dst_isnull[colidx] = true;
		}
		else
		{
			char	   *alloc_ptr = ((char *)kds_dst + kds_dst->length -
									 (usage_prev + alloc_size));
			memcpy(alloc_ptr, vl_src.value, vl_len);
			dst_isnull[colidx] = false;
			dst_values[colidx] = PointerGetDatum(alloc_ptr);
		}
	}
	return alloc_size;
}


/*
 * gpupreagg_preparation - It translaes an input kern_data_store (that
 * reflects outer relation's tupdesc) into the form of running total
 * and final result of gpupreagg (that reflects target-list of GpuPreAgg).
 *
 * Pay attention on a case when the kern_data_store with row-format is
 * translated. Row-format does not have toast buffer because variable-
 * length fields are in-place. gpupreagg_projection() treats the input
 * kern_data_store as toast buffer of the later stage. So, caller has to
 * give this kern_data_store (never used for data-store in the later
 * stage) as toast buffer if the source kds has row-format.
 */
KERNEL_FUNCTION(void)
gpupreagg_preparation(kern_gpupreagg *kgpreagg,
					  kern_data_store *kds_in,	/* in: KDS_FORMAT_ROW */
					  kern_data_store *kds_src,	/* out: KDS_FORMAT_SLOT */
					  kern_global_hashslot *g_hash)
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_context	kcxt;
	cl_uint			offset;
	cl_uint			nitems;
	size_t			kds_index = get_global_id();
	size_t			hash_size;
	size_t			hash_index;
	__shared__ cl_uint base;

	INIT_KERNEL_CONTEXT(&kcxt,gpupreagg_preparation,kparams);

	/* sanity checks */
	assert(kgpreagg->key_dist_salt > 0);
	assert(kds_in->format == KDS_FORMAT_ROW);
	assert(kds_src->format == KDS_FORMAT_SLOT);

	/* init global hash slot */
	hash_size = kgpreagg->hash_size;;
	if (get_global_id() == 0)
	{
		g_hash->hash_usage = 0;
		g_hash->hash_size = hash_size;
	}
	for (hash_index = get_global_id();
		 hash_index < hash_size;
		 hash_index += get_global_size())
	{
		g_hash->hash_slot[hash_index].s.hash = 0;
		g_hash->hash_slot[hash_index].s.index = (cl_uint)(0xffffffff);
	}

	/* check qualifiers */
	if (kds_index < kds_in->nitems)
	{
		if (!gpupreagg_qual_eval(&kcxt, kds_in, kds_index))
			kds_index = kds_in->nitems;	/* ensure this thread is not valid */
	}

	/* calculation of total number of rows to be processed in this work-
	 * group.
	 */
	offset = arithmetic_stairlike_add(kds_index < kds_in->nitems ? 1 : 0,
									  &nitems);

	/* Allocation of the result slot on the kds_src. */
	if (get_local_id() == 0)
	{
		if (nitems > 0)
			base = atomicAdd(&kds_src->nitems, nitems);
		else
			base = 0;
	}
	__syncthreads();

	/* GpuPreAgg should never increase number of items */
	assert(base + nitems <= kds_src->nrooms);

	/* do projection */
	if (kds_index < kds_in->nitems)
	{
		gpupreagg_projection(&kcxt,
							 kds_in,			/* input kds */
							 kds_src,			/* source of reduction kds */
							 kds_index,			/* rowidx of kds_in */
							 base + offset);	/* rowidx of kds_src */
	}
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kgpreagg->kerror, kcxt.e);
}

/*
 * gpupreagg_local_reduction
 */
KERNEL_FUNCTION_MAXTHREADS(void)
gpupreagg_local_reduction(kern_gpupreagg *kgpreagg,
						  kern_data_store *kds_src,
						  kern_data_store *kds_dst)
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_context	kcxt;
	varlena		   *kparam_0 = (varlena *)kparam_get_value(kparams, 0);
	cl_char		   *gpagg_atts = (cl_char *) VARDATA(kparam_0);
	size_t			hash_size = 2 * get_local_size();
	size_t			dest_index;
	cl_uint			owner_index;
	cl_uint			key_dist_salt = kgpreagg->key_dist_salt;
	cl_uint			key_dist_index = 0;
	cl_uint			hash_value;
	cl_uint			hash_value_base;
	cl_uint			nitems = kds_src->nitems;
	cl_uint			nattrs = kds_src->ncols;
	cl_uint			ngroups;
	cl_uint			index;
	cl_uint			attnum;
	pg_int4_t		key_dist_factor;
	pagg_hashslot	old_slot;
	pagg_hashslot	new_slot;
	pagg_hashslot	cur_slot;
	pagg_datum	   *l_datum;
	pagg_hashslot  *l_hashslot;
	__shared__ cl_uint	crc32_table[256];
	__shared__ size_t	base_index;

	INIT_KERNEL_CONTEXT(&kcxt,gpupreagg_local_reduction,kparams);

	/*
	 * calculation of the hash value of grouping keys in this record.
	 * It tends to take massive amount of random access on global memory,
	 * so it makes performance advantage to move the master table from
	 * gloabl to the local memory first.
	 */
	for (index = get_local_id();
		 index < lengthof(kgpreagg->pg_crc32_table);
		 index += get_local_size())
		crc32_table[index] = kgpreagg->pg_crc32_table[index];
	__syncthreads();

	INIT_LEGACY_CRC32(hash_value);
	if (get_global_id() < nitems)
		hash_value = gpupreagg_hashvalue(&kcxt, crc32_table,
										 hash_value,
										 kds_src,
										 get_global_id());
	hash_value_base = hash_value;
	if (key_dist_salt > 1)
	{
		key_dist_factor.isnull = false;
		key_dist_factor.value = (get_global_id() % key_dist_salt);
		hash_value = pg_int4_comp_crc32(crc32_table,
										hash_value,
										key_dist_factor);
	}
	FIN_LEGACY_CRC32(hash_value);

	/*
	 * Find a hash-slot to determine the item index that represents
	 * a particular group-keys.
	 * The array of global hash-slot should be initialized to 'all
	 * empty' state on the projection kernel.
	 * one will take a place using atomic operation. Then. here are
	 * two cases for hash conflicts; case of same grouping-key, or
	 * case of different grouping-key but same hash-value.
	 * The first conflict case informs us the item-index responsible
	 * to the grouping key. We cannot help the later case, so retry
	 * the steps with next hash-slot.
	 */
	l_hashslot = SHARED_WORKMEM(pagg_hashslot);
	for (index = get_local_id();
		 index < hash_size;
		 index += get_local_size())
	{
		l_hashslot[index].s.hash = 0;
		l_hashslot[index].s.index = (cl_uint)(0xffffffff);
	}
	__syncthreads();

	if (get_global_id() < nitems)
	{
	retry_major:
		new_slot.s.hash = hash_value;
		new_slot.s.index = get_local_id();
		old_slot.s.hash = 0;
		old_slot.s.index = (cl_uint)(0xffffffff);
		index = hash_value % hash_size;
	retry_minor:
		cur_slot.value = atomicCAS(&l_hashslot[index].value,
								   old_slot.value,
								   new_slot.value);
		if (cur_slot.value == old_slot.value)
		{
			/* Hash slot was empty, so this thread shall be responsible
			 * to this grouping-key.
			 */
			owner_index = new_slot.s.index;
		}
		else
		{
			size_t	buddy_index
				= (get_global_id() - get_local_id() + cur_slot.s.index);

			if (cur_slot.s.hash == new_slot.s.hash &&
				gpupreagg_keymatch(&kcxt,
								   kds_src, get_global_id(),
								   kds_src, buddy_index))
			{
				owner_index = cur_slot.s.index;
			}
			else
			{
				if (key_dist_salt > 1 && ++key_dist_index < key_dist_salt)
				{
					hash_value = hash_value_base;
					key_dist_factor.isnull = false;
					key_dist_factor.value =
						(get_global_id() + key_dist_index) % key_dist_salt;
					hash_value = pg_int4_comp_crc32(crc32_table,
													hash_value,
													key_dist_factor);
					FIN_LEGACY_CRC32(hash_value);
					goto retry_major;
				}
				index = (index + 1) % hash_size;
				goto retry_minor;
			}
		}
	}
	else
		owner_index = (cl_uint)(0xffffffff);
    __syncthreads();

	/*
	 * Make a reservation on the destination kern_data_store
	 * Only thread that is responsible to grouping-key (also, it shall
	 * have same hash-index with get_local_id(0)) takes a place on the
	 * destination kern_data_store.
	 */
	index = arithmetic_stairlike_add(get_local_id() == owner_index ? 1 : 0,
									 &ngroups);
	if (get_local_id() == 0)
		base_index = atomicAdd(&kds_dst->nitems, ngroups);
	__syncthreads();

	/* should not growth the number of items over the nrooms */
	assert(base_index + ngroups <= kds_dst->nrooms);
	dest_index = base_index + index;

	/*
	 * Local reduction for each column
	 *
	 * Any threads that are NOT responsible to grouping-key calculates
	 * aggregation on the item that is responsibles.
	 * Once atomic operations got finished, values of pagg_datum in the
	 * respobsible thread will have partially aggregated one.
	 *
	 * NOTE: local memory shall be reused to l_datum array, so l_hashslot[]
	 * array is no longer available across here
	 */
	l_datum = SHARED_WORKMEM(pagg_datum);
	for (attnum = 0; attnum < nattrs; attnum++)
	{
		/*
		 * In case when this column is either a grouping-key or not-
		 * referenced one (thus, not a partial aggregation), all we
		 * need to do is copying the data from the source to the
		 * destination; without modification anything.
		 */
		if (gpagg_atts[attnum] != GPUPREAGG_FIELD_IS_AGGFUNC)
		{
			if (owner_index == get_local_id())
			{
				gpupreagg_data_move(&kcxt,
									attnum,
									kds_src, get_global_id(),
									kds_dst, dest_index);
			}
			continue;
		}

		/* Load aggregation item to pagg_datum */
		if (get_global_id() < nitems)
		{
			gpupreagg_data_load(l_datum + get_local_id(),
								&kcxt,
								kds_src,
								attnum,
								get_global_id());
		}
		__syncthreads();

		/* Reduction, using local atomic operation */
		if (get_global_id() < nitems &&
			get_local_id() != owner_index)
		{
			gpupreagg_local_calc(&kcxt,
								 attnum,
								 l_datum + owner_index,
								 l_datum + get_local_id());
		}
		__syncthreads();

		/* Move the value that is aggregated */
		if (owner_index == get_local_id())
		{
			gpupreagg_data_store(l_datum + owner_index,
								 &kcxt,
								 kds_dst,
								 attnum,
								 dest_index);
			/*
			 * varlena should never appear here, so we don't need to
			 * put pg_fixup_tupslot_varlena() here
			 */
		}
		__syncthreads();
	}
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kgpreagg->kerror, kcxt.e);
}

/*
 * Check whether the global hash-slot has enough free space at this moment.
 */
STATIC_INLINE(cl_bool)
check_global_hashslot_usage(kern_context *kcxt,
							size_t g_hashusage, size_t g_hashlimit)
{
	if (g_hashusage >= g_hashlimit)
	{
		STROM_SET_ERROR(&kcxt->e, StromError_DataStoreNoSpace);
		return false;	/* hash usage exceeds the limitation */
	}
	return true;		/* ok, we still have rooms */
}

/*
 * gpupreagg_global_reduction
 */
KERNEL_FUNCTION(void)
gpupreagg_global_reduction(kern_gpupreagg *kgpreagg,
						   kern_data_store *kds_dst,
						   kern_global_hashslot *g_hash)
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_resultbuf *kresults = KERN_GPUPREAGG_RESULTBUF(kgpreagg);
	kern_context	kcxt;
	varlena		   *kparam_0 = (varlena *) kparam_get_value(kparams, 0);
	cl_char		   *gpagg_atts = (cl_char *) VARDATA(kparam_0);
	size_t			g_hashsize = g_hash->hash_size;
	size_t			g_hashlimit = GLOBAL_HASHSLOT_THRESHOLD(g_hashsize);
	size_t			dest_index;
	size_t			owner_index;
	cl_uint			key_dist_salt = kgpreagg->key_dist_salt;
	cl_uint			key_dist_index = 0;
	cl_uint			hash_value;
	cl_uint			hash_value_base;
	cl_uint			ngroups;
	cl_uint			index;
	cl_uint			nattrs = kds_dst->ncols;
	cl_uint			attnum;
	cl_uint			nconflicts = 0;
	cl_uint			count;
	pg_int4_t		key_dist_factor;
	pagg_hashslot	old_slot;
	pagg_hashslot	new_slot;
	pagg_hashslot	cur_slot;
	__shared__ cl_uint	crc32_table[256];
	__shared__ cl_uint	base;
	__shared__ cl_uint	g_hashusage;

	INIT_KERNEL_CONTEXT(&kcxt,gpupreagg_global_reduction,kparams);

	/*
	 * calculation of the hash value of grouping keys in this record.
	 * It tends to take massive amount of random access on global memory,
	 * so it makes performance advantage to move the master table from
	 * gloabl to the local memory first.
	 */
	for (index = get_local_id();
		 index < lengthof(kgpreagg->pg_crc32_table);
		 index += get_local_size())
		crc32_table[index] = kgpreagg->pg_crc32_table[index];
	__syncthreads();

	/*
	 * check g_hash slot usage first - this kernel uses staircase operation,
	 * so quick exit must be atomically.
	 */
	__threadfence();
	if (get_local_id() == 0)
		g_hashusage = g_hash->hash_usage;
	__syncthreads();
	if (!check_global_hashslot_usage(&kcxt, g_hashusage, g_hashlimit))
		goto out;

	if (get_global_id() < kds_dst->nitems)
	{
		/*
		 * Calculation of initial hash value
		 */
		INIT_LEGACY_CRC32(hash_value);
		hash_value = gpupreagg_hashvalue(&kcxt, crc32_table,
										 hash_value,
										 kds_dst,
										 get_global_id());
		hash_value_base = hash_value;
		if (key_dist_salt > 1)
		{
			key_dist_factor.isnull = false;
			key_dist_factor.value = (get_global_id() % key_dist_salt);
			hash_value = pg_int4_comp_crc32(crc32_table,
											hash_value,
											key_dist_factor);
		}
		FIN_LEGACY_CRC32(hash_value);

		/*
		 * Find a hash-slot to determine the item index that represents
		 * a particular group-keys.
		 * The array of hash-slot is initialized to 'all empty', so first
		 * one will take a place using atomic operation. Then. here are
		 * two cases for hash conflicts; case of same grouping-key, or
		 * case of different grouping-key but same hash-value.
		 * The first conflict case informs us the item-index responsible
		 * to the grouping key. We cannot help the later case, so retry
		 * the steps with next hash-slot.
		 */
	retry_major:
		new_slot.s.hash = hash_value;
		new_slot.s.index = get_global_id();
		old_slot.s.hash = 0;
		old_slot.s.index = (cl_uint)(0xffffffff);
		index = hash_value % g_hashsize;
	retry_minor:
		cur_slot.value = atomicCAS(&g_hash->hash_slot[index].value,
								   old_slot.value,
								   new_slot.value);
		if (cur_slot.value == old_slot.value)
		{
			cl_uint		g_hashusage = atomicAdd(&g_hash->hash_usage, 1);
			assert(g_hashusage < g_hashsize);
			/*
			 * Hash slot was empty, so this thread shall be responsible
			 * to this grouping-key.
			 */
			owner_index = new_slot.s.index;

		}
		else if (cur_slot.s.hash == new_slot.s.hash &&
				 gpupreagg_keymatch(&kcxt,
									kds_dst, get_global_id(),
									kds_dst, cur_slot.s.index))
		{
			assert(cur_slot.s.index < kds_dst->nitems);
			owner_index = cur_slot.s.index;
		}
		else
		{
			nconflicts++;
			if (key_dist_salt > 1 && ++key_dist_index < key_dist_salt)
			{
				hash_value = hash_value_base;
				key_dist_factor.isnull = false;
				key_dist_factor.value =
					(get_global_id() + key_dist_index) % key_dist_salt;
				hash_value = pg_int4_comp_crc32(crc32_table,
												hash_value,
												key_dist_factor);
				FIN_LEGACY_CRC32(hash_value);
				goto retry_major;
			}
			__threadfence();
			if (check_global_hashslot_usage(&kcxt, g_hash->hash_usage,
											g_hashlimit))
			{
				index = (index + 1) % g_hashsize;
				goto retry_minor;
			}
			owner_index = (cl_uint)(0xffffffff);
		}
	}
	else
		owner_index = (cl_uint)(0xffffffff);

	/*
	 * Allocation of a slot of kern_rowmap to point which slot is
	 * responsible to grouping key.
	 * All the threads that are not responsible to the grouping-key,
	 * it updates the value of responsible thread.
	 *
	 * NOTE: Length of kern_row_map should be same as kds->nrooms.
	 * So, we can use kds->nrooms to check array boundary.
	 */
	__syncthreads();
	index = arithmetic_stairlike_add(get_global_id() == owner_index ? 1 : 0,
									 &ngroups);
	__syncthreads();
	if (ngroups > 0)
	{
		if (get_local_id() == 0)
			base = atomicAdd(&kresults->nitems, ngroups);
		__syncthreads();
		assert(base + ngroups <= kresults->nrooms);
		dest_index = base + index;

		if (get_global_id() < kds_dst->nitems &&
			get_global_id() == owner_index)
		{
			kresults->results[dest_index] = get_global_id();
		}
	}

	/*
	 * Global reduction for each column
	 *
	 * Any threads that are NOT responsible to grouping-key calculates
	 * aggregation on the item that is responsibles.
	 * Once atomic operations got finished, values of pagg_datum in the
	 * respobsible thread will have partially aggregated one.
	 */
	for (attnum = 0; attnum < nattrs; attnum++)
	{
		if (gpagg_atts[attnum] != GPUPREAGG_FIELD_IS_AGGFUNC)
			continue;

		/*
		 * Reduction, using global atomic operation
		 *
		 * If thread is responsible to the grouping-key, other threads but
		 * NOT responsible will accumlate their values here, then it shall
		 * become aggregated result. So, we mark the "responsible" thread
		 * identifier on the kern_row_map. Once kernel execution gets done,
		 * this index points the location of aggregate value.
		 */
		if (get_global_id() < kds_dst->nitems &&
			get_global_id() != owner_index)
		{
			assert(owner_index < kds_dst->nrooms);
			gpupreagg_global_calc(&kcxt,
								  attnum,
								  kds_dst, owner_index,
								  kds_dst, get_global_id());
		}
	}
out:
	/* collect run-time statistics */
	arithmetic_stairlike_add(nconflicts, &count);
	if (count > 0 && get_local_id() == 0)
		atomicAdd(&kgpreagg->ghash_conflicts, count);
	__syncthreads();

	/* write-back execution status into host-side */
	kern_writeback_error_status(&kgpreagg->kerror, kcxt.e);
}

/*
 * gpupreagg_final_preparation
 *
 * It initializes the f_hash prior to gpupreagg_final_reduction
 */
KERNEL_FUNCTION(void)
gpupreagg_final_preparation(size_t f_hashsize,
							kern_global_hashslot *f_hash)
{
	size_t		hash_index;

	if (get_global_id() == 0)
	{
		f_hash->hash_usage = 0;
		f_hash->hash_size = f_hashsize;
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
 * gpupreagg_final_reduction
 *
 * kds_dst = result of local or global reduction
 * kds_final = destination buffer in this case.
 *             kds_final->usage points current available variable length
 *             area, until kds_final->length. Use atomicAdd().
 * f_hash = hash slot of the final buffer
 */
KERNEL_FUNCTION(void)
gpupreagg_final_reduction(kern_gpupreagg *kgpreagg,		/* in */
						  kern_data_store *kds_dst,		/* in */
						  kern_data_store *kds_final,		/* out */
						  kern_global_hashslot *f_hash)	/* only internal */
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_resultbuf *kresults = KERN_GPUPREAGG_RESULTBUF(kgpreagg);
	kern_context	kcxt;
	varlena		   *kparam_0 = (varlena *) kparam_get_value(kparams, 0);
	cl_char		   *gpagg_atts = (cl_char *) VARDATA(kparam_0);
	cl_uint			kds_index;
	cl_uint			owner_index;
	size_t			f_hashsize = f_hash->hash_size;
	size_t			f_hashlimit = GLOBAL_HASHSLOT_THRESHOLD(f_hashsize);
	cl_uint			key_dist_salt = kgpreagg->key_dist_salt;
	cl_uint			key_dist_index = 0;
	cl_uint			hash_value;
	cl_uint			hash_value_base;
	cl_uint			index;
	cl_uint			nattrs = kds_dst->ncols;
	cl_uint			attnum;
	cl_uint			count;
	cl_uint			nconflicts = 0;
	cl_uint			allocated = 0;
	cl_bool			isOwner = false;
	pg_int4_t		key_dist_factor;
	pagg_hashslot	old_slot;
	pagg_hashslot	new_slot;
	pagg_hashslot	cur_slot;
	__shared__ cl_uint	crc32_table[256];
	__shared__ cl_uint	f_hashusage;

	INIT_KERNEL_CONTEXT(&kcxt,gpupreagg_final_reduction,kparams);

	/*
	 * check availability of final hashslot usage
	 */
	__threadfence();
	if (get_local_id() == 0)
		f_hashusage = f_hash->hash_usage;
	__syncthreads();
	if (!check_global_hashslot_usage(&kcxt, f_hashusage, f_hashlimit))
		goto out;

	/*
	 * calculation of the hash value of grouping keys in this record.
	 * It tends to take massive amount of random access on global memory,
	 * so it makes performance advantage to move the master table from
	 * gloabl to the local memory first.
	 */
	for (index = get_local_id();
		 index < lengthof(kgpreagg->pg_crc32_table);
		 index += get_local_size())
		crc32_table[index] = kgpreagg->pg_crc32_table[index];
	__syncthreads();

	/* row-index on the kds_dst buffer */
	if (kresults->all_visible)
	{
		if (get_global_id() < kds_dst->nitems)
			kds_index = get_global_id();
		else
			goto out;
	}
	else
	{
		if (get_global_id() < min(kresults->nitems,
								  kresults->nrooms))
			kds_index = kresults->results[get_global_id()];
		else
			goto out;
		assert(kds_index < kds_dst->nitems);
	}

	/*
	 * Calculation of initial hash value
	 */
	INIT_LEGACY_CRC32(hash_value);
	hash_value = gpupreagg_hashvalue(&kcxt, crc32_table,
									 hash_value,
									 kds_dst,
									 kds_index);
	hash_value_base = hash_value;
	if (key_dist_salt > 1)
	{
		key_dist_factor.isnull = false;
		key_dist_factor.value = (get_global_id() % key_dist_salt);
		hash_value = pg_int4_comp_crc32(crc32_table,
										hash_value,
										key_dist_factor);
	}
	FIN_LEGACY_CRC32(hash_value);

	/*
	 * Find a hash-slot to determine the item index that represents
	 * a particular group-keys.
	 * The array of hash-slot is initialized to 'all empty', so first
	 * one will take a place using atomic operation. Then. here are
	 * two cases for hash conflicts; case of same grouping-key, or
	 * case of different grouping-key but same hash-value.
	 * The first conflict case informs us the item-index responsible
	 * to the grouping key. We cannot help the later case, so retry
	 * the steps with next hash-slot.
	 */
retry_major:
	new_slot.s.hash  = hash_value;
	new_slot.s.index = (cl_uint)(0xfffffffe); /* LOCK */
	old_slot.s.hash  = 0;
	old_slot.s.index = (cl_uint)(0xffffffff); /* INVALID */
	index  = hash_value % f_hashsize;
retry_minor:
	cur_slot.value = atomicCAS(&f_hash->hash_slot[index].value,
							   old_slot.value, new_slot.value);

	if (cur_slot.value == old_slot.value)
	{
		/*
		 * We could get an empty slot, so hash_usage should be smaller
		 * than hash_size itself, at least.
		 */
		cl_uint		f_hashusage = atomicAdd(&f_hash->hash_usage, 1);
		assert(f_hashusage < f_hashsize);

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
			for (attnum = 0; attnum < nattrs; attnum++)
			{
				allocated += gpupreagg_final_data_move(&kcxt,
													   attnum,
													   kds_dst,
													   kds_index,
													   kds_final,
													   new_slot.s.index);
			}
		}
		else
		{
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		}
		__threadfence();
		f_hash->hash_slot[index].s.index = new_slot.s.index;
	}
	/* wait updating the hash slot. */
	while (cur_slot.s.index == (cl_uint)(0xfffffffe))
	{
		__threadfence();
		cur_slot.s.index = f_hash->hash_slot[index].s.index;
	}

	if (cur_slot.value == old_slot.value)
	{
		owner_index = new_slot.s.index;
		isOwner = true;
	}
	else if (cur_slot.s.hash == new_slot.s.hash &&
			 (cur_slot.s.index < kds_final->nrooms
			  ? gpupreagg_keymatch(&kcxt,
								   kds_dst, kds_index,
								   kds_final, cur_slot.s.index)
			  : true))
	{
		/*
		 * NOTE: If hash value was identical but cur_slot.s.index is out
		 * of range thus we cannot use gpupreagg_keymatch, we cannot
		 * determine whether this hash slot is actually owned by other
		 * row that has identical grouping keys.
		 * However, it is obvious this kernel invocation will return
		 * DataStoreNoSpace error, then CPU fallback will work.
		 */
		owner_index = cur_slot.s.index;
	}
	else
	{
		/* hash slot conflicts */
		nconflicts++;
		if (key_dist_salt > 1 && ++key_dist_index < key_dist_salt)
		{
			hash_value = hash_value_base;
			key_dist_factor.isnull = false;
			key_dist_factor.value =
				(get_global_id() + key_dist_index) % key_dist_salt;
			hash_value = pg_int4_comp_crc32(crc32_table,
											hash_value,
											key_dist_factor);
			FIN_LEGACY_CRC32(hash_value);
			goto retry_major;
		}
		__threadfence();
		if (!check_global_hashslot_usage(&kcxt, f_hash->hash_usage,
										 f_hashlimit))
			goto out;
		index = (index + 1) % f_hashsize;
		goto retry_minor;
	}

	/*
	 * Global reduction for each column
	 *
	 * Any threads that are NOT responsible to grouping-key calculates
	 * aggregation on the item that is responsibles.
	 * Once atomic operations got finished, values of pagg_datum in the
	 * respobsible thread will have partially aggregated one.
	 */
	if (!isOwner && owner_index < kds_final->nrooms)
	{
		for (attnum = 0; attnum < nattrs; attnum++)
		{
			if (gpagg_atts[attnum] != GPUPREAGG_FIELD_IS_AGGFUNC)
				continue;
			
			/*
			 * Reduction, using global atomic operation
			 *
			 * If thread is responsible to the grouping-key, other threads but
			 * NOT responsible will accumlate their values here, then it shall
			 * become aggregated result. So, we mark the "responsible" thread
			 * identifier on the kern_row_map. Once kernel execution gets done,
			 * this index points the location of aggregate value.
			 */
			gpupreagg_global_calc(&kcxt,
								  attnum,
								  kds_final, owner_index,
								  kds_dst, kds_index);
		}
	}

out:
	/* update run-time statistics */
	arithmetic_stairlike_add(isOwner ? 1 : 0, &count);
	if (count > 0 && get_local_id() == 0)
		atomicAdd(&kgpreagg->num_groups, count);
	__syncthreads();

	arithmetic_stairlike_add(allocated, &count);
	if (count > 0 && get_local_id() == 0)
		atomicAdd(&kgpreagg->varlena_usage, count);
	__syncthreads();

	arithmetic_stairlike_add(nconflicts, &count);
	if (count > 0 && get_local_id() == 0)
		atomicAdd(&kgpreagg->fhash_conflicts, count);
	__syncthreads();

	/* write-back execution status into host-side */
	kern_writeback_error_status(&kgpreagg->kerror, kcxt.e);
}

/*
 * gpupreagg_nogroup_reduction
 *
 * It makes aggregation if no GROUP-BY clause given. We can omit atomic-
 * operations in this case, because all the rows are eventually consolidated
 * to just one record, thus usual reduction operation is sufficient.
 */
KERNEL_FUNCTION_MAXTHREADS(void)
gpupreagg_nogroup_reduction(kern_gpupreagg *kgpreagg,
							kern_data_store *kds_src,
							kern_data_store *kds_dst)
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_resultbuf *kresults = KERN_GPUPREAGG_RESULTBUF(kgpreagg);
	kern_context	kcxt;
	varlena		   *kparam_0 = (varlena *)kparam_get_value(kparams, 0);
	cl_char		   *gpagg_atts = (cl_char *)VARDATA(kparam_0);
	pagg_datum	   *l_datum = SHARED_WORKMEM(pagg_datum);
	cl_uint			nitems = kds_src->nitems;
	cl_uint			nattrs = kds_src->ncols;
	size_t			lid = get_local_id();
	size_t			gid = get_global_id();
	size_t			lsz = get_local_size();
	size_t			dest_index	= gid / lsz;
	int				attnum;

	INIT_KERNEL_CONTEXT(&kcxt, gpupreagg_nogroup_reduction, kparams);

	/* loop for each columns */
	for (attnum = 0; attnum < nattrs; attnum++)
	{
		size_t	distance;

		/* if not GPUPREAGG_FIELD_IS_AGGFUNC, do nothing */
		if (gpagg_atts[attnum] != GPUPREAGG_FIELD_IS_AGGFUNC)
		{
			if (gid < nitems  && lid == 0)
				gpupreagg_data_move(&kcxt, attnum,
									kds_src, gid,
									kds_dst, dest_index);
			continue;
		}

		/* load this value from kds_src onto datum */
		if (gid < nitems)
			gpupreagg_data_load(&l_datum[lid], &kcxt,
								kds_src, attnum, gid);

		__syncthreads();

		/* do reduction */
		for (distance = 2; distance <= lsz; distance *= 2)
		{
			if (lid % distance == 0 && (gid + distance / 2) < nitems)
				gpupreagg_nogroup_calc(&kcxt,
									   attnum,
									   &l_datum[lid],
									   &l_datum[lid + distance / 2]);
			__syncthreads();
		}

		/* store this value to kds_dst from datum */
		if (gid < nitems  &&  lid == 0)
			gpupreagg_data_store(&l_datum[lid], &kcxt,
								 kds_dst, attnum, dest_index);
		__syncthreads();
	}

	/*
	 * Fixup kern_rowmap/kds->nitems
	 */
	if (gid == 0)
	{
		kresults->nitems = (nitems + lsz - 1) / lsz;
		kds_dst->nitems = (nitems + lsz - 1) / lsz;
	}
	if (lid == 0)
	{
		kresults->results[dest_index] = dest_index;
	}

	/* write-back execution status into host-side */
	kern_writeback_error_status(&kgpreagg->kerror, kcxt.e);
}


/*
 * gpupreagg_fixup_varlena
 *
 * In case when varlena datum (excludes numeric) is used in grouping-key,
 * datum on kds with tupslot format has not-interpretable for host systems.
 * So, we need to fix up its value to adjust offset by hostptr.
 */
KERNEL_FUNCTION(void)
gpupreagg_fixup_varlena(kern_gpupreagg *kgpreagg, kern_data_store *kds_final)
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_context	kcxt;
	varlena		   *kparam_0 = (varlena *) kparam_get_value(kparams, 0);
	cl_char		   *gpagg_atts = (cl_char *) VARDATA(kparam_0);
	cl_uint			nattrs = kds_final->ncols;
	cl_uint			colidx;
	size_t			offset;
	kern_colmeta	cmeta;

	/* Sanity checks */
	assert(kds_final->format == KDS_FORMAT_SLOT);

	INIT_KERNEL_CONTEXT(&kcxt,gpupreagg_fixup_varlena,kparams);

	if (get_global_id() < kds_final->nitems)
	{
		size_t		kds_index = get_global_id();
		Datum	   *ts_values = KERN_DATA_STORE_VALUES(kds_final, kds_index);
		cl_bool	   *ts_isnull = KERN_DATA_STORE_ISNULL(kds_final, kds_index);

		for (colidx = 0; colidx < nattrs; colidx++)
		{
			if (gpagg_atts[colidx] != GPUPREAGG_FIELD_IS_GROUPKEY)
				continue;
			/* fixed length variable? */
			cmeta = kds_final->colmeta[colidx];
			if (cmeta.attbyval)
				continue;
			/* null variable? */
			if (ts_isnull[colidx])
				continue;
			/* fixup pointer variables */
			offset = ((size_t)ts_values[colidx] -
					  (size_t)&kds_final->hostptr);
			if (offset < kds_final->length)
				ts_values[colidx] = kds_final->hostptr + offset;
			else
			{
				STROM_SET_ERROR(&kcxt.e, StromError_DataStoreOutOfRange);
				break;
			}
		}
	}
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kgpreagg->kerror, kcxt.e);
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
#define Min(x,y)	((x) < (y) ? (x) : (y))
#define Max(x,y)	((x) > (y) ? (x) : (y))
#define Add(x,y)	((x)+(y))

STATIC_INLINE(cl_int)
pg_atomic_min_int(cl_int *addr, cl_int value)
{
	return __iAtomicMin(addr, value);
}
STATIC_INLINE(cl_int)
pg_atomic_max_int(cl_int *addr, cl_int value)
{
	return __iAtomicMax(addr, value);
}
STATIC_INLINE(cl_int)
pg_atomic_add_int(cl_int *addr, cl_int value)
{
	return __iAtomicAdd(addr, value);
}


#define PG_ATOMIC_LONG_TEMPLATE(operation,addr,value)				\
	do {															\
		cl_ulong	curval = *((cl_ulong *) (addr));				\
		cl_ulong	oldval;											\
		cl_ulong	newval;											\
																	\
		do {														\
			oldval = curval;										\
			newval = operation(oldval, (value));					\
		} while ((curval = atomicCAS((cl_ulong *) (addr),			\
									 oldval, newval)) != oldval);	\
		return (cl_long) oldval;									\
	} while(0)

STATIC_INLINE(cl_long)
pg_atomic_min_long(cl_long *addr, cl_long value)
{
#if __CUDA_ARCH__ < 350
	PG_ATOMIC_LONG_TEMPLATE(Min, addr, value);
#else
	return __illAtomicMin(addr, value);	
#endif
}

STATIC_INLINE(cl_long)
pg_atomic_max_long(cl_long *addr, cl_long value)
{
#if __CUDA_ARCH__ < 350
	PG_ATOMIC_LONG_TEMPLATE(Max, addr, value);
#else
	return __illAtomicMax(addr, value);	
#endif
}

STATIC_INLINE(cl_long)
pg_atomic_add_long(cl_long *addr, cl_long value)
{
	return (cl_long) atomicAdd((cl_ulong *) addr, (cl_ulong) value);
}

#define PG_ATOMIC_FLOAT_TEMPLATE(operation,addr,value)					\
	do {																\
		cl_uint		curval = __float_as_int(*(addr));					\
		cl_uint		oldval;												\
		cl_uint		newval;												\
		float		temp;												\
																		\
		do {															\
			oldval = curval;											\
			temp = operation(__int_as_float(oldval), (value));			\
			newval = __float_as_int(temp);								\
		} while ((curval = atomicCAS((cl_uint *) (addr),				\
									 oldval, newval)) != oldval);		\
		return __int_as_float(oldval);									\
	} while(0)

STATIC_INLINE(cl_float)
pg_atomic_min_float(cl_float *addr, float value)
{
	PG_ATOMIC_FLOAT_TEMPLATE(Min, addr, value);
}

STATIC_INLINE(cl_float)
pg_atomic_max_float(cl_float *addr, float value)
{
	PG_ATOMIC_FLOAT_TEMPLATE(Max, addr, value);
}

STATIC_INLINE(cl_float)
pg_atomic_add_float(cl_float *addr, float value)
{
#if __CUDA_ARCH__ < 350
	PG_ATOMIC_FLOAT_TEMPLATE(Add, addr, value);
#else
	return atomicAdd(addr, value);
#endif
}

#define PG_ATOMIC_DOUBLE_TEMPLATE(operation,addr,value)					\
	do {																\
		cl_ulong	curval = __double_as_longlong(*(addr));				\
		cl_ulong	oldval;												\
		cl_ulong	newval;												\
		double		temp;												\
																		\
		do {															\
			oldval = curval;											\
			temp = operation(__longlong_as_double(oldval), (value));	\
			newval = __double_as_longlong(temp);						\
		} while ((curval = atomicCAS((cl_ulong *) (addr),				\
									 oldval, newval)) != oldval);		\
		return __longlong_as_double(oldval);							\
	} while(0)

STATIC_INLINE(cl_double)
pg_atomic_min_double(cl_double *addr, cl_double value)
{
	PG_ATOMIC_DOUBLE_TEMPLATE(Min, addr, value);
}

STATIC_INLINE(cl_double)
pg_atomic_max_double(cl_double *addr, cl_double value)
{
	PG_ATOMIC_DOUBLE_TEMPLATE(Max, addr, value);
}

STATIC_INLINE(cl_double)
pg_atomic_add_double(cl_double *addr, cl_double value)
{
	PG_ATOMIC_DOUBLE_TEMPLATE(Add, addr, value);
}

/* macro to check overflow on accumlate operation */
#define CHECK_OVERFLOW_NONE(x,y)		(0)

#define CHECK_OVERFLOW_SHORT(x,y)		\
	(((x)+(y)) < SHRT_MIN || SHRT_MAX < ((x)+(y)))

#define CHECK_OVERFLOW_INT(x,y)			\
	((((x) < 0) == ((y) < 0)) && (((x) + (y) < 0) != ((x) < 0)))
	
#define CHECK_OVERFLOW_FLOAT(x,y)		\
	(isinf((x) + (y)) && !isinf(x) && !isinf(y))

#define CHECK_OVERFLOW_NUMERIC(x,y)		CHECK_OVERFLOW_NONE(x,y)

/*
 * Helper macros for gpupreagg_local_calc
 */
#define AGGCALC_LOCAL_TEMPLATE(TYPE,kcxt,								\
							   accum_isnull,accum_val,					\
							   newval_isnull,newval_val,				\
							   OVERFLOW_CHECK,ATOMIC_FUNC_CALL)			\
	do {																\
		if (!(newval_isnull))											\
		{																\
			TYPE old = ATOMIC_FUNC_CALL;								\
			if (OVERFLOW_CHECK(old, (newval_val)))						\
			{															\
				STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);		\
			}															\
			(accum_isnull) = false;										\
		}																\
	} while (0)

#define AGGCALC_LOCAL_TEMPLATE_SHORT(kcxt,accum,newval,					\
									 OVERFLOW_CHECK,ATOMIC_FUNC_CALL)	\
	AGGCALC_LOCAL_TEMPLATE(cl_int,kcxt,									\
						   (accum)->isnull,(accum)->int_val,			\
						   (newval)->isnull,(newval)->int_val,			\
						   OVERFLOW_CHECK,ATOMIC_FUNC_CALL)
#define AGGCALC_LOCAL_TEMPLATE_INT(kcxt,accum,newval,					\
								   OVERFLOW_CHECK,ATOMIC_FUNC_CALL)		\
	AGGCALC_LOCAL_TEMPLATE(cl_int,kcxt,									\
						   (accum)->isnull,(accum)->int_val,			\
						   (newval)->isnull,(newval)->int_val,			\
						   OVERFLOW_CHECK,ATOMIC_FUNC_CALL)
#define AGGCALC_LOCAL_TEMPLATE_LONG(kcxt,accum,newval,					\
									OVERFLOW_CHECK,ATOMIC_FUNC_CALL)	\
	AGGCALC_LOCAL_TEMPLATE(cl_long,kcxt,								\
						   (accum)->isnull,(accum)->long_val,			\
						   (newval)->isnull,(newval)->long_val,			\
						   OVERFLOW_CHECK,ATOMIC_FUNC_CALL)
#define AGGCALC_LOCAL_TEMPLATE_FLOAT(kcxt,accum,newval,					\
									 OVERFLOW_CHECK,ATOMIC_FUNC_CALL)	\
	AGGCALC_LOCAL_TEMPLATE(cl_float,kcxt,								\
						   (accum)->isnull,(accum)->float_val,			\
						   (newval)->isnull,(newval)->float_val,		\
						   OVERFLOW_CHECK,ATOMIC_FUNC_CALL)
#define AGGCALC_LOCAL_TEMPLATE_DOUBLE(kcxt,accum,newval,				\
									  OVERFLOW_CHECK,ATOMIC_FUNC_CALL)	\
	AGGCALC_LOCAL_TEMPLATE(cl_double,kcxt,								\
						   (accum)->isnull,(accum)->double_val,			\
						   (newval)->isnull,(newval)->double_val,		\
						   OVERFLOW_CHECK,ATOMIC_FUNC_CALL)
#define AGGCALC_LOCAL_TEMPLATE_NUMERIC(kcxt,accum,newval,				\
									   OVERFLOW_CHECK,ATOMIC_FUNC_CALL)	\
	AGGCALC_LOCAL_TEMPLATE(cl_ulong,kcxt,								\
						   (accum)->isnull,(accum)->ulong_val,			\
						   (newval)->isnull,(newval)->ulong_val,		\
						   OVERFLOW_CHECK,ATOMIC_FUNC_CALL)

/* calculation for local partial min */
#define AGGCALC_LOCAL_PMIN_SHORT(kcxt,accum,newval)						\
	AGGCALC_LOCAL_TEMPLATE_SHORT((kcxt),accum,newval,CHECK_OVERFLOW_NONE, \
		pg_atomic_min_int(&(accum)->int_val,(newval)->int_val))
#define AGGCALC_LOCAL_PMIN_INT(kcxt,accum,newval)						\
	AGGCALC_LOCAL_TEMPLATE_INT((kcxt),accum,newval,CHECK_OVERFLOW_NONE, \
		pg_atomic_min_int(&(accum)->int_val,(newval)->int_val))
#define AGGCALC_LOCAL_PMIN_LONG(kcxt,accum,newval)						\
	AGGCALC_LOCAL_TEMPLATE_LONG((kcxt),accum,newval,CHECK_OVERFLOW_NONE, \
		pg_atomic_min_long(&(accum)->long_val,(newval)->long_val))
#define AGGCALC_LOCAL_PMIN_FLOAT(kcxt,accum,newval)						\
	AGGCALC_LOCAL_TEMPLATE_FLOAT((kcxt),accum,newval,CHECK_OVERFLOW_NONE, \
		pg_atomic_min_float(&(accum)->float_val,(newval)->float_val))
#define AGGCALC_LOCAL_PMIN_DOUBLE(kcxt,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_DOUBLE((kcxt),accum,newval,CHECK_OVERFLOW_NONE, \
		pg_atomic_min_double(&(accum)->double_val,(newval)->double_val))
#define AGGCALC_LOCAL_PMIN_NUMERIC(kcxt,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_NUMERIC(kcxt,accum,newval,CHECK_OVERFLOW_NONE, \
		pg_atomic_min_numeric((kcxt),&(accum)->ulong_val,(newval)->ulong_val))

/* calculation for local partial max */
#define AGGCALC_LOCAL_PMAX_SHORT(kcxt,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_SHORT((kcxt),accum,newval,CHECK_OVERFLOW_NONE,	\
		pg_atomic_max_int(&(accum)->int_val,(newval)->int_val))
#define AGGCALC_LOCAL_PMAX_INT(kcxt,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_INT((kcxt),accum,newval,CHECK_OVERFLOW_NONE,	\
		pg_atomic_max_int(&(accum)->int_val,(newval)->int_val))
#define AGGCALC_LOCAL_PMAX_LONG(kcxt,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_LONG((kcxt),accum,newval,CHECK_OVERFLOW_NONE, \
		pg_atomic_max_long(&(accum)->long_val,(newval)->long_val))
#define AGGCALC_LOCAL_PMAX_FLOAT(kcxt,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_FLOAT((kcxt),accum,newval,CHECK_OVERFLOW_NONE, \
		pg_atomic_max_float(&(accum)->float_val,(newval)->float_val))
#define AGGCALC_LOCAL_PMAX_DOUBLE(kcxt,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_DOUBLE((kcxt),accum,newval,CHECK_OVERFLOW_NONE, \
		pg_atomic_max_double(&(accum)->double_val,(newval)->double_val))
#define AGGCALC_LOCAL_PMAX_NUMERIC(kcxt,accum,newval)				\
	AGGCALC_LOCAL_TEMPLATE_NUMERIC((kcxt),accum,newval,CHECK_OVERFLOW_NONE,	\
		pg_atomic_max_numeric((kcxt),&(accum)->ulong_val,(newval)->ulong_val))

/* calculation for local partial add */
#define AGGCALC_LOCAL_PADD_SHORT(kcxt,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_SHORT((kcxt),accum,newval,CHECK_OVERFLOW_SHORT, \
		pg_atomic_add_int(&(accum)->int_val,(newval)->int_val))
#define AGGCALC_LOCAL_PADD_INT(kcxt,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_INT((kcxt),accum,newval,CHECK_OVERFLOW_INT,	\
		pg_atomic_add_int(&(accum)->int_val,(newval)->int_val))
#define AGGCALC_LOCAL_PADD_LONG(kcxt,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_LONG((kcxt),accum,newval,CHECK_OVERFLOW_INT,	\
		pg_atomic_add_long(&(accum)->long_val,(newval)->long_val))
#define AGGCALC_LOCAL_PADD_FLOAT(kcxt,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_FLOAT((kcxt),accum,newval,CHECK_OVERFLOW_FLOAT, \
		pg_atomic_add_float(&(accum)->float_val,(newval)->float_val))
#define AGGCALC_LOCAL_PADD_DOUBLE(kcxt,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_DOUBLE((kcxt),accum,newval,CHECK_OVERFLOW_FLOAT,	\
		pg_atomic_add_double(&(accum)->double_val,(newval)->double_val))
#define AGGCALC_LOCAL_PADD_NUMERIC(kcxt,accum,newval)				\
	AGGCALC_LOCAL_TEMPLATE_NUMERIC((kcxt),accum,newval,					\
								   CHECK_OVERFLOW_NUMERIC,				\
		pg_atomic_add_numeric((kcxt),&(accum)->ulong_val,(newval)->ulong_val))

/*
 * Helper macros for gpupreagg_global_calc
 *
 * NOTE: please assume the variables below are available in the context
 * these macros in use.
 *   char            new_isnull;
 *   Datum           new_value;
 *   __global char  *accum_isnull;
 *   __global Datum *accum_value;
 */
#define AGGCALC_GLOBAL_TEMPLATE(TYPE,kcxt,							\
								accum_isnull,accum_value,			\
								new_isnull,new_value,				\
								OVERFLOW_CHECK,ATOMIC_FUNC_CALL)	\
	if (!(new_isnull))												\
	{																\
		TYPE old = ATOMIC_FUNC_CALL;								\
		if (OVERFLOW_CHECK(old, (new_value)))						\
		{															\
			STROM_SET_ERROR(&(kcxt)->e, StromError_CpuReCheck);		\
		}															\
		*(accum_isnull) = false;									\
	}

#define AGGCALC_GLOBAL_TEMPLATE_SHORT(kcxt,							\
									  accum_isnull,accum_value,		\
									  new_isnull,new_value,			\
									  OVERFLOW_CHECK,ATOMIC_FUNC_CALL)	\
	AGGCALC_GLOBAL_TEMPLATE(cl_int, kcxt,						\
							accum_isnull,accum_value,				\
							new_isnull,new_value,					\
							OVERFLOW_CHECK,ATOMIC_FUNC_CALL)
#define AGGCALC_GLOBAL_TEMPLATE_INT(kcxt,							\
									accum_isnull,accum_value,		\
									new_isnull,new_value,			\
									OVERFLOW_CHECK,ATOMIC_FUNC_CALL)	\
	AGGCALC_GLOBAL_TEMPLATE(cl_int,kcxt,								\
							accum_isnull,accum_value,				\
							new_isnull,new_value,					\
							OVERFLOW_CHECK,ATOMIC_FUNC_CALL)
#define AGGCALC_GLOBAL_TEMPLATE_LONG(kcxt,							\
									 accum_isnull,accum_value,		\
									 new_isnull,new_value,			\
									 OVERFLOW_CHECK,ATOMIC_FUNC_CALL)	\
	AGGCALC_GLOBAL_TEMPLATE(cl_long,kcxt,							\
							accum_isnull,accum_value,				\
							new_isnull,new_value,					\
							OVERFLOW_CHECK,ATOMIC_FUNC_CALL)
#define AGGCALC_GLOBAL_TEMPLATE_FLOAT(kcxt,							\
									  accum_isnull,accum_value,		\
									  new_isnull,new_value,			\
									  OVERFLOW_CHECK,ATOMIC_FUNC_CALL)	\
	AGGCALC_GLOBAL_TEMPLATE(cl_float,kcxt,							\
							accum_isnull,accum_value,				\
							new_isnull,new_value,					\
							OVERFLOW_CHECK,ATOMIC_FUNC_CALL)
#define AGGCALC_GLOBAL_TEMPLATE_DOUBLE(kcxt,						\
									   accum_isnull,accum_value,	\
									   new_isnull,new_value,		\
									   OVERFLOW_CHECK,ATOMIC_FUNC_CALL)	\
	AGGCALC_GLOBAL_TEMPLATE(cl_double,kcxt,							\
							accum_isnull,accum_value,				\
							new_isnull,new_value,					\
							OVERFLOW_CHECK,ATOMIC_FUNC_CALL)
#define AGGCALC_GLOBAL_TEMPLATE_NUMERIC(kcxt,						\
										accum_isnull,accum_value,	\
										new_isnull,new_value,		\
										OVERFLOW_CHECK,ATOMIC_FUNC_CALL) \
	AGGCALC_GLOBAL_TEMPLATE(cl_ulong,kcxt,							\
							accum_isnull,accum_value,				\
							new_isnull,new_value,					\
							OVERFLOW_CHECK,ATOMIC_FUNC_CALL)

/* calculation for global partial max */
#define AGGCALC_GLOBAL_PMAX_SHORT(kcxt,accum_isnull,accum_value,	\
								  new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_SHORT(									\
		kcxt,accum_isnull,accum_value,								\
		new_isnull,(cl_int)(new_value), CHECK_OVERFLOW_NONE,		\
		pg_atomic_max_int((cl_int *)(accum_value),(cl_int)(new_value)))
#define AGGCALC_GLOBAL_PMAX_INT(kcxt,accum_isnull,accum_value,		\
								new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_INT(									\
		kcxt,accum_isnull,accum_value,								\
		new_isnull,(cl_int)(new_value),CHECK_OVERFLOW_NONE,			\
		pg_atomic_max_int((cl_int *)(accum_value),(cl_int)(new_value)))
#define AGGCALC_GLOBAL_PMAX_LONG(kcxt,accum_isnull,accum_value,		\
								 new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_LONG(\
		kcxt,accum_isnull,accum_value,								\
		new_isnull,(cl_long)(new_value),CHECK_OVERFLOW_NONE,		\
		pg_atomic_max_long((cl_long *)(accum_value),(cl_long)(new_value)))
#define AGGCALC_GLOBAL_PMAX_FLOAT(kcxt,accum_isnull,accum_value,	\
								  new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_FLOAT(									\
		kcxt,accum_isnull,accum_value,								\
		new_isnull, __int_as_float((cl_uint)(new_value)),			\
		CHECK_OVERFLOW_NONE,										\
		pg_atomic_max_float((cl_float *)(accum_value),				\
							__int_as_float((cl_uint)(new_value))))
#define AGGCALC_GLOBAL_PMAX_DOUBLE(kcxt,accum_isnull,accum_value,	\
								   new_isnull,new_value)			\
	AGGCALC_GLOBAL_TEMPLATE_DOUBLE(									\
		kcxt,accum_isnull,accum_value,								\
		new_isnull, __longlong_as_double(new_value),				\
		CHECK_OVERFLOW_NONE,										\
		pg_atomic_max_double((cl_double *)(accum_value),			\
							 __longlong_as_double(new_value)))
#define AGGCALC_GLOBAL_PMAX_NUMERIC(kcxt,accum_isnull,accum_value,	\
									new_isnull,new_value)			\
	AGGCALC_GLOBAL_TEMPLATE_NUMERIC(								\
		kcxt, accum_isnull, accum_value,							\
		new_isnull, (cl_ulong)(new_value),							\
		CHECK_OVERFLOW_NONE,										\
		pg_atomic_max_numeric((kcxt),(cl_ulong *)(accum_value),		\
							  (cl_ulong)(new_value)))
/* calculation for global partial min */
#define AGGCALC_GLOBAL_PMIN_SHORT(kcxt,accum_isnull,accum_value,	\
								  new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_SHORT(									\
		kcxt,accum_isnull,accum_value,								\
		new_isnull,(cl_int)(new_value),								\
		CHECK_OVERFLOW_NONE,										\
		pg_atomic_min_int((cl_int *)(accum_value),(cl_int)(new_value)))
#define AGGCALC_GLOBAL_PMIN_INT(kcxt,accum_isnull,accum_value,		\
								new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_INT(									\
		kcxt,accum_isnull,accum_value,								\
		new_isnull,(cl_int)(new_value),								\
		CHECK_OVERFLOW_NONE,										\
		pg_atomic_min_int((cl_int *)(accum_value),(cl_int)(new_value)))
#define AGGCALC_GLOBAL_PMIN_LONG(kcxt,accum_isnull,accum_value,		\
								 new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_LONG(\
		kcxt,accum_isnull,accum_value,								\
		new_isnull,(cl_long)(new_value),							\
		CHECK_OVERFLOW_NONE,										\
		pg_atomic_min_long((cl_long *)(accum_value),(cl_long)(new_value)))
#define AGGCALC_GLOBAL_PMIN_FLOAT(kcxt,accum_isnull,accum_value,	\
								  new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_FLOAT(									\
		kcxt,accum_isnull,accum_value,								\
		new_isnull,__int_as_float((cl_uint)(new_value)),			\
		CHECK_OVERFLOW_NONE,										\
		pg_atomic_min_float((cl_float *)(accum_value),				\
							__int_as_float((cl_uint)(new_value))))
#define AGGCALC_GLOBAL_PMIN_DOUBLE(kcxt,accum_isnull,accum_value,	\
								   new_isnull,new_value)			\
	AGGCALC_GLOBAL_TEMPLATE_DOUBLE(									\
		kcxt, accum_isnull, accum_value,							\
		new_isnull,__longlong_as_double(new_value),					\
		CHECK_OVERFLOW_NONE,										\
		pg_atomic_min_double((cl_double *)(accum_value),			\
							 __longlong_as_double(new_value)))
#define AGGCALC_GLOBAL_PMIN_NUMERIC(kcxt,accum_isnull,accum_value,	\
									new_isnull,new_value)			\
	AGGCALC_GLOBAL_TEMPLATE_NUMERIC(								\
		kcxt,accum_isnull,accum_value,								\
		new_isnull,(cl_ulong)(new_value),CHECK_OVERFLOW_NONE,		\
		pg_atomic_min_numeric((kcxt),(cl_ulong *)(accum_value),		\
							  (cl_ulong)(new_value)))
/* calculation for global partial add */
#define AGGCALC_GLOBAL_PADD_SHORT(kcxt,accum_isnull,accum_value,	\
								  new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_SHORT(									\
		kcxt,accum_isnull,accum_value,								\
		new_isnull,(cl_int)(new_value),								\
		CHECK_OVERFLOW_SHORT,										\
		pg_atomic_add_int((cl_int *)(accum_value), (cl_int)(new_value)))
#define AGGCALC_GLOBAL_PADD_INT(kcxt,accum_isnull,accum_value,		\
								new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_INT(									\
		kcxt,accum_isnull,accum_value,								\
		new_isnull,(cl_int)(new_value),								\
		CHECK_OVERFLOW_INT,											\
		pg_atomic_add_int((cl_int *)(accum_value), (cl_int)(new_value)))
#define AGGCALC_GLOBAL_PADD_LONG(kcxt,accum_isnull,accum_value,		\
								 new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_LONG(									\
		kcxt,accum_isnull,accum_value,								\
		new_isnull,(cl_long)(new_value),							\
		CHECK_OVERFLOW_INT,											\
		pg_atomic_add_long((cl_long *)(accum_value), (cl_long)(new_value)))
#define AGGCALC_GLOBAL_PADD_FLOAT(kcxt,accum_isnull,accum_value,	\
								  new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_FLOAT(									\
		kcxt,accum_isnull,accum_value,								\
	    new_isnull,__int_as_float((cl_uint)(new_value)),			\
		CHECK_OVERFLOW_FLOAT,										\
		pg_atomic_add_float((cl_float *)(accum_value),				\
							__int_as_float((cl_uint)(new_value))))
#define AGGCALC_GLOBAL_PADD_DOUBLE(kcxt,accum_isnull,accum_value,	\
								   new_isnull,new_value)			\
	AGGCALC_GLOBAL_TEMPLATE_DOUBLE(									\
		kcxt,accum_isnull,accum_value,								\
		new_isnull,__longlong_as_double(new_value),					\
		CHECK_OVERFLOW_FLOAT,										\
		pg_atomic_add_double((cl_double *)(accum_value),			\
							 __longlong_as_double(new_value)))
#define AGGCALC_GLOBAL_PADD_NUMERIC(kcxt,accum_isnull,accum_value,	\
									new_isnull,new_value)			\
	AGGCALC_GLOBAL_TEMPLATE_NUMERIC(								\
		kcxt,accum_isnull,accum_value,								\
		new_isnull,(cl_ulong)(new_value),							\
		CHECK_OVERFLOW_NUMERIC,										\
		pg_atomic_add_numeric((kcxt),(cl_ulong *)(accum_value),		\
							  (cl_ulong)(new_value)))

/*
 * Helper macros for gpupreagg_nogroup_calc
 */
#define AGGCALC_NOGROUP_TEMPLATE(TYPE,kcxt,							\
								 accum_isnull,accum_val,			\
								 newval_isnull,newval_val,			\
								 OVERFLOW_CHECK,FUNC_CALL)			\
	do {															\
		if (!(newval_isnull))										\
		{															\
			TYPE tmp = FUNC_CALL;									\
			if (OVERFLOW_CHECK((accum_val), (newval_val)))			\
			{														\
				STROM_SET_ERROR(&(kcxt)->e, StromError_CpuReCheck);	\
			}														\
			(accum_val)    = tmp;									\
			(accum_isnull) = false;									\
		}															\
	} while (0)

#define AGGCALC_NOGROUP_TEMPLATE_SHORT(kcxt,accum,newval,			\
									   OVERFLOW_CHECK,FUNC_CALL)	\
	AGGCALC_NOGROUP_TEMPLATE(cl_int,(kcxt),							\
							 (accum)->isnull,(accum)->int_val,		\
							 (newval)->isnull,(newval)->int_val,	\
							 OVERFLOW_CHECK,FUNC_CALL)
#define AGGCALC_NOGROUP_TEMPLATE_INT(kcxt,accum,newval,				\
									 OVERFLOW_CHECK,FUNC_CALL)		\
	AGGCALC_NOGROUP_TEMPLATE(cl_int,kcxt,							\
							 (accum)->isnull,(accum)->int_val,		\
							 (newval)->isnull,(newval)->int_val,	\
							 OVERFLOW_CHECK,FUNC_CALL)
#define AGGCALC_NOGROUP_TEMPLATE_LONG(kcxt,accum,newval,			\
									  OVERFLOW_CHECK,FUNC_CALL)		\
	AGGCALC_NOGROUP_TEMPLATE(cl_long,(kcxt),						\
							 (accum)->isnull,(accum)->long_val,		\
							 (newval)->isnull,(newval)->long_val,	\
							 OVERFLOW_CHECK,FUNC_CALL)
#define AGGCALC_NOGROUP_TEMPLATE_FLOAT(kcxt,accum,newval,			\
									   OVERFLOW_CHECK,FUNC_CALL)	\
	AGGCALC_NOGROUP_TEMPLATE(cl_float,(kcxt),						\
							 (accum)->isnull,(accum)->float_val,	\
							 (newval)->isnull,(newval)->float_val,	\
							 OVERFLOW_CHECK,FUNC_CALL)
#define AGGCALC_NOGROUP_TEMPLATE_DOUBLE(kcxt,accum,newval,			\
										OVERFLOW_CHECK,FUNC_CALL)	\
	AGGCALC_NOGROUP_TEMPLATE(cl_double,(kcxt),						\
							 (accum)->isnull,(accum)->double_val,	\
							 (newval)->isnull,(newval)->double_val,	\
							 OVERFLOW_CHECK,FUNC_CALL)

#ifdef PG_NUMERIC_TYPE_DEFINED
#define AGGCALC_NOGROUP_TEMPLATE_NUMERIC(kcxt,accum_val,new_val,	\
										 FUNC_CALL)					\
	do {															\
		pg_numeric_t	x, y, z;									\
																	\
		if (!(new_val)->isnull)										\
		{															\
			x.isnull = false;										\
			x.value = (accum_val)->ulong_val;						\
			y.isnull = false;										\
			y.value = (new_val)->ulong_val;							\
			z = FUNC_CALL((kcxt), x, y);							\
			if (z.isnull)											\
				STROM_SET_ERROR(&(kcxt)->e, StromError_CpuReCheck);	\
			(accum_val)->ulong_val = z.value;						\
			(accum_val)->isnull = z.isnull;							\
		}															\
	} while (0)
#endif

/* calculation for no group partial max */
#define AGGCALC_NOGROUP_PMAX_SHORT(kcxt,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_SHORT((kcxt),accum,newval,		\
								   CHECK_OVERFLOW_NONE,		\
								   Max((accum)->int_val,	\
									   (newval)->int_val))
#define AGGCALC_NOGROUP_PMAX_INT(kcxt,accum,newval)			\
	AGGCALC_NOGROUP_TEMPLATE_INT((kcxt),accum,newval,		\
								 CHECK_OVERFLOW_NONE,		\
								 Max((accum)->int_val,		\
									 (newval)->int_val))
#define AGGCALC_NOGROUP_PMAX_LONG(kcxt,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_LONG((kcxt),accum,newval,		\
								  CHECK_OVERFLOW_NONE,		\
								  Max((accum)->long_val,	\
									  (newval)->long_val))
#define AGGCALC_NOGROUP_PMAX_FLOAT(kcxt,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_FLOAT((kcxt),accum,newval,		\
								   CHECK_OVERFLOW_NONE,		\
								   Max((accum)->float_val,	\
									   (newval)->float_val))
#define AGGCALC_NOGROUP_PMAX_DOUBLE(kcxt,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_DOUBLE((kcxt),accum,newval,	\
									CHECK_OVERFLOW_NONE,	\
									Max((accum)->double_val,\
										(newval)->double_val))
#define AGGCALC_NOGROUP_PMAX_NUMERIC(kcxt,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_NUMERIC((kcxt),accum,newval,	\
									 pgfn_numeric_max)

/* calculation for no group partial min */
#define AGGCALC_NOGROUP_PMIN_SHORT(kcxt,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_SHORT((kcxt),accum,newval,		\
								   CHECK_OVERFLOW_NONE,		\
								   Min((accum)->int_val,	\
									   (newval)->int_val))
#define AGGCALC_NOGROUP_PMIN_INT(kcxt,accum,newval)			\
	AGGCALC_NOGROUP_TEMPLATE_INT((kcxt),accum,newval,		\
								 CHECK_OVERFLOW_NONE,		\
								 Min((accum)->int_val,		\
									 (newval)->int_val))
#define AGGCALC_NOGROUP_PMIN_LONG(kcxt,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_LONG((kcxt),accum,newval,		\
								  CHECK_OVERFLOW_NONE,		\
								  Min((accum)->long_val,	\
									  (newval)->long_val))
#define AGGCALC_NOGROUP_PMIN_FLOAT(kcxt,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_FLOAT((kcxt),accum,newval,		\
								   CHECK_OVERFLOW_NONE,		\
								   Min((accum)->float_val,	\
									   (newval)->float_val))
#define AGGCALC_NOGROUP_PMIN_DOUBLE(kcxt,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_DOUBLE((kcxt),accum,newval,	\
									CHECK_OVERFLOW_NONE,	\
									Min((accum)->double_val,\
										(newval)->double_val))
#define AGGCALC_NOGROUP_PMIN_NUMERIC(kcxt,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_NUMERIC((kcxt),accum,newval,	\
									 pgfn_numeric_min)

/* calculation for no group partial add */
#define AGGCALC_NOGROUP_PADD_SHORT(kcxt,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_SHORT((kcxt),accum,newval,		\
								   CHECK_OVERFLOW_SHORT,	\
								   Add((accum)->int_val,	\
									   (newval)->int_val))
#define AGGCALC_NOGROUP_PADD_INT(kcxt,accum,newval)			\
	AGGCALC_NOGROUP_TEMPLATE_INT((kcxt),accum,newval,		\
								 CHECK_OVERFLOW_INT,		\
								 Add((accum)->int_val,		\
									 (newval)->int_val))
#define AGGCALC_NOGROUP_PADD_LONG(kcxt,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_LONG((kcxt),accum,newval,		\
								  CHECK_OVERFLOW_INT,		\
								  Add((accum)->long_val,	\
									  (newval)->long_val))
#define AGGCALC_NOGROUP_PADD_FLOAT(kcxt,accum,newval)		\
    AGGCALC_NOGROUP_TEMPLATE_FLOAT((kcxt),accum,newval,		\
								   CHECK_OVERFLOW_FLOAT,	\
								   Add((accum)->float_val,	\
									   (newval)->float_val))
#define AGGCALC_NOGROUP_PADD_DOUBLE(kcxt,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_DOUBLE((kcxt),accum,newval,	\
									CHECK_OVERFLOW_FLOAT,	\
									Add((accum)->double_val,\
										(newval)->double_val))
#define AGGCALC_NOGROUP_PADD_NUMERIC(kcxt,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_NUMERIC((kcxt),accum,newval,	\
									 pgfn_numeric_add)

#endif	/* __CUDACC__ */
#endif	/* CUDA_GPUPREAGG_H */
