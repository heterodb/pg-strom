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
	cl_uint			hash_size;				/* size of global hash-slots */
	char			__padding__[4];			/* alignment */
	cl_uint			pg_crc32_table[256];	/* master CRC32 table */
	kern_parambuf	kparams;
	/*
	 * kern_resultbuf with nrels==1 shall be located next to kern_parambuf
	 */
} kern_gpupreagg;

/* macro definitions to reference packed values */
#define KERN_GPUPREAGG_PARAMBUF(kgpreagg)			\
	((kern_parambuf *)(&(kgpreagg)->kparams))
#define KERN_GPUPREAGG_PARAMBUF_LENGTH(kgpreagg)	\
	(KERN_GPUPREAGG_PARAMBUF(kgpreagg)->length)
#define KERN_GPUPREAGG_RESULTBUF(kgpreagg)							\
	((kern_resultbuf *)((char *)KERN_GPUPREAGG_PARAMBUF(kgpreagg)	\
						+ KERN_GPUPREAGG_PARAMBUF_LENGTH(kgpreagg)))
#define KERN_GPUPREAGG_LENGTH(kgpreagg,nitems)			\
	((uintptr_t)(KERN_GPUPREAGG_RESULTBUF(kgpreagg)->results + (nitems)) - \
	 (uintptr_t)(kgpreagg))

#define KERN_GPUPREAGG_DMASEND_OFFSET(kgpreagg)			0
#define KERN_GPUPREAGG_DMASEND_LENGTH(kgpreagg)			\
	(offsetof(kern_gpupreagg, kparams) +				\
	 KERN_GPUPREAGG_PARAMBUF_LENGTH(kgpreagg))
#define KERN_GPUPREAGG_DMARECV_OFFSET(kgpreagg)			\
	((uintptr_t)KERN_GPUPREAGG_RESULTBUF(kgpreagg) -	\
	 (uintptr_t)(kgpreagg))
#define KERN_GPUPREAGG_DMARECV_LENGTH(kgpreagg,nitems)	\
	KERN_GPUPREAGG_LENGTH(kgpreagg,nitems)

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
	};
} pagg_hashslot;

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
#define GPUPREAGG_FIELD_IS_NULL			0
#define GPUPREAGG_FIELD_IS_GROUPKEY		1
#define GPUPREAGG_FIELD_IS_AGGFUNC		2

#ifdef __CUDACC__

/* macro to check overflow on accumlate operation*/
#define CHECK_OVERFLOW_NONE(x,y)		(0)

#define CHECK_OVERFLOW_SHORT(x,y)				\
	(((x)+(y)) < SHRT_MIN || SHRT_MAX < ((x)+(y)))

#define CHECK_OVERFLOW_INT(x,y)					\
	((((x) < 0) == ((y) < 0)) && (((x) + (y) < 0) != ((x) < 0)))
	
#define CHECK_OVERFLOW_FLOAT(x,y)				\
	(isinf((x) + (y)) && !isinf(x) && !isinf(y))

#define CHECK_OVERFLOW_NUMERIC(x,y)		CHECK_OVERFLOW_NONE(x,y)


/*
 * hash value calculation function - to be generated by PG-Strom on the fly
 */
STATIC_FUNCTION(cl_uint)
gpupreagg_hashvalue(cl_int *errcode,
					cl_uint *crc32_table,	/* __shared__ memory */
					kern_data_store *kds,
					kern_data_store *ktoast,
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
gpupreagg_keymatch(cl_int *errcode,
				   kern_data_store *kds,
				   kern_data_store *ktoast,
				   size_t x_index,
				   size_t y_index);

/*
 * local calculation function - to be generated by PG-Strom on the fly
 *
 * It aggregates the newval to accum using atomic operation on the
 * local pagg_datum array
 */
STATIC_FUNCTION(void)
gpupreagg_local_calc(cl_int *errcode,
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
gpupreagg_global_calc(cl_int *errcode,
					  cl_int attnum,
					  kern_data_store *kds,
					  kern_data_store *ktoast,
					  size_t accum_index,
					  size_t newval_index);

/*
 * Reduction operation with no atomic operations. It can be used if no
 * GROUP-BY clause is given, because all the rows shall be eventually
 * consolidated into one aggregated row.
 */
STATIC_FUNCTION(void)
gpupreagg_nogroup_calc(cl_int *errcode,
					   cl_int attnum,
					   pagg_datum *accum,
					   pagg_datum *newval);

/*
 * translate a kern_data_store (input) into an output form
 * (auto generated function)
 */
STATIC_FUNCTION(void)
gpupreagg_projection(cl_int *errcode,
					 kern_parambuf *kparams,
					 kern_data_store *kds_in,
					 kern_data_store *kds_src,
					 kern_data_store *ktoast,	/* never used */
					 size_t rowidx_in,
					 size_t rowidx_out);
/*
 * check qualifiers being pulled-up from the outer relation.
 * if not valid, this record shall not be processed.
 */
STATIC_FUNCTION(bool)
gpupreagg_qual_eval(cl_int *errcode,
					kern_parambuf *kparams,
					kern_data_store *kds,
					kern_data_store *ktoast,
					size_t kds_index);

/*
 * load the data from kern_data_store to pagg_datum structure
 */
STATIC_FUNCTION(void)
gpupreagg_data_load(pagg_datum *pdatum,		/* __shared__ */
					cl_int *errcode,
					kern_data_store *kds,
					kern_data_store *ktoast,
					cl_uint colidx, cl_uint rowidx)
{
	kern_colmeta	cmeta;
	Datum		   *values;
	cl_char		   *isnull;

	if (kds->format != KDS_FORMAT_TUPSLOT ||
		colidx >= kds->ncols)
	{
		STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
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
		STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
	}
}

/*
 * store the data from pagg_datum structure to kern_data_store
 */
STATIC_FUNCTION(void)
gpupreagg_data_store(pagg_datum *pdatum,	/* __shared__ */
					 cl_int *errcode,
					 kern_data_store *kds,
					 kern_data_store *ktoast,
					 cl_uint colidx, cl_uint rowidx)
{
	kern_colmeta	cmeta;
	Datum		   *values;
	cl_char		   *isnull;

	if (kds->format != KDS_FORMAT_TUPSLOT ||
		colidx >= kds->ncols)
	{
		STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
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
		STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
	}
}

/* gpupreagg_data_move - it moves grouping key from the source kds to
 * the destination kds as is. We assume toast buffer is shared and
 * resource number of varlena key is not changed. So, all we need to
 * do is copying the offset value, not varlena body itself.
 */
STATIC_FUNCTION(void)
gpupreagg_data_move(cl_int *errcode,
					kern_data_store *kds_src,
					kern_data_store *kds_dst,
					kern_data_store *ktoast,
					cl_uint colidx,
					cl_uint rowidx_src,
					cl_uint rowidx_dst)
{
	Datum	   *src_values;
	Datum	   *dst_values;
	cl_char	   *src_isnull;
	cl_char	   *dst_isnull;

	/*
	 * XXX - Paranoire checks?
	 */
	if (kds_src->format != KDS_FORMAT_TUPSLOT ||
		kds_dst->format != KDS_FORMAT_TUPSLOT)
	{
		STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
		return;
	}
	if (colidx >= kds_src->ncols || colidx >= kds_dst->ncols)
	{
		STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
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
					  pagg_hashslot *g_hashslot)
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_resultbuf *kresults = KERN_GPUPREAGG_RESULTBUF(kgpreagg);
	cl_int			errcode = StromError_Success;
	cl_uint			offset;
	cl_uint			nitems;
	size_t			kds_index = get_global_id();
	size_t			hash_size;
	size_t			hash_index;
	__shared__ cl_uint base;

	/* init global hash slot */
	hash_size = kgpreagg->hash_size;;
	for (hash_index = get_global_id(0);
		 hash_index < hash_size;
		 hash_index += get_global_size(0))
	{
		g_hashslot[hash_index].hash = 0;
		g_hashslot[hash_index].index = (cl_uint)(0xffffffff);
	}

	/* check qualifiers */
	if (kds_index < kds_in->nitems)
	{
		if (!gpupreagg_qual_eval(&errcode, kparams, kds_in, NULL, kds_index))
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

	/* out of range check -- usually, should not happen */
	if (base + nitems > kds_src->nrooms)
	{
		errcode = StromError_DataStoreNoSpace;
		goto out;
	}

	/* do projection */
	if (kds_index < kds_in->nitems)
	{
		gpupreagg_projection(&errcode,
							 kparams,
							 kds_in,			/* input kds */
							 kds_src,			/* source of reduction kds */
							 NULL,				/* never use toast */
							 kds_index,			/* rowidx of kds_in */
							 base + offset);	/* rowidx of kds_src */
	}
out:
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kresults->errcode, errcode);
}

/*
 * gpupreagg_local_reduction
 */
KERNEL_FUNCTION(void)
gpupreagg_local_reduction(kern_gpupreagg *kgpreagg,
						  kern_data_store *kds_src,
						  kern_data_store *kds_dst,
						  kern_data_store *ktoast)
{
	kern_parambuf	   *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	varlena			   *kparam_0 = kparam_get_value(kparams, 0);
	cl_char			   *gpagg_atts = (cl_char *) VARDATA(kparam_0);
	size_t				hash_size = 2 * get_local_size();
	size_t				dest_index;
	cl_uint				owner_index;
	cl_uint				hash_value;
	cl_uint				nitems = kds_src->nitems;
	cl_uint				nattrs = kds_src->ncols;
	cl_uint				ngroups;
	cl_uint				index;
	cl_uint				attnum;
	cl_int				errcode = StromError_Success;
	pagg_hashslot		old_slot;
	pagg_hashslot		new_slot;
	pagg_hashslot		cur_slot;
	cl_uint			   *crc32_table;
	pagg_datum		   *l_datum;
	pagg_hashslot	   *l_hashslot;
	__shared__ size_t	base_index;

	/*
	 * calculation of the hash value of grouping keys in this record.
	 * It tends to take massive amount of random access on global memory,
	 * so it makes performance advantage to move the master table from
	 * gloabl to the local memory first.
	 */
	crc32_table = SHARED_WORKMEM(cl_uint);
	for (index = get_local_id();
		 index < lengthof(crc32_table);
		 index += get_local_size())
		crc32_table[index] = kgpreagg->pg_crc32_table[index];
	__syncthreads();

	if (get_global_id() < nitems)
		hash_value = gpupreagg_hashvalue(&errcode, crc32_table,
										 kds_src, ktoast,
										 get_global_id());

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
		l_hashslot[index].hash = 0;
		l_hashslot[index].index = (cl_uint)(0xffffffff);
	}
	__syncthreads();

	if (get_global_id() < nitems)
	{
		new_slot.hash = hash_value;
		new_slot.index = get_local_id(0);
		old_slot.hash = 0;
		old_slot.index = (cl_uint)(0xffffffff);
		index = hash_value % hash_size;

	retry:
		cur_slot.value = atom_cmpxchg(&l_hashslot[index].value,
									  old_slot.value,
									  new_slot.value);
		if (cur_slot.value == old_slot.value)
		{
			/* Hash slot was empty, so this thread shall be responsible
			 * to this grouping-key.
			 */
			owner_index = new_slot.index;
		}
		else
		{
			size_t	buddy_index
				= (get_global_id() - get_local_id() + cur_slot.index);

			if (cur_slot.hash == new_slot.hash &&
				gpupreagg_keymatch(&errcode,
								   kds_src, ktoast,
								   get_global_id(),
								   buddy_index))
			{
				owner_index = cur_slot.index;
			}
			else
			{
				index = (index + 1) % hash_size;
				goto retry;
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
	if (kds_dst->nrooms < base_index + ngroups)
	{
		errcode = StromError_DataStoreNoSpace;
		goto out;
	}
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
				gpupreagg_data_move(&errcode,
									kds_src, kds_dst, ktoast,
									attnum,
									get_global_id(0),
									dest_index);
			}
			continue;
		}

		/* Load aggregation item to pagg_datum */
		if (get_global_id() < nitems)
		{
			gpupreagg_data_load(l_datum + get_local_id(),
								&errcode,
								kds_src, ktoast,
								attnum, get_global_id());
		}
		__syncthreads();

		/* Reduction, using local atomic operation */
		if (get_global_id() < nitems &&
			get_local_id() != owner_index)
		{
			gpupreagg_local_calc(&errcode,
								 attnum,
								 l_datum + owner_index,
								 l_datum + get_local_id());
		}
		__syncthreads();

		/* Move the value that is aggregated */
		if (owner_index == get_local_id())
		{
			gpupreagg_data_store(l_datum + owner_index,
								 &errcode,
								 kds_dst, ktoast,
								 attnum, dest_index);
			/*
			 * varlena should never appear here, so we don't need to
			 * put pg_fixup_tupslot_varlena() here
			 */
		}
		__syncthreads();
	}
out:
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kresults->errcode, errcode);
}

/*
 * gpupreagg_global_reduction
 */
KERNEL_FUNCTION(void)
gpupreagg_global_reduction(kern_gpupreagg *kgpreagg,
						   kern_data_store *kds_dst,
						   kern_data_store *ktoast,
						   pagg_hashslot *g_hashslot)
{
	kern_parambuf	   *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_resultbuf	   *kresults = KERN_GPUPREAGG_RESULTBUF(kgpreagg);
	varlena			   *kparam_0 = kparam_get_value(kparams, 0);
	cl_char			   *gpagg_atts = (cl_char *) VARDATA(kparam_0);
	size_t				hash_size = kgpreagg->hash_size;
	size_t				dest_index;
	size_t				owner_index;
	cl_uint				hash_value;
	cl_uint				nitems = kds_dst->nitems;
	cl_uint				ngroups;
	cl_uint				index;
	cl_uint				nattrs = kds_dst->ncols;
	cl_uint				attnum;
	cl_int				errcode = StromError_Success;
	pagg_hashslot		old_slot;
	pagg_hashslot		new_slot;
	pagg_hashslot		cur_slot;
	cl_uint			   *crc32_table;
	__shared__ size_t	base_index;

	/*
	 * calculation of the hash value of grouping keys in this record.
	 * It tends to take massive amount of random access on global memory,
	 * so it makes performance advantage to move the master table from
	 * gloabl to the local memory first.
	 */
	crc32_table = SHARED_WORKMEM(cl_uint);	/* 1KB */
	for (index = get_local_id();
		 index < lengthof(crc32_table);
		 index += get_local_size())
		crc32_table[index] = kgpreagg->pg_crc32_table[index];
	__syncthreads();

	if (get_global_id() < nitems)
	{
		hash_value = gpupreagg_hashvalue(&errcode, crc32_table,
										 kds_dst, ktoast,
										 get_global_id());
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
		new_slot.hash = hash_value;
		new_slot.index = get_global_id(0);
		old_slot.hash = 0;
		old_slot.index = (cl_uint)(0xffffffff);
		index = hash_value % hash_size;
	retry:
		cur_slot.value = atom_cmpxchg(&g_hashslot[index].value,
									  old_slot.value,
									  new_slot.value);
		if (cur_slot.value == old_slot.value)
		{
			/* Hash slot was empty, so this thread shall be responsible
			 * to this grouping-key.
			 */
			owner_index = new_slot.index;
		}
		else if (cur_slot.hash == new_slot.hash &&
				 gpupreagg_keymatch(&errcode,
									kds_dst, ktoast,
									get_global_id(),
									cur_slot.index))
		{
			owner_index = cur_slot.index;
		}
		else
		{
			index = (index + 1) % hash_size;
			goto retry;
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
	if (get_local_id() == 0)
		base_index = atomicAdd(&kresults->nitems, ngroups);
	__syncthreads();
	if (kresults->nrooms <= base_index + ngroups)
	{
		errcode = StromError_DataStoreNoSpace;
		goto out;
	}
	dest_index = base_index + index;

	if (get_global_id() < nitems &&
		get_global_id() == owner_index)
		kresults->results[dest_index] = get_global_id();

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
		if (get_global_id(0) < nitems &&
			get_global_id(0) != owner_index)
		{
			gpupreagg_global_calc(&errcode,
								  attnum,
								  kds_dst,
								  ktoast,
								  owner_index,
								  get_global_id());
		}
	}
out:
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kresults->errcode, errcode);
}

/*
 * gpupreagg_nogroup_reduction
 *
 * It makes aggregation if no GROUP-BY clause given. We can omit atomic-
 * operations in this case, because all the rows are eventually consolidated
 * to just one record, thus usual reduction operation is sufficient.
 */
KERNEL_FUNCTION(void)
gpupreagg_nogroup_reduction(kern_gpupreagg *kgpreagg,
							kern_data_store *kds_src,
							kern_data_store *kds_dst,
							kern_data_store *ktoast)
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_resultbuf *kresults = KERN_GPUPREAGG_RESULTBUF(kgpreagg);
	varlena		   *kparam_0   = kparam_get_value(kparams, 0);
	cl_char		   *gpagg_atts = (cl_char *)VARDATA(kparam_0);
	pagg_datum	   *l_datum = SHARED_WORKMEM(pagg_datum);
	cl_uint			nitems = kds_src->nitems;
	cl_uint			nattrs = kds_src->ncols;
	cl_int			errcode = StromError_Success;
	size_t			lid = get_local_id();
	size_t			gid = get_global_id();
	size_t			lsz = get_local_size();
	size_t			dest_index	= gid / lsz;
	int				attnum;

	/* loop for each columns */
	for (attnum = 0; attnum < nattrs; attnum++)
	{
		size_t	distance;

		/* if not GPUPREAGG_FIELD_IS_AGGFUNC, do nothing */
		if (gpagg_atts[attnum] != GPUPREAGG_FIELD_IS_AGGFUNC)
		{
			if (gid < nitems  && lid == 0)
				gpupreagg_data_move(&errcode, kds_src, kds_dst, ktoast,
									attnum,	gid, dest_index);
			continue;
		}

		/* load this value from kds_src onto datum */
		if (gid < nitems)
			gpupreagg_data_load(&l_datum[get_lobal_id()], &errcode,
								kds_src, ktoast, attnum, gid);
		__syncthreads();

		/* do reduction */
		for (distance = 2; distance <= lsz; distance *= 2)
		{
			if (lid % distance == 0 && (gid + distance / 2) < nitems)
				gpupreagg_nogroup_calc(&errcode,
									   attnum,
									   &l_datum[lid],
									   &l_datum[lid + distance / 2]);
			__syncthreads();
		}

		/* store this value to kds_dst from datum */
		if (gid < nitems  &&  lid == 0)
			gpupreagg_data_store(&l_datum[lid], &errcode,
								 kds_dst, ktoast, attnum, dest_index);
		__syncthreads();
	}

	/*
	 * Fixup kern_rowmap/kds->nitems
	 */
	if (gid == 0)
	{
		krowmap->nvalids = (nitems + lsz - 1) / lsz;
		kds_dst->nitems = (nitems + lsz - 1) / lsz;
	}
	if (lid == 0)
	{
		krowmap->rindex[dest_index] = dest_index;
	}

	/* write-back execution status into host-side */
	kern_writeback_error_status(&kresults->errcode, errcode);
}


/*
 * gpupreagg_fixup_varlena
 *
 * In case when varlena datum (excludes numeric) is used in grouping-key,
 * datum on kds with tupslot format has not-interpretable for host systems.
 * So, we need to fix up its value to adjust offset by hostptr.
 */
KERNEL_FUNCTION(void)
gpupreagg_fixup_varlena(kern_gpupreagg *kgpreagg,
						kern_data_store *kds_dst,
						kern_data_store *ktoast)
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_resultbuf *kresults = KERN_GPUPREAGG_RESULTBUF(kgpreagg);
	varlena		   *kparam_0 = kparam_get_value(kparams, 0);
	cl_char		   *gpagg_atts = (cl_char *) VARDATA(kparam_0);
	cl_int			errcode = StromError_Success;
	cl_uint			nattrs = kds_dst->ncols;
	cl_uint			nitems = krowmap->nvalids;

	if (get_global_id() < nitems)
	{
		cl_uint			attnum;
		cl_uint			rowidx;

		for (attnum = 0; attnum < nattrs; attnum++)
		{
			if (gpagg_atts[attnum] != GPUPREAGG_FIELD_IS_GROUPKEY)
				continue;

			rowidx = krowmap->rindex[get_global_id(0)];
			pg_fixup_tupslot_varlena(&errcode,
									 kds_dst, ktoast,
									 attnum, rowidx);
		}
	}
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kresults->errcode, errcode);
}

/* ----------------------------------------------------------------
 *
 * Own version of atomic functions; for float, double and numeric
 *
 * ----------------------------------------------------------------
 */
#define add(x,y)	(x)+(y)

#define ATOMIC_FLOAT_TEMPLATE(op_name)									\
	STATIC_INLINE(float)												\
	atomic_##op_name##_float(volatile float *ptr, float value)			\
	{																	\
		cl_int		curval = __float_as_int(*ptr);						\
		cl_int		oldval;												\
		cl_int		newval;												\
																		\
		do {															\
			oldval = curval;											\
			newval = __float_as_int(op_name(__int_as_float(oldval),		\
											value));					\
		} while ((curval = atomicCAS((cl_int *) ptr,					\
									 oldval, newval)) != oldval);		\
		return __int_as_float(oldval);									\
	}

ATOMIC_FLOAT_TEMPLATE(max)
ATOMIC_FLOAT_TEMPLATE(min)
#if __CUDA_ARCH__ < 350
ATOMIC_FLOAT_TEMPLATE(add)
#else
STATIC_INLINE(float)
atomic_add_float(volatile float *ptr, float value)
{
	return atomicAdd(ptr, value);
}
#endif

#define ATOMIC_DOUBLE_TEMPLATE(prefix, op_name)							\
	STATIC_INLINE(double)												\
	atomic_##op_name##_double(volatile double *ptr, double value)		\
	{																	\
		cl_long		curval = __double_as_longlong(*ptr);				\
		cl_long		oldval;												\
		cl_long		newval;												\
		double		temp;												\
																		\
		do {															\
			oldval = curval;											\
			temp = op_name(__longlong_as_double(oldval), temp);			\
			newval = __double_as_longlong(temp);						\
		} while ((curval = atomicCAS((cl_long *) ptr,					\
									 oldval, newval)) != oldval);		\
		return __longlong_as_double(oldval);							\
	}

ATOMIC_DOUBLE_TEMPLATE(min)
ATOMIC_DOUBLE_TEMPLATE(max)
ATOMIC_DOUBLE_TEMPLATE(add)

#undef add

#ifdef PG_NUMERIC_TYPE_DEFINED

STATIC_INLINE(cl_ulong)
atomic_min_numeric(cl_int *errcode,
				   volatile cl_ulong *ptr,
				   cl_ulong numeric_value)
{
	pg_numeric_t	x, y;
	pg_int4_t		comp;
	cl_ulong		oldval;
	cl_ulong		curval = *ptr;
	cl_ulong		newval;

	do {
		x.isnull = false;
		y.isnull = false;
		x.value = oldval = curval;
		y.value = numeric_value;
		comp = pgfn_numeric_cmp(errcode, x, y);
		if (comp.value < 0)
			break;
	} while ((curval = atomicCAS(ptr, oldval, newval)) != oldval);

	return oldval;
}

STATIC_INLINE(cl_ulong)
atomic_max_numeric(cl_int *errcode,
				   volatile cl_ulong *ptr,
				   cl_ulong numeric_value)
{
	pg_numeric_t	x, y;
	pg_int4_t		comp;
	cl_ulong		oldval;
	cl_ulong		curval = *ptr;
	cl_ulong		newval;

	do {
		x.isnull = false;
		y.isnull = false;
		x.value = oldval = curval;
		y.value = numeric_value;
		comp = pgfn_numeric_cmp(errcode, x, y);
		if (comp.value > 0)
			break;
	} while ((curval = atomicCAS(ptr, oldval, newval)) != oldval);

	return oldval;
}

STATIC_INLINE(cl_ulong)
atomic_add_numeric(cl_int *errcode,
				   volatile cl_ulong *ptr,
				   cl_ulong numeric_value)
{
	pg_numeric_t x, y, z;
	cl_ulong	oldval;
	cl_ulong	curval = *ptr;
	cl_ulong	newval;

	do {
		x.isnull = false;
		y.isnull = false;
		x.value = oldval = curval;
		y.value = numeric_value;
		z = pgfn_numeric_add(errcode, x, y);
		newval = z.value;
	} while ((curval = atomicCAS(ptr, oldval, newval)) != oldval);

	return oldval;
}
#endif

/*
 * Helper macros for gpupreagg_local_calc
 */
#define AGGCALC_LOCAL_TEMPLATE(TYPE,errcode,							\
							   accum_isnull,accum_val,					\
							   newval_isnull,newval_val,				\
							   OVERFLOW,ATOMIC_FUNC_CALL)				\
	do {																\
		if (!(newval_isnull))											\
		{																\
			TYPE old = ATOMIC_FUNC_CALL;								\
			if (OVERFLOW(old, (newval_val)))							\
			{															\
				STROM_SET_ERROR(errcode, StromError_CpuReCheck);		\
			}															\
			(accum_isnull) = false;										\
		}																\
	} while (0)

#define AGGCALC_LOCAL_TEMPLATE_SHORT(errcode,accum,newval,				\
									 OVERFLOW,ATOMIC_FUNC_CALL)			\
	AGGCALC_LOCAL_TEMPLATE(cl_int,errcode,								\
						   (accum)->isnull,(accum)->int_val,			\
						   (newval)->isnull,(newval)->int_val,			\
						   OVERFLOW,ATOMIC_FUNC_CALL)
#define AGGCALC_LOCAL_TEMPLATE_INT(errcode,accum,newval,				\
								   OVERFLOW,ATOMIC_FUNC_CALL)			\
	AGGCALC_LOCAL_TEMPLATE(cl_int,errcode,								\
						   (accum)->isnull,(accum)->int_val,			\
						   (newval)->isnull,(newval)->int_val,			\
						   OVERFLOW,ATOMIC_FUNC_CALL)
#define AGGCALC_LOCAL_TEMPLATE_LONG(errcode,accum,newval,				\
									OVERFLOW,ATOMIC_FUNC_CALL)			\
	AGGCALC_LOCAL_TEMPLATE(cl_long,errcode,								\
						   (accum)->isnull,(accum)->long_val,			\
						   (newval)->isnull,(newval)->long_val,			\
						   OVERFLOW,ATOMIC_FUNC_CALL)
#define AGGCALC_LOCAL_TEMPLATE_FLOAT(errcode,accum,newval,				\
									 OVERFLOW,ATOMIC_FUNC_CALL)			\
	AGGCALC_LOCAL_TEMPLATE(cl_float,errcode,							\
						   (accum)->isnull,(accum)->float_val,			\
						   (newval)->isnull,(newval)->float_val,		\
						   OVERFLOW,ATOMIC_FUNC_CALL)
#define AGGCALC_LOCAL_TEMPLATE_DOUBLE(errcode,accum,newval,				\
									  OVERFLOW,ATOMIC_FUNC_CALL)		\
	AGGCALC_LOCAL_TEMPLATE(cl_double,errcode,							\
						   (accum)->isnull,(accum)->double_val,			\
						   (newval)->isnull,(newval)->double_val,		\
						   OVERFLOW,ATOMIC_FUNC_CALL)
#define AGGCALC_LOCAL_TEMPLATE_NUMERIC(errcode,accum,newval,			\
									   OVERFLOW,ATOMIC_FUNC_CALL)		\
	AGGCALC_LOCAL_TEMPLATE(cl_ulong,errcode,							\
						   (accum)->isnull,(accum)->ulong_val,			\
						   (newval)->isnull,(newval)->ulong_val,		\
						   OVERFLOW,ATOMIC_FUNC_CALL)

/* calculation for local partial max */
#define AGGCALC_LOCAL_PMAX_SHORT(errcode,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_SHORT(errcode,accum,newval,CHECK_OVERFLOW_NONE,	\
			atomicMax(&(accum)->int_val, (newval)->int_val))
#define AGGCALC_LOCAL_PMAX_INT(errcode,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_INT(errcode,accum,newval,CHECK_OVERFLOW_NONE, \
			atomicMax(&(accum)->int_val, (newval)->int_val))
#define AGGCALC_LOCAL_PMAX_LONG(errcode,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_LONG(errcode,accum,newval,CHECK_OVERFLOW_NONE, \
			atomicMax(&(accum)->long_val, (newval)->long_val))
#define AGGCALC_LOCAL_PMAX_FLOAT(errcode,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_FLOAT(errcode,accum,newval,CHECK_OVERFLOW_NONE, \
			atomic_max_float(&(accum)->float_val, (newval)->float_val))
#define AGGCALC_LOCAL_PMAX_DOUBLE(errcode,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_DOUBLE(errcode,accum,newval,CHECK_OVERFLOW_NONE, \
			atomic_max_double(&(accum)->double_val, (newval)->double_val))
#define AGGCALC_LOCAL_PMAX_NUMERIC(errcode,accum,newval)				\
	AGGCALC_LOCAL_TEMPLATE_NUMERIC(errcode,accum,newval,CHECK_OVERFLOW_NONE, \
			atomic_max_numeric((errcode), &(accum)->ulong_val,			\
							   (newval)->ulong_val))

/* calculation for local partial min */
#define AGGCALC_LOCAL_PMIN_SHORT(errcode,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_SHORT(errcode,accum,newval,CHECK_OVERFLOW_NONE, \
			atomicMin(&(accum)->int_val, (newval)->int_val))
#define AGGCALC_LOCAL_PMIN_INT(errcode,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_INT(errcode,accum,newval,CHECK_OVERFLOW_NONE, \
			atomicMin(&(accum)->int_val, (newval)->int_val))
#define AGGCALC_LOCAL_PMIN_LONG(errcode,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_LONG(errcode,accum,newval,CHECK_OVERFLOW_NONE, \
			atomicMin(&(accum)->long_val, (newval)->long_val))
#define AGGCALC_LOCAL_PMIN_FLOAT(errcode,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_FLOAT(errcode,accum,newval,CHECK_OVERFLOW_NONE, \
			atomic_min_float(&(accum)->float_val, (newval)->float_val))
#define AGGCALC_LOCAL_PMIN_DOUBLE(errcode,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_DOUBLE(errcode,accum,newval,CHECK_OVERFLOW_NONE, \
			atomic_min_double(&(accum)->double_val, (newval)->double_val))
#define AGGCALC_LOCAL_PMIN_NUMERIC(errcode,accum,newval)				\
	AGGCALC_LOCAL_TEMPLATE_NUMERIC(errcode,accum,newval,CHECK_OVERFLOW_NONE, \
			atomic_min_numeric((errcode), &(accum)->ulong_val,			\
							   (newval)->ulong_val))

/* calculation for local partial add */
#define AGGCALC_LOCAL_PADD_SHORT(errcode,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_SHORT(errcode,accum,newval,CHECK_OVERFLOW_SHORT,	\
			atomicAdd(&(accum)->int_val, (newval)->int_val))
#define AGGCALC_LOCAL_PADD_INT(errcode,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_INT(errcode,accum,newval,CHECK_OVERFLOW_INT,	\
			atomicAdd(&(accum)->int_val, (newval)->int_val))
#define AGGCALC_LOCAL_PADD_LONG(errcode,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_LONG(errcode,accum,newval,CHECK_OVERFLOW_INT, \
			atomAdd(&(accum)->long_val, (newval)->long_val))
#define AGGCALC_LOCAL_PADD_FLOAT(errcode,accum,newval)					\
    AGGCALC_LOCAL_TEMPLATE_FLOAT(errcode,accum,newval,CHECK_OVERFLOW_FLOAT, \
			atomic_add_float(&(accum)->float_val, (newval)->float_val))
#define AGGCALC_LOCAL_PADD_DOUBLE(errcode,accum,newval)					\
	AGGCALC_LOCAL_TEMPLATE_DOUBLE(errcode,accum,newval,CHECK_OVERFLOW_FLOAT, \
			atomic_add_double(&(accum)->double_val, (newval)->double_val))
#define AGGCALC_LOCAL_PADD_NUMERIC(errcode,accum,newval)				\
	AGGCALC_LOCAL_TEMPLATE_NUMERIC(errcode,accum,newval,				\
		    CHECK_OVERFLOW_NUMERIC,										\
			atomic_add_numeric((errcode),&(accum)->ulong_val,			\
							   (newval)->ulong_val))

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
#define AGGCALC_GLOBAL_TEMPLATE(TYPE,errcode,						\
								accum_isnull,accum_value,			\
								new_isnull,new_value,OVERFLOW,		\
								ATOMIC_FUNC_CALL)					\
	if (!(new_isnull))												\
	{																\
		TYPE old = ATOMIC_FUNC_CALL;								\
		if (OVERFLOW(old, (new_value)))								\
		{															\
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);		\
		}															\
		*(accum_isnull) = false;									\
	}

#define AGGCALC_GLOBAL_TEMPLATE_SHORT(errcode,						\
									  accum_isnull,accum_value,		\
									  new_isnull,new_value,			\
									  OVERFLOW,ATOMIC_FUNC_CALL)	\
	AGGCALC_GLOBAL_TEMPLATE(cl_int, errcode,						\
							accum_isnull,accum_value,				\
							new_isnull,new_value,					\
							OVERFLOW,ATOMIC_FUNC_CALL)
#define AGGCALC_GLOBAL_TEMPLATE_INT(errcode,						\
									accum_isnull,accum_value,		\
									new_isnull,new_value,			\
									OVERFLOW,ATOMIC_FUNC_CALL)		\
	AGGCALC_GLOBAL_TEMPLATE(cl_int,errcode,							\
							accum_isnull,accum_value,				\
							new_isnull,new_value,					\
							OVERFLOW,ATOMIC_FUNC_CALL)
#define AGGCALC_GLOBAL_TEMPLATE_LONG(errcode,						\
									 accum_isnull,accum_value,		\
									 new_isnull,new_value,			\
									 OVERFLOW,ATOMIC_FUNC_CALL)		\
	AGGCALC_GLOBAL_TEMPLATE(cl_long,errcode,						\
							accum_isnull,accum_value,				\
							new_isnull,new_value,					\
							OVERFLOW,ATOMIC_FUNC_CALL)
#define AGGCALC_GLOBAL_TEMPLATE_FLOAT(errcode,						\
									  accum_isnull,accum_value,		\
									  new_isnull,new_value,			\
									  OVERFLOW,ATOMIC_FUNC_CALL)	\
	AGGCALC_GLOBAL_TEMPLATE(cl_float,errcode,						\
							accum_isnull,accum_value,				\
							new_isnull,new_value,					\
							OVERFLOW,ATOMIC_FUNC_CALL)
#define AGGCALC_GLOBAL_TEMPLATE_DOUBLE(errcode,						\
									   accum_isnull,accum_value,	\
									   new_isnull,new_value,		\
									   OVERFLOW,ATOMIC_FUNC_CALL)	\
	AGGCALC_GLOBAL_TEMPLATE(cl_double,errcode,						\
							accum_isnull,accum_value,				\
							new_isnull,new_value,					\
							OVERFLOW,ATOMIC_FUNC_CALL)
#define AGGCALC_GLOBAL_TEMPLATE_NUMERIC(errcode,					\
										accum_isnull,accum_value,	\
										new_isnull,new_value,		\
										OVERFLOW,ATOMIC_FUNC_CALL)	\
	AGGCALC_GLOBAL_TEMPLATE(cl_ulong,errcode,						\
							accum_isnull,accum_value,				\
							new_isnull,new_value,					\
							OVERFLOW,ATOMIC_FUNC_CALL)

/* calculation for global partial max */
#define AGGCALC_GLOBAL_PMAX_SHORT(errcode,accum_isnull,accum_value,	\
								  new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_SHORT(									\
		errcode,accum_isnull,accum_value,							\
		new_isnull,(cl_int)(new_value), CHECK_OVERFLOW_NONE,		\
		atomicMax((cl_int *)(accum_value), (cl_int)(new_value)))
#define AGGCALC_GLOBAL_PMAX_INT(errcode,accum_isnull,accum_value,	\
								new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_INT(									\
		errcode,accum_isnull,accum_value,							\
		new_isnull,(cl_int)(new_value),CHECK_OVERFLOW_NONE,			\
		atomicMax((cl_int *)(accum_value), (cl_int)(new_value)))
#define AGGCALC_GLOBAL_PMAX_LONG(errcode,accum_isnull,accum_value,	\
								 new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_LONG(\
		errcode,accum_isnull,accum_value,							\
		new_isnull,(cl_long)(new_value),CHECK_OVERFLOW_NONE,		\
		atomicMax((cl_long *)(accum_value), (cl_long)(new_value)))
#define AGGCALC_GLOBAL_PMAX_FLOAT(errcode,accum_isnull,accum_value,	\
								  new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_FLOAT(									\
		errcode,accum_isnull,accum_value,							\
		new_isnull, __int_as_float((cl_uint)(new_value)),			\
		CHECK_OVERFLOW_NONE,										\
		atomic_max_float((cl_float *)(accum_value),					\
						 __int_as_float((cl_uint)(new_value))))
#define AGGCALC_GLOBAL_PMAX_DOUBLE(errcode,accum_isnull,accum_value,\
								   new_isnull,new_value)			\
	AGGCALC_GLOBAL_TEMPLATE_DOUBLE(									\
		errcode,accum_isnull,accum_value,							\
		new_isnull, __longlong_as_double(new_value),				\
		CHECK_OVERFLOW_NONE,										\
		atomic_max_double((cl_double *)(accum_value),				\
						  __longlong_as_double(new_value)))
#define AGGCALC_GLOBAL_PMAX_NUMERIC(errcode,accum_isnull,accum_value, \
									new_isnull,new_value)			\
	AGGCALC_GLOBAL_TEMPLATE_NUMERIC(								\
		errcode, accum_isnull, accum_value,							\
		new_isnull, (cl_ulong)(new_value),							\
		CHECK_OVERFLOW_NONE,										\
		atomic_max_numeric((errcode),(cl_ulong *)(accum_value),		\
						   (cl_ulong)(new_value)))
/* calculation for global partial min */
#define AGGCALC_GLOBAL_PMIN_SHORT(errcode,accum_isnull,accum_value,	\
								  new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_SHORT(									\
		errcode,accum_isnull,accum_value,							\
		new_isnull,(cl_int)(new_value),								\
		CHECK_OVERFLOW_NONE,										\
		atomicMin((cl_int *)(accum_value),(cl_int)(new_value)))
#define AGGCALC_GLOBAL_PMIN_INT(errcode,accum_isnull,accum_value,	\
								new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_INT(									\
		errcode,accum_isnull,accum_value,							\
		new_isnull,(cl_int)(new_value),								\
		CHECK_OVERFLOW_NONE,										\
		atomicMin((cl_int *)(accum_value),(cl_int)(new_value)))
#define AGGCALC_GLOBAL_PMIN_LONG(errcode,accum_isnull,accum_value,	\
								 new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_LONG(\
		errcode,accum_isnull,accum_value,							\
		new_isnull,(cl_long)(new_value),							\
		CHECK_OVERFLOW_NONE,										\
		atomicMin((cl_long *)(accum_value),(cl_long)(new_value)))
#define AGGCALC_GLOBAL_PMIN_FLOAT(errcode,accum_isnull,accum_value,	\
								  new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_FLOAT(									\
		errcode,accum_isnull,accum_value,							\
		new_isnull,__int_as_float((cl_uint)(new_value)),			\
		CHECK_OVERFLOW_NONE,										\
		atomic_min_float((cl_float *)(accum_value),					\
						 __int_as_float((cl_uint)(new_value))))
#define AGGCALC_GLOBAL_PMIN_DOUBLE(errcode,accum_isnull,accum_value,\
								   new_isnull,new_value)			\
	AGGCALC_GLOBAL_TEMPLATE_DOUBLE(									\
		errcode, accum_isnull, accum_value,							\
		new_isnull,__longlong_as_double(new_value),					\
		CHECK_OVERFLOW_NONE,										\
		atomic_min_double((cl_double *)(accum_value),				\
						  __longlong_as_double(new_value)))
#define AGGCALC_GLOBAL_PMIN_NUMERIC(errcode,accum_isnull,accum_value, \
									new_isnull,new_value)			\
	AGGCALC_GLOBAL_TEMPLATE_NUMERIC(								\
		errcode,accum_isnull,accum_value,							\
		new_isnull,(cl_ulong)(new_value),CHECK_OVERFLOW_NONE,		\
		atomic_min_numeric((errcode),(cl_ulong *)(accum_value),		\
						   (cl_ulong)(new_value)))
/* calculation for global partial add */
#define AGGCALC_GLOBAL_PADD_SHORT(errcode,accum_isnull,accum_value,	\
								  new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_SHORT(									\
		errcode,accum_isnull,accum_value,							\
		new_isnull,(cl_int)(new_value),								\
		CHECK_OVERFLOW_SHORT,										\
		atomicAdd((cl_int *)(accum_value), (cl_int)(new_value)))
#define AGGCALC_GLOBAL_PADD_INT(errcode,accum_isnull,accum_value,	\
								new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_INT(									\
		errcode,accum_isnull,accum_value,							\
		new_isnull,(cl_int)(new_value),								\
		CHECK_OVERFLOW_INT,											\
		atomicAdd((cl_int *)(accum_value), (cl_int)(new_value)))
#define AGGCALC_GLOBAL_PADD_LONG(errcode,accum_isnull,accum_value,	\
								 new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_LONG(									\
		errcode,accum_isnull,accum_value,							\
		new_isnull,(cl_long)(new_value),							\
		CHECK_OVERFLOW_INT,											\
		atomicAdd((cl_long *)(accum_value), (cl_long)(new_value)))
#define AGGCALC_GLOBAL_PADD_FLOAT(errcode,accum_isnull,accum_value,	\
								  new_isnull,new_value)				\
	AGGCALC_GLOBAL_TEMPLATE_FLOAT(									\
		errcode,accum_isnull,accum_value,							\
	    new_isnull,__int_as_float((cl_uint)(new_value)),			\
		CHECK_OVERFLOW_FLOAT,										\
		atomic_add_float((cl_float *)(accum_value),					\
						 __int_as_float((cl_uint)(new_value))))
#define AGGCALC_GLOBAL_PADD_DOUBLE(errcode,accum_isnull,accum_value,\
								   new_isnull,new_value)			\
	AGGCALC_GLOBAL_TEMPLATE_DOUBLE(									\
		errcode,accum_isnull,accum_value,							\
		new_isnull,__longlong_as_double(new_value),					\
		CHECK_OVERFLOW_FLOAT,										\
		atomic_add_double((cl_double *)(accum_value),				\
						  __longlong_as_double(new_value)))
#define AGGCALC_GLOBAL_PADD_NUMERIC(errcode,accum_isnull,accum_value, \
									new_isnull,new_value)			\
	AGGCALC_GLOBAL_TEMPLATE(										\
		ulong,errcode,accum_isnull,accum_value,						\
		new_isnull,(cl_ulong)(new_value),							\
		CHECK_OVERFLOW_NUMERIC,										\
		atomic_add_numeric((errcode),								\
						   (cl_ulong *)(accum_value),				\
						   (cl_ulong)(new_value)))

/*
 * Helper macros for gpupreagg_nogroup_calc
 */
#define MAX(x,y)			(((x)<(y)) ? (y) : (x))
#define MIN(x,y)			(((x)<(y)) ? (x) : (y))
#define ADD(x,y)			((x) + (y))

#define AGGCALC_NOGROUP_TEMPLATE(TYPE,errcode,						\
								 accum_isnull,accum_val,			\
								 newval_isnull,newval_val,			\
								 OVERFLOW,FUNC_CALL)				\
	do {															\
		if (!(newval_isnull))										\
		{															\
			TYPE tmp = FUNC_CALL;									\
			if (OVERFLOW((accum_val), (newval_val)))				\
			{														\
				STROM_SET_ERROR(errcode, StromError_CpuReCheck);	\
			}														\
			(accum_val)    = tmp;									\
			(accum_isnull) = false;									\
		}															\
	} while (0)

#define AGGCALC_NOGROUP_TEMPLATE_SHORT(errcode,accum,newval,		\
									   OVERFLOW,FUNC_CALL)			\
	AGGCALC_NOGROUP_TEMPLATE(cl_int,errcode,						\
							 (accum)->isnull,(accum)->int_val,		\
							 (newval)->isnull,(newval)->int_val,	\
							 OVERFLOW,FUNC_CALL)
#define AGGCALC_NOGROUP_TEMPLATE_INT(errcode,accum,newval,			\
									 OVERFLOW,FUNC_CALL)			\
	AGGCALC_NOGROUP_TEMPLATE(cl_int,errcode,						\
							 (accum)->isnull,(accum)->int_val,		\
							 (newval)->isnull,(newval)->int_val,	\
							 OVERFLOW,FUNC_CALL)
#define AGGCALC_NOGROUP_TEMPLATE_LONG(errcode,accum,newval,			\
									  OVERFLOW,FUNC_CALL)			\
	AGGCALC_NOGROUP_TEMPLATE(cl_long,errcode,						\
							 (accum)->isnull,(accum)->long_val,		\
							 (newval)->isnull,(newval)->long_val,	\
							 OVERFLOW,FUNC_CALL)
#define AGGCALC_NOGROUP_TEMPLATE_FLOAT(errcode,accum,newval,		\
									   OVERFLOW,FUNC_CALL)			\
	AGGCALC_NOGROUP_TEMPLATE(cl_float,errcode,						\
							 (accum)->isnull,(accum)->float_val,	\
							 (newval)->isnull,(newval)->float_val,	\
							 OVERFLOW,FUNC_CALL)
#define AGGCALC_NOGROUP_TEMPLATE_DOUBLE(errcode,accum,newval,		\
										OVERFLOW,FUNC_CALL)			\
	AGGCALC_NOGROUP_TEMPLATE(cl_double,errcode,						\
							 (accum)->isnull,(accum)->double_val,	\
							 (newval)->isnull,(newval)->double_val,	\
							 OVERFLOW,FUNC_CALL)

#ifdef PG_NUMERIC_TYPE_DEFINED
#define AGGCALC_NOGROUP_TEMPLATE_NUMERIC(errcode,accum_val,new_val,	\
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
			z = FUNC_CALL((errcode), x, y);							\
			if (z.isnull)											\
				STROM_SET_ERROR(errcode, StromError_CpuReCheck);	\
			(accum_val)->ulong_val = z.value;						\
			(accum_val)->isnull = z.isnull;							\
		}															\
	} while (0)
#endif

/* calculation for no group partial max */
#define AGGCALC_NOGROUP_PMAX_SHORT(errcode,accum,newval)	\
	AGGCALC_NOGROUP_TEMPLATE_SHORT(errcode,accum,newval,	\
								   CHECK_OVERFLOW_NONE,		\
								   MAX((accum)->int_val,	\
									   (newval)->int_val))
#define AGGCALC_NOGROUP_PMAX_INT(errcode,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_INT(errcode,accum,newval,		\
								 CHECK_OVERFLOW_NONE,		\
								 MAX((accum)->int_val,		\
									 (newval)->int_val))
#define AGGCALC_NOGROUP_PMAX_LONG(errcode,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_LONG(errcode,accum,newval,		\
								  CHECK_OVERFLOW_NONE,		\
								  MAX((accum)->long_val,	\
									  (newval)->long_val))
#define AGGCALC_NOGROUP_PMAX_FLOAT(errcode,accum,newval)	\
	AGGCALC_NOGROUP_TEMPLATE_FLOAT(errcode,accum,newval,	\
								   CHECK_OVERFLOW_NONE,		\
								   MAX((accum)->float_val,	\
									   (newval)->float_val))
#define AGGCALC_NOGROUP_PMAX_DOUBLE(errcode,accum,newval)	\
	AGGCALC_NOGROUP_TEMPLATE_DOUBLE(errcode,accum,newval,	\
									CHECK_OVERFLOW_NONE,	\
									MAX((accum)->double_val,\
										(newval)->double_val))
#define AGGCALC_NOGROUP_PMAX_NUMERIC(errcode,accum,newval)	\
	AGGCALC_NOGROUP_TEMPLATE_NUMERIC(errcode,accum,newval,	\
									 pgfn_numeric_max)

/* calculation for no group partial min */
#define AGGCALC_NOGROUP_PMIN_SHORT(errcode,accum,newval)	\
	AGGCALC_NOGROUP_TEMPLATE_SHORT(errcode,accum,newval,	\
								   CHECK_OVERFLOW_NONE,		\
								   MIN((accum)->int_val,	\
									   (newval)->int_val))
#define AGGCALC_NOGROUP_PMIN_INT(errcode,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_INT(errcode,accum,newval,		\
								 CHECK_OVERFLOW_NONE,		\
								 MIN((accum)->int_val,		\
									 (newval)->int_val))
#define AGGCALC_NOGROUP_PMIN_LONG(errcode,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_LONG(errcode,accum,newval,		\
								  CHECK_OVERFLOW_NONE,		\
								  MIN((accum)->long_val,	\
									  (newval)->long_val))
#define AGGCALC_NOGROUP_PMIN_FLOAT(errcode,accum,newval)	\
	AGGCALC_NOGROUP_TEMPLATE_FLOAT(errcode,accum,newval,	\
								   CHECK_OVERFLOW_NONE,		\
								   MIN((accum)->float_val,	\
									   (newval)->float_val))
#define AGGCALC_NOGROUP_PMIN_DOUBLE(errcode,accum,newval)	\
	AGGCALC_NOGROUP_TEMPLATE_DOUBLE(errcode,accum,newval,	\
									CHECK_OVERFLOW_NONE,	\
									MIN((accum)->double_val,\
										(newval)->double_val))
#define AGGCALC_NOGROUP_PMIN_NUMERIC(errcode,accum,newval)	\
	AGGCALC_NOGROUP_TEMPLATE_NUMERIC(errcode,accum,newval,	\
									 pgfn_numeric_min)

/* calculation for no group partial add */
#define AGGCALC_NOGROUP_PADD_SHORT(errcode,accum,newval)	\
	AGGCALC_NOGROUP_TEMPLATE_SHORT(errcode,accum,newval,	\
								   CHECK_OVERFLOW_SHORT,	\
								   ADD((accum)->int_val,	\
									   (newval)->int_val))
#define AGGCALC_NOGROUP_PADD_INT(errcode,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_INT(errcode,accum,newval,		\
								 CHECK_OVERFLOW_INT,		\
								 ADD((accum)->int_val,		\
									 (newval)->int_val))
#define AGGCALC_NOGROUP_PADD_LONG(errcode,accum,newval)		\
	AGGCALC_NOGROUP_TEMPLATE_LONG(errcode,accum,newval,		\
								  CHECK_OVERFLOW_INT,		\
								  ADD((accum)->long_val,	\
									  (newval)->long_val))
#define AGGCALC_NOGROUP_PADD_FLOAT(errcode,accum,newval)	\
    AGGCALC_NOGROUP_TEMPLATE_FLOAT(errcode,accum,newval,	\
								   CHECK_OVERFLOW_FLOAT,	\
								   ADD((accum)->float_val,	\
									   (newval)->float_val))
#define AGGCALC_NOGROUP_PADD_DOUBLE(errcode,accum,newval)	\
	AGGCALC_NOGROUP_TEMPLATE_DOUBLE(errcode,accum,newval,	\
									CHECK_OVERFLOW_FLOAT,	\
									ADD((accum)->double_val,\
										(newval)->double_val))
#define AGGCALC_NOGROUP_PADD_NUMERIC(errcode,accum,newval)	\
	AGGCALC_NOGROUP_TEMPLATE_NUMERIC(errcode,accum,newval,	\
									 pgfn_numeric_add)
#endif	/* __CUDACC__ */
#endif	/* CUDA_GPUPREAGG_H */
