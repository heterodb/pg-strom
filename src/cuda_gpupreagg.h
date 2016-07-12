/*
 * cuda_gpupreagg.h
 *
 * Preprocess of aggregate using GPU acceleration, to reduce number of
 * rows to be processed by CPU; including the Sort reduction.
 * --
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
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
 * Control data structure for GpuPreAgg kernel functions
 */
#define GPUPREAGG_NOGROUP_REDUCTION		1
#define GPUPREAGG_LOCAL_REDUCTION		2
#define GPUPREAGG_GLOBAL_REDUCTION		3
#define GPUPREAGG_FINAL_REDUCTION		4
#define GPUPREAGG_ONLY_TERMINATION		99	/* used to urgent termination */

/*
 * +--------------------+
 * | kern_gpupreagg     |
 * |     :              |
 * | kresults_1_offset -------+
 * | kresults_2_offset -----+ |
 * | +------------------+   | |
 * | | kern_parambuf    |   | |
 * | |    :             |   | |
 * +-+------------------+ <-|-+
 * | kern_resultbuf(1st)|   |
 * |      :             |   |
 * |      :             |   |
 * +--------------------+ <-+
 * | kern_resultbuf(2nd)|
 * |      :             |
 * |      :             |
 * +--------------------+
 */
typedef struct
{
	kern_errorbuf	kerror;				/* kernel error information */
	cl_uint			reduction_mode;		/* one of GPUPREAGG_* above */
	/* -- runtime statistics -- */
	cl_uint			num_conflicts;		/* only used in kernel space */
	cl_uint			num_groups;			/* out: # of new groups */
	cl_uint			varlena_usage;		/* out: size of varlena usage */
	cl_uint			ghash_conflicts;	/* out: # of ghash conflicts */
	cl_uint			fhash_conflicts;	/* out: # of fhash conflicts */
	/* -- performance monitor -- */
	struct {
		cl_uint		num_kern_prep;		/* # of kern_preparation calls */
		cl_uint		num_kern_nogrp;		/* # of kern_nogroup calls */
		cl_uint		num_kern_lagg;		/* # of kern_local_reduction calls */
		cl_uint		num_kern_gagg;		/* # of kern_global_reducation calls */
		cl_uint		num_kern_fagg;		/* # of kern_final_reduction calls */
		cl_float	tv_kern_prep;		/* msec of kern_preparation */
		cl_float	tv_kern_nogrp;		/* msec of kern_nogroup */
		cl_float	tv_kern_lagg;		/* msec of kern_local_reduction */
		cl_float	tv_kern_gagg;		/* msec of kern_global_reducation */
		cl_float	tv_kern_fagg;		/* msec of kern_final_reduction */
	} pfm;
	/* -- other hashing parameters -- */
	cl_uint			key_dist_salt;			/* hashkey distribution salt */
	cl_uint			hash_size;				/* size of global hash-slots */
	cl_uint			pg_crc32_table[256];	/* master CRC32 table */
	cl_uint			kresults_1_offset;		/* offset to 1st kresults */
	cl_uint			kresults_2_offset;		/* offset to 2nd kresults */
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
#define KERN_GPUPREAGG_1ST_RESULTBUF(kgpreagg)			\
	((kern_resultbuf *)									\
	 ((char *)(kgpreagg) + (kgpreagg)->kresults_1_offset))
#define KERN_GPUPREAGG_2ND_RESULTBUF(kgpreagg)			\
	((kern_resultbuf *)									\
	 ((char *)(kgpreagg) + (kgpreagg)->kresults_2_offset))

#define KERN_GPUPREAGG_LENGTH(kgpreagg)					\
	((uintptr_t)(kgpreagg)->kresults_1_offset +			\
	 2 * ((uintptr_t)(kgpreagg)->kresults_2_offset -	\
		  (uintptr_t)(kgpreagg)->kresults_1_offset))
#define KERN_GPUPREAGG_DMASEND_OFFSET(kgpreagg)		0
#define KERN_GPUPREAGG_DMASEND_LENGTH(kgpreagg)			\
	(offsetof(kern_gpupreagg, kparams) +				\
	 KERN_GPUPREAGG_PARAMBUF_LENGTH(kgpreagg))
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
					 kern_data_store *kds_src,	/* in */
					 kern_tupitem *tupitem,		/* in */
					 kern_data_store *kds_dst,	/* out */
					 Datum *dst_values,			/* out */
					 cl_char *dst_isnull);		/* out */

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

	assert(kds->format == KDS_FORMAT_SLOT);
	assert(colidx < kds->ncols);

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
	else if (cmeta.attlen == sizeof(cl_long) ||	/* also, cl_double */
			 cmeta.atttypid == PG_NUMERICOID)	/* internal of numeric */
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

	assert(kds->format == KDS_FORMAT_SLOT);
	assert(colidx < kds->ncols);

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
	else if (cmeta.attlen == sizeof(cl_long) ||	/* also, cl_double */
			 cmeta.atttypid == PG_NUMERICOID)	/* internal of numeric */
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
						  kern_data_store *kds_src, cl_uint rowidx_src,
						  kern_data_store *kds_dst, cl_uint rowidx_dst)
{
	Datum	   *src_values = KERN_DATA_STORE_VALUES(kds_src, rowidx_src);
	Datum	   *dst_values = KERN_DATA_STORE_VALUES(kds_dst, rowidx_dst);
	cl_char	   *src_isnull = KERN_DATA_STORE_ISNULL(kds_src, rowidx_src);
	cl_char	   *dst_isnull = KERN_DATA_STORE_ISNULL(kds_dst, rowidx_dst);
	cl_uint		i, ncols = kds_src->ncols;
	cl_uint		alloc_size = 0;
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
		if (cmeta.atttypid == PG_NUMERICOID)
			continue;	/* special internal data format */
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
		cl_uint		usage_prev = atomicAdd(&kds_dst->usage, alloc_size);

		if (KERN_DATA_STORE_SLOT_LENGTH(kds_dst, kds_dst->nrooms) +
			usage_prev + alloc_size >= kds_dst->length)
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
			return 0;
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
		else if (cmeta.atttypid == PG_NUMERICOID ||
				 cmeta.attbyval)
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
	kern_tupitem   *tupitem;
	cl_uint			offset;
	cl_uint			count;
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
	if (kds_index < kds_in->nitems &&
		gpupreagg_qual_eval(&kcxt, kds_in, kds_index))
		tupitem = KERN_DATA_STORE_TUPITEM(kds_in, kds_index);
	else
		tupitem = NULL;		/* not a visible tuple */

	/* calculation of total number of rows to be processed in this work-
	 * group.
	 */
	offset = pgstromStairlikeSum(tupitem != NULL ? 1 : 0, &count);

	/* Allocation of the result slot on the kds_src. */
	if (get_local_id() == 0)
	{
		if (count > 0)
			base = atomicAdd(&kds_src->nitems, count);
		else
			base = 0;
	}
	__syncthreads();

	/* GpuPreAgg should never increase number of items */
	assert(base + count <= kds_src->nrooms);

	/* do projection */
	if (tupitem != NULL)
	{
		cl_uint		dst_index = base + offset;
		Datum	   *dst_values = KERN_DATA_STORE_VALUES(kds_src, dst_index);
		cl_char	   *dst_isnull = KERN_DATA_STORE_ISNULL(kds_src, dst_index);

		gpupreagg_projection(&kcxt,
							 kds_in,		/* input kds */
							 tupitem,		/* input tuple */
							 kds_src,		/* destination kds */
							 dst_values,	/* destination values[] array */
							 dst_isnull);	/* destination isnull[] array */
	}
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
							kern_data_store *kds_slot,
							kern_resultbuf *kresults_src,
							kern_resultbuf *kresults_dst)
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_context	kcxt;
	varlena		   *kparam_0 = (varlena *)kparam_get_value(kparams, 0);
	cl_char		   *gpagg_atts = (cl_char *)VARDATA(kparam_0);
	pagg_datum	   *l_datum = SHARED_WORKMEM(pagg_datum);
	cl_uint			i, ncols = kds_slot->ncols;
	cl_uint			nvalids = 0;
	cl_uint			kds_index = UINT_MAX;

	INIT_KERNEL_CONTEXT(&kcxt, gpupreagg_nogroup_reduction, kparams);
	if (kresults_src->all_visible)
	{
		/* scope of this block */
		if (get_global_base() < kds_slot->nitems)
			nvalids = min(kds_slot->nitems - get_global_base(),
						  get_local_size());
		else
			goto out;	/* should not happen */

		/* global index on the kds_slot */
		if (get_local_id() < nvalids)
			kds_index = get_global_base() + get_local_id();
		__syncthreads();
	}
	else
	{
		/* scope of this block */
		if (get_global_base() < kresults_src->nitems)
			nvalids = min(kresults_src->nitems - get_global_base(),
						  get_local_size());
		else
			goto out;	/* should not happen */

		/* global index on the kds_slot */

		/* MEMO: On the CUDA 7.5 NVRTC, we met a strange behavior.
		 * If and when a part of threads in warp didn't fit nvalids,
		 * it is ought to ignore the following if-then block unless
		 * else-block. At that time, our test workloads has nvalids=196,
		 * thus 4 threads match the if-condition below.
		 * According to the observation, these 4 threads didn't resume
		 * until exit of any other threads (thread 0-191 and 196-223!).
		 * Thus, final result lacks the values to be accumulated by the
		 * 4 threads. This strange behavior was eliminated if we added
		 * an else-block. It seems to me a bug of NVRTC 7.5?
		 *
		 * MEMO: additional investigation - This problem happens on
		 * GTX980 (GM104) but didn't happen in Tesla K20c (GK110).
		 * Here is no difference in the PTX image, expect for .target
		 * directive. One strange is, the reduction loop below contains
		 * three __syncthreads(), but I could find only two bar.sync
		 * operation in this function. So, I doubt the threads out of
		 * range unexpectedly reached end of the function, then it
		 * made H/W synchronization mechanism confused.
		 *
		 * As a workaround, I could find an extra '__syncthreads()'
		 * just after that if-block below can hide the problem.
		 * However, it shouldn't be necessary
		 *
		 * This strange behavior was reported to NVIDIA at 22-03-2016,
		 * as a bug report bug#160322-000527. The PG-Strom development
		 * team is now waiting for their response.
		 */
		if (get_local_id() < nvalids)
			kds_index = kresults_src->results[get_global_id()];
		__syncthreads();
	}

	/* loop for each columns */
	for (i=0; i < ncols; i++)
	{
		size_t		dist;

		/* if not GPUPREAGG_FIELD_IS_AGGFUNC, do nothing */
		if (gpagg_atts[i] != GPUPREAGG_FIELD_IS_AGGFUNC)
			continue;

		/* load this value from kds_slot onto pagg_datum */
		if (get_local_id() < nvalids)
		{
			gpupreagg_data_load(l_datum + get_local_id(),
								&kcxt, kds_slot, i, kds_index);
		}

		/* do reduction */
		for (dist = 2; dist < 2 * nvalids; dist *= 2)
		{
			if ((get_local_id() % dist) == 0 &&
				(get_local_id() + dist / 2) < nvalids)
			{
				gpupreagg_nogroup_calc(&kcxt,
									   i,
									   l_datum + get_local_id(),
									   l_datum + get_local_id() + dist / 2);
			}
			__syncthreads();
		}

		/* store this value to kds_dst from datum */
		if (get_local_id() == 0)
		{
			gpupreagg_data_store(l_datum + get_local_id(),
								 &kcxt, kds_slot, i, kds_index);
		}
		__syncthreads();
	}

	/* update kresults_dst */
	if (get_local_id() == 0)
	{
		cl_uint		dest_index
			= atomicAdd(&kresults_dst->nitems, 1);
		assert(dest_index < kresults_dst->nrooms);
		kresults_dst->results[dest_index] = kds_index;
	}
out:
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kgpreagg->kerror, kcxt.e);
}

/*
 * gpupreagg_local_reduction
 */
KERNEL_FUNCTION_MAXTHREADS(void)
gpupreagg_local_reduction(kern_gpupreagg *kgpreagg,
						  kern_data_store *kds_slot,
						  kern_resultbuf *kresults)
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_context	kcxt;
	varlena		   *kparam_0 = (varlena *)kparam_get_value(kparams, 0);
	cl_char		   *gpagg_atts = (cl_char *) VARDATA(kparam_0);
	size_t			hash_size = 2 * get_local_size();
	cl_uint			owner_index;
	cl_uint			key_dist_salt = kgpreagg->key_dist_salt;
	cl_uint			key_dist_index = 0;
	cl_uint			hash_value;
	cl_uint			hash_value_base;
	cl_uint			nitems = kds_slot->nitems;
	cl_uint			i, ncols = kds_slot->ncols;
	cl_uint			count;
	cl_uint			index;
	pg_int4_t		key_dist_factor;
	pagg_hashslot	old_slot;
	pagg_hashslot	new_slot;
	pagg_hashslot	cur_slot;
	pagg_datum	   *l_datum;
	pagg_hashslot  *l_hashslot;
	__shared__ cl_uint	crc32_table[256];
	__shared__ size_t	base;

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
										 kds_slot,
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
								   kds_slot, get_global_id(),
								   kds_slot, buddy_index))
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
	index = pgstromStairlikeSum(get_local_id() == owner_index ? 1 : 0,
								&count);
	if (get_local_id() == 0)
		base = atomicAdd(&kresults->nitems, count);
	__syncthreads();
	if (get_local_id() == owner_index)
		kresults->results[base + index] = get_global_id();
	/* Number of items should not be larger than nrooms  */
	assert(base + count <= kresults->nrooms);

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
	for (i=0; i < ncols; i++)
	{
		/*
		 * In case when this column is either a grouping-key or not-
		 * referenced one (thus, not a partial aggregation), all we
		 * need to do is copying the data from the source to the
		 * destination; without modification anything.
		 */
		if (gpagg_atts[i] != GPUPREAGG_FIELD_IS_AGGFUNC)
			continue;

		/* Load aggregation item to pagg_datum */
		if (get_global_id() < nitems)
		{
			gpupreagg_data_load(l_datum + get_local_id(),
								&kcxt,
								kds_slot,
								i,
								get_global_id());
		}
		__syncthreads();

		/* Reduction, using local atomic operation */
		if (get_global_id() < nitems &&
			get_local_id() != owner_index)
		{
			gpupreagg_local_calc(&kcxt,
								 i,
								 l_datum + owner_index,
								 l_datum + get_local_id());
		}
		__syncthreads();

		/* Move the value that is aggregated */
		if (owner_index == get_local_id())
		{
			assert(get_global_id() < nitems);
			gpupreagg_data_store(l_datum + owner_index,
								 &kcxt,
								 kds_slot,
								 i,
								 get_global_id());
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
						   kern_data_store *kds_slot,
						   kern_resultbuf *kresults_src,
						   kern_resultbuf *kresults_dst,
						   kern_global_hashslot *g_hash)
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_context	kcxt;
	varlena		   *kparam_0 = (varlena *) kparam_get_value(kparams, 0);
	cl_char		   *gpagg_atts = (cl_char *) VARDATA(kparam_0);
	size_t			g_hashsize = g_hash->hash_size;
	size_t			g_hashlimit = GLOBAL_HASHSLOT_THRESHOLD(g_hashsize);
	size_t			owner_index;
	size_t			kds_index;
	cl_bool			is_valid_slot = false;
	cl_uint			key_dist_salt = kgpreagg->key_dist_salt;
	cl_uint			key_dist_index = 0;
	cl_uint			hash_value;
	cl_uint			hash_value_base;
	cl_uint			index;
	cl_uint			i, ncols = kds_slot->ncols;
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

	if (kresults_src->all_visible)
	{
		if (get_global_id() < kds_slot->nitems)
		{
			kds_index = get_global_id();
			is_valid_slot = true;
		}
	}
	else
	{
		if (get_global_id() < kresults_src->nitems)
		{
			kds_index = kresults_src->results[get_global_id()];
			assert(kds_index < kds_slot->nitems);
			is_valid_slot = true;
		}
	}

	if (is_valid_slot)
	{
		/*
		 * Calculation of initial hash value
		 */
		INIT_LEGACY_CRC32(hash_value);
		hash_value = gpupreagg_hashvalue(&kcxt, crc32_table,
										 hash_value,
										 kds_slot,
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
		new_slot.s.hash = hash_value;
		new_slot.s.index = kds_index;
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
									kds_slot, kds_index,
									kds_slot, cur_slot.s.index))
		{
			assert(cur_slot.s.index < kds_slot->nitems);
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
	index = pgstromStairlikeSum(owner_index != (cl_uint)(0xffffffff) &&
								owner_index == kds_index ? 1 : 0,
								&count);
	if (count > 0)
	{
		if (get_local_id() == 0)
			base = atomicAdd(&kresults_dst->nitems, count);
		__syncthreads();
		assert(base + count <= kresults_dst->nrooms);
		if (kds_index == owner_index)
			kresults_dst->results[base + index] = kds_index;
	}
	__syncthreads();

	/*
	 * Quick bailout if thread is not valid, or no hash slot is available.
	 * check_global_hashslot_usage() already put an error code, for CPU
	 * fallback logic, so we can exit anyway.
	 */
	if (owner_index == (cl_uint)(0xffffffff))
		goto out;

	/*
	 * Global reduction for each column
	 *
	 * Any non-owner thread shall accumulate its own value onto the value
	 * owned by grouping-key owner. Once atomic operations got done, the
	 * accumulated value means the partial aggregation.
	 * In case when thread points invalid item or thread could not find
	 * a hash slot (decision by check_global_hashslot_usage), this thread
	 * will skip the reduction phase.
	 * Even in the later case, check_global_hashslot_usage already set an
	 * error code, so we don't care about here.
	 */
	if (owner_index != 0xffffffffU &&		/* a valid thread? */
		owner_index != kds_index)			/* not a owner thread? */
	{
		for (i=0; i < ncols; i++)
		{
			if (gpagg_atts[i] != GPUPREAGG_FIELD_IS_AGGFUNC)
				continue;
			assert(owner_index < kds_slot->nrooms);
			gpupreagg_global_calc(&kcxt,
								  i,
								  kds_slot, owner_index,
								  kds_slot, kds_index);
		}
	}
out:
	/* collect run-time statistics */
	pgstromStairlikeSum(nconflicts, &count);
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
 */
KERNEL_FUNCTION(void)
gpupreagg_final_reduction(kern_gpupreagg *kgpreagg,		/* in */
						  kern_data_store *kds_slot,	/* in */
						  kern_data_store *kds_final,	/* out */
						  kern_resultbuf *kresults_src,	/* in */
						  kern_resultbuf *kresults_dst,	/* out, locked only */
						  kern_global_hashslot *f_hash)	/* only internal */
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_context	kcxt;
	varlena		   *kparam_0 = (varlena *) kparam_get_value(kparams, 0);
	cl_char		   *gpagg_atts = (cl_char *) VARDATA(kparam_0);
	cl_uint			kds_index;
	cl_uint			owner_index = (cl_uint)(0xffffffff); //INVALID
	size_t			f_hashsize = f_hash->hash_size;
	size_t			f_hashlimit = GLOBAL_HASHSLOT_THRESHOLD(f_hashsize);
	cl_uint			key_dist_salt = kgpreagg->key_dist_salt;
	cl_uint			key_dist_index = 0;
	cl_uint			hash_value;
	cl_uint			hash_value_base;
	cl_uint			i, ncols = kds_slot->ncols;
	cl_uint			index;
	cl_uint			count;
	cl_uint			nconflicts = 0;
	cl_uint			allocated = 0;
	cl_bool			isOwner = false;
	cl_bool			meet_locked = false;
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

	/* row-index on the kds_slot buffer */
	if (kresults_src->all_visible)
	{
		if (get_global_id() < kds_slot->nitems)
			kds_index = get_global_id();
		else
			goto out;
	}
	else
	{
		if (get_global_id() < min(kresults_src->nitems,
								  kresults_src->nrooms))
			kds_index = kresults_src->results[get_global_id()];
		else
			goto out;
		assert(kds_index < kds_slot->nitems);
	}

	/*
	 * Calculation of initial hash value
	 */
	INIT_LEGACY_CRC32(hash_value);
	hash_value = gpupreagg_hashvalue(&kcxt, crc32_table,
									 hash_value,
									 kds_slot,
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
			allocated += gpupreagg_final_data_move(&kcxt,
												   kds_slot,
												   kds_index,
												   kds_final,
												   new_slot.s.index);
		}
		else
		{
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		}
		__threadfence();
		f_hash->hash_slot[index].s.index = new_slot.s.index;
		/* this thread is the owner of this slot */
		owner_index = new_slot.s.index;
		cur_slot.value = new_slot.value;
		isOwner = true;
	}
	else
	{
		/* it may be a slot we can use later */
		if (cur_slot.s.index == (cl_uint)(0xfffffffe) &&
			cur_slot.s.hash == hash_value)
			meet_locked = true;

		if (cur_slot.s.index != (cl_uint)(0xfffffffe) &&
			cur_slot.s.hash  == hash_value &&
			gpupreagg_keymatch(&kcxt,
							   kds_slot, kds_index,
							   kds_final, cur_slot.s.index))
		{
			/* grouping key matched */
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
			/*
			 * If we already meet a hash entry that was locked but has
			 * same hash value, we try to walk on the next kernel
			 * invocation, than enlarge hash slot.
			 */
			if (meet_locked)
			{
				owner_index = (cl_uint)(0xfffffffe); // LOCKED
				goto out;
			}

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
		if (owner_index < Min(kds_final->nitems,
							  kds_final->nrooms))
		{
			for (i=0; i < ncols; i++)
			{
				if (gpagg_atts[i] != GPUPREAGG_FIELD_IS_AGGFUNC)
					continue;
			
				/*
				 * Reduction, using global atomic operation
				 *
				 * If thread is responsible to the grouping-key, other
				 * threads but NOT responsible will accumlate their values
				 * here, then it shall become aggregated result. So, we mark
				 * the "responsible" thread identifier on the kern_row_map.
				 * Once kernel execution gets done, this index points the
				 * location of aggregate value.
				 */
				gpupreagg_global_calc(&kcxt,
									  i,
									  kds_final, owner_index,
									  kds_slot, kds_index);
			}
		}
	}
out:
	/* Do we try to update kds_final again on the next kernel call? */
	__syncthreads();
	index = pgstromStairlikeSum(owner_index == 0xfffffffeU ? 1 : 0,
								&count);
	if (count > 0)
	{
		__shared__ cl_uint base;

		if (get_local_id() == 0)
			base = atomicAdd(&kresults_dst->nitems, count);
		__syncthreads();
		if (owner_index == 0xfffffffeU)
			kresults_dst->results[base + index] = kds_index;
	}

	/* update run-time statistics */
	pgstromStairlikeSum(isOwner ? 1 : 0, &count);
	if (count > 0 && get_local_id() == 0)
		atomicAdd(&kgpreagg->num_groups, count);
	__syncthreads();

	pgstromStairlikeSum(allocated, &count);
	if (count > 0 && get_local_id() == 0)
		atomicAdd(&kgpreagg->varlena_usage, count);
	__syncthreads();

	pgstromStairlikeSum(nconflicts, &count);
	if (count > 0 && get_local_id() == 0)
		atomicAdd(&kgpreagg->fhash_conflicts, count);
	__syncthreads();

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
gpupreagg_fixup_varlena(kern_gpupreagg *kgpreagg,
						kern_data_store *kds_final)
{
	kern_parambuf  *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_context	kcxt;
	varlena		   *kparam_0 = (varlena *) kparam_get_value(kparams, 0);
	cl_uint			i, ncols = kds_final->ncols;
	kern_colmeta	cmeta;
	size_t			kds_index = get_global_id();
	Datum		   *dst_values = KERN_DATA_STORE_VALUES(kds_final, kds_index);
	cl_bool		   *dst_isnull = KERN_DATA_STORE_ISNULL(kds_final, kds_index);
	cl_char		   *numeric_ptr = NULL;
	cl_uint			numeric_len = 0;

	/* Sanity checks */
	assert(kds_final->format == KDS_FORMAT_SLOT);
	assert(kds_final->has_notbyval);

	INIT_KERNEL_CONTEXT(&kcxt,gpupreagg_fixup_varlena,kparams);

	/*
	 * Expand extra field to fixup numeric data type; varlena or indirect
	 * data types are already copied to the extra field, so all we have to
	 * fixup here is numeric data type.
	 */
	if (kds_final->has_numeric)
	{
		cl_uint		offset;
		cl_uint		count;
		__shared__ cl_uint base;

		if (kds_index < kds_final->nitems)
		{
			for (i=0; i < ncols; i++)
			{
				cmeta = kds_final->colmeta[i];

				if (cmeta.atttypid != PG_NUMERICOID)
					continue;
				if (dst_isnull[i])
					continue;
				numeric_len += MAXALIGN(pg_numeric_to_varlena(&kcxt, NULL,
															  dst_values[i],
															  dst_isnull[i]));
			}
		}
		/* allocation of the extra buffer on demand */
		offset = pgstromStairlikeSum(numeric_len, &count);
		if (get_local_id() == 0)
		{
			if (count > 0)
				base = atomicAdd(&kds_final->usage, count);
			else
				base = 0;
		}
		__syncthreads();

		/*
		 * At this point, number of items will be never increased any more,
		 * so extra area is limited by nitems, not nrooms. It is actually
		 * 'extra' area than final_reduction phase. :-)
		 */
		if (KERN_DATA_STORE_SLOT_LENGTH(kds_final, kds_final->nitems) +
			base + count >= kds_final->length)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			goto out;
		}
		numeric_ptr = ((char *)kds_final + kds_final->length -
					   (base + offset + numeric_len));
	}

	if (kds_index < kds_final->nitems)
	{
		for (i=0; i < ncols; i++)
		{
			/* No need to fixup NULL value anyway */
			if (dst_isnull[i])
				continue;

			cmeta = kds_final->colmeta[i];
			if (cmeta.atttypid == PG_NUMERICOID)
			{
				assert(numeric_ptr != NULL);
				numeric_len = pg_numeric_to_varlena(&kcxt, numeric_ptr,
													dst_values[i],
													dst_isnull[i]);
				dst_values[i] = devptr_to_host(kds_final, numeric_ptr);
				numeric_ptr += MAXALIGN(numeric_len);
			}
			else if (!cmeta.attbyval)
			{
				/* validation of the device pointer */
				assert(dst_values[i] >= ((size_t)kds_final +
										 kds_final->length -
										 kds_final->usage) &&
					   dst_values[i] <  ((size_t)kds_final +
										 kds_final->length));
				dst_values[i] = devptr_to_host(kds_final, dst_values[i]);
			}
		}
	}
out:
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kgpreagg->kerror, kcxt.e);
}

/*
 * gpupreagg_main
 *
 * The controller kernel function that launches a 
 *
 *
 *
 *
 *
 */
KERNEL_FUNCTION(void)
gpupreagg_main(kern_gpupreagg *kgpreagg,
			   kern_data_store *kds_row,		/* KDS_FORMAT_ROW */
			   kern_data_store *kds_slot,		/* KDS_FORMAT_SLOT */
			   kern_global_hashslot *g_hash,	/* For global reduction */
			   kern_data_store *kds_final,		/* KDS_FORMAT_SLOT + Extra */
			   kern_global_hashslot *f_hash)
{
	kern_parambuf	   *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	kern_resultbuf	   *kresults_src = KERN_GPUPREAGG_1ST_RESULTBUF(kgpreagg);
	kern_resultbuf	   *kresults_dst = KERN_GPUPREAGG_2ND_RESULTBUF(kgpreagg);
	kern_resultbuf	   *kresults_tmp;
	cl_uint				kresults_nrooms = kds_row->nitems;
	kern_context		kcxt;
	void			  **kern_args;
	dim3				grid_sz;
	dim3				block_sz;
	cl_ulong			tv_start;
	cudaError_t			status = cudaSuccess;

	/* Init kernel context */
	INIT_KERNEL_CONTEXT(&kcxt, gpupreagg_main, kparams);
	assert(get_global_size() == 1);	/* !!single thread!! */
	assert(kgpreagg->reduction_mode != GPUPREAGG_ONLY_TERMINATION);

	/* Launch:
	 * KERNEL_FUNCTION(void)
	 * gpupreagg_preparation(kern_gpupreagg *kgpreagg,
	 *                       kern_data_store *kds_row,
	 *                       kern_data_store *kds_slot,
	 *                       kern_global_hashslot *g_hash)
	 */
	tv_start = GlobalTimer();
	kern_args = (void **)cudaGetParameterBuffer(sizeof(void *),
												sizeof(void *) * 4);
	if (!kern_args)
	{
		STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
		goto out;
	}
	kern_args[0] = kgpreagg;
	kern_args[1] = kds_row;
	kern_args[2] = kds_slot;
	kern_args[3] = g_hash;

	status = optimal_workgroup_size(&grid_sz,
									&block_sz,
									(const void *)
									gpupreagg_preparation,
									kds_row->nitems,
									0, sizeof(kern_errorbuf));
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
		goto out;
	}

	status = cudaLaunchDevice((void *)gpupreagg_preparation,
							  kern_args, grid_sz, block_sz,
							  sizeof(kern_errorbuf) * block_sz.x,
							  NULL);
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
		goto out;
	}

	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
		goto out;
	}
	else if (kgpreagg->kerror.errcode != StromError_Success)
		return;

	TIMEVAL_RECORD(kgpreagg,kern_prep,tv_start);

	if (kgpreagg->reduction_mode == GPUPREAGG_NOGROUP_REDUCTION)
	{
		/* Launch:
		 * KERNEL_FUNCTION(void)
		 * gpupreagg_nogroup_reduction(kern_gpupreagg *kgpreagg,
		 *                             kern_data_store *kds_slot,
		 *                             kern_resultbuf *kresults_src,
		 *                             kern_resultbuf *kresults_dst)
		 */
		tv_start = GlobalTimer();

		/* setup kern_resultbuf */
		memset(kresults_src, 0, offsetof(kern_resultbuf, results[0]));
		kresults_src->nrels = 1;
		kresults_src->nrooms = kresults_nrooms;
		kresults_src->all_visible = true;

		memset(kresults_dst, 0, offsetof(kern_resultbuf, results[0]));
		kresults_dst->nrels = 1;
		kresults_dst->nrooms = kresults_nrooms;

		/* 1st trial of the reduction */
		kern_args = (void **)cudaGetParameterBuffer(sizeof(void *),
													sizeof(void *) * 4);
		if (!kern_args)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
			goto out;
		}
		kern_args[0] = kgpreagg;
		kern_args[1] = kds_slot;
		kern_args[2] = kresults_src;
		kern_args[3] = kresults_dst;
		status = largest_workgroup_size(&grid_sz,
										&block_sz,
										(const void *)
										gpupreagg_nogroup_reduction,
										kds_slot->nitems,
										0, Max(sizeof(pagg_datum),
											   sizeof(kern_errorbuf)));
		if (status != cudaSuccess)
		{
			STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
			goto out;
		}

		status = cudaLaunchDevice((void *)gpupreagg_nogroup_reduction,
								  kern_args, grid_sz, block_sz,
								  Max(sizeof(pagg_datum),
									  sizeof(kern_errorbuf)) * block_sz.x,
								  NULL);
        if (status != cudaSuccess)
        {
            STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
            goto out;
        }

		status = cudaDeviceSynchronize();
		if (status != cudaSuccess)
		{
			STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
			goto out;
		}
		else if (kgpreagg->kerror.errcode != StromError_Success)
			return;

		/* 2nd trial of the reduction */
		memset(kresults_src, 0, offsetof(kern_resultbuf, results[0]));
		kresults_src->nrels = 1;
		kresults_src->nrooms = kresults_nrooms;

		kern_args = (void **)cudaGetParameterBuffer(sizeof(void *),
													sizeof(void *) * 4);
		if (!kern_args)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
			goto out;
		}
		kern_args[0] = kgpreagg;
        kern_args[1] = kds_slot;
		kern_args[2] = kresults_dst;	/* reverse */
        kern_args[3] = kresults_src;	/* reverse */
		status = largest_workgroup_size(&grid_sz,
										&block_sz,
										(const void *)
										gpupreagg_nogroup_reduction,
										kresults_dst->nitems,
										0, Max(sizeof(pagg_datum),
											   sizeof(kern_errorbuf)));
		if (status != cudaSuccess)
		{
			STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
			goto out;
		}

		status = cudaLaunchDevice((void *)gpupreagg_nogroup_reduction,
								  kern_args, grid_sz, block_sz,
								  Max(sizeof(pagg_datum),
									  sizeof(kern_errorbuf)) * block_sz.x,
								  NULL);
		if (status != cudaSuccess)
		{
			STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
			goto out;
		}

		status = cudaDeviceSynchronize();
		if (status != cudaSuccess)
		{
			STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
			goto out;
		}
		else if (kgpreagg->kerror.errcode != StromError_Success)
			return;

		TIMEVAL_RECORD(kgpreagg,kern_nogrp,tv_start);
	}
	else if (kgpreagg->reduction_mode == GPUPREAGG_LOCAL_REDUCTION ||
			 kgpreagg->reduction_mode == GPUPREAGG_GLOBAL_REDUCTION)
	{
		memset(kresults_src, 0, offsetof(kern_resultbuf, results[0]));
		kresults_src->nrels = 1;
		kresults_src->nrooms = kresults_nrooms;
		memset(kresults_dst, 0, offsetof(kern_resultbuf, results[0]));
		kresults_dst->nrels = 1;
		kresults_dst->nrooms = kresults_nrooms;

		if (kgpreagg->reduction_mode == GPUPREAGG_LOCAL_REDUCTION)
		{
			size_t		dynamic_shmem_unitsz = Max3(sizeof(kern_errorbuf),
													sizeof(pagg_hashslot) * 2,
													sizeof(pagg_datum));
			/*
			 * Launch:
			 * KERNEL_FUNCTION_MAXTHREADS(void)
			 * gpupreagg_local_reduction(kern_gpupreagg *kgpreagg,
			 *                           kern_data_store *kds_slot,
			 *                           kern_resultbuf *kresults
			 */
			tv_start = GlobalTimer();
			kern_args = (void **)cudaGetParameterBuffer(sizeof(void *),
														sizeof(void *) * 3);
			if (!kern_args)
			{
				STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
				goto out;
			}
			kern_args[0] = kgpreagg;
			kern_args[1] = kds_slot;		/* in */
			kern_args[2] = kresults_src;	/* out */
			status = largest_workgroup_size(&grid_sz,
											&block_sz,
											(const void *)
											gpupreagg_local_reduction,
											kds_slot->nitems,
											0, dynamic_shmem_unitsz);
			if (status != cudaSuccess)
			{
				STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
				goto out;
			}

			status = cudaLaunchDevice((void *)gpupreagg_local_reduction,
									  kern_args,
									  grid_sz,
									  block_sz,
									  dynamic_shmem_unitsz * block_sz.x,
									  NULL);
			if (status != cudaSuccess)
			{
				STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
				goto out;
			}

			status = cudaDeviceSynchronize();
			if (status != cudaSuccess)
			{
				STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
				goto out;
			}
			else if (kgpreagg->kerror.errcode != StromError_Success)
				return;
			/* perfmon */
			TIMEVAL_RECORD(kgpreagg,kern_lagg,tv_start);
		}
		else
		{
			/* if no local reduction, global reduction takes all the rows */
			kresults_src->all_visible = true;
		}

		/*
		 * Launch:
		 * KERNEL_FUNCTION(void)
		 * gpupreagg_global_reduction(kern_gpupreagg *kgpreagg,
		 *                            kern_data_store *kds_slot,
		 *                            kern_resultbuf *kresults_src,
		 *                            kern_resultbuf *kresults_dst,
		 *                            kern_global_hashslot *g_hash)
		 */
		tv_start = GlobalTimer();
		kern_args = (void **)cudaGetParameterBuffer(sizeof(void *),
													sizeof(void *) * 5);
		if (!kern_args)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
			goto out;
		}
		kern_args[0] = kgpreagg;
		kern_args[1] = kds_slot;
		kern_args[2] = kresults_src;
		kern_args[3] = kresults_dst;
		kern_args[4] = g_hash;
		status = largest_workgroup_size(&grid_sz,
										&block_sz,
										(const void *)
										gpupreagg_global_reduction,
										kresults_src->all_visible
										? kds_slot->nitems
										: kresults_src->nitems,
										0, sizeof(kern_errorbuf));
		if (status != cudaSuccess)
		{
			STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
			goto out;
		}

		status = cudaLaunchDevice((void *)gpupreagg_global_reduction,
								  kern_args,
								  grid_sz,
								  block_sz,
								  sizeof(kern_errorbuf) * block_sz.x,
								  NULL);
		if (status != cudaSuccess)
		{
			STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
			goto out;
		}

		status = cudaDeviceSynchronize();
		if (status != cudaSuccess)
		{
			STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
			goto out;
		}
		else if (kgpreagg->kerror.errcode != StromError_Success)
			return;

		TIMEVAL_RECORD(kgpreagg,kern_gagg,tv_start);

		/* swap */
		kresults_tmp = kresults_src;
		kresults_src = kresults_dst;
		kresults_dst = kresults_tmp;
	}
	else
	{
		/* only final reduction - all input slot should be visible */
		memset(kresults_src, 0, offsetof(kern_resultbuf, results[0]));
		kresults_src->nrels = 1;
		kresults_src->nrooms = kresults_nrooms;
		kresults_src->all_visible = true;
	}

	/* Launch:
	 * KERNEL_FUNCTION(void)
	 * gpupreagg_final_reduction(kern_gpupreagg *kgpreagg,
	 *                           kern_data_store *kds_slot,
	 *                           kern_data_store *kds_final,
	 *                           kern_resultbuf *kresults_src,
	 *                           kern_resultbuf *kresults_dst,
	 *                           kern_global_hashslot *f_hash)
	 */
	tv_start = GlobalTimer();
final_retry:
	/* init destination kern_resultbuf */
	memset(kresults_dst, 0, offsetof(kern_resultbuf, results[0]));
	kresults_dst->nrels = 1;
	kresults_dst->nrooms = kresults_nrooms;

	kern_args = (void **)cudaGetParameterBuffer(sizeof(void *),
												sizeof(void *) * 6);
	if (!kern_args)
	{
		STROM_SET_ERROR(&kcxt.e, StromError_OutOfKernelArgs);
		goto out;
	}
	kern_args[0] = kgpreagg;
	kern_args[1] = kds_slot;
	kern_args[2] = kds_final;
	kern_args[3] = kresults_src;
	kern_args[4] = kresults_dst;
	kern_args[5] = f_hash;

	status = optimal_workgroup_size(&grid_sz,
									&block_sz,
									(const void *)
									gpupreagg_final_reduction,
									kresults_src->all_visible
									? kds_slot->nitems
									: kresults_src->nitems,
									0, sizeof(kern_errorbuf));
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
		goto out;
	}
	status = cudaLaunchDevice((void *)gpupreagg_final_reduction,
							  kern_args,
							  grid_sz,
							  block_sz,
							  sizeof(kern_errorbuf) * block_sz.x,
							  NULL);
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
		goto out;
	}
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kcxt.e, status);
		goto out;
	}
	else if (kgpreagg->kerror.errcode != StromError_Success)
		return;

	if (kresults_dst->nitems > 0)
	{
		/* swap */
		kresults_tmp = kresults_src;
		kresults_src = kresults_dst;
		kresults_dst = kresults_tmp;
		/* increment num_kern_fagg */
		kgpreagg->pfm.num_kern_fagg++;
		goto final_retry;
	}
	TIMEVAL_RECORD(kgpreagg,kern_fagg,tv_start);

	/*
	 * NOTE: gpupreagg_fixup_varlena shall be launched by CPU thread
	 * after synchronization of all the concurrent tasks within same
	 * segments. So, we have no chance to launch this kernel by GPU.
	 */
out:
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
