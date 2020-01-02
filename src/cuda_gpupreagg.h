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
#ifndef CUDA_GPUPREAGG_H
#define CUDA_GPUPREAGG_H

struct kern_gpupreagg
{
	kern_errorbuf	kerror;				/* kernel error information */
	cl_uint			num_group_keys;		/* nogroup reduction, if 0 */
	cl_uint			read_slot_pos;		/* offset to read kds_slot */
	cl_uint			grid_sz;			/* grid-size of setup/join kernel */
	cl_uint			block_sz;			/* block-size of setup/join kernel */
	cl_uint			row_inval_map_size;	/* length of row-invalidation-map */
	cl_bool			setup_slot_done;	/* setup stage is done, if true */
	/* -- suspend/resume (KDS_FORMAT_BLOCK) */
	cl_bool			resume_context;		/* resume kernel, if true */
	cl_uint			suspend_count;		/* number of suspended blocks */
	cl_uint			suspend_size;		/* offset to suspend buffer, if any */
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
	kern_parambuf	kparams;
	/* <-- gpupreaggSuspendContext[], if any --> */
	/* <-- gpupreaggRowInvalidationMap. if any --> */
};
typedef struct kern_gpupreagg	kern_gpupreagg;

/*
 * gpupreaggSuspendContext is used to suspend gpupreagg_setup_block kernel.
 * Because KDS_FORMAT_BLOCK can have more items than estimation, so we cannot
 * avoid overflow of @kds_slot buffer preliminary. If @nitems exceeds @nrooms,
 * gpupreagg_setup_block will exit immediately, and save the current context
 * on the gpupreagg_suspend_context array to resume later.
 */
typedef union
{
	struct {
		size_t		src_base;
	} r;	/* row-format */
	struct {
		cl_uint		part_index;
		cl_uint		line_index;
	} b;	/* block-format */
	struct {
		size_t		src_base;
	} c;	/* arrow-format */
} gpupreaggSuspendContext;

/* macro definitions to reference packed values */
#define KERN_GPUPREAGG_PARAMBUF(kgpreagg)				\
	(&(kgpreagg)->kparams)
#define KERN_GPUPREAGG_PARAMBUF_LENGTH(kgpreagg)		\
	((kgpreagg)->kparams.length)
#define KERN_GPUPREAGG_LENGTH(kgpreagg)					\
	(offsetof(kern_gpupreagg, kparams) +				\
	 KERN_GPUPREAGG_PARAMBUF_LENGTH(kgpreagg))
/* suspend/resume buffer for KDS_FORMAT_BLOCK */
#define KERN_GPUPREAGG_SUSPEND_CONTEXT(kgpreagg,group_id)	\
	((kgpreagg)->suspend_size > 0							\
	 ? ((gpupreaggSuspendContext *)							\
		((char *)KERN_GPUPREAGG_PARAMBUF(kgpreagg) +		\
		 KERN_GPUPREAGG_PARAMBUF_LENGTH(kgpreagg))) + (group_id) \
	 : NULL)
/* row-invalidation map */
#define KERN_GPUPREAGG_ROW_INVALIDATION_MAP(kgpreagg)		\
	((cl_char *)KERN_GPUPREAGG_PARAMBUF(kgpreagg) +			\
	 KERN_GPUPREAGG_PARAMBUF_LENGTH(kgpreagg) +				\
	 (kgpreagg)->suspend_size)

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

#ifndef __CUDACC__
/*
 * gpupreagg_reset_kernel_task - reset kern_gpupreagg status prior to resume
 */
STATIC_INLINE(void)
gpupreagg_reset_kernel_task(kern_gpupreagg *kgpreagg, bool resume_context)
{
	cl_char	   *ri_map;

	memset(&kgpreagg->kerror, 0, sizeof(kern_errorbuf));
	kgpreagg->read_slot_pos   = 0;
	kgpreagg->setup_slot_done = false;
	kgpreagg->resume_context  = resume_context;
	kgpreagg->suspend_count   = 0;

	ri_map = KERN_GPUPREAGG_ROW_INVALIDATION_MAP(kgpreagg);
	memset(ri_map, 0, kgpreagg->row_inval_map_size);
}
#else	/* __CUDACC__ */
/*
 * gpupreagg_quals_eval(_arrow) - qualifier of outer scan
 */
DEVICE_FUNCTION(cl_bool)
gpupreagg_quals_eval(kern_context *kcxt,
					 kern_data_store *kds,
					 ItemPointerData *t_self,
					 HeapTupleHeaderData *htup);
DEVICE_FUNCTION(cl_bool)
gpupreagg_quals_eval_arrow(kern_context *kcxt,
						   kern_data_store *kds,
						   cl_uint row_index);

/*
 * hash value calculation function - to be generated by PG-Strom on the fly
 */
DEVICE_FUNCTION(cl_uint)
gpupreagg_hashvalue(kern_context *kcxt,
					cl_char *slot_dclass,
					Datum   *slot_values);

/*
 * comparison function - to be generated by PG-Strom on the fly
 *
 * It compares two records indexed by 'x_index' and 'y_index' on the supplied
 * kern_data_store, then returns -1 if record[X] is less than record[Y],
 * 0 if record[X] is equivalent to record[Y], or 1 if record[X] is greater
 * than record[Y].
 * (auto generated function)
 */
DEVICE_FUNCTION(cl_bool)
gpupreagg_keymatch(kern_context *kcxt,
				   kern_data_store *x_kds, size_t x_index,
				   kern_data_store *y_kds, size_t y_index);

/*
 * nogroup calculation function - to be generated by PG-Strom on the fly
 */
DEVICE_FUNCTION(void)
gpupreagg_nogroup_calc(cl_int attnum,
					   cl_char *p_accum_dclass,
					   Datum   *p_accum_datum,
					   cl_char  newval_dclass,
					   Datum    newval_datum);


/*
 * local calculation function - to be generated by PG-Strom on the fly
 */
DEVICE_FUNCTION(void)
gpupreagg_local_calc(cl_int attnum,
					 cl_char *p_accum_dclass,
					 Datum   *p_accum_datum,
					 cl_char  newval_dclass,
					 Datum    newval_datum);

/*
 * global calculation function - to be generated by PG-Strom on the fly
 */
DEVICE_FUNCTION(void)
gpupreagg_global_calc(cl_char *accum_dclass,
					  Datum   *accum_values,
					  cl_char *newval_dclass,
					  Datum   *newval_values);
/*
 * translate a kern_data_store (input) into an output form
 * (auto generated function)
 */
DEVICE_FUNCTION(void)
gpupreagg_projection_row(kern_context *kcxt,
						 kern_data_store *kds_src,	/* in */
						 HeapTupleHeaderData *htup,	/* in */
						 cl_char *dst_dclass,		/* out */
						 Datum   *dst_values);		/* out */

DEVICE_FUNCTION(void)
gpupreagg_projection_arrow(kern_context *kcxt,
						   kern_data_store *kds_src,	/* in */
						   cl_uint src_index,			/* out */
						   cl_char *dst_dclass,		/* out */
						   Datum   *dst_values);		/* out */

/*
 * GpuPreAgg initial projection
 */
DEVICE_FUNCTION(void)
gpupreagg_setup_row(kern_context *kcxt,
					kern_gpupreagg *kgpreagg,
					kern_data_store *kds_src,	/* in: KDS_FORMAT_ROW */
					kern_data_store *kds_slot);	/* out: KDS_FORMAT_SLOT */
DEVICE_FUNCTION(void)
gpupreagg_setup_block(kern_context *kcxt,
					  kern_gpupreagg *kgpreagg,
					  kern_data_store *kds_src,
					  kern_data_store *kds_slot);
DEVICE_FUNCTION(void)
gpupreagg_setup_arrow(kern_context *kcxt,
					  kern_gpupreagg *kgpreagg,
					  kern_data_store *kds_src,	  /* in: KDS_FORMAT_ARROW */
					  kern_data_store *kds_slot); /* out: KDS_FORMAT_SLOT */

/*
 * GpuPreAgg reduction functions
 */
DEVICE_FUNCTION(void)
gpupreagg_nogroup_reduction(kern_context *kcxt,
							kern_gpupreagg *kgpreagg,		/* in/out */
							kern_errorbuf *kgjoin_errorbuf,	/* in */
							kern_data_store *kds_slot,		/* in */
							kern_data_store *kds_final,		/* shared out */
							kern_global_hashslot *f_hash);	/* shared out */
DEVICE_FUNCTION(void)
gpupreagg_groupby_reduction(kern_context *kcxt,
							kern_gpupreagg *kgpreagg,		/* in/out */
							kern_errorbuf *kgjoin_errorbuf,	/* in */
							kern_data_store *kds_slot,		/* in */
							kern_data_store *kds_final,		/* shared out */
							kern_global_hashslot *f_hash);	/* shared out */
#endif /* __CUDACC__ */

/* ----------------------------------------------------------------
 *
 * A thin abstraction layer for atomic functions
 *
 * ---------------------------------------------------------------- */
#ifdef __CUDACC__
STATIC_INLINE(void)
aggcalc_atomic_min_int(cl_char *p_accum_dclass, Datum *p_accum_datum,
					   cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		cl_int	newval_int = (cl_int)(newval_datum & 0xffffffffU);

		atomicMin((cl_int *)p_accum_datum, newval_int);
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_atomic_max_int(cl_char *p_accum_dclass, Datum *p_accum_datum,
					   cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		cl_int	newval_int = (cl_int)(newval_datum & 0xffffffffU);

		atomicMax((cl_int *)p_accum_datum, newval_int);
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_atomic_add_int(cl_char *p_accum_dclass, Datum *p_accum_datum,
					   cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		cl_int		newval_int = (cl_int)(newval_datum & 0xffffffff);

		atomicAdd((cl_int *)p_accum_datum, newval_int);
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_atomic_min_long(cl_char *p_accum_dclass, Datum *p_accum_datum,
						cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		atomicMin((cl_long *)p_accum_datum, (cl_long)newval_datum);
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}


STATIC_INLINE(void)
aggcalc_atomic_max_long(cl_char *p_accum_dclass, Datum *p_accum_datum,
						cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		atomicMax((cl_long *)p_accum_datum, (cl_long)newval_datum);
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_atomic_add_long(cl_char *p_accum_dclass, Datum *p_accum_datum,
						cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		atomicAdd((cl_ulong *)p_accum_datum, (cl_ulong)newval_datum);
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_atomic_min_float(cl_char *p_accum_dclass, Datum *p_accum_datum,
						 cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
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
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_atomic_max_float(cl_char *p_accum_dclass, Datum *p_accum_datum,
						 cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
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
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_atomic_add_float(cl_char *p_accum_dclass, Datum *p_accum_datum,
						 cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		atomicAdd((cl_float *)p_accum_datum,
				  __int_as_float(newval_datum & 0xffffffff));
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_atomic_min_double(cl_char *p_accum_dclass, Datum *p_accum_datum,
						  cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
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
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_atomic_max_double(cl_char *p_accum_dclass, Datum *p_accum_datum,
						  cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
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
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_atomic_add_double(cl_char *p_accum_dclass, Datum *p_accum_datum,
						  cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		atomicAdd((cl_double *)p_accum_datum,
				  __longlong_as_double(newval_datum));
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_normal_min_int(cl_char *p_accum_dclass, Datum *p_accum_datum,
					   cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		cl_int		newval_int = (cl_int)(newval_datum & 0xffffffffU);

		*((cl_int *)p_accum_datum) = Min(*((cl_int *)p_accum_datum),
										 newval_int);
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_normal_max_int(cl_char *p_accum_dclass, Datum *p_accum_datum,
					   cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		cl_int		newval_int = (cl_int)(newval_datum & 0xffffffffU);

		*((cl_int *)p_accum_datum) = Max(*((cl_int *)p_accum_datum),
										 newval_int);
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}


STATIC_INLINE(void)
aggcalc_normal_add_int(cl_char *p_accum_dclass, Datum *p_accum_datum,
					   cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		*((cl_int *)p_accum_datum) += (cl_int)(newval_datum & 0xffffffff);
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_normal_min_long(cl_char *p_accum_dclass, Datum *p_accum_datum,
						cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		*((cl_long *)p_accum_datum) = Min(*((cl_long *)p_accum_datum),
										  (cl_long)newval_datum);
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_normal_max_long(cl_char *p_accum_dclass, Datum *p_accum_datum,
						cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		*((cl_long *)p_accum_datum) = Max(*((cl_long *)p_accum_datum),
										  (cl_long)newval_datum);
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_normal_add_long(cl_char *p_accum_dclass, Datum *p_accum_datum,
						cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		*((cl_long *)p_accum_datum) += (cl_long)newval_datum;
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_normal_min_float(cl_char *p_accum_dclass, Datum *p_accum_datum,
						 cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		*((cl_float *)p_accum_datum)
			= Min(*((cl_float *)p_accum_datum),
				  __int_as_float(newval_datum & 0xffffffff));
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_normal_max_float(cl_char *p_accum_dclass, Datum *p_accum_datum,
						 cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		*((cl_float *)p_accum_datum)
			= Max(*((cl_float *)p_accum_datum),
				  __int_as_float(newval_datum & 0xffffffff));
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_normal_add_float(cl_char *p_accum_dclass, Datum *p_accum_datum,
						 cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		*((cl_float *)p_accum_datum)
			+= __int_as_float(newval_datum & 0xffffffff);
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_normal_min_double(cl_char *p_accum_dclass, Datum *p_accum_datum,
						  cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		*((cl_double *)p_accum_datum)
			= Min(*((cl_double *)p_accum_datum),
				  __longlong_as_double((cl_ulong)newval_datum));
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_normal_max_double(cl_char *p_accum_dclass, Datum *p_accum_datum,
						  cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		*((cl_double *)p_accum_datum)
			= Max(*((cl_double *)p_accum_datum),
				  __longlong_as_double((cl_ulong)newval_datum));
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}

STATIC_INLINE(void)
aggcalc_normal_add_double(cl_char *p_accum_dclass, Datum *p_accum_datum,
						  cl_char newval_dclass, Datum newval_datum)
{
	if (newval_dclass == DATUM_CLASS__NORMAL)
	{
		*((cl_double *)p_accum_datum)
			+= __longlong_as_double((cl_ulong)newval_datum);
		*p_accum_dclass = DATUM_CLASS__NORMAL;
	}
	else
		assert(newval_dclass == DATUM_CLASS__NULL);
}
#endif	/* __CUDACC__ */

#ifdef __CUDACC_RTC__
/*
 * GPU kernel entrypoint - valid only NVRTC
 */
KERNEL_FUNCTION(void)
kern_gpupreagg_setup_row(kern_gpupreagg *kgpreagg,
						 kern_data_store *kds_src,
						 kern_data_store *kds_slot)
{
	kern_parambuf *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	DECL_KERNEL_CONTEXT(u);

	INIT_KERNEL_CONTEXT(&u.kcxt, kparams);
	gpupreagg_setup_row(&u.kcxt, kgpreagg, kds_src, kds_slot);
	kern_writeback_error_status(&kgpreagg->kerror, &u.kcxt);
}

KERNEL_FUNCTION(void)
kern_gpupreagg_setup_block(kern_gpupreagg *kgpreagg,
						   kern_data_store *kds_src,
						   kern_data_store *kds_slot)
{
	kern_parambuf *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	DECL_KERNEL_CONTEXT(u);

	INIT_KERNEL_CONTEXT(&u.kcxt, kparams);
	gpupreagg_setup_block(&u.kcxt, kgpreagg, kds_src, kds_slot);
	kern_writeback_error_status(&kgpreagg->kerror, &u.kcxt);
}

KERNEL_FUNCTION(void)
kern_gpupreagg_setup_arrow(kern_gpupreagg *kgpreagg,
						   kern_data_store *kds_src,
						   kern_data_store *kds_slot)
{
	kern_parambuf *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	DECL_KERNEL_CONTEXT(u);

	INIT_KERNEL_CONTEXT(&u.kcxt, kparams);
	gpupreagg_setup_arrow(&u.kcxt, kgpreagg, kds_src, kds_slot);
	kern_writeback_error_status(&kgpreagg->kerror, &u.kcxt);
}

KERNEL_FUNCTION(void)
kern_gpupreagg_nogroup_reduction(kern_gpupreagg *kgpreagg,
								 kern_errorbuf *kgjoin_errorbuf,
								 kern_data_store *kds_slot,
								 kern_data_store *kds_final,
								 kern_global_hashslot *f_hash)
{
	kern_parambuf *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	DECL_KERNEL_CONTEXT(u);

	INIT_KERNEL_CONTEXT(&u.kcxt, kparams);
	gpupreagg_nogroup_reduction(&u.kcxt,
								kgpreagg,
								kgjoin_errorbuf,
								kds_slot,
								kds_final,
								f_hash);
	kern_writeback_error_status(&kgpreagg->kerror, &u.kcxt);
}

KERNEL_FUNCTION(void)
kern_gpupreagg_groupby_reduction(kern_gpupreagg *kgpreagg,
								 kern_errorbuf *kgjoin_errorbuf,
								 kern_data_store *kds_slot,
								 kern_data_store *kds_final,
								 kern_global_hashslot *f_hash)
{
	kern_parambuf *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	DECL_KERNEL_CONTEXT(u);

	INIT_KERNEL_CONTEXT(&u.kcxt, kparams);
	gpupreagg_groupby_reduction(&u.kcxt,
								kgpreagg,
								kgjoin_errorbuf,
								kds_slot,
								kds_final,
								f_hash);
	kern_writeback_error_status(&kgpreagg->kerror, &u.kcxt);
}
#endif /* __CUDACC_RTC__ */
#endif /* CUDA_GPUPREAGG_H */
