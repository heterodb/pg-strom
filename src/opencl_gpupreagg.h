/*
 * opencl_gpupreagg.h
 *
 * Preprocess of aggregate using GPU acceleration, to reduce number of
 * rows to be processed by CPU; including the Sort reduction.
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
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
#ifndef OPENCL_GPUPREAGG_H
#define OPENCL_GPUPREAGG_H

/*
 * Sequential Scan using GPU/MIC acceleration
 *
 * It packs a kern_parambuf and kern_resultbuf structure within a continuous
 * memory ares, to transfer (usually) small chunk by one DMA call.
 *
 * +----------------+  -----
 * | status         |    ^
 * +----------------+    |
 * | hash_size      |    |
 * +----------------+    |
 * | pg_crc32_table |    |
 * |      :         |    |
 * +----------------+    |
 * | kern_parambuf  |    |
 * | +--------------+    |
 * | | length   o--------------+
 * | +--------------+    |     | kern_row_map is located just after
 * | | nparams      |    |     | the kern_parambuf (because of DMA
 * | +--------------+    |     | optimization), so head address of
 * | | poffset[0]   |    |     | kern_gpuscan + parambuf.length
 * | | poffset[1]   |    |     | points kern_row_map.
 * | |    :         |    |     |
 * | | poffset[M-1] |    |     |
 * | +--------------+    |     |
 * | | variable     |    |     |
 * | | length field |    |     |
 * | | for Param /  |    |     |
 * | | Const values |    |     |
 * | |     :        |    |     |
 * +-+--------------+ <--------+
 * | kern_row_map   |    |
 * | +--------------+    |
 * | | nvalids (=N) |    |
 * | +--------------+    |
 * | | rindex[0]    |    |
 * | | rindex[1]    |    |
 * | |    :         |    |
 * | | rindex[N]    |    V
 * +-+--------------+  -----
 */

typedef struct
{
	cl_int			status;		/* result of kernel execution */
	cl_uint			hash_size;	/* size of global hash-slots */
	cl_uint			pg_crc32_table[256];	/* master CRC32 table */
	char			__padding__[8];		/* alignment */
	kern_parambuf	kparams;
	/*
	 *  kern_row_map shall be located next to kern_parmbuf
	 */
} kern_gpupreagg;

/* macro definitions to reference packed values */
#define KERN_GPUPREAGG_PARAMBUF(kgpreagg)				\
	((__global kern_parambuf *)(&(kgpreagg)->kparams))
#define KERN_GPUPREAGG_PARAMBUF_LENGTH(kgpreagg)		\
	(KERN_GPUPREAGG_PARAMBUF(kgpreagg)->length)
#define KERN_GPUPREAGG_KROWMAP(kgpreagg)				\
	((__global kern_row_map *)							\
	 ((__global char *)(kgpreagg) +						\
	  STROMALIGN(offsetof(kern_gpupreagg, kparams) +	\
				 KERN_GPUPREAGG_PARAMBUF_LENGTH(kgpreagg))))
#define KERN_GPUPREAGG_BUFFER_SIZE(kgpreagg, nitems)	\
	((uintptr_t)(KERN_GPUPREAGG_KROWMAP(kgpreagg)->rindex +	(nitems)) -	\
	 (uintptr_t)(kgpreagg))
#define KERN_GPUPREAGG_DMASEND_OFFSET(kgpreagg)			0
#define KERN_GPUPREAGG_DMASEND_LENGTH(kgpreagg)						\
	((uintptr_t)(KERN_GPUPREAGG_KROWMAP(kgpreagg)->rindex +			\
				 KERN_GPUPREAGG_KROWMAP(kgpreagg)->nvalids < 0 ?	\
				 0 : KERN_GPUPREAGG_KROWMAP(kgpreagg)->nvalids) -	\
	 (uintptr_t)(kgpreagg))
#define KERN_GPUPREAGG_DMARECV_OFFSET(kgpreagg)			\
	offsetof(kern_gpupreagg, status)
#define KERN_GPUPREAGG_DMARECV_LENGTH(kgpreagg)			\
	sizeof(cl_uint)

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
 * of gpupreagg_keycomp().
 * isnull indicates whether the current running total is NULL, or not.
 * XXX_val is a running total itself.
 */
typedef struct
{
//	cl_uint			group_id;
	cl_int			isnull;		/* 0: not null 1: null 2: now updating */
	union {
		cl_short	short_val;
		cl_int		int_val;
		cl_long		long_val;
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

#ifdef OPENCL_DEVICE_CODE

/* macro to check overflow on accumlate operation*/
#define CHECK_OVERFLOW_INT(x, y)				\
	((((x) < 0) == ((y) < 0)) && (((x) + (y) < 0) != ((x) < 0)))
	
#define CHECK_OVERFLOW_FLOAT(x, y)				\
	(isinf((x) + (y)) && !isinf(x) && !isinf(y))

/*
 * hash value calculation function - to be generated by PG-Strom on the fly
 */
gpupreagg_hashvalue(__private cl_int *errcode,
					__global kern_data_store *kds,
					__global kern_data_store *ktoast,
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
static cl_int
gpupreagg_keycomp(__private cl_int *errcode,
				  __global kern_data_store *kds,
				  __global kern_data_store *ktoast,
				  size_t x_index,
				  size_t y_index);

/*
 * local calculation function - to be generated by PG-Strom on the fly
 *
 * It aggregates the newval to accum using atomic operation on the
 * local pagg_datum array
 */
static void
gpupreagg_local_calc(__private cl_int *errcode,
					 cl_int attnum,
					 __local pagg_datum *accum,
					 __local pagg_datum *newval);

/*
 * global calculation function - to be generated by PG-Strom on the fly
 *
 * It also aggregates the newval to accum using atomic operation on
 * the global kern_data_store
 */
static void
gpupreagg_global_calc(__private cl_int *errcode,
					  cl_int attnum,
					  __global kern_data_store *kds,
					  __global kern_data_store *ktoast,
					  size_t accum_index,
					  size_t newval_index);

/*
 * translate a kern_data_store (input) into an output form
 * (auto generated function)
 */
static void
gpupreagg_projection(__private cl_int *errcode,
					 __global kern_parambuf *kparams,
					 __global kern_data_store *kds_in,
					 __global kern_data_store *kds_src,
					 __global void *ktoast,
					 size_t rowidx_in,
					 size_t rowidx_out);
/*
 * check qualifiers being pulled-up from the outer relation.
 * if not valid, this record shall not be processed.
 */
static bool
gpupreagg_qual_eval(__private cl_int *errcode,
					__global kern_parambuf *kparams,
					__global kern_data_store *kds,
					__global kern_data_store *ktoast,
					size_t kds_index);

/*
 * load the data from kern_data_store to pagg_datum structure
 */
static void
gpupreagg_data_load(__local pagg_datum *pdatum,
					__private cl_int *errcode,
					__global kern_data_store *kds,
					__global kern_data_store *ktoast,
					cl_uint colidx, cl_uint rowidx)
{
	kern_colmeta	cmeta;

	if (colidx >= kds->ncols)
	{
		STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
		return;
	}
	cmeta = kds->colmeta[colidx];
	/*
	 * Right now, expected data length for running total of partial aggregate
	 * are 2, 4, or 8. Elasewhere, it may be a bug.
	 */
	if (cmeta.attlen == sizeof(cl_short))
	{
		__global cl_short  *addr = kern_get_datum(kds,ktoast,colidx,rowidx);
		if (!addr)
			pdatum->isnull = true;
		else
		{
			pdatum->isnull = false;
			pdatum->short_val = *addr;
		}
	}
	else if (cmeta.attlen == sizeof(cl_int))		/* also, cl_float */
	{
		__global cl_int   *addr = kern_get_datum(kds,ktoast,colidx,rowidx);
		if (!addr)
			pdatum->isnull	= true;
		else
		{
			pdatum->isnull	= false;
			pdatum->int_val	= *addr;
		}
	}
	else if (cmeta.attlen == sizeof(cl_long))	/* also, cl_double */
	{
		__global cl_long  *addr = kern_get_datum(kds,ktoast,colidx,rowidx);
		if (!addr)
			pdatum->isnull	= true;
		else
		{
			pdatum->isnull	= false;
			pdatum->long_val= *addr;
		}
	}
	else
	{
		STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
	}
}

/*
 * store the data from pagg_datum structure to kern_data_store
 */
static void
gpupreagg_data_store(__local pagg_datum *pdatum,
					 __private cl_int *errcode,
					 __global kern_data_store *kds,
					 __global kern_data_store *ktoast,
					 cl_uint colidx, cl_uint rowidx)
{
	kern_colmeta	cmeta;

	if (colidx >= kds->ncols)
	{
		STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
		return;
	}
	cmeta = kds->colmeta[colidx];
	/*
	 * Right now, expected data length for running total of partial aggregate
	 * are 2, 4, or 8. Elasewhere, it may be a bug.
	 */
	if (cmeta.attlen == sizeof(cl_short))
	{
		pg_int2_t	temp;

		temp.isnull = pdatum->isnull;
		temp.value  = pdatum->short_val;
		pg_int2_vstore(kds, ktoast, errcode, colidx, rowidx, temp);
	}
	else if (cmeta.attlen == sizeof(cl_uint))		/* also, cl_float */
	{
		pg_int4_t	temp;

		temp.isnull	= pdatum->isnull;
		temp.value	= pdatum->int_val;
		pg_int4_vstore(kds, ktoast, errcode, colidx, rowidx, temp);
	}
	else if (cmeta.attlen == sizeof(cl_ulong))	/* also, cl_double */
	{
		pg_int8_t	temp;

		temp.isnull	= pdatum->isnull;
		temp.value	= pdatum->long_val;
		pg_int8_vstore(kds, ktoast, errcode, colidx, rowidx, temp);
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
static void
gpupreagg_data_move(__private cl_int *errcode,
					__global kern_data_store *kds_src,
					__global kern_data_store *kds_dst,
					__global kern_data_store *ktoast,
					cl_uint colidx,
					cl_uint rowidx_src,
					cl_uint rowidx_dst)
{
	__global Datum	   *src_values;
	__global Datum	   *dst_values;
	__global cl_char   *src_isnull;
	__global cl_char   *dst_isnull;

	if (colidx >= kds_src->ncols || colidx >= kds_dst->ncols)
	{
		STROM_SET_ERROR(errcode, StromError_DataStoreCorruption);
		return;
	}

	src_values = KERN_DATA_STORE_VALUES(kds_src, rowidx_src);
	src_isnull = KERN_DATA_STORE_ISNULL(kds_src, rowidx_src);
	dst_values = KERN_DATA_STORE_VALUES(kds_dst, rowidx_dst);
	dst_isnull = KERN_DATA_STORE_ISNULL(kds_dst, rowidx_dst);

	if (src_isnull[colidx])
	{
		dst_isnull[colidx] = (cl_char) 1;
		dst_values[colidx] = (Datum) 0;
	}
	else
	{
		dst_isnull[colidx] = (cl_char) 0;
		dst_values[colidx] = src_values[colidx];
	}
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
__kernel void
gpupreagg_preparation(__global kern_gpupreagg *kgpreagg,
					  __global kern_data_store *kds_in,
					  __global kern_data_store *kds_src,
					  __global pagg_hashslot *g_hashslot,
					  KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
{
	__global kern_parambuf *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	__global kern_row_map  *krowmap = KERN_GPUPREAGG_KROWMAP(kgpreagg);
	cl_int					errcode = StromError_Success;
	cl_uint					offset;
	cl_uint					nitems;
	size_t					hash_size;
	size_t					curr_index;
	size_t					kds_index;
	__local cl_uint			base;

	/*
	 * filters out invisible rows
	 */
	if (krowmap->nvalids < 0)
		kds_index = get_global_id(0);
	else if (get_global_id(0) < krowmap->nvalids)
		kds_index = (size_t) krowmap->rindex[get_global_id(0)];
	else
		kds_index = kds_in->nitems;	/* ensure this thread is out of range */

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
									  LOCAL_WORKMEM,
									  &nitems);

	/* Allocation of the result slot on the kds_src. */
	if (get_local_id(0) == 0)
	{
		if (nitems > 0)
			base = atomic_add(&kds_src->nitems, nitems);
		else
			base = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

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
	kern_writeback_error_status(&kgpreagg->status, errcode, LOCAL_WORKMEM);
}

/*
 * gpupreagg_init_global_hashslot
 *
 * It intends to be called prior to gpupreagg_global_reduction(),
 * if gpupreagg_local_reduction() is not called. It initialized
 * the global hash-table and kern_row_map->nvalids.
 */
__kernel void
gpupreagg_init_global_hashslot(__global kern_gpupreagg *kgpreagg,
							   __global pagg_hashslot *g_hashslot)
{
	__global kern_row_map *krowmap = KERN_GPUPREAGG_KROWMAP(kgpreagg);
	size_t		hash_size;
	size_t		curr_index;

	if (get_global_id(0) == 0)
		krowmap->nvalids = 0;

	hash_size = kgpreagg->hash_size;
	for (curr_index = get_global_id(0);
		 curr_index < hash_size;
		 curr_index += get_global_size(0))
	{
		g_hashslot[curr_index].hash = 0;
        g_hashslot[curr_index].index = (cl_uint)(0xffffffff);
	}
}

/*
 * gpupreagg_local_reduction
 */
__kernel void
gpupreagg_local_reduction(__global kern_gpupreagg *kgpreagg,
						  __global kern_data_store *kds_src,
						  __global kern_data_store *kds_dst,
						  __global kern_data_store *ktoast,
						  __global pagg_hashslot *g_hashslot,
						  KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
{
	size_t					hash_size = 2 * get_local_size(0);
	size_t					dest_index;
	cl_uint					owner_index;
	cl_uint					hash_value;
	cl_uint					nitems = kds_src->nitems;
	cl_uint					ngroups;
	cl_uint					index;
	cl_uint					nattrs = kds->ncols;
	cl_uint					attnum;
	pagg_hashslot			old_slot;
	pagg_hashslot			new_slot;
	pagg_hashslot			cur_slot;
	__local cl_uint			crc32_table[256];
	__local size_t			base_index;
	__local pagg_datum	   *l_datum;
	__local pagg_hashslot  *l_hashslot;

	/* next stage expect g_hashslot is correctly initialized */
	gpupreagg_init_global_hashslot(kgpreagg, g_hashslot);

	/*
	 * calculation of the hash value of grouping keys in this record.
	 * It tends to take massive amount of random access on global memory,
	 * so it makes performance advantage to move the master table from
	 * gloabl to the local memory first.
	 */
	for (index = get_local_id(0);
		 index < lengthof(crc32_table);
		 index += get_local_size(0))
		crc32_table[index] = kgpreagg->pg_crc32_table[index];
	barrier(CLK_LOCAL_MEM_FENCE);

	if (get_global_id(0) < nitems)
		hash_value = gpupreagg_hashvalue(&errcode, kds_src, ktoast,
										 get_global_id(0));
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
	l_hashslot = (__local pagg_hashslot *)STROMALIGN(LOCAL_WORKMEM);
	for (index = get_local_id(0);
		 index < hash_size;
		 index += get_local_size(0))
	{
		l_hashslot[index].hash = 0;
		l_hashslot[index].index = (cl_uint)(0xffffffff);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	new_slot.hash = hash_value;
	new_slot.index = get_local_id(0);
	old_slot.hash = 0;
	old_slot.index = (cl_uint)(0xffffffff);
	index = hash_value % hash_size;

	if (get_global_id(0) < nitems)
	{
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
		else
		{
			size_t	buddy_index
				= (get_global_id(0) - get_local_id(0) + cur_slot.index);

			if (cur_slot.hash == new_slot.hash &&
				gpupreagg_keycomp(&errcode,
								  kds_src, ktoast,
								  get_global_id(0),
								  buddy_index) == 0)
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
    barrier(CLK_LOCAL_MEM_FENCE);

	/*
	 * Make a reservation on the destination kern_data_store
	 * Only thread that is responsible to grouping-key (also, it shall
	 * have same hash-index with get_local_id(0)) takes a place on the
	 * destination kern_data_store.
	 */
	index = arithmetic_stairlike_add(get_local_id(0) == owner_index ? 1 : 0,
									 LOCAL_WORKMEM, &ngroups);
	if (get_local_id(0) == 0)
		base_index = atomic_add(&kds_dst->nitems, ngroups);
	barrier(CLK_LOCAL_MEM_FENCE);
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
	l_datum = (__local pagg_datum *)STROMALIGN(LOCAL_WORKMEM);
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
			if (owner_index == get_local_id(0))
			{
				gpupreagg_data_move(&errcode,
									kds_src, kds_dst, ktoast,
									attnum,
									get_global_id(0),
									dest_index);
				/* also, fixup varlena datum if needed */
				pg_fixup_tupslot_varlena(&errcode, kds_dst, ktoast,
										 attnum, dest_index);
			}
			continue;
		}

		/* Load aggregation item to pagg_datum */
		if (get_global_id(0) < nitems)
		{
            gpupreagg_data_load(l_datum + get_local_id(0),
                                &errcode,
                                kds_src, ktoast,
                                cindex, get_global_id(0));
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Reduction, using local atomic operation */
		if (get_global_id(0) < nitems &&
			get_global_id(0) != owner_index)
		{
			gpupreagg_local_calc(&errcode,
								 attnum,
								 l_datum + owner_index,
								 l_datum + get_local_id(0));
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		/* Move the value that is aggregated */
		if (owner_index == get_local_id(0))
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
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

/*
 * gpupreagg_global_reduction
 */
__kernel void
gpupreagg_global_reduction(__global kern_gpupreagg *kgpreagg,
                           __global kern_data_store *kds_dst,
                           __global kern_data_store *ktoast,
                           __global pagg_hashslot *g_hashslot,
                           KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
{
	__global kern_row_map *krowmap = KERN_GPUPREAGG_KROWMAP(gpreagg);
	size_t			hash_size = kgpreagg->hash_size;
	size_t			dest_index;
	size_t			owner_index;
	cl_uint			hash_value;
	cl_uint			nitems = kds_dst->nitems;
	cl_uint			ngroups;
	cl_uint			index;
	cl_uint			nattrs = kds_dst->ncols;
	cl_uint			attnum;
	pagg_hashslot	old_slot;
	pagg_hashslot	new_slot;
	pagg_hashslot	cur_slot;
	__local cl_uint	crc32_table[256];
	__local size_t	base_index;

	/*
	 * calculation of the hash value of grouping keys in this record.
	 * It tends to take massive amount of random access on global memory,
	 * so it makes performance advantage to move the master table from
	 * gloabl to the local memory first.
	 */
	for (index = get_local_id(0);
		 index < lengthof(crc32_table);
		 index += get_local_size(0))
		crc32_table[index] = kgpreagg->pg_crc32_table[index];
	barrier(CLK_LOCAL_MEM_FENCE);

	if (get_global_id(0) < nitems)
		hash_value = gpupreagg_hashvalue(&errcode, kds_src, ktoast,
										 get_global_id(0));
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
	if (get_global_id(0) < nitems)
	{
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
				 gpupreagg_keycomp(&errcode,
								   kds_src, ktoast,
								   get_global_id(0),
								   cur_slot.index) == 0)
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
	 *
	 * NOTE: Length of kern_row_map should be same as kds->nrooms.
	 * So, we can use kds->nrooms to check array boundary.
	 */
	barrier(CLK_LOCAL_MEM_FENCE);
	index = arithmetic_stairlike_add(get_global_id(0) == owner_index ? 1 : 0,
									 LOCAL_WORKMEM, &ngroups);
	if (get_local_id(0) == 0)
		base_index = atomic_add(&krowmap->nvalids, ngroups);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (kds_dst->nrooms < base_index + ngroups)
	{
		errcode = StromError_DataStoreNoSpace;
		goto out;
	}
	dest_index = base_index + index;

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
		/*
		 * nothing to do for grouping-keys
		 */
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
		if (get_global_id(0) < nitems)
		{
			if (get_global_id(0) == owner_index)
				krowmap->rindex[dest_index] = get_global_id(0);
			else
			{
				gpupreagg_global_calc(&errcode,
									  kds_dst,
									  ktoast,
									  owner_index,
									  get_global_id(0));
			}
		}
	}
}

/*
 * Helper macros for gpupreagg_local_calc
 *
 * NOTE: please assume the variables below are available in the context
 * these macros in use.
 *   __local pagg_datum *accum;
 *   __local pagg_datum *newval;
 *   int                 state;
 */
#define AGGCALC_LOCAL32_TEMPLATE(ATOMIC_FUNC_CALL)		\
	do {												\
		if (!newval->isnull)							\
			break;										\
		state = atomic_cmpxchg(&accum->isnull, 1, 2);	\
		if (state == 0)			/* it was NOT NULL */	\
		{												\
			ATOMIC_FUNC_CALL;							\
		}												\
		else if (state == 1)	/* it was NULL */		\
		{												\
			newval->int_val = newval->int_val;			\
			newval->isnull = 0;							\
		}												\
	} while (state > 1)

#define AGGCALC_LOCAL64_TEMPLATE(ATOMIC_FUNC_CALL)		\
	do {												\
		if (!newval->isnull)							\
			break;										\
		state = atomic_cmpxchg(&accum->isnull, 1, 2);	\
		if (state == 0)			/* it was NOT NULL */	\
		{												\
			ATOMIC_FUNC_CALL;							\
		}												\
		else if (state == 1)	/* it was NULL */		\
		{												\
			newval->long_val = newval->long_val;		\
			newval->isnull = 0;							\
		}												\
	} while (state > 1)

/* original atomic functions */
static inline float
latomic_max_float(volatile __local float *ptr, float value)
{
	uint	oldval = to_uint(*ptr);
	uint	newval = to_uint(max(to_float(oldval), value));
	uint	curval;

	while ((curval = atomic_cmpxchg((__global uint *)ptr,
									oldval, newval)) != oldval)
	{
		oldval = curval;
		newval = to_uint(max(to_float(oldval),value));
	}
	return to_float(oldval);
}

static inline float
latomic_min_float(volatile __local float *ptr, float value)
{
	uint	oldval = to_uint(*ptr);
	uint	newval = to_uint(min(to_float(oldval), value));
	uint	curval;

	while ((curval = atomic_cmpxchg((__global uint *)ptr,
									oldval, newval)) != oldval)
	{
		oldval = curval;
		newval = to_uint(min(to_float(oldval),value));
	}
	return to_float(oldval);
}

static inline float
latomic_add_float(volatile __local float *ptr, float value)
{
	uint	oldval = to_uint(*ptr);
	uint	newval = to_uint(to_float(oldval) + value));
	uint	curval;

	while ((curval = atomic_cmpxchg((__global uint *)ptr,
									oldval, newval)) != oldval)
	{
		oldval = curval;
		newval = to_uint(to_float(oldval) + value));
	}
	return to_float(oldval);
}

static inline double
latomic_max_double(volatile __local double *ptr, float value)
{
	ulong	oldval = to_ulong(*ptr);
	ulong	newval = to_ulong(max(to_double(oldval), value));
	ulong	curval;

	while ((curval = atomic_cmpxchg((__global ulong *)ptr,
									oldval, newval)) != oldval)
	{
		oldval = curval;
		newval = to_ulong(max(to_double(oldval),value));
	}
	return to_double(oldval);
}

static inline double
latomic_min_double(volatile __local double *ptr, float value)
{
	ulong	oldval = to_ulong(*ptr);
	ulong	newval = to_ulong(min(to_double(oldval), value));
	ulong	curval;

	while ((curval = atomic_cmpxchg((__global ulong *)ptr,
									oldval, newval)) != oldval)
	{
		oldval = curval;
		newval = to_ulong(min(to_double(oldval),value));
	}
	return to_double(oldval);
}

static inline double
latomic_add_double(volatile __local double *ptr, float value)
{
	ulong	oldval = to_ulong(*ptr);
	ulong	newval = to_ulong(to_double(oldval) + value);
	ulong	curval;

	while ((curval = atomic_cmpxchg((__global ulong *)ptr,
									oldval, newval)) != oldval)
	{
		oldval = curval;
		newval = to_ulong(to_double(oldval) + value);
	}
	return to_double(oldval);
}

#ifdef PG_NUMERIC_TYPE_DEFINED
static inline cl_ulong
latomic_max_numeric(__private int *errcode,
					volatile __local cl_ulong *ptr, cl_ulong value)
{
	pg_numeric_t	x, y, z;
	int				comp;

	x.isnull = false;
	y.isnull = false;
	x.value  = *ptr;
	y.value  = value;
	comp = pgfn_numeric_cmp(errcode, x, y);
}

static inline cl_ulong
latomic_min_numeric(__private int *errcode,
					volatile __local cl_ulong *ptr, cl_ulong value)
{}

static inline cl_ulong
latomic_add_numeric(__private int *errcode,
					volatile __local cl_ulong *ptr, cl_ulong value)
{}
#endif

/* calculation for partial max */
#define GPUPREAGG_AGGCALC_PMAX_SHORT(errcode)							\
	AGGCALC_LOCAL32_TEMPLATE(atomic_max(&accum->int_val,				\
										(int)newval->short_val))
#define GPUPREAGG_AGGCALC_PMAX_INT(errcode)								\
	AGGCALC_LOCAL32_TEMPLATE(atomic_max(&accum->int_val, newval->int_val))
#define GPUPREAGG_AGGCALC_PMAX_LONG(errcode)							\
	AGGCALC_LOCAL64_TEMPLATE(atom_max(&accum->long_val, newval->long_val))
#define GPUPREAGG_AGGCALC_PMAX_FLOAT(errcode)							\
    AGGCALC_LOCAL32_TEMPLATE(latomic_max_float(&accum->float_val,		\
											   newval->float_val))
#define GPUPREAGG_AGGCALC_PMAX_DOUBLE(errcode)							\
	AGGCALC_LOCAL64_TEMPLATE(latomic_max_double(&accum->double_val,		\
												newval->double_val))
#define GPUPREAGG_AGGCALC_PMAX_NUMERIC(errcode)							\
	AGGCALC_LOCAL64_TEMPLATE(latomic_max_numeric(&accum->long_val,		\
												 newval->long_val))

/* calculation for partial min */
#define GPUPREAGG_AGGCALC_PMIN_SHORT(errcode)							\
	AGGCALC_LOCAL32_TEMPLATE(atomic_min(&accum->int_val,				\
										(int)newval->short_val))
#define GPUPREAGG_AGGCALC_PMIN_INT(errcode)								\
	AGGCALC_LOCAL32_TEMPLATE(atomic_min(&accum->int_val, newval->int_val))
#define GPUPREAGG_AGGCALC_PMIN_LONG(errcode)							\
	AGGCALC_LOCAL64_TEMPLATE(atom_min(&accum->long_val, newval->long_val))
#define GPUPREAGG_AGGCALC_PMIN_FLOAT(errcode)							\
    AGGCALC_LOCAL32_TEMPLATE(latomic_min_float(&accum->float_val,		\
											   newval->float_val))
#define GPUPREAGG_AGGCALC_PMIN_DOUBLE(errcode)							\
	AGGCALC_LOCAL64_TEMPLATE(latomic_min_double(&accum->double_val,		\
												newval->double_val))
#define GPUPREAGG_AGGCALC_PMIN_NUMERIC(errcode)							\
	AGGCALC_LOCAL64_TEMPLATE(latomic_min_numeric(&accum->long_val,		\
												 newval->long_val))

/* calculation for partial sum */
#define GPUPREAGG_AGGCALC_PSUM_SHORT(errcode)							\
	AGGCALC_LOCAL32_TEMPLATE(atomic_add(&accum->int_val,				\
										(int)newval->short_val))
#define GPUPREAGG_AGGCALC_PSUM_INT(errcode)								\
	AGGCALC_LOCAL32_TEMPLATE(atomic_add(&accum->int_val, newval->int_val))
#define GPUPREAGG_AGGCALC_PSUM_LONG(errcode)							\
	AGGCALC_LOCAL64_TEMPLATE(atom_add(&accum->long_val, newval->long_val))
#define GPUPREAGG_AGGCALC_PSUM_FLOAT(errcode)							\
    AGGCALC_LOCAL32_TEMPLATE(latomic_add_float(&accum->float_val,		\
											   newval->float_val))
#define GPUPREAGG_AGGCALC_PSUM_DOUBLE(errcode)							\
	AGGCALC_LOCAL64_TEMPLATE(latomic_add_double(&accum->double_val,		\
												newval->double_val))
#define GPUPREAGG_AGGCALC_PSUM_NUMERIC(errcode)							\
	AGGCALC_LOCAL64_TEMPLATE(latomic_add_numeric(&accum->long_val,		\
												 newval->long_val))




/* calculation for partial min */


#define GPUPREAGG_AGGCALC_PMAX_SHORT(errcode,accum,newval)      \
    GPUPREAGG_AGGCALC_PMAX_TEMPLATE(short,(accum),(newval))


GPUPREAGG_AGGCALC_PMAX_SHORT(

/* Partial Max functions */
AGGCALC_LOCAL32_TEMPLATE(pmax,short,
						 atomic_max(&accum->int_val,
									(cl_int) newval->short_val))
AGGCALC_LOCAL32_TEMPLATE(pmax,int,
						 atomic_max(&accum->int_val,
									(cl_int) newval->int_val))
AGGCALC_LOCAL64_TEMPLATE(pmax,long,
						 atom_max(&accum->long_val,
								  newval->long_val))
AGGCALC_LOCAL32_TEMPLATE(pmax,float,
						 atomic_max_local_float(&accum->float_val,
												newval->float_val))
AGGCALC_LOCAL64_TEMPLATE(pmax,double,
						 atomic_max_local_double(&accum->double_val,
												 newval->double_val))
#ifdef PG_NUMERIC_TYPE_DEFINED
AGGCALC_LOCAL64_TEMPLATE(pmax,numeric,
						 atomic_max_local_numeric(&accum->long_val,
												  newval->long_val))
#endif

/* Partial Min functions */
AGGCALC_LOCAL32_TEMPLATE(pmin,short,
						 atomic_min(&accum->int_val,
									(cl_int) newval->short_val))
AGGCALC_LOCAL32_TEMPLATE(pmin,int,
						 atomic_min(&accum->int_val,
									(cl_int) newval->int_val))
AGGCALC_LOCAL64_TEMPLATE(pmin,long,
						 atom_min(&accum->long_val,
								  newval->long_val))
AGGCALC_LOCAL32_TEMPLATE(pmin,float,
						 atomic_min_local_float(&accum->float_val,
												newval->float_val))
AGGCALC_LOCAL64_TEMPLATE(pmin,double,
						 atomic_min_local_double(&accum->double_val,
												 newval->double_val))
#ifdef PG_NUMERIC_TYPE_DEFINED
AGGCALC_LOCAL64_TEMPLATE(pmax,numeric,
						 atomic_max_local_numeric(&accum->long_val,
												  newval->long_val))
#endif











#define GPUPREAGG_AGGCALC_PMAX_TEMPLATE(FIELD,accum,newval)		\
	if (!(newval)->isnull)										\
	{															\
		if ((accum)->isnull)									\
			(accum)->FIELD##_val = (newval)->FIELD##_val;		\
		else													\
			(accum)->FIELD##_val = max((accum)->FIELD##_val,	\
									   (newval)->FIELD##_val);	\
		(accum)->isnull = false;								\
	}

#define GPUPREAGG_AGGCALC_PMIN_TEMPLATE(FIELD,accum,newval)		\
	if (!(newval)->isnull)										\
	{															\
		if ((accum)->isnull)									\
			(accum)->FIELD##_val = (newval)->FIELD##_val;		\
		else													\
			(accum)->FIELD##_val = min((accum)->FIELD##_val,	\
									   (newval)->FIELD##_val);	\
		(accum)->isnull = false;								\
	}

#define GPUPREAGG_AGGCALC_PMINMAX_NUMERIC_TEMPLATE(OP,errcode,accum,newval) \
	if (!(newval)->isnull)										\
	{															\
		if ((accum)->isnull)									\
			(accum)->long_val = (newval)->long_val;				\
		else													\
		{														\
			pg_numeric_t	x;									\
			pg_numeric_t	y;									\
																\
			x.isnull = y.isnull = false;						\
			x.value = (accum)->long_val;						\
			y.value = (newval)->long_val;						\
																\
			if (numeric_cmp(errcode,x,y) OP 0)					\
				(accum)->long_val = (newval)->long_val;			\
		}														\
		(accum)->isnull = false;								\
	}

/* In-kernel PMAX() implementation */
#define GPUPREAGG_AGGCALC_PMAX_SHORT(errcode,accum,newval)		\
	GPUPREAGG_AGGCALC_PMAX_TEMPLATE(short,(accum),(newval))
#define GPUPREAGG_AGGCALC_PMAX_INT(errcode,accum,newval)		\
	GPUPREAGG_AGGCALC_PMAX_TEMPLATE(int,(accum),(newval))
#define GPUPREAGG_AGGCALC_PMAX_LONG(errcode,accum,newval)		\
	GPUPREAGG_AGGCALC_PMAX_TEMPLATE(long,(accum),(newval))
#define GPUPREAGG_AGGCALC_PMAX_FLOAT(errcode,accum,newval)		\
	GPUPREAGG_AGGCALC_PMAX_TEMPLATE(float,(accum),(newval))
#define GPUPREAGG_AGGCALC_PMAX_DOUBLE(errcode,accum,newval)		\
	GPUPREAGG_AGGCALC_PMAX_TEMPLATE(double,(accum),(newval))
#define GPUPREAGG_AGGCALC_PMAX_NUMERIC(errcode,accum,newval)	\
	GPUPREAGG_AGGCALC_PMINMAX_NUMERIC_TEMPLATE(<,errcode,accum,newval)

/* In-kernel PMIN() implementation */
#define GPUPREAGG_AGGCALC_PMIN_SHORT(errcode,accum,newval)		\
	GPUPREAGG_AGGCALC_PMIN_TEMPLATE(short,(accum),(newval))
#define GPUPREAGG_AGGCALC_PMIN_INT(errcode,accum,newval)		\
	GPUPREAGG_AGGCALC_PMIN_TEMPLATE(int,(accum),(newval))
#define GPUPREAGG_AGGCALC_PMIN_LONG(errcode,accum,newval)		\
	GPUPREAGG_AGGCALC_PMIN_TEMPLATE(long,(accum),(newval))
#define GPUPREAGG_AGGCALC_PMIN_FLOAT(errcode,accum,newval)		\
	GPUPREAGG_AGGCALC_PMIN_TEMPLATE(float,(accum),(newval))
#define GPUPREAGG_AGGCALC_PMIN_DOUBLE(errcode,accum,newval)		\
	GPUPREAGG_AGGCALC_PMIN_TEMPLATE(double,(accum),(newval))
#define GPUPREAGG_AGGCALC_PMIN_NUMERIC(errcode,accum,newval)	\
	GPUPREAGG_AGGCALC_PMINMAX_NUMERIC_TEMPLATE(>,errcode,accum,newval)

/* In-kernel PSUM() implementation */
#define GPUPREAGG_AGGCALC_PSUM_TEMPLATE(FIELD,OVERFLOW,errcode,accum,newval) \
	if (!(accum)->isnull)											\
	{																\
		if (!(newval)->isnull)										\
		{															\
			if (OVERFLOW((accum)->FIELD##_val,						\
						 (newval)->FIELD##_val))					\
				STROM_SET_ERROR(errcode, StromError_CpuReCheck);	\
			(accum)->FIELD##_val += (newval)->FIELD##_val;			\
		}															\
	}																\
	else if (!(newval)->isnull)										\
	{																\
		(accum)->isnull = (newval)->isnull;							\
		(accum)->FIELD##_val = (newval)->FIELD##_val;				\
	}

#define GPUPREAGG_AGGCALC_PSUM_SHORT(errcode,accum,newval)			\
	GPUPREAGG_AGGCALC_PSUM_TEMPLATE(short,CHECK_OVERFLOW_INT,		\
									(errcode),(accum),(newval))
#define GPUPREAGG_AGGCALC_PSUM_INT(errcode,accum,newval)			\
	GPUPREAGG_AGGCALC_PSUM_TEMPLATE(int,CHECK_OVERFLOW_INT,			\
									(errcode),(accum),(newval))
#define GPUPREAGG_AGGCALC_PSUM_LONG(errcode,accum,newval)			\
	GPUPREAGG_AGGCALC_PSUM_TEMPLATE(long,CHECK_OVERFLOW_INT,		\
									(errcode),(accum),(newval))
#define GPUPREAGG_AGGCALC_PSUM_FLOAT(errcode,accum,newval)			\
	GPUPREAGG_AGGCALC_PSUM_TEMPLATE(float,CHECK_OVERFLOW_FLOAT,		\
									(errcode),(accum),(newval))
#define GPUPREAGG_AGGCALC_PSUM_DOUBLE(errcode,accum,newval)			\
	GPUPREAGG_AGGCALC_PSUM_TEMPLATE(double,CHECK_OVERFLOW_FLOAT,	\
									(errcode),(accum),(newval))
#define GPUPREAGG_AGGCALC_PSUM_NUMERIC(errcode,accum,newval)		\
	if (!(accum)->isnull)											\
	{																\
		if (!(newval)->isnull)										\
		{															\
			pg_numeric_t	x;										\
			pg_numeric_t	y;										\
			pg_numeric_t	r;										\
																	\
			x.isnull = y.isnull = false;							\
			x.value = (accum)->long_val;							\
			y.value = (newval)->long_val;							\
																	\
			r = pgfn_numeric_add(errcode,x,y);						\
																	\
			(accum)->long_val = r.value;							\
		}															\
	}																\
	else if (!(newval)->isnull)										\
	{																\
		(accum)->isnull = (newval)->isnull;							\
		(accum)->long_val = (newval)->long_val;						\
	}

#else
/* Host side representation of kern_gpupreagg. It can perform as a message
 * object of PG-Strom, has key of OpenCL device program, a source row/column
 * store and a destination kern_data_store.
 */
typedef struct
{
	pgstrom_message	msg;		/* = StromTag_GpuPreAgg */
	Datum			dprog_key;	/* key of device program */
	bool			skip_local_reduction;
	bool			needs_grouping;	/* true, if it needs grouping step */
	double			num_groups;	/* estimated number of groups */
	pgstrom_data_store *pds;	/* source data-store */
	pgstrom_data_store *pds_dest; /* result data-store */
	kern_gpupreagg	kern;		/* kernel portion to be sent */
} pgstrom_gpupreagg;
#endif	/* OPENCL_DEVICE_CODE */
#endif	/* OPENCL_GPUPREAGG_H */
