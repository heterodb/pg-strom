/*
 * opencl_gpupreagg.h
 *
 * Preprocess of aggregate using GPU acceleration, to reduce number of
 * rows to be processed by CPU; including the Sort reduction.
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
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
 * | rindex_len     |    |
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
 * | rindex[] for   |    ^
 * |  working of    |    |
 * |  bitonic sort  |  device onlye memory
 * |       :        |    |
 * |       :        |    V
 * +----------------+  -----
 */

typedef struct
{
	cl_int			status;		/* result of kernel execution */
	cl_int			sortbuf_len;/* length of sorting rindex[] that holds
								 * row-index being sorted; must be length
								 * of get_next_log2(nitems)
								 */
	char			__padding[8];	/* align to 128bits */
	kern_parambuf	kparams;
	/*
	 * kern_row_map and rindexp[] for sorting will be here
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
#define KERN_GPUPREAGG_SORT_RINDEX(kgpreagg)			\
	(KERN_GPUPREAGG_KROWMAP(kgpreagg)->rindex +			\
	 (KERN_GPUPREAGG_KROWMAP(kgpreagg)->nvalids < 0		\
	  ? 0 : KERN_GPUPREAGG_KROWMAP(kgpreagg)->nvalids))
#define KERN_GPUPREAGG_BUFFER_SIZE(kgpreagg)			\
	((uintptr_t)(KERN_GPUPREAGG_SORT_RINDEX(kgpreagg) +	\
				 (kgpreagg)->sortbuf_len) -				\
	 (uintptr_t)(kgpreagg))
#define KERN_GPUPREAGG_DMASEND_OFFSET(kgpreagg)			0
#define KERN_GPUPREAGG_DMASEND_LENGTH(kgpreagg)			\
	((uintptr_t)KERN_GPUPREAGG_SORT_RINDEX(kgpreagg) -	\
	 (uintptr_t)(kgpreagg))
#define KERN_GPUPREAGG_DMARECV_OFFSET(kgpreagg)			\
	offsetof(kern_gpupreagg, status)
#define KERN_GPUPREAGG_DMARECV_LENGTH(kgpreagg)			\
	sizeof(cl_uint)

/*
 * NOTE: pagg_datum is a set of information to calculate running total.
 * group_id indicates which group does this work-item belong to, instead
 * of gpupreagg_keycomp().
 * isnull indicates whether the current running total is NULL, or not.
 * XXX_val is a running total itself.
 */
typedef struct
{
	cl_uint			group_id;
	cl_char			isnull;
	cl_char			__padding__[3];
	union {
		cl_uint		int_val;
		cl_ulong	long_val;
		cl_float	float_val;
		cl_double	double_val;
	};
} pagg_datum;

#ifdef OPENCL_DEVICE_CODE

/*
 * pg_XXX_vstore - an interface function that stores a datum on
 * the destination kern_data_store.
 */
static __global void *
pg_common_vstore(__private cl_int *errcode,
				 __global kern_data_store *kds,
				 cl_uint colidx,
				 cl_uint rowidx,
				 bool isnull)
{
	kern_colmeta		cmeta = kds->colmeta[colidx];
	__global cl_uint   *nullmap;
	cl_uint				nullmask;
	cl_uint				offset = 0;

	/* only column-store can be written in the kernel space */
	if (!kds->column_form)
	{
		if (!StromErrorIsSignificant(*errcode))
			*errcode = StromError_DataStoreCorruption;
		return NULL;
	}
	/* out of range? */
	if (colidx >= kds->ncols || rowidx >= kds->nrooms)
	{
		if (!StromErrorIsSignificant(*errcode))
			*errcode = StromError_DataStoreOutOfRange;
		return NULL;
	}
	cmeta = kds->colmeta[colidx];
	/* only null can be allowed to store value on invalid column */
	if (!cmeta.attvalid)
	{
		if (!isnull && !StromErrorIsSignificant(*errcode))
			*errcode = StromError_DataStoreCorruption;
		return NULL;
	}

	offset = cmeta.cs_offset;
	nullmap = ((__global cl_uint *)
			   ((__global cl_char *)kds + offset)) + (rowidx >> 5);
	nullmask = (1U << (rowidx & 0x1f));
	if (isnull)
	{
		if (!cmeta.attnotnull)
			atomic_and(nullmap, ~nullmask);
		else
		{
			if (!StromErrorIsSignificant(*errcode))
				*errcode = StromError_DataStoreCorruption;
		}
		return NULL;
	}

	if (!cmeta.attnotnull)
	{
		atomic_or(nullmap, nullmask);
		offset += STROMALIGN(bitmaplen(kds->nrooms));
	}
	__global cl_char *hoge
		= (__global cl_char *)kds + offset + (cmeta.attlen > 0 ?
											   cmeta.attlen :
											   sizeof(cl_uint)) * rowidx;
	printf("vstore col=%d row=%d offset=%u hoge-kds=%zu\n", colidx, rowidx, offset, (uintptr_t)hoge - (uintptr_t)kds);
	return hoge;
}

#define STROMCL_SIMPLE_VARSTORE_TEMPLATE(NAME,BASE)			\
	static void												\
	pg_##NAME##_vstore(__global kern_data_store *kds,		\
					   __global kern_toastbuf *ktoast,		\
					   __private int *errcode,				\
					   cl_uint colidx,						\
					   cl_uint rowidx,						\
					   pg_##NAME##_t datum)					\
	{														\
		__global BASE  *cs_addr								\
			= pg_common_vstore(errcode, kds,				\
							   colidx, rowidx,				\
							   datum.isnull);				\
		if (cs_addr)										\
			*cs_addr = datum.value;							\
	}

#define STROMCL_VARLENA_VARSTORE_TEMPLATE(NAME)				\
	static void												\
	pg_##NAME##_vstore(__global kern_data_store *kds,		\
					   __global kern_toastbuf *ktoast,		\
					   __private int *errcode,				\
					   cl_uint colidx,						\
					   cl_uint rowidx,						\
					   pg_##NAME##_t datum)					\
	{														\
		__global cl_uint   *cs_addr							\
			= pg_common_vstore(errcode, kds,				\
							   colidx, rowidx,				\
							   datum.isnull);				\
		if (cs_addr)										\
		{													\
			cl_uint		vl_offset							\
				= (cl_uint)((uintptr_t)datum.value -		\
							(uintptr_t)ktoast);				\
			if (ktoast->length == TOASTBUF_MAGIC)			\
				vl_offset -= ktoast->coldir[colidx];		\
															\
			*cs_addr = vl_offset;							\
		}													\
	}

/* built-in declarations */
#ifndef PG_INT4_TYPE_DEFINED
#define PG_INT4_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int4,cl_int)
#endif
#ifndef PG_INT8_TYPE_DEFINED
#define PG_INT8_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int8,cl_long)
#endif

STROMCL_SIMPLE_VARSTORE_TEMPLATE(int4,cl_int);
STROMCL_SIMPLE_VARSTORE_TEMPLATE(int8,cl_long);

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
				  __global kern_toastbuf *ktoast,
				  size_t x_index,
				  size_t y_index);
/*
 * calculation function - to be generated by PG-Strom on the fly
 *
 * It updates the supplied 'accum' value by 'newval' value. Both of data
 * structure is expected to be on the local memory.
 * (auto generated function)
 */
static void
gpupreagg_aggcalc(__private cl_int *errcode,
				  cl_int resno,
				  __local pagg_datum *accum,
				  __local pagg_datum *newval);

/*
 * translate a kern_data_store (input) into an output form
 * (auto generated function)
 */
static void
gpupreagg_projection(__private cl_int *errcode,
					 __global kern_data_store *kds_in,
					 __global kern_data_store *kds_out,
					 __global kern_toastbuf *ktoast,
					 size_t rowidx_in,
					 size_t rowidx_out);

/*
 * load the data from kern_data_store to pagg_datum structure
 */
static void
gpupreagg_data_load(__local pagg_datum *pdatum,
					__private cl_int *errcode,
					__global kern_data_store *kds,
					__global kern_toastbuf *ktoast,
					cl_uint colidx, cl_uint rowidx)
{
	kern_colmeta	cmeta;

	if (colidx >= kds->ncols)
	{
		if (!StromErrorIsSignificant(*errcode))
			*errcode = StromError_DataStoreCorruption;
		return;
	}
	cmeta = kds->colmeta[colidx];
	/*
	 * Right now, expected data length for running total of partial aggregate
	 * are 4, or 8. Elasewhere, it may be a bug.
	 */
	if (cmeta.attlen == sizeof(cl_uint))		/* also, cl_float */
	{
		__global cl_uint   *addr = kern_get_datum(kds,ktoast,colidx,rowidx);
		if (!addr)
			pdatum->isnull	= true;
		else
		{
			pdatum->isnull	= false;
			pdatum->int_val	= *addr;
		}
	}
	else if (cmeta.attlen == sizeof(cl_ulong))	/* also, cl_double */
	{
		__global cl_ulong  *addr = kern_get_datum(kds,ktoast,colidx,rowidx);
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
		if (!StromErrorIsSignificant(*errcode))
			*errcode = StromError_DataStoreCorruption;
	}
}

/*
 * store the data from pagg_datum structure to kern_data_store
 */
static void
gpupreagg_data_store(__local pagg_datum *pdatum,
					 __private cl_int *errcode,
					 __global kern_data_store *kds,
					 __global kern_toastbuf *ktoast,
					 cl_uint colidx, cl_uint rowidx,
					 __local void *local_workbuf)
{
	kern_colmeta	cmeta;

	if (colidx >= kds->ncols)
	{
		if (!StromErrorIsSignificant(*errcode))
			*errcode = StromError_DataStoreCorruption;
		return;
	}
	cmeta = kds->colmeta[colidx];
	/*
	 * Right now, expected data length for running total of partial aggregate
	 * are 4, or 8. Elasewhere, it may be a bug.
	 */
	if (cmeta.attlen == sizeof(cl_uint))		/* also, cl_float */
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
		if (!StromErrorIsSignificant(*errcode))
			*errcode = StromError_DataStoreCorruption;
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
					__global kern_toastbuf *ktoast,
					cl_uint colidx,
					cl_uint rowidx_src,
					cl_uint rowidx_dst)
{
	__global char	   *addr_src;
	__global cl_uint   *nullmap;
	__global void	   *src_datum;
	cl_uint				nullmask;
	cl_uint				cs_offset;

	if (kds_src->ncols != kds_dst->ncols ||
		colidx >= kds_src->ncols ||
		!kds_src->colmeta[colidx].attvalid ||
		!kds_dst->colmeta[colidx].attvalid ||
		kds_src->colmeta[colidx].attlen != kds_dst->colmeta[colidx].attlen)
	{
		*errcode = StromError_DataStoreCorruption;
		return;
	}

	cs_offset = kds_src->colmeta[colidx].cs_offset;
	nullmap = (__global cl_uint *)
		((__global char *)kds_dst + cs_offset + (rowidx_dst >> 5));
	nullmask = (1U << (rowidx_dst & 0x1f));

	src_datum = kern_get_datum(kds_src, ktoast, colidx, rowidx_src);
	if (!src_datum)
	{
		if (!kds_dst->colmeta[colidx].attnotnull)
			atomic_and(nullmap, ~nullmask);
		else
			*errcode = StromError_DataStoreCorruption;
	}
	else
	{
		__global cl_char   *dest_addr;
		cl_short			attlen = kds_dst->colmeta[colidx].attlen;

		if (!kds_dst->colmeta[colidx].attnotnull)
		{
			atomic_or(nullmap, nullmask);
			cs_offset += STROMALIGN(bitmaplen(kds_dst->nrooms));
		}
		dest_addr = ((__global cl_char *) kds_dst +
					 cs_offset + attlen * rowidx_dst);
		switch (attlen)
		{
			case sizeof(cl_char):
				*((__global cl_char *) dest_addr)
					= *((__global cl_char *) src_datum);
				break;
			case sizeof(cl_short):
				*((__global cl_short *) dest_addr)
					= *((__global cl_short *) src_datum);
				break;
			case sizeof(cl_int):
				*((__global cl_int *) dest_addr)
					= *((__global cl_int *) src_datum);
				break;
			case sizeof(cl_long):
				*((__global cl_long *) dest_addr)
					= *((__global cl_long *) src_datum);
				break;
			default:
				if (attlen > 0)
					memcpy(dest_addr, src_datum, attlen);
				else
					*((__global cl_uint *) dest_addr) =
						*((__global cl_uint *) src_datum);
		}
	}
}

/*
 * gpupreagg_preparation - It translaes an input kern_data_store (that
 * reflects outer relation's tupdesc) into the form of running total
 * and final result of gpupreagg (that reflects target-list of GpuPreAgg).
 */
__kernel void
gpupreagg_preparation(__global kern_gpupreagg *kgpreagg,
					  __global kern_data_store *kds_in,
					  __global kern_data_store *kds_out,
					  __global kern_toastbuf *ktoast,
					  __local void *local_memory)
{
	__global kern_parambuf *kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	__global kern_row_map  *krowmap = KERN_GPUPREAGG_KROWMAP(kgpreagg);
//	__global cl_int		   *rindex = KERN_GPUPREAGG_ROW_INDEX(kgpreagg);
	cl_int					errcode = StromError_Success;
	cl_uint					offset;
	cl_uint					nitems;
	size_t					kds_index;
	__local cl_uint			base;

	if (krowmap->nvalids < 0)
		kds_index = get_global_id(0);
	else if (get_global_id(0) < krowmap->nvalids)
		kds_index = (size_t) krowmap->rindex[get_global_id(0)];
	else
		kds_index = kds_in->nitems;	/* ensure this thread is out of range */

	/* calculation of total number of rows to be processed in this work-
	 * group.
	 */
	offset = arithmetic_stairlike_add(kds_index < kds_in->nitems ? 1 : 0,
									  local_memory,
									  &nitems);

	/* Allocation of the result slot on the kds_out. */
	if (get_local_id(0) == 0)
	{
		if (nitems > 0)
			base = atomic_add(&kds_out->nitems, nitems);
		else
			base = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	/* out of range check -- usually, should not happen */
	if (base + nitems > kds_out->nrooms)
	{
		errcode = StromError_DataStoreNoSpace;
		goto out;
	}

	/* do projection */
	if (kds_index < kds_in->nitems)
	{
		gpupreagg_projection(&errcode,
							 kds_in, kds_out, ktoast,
							 kds_index,			/* rowidx of kds_in */
							 base + offset);	/* rowidx of kds_out */
	}
out:
	/* write-back execution status into host-side */
	kern_writeback_error_status(&kgpreagg->status, errcode, local_memory);
}

/*
 * gpupreagg_reduction - entrypoint of the main logic for GpuPreAgg.
 * The both of kern_data_store have identical form that reflects running 
 * total and final results. rindex will show the sorted order according
 * to the gpupreagg_keycomp() being constructed on the fly.
 * This function makes grouping at first, then run data reduction within
 * the same group. 
 */
__kernel void
gpupreagg_reduction(__global kern_gpupreagg *kgpreagg,
					__global kern_data_store *kds_src,
					__global kern_data_store *kds_dst,
					__global kern_toastbuf *ktoast,
					__local void *local_memory)
{
	__global kern_parambuf	*kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	__global cl_int			*rindex  = KERN_GPUPREAGG_SORT_RINDEX(kgpreagg);
	__local pagg_datum		*l_data  = local_memory;
	__local void	 *l_workbuf = (__local void *)&l_data[get_local_size(0)];
	__global varlena *kparam_3 = kparam_get_value(kparams, 3);
	cl_uint	pagg_natts = VARSIZE_EXHDR(kparam_3) / sizeof(cl_uint);
	__global cl_uint *pagg_anums = (__global cl_uint *) VARDATA(kparam_3);

	cl_int pindex		= 0;

	cl_int ncols		= kds_src->ncols;
	cl_int nrows		= kds_src->nitems;
	cl_int errcode		= StromError_Success;

	cl_int localID     = get_local_id(0);
	cl_int globalID        = get_global_id(0);
    cl_int localSize   = get_local_size(0);

	cl_int prtID		= globalID / localSize;	/* partition ID */
	cl_int prtSize		= localSize;			/* partition Size */
	cl_int prtMask		= prtSize - 1;			/* partition Mask */
	cl_int prtPos		= prtID * prtSize;		/* partition Position */

	cl_int localEntry  = (prtPos+prtSize < nrows) ? prtSize : (nrows-prtPos);
	/* Check no data for this work group */
	if(localEntry <= 0) {
		goto out;
	}

	/* Generate group id of local work group. */
	cl_int groupID;
	cl_uint ngroups;
	{
		cl_int isNewID = 0;

		if(localID == 0)
		{
	        isNewID = 1;
		}
		else if(localID < localEntry)
		{
			int rv = gpupreagg_keycomp(&errcode, kds_src, ktoast,
									   rindex[globalID-1], rindex[globalID]);
			isNewID = (rv != 0) ? 1 : 0;
		}
		groupID = arithmetic_stairlike_add(isNewID, local_memory, &ngroups);
	}

	/* allocation of result buffer */
	__local cl_uint base;
	{
		if (get_local_id(0) == 0)
			base = atomic_add(&kds_dst->nitems, ngroups);
		barrier(CLK_LOCAL_MEM_FENCE);

		if (kds_dst->nrooms <= base + ngroups) {
			errcode = StromError_DataStoreNoSpace;
			goto out;
		}
	}
	/* Aggregate for each item. */
	for (cl_int cindex=0; cindex < ncols; cindex++)
	{
		/* In case when column is neither grouping-key nor partial
		 * aggregation, we have nothing to do. So, move to the next
		 * column.
		 */
		if (!kds_src->colmeta[cindex].attvalid)
			continue;

		/* In case when column is a grouping-key (thus, no partial
		 * aggregation is defined), all we need to do is copying
		 * the data from source to destination.
		 */
		if (pindex < pagg_natts && cindex == pagg_anums[pindex])
		{
			gpupreagg_data_move(&errcode, kds_src, kds_dst, ktoast,
								cindex,
								rindex[globalID],	/* source rowid */
								base + groupID);	/* destination rowid */
			pindex++;
			continue;
		}

		/* Load aggregate item */
		l_data[localID].group_id = -1;
		if(localID < localEntry) {
			gpupreagg_data_load(&l_data[localID], &errcode, kds_src, ktoast,
								cindex, rindex[globalID]);
			l_data[localID].group_id = groupID;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		// Reduction
		for(int unitSize=2; unitSize<=prtSize; unitSize*=2) {
			if(localID % unitSize == unitSize/2  &&  localID < localEntry) {
				cl_int  dstID;

				dstID = localID - unitSize/2;
				if(l_data[localID].group_id == l_data[dstID].group_id) {
					// Marge this aggregate data to lower.
					gpupreagg_aggcalc(&errcode, cindex,
									  &l_data[dstID], &l_data[localID]);
					l_data[localID].group_id = -1;
				}
				barrier(CLK_LOCAL_MEM_FENCE);

				if(l_data[localID].group_id != -1  &&
				   localID + unitSize/2 < localEntry) {
					dstID = localID + unitSize/2;
					if(l_data[localID].group_id == l_data[dstID].group_id) {
						// Marge this aggregate data to upper.
						gpupreagg_aggcalc(&errcode, cindex,
										  &l_data[dstID], &l_data[localID]);
						l_data[localID].group_id = -1;
					}
				}
				barrier(CLK_LOCAL_MEM_FENCE);
			}
		}
		// write back aggregate data
		if(l_data[localID].group_id != -1) {
			gpupreagg_data_store(&l_data[localID], &errcode, kds_dst, ktoast,
								 cindex, base + groupID, l_workbuf);
		}
    }
out:
	kern_writeback_error_status(&kgpreagg->status, errcode, l_workbuf);
}

/*
 * gpupreagg_bitonic_local
 *
 * It tries to apply each steps of bitonic-sorting until its unitsize
 * reaches the workgroup-size (that is expected to power of 2).
 */
__kernel void
gpupreagg_bitonic_local(__global kern_gpupreagg *kgpreagg,
						__global kern_data_store *kds,
						__global kern_toastbuf *ktoast,
						__local void *local_memory)
{
	__global kern_parambuf	*kparams  = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	__global cl_int			*rindex	  = KERN_GPUPREAGG_SORT_RINDEX(kgpreagg);
	__local  cl_int			*localIdx = local_memory;

	cl_int nrows		= kds->nitems;
	cl_int errcode		= StromError_Success;

    cl_int localID		= get_local_id(0);
    cl_int globalID		= get_global_id(0);
    cl_int localSize	= get_local_size(0);

    cl_int prtID		= globalID / localSize; /* partition ID */
    cl_int prtSize		= localSize * 2;		/* partition Size */
    cl_int prtMask		= prtSize - 1;			/* partition Mask */
    cl_int prtPos		= prtID * prtSize;		/* partition Position */

    cl_int localEntry	= ((prtPos + prtSize < nrows)
						   ? prtSize
						   : (nrows - prtPos));

    // create row index and then store to localIdx
    if(localID < localEntry)
		localIdx[localID] = prtPos + localID;

    if(localSize + localID < localEntry)
		localIdx[localSize + localID] = prtPos + localSize + localID;

    barrier(CLK_LOCAL_MEM_FENCE);


	// bitonic sort
	for(int blockSize=2; blockSize<=prtSize; blockSize*=2)
	{
		int blockMask		= blockSize - 1;
		int halfBlockSize	= blockSize / 2;
		int halfBlockMask	= halfBlockSize -1;

		for(int unitSize=blockSize; 2<=unitSize; unitSize/=2)
		{
			int unitMask		= unitSize - 1;
			int halfUnitSize	= unitSize / 2;
			int halfUnitMask	= halfUnitSize - 1;

			bool reversing	= unitSize == blockSize ? true : false;
			int idx0 = ((localID / halfUnitSize) * unitSize
						+ localID % halfUnitSize);
			int idx1 = ((reversing == true)
						? ((idx0 & ~unitMask) | (~idx0 & unitMask))
						: (halfUnitSize + idx0));

			if(idx1 < localEntry) {
				cl_int pos0 = localIdx[idx0];
				cl_int pos1 = localIdx[idx1];
				cl_int rv   = gpupreagg_keycomp(&errcode, kds, ktoast,
												pos0, pos1);

				if(0 < rv) {
					// swap
					localIdx[idx0] = pos1;
					localIdx[idx1] = pos0;
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
    }

    if(localID < localEntry)
		rindex[prtPos + localID] = localIdx[localID];

    if(localSize + localID < localEntry)
		rindex[prtPos + localSize + localID] = localIdx[localSize + localID];

	kern_writeback_error_status(&kgpreagg->status, errcode, local_memory);
}



/*
 * gpupreagg_bitonic_step
 *
 * It tries to apply individual steps of bitonic-sorting for each step,
 * but does not have restriction of workgroup size. The host code has to
 * control synchronization of each step not to overrun.
 */
__kernel void
gpupreagg_bitonic_step(__global kern_gpupreagg *kgpreagg,
					   cl_int bitonic_unitsz,
					   __global kern_data_store *kds,
					   __global kern_toastbuf *ktoast,
					   __local void *local_memory)
{
	__global kern_parambuf	*kparams = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	__global cl_int			*rindex	 = KERN_GPUPREAGG_SORT_RINDEX(kgpreagg);

	cl_int	nrows	  = kds->nitems;
	cl_bool reversing = (bitonic_unitsz < 0 ? true : false);
	size_t	unitsz    = (bitonic_unitsz < 0
						 ? 1U << -bitonic_unitsz
						 : 1U << bitonic_unitsz);
	cl_int	errcode	  = StromError_Success;

	cl_int	threadID		= get_global_id(0);
	cl_int	halfUnitSize	= unitsz / 2;
	cl_int	unitMask		= unitsz - 1;

	cl_int	idx0;
	cl_int	idx1;

	idx0 = (threadID / halfUnitSize) * unitsz + threadID % halfUnitSize;
	idx1 = (reversing
			? ((idx0 & ~unitMask) | (~idx0 & unitMask))
			: (idx0 + halfUnitSize));
	if(nrows <= idx1)
		return;

	cl_int	pos0	= rindex[idx0];
	cl_int	pos1	= rindex[idx1];
	cl_int	rv;

	rv = gpupreagg_keycomp(&errcode, kds, ktoast, pos0, pos1);
	if(0 < rv) {
		/* Swap */
		rindex[idx0] = pos1;
		rindex[idx1] = pos0;
	}

	kern_writeback_error_status(&kgpreagg->status, errcode, local_memory);
}

/*
 * gpupreagg_bitonic_merge
 *
 * It handles the merging step of bitonic-sorting if unitsize becomes less
 * than or equal to the workgroup size.
 */
__kernel void
gpupreagg_bitonic_merge(__global kern_gpupreagg *kgpreagg,
						__global kern_data_store *kds,
						__global kern_toastbuf *ktoast,
						__local void *local_memory)
{
	__global kern_parambuf	*kparams  = KERN_GPUPREAGG_PARAMBUF(kgpreagg);
	__global cl_int			*rindex	  = KERN_GPUPREAGG_SORT_RINDEX(kgpreagg);
	__local	 cl_int			*localIdx = local_memory;

	cl_int nrows		= kds->nitems;
	cl_int errcode		= StromError_Success;

    cl_int localID		= get_local_id(0);
    cl_int globalID		= get_global_id(0);
    cl_int localSize	= get_local_size(0);

    cl_int prtID		= globalID / localSize; /* partition ID */
    cl_int prtSize		= localSize * 2;		/* partition Size */
    cl_int prtMask		= prtSize - 1;			/* partition Mask */
    cl_int prtPos		= prtID * prtSize;		/* partition Position */

    cl_int localEntry	= (prtPos+prtSize < nrows) ? prtSize : (nrows-prtPos);


    // load index to localIdx
    if(localID < localEntry)
		localIdx[localID] = rindex[prtPos + localID];

    if(localSize + localID < localEntry)
		localIdx[localSize + localID] = rindex[prtPos + localSize + localID];

    barrier(CLK_LOCAL_MEM_FENCE);


	// marge sorted block
	int blockSize		= prtSize;
	int blockMask		= blockSize - 1;
	int halfBlockSize	= blockSize / 2;
	int halfBlockMask	= halfBlockSize -1;

	for(int unitSize=blockSize; 2<=unitSize; unitSize/=2)
	{
		int unitMask		= unitSize - 1;
		int halfUnitSize	= unitSize / 2;
		int halfUnitMask	= halfUnitSize - 1;

		int idx0 = localID / halfUnitSize * unitSize + localID % halfUnitSize;
		int idx1 = halfUnitSize + idx0;

		if(idx1 < localEntry) {
			cl_int pos0 = localIdx[idx0];
			cl_int pos1 = localIdx[idx1];
			cl_int rv = gpupreagg_keycomp(&errcode, kds, ktoast, pos0, pos1);

			if(0 < rv) {
				// swap
				localIdx[idx0] = pos1;
				localIdx[idx1] = pos0;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(localID < localEntry)
		rindex[prtPos + localID] = localIdx[localID];

    if(localSize + localID < localEntry)
		rindex[prtPos + localSize + localID] = localIdx[localSize + localID];

	kern_writeback_error_status(&kgpreagg->status, errcode, local_memory);
}




/*
 * gpupreagg_check_next - decision making whether we need to run next
 * reduction step actually.
 */
__kernel void
gpupreagg_check_next(__global kern_gpupreagg *kgpreagg,
					 __global kern_data_store *kds_old,
					 __global kern_data_store *kds_new)
{
#if 0
	/* used this function ? */
	size_t	nrows_old	= kds_old->nitems;
	size_t	nrows_new	= kds_new->nitems;

	bool	needNextReduction;

	if(nrows_old->flag_needNextReduction == false) {
		needNextReduction = false;
	} else {
		if((nrows_old - nrows_new) < nrows_old / 10) {
			needNextReduction = false;
		} else {
			needNextReduction = true;
		}
	}

	nrows_new->flag_needNextReduction = true;
#endif
}

#else
/*
 * special system parameter of gpupreagg
 * KPARAM_2 - kds_head of the source/target kern_data_store
 * KPARAM_3 - array of referenced and aggregated column index
 */
static inline kern_data_store *
KPARAM_GET_KDS_HEAD_DEST(kern_parambuf *kparams)
{
	bytea  *vl_datum = kparam_get_value(kparams, 2);

	if (!vl_datum)
		return NULL;
	return (kern_data_store *)VARDATA_ANY(vl_datum);
}




/* Host side representation of kern_gpupreagg. It can perform as a message
 * object of PG-Strom, has key of OpenCL device program, a source row/column
 * store and a destination kern_data_store.
 */
typedef struct
{
	pgstrom_message		msg;		/* = StromTag_GpuPreAgg */
	Datum				dprog_key;	/* key of device program */
	StromObject		   *rcstore;	/* source row/column store as input */
	kern_data_store	   *kds_dst;	/* result buffer of partial aggregate */
	kern_gpupreagg		kern;		/* kernel portion to be sent */
} pgstrom_gpupreagg;
#endif	/* OPENCL_DEVICE_CODE */
#endif	/* OPENCL_GPUPREAGG_H */
