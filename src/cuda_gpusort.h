/*
 * cuda_gpusort.h
 *
 * CUDA device code for GpuSort logic
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
#ifndef CUDA_GPUSORT_H
#define CUDA_GPUSORT_H

/*
 * kern_gpusort
 */
typedef struct {
	kern_errorbuf	kerror;
	cl_uint			nitems_in;
	cl_uint			nitems_out;
	kern_parambuf	kparams;
	/* <-- gpusortResultIndex --> */
} kern_gpusort;

typedef struct {
	cl_uint			nitems;
	cl_uint			results[FLEXIBLE_ARRAY_MEMBER];
} gpusortResultIndex;

#define KERN_GPUSORT_PARAMBUF(kgpusort)			\
	(&((kern_gpusort *)(kgpusort))->kparams)
#define KERN_GPUSORT_PARAMBUF_LENGTH(kgpusort)	\
	STROMALIGN(KERN_GPUSORT_PARAMBUF(kgpusort)->length)
#define KERN_GPUSORT_RESULT_INDEX(kgpusort)		\
	((gpusortResultIndex *)						\
	 ((char *)KERN_GPUSORT_PARAMBUF(kgpusort) + \
	  KERN_GPUSORT_PARAMBUF_LENGTH(kgpusort)))

#define BITONIC_MAX_LOCAL_SHIFT		12
#define BITONIC_MAX_LOCAL_SZ		(1<<BITONIC_MAX_LOCAL_SHIFT)

#ifdef __CUDACC__
/*
 * gpusort_quals_eval - evaluation of device qualifier
 */
STATIC_FUNCTION(cl_bool)
gpusort_quals_eval(kern_context *kcxt,
				   kern_data_store *kds,
				   cl_uint row_index);
/*
 * gpusort_keycomp - comparison of two keys for sorting
 */
STATIC_FUNCTION(cl_int)
gpusort_keycomp(kern_context *kcxt,
				kern_data_store *kds_src,
				cl_uint x_index,
				cl_uint y_index);
/*
 * gpusort_setup_column
 */
KERNEL_FUNCTION(void)
gpusort_setup_column(kern_gpusort *kgpusort,
					 kern_data_store *kds_src)
{
	gpusortResultIndex *kresults = KERN_GPUSORT_RESULT_INDEX(kgpusort);
	kern_context	kcxt;
	cl_uint			globalSz = get_global_size();
	cl_uint			loop, nloops;
	__shared__ cl_uint pos;

	INIT_KERNEL_CONTEXT(&kcxt, gpusort_setup_column, &kgpusort->kparams);
	assert(kds_src->format == KDS_FORMAT_COLUMN);
	nloops = (kds_src->nitems + globalSz - 1) / globalSz;
	for (loop=0; loop < nloops; loop++)
	{
		cl_uint		index = loop * globalSz + get_global_id();
		cl_uint		offset;
		cl_uint		count;
		cl_bool		retval;

		if (index < kds_src->nitems)
			retval = gpusort_quals_eval(&kcxt, kds_src, index);
		else
			retval = false;
		/* bailout if any error */
		if (__syncthreads_count(kcxt.e.errcode) > 0)
			break;
		offset = pgstromStairlikeSum(retval ? 1 : 0, &count);
		if (get_local_id() == 0 && count > 0)
			pos = atomicAdd(&kresults->nitems, count);
		__syncthreads();
		if (retval)
			kresults->results[pos + offset] = index;
		__syncthreads();
	}
	kern_writeback_error_status(&kgpusort->kerror, &kcxt.e);
}

KERNEL_FUNCTION_MAXTHREADS(void)
gpusort_bitonic_local(kern_gpusort *kgpusort,
					  kern_data_store *kds_src)
{
	gpusortResultIndex *kresults = KERN_GPUSORT_RESULT_INDEX(kgpusort);
	kern_context	kcxt;
	cl_uint			localLimit;
	cl_uint			nitems = kresults->nitems;
	cl_uint			partSize;
	cl_uint			partBase;
	cl_uint			blockSize;
	cl_uint			unitSize;
	cl_uint			i;
	__shared__ cl_uint localIdx[2 * BITONIC_MAX_LOCAL_SZ];		/* 32kB */

	/* quick bailout if any error happen in the prior kernel */
	if (__syncthreads_count(kgpusort->kerror.errcode) != 0)
		return;
	INIT_KERNEL_CONTEXT(&kcxt, gpusort_bitonic_local, &kgpusort->kparams);
	/* Adjust partition size if nitems is enough small */
	partSize = 2 * BITONIC_MAX_LOCAL_SZ;
	while (partSize / 2 > nitems)
		partSize /= 2;
	partBase = get_group_id() * partSize;

	/* Load index to localIdx[] */
	if (partBase + partSize <= nitems)
		localLimit = partSize;
	else if (partBase < nitems)
		localLimit = nitems - partBase;
	else
		return;		/* too much thread-blocks are launched? */

	for (i = get_local_id(); i < localLimit; i += get_local_size())
		localIdx[i] = kresults->results[partBase + i];
	__syncthreads();

	for (blockSize = 2; blockSize <= partSize; blockSize *= 2)
	{
		for (unitSize = blockSize; unitSize >= 2; unitSize /= 2)
		{
			cl_uint		unitMask		= (unitSize - 1);
			cl_uint		halfUnitSize	= (unitSize >> 1);
			cl_uint		halfUnitMask	= (halfUnitSize - 1);
			cl_bool		reversing		= (unitSize == blockSize);
			cl_uint		localId;
			cl_uint		idx0, idx1;

			for (localId = get_local_id();
				 localId < partSize;
				 localId += get_local_size())
			{
				idx0 = (((localId  & ~halfUnitMask) << 1) +
						 (localId  &  halfUnitMask));
				idx1 = (reversing
						? ((idx0 & ~unitMask) | (~idx0 & unitMask))
						: (halfUnitSize + idx0));
				if (idx1 < localLimit)
				{
					cl_uint		pos0 = localIdx[idx0];
					cl_uint		pos1 = localIdx[idx1];
					cl_int		comp;

					comp = gpusort_keycomp(&kcxt, kds_src, pos0, pos1);
					if (comp > 0)
					{
						/* swap */
						localIdx[idx0] = pos1;
						localIdx[idx1] = pos0;
					}
				}
			}
			__syncthreads();
		}
	}
	/* Store index on localIdx[] */
	for (i = get_local_id(); i < localLimit; i += get_local_size())
		kresults->results[partBase + i] = localIdx[i];
	__syncthreads();
	/* any errors on run-time? */
	kern_writeback_error_status(&kgpusort->kerror, &kcxt.e);
}

KERNEL_FUNCTION_MAXTHREADS(void)
gpusort_bitonic_step(kern_gpusort *kgpusort,
					 kern_data_store *kds_src,
					 cl_uint unitsz,
					 cl_bool reversing)
{
	gpusortResultIndex *kresults = KERN_GPUSORT_RESULT_INDEX(kgpusort);
	kern_context	kcxt;
	cl_uint			nitems = kresults->nitems;
	cl_uint			halfUnitSize = unitsz >> 1;
	cl_uint			halfUnitMask = halfUnitSize - 1;
	cl_uint			unitMask = unitsz - 1;
	cl_uint			idx0, idx1;
	cl_uint			pos0, pos1;

	/* quick bailout if any error happen in the prior kernel */
	if (__syncthreads_count(kgpusort->kerror.errcode) != 0)
		return;
	INIT_KERNEL_CONTEXT(&kcxt, gpusort_bitonic_step, &kgpusort->kparams);

	idx0 = (((get_global_id() & ~halfUnitMask) << 1)
			+ (get_global_id() & halfUnitMask));
	idx1 = (reversing
			? ((idx0 & ~unitMask) | (~idx0 & unitMask))
			: (idx0 + halfUnitSize));
	if (idx1 < nitems)
	{
		pos0 = kresults->results[idx0];
		pos1 = kresults->results[idx1];
        if (gpusort_keycomp(&kcxt, kds_src, pos0, pos1) > 0)
        {
            /* swap */
            kresults->results[idx0] = pos1;
			kresults->results[idx1] = pos0;
        }
    }
	kern_writeback_error_status(&kgpusort->kerror, &kcxt.e);
}

KERNEL_FUNCTION_MAXTHREADS(void)
gpusort_bitonic_merge(kern_gpusort *kgpusort,
					  kern_data_store *kds_src)
{
	gpusortResultIndex *kresults = KERN_GPUSORT_RESULT_INDEX(kgpusort);
	kern_context	kcxt;
	cl_uint			localLimit;
	cl_uint			nitems = kresults->nitems;
	cl_uint			partSize = 2 * BITONIC_MAX_LOCAL_SZ;
	cl_uint			partBase = get_group_id() * partSize;
	cl_uint			blockSize = partSize;
	cl_uint			unitSize;
	cl_uint			i;
	__shared__ cl_uint localIdx[2 * BITONIC_MAX_LOCAL_SZ];		/* 32kB */

	/* quick bailout if any error happen in the prior kernel */
	if (__syncthreads_count(kgpusort->kerror.errcode) != 0)
		return;
	INIT_KERNEL_CONTEXT(&kcxt, gpusort_bitonic_merge, &kgpusort->kparams);
	/* Load index to localIdx[] */
	if (partBase + partSize <= nitems)
		localLimit = partSize;
	else if (partBase < nitems)
		localLimit = nitems - partBase;
	else
		return;		/* out of range */
	for (i = get_local_id(); i < localLimit; i += get_local_size())
		localIdx[i] = kresults->results[partBase + i];
	__syncthreads();

	/* merge two sorted blocks */
	for (unitSize = blockSize; unitSize >= 2; unitSize >>= 1)
	{
		cl_uint		halfUnitSize = (unitSize >> 1);
		cl_uint		halfUnitMask = (halfUnitSize - 1);
		cl_uint		idx0, idx1;
		cl_uint		localId;

		for (localId = get_local_id();
			 localId < BITONIC_MAX_LOCAL_SZ;
			 localId += get_local_size())
		{
			idx0 = (((localId & ~halfUnitMask) << 1)
					+ (localId & halfUnitMask));
			idx1 = halfUnitSize + idx0;
			if (idx1 < localLimit)
			{
				cl_uint		pos0 = localIdx[idx0];
				cl_uint		pos1 = localIdx[idx1];

				if (gpusort_keycomp(&kcxt, kds_src, pos0, pos1) > 0)
				{
					localIdx[idx0] = pos1;
					localIdx[idx1] = pos0;
				}
			}
		}
		__syncthreads();
	}
	/* Store index from localIdx[] */
	for (i = get_local_id(); i < localLimit; i += get_local_size())
		kresults->results[partBase + i] = localIdx[i];
	__syncthreads();
	/* any error status? */
	kern_writeback_error_status(&kgpusort->kerror, &kcxt.e);
}
#endif	/* __CUDACC__ */
#endif	/* CUDA_GPUSORT_H */
