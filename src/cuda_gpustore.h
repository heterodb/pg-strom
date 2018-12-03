/*
 * cuda_gstore.h
 *
 * CUDA device code specific to GpuStoreFdw
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
#ifndef CUDA_GSTORE_H
#define CUDA_GSTORE_H

/*
 * kern_gpustore
 */
typedef struct
{
	kern_errorbuf	kerror;
	cl_uint			grid_sz;
	cl_uint			block_sz;
	/* runtime statistics */
	cl_uint			nitems_in;
	cl_uint			nitems_out;
	kern_parambuf	kparams;
	/* <-- gpustoreSuspendContext --> */
	/* <-- gpustoreQualResults --> */
} kern_gpustore;

typedef struct
{
	cl_uint			part_index;
} gpustoreSuspendContext;

typedef struct
{
	cl_uint			nitems;
	cl_uint			row_index[FLEXIBLE_ARRAY_MEMBER];
} gpustoreQualResults;

#define KERN_GPUSTORE_PARAMBUF(kgstore)			\
	(&(kgstore)->kparams)
#define KERN_GPUSTORE_PARAMBUF_LENGTH(kgstore)	\
	STROMALIGN((kgstore)->kparams.length)
#define KERN_GPUSTORE_SUSPEND_CONTEXT(kgstore, group_id)	\
	(((gpustoreSuspendContext *)							\
	  ((char *)KERN_GPUSTORE_PARAMBUF(kgstore) +			\
	   KERN_GPUSTORE_PARAMBUF_LENGTH(kgstore))) + (group_id))
#define KERN_GPUSTORE_QUAL_RESULTS(kgstore)		\
	((gpustoreQualResults *)					\
	 ((char *)KERN_GPUSTORE_PARAMBUF(kgstore) +	\
	  KERN_GPUSTORE_PARAMBUF_LENGTH(kgstore)))

/*
 * gpustore_quals_eval - evaluates device qualifier
 */
STATIC_FUNCTION(cl_bool)
gpustore_quals_eval(kern_context *kcxt,
					kern_data_store *kds_src,
					cl_uint row_index);

/*
 * gpustore_keycomp - compares two keys for sorting
 */
STATIC_FUNCTION(cl_int)
gpustore_keycomp(kern_context *kcxt,
				 kern_data_store *kds_src,
				 cl_uint x_index,
				 cl_uint y_index);

/*
 * gpustore_projection - setup result tuple
 */
STATIC_FUNCTION(void)
gpustore_projection(kern_context *kcxt,
					kern_data_store *kds_src,
					cl_uint row_index,
					Datum *tup_values,
					cl_bool *tup_isnull);

KERNEL_FUNCTION(void)
gpustore_exec_quals(kern_gpustore *kgstore,
					kern_data_store *kds_src)
{
	gpustoreQualResults *qresults = KERN_GPUSTORE_QUAL_RESULTS(kgstore);
	kern_context	kcxt;
	cl_uint			base;
	__shared__ cl_uint pos;

	INIT_KERNEL_CONTEXT(&kcxt, gpustore_exec_quals, &kgstore->kparams);
	for (base = get_global_base();
		 base < kds_src->nitems;
		 base += get_global_size())
	{
		cl_uint		index = base + get_local_id();
		cl_uint		offset;
		cl_uint		count;
		cl_bool		result;

		result = gpustore_quals_eval(&kcxt, kds_src, index);
		/* bailout if any error */
		if (__syncthreads_count(kcxt.e.errcode) > 0)
			break;
		offset = pgstromStairlikeSum(result ? 1 : 0, &count);
		if (get_local_id() == 0 && count > 0)
			pos = atomicAdd(&gr_index->nitems, count);
		__syncthreads();
		if (result)
			gr_index->row_index[pos + offset] = index;
		__syncthreads();
	}
	kern_writeback_error_status(&kgstore->kerror, &kcxt.e);
}

#define BITONIC_MAX_LOCAL_SHIFT		13
#define BITONIC_MAX_LOCAL_SZ		(1<<13)

KERNEL_FUNCTION_MAXTHREADS(void)
gpustore_bitonic_local(kern_gpustore *kgstore,
					   kern_data_store *kds_src)
{
	gpustoreQualResults *qresults = KERN_GPUSTORE_QUAL_RESULTS(kgstore);
	kern_context	kcxt;
	__shared__ cl_uint localIdx[BITONIC_MAX_LOCAL_SZ];		/* 32kB */
	cl_uint			localLimit;
	cl_uint			nitems = qresults->nitems;
	cl_uint			partSize = 2 * BITONIC_MAX_LOCAL_SZ;
	cl_uint			partBase = get_global_index() * partSize;
	cl_uint			blockSize;
	cl_uint			unitSize;
	cl_uint			i;

	/* quick bailout if any error happen in the prior kernel */
	if (__syncthreads_count(kgstore->kerror.errcode) != 0)
		return;
	INIT_KERNEL_CONTEXT(&kcxt, gpustore_bitonic_local, &kgstore->kparams);

	/* Load index to localIdx[] */
	if (partBase + partSize <= nitems)
		localLimit = partSize;
	else if (partBase < nitems)
		localLimit = nitems - partBase;
	else
		return;		/* too much thread-blocks are launched? */

	for (i = get_local_id(); i < localLimit; i += get_local_size())
		localIdx[i] = qresults->row_index[partBase + i];
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

					if (gpustore_keycomp(&kcxt, kds_src, pos0, pos1) > 0)
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
		qresults->row_index[partBase + i] = localIdx[i];
	__syncthreads();

	/* any errors on run-time? */
	kern_writeback_error_status(&kgstore->kerror, &kcxt.e);
}

KERNEL_FUNCTION_MAXTHREADS(void)
gpustore_bitonic_step(kern_gpustore *kgstore,
					  kern_data_store *kds_src,
					  cl_uint unitsz,
					  cl_bool reversing)
{
	gpustoreQualResults *qresults = KERN_GPUSTORE_QUAL_RESULTS(kgstore);
	kern_context	kcxt;
	cl_uint			nitems = qresults->nitems;
	cl_uint			halfUnitSize = unitsz >> 1;
	cl_uint			halfUnitMask = halfUnitSize - 1;
	cl_uint			unitMask = unitsz - 1;
	cl_uint			idx0, idx1;
	cl_uint			pos0, pos1;
	cl_uint			index;

	/* quick bailout if any error happen in the prior kernel */
	if (__syncthreads_count(kgstore->kerror.errcode) != 0)
		return;
	INIT_KERNEL_CONTEXT(&kcxt, gpustore_bitonic_step, &kgstore->kparams);

	idx0 = (((get_global_id() & ~halfUnitMask) << 1)
			+ (get_global_id() & halfUnitMask));
	idx1 = (reversing
			? ((idx0 & ~unitMask) | (~idx0 & unitMask))
			: (idx0 + halfUnitSize));
	if (idx1 < nitems)
	{
		pos0 = qresults->row_index[idx0];
		pos1 = qresults->row_index[idx1];
        if (gpusort_keycomp(&kcxt, kds_slot, pos0, pos1) > 0)
        {
            /* swap */
            qresults->row_index[idx0] = pos1;
            qresults->row_index[idx1] = pos0;
        }
    }
	kern_writeback_error_status(&kgstore->kerror);
}

KERNEL_FUNCTION_MAXTHREADS(void)
gpustore_bitonic_merge(kern_gpustore *kgstore,
					   kern_data_store *kds_src)
{
	gpustoreQualResults *qresults = KERN_GPUSTORE_QUAL_RESULTS(kgstore);
	kern_context	kcxt;
	__shared__ cl_uint localIdx[BITONIC_MAX_LOCAL_SZ];		/* 32kB */
	cl_uint			localLimit;
	cl_uint			nitems = qresults->nitems;
	cl_uint			partSize = 2 * BITONIC_MAX_LOCAL_SZ;
	cl_uint			partBase = get_global_index() * partSize;
	cl_uint			blockSize = partSize;
	cl_uint			unitSize;
	cl_uint			i;

	/* quick bailout if any error happen in the prior kernel */
	if (__syncthreads_count(kgstore->kerror.errcode) != 0)
		return;
	INIT_KERNEL_CONTEXT(&kcxt, gpustore_bitonic_merge, &kgstore->kparams);
	/* Load index to localIdx[] */
	if (partBase + partSize <= nitems)
		localLimit = partSize;
	else if (partBase < nitems)
		localLimit = nitems - partBase;
	else
		return;		/* out of range */
	for (i = get_local_id(); i < localLimit; i += get_local_size())
		localIdx[i] = qresults->row_index[partBase + i];
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
		qresults->row_index[partBase + i] = localIdx[i];
	__syncthreads();
	/* any error status? */
	kern_writeback_error_status(&kgstore->kerror);
}


KERNEL_FUNCTION(void)
gpustore_projection(kern_gpustore *kgstore,
					kern_data_store *kds_src,
					kern_data_store *kds_dst)
{
	
	gpustoreQualResults *qresults = KERN_GPUSTORE_QUAL_RESULTS(kgstore);
	kern_context	kcxt;
	cl_uint			part_index = 0;
	cl_uint			base;
	cl_uint			index;
	cl_uint			required;
#if GPUSTORE_DEVICE_PROJECTION_NFIELDS > 0
	Datum			tup_values[GPUSTORE_DEVICE_PROJECTION_NFIELDS];
	cl_bool			tup_isnull[GPUSTORE_DEVICE_PROJECTION_NFIELDS];
#else
	Datum		   *tup_values = NULL;
	cl_bool		   *tup_isnull = NULL;
#endif
	__shared__ cl_uint pos;

	/* quick bailout if any error happen in the prior kernel */
	if (__syncthreads_count(kgstore->kerror.errcode) != 0)
		return;
	INIT_KERNEL_CONTEXT(&kcxt, gpustore_projection, &kgstore->kparams);
	/* resume kernel from the point where suspended, if any */
	if (kgstore->resume_context)
	{
		assert(my_suspend != NULL);
		part_index = my_suspend->part_index;
	}

	for (base = get_global_base() + part_index * get_global_size();
		 base < qresults->nitems;
		 base += get_global_size(), part_index++)
	{
		if (base + get_local_id() < qresults->nitems)
		{
			index = qresults->row_index[base + get_local_id()];
			assert(index < kds_src->nitems);
			gpustore_projection(&kcxt,
								kds_src,
								index,
								tup_values,
								tup_isnull);
			required = MAXALIGN(offsetof(kern_tupitem, htup) +
								compute_heaptuple_size(&kcxt,
													   kds_dst,
													   tup_values,
													   tup_isnull));
		}
		else
			required = 0;

		usage_offset = pgstromStairlikeSum(required, &extra_sz);
		if (get_local_id() == 0)
		{
			union {
				struct {
					cl_uint		nitems;
					cl_uint		usage;
				} i;
				cl_ulong		v64;
			} oldval, curval, newval;

			nvalids = Min(qresults->nitems - base, get_local_size());

			curval.i.nitems = kds_dst->nitems;
            curval.i.usage  = kds_dst->usage;
			do {
				newval = oldval = curval;
				newval.i.nitems += nvalids;
				newval.i.usage  += __kds_packed(extra_sz);

				if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
					STROMALIGN(sizeof(cl_uint) * newval.i.nitems) +
					__kds_unpack(newval.i.usage) > kds_dst->length)
				{
					atomicAdd(&kgstore->suspend_count, 1);
					suspend_kernel = 1;
					break;
				}
			} while ((curval.v64 = atomicCAS((cl_ulong *)&kds_dst->nitems,
											 oldval.v64,
											 newval.v64)) != oldval.v64);
			nitems_base = oldval.i.nitems;
			usage_base = __kds_unpack(oldval.i.usage);
		}
		if (__syncthreads_count(suspend_kernel) > 0)
			break;

		/* store the result heap-tuple on the destination buffer */
		if (required > 0)
		{
			cl_uint	   *dst_index = KERN_DATA_STORE_ROWINDEX(kds_dst);
			size_t		pos;

			pos = kds_dst->length - (usage_base + usage_offset + required);
			dst_index[nitems_base + nitems_offset] = __kds_packed(pos);
			form_kern_heaptuple((kern_tupitem *)((char *)kds_dst + pos),
								kds_dst->ncols,
								kds_dst->colmeta,
								NULL,	/* ItemPointerData */
								NULL,	/* HeapTupleFields */
								kds_dst->tdhasoid ? 0xffffffff : 0,
								tup_values,
								tup_isnull);
		}
		/* bailout if any error */
		if (__syncthreads_count(kcxt.e.errcode) > 0)
			break;
	}
	/* suspend the current position (even if normal exit) */
	if (my_suspend && get_local_id() == 0)
	{
		my_suspend->part_index = part_index;
	}
	kern_writeback_error_status(&kgstore->kerror, &kcxt.e);
}

#endif	/* CUDA_GSTORE_H */
