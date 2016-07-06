/*
 * cuda_gpuscan.h
 *
 * CUDA device code specific to GpuScan logic
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
#ifndef CUDA_GPUSCAN_H
#define CUDA_GPUSCAN_H

/*
 * Sequential Scan using GPU/MIC acceleration
 *
 * It packs a kern_parambuf and kern_resultbuf structure within a continuous
 * memory ares, to transfer (usually) small chunk by one DMA call.
 *
 * +----------------+       -----
 * | kern_parambuf  |         ^
 * | +--------------+         |
 * | | length   o--------------------+
 * | +--------------+         |      | kern_vrelation is located just after
 * | | nparams      |         |      | the kern_parambuf (because of DMA
 * | +--------------+         |      | optimization), so head address of
 * | | poffset[0]   |         |      | kern_gpuscan + parambuf.length
 * | | poffset[1]   |         |      | points kern_resultbuf.
 * | |    :         |         |      |
 * | | poffset[M-1] |         |      |
 * | +--------------+         |      |
 * | | variable     |         |      |
 * | | length field |         |      |
 * | | for Param /  |         |      |
 * | | Const values |         |      |
 * | |     :        |         |      |
 * +-+--------------+  -----  |  <---+
 * | kern_resultbuf |    ^    |
 * | +--------------+    |    |  Area to be sent to OpenCL device.
 * | | nrels (=1)   |    |    |  Forward DMA shall be issued here.
 * | +--------------+    |    |
 * | | nitems       |    |    |
 * | +--------------+    |    |
 * | | nrooms (=N)  |    |    |
 * | +--------------+    |    |
 * | | errcode      |    |    V
 * | +--------------+    |  -----
 * | | rindex[0]    |    |
 * | | rindex[1]    |    |  Area to be written back from OpenCL device.
 * | |     :        |    |  Reverse DMA shall be issued here.
 * | | rindex[N-1]  |    V
 * +-+--------------+  -----
 *
 * Gpuscan kernel code assumes all the fields shall be initialized to zero.
 */
typedef struct {
	kern_errorbuf	kerror;
	kern_parambuf	kparams;
} kern_gpuscan;

#define KERN_GPUSCAN_PARAMBUF(kgpuscan)			\
	((kern_parambuf *)(&(kgpuscan)->kparams))
#define KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan)	\
	STROMALIGN((kgpuscan)->kparams.length)
#define KERN_GPUSCAN_RESULTBUF(kgpuscan)		\
	((kern_resultbuf *)((char *)&(kgpuscan)->kparams +				\
						STROMALIGN((kgpuscan)->kparams.length)))
#define KERN_GPUSCAN_RESULTBUF_LENGTH(kgpuscan)	\
	STROMALIGN(offsetof(kern_resultbuf,			\
		results[KERN_GPUSCAN_RESULTBUF(kgpuscan)->nrels * \
				KERN_GPUSCAN_RESULTBUF(kgpuscan)->nrooms]))
#define KERN_GPUSCAN_LENGTH(kgpuscan)			\
	(offsetof(kern_gpuscan, kparams) +			\
	 KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan) +	\
	 KERN_GPUSCAN_RESULTBUF_LENGTH(kgpuscan))
#define KERN_GPUSCAN_DMASEND_OFFSET(kgpuscan)	0
#define KERN_GPUSCAN_DMASEND_LENGTH(kgpuscan)	\
	(KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan) +	\
	 offsetof(kern_resultbuf, results[0]))
#define KERN_GPUSCAN_DMARECV_OFFSET(kgpuscan)	0
#define KERN_GPUSCAN_DMARECV_LENGTH(kgpuscan, nitems)	\
	(KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan) +			\
	 offsetof(kern_resultbuf, results[(nitems)]))

#ifdef __CUDACC__
/*
 * forward declaration of the function to be generated on the fly
 */
STATIC_FUNCTION(cl_bool)
gpuscan_quals_eval(kern_context *kcxt,
				   kern_data_store *kds,
				   size_t kds_index);

/*
 * forward declaration of the function to be generated on the fly
 */
STATIC_FUNCTION(void)
gpuscan_projection(kern_context *kcxt,
				   kern_data_store *kds_src,
				   kern_tupitem *tupitem,
				   kern_data_store *kds_dst,
				   cl_uint dst_nitems,
				   Datum *tup_values,
				   cl_bool *tup_isnull,
				   cl_bool *tup_internal);

/*
 * kernel entrypoint of gpuscan
 */
KERNEL_FUNCTION(void)
gpuscan_exec_quals(kern_gpuscan *kgpuscan,
				   kern_data_store *kds_src)
{
	kern_parambuf  *kparams = KERN_GPUSCAN_PARAMBUF(kgpuscan);
	kern_resultbuf *kresults = KERN_GPUSCAN_RESULTBUF(kgpuscan);
	kern_context	kcxt;
	size_t			kds_index = get_global_id();
	cl_bool			rc;
	cl_uint			offset;
	cl_uint			count;
	__shared__ cl_uint base;

	/* sanity checks */
	assert(kds_src->format == KDS_FORMAT_ROW);
	assert(!kresults->all_visible);

	INIT_KERNEL_CONTEXT(&kcxt,gpuscan_exec_quals,kparams);

	/* evaluate device qualifier */
	if (kds_index < kds_src->nitems)
		rc = gpuscan_quals_eval(&kcxt, kds_src, kds_index);
	else
		rc = false;

	/* expand kresults buffer */
	offset = pgstromStairlikeSum(rc ? 1 : 0, &count);
	if (count > 0)
	{
		if (get_local_id() == 0)
			base = atomicAdd(&kresults->nitems, count);
		__syncthreads();

		if (base + count > kresults->nrooms)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			goto out;
		}
		else if (rc)
		{
			/* OK, store the result */
			kresults->results[base + offset] = (cl_uint)
				((char *)KERN_DATA_STORE_TUPITEM(kds_src, kds_index) -
				 (char *)kds_src);
		}
	}
	__syncthreads();
out:
	/* write back error status if any */
	kern_writeback_error_status(&kgpuscan->kerror, kcxt.e);
}

#ifdef GPUSCAN_DEVICE_PROJECTION
/*
 * gpuscan_projection_row
 *
 * It constructs a result tuple of GpuScan according to the required layout
 * of the result tuple. In case when row-format is required, host code never
 * call the device projection kernel unless result layout is not compatible.
 * So, entire kernel function is within #ifdef ... #endif block
 */
KERNEL_FUNCTION(void)
gpuscan_projection_row(kern_gpuscan *kgpuscan,
					   kern_data_store *kds_src,
					   kern_data_store *kds_dst)
{
	kern_parambuf  *kparams = KERN_GPUSCAN_PARAMBUF(kgpuscan);
	kern_resultbuf *kresults = KERN_GPUSCAN_RESULTBUF(kgpuscan);
	kern_context	kcxt;
	cl_uint			dst_nitems;
	cl_uint			offset;
	cl_uint			count;
	__shared__ cl_uint base;
	cl_uint			required;
	Datum			tup_values[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
	cl_bool			tup_isnull[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
	cl_bool			tup_internal[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
	cl_uint		   *tup_index = KERN_DATA_STORE_ROWINDEX(kds_dst);

	/*
	 * immediate bailout if previous stage already have error status
	 */
	kcxt.e = kgpuscan->kerror;
	if (kcxt.e.errcode != StromError_Success)
		goto out;
	INIT_KERNEL_CONTEXT(&kcxt, gpuscan_projection_row, kparams);

	/* sanity checks */
	assert(kresults->nrels == 1);
	assert(kds_src->format == KDS_FORMAT_ROW);
	assert(kds_dst->format == KDS_FORMAT_ROW && kds_dst->nslots == 0);
	/* update number of visible items */
	dst_nitems = (kresults->all_visible ? kds_src->nitems : kresults->nitems);
	if (get_global_id() == 0)
		kds_dst->nitems = dst_nitems;
	if (dst_nitems > kds_dst->nrooms)
	{
		STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		goto out;
	}

	/*
	 * step.1 - compute length of the result tuple to be written
	 */
	memset(tup_internal, 0, sizeof(tup_internal));

	if (get_global_id() < dst_nitems)
	{
		kern_tupitem   *tupitem_src;

		if (kresults->all_visible)
			tupitem_src = KERN_DATA_STORE_TUPITEM(kds_src, get_global_id());
		else
			tupitem_src = (kern_tupitem *)((char *)kds_src +
										   kresults->results[get_global_id()]);
		gpuscan_projection(&kcxt,
						   kds_src,
						   tupitem_src,
						   kds_dst,
						   dst_nitems,
						   tup_values,
						   tup_isnull,
						   tup_internal);
		required = MAXALIGN(offsetof(kern_tupitem, htup) +
							compute_heaptuple_size(&kcxt,
												   kds_dst,
												   tup_values,
												   tup_isnull,
												   tup_internal));
	}
	else
		required = 0;		/* not consume any buffer */

	/*
	 * step.2 - increment the buffer usage of kds_dst
	 */
	offset = pgstromStairlikeSum(required, &count);
	if (count > 0)
	{
		if (get_local_id() == 0)
			base = atomicAdd(&kds_dst->usage, count);
		__syncthreads();

		if (KERN_DATA_STORE_HEAD_LENGTH(kds_dst) +
			STROMALIGN(sizeof(cl_uint) * kresults->nitems) +
			base + count > kds_dst->length)
		{
			STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
			goto out;
		}
		else if (required > 0)
		{
			/*
			 * step.3 - extract the result heap-tuple
			 */
			cl_uint			pos = kds_dst->length - (base + offset + required);
			kern_tupitem   *tupitem_dst
				= (kern_tupitem *)((char *)kds_dst + pos);

			tup_index[get_global_id()] = pos;
			form_kern_heaptuple(&kcxt, kds_dst, tupitem_dst,
								tup_values, tup_isnull, tup_internal);
		}
	}
	__syncthreads();
out:
	/* write back error status if any */
	kern_writeback_error_status(&kgpuscan->kerror, kcxt.e);
}
#endif

KERNEL_FUNCTION(void)
gpuscan_projection_slot(kern_gpuscan *kgpuscan,
						kern_data_store *kds_src,
						kern_data_store *kds_dst)
{
	kern_parambuf  *kparams = KERN_GPUSCAN_PARAMBUF(kgpuscan);
	kern_resultbuf *kresults = KERN_GPUSCAN_RESULTBUF(kgpuscan);
	kern_context	kcxt;
	cl_uint			dst_nitems;
	kern_tupitem   *tupitem;
	Datum		   *tup_values;
	cl_bool		   *tup_isnull;
#ifdef GPUSCAN_DEVICE_PROJECTION
	cl_bool			tup_internal[GPUSCAN_DEVICE_PROJECTION_NFIELDS];
#endif

	/*
	 * immediate bailout if previous stage already have error status
	 */
	kcxt.e = kgpuscan->kerror;
	if (kcxt.e.errcode != StromError_Success)
		goto out;
	INIT_KERNEL_CONTEXT(&kcxt, gpuscan_projection_row, kparams);

	/* sanity checks */
	assert(kresults->nrels == 1);
	assert(kds_src->format == KDS_FORMAT_ROW);
	assert(kds_dst->format == KDS_FORMAT_SLOT);
	if (kresults->nitems > kds_dst->nrooms)
	{
		STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		goto out;
	}
	/* update number of visible items */
	dst_nitems = (kresults->all_visible ? kds_src->nitems : kresults->nitems);
	if (get_global_id() == 0)
		kds_dst->nitems = dst_nitems;
	if (dst_nitems > kds_dst->nrooms)
	{
		STROM_SET_ERROR(&kcxt.e, StromError_DataStoreNoSpace);
		goto out;
	}
	/* fetch the source tuple */
	if (get_global_id() < dst_nitems)
	{
		if (kresults->all_visible)
			tupitem = KERN_DATA_STORE_TUPITEM(kds_src, get_global_id());
		else
			tupitem = (kern_tupitem *)((char *)kds_src +
									   kresults->results[get_global_id()]);
	}
	else
		tupitem = NULL;

	tup_values = KERN_DATA_STORE_VALUES(kds_dst, get_global_id());
	tup_isnull = KERN_DATA_STORE_ISNULL(kds_dst, get_global_id());
#ifdef GPUSCAN_DEVICE_PROJECTION
	assert(kds_dst->ncols == GPUSCAN_DEVICE_PROJECTION_NFIELDS);
	gpuscan_projection(&kcxt,
					   kds_src,
					   tupitem,
					   kds_dst,
					   dst_nitems,
					   tup_values,
					   tup_isnull,
					   tup_internal);
#else
	if (tupitem != NULL)
	{
		deform_kern_heaptuple(&kcxt,
							  kds_src,
							  tupitem,
							  kds_dst->ncols,
							  true,
							  tup_values,
							  tup_isnull);
	}
#endif
out:
	/* write back error status if any */
	kern_writeback_error_status(&kgpuscan->kerror, kcxt.e);
}
#endif	/* __CUDACC__ */
#endif	/* CUDA_GPUSCAN_H */
