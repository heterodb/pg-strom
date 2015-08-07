/*
 * cuda_gpuscan.h
 *
 * CUDA device code specific to GpuScan logic
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
#define KERN_GPUSCAN_DMARECV_OFFSET(kgpuscan)	\
	KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan)
#define KERN_GPUSCAN_DMARECV_LENGTH(kgpuscan)	\
	KERN_GPUSCAN_RESULTBUF_LENGTH(kgpuscan)

#ifdef __CUDACC__
/*
 * gpuscan_writeback_results
 *
 * It writes back the calculation result of gpuscan.
 */
STATIC_FUNCTION(void)
gpuscan_writeback_results(kern_resultbuf *kresults, int result)
{
	__shared__ cl_uint base;
	size_t		result_index = get_global_id() + 1;
	cl_uint		binary;
	cl_uint		offset;
	cl_uint		nitems;

	assert(kresults->nrels == 1);

	/*
	 * A typical usecase of arithmetic_stairlike_add with binary value:
	 * It takes 1 if thread wants to return a status to the host side,
	 * then stairlike-add returns a relative offset within workgroup,
	 * and we can adjust this offset by global base index.
	 */
	binary = (result != 0 ? 1 : 0);
	offset = arithmetic_stairlike_add(binary, &nitems);
	if (get_local_id() == 0)
		base = atomicAdd(&kresults->nitems, nitems);
	__syncthreads();

	/*
	 * Write back the row-index that passed evaluation of the qualifier,
	 * or needs re-check on the host side. In case of re-check, row-index
	 * shall be a negative number.
	 */
	if (result > 0)
		kresults->results[base + offset] =  result_index;
	else if (result < 0)
		kresults->results[base + offset] = -result_index;
}

/*
 * forward declaration of the function to be generated on the fly
 */
STATIC_FUNCTION(cl_bool)
gpuscan_qual_eval(kern_context *kcxt,
				  kern_data_store *kds,
				  kern_data_store *ktoast,
				  size_t kds_index);
/*
 * kernel entrypoint of gpuscan
 */
KERNEL_FUNCTION(void)
gpuscan_qual(kern_gpuscan *kgpuscan,	/* in/out */
			 kern_data_store *kds,		/* in */
			 kern_data_store *ktoast)	/* always NULL */
{
	kern_parambuf  *kparams = KERN_GPUSCAN_PARAMBUF(kgpuscan);
	kern_resultbuf *kresults = KERN_GPUSCAN_RESULTBUF(kgpuscan);
	kern_context	kcxt;
	size_t			kds_index = get_global_id();
	cl_int			rc = 0;

	INIT_KERNEL_CONTEXT(&kcxt,gpuscan_qual,kparams);

	if (kds_index < kds->nitems)
	{
		if (gpuscan_qual_eval(&kcxt, kds, ktoast, kds_index))
		{
			if (kcxt.e.errcode == StromError_Success)
				rc = 1;
			else if (kcxt.e.errcode == StromError_CpuReCheck)
			{
				rc = -1;
				kcxt.e.errcode = StromError_Success;	/* CPU rechecks */
			}
		}
		else if (kcxt.e.errcode == StromError_CpuReCheck)
		{
			rc = -1;
			kcxt.e.errcode = StromError_Success;		/* CPU rechecks */
		}
	}
	/* writeback the results */
	gpuscan_writeback_results(kresults, rc);
	/* chunk level error, if any */
	kern_writeback_error_status(&kresults->kerror, kcxt.e);
}

#endif	/* __CUDACC__ */
#endif	/* CUDA_GPUSCAN_H */
