/*
 * device_gpuscan.h
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
#ifndef DEVICE_GPUSCAN_H
#define DEVICE_GPUSCAN_H

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
	((__global kern_parambuf *)(&(kgpuscan)->kparams))
#define KERN_GPUSCAN_PARAMBUF_LENGTH(kgpuscan)	\
	STROMALIGN((kgpuscan)->kparams.length)
#define KERN_GPUSCAN_RESULTBUF(kgpuscan)		\
	((__global kern_resultbuf *)				\
	 ((__global char *)&(kgpuscan)->kparams +	\
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

#ifdef OPENCL_DEVICE_CODE
/*
 * gpuscan_writeback_row_error
 *
 * It writes back the calculation result of gpuscan.
 */
static void
gpuscan_writeback_row_error(__global kern_resultbuf *kresults,
							int errcode,
							__local void *workmem)
{
	__local cl_uint *p_base = workmem;
	cl_uint		base;
	cl_uint		binary;
	cl_uint		offset;
	cl_uint		nitems;

	/*
	 * A typical usecase of arithmetic_stairlike_add with binary value:
	 * It takes 1 if thread wants to return a status to the host side,
	 * then stairlike-add returns a relative offset within workgroup,
	 * and we can adjust this offset by global base index.
	 */
	binary = (get_global_id(0) < kresults->nrooms &&
			  (errcode == StromError_Success ||
			   errcode == StromError_CpuReCheck)) ? 1 : 0;

	offset = arithmetic_stairlike_add(binary, workmem, &nitems);

	if (get_local_id(0) == 0)
		*p_base = atomic_add(&kresults->nitems, nitems);
	barrier(CLK_LOCAL_MEM_FENCE);
	base = *p_base;

	/*
	 * Write back the row-index that passed evaluation of the qualifier,
	 * or needs re-check on the host side. In case of re-check, row-index
	 * shall be a negative number.
	 */
	if (get_global_id(0) >= kresults->nrooms)
		return;

	if (errcode == StromError_Success)
		kresults->results[base + offset] = (get_global_id(0) + 1);
	else if (errcode == StromError_CpuReCheck)
		kresults->results[base + offset] = -(get_global_id(0) + 1);
}

/*
 * forward declaration of the function to be generated on the fly
 */
static pg_bool_t
gpuscan_qual_eval(__private cl_int *errcode,
				  __global kern_parambuf *kparams,
				  __global kern_data_store *kds,
				  __global kern_data_store *ktoast,
				  size_t kds_index);
/*
 * kernel entrypoint of gpuscan
 */
__kernel void
gpuscan_qual(__global kern_gpuscan *kgpuscan,	/* in/out */
			 __global kern_data_store *kds,		/* in */
			 __global kern_data_store *ktoast,	/* always NULL */
			 KERN_DYNAMIC_LOCAL_WORKMEM_ARG)	/* in */
{
	pg_bool_t	rc;
	cl_int		errcode = StromError_Success;
	__global kern_parambuf *kparams = KERN_GPUSCAN_PARAMBUF(kgpuscan);
	__global kern_resultbuf *kresults = KERN_GPUSCAN_RESULTBUF(kgpuscan);

	if (get_global_id(0) < kds->nitems)
		rc = gpuscan_qual_eval(&errcode, kparams, kds, ktoast,
							   get_global_id(0));
	else
		rc.isnull = true;

	STROM_SET_ERROR(&errcode,
					!rc.isnull && rc.value != 0
					? StromError_Success
					: StromError_RowFiltered);

	/* writeback error code */
	gpuscan_writeback_row_error(kresults, errcode, LOCAL_WORKMEM);
	if (!StromErrorIsSignificant(errcode))
		errcode = StromError_Success;	/* clear the minor error */
	kern_writeback_error_status(&kresults->errcode, errcode, LOCAL_WORKMEM);
}

#endif	/* OPENCL_DEVICE_CODE */
#endif	/* DEVICE_GPUSCAN_H */
