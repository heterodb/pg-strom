/*
 * opencl_gpuscan.h
 *
 * OpenCL device code specific to GpuScan logic
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef OPENCL_GPUSCAN_H
#define OPENCL_GPUSCAN_H

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
 * | +--------------+         |      | kern_resultbuf is located just after
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
 * | | nrooms       |    |    |  Forward DMA shall be issued here.
 * | +--------------+    |    |
 * | | nitems       |    |    |
 * | +--------------+    |    |
 * | | debug_usage  |    |    |
 * | +--------------+    |    |
 * | | errcode      |    |    V
 * | +--------------+    |  -----
 * | | results[0]   |    |
 * | | results[1]   |    |  Area to be written back from OpenCL device.
 * | |     :        |    |  Reverse DMA shall be issued here.
 * | | results[N-1] |    V
 * | +--------------+  --+--
 * | | debug buffer |    |
 * | /  (if used)   /    |
 * | /              /    |
 * | |              |    V
 * +-+--------------+  -----
 *
 * Gpuscan kernel code assumes all the fields shall be initialized to zero.
 */
typedef struct {
	kern_parambuf	kparam;
	/*
	 * as above, kern_resultbuf shall be located next to the parambuf
	 */
} kern_gpuscan;

#define KERN_GPUSCAN_PARAMBUF(kgscan)			\
	((__global kern_parambuf *)(&(kgscan)->kparam))
#define KERN_GPUSCAN_RESULTBUF(kgscan)			\
	((__global kern_resultbuf *)((char *)(kgscan) + (kgscan)->kparam.length))
#define KERN_GPUSCAN_LENGTH(kgscan)										\
	(offsetof(kern_gpuscan, kparam) +									\
	 (kgscan)->kparam.length +											\
	 offsetof(kern_resultbuf,											\
			  results[KERN_GPUSCAN_RESULTBUF(kgscan)->nrooms]) +		\
	 (KERN_GPUSCAN_RESULTBUF(kgscan)->debug_usage == KERN_DEBUG_UNAVAILABLE ? \
	  0 : KERNEL_DEBUG_BUFSIZE))
#define KERN_GPUSCAN_DMA_SENDLEN(kgscan)		\
	((kgscan)->kparam.length +					\
	 offsetof(kern_resultbuf, results[0]))

#define KERN_GPUSCAN_DMA_RECVLEN(kgscan)								\
	(offsetof(kern_resultbuf,											\
			  results[KERN_GPUSCAN_RESULTBUF(kgscan)->nrooms]) +		\
	 (KERN_GPUSCAN_RESULTBUF(kgscan)->debug_usage == KERN_DEBUG_UNAVAILABLE ? \
	  0 : KERNEL_DEBUG_BUFSIZE))

#ifdef OPENCL_DEVICE_CODE
/* macro for error setting */
#define STROM_SET_ERROR(errcode)				\
	gpuscan_set_error(errcode,local_workmem)

/*
 * Usage of local memory on gpuscan logic.
 * 
 * Gpuscan requires to allocate 2 * sizeof(cl_int) * get_local_size(0) length
 * of local memory on its invocation, to handle tuple's visibility and error
 * status during evaluation of qualifiers.
 */
static inline void
gpuscan_workmem_init(__local void *workmem)
{
	__local cl_int *local_error = workmem;

	local_error[get_local_id(0)] = StromError_Success;
	barrier(CLK_LOCAL_MEM_FENCE);
}

/*
 * It sets an error code unless no significant error code is already set.
 * Also, RowReCheck has higher priority than RowFiltered because RowReCheck
 * implies device cannot run the given expression completely.
 * (Usually, due to compressed or external varlena datum)
 */
static inline void
gpuscan_set_error(cl_int errcode, __local void *workmem)
{
	__local cl_int *local_error = workmem;
	cl_int	oldcode = local_error[get_local_id(0)];

	if (StromErrorIsSignificant(errcode))
	{
		if (!StromErrorIsSignificant(oldcode))
			local_error[get_local_id(0)] = errcode;
	}
	else if (errcode > oldcode)
		local_error[get_local_id(0)] = errcode;
}

/*
 * Get an error code to be returned in statement level
 */
static void
gpuscan_writeback_statement_error(__global kern_resultbuf *kresbuf,
								  __local void *workmem)
{
	__local cl_int *local_error = workmem;
	__local cl_int *local_temp = local_error + get_local_size(0);
	cl_uint		wkgrp_id = get_local_id(0);
	cl_uint		wkgrp_sz;
	cl_int		errcode1;
	cl_int		errcode2;
	cl_int		i = 0;

	local_temp[wkgrp_id] = local_error[wkgrp_id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (i=0, wkgrp_sz = get_local_size(0) - 1;
		 wkgrp_sz != 0;
		 i++, wkgrp_sz >>= 1)
	{
		/* if least (i+1) bits of this wkgrp_id are zero? */
		if ((wkgrp_id & ((1<<(i+1))-1)) == 0)
		{
			errcode1 = local_temp[wkgrp_id];
			errcode2 = (wkgrp_id + (1<<i) < get_local_size(0)
						? local_temp[wkgrp_id + (1<<i)]
						: StromError_Success);

			if (!StromErrorIsSignificant(errcode1) &&
				StromErrorIsSignificant(errcode2))
				local_temp[wkgrp_id] = errcode2;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/*
	 * It writes back a statement level error, unless no other workgroup
	 * put a significant statement-level error.
	 * This atomic operation set an error code, if it is still
	 * StromError_Success.
	 */
	errcode1 = local_temp[0];
	if (get_local_id(0) == 0 && StromErrorIsSignificant(errcode1))
	{
		KDEBUG_INT("errcode", errcode1);
		atomic_cmpxchg(&kresbuf->errcode, StromError_Success, errcode1);
	}
}

/*
 * gpuscan_writeback_row_error
 *
 * It writes back the calculation result of gpuscan.
 */
static void
gpuscan_writeback_row_error(__global kern_resultbuf *kresbuf,
							__local void *workmem)
{
	__local cl_int *local_error = workmem;
	__local cl_int *local_temp = local_error + get_local_size(0);
	cl_uint		wkgrp_sz = get_local_size(0);
	cl_uint		wkgrp_id = get_local_id(0);
	cl_uint		offset;
	cl_uint		nitems;
	cl_uint		i;

	/*
	 * NOTE: At the begining, kern_local_error_work has either 1 or 0
	 * according to the row-level error code. This logic tries to count
	 * number of elements with 1,
	 * example)
	 * X[0] - 1 -> 1 (X[0])      -> 1 (X[0])   -> 1 (X[0])   -> 1 *
	 * X[1] - 0 -> 1 (X[0]+X[1]) -> 1 (X[0-1]) -> 1 (X[0-1]) -> 1
	 * X[2] - 0 -> 0 (X[2])      -> 1 (X[0-2]) -> 1 (X[0-2]) -> 1
	 * X[3] - 1 -> 1 (X[2]+X[3]) -> 2 (X[0-3]) -> 2 (X[0-3]) -> 2 *
	 * X[4] - 0 -> 0 (X[4])      -> 0 (X[4])   -> 2 (X[0-4]) -> 2
	 * X[5] - 0 -> 0 (X[4]+X[5]) -> 0 (X[4-5]) -> 2 (X[0-5]) -> 2
	 * X[6] - 1 -> 1 (X[6])      -> 1 (X[4-6]) -> 3 (X[0-6]) -> 3 *
	 * X[7] - 1 -> 2 (X[6]+X[7]) -> 2 (X[4-7]) -> 4 (X[0-7]) -> 4 *
	 * X[8] - 0 -> 0 (X[8])      -> 0 (X[7])   -> 0 (X[7])   -> 4
	 * X[9] - 1 -> 1 (X[8]+X[9]) -> 1 (X[7-8]) -> 1 (X7-8])  -> 5 *
	 */
	local_temp[wkgrp_id]
		= (get_global_id(0) < kresbuf->nrooms &&
		   (local_error[wkgrp_id] == StromError_Success ||
			local_error[wkgrp_id] == StromError_RowReCheck)) ? 1 : 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	for (i=0; wkgrp_sz != 0; i++, wkgrp_sz >>= 1)
	{
		if ((wkgrp_id & (1 << i)) != 0)
		{
			cl_int	i_source = (wkgrp_id & ~(1 << i)) | ((1 << i) - 1);

			local_temp[wkgrp_id] += local_temp[i_source];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/*
	 * After the loop, each entry of kern_local_error_work[] shall have
	 * sum(0..i) of kern_local_error[]; that means index of the item that
	 * has Success or ReCheck status. It also means tha last item of
	 * kern_local_error_work[] array is total number of items to be written
	 * back (because it it sum(0..N-1)), so we acquire this number of rooms
	 * with atomic operation, then write them back.
	 */
	nitems = local_temp[get_local_size(0) - 1];

	barrier(CLK_LOCAL_MEM_FENCE);
	if (get_local_id(0) == 0)
	{
		offset = atomic_add(&kresbuf->nitems, nitems);
		for (i=0; i < get_local_size(0); i++)
			local_temp[i] += offset;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	/*
	 * Write back the row-index that passed evaluation of the qualifier,
	 * or needs re-check on the host side. In case of re-check, row-index
	 * shall be a negative number.
	 */
	if (get_global_id(0) >= kresbuf->nrooms)
		return;

	if (local_error[wkgrp_id] == StromError_Success)
	{
		i = local_temp[wkgrp_id];
		kresbuf->results[i - 1] = (get_global_id(0) + 1);
	}
	else if (local_error[wkgrp_id] == StromError_RowReCheck)
	{
		i = local_temp[wkgrp_id];
		kresbuf->results[i - 1] = -(get_global_id(0) + 1);
	}
}

static inline void
gpuscan_writeback_result(__global kern_gpuscan *kgpuscan,
						 __local void *local_workmem)
{
	__global kern_resultbuf *kresbuf = KERN_GPUSCAN_RESULTBUF(kgpuscan);

	gpuscan_writeback_statement_error(kresbuf, local_workmem);
	gpuscan_writeback_row_error(kresbuf, local_workmem);
}

#else	/* OPENCL_DEVICE_CODE */

/*
 * Host side representation of kern_gpuscan. It has a program-id to be
 * executed on the OpenCL device, and either of row- or column- store
 * to be processed, in addition to the kern_gpuscan buffer including
 * kern_parambuf for constant values.
 */
typedef struct {
	pgstrom_message	msg;	/* = StromTag_GpuScan */
	Datum			dprog_key;	/* key of device program */
	dlist_head		rc_store;	/* a row- or column store */
	kern_gpuscan	kern;
} pgstrom_gpuscan;

#endif	/* OPENCL_DEVICE_CODE */
#endif	/* OPENCL_GPUSCAN_H */
