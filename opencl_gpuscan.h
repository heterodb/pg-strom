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
 * Gpuscan kernel code assumes all the fields shall be initialized to zero.
 */
typedef struct {
	cl_uint		errcode;		/* statement level errorcode */
	cl_int		nrows;
	cl_int		results[FLEXIBLE_ARRAY_MEMBER];
} kern_gpuscan;

#ifdef OPENCL_DEVICE_CODE
/* macro for error setting */
#define STROM_SET_ERROR(errcode)	gpuscan_set_error(errcode)

/*
 * Usage of local memory on gpuscan logic.
 * 
 * Gpuscan requires to allocate 2 * sizeof(cl_int) * get_local_size(0) length
 * of local memory on its invocation, to handle tuple's visibility and error
 * status during evaluation of qualifiers.
 */
static __local cl_int  *kern_local_error;
static __local cl_int  *kern_local_error_work;

static inline void
gpuscan_local_init(__local cl_int *karg_local_buffer)
{
	kern_local_error = karg_local_buffer;
	kern_local_error_work = kern_local_error + get_local_size(0);
	kern_local_error[get_local_id(0)] = StromError_Success;
}

/*
 * It sets an error code unless no significant error code is already set.
 * Also, RowReCheck has higher priority than RowFiltered because RowReCheck
 * implies device cannot run the given expression completely.
 * (Usually, due to compressed or external varlena datum)
 */
static inline void
gpuscan_set_error(cl_int errcode)
{
	cl_int	oldcode = kern_local_error[get_local_id(0)];

	if (StromErrorIsSignificant(errcode))
	{
		if (!StromErrorIsSignificant(oldcode))
			kern_local_error[get_local_id(0)] = errcode;
	}
	else if (errcode > oldcode)
		kern_local_error[get_local_id(0)] = errcode;
}

/*
 * Get an error code to be returned in statement level
 */
static void
gpuscan_writeback_statement_error(__global kern_gpuscan *gpuscan)
{
	cl_uint		wkgrp_sz = get_local_size(0);
	cl_uint		wkgrp_id = get_local_id(0);
	cl_int		i = 0;

	kern_local_error_work[wkgrp_id] = kern_local_error[wkgrp_id];
	while (wkgrp_sz != 0)
	{
		if (wkgrp_id & ((1<<(i+1))-1) == 0)
		{
			cl_int	errcode1 = kern_local_error_work[wkgrp_id];
			cl_int	errcode2 = kern_local_error_work[wkgrp_id + (1<<i)];

			if (!StromErrorIsStmtLevel(errcode1) &&
				StromErrorIsStmtLevel(errcode2))
				kern_local_error_work[wkgrp_id] = errcode2;
		}
		wkgrp_sz >>= 1;
		i++;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/*
	 * It writes back a statement level error, unless no other workgroup
	 * put a significant statement-level error.
	 * This atomic operation set an error code, if it is still
	 * StromError_Success.
	 */
	i = kern_local_error_work[0];
	if (get_local_id(0) == 0 && StromErrorIsSignificant(i))
		atomic_cmpxchg(&gpuscan->errcode, StromError_Success, i);

	return kern_local_error_work[0];
}

/*
 * gpuscan_writeback_row_error
 *
 * It writes back the calculation result of gpuscan.
 */
static void
gpuscan_writeback_row_error(__global kern_gpuscan *gpuscan)
{
	cl_uint		wkgrp_sz = get_local_size(0);
	cl_uint		wkgrp_id = get_local_id(0);
	cl_uint		offset;
	cl_uint		nrooms;
	cl_int		i = 0;

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
	kern_local_error_work[wkgrp_id]
		= (kern_local_error[wkgrp_id] == StromError_Success ||
		   kern_local_error[wkgrp_id] == StromError_RowReCheck ? 1 : 0);

	while (wkgrp_sz != 0)
	{
		if (wkgrp_id & (1 << i) != 0)
		{
			cl_int	i_source = (wkgrp_id & ~(1 << i)) | ((1 << i) - 1);

			kern_local_error_work[wkgrp_id] += kern_local_error_work[i_source];
		}
		wkgrp_sz >>= 1;
		i++;
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
	nrooms = kern_local_error_work[get_local_size(0) - 1];
	offset = atomic_add(&gpuscan->nrows, nrooms);

	/*
	 * Write back the row-index that passed evaluation of the qualifier,
	 * or needs re-check on the host side. In case of re-check, row-index
	 * shall be a negative number.
	 */
	if (kern_local_error[wkgrp_id] == StromError_Success)
	{
		i = kern_local_error_work[wkgrp_id];
		result[offset + i - 1] = (get_global_id(0) + 1);
	}
	else if (kern_local_error[wkgrp_id] == StromError_RowReCheck)
	{
		i = kern_local_error_work[wkgrp_id];
		results[offset + i - 1] = -(get_global_id(0) + 1);
	}
}

static inline void
gpuscan_writeback_result(__global kern_gpuscan *gpuscan)
{
	gpuscan_writeback_statement_error(gpuscan);
	gpuscan_writeback_row_error(gpuscan);
}

#else	/* OPENCL_DEVICE_CODE */

/*
 * Host side representation of kern_gpuscan. It has a program-id to be
 * executed on the OpenCL device, and either of row- or column- store
 * to be processed, in addition to the kern_gpuscan buffer.
 */
typedef struct {
	MessageTag		type;	/* StromMsg_GpuScan */
	Datum			program_id;
	union {
		MessageTag			   *head;
		pgstrom_row_store	   *rs;
		pgstrom_column_store   *cs;
	} store;
	kern_gpuscan	kern;
} pgstrom_gpuscan;

#endif	/* OPENCL_DEVICE_CODE */
#endif	/* OPENCL_GPUSCAN_H */
