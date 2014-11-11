/*
 * opencl_gpusort.h
 *
 * Sort logic accelerated by OpenCL devices
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef OPENCL_GPUSORT_H
#define OPENCL_GPUSORT_H

/*
 * Sort acceleration using GPU/MIC devices
 *
 * Because of device memory restriction, we have implemented two different 
 * sorting logic. One is in-chunk sort using bitonic-sort, the other is
 * inter-chunk sort using merge-sort.
 * DRAM capacity of usual discrete GPU/MIC devices is much less than host
 * system (for more correctness, it depends on maximum alloc size being
 * supported by OpenCL platform), so the algorithm needs to work even if
 * only a limited portion of the data to be sorted is visible; like a window
 * towards the whole landscape.
 * Our expectation is, our supported OpenCL device can load 4-5 chunks
 * simultaneously at leas, and each chunk has 50MB-100MB capacity.
 * 
 * Preprocess
 * ----------
 * Even though a chunk has 50MB-100MB capacity, it is much larger than
 * the size of usual data unit that PG-Strom performs on. (Also, column-
 * store contains "junk" records to be filtered on scan stage. We need
 * to remove them prior to sorting),
 * So, we takes a preprocess step that construct a larger column-store
 * (here, we call it sort-chunk), prior to main sort logic. It copies
 * the contents of usual row- and column- stores into the sort-chunk,
 * and set up index array; being used in the in-chunk sorting below.
 *
 * In-chunk sorting
 * ----------------
 * Prior to inter-chunks sorting, we sort the items within a particular
 * chunk. Here is nothing difficult to do because all the items are visible
 * for a kernel invocation, thus, all we are doing is as texebook says.
 *   Bitonic-sorter
 *   http://en.wikipedia.org/wiki/Bitonic_sorter
 * Host-side kicks an OpenCL kernel with a chunk in row- or column-
 * format. Then, kernel generate an array of sorted index.
 *
 * Inter-chunk sorting
 * -------------------
 * If data set is larger than capacity of a chunk, we needs to take another
 * logic to merge preliminary sorted chunks (by bitonic-sort).
 * Because of the DRAM size restriction, all kernel can see simultaneously
 * is at most 4-5 chunks. A regular merge-sort is designed to sort two
 * preliminary sorted smaller array; usually stored in sequential devices.
 * We deal with GPU/MIC DRAM as if a small window towards whole of data set.
 * Let's assume Please assume we try to merge 
 *
 *
 */






/*
 * kern_gpusort packs three structures but not explicitly shows because of
 * variable length fields.
 * The kern_parambuf (a structure for Param/Const values) is located on
 * the head of kern_gpusort structure.
 * Then, kern_column_store should be located on the next, and
 * kern_toastbuf should be last. We allocate toastbuf anyway.
 */
typedef struct
{
	kern_parambuf		kparam;

	/*
	 * variable length fields below
	 * -----------------------------
	 * kern_column_store	kchunk
	 * cl_int				status
	 * kern_toastbuf		ktoast
	 *
	 * On gpusort_setup_chunk_(rs|cs), whole of the kern_gpusort shall
	 * be written back.
	 * On gpusort_single, result buffer (a part of kchunk) and status
	 * shall be written back.
	 * On gpusort_multi, whole of the kern_gpusort shall be written
	 * back.
	 */
} kern_gpusort;

/* macro definitions to reference packed values */
#define KERN_GPUSORT_PARAMBUF(kgpusort)						\
	((__global kern_parambuf *)(&(kgpusort)->kparam))
#define KERN_GPUSORT_PARAMBUF_LENGTH(kgpusort)				\
	STROMALIGN(KERN_GPUSORT_PARAMBUF(kgpusort)->length)

#define KERN_GPUSORT_CHUNK(kgpusort)						\
	((__global kern_column_store *)							\
	 ((__global char *)KERN_GPUSORT_PARAMBUF(kgpusort) +	\
	  KERN_GPUSORT_PARAMBUF_LENGTH(kgpusort)))
#define KERN_GPUSORT_CHUNK_LENGTH(kgpusort)					\
	STROMALIGN(KERN_GPUSORT_CHUNK(kgpusort)->length)

#define KERN_GPUSORT_STATUS(kgpusort)						\
	((__global cl_int *)									\
	 ((__global char *)KERN_GPUSORT_CHUNK(kgpusort) +		\
	  KERN_GPUSORT_CHUNK_LENGTH(kgpusort)))
#define KERN_GPUSORT_STATUS_LENGTH(kgpusort)				\
	STROMALIGN(sizeof(cl_int))

#define KERN_GPUSORT_TOASTBUF(kgpusort)						\
	((__global kern_toastbuf *)								\
	 ((__global char *)KERN_GPUSORT_STATUS(kgpusort) +		\
	  KERN_GPUSORT_STATUS_LENGTH(kgpusort)))
#define KERN_GPUSORT_TOASTBUF_LENGTH(kgpusort)				\
	STROMALIGN(KERN_GPUSORT_TOASTBUF(kgpusort)->length)

/* last column of kchunk is index array of the chunk */
#define KERN_GPUSORT_RESULT_INDEX(kchunk)					\
	((__global cl_int *)									\
	 ((__global char *)(kchunk) +							\
	  (kchunk)->colmeta[(kchunk)->ncols - 1].cs_ofs))
/* second last column of kchunk is global-rowid */
#define KERN_GPUSORT_GLOBAL_ROWID(kchunk)				\
	((__global cl_ulong *)								\
	 ((__global char *)(kchunk) +						\
	  (kchunk)->colmeta[(kchunk)->ncols - 2].cs_ofs))


#define KERN_GPUSORT_TOTAL_LENGTH(kchunk)		\
	(KERN_GPUSORT_PARAMBUF_LENGTH(kchunk) +		\
	 KERN_GPUSORT_CHUNK_LENGTH(kchunk) +		\
	 KERN_GPUSORT_STATUS_LENGTH(kchunk) +		\
	 KERN_GPUSORT_TOASTBUF_LENGTH(kchunk))

#ifdef OPENCL_DEVICE_CODE
/*
 * comparison function - to be generated by PG-Strom on the fly
 */
static cl_int gpusort_comp(__private int *errcode,
						   __global kern_column_store *kcs_x,
						   __global kern_toastbuf *ktoast_x,
						   __private cl_int x_index,
						   __global kern_column_store *kcs_y,
						   __global kern_toastbuf *ktoast_y,
						   __private cl_int y_index);







/*
 * device only code below
 */


/* expected kernel prototypes */
static void
run_gpusort_single_step(
	__global kern_parambuf *kparams,
	cl_bool reversing,					/* in */
	cl_uint unitsz,						/* in */
	__global kern_column_store *kchunk,	/* in */
	__global kern_toastbuf *ktoast,		/* in */
	__private cl_int *errcode			/* out */
	)
{
	__global cl_int	*results = KERN_GPUSORT_RESULT_INDEX(kchunk);
	cl_int	nrows			 = (kchunk)->nrows;

	/*
	 * sort the supplied kchunk according to the supplied
	 * compare function, then it put index of sorted array
	 * on the rindex buffer.
	 * (rindex array has the least 2^N capacity larger than nrows)
	 */

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

	cl_int	pos0			= results[idx0];
	cl_int	pos1			= results[idx1];
	cl_int	rv;

	rv = gpusort_comp(errcode, kchunk, ktoast, pos0, kchunk, ktoast, pos1);
	if(0 < rv)
	{
		/* Swap */
		results[idx0] = pos1;
		results[idx1] = pos0;
	}
	return;
}

/* expected kernel prototypes */
static void
run_gpusort_single_marge(
	__global kern_parambuf *kparams,
	__global kern_column_store *kchunk,	/* in */
	__global kern_toastbuf *ktoast,		/* in */
	__private cl_int *errcode,			/* out */
	__local int localIdx[] )
{
	__global cl_int	*results = KERN_GPUSORT_RESULT_INDEX(kchunk);
	cl_int			nrows	 = (kchunk)->nrows;

	/*
	 * sort the supplied kchunk according to the supplied
	 * compare function, then it put index of sorted array
	 * on the rindex buffer.
	 * (rindex array has the least 2^N capacity larger than nrows)
	 */
	
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
		localIdx[localID] = results[prtPos + localID];

    if(localSize + localID < localEntry)
		localIdx[localSize + localID] = results[prtPos + localSize + localID];

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
			cl_int rv = gpusort_comp(errcode,
									 kchunk, ktoast, pos0,
									 kchunk, ktoast, pos1);

			if(0 < rv) {
				// swap
				localIdx[idx0] = pos1;
				localIdx[idx1] = pos0;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(localID < localEntry)
		results[prtPos + localID] = localIdx[localID];

    if(localSize + localID < localEntry)
		results[prtPos + localSize + localID] = localIdx[localSize + localID];

	barrier(CLK_LOCAL_MEM_FENCE);

	return;
}

/* expected kernel prototypes */
static void
run_gpusort_single_sort(
	__global kern_parambuf *kparams,
	__global kern_column_store *kchunk,	/* in */
	__global kern_toastbuf *ktoast,		/* in */
	__private cl_int *errcode,			/* out */
	__local int localIdx[] )
{
	__global cl_int	*results = KERN_GPUSORT_RESULT_INDEX(kchunk);
	cl_int			nrows	 = (kchunk)->nrows;

	/*
	 * sort the supplied kchunk according to the supplied
	 * compare function, then it put index of sorted array
	 * on the rindex buffer.
	 * (rindex array has the least 2^N capacity larger than nrows)
	 */
	
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

    // load index to localIdx
    if(localID < localEntry)
		localIdx[localID] = results[prtPos + localID];

    if(localSize + localID < localEntry)
		localIdx[localSize + localID] = results[prtPos + localSize + localID];

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
				cl_int rv = gpusort_comp(errcode,
										 kchunk, ktoast, pos0,
										 kchunk, ktoast, pos1);

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
		results[prtPos + localID] = localIdx[localID];

    if(localSize + localID < localEntry)
		results[prtPos + localSize + localID] = localIdx[localSize + localID];

	barrier(CLK_LOCAL_MEM_FENCE);

	return;
}

#if 0
static void
gpusort_set_record(__global kern_parambuf		*kparams,
				   cl_int			 			index,
				   cl_int						N,
				   cl_int						nrowsDst0,
				   __global kern_column_store	*chunkDst0,
				   __global kern_toastbuf		*toastDst0,
				   __global kern_column_store	*chunkDst1,
				   __global kern_toastbuf		*toastDst1,
				   __global kern_column_store	*chunkSrc0,
				   __global kern_toastbuf		*toastSrc0,
				   __global kern_column_store	*chunkSrc1,
				   __global kern_toastbuf		*toastSrc1,
				   __global cl_int				*result0,
				   __global cl_int				*result1,
				   __private cl_int				*errcode,
				   __local void				 	*local_workbuf)
{
	cl_int N2			= N / 2;
	cl_int posDst		= index;
	cl_int posSrc 		= (index < N2) ? result0[index] : result1[index - N2];
	cl_int chunkPosDst	= (posDst < nrowsDst0) ? posDst : posDst - nrowsDst0;
	cl_int chunkPosSrc	= (posSrc < N2) ? posSrc : posSrc - N2;

	__global kern_column_store *chunkDst;
	__global kern_column_store *chunkSrc;
	__global kern_toastbuf     *toastDst;
	__global kern_toastbuf     *toastSrc;

	chunkDst = (posDst < nrowsDst0) ? chunkDst0 : chunkDst1;
	toastDst = (posDst < nrowsDst0) ? toastDst0 : toastDst1;
	chunkSrc = (posSrc < N2) ? chunkSrc0 : chunkSrc1;
	toastSrc = (posSrc < N2) ? toastSrc0 : toastSrc1;


	// set nrows
	if(chunkPosDst == N2 - 1  &&  posSrc < N)
		(chunkDst)->nrows = N2;

	else if(N <= posSrc)
	{
		cl_int flagLastPlus1 = true;

		if(0 < chunkPosDst)
		{
			cl_int indexPrev = index - 1;
			cl_int posPrev	 = ((indexPrev < N2)
								? result0[indexPrev]
								: result1[indexPrev - N2]);
			if(N <= posPrev)
			  flagLastPlus1 = false;
		}

		if(flagLastPlus1 == true)
			(chunkDst)->nrows = chunkPosDst;
	}


	// set index
	__global cl_int *resultDst = KERN_GPUSORT_RESULT_INDEX(chunkDst);
	resultDst[chunkPosDst]     = (posSrc < N) ? chunkPosDst : N;


	// set row data
	if(posSrc  < N)
	{
		kern_column_to_column(errcode,
							  chunkDst, toastDst, chunkPosDst,
							  chunkSrc, toastSrc, chunkPosSrc, local_workbuf);
	}

	return;
}

static void
run_gpusort_multi(__global kern_parambuf *kparams,
				  cl_bool reversing,		/* in */
				  cl_uint unitsz,			/* out */
				  __global kern_column_store *x_chunk,
				  __global kern_toastbuf     *x_toast,
				  __global kern_column_store *y_chunk,
				  __global kern_toastbuf     *y_toast,
				  __global kern_column_store *z_chunk0,
				  __global kern_toastbuf     *z_toast0,
				  __global kern_column_store *z_chunk1,
				  __global kern_toastbuf     *z_toast1,
				  __private cl_int *errcode,
				  __local void *local_workbuf)
{
	__global cl_int	*x_results = KERN_GPUSORT_RESULT_INDEX(x_chunk);
	__global cl_int	*y_results = KERN_GPUSORT_RESULT_INDEX(y_chunk);

	/*
	 * Run merge sort logic on the supplied x_chunk and y_chunk.
	 * Its results shall be stored into z_chunk0 and z_chunk1,
	 *
	 */

	cl_int	threadID		= get_global_id(0);
	cl_int  x_nrows			= (x_chunk)->nrows;
	cl_int	y_nrows			= (y_chunk)->nrows;
	cl_int	halfUnitSize	= unitsz / 2;
	cl_int	unitMask		= unitsz - 1;
	cl_int	idx0;
	cl_int	idx1;

	idx0 = (threadID / halfUnitSize) * unitsz + threadID % halfUnitSize;
	idx1 = (reversing
			? ((idx0 & ~unitMask) | (~idx0 & unitMask))
			: (idx0 + halfUnitSize));

	cl_int	N;

	for(int i=1; i<x_nrows+y_nrows; i<<=1) {
	}

	cl_int	N2	= N / 2; /* Starting index number of y_chunk */
	if(N2 <= threadID)
		return;

	/* Re-numbering the index at first times. */
	if(reversing)
	{
		if(x_nrows <= threadID)
			x_results[threadID] = N;

		y_results[idx1 - N2] = ((idx1 - N2 < y_nrows)
								? (y_results[idx1 - N2] + N2)
								: N);
	}


	__global cl_int	*result0;
	__global cl_int	*result1;

	result0 = (idx0 < N2) ? &x_results[idx0] : &y_results[idx0 - N2];
	result1 = (idx1 < N2) ? &x_results[idx1] : &y_results[idx1 - N2];

	cl_int	pos0	= *result0;
	cl_int	pos1	= *result1;

	if(N <= pos1)
	{
		/* pos1 is empry(maximum) */
	}

	else if (N <= pos0)
	{
		/* swap, pos0 is empry(maximum) */
		*result0 = pos1;
		*result1 = pos0;
	}

	else
	{
		/* sorting by data */
		__global kern_column_store	*chunk0 = (pos0 < N2) ? x_chunk : y_chunk;
		__global kern_column_store	*chunk1 = (pos1 < N2) ? x_chunk : y_chunk;
		__global kern_toastbuf		*toast0 = (pos0 < N2) ? x_toast : y_toast;
		__global kern_toastbuf		*toast1 = (pos1 < N2) ? x_toast : y_toast;
		cl_int						chkPos0 = (pos0 < N2) ? pos0 : (pos0 - N2);
		cl_int						chkPos1 = (pos1 < N2) ? pos1 : (pos1 - N2);

		cl_int rv = gpusort_comp(errcode,
								 chunk0, toast0, chkPos0,
								 chunk1, toast1, chkPos1);
		if(0 < rv)
		{
			/* swap */
			*result0 = pos1;
			*result1 = pos0;
		}
	}

	/* Update output chunk at last kernel. */
	if(unitsz == 2)
	{
		gpusort_set_record(kparams, idx0, N, N2,
						   z_chunk0, z_toast0, z_chunk1, z_toast1,
						   x_chunk, x_toast, y_chunk, y_toast,
						   x_results, y_results,
						   errcode, local_workbuf);

		gpusort_set_record(kparams, idx1, N, N2,
						   z_chunk0, z_toast0, z_chunk1, z_toast1,
						   x_chunk, x_toast, y_chunk, y_toast,
						   x_results, y_results,
						   errcode, local_workbuf);
	}

	return;
}
#endif

/*
 * gpusort_single
 *
 * Entrypoint of the kernel function for single chunk sorting. It takes
 * a gpusort-chunk, to be sorted according to gpusort_comp() being
 * generated on the fly.
 */
__kernel void
gpusort_single_step(
	cl_int bitonic_unitsz,
	__global kern_gpusort *kgsort,
	__local void *local_workbuf)
{
	__global kern_parambuf *kparams		= KERN_GPUSORT_PARAMBUF(kgsort);
	__global kern_column_store *kchunk	= KERN_GPUSORT_CHUNK(kgsort);
	__global kern_toastbuf *ktoast		= KERN_GPUSORT_TOASTBUF(kgsort);
	__global cl_int		   *kstatus		= KERN_GPUSORT_STATUS(kgsort);
	__global cl_int		   *results		= KERN_GPUSORT_RESULT_INDEX(kchunk);
	cl_bool		reversing = (bitonic_unitsz < 0 ? true : false);
	cl_uint		unitsz = (bitonic_unitsz < 0
						  ? 1U << -bitonic_unitsz
						  : 1U << bitonic_unitsz);
	cl_int		errcode = StromError_Success;

	run_gpusort_single_step(kparams, reversing, unitsz, kchunk, ktoast,
							&errcode);
	kern_writeback_error_status(kstatus, errcode, local_workbuf);
}

__kernel void
gpusort_single_marge(
	__global kern_gpusort *kgsort,
	__local void *local_workbuf)
{
	__global kern_parambuf *kparams		= KERN_GPUSORT_PARAMBUF(kgsort);
	__global kern_column_store *kchunk	= KERN_GPUSORT_CHUNK(kgsort);
	__global kern_toastbuf *ktoast		= KERN_GPUSORT_TOASTBUF(kgsort);
	__global cl_int		   *kstatus		= KERN_GPUSORT_STATUS(kgsort);
	__global cl_int		   *results		= KERN_GPUSORT_RESULT_INDEX(kchunk);
	cl_int errcode = StromError_Success;

	run_gpusort_single_marge(kparams, kchunk, ktoast, &errcode, local_workbuf);
	kern_writeback_error_status(kstatus, errcode, local_workbuf);
}

__kernel void
gpusort_single_sort(
	__global kern_gpusort *kgsort,
	__local void *local_workbuf)
{
	__global kern_parambuf *kparams		= KERN_GPUSORT_PARAMBUF(kgsort);
	__global kern_column_store *kchunk	= KERN_GPUSORT_CHUNK(kgsort);
	__global kern_toastbuf *ktoast		= KERN_GPUSORT_TOASTBUF(kgsort);
	__global cl_int		   *kstatus		= KERN_GPUSORT_STATUS(kgsort);
	__global cl_int		   *results		= KERN_GPUSORT_RESULT_INDEX(kchunk);
	cl_int errcode = StromError_Success;

	run_gpusort_single_sort(kparams, kchunk, ktoast, &errcode, local_workbuf);
	kern_writeback_error_status(kstatus, errcode, local_workbuf);
}

/*
 * gpusort_multi
 *
 * Entrypoint of the kernel function for multi chunks sorting. It takes
 * two input chunks and two output chunks. Records in the both input
 * chunks are sorted according to gpusort_comp() being generated on the
 * fly, then written to the output chunks; smaller half shall be kgsort_z1,
 * larger half shall be kgsort_z2.
 */
__kernel void
gpusort_multi(cl_int mergesort_unitsz,
			  __global kern_gpusort *kgsort_x,
			  __global kern_gpusort *kgsort_y,
			  __global kern_gpusort *kgsort_z1,
			  __global kern_gpusort *kgsort_z2,
			  __local void *local_workbuf)
{
	__global kern_parambuf *kparams		= KERN_GPUSORT_PARAMBUF(kgsort_x);
	__global kern_column_store *x_chunk = KERN_GPUSORT_CHUNK(kgsort_x);
	__global kern_column_store *y_chunk = KERN_GPUSORT_CHUNK(kgsort_y);
	__global kern_column_store *z_chunk1 = KERN_GPUSORT_CHUNK(kgsort_z1);
	__global kern_column_store *z_chunk2 = KERN_GPUSORT_CHUNK(kgsort_z2);
	__global kern_toastbuf *x_toast = KERN_GPUSORT_TOASTBUF(kgsort_x);
	__global kern_toastbuf *y_toast = KERN_GPUSORT_TOASTBUF(kgsort_y);
	__global kern_toastbuf *z_toast1 = KERN_GPUSORT_TOASTBUF(kgsort_z1);
	__global kern_toastbuf *z_toast2 = KERN_GPUSORT_TOASTBUF(kgsort_z2);
	__global cl_int		   *kstatus		= KERN_GPUSORT_STATUS(kgsort_x);
	cl_bool		reversing = (mergesort_unitsz < 0 ? true : false);
	cl_int		unitsz = (mergesort_unitsz < 0
						  ? 1U << -mergesort_unitsz
						  : 1U << mergesort_unitsz);
	cl_int		errcode = StromError_Success;

#if 0
	run_gpusort_multi(kparams,
					  reversing, unitsz,
					  x_chunk, x_toast,
					  y_chunk, y_toast,
					  z_chunk1, z_toast1,
					  z_chunk2, z_toast2,
					  &errcode, local_workbuf);
	kern_writeback_error_status(kstatus, errcode, local_workbuf);
#endif
}

/*
 * gpusort_setup_chunk_rs
 *
 * This routine move records from usual row-store (smaller) into
 * sorting chunk (a larger column store).
 *
 * The first column of the sorting chunk (cl_long) is identifier
 * of individual rows on the host side. The last column of the
 * sorting chunk (cl_uint) can be used as index of array.
 * Usually, this index is initialized to a sequential number,
 * then gpusort_single modifies this index array later.
 */
__kernel void
gpusort_setup_chunk_rs(cl_uint rcs_gindex,
					   __global kern_gpusort *kgpusort,
					   __global kern_row_store *krs,
					   cl_int	krs_nitems,
					   __local void *local_workmem)
{
	__global kern_parambuf	   *kparams = KERN_GPUSORT_PARAMBUF(kgpusort);
	__global kern_column_store *kcs = KERN_GPUSORT_CHUNK(kgpusort);
	__global kern_toastbuf	   *ktoast = KERN_GPUSORT_TOASTBUF(kgpusort);
	__global cl_int			   *kstatus = KERN_GPUSORT_STATUS(kgpusort);
	__global cl_char		   *attrefs;
	__global cl_uint		   *rindex;
	__local size_t	kcs_offset;
	__local size_t	kcs_nitems;
	size_t			kcs_index;	/* destination */
	size_t			krs_index;	/* source */
	pg_bytea_t		kparam_0;
	cl_int			errcode = StromError_Success;

	/* if number of valid items are negative, it means all the items
	 * are valid. So, no need to use rindex
	 */
	if (krs_nitems < 0)
	{
		rindex = NULL;
		krs_nitems = krs->nrows;
	}
	else
	{
		rindex = (__global cl_uint *)((__global char *)krs +
									  STROMALIGN(krs->length));
	}

	/* determine number of items to be moved */
	if (get_local_id(0) == 0)
	{
		if (get_global_id(0) + get_local_size(0) < krs_nitems)
			kcs_nitems = get_local_size(0);
		else if (get_global_id(0) < krs_nitems)
			kcs_nitems = krs_nitems - get_global_id(0);
		else
			kcs_nitems = 0;
		kcs_offset = atomic_add(&kcs->nrows, kcs_nitems);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	kcs_index = kcs_offset + get_local_id(0);

	/*
	 * fetch a valid row on the krs; pay attention row-store may contain
	 * rows being already filtered out.
	 */
	if (!rindex)
		krs_index = get_global_id(0);
	else if (get_global_id(0) < krs_nitems)
		krs_index = rindex[get_global_id(0)];
	else
		krs_index = krs->nrows;		/* dealt with invalid row */

	/* flag of referenced columns */
	kparam_0 = pg_bytea_param(kparams, &errcode, 0);

	kern_row_to_column(&errcode,
					   (__global cl_char *)VARDATA(kparam_0.value),
					   krs,
					   krs_index,
					   kcs,
					   ktoast,
					   kcs_offset,
					   kcs_nitems,
					   local_workmem);

	if (get_local_id(0) < kcs_nitems)
	{
		cl_uint		ncols = kcs->ncols;
		cl_ulong	growid = (cl_ulong)rcs_gindex << 32 | krs_index;
		__global cl_char   *addr;

		/* second last column is global row-id */
		addr = kern_get_datum(kcs, ncols - 2, kcs_index);
		*((__global cl_ulong *)addr) = growid;
		/* last column is index number within a chunk */
		addr = kern_get_datum(kcs, ncols - 1, kcs_index);
		*((__global cl_uint *)addr) = kcs_index;
	}
	kern_writeback_error_status(kstatus, errcode, local_workmem);
}

__kernel void
gpusort_setup_chunk_cs(cl_uint rcs_gindex,
					   __global kern_gpusort *kgsort,
					   __global kern_column_store *kcs_src,
					   cl_int   src_nitems,
					   __local void *local_workmem)
{
	__global kern_parambuf	   *kparams = KERN_GPUSORT_PARAMBUF(kgsort);
	__global kern_column_store *kcs_dst = KERN_GPUSORT_CHUNK(kgsort);
	__global kern_toastbuf	   *ktoast_dst = KERN_GPUSORT_TOASTBUF(kgsort);
	__global cl_int			   *kstatus = KERN_GPUSORT_STATUS(kgsort);
	__global kern_toastbuf	   *ktoast_src =
		(__global kern_toastbuf *)((__global char *)kcs_src + kcs_src->length);
	__global cl_int			   *rindex;
	__local size_t	dst_offset;
	__local size_t	dst_nitems;
	size_t			dst_index;	/* index on the destination column-store */
	size_t			src_index;	/* index on the source column-store */
	cl_uint			ncols = kcs_src->ncols;
	cl_int			errcode = StromError_Success;

	/*
	 * If number of valid rows are negative value, it means all the
	 * rows are valid, so no need to use rindex here.
	 */
	if (src_nitems < 0)
	{
		rindex = NULL;
		src_nitems = kcs_src->nrows;
	}
	else
	{
		rindex = KERN_GPUSORT_RESULT_INDEX(kcs_src);
	}

	/* determine number of items to be moved */
	if (get_local_id(0) == 0)
	{
		if (get_global_id(0) + get_local_size(0) < src_nitems)
			dst_nitems = get_local_size(0);
		else if (get_global_id(0) < src_nitems)
			dst_nitems = src_nitems - get_global_id(0);
		else
			dst_nitems = 0;
		dst_offset = atomic_add(&kcs_dst->nrows, dst_nitems);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	dst_index = dst_offset + get_local_id(0);

	/*
	 * fetch a valid row on the krs; pay attention row-store may contain
	 * rows being already filtered out.
	 */
	if (!rindex)
		src_index = get_global_id(0);
	else if (get_global_id(0) < src_nitems)
		src_index = rindex[get_global_id(0)];
	else
		src_index = kcs_src->nrows;		/* performs as an invalid row */

	/* move data from source to destination column-store */
	kern_column_to_column(&errcode,
						  ncols - 2,
						  kcs_dst,
						  ktoast_dst,
						  dst_index,
						  kcs_src,
						  ktoast_src,
						  src_index,
						  local_workmem);

	/* also, growid and rindex shall be put */
	if (get_local_id(0) < dst_nitems)
	{
		cl_uint		ncols = kcs_dst->ncols;
		cl_ulong	growid = (cl_ulong)rcs_gindex << 32 | src_index;
		__global cl_char   *addr;

		/* second last column is global row-id */
		addr = kern_get_datum(kcs_dst, ncols - 2, dst_index);
		*((__global cl_ulong *)addr) = growid;
		/* last column is index number within a chunk */
		addr = kern_get_datum(kcs_dst, ncols - 1, dst_index);
		*((__global cl_uint *)addr) = dst_index;
	}
	kern_writeback_error_status(kstatus, errcode, local_workmem);
}

#else	/* OPENCL_DEVICE_CODE */

typedef struct
{
	dlist_node		chain;		/* to be linked to pgstrom_gpusort */
	cl_uint			scan_pos;	/* current scan position, after the sorting */
	struct {
		StromObject	   *rcstore;
		cl_int			nitems;
		cl_int		   *rindex;	/* non-null, if rindex is available */
	} rcs_slot[400];	/* <- max number of r/c stores per chunk */
	cl_uint			rcs_nums;	/* number of r/c stores associated */
	cl_uint			rcs_head;	/* head index of rcs array in GpuSortstate */
	kern_gpusort	kern;
} pgstrom_gpusort_chunk;

typedef struct
{
	pgstrom_message	msg;		/* = StromTag_GpuSort */
	Datum			dprog_key;	/* key of device program object */
	dlist_node		chain;		/* be linked to free list */
	bool			has_rindex;	/* true. if some rows may not be valid */
	bool			is_sorted;	/* true, if already sorted */
	dlist_head		gs_chunks;	/* chunked being sorted, or to be sorted */
} pgstrom_gpusort;

#endif	/* !OPENCL_DEVICE_CODE */
#endif	/* OPENCL_GPUSORT_H */
