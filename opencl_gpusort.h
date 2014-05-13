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
 * Layout of kern_parambuf
 *
 *
 * +----------------+    -----
 * | kern_parambuf  |      ^
 * | +--------------+      |
 * | | length   o-------+  | GpuSort will store its results on the
 * | +--------------+   |  | kern_column_store unlike GpuScan, so
 * | | nparams      |   |  | it does not have kern_resultbuf here.
 * | +--------------+   |  | 
 * | | poffset[0]   |   |  | 
 * | | poffset[1]   |   |  | 
 * | |    :         |   |  | 
 * | | poffset[M-1] |   |  | 
 * | +--------------+   |  | Area to be sent to OpenCL device.
 * | | variable     |   |  | Forward DMA shall be issued here.
 * | | length field |   |  | 
 * | | for Param /  |   |  | 
 * | | Const values |   |  | 
 * | |     :        |   V  V 
 * +----------------+ -------
 * |                |   |
 * /  Debug Buffer  /   |  Area to be written back from OpenCL device.
 * /   (If used)    /   |  Reverse DMA shall be issued here.
 * | |              |   V
 * +-+--------------+ -----
 */
typedef struct
{
	kern_parambuf	kparam;


} kern_gpusort;




#if 0
/* expected kernel prototypes */
static void
gpusort_bitonic_single(__global kern_parambuf *kparams,
					   __global kern_column_store *kcs,
					   __global kern_toastbuf *toast,
					   __private cl_int *errcode,
					   __global cl_int *results,
					   __lobal void *workbuf)
{

	
}

__kernel void
gpusort_bitonic_cs(__global kern_gpusort *kgsort,
				   __global kern_column_store *kcs,
				   __global kern_toastbuf *toast,
				   __lobal void *workbuf)
{
	cl_int		errcode = StromError_Success;
	__global kern_parambuf *kparams
		= KERN_GPUSORT_PARAMBUF(kgsort);
	__global cl_uint   *results
		= KERN_COLSTORE_VALUES(kcs->ncols);

	gpusort_bitonic_single(kparams, kcs, toast, &errcode, results, workbuf);
}








#endif


#ifdef OPENCL_DEVICE_CODE

#else	/* OPENCL_DEVICE_CODE */


typedef struct
{
	pgstrom_message	msg;	/* = StromTag_GpuSort */
	Datum			dprog_key;
	
	kern_gpusort	kern;
} pgstrom_gpusort;

#endif	/* !OPENCL_DEVICE_CODE */
#endif	/* OPENCL_GPUSORT_H */
