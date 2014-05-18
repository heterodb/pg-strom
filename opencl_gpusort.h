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

#ifdef OPENCL_DEVICE_CODE
/*
 * device only code below
 */


/* expected kernel prototypes */
static void
run_gpusort_single(__global kern_parambuf *kparams,
				   __global kern_column_store *kchunk,	/* in */
				   __global kern_toastbuf *ktoast,		/* in */
				   __private cl_int *errcode,			/* out */
				   __local void *local_workbuf)
{
	__global cl_int	*results = KERN_GPUSORT_RESULT_INDEX(kchunk);

	/*
	 * sort the supplied kchunk according to the supplied
	 * compare function, then it put index of sorted array
	 * on the rindex buffer.
	 * (rindex array has the least 2^N capacity larger than nrows)
	 */
}

static void
run_gpusort_multi(__global kern_parambuf *kparams,
				  __global kern_column_store *x_chunk,
				  __global kern_toastbuf     *x_toast,
				  __global kern_column_store *y_chunk,
				  __global kern_toastbuf     *y_toast,
				  __global kern_column_store *z_chunk1,
				  __global kern_toastbuf     *z_toast1,
				  __global kern_column_store *z_chunk2,
				  __global kern_toastbuf     *z_toast2,
				  __private cl_int *errcode,
				  __local void *local_workbuf)
{
	/*
	 * Run merge sort logic on the supplied x_chunk and y_chunk.
	 * Its results shall be stored into z_chunk1 and z_chunk2,
	 *
	 */
}






__kernel void
gpusort_single(__global kern_gpusort *kgsort,
			   __local void *local_workbuf)
{
	__global kern_parambuf *kparams		= KERN_GPUSORT_PARAMBUF(kgsort);
	__global kern_column_store *kchunk	= KERN_GPUSORT_CHUNK(kgsort);
	__global kern_toastbuf *ktoast		= KERN_GPUSORT_TOASTBUF(kgsort);
	__global cl_int		   *results		= KERN_GPUSORT_RESULT_INDEX(kchunk);
	cl_int		errcode = StromError_Success;

	run_gpusort_single(kparams, kchunk, ktoast, &errcode, local_workbuf);
}

__kernel void
gpusort_multi(__global kern_gpusort *kgsort_x,
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
	cl_int		errcode = StromError_Success;

	run_gpusort_multi(kparams,
					  x_chunk, x_toast,
					  y_chunk, y_toast,
					  z_chunk1, z_toast1,
					  z_chunk2, z_toast2,
					  &errcode, local_workbuf);
}

kernel void
gpusort_setup_chunk_rs(__global kern_gpusort *kgsort,
					   __global kern_row_store *krs,
					   __local void *local_workmem)
{
	/*
	 * This routine move records from usual row-store (smaller) into
	 * sorting chunk (a larger column store).
	 * Note: get_global_offset(1) shows index of row-store on host.
	 *
	 * The first column of the sorting chunk (cl_long) is identifier
	 * of individual rows on the host side. The last column of the
	 * sorting chunk (cl_uint) can be used as index of array.
	 * Usually, this index is initialized to a sequential number,
	 * then gpusort_single modifies this index array later.
	 */
}

kernel void
gpusort_setup_chunk_cs(__global kern_gpusort *kgsort,
					   __global kern_column_store *kcs,
					   __global kern_toastbuf *ktoast,
					   __local void *local_workmem)
{
	/*
	 * This routine moves records from usual column-store (smaller)
	 * into sorting chunk (a larger column store), as a preprocess
	 * of GPU sorting.
	 * Note: get_global_offset(1) shows index of row-store on host.
	 */
}

#else	/* OPENCL_DEVICE_CODE */

typedef struct
{
	pgstrom_message	msg;		/* = StromTag_GpuSort */
	Datum			dprog_key;	/* key of device program object */
	dlist_node		chain;		/* be linked to pgstrom_gpusort_multi */
	StromObject	  **rcs_slot;	/* array of underlying row/column-store */
	cl_uint			rcs_slotsz;	/* length of the array */
	cl_uint			rcs_nums;	/* current usage of the array */
	cl_uint			rcs_global_index;	/* starting offset in GpuSortState */
	kern_gpusort	kern;
} pgstrom_gpusort;

typedef struct
{
	pgstrom_message	msg;		/* = StromTag_GpuSortMulti */
	Datum			dprog_key;	/* key of device program object */
	dlist_node		chain;		/* be linked to free list */
	dlist_head		in_chunk1;	/* sorted chunks to be merged */
	dlist_head		in_chunk2;	/* sorted chunks to be merged */
	dlist_head		out_chunk;	/* merged output chunks */
	dlist_head		work_chunk;	/* working buffer during merge sort */
} pgstrom_gpusort_multi;

#define GPUSORT_MULTI_PER_BLOCK				\
	((SHMEM_BLOCKSZ - SHMEM_ALLOC_COST		\
	  - sizeof(dlist_node)) / sizeof(pgstrom_gpusort_multi))

#endif	/* !OPENCL_DEVICE_CODE */
#endif	/* OPENCL_GPUSORT_H */
