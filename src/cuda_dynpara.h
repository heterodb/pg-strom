/*
 * cuda_dynpara.h
 *
 * Support routine for dynamic parallelism on CUDA devices.
 * Inclusion of this library makes cuda_program.c to link libcudadevrt.a.
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
#ifndef CUDA_DYNPARA_H
#define CUDA_DYNPARA_H
#ifdef __CUDACC__
#include <device_launch_parameters.h>
#define WORKGROUPSIZE_RESULT_TYPE		cudaError_t
#define WORKGROUPSIZE_RESULT_SUCCESS	cudaSuccess
#define WORKGROUPSIZE_RESULT_EINVAL		cudaErrorInvalidValue
#else
#define WORKGROUPSIZE_RESULT_TYPE		CUresult
#define WORKGROUPSIZE_RESULT_SUCCESS	CUDA_SUCCESS
#define WORKGROUPSIZE_RESULT_EINVAL		CUDA_ERROR_INVALID_VALUE
#endif

#ifdef __CUDACC__
/*
 * Macro to track timeval and increment usage.
 */
#define TIMEVAL_RECORD(ktask,field,tv1,tv2,smx_clock)				\
	do {															\
		(ktask)->num_##field++;										\
		(ktask)->tv_##field += (((cl_float)(((tv2) - (tv1)))) /		\
								 ((cl_float)((smx_clock) * 1000)));	\
	} while(0)
#endif

/*
 * __pgstrom_optimal_workgroup_size
 *
 * Logic is equivalent to cudaOccupancyMaxPotentialBlockSize(),
 * but is not supported by the device runtime at CUDA-7.5.
 * So, we ported the same calculation logic here.
 */
STATIC_INLINE(WORKGROUPSIZE_RESULT_TYPE)
__pgstrom_optimal_workgroup_size(cl_uint *p_grid_sz,
								 cl_uint *p_block_sz,
								 cl_uint nitems,
#ifdef __CUDACC__
								 const void *kernel_func,
#else
								 CUfunction kernel_func,
#endif
								 cl_int funcMaxThreadsPerBlock,
                                 cl_uint staticShmemSize,
                                 cl_uint dynamicShmemPerThread,
								 cl_int warpSize,
								 cl_int devMaxThreadsPerBlock,
								 cl_int maxThreadsPerMultiProcessor,
								 cl_int flags)
{
	WORKGROUPSIZE_RESULT_TYPE status;
	/* Limits */
	cl_int		blockSizeLimit = (nitems > 0 ? nitems : 1);
	cl_int		occupancyLimit = maxThreadsPerMultiProcessor;

    /* Recorded maximum */
	cl_int		maxBlockSize = 0;
	cl_int		maxOccupancy = 0;

    /* Temporary */
	cl_int		blockSizeToTryAligned;
	cl_int		blockSizeToTry;
	cl_int		blockSizeLimitAligned;
	cl_int		occupancyInBlocks;
	cl_int		occupancyInThreads;

	/*
	 * Try each block size, and pick the block size with maximum occupancy
	 */
	if (devMaxThreadsPerBlock < blockSizeLimit)
		blockSizeLimit = devMaxThreadsPerBlock;

	if (funcMaxThreadsPerBlock < blockSizeLimit)
		blockSizeLimit = funcMaxThreadsPerBlock;

	blockSizeLimitAligned =
		((blockSizeLimit + (warpSize - 1)) / warpSize) * warpSize;

	for (blockSizeToTryAligned = blockSizeLimitAligned;
		 blockSizeToTryAligned > 0;
		 blockSizeToTryAligned -= warpSize)
	{
		/*
		 * This is needed for the first iteration, because
		 * blockSizeLimitAligned could be greater than blockSizeLimit
		 */
		if (blockSizeLimit < blockSizeToTryAligned)
			blockSizeToTry = blockSizeLimit;
		else
			blockSizeToTry = blockSizeToTryAligned;
#ifdef __CUDACC__
		status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
			&occupancyInBlocks,
			kernel_func,
			blockSizeToTry,
			dynamicShmemPerThread * blockSizeToTry,
			flags);
		if (status != cudaSuccess)
			return status;
#else
		status = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
			&occupancyInBlocks,
			kernel_func,
			blockSizeToTry,
			dynamicShmemPerThread * blockSizeToTry,
			flags);
		if (status != CUDA_SUCCESS)
			return status;
#endif
		occupancyInThreads = blockSizeToTry * occupancyInBlocks;

		if (occupancyInThreads > maxOccupancy)
		{
			maxBlockSize = blockSizeToTry;
			maxOccupancy = occupancyInThreads;
		}

        /* Early out if we have reached the maximum */
		if (occupancyLimit == maxOccupancy)
			break;
	}

	/*
	 * Return best availability
	 */
	*p_grid_sz = Max((nitems + maxBlockSize - 1) / maxBlockSize, 1);
	*p_block_sz = Max(maxBlockSize, 1);

	return WORKGROUPSIZE_RESULT_SUCCESS;
}

#ifdef __CUDACC__
/*
 *
 */
STATIC_FUNCTION(cudaError_t)
pgstrom_optimal_workgroup_size(dim3 *p_grid_sz,
                               dim3 *p_block_sz,
							   const void *kernel_func,
                               size_t nitems,
                               size_t dynamic_shmem_per_thread)
{
	cudaError_t	status;
	/* Device and function properties */
	cudaFuncAttributes attrs;
	int			device;
	int			maxThreadsPerMultiProcessor;
	int			warpSize;
	int			devMaxThreadsPerBlock;

	/* Sanity checks */
	if (!p_grid_sz || !p_block_sz || !kernel_func)
		return cudaErrorInvalidValue;

	/* Obtain device and function properties */
	status = cudaGetDevice(&device);
	if (status != cudaSuccess)
		return status;

	status = cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor,
									cudaDevAttrMaxThreadsPerMultiProcessor,
									device);
	if (status != cudaSuccess)
		return status;

	status = cudaDeviceGetAttribute(&warpSize,
									cudaDevAttrWarpSize,
									device);
	if (status != cudaSuccess)
		return status;

	status = cudaDeviceGetAttribute(&devMaxThreadsPerBlock,
									cudaDevAttrMaxThreadsPerBlock,
									device);
	if (status != cudaSuccess)
		return status;

	status = cudaFuncGetAttributes(&attrs, kernel_func);
	if (status != cudaSuccess)
		return status;

	/*
	 * Estimate an optimal workgroup size
	 */
	status = __pgstrom_optimal_workgroup_size(&p_grid_sz->x,
											  &p_block_sz->x,
											  nitems,
											  kernel_func,
											  attrs.maxThreadsPerBlock,
											  attrs.sharedSizeBytes,
											  dynamic_shmem_per_thread,
											  warpSize,
											  devMaxThreadsPerBlock,
											  maxThreadsPerMultiProcessor,
											  0);
	if (status != cudaSuccess)
		return status;

	p_grid_sz->y = 1;
	p_grid_sz->z = 1;
	p_block_sz->y = 1;
	p_block_sz->z = 1;

	return cudaSuccess;
}
#endif	/* __CADACC__ */

STATIC_INLINE(WORKGROUPSIZE_RESULT_TYPE)
__pgstrom_largest_workgroup_size(cl_uint *p_grid_sz,
								 cl_uint *p_block_sz,
								 cl_uint nitems,
								 cl_uint kernel_max_blocksz,
								 cl_uint static_shmem_sz,
								 cl_uint dynamic_shmem_per_thread,
								 cl_uint warp_size,
								 cl_uint max_shmem_per_block)
{
	cl_uint		block_size = kernel_max_blocksz;

	/* special case if nitems == 0 */
	if (nitems == 0)
		nitems = 1;
	if (nitems < block_size)
		block_size = (nitems + warp_size - 1) & ~(warp_size - 1);
	if (static_shmem_sz +
		dynamic_shmem_per_thread * block_size > max_shmem_per_block)
	{
		block_size = (max_shmem_per_block -
					  static_shmem_sz) / dynamic_shmem_per_thread;
		block_size &= ~(warp_size - 1);

		if (block_size < warp_size)
			return WORKGROUPSIZE_RESULT_EINVAL;
	}
	*p_block_sz = block_size;
	*p_grid_sz = (nitems + block_size - 1) / block_size;

	return WORKGROUPSIZE_RESULT_SUCCESS;
}

#ifdef __CUDACC__
STATIC_FUNCTION(cudaError_t)
pgstrom_largest_workgroup_size(dim3 *p_grid_sz,
							   dim3 *p_block_sz,
							   const void *kernel_func,
							   size_t nitems,
							   size_t dynamic_shmem_per_thread)
{
	cudaError_t	status;
	cl_int		device;
	cl_int		warp_size;
	cl_int		max_shmem_per_block;
	cudaFuncAttributes attrs;

	/* Sanity checks */
	if (!p_grid_sz || !p_block_sz || !kernel_func)
		return cudaErrorInvalidValue;

	/* Get device and function's attribute */
	status = cudaGetDevice(&device);
	if (status != cudaSuccess)
		return status;

	status = cudaDeviceGetAttribute(&warp_size,
									cudaDevAttrWarpSize,
									device);
	if (status != cudaSuccess)
		return status;

	status = cudaDeviceGetAttribute(&max_shmem_per_block,
									cudaDevAttrMaxSharedMemoryPerBlock,
									device);
	if (status != cudaSuccess)
		return status;

	status = cudaFuncGetAttributes(&attrs, kernel_func);
	if (status != cudaSuccess)
		return status;

	status = __pgstrom_largest_workgroup_size(&p_grid_sz->x,
											  &p_block_sz->x,
											  nitems,
											  attrs.maxThreadsPerBlock,
											  attrs.sharedSizeBytes,
											  dynamic_shmem_per_thread,
											  warp_size,
											  max_shmem_per_block);
	if (status != cudaSuccess)
		return status;

	p_grid_sz->y = 1;
	p_grid_sz->z = 1;
	p_block_sz->y = 1;
	p_block_sz->z = 1;

	return WORKGROUPSIZE_RESULT_SUCCESS;
}
#endif	/* __CUDACC__ */

STATIC_INLINE(WORKGROUPSIZE_RESULT_TYPE)
__pgstrom_largest_workgroup_size_2d(cl_uint *p_grid_xsize,
									cl_uint *p_grid_ysize,
									cl_uint *p_block_xsize,
									cl_uint *p_block_ysize,
									cl_uint x_nitems,
									cl_uint y_nitems,
									cl_uint kernel_max_blocksz,
									cl_uint static_shmem_sz,
									cl_uint dynamic_shmem_per_xitem,
									cl_uint dynamic_shmem_per_yitem,
									cl_uint dynamic_shmem_per_thread,
									cl_uint warp_size,
									cl_uint max_shmem_per_block)
{
	cl_uint		block_total_size = kernel_max_blocksz;
	cl_uint		block_xsize;
	cl_uint		block_ysize;

	/* Special case if nitems == 0 */
	if (x_nitems == 0)
		x_nitems = 1;
	if (y_nitems == 0)
		y_nitems = 1;
	/* adjust entire block size */
	if (x_nitems * y_nitems < block_total_size)
        block_total_size = (x_nitems * y_nitems +
							(warp_size - 1)) & ~(warp_size - 1);
	/*
	 * reduction of block size according to the shared memory consumption
	 * per thread basis
	 */
	if (static_shmem_sz +
		block_total_size * dynamic_shmem_per_thread > max_shmem_per_block)
	{
		block_total_size = ((max_shmem_per_block -
							 static_shmem_sz) / dynamic_shmem_per_thread);
		block_total_size &= ~(warp_size - 1);

		if (block_total_size < warp_size)
			return WORKGROUPSIZE_RESULT_EINVAL;
	}

	/*
	 * adjust block_xsize and _ysize according to the expected shared memory
	 * consumption, and scale of nitems
	 */
	block_xsize = warp_size;
	block_ysize = (block_total_size / block_xsize);
	if (block_ysize < y_nitems)
	{
		block_ysize = y_nitems;
		block_xsize = (block_total_size / block_ysize) & ~(warp_size - 1);
	}

	while (static_shmem_sz +
		   block_xsize * dynamic_shmem_per_xitem +
		   block_ysize * dynamic_shmem_per_yitem > max_shmem_per_block)
	{
		if (block_ysize == 1 ||
			(block_xsize - warp_size) * dynamic_shmem_per_xitem >
			(block_ysize - 1) * dynamic_shmem_per_yitem)
		{
			if (block_xsize <= warp_size)
				return WORKGROUPSIZE_RESULT_EINVAL;
			block_xsize -= warp_size;
			block_ysize = ((max_shmem_per_block - static_shmem_sz -
							block_xsize * dynamic_shmem_per_xitem)
						   / dynamic_shmem_per_yitem);
		}
		else
		{
			if (block_ysize <= 1)
				return WORKGROUPSIZE_RESULT_EINVAL;
			block_ysize--;
			block_xsize = ((max_shmem_per_block - static_shmem_sz -
							block_ysize * dynamic_shmem_per_yitem)
						   / dynamic_shmem_per_yitem) & ~(warp_size - 1);
		}
	}
	/* results */
	*p_grid_xsize = (x_nitems + block_xsize - 1) / block_xsize;
	*p_grid_ysize = (y_nitems + block_ysize - 1) / block_ysize;
	*p_block_xsize = block_xsize;
	*p_block_ysize = block_ysize;

	return WORKGROUPSIZE_RESULT_SUCCESS;
}

#ifdef __CUDACC__
STATIC_FUNCTION(cudaError_t)
pgstrom_largest_workgroup_size_2d(dim3 *p_grid_sz,
								  dim3 *p_block_sz,
								  const void *kernel_func,
								  cl_uint x_nitems,
								  cl_uint y_nitems,
								  cl_uint dynamic_shmem_per_xitem,
								  cl_uint dynamic_shmem_per_yitem,
								  cl_uint dynamic_shmem_per_thread)
{
	cudaError_t	status;
	cl_int		device;
	cl_int		warp_size;
	cl_int		max_shmem_per_block;
	cudaFuncAttributes attrs;

	/* Sanity checks */
	if (!p_grid_sz || !p_block_sz || !kernel_func)
		return cudaErrorInvalidValue;

	/* Get device and function's attribute */
	status = cudaGetDevice(&device);
	if (status != cudaSuccess)
		return status;

	status = cudaDeviceGetAttribute(&warp_size,
									cudaDevAttrWarpSize,
									device);
	if (status != cudaSuccess)
		return status;

	status = cudaDeviceGetAttribute(&max_shmem_per_block,
									cudaDevAttrMaxSharedMemoryPerBlock,
									device);
	if (status != cudaSuccess)
		return status;

	status = cudaFuncGetAttributes(&attrs, kernel_func);
	if (status != cudaSuccess)
		return status;

	status = __pgstrom_largest_workgroup_size_2d(&p_grid_sz->x,
												 &p_grid_sz->y,
												 &p_block_sz->x,
												 &p_block_sz->y,
												 x_nitems,
												 y_nitems,
												 attrs.maxThreadsPerBlock,
												 attrs.sharedSizeBytes,
												 dynamic_shmem_per_xitem,
												 dynamic_shmem_per_yitem,
												 dynamic_shmem_per_thread,
												 warp_size,
												 max_shmem_per_block);
	if (status != cudaSuccess)
		return status;

	p_grid_sz->z = 1;
	p_block_sz->z = 1;

	return cudaSuccess;
}
#endif	/* __CUDACC__ */
#undef WORKGROUPSIZE_RESULT_TYPE
#undef WORKGROUPSIZE_RESULT_SUCCESS
#undef WORKGROUPSIZE_RESULT_EINVAL
#endif	/* CUDA_DYNPARA_H */
