/*
 * cuda_dynpara.h
 *
 * Support routine for dynamic parallelism on CUDA devices.
 * Inclusion of this library makes cuda_program.c to link libcudadevrt.a.
 * --
 * Copyright 2011-2019 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2019 (C) The PG-Strom Development Team
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
/*
 * Macro to track timeval and increment usage.
 */
#define TIMEVAL_RECORD(ktask,field,tv_start)						\
	do {															\
		(ktask)->pfm.num_##field++;									\
		(ktask)->pfm.tv_##field +=									\
			((cl_float)(GlobalTimer() - (tv_start)) / 1000000.0);	\
	} while(0)

/*
 * __occupancy_max_potential_block_size
 *
 * Equivalent logic with cudaOccupancyMaxPotentialBlockSize, but not supported
 * by the device runtime at CUDA 7.5, so we ported the same calculation logic
 * here.
 */
STATIC_INLINE(cudaError_t)
__occupancy_max_potential_block_size(cl_uint *p_minGridSize,
									 cl_uint *p_maxBlockSize,
									 const void *kernel_function,
									 cl_uint dynamicShmemPerBlock,
									 cl_uint dynamicShmemPerThread,
									 size_t blockSizeLimit)
{
	cudaError_t		status;

	/* Device and function properties */
	struct cudaFuncAttributes attr;
	int				device;

    /* Limits */
	int				maxThreadsPerMultiProcessor;
	int				warpSize;
	int				devMaxThreadsPerBlock;
	int				multiProcessorCount;
	int				funcMaxThreadsPerBlock;
	int				occupancyLimit;

    /* Recorded maximum */
    int				maxBlockSize = 0;
    int				numBlocks    = 0;
    int				maxOccupancy = 0;

    /* Temporary */
    int				blockSizeToTryAligned;
    int				blockSizeToTry;
    int				blockSizeLimitAligned;
    int				occupancyInBlocks;
    int				occupancyInThreads;
    int				dynamicSMemSize;

	/* ------------------------------------------------
	 *  Sanity checks
	 * ------------------------------------------------ */
	if (!p_minGridSize || !p_maxBlockSize || !kernel_function)
		return cudaErrorInvalidValue;

	/* ------------------------------------------------
	 *  Obtain device and function properties
	 * ------------------------------------------------ */
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

    status = cudaDeviceGetAttribute(&multiProcessorCount,
									cudaDevAttrMultiProcessorCount,
									device);
    if (status != cudaSuccess)
        return status;

	status = cudaFuncGetAttributes(&attr, kernel_function);
	if (status != cudaSuccess)
		return status;
	funcMaxThreadsPerBlock = attr.maxThreadsPerBlock;

	/* ------------------------------------------------
	 *  Try each block size, and pick the block size with
	 *  maximum occupancy
	 * ------------------------------------------------ */
	occupancyLimit = maxThreadsPerMultiProcessor;

	if (blockSizeLimit == 0)
		blockSizeLimit = devMaxThreadsPerBlock;

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

		dynamicSMemSize = (dynamicShmemPerBlock +
						   dynamicShmemPerThread * blockSizeToTry);

        status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
			&occupancyInBlocks,
			kernel_function,
			blockSizeToTry,
			dynamicSMemSize);
		if (status != cudaSuccess)
			return status;

		occupancyInThreads = blockSizeToTry * occupancyInBlocks;

		if (occupancyInThreads > maxOccupancy)
		{
			maxBlockSize = blockSizeToTry;
			numBlocks    = occupancyInBlocks;
			maxOccupancy = occupancyInThreads;
		}

		/*
		 * Early out if we have reached the maximum
		 */
		if (occupancyLimit == maxOccupancy)
			break;
	}

	/* --------------------------------
	 *  Return best available
	 * -------------------------------- */
	*p_minGridSize  = numBlocks * multiProcessorCount;
	*p_maxBlockSize = maxBlockSize;

	return status;
}

/*
 * optimal_workgroup_size - lead an optimal block_size from the standpoint
 * of performance.
 */
STATIC_FUNCTION(cudaError_t)
optimal_workgroup_size(dim3 *p_grid_sz,
					   dim3 *p_block_sz,
					   const void *kernel_function,
					   size_t nitems,
					   size_t dynamic_shmem_per_block,
					   size_t dynamic_shmem_per_thread)
{
	cudaError_t	status;
	cl_uint		minGridSize;
	cl_uint		maxBlockSize;

	status = __occupancy_max_potential_block_size(&minGridSize,
												  &maxBlockSize,
												  kernel_function,
												  dynamic_shmem_per_block,
												  dynamic_shmem_per_thread,
												  nitems);
	if (status != cudaSuccess)
		return status;

	/* nitems must be less than blockSz.x * gridSz.x */
	if ((size_t)maxBlockSize * (size_t)INT_MAX < nitems)
		return cudaErrorInvalidValue;

	p_block_sz->x = maxBlockSize;
	p_block_sz->y = 1;
	p_block_sz->z = 1;
	p_grid_sz->x = (nitems + (size_t)maxBlockSize - 1) / (size_t)maxBlockSize;
	p_grid_sz->y = 1;
	p_grid_sz->z = 1;

	return cudaSuccess;
}

/*
 * largest_workgroup_size - lead an optimal block_size from the standpoint
 * of number of threads per block
 */
STATIC_FUNCTION(cudaError_t)
largest_workgroup_size(dim3 *p_grid_sz,
					   dim3 *p_block_sz,
					   const void *kernel_function,
					   size_t nitems,
					   size_t dynamic_shmem_per_block,
					   size_t dynamic_shmem_per_thread)
{
	cudaError_t	status;
	cl_int		device;
	cl_int		warpSize;
	cl_int		maxBlockSize;
	cl_int		staticShmemSize;
	cl_int		maxShmemSize;
	cl_int		shmemSizeTotal;
	cudaFuncAttributes attrs;

	/* Sanity checks */
	if (!p_grid_sz || !p_block_sz || !kernel_function)
		return cudaErrorInvalidValue;

	/* Get device and function's attribute */
	status = cudaGetDevice(&device);
	if (status != cudaSuccess)
		return status;

	status = cudaDeviceGetAttribute(&warpSize,
									cudaDevAttrWarpSize,
									device);
	if (status != cudaSuccess)
		return status;

	status = cudaDeviceGetAttribute(&maxShmemSize,
									cudaDevAttrMaxSharedMemoryPerBlock,
									device);
	if (status != cudaSuccess)
		return status;

	status = cudaFuncGetAttributes(&attrs, kernel_function);
	if (status != cudaSuccess)
		return status;
	maxBlockSize    = attrs.maxThreadsPerBlock;
	staticShmemSize = attrs.sharedSizeBytes;

	/* only shared memory consumption is what we have to control */
	shmemSizeTotal = (staticShmemSize +
					  dynamic_shmem_per_block +
					  dynamic_shmem_per_thread * maxBlockSize);
	if (shmemSizeTotal > maxShmemSize)
	{
		if (dynamic_shmem_per_thread > 0 &&
			staticShmemSize +
			dynamic_shmem_per_block +
			dynamic_shmem_per_thread * warpSize <= maxShmemSize)
		{
			maxBlockSize = (maxShmemSize -
							staticShmemSize -
							dynamic_shmem_per_block)/dynamic_shmem_per_thread;
			maxBlockSize = (maxBlockSize / warpSize) * warpSize;
			if (maxBlockSize < warpSize)
				return cudaErrorInvalidValue;
		}
		else
		{
			/* adjust of block-size makes no sense! */
			return cudaErrorInvalidValue;
		}
	}
	/* nitems must be less than blockSz.x * gridSz.x */
	if ((size_t)maxBlockSize * (size_t)INT_MAX < nitems)
		return cudaErrorInvalidValue;

	p_block_sz->x = maxBlockSize;
	p_block_sz->y = 1;
	p_block_sz->z = 1;
	p_grid_sz->x = (nitems + maxBlockSize - 1) / maxBlockSize;
	p_grid_sz->y = 1;
	p_grid_sz->z = 1;

	return cudaSuccess;
}

/*
 * kern_arg_t - a uniformed data type to deliver kernel arguments.
 */
typedef devptr_t		kern_arg_t;

/*
 * pgstromLaunchDynamicKernelXX 
 *
 * A utility routine to launch a kernel function, and then wait for its
 * completion
 */
STATIC_FUNCTION(cudaError_t)
__pgstromLaunchDynamicKernel(void		   *kern_function,
							 kern_arg_t	   *kern_argbuf,
							 size_t			num_threads,
							 cl_uint		shmem_per_block,
							 cl_uint		shmem_per_thread)
{
	dim3		grid_sz;
	dim3		block_sz;
	cudaError_t	status;

	status = optimal_workgroup_size(&grid_sz,
									&block_sz,
									kern_function,
									num_threads,
									shmem_per_block,
									shmem_per_thread);
	if (status != cudaSuccess)
		return status;

	status = cudaLaunchDevice(kern_function,
							  kern_argbuf,
							  grid_sz, block_sz,
							  shmem_per_block +
							  shmem_per_thread * block_sz.x,
							  cudaStreamPerThread);
	if (status != cudaSuccess)
		return status;

	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
		return status;

	return cudaSuccess;
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernel0(void	   *kern_function,
							size_t		num_threads,
							cl_uint		shmem_per_block,
							cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = NULL;

	return __pgstromLaunchDynamicKernel(kern_function,
										kern_args,
										num_threads,
										shmem_per_block,
										shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernel1(void	   *kern_function,
							kern_arg_t	karg0,
							size_t		num_threads,
							cl_uint		shmem_per_block,
							cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 1);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	return __pgstromLaunchDynamicKernel(kern_function,
										kern_args,
										num_threads,
										shmem_per_block,
										shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernel2(void	   *kern_function,
							kern_arg_t	karg0,
							kern_arg_t	karg1,
							size_t		num_threads,
							cl_uint		shmem_per_block,
							cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 2);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	kern_args[1] = karg1;
	return __pgstromLaunchDynamicKernel(kern_function,
										kern_args,
										num_threads,
										shmem_per_block,
										shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernel3(void	   *kern_function,
							kern_arg_t	karg0,
							kern_arg_t	karg1,
							kern_arg_t	karg2,
							size_t		num_threads,
							cl_uint		shmem_per_block,
							cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 3);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	kern_args[1] = karg1;
	kern_args[2] = karg2;
	return __pgstromLaunchDynamicKernel(kern_function,
										kern_args,
										num_threads,
										shmem_per_block,
										shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernel4(void	   *kern_function,
							kern_arg_t	karg0,
							kern_arg_t	karg1,
							kern_arg_t	karg2,
							kern_arg_t	karg3,
							size_t		num_threads,
							cl_uint		shmem_per_block,
							cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 4);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	kern_args[1] = karg1;
	kern_args[2] = karg2;
	kern_args[3] = karg3;
	return __pgstromLaunchDynamicKernel(kern_function,
										kern_args,
										num_threads,
										shmem_per_block,
										shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernel5(void	   *kern_function,
							kern_arg_t	karg0,
							kern_arg_t	karg1,
							kern_arg_t	karg2,
							kern_arg_t	karg3,
							kern_arg_t	karg4,
							size_t		num_threads,
							cl_uint		shmem_per_block,
							cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 5);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	kern_args[1] = karg1;
	kern_args[2] = karg2;
	kern_args[3] = karg3;
	kern_args[4] = karg4;
	return __pgstromLaunchDynamicKernel(kern_function,
										kern_args,
										num_threads,
										shmem_per_block,
										shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernel6(void	   *kern_function,
							kern_arg_t	karg0,
							kern_arg_t	karg1,
							kern_arg_t	karg2,
							kern_arg_t	karg3,
							kern_arg_t	karg4,
							kern_arg_t	karg5,
							size_t		num_threads,
							cl_uint		shmem_per_block,
							cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 6);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	kern_args[1] = karg1;
	kern_args[2] = karg2;
	kern_args[3] = karg3;
	kern_args[4] = karg4;
	kern_args[5] = karg5;
	return __pgstromLaunchDynamicKernel(kern_function,
										kern_args,
										num_threads,
										shmem_per_block,
										shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernel7(void	   *kern_function,
							kern_arg_t	karg0,
							kern_arg_t	karg1,
							kern_arg_t	karg2,
							kern_arg_t	karg3,
							kern_arg_t	karg4,
							kern_arg_t	karg5,
							kern_arg_t	karg6,
							size_t		num_threads,
							cl_uint		shmem_per_block,
							cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 7);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	kern_args[1] = karg1;
	kern_args[2] = karg2;
	kern_args[3] = karg3;
	kern_args[4] = karg4;
	kern_args[5] = karg5;
	kern_args[6] = karg6;
	return __pgstromLaunchDynamicKernel(kern_function,
										kern_args,
										num_threads,
										shmem_per_block,
										shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernel8(void	   *kern_function,
							kern_arg_t	karg0,
							kern_arg_t	karg1,
							kern_arg_t	karg2,
							kern_arg_t	karg3,
							kern_arg_t	karg4,
							kern_arg_t	karg5,
							kern_arg_t	karg6,
							kern_arg_t	karg7,
							size_t		num_threads,
							cl_uint		shmem_per_block,
							cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 8);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	kern_args[1] = karg1;
	kern_args[2] = karg2;
	kern_args[3] = karg3;
	kern_args[4] = karg4;
	kern_args[5] = karg5;
	kern_args[6] = karg6;
	kern_args[7] = karg7;
	return __pgstromLaunchDynamicKernel(kern_function,
										kern_args,
										num_threads,
										shmem_per_block,
										shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernel9(void	   *kern_function,
							kern_arg_t	karg0,
							kern_arg_t	karg1,
							kern_arg_t	karg2,
							kern_arg_t	karg3,
							kern_arg_t	karg4,
							kern_arg_t	karg5,
							kern_arg_t	karg6,
							kern_arg_t	karg7,
							kern_arg_t	karg8,
							size_t		num_threads,
							cl_uint		shmem_per_block,
							cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 9);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	kern_args[1] = karg1;
	kern_args[2] = karg2;
	kern_args[3] = karg3;
	kern_args[4] = karg4;
	kern_args[5] = karg5;
	kern_args[6] = karg6;
	kern_args[7] = karg7;
	kern_args[8] = karg8;
	return __pgstromLaunchDynamicKernel(kern_function,
										kern_args,
										num_threads,
										shmem_per_block,
										shmem_per_thread);
}

/*
 * pgstromLaunchDynamicKernelMaxThreadsXX 
 *
 * A utility routine to launch a kernel function, with largest available
 * number of threads (but may not optimal), and then wait for its completion.
 */
STATIC_FUNCTION(cudaError_t)
__pgstromLaunchDynamicKernelMaxThreads(void		   *kern_function,
									   kern_arg_t  *kern_argbuf,
									   cl_uint		threads_unitsz,
									   cl_uint		num_thread_units,
									   cl_uint		shmem_per_block,
									   cl_uint		shmem_per_thread)
{
	dim3		grid_sz;
	dim3		block_sz;
	cudaError_t	status;

	status = largest_workgroup_size(&grid_sz,
									&block_sz,
									kern_function,
									threads_unitsz * num_thread_units,
									shmem_per_block,
									shmem_per_thread);
	if (status != cudaSuccess)
		return status;
	if (threads_unitsz > block_sz.x)
		return cudaErrorInvalidValue;
	if (threads_unitsz > 1)
	{
		cl_uint		npacked = block_sz.x / threads_unitsz;

		block_sz.x = (npacked * threads_unitsz +
					  warpSize - 1) & ~(warpSize - 1);
		grid_sz.x = (num_thread_units + npacked - 1) / npacked;
	}

	status = cudaLaunchDevice(kern_function,
							  kern_argbuf,
							  grid_sz, block_sz,
							  shmem_per_block +
							  shmem_per_thread * block_sz.x,
							  cudaStreamPerThread);
	if (status != cudaSuccess)
		return status;

	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
		return status;

	return cudaSuccess;
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernelMaxThreads0(void		   *kern_function,
									  cl_uint		threads_unitsz,
									  cl_uint		num_thread_units,
									  cl_uint		shmem_per_block,
									  cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = NULL;

	return __pgstromLaunchDynamicKernelMaxThreads(kern_function,
												  kern_args,
												  threads_unitsz,
												  num_thread_units,
												  shmem_per_block,
												  shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernelMaxThreads1(void		   *kern_function,
									  kern_arg_t	karg0,
									  cl_uint		threads_unitsz,
									  cl_uint		num_thread_units,
									  cl_uint		shmem_per_block,
									  cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 1);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	return __pgstromLaunchDynamicKernelMaxThreads(kern_function,
												  kern_args,
												  threads_unitsz,
												  num_thread_units,
												  shmem_per_block,
												  shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernelMaxThreads2(void		   *kern_function,
									  kern_arg_t	karg0,
									  kern_arg_t	karg1,
									  cl_uint		threads_unitsz,
									  cl_uint		num_thread_units,
									  cl_uint		shmem_per_block,
									  cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 2);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	kern_args[1] = karg1;
	return __pgstromLaunchDynamicKernelMaxThreads(kern_function,
												  kern_args,
												  threads_unitsz,
												  num_thread_units,
												  shmem_per_block,
												  shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernelMaxThreads3(void		   *kern_function,
									  kern_arg_t	karg0,
									  kern_arg_t	karg1,
									  kern_arg_t	karg2,
									  cl_uint		threads_unitsz,
									  cl_uint		num_thread_units,
									  cl_uint		shmem_per_block,
									  cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 3);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	kern_args[1] = karg1;
	kern_args[2] = karg2;
	return __pgstromLaunchDynamicKernelMaxThreads(kern_function,
												  kern_args,
												  threads_unitsz,
												  num_thread_units,
												  shmem_per_block,
												  shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernelMaxThreads4(void		   *kern_function,
									  kern_arg_t	karg0,
									  kern_arg_t	karg1,
									  kern_arg_t	karg2,
									  kern_arg_t	karg3,
									  cl_uint		threads_unitsz,
									  cl_uint		num_thread_units,
									  cl_uint		shmem_per_block,
									  cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 4);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	kern_args[1] = karg1;
	kern_args[2] = karg2;
	kern_args[3] = karg3;
	return __pgstromLaunchDynamicKernelMaxThreads(kern_function,
												  kern_args,
												  threads_unitsz,
												  num_thread_units,
												  shmem_per_block,
												  shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernelMaxThreads5(void		   *kern_function,
									  kern_arg_t	karg0,
									  kern_arg_t	karg1,
									  kern_arg_t	karg2,
									  kern_arg_t	karg3,
									  kern_arg_t	karg4,
									  cl_uint		threads_unitsz,
									  cl_uint		num_thread_units,
									  cl_uint		shmem_per_block,
									  cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 5);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	kern_args[1] = karg1;
	kern_args[2] = karg2;
	kern_args[3] = karg3;
	kern_args[4] = karg4;
	return __pgstromLaunchDynamicKernelMaxThreads(kern_function,
												  kern_args,
												  threads_unitsz,
												  num_thread_units,
												  shmem_per_block,
												  shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernelMaxThreads6(void		   *kern_function,
									  kern_arg_t	karg0,
									  kern_arg_t	karg1,
									  kern_arg_t	karg2,
									  kern_arg_t	karg3,
									  kern_arg_t	karg4,
									  kern_arg_t	karg5,
									  cl_uint		threads_unitsz,
									  cl_uint		num_thread_units,
									  cl_uint		shmem_per_block,
									  cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 6);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	kern_args[1] = karg1;
	kern_args[2] = karg2;
	kern_args[3] = karg3;
	kern_args[4] = karg4;
	kern_args[5] = karg5;
	return __pgstromLaunchDynamicKernelMaxThreads(kern_function,
												  kern_args,
												  threads_unitsz,
												  num_thread_units,
												  shmem_per_block,
												  shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernelMaxThreads7(void		   *kern_function,
									  kern_arg_t	karg0,
									  kern_arg_t	karg1,
									  kern_arg_t	karg2,
									  kern_arg_t	karg3,
									  kern_arg_t	karg4,
									  kern_arg_t	karg5,
									  kern_arg_t	karg6,
									  cl_uint		threads_unitsz,
									  cl_uint		num_thread_units,
									  cl_uint		shmem_per_block,
									  cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 7);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	kern_args[1] = karg1;
	kern_args[2] = karg2;
	kern_args[3] = karg3;
	kern_args[4] = karg4;
	kern_args[5] = karg5;
	kern_args[6] = karg6;
	return __pgstromLaunchDynamicKernelMaxThreads(kern_function,
												  kern_args,
												  threads_unitsz,
												  num_thread_units,
												  shmem_per_block,
												  shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernelMaxThreads8(void		   *kern_function,
									  kern_arg_t	karg0,
									  kern_arg_t	karg1,
									  kern_arg_t	karg2,
									  kern_arg_t	karg3,
									  kern_arg_t	karg4,
									  kern_arg_t	karg5,
									  kern_arg_t	karg6,
									  kern_arg_t	karg7,
									  cl_uint		threads_unitsz,
									  cl_uint		num_thread_units,
									  cl_uint		shmem_per_block,
									  cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 8);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	kern_args[1] = karg1;
	kern_args[2] = karg2;
	kern_args[3] = karg3;
	kern_args[4] = karg4;
	kern_args[5] = karg5;
	kern_args[6] = karg6;
	kern_args[7] = karg7;
	return __pgstromLaunchDynamicKernelMaxThreads(kern_function,
												  kern_args,
												  threads_unitsz,
												  num_thread_units,
												  shmem_per_block,
												  shmem_per_thread);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernelMaxThreads9(void		   *kern_function,
									  kern_arg_t	karg0,
									  kern_arg_t	karg1,
									  kern_arg_t	karg2,
									  kern_arg_t	karg3,
									  kern_arg_t	karg4,
									  kern_arg_t	karg5,
									  kern_arg_t	karg6,
									  kern_arg_t	karg7,
									  kern_arg_t	karg8,
									  cl_uint		threads_unitsz,
									  cl_uint		num_thread_units,
									  cl_uint		shmem_per_block,
									  cl_uint		shmem_per_thread)
{
	kern_arg_t  *kern_args = (kern_arg_t *)
		cudaGetParameterBuffer(sizeof(kern_arg_t),
							   sizeof(kern_arg_t) * 9);
	if (!kern_args)
		return cudaErrorLaunchOutOfResources;

	kern_args[0] = karg0;
	kern_args[1] = karg1;
	kern_args[2] = karg2;
	kern_args[3] = karg3;
	kern_args[4] = karg4;
	kern_args[5] = karg5;
	kern_args[6] = karg6;
	kern_args[7] = karg7;
	kern_args[8] = karg8;
	return __pgstromLaunchDynamicKernelMaxThreads(kern_function,
												  kern_args,
												  threads_unitsz,
												  num_thread_units,
												  shmem_per_block,
												  shmem_per_thread);
}
#endif	/* __CUDACC__ */
#undef WORKGROUPSIZE_RESULT_TYPE
#undef WORKGROUPSIZE_RESULT_SUCCESS
#undef WORKGROUPSIZE_RESULT_EINVAL
#endif	/* CUDA_DYNPARA_H */
