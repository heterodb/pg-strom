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
#define TIMEVAL_RECORD(ktask,field,tv_start)						\
	do {															\
		(ktask)->pfm.num_##field++;									\
		(ktask)->pfm.tv_##field +=									\
			((cl_float)(GlobalTimer() - (tv_start)) / 1000000.0);	\
	} while(0)
#endif	/* __CUDACC__ */

/*
 * __optimal_workgroup_size
 *
 * Logic is equivalent to cudaOccupancyMaxPotentialBlockSize(),
 * but is not supported by the device runtime at CUDA-7.5.
 * So, we ported the same calculation logic here.
 */
STATIC_INLINE(WORKGROUPSIZE_RESULT_TYPE)
__optimal_workgroup_size(cl_uint *p_grid_sz,
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
optimal_workgroup_size(dim3 *p_grid_sz,
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
	status = __optimal_workgroup_size(&p_grid_sz->x,
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
__largest_workgroup_size(cl_uint *p_grid_sz,
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
largest_workgroup_size(dim3 *p_grid_sz,
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

	status = __largest_workgroup_size(&p_grid_sz->x,
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
							 cl_uint		shmem_per_thread,
							 cl_uint		shmem_per_block)
{
	dim3		grid_sz;
	dim3		block_sz;
	cudaError_t	status;

	status = optimal_workgroup_size(&grid_sz,
									&block_sz,
									kern_function,
									num_threads,
									shmem_per_thread);
	if (status != cudaSuccess)
		return status;

	status = cudaLaunchDevice(kern_function,
							  kern_argbuf,
							  grid_sz, block_sz,
							  shmem_per_block +
							  shmem_per_thread * block_sz.x,
							  NULL);
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
							cl_uint		shmem_per_thread,
							cl_uint		shmem_per_block)
{
	kern_arg_t  *kern_args = NULL;

	return __pgstromLaunchDynamicKernel(kern_function,
										kern_args,
										num_threads,
										shmem_per_thread,
										shmem_per_block);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernel1(void	   *kern_function,
							kern_arg_t	karg0,
							size_t		num_threads,
							cl_uint		shmem_per_thread,
							cl_uint		shmem_per_block)
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
										shmem_per_thread,
										shmem_per_block);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernel2(void	   *kern_function,
							kern_arg_t	karg0,
							kern_arg_t	karg1,
							size_t		num_threads,
							cl_uint		shmem_per_thread,
							cl_uint		shmem_per_block)
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
										shmem_per_thread,
										shmem_per_block);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernel3(void	   *kern_function,
							kern_arg_t	karg0,
							kern_arg_t	karg1,
							kern_arg_t	karg2,
							size_t		num_threads,
							cl_uint		shmem_per_thread,
							cl_uint		shmem_per_block)
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
										shmem_per_thread,
										shmem_per_block);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernel4(void	   *kern_function,
							kern_arg_t	karg0,
							kern_arg_t	karg1,
							kern_arg_t	karg2,
							kern_arg_t	karg3,
							size_t		num_threads,
							cl_uint		shmem_per_thread,
							cl_uint		shmem_per_block)
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
										shmem_per_thread,
										shmem_per_block);
}

STATIC_FUNCTION(cudaError_t)
pgstromLaunchDynamicKernel5(void	   *kern_function,
							kern_arg_t	karg0,
							kern_arg_t	karg1,
							kern_arg_t	karg2,
							kern_arg_t	karg3,
							kern_arg_t	karg4,
							size_t		num_threads,
							cl_uint		shmem_per_thread,
							cl_uint		shmem_per_block)
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
										shmem_per_thread,
										shmem_per_block);
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
							cl_uint		shmem_per_thread,
							cl_uint		shmem_per_block)
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
										shmem_per_thread,
										shmem_per_block);
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
							cl_uint		shmem_per_thread,
							cl_uint		shmem_per_block)
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
										shmem_per_thread,
										shmem_per_block);
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
							cl_uint		shmem_per_thread,
							cl_uint		shmem_per_block)
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
										shmem_per_thread,
										shmem_per_block);
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
							cl_uint		shmem_per_thread,
							cl_uint		shmem_per_block)
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
										shmem_per_thread,
										shmem_per_block);
}

#endif	/* __CUDACC__ */
#undef WORKGROUPSIZE_RESULT_TYPE
#undef WORKGROUPSIZE_RESULT_SUCCESS
#undef WORKGROUPSIZE_RESULT_EINVAL
#endif	/* CUDA_DYNPARA_H */
