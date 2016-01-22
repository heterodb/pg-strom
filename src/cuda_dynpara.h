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



extern device cudaError_t cudaGetParameterBuffer(void **params);
extern __device__ cudaError_t

 cudaLaunchDevice(void *kernel, void *params, dim3 gridDim, dim3 blockDim, unsigned int sharedMemSize = 0, cudaStream_t stream = 0);


/*
 * pg_optimal_workgroup_size - calculation of the optimal grid/block size
 * for 1-dimensional kernel launch.
 */
STATIC_FUNCTION(cl_bool)
pg_optimal_workgroup_size(kern_context *kcxt,
						  dim *p_grid_size,
						  dim *p_block_size,
						  const void *kernel_func,
						  cl_bool maximize_blocksize,
						  cl_uint nitems,
						  cl_uint dyn_shmem_sz)
{
	cudaFuncAttributes	fattrs;
	cl_int				device;
	cudaError_t			rc;

	/* Get the current device handler */
	rc = cudaGetDevice(&device);
	if (rc != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kcxt->e, rc);
		return false;
	}

	/* Get function's attribute */
	rc = cudaFuncGetAttributes(&fattrs, kernel_func);
	if (rc != cudaSuccess)
	{
		STROM_SET_RUNTIME_ERROR(&kcxt->e, rc);
		return false;
	}
	if (fattrs.localSizeBytes <= max_shmem_sz)
	{
		STROM_SET_RUNTIME_ERROR(&kcxt->e, cudaErrorLaunchOutOfResources);
		return false;
	}
	block_size = min(fattrs.maxThreadsPerBlock,
					 (nitems + (warp_size - 1)) & ~(warp_size - 1));

	if (maximize_blocksize)
	{
		if (dyn_shmem_sz > 0 &&
			fattrs.localSizeBytes + dyn_shmem_sz * block_size > max_shmem_sz)
		{
			block_size = ((max_shmem_sz - fattrs.localSizeBytes)
						  / dyn_shmem_sz) & ~(warp_size - 1);
			if (block_size < warp_size)
			{
				rc = cudaErrorLaunchOutOfResources;
				STROM_SET_RUNTIME_ERROR(&kcxt->e, rc);
				return false;
			}
		}
	}
	else
	{
		rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&block_size,
														   kernel_func,
														   block_size,
														   dyn_shmem_sz);
		if (rc != cudaSuccess)
		{
			STROM_SET_RUNTIME_ERROR(&kcxt->e, rc);
			return false;
		}
	}




	? __device__ ?cudaError_t cudaDeviceGetAttribute ( int* value, cudaDeviceAttr attr, int  device )





	_device__ ?cudaError_t cudaGetDevice ( int* device )

	cudaDeviceGetAttribute();




	cudaFuncAttributes	fattr;
}

#if 0
// to be implemented later
/*
 * pg_optimal_workgroup_size2D - calculation of the optimal grid/block size
 * for 2-dimensional kernel launch.
 */
STATIC_FUNCTION(void)
pg_optimal_workgroup_size2D(kern_context *kcxt,
							dim *p_grid_size,
							dim *p_block_size,
							void *kernel_func,
							cl_bool maximize_blocksize,
							size_t x_nitems,
							size_t y_nitems,
							size_t x_shmem_sz,
							size_t y_shmem_sz)
{


}
#endif

#endif	/* __CUDACC__ */
#endif	/* CUDA_DYNPARA_H */
