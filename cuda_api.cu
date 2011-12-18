/*
 * cuda_api.cu
 *
 * A set of simple wrappers to CUDA runtime APIs, because some of basic
 * declarations are conflicts between PostgreSQL and CUDA, thus we need
 * to invoke CUDA runtime API from files that does not include any
 * header files of PostgreSQL. That's too bad. :(
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "pg_strom_cuda.h"
#include <cuda_runtime.h>

const char *
pgcuda_get_error_string(cudaError_t error)
{
	return cudaGetErrorString(error);
}

cudaError_t
pgcuda_get_device_count(int *count)
{
	return cudaGetDeviceCount(count);
}	

cudaError_t
pgcuda_set_device(int device)
{
	return cudaSetDevice(device);
}

cudaError_t
pgcuda_get_device(int *device)
{
	return cudaGetDevice(device);
}

cudaError_t
pgcuda_get_device_properties(struct cudaDeviceProp *prop, int device)
{
	return cudaGetDeviceProperties(prop, device);
}

cudaError_t
pgcuda_malloc(void **devptr, size_t size)
{
	return cudaMalloc(devptr, size);
}

cudaError_t
pgcuda_free(void *devptr)
{
	return pgcuda_free(devptr);
}

cudaError_t
pgcuda_malloc_host(void **ptr, size_t size)
{
	return cudaMallocHost(ptr, size);
}

cudaError_t
pgcuda_free_host(void *ptr)
{
	return cudaFreeHost(ptr);
}

cudaError_t
pgcuda_memcpy(void *dst, const void *src, size_t count,
			  enum cudaMemcpyKind kind)
{
	return cudaMemcpy(dst, src, count, kind);
}

cudaError_t
pgcuda_memcpy_async(void *dst, const void *src, size_t count,
					enum cudaMemcpyKind kind, cudaStream_t stream)
{
	return cudaMemcpyAsync(dst, src, count, kind, stream);
}

cudaError_t
pgcuda_stream_create(cudaStream_t *p_stream)
{
	return cudaStreamCreate(p_stream);
}

cudaError_t
pgcuda_stream_destroy(cudaStream_t stream)
{
	return cudaStreamDestroy(stream);
}

cudaError_t
pgcuda_stream_synchronize(cudaStream_t stream)
{
	return cudaStreamSynchronize(stream);
}
