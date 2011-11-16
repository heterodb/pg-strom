/*
 * cuda.c
 *
 * A set of simple wrappers to CUDA runtime APIs, because some of basic
 * declarations are conflicts between PostgreSQL and CUDA, thus we need
 * to invoke CUDA runtime API from files that does not include any
 * header files of PostgreSQL. That's too bad.
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "pg_rapid.h"
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
pgcuda_get_device_properties(struct cudaDeviceProp *prop, int device)
{
	return cudaGetDeviceProperties(prop, device);
}
