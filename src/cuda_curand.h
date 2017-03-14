/*
 * cuda_curand.h
 *
 * Switch to link cuRand library for random number generation in GPU kernel.
 * --
 * Copyright 2011-2017 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2017 (C) The PG-Strom Development Team
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
#ifndef CUDA_CURAND_H
#define CUDA_CURAND_H
#ifdef __CUDACC__
/*
 * NOTE: Because of several troubles of cuRAND library when we try to use it
 * on NVRTC environment, right now, we give up to link full set of this
 * library.
 * It tries to include files out of the default include paths, then raise
 * errors no qualifiers of functions, but __device__ as default configuration
 * makes conflicts.
 */
#define __x86_64__
#error "cuRAND on NVRTC still has several problems"
#include <curand_kernel.h>
#endif	/* __CUDACC__ */
#ifdef __cplusplus
	extern "C" {
#endif
#endif	/* CUDA_CURAND_H */
