/*
 * cuda_cublas.h
 *
 * Switch to link cuBLAS library for matrix handling in GPU kernel.
 * --
 * Copyright 2011-2018 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2018 (C) The PG-Strom Development Team
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
#ifndef CUDA_CUBLAS_H
#define CUDA_CUBLAS_H
#ifdef __CUDACC__
#include <cublas.h>
#endif	/* __CUDACC__ */
#endif	/* CUDA_CUBLAS_H */
