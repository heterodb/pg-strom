/*
 * cuda_plcuda.h
 *
 * GPU accelerated parallel relations join based on hash-join or
 * nested-loop logic.
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
#ifndef CUDA_PLCUDA_H
#define CUDA_PLCUDA_H

#define __DATATYPE_MAX_WIDTH	80

typedef struct
{
	kern_errorbuf	kerror;
	/*
	 * NOTE: __retval is the primary result buffer. It shall be initialized
	 * on kernel invocation (prior to the prep-kernel) as follows:
	 *
	 * If result is fixed-length data type:
	 *   --> all zero clear (that implies not-null)
	 * If result is variable-length data type:
	 *   --> NULL varlena if 'results' == NULL
	 *   --> valid varlena if 'results' != NULL
	 */
	char			__retval[__DATATYPE_MAX_WIDTH];
	cl_uint			prep_num_threads;
	cl_uint			prep_shmem_size;
	cl_uint			body_num_threads;
	cl_uint			body_shmem_size;
	cl_uint			post_num_threads;
	cl_uint			post_shmem_size;
	cl_ulong		working_bufsz;
	cl_ulong		working_usage;
	cl_ulong		results_bufsz;
	cl_ulong		results_usage;
	cl_uint			total_length;	/* total lenght including kparams buffer */
	cl_uint			nargs;
	kern_colmeta	retmeta;
	kern_colmeta	argmeta[FLEXIBLE_ARRAY_MEMBER];	/* metadata of arguments */
} kern_plcuda;

#define KERN_PLCUDA_PARAMBUF(kplcuda)			\
	((kern_parambuf *)((char *)(kplcuda) +		\
					   STROMALIGN(offsetof(kern_plcuda,		\
										   argmeta[(kplcuda)->nargs]))))
#define KERN_PLCUDA_PARAMBUF_LENGTH(kplcuda)	\
	(KERN_PLCUDA_PARAMBUF(kplcuda)->length)
#define KERN_PLCUDA_DMASEND_LENGTH(kplcuda)		\
	((kplcuda)->total_length)
#define KERN_PLCUDA_DMARECV_LENGTH(kplcuda)		\
	(offsetof(kern_plcuda, retmeta))

#endif	/* CUDA_PLCUDA.H */
