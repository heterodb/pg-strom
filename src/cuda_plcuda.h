/*
 * cuda_plcuda.h
 *
 * GPU accelerated parallel relations join based on hash-join or
 * nested-loop logic.
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
#ifndef CUDA_PLCUDA_H
#define CUDA_PLCUDA_H

#define __DATATYPE_MAX_WIDTH	80

typedef struct
{
	size_t			length;
	kern_errorbuf	kerror_prep;
	kern_errorbuf	kerror_main;
	kern_errorbuf	kerror_post;

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
	char			__vl_buffer[512];	/* short varlena buffer */

	/*
	 * NOTE: The PL/CUDA code can use the debug counter below. If and when
	 * non-zero value is set on the variables below.
	 */
	cl_ulong		plcuda_debug_count0;
	cl_ulong		plcuda_debug_count1;
	cl_ulong		plcuda_debug_count2;
	cl_ulong		plcuda_debug_count3;
	cl_ulong		plcuda_debug_count4;
	cl_ulong		plcuda_debug_count5;
	cl_ulong		plcuda_debug_count6;
	cl_ulong		plcuda_debug_count7;

	/* parameters to launch kernel */
	cl_ulong		prep_num_threads;
	cl_uint			prep_kern_blocksz;
	cl_uint			prep_shmem_unitsz;
	cl_uint			prep_shmem_blocksz;
	cl_ulong		main_num_threads;
	cl_uint			main_kern_blocksz;
	cl_uint			main_shmem_unitsz;
	cl_uint			main_shmem_blocksz;
	cl_ulong		post_num_threads;
	cl_uint			post_kern_blocksz;
	cl_uint			post_shmem_unitsz;
	cl_uint			post_shmem_blocksz;
	cl_ulong		working_bufsz;
	cl_ulong		working_usage;
	cl_ulong		results_bufsz;
	cl_ulong		results_usage;
	cl_int			nargs;
	kern_colmeta	retmeta;	/* result data type */
	kern_colmeta	argmeta[FLEXIBLE_ARRAY_MEMBER];	/* argument's data types */
} kern_plcuda;

#define KERN_PLCUDA_PARAMBUF(kplcuda)			\
	((kern_parambuf *)((char *)(kplcuda) + (kplcuda)->length))
#define KERN_PLCUDA_PARAMBUF_LENGTH(kplcuda)	\
	(KERN_PLCUDA_PARAMBUF(kplcuda)->length)
#define KERN_PLCUDA_DMASEND_LENGTH(kplcuda)		\
	((kplcuda)->length + KERN_PLCUDA_PARAMBUF_LENGTH(kplcuda))
#define KERN_PLCUDA_DMARECV_LENGTH(kplcuda)		\
	(offsetof(kern_plcuda, retmeta))
#define PLCUDA_ERROR_RETURN(errcode)			\
	do {										\
		STROM_SET_ERROR(&kcxt->e, (errcode));	\
		return;									\
	} while(0)
#define PLCUDA_RUNTIME_ERROR_RETURN(errcode)	\
	do {										\
		STROM_SET_RUNTIME_ERROR(&kcxt->e, (errcode));	\
		return;									\
	} while(0)

/*
 * composite data type support in kernel space
 */
typedef struct
{
	cl_uint			vl_len_;	/* varlena header */
	cl_uint			type_oid;	/* oid of the composite type */
	cl_int			nattrs;		/* number of attributes */
	cl_int			__padding__;/* for alignment */
	kern_colmeta	colmeta[FLEXIBLE_ARRAY_MEMBER];
} pg_composite_typedesc;

#define PLCUDA_SETUP_COMPOSITE_RESULT(kplcuda,buffer,tup_datum,tup_isnull) \
	do {																\
		kern_parambuf *kparams = KERN_PLCUDA_PARAMBUF(kplcuda);			\
		cl_uint        type_oid = kplcuda->retmeta.atttypid;			\
		pg_composite_typedesc *ct_desc;									\
																		\
		ct_desc = pg_lookup_composite_typedesc(kparams, type_oid);		\
		assert(ct_desc != NULL);										\
		form_kern_composite_type((buffer),								\
								 &kplcuda->retmeta,						\
								 ct_desc->nattrs,						\
								 ct_desc->colmeta,						\
								 (tup_datum),							\
								 (tup_isnull));							\
	} while(0)

/*
 * gstore_fdw.c related stuff
 */
/* relation 'format' option */
#define GSTORE_FDW_FORMAT__PGSTROM		500		/* KDS with column format */
//#define GSTORE_FDW_FORMAT__PGARRAY
//#define GSTORE_FDW_FORMAT__NUMPY

/* column 'compression' option */
#define GSTORE_COMPRESSION__NONE		0
#define GSTORE_COMPRESSION__PGLZ		1

#ifdef __CUDACC__
typedef union {
	devptr_t			ptr;
	kern_data_store	   *kds;	/* GSTORE_FDW_FORMAT__PGSTROM */
} kern_reggstore_t;

STATIC_INLINE(kern_reggstore_t)
pg_reggstore_param(kern_context *kcxt, cl_uint param_id)
{
	kern_parambuf	   *kparams = kcxt->kparams;
	kern_reggstore_t	retval;

	if (param_id < kparams->nparams &&
		kparams->poffset[param_id] > 0)
	{
		retval.ptr = *((devptr_t *)((char *)kparams +
									kparams->poffset[param_id]));
	}
	else
		retval.ptr = 0L;

	return retval;
}

#endif

#endif	/* CUDA_PLCUDA.H */
