/*
 * cuda_matrix.h
 *
 * collection of matrix/vector/array support routines for CUDA devices
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
#ifndef CUDA_MATRIX_H
#define CUDA_MATRIX_H
#ifdef __CUDACC__

/* ------------------------------------------------------------------
 *
 * Support macros for PostgreSQL's array structure
 *
 * ------------------------------------------------------------------ */

typedef struct
{
	/* XXX - NOTE: varlena header is extracted by VARDATA_ANY */
	cl_int		ndim;			/* # of dimensions */
	cl_int		dataoffset;		/* offset to data, or 0 if no bitmap */
	cl_uint		elemtype;		/* element type OID */
} ArrayType;

#define MAXDIM			6

#define ARR_SIZE(a)		VARSIZE_ANY(a)
#define ARR_NDIM(a)		(((ArrayType *)(a))->ndim)
#define ARR_HASNULL(a)	(((ArrayType *)(a))->dataoffset != 0)
#define ARR_ELEMTYPE(a)	(((ArrayType *)(a))->elemtype)
#define ARR_DIMS(a)									\
	((int *) (((char *) (a)) + sizeof(ArrayType)))
#define ARR_LBOUND(a)								\
	((int *) (((char *) (a)) + sizeof(ArrayType) +	\
			  sizeof(int) * ARR_NDIM(a)))
#define ARR_NULLBITMAP(a)							\
	(ARR_HASNULL(a)									\
	 ? (((char *) (a)) + sizeof(ArrayType) +		\
		2 * sizeof(int) * ARR_NDIM(a))				\
	 : (char *) NULL)
/*
 * The total array header size (in bytes) for an array with the specified
 * number of dimensions and total number of items.
 */
#define ARR_OVERHEAD_NONULLS(ndims)					\
	MAXALIGN(sizeof(ArrayType) + 2 * sizeof(int) * (ndims))
#define ARR_OVERHEAD_WITHNULLS(ndims, nitems)		\
	MAXALIGN(sizeof(ArrayType) + 2 * sizeof(int) * (ndims) +	\
			 ((nitems) + 7) / 8)
/*
 * Returns a pointer to the actual array data.
 */
#define ARR_DATA_OFFSET(a)					\
	(ARR_HASNULL(a)							\
	 ? ((ArrayType *)(a))->dataoffset		\
	 : ARR_OVERHEAD_NONULLS(ARR_NDIM(a)))

#define ARR_DATA_PTR(a)		(((char *) (a)) + ARR_DATA_OFFSET(a))

#ifndef PG_ARRAY_TYPE_DEFINED
#define PG_ARRAY_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(array)
#endif

STATIC_INLINE(cl_int)
ArrayGetNItems(kern_context *kcxt, cl_int ndim, const cl_int *dims)
{
	cl_int		i, ret;
	cl_long		prod;

	if (ndim <= 0)
		return 0;

	ret = 1;
	for (i=0; i < ndim; i++)
	{

		if (dims[i] < 0)
		{
			/* negative dimension implies an error... */
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
			return 0;
		}
		prod = (cl_long) ret * (cl_long) dims[i];
		ret = (cl_int) prod;
		if ((cl_long) ret != prod)
		{
			/* array size exceeds the maximum allowed... */
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
			return 0;
		}
	}
	assert(ret >= 0);

	return ret;
}

#endif	/* __CUDACC__ */
#endif	/* CUDA_MATRIX_H */
