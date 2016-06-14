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
	/*
	 * NOTE: We assume 4bytes varlena header for array type. It allows
	 * aligned references to the array elements. Unlike CPU side, we
	 * cannot have extra malloc to ensure 4bytes varlena header. It is
	 * the reason why our ScalarArrayOp implementation does not support
	 * array data type referenced by Var node; which is potentially has
	 * short format.
	 */
	cl_uint		vl_len_;		/* don't touch this field */
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

/* ----------------------------------------------------------------
 *
 * MATRIX data type support
 *
 * ---------------------------------------------------------------- */

#ifndef PG_MATRIX_TYPE_DEFINED
#define PG_MATRIX_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(matrix)
#endif

#endif	/* __CUDACC__ */

typedef struct
{
	cl_uint		__vl_len;	/* varlena header (only 4B header) */
	cl_int		ndim;		/* always 2 for matrix */
	cl_int		dataoffset;	/* always 0 for matrix */
	cl_uint		elemtype;	/* always FLOAT4OID for matrix */
	union {
		struct {
			cl_int	width;		/* height of the matrix (=dim1) */
			cl_int	height;		/* width of the matrix (=dim2) */
			cl_int	lbound1;	/* always 1 for matrix */
			cl_int	lbound2;	/* always 1 for matrix */
			char	values[1];	/* to be variable length */
		} d2;
		struct {
			cl_int	height;		/* height of the vector */
			cl_int	lbound1;	/* always 1 for vector */
			char	values[1];	/* to be variable length */
		} d1;
	};
} MatrixType;

STATIC_INLINE(cl_bool)
VALIDATE_ARRAY_MATRIX(MatrixType *matrix)
{
	if (!VARATT_IS_4B(matrix))
		return false;
	if (matrix->dataoffset == 0 &&
#ifdef __CUDACC__
		(matrix->elemtype == PG_INT2OID ||
		 matrix->elemtype == PG_INT4OID ||
		 matrix->elemtype == PG_INT8OID ||
		 matrix->elemtype == PG_FLOAT4OID ||
		 matrix->elemtype == PG_FLOAT8OID)
#else	/* __CUDACC__ */
		(matrix->elemtype == INT2OID ||
		 matrix->elemtype == INT4OID ||
		 matrix->elemtype == INT8OID ||
		 matrix->elemtype == FLOAT4OID ||
		 matrix->elemtype == FLOAT8OID)
#endif	/* __CUDACC__ */
		)
	{
		if (matrix->ndim == 2)
		{
			if (matrix->d2.width > 0 &&
				matrix->d2.height > 0 &&
				matrix->d2.lbound1 == 1 &&
				matrix->d2.lbound2 == 1)
				return true;
		}
		else if (matrix->ndim == 1)
		{
			if (matrix->d1.height > 0 &&
				matrix->d1.lbound1 == 1)
				return true;
		}
		else
			return false;
	}
	return false;
}

#define ARRAY_MATRIX_ELEMTYPE(X)			\
	(((MatrixType *)(X))->elemtype)
#define ARRAY_MATRIX_HEIGHT(X)											\
	(((MatrixType *)(X))->ndim == 2 ? ((MatrixType *)(X))->d2.height :	\
	 ((MatrixType *)(X))->ndim == 1 ? ((MatrixType *)(X))->d1.height : -1)
#define ARRAY_MATRIX_WIDTH(X)											\
	(((MatrixType *)(X))->ndim == 2 ? ((MatrixType *)(X))->d2.width :	\
	 ((MatrixType *)(X))->ndim == 1 ? 1 : -1)
#define ARRAY_MATRIX_DATAPTR(X)											\
	(((MatrixType *)(X))->ndim == 2 ? ((MatrixType *)(X))->d2.values :	\
	 ((MatrixType *)(X))->ndim == 1 ? ((MatrixType *)(X))->d1.values : NULL)
#define ARRAY_MATRIX_RAWSIZE(typlen,height,width)		\
	offsetof(MatrixType, d2.values[(size_t)(typlen) *	\
								   (size_t)(height) *	\
								   (size_t)(width)])
#define ARRAY_VECTOR_RAWSIZE(typlen,nitems)				\
	offsetof(MatrixType, d1.values[(size_t)(typlen) *	\
								   (size_t)(nitems)])

#define INIT_ARRAY_VECTOR(X,_elemtype,_typlen,_nitems)			\
	do {														\
		size_t	__len = ARRAY_VECTOR_RAWSIZE(_typlen,_nitems);	\
																\
		SET_VARSIZE(X, __len);									\
		((MatrixType *)(X))->ndim = 1;							\
		((MatrixType *)(X))->dataoffset = 0;					\
		((MatrixType *)(X))->elemtype = (_elemtype);			\
		((MatrixType *)(X))->d1.height = (_nitems);				\
		((MatrixType *)(X))->d1.lbound1 = 1;					\
	} while(0)

#define INIT_ARRAY_MATRIX(X,_elemtype,_typlen,_height,_width)	\
	do {														\
		size_t	__len = ARRAY_MATRIX_RAWSIZE((_typlen),			\
											 (_height),			\
											 (_width));			\
		SET_VARSIZE(X, __len);									\
		((MatrixType *)(X))->ndim = 2;							\
		((MatrixType *)(X))->dataoffset = 0;					\
		((MatrixType *)(X))->elemtype = (_elemtype);			\
		((MatrixType *)(X))->d2.height = (_height);				\
		((MatrixType *)(X))->d2.width = (_width);				\
		((MatrixType *)(X))->d2.lbound1 = 1;					\
		((MatrixType *)(X))->d2.lbound2 = 1;					\
	} while(0)

#endif	/* CUDA_MATRIX_H */
