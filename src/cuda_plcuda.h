/*
 * cuda_plcuda.h
 *
 * CUDA device code for PL/CUDA
 * --
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
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
/*
 * GstoreIpcMapping
 */
typedef struct
{
	GstoreIpcHandle	h;		/* unique identifier of Gstore_Fdw */
	void		   *map;	/* mapped device address
							 * in the CUDA program context */
} GstoreIpcMapping;

#define PLCUDA_ARGMENT_FDESC	4
#define PLCUDA_RESULT_FDESC		5

/*
 * Error handling
 */
#define EEXIT(fmt,...)									\
	do {												\
		fprintf(stderr, "Error(L%d): " fmt "\n",		\
				__LINE__, ##__VA_ARGS__);				\
		exit(1);										\
	} while(0)

#define CUEXIT(rc,fmt,...)								\
	do {												\
		fprintf(stderr, "Error(L%d): " fmt " (%s)\n",	\
				__LINE__, ##__VA_ARGS__,				\
				cudaGetErrorName(rc));					\
	} while(0)

/* ----------------------------------------------------------------
 *
 * Support macros for array-based matrix (only used by PL/CUDA)
 *
 * ---------------------------------------------------------------- */

#define TYPEDEF_VECTOR_TEMPLATE(SUFFIX,ELEMTYPENAME)		\
	typedef struct											\
	{														\
		cl_uint		__vl_len;	/* varlena header */		\
		cl_int		ndim;		/* always 1 for vector */	\
		cl_int		dataoffset;	/* always 0 for vector */	\
		cl_uint		elemtype;	/* element type */			\
		cl_int		height;		/* =dim1 */					\
		cl_int		lbound1;	/* always 1 for vector */	\
		ELEMTYPENAME values[1];	/* arrays of values */		\
	} VectorType##SUFFIX

TYPEDEF_VECTOR_TEMPLATE(,cl_char);
TYPEDEF_VECTOR_TEMPLATE(Bool,cl_bool);
TYPEDEF_VECTOR_TEMPLATE(Short,cl_short);
TYPEDEF_VECTOR_TEMPLATE(Int,cl_int);
TYPEDEF_VECTOR_TEMPLATE(Long,cl_long);
TYPEDEF_VECTOR_TEMPLATE(Float,cl_float);
TYPEDEF_VECTOR_TEMPLATE(Double,cl_double);

#define TYPEDEF_MATRIX_TEMPLATE(SUFFIX,ELEMTYPENAME)		\
	typedef struct											\
	{														\
		cl_uint		__vl_len;	/* varlena header */		\
		cl_int		ndim;		/* always 2 for matrix */	\
		cl_int		dataoffset;	/* always 0 for matrix */	\
		cl_uint		elemtype;	/* element type */			\
		cl_int		height;		/* =dim1 */					\
		cl_int		width;		/* =dim2 */					\
		cl_int		lbound1;	/* always 1 for matrix */	\
		cl_int		lbound2;	/* always 1 for matrix */	\
		ELEMTYPENAME values[1];	/* arrays of values */		\
	} MatrixType##SUFFIX

TYPEDEF_MATRIX_TEMPLATE(,cl_char);
TYPEDEF_MATRIX_TEMPLATE(Bool,cl_bool);
TYPEDEF_MATRIX_TEMPLATE(Short,cl_short);
TYPEDEF_MATRIX_TEMPLATE(Int,cl_int);
TYPEDEF_MATRIX_TEMPLATE(Long,cl_long);
TYPEDEF_MATRIX_TEMPLATE(Float,cl_float);
TYPEDEF_MATRIX_TEMPLATE(Double,cl_double);

#define TYPEDEF_CUBE_TEMPLATE(SUFFIX,ELEMTYPENAME)			\
	typedef struct											\
	{														\
		cl_uint		__vl_len;	/* varlena header */		\
		cl_int		ndim;		/* always 3 for cube */		\
		cl_int		dataoffset;	/* always 0 for cube */		\
		cl_uint		elemtype;	/* element type */			\
		cl_int		depth;		/* =dim1 */					\
		cl_int		width;		/* =dim2 */					\
		cl_int		height;		/* =dim3 */					\
		cl_int		lbound1;	/* always 1 for cube */		\
		cl_int		lbound2;	/* always 1 for cube */		\
		cl_int		lbound3;	/* always 1 for cube */		\
		ELEMTYPENAME values[1];	/* arrays of values */		\
	} CubeType##SUFFIX

TYPEDEF_CUBE_TEMPLATE(,cl_char);
TYPEDEF_CUBE_TEMPLATE(Bool,cl_bool);
TYPEDEF_CUBE_TEMPLATE(Short,cl_short);
TYPEDEF_CUBE_TEMPLATE(Int,cl_int);
TYPEDEF_CUBE_TEMPLATE(Long,cl_long);
TYPEDEF_CUBE_TEMPLATE(Float,cl_float);
TYPEDEF_CUBE_TEMPLATE(Double,cl_double);

#ifdef __CUDACC__
#define IS_ARRAY_MATRIX_TYPE(elemtype)	\
	((elemtype) == PG_BOOLOID ||		\
	 (elemtype) == PG_INT2OID ||		\
	 (elemtype) == PG_INT4OID ||		\
	 (elemtype) == PG_INT8OID ||		\
	 (elemtype) == PG_FLOAT4OID ||		\
	 (elemtype) == PG_FLOAT8OID)
#else
#define IS_ARRAY_MATRIX_TYPE(elemtype)	\
	((elemtype) == BOOLOID ||			\
	 (elemtype) == INT2OID ||			\
	 (elemtype) == INT4OID ||			\
	 (elemtype) == INT8OID ||			\
	 (elemtype) == FLOAT4OID ||			\
	 (elemtype) == FLOAT8OID)
#endif /* __CUDACC__ */

#define ARRAY_VECTOR_RAWSIZE(type_len,height)	\
	offsetof(VectorType,values[(size_t)(type_len) * (size_t)(height)])

#define INIT_ARRAY_VECTOR(X,_elemtype,_typlen,_height)			\
	do {														\
		size_t __len = ARRAY_VECTOR_RAWSIZE(_typlen,_height);	\
		SET_VARSIZE(X, __len);									\
		Assert(IS_ARRAY_MATRIX_TYPE(_elemtype));				\
		((VectorType *)(X))->ndim = 1;							\
		((VectorType *)(X))->dataoffset = 0;					\
		((VectorType *)(X))->elemtype = (_elemtype);			\
		((VectorType *)(X))->height = (_height);				\
		((VectorType *)(X))->lbound1 = 1;						\
	} while(0)

STATIC_INLINE(cl_bool)
__VALIDATE_ARRAY_VECTOR(ArrayType *X, cl_uint elemtype, cl_bool strict)
{
	if (VARATT_IS_4B(X) &&
		!ARR_HASNULL(X) &&
		(elemtype == 0
		 ? IS_ARRAY_MATRIX_TYPE(ARR_ELEMTYPE(X))
		 : ARR_ELEMTYPE(X) == elemtype))
	{
		if (X->ndim == 1)
		{
			if (((VectorType *)X)->lbound1 == 1 &&
				((VectorType *)X)->height > 0)
				return true;
		}
		else if (!strict && X->ndim == 2)
		{
			if (((MatrixType *)X)->lbound1 == 1 &&
				((MatrixType *)X)->lbound2 == 1 &&
				((MatrixType *)X)->width == 1 &&
				((MatrixType *)X)->height > 0)
				return true;
		}
		else if (!strict && X->ndim == 3)
		{
			if (((CubeType *)X)->lbound1 == 1 &&
				((CubeType *)X)->lbound2 == 1 &&
				((CubeType *)X)->lbound3 == 1 &&
				((CubeType *)X)->depth == 1 &&
				((CubeType *)X)->width == 1 &&
				((CubeType *)X)->height > 0)
				return true;
		}
	}
	return false;
}
#define VALIDATE_ARRAY_VECTOR(X)				\
	__VALIDATE_ARRAY_VECTOR((ArrayType *)(X),0,false)
#define VALIDATE_ARRAY_VECTOR_STRICT(X)			\
	__VALIDATE_ARRAY_VECTOR((ArrayType *)(X),0,type)
#define VALIDATE_ARRAY_VECTOR_TYPE(X,_elemtype)	\
	__VALIDATE_ARRAY_VECTOR((ArrayType *)(X),(_elemtype),false)
#define VALIDATE_ARRAY_VECTOR_TYPE_STRICT(X,_elemtype) \
	__VALIDATE_ARRAY_VECTOR((ArrayType *)(X),(_elemtype),true)
#define ARRAY_VECTOR_HEIGHT(X)										\
	(((ArrayType *)(X))->ndim == 1 ? ((VectorType *)(X))->height :		\
	 (((ArrayType *)(X))->ndim == 2 ? ((MatrixType *)(X))->height :		\
	  (((ArrayType *)(X))->ndim == 3 ? ((CubeType *)(X))->height : -1)))

#define ARRAY_MATRIX_RAWSIZE(type_len,height,width)		\
	offsetof(MatrixType, values[(size_t)(type_len) *	\
								(size_t)(height) *		\
								(size_t)(width)])
#define INIT_ARRAY_MATRIX(X,_elemtype,_typlen,_height,_width)	\
	do {														\
		size_t	__len = ARRAY_MATRIX_RAWSIZE(_typlen,			\
											 _height,			\
											 _width);			\
		SET_VARSIZE(X, __len);									\
		Assert(IS_ARRAY_MATRIX_TYPE(_elemtype));				\
		((MatrixType *)(X))->ndim = 2;							\
		((MatrixType *)(X))->dataoffset = 0;					\
		((MatrixType *)(X))->elemtype = (_elemtype);			\
		((MatrixType *)(X))->height = (_height);				\
		((MatrixType *)(X))->width = (_width);					\
		((MatrixType *)(X))->lbound1 = 1;						\
		((MatrixType *)(X))->lbound2 = 1;						\
	} while(0)

STATIC_INLINE(cl_bool)
__VALIDATE_ARRAY_MATRIX(ArrayType *X, cl_uint elemtype, cl_bool strict)
{
	if (VARATT_IS_4B(X) &&
		!ARR_HASNULL(X) &&
		(elemtype == 0
		 ? IS_ARRAY_MATRIX_TYPE(ARR_ELEMTYPE(X))
		 : ARR_ELEMTYPE(X) == elemtype))
	{
		if (!strict && X->ndim == 1)
		{
			if (((VectorType *)X)->lbound1 == 1 &&
				((VectorType *)X)->height  > 0)
				return true;
		}
		else if (X->ndim == 2)
		{
			if (((MatrixType *)X)->lbound1 == 1 &&
				((MatrixType *)X)->lbound2 == 1 &&
				((MatrixType *)X)->width  > 0 &&
				((MatrixType *)X)->height > 0)
				return true;
		}
		else if (!strict && X->ndim == 3)
		{
			if (((CubeType *)X)->lbound1 == 1 &&
				((CubeType *)X)->lbound2 == 1 &&
				((CubeType *)X)->lbound3 == 1 &&
				((CubeType *)X)->depth == 1 &&
				((CubeType *)X)->width  > 0 &&
				((CubeType *)X)->height > 0)
				return true;
		}
	}
	return false;
}
#define VALIDATE_ARRAY_MATRIX(X)				\
	__VALIDATE_ARRAY_MATRIX((ArrayType *)(X),0,false)
#define VALIDATE_ARRAY_MATRIX_TYPE(X,_elemtype)	\
	__VALIDATE_ARRAY_MATRIX((ArrayType *)(X),(_elemtype),false)
#define VALIDATE_ARRAY_MATRIX_STRICT(X)					\
	__VALIDATE_ARRAY_MATRIX((ArrayType *)(X),0,true)
#define VALIDATE_ARRAY_MATRIX_TYPE_STRICT(X,_elemtype)	\
	__VALIDATE_ARRAY_MATRIX((ArrayType *)(X),(_elemtype),true)

#define ARRAY_MATRIX_WIDTH(X)											\
	(((ArrayType *)(X))->ndim == 1 ? 1 :								\
	 (((ArrayType *)(X))->ndim == 2 ? ((MatrixType *)(X))->width :		\
	  (((ArrayType *)(X))->ndim == 3 ? ((CubeType *)(X))->width : -1)))
#define ARRAY_MATRIX_HEIGHT(X)											\
	ARRAY_VECTOR_HEIGHT(X)


#define ARRAY_CUBE_RAWSIZE(type_len,height,width,depth)	\
	offsetof(CubeType, values[(size_t)(type_len) *		\
							  (size_t)(height) *			\
							  (size_t)(width) *			\
							  (size_t)(depth)])
#define INIT_ARRAY_CUBE(X,_elemtype,_typlen,_height,_width,_depth)	\
	do {														\
		size_t	__len = ARRAY_CUBE_RAWSIZE(_typlen,				\
										   _height,				\
										   _width,				\
										   _depth);				\
		SET_VARSIZE(X, __len);									\
		Assert(IS_ARRAY_MATRIX_TYPE(_elemtype));				\
		((CubeType *)(X))->ndim = 3;							\
		((CubeType *)(X))->dataoffset = 0;						\
		((CubeType *)(X))->elemtype = (_elemtype);				\
		((CubeType *)(X))->height = (_height);					\
		((CubeType *)(X))->width = (_width);					\
		((CubeType *)(X))->depth = (_depth);					\
		((CubeType *)(X))->lbound1 = 1;							\
		((CubeType *)(X))->lbound2 = 1;							\
		((CubeType *)(X))->lbound3 = 1;							\
	} while(0)

STATIC_INLINE(cl_bool)
__VALIDATE_ARRAY_CUBE(ArrayType *X, cl_uint elemtype, cl_bool strict)
{
	if (VARATT_IS_4B(X) &&
		!ARR_HASNULL(X) &&
		(elemtype == 0
		 ? IS_ARRAY_MATRIX_TYPE(ARR_ELEMTYPE(X))
		 : ARR_ELEMTYPE(X) == elemtype))
	{
		if (!strict && X->ndim == 1)
		{
			if (((VectorType *)X)->lbound1 == 1 &&
				((VectorType *)X)->height  > 0)
				return true;
		}
		else if (!strict && X->ndim == 2)
		{
			if (((MatrixType *)X)->lbound1 == 1 &&
				((MatrixType *)X)->lbound2 == 1 &&
				((MatrixType *)X)->width  > 0 &&
				((MatrixType *)X)->height > 0)
				return true;
		}
		else if (X->ndim == 3)
		{
			if (((CubeType *)X)->lbound1 == 1 &&
				((CubeType *)X)->lbound2 == 1 &&
				((CubeType *)X)->lbound3 == 1 &&
				((CubeType *)X)->depth  > 0 &&
				((CubeType *)X)->width  > 0 &&
				((CubeType *)X)->height > 0)
				return true;
		}
	}
	return false;
}
#define VALIDATE_ARRAY_CUBE(X)						\
	__VALIDATE_ARRAY_CUBE((ArrayType *)(X),0,false)
#define VALIDATE_ARRAY_CUBE_TYPE(X,_elemtype)		\
	__VALIDATE_ARRAY_CUBE((ArrayType *)(X),(_elemtype),false)
#define VALIDATE_ARRAY_CUBE_STRICT(X)				\
	__VALIDATE_ARRAY_CUBE((ArrayType *)(X),0,true)
#define VALIDATE_ARRAY_CUBE_TYPE_STRICT(X,_elemtype)\
	__VALIDATE_ARRAY_CUBE((ArrayType *)(X),(_elemtype),true)

#define ARRAY_CUBE_DEPTH(X)												\
	(((ArrayType *)(X))->ndim == 1 ? 1 :								\
	 (((ArrayType *)(X))->ndim == 1 ? 1 :								\
	  (((ArrayType *)(X))->ndim == 1 ? ((CubeType *)(X))->depth : -1)))
#define ARRAY_CUBE_WIDTH(X)			ARRAY_MATRIX_WIDTH(X)
#define ARRAY_CUBE_HEIGHT(X)		ARRAY_MATRIX_HEIGHT(X)

#endif	/* CUDA_PLCUDA_H */
