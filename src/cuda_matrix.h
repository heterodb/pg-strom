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

#if 0
/* ----------------------------------------------------------------
 *
 * Bitonic Sorting Support for PL/CUDA functions
 *
 * pgstromBitonicSortFP32
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 * ----------------------------------------------------------------
 */

#define PGSTROM_BITONIC_SORT_TEMPLATE(SUFFIX,BASETYPE)					\
	STATIC_INLINE(cl_int)												\
	pgstrom_bitonic_keycomp_#SUFFIX(BASETYPE *data_ptr,					\
									cl_uint	height,						\
									cl_uint	width,						\
									cl_uint  *sortkeys,					\
									cl_uint	num_keys,					\
									cl_uint	x_index,					\
									cl_uint	y_index)					\
	{																	\
		cl_uint		i, j;												\
		BASETYPE	x_value;											\
		BASETYPE	y_value;											\
																		\
		for (i=0; i < num_keys; i++)									\
		{																\
			j = sortkeys[i];											\
			assert(j < width);											\
																		\
			x_value = data_ptr[j * height + x_index];					\
			y_value = data_ptr[j * height + y_index];					\
																		\
			if (x_value < y_value)										\
				return -1;												\
			else if (x_value > y_value)									\
				return -1;												\
		}																\
		return 0;	/* both rows are equivalent */						\
	}																	\
																		\
	KERNEL_FUNCTION_MAXTHREADS(void)									\
	pgstrom_bitonic_local_#SUFFIX(BASETYPE *data_ptr,					\
								  cl_uint height,						\
								  cl_uint width,						\
								  cl_uint *row_index,					\
								  cl_uint *sortkeys,					\
								  cl_uint num_keys,						\
								  cl_int direction)						\
	{																	\
		cl_uint	   *localIdx = SHARED_WORKMEM(cl_uint);					\
		cl_uint		localLimit;											\
		cl_uint		partSize = 2 * get_local_size();					\
		cl_uint		partBase = get_global_index() * partSize;			\
		cl_uint		blockSize;											\
		cl_uint		unitSize;											\
		cl_uint		i;													\
																		\
		/* Load index to localIdx[] */									\
		localLimit = (partBase + partSize <= height						\
					  ? partSize										\
					  : height - partBase);								\
		for (i = get_local_id();										\
			 i < localLimit;											\
			 i += get_local_size())										\
			localIdx[i] = partBase + i;									\
		__syncthreads();												\
																		\
		for (blockSize = 2;	blockSize <= partSize; blockSize *= 2)		\
		{																\
			for (unitSize = blockSize; unitSize >= 2; unitSize /= 2)	\
			{															\
				cl_uint		unitMask		= (unitSize - 1);			\
				cl_uint		halfUnitSize	= (unitSize >> 1);			\
				cl_uint		halfUnitMask	= (halfUnitSize - 1);		\
				cl_uint		idx0, idx1;									\
																		\
				idx0 = (((get_local_id() & ~halfUnitMask) << 1) +		\
						(get_local_id() & halfUnitMask));				\
				idx1 = (unitSize == blockSize							\
						? ((idx0 & ~unitMask) | (~idx0 & unitMask))		\
						: (halfUnitSize + idx0));						\
				if (idx1 < localLimit)									\
				{														\
					cl_uint		pos0 = localIdx[idx0];					\
					cl_uint		pos1 = localIdx[idx1];					\
																		\
					if (__pgstrom_bitonic_keycomp(data_ptr,				\
												  height,				\
												  width,				\
												  sortkeys,				\
												  num_keys,				\
												  pos0,					\
												  pos1) == direction)	\
					{													\
						/* swap */										\
						localIdx[idx0] = pos1;							\
						localIdx[idx1] = pos0;							\
					}													\
				}														\
				__syncthreads();										\
			}															\
		}																\
		/* write back the sorting result of this local block */			\
		for (i = get_local_id(); i < partSize; i += get_local_size())	\
			row_index[partBase + i] = localIdx[i];						\
		__syncthreads();												\
	}																	\
																		\
	KERNEL_FUNCTION_MAXTHREADS(void)									\
	pgstrom_bitonic_step_#SUFFIX(BASETYPE  *data_ptr,					\
								 cl_uint	height,						\
								 cl_uint	width,						\
								 cl_uint   *row_index,					\
								 cl_uint   *sortkeys,					\
								 cl_uint	num_keys,					\
								 cl_int		direction,					\
								 cl_uint	unitSize,					\
								 cl_bool	reversing)					\
	{																	\
		cl_uint		unitMask = unitSize - 1;							\
		cl_uint		halfUnitSize = unitSize >> 1;						\
		cl_uint		halfUnitMask = halfUnitSize - 1;					\
		cl_uint		idx0, idx1;											\
		cl_uint		pos0, pos1;											\
																		\
		idx0 = (((get_global_id() & ~halfUnitMask) << 1)				\
				+ (get_global_id() & halfUnitMask));					\
		idx1 = (reversing												\
				? ((idx0 & ~unitMask) | (~idx0 & unitMask))				\
				: (idx0 + halfUnitSize));								\
		if (idx1 < height)												\
		{																\
			pos0 = row_index[idx0];										\
			pos1 = row_index[idx1];										\
																		\
			if (__pgstrom_bitonic_keycomp(data_ptr,						\
										  height,						\
										  width,						\
										  sortkeys,						\
										  num_keys,						\
										  pos0,							\
										  pos1) == direction)			\
			{															\
				/* swap */												\
				row_index[idx0] = pos1;									\
				row_index[idx1] = pos0;									\
			}															\
		}																\
	}																	\
																		\
	KERNEL_FUNCTION_MAXTHREADS(void)									\
	pgstrom_bitonic_merge_#SUFFIX(BASETYPE *data_ptr,					\
								  cl_uint	height,						\
								  cl_uint	width,						\
								  cl_uint  *row_index,					\
								  cl_uint  *sortkeys,					\
								  cl_uint	num_keys,					\
								  cl_int	direction)					\
	{																	\
		cl_uint	   *localIdx = SHARED_WORKMEM(cl_uint);					\
		cl_uint		localLimit;											\
		cl_uint		partSize = 2 * get_local_size();					\
		cl_uint		partBase = get_global_index() * partSize;			\
		cl_uint		blockSize = partSize;								\
		cl_uint		unitSize;											\
		cl_uint		i;													\
																		\
		/* Load index to localIdx[] */									\
		localLimit = (partBase + partSize <= height						\
					  ? partSize										\
					  : height - partBase);								\
		for (i = get_local_id(); i < localLimit; i += get_local_size())	\
			localIdx[i] = row_index[partBase + i];						\
		__syncthreads();												\
																		\
		/* merge two sorted blocks */									\
		for (unitSize = blockSize; unitSize >= 2; unitSize >>= 1)		\
		{																\
			cl_uint		halfUnitSize = (unitSize >> 1);					\
			cl_uint		halfUnitMask = (halfUnitSize - 1);				\
			cl_uint		idx0, idx1;										\
																		\
			idx0 = (((get_local_id() & ~halfUnitMask) << 1)				\
					+ (get_local_id() & halfUnitMask));					\
			idx1 = halfUnitSize + idx0;									\
																		\
			if (idx1 < localLimit)										\
			{															\
				cl_uint		pos0 = localIdx[idx0];						\
				cl_uint		pos1 = localIdx[idx1];						\
																		\
				if (__pgstrom_bitonic_keycomp(data_ptr,					\
											  height,					\
											  width,					\
											  sortkeys,					\
											  num_keys,					\
											  pos0,						\
											  pos1) == direction)		\
				{														\
					/* swap */											\
					row_index[idx0] = pos1;								\
					row_index[idx1] = pos0;								\
				}														\
			}															\
			__syncthreads();											\
		}																\
		/* update the row_index[] */									\
		for (i = get_local_id(); i < partSize; i += get_local_size())	\
			row_index[partBase + i] = localIdx[i];						\
		__syncthreads();												\
	}																	\
																		\
	KERNEL_FUNCTION(void)												\
	pgstrom_matrix_copy_#SUFFIX(BASETYPE   *src_ptr,					\
								BASETYPE   *dst_ptr,					\
								cl_uint		height,						\
								cl_uint		width,						\
								cl_uint	   *row_index)					\
	{																	\
		if (get_global_id() < height)									\
		{																\
			cl_uint		src_idx = row_index[get_global_id()];			\
			cl_uint		dst_idx = get_global_id();						\
			cl_uint		i, offset;										\
																		\
			for (i=0, offset=0; i < width; i++, offset += height)		\
			{															\
				dst_ptr[offset + dst_idx] = src_ptr[offset + src_idx];	\
			}															\
		}																\
	}

PGSTROM_BITONIC_SORT_TEMPLATE(fp32, cl_float)
PGSTROM_BITONIC_SORT_TEMPLATE(fp64, cl_double)

STATIC_FUNCTION(cudaError_t)
pgstrom_bitonic_sort_fp32(MatrixType   *M,
						  MatrixType   *R,
						  void		   *row_index,
						  cl_uint	   *sort_keys,
						  cl_uint		num_keys,
						  cl_bool		is_descending)
{
	cl_uint		height = ARRAY_MATRIX_HEIGHT(M);
	cl_uint		width = ARRAY_MATRIX_WIDTH(M);
	cl_int		direction = (is_descending ? 1 : -1);
	dim3		grid_sz;
	dim3		block_sz;
	cl_uint		__block_sz = UINT_MAX;
	Datum		__kern_args[7];
	Datum	   *kern_args;
	void	   *kern_funcs[3];
	cudaError_t	status = cudaSuccess;

	if (ARRAY_MATRIX_ELEMTYPE(M) != PG_FLOAT4OID)
		return cudaErrorInvalidValue;

	/* nothing to sort */
	if (num_keys == 0)
		return cudaSuccess;

	/* setup common kernel arguments */
	__sort_keys = malloc(sizeof(cl_uint) * num_keys);
	if (!__sort_keys)
		return cudaErrorMemoryAllocation;
	memcpy(__sort_keys, sort_keys, sizeof(cl_uint) * num_keys);

	__kern_args[0] = (Datum)ARRAY_MATRIX_DATAPTR(M);
	__kern_args[1] = (Datum)(height);
	__kern_args[2] = (Datum)(width);
	__kern_args[3] = (Datum)(row_index);
	__kern_args[4] = (Datum)(__sort_keys);
	__kern_args[5] = (Datum)(num_keys);
	__kern_args[6] = (Datum)(is_descending ? 1 : -1);

	/*
	 * Ensure max available block size for each kernel functions.
	 * These are declared with KERNEL_FUNCTION_MAXTHREADS, we
	 * expect largest workgroup size is equivalent to H/W limit.
	 */
	kern_funcs[0] = (void *)pgstrom_bitonic_local_fp32;
	kern_funcs[1] = (void *)pgstrom_bitonic_step_fp32;
	kern_funcs[2] = (void *)pgstrom_bitonic_merge_fp32;
	for (i=0; i < 3; i++)
	{
		cl_uint		__temp_sz;

		status = pgstrom_largest_workgroup_size(
			&grid_sz,
			&block_sz,
			kern_funcs[i],
			(height + 1) / 2,
			2 * sizeof(cl_uint));
		if (status != cudaSuccess)
			goto out;
		__temp_sz = 1 << (get_next_log2(block_sz.x + 1) - 1);
		__block_sz = Min(__block_sz, __temp_sz);
	}
	assert((__block_sz & (__block_sz - 1)) == 0);	/* to be 2^N */
	block_sz.x = __block_sz;
	block_sz.y = 1;
	block_sz.z = 1;

	/* nhalf is the least power of two value that is larger than
	 * or equal to half of the nitems. */
	nhalf = 1UL << (get_next_log2(height + 1) - 1);

	/*
	 * KERNEL_FUNCTION_MAXTHREADS(void)
	 * pgstrom_bitonic_local_#SUFFIX(...)
	 */
	kern_args = (Datum *)
		cudaGetParameterBuffer(sizeof(Datum)
							   sizeof(Datum) * 7);
	if (!kern_args)
	{
		status = cudaErrorLaunchOutOfResources;
		goto out;
	}
	memcpy(kern_args, __kern_args, sizeof(Datum) * 7);

	status = cudaLaunchDevice((void *)pgstrom_bitonic_local_fp32,
							  kern_args, grid_sz, block_sz,
							  2 * sizeof(cl_uint) * block_sz.x,
							  NULL);
	if (status != cudaSuccess)
		goto out;
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
		goto out;

	/* inter blocks bitonic sorting */
	for (i = block_sz.x; i < nhalf; i *= 2)
	{
		for (j = 2 * i; j > block_sz.x; j /= 2)
		{
			cl_uint		unitSize = 2 * j;
			cl_uint		workSize;

			/*
			 * KERNEL_FUNCTION_MAXTHREADS(void)
			 * pgstrom_bitonic_step_#SUFFIX(...)
			 */
			kern_args = (Datum *)
				cudaGetParameterBuffer(sizeof(Datum)
									   sizeof(Datum) * 9);
			if (!kern_args)
			{
				status = cudaErrorLaunchOutOfResources;
				goto out;
			}
			memcpy(kern_args, __kern_args, sizeof(Datum) * 7);
			kern_args[7] = (Datum)(unitSize);
			kern_args[8] = (Datum)(j == 2 * i ? true : false);

			workSize = (((height + unitSize - 1)
						 / unitSize) * unitSize / 2);
			grid_sz.x = (work_size + block_sz.x - 1) / block_sz.x;
			grid_sz.y = 1;
			grid_sz.z = 1;

			status = cudaLaunchDevice((void *)pgstrom_bitonic_step_fp32,
									  kern_args, grid_sz, block_sz,
									  0,
									  NULL);
			if (status != cudaSuccess)
				goto out;
			status = cudaDeviceSynchronize();
			if (status != cudaSuccess)
				goto out;
		}

		/*
		 * Launch: pgstrom_bitonic_merge_SUFFIX
		 */
		kern_args = (Datum *)
			cudaGetParameterBuffer(sizeof(Datum)
								   sizeof(Datum) * 7);
		if (!kern_args)
		{
			status = cudaErrorLaunchOutOfResources;
			goto out;
		}
		memcpy(kern_args, __kern_args, sizeof(Datum) * 7);

		grid_sz.x = ((height + 1) / 2 + block_sz.x - 1) / block_sz.x;
		grid_sz.y = 1;
		grid_sz.z = 1;

		status = cudaLaunchDevice((void *)pgstrom_bitonic_merge_fp32,
								  kern_args, grid_sz, block_sz,
								  2 * sizeof(cl_uint) * block_sz.x,
								  NULL);
		if (status != cudaSuccess)
			goto out;
		status = cudaDeviceSynchronize();
		if (status != cudaSuccess)
			goto out;
	}

	/*
	 * KERNEL_FUNCTION(void)
	 * pgstrom_matrix_copy_#SUFFIX(...)
	 */
	__kern_args[0] = (Datum)ARRAY_MATRIX_DATAPTR(M);
	__kern_args[1] = (Datum)ARRAY_MATRIX_DATAPTR(R);
	__kern_args[2] = (Datum)(height);
	__kern_args[3] = (Datum)(width);
	__kern_args[4] = (Datum)(row_index);

	status = pgstromLaunchDynamicKernel(pgstrom_matrix_copy_fp32,
										__kern_args, 5,
										height,
										0);
out:
	free(__sort_keys);
	return status;
}
#endif



#endif	/* CUDA_MATRIX_H */
