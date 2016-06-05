/*
 * matrix.c
 *
 * Matrix data type support
 * ----
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
#include "postgres.h"
#include "catalog/pg_type.h"
#include "utils/array.h"
#include "utils/arrayaccess.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/syscache.h"
#include "pg_strom.h"
#include "cuda_matrix.h"
#include <math.h>

/* fmgr macros for regular varlena matrix  objects */
#define DatumGetMatrixTypeP(X)					\
	((MatrixType *) PG_DETOAST_DATUM(X))
#define DatumGetMatrixTypePCopy(X)				\
	((MatrixType *) PG_DETOAST_DATUM_COPY(X))
#define PG_GETARG_MATRIXTYPE_P(n)				\
	DatumGetMatrixTypeP(PG_GETARG_DATUM(n))
#define PG_GETARG_MATRIXTYPE_P_COPY(n)			\
	DatumGetMatrixTypePCopy(PG_GETARG_DATUM(n))
#define PG_RETURN_MATRIXTYPE_P(x)		PG_RETURN_POINTER(x)

#define MATRIX_INIT_FIELDS(matrix,_height,_width)	\
	do {											\
		(matrix)->ndim = 2;							\
		(matrix)->dataoffset = 0;					\
		(matrix)->elemtype = FLOAT4OID;				\
		(matrix)->height = (_height);				\
		(matrix)->width = (_width);					\
		(matrix)->lbound1 = 1;						\
		(matrix)->lbound2 = 1;						\
	} while(0)										\

#define MATRIX_SANITYCHECK_NOERROR(matrix)						\
	((((MatrixType *)(matrix))->ndim == 2 &&					\
	  ((MatrixType *)(matrix))->dataoffset == 0 &&				\
	  ((MatrixType *)(matrix))->elemtype == FLOAT4OID &&		\
	  ((MatrixType *)(matrix))->lbound1 == 1 &&					\
	  ((MatrixType *)(matrix))->lbound2 == 1) ? true : false)

#define MATRIX_SANITYCHECK(matrix)										\
	do {																\
		if (!MATRIX_SANITYCHECK_NOERROR((matrix)))						\
		{																\
			if (client_min_messages <= DEBUG1)							\
				elog(ERROR, "Invalid matrix format: "					\
					 "__vl_len=%u ndim=%d dataoffset=%d elemtype=%u "	\
					 "width=%d height=%d lbound1=%d lbound2=%d",		\
					 ((MatrixType *)(matrix))->__vl_len,				\
					 ((MatrixType *)(matrix))->ndim,					\
					 ((MatrixType *)(matrix))->dataoffset,				\
					 ((MatrixType *)(matrix))->elemtype,				\
					 ((MatrixType *)(matrix))->width,					\
					 ((MatrixType *)(matrix))->height,					\
					 ((MatrixType *)(matrix))->lbound1,					\
					 ((MatrixType *)(matrix))->lbound2);				\
			else														\
				elog(ERROR, "Invalid matrix format");					\
		}																\
	} while(0)

/*
 * Type cast functions
 */
static MatrixType *
__array_to_matrix(ArrayType *array, Oid cast_func)
{
	MatrixType *matrix;
	cl_uint		x_offset;
	cl_uint		y_offset;
	cl_uint		x_nitems;
	cl_uint		y_nitems;
	Size		len;
	cl_uint		index = 0;
	cl_uint		i, j;
	cl_float   *dst;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	array_iter	iter;
	FmgrInfo	flinfo;
    FunctionCallInfoData fcinfo;

	if (OidIsValid(cast_func))
	{
		fmgr_info(cast_func, &flinfo);
		InitFunctionCallInfoData(fcinfo, &flinfo, 1, InvalidOid, NULL, NULL);
	}

	if (ARR_NDIM(array) > 2)
		elog(ERROR, "Unable to transform %d-dimensional array to matrix",
			 ARR_NDIM(array));
	else if (ARR_NDIM(array) == 2)
	{
		int	   *lbounds = ARR_LBOUND(array);
		int	   *dims = ARR_DIMS(array);

		/* fast path - if array is binary compatible to matrix */
		if (MATRIX_SANITYCHECK_NOERROR((MatrixType *) array))
			return (MatrixType *) array;

		/* make a matrix */
		y_offset = lbounds[0] - 1;
		x_offset = lbounds[1] - 1;
		y_nitems = dims[0];
		x_nitems = dims[1];
	}
	else if (ARR_NDIM(array) == 1)
	{
		int	   *lbounds = ARR_LBOUND(array);
		int	   *dims = ARR_DIMS(array);

		/* make a matrix as a representation of vector */
		y_offset = lbounds[0] - 1;
		x_offset = 0;
		y_nitems = dims[0];
		x_nitems = 1;
	}
	else
		elog(ERROR, "unexpected array dimension: %d", ARR_NDIM(array));

	len = offsetof(MatrixType, values[(Size)(x_offset + x_nitems) *
									  (Size)(y_offset + y_nitems)]);
	if (!AllocSizeIsValid(len))
		elog(ERROR, "matrix size too large");

	matrix = palloc(len);
	SET_VARSIZE(matrix, len);
	MATRIX_INIT_FIELDS(matrix, y_offset + y_nitems, x_offset + x_nitems);
	dst = matrix->values;

	/* Loop over the source array */
	array_iter_setup(&iter, (AnyArrayType *)array);
	get_typlenbyvalalign(ARR_ELEMTYPE(array),
						 &typlen, &typbyval, &typalign);
	if (y_offset > 0)
	{
		memset(dst, 0, sizeof(float) * y_offset * matrix->width);
		dst += y_offset * matrix->width;
	}
	for (j=0; j < y_nitems; j++)
	{
		if (x_offset > 0)
		{
			memset(dst, 0, sizeof(float) * x_offset);
			dst += x_offset;
		}

		for (i=0; i < x_nitems; i++)
		{
			Datum	datum;
			bool	isnull;

			datum = array_iter_next(&iter, &isnull, index,
									typlen, typbyval, typalign);
			if (isnull)
				*dst++ = 0.0;
			else if (!OidIsValid(cast_func))	/* binary compatible */
				*dst++ = DatumGetFloat4(datum);
			else
			{
				fcinfo.arg[0] = datum;
				fcinfo.argnull[0] = false;
				fcinfo.isnull = false;

				datum = FunctionCallInvoke(&fcinfo);
				if (fcinfo.isnull)
					*dst++ = 0.0;
				else
					*dst++ = DatumGetFloat4(datum);
			}
			index++;
		}
	}
	Assert(index == (Size)x_nitems * (Size)y_nitems);
	Assert((uintptr_t)dst == (uintptr_t)((char *)matrix + len));

	return matrix;
}

static ArrayType *
__matrix_to_array(MatrixType *matrix, Oid dest_type, Oid cast_func)
{
	ArrayType  *array;
	Size		i, nitems;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	Datum	   *elem_values;
	bool	   *elem_isnull;
	float	   *src;
	int			dims[2];
	int			lbounds[2];
	FmgrInfo	flinfo;
    FunctionCallInfoData fcinfo;

	MATRIX_SANITYCHECK(matrix);
	if (dest_type == FLOAT4OID)
		return (ArrayType *) matrix;	/* super fast path if float4[] */

	if (OidIsValid(cast_func))
	{
		fmgr_info(cast_func, &flinfo);
		InitFunctionCallInfoData(fcinfo, &flinfo, 1, InvalidOid, NULL, NULL);
	}
	get_typlenbyvalalign(dest_type, &typlen, &typbyval, &typalign);

	nitems = (Size)matrix->height * (Size)matrix->width;
	elem_values = palloc(sizeof(Datum) * nitems);
	elem_isnull = palloc(sizeof(bool) * nitems);

	src = (float *)ARR_DATA_PTR(matrix);
	for (i=0; i < nitems; i++)
	{
		Datum	newval;

		fcinfo.isnull = false;
		fcinfo.arg[0] = Float4GetDatum(*src);
		fcinfo.argnull[0] = false;
		newval = FunctionCallInvoke(&fcinfo);
		if (fcinfo.isnull)
			elem_isnull[i] = true;
		else
		{
			elem_isnull[i] = false;
			elem_values[i] = newval;
		}
		src++;
	}
	Assert(ARR_DATA_PTR(matrix) + sizeof(float) * nitems == (char *)src);
	dims[0] = matrix->height;
	dims[1] = matrix->width;
	lbounds[0] = 1;
	lbounds[1] = 1;

	array = construct_md_array(elem_values, elem_isnull, 2, dims, lbounds,
							   dest_type, typlen, typbyval, typalign);
	return array;
}

/*
 * matrix type in/out function
 */
Datum
matrix_in(PG_FUNCTION_ARGS)
{
	ArrayType  *array;

	array = (ArrayType *) OidFunctionCall3Coll(F_ARRAY_IN,
											   fcinfo->fncollation,
											   fcinfo->arg[0],
											   ObjectIdGetDatum(FLOAT4OID),
											   Int32GetDatum(-1));
	if (MATRIX_SANITYCHECK_NOERROR((MatrixType *)array))
		return PointerGetDatum(array);
	if (ARR_ELEMTYPE(array) != FLOAT4OID)
		elog(ERROR, "Bug? array is not float4[]");
	return PointerGetDatum(__array_to_matrix(array, InvalidOid));
}
PG_FUNCTION_INFO_V1(matrix_in);

Datum
matrix_out(PG_FUNCTION_ARGS)
{
	MatrixType *matrix = PG_GETARG_MATRIXTYPE_P(0);

	MATRIX_SANITYCHECK(matrix);

	return OidFunctionCall1Coll(F_ARRAY_OUT,
								fcinfo->fncollation,
								PointerGetDatum(matrix));
}
PG_FUNCTION_INFO_V1(matrix_out);

Datum
matrix_recv(PG_FUNCTION_ARGS)
{
	ArrayType  *array;

	array = (ArrayType *)OidFunctionCall3Coll(F_ARRAY_RECV,
											  fcinfo->fncollation,
											  fcinfo->arg[0],
											  ObjectIdGetDatum(FLOAT8OID),
											  Int32GetDatum(-1));
	if (MATRIX_SANITYCHECK_NOERROR((MatrixType *)array))
		return PointerGetDatum(array);
	if (ARR_ELEMTYPE(array) != FLOAT4OID)
		elog(ERROR, "Bug? array is not float4[]");
	return PointerGetDatum(__array_to_matrix(array, InvalidOid));
}
PG_FUNCTION_INFO_V1(matrix_recv);

Datum
matrix_send(PG_FUNCTION_ARGS)
{
	MatrixType *matrix = PG_GETARG_MATRIXTYPE_P(0);

	MATRIX_SANITYCHECK(matrix);

	return OidFunctionCall1Coll(F_ARRAY_SEND,
								fcinfo->fncollation,
								fcinfo->arg[0]);
}
PG_FUNCTION_INFO_V1(matrix_send);

Datum
float4array_to_matrix(PG_FUNCTION_ARGS)
{
	ArrayType  *array = PG_GETARG_ARRAYTYPE_P(0);
	PG_RETURN_MATRIXTYPE_P(__array_to_matrix(array, InvalidOid));
}
PG_FUNCTION_INFO_V1(float4array_to_matrix);

Datum
matrix_to_float4array(PG_FUNCTION_ARGS)
{
	MatrixType *matrix = PG_GETARG_MATRIXTYPE_P(0);
	PG_RETURN_ARRAYTYPE_P(__matrix_to_array(matrix, FLOAT4OID, InvalidOid));
}
PG_FUNCTION_INFO_V1(matrix_to_float4array);

Datum
float8array_to_matrix(PG_FUNCTION_ARGS)
{
	ArrayType  *array = PG_GETARG_ARRAYTYPE_P(0);
	PG_RETURN_MATRIXTYPE_P(__array_to_matrix(array, F_DTOF));
}
PG_FUNCTION_INFO_V1(float8array_to_matrix);

Datum
matrix_to_float8array(PG_FUNCTION_ARGS)
{
	MatrixType *matrix = PG_GETARG_MATRIXTYPE_P(0);
	PG_RETURN_ARRAYTYPE_P(__matrix_to_array(matrix, FLOAT8OID, F_FTOD));
}
PG_FUNCTION_INFO_V1(matrix_to_float8array);

Datum
numericarray_to_matrix(PG_FUNCTION_ARGS)
{
	ArrayType  *array = PG_GETARG_ARRAYTYPE_P(0);
	PG_RETURN_MATRIXTYPE_P(__array_to_matrix(array, F_NUMERIC_FLOAT4));
}
PG_FUNCTION_INFO_V1(numericarray_to_matrix);

Datum
matrix_to_numericarray(PG_FUNCTION_ARGS)
{
	MatrixType *matrix = PG_GETARG_MATRIXTYPE_P(0);
	PG_RETURN_ARRAYTYPE_P(__matrix_to_array(matrix, NUMERICOID,
											F_FLOAT4_NUMERIC));
}
PG_FUNCTION_INFO_V1(matrix_to_numericarray);

/*
 * make_matrix aggregate function
 */
typedef struct
{
	Oid		elemtype;	/* element type of the input array */
	cl_int	width;		/* maximum width of input vector */
	List   *rows;		/* list of the supplied vector */
} make_matrix_state;

Datum
make_matrix_accum(PG_FUNCTION_ARGS)
{
	make_matrix_state  *mstate;
	MemoryContext		aggcxt;
	MemoryContext		oldcxt;
	ArrayType		   *array;
	cl_uint				width;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");

	if (PG_ARGISNULL(1))
		elog(ERROR, "null-array was supplied");

	oldcxt = MemoryContextSwitchTo(aggcxt);
	array = PG_GETARG_ARRAYTYPE_P_COPY(1);

	/* sanity check */
	if (ARR_NDIM(array) != 1)
		elog(ERROR, "input array was not 1-dimension array");
	if (ARR_ELEMTYPE(array) != INT2OID &&
		ARR_ELEMTYPE(array) != INT4OID &&
		ARR_ELEMTYPE(array) != INT8OID &&
		ARR_ELEMTYPE(array) != FLOAT4OID &&
		ARR_ELEMTYPE(array) != FLOAT8OID &&
		ARR_ELEMTYPE(array) != NUMERICOID)
		elog(ERROR, "unsupported element type: %s",
			 format_type_be(ARR_ELEMTYPE(array)));

	width = ARR_LBOUND(array)[0] + ARR_DIMS(array)[0] - 1;

	if (PG_ARGISNULL(0))
	{
		mstate = palloc0(sizeof(make_matrix_state));
		mstate->elemtype = array->elemtype;
	}
	else
		mstate = (make_matrix_state *)PG_GETARG_POINTER(0);

	if (mstate->elemtype != array->elemtype)
		elog(ERROR, "element type mismatch!");

	mstate->width = Max(mstate->width, width);
	mstate->rows = lappend(mstate->rows, array);

	MemoryContextSwitchTo(oldcxt);

	PG_RETURN_POINTER(mstate);
}
PG_FUNCTION_INFO_V1(make_matrix_accum);

Datum
make_matrix_final(PG_FUNCTION_ARGS)
{
	make_matrix_state *mstate;
	MatrixType *matrix;
	ListCell   *lc;
	Oid			cast_fnoid;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	array_iter	iter;
	FmgrInfo	cast_flinfo;
	FunctionCallInfoData cast_fcinfo;
	Size		width;
	Size		height;
	Size		len;
	Size		row_index;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	mstate = (make_matrix_state *)PG_GETARG_POINTER(0);

	switch (mstate->elemtype)
	{
		case INT2OID:
			cast_fnoid = F_I2TOF;
			break;
		case INT4OID:
			cast_fnoid = F_I4TOF;
			break;
		case INT8OID:
			cast_fnoid = F_I8TOF;
			break;
		case FLOAT4OID:
			cast_fnoid = InvalidOid;
			break;
		case FLOAT8OID:
			cast_fnoid = F_DTOF;
			break;
		case NUMERICOID:
			cast_fnoid = F_NUMERIC_FLOAT4;
			break;
		default:
			elog(ERROR, "matrix does not support %s as element type",
				 format_type_be(mstate->elemtype));
	}

	if (OidIsValid(cast_fnoid))
	{
		fmgr_info(cast_fnoid, &cast_flinfo);
		InitFunctionCallInfoData(cast_fcinfo, &cast_flinfo,
								 1, InvalidOid, NULL, NULL);
	}

	width = mstate->width;
	height = list_length(mstate->rows);
	len = sizeof(MatrixType) + sizeof(float) * width * height;
	if (!AllocSizeIsValid(len))
		elog(ERROR, "supplied matrix is too big");
	matrix = palloc(len);
	SET_VARSIZE(matrix, len);
	MATRIX_INIT_FIELDS(matrix, height, width);

	get_typlenbyvalalign(mstate->elemtype, &typlen, &typbyval, &typalign);

	row_index = 0;
	foreach (lc, mstate->rows)
	{
		ArrayType  *array = lfirst(lc);
		Size		offset = ARR_LBOUND(array)[0] - 1;
		Size		i, nitems = ARR_DIMS(array)[0];
		cl_float   *dst = matrix->values + row_index;
		Datum		datum;
		bool		isnull;

		/* sanity checks */
		Assert(ARR_ELEMTYPE(array) == mstate->elemtype &&
			   ARR_NDIM(array) == 1 &&
			   offset + nitems <= mstate->width);
		if (offset > 0)
		{
			for (i=0; i < offset; i++)
			{
				*dst = 0.0;
				dst += height;
			}
		}

		array_iter_setup(&iter, (AnyArrayType *)array);
		for (i=0; i < nitems; i++)
		{
			datum = array_iter_next(&iter, &isnull, i,
									typlen, typbyval, typalign);
			if (isnull)
				*dst = 0.0;
			else if (!OidIsValid(cast_fnoid))
				*dst = DatumGetFloat4(datum);
			else
			{
				cast_fcinfo.arg[0] = datum;
				cast_fcinfo.argnull[0] = false;
				cast_fcinfo.isnull = false;

				datum = FunctionCallInvoke(&cast_fcinfo);
				if (cast_fcinfo.isnull)
					*dst = 0.0;
				else
					*dst = DatumGetFloat4(datum);
			}
			dst += height;
		}

		for (i=offset + nitems; i < width; i++)
		{
			*dst = 0.0;
			dst += height;
		}
		pfree(array);
		row_index++;
	}
	PG_RETURN_POINTER(matrix);
}
PG_FUNCTION_INFO_V1(make_matrix_final);

/*
 * Get properties of matrix
 */
Datum
matrix_height(PG_FUNCTION_ARGS)
{
	MatrixType *X = PG_GETARG_MATRIXTYPE_P(0);

	MATRIX_SANITYCHECK(X);

	PG_RETURN_INT32(X->height);
}
PG_FUNCTION_INFO_V1(matrix_height);

Datum
matrix_width(PG_FUNCTION_ARGS)
{
	MatrixType *X = PG_GETARG_MATRIXTYPE_P(0);

	MATRIX_SANITYCHECK(X);

	PG_RETURN_INT32(X->width);
}
PG_FUNCTION_INFO_V1(matrix_width);

Datum
matrix_rawsize(PG_FUNCTION_ARGS)
{
	int32	height = PG_GETARG_INT32(0);
	int32	width = PG_GETARG_INT32(1);

	PG_RETURN_INT64(offsetof(MatrixType,
							 values[(Size)height * (Size)width]));
}
PG_FUNCTION_INFO_V1(matrix_rawsize);

/*
 * matrix_transpose
 */
Datum
matrix_transpose(PG_FUNCTION_ARGS)
{
	MatrixType *X = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType *T;
	Size		len;
	Size		i, nitems;

	MATRIX_SANITYCHECK(X);

	/* make a new matrix */
	len = offsetof(MatrixType,
				   values[(Size)X->width * (Size)X->height]);
	if (!AllocSizeIsValid(len))
		elog(ERROR, "matrix size too large");
	T = palloc(len);
	SET_VARSIZE(T, len);
	MATRIX_INIT_FIELDS(T, X->width, X->height);

	nitems = (Size)X->height * (Size)X->width;
	for (i=0; i < nitems; i++)
	{
		T->values[(i % X->height) * T->height +
				  (i / X->height)] = X->values[i];
	}
	PG_RETURN_MATRIXTYPE_P(T);
}
PG_FUNCTION_INFO_V1(matrix_transpose);

/*
 * matrix_add
 */
Datum
matrix_add(PG_FUNCTION_ARGS)
{
	MatrixType *X = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType *Y = PG_GETARG_MATRIXTYPE_P(1);
	MatrixType *R;
	Size		i, nitems;
	Size		len;
	float	   *xval;
	float	   *yval;
	float	   *rval;

	/* sanity check */
	MATRIX_SANITYCHECK(X);
	MATRIX_SANITYCHECK(Y);
	if (X->height != Y->height || X->width != Y->width)
		elog(ERROR, "matrix size mismatch (%u,%u) + (%u,%u)",
			 X->height, X->width, Y->height, Y->width);
	/* make a new matrix */
	len = offsetof(MatrixType,
				   values[(Size)X->height * (Size)X->width]);
	if (!AllocSizeIsValid(len))
		elog(ERROR, "matrix size too large");
	R = palloc(len);
	SET_VARSIZE(R, len);
	MATRIX_INIT_FIELDS(R, X->height, X->width);

	nitems = X->height * X->width;
	xval = X->values;
	yval = Y->values;
	rval = R->values;
	for (i=0; i < nitems; i++, xval++, yval++, rval++)
		*rval = *xval + *yval;

	PG_RETURN_MATRIXTYPE_P(R);
}
PG_FUNCTION_INFO_V1(matrix_add);

/*
 * matrix_sub
 */
Datum
matrix_sub(PG_FUNCTION_ARGS)
{
	MatrixType *X = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType *Y = PG_GETARG_MATRIXTYPE_P(1);
	MatrixType *R;
	Size		i, nitems;
	Size		len;
	float	   *xval;
	float	   *yval;
	float	   *rval;

	/* sanity check */
	MATRIX_SANITYCHECK(X);
	MATRIX_SANITYCHECK(Y);
	if (X->height != Y->height || X->width != Y->width)
		elog(ERROR, "matrix size mismatch (%u,%u) - (%u,%u)",
			 X->height, X->width, Y->height, Y->width);
	/* make a new matrix */
	len = offsetof(MatrixType,
				   values[(Size)X->height * (Size)X->width]);
	if (!AllocSizeIsValid(len))
		elog(ERROR, "matrix size too large");
	R = palloc(len);
	SET_VARSIZE(R, len);
	MATRIX_INIT_FIELDS(R, X->height, X->width);

	nitems = X->height * X->width;
	xval = X->values;
	yval = Y->values;
	rval = R->values;
	for (i=0; i < nitems; i++, xval++, yval++, rval++)
		*rval = *xval - *yval;

	PG_RETURN_MATRIXTYPE_P(R);
}
PG_FUNCTION_INFO_V1(matrix_sub);

/*
 * matrix_mul
 */
Datum
matrix_mul(PG_FUNCTION_ARGS)
{
	MatrixType *X = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType *Y = PG_GETARG_MATRIXTYPE_P(1);
	MatrixType *R;
	Size		len;
	cl_uint		nloops;
	cl_uint		i, j, k;
	float	   *dst;

	/* sanity check */
	MATRIX_SANITYCHECK(X);
	MATRIX_SANITYCHECK(Y);
	if (X->width != Y->height)
		elog(ERROR, "matrix size mismatch (%u,%u) - (%u,%u)",
			 X->height, X->width, Y->height, Y->width);
	nloops = X->width;

	/* make a new matrix */
	len = offsetof(MatrixType,
				   values[(Size)X->height * (Size)Y->width]);
	if (!AllocSizeIsValid(len))
		elog(ERROR, "matrix size too large");
	R = palloc(len);
	SET_VARSIZE(R, len);
	MATRIX_INIT_FIELDS(R, X->height, Y->width);

	dst = R->values;
	for (j=0; j < X->height; j++)
	{
		for (i=0; i < Y->width; i++)
		{
			double	sum = 0.0;
			float  *x_val = X->values + j * X->width;
			float  *y_val = Y->values + i;

			for (k=0; k < nloops; k++)
			{
				sum += *x_val * *y_val;

				x_val++;
				y_val += Y->width;
			}
			*dst++ = (float)sum;
		}
	}
	PG_RETURN_MATRIXTYPE_P(R);
}
PG_FUNCTION_INFO_V1(matrix_mul);
