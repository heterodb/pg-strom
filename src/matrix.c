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
#include "catalog/pg_cast.h"
#include "catalog/pg_type.h"
#include "utils/array.h"
#include "utils/arrayaccess.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
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

#define MATRIX_SANITYCHECK_NOERROR(matrix)			\
	(((matrix)->ndim == 2 &&						\
	  (matrix)->dataoffset == 0 &&					\
	  (matrix)->elemtype == FLOAT4OID &&			\
	  (matrix)->lbound1 == 1 &&						\
	  (matrix)->lbound2 == 1) ? true : false)
#define MATRIX_SANITYCHECK(matrix)					\
	do {											\
		if (!MATRIX_SANITYCHECK_NOERROR((matrix)))	\
			elog(ERROR, "Invalid matrix format");	\
	} while(0)

/*
 * Type cast functions
 */
static MatrixType *
__array_to_matrix(ArrayType *array)
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
	Oid			cast_func = InvalidOid;
	char		cast_method = COERCION_METHOD_BINARY;
	FmgrInfo	cast_flinfo;
	FunctionCallInfoData cast_fcinfo;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	array_iter	iter;

	if (ARR_ELEMTYPE(array) != FLOAT4OID)
	{
		HeapTuple		tuple;
		Form_pg_cast	castForm;

		tuple = SearchSysCache2(CASTSOURCETARGET,
								ObjectIdGetDatum(ARR_ELEMTYPE(array)),
								ObjectIdGetDatum(FLOAT4OID));
		if (!HeapTupleIsValid(tuple))
			elog(ERROR, "failed to lookup cast %s to %s",
				 format_type_be(ARR_ELEMTYPE(array)),
				 format_type_be(FLOAT4OID));
		castForm = (Form_pg_cast) GETSTRUCT(tuple);
		cast_func = castForm->castfunc;
		cast_method = castForm->castmethod;
		Assert(OidIsValid(cast_func) || cast_method == COERCION_METHOD_BINARY);
		ReleaseSysCache(tuple);

		if (OidIsValid(cast_func))
		{
			fmgr_info(cast_func, &cast_flinfo);
			InitFunctionCallInfoData(cast_fcinfo, &cast_flinfo,
									 1, InvalidOid, NULL, NULL);
		}
	}

	if (ARR_NDIM(array) > 2)
		elog(ERROR, "Unable to transform %d-dimensional array to matrix",
			 ARR_NDIM(array));
	else if (ARR_NDIM(array) == 2)
	{
		int	   *lbounds = ARR_LBOUND(array);
		int	   *dims = ARR_DIMS(array);

		/*
		 * fast path - if supplied array is entirely compatible to matrix,
		 * we don't need to transform the data.
		 */
		if (ARR_ELEMTYPE(array) == FLOAT4OID &&
			!ARR_HASNULL(array) &&
			lbounds[0] == 1 &&
			lbounds[1] == 1)
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
			else if (cast_method == COERCION_METHOD_BINARY)
				*dst++ = DatumGetFloat4(datum);
			else
			{
				Datum	newval;

				cast_fcinfo.isnull = false;
				cast_fcinfo.arg[0] = datum;
				cast_fcinfo.argnull[0] = false;

				newval = FunctionCallInvoke(&cast_fcinfo);
				if (cast_fcinfo.isnull)
					*dst++ = 0.0;
				else
					*dst++ = DatumGetFloat4(newval);
			}
			index++;
		}
	}
	Assert(index == (Size)x_nitems * (Size)y_nitems);
	Assert((uintptr_t)dst == (uintptr_t)((char *)matrix + len));

	return matrix;
}

static ArrayType *
__matrix_to_array(MatrixType *matrix, Oid dest_type)
{
	ArrayType  *array;
	Size		i, nitems;
	Size		len;
	int			typlen;
	float	   *src;
	Oid			cast_func = InvalidOid;
	char		cast_method = COERCION_METHOD_BINARY;
	FmgrInfo	cast_flinfo;
	FunctionCallInfoData cast_fcinfo;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	MATRIX_SANITYCHECK(matrix);
	if (dest_type == FLOAT4OID)
		return (ArrayType *) matrix;	/* super fast path if float4[] */
	// check binary comatible

	// if function cast, setup cast function
	// elsewhere, use in/out function
	// walk on the source matrix then set up values/isnull
	// finally construct_md_array()

	else
	{
		HeapTuple		tuple;
		Form_pg_cast	castForm;

		tuple = SearchSysCache2(CASTSOURCETARGET,
                                ObjectIdGetDatum(FLOAT4OID),
								ObjectIdGetDatum(dest_type));
		if (!HeapTupleIsValid(tuple))
            elog(ERROR, "failed to lookup cast %s to %s",
				 format_type_be(FLOAT4OID),
				 format_type_be(dest_type));
		castForm = (Form_pg_cast) GETSTRUCT(tuple);
		cast_func = castForm->castfunc;
		cast_method = castForm->castmethod;
		Assert(OidIsValid(cast_func) || cast_method == COERCION_METHOD_BINARY);
		ReleaseSysCache(tuple);

		if (OidIsValid(cast_func))
		{
			fmgr_info(cast_func, &cast_flinfo);
			InitFunctionCallInfoData(cast_fcinfo, &cast_flinfo,
									 1, InvalidOid, NULL, NULL);
		}
	}
	get_typlenbyvalalign(dest_type, &typlen, &typbyval, &typalign);
	nitems = (Size)matrix->height * (Size)matrix->width;

	if (cast_method == COERCION_METHOD_BINARY)
	{
		len = sizeof(ArrayType) + 4 * sizeof(int) + typlen * nitems;
		if (!AllocSizeIsValid(len))
			elog(ERROR, "array size too large");
		array = palloc0(len);
		SET_VARSIZE(array,len);
		array->ndim = 2;
		array->dataoffset = 0;
		array->elemtype = dest_type;
		ARR_DIMS(array)[0] = matrix->height;
		ARR_DIMS(array)[1] = matrix->width;
		ARR_LBOUND(array)[0] = 1;
		ARR_LBOUND(array)[1] = 1;

		Assert(typlen == sizeof(float));
		memcpy(ARR_DATA_PTR(array), matrix->values, sizeof(float) * nitems);
	}
	else
	{
		Datum  *values = palloc(sizeof(Datum) * nitems);
		bool   *isnull = palloc(sizeof(bool) * nitems);
		float  *src = (float *)ARR_DATA_PTR(matrix);
		int		dims[2];
		int		lbounds[2];

		for (i=0; i < nitems; i++)
		{
			Datum	newval;

			cast_fcinfo.isnull = false;
			cast_fcinfo.arg[0] = Float4GetDatum(*src);
			cast_fcinfo.argnull[0] = false;
			newval = FunctionCallInvoke(&cast_fcinfo);
			if (cast_fcinfo.isnull)
				isnull[i] = true;
			else
			{
				isnull[i] = false;
				values[i] = newval;
			}
			src++;
		}
		dims[0] = matrix->height;
		dims[1] = matrix->width;
		lbounds[0] = matrix->lbound1;
		lbounds[1] = matrix->lbound2;

		array = construct_md_array(values, isnull, 2, dims, lbounds,
								   dest_type, typlen, typbyval, typalign);
	}
	return array;
}

/*
 * matrix type in/out function
 */
Datum
matrix_in(PG_FUNCTION_ARGS)
{
	Datum	array;

	elog(INFO, "matrix_in arg %lu %lu %lu",
		 fcinfo->arg[0], fcinfo->arg[1], fcinfo->arg[2]);

	array = OidFunctionCall3Coll(F_ARRAY_IN,
								 fcinfo->fncollation,
								 fcinfo->arg[0],
								 ObjectIdGetDatum(FLOAT8OID),
								 Int32GetDatum(-1));
	if (MATRIX_SANITYCHECK_NOERROR((MatrixType *)array))
		return array;

	return PointerGetDatum(array_to_matrix((ArrayType *) array));
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
	Datum   array;

	elog(INFO, "matrix_recv arg %lu %lu %lu",
		 fcinfo->arg[0], fcinfo->arg[1], fcinfo->arg[2]);

	array = OidFunctionCall3Coll(F_ARRAY_RECV,
								 fcinfo->fncollation,
								 fcinfo->arg[0],
								 fcinfo->arg[1],
								 fcinfo->arg[2]);
	if (MATRIX_SANITYCHECK_NOERROR((MatrixType *)array))
		return array;

	return PointerGetDatum(array_to_matrix((ArrayType *) array));
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
anyarray_to_matrix(PG_FUNCTION_ARGS)
{
	ArrayType  *array = PG_GETARG_ARRAYTYPE_P(0);
	elog(INFO, "%s", __FUNCTION__);
	PG_RETURN_MATRIXTYPE_P(__array_to_matrix(array));
}
PG_FUNCTION_INFO_V1(anyarray_to_matric);

Datum
matrix_to_anyarray(PG_FUNCTION_ARGS)
{
	MatrixType *matrix = PG_GETARG_MATRIXTYPE_P(0);
	Oid			fn_oid = fcinfo->flinfo->fn_oid;
	Oid			rettype = get_func_rettype(fn_oid);
	Oid			elemtype = get_element_type(rettype);

	elog(INFO, "%s", __FUNCTION__);
	PG_RETURN_ARRAYTYPE_P(__matrix_to_array(matrix, elemtype));
}
PG_FUNCTION_INFO_V1(matrix_to_anyarray);

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
	if (ARR_ELEMTYPE(array) != FLOAT4OID &&
		ARR_ELEMTYPE(array) != FLOAT8OID)
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
	make_matrix_state  *mstate;
	MatrixType		   *matrix;
	ListCell		   *lc;
	cl_float		   *dst;
	cl_int				typlen;
	Size				len;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	mstate = (make_matrix_state *)PG_GETARG_POINTER(0);
	if (mstate->elemtype == FLOAT4OID)
		typlen = sizeof(cl_float);
	else if (mstate->elemtype == FLOAT8OID)
		typlen = sizeof(cl_double);
	else
		elog(ERROR, "matrix does not support %s as element type",
			 format_type_be(mstate->elemtype));

	len = sizeof(MatrixType) +
		typlen * (Size)mstate->width * (Size)list_length(mstate->rows);
	if (!AllocSizeIsValid(len))
		elog(ERROR, "supplied matrix is too big");
	matrix = palloc(len);
	SET_VARSIZE(matrix, len);
	MATRIX_INIT_FIELDS(matrix, list_length(mstate->rows), mstate->width);

	dst = matrix->values;
	foreach (lc, mstate->rows)
	{
		ArrayType  *array = lfirst(lc);
		cl_uint		offset = ARR_LBOUND(array)[0] - 1;
		cl_uint		i, nitems = ARR_DIMS(array)[0];

		/* sanity checks */
		Assert((ARR_ELEMTYPE(array) == FLOAT4OID ||
				ARR_ELEMTYPE(array) == FLOAT8OID) &&
			   ARR_NDIM(array) == 1 &&
			   offset + nitems <= mstate->width);

		if (offset > 0)
		{
			memset(dst, 0, sizeof(float) * offset);
			dst += offset;
		}

		if (mstate->elemtype == FLOAT4OID)
		{
			/* no need to transform, if float4 */
			memcpy(dst, ARR_DATA_PTR(array), sizeof(float) * nitems);
			dst += nitems;
		}
		else if (mstate->elemtype == FLOAT8OID)
		{
			double *src = (double *) ARR_DATA_PTR(array);

			for (i=0; i < nitems; i++)
			{
				double	oldval = *src++;
				float	newval = (float)oldval;

				if (!isinf(oldval) && isinf(newval))
					ereport(ERROR,
							(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
							 errmsg("value out of range: overflow")));
				if (oldval != 0.0 && newval == 0.0)
					ereport(ERROR,
							(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
							 errmsg("value out of range: underflow")));
				*dst++ = newval;
			}
		}
		else
			elog(ERROR, "unexpected element type");

		if (offset + nitems < mstate->width)
		{
			cl_uint		n = mstate->width - (offset + nitems);

			memset(dst, 0, sizeof(float) * n);
			dst += n;
		}

		Assert((dst - matrix->values) % mstate->width == 0);
		pfree(array);
	}
	PG_RETURN_POINTER(matrix);
}
PG_FUNCTION_INFO_V1(make_matrix_final);

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
