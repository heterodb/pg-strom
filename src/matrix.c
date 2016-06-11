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
#include "funcapi.h"
#include "utils/array.h"
#include "utils/arrayaccess.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/syscache.h"
#include "utils/varbit.h"
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

/*
 * Constructor of Matrix-like Array
 */
typedef struct
{
	Oid			elemtype;	/* element type of the input array */
	cl_uint		width;		/* max width of the input vector */
	List	   *rows;		/* list of the supplied vector */
} array_matrix_state;

Datum
array_matrix_accum(PG_FUNCTION_ARGS)
{
	array_matrix_state *amstate;
	MemoryContext	aggcxt;
	MemoryContext	oldcxt;
	ArrayType	   *array;
	cl_uint			width;

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
		ARR_ELEMTYPE(array) != FLOAT8OID)
		elog(ERROR, "unsupported element type: %s",
			 format_type_be(ARR_ELEMTYPE(array)));

	width = ARR_LBOUND(array)[0] + ARR_DIMS(array)[0] - 1;

	if (PG_ARGISNULL(0))
	{
		amstate = palloc0(sizeof(array_matrix_state));
		amstate->elemtype = array->elemtype;
	}
	else
		amstate = (array_matrix_state *)PG_GETARG_POINTER(0);

	amstate->width = Max(amstate->width, width);
	amstate->rows = lappend(amstate->rows, array);

	MemoryContextSwitchTo(oldcxt);

	PG_RETURN_POINTER(amstate);
}
PG_FUNCTION_INFO_V1(array_matrix_accum);

Datum
array_matrix_accum_varbit(PG_FUNCTION_ARGS)
{
	array_matrix_state *amstate;
	MemoryContext	aggcxt;
	MemoryContext	oldcxt;
	ArrayType	   *array;
	Datum		   *values;
	Size			i, nitems;
	int16			typlen;
	bool			typbyval;
	char			typalign;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	get_typlenbyvalalign(INT4OID, &typlen, &typbyval, &typalign);

	if (PG_ARGISNULL(1))
	{
		oldcxt = MemoryContextSwitchTo(aggcxt);
		/* an empty int32 array */
		array = construct_array(NULL, 0, INT4OID,
								typlen, typbyval, typalign);
		nitems = 0;
	}
	else
	{
		VarBit	   *varbit = PG_GETARG_VARBIT_P(1);
		Size		pos;

		nitems = (varbit->bit_len / BITS_PER_BYTE +
				  sizeof(cl_int) - 1) / sizeof(cl_int);
		values = palloc0(sizeof(Datum) * nitems);
		for (i=0, pos = sizeof(cl_int) * BITS_PER_BYTE;
			 i < nitems;
			 i++, pos += sizeof(cl_int) * BITS_PER_BYTE)
		{
			cl_int	code = ((cl_int *)varbit->bit_dat)[i];
			cl_int	shift;

			if (varbit->bit_len < pos)
			{
				Assert(pos - varbit->bit_len < sizeof(cl_int) * BITS_PER_BYTE);
				shift = (sizeof(cl_int) * BITS_PER_BYTE -
						 (pos - varbit->bit_len));
				code &= (1UL << shift) - 1;
			}
			values[i] = Int32GetDatum(code);
		}
		oldcxt = MemoryContextSwitchTo(aggcxt);
		array = construct_array(values, nitems, INT4OID,
								typlen, typbyval, typalign);
	}

	if (PG_ARGISNULL(0))
	{
		amstate = palloc0(sizeof(array_matrix_state));
		amstate->elemtype = INT4OID;
	}
	else
		amstate = (array_matrix_state *)PG_GETARG_POINTER(0);

	amstate->width = Max(amstate->width, nitems);
	amstate->rows = lappend(amstate->rows, array);

	MemoryContextSwitchTo(oldcxt);

	PG_RETURN_POINTER(amstate);
}
PG_FUNCTION_INFO_V1(array_matrix_accum_varbit);

#define ARRAY_MATRIX_FINAL_TEMPLATE(R,amstate,BASETYPE)					\
	do {																\
		Size		width = (amstate)->width;							\
		Size		height = list_length((amstate)->rows);				\
		Size		length;												\
		Size		row_index;											\
		int16		typlen;												\
		bool		typbyval;											\
		char		typalign;											\
		ListCell   *lc;													\
																		\
		get_typlenbyvalalign((amstate)->elemtype,						\
							 &typlen, &typbyval, &typalign);			\
		Assert(typlen == sizeof(BASETYPE));								\
		length = offsetof(MatrixType,									\
						  values[(Size)typlen * width * height]);		\
		if (!AllocSizeIsValid(length))									\
			elog(ERROR, "supplied array-matrix is too big");			\
		R = palloc(length);												\
		SET_VARSIZE(R, length);											\
		INIT_ARRAY_MATRIX(R, (amstate)->elemtype, height, width);		\
																		\
		row_index = 0;													\
		foreach (lc, (amstate)->rows)									\
		{																\
			ArrayType  *array = lfirst(lc);								\
			Size		offset = ARR_LBOUND(array)[0] - 1;				\
			Size		i, nitems = ARR_DIMS(array)[0];					\
			BASETYPE   *dest;											\
			array_iter	iter;											\
			Datum		datum;											\
			Datum		mask;											\
			bool		isnull;											\
																		\
			/* sanity checks */											\
			Assert(ARR_ELEMTYPE(array) == (amstate)->elemtype &&		\
				   ARR_NDIM(array) == 1);								\
			dest = ((BASETYPE *)(R)->values) + row_index;				\
			mask = (sizeof(BASETYPE) < sizeof(Datum)					\
					? ((1UL << (sizeof(BASETYPE) * 8)) - 1)				\
					: ~0UL);											\
			for (i=0; i < offset; i++, dest += height)					\
				*dest = 0;												\
			array_iter_setup(&iter, (AnyArrayType *)array);				\
			for (i=0; i < nitems; i++, dest += height)					\
			{															\
				datum = array_iter_next(&iter, &isnull, i,				\
										typlen, typbyval, typalign);	\
				if (isnull)												\
					datum = 0;											\
				*dest = (BASETYPE)(datum & mask);						\
			}															\
																		\
			for (i = offset + nitems; i < width; i++, dest += height)	\
				*dest = 0;												\
																		\
			row_index++;												\
		}																\
	} while(0)

Datum
array_matrix_final_int2(PG_FUNCTION_ARGS)
{
	array_matrix_state *amstate;
	MatrixType *R;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	amstate = (array_matrix_state *)PG_GETARG_POINTER(0);
	Assert(amstate->elemtype == INT2OID);
	ARRAY_MATRIX_FINAL_TEMPLATE(R,amstate,cl_ushort);
	PG_RETURN_POINTER(R);
}
PG_FUNCTION_INFO_V1(array_matrix_final_int2);

Datum
array_matrix_final_int4(PG_FUNCTION_ARGS)
{
	array_matrix_state *amstate;
	MatrixType *R;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	amstate = (array_matrix_state *)PG_GETARG_POINTER(0);
	Assert(amstate->elemtype == INT4OID);
	ARRAY_MATRIX_FINAL_TEMPLATE(R,amstate,cl_uint);
	PG_RETURN_POINTER(R);
}
PG_FUNCTION_INFO_V1(array_matrix_final_int4);

Datum
array_matrix_final_int8(PG_FUNCTION_ARGS)
{
	array_matrix_state *amstate;
	MatrixType *R;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	amstate = (array_matrix_state *)PG_GETARG_POINTER(0);
	Assert(amstate->elemtype == INT8OID);
	ARRAY_MATRIX_FINAL_TEMPLATE(R,amstate,cl_ulong);
	PG_RETURN_POINTER(R);
}
PG_FUNCTION_INFO_V1(array_matrix_final_int8);

Datum
array_matrix_final_float4(PG_FUNCTION_ARGS)
{
	array_matrix_state *amstate;
	MatrixType *R;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	amstate = (array_matrix_state *)PG_GETARG_POINTER(0);
	Assert(amstate->elemtype == FLOAT4OID);
	ARRAY_MATRIX_FINAL_TEMPLATE(R,amstate,cl_uint);
	PG_RETURN_POINTER(R);
}
PG_FUNCTION_INFO_V1(array_matrix_final_float4);

Datum
array_matrix_final_float8(PG_FUNCTION_ARGS)
{
	array_matrix_state *amstate;
	MatrixType *R;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	amstate = (array_matrix_state *)PG_GETARG_POINTER(0);
	Assert(amstate->elemtype == FLOAT8OID);
	ARRAY_MATRIX_FINAL_TEMPLATE(R,amstate,cl_ulong);
	PG_RETURN_POINTER(R);
}
PG_FUNCTION_INFO_V1(array_matrix_final_float8);

/*
 * Validator of matrix-like array
 */
Datum
array_matrix_validation(PG_FUNCTION_ARGS)
{
	MatrixType *matrix = PG_GETARG_MATRIXTYPE_P(0);

	PG_RETURN_BOOL(VALIDATE_ARRAY_MATRIX(matrix));
}
PG_FUNCTION_INFO_V1(array_matrix_validation);

/*
 * Size estimator
 */
Datum
array_matrix_rawsize(PG_FUNCTION_ARGS)
{
	Oid			elemtype = PG_GETARG_OID(0);
	int32		height = PG_GETARG_INT32(1);
	int32		width = PG_GETARG_INT32(2);
	int16		typlen;
	Size		length;

	switch (elemtype)
	{
		case INT2OID:
		case INT4OID:
		case INT8OID:
		case FLOAT4OID:
		case FLOAT8OID:
			typlen = get_typlen(elemtype);
			break;
		default:
			elog(ERROR, "unable to make array-matrix with '%s' type",
				 format_type_be(elemtype));
	}
	length = MAXALIGN(offsetof(MatrixType, values) +
					  (Size)typlen * (Size)height * (Size)width);
	PG_RETURN_INT64(length);
}
PG_FUNCTION_INFO_V1(array_matrix_rawsize);

Datum
array_matrix_height(PG_FUNCTION_ARGS)
{
	MatrixType *M = PG_GETARG_MATRIXTYPE_P(0);

	if (VARATT_IS_EXPANDED_HEADER(M) ||
		!VALIDATE_ARRAY_MATRIX(M))
		elog(ERROR, "not a matrix-like array");
	PG_RETURN_INT32(M->height);
}
PG_FUNCTION_INFO_V1(array_matrix_height);

Datum
array_matrix_width(PG_FUNCTION_ARGS)
{
	MatrixType *M = PG_GETARG_MATRIXTYPE_P(0);

	if (VARATT_IS_EXPANDED_HEADER(M) ||
		!VALIDATE_ARRAY_MATRIX(M))
		elog(ERROR, "not a matrix-like array");
	PG_RETURN_INT32(M->width);
}
PG_FUNCTION_INFO_V1(array_matrix_width);

Datum
array_matrix_unnest(PG_FUNCTION_ARGS)
{
	struct {
		MatrixType	   *matrix;
		TupleTableSlot *slot;
		int16			typlen;
		bool			typbyval;
		char			typalign;
	}				   *state;
	FuncCallContext	   *fncxt;
	MatrixType		   *matrix;
	TupleTableSlot	   *slot;
	HeapTuple			tuple;
	cl_int				i;
	char			   *source;

	if (SRF_IS_FIRSTCALL())
	{
		TupleDesc		tupdesc;
		MemoryContext	oldcxt;
		MatrixType	   *matrix;

		fncxt = SRF_FIRSTCALL_INIT();
		oldcxt = MemoryContextSwitchTo(fncxt->multi_call_memory_ctx);
		state = palloc0(sizeof(*state));
		matrix = PG_GETARG_MATRIXTYPE_P(0);

		/*
		 * TODO: Allow general 1D/2D array to unnest
		 */
		if (VARATT_IS_EXPANDED_HEADER(matrix))
			elog(ERROR, "ExpandedArrayHeader is not supported");
		if (!VALIDATE_ARRAY_MATRIX(matrix))
			elog(ERROR, "Not a matrix-like array");
		get_typlenbyvalalign(matrix->elemtype,
							 &state->typlen,
							 &state->typbyval,
							 &state->typalign);
		tupdesc = CreateTemplateTupleDesc(matrix->width, false);
		for (i=0; i < matrix->width; i++)
		{
			TupleDescInitEntry(tupdesc,
							   (AttrNumber) i+1,
							   psprintf("c%u", i+1),
							   matrix->elemtype, -1, 0);
		}
		fncxt->tuple_desc = BlessTupleDesc(tupdesc);

		state->matrix = matrix;
		state->slot = MakeSingleTupleTableSlot(fncxt->tuple_desc);
		fncxt->user_fctx = state;

		MemoryContextSwitchTo(oldcxt);
	}
	fncxt = SRF_PERCALL_SETUP();
	state = fncxt->user_fctx;
	matrix = state->matrix;
	slot = state->slot;

	if (fncxt->call_cntr >= matrix->height)
		SRF_RETURN_DONE(fncxt);

	source = matrix->values + state->typlen * fncxt->call_cntr;
	ExecClearTuple(slot);
	memset(slot->tts_isnull, 0, sizeof(bool) * matrix->width);
	for (i=0; i < matrix->width; i++)
	{
		switch (state->typlen)
		{
			case sizeof(cl_ushort):
				slot->tts_values[i] = *((cl_ushort *)source);
				source += sizeof(cl_ushort) * matrix->height;
				break;
			case sizeof(cl_uint):
				slot->tts_values[i] = *((cl_uint *)source);
				source += sizeof(cl_uint) * matrix->height;
				break;
			case sizeof(cl_ulong):
				slot->tts_values[i] = *((cl_ulong *)source);
				source += sizeof(cl_ulong) * matrix->height;
				break;
			default:
				elog(ERROR, "unexpecter type length: %d", state->typlen);
		}
	}
	ExecStoreVirtualTuple(slot);

	tuple = ExecMaterializeSlot(slot);
	SRF_RETURN_NEXT(fncxt, HeapTupleGetDatum(tuple));
}
PG_FUNCTION_INFO_V1(array_matrix_unnest);

/*
 * rbind that takes two arrays
 */
static MatrixType *
array_martix_rbind(Oid elemtype, MatrixType *X, MatrixType *Y)
{
	MatrixType *R;
	cl_uint		height;
	cl_uint		width;
	int			typlen;
	Size		length;
	int			i;
	char	   *src, *dst;

	/* sanity checks */
	if (VARATT_IS_EXPANDED_HEADER(X) || VARATT_IS_EXPANDED_HEADER(Y))
		elog(ERROR, "ExpandedArrayHeader is not supported");
	if (!VALIDATE_ARRAY_MATRIX(X) || !VALIDATE_ARRAY_MATRIX(Y))
		elog(ERROR, "Not a matrix-like array");
	if (elemtype != X->elemtype || elemtype != Y->elemtype)
		elog(ERROR, "Bug? not expected type");
	typlen = get_typlen(elemtype);

	width = Max(X->width, Y->width);
	height = X->height + Y->height;
	length = offsetof(MatrixType,
					  values[typlen * (Size)width * (Size)height]);
	R = palloc(length);
	SET_VARSIZE(R, length);
	INIT_ARRAY_MATRIX(R, elemtype, height, width);

	/* copy from the top-matrix */
	for (i=0, dst = R->values, src = X->values;
		 i < width;
		 i++, dst += typlen * height, src += typlen * X->height)
	{
		if (i < X->width)
			memcpy(dst, src, typlen * X->height);
		else
			memset(dst, 0, typlen * X->height);
	}

	/* copy from the bottom-matrix */
	for (i=0, dst = R->values + typlen * X->height, src = Y->values;
		 i < width;
		 i++, dst += typlen * height, src += typlen * Y->height)
	{
		if (i < Y->width)
			memcpy(dst, src, typlen * Y->height);
		else
			memset(dst, 0, typlen * Y->height);
	}
	return R;
}

Datum
array_matrix_rbind_int2(PG_FUNCTION_ARGS)
{
	MatrixType	   *X = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType	   *Y = PG_GETARG_MATRIXTYPE_P(1);
	PG_RETURN_MATRIXTYPE_P(array_martix_rbind(INT2OID, X, Y));
}
PG_FUNCTION_INFO_V1(array_matrix_rbind_int2);

Datum
array_matrix_rbind_int4(PG_FUNCTION_ARGS)
{
	MatrixType	   *X = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType	   *Y = PG_GETARG_MATRIXTYPE_P(1);
	PG_RETURN_MATRIXTYPE_P(array_martix_rbind(INT4OID, X, Y));
}
PG_FUNCTION_INFO_V1(array_matrix_rbind_int4);

Datum
array_matrix_rbind_int8(PG_FUNCTION_ARGS)
{
	MatrixType	   *X = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType	   *Y = PG_GETARG_MATRIXTYPE_P(1);
	PG_RETURN_MATRIXTYPE_P(array_martix_rbind(INT8OID, X, Y));
}
PG_FUNCTION_INFO_V1(array_matrix_rbind_int8);

Datum
array_matrix_rbind_float4(PG_FUNCTION_ARGS)
{
	MatrixType	   *X = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType	   *Y = PG_GETARG_MATRIXTYPE_P(1);
	PG_RETURN_MATRIXTYPE_P(array_martix_rbind(FLOAT4OID, X, Y));
}
PG_FUNCTION_INFO_V1(array_matrix_rbind_float4);

Datum
array_matrix_rbind_float8(PG_FUNCTION_ARGS)
{
	MatrixType	   *X = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType	   *Y = PG_GETARG_MATRIXTYPE_P(1);
	PG_RETURN_MATRIXTYPE_P(array_martix_rbind(FLOAT8OID, X, Y));
}
PG_FUNCTION_INFO_V1(array_matrix_rbind_float8);

/*
 * cbind that takes two arrays
 */
static MatrixType *
array_martix_cbind(Oid elemtype, MatrixType *X, MatrixType *Y)
{
	MatrixType *R;
	cl_uint		height;
	cl_uint		width;
	int			typlen;
	Size		length;
	int			i;
	char	   *src, *dst;

	/* sanity checks */
	if (VARATT_IS_EXPANDED_HEADER(X) || VARATT_IS_EXPANDED_HEADER(Y))
		elog(ERROR, "ExpandedArrayHeader is not supported");
	if (!VALIDATE_ARRAY_MATRIX(X) || !VALIDATE_ARRAY_MATRIX(Y))
		elog(ERROR, "Not a matrix-like array");
	if (elemtype != X->elemtype || elemtype != Y->elemtype)
		elog(ERROR, "Bug? not expected type");
	typlen = get_typlen(elemtype);

	width = X->width + Y->width;
	height = Max(X->height, Y->height);
	length = offsetof(MatrixType,
					  values[typlen * (Size)width * (Size)height]);
	R = palloc(length);
	SET_VARSIZE(R, length);
	INIT_ARRAY_MATRIX(R, elemtype, height, width);

	for (i=0, src = X->values, dst = R->values;
		 i < X->width;
		 i++, src += typlen * X->height, dst += typlen * height)
	{
		memcpy(dst, src, typlen * X->height);
		if (X->height < height)
			memset(dst + typlen * X->height, 0,
				   typlen * (height - X->height));
	}

	for (i=0, src = Y->values, dst = (R->values +
										 typlen * X->width * height);
		 i < Y->width;
		 i++, src += typlen * Y->height, dst += typlen * height)
	{
		memcpy(dst, src, typlen * Y->height);
		if (Y->height < height)
			memset(dst + typlen * Y->height, 0,
				   typlen * (height - Y->height));
	}
	return R;
}

Datum
array_matrix_cbind_int2(PG_FUNCTION_ARGS)
{
	MatrixType	   *X = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType	   *Y = PG_GETARG_MATRIXTYPE_P(1);
	PG_RETURN_MATRIXTYPE_P(array_martix_cbind(INT2OID, X, Y));
}
PG_FUNCTION_INFO_V1(array_matrix_cbind_int2);

Datum
array_matrix_cbind_int4(PG_FUNCTION_ARGS)
{
	MatrixType	   *X = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType	   *Y = PG_GETARG_MATRIXTYPE_P(1);
	PG_RETURN_MATRIXTYPE_P(array_martix_cbind(INT4OID, X, Y));
}
PG_FUNCTION_INFO_V1(array_matrix_cbind_int4);

Datum
array_matrix_cbind_int8(PG_FUNCTION_ARGS)
{
	MatrixType	   *X = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType	   *Y = PG_GETARG_MATRIXTYPE_P(1);
	PG_RETURN_MATRIXTYPE_P(array_martix_cbind(INT8OID, X, Y));
}
PG_FUNCTION_INFO_V1(array_matrix_cbind_int8);

Datum
array_matrix_cbind_float4(PG_FUNCTION_ARGS)
{
	MatrixType	   *X = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType	   *Y = PG_GETARG_MATRIXTYPE_P(1);
	PG_RETURN_MATRIXTYPE_P(array_martix_cbind(FLOAT4OID, X, Y));
}
PG_FUNCTION_INFO_V1(array_matrix_cbind_float4);

Datum
array_matrix_cbind_float8(PG_FUNCTION_ARGS)
{
	MatrixType	   *X = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType	   *Y = PG_GETARG_MATRIXTYPE_P(1);
	PG_RETURN_MATRIXTYPE_P(array_martix_cbind(FLOAT8OID, X, Y));
}
PG_FUNCTION_INFO_V1(array_matrix_cbind_float8);

/*
 * rbind as aggregate function
 */
typedef struct
{
	Oid		elemtype;
	Size	width;
	Size	height;
	List   *matrix_list;
} matrix_rbind_state;

Datum
array_matrix_rbind_accum(PG_FUNCTION_ARGS)
{
	matrix_rbind_state *mrstate;
	MemoryContext	aggcxt;
	MemoryContext	oldcxt;
	MatrixType	   *X;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");

	if (PG_ARGISNULL(1))
		elog(ERROR, "null-array was supplied");

	oldcxt = MemoryContextSwitchTo(aggcxt);
    X = PG_GETARG_MATRIXTYPE_P_COPY(1);
	if (!VALIDATE_ARRAY_MATRIX(X))
		elog(ERROR, "input array is not a valid matrix-like array");

	if (PG_ARGISNULL(0))
	{
		mrstate = palloc0(sizeof(matrix_rbind_state));
		mrstate->elemtype = X->elemtype;
	}
	else
	{
		mrstate = (matrix_rbind_state *)PG_GETARG_POINTER(0);
		if (mrstate->elemtype == X->elemtype)
			elog(ERROR, "element type of input array mismatch '%s' for '%s'",
				 format_type_be(X->elemtype),
				 format_type_be(mrstate->elemtype));
	}

	mrstate->width = Max(mrstate->width, X->width);
	mrstate->height += X->height;
	mrstate->matrix_list = lappend(mrstate->matrix_list, X);

	MemoryContextSwitchTo(oldcxt);

	PG_RETURN_POINTER(mrstate);
}
PG_FUNCTION_INFO_V1(array_matrix_rbind_accum);

static MatrixType *
array_matrix_rbind_final(matrix_rbind_state *mrstate)
{
	MatrixType *R;
	int			typlen;
	Size		height = mrstate->height;
	Size		width = mrstate->width;
	Size		length;
	Size		row_index;
	char	   *src, *dst;
	ListCell   *lc;

	switch (mrstate->elemtype)
	{
		case INT2OID:
			typlen = sizeof(cl_short);
			break;
		case INT4OID:
		case FLOAT4OID:
			typlen = sizeof(cl_int);
			break;
		case INT8OID:
		case FLOAT8OID:
			typlen = sizeof(cl_long);
			break;
		default:
			elog(ERROR, "unsupported element type: %s",
				 format_type_be(mrstate->elemtype));
	}
	length = MAXALIGN(offsetof(MatrixType,
							   values[(Size)typlen * width * height]));
	if (!AllocSizeIsValid(length))
		elog(ERROR, "supplied array-matrix is too big");
	R = palloc(length);
	SET_VARSIZE(R, length);
	INIT_ARRAY_MATRIX(R, mrstate->elemtype, height, width);

	row_index = 0;
	foreach (lc, mrstate->matrix_list)
	{
		MatrixType *X = lfirst(lc);
		cl_int		i;

		Assert(VALIDATE_ARRAY_MATRIX(matrix));
		for (i=0, src = X->values, dst = R->values + typlen * row_index;
			 i < width;
			 i++, src += typlen * X->height, dst += typlen * height)
		{
			if (i < X->width)
				memcpy(dst, src, typlen * X->height);
			else
				memset(dst, 0, typlen * X->height);
		}
		row_index += X->height;
	}
	return R;
}

Datum
array_matrix_rbind_final_int2(PG_FUNCTION_ARGS)
{
	matrix_rbind_state *mrstate;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	mrstate = (matrix_rbind_state *)PG_GETARG_POINTER(0);
	Assert(mrstate->elemtype == INT2OID);
	PG_RETURN_POINTER(array_matrix_rbind_final(mrstate));
}
PG_FUNCTION_INFO_V1(array_matrix_rbind_final_int2);

Datum
array_matrix_rbind_final_int4(PG_FUNCTION_ARGS)
{
	matrix_rbind_state *mrstate;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	mrstate = (matrix_rbind_state *)PG_GETARG_POINTER(0);
	Assert(mrstate->elemtype == INT4OID);
	PG_RETURN_POINTER(array_matrix_rbind_final(mrstate));
}
PG_FUNCTION_INFO_V1(array_matrix_rbind_final_int4);

Datum
array_matrix_rbind_final_int8(PG_FUNCTION_ARGS)
{
	matrix_rbind_state *mrstate;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	mrstate = (matrix_rbind_state *)PG_GETARG_POINTER(0);
	Assert(mrstate->elemtype == INT8OID);
	PG_RETURN_POINTER(array_matrix_rbind_final(mrstate));
}
PG_FUNCTION_INFO_V1(array_matrix_rbind_final_int8);

Datum
array_matrix_rbind_final_float4(PG_FUNCTION_ARGS)
{
	matrix_rbind_state *mrstate;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	mrstate = (matrix_rbind_state *)PG_GETARG_POINTER(0);
	Assert(mrstate->elemtype == FLOAT4OID);
	PG_RETURN_POINTER(array_matrix_rbind_final(mrstate));
}
PG_FUNCTION_INFO_V1(array_matrix_rbind_final_float4);

Datum
array_matrix_rbind_final_float8(PG_FUNCTION_ARGS)
{
	matrix_rbind_state *mrstate;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	mrstate = (matrix_rbind_state *)PG_GETARG_POINTER(0);
	Assert(mrstate->elemtype == FLOAT8OID);
	PG_RETURN_POINTER(array_matrix_rbind_final(mrstate));
}
PG_FUNCTION_INFO_V1(array_matrix_rbind_final_float8);

/*
 * cbind as aggregate function
 */
typedef struct
{
	Oid		elemtype;
	Size	width;
	Size	height;
	List   *matrix_list;
} matrix_cbind_state;

Datum
array_matrix_cbind_accum(PG_FUNCTION_ARGS)
{
	matrix_cbind_state *mcstate;
	MemoryContext	aggcxt;
	MemoryContext	oldcxt;
	MatrixType	   *X;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");

	if (PG_ARGISNULL(1))
		elog(ERROR, "null-array was supplied");

	oldcxt = MemoryContextSwitchTo(aggcxt);
	X = PG_GETARG_MATRIXTYPE_P_COPY(1);
	if (!VALIDATE_ARRAY_MATRIX(X))
		elog(ERROR, "input array is not a valid matrix-like array");

	if (PG_ARGISNULL(0))
	{
		mcstate = palloc0(sizeof(matrix_cbind_state));
		mcstate->elemtype = X->elemtype;
	}
	else
	{
		mcstate = (matrix_cbind_state *)PG_GETARG_POINTER(0);
		if (mcstate->elemtype == X->elemtype)
			elog(ERROR, "element type of input array mismatch '%s' for '%s'",
				 format_type_be(X->elemtype),
				 format_type_be(mcstate->elemtype));
	}
	mcstate->width += X->width;
	mcstate->height = Max(mcstate->height, X->height);
	mcstate->matrix_list = lappend(mcstate->matrix_list, X);

	MemoryContextSwitchTo(oldcxt);

	PG_RETURN_POINTER(mcstate);
}
PG_FUNCTION_INFO_V1(array_matrix_cbind_accum);

static MatrixType *
array_matrix_cbind_final(matrix_cbind_state *mcstate)
{
	MatrixType *R;
	int			typlen;
	Size		height = mcstate->height;
	Size		width = mcstate->width;
	Size		length;
	ListCell   *lc;
	char	   *src, *dst;

	switch (mcstate->elemtype)
	{
		case INT2OID:
			typlen = sizeof(cl_short);
			break;
		case INT4OID:
		case FLOAT4OID:
			typlen = sizeof(cl_int);
			break;
		case INT8OID:
		case FLOAT8OID:
			typlen = sizeof(cl_long);
			break;
		default:
			elog(ERROR, "unsupported element type: %s",
				 format_type_be(mcstate->elemtype));
    }
	length = MAXALIGN(offsetof(MatrixType,
							   values[(Size)typlen * width * height]));
	if (!AllocSizeIsValid(length))
		elog(ERROR, "supplied array-matrix is too big");
	R = palloc(length);
	SET_VARSIZE(R, length);
	INIT_ARRAY_MATRIX(R, mcstate->elemtype, height, width);

	dst = R->values;
	foreach (lc, mcstate->matrix_list)
	{
		MatrixType *X = lfirst(lc);
		cl_uint		i;

		Assert(VALIDATE_ARRAY_MATRIX(X));
		for (i=0, src = X->values;
			 i < X->width;
			 i++, src += typlen * X->height, dst += typlen * height)
		{
			memcpy(dst, src, typlen * X->height);
			if (X->height < height)
				memset(dst + typlen * X->height, 0,
					   typlen * (height - X->height));
		}
	}
	return R;
}

Datum
array_matrix_cbind_final_int2(PG_FUNCTION_ARGS)
{
	matrix_cbind_state *mcstate;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	mcstate = (matrix_cbind_state *)PG_GETARG_POINTER(0);
	Assert(mcstate->elemtype == INT2OID);
	PG_RETURN_POINTER(array_matrix_cbind_final(mcstate));
}
PG_FUNCTION_INFO_V1(array_matrix_cbind_final_int2);

Datum
array_matrix_cbind_final_int4(PG_FUNCTION_ARGS)
{
	matrix_cbind_state *mcstate;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	mcstate = (matrix_cbind_state *)PG_GETARG_POINTER(0);
	Assert(mcstate->elemtype == INT4OID);
	PG_RETURN_POINTER(array_matrix_cbind_final(mcstate));
}
PG_FUNCTION_INFO_V1(array_matrix_cbind_final_int4);

Datum
array_matrix_cbind_final_int8(PG_FUNCTION_ARGS)
{
	matrix_cbind_state *mcstate;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	mcstate = (matrix_cbind_state *)PG_GETARG_POINTER(0);
	Assert(mcstate->elemtype == INT8OID);
	PG_RETURN_POINTER(array_matrix_cbind_final(mcstate));
}
PG_FUNCTION_INFO_V1(array_matrix_cbind_final_int8);

Datum
array_matrix_cbind_final_float4(PG_FUNCTION_ARGS)
{
	matrix_cbind_state *mcstate;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	mcstate = (matrix_cbind_state *)PG_GETARG_POINTER(0);
	Assert(mcstate->elemtype == FLOAT4OID);
	PG_RETURN_POINTER(array_matrix_cbind_final(mcstate));
}
PG_FUNCTION_INFO_V1(array_matrix_cbind_final_float4);

Datum
array_matrix_cbind_final_float8(PG_FUNCTION_ARGS)
{
	matrix_cbind_state *mcstate;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	mcstate = (matrix_cbind_state *)PG_GETARG_POINTER(0);
	Assert(mcstate->elemtype == FLOAT8OID);
	PG_RETURN_POINTER(array_matrix_cbind_final(mcstate));
}
PG_FUNCTION_INFO_V1(array_matrix_cbind_final_float8);

/*
 * matrix_transpose
 */
#define ARRAY_MATRIX_TRANSPOSE_TEMPLATE(T,M,BASETYPE)					\
	do {																\
		Size	height = (M)->height;									\
		Size	width = (M)->width;										\
		Size	i, nitems = width * height;								\
		Size	length;													\
																		\
		length = offsetof(MatrixType,									\
						  values[sizeof(BASETYPE) * width * height]);	\
		if (!AllocSizeIsValid(length))									\
			elog(ERROR, "matrix array size too large");					\
		T = palloc(length);												\
		SET_VARSIZE(T, length);											\
		INIT_ARRAY_MATRIX(T, (M)->elemtype, width, height);				\
																		\
		for (i=0; i < nitems; i++)										\
		{																\
			*((BASETYPE *)(T->values + sizeof(BASETYPE) *				\
						   ((i % height) * width + (i / height)))) =	\
				*((BASETYPE *)((M)->values + sizeof(BASETYPE) * i));	\
		}																\
	} while(0)

Datum
array_matrix_transpose_int2(PG_FUNCTION_ARGS)
{
	MatrixType *matrix = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType *result;

	if (VARATT_IS_EXPANDED_HEADER(matrix) ||
		!VALIDATE_ARRAY_MATRIX(matrix))
		elog(ERROR, "Array is not like Matrix");
	Assert(matrix->elemtype == INT2OID);
	ARRAY_MATRIX_TRANSPOSE_TEMPLATE(result,matrix,cl_ushort);
	PG_RETURN_POINTER(result);
}
PG_FUNCTION_INFO_V1(array_matrix_transpose_int2);

Datum
array_matrix_transpose_int4(PG_FUNCTION_ARGS)
{
	MatrixType *matrix = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType *result;

	if (VARATT_IS_EXPANDED_HEADER(matrix) ||
		!VALIDATE_ARRAY_MATRIX(matrix))
		elog(ERROR, "Array is not like Matrix");
	Assert(matrix->elemtype == INT4OID);
	ARRAY_MATRIX_TRANSPOSE_TEMPLATE(result,matrix,cl_ushort);
	PG_RETURN_POINTER(result);
}
PG_FUNCTION_INFO_V1(array_matrix_transpose_int4);

Datum
array_matrix_transpose_int8(PG_FUNCTION_ARGS)
{
	MatrixType *matrix = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType *result;

	if (VARATT_IS_EXPANDED_HEADER(matrix) ||
		!VALIDATE_ARRAY_MATRIX(matrix))
		elog(ERROR, "Array is not like Matrix");
	Assert(matrix->elemtype == INT8OID);
	ARRAY_MATRIX_TRANSPOSE_TEMPLATE(result,matrix,cl_ushort);
	PG_RETURN_POINTER(result);
}
PG_FUNCTION_INFO_V1(array_matrix_transpose_int8);

Datum
array_matrix_transpose_float4(PG_FUNCTION_ARGS)
{
	MatrixType *matrix = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType *result;

	if (VARATT_IS_EXPANDED_HEADER(matrix) ||
		!VALIDATE_ARRAY_MATRIX(matrix))
		elog(ERROR, "Array is not like Matrix");
	Assert(matrix->elemtype == FLOAT4OID);
	ARRAY_MATRIX_TRANSPOSE_TEMPLATE(result,matrix,cl_ushort);
	PG_RETURN_POINTER(result);
}
PG_FUNCTION_INFO_V1(array_matrix_transpose_float4);

Datum
array_matrix_transpose_float8(PG_FUNCTION_ARGS)
{
	MatrixType *matrix = PG_GETARG_MATRIXTYPE_P(0);
	MatrixType *result;

	if (VARATT_IS_EXPANDED_HEADER(matrix) ||
		!VALIDATE_ARRAY_MATRIX(matrix))
		elog(ERROR, "Array is not like Matrix");
	Assert(matrix->elemtype == FLOAT8OID);
	ARRAY_MATRIX_TRANSPOSE_TEMPLATE(result,matrix,cl_ushort);
	PG_RETURN_POINTER(result);
}
PG_FUNCTION_INFO_V1(array_matrix_transpose_float8);
