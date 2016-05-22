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
#include "utils/builtins.h"
#include "utils/fmgroids.h"
#include "pg_strom.h"

/*
 * matrix in/out functions - almost same as array functions, but extra
 * sanity checks are needed.
 */
static void
matrix_sanity_check(ArrayType *matrix, Oid element_type)
{
	int	   *lbounds;
	int		i;

	if (ARR_ELEMTYPE(matrix) != element_type)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("matrix with %s are not supported",
						format_type_be(ARR_ELEMTYPE(matrix)))));

	if (ARR_NDIM(matrix) != 2)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("matrix has to be 2-dimensional array")));

	if (ARR_HASNULL(matrix))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("matrix should not have null value")));

	lbounds = ARR_LBOUND(matrix);
	for (i=0; i < ARR_NDIM(matrix); i++)
	{
		if (lbounds[i] != 1)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("matrix cannot have lower bounds")));
	}
}

Datum
matrix_in(PG_FUNCTION_ARGS)
{
	Datum	matrix;

	matrix = OidFunctionCall3Coll(F_ARRAY_IN,
								  fcinfo->fncollation,
								  fcinfo->arg[0],
								  fcinfo->arg[1],
								  fcinfo->arg[2]);
	matrix_sanity_check(DatumGetArrayTypeP(matrix), FLOAT4OID);
	return matrix;
}
PG_FUNCTION_INFO_V1(matrix_in);

Datum
matrix_out(PG_FUNCTION_ARGS)
{
	ArrayType  *matrix = PG_GETARG_ARRAYTYPE_P(0);

	matrix_sanity_check(matrix, FLOAT4OID);

	return OidFunctionCall1Coll(F_ARRAY_OUT,
								fcinfo->fncollation,
								PointerGetDatum(matrix));
}
PG_FUNCTION_INFO_V1(matrix_out);

Datum
matrix_recv(PG_FUNCTION_ARGS)
{
	Datum	matrix;

	matrix = OidFunctionCall3Coll(F_ARRAY_RECV,
								  fcinfo->fncollation,
								  fcinfo->arg[0],
								  fcinfo->arg[1],
								  fcinfo->arg[2]);
	matrix_sanity_check(DatumGetArrayTypeP(matrix), FLOAT4OID);
	return matrix;
}
PG_FUNCTION_INFO_V1(matrix_recv);

Datum
matrix_send(PG_FUNCTION_ARGS)
{
	ArrayType  *matrix = PG_GETARG_ARRAYTYPE_P(0);

	matrix_sanity_check(matrix, FLOAT4OID);
	return OidFunctionCall1Coll(F_ARRAY_SEND,
								fcinfo->fncollation,
								PointerGetDatum(matrix));
}
PG_FUNCTION_INFO_V1(matrix_send);

/*
 * Type cast functions
 */
Datum
float4_to_matrix(PG_FUNCTION_ARGS)
{
	ArrayType  *array = PG_GETARG_ARRAYTYPE_P(0);
	ArrayType  *matrix;
	cl_uint		x_offset;
	cl_uint		y_offset;
	cl_uint		x_nitems;
	cl_uint		y_nitems;
	Size		len;
	bits8	   *nullmap;
	cl_uint		bitmask;
	cl_uint		i, j;
	cl_float   *src;
	cl_float   *dst;

	if (ARR_ELEMTYPE(array) != FLOAT4OID)
		elog(ERROR, "Bug? input array is not float4[]");
	if (ARR_NDIM(array) > 2)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("Unable to transform 3 or more dimension's array")));
	else if (ARR_NDIM(array) == 2)
	{
		int		   *lbounds = ARR_LBOUND(array);
		int		   *dims = ARR_DIMS(array);

		/* no need to adjust it? */
		if (!ARR_HASNULL(array) &&
			lbounds[0] == 1 &&
			lbounds[1] == 1)
			PG_RETURN_POINTER(array);

		/* make a new 2D-matrix */
		x_offset = lbounds[0] - 1;
		y_offset = lbounds[1] - 1;
		x_nitems = dims[0];
		y_nitems = dims[1];
	}
	else if (ARR_NDIM(array) == 1)
	{
		int		   *lbounds = ARR_LBOUND(array);
		int		   *dims = ARR_DIMS(array);

		/* 1D-array to 2D-matrix */
		x_offset = lbounds[0] - 1;
		y_offset = 0;
		x_nitems = dims[0];
		y_nitems = 1;
	}
	else
		elog(ERROR, "unexpected array dimension: %d", ARR_NDIM(array));

	/* construct a new 2D matrix, then copy data */
	if ((Size)(x_offset + x_nitems) * (Size)(y_offset + y_nitems) > INT_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("supplied array is too big")));

	len = sizeof(ArrayType) + 4 * sizeof(int) +
		sizeof(float) * (x_offset + x_nitems) * (y_offset + y_nitems);
	matrix = palloc(len);
	SET_VARSIZE(matrix, len);
	matrix->ndim = 2;
	matrix->dataoffset = 0;
	matrix->elemtype = FLOAT4OID;
	ARR_DIMS(matrix)[0] = x_offset + x_nitems;
	ARR_DIMS(matrix)[1] = y_offset + y_nitems;
	ARR_LBOUND(matrix)[0] = 1;
	ARR_LBOUND(matrix)[1] = 1;

	src = (float *)ARR_DATA_PTR(array);
	dst = (float *)ARR_DATA_PTR(matrix);
	nullmap = ARR_NULLBITMAP(array);
	bitmask = 1;

	if (y_offset > 0)
	{
		memset(dst, 0, sizeof(float) * y_offset * (x_offset + x_nitems));
		dst += y_offset * (x_offset + x_nitems);
	}

	for (j=0; j < y_nitems; j++)
	{
		if (x_offset > 0)
		{
			memset(dst, 0, sizeof(float) * x_offset);
			dst += x_offset;
		}
		for (i=0; i < x_nitems; i++, dst++)
		{
			if (nullmap && (*nullmap & bitmask) != 0)
			{
				*dst = 0.0;

				bitmask <<= 1;
				if (bitmask == 0x100)
				{
					bitmask = 1;
					nullmap++;
				}
			}
			else
			{
				*dst = *src;
				src++;
			}
		}
	}
	PG_RETURN_POINTER(matrix);
}
PG_FUNCTION_INFO_V1(float4_to_matrix);

Datum
matrix_to_float4(PG_FUNCTION_ARGS)
{
	ArrayType  *matrix = PG_GETARG_ARRAYTYPE_P(0);

	matrix_sanity_check(matrix, FLOAT4OID);

	PG_RETURN_POINTER(matrix);
}
PG_FUNCTION_INFO_V1(matrix_to_float4);
