/*
 * aggfuncs.c
 *
 * Definition of self-defined aggregate functions, used by GpuPreAgg
 * ----
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
#include "pg_strom.h"
#include "cuda_numeric.h"

/*
 * declarations
 */
Datum pgstrom_partial_nrows(PG_FUNCTION_ARGS);
Datum pgstrom_partial_avg_int8(PG_FUNCTION_ARGS);
Datum pgstrom_partial_avg_float8(PG_FUNCTION_ARGS);
Datum pgstrom_final_avg_int8_accum(PG_FUNCTION_ARGS);
Datum pgstrom_final_avg_int8_final(PG_FUNCTION_ARGS);
Datum pgstrom_final_avg_float8_accum(PG_FUNCTION_ARGS);
Datum pgstrom_final_avg_float8_final(PG_FUNCTION_ARGS);
Datum pgstrom_final_avg_numeric_final(PG_FUNCTION_ARGS);
Datum pgstrom_partial_min_any(PG_FUNCTION_ARGS);
Datum pgstrom_partial_max_any(PG_FUNCTION_ARGS);
Datum pgstrom_partial_sum_any(PG_FUNCTION_ARGS);
Datum pgstrom_partial_sum_x2_float4(PG_FUNCTION_ARGS);
Datum pgstrom_partial_sum_x2_float8(PG_FUNCTION_ARGS);
Datum pgstrom_partial_sum_x2_numeric(PG_FUNCTION_ARGS);
Datum pgstrom_partial_cov_x(PG_FUNCTION_ARGS);
Datum pgstrom_partial_cov_y(PG_FUNCTION_ARGS);
Datum pgstrom_partial_cov_x2(PG_FUNCTION_ARGS);
Datum pgstrom_partial_cov_y2(PG_FUNCTION_ARGS);
Datum pgstrom_partial_cov_xy(PG_FUNCTION_ARGS);
Datum pgstrom_partial_variance_float8(PG_FUNCTION_ARGS);
Datum pgstrom_partial_covariance_float8(PG_FUNCTION_ARGS);
Datum pgstrom_float8_stddev_samp_numeric(PG_FUNCTION_ARGS);
Datum pgstrom_float8_stddev_pop_numeric(PG_FUNCTION_ARGS);
Datum pgstrom_float8_var_samp_numeric(PG_FUNCTION_ARGS);
Datum pgstrom_float8_var_pop_numeric(PG_FUNCTION_ARGS);

/* utility to reference numeric[] */
static inline Datum
numeric_array_ref(ArrayType *array, int index, bool *p_isnull)
{
	return array_ref(array, 1, &index, -1, -1, false, 'i', p_isnull);
}

Datum
pgstrom_partial_nrows(PG_FUNCTION_ARGS)
{
	int		i;

	for (i=0; i < PG_NARGS(); i++)
	{
		if (PG_ARGISNULL(i) || !PG_GETARG_BOOL(i))
			PG_RETURN_INT64(0);
	}
	PG_RETURN_INT64(1);
}
PG_FUNCTION_INFO_V1(pgstrom_partial_nrows);

Datum
pgstrom_partial_avg_int8(PG_FUNCTION_ARGS)
{
	ArrayType  *result;
	Datum		items[2];

	items[0] = PG_GETARG_DATUM(0);	/* nrows(int8) */
	items[1] = PG_GETARG_DATUM(1);	/* p_sum(int8) */
	result = construct_array(items, 2, INT8OID,
							 sizeof(int64), FLOAT8PASSBYVAL, 'd');
	PG_RETURN_ARRAYTYPE_P(result);
}
PG_FUNCTION_INFO_V1(pgstrom_partial_avg_int8);

Datum
pgstrom_partial_avg_float8(PG_FUNCTION_ARGS)
{
	int64		nrows = PG_GETARG_INT64(0);
	ArrayType  *result;
	Datum		items[2];

	items[0] = Float8GetDatum((float8)nrows);
	items[1] = PG_GETARG_DATUM(1);	/* p_sum(float8) */
	result = construct_array(items, 2, FLOAT8OID,
							 sizeof(float8), FLOAT8PASSBYVAL, 'd');
	PG_RETURN_ARRAYTYPE_P(result);
}
PG_FUNCTION_INFO_V1(pgstrom_partial_avg_float8);

Datum
pgstrom_final_avg_int8_accum(PG_FUNCTION_ARGS)
{
	MemoryContext	aggcxt;
	MemoryContext	oldcxt;
	ArrayType	   *xarray;
	ArrayType	   *yarray;
	int64		   *x, *y;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	if (PG_ARGISNULL(1))
		elog(ERROR, "Null state was supplied");

	if (PG_ARGISNULL(0))
	{
		oldcxt = MemoryContextSwitchTo(aggcxt);
		xarray = PG_GETARG_ARRAYTYPE_P_COPY(1);
		MemoryContextSwitchTo(oldcxt);
	}
	else
	{
		xarray = PG_GETARG_ARRAYTYPE_P(0);
		yarray = PG_GETARG_ARRAYTYPE_P(1);
		x = (int64 *)ARR_DATA_PTR(xarray);
		y = (int64 *)ARR_DATA_PTR(yarray);

		x[0] += y[0];
		x[1] += y[1];
	}
	PG_RETURN_POINTER(xarray);
}
PG_FUNCTION_INFO_V1(pgstrom_final_avg_int8_accum);

Datum
pgstrom_final_avg_int8_final(PG_FUNCTION_ARGS)
{
	ArrayType	   *xarray = PG_GETARG_ARRAYTYPE_P(0);
	int64		   *x = (int64 *)ARR_DATA_PTR(xarray);

	return DirectFunctionCall2(numeric_div,
							   DirectFunctionCall1(int8_numeric,
												   Int64GetDatum(x[1])),
							   DirectFunctionCall1(int8_numeric,
												   Int64GetDatum(x[0])));
}
PG_FUNCTION_INFO_V1(pgstrom_final_avg_int8_final);

Datum
pgstrom_final_avg_float8_accum(PG_FUNCTION_ARGS)
{
	MemoryContext	aggcxt;
	MemoryContext	oldcxt;
	ArrayType	   *xarray;
	ArrayType	   *yarray;
	float8		   *x, *y;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	if (PG_ARGISNULL(1))
		elog(ERROR, "Null state was supplied");

	if (PG_ARGISNULL(0))
	{
		oldcxt = MemoryContextSwitchTo(aggcxt);
		xarray = PG_GETARG_ARRAYTYPE_P_COPY(1);
		MemoryContextSwitchTo(oldcxt);
	}
	else
	{
		xarray = PG_GETARG_ARRAYTYPE_P(0);
		yarray = PG_GETARG_ARRAYTYPE_P(1);
		x = (float8 *)ARR_DATA_PTR(xarray);
		y = (float8 *)ARR_DATA_PTR(yarray);

		x[0] += y[0];
		x[1] += y[1];
	}
	PG_RETURN_POINTER(xarray);
}
PG_FUNCTION_INFO_V1(pgstrom_final_avg_float8_accum);

Datum
pgstrom_final_avg_float8_final(PG_FUNCTION_ARGS)
{
	ArrayType	   *xarray = PG_GETARG_ARRAYTYPE_P(0);
	float8		   *x = (float8 *)ARR_DATA_PTR(xarray);

	PG_RETURN_FLOAT8(x[1] / x[0]);
}
PG_FUNCTION_INFO_V1(pgstrom_final_avg_float8_final);

Datum
pgstrom_final_avg_numeric_final(PG_FUNCTION_ARGS)
{
	ArrayType	   *xarray = PG_GETARG_ARRAYTYPE_P(0);
	float8		   *x = (float8 *)ARR_DATA_PTR(xarray);
	Datum			nrows, sum;

	nrows = DirectFunctionCall1(float8_numeric, Float8GetDatum(x[0]));
	sum   = DirectFunctionCall1(float8_numeric, Float8GetDatum(x[1]));

	return DirectFunctionCall2(numeric_div, sum, nrows);
}
PG_FUNCTION_INFO_V1(pgstrom_final_avg_numeric_final);

/*
 * pgstrom.pmin(anyelement)
 */
Datum
pgstrom_partial_min_any(PG_FUNCTION_ARGS)
{
	PG_RETURN_DATUM(PG_GETARG_DATUM(0));
}
PG_FUNCTION_INFO_V1(pgstrom_partial_min_any);

/*
 * pgstrom.pmax(anyelement)
 */
Datum
pgstrom_partial_max_any(PG_FUNCTION_ARGS)
{
	PG_RETURN_DATUM(PG_GETARG_DATUM(0));
}
PG_FUNCTION_INFO_V1(pgstrom_partial_max_any);

/*
 * pgstrom.psum(anyelement)
 */
Datum
pgstrom_partial_sum_any(PG_FUNCTION_ARGS)
{
	PG_RETURN_DATUM(PG_GETARG_DATUM(0));
}
PG_FUNCTION_INFO_V1(pgstrom_partial_sum_any);

/*
 * pgstrom.psum_x2(float4)
 */
Datum
pgstrom_partial_sum_x2_float4(PG_FUNCTION_ARGS)
{
	float4		value = (PG_ARGISNULL(0) ? 0.0 : PG_GETARG_FLOAT4(0));

	PG_RETURN_FLOAT4(value * value);
}
PG_FUNCTION_INFO_V1(pgstrom_partial_sum_x2_float4);

/*
 * pgstrom.psum_x2(float8)
 */
Datum
pgstrom_partial_sum_x2_float8(PG_FUNCTION_ARGS)
{
	float8		value = (PG_ARGISNULL(0) ? 0.0 : PG_GETARG_FLOAT8(0));

	PG_RETURN_FLOAT8(value * value);	
}
PG_FUNCTION_INFO_V1(pgstrom_partial_sum_x2_float8);

/*
 * pgstrom.psum_x2(numeric)
 */
Datum
pgstrom_partial_sum_x2_numeric(PG_FUNCTION_ARGS)
{
	Datum		value;

	if (!PG_ARGISNULL(0))
		value = PG_GETARG_DATUM(0);	/* a valid numeric value */
	else
		value = DirectFunctionCall3(numeric_in,
									CStringGetDatum("0"),
									ObjectIdGetDatum(InvalidOid),
									Int32GetDatum(-1));
	return DirectFunctionCall2(numeric_mul, value, value);
}
PG_FUNCTION_INFO_V1(pgstrom_partial_sum_x2_numeric);

/*
 * pgstrom.pcov_x(float8)
 */
Datum
pgstrom_partial_cov_x(PG_FUNCTION_ARGS)
{
	if (!PG_GETARG_BOOL(0))
		PG_RETURN_NULL();
	PG_RETURN_DATUM(PG_GETARG_DATUM(1));
}
PG_FUNCTION_INFO_V1(pgstrom_partial_cov_x);

/*
 * pgstrom.pcov_y(float8)
 */
Datum
pgstrom_partial_cov_y(PG_FUNCTION_ARGS)
{
	if (!PG_GETARG_BOOL(0))
		PG_RETURN_NULL();
	PG_RETURN_DATUM(PG_GETARG_DATUM(2));
}
PG_FUNCTION_INFO_V1(pgstrom_partial_cov_y);

/*
 * pgstrom.pcov_x2(float8)
 */
Datum
pgstrom_partial_cov_x2(PG_FUNCTION_ARGS)
{
	float8		value = PG_GETARG_FLOAT8(1);

	if (!PG_GETARG_BOOL(0))
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(value * value);
}
PG_FUNCTION_INFO_V1(pgstrom_partial_cov_x2);

/*
 * pgstrom.pcov_y2(float8)
 */
Datum
pgstrom_partial_cov_y2(PG_FUNCTION_ARGS)
{
	float8		value = PG_GETARG_FLOAT8(2);

	if (!PG_GETARG_BOOL(0))
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(value * value);
}
PG_FUNCTION_INFO_V1(pgstrom_partial_cov_y2);

/*
 * pgstrom.pcov_xy(float8)
 */
Datum
pgstrom_partial_cov_xy(PG_FUNCTION_ARGS)
{
	float8	x_value = PG_GETARG_FLOAT8(1);
	float8	y_value = PG_GETARG_FLOAT8(2);

	if (!PG_GETARG_BOOL(0))
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(x_value * y_value);
}
PG_FUNCTION_INFO_V1(pgstrom_partial_cov_xy);

/*
 * pgstrom_partial_variance_float8
 */
Datum
pgstrom_partial_variance_float8(PG_FUNCTION_ARGS)
{
	ArrayType  *state;
	Datum		items[3];

	items[0] = Float8GetDatum((double)PG_GETARG_INT64(0));	/* nrows(int8) */
	items[1] = PG_GETARG_DATUM(1);	/* sum of X */
	items[2] = PG_GETARG_DATUM(2);	/* sum of X^2 */
    state = construct_array(items, 3, FLOAT8OID,
							sizeof(float8), FLOAT8PASSBYVAL, 'd');
	PG_RETURN_ARRAYTYPE_P(state);
}
PG_FUNCTION_INFO_V1(pgstrom_partial_variance_float8);

/*
 * pgstrom_partial_covariance_float8
 */
Datum
pgstrom_partial_covariance_float8(PG_FUNCTION_ARGS)
{
	ArrayType  *state;
	Datum		items[6];

	items[0] = Float8GetDatum((double)PG_GETARG_INT64(0));	/* nrows(int8) */
	items[1] = PG_GETARG_DATUM(1);	/* sum of X */
	items[2] = PG_GETARG_DATUM(2);	/* sum of X^2 */
	items[3] = PG_GETARG_DATUM(3);	/* sum of Y */
	items[4] = PG_GETARG_DATUM(4);	/* sum of Y^2 */
	items[5] = PG_GETARG_DATUM(5);	/* sum of X*Y */
	state = construct_array(items, 6, FLOAT8OID,
							sizeof(float8), FLOAT8PASSBYVAL, 'd');
	PG_RETURN_ARRAYTYPE_P(state);
}
PG_FUNCTION_INFO_V1(pgstrom_partial_covariance_float8);

/*
 * pgstrom_float8_stddev_samp_numeric
 */
Datum
pgstrom_float8_stddev_samp_numeric(PG_FUNCTION_ARGS)
{
	Datum	datum = DirectFunctionCall1(float8_stddev_samp,
										PG_GETARG_DATUM(0));
	PG_RETURN_NUMERIC(DirectFunctionCall1(float8_numeric, datum));
}
PG_FUNCTION_INFO_V1(pgstrom_float8_stddev_samp_numeric);

/*
 * pgstrom_float8_stddev_pop_numeric
 */
Datum
pgstrom_float8_stddev_pop_numeric(PG_FUNCTION_ARGS)
{
	Datum	datum = DirectFunctionCall1(float8_stddev_pop,
										PG_GETARG_DATUM(0));
	PG_RETURN_NUMERIC(DirectFunctionCall1(float8_numeric, datum));
}
PG_FUNCTION_INFO_V1(pgstrom_float8_stddev_pop_numeric);

/*
 * pgstrom_float8_var_samp_numeric
 */
Datum
pgstrom_float8_var_samp_numeric(PG_FUNCTION_ARGS)
{
	Datum	datum = DirectFunctionCall1(float8_var_samp,
										PG_GETARG_DATUM(0));
	PG_RETURN_NUMERIC(DirectFunctionCall1(float8_numeric, datum));
}
PG_FUNCTION_INFO_V1(pgstrom_float8_var_samp_numeric);

/*
 * pgstrom_float8_var_pop_numeric
 */
Datum
pgstrom_float8_var_pop_numeric(PG_FUNCTION_ARGS)
{
	Datum	datum = DirectFunctionCall1(float8_var_pop,
										PG_GETARG_DATUM(0));
	PG_RETURN_NUMERIC(DirectFunctionCall1(float8_numeric, datum));
}
PG_FUNCTION_INFO_V1(pgstrom_float8_var_pop_numeric);
