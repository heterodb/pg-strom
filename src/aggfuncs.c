/*
 * aggfuncs.c
 *
 * Definition of self-defined aggregate functions, used by GpuPreAgg
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "cuda_numeric.h"

/*
 * declarations
 */
PG_FUNCTION_INFO_V1(pgstrom_partial_nrows);
PG_FUNCTION_INFO_V1(pgstrom_partial_avg_int8);
PG_FUNCTION_INFO_V1(pgstrom_partial_avg_float8);
PG_FUNCTION_INFO_V1(pgstrom_final_avg_int8_accum);
PG_FUNCTION_INFO_V1(pgstrom_final_avg_int8_final);
PG_FUNCTION_INFO_V1(pgstrom_final_avg_float8_accum);
PG_FUNCTION_INFO_V1(pgstrom_final_avg_float8_final);
PG_FUNCTION_INFO_V1(pgstrom_final_avg_numeric_final);
PG_FUNCTION_INFO_V1(pgstrom_partial_min_any);
PG_FUNCTION_INFO_V1(pgstrom_partial_max_any);
PG_FUNCTION_INFO_V1(pgstrom_partial_sum_any);
PG_FUNCTION_INFO_V1(pgstrom_partial_sum_x2_float4);
PG_FUNCTION_INFO_V1(pgstrom_partial_sum_x2_float8);
PG_FUNCTION_INFO_V1(pgstrom_partial_sum_x2_numeric);
PG_FUNCTION_INFO_V1(pgstrom_partial_cov_x);
PG_FUNCTION_INFO_V1(pgstrom_partial_cov_y);
PG_FUNCTION_INFO_V1(pgstrom_partial_cov_x2);
PG_FUNCTION_INFO_V1(pgstrom_partial_cov_y2);
PG_FUNCTION_INFO_V1(pgstrom_partial_cov_xy);
PG_FUNCTION_INFO_V1(pgstrom_partial_variance_float8);
PG_FUNCTION_INFO_V1(pgstrom_partial_covariance_float8);
PG_FUNCTION_INFO_V1(pgstrom_float8_combine);
PG_FUNCTION_INFO_V1(pgstrom_float8_stddev_samp);
PG_FUNCTION_INFO_V1(pgstrom_float8_stddev_pop);
PG_FUNCTION_INFO_V1(pgstrom_float8_stddev_samp_numeric);
PG_FUNCTION_INFO_V1(pgstrom_float8_stddev_pop_numeric);
PG_FUNCTION_INFO_V1(pgstrom_float8_var_samp);
PG_FUNCTION_INFO_V1(pgstrom_float8_var_pop);
PG_FUNCTION_INFO_V1(pgstrom_float8_var_samp_numeric);
PG_FUNCTION_INFO_V1(pgstrom_float8_var_pop_numeric);
PG_FUNCTION_INFO_V1(pgstrom_float8_regr_combine);
PG_FUNCTION_INFO_V1(pgstrom_float8_corr);
PG_FUNCTION_INFO_V1(pgstrom_float8_covar_pop);
PG_FUNCTION_INFO_V1(pgstrom_float8_covar_samp);
PG_FUNCTION_INFO_V1(pgstrom_float8_regr_avgx);
PG_FUNCTION_INFO_V1(pgstrom_float8_regr_avgy);
PG_FUNCTION_INFO_V1(pgstrom_float8_regr_intercept);
PG_FUNCTION_INFO_V1(pgstrom_float8_regr_r2);
PG_FUNCTION_INFO_V1(pgstrom_float8_regr_slope);
PG_FUNCTION_INFO_V1(pgstrom_float8_regr_sxx);
PG_FUNCTION_INFO_V1(pgstrom_float8_regr_sxy);
PG_FUNCTION_INFO_V1(pgstrom_float8_regr_syy);

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

Datum
pgstrom_final_avg_float8_final(PG_FUNCTION_ARGS)
{
	ArrayType	   *xarray = PG_GETARG_ARRAYTYPE_P(0);
	float8		   *x = (float8 *)ARR_DATA_PTR(xarray);

	PG_RETURN_FLOAT8(x[1] / x[0]);
}

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

/*
 * pgstrom.pmin(anyelement)
 */
Datum
pgstrom_partial_min_any(PG_FUNCTION_ARGS)
{
	PG_RETURN_DATUM(PG_GETARG_DATUM(0));
}

/*
 * pgstrom.pmax(anyelement)
 */
Datum
pgstrom_partial_max_any(PG_FUNCTION_ARGS)
{
	PG_RETURN_DATUM(PG_GETARG_DATUM(0));
}

/*
 * pgstrom.psum(anyelement)
 */
Datum
pgstrom_partial_sum_any(PG_FUNCTION_ARGS)
{
	PG_RETURN_DATUM(PG_GETARG_DATUM(0));
}

/*
 * pgstrom.psum_x2(float4)
 */
Datum
pgstrom_partial_sum_x2_float4(PG_FUNCTION_ARGS)
{
	float4		value = (PG_ARGISNULL(0) ? 0.0 : PG_GETARG_FLOAT4(0));

	PG_RETURN_FLOAT4(value * value);
}

/*
 * pgstrom.psum_x2(float8)
 */
Datum
pgstrom_partial_sum_x2_float8(PG_FUNCTION_ARGS)
{
	float8		value = (PG_ARGISNULL(0) ? 0.0 : PG_GETARG_FLOAT8(0));

	PG_RETURN_FLOAT8(value * value);	
}

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

/*
 * float8 validator
 */
static inline float8 *
check_float8_array(ArrayType *transarray, const char *caller, int n)
{
	if (ARR_NDIM(transarray) != 1 ||
		ARR_DIMS(transarray)[0] != n ||
		ARR_HASNULL(transarray) ||
		ARR_ELEMTYPE(transarray) != FLOAT8OID)
		elog(ERROR, "%s: expected %d-element float8 array", caller, n);
	return (float8 *) ARR_DATA_PTR(transarray);
}

static inline void
check_float8_value(float8 value, bool inf_is_valid, bool zero_is_valid)
{
	if (isinf(value) && !inf_is_valid)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("value out of range: overflow")));
	if (value == 0.0 && !zero_is_valid)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("value out of range: underflow")));
}

/*
 * pgstrom_float8_combine
 */
Datum
pgstrom_float8_combine(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray1 = PG_GETARG_ARRAYTYPE_P(0);
	ArrayType  *transarray2 = PG_GETARG_ARRAYTYPE_P(1);
	float8	   *transvalues1;
	float8	   *transvalues2;
	float8		N, sumX, sumX2;

	if (!AggCheckCallContext(fcinfo, NULL))
		elog(ERROR, "aggregate function called in non-aggregate context");
	transvalues1 = check_float8_array(transarray1, __FUNCTION__, 3);
	N     = transvalues1[0];
	sumX  = transvalues1[1];
	sumX2 = transvalues1[2];

	transvalues2 = check_float8_array(transarray2, __FUNCTION__, 3);
	N     += transvalues2[0];
    sumX  += transvalues2[1];
	sumX2 += transvalues2[2];
	check_float8_value(sumX,  isinf(transvalues1[1]) || isinf(transvalues2[1]), true);
	check_float8_value(sumX2, isinf(transvalues1[2]) || isinf(transvalues2[2]), true);

	transvalues1[0] = N;
	transvalues1[1] = sumX;
	transvalues1[2] = sumX2;

	PG_RETURN_ARRAYTYPE_P(transarray1);
}

/*
 * pgstrom_float8_var_samp
 */
Datum
pgstrom_float8_var_samp(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	float8	   *transvalues;
	float8		N, sumX, sumX2;
	float8		numerator;

	transvalues = check_float8_array(transarray, "float8_stddev_pop", 3);
	N = transvalues[0];
	sumX = transvalues[1];
	sumX2 = transvalues[2];
	/* Population stddev is undefined when N is 0, so return NULL */
	if (N == 0.0)
		PG_RETURN_NULL();

	numerator = N * sumX2 - sumX * sumX;
	check_float8_value(numerator, isinf(sumX2) || isinf(sumX), true);

	/* Watch out for roundoff error producing a negative numerator */
	if (numerator <= 0.0)
		PG_RETURN_FLOAT8(0.0);

	PG_RETURN_FLOAT8(numerator / (N * (N - 1.0)));
}

/*
 * pgstrom_float8_var_pop
 */
Datum
pgstrom_float8_var_pop(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	float8	   *transvalues;
	float8		N, sumX, sumX2;
	float8		numerator;

	transvalues = check_float8_array(transarray, "float8_stddev_pop", 3);
	N = transvalues[0];
	sumX = transvalues[1];
	sumX2 = transvalues[2];
	/* Population stddev is undefined when N is 0, so return NULL */
	if (N == 0.0)
		PG_RETURN_NULL();

	numerator = N * sumX2 - sumX * sumX;
	check_float8_value(numerator, isinf(sumX2) || isinf(sumX), true);

	/* Watch out for roundoff error producing a negative numerator */
	if (numerator <= 0.0)
		PG_RETURN_FLOAT8(0.0);

	PG_RETURN_FLOAT8(numerator / (N * N));
}

/*
 * pgstrom_float8_stddev_samp
 */
Datum
pgstrom_float8_stddev_samp(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	float8	   *transvalues;
	float8		N, sumX, sumX2;
	float8		numerator;

	transvalues = check_float8_array(transarray, "float8_stddev_pop", 3);
	N = transvalues[0];
	sumX = transvalues[1];
	sumX2 = transvalues[2];
	/* Population stddev is undefined when N is 0, so return NULL */
	if (N == 0.0)
		PG_RETURN_NULL();

	numerator = N * sumX2 - sumX * sumX;
	check_float8_value(numerator, isinf(sumX2) || isinf(sumX), true);

	/* Watch out for roundoff error producing a negative numerator */
	if (numerator <= 0.0)
		PG_RETURN_FLOAT8(0.0);

	PG_RETURN_FLOAT8(sqrt(numerator / (N * (N - 1.0))));
}

/*
 * pgstrom_float8_stddev_pop
 */
Datum
pgstrom_float8_stddev_pop(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	float8	   *transvalues;
	float8		N, sumX, sumX2;
	float8		numerator;

	transvalues = check_float8_array(transarray, "float8_stddev_pop", 3);
	N = transvalues[0];
	sumX = transvalues[1];
	sumX2 = transvalues[2];
	/* Population stddev is undefined when N is 0, so return NULL */
	if (N == 0.0)
		PG_RETURN_NULL();

	numerator = N * sumX2 - sumX * sumX;
	check_float8_value(numerator, isinf(sumX2) || isinf(sumX), true);

	/* Watch out for roundoff error producing a negative numerator */
	if (numerator <= 0.0)
		PG_RETURN_FLOAT8(0.0);

	PG_RETURN_FLOAT8(sqrt(numerator / (N * N)));
}

/*
 * pgstrom_float8_stddev_samp_numeric
 */
Datum
pgstrom_float8_stddev_samp_numeric(PG_FUNCTION_ARGS)
{
	Datum	datum = pgstrom_float8_stddev_samp(fcinfo);

	PG_RETURN_NUMERIC(DirectFunctionCall1(float8_numeric, datum));
}

/*
 * pgstrom_float8_stddev_pop_numeric
 */
Datum
pgstrom_float8_stddev_pop_numeric(PG_FUNCTION_ARGS)
{
	Datum	datum = pgstrom_float8_stddev_pop(fcinfo);

	PG_RETURN_NUMERIC(DirectFunctionCall1(float8_numeric, datum));
}

/*
 * pgstrom_float8_var_samp_numeric
 */
Datum
pgstrom_float8_var_samp_numeric(PG_FUNCTION_ARGS)
{
	Datum	datum = pgstrom_float8_var_samp(fcinfo);

	PG_RETURN_NUMERIC(DirectFunctionCall1(float8_numeric, datum));
}

/*
 * pgstrom_float8_var_pop_numeric
 */
Datum
pgstrom_float8_var_pop_numeric(PG_FUNCTION_ARGS)
{
	Datum	datum = pgstrom_float8_var_pop(fcinfo);

	PG_RETURN_NUMERIC(DirectFunctionCall1(float8_numeric, datum));
}

/*
 * pgstrom_float8_regr_combine
 */
Datum
pgstrom_float8_regr_combine(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray1 = PG_GETARG_ARRAYTYPE_P(0);
	ArrayType  *transarray2 = PG_GETARG_ARRAYTYPE_P(1);
	float8	   *transvalues1;
	float8	   *transvalues2;
	float8		N, sumX, sumX2, sumY, sumY2, sumXY;

	if (!AggCheckCallContext(fcinfo, NULL))
		elog(ERROR, "aggregate function called in non-aggregate context");

	transvalues1 = check_float8_array(transarray1, __FUNCTION__, 6);
	transvalues2 = check_float8_array(transarray2, __FUNCTION__, 6);
	N     = transvalues1[0] + transvalues2[0];
	sumX  = transvalues1[1] + transvalues2[1];
	sumX2 = transvalues1[2] + transvalues2[2];
	sumY  = transvalues1[3] + transvalues2[3];
	sumY2 = transvalues1[4] + transvalues2[4];
	sumXY = transvalues1[5] + transvalues2[5];

	check_float8_value(sumX,  isinf(transvalues1[1]) || isinf(transvalues2[1]), true);
	check_float8_value(sumX2, isinf(transvalues1[2]) || isinf(transvalues2[2]), true);
	check_float8_value(sumY,  isinf(transvalues1[3]) || isinf(transvalues2[3]), true);
	check_float8_value(sumY2, isinf(transvalues1[4]) || isinf(transvalues2[4]), true);
	check_float8_value(sumXY, isinf(transvalues1[5]) || isinf(transvalues2[5]), true);

	transvalues1[0] = N;
	transvalues1[1] = sumX;
	transvalues1[2] = sumX2;
	transvalues1[3] = sumY;
	transvalues1[4] = sumY2;
	transvalues1[5] = sumXY;

	PG_RETURN_ARRAYTYPE_P(transarray1);
}

/*
 * pgstrom_float8_covar_pop
 */
Datum
pgstrom_float8_covar_pop(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	float8     *transvalues;
	float8      N, sumX, sumY, sumXY;
	float8		numerator;

	transvalues = check_float8_array(transarray, __FUNCTION__, 6);
	N = transvalues[0];
	sumX = transvalues[1];
	sumY = transvalues[3];
	sumXY = transvalues[5];

	/* if N is 0 we should return NULL */
	if (N < 1.0)
		PG_RETURN_NULL();
	numerator = N * sumXY - sumX * sumX;
	check_float8_value(numerator, isinf(sumXY) || isinf(sumX) || isinf(sumY), true);

	PG_RETURN_FLOAT8(numerator / (N * N));
}

/*
 * pgstrom_float8_covar_samp
 */
Datum
pgstrom_float8_covar_samp(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	float8     *transvalues;
	float8      N, sumX, sumY, sumXY;
	float8		numerator;

	transvalues = check_float8_array(transarray, __FUNCTION__, 6);
	N = transvalues[0];
	sumX = transvalues[1];
	sumY = transvalues[3];
	sumXY = transvalues[5];

	/* if N is <= 1 we should return NULL */
	if (N < 2.0)
		PG_RETURN_NULL();
	numerator = N * sumXY - sumX * sumX;
	check_float8_value(numerator, isinf(sumXY) || isinf(sumX) || isinf(sumY), true);

	PG_RETURN_FLOAT8(numerator / (N * (N - 1.0)));
}

/*
 * pgstrom_float8_corr
 */
Datum
pgstrom_float8_corr(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	float8     *transvalues;
	float8      N, sumX, sumX2, sumY, sumY2, sumXY;
	float8		numeratorX, numeratorY, numeratorXY;

	transvalues = check_float8_array(transarray, __FUNCTION__, 6);
	N = transvalues[0];
	sumX = transvalues[1];
	sumX2 = transvalues[2];
	sumY = transvalues[3];
	sumY2 = transvalues[4];
	sumXY = transvalues[5];

	/* if N is 0 we should return NULL */
	if (N < 1.0)
		PG_RETURN_NULL();
	numeratorX = N * sumX2 - sumX * sumX;
	numeratorY = N * sumY2 - sumY * sumY;
	numeratorXY = N * sumXY - sumX * sumY;
	check_float8_value(numeratorX, isinf(sumX) || isinf(sumX2), true);
	check_float8_value(numeratorY, isinf(sumY) || isinf(sumY2), true);
	check_float8_value(numeratorXY, isinf(sumXY) || isinf(sumX) || isinf(sumY), true);

	if (numeratorX <= 0 || numeratorY <= 0)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(numeratorXY / sqrt(numeratorX * numeratorY));
}

/*
 * pgstrom_float8_regr_avgx
 */
Datum
pgstrom_float8_regr_avgx(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	float8     *transvalues;
	float8      N, sumX;

	transvalues = check_float8_array(transarray, __FUNCTION__, 6);
	N = transvalues[0];
	sumX = transvalues[1];

	/* if N is 0 we should return NULL */
	if (N < 1.0)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(sumX / N);
}

/*
 * pgstrom_float8_regr_avgy
 */
Datum
pgstrom_float8_regr_avgy(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	float8     *transvalues;
	float8      N, sumY;

	transvalues = check_float8_array(transarray, __FUNCTION__, 6);
	N = transvalues[0];
	sumY = transvalues[3];

	/* if N is 0 we should return NULL */
	if (N < 1.0)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(sumY / N);
}

/*
 * pgstrom_float8_regr_intercept
 */
Datum
pgstrom_float8_regr_intercept(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	float8     *transvalues;
	float8      N, sumX, sumX2, sumY, sumXY;
	float8		numeratorX, numeratorXXY;

	transvalues = check_float8_array(transarray, __FUNCTION__, 6);
	N = transvalues[0];
	sumX = transvalues[1];
	sumX2 = transvalues[2];
	sumY = transvalues[3];
	sumXY = transvalues[5];

	/* if N is 0 we should return NULL */
	if (N < 1.0)
		PG_RETURN_NULL();
	numeratorX = N * sumX2 - sumX * sumX;
	numeratorXXY = sumY * sumX2 - sumX * sumXY;
	check_float8_value(numeratorX, isinf(sumX) || isinf(sumX2), true);
	check_float8_value(numeratorXXY, (isinf(sumY) || isinf(sumX2) ||
									  isinf(sumX) || isinf(sumXY)), true);
	if (numeratorX <= 0)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(numeratorXXY / numeratorX);
}

/*
 * pgstrom_float8_regr_r2
 */
Datum
pgstrom_float8_regr_r2(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	float8     *transvalues;
	float8      N, sumX, sumX2, sumY, sumY2, sumXY;
	float8		numeratorX, numeratorY, numeratorXY;

	transvalues = check_float8_array(transarray, __FUNCTION__, 6);
	N = transvalues[0];
	sumX = transvalues[1];
	sumX2 = transvalues[2];
	sumY = transvalues[3];
	sumY2 = transvalues[4];
	sumXY = transvalues[5];

	/* if N is 0 we should return NULL */
	if (N < 1.0)
		PG_RETURN_NULL();
	numeratorX = N * sumX2 - sumX * sumX;
	numeratorY = N * sumY2 - sumY * sumY;
	numeratorXY = N * sumXY - sumX * sumY;
	check_float8_value(numeratorX, isinf(sumX) || isinf(sumX2), true);
	check_float8_value(numeratorY, isinf(sumY) || isinf(sumY2), true);
	check_float8_value(numeratorXY, isinf(sumXY) || isinf(sumX) || isinf(sumY), true);

	if (numeratorX <= 0.0)
		PG_RETURN_NULL();
	if (numeratorY <= 0.0)
		PG_RETURN_FLOAT8(1.0);
	PG_RETURN_FLOAT8((numeratorXY * numeratorXY) / (numeratorX * numeratorY));
}

/*
 * pgstrom_float8_regr_slope
 */
Datum
pgstrom_float8_regr_slope(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	float8     *transvalues;
	float8      N, sumX, sumX2, sumY, sumXY;
	float8		numeratorX, numeratorXY;

	transvalues = check_float8_array(transarray, __FUNCTION__, 6);
	N = transvalues[0];
	sumX = transvalues[1];
	sumX2 = transvalues[2];
	sumY = transvalues[3];
	sumXY = transvalues[5];

	/* if N is 0 we should return NULL */
	if (N < 1.0)
		PG_RETURN_NULL();
	numeratorX = N * sumX2 - sumX * sumX;
	numeratorXY = N * sumXY - sumX * sumY;
	check_float8_value(numeratorX, isinf(sumX) || isinf(sumX2), true);
	check_float8_value(numeratorXY, isinf(sumXY) || isinf(sumX) || isinf(sumY), true);

	if (numeratorX <= 0)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(numeratorXY / numeratorX);
}

/*
 * pgstrom_float8_regr_sxx
 */
Datum
pgstrom_float8_regr_sxx(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	float8     *transvalues;
	float8      N, sumX, sumX2;
	float8		numeratorX;

	transvalues = check_float8_array(transarray, __FUNCTION__, 6);
	N = transvalues[0];
	sumX = transvalues[1];
	sumX2 = transvalues[2];

	/* if N is 0 we should return NULL */
	if (N < 1.0)
		PG_RETURN_NULL();
	numeratorX = N * sumX2 - sumX * sumX;
	check_float8_value(numeratorX, isinf(sumX) || isinf(sumX2), true);

	if (numeratorX <= 0)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(numeratorX / N);
}

/*
 * pgstrom_float8_regr_syy
 */
Datum
pgstrom_float8_regr_syy(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	float8     *transvalues;
	float8      N, sumY, sumY2;
	float8		numeratorY;

	transvalues = check_float8_array(transarray, __FUNCTION__, 6);
	N = transvalues[0];
	sumY = transvalues[3];
	sumY2 = transvalues[4];

	/* if N is 0 we should return NULL */
	if (N < 1.0)
		PG_RETURN_NULL();
	numeratorY = N * sumY2 - sumY * sumY;
	check_float8_value(numeratorY, isinf(sumY) || isinf(sumY2), true);

	if (numeratorY <= 0)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(numeratorY / N);
}

/*
 * pgstrom_float8_regr_sxy
 */
Datum
pgstrom_float8_regr_sxy(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	float8     *transvalues;
	float8      N, sumX, sumY, sumXY;
	float8		numeratorXY;

	transvalues = check_float8_array(transarray, __FUNCTION__, 6);
	N = transvalues[0];
	sumX = transvalues[1];
	sumY = transvalues[3];
	sumXY = transvalues[5];

	/* if N is 0 we should return NULL */
	if (N < 1.0)
		PG_RETURN_NULL();
	numeratorXY = N * sumXY - sumX * sumY;
	check_float8_value(numeratorXY, isinf(sumXY) || isinf(sumX) || isinf(sumY), true);

	PG_RETURN_FLOAT8(numeratorXY / N);
}
