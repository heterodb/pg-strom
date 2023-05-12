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
PG_FUNCTION_INFO_V1(pgstrom_hll_sketch_new);
PG_FUNCTION_INFO_V1(pgstrom_hll_sketch_merge);
PG_FUNCTION_INFO_V1(pgstrom_hll_count_final);
PG_FUNCTION_INFO_V1(pgstrom_hll_sketch_histogram);

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

/*
 * ----------------------------------------------------------------
 *
 * Hyper-Log-Log support functions
 *
 * ----------------------------------------------------------------
 */

/*
 * Hash-function based on Sip-Hash
 *
 * See https://en.wikipedia.org/wiki/SipHash
 *     and https://github.com/veorq/SipHash
 */
/* default: SipHash-2-4 */
#define cROUNDS 2
#define dROUNDS 4
#define ROTL(x, b) (uint64_t)(((x) << (b)) | ((x) >> (64 - (b))))

#define U8TO64_LE(p)											\
    (((uint64_t)((p)[0]))       | ((uint64_t)((p)[1]) <<  8) |	\
     ((uint64_t)((p)[2]) << 16) | ((uint64_t)((p)[3]) << 24) |	\
     ((uint64_t)((p)[4]) << 32) | ((uint64_t)((p)[5]) << 40) |	\
     ((uint64_t)((p)[6]) << 48) | ((uint64_t)((p)[7]) << 56))

#define SIPROUND					\
    do {							\
        v0 += v1;					\
        v1 = ROTL(v1, 13);			\
        v1 ^= v0;					\
        v0 = ROTL(v0, 32);			\
        v2 += v3;					\
        v3 = ROTL(v3, 16);			\
        v3 ^= v2;					\
        v0 += v3;					\
        v3 = ROTL(v3, 21);			\
        v3 ^= v0;					\
        v2 += v1;					\
        v1 = ROTL(v1, 17);			\
        v1 ^= v2;					\
        v2 = ROTL(v2, 32);			\
    } while (0)

static uint64_t
__pgstrom_hll_siphash_value(const void *ptr, const size_t len)
{
	const unsigned char *ni = (const unsigned char *)ptr;
	uint64_t	v0 = 0x736f6d6570736575UL;
	uint64_t	v1 = 0x646f72616e646f6dUL;
	uint64_t	v2 = 0x6c7967656e657261UL;
	uint64_t	v3 = 0x7465646279746573UL;
	uint64_t	k0 = 0x9c38151cda15a76bUL;	/* random key-0 */
	uint64_t	k1 = 0xfb4ff68fbd3e6658UL;	/* random key-1 */
	uint64_t	m;
	int			i;
    const unsigned char *end = ni + len - (len % sizeof(uint64_t));
    const int	left = len & 7;
    uint64_t	b = ((uint64_t)len) << 56;

    v3 ^= k1;
    v2 ^= k0;
    v1 ^= k1;
    v0 ^= k0;

	for (; ni != end; ni += 8)
	{
		m = U8TO64_LE(ni);
		v3 ^= m;

		for (i = 0; i < cROUNDS; ++i)
			SIPROUND;

		v0 ^= m;
	}

#if 1
	if (left > 0)
	{
		uint64_t	temp = 0;

		memcpy(&temp, ni, left);
		b |= (temp & ((1UL << (BITS_PER_BYTE * left)) - 1));
	}
#else
	/* original code */
	switch (left)
	{
		case 7:
			b |= ((uint64_t)ni[6]) << 48;		__attribute__ ((fallthrough));
		case 6:
			b |= ((uint64_t)ni[5]) << 40;		__attribute__ ((fallthrough));
		case 5:
			b |= ((uint64_t)ni[4]) << 32;		__attribute__ ((fallthrough));
		case 4:
			b |= ((uint64_t)ni[3]) << 24;		__attribute__ ((fallthrough));
		case 3:
			b |= ((uint64_t)ni[2]) << 16;		__attribute__ ((fallthrough));
		case 2:
			b |= ((uint64_t)ni[1]) << 8;		__attribute__ ((fallthrough));
		case 1:
			b |= ((uint64_t)ni[0]);
			break;
		case 0:
			break;
    }
#endif

    v3 ^= b;
	for (i = 0; i < cROUNDS; ++i)
		SIPROUND;

	v0 ^= b;

	v2 ^= 0xff;

	for (i = 0; i < dROUNDS; ++i)
		SIPROUND;

	b = v0 ^ v1 ^ v2 ^ v3;

	return b;
}

/*
 * pgstrom_hll_hash_xxxx functions
 */
static uint64
__pgstrom_hll_hash_int1(Datum datum)
{
	return __pgstrom_hll_siphash_value(&datum, sizeof(int8));
}

static uint64
__pgstrom_hll_hash_int2(Datum datum)
{
	return __pgstrom_hll_siphash_value(&datum, sizeof(int16));
}

static uint64
__pgstrom_hll_hash_int4(Datum datum)
{
	return __pgstrom_hll_siphash_value(&datum, sizeof(int32));
}

static uint64
__pgstrom_hll_hash_int8(Datum datum)
{
	return __pgstrom_hll_siphash_value(&datum, sizeof(int64));
}

static uint64
__pgstrom_hll_hash_numeric(Datum datum)
{
	kern_context kcxt;
	pg_numeric_t num;
	size_t		sz;

	memset(&kcxt, 0, sizeof(kcxt));
	num = pg_numeric_from_varlena(&kcxt, (struct varlena *)datum);
	if (kcxt.errcode != ERRCODE_STROM_SUCCESS)
		elog(ERROR, "failed on hash calculation of device numeric: %s",
			 DatumGetCString(DirectFunctionCall1(numeric_out, datum)));
	sz = offsetof(pg_numeric_t, weight) + sizeof(cl_short);
	return __pgstrom_hll_siphash_value(&num, sz);
}

static uint64
__pgstrom_hll_hash_date(Datum datum)
{
	return __pgstrom_hll_siphash_value(&datum, sizeof(DateADT));
}

static uint64
__pgstrom_hll_hash_time(Datum datum)
{
	return __pgstrom_hll_siphash_value(&datum, sizeof(TimeADT));
}

static uint64
__pgstrom_hll_hash_timetz(Datum datum)
{
	return __pgstrom_hll_siphash_value(DatumGetPointer(datum), sizeof(TimeTzADT));
}

static uint64
__pgstrom_hll_hash_timestamp(Datum datum)
{
	return __pgstrom_hll_siphash_value(&datum, sizeof(Timestamp));
}

static uint64
__pgstrom_hll_hash_timestamptz(Datum datum)
{
	return __pgstrom_hll_siphash_value(&datum, sizeof(TimestampTz));
}

static uint64
__pgstrom_hll_hash_bpchar(Datum datum)
{
	BpChar	   *val = DatumGetBpCharPP(datum);
	int			len = bpchartruelen(VARDATA_ANY(val),
									VARSIZE_ANY_EXHDR(val));
	return __pgstrom_hll_siphash_value(VARDATA_ANY(val), len);
}

static uint64
__pgstrom_hll_hash_varlena(Datum datum)
{
	struct varlena *val = PG_DETOAST_DATUM(datum);

	return __pgstrom_hll_siphash_value(VARDATA_ANY(val), VARSIZE_ANY_EXHDR(val));
}

static uint64
__pgstrom_hll_hash_uuid(Datum datum)
{
	return __pgstrom_hll_siphash_value(DatumGetUUIDP(datum), sizeof(pg_uuid_t));
}

static bytea *
__pgstrom_hll_sketch_update_common(PG_FUNCTION_ARGS, uint64 hash)
{
	MemoryContext	aggcxt;
	bytea		   *hll_state;
	uint8		   *hll_regs;
	uint64			nrooms;
	uint32			index;
	uint32			count;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	nrooms = (1UL << pgstrom_hll_register_bits);
	if (PG_ARGISNULL(0))
	{
		size_t	sz = VARHDRSZ + sizeof(uint8) * nrooms;
		hll_state = MemoryContextAllocZero(aggcxt, sz);
		SET_VARSIZE(hll_state, sz);
	}
	else
	{
		hll_state = PG_GETARG_BYTEA_P(0);
	}
	Assert(VARSIZE(hll_state) == VARHDRSZ + sizeof(uint8) * nrooms);
	hll_regs = (uint8 *)VARDATA(hll_state);

	index = hash & (nrooms - 1);
	count = __builtin_ctzll(hash >> pgstrom_hll_register_bits) + 1;
	if (hll_regs[index] < count)
		hll_regs[index] = count;
	return hll_state;
}

#define PGSTROM_HLL_HANDLER_TEMPLATE(NAME)								\
	PG_FUNCTION_INFO_V1(pgstrom_hll_hash_##NAME);						\
	PG_FUNCTION_INFO_V1(pgstrom_hll_sketch_update_##NAME);				\
	Datum																\
	pgstrom_hll_hash_##NAME(PG_FUNCTION_ARGS)							\
	{																	\
		Datum	arg = PG_GETARG_DATUM(0);								\
		PG_RETURN_UINT64(__pgstrom_hll_hash_##NAME(arg));				\
	}																	\
	Datum																\
	pgstrom_hll_sketch_update_##NAME(PG_FUNCTION_ARGS)					\
	{																	\
		if (PG_ARGISNULL(1))											\
		{																\
			if (PG_ARGISNULL(0))										\
				PG_RETURN_NULL();										\
			PG_RETURN_DATUM(PG_GETARG_DATUM(0));						\
		}																\
		else															\
		{																\
			Datum	arg = PG_GETARG_DATUM(1);							\
			uint64  hash = __pgstrom_hll_hash_##NAME(arg);				\
			bytea  *state;												\
																		\
			state = __pgstrom_hll_sketch_update_common(fcinfo, hash);	\
			PG_RETURN_BYTEA_P(state);									\
		}																\
	}

PGSTROM_HLL_HANDLER_TEMPLATE(int1)
PGSTROM_HLL_HANDLER_TEMPLATE(int2)
PGSTROM_HLL_HANDLER_TEMPLATE(int4)
PGSTROM_HLL_HANDLER_TEMPLATE(int8)
PGSTROM_HLL_HANDLER_TEMPLATE(numeric)
PGSTROM_HLL_HANDLER_TEMPLATE(date)
PGSTROM_HLL_HANDLER_TEMPLATE(time)
PGSTROM_HLL_HANDLER_TEMPLATE(timetz)
PGSTROM_HLL_HANDLER_TEMPLATE(timestamp)
PGSTROM_HLL_HANDLER_TEMPLATE(timestamptz)
PGSTROM_HLL_HANDLER_TEMPLATE(bpchar)
PGSTROM_HLL_HANDLER_TEMPLATE(varlena)
PGSTROM_HLL_HANDLER_TEMPLATE(uuid)

/*
 * pgstrom_hll_sketch_new
 */
Datum
pgstrom_hll_sketch_new(PG_FUNCTION_ARGS)
{
	uint64		nrooms = (1UL << pgstrom_hll_register_bits);
	uint64		hll_hash = DatumGetUInt64(PG_GETARG_DATUM(0));
	bytea	   *hll_state;
	uint8	   *hll_regs;
	uint32		count;
	uint32		index;

	hll_state = palloc0(VARHDRSZ + sizeof(uint8) * nrooms);
	SET_VARSIZE(hll_state, VARHDRSZ + sizeof(uint8) * nrooms);
	hll_regs = (uint8 *)VARDATA(hll_state);

	index = hll_hash & (nrooms - 1);
	Assert(index < nrooms);
	count = __builtin_ctzll(hll_hash >> pgstrom_hll_register_bits) + 1;
	if (hll_regs[index] < count)
		hll_regs[index] = count;

	PG_RETURN_BYTEA_P(hll_state);
}

/*
 * pgstrom_hll_sketch_merge
 */
Datum
pgstrom_hll_sketch_merge(PG_FUNCTION_ARGS)
{
	MemoryContext	aggcxt;
	bytea		   *hll_state = NULL;
	uint8		   *hll_regs;
	bytea		   *new_state;
	uint8		   *new_regs;
	uint32			nrooms;
	uint32			index;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	if (PG_ARGISNULL(0))
	{
		if (PG_ARGISNULL(1))
			PG_RETURN_NULL();
		new_state = PG_GETARG_BYTEA_P(1);
		nrooms = VARSIZE_ANY_EXHDR(new_state);
		if (nrooms < 1 || (nrooms & (nrooms - 1)) != 0)
			elog(ERROR, "HLL sketch must have 2^N rooms (%u)", nrooms);
		hll_state = MemoryContextAllocZero(aggcxt, VARHDRSZ + nrooms);
		SET_VARSIZE(hll_state, VARHDRSZ + nrooms);
		memcpy(VARDATA_ANY(hll_state), VARDATA_ANY(new_state), nrooms);
	}
	else
	{
		hll_state = PG_GETARG_BYTEA_P(0);
		nrooms = VARSIZE_ANY_EXHDR(hll_state);
		if (nrooms < 1 || (nrooms & (nrooms - 1)) != 0)
			elog(ERROR, "HLL sketch must have 2^N rooms (%u)", nrooms);
		if (!PG_ARGISNULL(1))
		{
			new_state = PG_GETARG_BYTEA_P(1);
			if (VARSIZE_ANY_EXHDR(hll_state) != VARSIZE_ANY_EXHDR(new_state))
				elog(ERROR, "incompatible HLL sketch");
			hll_regs = (uint8 *)VARDATA_ANY(hll_state);
			new_regs = (uint8 *)VARDATA_ANY(new_state);
			for (index=0; index < nrooms; index++)
			{
				if (hll_regs[index] < new_regs[index])
                    hll_regs[index] = new_regs[index];
			}
		}
	}
	PG_RETURN_POINTER(hll_state);
}

/*
 * pgstrom_hll_count_final
 */
Datum
pgstrom_hll_count_final(PG_FUNCTION_ARGS)
{
	bytea	   *hll_state;
	uint8	   *hll_regs;
	uint32		nrooms;
	uint32		index;
	double		divider = 0.0;
	double		weight;
	double		estimate;

#if 0
	/*
	 * MEMO: Here to no reason to prohibit to use pgstrom.hll_count_final()
	 * towards preliminary calculated HLL sketch.
	 */
	if (!AggCheckCallContext(fcinfo, NULL))
		elog(ERROR, "aggregate function called in non-aggregate context");
#endif
	if (PG_ARGISNULL(0))
		PG_RETURN_INT64(0);
	/*
	 * MEMO: Hyper-Log-Log merge algorithm
	 * https://ja.wikiqube.net/wiki/HyperLogLog
	 */
	hll_state = PG_GETARG_BYTEA_P(0);
	nrooms = VARSIZE_ANY_EXHDR(hll_state);
	if (nrooms < 1 || (nrooms & (nrooms - 1)) != 0)
		elog(ERROR, "HLL sketch must have 2^N rooms (%u)", nrooms);
	hll_regs = (uint8 *)VARDATA(hll_state);

	for (index = 0; index < nrooms; index++)
		divider += 1.0 / (double)(1UL << hll_regs[index]);
	if (nrooms <= 16)
		weight = 0.673;
	else if (nrooms <= 32)
		weight = 0.697;
	else if (nrooms <= 64)
		weight = 0.709;
	else
		weight = 0.7213 / (1.0 + 1.079 / (double)nrooms);

	estimate = (weight * (double)nrooms * (double)nrooms) / divider;
	PG_RETURN_INT64((int64)estimate);
}



/*
 * pgstrom_hll_sketch_histogram
 */
Datum
pgstrom_hll_sketch_histogram(PG_FUNCTION_ARGS)
{
	bytea	   *hll_state = PG_GETARG_BYTEA_P(0);
	uint8	   *hll_regs;
	uint32		nrooms;
	uint32		index;
	Datum		hll_hist[64];
	int			max_hist = -1;
	ArrayType  *result;

	nrooms = VARSIZE_ANY_EXHDR(hll_state);
	if (nrooms < 1 || (nrooms & (nrooms - 1)) != 0)
		elog(ERROR, "HLL sketch must have 2^N rooms (%u)", nrooms);
	hll_regs = (uint8 *)VARDATA(hll_state);

	memset(hll_hist, 0, sizeof(hll_hist));
	for (index=0; index < nrooms; index++)
	{
		int		value = (int)hll_regs[index];

		if (value < 0 || value >= 64)
			elog(ERROR, "HLL sketch looks corrupted");
		hll_hist[value]++;
		if (max_hist < value)
			max_hist = value;
	}

	if (max_hist < 0)
		PG_RETURN_NULL();

	result = construct_array(hll_hist,
							 max_hist + 1,
							 INT4OID,
							 sizeof(int32),
							 true,
							 'i');
	PG_RETURN_POINTER(result);
}
