/*
 * aggfuncs.c
 *
 * Definition of self-defined aggregate functions, used by GpuPreAgg
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
#include "fmgr.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/cash.h"
#include "utils/numeric.h"
#include <math.h>
#include "pg_strom.h"
#include "cuda_numeric.h"

/*
 * declarations
 */
Datum pgstrom_partial_nrows(PG_FUNCTION_ARGS);
Datum pgstrom_partial_avg_int8(PG_FUNCTION_ARGS);
Datum pgstrom_partial_avg_float8(PG_FUNCTION_ARGS);
Datum pgstrom_partial_avg_numeric(PG_FUNCTION_ARGS);
Datum pgstrom_final_avg_int8_accum(PG_FUNCTION_ARGS);
Datum pgstrom_final_avg_int8(PG_FUNCTION_ARGS);
Datum pgstrom_final_avg_float8_accum(PG_FUNCTION_ARGS);
Datum pgstrom_final_avg_float8(PG_FUNCTION_ARGS);
Datum pgstrom_final_avg_numeric_accum(PG_FUNCTION_ARGS);
Datum pgstrom_final_avg_numeric(PG_FUNCTION_ARGS);
Datum pgstrom_partial_min(PG_FUNCTION_ARGS);
Datum pgstrom_partial_max(PG_FUNCTION_ARGS);
Datum pgstrom_partial_sum_any(PG_FUNCTION_ARGS);
Datum pgstrom_partial_sum_x2_float4(PG_FUNCTION_ARGS);
Datum pgstrom_partial_sum_x2_float8(PG_FUNCTION_ARGS);
Datum pgstrom_partial_sum_x2_numeric(PG_FUNCTION_ARGS);
Datum pgstrom_partial_cov_x(PG_FUNCTION_ARGS);
Datum pgstrom_partial_cov_y(PG_FUNCTION_ARGS);
Datum pgstrom_partial_cov_x2(PG_FUNCTION_ARGS);
Datum pgstrom_partial_cov_y2(PG_FUNCTION_ARGS);
Datum pgstrom_partial_cov_xy(PG_FUNCTION_ARGS);


#if 0
Datum pgstrom_partial_avg_fp8(PG_FUNCTION_ARGS);
Datum pgstrom_partial_sum_int8(PG_FUNCTION_ARGS);
Datum pgstrom_partial_sum_numeric(PG_FUNCTION_ARGS);
Datum pgstrom_partial_variance_fp8(PG_FUNCTION_ARGS);
Datum pgstrom_partial_covar_fp8(PG_FUNCTION_ARGS);



/* copy from utils/adt/numeric.c */
typedef int16 NumericDigit;

/* copy from utils/adt/numeric.c */
typedef struct NumericVar
{
	int			ndigits;		/* # of digits in digits[] - can be 0! */
	int			weight;			/* weight of first digit */
	int			sign;			/* NUMERIC_POS, NUMERIC_NEG, or NUMERIC_NAN */
	int			dscale;			/* display scale */
	NumericDigit *buf;			/* start of palloc'd space for digits[] */
	NumericDigit *digits;		/* base-NBASE digits */
} NumericVar;

/* copy from utils/adt/numeric.c */
typedef struct NumericAggState
{
	bool		calcSumX2;		/* if true, calculate sumX2 */
	MemoryContext agg_context;	/* context we're calculating in */
	int64		N;				/* count of processed numbers */
	NumericVar	sumX;			/* sum of processed numbers */
	NumericVar	sumX2;			/* sum of squares of processed numbers */
	int			maxScale;		/* maximum scale seen so far */
	int64		maxScaleCount;	/* number of values seen with maximum scale */
	int64		NaNcount;		/* count of NaN values (not included in N!) */
} NumericAggState;

/* copy from utils/adt/numeric.c */
static inline NumericAggState *
makeNumericAggState(bool calcSumX2)
{
	NumericAggState *state = palloc0(sizeof(NumericAggState));

	state->calcSumX2 = calcSumX2;
	state->agg_context = CurrentMemoryContext;

	return state;
}

/* copy from utils/adt/numeric.c */
#ifdef HAVE_INT128
typedef struct Int128AggState
{
	bool		calcSumX2;      /* if true, calculate sumX2 */
	int64		N;              /* count of processed numbers */
	int128		sumX;           /* sum of processed numbers */
	int128		sumX2;          /* sum of squares of processed numbers */
} Int128AggState;

static inline Int128AggState *
makeInt128AggState(bool calcSumX2)
{
	Int128AggState *state = palloc0(sizeof(Int128AggState));

	state->calcSumX2 = calcSumX2;

	return state;
}
#define makePolyNumAggState	makeInt128AggState
typedef Int128AggState		PolyNumAggState;
#else
#define makePolyNumAggState	makeNumericAggState
typedef NumericAggState		PolyNumAggState;
#endif
#endif




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
pgstrom_partial_avg_numeric(PG_FUNCTION_ARGS)
{
	ArrayType  *result;
	Datum		items[2];

	items[0] = DirectFunctionCall1(int8_numeric,
								   PG_GETARG_DATUM(0));	/* nrows(int8) */
	items[1] = PG_GETARG_DATUM(1);	/* p_sum(numeric) */
	result = construct_array(items, 2, NUMERICOID,
							 -1, false, 'i');
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
		x[1] += y[0];
	}
	PG_RETURN_POINTER(xarray);
}

Datum
pgstrom_final_avg_int8(PG_FUNCTION_ARGS)
{
	ArrayType	   *xarray = PG_GETARG_ARRAYTYPE_P(0);
	int64		   *x = (int64 *)ARR_DATA_PTR(xarray);

	return DirectFunctionCall2(numeric_div,
							   DirectFunctionCall1(int8_numeric,
												   Int64GetDatum(x[0])),
							   DirectFunctionCall1(int8_numeric,
												   Int64GetDatum(x[1])));
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
		x[1] += y[0];
	}
	PG_RETURN_POINTER(xarray);
}

Datum
pgstrom_final_avg_float8(PG_FUNCTION_ARGS)
{
	ArrayType	   *xarray = PG_GETARG_ARRAYTYPE_P(0);
	float8		   *x = (float8 *)ARR_DATA_PTR(xarray);

	PG_RETURN_FLOAT8(x[1] / x[0]);
}

Datum
pgstrom_final_avg_numeric_accum(PG_FUNCTION_ARGS)
{
	MemoryContext	aggcxt;
	MemoryContext	oldcxt;
	ArrayType	   *xarray;
	ArrayType	   *yarray;
	Datum			x0, x1;
	Datum			y0, y1;
	Datum			items[2];
	bool			isnull[4];
	int				index;

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

		index = 0;
		x0 = array_ref(xarray, 1, &index, -1, -1, false, 'i', &isnull[0]);
		y0 = array_ref(yarray, 1, &index, -1, -1, false, 'i', &isnull[1]);
		index = 1;
		x1 = array_ref(xarray, 1, &index, -1, -1, false, 'i', &isnull[2]);
		y1 = array_ref(yarray, 1, &index, -1, -1, false, 'i', &isnull[3]);

		if (isnull[0] || isnull[1] || isnull[2] || isnull[3])
			elog(ERROR, "unexpected internal state");

		items[0] = DirectFunctionCall2(numeric_add, x0, y0);
		items[1] = DirectFunctionCall2(numeric_add, x1, y1);

		oldcxt = MemoryContextSwitchTo(aggcxt);
		xarray = construct_array(items, 2, NUMERICOID,
								 -1, false, 'i');
		MemoryContextSwitchTo(oldcxt);
	}
	PG_RETURN_POINTER(xarray);
}

Datum
pgstrom_final_avg_numeric(PG_FUNCTION_ARGS)
{
	ArrayType  *xarray = PG_GETARG_ARRAYTYPE_P(0);
	Datum		nrows;
	Datum		sum;
	int			index = 0;
	bool		isnull[2];

	nrows = array_ref(xarray, 1, &index, -1, -1, false, 'i', &isnull[0]);
	index++;
	sum = array_ref(xarray, 1, &index, -1, -1, false, 'i', &isnull[1]);

	if (isnull[0] || isnull[1])
		elog(ERROR, "unexpected internal state");

	return DirectFunctionCall2(numeric_div, sum, nrows);
}

/*
 * pgstrom.pmin(anyelement)
 */
Datum
pgstrom_partial_min(PG_FUNCTION_ARGS)
{
	PG_RETURN_DATUM(PG_GETARG_DATUM(0));
}

/*
 * pgstrom.pmax(anyelement)
 */
Datum
pgstrom_partial_max(PG_FUNCTION_ARGS)
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
















#ifdef NOT_USED
/*
 * gpupreagg_psum_int8 - alternative function for SUM(int8)
 */
Datum
pgstrom_partial_sum_int8(PG_FUNCTION_ARGS)
{
	PolyNumAggState *state = makePolyNumAggState(false);
	DirectFunctionCall2(int8_avg_accum,
						PointerGetDatum(state),
						PG_GETARG_DATUM(0));    /* partial sum */
	PG_RETURN_POINTER(state);
}

/*
 * gpupreagg_psum_numeric - alternative function for SUM(numeric)
 */
Datum
pgstrom_partial_sum_numeric(PG_FUNCTION_ARGS)
{
	NumericAggState *state = makeNumericAggState(false);
	DirectFunctionCall2(numeric_avg_accum,
						PointerGetDatum(state),
						PG_GETARG_DATUM(0));	/* partial sum */
	PG_RETURN_POINTER(state);
}

/*
 * pgstrom_partial_variance_fp8 - alternative function for variance and similar
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
 * pgstrom_partial_covar_fp8 - alternative function for covariance and similar
 */
Datum
pgstrom_partial_covar_fp8(PG_FUNCTION_ARGS)
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

/* gpupreagg_psum_* - placeholder function that generates partial sum
 * of the arguments. _x2 generates square value of the input
 */
Datum
gpupreagg_psum_int(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 1);
	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	PG_RETURN_INT64(PG_GETARG_INT64(0));
}
PG_FUNCTION_INFO_V1(gpupreagg_psum_int);

Datum
gpupreagg_psum_float4(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 1);
	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	PG_RETURN_FLOAT4(PG_GETARG_FLOAT4(0));
}
PG_FUNCTION_INFO_V1(gpupreagg_psum_float4);

Datum
gpupreagg_psum_float8(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 1);
	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(PG_GETARG_FLOAT8(0));
}
PG_FUNCTION_INFO_V1(gpupreagg_psum_float8);

Datum
gpupreagg_psum_x2_float(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 1);
	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	PG_RETURN_DATUM(DirectFunctionCall2(float8mul,
										PG_GETARG_FLOAT8(0),
										PG_GETARG_FLOAT8(0)));
}
PG_FUNCTION_INFO_V1(gpupreagg_psum_x2_float);

Datum
gpupreagg_psum_numeric(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 1);
	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	PG_RETURN_NUMERIC(PG_GETARG_NUMERIC(0));
}
PG_FUNCTION_INFO_V1(gpupreagg_psum_numeric);

Datum
gpupreagg_psum_x2_numeric(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 1);
	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();

	PG_RETURN_NUMERIC(DirectFunctionCall2(numeric_mul,
										  PG_GETARG_DATUM(0),
										  PG_GETARG_DATUM(0)));
}
PG_FUNCTION_INFO_V1(gpupreagg_psum_x2_numeric);

Datum
gpupreagg_psum_money(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 1);
	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	PG_RETURN_CASH(PG_GETARG_CASH(0));
}
PG_FUNCTION_INFO_V1(gpupreagg_psum_money);

/* gpupreagg_corr_psum - placeholder function that generates partial sum
 * of the arguments. _x2 generates square value of the input
 */
Datum
gpupreagg_corr_psum_x(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 3);
	/* Aggregate Filter */
	if (PG_ARGISNULL(0) || !PG_GETARG_BOOL(0))
		PG_RETURN_NULL();
	/* NULL checks */
	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(PG_GETARG_FLOAT8(1));
}
PG_FUNCTION_INFO_V1(gpupreagg_corr_psum_x);

Datum
gpupreagg_corr_psum_y(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 3);
	/* Aggregate Filter */
	if (PG_ARGISNULL(0) || !PG_GETARG_BOOL(0))
		PG_RETURN_NULL();
	/* NULL checks */
	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(PG_GETARG_FLOAT8(2));
}
PG_FUNCTION_INFO_V1(gpupreagg_corr_psum_y);

Datum
gpupreagg_corr_psum_x2(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 3);
	/* Aggregate Filter */
	if (PG_ARGISNULL(0) || !PG_GETARG_BOOL(0))
		PG_RETURN_NULL();
	/* NULL checks */
	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		PG_RETURN_NULL();
	/* calculation of X*X with overflow checks */
	PG_RETURN_DATUM(DirectFunctionCall2(float8mul,
										PG_GETARG_FLOAT8(1),
										PG_GETARG_FLOAT8(1)));
}
PG_FUNCTION_INFO_V1(gpupreagg_corr_psum_x2);

Datum
gpupreagg_corr_psum_y2(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 3);
	/* Aggregate Filter */
	if (PG_ARGISNULL(0) || !PG_GETARG_BOOL(0))
		PG_RETURN_NULL();
	/* NULL checks */
	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		PG_RETURN_NULL();
	/* calculation of X*X with overflow checks */
	PG_RETURN_DATUM(DirectFunctionCall2(float8mul,
										PG_GETARG_FLOAT8(2),
										PG_GETARG_FLOAT8(2)));
}
PG_FUNCTION_INFO_V1(gpupreagg_corr_psum_y2);

Datum
gpupreagg_corr_psum_xy(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 3);
	/* Aggregate Filter */
	if (PG_ARGISNULL(0) || !PG_GETARG_BOOL(0))
		PG_RETURN_NULL();
	/* NULL checks */
	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		PG_RETURN_NULL();
	/* calculation of X*X with overflow checks */
	PG_RETURN_DATUM(DirectFunctionCall2(float8mul,
										PG_GETARG_FLOAT8(1),
										PG_GETARG_FLOAT8(2)));
}
PG_FUNCTION_INFO_V1(gpupreagg_corr_psum_xy);

/*
 * ex_avg() - an enhanced average calculation that takes two arguments;
 * number of rows in this group and partial sum of the value.
 * Then, it eventually generate mathmatically compatible average value.
 */
static int64 *
check_int64_array(ArrayType *transarray, int n)
{
	if (ARR_NDIM(transarray) != 1 ||
		ARR_DIMS(transarray)[0] != n ||
		ARR_HASNULL(transarray) ||
		ARR_ELEMTYPE(transarray) != INT8OID)
		elog(ERROR, "Two elements int8 array is expected");
	return (int64 *) ARR_DATA_PTR(transarray);
}

Datum
pgstrom_avg_int8_accum(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	int32		nrows = PG_GETARG_INT32(1);
	int64		psumX = PG_GETARG_INT64(2);
	int64	   *transvalues;
	int64		newN;
	int64		newSumX;

	transvalues = check_int64_array(transarray, 2);
	newN = transvalues[0] + nrows;
	newSumX = transvalues[1] + psumX;

	if (AggCheckCallContext(fcinfo, NULL))
	{
		transvalues[0] = newN;
		transvalues[1] = newSumX;

		PG_RETURN_ARRAYTYPE_P(transarray);
	}
	else
	{
		Datum		transdatums[2];
		ArrayType  *result;

		transdatums[0] = Int64GetDatumFast(newN);
		transdatums[1] = Int64GetDatumFast(newSumX);

		result = construct_array(transdatums, 2,
								 INT8OID,
								 sizeof(int64), FLOAT8PASSBYVAL, 'd');
		PG_RETURN_ARRAYTYPE_P(result);
	}
}
PG_FUNCTION_INFO_V1(pgstrom_avg_int8_accum);

Datum
pgstrom_sum_int8_accum(PG_FUNCTION_ARGS)
{
	int64		newval;

	if (PG_ARGISNULL(0))
	{
		if (PG_ARGISNULL(1))
			PG_RETURN_NULL();   /* still no non-null */
		/* This is the first non-null input. */
		newval = PG_GETARG_INT64(1);
	}
	else
	{
		newval = PG_GETARG_INT64(0);

		if (!PG_ARGISNULL(1))
			newval += PG_GETARG_INT64(1);
	}
	PG_RETURN_INT64(newval);
}
PG_FUNCTION_INFO_V1(pgstrom_sum_int8_accum);

/*
 * numeric_agg_state - self version of aggregation internal state; that
 * can keep N, sum(X) and sum(X*X) in numeric data-type.
 */
typedef struct
{
	int64	N;
	Datum	sumX;
	Datum	sumX2;
} numeric_agg_state;

Datum
pgstrom_int8_avg_accum(PG_FUNCTION_ARGS)
{
	int32			nrows = PG_GETARG_INT32(1);
	Datum			addNum;
	MemoryContext	aggcxt;
	MemoryContext	oldcxt;
	numeric_agg_state *state;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");

	if (PG_ARGISNULL(1))
		nrows = 0;
	else if (nrows < 0)
		elog(ERROR, "Bug? negative nrows were given");

	/* make a state object and update it */
	oldcxt = MemoryContextSwitchTo(aggcxt);
	state = PG_ARGISNULL(0) ? NULL : (numeric_agg_state *)PG_GETARG_POINTER(0);
	if (!state)
	{
		state = palloc0(sizeof(numeric_agg_state));
		state->N = 0;
		state->sumX = DirectFunctionCall3(numeric_in,
										  CStringGetDatum("0"),
										  ObjectIdGetDatum(0),
										  Int32GetDatum(-1));
	}

	if (nrows > 0 && !PG_ARGISNULL(2))
	{
		state->N += nrows;
		addNum = DirectFunctionCall1(int8_numeric, PG_GETARG_DATUM(2));
		state->sumX = DirectFunctionCall2(numeric_add, state->sumX, addNum);
	}
	MemoryContextSwitchTo(oldcxt);

	PG_RETURN_POINTER(state);
}
PG_FUNCTION_INFO_V1(pgstrom_int8_avg_accum);

Datum
pgstrom_numeric_avg_accum(PG_FUNCTION_ARGS)
{
	int32			nrows = PG_GETARG_INT32(1);
	MemoryContext	aggcxt;
	MemoryContext	oldcxt;
	numeric_agg_state *state;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");

	if (PG_ARGISNULL(1))
		nrows = 0;
	else if (nrows < 0)
		elog(ERROR, "Bug? negative nrows were given");

	/* make a state object and update it */
	oldcxt = MemoryContextSwitchTo(aggcxt);
	state = PG_ARGISNULL(0) ? NULL : (numeric_agg_state *)PG_GETARG_POINTER(0);
	if (!state)
	{
		state = palloc0(sizeof(numeric_agg_state));
		state->N = 0;
		state->sumX = DirectFunctionCall3(numeric_in,
										  CStringGetDatum("0"),
										  ObjectIdGetDatum(0),
										  Int32GetDatum(-1));
	}

	if (nrows > 0 && !PG_ARGISNULL(2))
	{
		state->N += nrows;
		state->sumX = DirectFunctionCall2(numeric_add,
										  state->sumX,
										  PG_GETARG_DATUM(2));
	}
	MemoryContextSwitchTo(oldcxt);

	PG_RETURN_POINTER(state);
}
PG_FUNCTION_INFO_V1(pgstrom_numeric_avg_accum);

Datum
pgstrom_numeric_avg_final(PG_FUNCTION_ARGS)
{
	numeric_agg_state *state;
	Datum		vN;
	Datum		result;

	state = PG_ARGISNULL(0) ? NULL : (numeric_agg_state *)PG_GETARG_POINTER(0);

	/* If there were no non-null inputs, return NULL */
	if (state == NULL || state->N == 0)
		PG_RETURN_NULL();
	/* If any NaN value is accumlated, return NaN */
	if (numeric_is_nan(DatumGetNumeric(state->sumX)))
		PG_RETURN_NUMERIC(state->sumX);

	vN = DirectFunctionCall1(int8_numeric, Int64GetDatum(state->N));
	result = DirectFunctionCall2(numeric_div, state->sumX, vN);

	PG_RETURN_NUMERIC(result);
}
PG_FUNCTION_INFO_V1(pgstrom_numeric_avg_final);

/* logic copied from utils/adt/float.c */
static inline float8 *
check_float8_array(ArrayType *transarray, int nitems)
{
	/*
	 * We expect the input to be an N-element float array; verify that. We
	 * don't need to use deconstruct_array() since the array data is just
	 * going to look like a C array of N float8 values.
	 */
	if (ARR_NDIM(transarray) != 1 ||
		ARR_DIMS(transarray)[0] != nitems ||
		ARR_HASNULL(transarray) ||
		ARR_ELEMTYPE(transarray) != FLOAT8OID)
		elog(ERROR, "%d-elements float8 array is expected", nitems);
	return (float8 *) ARR_DATA_PTR(transarray);
}

/* logic copied from utils/adt/float.c */
static inline void
check_float8_valid(float8 value, bool inf_is_valid, bool zero_is_valid)
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

Datum
pgstrom_sum_float8_accum(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	int32		nrows = PG_GETARG_INT32(1);
	float8		psumX = PG_GETARG_FLOAT8(2);
	float8	   *transvalues;
	float8		newN;
	float8		newSumX;

	transvalues = check_float8_array(transarray, 3);
	newN = transvalues[0] + (float8) nrows;
	newSumX = transvalues[1] + psumX;
	check_float8_valid(newSumX, isinf(transvalues[1]) || isinf(psumX), true);

	if (AggCheckCallContext(fcinfo, NULL))
	{
		transvalues[0] = newN;
		transvalues[1] = newSumX;
		transvalues[2] = 0.0;	/* dummy */

		PG_RETURN_ARRAYTYPE_P(transarray);
	}
	else
	{
		Datum		transdatums[3];
		ArrayType  *result;

		transdatums[0] = Float8GetDatumFast(newN);
		transdatums[1] = Float8GetDatumFast(newSumX);
		transdatums[2] = Float8GetDatumFast(0.0);

		result = construct_array(transdatums, 3,
								 FLOAT8OID,
								 sizeof(float8), FLOAT8PASSBYVAL, 'd');
		PG_RETURN_ARRAYTYPE_P(result);
	}
}
PG_FUNCTION_INFO_V1(pgstrom_sum_float8_accum);

/*
 * variance and stddev - mathmatical compatible result can be lead using
 * nrows, psum(X) and psum(X*X). So, we track these variables.
 */
Datum
pgstrom_variance_float8_accum(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	int32		nrows = PG_GETARG_INT32(1);
	float8		psumX = PG_GETARG_FLOAT8(2);
	float8		psumX2 = PG_GETARG_FLOAT8(3);
	float8	   *transvalues;
	float8		newN;
	float8		newSumX;
	float8		newSumX2;

	transvalues = check_float8_array(transarray, 3);
	newN = transvalues[0] + (float8) nrows;
	newSumX = transvalues[1] + psumX;
	check_float8_valid(newSumX, isinf(transvalues[1]) || isinf(psumX), true);
	newSumX2 = transvalues[2] + psumX2;
	check_float8_valid(newSumX2, isinf(transvalues[2]) || isinf(psumX2), true);

	if (AggCheckCallContext(fcinfo, NULL))
	{
		transvalues[0] = newN;
		transvalues[1] = newSumX;
		transvalues[2] = newSumX2;

		PG_RETURN_ARRAYTYPE_P(transarray);
	}
	else
	{
		Datum		transdatums[3];
		ArrayType  *result;

		transdatums[0] = Float8GetDatumFast(newN);
		transdatums[1] = Float8GetDatumFast(newSumX);
		transdatums[2] = Float8GetDatumFast(newSumX2);

		result = construct_array(transdatums, 3,
								 FLOAT8OID,
								 sizeof(float8), FLOAT8PASSBYVAL, 'd');
		PG_RETURN_ARRAYTYPE_P(result);
	}
}
PG_FUNCTION_INFO_V1(pgstrom_variance_float8_accum);

Datum
pgstrom_numeric_var_accum(PG_FUNCTION_ARGS)
{
	int32			nrows = PG_GETARG_INT32(1);
	MemoryContext	aggcxt;
	MemoryContext	oldcxt;
	numeric_agg_state *state;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");

	if (PG_ARGISNULL(1))
		nrows = 0;
	else if (nrows < 0)
		elog(ERROR, "Bug? negative nrows were given");

	/* make a state object and update it */
	oldcxt = MemoryContextSwitchTo(aggcxt);
	state = PG_ARGISNULL(0) ? NULL : (numeric_agg_state *)PG_GETARG_POINTER(0);
	if (!state)
	{
		state = palloc0(sizeof(numeric_agg_state));
		state->N = 0;
		state->sumX = DirectFunctionCall3(numeric_in,
										  CStringGetDatum("0"),
										  ObjectIdGetDatum(0),
										  Int32GetDatum(-1));
		state->sumX2 = DirectFunctionCall3(numeric_in,
										   CStringGetDatum("0"),
										   ObjectIdGetDatum(0),
										   Int32GetDatum(-1));
	}

	if (nrows > 0 && !PG_ARGISNULL(2) && !PG_ARGISNULL(3))
	{
		state->N += nrows;
		state->sumX = DirectFunctionCall2(numeric_add,
										  state->sumX,
										  PG_GETARG_DATUM(2));
		state->sumX2 = DirectFunctionCall2(numeric_add,
										   state->sumX2,
										   PG_GETARG_DATUM(3));
	}
	MemoryContextSwitchTo(oldcxt);

	PG_RETURN_POINTER(state);
}
PG_FUNCTION_INFO_V1(pgstrom_numeric_var_accum);

static Numeric
pgstrom_numeric_stddev_internal(numeric_agg_state *state,
								bool variance, bool sample)
{
	Datum	vZero;
	Datum	vN;
	Datum	vN2;
	Datum	vSumX;
	Datum	vSumX2;
	Datum	result;

	if (state == NULL)
		return NULL;
	/* NaN checks */
	if (numeric_is_nan(DatumGetNumeric(state->sumX)))
		return DatumGetNumeric(state->sumX);
	if (numeric_is_nan(DatumGetNumeric(state->sumX2)))
		return DatumGetNumeric(state->sumX2);

	/*
	 * Sample stddev and variance are undefined when N <= 1; population stddev
	 * is undefined when N == 0. Return NULL in either case.
	 */
	if (sample ? state->N <= 1 : state->N <= 0)
		return NULL;

	/* const_zero = (Numeric)0 */
	vZero  = DirectFunctionCall3(numeric_in,
								 CStringGetDatum("0"),
								 ObjectIdGetDatum(0),
								 Int32GetDatum(-1));
	/* vN = (Numeric)N */
	vN = DirectFunctionCall1(int8_numeric, Int64GetDatum(state->N));
	/* vsumX = sumX * sumX */
	vSumX = DirectFunctionCall2(numeric_mul, state->sumX, state->sumX);
	/* vsumX2 = N * sumX2 */
	vSumX2 = DirectFunctionCall2(numeric_mul, state->sumX2, vN);
	/* N * sumX2 - sumX * sumX */
	vSumX2 = DirectFunctionCall2(numeric_sub, vSumX2, vSumX);

	/* Watch out for roundoff error producing a negative numerator */
	if (DirectFunctionCall2(numeric_cmp, vSumX2, vZero) <= 0)
		return DatumGetNumeric(vZero);

	if (!sample)
		vN2 = DirectFunctionCall2(numeric_mul, vN, vN);	/* N * N */
	else
	{
		Datum	vOne;
		Datum	vNminus;

		vOne = DirectFunctionCall3(numeric_in,
								   CStringGetDatum("1"),
								   ObjectIdGetDatum(0),
								   Int32GetDatum(-1));
		vNminus = DirectFunctionCall2(numeric_sub, vN, vOne);
		vN2 = DirectFunctionCall2(numeric_mul, vN, vNminus); /* N * (N - 1) */
	}
	/* variance */
	result = DirectFunctionCall2(numeric_div, vSumX2, vN2);
	/* stddev? */
	if (!variance)
		result = DirectFunctionCall1(numeric_sqrt, result);

	return DatumGetNumeric(result);
}

Datum
pgstrom_numeric_var_samp(PG_FUNCTION_ARGS)
{
	numeric_agg_state *state;
	Numeric		result;

	state = PG_ARGISNULL(0) ? NULL : (numeric_agg_state *)PG_GETARG_POINTER(0);

	result = pgstrom_numeric_stddev_internal(state, true, true);
	if (!result)
		PG_RETURN_NULL();

	PG_RETURN_NUMERIC(result);
}
PG_FUNCTION_INFO_V1(pgstrom_numeric_var_samp);

Datum
pgstrom_numeric_stddev_samp(PG_FUNCTION_ARGS)
{
	numeric_agg_state *state;
	Numeric		result;

	state = PG_ARGISNULL(0) ? NULL : (numeric_agg_state *)PG_GETARG_POINTER(0);

	result = pgstrom_numeric_stddev_internal(state, false, true);
	if (!result)
		PG_RETURN_NULL();

	PG_RETURN_NUMERIC(result);
}
PG_FUNCTION_INFO_V1(pgstrom_numeric_stddev_samp);

Datum
pgstrom_numeric_var_pop(PG_FUNCTION_ARGS)
{
	numeric_agg_state *state;
	Numeric		result;

	state = PG_ARGISNULL(0) ? NULL : (numeric_agg_state *)PG_GETARG_POINTER(0);

	result = pgstrom_numeric_stddev_internal(state, true, false);
	if (!result)
		PG_RETURN_NULL();

	PG_RETURN_NUMERIC(result);
}
PG_FUNCTION_INFO_V1(pgstrom_numeric_var_pop);

Datum
pgstrom_numeric_stddev_pop(PG_FUNCTION_ARGS)
{
	numeric_agg_state *state;
	Numeric		result;

	state = PG_ARGISNULL(0) ? NULL : (numeric_agg_state *)PG_GETARG_POINTER(0);

	result = pgstrom_numeric_stddev_internal(state, false, false);
	if (!result)
		PG_RETURN_NULL();

	PG_RETURN_NUMERIC(result);
}
PG_FUNCTION_INFO_V1(pgstrom_numeric_stddev_pop);

/*
 * covariance - mathmatical compatible result can be lead using
 * nrows, psum(X), psum(X*X), psum(Y), psum(Y*Y), psum(X*Y)
 */
Datum
pgstrom_covariance_float8_accum(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	int32		nrows  = PG_GETARG_INT32(1);
	float8		psumX  = PG_GETARG_FLOAT8(2);
	float8		psumX2 = PG_GETARG_FLOAT8(3);
	float8		psumY  = PG_GETARG_FLOAT8(4);
	float8		psumY2 = PG_GETARG_FLOAT8(5);
	float8		psumXY = PG_GETARG_FLOAT8(6);
	float8	   *transvalues;
	float8		newN;
	float8		newSumX;
	float8		newSumX2;
	float8		newSumY;
	float8		newSumY2;
	float8		newSumXY;

	transvalues = check_float8_array(transarray, 6);
	newN = transvalues[0] + (float8) nrows;
	newSumX = transvalues[1] + psumX;
	check_float8_valid(newSumX, isinf(transvalues[1]) || isinf(psumX), true);
	newSumX2 = transvalues[2] + psumX2;
	check_float8_valid(newSumX2, isinf(transvalues[2]) || isinf(psumX2), true);
	newSumY = transvalues[3] + psumY;
	check_float8_valid(newSumY, isinf(transvalues[3]) || isinf(psumY), true);
	newSumY2 = transvalues[4] + psumY2;
	check_float8_valid(newSumY2, isinf(transvalues[4]) || isinf(psumY2), true);
	newSumXY = transvalues[5] + psumXY;
	check_float8_valid(newSumXY, isinf(transvalues[5]) || isinf(psumXY), true);

	if (AggCheckCallContext(fcinfo, NULL))
	{
		transvalues[0] = newN;
		transvalues[1] = newSumX;
		transvalues[2] = newSumX2;
		transvalues[3] = newSumY;
		transvalues[4] = newSumY2;
		transvalues[5] = newSumXY;

		PG_RETURN_ARRAYTYPE_P(transarray);
	}
	else
	{
		Datum		transdatums[6];
		ArrayType  *result;

		transdatums[0] = Float8GetDatumFast(newN);
		transdatums[1] = Float8GetDatumFast(newSumX);
		transdatums[2] = Float8GetDatumFast(newSumX2);
		transdatums[3] = Float8GetDatumFast(newSumY);
		transdatums[4] = Float8GetDatumFast(newSumY2);
		transdatums[5] = Float8GetDatumFast(newSumXY);

		result = construct_array(transdatums, 6,
								 FLOAT8OID,
								 sizeof(float8), FLOAT8PASSBYVAL, 'd');
		PG_RETURN_ARRAYTYPE_P(result);
	}
}
PG_FUNCTION_INFO_V1(pgstrom_covariance_float8_accum);
#endif
