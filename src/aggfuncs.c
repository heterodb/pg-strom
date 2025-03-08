/*
 * aggfuncs.c
 *
 * Definition of self-defined aggregate functions, used by GpuPreAgg
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "float2.h"

/*
 * Functions Declaration
 */
PG_FUNCTION_INFO_V1(pgstrom_partial_nrows);
PG_FUNCTION_INFO_V1(pgstrom_partial_nrows_filtered);

PG_FUNCTION_INFO_V1(pgstrom_partial_minmax_int64);
PG_FUNCTION_INFO_V1(pgstrom_partial_minmax_fp64);
PG_FUNCTION_INFO_V1(pgstrom_fmin_trans_int64);
PG_FUNCTION_INFO_V1(pgstrom_fmin_trans_fp64);
PG_FUNCTION_INFO_V1(pgstrom_fmax_trans_int64);
PG_FUNCTION_INFO_V1(pgstrom_fmax_trans_fp64);
PG_FUNCTION_INFO_V1(pgstrom_fminmax_final_int8);
PG_FUNCTION_INFO_V1(pgstrom_fminmax_final_int16);
PG_FUNCTION_INFO_V1(pgstrom_fminmax_final_int32);
PG_FUNCTION_INFO_V1(pgstrom_fminmax_final_int64);
PG_FUNCTION_INFO_V1(pgstrom_fminmax_final_fp16);
PG_FUNCTION_INFO_V1(pgstrom_fminmax_final_fp32);
PG_FUNCTION_INFO_V1(pgstrom_fminmax_final_fp64);
PG_FUNCTION_INFO_V1(pgstrom_fminmax_final_numeric);

PG_FUNCTION_INFO_V1(pgstrom_partial_sum_int);
PG_FUNCTION_INFO_V1(pgstrom_partial_sum_int64);
PG_FUNCTION_INFO_V1(pgstrom_partial_sum_fp);
PG_FUNCTION_INFO_V1(pgstrom_partial_sum_numeric);
PG_FUNCTION_INFO_V1(pgstrom_partial_sum_cash);
PG_FUNCTION_INFO_V1(pgstrom_fsum_trans_int);
PG_FUNCTION_INFO_V1(pgstrom_fsum_trans_fp);
PG_FUNCTION_INFO_V1(pgstrom_fsum_trans_numeric);
PG_FUNCTION_INFO_V1(pgstrom_fsum_final_int);
PG_FUNCTION_INFO_V1(pgstrom_fsum_final_int_as_numeric);
PG_FUNCTION_INFO_V1(pgstrom_fsum_final_int_as_cash);
PG_FUNCTION_INFO_V1(pgstrom_fsum_final_fp32);
PG_FUNCTION_INFO_V1(pgstrom_fsum_final_fp64);
PG_FUNCTION_INFO_V1(pgstrom_fsum_final_fp64_as_numeric);
PG_FUNCTION_INFO_V1(pgstrom_fsum_final_numeric);
PG_FUNCTION_INFO_V1(pgstrom_favg_final_int);
PG_FUNCTION_INFO_V1(pgstrom_favg_final_fp);
PG_FUNCTION_INFO_V1(pgstrom_favg_final_num);
PG_FUNCTION_INFO_V1(pgstrom_favg_final_numeric);

PG_FUNCTION_INFO_V1(pgstrom_partial_variance);
PG_FUNCTION_INFO_V1(pgstrom_stddev_trans);
PG_FUNCTION_INFO_V1(pgstrom_stddev_samp_final);
PG_FUNCTION_INFO_V1(pgstrom_stddev_sampf_final);
PG_FUNCTION_INFO_V1(pgstrom_stddev_pop_final);
PG_FUNCTION_INFO_V1(pgstrom_stddev_popf_final);
PG_FUNCTION_INFO_V1(pgstrom_var_samp_final);
PG_FUNCTION_INFO_V1(pgstrom_var_sampf_final);
PG_FUNCTION_INFO_V1(pgstrom_var_pop_final);
PG_FUNCTION_INFO_V1(pgstrom_var_popf_final);

PG_FUNCTION_INFO_V1(pgstrom_partial_covar);
PG_FUNCTION_INFO_V1(pgstrom_covar_accum);
PG_FUNCTION_INFO_V1(pgstrom_correlation_final);
PG_FUNCTION_INFO_V1(pgstrom_covar_samp_final);
PG_FUNCTION_INFO_V1(pgstrom_covar_pop_final);

PG_FUNCTION_INFO_V1(pgstrom_regr_avgx_final);
PG_FUNCTION_INFO_V1(pgstrom_regr_avgy_final);
PG_FUNCTION_INFO_V1(pgstrom_regr_count_final);
PG_FUNCTION_INFO_V1(pgstrom_regr_intercept_final);
PG_FUNCTION_INFO_V1(pgstrom_regr_r2_final);
PG_FUNCTION_INFO_V1(pgstrom_regr_slope_final);
PG_FUNCTION_INFO_V1(pgstrom_regr_sxx_final);
PG_FUNCTION_INFO_V1(pgstrom_regr_sxy_final);
PG_FUNCTION_INFO_V1(pgstrom_regr_syy_final);

/*
 * float8 validator
 */
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
 * NROWS
 */
PUBLIC_FUNCTION(Datum)
pgstrom_partial_nrows(PG_FUNCTION_ARGS)
{
	int		nargs = PG_NARGS();;

	for (int i=0; i < nargs; i++)
	{
		if (PG_ARGISNULL(i))
			PG_RETURN_INT64(0);
	}
	PG_RETURN_INT64(1);
}

PUBLIC_FUNCTION(Datum)
pgstrom_partial_nrows_filtered(PG_FUNCTION_ARGS)
{
	int		nargs = PG_NARGS();

	Assert(nargs > 0);
	if (PG_ARGISNULL(nargs-1) || !PG_GETARG_BOOL(nargs-1))
		PG_RETURN_INT64(0);
	for (int i=1; i < nargs; i++)
	{
		if (PG_ARGISNULL(i-1))
			PG_RETURN_INT64(0);
	}
	PG_RETURN_INT64(1);
}

/*
 * MIN(X) and MAX(X) functions
 */
PUBLIC_FUNCTION(Datum)
pgstrom_partial_minmax_int64(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_int64_packed *r;

	if (PG_NARGS() > 1 && (PG_ARGISNULL(1) || !PG_GETARG_BOOL(1)))
		PG_RETURN_NULL();
	r = palloc(sizeof(kagg_state__pminmax_int64_packed));
	r->attrs = __PAGG_MINMAX_ATTRS__VALID;
	r->value = PG_GETARG_INT64(0);
	SET_VARSIZE(r, sizeof(kagg_state__pminmax_int64_packed));

	PG_RETURN_POINTER(r);
}

PUBLIC_FUNCTION(Datum)
pgstrom_partial_minmax_fp64(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_fp64_packed *r;

	if (PG_NARGS() > 1 && (PG_ARGISNULL(1) || !PG_GETARG_BOOL(1)))
		PG_RETURN_NULL();
	r = palloc(sizeof(kagg_state__pminmax_fp64_packed));
	r->attrs = __PAGG_MINMAX_ATTRS__VALID;
	r->value = PG_GETARG_FLOAT8(0);
	SET_VARSIZE(r, sizeof(kagg_state__pminmax_fp64_packed));

	PG_RETURN_POINTER(r);
}

#define __MINMAX_TRANS_TEMPLATE(TYPE,OPER)								\
	kagg_state__pminmax_##TYPE##_packed *state;							\
	kagg_state__pminmax_##TYPE##_packed *arg;							\
	MemoryContext	aggcxt;												\
																		\
	if (!AggCheckCallContext(fcinfo, &aggcxt))							\
		elog(ERROR, "aggregate function called in non-aggregate context"); \
	if (PG_ARGISNULL(0))												\
	{																	\
		if (PG_ARGISNULL(1))											\
			PG_RETURN_NULL();											\
		arg = (kagg_state__pminmax_##TYPE##_packed *)					\
			PG_GETARG_BYTEA_P(1);										\
		state = MemoryContextAlloc(aggcxt, sizeof(*state));				\
		memcpy(state, arg, sizeof(*state));								\
	}																	\
	else																\
	{																	\
		state = (kagg_state__pminmax_##TYPE##_packed *)					\
			PG_GETARG_BYTEA_P(0);										\
		if (!PG_ARGISNULL(1))											\
		{																\
			arg = (kagg_state__pminmax_##TYPE##_packed *)				\
				PG_GETARG_BYTEA_P(1);									\
			if ((arg->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)			\
			{															\
				if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)	\
					memcpy(state, arg, sizeof(*state));					\
				else													\
					state->value = OPER(state->value, arg->value);		\
			}															\
		}																\
	}																	\
	PG_RETURN_POINTER(state);

PUBLIC_FUNCTION(Datum)
pgstrom_fmin_trans_int64(PG_FUNCTION_ARGS)
{
	__MINMAX_TRANS_TEMPLATE(int64,Min);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fmin_trans_fp64(PG_FUNCTION_ARGS)
{
	__MINMAX_TRANS_TEMPLATE(fp64,Min);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fmax_trans_int64(PG_FUNCTION_ARGS)
{
	__MINMAX_TRANS_TEMPLATE(int64,Max);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fmax_trans_fp64(PG_FUNCTION_ARGS)
{
	__MINMAX_TRANS_TEMPLATE(fp64,Max);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fminmax_final_int8(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_int64_packed *state
		= (kagg_state__pminmax_int64_packed *)PG_GETARG_BYTEA_P(0);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)
		PG_RETURN_NULL();
	if (state->value < SCHAR_MIN || state->value > SCHAR_MAX)
		elog(ERROR, "min(int8) out of range");
	PG_RETURN_INT32(state->value);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fminmax_final_int16(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_int64_packed *state
		= (kagg_state__pminmax_int64_packed *)PG_GETARG_BYTEA_P(0);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)
		PG_RETURN_NULL();
	if (state->value < SHRT_MIN || state->value > SHRT_MAX)
		elog(ERROR, "min(int16) out of range");
	PG_RETURN_INT32(state->value);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fminmax_final_int32(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_int64_packed *state
		= (kagg_state__pminmax_int64_packed *)PG_GETARG_BYTEA_P(0);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)
		PG_RETURN_NULL();
	if (state->value < INT_MIN || state->value > INT_MAX)
		elog(ERROR, "min(int32) out of range");
	PG_RETURN_INT32(state->value);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fminmax_final_int64(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_int64_packed *state
		= (kagg_state__pminmax_int64_packed *)PG_GETARG_BYTEA_P(0);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)
		PG_RETURN_NULL();
	PG_RETURN_INT64(state->value);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fminmax_final_fp16(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_fp64_packed *state
		= (kagg_state__pminmax_fp64_packed *)PG_GETARG_BYTEA_P(0);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)
		PG_RETURN_NULL();
	PG_RETURN_UINT16(__half_as_short__(fp64_to_fp16(state->value)));
}

PUBLIC_FUNCTION(Datum)
pgstrom_fminmax_final_fp32(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_fp64_packed *state
		= (kagg_state__pminmax_fp64_packed *)PG_GETARG_BYTEA_P(0);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT4(state->value);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fminmax_final_fp64(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_fp64_packed *state
		= (kagg_state__pminmax_fp64_packed *)PG_GETARG_BYTEA_P(0);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(state->value);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fminmax_final_numeric(PG_FUNCTION_ARGS)
{
	kagg_state__pminmax_fp64_packed *state
		= (kagg_state__pminmax_fp64_packed *)PG_GETARG_BYTEA_P(0);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) == 0)
		PG_RETURN_NULL();
	return DirectFunctionCall1(float8_numeric,
							   Float8GetDatum(state->value));
}

/*
 * SUM(X),AVG(X) functions
 */
PUBLIC_FUNCTION(Datum)
pgstrom_partial_sum_int(PG_FUNCTION_ARGS)
{
	kagg_state__psum_int_packed *r;

	if (PG_NARGS() > 1 && (PG_ARGISNULL(1) || !PG_GETARG_BOOL(1)))
		PG_RETURN_NULL();
	r = palloc(sizeof(kagg_state__psum_int_packed));
	r->nitems = 1;
	r->sum = PG_GETARG_INT64(0);
	SET_VARSIZE(r, sizeof(kagg_state__psum_int_packed));

	PG_RETURN_POINTER(r);
}

PUBLIC_FUNCTION(Datum)
pgstrom_partial_sum_int64(PG_FUNCTION_ARGS)
{
	kagg_state__psum_numeric_packed *r;

	if (PG_NARGS() > 1 && (PG_ARGISNULL(1) || !PG_GETARG_BOOL(1)))
		PG_RETURN_NULL();
	r = palloc(sizeof(kagg_state__psum_numeric_packed));
	r->attrs = 0;
	r->nitems = 1;
	__store_int128_packed(&r->sum, PG_GETARG_INT64(0));
	SET_VARSIZE(r, sizeof(kagg_state__psum_numeric_packed));

	PG_RETURN_POINTER(r);
}

PUBLIC_FUNCTION(Datum)
pgstrom_partial_sum_fp(PG_FUNCTION_ARGS)
{
	kagg_state__psum_fp_packed *r;

	if (PG_NARGS() > 1 && (PG_ARGISNULL(1) || !PG_GETARG_BOOL(1)))
		PG_RETURN_NULL();
	r = palloc(sizeof(kagg_state__psum_fp_packed));
	r->nitems = 1;
	r->sum = PG_GETARG_FLOAT8(0);
	SET_VARSIZE(r, sizeof(kagg_state__psum_fp_packed));

	PG_RETURN_POINTER(r);
}

PUBLIC_FUNCTION(Datum)
pgstrom_partial_sum_numeric(PG_FUNCTION_ARGS)
{
	kagg_state__psum_numeric_packed *r;
	uint8_t		kind;
	int16_t		weight;
	int128_t	value;
	const char *emsg;

	if (PG_NARGS() > 1 && !PG_GETARG_BOOL(1))
		PG_RETURN_NULL();
	emsg = __xpu_numeric_from_varlena(&kind,
									  &weight,
									  &value,
									  (varlena *)PG_GETARG_NUMERIC(0));
	if (emsg)
		elog(ERROR, "%s: %s", __FUNCTION__, emsg);

	r = palloc0(sizeof(kagg_state__psum_numeric_packed));
	if (kind == XPU_NUMERIC_KIND__NAN)
		r->attrs |= __PAGG_NUMERIC_ATTRS__NAN;
	else if (kind == XPU_NUMERIC_KIND__POS_INF)
		r->attrs |= __PAGG_NUMERIC_ATTRS__PINF;
	else if (kind == XPU_NUMERIC_KIND__NEG_INF)
		r->attrs |= __PAGG_NUMERIC_ATTRS__NINF;
	else
	{
		Assert(kind == XPU_NUMERIC_KIND__VALID);
		r->attrs |= ((uint32_t)weight & __PAGG_NUMERIC_ATTRS__WEIGHT);
		__store_int128_packed(&r->sum, value);
	}
	r->nitems = 1;
	SET_VARSIZE(r, sizeof(kagg_state__psum_numeric_packed));
	PG_RETURN_POINTER(r);
}

PUBLIC_FUNCTION(Datum)
pgstrom_partial_sum_cash(PG_FUNCTION_ARGS)
{
	kagg_state__psum_int_packed *r;

	if (PG_NARGS() > 1 && (PG_ARGISNULL(1) && !PG_GETARG_BOOL(1)))
		PG_RETURN_NULL();
	r = palloc(sizeof(kagg_state__psum_int_packed));
	r->nitems = 1;
	r->sum = PG_GETARG_CASH(0);
	SET_VARSIZE(r, sizeof(kagg_state__psum_int_packed));

	PG_RETURN_POINTER(r);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_trans_int(PG_FUNCTION_ARGS)
{
	kagg_state__psum_int_packed *state;
	kagg_state__psum_int_packed *arg;
	MemoryContext	aggcxt;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	if (PG_ARGISNULL(0))
	{
		if (PG_ARGISNULL(1))
			PG_RETURN_NULL();
		arg = (kagg_state__psum_int_packed *)PG_GETARG_BYTEA_P(1);
		state = MemoryContextAlloc(aggcxt, sizeof(*state));
		memcpy(state, arg, sizeof(*state));
	}
	else
	{
		state = (kagg_state__psum_int_packed *)PG_GETARG_BYTEA_P(0);
		if (!PG_ARGISNULL(1))
		{
			arg = (kagg_state__psum_int_packed *)PG_GETARG_BYTEA_P(1);
			state->nitems += arg->nitems;
			state->sum    += arg->sum;
		}
	}
	PG_RETURN_POINTER(state);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_trans_fp(PG_FUNCTION_ARGS)
{
	kagg_state__psum_fp_packed *state;
	kagg_state__psum_fp_packed *arg;
	MemoryContext	aggcxt;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	if (PG_ARGISNULL(0))
	{
		if (PG_ARGISNULL(1))
			PG_RETURN_NULL();
		arg = (kagg_state__psum_fp_packed *)PG_GETARG_BYTEA_P(1);
		state = MemoryContextAlloc(aggcxt, sizeof(*state));
		memcpy(state, arg, sizeof(*state));
	}
	else
	{
		state = (kagg_state__psum_fp_packed *)PG_GETARG_BYTEA_P(0);
		if (!PG_ARGISNULL(1))
		{
			arg = (kagg_state__psum_fp_packed *)PG_GETARG_BYTEA_P(1);
			state->nitems += arg->nitems;
			state->sum    += arg->sum;
		}
	}
	PG_RETURN_POINTER(state);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_trans_numeric(PG_FUNCTION_ARGS)
{
	kagg_state__psum_numeric_packed *state;
	kagg_state__psum_numeric_packed *arg;
	MemoryContext		aggcxt;
	static uint64_t		__pow10[] = {
		1UL,						/* 10^0 */
		10UL,						/* 10^1 */
		100UL,						/* 10^2 */
		1000UL,						/* 10^3 */
		10000UL,					/* 10^4 */
		100000UL,					/* 10^5 */
		1000000UL,					/* 10^6 */
		10000000UL,					/* 10^7 */
		100000000UL,				/* 10^8 */
		1000000000UL,				/* 10^9 */
		10000000000UL,				/* 10^10 */
		100000000000UL,				/* 10^11 */
		1000000000000UL,			/* 10^12 */
		10000000000000UL,			/* 10^13 */
		100000000000000UL,			/* 10^14 */
		1000000000000000UL,			/* 10^15 */
		10000000000000000UL,		/* 10^16 */
		100000000000000000UL,		/* 10^17 */
		1000000000000000000UL,		/* 10^18 */
	};

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	if (PG_ARGISNULL(0))
	{
		if (PG_ARGISNULL(1))
			PG_RETURN_NULL();
		arg = (kagg_state__psum_numeric_packed *)PG_GETARG_BYTEA_P(1);
		state = MemoryContextAlloc(aggcxt, sizeof(*state));
		memcpy(state, arg, sizeof(*state));
	}
	else
	{
		uint32_t	special;

		state = (kagg_state__psum_numeric_packed *)PG_GETARG_BYTEA_P(0);
		if (!PG_ARGISNULL(1))
		{
			arg = (kagg_state__psum_numeric_packed *)PG_GETARG_BYTEA_P(1);
			special = (arg->attrs & __PAGG_NUMERIC_ATTRS__MASK);
			if (special != 0)
				state->attrs |= special;
			else
			{
				int16_t		weight_s = (int16_t)(state->attrs & __PAGG_NUMERIC_ATTRS__WEIGHT);
				int16_t		weight_a = (int16_t)(arg->attrs & __PAGG_NUMERIC_ATTRS__WEIGHT);
				int128_t	ival = __fetch_int128_packed(&arg->sum);

				if (weight_s > weight_a)
				{
					int		shift = (weight_s - weight_a);

					while (shift > 0)
					{
						int		k = Min(shift, 18);

						ival *= (int128_t)__pow10[k];
						shift -= k;
					}
				}
				else if (weight_s < weight_a)
				{
					int		shift = (weight_a - weight_s);

					while (shift > 0)
					{
						int		k = Min(shift, 18);

						ival /= (int128_t)__pow10[k];
						shift -= k;
					}
				}
				__store_int128_packed(&state->sum, ival +
									  __fetch_int128_packed(&state->sum));
			}
			state->nitems += arg->nitems;
		}
	}
	PG_RETURN_POINTER(state);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_final_int(PG_FUNCTION_ARGS)
{
	kagg_state__psum_int_packed *state;

	state = (kagg_state__psum_int_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	PG_RETURN_INT64(state->sum);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_final_int_as_numeric(PG_FUNCTION_ARGS)
{
	kagg_state__psum_int_packed *state;

	state = (kagg_state__psum_int_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	PG_RETURN_DATUM(DirectFunctionCall1(int8_numeric,
										Int64GetDatum(state->sum)));
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_final_int_as_cash(PG_FUNCTION_ARGS)
{
	kagg_state__psum_int_packed *state;

	state = (kagg_state__psum_int_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	PG_RETURN_CASH(state->sum);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_final_fp32(PG_FUNCTION_ARGS)
{
	kagg_state__psum_fp_packed *state
		= (kagg_state__psum_fp_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT4(state->sum);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_final_fp64(PG_FUNCTION_ARGS)
{
	kagg_state__psum_fp_packed *state
		= (kagg_state__psum_fp_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(state->sum);
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_final_fp64_as_numeric(PG_FUNCTION_ARGS)
{
	kagg_state__psum_fp_packed *state
		= (kagg_state__psum_fp_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	PG_RETURN_DATUM(DirectFunctionCall1(float8_numeric,
										Float8GetDatum(state->sum)));
}

PUBLIC_FUNCTION(Datum)
pgstrom_fsum_final_numeric(PG_FUNCTION_ARGS)
{
	kagg_state__psum_numeric_packed *state;
	uint32_t	special;
	Datum		datum;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();

	state = (kagg_state__psum_numeric_packed *)PG_GETARG_BYTEA_P(0);
	special = (state->attrs & __PAGG_NUMERIC_ATTRS__MASK);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	if (special != 0)
	{
		const char *str;

		if (special == __PAGG_NUMERIC_ATTRS__PINF)
			str = "+Inf";
		else if (special == __PAGG_NUMERIC_ATTRS__NINF)
			str = "-Inf";
		else
			str = "NaN";
		datum = DirectFunctionCall1(numeric_in, CStringGetDatum(str));
	}
	else
	{
		int16_t		weight = (state->attrs & __PAGG_NUMERIC_ATTRS__WEIGHT);
		int128_t	ival = __fetch_int128_packed(&state->sum);
		int			bufsz = __xpu_numeric_to_varlena(NULL, weight, ival);
		char	   *buf = palloc(bufsz);

		__xpu_numeric_to_varlena(buf, weight, ival);
		datum = PointerGetDatum(buf);
	}
	PG_RETURN_DATUM(datum);
}

PUBLIC_FUNCTION(Datum)
pgstrom_favg_final_int(PG_FUNCTION_ARGS)
{
	kagg_state__psum_int_packed *state;
	Datum	n, sum;

	state = (kagg_state__psum_int_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	n = DirectFunctionCall1(int8_numeric, Int64GetDatum(state->nitems));
	sum = DirectFunctionCall1(int8_numeric, Int64GetDatum(state->sum));

	PG_RETURN_DATUM(DirectFunctionCall2(numeric_div, sum, n));
}

PUBLIC_FUNCTION(Datum)
pgstrom_favg_final_fp(PG_FUNCTION_ARGS)
{
	kagg_state__psum_fp_packed *state;

	state = (kagg_state__psum_fp_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8((double)state->sum / (double)state->nitems);
}

PUBLIC_FUNCTION(Datum)
pgstrom_favg_final_num(PG_FUNCTION_ARGS)
{
	kagg_state__psum_fp_packed *state;
	Datum	n, sum;

	state = (kagg_state__psum_fp_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	n = DirectFunctionCall1(int8_numeric, Int64GetDatum(state->nitems));
	sum = DirectFunctionCall1(float8_numeric, Float8GetDatum(state->sum));

	PG_RETURN_DATUM(DirectFunctionCall2(numeric_div, sum, n));
}

PUBLIC_FUNCTION(Datum)
pgstrom_favg_final_numeric(PG_FUNCTION_ARGS)
{
	kagg_state__psum_numeric_packed *state;
	uint32_t	special;
	Datum		datum;

	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();

	state = (kagg_state__psum_numeric_packed *)PG_GETARG_BYTEA_P(0);
	special = (state->attrs & __PAGG_NUMERIC_ATTRS__MASK);
	if (state->nitems == 0)
		PG_RETURN_NULL();
	if (special != 0)
	{
		const char *str;

		if (special == __PAGG_NUMERIC_ATTRS__PINF)
			str = "+Inf";
		else if (special == __PAGG_NUMERIC_ATTRS__NINF)
			str = "-Inf";
		else
			str = "NaN";
		datum = DirectFunctionCall1(numeric_in, CStringGetDatum(str));
	}
	else
	{
		int16_t		weight = (state->attrs & __PAGG_NUMERIC_ATTRS__WEIGHT);
		int128_t	ival = __fetch_int128_packed(&state->sum);
		int			bufsz = __xpu_numeric_to_varlena(NULL, weight, ival);
		Numeric		sum = palloc(bufsz);
		Numeric		div = int64_to_numeric(state->nitems);

		__xpu_numeric_to_varlena((char *)sum, weight, ival);
		datum = DirectFunctionCall2(numeric_div,
									NumericGetDatum(sum),
									NumericGetDatum(div));
	}
	PG_RETURN_DATUM(datum);
}

/*
 * STDDEV/VARIANCE
 */
PUBLIC_FUNCTION(Datum)
pgstrom_partial_variance(PG_FUNCTION_ARGS)
{
	kagg_state__stddev_packed *r;
	float8_t	fval;

	if (PG_NARGS() > 1 && (PG_ARGISNULL(1) || !PG_GETARG_BOOL(1)))
		PG_RETURN_NULL();
	fval = PG_GETARG_FLOAT8(0);
	r = palloc(sizeof(kagg_state__stddev_packed));
	r->attrs = 0;
	r->nitems = 1;
	r->sum_x2 = fval * fval;
	SET_VARSIZE(r, sizeof(kagg_state__stddev_packed));

	PG_RETURN_POINTER(r);
}

PUBLIC_FUNCTION(Datum)
pgstrom_stddev_trans(PG_FUNCTION_ARGS)
{
	kagg_state__stddev_packed *state;
	kagg_state__stddev_packed *arg;
	MemoryContext	aggcxt;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	if (PG_ARGISNULL(0))
	{
		if (PG_ARGISNULL(1))
			PG_RETURN_NULL();
		arg = (kagg_state__stddev_packed *)PG_GETARG_BYTEA_P(1);
		state = MemoryContextAlloc(aggcxt, sizeof(*state));
		memcpy(state, arg, sizeof(*state));
	}
	else
	{
		state = (kagg_state__stddev_packed *)PG_GETARG_BYTEA_P(0);
		if (!PG_ARGISNULL(1))
		{
			arg = (kagg_state__stddev_packed *)PG_GETARG_BYTEA_P(1);

			state->attrs  |= arg->attrs;
			state->nitems += arg->nitems;
			state->sum_x  += arg->sum_x;
			state->sum_x2 += arg->sum_x2;
		}
	}
	PG_RETURN_POINTER(state);
}

PUBLIC_FUNCTION(Datum)
pgstrom_var_sampf_final(PG_FUNCTION_ARGS)
{
	if (!PG_ARGISNULL(0))
	{
		kagg_state__stddev_packed *state
			= (kagg_state__stddev_packed *)PG_GETARG_BYTEA_P(0);
		if (state->nitems > 1)
		{
			float8_t	N = (double)state->nitems;
			float8_t	fval = N * state->sum_x2 - state->sum_x * state->sum_x;

			PG_RETURN_FLOAT8(fval / (N * (N - 1.0)));
		}
	}
	PG_RETURN_NULL();
}

PUBLIC_FUNCTION(Datum)
pgstrom_var_samp_final(PG_FUNCTION_ARGS)
{
	Datum	datum = pgstrom_var_sampf_final(fcinfo);

	if (fcinfo->isnull)
		PG_RETURN_NULL();
	PG_RETURN_DATUM(DirectFunctionCall1(float8_numeric, datum));
}

PUBLIC_FUNCTION(Datum)
pgstrom_var_popf_final(PG_FUNCTION_ARGS)
{
	if (!PG_ARGISNULL(0))
	{
		kagg_state__stddev_packed *state
			= (kagg_state__stddev_packed *)PG_GETARG_BYTEA_P(0);
		if (state->nitems > 0)
		{
			float8_t	N = (double)state->nitems;
			float8_t	fval = N * state->sum_x2 - state->sum_x * state->sum_x;

			PG_RETURN_FLOAT8(fval / (N * N));
		}
	}
	PG_RETURN_NULL();
}

PUBLIC_FUNCTION(Datum)
pgstrom_var_pop_final(PG_FUNCTION_ARGS)
{
	Datum	datum = pgstrom_var_popf_final(fcinfo);

	if (fcinfo->isnull)
		PG_RETURN_NULL();
	PG_RETURN_DATUM(DirectFunctionCall1(float8_numeric, datum));
}

PUBLIC_FUNCTION(Datum)
pgstrom_stddev_sampf_final(PG_FUNCTION_ARGS)
{
	Datum   datum = pgstrom_var_sampf_final(fcinfo);

	if (fcinfo->isnull)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(sqrt(DatumGetFloat8(datum)));
}

PUBLIC_FUNCTION(Datum)
pgstrom_stddev_samp_final(PG_FUNCTION_ARGS)
{
	Datum	datum = pgstrom_stddev_sampf_final(fcinfo);

	if (fcinfo->isnull)
		PG_RETURN_NULL();
	PG_RETURN_DATUM(DirectFunctionCall1(float8_numeric, datum));
}

PUBLIC_FUNCTION(Datum)
pgstrom_stddev_popf_final(PG_FUNCTION_ARGS)
{
	Datum	datum = pgstrom_var_popf_final(fcinfo);

	if (fcinfo->isnull)
        PG_RETURN_NULL();
    PG_RETURN_FLOAT8(sqrt(DatumGetFloat8(datum)));
}

PUBLIC_FUNCTION(Datum)
pgstrom_stddev_pop_final(PG_FUNCTION_ARGS)
{
	Datum	datum = pgstrom_stddev_popf_final(fcinfo);

	if (fcinfo->isnull)
		PG_RETURN_NULL();
	PG_RETURN_DATUM(DirectFunctionCall1(float8_numeric, datum));
}

/*
 * COVAR/REGR_*
 */
PUBLIC_FUNCTION(Datum)
pgstrom_partial_covar(PG_FUNCTION_ARGS)
{
	kagg_state__covar_packed *r;
	float8_t	x = PG_GETARG_FLOAT8(0);
	float8_t	y = PG_GETARG_FLOAT8(1);

	if (PG_NARGS() > 2 && (PG_ARGISNULL(2) || !PG_GETARG_BOOL(2)))
		PG_RETURN_NULL();
	r = palloc(sizeof(kagg_state__covar_packed));
	r->attrs  = 0;
	r->nitems = 1;
	r->sum_x  = x;
	r->sum_xx = x * x;
	r->sum_y  = y;
	r->sum_yy = y * y;
	r->sum_xy = x * y;
	SET_VARSIZE(r, sizeof(kagg_state__covar_packed));

	PG_RETURN_POINTER(r);
}

PUBLIC_FUNCTION(Datum)
pgstrom_covar_accum(PG_FUNCTION_ARGS)
{
	kagg_state__covar_packed *state;
	kagg_state__covar_packed *arg;
	MemoryContext	aggcxt;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	if (PG_ARGISNULL(0))
	{
		if (PG_ARGISNULL(1))
			PG_RETURN_NULL();
		arg = (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(1);
		state = MemoryContextAlloc(aggcxt, sizeof(*state));
		memcpy(state, arg, sizeof(*state));
	}
	else
	{
		state = (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(0);
		if (!PG_ARGISNULL(1))
		{
			arg = (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(1);

			state->attrs  |= arg->attrs;
			state->nitems += arg->nitems;
			state->sum_x  += arg->sum_x;
			state->sum_xx += arg->sum_xx;
			state->sum_y  += arg->sum_y;
			state->sum_yy += arg->sum_yy;
			state->sum_xy += arg->sum_xy;
		}
	}
	PG_RETURN_POINTER(state);
}

PUBLIC_FUNCTION(Datum)
pgstrom_correlation_final(PG_FUNCTION_ARGS)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(0);

	if (state->nitems < 1 ||
		state->sum_xx == 0.0 ||
		state->sum_yy == 0.0)
		PG_RETURN_NULL();

	PG_RETURN_FLOAT8(state->sum_xy / sqrt(state->sum_xx * state->sum_yy));
}

PUBLIC_FUNCTION(Datum)
pgstrom_covar_samp_final(PG_FUNCTION_ARGS)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(0);

	if (state->nitems < 2)
		PG_RETURN_NULL();

	PG_RETURN_FLOAT8(state->sum_xy / (double)(state->nitems - 1));
}

PUBLIC_FUNCTION(Datum)
pgstrom_covar_pop_final(PG_FUNCTION_ARGS)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(0);

	if (state->nitems < 1)
		PG_RETURN_NULL();

	PG_RETURN_FLOAT8(state->sum_xy / (double)state->nitems);
}

PUBLIC_FUNCTION(Datum)
pgstrom_regr_avgx_final(PG_FUNCTION_ARGS)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(0);

	if (state->nitems < 1)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(state->sum_x / (double)state->nitems);
}

PUBLIC_FUNCTION(Datum)
pgstrom_regr_avgy_final(PG_FUNCTION_ARGS)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems < 1)
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(state->sum_y / (double)state->nitems);
}

PUBLIC_FUNCTION(Datum)
pgstrom_regr_count_final(PG_FUNCTION_ARGS)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(0);

	PG_RETURN_FLOAT8((double)state->nitems);
}

PUBLIC_FUNCTION(Datum)
pgstrom_regr_intercept_final(PG_FUNCTION_ARGS)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(0);

	if (state->nitems < 1 || state->sum_xx == 0.0)
		PG_RETURN_NULL();

	PG_RETURN_FLOAT8((state->sum_y -
					  state->sum_x * state->sum_xy / state->sum_xx) / (double)state->nitems);
}

PUBLIC_FUNCTION(Datum)
pgstrom_regr_r2_final(PG_FUNCTION_ARGS)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems > 0 &&
		state->sum_xx != 0.0 &&
		state->sum_yy != 0.0)
	{
		PG_RETURN_FLOAT8((state->sum_xy * state->sum_xy) /
						 (state->sum_xx * state->sum_yy));
	}
	PG_RETURN_NULL();
}

PUBLIC_FUNCTION(Datum)
pgstrom_regr_slope_final(PG_FUNCTION_ARGS)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems > 0 && state->sum_xx != 0.0)
	{
		PG_RETURN_FLOAT8(state->sum_xy / state->sum_xx);
	}
	PG_RETURN_NULL();
}

PUBLIC_FUNCTION(Datum)
pgstrom_regr_sxx_final(PG_FUNCTION_ARGS)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems > 0)
	{
		PG_RETURN_FLOAT8(state->sum_xx);
	}
	PG_RETURN_NULL();
}

PUBLIC_FUNCTION(Datum)
pgstrom_regr_sxy_final(PG_FUNCTION_ARGS)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems > 0)
	{
		PG_RETURN_FLOAT8(state->sum_xy);
	}
	PG_RETURN_NULL();
}

PUBLIC_FUNCTION(Datum)
pgstrom_regr_syy_final(PG_FUNCTION_ARGS)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)PG_GETARG_BYTEA_P(0);
	if (state->nitems > 0)
	{
		PG_RETURN_FLOAT8(state->sum_yy);
	}
	PG_RETURN_NULL();
}

#if 0
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
	xpu_numeric_t	num;
	const char	   *emsg;

	memset(&num, 0, sizeof(num));
	emsg = __xpu_numeric_from_varlena(&num, (struct varlena *)datum);
	if (emsg)
		elog(ERROR, "failed on hash calculation of device numeric: %s", emsg);
	return __pgstrom_hll_siphash_value(&num.weight,
									   offsetof(xpu_numeric_t, value)
									   + sizeof(int128_t)
									   - offsetof(xpu_numeric_t, weight));
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
	PUBLIC_FUNCTION(Datum)												\
	pgstrom_hll_hash_##NAME(PG_FUNCTION_ARGS)							\
	{																	\
		Datum	arg = PG_GETARG_DATUM(0);								\
		PG_RETURN_UINT64(__pgstrom_hll_hash_##NAME(arg));				\
	}																	\
	PUBLIC_FUNCTION(Datum)												\
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
PUBLIC_FUNCTION(Datum)
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
PUBLIC_FUNCTION(Datum)
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
PUBLIC_FUNCTION(Datum)
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
PUBLIC_FUNCTION(Datum)
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
#endif
