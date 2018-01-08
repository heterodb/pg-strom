/*
 * float2.c
 *
 * half-precision floating point data type support
 * ----
 * Copyright 2018 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2018 (C) The PG-Strom Development Team
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

typedef unsigned short	half_t;
#define PG_GETARG_FLOAT2(x)	PG_GETARG_INT16(x)
#define PG_RETURN_FLOAT2(x)	PG_RETURN_INT16(x)
#define DatumGetFloat2(x)	DatumGetInt16(x)
#define Float2GetDatum(x)	Int16GetDatum(x)

#define FP16_FRAC_BITS		(10)
#define FP16_EXPO_BITS		(5)
#define FP16_EXPO_MIN		(-14)
#define FP16_EXPO_MAX		(15)
#define FP16_EXPO_BIAS		(15)

#define FP32_FRAC_BITS		(23)
#define FP32_EXPO_BITS		(8)
#define FP32_EXPO_MIN		(-126)
#define FP32_EXPO_MAX		(127)
#define FP32_EXPO_BIAS		(127)

#define FP64_FRAC_BITS		(52)
#define FP64_EXPO_BITS		(11)
#define FP64_EXPO_MIN		(-1022)
#define FP64_EXPO_MAX		(1023)
#define FP64_EXPO_BIAS		(1023)

/* type input/output */
Datum pgstrom_float2_in(PG_FUNCTION_ARGS);
Datum pgstrom_float2_out(PG_FUNCTION_ARGS);
Datum pgstrom_float2_send(PG_FUNCTION_ARGS);
Datum pgstrom_float2_recv(PG_FUNCTION_ARGS);
/* type cast functions */
Datum pgstrom_float2_to_float4(PG_FUNCTION_ARGS);
Datum pgstrom_float2_to_float8(PG_FUNCTION_ARGS);
Datum pgstrom_float2_to_int2(PG_FUNCTION_ARGS);
Datum pgstrom_float2_to_int4(PG_FUNCTION_ARGS);
Datum pgstrom_float2_to_int8(PG_FUNCTION_ARGS);
Datum pgstrom_float2_to_numeric(PG_FUNCTION_ARGS);
Datum pgstrom_float4_to_float2(PG_FUNCTION_ARGS);
Datum pgstrom_float8_to_float2(PG_FUNCTION_ARGS);
Datum pgstrom_int2_to_float2(PG_FUNCTION_ARGS);
Datum pgstrom_int4_to_float2(PG_FUNCTION_ARGS);
Datum pgstrom_int8_to_float2(PG_FUNCTION_ARGS);
Datum pgstrom_numeric_to_float2(PG_FUNCTION_ARGS);
/* comparison */
Datum pgstrom_float2_eq(PG_FUNCTION_ARGS);
Datum pgstrom_float2_ne(PG_FUNCTION_ARGS);
Datum pgstrom_float2_lt(PG_FUNCTION_ARGS);
Datum pgstrom_float2_le(PG_FUNCTION_ARGS);
Datum pgstrom_float2_gt(PG_FUNCTION_ARGS);
Datum pgstrom_float2_ge(PG_FUNCTION_ARGS);
Datum pgstrom_float2_larger(PG_FUNCTION_ARGS);
Datum pgstrom_float2_smaller(PG_FUNCTION_ARGS);

Datum pgstrom_float42_eq(PG_FUNCTION_ARGS);
Datum pgstrom_float42_ne(PG_FUNCTION_ARGS);
Datum pgstrom_float42_lt(PG_FUNCTION_ARGS);
Datum pgstrom_float42_le(PG_FUNCTION_ARGS);
Datum pgstrom_float42_gt(PG_FUNCTION_ARGS);
Datum pgstrom_float42_ge(PG_FUNCTION_ARGS);

Datum pgstrom_float82_eq(PG_FUNCTION_ARGS);
Datum pgstrom_float82_ne(PG_FUNCTION_ARGS);
Datum pgstrom_float82_lt(PG_FUNCTION_ARGS);
Datum pgstrom_float82_le(PG_FUNCTION_ARGS);
Datum pgstrom_float82_gt(PG_FUNCTION_ARGS);
Datum pgstrom_float82_ge(PG_FUNCTION_ARGS);

Datum pgstrom_float24_eq(PG_FUNCTION_ARGS);
Datum pgstrom_float24_ne(PG_FUNCTION_ARGS);
Datum pgstrom_float24_lt(PG_FUNCTION_ARGS);
Datum pgstrom_float24_le(PG_FUNCTION_ARGS);
Datum pgstrom_float24_gt(PG_FUNCTION_ARGS);
Datum pgstrom_float24_ge(PG_FUNCTION_ARGS);

Datum pgstrom_float28_eq(PG_FUNCTION_ARGS);
Datum pgstrom_float28_ne(PG_FUNCTION_ARGS);
Datum pgstrom_float28_lt(PG_FUNCTION_ARGS);
Datum pgstrom_float28_le(PG_FUNCTION_ARGS);
Datum pgstrom_float28_gt(PG_FUNCTION_ARGS);
Datum pgstrom_float28_ge(PG_FUNCTION_ARGS);

/* unary operators */
Datum pgstrom_float2_up(PG_FUNCTION_ARGS);
Datum pgstrom_float2_um(PG_FUNCTION_ARGS);
Datum pgstrom_float2_abs(PG_FUNCTION_ARGS);

/* arithmetic operators */
Datum pgstrom_float2_pl(PG_FUNCTION_ARGS);
Datum pgstrom_float2_mi(PG_FUNCTION_ARGS);
Datum pgstrom_float2_mul(PG_FUNCTION_ARGS);
Datum pgstrom_float2_div(PG_FUNCTION_ARGS);

Datum pgstrom_float24_pl(PG_FUNCTION_ARGS);
Datum pgstrom_float24_mi(PG_FUNCTION_ARGS);
Datum pgstrom_float24_mul(PG_FUNCTION_ARGS);
Datum pgstrom_float24_div(PG_FUNCTION_ARGS);

Datum pgstrom_float28_pl(PG_FUNCTION_ARGS);
Datum pgstrom_float28_mi(PG_FUNCTION_ARGS);
Datum pgstrom_float28_mul(PG_FUNCTION_ARGS);
Datum pgstrom_float28_div(PG_FUNCTION_ARGS);

Datum pgstrom_float42_pl(PG_FUNCTION_ARGS);
Datum pgstrom_float42_mi(PG_FUNCTION_ARGS);
Datum pgstrom_float42_mul(PG_FUNCTION_ARGS);
Datum pgstrom_float42_div(PG_FUNCTION_ARGS);

Datum pgstrom_float82_pl(PG_FUNCTION_ARGS);
Datum pgstrom_float82_mi(PG_FUNCTION_ARGS);
Datum pgstrom_float82_mul(PG_FUNCTION_ARGS);
Datum pgstrom_float82_div(PG_FUNCTION_ARGS);

/* misc functions */
Datum pgstrom_cash_mul_float2(PG_FUNCTION_ARGS);
Datum pgstrom_float2_mul_cash(PG_FUNCTION_ARGS);
Datum pgstrom_cash_div_float2(PG_FUNCTION_ARGS);
Datum pgstrom_float8_as_int8(PG_FUNCTION_ARGS);
Datum pgstrom_float4_as_int4(PG_FUNCTION_ARGS);
Datum pgstrom_float2_as_int2(PG_FUNCTION_ARGS);
Datum pgstrom_int8_as_float8(PG_FUNCTION_ARGS);
Datum pgstrom_int4_as_float4(PG_FUNCTION_ARGS);
Datum pgstrom_int2_as_float2(PG_FUNCTION_ARGS);

/* aggregate functions */
Datum pgstrom_float2_accum(PG_FUNCTION_ARGS);
Datum pgstrom_float2_sum(PG_FUNCTION_ARGS);

//#define DEBUG_FP16 1

static inline void
print_fp16(const char *prefix, cl_uint value)
{
#ifdef DEBUG_FP16
	elog(INFO, "%sFP16 0x%04x = %d + %d + 0x%04x",
		 prefix ? prefix : "",
		 value,
		 (value & 0x8000) ? 1 : 0,
		 ((value >> FP16_FRAC_BITS) & 0x001f) - FP16_EXPO_BIAS,
		 (value & 0x03ff));
#endif
}

static inline void
print_fp32(const char *prefix, cl_uint value)
{
#ifdef DEBUG_FP16
	elog(INFO, "%sFP32 0x%08x = %d + %d + 0x%08x",
		 prefix ? prefix : "",
         value,
		 (value & 0x80000000U) ? 1 : 0,
		 ((value >> FP32_FRAC_BITS) & 0x00ff) - FP32_EXPO_BIAS,
		 (value & 0x7fffff));
#endif
}

static inline void
print_fp64(const char *prefix, cl_ulong value)
{
#ifdef DEBUG_FP16
	elog(INFO, "%sFP64 0x%016lx = %d + %ld + %014lx",
		 prefix ? prefix : "",
         value,
		 (value & 0x8000000000000000UL) ? 1 : 0,
		 ((value >> FP64_FRAC_BITS) & 0x07ff) - FP64_EXPO_BIAS,
		 (value & ((1UL << FP64_FRAC_BITS) - 1)));
#endif
}

/*
 * check to see if a float4/8 val has underflowed or overflowed
 */
#define CHECKFLOATVAL(val, inf_is_valid, zero_is_valid)				\
	do {															\
		if (isinf(val) && !(inf_is_valid))							\
			ereport(ERROR,											\
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),	\
					 errmsg("value out of range: overflow")));		\
																	\
		if ((val) == 0.0 && !(zero_is_valid))						\
			ereport(ERROR,											\
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),	\
					 errmsg("value out of range: underflow")));		\
	} while(0)

/*
 * cast functions across floating point
 */
static half_t
fp32_to_fp16(float value)
{
	cl_uint		fp32val = float_as_int(value);
	cl_uint		sign = ((fp32val & 0x80000000U) >> 16);
	cl_int		expo = ((fp32val & 0x7f800000U) >> 23);
	cl_int		frac = ((fp32val & 0x007fffffU));
	half_t		result;

	print_fp32("->", fp32val);

	/* special cases */
	if (expo == 0xff)
	{
		if (frac == 0)
			result = sign | 0x7c00;		/* -/+Infinity */
		else
			result = 0xffff;			/* NaN */
	}
	else if (expo == 0)
		result = sign;					/* -/+0.0 */
	else
	{
		expo -= FP32_EXPO_BIAS;

		frac = ((frac >> 13) | 0x400) + ((frac >> 12) & 1);
		while ((frac & 0xfc00) != 0x400)
		{
			frac >>= 1;
			expo++;
		}

		if (expo < FP16_EXPO_MIN)
		{
			/* try non-uniformed fraction for small numbers */
			if (FP16_EXPO_MIN - expo <= FP16_FRAC_BITS)
				frac >>= (FP16_EXPO_MIN - expo);
			else
				frac = 0;
			expo = 0;
		}
		else if (expo > FP16_EXPO_MAX)
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("\"%f\" is out of range for type float2", value)));
		else
		{
			frac &= 0x3ff;
			expo += FP16_EXPO_BIAS;
		}
		result = sign | (expo << FP16_FRAC_BITS) | frac;
	}
	print_fp16("<-", result);
	return result;
}

static float
fp16_to_fp32(half_t fp16val)
{
	cl_uint		sign = ((cl_uint)(fp16val & 0x8000) << 16);
	cl_int		expo = ((fp16val & 0x7c00) >> 10);
	cl_int		frac = ((fp16val & 0x03ff));
	cl_uint		result;

	print_fp16("->", fp16val);

	if (expo == 0x1f)
	{
		if (frac == 0)
			result = (sign | 0x7f800000);	/* +/-Infinity */
		else
			result = 0xffffffff;			/* NaN */
	}
	else if (expo == 0 && frac == 0)
		result = sign;						/* +/-0.0 */
	else
	{
		if (expo == 0)
		{
			expo = FP16_EXPO_MIN;
			while ((frac & 0x400) == 0)
			{
				frac <<= 1;
				expo--;
			}
			frac &= 0x3ff;
		}
		else
			expo -= FP16_EXPO_BIAS;

		expo += FP32_EXPO_BIAS;
		result = (sign | (expo << FP32_FRAC_BITS) | (frac << 13));
	}
	print_fp32("<-", result);
	return int_as_float(result);
}

static half_t
fp64_to_fp16(double value)
{
	cl_ulong	fp64val = double_as_long(value);
	cl_uint		sign = ((fp64val >> 48) & 0x8000);
	cl_long		expo = ((fp64val >> 52) & 0x07ff);
	cl_long		frac = ((fp64val & 0x000fffffffffffffUL));
	half_t		result;

	print_fp64("->", fp64val);

	if (expo == 0x7ff)
	{
		if (frac == 0)
			result = sign | 0x7c00;		/* -/+Infinity */
		else
			result = 0xffff;			/* NaN */
	}
	else if (expo == 0)
		result = sign;					/* -/+0.0 */
	else
	{
		expo -= FP64_EXPO_BIAS;

		frac = ((frac >> 42) | 0x400) + ((frac >> 41) & 1);
		while ((frac & 0xfc00) != 0x400)
		{
			frac >>= 1;
			expo++;
		}

		if (expo < FP16_EXPO_MIN)
		{
			/* try non-uniformed fraction for small numbers */
			if (FP16_EXPO_MIN - expo <= FP16_FRAC_BITS)
				frac >>= (FP16_EXPO_MIN - expo);
			else
				frac = 0;
			expo = 0;
		}
		else if (expo > FP16_EXPO_MAX)
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("\"%f\" is out of range for type float2", value)));
		else
		{
			frac &= 0x3ff;
			expo += FP16_EXPO_BIAS;
		}
		result = sign | (expo << FP16_FRAC_BITS) | frac;
	}
	print_fp16("<-", result);
	return result;
}

static float
fp16_to_fp64(half_t fp16val)
{
	cl_uint		sign = ((cl_uint)(fp16val & 0x8000) << 16);
	cl_long		expo = ((fp16val & 0x7c00) >> 10);
	cl_long		frac = ((fp16val & 0x03ff));
	cl_ulong	result;

	print_fp16("->",fp16val);
	if (expo == 0x1f)
	{
		if (frac == 0)
			result = (sign | 0x7f800000);	/* +/-Infinity */
		else
			result = 0xffffffff;			/* NaN */
	}
	else if (expo == 0 && frac == 0)
		result = sign;						/* +/-0.0 */
	else
	{
		if (expo == 0)
		{
			expo = FP16_EXPO_MIN;
			while ((frac & 0x400) == 0)
			{
				frac <<= 1;
				expo--;
			}
			frac &= 0x3ff;
		}
		else
			expo -= FP16_EXPO_BIAS;

		expo += FP64_EXPO_BIAS;
		result = (sign | (expo << FP64_FRAC_BITS) | (frac << 42));
	}
	print_fp64("<-", result);
	return long_as_double(result);
}

/*
 * pgstrom_float2_in
 */
Datum
pgstrom_float2_in(PG_FUNCTION_ARGS)
{
	Datum	datum = float4in(fcinfo);
	float	fval;

	if (fcinfo->isnull)
		PG_RETURN_NULL();
	fval = DatumGetFloat4(datum);

	PG_RETURN_FLOAT2(fp32_to_fp16(fval));
}
PG_FUNCTION_INFO_V1(pgstrom_float2_in);

/*
 * pgstrom_float2_out
 */
Datum
pgstrom_float2_out(PG_FUNCTION_ARGS)
{
	float	fval = fp16_to_fp32((half_t)PG_GETARG_FLOAT2(0));

	return DirectFunctionCall1(float4out, Float4GetDatum(fval));
}
PG_FUNCTION_INFO_V1(pgstrom_float2_out);

/*
 * pgstrom_float2_recv
 */
Datum
pgstrom_float2_recv(PG_FUNCTION_ARGS)
{
	return int2recv(fcinfo);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_recv);

/*
 * pgstrom_float2_send
 */
Datum
pgstrom_float2_send(PG_FUNCTION_ARGS)
{
	return int2send(fcinfo);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_send);

/*
 * pgstrom_float2_to_float4
 */
Datum
pgstrom_float2_to_float4(PG_FUNCTION_ARGS)
{
	half_t	fval = PG_GETARG_FLOAT2(0);

	PG_RETURN_FLOAT4(fp16_to_fp32(fval));
}
PG_FUNCTION_INFO_V1(pgstrom_float2_to_float4);

/*
 * pgstrom_float2_to_float8
 */
Datum
pgstrom_float2_to_float8(PG_FUNCTION_ARGS)
{
	half_t	fval = PG_GETARG_FLOAT2(0);

	PG_RETURN_FLOAT8(fp16_to_fp64(fval));
}
PG_FUNCTION_INFO_V1(pgstrom_float2_to_float8);

/*
 * pgstrom_float2_to_int2
 */
Datum
pgstrom_float2_to_int2(PG_FUNCTION_ARGS)
{
	float	fval = fp16_to_fp32(PG_GETARG_FLOAT2(0));

	return DirectFunctionCall1(ftoi2, Float2GetDatum(fval));
}
PG_FUNCTION_INFO_V1(pgstrom_float2_to_int2);

/*
 * pgstrom_float2_to_int4
 */
Datum
pgstrom_float2_to_int4(PG_FUNCTION_ARGS)
{
	float	fval = fp16_to_fp32(PG_GETARG_FLOAT2(0));

	return DirectFunctionCall1(ftoi4, Float2GetDatum(fval));
}
PG_FUNCTION_INFO_V1(pgstrom_float2_to_int4);

/*
 * pgstrom_float2_to_int8
 */
Datum
pgstrom_float2_to_int8(PG_FUNCTION_ARGS)
{
	double	fval = fp16_to_fp64(PG_GETARG_FLOAT2(0));

	return DirectFunctionCall1(dtoi8, Float2GetDatum(fval));
}
PG_FUNCTION_INFO_V1(pgstrom_float2_to_int8);

/*
 * pgstrom_float2_to_numeric
 */
Datum
pgstrom_float2_to_numeric(PG_FUNCTION_ARGS)
{
	float	fval = PG_GETARG_FLOAT2(0);

	return DirectFunctionCall1(float4_numeric, Float4GetDatum(fval));
}
PG_FUNCTION_INFO_V1(pgstrom_float2_to_numeric);

/*
 * pgstrom_float4_to_float2
 */
Datum
pgstrom_float4_to_float2(PG_FUNCTION_ARGS)
{
	float	fval = PG_GETARG_FLOAT4(0);

	PG_RETURN_FLOAT2(fp32_to_fp16(fval));
}
PG_FUNCTION_INFO_V1(pgstrom_float4_to_float2);

/*
 * pgstrom_float8_to_float2
 */
Datum
pgstrom_float8_to_float2(PG_FUNCTION_ARGS)
{
	double	fval = PG_GETARG_FLOAT8(0);

	PG_RETURN_FLOAT2(fp64_to_fp16(fval));
}
PG_FUNCTION_INFO_V1(pgstrom_float8_to_float2);

/*
 * pgstrom_int2_to_float2
 */
Datum
pgstrom_int2_to_float2(PG_FUNCTION_ARGS)
{
	float	fval = (float) PG_GETARG_INT16(0);

	PG_RETURN_FLOAT2(fp32_to_fp16(fval));
}
PG_FUNCTION_INFO_V1(pgstrom_int2_to_float2);

/*
 * pgstrom_int4_to_float2
 */
Datum
pgstrom_int4_to_float2(PG_FUNCTION_ARGS)
{
	float	fval = (float) PG_GETARG_INT32(0);

	PG_RETURN_FLOAT2(fp32_to_fp16(fval));
}
PG_FUNCTION_INFO_V1(pgstrom_int4_to_float2);

/*
 * pgstrom_int8_to_float2
 */
Datum
pgstrom_int8_to_float2(PG_FUNCTION_ARGS)
{
	double	fval = (double) PG_GETARG_INT64(0);

	PG_RETURN_FLOAT2(fp64_to_fp16(fval));
}
PG_FUNCTION_INFO_V1(pgstrom_int8_to_float2);

/*
 * pgstrom_numeric_to_float2
 */
Datum
pgstrom_numeric_to_float2(PG_FUNCTION_ARGS)
{
	float	fval = DatumGetFloat4(numeric_float4(fcinfo));

	PG_RETURN_FLOAT2(fp32_to_fp16(fval));
}
PG_FUNCTION_INFO_V1(pgstrom_numeric_to_float2);

/*
 * Comparison operators
 */
Datum
pgstrom_float2_eq(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) == 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_eq);

Datum
pgstrom_float2_ne(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) != 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_ne);

Datum
pgstrom_float2_lt(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) < 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_lt);

Datum
pgstrom_float2_le(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) <= 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_le);

Datum
pgstrom_float2_gt(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) > 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_gt);

Datum
pgstrom_float2_ge(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) >= 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_ge);

Datum
pgstrom_float2_larger(PG_FUNCTION_ARGS)
{
	half_t	arg1 = PG_GETARG_FLOAT2(0);
	half_t	arg2 = PG_GETARG_FLOAT2(1);

	PG_RETURN_FLOAT2(fp16_to_fp32(arg1) > fp16_to_fp32(arg2) ? arg1 : arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_larger);

Datum
pgstrom_float2_smaller(PG_FUNCTION_ARGS)
{
	half_t	arg1 = PG_GETARG_FLOAT2(0);
	half_t	arg2 = PG_GETARG_FLOAT2(1);

	PG_RETURN_FLOAT2(fp16_to_fp32(arg1) < fp16_to_fp32(arg2) ? arg1 : arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_smaller);

Datum
pgstrom_float42_eq(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) == 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float42_eq);

Datum
pgstrom_float42_ne(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) != 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float42_ne);

Datum
pgstrom_float42_lt(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) < 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float42_lt);

Datum
pgstrom_float42_le(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) <= 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float42_le);

Datum
pgstrom_float42_gt(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) > 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float42_gt);

Datum
pgstrom_float42_ge(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) >= 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float42_ge);

Datum
pgstrom_float82_eq(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = fp16_to_fp64(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) == 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float82_eq);

Datum
pgstrom_float82_ne(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = fp16_to_fp64(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) != 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float82_ne);

Datum
pgstrom_float82_lt(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = fp16_to_fp64(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) < 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float82_lt);

Datum
pgstrom_float82_le(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = fp16_to_fp64(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) <= 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float82_le);

Datum
pgstrom_float82_gt(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = fp16_to_fp64(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) > 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float82_gt);

Datum
pgstrom_float82_ge(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = fp16_to_fp64(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) >= 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float82_ge);

Datum
pgstrom_float24_eq(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) == 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float24_eq);

Datum
pgstrom_float24_ne(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) != 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float24_ne);

Datum
pgstrom_float24_lt(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) < 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float24_lt);

Datum
pgstrom_float24_le(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) <= 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float24_le);

Datum
pgstrom_float24_gt(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) > 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float24_gt);

Datum
pgstrom_float24_ge(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) >= 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float24_ge);

Datum
pgstrom_float28_eq(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) == 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float28_eq);

Datum
pgstrom_float28_ne(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) != 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float28_ne);

Datum
pgstrom_float28_lt(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) < 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float28_lt);

Datum
pgstrom_float28_le(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) <= 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float28_le);

Datum
pgstrom_float28_gt(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) > 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float28_gt);

Datum
pgstrom_float28_ge(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) >= 0);
}
PG_FUNCTION_INFO_V1(pgstrom_float28_ge);

/*
 * unary operators
 */
Datum
pgstrom_float2_up(PG_FUNCTION_ARGS)
{
	half_t	fval = PG_GETARG_FLOAT2(0);

	PG_RETURN_FLOAT2(fval);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_up);

Datum
pgstrom_float2_um(PG_FUNCTION_ARGS)
{
	half_t	fval = PG_GETARG_FLOAT2(0);

	fval ^= 0x8000;

	PG_RETURN_FLOAT2(fval);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_um);

Datum
pgstrom_float2_abs(PG_FUNCTION_ARGS)
{
	half_t	fval = PG_GETARG_FLOAT2(0);

	fval &= ~0x8000;

	PG_RETURN_FLOAT2(fval);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_abs);

/*
 * arithmetic operations
 */
Datum
pgstrom_float2_pl(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));
	float	result;

	result = arg1 + arg2;

	elog(INFO, "arg1 %f arg2 %f result %f", arg1, arg2, result);

	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT4(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_pl);

Datum
pgstrom_float2_mi(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));
	float	result;

	result = arg1 - arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT4(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_mi);

Datum
pgstrom_float2_mul(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));
	float	result;

	result = arg1 * arg2;

	CHECKFLOATVAL(result,
				  isinf(arg1) || isinf(arg2),
				  arg1 == 0 || arg2 == 0);
	PG_RETURN_FLOAT4(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_mul);

Datum
pgstrom_float2_div(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));
	float	result;

	if (arg2 == 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));
	result = arg1 / arg2;

	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), arg1 == 0.0);
	PG_RETURN_FLOAT4(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_div);

Datum
pgstrom_float24_pl(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = PG_GETARG_FLOAT2(1);
	float	result;

	result = arg1 + arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT4(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float24_pl);

Datum
pgstrom_float24_mi(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = PG_GETARG_FLOAT2(1);
	float	result;

	result = arg1 - arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT4(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float24_mi);

Datum
pgstrom_float24_mul(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = PG_GETARG_FLOAT2(1);
	float	result;

	result = arg1 * arg2;
	CHECKFLOATVAL(result,
				  isinf(arg1) || isinf(arg2),
				  arg1 == 0 || arg2 == 0);
	PG_RETURN_FLOAT4(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float24_mul);

Datum
pgstrom_float24_div(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = PG_GETARG_FLOAT2(1);
	float	result;

	if (arg2 == 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));

	result = arg1 / arg2;

	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), arg1 == 0.0);
	PG_RETURN_FLOAT4(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float24_div);

Datum
pgstrom_float28_pl(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
	double	arg2 = PG_GETARG_FLOAT8(1);
	double	result;

	result = arg1 + arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);

	PG_RETURN_FLOAT8(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float28_pl);

Datum
pgstrom_float28_mi(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
	double	arg2 = PG_GETARG_FLOAT8(1);
	double	result;

	result = arg1 - arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);

	PG_RETURN_FLOAT8(result);	
}
PG_FUNCTION_INFO_V1(pgstrom_float28_mi);

Datum
pgstrom_float28_mul(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
	double	arg2 = PG_GETARG_FLOAT8(1);
	double	result;

	result = arg1 * arg2;
	CHECKFLOATVAL(result,
				  isinf(arg1) || isinf(arg2),
				  arg1 == 0.0 || arg2 == 0.0);
	PG_RETURN_FLOAT8(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float28_mul);

Datum
pgstrom_float28_div(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
	double	arg2 = PG_GETARG_FLOAT8(1);
	double	result;

	if (arg2 == 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));

	result = arg1 / arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), arg1 == 0.0);

	PG_RETURN_FLOAT8(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float28_div);

Datum
pgstrom_float42_pl(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));
	float	result;

	result = arg1 + arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT4(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float42_pl);

Datum
pgstrom_float42_mi(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));
	float	result;

	result = arg1 - arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT4(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float42_mi);

Datum
pgstrom_float42_mul(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));
	float	result;

	result = arg1 * arg2;
	CHECKFLOATVAL(result,
				  isinf(arg1) || isinf(arg2),
				  arg1 == 0.0 || arg2 == 0.0);
	PG_RETURN_FLOAT4(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float42_mul);

Datum
pgstrom_float42_div(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));
	float	result;

	if (arg2 == 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));

	result = arg1 / arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT4(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float42_div);

Datum
pgstrom_float82_pl(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = fp16_to_fp64(PG_GETARG_FLOAT2(1));
	double	result;

	result = arg1 + arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT8(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float82_pl);

Datum
pgstrom_float82_mi(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = fp16_to_fp64(PG_GETARG_FLOAT2(1));
	double	result;

	result = arg1 - arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT8(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float82_mi);

Datum
pgstrom_float82_mul(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = fp16_to_fp64(PG_GETARG_FLOAT2(1));
	double	result;

	result = arg1 * arg2;

	CHECKFLOATVAL(result,
				  isinf(arg1) || isinf(arg2),
				  arg1 == 0.0 || arg2 == 0.0);
	PG_RETURN_FLOAT8(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float82_mul);

Datum
pgstrom_float82_div(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = fp16_to_fp64(PG_GETARG_FLOAT2(1));
	double	result;

	if (arg2 == 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));

	result = arg1 / arg2;

	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), arg1 == 0.0);
	PG_RETURN_FLOAT8(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float82_div);

/*
 * Misc functions
 */
Datum
pgstrom_cash_mul_float2(PG_FUNCTION_ARGS)
{
	Cash		c = PG_GETARG_CASH(0);
	float8		f = fp16_to_fp64(PG_GETARG_FLOAT2(1));
	Cash		result;

	result = rint(c * f);
	PG_RETURN_CASH(result);
}
PG_FUNCTION_INFO_V1(pgstrom_cash_mul_float2);

Datum
pgstrom_float2_mul_cash(PG_FUNCTION_ARGS)
{
	float8		f = fp16_to_fp64(PG_GETARG_FLOAT2(0));
	Cash		c = PG_GETARG_CASH(1);
	Cash		result;

	result = rint(f * c);
	PG_RETURN_CASH(result);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_mul_cash);

Datum
pgstrom_cash_div_float2(PG_FUNCTION_ARGS)
{
	Cash		c = PG_GETARG_CASH(0);
	float8		f = fp16_to_fp64(PG_GETARG_FLOAT2(1));
	Cash		result;

	if (f == 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));

	result = rint(c / f);
	PG_RETURN_CASH(result);
}
PG_FUNCTION_INFO_V1(pgstrom_cash_div_float2);

Datum
pgstrom_float8_as_int8(PG_FUNCTION_ARGS)
{
	float8	fval = PG_GETARG_FLOAT8(0);

	PG_RETURN_INT64(double_as_long(fval));
}
PG_FUNCTION_INFO_V1(pgstrom_float8_as_int8);

Datum
pgstrom_float4_as_int4(PG_FUNCTION_ARGS)
{
	float4	fval = PG_GETARG_FLOAT4(0);

	PG_RETURN_INT32(float_as_int(fval));
}
PG_FUNCTION_INFO_V1(pgstrom_float4_as_int4);

Datum
pgstrom_float2_as_int2(PG_FUNCTION_ARGS)
{
	half_t	fval = PG_GETARG_FLOAT2(0);

	PG_RETURN_INT16(fval);	/* actually, half_t is unsigned short */
}
PG_FUNCTION_INFO_V1(pgstrom_float2_as_int2);

Datum
pgstrom_int8_as_float8(PG_FUNCTION_ARGS)
{
	int64	ival = PG_GETARG_INT64(0);

	PG_RETURN_FLOAT8(long_as_double(ival));
}
PG_FUNCTION_INFO_V1(pgstrom_int8_as_float8);

Datum
pgstrom_int4_as_float4(PG_FUNCTION_ARGS)
{
	int32	ival = PG_GETARG_INT32(0);

	PG_RETURN_FLOAT4(int_as_float(ival));
}
PG_FUNCTION_INFO_V1(pgstrom_int4_as_float4);

Datum
pgstrom_int2_as_float2(PG_FUNCTION_ARGS)
{
	int16	ival = PG_GETARG_INT16(0);

	PG_RETURN_FLOAT2(ival);	/* actually, half_t is unsigned short */
}
PG_FUNCTION_INFO_V1(pgstrom_int2_as_float2);

Datum
pgstrom_float2_accum(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	/* do computations as float8 */
	float8      newval = fp16_to_fp64(PG_GETARG_FLOAT2(1));
	float8     *transvalues;
	float8      N, sumX, sumX2;

	/* logic in check_float8_array at utils/adt/float.c */
	if (ARR_NDIM(transarray) != 1 ||
		ARR_DIMS(transarray)[0] != 3 ||
		ARR_HASNULL(transarray) ||
		ARR_ELEMTYPE(transarray) != FLOAT8OID)
		elog(ERROR, "float2_accum: expected 3-element float8 array");

	transvalues = (float8 *) ARR_DATA_PTR(transarray);
	N = transvalues[0];
	sumX = transvalues[1];
	sumX2 = transvalues[2];

	N += 1.0;
	sumX += newval;
	CHECKFLOATVAL(sumX, isinf(transvalues[1]) || isinf(newval), true);
	sumX2 += newval * newval;
	CHECKFLOATVAL(sumX2, isinf(transvalues[2]) || isinf(newval), true);

	/*
	 * If we're invoked as an aggregate, we can cheat and modify our first
	 * parameter in-place to reduce palloc overhead. Otherwise we construct a
	 * new array with the updated transition data and return it.
	 */
	if (AggCheckCallContext(fcinfo, NULL))
	{
		transvalues[0] = N;
		transvalues[1] = sumX;
		transvalues[2] = sumX2;

		PG_RETURN_ARRAYTYPE_P(transarray);
	}
	else
	{
		Datum		transdatums[3];
		ArrayType  *result;

		transdatums[0] = Float8GetDatumFast(N);
		transdatums[1] = Float8GetDatumFast(sumX);
		transdatums[2] = Float8GetDatumFast(sumX2);

		result = construct_array(transdatums, 3,
								 FLOAT8OID,
								 sizeof(float8), FLOAT8PASSBYVAL, 'd');

		PG_RETURN_ARRAYTYPE_P(result);
	}
}
PG_FUNCTION_INFO_V1(pgstrom_float2_accum);

Datum
pgstrom_float2_sum(PG_FUNCTION_ARGS)
{
	float8		newval;

	if (PG_ARGISNULL(0))
	{
		if (PG_ARGISNULL(1))
			PG_RETURN_NULL();	/* still no non-null */
		newval = fp16_to_fp64(PG_GETARG_FLOAT2(1));
	}
	else
	{
		newval = PG_GETARG_FLOAT8(0);

		if (!PG_ARGISNULL(1))
			newval += fp16_to_fp64(PG_GETARG_FLOAT2(1));
	}
	PG_RETURN_FLOAT8(newval);
}
PG_FUNCTION_INFO_V1(pgstrom_float2_sum);
