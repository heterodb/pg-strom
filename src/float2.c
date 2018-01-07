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
//#define DEBUG_FP16 1

static inline void
print_fp16(const char *prefix, cl_uint value)
{
#ifdef DEBUG_FP16
	elog(INFO, "%sFP16 0x%04x = %d + %d + 0x%04x",
		 prefix ? prefix : "",
		 value,
		 (value & 0x8000) ? 1 : 0,
		 ((value >> FP16_FRAC_BITS) & 0x0015) - FP16_EXPO_BIAS,
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
		 (value & (1UL<<63)) ? 1 : 0,
		 ((value >> FP32_FRAC_BITS) & 0x03ff) - FP64_EXPO_BIAS,
		 (value & ((1UL << 53) - 1)));
#endif
}

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

		frac = ((frac >> 12) | 0x400) + ((frac >> 11) & 1);
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
		result = (sign | (expo << FP32_FRAC_BITS) | (frac << 12));
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

		frac = ((frac >> 41) | 0x400) + ((frac >> 40) & 1);
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

		expo += FP32_EXPO_BIAS;
		result = (sign | (expo << FP32_FRAC_BITS) | frac);
	}
	print_fp64("<-", result);
	return int_as_float(result);
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
