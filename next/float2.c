/*
 * float2.c
 *
 * half-precision floating point data type support
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "float2.h"

#ifndef EMULATE_FLOAT2
#define PG_GETARG_FP16(x)			__short_as_half__(PG_GETARG_UINT16(x))
#define PG_GETARG_FP16_AS_FP32(x)	((float)PG_GETARG_FP16(x))
#define PG_GETARG_FP16_AS_FP64(x)	((double)PG_GETARG_FP16(x))
#define PG_RETURN_FP16(x)			PG_RETURN_UINT16(__half_as_short__(x))
#define PG_RETURN_FP32_AS_FP16(x)	PG_RETURN_FP16((float2_t)(x))
#define PG_RETURN_FP64_AS_FP16(x)	PG_RETURN_FP16((float2_t)(x))
#else
#define PG_GETARG_FP16(x)			PG_GETARG_UINT16(x)
#define PG_GETARG_FP16_AS_FP32(x)	fp16_to_fp32(PG_GETARG_FP16(x))
#define PG_GETARG_FP16_AS_FP64(x)	fp16_to_fp64(PG_GETARG_FP16(x))
#define PG_RETURN_FP16(x)			PG_RETURN_UINT16(x)
#define PG_RETURN_FP32_AS_FP16(x)	PG_RETURN_FP16(fp32_to_fp16(x))
#define PG_RETURN_FP64_AS_FP16(x)	PG_RETURN_FP16(fp64_to_fp16(x))
#endif

/* type i/o handler */
PG_FUNCTION_INFO_V1(pgstrom_float2in);
PG_FUNCTION_INFO_V1(pgstrom_float2out);
PG_FUNCTION_INFO_V1(pgstrom_float2recv);
PG_FUNCTION_INFO_V1(pgstrom_float2send);
/* type cast */
PG_FUNCTION_INFO_V1(pgstrom_float2_to_float4);
PG_FUNCTION_INFO_V1(pgstrom_float2_to_float8);
PG_FUNCTION_INFO_V1(pgstrom_float2_to_int1);
PG_FUNCTION_INFO_V1(pgstrom_float2_to_int2);
PG_FUNCTION_INFO_V1(pgstrom_float2_to_int4);
PG_FUNCTION_INFO_V1(pgstrom_float2_to_int8);
PG_FUNCTION_INFO_V1(pgstrom_float2_to_numeric);
PG_FUNCTION_INFO_V1(pgstrom_float4_to_float2);
PG_FUNCTION_INFO_V1(pgstrom_float8_to_float2);
PG_FUNCTION_INFO_V1(pgstrom_int1_to_float2);
PG_FUNCTION_INFO_V1(pgstrom_int2_to_float2);
PG_FUNCTION_INFO_V1(pgstrom_int4_to_float2);
PG_FUNCTION_INFO_V1(pgstrom_int8_to_float2);
PG_FUNCTION_INFO_V1(pgstrom_numeric_to_float2);
/* type comparison */
PG_FUNCTION_INFO_V1(pgstrom_float2eq);
PG_FUNCTION_INFO_V1(pgstrom_float2ne);
PG_FUNCTION_INFO_V1(pgstrom_float2lt);
PG_FUNCTION_INFO_V1(pgstrom_float2le);
PG_FUNCTION_INFO_V1(pgstrom_float2gt);
PG_FUNCTION_INFO_V1(pgstrom_float2ge);
PG_FUNCTION_INFO_V1(pgstrom_float2cmp);
PG_FUNCTION_INFO_V1(pgstrom_float2larger);
PG_FUNCTION_INFO_V1(pgstrom_float2smaller);
PG_FUNCTION_INFO_V1(pgstrom_float2hash);

PG_FUNCTION_INFO_V1(pgstrom_float42eq);
PG_FUNCTION_INFO_V1(pgstrom_float42ne);
PG_FUNCTION_INFO_V1(pgstrom_float42lt);
PG_FUNCTION_INFO_V1(pgstrom_float42le);
PG_FUNCTION_INFO_V1(pgstrom_float42gt);
PG_FUNCTION_INFO_V1(pgstrom_float42ge);
PG_FUNCTION_INFO_V1(pgstrom_float42cmp);

PG_FUNCTION_INFO_V1(pgstrom_float82eq);
PG_FUNCTION_INFO_V1(pgstrom_float82ne);
PG_FUNCTION_INFO_V1(pgstrom_float82lt);
PG_FUNCTION_INFO_V1(pgstrom_float82le);
PG_FUNCTION_INFO_V1(pgstrom_float82gt);
PG_FUNCTION_INFO_V1(pgstrom_float82ge);
PG_FUNCTION_INFO_V1(pgstrom_float82cmp);

PG_FUNCTION_INFO_V1(pgstrom_float24eq);
PG_FUNCTION_INFO_V1(pgstrom_float24ne);
PG_FUNCTION_INFO_V1(pgstrom_float24lt);
PG_FUNCTION_INFO_V1(pgstrom_float24le);
PG_FUNCTION_INFO_V1(pgstrom_float24gt);
PG_FUNCTION_INFO_V1(pgstrom_float24ge);
PG_FUNCTION_INFO_V1(pgstrom_float24cmp);

PG_FUNCTION_INFO_V1(pgstrom_float28eq);
PG_FUNCTION_INFO_V1(pgstrom_float28ne);
PG_FUNCTION_INFO_V1(pgstrom_float28lt);
PG_FUNCTION_INFO_V1(pgstrom_float28le);
PG_FUNCTION_INFO_V1(pgstrom_float28gt);
PG_FUNCTION_INFO_V1(pgstrom_float28ge);
PG_FUNCTION_INFO_V1(pgstrom_float28cmp);

/* unary operators */
PG_FUNCTION_INFO_V1(pgstrom_float2up);
PG_FUNCTION_INFO_V1(pgstrom_float2um);
PG_FUNCTION_INFO_V1(pgstrom_float2abs);
/* arithmetric operators */
PG_FUNCTION_INFO_V1(pgstrom_float2pl);
PG_FUNCTION_INFO_V1(pgstrom_float2mi);
PG_FUNCTION_INFO_V1(pgstrom_float2mul);
PG_FUNCTION_INFO_V1(pgstrom_float2div);

PG_FUNCTION_INFO_V1(pgstrom_float24pl);
PG_FUNCTION_INFO_V1(pgstrom_float24mi);
PG_FUNCTION_INFO_V1(pgstrom_float24mul);
PG_FUNCTION_INFO_V1(pgstrom_float24div);

PG_FUNCTION_INFO_V1(pgstrom_float28pl);
PG_FUNCTION_INFO_V1(pgstrom_float28mi);
PG_FUNCTION_INFO_V1(pgstrom_float28mul);
PG_FUNCTION_INFO_V1(pgstrom_float28div);

PG_FUNCTION_INFO_V1(pgstrom_float42pl);
PG_FUNCTION_INFO_V1(pgstrom_float42mi);
PG_FUNCTION_INFO_V1(pgstrom_float42mul);
PG_FUNCTION_INFO_V1(pgstrom_float42div);

PG_FUNCTION_INFO_V1(pgstrom_float82pl);
PG_FUNCTION_INFO_V1(pgstrom_float82mi);
PG_FUNCTION_INFO_V1(pgstrom_float82mul);
PG_FUNCTION_INFO_V1(pgstrom_float82div);

/* misc functions */
PG_FUNCTION_INFO_V1(pgstrom_cash_mul_flt2);
PG_FUNCTION_INFO_V1(pgstrom_flt2_mul_cash);
PG_FUNCTION_INFO_V1(pgstrom_cash_div_flt2);
PG_FUNCTION_INFO_V1(pgstrom_float2_accum);
PG_FUNCTION_INFO_V1(pgstrom_float2_sum);

static inline void
print_fp16(const char *prefix, uint32 value)
{
	elog(INFO, "%sFP16 0x%04x = %d + %d + 0x%04x",
		 prefix ? prefix : "",
		 value,
		 (value & 0x8000) ? 1 : 0,
		 ((value >> FP16_FRAC_BITS) & 0x001f) - FP16_EXPO_BIAS,
		 (value & 0x03ff));
}

static inline void
print_fp32(const char *prefix, uint32 value)
{
	elog(INFO, "%sFP32 0x%08x = %d + %d + 0x%08x",
		 prefix ? prefix : "",
         value,
		 (value & 0x80000000U) ? 1 : 0,
		 ((value >> FP32_FRAC_BITS) & 0x00ff) - FP32_EXPO_BIAS,
		 (value & 0x7fffff));
}

static inline void
print_fp64(const char *prefix, uint64 value)
{
	elog(INFO, "%sFP64 0x%016lx = %d + %ld + %014lx",
		 prefix ? prefix : "",
         value,
		 (value & 0x8000000000000000UL) ? 1 : 0,
		 ((value >> FP64_FRAC_BITS) & 0x07ff) - FP64_EXPO_BIAS,
		 (value & ((1UL << FP64_FRAC_BITS) - 1)));
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
 * pgstrom_float2in
 */
Datum
pgstrom_float2in(PG_FUNCTION_ARGS)
{
	float	fval = DatumGetFloat4(float4in(fcinfo));

	PG_RETURN_FP32_AS_FP16(fval);
}

/*
 * pgstrom_float2out
 */
Datum
pgstrom_float2out(PG_FUNCTION_ARGS)
{
	float	fval = PG_GETARG_FP16_AS_FP32(0);

	return DirectFunctionCall1(float4out, Float4GetDatum(fval));
}

/*
 * pgstrom_float2recv
 */
Datum
pgstrom_float2recv(PG_FUNCTION_ARGS)
{
	return int2recv(fcinfo);
}

/*
 * pgstrom_float2send
 */
Datum
pgstrom_float2send(PG_FUNCTION_ARGS)
{
	return int2send(fcinfo);
}

/*
 * pgstrom_float2_to_float4
 */
Datum
pgstrom_float2_to_float4(PG_FUNCTION_ARGS)
{
	float	fval = PG_GETARG_FP16_AS_FP32(0);

	PG_RETURN_FLOAT4(fval);
}

/*
 * pgstrom_float2_to_float8
 */
Datum
pgstrom_float2_to_float8(PG_FUNCTION_ARGS)
{
	double	fval = PG_GETARG_FP16_AS_FP64(0);

	PG_RETURN_FLOAT8(fval);
}

/*
 * pgstrom_float2_to_int1
 */
Datum
pgstrom_float2_to_int1(PG_FUNCTION_ARGS)
{
	float	fval = PG_GETARG_FP16_AS_FP32(0);
	Datum	ival = DirectFunctionCall1(ftoi4, Float4GetDatum(fval));

	if (DatumGetInt32(ival) < SCHAR_MIN ||
		DatumGetInt32(ival) > SCHAR_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("tinyint out of range")));
	return ival;
}

/*
 * pgstrom_float2_to_int2
 */
Datum
pgstrom_float2_to_int2(PG_FUNCTION_ARGS)
{
	float	fval = PG_GETARG_FP16_AS_FP32(0);

	return DirectFunctionCall1(ftoi2, Float4GetDatum(fval));
}

/*
 * pgstrom_float2_to_int4
 */
Datum
pgstrom_float2_to_int4(PG_FUNCTION_ARGS)
{
	float	fval = PG_GETARG_FP16_AS_FP32(0);

	return DirectFunctionCall1(ftoi4, Float4GetDatum(fval));
}

/*
 * pgstrom_float2_to_int8
 */
Datum
pgstrom_float2_to_int8(PG_FUNCTION_ARGS)
{
	float	fval = PG_GETARG_FP16_AS_FP32(0);

	return DirectFunctionCall1(ftoi8, Float4GetDatum(fval));
}

/*
 * pgstrom_float2_to_numeric
 */
Datum
pgstrom_float2_to_numeric(PG_FUNCTION_ARGS)
{
	float	fval = PG_GETARG_FP16_AS_FP32(0);

	return DirectFunctionCall1(float4_numeric, Float4GetDatum(fval));
}

/*
 * pgstrom_float4_to_float2
 */
Datum
pgstrom_float4_to_float2(PG_FUNCTION_ARGS)
{
	float	fval = PG_GETARG_FLOAT4(0);

	PG_RETURN_FP32_AS_FP16(fval);
}

/*
 * pgstrom_float8_to_float2
 */
Datum
pgstrom_float8_to_float2(PG_FUNCTION_ARGS)
{
	double	fval = PG_GETARG_FLOAT8(0);

	PG_RETURN_FP64_AS_FP16(fval);
}

/*
 * pgstrom_int1_to_float2
 */
Datum
pgstrom_int1_to_float2(PG_FUNCTION_ARGS)
{
	int32	ival = (int32)PG_GETARG_DATUM(0);

	PG_RETURN_FP32_AS_FP16((float)ival);
}

/*
 * pgstrom_int2_to_float2
 */
Datum
pgstrom_int2_to_float2(PG_FUNCTION_ARGS)
{
	float	fval = (float) PG_GETARG_INT16(0);

	PG_RETURN_FP32_AS_FP16(fval);
}

/*
 * pgstrom_int4_to_float2
 */
Datum
pgstrom_int4_to_float2(PG_FUNCTION_ARGS)
{
	double	fval = (double) PG_GETARG_INT32(0);

	PG_RETURN_FP64_AS_FP16(fval);
}

/*
 * pgstrom_int8_to_float2
 */
Datum
pgstrom_int8_to_float2(PG_FUNCTION_ARGS)
{
	double	fval = (double) PG_GETARG_INT64(0);

	PG_RETURN_FP64_AS_FP16(fval);
}

/*
 * pgstrom_numeric_to_float2
 */
Datum
pgstrom_numeric_to_float2(PG_FUNCTION_ARGS)
{
	float	fval = DatumGetFloat4(numeric_float4(fcinfo));

	PG_RETURN_FP32_AS_FP16(fval);
}

/*
 * Comparison operators
 */
Datum
pgstrom_float2eq(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) == 0);
}

Datum
pgstrom_float2ne(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) != 0);
}

Datum
pgstrom_float2lt(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) < 0);
}

Datum
pgstrom_float2le(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) <= 0);
}

Datum
pgstrom_float2gt(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) > 0);
}

Datum
pgstrom_float2ge(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) >= 0);
}

Datum
pgstrom_float2cmp(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);
	int		comp = float4_cmp_internal(arg1, arg2);

	PG_RETURN_INT32(comp > 0 ? 1 : (comp < 0 ? -1 : 0));
}

Datum
pgstrom_float2larger(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);

	PG_RETURN_DATUM(arg1 > arg2 ? PG_GETARG_DATUM(0) : PG_GETARG_DATUM(1));
}

Datum
pgstrom_float2smaller(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);

	PG_RETURN_DATUM(arg1 < arg2 ? PG_GETARG_DATUM(0) : PG_GETARG_DATUM(1));
}

Datum
pgstrom_float2hash(PG_FUNCTION_ARGS)
{
	half_t	fval = PG_GETARG_UINT16(0);
	int32	sign = (fval & 0x8000);
	int32	expo = (fval & 0x7c00) >> 10;
	int32	frac = (fval & 0x03ff);

	if (expo == 0x1f)
	{
		if (frac == 0)
			PG_RETURN_INT32(sign ? -INT_MAX : INT_MAX);	/* +/-Infinity */
		else
			PG_RETURN_INT32(UINT_MAX);					/* NaN */
	}
	else if (expo == 0 && frac == 0)
		PG_RETURN_INT32(0);								/* +/-0.0 */

	/* elsewhere, normal finite values */
	return hash_any((unsigned char *)&fval, sizeof(half_t));
}

Datum
pgstrom_float42eq(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) == 0);
}

Datum
pgstrom_float42ne(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) != 0);
}

Datum
pgstrom_float42lt(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) < 0);
}

Datum
pgstrom_float42le(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) <= 0);
}

Datum
pgstrom_float42gt(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) > 0);
}

Datum
pgstrom_float42ge(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) >= 0);
}

Datum
pgstrom_float42cmp(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);
	int		comp = float4_cmp_internal(arg1, arg2);

	PG_RETURN_INT32(comp > 0 ? 1 : (comp < 0 ? -1 : 0));
}

Datum
pgstrom_float82eq(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = PG_GETARG_FP16_AS_FP64(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) == 0);
}

Datum
pgstrom_float82ne(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = PG_GETARG_FP16_AS_FP64(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) != 0);
}

Datum
pgstrom_float82lt(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = PG_GETARG_FP16_AS_FP64(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) < 0);
}

Datum
pgstrom_float82le(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = PG_GETARG_FP16_AS_FP64(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) <= 0);
}

Datum
pgstrom_float82gt(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = PG_GETARG_FP16_AS_FP64(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) > 0);
}

Datum
pgstrom_float82ge(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = PG_GETARG_FP16_AS_FP64(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) >= 0);
}

Datum
pgstrom_float82cmp(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = PG_GETARG_FP16_AS_FP64(1);
	int		comp = float8_cmp_internal(arg1, arg2);

	PG_RETURN_INT32(comp > 0 ? 1 : (comp < 0 ? -1 : 0));
}

Datum
pgstrom_float24eq(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) == 0);
}

Datum
pgstrom_float24ne(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) != 0);
}

Datum
pgstrom_float24lt(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) < 0);
}

Datum
pgstrom_float24le(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) <= 0);
}

Datum
pgstrom_float24gt(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) > 0);
}

Datum
pgstrom_float24ge(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) >= 0);
}

Datum
pgstrom_float24cmp(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
  	float	arg2 = PG_GETARG_FLOAT4(1);
	int		comp = float4_cmp_internal(arg1, arg2);

	PG_RETURN_INT32(comp > 0 ? 1 : (comp < 0 ? -1 : 0));
}

Datum
pgstrom_float28eq(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FP16_AS_FP64(0);
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) == 0);
}

Datum
pgstrom_float28ne(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FP16_AS_FP64(0);
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) != 0);
}

Datum
pgstrom_float28lt(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FP16_AS_FP64(0);
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) < 0);
}

Datum
pgstrom_float28le(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FP16_AS_FP64(0);
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) <= 0);
}

Datum
pgstrom_float28gt(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FP16_AS_FP64(0);
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) > 0);
}

Datum
pgstrom_float28ge(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FP16_AS_FP64(0);
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) >= 0);
}

Datum
pgstrom_float28cmp(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FP16_AS_FP64(0);
  	double	arg2 = PG_GETARG_FLOAT8(1);
	int		comp = float8_cmp_internal(arg1, arg2);

	PG_RETURN_INT32(comp > 0 ? 1 : (comp < 0 ? -1 : 0));
}

/*
 * unary operators
 */
Datum
pgstrom_float2up(PG_FUNCTION_ARGS)
{
	float2_t	fval = PG_GETARG_FP16(0);

	PG_RETURN_FP16(fval);
}

Datum
pgstrom_float2um(PG_FUNCTION_ARGS)
{
	float2_t	fval = PG_GETARG_FP16(0);
#ifndef EMULATE_FLOAT2
	fval = -fval;
#else
	fval ^= ~0x8000;
#endif
	PG_RETURN_FP16(fval);
}

Datum
pgstrom_float2abs(PG_FUNCTION_ARGS)
{
	float2_t	fval = PG_GETARG_FP16(0);
#ifndef EMULATE_FLOAT2
	fval = abs(fval);
#else
	fval &= ~0x8000;
#endif
	PG_RETURN_FP16(fval);
}

/*
 * arithmetic operations
 */
Datum
pgstrom_float2pl(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);
	float	result;

	result = arg1 + arg2;

	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT4(result);
}

Datum
pgstrom_float2mi(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);
	float	result;

	result = arg1 - arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT4(result);
}

Datum
pgstrom_float2mul(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);
	float	result;

	result = arg1 * arg2;

	CHECKFLOATVAL(result,
				  isinf(arg1) || isinf(arg2),
				  arg1 == 0 || arg2 == 0);
	PG_RETURN_FLOAT4(result);
}

Datum
pgstrom_float2div(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);
	float	result;

	if (arg2 == 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));
	result = arg1 / arg2;

	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), arg1 == 0.0);
	PG_RETURN_FLOAT4(result);
}

Datum
pgstrom_float24pl(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
	float	arg2 = PG_GETARG_FLOAT4(1);
	float	result;

	result = arg1 + arg2;

	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT4(result);
}

Datum
pgstrom_float24mi(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
	float	arg2 = PG_GETARG_FLOAT4(1);
	float	result;

	result = arg1 - arg2;

	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT4(result);
}

Datum
pgstrom_float24mul(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
	float	arg2 = PG_GETARG_FLOAT4(1);
	float	result;

	result = arg1 * arg2;
	CHECKFLOATVAL(result,
				  isinf(arg1) || isinf(arg2),
				  arg1 == 0 || arg2 == 0);
	PG_RETURN_FLOAT4(result);
}

Datum
pgstrom_float24div(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FP16_AS_FP32(0);
	float	arg2 = PG_GETARG_FLOAT4(1);
	float	result;

	if (arg2 == 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));

	result = arg1 / arg2;

	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), arg1 == 0.0);
	PG_RETURN_FLOAT4(result);
}

Datum
pgstrom_float28pl(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FP16_AS_FP32(0);
	double	arg2 = PG_GETARG_FLOAT8(1);
	double	result;

	result = arg1 + arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);

	PG_RETURN_FLOAT8(result);
}

Datum
pgstrom_float28mi(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FP16_AS_FP64(0);
	double	arg2 = PG_GETARG_FLOAT8(1);
	double	result;

	result = arg1 - arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);

	PG_RETURN_FLOAT8(result);	
}

Datum
pgstrom_float28mul(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FP16_AS_FP64(0);
	double	arg2 = PG_GETARG_FLOAT8(1);
	double	result;

	result = arg1 * arg2;
	CHECKFLOATVAL(result,
				  isinf(arg1) || isinf(arg2),
				  arg1 == 0.0 || arg2 == 0.0);
	PG_RETURN_FLOAT8(result);
}

Datum
pgstrom_float28div(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FP16_AS_FP64(0);
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

Datum
pgstrom_float42pl(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);
	float	result;

	result = arg1 + arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT4(result);
}

Datum
pgstrom_float42mi(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);
	float	result;

	result = arg1 - arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT4(result);
}

Datum
pgstrom_float42mul(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);
	float	result;

	result = arg1 * arg2;
	CHECKFLOATVAL(result,
				  isinf(arg1) || isinf(arg2),
				  arg1 == 0.0 || arg2 == 0.0);
	PG_RETURN_FLOAT4(result);
}

Datum
pgstrom_float42div(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = PG_GETARG_FP16_AS_FP32(1);
	float	result;

	if (arg2 == 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));

	result = arg1 / arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT4(result);
}

Datum
pgstrom_float82pl(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = PG_GETARG_FP16_AS_FP64(1);
	double	result;

	result = arg1 + arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT8(result);
}

Datum
pgstrom_float82mi(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = PG_GETARG_FP16_AS_FP64(1);
	double	result;

	result = arg1 - arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT8(result);
}

Datum
pgstrom_float82mul(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = PG_GETARG_FP16_AS_FP64(1);
	double	result;

	result = arg1 * arg2;

	CHECKFLOATVAL(result,
				  isinf(arg1) || isinf(arg2),
				  arg1 == 0.0 || arg2 == 0.0);
	PG_RETURN_FLOAT8(result);
}

Datum
pgstrom_float82div(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = PG_GETARG_FP16_AS_FP64(1);
	double	result;

	if (arg2 == 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));

	result = arg1 / arg2;

	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), arg1 == 0.0);
	PG_RETURN_FLOAT8(result);
}

/*
 * Misc functions
 */
Datum
pgstrom_cash_mul_flt2(PG_FUNCTION_ARGS)
{
	Cash		c = PG_GETARG_CASH(0);
	float8		f = PG_GETARG_FP16_AS_FP64(1);
	Cash		result;

	result = rint(c * f);
	PG_RETURN_CASH(result);
}

Datum
pgstrom_flt2_mul_cash(PG_FUNCTION_ARGS)
{
	float8		f = PG_GETARG_FP16_AS_FP64(0);
	Cash		c = PG_GETARG_CASH(1);
	Cash		result;

	result = rint(f * c);
	PG_RETURN_CASH(result);
}

Datum
pgstrom_cash_div_flt2(PG_FUNCTION_ARGS)
{
	Cash		c = PG_GETARG_CASH(0);
	float8		f = PG_GETARG_FP16_AS_FP64(1);
	Cash		result;

	if (f == 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));

	result = rint(c / f);
	PG_RETURN_CASH(result);
}

Datum
pgstrom_float2_accum(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	/* do computations as float8 */
	float8      newval = PG_GETARG_FP16_AS_FP64(1);
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

Datum
pgstrom_float2_sum(PG_FUNCTION_ARGS)
{
	float8		newval;

	if (PG_ARGISNULL(0))
	{
		if (PG_ARGISNULL(1))
			PG_RETURN_NULL();	/* still no non-null */
		newval = PG_GETARG_FP16_AS_FP64(1);
	}
	else
	{
		newval = PG_GETARG_FLOAT8(0);

		if (!PG_ARGISNULL(1))
			newval += PG_GETARG_FP16_AS_FP64(1);
	}
	PG_RETURN_FLOAT8(newval);
}
