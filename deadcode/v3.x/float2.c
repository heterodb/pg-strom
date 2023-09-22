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

#define PG_GETARG_FLOAT2(x)	PG_GETARG_INT16(x)
#define PG_RETURN_FLOAT2(x)	PG_RETURN_INT16(x)
#define DatumGetFloat2(x)	DatumGetInt16(x)
#define Float2GetDatum(x)	Int16GetDatum(x)

/* type i/o handler */
PG_FUNCTION_INFO_V1(pgstrom_float2_in);
PG_FUNCTION_INFO_V1(pgstrom_float2_out);
PG_FUNCTION_INFO_V1(pgstrom_float2_recv);
PG_FUNCTION_INFO_V1(pgstrom_float2_send);
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
PG_FUNCTION_INFO_V1(pgstrom_float2_eq);
PG_FUNCTION_INFO_V1(pgstrom_float2_ne);
PG_FUNCTION_INFO_V1(pgstrom_float2_lt);
PG_FUNCTION_INFO_V1(pgstrom_float2_le);
PG_FUNCTION_INFO_V1(pgstrom_float2_gt);
PG_FUNCTION_INFO_V1(pgstrom_float2_ge);
PG_FUNCTION_INFO_V1(pgstrom_float2_cmp);
PG_FUNCTION_INFO_V1(pgstrom_float2_larger);
PG_FUNCTION_INFO_V1(pgstrom_float2_smaller);
PG_FUNCTION_INFO_V1(pgstrom_float2_hash);

PG_FUNCTION_INFO_V1(pgstrom_float42_eq);
PG_FUNCTION_INFO_V1(pgstrom_float42_ne);
PG_FUNCTION_INFO_V1(pgstrom_float42_lt);
PG_FUNCTION_INFO_V1(pgstrom_float42_le);
PG_FUNCTION_INFO_V1(pgstrom_float42_gt);
PG_FUNCTION_INFO_V1(pgstrom_float42_ge);
PG_FUNCTION_INFO_V1(pgstrom_float42_cmp);

PG_FUNCTION_INFO_V1(pgstrom_float82_eq);
PG_FUNCTION_INFO_V1(pgstrom_float82_ne);
PG_FUNCTION_INFO_V1(pgstrom_float82_lt);
PG_FUNCTION_INFO_V1(pgstrom_float82_le);
PG_FUNCTION_INFO_V1(pgstrom_float82_gt);
PG_FUNCTION_INFO_V1(pgstrom_float82_ge);
PG_FUNCTION_INFO_V1(pgstrom_float82_cmp);

PG_FUNCTION_INFO_V1(pgstrom_float24_eq);
PG_FUNCTION_INFO_V1(pgstrom_float24_ne);
PG_FUNCTION_INFO_V1(pgstrom_float24_lt);
PG_FUNCTION_INFO_V1(pgstrom_float24_le);
PG_FUNCTION_INFO_V1(pgstrom_float24_gt);
PG_FUNCTION_INFO_V1(pgstrom_float24_ge);
PG_FUNCTION_INFO_V1(pgstrom_float24_cmp);

PG_FUNCTION_INFO_V1(pgstrom_float28_eq);
PG_FUNCTION_INFO_V1(pgstrom_float28_ne);
PG_FUNCTION_INFO_V1(pgstrom_float28_lt);
PG_FUNCTION_INFO_V1(pgstrom_float28_le);
PG_FUNCTION_INFO_V1(pgstrom_float28_gt);
PG_FUNCTION_INFO_V1(pgstrom_float28_ge);
PG_FUNCTION_INFO_V1(pgstrom_float28_cmp);

/* unary operators */
PG_FUNCTION_INFO_V1(pgstrom_float2_up);
PG_FUNCTION_INFO_V1(pgstrom_float2_um);
PG_FUNCTION_INFO_V1(pgstrom_float2_abs);
/* arithmetric operators */
PG_FUNCTION_INFO_V1(pgstrom_float2_pl);
PG_FUNCTION_INFO_V1(pgstrom_float2_mi);
PG_FUNCTION_INFO_V1(pgstrom_float2_mul);
PG_FUNCTION_INFO_V1(pgstrom_float2_div);

PG_FUNCTION_INFO_V1(pgstrom_float24_pl);
PG_FUNCTION_INFO_V1(pgstrom_float24_mi);
PG_FUNCTION_INFO_V1(pgstrom_float24_mul);
PG_FUNCTION_INFO_V1(pgstrom_float24_div);

PG_FUNCTION_INFO_V1(pgstrom_float28_pl);
PG_FUNCTION_INFO_V1(pgstrom_float28_mi);
PG_FUNCTION_INFO_V1(pgstrom_float28_mul);
PG_FUNCTION_INFO_V1(pgstrom_float28_div);

PG_FUNCTION_INFO_V1(pgstrom_float42_pl);
PG_FUNCTION_INFO_V1(pgstrom_float42_mi);
PG_FUNCTION_INFO_V1(pgstrom_float42_mul);
PG_FUNCTION_INFO_V1(pgstrom_float42_div);

PG_FUNCTION_INFO_V1(pgstrom_float82_pl);
PG_FUNCTION_INFO_V1(pgstrom_float82_mi);
PG_FUNCTION_INFO_V1(pgstrom_float82_mul);
PG_FUNCTION_INFO_V1(pgstrom_float82_div);

/* misc functions */
PG_FUNCTION_INFO_V1(pgstrom_cash_mul_flt2);
PG_FUNCTION_INFO_V1(pgstrom_flt2_mul_cash);
PG_FUNCTION_INFO_V1(pgstrom_cash_div_flt2);
PG_FUNCTION_INFO_V1(pgstrom_float8_as_int8);
PG_FUNCTION_INFO_V1(pgstrom_float4_as_int4);
PG_FUNCTION_INFO_V1(pgstrom_float2_as_int2);
PG_FUNCTION_INFO_V1(pgstrom_int8_as_float8);
PG_FUNCTION_INFO_V1(pgstrom_int4_as_float4);
PG_FUNCTION_INFO_V1(pgstrom_int2_as_float2);
PG_FUNCTION_INFO_V1(pgstrom_float2_accum);
PG_FUNCTION_INFO_V1(pgstrom_float2_sum);
PG_FUNCTION_INFO_V1(pgstrom_define_shell_type);

static inline void
print_fp16(const char *prefix, cl_uint value)
{
	elog(INFO, "%sFP16 0x%04x = %d + %d + 0x%04x",
		 prefix ? prefix : "",
		 value,
		 (value & 0x8000) ? 1 : 0,
		 ((value >> FP16_FRAC_BITS) & 0x001f) - FP16_EXPO_BIAS,
		 (value & 0x03ff));
}

static inline void
print_fp32(const char *prefix, cl_uint value)
{
	elog(INFO, "%sFP32 0x%08x = %d + %d + 0x%08x",
		 prefix ? prefix : "",
         value,
		 (value & 0x80000000U) ? 1 : 0,
		 ((value >> FP32_FRAC_BITS) & 0x00ff) - FP32_EXPO_BIAS,
		 (value & 0x7fffff));
}

static inline void
print_fp64(const char *prefix, cl_ulong value)
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

/*
 * pgstrom_float2_out
 */
Datum
pgstrom_float2_out(PG_FUNCTION_ARGS)
{
	float	fval = fp16_to_fp32((half_t)PG_GETARG_FLOAT2(0));

	return DirectFunctionCall1(float4out, Float4GetDatum(fval));
}

/*
 * pgstrom_float2_recv
 */
Datum
pgstrom_float2_recv(PG_FUNCTION_ARGS)
{
	return int2recv(fcinfo);
}

/*
 * pgstrom_float2_send
 */
Datum
pgstrom_float2_send(PG_FUNCTION_ARGS)
{
	return int2send(fcinfo);
}

/*
 * pgstrom_float2_to_float4
 */
Datum
pgstrom_float2_to_float4(PG_FUNCTION_ARGS)
{
	half_t	fval = PG_GETARG_FLOAT2(0);

	PG_RETURN_FLOAT4(fp16_to_fp32(fval));
}

/*
 * pgstrom_float2_to_float8
 */
Datum
pgstrom_float2_to_float8(PG_FUNCTION_ARGS)
{
	half_t	fval = PG_GETARG_FLOAT2(0);

	PG_RETURN_FLOAT8(fp16_to_fp64(fval));
}

/*
 * pgstrom_float2_to_int1
 */
Datum
pgstrom_float2_to_int1(PG_FUNCTION_ARGS)
{
	float	fval = fp16_to_fp32(PG_GETARG_FLOAT2(0));
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
	float	fval = fp16_to_fp32(PG_GETARG_FLOAT2(0));

	return DirectFunctionCall1(ftoi2, Float4GetDatum(fval));
}

/*
 * pgstrom_float2_to_int4
 */
Datum
pgstrom_float2_to_int4(PG_FUNCTION_ARGS)
{
	float	fval = fp16_to_fp32(PG_GETARG_FLOAT2(0));

	return DirectFunctionCall1(ftoi4, Float4GetDatum(fval));
}

/*
 * pgstrom_float2_to_int8
 */
Datum
pgstrom_float2_to_int8(PG_FUNCTION_ARGS)
{
	double	fval = fp16_to_fp64(PG_GETARG_FLOAT2(0));

	return DirectFunctionCall1(dtoi8, Float8GetDatum(fval));
}

/*
 * pgstrom_float2_to_numeric
 */
Datum
pgstrom_float2_to_numeric(PG_FUNCTION_ARGS)
{
	float	fval = fp16_to_fp32(PG_GETARG_FLOAT2(0));

	return DirectFunctionCall1(float4_numeric, Float4GetDatum(fval));
}

/*
 * pgstrom_float4_to_float2
 */
Datum
pgstrom_float4_to_float2(PG_FUNCTION_ARGS)
{
	float	fval = PG_GETARG_FLOAT4(0);

	PG_RETURN_FLOAT2(fp32_to_fp16(fval));
}

/*
 * pgstrom_float8_to_float2
 */
Datum
pgstrom_float8_to_float2(PG_FUNCTION_ARGS)
{
	double	fval = PG_GETARG_FLOAT8(0);

	PG_RETURN_FLOAT2(fp64_to_fp16(fval));
}

/*
 * pgstrom_int1_to_float2
 */
Datum
pgstrom_int1_to_float2(PG_FUNCTION_ARGS)
{
	float	fval = (float)((int32)PG_GETARG_DATUM(0));

	PG_RETURN_FLOAT2(fp32_to_fp16(fval));
}

/*
 * pgstrom_int2_to_float2
 */
Datum
pgstrom_int2_to_float2(PG_FUNCTION_ARGS)
{
	float	fval = (float) PG_GETARG_INT16(0);

	PG_RETURN_FLOAT2(fp32_to_fp16(fval));
}

/*
 * pgstrom_int4_to_float2
 */
Datum
pgstrom_int4_to_float2(PG_FUNCTION_ARGS)
{
	double	fval = (double) PG_GETARG_INT32(0);

	PG_RETURN_FLOAT2(fp64_to_fp16(fval));
}

/*
 * pgstrom_int8_to_float2
 */
Datum
pgstrom_int8_to_float2(PG_FUNCTION_ARGS)
{
	double	fval = (double) PG_GETARG_INT64(0);

	PG_RETURN_FLOAT2(fp64_to_fp16(fval));
}

/*
 * pgstrom_numeric_to_float2
 */
Datum
pgstrom_numeric_to_float2(PG_FUNCTION_ARGS)
{
	float	fval = DatumGetFloat4(numeric_float4(fcinfo));

	PG_RETURN_FLOAT2(fp32_to_fp16(fval));
}

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

Datum
pgstrom_float2_ne(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) != 0);
}

Datum
pgstrom_float2_lt(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) < 0);
}

Datum
pgstrom_float2_le(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) <= 0);
}

Datum
pgstrom_float2_gt(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) > 0);
}

Datum
pgstrom_float2_ge(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) >= 0);
}

Datum
pgstrom_float2_cmp(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));
	int		comp = float4_cmp_internal(arg1, arg2);

	PG_RETURN_INT32(comp > 0 ? 1 : (comp < 0 ? -1 : 0));
}

Datum
pgstrom_float2_larger(PG_FUNCTION_ARGS)
{
	half_t	arg1 = PG_GETARG_FLOAT2(0);
	half_t	arg2 = PG_GETARG_FLOAT2(1);

	PG_RETURN_FLOAT2(fp16_to_fp32(arg1) > fp16_to_fp32(arg2) ? arg1 : arg2);
}

Datum
pgstrom_float2_smaller(PG_FUNCTION_ARGS)
{
	half_t	arg1 = PG_GETARG_FLOAT2(0);
	half_t	arg2 = PG_GETARG_FLOAT2(1);

	PG_RETURN_FLOAT2(fp16_to_fp32(arg1) < fp16_to_fp32(arg2) ? arg1 : arg2);
}

Datum
pgstrom_float2_hash(PG_FUNCTION_ARGS)
{
	half_t	fval = PG_GETARG_FLOAT2(0);
	cl_int	sign = (fval & 0x8000);
	cl_int	expo = (fval & 0x7c00) >> 10;
	cl_int	frac = (fval & 0x03ff);

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
pgstrom_float42_eq(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) == 0);
}

Datum
pgstrom_float42_ne(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) != 0);
}

Datum
pgstrom_float42_lt(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) < 0);
}

Datum
pgstrom_float42_le(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) <= 0);
}

Datum
pgstrom_float42_gt(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) > 0);
}

Datum
pgstrom_float42_ge(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) >= 0);
}

Datum
pgstrom_float42_cmp(PG_FUNCTION_ARGS)
{
	float	arg1 = PG_GETARG_FLOAT4(0);
	float	arg2 = fp16_to_fp32(PG_GETARG_FLOAT2(1));
	int		comp = float4_cmp_internal(arg1, arg2);

	PG_RETURN_INT32(comp > 0 ? 1 : (comp < 0 ? -1 : 0));
}

Datum
pgstrom_float82_eq(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = fp16_to_fp64(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) == 0);
}

Datum
pgstrom_float82_ne(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = fp16_to_fp64(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) != 0);
}

Datum
pgstrom_float82_lt(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = fp16_to_fp64(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) < 0);
}

Datum
pgstrom_float82_le(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = fp16_to_fp64(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) <= 0);
}

Datum
pgstrom_float82_gt(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = fp16_to_fp64(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) > 0);
}

Datum
pgstrom_float82_ge(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = fp16_to_fp64(PG_GETARG_FLOAT2(1));

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) >= 0);
}

Datum
pgstrom_float82_cmp(PG_FUNCTION_ARGS)
{
	double	arg1 = PG_GETARG_FLOAT8(0);
	double	arg2 = fp16_to_fp64(PG_GETARG_FLOAT2(1));
	int		comp = float8_cmp_internal(arg1, arg2);

	PG_RETURN_INT32(comp > 0 ? 1 : (comp < 0 ? -1 : 0));
}

Datum
pgstrom_float24_eq(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) == 0);
}

Datum
pgstrom_float24_ne(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) != 0);
}

Datum
pgstrom_float24_lt(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) < 0);
}

Datum
pgstrom_float24_le(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) <= 0);
}

Datum
pgstrom_float24_gt(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) > 0);
}

Datum
pgstrom_float24_ge(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
  	float	arg2 = PG_GETARG_FLOAT4(1);

	PG_RETURN_BOOL(float4_cmp_internal(arg1, arg2) >= 0);
}

Datum
pgstrom_float24_cmp(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
  	float	arg2 = PG_GETARG_FLOAT4(1);
	int		comp = float4_cmp_internal(arg1, arg2);

	PG_RETURN_INT32(comp > 0 ? 1 : (comp < 0 ? -1 : 0));
}

Datum
pgstrom_float28_eq(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) == 0);
}

Datum
pgstrom_float28_ne(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) != 0);
}

Datum
pgstrom_float28_lt(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) < 0);
}

Datum
pgstrom_float28_le(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) <= 0);
}

Datum
pgstrom_float28_gt(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) > 0);
}

Datum
pgstrom_float28_ge(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
  	double	arg2 = PG_GETARG_FLOAT8(1);

	PG_RETURN_BOOL(float8_cmp_internal(arg1, arg2) >= 0);
}

Datum
pgstrom_float28_cmp(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
  	double	arg2 = PG_GETARG_FLOAT8(1);
	int		comp = float8_cmp_internal(arg1, arg2);

	PG_RETURN_INT32(comp > 0 ? 1 : (comp < 0 ? -1 : 0));
}

/*
 * unary operators
 */
Datum
pgstrom_float2_up(PG_FUNCTION_ARGS)
{
	half_t	fval = PG_GETARG_FLOAT2(0);

	PG_RETURN_FLOAT2(fval);
}

Datum
pgstrom_float2_um(PG_FUNCTION_ARGS)
{
	half_t	fval = PG_GETARG_FLOAT2(0);

	fval ^= 0x8000;

	PG_RETURN_FLOAT2(fval);
}

Datum
pgstrom_float2_abs(PG_FUNCTION_ARGS)
{
	half_t	fval = PG_GETARG_FLOAT2(0);

	fval &= ~0x8000;

	PG_RETURN_FLOAT2(fval);
}

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

	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT4(result);
}

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

Datum
pgstrom_float24_pl(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = PG_GETARG_FLOAT4(1);
	float	result;

	result = arg1 + arg2;

	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT4(result);
}

Datum
pgstrom_float24_mi(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = PG_GETARG_FLOAT4(1);
	float	result;

	result = arg1 - arg2;

	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);
	PG_RETURN_FLOAT4(result);
}

Datum
pgstrom_float24_mul(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
	float	arg2 = PG_GETARG_FLOAT4(1);
	float	result;

	result = arg1 * arg2;
	CHECKFLOATVAL(result,
				  isinf(arg1) || isinf(arg2),
				  arg1 == 0 || arg2 == 0);
	PG_RETURN_FLOAT4(result);
}

Datum
pgstrom_float24_div(PG_FUNCTION_ARGS)
{
	float	arg1 = fp16_to_fp32(PG_GETARG_FLOAT2(0));
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
pgstrom_float28_pl(PG_FUNCTION_ARGS)
{
	double	arg1 = fp16_to_fp64(PG_GETARG_FLOAT2(0));
	double	arg2 = PG_GETARG_FLOAT8(1);
	double	result;

	result = arg1 + arg2;
	CHECKFLOATVAL(result, isinf(arg1) || isinf(arg2), true);

	PG_RETURN_FLOAT8(result);
}

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

/*
 * Misc functions
 */
Datum
pgstrom_cash_mul_flt2(PG_FUNCTION_ARGS)
{
	Cash		c = PG_GETARG_CASH(0);
	float8		f = fp16_to_fp64(PG_GETARG_FLOAT2(1));
	Cash		result;

	result = rint(c * f);
	PG_RETURN_CASH(result);
}

Datum
pgstrom_flt2_mul_cash(PG_FUNCTION_ARGS)
{
	float8		f = fp16_to_fp64(PG_GETARG_FLOAT2(0));
	Cash		c = PG_GETARG_CASH(1);
	Cash		result;

	result = rint(f * c);
	PG_RETURN_CASH(result);
}

Datum
pgstrom_cash_div_flt2(PG_FUNCTION_ARGS)
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

Datum
pgstrom_float8_as_int8(PG_FUNCTION_ARGS)
{
	float8	fval = PG_GETARG_FLOAT8(0);

	PG_RETURN_INT64(double_as_long(fval));
}

Datum
pgstrom_float4_as_int4(PG_FUNCTION_ARGS)
{
	float4	fval = PG_GETARG_FLOAT4(0);

	PG_RETURN_INT32(float_as_int(fval));
}

Datum
pgstrom_float2_as_int2(PG_FUNCTION_ARGS)
{
	half_t	fval = PG_GETARG_FLOAT2(0);

	PG_RETURN_INT16(fval);	/* actually, half_t is unsigned short */
}

Datum
pgstrom_int8_as_float8(PG_FUNCTION_ARGS)
{
	int64	ival = PG_GETARG_INT64(0);

	PG_RETURN_FLOAT8(long_as_double(ival));
}

Datum
pgstrom_int4_as_float4(PG_FUNCTION_ARGS)
{
	int32	ival = PG_GETARG_INT32(0);

	PG_RETURN_FLOAT4(int_as_float(ival));
}

Datum
pgstrom_int2_as_float2(PG_FUNCTION_ARGS)
{
	int16	ival = PG_GETARG_INT16(0);

	PG_RETURN_FLOAT2(ival);	/* actually, half_t is unsigned short */
}

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

Datum
pgstrom_define_shell_type(PG_FUNCTION_ARGS)
{
	Name		type_name = PG_GETARG_NAME(0);
	Oid			type_oid = PG_GETARG_OID(1);
	Oid			type_namespace = PG_GETARG_OID(2);
	Relation	type_rel;
	TupleDesc	tupdesc;
	HeapTuple	tup;
	Datum		values[Natts_pg_type];
	bool		isnull[Natts_pg_type];

	if (!superuser())
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
				 errmsg("must be superuser to create a shell type")));
	/* see TypeShellMake */
	type_rel = table_open(TypeRelationId, RowExclusiveLock);
	tupdesc = RelationGetDescr(type_rel);

	memset(values, 0, sizeof(values));
	memset(isnull, 0, sizeof(isnull));
#if PG_VERSION_NUM >= 120000
	values[Anum_pg_type_oid-1] = type_oid;
#endif
	values[Anum_pg_type_typname-1] = NameGetDatum(type_name);
    values[Anum_pg_type_typnamespace-1] = ObjectIdGetDatum(type_namespace);
    values[Anum_pg_type_typowner-1] = ObjectIdGetDatum(GetUserId());
    values[Anum_pg_type_typlen-1] = Int16GetDatum(sizeof(int32));
    values[Anum_pg_type_typbyval-1] = BoolGetDatum(true);
    values[Anum_pg_type_typtype-1] = CharGetDatum(TYPTYPE_PSEUDO);
	values[Anum_pg_type_typcategory-1] =CharGetDatum(TYPCATEGORY_PSEUDOTYPE);
	values[Anum_pg_type_typispreferred-1] = BoolGetDatum(false);
	values[Anum_pg_type_typisdefined-1] = BoolGetDatum(false);
	values[Anum_pg_type_typdelim-1] = CharGetDatum(DEFAULT_TYPDELIM);
	values[Anum_pg_type_typrelid-1] = ObjectIdGetDatum(InvalidOid);
	values[Anum_pg_type_typelem-1] = ObjectIdGetDatum(InvalidOid);
	values[Anum_pg_type_typarray-1] = ObjectIdGetDatum(InvalidOid);
	values[Anum_pg_type_typinput-1] = ObjectIdGetDatum(F_SHELL_IN);
	values[Anum_pg_type_typoutput-1] = ObjectIdGetDatum(F_SHELL_OUT);
	values[Anum_pg_type_typreceive-1] = ObjectIdGetDatum(InvalidOid);
	values[Anum_pg_type_typsend-1] = ObjectIdGetDatum(InvalidOid);
	values[Anum_pg_type_typmodin-1] = ObjectIdGetDatum(InvalidOid);
	values[Anum_pg_type_typmodout-1] = ObjectIdGetDatum(InvalidOid);
	values[Anum_pg_type_typanalyze-1] = ObjectIdGetDatum(InvalidOid);
	values[Anum_pg_type_typalign-1] = CharGetDatum('i');
	values[Anum_pg_type_typstorage-1] = CharGetDatum('p');
	values[Anum_pg_type_typnotnull-1] = BoolGetDatum(false);
	values[Anum_pg_type_typbasetype-1] = ObjectIdGetDatum(InvalidOid);
	values[Anum_pg_type_typtypmod-1] = Int32GetDatum(-1);
	values[Anum_pg_type_typndims-1] = Int32GetDatum(0);
	values[Anum_pg_type_typcollation-1] = ObjectIdGetDatum(InvalidOid);
	isnull[Anum_pg_type_typdefaultbin-1] = true;
	isnull[Anum_pg_type_typdefault-1] = true;
	isnull[Anum_pg_type_typacl-1] = true;

	/* create a new type tuple, and insert */
	tup = heap_form_tuple(tupdesc, values, isnull);
#if PG_VERSION_NUM < 120000
	HeapTupleSetOid(tup, type_oid);
#endif
	CatalogTupleInsert(type_rel, tup);

	/* create dependencies */
	GenerateTypeDependencies(tup,
							 type_rel,
							 NULL,
							 NULL,
							 0,
							 false,
							 false,
							 true,
							 false);
	/* Post creation hook for new shell type */
	InvokeObjectPostCreateHook(TypeRelationId, type_oid, 0);

	heap_freetuple(tup);
	table_close(type_rel, RowExclusiveLock);

	PG_RETURN_OID(type_oid);
}
