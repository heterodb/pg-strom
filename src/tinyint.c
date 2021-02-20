/*
 * src/tinyint.c
 *
 * 8bit-width integer data type support
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

#define DatumGetInt8(x)			((int8) (x))
#define PG_GETARG_INT8(x)		DatumGetInt8(PG_GETARG_DATUM(x))
#define PG_RETURN_INT8(x)		return Int8GetDatum(x)

/* type input/output */
Datum pgstrom_tinyint_in(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_out(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_send(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_recv(PG_FUNCTION_ARGS);

/* type cast functions */
Datum pgstrom_tinyint_to_int2(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_to_int4(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_to_int8(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_to_float4(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_to_float8(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_to_numeric(PG_FUNCTION_ARGS);
Datum pgstrom_int2_to_tinyint(PG_FUNCTION_ARGS);
Datum pgstrom_int4_to_tinyint(PG_FUNCTION_ARGS);
Datum pgstrom_int8_to_tinyint(PG_FUNCTION_ARGS);
Datum pgstrom_float4_to_tinyint(PG_FUNCTION_ARGS);
Datum pgstrom_float8_to_tinyint(PG_FUNCTION_ARGS);
Datum pgstrom_numeric_to_tinyint(PG_FUNCTION_ARGS);

/* comparison */
Datum pgstrom_tinyint_eq(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_ne(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_lt(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_le(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_gt(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_ge(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_larger(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_smaller(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_hash(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_cmp(PG_FUNCTION_ARGS);
Datum pgstrom_btint12_cmp(PG_FUNCTION_ARGS);
Datum pgstrom_btint14_cmp(PG_FUNCTION_ARGS);
Datum pgstrom_btint18_cmp(PG_FUNCTION_ARGS);
Datum pgstrom_btint21_cmp(PG_FUNCTION_ARGS);
Datum pgstrom_btint41_cmp(PG_FUNCTION_ARGS);
Datum pgstrom_btint81_cmp(PG_FUNCTION_ARGS);

Datum pgstrom_int12_eq(PG_FUNCTION_ARGS);
Datum pgstrom_int12_ne(PG_FUNCTION_ARGS);
Datum pgstrom_int12_lt(PG_FUNCTION_ARGS);
Datum pgstrom_int12_le(PG_FUNCTION_ARGS);
Datum pgstrom_int12_gt(PG_FUNCTION_ARGS);
Datum pgstrom_int12_ge(PG_FUNCTION_ARGS);

Datum pgstrom_int14_eq(PG_FUNCTION_ARGS);
Datum pgstrom_int14_ne(PG_FUNCTION_ARGS);
Datum pgstrom_int14_lt(PG_FUNCTION_ARGS);
Datum pgstrom_int14_le(PG_FUNCTION_ARGS);
Datum pgstrom_int14_gt(PG_FUNCTION_ARGS);
Datum pgstrom_int14_ge(PG_FUNCTION_ARGS);

Datum pgstrom_int18_eq(PG_FUNCTION_ARGS);
Datum pgstrom_int18_ne(PG_FUNCTION_ARGS);
Datum pgstrom_int18_lt(PG_FUNCTION_ARGS);
Datum pgstrom_int18_le(PG_FUNCTION_ARGS);
Datum pgstrom_int18_gt(PG_FUNCTION_ARGS);
Datum pgstrom_int18_ge(PG_FUNCTION_ARGS);

Datum pgstrom_int21_eq(PG_FUNCTION_ARGS);
Datum pgstrom_int21_ne(PG_FUNCTION_ARGS);
Datum pgstrom_int21_lt(PG_FUNCTION_ARGS);
Datum pgstrom_int21_le(PG_FUNCTION_ARGS);
Datum pgstrom_int21_gt(PG_FUNCTION_ARGS);
Datum pgstrom_int21_ge(PG_FUNCTION_ARGS);

Datum pgstrom_int41_eq(PG_FUNCTION_ARGS);
Datum pgstrom_int41_ne(PG_FUNCTION_ARGS);
Datum pgstrom_int41_lt(PG_FUNCTION_ARGS);
Datum pgstrom_int41_le(PG_FUNCTION_ARGS);
Datum pgstrom_int41_gt(PG_FUNCTION_ARGS);
Datum pgstrom_int41_ge(PG_FUNCTION_ARGS);

Datum pgstrom_int81_eq(PG_FUNCTION_ARGS);
Datum pgstrom_int81_ne(PG_FUNCTION_ARGS);
Datum pgstrom_int81_lt(PG_FUNCTION_ARGS);
Datum pgstrom_int81_le(PG_FUNCTION_ARGS);
Datum pgstrom_int81_gt(PG_FUNCTION_ARGS);
Datum pgstrom_int81_ge(PG_FUNCTION_ARGS);

/* unary operators */
Datum pgstrom_tinyint_up(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_um(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_abs(PG_FUNCTION_ARGS);

/* arithmetic operators */
Datum pgstrom_tinyint_pl(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_mi(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_mul(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_div(PG_FUNCTION_ARGS);

Datum pgstrom_int12_pl(PG_FUNCTION_ARGS);
Datum pgstrom_int12_mi(PG_FUNCTION_ARGS);
Datum pgstrom_int12_mul(PG_FUNCTION_ARGS);
Datum pgstrom_int12_div(PG_FUNCTION_ARGS);

Datum pgstrom_int14_pl(PG_FUNCTION_ARGS);
Datum pgstrom_int14_mi(PG_FUNCTION_ARGS);
Datum pgstrom_int14_mul(PG_FUNCTION_ARGS);
Datum pgstrom_int14_div(PG_FUNCTION_ARGS);

Datum pgstrom_int18_pl(PG_FUNCTION_ARGS);
Datum pgstrom_int18_mi(PG_FUNCTION_ARGS);
Datum pgstrom_int18_mul(PG_FUNCTION_ARGS);
Datum pgstrom_int18_div(PG_FUNCTION_ARGS);

Datum pgstrom_int21_pl(PG_FUNCTION_ARGS);
Datum pgstrom_int21_mi(PG_FUNCTION_ARGS);
Datum pgstrom_int21_mul(PG_FUNCTION_ARGS);
Datum pgstrom_int21_div(PG_FUNCTION_ARGS);

Datum pgstrom_int41_pl(PG_FUNCTION_ARGS);
Datum pgstrom_int41_mi(PG_FUNCTION_ARGS);
Datum pgstrom_int41_mul(PG_FUNCTION_ARGS);
Datum pgstrom_int41_div(PG_FUNCTION_ARGS);

Datum pgstrom_int81_pl(PG_FUNCTION_ARGS);
Datum pgstrom_int81_mi(PG_FUNCTION_ARGS);
Datum pgstrom_int81_mul(PG_FUNCTION_ARGS);
Datum pgstrom_int81_div(PG_FUNCTION_ARGS);

/* bit operations */
Datum pgstrom_tinyint_and(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_or(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_xor(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_not(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_shl(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_shr(PG_FUNCTION_ARGS);

/* misc functions */
Datum pgstrom_cash_mul_tinyint(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_mul_cash(PG_FUNCTION_ARGS);
Datum pgstrom_cash_div_tinyint(PG_FUNCTION_ARGS);

/* aggregate functions */
Datum pgstrom_tinyint_avg_accum(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_avg_accum_inv(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_var_accum(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_var_accum_inv(PG_FUNCTION_ARGS);
Datum pgstrom_tinyint_sum(PG_FUNCTION_ARGS);

/*
 * Type input / output functions
 */
Datum
pgstrom_tinyint_in(PG_FUNCTION_ARGS)
{
	char   *num = PG_GETARG_CSTRING(0);
	int32	ival = pg_strtoint32(num);

	if (ival < PG_INT8_MIN || ival > PG_INT8_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("value \"%s\" is out of range for type %s",
						num, "tinyint")));
	PG_RETURN_INT8(ival);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_in);

Datum
pgstrom_tinyint_out(PG_FUNCTION_ARGS)
{
	int8	ival = PG_GETARG_INT8(0);
	
	PG_RETURN_CSTRING(psprintf("%d", (int)ival));
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_out);

Datum
pgstrom_tinyint_send(PG_FUNCTION_ARGS)
{
	int8	ival = PG_GETARG_INT8(0);
	StringInfoData buf;

	pq_begintypsend(&buf);
	pq_sendint8(&buf, ival);
	PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_send);

Datum
pgstrom_tinyint_recv(PG_FUNCTION_ARGS)
{
	StringInfo	buf = (StringInfo) PG_GETARG_POINTER(0);

	PG_RETURN_INT8((int8) pq_getmsgint(buf, sizeof(int8)));
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_recv);

/*
 * Type cast functions
 */
Datum
pgstrom_tinyint_to_int2(PG_FUNCTION_ARGS)
{
	int8	ival = PG_GETARG_INT8(0);

	PG_RETURN_INT16(ival);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_to_int2);

Datum
pgstrom_tinyint_to_int4(PG_FUNCTION_ARGS)
{
	int8	ival = PG_GETARG_INT8(0);

	PG_RETURN_INT32(ival);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_to_int4);

Datum
pgstrom_tinyint_to_int8(PG_FUNCTION_ARGS)
{
	int8	ival = PG_GETARG_INT8(0);

	PG_RETURN_INT64(ival);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_to_int8);

Datum
pgstrom_tinyint_to_float4(PG_FUNCTION_ARGS)
{
	int8	ival = PG_GETARG_INT8(0);

	PG_RETURN_FLOAT4((float4) ival);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_to_float4);

Datum
pgstrom_tinyint_to_float8(PG_FUNCTION_ARGS)
{
	int8	ival = PG_GETARG_INT8(0);

	PG_RETURN_FLOAT8((float8) ival);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_to_float8);

Datum
pgstrom_tinyint_to_numeric(PG_FUNCTION_ARGS)
{
	int32	ival = (int32)PG_GETARG_INT8(0);

	return DirectFunctionCall1(int4_numeric, Int32GetDatum(ival));
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_to_numeric);

Datum
pgstrom_int2_to_tinyint(PG_FUNCTION_ARGS)
{
	int16	ival = PG_GETARG_INT16(0);

	if (ival < PG_INT8_MIN || ival > PG_INT8_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("tinyint out of range")));
	PG_RETURN_INT8(ival);
}
PG_FUNCTION_INFO_V1(pgstrom_int2_to_tinyint);

Datum
pgstrom_int4_to_tinyint(PG_FUNCTION_ARGS)
{
	int32	ival = PG_GETARG_INT32(0);

	if (ival < PG_INT8_MIN || ival > PG_INT8_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("tinyint out of range")));
	PG_RETURN_INT8(ival);
}
PG_FUNCTION_INFO_V1(pgstrom_int4_to_tinyint);

Datum
pgstrom_int8_to_tinyint(PG_FUNCTION_ARGS)
{
	int64	ival = PG_GETARG_INT64(0);

	if (ival < PG_INT8_MIN || ival > PG_INT8_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("tinyint out of range")));
	PG_RETURN_INT8(ival);
}
PG_FUNCTION_INFO_V1(pgstrom_int8_to_tinyint);

Datum
pgstrom_float4_to_tinyint(PG_FUNCTION_ARGS)
{
	float4	fval = PG_GETARG_FLOAT4(0);

	if (fval < (float4)PG_INT8_MIN || fval > (float4)PG_INT8_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("tinyint out of range")));
	PG_RETURN_INT8(fval);
}
PG_FUNCTION_INFO_V1(pgstrom_float4_to_tinyint);

Datum
pgstrom_float8_to_tinyint(PG_FUNCTION_ARGS)
{
	float8	fval = PG_GETARG_FLOAT8(0);

	if (fval < (float8)PG_INT8_MIN || fval > (float8)PG_INT8_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("tinyint out of range")));
	PG_RETURN_INT8((int8)fval);
}
PG_FUNCTION_INFO_V1(pgstrom_float8_to_tinyint);

Datum
pgstrom_numeric_to_tinyint(PG_FUNCTION_ARGS)
{
	int32	ival = DatumGetInt32(numeric_int4(fcinfo));

	if (ival < PG_INT8_MIN || ival > PG_INT8_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("tinyint out of range")));
	PG_RETURN_INT8((int8)ival);
}
PG_FUNCTION_INFO_V1(pgstrom_numeric_to_tinyint);

/*
 * Comparison functions
 */
Datum
pgstrom_tinyint_eq(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 == arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_eq);

Datum
pgstrom_tinyint_ne(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 != arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_ne);

Datum
pgstrom_tinyint_lt(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 < arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_lt);

Datum
pgstrom_tinyint_le(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 <= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_le);

Datum
pgstrom_tinyint_gt(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 > arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_gt);

Datum
pgstrom_tinyint_ge(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 >= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_ge);

Datum
pgstrom_tinyint_larger(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_INT8(arg1 > arg2 ? arg1 : arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_larger);

Datum
pgstrom_tinyint_smaller(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_INT8(arg1 < arg2 ? arg1 : arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_smaller);

Datum
pgstrom_tinyint_hash(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);

	/* Does it really make sense? */
	return hash_any((unsigned char *)&arg1, sizeof(int8));
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_hash);

Datum
pgstrom_tinyint_cmp(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	if (arg1 > arg2)
		PG_RETURN_INT32(1);
	else if (arg1 < arg2)
		PG_RETURN_INT32(-1);
	else
		PG_RETURN_INT32(0);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_cmp);

Datum
pgstrom_btint12_cmp(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int16	arg2 = PG_GETARG_INT16(1);

	if (arg1 > arg2)
		PG_RETURN_INT32(1);
	else if (arg1 < arg2)
		PG_RETURN_INT32(-1);
	else
		PG_RETURN_INT32(0);
}
PG_FUNCTION_INFO_V1(pgstrom_btint12_cmp);

Datum
pgstrom_btint14_cmp(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	if (arg1 > arg2)
		PG_RETURN_INT32(1);
	else if (arg1 < arg2)
		PG_RETURN_INT32(-1);
	else
		PG_RETURN_INT32(0);
}
PG_FUNCTION_INFO_V1(pgstrom_btint14_cmp);

Datum
pgstrom_btint18_cmp(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int64	arg2 = PG_GETARG_INT64(1);

	if (arg1 > arg2)
		PG_RETURN_INT32(1);
	else if (arg1 < arg2)
		PG_RETURN_INT32(-1);
	else
		PG_RETURN_INT32(0);
}
PG_FUNCTION_INFO_V1(pgstrom_btint18_cmp);

Datum
pgstrom_btint21_cmp(PG_FUNCTION_ARGS)
{
	int16	arg1 = PG_GETARG_INT16(0);
	int8	arg2 = PG_GETARG_INT8(1);

	if (arg1 > arg2)
		PG_RETURN_INT32(1);
	else if (arg1 < arg2)
		PG_RETURN_INT32(-1);
	else
		PG_RETURN_INT32(0);
}
PG_FUNCTION_INFO_V1(pgstrom_btint21_cmp);

Datum
pgstrom_btint41_cmp(PG_FUNCTION_ARGS)
{
	int32	arg1 = PG_GETARG_INT32(0);
	int8	arg2 = PG_GETARG_INT8(1);

	if (arg1 > arg2)
		PG_RETURN_INT32(1);
	else if (arg1 < arg2)
		PG_RETURN_INT32(-1);
	else
		PG_RETURN_INT32(0);
}
PG_FUNCTION_INFO_V1(pgstrom_btint41_cmp);

Datum
pgstrom_btint81_cmp(PG_FUNCTION_ARGS)
{
	int64	arg1 = PG_GETARG_INT64(0);
	int8	arg2 = PG_GETARG_INT8(1);

	if (arg1 > arg2)
		PG_RETURN_INT32(1);
	else if (arg1 < arg2)
		PG_RETURN_INT32(-1);
	else
		PG_RETURN_INT32(0);
}
PG_FUNCTION_INFO_V1(pgstrom_btint81_cmp);

Datum
pgstrom_int12_eq(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int16	arg2 = PG_GETARG_INT16(1);

	PG_RETURN_BOOL(arg1 == arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int12_eq);

Datum
pgstrom_int12_ne(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int16	arg2 = PG_GETARG_INT16(1);

	PG_RETURN_BOOL(arg1 != arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int12_ne);

Datum
pgstrom_int12_lt(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int16	arg2 = PG_GETARG_INT16(1);

	PG_RETURN_BOOL(arg1 < arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int12_lt);

Datum
pgstrom_int12_le(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int16	arg2 = PG_GETARG_INT16(1);

	PG_RETURN_BOOL(arg1 <= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int12_le);

Datum
pgstrom_int12_gt(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int16	arg2 = PG_GETARG_INT16(1);

	PG_RETURN_BOOL(arg1 > arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int12_gt);

Datum
pgstrom_int12_ge(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int16	arg2 = PG_GETARG_INT16(1);

	PG_RETURN_BOOL(arg1 >= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int12_ge);

Datum
pgstrom_int14_eq(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	PG_RETURN_BOOL(arg1 == arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int14_eq);

Datum
pgstrom_int14_ne(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	PG_RETURN_BOOL(arg1 != arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int14_ne);

Datum
pgstrom_int14_lt(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	PG_RETURN_BOOL(arg1 < arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int14_lt);

Datum
pgstrom_int14_le(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	PG_RETURN_BOOL(arg1 <= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int14_le);

Datum
pgstrom_int14_gt(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	PG_RETURN_BOOL(arg1 > arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int14_gt);

Datum
pgstrom_int14_ge(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	PG_RETURN_BOOL(arg1 >= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int14_ge);

Datum
pgstrom_int18_eq(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int64	arg2 = PG_GETARG_INT64(1);

	PG_RETURN_BOOL(arg1 == arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int18_eq);

Datum
pgstrom_int18_ne(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int64	arg2 = PG_GETARG_INT64(1);

	PG_RETURN_BOOL(arg1 != arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int18_ne);

Datum
pgstrom_int18_lt(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int64	arg2 = PG_GETARG_INT64(1);

	PG_RETURN_BOOL(arg1 < arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int18_lt);

Datum
pgstrom_int18_le(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int64	arg2 = PG_GETARG_INT64(1);

	PG_RETURN_BOOL(arg1 <= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int18_le);

Datum
pgstrom_int18_gt(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int64	arg2 = PG_GETARG_INT64(1);

	PG_RETURN_BOOL(arg1 > arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int18_gt);

Datum
pgstrom_int18_ge(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int64	arg2 = PG_GETARG_INT64(1);

	PG_RETURN_BOOL(arg1 >= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int18_ge);

Datum
pgstrom_int21_eq(PG_FUNCTION_ARGS)
{
	int16	arg1 = PG_GETARG_INT16(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 == arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int21_eq);

Datum
pgstrom_int21_ne(PG_FUNCTION_ARGS)
{
	int16	arg1 = PG_GETARG_INT16(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 != arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int21_ne);

Datum
pgstrom_int21_lt(PG_FUNCTION_ARGS)
{
	int16	arg1 = PG_GETARG_INT16(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 < arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int21_lt);

Datum
pgstrom_int21_le(PG_FUNCTION_ARGS)
{
	int16	arg1 = PG_GETARG_INT16(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 <= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int21_le);

Datum
pgstrom_int21_gt(PG_FUNCTION_ARGS)
{
	int16	arg1 = PG_GETARG_INT16(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 > arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int21_gt);

Datum
pgstrom_int21_ge(PG_FUNCTION_ARGS)
{
	int16	arg1 = PG_GETARG_INT16(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 >= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int21_ge);

Datum
pgstrom_int41_eq(PG_FUNCTION_ARGS)
{
	int32	arg1 = PG_GETARG_INT32(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 == arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int41_eq);

Datum
pgstrom_int41_ne(PG_FUNCTION_ARGS)
{
	int32	arg1 = PG_GETARG_INT32(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 != arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int41_ne);

Datum
pgstrom_int41_lt(PG_FUNCTION_ARGS)
{
	int32	arg1 = PG_GETARG_INT32(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 < arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int41_lt);

Datum
pgstrom_int41_le(PG_FUNCTION_ARGS)
{
	int32	arg1 = PG_GETARG_INT32(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 <= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int41_le);

Datum
pgstrom_int41_gt(PG_FUNCTION_ARGS)
{
	int32	arg1 = PG_GETARG_INT32(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 > arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int41_gt);

Datum
pgstrom_int41_ge(PG_FUNCTION_ARGS)
{
	int32	arg1 = PG_GETARG_INT32(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 >= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int41_ge);

Datum
pgstrom_int81_eq(PG_FUNCTION_ARGS)
{
	int64	arg1 = PG_GETARG_INT64(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 == arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int81_eq);

Datum
pgstrom_int81_ne(PG_FUNCTION_ARGS)
{
	int64	arg1 = PG_GETARG_INT64(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 != arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int81_ne);

Datum
pgstrom_int81_lt(PG_FUNCTION_ARGS)
{
	int64	arg1 = PG_GETARG_INT64(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 < arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int81_lt);

Datum
pgstrom_int81_le(PG_FUNCTION_ARGS)
{
	int64	arg1 = PG_GETARG_INT64(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 <= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int81_le);

Datum
pgstrom_int81_gt(PG_FUNCTION_ARGS)
{
	int64	arg1 = PG_GETARG_INT64(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 > arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int81_gt);

Datum
pgstrom_int81_ge(PG_FUNCTION_ARGS)
{
	int64	arg1 = PG_GETARG_INT64(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 >= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int81_ge);

/* unary operators */
Datum
pgstrom_tinyint_up(PG_FUNCTION_ARGS)
{
	int8	arg = PG_GETARG_INT8(0);

	PG_RETURN_INT8(arg);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_up);

Datum
pgstrom_tinyint_um(PG_FUNCTION_ARGS)
{
	int8	arg = PG_GETARG_INT8(0);

	PG_RETURN_INT8(-arg);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_um);

Datum
pgstrom_tinyint_abs(PG_FUNCTION_ARGS)
{
	int8	arg = PG_GETARG_INT8(0);

	PG_RETURN_INT8(arg < 0 ? -arg : arg);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_abs);

/*
 * Arithmetic operators
 */
Datum
pgstrom_tinyint_pl(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);
	int32	retval = (int32)arg1 + (int32)arg2;

	if (retval < PG_INT8_MIN || retval > PG_INT8_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("tinyint out of range")));
	PG_RETURN_INT8(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_pl);

Datum
pgstrom_tinyint_mi(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);
	int32	retval = (int32)arg1 - (int32)arg2;

	if (retval < PG_INT8_MIN || retval > PG_INT8_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("tinyint out of range")));
	PG_RETURN_INT8(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_mi);

Datum
pgstrom_tinyint_mul(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);
	int32	retval = (int32)arg1 * (int32)arg2;

	if (retval < PG_INT8_MIN || retval > PG_INT8_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("tinyint out of range")));
	PG_RETURN_INT8(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_mul);

Datum
pgstrom_tinyint_div(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);
	int8	retval;

	if (arg2 == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));
	if (arg2 == -1)
	{
		if (arg1 == PG_INT8_MAX)
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("tinyint out of range")));
		retval = -arg1;
	}
	else
	{
		retval = arg1 / arg2;
	}
	PG_RETURN_INT8(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_div);

Datum
pgstrom_int12_pl(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int16	arg2 = PG_GETARG_INT16(1);
	int16	retval;

	if (pg_add_s16_overflow((int16)arg1, arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("smallint out of range")));
	PG_RETURN_INT16(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int12_pl);

Datum
pgstrom_int12_mi(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int16	arg2 = PG_GETARG_INT16(1);
	int16	retval;

	if (pg_sub_s16_overflow((int16)arg1, arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("smallint out of range")));
	PG_RETURN_INT16(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int12_mi);

Datum
pgstrom_int12_mul(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int16	arg2 = PG_GETARG_INT16(1);
	int16	retval;

	if (pg_mul_s16_overflow((int16)arg1, arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("smallint out of range")));
	PG_RETURN_INT16(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int12_mul);

Datum
pgstrom_int12_div(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int16	arg2 = PG_GETARG_INT8(1);

	if (arg2 == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));
	/* no overflow is possible */
	PG_RETURN_INT16((int16)arg1 / arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int12_div);

Datum
pgstrom_int14_pl(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);
	int32	retval;

	if (pg_add_s32_overflow((int32)arg1, arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("integer out of range")));
	PG_RETURN_INT32(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int14_pl);

Datum
pgstrom_int14_mi(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);
	int32	retval;

	if (pg_sub_s32_overflow((int32)arg1, arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("integer out of range")));
	PG_RETURN_INT32(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int14_mi);

Datum
pgstrom_int14_mul(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);
	int32	retval;

	if (pg_mul_s32_overflow((int32)arg1, arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("integer out of range")));
	PG_RETURN_INT32(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int14_mul);

Datum
pgstrom_int14_div(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	if (arg2 == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));
	/* no overflow is possible */
	PG_RETURN_INT32((int32)arg1 / arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int14_div);

Datum
pgstrom_int18_pl(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int64	arg2 = PG_GETARG_INT64(1);
	int64	retval;

	if (pg_add_s64_overflow((int64)arg1, arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("bigint out of range")));
	PG_RETURN_INT64(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int18_pl);

Datum
pgstrom_int18_mi(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int64	arg2 = PG_GETARG_INT64(1);
	int64	retval;

	if (pg_sub_s64_overflow((int64)arg1, arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("bigint out of range")));
	PG_RETURN_INT64(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int18_mi);

Datum
pgstrom_int18_mul(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int64	arg2 = PG_GETARG_INT64(1);
	int64	retval;

	if (pg_mul_s64_overflow((int64)arg1, arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("bigint out of range")));
	PG_RETURN_INT64(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int18_mul);

Datum
pgstrom_int18_div(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int64	arg2 = PG_GETARG_INT64(1);

	if (arg2 == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));
	/* no overflow is possible */
	PG_RETURN_INT64((int64)arg1 / arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int18_div);

Datum
pgstrom_int21_pl(PG_FUNCTION_ARGS)
{
	int16	arg1 = PG_GETARG_INT16(0);
	int8	arg2 = PG_GETARG_INT8(1);
	int16	retval;

	if (pg_add_s16_overflow(arg1, (int16)arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("smallint out of range")));
	PG_RETURN_INT16(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int21_pl);

Datum
pgstrom_int21_mi(PG_FUNCTION_ARGS)
{
	int16	arg1 = PG_GETARG_INT16(0);
	int8	arg2 = PG_GETARG_INT8(1);
	int16	retval;

	if (pg_sub_s16_overflow(arg1, (int16)arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("smallint out of range")));
	PG_RETURN_INT16(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int21_mi);

Datum
pgstrom_int21_mul(PG_FUNCTION_ARGS)
{
	int16	arg1 = PG_GETARG_INT16(0);
	int8	arg2 = PG_GETARG_INT8(1);
	int16	retval;

	if (pg_mul_s16_overflow(arg1, (int16)arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("smallint out of range")));
	PG_RETURN_INT16(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int21_mul);

Datum
pgstrom_int21_div(PG_FUNCTION_ARGS)
{
	int16	arg1 = PG_GETARG_INT16(0);
	int8	arg2 = PG_GETARG_INT8(1);
	int16	retval;

	if (arg2 == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));

	if (arg2 == -1)
	{
		if (arg1 == PG_INT16_MAX)
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("smallint out of range")));
		retval = -arg1;
	}
	else
	{
		retval = arg1 / arg2;
	}
	PG_RETURN_INT16(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int21_div);

Datum
pgstrom_int41_pl(PG_FUNCTION_ARGS)
{
	int32	arg1 = PG_GETARG_INT32(0);
	int8	arg2 = PG_GETARG_INT8(1);
	int32	retval;

	if (pg_add_s32_overflow(arg1, (int32)arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("integer out of range")));
	PG_RETURN_INT32(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int41_pl);

Datum
pgstrom_int41_mi(PG_FUNCTION_ARGS)
{
	int32	arg1 = PG_GETARG_INT32(0);
	int8	arg2 = PG_GETARG_INT8(1);
	int32	retval;

	if (pg_sub_s32_overflow(arg1, (int32)arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("integer out of range")));
	PG_RETURN_INT32(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int41_mi);

Datum
pgstrom_int41_mul(PG_FUNCTION_ARGS)
{
	int32	arg1 = PG_GETARG_INT32(0);
	int8	arg2 = PG_GETARG_INT8(1);
	int32	retval;

	if (pg_mul_s32_overflow(arg1, (int32)arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("integer out of range")));
	PG_RETURN_INT32(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int41_mul);

Datum
pgstrom_int41_div(PG_FUNCTION_ARGS)
{
	int32	arg1 = PG_GETARG_INT32(0);
	int8	arg2 = PG_GETARG_INT8(1);
	int32	retval;

	if (arg2 == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));
	if (arg2 == -1)
	{
		if (arg1 == PG_INT32_MAX)
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("integer out of range")));
		retval = -arg1;
	}
	else
	{
		retval = arg1 / arg2;
	}
	PG_RETURN_INT32(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int41_div);

Datum
pgstrom_int81_pl(PG_FUNCTION_ARGS)
{
	int64	arg1 = PG_GETARG_INT64(0);
	int8	arg2 = PG_GETARG_INT8(1);
	int64	retval;

	if (pg_add_s64_overflow(arg1, (int64)arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("bigint out of range")));
	PG_RETURN_INT64(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int81_pl);

Datum
pgstrom_int81_mi(PG_FUNCTION_ARGS)
{
	int64	arg1 = PG_GETARG_INT64(0);
	int8	arg2 = PG_GETARG_INT8(1);
	int64	retval;

	if (pg_sub_s64_overflow(arg1, (int64)arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("bigint out of range")));
	PG_RETURN_INT64(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int81_mi);

Datum
pgstrom_int81_mul(PG_FUNCTION_ARGS)
{
	int64	arg1 = PG_GETARG_INT64(0);
	int8	arg2 = PG_GETARG_INT8(1);
	int64	retval;

	if (pg_mul_s64_overflow(arg1, (int64)arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("bigint out of range")));
	PG_RETURN_INT64(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int81_mul);

Datum
pgstrom_int81_div(PG_FUNCTION_ARGS)
{
	int64	arg1 = PG_GETARG_INT64(0);
	int8	arg2 = PG_GETARG_INT8(1);
	int64	retval;

	if (arg2 == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));
	if (arg2 == -1)
	{
		if (arg1 == PG_INT64_MAX)
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("bigint out of range")));
		retval = -arg1;
	}
	else
	{
		retval = arg1 / arg2;
	}
	PG_RETURN_INT64(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_int81_div);

/*
 * Bit operations
 */
Datum
pgstrom_tinyint_and(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_INT8(arg1 & arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_and);

Datum
pgstrom_tinyint_or(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_INT8(arg1 | arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_or);

Datum
pgstrom_tinyint_xor(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_INT8(arg1 ^ arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_xor);

Datum
pgstrom_tinyint_not(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);

	PG_RETURN_INT8(~arg1);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_not);

Datum
pgstrom_tinyint_shl(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	PG_RETURN_INT8(arg1 << arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_shl);

Datum
pgstrom_tinyint_shr(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	PG_RETURN_INT8(arg1 >> arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_shr);

/*
 * misc functions
 */
Datum
pgstrom_cash_mul_tinyint(PG_FUNCTION_ARGS)
{
	Cash	arg1 = PG_GETARG_CASH(0);
	int8	arg2 = PG_GETARG_INT8(1);
	Cash	retval;

	if (pg_mul_s64_overflow(arg1, (int64)arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("money out of range")));
	PG_RETURN_CASH(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_cash_mul_tinyint);

Datum
pgstrom_tinyint_mul_cash(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	Cash	arg2 = PG_GETARG_CASH(1);
	Cash	retval;

	if (pg_mul_s64_overflow((int64)arg1, arg2, &retval))
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("money out of range")));
	PG_RETURN_CASH(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_mul_cash);

Datum
pgstrom_cash_div_tinyint(PG_FUNCTION_ARGS)
{
	Cash	arg1 = PG_GETARG_CASH(0);
	int8	arg2 = PG_GETARG_INT8(1);
	Cash	retval;

	if (arg2 == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));
	if (arg2 == -1)
	{
		if (arg1 == PG_INT64_MAX)
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("money out of range")));
		retval = -arg1;
	}
	else
	{
		retval = arg1 / arg2;
	}
	PG_RETURN_CASH(retval);
}
PG_FUNCTION_INFO_V1(pgstrom_cash_div_tinyint);

/* aggregate functions */
Datum
pgstrom_tinyint_sum(PG_FUNCTION_ARGS)
{
	int64	newval;

	if (PG_ARGISNULL(0))
	{
		if (PG_ARGISNULL(1))
			PG_RETURN_NULL();	/* still keep NULL */
		newval = (int64)PG_GETARG_INT8(1);
	}
	else
	{
		newval = PG_GETARG_INT64(0);
		if (!PG_ARGISNULL(1))
			newval += (int64)PG_GETARG_INT8(1);
	}
	PG_RETURN_INT64(newval);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_sum);

Datum
pgstrom_tinyint_avg_accum(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	int32		ival = (int32)PG_GETARG_INT8(1);
	int64	   *transvalues;

	if (ARR_NDIM(transarray) != 1 ||
		ARR_DIMS(transarray)[0] != 2 ||
		ARR_HASNULL(transarray) ||
		ARR_ELEMTYPE(transarray) != INT8OID)
		elog(ERROR, "expected 2-element int8 array");
	/* see definition of Int8TransTypeData */
	transvalues = (int64 *) ARR_DATA_PTR(transarray);

	/*
	 * If we're invoked as an aggregate, we can cheat and modify our first
	 * parameter in-place to reduce palloc overhead. Otherwise we construct a
	 * new array with the updated transition data and return it.
	 */
	if (AggCheckCallContext(fcinfo, NULL))
	{
		transvalues[0] += 1;
		transvalues[1] += ival;
		PG_RETURN_ARRAYTYPE_P(transarray);
    }
	else
	{
		Datum	transdatums[3];
		ArrayType *result;

		transdatums[0] = Int64GetDatum(transvalues[0] + 1);
		transdatums[1] = Int64GetDatum(transvalues[1] + ival);
		result = construct_array(transdatums,
								 2,
								 INT8OID,
								 8, FLOAT8PASSBYVAL, 'd');
		PG_RETURN_ARRAYTYPE_P(result);
	}
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_avg_accum);

Datum
pgstrom_tinyint_avg_accum_inv(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray;
	int32		ival = (int32)PG_GETARG_INT8(1);
	int64	   *transvalues;

	/*
	 * If we're invoked as an aggregate, we can cheat and modify our first
	 * parameter in-place to reduce palloc overhead. Otherwise we construct a
	 * new array with the updated transition data and return it.
	 */
	if (AggCheckCallContext(fcinfo, NULL))
		transarray = PG_GETARG_ARRAYTYPE_P(0);
	else
		transarray = PG_GETARG_ARRAYTYPE_P_COPY(0);

	if (ARR_NDIM(transarray) != 1 ||
		ARR_DIMS(transarray)[0] != 2 ||
		ARR_HASNULL(transarray) ||
		ARR_ELEMTYPE(transarray) != INT8OID)
		elog(ERROR, "expected 2-element int8 array");
	/* see definition of Int8TransTypeData */
	transvalues = (int64 *) ARR_DATA_PTR(transarray);
	transvalues[0] -= 1;
	transvalues[1] -= ival;
	PG_RETURN_ARRAYTYPE_P(transarray);
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_avg_accum_inv);

Datum
pgstrom_tinyint_var_accum(PG_FUNCTION_ARGS)
{
	int32		ival = (int32)PG_GETARG_INT8(1);

	return DirectFunctionCall2(int4_accum,
							   PG_GETARG_DATUM(0),
							   Int32GetDatum(ival));
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_var_accum);

Datum
pgstrom_tinyint_var_accum_inv(PG_FUNCTION_ARGS)
{
	int32		ival = (int32)PG_GETARG_INT8(1);

	return DirectFunctionCall2(int4_accum_inv,
							   PG_GETARG_DATUM(0),
							   Int32GetDatum(ival));
}
PG_FUNCTION_INFO_V1(pgstrom_tinyint_var_accum_inv);
