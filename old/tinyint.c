/*
 * tinyint.c
 *
 * 8bit-width integer data type support
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

#define DatumGetInt8(x)			((int8) (x))
#define PG_GETARG_INT8(x)		DatumGetInt8(PG_GETARG_DATUM(x))
#define PG_RETURN_INT8(x)		return Int8GetDatum(x)

/* type input/output */
Datum pgstrom_int1in(PG_FUNCTION_ARGS);
Datum pgstrom_int1out(PG_FUNCTION_ARGS);
Datum pgstrom_int1send(PG_FUNCTION_ARGS);
Datum pgstrom_int1recv(PG_FUNCTION_ARGS);

/* type cast functions */
Datum pgstrom_int1_to_int2(PG_FUNCTION_ARGS);
Datum pgstrom_int1_to_int4(PG_FUNCTION_ARGS);
Datum pgstrom_int1_to_int8(PG_FUNCTION_ARGS);
Datum pgstrom_int1_to_float4(PG_FUNCTION_ARGS);
Datum pgstrom_int1_to_float8(PG_FUNCTION_ARGS);
Datum pgstrom_int1_to_numeric(PG_FUNCTION_ARGS);
Datum pgstrom_int2_to_int1(PG_FUNCTION_ARGS);
Datum pgstrom_int4_to_int1(PG_FUNCTION_ARGS);
Datum pgstrom_int8_to_int1(PG_FUNCTION_ARGS);
Datum pgstrom_float4_to_int1(PG_FUNCTION_ARGS);
Datum pgstrom_float8_to_int1(PG_FUNCTION_ARGS);
Datum pgstrom_numeric_to_int1(PG_FUNCTION_ARGS);

/* comparison */
Datum pgstrom_int1eq(PG_FUNCTION_ARGS);
Datum pgstrom_int1ne(PG_FUNCTION_ARGS);
Datum pgstrom_int1lt(PG_FUNCTION_ARGS);
Datum pgstrom_int1le(PG_FUNCTION_ARGS);
Datum pgstrom_int1gt(PG_FUNCTION_ARGS);
Datum pgstrom_int1ge(PG_FUNCTION_ARGS);
Datum pgstrom_int1larger(PG_FUNCTION_ARGS);
Datum pgstrom_int1smaller(PG_FUNCTION_ARGS);

Datum pgstrom_int12eq(PG_FUNCTION_ARGS);
Datum pgstrom_int12ne(PG_FUNCTION_ARGS);
Datum pgstrom_int12lt(PG_FUNCTION_ARGS);
Datum pgstrom_int12le(PG_FUNCTION_ARGS);
Datum pgstrom_int12gt(PG_FUNCTION_ARGS);
Datum pgstrom_int12ge(PG_FUNCTION_ARGS);

Datum pgstrom_int14eq(PG_FUNCTION_ARGS);
Datum pgstrom_int14ne(PG_FUNCTION_ARGS);
Datum pgstrom_int14lt(PG_FUNCTION_ARGS);
Datum pgstrom_int14le(PG_FUNCTION_ARGS);
Datum pgstrom_int14gt(PG_FUNCTION_ARGS);
Datum pgstrom_int14ge(PG_FUNCTION_ARGS);

Datum pgstrom_int18eq(PG_FUNCTION_ARGS);
Datum pgstrom_int18ne(PG_FUNCTION_ARGS);
Datum pgstrom_int18lt(PG_FUNCTION_ARGS);
Datum pgstrom_int18le(PG_FUNCTION_ARGS);
Datum pgstrom_int18gt(PG_FUNCTION_ARGS);
Datum pgstrom_int18ge(PG_FUNCTION_ARGS);

Datum pgstrom_int21eq(PG_FUNCTION_ARGS);
Datum pgstrom_int21ne(PG_FUNCTION_ARGS);
Datum pgstrom_int21lt(PG_FUNCTION_ARGS);
Datum pgstrom_int21le(PG_FUNCTION_ARGS);
Datum pgstrom_int21gt(PG_FUNCTION_ARGS);
Datum pgstrom_int21ge(PG_FUNCTION_ARGS);

Datum pgstrom_int41eq(PG_FUNCTION_ARGS);
Datum pgstrom_int41ne(PG_FUNCTION_ARGS);
Datum pgstrom_int41lt(PG_FUNCTION_ARGS);
Datum pgstrom_int41le(PG_FUNCTION_ARGS);
Datum pgstrom_int41gt(PG_FUNCTION_ARGS);
Datum pgstrom_int41ge(PG_FUNCTION_ARGS);

Datum pgstrom_int81eq(PG_FUNCTION_ARGS);
Datum pgstrom_int81ne(PG_FUNCTION_ARGS);
Datum pgstrom_int81lt(PG_FUNCTION_ARGS);
Datum pgstrom_int81le(PG_FUNCTION_ARGS);
Datum pgstrom_int81gt(PG_FUNCTION_ARGS);
Datum pgstrom_int81ge(PG_FUNCTION_ARGS);

Datum pgstrom_int1hash(PG_FUNCTION_ARGS);
Datum pgstrom_btint1cmp(PG_FUNCTION_ARGS);
Datum pgstrom_btint12cmp(PG_FUNCTION_ARGS);
Datum pgstrom_btint14cmp(PG_FUNCTION_ARGS);
Datum pgstrom_btint18cmp(PG_FUNCTION_ARGS);
Datum pgstrom_btint21cmp(PG_FUNCTION_ARGS);
Datum pgstrom_btint41cmp(PG_FUNCTION_ARGS);
Datum pgstrom_btint81cmp(PG_FUNCTION_ARGS);

/* unary operators */
Datum pgstrom_int1up(PG_FUNCTION_ARGS);
Datum pgstrom_int1um(PG_FUNCTION_ARGS);
Datum pgstrom_int1abs(PG_FUNCTION_ARGS);

/* arithmetic operators */
Datum pgstrom_int1pl(PG_FUNCTION_ARGS);
Datum pgstrom_int1mi(PG_FUNCTION_ARGS);
Datum pgstrom_int1mul(PG_FUNCTION_ARGS);
Datum pgstrom_int1div(PG_FUNCTION_ARGS);
Datum pgstrom_int1mod(PG_FUNCTION_ARGS);

Datum pgstrom_int12pl(PG_FUNCTION_ARGS);
Datum pgstrom_int12mi(PG_FUNCTION_ARGS);
Datum pgstrom_int12mul(PG_FUNCTION_ARGS);
Datum pgstrom_int12div(PG_FUNCTION_ARGS);

Datum pgstrom_int14pl(PG_FUNCTION_ARGS);
Datum pgstrom_int14mi(PG_FUNCTION_ARGS);
Datum pgstrom_int14mul(PG_FUNCTION_ARGS);
Datum pgstrom_int14div(PG_FUNCTION_ARGS);

Datum pgstrom_int18pl(PG_FUNCTION_ARGS);
Datum pgstrom_int18mi(PG_FUNCTION_ARGS);
Datum pgstrom_int18mul(PG_FUNCTION_ARGS);
Datum pgstrom_int18div(PG_FUNCTION_ARGS);

Datum pgstrom_int21pl(PG_FUNCTION_ARGS);
Datum pgstrom_int21mi(PG_FUNCTION_ARGS);
Datum pgstrom_int21mul(PG_FUNCTION_ARGS);
Datum pgstrom_int21div(PG_FUNCTION_ARGS);

Datum pgstrom_int41pl(PG_FUNCTION_ARGS);
Datum pgstrom_int41mi(PG_FUNCTION_ARGS);
Datum pgstrom_int41mul(PG_FUNCTION_ARGS);
Datum pgstrom_int41div(PG_FUNCTION_ARGS);

Datum pgstrom_int81pl(PG_FUNCTION_ARGS);
Datum pgstrom_int81mi(PG_FUNCTION_ARGS);
Datum pgstrom_int81mul(PG_FUNCTION_ARGS);
Datum pgstrom_int81div(PG_FUNCTION_ARGS);

/* bit operations */
Datum pgstrom_int1and(PG_FUNCTION_ARGS);
Datum pgstrom_int1or(PG_FUNCTION_ARGS);
Datum pgstrom_int1xor(PG_FUNCTION_ARGS);
Datum pgstrom_int1not(PG_FUNCTION_ARGS);
Datum pgstrom_int1shl(PG_FUNCTION_ARGS);
Datum pgstrom_int1shr(PG_FUNCTION_ARGS);

/* misc functions */
Datum pgstrom_cash_mul_int1(PG_FUNCTION_ARGS);
Datum pgstrom_int1_mul_cash(PG_FUNCTION_ARGS);
Datum pgstrom_cash_div_int1(PG_FUNCTION_ARGS);

/* aggregate functions */
Datum pgstrom_int1_sum(PG_FUNCTION_ARGS);
Datum pgstrom_int1_avg_accum(PG_FUNCTION_ARGS);
Datum pgstrom_int1_avg_accum_inv(PG_FUNCTION_ARGS);
Datum pgstrom_int1_var_accum(PG_FUNCTION_ARGS);
Datum pgstrom_int1_var_accum_inv(PG_FUNCTION_ARGS);

/*
 * Type input / output functions
 */
Datum
pgstrom_int1in(PG_FUNCTION_ARGS)
{
	char   *num = PG_GETARG_CSTRING(0);
	char   *end;
	long	ival;

	if (!num)
		elog(ERROR, "NULL pointer");
	ival = strtol(num, &end, 10);
	if (*num == '\0' || *end != '\0')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("invalid input syntax for tinyint: \"%s\"", num)));
	if (ival < PG_INT8_MIN || ival > PG_INT8_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("value \"%s\" is out of range for type %s",
						num, "tinyint")));
	PG_RETURN_INT8(ival);
}
PG_FUNCTION_INFO_V1(pgstrom_int1in);

Datum
pgstrom_int1out(PG_FUNCTION_ARGS)
{
	int8	ival = PG_GETARG_INT8(0);
	
	PG_RETURN_CSTRING(psprintf("%d", (int)ival));
}
PG_FUNCTION_INFO_V1(pgstrom_int1out);

Datum
pgstrom_int1send(PG_FUNCTION_ARGS)
{
	int8	ival = PG_GETARG_INT8(0);
	StringInfoData buf;

	pq_begintypsend(&buf);
	pq_sendint8(&buf, ival);
	PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
}
PG_FUNCTION_INFO_V1(pgstrom_int1send);

Datum
pgstrom_int1recv(PG_FUNCTION_ARGS)
{
	StringInfo	buf = (StringInfo) PG_GETARG_POINTER(0);

	PG_RETURN_INT8((int8) pq_getmsgint(buf, sizeof(int8)));
}
PG_FUNCTION_INFO_V1(pgstrom_int1recv);

/*
 * Type cast functions
 */
Datum
pgstrom_int1_to_int2(PG_FUNCTION_ARGS)
{
	int8	ival = PG_GETARG_INT8(0);

	PG_RETURN_INT16(ival);
}
PG_FUNCTION_INFO_V1(pgstrom_int1_to_int2);

Datum
pgstrom_int1_to_int4(PG_FUNCTION_ARGS)
{
	int8	ival = PG_GETARG_INT8(0);

	PG_RETURN_INT32(ival);
}
PG_FUNCTION_INFO_V1(pgstrom_int1_to_int4);

Datum
pgstrom_int1_to_int8(PG_FUNCTION_ARGS)
{
	int8	ival = PG_GETARG_INT8(0);

	PG_RETURN_INT64(ival);
}
PG_FUNCTION_INFO_V1(pgstrom_int1_to_int8);

Datum
pgstrom_int1_to_float4(PG_FUNCTION_ARGS)
{
	int8	ival = PG_GETARG_INT8(0);

	PG_RETURN_FLOAT4((float4) ival);
}
PG_FUNCTION_INFO_V1(pgstrom_int1_to_float4);

Datum
pgstrom_int1_to_float8(PG_FUNCTION_ARGS)
{
	int8	ival = PG_GETARG_INT8(0);

	PG_RETURN_FLOAT8((float8) ival);
}
PG_FUNCTION_INFO_V1(pgstrom_int1_to_float8);

Datum
pgstrom_int1_to_numeric(PG_FUNCTION_ARGS)
{
	int32	ival = (int32)PG_GETARG_INT8(0);

	return DirectFunctionCall1(int4_numeric, Int32GetDatum(ival));
}
PG_FUNCTION_INFO_V1(pgstrom_int1_to_numeric);

Datum
pgstrom_int2_to_int1(PG_FUNCTION_ARGS)
{
	int16	ival = PG_GETARG_INT16(0);

	if (ival < PG_INT8_MIN || ival > PG_INT8_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("tinyint out of range")));
	PG_RETURN_INT8(ival);
}
PG_FUNCTION_INFO_V1(pgstrom_int2_to_int1);

Datum
pgstrom_int4_to_int1(PG_FUNCTION_ARGS)
{
	int32	ival = PG_GETARG_INT32(0);

	if (ival < PG_INT8_MIN || ival > PG_INT8_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("tinyint out of range")));
	PG_RETURN_INT8(ival);
}
PG_FUNCTION_INFO_V1(pgstrom_int4_to_int1);

Datum
pgstrom_int8_to_int1(PG_FUNCTION_ARGS)
{
	int64	ival = PG_GETARG_INT64(0);

	if (ival < PG_INT8_MIN || ival > PG_INT8_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("tinyint out of range")));
	PG_RETURN_INT8(ival);
}
PG_FUNCTION_INFO_V1(pgstrom_int8_to_int1);

Datum
pgstrom_float4_to_int1(PG_FUNCTION_ARGS)
{
	float4	fval = PG_GETARG_FLOAT4(0);

	if (fval < (float4)PG_INT8_MIN || fval > (float4)PG_INT8_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("tinyint out of range")));
	PG_RETURN_INT8(fval);
}
PG_FUNCTION_INFO_V1(pgstrom_float4_to_int1);

Datum
pgstrom_float8_to_int1(PG_FUNCTION_ARGS)
{
	float8	fval = PG_GETARG_FLOAT8(0);

	if (fval < (float8)PG_INT8_MIN || fval > (float8)PG_INT8_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("tinyint out of range")));
	PG_RETURN_INT8((int8)fval);
}
PG_FUNCTION_INFO_V1(pgstrom_float8_to_int1);

Datum
pgstrom_numeric_to_int1(PG_FUNCTION_ARGS)
{
	int32	ival = DatumGetInt32(numeric_int4(fcinfo));

	if (ival < PG_INT8_MIN || ival > PG_INT8_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("tinyint out of range")));
	PG_RETURN_INT8((int8)ival);
}
PG_FUNCTION_INFO_V1(pgstrom_numeric_to_int1);

/*
 * Comparison functions
 */
Datum
pgstrom_int1eq(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 == arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int1eq);

Datum
pgstrom_int1ne(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 != arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int1ne);

Datum
pgstrom_int1lt(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 < arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int1lt);

Datum
pgstrom_int1le(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 <= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int1le);

Datum
pgstrom_int1gt(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 > arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int1gt);

Datum
pgstrom_int1ge(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 >= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int1ge);

Datum
pgstrom_int1larger(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_INT8(arg1 > arg2 ? arg1 : arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int1larger);

Datum
pgstrom_int1smaller(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_INT8(arg1 < arg2 ? arg1 : arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int1smaller);

Datum
pgstrom_int12eq(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int16	arg2 = PG_GETARG_INT16(1);

	PG_RETURN_BOOL(arg1 == arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int12eq);

Datum
pgstrom_int12ne(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int16	arg2 = PG_GETARG_INT16(1);

	PG_RETURN_BOOL(arg1 != arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int12ne);

Datum
pgstrom_int12lt(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int16	arg2 = PG_GETARG_INT16(1);

	PG_RETURN_BOOL(arg1 < arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int12lt);

Datum
pgstrom_int12le(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int16	arg2 = PG_GETARG_INT16(1);

	PG_RETURN_BOOL(arg1 <= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int12le);

Datum
pgstrom_int12gt(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int16	arg2 = PG_GETARG_INT16(1);

	PG_RETURN_BOOL(arg1 > arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int12gt);

Datum
pgstrom_int12ge(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int16	arg2 = PG_GETARG_INT16(1);

	PG_RETURN_BOOL(arg1 >= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int12ge);

Datum
pgstrom_int14eq(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	PG_RETURN_BOOL(arg1 == arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int14eq);

Datum
pgstrom_int14ne(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	PG_RETURN_BOOL(arg1 != arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int14ne);

Datum
pgstrom_int14lt(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	PG_RETURN_BOOL(arg1 < arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int14lt);

Datum
pgstrom_int14le(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	PG_RETURN_BOOL(arg1 <= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int14le);

Datum
pgstrom_int14gt(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	PG_RETURN_BOOL(arg1 > arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int14gt);

Datum
pgstrom_int14ge(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	PG_RETURN_BOOL(arg1 >= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int14ge);

Datum
pgstrom_int18eq(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int64	arg2 = PG_GETARG_INT64(1);

	PG_RETURN_BOOL(arg1 == arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int18eq);

Datum
pgstrom_int18ne(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int64	arg2 = PG_GETARG_INT64(1);

	PG_RETURN_BOOL(arg1 != arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int18ne);

Datum
pgstrom_int18lt(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int64	arg2 = PG_GETARG_INT64(1);

	PG_RETURN_BOOL(arg1 < arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int18lt);

Datum
pgstrom_int18le(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int64	arg2 = PG_GETARG_INT64(1);

	PG_RETURN_BOOL(arg1 <= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int18le);

Datum
pgstrom_int18gt(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int64	arg2 = PG_GETARG_INT64(1);

	PG_RETURN_BOOL(arg1 > arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int18gt);

Datum
pgstrom_int18ge(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int64	arg2 = PG_GETARG_INT64(1);

	PG_RETURN_BOOL(arg1 >= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int18ge);

Datum
pgstrom_int21eq(PG_FUNCTION_ARGS)
{
	int16	arg1 = PG_GETARG_INT16(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 == arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int21eq);

Datum
pgstrom_int21ne(PG_FUNCTION_ARGS)
{
	int16	arg1 = PG_GETARG_INT16(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 != arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int21ne);

Datum
pgstrom_int21lt(PG_FUNCTION_ARGS)
{
	int16	arg1 = PG_GETARG_INT16(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 < arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int21lt);

Datum
pgstrom_int21le(PG_FUNCTION_ARGS)
{
	int16	arg1 = PG_GETARG_INT16(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 <= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int21le);

Datum
pgstrom_int21gt(PG_FUNCTION_ARGS)
{
	int16	arg1 = PG_GETARG_INT16(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 > arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int21gt);

Datum
pgstrom_int21ge(PG_FUNCTION_ARGS)
{
	int16	arg1 = PG_GETARG_INT16(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 >= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int21ge);

Datum
pgstrom_int41eq(PG_FUNCTION_ARGS)
{
	int32	arg1 = PG_GETARG_INT32(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 == arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int41eq);

Datum
pgstrom_int41ne(PG_FUNCTION_ARGS)
{
	int32	arg1 = PG_GETARG_INT32(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 != arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int41ne);

Datum
pgstrom_int41lt(PG_FUNCTION_ARGS)
{
	int32	arg1 = PG_GETARG_INT32(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 < arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int41lt);

Datum
pgstrom_int41le(PG_FUNCTION_ARGS)
{
	int32	arg1 = PG_GETARG_INT32(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 <= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int41le);

Datum
pgstrom_int41gt(PG_FUNCTION_ARGS)
{
	int32	arg1 = PG_GETARG_INT32(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 > arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int41gt);

Datum
pgstrom_int41ge(PG_FUNCTION_ARGS)
{
	int32	arg1 = PG_GETARG_INT32(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 >= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int41ge);

Datum
pgstrom_int81eq(PG_FUNCTION_ARGS)
{
	int64	arg1 = PG_GETARG_INT64(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 == arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int81eq);

Datum
pgstrom_int81ne(PG_FUNCTION_ARGS)
{
	int64	arg1 = PG_GETARG_INT64(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 != arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int81ne);

Datum
pgstrom_int81lt(PG_FUNCTION_ARGS)
{
	int64	arg1 = PG_GETARG_INT64(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 < arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int81lt);

Datum
pgstrom_int81le(PG_FUNCTION_ARGS)
{
	int64	arg1 = PG_GETARG_INT64(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 <= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int81le);

Datum
pgstrom_int81gt(PG_FUNCTION_ARGS)
{
	int64	arg1 = PG_GETARG_INT64(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 > arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int81gt);

Datum
pgstrom_int81ge(PG_FUNCTION_ARGS)
{
	int64	arg1 = PG_GETARG_INT64(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_BOOL(arg1 >= arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int81ge);

Datum
pgstrom_int1hash(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);

	/* Does it really make sense? */
	return hash_any((unsigned char *)&arg1, sizeof(int8));
}
PG_FUNCTION_INFO_V1(pgstrom_int1hash);

Datum
pgstrom_btint1cmp(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_btint1cmp);

Datum
pgstrom_btint12cmp(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_btint12cmp);

Datum
pgstrom_btint14cmp(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_btint14cmp);

Datum
pgstrom_btint18cmp(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_btint18cmp);

Datum
pgstrom_btint21cmp(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_btint21cmp);

Datum
pgstrom_btint41cmp(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_btint41cmp);

Datum
pgstrom_btint81cmp(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_btint81cmp);

/* unary operators */
Datum
pgstrom_int1up(PG_FUNCTION_ARGS)
{
	int8	arg = PG_GETARG_INT8(0);

	PG_RETURN_INT8(arg);
}
PG_FUNCTION_INFO_V1(pgstrom_int1up);

Datum
pgstrom_int1um(PG_FUNCTION_ARGS)
{
	int8	arg = PG_GETARG_INT8(0);

	PG_RETURN_INT8(-arg);
}
PG_FUNCTION_INFO_V1(pgstrom_int1um);

Datum
pgstrom_int1abs(PG_FUNCTION_ARGS)
{
	int8	arg = PG_GETARG_INT8(0);

	PG_RETURN_INT8(arg < 0 ? -arg : arg);
}
PG_FUNCTION_INFO_V1(pgstrom_int1abs);

/*
 * Arithmetic operators
 */
Datum
pgstrom_int1pl(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int1pl);

Datum
pgstrom_int1mi(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int1mi);

Datum
pgstrom_int1mul(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int1mul);

Datum
pgstrom_int1div(PG_FUNCTION_ARGS)
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
		if (arg1 == PG_INT8_MIN)
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
PG_FUNCTION_INFO_V1(pgstrom_int1div);

Datum
pgstrom_int1mod(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	if (arg2 == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DIVISION_BY_ZERO),
				 errmsg("division by zero")));
	if (arg2 == -1)
		PG_RETURN_INT8(0);
	/* no overflow is possible */
	PG_RETURN_INT8(arg1 % arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int1mod);

Datum
pgstrom_int12pl(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int12pl);

Datum
pgstrom_int12mi(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int12mi);

Datum
pgstrom_int12mul(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int12mul);

Datum
pgstrom_int12div(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int12div);

Datum
pgstrom_int14pl(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int14pl);

Datum
pgstrom_int14mi(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int14mi);

Datum
pgstrom_int14mul(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int14mul);

Datum
pgstrom_int14div(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int14div);

Datum
pgstrom_int18pl(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int18pl);

Datum
pgstrom_int18mi(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int18mi);

Datum
pgstrom_int18mul(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int18mul);

Datum
pgstrom_int18div(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int18div);

Datum
pgstrom_int21pl(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int21pl);

Datum
pgstrom_int21mi(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int21mi);

Datum
pgstrom_int21mul(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int21mul);

Datum
pgstrom_int21div(PG_FUNCTION_ARGS)
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
		if (arg1 == PG_INT16_MIN)
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
PG_FUNCTION_INFO_V1(pgstrom_int21div);

Datum
pgstrom_int41pl(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int41pl);

Datum
pgstrom_int41mi(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int41mi);

Datum
pgstrom_int41mul(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int41mul);

Datum
pgstrom_int41div(PG_FUNCTION_ARGS)
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
		if (arg1 == PG_INT32_MIN)
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
PG_FUNCTION_INFO_V1(pgstrom_int41div);

Datum
pgstrom_int81pl(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int81pl);

Datum
pgstrom_int81mi(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int81mi);

Datum
pgstrom_int81mul(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int81mul);

Datum
pgstrom_int81div(PG_FUNCTION_ARGS)
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
		if (arg1 == PG_INT64_MIN)
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
PG_FUNCTION_INFO_V1(pgstrom_int81div);

/*
 * Bit operations
 */
Datum
pgstrom_int1and(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_INT8(arg1 & arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int1and);

Datum
pgstrom_int1or(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_INT8(arg1 | arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int1or);

Datum
pgstrom_int1xor(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int8	arg2 = PG_GETARG_INT8(1);

	PG_RETURN_INT8(arg1 ^ arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int1xor);

Datum
pgstrom_int1not(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);

	PG_RETURN_INT8(~arg1);
}
PG_FUNCTION_INFO_V1(pgstrom_int1not);

Datum
pgstrom_int1shl(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	PG_RETURN_INT8(arg1 << arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int1shl);

Datum
pgstrom_int1shr(PG_FUNCTION_ARGS)
{
	int8	arg1 = PG_GETARG_INT8(0);
	int32	arg2 = PG_GETARG_INT32(1);

	PG_RETURN_INT8(arg1 >> arg2);
}
PG_FUNCTION_INFO_V1(pgstrom_int1shr);

/*
 * misc functions
 */
Datum
pgstrom_cash_mul_int1(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_cash_mul_int1);

Datum
pgstrom_int1_mul_cash(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int1_mul_cash);

Datum
pgstrom_cash_div_int1(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_cash_div_int1);

/* aggregate functions */
Datum
pgstrom_int1_sum(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int1_sum);

Datum
pgstrom_int1_avg_accum(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int1_avg_accum);

Datum
pgstrom_int1_avg_accum_inv(PG_FUNCTION_ARGS)
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
PG_FUNCTION_INFO_V1(pgstrom_int1_avg_accum_inv);

Datum
pgstrom_int1_var_accum(PG_FUNCTION_ARGS)
{
	int32		ival = (int32)PG_GETARG_INT8(1);

	PG_GETARG_DATUM(1) = Int32GetDatum(ival);
	return int4_accum(fcinfo);
}
PG_FUNCTION_INFO_V1(pgstrom_int1_var_accum);

Datum
pgstrom_int1_var_accum_inv(PG_FUNCTION_ARGS)
{
	int32		ival = (int32)PG_GETARG_INT8(1);

	PG_GETARG_DATUM(1) = Int32GetDatum(ival);
	return int4_accum_inv(fcinfo);
}
PG_FUNCTION_INFO_V1(pgstrom_int1_var_accum_inv);
