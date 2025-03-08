/*
 * arrow_pgsql.c
 *
 * Routines to intermediate PostgreSQL and Apache Arrow data types.
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifdef __PGSTROM_MODULE__
#include "postgres.h"
#if PG_VERSION_NUM < 130000
#include "access/hash.h"
#endif
#include "access/htup_details.h"
#if PG_VERSION_NUM >= 130000
#include "common/hashfn.h"
#endif
#include "port/pg_bswap.h"
#include "utils/array.h"
#include "utils/date.h"
#include "utils/timestamp.h"
#else	/* !__PGSTROM_MODULE__! */
/* if built as a part of standalone software */
#include "sql2arrow.h"
#include <arpa/inet.h>
#include <endian.h>

#define VARHDRSZ			((int32_t) sizeof(int32_t))
#define Min(x,y)			((x) < (y) ? (x) : (y))
#define Max(x,y)			((x) > (y) ? (x) : (y))

/* PostgreSQL type definitions */
typedef int32_t				DateADT;
typedef int64_t				TimeADT;
typedef int64_t				Timestamp;
typedef int64_t				TimeOffset;

#define UNIX_EPOCH_JDATE		2440588 /* == date2j(1970, 1, 1) */
#define POSTGRES_EPOCH_JDATE	2451545 /* == date2j(2000, 1, 1) */
#define USECS_PER_DAY			86400000000UL

typedef struct
{
	TimeOffset	time;
	int32_t		day;
	int32_t		month;
} Interval;
#endif

#include "arrow_ipc.h"
#include "float2.h"

/*
 * callbacks to write out min/max statistics
 */
static int
write_null_stat(SQLfield *attr, char *buf, size_t len,
				const SQLstat__datum *datum)
{
	return snprintf(buf, len, "null");
}

static int
write_int8_stat(SQLfield *attr, char *buf, size_t len,
				const SQLstat__datum *datum)
{
	return snprintf(buf, len, "%d", (int32_t)datum->i8);
}

static int
write_int16_stat(SQLfield *attr, char *buf, size_t len,
				 const SQLstat__datum *datum)
{
	return snprintf(buf, len, "%d", (int32_t)datum->i16);
}

static int
write_int32_stat(SQLfield *attr, char *buf, size_t len,
				 const SQLstat__datum *datum)
{
	return snprintf(buf, len, "%d", datum->i32);
}

static int
write_int64_stat(SQLfield *attr, char *buf, size_t len,
				 const SQLstat__datum *datum)
{
	return snprintf(buf, len, "%ld", datum->i64);
}

static int
write_int128_stat(SQLfield *attr, char *buf, size_t len,
				  const SQLstat__datum *datum)
{
	int128_t	ival = datum->i128;
	char		temp[64];
	char	   *pos = temp + sizeof(temp) - 1;
	bool		is_minus = false;

	/* special case handling if INT128 min value */
	if (~ival == (int128_t)0)
		return snprintf(buf, len, "-170141183460469231731687303715884105728");
	if (ival < 0)
	{
		is_minus = true;
		ival = -ival;
	}

	*pos = '\0';
	do {
		int		dig = ival % 10;

		*--pos = ('0' + dig);
		ival /= 10;
	} while (ival != 0);

	return snprintf(buf, len, "%s%s", (is_minus ? "-" : ""), pos);
}

/* ----------------------------------------------------------------
 *
 * Put value handler for each data types
 *
 * ----------------------------------------------------------------
 */

/*
 * MEMO: __fetch_XXbit() is a wrapper function when put-value handler is
 * called on pg2arrow that fetches values over the libpq binary protocol.
 * This byte-swapping is not necessary at the PG-Strom module context.
 */
static inline uint8_t __fetch_8bit(const void *addr)
{
	return *((uint8_t *)addr);
}

static inline uint16_t __fetch_16bit(const void *addr)
{
#ifdef __PGSTROM_MODULE__
	return *((uint16_t *)addr);
#else
	return be16toh(*((uint16_t *)addr));
#endif
}

static inline int16_t __fetch_16bit_signed(const void *addr)
{
#ifdef __PGSTROM_MODULE__
	return *((int16_t *)addr);
#else
	return (int16_t)be16toh(*((uint16_t *)addr));
#endif
}

static inline uint32_t __fetch_32bit(const void *addr)
{
#ifdef __PGSTROM_MODULE__
	return *((uint32_t *)addr);
#else
	return be32toh(*((uint32_t *)addr));
#endif
}

static inline uint64_t __fetch_64bit(const void *addr)
{
#ifdef __PGSTROM_MODULE__
	return *((uint64_t *)addr);
#else
	return be64toh(*((uint64_t *)addr));
#endif
}

#define STAT_UPDATES(COLUMN,FIELD,VALUE)					\
	do {													\
		if ((COLUMN)->stat_enabled)							\
		{													\
			if (!(COLUMN)->stat_datum.is_valid)				\
			{												\
				(COLUMN)->stat_datum.min.FIELD = VALUE;		\
				(COLUMN)->stat_datum.max.FIELD = VALUE;		\
				(COLUMN)->stat_datum.is_valid = true;		\
			}												\
			else											\
			{												\
				if ((COLUMN)->stat_datum.min.FIELD > VALUE)	\
					(COLUMN)->stat_datum.min.FIELD = VALUE;	\
				if ((COLUMN)->stat_datum.max.FIELD < VALUE)	\
					(COLUMN)->stat_datum.max.FIELD = VALUE;	\
			}												\
		}													\
	} while(0)

static size_t
put_bool_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	int8_t		value;

	if (!addr)
	{
		column->nullcount++;
		sql_buffer_clrbit(&column->nullmap, row_index);
		sql_buffer_clrbit(&column->values,  row_index);
	}
	else
	{
		value = *((const int8_t *)addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		if (value)
			sql_buffer_setbit(&column->values, row_index);
		else
			sql_buffer_clrbit(&column->values, row_index);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
move_bool_value(SQLfield *dest, const SQLfield *src, long sindex)
{
	size_t	dindex = dest->nitems++;

	if (!sql_buffer_getbit(&src->nullmap, sindex))
	{
		dest->nullcount++;
        sql_buffer_clrbit(&dest->nullmap, dindex);
        sql_buffer_clrbit(&dest->values,  dindex);
	}
	else
	{
		sql_buffer_setbit(&dest->nullmap, dindex);
		if (sql_buffer_getbit(&src->values, sindex))
			sql_buffer_setbit(&dest->values,  dindex);
		else
			sql_buffer_clrbit(&dest->values,  dindex);
	}
	return __buffer_usage_inline_type(dest);
}

/*
 * utility function to set NULL value
 */
static inline void
__put_inline_null_value(SQLfield *column, size_t row_index, int sz)
{
	column->nullcount++;
	sql_buffer_clrbit(&column->nullmap, row_index);
	sql_buffer_append_zero(&column->values, sz);
}

/*
 * IntXX/UintXX
 */
static size_t
put_int8_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	int8_t		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int8_t));
	else
	{
		assert(sz == sizeof(int8_t));
		value = *((const int8_t *)addr);

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int8_t));

		STAT_UPDATES(column,i8,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_uint8_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint8_t		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint8_t));
	else
	{
		assert(sz == sizeof(uint8_t));
		value = *((const uint8_t *)addr);
		if (value > INT8_MAX)
			Elog("Uint8 cannot store negative values");
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(uint8_t));

		STAT_UPDATES(column,u8,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_int16_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	int16_t		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int16_t));
	else
	{
		assert(sz == sizeof(int16_t));
		value = __fetch_16bit(addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sz);

		STAT_UPDATES(column,i16,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_uint16_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint16_t	value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint16_t));
	else
	{
		assert(sz == sizeof(uint16_t));
		value = __fetch_16bit(addr);
		if (value > INT16_MAX)
			Elog("Uint16 cannot store negative values");
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sz);

		STAT_UPDATES(column,u16,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_int32_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	int32_t		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint32_t));
	else
	{
		assert(sz == sizeof(uint32_t));
		value = __fetch_32bit(addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sz);
		STAT_UPDATES(column,i32,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_uint32_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint32_t	value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint32_t));
	else
	{
		assert(sz == sizeof(uint32_t));
		value = __fetch_32bit(addr);
		if (value > INT32_MAX)
			Elog("Uint32 cannot store negative values");
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sz);

		STAT_UPDATES(column,u32,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_int64_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	int64_t		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint64_t));
	else
	{
		assert(sz == sizeof(uint64_t));
		value = __fetch_64bit(addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sz);
		
		STAT_UPDATES(column,i64,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_uint64_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint64_t	value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint64_t));
	else
	{
		assert(sz == sizeof(uint64_t));
		value = __fetch_64bit(addr);
		if (value > INT64_MAX)
			Elog("Uint64 cannot store negative values");
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sz);
		
		STAT_UPDATES(column,u64,value);
	}
	return __buffer_usage_inline_type(column);
}

/*
 * FloatingPointXX
 */
static size_t
put_float16_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	half_t		value;
	float		fval;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint16_t));
	else
	{
		assert(sz == sizeof(uint16_t));
		value = __fetch_16bit(addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sz);

		fval = fp16_to_fp32(value);
		STAT_UPDATES(column,f32,fval);
	}
	return __buffer_usage_inline_type(column);
}

static int
write_float16_stat(SQLfield *attr, char *buf, size_t len,
				   const SQLstat__datum *datum)
{
	half_t		ival = fp32_to_fp16(datum->f32);

	return snprintf(buf, len, "%u", (uint32_t)ival);
}

static size_t
put_float32_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	int32_t		value;
	float		fval;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint32_t));
	else
	{
		assert(sz == sizeof(uint32_t));
		value = __fetch_32bit(addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sz);

		memcpy(&fval, &value, sizeof(float));
		STAT_UPDATES(column,f32,fval);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_float64_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	int64_t		value;
	double		fval;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint64_t));
	else
	{
		assert(sz == sizeof(uint64_t));
		value = __fetch_64bit(addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sz);

		memcpy(&fval, &value, sizeof(double));
		STAT_UPDATES(column,f64,fval);
	}
	return __buffer_usage_inline_type(column);
}

/*
 * Decimal
 */

/* parameters of Numeric type */
#define NUMERIC_DSCALE_MASK	0x3FFF
#define NUMERIC_SIGN_MASK	0xC000
#define NUMERIC_POS			0x0000
#define NUMERIC_NEG         0x4000
#define NUMERIC_NAN			0xC000
#define NUMERIC_PINF		0xD000
#define NUMERIC_NINF		0xF000

#define NBASE				10000
#define HALF_NBASE			5000
#define DEC_DIGITS			4	/* decimal digits per NBASE digit */
#define MUL_GUARD_DIGITS    2	/* these are measured in NBASE digits */
#define DIV_GUARD_DIGITS	4
typedef int16_t				NumericDigit;
typedef struct NumericVar
{
	int			ndigits;	/* # of digits in digits[] - can be 0! */
	int			weight;		/* weight of first digit */
	int			sign;		/* NUMERIC_POS, NUMERIC_NEG, or NUMERIC_NAN */
	int			dscale;		/* display scale */
	NumericDigit *digits;	/* base-NBASE digits */
} NumericVar;

#ifdef  __PGSTROM_MODULE__
#define NUMERIC_SHORT_SIGN_MASK			0x2000
#define NUMERIC_SHORT_DSCALE_MASK		0x1F80
#define NUMERIC_SHORT_DSCALE_SHIFT		7
#define NUMERIC_SHORT_WEIGHT_SIGN_MASK	0x0040
#define NUMERIC_SHORT_WEIGHT_MASK		0x003F

static void
init_var_from_num(NumericVar *nv, const char *addr, int sz)
{
	uint16_t		n_header = *((uint16_t *)addr);

	/* NUMERIC_HEADER_IS_SHORT */
	if ((n_header & 0x8000) != 0)
	{
		/* short format */
		const struct {
			uint16_t	n_header;
			NumericDigit n_data[FLEXIBLE_ARRAY_MEMBER];
		}  *n_short = (const void *)addr;
		size_t		hoff = ((uintptr_t)n_short->n_data - (uintptr_t)n_short);

		nv->ndigits = (sz - hoff) / sizeof(NumericDigit);
		nv->weight = (n_short->n_header & NUMERIC_SHORT_WEIGHT_MASK);
		if ((n_short->n_header & NUMERIC_SHORT_WEIGHT_SIGN_MASK) != 0)
			nv->weight |= NUMERIC_SHORT_WEIGHT_MASK;	/* negative value */
		nv->sign = ((n_short->n_header & NUMERIC_SHORT_SIGN_MASK) != 0
					? NUMERIC_NEG
					: NUMERIC_POS);
		nv->dscale = (n_short->n_header & NUMERIC_SHORT_DSCALE_MASK) >> NUMERIC_SHORT_DSCALE_SHIFT;
		nv->digits = (NumericDigit *)n_short->n_data;
	}
	else
	{
		/* long format */
		const struct {
			uint16_t      n_sign_dscale;  /* Sign + display scale */
			int16_t       n_weight;       /* Weight of 1st digit  */
			NumericDigit n_data[FLEXIBLE_ARRAY_MEMBER]; /* Digits */
		}  *n_long = (const void *)addr;
		size_t		hoff = ((uintptr_t)n_long->n_data - (uintptr_t)n_long);

		assert(sz >= hoff);
		nv->ndigits = (sz - hoff) / sizeof(NumericDigit);
		nv->weight = n_long->n_weight;
		nv->sign   = (n_long->n_sign_dscale & NUMERIC_SIGN_MASK);
		nv->dscale = (n_long->n_sign_dscale & NUMERIC_DSCALE_MASK);
		nv->digits = (NumericDigit *)n_long->n_data;
	}
}
#endif	/* __PGSTROM_MODULE__ */

static size_t
put_decimal_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int128_t));
	else
	{
		NumericVar		nv;
		int				scale = column->arrow_type.Decimal.scale;
		int128_t		value = 0;
		int				d, dig;
#ifdef __PGSTROM_MODULE__
		init_var_from_num(&nv, addr, sz);
#else
		struct {
			uint16_t	ndigits;	/* number of digits */
			uint16_t	weight;		/* weight of first digit */
			uint16_t	sign;		/* NUMERIC_(POS|NEG|NAN) */
			uint16_t	dscale;		/* display scale */
			NumericDigit digits[FLEXIBLE_ARRAY_MEMBER];
		}  *rawdata = (void *)addr;
		nv.ndigits	= __fetch_16bit(&rawdata->ndigits);
		nv.weight	= __fetch_16bit_signed(&rawdata->weight);
		nv.sign		= __fetch_16bit(&rawdata->sign);
		nv.dscale	= __fetch_16bit(&rawdata->dscale);
		nv.digits	= rawdata->digits;
#endif	/* __PGSTROM_MODULE__ */
		if ((nv.sign & NUMERIC_SIGN_MASK) == NUMERIC_SIGN_MASK)
			Elog("Decimal128 cannot map NaN, +Inf or -Inf in PostgreSQL Numeric");
		/* makes integer portion first */
		for (d=0; d <= nv.weight; d++)
		{
			dig = (d < nv.ndigits) ? __fetch_16bit(&nv.digits[d]) : 0;
			if (dig < 0 || dig >= NBASE)
				Elog("Numeric digit is out of range: %d", (int)dig);
			value = NBASE * value + (int128_t)dig;
		}
		/* makes floating point portion if any */
		while (scale > 0)
		{
			dig = (d >= 0 && d < nv.ndigits) ? __fetch_16bit(&nv.digits[d]) : 0;
			if (dig < 0 || dig >= NBASE)
				Elog("Numeric digit is out of range: %d", (int)dig);

			if (scale >= DEC_DIGITS)
				value = NBASE * value + dig;
			else if (scale == 3)
				value = 1000L * value + dig / 10L;
			else if (scale == 2)
				value =  100L * value + dig / 100L;
			else if (scale == 1)
				value =   10L * value + dig / 1000L;
			else
				Elog("internal bug");
			scale -= DEC_DIGITS;
			d++;
		}
		/* is it a negative value? */
		if ((nv.sign & NUMERIC_NEG) != 0)
			value = -value;

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(value));

		STAT_UPDATES(column,i128,value);
	}
	return __buffer_usage_inline_type(column);
}

#define MOVE_SCALAR_TEMPLATE(NAME,VALUE_TYPE,STAT_NAME)					\
	static size_t														\
	move_##NAME##_value(SQLfield *dest, const SQLfield *src, long sindex) \
	{																	\
		size_t	dindex = dest->nitems++;								\
																		\
		if (!sql_buffer_getbit(&src->nullmap, sindex))					\
			__put_inline_null_value(dest, dindex, sizeof(VALUE_TYPE));	\
		else															\
		{																\
			VALUE_TYPE	value;											\
																		\
			value = ((VALUE_TYPE *)src->values.data)[sindex];			\
			sql_buffer_setbit(&dest->nullmap, dindex);					\
			sql_buffer_append(&dest->values, &value,					\
							  sizeof(VALUE_TYPE));						\
			STAT_UPDATES(dest,STAT_NAME,value);							\
		}																\
		return __buffer_usage_inline_type(dest);						\
	}
MOVE_SCALAR_TEMPLATE(int8,     int8_t,  i8)
MOVE_SCALAR_TEMPLATE(uint8,   uint8_t,  u8)
MOVE_SCALAR_TEMPLATE(int16,   int16_t, i16)
MOVE_SCALAR_TEMPLATE(uint16, uint16_t, u16)
MOVE_SCALAR_TEMPLATE(int32,   int32_t, i32)
MOVE_SCALAR_TEMPLATE(uint32, uint32_t, u32)
MOVE_SCALAR_TEMPLATE(int64,   int64_t, i64)
MOVE_SCALAR_TEMPLATE(uint64, uint64_t, u64)
static size_t
move_float16_value(SQLfield *dest, const SQLfield *src, long sindex)
{
	size_t	dindex = dest->nitems++;

	if (!sql_buffer_getbit(&src->nullmap, sindex))
		__put_inline_null_value(dest, dindex, sizeof(float2_t));
	else
	{
		float2_t	value;
		float4_t	fval;

		value = ((float2_t *)src->values.data)[sindex];
		sql_buffer_setbit(&dest->nullmap, dindex);
		sql_buffer_append(&dest->values, &value, sizeof(float2_t));
		fval = fp16_to_fp32(value);
		STAT_UPDATES(dest, f32, fval);
	}
	return __buffer_usage_inline_type(dest);
}
MOVE_SCALAR_TEMPLATE(float32, float4_t, f32)
MOVE_SCALAR_TEMPLATE(float64, float8_t, f64)
MOVE_SCALAR_TEMPLATE(decimal, int128_t, i128)

/*
 * Date
 */
static size_t
__put_date_day_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	int32_t		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int32_t));
	else
	{
		assert(sz == sizeof(DateADT));
		value = __fetch_32bit(addr);
		value += (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int32_t));
		STAT_UPDATES(column,i32,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
__put_date_ms_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	int64_t		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int64_t));
	else
	{
		assert(sz == sizeof(DateADT));
		value = __fetch_32bit(addr);
		value += (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
		/* adjust ArrowDateUnit__Day to __MilliSecond */
		value *= 86400000L;

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int64_t));
		STAT_UPDATES(column,i64,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_date_value(SQLfield *column, const char *addr, int sz)
{
	/* validation checks only first call */
	switch (column->arrow_type.Date.unit)
	{
		case ArrowDateUnit__Day:
			column->put_value = __put_date_day_value;
			column->write_stat = write_int32_stat;
			break;
		case ArrowDateUnit__MilliSecond:
			column->put_value = __put_date_ms_value;
			column->write_stat = write_int64_stat;
			break;
		default:
			Elog("ArrowTypeDate has unknown unit (%d)",
				 column->arrow_type.Date.unit);
			break;
	}
	return column->put_value(column, addr, sz);
}


static size_t
move_date_value(SQLfield *dest, const SQLfield *src, long sindex)
{
	assert(src->arrow_type.Date.unit == dest->arrow_type.Date.unit);
	switch (src->arrow_type.Date.unit)
	{
		case ArrowDateUnit__Day:
			return move_int32_value(dest, src, sindex);
		case ArrowDateUnit__MilliSecond:
			return move_int64_value(dest, src, sindex);
		default:
			break;
	}
	Elog("ArrowTypeDate has unknown unit (%d)",
		 src->arrow_type.Date.unit);
}

/*
 * Time
 */
static size_t
__put_time_sec_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	TimeADT		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int32_t));
	else
	{
		assert(sz == sizeof(TimeADT));
		/* convert from ArrowTimeUnit__MicroSecond to __Second */
		value = __fetch_64bit(addr) / 1000000L;
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int32_t));
		STAT_UPDATES(column,i32,value);
	}
	return __buffer_usage_inline_type(column);

}

static size_t
__put_time_ms_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	TimeADT		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int32_t));
	else
	{
		assert(sz == sizeof(TimeADT));
		/* convert from ArrowTimeUnit__MicroSecond to __MiliSecond */
		value = __fetch_64bit(addr) / 1000L;
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int32_t));
		STAT_UPDATES(column,i32,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
__put_time_us_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	TimeADT		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int64_t));
	else
	{
		assert(sz == sizeof(TimeADT));
		/* PostgreSQL native is ArrowTimeUnit__MicroSecond */
		value = __fetch_64bit(addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int64_t));
		STAT_UPDATES(column,i64,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
__put_time_ns_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	TimeADT		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int64_t));
	else
	{
		assert(sz == sizeof(TimeADT));
		/* convert from ArrowTimeUnit__MicroSecond to __NanoSecond */
		value = __fetch_64bit(addr) * 1000L;
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int64_t));
		STAT_UPDATES(column,i64,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_time_value(SQLfield *column, const char *addr, int sz)
{
	switch (column->arrow_type.Time.unit)
	{
		case ArrowTimeUnit__Second:
			if (column->arrow_type.Time.bitWidth != 32)
				Elog("ArrowTypeTime has inconsistent bitWidth(%d) for [sec]",
					 column->arrow_type.Time.bitWidth);
			column->put_value = __put_time_sec_value;
			column->write_stat = write_int32_stat;
			break;
		case ArrowTimeUnit__MilliSecond:
			if (column->arrow_type.Time.bitWidth != 32)
				Elog("ArrowTypeTime has inconsistent bitWidth(%d) for [ms]",
					 column->arrow_type.Time.bitWidth);
			column->put_value = __put_time_ms_value;
			column->write_stat = write_int32_stat;
			break;
		case ArrowTimeUnit__MicroSecond:
			if (column->arrow_type.Time.bitWidth != 64)
				Elog("ArrowTypeTime has inconsistent bitWidth(%d) for [us]",
					 column->arrow_type.Time.bitWidth);
			column->put_value = __put_time_us_value;
			column->write_stat = write_int64_stat;
			break;
		case ArrowTimeUnit__NanoSecond:
			if (column->arrow_type.Time.bitWidth != 64)
				Elog("ArrowTypeTime has inconsistent bitWidth(%d) for [ns]",
					 column->arrow_type.Time.bitWidth);
			column->put_value = __put_time_ns_value;
			column->write_stat = write_int64_stat;
			break;
		default:
			Elog("ArrowTypeTime has unknown unit (%d)",
				 column->arrow_type.Time.unit);
			break;
	}
	return column->put_value(column, addr, sz);
}

static size_t
move_time_value(SQLfield *dest, const SQLfield *src, long sindex)
{
	assert(src->arrow_type.Time.unit == dest->arrow_type.Time.unit);
	switch (src->arrow_type.Time.unit)
	{
		case ArrowTimeUnit__Second:
		case ArrowTimeUnit__MilliSecond:
			return move_int32_value(dest, src, sindex);
		case ArrowTimeUnit__MicroSecond:
		case ArrowTimeUnit__NanoSecond:
			return move_int64_value(dest, src, sindex);
		default:
			break;
	}
	Elog("ArrowTypeTime has unknown unit (%d)",
		 src->arrow_type.Time.unit);
}

/*
 * Timestamp
 */
static size_t
__put_timestamp_sec_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	Timestamp	value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int64_t));
	else
	{
		assert(sz == sizeof(Timestamp));
		value = __fetch_64bit(addr);
		/* convert PostgreSQL epoch to UNIX epoch */
		value += (POSTGRES_EPOCH_JDATE -
				  UNIX_EPOCH_JDATE) * USECS_PER_DAY;
		/* convert ArrowTimeUnit__MicroSecond to __Second */
		value /= 1000000L;
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int64_t));
		STAT_UPDATES(column,i64,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
__put_timestamp_ms_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	Timestamp	value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int64_t));
	else
	{
		assert(sz == sizeof(Timestamp));
		value = __fetch_64bit(addr);
		/* convert PostgreSQL epoch to UNIX epoch */
		value += (POSTGRES_EPOCH_JDATE -
				  UNIX_EPOCH_JDATE) * USECS_PER_DAY;
		/* convert ArrowTimeUnit__MicroSecond to __MilliSecond */
		value /= 1000L;
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int64_t));
		STAT_UPDATES(column,i64,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
__put_timestamp_us_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	Timestamp	value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int64_t));
	else
	{
		assert(sz == sizeof(Timestamp));
		value = __fetch_64bit(addr);
		/* convert PostgreSQL epoch to UNIX epoch */
		value += (POSTGRES_EPOCH_JDATE -
				  UNIX_EPOCH_JDATE) * USECS_PER_DAY;
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int64_t));
		STAT_UPDATES(column,i64,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
__put_timestamp_ns_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	Timestamp	value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int64_t));
	else
	{
		assert(sz == sizeof(Timestamp));
		value = __fetch_64bit(addr);
		/* convert PostgreSQL epoch to UNIX epoch */
		value += (POSTGRES_EPOCH_JDATE -
				  UNIX_EPOCH_JDATE) * USECS_PER_DAY;
		/* convert ArrowTimeUnit__MicroSecond to __MilliSecond */
		value *= 1000L;
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int64_t));
		STAT_UPDATES(column,i64,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_timestamp_value(SQLfield *column, const char *addr, int sz)
{
	switch (column->arrow_type.Timestamp.unit)
	{
		case ArrowTimeUnit__Second:
			column->put_value = __put_timestamp_sec_value;
			column->write_stat = write_int64_stat;
			break;
		case ArrowTimeUnit__MilliSecond:
			column->put_value = __put_timestamp_ms_value;
			column->write_stat = write_int64_stat;
			break;
		case ArrowTimeUnit__MicroSecond:
			column->put_value = __put_timestamp_us_value;
			column->write_stat = write_int64_stat;
			break;
		case ArrowTimeUnit__NanoSecond:
			column->put_value = __put_timestamp_ns_value;
			column->write_stat = write_int64_stat;
			break;
		default:
			Elog("ArrowTypeTimestamp has unknown unit (%d)",
				column->arrow_type.Timestamp.unit);
			break;
	}
	return column->put_value(column, addr, sz);
}

static size_t
move_timestamp_value(SQLfield *dest, const SQLfield *src, long sindex)
{
	assert(src->arrow_type.Timestamp.unit == dest->arrow_type.Timestamp.unit);
	switch (src->arrow_type.Timestamp.unit)
	{
		case ArrowTimeUnit__Second:
		case ArrowTimeUnit__MilliSecond:
		case ArrowTimeUnit__MicroSecond:
		case ArrowTimeUnit__NanoSecond:
			return move_int64_value(dest, src, sindex);
		default:
			break;
	}
	Elog("ArrowTypeTimestamp has unknown unit (%d)",
		 dest->arrow_type.Timestamp.unit);
}

/*
 * Interval
 */
#define DAYS_PER_MONTH	30		/* assumes exactly 30 days per month */
#define HOURS_PER_DAY	24		/* assume no daylight savings time changes */

static size_t
__put_interval_year_month_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint32_t));
	else
	{
		uint32_t	m;

		assert(sz == sizeof(Interval));
		m = __fetch_32bit(&((const Interval *)addr)->month);
		sql_buffer_append(&column->values, &m, sizeof(uint32_t));
	}
	return __buffer_usage_inline_type(column);
}

static size_t
__put_interval_day_time_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;

	if (!addr)
		__put_inline_null_value(column, row_index, 2 * sizeof(uint32_t));
	else
	{
		Interval	iv;
		uint32_t	value;

		assert(sz == sizeof(Interval));
		iv.time  = __fetch_64bit(&((const Interval *)addr)->time);
		iv.day   = __fetch_32bit(&((const Interval *)addr)->day);
		iv.month = __fetch_32bit(&((const Interval *)addr)->month);

		/*
		 * Unit of PostgreSQL Interval is micro-seconds. Arrow Interval::time
		 * is represented as a pair of elapsed days and milli-seconds; needs
		 * to be adjusted.
		 */
		value = iv.month + DAYS_PER_MONTH * iv.day;
		sql_buffer_append(&column->values, &value, sizeof(uint32_t));
		value = iv.time / 1000;
		sql_buffer_append(&column->values, &value, sizeof(uint32_t));
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_interval_value(SQLfield *sql_field, const char *addr, int sz)
{
	switch (sql_field->arrow_type.Interval.unit)
	{
		case ArrowIntervalUnit__Year_Month:
			sql_field->put_value = __put_interval_year_month_value;
			break;
		case ArrowIntervalUnit__Day_Time:
			sql_field->put_value = __put_interval_day_time_value;
			break;
		default:
			Elog("column attribute \"%s\" has unknown Arrow::Interval.unit(%d)",
				 sql_field->field_name,
				 sql_field->arrow_type.Interval.unit);
			break;
	}
	return sql_field->put_value(sql_field, addr, sz);
}

static size_t
move_interval_value(SQLfield *dest, const SQLfield *src, long sindex)
{
	assert(src->arrow_type.Interval.unit == dest->arrow_type.Interval.unit);
	switch (src->arrow_type.Interval.unit)
	{
		case ArrowIntervalUnit__Year_Month:
			return move_uint32_value(dest, src, sindex);
		case ArrowIntervalUnit__Day_Time:
			return move_uint64_value(dest, src, sindex);
		default:
			break;
	}
	Elog("Arrow::Interval.unit is unknown (%d)",
		 src->arrow_type.Interval.unit);
}

/*
 * Utf8, Binary
 */
static size_t
put_variable_value(SQLfield *column,
				   const char *addr, int sz)
{
	size_t		row_index = column->nitems++;

	if (row_index == 0)
		sql_buffer_append_zero(&column->values, sizeof(uint32_t));
	if (!addr)
	{
		column->nullcount++;
		sql_buffer_clrbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values,
						  &column->extra.usage, sizeof(uint32_t));
	}
	else
	{
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->extra, addr, sz);
		sql_buffer_append(&column->values,
						  &column->extra.usage, sizeof(uint32_t));
	}
	return __buffer_usage_varlena_type(column);
}


static size_t
move_variable_value(SQLfield *dest, const SQLfield *src, long sindex)
{
	const char *addr = NULL;
	int			sz = 0;

	if (sql_buffer_getbit(&src->nullmap, sindex))
	{
		uint32_t	head = ((uint32_t *)src->values.data)[sindex];
		uint32_t	tail = ((uint32_t *)src->values.data)[sindex+1];

		assert(head <= tail && tail <= src->extra.usage);
		if (tail - head >= INT_MAX)
			Elog("too large variable data (len: %u)", tail - head);
		addr = src->extra.data + head;
		sz   = tail - head;
	}
	return put_variable_value(dest, addr, sz);
}

/*
 * FixedSizeBinary
 */
static size_t
put_bpchar_value(SQLfield *column,
				 const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	int			len = column->arrow_type.FixedSizeBinary.byteWidth;
	char	   *temp = alloca(len);

	assert(len > 0);
	memset(temp, ' ', len);
	if (!addr)
	{
		column->nullcount++;
		sql_buffer_clrbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, temp, len);
	}
	else
	{
		memcpy(temp, addr, Min(sz, len));
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, temp, len);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
move_bpchar_value(SQLfield *dest, const SQLfield *src, long sindex)
{
	const char *addr = NULL;
	int		unitsz = src->arrow_type.FixedSizeBinary.byteWidth;

	if (sql_buffer_getbit(&src->nullmap, sindex))
	{
		addr = src->values.data + unitsz * sindex;
	}
	return put_bpchar_value(dest, addr, unitsz);
}

/*
 * List::<element> type
 */
static size_t
put_array_value(SQLfield *column,
				const char *addr, int sz)
{
	SQLfield   *element = column->element;
	size_t		row_index = column->nitems++;

	if (row_index == 0)
		sql_buffer_append_zero(&column->values, sizeof(uint32_t));
	if (!addr)
	{
		column->nullcount++;
		sql_buffer_clrbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &element->nitems, sizeof(int32_t));
	}
	else
	{
#ifdef __PGSTROM_MODULE__
		/*
		 * NOTE: varlena of ArrayType may have short-header (1b, not 4b).
		 * We assume (addr - VARHDRSZ) is a head of ArrayType for performance
		 * benefit by elimination of redundant copy just for header.
		 * Due to the reason, we should never rely on varlena header, thus,
		 * unable to use VARSIZE() or related ones.
		 */
		ArrayType  *array = (ArrayType *)(addr - VARHDRSZ);
		size_t		i, nitems = 1;
		bits8	   *nullmap;
		char	   *base;
		size_t		off = 0;

		for (i=0; i < ARR_NDIM(array); i++)
			nitems *= ARR_DIMS(array)[i];
		nullmap = ARR_NULLBITMAP(array);
		base = ARR_DATA_PTR(array);
		for (i=0; i < nitems; i++)
		{
			if (nullmap && att_isnull(i, nullmap))
			{
				element->put_value(element, NULL, 0);
			}
			else if (element->sql_type.pgsql.typbyval)
			{
				Assert(element->sql_type.pgsql.typlen > 0 &&
					   element->sql_type.pgsql.typlen <= sizeof(Datum));
				element->put_value(element, base + off,
								   element->sql_type.pgsql.typlen);
				off = TYPEALIGN(element->sql_type.pgsql.typalign,
								off + element->sql_type.pgsql.typlen);
			}
			else if (element->sql_type.pgsql.typlen == -1)
			{
				int		vl_len = VARSIZE_ANY_EXHDR(base + off);
				char   *vl_data = VARDATA_ANY(base + off);

				element->put_value(element, vl_data, vl_len);
				off = TYPEALIGN(element->sql_type.pgsql.typalign,
								off + VARSIZE_ANY(base + off));
			}
			else
			{
				Elog("Bug? PostgreSQL Array has unsupported element type");
			}
		}
#else  /* __PGSTROM_MODULE__ */
		struct {
			int32_t		ndim;
			int32_t		hasnull;
			int32_t		element_type;
			struct {
				int32_t	sz;
				int32_t	lb;
			} dim[FLEXIBLE_ARRAY_MEMBER];
		}  *rawdata = (void *) addr;
		int32_t		ndim = __fetch_32bit(&rawdata->ndim);
		//int32_t		hasnull = __fetch_32bit(&rawdata->hasnull);
		Oid			element_typeid = __fetch_32bit(&rawdata->element_type);
		size_t		i, nitems = 1;
		int			item_sz;
		char	   *pos;

		if (element_typeid != element->sql_type.pgsql.typeid)
			Elog("PostgreSQL array type mismatch");
		if (ndim < 1)
			Elog("Invalid dimension size of PostgreSQL Array (ndim=%d)", ndim);
		for (i=0; i < ndim; i++)
			nitems *= __fetch_32bit(&rawdata->dim[i].sz);

		pos = (char *)&rawdata->dim[ndim];
		for (i=0; i < nitems; i++)
		{
			if (pos + sizeof(int32_t) > addr + sz)
				Elog("out of range - binary array has corruption");
			item_sz = __fetch_32bit(pos);
			pos += sizeof(int32_t);
			if (item_sz < 0)
				sql_field_put_value(element, NULL, 0);
			else
			{
				sql_field_put_value(element, pos, item_sz);
				pos += item_sz;
			}
		}
#endif /* __PGSTROM_MODULE__ */
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &element->nitems, sizeof(int32_t));
	}
	return __buffer_usage_inline_type(column) + element->__curr_usage__;
}

static size_t
move_array_value(SQLfield *dest, const SQLfield *src, long sindex)
{
	SQLfield   *d_elem = dest->element;
	long		dindex = dest->nitems++;

	if (!sql_buffer_getbit(&src->nullmap, sindex))
	{
		/* add NULL */
		dest->nullcount++;
		sql_buffer_clrbit(&dest->nullmap, sindex);
		sql_buffer_append(&dest->values, &d_elem->nitems, sizeof(int32_t));
	}
	else
	{
		SQLfield   *s_elem = src->element;
		uint32_t	head = ((uint32_t *)src->values.data)[sindex];
		uint32_t	tail = ((uint32_t *)src->values.data)[sindex+1];
		uint32_t	curr;

		assert(head <= tail);
		assert(IsSQLfieldCompatible(d_elem, s_elem));
		for (curr = head; curr < tail; curr++)
			sql_field_move_value(d_elem, s_elem, curr);

		sql_buffer_setbit(&dest->nullmap, dindex);
		sql_buffer_append(&dest->values, &d_elem->nitems, sizeof(int32_t));
	}
	return __buffer_usage_inline_type(dest) + d_elem->__curr_usage__;
}

/*
 * Arrow::Struct
 */
static size_t
put_composite_value(SQLfield *column,
					const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	size_t		usage = 0;
	int			j;

	if (!addr)
	{
		column->nullcount++;
		sql_buffer_clrbit(&column->nullmap, row_index);
		/* NULL for all the subtypes */
		for (j=0; j < column->nfields; j++)
		{
			usage += sql_field_put_value(&column->subfields[j], NULL, 0);
		}
	}
	else
	{
#ifdef __PGSTROM_MODULE__
		HeapTupleHeader htup = (HeapTupleHeader)(addr - VARHDRSZ);
		bits8	   *nullmap = NULL;
		int			j, nvalids;
		char	   *base = (char *)htup + htup->t_hoff;
		size_t		off = 0;

		if ((htup->t_infomask & HEAP_HASNULL) != 0)
			nullmap = htup->t_bits;
		nvalids = HeapTupleHeaderGetNatts(htup);

		for (j=0; j < column->nfields; j++)
		{
			SQLfield   *field = &column->subfields[j];
			int			vl_len;
			char	   *vl_dat;

			if (j >= nvalids || (nullmap && att_isnull(j, nullmap)))
			{
				usage += sql_field_put_value(field, NULL, 0);
			}
			else if (field->sql_type.pgsql.typbyval)
			{
				Assert(field->sql_type.pgsql.typlen > 0 &&
					   field->sql_type.pgsql.typlen <= sizeof(Datum));

				off = TYPEALIGN(field->sql_type.pgsql.typalign, off);
				usage += sql_field_put_value(field, base + off,
											 field->sql_type.pgsql.typlen);
				off += field->sql_type.pgsql.typlen;
			}
			else if (field->sql_type.pgsql.typlen == -1)
			{
				if (!VARATT_NOT_PAD_BYTE(base + off))
					off = TYPEALIGN(field->sql_type.pgsql.typalign, off);
				vl_dat = VARDATA_ANY(base + off);
				vl_len = VARSIZE_ANY_EXHDR(base + off);
				usage += sql_field_put_value(field, vl_dat, vl_len);
				off += VARSIZE_ANY(base + off);
			}
			else
			{
				Elog("Bug? sub-field '%s' of column '%s' has unsupported type",
					 field->field_name,
					 column->field_name);
			}
			assert(column->nitems == field->nitems);
		}
#else  /* __PGSTROM_MODULE__ */
		const char *pos = addr;
		int			j, nvalids;

		if (sz < sizeof(uint32_t))
			Elog("binary composite record corruption");
		nvalids = __fetch_32bit(pos);
		pos += sizeof(int);
		for (j=0; j < column->nfields; j++)
		{
			SQLfield *sub_field = &column->subfields[j];
			Oid		typeid;
			int32_t	len;

			if (j >= nvalids)
			{
				usage += sql_field_put_value(sub_field, NULL, 0);
				continue;
			}
			if ((pos - addr) + sizeof(Oid) + sizeof(int) > sz)
				Elog("binary composite record corruption");
			typeid = __fetch_32bit(pos);
			pos += sizeof(Oid);
			if (sub_field->sql_type.pgsql.typeid != typeid)
				Elog("composite subtype mismatch");
			len = __fetch_32bit(pos);
			pos += sizeof(int32_t);
			if (len == -1)
			{
				usage += sql_field_put_value(sub_field, NULL, 0);
			}
			else
			{
				if ((pos - addr) + len > sz)
					Elog("binary composite record corruption");
				usage += sql_field_put_value(sub_field, pos, len);
				pos += len;
			}
			assert(column->nitems == sub_field->nitems);
		}
#endif /* __PGSTROM_MODULE__ */
		sql_buffer_setbit(&column->nullmap, row_index);
	}
	if (column->nullcount > 0)
		usage += ARROWALIGN(column->nullmap.usage);
	return usage;
}

static size_t
move_composite_value(SQLfield *dest, const SQLfield *src, long sindex)
{
	long	dindex = dest->nitems++;
	size_t	usage = 0;

	if (!sql_buffer_getbit(&src->nullmap, sindex))
	{
		dest->nullcount++;
		sql_buffer_clrbit(&dest->nullmap, dindex);
		for (int j=0; j < dest->nfields; j++)
		{
			usage += sql_field_put_value(&dest->subfields[j], NULL, 0);
		}
	}
	else
	{
		for (int j=0; j < dest->nfields; j++)
		{
			usage += sql_field_move_value(&dest->subfields[j],
										  &src->subfields[j], sindex);
		}
		sql_buffer_setbit(&dest->nullmap, dindex);
	}
	if (dest->nullcount > 0)
		usage += ARROWALIGN(dest->nullmap.usage);
	return usage;
}

/*
 * Enum values
 */
static size_t
put_dictionary_value(SQLfield *column,
					 const char *addr, int sz)
{
	size_t		row_index = column->nitems++;

	if (!addr)
	{
		column->nullcount++;
		sql_buffer_clrbit(&column->nullmap, row_index);
		sql_buffer_append_zero(&column->values, sizeof(uint32_t));
	}
	else
	{
		SQLdictionary *enumdict = column->enumdict;
		hashItem   *hitem;
		uint32_t	hash;

		hash = hash_any((const unsigned char *)addr, sz);
		for (hitem = enumdict->hslots[hash % enumdict->nslots];
			 hitem != NULL;
			 hitem = hitem->next)
		{
			if (hitem->hash == hash &&
				hitem->label_sz == sz &&
				memcmp(hitem->label, addr, sz) == 0)
				break;
		}
		if (!hitem)
			Elog("Enum label was not found in pg_enum result");
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values,  &hitem->index, sizeof(int32_t));
	}
	return __buffer_usage_inline_type(column);
}

static size_t
move_dictionary_value(SQLfield *dest, const SQLfield *src, long sindex)
{
	if (!sql_buffer_getbit(&src->nullmap, sindex))
		return put_dictionary_value(dest, NULL, 0);
	if (dest->enumdict == src->enumdict)
	{
		uint32_t	enum_id = ((uint32_t *)src->values.data)[sindex];

		return put_uint32_value(dest, (char *)&enum_id, sizeof(uint32_t));
	}
	Elog("Different Enum dictionary is not compatible");
}

/*
 * put_value handler for contrib/cube module
 */
static size_t
put_extra_cube_value(SQLfield *column,
					 const char *addr, int sz)
{
	size_t		row_index = column->nitems++;

	if (row_index == 0)
		sql_buffer_append_zero(&column->values, sizeof(uint32_t));
	if (!addr)
	{
		column->nullcount++;
		sql_buffer_clrbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values,
						  &column->extra.usage, sizeof(uint32_t));
	}
	else
	{
		uint32_t	header = __fetch_32bit(addr);
		uint32_t	i, nitems = (header & 0x7fffffffU);
		uint64_t	value;

		if ((header & 0x80000000U) == 0)
			nitems += nitems;
		if (sz != sizeof(uint32_t) + sizeof(uint64_t) * nitems)
			Elog("cube binary data looks broken");
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->extra, &header, sizeof(uint32_t));
		addr += sizeof(uint32_t);
		for (i=0; i < nitems; i++)
		{
			value = __fetch_64bit(addr + sizeof(uint64_t) * i);
			sql_buffer_append(&column->extra, &value, sizeof(uint64_t));
		}
		sql_buffer_append(&column->values,
						  &column->extra.usage, sizeof(uint32_t));
	}
	return __buffer_usage_varlena_type(column);
}

/* ----------------------------------------------------------------
 *
 * setup handler for each data types
 *
 * ----------------------------------------------------------------
 */
static int
assignArrowTypeInt(SQLfield *column, bool is_signed,
				   ArrowField *arrow_field)
{
	initArrowNode(&column->arrow_type, Int);
	column->arrow_type.Int.is_signed = is_signed;
	switch (column->sql_type.pgsql.typlen)
	{
		case sizeof(char):
			column->arrow_type.Int.bitWidth = 8;
			column->put_value = (is_signed ? put_int8_value : put_uint8_value);
			column->move_value = (is_signed ? move_int8_value : move_uint8_value);
			column->write_stat = write_int8_stat;
			break;
		case sizeof(short):
			column->arrow_type.Int.bitWidth = 16;
			column->put_value = (is_signed ? put_int16_value : put_uint16_value);
			column->move_value = (is_signed ? move_int16_value : move_uint16_value);
			column->write_stat = write_int16_stat;
			break;
		case sizeof(int):
			column->arrow_type.Int.bitWidth = 32;
			column->put_value = (is_signed ? put_int32_value : put_uint32_value);
			column->move_value = (is_signed ? move_int32_value : move_uint32_value);
			column->write_stat = write_int32_stat;
			break;
		case sizeof(long):
			column->arrow_type.Int.bitWidth = 64;
			column->put_value = (is_signed ? put_int64_value : put_uint64_value);
			column->move_value = (is_signed ? move_int64_value : move_uint64_value);
			column->write_stat = write_int64_stat;
			break;
		default:
			Elog("unsupported Int width: %d",
				 column->sql_type.pgsql.typlen);
			break;
	}

	if (arrow_field)
	{
		int32_t		bitWidth = column->arrow_type.Int.bitWidth;

		if (arrow_field->type.node.tag != ArrowNodeTag__Int ||
			arrow_field->type.Int.bitWidth != bitWidth ||
			arrow_field->type.Int.is_signed != is_signed)
			Elog("attribute '%s' is not compatible", column->field_name);
	}
	return 2;		/* null map + values */
}

static int
assignArrowTypeFloatingPoint(SQLfield *column, ArrowField *arrow_field)
{
	initArrowNode(&column->arrow_type, FloatingPoint);
	switch (column->sql_type.pgsql.typlen)
	{
		case sizeof(short):		/* half */
			column->arrow_type.FloatingPoint.precision
				= ArrowPrecision__Half;
			column->put_value = put_float16_value;
			column->move_value = move_float16_value;
			column->write_stat = write_float16_stat;
			break;
		case sizeof(float):
			column->arrow_type.FloatingPoint.precision
				= ArrowPrecision__Single;
			column->put_value = put_float32_value;
			column->move_value = move_float32_value;
			column->write_stat = write_int32_stat;
			break;
		case sizeof(double):
			column->arrow_type.FloatingPoint.precision
				= ArrowPrecision__Double;
			column->put_value = put_float64_value;
			column->move_value = move_float64_value;
			column->write_stat = write_int64_stat;
			break;
		default:
			Elog("unsupported floating point width: %d",
				 column->sql_type.pgsql.typlen);
			break;
	}

	if (arrow_field)
	{
		ArrowPrecision precision = column->arrow_type.FloatingPoint.precision;

		if (arrow_field->type.node.tag != ArrowNodeTag__FloatingPoint ||
			arrow_field->type.FloatingPoint.precision != precision)
			Elog("attribute '%s' is not compatible", column->field_name);
	}
	return 2;		/* nullmap + values */
}

static int
assignArrowTypeBinary(SQLfield *column, ArrowField *arrow_field)
{
	if (arrow_field &&
		arrow_field->type.node.tag != ArrowNodeTag__Binary)
		Elog("attribute '%s' is not compatible", column->field_name);
	initArrowNode(&column->arrow_type, Binary);
	column->put_value = put_variable_value;
	column->move_value = move_variable_value;
	return 3;		/* nullmap + index + extra */
}

static int
assignArrowTypeUtf8(SQLfield *column, ArrowField *arrow_field)
{
	if (arrow_field &&
		arrow_field->type.node.tag != ArrowNodeTag__Utf8)
		Elog("attribute '%s' is not compatible", column->field_name);
	initArrowNode(&column->arrow_type, Utf8);
	column->put_value = put_variable_value;
	column->move_value = move_variable_value;
	return 3;		/* nullmap + index + extra */
}

static int
assignArrowTypeBpchar(SQLfield *column, ArrowField *arrow_field)
{
	int32_t		byteWidth;

	if (column->sql_type.pgsql.typmod <= VARHDRSZ)
		Elog("unexpected Bpchar definition (typmod=%d)",
			 column->sql_type.pgsql.typmod);
	byteWidth = column->sql_type.pgsql.typmod - VARHDRSZ;
	if (arrow_field &&
		(arrow_field->type.node.tag != ArrowNodeTag__FixedSizeBinary ||
		 arrow_field->type.FixedSizeBinary.byteWidth != byteWidth))
		Elog("attribute '%s' is not compatible", column->field_name);

	initArrowNode(&column->arrow_type, FixedSizeBinary);
	column->arrow_type.FixedSizeBinary.byteWidth = byteWidth;
	column->put_value = put_bpchar_value;
	column->move_value = move_bpchar_value;
	return 2;		/* nullmap + values */
}

static int
assignArrowTypeBool(SQLfield *column, ArrowField *arrow_field)
{
	if (arrow_field &&
		arrow_field->type.node.tag != ArrowNodeTag__Bool)
		Elog("attribute %s is not compatible", column->field_name);

	initArrowNode(&column->arrow_type, Bool);
	column->put_value = put_bool_value;
	column->move_value = move_bool_value;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeDecimal(SQLfield *column, ArrowField *arrow_field)
{
	int		typmod			= column->sql_type.pgsql.typmod;
	int		precision		= 30;	/* default, if typmod == -1 */
	int		scale			=  8;	/* default, if typmod == -1 */

	if (typmod >= VARHDRSZ)
	{
		typmod -= VARHDRSZ;
		precision = (typmod >> 16) & 0xffff;
		scale = (typmod & 0xffff);
	}
	if (arrow_field)
	{
		if (arrow_field->type.node.tag != ArrowNodeTag__Decimal)
			Elog("attribute %s is not compatible", column->field_name);
		precision = arrow_field->type.Decimal.precision;
		scale = arrow_field->type.Decimal.scale;
	}
	initArrowNode(&column->arrow_type, Decimal);
	column->arrow_type.Decimal.precision = precision;
	column->arrow_type.Decimal.scale = scale;
	column->arrow_type.Decimal.bitWidth = 128;
	column->put_value = put_decimal_value;
	column->move_value = move_decimal_value;
	column->write_stat = write_int128_stat;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeDate(SQLfield *column, ArrowField *arrow_field)
{
	ArrowDateUnit	unit = ArrowDateUnit__Day;

	if (arrow_field)
	{
		if (arrow_field->type.node.tag != ArrowNodeTag__Date)
			Elog("attribute %s is not compatible", column->field_name);
		unit = arrow_field->type.Date.unit;
	}
	initArrowNode(&column->arrow_type, Date);
	column->arrow_type.Date.unit = unit;
	column->put_value = put_date_value;
	column->move_value = move_date_value;
	column->write_stat = write_null_stat;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeTime(SQLfield *column, ArrowField *arrow_field)
{
	ArrowTimeUnit	unit = ArrowTimeUnit__MicroSecond;

	if (arrow_field)
	{
		if (arrow_field->type.node.tag != ArrowNodeTag__Time)
			Elog("attribute %s is not compatible", column->field_name);
		unit = arrow_field->type.Time.unit;
	}
	initArrowNode(&column->arrow_type, Time);
	column->arrow_type.Time.unit = unit;
	column->arrow_type.Time.bitWidth = 64;
	column->put_value = put_time_value;
	column->move_value = move_time_value;
	column->write_stat = write_null_stat;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeTimestamp(SQLfield *column, const char *tz_name,
						 ArrowField *arrow_field)
{
	ArrowTimeUnit	unit = ArrowTimeUnit__MicroSecond;

	if (arrow_field)
	{
		if (arrow_field->type.node.tag != ArrowNodeTag__Timestamp)
			Elog("attribute %s is not compatible", column->field_name);
		unit = arrow_field->type.Timestamp.unit;
	}
	initArrowNode(&column->arrow_type, Timestamp);
	column->arrow_type.Timestamp.unit = unit;
	if (tz_name)
	{
		column->arrow_type.Timestamp.timezone = pstrdup(tz_name);
		column->arrow_type.Timestamp._timezone_len = strlen(tz_name);
	}
	column->put_value = put_timestamp_value;
	column->move_value = move_timestamp_value;
	column->write_stat = write_null_stat;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeInterval(SQLfield *column, ArrowField *arrow_field)
{
	ArrowIntervalUnit	unit = ArrowIntervalUnit__Day_Time;

	if (arrow_field)
	{
		if (arrow_field->type.node.tag != ArrowNodeTag__Interval)
			Elog("attribute %s is not compatible", column->field_name);
		unit = arrow_field->type.Interval.unit;
	}
	initArrowNode(&column->arrow_type, Interval);
	column->arrow_type.Interval.unit = unit;
	column->put_value = put_interval_value;
	column->move_value = move_interval_value;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeList(SQLfield *column, ArrowField *arrow_field)
{
	if (arrow_field &&
		arrow_field->type.node.tag != ArrowNodeTag__List)
		Elog("attribute %s is not compatible", column->field_name);

	initArrowNode(&column->arrow_type, List);
	column->put_value = put_array_value;
	column->move_value = move_array_value;

	return 2;		/* nullmap + offset vector */
}

static int
assignArrowTypeStruct(SQLfield *column, ArrowField *arrow_field)
{
	if (arrow_field &&
		arrow_field->type.node.tag != ArrowNodeTag__Struct)
		Elog("attribute %s is not compatible", column->field_name);

	initArrowNode(&column->arrow_type, Struct);
	column->put_value = put_composite_value;
	column->move_value = move_composite_value;

	return 1;	/* only nullmap */
}

static int
assignArrowTypeDictionary(SQLfield *column, ArrowField *arrow_field)
{
	if (arrow_field)
	{
		ArrowTypeInt   *indexType;

		if (arrow_field->type.node.tag != ArrowNodeTag__Utf8)
			Elog("attribute %s is not compatible", column->field_name);
		if (!arrow_field->dictionary)
			Elog("attribute has no dictionary");
		indexType = &arrow_field->dictionary->indexType;
		if (indexType->node.tag == ArrowNodeTag__Int &&
			indexType->bitWidth == sizeof(uint32_t) &&
			!indexType->is_signed)
			Elog("IndexType of ArrowDictionaryEncoding must be Int32");
	}

	initArrowNode(&column->arrow_type, Utf8);
	column->put_value = put_dictionary_value;
	column->move_value = move_dictionary_value;

	return 2;	/* nullmap + values */
}

static int
assignArrowTypeExtraCube(SQLfield *column, ArrowField *arrow_field)
{
	if (arrow_field &&
		arrow_field->type.node.tag != ArrowNodeTag__Binary)
		Elog("attribute %s is not compatible", column->field_name);

	initArrowNode(&column->arrow_type, Binary);
	column->put_value = put_extra_cube_value;
	column->move_value = move_variable_value;
	return 3;		/* nullmap + index + extra */
}

/*
 * __assignArrowTypeHint
 */
static void
__assignArrowTypeHint(SQLfield *column,
					  const char *typname,
					  const char *typnamespace,
					  const char *typextension)
{
	int		index = column->numCustomMetadata++;
	ArrowKeyValue *kv;
	char	buf[300];
	int		sz = 0;

	if (!column->customMetadata)
		column->customMetadata = palloc(sizeof(ArrowKeyValue) * (index+1));
	else
		column->customMetadata = repalloc(column->customMetadata,
										  sizeof(ArrowKeyValue) * (index+1));
	kv = &column->customMetadata[index];
	__initArrowNode(&kv->node, ArrowNodeTag__KeyValue);
	kv->key = pstrdup("pg_type");
	kv->_key_len = 7;

	if (!typextension && strcmp(typnamespace, "pg_catalog") != 0)
	{
		strcpy(buf+sz, typnamespace);
		sz += strlen(typnamespace);
		buf[sz++] = '.';
	}
	strcpy(buf+sz, typname);
	sz += strlen(typname);
	if (typextension)
	{
		buf[sz++] = '@';
		strcpy(buf+sz, typextension);
		sz += strlen(typextension);
	}
	kv->value = pstrdup(buf);
	kv->_value_len = sz;
}

/*
 * assignArrowTypePgSQL
 */
int
assignArrowTypePgSQL(SQLfield *column,
					 const char *field_name,
					 Oid typeid,
					 int typmod,
					 const char *typname,
					 const char *typnamespace,
					 short typlen,
					 bool typbyval,
					 char typtype,
					 char typalign,
					 Oid typrelid,
					 Oid typelemid,
					 const char *tz_name,
					 const char *extname,
					 ArrowField *arrow_field)
{
	SQLtype__pgsql	   *pgtype = &column->sql_type.pgsql;
	
	memset(column, 0, sizeof(SQLfield));
	column->field_name = pstrdup(field_name);
	pgtype->typeid = typeid;
	pgtype->typmod = typmod;
	pgtype->typname = pstrdup(typname);
	pgtype->typnamespace = typnamespace;
	pgtype->typlen = typlen;
	pgtype->typbyval = typbyval;
	pgtype->typtype = typtype;
	if (typalign == 'c')
		pgtype->typalign = sizeof(char);
	else if (typalign == 's')
		pgtype->typalign = sizeof(short);
	else if (typalign == 'i')
		pgtype->typalign = sizeof(int);
	else if (typalign == 'd')
		pgtype->typalign = sizeof(double);

	/* array type */
	if (typelemid != 0)
	{
		if (typlen != -1)
			Elog("Bug? array type is not varlena (typlen != -1)");
		return assignArrowTypeList(column, arrow_field);
	}

	/* composite type */
	if (typrelid != 0)
	{
		__assignArrowTypeHint(column, typname, typnamespace, NULL);
		return assignArrowTypeStruct(column, arrow_field);
	}

	/* enum type */
	if (typtype == 'e')
	{
		__assignArrowTypeHint(column, typname, typnamespace, NULL);
		return assignArrowTypeDictionary(column, arrow_field);
	}

	/* several known types provided by extension */
	if (extname != NULL)
	{
		/* contrib/cube (relocatable) */
		if (strcmp(typname, "cube") == 0 &&
			strcmp(extname, "cube") == 0)
		{
			__assignArrowTypeHint(column, typname, typnamespace, extname);
			return assignArrowTypeExtraCube(column, arrow_field);
		}
	}

	/* other built-in types */
	if (strcmp(typnamespace, "pg_catalog") == 0)
	{
		/* well known built-in data types? */
		if (strcmp(typname, "bool") == 0)
		{
			return assignArrowTypeBool(column, arrow_field);
		}
		else if (strcmp(typname, "int2") == 0 ||
				 strcmp(typname, "int4") == 0 ||
				 strcmp(typname, "int8") == 0)
		{
			return assignArrowTypeInt(column, true, arrow_field);
		}
		else if (strcmp(typname, "float2") == 0 ||
				 strcmp(typname, "float4") == 0 ||
				 strcmp(typname, "float8") == 0)
		{
			return assignArrowTypeFloatingPoint(column, arrow_field);
		}
		else if (strcmp(typname, "date") == 0)
		{
			return assignArrowTypeDate(column, arrow_field);
		}
		else if (strcmp(typname, "time") == 0)
		{
			return assignArrowTypeTime(column, arrow_field);
		}
		else if (strcmp(typname, "timestamp") == 0)
		{
			return assignArrowTypeTimestamp(column, NULL, arrow_field);
		}
		else if (strcmp(typname, "timestamptz") == 0)
		{
			return assignArrowTypeTimestamp(column, tz_name, arrow_field);
		}
		else if (strcmp(typname, "interval") == 0)
		{
			return assignArrowTypeInterval(column, arrow_field);
		}
		else if (strcmp(typname, "text") == 0 ||
				 strcmp(typname, "varchar") == 0)
		{
			return assignArrowTypeUtf8(column, arrow_field);
		}
		else if (strcmp(typname, "bpchar") == 0)
		{
			return assignArrowTypeBpchar(column, arrow_field);
		}
		else if (strcmp(typname, "numeric") == 0)
		{
			return assignArrowTypeDecimal(column, arrow_field);
		}
	}
	/* elsewhere, we save the values just bunch of binary data */
	if (typlen > 0)
	{
		if (typlen == sizeof(char) ||
			typlen == sizeof(short) ||
			typlen == sizeof(int) ||
			typlen == sizeof(double))
		{
			__assignArrowTypeHint(column, typname, typnamespace, NULL);
			return assignArrowTypeInt(column, false, arrow_field);
		}
		/*
		 * MEMO: Unfortunately, we have no portable way to pack user defined
		 * fixed-length binary data types, because their 'send' handler often
		 * manipulate its internal data representation.
		 * Please check box_send() for example. It sends four float8 (which
		 * is reordered to bit-endien) values in 32bytes. We cannot understand
		 * its binary format without proper knowledge.
		 */
	}
	else if (typlen == -1)
	{
		__assignArrowTypeHint(column, typname, typnamespace, NULL);
		return assignArrowTypeBinary(column, arrow_field);
	}
	Elog("PostgreSQL type: '%s' is not supported", typname);
}
