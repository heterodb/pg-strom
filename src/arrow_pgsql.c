/*
 * arrow_pgsql.c
 *
 * Routines to intermediate PostgreSQL and Apache Arrow data types.
 * ----
 * Copyright 2011-2019 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2019 (C) The PG-Strom Development Team
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
#ifdef __PG2ARROW__
#include "postgres.h"
#include "port/pg_bswap.h"
#include "utils/date.h"
#include "utils/timestamp.h"
typedef struct SQLbuffer	StringInfoData;
typedef struct SQLbuffer   *StringInfo;
#if PG_VERSION_NUM < 110000
#ifdef WORDS_BIGENDIAN
#define __ntoh16(x)			(x)
#define __ntoh32(x)			(x)
#define __ntoh64(x)			(x)
#else
#define __ntoh16(x)			ntohs(x)
#define __ntoh32(x)			BSWAP32(x)
#define __ntoh64(x)			BSWAP64(x)
#endif
#else	/* >=PG11 */
#define __ntoh16(x)			pg_ntoh16(x)
#define __ntoh32(x)			pg_ntoh32(x)
#define __ntoh64(x)			pg_ntoh64(x)
#endif	/* >=PG11 */
#else	/* __PG2ARROW__ */
#include "pg_strom.h"
#define __ntoh16(x)			(x)
#define __ntoh32(x)			(x)
#define __ntoh64(x)			(x)
#endif	/* !__PG2ARROW__ */
#include "arrow_ipc.h"

/* ----------------------------------------------------------------
 *
 * Put value handler for each data types
 *
 * ----------------------------------------------------------------
 */
static inline size_t
__buffer_usage_inline_type(SQLfield *column)
{
	size_t		usage;

	usage = ARROWALIGN(column->values.usage);
	if (column->nullcount > 0)
		usage += ARROWALIGN(column->nullmap.usage);
	return usage;
}

static inline size_t
__buffer_usage_varlena_type(SQLfield *column)
{
	size_t		usage;

	usage = (ARROWALIGN(column->values.usage) +
			 ARROWALIGN(column->extra.usage));
	if (column->nullcount > 0)
		usage += ARROWALIGN(column->nullmap.usage);
	return usage;
}

static size_t
put_bool_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	int8		value;

	if (!addr)
	{
		column->nullcount++;
		sql_buffer_clrbit(&column->nullmap, row_index);
		sql_buffer_clrbit(&column->values,  row_index);
	}
	else
	{
		value = *((const int8 *)addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		if (value)
			sql_buffer_setbit(&column->values,  row_index);
		else
			sql_buffer_clrbit(&column->values,  row_index);
	}
	return __buffer_usage_inline_type(column);
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
	uint8		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint8));
	else
	{
		assert(sz == sizeof(uint8));
		value = *((const uint8 *)addr);

		if (!column->arrow_type.Int.is_signed && value > INT8_MAX)
			Elog("Uint8 cannot store negative values");

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(uint8));
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_int16_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint16		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint16));
	else
	{
		assert(sz == sizeof(uint16));
		value = __ntoh16(*((const uint16 *)addr));
		if (!column->arrow_type.Int.is_signed && value > INT16_MAX)
			Elog("Uint16 cannot store negative values");
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sz);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_int32_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint32		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint32));
	else
	{
		assert(sz == sizeof(uint32));
		value = __ntoh32(*((const uint32 *)addr));
		if (!column->arrow_type.Int.is_signed && value > INT32_MAX)
			Elog("Uint32 cannot store negative values");
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sz);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_int64_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint64		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint64));
	else
	{
		assert(sz == sizeof(uint64));
		value = __ntoh64(*((const uint64 *)addr));
		if (!column->arrow_type.Int.is_signed && value > INT64_MAX)
			Elog("Uint64 cannot store negative values");
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sz);
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
	uint16		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint16));
	else
	{
		assert(sz == sizeof(uint16));
		value = __ntoh16(*((const uint16 *)addr));
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sz);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_float32_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint32		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint32));
	else
	{
		assert(sz == sizeof(uint32));
		value = __ntoh32(*((const uint32 *)addr));
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sz);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_float64_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint64		value;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint64));
	else
	{
		assert(sz == sizeof(uint64));
		value = __ntoh64(*((const uint64 *)addr));
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sz);
	}
	return __buffer_usage_inline_type(column);
}

/*
 * Decimal
 */
#ifdef PG_INT128_TYPE
/* parameters of Numeric type */
#define NUMERIC_DSCALE_MASK	0x3FFF
#define NUMERIC_SIGN_MASK	0xC000
#define NUMERIC_POS         0x0000
#define NUMERIC_NEG         0x4000
#define NUMERIC_NAN         0xC000

#define NBASE				10000
#define HALF_NBASE			5000
#define DEC_DIGITS			4	/* decimal digits per NBASE digit */
#define MUL_GUARD_DIGITS    2	/* these are measured in NBASE digits */
#define DIV_GUARD_DIGITS	4
typedef int16				NumericDigit;
typedef struct NumericVar
{
	int			ndigits;	/* # of digits in digits[] - can be 0! */
	int			weight;		/* weight of first digit */
	int			sign;		/* NUMERIC_POS, NUMERIC_NEG, or NUMERIC_NAN */
	int			dscale;		/* display scale */
	NumericDigit *digits;	/* base-NBASE digits */
} NumericVar;

#ifndef __PG2ARROW__
#define NUMERIC_SHORT_SIGN_MASK			0x2000
#define NUMERIC_SHORT_DSCALE_MASK		0x1F80
#define NUMERIC_SHORT_DSCALE_SHIFT		7
#define NUMERIC_SHORT_WEIGHT_SIGN_MASK	0x0040
#define NUMERIC_SHORT_WEIGHT_MASK		0x003F

static void
init_var_from_num(NumericVar *nv, const char *addr, int sz)
{
	uint16		n_header = *((uint16 *)addr);

	/* NUMERIC_HEADER_IS_SHORT */
	if ((n_header & 0x8000) != 0)
	{
		/* short format */
		const struct {
			uint16	n_header;
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
			uint16      n_sign_dscale;  /* Sign + display scale */
			int16       n_weight;       /* Weight of 1st digit  */
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
#endif	/* !__PG2ARROW__ */

static size_t
put_decimal_value(SQLfield *column,
			const char *addr, int sz)
{
	size_t		row_index = column->nitems++;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int128));
	else
	{
		NumericVar		nv;
		int				scale = column->arrow_type.Decimal.scale;
		int128			value = 0;
		int				d, dig;
#ifdef __PG2ARROW__
		struct {
			uint16		ndigits;	/* number of digits */
			uint16		weight;		/* weight of first digit */
			uint16		sign;		/* NUMERIC_(POS|NEG|NAN) */
			uint16		dscale;		/* display scale */
			NumericDigit digits[FLEXIBLE_ARRAY_MEMBER];
		}  *rawdata = (void *)addr;
		nv.ndigits	= (int16)__ntoh16(rawdata->ndigits);
		nv.weight	= (int16)__ntoh16(rawdata->weight);
		nv.sign		= (int16)__ntoh16(rawdata->sign);
		nv.dscale	= (int16)__ntoh16(rawdata->dscale);
		nv.digits	= rawdata->digits;
#else	/* __PG2ARROW__ */
		init_var_from_num(&nv, addr, sz);
#endif	/* __PG2ARROW__ */
		if ((nv.sign & NUMERIC_SIGN_MASK) == NUMERIC_NAN)
			Elog("Decimal128 cannot map NaN in PostgreSQL Numeric");

		/* makes integer portion first */
		for (d=0; d <= nv.weight; d++)
		{
			dig = (d < nv.ndigits) ? __ntoh16(nv.digits[d]) : 0;
			if (dig < 0 || dig >= NBASE)
				Elog("Numeric digit is out of range: %d", (int)dig);
			value = NBASE * value + (int128)dig;
		}
		/* makes floating point portion if any */
		while (scale > 0)
		{
			dig = (d >= 0 && d < nv.ndigits) ? __ntoh16(nv.digits[d]) : 0;
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
	}
	return __buffer_usage_inline_type(column);
}
#endif	/* PG_INT128_TYPE */

/*
 * Date
 */
static inline size_t
__put_date_value_generic(SQLfield *column, const char *addr, int pgsql_sz,
						 int64 adjustment, int arrow_sz)
{
	size_t		row_index = column->nitems++;
	uint64		value;

	if (!addr)
		__put_inline_null_value(column, row_index, arrow_sz);
	else
	{
		assert(pgsql_sz == sizeof(DateADT));
		sql_buffer_setbit(&column->nullmap, row_index);
		value = __ntoh32(*((const DateADT *)addr));
		value += (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
		/*
		 * PostgreSQL native is ArrowDateUnit__Day.
		 * Compiler optimization will remove the if-block below by constant
		 * 'adjustment' argument.
		 */
		if (adjustment > 0)
			value *= adjustment;
		else if (adjustment < 0)
			value /= adjustment;
		sql_buffer_append(&column->values, &value, arrow_sz);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
__put_date_day_value(SQLfield *column, const char *addr, int sz)
{
	return __put_date_value_generic(column, addr, sz,
									0, sizeof(int32));
}

static size_t
__put_date_ms_value(SQLfield *column, const char *addr, int sz)
{
	return __put_date_value_generic(column, addr, sz,
									86400000L, sizeof(int64));
}

static size_t
put_date_value(SQLfield *column, const char *addr, int sz)
{
	/* validation checks only first call */
	switch (column->arrow_type.Date.unit)
	{
		case ArrowDateUnit__Day:
			column->put_value = __put_date_day_value;
			break;
		case ArrowDateUnit__MilliSecond:
			column->put_value = __put_date_ms_value;
			break;
		default:
			Elog("ArrowTypeDate has unknown unit (%d)",
				 column->arrow_type.Date.unit);
			break;
	}
	return column->put_value(column, addr, sz);
}

/*
 * Time
 */
static inline size_t
__put_time_value_generic(SQLfield *column, const char *addr, int pgsql_sz,
						 int64 adjustment, int arrow_sz)
{
	size_t		row_index = column->nitems++;
	TimeADT		value;

	if (!addr)
		__put_inline_null_value(column, row_index, arrow_sz);
	else
	{
		assert(pgsql_sz == sizeof(TimeADT));
		value = __ntoh64(*((const TimeADT *)addr));
		/*
		 * PostgreSQL native is ArrowTimeUnit__MicroSecond
		 * Compiler optimization will remove the if-block below by constant
		 * 'adjustment' argument.
		 */
		if (adjustment > 0)
			value *= adjustment;
		else if (adjustment < 0)
			value /= -adjustment;
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, arrow_sz);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
__put_time_sec_value(SQLfield *column, const char *addr, int sz)
{
	return __put_time_value_generic(column, addr, sz,
									-1000000L, sizeof(int32));
}

static size_t
__put_time_ms_value(SQLfield *column, const char *addr, int sz)
{
	return __put_time_value_generic(column, addr, sz,
									-1000L, sizeof(int32));
}

static size_t
__put_time_us_value(SQLfield *column, const char *addr, int sz)
{
	return __put_time_value_generic(column, addr, sz,
									0L, sizeof(int64));
}

static size_t
__put_time_ns_value(SQLfield *column, const char *addr, int sz)
{
	return __put_time_value_generic(column, addr, sz,
									1000L, sizeof(int64));
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
			break;
		case ArrowTimeUnit__MilliSecond:
			if (column->arrow_type.Time.bitWidth != 32)
				Elog("ArrowTypeTime has inconsistent bitWidth(%d) for [ms]",
					 column->arrow_type.Time.bitWidth);
			column->put_value = __put_time_ms_value;
			break;
		case ArrowTimeUnit__MicroSecond:
			if (column->arrow_type.Time.bitWidth != 64)
				Elog("ArrowTypeTime has inconsistent bitWidth(%d) for [us]",
					 column->arrow_type.Time.bitWidth);
			column->put_value = __put_time_us_value;
			break;
		case ArrowTimeUnit__NanoSecond:
			if (column->arrow_type.Time.bitWidth != 64)
				Elog("ArrowTypeTime has inconsistent bitWidth(%d) for [ns]",
					 column->arrow_type.Time.bitWidth);
			column->put_value = __put_time_ns_value;
		default:
			Elog("ArrowTypeTime has unknown unit (%d)",
				 column->arrow_type.Time.unit);
			break;
	}
	return column->put_value(column, addr, sz);
}

/*
 * Timestamp
 */
static inline size_t
__put_timestamp_value_generic(SQLfield *column,
							  const char *addr, int pgsql_sz,
							  int64 adjustment, int arrow_sz)
{
	size_t		row_index = column->nitems++;
	Timestamp	value;

	if (!addr)
		__put_inline_null_value(column, row_index, arrow_sz);
	else
	{
		assert(pgsql_sz == sizeof(Timestamp));
		value = __ntoh64(*((const Timestamp *)addr));
		/* convert PostgreSQL epoch to UNIX epoch */
		value += (POSTGRES_EPOCH_JDATE -
				  UNIX_EPOCH_JDATE) * USECS_PER_DAY;
		/*
		 * PostgreSQL native is ArrowTimeUnit__MicroSecond
		 * Compiler optimization will remove the if-block below by constant
		 * 'adjustment' argument.
		 */
		if (adjustment > 0)
			value *= adjustment;
		else if (adjustment < 0)
			value /= adjustment;
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, arrow_sz);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
__put_timestamp_sec_value(SQLfield *column, const char *addr, int sz)
{
	return __put_timestamp_value_generic(column, addr, sz,
										 -1000000L, sizeof(int64));
}

static size_t
__put_timestamp_ms_value(SQLfield *column, const char *addr, int sz)
{
	return __put_timestamp_value_generic(column, addr, sz,
										 -1000L, sizeof(int64));
}

static size_t
__put_timestamp_us_value(SQLfield *column, const char *addr, int sz)
{
	return __put_timestamp_value_generic(column, addr, sz,
										 0L, sizeof(int64));
}

static size_t
__put_timestamp_ns_value(SQLfield *column, const char *addr, int sz)
{
	return __put_timestamp_value_generic(column, addr, sz,
										 -1000L, sizeof(int64));
}

static size_t
put_timestamp_value(SQLfield *column, const char *addr, int sz)
{
	switch (column->arrow_type.Timestamp.unit)
	{
		case ArrowTimeUnit__Second:
			column->put_value = __put_timestamp_sec_value;
			break;
		case ArrowTimeUnit__MilliSecond:
			column->put_value = __put_timestamp_ms_value;
			break;
		case ArrowTimeUnit__MicroSecond:
			column->put_value = __put_timestamp_us_value;
			break;
		case ArrowTimeUnit__NanoSecond:
			column->put_value = __put_timestamp_ns_value;
			break;
		default:
			Elog("ArrowTypeTimestamp has unknown unit (%d)",
				column->arrow_type.Timestamp.unit);
			break;
	}
	return column->put_value(column, addr, sz);
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
		__put_inline_null_value(column, row_index, sizeof(uint32));
	else
	{
		uint32	m;

		assert(sz == sizeof(Interval));
		m = __ntoh32(((const Interval *)addr)->month);
		sql_buffer_append(&column->values, &m, sizeof(uint32));
	}
	return __buffer_usage_inline_type(column);
}

static size_t
__put_interval_day_time_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;

	if (!addr)
		__put_inline_null_value(column, row_index, 2 * sizeof(uint32));
	else
	{
		Interval	iv;
		uint32		value;

		assert(sz == sizeof(Interval));
		iv.time  = __ntoh64(((const Interval *)addr)->time);
		iv.day   = __ntoh32(((const Interval *)addr)->day);
		iv.month = __ntoh32(((const Interval *)addr)->month);

		/*
		 * Unit of PostgreSQL Interval is micro-seconds. Arrow Interval::time
		 * is represented as a pair of elapsed days and milli-seconds; needs
		 * to be adjusted.
		 */
		value = iv.month + DAYS_PER_MONTH * iv.day;
		sql_buffer_append(&column->values, &value, sizeof(uint32));
		value = iv.time / 1000;
		sql_buffer_append(&column->values, &value, sizeof(uint32));
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
			Elog("columnibute \"%s\" has unknown Arrow::Interval.unit(%d)",
				 sql_field->field_name,
				 sql_field->arrow_type.Interval.unit);
			break;
	}
	return sql_field->put_value(sql_field, addr, sz);
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
		sql_buffer_append_zero(&column->values, sizeof(uint32));
	if (!addr)
	{
		column->nullcount++;
		sql_buffer_clrbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values,
						  &column->extra.usage, sizeof(uint32));
	}
	else
	{
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->extra, addr, sz);
		sql_buffer_append(&column->values,
						  &column->extra.usage, sizeof(uint32));
	}
	return __buffer_usage_varlena_type(column);
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
		sql_buffer_append_zero(&column->values, sizeof(uint32));
	if (!addr)
	{
		column->nullcount++;
		sql_buffer_clrbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &element->nitems, sizeof(int32));
	}
	else
	{
#ifdef __PG2ARROW__
		struct {
			int32		ndim;
			int32		hasnull;
			int32		element_type;
			struct {
				int32	sz;
				int32	lb;
			} dim[FLEXIBLE_ARRAY_MEMBER];
		}  *rawdata = (void *) addr;
		int32		ndim = __ntoh32(rawdata->ndim);
		//int32		hasnull = __ntoh32(rawdata->hasnull);
		Oid			element_typeid = __ntoh32(rawdata->element_type);
		size_t		i, nitems = 1;
		int			item_sz;
		char	   *pos;

		if (element_typeid != element->sql_type.pgsql.typeid)
			Elog("PostgreSQL array type mismatch");
		if (ndim < 1)
			Elog("Invalid dimension size of PostgreSQL Array (ndim=%d)", ndim);
		for (i=0; i < ndim; i++)
			nitems *= __ntoh32(rawdata->dim[i].sz);

		pos = (char *)&rawdata->dim[ndim];
		for (i=0; i < nitems; i++)
		{
			if (pos + sizeof(int32) > addr + sz)
				Elog("out of range - binary array has corruption");
			item_sz = __ntoh32(*((int32 *)pos));
			pos += sizeof(int32);
			if (item_sz < 0)
				arrowFieldPutValue(element, NULL, 0);
			else
			{
				arrowFieldPutValue(element, pos, item_sz);
				pos += item_sz;
			}
		}
#else	/* __PG2ARROW__ */
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
#endif	/* !__PG2ARROW__ */
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &element->nitems, sizeof(int32));
	}
	return __buffer_usage_inline_type(column) + element->__curr_usage__;
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
			usage += arrowFieldPutValue(&column->subfields[j], NULL, 0);
		}
	}
	else
	{
#ifdef __PG2ARROW__
		const char *pos = addr;
		int			j, nvalids;

		if (sz < sizeof(uint32))
			Elog("binary composite record corruption");
		nvalids = __ntoh32(*((const int *)pos));
		pos += sizeof(int);
		for (j=0; j < column->nfields; j++)
		{
			SQLfield *sub_field = &column->subfields[j];
			Oid		typeid;
			int32	len;

			if (j >= nvalids)
			{
				usage += arrowFieldPutValue(sub_field, NULL, 0);
				continue;
			}
			if ((pos - addr) + sizeof(Oid) + sizeof(int) > sz)
				Elog("binary composite record corruption");
			typeid = __ntoh32(*((Oid *)pos));
			pos += sizeof(Oid);
			if (sub_field->sql_type.pgsql.typeid != typeid)
				Elog("composite subtype mismatch");
			len = __ntoh32(*((int *)pos));
			pos += sizeof(int32);
			if (len == -1)
			{
				usage += arrowFieldPutValue(sub_field, NULL, 0);
			}
			else
			{
				if ((pos - addr) + len > sz)
					Elog("binary composite record corruption");
				usage += arrowFieldPutValue(sub_field, pos, len);
				pos += len;
			}
			assert(column->nitems == sub_field->nitems);
		}
#else	/* __PG2ARROW__ */
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

			if (j >= nvalids || (nullmap && att_isnull(j, nullmap)))
			{
				usage += arrowFieldPutValue(field, NULL, 0);
			}
			else if (field->sql_type.pgsql.typbyval)
			{
				Assert(field->sql_type.pgsql.typlen > 0 &&
					   field->sql_type.pgsql.typlen <= sizeof(Datum));
				usage += arrowFieldPutValue(field, base + off,
											field->sql_type.pgsql.typlen);
				off = TYPEALIGN(field->sql_type.pgsql.typalign,
								off + field->sql_type.pgsql.typlen);
			}
			else if (field->sql_type.pgsql.typlen == -1)
			{
				int		vl_len = VARSIZE_ANY_EXHDR(base + off);
				char   *vl_dat = VARDATA_ANY(base + off);

				usage += arrowFieldPutValue(field, vl_dat, vl_len);
				off = TYPEALIGN(field->sql_type.pgsql.typalign,
								off + VARSIZE_ANY(base + off));
			}
			else
			{
				Elog("Bug? sub-field '%s' of column '%s' has unsupported type",
					 field->field_name,
					 column->field_name);
			}
			assert(column->nitems == field->nitems);
		}
#endif	/* !__PG2ARROW__ */
		sql_buffer_setbit(&column->nullmap, row_index);
	}
	if (column->nullcount > 0)
		usage += ARROWALIGN(column->nullmap.usage);
	return usage;
}

static size_t
put_dictionary_value(SQLfield *column,
					 const char *addr, int sz)
{
	size_t		row_index = column->nitems++;

	if (!addr)
	{
		column->nullcount++;
		sql_buffer_clrbit(&column->nullmap, row_index);
		sql_buffer_append_zero(&column->values, sizeof(uint32));
	}
	else
	{
		SQLdictionary *enumdict = column->enumdict;
		hashItem   *hitem;
		uint32		hash;

		hash = hash_any((const unsigned char *)addr, sz);
		for (hitem = enumdict->hslots[hash % enumdict->nslots];
			 hitem != NULL;
			 hitem = hitem->next)
		{
			if (hitem->hash == hash &&
				hitem->label_len == sz &&
				memcmp(hitem->label, addr, sz) == 0)
				break;
		}
		if (!hitem)
			Elog("Enum label was not found in pg_enum result");

		sql_buffer_setbit(&column->nullmap, row_index);
        sql_buffer_append(&column->values,  &hitem->index, sizeof(int32));
	}
	return __buffer_usage_inline_type(column);
}

/* ----------------------------------------------------------------
 *
 * Gut value handler for each data types
 *
 * ----------------------------------------------------------------
 */
static inline bool
__get_value_isnull(SQLfield *column, size_t row_index)
{
	if (row_index < column->nitems)
	{
		size_t	index = (row_index >> 3);
		uint32	mask = (1U << (row_index & 7));

		assert(index < column->nullmap.usage);
		if ((column->nullmap.data[index] & mask) != 0)
			return false;
	}
	return true;
}

static inline void *
__get_inline_addr(SQLfield *column, size_t row_index, int unitsz, bool *isnull)
{
	if (!__get_value_isnull(column, row_index))
	{
		assert(unitsz * (row_index + 1) <= column->values.usage);
		*isnull = false;
		return column->values.data + unitsz * row_index;
	}
	*isnull = true;
	return NULL;
}

static Datum
get_bool_value(SQLfield *column, size_t row_index, bool *isnull)
{
	if (row_index < column->nitems)
	{
		size_t	index = (row_index >> 3);
		uint32	mask = (1U << (row_index & 7));

		assert(index < column->nullmap.usage &&
			   index < column->values.usage);
		if ((column->nullmap.data[index] & mask) != 0)
		{
			*isnull = false;
			if ((column->values.data[index] & mask) != 0)
				return true;
			else
				return false;
		}
	}
	*isnull = true;
	return 0;
}

static Datum
get_int8_value(SQLfield *column, size_t row_index, bool *isnull)
{
	int8   *val = __get_inline_addr(column, row_index, sizeof(int8), isnull);

	if (val)
		return *val;
	return 0;
}

static Datum
get_int16_value(SQLfield *column, size_t row_index, bool *isnull)
{
	int16  *val = __get_inline_addr(column, row_index, sizeof(int16), isnull);

	if (val)
		return *val;
	return 0;
}

static Datum
get_int32_value(SQLfield *column, size_t row_index, bool *isnull)
{
	int32  *val = __get_inline_addr(column, row_index, sizeof(int32), isnull);

	if (val)
		return *val;
	return 0;
}

static Datum
get_int64_value(SQLfield *column, size_t row_index, bool *isnull)
{
	int64  *val = __get_inline_addr(column, row_index, sizeof(int32), isnull);

	if (val)
		return *val;
	return 0;
}

static Datum
get_decimal_value(SQLfield *column, size_t row_index, bool *isnull)
{
#ifdef PG_INT128_TYPE
	int128	   *addr, value;
	struct {
		int32	vl_len_;
		uint16	n_sign_dscale;
		int16	n_weight;
		NumericDigit n_data[16];	/* sufficient for 128bit value */
	} nm;
	int			scale = column->arrow_type.Decimal.scale;
	int			i, len, sz;
	bool		is_negative = false;
	void	   *result;

	addr = __get_inline_addr(column, row_index, sizeof(int128), isnull);
	if (!addr)
		return 0;

	value = *addr;
	if (value < 0)
	{
		is_negative = false;
		value = -value;
	}

	switch (scale % 4)
	{
		case 0:
			break;
		case 1:
			scale += 3;
			value *= 1000L;
			break;
		case 2:
			scale += 2;
			value *= 100L;
			break;
		case 3:
			scale += 1;
			value *= 10L;
			break;
	}

	i = 16;
	memset(&nm, 0, sizeof(nm));
	for (i=15, len=0; value != 0; i++)
	{
		assert(i >= 0);
		nm.n_data[i] = value % NBASE;
		value = value / NBASE;
		len++;
	}
	memmove(nm.n_data, nm.n_data + i, sizeof(NumericDigit) * len);
	nm.n_weight = len;
	nm.n_sign_dscale = (NUMERIC_DSCALE_MASK & (scale / DEC_DIGITS));
	if (is_negative)
		nm.n_sign_dscale |= NUMERIC_NEG;
	sz = (char *)&nm.n_data[len] - (char *)&nm;
	SET_VARSIZE(&nm, sz);

	result = palloc(sz);
	memcpy(result, &nm, sz);

	return (Datum)result;
#else
#error "Int128 must be enabled for Arrow::Decimal support"
#endif
}

static Datum
__get_date_day_value(SQLfield *column, size_t row_index, bool *isnull)
{
	int32  *pos = __get_inline_addr(column, row_index, sizeof(int32), isnull);

	if (pos)
		return *pos;
	return 0;
}

static Datum
__get_date_ms_value(SQLfield *column, size_t row_index, bool *isnull)
{
	int64  *pos = __get_inline_addr(column, row_index, sizeof(int32), isnull);

	if (pos)
		return (DateADT)(*pos / 86400000L);
	return 0;
}

static Datum
get_date_value(SQLfield *column, size_t row_index, bool *isnull)
{
	/* validation checks only first call */
	switch (column->arrow_type.Date.unit)
	{
		case ArrowDateUnit__Day:
			column->get_value = __get_date_day_value;
			break;
		case ArrowDateUnit__MilliSecond:
			column->get_value = __get_date_ms_value;
			break;
		default:
			Elog("ArrowTypeDate has unknown unit (%d)",
				 column->arrow_type.Date.unit);
			break;
	}
	return column->get_value(column, row_index, isnull);
}

static Datum
__get_time_sec_value(SQLfield *column, size_t row_index, bool *isnull)
{
	int32  *pos = __get_inline_addr(column, row_index, sizeof(int32), isnull);

	if (pos)
		return ((int64)(*pos)) * 1000000L;
	return 0;
}

static Datum
__get_time_ms_value(SQLfield *column, size_t row_index, bool *isnull)
{
	int32  *pos = __get_inline_addr(column, row_index, sizeof(int32), isnull);

	if (pos)
		return ((int64)(*pos)) * 1000L;
	return 0;
}

static Datum
__get_time_us_value(SQLfield *column, size_t row_index, bool *isnull)
{
	int64  *pos = __get_inline_addr(column, row_index, sizeof(int64), isnull);

	if (pos)
		return *pos;
	return 0;
}

static Datum
__get_time_ns_value(SQLfield *column, size_t row_index, bool *isnull)
{
	int64  *pos = __get_inline_addr(column, row_index, sizeof(int64), isnull);

	if (pos)
		return *pos / 1000L;
	return 0;
}

static Datum
get_time_value(SQLfield *column, size_t row_index, bool *isnull)
{
	switch (column->arrow_type.Time.unit)
	{
		case ArrowTimeUnit__Second:
			if (column->arrow_type.Time.bitWidth != 32)
				Elog("ArrowTypeTime has inconsistent bitWidth(%d) for [sec]",
					 column->arrow_type.Time.bitWidth);
			column->get_value = __get_time_sec_value;
			break;
		case ArrowTimeUnit__MilliSecond:
			if (column->arrow_type.Time.bitWidth != 32)
				Elog("ArrowTypeTime has inconsistent bitWidth(%d) for [ms]",
					 column->arrow_type.Time.bitWidth);
			column->get_value = __get_time_ms_value;
			break;
		case ArrowTimeUnit__MicroSecond:
			if (column->arrow_type.Time.bitWidth != 64)
				Elog("ArrowTypeTime has inconsistent bitWidth(%d) for [us]",
					 column->arrow_type.Time.bitWidth);
			column->get_value = __get_time_us_value;
			break;
		case ArrowTimeUnit__NanoSecond:
			if (column->arrow_type.Time.bitWidth != 64)
				Elog("ArrowTypeTime has inconsistent bitWidth(%d) for [ns]",
					 column->arrow_type.Time.bitWidth);
			column->get_value = __get_time_ns_value;
		default:
			Elog("ArrowTypeTime has unknown unit (%d)",
				 column->arrow_type.Time.unit);
			break;
	}
	return column->get_value(column, row_index, isnull);
}

static Datum
__get_timestamp_sec_value(SQLfield *column, size_t row_index, bool *isnull)
{
	int64  *pos = __get_inline_addr(column, row_index, sizeof(int64), isnull);

	if (pos)
		return *pos * 1000000L;
    return 0;
}

static Datum
__get_timestamp_ms_value(SQLfield *column, size_t row_index, bool *isnull)
{
	int64  *pos = __get_inline_addr(column, row_index, sizeof(int64), isnull);

	if (pos)
		return *pos * 1000L;
	return 0;
}

static Datum
__get_timestamp_us_value(SQLfield *column, size_t row_index, bool *isnull)
{
	int64  *pos = __get_inline_addr(column, row_index, sizeof(int64), isnull);

	if (pos)
		return *pos;
	return 0;
}

static Datum
__get_timestamp_ns_value(SQLfield *column, size_t row_index, bool *isnull)
{
	int64  *pos = __get_inline_addr(column, row_index, sizeof(int64), isnull);

	if (pos)
		return *pos / 1000L;
	return 0;
}

static Datum
get_timestamp_value(SQLfield *column, size_t row_index, bool *isnull)
{
	switch (column->arrow_type.Timestamp.unit)
	{
		case ArrowTimeUnit__Second:
			column->get_value = __get_timestamp_sec_value;
			break;
		case ArrowTimeUnit__MilliSecond:
			column->get_value = __get_timestamp_ms_value;
			break;
		case ArrowTimeUnit__MicroSecond:
			column->get_value = __get_timestamp_us_value;
			break;
		case ArrowTimeUnit__NanoSecond:
			column->get_value = __get_timestamp_ns_value;
			break;
		default:
			Elog("ArrowTypeTimestamp has unknown unit (%d)",
				 column->arrow_type.Timestamp.unit);
			break;
	}
	return column->get_value(column, row_index, isnull);
}

static Datum
__get_interval_year_month_value(SQLfield *column,
								size_t row_index, bool *isnull)
{
	Interval   *iv;
	uint32	   *pos;

	pos = __get_inline_addr(column, row_index, sizeof(uint32), isnull);
	if (!pos)
		return 0;
	iv = palloc0(sizeof(Interval));
	iv->month = *pos;

	return (Datum)iv;
}

static Datum
__get_interval_day_time_value(SQLfield *column,
							  size_t row_index, bool *isnull)
{
	Interval   *iv;
	uint32	   *pos;
	uint32		l, h;

	pos = __get_inline_addr(column, row_index, sizeof(uint64), isnull);
	if (!pos)
		return 0;
	l = pos[0];
	h = pos[1];

	iv = palloc0(sizeof(Interval));
	iv->time  = h * 1000;
	iv->day   = l / DAYS_PER_MONTH;
	iv->month = l % DAYS_PER_MONTH;

	return (Datum)iv;
}

static Datum
get_interval_value(SQLfield *column, size_t row_index, bool *isnull)
{
	switch (column->arrow_type.Interval.unit)
	{
		case ArrowIntervalUnit__Year_Month:
			column->get_value = __get_interval_year_month_value;
			break;
		case ArrowIntervalUnit__Day_Time:
			column->get_value = __get_interval_day_time_value;
			break;
		default:
			Elog("column \"%s\" has unknown Arrow::Interval.unit(%d)",
				 column->field_name,
				 column->arrow_type.Interval.unit);
			break;
	}
	return column->get_value(column, row_index, isnull);
}

static Datum
get_variable_value(SQLfield *column, size_t row_index, bool *isnull)
{
	uint32	   *pos;
	uint32		start, end, sz;
	char	   *result;

	pos = __get_inline_addr(column, row_index, sizeof(int32), isnull);
	if (!pos)
		return 0;
	start = *pos++;
	end   = *pos++;
	assert(start <= end && end <= column->extra.usage);
	sz = end - start;
	result = palloc(VARHDRSZ + sz);
	memcpy(result + VARHDRSZ, column->extra.data + start, sz);
	SET_VARSIZE(result, VARHDRSZ + sz);

	return (Datum)result;
}

static Datum
get_bpchar_value(SQLfield *column, size_t row_index, bool *isnull)
{
	int		unitsz = column->arrow_type.FixedSizeBinary.byteWidth;
	char   *pos, *result;

	pos = __get_inline_addr(column, row_index, unitsz, isnull);
	if (!pos)
		return 0;
	result = palloc(VARHDRSZ + unitsz);
	memcpy(result + VARHDRSZ, pos, unitsz);
	SET_VARSIZE(result, VARHDRSZ + unitsz);

	return (Datum)result;
}

static Datum
get_array_value(SQLfield *column, size_t row_index, bool *p_isnull)
{
#ifndef __PG2ARROW__
	uint32	   *pos;

	pos = __get_inline_addr(column, row_index, sizeof(int32), p_isnull);
	if (pos)
	{
		SQLfield   *element = column->element;
		uint32		l = pos[0];
		uint32		h = pos[1];
		uint32		i, nitems;
		Datum	   *values;
		bool	   *isnull;
		bool		hasnull = false;
		uint8	   *nullmap = NULL;
		uint32		mask = 1;
		size_t		usage = 0;
		struct {
			int32	vl_len_;
			int		ndim;
			int		dataoffset;
			Oid		elemtype;
			int		dim1;
			int		lbound1;
			char	data[FLEXIBLE_ARRAY_MEMBER];
		} *r = NULL;

		/* fetch values from the element buffer */
		assert(l <= h);
		nitems = h - l;
		values = alloca(sizeof(Datum) * nitems);
		isnull = alloca(sizeof(bool) * nitems);
		for (i=0; i < nitems; i++)
		{
			values[i] = arrowFieldGetValue(element, l+i, &isnull[i]);
			if (isnull[i])
				hasnull = true;
			else
			{
				if (element->sql_type.pgsql.typlen > 0)
					usage += element->sql_type.pgsql.typlen;
				else if (element->sql_type.pgsql.typlen == -1)
					usage += VARSIZE_ANY(values[i]);
				usage = TYPEALIGN(element->sql_type.pgsql.typalign, usage);
			}
		}
		/* build an ArrayType object */
		if (hasnull)
			usage += MAXALIGN(BITMAPLEN(nitems));
		r = palloc((char *)(&r->data[usage]) - (char *)r);
		r->ndim = 1;
		r->dataoffset = (hasnull ? MAXALIGN((nitems + 7) / 8) : 0);
		r->elemtype = element->sql_type.pgsql.typeid;
		r->dim1 = nitems;
		r->lbound1 = 1;

		if (!hasnull)
		{
			nullmap = NULL;
			usage = 0;
		}
		else
		{
			nullmap = (uint8 *)r->data;
			usage = MAXALIGN(BITMAPLEN(nitems));
		}

		for (i=0; i < nitems; i++)
		{
			if (isnull[i])
				*nullmap &= ~mask;
			else
			{
				if (nullmap)
					*nullmap |= mask;
				if (element->sql_type.pgsql.typbyval)
				{
					assert(element->sql_type.pgsql.typlen > 0 &&
						   element->sql_type.pgsql.typlen <= sizeof(Datum));
					memcpy(r->data + usage, &values[i],
						   element->sql_type.pgsql.typlen);
					usage += element->sql_type.pgsql.typlen;
				}
				else if (element->sql_type.pgsql.typlen > 0)
				{
					memcpy(r->data + usage, (char *)values[i],
						   element->sql_type.pgsql.typlen);
					usage += element->sql_type.pgsql.typlen;
				}
				else if (element->sql_type.pgsql.typlen == -1)
				{
					int		__len = VARSIZE_ANY(values[i]);

					memcpy(r->data + usage, (char *)values[i], __len);
					usage += __len;
				}
				else
					Elog("Bug? unsupported element type");
				usage = TYPEALIGN(element->sql_type.pgsql.typalign, usage);

				if (nullmap)
				{
					mask <<= 1;
					if (mask == 0x100)
					{
						mask = 1;
						nullmap++;
					}
				}
			}
		}
		SET_VARSIZE(r, (char *)&r->data[usage] - (char *)r);
		return (Datum) r;
	}
#else		/* __PG2ARROW__ */
	Elog("get_array_value is enabled at only server side");
#endif		/* __PG2ARROW__ */
	*p_isnull = true;
	return 0;
}

static Datum
get_composite_value(SQLfield *column, size_t row_index, bool *p_isnull)
{
#ifndef __PG2ARROW__
	if (!__get_value_isnull(column, row_index))
	{
		int			j, nfields = column->nfields;
		Datum	   *values = alloca(sizeof(Datum) * nfields);
		bool	   *isnull = alloca(sizeof(bool) * nfields);
		bool		hasnull = false;
		size_t		datalen = 0;
		size_t		hoff;
		uint8	   *nullmap = NULL;
		uint32		mask = 1U;
		HeapTupleHeader htup;

		for (j=0; j < nfields; j++)
		{
			SQLfield *field = &column->subfields[j];

			values[j] = arrowFieldGetValue(field, row_index, &isnull[j]);
			if (isnull[j])
				hasnull = true;
			else
			{
				if (field->sql_type.pgsql.typlen > 0)
					datalen += field->sql_type.pgsql.typlen;
				else if (field->sql_type.pgsql.typlen == -1)
					datalen += VARSIZE_ANY(values[j]);
				else
					Elog("unsupported data type");
				datalen = TYPEALIGN(field->sql_type.pgsql.typalign, datalen);
			}
		}
		hoff = offsetof(HeapTupleHeaderData, t_bits);
		if (hasnull)
			hoff += BITMAPLEN(nfields);
		hoff = MAXALIGN(hoff);

		htup = palloc(hoff + datalen);
		memset(htup, 0, hoff);
		HeapTupleHeaderSetNatts(htup, nfields);
		htup->t_hoff = hoff;
		if (hasnull)
			nullmap = htup->t_bits;
		for (j=0; j < nfields; j++)
		{
			SQLfield   *field = &column->subfields[j];

			if (isnull[j])
			{
				*nullmap &= ~mask;
				htup->t_infomask |= HEAP_HASNULL;
			}
			else
			{
				if (nullmap)
					*nullmap |= mask;
				if (field->sql_type.pgsql.typbyval)
				{
					memcpy((char *)htup + hoff, &values[j],
						   field->sql_type.pgsql.typlen);
					hoff += field->sql_type.pgsql.typlen;
				}
				else if (field->sql_type.pgsql.typlen > 0)
				{
					memcpy((char *)htup + hoff, (char *)values[j],
						   field->sql_type.pgsql.typlen);
					hoff += field->sql_type.pgsql.typlen;
				}
				else if (field->sql_type.pgsql.typlen == -1)
				{
					int		vl_len = VARSIZE_ANY(values[j]);

					memcpy((char *)htup + hoff, (char *)values[j], vl_len);
					hoff += vl_len;
					htup->t_infomask |= HEAP_HASVARWIDTH;
				}
				else
				{
					Elog("Bug? unsupported data type");
				}
				hoff = TYPEALIGN(field->sql_type.pgsql.typalign, hoff);
			}
		}
		HeapTupleHeaderSetDatumLength(htup, hoff);
		HeapTupleHeaderSetTypeId(htup, column->sql_type.pgsql.typeid);
		HeapTupleHeaderSetTypMod(htup, column->sql_type.pgsql.typmod);
		ItemPointerSetInvalid(&htup->t_ctid);

		*p_isnull = false;
		return (Datum)htup;
	}
#else		/* __PG2ARROW__ */
	Elog("get_composite_value is enabled at only server-side");
#endif		/* __PG2ARROW__ */
	*p_isnull = true;
	return 0;
}

static Datum
get_dictionary_value(SQLfield *column, size_t row_index, bool *isnull)
{
	Elog("Enum type is not supported yet");
	return 0;
}

/* ----------------------------------------------------------------
 *
 * setup handler for each data types
 *
 * ----------------------------------------------------------------
 */
static int
assignArrowTypeInt(SQLfield *column, bool is_signed)
{
	initArrowNode(&column->arrow_type, Int);
	column->arrow_type.Int.is_signed = is_signed;
	switch (column->sql_type.pgsql.typlen)
	{
		case sizeof(char):
			column->arrow_type.Int.bitWidth = 8;
			column->arrow_typename = (is_signed ? "Int8" : "Uint8");
			column->put_value = put_int8_value;
			column->get_value = get_int8_value;
			break;
		case sizeof(short):
			column->arrow_type.Int.bitWidth = 16;
			column->arrow_typename = (is_signed ? "Int16" : "Uint16");
			column->put_value = put_int16_value;
			column->get_value = get_int16_value;
			break;
		case sizeof(int):
			column->arrow_type.Int.bitWidth = 32;
			column->arrow_typename = (is_signed ? "Int32" : "Uint32");
			column->put_value = put_int32_value;
			column->get_value = get_int32_value;
			break;
		case sizeof(long):
			column->arrow_type.Int.bitWidth = 64;
			column->arrow_typename = (is_signed ? "Int64" : "Uint64");
			column->put_value = put_int64_value;
			column->get_value = get_int64_value;
			break;
		default:
			Elog("unsupported Int width: %d",
				 column->sql_type.pgsql.typlen);
			break;
	}
	return 2;		/* null map + values */
}

static int
assignArrowTypeFloatingPoint(SQLfield *column)
{
	initArrowNode(&column->arrow_type, FloatingPoint);
	switch (column->sql_type.pgsql.typlen)
	{
		case sizeof(short):		/* half */
			column->arrow_type.FloatingPoint.precision
				= ArrowPrecision__Half;
			column->arrow_typename = "Float16";
			column->put_value = put_float16_value;
			column->get_value = get_int16_value;
			break;
		case sizeof(float):
			column->arrow_type.FloatingPoint.precision
				= ArrowPrecision__Single;
			column->arrow_typename = "Float32";
			column->put_value = put_float32_value;
			column->get_value = get_int32_value;
			break;
		case sizeof(double):
			column->arrow_type.FloatingPoint.precision
				= ArrowPrecision__Double;
			column->arrow_typename = "Float64";
			column->put_value = put_float64_value;
			column->get_value = get_int64_value;
			break;
		default:
			Elog("unsupported floating point width: %d",
				 column->sql_type.pgsql.typlen);
			break;
	}
	return 2;		/* nullmap + values */
}

static int
assignArrowTypeBinary(SQLfield *column)
{
	initArrowNode(&column->arrow_type, Binary);
	column->arrow_typename	= "Binary";
	column->put_value		= put_variable_value;
	column->get_value		= get_variable_value;

	return 3;		/* nullmap + index + extra */
}

static int
assignArrowTypeUtf8(SQLfield *column)
{
	initArrowNode(&column->arrow_type, Utf8);
	column->arrow_typename	= "Utf8";
	column->put_value		= put_variable_value;
	column->get_value		= get_variable_value;

	return 3;		/* nullmap + index + extra */
}

static int
assignArrowTypeBpchar(SQLfield *column)
{
	if (column->sql_type.pgsql.typmod <= VARHDRSZ)
		Elog("unexpected Bpchar definition (typmod=%d)",
			 column->sql_type.pgsql.typmod);

	initArrowNode(&column->arrow_type, FixedSizeBinary);
	column->arrow_type.FixedSizeBinary.byteWidth
		= column->sql_type.pgsql.typmod - VARHDRSZ;
	column->arrow_typename	= "FixedSizeBinary";
	column->put_value		= put_bpchar_value;
	column->get_value		= get_bpchar_value;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeBool(SQLfield *column)
{
	initArrowNode(&column->arrow_type, Bool);
	column->arrow_typename	= "Bool";
	column->put_value		= put_bool_value;
	column->get_value		= get_bool_value;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeDecimal(SQLfield *column)
{
#ifdef PG_INT128_TYPE
	int		typmod			= column->sql_type.pgsql.typmod;
	int		precision		= 30;	/* default, if typmod == -1 */
	int		scale			=  8;	/* default, if typmod == -1 */

	if (typmod >= VARHDRSZ)
	{
		typmod -= VARHDRSZ;
		precision = (typmod >> 16) & 0xffff;
		scale = (typmod & 0xffff);
	}
	initArrowNode(&column->arrow_type, Decimal);
	column->arrow_type.Decimal.precision = precision;
	column->arrow_type.Decimal.scale = scale;
	column->arrow_typename	= "Decimal";
	column->put_value		= put_decimal_value;
	column->get_value		= get_decimal_value;
#else
#error "Int128 must be enabled for Arrow::Decimal support"
#endif
	return 2;		/* nullmap + values */
}

static int
assignArrowTypeDate(SQLfield *column)
{
	initArrowNode(&column->arrow_type, Date);
	column->arrow_type.Date.unit = ArrowDateUnit__Day;
	column->arrow_typename	= "Date";
	column->put_value		= put_date_value;
	column->get_value		= get_date_value;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeTime(SQLfield *column)
{
	initArrowNode(&column->arrow_type, Time);
	column->arrow_type.Time.unit = ArrowTimeUnit__MicroSecond;
	column->arrow_type.Time.bitWidth = 64;
	column->arrow_typename	= "Time";
	column->put_value		= put_time_value;
	column->get_value		= get_time_value;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeTimestamp(SQLfield *column)
{
	initArrowNode(&column->arrow_type, Timestamp);
	column->arrow_type.Timestamp.unit = ArrowTimeUnit__MicroSecond;
	column->arrow_typename	= "Timestamp";
	column->put_value		= put_timestamp_value;
	column->get_value		= get_timestamp_value;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeInterval(SQLfield *column)
{
	initArrowNode(&column->arrow_type, Interval);
	column->arrow_type.Interval.unit = ArrowIntervalUnit__Day_Time;
	column->arrow_typename	= "Interval";
	column->put_value       = put_interval_value;
	column->get_value		= get_interval_value;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeList(SQLfield *column)
{
	initArrowNode(&column->arrow_type, List);
	column->arrow_typename	= "List";
	column->put_value		= put_array_value;
	column->get_value		= get_array_value;

	return 2;		/* nullmap + offset vector */
}

static int
assignArrowTypeStruct(SQLfield *column)
{
	initArrowNode(&column->arrow_type, Struct);
	column->arrow_typename	= "Struct";
	column->put_value		= put_composite_value;
	column->get_value		= get_composite_value;

	return 1;	/* only nullmap */
}

static int
assignArrowTypeDictionary(SQLfield *column)
{
	initArrowNode(&column->arrow_type, Utf8);
	column->arrow_typename	= psprintf("Enum; dictionary=%u",
									   column->sql_type.pgsql.typeid);
	column->put_value		= put_dictionary_value;
	column->get_value		= get_dictionary_value;

	return 2;	/* nullmap + values */
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
					 Oid typelemid)

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

	if (typelemid != 0)
	{
		/* array type */
		if (typlen != -1)
			Elog("Bug? array type is not varlena (typlen != -1)");
		return assignArrowTypeList(column);
	}
	else if (typrelid != 0)
	{
		/* composite type */
		return assignArrowTypeStruct(column);
	}
	else if (typtype == 'e')
	{
		/* enum type */
		return assignArrowTypeDictionary(column);
	}
	else if (strcmp(typnamespace, "pg_catalog") == 0)
	{
		/* well known built-in data types? */
		if (strcmp(typname, "bool") == 0)
		{
			return assignArrowTypeBool(column);
		}
		else if (strcmp(typname, "int2") == 0 ||
				 strcmp(typname, "int4") == 0 ||
				 strcmp(typname, "int8") == 0)
		{
			return assignArrowTypeInt(column, true);
		}
		else if (strcmp(typname, "float2") == 0 ||
				 strcmp(typname, "float4") == 0 ||
				 strcmp(typname, "float8") == 0)
		{
			return assignArrowTypeFloatingPoint(column);
		}
		else if (strcmp(typname, "date") == 0)
		{
			return assignArrowTypeDate(column);
		}
		else if (strcmp(typname, "time") == 0)
		{
			return assignArrowTypeTime(column);
		}
		else if (strcmp(typname, "timestamp") == 0 ||
				 strcmp(typname, "timestamptz") == 0)
		{
			return assignArrowTypeTimestamp(column);
		}
		else if (strcmp(typname, "interval") == 0)
		{
			return assignArrowTypeInterval(column);
		}
		else if (strcmp(typname, "text") == 0 ||
				 strcmp(typname, "varchar") == 0)
		{
			return assignArrowTypeUtf8(column);
		}
		else if (strcmp(typname, "bpchar") == 0)
		{
			return assignArrowTypeBpchar(column);
		}
		else if (strcmp(typname, "numeric") == 0)
		{
			return assignArrowTypeDecimal(column);
		}
	}
	/* elsewhere, we save the values just bunch of binary data */
	if (typlen > 0)
	{
		if (typlen == sizeof(char) ||
			typlen == sizeof(short) ||
			typlen == sizeof(int) ||
			typlen == sizeof(double))
			return assignArrowTypeInt(column, false);
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
		return assignArrowTypeBinary(column);
	}
	Elog("PostgreSQL type: '%s' is not supported", typname);
}
