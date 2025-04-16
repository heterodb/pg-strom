/*
 * mysql_client.c - MySQL specific portion for mysql2arrow command
 *
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <mysql/mysql.h>
#include "sql2arrow.h"
#include <ctype.h>
#include <limits.h>
#include <stdarg.h>

/* static variables */
static char	   *mysql_timezone = NULL;
static int		__exp10[] = {1,
							 10,
							 100,
							 1000,
							 10000,
							 100000,
							 1000000,
							 10000000,
							 100000000,
							 1000000000 };

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

/*
 * put values handlers
 */
static inline void
__put_inline_null_value(SQLfield *column, size_t row_index, int sz)
{
	column->nullcount++;
	sql_buffer_clrbit(&column->nullmap, row_index);
	sql_buffer_append_zero(&column->values, sz);
}

static size_t
put_int8_value(SQLfield *column, const char *addr, int sz)
{
	size_t	row_index = column->nitems++;
	char   *end;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int8_t));
	else
	{
		int64_t		ival = strtol(addr, &end, 10);

		if (*end != '\0' || errno != 0)
			Elog("value '%s' is not valid", addr);
		if (ival < SCHAR_MIN || ival > SCHAR_MAX)
			Elog("value '%s' is out of range for Arrow::Int8", addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &ival, sizeof(int8_t));
		STAT_UPDATES(column, i8, ival);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_uint8_value(SQLfield *column, const char *addr, int sz)
{
	size_t	row_index = column->nitems++;
	char   *end;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int8_t));
	else
	{
		uint64_t	ival = strtol(addr, &end, 10);

		if (*end != '\0' || errno != 0)
			Elog("value '%s' is not valid", addr);
		if (ival > UCHAR_MAX)
			Elog("value '%s' is out of range for Arrow::Uint8", addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &ival, sizeof(uint8_t));
		STAT_UPDATES(column, u8, ival);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_int16_value(SQLfield *column, const char *addr, int sz)
{
	size_t	row_index = column->nitems++;
	char   *end;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int16_t));
	else
	{
		int64_t		ival = strtol(addr, &end, 10);

		if (*end != '\0' || errno != 0)
			Elog("value '%s' is not valid", addr);
		if (ival < SHRT_MIN || ival > SHRT_MAX)
			Elog("value '%s' is out of range for Arrow::Int16", addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &ival, sizeof(int16_t));
		STAT_UPDATES(column, i16, ival);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_uint16_value(SQLfield *column, const char *addr, int sz)
{
	size_t	row_index = column->nitems++;
	char   *end;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int16_t));
	else
	{
		uint64_t	ival = strtoul(addr, &end, 10);

		if (*end != '\0' || errno != 0)
			Elog("value '%s' is not valid", addr);
		if (ival > USHRT_MAX)
			Elog("value '%s' is out of range for Arrow::Uint16", addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &ival, sizeof(uint16_t));
		STAT_UPDATES(column, u16, ival);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_int32_value(SQLfield *column, const char *addr, int sz)
{
	size_t	row_index = column->nitems++;
	char   *end;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int32_t));
	else
	{
		int64_t		ival = strtol(addr, &end, 10);

		if (*end != '\0' || errno != 0)
			Elog("value '%s' is not valid", addr);
		if (ival < INT_MIN || ival > INT_MAX)
			Elog("value '%s' is out of range for Arrow::Int32", addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &ival, sizeof(int32_t));
		STAT_UPDATES(column, i32, ival);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_uint32_value(SQLfield *column, const char *addr, int sz)
{
	size_t	row_index = column->nitems++;
	char   *end;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int32_t));
	else
	{
		uint64_t	ival = strtoul(addr, &end, 10);

		if (*end != '\0' || errno != 0)
			Elog("value '%s' is not valid", addr);
		if (ival > UINT_MAX)
			Elog("value '%s' is out of range for Arrow::Uint32", addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &ival, sizeof(uint32_t));
		STAT_UPDATES(column, u32, ival);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_int64_value(SQLfield *column, const char *addr, int sz)
{
	size_t	row_index = column->nitems++;
	char   *end;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int64_t));
	else
	{
		int64_t		ival = strtol(addr, &end, 10);

		if (*end != '\0' || errno != 0)
			Elog("value '%s' is not valid", addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &ival, sizeof(int64_t));
		STAT_UPDATES(column, i64, ival);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_uint64_value(SQLfield *column, const char *addr, int sz)
{
	size_t	row_index = column->nitems++;
	char   *end;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int64_t));
	else
	{
		uint64_t	ival = strtoul(addr, &end, 10);

		if (*end != '\0' || errno != 0)
			Elog("value '%s' is not valid", addr);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &ival, sizeof(uint64_t));
		STAT_UPDATES(column, u64, ival);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_float32_value(SQLfield *column, const char *addr, int sz)
{
	size_t	row_index = column->nitems++;
	char   *end;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(float));
	else
	{
		float	fval = strtof(addr, &end);

		if (*end != '\0' || errno != 0)
			Elog("value '%s' is out of range for %s",
				 addr, arrowNodeName(&column->arrow_type.node));
		sql_buffer_setbit(&column->nullmap, row_index);
        sql_buffer_append(&column->values, &fval, sizeof(float));
		STAT_UPDATES(column, f32, fval);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_float64_value(SQLfield *column, const char *addr, int sz)
{
	size_t	row_index = column->nitems++;
	char   *end;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(double));
	else
	{
		double	fval = strtod(addr, &end);

		if (*end != '\0' || errno != 0)
			Elog("value '%s' is out of range for %s",
				 addr, arrowNodeName(&column->arrow_type.node));
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &fval, sizeof(double));
		STAT_UPDATES(column, f64, fval);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_decimal_value(SQLfield *column, const char *addr, int sz)
{
	size_t	row_index = column->nitems++;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(__int128));
	else
	{
		char	   *pos = (char *)addr;
		char	   *end;
		int			dscale = column->arrow_type.Decimal.scale;
		bool		negative = false;
		__int128	value;

		if (*pos == '-')
		{
			negative = true;
			pos++;
		}
		else if (*pos == '+')
			pos++;

		end = strchr(pos, '.');
		if (!end)
			end = pos + strlen(pos);
		if (end - pos < 20)
		{
			char   *point;

			value = strtoul(pos, &point, 10);
			if (*point != '.' && *point != '\0')
				Elog("invalid Decimal value [%s]", addr);			
		}
		else
		{
			if (dscale < 0)
				end += dscale;
			value = 0;
			while (pos < end)
				value = 10 * value + (*pos++ - '0');
		}

		if (dscale > 0 && *end != '\0')
		{
			assert(*end == '.');
			for (pos = end + 1; dscale > 0; dscale--)
			{
				if (!isdigit(*pos))
					value = 10 * value;
				else
					value = 10 * value + (*pos++ - '0');
			}
		}
		if (negative)
			value = -value;		
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(__int128));
	}
	return __buffer_usage_inline_type(column);
}

static long
date2j(int y, int m, int d)
{
	long	julian;
	long	century;

	if (m > 2)
	{
		m += 1;
		y += 4800;
	}
	else
	{
		m += 13;
		y += 4799;
	}
	century = y / 100;
	julian = y * 365 - 32167;
	julian += y / 4 - century + century / 4;
	julian += 7834 * m / 256 + d;

	return julian;
}
#define UNIX_EPOCH_JDATE	2440588 /* == date2j(1970, 1, 1) */

/*
 * Date
 */
static size_t
put_date_day_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int32_t));
	else
	{
		int			y, m, d;
		int64_t		value;

		if (sscanf(addr, "%d-%d-%d", &y, &m, &d) != 3)
			Elog("invalid Date value: [%s]", addr);
		value = date2j(y, m, d) - UNIX_EPOCH_JDATE;
		if (value < INT_MAX || value > INT_MAX)
			Elog("Arrow::Date[day] out of range");
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int32_t));
		STAT_UPDATES(column, i32, value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_date_ms_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(int32_t));
	else
	{
		int			y, m, d;
		int64_t		value;

		if (sscanf(addr, "%d-%d-%d", &y, &m, &d) != 3)
			Elog("invalid Date value: [%s]", addr);
		value = date2j(y, m, d) - UNIX_EPOCH_JDATE;
		if (value <= LONG_MIN / 1000000 || value >= LONG_MAX / 1000000)
			Elog("Arrow::Date[ms] out of range");
		value *= 1000000;
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int64_t));
		STAT_UPDATES(column, i64, value);
	}
	return __buffer_usage_inline_type(column);
}

/*
 * Time
 */
#define __PUT_TIME_VALUE_TEMPLATE(NAME,ARROW_SCALE,ARROW_SZ,STAT_VALUE)	\
	static size_t														\
	put_time_##NAME##_value(SQLfield *column, const char *addr, int sz)	\
	{																	\
		size_t	row_index = column->nitems++;							\
																		\
		assert(sz == ARROW_SZ);											\
		if (!addr)														\
			__put_inline_null_value(column, row_index, ARROW_SZ);		\
		else															\
		{																\
			int		h, m, s, frac = 0;									\
			int64_t	value;												\
			char   *pos = strchr(addr, '.');							\
																		\
			if (pos)													\
			{															\
				int	scale = strlen(pos + 1);							\
																		\
				if (scale < 0 || scale > 6)								\
					Elog("invalid Time value [%s]", addr);				\
				scale -= ARROW_SCALE;									\
				if (sscanf(addr, "%d:%d:%d.%d",							\
						   &h, &m, &s, &frac) != 4)						\
					Elog("invalid Time value [%s]", addr);				\
				if (scale < 0)											\
					frac *= __exp10[-scale];							\
				else if (scale > 0)										\
					frac /= __exp10[scale];								\
			}															\
			else														\
			{															\
				if (sscanf(addr, "%d:%d:%d",							\
						   &h, &m, &s) != 3)							\
					Elog("invalid Time value [%s]", addr);				\
			}															\
			value = 3600L * (long)h + 60L * (long)m + (long)s;			\
			value = value * __exp10[ARROW_SCALE] + frac;				\
																		\
			sql_buffer_setbit(&column->nullmap, row_index);				\
			sql_buffer_append(&column->values, &value, ARROW_SZ);		\
			STAT_UPDATES(column, STAT_VALUE, value);					\
		}																\
		return __buffer_usage_inline_type(column);						\
	}
__PUT_TIME_VALUE_TEMPLATE(sec,0,sizeof(int32_t),i32)
__PUT_TIME_VALUE_TEMPLATE( ms,3,sizeof(int32_t),i32)
__PUT_TIME_VALUE_TEMPLATE( us,6,sizeof(int64_t),i64)
__PUT_TIME_VALUE_TEMPLATE( ns,9,sizeof(int64_t),i64)

/*
 * Timestamp
 */
#define __PUT_TIMESTAMP_VALUE_TEMPLATE(NAME,ARROW_SCALE,ARROW_SZ)		\
	static size_t														\
	put_timestamp_##NAME##_value(SQLfield *column, const char *addr, int sz) \
	{																	\
		size_t	row_index = column->nitems++;							\
																		\
		assert(sz == ARROW_SZ);											\
		if (!addr)														\
			__put_inline_null_value(column, row_index, ARROW_SZ);		\
		else															\
		{																\
			int		year, mon, day, hour, min, sec, frac = 0;			\
			int64_t	value;												\
			char   *pos = strchr(addr, '.');							\
																		\
			if (pos != NULL)											\
			{															\
				int	scale = strlen(pos + 1);							\
																		\
				scale -= ARROW_SCALE;									\
				if (sscanf(addr, "%04d-%02d-%02d %d:%d:%d.%d",			\
						   &year, &mon, &day,							\
						   &hour, &min, &sec, &frac) != 7)				\
					Elog("invalid Time value [%s]", addr);				\
				if (scale < 0)											\
					frac *= __exp10[-scale];							\
				else if (scale > 0)										\
					frac /= __exp10[scale];								\
			}															\
			else														\
			{															\
				if (sscanf(addr, "%04d-%02d-%02d %d:%d:%d",				\
						   &year, &mon, &day,							\
						   &hour, &min, &sec) != 6)						\
					Elog("invalid Timestamp value [%s]", addr);			\
			}															\
			value = date2j(year, mon, day) - UNIX_EPOCH_JDATE;			\
			value = 86400L * value + (3600L * hour + 60L * min + sec);	\
			value = value * __exp10[ARROW_SCALE] + frac;				\
																		\
			sql_buffer_setbit(&column->nullmap, row_index);				\
			sql_buffer_append(&column->values, &value, ARROW_SZ);		\
			STAT_UPDATES(column, i64, value);							\
		}																\
		return __buffer_usage_inline_type(column);						\
	}

__PUT_TIMESTAMP_VALUE_TEMPLATE(sec,0,sizeof(int64_t))
__PUT_TIMESTAMP_VALUE_TEMPLATE( ms,3,sizeof(int64_t))
__PUT_TIMESTAMP_VALUE_TEMPLATE( us,6,sizeof(int64_t))
__PUT_TIMESTAMP_VALUE_TEMPLATE( ns,9,sizeof(int64_t))

static size_t
put_variable_value(SQLfield *column, const char *addr, int sz)
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

#if 0
/*
 * NOTE: Right now, we have no way to fetch column's attribute during
 * query execution (for more correctness, libmysqlclient does not allow
 * to run another query before mysql_free_result(), so we cannot run
 *   SHOW COLUMNS FROM <tablename> LIKE '<columnname>'
 * according to the query results.
 * So, mysql2arrow saves Enum data type in MySQL as normal Utf8/Binary
 * values, without creation of DictionaryBatch chunks.
 */
static size_t
put_dictionary_value(SQLfield *column, const char *addr, int sz)
{
	SQLdictionary *enumdict = column->enumdict;
	hashItem   *hitem;
	uint32		hash, hindex;
	size_t		row_index = column->nitems++;

	if (!addr)
		__put_inline_null_value(column, row_index, sizeof(uint32));
	else
	{
		hash = hash_any((unsigned char *)addr, sz);
		hindex = hash % enumdict->nslots;
		for (hitem = enumdict->hslots[hindex];
			 hitem != NULL;
			 hitem = hitem->next)
		{
			if (hitem->hash == hash &&
				hitem->label_sz == sz &&
				memcmp(hitem->label, addr, sz) == 0)
				break;
		}
		if (!hitem)
			Elog("Enum label was not found");
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &hitem->index, sizeof(uint32));
	}
	return __buffer_usage_varlena_type(column);
}
#endif

/*
 * Callbacks for move values between SQLtables (for parallel execution)
 */
#define MOVE_SCALAR_TEMPLATE(NAME,VALUE_TYPE,STAT_NAME)                 \
	static size_t                                                       \
	move_##NAME##_value(SQLfield *dest, const SQLfield *src, long sindex) \
	{                                                                   \
		size_t  dindex = dest->nitems++;                                \
                                                                        \
		if (!sql_buffer_getbit(&src->nullmap, sindex))                  \
			__put_inline_null_value(dest, dindex, sizeof(VALUE_TYPE));  \
		else                                                            \
		{                                                               \
			VALUE_TYPE  value;                                          \
                                                                        \
			value = ((VALUE_TYPE *)src->values.data)[sindex];           \
			sql_buffer_setbit(&dest->nullmap, dindex);                  \
			sql_buffer_append(&dest->values, &value,                    \
							  sizeof(VALUE_TYPE));                      \
			STAT_UPDATES(dest,STAT_NAME,value);                         \
		}                                                               \
		return __buffer_usage_inline_type(dest);                        \
	}
MOVE_SCALAR_TEMPLATE(int8, int8_t, i8)
MOVE_SCALAR_TEMPLATE(uint8, uint8_t, u8)
MOVE_SCALAR_TEMPLATE(int16, int16_t, i16)
MOVE_SCALAR_TEMPLATE(uint16, uint16_t, u16)
MOVE_SCALAR_TEMPLATE(int32, int32_t, i32)
MOVE_SCALAR_TEMPLATE(uint32, uint32_t, u32)
MOVE_SCALAR_TEMPLATE(int64, int64_t, i64)
MOVE_SCALAR_TEMPLATE(uint64, uint64_t, u64)
MOVE_SCALAR_TEMPLATE(float32, float, f32)
MOVE_SCALAR_TEMPLATE(float64, double, f64)
MOVE_SCALAR_TEMPLATE(decimal, int128_t, i128)

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
 * mysql_setup_attribute
 */
static int
mysql_setup_attribute(MYSQL *conn,
					  MYSQL_FIELD *my_field,
					  ArrowField *arrow_field,
					  SQLtable *table,
					  SQLfield *column,
					  int attnum)
{
	ArrowType  *arrow_type = (arrow_field ? &arrow_field->type : NULL);
	bool		is_signed = ((my_field->flags & UNSIGNED_FLAG) == 0);
	bool		is_binary = ((my_field->flags & BINARY_FLAG) != 0);
//	bool		is_enum   = ((my_field->flags & ENUM_FLAG) != 0);
	int			bitWidth = -1;
	int			precision = -1;
	int			dscale;
	int			unit;
	const char *tz_name = NULL;
	
	memset(column, 0, sizeof(SQLfield));
	column->field_name = pstrdup(my_field->name);
	column->sql_type.mysql.typeid = (int)my_field->type;
	switch (my_field->type)
	{
		/*
		 * ArrowTypeInt
		 */
		case MYSQL_TYPE_TINY:
			if (bitWidth < 0)
			{
				if (!arrow_type)
					bitWidth = 8;
				else if (arrow_type->node.tag == ArrowNodeTag__Int &&
						 (is_signed
						  ? arrow_type->Int.is_signed && arrow_type->Int.bitWidth >= 8
						  : arrow_type->Int.bitWidth > 8))
				{
					bitWidth  = arrow_type->Int.bitWidth;
					is_signed = arrow_type->Int.is_signed;
				}
				else
				{
					Elog("attribute '%s' is not compatible", my_field->name);
				}
			}

		case MYSQL_TYPE_SHORT:
		case MYSQL_TYPE_YEAR:
			if (bitWidth < 0)
			{
				if (!arrow_type)
					bitWidth = 16;
				else if (arrow_type->node.tag == ArrowNodeTag__Int &&
						 (is_signed
						  ? arrow_type->Int.is_signed && arrow_type->Int.bitWidth >= 16
						  : arrow_type->Int.bitWidth > 16))
				{
					bitWidth  = arrow_type->Int.bitWidth;
					is_signed = arrow_type->Int.is_signed;
				}
				else
				{
					Elog("attribute '%s' is not compatible", my_field->name);
				}
			}

		case MYSQL_TYPE_INT24:
		case MYSQL_TYPE_LONG:
			if (bitWidth < 0)
			{
				if (!arrow_type)
					bitWidth = 32;
				else if (arrow_type->node.tag == ArrowNodeTag__Int &&
						 (is_signed
						  ? arrow_type->Int.is_signed && arrow_type->Int.bitWidth >= 32
						  : arrow_type->Int.bitWidth > 32))
				{
					bitWidth  = arrow_type->Int.bitWidth;
					is_signed = arrow_type->Int.is_signed;
				}
				else
				{
					Elog("attribute '%s' is not compatible", my_field->name);
				}
			}

		case MYSQL_TYPE_LONGLONG:
			if (bitWidth < 0)
			{
				if (!arrow_type)
					bitWidth = 64;
				else if (arrow_type->node.tag == ArrowNodeTag__Int &&
						 (is_signed
						  ? arrow_type->Int.is_signed && arrow_type->Int.bitWidth >= 64
						  : arrow_type->Int.bitWidth > 64))
				{
					bitWidth  = arrow_type->Int.bitWidth;
					is_signed = arrow_type->Int.is_signed;
				}
				else
				{
					Elog("attribute '%s' is not compatible", my_field->name);
				}
			}
			initArrowNode(&column->arrow_type, Int);
			column->arrow_type.Int.is_signed = is_signed;
			column->arrow_type.Int.bitWidth = bitWidth;
			switch (bitWidth)
			{
				case 8:
					column->put_value  = (is_signed ? put_int8_value  : put_uint8_value);
					column->move_value = (is_signed ? move_int8_value : move_uint8_value);
					break;
				case 16:
					column->put_value  = (is_signed ? put_int16_value  : put_uint16_value);
					column->move_value = (is_signed ? move_int16_value : move_uint16_value);
					break;
				case 32:
					column->put_value  = (is_signed ? put_int32_value  : put_uint32_value);
					column->move_value = (is_signed ? move_int32_value : move_uint32_value);
					break;
				case 64:
					column->put_value  = (is_signed ? put_int64_value  : put_uint64_value);
					column->move_value = (is_signed ? move_int64_value : move_uint64_value);
					break;
				default:
					Elog("attribute '%s' try to use unsupported Int::bitWidth(%d)",
						 my_field->name, bitWidth);
					break;
			}
			return 2;		/* nullmap + values */

		/*
		 * ArrowTypeFloatingPoint
		 */
		case MYSQL_TYPE_FLOAT:
			if (precision < 0)
			{
				if (!arrow_type)
					precision = ArrowPrecision__Single;
				else if (arrow_type->node.tag == ArrowNodeTag__FloatingPoint &&
						 (arrow_type->FloatingPoint.precision == ArrowPrecision__Single ||
						  arrow_type->FloatingPoint.precision == ArrowPrecision__Double))
					precision = arrow_type->FloatingPoint.precision;
				else
					Elog("attribute '%s' is not compatible", my_field->name);
			}
		case MYSQL_TYPE_DOUBLE:
			if (precision < 0)
			{
				if (!arrow_type)
					precision = ArrowPrecision__Double;
				else if (arrow_type->node.tag == ArrowNodeTag__FloatingPoint &&
						 (arrow_type->FloatingPoint.precision == ArrowPrecision__Single ||
						  arrow_type->FloatingPoint.precision == ArrowPrecision__Double))
					precision = arrow_type->FloatingPoint.precision;
				else
					Elog("attribute '%s' is not compatible", my_field->name);
			}
			initArrowNode(&column->arrow_type, FloatingPoint);
			column->arrow_type.FloatingPoint.precision = precision;
			switch (precision)
			{
				case ArrowPrecision__Single:
					column->put_value  = put_float32_value;
					column->move_value = move_float32_value;
					break;
				case ArrowPrecision__Double:
					column->put_value  = put_float64_value;
					column->move_value = move_float64_value;
					break;
				default:
					Elog("attribute '%s' try to use unknown FloatingPoint::precision(%d)",
						 my_field->name, precision);
					break;
			}
			return 2;		/* nullmap + values */
		/*
		 * ArrowTypeDecimal
		 */
		case MYSQL_TYPE_DECIMAL:
		case MYSQL_TYPE_NEWDECIMAL:
			precision = my_field->max_length;
			dscale = my_field->decimals;
			if (arrow_type)
			{
				if (arrow_type->node.tag != ArrowNodeTag__Decimal)
					Elog("attribute %d is not compatible to %s",
						 attnum, arrow_type->node.tagName);
				precision = arrow_type->Decimal.precision;
				dscale = arrow_type->Decimal.scale;
			}
			initArrowNode(&column->arrow_type, Decimal);
			column->arrow_type.Decimal.precision = precision;
			column->arrow_type.Decimal.scale = dscale;
			column->put_value = put_decimal_value;
			column->move_value = move_decimal_value;
			return 2;		/* nullmap + values */
		/*
		 * ArrowTypeDate
		 */
		case MYSQL_TYPE_DATE:
			if (!arrow_type)
				unit = ArrowDateUnit__Day;
			else if (arrow_type->node.tag == ArrowNodeTag__Date &&
					 (arrow_type->Date.unit == ArrowDateUnit__Day ||
					  arrow_type->Date.unit == ArrowDateUnit__MilliSecond))
				unit = arrow_type->Date.unit;
			else
				Elog("attribute '%s' is not compatible", my_field->name);
			initArrowNode(&column->arrow_type, Date);
			column->arrow_type.Date.unit = unit;
			switch (unit)
			{
				case ArrowDateUnit__Day:
					column->put_value  = put_date_day_value;
					column->move_value = move_int32_value;
					break;
				case ArrowDateUnit__MilliSecond:
					column->put_value  = put_date_ms_value;
					column->move_value = move_int64_value;
					break;
			}
			return 2;		/* nullmap + values */
		/*
		 * ArrowTypeTime
		 */
		case MYSQL_TYPE_TIME:
			if (!arrow_type)
			{
				if (my_field->decimals == 0)
					unit = ArrowTimeUnit__Second;
				else if (my_field->decimals <= 3)
					unit = ArrowTimeUnit__MilliSecond;
				else if (my_field->decimals <= 6)
					unit = ArrowTimeUnit__MicroSecond;
				else
					unit = ArrowTimeUnit__NanoSecond;
			}
			else if (arrow_type->node.tag == ArrowNodeTag__Time)
			{
				switch (arrow_type->Time.unit)
				{
					case ArrowTimeUnit__Second:
					case ArrowTimeUnit__MilliSecond:
						if (arrow_type->Time.bitWidth != 32)
							Elog("Arrow::Time has wrong bitWidth (%d)",
								 arrow_type->Time.bitWidth);
						break;
					case ArrowTimeUnit__MicroSecond:
					case ArrowTimeUnit__NanoSecond:
						if (arrow_type->Time.bitWidth != 64)
							Elog("Arrow::Time has wrong bitWidth (%d)",
								 arrow_type->Time.bitWidth);
						break;
					default:
						Elog("Arrow::Time has unknown unit (%d)",
							 arrow_type->Time.unit);
						break;
				}
				unit = arrow_type->Time.unit;
			}
			else
			{
				Elog("attribute '%s' is not compatible", my_field->name);
			}

			initArrowNode(&column->arrow_type, Time);
			column->arrow_type.Time.unit = unit;
			switch (unit)
			{
				case ArrowTimeUnit__Second:
					column->arrow_type.Time.bitWidth = 32;
					column->put_value  = put_time_sec_value;
					column->move_value = move_int32_value;
					break;
				case ArrowTimeUnit__MilliSecond:
					column->arrow_type.Time.bitWidth = 32;
					column->put_value  = put_time_ms_value;
					column->move_value = move_int32_value;
					break;
				case ArrowTimeUnit__MicroSecond:
					column->arrow_type.Time.bitWidth = 64;
					column->put_value  = put_time_us_value;
					column->move_value = move_uint64_value;
					break;
				case ArrowTimeUnit__NanoSecond:
					column->arrow_type.Time.bitWidth = 64;
					column->put_value  = put_time_ns_value;
					column->move_value = move_uint64_value;
					break;
				default:
					Elog("Bug? unknown Arrow::Time unit (%d)", unit);
					break;
			}
			return 2;		/* nullmap + values */
			
		/*
		 * ArrowTypeTimestamp
		 */
		case MYSQL_TYPE_TIMESTAMP:
			tz_name = mysql_timezone;
		case MYSQL_TYPE_DATETIME:
			if (!arrow_type)
			{
				if (my_field->decimals == 0)
					unit = ArrowTimeUnit__Second;
				else if (my_field->decimals <= 3)
					unit = ArrowTimeUnit__MilliSecond;
				else if (my_field->decimals <= 6)
					unit = ArrowTimeUnit__MicroSecond;
				else
					unit = ArrowTimeUnit__NanoSecond;
			}
			else if (arrow_type->node.tag == ArrowNodeTag__Timestamp)
			{
				unit = arrow_type->Timestamp.unit;
				if (unit != ArrowTimeUnit__Second &&
					unit != ArrowTimeUnit__MilliSecond &&
					unit != ArrowTimeUnit__MicroSecond &&
					unit != ArrowTimeUnit__NanoSecond)
					Elog("Arrow::Timestamp has unknown unit (%d)", unit);
			}
			else
			{
				Elog("attribute '%s' is not compatible", my_field->name);
			}
			initArrowNode(&column->arrow_type, Timestamp);
			column->arrow_type.Timestamp.unit = unit;
			if (tz_name)
			{
				column->arrow_type.Timestamp.timezone = pstrdup(tz_name);
				column->arrow_type.Timestamp._timezone_len = strlen(tz_name);
			}
			switch (unit)
			{
				case ArrowTimeUnit__Second:
					column->put_value  = put_timestamp_sec_value;
					column->move_value = move_int64_value;
					break;
				case ArrowTimeUnit__MilliSecond:
					column->put_value  = put_timestamp_ms_value;
					column->move_value = move_int64_value;
					break;
				case ArrowTimeUnit__MicroSecond:
					column->put_value  = put_timestamp_us_value;
					column->move_value = move_int64_value;
					break;
				case ArrowTimeUnit__NanoSecond:
					column->put_value  = put_timestamp_ns_value;
					column->move_value = move_int64_value;
					break;
				default:
					Elog("Bug? unknown Arrow::Timestamp unit(%d)", unit);
					break;
			}
			return 2;		/* nullmap + values */

		case MYSQL_TYPE_STRING:
		case MYSQL_TYPE_VAR_STRING:
		case MYSQL_TYPE_VARCHAR:
		case MYSQL_TYPE_BLOB:
		case MYSQL_TYPE_TINY_BLOB:
		case MYSQL_TYPE_MEDIUM_BLOB:
		case MYSQL_TYPE_LONG_BLOB:
			if (!is_binary)
			{
				/*
				 * ArrowTypeUtf8
				 */
				initArrowNode(&column->arrow_type, Utf8);
				column->put_value  = put_variable_value;
				column->move_value = move_variable_value;
			}
			else
			{
				/*
				 * ArrowTypeBinary
				 */
				initArrowNode(&column->arrow_type, Binary);
				column->put_value  = put_variable_value;
				column->move_value = move_variable_value;
			}
			return 3;	/* nullmap + index + extra */

		case MYSQL_TYPE_NULL:
			Elog("MySQL Null data type is not supported");
		case MYSQL_TYPE_ENUM:
			Elog("MySQL Enum data type is not supported");
		case MYSQL_TYPE_SET:
			Elog("MySQL SET data type is not supported");
		case MYSQL_TYPE_BIT:
			Elog("MySQL Bit data type is not supported");
#if MYSQL_VERSION_ID >= 50700
		case MYSQL_TYPE_JSON:
			Elog("MySQL JSON data type is not supported");
#endif /* >= MySQL 5.7.00 */
		case MYSQL_TYPE_GEOMETRY:
			Elog("MySQL Geometry data type is not supported");
		default:
			Elog("unsupported MySQL data type: %d", (int)my_field->type);
	}
	return -1;
}

/*
 * callbacks from sql2arrow main logic
 */
typedef struct {
	MYSQL	   *conn;
	MYSQL_RES  *res;
} MYSTATE;

/*
 * sqldb_server_connect
 */
void *
sqldb_server_connect(const char *sqldb_hostname,
					 const char *sqldb_port_num,
					 const char *sqldb_username,
					 const char *sqldb_password,
					 const char *sqldb_database,
					 userConfigOption *sqldb_session_configs)
{
	MYSTATE	   *mystate = palloc0(sizeof(MYSTATE));
	MYSQL	   *conn;
	MYSQL_RES  *res;
	MYSQL_ROW	row;
	int			port_num = 0;
	userConfigOption *conf;
	const char *query;

	conn = mysql_init(NULL);
	if (!conn)
		Elog("failed on mysql_init");

	if (sqldb_port_num)
		port_num = atoi(sqldb_port_num);
	if (!mysql_real_connect(conn,
							sqldb_hostname,
							sqldb_username,
							sqldb_password,
							sqldb_database,
							port_num,
							NULL, 0))
		Elog("failed on mysql_real_connect: %s", mysql_error(conn));

	/*
	 * Preset user's config options
	 */
	for (conf = sqldb_session_configs; conf != NULL; conf = conf->next)
	{
		if (mysql_query(conn, conf->query) != 0)
			Elog("failed on mysql_query('%s'): %s",
				 conf->query, mysql_error(conn));
	}

	/*
     * ensure client encoding is UTF-8
     */
	query = "SET character_set_results = 'utf8'";
	if (mysql_query(conn, query) != 0)
		Elog("failed on mysql_query('%s'): %s",
			 query, mysql_error(conn));

	/*
	 * check the current timezone setting, and temporary change
	 * the configuration to UTC for Apache Arrow's binary representation
	 */
	query = "SELECT @@time_zone, @@system_time_zone";
	if (mysql_query(conn, query) != 0)
		Elog("failed on mysql_query('%s'): %s",
			 query, mysql_error(conn));
	res = mysql_store_result(conn);
	if (!res)
		Elog("failed on mysql_store_result: %s", mysql_error(conn));
	if (mysql_num_fields(res) != 2 ||
		mysql_num_rows(res) != 1)
		Elog("unexpected query result for '%s'", query);
	row = mysql_fetch_row(res);
	if (strcmp(row[0], "SYSTEM") == 0)
		mysql_timezone = pstrdup(row[1]);
	else
		mysql_timezone = pstrdup(row[0]);
	mysql_free_result(res);

	query = "SET time_zone = 'UTC'";
	if (mysql_query(conn, query) != 0)
	{
		Elog("unable to set time_zone to 'UTC': %s\n"
			 "\n"
			 "HINT: try this script below.\n"
			 "  $ mysql_tzinfo_to_sql /usr/share/zoneinfo | mysql -u root DBNAME\n",
			 mysql_error(conn));
	}
	mystate->conn = conn;
	return mystate;
}

/*
 * sqldb_begin_query
 */
SQLtable *
sqldb_begin_query(void *sqldb_state,
				  const char *sqldb_command,
				  ArrowFileInfo *af_info,
				  SQLdictionary *sql_dict_list)
{
	MYSTATE	   *mystate = (MYSTATE *)sqldb_state;
	MYSQL	   *conn = mystate->conn;
	MYSQL_RES  *res;
	SQLtable   *table;
	int			j, nfields;
	const char *query;

	/* start transaction with read-only mode */
	query = "START TRANSACTION READ ONLY";
	if (mysql_query(conn, query) != 0)
		Elog("failed on mysql_query('%s'): %s",
			 query, mysql_error(conn));

	/* exec SQL command  */
	if (mysql_query(conn, sqldb_command) != 0)
		Elog("failed on mysql_query('%s'): %s",
			 sqldb_command, mysql_error(conn));
	res = mysql_use_result(conn);
	if (!res)
		Elog("failed on mysql_use_result: %s",
			 mysql_error(conn));
	mystate->res = res;

	nfields = mysql_num_fields(res);
	if (af_info &&
		af_info->footer.schema._num_fields != nfields)
		Elog("--append is given, but number of columns are different.");
	
	/* create SQLtable buffer */
	table = palloc0(offsetof(SQLtable, columns[nfields]));
	table->nfields = nfields;
	table->sql_dict_list = sql_dict_list;
	for (j=0; j < nfields; j++)
	{
		MYSQL_FIELD *my_field = mysql_fetch_field_direct(res, j);
		ArrowField	*arrow_field = NULL;

		if (af_info)
			arrow_field = &af_info->footer.schema.fields[j];

		table->numBuffers +=
			mysql_setup_attribute(conn,
								  my_field,
								  arrow_field,
								  table,
								  &table->columns[j],
								  j+1);
		table->numFieldNodes++;
	}
	return table;
}

bool
sqldb_fetch_results(void *sqldb_state, SQLtable *table)
{
	MYSTATE	   *mystate = (MYSTATE *)sqldb_state;
	MYSQL_ROW	row;
	unsigned long *row_sz;
	size_t		usage = 0;
	int			j;

	row = mysql_fetch_row(mystate->res);
	if (!row)
		return false;

	row_sz = mysql_fetch_lengths(mystate->res);
	for (j=0; j < table->nfields; j++)
	{
		SQLfield   *column = &table->columns[j];
		size_t		sz = (row[j] != NULL ? row_sz[j] : 0);

		assert(table->nitems == column->nitems);
		usage += sql_field_put_value(column, row[j], sz);
	}
	table->usage = usage;
	table->nitems++;

	return true;
}

void
sqldb_close_connection(void *sqldb_state)
{
	MYSTATE	   *mystate = (MYSTATE *)sqldb_state;

	mysql_free_result(mystate->res);
	mysql_close(mystate->conn);
}

char *
sqldb_build_simple_command(void *sqldb_state,
						   const char *simple_table_name,
						   int num_worker_threads,
						   size_t segment_sz)
{
	char   *buf = alloca(strlen(simple_table_name) + 100);

	assert(num_worker_threads == 1);
	sprintf(buf, "SELECT * FROM %s", simple_table_name);

	return pstrdup(buf);
}

/*
 * PG12 or later replaces XXprintf by pg_XXprintf
 */
#ifdef vprintf
#undef vprintf
#endif
#ifdef vfprintf
#undef vfprintf
#endif
#ifdef vsprintf
#undef vsprintf
#endif
#ifdef vsnprintf
#undef vsnprintf
#endif

int
pg_printf(const char *fmt,...)
{
	va_list	args;
	int		r;

	va_start(args, fmt);
	r = vprintf(fmt, args);
	va_end(args);

	return r;
}

int
pg_fprintf(FILE *stream, const char *fmt,...)
{
	va_list args;
	int		r;

	va_start(args, fmt);
	r = vfprintf(stream, fmt, args);
    va_end(args);

	return r;
}

int
pg_sprintf(char *str, const char *fmt,...)
{
	va_list	args;
	int		r;

	va_start(args, fmt);
	r = vsprintf(str, fmt, args);
	va_end(args);

	return r;
}

int
pg_snprintf(char *str, size_t count, const char *fmt,...)
{
	va_list	args;
	int		r;

	va_start(args, fmt);
	r = vsnprintf(str, count, fmt, args);
	va_end(args);

	return r;
}

int
pg_vsnprintf(char *str, size_t count, const char *fmt, va_list args)
{
	return vsnprintf(str, count, fmt, args);
}
