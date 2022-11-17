/*
 * arrow_ruby.c
 *
 * A Ruby language extension to write out data as Apache Arrow files.
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <ruby.h>
#include <ctype.h>
#include <libgen.h>
#include <sys/file.h>
#include "float2.h"
#define Elog(fmt,...)							\
	rb_raise(rb_eException, "%s:%d " fmt,		\
			 __FILE__, __LINE__, ##__VA_ARGS__)
#include "arrow_ipc.h"

/*
 * Misc definitions
 */
#define SECS_PER_DAY	86400UL
#define IP4ADDR_LEN		4
#define IP6ADDR_LEN		16

static inline char *
trim_cstring(char *str)
{
	char   *end;

	while (isspace(*str))
		str++;
	end = str + strlen(str) - 1;
	while (end >= str && isspace(*end))
		*end-- = '\0';

	return str;
}

static int128_t
__atoi128(const char *tok, bool *p_isnull)
{
	int128_t	ival = 0;
	bool		is_minus = false;

	if (*tok == '-')
	{
		is_minus = true;
		tok++;
	}
	while (isdigit(*tok))
	{
		ival = 10 * ival + (*tok - '0');
		tok++;
	}
	if (*tok != '\0')
		*p_isnull = true;
	if (is_minus)
	{
		if (ival == 0)
			*p_isnull = true;
		ival = -ival;
	}
	return ival;
}


static inline VALUE
rb_puts(VALUE obj)
{
	return rb_funcall(rb_mKernel, rb_intern("puts"), 1, obj);
}

/*
 * memory allocation wrapper
 */
void *
palloc(size_t sz)
{
	return ruby_xmalloc(sz);
}

void *
palloc0(size_t sz)
{
	void   *ptr = ruby_xmalloc(sz);

	memset(ptr, 0, sz);

	return ptr;
}

char *
pstrdup(const char *str)
{
	void   *dst = palloc(strlen(str) + 1);

	strcpy(dst, str);

	return dst;
}

char *
pstrdup_ruby(VALUE datum)
{
	size_t	len = RSTRING_LEN(datum);
	char   *dst = palloc(len + 1);

	memcpy(dst, RSTRING_PTR(datum), len);
	dst[len] = '\0';

	return dst;
}

void *
repalloc(void *old, size_t sz)
{
	return ruby_xrealloc(old, sz);
}

void
pfree(void *ptr)
{
	ruby_xfree(ptr);
}

/* ----------------------------------------------------------------
 *
 * Put values handler for Ruby VALUE
 *
 * ----------------------------------------------------------------
 */
static inline void
__put_inline_null_value(SQLfield *column, size_t row_index, int sz)
{
	column->nullcount++;
	sql_buffer_clrbit(&column->nullmap, row_index);
	sql_buffer_append_zero(&column->values, sz);
}

#define STAT_UPDATES(COLUMN,FIELD,VALUE)					\
	do {													\
		if ((COLUMN)->stat_enabled)							\
		{													\
			if (!(COLUMN)->stat_datum.is_valid)				\
			{												\
				(COLUMN)->stat_datum.min.FIELD = VALUE;     \
				(COLUMN)->stat_datum.max.FIELD = VALUE;     \
				(COLUMN)->stat_datum.is_valid = true;		\
			}												\
			else											\
			{												\
				if ((COLUMN)->stat_datum.min.FIELD > VALUE) \
					(COLUMN)->stat_datum.min.FIELD = VALUE; \
				if ((COLUMN)->stat_datum.max.FIELD < VALUE) \
					(COLUMN)->stat_datum.max.FIELD = VALUE; \
			}												\
		}													\
	} while(0)

/*
 * Bool
 */
static int
__ruby_fetch_bool_value(VALUE datum)
{
	if (datum == Qnil)
		return -1;		/* null */
	if (datum == Qtrue)
		return 1;
	else if (datum == Qfalse)
		return 0;
	else
	{
		if (CLASS_OF(datum) == rb_cString)
		{
			const char *ptr = RSTRING_PTR(datum);
			size_t		len = RSTRING_LEN(datum);

			if ((len == 4 && (memcmp(ptr, "true", 4) == 0 ||
							  memcmp(ptr, "True", 4) == 0 ||
							  memcmp(ptr, "TRUE", 4) == 0)) ||
				(len == 1 && (memcmp(ptr, "t", 1) == 0 ||
							  memcmp(ptr, "T", 1) == 0)))
				return 1;
			if ((len == 5 && (memcmp(ptr, "false", 5) == 0 ||
							  memcmp(ptr, "False", 5) == 0 ||
							  memcmp(ptr, "FALSE", 5) == 0)) ||
				(len == 1 && (memcmp(ptr, "f", 1) == 0 ||
							  memcmp(ptr, "F", 1) == 0)))
				return 0;
			/* elsewhere, try to convert to Integer */
			datum = rb_funcall(datum, rb_intern("to_i"), 0);
		}

		if (CLASS_OF(datum) == rb_cInteger ||
			CLASS_OF(datum) == rb_cFloat ||
			CLASS_OF(datum) == rb_cRational)
		{
			int		ival = NUM2INT(datum);

			return (ival == 0 ? 0 : 1);
		}
	}
	Elog("unable to convert to boolean value");
}

static size_t
ruby_put_bool_value(SQLfield *column, const char *addr, int sz)
{
	size_t	row_index = column->nitems++;
	int		bval = __ruby_fetch_bool_value((VALUE)addr);

	if (bval < 0)
	{
		/* null */
		column->nullcount++;
		sql_buffer_clrbit(&column->nullmap, row_index);
		sql_buffer_clrbit(&column->values,  row_index);
	}
	else
	{
		sql_buffer_setbit(&column->nullmap, row_index);
		if (bval != 0)
			sql_buffer_setbit(&column->values, row_index);
		else
			sql_buffer_clrbit(&column->values, row_index);		
	}
	return __buffer_usage_inline_type(column);
}

/*
 * IntXX/UintXX
 */
static VALUE
__ruby_fetch_integer_value(VALUE datum)
{
	if (datum != Qnil && CLASS_OF(datum) != rb_cInteger)
	{
		if (!rb_respond_to(datum, rb_intern("to_i")))
			datum = rb_funcall(datum, rb_intern("to_s"), 0);
		datum = rb_funcall(datum, rb_intern("to_i"), 0);
	}
	return datum;
}

static size_t
put_ruby_int8_value(SQLfield *column, const char *addr, int sz)
{
	VALUE		datum = __ruby_fetch_integer_value((VALUE)addr);
	size_t		row_index = column->nitems++;

	if (datum == Qnil)
		__put_inline_null_value(column, row_index, sizeof(int8_t));
	else
	{
		int8_t		value = NUM2CHR(datum);

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int8_t));

		STAT_UPDATES(column,i8,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_ruby_int16_value(SQLfield *column, const char *addr, int sz)
{
	VALUE		datum = __ruby_fetch_integer_value((VALUE)addr);
	size_t		row_index = column->nitems++;

	if (datum == Qnil)
		__put_inline_null_value(column, row_index, sizeof(int16_t));
	else
	{
		int16_t		value = NUM2SHORT(datum);

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int16_t));

		STAT_UPDATES(column,i16,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_ruby_int32_value(SQLfield *column, const char *addr, int sz)
{
	VALUE		datum = __ruby_fetch_integer_value((VALUE)addr);
	size_t		row_index = column->nitems++;

	if (datum == Qnil)
		__put_inline_null_value(column, row_index, sizeof(int32_t));
	else
	{
		int32_t		value = NUM2INT(datum);

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int32_t));

		STAT_UPDATES(column,i32,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_ruby_int64_value(SQLfield *column, const char *addr, int sz)
{
	VALUE		datum = __ruby_fetch_integer_value((VALUE)addr);
	size_t		row_index = column->nitems++;

	if (datum == Qnil)
		__put_inline_null_value(column, row_index, sizeof(int64_t));
	else
	{
		int64_t		value = NUM2LONG(datum);

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int64_t));

		STAT_UPDATES(column,i64,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_ruby_uint8_value(SQLfield *column, const char *addr, int sz)
{
	VALUE		datum = __ruby_fetch_integer_value((VALUE)addr);
	size_t		row_index = column->nitems++;

	if (datum == Qnil)
		__put_inline_null_value(column, row_index, sizeof(uint8_t));
	else
	{
		uint32_t	value = NUM2UINT(datum);

		if (value > UCHAR_MAX)
			rb_raise(rb_eRangeError, "Uint8 out of range (%u)", value);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(uint8_t));

		STAT_UPDATES(column,u8,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_ruby_uint16_value(SQLfield *column, const char *addr, int sz)
{
	VALUE		datum = __ruby_fetch_integer_value((VALUE)addr);
	size_t		row_index = column->nitems++;

	if (datum == Qnil)
		__put_inline_null_value(column, row_index, sizeof(uint16_t));
	else
	{
		uint32_t	value = NUM2UINT(datum);

		if (value > USHRT_MAX)
			rb_raise(rb_eRangeError, "Uint16 out of range (%u)", value);
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(uint16_t));

		STAT_UPDATES(column,u16,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_ruby_uint32_value(SQLfield *column, const char *addr, int sz)
{
	VALUE		datum = __ruby_fetch_integer_value((VALUE)addr);
	size_t		row_index = column->nitems++;

	if (datum == Qnil)
		__put_inline_null_value(column, row_index, sizeof(uint32_t));
	else
	{
		uint32_t	value = NUM2UINT(datum);

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(uint32_t));

		STAT_UPDATES(column,u32,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_ruby_uint64_value(SQLfield *column, const char *addr, int sz)
{
	VALUE		datum = __ruby_fetch_integer_value((VALUE)addr);
	size_t		row_index = column->nitems++;

	if (datum == Qnil)
		__put_inline_null_value(column, row_index, sizeof(uint64_t));
	else
	{
		uint64_t	value = NUM2UINT(datum);

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(uint64_t));

		STAT_UPDATES(column,u64,value);
	}
	return __buffer_usage_inline_type(column);
}

/*
 * FloatingPointXX
 */
static VALUE
__ruby_fetch_float_value(VALUE datum)
{
	if (datum != Qnil && CLASS_OF(datum) != rb_cFloat)
	{
		if (!rb_respond_to(datum, rb_intern("to_f")))
			datum = rb_funcall(datum, rb_intern("to_s"), 0);
		datum = rb_funcall(datum, rb_intern("to_f"), 0);
	}
	return datum;
}

static size_t
put_ruby_float16_value(SQLfield *column, const char *addr, int sz)
{
	VALUE		datum = __ruby_fetch_float_value((VALUE)addr);
	size_t		row_index = column->nitems++;

	if (datum == Qnil)
		__put_inline_null_value(column, row_index, sizeof(half_t));
	else
	{
		double	fval = NUM2DBL(datum);
		half_t	value = fp64_to_fp16(fval);

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(half_t));

		STAT_UPDATES(column,f64,fval);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_ruby_float32_value(SQLfield *column, const char *addr, int sz)
{
	VALUE		datum = __ruby_fetch_float_value((VALUE)addr);
	size_t		row_index = column->nitems++;

	if (datum == Qnil)
		__put_inline_null_value(column, row_index, sizeof(float));
	else
	{
		float	value = (float)NUM2DBL(datum);

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(float));

		STAT_UPDATES(column,f32,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_ruby_float64_value(SQLfield *column, const char *addr, int sz)
{
	VALUE		datum = __ruby_fetch_float_value((VALUE)addr);
	size_t		row_index = column->nitems++;

	if (datum == Qnil)
		__put_inline_null_value(column, row_index, sizeof(double));
	else
	{
		double	value = NUM2DBL(datum);

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(double));

		STAT_UPDATES(column,f64,value);
	}
	return __buffer_usage_inline_type(column);
}

/*
 * Decimal
 */
static bool
__ruby_fetch_decimal_value(VALUE datum, int128_t *p_value, int scale)
{
	VALUE		datum_klass;
	bool		retry = false;

	if (datum == Qnil)
		return false;
retry_again:
	datum_klass = CLASS_OF(datum);
	if (datum_klass == rb_cInteger ||
		datum_klass == rb_cFloat ||
		datum_klass == rb_cRational)
	{
		VALUE	ival;

		if (scale > 0)
		{
			ival = rb_funcall(INT2NUM(10), rb_intern("**"),
							  1, INT2NUM(scale));
			datum = rb_funcall(datum, rb_intern("*"), 1, ival);
		}
		else if (scale < 0)
		{
			ival = rb_funcall(INT2NUM(10), rb_intern("**"),
							  1, INT2NUM(-scale));
			datum = rb_funcall(datum, rb_intern("/"), 1, ival);
		}
		/* convert to integer */
		if (CLASS_OF(datum) != rb_cInteger)
			datum = rb_funcall(datum, rb_intern("to_i"), 0);
		/* overflow check */
		ival = rb_funcall(datum, rb_intern("bit_length"), 0);
		if (NUM2INT(ival) > 128)
			Elog("decimal value out of range");
		rb_integer_pack(datum, p_value, sizeof(int128_t), 1, 0,
						INTEGER_PACK_LITTLE_ENDIAN);
		return true;
	}
	else if (!retry)
	{
		/* convert to String once, if not yet */
		if (datum_klass != rb_cString)
			datum = rb_funcall(datum, rb_intern("to_s"), 0);
		/* then, convert to Retional */
		datum = rb_Rational1(datum);
		retry = true;
		goto retry_again;
	}
	Elog("cannot convert to decimal value");
}

static size_t
put_ruby_decimal_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	int128_t	value = 0;

	if (!__ruby_fetch_decimal_value((VALUE)addr, &value,
									column->arrow_type.Decimal.scale))
		__put_inline_null_value(column, row_index, sizeof(int128_t));
	else
	{
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(int128_t));

		STAT_UPDATES(column,i128,value);
	}
	return __buffer_usage_inline_type(column);
}

/*
 * common routine to fetch date/time value
 */
static bool
__ruby_fetch_timestamp_value(VALUE datum,
							 uint64_t *p_sec,	/* seconds from UTC */
							 uint64_t *p_nsec,	/* nano-seconds in the day */
							 bool convert_to_utc)
{
	const char *cname;
	VALUE		sec;
	VALUE		nsec;
	bool		retry = false;

	if (datum == Qnil)
		return false;	/* NULL */
try_again:
	/* Is it Fluent::EventTime? */
	cname = rb_class2name(CLASS_OF(datum));
	if (strcmp(cname, "EventTime") == 0 &&
		rb_respond_to(datum, rb_intern("sec")) &&
		rb_respond_to(datum, rb_intern("nsec")))
	{
		sec = rb_funcall(datum, rb_intern("sec"), 0);
		nsec = rb_funcall(datum, rb_intern("nsec"), 0);

		*p_sec = NUM2ULONG(sec);
		*p_nsec = NUM2ULONG(nsec);
		return true;
	}
	/* Is it Integer (elapsed seconds from Epoch) */
	if (CLASS_OF(datum) == rb_cInteger)
	{
		*p_sec = NUM2ULONG(datum);
		*p_nsec = 0;
		return true;
	}
	/* Is convertible to UTC? (should happen only once) */
	if (convert_to_utc && rb_respond_to(datum, rb_intern("getutc")))
	{
		datum = rb_funcall(datum, rb_intern("getutc"), 0);
		convert_to_utc = false;
	}
	/* Is it Time? */
	if (rb_respond_to(datum, rb_intern("tv_sec")) &&
		rb_respond_to(datum, rb_intern("tv_nsec")))
	{
		sec = rb_funcall(datum, rb_intern("tv_sec"), 0);
		nsec = rb_funcall(datum, rb_intern("tv_nsec"), 0);

		*p_sec = NUM2ULONG(sec);
		*p_nsec = NUM2ULONG(nsec);
		return true;
	}
	/* Convertible to Time? (maybe String or Date) */
	if (rb_respond_to(datum, rb_intern("to_time")))
		datum = rb_funcall(datum, rb_intern("to_time"), 0);
	else
	{
		/* elsewhere, convert to String, then create a new Time object */
		datum = rb_funcall(datum, rb_intern("to_s"), 0);
		datum = rb_funcall(rb_cTime, rb_intern("parse"), 1, datum);
	}

	if (!retry)
	{
		retry = true;
		goto try_again;
	}
	Elog("unable to extract sec/nsec from the supplied object");
}

/*
 * Date
 */
static size_t
put_ruby_date_day_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint64_t	sec;
	uint64_t	nsec;

	if (!__ruby_fetch_timestamp_value((VALUE)addr, &sec, &nsec, false))
		__put_inline_null_value(column, row_index, sizeof(int32_t));
	else
	{
		uint32_t	value = sec / SECS_PER_DAY;

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(uint32_t));

		STAT_UPDATES(column,u32,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_ruby_date_ms_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint64_t	sec;
	uint64_t	nsec;

	if (!__ruby_fetch_timestamp_value((VALUE)addr, &sec, &nsec, false))
		__put_inline_null_value(column, row_index, sizeof(int64_t));
	else
	{
		uint64_t	value = (sec * 1000L) + (nsec / 1000000L);

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(uint64_t));

		STAT_UPDATES(column,u64,value);
	}
	return __buffer_usage_inline_type(column);
}

/*
 * Time
 */
static size_t
put_ruby_time_sec_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint64_t	sec;
	uint64_t	nsec;

	if (!__ruby_fetch_timestamp_value((VALUE)addr, &sec, &nsec, false))
		__put_inline_null_value(column, row_index, sizeof(uint32_t));
	else
	{
		uint32_t	value = sec % SECS_PER_DAY;

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(uint32_t));

		STAT_UPDATES(column,u32,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_ruby_time_ms_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint64_t	sec;
	uint64_t	nsec;

	if (!__ruby_fetch_timestamp_value((VALUE)addr, &sec, &nsec, false))
		__put_inline_null_value(column, row_index, sizeof(uint32_t));
	else
	{
		uint32_t	value = (sec % SECS_PER_DAY) * 1000 + (nsec / 1000000);

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(uint32_t));

		STAT_UPDATES(column,u32,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_ruby_time_us_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint64_t	sec;
	uint64_t	nsec;

	if (!__ruby_fetch_timestamp_value((VALUE)addr, &sec, &nsec, false))
		__put_inline_null_value(column, row_index, sizeof(uint64_t));
	else
	{
		uint64_t	value = (sec % SECS_PER_DAY) * 1000000L + (nsec / 1000L);

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(uint64_t));

		STAT_UPDATES(column,u64,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_ruby_time_ns_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint64_t	sec;
	uint64_t	nsec;

	if (!__ruby_fetch_timestamp_value((VALUE)addr, &sec, &nsec, false))
		__put_inline_null_value(column, row_index, sizeof(int64_t));
	else
	{
		uint64_t	value = (sec % SECS_PER_DAY) * 1000000000L + nsec;

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(uint64_t));

		STAT_UPDATES(column,u64,value);
	}
	return __buffer_usage_inline_type(column);
}

/*
 * Timestamp
 */
static size_t
put_ruby_timestamp_sec_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint64_t	sec;
	uint64_t	nsec;

	if (!__ruby_fetch_timestamp_value((VALUE)addr, &sec, &nsec, true))
		__put_inline_null_value(column, row_index, sizeof(int64_t));
	else
	{
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &sec, sizeof(uint64_t));

		STAT_UPDATES(column,u64,sec);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_ruby_timestamp_ms_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint64_t	sec;
	uint64_t	nsec;

	if (!__ruby_fetch_timestamp_value((VALUE)addr, &sec, &nsec, true))
		__put_inline_null_value(column, row_index, sizeof(int64_t));
	else
	{
		uint64_t	value = sec * 1000L + nsec / 1000000L;

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(uint64_t));

		STAT_UPDATES(column,u64,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_ruby_timestamp_us_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint64_t	sec;
	uint64_t	nsec;

	if (!__ruby_fetch_timestamp_value((VALUE)addr, &sec, &nsec, true))
		__put_inline_null_value(column, row_index, sizeof(int64_t));
	else
	{
		uint64_t	value = sec * 1000000L + nsec / 1000L;

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(uint64_t));

		STAT_UPDATES(column,u64,value);
	}
	return __buffer_usage_inline_type(column);
}

static size_t
put_ruby_timestamp_ns_value(SQLfield *column, const char *addr, int sz)
{
	size_t		row_index = column->nitems++;
	uint64_t	sec;
	uint64_t	nsec;

	if (!__ruby_fetch_timestamp_value((VALUE)addr, &sec, &nsec, true))
		__put_inline_null_value(column, row_index, sizeof(int64_t));
	else
	{
		uint64_t	value = sec * 1000000000L + nsec;

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, &value, sizeof(uint64_t));

		STAT_UPDATES(column,u64,value);
	}
	return __buffer_usage_inline_type(column);
}

/*
 * Utf8 (String)
 */
static size_t
put_ruby_utf8_value(SQLfield *column, const char *addr, int sz)
{
	static VALUE utf8_encoding = Qnil;
	VALUE		datum = (VALUE)addr;
	size_t		row_index = column->nitems++;

	if (row_index == 0)
		sql_buffer_append_zero(&column->values, sizeof(uint32_t));
	if (datum == Qnil)
	{
		column->nullcount++;
		sql_buffer_clrbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values,
						  &column->extra.usage, sizeof(uint32_t));
	}
	else
	{
		VALUE		encoding;

		if (TYPE(datum) != T_STRING)
			datum = rb_funcall(datum, rb_intern("to_s"), 0);
		if (utf8_encoding == Qnil)
		{
			VALUE	klass = rb_path2class("Encoding");

			utf8_encoding = rb_const_get(klass, rb_intern("UTF_8"));
		}
		/* force to convert UTF-8 string, if needed */
		encoding = rb_funcall(datum, rb_intern("encoding"), 0);
		if (encoding != utf8_encoding)
			datum = rb_funcall(datum, rb_intern("encode"), 1, utf8_encoding);

		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->extra,
						  RSTRING_PTR(datum),
						  RSTRING_LEN(datum));
		sql_buffer_append(&column->values,
						  &column->extra.usage, sizeof(uint32_t));
	}
	return __buffer_usage_varlena_type(column);
}

/*
 * common routine to fetch ip address
 */
static bool
__ruby_fetch_ipaddr_value(VALUE datum, unsigned char *buf, int ip_version)
{
	bool		retry = false;

	if (datum == Qnil)
		return false;	/* NULL */

retry_again:
	/* Is IPAddr object? */
	if (rb_respond_to(datum, rb_intern("ipv4?")) &&
		rb_respond_to(datum, rb_intern("ipv6?")) &&
		rb_respond_to(datum, rb_intern("to_i")))
	{
		VALUE	bval;
		VALUE	ival;

		if ((ip_version == 4 || ip_version < 0) &&
			(bval = rb_funcall(datum, rb_intern("ipv4?"), 0)) == Qtrue)
		{
			ival = rb_funcall(datum, rb_intern("to_i"), 0);

			rb_integer_pack(ival, buf, IP4ADDR_LEN, 1, 0,
							INTEGER_PACK_BIG_ENDIAN);
			return true;
		}
		if ((ip_version == 6 || ip_version < 0) &&
			(bval = rb_funcall(datum, rb_intern("ipv6?"), 0)) == Qtrue)
		{
			ival = rb_funcall(datum, rb_intern("to_i"), 0);

			rb_integer_pack(ival, buf, IP6ADDR_LEN, 1, 0,
							INTEGER_PACK_BIG_ENDIAN);
			return true;
		}
		Elog("IPAddr is not IPv%d format", ip_version);
	}

	/* Elsewhere try to convert to IPAddr */
	if (!retry)
	{
		static VALUE ipaddr_klass = Qnil;

		/* Load 'ipaddr' module once */
		if (ipaddr_klass == Qnil)
		{
			rb_require("ipaddr");
			ipaddr_klass = rb_path2class("IPAddr");
		}
		if (TYPE(datum) != T_STRING)
			datum = rb_funcall(datum, rb_intern("to_s"), 0);
		datum = rb_class_new_instance(1, &datum, ipaddr_klass);
		retry = true;
		goto retry_again;
	}
	Elog("unable to convert datum to logical Arrow::Ipaddr4/6");
}


/*
 * Ipaddr4
 */
static size_t
put_ruby_logical_ip4addr_value(SQLfield *column, const char *addr, int sz)
{
	size_t	row_index = column->nitems++;
	unsigned char buf[IP6ADDR_LEN];

	if (!__ruby_fetch_ipaddr_value((VALUE)addr, buf, 4))
		__put_inline_null_value(column, row_index, IP4ADDR_LEN);
	else
	{
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, buf, IP4ADDR_LEN);
	}
	return __buffer_usage_inline_type(column);
}

/*
 * Ipaddr6
 */
static size_t
put_ruby_logical_ip6addr_value(SQLfield *column, const char *addr, int sz)
{
	size_t	row_index = column->nitems++;
	unsigned char buf[IP6ADDR_LEN];

	if (!__ruby_fetch_ipaddr_value((VALUE)addr, buf, 6))
		__put_inline_null_value(column, row_index, IP6ADDR_LEN);
	else
	{
		sql_buffer_setbit(&column->nullmap, row_index);
		sql_buffer_append(&column->values, buf, IP6ADDR_LEN);
	}
	return __buffer_usage_inline_type(column);
}

/* ----------------------------------------------------------------
 *
 * Write out min/max statistics handler
 *
 * ---------------------------------------------------------------- */
static int
write_ruby_int8_stat(SQLfield *column, char *buf, size_t len,
					 const SQLstat__datum *stat_datum)
{
	return snprintf(buf, len, "%d", (int32_t)stat_datum->i8);
}

static int
write_ruby_int16_stat(SQLfield *column, char *buf, size_t len,
					  const SQLstat__datum *stat_datum)
{
	return snprintf(buf, len, "%d", (int32_t)stat_datum->i16);
}

static int
write_ruby_int32_stat(SQLfield *column, char *buf, size_t len,
					  const SQLstat__datum *stat_datum)
{
	return snprintf(buf, len, "%d", stat_datum->i32);
}

static int
write_ruby_int64_stat(SQLfield *column, char *buf, size_t len,
					  const SQLstat__datum *stat_datum)
{
	return snprintf(buf, len, "%ld", stat_datum->i64);
}

static int
write_ruby_int128_stat(SQLfield *column, char *buf, size_t len,
					   const SQLstat__datum *stat_datum)
{
	int128_t	ival = stat_datum->i128;
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

static int
write_ruby_uint8_stat(SQLfield *column, char *buf, size_t len,
					  const SQLstat__datum *stat_datum)
{
	return snprintf(buf, len, "%u", (uint32_t)stat_datum->i8);
}

static int
write_ruby_uint16_stat(SQLfield *column, char *buf, size_t len,
					   const SQLstat__datum *stat_datum)
{
	return snprintf(buf, len, "%u", (uint32_t)stat_datum->i16);
}

static int
write_ruby_uint32_stat(SQLfield *column, char *buf, size_t len,
					   const SQLstat__datum *stat_datum)
{
	return snprintf(buf, len, "%u", (uint32_t)stat_datum->i32);
}


static int
write_ruby_uint64_stat(SQLfield *column, char *buf, size_t len,
					   const SQLstat__datum *stat_datum)
{
	return snprintf(buf, len, "%lu", (uint64_t)stat_datum->i64);
}

/* ----------------------------------------------------------------
 *
 * Routines related to initializer
 *
 * ----------------------------------------------------------------
 */
static void
__arrowFileWritePathnameValidator(VALUE self, VALUE __pathname)
{
	const char *str;
	uint32_t	len, i;
	VALUE		pathname;

	pathname = rb_funcall(__pathname, rb_intern("to_s"), 0);
	str = RSTRING_PTR(pathname);
	len = RSTRING_LEN(pathname);
	if (len == 0)
		Elog("pathname must not be empty");
	if (*str != '/')
		Elog("pathname must be absolute path: %.*s", len, str);

	for (i=0; i < len; i++)
	{
		if (str[i] != '%')
			continue;
		if (++i >= len)
			Elog("invalid pathname configuration: %.*s", len, str);
		switch (str[i])
		{
			case 'Y':	/* Year (4-digits) */
			case 'y':	/* Year (2-digits) */
			case 'm':	/* month (01-12) */
			case 'd':	/* day (01-31) */
			case 'H':	/* hour (00-23) */
			case 'M':	/* minute (00-59) */
			case 'S':	/* second (00-59) */
			case 'p':	/* process's PID */
			case 'q':	/* sequence number */
				break;
			default:
				Elog("unknown format character: '%c' in '%.*s'",
					 str[i], len, str);
		}
	}
	rb_ivar_set(self, rb_intern("pathname"), pathname);
}

static void
arrowFieldAddCustomMetadata(SQLfield *column,
                            const char *key,
                            const char *value)
{
	ArrowKeyValue *kv;
	size_t		sz;

	sz = sizeof(ArrowKeyValue) * (column->numCustomMetadata + 1);
	column->customMetadata = repalloc(column->customMetadata, sz);
	kv = &column->customMetadata[column->numCustomMetadata++];
	initArrowNode(kv, KeyValue);
	kv->key = pstrdup(key);
	kv->_key_len = strlen(key);
	kv->value = pstrdup(value);
	kv->_value_len = strlen(value);
}

static int
__assignFieldTypeBool(SQLfield *column)
{
	initArrowNode(&column->arrow_type, Bool);
	column->put_value = ruby_put_bool_value;

	return 2;		/* nullmap + values */
}

static int
__assignFieldTypeInt(SQLfield *column, const char *extra)
{
	initArrowNode(&column->arrow_type, Int);
	column->arrow_type.Int.is_signed = true;
	if (strcmp(extra, "8") == 0)
	{
		column->arrow_type.Int.bitWidth = 8;
		column->put_value = put_ruby_int8_value;
		column->write_stat = write_ruby_int8_stat;
	}
	else if (strcmp(extra, "16") == 0)
	{
		column->arrow_type.Int.bitWidth = 16;
		column->put_value = put_ruby_int16_value;
		column->write_stat = write_ruby_int16_stat;
	}
	else if (strcmp(extra, "32") == 0)
	{
		column->arrow_type.Int.bitWidth = 32;
		column->put_value = put_ruby_int32_value;
		column->write_stat = write_ruby_int32_stat;
	}
	else if (strcmp(extra, "64") == 0)
	{
		column->arrow_type.Int.bitWidth = 64;
		column->put_value = put_ruby_int64_value;
		column->write_stat = write_ruby_int64_stat;
	}
	else
		Elog("Not a supported Int width (%s)", extra);

	return 2;	/* nullmap + values */
}

static int
__assignFieldTypeUint(SQLfield *column, const char *extra)
{
	initArrowNode(&column->arrow_type, Int);
	column->arrow_type.Int.is_signed = false;
	if (strcmp(extra, "8") == 0)
	{
		column->arrow_type.Int.bitWidth = 8;
		column->put_value = put_ruby_uint8_value;
		column->write_stat = write_ruby_uint8_stat;
	}
	else if (strcmp(extra, "16") == 0)
	{
		column->arrow_type.Int.bitWidth = 16;
		column->put_value = put_ruby_uint16_value;
		column->write_stat = write_ruby_uint16_stat;
	}
	else if (strcmp(extra, "32") == 0)
	{
		column->arrow_type.Int.bitWidth = 32;
		column->put_value = put_ruby_uint32_value;
		column->write_stat = write_ruby_uint32_stat;
	}
	else if (strcmp(extra, "64") == 0)
	{
		column->arrow_type.Int.bitWidth = 64;
		column->put_value = put_ruby_uint64_value;
		column->write_stat = write_ruby_uint64_stat;
	}
	else
		Elog("Not a supported Uint width (%s)", extra);

	return 2;	/* nullmap + values */
}

static int
__assignFieldTypeFloatingPoint(SQLfield *column, const char *extra)
{
	initArrowNode(&column->arrow_type, FloatingPoint);
	if (strcmp(extra, "16") == 0)
	{
		column->arrow_type.FloatingPoint.precision
			= ArrowPrecision__Half;
		column->put_value = put_ruby_float16_value;
		column->write_stat = write_ruby_int16_stat;
	}
	else if (strcmp(extra, "32") == 0)
	{
		column->arrow_type.FloatingPoint.precision
			= ArrowPrecision__Single;
		column->put_value = put_ruby_float32_value;
		column->write_stat = write_ruby_int32_stat;
	}
	else if (strcmp(extra, "64") == 0)
	{
		column->arrow_type.FloatingPoint.precision
			= ArrowPrecision__Double;
		column->put_value = put_ruby_float64_value;
		column->write_stat = write_ruby_int64_stat;
	}
	else
		Elog("Not a supported FloatingPoint width (%s)", extra);

	return 2;	/* nullmap + values */
}

static int
__assignFieldTypeDecimal(SQLfield *column, const char *extra)
{
	int		bitWidth = 128;
	int		precision = 30;
	int		scale = 8;

	if (strncmp(extra, "128", 3) == 0)
		extra += 3;
	if (*extra == '(')
	{
		const char *tail = extra + strlen(extra) - 1;

		if (*tail != ')')
			Elog("Arrow::Decimal definition syntax error");
		if (sscanf(extra, "(%d)", &scale) != 1)
			Elog("Arrow::Decimal definition syntax error");
	}
	else if (*extra != '\0')
		Elog("Arrow::Decimal definition syntax error");

	initArrowNode(&column->arrow_type, Decimal);
	column->arrow_type.Decimal.precision = precision;
	column->arrow_type.Decimal.scale = scale;
	column->arrow_type.Decimal.bitWidth = bitWidth;
	column->put_value = put_ruby_decimal_value;
	column->write_stat = write_ruby_int128_stat;

	return 2;	/* nullmap + values */
}

static int
__assignFieldTypeDate(SQLfield *column, const char *extra)
{
	initArrowNode(&column->arrow_type, Date);
	if (strcmp(extra, "[day]") == 0 || strcmp(extra, "") == 0)
	{
		column->arrow_type.Date.unit = ArrowDateUnit__Day;
		column->put_value = put_ruby_date_day_value;
		column->write_stat = write_ruby_int32_stat;
	}
	else if (strcmp(extra, "[ms]") == 0)
	{
		column->arrow_type.Date.unit = ArrowDateUnit__MilliSecond;
		column->put_value = put_ruby_date_ms_value;
		column->write_stat = write_ruby_int64_stat;
	}
	else
		Elog("Arrow::Date - not a supported unit size: %s", extra);

	return 2;
}

static int
__assignFieldTypeTime(SQLfield *column, const char *extra)
{
	initArrowNode(&column->arrow_type, Time);
	if (strcmp(extra, "[sec]") == 0 || strcmp(extra, "") == 0)
	{
		column->arrow_type.Time.unit = ArrowTimeUnit__Second;
		column->arrow_type.Time.bitWidth = 32;
		column->put_value = put_ruby_time_sec_value;
		column->write_stat = write_ruby_int32_stat;
	}
	else if (strcmp(extra, "[ms]") == 0)
	{
		column->arrow_type.Time.unit = ArrowTimeUnit__MilliSecond;
		column->arrow_type.Time.bitWidth = 32;
		column->put_value = put_ruby_time_ms_value;
		column->write_stat = write_ruby_int32_stat;
	}
	else if (strcmp(extra, "[us]") == 0)
	{
		column->arrow_type.Time.unit = ArrowTimeUnit__MicroSecond;
		column->arrow_type.Time.bitWidth = 64;
		column->put_value = put_ruby_time_us_value;
		column->write_stat = write_ruby_int64_stat;
	}
	else if (strcmp(extra, "[ns]") == 0)
	{
		column->arrow_type.Time.unit = ArrowTimeUnit__NanoSecond;
		column->arrow_type.Time.bitWidth = 64;
		column->put_value = put_ruby_time_ns_value;
		column->write_stat = write_ruby_int64_stat;
	}
	else
		Elog("Arrow::Time - not a supported unit size: %s", extra);

	return 2;	/* nullmap + values */
}

static int
__assignFieldTypeTimestamp(SQLfield *column, const char *extra)
{
	initArrowNode(&column->arrow_type, Timestamp);
	/* with timezone? */
	if (strncmp(extra, "Tz", 2) == 0)
	{
		struct tm	tm;
		time_t		t;

		t = time(NULL);
		localtime_r(&t, &tm);
		if (tm.tm_zone)
		{
			column->arrow_type.Timestamp.timezone = pstrdup(tm.tm_zone);
			column->arrow_type.Timestamp._timezone_len = strlen(tm.tm_zone);
		}
		extra += 2;
	}

	if (strcmp(extra, "[sec]") == 0)
	{
		column->arrow_type.Timestamp.unit = ArrowTimeUnit__Second;
		column->put_value = put_ruby_timestamp_sec_value;
		column->write_stat = write_ruby_int64_stat;
	}
	else if (strcmp(extra, "[ms]") == 0)
	{
		column->arrow_type.Timestamp.unit = ArrowTimeUnit__MilliSecond;
		column->put_value = put_ruby_timestamp_ms_value;
		column->write_stat = write_ruby_int64_stat;
	}
	else if (strcmp(extra, "[us]") == 0 || strcmp(extra,"") == 0)
	{
		column->arrow_type.Timestamp.unit = ArrowTimeUnit__MicroSecond;
		column->put_value = put_ruby_timestamp_us_value;
		column->write_stat = write_ruby_int64_stat;
	}
	else if (strcmp(extra, "[ns]") == 0)
	{
		column->arrow_type.Timestamp.unit = ArrowTimeUnit__NanoSecond;
        column->put_value = put_ruby_timestamp_ns_value;
        column->write_stat = write_ruby_int64_stat;
	}
	else
		Elog("Arrow::Timestamp - not a supported unit size: %s", extra);

	return 2;
}

static int
__assignFieldTypeUtf8(SQLfield *column)
{
	initArrowNode(&column->arrow_type, Utf8);
	column->put_value = put_ruby_utf8_value;

	return 3;	/* nullmap + index + extra */
}

static int
__assignFieldTypeIpaddr4(SQLfield *column)
{
	initArrowNode(&column->arrow_type, FixedSizeBinary);
	column->arrow_type.FixedSizeBinary.byteWidth = IP4ADDR_LEN;
	column->put_value = put_ruby_logical_ip4addr_value;
	arrowFieldAddCustomMetadata(column, "pg_type", "pg_catalog.inet");

	return 2;	/* nullmap + values */
}

static int
__assignFieldTypeIpaddr6(SQLfield *column)
{
	initArrowNode(&column->arrow_type, FixedSizeBinary);
	column->arrow_type.FixedSizeBinary.byteWidth = IP6ADDR_LEN;
	column->put_value = put_ruby_logical_ip6addr_value;
	arrowFieldAddCustomMetadata(column, "pg_type", "pg_catalog.inet");

	return 2;	/* nullmap + values */
}

static int
__arrowFileAssignFieldType(SQLfield *column,
						   const char *field_name,
						   const char *field_type,
						   bool stat_enabled,
						   bool ts_column,
						   bool tag_column)
{
	column->field_name = pstrdup(field_name);
	column->stat_enabled = stat_enabled;
	column->sql_type.fluent.ts_column = ts_column;
	column->sql_type.fluent.tag_column = tag_column;

	if (strcmp(field_type, "Bool") == 0)
		return __assignFieldTypeBool(column);
	else if (strncmp(field_type, "Int", 3) == 0)
		return  __assignFieldTypeInt(column, field_type + 3);
	else if (strncmp(field_type, "Uint", 4) == 0)
		return  __assignFieldTypeUint(column, field_type + 4);
	else if (strncmp(field_type, "Float", 5) == 0)
		return  __assignFieldTypeFloatingPoint(column, field_type + 5);
	else if (strncmp(field_type, "Decimal", 7) == 0)
		return  __assignFieldTypeDecimal(column, field_type + 7);
	else if (strncmp(field_type, "Timestamp", 9) == 0)
		return  __assignFieldTypeTimestamp(column, field_type + 9);
	else if (strncmp(field_type, "Date", 4) == 0)
		return  __assignFieldTypeDate(column, field_type + 4);
	else if (strncmp(field_type, "Time", 4) == 0)
		return  __assignFieldTypeTime(column, field_type + 4);
    else if (strcmp(field_type, "Utf8") == 0)
		return  __assignFieldTypeUtf8(column);
	else if (strcmp(field_type, "Ipaddr4") == 0)
		return  __assignFieldTypeIpaddr4(column);
	else if (strcmp(field_type, "Ipaddr6") == 0)
		return  __assignFieldTypeIpaddr6(column);
	Elog("ArrowFileWrite: not a supported type");
}

/*
 * Parsing the Schema Definition
 */
static void
__arrowFileWriteParseSchemaDefs(VALUE self, VALUE __schema_defs)
{
	VALUE		schema_defs;
	VALUE		schema = Qnil;
	char	   *buf;
	int			len;
	char	   *tok, *saveptr;

	schema_defs = rb_funcall(__schema_defs, rb_intern("to_s"), 0);
	len = RSTRING_LEN(schema_defs);
	buf = alloca(len+1);
	memcpy(buf, RSTRING_PTR(schema_defs), len);
	buf[len] = '\0';

	for (tok = strtok_r(buf, ",", &saveptr);
		 tok != NULL;
		 tok = strtok_r(NULL, ",", &saveptr))
	{
		/* <column_name>=<column_type>[;<column_attr>;...] */
		VALUE		hash = rb_hash_new();
		char	   *field_name = tok;
		char	   *field_type;
		char	   *__extra;
		char	   *__tok, *__saveptr;
		bool		stat_enabled = false;
		SQLfield	__dummy;

		field_type = strchr(field_name, '=');
		if (!field_type)
			Elog("syntax error in schema definition");
		*field_type++ = '\0';

		__extra = strchr(field_type, ';');
		if (__extra)
		{
			*__extra++ = '\0';
			for (__tok = strtok_r(__extra, ";", &__saveptr);
				 __tok != NULL;
				 __tok = strtok_r(NULL, ";", &__saveptr))
			{
				char   *attr = trim_cstring(__tok);

				if (strcmp(attr, "stat_enabled") == 0)
				{
					if (stat_enabled)
						Elog("duplicated column attribute: %s", attr);
					stat_enabled = true;
				}
				else
				{
					Elog("unknown column attribute: %s", attr);
				}
			}
		}
		field_name = trim_cstring(field_name);
		field_type = trim_cstring(field_type);

		/* validation */
		__arrowFileAssignFieldType(&__dummy,
								   field_name,
								   field_type,
								   stat_enabled,
								   false,
								   false);

		rb_funcall(hash, rb_intern("store"), 2,
				   rb_str_new_cstr("name"),
				   rb_str_new_cstr(field_name));
		rb_funcall(hash, rb_intern("store"), 2,
				   rb_str_new_cstr("type"),
				   rb_str_new_cstr(field_type));
		rb_funcall(hash, rb_intern("store"), 2,
				   rb_str_new_cstr("stat_enabled"),
				   stat_enabled ? Qtrue : Qfalse);

		if (schema == Qnil)
			schema = rb_ary_new();
		rb_funcall(schema, rb_intern("append"), 1, hash);
	}
	if (schema == Qnil)
		Elog("no valid schema definition");
	rb_ivar_set(self, rb_intern("schema"), schema);
}

static void
__arrowFileWriteParseParams(VALUE self,
							VALUE __params)
{
	VALUE		datum;
	VALUE		schema;
	VALUE		ts_column = Qnil;
	VALUE		tag_column = Qnil;
	long		f_threshold = 10000;
	int			i, count;

	if (CLASS_OF(__params) == rb_cHash)
	{
		datum = rb_funcall(__params, rb_intern("fetch"), 2,
						   rb_str_new_cstr("ts_column"), Qnil);
		if (datum != Qnil)
			ts_column = rb_funcall(datum, rb_intern("to_s"), 0);

		datum = rb_funcall(__params, rb_intern("fetch"), 2,
						   rb_str_new_cstr("tag_column"), Qnil);
		if (datum != Qnil)
			tag_column = rb_funcall(datum, rb_intern("to_s"), 0);

		datum = rb_funcall(__params, rb_intern("fetch"), 2,
						   rb_str_new_cstr("filesize_threshold"), Qnil);
		if (datum != Qnil)
		{
			datum = rb_funcall(datum, rb_intern("to_i"), 0);
			f_threshold = NUM2LONG(datum);
			if (f_threshold < 16 || f_threshold > 1048576)
				Elog("filesize_threshold must be [16...1048576]");
		}
	}
	else if (__params != Qnil)
		Elog("ArrowFileWrite: parameters must be Hash");

	schema = rb_ivar_get(self, rb_intern("schema"));
	datum = rb_funcall(schema, rb_intern("count"), 0);
	count = NUM2INT(datum);

	for (i=0; i < count; i++)
	{
		VALUE	field;
		VALUE	fname;

		field = rb_funcall(schema, rb_intern("fetch"),
						   1, INT2NUM(i));
		fname = rb_funcall(field,  rb_intern("fetch"),
						   1, rb_str_new_cstr("name"));
		if (ts_column != Qnil &&
			rb_funcall(fname, rb_intern("=="), 1, ts_column) == Qtrue)
		{
			rb_funcall(field, rb_intern("store"), 2,
					   rb_str_new_cstr("ts_column"), Qtrue);
		}
		else if (tag_column != Qnil &&
				 rb_funcall(fname, rb_intern("=="), 1, tag_column) == Qtrue)
		{
			rb_funcall(field, rb_intern("store"), 2,
					   rb_str_new_cstr("tag_column"), Qtrue);
		}
	}
	rb_ivar_set(self, rb_intern("filesize_threshold"),
				LONG2NUM(f_threshold << 20));
}

static VALUE
rb_ArrowFileWrite__initialize(VALUE self,
							  VALUE __pathname,
							  VALUE __schema_defs,
							  VALUE __params)
{
	rb_require("time");

	__arrowFileWritePathnameValidator(self, __pathname);
	__arrowFileWriteParseSchemaDefs(self, __schema_defs);
	__arrowFileWriteParseParams(self, __params);

	return self;
}

/* ----------------------------------------------------------------
 *
 * Routines related to ArrowFile::open / close
 *
 * ----------------------------------------------------------------
 */
static void
__arrowFileSwitchFile(SQLtable *table, struct stat *st_buf_new)
{
	struct stat	st_buf_cur;
	char   *d_name, *__d_buf;
	char   *b_name, *__b_buf;
	int		d_desc = -1;

	__d_buf = alloca(strlen(table->filename) + 1);
	strcpy(__d_buf, table->filename);
	d_name = dirname(__d_buf);

	__b_buf = alloca(strlen(table->filename) + 1);
	strcpy(__b_buf, table->filename);
	b_name = basename(__b_buf);

	d_desc = open(d_name, O_RDONLY | O_DIRECTORY);
	if (d_desc < 0)
		Elog("failed on open('%s'): %m", d_name);
	if (flock(d_desc, LOCK_EX) != 0)
	{
		close(d_desc);
		Elog("failed on flock('%s'): %m", d_name);
	}
	/* <-- exclusive lock on the directory --> */
	if (stat(table->filename, &st_buf_cur) != 0)
	{
		close(d_desc);
		Elog("failed on stat('%s'): %m", table->filename);
	}
	/* pathname is not renamed yet? */
	if (st_buf_cur.st_dev == st_buf_new->st_dev &&
		st_buf_cur.st_ino == st_buf_new->st_ino)
	{
		char   *n_name = alloca(strlen(b_name) + 100);
		int		suffix;

		for (suffix = 1; ; suffix++)
		{
			sprintf(n_name, "%s.%d", b_name, suffix);
			if (faccessat(d_desc, n_name, F_OK, 0) != 0)
			{
				if (errno != ENOENT)
				{
					close(d_desc);
					Elog("failed on faccessat('%s/%s'): %m", d_name, n_name);
				}
				if (renameat(d_desc, b_name,
							 d_desc, n_name) != 0)
				{
					close(d_desc);
					Elog("failed on renameat('%s/%s' -> '%s/%s'): %m",
						 d_name, b_name,
						 d_name, n_name);
				}
				break;
			}
		}
	}
	close(d_desc);		/* also, unlock */	
}

static void
arrowFileCloseFile(SQLtable *table)
{
	if (table->fdesc >= 0)
	{
		close(table->fdesc);
		pfree((void *)table->filename);
		table->fdesc = -1;
		table->filename = NULL;
	}
}

static bool
arrowFileOpenFile(VALUE self, SQLtable *table)
{
	VALUE		pathname = rb_ivar_get(self, rb_intern("pathname"));
	VALUE		threshold = rb_ivar_get(self, rb_intern("filesize_threshold"));
	const char *str;
	char	   *buf = alloca(2000);
	uint32_t	bufsz = 2000;
	uint32_t	i, j, len;
	int			fdesc;
	time_t		__time;
	struct tm	tm;

	__time = time(NULL);
	localtime_r(&__time, &tm);

	assert(CLASS_OF(pathname) == rb_cString);
	str = RSTRING_PTR(pathname);
	len = RSTRING_LEN(pathname);
	for (i=0, j=0; i < len; i++)
	{
		int		c = str[i];

		assert(j + 20 < bufsz);
		if (c != '%')
			buf[j++] = c;
		else if (++i < len)
		{
			switch (str[i])
			{
				case 'Y':
					j += snprintf(buf+j, bufsz-j, "%04u", tm.tm_year + 1900);
					break;
				case 'y':
					j += snprintf(buf+j, bufsz-j, "%02u", tm.tm_year % 100);
					break;
				case 'm':
					j += snprintf(buf+j, bufsz-j, "%02u", tm.tm_mon + 1);
					break;
				case 'd':
					j += snprintf(buf+j, bufsz-j, "%02u", tm.tm_mday);
					break;
				case 'H':
					j += snprintf(buf+j, bufsz-j, "%02u", tm.tm_hour);
					break;
				case 'M':
					j += snprintf(buf+j, bufsz-j, "%02u", tm.tm_min);
					break;
				case 'S':
					j += snprintf(buf+j, bufsz-j, "%02u", tm.tm_sec);
					break;
				case 'p':
					j += snprintf(buf+j, bufsz-j, "%u", getpid());
					break;
				default:
					Elog("unknown format character at: %.*s", len, str);
			}
		}
		else
		{
			Elog("Bug? unclosed format charaster at: %.*s", len, str);
		}

		/* expand the buffer if little margin */
		if (j + 20 >= bufsz)
		{
			size_t	__bufsz = 2 * bufsz + 1000;
			char   *__buf = alloca(__bufsz);

			memcpy(__buf, buf, j);
			bufsz = __bufsz;
			buf = __buf;
		}
	}
	buf[j] = '\0';
	/* open the file */
	for (;;)
	{
		struct stat	stat_buf;

		fdesc = open(buf, O_RDWR | O_CREAT, 0644);
		if (fdesc < 0)
			Elog("ArrowWrite: failed to open '%s': %m", buf);
		table->fdesc = fdesc;
		table->filename = pstrdup(buf);
		if (flock(fdesc, LOCK_EX) != 0)
			Elog("failed on flock('%s'): %m", buf);
		if (fstat(fdesc, &stat_buf) != 0)
			Elog("failed on fstat('%s'): %m", buf);
		/* check threshold */
		if (stat_buf.st_size < NUM2LONG(threshold))
			return (stat_buf.st_size == 0);		/* true, if new file */
		/* file rotation, then retry */
		__arrowFileSwitchFile(table, &stat_buf);
		arrowFileCloseFile(table);
	}
}

static void
arrowFileSetupNewFile(SQLtable *table)
{
	arrowFileWrite(table, "ARROW1\0\0", 8);
	writeArrowSchema(table);
}

static SQLstat *
__arrowFieldParseStats(ArrowField *af_field)
{
	SQLstat	   *stat_values;
	char	   *min_values = NULL;
	char	   *max_values = NULL;
	char	   *tok1, *pos1;
	char	   *tok2, *pos2;
	uint32_t	nrooms = 500;
	uint32_t	i, nitems;

	for (i=0; i < af_field->_num_custom_metadata; i++)
	{
		ArrowKeyValue  *kv = &af_field->custom_metadata[i];

		if (strcmp(kv->key, "min_values") == 0)
		{
			min_values = alloca(kv->_value_len + 1);
			memcpy(min_values, kv->value, kv->_value_len);
			min_values[kv->_value_len] = '\0';
		}
		else if (strcmp(kv->key, "max_values") == 0)
		{
			max_values = alloca(kv->_value_len + 1);
			memcpy(max_values, kv->value, kv->_value_len);
			max_values[kv->_value_len] = '\0';
		}
	}
	if (!min_values || !max_values)
		Elog("column [%s] has no min/max statistics", af_field->name);

	stat_values = palloc(sizeof(SQLstat) * nrooms);
	for (tok1 = strtok_r(min_values, ",", &pos1),
		 tok2 = strtok_r(max_values, ",", &pos2), nitems = 0;
		 tok1 != NULL && tok2 != NULL;
		 tok1 = strtok_r(NULL, ",", &pos1),
		 tok2 = strtok_r(NULL, ",", &pos2), nitems++)
	{
		SQLstat	   *stat;
		bool		__isnull = false;

		if (nitems >= nrooms)
		{
			nrooms = 2 * nrooms;
			stat_values = repalloc(stat_values, sizeof(SQLstat) * nrooms);
		}
		stat = &stat_values[nitems];
		stat->next = NULL;	/* set later */
		stat->rb_index = nitems;
		stat->min.i128 = __atoi128(trim_cstring(tok1), &__isnull);
		stat->max.i128 = __atoi128(trim_cstring(tok2), &__isnull);
		stat->is_valid = !__isnull;
	}
	for (i=1; i < nitems; i++)
		stat_values[i-1].next = &stat_values[i];
	return stat_values;
}

static void
__arrowFileValidateColumn(SQLfield *column,
						  ArrowField *af_field,
						  int numRecordBatches)
{
	ArrowType  *c = &column->arrow_type;
	ArrowType  *f = &af_field->type;

	if (c->node.tag != f->node.tag)
		Elog("Field type mismatch [%s] <-> [%s]",
			 c->node.tagName, f->node.tagName);
	switch (c->node.tag)
	{
		case ArrowNodeTag__Bool:
			break;

		case ArrowNodeTag__Int:
			if ((c->Int.is_signed && !f->Int.is_signed) ||
				(!c->Int.is_signed && f->Int.is_signed))
				Elog("Int signed/unsigned mismatch");
			if (c->Int.bitWidth != f->Int.bitWidth)
				Elog("Int bitWidth mismatch");
			break;

		case ArrowNodeTag__FloatingPoint:
			if (c->FloatingPoint.precision != f->FloatingPoint.precision)
				Elog("FloatingPoint precision mismatch");
			break;

		case ArrowNodeTag__Utf8:
			break;

		case ArrowNodeTag__Decimal:
			if (c->Decimal.scale != f->Decimal.scale ||
				c->Decimal.bitWidth != f->Decimal.bitWidth)
				Elog("Not a compatible Decimal: Decimal%d(%d,%d) <-> Decimal%d(%d,%d)",
					 c->Decimal.bitWidth,
					 c->Decimal.precision,
					 c->Decimal.scale,
					 f->Decimal.bitWidth,
					 f->Decimal.precision,
					 f->Decimal.scale);
			break;

		case ArrowNodeTag__Date:
			if (c->Date.unit != f->Date.unit)
				Elog("Date has incompatible unit");
			break;

		case ArrowNodeTag__Time:
			if (c->Time.unit != f->Time.unit)
				Elog("Time has incompatible unit");
			break;

		case ArrowNodeTag__Timestamp:
			if (c->Timestamp.unit != f->Timestamp.unit)
				Elog("Timestamp has incompatible unit");
			break;

		case ArrowNodeTag__FixedSizeBinary:
			if (c->FixedSizeBinary.byteWidth != f->FixedSizeBinary.byteWidth)
				Elog("FixedSizeBinary has incompatible byteWidth");
			break;

		default:
			Elog("Bug? not a supported Arrow Type [%s]",
				 column->arrow_type.node.tagName);
	}

	if (column->stat_enabled)
		column->stat_list = __arrowFieldParseStats(af_field);
}

static void
arrowFileSetupAppend(SQLtable *table)
{
	ArrowFileInfo af_info;
	uint32_t	nitems;
	size_t		nbytes;
	size_t		offset;
	int			j;
	char		buffer[100];

	readArrowFileDesc(table->fdesc, &af_info);
	if (table->nfields != af_info.footer.schema._num_fields)
		Elog("number of fields mismatch %d <-> %d at %s",
			 table->nfields,
			 af_info.footer.schema._num_fields,
			 table->filename);
	for (j=0; j < table->nfields; j++)
		__arrowFileValidateColumn(&table->columns[j],
								  &af_info.footer.schema.fields[j],
								  af_info.footer._num_recordBatches);

	/* restore RecordBatches already in the file */
	nitems = af_info.footer._num_recordBatches;
	table->numRecordBatches = nitems;
	table->recordBatches = palloc(sizeof(ArrowBlock) * nitems);
	memcpy(table->recordBatches,
		   af_info.footer.recordBatches,
		   sizeof(ArrowBlock) * nitems);

	/* move to the file offset in front of the Footer */
	nbytes = sizeof(int32_t) + 6;   /* strlen("ARROW1") */
	offset = af_info.stat_buf.st_size - nbytes;
	if (pread(table->fdesc, buffer, nbytes, offset) != nbytes)
		Elog("failed on pread('%s'): %m", table->filename);
	offset -= *((uint32_t *)buffer);
	if (lseek(table->fdesc, offset, SEEK_SET) < 0)
		Elog("failed on lseek('%s'): %m", table->filename);
	table->f_pos = offset;
}

typedef struct
{
	VALUE		self;
	VALUE		chunk;
	SQLtable   *table;
} WriteChunkArgs;

static SQLtable *
__arrowFileCreateTable(VALUE self)
{
	VALUE		schema = rb_ivar_get(self, rb_intern("schema"));
	VALUE		field;
	VALUE		datum;
	int			j, count;
	int			nbuffers = 0;
	SQLtable   *table;

	datum = rb_funcall(schema, rb_intern("count"), 0);
	count = NUM2INT(datum);
	table = palloc0(offsetof(SQLtable, columns[count]));
	table->fdesc = -1;
	for (j=0; j < count; j++)
	{
		char   *__fname;
		char   *__ftype;
		VALUE	__stat_enabled;
		VALUE	__ts_column;
		VALUE	__tag_column;
		size_t	len;

		field = rb_funcall(schema, rb_intern("fetch"), 1, INT2NUM(j));
		datum = rb_funcall(field, rb_intern("fetch"),
						   1, rb_str_new_cstr("name"));
		len = RSTRING_LEN(datum);
		__fname = alloca(len+1);
		memcpy(__fname, RSTRING_PTR(datum), len);
		__fname[len] = '\0';

		datum = rb_funcall(field, rb_intern("fetch"),
						   1, rb_str_new_cstr("type"));
		len = RSTRING_LEN(datum);
		__ftype = alloca(len+1);
		memcpy(__ftype, RSTRING_PTR(datum), len);
		__ftype[len] = '\0';

		__stat_enabled = rb_funcall(field, rb_intern("fetch"),
									1, rb_str_new_cstr("stat_enabled"));
		__ts_column = rb_funcall(field, rb_intern("fetch"),
								 2, rb_str_new_cstr("ts_column"), Qfalse);
		__tag_column = rb_funcall(field, rb_intern("fetch"),
								  2, rb_str_new_cstr("tag_column"), Qfalse);

		nbuffers += __arrowFileAssignFieldType(&table->columns[j],
											   __fname,
											   __ftype,
											   __stat_enabled == Qtrue,
											   __ts_column == Qtrue,
											   __tag_column == Qtrue);
		if (__stat_enabled == Qtrue)
			table->has_statistics = true;
	}
	table->numFieldNodes = count;
	table->numBuffers = nbuffers;
	table->nfields = count;

	return table;
}

/*
 * __arrowFileReleaseTable
 *
 * release buffer memory unless wait for GC activation
 */
static void
__arrowFileReleaseTable(SQLtable *table)
{
	int		j;

	if (table->__iov)
		pfree(table->__iov);
	if (table->recordBatches)
		pfree(table->recordBatches);
	assert(!table->dictionaries);
	if (table->customMetadata)
		pfree(table->customMetadata);
	assert(!table->sql_dict_list);
	for (j=0; j < table->numFieldNodes; j++)
	{
		SQLfield   *column = &table->columns[j];

		if (column->field_name)
			pfree(column->field_name);
		assert(!column->element);
		assert(!column->subfields);
		if (column->nullmap.data)
			pfree(column->nullmap.data);
		if (column->values.data)
			pfree(column->values.data);
		if (column->extra.data)
			pfree(column->extra.data);
		if (column->customMetadata)
			pfree(column->customMetadata);
	}
}

static VALUE
__arrowFileWriteRow(RB_BLOCK_CALL_FUNC_ARGLIST(__yield, __private))
{
	WriteChunkArgs *args = (WriteChunkArgs *)__private;
	SQLtable   *table = args->table;
	VALUE		tag;
	VALUE		ts;
	VALUE		record;
	int			j;

	tag = rb_funcall(__yield, rb_intern("fetch"), 1, INT2NUM(0));
	ts = rb_funcall(__yield, rb_intern("fetch"), 1, INT2NUM(1));
	record = rb_funcall(__yield, rb_intern("fetch"), 1, INT2NUM(2));
	for (j=0; j < table->nfields; j++)
	{
		SQLfield   *column = &table->columns[j];
		VALUE		datum;

		if (column->sql_type.fluent.ts_column)
			datum = ts;
		else if (column->sql_type.fluent.tag_column)
			datum = tag;
		else
		{
			datum = rb_funcall(record, rb_intern("fetch"), 2,
							   rb_str_new_cstr(column->field_name), Qnil);
		}
		column->put_value(column, (const char *)datum, -1);
	}
	table->nitems++;

	return Qtrue;
}

static VALUE
__arrowFileWriteChunk(VALUE __args)
{
	WriteChunkArgs *args = (WriteChunkArgs *)__args;
	SQLtable   *table;

	/* setup SQLtable buffer */
	args->table = table = __arrowFileCreateTable(args->self);
	/* iterate chunk to fill up the buffer */
	rb_block_call(args->chunk,
				  rb_intern("each"),
				  0,
				  NULL,
				  __arrowFileWriteRow,
				  __args);
	/* open the destination file */
	if (arrowFileOpenFile(args->self, args->table))
		arrowFileSetupNewFile(args->table);
	else
		arrowFileSetupAppend(args->table);
	/* write out a new record-batch */
	writeArrowRecordBatch(table);
	/* write out a new footer */
	writeArrowFooter(table);
	/* close the file, and unlock */
	arrowFileCloseFile(table);

	return Qtrue;
}

static VALUE
rb_ArrowFileWrite__writeChunk(VALUE self,
							  VALUE chunk)
{
	WriteChunkArgs args;
	VALUE		retval;
	int			status;

	memset(&args, 0, sizeof(WriteChunkArgs));
	args.self  = self;
	args.chunk = chunk;

	retval = rb_protect(__arrowFileWriteChunk, (VALUE)&args, &status);
	if (status != 0)
	{
		if (args.table)
		{
			if (args.table->fdesc >= 0)
				close(args.table->fdesc);
			__arrowFileReleaseTable(args.table);
		}
		rb_jump_tag(status);
	}
	assert(args.table->fdesc < 0);
	__arrowFileReleaseTable(args.table);

	return retval;
}

void
Init_arrow_file_write(void)
{
	VALUE	klass;

	klass = rb_define_class("ArrowFileWrite",  rb_cObject);
	rb_define_method(klass, "initialize", rb_ArrowFileWrite__initialize, 3);
	rb_define_method(klass, "writeChunk", rb_ArrowFileWrite__writeChunk, 1);
}
