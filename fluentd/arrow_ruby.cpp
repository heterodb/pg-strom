/*
 * arrow_ruby.cpp
 *
 * A Ruby language extension to write out data as Apache Arrow files.
 * --
 * Copyright 2011-2025 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2025 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/util/value_parsing.h>
#include <ruby.h>
#include <ctype.h>
#include <limits.h>
#include <string>
#include "float2.h"

using	arrowField		= std::shared_ptr<arrow::Field>;
using	arrowBuilder	= std::shared_ptr<arrow::ArrayBuilder>;
using	arrowArray		= std::shared_ptr<arrow::Array>;

#define Elog(fmt,...)							\
	rb_raise(rb_eException, "%s:%d " fmt,		\
			 __FILE__, __LINE__, ##__VA_ARGS__)

static inline void *
palloc(size_t sz)
{
	return ruby_xmalloc(sz);
}
static inline void *
palloc0(size_t sz)
{
	void   *ptr = palloc(sz);
	memset(ptr, 0, sz);
	return ptr;
}
static inline char *
pstrdup(const char *str)
{
	char   *result = (char *)palloc(strlen(str)+1);
	strcpy(result, str);
	return result;
}
static inline char *
pstrdup(const char *str, uint32_t len)
{
	char   *result = (char *)palloc(len + 1);
	memcpy(result, str, len);
	result[len] = '\0';
	return result;
}
static inline void
pfree(void *ptr)
{
	ruby_xfree(ptr);
}
static inline char *
__trim(char *token)
{
	if (token)
	{
		char   *tail = token + strlen(token) - 1;

		while (isspace(*token))
			token++;
		while (tail >= token && isspace(*tail))
			*tail-- = '\0';
	}
	return token;
}

class arrowFileWrite {
public:
	arrowFileWrite()
	{
		pathname = nullptr;
		ts_column = nullptr;
		tag_column = nullptr;
		parquet_mode = false;
		default_compression = arrow::Compression::type::UNCOMPRESSED;
	}
	~arrowFileWrite()
	{
		if (pathname)
			pfree((void *)pathname);
		if (ts_column)
			pfree((void *)ts_column);
		if (tag_column)
			pfree((void *)tag_column);
	}
	const char	   *pathname;
	const char	   *ts_column;
	const char	   *tag_column;
	bool			parquet_mode;
	arrow::Compression::type default_compression;
	std::vector<std::shared_ptr<class arrowFileWriteField>> arrow_fields;
};

/* Arrow/Parquet compression options */
static struct {
	const char	   *name;
	arrow::Compression::type	code;
}	compression_options[] = {
	{"none",	arrow::Compression::type::UNCOMPRESSED},
	{"snappy",	arrow::Compression::type::SNAPPY},
	{"gzip",	arrow::Compression::type::GZIP},
	{"brotli",	arrow::Compression::type::BROTLI},
	{"zstd",	arrow::Compression::type::ZSTD},
	{"lz4",		arrow::Compression::type::LZ4},
	{"lzo",		arrow::Compression::type::LZO},
	{"bz2",		arrow::Compression::type::BZ2},
	{NULL,		arrow::Compression::type::UNCOMPRESSED}
};

class arrowFileWriteField {
public:
	std::string		attname;
	std::string		typname;
	VALUE			fetch_key;
	bool			stats_enabled;
	bool			ts_column;
	bool			tag_column;
	arrow::Compression::type compression;	/* valid only if parquet-mode */
	arrowBuilder	arrow_builder;
	arrowArray		arrow_array;
	arrowFileWriteField(const char *__attname,
						const char *__typname,
						bool __stats_enabled,
						arrow::Compression::type __compression)
	{
		attname = std::string(__attname);
		typname = std::string(__typname);
		fetch_key = rb_str_new_cstr(__attname);
		ts_column = false;
		tag_column = false;
		stats_enabled = __stats_enabled;
		compression = __compression;
	}
	virtual void	putValue(VALUE datum) = 0;
	void	Reset(void)
	{
		arrow_builder->Reset();
		if (arrow_array)
			arrow_array = nullptr;
	}
	void	__errorReportOnPutValue(arrow::Status rv, VALUE datum)
	{
		const char *str;
		size_t		len;

		if (datum == Qnil)
		{
			str = "NULL";
			len = 4;
		}
		else
		{
			VALUE	sval = rb_String(datum);
			str = RSTRING_PTR(sval);
			len = RSTRING_LEN(sval);
		}
		Elog("unable to put '%.*s' to '%s' field (%s): %s",
			 len, str,
			 attname.c_str(),
			 typname.c_str(),
			 rv.ToString().c_str());
	}
};

#define ARROW_FILE_WRITE_FIELD_TEMPLATE(NAME,BUILDER_TYPE,ARROW_TYPE,C_TYPE) \
	class arrowFileWrite##NAME##Field : public arrowFileWriteField		\
	{																	\
	public:																\
		arrowFileWrite##NAME##Field(const char *__attname,				\
									bool __stats_enabled,				\
									arrow::Compression::type __compression)	\
		: arrowFileWriteField(__attname, #NAME, __stats_enabled, __compression)	\
		{																\
			arrow_builder = std::make_shared<arrow::BUILDER_TYPE>(arrow::ARROW_TYPE, \
																  arrow::default_memory_pool()); \
		}																\
		C_TYPE	fetchValue(VALUE datum);								\
		void	putValue(VALUE datum)									\
		{																\
			auto	builder = std::dynamic_pointer_cast<arrow::BUILDER_TYPE>(this->arrow_builder); \
			arrow::Status rv;											\
																		\
			if (datum == Qnil)											\
				rv = builder->AppendNull();								\
			else														\
				rv = builder->Append(fetchValue(datum));				\
			if (!rv.ok())												\
				__errorReportOnPutValue(rv, datum);						\
		}																\
	}

ARROW_FILE_WRITE_FIELD_TEMPLATE(Boolean,   BooleanBuilder,   boolean(), bool);
ARROW_FILE_WRITE_FIELD_TEMPLATE(Int8,      Int8Builder,      int8(),    int8_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(Int16,     Int16Builder,     int16(),   int16_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(Int32,     Int32Builder,     int32(),   int32_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(Int64,     Int64Builder,     int64(),   int64_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(UInt8,     UInt8Builder,     uint8(),   uint8_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(UInt16,    UInt16Builder,    uint16(),  uint16_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(UInt32,    UInt32Builder,    uint32(),  uint32_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(UInt64,    UInt64Builder,    uint64(),  uint64_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(HalfFloat, HalfFloatBuilder, float16(), half_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(Float,     FloatBuilder,     float32(), float);
ARROW_FILE_WRITE_FIELD_TEMPLATE(Double,	   DoubleBuilder,    float64(), double);
ARROW_FILE_WRITE_FIELD_TEMPLATE(Date_sec, Date32Builder, date32(), int32_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(Date_ms,  Date64Builder, date64(), int64_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(Time_sec, Time32Builder, time32(arrow::TimeUnit::SECOND), int32_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(Time_ms,  Time32Builder, time32(arrow::TimeUnit::MILLI),  int32_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(Time_us,  Time64Builder, time64(arrow::TimeUnit::MICRO),  int64_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(Time_ns,  Time64Builder, time64(arrow::TimeUnit::NANO),   int64_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(Timestamp_sec, TimestampBuilder, timestamp(arrow::TimeUnit::SECOND), int64_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(Timestamp_ms,  TimestampBuilder, timestamp(arrow::TimeUnit::MILLI),  int64_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(Timestamp_us,  TimestampBuilder, timestamp(arrow::TimeUnit::MICRO),  int64_t);
ARROW_FILE_WRITE_FIELD_TEMPLATE(Timestamp_ns,  TimestampBuilder, timestamp(arrow::TimeUnit::NANO),   int64_t);

bool
arrowFileWriteBooleanField::fetchValue(VALUE datum)
{
	if (datum == Qtrue)
		return true;
	if (datum == Qfalse)
		return false;
	if (CLASS_OF(datum) == rb_cString)
	{
		const char *ptr = RSTRING_PTR(datum);
		size_t		len = RSTRING_LEN(datum);

		if (len == 4 && (memcmp(ptr, "true", 4) == 0 ||
						 memcmp(ptr, "True", 4) == 0 ||
						 memcmp(ptr, "TRUE", 4) == 0))
			return true;
		if (len == 1 && (memcmp(ptr, "t", 1) == 0 ||
						 memcmp(ptr, "T", 1) == 0))
			return true;
		if (len == 5 && (memcmp(ptr, "false", 5) == 0 ||
						 memcmp(ptr, "False", 5) == 0 ||
						 memcmp(ptr, "FALSE", 5) == 0))
			return false;
		if (len == 1 && (memcmp(ptr, "f", 1) == 0 ||
						 memcmp(ptr, "F", 1) == 0))
			return false;
		/* elsewhere, try to convert to Integer */
		datum = rb_funcall(datum, rb_intern("to_i"), 0);
	}
	return (NUM2INT(datum) != 0 ? true : false);
}

static inline VALUE
__fetchIntValue(VALUE datum)
{
	assert(datum != Qnil);
	if (CLASS_OF(datum) == rb_cInteger)
		return datum;
	if (!rb_respond_to(datum, rb_intern("to_i")))
		datum = rb_funcall(datum, rb_intern("to_s"), 0);
	return rb_funcall(datum, rb_intern("to_i"), 0);
}

int8_t
arrowFileWriteInt8Field::fetchValue(VALUE datum)
{
	int		ival = NUM2INT(__fetchIntValue(datum));
	if (ival < SCHAR_MIN || ival > SCHAR_MAX)
		Elog("value %d is out of range for '%s' (%s)",
			 ival, attname.c_str(), typname.c_str());
	return (int8_t)ival;
}

int16_t
arrowFileWriteInt16Field::fetchValue(VALUE datum)
{
	return NUM2SHORT(__fetchIntValue(datum));
}

int32_t
arrowFileWriteInt32Field::fetchValue(VALUE datum)
{
	return NUM2INT(__fetchIntValue(datum));
}

int64_t
arrowFileWriteInt64Field::fetchValue(VALUE datum)
{
	return NUM2LONG(__fetchIntValue(datum));
}

uint8_t
arrowFileWriteUInt8Field::fetchValue(VALUE datum)
{
	int		ival = NUM2INT(__fetchIntValue(datum));
	if (ival < 0 || ival > UCHAR_MAX)
		Elog("value %d is out of range for '%s' (%s)",
			 ival, attname.c_str(), typname.c_str());
	return (uint8_t)ival;
}

uint16_t
arrowFileWriteUInt16Field::fetchValue(VALUE datum)
{
	return NUM2USHORT(__fetchIntValue(datum));
}

uint32_t
arrowFileWriteUInt32Field::fetchValue(VALUE datum)
{
	return NUM2UINT(__fetchIntValue(datum));
}

uint64_t
arrowFileWriteUInt64Field::fetchValue(VALUE datum)
{
	return NUM2ULONG(__fetchIntValue(datum));
}

static inline double
__fetchFloatValue(VALUE datum)
{
	assert(datum != Qnil);
	if (CLASS_OF(datum) != rb_cFloat)
	{
		if (!rb_respond_to(datum, rb_intern("to_f")))
			datum = rb_funcall(datum, rb_intern("to_s"), 0);
		datum = rb_funcall(datum, rb_intern("to_f"), 0);
	}
	return NUM2DBL(datum);
}

half_t
arrowFileWriteHalfFloatField::fetchValue(VALUE datum)
{
	return fp64_to_fp16(__fetchFloatValue(datum));
}

float
arrowFileWriteFloatField::fetchValue(VALUE datum)
{
	return (float)__fetchFloatValue(datum);
}

double
arrowFileWriteDoubleField::fetchValue(VALUE datum)
{
	return __fetchFloatValue(datum);
}

#define SECS_PER_DAY		86400

static inline bool
__tryFetchEventTime(VALUE datum,
					uint64_t *p_sec,	/* seconds from UTC */
					uint64_t *p_nsec)	/* nano-seconds in the day */
{
	const char *cname;

	assert(datum != Qnil);
	/* Is it Fluent::EventTime? */
	cname = rb_class2name(CLASS_OF(datum));
	if (strcmp(cname, "EventTime") == 0 &&
		rb_respond_to(datum, rb_intern("sec")) &&
		rb_respond_to(datum, rb_intern("nsec")))
	{
		VALUE	sec = rb_funcall(datum, rb_intern("sec"), 0);
		VALUE	nsec = rb_funcall(datum, rb_intern("nsec"), 0);
		*p_sec = NUM2ULONG(sec);
		*p_nsec = NUM2ULONG(nsec);
		return true;
	}
	return false;
}

template <typename B_TYPE, typename A_TYPE, typename C_TYPE>
static inline C_TYPE
__fetchRubyDateTimeValue(arrowBuilder arrow_builder,
						 VALUE datum,
						 std::string &attname,
						 std::string &typname)
{
	auto	builder = std::dynamic_pointer_cast<B_TYPE>(arrow_builder);
	auto	ts_type = std::dynamic_pointer_cast<A_TYPE>(builder->type());
	VALUE	sval = rb_String(datum);
	C_TYPE	tval;

	if (!arrow::internal::ParseValue<A_TYPE>(*ts_type,
											 RSTRING_PTR(sval),
											 RSTRING_LEN(sval),
											 &tval))
		Elog("unable to convert token '%*s' for '%s' field (%s)",
			 RSTRING_LEN(sval),
			 RSTRING_PTR(sval),
			 attname.c_str(),
			 typname.c_str());
	return tval;
}
#define FETCH_RUBY_DATE_TIME_VALUE(NAME,C_TYPE)							\
	__fetchRubyDateTimeValue<arrow::NAME##Builder,						\
							 arrow::NAME##Type,							\
							 C_TYPE>(arrow_builder, datum, attname, typname)
int32_t
arrowFileWriteDate_secField::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Date32, int32_t);
	return (int32_t)(sec / SECS_PER_DAY);
}

int64_t
arrowFileWriteDate_msField::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Date64, int64_t);
	return (sec * 1000L) + (nsec / 1000000L);
}

int32_t
arrowFileWriteTime_secField::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Time32, int32_t);
	return (int32_t)(sec % SECS_PER_DAY);
}

int32_t
arrowFileWriteTime_msField::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Time32, int32_t);
	return (sec % SECS_PER_DAY) * 1000 + (nsec / 1000000);
}

int64_t
arrowFileWriteTime_usField::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Time64, int64_t);
	return (sec % SECS_PER_DAY) * 1000000L + (nsec / 1000);
}

int64_t
arrowFileWriteTime_nsField::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Time64, int64_t);
	return (sec % SECS_PER_DAY) * 1000000000L + nsec;
}

int64_t
arrowFileWriteTimestamp_secField::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Timestamp, int64_t);
	return sec;
}

int64_t
arrowFileWriteTimestamp_msField::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Timestamp, int64_t);
	return (sec * 1000) + (nsec / 1000000);
}

int64_t
arrowFileWriteTimestamp_usField::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Timestamp, int64_t);
	return (sec * 1000000) + (nsec / 1000);
}

int64_t
arrowFileWriteTimestamp_nsField::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Timestamp, int64_t);
	return (sec * 1000000000L) + nsec;
}

/*
 * Decimal128 type handler
 */
class arrowFileWriteDecimal128Field : public arrowFileWriteField
{
	int		dprecision;
	int		dscale;
public:
	arrowFileWriteDecimal128Field(const char *__attname,
								  bool __stats_enabled,
								  arrow::Compression::type __compression)
		: arrowFileWriteField(__attname, "Decimal128", __stats_enabled, __compression)
	{
		const char *extra = strchr(__attname, '(');

		dprecision = 38;	/* default */
		dscale = 6;			/* default */
		if (extra)
		{
			int		__precision;
			int		__scale;
			char   *__dummy;

			if (sscanf(extra, "(%u,%u)%s",
					   &__precision,
					   &__scale,
					   &__dummy) == 2)
			{
				dprecision = __precision;
				dscale = __scale;
			}
			else if (sscanf(extra, "(%u)%s",
							&__scale,
							&__dummy) == 1)
			{
				dscale = __scale;
			}
			else
				Elog("invalid Decimal128 precision and scale '%s'", extra);
		}
		arrow_builder = std::make_shared<arrow::Decimal128Builder>(arrow::decimal128(dprecision, dscale),
																   arrow::default_memory_pool());
	}
	arrow::Decimal128	fetchValue(VALUE datum)
	{
		assert(datum != Qnil);
	retry:
		if (CLASS_OF(datum) == rb_cInteger ||
			CLASS_OF(datum) == rb_cFloat ||
			CLASS_OF(datum) == rb_cRational)
		{
			arrow::Decimal128 result;
			VALUE		ival;

			if (dscale > 0)
			{
				ival = rb_funcall(INT2NUM(10),
								  rb_intern("**"),
								  1, INT2NUM(dscale));
				datum = rb_funcall(datum, rb_intern("*"), 1, ival);
			}
			else if (dscale < 0)
			{
				ival = rb_funcall(INT2NUM(10),
								  rb_intern("**"),
								  1, INT2NUM(-dscale));
				datum = rb_funcall(datum, rb_intern("/"), 1, ival);
			}
			/* convert to integer */
			if (CLASS_OF(datum) != rb_cInteger)
				datum = rb_funcall(datum, rb_intern("to_i"), 0);
			/* overflow check */
			ival = rb_funcall(datum, rb_intern("bit_length"), 0);
			if (NUM2INT(ival) > 128)
				Elog("decimal value out of range");
			rb_integer_pack(datum,
							&result, sizeof(arrow::Decimal128), 1, 0,
							INTEGER_PACK_LITTLE_ENDIAN);
			return result;
		}
		else
		{
			if (CLASS_OF(datum) != rb_cString)
				datum = rb_String(datum);
			datum = rb_Rational1(datum);
			goto retry;
		}
		Elog("cannot convert to decimal value");
	}
	void	putValue(VALUE datum)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::Decimal128Builder>(this->arrow_builder);
		arrow::Status rv;

		if (datum == Qnil)
			rv = builder->AppendNull();
		else
			rv = builder->Append(fetchValue(datum));
		if (!rv.ok())
			__errorReportOnPutValue(rv, datum);
	}
};

class arrowFileWriteUtf8Field : public arrowFileWriteField
{
public:
	arrowFileWriteUtf8Field(const char *__attname,
							bool __stats_enabled,
							arrow::Compression::type __compression)
		: arrowFileWriteField(__attname, "Utf8", __stats_enabled, __compression)
	{
		arrow_builder = std::make_shared<arrow::StringBuilder>(arrow::utf8(),
															   arrow::default_memory_pool());
	}
	void	putValue(VALUE datum)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::StringBuilder>(this->arrow_builder);
		arrow::Status rv;

		if (datum == Qnil)
			rv = builder->AppendNull();
		else
		{
			VALUE	sval = rb_String(datum);

			rv = builder->Append(RSTRING_PTR(sval),
								 RSTRING_LEN(sval));
		}
		if (!rv.ok())
			__errorReportOnPutValue(rv, datum);
	}
};

/* ================================================================
 *
 * IPAddr4 / IPAddr6
 *
 * ================================================================ */
static VALUE	ipaddr_klass = Qnil;

class arrowFileWriteIpAddr4Field : public arrowFileWriteField
{
public:
	arrowFileWriteIpAddr4Field(const char *__attname,
							   bool __stats_enabled,
							   arrow::Compression::type __compression)
		: arrowFileWriteField(__attname, "IpAddr4", __stats_enabled, __compression)
	{
		arrow_builder = std::make_shared<arrow::FixedSizeBinaryBuilder>(arrow::fixed_size_binary(4),
																		arrow::default_memory_pool());
	}
	void	putValue(VALUE datum)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::FixedSizeBinaryBuilder>(this->arrow_builder);
		arrow::Status rv;

		if (datum == Qnil)
			rv = builder->AppendNull();
		else
		{
			uint8_t	buf[4];
			int		nloops = 0;
		retry:
			if (rb_respond_to(datum, rb_intern("ipv4?")) &&
				rb_funcall(datum, rb_intern("ipv4?"), 0) == Qtrue)
			{
				VALUE	ival = rb_funcall(datum, rb_intern("to_i"), 0);

				rb_integer_pack(ival, buf, sizeof(buf), 1, 0,
								INTEGER_PACK_BIG_ENDIAN);
			}
			else if (nloops++ == 0)
			{
				if (ipaddr_klass == Qnil)
					ipaddr_klass = rb_path2class("IPAddr");
				datum = rb_String(datum);
				datum = rb_class_new_instance(1, &datum, ipaddr_klass);
				goto retry;
			}
			else
			{
				Elog("unable to convert datum to logical arrow::IpAddr4");
			}
			rv = builder->Append(buf);
		}
		if (!rv.ok())
			__errorReportOnPutValue(rv, datum);
	}
};

class arrowFileWriteIpAddr6Field : public arrowFileWriteField
{
public:
	arrowFileWriteIpAddr6Field(const char *__attname,
							   bool __stats_enabled,
							   arrow::Compression::type __compression)
		: arrowFileWriteField(__attname, "IpAddr6", __stats_enabled, __compression)
	{
		arrow_builder = std::make_shared<arrow::FixedSizeBinaryBuilder>(arrow::fixed_size_binary(16),
																		arrow::default_memory_pool());
	}
	void	putValue(VALUE datum)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::FixedSizeBinaryBuilder>(this->arrow_builder);
		arrow::Status rv;

		if (datum == Qnil)
			rv = builder->AppendNull();
		else
		{
			uint8_t	buf[16];
			int		nloops = 0;
		retry:
			if (rb_respond_to(datum, rb_intern("ipv6?")) &&
				rb_funcall(datum, rb_intern("ipv6?"), 0) == Qtrue)
			{
				VALUE	ival = rb_funcall(datum, rb_intern("to_i"), 0);

				rb_integer_pack(ival, buf, sizeof(buf), 1, 0,
								INTEGER_PACK_BIG_ENDIAN);
			}
			else if (nloops++ == 0)
			{
				/* Load 'ipaddr' module once */
				if (ipaddr_klass == Qnil)
					ipaddr_klass = rb_path2class("IPAddr");
				datum = rb_String(datum);
				datum = rb_class_new_instance(1, &datum, ipaddr_klass);
				goto retry;
			}
			else
			{
				Elog("unable to convert datum to logical arrow::IpAddr4");
			}
			rv = builder->Append(buf);
		}
		if (!rv.ok())
			__errorReportOnPutValue(rv, datum);
	}
};

// ----------------------------------------------------------------
//
// Interface for Ruby ABI
//
// ----------------------------------------------------------------
extern "C" {
	static VALUE	arrowFileWrite__initialize(VALUE self,
											   VALUE __pathname,
											   VALUE __schema_defs,
											   VALUE __params);
	static VALUE	arrowFileWrite__writeChunk(VALUE self,
											   VALUE __chunk);
	static void		arrowFileWriteType__dmark(void *data);
	static void		arrowFileWriteType__free(void *data);
	static size_t	arrowFileWriteType__size(const void *data);
	void			Init_arrow_file_write(void);
};

static const rb_data_type_t	arrowFileWriteType = {
	"ArrowFileWrite",
	{
		arrowFileWriteType__dmark,	/* dmark */
		arrowFileWriteType__free,	/* dfree */
		arrowFileWriteType__size,	/* dsize */
		NULL, NULL,		/* reserved */
	},
	NULL,	/* parent */
	NULL,	/* data */
	0,		/* flags */
};

static void
__arrowFileWritePathnameValidator(arrowFileWrite *fw_state, VALUE __pathname)
{
	const char *str;
	uint32_t	len;
	VALUE		pathname;

	pathname = rb_funcall(__pathname, rb_intern("to_s"), 0);
	str = RSTRING_PTR(pathname);
	len = RSTRING_LEN(pathname);
	if (len == 0)
		Elog("pathname must not be empty");
	for (int i=0; i < len; i++)
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
				Elog("unknown format character: '%%%c' in '%.*s'",
					 str[i], len, str);
		}
	}
	fw_state->pathname = pstrdup(str, len);
}

static void
__arrowFileWriteParseParamDefs(arrowFileWrite *fw_state, VALUE __params)
{
	VALUE		datum;

	if (CLASS_OF(__params) == rb_cHash)
	{
		datum = rb_funcall(__params, rb_intern("fetch"), 2,
						   rb_str_new_cstr("ts_column"), Qnil);
		if (datum != Qnil)
		{
			VALUE	ts_column = rb_funcall(datum, rb_intern("to_s"), 0);

			fw_state->ts_column = pstrdup(RSTRING_PTR(ts_column),
										  RSTRING_LEN(ts_column));
		}
		datum = rb_funcall(__params, rb_intern("fetch"), 2,
						   rb_str_new_cstr("tag_column"), Qnil);
		if (datum != Qnil)
		{
			VALUE	tag_column = rb_funcall(datum, rb_intern("to_s"), 0);
			fw_state->tag_column = pstrdup(RSTRING_PTR(tag_column),
										   RSTRING_LEN(tag_column));
		}
		datum = rb_funcall(__params, rb_intern("fetch"), 2,
						   rb_str_new_cstr("format"), Qnil);
		if (datum != Qnil)
		{
			VALUE		format = rb_funcall(datum, rb_intern("to_s"), 0);
			const char *str = RSTRING_PTR(format);
			uint32_t	len = RSTRING_LEN(format);

			if (len == 5 && strncmp("arrow", str, 5) == 0)
				fw_state->parquet_mode = false;
			else if (len == 7 && strncmp("parquet", str, 7) == 0)
				fw_state->parquet_mode = true;
			else
				Elog("unsupported format option [%.*s]", len, str);
		}
		datum = rb_funcall(__params, rb_intern("fetch"), 2,
						   rb_str_new_cstr("compression"), Qnil);
		if (datum != Qnil)
		{
			VALUE		comp = rb_funcall(datum, rb_intern("to_s"), 0);
			const char *str = RSTRING_PTR(comp);
			uint32_t	len = RSTRING_LEN(comp);

			for (int i=0; compression_options[i].name != NULL; i++)
			{
				const char *name = compression_options[i].name;

				if (strncmp(name, str, len) == 0 && name[len] == '\0')
				{
					fw_state->default_compression = compression_options[i].code;
					goto found;
				}
			}
			Elog("unsupported compression option [%.*s]", len, str);
		found:
			;
		}
	}
	else if (__params != Qnil)
		Elog("ArrowFileWrite: parameters must be Hash");
}

static void
__arrowFileWriteParseSchemaDefs(arrowFileWrite *fw_state, VALUE __schema_defs)
{
	VALUE		schema_defs;
	char	   *buf;
	int			len;
	char	   *tok, *saveptr;

	schema_defs = rb_funcall(__schema_defs, rb_intern("to_s"), 0);
	len = RSTRING_LEN(schema_defs);
	buf = (char *)alloca(len+1);
	memcpy(buf, RSTRING_PTR(schema_defs), len);
	buf[len] = '\0';

	for (tok = strtok_r(buf, ",", &saveptr);
		 tok != NULL;
		 tok = strtok_r(NULL, ",", &saveptr))
	{
		/* <column_name>=<column_type>[;<column_attr>;...] */
		std::shared_ptr<class arrowFileWriteField> arrow_field;
		char   *field_name = tok;
		char   *field_type;
		char   *field_extra;
		int		stat_enabled = -1;
		arrow::Compression::type compression = arrow::Compression::type::UNCOMPRESSED;

		field_type = strchr(field_name, '=');
		if (!field_type)
			Elog("syntax error in schema definition [%s]", field_name);
		*field_type++ = '\0';

		field_extra = strchr(field_type, ';');
		if (field_extra)
		{
			char   *__tok, *__saveptr;

			*field_extra++ = '\0';
			for (__tok = strtok_r(field_extra, ";", &__saveptr);
				 __tok != NULL;
				 __tok = strtok_r(NULL, ";", &__saveptr))
			{
				__tok = __trim(__tok);
				if (strcmp(__tok, "stat_enabled") == 0)
					stat_enabled = 1;
				else if (strcmp(__tok, "stat_disabled") == 0)
					stat_enabled = 0;
				else
				{
					for (int i=0; compression_options[i].name != NULL; i++)
					{
						const char *name = compression_options[i].name;

						if (strcmp(__tok, name) == 0)
						{
							compression = compression_options[i].code;
							goto found;
						}
					}
					Elog("unknown field attribute [%s]", __tok);
				found:
					;
				}
			}
		}
		field_name = __trim(field_name);
		field_type = __trim(field_type);

		if (strcasecmp(field_name, "bool") == 0)
			arrow_field = std::make_shared<arrowFileWriteBooleanField>(field_name, stat_enabled, compression);
		else if (strcasecmp(field_name, "int8") == 0)
			arrow_field = std::make_shared<arrowFileWriteInt8Field>(field_name, stat_enabled, compression);
		else if (strcasecmp(field_name, "int16") == 0)
			arrow_field = std::make_shared<arrowFileWriteInt16Field>(field_name, stat_enabled, compression);
		else if (strcasecmp(field_name, "int32") == 0)
			arrow_field = std::make_shared<arrowFileWriteInt32Field>(field_name, stat_enabled, compression);
		else if (strcasecmp(field_name, "int64") == 0)
			arrow_field = std::make_shared<arrowFileWriteInt64Field>(field_name, stat_enabled, compression);
		else if (strcasecmp(field_name, "uint8") == 0)
			arrow_field = std::make_shared<arrowFileWriteUInt8Field>(field_name, stat_enabled, compression);
		else if (strcasecmp(field_name, "uint16") == 0)
			arrow_field = std::make_shared<arrowFileWriteUInt16Field>(field_name, stat_enabled, compression);
		else if (strcasecmp(field_name, "uint32") == 0)
			arrow_field = std::make_shared<arrowFileWriteUInt32Field>(field_name, stat_enabled, compression);
		else if (strcasecmp(field_name, "uint64") == 0)
			arrow_field = std::make_shared<arrowFileWriteUInt64Field>(field_name, stat_enabled, compression);
		else if (strcasecmp(field_name, "float16") == 0)
			arrow_field = std::make_shared<arrowFileWriteHalfFloatField>(field_name, stat_enabled, compression);
		else if (strcasecmp(field_name, "float32") == 0)
			arrow_field = std::make_shared<arrowFileWriteFloatField>(field_name, stat_enabled, compression);
		else if (strcasecmp(field_name, "float64") == 0)
			arrow_field = std::make_shared<arrowFileWriteDoubleField>(field_name, stat_enabled, compression);
		else if (strcasecmp(field_name, "decimal") == 0 ||
				 strcasecmp(field_name, "decimal128") == 0 ||
				 strncasecmp(field_name, "decimal(", 8) == 0 ||		/* with precision, scale */
				 strncasecmp(field_name, "decimal128(", 11) == 0)	/* with precision, scale */
			arrow_field = std::make_shared<arrowFileWriteDecimal128Field>(field_name, stat_enabled, compression);
		else if (strcasecmp(field_name, "text") == 0 ||
				 strcasecmp(field_name, "utf8") == 0)
			arrow_field = std::make_shared<arrowFileWriteUtf8Field>(field_name, stat_enabled, compression);
		else if (strcasecmp(field_name, "ipaddr4") == 0)
			arrow_field = std::make_shared<arrowFileWriteIpAddr4Field>(field_name, stat_enabled, compression);
		else if (strcasecmp(field_name, "ipaddr6") == 0)
			arrow_field = std::make_shared<arrowFileWriteIpAddr6Field>(field_name, stat_enabled, compression);
		else
			Elog("ArrowFileWrite: not a supported type '%s' for '%s'", field_type, field_name);

		fw_state->arrow_fields.push_back(arrow_field);
	}
}







static VALUE
arrowFileWrite__initialize(VALUE self,
						   VALUE __pathname,
						   VALUE __schema_defs,
						   VALUE __params)
{
	arrowFileWrite *fw_state = (arrowFileWrite *)self;
	char   *emsg = NULL;

	rb_require("time");
	rb_require("ipaddr");
	try {
		__arrowFileWritePathnameValidator(fw_state, __pathname);
		__arrowFileWriteParseParamDefs(fw_state, __params);
		__arrowFileWriteParseSchemaDefs(fw_state, __schema_defs);
	}
	catch (const std::exception &e) {
		const char *estr = e.what();
		emsg = (char *)alloca(strlen(estr)+1);
		strcpy(emsg, estr);
	}
	/* transform libarrow/libparquet exception to ruby exception */
	if (emsg)
		Elog("arrow-write: %s", emsg);
	return self;
}

static VALUE
arrowFileWrite__writeChunk(VALUE self,
						   VALUE __chunk)
{
	arrowFileWrite *fw_state = (arrowFileWrite *)self;
	char   *emsg = NULL;
	VALUE	retval = 0;

	//move to buffer


	
	//add try{}catch{};

	return retval;
}

static VALUE
arrowFileWriteType__alloc(VALUE klass)
{
	arrowFileWrite *fw_state = new arrowFileWrite;
	return TypedData_Wrap_Struct(klass, &arrowFileWriteType, fw_state);
}

static void
arrowFileWriteType__dmark(void *data)
{
	arrowFileWrite *fw_state = (arrowFileWrite *)data;

	for (int j=0; j < fw_state->arrow_fields.size(); j++)
	{
		auto	field = fw_state->arrow_fields[j];

		rb_gc_mark_maybe(field->fetch_key);
	}
}

static void
arrowFileWriteType__free(void *data)
{
	arrowFileWrite *fw_state = (arrowFileWrite *)data;

	delete(fw_state);
}

static size_t
arrowFileWriteType__size(const void *data)
{
	//FIXME: it reports GC internal memory consumption
	return sizeof(arrowFileWrite);
}

void
Init_arrow_file_write(void)
{
	VALUE	klass;

	klass = rb_define_class("ArrowFileWrite",  rb_cObject);
	rb_define_alloc_func(klass, arrowFileWriteType__alloc);
	rb_define_method(klass, "initialize", arrowFileWrite__initialize, 3);
	rb_define_method(klass, "writeChunk", arrowFileWrite__writeChunk, 1);
}
