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
#include <parquet/arrow/writer.h>
#include <ruby.h>
#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <string>
#include <syslog.h>
#include "float2.h"

using	arrowSchema		= std::shared_ptr<arrow::Schema>;
using	arrowField		= std::shared_ptr<arrow::Field>;
using	arrowBuilder	= std::shared_ptr<arrow::ArrayBuilder>;
using	arrowArray		= std::shared_ptr<arrow::Array>;

#define Min(a,b)		((a) < (b) ? (a) : (b))
#define Max(a,b)		((a) > (b) ? (a) : (b))
#define Elog(fmt,...)							\
	rb_raise(rb_eException, "%s:%d " fmt,		\
			 __FILE__, __LINE__, ##__VA_ARGS__)
#define Info(fmt,...)								\
	syslog(LOG_USER, "[fluentd-arrow %s:%d] " fmt,	\
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
	std::shared_ptr<arrow::io::FileOutputStream> OpenFile(std::string &filename);
public:
	arrowFileWrite()
	{
		pathname = nullptr;
		ts_column = nullptr;
		tag_column = nullptr;
		filename_seqno = 0;
		chunk_nitems = 0;
		parquet_mode = false;
		default_compression = arrow::Compression::type::ZSTD;
		arrow_schema = nullptr;
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
	uint32_t		filename_seqno;
	uint64_t		chunk_nitems;
	bool			parquet_mode;
	arrow::Compression::type default_compression;
	arrowSchema		arrow_schema;
	std::vector<std::shared_ptr<class arrowFileWriteColumn>> arrow_columns;
	void			ResetBuffers(void);
	std::shared_ptr<arrow::RecordBatch> BuildBuffers(void);
	void			writeArrowFile(void);
	void			writeParquetFile(void);
	std::shared_ptr<parquet::WriterProperties> buildParquetWriterProperties(void);
	std::shared_ptr<arrow::KeyValueMetadata> buildArrowMinMaxStats(void);
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

class arrowFileWriteColumn {
protected:
	void	__ResetBase(void)
	{
		arrow_builder->Reset();
		arrow_array = nullptr;
	}
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
	arrowFileWriteColumn(const char *__attname,
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
	virtual void	appendStats(std::shared_ptr<arrow::KeyValueMetadata> custom_metadata) = 0;
	virtual void	putValue(VALUE datum) = 0;
	virtual void	Reset(void)
	{
		__ResetBase();
	}
	virtual uint64_t	Finish(void)
	{
		auto	rv = arrow_builder->Finish(&arrow_array);
		if (!rv.ok())
			Elog("failed on arrow::ArrayBuilder::Finish: %s",
				 rv.ToString().c_str());
		return arrow_array->length();
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
	class arrowFileWriteColumn##NAME : public arrowFileWriteColumn		\
	{																	\
		bool	stats_is_valid;											\
		C_TYPE	stats_min_value;										\
		C_TYPE	stats_max_value;										\
	public:																\
		arrowFileWriteColumn##NAME(const char *__attname,				\
								   bool __stats_enabled,				\
								   arrow::Compression::type __compression) \
		: arrowFileWriteColumn(__attname, #NAME, __stats_enabled, __compression) \
		{																\
			arrow_builder = std::make_shared<arrow::BUILDER_TYPE>(arrow::ARROW_TYPE, \
																  arrow::default_memory_pool()); \
			stats_is_valid = false;										\
		}																\
		void	appendStats(std::shared_ptr<arrow::KeyValueMetadata> custom_metadata) \
		{																\
			if (!stats_is_valid)										\
				return;													\
			custom_metadata->Append(std::string("min_max_stats.") + attname, \
									std::to_string(stats_min_value) +	\
									std::string(",") +					\
									std::to_string(stats_max_value));	\
		}																\
		C_TYPE	fetchValue(VALUE datum);								\
		void	putValue(VALUE datum)									\
		{																\
			auto	builder = std::dynamic_pointer_cast<arrow::BUILDER_TYPE>(arrow_builder); \
			arrow::Status rv;											\
																		\
			if (datum == Qnil)											\
				rv = builder->AppendNull();								\
			else														\
			{															\
				C_TYPE	value = fetchValue(datum);						\
				if (stats_enabled)										\
				{														\
					if (!stats_is_valid)								\
					{													\
						stats_min_value = stats_max_value = value;		\
						stats_is_valid = true;							\
					}													\
					else if (value < stats_min_value)					\
						stats_min_value = value;						\
					else if (value > stats_max_value)					\
						stats_max_value = value;						\
				}														\
				rv = builder->Append(value);							\
			}															\
			if (!rv.ok())												\
				__errorReportOnPutValue(rv, datum);						\
		}																\
		void	Reset(void)												\
		{																\
			__ResetBase();												\
			stats_is_valid = false;										\
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
arrowFileWriteColumnBoolean::fetchValue(VALUE datum)
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
arrowFileWriteColumnInt8::fetchValue(VALUE datum)
{
	int		ival = NUM2INT(__fetchIntValue(datum));
	if (ival < SCHAR_MIN || ival > SCHAR_MAX)
		Elog("value %d is out of range for '%s' (%s)",
			 ival, attname.c_str(), typname.c_str());
	return (int8_t)ival;
}

int16_t
arrowFileWriteColumnInt16::fetchValue(VALUE datum)
{
	return NUM2SHORT(__fetchIntValue(datum));
}

int32_t
arrowFileWriteColumnInt32::fetchValue(VALUE datum)
{
	return NUM2INT(__fetchIntValue(datum));
}

int64_t
arrowFileWriteColumnInt64::fetchValue(VALUE datum)
{
	return NUM2LONG(__fetchIntValue(datum));
}

uint8_t
arrowFileWriteColumnUInt8::fetchValue(VALUE datum)
{
	int		ival = NUM2INT(__fetchIntValue(datum));
	if (ival < 0 || ival > UCHAR_MAX)
		Elog("value %d is out of range for '%s' (%s)",
			 ival, attname.c_str(), typname.c_str());
	return (uint8_t)ival;
}

uint16_t
arrowFileWriteColumnUInt16::fetchValue(VALUE datum)
{
	return NUM2USHORT(__fetchIntValue(datum));
}

uint32_t
arrowFileWriteColumnUInt32::fetchValue(VALUE datum)
{
	return NUM2UINT(__fetchIntValue(datum));
}

uint64_t
arrowFileWriteColumnUInt64::fetchValue(VALUE datum)
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
arrowFileWriteColumnHalfFloat::fetchValue(VALUE datum)
{
	return fp64_to_fp16(__fetchFloatValue(datum));
}

float
arrowFileWriteColumnFloat::fetchValue(VALUE datum)
{
	return (float)__fetchFloatValue(datum);
}

double
arrowFileWriteColumnDouble::fetchValue(VALUE datum)
{
	return __fetchFloatValue(datum);
}

#define SECS_PER_DAY		86400
#include <syslog.h>

static inline bool
__tryFetchEventTime(VALUE datum,
					uint64_t *p_sec,	/* seconds from UTC */
					uint64_t *p_nsec)	/* nano-seconds in the day */
{
	const char *cname;

	assert(datum != Qnil);
	/* Is it Fluent::EventTime? */
	cname = rb_class2name(CLASS_OF(datum));
	syslog(4, "EventTime = [%s]", cname);
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
	/* Is it Integer (elapsed seconds from Epoch) */
    if (CLASS_OF(datum) == rb_cInteger)
	{
		*p_sec = NUM2ULONG(datum);
		*p_nsec = 0;
		return true;
	}
	/* Is convertible to UTC? */
	if (rb_respond_to(datum, rb_intern("getutc")))
		datum = rb_funcall(datum, rb_intern("getutc"), 0);
	/* Is it Time? */
	if (rb_respond_to(datum, rb_intern("tv_sec")) &&
		rb_respond_to(datum, rb_intern("tv_nsec")))
	{
		VALUE	sec = rb_funcall(datum, rb_intern("tv_sec"), 0);
		VALUE	nsec = rb_funcall(datum, rb_intern("tv_nsec"), 0);

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
arrowFileWriteColumnDate_sec::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Date32, int32_t);
	return (int32_t)(sec / SECS_PER_DAY);
}

int64_t
arrowFileWriteColumnDate_ms::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Date64, int64_t);
	return (sec * 1000L) + (nsec / 1000000L);
}

int32_t
arrowFileWriteColumnTime_sec::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Time32, int32_t);
	return (int32_t)(sec % SECS_PER_DAY);
}

int32_t
arrowFileWriteColumnTime_ms::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Time32, int32_t);
	return (sec % SECS_PER_DAY) * 1000 + (nsec / 1000000);
}

int64_t
arrowFileWriteColumnTime_us::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Time64, int64_t);
	return (sec % SECS_PER_DAY) * 1000000L + (nsec / 1000);
}

int64_t
arrowFileWriteColumnTime_ns::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Time64, int64_t);
	return (sec % SECS_PER_DAY) * 1000000000L + nsec;
}

int64_t
arrowFileWriteColumnTimestamp_sec::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Timestamp, int64_t);
	return sec;
}

int64_t
arrowFileWriteColumnTimestamp_ms::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Timestamp, int64_t);
	return (sec * 1000) + (nsec / 1000000);
}

int64_t
arrowFileWriteColumnTimestamp_us::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Timestamp, int64_t);
	return (sec * 1000000) + (nsec / 1000);
}

int64_t
arrowFileWriteColumnTimestamp_ns::fetchValue(VALUE datum)
{
	uint64_t	sec, nsec;

	if (!__tryFetchEventTime(datum, &sec, &nsec))
		return FETCH_RUBY_DATE_TIME_VALUE(Timestamp, int64_t);
	return (sec * 1000000000L) + nsec;
}

/*
 * Decimal128 type handler
 */
class arrowFileWriteColumnDecimal128 : public arrowFileWriteColumn
{
	int		dprecision;
	int		dscale;
	bool	stats_is_valid;
	arrow::Decimal128 stats_min_value;
	arrow::Decimal128 stats_max_value;
public:
	arrowFileWriteColumnDecimal128(const char *__atttype,
								   const char *__attname,
								   bool __stats_enabled,
								   arrow::Compression::type __compression)
		: arrowFileWriteColumn(__attname, "Decimal128", __stats_enabled, __compression)
	{
		const char *extra = strchr(__atttype, '(');

		dprecision = 38;	/* default */
		dscale = 6;			/* default */
		if (extra)
		{
			long	ival;
			char   *end;

			ival = strtol(extra+1, &end, 10);
			while (isspace(*end))
				end++;
			if (*end == ',')
			{
				dscale = ival;
				ival = strtol(end+1, &end, 10);
				while (isspace(*end))
					end++;
				if (*end != ')')
					Elog("invalid Decimal128 precision and scale '%s'", __atttype);
				dprecision = ival;
			}
			else if (*end == ')')
			{
				dscale = ival;
			}
			else
				Elog("invalid Decimal128 precision and scale '%s'", __atttype);
			while (isspace(*end))
				end++;
			if (*end != '\0')
				Elog("invalid Decimal128 precision and scale '%s'", __atttype);
		}
		arrow_builder = std::make_shared<arrow::Decimal128Builder>(arrow::decimal128(dprecision, dscale),
																   arrow::default_memory_pool());
		stats_is_valid = false;
	}
	void	appendStats(std::shared_ptr<arrow::KeyValueMetadata> custom_metadata)
	{
		if (stats_is_valid)
		{
			auto	builder = std::dynamic_pointer_cast<arrow::Decimal128Builder>(arrow_builder);
			auto	d_type = std::static_pointer_cast<arrow::Decimal128Type>(builder->type());
			int		scale = d_type->scale();

			custom_metadata->Append(std::string("min_max_stats.") + attname,
									stats_min_value.ToString(scale) +
									std::string(",") +
									stats_max_value.ToString(scale));
		}
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
		auto	builder = std::dynamic_pointer_cast<arrow::Decimal128Builder>(arrow_builder);
		arrow::Status rv;

		if (datum == Qnil)
			rv = builder->AppendNull();
		else
		{
			auto	value = fetchValue(datum);
			if (stats_enabled)
			{
				if (!stats_is_valid)
				{
					stats_min_value = stats_max_value = value;
					stats_is_valid = true;
				}
				else if (value < stats_min_value)
					stats_min_value = value;
				else if (value > stats_max_value)
					stats_max_value = value;
			}
			rv = builder->Append(value);
		}
		if (!rv.ok())
			__errorReportOnPutValue(rv, datum);
	}
	void	Reset(void)
	{
		__ResetBase();
		stats_is_valid = false;
	}
};

class arrowFileWriteColumnUtf8 : public arrowFileWriteColumn
{
	static constexpr size_t STATS_VALUE_LEN = 60;
	bool	stats_is_valid;
	char	stats_min_value[STATS_VALUE_LEN+1];
    char	stats_max_value[STATS_VALUE_LEN+1];
    int		stats_min_len;
    int		stats_max_len;
	void	updateStats(const char *addr, int sz)
	{
		int		__len, status;

		if (!stats_is_valid)
		{
			__len = Min(sz, STATS_VALUE_LEN);
			memcpy(stats_min_value, addr, __len);
			memcpy(stats_max_value, addr, __len);
			stats_min_value[__len] = '\0';
			stats_max_value[__len] = '\0';
			stats_min_len = __len;
			stats_max_len = __len;
			stats_is_valid = true;
		}
		else
		{
			__len = Min(sz, stats_min_len);
			status = memcmp(addr, stats_min_value, __len);
			if (status < 0 || (status == 0 && stats_min_len > sz))
			{
				__len = Min(sz, STATS_VALUE_LEN);
				memcpy(stats_min_value, addr, __len);
				stats_min_value[__len] = '\0';
			}
			__len = Min(sz, stats_max_len);
			status = memcmp(addr, stats_max_value, __len);
			if (status > 0 || (status == 0 && stats_max_len < sz))
			{
				__len = Min(sz, STATS_VALUE_LEN);
				memcpy(stats_max_value, addr, __len);
				stats_max_value[__len] = '\0';
			}
		}
	}
public:
	arrowFileWriteColumnUtf8(const char *__attname,
							 bool __stats_enabled,
							 arrow::Compression::type __compression)
		: arrowFileWriteColumn(__attname, "Utf8", __stats_enabled, __compression)
	{
		arrow_builder = std::make_shared<arrow::StringBuilder>(arrow::utf8(),
															   arrow::default_memory_pool());
	}
	void	putValue(VALUE datum)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::StringBuilder>(arrow_builder);
		arrow::Status rv;

		if (datum == Qnil)
			rv = builder->AppendNull();
		else
		{
			VALUE	sval = rb_String(datum);

			if (stats_enabled)
				updateStats(RSTRING_PTR(sval),
							RSTRING_LEN(sval));
			rv = builder->Append(RSTRING_PTR(sval),
								 RSTRING_LEN(sval));
		}
		if (!rv.ok())
			__errorReportOnPutValue(rv, datum);
	}
	void	appendStats(std::shared_ptr<arrow::KeyValueMetadata> custom_metadata)
	{
		if (stats_is_valid)
		{
			/* NOTE: we cannot ',' character to use it for delimiter. */
			char   *pos;

			pos = strchr(stats_min_value, ',');
			if (pos)
				*pos = '\0';
			pos = strchr(stats_max_value, ',');
			if (pos)
			{
				pos[0] = (*pos + 1);
				pos[1] = '\0';
			}
			custom_metadata->Append(std::string("min_max_stats.") + attname,
									std::string(stats_min_value) +
									std::string(",") +
									std::string(stats_max_value));
		}
	}
};

/* ================================================================
 *
 * IPAddr4 / IPAddr6
 *
 * ================================================================ */
static VALUE	ipaddr_klass = Qnil;

class arrowFileWriteColumnIpAddr4 : public arrowFileWriteColumn
{
public:
	arrowFileWriteColumnIpAddr4(const char *__attname,
								bool __stats_enabled,
								arrow::Compression::type __compression)
		: arrowFileWriteColumn(__attname, "IpAddr4", __stats_enabled, __compression)
	{
		arrow_builder = std::make_shared<arrow::FixedSizeBinaryBuilder>(arrow::fixed_size_binary(4),
																		arrow::default_memory_pool());
	}
	void	putValue(VALUE datum)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::FixedSizeBinaryBuilder>(arrow_builder);
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
	void	appendStats(std::shared_ptr<arrow::KeyValueMetadata> custom_metadata)
	{
		/* min/max statistics are not supported */
	}
};

class arrowFileWriteColumnIpAddr6 : public arrowFileWriteColumn
{
public:
	arrowFileWriteColumnIpAddr6(const char *__attname,
								bool __stats_enabled,
								arrow::Compression::type __compression)
		: arrowFileWriteColumn(__attname, "IpAddr6", __stats_enabled, __compression)
	{
		arrow_builder = std::make_shared<arrow::FixedSizeBinaryBuilder>(arrow::fixed_size_binary(16),
																		arrow::default_memory_pool());
	}
	void	putValue(VALUE datum)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::FixedSizeBinaryBuilder>(arrow_builder);
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
	void	appendStats(std::shared_ptr<arrow::KeyValueMetadata> custom_metadata)
	{
		/* min/max statistics are not supported */
	}
};

// ----------------------------------------------------------------
//
// arrowFileWrite class methods
//
// ----------------------------------------------------------------
std::shared_ptr<arrow::io::FileOutputStream>
arrowFileWrite::OpenFile(std::string &filename)
{
	int		fdesc = -1;

	for (int retry=0; fdesc < 0; retry++)
	{
		time_t		tval = time(NULL);
		struct tm	tm;
		int			c;
		char		temp[40];

		localtime_r(&tval, &tm);
		/* setup a new filename */
		filename.clear();
		for (int i=0; pathname[i] != '\0'; i++)
		{
			int		c = pathname[i];

			if (c != '%')
				filename += c;
			else
			{
				c = pathname[++i];
				assert(c != '\0');
				switch (c)
				{
					case 'Y':   /* Year (4-digits) */
						sprintf(temp, "%04d", tm.tm_year + 1900);
						break;
					case 'y':   /* Year (2-digits) */
						sprintf(temp, "%02d", tm.tm_year % 100);
						break;
					case 'm':   /* month (01-12) */
						sprintf(temp, "%02d", tm.tm_mon + 1);
						break;
					case 'd':   /* day (01-31) */
						sprintf(temp, "%02d", tm.tm_mday);
						break;
					case 'H':   /* hour (00-23) */
						sprintf(temp, "%02d", tm.tm_hour);
						break;
					case 'M':   /* minute (00-59) */
						sprintf(temp, "%02d", tm.tm_min);
						break;
					case 'S':   /* second (00-59) */
						sprintf(temp, "%02d", tm.tm_sec);
						break;
					case 'p':   /* process's PID */
						sprintf(temp, "%u", getpid());
						break;
					case 'q':   /* sequence number */
						sprintf(temp, "%u", filename_seqno++);
						break;
					default:
						Elog("unknown format character: '%%%c' in '%s'",
							 c, pathname);
				}
				filename += temp;
			}
		}
		if (retry > 0)
		{
			sprintf(temp, ".%d", retry);
			filename += temp;
		}
		/* open the output file */
		fdesc = open(filename.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
		if (fdesc < 0 && errno != EEXIST)
			Elog("failed on open('%s'): %m", filename.c_str());
	}
	/* Open FileOutputStream */
	auto    rv = arrow::io::FileOutputStream::Open(fdesc);

	if (!rv.ok())
		Elog("failed on arrow::io::FileOutputStream::Open('%s'): %s",
			 filename.c_str(), rv.status().ToString().c_str());
	return rv.ValueOrDie();
}

void
arrowFileWrite::ResetBuffers(void)
{
	for (int j=0; j < arrow_columns.size(); j++)
		arrow_columns[j]->Reset();
	chunk_nitems = 0;
}

std::shared_ptr<arrow::RecordBatch>
arrowFileWrite::BuildBuffers(void)
{
	std::vector<arrowArray> arrow_arrays_vector;
	int64_t		nrows = -1;

	for (int j=0; j < arrow_columns.size(); j++)
	{
		auto		column = arrow_columns[j];
		int64_t		__nrows = column->Finish();

		if (nrows < 0)
			nrows = __nrows;
		else if (nrows != __nrows)
			Elog("Bug? number of rows mismatch across the buffers");
		assert(column->arrow_array != NULL);
		arrow_arrays_vector.push_back(column->arrow_array);
	}
	return arrow::RecordBatch::Make(arrow_schema, nrows, arrow_arrays_vector);
}

std::shared_ptr<arrow::KeyValueMetadata>
arrowFileWrite::buildArrowMinMaxStats(void)
{
	auto	custom_metadata = std::make_shared<arrow::KeyValueMetadata>();

	for (int j=0; j < arrow_columns.size(); j++)
	{
		auto	column = arrow_columns[j];

		if (column->stats_enabled)
			column->appendStats(custom_metadata);
	}
	return custom_metadata;
}

void
arrowFileWrite::writeArrowFile(void)
{
	std::string	filename;
	auto	file_out_stream = OpenFile(filename);
	auto	rbatch = BuildBuffers();

	std::shared_ptr<arrow::ipc::RecordBatchWriter> arrow_file_writer;
	{
		auto	rv = arrow::ipc::MakeFileWriter(file_out_stream, arrow_schema);
		if (!rv.ok())
			Elog("failed on arrow::ipc::MakeFileWriter for '%s': %s",
				 filename.c_str(),
				 rv.status().ToString().c_str());
		arrow_file_writer = rv.ValueOrDie();
	}
	/* write out record-batch */
	{
		auto	rv = arrow_file_writer->WriteRecordBatch(*rbatch, buildArrowMinMaxStats());
		if (!rv.ok())
			Elog("failed on arrow::ipc::RecordBatchWriter::WriteRecordBatch: %s",
				 rv.ToString().c_str());
	}
	/* write out arrow footer */
	{
		auto	rv = arrow_file_writer->Close();
		if (!rv.ok())
			Elog("failed on arrow::ipc::RecordBatchWriter::Close: %s",
				 rv.ToString().c_str());
	}
	/* close the file */
	{
		auto	rv = file_out_stream->Close();
		if (!rv.ok())
			Elog("failed on arrow::io::FileOutputStream::Close(): %s",
				 rv.ToString().c_str());
	}
}

std::shared_ptr<parquet::WriterProperties>
arrowFileWrite::buildParquetWriterProperties(void)
{
	parquet::WriterProperties::Builder builder;

	/* created by "fluentd-arrow-write" */
	builder.created_by(std::string("fluentd-arrow-write"));
	/* don't split chunks by number of lines basis */
	builder.max_row_group_length(INT64_MAX);
	/* setup compression method */
	builder.compression(default_compression);
	for (int j=0; j < arrow_columns.size(); j++)
	{
		auto	column = arrow_columns[j];

		if (default_compression != column->compression)
			builder.compression(column->attname,
								column->compression);
		if (!column->stats_enabled)
			builder.disable_statistics(column->attname);
	}
	return builder.build();
}

void
arrowFileWrite::writeParquetFile(void)
{
	std::string filename;
	auto	file_out_stream = OpenFile(filename);
	auto	rbatch = BuildBuffers();

	std::unique_ptr<parquet::arrow::FileWriter>	parquet_file_writer;
	{
		auto	pq_writer_props = buildParquetWriterProperties();
		auto	pq_arrow_writer_props = parquet::default_arrow_writer_properties();
		auto	rv = parquet::arrow::FileWriter::Open(*arrow_schema,
													  arrow::default_memory_pool(),
													  file_out_stream,
													  buildParquetWriterProperties(),
													  pq_arrow_writer_props);
		if (!rv.ok())
			Elog("failed on parquet::arrow::FileWriter::Open('%s'): %s",
				 filename.c_str(),
				 rv.status().ToString().c_str());
		parquet_file_writer = std::move(rv).ValueOrDie();
	}
	/* write out row-group */
	{
		auto	rv = parquet_file_writer->WriteRecordBatch(*rbatch);
		if (!rv.ok())
			Elog("failed on parquet::arrow::FileWriter::WriteRecordBatch: %s",
				 rv.ToString().c_str());
	}
	/* write out parquet footer */
	{
		auto	rv = parquet_file_writer->Close();
		if (!rv.ok())
			Elog("failed on parquet::arrow::FileWriter::Close: %s",
				 rv.ToString().c_str());
	}
	/* close the file */
	{
		auto	rv = file_out_stream->Close();
		if (!rv.ok())
			Elog("failed on arrow::io::FileOutputStream::Close(): %s",
				 rv.ToString().c_str());
	}
}

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
		std::shared_ptr<class arrowFileWriteColumn> arrow_column;
		char   *field_name = tok;
		char   *field_type;
		char   *field_extra;
		bool	ts_column = false;
		bool	tag_column = false;
		int		stats_enabled = -1;
		arrow::Compression::type compression = fw_state->default_compression;

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
				if (strcmp(__tok, "stats_enabled") == 0)
					stats_enabled = 1;
				else if (strcmp(__tok, "stats_disabled") == 0)
					stats_enabled = 0;
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
		if (stats_enabled < 0)
			stats_enabled = (fw_state->parquet_mode ? 1 : 0);
		assert(stats_enabled == 0 || stats_enabled == 1);
		if (fw_state->ts_column && strcasecmp(fw_state->ts_column, field_name) == 0)
			ts_column = true;
		if (fw_state->tag_column && strcasecmp(fw_state->tag_column, field_name) == 0)
			tag_column = true;

		if (strcasecmp(field_type, "bool") == 0 ||
			strcasecmp(field_type, "boolean") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnBoolean>(field_name,
																		 stats_enabled,
																		 compression);
		else if (strcasecmp(field_type, "int8") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnInt8>(field_name,
																	  stats_enabled,
																	  compression);
		else if (strcasecmp(field_type, "int16") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnInt16>(field_name,
																	   stats_enabled,
																	   compression);
		else if (strcasecmp(field_type, "int32") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnInt32>(field_name,
																	   stats_enabled,
																	   compression);
		else if (strcasecmp(field_type, "int64") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnInt64>(field_name,
																	   stats_enabled,
																	   compression);
		else if (strcasecmp(field_type, "uint8") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnUInt8>(field_name,
																	   stats_enabled,
																	   compression);
		else if (strcasecmp(field_type, "uint16") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnUInt16>(field_name,
																		stats_enabled,
																		compression);
		else if (strcasecmp(field_type, "uint32") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnUInt32>(field_name,
																		stats_enabled,
																		compression);
		else if (strcasecmp(field_type, "uint64") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnUInt64>(field_name,
																		stats_enabled,
																		compression);
		else if (strcasecmp(field_type, "float16") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnHalfFloat>(field_name,
																		   stats_enabled,
																		   compression);
		else if (strcasecmp(field_type, "float32") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnFloat>(field_name,
																	   stats_enabled,
																	   compression);
		else if (strcasecmp(field_type, "float64") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnDouble>(field_name,
																		stats_enabled,
																		compression);
		else if (strcasecmp(field_type, "decimal") == 0 ||
				 strcasecmp(field_type, "decimal128") == 0 ||
				 strncasecmp(field_type, "decimal(", 8) == 0 ||		/* with precision, scale */
				 strncasecmp(field_type, "decimal128(", 11) == 0)	/* with precision, scale */
			arrow_column = std::make_shared<arrowFileWriteColumnDecimal128>(field_type,
																			field_name,
																			stats_enabled,
																			compression);
		else if (strcasecmp(field_type, "date[day]") == 0 ||
				 strcasecmp(field_type, "date") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnDate_sec>(field_name,
																		  stats_enabled,
																		  compression);
		else if (strcasecmp(field_type, "date[ms]") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnDate_ms>(field_name,
																		 stats_enabled,
																		 compression);
		else if (strcasecmp(field_type, "time[sec]") == 0 ||
				 strcasecmp(field_type, "time") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnTime_sec>(field_name,
																		  stats_enabled,
																		  compression);
		else if (strcasecmp(field_type, "time[ms]") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnTime_ms>(field_name,
																		 stats_enabled,
																		 compression);
		else if (strcasecmp(field_type, "time[us]") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnTime_us>(field_name,
																		 stats_enabled,
																		 compression);
		else if (strcasecmp(field_type, "time[ns]") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnTime_ns>(field_name,
																		 stats_enabled,
																		 compression);
		else if (strcasecmp(field_type, "timestamp[sec]") == 0)
		{
			arrow_column = std::make_shared<arrowFileWriteColumnTimestamp_sec>(field_name,
																			   stats_enabled,
																			   compression);
			assert(false);
		}
		else if (strcasecmp(field_type, "timestamp[ms]") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnTimestamp_ms>(field_name,
																			  stats_enabled,
																			  compression);
		else if (strcasecmp(field_type, "timestamp[us]") == 0 ||
				 strcasecmp(field_type, "timestamp") == 0)			/* default precision */
			arrow_column = std::make_shared<arrowFileWriteColumnTimestamp_us>(field_name,
																			  stats_enabled,
																			  compression);
		else if (strcasecmp(field_type, "timestamp[ns]") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnTimestamp_ns>(field_name,
																			  stats_enabled,
																			  compression);
		else if (strcasecmp(field_type, "text") == 0 ||
				 strcasecmp(field_type, "utf8") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnUtf8>(field_name,
																	  stats_enabled,
																	  compression);
		else if (strcasecmp(field_type, "ipaddr4") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnIpAddr4>(field_name,
																		 stats_enabled,
																		 compression);
		else if (strcasecmp(field_type, "ipaddr6") == 0)
			arrow_column = std::make_shared<arrowFileWriteColumnIpAddr6>(field_name,
																		 stats_enabled,
																		 compression);
		else
			Elog("ArrowFileWrite: not a supported type '%s' for '%s'", field_type, field_name);
		arrow_column->ts_column = ts_column;
		arrow_column->tag_column = tag_column;
		fw_state->arrow_columns.push_back(arrow_column);
	}
}

static VALUE
arrowFileWrite__initialize(VALUE __self,
						   VALUE __pathname,
						   VALUE __schema_defs,
						   VALUE __params)
{
	char   *emsg = NULL;
	arrowFileWrite *fw_state;

	rb_require("time");
	rb_require("ipaddr");
	TypedData_Get_Struct(__self, arrowFileWrite, &arrowFileWriteType, fw_state);
	try {
		std::vector<arrowField>	arrow_fields;

		__arrowFileWritePathnameValidator(fw_state, __pathname);
		__arrowFileWriteParseParamDefs(fw_state, __params);
		__arrowFileWriteParseSchemaDefs(fw_state, __schema_defs);
		/* build arrowSchema */
		for (int i=0; i < fw_state->arrow_columns.size(); i++)
		{
			auto	column = fw_state->arrow_columns[i];
			auto	builder = column->arrow_builder;
			arrow_fields.push_back(arrow::field(column->attname, builder->type()));
		}
		fw_state->arrow_schema = arrow::schema(arrow_fields);
	}
	catch (const std::exception &e) {
		const char *estr = e.what();
		emsg = (char *)alloca(strlen(estr)+1);
		strcpy(emsg, estr);
	}
	/* transform libarrow/libparquet exception to ruby exception */
	if (emsg)
		Elog("arrow-write: %s", emsg);
	return __self;
}

static VALUE
arrowFileWrite__processOneRow(RB_BLOCK_CALL_FUNC_ARGLIST(__yield, __self))
{
	VALUE	tag = rb_funcall(__yield, rb_intern("fetch"), 1, INT2NUM(0));
	VALUE	ts = rb_funcall(__yield, rb_intern("fetch"), 1, INT2NUM(1));
	VALUE	record = rb_funcall(__yield, rb_intern("fetch"), 1, INT2NUM(2));
	arrowFileWrite *fw_state;

	TypedData_Get_Struct(__self, arrowFileWrite, &arrowFileWriteType, fw_state);
	for (int j=0; j < fw_state->arrow_columns.size(); j++)
	{
		auto	column = fw_state->arrow_columns[j];
		VALUE	datum;

		if (column->ts_column)
			datum = ts;
		else if (column->tag_column)
			datum = tag;
		else
			datum = rb_funcall(record, rb_intern("fetch"), 2,
							   rb_str_new_cstr(column->attname.c_str()), Qnil);
		column->putValue(datum);
	}
	fw_state->chunk_nitems++;
	return Qtrue;
}

static VALUE
arrowFileWrite__writeChunk(VALUE __self,
						   VALUE __chunk)
{
	arrowFileWrite *fw_state;

	TypedData_Get_Struct(__self, arrowFileWrite, &arrowFileWriteType, fw_state);
	fw_state->ResetBuffers();
	rb_block_call(__chunk,
				  rb_intern("each"),
				  0,
				  NULL,
				  arrowFileWrite__processOneRow,
				  __self);
	if (fw_state->chunk_nitems > 0)
	{
		if (!fw_state->parquet_mode)
			fw_state->writeArrowFile();
		else
			fw_state->writeParquetFile();
	}
	return INT2NUM(0);
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

	for (int j=0; j < fw_state->arrow_columns.size(); j++)
	{
		auto	field = fw_state->arrow_columns[j];

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
