/*
 * arrow_write.h
 *
 * Common routines to write out Apache Arrow/Parquet files
 * ----
 * Copyright 2011-2025 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2025 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef _ARROW_WRITE_H_
#define _ARROW_WRITE_H_
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/util/value_parsing.h>
#include <ctype.h>
#include <parquet/arrow/writer.h>
#include <stdarg.h>
#include <string>

#ifndef Elog
#define Elog(fmt,...)	__throw_error("[ERROR %s:%d] " fmt,			\
									  __FILE__, __LINE__, ##__VA_ARGS__)
#endif
#ifndef Info
#define Info(fmt,...)	fprintf(stderr, "[INFO: %s:%d] " fmt "\n",	\
								__FILE__, __LINE__, ##__VA_ARGS__)
#endif
#define Max(a,b)	((a)>(b) ? (a) : (b))
#define Min(a,b)	((a)<(b) ? (a) : (b))
#define BITMAPLEN(NITEMS)	(((NITEMS) + 7) / 8)
#define ARROW_ALIGN(LEN)	(((uintptr_t)(LEN) + 63UL) & ~63UL)

/*
 * utility functions
 */
[[noreturn]] static inline void
__throw_error(const char *fmt, ...)
{
	va_list ap;
	int		n, bufsz = 400;
	char   *buf = (char *)alloca(bufsz);

	va_start(ap, fmt);
	for (;;)
	{
		n = vsnprintf(buf, bufsz, fmt, ap);
		if (n < bufsz)
			break;
		bufsz += (bufsz + 1000);
		buf = (char *)alloca(bufsz);
	}
	va_end(ap);

	throw std::runtime_error(buf);
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

static inline char *
__trim_quotable(char *token)
{
	token = __trim(token);
	if (token)
	{
		char   *tail = token + strlen(token) - 1;

		if (*token == '"' && token < tail && *tail == '"')
		{
			*tail = '\0';
			return token + 1;
		}
	}
	return token;
}

static inline char *
__strtok_quotable(char *line, char delim, char **saveptr)
{
	char   *r_pos = (line ? line : *saveptr);
	char   *w_pos = r_pos;
	char   *start = w_pos;
	bool	in_quote = false;

	assert(saveptr != NULL);
	if (!r_pos)
		return NULL;	/* already touched to the end */
	/* skip white-spaces in the head */
	while (isspace(*r_pos))
		r_pos++;
	for (;;)
	{
		int		c = *r_pos++;

		if (c == '\0')
		{
			*saveptr = NULL;
			*w_pos++ = '\0';
			break;
		}
		else if (!in_quote && c == delim)
		{
			*saveptr = r_pos;
			*w_pos++ = '\0';
			break;
		}
		else if (c == '"')
		{
			if (*r_pos == '"')	/* "" is just an escaped '"' */
				*w_pos++ = *r_pos++;
			else
				in_quote = !in_quote;
		}
		else if (c == '\\')
		{
			c = *r_pos++;
			switch (c)
			{
				case '\0':
					*saveptr = NULL;
					*w_pos++ = '\0';
					goto out;
				case '0':
					*w_pos++ = '\0';
					break;
				case 'n':
					*w_pos++ = '\n';
					break;
				case 'r':
					*w_pos++ = '\r';
					break;
				case 't':
					*w_pos++ = '\t';
					break;
				default:
					*w_pos++ = c;
					break;
			}
		}
		else
		{
			*w_pos++ = c;
		}
	}
out:
	return __trim(start);
}

/*
 * Definition of arrowFileWriter
 */
using	arrowSchema		= std::shared_ptr<arrow::Schema>;
using	arrowField		= std::shared_ptr<arrow::Field>;
using	arrowBuilder	= std::shared_ptr<arrow::ArrayBuilder>;
using	arrowArray		= std::shared_ptr<arrow::Array>;
using	arrowMetadata	= std::shared_ptr<arrow::KeyValueMetadata>;

class arrowFileWriter
{
protected:
	std::shared_ptr<arrow::io::FileOutputStream>	file_out_stream;
	std::shared_ptr<arrow::ipc::RecordBatchWriter>	arrow_file_writer;
	std::unique_ptr<parquet::arrow::FileWriter>		parquet_file_writer;
public:
	std::string		pathname;
	bool			parquet_mode;
	arrow::Compression::type default_compression;
	arrowSchema		arrow_schema;
	arrowMetadata	table_metadata;
	std::vector<std::shared_ptr<class arrowFileWriterColumn>> arrow_columns;
	arrowFileWriter(bool __parquet_mode = false,
					arrow::Compression::type __compression = arrow::Compression::type::ZSTD)
	{
		file_out_stream = nullptr;
		arrow_file_writer = nullptr;
		parquet_file_writer = nullptr;
		parquet_mode = __parquet_mode;
		default_compression = __compression;
		arrow_schema = nullptr;
		table_metadata = std::make_shared<arrow::KeyValueMetadata>();
	}
	bool		Open(const char *__pathname);
	void		Close();
	void		ParseSchemaDefs(const char *schema_defs);
};

/*
 * arrowFileWriterColumn
 *
 * arrowFileWriterColumn
 *  | + arrowFileWriterScalarColumn
 *  |    + arrowFileWriterSimpleColumn
 *  |    + arrowFileWriterDecimalColumn
 *  |    + arrowFileWriterVariablelColumn
 *  + arrowFileWriterArrayColumn
 *  + arrowFileWriterStructColumn
 */
class arrowFileWriterColumn
{
public:
	std::shared_ptr<arrow::DataType> arrow_type;
	std::string		field_name;
	bool			stats_enabled;
	arrow::Compression::type compression;	/* valid only if parquet-mode */
	arrowBuilder	arrow_builder;
	arrowArray		arrow_array;
	std::vector<std::shared_ptr<class arrowFileWriterColumn>> children;	/* only Struct/List */
	arrowMetadata	field_metadata;
	arrowFileWriterColumn(std::shared_ptr<arrow::DataType> __arrow_type,
						  const char *__field_name,
						  bool __stats_enabled,
						  arrow::Compression::type __compression = arrow::Compression::type::UNCOMPRESSED)
	{
		field_name = std::string(__field_name);
		arrow_type = __arrow_type;
		stats_enabled = __stats_enabled;
		compression = __compression;
		arrow_builder = nullptr;
		arrow_array = nullptr;
		field_metadata = std::make_shared<arrow::KeyValueMetadata>();
	}
	virtual size_t	chunkSize() = 0;
	virtual size_t	putValue(const void *ptr, size_t sz) = 0;
	virtual size_t	moveValue(std::shared_ptr<class arrowFileWriterColumn> buddy, int64_t index) = 0;
	virtual void	Reset(void)
	{
		arrow_builder->Reset();
		arrow_array = nullptr;
	}
	virtual void	Finish(void)
	{
		auto	rv = arrow_builder->Finish(&arrow_array);
		if (!rv.ok())
			Elog("failed on arrow::ArrayBuilder::Finish: %s",
				 rv.ToString().c_str());
	}
};

/*
 * Field for a basic scalar type
 */
template <typename BUILDER_TYPE,
		  typename ARRAY_TYPE,
          typename CPP_TYPE>
class arrowFileWriterScalarColumn : public arrowFileWriterColumn
{
public:
	arrowFileWriterScalarColumn(std::shared_ptr<arrow::DataType> __arrow_type,
								const char *__field_name,
								bool __stats_enabled,
								arrow::Compression::type __compression)
		: arrowFileWriterColumn(__arrow_type,
								__field_name,
								__stats_enabled,
								__compression)
	{
		arrow_builder = std::make_shared<BUILDER_TYPE>(__arrow_type, arrow::default_memory_pool());
	}
	virtual void
	appendStats(arrowMetadata custom_metadata)
	{
		/* do nothing if no statistics support */
	}
	virtual void
	updateStats(CPP_TYPE &datum)
	{
		/* do nothing if no statistics support */
	}
	virtual CPP_TYPE	fetchValue(const void *ptr, size_t sz) = 0;
	virtual CPP_TYPE	fetchArrayValue(arrowArray array, int64_t index) = 0;
	size_t
	putValue(const void *ptr, size_t sz)
	{
		auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(arrow_builder);
		arrow::Status rv;

		if (!ptr)
			rv = builder->AppendNull();
		else
		{
			CPP_TYPE	datum = fetchValue(ptr, sz);
			updateStats(datum);
			rv = builder->Append(datum);
		}
		if (!rv.ok())
			Elog("unable to put value to '%s' field (%s): %s",
				 field_name.c_str(),
				 arrow_type->name().c_str(),
				 rv.ToString().c_str());
		return chunkSize();
	}
	size_t
	moveValue(std::shared_ptr<class arrowFileWriterColumn> buddy, int64_t index)
	{
		auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(arrow_builder);
		auto	buddy_array = std::dynamic_pointer_cast<ARRAY_TYPE>(buddy->arrow_array);
		arrow::Status rv;

		assert(index < buddy_array->length());
		if (buddy_array->IsNull(index))
			rv = builder->AppendNull();
		else
		{
			CPP_TYPE	datum = fetchArrayValue(buddy_array, index);
			updateStats(datum);
			rv = builder->Append(datum);
		}
		if (!rv.ok())
			Elog("unable to move value to '%s' field (%s): %s",
				 field_name.c_str(),
				 arrow_type->name().c_str(),
				 rv.ToString().c_str());
		return chunkSize();
	}
};

/*
 * Simple scalar types
 */
template <typename BUILDER_TYPE,
		  typename ARRAY_TYPE,
		  typename CPP_TYPE>
class arrowFileWriterSimpleColumn : public arrowFileWriterScalarColumn<BUILDER_TYPE,
																	   ARRAY_TYPE,
																	   CPP_TYPE>
{
	bool		stats_is_valid;
	CPP_TYPE	stats_min_value;
	CPP_TYPE	stats_max_value;
public:
	arrowFileWriterSimpleColumn(std::shared_ptr<arrow::DataType> __arrow_type,
								const char *__field_name,
								bool __stats_enabled,
								arrow::Compression::type __compression)
		: arrowFileWriterScalarColumn<BUILDER_TYPE,
									  ARRAY_TYPE,
									  CPP_TYPE>(__arrow_type,
												__field_name,
												__stats_enabled,
												__compression)
	{
		stats_is_valid = false;
	}
	void
	appendStats(arrowMetadata custom_metadata)
	{
		if (!stats_is_valid)
			return;
		custom_metadata->Append(std::string("min_max_stats.") + this->field_name,
								std::to_string(stats_min_value) +
								std::string(",") +
								std::to_string(stats_max_value));
	}
	void
	updateStats(CPP_TYPE datum)
	{
		if (this->stats_enabled)
		{
			if (!stats_is_valid)
			{
				stats_min_value = datum;
				stats_max_value = datum;
				stats_is_valid = true;
			}
			else if (datum < stats_min_value)
				stats_min_value = datum;
			else if (datum > stats_max_value)
				stats_max_value = datum;
		}
	}
	size_t
	chunkSize(void)
	{
		auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(this->arrow_builder);

		return (ARROW_ALIGN(builder->null_count() > 0 ? BITMAPLEN(builder->length()) : 0) +
				ARROW_ALIGN(sizeof(CPP_TYPE) * builder->length()));
	}
	CPP_TYPE
	fetchArrayValue(arrowArray buddy_array, int64_t index)
	{
		auto	array = std::dynamic_pointer_cast<ARRAY_TYPE>(buddy_array);

		return array->Value(index);
	}
	void
	Reset(void)
	{
		arrowFileWriterColumn::Reset();
		stats_is_valid = false;
	}
};

#define __ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(NAME,PREFIX,CPP_TYPE,TYPE_OBJ) \
	class arrowFileWriterColumn##NAME : public arrowFileWriterSimpleColumn<arrow::PREFIX##Builder, \
																		   arrow::PREFIX##Array, \
																		   CPP_TYPE> \
	{																	\
	public:																\
		arrowFileWriterColumn##NAME(const char *__field_name,			\
									bool __stats_enabled,				\
									arrow::Compression::type __compression)	\
		: arrowFileWriterSimpleColumn<arrow::PREFIX##Builder,			\
									  arrow::PREFIX##Array,				\
									  CPP_TYPE>(arrow::TYPE_OBJ,		\
												__field_name,			\
												__stats_enabled,		\
												__compression)			\
		{																\
		}																\
		CPP_TYPE	fetchValue(const void *ptr, size_t sz);				\
	};
#define ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(NAME,PREFIX,CPP_TYPE,TYPE_OBJ) \
	__ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(NAME,PREFIX,CPP_TYPE,TYPE_OBJ) \
	CPP_TYPE															\
	arrowFileWriterColumn##NAME::fetchValue(const void *ptr, size_t sz)	\
	{																	\
		auto	ts_type = std::dynamic_pointer_cast<arrow::PREFIX##Type>(arrow_type); \
		const char *token = (const char *)ptr;							\
		CPP_TYPE	datum;												\
																		\
		if (!arrow::internal::ParseValue<arrow::PREFIX##Type>(*ts_type, token, sz, &datum)) \
			Elog("unable to fetch token [%.*s] for '%s' (%s)",			\
				 (int)sz, token,										\
				 field_name.c_str(),									\
				 arrow_type->name().c_str());							\
		return datum;													\
	}
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Boolean,       Boolean,   bool,     boolean())
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Int8,          Int8,      int8_t,   int8())
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Int16,         Int16,     int16_t,  int16())
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Int32,         Int32,     int32_t,  int32())
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Int64,         Int64,     int64_t,  int64())
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(UInt8,         UInt8,     uint8_t,  uint8())
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(UInt16,        UInt16,    uint16_t, uint16())
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(UInt32,        UInt32,    uint32_t, uint32())
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(UInt64,        UInt64,    uint64_t, uint64())
__ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Float16,     HalfFloat, uint16_t, float16())
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Float32,       Float,     float,    float32())
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Float64,       Double,    double,   float64())
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Date_sec,      Date32,    int32_t,  date32())
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Date_ms,       Date64,    int64_t,  date64())
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Time_sec,      Time32,    int32_t,  time32(arrow::TimeUnit::SECOND))
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Time_ms,       Time32,    int32_t,  time32(arrow::TimeUnit::MILLI))
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Time_us,       Time64,    int64_t,  time64(arrow::TimeUnit::MICRO))
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Time_ns,       Time64,    int64_t,  time64(arrow::TimeUnit::NANO))
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Timestamp_sec, Timestamp, int64_t,  timestamp(arrow::TimeUnit::SECOND))
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Timestamp_ms,  Timestamp, int64_t,  timestamp(arrow::TimeUnit::MILLI))
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Timestamp_us,  Timestamp, int64_t,  timestamp(arrow::TimeUnit::MICRO))
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Timestamp_ns,  Timestamp, int64_t,  timestamp(arrow::TimeUnit::NANO))

/* special handling for float16 */
uint16_t
arrowFileWriterColumnFloat16::fetchValue(const void *ptr, size_t sz)
{
	const char *token = (const char *)ptr;
	arrow::util::Float16 datum;

	if (!arrow::internal::ParseValue<arrow::HalfFloatType>(token, sz, &datum))
		Elog("unable to fetch token [%.*s] for '%s' (%s)",
			 (int)sz, token,
			 field_name.c_str(),
			 arrow_type->name().c_str());
	return datum.bits();
}

/*
 * Decimal Field template
 */
template <typename BUILDER_TYPE,
		  typename ARRAY_TYPE,
		  typename ARROW_TYPE,
		  typename CPP_TYPE>
class arrowFileWriterDecimalColumn : public arrowFileWriterScalarColumn<BUILDER_TYPE,
																		ARRAY_TYPE,
																		CPP_TYPE>
{
	bool		stats_is_valid;
	CPP_TYPE	stats_min_value;
	CPP_TYPE	stats_max_value;
public:
	arrowFileWriterDecimalColumn(std::shared_ptr<arrow::DataType> __arrow_type,
								 const char *__field_name,
								 bool __stats_enabled,
								 arrow::Compression::type __compression)
		: arrowFileWriterScalarColumn<BUILDER_TYPE,
									  ARRAY_TYPE,
									  CPP_TYPE>(__arrow_type,
												__field_name,
												__stats_enabled,
												__compression)
	{
		stats_is_valid = false;
	}
	void
	appendStats(arrowMetadata custom_metadata)
	{
		if (stats_is_valid)
		{
			auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(this->arrow_builder);
			auto	d_type = std::dynamic_pointer_cast<ARROW_TYPE>(builder->type());

			custom_metadata->Append(std::string("min_max_stats.") + this->field_name,
									stats_min_value.ToString(d_type->scale()) +
									std::string(",") +
									stats_max_value.ToString(d_type->scale()));
		}
	}
	void
	updateStats(CPP_TYPE datum)
	{
		if (this->stats_enabled)
		{
			if (!stats_is_valid)
			{
				stats_min_value = datum;
				stats_max_value = datum;
				stats_is_valid = true;
			}
			else if (datum < stats_min_value)
				stats_min_value = datum;
			else if (datum > stats_max_value)
				stats_max_value = datum;
		}
	}
	size_t
	chunkSize(void)
	{
		auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(this->arrow_builder);

		return (ARROW_ALIGN(builder->null_count() > 0 ? BITMAPLEN(builder->length()) : 0) +
				ARROW_ALIGN(sizeof(CPP_TYPE) * builder->length()));
	}
	CPP_TYPE
	fetchValue(const void *ptr, size_t sz)
	{
		auto		builder = std::dynamic_pointer_cast<BUILDER_TYPE>(this->arrow_builder);
		auto		d_type = std::dynamic_pointer_cast<ARROW_TYPE>(this->arrow_type);
		const char *token = (const char *)ptr;
		CPP_TYPE	datum;
		int			precision, scale;
		arrow::Status rv;

		rv = CPP_TYPE::FromString(token, &datum, &precision, &scale);
		if (!rv.ok())
			Elog("unable to fetch token [%.*s] for '%s' (%s)",
				 (int)sz, token,
				 this->field_name.c_str(),
				 d_type->name().c_str());
		while (scale < d_type->scale())
		{
			scale++;
			datum *= 10;
		}
		while (scale > d_type->scale())
		{
			if (--scale == d_type->scale())
				datum += 5;
			datum /= 10;
		}
		return datum;
	}
	CPP_TYPE
	fetchArrayValue(arrowArray buddy_array, int64_t index)
	{
		auto	array = std::dynamic_pointer_cast<ARRAY_TYPE>(buddy_array);

		return *((const CPP_TYPE *)array->Value(index));
	}
	void
	Reset(void)
	{
		arrowFileWriterColumn::Reset();
		stats_is_valid = false;
	}
};
#define ARROW_FILE_WRITER_COLUMN_DECIMAL_TEMPLATE(NAME, TYPE_OBJECT_NAME) \
	class arrowFileWriterColumn##NAME : public arrowFileWriterDecimalColumn<arrow::NAME##Builder, \
																			arrow::NAME##Array,	\
																			arrow::NAME##Type, \
																			arrow::NAME> \
	{																	\
	public:																\
		arrowFileWriterColumn##NAME(const char *__field_name,			\
									int __precision,					\
									int __scale,						\
									bool __stats_enabled,				\
									arrow::Compression::type __compression)	\
		: arrowFileWriterDecimalColumn<arrow::NAME##Builder,			\
									   arrow::NAME##Array,				\
									   arrow::NAME##Type,				\
									   arrow::NAME>(arrow::TYPE_OBJECT_NAME(__precision, \
																			__scale), \
													__field_name,		\
													__stats_enabled,	\
													__compression)		\
		{}																\
	};
ARROW_FILE_WRITER_COLUMN_DECIMAL_TEMPLATE(Decimal128, decimal128)

/*
 * Interval field type
 */
template <typename BUILDER_TYPE,
		  typename ARRAY_TYPE,
		  typename ARROW_TYPE,
		  typename CPP_TYPE,
		  arrow::Type::type ARROW_TYPE_ID>
class __arrowFileWriterIntervalColumn : public arrowFileWriterScalarColumn<BUILDER_TYPE,
																		   ARRAY_TYPE,
																		   CPP_TYPE>
{
	static std::shared_ptr<arrow::DataType>
	__build_interval_arrow_type(void)
	{
		return (ARROW_TYPE_ID == arrow::Type::INTERVAL_MONTHS ? arrow::month_interval() :
				ARROW_TYPE_ID == arrow::Type::INTERVAL_DAY_TIME ? arrow::day_time_interval() :
				ARROW_TYPE_ID == arrow::Type::INTERVAL_MONTH_DAY_NANO ? arrow::month_day_nano_interval() : nullptr);
	}
public:
	__arrowFileWriterIntervalColumn(const char *__field_name,
									bool __stats_enabled,
									arrow::Compression::type __compression)
		: arrowFileWriterScalarColumn<BUILDER_TYPE,
									  ARRAY_TYPE,
									  CPP_TYPE>(__build_interval_arrow_type(),
												__field_name,
												__stats_enabled,
												__compression)
	{}
    size_t      chunkSize()
	{
		auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(this->arrow_builder);

		return (ARROW_ALIGN(builder->null_count() > 0 ? BITMAPLEN(builder->length()) : 0) +
				ARROW_ALIGN(sizeof(CPP_TYPE) * builder->length()));
	}
	CPP_TYPE	fetchValue(const void *ptr, size_t sz)
	{
		auto		ts_type = std::dynamic_pointer_cast<ARROW_TYPE>(this->arrow_type);
		const char *token = (const char *)ptr;
		CPP_TYPE	datum;

		if (!arrow::internal::ParseValue<ARROW_TYPE>(*ts_type, token, sz, &datum))
			Elog("unable to fetch token [%.*s] for '%s' (%s)",
				 (int)sz, token,
				 this->field_name.c_str(),
				 this->arrow_type->name().c_str());
		return datum;
	}
	CPP_TYPE	fetchArrayValue(arrowArray buddy_array, int64_t index)
	{
		auto	array = std::dynamic_pointer_cast<ARRAY_TYPE>(buddy_array);

		return array->Value(index);
	}
};
using arrowFileWriterIntervalMonth
	= __arrowFileWriterIntervalColumn<arrow::MonthIntervalBuilder,
									  arrow::MonthIntervalArray,
									  arrow::MonthIntervalType,
									  arrow::MonthIntervalType,
									  arrow::Type::INTERVAL_MONTHS>;
using arrowFileWriterIntervalDayTime
	= __arrowFileWriterIntervalColumn<arrow::DayTimeIntervalBuilder,
									  arrow::DayTimeIntervalArray,
									  arrow::DayTimeIntervalType,
									  arrow::DayTimeIntervalType,
									  arrow::Type::INTERVAL_DAY_TIME>;
using arrowFileWriterIntervalMonthDayNano
	= __arrowFileWriterIntervalColumn<arrow::MonthDayNanoIntervalBuilder,
									  arrow::MonthDayNanoIntervalArray,
									  arrow::MonthDayNanoIntervalType,
									  arrow::MonthDayNanoIntervalType,
									  arrow::Type::INTERVAL_MONTH_DAY_NANO>;
/*
 * variable length type (String, Binary, ...)
 */
template <typename BUILDER_TYPE,
		  typename ARRAY_TYPE,
		  typename CPP_TYPE,
		  typename OFFSET_TYPE>
class arrowFileWriterVariableColumn : public arrowFileWriterScalarColumn<BUILDER_TYPE,
																		 ARRAY_TYPE,
																		 CPP_TYPE>
{
public:
	arrowFileWriterVariableColumn(std::shared_ptr<arrow::DataType> __arrow_type,
								  const char *__field_name,
								  bool __stats_enabled,
								  arrow::Compression::type __compression)
		: arrowFileWriterScalarColumn<BUILDER_TYPE,
									  ARRAY_TYPE,
									  CPP_TYPE>(__arrow_type,
												__field_name,
												__stats_enabled,
												__compression)
	{}
	size_t
	chunkSize(void)
	{
		auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(this->arrow_builder);

		return (ARROW_ALIGN(builder->null_count() > 0 ? BITMAPLEN(builder->length()) : 0) +
				ARROW_ALIGN(sizeof(OFFSET_TYPE) * builder->length()) +
				ARROW_ALIGN(builder->value_data_length()));
	}
	CPP_TYPE
	fetchValue(const void *ptr, size_t sz)
	{
		return std::string_view((const char *)ptr, sz);
	}
	CPP_TYPE
	fetchArrayValue(arrowArray buddy_array, int64_t index)
	{
		auto	array = std::dynamic_pointer_cast<ARRAY_TYPE>(buddy_array);

		return array->Value(index);
	}
};

/* Binary and LargeBinary field */
template <typename BUILDER_TYPE,
          typename ARRAY_TYPE,
          typename OFFSET_TYPE>
class __arrowFileWriterBinaryColumn : public arrowFileWriterVariableColumn<BUILDER_TYPE,
																		   ARRAY_TYPE,
																		   std::string_view,
																		   OFFSET_TYPE>
{
public:
	__arrowFileWriterBinaryColumn(const char *__field_name,
								  bool __stats_enabled,
								  arrow::Compression::type __compression)
		: arrowFileWriterVariableColumn<BUILDER_TYPE,
										ARRAY_TYPE,
										std::string_view,
										OFFSET_TYPE>(sizeof(OFFSET_TYPE) == sizeof(int32_t)
													 ? arrow::binary()
													 : arrow::large_binary(),
													 __field_name,
													 __stats_enabled,
													 __compression)
	{}
};
using arrowFileWriterColumnBinary = __arrowFileWriterBinaryColumn<arrow::BinaryBuilder,
																  arrow::BinaryArray,
																  int32_t>;
using arrowFileWriterColumnLargeBinary = __arrowFileWriterBinaryColumn<arrow::LargeBinaryBuilder,
																	   arrow::LargeBinaryArray,
																	   int64_t>;
/* String and LargeString field */
template <typename BUILDER_TYPE,
		  typename ARRAY_TYPE,
		  typename OFFSET_TYPE>
class __arrowFileWriterUtf8Column : public arrowFileWriterVariableColumn<BUILDER_TYPE,
																		 ARRAY_TYPE,
																		 std::string_view,
																		 OFFSET_TYPE>
{
	static constexpr size_t STATS_VALUE_LEN = 60;
	bool	stats_is_valid;
	char	stats_min_value[STATS_VALUE_LEN+1];
	char	stats_max_value[STATS_VALUE_LEN+1];
	int		stats_min_len;
	int		stats_max_len;
public:
	__arrowFileWriterUtf8Column(const char *__field_name,
								bool __stats_enabled,
								arrow::Compression::type __compression)
		: arrowFileWriterVariableColumn<BUILDER_TYPE,
										ARRAY_TYPE,
										std::string_view,
										OFFSET_TYPE>(sizeof(OFFSET_TYPE) == sizeof(int32_t)
													 ? arrow::utf8()
													 : arrow::large_utf8(),
													 __field_name,
													 __stats_enabled,
													 __compression)
	{
		stats_is_valid = false;
	}
	virtual void
	appendStats(arrowMetadata custom_metadata)
	{
		/* NOTE: cannot contain ',' to be used for delimiter */
		if (stats_is_valid)
		{
			char   *pos;

			pos = strchr(stats_min_value, ',');
			if (pos)
				*pos = '\0';
			pos = strchr(stats_max_value, ',');
			if (pos)
			{
				*pos++ = (',' + 1);
				*pos = '\0';
			}
			custom_metadata->Append(std::string("min_max_stats.") + this->field_name,
									std::string(stats_min_value) +
									std::string(",") +
									std::string(stats_max_value));
		}
	}
	virtual void
	updateStats(std::string_view &str)
	{
		int		__len, status;

		if (!stats_is_valid)
		{
			__len = Min(str.size(), STATS_VALUE_LEN);
			memcpy(stats_min_value, str.data(), __len);
			memcpy(stats_max_value, str.data(), __len);
			stats_min_value[__len] = '\0';
			stats_max_value[__len] = '\0';
			stats_min_len = __len;
			stats_max_len = __len;
			stats_is_valid = true;
		}
		else
		{
			__len = Min(str.size(), stats_min_len);
			status = memcmp(str.data(), stats_min_value, __len);
			if (status < 0 || (status == 0 && stats_min_len > str.size()))
			{
				__len = Min(str.size(), STATS_VALUE_LEN);
				memcpy(stats_min_value, str.data(), __len);
				stats_min_value[__len] = '\0';
			}
			__len = Min(str.size(), stats_max_len);
			status = memcmp(str.data(), stats_max_value, __len);
			if (status > 0 || (status == 0 && stats_max_len < str.size()))
			{
				__len = Min(str.size(), STATS_VALUE_LEN);
				memcpy(stats_max_value, str.data(), __len);
				stats_max_value[__len] = '\0';
			}
		}
	}
};
using arrowFileWriterColumnUtf8 = __arrowFileWriterUtf8Column<arrow::StringBuilder,
															  arrow::StringArray,
															  int32_t>;
using arrowFileWriterColumnLargeUtf8 = __arrowFileWriterUtf8Column<arrow::LargeStringBuilder,
																   arrow::LargeStringArray,
																   int64_t>;
/* FixedSizeBinary fields */
template <typename BUILDER_TYPE,
          typename ARRAY_TYPE>
class __arrowFileWriterFixedSizeBinaryColumn : public arrowFileWriterScalarColumn<BUILDER_TYPE,
																				  ARRAY_TYPE,
																				  std::string_view>
{
	uint32_t	unitsz;
public:
	__arrowFileWriterFixedSizeBinaryColumn(const char *__field_name,
										   uint32_t __unitsz,
										   bool __stats_enabled,
										   arrow::Compression::type __compression)
		: arrowFileWriterScalarColumn<BUILDER_TYPE,
									  ARRAY_TYPE,
									  std::string_view>(arrow::fixed_size_binary(__unitsz),
														__field_name,
														__stats_enabled,
														__compression)
	{
		unitsz = __unitsz;
	}
	std::string_view
	fetchValue(const void *ptr, size_t sz)
	{
		return std::string_view((const char *)ptr, unitsz);
	}
    std::string_view
	fetchArrayValue(arrowArray buddy_array, int64_t index)
	{
		auto	array = std::dynamic_pointer_cast<ARRAY_TYPE>(buddy_array);

		return std::string_view((const char *)array->Value(index), unitsz);
	}
	size_t
	chunkSize(void)
	{
		auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(this->arrow_builder);

		return (ARROW_ALIGN(builder->null_count() > 0 ? BITMAPLEN(builder->length()) : 0) +
				ARROW_ALIGN(unitsz * builder->length()));
	}
};
using arrowFileWriterColumnFixedSizeBinary = __arrowFileWriterFixedSizeBinaryColumn<arrow::FixedSizeBinaryBuilder,
																					arrow::FixedSizeBinaryArray>;
/*
 * List field type
 */
template <typename BUILDER_TYPE,
		  typename ARRAY_TYPE,
		  typename OFFSET_TYPE>
class __arrowFileWriterListColumn : public arrowFileWriterColumn
{
public:
	__arrowFileWriterListColumn(const char *__field_name,
								std::shared_ptr<arrowFileWriterColumn> element,
								bool __stats_enabled,
								arrow::Compression::type __compression)
		: arrowFileWriterColumn(sizeof(OFFSET_TYPE) == sizeof(int32_t)
								? arrow::list(element->arrow_type)
								: arrow::large_list(element->arrow_type),
								__field_name,
								__stats_enabled,
								__compression)
	{
		arrow_builder = std::make_shared<BUILDER_TYPE>(arrow::default_memory_pool(),
													   element->arrow_type,
													   this->arrow_type);
		children.push_back(element);
	}
	size_t	chunkSize(void)
	{
		auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(arrow_builder);
		size_t	length = 0;

		assert(children.size() == 1);
		if (builder->null_count() > 0)
			length = ARROW_ALIGN(BITMAPLEN(builder->length()));
		return length + children[0]->chunkSize();
	}
	size_t	putValue(const void *ptr, size_t sz)
	{
		auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(arrow_builder);
		arrow::Status rv;

		assert(children.size() == 1);
		if (!ptr)
		{
			rv = builder->AppendNull();
			if (!rv.ok())
				Elog("unable to put NULL to '%s' field (%s): %s",
					 field_name.c_str(),
					 arrow_type->name().c_str(),
					 rv.ToString().c_str());
		}
		else
		{
			auto	element = children[0];
			char   *temp = (char *)alloca(sz + 1);
			char   *tok, *pos;
			int		index = 0;

			rv = builder->Append();
			if (!rv.ok())
				Elog("unable to put value to '%s' field (%s): %s",
					 field_name.c_str(),
					 arrow_type->name().c_str(),
					 rv.ToString().c_str());
			memcpy(temp, ptr, sz);
			temp[sz] = '\0';
			for (tok = __strtok_quotable(temp, ',', &pos);
				 index < children.size();
				 tok = __strtok_quotable(NULL, ',', &pos))
			{
				element->putValue(tok, strlen(tok));
			}
		}
		return chunkSize();
	}
	size_t	moveValue(std::shared_ptr<class arrowFileWriterColumn> buddy, int64_t index)
	{
		auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(arrow_builder);
		auto	buddy_array = std::dynamic_pointer_cast<ARRAY_TYPE>(buddy->arrow_array);
		arrow::Status rv;

		assert(children.size() == buddy->children.size());
		if (buddy_array->IsNull(index))
			rv = builder->AppendNull();
		else
		{
			rv = builder->Append();
			if (rv.ok())
			{
				OFFSET_TYPE	head = buddy_array->value_offset(index);
				OFFSET_TYPE	tail = buddy_array->value_offset(index+1);
				auto		elem_dst = children[0];
				auto		elem_src = buddy->children[0];

				for (OFFSET_TYPE curr=head; curr < tail; curr++)
				{
					elem_dst->moveValue(elem_src, curr);
				}
			}
		}
		if (!rv.ok())
			Elog("unable to move value to '%s' field (%s): %s",
				 field_name.c_str(),
				 arrow_type->name().c_str(),
				 rv.ToString().c_str());
		return chunkSize();
	}
	void	Reset(void)
	{
		assert(children.size() == 1);
		children[0]->Reset();
		arrowFileWriterColumn::Reset();
	}
	void	Finish(void)
	{
		assert(children.size() == 1);
		children[0]->Finish();
		arrowFileWriterColumn::Finish();
	}
};
using arrowFileWriterColumnList = __arrowFileWriterListColumn<arrow::ListBuilder,
															  arrow::ListArray,
															  int32_t>;
using arrowFileWriterColumnLargeList = __arrowFileWriterListColumn<arrow::LargeListBuilder,
																   arrow::LargeListArray,
																   int64_t>;
/*
 * Composite field type
 */
class arrowFileWriterColumnStruct : public arrowFileWriterColumn
{
	static std::shared_ptr<arrow::DataType>
	__build_composite_arrow_type(std::vector<std::shared_ptr<arrowFileWriterColumn>> &__children)
	{
		arrow::FieldVector children_fields;

		for (int i=0; i < __children.size(); i++)
		{
			auto	child = __children[i];
			auto	field = arrow::field(child->field_name,
										 child->arrow_type,
										 true,	/* nullable */
										 child->field_metadata);
			children_fields.push_back(field);
		}
		return struct_(children_fields);
	}
public:
	arrowFileWriterColumnStruct(const char *__field_name,
								std::vector<std::shared_ptr<arrowFileWriterColumn>> &__children,
								bool __stats_enabled,
								arrow::Compression::type __compression)
		: arrowFileWriterColumn(__build_composite_arrow_type(__children),
								__field_name,
								__stats_enabled,
								__compression)
	{
		std::vector<arrowBuilder>	children_builders;

		for (int i=0; i < __children.size(); i++)
		{
			auto	__child = __children[i];

			children_builders.push_back(__child->arrow_builder);
			children.push_back(__child);
		}
		arrow_builder = std::make_shared<arrow::StructBuilder>(arrow_type,
															   arrow::default_memory_pool(),
															   children_builders);
	}
	size_t	chunkSize(void)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::StructBuilder>(arrow_builder);
		size_t	length = 0;

		if (builder->null_count() > 0)
			length = ARROW_ALIGN(BITMAPLEN(builder->length()));
		for (int i=0; i < children.size(); i++)
		{
			auto	child = children[i];

			length += child->chunkSize();
		}
		return length;
	}
	size_t	putValue(const void *ptr, size_t sz)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::StructBuilder>(arrow_builder);
		arrow::Status rv;

		if (!ptr)
			rv = builder->AppendNull();
		else
		{
			rv = builder->Append();
			if (rv.ok())
			{
				char   *temp = (char *)alloca(sz + 1);
				char   *tok, *pos;
				int		index = 0;

				memcpy(temp, ptr, sz);
				temp[sz] = '\0';
				for (tok = __strtok_quotable(temp, ',', &pos);
					 index < children.size();
					 tok = __strtok_quotable(NULL, ',', &pos))
				{
					children[index++]->putValue(tok, strlen(tok));
				}
			}
		}
		if (!rv.ok())
			Elog("unable to put value to '%s' field (Struct): %s",
				 field_name.c_str(),
				 rv.ToString().c_str());
		return chunkSize();
	}
	size_t	moveValue(std::shared_ptr<class arrowFileWriterColumn> buddy, int64_t index)
	{
		auto	builder = std::dynamic_pointer_cast<arrow::StructBuilder>(arrow_builder);
		auto	buddy_array = std::dynamic_pointer_cast<arrow::StructArray>(buddy->arrow_array);
		size_t	length = 0;
		arrow::Status rv;

		assert(children.size() == buddy->children.size());
		if (buddy_array->IsNull(index))
		{
			for (int i=0; i < children.size(); i++)
				length += children[i]->putValue(NULL, 0);
			rv = builder->AppendNull();
		}
		else
		{
			rv = builder->Append();
			if (rv.ok())
			{
				for (int i=0; i < children.size(); i++)
				{
					auto	child = children[i];
					auto	buddy_child = buddy->children[i];

					child->moveValue(buddy_child, index);
				}
			}
		}
		if (!rv.ok())
			Elog("unable to move value to '%s' field (Struct): %s",
				 field_name.c_str(),
				 rv.ToString().c_str());
		return chunkSize();
	}
	void	Reset(void)
	{
		for (int i=0; i < children.size(); i++)
			children[i]->Reset();
		arrowFileWriterColumn::Reset();
	}
	void	Finish(void)
	{
		for (int i=0; i < children.size(); i++)
			children[i]->Finish();
		arrowFileWriterColumn::Finish();
	}
};


	void		ParseSchemaDefs(const char *schema_defs);



#endif	/* _ARROW_WRITE_H_ */
