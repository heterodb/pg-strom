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

static std::string
__quote_ident(const char *ident)
{
	std::string	buf;

	for (const char *pos = ident; *pos != '\0'; pos++)
	{
		if (islower(*pos) || isdigit(*pos) || *pos == '_')
			buf += *pos;
		else
			goto do_quote;
	}
	return buf;
do_quote:
	buf.clear();
	buf += '"';
	for (const char *pos = ident; *pos != '\0'; pos++)
	{
		if (*pos == '"')
			buf += '"';
		buf += *pos;
	}
	buf += '"';
	return buf;
}

/*
 * Definition of ArrowFileBuilderTable
 */
using	arrowSchema		= std::shared_ptr<arrow::Schema>;
using	arrowField		= std::shared_ptr<arrow::Field>;
using	arrowBuilder	= std::shared_ptr<arrow::ArrayBuilder>;
using	arrowArray		= std::shared_ptr<arrow::Array>;
using	arrowRecordBatch = std::shared_ptr<arrow::RecordBatch>;
using	arrowMetadata	= std::shared_ptr<arrow::KeyValueMetadata>;
using	arrowFileBuilderTable  = std::shared_ptr<class ArrowFileBuilderTable>;
using	arrowFileBuilderColumn = std::shared_ptr<class ArrowFileBuilderColumn>;

class ArrowFileBuilderTable
{
	arrowRecordBatch	__buildRecordBatch();
	arrowMetadata		__buildMinMaxStats();
	std::shared_ptr<parquet::WriterProperties>		__parquetWriterProperties();
	std::shared_ptr<parquet::ArrowWriterProperties>	__parquetArrowProperties();
protected:
	std::shared_ptr<arrow::io::FileOutputStream>	file_out_stream;
	std::shared_ptr<arrow::ipc::RecordBatchWriter>	arrow_file_writer;
	std::unique_ptr<parquet::arrow::FileWriter>		parquet_file_writer;
	bool			parquet_mode;
public:
	std::string		appname;	/* application name (optional) */
	std::string		pathname;
	arrow::Compression::type default_compression;
	arrowSchema		arrow_schema;
	arrowMetadata	table_metadata;
	std::vector<arrowFileBuilderColumn> arrow_columns;
	std::unordered_map<std::string, arrowFileBuilderColumn> columns_htable;
	/* --- methods --- */
	ArrowFileBuilderTable(bool __parquet_mode,
						  arrow::Compression::type __compression)
	{
		file_out_stream = nullptr;
		arrow_file_writer = nullptr;
		parquet_file_writer = nullptr;
		parquet_mode = __parquet_mode;
		default_compression = __compression;
		arrow_schema = nullptr;
		table_metadata = std::make_shared<arrow::KeyValueMetadata>();
	}
	bool		Open(const char *__pathname, bool allow_overwrite = false);
	void		Close();
	void		AssignSchema(void);
	void		WriteChunk(void);
	bool		checkCompatibility(arrowFileBuilderTable buddy);
	std::string PrintSchema(const char *ftable_name, const char *arrow_filename);
};

static inline arrowFileBuilderTable
makeArrowFileBuilderTable(bool __parquet_mode = false,
						  arrow::Compression::type __compression = arrow::Compression::type::ZSTD)
{
	return std::make_shared<ArrowFileBuilderTable>(__parquet_mode,
												   __compression);
}

/*
 * ArrowFileBuilderColumn
 *
 * Base class of individual fields.
 */
class ArrowFileBuilderColumn
{
	friend class ArrowFileBuilderTable;
public:
	std::shared_ptr<arrow::DataType> arrow_type;
	std::string		sql_type_name;			/* just for display */
	std::string		field_name;
	int				field_index;
	bool			stats_enabled;
	arrow::Compression::type compression;	/* valid only if parquet-mode */
	arrowBuilder	arrow_builder;
	arrowArray		arrow_array;
	std::vector<std::shared_ptr<class ArrowFileBuilderColumn>> children;	/* only Struct/List */
	arrowMetadata	field_metadata;
	/* ---- vcf2arrow specific fields ---- */
	char			vcf_allele_policy;
	/* ---- methods ---- */
	ArrowFileBuilderColumn(std::shared_ptr<arrow::DataType> __arrow_type,
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
		/* for vcf2arrow */
		vcf_allele_policy = ' ';
	}
	virtual void
	appendStats(arrowMetadata custom_metadata)
	{
		/* do nothing if no statistics support */
	}
	virtual size_t	chunkSize() = 0;
	virtual size_t	putValue(const void *ptr, size_t sz) = 0;
	virtual size_t	moveValue(arrowFileBuilderColumn buddy, int64_t index) = 0;
	virtual void	Reset(void)
	{
		arrow_builder->Reset();
		arrow_array = nullptr;
	}
	virtual size_t	Finish(void)
	{
		auto	rv = arrow_builder->Finish(&arrow_array);
		if (!rv.ok())
			Elog("failed on arrow::ArrayBuilder::Finish: %s",
				 rv.ToString().c_str());
		return arrow_array->length();
	}
	virtual bool	checkCompatibility(arrowFileBuilderColumn buddy)
	{
		if (field_name == buddy->field_name &&
			stats_enabled == buddy->stats_enabled &&
			compression == buddy->compression &&
			arrow_type->Equals(buddy->arrow_type) &&
			vcf_allele_policy == buddy->vcf_allele_policy)
			return true;
		return false;
	}
	virtual void	printSchema(std::string &sql)
	{
		size_t	len = sql.length();
		sql += __quote_ident(field_name.c_str());
		len = sql.length() - len;
		if (len < 32)
			sql += std::string(32-len, ' ');
		else
			sql += " ";
		sql += sql_type_name;
	}
protected:
	virtual void	__parquetWriterProperties(parquet::WriterProperties::Builder &builder,
											  arrow::Compression::type default_compression)
	{
		// if column level compression is different from default_compression,
		// set its own compression
		for (int j=0; j < children.size(); j++)
			children[j]->__parquetWriterProperties(builder, default_compression);
	}
};

/*
 * Field for a basic scalar type
 */
template <typename BUILDER_TYPE,
		  typename ARRAY_TYPE,
          typename CPP_TYPE>
class ArrowFileBuilderScalarColumn : public ArrowFileBuilderColumn
{
public:
	ArrowFileBuilderScalarColumn(std::shared_ptr<arrow::DataType> __arrow_type,
								 const char *__field_name,
								 bool __stats_enabled,
								 arrow::Compression::type __compression)
		: ArrowFileBuilderColumn(__arrow_type,
								__field_name,
								__stats_enabled,
								__compression)
	{
		arrow_builder = std::make_shared<BUILDER_TYPE>(__arrow_type, arrow::default_memory_pool());
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
	moveValue(arrowFileBuilderColumn buddy, int64_t index)
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
class ArrowFileBuilderSimpleColumn : public ArrowFileBuilderScalarColumn<BUILDER_TYPE,
																		 ARRAY_TYPE,
																		 CPP_TYPE>
{
	bool		stats_is_valid;
	CPP_TYPE	stats_min_value;
	CPP_TYPE	stats_max_value;
public:
	ArrowFileBuilderSimpleColumn(std::shared_ptr<arrow::DataType> __arrow_type,
								 const char *__field_name,
								 bool __stats_enabled,
								 arrow::Compression::type __compression)
		: ArrowFileBuilderScalarColumn<BUILDER_TYPE,
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
		ArrowFileBuilderColumn::Reset();
		stats_is_valid = false;
	}
};

#define __ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(NAME,PREFIX,CPP_TYPE,TYPE_OBJ,SQL_TYPE) \
	class ArrowFileBuilderColumn##NAME									\
		: public ArrowFileBuilderSimpleColumn<arrow::PREFIX##Builder,	\
											  arrow::PREFIX##Array,		\
											  CPP_TYPE>					\
	{																	\
	public:																\
		ArrowFileBuilderColumn##NAME(const char *__field_name,			\
									 bool __stats_enabled,				\
									 arrow::Compression::type __compression) \
			: ArrowFileBuilderSimpleColumn<arrow::PREFIX##Builder,		\
										   arrow::PREFIX##Array,		\
										   CPP_TYPE>(arrow::TYPE_OBJ,	\
													 __field_name,		\
													 __stats_enabled,	\
													 __compression)		\
		{																\
			sql_type_name = std::string(#SQL_TYPE);						\
		}																\
		CPP_TYPE	fetchValue(const void *ptr, size_t sz);				\
	};
#define ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(NAME,PREFIX,CPP_TYPE,TYPE_OBJ,SQL_TYPE) \
	__ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(NAME,PREFIX,CPP_TYPE,TYPE_OBJ,SQL_TYPE) \
	CPP_TYPE															\
	ArrowFileBuilderColumn##NAME::fetchValue(const void *ptr, size_t sz)	\
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
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Boolean,       Boolean,   bool,     boolean(), bool)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Int8,          Int8,      int8_t,   int8(),    int1)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Int16,         Int16,     int16_t,  int16(),   int2)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Int32,         Int32,     int32_t,  int32(),   int4)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Int64,         Int64,     int64_t,  int64(),   int8)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(UInt8,         UInt8,     uint8_t,  uint8(),   int1)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(UInt16,        UInt16,    uint16_t, uint16(),  int2)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(UInt32,        UInt32,    uint32_t, uint32(),  int4)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(UInt64,        UInt64,    uint64_t, uint64(),  int8)
__ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Float16,     HalfFloat, uint16_t, float16(), float2)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Float32,       Float,     float,    float32(), float4)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Float64,       Double,    double,   float64(), float8)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Date_sec,      Date32,    int32_t,  date32(),  date)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Date_ms,       Date64,    int64_t,  date64(),  date)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Time_sec,      Time32,    int32_t,  time32(arrow::TimeUnit::SECOND), time)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Time_ms,       Time32,    int32_t,  time32(arrow::TimeUnit::MILLI),  time)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Time_us,       Time64,    int64_t,  time64(arrow::TimeUnit::MICRO),  time)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Time_ns,       Time64,    int64_t,  time64(arrow::TimeUnit::NANO),   time)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Timestamp_sec, Timestamp, int64_t,  timestamp(arrow::TimeUnit::SECOND), timestamp)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Timestamp_ms,  Timestamp, int64_t,  timestamp(arrow::TimeUnit::MILLI),  timestamp)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Timestamp_us,  Timestamp, int64_t,  timestamp(arrow::TimeUnit::MICRO),  timestamp)
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Timestamp_ns,  Timestamp, int64_t,  timestamp(arrow::TimeUnit::NANO),   timestamp)

/* special handling for float16 */
uint16_t
ArrowFileBuilderColumnFloat16::fetchValue(const void *ptr, size_t sz)
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
class __arrowFileBuilderDecimalColumn
	: public ArrowFileBuilderScalarColumn<BUILDER_TYPE,
										  ARRAY_TYPE,
										  CPP_TYPE>
{
	bool		stats_is_valid;
	CPP_TYPE	stats_min_value;
	CPP_TYPE	stats_max_value;
	static std::shared_ptr<arrow::DataType>
	__decimal_arrow_type(int32_t __precision, int32_t __scale)
	{
		return (sizeof(CPP_TYPE) == 4  ? arrow::decimal32(__precision, __scale) :
				sizeof(CPP_TYPE) == 8  ? arrow::decimal64(__precision, __scale) :
				sizeof(CPP_TYPE) == 16 ? arrow::decimal128(__precision, __scale) :
				sizeof(CPP_TYPE) == 32 ? arrow::decimal256(__precision, __scale) : nullptr);
	}
public:
	__arrowFileBuilderDecimalColumn(const char *__field_name,
									int32_t __precision,
									int32_t __scale,
									bool __stats_enabled,
									arrow::Compression::type __compression)
		: ArrowFileBuilderScalarColumn<BUILDER_TYPE,
									   ARRAY_TYPE,
									   CPP_TYPE>(__decimal_arrow_type(__precision,
																	  __scale),
												 __field_name,
												 __stats_enabled,
												 __compression)
	{
		char	namebuf[80];

		stats_is_valid = false;
		this->sql_type_name = (std::string("numeric(") +
							   std::to_string(__precision) +
							   std::string(",") +
							   std::to_string(__scale) +
							   std::string(")"));
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
		ArrowFileBuilderColumn::Reset();
		stats_is_valid = false;
	}
};
using	ArrowFileBuilderColumnDecimal128 = __arrowFileBuilderDecimalColumn<arrow::Decimal128Builder,
																		   arrow::Decimal128Array,
																		   arrow::Decimal128Type,
																		   arrow::Decimal128>;
/*
 * Interval field type
 */
template <typename BUILDER_TYPE,
		  typename ARRAY_TYPE,
		  typename ARROW_TYPE,
		  typename CPP_TYPE,
		  arrow::Type::type ARROW_TYPE_ID>
class __ArrowFileBuilderIntervalColumn
	: public ArrowFileBuilderScalarColumn<BUILDER_TYPE,
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
	__ArrowFileBuilderIntervalColumn(const char *__field_name,
									 bool __stats_enabled,
									 arrow::Compression::type __compression)
		: ArrowFileBuilderScalarColumn<BUILDER_TYPE,
									   ARRAY_TYPE,
									   CPP_TYPE>(__build_interval_arrow_type(),
												 __field_name,
												 __stats_enabled,
												 __compression)
	{
		this->sql_type_name = std::string("interval");
	}
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
using ArrowFileBuilderIntervalMonth
	= __ArrowFileBuilderIntervalColumn<arrow::MonthIntervalBuilder,
									  arrow::MonthIntervalArray,
									  arrow::MonthIntervalType,
									  arrow::MonthIntervalType,
									  arrow::Type::INTERVAL_MONTHS>;
using ArrowFileBuilderIntervalDayTime
	= __ArrowFileBuilderIntervalColumn<arrow::DayTimeIntervalBuilder,
									  arrow::DayTimeIntervalArray,
									  arrow::DayTimeIntervalType,
									  arrow::DayTimeIntervalType,
									  arrow::Type::INTERVAL_DAY_TIME>;
using ArrowFileBuilderIntervalMonthDayNano
	= __ArrowFileBuilderIntervalColumn<arrow::MonthDayNanoIntervalBuilder,
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
class ArrowFileBuilderVariableColumn
	: public ArrowFileBuilderScalarColumn<BUILDER_TYPE,
										  ARRAY_TYPE,
										  CPP_TYPE>
{
public:
	ArrowFileBuilderVariableColumn(std::shared_ptr<arrow::DataType> __arrow_type,
								   const char *__field_name,
								   bool __stats_enabled,
								   arrow::Compression::type __compression)
		: ArrowFileBuilderScalarColumn<BUILDER_TYPE,
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
class __ArrowFileBuilderBinaryColumn
	: public ArrowFileBuilderVariableColumn<BUILDER_TYPE,
											ARRAY_TYPE,
											std::string_view,
											OFFSET_TYPE>
{
public:
	__ArrowFileBuilderBinaryColumn(const char *__field_name,
								   bool __stats_enabled,
								   arrow::Compression::type __compression)
		: ArrowFileBuilderVariableColumn<BUILDER_TYPE,
										 ARRAY_TYPE,
										 std::string_view,
										 OFFSET_TYPE>(sizeof(OFFSET_TYPE) == sizeof(int32_t)
													  ? arrow::binary()
													  : arrow::large_binary(),
													  __field_name,
													  __stats_enabled,
													  __compression)
	{
		this->sql_type_name = std::string("bytea");
	}
};
using ArrowFileBuilderColumnBinary
	= __ArrowFileBuilderBinaryColumn<arrow::BinaryBuilder,
									 arrow::BinaryArray,
									 int32_t>;
using ArrowFileBuilderColumnLargeBinary
	= __ArrowFileBuilderBinaryColumn<arrow::LargeBinaryBuilder,
									 arrow::LargeBinaryArray,
									 int64_t>;
/* String and LargeString field */
template <typename BUILDER_TYPE,
		  typename ARRAY_TYPE,
		  typename OFFSET_TYPE>
class __ArrowFileBuilderUtf8Column
	: public ArrowFileBuilderVariableColumn<BUILDER_TYPE,
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
	__ArrowFileBuilderUtf8Column(const char *__field_name,
								 bool __stats_enabled,
								 arrow::Compression::type __compression)
		: ArrowFileBuilderVariableColumn<BUILDER_TYPE,
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
		this->sql_type_name = std::string("text");
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
using ArrowFileBuilderColumnUtf8 = __ArrowFileBuilderUtf8Column<arrow::StringBuilder,
															   arrow::StringArray,
															  int32_t>;
using ArrowFileBuilderColumnLargeUtf8 = __ArrowFileBuilderUtf8Column<arrow::LargeStringBuilder,
																	 arrow::LargeStringArray,
																	 int64_t>;
/* FixedSizeBinary fields */
template <typename BUILDER_TYPE,
          typename ARRAY_TYPE>
class __ArrowFileBuilderFixedSizeBinaryColumn
	: public ArrowFileBuilderScalarColumn<BUILDER_TYPE,
										  ARRAY_TYPE,
										  std::string_view>
{
	uint32_t	unitsz;
public:
	__ArrowFileBuilderFixedSizeBinaryColumn(const char *__field_name,
											uint32_t __unitsz,
											bool __stats_enabled,
											arrow::Compression::type __compression)
		: ArrowFileBuilderScalarColumn<BUILDER_TYPE,
									   ARRAY_TYPE,
									   std::string_view>(arrow::fixed_size_binary(__unitsz),
														 __field_name,
														 __stats_enabled,
														 __compression)
	{
		unitsz = __unitsz;
		this->sql_type_name = (std::string("char(") +
							   std::to_string(__unitsz) +
							   std::string(")"));
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
using ArrowFileBuilderColumnFixedSizeBinary
	= __ArrowFileBuilderFixedSizeBinaryColumn<arrow::FixedSizeBinaryBuilder,
											  arrow::FixedSizeBinaryArray>;
/*
 * List field type
 */
template <typename BUILDER_TYPE,
		  typename ARRAY_TYPE,
		  typename OFFSET_TYPE>
class __ArrowFileBuilderListColumn : public ArrowFileBuilderColumn
{
public:
	__ArrowFileBuilderListColumn(const char *__field_name,
								 std::shared_ptr<ArrowFileBuilderColumn> element,
								 bool __stats_enabled,
								 arrow::Compression::type __compression)
		: ArrowFileBuilderColumn(sizeof(OFFSET_TYPE) == sizeof(int32_t)
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
		this->sql_type_name = element->sql_type_name + std::string("[]");
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
	size_t	moveValue(std::shared_ptr<class ArrowFileBuilderColumn> buddy, int64_t index)
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
		ArrowFileBuilderColumn::Reset();
	}
	size_t	Finish(void)
	{
		size_t	nrows = ArrowFileBuilderColumn::Finish();

		assert(children.size() == 1);
		if (children[0]->Finish() != nrows)
			Elog("Bug? element of '%s' has inconsist number of items",
				 field_name.c_str());
		return nrows;
	}
	bool    checkCompatibility(arrowFileBuilderColumn buddy)
	{
		assert(children.size() == 1);
		if (ArrowFileBuilderColumn::checkCompatibility(buddy) &&
			buddy->children.size() == 1 &&
			children[0]->checkCompatibility(buddy->children[0]))
			return true;
		return false;
	}
	virtual void	printSchema(std::string &sql)
	{
		assert(children.size() == 1);
		children[0]->printSchema(sql);
		sql += std::string("[]");
	}
};
using ArrowFileBuilderColumnList = __ArrowFileBuilderListColumn<arrow::ListBuilder,
																arrow::ListArray,
																int32_t>;
using ArrowFileBuilderColumnLargeList = __ArrowFileBuilderListColumn<arrow::LargeListBuilder,
																	 arrow::LargeListArray,
																	 int64_t>;
/*
 * Composite field type
 */
class ArrowFileBuilderColumnStruct : public ArrowFileBuilderColumn
{
	static std::shared_ptr<arrow::DataType>
	__build_composite_arrow_type(std::vector<std::shared_ptr<ArrowFileBuilderColumn>> &__children)
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
	ArrowFileBuilderColumnStruct(const char *__field_name,
								std::vector<std::shared_ptr<ArrowFileBuilderColumn>> &__children,
								bool __stats_enabled,
								arrow::Compression::type __compression)
		: ArrowFileBuilderColumn(__build_composite_arrow_type(__children),
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
		sql_type_name = (std::string("comp_") +
						 std::string(__field_name) +
						 std::string("_t"));
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
	size_t	moveValue(std::shared_ptr<class ArrowFileBuilderColumn> buddy, int64_t index)
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
		ArrowFileBuilderColumn::Reset();
	}
	size_t	Finish(void)
	{
		size_t	nrows = ArrowFileBuilderColumn::Finish();

		for (int i=0; i < children.size(); i++)
		{
			if (children[i]->Finish() != nrows)
				Elog("Bug? children of '%s' has inconsist number of items",
					 field_name.c_str());
		}
		return nrows;
	}
	bool    checkCompatibility(arrowFileBuilderColumn buddy)
	{
		if (ArrowFileBuilderColumn::checkCompatibility(buddy) &&
			children.size() == buddy->children.size())
		{
			for (int i=0; i < children.size(); i++)
			{
				if (!children[i]->checkCompatibility(buddy->children[i]))
					return false;
			}
			return true;
		}
		return false;
	}
	void	printSchema(std::string &sql)
	{
		std::string	decl;

		decl += std::string("CREATE TYPE comp_") + field_name + std::string("_t AS (\n");
		for (int i=0; i < children.size(); i++)
		{
			auto	child = children[i];

			if (i > 0)
				decl += std::string(",\n");
			decl += std::string("    ");
			child->printSchema(decl);
		}
		decl += std::string("\n);\n\n");
		decl += sql;
		sql = decl;
		ArrowFileBuilderColumn::printSchema(sql);
	}
};

/*
 * ArrowFileBuilderTable methods
 */
bool
ArrowFileBuilderTable::Open(const char *__pathname, bool allow_overwrite)
{
	int		fdesc;
	int		flags = O_RDWR | O_CREAT | O_TRUNC;

	if (!allow_overwrite)
		flags |= O_EXCL;
	fdesc = open(__pathname, flags, 0644);
	if (fdesc >= 0)
	{
		auto	rv = arrow::io::FileOutputStream::Open(fdesc);

		if (!rv.ok())
			Elog("failed on arrow::io::FileOutputStream::Open('%s'): %s",
				 __pathname,
				 rv.status().ToString().c_str());
		file_out_stream = rv.ValueOrDie();
		pathname = std::string(__pathname);
		return true;
	}
	return false;
}

void
ArrowFileBuilderTable::Close(void)
{
	arrow::Status	rv;
	int64_t			nrows = -1;

	/* flush if remaining record exist */
	for (int i=0; i < arrow_columns.size(); i++)
	{
		auto	column = arrow_columns[i];
		auto	builder = column->arrow_builder;

		if (nrows < 0)
			nrows = builder->length();
		else if (nrows != builder->length())
			Elog("Bug? number of rows in arrow_builder is not consistent");
	}
	if (nrows > 0)
		WriteChunk();
	/* write out arrow footer */
	if (arrow_file_writer)
	{
		rv = arrow_file_writer->Close();
		if (!rv.ok())
			Elog("failed on arrow::ipc::RecordBatchWriter::Close: %s",
				 rv.ToString().c_str());
		arrow_file_writer = nullptr;
	}
	/* write out parquet footer */
	if (parquet_file_writer)
	{
		rv = parquet_file_writer->Close();
		if (!rv.ok())
			Elog("failed on parquet::arrow::FileWriter::Close: %s",
				 rv.ToString().c_str());
		parquet_file_writer = nullptr;
	}
	/* close the output stream */
	if (file_out_stream)
	{
		rv = file_out_stream->Close();
		if (!rv.ok())
			Elog("failed on arrow::ipc::io::FileOutputStream::Close: %s",
				 rv.ToString().c_str());
		file_out_stream = nullptr;
	}
}

void
ArrowFileBuilderTable::AssignSchema(void)
{
	arrow::FieldVector arrow_fields;

	if (arrow_schema)
		Elog("ArrowFileBuilderTable: schema is already assigned");
	for (int i=0; i < arrow_columns.size(); i++)
	{
		auto	column = arrow_columns[i];
		auto	field = arrow::field(column->field_name,
									 column->arrow_type,
									 true,
									 column->field_metadata);
		arrow_fields.push_back(field);

		column->field_index = i;
		if (columns_htable.count(column->field_name) != 0)
			Elog("multiple field '%s' exist", column->field_name.c_str());
		columns_htable[column->field_name] = column;
	}
	arrow_schema = arrow::schema(arrow_fields);
}

std::string
ArrowFileBuilderTable::PrintSchema(const char *ftable_name,
								   const char *arrow_filename)
{
	std::string	sql;

	sql += "CREATE FOREIGN TABLE ";
	sql += __quote_ident(ftable_name);
	sql += "\n(\n";
	for (int i=0; i < arrow_columns.size(); i++)
	{
		auto	column = arrow_columns[i];

		if (i > 0)
			sql += ",\n";
		sql += "    ";
		column->printSchema(sql);
	}
	sql += "\n) SERVER arrow_fdw\n"
		"  OPTIONS (file '";
	if (arrow_filename)
		sql += arrow_filename;
	else
		sql += "PATH_TO_ARROW_FILE";
	sql += "');\n\n";

	return sql;
}

arrowRecordBatch
ArrowFileBuilderTable::__buildRecordBatch(void)
{
	std::vector<arrowArray> results;
	int64_t		nrows = -1;

	for (int j=0; j < arrow_columns.size(); j++)
	{
		auto	column = arrow_columns[j];
		int64_t	__nrows = column->Finish();

		if (nrows < 0)
			nrows = __nrows;
		else if (nrows != __nrows)
			Elog("Bug? number of rows mismatch across the buffers");
		assert(column->arrow_array != NULL);
		results.push_back(column->arrow_array);
	}
	return arrow::RecordBatch::Make(arrow_schema, nrows, results);
}

arrowMetadata
ArrowFileBuilderTable::__buildMinMaxStats(void)
{
	auto	metadata = std::make_shared<arrow::KeyValueMetadata>();

	for (int j=0; j < arrow_columns.size(); j++)
	{
		auto	column = arrow_columns[j];

		if (column->stats_enabled)
			column->appendStats(metadata);
	}
	return metadata;
}

std::shared_ptr<parquet::WriterProperties>
ArrowFileBuilderTable::__parquetWriterProperties()
{
	parquet::WriterProperties::Builder builder;
	/* application name, if any */
	if (appname.size() > 0)
		builder.created_by(appname);
	/* we don't switch row-group by number of lines basis  */
	builder.max_row_group_length(INT64_MAX);
	/* default compression method */
	builder.compression(default_compression);
	for (int j=0; j < arrow_columns.size(); j++)
		arrow_columns[j]->__parquetWriterProperties(builder, default_compression);
	// MEMO: Parquet enables min/max/null-count statistics in the default,
	// so we don't need to touch something special ...(like enable_statistics())
	return builder.build();
}

std::shared_ptr<parquet::ArrowWriterProperties>
ArrowFileBuilderTable::__parquetArrowProperties()
{
	return parquet::default_arrow_writer_properties();
}

void
ArrowFileBuilderTable::WriteChunk(void)
{
	if (!file_out_stream)
		Elog("ArrowFileBuilderTable::Open() must be called before WriteChunk()");
	if (!arrow_schema)
		Elog("ArrowFileBuilderTable::AssignSchema() must be called before WriteChunk()");
	if (!parquet_mode)
	{
		/* write out arrow file */
		if (!arrow_file_writer)
		{
			auto    rv = arrow::ipc::MakeFileWriter(file_out_stream, arrow_schema);
			if (!rv.ok())
				Elog("failed on arrow::ipc::MakeFileWriter for '%s': %s",
					 pathname.c_str(),
					 rv.status().ToString().c_str());
			arrow_file_writer = rv.ValueOrDie();
		}
		/* write out record batch */
		auto	rv = arrow_file_writer->WriteRecordBatch(*__buildRecordBatch(),
														 __buildMinMaxStats());

		if (!rv.ok())
			Elog("failed on arrow::ipc::RecordBatchWriter::WriteRecordBatch: %s",
				 rv.ToString().c_str());
	}
	else
	{
		/* write out parquet file */
		if (!parquet_file_writer)
		{
			auto	rv = parquet::arrow::FileWriter::Open(*arrow_schema,
														  arrow::default_memory_pool(),
														  file_out_stream,
														  __parquetWriterProperties(),
														  __parquetArrowProperties());
			if (!rv.ok())
				Elog("failed on parquet::arrow::FileWriter::Open('%s'): %s",
					 pathname.c_str(), rv.status().ToString().c_str());
			parquet_file_writer = std::move(rv).ValueOrDie();
		}
		/* write out row-group */
		{
			auto	rv = parquet_file_writer->WriteRecordBatch(*__buildRecordBatch());
			if (!rv.ok())
				Elog("failed on parquet::arrow::FileWriter::WriteRecordBatch: %s",
					 rv.ToString().c_str());

		}
		/* flush to the disk */
		{
			auto	rv = parquet_file_writer->NewBufferedRowGroup();
			if (!rv.ok())
				Elog("failed on parquet::arrow::FileWriter::NewBufferedRowGroup: %s",
					 rv.ToString().c_str());
		}
	}
}

bool
ArrowFileBuilderTable::checkCompatibility(arrowFileBuilderTable buddy)
{
	if (parquet_mode != buddy->parquet_mode ||
		default_compression != buddy->default_compression)
		return false;
	if (arrow_columns.size() != buddy->arrow_columns.size())
		return false;
	for (int i=0; i < arrow_columns.size(); i++)
	{
		auto	column = arrow_columns[i];
		auto	buddy_column = buddy->arrow_columns[i];

		if (!column->checkCompatibility(buddy_column))
			return false;
	}
	//TODO: metadata compatibility checks
	return true;
}

#endif	/* _ARROW_WRITE_H_ */
