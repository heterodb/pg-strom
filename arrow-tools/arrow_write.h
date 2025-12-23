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
#include <parquet/arrow/writer.h>
#include <string>
#include "float2.h"

#ifndef Elog
#define Elog(fmt,...)								\
	do {											\
		fprintf(stderr, "[Error: %s:%d] " fmt "\n",	\
				__FILE__, __LINE__, ##__VA_ARGS__);	\
		_exit(1);									\
	} while(0)
#endif
#ifndef Info
#define Info(fmt,...)							\
	fprintf(stderr, "[Info: %s:%d] " fmt "\n",	\
			__FILE__, __LINE__, ##__VA_ARGS__)
#endif
#define Max(a,b)	((a)>(b) ? (a) : (b))
#define Min(a,b)	((a)<(b) ? (a) : (b))
#define BITMAPLEN(NITEMS)	(((NITEMS) + 7) / 8)
#define ARROW_ALIGN(LEN)	(((uintptr_t)(LEN) + 63UL) & ~63UL)

using	arrowSchema		= std::shared_ptr<arrow::Schema>;
using	arrowField		= std::shared_ptr<arrow::Field>;
using	arrowBuilder	= std::shared_ptr<arrow::ArrayBuilder>;
using	arrowArray		= std::shared_ptr<arrow::Array>;

class arrowFileWriter
{
	std::shared_ptr<arrow::io::FileOutputStream>	file_out_stream;
	std::shared_ptr<arrow::ipc::RecordBatchWriter>	arrow_file_writer;
	std::unique_ptr<parquet::arrow::FileWriter>		parquet_file_writer;
public:
	std::string		pathname;
	bool			parquet_mode;
	arrow::Compression::type default_compression;
	arrowSchema		arrow_schema;
	std::vector<std::shared_ptr<class arrowFileWriterColumn>> arrow_columns;
	arrowFileWriter(void)
	{
		file_out_stream = nullptr;
		arrow_file_writer = nullptr;
		parquet_file_writer = nullptr;
		parquet_mode = false;
		default_compression = arrow::Compression::type::ZSTD;
		arrow_schema = nullptr;
	}
	arrowSchema	ParseSchemaDefs(const char *schema_defs);
	bool		Open(const char *__pathname);
	void		Close();
};

/*
 * arrowFileWriterColumn - A common base class for Arrow data types
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
	uint64_t		nitems;
	uint64_t		nullcount;
	arrowBuilder	arrow_builder;
	arrowArray		arrow_array;
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
		nitems = 0;
		nullcount = 0;
	}
	virtual size_t	putValue(const void *ptr, size_t sz) = 0;
	virtual size_t	moveValue(arrowFileWriterColumn &buddy, uint64_t index) = 0;
	virtual void	Reset(void)
	{
		arrow_builder->Reset();
		arrow_array = nullptr;
		nitems = 0;
		nullcount = 0;
	}
	virtual size_t	Finish(void)
	{
		auto	rv = arrow_builder->Finish(&arrow_array);
		if (!rv.ok())
			Elog("failed on arrow::ArrayBuilder::Finish: %s",
				 rv.ToString().c_str());
		return arrow_array->length();
	}
};

/*
 * Simple scalar types
 */
template <typename BUILDER_TYPE,
		  typename ARRAY_TYPE,
		  typename CPP_TYPE>
class arrowFileWriterSimpleColumn : public arrowFileWriterColumn
{
protected:
	bool		stats_is_valid;
	CPP_TYPE	stats_min_value;
	CPP_TYPE	stats_max_value;
	virtual void
	appendStats(std::shared_ptr<arrow::KeyValueMetadata> custom_metadata)
	{
		if (!stats_is_valid)
			return;
		custom_metadata->Append(std::string("min_max_stats.") + this->field_name,
								std::to_string(stats_min_value) +
								std::string(",") +
								std::to_string(stats_max_value));
	}
	virtual void
	updateStats(CPP_TYPE datum)
	{
		if (!this->stats_enabled)
			return;
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
	size_t
	chunkSize(void)
	{
		return (ARROW_ALIGN(this->nullcount > 0 ? BITMAPLEN(this->nitems) : 0) +
				ARROW_ALIGN(sizeof(CPP_TYPE) * this->nitems));
	}
public:
	arrowFileWriterSimpleColumn(std::shared_ptr<arrow::DataType> __arrow_type,
								const char *__field_name,
								bool __stats_enabled,
								arrow::Compression::type __compression)
		: arrowFileWriterColumn(__arrow_type, __field_name, __stats_enabled, __compression)
	{
		arrow_builder = std::make_shared<BUILDER_TYPE>(__arrow_type, arrow::default_memory_pool());
		stats_is_valid = false;
	}
	virtual CPP_TYPE
	fetchValue(const void *ptr, size_t sz);
	size_t
	putValue(const void *ptr, size_t sz)
	{
		auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(arrow_builder);
		arrow::Status rv;

		if (!ptr)
		{
			rv = builder->AppendNull();
			nullcount++;
		}
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
		nitems++;
		return chunkSize();
	}
	size_t
	moveValue(arrowFileWriterColumn &buddy, uint64_t index)
	{
		auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(arrow_builder);
		auto	array = std::dynamic_pointer_cast<ARRAY_TYPE>(buddy.arrow_array);
		arrow::Status rv;

		assert(index < array->length());
		if (array->IsNull(index))
		{
			rv = builder->AppendNull();
			nullcount++;
		}
		else
		{
			auto	datum = array->Value(index);
			updateStats(datum);
			rv = builder->Append(datum);
		}
		if (!rv.ok())
			Elog("unable to move value to '%s' field (%s): %s",
				 field_name.c_str(),
				 arrow_type->name().c_str(),
				 rv.ToString().c_str());
		nitems++;
		return chunkSize();
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
__ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(HalfFloat,   HalfFloat, uint16_t, float16())
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Float,         Float,     float,    float32())
ARROW_FILE_WRITER_SIMPLE_COLUMN_TEMPLATE(Double,        Double,    double,   float64())
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
half_t
arrowFileWriterColumnHalfFloat::fetchValue(const void *ptr, size_t sz)
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
class arrowFileWriterDecimalColumn : public arrowFileWriterColumn
{
protected:
	bool		stats_is_valid;
	CPP_TYPE	stats_min_value;
	CPP_TYPE	stats_max_value;
	void
	appendStats(std::shared_ptr<arrow::KeyValueMetadata> custom_metadata)
	{
		if (stats_is_valid)
		{
			auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(arrow_builder);
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
		if (!this->stats_enabled)
			return;
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
	size_t
	chunkSize(void)
	{
		return (ARROW_ALIGN(this->nullcount > 0 ? BITMAPLEN(this->nitems) : 0) +
				ARROW_ALIGN(sizeof(CPP_TYPE) * this->nitems));
	}
public:
	arrowFileWriterDecimalColumn(std::shared_ptr<arrow::DataType> __arrow_type,
								 const char *__field_name,
								 bool __stats_enabled,
								 arrow::Compression::type __compression)
		: arrowFileWriterColumn(__arrow_type, __field_name, __stats_enabled, __compression)
	{
		arrow_builder = std::make_shared<BUILDER_TYPE>(__arrow_type, arrow::default_memory_pool());
		stats_is_valid = false;
	}
	virtual CPP_TYPE
	fetchValue(const void *ptr, size_t sz);
	size_t
	putValue(const void *ptr, size_t sz)
	{
		auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(arrow_builder);
		arrow::Status rv;

		if (!ptr)
		{
			rv = builder->AppendNull();
			nullcount++;
		}
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
		nitems++;
		return chunkSize();
	}
	size_t
	moveValue(arrowFileWriterColumn &buddy, uint64_t index)
	{
		auto	builder = std::dynamic_pointer_cast<BUILDER_TYPE>(arrow_builder);
		auto	array = std::dynamic_pointer_cast<ARRAY_TYPE>(buddy.arrow_array);
		arrow::Status rv;

		assert(index < array->length());
		if (array->IsNull(index))
		{
			rv = builder->AppendNull();
			nullcount++;
		}
		else
		{
			auto	datum = *((CPP_TYPE *)array->Value(index));
			updateStats(datum);
			rv = builder->Append(datum);
		}
		if (!rv.ok())
			Elog("unable to move value to '%s' field (%s): %s",
				 field_name.c_str(),
				 arrow_type->name().c_str(),
				 rv.ToString().c_str());
		nitems++;
		return chunkSize();
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

STRING
BINARY
FIXED_SIZE_BINARY
INTERVAL_MONTHS
/// Like STRING, but with 64-bit offsets
    LARGE_STRING = 34,

    /// Like BINARY, but with 64-bit offsets
    LARGE_BINARY = 35,
*/



#endif	/* _ARROW_WRITE_H_ */
