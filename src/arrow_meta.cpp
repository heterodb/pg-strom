/*
 * arrow_meta.cc
 *
 * Routines to handle Apache Arrow/Parquet metadata
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "arrow_defs.h"
#include <arrow/api.h>			/* dnf install arrow-devel, or apt install libarrow-dev */
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/reader.h>
#include <arrow/util/decimal.h>
#ifdef HAS_PARQUET
#include <parquet/arrow/schema.h>	/* dnf install parquet-devel, or apt install libparquet-dev */
#include <parquet/metadata.h>
#include <parquet/statistics.h>
#include <parquet/file_reader.h>
#endif
#include <fcntl.h>
#include <sstream>
#include <string>

#include "flatbuffers/flatbuffers.h"			/* copied from arrow */
#include "flatbuffers/File_generated.h"			/* copied from arrow */
#include "flatbuffers/Message_generated.h"		/* copied from arrow */
#include "flatbuffers/Schema_generated.h"		/* copied from arrow */
#include "flatbuffers/SparseTensor_generated.h"		/* copied from arrow */
#include "flatbuffers/Tensor_generated.h"		/* copied from arrow */
#define ARROW_SIGNATURE			"ARROW1"
#define ARROW_SIGNATURE_SZ		(sizeof(ARROW_SIGNATURE)-1)
#define PARQUET_SIGNATURE		"PAR1"
#define PARQUET_SIGNATURE_SZ	(sizeof(PARQUET_SIGNATURE)-1)

/*
 * Error Reporting
 */
#ifdef __PGSTROM_MODULE__
extern "C" {
#include "postgres.h"
}
#else
#define Max(a,b)		((a) > (b) ? (a) : (b))
#define Min(a,b)		((a) < (b) ? (a) : (b))
#endif
#define Elog(fmt,...)								\
	do {											\
		char   *ebuf = (char *)alloca(320);			\
		snprintf(ebuf, 320, "(%s:%d) " fmt,			\
				 __FILE__,__LINE__, ##__VA_ARGS__);	\
		throw std::runtime_error(ebuf);				\
	} while(0)
static inline void
ErrorReport(const char *emsg)
{
#ifdef __PGSTROM_MODULE__
	elog(ERROR, "%s", emsg);
#else
	fprintf(stderr, "(ERROR %s\n", emsg+1);
	exit(1);
#endif
}

/*
 * Memory Allocation
 */
#ifdef __PGSTROM_MODULE__
extern "C" {
#include "utils/palloc.h"
}
static inline void *__palloc(size_t sz)
{
	void   *ptr = palloc_extended(sz, MCXT_ALLOC_HUGE | MCXT_ALLOC_NO_OOM);
	if (!ptr)
		Elog("out of memory (sz=%lu)", sz);
	return ptr;
}

static inline void *__palloc0(size_t sz)
{
	void   *ptr = palloc_extended(sz, MCXT_ALLOC_HUGE | MCXT_ALLOC_NO_OOM | MCXT_ALLOC_ZERO);
	if (!ptr)
		Elog("out of memory (sz=%lu)", sz);
	return ptr;
}

static inline void *__repalloc(void *ptr, size_t sz)
{
	ptr = repalloc_extended(ptr, sz, MCXT_ALLOC_HUGE | MCXT_ALLOC_NO_OOM);
	if (!ptr)
		Elog("out of memory (sz=%lu)", sz);
	return ptr;
}
#else
static inline void *__palloc(size_t sz)
{
	void   *ptr = malloc(sz);
	if (!ptr)
		Elog("out of memory (sz=%lu)", sz);
	return ptr;
}
static inline void *__palloc0(size_t sz)
{
	void   *ptr = malloc(sz);
	if (!ptr)
		Elog("out of memory (sz=%lu)", sz);
	memset(ptr, 0, sz);
	return ptr;
}
static inline void *__repalloc(void *ptr, size_t sz)
{
	ptr = realloc(ptr, sz);
	if (!ptr)
		Elog("out of memory (sz=%lu)", sz);
	return ptr;
}
#endif
static inline char *__pstrdup(const char *str)
{
	size_t	sz = strlen(str);
	char   *result = (char *)__palloc(sz+1);
	memcpy(result, str, sz);
	result[sz] = '\0';
	return result;
}
static inline char *__pstrdup(const flatbuffers::String *str)
{
	size_t	sz = str->size();
	char   *result = (char *)__palloc(sz+1);
	memcpy(result, str->data(), sz);
	result[sz] = '\0';
	return result;
}
// ============================================================
//
// Routines to dump Apache Arrow nodes in JSON string
//
// ============================================================
static void
__dumpArrowNode(std::ostringstream &json,
				const ArrowNode *node,
				std::string NewLine);

static inline std::string
__escape_json(const char *str)
{
	std::string	res;

	res += "\"";
	for (const char *pos = str; *pos != '\0'; pos++)
	{
		switch (*pos)
		{
            case '\b':
				res += "\b";
				break;
            case '\f':
				res += "\\f";
				break;
            case '\n':
				res += "\\n";
				break;
            case '\r':
				res += "\\r";
                break;
            case '\t':
				res += "\\t";
                break;
            case '"':
				res += "\\\"";
                break;
            case '\\':
				res += "\\\\";
                break;
            default:
				if ((unsigned char) *pos < ' ')
				{
					char	buf[32];
					snprintf(buf, 32, "\\u%04x", (int)*pos);
					res += buf;
				}
				else
				{
					res.push_back(*pos);
				}
				break;
		}
	}
	res += "\"";
	return res;
}

#define __SPACES(JSON_KEY)		std::string(std::strlen(JSON_KEY)+7,' ')
#define __ARRAY_INDENT			std::string("  ")

static inline void
__dumpArrowTypeInt(std::ostringstream &json,
				   const ArrowTypeInt *node,
				   std::string NewLine)
{
	const char *tag = (node->is_signed ? "Int" : "UInt");
	json << "\"" << tag << node->bitWidth  << "\"";
}
static inline void
__dumpArrowTypeFloatingPoint(std::ostringstream &json,
							 const ArrowTypeFloatingPoint *node,
							 std::string NewLine)
{
	const char *suffix = (node->precision == ArrowPrecision__Double ? "64" :
						  node->precision == ArrowPrecision__Single ? "32" :
						  node->precision == ArrowPrecision__Half   ? "16" : "???");
	json << "\"" << suffix << "\"";
}
static inline void
__dumpArrowTypeDecimal(std::ostringstream &json,
					   const ArrowTypeDecimal *node,
					   std::string NewLine)
{
	json << "\"Decimal" << node->bitWidth << "\", "
		 << "\"precision\" : " << node->precision << ", "
		 << "\"scale\" : " << node->scale;
}
static inline void
__dumpArrowTypeDate(std::ostringstream &json,
					const ArrowTypeDate *node,
					std::string NewLine)
{
	const char *unit = ArrowDateUnitAsCString(node->unit);
	const char *suffix = (node->unit == ArrowDateUnit__Day ? "32" : "64");

	json << "\"Date" << suffix << "\", "
		 << "\"unit\" : \"" << unit << "\"";
}
static inline void
__dumpArrowTypeTime(std::ostringstream &json,
					const ArrowTypeTime *node,
					std::string NewLine)
{
	const char *unit = ArrowTimeUnitAsCString(node->unit);
	const char *suffix = (node->unit == ArrowTimeUnit__Second ||
						  node->unit == ArrowTimeUnit__MilliSecond) ? "32" : "64";
	json << "\"Time" << suffix << "\", "
		 << "\"unit\" : \"" << unit << "\"";
}
static inline void
__dumpArrowTypeTimestamp(std::ostringstream &json,
						 const ArrowTypeTimestamp *node,
						 std::string NewLine)
{
	const char *unit = ArrowTimeUnitAsCString(node->unit);
	json << "\"Timestamp\", "
		 << "\"unit\" : \"" << unit << "\"";
	if (node->timezone)
		json << ", \"timezone\" : " << __escape_json(node->timezone);
}
static inline void
__dumpArrowTypeInterval(std::ostringstream &json,
						const ArrowTypeInterval *node,
						std::string NewLine)
{
	const char *unit = ArrowIntervalUnitAsCString(node->unit);
	json << "\"Interval\", \"unit\" : \"" << unit << "\"";
}
static inline void
__dumpArrowTypeUnion(std::ostringstream &json,
					 const ArrowTypeUnion *node,
					 std::string NewLine)
{
	const char *mode = ArrowUnionModeAsCString(node->mode);
	json << "\"Union\", "
		 << "\"mode\" : \""<< mode << "\", "
		 << "\"typeIds\" : [";
	for (int i=0; i < node->_num_typeIds; i++)
	{
		if (i > 0)
			json << ", ";
		json << node->typeIds[i];
	}
	json << "]";
}
static inline void
__dumpArrowTypeFixedSizeBinary(std::ostringstream &json,
							   const ArrowTypeFixedSizeBinary *node,
							   std::string NewLine)
{
	json << "\"FixedSizeBinary\", "
		 << "\"byteWidth\" : " << node->byteWidth;
}
static inline void
__dumpArrowTypeFixedSizeList(std::ostringstream &json,
							 const ArrowTypeFixedSizeList *node,
							 std::string NewLine)
{
	json << "\"FixedSizeList\", "
		 << "\"listSize\" : " << node->listSize;
}
static inline void
__dumpArrowTypeMap(std::ostringstream &json,
				   const ArrowTypeMap *node,
				   std::string NewLine)
{
	json << "\"Map\", "
		 << "\"keysSorted\" : " << (node->keysSorted ? "true" : "false");
}
static inline void
__dumpArrowTypeDuration(std::ostringstream &json,
						const ArrowTypeDuration *node,
						std::string NewLine)
{
	const char *unit = ArrowTimeUnitAsCString(node->unit);
	json << "\"Duration\", "
		 << "\"unit\" : " << __escape_json(unit);
}
static inline void
__dumpArrowKeyValue(std::ostringstream &json,
					const ArrowKeyValue *node,
					std::string NewLine)
{
	if (node->_key_len + node->_value_len <= 64)
	{
		json << "\"KeyValue\", "
			 << "\"key\" : " << __escape_json(node->key) << ", "
			 << "\"value\" : "<< __escape_json(node->value);
	}
	else
	{
		json << "\"KeyValue\"," << NewLine
			 << "\"key\" : " << __escape_json(node->key) << "," << NewLine
			 << "\"value\" : "<< __escape_json(node->value);
	}
}
static inline void
__dumpArrowDictionaryEncoding(std::ostringstream &json,
							  const ArrowDictionaryEncoding *node,
							  std::string NewLine)
{
	json << "\"DictionaryEncoding\"," << NewLine
		 << "\"id\" : " << node->id << "," << NewLine
		 << ", \"indexType\" : ";
	__dumpArrowNode(json, &node->indexType.node, NewLine);
	json << "," << NewLine
		 <<"\"isOrdered\" : " << (node->isOrdered ? "true" : "false");
}
static inline const char *
__ParquetPhysicalTypeAsCString(int code)
{
	switch (code)
	{
		case parquet::Type::BOOLEAN:	return "BOOLEAN";
		case parquet::Type::INT32:		return "INT32";
		case parquet::Type::INT64:		return "INT64";
		case parquet::Type::INT96:		return "INT96";
		case parquet::Type::FLOAT:		return "FLOAT";
		case parquet::Type::DOUBLE:		return "DOUBLE";
		case parquet::Type::BYTE_ARRAY:	return "BYTE_ARRAY";
		case parquet::Type::FIXED_LEN_BYTE_ARRAY: return "FIXED_LEN_BYTE_ARRAY";
		default:						return "???";
	}
}
static inline const char *
__ParquetConvertedTypeAsCString(int code)
{
	switch (code)
	{
		case parquet::ConvertedType::NONE:		return "NONE";
		case parquet::ConvertedType::UTF8:		return "UTF8";
		case parquet::ConvertedType::MAP:		return "MAP";
		case parquet::ConvertedType::MAP_KEY_VALUE: return "MAP_KEY_VALUE";
		case parquet::ConvertedType::LIST:		return "LIST";
		case parquet::ConvertedType::ENUM:		return "ENUM";
		case parquet::ConvertedType::DECIMAL:	return "DECIMAL";
		case parquet::ConvertedType::DATE:		return "DATE";
		case parquet::ConvertedType::TIME_MILLIS:		return "TIME_MILLIS";
		case parquet::ConvertedType::TIME_MICROS:		return "TIME_MICROS";
		case parquet::ConvertedType::TIMESTAMP_MILLIS:	return "TIMESTAMP_MILLIS";
		case parquet::ConvertedType::TIMESTAMP_MICROS:	return "TIMESTAMP_MICROS";
		case parquet::ConvertedType::UINT_8:	return "UINT_8";
		case parquet::ConvertedType::UINT_16:	return "UINT_16";
		case parquet::ConvertedType::UINT_32:	return "UINT_32";
		case parquet::ConvertedType::UINT_64:	return "UINT_64";
		case parquet::ConvertedType::INT_8:		return "INT_8";
		case parquet::ConvertedType::INT_16:	return "INT_16";
		case parquet::ConvertedType::INT_32:	return "INT_32";
		case parquet::ConvertedType::INT_64:	return "INT_64";
		case parquet::ConvertedType::JSON:		return "JSON";
		case parquet::ConvertedType::BSON:		return "BSON";
		case parquet::ConvertedType::INTERVAL:	return "INTERVAL";
		default:								return "???";
	}
}
static inline const char *
__ParquetLogicalTypeAsCString(int code)
{
	switch (code)
	{
		case parquet::LogicalType::Type::STRING:	return "STRING";
		case parquet::LogicalType::Type::MAP:		return "MAP";
		case parquet::LogicalType::Type::LIST:		return "LIST";
		case parquet::LogicalType::Type::ENUM:		return "ENUM";
		case parquet::LogicalType::Type::DECIMAL:	return "DECIMAL";
		case parquet::LogicalType::Type::DATE:		return "DATE";
		case parquet::LogicalType::Type::TIME:		return "TIME";
		case parquet::LogicalType::Type::TIMESTAMP:	return "TIMESTAMP";
		case parquet::LogicalType::Type::INTERVAL:	return "INTERVAL";
		case parquet::LogicalType::Type::INT:		return "INT";
		case parquet::LogicalType::Type::NIL:		return "NIL";
		case parquet::LogicalType::Type::JSON:		return "JSON";
		case parquet::LogicalType::Type::BSON:		return "BSON";
		case parquet::LogicalType::Type::UUID:		return "UUID";
		default:									return "???";
	}
}
static inline const char *
__ParquetLogicalTimeUnitAsCString(int code)
{
	switch (code)
	{
		case parquet::LogicalType::TimeUnit::MILLIS:	return "ms";
		case parquet::LogicalType::TimeUnit::MICROS:	return "us";
		case parquet::LogicalType::TimeUnit::NANOS:		return "ns";
		default:										return "???";
	}
}
static inline void
__dumpArrowField(std::ostringstream &json,
				 const ArrowField *node,
				 std::string NewLine)
{
	bool	multiline = (node->dictionary ||
						 node->_num_children > 0 ||
						 node->_num_custom_metadata > 0);
#ifdef HAS_PARQUET
	if (node->parquet.max_definition_level != 0 ||
		node->parquet.max_repetition_level != 0 ||
		node->parquet.physical_type  != (int16_t)parquet::Type::UNDEFINED ||
		node->parquet.converted_type != (int16_t)parquet::ConvertedType::UNDEFINED ||
		node->parquet.logical_type   != (int16_t)parquet::LogicalType::Type::UNDEFINED)
		multiline = true;
#endif
	json << "\"Field\"," << (multiline ? NewLine : " ")
		 << "\"name\" : " << __escape_json(node->name) << "," << (multiline ? NewLine : " ")
		 << "\"type\" : ";
	__dumpArrowNode(json, &node->type.node,
					NewLine + __SPACES("type"));
	json << "," << (multiline ? NewLine : " ")
		 << "\"nullable\" : " << (node->nullable ? "true" : "false");
	if (node->dictionary)
	{
		json << "," << (multiline ? NewLine : " ")
			 << "\"dictionary\" : ";
		__dumpArrowNode(json, (ArrowNode *)node->dictionary,
						NewLine + __SPACES("dictionary"));
	}
	if (node->children && node->_num_children > 0)
	{
		auto	__NewLine = NewLine + __SPACES("children");
		json << "," << NewLine << "\"children\" : [";
		for (int i=0; i < node->_num_children; i++)
		{
			if (i == 0)
				json << " ";
			else
				json << "," << (multiline ? __NewLine : " ");
			__dumpArrowNode(json, &node->children[i].node,
							__NewLine + __ARRAY_INDENT);
		}
		json << " ]";
	}
	if (node->custom_metadata && node->_num_custom_metadata > 0)
	{
		auto	__NewLine = NewLine + __SPACES("custom_metadata");
		json << "," << NewLine << "\"custom_metadata\" : [";
		for (int i=0; i < node->_num_custom_metadata; i++)
		{
			if (i == 0)
				json << " ";
			else
				json << "," << (multiline ? __NewLine : " ");
			__dumpArrowNode(json, &node->custom_metadata[i].node,
							__NewLine + __ARRAY_INDENT);
		}
		json << " ]";
	}
#if HAS_PARQUET
	if (node->parquet.max_definition_level != 0 ||
        node->parquet.max_repetition_level != 0 ||
		node->parquet.physical_type  != (int16_t)parquet::Type::UNDEFINED ||
        node->parquet.converted_type != (int16_t)parquet::ConvertedType::UNDEFINED ||
        node->parquet.logical_type   != (int16_t)parquet::LogicalType::Type::UNDEFINED)
	{
		const char *indent = "              ";
		json << "," << NewLine
			 << "\"parquet\" : { "
			 << "\"max_definition_level\" : " << node->parquet.max_definition_level
			 << "," << NewLine << indent
			 << "\"max_repetition_level\" : " << node->parquet.max_repetition_level;
		if (node->parquet.physical_type  != (int16_t)parquet::Type::UNDEFINED)
			json << "," << NewLine << indent
				 << "\"physical_type\" : \""<< __ParquetPhysicalTypeAsCString(node->parquet.physical_type) << "\"";
		if (node->parquet.converted_type != (int16_t)parquet::ConvertedType::UNDEFINED)
			json << "," << NewLine << indent
				 << "\"converted_type\" : \"" << __ParquetConvertedTypeAsCString(node->parquet.converted_type) << "\"";
		if (node->parquet.logical_type != (int16_t)parquet::LogicalType::Type::UNDEFINED)
		{
			json << "," << NewLine << indent
				 << "\"logical_type\" : \"" << __ParquetLogicalTypeAsCString(node->parquet.logical_type) << "\"";
			if (node->parquet.logical_type == parquet::LogicalType::Type::TIME ||
				node->parquet.logical_type == parquet::LogicalType::Type::TIMESTAMP)
			{
				json << "," << NewLine << indent
					 << "\"logical_time_unit\" : \"" << __ParquetLogicalTimeUnitAsCString(node->parquet.logical_time_unit) << "\"";
			}
			else if (node->parquet.logical_type == parquet::LogicalType::Type::DECIMAL)
			{
				json << "," << NewLine << indent
					 << "\"logical_decimal_precision\" : " << node->parquet.logical_decimal_precision
					 << "," << NewLine << indent
					 << "\"logical_decimal_scale\" : " << node->parquet.logical_decimal_scale;
			}
		}
		json << "}";
	}
#endif
}

#ifdef HAS_PARQUET
/*
 * Parquet Encoding Type
 */
static inline const char *
__ParquetEncodingTypeAsCString(int code)
{
	switch (code)
	{
		case parquet::Encoding::type::PLAIN:
			return "PLAIN";
		case parquet::Encoding::type::PLAIN_DICTIONARY:
			return "PLAIN_DICTIONARY";
		case parquet::Encoding::type::RLE:
			return "RLE";
		case parquet::Encoding::type::BIT_PACKED:
			return "BIT_PACKED";
		case parquet::Encoding::type::DELTA_BINARY_PACKED:
			return "DELTA_BINARY_PACKED";
		case parquet::Encoding::type::DELTA_LENGTH_BYTE_ARRAY:
			return "DELTA_LENGTH_BYTE_ARRAY";
		case parquet::Encoding::type::DELTA_BYTE_ARRAY:
			return "DELTA_BYTE_ARRAY";
		case parquet::Encoding::type::RLE_DICTIONARY:
			return "RLE_DICTIONARY";
		case parquet::Encoding::type::BYTE_STREAM_SPLIT:
			return "BYTE_STREAM_SPLIT";
		default:
			return "???";
	}
}

/*
 * Parquet PageType
 */
static inline const char *
__ParquetPageTypeAsCString(int code)
{
	switch (code)
	{
		case parquet::PageType::type::DATA_PAGE:		return "DATA_PAGE";
		case parquet::PageType::type::INDEX_PAGE:		return "INDEX_PAGE";
		case parquet::PageType::type::DICTIONARY_PAGE:	return "DICTIONARY_PAGE";
		case parquet::PageType::type::DATA_PAGE_V2:		return "DATA_PAGE_V2";
		default:										return "???";
	}
}
#endif

static inline void
__dumpArrowFieldNode(std::ostringstream &json,
					 const ArrowFieldNode *node,
					 std::string NewLine)
{
	bool	multiline = (node->_stat_min_value_len +
						 node->_stat_max_value_len > 48);
#ifdef HAS_PARQUET
	if (node->parquet.dictionary_page_offset != 0 ||
		node->parquet.data_page_offset != 0 ||
		node->parquet.index_page_offset != 0 ||
		node->parquet.total_compressed_size != 0 ||
		node->parquet.total_uncompressed_size != 0 ||
		node->parquet.compression_type != ArrowCompressionType__UNKNOWN ||
		node->parquet._num_encodings > 0 ||
		node->parquet._num_encoding_stats > 0)
		multiline = true;
#endif
	json << "\"FieldNode\", " << (multiline ? NewLine : " ")
		 << "\"length\" : " << node->length << "," << (multiline ? NewLine : " ")
		 << "\"null_count\" : " << node->null_count;
	if (node->stat_min_value)
	{
		json << "," << (multiline ? NewLine : " ")
			 << "\"stat_min_value\" : " << __escape_json(node->stat_min_value);
	}
	if (node->stat_max_value)
	{
		json << "," << (multiline ? NewLine : " ")
			 << "\"stat_max_value\" : " << __escape_json(node->stat_max_value);
	}
#ifdef HAS_PARQUET
	if (node->parquet.dictionary_page_offset != 0 ||
		node->parquet.data_page_offset != 0 ||
		node->parquet.index_page_offset != 0 ||
		node->parquet.total_compressed_size != 0 ||
		node->parquet.total_uncompressed_size != 0 ||
		node->parquet.compression_type != ArrowCompressionType__UNKNOWN ||
		node->parquet._num_encodings > 0 ||
		node->parquet._num_encoding_stats > 0)
	{
		const char *indent = "             ";

		assert(multiline);
		json << "," << NewLine
			 << "\"parquet\" : {"
			 << "\"total_compressed_size\" : " << node->parquet.total_compressed_size << ","
			 << NewLine << indent
			 << "\"total_uncompressed_size\" : " << node->parquet.total_uncompressed_size << ","
			 << NewLine << indent
			 << "\"data_page_offset\" : " << node->parquet.data_page_offset;
		if (node->parquet.index_page_offset != 0)
			json << "," << NewLine << indent
				 << "\"index_page_offset\" : " << node->parquet.index_page_offset;
		if (node->parquet.dictionary_page_offset != 0)
			json << "," << NewLine << indent
				 << "\"dictionary_page_offset\" : " << node->parquet.dictionary_page_offset;
		json << "," << NewLine << indent
			 << "\"compression_type\" : \"" << ArrowCompressionTypeAsCString(node->parquet.compression_type) << "\"";
		if (node->parquet._num_encodings > 0)
		{
			json << "," << NewLine << indent
				 << "\"encodings\" : [";
			for (int k=0; k < node->parquet._num_encodings; k++)
			{
				const char *enc = __ParquetEncodingTypeAsCString(node->parquet.encodings[k]);
				if (k > 0)
					json << ", ";
				json << "\"" << enc << "\"";
			}
			json << "]";
		}
		if (node->parquet._num_encoding_stats > 0)
		{
			const char *__indent = "                    ";
			json << "," << NewLine << indent
				 << "\"encoding_stats\" : [";
			for (int k=0; k < node->parquet._num_encoding_stats; k++)
			{
				auto	est = &node->parquet.encoding_stats[k];
				if (k > 0)
					json << "," << NewLine << indent << __indent;
				json << "{ \"page_type\" : \""
					 << __ParquetPageTypeAsCString(est->page_type)
					 << "\", \"encoding\" : \""
					 << __ParquetEncodingTypeAsCString(est->encoding)
					 << "\", \"count\" : " << est->count << "}";
			}
			json << "]";
		}
	}
#endif
}
static inline void
__dumpArrowBuffer(std::ostringstream &json,
				  const ArrowBuffer *node,
				  std::string NewLine)
{
	json << "\"Buffer\", "
		 << "\"offset\" : " << node->offset << ", "
		 << "\"length\" : " << node->length;
}
static inline void
__dumpArrowSchema(std::ostringstream &json,
				  const ArrowSchema *node,
				  std::string NewLine)
{
	const char *endian = ArrowEndiannessAsCString(node->endianness);
	json << "\"Schema\"," << NewLine
		 << "\"endianness\" : \""<< endian << "\"";
	if (node->fields && node->_num_fields > 0)
	{
		auto	__NewLine = NewLine + __SPACES("fields");
		json << "," << NewLine
			 << "\"fields\" : [";
		for (int i=0; i < node->_num_fields; i++)
		{
			if (i == 0)
				json << " ";
			else
				json << "," << __NewLine;
			__dumpArrowNode(json, &node->fields[i].node,
							__NewLine + __ARRAY_INDENT);
		}
		json << " ]";
	}
	if (node->custom_metadata && node->_num_custom_metadata > 0)
	{
		auto	__NewLine = NewLine + __SPACES("custom_metadata");
		json << "," << NewLine
			 << "\"custom_metadata\" : [";
		for (int i=0; i < node->_num_custom_metadata; i++)
		{
			if (i == 0)
				json << " ";
			else
				json << "," << __NewLine;
			__dumpArrowNode(json, &node->custom_metadata[i].node,
							__NewLine + __ARRAY_INDENT);
		}
		json << " ]";
	}
	if (node->features && node->_num_features > 0)
	{
		json << "," << NewLine
			 << "\"features\" : [";
		for (int i=0; i < node->_num_features; i++)
		{
			const char *feature = ArrowFeatureAsCString(node->features[i]);

			json << (i==0 ? " " : ", ");
			json << "\"" << feature << "\"";
		}
		json << " ]";
	}
}
static inline void
__dumpArrowRecordBatch(std::ostringstream &json,
					   const ArrowRecordBatch *node,
					   std::string NewLine)
{
	json << "\"ArrowRecordBatch\"," << NewLine
		 << "\"length\" : " << node->length;
	if (node->nodes && node->_num_nodes > 0)
	{
		auto	__NewLine = NewLine + __SPACES("nodes");
		json << "," << NewLine
			 << "\"nodes\" : [";
		for (int i=0; i < node->_num_nodes; i++)
		{
			if (i == 0)
				json << " ";
			else
				json << "," << __NewLine;
			__dumpArrowNode(json, &node->nodes[i].node,
							__NewLine + __ARRAY_INDENT);
		}
		json << " ]";
	}
	if (node->buffers && node->_num_buffers > 0)
	{
		auto	__NewLine = NewLine + __SPACES("buffers");

		json << "," << NewLine
			 << "\"buffers\" : [";
		for (int i=0; i < node->_num_buffers; i++)
		{
			if (i == 0)
				json << " ";
			else
				json << "," << __NewLine;
			__dumpArrowNode(json, &node->buffers[i].node,
							__NewLine + __ARRAY_INDENT);
		}
		json << " ]";
	}
	if (node->compression)
	{
		json << "," << NewLine
			 << "\"compression\" : ";
		__dumpArrowNode(json, (const ArrowNode *)node->compression,
						NewLine + __SPACES("compression"));
	}
}
static inline void
__dumpArrowDictionaryBatch(std::ostringstream &json,
						   const ArrowDictionaryBatch *node,
						   std::string NewLine)
{
	json << "\"DictionaryBatch\"," << NewLine
		 << "\"id\" : " << node->id << NewLine
		 << "\"data\" : ";
	__dumpArrowNode(json, &node->data.node,
					NewLine + __SPACES("data"));
	json << "," << NewLine
		 << "\"isDelta\" : " << (node->isDelta ? "true" : "false");
}
static inline void
__dumpArrowMessage(std::ostringstream &json,
				   const ArrowMessage *node,
				   std::string NewLine)
{
	const char *version = ArrowMetadataVersionAsCString(node->version);
	json << "\"Message\"," << NewLine
		 << "\"version\" : \"" << version << "\"," << NewLine
		 << "\"body\" : ";
	__dumpArrowNode(json, &node->body.node,
					NewLine + __SPACES("body"));
	json << "," << NewLine
		 <<"\"bodyLength\" : " << node->bodyLength;
	if (node->_num_custom_metadata > 0)
	{
		auto	__NewLine = NewLine + __SPACES("custom_metadata");
		json << "," << NewLine << "\"custom_metadata\" : [";
		for (int i=0; i < node->_num_custom_metadata; i++)
		{
			if (i == 0)
				json << " ";
			else
				json << "," << __NewLine;
			__dumpArrowNode(json, &node->custom_metadata[i].node,
							__NewLine + __ARRAY_INDENT);
		}
	}	
}
static inline void
__dumpArrowBlock(std::ostringstream &json,
				 const ArrowBlock *node,
				 std::string NewLine)
{
	json << "\"Block\", "
		 << "\"offset\" : " << node->offset << ", "
		 << "\"metaDataLength\" : " << node->metaDataLength << ", "
		 << "\"bodyLength\" : " << node->bodyLength;
}
static inline void
__dumpArrowFooter(std::ostringstream &json,
				  const ArrowFooter *node,
				  std::string NewLine)
{
	const char *version = ArrowMetadataVersionAsCString(node->version);
	json << "\"Footer\"," << NewLine
		 << "\"version\" : \"" << version << "\"," << NewLine
		 << "\"schema\" : ";
	__dumpArrowNode(json, &node->schema.node,
					NewLine + __SPACES("schema"));
	if (node->dictionaries && node->_num_dictionaries > 0)
	{
		auto	__NewLine = NewLine + __SPACES("dictionaries  ");
		json << "," << NewLine
			 << "\"dictionaries\" : [";
		for (int i=0; i < node->_num_dictionaries; i++)
		{
			if (i == 0)
				json << " ";
			else
				json << "," << __NewLine;
			__dumpArrowNode(json, &node->dictionaries[i].node,
							__NewLine);
		}
		json << " ]";
	}
	if (node->recordBatches && node->_num_recordBatches > 0)
	{
		auto	__NewLine = NewLine + __SPACES("recordBatches");
		json << "," << NewLine
			 << "\"recordBatches\" : [";
		for (int i=0; i < node->_num_recordBatches; i++)
		{
			if (i == 0)
				json << " ";
			else
				json << "," << __NewLine;
			__dumpArrowNode(json, &node->recordBatches[i].node,
							__NewLine);
		}
		json << " ]";
	}
	if (node->custom_metadata && node->_num_custom_metadata > 0)
	{
		auto	__NewLine = NewLine + __SPACES("custom_metadata  ");
		json << "," << NewLine
			 <<"\"custom_metadata\" : [";
		for (int i=0; i < node->_num_custom_metadata; i++)
		{
			if (i == 0)
				json << " ";
			else
				json << "," << __NewLine;
			__dumpArrowNode(json, &node->custom_metadata[i].node,
							__NewLine);
		}
		json << " ]";
	}
}
static inline void
__dumpArrowBodyCompression(std::ostringstream &json,
						   const ArrowBodyCompression *node,
						   std::string NewLine)
{
	const char *codec = ArrowCompressionTypeAsCString(node->codec);
	const char *method = ArrowBodyCompressionMethodAsCString(node->method);
	json << "\"BodyCompression\"," << NewLine
		 << "\"codec\" : \"" << codec << "\"," << NewLine
		 << "\"method\" : \"" << method << "\"";
}
static void
__dumpArrowNode(std::ostringstream &json, const ArrowNode *node, std::string NewLine)
{
	json << "{ \"tag\" : ";
	switch (node->tag)
	{
		case ArrowNodeTag__Null:
		case ArrowNodeTag__Utf8:
		case ArrowNodeTag__Binary:
		case ArrowNodeTag__Bool:
		case ArrowNodeTag__List:
		case ArrowNodeTag__Struct:
		case ArrowNodeTag__LargeBinary:
		case ArrowNodeTag__LargeUtf8:
		case ArrowNodeTag__LargeList:
			json << __escape_json(node->tagName);
			break;		/* nothing to special */
		case ArrowNodeTag__Int:
			__dumpArrowTypeInt(json, (const ArrowTypeInt *)node, NewLine);
			break;
		case ArrowNodeTag__FloatingPoint:
			__dumpArrowTypeFloatingPoint(json, (const ArrowTypeFloatingPoint *)node, NewLine);
			break;
		case ArrowNodeTag__Decimal:
			__dumpArrowTypeDecimal(json, (const ArrowTypeDecimal *)node, NewLine);
			break;
		case ArrowNodeTag__Date:
			__dumpArrowTypeDate(json, (const ArrowTypeDate *)node, NewLine);
			break;
		case ArrowNodeTag__Time:
			__dumpArrowTypeTime(json, (const ArrowTypeTime *)node, NewLine);
			break;
		case ArrowNodeTag__Timestamp:
			__dumpArrowTypeTimestamp(json, (const ArrowTypeTimestamp *)node, NewLine);
			break;
		case ArrowNodeTag__Interval:
			__dumpArrowTypeInterval(json, (const ArrowTypeInterval *)node, NewLine);
			break;
		case ArrowNodeTag__Union:
			__dumpArrowTypeUnion(json, (const ArrowTypeUnion *)node, NewLine);
			break;
		case ArrowNodeTag__FixedSizeBinary:
			__dumpArrowTypeFixedSizeBinary(json, (const ArrowTypeFixedSizeBinary *)node, NewLine);
			break;
		case ArrowNodeTag__FixedSizeList:
			__dumpArrowTypeFixedSizeList(json, (const ArrowTypeFixedSizeList *)node, NewLine);
			break;
		case ArrowNodeTag__Map:
			__dumpArrowTypeMap(json, (const ArrowTypeMap *)node, NewLine);
			break;
		case ArrowNodeTag__Duration:
			__dumpArrowTypeDuration(json, (const ArrowTypeDuration *)node, NewLine);
			break;
		case ArrowNodeTag__KeyValue:
			__dumpArrowKeyValue(json, (const ArrowKeyValue *)node, NewLine);
			break;
		case ArrowNodeTag__DictionaryEncoding:
			__dumpArrowDictionaryEncoding(json, (const ArrowDictionaryEncoding *)node, NewLine);
			break;
		case ArrowNodeTag__Field:
			__dumpArrowField(json, (const ArrowField *)node, NewLine);
			break;
		case ArrowNodeTag__FieldNode:
			__dumpArrowFieldNode(json, (const ArrowFieldNode *)node, NewLine);
			break;
		case ArrowNodeTag__Buffer:
			__dumpArrowBuffer(json, (const ArrowBuffer *)node, NewLine);
			break;
		case ArrowNodeTag__Schema:
			__dumpArrowSchema(json, (const ArrowSchema *)node, NewLine);
			break;
		case ArrowNodeTag__RecordBatch:
			__dumpArrowRecordBatch(json, (const ArrowRecordBatch *)node, NewLine);
			break;
		case ArrowNodeTag__DictionaryBatch:
			__dumpArrowDictionaryBatch(json, (const ArrowDictionaryBatch *)node, NewLine);
			break;
		case ArrowNodeTag__Message:
			__dumpArrowMessage(json, (const ArrowMessage *)node, NewLine);
			break;
		case ArrowNodeTag__Block:
			__dumpArrowBlock(json, (const ArrowBlock *)node, NewLine);
			break;
		case ArrowNodeTag__Footer:
			__dumpArrowFooter(json, (const ArrowFooter *)node, NewLine);
			break;
		case ArrowNodeTag__BodyCompression:
			__dumpArrowBodyCompression(json, (const ArrowBodyCompression *)node, NewLine);
			break;
		default:
			Elog("unknown ArrowNodeTag (%d)", (int)node->tag);
	}
	json << " }";
}

char *
dumpArrowNode(const ArrowNode *node)
{
	char   *emsg = NULL;
	char   *result;
	try {
		std::ostringstream json;
		std::string temp;

		__dumpArrowNode(json, node, std::string("\n  "));
		temp = json.str();
		result = (char *)__palloc(temp.size() + 1);
		memcpy(result, temp.data(), temp.size());
		result[temp.size()] = '\0';
	}
	catch (const std::exception &e) {
		const char *estr = e.what();
		emsg = (char *)alloca(std::strlen(estr)+1);
		strcpy(emsg, estr);
	}
	/* error report */
	if (emsg)
		ErrorReport(emsg);
	return result;
}

/*
 * dumpArrowFileInfo
 */
char *
dumpArrowFileInfo(const ArrowFileInfo *af_info)
{
	char   *emsg = NULL;
	char   *result;
	try {
		std::ostringstream json;
		std::string temp;

		json << "{ \"filename\" : \"" << af_info->filename << "\",\n"
			 << "  \"filesize\" : "   << af_info->stat_buf.st_size << ",\n"
			 << "  \"footer\" : ";
		__dumpArrowNode(json, (const ArrowNode *)&af_info->footer,
						std::string("\n               "));
		if (af_info->_num_dictionaries > 0)
		{
			json << ",\n"
				 << "  \"dictionaries\" : [\n          ";
			for (int k=0; k < af_info->_num_dictionaries; k++)
			{
				if (k > 0)
					json << ",\n          ";
				__dumpArrowNode(json, (const ArrowNode *)&af_info->dictionaries[k],
								std::string("\n          "));
			}
			json << "]";
		}
		if (af_info->_num_recordBatches > 0)
		{
			json << ",\n"
				 << "  \"record-batches\" : [\n          ";
			for (int k=0; k < af_info->_num_recordBatches; k++)
			{
				if (k > 0)
					json << ",\n";
				__dumpArrowNode(json, (const ArrowNode *)&af_info->recordBatches[k],
								std::string("\n          "));
			}
			json << "]";
		}
		json << "\n}\n";
		/* result as cstring */
		temp = json.str();
		result = (char *)__palloc(temp.size() + 1);
		memcpy(result, temp.data(), temp.size());
		result[temp.size()] = '\0';
	}
	catch (const std::exception &e) {
		const char *estr = e.what();
		emsg = (char *)alloca(std::strlen(estr)+1);
		strcpy(emsg, estr);
	}
	/* error report */
	if (emsg)
		ErrorReport(emsg);
	return result;
}
#undef __SPACES

// ============================================================
//
// Routines to copy two Apache Arrow nodes
//
// ============================================================
static void
__copyArrowNode(ArrowNode *dest, const ArrowNode *src);

#define COPY_SCALAR(FIELD)			(dest)->FIELD = (src)->FIELD
#define COPY_CSTRING(FIELD)											\
	do {															\
		if ((src)->FIELD)											\
		{															\
			(dest)->FIELD = __pstrdup((src)->FIELD);				\
			(dest)->_##FIELD##_len = strlen((dest)->FIELD);			\
		}															\
		else														\
		{															\
			(dest)->FIELD = NULL;									\
			(dest)->_##FIELD##_len = 0;								\
		}															\
	} while(0)
#define COPY_VECTOR(FIELD,NODETYPE)									\
	do {															\
		(dest)->_num_##FIELD = (src)->_num_##FIELD;					\
        if ((dest)->_num_##FIELD == 0)								\
            (dest)->FIELD = NULL;									\
        else														\
        {															\
			(dest)->FIELD = (NODETYPE *)							\
				__palloc(sizeof(NODETYPE) *	(src)->_num_##FIELD);	\
			for (int j=0; j < (src)->_num_##FIELD; j++)				\
                __copyArrowNode(&(dest)->FIELD[j].node,				\
								&(src)->FIELD[j].node);				\
        }															\
    } while(0)

static inline void
__copyArrowTypeInt(ArrowTypeInt *dest, const ArrowTypeInt *src)
{
	COPY_SCALAR(bitWidth);
	COPY_SCALAR(is_signed);
}

static inline void
__copyArrowTypeFloatingPoint(ArrowTypeFloatingPoint *dest,
                             const ArrowTypeFloatingPoint *src)
{
	COPY_SCALAR(precision);
}

static inline void
__copyArrowTypeDecimal(ArrowTypeDecimal *dest, const ArrowTypeDecimal *src)
{
	COPY_SCALAR(precision);
	COPY_SCALAR(scale);
	COPY_SCALAR(bitWidth);
}

static inline void
__copyArrowTypeDate(ArrowTypeDate *dest, const ArrowTypeDate *src)
{
	COPY_SCALAR(unit);
}

static inline void
__copyArrowTypeTime(ArrowTypeTime *dest, const ArrowTypeTime *src)
{
	COPY_SCALAR(unit);
	COPY_SCALAR(bitWidth);
}

static inline void
__copyArrowTypeTimestamp(ArrowTypeTimestamp *dest,
						 const ArrowTypeTimestamp *src)
{
	COPY_SCALAR(unit);
	COPY_CSTRING(timezone);
}

static inline void
__copyArrowTypeInterval(ArrowTypeInterval *dest,
						const ArrowTypeInterval *src)
{
	COPY_SCALAR(unit);
}
static inline void
__copyArrowTypeUnion(ArrowTypeUnion *dest, const ArrowTypeUnion *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(mode);
	if (src->_num_typeIds == 0)
		dest->typeIds = NULL;
	else
	{
		dest->typeIds = (int32_t *)
			__palloc(sizeof(int32_t) * src->_num_typeIds);
		memcpy(dest->typeIds, src->typeIds,
			   sizeof(int32_t) * src->_num_typeIds);
	}
	dest->_num_typeIds = src->_num_typeIds;
}

static inline void
__copyArrowTypeDuration(ArrowTypeDuration *dest, const ArrowTypeDuration *src)
{
	COPY_SCALAR(unit);
}

static inline void
__copyArrowTypeFixedSizeBinary(ArrowTypeFixedSizeBinary *dest,
							   const ArrowTypeFixedSizeBinary *src)
{
	COPY_SCALAR(byteWidth);
}

static inline void
__copyArrowTypeFixedSizeList(ArrowTypeFixedSizeList *dest,
							 const ArrowTypeFixedSizeList *src)
{
	COPY_SCALAR(listSize);
}

static inline void
__copyArrowTypeMap(ArrowTypeMap *dest, const ArrowTypeMap *src)
{
	COPY_SCALAR(keysSorted);
}

static inline void
__copyArrowBuffer(ArrowBuffer *dest, const ArrowBuffer *src)
{
	COPY_SCALAR(offset);
	COPY_SCALAR(length);
}

static inline void
__copyArrowKeyValue(ArrowKeyValue *dest, const ArrowKeyValue *src)
{
	COPY_CSTRING(key);
	COPY_CSTRING(value);
}

static inline void
__copyArrowDictionaryEncoding(ArrowDictionaryEncoding *dest,
							  const ArrowDictionaryEncoding *src)
{
	COPY_SCALAR(id);
	__copyArrowNode(&dest->indexType.node,
					&src->indexType.node);
	COPY_SCALAR(isOrdered);
}

static inline void
__copyArrowField(ArrowField *dest, const ArrowField *src)
{
	COPY_CSTRING(name);
	COPY_SCALAR(nullable);
	__copyArrowNode(&dest->type.node, &src->type.node);
	if (!src->dictionary)
		dest->dictionary = NULL;
	else
	{
		dest->dictionary = (ArrowDictionaryEncoding *)
			__palloc(sizeof(ArrowDictionaryEncoding));
		__copyArrowNode((ArrowNode *)dest->dictionary,
						(const ArrowNode *)src->dictionary);
	}
	COPY_VECTOR(children, ArrowField);
	COPY_VECTOR(custom_metadata, ArrowKeyValue);
#ifdef HAS_PARQUET
	COPY_SCALAR(parquet.max_definition_level);
	COPY_SCALAR(parquet.max_repetition_level);
	COPY_SCALAR(parquet.physical_type);
	COPY_SCALAR(parquet.converted_type);
	COPY_SCALAR(parquet.logical_type);
	COPY_SCALAR(parquet.logical_time_unit);
	COPY_SCALAR(parquet.logical_decimal_precision);
	COPY_SCALAR(parquet.logical_decimal_scale);
#endif
}

static inline void
__copyArrowFieldNode(ArrowFieldNode *dest, const ArrowFieldNode *src)
{
	COPY_SCALAR(length);
	COPY_SCALAR(null_count);
	COPY_CSTRING(stat_min_value);
	COPY_CSTRING(stat_max_value);
#ifdef HAS_PARQUET
	COPY_SCALAR(parquet.total_compressed_size);
	COPY_SCALAR(parquet.total_uncompressed_size);
	COPY_SCALAR(parquet.data_page_offset);
	COPY_SCALAR(parquet.dictionary_page_offset);
	COPY_SCALAR(parquet.index_page_offset);
	COPY_SCALAR(parquet.compression_type);
	if (src->parquet._num_encodings == 0)
		dest->parquet.encodings = NULL;
	else
	{
		dest->parquet.encodings = (int16_t *)
			__palloc(sizeof(int16_t) * src->parquet._num_encodings);
		memcpy(dest->parquet.encodings,
			   src->parquet.encodings,
			   sizeof(int16_t) * src->parquet._num_encodings);
	}
	dest->parquet._num_encodings = src->parquet._num_encodings;
	if (src->parquet._num_encoding_stats == 0)
		dest->parquet.encoding_stats = NULL;
	else
	{
		dest->parquet.encoding_stats = (ParquetEncodingStats *)
			__palloc(sizeof(ParquetEncodingStats) * src->parquet._num_encoding_stats);
		memcpy(dest->parquet.encoding_stats,
			   src->parquet.encoding_stats,
			   sizeof(ParquetEncodingStats) * src->parquet._num_encoding_stats);
	}
	dest->parquet._num_encoding_stats = src->parquet._num_encoding_stats;
#endif
}

static inline void
__copyArrowSchema(ArrowSchema *dest, const ArrowSchema *src)
{
	COPY_SCALAR(endianness);
	COPY_VECTOR(fields, ArrowField);
	COPY_VECTOR(custom_metadata, ArrowKeyValue);
	if (src->_num_features == 0)
		dest->features = NULL;
	else
	{
		dest->features = (ArrowFeature *)
			__palloc(sizeof(ArrowFeature) * src->_num_features);
		memcpy(dest->features, src->features,
			   sizeof(ArrowFeature) * src->_num_features);
	}
	dest->_num_features = src->_num_features;
}

static inline void
__copyArrowRecordBatch(ArrowRecordBatch *dest, const ArrowRecordBatch *src)
{
	COPY_SCALAR(length);
	COPY_VECTOR(nodes, ArrowFieldNode);
	COPY_VECTOR(buffers, ArrowBuffer);
}

static inline void
__copyArrowDictionaryBatch(ArrowDictionaryBatch *dest,
						   const ArrowDictionaryBatch *src)
{
	COPY_SCALAR(id);
	__copyArrowNode(&dest->data.node,
					&src->data.node);
	COPY_SCALAR(isDelta);
}

static inline void
__copyArrowMessage(ArrowMessage *dest, const ArrowMessage *src)
{
	COPY_SCALAR(version);
	__copyArrowNode(&dest->body.node, &src->body.node);
	COPY_SCALAR(bodyLength);
	COPY_VECTOR(custom_metadata, ArrowKeyValue);
}

static void
__copyArrowBlock(ArrowBlock *dest, const ArrowBlock *src)
{
	COPY_SCALAR(offset);
	COPY_SCALAR(metaDataLength);
	COPY_SCALAR(bodyLength);
}

static void
__copyArrowFooter(ArrowFooter *dest, const ArrowFooter *src)
{
	COPY_SCALAR(version);
	__copyArrowSchema(&dest->schema, &src->schema);
	COPY_VECTOR(dictionaries, ArrowBlock);
	COPY_VECTOR(recordBatches, ArrowBlock);
}

static void
__copyArrowBodyCompression(ArrowBodyCompression *dest, const ArrowBodyCompression *src)
{
	COPY_SCALAR(codec);
	COPY_SCALAR(method);
}

static void
__copyArrowNode(ArrowNode *dest, const ArrowNode *src)
{
	/* common portion */
	COPY_SCALAR(tag);
	COPY_SCALAR(tagName);
	COPY_SCALAR(dumpArrowNode);	//deprecate
	COPY_SCALAR(copyArrowNode);	//deprecate

	switch (src->tag)
	{
		case ArrowNodeTag__Null:
		case ArrowNodeTag__Utf8:
		case ArrowNodeTag__Binary:
		case ArrowNodeTag__Bool:
		case ArrowNodeTag__List:
		case ArrowNodeTag__Struct:
		case ArrowNodeTag__LargeBinary:
		case ArrowNodeTag__LargeUtf8:
		case ArrowNodeTag__LargeList:
			break;
		case ArrowNodeTag__Int:
			__copyArrowTypeInt((ArrowTypeInt *)dest,
							   (const ArrowTypeInt *)src);
			break;
		case ArrowNodeTag__FloatingPoint:
			__copyArrowTypeFloatingPoint((ArrowTypeFloatingPoint *)dest,
										 (const ArrowTypeFloatingPoint *)src);
			break;
		case ArrowNodeTag__Decimal:
			__copyArrowTypeDecimal((ArrowTypeDecimal *)dest,
								   (const ArrowTypeDecimal *)src);
			break;
		case ArrowNodeTag__Date:
			__copyArrowTypeDate((ArrowTypeDate *)dest,
								(const ArrowTypeDate *)src);
			break;
		case ArrowNodeTag__Time:
			__copyArrowTypeTime((ArrowTypeTime *)dest,
								(const ArrowTypeTime *)src);
			break;
		case ArrowNodeTag__Timestamp:
			__copyArrowTypeTimestamp((ArrowTypeTimestamp *)dest,
									 (const ArrowTypeTimestamp *)src);
			break;
		case ArrowNodeTag__Interval:
			__copyArrowTypeInterval((ArrowTypeInterval *)dest,
									(const ArrowTypeInterval *)src);
			break;
		case ArrowNodeTag__Union:
			__copyArrowTypeUnion((ArrowTypeUnion *)dest,
								 (const ArrowTypeUnion *)src);
			break;
		case ArrowNodeTag__FixedSizeBinary:
			__copyArrowTypeFixedSizeBinary((ArrowTypeFixedSizeBinary *)dest,
										   (const ArrowTypeFixedSizeBinary *)src);
			break;
		case ArrowNodeTag__FixedSizeList:
			__copyArrowTypeFixedSizeList((ArrowTypeFixedSizeList *)dest,
										 (const ArrowTypeFixedSizeList *)src);
			break;
		case ArrowNodeTag__Map:
			__copyArrowTypeMap((ArrowTypeMap *)dest,
							   (const ArrowTypeMap *)src);
			break;
		case ArrowNodeTag__Duration:
			__copyArrowTypeDuration((ArrowTypeDuration *)dest,
									(const ArrowTypeDuration *)src);
			break;
		case ArrowNodeTag__KeyValue:
			__copyArrowKeyValue((ArrowKeyValue *)dest,
								(const ArrowKeyValue *)src);
			break;
		case ArrowNodeTag__DictionaryEncoding:
			__copyArrowDictionaryEncoding((ArrowDictionaryEncoding *)dest,
										  (const ArrowDictionaryEncoding *)src);
			break;
		case ArrowNodeTag__Field:
			__copyArrowField((ArrowField *)dest,
							 (const ArrowField *)src);
			break;
		case ArrowNodeTag__FieldNode:
			__copyArrowFieldNode((ArrowFieldNode *)dest,
								 (const ArrowFieldNode *)src);
			break;
		case ArrowNodeTag__Buffer:
			__copyArrowBuffer((ArrowBuffer *)dest,
							  (const ArrowBuffer *)src);
			break;
		case ArrowNodeTag__Schema:
			__copyArrowSchema((ArrowSchema *)dest,
							  (const ArrowSchema *)src);
			break;
		case ArrowNodeTag__RecordBatch:
			__copyArrowRecordBatch((ArrowRecordBatch *)dest,
								   (const ArrowRecordBatch *)src);
			break;
		case ArrowNodeTag__DictionaryBatch:
			__copyArrowDictionaryBatch((ArrowDictionaryBatch *)dest,
									   (const ArrowDictionaryBatch *)src);
			break;
		case ArrowNodeTag__Message:
			__copyArrowMessage((ArrowMessage *)dest,
							   (const ArrowMessage *)src);
			break;
		case ArrowNodeTag__Block:
			__copyArrowBlock((ArrowBlock *)dest,
							 (const ArrowBlock *)src);
			break;
		case ArrowNodeTag__Footer:
			__copyArrowFooter((ArrowFooter *)dest,
							  (const ArrowFooter *)src);
			break;
		case ArrowNodeTag__BodyCompression:
			__copyArrowBodyCompression((ArrowBodyCompression *)dest,
									   (const ArrowBodyCompression *)src);
			break;
		default:
			Elog("unknown ArrowNodeTag (%d)", (int)src->tag);
	}
}

void
copyArrowNode(ArrowNode *dest, const ArrowNode *src)
{
	char   *emsg = NULL;
	try {
		__copyArrowNode(dest, src);
	}
	catch (const std::exception &e) {
		const char *estr = e.what();
		emsg = (char *)alloca(std::strlen(estr)+1);
		strcpy(emsg, estr);
	}
	/* error report */
	if (emsg)
		ErrorReport(emsg);
}

// ============================================================
//
// Routines to compare two Apache Arrow nodes
//
// ============================================================
static inline bool
__equalArrowNode(const ArrowNode *a, const ArrowNode *b);

static inline bool
__equalBoolean(bool a, bool b)
{
	return (a && b) || (!a && !b);
}

static inline bool
__equalArrowTypeInt(const ArrowTypeInt *a,
					const ArrowTypeInt *b)
{
	return (a->bitWidth == b->bitWidth &&
			__equalBoolean(a->is_signed, b->is_signed));
}

static inline bool
__equalArrowTypeFloatingPoint(const ArrowTypeFloatingPoint *a,
							  const ArrowTypeFloatingPoint *b)
{
	return (a->precision == b->precision);
}

static inline bool
__equalArrowTypeDeciaml(const ArrowTypeDecimal *a,
						const ArrowTypeDecimal *b)
{
	return (a->precision == b->precision &&
			a->scale     == b->scale &&
			a->bitWidth  == b->bitWidth);
}

static inline bool
__equalArrowTypeDate(const ArrowTypeDate *a,
							const ArrowTypeDate *b)
{
	return (a->unit == b->unit);
}

static inline bool
__equalArrowTypeTime(const ArrowTypeTime *a,
					 const ArrowTypeTime *b)
{
	return (a->unit     == b->unit &&
			a->bitWidth == b->bitWidth);
}

static inline bool
__equalArrowTypeTimestamp(const ArrowTypeTimestamp *a,
						  const ArrowTypeTimestamp *b)
{
	if (a->unit == b->unit)
	{
		if (!a->timezone && !b->timezone)
			return true;
		if (a->timezone && b->timezone &&
			std::strcmp(a->timezone, b->timezone) == 0)
			return true;
	}
	return false;
}

static inline bool
__equalArrowTypeInterval(const ArrowTypeInterval *a,
						 const ArrowTypeInterval *b)
{
	return (a->unit == b->unit);
}

static inline bool
__equalArrowTypeUnion(const ArrowTypeUnion *a,
					  const ArrowTypeUnion *b)
{
	if (a->mode == b->mode)
	{
		if (a->_num_typeIds == b->_num_typeIds)
		{
			if (!a->typeIds && !b->typeIds)
				return true;
			assert(a->typeIds && b->typeIds);
			for (int i=0; i < a->_num_typeIds; i++)
			{
				if (a->typeIds[i] != b->typeIds[i])
					return false;
			}
			return true;
		}
	}
	return false;
}

static inline bool
__equalArrowTypeFixedSizeBinary(const ArrowTypeFixedSizeBinary *a,
								const ArrowTypeFixedSizeBinary *b)
{
	return (a->byteWidth == b->byteWidth);
}

static inline bool
__equalArrowTypeFixedSizeList(const ArrowTypeFixedSizeList *a,
							  const ArrowTypeFixedSizeList *b)
{
	return (a->listSize == b->listSize);
}

static inline bool
__equalArrowTypeMap(const ArrowTypeMap *a,
					const ArrowTypeMap *b)
{
	return __equalBoolean(a->keysSorted, b->keysSorted);
}

static inline bool
__equalArrowTypeDuration(const ArrowTypeDuration *a,
						 const ArrowTypeDuration *b)
{
	return (a->unit == b->unit);
}

static inline bool
__equalArrowKeyValue(const ArrowKeyValue *a,
					 const ArrowKeyValue *b)
{
	return (a->_key_len == b->_key_len &&
			a->_value_len == b->_value_len &&
			std::strcmp(a->key, b->key) == 0 &&
			std::strcmp(a->value, b->value) == 0);
}

static inline bool
__equalArrowDictionaryEncoding(const ArrowDictionaryEncoding *a,
							   const ArrowDictionaryEncoding *b)
{
	return (a->id == b->id &&
			__equalArrowNode(&a->indexType.node,
							 &b->indexType.node) &&
			__equalBoolean(a->isOrdered,
						   b->isOrdered));
}

static inline bool
__equalArrowField(const ArrowField *a,
				  const ArrowField *b)
{
	if (a->_name_len == b->_name_len &&
		std::strcmp(a->name, b->name) == 0 &&
		__equalBoolean(a->nullable, b->nullable) &&
		__equalArrowNode(&a->type.node, &b->type.node) &&
		__equalArrowNode((ArrowNode *)a->dictionary,
						 (ArrowNode *)b->dictionary))
	{
		if (a->_num_children != b->_num_children)
			return false;
		for (int i=0; i < a->_num_children; i++)
		{
			if (!__equalArrowNode(&a->children[i].node,
								  &b->children[i].node))
				return false;
		}
		if (a->_num_custom_metadata != b->_num_custom_metadata)
			return false;
		for (int i=0; i < a->_num_custom_metadata; i++)
		{
			if (!__equalArrowNode(&a->custom_metadata[i].node,
								  &b->custom_metadata[i].node))
				return false;
		}
#ifdef HAS_PARQUET
		if (a->parquet.max_definition_level != b->parquet.max_definition_level ||
			a->parquet.max_repetition_level != b->parquet.max_repetition_level ||
			a->parquet.physical_type        != b->parquet.physical_type ||
			a->parquet.converted_type       != b->parquet.converted_type ||
			a->parquet.logical_type         != b->parquet.logical_type ||
			a->parquet.logical_time_unit    != b->parquet.logical_time_unit ||
			a->parquet.logical_decimal_precision != b->parquet.logical_decimal_precision ||
			a->parquet.logical_decimal_scale != b->parquet.logical_decimal_scale)
			return false;
#endif
		return true;
	}
	return false;
}

static inline bool
__equalArrowFieldNode(const ArrowFieldNode *a,
					  const ArrowFieldNode *b)
{
	if (a->length == b->length &&
		a->null_count == b->null_count &&
		a->_stat_min_value_len == b->_stat_min_value_len &&
		(a->_stat_min_value_len == 0 ||
		 memcmp(a->stat_min_value,
				b->stat_min_value, a->_stat_min_value_len) == 0) &&
		a->_stat_max_value_len == b->_stat_max_value_len &&
		(a->_stat_max_value_len == 0 ||
		 memcmp(a->stat_max_value,
				b->stat_max_value, a->_stat_max_value_len) == 0))
	{
#ifdef HAS_PARQUET
		if (a->parquet.total_compressed_size   != b->parquet.total_compressed_size ||
			a->parquet.total_uncompressed_size != b->parquet.total_uncompressed_size ||
			a->parquet.data_page_offset        != b->parquet.data_page_offset ||
			a->parquet.dictionary_page_offset  != b->parquet.dictionary_page_offset ||
			a->parquet.index_page_offset       != b->parquet.index_page_offset ||
			a->parquet.compression_type        != b->parquet.compression_type ||
			a->parquet._num_encodings          != b->parquet._num_encodings ||
			a->parquet._num_encoding_stats     != b->parquet._num_encoding_stats ||
			(a->parquet._num_encodings > 0 &&
			 memcmp(a->parquet.encodings,
					b->parquet.encodings,
					sizeof(int16_t) * a->parquet._num_encodings) != 0) ||
			(a->parquet._num_encoding_stats > 0 &&
			 memcmp(a->parquet.encoding_stats,
					b->parquet.encoding_stats,
					sizeof(ParquetEncodingStats) * a->parquet._num_encoding_stats) != 0))
			return false;
#endif
		return true;
	}
	return false;
}

static inline bool
__equalArrowBuffer(const ArrowBuffer *a,
				   const ArrowBuffer *b)
{
	return (a->offset == b->offset &&
			a->length == b->length);
}

static inline bool
__equalArrowSchema(const ArrowSchema *a,
				   const ArrowSchema *b)
{
	if (a->endianness != b->endianness)
		return false;
	if (a->_num_fields != b->_num_fields)
		return false;
	for (int i=0; i < a->_num_fields; i++)
	{
		if (!__equalArrowNode(&a->fields[i].node,
							  &b->fields[i].node))
			return false;
	}
	if (a->_num_custom_metadata != b->_num_custom_metadata)
		return false;
	for (int i=0; i < a->_num_custom_metadata; i++)
	{
		if (!__equalArrowNode(&a->custom_metadata[i].node,
							  &b->custom_metadata[i].node))
			return false;
	}
	if (a->_num_features != b->_num_features)
		return false;
	for (int i=0; i < a->_num_features; i++)
	{
		if (a->features[i] != b->features[i])
			return false;
	}
	return true;
}

static inline bool
__equalArrowRecordBatch(const ArrowRecordBatch *a,
						const ArrowRecordBatch *b)
{
	if (a->length != b->length)
		return false;
	if (a->_num_nodes != b->_num_nodes)
		return false;
	for (int i=0; i < a->_num_nodes; i++)
	{
		if (!__equalArrowNode(&a->nodes[i].node,
							  &b->nodes[i].node))
			return false;
	}
	if (a->_num_buffers != b->_num_buffers)
		return false;
	for (int i=0; i < a->_num_buffers; i++)
	{
		if (!__equalArrowNode(&a->buffers[i].node,
							  &b->buffers[i].node))
			return false;
	}
	if (!__equalArrowNode((ArrowNode *)a->compression,
						  (ArrowNode *)b->compression))
		return false;
	return true;
}

static inline bool
__equalArrowDictionaryBatch(const ArrowDictionaryBatch *a,
							const ArrowDictionaryBatch *b)
{
	return (a->id == b->id &&
			__equalArrowRecordBatch(&a->data, &b->data) &&
			__equalBoolean(a->isDelta, b->isDelta));
}

static inline bool
__equalArrowMessage(const ArrowMessage *a,
					const ArrowMessage *b)
{
	if (a->version != b->version)
		return false;
	if (!__equalArrowNode(&a->body.node,
						  &b->body.node))
		return false;
	if (a->bodyLength != b->bodyLength)
		return false;
	if (a->_num_custom_metadata != b->_num_custom_metadata)
		return false;
	for (int i=0; i < a->_num_custom_metadata; i++)
	{
		if (!__equalArrowNode(&a->custom_metadata[i].node,
							  &b->custom_metadata[i].node))
			return false;
	}
	return true;
}

static inline bool
__equalArrowBlock(const ArrowBlock *a,
				  const ArrowBlock *b)
{
	return (a->offset         == b->offset &&
			a->metaDataLength == b->metaDataLength &&
			a->bodyLength     == b->bodyLength);

}

static inline bool
__equalArrowFooter(const ArrowFooter *a,
				   const ArrowFooter *b)
{
	if (a->version != b->version ||
		!__equalArrowNode(&a->schema.node,
						  &b->schema.node))
		return false;
	if (a->_num_dictionaries != b->_num_dictionaries)
		return false;
	for (int i=0; i < a->_num_dictionaries; i++)
	{
		if (!__equalArrowNode(&a->dictionaries[i].node,
							  &b->dictionaries[i].node))
			return false;
	}
	if (a->_num_recordBatches != b->_num_recordBatches)
		return false;
	for (int i=0; i < a->_num_recordBatches; i++)
	{
		if (!__equalArrowNode(&a->recordBatches[i].node,
							  &b->recordBatches[i].node))
			return false;
	}
	if (a->_num_custom_metadata != b->_num_custom_metadata)
		return false;
	for (int i=0; i < a->_num_custom_metadata; i++)
	{
		if (!__equalArrowNode(&a->custom_metadata[i].node,
							  &b->custom_metadata[i].node))
			return false;
	}
	return true;
}

static inline bool
__equalArrowBodyCompression(const ArrowBodyCompression *a,
							const ArrowBodyCompression *b)
{
	return (a->codec  == b->codec &&
			a->method == b->method);
}

static bool
__equalArrowNode(const ArrowNode *a, const ArrowNode *b)
{
	if (!a || !b)
		return (!a && !b);
	if (a->tag != b->tag)
		return false;
	switch (a->tag)
	{
		case ArrowNodeTag__Null:
		case ArrowNodeTag__Utf8:
		case ArrowNodeTag__Binary:
		case ArrowNodeTag__Bool:
		case ArrowNodeTag__List:
		case ArrowNodeTag__Struct:
		case ArrowNodeTag__LargeBinary:
		case ArrowNodeTag__LargeUtf8:
		case ArrowNodeTag__LargeList:
			return true;	/* no special properties */
		case ArrowNodeTag__Int:
			return __equalArrowTypeInt((const ArrowTypeInt *)a,
									   (const ArrowTypeInt *)b);
		case ArrowNodeTag__FloatingPoint:
			return __equalArrowTypeFloatingPoint((const ArrowTypeFloatingPoint *)a,
												 (const ArrowTypeFloatingPoint *)b);
		case ArrowNodeTag__Decimal:
			return __equalArrowTypeDeciaml((const ArrowTypeDecimal *)a,
										   (const ArrowTypeDecimal *)b);
		case ArrowNodeTag__Date:
			return __equalArrowTypeDate((const ArrowTypeDate *)a,
										(const ArrowTypeDate *)b);
		case ArrowNodeTag__Time:
			return __equalArrowTypeTime((const ArrowTypeTime *)a,
										(const ArrowTypeTime *)b);
		case ArrowNodeTag__Timestamp:
			return __equalArrowTypeTimestamp((const ArrowTypeTimestamp *)a,
											 (const ArrowTypeTimestamp *)b);
		case ArrowNodeTag__Interval:
			return __equalArrowTypeInterval((const ArrowTypeInterval *)a,
											(const ArrowTypeInterval *)b);
		case ArrowNodeTag__Union:
			return __equalArrowTypeUnion((const ArrowTypeUnion *)a,
										 (const ArrowTypeUnion *)b);
		case ArrowNodeTag__FixedSizeBinary:
			return __equalArrowTypeFixedSizeBinary((const ArrowTypeFixedSizeBinary *)a,
												   (const ArrowTypeFixedSizeBinary *)b);
		case ArrowNodeTag__FixedSizeList:
			return __equalArrowTypeFixedSizeList((const ArrowTypeFixedSizeList *)a,
												 (const ArrowTypeFixedSizeList *)b);
		case ArrowNodeTag__Map:
			return __equalArrowTypeMap((const ArrowTypeMap *)a,
									   (const ArrowTypeMap *)b);
		case ArrowNodeTag__Duration:
			return __equalArrowTypeDuration((const ArrowTypeDuration *)a,
											(const ArrowTypeDuration *)b);
			/* others */
		case ArrowNodeTag__KeyValue:
			return __equalArrowKeyValue((const ArrowKeyValue *)a,
										(const ArrowKeyValue *)b);
		case ArrowNodeTag__DictionaryEncoding:
			return __equalArrowDictionaryEncoding((const ArrowDictionaryEncoding *)a,
												  (const ArrowDictionaryEncoding *)b);
		case ArrowNodeTag__Field:
			return __equalArrowField((const ArrowField *)a,
									 (const ArrowField *)b);
		case ArrowNodeTag__FieldNode:
			return __equalArrowFieldNode((const ArrowFieldNode *)a,
										 (const ArrowFieldNode *)b);
		case ArrowNodeTag__Buffer:
			return __equalArrowBuffer((const ArrowBuffer *)a,
									  (const ArrowBuffer *)b);
		case ArrowNodeTag__Schema:
			return __equalArrowSchema((const ArrowSchema *)a,
									  (const ArrowSchema *)b);
		case ArrowNodeTag__RecordBatch:
			return __equalArrowRecordBatch((const ArrowRecordBatch *)a,
										   (const ArrowRecordBatch *)b);
		case ArrowNodeTag__DictionaryBatch:
			return __equalArrowDictionaryBatch((const ArrowDictionaryBatch *)a,
											   (const ArrowDictionaryBatch *)b);
		case ArrowNodeTag__Message:
			return __equalArrowMessage((const ArrowMessage *)a,
									   (const ArrowMessage *)b);
		case ArrowNodeTag__Block:
			return __equalArrowBlock((const ArrowBlock *)a,
									 (const ArrowBlock *)b);
		case ArrowNodeTag__Footer:
			return __equalArrowFooter((const ArrowFooter *)a,
									  (const ArrowFooter *)b);
		case ArrowNodeTag__BodyCompression:
			return __equalArrowBodyCompression((const ArrowBodyCompression *)a,
											   (const ArrowBodyCompression *)b);
		default:
			Elog("unknown ArrowNodeTag (%d)", (int)a->tag);
	}
	return false;
}

bool
equalArrowNode(const ArrowNode *a, const ArrowNode *b)
{
	char   *emsg = NULL;
	bool	rv;
	try {
		rv = __equalArrowNode(a, b);
	}
	catch (const std::exception &e) {
		const char *estr = e.what();
		emsg = (char *)alloca(std::strlen(estr)+1);
		strcpy(emsg, estr);
	}
	/* error report */
	if (emsg)
		ErrorReport(emsg);
	return rv;
}

// ============================================================
//
// Routines to initialize Apache Arrow nodes
//
// ============================================================
#define __INIT_ARROW_NODE(PTR,TYPENAME,NAME)                \
	do {                                                    \
		memset((PTR),0,sizeof(Arrow##TYPENAME));			\
		((ArrowNode *)(PTR))->tag = ArrowNodeTag__##NAME;   \
		((ArrowNode *)(PTR))->tagName = #NAME;              \
	} while(0)
#define INIT_ARROW_NODE(PTR,NAME)		__INIT_ARROW_NODE(PTR,NAME,NAME)
#define INIT_ARROW_TYPE_NODE(PTR,NAME)	__INIT_ARROW_NODE(PTR,Type##NAME,NAME)

void
__initArrowNode(ArrowNode *node, ArrowNodeTag tag)
{
#define CASE_ARROW_NODE(NAME)                       \
	case ArrowNodeTag__##NAME:                      \
		INIT_ARROW_NODE(node,NAME);                 \
		break
#define CASE_ARROW_TYPE_NODE(NAME)                  \
	case ArrowNodeTag__##NAME:                      \
		INIT_ARROW_TYPE_NODE(node,NAME);            \
		break

	switch (tag)
	{
		CASE_ARROW_TYPE_NODE(Null);
		CASE_ARROW_TYPE_NODE(Int);
		CASE_ARROW_TYPE_NODE(FloatingPoint);
		CASE_ARROW_TYPE_NODE(Utf8);
		CASE_ARROW_TYPE_NODE(Binary);
		CASE_ARROW_TYPE_NODE(Bool);
		CASE_ARROW_TYPE_NODE(Decimal);
		CASE_ARROW_TYPE_NODE(Date);
		CASE_ARROW_TYPE_NODE(Time);
		CASE_ARROW_TYPE_NODE(Timestamp);
		CASE_ARROW_TYPE_NODE(Interval);
		CASE_ARROW_TYPE_NODE(List);
		CASE_ARROW_TYPE_NODE(Struct);
		CASE_ARROW_TYPE_NODE(Union);
		CASE_ARROW_TYPE_NODE(FixedSizeBinary);
		CASE_ARROW_TYPE_NODE(FixedSizeList);
		CASE_ARROW_TYPE_NODE(Map);
		CASE_ARROW_TYPE_NODE(Duration);
		CASE_ARROW_TYPE_NODE(LargeBinary);
		CASE_ARROW_TYPE_NODE(LargeUtf8);
		CASE_ARROW_TYPE_NODE(LargeList);

		CASE_ARROW_NODE(KeyValue);
		CASE_ARROW_NODE(DictionaryEncoding);
		CASE_ARROW_NODE(Field);
		CASE_ARROW_NODE(FieldNode);
		CASE_ARROW_NODE(Buffer);
		CASE_ARROW_NODE(Schema);
		CASE_ARROW_NODE(RecordBatch);
		CASE_ARROW_NODE(DictionaryBatch);
		CASE_ARROW_NODE(Message);
		CASE_ARROW_NODE(Block);
		CASE_ARROW_NODE(Footer);
		default:
			Elog("unknown ArrowNodeTag: %d", tag);
	}
#undef CASE_ARROW_NODE
#undef CASE_ARROW_TYPE_NODE
}

// ============================================================
//
// readArrowFileInfo
//
// ============================================================

static void
readArrowBlock(ArrowBlock *node,
			   const org::apache::arrow::flatbuf::Block *block)
{
	INIT_ARROW_NODE(node, Block);
	node->offset         = block->offset();
	node->metaDataLength = block->metaDataLength();
	node->bodyLength     = block->bodyLength();
}

static void
readArrowKeyValue(ArrowKeyValue *node,
				  const std::string &key,
				  const std::string &value)
{
	INIT_ARROW_NODE(node, KeyValue);
	node->key = __pstrdup(key.c_str());
	node->_key_len = std::strlen(node->key);
	node->value = __pstrdup(value.c_str());
	node->_value_len = std::strlen(node->value);
}

static void
readArrowFieldNode(ArrowFieldNode *node,
				   const org::apache::arrow::flatbuf::FieldNode *fnode)
{
	INIT_ARROW_NODE(node, FieldNode);
	node->length     = fnode->length();
	node->null_count = fnode->null_count();
	/* max_value/min_value shall be set later */
#ifdef HAS_PARQUET
	node->parquet.compression_type = ArrowCompressionType__UNKNOWN;
#endif
}

static void
readArrowBuffer(ArrowBuffer *node,
				const org::apache::arrow::flatbuf::Buffer *buffer)
{
	INIT_ARROW_NODE(node, Buffer);
	node->offset = buffer->offset();
	node->length = buffer->length();
}

static void
readArrowTypeInt(ArrowTypeInt *node,
				  const org::apache::arrow::flatbuf::Int *__type)
{
	INIT_ARROW_TYPE_NODE(node, Int);
	node->bitWidth  = __type->bitWidth();
	node->is_signed = __type->is_signed();
}

static void
readArrowTypeFloatingPoint(ArrowTypeFloatingPoint *node,
						   const org::apache::arrow::flatbuf::FloatingPoint *__type)
{
	INIT_ARROW_TYPE_NODE(node, FloatingPoint);
	switch (__type->precision())
	{
		case org::apache::arrow::flatbuf::Precision_HALF:
			node->precision = ArrowPrecision__Half;
			break;
		case org::apache::arrow::flatbuf::Precision_SINGLE:
			node->precision = ArrowPrecision__Single;
			break;
		case org::apache::arrow::flatbuf::Precision_DOUBLE:
			node->precision = ArrowPrecision__Double;
			break;
		default:
			Elog("unknown FloatingPoint precision (%d)", (int)__type->precision());
			break;
	}
}

static void
readArrowTypeDecimal(ArrowTypeDecimal *node,
					 const org::apache::arrow::flatbuf::Decimal *__type)
{
	INIT_ARROW_TYPE_NODE(node, Decimal);
	node->precision = __type->precision();
	node->scale     = __type->scale();
	node->bitWidth  = __type->bitWidth();
	if (node->bitWidth != 128 && node->bitWidth != 256)
		Elog("unknown bitWidth (%d) for Arrow::Decimal", node->bitWidth);
}

static void
readArrowTypeDate(ArrowTypeDate *node,
				  const org::apache::arrow::flatbuf::Date *__type)
{
	INIT_ARROW_TYPE_NODE(node, Date);
	switch (__type->unit())
	{
		case org::apache::arrow::flatbuf::DateUnit_DAY:
			node->unit = ArrowDateUnit__Day;
			break;
		case org::apache::arrow::flatbuf::DateUnit_MILLISECOND:
			node->unit = ArrowDateUnit__MilliSecond;
			break;
		default:
			Elog("unknown Arrow::Date unit (%d)", (int)__type->unit());
			break;
	}
}

static void
readArrowTypeTime(ArrowTypeTime *node,
				  const org::apache::arrow::flatbuf::Time *__type)
{
	INIT_ARROW_TYPE_NODE(node, Time);
	switch (__type->unit())
	{
		case org::apache::arrow::flatbuf::TimeUnit_SECOND:
			node->unit = ArrowTimeUnit__Second;
			break;
		case org::apache::arrow::flatbuf::TimeUnit_MILLISECOND:
			node->unit = ArrowTimeUnit__MilliSecond;
			break;
		case org::apache::arrow::flatbuf::TimeUnit_MICROSECOND:
			node->unit = ArrowTimeUnit__MicroSecond;
			break;
		case org::apache::arrow::flatbuf::TimeUnit_NANOSECOND:
			node->unit = ArrowTimeUnit__NanoSecond;
			break;
		default:
			Elog("unknown Arrow::Time unit (%d)", (int)__type->unit());
			break;
	}
	node->bitWidth = __type->bitWidth();
	if (node->bitWidth != 32 && node->bitWidth != 64)
		Elog("unknown Arrow::Time unit bitWidth (%d)", node->bitWidth);
}

static void
readArrowTypeTimestamp(ArrowTypeTimestamp *node,
					   const org::apache::arrow::flatbuf::Timestamp *__type)
{
	INIT_ARROW_TYPE_NODE(node, Timestamp);
	switch (__type->unit())
	{
		case org::apache::arrow::flatbuf::TimeUnit_SECOND:
			node->unit = ArrowTimeUnit__Second;
			break;
		case org::apache::arrow::flatbuf::TimeUnit_MILLISECOND:
			node->unit = ArrowTimeUnit__MilliSecond;
			break;
		case org::apache::arrow::flatbuf::TimeUnit_MICROSECOND:
			node->unit = ArrowTimeUnit__MicroSecond;
			break;
		case org::apache::arrow::flatbuf::TimeUnit_NANOSECOND:
			node->unit = ArrowTimeUnit__NanoSecond;
			break;
		default:
			Elog("unknown Arrow::Timestamp unit (%d)", (int)__type->unit());
			break;
	}
	auto	__timezone = __type->timezone();
	if (__timezone)
	{
		node->timezone = __pstrdup(__timezone);
		node->_timezone_len = __timezone->size();
	}
}

static void
readArrowTypeInterval(ArrowTypeInterval *node,
					  const org::apache::arrow::flatbuf::Interval *__type)
{
	INIT_ARROW_TYPE_NODE(node, Interval);
	switch (__type->unit())
	{
		case org::apache::arrow::flatbuf::IntervalUnit_YEAR_MONTH:
			node->unit = ArrowIntervalUnit__Year_Month;
			break;
		case org::apache::arrow::flatbuf::IntervalUnit_DAY_TIME:
			node->unit = ArrowIntervalUnit__Day_Time;
			break;
		case org::apache::arrow::flatbuf::IntervalUnit_MONTH_DAY_NANO:
			node->unit = ArrowIntervalUnit__Month_Day_Nano;
			break;
		default:
			Elog("unknown Arrow::Interval unit (%d)", (int)__type->unit());
	}
}

static void
readArrowTypeUnion(ArrowTypeUnion *node,
				   const org::apache::arrow::flatbuf::Union *__type)
{
	INIT_ARROW_TYPE_NODE(node, Union);
	switch (__type->mode())
	{
		case org::apache::arrow::flatbuf::UnionMode_Sparse:
			node->mode = ArrowUnionMode__Sparse;
			break;
		case org::apache::arrow::flatbuf::UnionMode_Dense:
			node->mode = ArrowUnionMode__Dense;
			break;
		default:
			Elog("unknown Arrow::Union mode (%d)", (int)__type->mode());
	}

	auto	__typeIds = __type->typeIds();
	if (__typeIds && __typeIds->size() > 0)
	{
		node->_num_typeIds = __typeIds->size();
		node->typeIds = (int32_t *)__palloc(sizeof(int32_t) * node->_num_typeIds);
		for (int i=0; i < node->_num_typeIds; i++)
			node->typeIds[i] = (*__typeIds)[i];
	}
}

static void
readArrowTypeFixedSizeBinary(ArrowTypeFixedSizeBinary *node,
							 const org::apache::arrow::flatbuf::FixedSizeBinary *__type)
{
	INIT_ARROW_TYPE_NODE(node,FixedSizeBinary);
	node->byteWidth = __type->byteWidth();
}

static void
readArrowTypeFixedSizeList(ArrowTypeFixedSizeList *node,
						   const org::apache::arrow::flatbuf::FixedSizeList *__type)
{
	INIT_ARROW_TYPE_NODE(node,FixedSizeList);
	node->listSize = __type->listSize();
}

static void
readArrowTypeMap(ArrowTypeMap *node,
				 const org::apache::arrow::flatbuf::Map *__type)
{
	INIT_ARROW_TYPE_NODE(node,Map);
	node->keysSorted = __type->keysSorted();
}

static void
readArrowTypeDuration(ArrowTypeDuration *node,
					  const org::apache::arrow::flatbuf::Duration *__type)
{
	INIT_ARROW_TYPE_NODE(node,Duration);
	switch (__type->unit())
	{
		case org::apache::arrow::flatbuf::TimeUnit_SECOND:
			node->unit = ArrowTimeUnit__Second;
			break;
		case org::apache::arrow::flatbuf::TimeUnit_MILLISECOND:
			node->unit = ArrowTimeUnit__MilliSecond;
			break;
		case org::apache::arrow::flatbuf::TimeUnit_MICROSECOND:
			node->unit = ArrowTimeUnit__MicroSecond;
			break;
		case org::apache::arrow::flatbuf::TimeUnit_NANOSECOND:
			node->unit = ArrowTimeUnit__NanoSecond;
			break;
		default:
			Elog("unknown Arrow::Duration unit (%d)", (int)__type->unit());
			break;
	}
}

static void
readArrowDictionaryEncoding(ArrowDictionaryEncoding *node,
							const org::apache::arrow::flatbuf::DictionaryEncoding *dict)
{
	INIT_ARROW_NODE(node, DictionaryEncoding);
	node->id = dict->id();
	readArrowTypeInt(&node->indexType, dict->indexType());
	node->isOrdered = dict->isOrdered();
	dict->dictionaryKind();

}

static void
readArrowField(ArrowField *node,
			   const org::apache::arrow::flatbuf::Field *field)
{
	INIT_ARROW_NODE(node, Field);
	node->name = __pstrdup(field->name());
	node->_name_len = std::strlen(node->name);
	node->nullable = field->nullable();
	switch (field->type_type())
	{
		case org::apache::arrow::flatbuf::Type_Null:
			INIT_ARROW_TYPE_NODE(&node->type, Null);
			break;
		case org::apache::arrow::flatbuf::Type_Int:
			readArrowTypeInt(&node->type.Int,
							 field->type_as_Int());
			break;
		case org::apache::arrow::flatbuf::Type_FloatingPoint:
			readArrowTypeFloatingPoint(&node->type.FloatingPoint,
									   field->type_as_FloatingPoint());
			break;
		case org::apache::arrow::flatbuf::Type_Binary:
			INIT_ARROW_TYPE_NODE(&node->type, Binary);
			break;
		case org::apache::arrow::flatbuf::Type_Utf8:
			INIT_ARROW_TYPE_NODE(&node->type, Utf8);
			break;
		case org::apache::arrow::flatbuf::Type_Bool:
			INIT_ARROW_TYPE_NODE(&node->type, Bool);
			break;
		case org::apache::arrow::flatbuf::Type_Decimal:
			readArrowTypeDecimal(&node->type.Decimal,
								 field->type_as_Decimal());
			break;
		case org::apache::arrow::flatbuf::Type_Date:
			readArrowTypeDate(&node->type.Date,
							  field->type_as_Date());
			break;
		case org::apache::arrow::flatbuf::Type_Time:
			readArrowTypeTime(&node->type.Time,
							  field->type_as_Time());
			break;
		case org::apache::arrow::flatbuf::Type_Timestamp:
			readArrowTypeTimestamp(&node->type.Timestamp,
								   field->type_as_Timestamp());
			break;
		case org::apache::arrow::flatbuf::Type_Interval:
			readArrowTypeInterval(&node->type.Interval,
								  field->type_as_Interval());
			break;
		case org::apache::arrow::flatbuf::Type_List:
			INIT_ARROW_TYPE_NODE(&node->type, List);
			break;
		case org::apache::arrow::flatbuf::Type_Struct_:
			INIT_ARROW_TYPE_NODE(&node->type, Struct);
			break;
		case org::apache::arrow::flatbuf::Type_Union:
			readArrowTypeUnion(&node->type.Union,
							   field->type_as_Union());
			break;
		case org::apache::arrow::flatbuf::Type_FixedSizeBinary:
			readArrowTypeFixedSizeBinary(&node->type.FixedSizeBinary,
										 field->type_as_FixedSizeBinary());
			break;
		case org::apache::arrow::flatbuf::Type_FixedSizeList:
			readArrowTypeFixedSizeList(&node->type.FixedSizeList,
									   field->type_as_FixedSizeList());
			break;
		case org::apache::arrow::flatbuf::Type_Map:
			readArrowTypeMap(&node->type.Map,
							 field->type_as_Map());
			break;
		case org::apache::arrow::flatbuf::Type_Duration:
			readArrowTypeDuration(&node->type.Duration,
								  field->type_as_Duration());
			break;
		case org::apache::arrow::flatbuf::Type_LargeBinary:
			INIT_ARROW_TYPE_NODE(&node->type, LargeBinary);
			break;
		case org::apache::arrow::flatbuf::Type_LargeUtf8:
			INIT_ARROW_TYPE_NODE(&node->type, LargeUtf8);
			break;
		case org::apache::arrow::flatbuf::Type_LargeList:
			INIT_ARROW_TYPE_NODE(&node->type, LargeList);
			break;
		default:
			Elog("unknown org::apache::arrow::flatbuf::Type_XXX (%d)",
				 (int)field->type_type());
	}

	auto __dictionary = field->dictionary();
	if (__dictionary)
	{
		node->dictionary = (ArrowDictionaryEncoding *)
			__palloc(sizeof(ArrowDictionaryEncoding));
		readArrowDictionaryEncoding(node->dictionary, __dictionary);
	}

	auto __children = field->children();
	if ( __children && __children->size() > 0)
	{
		node->_num_children = __children->size();
		node->children = (ArrowField *)
			__palloc(sizeof(ArrowField) * node->_num_children);
		for (int i=0; i < node->_num_children; i++)
			readArrowField(&node->children[i], (*__children)[i]);
	}

	auto __custom_metadata = field->custom_metadata();
	if (__custom_metadata && __custom_metadata->size() > 0)
	{
		node->_num_custom_metadata = __custom_metadata->size();
		node->custom_metadata = (ArrowKeyValue *)
			__palloc(sizeof(ArrowKeyValue) * node->_num_custom_metadata);
		for (int i=0; i < node->_num_custom_metadata; i++)
		{
			auto	__kv = (*__custom_metadata)[i];
			readArrowKeyValue(&node->custom_metadata[i],
							  __kv->key()->str(),
							  __kv->value()->str());
		}
	}
#if HAS_PARQUET
	node->parquet.physical_type = parquet::Type::UNDEFINED;
	node->parquet.converted_type = parquet::ConvertedType::UNDEFINED;
	node->parquet.logical_type = parquet::LogicalType::Type::UNDEFINED;
#endif
}

static void
readArrowSchemaMessage(ArrowSchema *node,
					   const org::apache::arrow::flatbuf::Schema *schema)
{
	INIT_ARROW_NODE(node, Schema);
	switch (schema->endianness())
	{
		case org::apache::arrow::flatbuf::Endianness_Little:
			node->endianness = ArrowEndianness__Little;
			break;
		case org::apache::arrow::flatbuf::Endianness_Big:
			node->endianness = ArrowEndianness__Big;
			break;
		default:
			Elog("unknown Endianness (%d)", (int)schema->endianness());
			break;
	}

	auto	__fields = schema->fields();
	if (__fields && __fields->size() > 0)
	{
		node->_num_fields = __fields->size();
		node->fields = (ArrowField *)
			__palloc(sizeof(ArrowField) * node->_num_fields);
		for (int i=0; i < node->_num_fields; i++)
			readArrowField(&node->fields[i], (*__fields)[i]);
	}

	auto	__custom_metadata = schema->custom_metadata();
	if (__custom_metadata && __custom_metadata->size() > 0)
	{
		node->_num_custom_metadata = __custom_metadata->size();
		node->custom_metadata = (ArrowKeyValue *)
			__palloc(sizeof(ArrowKeyValue) * node->_num_custom_metadata);
		for (int i=0; i < node->_num_custom_metadata; i++)
		{
			auto	__kv = (*__custom_metadata)[i];
			readArrowKeyValue(&node->custom_metadata[i],
							  __kv->key()->str(),
							  __kv->value()->str());
		}
	}

	auto	__features = schema->features();
	if (__features && __features->size() > 0)
	{
		node->_num_features = __features->size();
		node->features = (ArrowFeature *)
			__palloc(sizeof(ArrowFeature) * node->_num_features);
		for (int i=0; i < node->_num_features; i++)
		{
			auto	feature_id = (*__features)[i];

			switch (org::apache::arrow::flatbuf::EnumValuesFeature()[feature_id])
			{
				case org::apache::arrow::flatbuf::Feature_UNUSED:
					node->features[i] = ArrowFeature__Unused;
					break;
				case org::apache::arrow::flatbuf::Feature_DICTIONARY_REPLACEMENT:
					node->features[i] = ArrowFeature__DictionaryReplacement;
					break;
				case org::apache::arrow::flatbuf::Feature_COMPRESSED_BODY:
					node->features[i] = ArrowFeature__CompressedBody;
					break;
				default:
					Elog("unknown Feature (%ld)", feature_id);
			}
		}
	}
}

static void
readArrowBodyCompression(ArrowBodyCompression *node,
						 const org::apache::arrow::flatbuf::BodyCompression *compression)
{
	INIT_ARROW_NODE(node,BodyCompression);
	switch (compression->codec())
	{
		case org::apache::arrow::flatbuf::CompressionType_LZ4_FRAME:
			node->codec = ArrowCompressionType__LZ4_FRAME;
			break;
		case org::apache::arrow::flatbuf::CompressionType_ZSTD:
			node->codec = ArrowCompressionType__ZSTD;
			break;
		default:
			Elog("unknown CompressionType (%d)",
				 (int)compression->codec());
			break;
	}
	switch (compression->method())
	{
		case org::apache::arrow::flatbuf::BodyCompressionMethod_BUFFER:
			node->method = ArrowBodyCompressionMethod__BUFFER;
			break;
		default:
			Elog("unknown BodyCompressionMethod (%d)",
				 (int)compression->method());
			break;
	}
}

static void
readArrowRecordBatchMessage(ArrowRecordBatch *node,
							const org::apache::arrow::flatbuf::RecordBatch *rbatch)
{
	INIT_ARROW_NODE(node,RecordBatch);
	node->length = rbatch->length();

	auto	__nodes = rbatch->nodes();
	if (__nodes && __nodes->size() > 0)
	{
		node->_num_nodes = __nodes->size();
		node->nodes = (ArrowFieldNode *)
			__palloc(sizeof(ArrowFieldNode) * node->_num_nodes);
		for (int i=0; i < node->_num_nodes; i++)
			readArrowFieldNode(&node->nodes[i], (*__nodes)[i]);
	}

	auto	__buffers = rbatch->buffers();
	if (__buffers && __buffers->size() > 0)
	{
		node->_num_buffers = __buffers->size();
		node->buffers = (ArrowBuffer *)
			__palloc(sizeof(ArrowBuffer) * node->_num_buffers);
		for (int i=0; i < node->_num_buffers; i++)
			readArrowBuffer(&node->buffers[i], (*__buffers)[i]);
	}

	auto	__compression = rbatch->compression();
	if (__compression)
	{
		node->compression = (ArrowBodyCompression *)
			__palloc(sizeof(ArrowBodyCompression));
		readArrowBodyCompression(node->compression, __compression);
	}
}

static void
readArrowDictionaryBatchMessage(ArrowDictionaryBatch *node,
								const org::apache::arrow::flatbuf::DictionaryBatch *dbatch)
{
	INIT_ARROW_NODE(node,DictionaryBatch);
	node->id = dbatch->id();
	readArrowRecordBatchMessage(&node->data, dbatch->data());
	node->isDelta = dbatch->isDelta();
}

static void
readArrowMessageBlock(ArrowMessage *node,
					  std::shared_ptr<arrow::io::ReadableFile> rfilp,
					  const org::apache::arrow::flatbuf::Block *block)
{
	auto	rv = rfilp->ReadAt(block->offset(),
							   block->metaDataLength());
	if (!rv.ok())
		Elog("failed on arrow::io::ReadableFile::ReadAt: %s",
			 rv.status().ToString().c_str());
	auto	buffer = rv.ValueOrDie();
	auto	fb_base = buffer->data() + sizeof(uint32_t);	/* Continuation token (0xffffffff) */
	auto	message = org::apache::arrow::flatbuf::GetSizePrefixedMessage(fb_base);
	auto	custom_metadata = message->custom_metadata();

	INIT_ARROW_NODE(node, Message);
	switch (message->header_type())
	{
		case org::apache::arrow::flatbuf::MessageHeader_Schema: {
			auto	__schema = message->header_as_Schema();
			readArrowSchemaMessage(&node->body.schema, __schema);
			break;
		}
		case org::apache::arrow::flatbuf::MessageHeader_DictionaryBatch: {
			auto	__dbatch = message->header_as_DictionaryBatch();
			readArrowDictionaryBatchMessage(&node->body.dictionaryBatch, __dbatch);
			break;
		}
		case org::apache::arrow::flatbuf::MessageHeader_RecordBatch: {
			auto	__rbatch = message->header_as_RecordBatch();
			readArrowRecordBatchMessage(&node->body.recordBatch, __rbatch);
			break;
		}
		case org::apache::arrow::flatbuf::MessageHeader_Tensor:
			Elog("MessageHeader::Tensor is not implemented right now");
		case org::apache::arrow::flatbuf::MessageHeader_SparseTensor:
			Elog("MessageHeader::SparseTensor is not implemented right now");
		default:
			Elog("corrupted arrow file? unknown message type %d",
				 (int)message->header_type());
	}
	if (custom_metadata && custom_metadata->size() > 0)
	{
		node->_num_custom_metadata = custom_metadata->size();
		node->custom_metadata = (ArrowKeyValue *)
			__palloc(sizeof(ArrowKeyValue) * node->_num_custom_metadata);
		for (int i=0; i < node->_num_custom_metadata; i++)
		{
			auto	kv = (*custom_metadata)[i];
			readArrowKeyValue(&node->custom_metadata[i],
							  kv->key()->str(),
							  kv->value()->str());
		}
	}
}

static void
readArrowFooter(ArrowFooter *node,
				const org::apache::arrow::flatbuf::Footer *footer)
{
	auto	__version = footer->version();
	auto	__schema = footer->schema();
	auto	__dictionaries = footer->dictionaries();
	auto	__record_batches = footer->recordBatches();
	auto	__custom_metadata = footer->custom_metadata();

	INIT_ARROW_NODE(node, Footer);
	/* extract version */
	switch (__version)
	{
		case org::apache::arrow::flatbuf::MetadataVersion_V1:
			node->version = ArrowMetadataVersion__V1;
			break;
		case org::apache::arrow::flatbuf::MetadataVersion_V2:
			node->version = ArrowMetadataVersion__V2;
			break;
		case org::apache::arrow::flatbuf::MetadataVersion_V3:
			node->version = ArrowMetadataVersion__V3;
			break;
		case org::apache::arrow::flatbuf::MetadataVersion_V4:
			node->version = ArrowMetadataVersion__V4;
			break;
		case org::apache::arrow::flatbuf::MetadataVersion_V5:
			node->version = ArrowMetadataVersion__V5;
			break;
		default:
			Elog("unknown Apache Arroe metadata version: %d", (int)__version);
	}
	/* extract schema */
	readArrowSchemaMessage(&node->schema, __schema);
	/* extract dictionary batch blocks */
	if (__dictionaries && __dictionaries->size() > 0)
	{
		node->dictionaries = (ArrowBlock *)
			__palloc(sizeof(ArrowBlock) * __dictionaries->size());
		for (uint32_t i=0; i < __dictionaries->size(); i++)
		{
			readArrowBlock(&node->dictionaries[i],
						   (*__dictionaries)[i]);
		}
		node->_num_dictionaries = __dictionaries->size();
	}
	/* extract record-batch */
	if (__record_batches && __record_batches->size() > 0)
	{
		node->recordBatches = (ArrowBlock *)
			__palloc(sizeof(ArrowBlock) * __record_batches->size());
		for (uint32_t i=0; i < __record_batches->size(); i++)
		{
			readArrowBlock(&node->recordBatches[i],
						   (*__record_batches)[i]);
		}
		node->_num_recordBatches = __record_batches->size();
	}
	/* extract key-value pairs */
	if (__custom_metadata && __custom_metadata->size() > 0)
	{
		node->custom_metadata = (ArrowKeyValue *)
			__palloc(sizeof(ArrowKeyValue) * node->_num_custom_metadata);
		for (uint32_t i=0; i < __custom_metadata->size(); i++)
		{
			auto	__kv = (*__custom_metadata)[i];
			readArrowKeyValue(&node->custom_metadata[i],
							  __kv->key()->str(),
							  __kv->value()->str());
		}
		node->_num_custom_metadata = __custom_metadata->size();
	}
}

/*
 * __readArrowFileMetadata
 */
static void
__readArrowFileMetadata(std::shared_ptr<arrow::io::ReadableFile> rfilp,
						ArrowFileInfo *af_info)
{
	
	auto	file_sz = rfilp->GetSize().ValueOrDie();
	auto	tail_sz = sizeof(uint32_t) + ARROW_SIGNATURE_SZ;
	int32_t	footer_sz;

	/* validate arrow file tail */
	{
		auto	rv = rfilp->ReadAt(file_sz - tail_sz, tail_sz);
		if (!rv.ok())
			Elog("failed on arrow::io::ReadableFile::ReadAt: %s",
				 rv.status().ToString().c_str());
		auto	buffer = rv.ValueOrDie();

		if (memcmp((char *)buffer->data() + sizeof(uint32_t),
				   ARROW_SIGNATURE,
				   ARROW_SIGNATURE_SZ) != 0)
			Elog("arrow: signature check failed");
		footer_sz = *((int32_t *)buffer->data());
	}
	/* read the footer flat-buffer */
	{
		auto	rv = rfilp->ReadAt(file_sz
								   - tail_sz
								   - footer_sz, footer_sz);
		if (!rv.ok())
			Elog("failed on arrow::io::ReadableFile::ReadAt: %s",
				 rv.status().ToString().c_str());
		auto	buffer = rv.ValueOrDie();
		auto	footer = org::apache::arrow::flatbuf::GetFooter(buffer->data());

		/* extract Footer */
		readArrowFooter(&af_info->footer, footer);

		/* extract DictionaryBatch message*/
		auto	__dictionaries = footer->dictionaries();
		if (__dictionaries && __dictionaries->size() > 0)
		{
			af_info->dictionaries = (ArrowMessage *)
				__palloc(sizeof(ArrowMessage) * __dictionaries->size());
			for (uint32_t i=0; i < __dictionaries->size(); i++)
			{
				auto	block = (*__dictionaries)[i];
				readArrowMessageBlock(&af_info->dictionaries[i],
									  rfilp, block);
			}
			af_info->_num_dictionaries = __dictionaries->size();
		}
		/* extract RecordBatch message */
		auto	__record_batches = footer->recordBatches();
		if (__record_batches && __record_batches->size() > 0)
		{
			af_info->recordBatches = (ArrowMessage *)
				__palloc(sizeof(ArrowMessage) * __record_batches->size());
			for (uint32_t i=0; i < __record_batches->size(); i++)
			{
				auto	block = (*__record_batches)[i];
				readArrowMessageBlock(&af_info->recordBatches[i],
									  rfilp, block);
			}
			af_info->_num_recordBatches = __record_batches->size();
		}
		/*
		 * extract min/max statistics embedded by pg2arrow
		 * (not a standard feature)
		 */
		const auto schema = &af_info->footer.schema;
		for (int j=0; j < schema->_num_fields; j++)
		{
			const ArrowField *field = &schema->fields[j];
			const char *min_values = NULL;
			const char *max_values = NULL;

			for (int k=0; k < field->_num_custom_metadata; k++)
			{
				const ArrowKeyValue *kv = &field->custom_metadata[k];

				if (std::strcmp(kv->key, "min_values") == 0)
					min_values = kv->value;
				else if (std::strcmp(kv->key, "max_values") == 0)
					max_values = kv->value;
			}
			if (min_values && max_values)
			{
				char   *min_buffer = (char *)alloca(std::strlen(min_values)+1);
				char   *max_buffer = (char *)alloca(std::strlen(max_values)+1);
				char   *tok1, *pos1;	/* for min values */
				char   *tok2, *pos2;	/* for max values */
				int		rb_index;

				strcpy(min_buffer, min_values);
				strcpy(max_buffer, max_values);
				for (tok1 = strtok_r(min_buffer, ",", &pos1),
					 tok2 = strtok_r(max_buffer, ",", &pos2),
					 rb_index = 0;
					 tok1 != NULL && tok2 != NULL &&
					 rb_index < af_info->_num_recordBatches;
					 tok1 = strtok_r(NULL, ",", &pos1),
					 tok2 = strtok_r(NULL, ",", &pos2),
					 rb_index++)
				{
					auto rbatch = &af_info->recordBatches[rb_index].body.recordBatch;
					assert(ArrowNodeIs(rbatch, RecordBatch));
					auto fnode = &rbatch->nodes[j];

					fnode->stat_min_value = __pstrdup(tok1);
					fnode->_stat_min_value_len = std::strlen(fnode->stat_min_value);
					fnode->stat_max_value = __pstrdup(tok2);
					fnode->_stat_max_value_len = std::strlen(fnode->stat_max_value);
				}
			}
		}
	}
}

#ifdef HAS_PARQUET
// ============================================================
//
// Routines to parse Parquet File metadata
//
// ============================================================

static inline ArrowDateUnit
__transformArrowDateUnit(arrow::DateUnit unit)
{
	switch (unit)
	{
		case arrow::DateUnit::DAY:
			return ArrowDateUnit__Day;
		case arrow::DateUnit::MILLI:
			return ArrowDateUnit__MilliSecond;
		default:
			Elog("Unknown DateUnit (%d)", (int)unit);
	}
}

static inline ArrowTimeUnit
__transformArrowTimeUnit(arrow::TimeUnit::type unit)
{
	switch (unit)
	{
		case arrow::TimeUnit::type::SECOND:
			return ArrowTimeUnit__Second;
		case arrow::TimeUnit::type::MILLI:
			return ArrowTimeUnit__MilliSecond;
		case arrow::TimeUnit::type::MICRO:
			return ArrowTimeUnit__MicroSecond;
		case arrow::TimeUnit::type::NANO:
			return ArrowTimeUnit__NanoSecond;
		default:
			Elog("Unknown TimeUnit (%d)", (int)unit);
	}
}

/*
 * __readParquetFieldMetadata
 */
static void
__readParquetFieldMetadata(ArrowField *node,
						   std::shared_ptr<arrow::Field> field,
						   const parquet::ColumnDescriptor *cdesc)
{
	INIT_ARROW_NODE(node, Field);
	node->name = __pstrdup(field->name().c_str());
	node->_name_len = std::strlen(node->name);
	node->nullable = field->nullable();

	auto	__type = field->type();
	switch (__type->id())
	{
		case arrow::Type::type::NA:
			INIT_ARROW_TYPE_NODE(&node->type, Null);
			break;
		case arrow::Type::type::BOOL:
			INIT_ARROW_TYPE_NODE(&node->type, Bool);
			break;
		case arrow::Type::type::UINT8:
			INIT_ARROW_TYPE_NODE(&node->type, Int);
			node->type.Int.bitWidth  = 8;
			node->type.Int.is_signed = false;
			break;
		case arrow::Type::type::INT8:
			INIT_ARROW_TYPE_NODE(&node->type, Int);
			node->type.Int.bitWidth  = 8;
			node->type.Int.is_signed = true;
			break;
		case arrow::Type::type::UINT16:
			INIT_ARROW_TYPE_NODE(&node->type, Int);
			node->type.Int.bitWidth  = 16;
			node->type.Int.is_signed = false;
			break;
		case arrow::Type::type::INT16:
			INIT_ARROW_TYPE_NODE(&node->type, Int);
			node->type.Int.bitWidth  = 16;
			node->type.Int.is_signed = true;
			break;
		case arrow::Type::type::UINT32:
			INIT_ARROW_TYPE_NODE(&node->type, Int);
			node->type.Int.bitWidth  = 32;
			node->type.Int.is_signed = false;
			break;
		case arrow::Type::type::INT32:
			INIT_ARROW_TYPE_NODE(&node->type, Int);
			node->type.Int.bitWidth  = 32;
			node->type.Int.is_signed = true;
			break;
		case arrow::Type::type::UINT64:
			INIT_ARROW_TYPE_NODE(&node->type, Int);
			node->type.Int.bitWidth  = 64;
			node->type.Int.is_signed = false;
			break;
		case arrow::Type::type::INT64:
			INIT_ARROW_TYPE_NODE(&node->type, Int);
			node->type.Int.bitWidth  = 64;
			node->type.Int.is_signed = true;
			break;
		case arrow::Type::type::HALF_FLOAT:
			INIT_ARROW_TYPE_NODE(&node->type, FloatingPoint);
			node->type.FloatingPoint.precision = ArrowPrecision__Half;
			break;
		case arrow::Type::type::FLOAT:
			INIT_ARROW_TYPE_NODE(&node->type, FloatingPoint);
			node->type.FloatingPoint.precision = ArrowPrecision__Single;
			break;
		case arrow::Type::type::DOUBLE:
			INIT_ARROW_TYPE_NODE(&node->type, FloatingPoint);
			node->type.FloatingPoint.precision = ArrowPrecision__Double;
			break;
		case arrow::Type::type::DECIMAL128:
		case arrow::Type::type::DECIMAL256: {
			const auto d_type = arrow::internal::checked_pointer_cast<arrow::DecimalType>(__type);
			INIT_ARROW_TYPE_NODE(&node->type, Decimal);
			node->type.Decimal.precision = d_type->precision();
			node->type.Decimal.scale = d_type->scale();
			node->type.Decimal.bitWidth = (__type->id() == arrow::Type::type::DECIMAL128 ? 128 : 256);
			break;
		}
		case arrow::Type::type::STRING:
			INIT_ARROW_TYPE_NODE(&node->type, Utf8);
			break;
		case arrow::Type::type::BINARY:
			INIT_ARROW_TYPE_NODE(&node->type, Binary);
			break;
		case arrow::Type::type::FIXED_SIZE_BINARY: {
			const auto f_type = arrow::internal::checked_pointer_cast<arrow::FixedSizeBinaryType>(__type);
			INIT_ARROW_TYPE_NODE(&node->type, FixedSizeBinary);
			node->type.FixedSizeBinary.byteWidth
				= f_type->byte_width();
			break;
		}
		case arrow::Type::type::DATE32:
		case arrow::Type::type::DATE64: {
			const auto d_type = arrow::internal::checked_pointer_cast<arrow::DateType>(__type);
			INIT_ARROW_TYPE_NODE(&node->type, Date);
			node->type.Date.unit = __transformArrowDateUnit(d_type->unit());
			break;
		}
		case arrow::Type::type::TIMESTAMP: {
			const auto ts_type = arrow::internal::checked_pointer_cast<arrow::TimestampType>(__type);
			const auto ts_tz = ts_type->timezone();
			INIT_ARROW_TYPE_NODE(&node->type, Timestamp);
			node->type.Timestamp.unit = __transformArrowTimeUnit(ts_type->unit());
			if (ts_tz.size() > 0)
			{
				node->type.Timestamp.timezone = __pstrdup(ts_tz.c_str());
				node->type.Timestamp._timezone_len = std::strlen(node->type.Timestamp.timezone);
			}
			break;
		}
		case arrow::Type::type::TIME32:
		case arrow::Type::type::TIME64: {
			const auto tm_type = arrow::internal::checked_pointer_cast<arrow::TimeType>(__type);
			INIT_ARROW_TYPE_NODE(&node->type, Time);
			node->type.Time.unit = __transformArrowTimeUnit(tm_type->unit());
			node->type.Time.bitWidth = (__type->id() == arrow::Type::type::TIME32 ? 32 : 64);
			break;
		}
		case arrow::Type::type::INTERVAL_MONTHS:
			INIT_ARROW_TYPE_NODE(&node->type, Interval);
			node->type.Interval.unit = ArrowIntervalUnit__Year_Month;
			break;
		case arrow::Type::type::INTERVAL_DAY_TIME:
			INIT_ARROW_TYPE_NODE(&node->type, Interval);
			node->type.Interval.unit = ArrowIntervalUnit__Day_Time;
			break;
		case arrow::Type::type::INTERVAL_MONTH_DAY_NANO:
			INIT_ARROW_TYPE_NODE(&node->type, Interval);
			node->type.Interval.unit = ArrowIntervalUnit__Month_Day_Nano;
			break;
		case arrow::Type::type::LIST:
			INIT_ARROW_TYPE_NODE(&node->type, List);
			break;
		case arrow::Type::type::STRUCT:
			INIT_ARROW_TYPE_NODE(&node->type, Struct);
			break;
		case arrow::Type::type::SPARSE_UNION:
		case arrow::Type::type::DENSE_UNION: {
			const auto un_type = arrow::internal::checked_pointer_cast<arrow::UnionType>(__type);
			INIT_ARROW_TYPE_NODE(&node->type, Union);
			node->type.Union.mode = (__type->id() == arrow::Type::type::SPARSE_UNION
									 ? ArrowUnionMode__Sparse
									 : ArrowUnionMode__Dense);
			auto	__type_codes = un_type->type_codes();
			if (__type_codes.size() > 0)
			{
				node->type.Union._num_typeIds = __type_codes.size();
				node->type.Union.typeIds = (int32_t *)
					__palloc(sizeof(int32_t) * node->type.Union._num_typeIds);
				for (int i=0; i < node->type.Union._num_typeIds; i++)
					node->type.Union.typeIds[i] = __type_codes[i];
			}
			break;
		}
		case arrow::Type::type::MAP: {
			const auto map_type = arrow::internal::checked_pointer_cast<arrow::MapType>(__type);
			INIT_ARROW_TYPE_NODE(&node->type, Map);
			node->type.Map.keysSorted = map_type->keys_sorted();
			break;
		}
		case arrow::Type::type::FIXED_SIZE_LIST: {
			const auto fl_type = arrow::internal::checked_pointer_cast<arrow::FixedSizeListType>(__type);
			INIT_ARROW_TYPE_NODE(&node->type, FixedSizeList);
			node->type.FixedSizeList.listSize = fl_type->list_size();
			break;
		}
		case arrow::Type::type::DURATION: {
			const auto du_type = arrow::internal::checked_pointer_cast<arrow::DurationType>(__type);
			INIT_ARROW_TYPE_NODE(&node->type, Duration);
			node->type.Duration.unit = __transformArrowTimeUnit(du_type->unit());
			break;
		}
		case arrow::Type::type::LARGE_STRING:
			INIT_ARROW_TYPE_NODE(&node->type, LargeUtf8);
			break;
		case arrow::Type::type::LARGE_BINARY:
			INIT_ARROW_TYPE_NODE(&node->type, LargeBinary);
			break;
		case arrow::Type::type::LARGE_LIST:
			INIT_ARROW_TYPE_NODE(&node->type, LargeList);
			break;
		default:
			Elog("unknown arrow type mapping (id=%d)", (int)__type->id());
	}
	/* subfields (List/Struct) */
	if (__type->num_fields() > 0)
	{
		node->_num_children = __type->num_fields();
		node->children = (ArrowField *)
			__palloc(sizeof(ArrowField) * node->_num_children);
		for (int i=0; i < node->_num_children; i++)
		{
			__readParquetFieldMetadata(&node->children[i],
									   __type->field(i),
									   NULL);
		}
	}

	/* custom metadata */
	auto	__custom_metadata = field->metadata();
	if (__custom_metadata && __custom_metadata->size() > 0)
	{
		node->_num_custom_metadata = __custom_metadata->size();
		node->custom_metadata = (ArrowKeyValue *)
			__palloc(sizeof(ArrowKeyValue) * node->_num_custom_metadata);
		for (int i=0; i < node->_num_custom_metadata; i++)
		{
			readArrowKeyValue(&node->custom_metadata[i],
							  __custom_metadata->key(i),
							  __custom_metadata->value(i));
		}
	}
	/* add extra stuff of parquet */
	if (cdesc)
	{
		auto	logical_type = cdesc->logical_type();

		node->parquet.max_definition_level   = cdesc->max_definition_level();
		node->parquet.max_repetition_level   = cdesc->max_repetition_level();
		node->parquet.physical_type  = (int16_t)cdesc->physical_type();
		node->parquet.converted_type = (int16_t)cdesc->converted_type();
		if (logical_type)
		{
			node->parquet.logical_type = (int16_t)logical_type->type();
			switch (logical_type->type())
			{
				case parquet::LogicalType::Type::DECIMAL: {
					auto decimal_type = std::static_pointer_cast<const parquet::DecimalLogicalType>(logical_type);
					node->parquet.logical_decimal_precision = decimal_type->precision();
					node->parquet.logical_decimal_scale = decimal_type->scale();
					break;
				}
				case parquet::LogicalType::Type::TIME: {
					auto time_type = std::static_pointer_cast<const parquet::TimeLogicalType>(logical_type);
					node->parquet.logical_time_unit = time_type->time_unit();
					break;
				}
				case parquet::LogicalType::Type::TIMESTAMP: {
					auto ts_type = std::static_pointer_cast<const parquet::TimestampLogicalType>(logical_type);
					node->parquet.logical_time_unit = ts_type->time_unit();
					break;
				}
				default:
					break;
			}
		}
	}
}

/*
 * __readParquetSchemaMetadata
 */
static void
__readParquetSchemaMetadata(ArrowSchema *node,
							const parquet::SchemaDescriptor *parquet_schema)
{
	std::shared_ptr<arrow::Schema> arrow_schema;

	auto	status = parquet::arrow::FromParquetSchema(parquet_schema,
													   &arrow_schema);
	if (!status.ok())
		Elog("failed on parquet::FromParquetSchema: %s",
			 status.ToString().c_str());

	INIT_ARROW_NODE(node, Schema);
	switch (arrow_schema->endianness())
	{
		case arrow::Endianness::Little:
			node->endianness = ArrowEndianness__Little;
			break;
		case arrow::Endianness::Big:
			node->endianness = ArrowEndianness__Big;
			break;
		default:
			Elog("unknown Endianness (%d)", (int)arrow_schema->endianness());
			break;
	}
	if (arrow_schema->num_fields() > 0)
	{
		node->_num_fields = arrow_schema->num_fields();
		node->fields = (ArrowField *)
			__palloc(sizeof(ArrowField) * node->_num_fields);
		for (int j=0; j < node->_num_fields; j++)
		{
			__readParquetFieldMetadata(&node->fields[j],
									   arrow_schema->field(j),
									   parquet_schema->Column(j));
		}
	}
	auto	__custom_metadata = arrow_schema->metadata();
	if (__custom_metadata && __custom_metadata->size() > 0)
	{
		node->_num_custom_metadata = __custom_metadata->size();
		node->custom_metadata = (ArrowKeyValue *)
            __palloc(sizeof(ArrowKeyValue) * node->_num_custom_metadata);
		for (int i=0; i < node->_num_custom_metadata; i++)
		{
			readArrowKeyValue(&node->custom_metadata[i],
							  __custom_metadata->key(i),
							  __custom_metadata->value(i));
		}
	}
}

/*
 * __readParquetMinMaxStats
 */
static void
__readParquetMinMaxStats(ArrowFieldNode *field,
						 std::shared_ptr<parquet::Statistics> stats)
{
	std::string	min_datum;
	std::string	max_datum;

	switch (stats->physical_type())
	{
		case parquet::Type::BOOLEAN: {
			auto __stat = std::dynamic_pointer_cast<const parquet::BoolStatistics>(stats);
			min_datum = (__stat->min() ? "true" : "false");
			max_datum = (__stat->max() ? "true" : "false");
			break;
		}
		case parquet::Type::INT32: {
			auto __stat = std::dynamic_pointer_cast<const parquet::Int32Statistics>(stats);
			min_datum = std::to_string(__stat->min());
			max_datum = std::to_string(__stat->max());
			break;
		}
		case parquet::Type::INT64: {
			auto __stat = std::dynamic_pointer_cast<const parquet::Int64Statistics>(stats);
			min_datum = std::to_string(__stat->min());
			max_datum = std::to_string(__stat->max());
			break;
		}
		case parquet::Type::FLOAT: {
			auto __stat = std::dynamic_pointer_cast<const parquet::FloatStatistics>(stats);
			min_datum = std::to_string(__stat->min());
			max_datum = std::to_string(__stat->max());
			break;
		}
		case parquet::Type::DOUBLE: {
			auto __stat = std::dynamic_pointer_cast<const parquet::DoubleStatistics>(stats);
			min_datum = std::to_string(__stat->min());
			max_datum = std::to_string(__stat->max());
			break;
		}
		case parquet::Type::FIXED_LEN_BYTE_ARRAY: {
			auto __stat = std::dynamic_pointer_cast<const parquet::FLBAStatistics>(stats);
			auto cdescr = __stat->descr();

			assert(cdescr->physical_type() == stats->physical_type());
			if (cdescr->converted_type() == parquet::ConvertedType::type::DECIMAL ||
				(cdescr->logical_type() && cdescr->logical_type()->is_decimal()))
			{
				/* only Decimal128 */
				auto	min_rv = arrow::Decimal128::FromBigEndian(__stat->min().ptr,
																  cdescr->type_length());
				auto	max_rv = arrow::Decimal128::FromBigEndian(__stat->max().ptr,
																  cdescr->type_length());
				if (!min_rv.ok() || !max_rv.ok())
					return;
				min_datum = min_rv.ValueOrDie().ToString(cdescr->type_scale());
				max_datum = max_rv.ValueOrDie().ToString(cdescr->type_scale());
				break;
			}
			return;
		}
		default:
			return;		/* not supported */
	}
	field->stat_min_value = __pstrdup(min_datum.c_str());
	field->_stat_min_value_len = min_datum.size();
	field->stat_max_value = __pstrdup(max_datum.c_str());
	field->_stat_max_value_len = max_datum.size();
}

/*
 * __arrowCompressionTypeFromMetadata
 */
static inline ArrowCompressionType
__arrowCompressionTypeFromMetadata(arrow::Compression::type code)
{
	switch (code)
	{
		case arrow::Compression::type::UNCOMPRESSED:
			return ArrowCompressionType__UNCOMPRESSED;
		case arrow::Compression::type::SNAPPY:
			return ArrowCompressionType__SNAPPY;
		case arrow::Compression::type::GZIP:
			return ArrowCompressionType__GZIP;
		case arrow::Compression::type::BROTLI:
			return ArrowCompressionType__BROTLI;
		case arrow::Compression::type::ZSTD:
			return ArrowCompressionType__ZSTD;
		case arrow::Compression::type::LZ4:
			return ArrowCompressionType__LZ4;
		case arrow::Compression::type::LZ4_FRAME:
			return ArrowCompressionType__LZ4_FRAME;
		case arrow::Compression::type::LZO:
			return ArrowCompressionType__LZO;
		case arrow::Compression::type::BZ2:
			return ArrowCompressionType__BZ2;
		case arrow::Compression::type::LZ4_HADOOP:
			return ArrowCompressionType__LZ4_HADOOP;
		default:
			return ArrowCompressionType__UNKNOWN;
	}
}

/*
 * __readParquetRowGroupMetadata
 */
static void
__readParquetRowGroupMetadata(ArrowMessage *rbatch_message,
							  ArrowBlock   *rbatch_block,
							  std::unique_ptr<parquet::RowGroupMetaData> rg_meta,
							  ArrowMetadataVersion metadata_version,
							  int64_t next_rowgroup_offset)
{
	int64_t		bodyLength = 0;

	/*
	 * RowGroup is similar to RecordBatch Message in Arrow
	 */
	INIT_ARROW_NODE(rbatch_message, Message);
	rbatch_message->version = metadata_version;
	rbatch_message->bodyLength = (next_rowgroup_offset - rg_meta->file_offset());

	/*
	 * ColumnChunkMetaData is tranformed as if FieldNodes/Buffers
	 */
	ArrowRecordBatch   *rbatch = &rbatch_message->body.recordBatch;

	INIT_ARROW_NODE(rbatch, RecordBatch);
	rbatch->length = rg_meta->num_rows();
	rbatch->_num_nodes = rg_meta->num_columns();
	rbatch->nodes = (ArrowFieldNode *)
		__palloc(sizeof(ArrowFieldNode) * rbatch->_num_nodes);
	for (int j=0; j < rg_meta->num_columns(); j++)
	{
		auto	col_meta = rg_meta->ColumnChunk(j);
		auto	field = &rbatch->nodes[j];
		auto	stats = col_meta->statistics();
		auto	encodings = col_meta->encodings();
		auto	encoding_stats = col_meta->encoding_stats();

		INIT_ARROW_NODE(field, FieldNode);
		field->length = col_meta->num_values();
		if (stats)
		{
			if (stats->HasNullCount())
				field->null_count = stats->null_count();
			if (stats->HasMinMax())
				__readParquetMinMaxStats(field, stats);
		}
		/*
		 * Some additional Parquet specific attrobutes for dump only
		 */
		field->parquet.dictionary_page_offset  = col_meta->dictionary_page_offset();
		field->parquet.data_page_offset        = col_meta->data_page_offset();
		field->parquet.index_page_offset       = col_meta->index_page_offset();
		field->parquet.total_compressed_size   = col_meta->total_compressed_size();
		field->parquet.total_uncompressed_size = col_meta->total_uncompressed_size();
		field->parquet.compression_type        =
			__arrowCompressionTypeFromMetadata(col_meta->compression());
		field->parquet._num_encodings = encodings.size();
		if (field->parquet._num_encodings > 0)
		{
			field->parquet.encodings = (int16_t *)
				__palloc(sizeof(int16_t) * field->parquet._num_encodings);
			for (int k=0; k < field->parquet._num_encodings; k++)
				field->parquet.encodings[k] = (int16_t)encodings[k];
		}
		field->parquet._num_encoding_stats = encoding_stats.size();
		if (field->parquet._num_encoding_stats > 0)
		{
			field->parquet.encoding_stats = (ParquetEncodingStats *)
				__palloc(sizeof(ParquetEncodingStats) * field->parquet._num_encoding_stats);
			for (int k=0; k < field->parquet._num_encoding_stats; k++)
			{
				auto	dst = &field->parquet.encoding_stats[k];
				auto	src = encoding_stats[k];

				dst->page_type = (int16_t)src.page_type;
				dst->encoding  = (int16_t)src.encoding;
				dst->count     = src.count;
			}
		}
		bodyLength = Max(bodyLength, (col_meta->data_page_offset() +
									  col_meta->total_compressed_size()));
	}

	/*
	 * Block (whole Row-Group)
	 */
	INIT_ARROW_NODE(rbatch_block, Block);
	rbatch_block->offset = rg_meta->file_offset();
	rbatch_block->metaDataLength = (next_rowgroup_offset - bodyLength);
	rbatch_block->bodyLength = bodyLength;
}

/*
 * __readParquetFileMetadata
 */
static void
__readParquetFileMetadata(std::shared_ptr<arrow::io::ReadableFile> rfilp,
						  ArrowFileInfo *af_info)
{
	ArrowFooter *footer = &af_info->footer;
	auto	parquet_reader = parquet::ParquetFileReader::Open(rfilp);
	auto	file_meta = parquet_reader->metadata();
	auto	file_size = rfilp->GetSize().ValueOrDie();

	INIT_ARROW_NODE(footer, Footer);
	/* Parquet file version */
	switch (file_meta->version())
	{
		case parquet::ParquetVersion::PARQUET_1_0:
			footer->version = ArrowMetadataVersion__Parquet_V1_0;
			break;
		case parquet::ParquetVersion::PARQUET_2_4:
			footer->version = ArrowMetadataVersion__Parquet_V2_4;
			break;
		case parquet::ParquetVersion::PARQUET_2_6:
			footer->version = ArrowMetadataVersion__Parquet_V2_6;
			break;
		default:
			Elog("unknown Parquet version code (%d)", (int)file_meta->version());
	}

	/*
	 * Read Schema definition
	 */
	__readParquetSchemaMetadata(&footer->schema, file_meta->schema());

	/*
	 * For each Row-Groups
	 */
	if (file_meta->num_row_groups() > 0)
	{
		auto	ngroups = file_meta->num_row_groups();
		int64_t	meta_file_offset = (file_size - (PARQUET_SIGNATURE_SZ
												 + sizeof(uint32_t)
												 + file_meta->size()));
		af_info->recordBatches = (ArrowMessage *)
			__palloc(sizeof(ArrowMessage) * ngroups);
		footer->recordBatches = (ArrowBlock *)
			__palloc(sizeof(ArrowBlock) * ngroups);
		for (int i=0; i < ngroups; i++)
		{
			int64_t		next_rowgroup_offset = (i+1 < ngroups
												? file_meta->RowGroup(i+1)->file_offset()
												: meta_file_offset);
			__readParquetRowGroupMetadata(&af_info->recordBatches[i],
										  &footer->recordBatches[i],
										  file_meta->RowGroup(i),
										  footer->version,
										  next_rowgroup_offset);
		}
		af_info->_num_recordBatches = ngroups;
		footer->_num_recordBatches = ngroups;
	}
	/* key-value metadata */
	auto	file_key_value = file_meta->key_value_metadata();
	if (file_key_value)
	{
		int		nitems = file_key_value->size();
		ArrowKeyValue  *custom_metadata = (ArrowKeyValue *)
			__palloc(sizeof(ArrowKeyValue) * nitems);
		for (int i=0; i < nitems; i++)
		{
			ArrowKeyValue *node = &custom_metadata[i];

			INIT_ARROW_NODE(node, KeyValue);
			node->key = __pstrdup(file_key_value->key(i).c_str());
			node->_key_len = std::strlen(node->key);
			node->value = __pstrdup(file_key_value->value(i).c_str());
			node->_value_len = std::strlen(node->value);
		}
		af_info->footer._num_custom_metadata = nitems;
		af_info->footer.custom_metadata = custom_metadata;
	}
}
#endif

/*
 * readArrowFileInfo
 */
static void
__readArrowFileInfo(int fdesc, ArrowFileInfo *af_info)
{
	char	magic[10];

	/* open the file stream */
	auto rv = arrow::io::ReadableFile::Open(fdesc);
	if (!rv.ok())
		Elog("failed on arrow::io::ReadableFile::Open: %s",
			 rv.status().ToString().c_str());
	/* std::shared_ptr<arrow::io::ReadableFile> */
	auto rfilp = rv.ValueOrDie();
	/* Quick check of the file format. */
	if (rfilp->ReadAt(0, 6, magic) != 6)
		Elog("failed on arrow::io::ReadableFile::ReadAt('%s')", af_info->filename);
	if (std::memcmp(magic, ARROW_SIGNATURE, ARROW_SIGNATURE_SZ) == 0)
		__readArrowFileMetadata(rfilp, af_info);
#if HAS_PARQUET
	else if (std::memcmp(magic, PARQUET_SIGNATURE, PARQUET_SIGNATURE_SZ) == 0)
		__readParquetFileMetadata(rfilp, af_info);
#endif
	else
		Elog("file is neither arrow nor parquet");
	/* file is automatically closed on destructor */
}

int
readArrowFileInfo(const char *filename, ArrowFileInfo *af_info)
{
	char   *emsg = NULL;
	int		fdesc;

	/* try open the file */
	fdesc = open(filename, O_RDONLY);
	if (fdesc < 0)
	{
		assert(errno != 0);
		return errno;
	}
	try {
		/* init ArrowFileInfo */
		memset(af_info, 0, sizeof(ArrowFileInfo));
		af_info->filename = __pstrdup(filename);
		if (fstat(fdesc, &af_info->stat_buf) != 0)
			Elog("failed on fstat('%s'): %m", filename);
		/* walk on the apache arrow/parquet */
		__readArrowFileInfo(fdesc, af_info);
	}
	catch (const std::exception &e) {
		const char *estr = e.what();
		emsg = (char *)alloca(std::strlen(estr)+1);
		strcpy(emsg, estr);
	}
	/* error report */
	if (emsg)
		ErrorReport(emsg);
	return 0;	/* success */
}
