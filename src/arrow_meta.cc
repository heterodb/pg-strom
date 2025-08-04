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
#include <arrow/api.h>			/* dnf install libarrow-devel */
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/ipc/reader.h>
#include <fcntl.h>
#include <sstream>
#include <string>

#include "flatbuffers/flatbuffers.h"			/* copied from arrow */
#include "flatbuffers/File_generated.h"			/* copied from arrow */
#include "flatbuffers/Message_generated.h"		/* copied from arrow */
#include "flatbuffers/Schema_generated.h"		/* copied from arrow */
#include "flatbuffers/SparseTensor_generated.h"	/* copied from arrow */
#include "flatbuffers/Tensor_generated.h"		/* copied from arrow */
#define ARROW_SIGNATURE		"ARROW1"
#define PARQUET_SIGNATURE	"PAR1"

/*
 * Error Reporting
 */
#ifdef PGSTROM_DEBUG_BUILD
extern "C" {
#include "postgres.h"
}
#endif
#define Elog(fmt,...)								\
	do {											\
		char   *ebuf = (char *)alloca(320);			\
		snprintf(ebuf, 320, "[ERROR %s:%d] " fmt,	\
				 __FILE__,__LINE__, ##__VA_ARGS__);	\
		throw std::runtime_error(ebuf);				\
	} while(0)

/*
 * Memory Allocation
 */
#ifdef PGSTROM_DEBUG_BUILD
extern "C" {
#include "utils/palloc.h"
}
inline void *__palloc(size_t sz)
{
	void   *ptr = palloc_extended(sz, MCXT_ALLOC_NO_OOM);
	if (!ptr)
		Elog("out of memory (sz=%lu)", sz);
	return ptr;
}

inline void *__palloc0(size_t sz)
{
	void   *ptr = palloc_extended(sz, MCXT_ALLOC_NO_OOM | MCXT_ALLOC_ZERO);
	if (!ptr)
		Elog("out of memory (sz=%lu)", sz);
	return ptr;
}

inline void *__repalloc(void *ptr, size_t sz)
{
	ptr = repalloc_extended(ptr, sz, MCXT_ALLOC_NO_OOM);
	if (!ptr)
		Elog("out of memory (sz=%lu)", sz);
	return ptr;
}
#else
inline void *__palloc(size_t sz)
{
	void   *ptr = malloc(sz);
	if (!ptr)
		Elog("out of memory (sz=%lu)", sz);
	return ptr;
}

inline void *__palloc0(size_t sz)
{
	void   *ptr = malloc(sz);
	if (!ptr)
		Elog("out of memory (sz=%lu)", sz);
	memset(ptr, 0, sz);
	return ptr;
}

inline void *__repalloc(void *ptr, size_t sz)
{
	ptr = realloc(ptr, sz);
	if (!ptr)
		Elog("out of memory (sz=%lu)", sz);
	return ptr;
}
#endif
inline char *__pstrdup(const char *str)
{
	size_t	sz = strlen(str);
	char   *result = (char *)__palloc(sz+1);
	memcpy(result, str, sz);
	result[sz] = '\0';
	return result;
}
inline char *__pstrdup(const flatbuffers::String *str)
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
__dumpArrowNode(std::ostringstream &json, const ArrowNode *node);

static inline std::string
__escape_json(const char *str)
{
	std::string	res;

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
	return res;
}

static inline void
__dumpArrowTypeInt(std::ostringstream &json, const ArrowTypeInt *node)
{
	const char *tag = (node->is_signed ? "Int" : "UInt");
	json << "\"" << tag << node->bitWidth  << "\"";
}
static inline void
__dumpArrowTypeFloatingPoint(std::ostringstream &json, const ArrowTypeFloatingPoint *node)
{
	const char *suffix = (node->precision == ArrowPrecision__Double ? "64" :
						  node->precision == ArrowPrecision__Single ? "32" :
						  node->precision == ArrowPrecision__Half   ? "16" : "???");
	json << "\"" << suffix << "\"";
}
static inline void
__dumpArrowTypeDecimal(std::ostringstream &json, const ArrowTypeDecimal *node)
{
	json << "\"Decimal" << node->bitWidth << "\""
		 << ", \"precision\" : " << node->precision
		 << ", \"scale\" : " << node->scale;
}
static inline void
__dumpArrowTypeDate(std::ostringstream &json, const ArrowTypeDate *node)
{
	const char *unit = ArrowDateUnitAsCString(node->unit);
	const char *suffix = (node->unit == ArrowDateUnit__Day ? "32" : "64");

	json << "\"Date" << suffix << "\""
		 << ", \"unit\" : \"" << unit << "\"";
}
static inline void
__dumpArrowTypeTime(std::ostringstream &json, const ArrowTypeTime *node)
{
	const char *unit = ArrowTimeUnitAsCString(node->unit);
	const char *suffix = (node->unit == ArrowTimeUnit__Second ||
						  node->unit == ArrowTimeUnit__MilliSecond) ? "32" : "64";
	json << "\"Time" << suffix << "\""
		 << ", \"unit\" : \"" << unit << "\"";
}
static inline void
__dumpArrowTypeTimestamp(std::ostringstream &json, const ArrowTypeTimestamp *node)
{
	const char *unit = ArrowTimeUnitAsCString(node->unit);
	json << "\"Timestamp\""
		 << ", \"unit\" : \"" << unit << "\"";
}
static inline void
__dumpArrowTypeInterval(std::ostringstream &json, const ArrowTypeInterval *node)
{
	const char *unit = ArrowIntervalUnitAsCString(node->unit);
	json << "\"Interval\""
		 << ", \"unit\" : \"" << unit << "\"";
}
static inline void
__dumpArrowTypeUnion(std::ostringstream &json, const ArrowTypeUnion *node)
{
	const char *mode = ArrowUnionModeAsCString(node->mode);
	json << "\"Union\""
		 << ", \"mode\" : \""<< mode << "\""
		 << ", \"typeIds\" : [";
	for (int i=0; i < node->_num_typeIds; i++)
	{
		if (i > 0)
			json << ", ";
		json << node->typeIds[i];
	}
	json << "]";
}
static inline void
__dumpArrowTypeFixedSizeBinary(std::ostringstream &json, const ArrowTypeFixedSizeBinary *node)
{
	json << "\"FixedSizeBinary\", \"byteWidth\" : " << node->byteWidth;
}
static inline void
__dumpArrowTypeFixedSizeList(std::ostringstream &json, const ArrowTypeFixedSizeList *node)
{
	json << "\"FixedSizeList\""
		 << ", \"listSize\" : " << node->listSize;
}
static inline void
__dumpArrowTypeMap(std::ostringstream &json, const ArrowTypeMap *node)
{
	json << "\"Map\""
		 << ", keysSorted : " << (node->keysSorted ? "true" : "false");
}
static inline void
__dumpArrowTypeDuration(std::ostringstream &json, const ArrowTypeDuration *node)
{
	const char *unit = ArrowTimeUnitAsCString(node->unit);
	json << "\"Duration\""
		 << ", \"unit\" : " << unit;
}
static inline void
__dumpArrowKeyValue(std::ostringstream &json, const ArrowKeyValue *node)
{
	json << "\"KeyValue\""
		 << ", \"key\" : \"" << __escape_json(node->key) << "\""
		 << ", \"value\" : \""<< __escape_json(node->value) << "\"";
}
static inline void
__dumpArrowDictionaryEncoding(std::ostringstream &json, const ArrowDictionaryEncoding *node)
{
	json << "\"DictionaryEncoding\", \"id\" : " << node->id
		 << ", \"indexType\" : ";
	__dumpArrowNode(json, &node->indexType.node);
	json << ", \"isOrdered\" : " << (node->isOrdered ? "true" : "false");
}
static inline void
__dumpArrowField(std::ostringstream &json, const ArrowField *node)
{
	json << "\"Field\""
		 << ", \"name\" : \"" << __escape_json(node->name) << "\""
		 << ", \"nullable\" : " << (node->nullable ? "true" : "false")
		 << ", \"type\" : ";
	__dumpArrowNode(json, &node->type.node);
	if (node->dictionary)
	{
		json << ", \"dictionary\" : ";
		__dumpArrowNode(json, (ArrowNode *)node->dictionary);
	}
	if (node->children && node->_num_children > 0)
	{
		json << ", \"children\" : [";
		for (int i=0; i < node->_num_children; i++)
		{
			json << (i > 0 ? ", " : " ");
			__dumpArrowNode(json, &node->children[i].node);
		}
		json << " ]";
	}
	if (node->custom_metadata && node->_num_custom_metadata > 0)
	{
		json << ", \"custom_metadata\" : [";
		for (int i=0; i < node->_num_custom_metadata; i++)
		{
			json << (i > 0 ? ", " : " ");
			__dumpArrowNode(json, &node->custom_metadata[i].node);
		}
		json << " ]";
	}
}
static inline void
__dumpArrowFieldNode(std::ostringstream &json, const ArrowFieldNode *node)
{
	json << "\"FieldNode\""
		 << ", \"length\" : " << node->length
		 << ", \"null_count\" : " << node->null_count;
}
static inline void
__dumpArrowBuffer(std::ostringstream &json, const ArrowBuffer *node)
{
	json << "\"Buffer\""
		 << ", \"offset\" : " << node->offset
		 << ", \"length\" : " << node->length;
}
static inline void
__dumpArrowSchema(std::ostringstream &json, const ArrowSchema *node)
{
	const char *endian = ArrowEndiannessAsCString(node->endianness);
	json << "\"Schema\""
		 << ", \"endianness\" : \""<< endian << "\"";
	if (node->fields && node->_num_fields > 0)
	{
		json << ", \"fields\" : [";
		for (int i=0; i < node->_num_fields; i++)
		{
			json << (i > 0 ? ", " : " ");
			__dumpArrowNode(json, &node->fields[i].node);
		}
		json << " ]";
	}
	if (node->custom_metadata && node->_num_custom_metadata > 0)
	{
		json << ", \"custom_metadata\" : [";
		for (int i=0; i < node->_num_custom_metadata; i++)
		{
			json << (i > 0 ? ", " : "");
			__dumpArrowNode(json, &node->custom_metadata[i].node);
		}
		json << " ]";
	}
	if (node->features && node->_num_features > 0)
	{
		json << ", \"features\" : [";
		for (int i=0; i < node->_num_features; i++)
		{
			const char *feature = ArrowFeatureAsCString(node->features[i]);

			json << (i > 0 ? ", " : " ");
			json << "\"" << feature << "\"";
		}
		json << " ]";
	}
}
static inline void
__dumpArrowRecordBatch(std::ostringstream &json, const ArrowRecordBatch *node)
{
	json << "\"ArrowRecordBatch\""
		 << ", \"length\" : " << node->length;
	if (node->nodes && node->_num_nodes > 0)
	{
		json << ", \"nodes\" : [";
		for (int i=0; i < node->_num_nodes; i++)
		{
			json << (i > 0 ? ", " : " ");
			__dumpArrowNode(json, &node->nodes[i].node);
		}
		json << " ]";
	}
	if (node->buffers && node->_num_buffers > 0)
	{
		json << ", \"buffers\" : [";
		for (int i=0; i < node->_num_buffers; i++)
		{
			json << (i > 0 ? ", " : " ");
			__dumpArrowNode(json, &node->buffers[i].node);
		}
		json << " ]";
	}
	if (node->compression)
	{
		json << ", \"compression\" : ";
		__dumpArrowNode(json, (const ArrowNode *)node->compression);
	}
}
static inline void
__dumpArrowDictionaryBatch(std::ostringstream &json, const ArrowDictionaryBatch *node)
{
	json << "\"DictionaryBatch\""
		 << ", \"id\" : " << node->id
		 << ", \"data\" : ";
	__dumpArrowNode(json, &node->data.node);
	json << ", \"isDelta\" : " << (node->isDelta ? "true" : "false");
}
static inline void
__dumpArrowMessage(std::ostringstream &json, const ArrowMessage *node)
{
	const char *version = ArrowMetadataVersionAsCString(node->version);
	json << "\"Message\""
		 << ", \"version\" : \"" << version << "\""
		 << ", \"body\" : ";
	__dumpArrowNode(json, &node->body.node);
	json << ", \"bodyLength\" : " << node->bodyLength;
}
static inline void
__dumpArrowBlock(std::ostringstream &json, const ArrowBlock *node)
{
	json << "\"Block\""
		 << ", \"offset\" : " << node->offset
		 << ", \"metaDataLength\" : " << node->metaDataLength
		 << ", \"bodyLength\" : " << node->bodyLength;
}
static inline void
__dumpArrowFooter(std::ostringstream &json, const ArrowFooter *node)
{
	const char *version = ArrowMetadataVersionAsCString(node->version);
	json << "\"Footer\""
		 << ", \"version\" : \"" << version << "\""
		 << ", \"schema\" : ";
	__dumpArrowNode(json, &node->schema.node);
	if (node->dictionaries && node->_num_dictionaries > 0)
	{
		json << ", \"dictionaries\" : [";
		for (int i=0; i < node->_num_dictionaries; i++)
		{
			json << (i > 0 ? ", " : " ");
			__dumpArrowNode(json, &node->dictionaries[i].node);
		}
		json << " ]";
	}
	if (node->recordBatches && node->_num_recordBatches > 0)
	{
		json << ", \"recordBatches\" : [";
		for (int i=0; i < node->_num_recordBatches; i++)
		{
			json << (i > 0 ? ", " : " ");
			__dumpArrowNode(json, &node->recordBatches[i].node);
		}
		json << " ]";
	}
	if (node->custom_metadata && node->_num_custom_metadata > 0)
	{
		json << ", \"custom_metadata\" : [";
		for (int i=0; i < node->_num_custom_metadata; i++)
		{
			json << (i > 0 ? "," : "");
			__dumpArrowNode(json, &node->custom_metadata[i].node);
		}
		json << " ]";
	}
}
static inline void
__dumpArrowBodyCompression(std::ostringstream &json, const ArrowBodyCompression *node)
{
	const char *codec = ArrowCompressionTypeAsCString(node->codec);
	const char *method = ArrowBodyCompressionMethodAsCString(node->method);
	json << "\"BodyCompression\""
		 << ", \"codec\" : \"" << codec << "\""
		 << ", \"method\" : \"" << method << "\"";
}

static void
__dumpArrowNode(std::ostringstream &json, const ArrowNode *node)
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
			json << "\"" << node->tagName << "\"";
			break;		/* nothing to special */
		case ArrowNodeTag__Int:break;
		case ArrowNodeTag__FloatingPoint:break;
		case ArrowNodeTag__Decimal:break;
		case ArrowNodeTag__Date:break;
		case ArrowNodeTag__Time:break;
		case ArrowNodeTag__Timestamp:break;
		case ArrowNodeTag__Interval:break;
		case ArrowNodeTag__Union:break;
		case ArrowNodeTag__FixedSizeBinary:break;
		case ArrowNodeTag__FixedSizeList:break;
		case ArrowNodeTag__Map:break;
		case ArrowNodeTag__Duration:break;
		case ArrowNodeTag__KeyValue:break;
		case ArrowNodeTag__DictionaryEncoding:break;
		case ArrowNodeTag__Field:break;
		case ArrowNodeTag__FieldNode:break;
		case ArrowNodeTag__Buffer:break;
		case ArrowNodeTag__Schema:break;
		case ArrowNodeTag__RecordBatch:break;
		case ArrowNodeTag__DictionaryBatch:break;
		case ArrowNodeTag__Message:break;
		case ArrowNodeTag__Block:break;
		case ArrowNodeTag__Footer:break;
		case ArrowNodeTag__BodyCompression:break;
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

		__dumpArrowNode(json, node);
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
	{
#ifdef PGSTROM_DEBUG_BUILD
		elog(ERROR, "%s", emsg);
#else
		fputs(emsg, stderr);
		exit(1);
#endif
	}
	return result;
}

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
        if ((src)->_num_##FIELD == 0)								\
        {															\
            (dest)->FIELD = NULL;									\
        }															\
        else														\
        {															\
			(dest)->FIELD = (NODETYPE *)							\
				__palloc(sizeof(NODETYPE) *	(src)->_num_##FIELD);	\
			for (int j=0; j < (src)->_num_##FIELD; j++)				\
                __copyArrowNode(&(dest)->FIELD[j].node,				\
								&(src)->FIELD[j].node);				\
        }															\
        (dest)->_num_##FIELD = (src)->_num_##FIELD;					\
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
	if (!src->typeIds)
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
}

static inline void
__copyArrowFieldNode(ArrowFieldNode *dest, const ArrowFieldNode *src)
{
	COPY_SCALAR(length);
	COPY_SCALAR(null_count);
}

static inline void
__copyArrowSchema(ArrowSchema *dest, const ArrowSchema *src)
{
	COPY_SCALAR(endianness);
	COPY_VECTOR(fields, ArrowField);
	COPY_VECTOR(custom_metadata, ArrowKeyValue);
	if (!src->features)
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
	{
#ifdef PGSTROM_DEBUG_BUILD
		elog(ERROR, "%s", emsg);
#else
		fputs(emsg, stderr);
		exit(1);
#endif
	}
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
		return true;
	}
	return false;
}

static inline bool
__equalArrowFieldNode(const ArrowFieldNode *a,
					  const ArrowFieldNode *b)
{
	return (a->length == b->length &&
			a->null_count == b->null_count);
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
	return (a->version == b->version &&
			__equalArrowNode(&a->body.node,
							 &b->body.node) &&
			a->bodyLength == b->bodyLength);
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
	{
#ifdef PGSTROM_DEBUG_BUILD
		elog(ERROR, "%s", emsg);
#else
		fputs(emsg, stderr);
		exit(1);
#endif
	}
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
				  const org::apache::arrow::flatbuf::KeyValue *kv)
{
	INIT_ARROW_NODE(node, KeyValue);
	node->key = __pstrdup(kv->key());
	node->_key_len = std::strlen(node->key);
	node->value = __pstrdup(kv->value());
	node->_value_len = std::strlen(node->value);
}

static void
readArrowFieldNode(ArrowFieldNode *node,
				   const org::apache::arrow::flatbuf::FieldNode *fnode)
{
	INIT_ARROW_NODE(node, FieldNode);
	node->length     = fnode->length();
	node->null_count = fnode->null_count();
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
		case org::apache::arrow::flatbuf::Precision::HALF:
			node->precision = ArrowPrecision__Half;
			break;
		case org::apache::arrow::flatbuf::Precision::SINGLE:
			node->precision = ArrowPrecision__Single;
			break;
		case org::apache::arrow::flatbuf::Precision::DOUBLE:
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
		case org::apache::arrow::flatbuf::DateUnit::DAY:
			node->unit = ArrowDateUnit__Day;
			break;
		case org::apache::arrow::flatbuf::DateUnit::MILLISECOND:
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
		case org::apache::arrow::flatbuf::TimeUnit::SECOND:
			node->unit = ArrowTimeUnit__Second;
			break;
		case org::apache::arrow::flatbuf::TimeUnit::MILLISECOND:
			node->unit = ArrowTimeUnit__MilliSecond;
			break;
		case org::apache::arrow::flatbuf::TimeUnit::MICROSECOND:
			node->unit = ArrowTimeUnit__MicroSecond;
			break;
		case org::apache::arrow::flatbuf::TimeUnit::NANOSECOND:
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
		case org::apache::arrow::flatbuf::TimeUnit::SECOND:
			node->unit = ArrowTimeUnit__Second;
			break;
		case org::apache::arrow::flatbuf::TimeUnit::MILLISECOND:
			node->unit = ArrowTimeUnit__MilliSecond;
			break;
		case org::apache::arrow::flatbuf::TimeUnit::MICROSECOND:
			node->unit = ArrowTimeUnit__MicroSecond;
			break;
		case org::apache::arrow::flatbuf::TimeUnit::NANOSECOND:
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
		case org::apache::arrow::flatbuf::IntervalUnit::YEAR_MONTH:
			node->unit = ArrowIntervalUnit__Year_Month;
			break;
		case org::apache::arrow::flatbuf::IntervalUnit::DAY_TIME:
			node->unit = ArrowIntervalUnit__Day_Time;
			break;
		case org::apache::arrow::flatbuf::IntervalUnit::MONTH_DAY_NANO:
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
		case org::apache::arrow::flatbuf::UnionMode::Sparse:
			node->mode = ArrowUnionMode__Sparse;
			break;
		case org::apache::arrow::flatbuf::UnionMode::Dense:
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
		case org::apache::arrow::flatbuf::TimeUnit::SECOND:
			node->unit = ArrowTimeUnit__Second;
			break;
		case org::apache::arrow::flatbuf::TimeUnit::MILLISECOND:
			node->unit = ArrowTimeUnit__MilliSecond;
			break;
		case org::apache::arrow::flatbuf::TimeUnit::MICROSECOND:
			node->unit = ArrowTimeUnit__MicroSecond;
			break;
		case org::apache::arrow::flatbuf::TimeUnit::NANOSECOND:
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
		case org::apache::arrow::flatbuf::Type::Null:
			INIT_ARROW_TYPE_NODE(&node->type, Null);
			break;
		case org::apache::arrow::flatbuf::Type::Int:
			readArrowTypeInt(&node->type.Int,
							 field->type_as_Int());
			break;
		case org::apache::arrow::flatbuf::Type::FloatingPoint:
			readArrowTypeFloatingPoint(&node->type.FloatingPoint,
									   field->type_as_FloatingPoint());
			break;
		case org::apache::arrow::flatbuf::Type::Binary:
			INIT_ARROW_TYPE_NODE(&node->type, Binary);
			break;
		case org::apache::arrow::flatbuf::Type::Utf8:
			INIT_ARROW_TYPE_NODE(&node->type, Utf8);
			break;
		case org::apache::arrow::flatbuf::Type::Bool:
			INIT_ARROW_TYPE_NODE(&node->type, Bool);
			break;
		case org::apache::arrow::flatbuf::Type::Decimal:
			readArrowTypeDecimal(&node->type.Decimal,
								 field->type_as_Decimal());
			break;
		case org::apache::arrow::flatbuf::Type::Date:
			readArrowTypeDate(&node->type.Date,
							  field->type_as_Date());
			break;
		case org::apache::arrow::flatbuf::Type::Time:
			readArrowTypeTime(&node->type.Time,
							  field->type_as_Time());
			break;
		case org::apache::arrow::flatbuf::Type::Timestamp:
			readArrowTypeTimestamp(&node->type.Timestamp,
								   field->type_as_Timestamp());
			break;
		case org::apache::arrow::flatbuf::Type::Interval:
			readArrowTypeInterval(&node->type.Interval,
								  field->type_as_Interval());
			break;
		case org::apache::arrow::flatbuf::Type::List:
			INIT_ARROW_TYPE_NODE(&node->type, List);
			break;
		case org::apache::arrow::flatbuf::Type::Struct_:
			INIT_ARROW_TYPE_NODE(&node->type, Struct);
			break;
		case org::apache::arrow::flatbuf::Type::Union:
			readArrowTypeUnion(&node->type.Union,
							   field->type_as_Union());
			break;
		case org::apache::arrow::flatbuf::Type::FixedSizeBinary:
			readArrowTypeFixedSizeBinary(&node->type.FixedSizeBinary,
										 field->type_as_FixedSizeBinary());
			break;
		case org::apache::arrow::flatbuf::Type::FixedSizeList:
			readArrowTypeFixedSizeList(&node->type.FixedSizeList,
									   field->type_as_FixedSizeList());
			break;
		case org::apache::arrow::flatbuf::Type::Map:
			readArrowTypeMap(&node->type.Map,
							 field->type_as_Map());
			break;
		case org::apache::arrow::flatbuf::Type::Duration:
			readArrowTypeDuration(&node->type.Duration,
								  field->type_as_Duration());
			break;
		case org::apache::arrow::flatbuf::Type::LargeBinary:
			INIT_ARROW_TYPE_NODE(&node->type, LargeBinary);
			break;
		case org::apache::arrow::flatbuf::Type::LargeUtf8:
			INIT_ARROW_TYPE_NODE(&node->type, LargeUtf8);
			break;
		case org::apache::arrow::flatbuf::Type::LargeList:
			INIT_ARROW_TYPE_NODE(&node->type, LargeList);
			break;
		default:
			Elog("unknown org::apache::arrow::flatbuf::Type (%d)",
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
			readArrowKeyValue(&node->custom_metadata[i], (*__custom_metadata)[i]);
	}
}

static void
readArrowSchemaMessage(ArrowSchema *node,
					   const org::apache::arrow::flatbuf::Schema *schema)
{
	INIT_ARROW_NODE(node, Schema);
	switch (schema->endianness())
	{
		case org::apache::arrow::flatbuf::Endianness::Little:
			node->endianness = ArrowEndianness__Little;
			break;
		case org::apache::arrow::flatbuf::Endianness::Big:
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
			readArrowKeyValue(&node->custom_metadata[i], (*__custom_metadata)[i]);
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
				case org::apache::arrow::flatbuf::Feature::UNUSED:
					node->features[i] = ArrowFeature__Unused;
					break;
				case org::apache::arrow::flatbuf::Feature::DICTIONARY_REPLACEMENT:
					node->features[i] = ArrowFeature__DictionaryReplacement;
					break;
				case org::apache::arrow::flatbuf::Feature::COMPRESSED_BODY:
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
		case org::apache::arrow::flatbuf::CompressionType::LZ4_FRAME:
			node->codec = ArrowCompressionType__LZ4_FRAME;
			break;
		case org::apache::arrow::flatbuf::CompressionType::ZSTD:
			node->codec = ArrowCompressionType__ZSTD;
			break;
		default:
			Elog("unknown CompressionType (%d)",
				 (int)compression->codec());
			break;
	}
	switch (compression->method())
	{
		case org::apache::arrow::flatbuf::BodyCompressionMethod::BUFFER:
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

	switch (message->header_type())
	{
		case org::apache::arrow::flatbuf::MessageHeader::Schema: {
			auto	__schema = message->header_as_Schema();
			readArrowSchemaMessage(&node->body.schema, __schema);
			break;
		}
		case org::apache::arrow::flatbuf::MessageHeader::DictionaryBatch: {
			auto	__dbatch = message->header_as_DictionaryBatch();
			readArrowDictionaryBatchMessage(&node->body.dictionaryBatch, __dbatch);
			break;
		}
		case org::apache::arrow::flatbuf::MessageHeader::RecordBatch: {
			auto	__rbatch = message->header_as_RecordBatch();
			readArrowRecordBatchMessage(&node->body.recordBatch, __rbatch);
			break;
		}
		case org::apache::arrow::flatbuf::MessageHeader::Tensor:
			Elog("MessageHeader::Tensor is not implemented right now");
		case org::apache::arrow::flatbuf::MessageHeader::SparseTensor:
			Elog("MessageHeader::SparseTensor is not implemented right now");
		default:
			Elog("corrupted arrow file? unknown message type %d",
				 (int)message->header_type());
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
		case org::apache::arrow::flatbuf::MetadataVersion::V1:
			node->version = ArrowMetadataVersion__V1;
			break;
		case org::apache::arrow::flatbuf::MetadataVersion::V2:
			node->version = ArrowMetadataVersion__V2;
			break;
		case org::apache::arrow::flatbuf::MetadataVersion::V3:
			node->version = ArrowMetadataVersion__V3;
			break;
		case org::apache::arrow::flatbuf::MetadataVersion::V4:
			node->version = ArrowMetadataVersion__V4;
			break;
		case org::apache::arrow::flatbuf::MetadataVersion::V5:
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
			readArrowKeyValue(&node->custom_metadata[i],
							  (*__custom_metadata)[i]);
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
	auto	tail_sz = sizeof(uint32_t) + sizeof(ARROW_SIGNATURE)-1;
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
				   sizeof(ARROW_SIGNATURE)-1) != 0)
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
		}
		/* this is Apache Arrow file */
		af_info->file_is_parquet = false;
	}
}

#ifdef HAS_PARQUET
/*
 * __readParquetFileMetadata
 */
static void
__readParquetFileMetadata(std::shared_ptr<arrow::io::ReadableFile> rfile,
						  ArrowFileInfo *af_info)
{

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
	/* check file format - arrow or parquet */
	if (rfilp->ReadAt(0, 6, magic) != 6)
		Elog("failed on arrow::io::ReadableFile::ReadAt");
	if (std::memcmp(magic, ARROW_SIGNATURE, sizeof(ARROW_SIGNATURE)-1) == 0)
		__readArrowFileMetadata(rfilp, af_info);
#if HAS_PARQUET
	else if (std::memcmp(magic, PARQUET_SIGNATURE, sizeof(PARQUET_SIGNATURE)-1) == 0)
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
	{
#ifdef PGSTROM_DEBUG_BUILD
		elog(ERROR, "%s", emsg);
#else
		fputs(emsg, stderr);
		exit(1);
#endif
	}
	return 0;	/* success */
}
