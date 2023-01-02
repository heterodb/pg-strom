/*
 * arrow_nodes.c
 *
 * Routines to handle ArrowNode objects, intermediation of PostgreSQL types
 * and Apache Arrow types.
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <stdarg.h>
#include "arrow_ipc.h"

static void		sql_buffer_printf(SQLbuffer *buf, const char *fmt, ...)
#ifdef pg_attribute_printf
	pg_attribute_printf(2,3)
#elif defined(__GNUC__)
	__attribute__((format(gnu_printf, 2, 3)))
#endif
	;

/*
 * Dump support of ArrowNode
 */
static void
__dumpArrowNode(SQLbuffer *buf, ArrowNode *node)
{
	node->dumpArrowNode(buf, node);
}

static void
__dumpArrowNodeSimple(SQLbuffer *buf, ArrowNode *node)
{
	sql_buffer_printf(buf, "{%s}", node->tagName);
}
#define __dumpArrowTypeNull			__dumpArrowNodeSimple
#define __dumpArrowTypeUtf8			__dumpArrowNodeSimple
#define __dumpArrowTypeBinary		__dumpArrowNodeSimple
#define __dumpArrowTypeBool			__dumpArrowNodeSimple
#define __dumpArrowTypeList			__dumpArrowNodeSimple
#define __dumpArrowTypeStruct		__dumpArrowNodeSimple
#define __dumpArrowTypeLargeBinary	__dumpArrowNodeSimple
#define __dumpArrowTypeLargeUtf8	__dumpArrowNodeSimple
#define __dumpArrowTypeLargeList	__dumpArrowNodeSimple

static inline const char *
ArrowPrecisionAsCstring(ArrowPrecision prec)
{
	switch (prec)
	{
		case ArrowPrecision__Half:
			return "16";
		case ArrowPrecision__Single:
			return "32";
		case ArrowPrecision__Double:
			return "64";
		default:
			return "??";
	}
}

static inline const char *
ArrowDateUnitAsCstring(ArrowDateUnit unit)
{
	switch (unit)
	{
		case ArrowDateUnit__Day:
			return "day";
		case ArrowDateUnit__MilliSecond:
			return "msec";
		default:
			return "???";
	}
}

static inline const char *
ArrowTimeUnitAsCstring(ArrowTimeUnit unit)
{
	switch (unit)
	{
		case  ArrowTimeUnit__Second:
			return "sec";
		case ArrowTimeUnit__MilliSecond:
			return "ms";
		case ArrowTimeUnit__MicroSecond:
			return "us";
		case ArrowTimeUnit__NanoSecond:
			return "ns";
		default:
			return "???";
	}
}

static inline const char *
ArrowIntervalUnitAsCstring(ArrowIntervalUnit unit)
{
	switch (unit)
	{
		case ArrowIntervalUnit__Year_Month:
			return "Year/Month";
		case ArrowIntervalUnit__Day_Time:
			return "Day/Time";
		default:
			return "???";
	}
}

static inline const char *
ArrowFeatureAsCstring(ArrowFeature feature)
{
	switch (feature)
	{
		case ArrowFeature__Unused:
			return "Unused";
		case ArrowFeature__DictionaryReplacement:
			return "DictionaryReplacement";
		case ArrowFeature__CompressedBody:
			return "CompressedBody";
		default:
			return "???";
	}
}

static inline const char *
ArrowCompressionTypeAsCstring(ArrowCompressionType codec)
{
	switch (codec)
	{
		case ArrowCompressionType__LZ4_FRAME:
			return "LZ4_FRAME";
		case ArrowCompressionType__ZSTD:
			return "ZSTD";
		default:
			return "???";
	}
}

static inline const char *
ArrowBodyCompressionMethodAsCstring(ArrowBodyCompressionMethod method)
{
	switch (method)
	{
		case ArrowBodyCompressionMethod__BUFFER:
			return "BUFFER";
		default:
			return "???";
	}
}

static void
__dumpArrowTypeInt(SQLbuffer *buf, ArrowNode *node)
{
	ArrowTypeInt   *i = (ArrowTypeInt *)node;

	sql_buffer_printf(
		buf, "{%s%d}",
		i->is_signed ? "Int" : "Uint",
		i->bitWidth);
}

static void
__dumpArrowTypeFloatingPoint(SQLbuffer *buf, ArrowNode *node)
{
	ArrowTypeFloatingPoint *f = (ArrowTypeFloatingPoint *)node;

	sql_buffer_printf(
		buf, "{Float%s}",
		ArrowPrecisionAsCstring(f->precision));
}

static void
__dumpArrowTypeDecimal(SQLbuffer *buf, ArrowNode *node)
{
	ArrowTypeDecimal *d = (ArrowTypeDecimal *)node;

	sql_buffer_printf(
		buf, "{Decimal: precision=%d, scale=%d, bitWidth=%d}",
		d->precision,
		d->scale,
		d->bitWidth);
}

static void
__dumpArrowTypeDate(SQLbuffer *buf, ArrowNode *node)
{
	ArrowTypeDate *d = (ArrowTypeDate *)node;

	sql_buffer_printf(
		buf, "{Date: unit=%s}",
		ArrowDateUnitAsCstring(d->unit));
}

static void
__dumpArrowTypeTime(SQLbuffer *buf, ArrowNode *node)
{
	ArrowTypeTime *t = (ArrowTypeTime *)node;

	sql_buffer_printf(
		buf, "{Time: unit=%s, bitWidth=%d}",
		ArrowTimeUnitAsCstring(t->unit), t->bitWidth);
}

static void
__dumpArrowTypeTimestamp(SQLbuffer *buf, ArrowNode *node)
{
	ArrowTypeTimestamp *t = (ArrowTypeTimestamp *)node;

	if (t->timezone)
	{
		sql_buffer_printf(
			buf, "{Timestamp: unit=%s, timezone='%*s'}",
			ArrowTimeUnitAsCstring(t->unit),
			t->_timezone_len, t->timezone);
	}
	else
	{
		sql_buffer_printf(
			buf, "{Timestamp: unit=%s}",
			ArrowTimeUnitAsCstring(t->unit));
	}
}

static void
__dumpArrowTypeInterval(SQLbuffer *buf, ArrowNode *node)
{
	ArrowTypeInterval *t = (ArrowTypeInterval *)node;

	sql_buffer_printf(
		buf, "{Interval: unit=%s}",
		ArrowIntervalUnitAsCstring(t->unit));
}

static void
__dumpArrowTypeUnion(SQLbuffer *buf, ArrowNode *node)
{
	ArrowTypeUnion *u = (ArrowTypeUnion *)node;
	int			i;

	sql_buffer_printf(
		buf, "{Union: mode=%s, typeIds=[",
		u->mode == ArrowUnionMode__Sparse ? "Sparse" :
		u->mode == ArrowUnionMode__Dense ? "Dense" : "???");
	for (i=0; i < u->_num_typeIds; i++)
		sql_buffer_printf(buf, "%s%d", i > 0 ? ", " : " ",
						  u->typeIds[i]);
	sql_buffer_printf(buf, "]}");
}

static void
__dumpArrowTypeFixedSizeBinary(SQLbuffer *buf, ArrowNode *node)
{
	ArrowTypeFixedSizeBinary *fb = (ArrowTypeFixedSizeBinary *)node;

	sql_buffer_printf(
		buf, "{FixedSizeBinary: byteWidth=%d}",
		fb->byteWidth);
}

static void
__dumpArrowTypeFixedSizeList(SQLbuffer *buf, ArrowNode *node)
{
	ArrowTypeFixedSizeList *fl = (ArrowTypeFixedSizeList *)node;

	sql_buffer_printf(
		buf, "{FixedSizeList: listSize=%d}",
		fl->listSize);
}

static void
__dumpArrowTypeMap(SQLbuffer *buf, ArrowNode *node)
{
	ArrowTypeMap *m = (ArrowTypeMap *)node;

	sql_buffer_printf(
		buf, "{Map: keysSorted=%s}",
		m->keysSorted ? "true" : "false");
}

static void
__dumpArrowTypeDuration(SQLbuffer *buf, ArrowNode *node)
{
	ArrowTypeDuration *d = (ArrowTypeDuration *)node;

	sql_buffer_printf(
		buf, "{Duration: unit=%s}",
		ArrowTimeUnitAsCstring(d->unit));
}

static void
__dumpArrowKeyValue(SQLbuffer *buf, ArrowNode *node)
{
	ArrowKeyValue *kv = (ArrowKeyValue *)node;

	sql_buffer_printf(
		buf, "{KeyValue: key=\"%s\" value=\"%s\"}",
		kv->key ? kv->key : "",
		kv->value ? kv->value : "");
}

static void
__dumpArrowDictionaryEncoding(SQLbuffer *buf, ArrowNode *node)
{
	ArrowDictionaryEncoding *d = (ArrowDictionaryEncoding *)node;

	sql_buffer_printf(
		buf, "{DictionaryEncoding: id=%ld, indexType=", d->id);
	__dumpArrowNode(buf, (ArrowNode *)&d->indexType);
	sql_buffer_printf(
		buf, ", isOrdered=%s}",
		d->isOrdered ? "true" : "false");
}

static void
__dumpArrowField(SQLbuffer *buf, ArrowNode *node)
{
	ArrowField *f = (ArrowField *)node;
	int		i;

	sql_buffer_printf(
		buf, "{Field: name=\"%s\", nullable=%s, type=",
		f->name ? f->name : "",
		f->nullable ? "true" : "false");
	__dumpArrowNode(buf, (ArrowNode *)&f->type);
	if (f->dictionary)
	{
		sql_buffer_printf(buf, ", dictionary=");
		__dumpArrowNode(buf, (ArrowNode *)f->dictionary);
	}
	sql_buffer_printf(buf, ", children=[");
	for (i=0; i < f->_num_children; i++)
	{
		if (i > 0)
			sql_buffer_printf(buf, ", ");
		__dumpArrowNode(buf, (ArrowNode *)&f->children[i]);
	}
	sql_buffer_printf(buf, "], custom_metadata=[");
	for (i=0; i < f->_num_custom_metadata; i++)
	{
		if (i > 0)
			sql_buffer_printf(buf, ", ");
		__dumpArrowNode(buf, (ArrowNode *)&f->custom_metadata[i]);
	}
	sql_buffer_printf(buf, "]}");
}

static void
__dumpArrowFieldNode(SQLbuffer *buf, ArrowNode *node)
{
	sql_buffer_printf(
		buf, "{FieldNode: length=%ld, null_count=%ld}",
		((ArrowFieldNode *)node)->length,
		((ArrowFieldNode *)node)->null_count);
}

static void
__dumpArrowBuffer(SQLbuffer *buf, ArrowNode *node)
{
	sql_buffer_printf(
		buf, "{Buffer: offset=%ld, length=%ld}",
		((ArrowBuffer *)node)->offset,
		((ArrowBuffer *)node)->length);
}

static void
__dumpArrowSchema(SQLbuffer *buf, ArrowNode *node)
{
	ArrowSchema *s = (ArrowSchema *)node;
	int		i;

	sql_buffer_printf(
		buf, "{Schema: endianness=%s, fields=[",
		s->endianness == ArrowEndianness__Little ? "little" :
		s->endianness == ArrowEndianness__Big ? "big" : "???");
	for (i=0; i < s->_num_fields; i++)
	{
		if (i > 0)
			sql_buffer_printf(buf, ", ");
		__dumpArrowNode(buf, (ArrowNode *)&s->fields[i]);
	}
	sql_buffer_printf(buf, "], custom_metadata=[");
	for (i=0; i < s->_num_custom_metadata; i++)
	{
		if (i > 0)
			sql_buffer_printf(buf, ", ");
		__dumpArrowNode(buf, (ArrowNode *)&s->custom_metadata[i]);
	}
	sql_buffer_printf(buf, "], features=[");
	for (i=0; i < s->_num_features; i++)
	{
		if (i > 0)
			sql_buffer_printf(buf, ", ");
		sql_buffer_printf(buf, "%s", ArrowFeatureAsCstring(s->features[i]));
	}
	sql_buffer_printf(buf, "]}");
}

static void
__dumpArrowRecordBatch(SQLbuffer *buf, ArrowNode *node)
{
	ArrowRecordBatch *r = (ArrowRecordBatch *) node;
	int		i;

	sql_buffer_printf(buf, "{RecordBatch: length=%ld, nodes=[", r->length);
	for (i=0; i < r->_num_nodes; i++)
	{
		if (i > 0)
			sql_buffer_printf(buf, ", ");
		__dumpArrowNode(buf, (ArrowNode *)&r->nodes[i]);
	}
	sql_buffer_printf(buf, "], buffers=[");
	for (i=0; i < r->_num_buffers; i++)
	{
		if (i > 0)
			sql_buffer_printf(buf, ", ");
		__dumpArrowNode(buf, (ArrowNode *)&r->buffers[i]);
	}
	sql_buffer_printf(buf,"]}");
}

static void
__dumpArrowDictionaryBatch(SQLbuffer *buf, ArrowNode *node)
{
	ArrowDictionaryBatch *d = (ArrowDictionaryBatch *)node;

	sql_buffer_printf(
		buf, "{DictionaryBatch: id=%ld, data=", d->id);
	__dumpArrowNode(buf, (ArrowNode *)&d->data);
	sql_buffer_printf(
		buf, ", isDelta=%s}",
		d->isDelta ? "true" : "false");
}

static void
__dumpArrowMessage(SQLbuffer *buf, ArrowNode *node)
{
	ArrowMessage *m = (ArrowMessage *)node;

	sql_buffer_printf(
		buf, "{Message: version=%s, body=",
		m->version == ArrowMetadataVersion__V1 ? "V1" :
		m->version == ArrowMetadataVersion__V2 ? "V2" :
		m->version == ArrowMetadataVersion__V3 ? "V3" :
		m->version == ArrowMetadataVersion__V4 ? "V4" : "???");
	__dumpArrowNode(buf, (ArrowNode *)&m->body);
	sql_buffer_printf(buf, ", bodyLength=%lu}", m->bodyLength);
}

static void
__dumpArrowBlock(SQLbuffer *buf, ArrowNode *node)
{
	sql_buffer_printf(
		buf, "{Block: offset=%ld, metaDataLength=%d bodyLength=%ld}",
		((ArrowBlock *)node)->offset,
		((ArrowBlock *)node)->metaDataLength,
		((ArrowBlock *)node)->bodyLength);
}

static void
__dumpArrowFooter(SQLbuffer *buf, ArrowNode *node)
{
	ArrowFooter *f = (ArrowFooter *)node;
	int		i;

	sql_buffer_printf(
		buf, "{Footer: version=%s, schema=",
		f->version == ArrowMetadataVersion__V1 ? "V1" :
		f->version == ArrowMetadataVersion__V2 ? "V2" :
		f->version == ArrowMetadataVersion__V3 ? "V3" :
		f->version == ArrowMetadataVersion__V4 ? "V4" :
		f->version == ArrowMetadataVersion__V5 ? "V5" : "???");
	__dumpArrowNode(buf, (ArrowNode *)&f->schema);
	sql_buffer_printf(buf, ", dictionaries=[");
	for (i=0; i < f->_num_dictionaries; i++)
	{
		if (i > 0)
			sql_buffer_printf(buf, ", ");
		__dumpArrowNode(buf, (ArrowNode *)&f->dictionaries[i]);
	}
	sql_buffer_printf(buf, "], recordBatches=[");
	for (i=0; i < f->_num_recordBatches; i++)
	{
		if (i > 0)
			sql_buffer_printf(buf, ", ");
		__dumpArrowNode(buf, (ArrowNode *)&f->recordBatches[i]);
	}
	sql_buffer_printf(buf, "]}");
}

static void
__dumpArrowBodyCompression(SQLbuffer *buf, ArrowNode *node)
{
	ArrowBodyCompression *compress = (ArrowBodyCompression *)node;

	sql_buffer_printf(
		buf, "{BodyCompression: codec=%s, method=%s}",
		ArrowCompressionTypeAsCstring(compress->codec),
		ArrowBodyCompressionMethodAsCstring(compress->method));
}

char *
dumpArrowNode(ArrowNode *node)
{
	SQLbuffer	buf;

	sql_buffer_init(&buf);
	__dumpArrowNode(&buf, node);

	return buf.data;
}

/*
 * sql_buffer_printf - equivalent to appendStringInfo in PG
 */
static void
sql_buffer_printf(SQLbuffer *buf, const char *fmt, ...)
{
	if (!buf->data)
		sql_buffer_expand(buf, 1024);
	for (;;)
	{
		va_list		va_args;
		size_t		avail = buf->length - buf->usage;
		size_t		nbytes;

		va_start(va_args, fmt);
		nbytes = vsnprintf(buf->data + buf->usage, avail, fmt, va_args);
		va_end(va_args);

		if (nbytes < avail)
		{
			buf->usage += nbytes;
			return;
		}
		sql_buffer_expand(buf, buf->usage + nbytes);
	}
}

/*
 * Copy support of ArrowNode
 */
#define COPY_SCALAR(FIELD)								\
	(dest)->FIELD = (src)->FIELD
#define COPY_CSTRING(FIELD)								\
	do {												\
		if ((src)->FIELD)								\
		{												\
			(dest)->FIELD = pstrdup((src)->FIELD);		\
			(dest)->_##FIELD##_len = strlen((dest)->FIELD);	\
		}												\
		else											\
		{												\
			(dest)->FIELD = NULL;						\
			(dest)->_##FIELD##_len = 0;					\
		}												\
	} while(0)
#define COPY_VECTOR(FIELD, NODETYPE)										\
	do {																	\
		int		j;															\
																			\
		if ((src)->_num_##FIELD == 0)										\
		{																	\
			(dest)->FIELD = NULL;											\
		}																	\
		else																\
		{																	\
			(dest)->FIELD = palloc(sizeof(NODETYPE) * (src)->_num_##FIELD);	\
			for (j=0; j < (src)->_num_##FIELD; j++)							\
				__copy##NODETYPE(&(dest)->FIELD[j], &(src)->FIELD[j]);		\
		}																	\
		(dest)->_num_##FIELD = (src)->_num_##FIELD;							\
	} while(0)

static void
__copyArrowNode(ArrowNode *dest, const ArrowNode *src)
{
	COPY_SCALAR(tag);
	COPY_SCALAR(tagName);
	COPY_SCALAR(dumpArrowNode);
	COPY_SCALAR(copyArrowNode);
}
#define __copyArrowTypeNull			__copyArrowNode
#define __copyArrowTypeUtf8			__copyArrowNode
#define __copyArrowTypeBinary		__copyArrowNode
#define __copyArrowTypeBool			__copyArrowNode
#define __copyArrowTypeList			__copyArrowNode
#define __copyArrowTypeStruct		__copyArrowNode
#define __copyArrowTypeLargeBinary	__copyArrowNode
#define __copyArrowTypeLargeUtf8	__copyArrowNode
#define __copyArrowTypeLargeList	__copyArrowNode

static void
__copyArrowTypeInt(ArrowTypeInt *dest, const ArrowTypeInt *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(bitWidth);
	COPY_SCALAR(is_signed);
}

static void
__copyArrowTypeFloatingPoint(ArrowTypeFloatingPoint *dest,
							 const ArrowTypeFloatingPoint *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(precision);
}

static void
__copyArrowTypeDecimal(ArrowTypeDecimal *dest, const ArrowTypeDecimal *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(precision);
	COPY_SCALAR(scale);
	COPY_SCALAR(bitWidth);
}

static void
__copyArrowTypeDate(ArrowTypeDate *dest, const ArrowTypeDate *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(unit);
}

static void
__copyArrowTypeTime(ArrowTypeTime *dest, const ArrowTypeTime *src)
{
	__copyArrowNode(&dest->node, &src->node);
    COPY_SCALAR(unit);
	COPY_SCALAR(bitWidth);
}

static void
__copyArrowTypeTimestamp(ArrowTypeTimestamp *dest,
						 const ArrowTypeTimestamp *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(unit);
	COPY_CSTRING(timezone);
}

static void
__copyArrowTypeInterval(ArrowTypeInterval *dest,
						const ArrowTypeInterval *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(unit);
}

static void
__copyArrowTypeUnion(ArrowTypeUnion *dest, const ArrowTypeUnion *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(mode);
	if (!src->typeIds)
		dest->typeIds = NULL;
	else
	{
		dest->typeIds = palloc(sizeof(int32_t) * src->_num_typeIds);
		memcpy(dest->typeIds, src->typeIds, sizeof(int32_t) * src->_num_typeIds);
	}
	dest->_num_typeIds = src->_num_typeIds;
}

static void
__copyArrowTypeDuration(ArrowTypeDuration *dest, const ArrowTypeDuration *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(unit);
}

static void
__copyArrowTypeFixedSizeBinary(ArrowTypeFixedSizeBinary *dest,
							   const ArrowTypeFixedSizeBinary *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(byteWidth);
}

static void
__copyArrowTypeFixedSizeList(ArrowTypeFixedSizeList *dest,
							 const ArrowTypeFixedSizeList *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(listSize);
}

static void
__copyArrowTypeMap(ArrowTypeMap *dest, const ArrowTypeMap *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(keysSorted);
}

static void
__copyArrowType(ArrowType *dest, const ArrowType *src)
{
	src->node.copyArrowNode(&dest->node, &src->node);
}

static void
__copyArrowBuffer(ArrowBuffer *dest, const ArrowBuffer *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(offset);
	COPY_SCALAR(length);
}

static void
__copyArrowKeyValue(ArrowKeyValue *dest, const ArrowKeyValue *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_CSTRING(key);
	COPY_CSTRING(value);
}

static void
__copyArrowDictionaryEncoding(ArrowDictionaryEncoding *dest,
							  const ArrowDictionaryEncoding *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(id);
	__copyArrowTypeInt(&dest->indexType, &src->indexType);
	COPY_SCALAR(isOrdered);
}

static void
__copyArrowField(ArrowField *dest, const ArrowField *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_CSTRING(name);
	COPY_SCALAR(nullable);
	__copyArrowType(&dest->type, &src->type);
	if (!src->dictionary)
		dest->dictionary = NULL;
	else
	{
		dest->dictionary = palloc0(sizeof(ArrowDictionaryEncoding));
		__copyArrowDictionaryEncoding(dest->dictionary, src->dictionary);
	}
	COPY_VECTOR(children, ArrowField);
	COPY_VECTOR(custom_metadata, ArrowKeyValue);
}

static void
__copyArrowFieldNode(ArrowFieldNode *dest, const ArrowFieldNode *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(length);
	COPY_SCALAR(null_count);
}

static void
__copyArrowSchema(ArrowSchema *dest, const ArrowSchema *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(endianness);
	COPY_VECTOR(fields, ArrowField);
	COPY_VECTOR(custom_metadata, ArrowKeyValue);
	if (!src->features)
		dest->features = NULL;
	else
	{
		dest->features = palloc0(sizeof(ArrowFeature) * src->_num_features);
		memcpy(dest->features, src->features,
			   sizeof(ArrowFeature) * src->_num_features);
	}
	dest->_num_features = src->_num_features;
}

static void
__copyArrowRecordBatch(ArrowRecordBatch *dest, const ArrowRecordBatch *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(length);
	COPY_VECTOR(nodes, ArrowFieldNode);
	COPY_VECTOR(buffers, ArrowBuffer);
}

static void
__copyArrowDictionaryBatch(ArrowDictionaryBatch *dest,
						   const ArrowDictionaryBatch *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(id);
	__copyArrowRecordBatch(&dest->data, &src->data);
	COPY_SCALAR(isDelta);
}

static void
__copyArrowMessage(ArrowMessage *dest, const ArrowMessage *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(version);
	switch (src->body.node.tag)
	{
		case ArrowNodeTag__Schema:
			__copyArrowSchema(&dest->body.schema, &src->body.schema);
			break;
		case ArrowNodeTag__RecordBatch:
			__copyArrowDictionaryBatch(&dest->body.dictionaryBatch,
									   &src->body.dictionaryBatch);
			break;
		case ArrowNodeTag__DictionaryBatch:
			__copyArrowRecordBatch(&dest->body.recordBatch,
								   &src->body.recordBatch);
			break;
		default:
			Elog("Bug? unknown ArrowMessageBody");
	}
	COPY_SCALAR(bodyLength);
}

static void
__copyArrowBlock(ArrowBlock *dest, const ArrowBlock *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(offset);
	COPY_SCALAR(metaDataLength);
	COPY_SCALAR(bodyLength);
}

static void
__copyArrowFooter(ArrowFooter *dest, const ArrowFooter *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(version);
	__copyArrowSchema(&dest->schema, &src->schema);
	COPY_VECTOR(dictionaries, ArrowBlock);
	COPY_VECTOR(recordBatches, ArrowBlock);
}

static void
__copyArrowBodyCompression(ArrowBodyCompression *dest, const ArrowBodyCompression *src)
{
	__copyArrowNode(&dest->node, &src->node);
	COPY_SCALAR(codec);
	COPY_SCALAR(method);
}

void
copyArrowNode(ArrowNode *dest, const ArrowNode *src)
{
	src->copyArrowNode(dest, src);
}

const char *
arrowNodeName(ArrowNode *node)
{
	static __thread char buf[128];
	ArrowPrecision	prep;
	ArrowDateUnit	d_unit;
	ArrowTimeUnit	t_unit;
	ArrowIntervalUnit i_unit;
	const char	   *timezone;
	int		off;

	switch (node->tag)
	{
		case ArrowNodeTag__Null:
			return "Arrow::Null";

		case ArrowNodeTag__Int:
			snprintf(buf, sizeof(buf), "Arrow::%s%d",
					 ((ArrowTypeInt *)node)->is_signed ? "Int" : "Uint",
					 ((ArrowTypeInt *)node)->bitWidth);
			return buf;

		case ArrowNodeTag__FloatingPoint:
			prep = ((ArrowTypeFloatingPoint*)node)->precision;
			snprintf(buf, sizeof(buf), "Arrow::Float%s",
					 ArrowPrecisionAsCstring(prep));
			return buf;

		case ArrowNodeTag__Utf8:
			return "Arrow::Utf8";

		case ArrowNodeTag__Binary:
			return "Arrow::Binary";

		case ArrowNodeTag__Bool:
			return "Arrow::Bool";

		case ArrowNodeTag__Decimal:
			if (((ArrowTypeDecimal *)node)->scale == 0)
				snprintf(buf, sizeof(buf), "Arrow::Decimal%d(%d)",
						 ((ArrowTypeDecimal *)node)->bitWidth,
						 ((ArrowTypeDecimal *)node)->precision);
			else
				snprintf(buf, sizeof(buf), "Arrow::Decimal%d(%d,%d)",
						 ((ArrowTypeDecimal *)node)->bitWidth,
						 ((ArrowTypeDecimal *)node)->precision,
						 ((ArrowTypeDecimal *)node)->scale);
			return buf;

		case ArrowNodeTag__Date:
			d_unit = ((ArrowTypeDate *)node)->unit;
			snprintf(buf, sizeof(buf), "Arrow::Date[%s]",
					 ArrowDateUnitAsCstring(d_unit));
			return buf;

		case ArrowNodeTag__Time:
			t_unit = ((ArrowTypeTime *)node)->unit;
			snprintf(buf, sizeof(buf), "Arrow::Time%d%s",
					 ((ArrowTypeTime *)node)->bitWidth,
					 ArrowTimeUnitAsCstring(t_unit));
			return buf;

		case ArrowNodeTag__Timestamp:
			t_unit = ((ArrowTypeTimestamp *)node)->unit;
			timezone = ((ArrowTypeTimestamp *)node)->timezone;
			off = snprintf(buf, sizeof(buf), "Arrow::Timestamp%s",
						   ArrowTimeUnitAsCstring(t_unit));
			if (timezone)
				snprintf(buf+off, sizeof(buf)-off, " <%s>", timezone);
			return buf;
			
		case ArrowNodeTag__Interval:
			i_unit = ((ArrowTypeInterval *)node)->unit;
			snprintf(buf, sizeof(buf), "Arrow::Interval[%s]",
					 ArrowIntervalUnitAsCstring(i_unit));
			return buf;

		case ArrowNodeTag__List:
			return "Arrow::List";

		case ArrowNodeTag__Struct:
			return "Arrow::Struct";

		case ArrowNodeTag__Union:
			return "Arrow::Union";

		case ArrowNodeTag__FixedSizeBinary:
			snprintf(buf, sizeof(buf), "Arrow::FixedSizeBinary<%d>",
					 ((ArrowTypeFixedSizeBinary *)node)->byteWidth);
			return buf;

		case ArrowNodeTag__FixedSizeList:
			snprintf(buf, sizeof(buf), "Arrow::FixedSizeList<%d>",
					 ((ArrowTypeFixedSizeList *)node)->listSize);
			return buf;

		case ArrowNodeTag__Map:
			return "Arrow::Map";

		case ArrowNodeTag__Duration:
			t_unit = ((ArrowTypeDuration *)node)->unit;
			snprintf(buf, sizeof(buf), "Arrow::Duration%s",
					 t_unit == ArrowTimeUnit__Second ? "" :
					 t_unit == ArrowTimeUnit__MilliSecond ? "[ms]" :
					 t_unit == ArrowTimeUnit__MicroSecond ? "[us]" :
					 t_unit == ArrowTimeUnit__NanoSecond ? "[ns]" : "[??]");
			return buf;

		case ArrowNodeTag__LargeBinary:
			return "Arrow::LargeBinary";

		case ArrowNodeTag__LargeUtf8:
			return "Arrow::LargeUtf8";

		case ArrowNodeTag__LargeList:
			return "Arrow::LargeList";

		case ArrowNodeTag__KeyValue:
			return "Arrow::KeyValue";

		case ArrowNodeTag__DictionaryEncoding:
			return "Arrow::DictionaryEncoding";

		case ArrowNodeTag__Field:
			return "Arrow::Field";

		case ArrowNodeTag__FieldNode:
			return "Arrow::FieldNode";

		case ArrowNodeTag__Buffer:
			return "Arrow::Buffer";

		case ArrowNodeTag__Schema:
			return "Arrow::Schema";

		case ArrowNodeTag__RecordBatch:
			return "Arrow::RecordBatch";

		case ArrowNodeTag__DictionaryBatch:
			return "Arrow::DictionaryBatch";

		case ArrowNodeTag__Message:
			return "Arrow::Message";

		case ArrowNodeTag__Block:
			return "Arrow::Block";

		case ArrowNodeTag__Footer:
			return "Arrow::Footer";

		case ArrowNodeTag__BodyCompression:
			return "Arrow::BodyCompression";

		default:
			break;
	}
	return "Unknown";
}

/* ------------------------------------------------
 *
 * Routines to compare Apache Arrow type nodes
 *
 * ------------------------------------------------
 */
static bool
__arrowFieldTypeIsEqual(ArrowField *a, ArrowField *b, int depth)
{
	int		j;

	if (a->type.node.tag != b->type.node.tag)
		return false;
	switch (a->node.tag)
	{
		case ArrowNodeTag__Int:
			if (a->type.Int.bitWidth != b->type.Int.bitWidth)
				return false;
			break;
		case ArrowNodeTag__FloatingPoint:
			{
				ArrowPrecision	p1 = a->type.FloatingPoint.precision;
				ArrowPrecision	p2 = b->type.FloatingPoint.precision;

				if (p1 != p2)
					return false;
			}
			break;
		case ArrowNodeTag__Utf8:
		case ArrowNodeTag__Binary:
		case ArrowNodeTag__Bool:
			break;

		case ArrowNodeTag__Decimal:
			if (a->type.Decimal.precision != b->type.Decimal.precision ||
				a->type.Decimal.scale     != b->type.Decimal.scale ||
				a->type.Decimal.bitWidth  != b->type.Decimal.bitWidth)
				return false;
			break;

		case ArrowNodeTag__Date:
			if (a->type.Date.unit != b->type.Date.unit)
				return false;
			break;

		case ArrowNodeTag__Time:
			if (a->type.Time.unit != b->type.Time.unit ||
				a->type.Time.bitWidth != b->type.Time.bitWidth)
				return false;
			break;

		case ArrowNodeTag__Timestamp:
			if (a->type.Timestamp.unit != b->type.Timestamp.unit ||
				a->type.Timestamp.timezone != NULL ||
				b->type.Timestamp.timezone != NULL)
				return false;
			break;

		case ArrowNodeTag__Interval:
			if (a->type.Interval.unit != b->type.Interval.unit)
				return false;
			break;

		case ArrowNodeTag__Struct:
			if (depth > 0)
				Elog("arrow: nested composite types are not supported");
			if (a->_num_children != b->_num_children)
				return false;
			for (j=0; j < a->_num_children; j++)
			{
				if (!__arrowFieldTypeIsEqual(&a->children[j],
											 &b->children[j],
											 depth + 1))
					return false;
			}
			break;

		case ArrowNodeTag__List:
			if (depth > 0)
				Elog("arrow_fdw: nested array types are not supported");
			if (a->_num_children != 1 || b->_num_children != 1)
				Elog("Bug? List of arrow type is corrupted.");
			if (!__arrowFieldTypeIsEqual(&a->children[0],
										 &b->children[0],
										 depth + 1))
				return false;
			break;

		default:
			Elog("'%s' of arrow type is not supported",
				 a->node.tagName);
	}
	return true;
}

bool
arrowFieldTypeIsEqual(ArrowField *a, ArrowField *b)
{
	return __arrowFieldTypeIsEqual(a, b, 0);
}

/* ------------------------------------------------
 *
 * Routines to initialize Apache Arrow nodes
 *
 * ------------------------------------------------
 */
typedef void (*dumpArrowNode_f)(SQLbuffer *buf, ArrowNode *node);
typedef void (*copyArrowNode_f)(ArrowNode *dest, const ArrowNode *src);

#define __INIT_ARROW_NODE(PTR,TYPENAME,NAME)				\
	do {													\
		((ArrowNode *)(PTR))->tag = ArrowNodeTag__##NAME;	\
		((ArrowNode *)(PTR))->tagName = #NAME;				\
		((ArrowNode *)(PTR))->dumpArrowNode					\
			= (dumpArrowNode_f)__dumpArrow##TYPENAME;		\
		((ArrowNode *)(PTR))->copyArrowNode					\
			= (copyArrowNode_f)__copyArrow##TYPENAME;		\
	} while(0)
#define INIT_ARROW_NODE(PTR,NAME)		__INIT_ARROW_NODE(PTR,NAME,NAME)
#define INIT_ARROW_TYPE_NODE(PTR,NAME)	__INIT_ARROW_NODE(PTR,Type##NAME,NAME)

void
__initArrowNode(ArrowNode *node, ArrowNodeTag tag)
{
#define CASE_ARROW_NODE(NAME)						\
	case ArrowNodeTag__##NAME:						\
		memset(node, 0, sizeof(Arrow##NAME));		\
		INIT_ARROW_NODE(node,NAME);					\
		break
#define CASE_ARROW_TYPE_NODE(NAME)					\
	case ArrowNodeTag__##NAME:						\
		memset(node, 0, sizeof(ArrowType##NAME));	\
		INIT_ARROW_TYPE_NODE(node,NAME);			\
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

/* ------------------------------------------------
 *
 * Routines to read Apache Arrow files
 *
 * ------------------------------------------------
 */

/* table/vtable of FlatBuffer */
typedef struct
{
	uint16_t	vlen;	/* vtable length */
	uint16_t	tlen;	/* table length */
	uint16_t	offset[FLEXIBLE_ARRAY_MEMBER];
} FBVtable;

typedef struct
{
	int32_t	   *table;
	FBVtable   *vtable;
} FBTable;

static inline FBTable
fetchFBTable(void *p_table)
{
	FBTable		t;

	t.table  = (int32_t *)p_table;
	t.vtable = (FBVtable *)((char *)p_table - *t.table);

	return t;
}

static inline void *
__fetchPointer(FBTable *t, int index)
{
	FBVtable   *vtable = t->vtable;

	if (offsetof(FBVtable, offset[index]) < vtable->vlen)
	{
		uint16_t	offset = vtable->offset[index];

		assert(offset < vtable->tlen);
		if (offset)
			return (char *)t->table + offset;
	}
	return NULL;
}

static inline bool
__fetchBool(FBTable *t, int index, bool __default)
{
	bool	   *ptr = __fetchPointer(t, index);

	return (ptr ? *ptr : __default);
}
	
static inline bool
fetchBool(FBTable *t, int index)
{
	return __fetchBool(t, index, false);
}

static inline int8_t
__fetchChar(FBTable *t, int index, int8_t __default)
{
	int8_t	   *ptr = __fetchPointer(t, index);

	return (ptr ? *ptr : __default);
}

static inline int8_t
fetchChar(FBTable *t, int index)
{
	return __fetchChar(t, index, 0);
}

static inline int16_t
__fetchShort(FBTable *t, int index, int16_t __default)
{
	int16_t	   *ptr = __fetchPointer(t, index);

	return (ptr ? *ptr : __default);
}

static inline int16_t
fetchShort(FBTable *t, int index)
{
	return __fetchShort(t, index, 0);
}

static inline int32_t
__fetchInt(FBTable *t, int index, int32_t __default)
{
	int32_t	   *ptr = __fetchPointer(t, index);

	return (ptr ? *ptr : __default);
}

static inline int32_t
fetchInt(FBTable *t, int index)
{
	return __fetchInt(t, index, 0);
}

static inline int64_t
__fetchLong(FBTable *t, int index, int64_t __default)
{
	int64_t	   *ptr = __fetchPointer(t, index);

	return (ptr ? *ptr : __default);
}

static inline int64_t
fetchLong(FBTable *t, int index)
{
	return __fetchLong(t, index, 0);
}

static inline void *
fetchOffset(FBTable *t, int index)
{
	int32_t	   *ptr = __fetchPointer(t, index);
	return (ptr ? (char *)ptr + *ptr : NULL);
}

static inline const char *
fetchString(FBTable *t, int index, int *p_strlen)
{
	int32_t	   *ptr = fetchOffset(t, index);
	int32_t		len = 0;
	char	   *temp;

	if (!ptr)
		temp = NULL;
	else
	{
		len = *ptr++;
		temp = palloc(len + 1);
		memcpy(temp, (char *)ptr, len);
		temp[len] = '\0';
	}
	*p_strlen = len;
	return temp;
}

static inline int32_t *
fetchVector(FBTable *t, int index, int *p_nitems)
{
	int32_t	   *vector = fetchOffset(t, index);

	if (!vector)
		*p_nitems = 0;
	else
		*p_nitems = *vector++;
	return vector;
}

static void
readArrowKeyValue(ArrowKeyValue *kv, const char *pos)
{
	FBTable		t = fetchFBTable((int32_t *)pos);

	memset(kv, 0, sizeof(ArrowKeyValue));
	INIT_ARROW_NODE(kv, KeyValue);
	kv->key     = fetchString(&t, 0, &kv->_key_len);
	kv->value   = fetchString(&t, 1, &kv->_value_len);
}

static void
readArrowTypeInt(ArrowTypeInt *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32_t *) pos);

	node->bitWidth  = fetchInt(&t, 0);
	node->is_signed = fetchBool(&t, 1);
	if (node->bitWidth != 8  && node->bitWidth != 16 &&
		node->bitWidth != 32 && node->bitWidth != 64)
		Elog("ArrowTypeInt has unknown bitWidth (%d)", node->bitWidth);
}

static void
readArrowTypeFloatingPoint(ArrowTypeFloatingPoint *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32_t *) pos);

	node->precision = fetchShort(&t, 0);
	if (node->precision != ArrowPrecision__Half &&
		node->precision != ArrowPrecision__Single &&
		node->precision != ArrowPrecision__Double)
		Elog("ArrowTypeFloatingPoint has unknown precision (%d)",
			 node->precision);
}

static void
readArrowTypeDecimal(ArrowTypeDecimal *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32_t *) pos);

	node->precision = fetchInt(&t, 0);
	node->scale     = fetchInt(&t, 1);
	node->bitWidth  = __fetchInt(&t, 2, 128);
	if (node->bitWidth != 128 &&
		node->bitWidth != 256)
		Elog("ArrowTypeDecimal has unsupported bitWidth (%d)",
			 node->bitWidth);
}

static void
readArrowTypeDate(ArrowTypeDate *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32_t *) pos);

	/* Date->unit has non-zero default value */
	node->unit = __fetchShort(&t, 0, ArrowDateUnit__MilliSecond);
	if (node->unit != ArrowDateUnit__Day &&
		node->unit != ArrowDateUnit__MilliSecond)
		Elog("ArrowTypeDate has unknown unit (%d)", node->unit);
}

static void
readArrowTypeTime(ArrowTypeTime *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32_t *) pos);

	node->unit = __fetchShort(&t, 0, ArrowTimeUnit__MilliSecond);
	node->bitWidth = __fetchInt(&t, 1, 32);
	switch (node->unit)
	{
		case ArrowTimeUnit__Second:
		case ArrowTimeUnit__MilliSecond:
			if (node->bitWidth != 32)
				Elog("ArrowTypeTime has inconsistent bitWidth(%d) for unit=%d",
					 node->bitWidth, node->unit);
			break;
		case ArrowTimeUnit__MicroSecond:
		case ArrowTimeUnit__NanoSecond:
			if (node->bitWidth != 64)
				Elog("ArrowTypeTime has inconsistent bitWidth(%d) for unit=%d",
					 node->bitWidth, node->unit);
			break;
		default:
			Elog("ArrowTypeTime has unknown unit (%d)", node->unit);
	}
}

static void
readArrowTypeTimestamp(ArrowTypeTimestamp *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32_t *) pos);

	node->unit = fetchShort(&t, 0);
	node->timezone = fetchString(&t, 1, &node->_timezone_len);
	if (node->unit != ArrowTimeUnit__Second &&
		node->unit != ArrowTimeUnit__MilliSecond &&
		node->unit != ArrowTimeUnit__MicroSecond &&
		node->unit != ArrowTimeUnit__NanoSecond)
		Elog("ArrowTypeTimestamp has unknown unit (%d)", node->unit);
}

static void
readArrowTypeInterval(ArrowTypeInterval *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32_t *) pos);

	node->unit = __fetchShort(&t, 0, ArrowTimeUnit__MilliSecond);
	if (node->unit != ArrowIntervalUnit__Year_Month &&
		node->unit != ArrowIntervalUnit__Day_Time)
		Elog("ArrowTypeInterval has unknown unit (%d)", node->unit);
}

static void
readArrowTypeUnion(ArrowTypeUnion *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32_t *) pos);
	int32_t	   *vector;
	int32_t		nitems;

	node->mode = fetchShort(&t, 0);
	vector = fetchVector(&t, 1, &nitems);
	if (nitems == 0)
		node->typeIds = NULL;
	else
	{
		node->typeIds = palloc0(sizeof(int32_t) * nitems);
		memcpy(node->typeIds, vector, sizeof(int32_t) * nitems);
	}
	node->_num_typeIds = nitems;
}

static void
readArrowTypeFixedSizeBinary(ArrowTypeFixedSizeBinary *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32_t *) pos);

	node->byteWidth = fetchInt(&t, 0);
}

static void
readArrowTypeFixedSizeList(ArrowTypeFixedSizeList *node, const char *pos)
{
	FBTable		t= fetchFBTable((int32_t *) pos);

	node->listSize = fetchInt(&t, 0);
}

static void
readArrowTypeMap(ArrowTypeMap *node, const char *pos)
{
	FBTable		t= fetchFBTable((int32_t *) pos);

	node->keysSorted = fetchBool(&t, 0);
}

static void
readArrowTypeDuration(ArrowTypeDuration *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32_t *) pos);

	node->unit = __fetchShort(&t, 0, ArrowTimeUnit__MilliSecond);
	if (node->unit != ArrowTimeUnit__Second &&
		node->unit != ArrowTimeUnit__MilliSecond &&
		node->unit != ArrowTimeUnit__MicroSecond &&
		node->unit != ArrowTimeUnit__NanoSecond)
		Elog("ArrowTypeDuration has unknown unit (%d)", node->unit);
}

static void
readArrowType(ArrowType *type, int type_tag, const char *type_pos)
{
	memset(type, 0, sizeof(ArrowType));
	switch (type_tag)
	{
		case ArrowType__Null:
			INIT_ARROW_TYPE_NODE(type, Null);
			break;
		case ArrowType__Int:
			INIT_ARROW_TYPE_NODE(type, Int);
			if (type_pos)
				readArrowTypeInt(&type->Int, type_pos);
			break;
		case ArrowType__FloatingPoint:
			INIT_ARROW_TYPE_NODE(type, FloatingPoint);
			if (type_pos)
				readArrowTypeFloatingPoint(&type->FloatingPoint, type_pos);
			break;
		case ArrowType__Binary:
			INIT_ARROW_TYPE_NODE(type, Binary);
			break;
		case ArrowType__Utf8:
			INIT_ARROW_TYPE_NODE(type, Utf8);
			break;
		case ArrowType__Bool:
			INIT_ARROW_TYPE_NODE(type, Bool);
			break;
		case ArrowType__Decimal:
			INIT_ARROW_TYPE_NODE(type, Decimal);
			if (type_pos)
				readArrowTypeDecimal(&type->Decimal, type_pos);
			break;
		case ArrowType__Date:
			INIT_ARROW_TYPE_NODE(type, Date);
			if (type_pos)
				readArrowTypeDate(&type->Date, type_pos);
			break;
		case ArrowType__Time:
			INIT_ARROW_TYPE_NODE(type, Time);
			if (type_pos)
				readArrowTypeTime(&type->Time, type_pos);
			break;
		case ArrowType__Timestamp:
			INIT_ARROW_TYPE_NODE(type, Timestamp);
			if (type_pos)
				readArrowTypeTimestamp(&type->Timestamp, type_pos);
			break;
		case ArrowType__Interval:
			INIT_ARROW_TYPE_NODE(type, Interval);
			if (type_pos)
				readArrowTypeInterval(&type->Interval, type_pos);
			break;
		case ArrowType__List:
			INIT_ARROW_TYPE_NODE(type, List);
			break;
		case ArrowType__Struct:
			INIT_ARROW_TYPE_NODE(type, Struct);
			break;
		case ArrowType__Union:
			INIT_ARROW_TYPE_NODE(type, Union);
			if (type_pos)
				readArrowTypeUnion(&type->Union, type_pos);
			break;
		case ArrowType__FixedSizeBinary:
			INIT_ARROW_TYPE_NODE(type, FixedSizeBinary);
			if (type_pos)
				readArrowTypeFixedSizeBinary(&type->FixedSizeBinary, type_pos);
			break;
		case ArrowType__FixedSizeList:
			INIT_ARROW_TYPE_NODE(type, FixedSizeList);
			if (type_pos)
				readArrowTypeFixedSizeList(&type->FixedSizeList, type_pos);
			break;
		case ArrowType__Map:
			INIT_ARROW_TYPE_NODE(type, Map);
			if (type_pos)
				readArrowTypeMap(&type->Map, type_pos);
			break;
		case ArrowType__Duration:
			INIT_ARROW_TYPE_NODE(type, Duration);
			if (type_pos)
				readArrowTypeDuration(&type->Duration, type_pos);
			break;
		case ArrowType__LargeBinary:
			INIT_ARROW_TYPE_NODE(type, LargeBinary);
			break;
		case ArrowType__LargeUtf8:
			INIT_ARROW_TYPE_NODE(type, LargeUtf8);
			break;
		case ArrowType__LargeList:
			INIT_ARROW_TYPE_NODE(type, LargeList);
			break;
		default:
			printf("no suitable ArrowType__* tag for the code = %d", type_tag);
			break;
	}
}

static void
readArrowDictionaryEncoding(ArrowDictionaryEncoding *dict, const char *pos)
{
	FBTable		t = fetchFBTable((int32_t *)pos);
	const char *type_pos;

	memset(dict, 0, sizeof(ArrowDictionaryEncoding));
	INIT_ARROW_NODE(dict, DictionaryEncoding);
	dict->id		= fetchLong(&t, 0);
	type_pos		= fetchOffset(&t, 1);
	INIT_ARROW_TYPE_NODE(&dict->indexType, Int);
	readArrowTypeInt(&dict->indexType, type_pos);
	dict->isOrdered	= fetchBool(&t, 2);
}

static void
readArrowField(ArrowField *field, const char *pos)
{
	FBTable		t = fetchFBTable((int32_t *)pos);
	int			type_tag;
	const char *type_pos;
	const char *dict_pos;
	int32_t	   *vector;
	int			i, nitems;

	memset(field, 0, sizeof(ArrowField));
	INIT_ARROW_NODE(field, Field);
	field->name		= fetchString(&t, 0, &field->_name_len);
	field->nullable	= fetchBool(&t, 1);
	/* type */
	type_tag		= fetchChar(&t, 2);
	type_pos		= fetchOffset(&t, 3);
	readArrowType(&field->type, type_tag, type_pos);

	/* dictionary */
	dict_pos = fetchOffset(&t, 4);
	if (!dict_pos)
		field->dictionary = NULL;
	else
	{
		field->dictionary = palloc0(sizeof(ArrowDictionaryEncoding));
		readArrowDictionaryEncoding(field->dictionary, dict_pos);
	}

	/* children */
	vector = fetchVector(&t, 5, &nitems);
	if (nitems > 0)
	{
		field->children = palloc0(sizeof(ArrowField) * nitems);
		for (i=0; i < nitems; i++)
		{
			int		offset = vector[i];

			if (offset == 0)
				Elog("ArrowField has NULL-element in children[]");
			readArrowField(&field->children[i],
						   (const char *)&vector[i] + offset);
		}
	}
	field->_num_children = nitems;

	/* custom_metadata */
	vector = fetchVector(&t, 6, &nitems);
	if (nitems > 0)
	{
		field->custom_metadata = palloc0(sizeof(ArrowKeyValue) * nitems);
		for (i=0; i < nitems; i++)
		{
			int		offset = vector[i];

			if (offset == 0)
				Elog("ArrowField has NULL-element in custom_metadata[]");
			readArrowKeyValue(&field->custom_metadata[i],
							  (const char *)&vector[i] + offset);
		}
	}
	field->_num_custom_metadata = nitems;
}

static void
readArrowBodyCompression(ArrowBodyCompression *compression, const char *pos)
{
	FBTable		t = fetchFBTable((int32_t *)pos);

	memset(compression, 0, sizeof(ArrowBodyCompression));
	INIT_ARROW_NODE(compression, BodyCompression);
	compression->codec	= fetchChar(&t, 0);
	compression->method	= fetchChar(&t, 1);
}

static void
readArrowSchema(ArrowSchema *schema, const char *pos)
{
	FBTable		t = fetchFBTable((int32_t *)pos);
	int32_t	   *vector;
	int32_t		i, nitems;

	memset(schema, 0, sizeof(ArrowSchema));
	INIT_ARROW_NODE(schema, Schema);
	schema->endianness	= fetchBool(&t, 0);
	/* [ fields ]*/
	vector = fetchVector(&t, 1, &nitems);
	if (nitems > 0)
	{
		schema->fields = palloc0(sizeof(ArrowField) * nitems);
		for (i=0; i < nitems; i++)
		{
			int		offset = vector[i];

			if (offset == 0)
				Elog("ArrowSchema has NULL-element in fields[]");
			readArrowField(&schema->fields[i],
						   (const char *)&vector[i] + offset);
		}
	}
	schema->_num_fields = nitems;

	/* [ custom_metadata ] */
	vector = fetchVector(&t, 2, &nitems);
	if (nitems > 0)
	{
		schema->custom_metadata = palloc0(sizeof(ArrowKeyValue) * nitems);
		for (i=0; i < nitems; i++)
		{
			int		offset = vector[i];

			if (offset == 0)
				Elog("ArrowSchema has NULL-item in custom_metadata[]");
			readArrowKeyValue(&schema->custom_metadata[i],
							  (const char *)&vector[i] + offset);
		}
	}
	schema->_num_custom_metadata = nitems;

	/* [ features ] */
	vector = fetchVector(&t, 3, &nitems);
	if (nitems == 0)
		schema->features = NULL;
	else
	{
		schema->features = palloc0(sizeof(ArrowFeature) * nitems);
		for (i=0; i < nitems; i++)
			schema->features[i] = ((const uint64_t *)vector)[i];
	}
	schema->_num_features = nitems;
}

static size_t
readArrowFieldNode(ArrowFieldNode *node, const char *pos)
{
	struct {
		int64_t		length		__attribute__ ((aligned(8)));
		int64_t		null_count	__attribute__ ((aligned(8)));
	} *fmap = (void *) pos;

	memset(node, 0, sizeof(ArrowFieldNode));
	INIT_ARROW_NODE(node, FieldNode);
	node->length		= fmap->length;
	node->null_count	= fmap->null_count;

	return sizeof(*fmap);
}

static size_t
readArrowBuffer(ArrowBuffer *node, const char *pos)
{
	struct {
		int64_t		offset		__attribute__ ((aligned(8)));
		int64_t		length		__attribute__ ((aligned(8)));
	} *fmap = (void *) pos;

	memset(node, 0, sizeof(ArrowBuffer));
	INIT_ARROW_NODE(node, Buffer);
	node->offset		= fmap->offset;
	node->length		= fmap->length;

	return sizeof(*fmap);


}

static void
readArrowRecordBatch(ArrowRecordBatch *rbatch, const char *pos)
{
	FBTable		t = fetchFBTable((int32_t *)pos);
	const char *next;
	int			i, nitems;

	memset(rbatch, 0, sizeof(ArrowRecordBatch));
	INIT_ARROW_NODE(rbatch, RecordBatch);
	rbatch->length	= fetchLong(&t, 0);
	/* nodes: [FieldNode] */
	next = (const char *)fetchVector(&t, 1, &nitems);
	if (nitems > 0)
	{
		rbatch->nodes = palloc0(sizeof(ArrowFieldNode) * nitems);
		for (i=0; i < nitems; i++)
			next += readArrowFieldNode(&rbatch->nodes[i], next);
	}
	rbatch->_num_nodes = nitems;

	/* buffers: [Buffer] */
	next = (const char *)fetchVector(&t, 2, &nitems);
	if (nitems > 0)
	{
		rbatch->buffers = palloc0(sizeof(ArrowBuffer) * nitems);
		for (i=0; i < nitems; i++)
			next += readArrowBuffer(&rbatch->buffers[i], next);
	}
	rbatch->_num_buffers = nitems;

	/* (optional) BodyCompression  */
	next = (const char *)fetchOffset(&t, 3);
	if (next)
	{
		rbatch->compression = palloc0(sizeof(ArrowBodyCompression));
		readArrowBodyCompression(rbatch->compression, next);
	}
}

static void
readArrowDictionaryBatch(ArrowDictionaryBatch *dbatch, const char *pos)
{
	FBTable		t = fetchFBTable((int32_t *)pos);
	const char *next;

	memset(dbatch, 0, sizeof(ArrowDictionaryBatch));
	INIT_ARROW_NODE(dbatch, DictionaryBatch);
	dbatch->id	= fetchLong(&t, 0);
	next		= fetchOffset(&t, 1);
	readArrowRecordBatch(&dbatch->data, next);
	dbatch->isDelta = fetchBool(&t, 2);
}

static void
readArrowMessage(ArrowMessage *message, const char *pos)
{
	FBTable			t = fetchFBTable((int32_t *)pos);
	int				mtype;
	const char	   *next;

	memset(message, 0, sizeof(ArrowMessage));
	INIT_ARROW_NODE(message, Message);
	message->version	= fetchShort(&t, 0);
	mtype				= fetchChar(&t, 1);
	next				= fetchOffset(&t, 2);
	message->bodyLength	= fetchLong(&t, 3);

	if (message->version < ArrowMetadataVersion__V4 ||
		message->version > ArrowMetadataVersion__V5)
		Elog("metadata version %d is not supported", message->version);

	switch (mtype)
	{
		case ArrowMessageHeader__Schema:
			readArrowSchema(&message->body.schema, next);
			break;
		case ArrowMessageHeader__DictionaryBatch:
			readArrowDictionaryBatch(&message->body.dictionaryBatch, next);
			break;
		case ArrowMessageHeader__RecordBatch:
			readArrowRecordBatch(&message->body.recordBatch, next);
			break;
		case ArrowMessageHeader__Tensor:
			Elog("message type: Tensor is not implemented");
			break;
		case ArrowMessageHeader__SparseTensor:
			Elog("message type: SparseTensor is not implemented");
			break;
		default:
			Elog("unknown message header type: %d", mtype);
			break;
	}
}

/*
 * readArrowBlock (read inline structure)
 */
static size_t
readArrowBlock(ArrowBlock *node, const char *pos)
{
	struct {
		int64_t		offset			__attribute__ ((aligned(8)));
		int32_t		metaDataLength	__attribute__ ((aligned(8)));
		int64_t		bodyLength		__attribute__ ((aligned(8)));
	} *fmap = (void *) pos;

	memset(node, 0, sizeof(ArrowBlock));
	INIT_ARROW_NODE(node, Block);
	node->offset         = fmap->offset;
	node->metaDataLength = fmap->metaDataLength;
	node->bodyLength     = fmap->bodyLength;

	return sizeof(*fmap);
}

/*
 * readArrowFooter
 */
static void
readArrowFooter(ArrowFooter *node, const char *pos)
{
	FBTable			t = fetchFBTable((int32_t *)pos);
	const char	   *next;
	int				i, nitems;

	memset(node, 0, sizeof(ArrowFooter));
	INIT_ARROW_NODE(node, Footer);
	node->version	= fetchShort(&t, 0);
	/* schema */
	next = fetchOffset(&t, 1);
	readArrowSchema(&node->schema, next);
	/* [dictionaries] */
	next = (const char *)fetchVector(&t, 2, &nitems);
	if (nitems > 0)
	{
		node->dictionaries = palloc0(sizeof(ArrowBlock) * nitems);
		for (i=0; i < nitems; i++)
			next += readArrowBlock(&node->dictionaries[i], next);
		node->_num_dictionaries = nitems;
	}

	/* [recordBatches] */
	next = (const char *)fetchVector(&t, 3, &nitems);
	if (nitems > 0)
	{
		node->recordBatches = palloc0(sizeof(ArrowBlock) * nitems);
		for (i=0; i < nitems; i++)
			next += readArrowBlock(&node->recordBatches[i], next);
		node->_num_recordBatches = nitems;
	}
}

/*
 * readArrowFile - read the supplied apache arrow file
 */
#define ARROW_FILE_HEAD_SIGNATURE		"ARROW1\0\0"
#define ARROW_FILE_HEAD_SIGNATURE_SZ	(sizeof(ARROW_FILE_HEAD_SIGNATURE) - 1)
#define ARROW_FILE_TAIL_SIGNATURE		"ARROW1"
#define ARROW_FILE_TAIL_SIGNATURE_SZ	(sizeof(ARROW_FILE_TAIL_SIGNATURE) - 1)

#ifndef __PGSTROM_MODULE__
#define PG_TRY()								\
	if (true) {									\
		bool	__dummy__ __attribute__((unused))
#define PG_FINALLY()							\
	} else {									\
		bool __dummy__ __attribute__((unused))
#define PG_END_TRY()							\
	}
#endif

void
readArrowFileDesc(int fdesc, ArrowFileInfo *af_info)
{
	static long		__PAGE_SIZE = 0;
	size_t			file_sz;
	size_t			mmap_sz;
	char		   *mmap_head = NULL;
	char		   *mmap_tail = NULL;
	const char	   *pos;
	int32_t			offset;
	int32_t			i, nitems;

	memset(af_info, 0, sizeof(ArrowFileInfo));
	if (fstat(fdesc, &af_info->stat_buf) != 0)
		Elog("failed on fstat: %m");
	file_sz = af_info->stat_buf.st_size;
	if (__PAGE_SIZE == 0)
		__PAGE_SIZE = sysconf(_SC_PAGESIZE);
	mmap_sz = ((file_sz + __PAGE_SIZE - 1) & ~(__PAGE_SIZE - 1));
	mmap_head = mmap(NULL, mmap_sz, PROT_READ, MAP_SHARED, fdesc, 0);
	if (mmap_head == MAP_FAILED)
		Elog("failed on mmap: %m");
	mmap_tail = mmap_head + file_sz - ARROW_FILE_TAIL_SIGNATURE_SZ;

	/* check signature */
	PG_TRY();
	{
		if (memcmp(mmap_head,
				   ARROW_FILE_HEAD_SIGNATURE,
				   ARROW_FILE_HEAD_SIGNATURE_SZ) != 0 ||
			memcmp(mmap_tail,
				   ARROW_FILE_TAIL_SIGNATURE,
				   ARROW_FILE_TAIL_SIGNATURE_SZ) != 0)
		{
			Elog("Signature mismatch on Apache Arrow file");
		}

		/* Read Footer chunk */
		pos = mmap_tail - sizeof(int32_t);
		offset = *((int32_t *)pos);
		pos -= offset;
		offset = *((int32_t *)pos);
		readArrowFooter(&af_info->footer, pos + offset);

		/* Read DictionaryBatch chunks */
		nitems = af_info->footer._num_dictionaries;
		if (nitems > 0)
		{
			af_info->dictionaries = palloc0(nitems * sizeof(ArrowMessage));
			for (i=0; i < nitems; i++)
			{
				ArrowBlock	   *b = &af_info->footer.dictionaries[i];
				ArrowMessage   *m = &af_info->dictionaries[i];
				int32_t		   *ival = (int32_t *)(mmap_head + b->offset);
				int32_t			metaLength	__attribute__((unused));
				int32_t		   *headOffset;

				if (*ival == 0xffffffff)
				{
					metaLength = ival[1];
					headOffset = ival + 2;
				}
				else
				{
					/* Older format prior to Arrow v0.15 */
					metaLength = *ival;
					headOffset = ival + 1;
				}
				pos = (const char *)headOffset + *headOffset;
				readArrowMessage(m, pos);
			}
		}

		/* Read RecordBatch chunks */
		nitems = af_info->footer._num_recordBatches;
		if (nitems > 0)
		{
			af_info->recordBatches = palloc0(nitems * sizeof(ArrowMessage));
			for (i=0; i < nitems; i++)
			{
				ArrowBlock	   *b = &af_info->footer.recordBatches[i];
				ArrowMessage   *m = &af_info->recordBatches[i];
				int32_t		   *ival = (int32_t *)(mmap_head + b->offset);
				int32_t			metaLength	__attribute__((unused));
				int32_t		   *headOffset;

				if (*ival == 0xffffffff)
				{
					metaLength = ival[1];
					headOffset = ival + 2;
				}
				else
				{
					/* Older format prior to Arrow v0.15 */
					metaLength = *ival;
					headOffset = ival + 1;
				}
				pos = (const char *)headOffset + *headOffset;
				readArrowMessage(m, pos);
			}
		}
		munmap(mmap_head, mmap_sz);
	}
	PG_FINALLY();
	{
		munmap(mmap_head, mmap_sz);
	}
	PG_END_TRY();
}
