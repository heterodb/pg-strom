/*
 * arrow_nodes.c
 *
 * Routines to handle ArrowNode objects, intermediation of PostgreSQL types
 * and Apache Arrow types.
 * ----
 * Copyright 2011-2019 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2019 (C) The PG-Strom Development Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */
#ifdef __PG2ARROW__
#include "postgres.h"
#include "port/pg_bswap.h"
#include "utils/date.h"
#include "utils/timestamp.h"
typedef struct SQLbuffer	StringInfoData;
typedef struct SQLbuffer   *StringInfo;
#if PG_VERSION_NUM < 110000
#ifdef WORDS_BIGENDIAN
#define __ntoh16(x)			(x)
#define __ntoh32(x)			(x)
#define __ntoh64(x)			(x)
#else
#define __ntoh16(x)			ntohs(x)
#define __ntoh32(x)			BSWAP32(x)
#define __ntoh64(x)			BSWAP64(x)
#endif
#else	/* >=PG11 */
#define __ntoh16(x)			pg_ntoh16(x)
#define __ntoh32(x)			pg_ntoh32(x)
#define __ntoh64(x)			pg_ntoh64(x)
#endif	/* >=PG11 */
#else	/* __PG2ARROW__ */
#include "pg_strom.h"
#define __ntoh16(x)			(x)
#define __ntoh32(x)			(x)
#define __ntoh64(x)			(x)
#endif	/* !__PG2ARROW__ */
#include "arrow_ipc.h"

/*
 * Dump support of ArrowNode
 */
static void
__dumpArrowNode(StringInfo str, ArrowNode *node)
{
	node->dumpArrowNode(str, node);
}

static void
__dumpArrowNodeSimple(StringInfo str, ArrowNode *node)
{
	appendStringInfo(str, "{%s}", node->tagName);
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

static void
__dumpArrowTypeInt(StringInfo str, ArrowNode *node)
{
	ArrowTypeInt   *i = (ArrowTypeInt *)node;

	appendStringInfo(str,"{%s%d}",
					 i->is_signed ? "Int" : "Uint",
					 i->bitWidth);
}

static void
__dumpArrowTypeFloatingPoint(StringInfo str, ArrowNode *node)
{
	ArrowTypeFloatingPoint *f = (ArrowTypeFloatingPoint *)node;

	appendStringInfo(
		str,"{Float%s}",
		ArrowPrecisionAsCstring(f->precision));
}

static void
__dumpArrowTypeDecimal(StringInfo str, ArrowNode *node)
{
	ArrowTypeDecimal *d = (ArrowTypeDecimal *)node;

	appendStringInfo(str,"{Decimal: precision=%d, scale=%d}",
					 d->precision,
					 d->scale);
}

static void
__dumpArrowTypeDate(StringInfo str, ArrowNode *node)
{
	ArrowTypeDate *d = (ArrowTypeDate *)node;

	appendStringInfo(
		str,"{Date: unit=%s}",
		ArrowDateUnitAsCstring(d->unit));
}

static void
__dumpArrowTypeTime(StringInfo str, ArrowNode *node)
{
	ArrowTypeTime *t = (ArrowTypeTime *)node;

	appendStringInfo(
		str,"{Time: unit=%s}",
		ArrowTimeUnitAsCstring(t->unit));
}

static void
__dumpArrowTypeTimestamp(StringInfo str, ArrowNode *node)
{
	ArrowTypeTimestamp *t = (ArrowTypeTimestamp *)node;

	appendStringInfo(
		str,"{Timestamp: unit=%s}",
		ArrowTimeUnitAsCstring(t->unit));
}

static void
__dumpArrowTypeInterval(StringInfo str, ArrowNode *node)
{
	ArrowTypeInterval *t = (ArrowTypeInterval *)node;

	appendStringInfo(
		str,"{Interval: unit=%s}",
		ArrowIntervalUnitAsCstring(t->unit));
}

static void
__dumpArrowTypeUnion(StringInfo str, ArrowNode *node)
{
	ArrowTypeUnion *u = (ArrowTypeUnion *)node;
	int			i;

	appendStringInfo(
		str,"{Union: mode=%s, typeIds=[",
		u->mode == ArrowUnionMode__Sparse ? "Sparse" :
		u->mode == ArrowUnionMode__Dense ? "Dense" : "???");
	for (i=0; i < u->_num_typeIds; i++)
		appendStringInfo(str, "%s%d", i > 0 ? ", " : " ",
						 u->typeIds[i]);
	appendStringInfo(str, "]}");
}

static void
__dumpArrowTypeFixedSizeBinary(StringInfo str, ArrowNode *node)
{
	ArrowTypeFixedSizeBinary *fb = (ArrowTypeFixedSizeBinary *)node;

	appendStringInfo(
		str,"{FixedSizeBinary: byteWidth=%d}", fb->byteWidth);
}

static void
__dumpArrowTypeFixedSizeList(StringInfo str, ArrowNode *node)
{
	ArrowTypeFixedSizeList *fl = (ArrowTypeFixedSizeList *)node;

	appendStringInfo(
		str,"{FixedSizeList: listSize=%d}", fl->listSize);
}

static void
__dumpArrowTypeMap(StringInfo str, ArrowNode *node)
{
	ArrowTypeMap *m = (ArrowTypeMap *)node;

	appendStringInfo(
		str,"{Map: keysSorted=%s}", m->keysSorted ? "true" : "false");
}

static void
__dumpArrowTypeDuration(StringInfo str, ArrowNode *node)
{
	ArrowTypeDuration *d = (ArrowTypeDuration *)node;

	appendStringInfo(
		str,"{Duration: unit=%s}",
		ArrowTimeUnitAsCstring(d->unit));
}

static void
__dumpArrowKeyValue(StringInfo str, ArrowNode *node)
{
	ArrowKeyValue *kv = (ArrowKeyValue *)node;

	appendStringInfo(str,"{KeyValue: key=\"%s\" value=\"%s\"}",
					 kv->key ? kv->key : "",
					 kv->value ? kv->value : "");
}

static void
__dumpArrowDictionaryEncoding(StringInfo str, ArrowNode *node)
{
	ArrowDictionaryEncoding *d = (ArrowDictionaryEncoding *)node;

	appendStringInfo(str,"{DictionaryEncoding: id=%ld, indexType=", d->id);
	__dumpArrowNode(str, (ArrowNode *)&d->indexType);
	appendStringInfo(str,", isOrdered=%s}",
					 d->isOrdered ? "true" : "false");
}

static void
__dumpArrowField(StringInfo str, ArrowNode *node)
{
	ArrowField *f = (ArrowField *)node;
	int		i;

	appendStringInfo(str, "{Field: name=\"%s\", nullable=%s, type=",
					 f->name ? f->name : "",
					 f->nullable ? "true" : "false");
	__dumpArrowNode(str, (ArrowNode *)&f->type);
	if (f->dictionary.indexType.node.tag == ArrowNodeTag__Int)
	{
		appendStringInfo(str, ", dictionary=");
		__dumpArrowNode(str, (ArrowNode *)&f->dictionary);
	}
	appendStringInfo(str, ", children=[");
	for (i=0; i < f->_num_children; i++)
	{
		if (i > 0)
			appendStringInfo(str, ", ");
		__dumpArrowNode(str, (ArrowNode *)&f->children[i]);
	}
	appendStringInfo(str, "], custom_metadata=[");
	for (i=0; i < f->_num_custom_metadata; i++)
	{
		if (i > 0)
			appendStringInfo(str, ", ");
		__dumpArrowNode(str, (ArrowNode *)&f->custom_metadata[i]);
	}
	appendStringInfo(str, "]}");
}

static void
__dumpArrowFieldNode(StringInfo str, ArrowNode *node)
{
	appendStringInfo(
		str, "{FieldNode: length=%ld, null_count=%ld}",
		((ArrowFieldNode *)node)->length,
		((ArrowFieldNode *)node)->null_count);
}

static void
__dumpArrowBuffer(StringInfo str, ArrowNode *node)
{
	appendStringInfo(
		str, "{Buffer: offset=%ld, length=%ld}",
		((ArrowBuffer *)node)->offset,
		((ArrowBuffer *)node)->length);
}

static void
__dumpArrowSchema(StringInfo str, ArrowNode *node)
{
	ArrowSchema *s = (ArrowSchema *)node;
	int		i;

	appendStringInfo(
		str, "{Schema: endianness=%s, fields=[",
		s->endianness == ArrowEndianness__Little ? "little" :
		s->endianness == ArrowEndianness__Big ? "big" : "???");
	for (i=0; i < s->_num_fields; i++)
	{
		if (i > 0)
			appendStringInfo(str, ", ");
		__dumpArrowNode(str, (ArrowNode *)&s->fields[i]);
	}
	appendStringInfo(str, "], custom_metadata=[");
	for (i=0; i < s->_num_custom_metadata; i++)
	{
		if (i > 0)
			appendStringInfo(str, ", ");
		__dumpArrowNode(str, (ArrowNode *)&s->custom_metadata[i]);
	}
	appendStringInfo(str, "]}");
}

static void
__dumpArrowRecordBatch(StringInfo str, ArrowNode *node)
{
	ArrowRecordBatch *r = (ArrowRecordBatch *) node;
	int		i;

	appendStringInfo(str, "{RecordBatch: length=%ld, nodes=[", r->length);
	for (i=0; i < r->_num_nodes; i++)
	{
		if (i > 0)
			appendStringInfo(str, ", ");
		__dumpArrowNode(str, (ArrowNode *)&r->nodes[i]);
	}
	appendStringInfo(str, "], buffers=[");
	for (i=0; i < r->_num_buffers; i++)
	{
		if (i > 0)
			appendStringInfo(str, ", ");
		__dumpArrowNode(str, (ArrowNode *)&r->buffers[i]);
	}
	appendStringInfo(str,"]}");
}

static void
__dumpArrowDictionaryBatch(StringInfo str, ArrowNode *node)
{
	ArrowDictionaryBatch *d = (ArrowDictionaryBatch *)node;

	appendStringInfo(str, "{DictionaryBatch: id=%ld, data=", d->id);
	__dumpArrowNode(str, (ArrowNode *)&d->data);
	appendStringInfo(str, ", isDelta=%s}",
					 d->isDelta ? "true" : "false");
}

static void
__dumpArrowMessage(StringInfo str, ArrowNode *node)
{
	ArrowMessage *m = (ArrowMessage *)node;

	appendStringInfo(
		str, "{Message: version=%s, body=",
		m->version == ArrowMetadataVersion__V1 ? "V1" :
		m->version == ArrowMetadataVersion__V2 ? "V2" :
		m->version == ArrowMetadataVersion__V3 ? "V3" :
		m->version == ArrowMetadataVersion__V4 ? "V4" : "???");
	__dumpArrowNode(str, (ArrowNode *)&m->body);
	appendStringInfo(str, ", bodyLength=%lu}", m->bodyLength);
}

static void
__dumpArrowBlock(StringInfo str, ArrowNode *node)
{
	appendStringInfo(
		str, "{Block: offset=%ld, metaDataLength=%d bodyLength=%ld}",
		((ArrowBlock *)node)->offset,
		((ArrowBlock *)node)->metaDataLength,
		((ArrowBlock *)node)->bodyLength);
}

static void
__dumpArrowFooter(StringInfo str, ArrowNode *node)
{
	ArrowFooter *f = (ArrowFooter *)node;
	int		i;

	appendStringInfo(
		str, "{Footer: version=%s, schema=",
		f->version == ArrowMetadataVersion__V1 ? "V1" :
		f->version == ArrowMetadataVersion__V2 ? "V2" :
		f->version == ArrowMetadataVersion__V3 ? "V3" :
		f->version == ArrowMetadataVersion__V4 ? "V4" : "???");
	__dumpArrowNode(str, (ArrowNode *)&f->schema);
	appendStringInfo(str, ", dictionaries=[");
	for (i=0; i < f->_num_dictionaries; i++)
	{
		if (i > 0)
			appendStringInfo(str, ", ");
		__dumpArrowNode(str, (ArrowNode *)&f->dictionaries[i]);
	}
	appendStringInfo(str, "], recordBatches=[");
	for (i=0; i < f->_num_recordBatches; i++)
	{
		if (i > 0)
			appendStringInfo(str, ", ");
		__dumpArrowNode(str, (ArrowNode *)&f->recordBatches[i]);
	}
	appendStringInfo(str, "]}");
}

char *
dumpArrowNode(ArrowNode *node)
{
	StringInfoData str;

	initStringInfo(&str);
	__dumpArrowNode(&str, node);

	return str.data;
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
#define COPY_VECTOR(FIELD, NODETYPE)								\
	do {															\
		int		j;													\
																	\
		(dest)->FIELD = palloc(sizeof(NODETYPE) * (src)->_num_##FIELD);	\
		for (j=0; j < (src)->_num_##FIELD; j++)						\
			__copy##NODETYPE(&(dest)->FIELD[j], &(src)->FIELD[j]);	\
		(dest)->_num_##FIELD = (src)->_num_##FIELD;					\
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
	dest->typeIds = palloc(sizeof(int32) * src->_num_typeIds);
	memcpy(dest->typeIds, src->typeIds, sizeof(int32) * src->_num_typeIds);
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
	__copyArrowDictionaryEncoding(&dest->dictionary, &src->dictionary);
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

void
copyArrowNode(ArrowNode *dest, const ArrowNode *src)
{
	src->copyArrowNode(dest, src);
}

/*
 * CString representation of arrow type name
 */
static size_t
__arrowTypeName(char *buf, size_t len, ArrowField *field)
{
	ArrowType  *t = &field->type;
	size_t		sz = 0;
	int			j;

	switch (t->node.tag)
	{
		case ArrowNodeTag__Null:
			sz = snprintf(buf, len, "Null");
			break;
		case ArrowNodeTag__Int:
			sz = snprintf(buf, len, "%s%d",
						  t->Int.is_signed ? "Int" : "Uint",
						  t->Int.bitWidth);
			break;
		case ArrowNodeTag__FloatingPoint:
			sz = snprintf(
				buf, len, "Float%s",
				ArrowPrecisionAsCstring(t->FloatingPoint.precision));
			break;
		case ArrowNodeTag__Utf8:
			sz = snprintf(buf, len, "Utf8");
			break;
		case ArrowNodeTag__Binary:
			sz = snprintf(buf, len, "Binary");
			break;
		case ArrowNodeTag__Bool:
			sz = snprintf(buf, len, "Bool");
			break;
		case ArrowNodeTag__Decimal:
			if (t->Decimal.scale == 0)
				sz = snprintf(buf, len, "Decimal(%d)",
							  t->Decimal.precision);
			else
				sz = snprintf(buf, len, "Decimal(%d,%d)",
							  t->Decimal.precision,
							  t->Decimal.scale);
			break;
		case ArrowNodeTag__Date:
			sz = snprintf(
				buf, len, "Date[%s]",
				ArrowDateUnitAsCstring(t->Date.unit));
			break;
		case ArrowNodeTag__Time:
			sz = snprintf(buf, len, "Time[%s]",
						  ArrowTimeUnitAsCstring(t->Time.unit));
			break;
		case ArrowNodeTag__Timestamp:
			sz = snprintf(buf, len, "Timestamp[%s]",
						  ArrowTimeUnitAsCstring(t->Timestamp.unit));
			break;
		case ArrowNodeTag__Interval:
			sz = snprintf(buf, len, "Interval[%s]",
						  ArrowIntervalUnitAsCstring(t->Interval.unit));
			break;
		case ArrowNodeTag__List:
			if (field->_num_children != 1)
				Elog("corrupted List data type");
			sz = __arrowTypeName(buf, len, &field->children[0]);
			sz += snprintf(buf+sz, len-sz, "[]");
			break;

		case ArrowNodeTag__Struct:
			sz += snprintf(buf+sz, len-sz, "Struct(");
			for (j=0; j < field->_num_children; j++)
			{
				if (j > 0)
					sz += snprintf(buf+sz, len-sz, ", ");
				sz += __arrowTypeName(buf+sz, len-sz, &field->children[j]);
			}
			sz += snprintf(buf+sz, len-sz, ")");
			break;

		case ArrowNodeTag__Union:
			sz = snprintf(buf, len, "Union");
			break;
		case ArrowNodeTag__FixedSizeBinary:
			sz = snprintf(buf, len, "FixedSizeBinary(%d)",
						  t->FixedSizeBinary.byteWidth);
			break;
		case ArrowNodeTag__FixedSizeList:
			sz = snprintf(buf, len, "FixedSizeList[%d]",
						  t->FixedSizeList.listSize);
			break;
		case ArrowNodeTag__Map:
			sz = snprintf(buf, len, "Map");
			break;
		case ArrowNodeTag__Duration:
			sz = snprintf(buf, len, "Duration[%s]",
						  ArrowTimeUnitAsCstring(t->Duration.unit));
			break;
		case ArrowNodeTag__LargeBinary:
			sz = snprintf(buf, len, "LargeBinary");
			break;
		case ArrowNodeTag__LargeUtf8:
			sz = snprintf(buf, len, "LargeUtf8");
			break;
		case ArrowNodeTag__LargeList:
			sz = snprintf(buf, len, "LargeList");
			break;
		default:
			Elog("unknown Arrow type");
	}
	return sz;
}

char *
arrowTypeName(ArrowField *field)
{
	char	buf[1024];

	__arrowTypeName(buf, sizeof(buf), field);

	return pstrdup(buf);
}

/* ------------------------------------------------
 *
 * Routines to initialize Apache Arrow nodes
 *
 * ------------------------------------------------
 */
typedef void (*dumpArrowNode_f)(StringInfo str, ArrowNode *node);
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
	uint16		vlen;	/* vtable length */
	uint16		tlen;	/* table length */
	uint16		offset[FLEXIBLE_ARRAY_MEMBER];
} FBVtable;

typedef struct
{
	int32	   *table;
	FBVtable   *vtable;
} FBTable;

static inline FBTable
fetchFBTable(void *p_table)
{
	FBTable		t;

	t.table  = (int32 *)p_table;
	t.vtable = (FBVtable *)((char *)p_table - *t.table);

	return t;
}

static inline void *
__fetchPointer(FBTable *t, int index)
{
	FBVtable   *vtable = t->vtable;

	if (offsetof(FBVtable, offset[index]) < vtable->vlen)
	{
		uint16		offset = vtable->offset[index];

		assert(offset < vtable->tlen);
		if (offset)
			return (char *)t->table + offset;
	}
	return NULL;
}

static inline bool
fetchBool(FBTable *t, int index)
{
	bool	   *ptr = __fetchPointer(t, index);
	return (ptr ? *ptr : false);
}

static inline int8
fetchChar(FBTable *t, int index)
{
	int8	   *ptr = __fetchPointer(t, index);
	return (ptr ? *ptr : 0);
}

static inline int16
fetchShort(FBTable *t, int index)
{
	int16	  *ptr = __fetchPointer(t, index);
	return (ptr ? *ptr : 0);
}

static inline int32
fetchInt(FBTable *t, int index)
{
	int32	  *ptr = __fetchPointer(t, index);
	return (ptr ? *ptr : 0);
}

static inline int64
fetchLong(FBTable *t, int index)
{
	int64	  *ptr = __fetchPointer(t, index);
	return (ptr ? *ptr : 0);
}

static inline void *
fetchOffset(FBTable *t, int index)
{
	int32  *ptr = __fetchPointer(t, index);
	return (ptr ? (char *)ptr + *ptr : NULL);
}

static inline const char *
fetchString(FBTable *t, int index, int *p_strlen)
{
	int32  *ptr = fetchOffset(t, index);
	int32	len = 0;
	char   *temp;

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

static inline int32 *
fetchVector(FBTable *t, int index, int *p_nitems)
{
	int32  *vector = fetchOffset(t, index);

	if (!vector)
		*p_nitems = 0;
	else
		*p_nitems = *vector++;
	return vector;
}

static void
readArrowKeyValue(ArrowKeyValue *kv, const char *pos)
{
	FBTable		t = fetchFBTable((int32 *)pos);

	memset(kv, 0, sizeof(ArrowKeyValue));
	INIT_ARROW_NODE(kv, KeyValue);
	kv->key     = fetchString(&t, 0, &kv->_key_len);
	kv->value   = fetchString(&t, 1, &kv->_value_len);
}

static void
readArrowTypeInt(ArrowTypeInt *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32 *) pos);

	node->bitWidth  = fetchInt(&t, 0);
	node->is_signed = fetchBool(&t, 1);
	if (node->bitWidth != 8  && node->bitWidth != 16 &&
		node->bitWidth != 32 && node->bitWidth != 64)
		Elog("ArrowTypeInt has unknown bitWidth (%d)", node->bitWidth);
}

static void
readArrowTypeFloatingPoint(ArrowTypeFloatingPoint *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32 *) pos);

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
	FBTable		t = fetchFBTable((int32 *) pos);

	node->precision = fetchInt(&t, 0);
	node->scale     = fetchInt(&t, 1);
}

static void
readArrowTypeDate(ArrowTypeDate *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32 *) pos);
	int16	   *ptr;

	/* Date->unit has non-zero default value */
	ptr = __fetchPointer(&t, 0);
	node->unit = (ptr != NULL ? *ptr : ArrowDateUnit__MilliSecond);
	if (node->unit != ArrowDateUnit__Day &&
		node->unit != ArrowDateUnit__MilliSecond)
		Elog("ArrowTypeDate has unknown unit (%d)", node->unit);
}

static void
readArrowTypeTime(ArrowTypeTime *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32 *) pos);

	node->unit = fetchShort(&t, 0);
	node->bitWidth = fetchInt(&t, 1);
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
	FBTable		t = fetchFBTable((int32 *) pos);

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
	FBTable		t = fetchFBTable((int32 *) pos);

	node->unit = fetchShort(&t, 0);
	if (node->unit != ArrowIntervalUnit__Year_Month &&
		node->unit != ArrowIntervalUnit__Day_Time)
		Elog("ArrowTypeInterval has unknown unit (%d)", node->unit);
}

static void
readArrowTypeUnion(ArrowTypeUnion *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32 *) pos);
	int32	   *vector;
	int32		nitems;

	node->mode = fetchShort(&t, 0);
	vector = fetchVector(&t, 1, &nitems);
	if (nitems == 0)
		node->typeIds = NULL;
	else
	{
		node->typeIds = palloc0(sizeof(int32) * nitems);
		memcpy(node->typeIds, vector, sizeof(int32) * nitems);
	}
	node->_num_typeIds = nitems;
}

static void
readArrowTypeFixedSizeBinary(ArrowTypeFixedSizeBinary *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32 *) pos);

	node->byteWidth = fetchInt(&t, 0);
}

static void
readArrowTypeFixedSizeList(ArrowTypeFixedSizeList *node, const char *pos)
{
	FBTable		t= fetchFBTable((int32 *) pos);

	node->listSize = fetchInt(&t, 0);
}

static void
readArrowTypeMap(ArrowTypeMap *node, const char *pos)
{
	FBTable		t= fetchFBTable((int32 *) pos);

	node->keysSorted = fetchBool(&t, 0);
}

static void
readArrowTypeDuration(ArrowTypeDuration *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32 *) pos);

	node->unit = fetchShort(&t, 0);
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
	FBTable		t = fetchFBTable((int32 *)pos);
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
	FBTable		t = fetchFBTable((int32 *)pos);
	int			type_tag;
	const char *type_pos;
	const char *dict_pos;
	int32	   *vector;
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
	if (dict_pos)
		readArrowDictionaryEncoding(&field->dictionary, dict_pos);

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
readArrowSchema(ArrowSchema *schema, const char *pos)
{
	FBTable		t = fetchFBTable((int32 *)pos);
	int32	   *vector;
	int32		i, nitems;

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
}

static size_t
readArrowFieldNode(ArrowFieldNode *node, const char *pos)
{
	struct {
		int64		length		__attribute__ ((aligned(8)));
		int64		null_count	__attribute__ ((aligned(8)));
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
		int64		offset		__attribute__ ((aligned(8)));
		int64		length		__attribute__ ((aligned(8)));
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
	FBTable		t = fetchFBTable((int32 *)pos);
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
}

static void
readArrowDictionaryBatch(ArrowDictionaryBatch *dbatch, const char *pos)
{
	FBTable		t = fetchFBTable((int32 *)pos);
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
	FBTable			t = fetchFBTable((int32 *)pos);
	int				mtype;
	const char	   *next;

	memset(message, 0, sizeof(ArrowMessage));
	INIT_ARROW_NODE(message, Message);
	message->version	= fetchShort(&t, 0);
	mtype				= fetchChar(&t, 1);
	next				= fetchOffset(&t, 2);
	message->bodyLength	= fetchLong(&t, 3);

	if (message->version != ArrowMetadataVersion__V4)
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
		int64		offset			__attribute__ ((aligned(8)));
		int32		metaDataLength	__attribute__ ((aligned(8)));
		int64		bodyLength		__attribute__ ((aligned(8)));
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
	FBTable			t = fetchFBTable((int32 *)pos);
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

void
readArrowFileDesc(int fdesc, ArrowFileInfo *af_info)
{
	size_t			file_sz;
	size_t			mmap_sz;
	char		   *mmap_head = NULL;
	char		   *mmap_tail = NULL;
	const char	   *pos;
	int32			offset;
	int32			i, nitems;

	memset(af_info, 0, sizeof(ArrowFileInfo));
	if (fstat(fdesc, &af_info->stat_buf) != 0)
		Elog("failed on fstat: %m");
	file_sz = af_info->stat_buf.st_size;
	mmap_sz = TYPEALIGN(sysconf(_SC_PAGESIZE), file_sz);
	mmap_head = mmap(NULL, mmap_sz, PROT_READ, MAP_SHARED, fdesc, 0);
	if (mmap_head == MAP_FAILED)
		Elog("failed on mmap: %m");
	mmap_tail = mmap_head + file_sz - ARROW_FILE_TAIL_SIGNATURE_SZ;

	/* check signature */
	if (memcmp(mmap_head,
			   ARROW_FILE_HEAD_SIGNATURE,
			   ARROW_FILE_HEAD_SIGNATURE_SZ) != 0 ||
		memcmp(mmap_tail,
			   ARROW_FILE_TAIL_SIGNATURE,
			   ARROW_FILE_TAIL_SIGNATURE_SZ) != 0)
	{
		munmap(mmap_head, mmap_sz);
		Elog("Signature mismatch on Apache Arrow file");
	}

	/* Read Footer chunk */
	pos = mmap_tail - sizeof(int32);
	offset = *((int32 *)pos);
	pos -= offset;
	offset = *((int32 *)pos);
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
			int32		   *ival = (int32 *)(mmap_head + b->offset);
			int32			metaLength	__attribute__((unused));
			int32		   *headOffset;

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
			int32		   *ival = (int32 *)(mmap_head + b->offset);
			int32			metaLength	__attribute__((unused));
			int32		   *headOffset;

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

/* ----------------------------------------------------------------
 *
 * put_value handler for each data types (optional)
 *
 * ----------------------------------------------------------------
 */
static void
put_bool_value(SQLfield *attr, const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;
	int8		value;

	if (!addr)
	{
		attr->nullcount++;
		sql_buffer_clrbit(&attr->nullmap, row_index);
		sql_buffer_clrbit(&attr->values,  row_index);
	}
	else
	{
		value = *((const int8 *)addr);
		sql_buffer_setbit(&attr->nullmap, row_index);
		if (value)
			sql_buffer_setbit(&attr->values,  row_index);
		else
			sql_buffer_clrbit(&attr->values,  row_index);
	}
}

static inline void
put_inline_null_value(SQLfield *attr, size_t row_index, int sz)
{
	attr->nullcount++;
	sql_buffer_clrbit(&attr->nullmap, row_index);
	sql_buffer_append_zero(&attr->values, sz);
}

/*
 * IntXX/UintXX
 */
static void
put_int8_value(SQLfield *attr, const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;
	uint8		value;

	if (!addr)
		put_inline_null_value(attr, row_index, sizeof(uint8));
	else
	{
		assert(sz == sizeof(uint8));
		value = *((const uint8 *)addr);

		if (!attr->arrow_type.Int.is_signed && value > INT8_MAX)
			Elog("Uint8 cannot store negative values");

		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &value, sizeof(uint8));
	}
}

static void
put_int16_value(SQLfield *attr, const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;
	uint16		value;

	if (!addr)
		put_inline_null_value(attr, row_index, sizeof(uint16));
	else
	{
		assert(sz == sizeof(uint16));
		value = __ntoh16(*((const uint16 *)addr));
		if (!attr->arrow_type.Int.is_signed && value > INT16_MAX)
			Elog("Uint16 cannot store negative values");
		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &value, sz);
	}
}

static void
put_int32_value(SQLfield *attr, const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;
	uint32		value;

	if (!addr)
		put_inline_null_value(attr, row_index, sizeof(uint32));
	else
	{
		assert(sz == sizeof(uint32));
		value = __ntoh32(*((const uint32 *)addr));
		if (!attr->arrow_type.Int.is_signed && value > INT32_MAX)
			Elog("Uint32 cannot store negative values");
		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &value, sz);
	}
}

static void
put_int64_value(SQLfield *attr, const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;
	uint64		value;

	if (!addr)
		put_inline_null_value(attr, row_index, sizeof(uint64));
	else
	{
		assert(sz == sizeof(uint64));
		value = __ntoh64(*((const uint64 *)addr));
		if (!attr->arrow_type.Int.is_signed && value > INT64_MAX)
			Elog("Uint64 cannot store negative values");
		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &value, sz);
	}
}

/*
 * FloatingPointXX
 */
static void
put_float16_value(SQLfield *attr, const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;
	uint16		value;

	if (!addr)
		put_inline_null_value(attr, row_index, sizeof(uint16));
	else
	{
		assert(sz == sizeof(uint16));
		value = __ntoh16(*((const uint16 *)addr));
		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &value, sz);
	}
}

static void
put_float32_value(SQLfield *attr, const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;
	uint32		value;

	if (!addr)
		put_inline_null_value(attr, row_index, sizeof(uint32));
	else
	{
		assert(sz == sizeof(uint32));
		value = __ntoh32(*((const uint32 *)addr));
		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &value, sz);
	}
}

static void
put_float64_value(SQLfield *attr, const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;
	uint64		value;

	if (!addr)
		put_inline_null_value(attr, row_index, sizeof(uint64));
	else
	{
		assert(sz == sizeof(uint64));
		value = __ntoh64(*((const uint64 *)addr));
		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &value, sz);
	}
}

/*
 * Decimal
 */
#ifdef PG_INT128_TYPE
/* parameters of Numeric type */
#define NUMERIC_DSCALE_MASK	0x3FFF
#define NUMERIC_SIGN_MASK	0xC000
#define NUMERIC_POS         0x0000
#define NUMERIC_NEG         0x4000
#define NUMERIC_NAN         0xC000

#define NBASE				10000
#define HALF_NBASE			5000
#define DEC_DIGITS			4	/* decimal digits per NBASE digit */
#define MUL_GUARD_DIGITS    2	/* these are measured in NBASE digits */
#define DIV_GUARD_DIGITS	4
typedef int16				NumericDigit;
typedef struct NumericVar
{
	int			ndigits;	/* # of digits in digits[] - can be 0! */
	int			weight;		/* weight of first digit */
	int			sign;		/* NUMERIC_POS, NUMERIC_NEG, or NUMERIC_NAN */
	int			dscale;		/* display scale */
	NumericDigit *digits;	/* base-NBASE digits */
} NumericVar;

#ifndef __PG2ARROW__
#define NUMERIC_SHORT_SIGN_MASK			0x2000
#define NUMERIC_SHORT_DSCALE_MASK		0x1F80
#define NUMERIC_SHORT_DSCALE_SHIFT		7
#define NUMERIC_SHORT_WEIGHT_SIGN_MASK	0x0040
#define NUMERIC_SHORT_WEIGHT_MASK		0x003F

static void
init_var_from_num(NumericVar *nv, const char *addr, int sz)
{
	uint16		n_header = *((uint16 *)addr);

	/* NUMERIC_HEADER_IS_SHORT */
	if ((n_header & 0x8000) != 0)
	{
		/* short format */
		const struct {
			uint16	n_header;
			NumericDigit n_data[FLEXIBLE_ARRAY_MEMBER];
		}  *n_short = (const void *)addr;
		size_t		hoff = ((uintptr_t)n_short->n_data - (uintptr_t)n_short);

		nv->ndigits = (sz - hoff) / sizeof(NumericDigit);
		nv->weight = (n_short->n_header & NUMERIC_SHORT_WEIGHT_MASK);
		if ((n_short->n_header & NUMERIC_SHORT_WEIGHT_SIGN_MASK) != 0)
			nv->weight |= NUMERIC_SHORT_WEIGHT_MASK;	/* negative value */
		nv->sign = ((n_short->n_header & NUMERIC_SHORT_SIGN_MASK) != 0
					? NUMERIC_NEG
					: NUMERIC_POS);
		nv->dscale = (n_short->n_header & NUMERIC_SHORT_DSCALE_MASK) >> NUMERIC_SHORT_DSCALE_SHIFT;
		nv->digits = (NumericDigit *)n_short->n_data;
	}
	else
	{
		/* long format */
		const struct {
			uint16      n_sign_dscale;  /* Sign + display scale */
			int16       n_weight;       /* Weight of 1st digit  */
			NumericDigit n_data[FLEXIBLE_ARRAY_MEMBER]; /* Digits */
		}  *n_long = (const void *)addr;
		size_t		hoff = ((uintptr_t)n_long->n_data - (uintptr_t)n_long);

		assert(sz >= hoff);
		nv->ndigits = (sz - hoff) / sizeof(NumericDigit);
		nv->weight = n_long->n_weight;
		nv->sign   = (n_long->n_sign_dscale & NUMERIC_SIGN_MASK);
		nv->dscale = (n_long->n_sign_dscale & NUMERIC_DSCALE_MASK);
		nv->digits = (NumericDigit *)n_long->n_data;
	}
}
#endif	/* !__PG2ARROW__ */

static void
put_decimal_value(SQLfield *attr,
			const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;

	if (!addr)
		put_inline_null_value(attr, row_index, sizeof(int128));
	else
	{
		NumericVar		nv;
		int				scale = attr->arrow_type.Decimal.scale;
		int128			value = 0;
		int				d, dig;
#ifdef __PG2ARROW__
		struct {
			uint16		ndigits;	/* number of digits */
			uint16		weight;		/* weight of first digit */
			uint16		sign;		/* NUMERIC_(POS|NEG|NAN) */
			uint16		dscale;		/* display scale */
			NumericDigit digits[FLEXIBLE_ARRAY_MEMBER];
		}  *rawdata = (void *)addr;
		nv.ndigits	= (int16)__ntoh16(rawdata->ndigits);
		nv.weight	= (int16)__ntoh16(rawdata->weight);
		nv.sign		= (int16)__ntoh16(rawdata->sign);
		nv.dscale	= (int16)__ntoh16(rawdata->dscale);
		nv.digits	= rawdata->digits;
#else	/* __PG2ARROW__ */
		init_var_from_num(&nv, addr, sz);
#endif	/* __PG2ARROW__ */
		if ((nv.sign & NUMERIC_SIGN_MASK) == NUMERIC_NAN)
			Elog("Decimal128 cannot map NaN in PostgreSQL Numeric");

		/* makes integer portion first */
		for (d=0; d <= nv.weight; d++)
		{
			dig = (d < nv.ndigits) ? __ntoh16(nv.digits[d]) : 0;
			if (dig < 0 || dig >= NBASE)
				Elog("Numeric digit is out of range: %d", (int)dig);
			value = NBASE * value + (int128)dig;
		}
		/* makes floating point portion if any */
		while (scale > 0)
		{
			dig = (d >= 0 && d < nv.ndigits) ? __ntoh16(nv.digits[d]) : 0;
			if (dig < 0 || dig >= NBASE)
				Elog("Numeric digit is out of range: %d", (int)dig);

			if (scale >= DEC_DIGITS)
				value = NBASE * value + dig;
			else if (scale == 3)
				value = 1000L * value + dig / 10L;
			else if (scale == 2)
				value =  100L * value + dig / 100L;
			else if (scale == 1)
				value =   10L * value + dig / 1000L;
			else
				Elog("internal bug");
			scale -= DEC_DIGITS;
			d++;
		}
		/* is it a negative value? */
		if ((nv.sign & NUMERIC_NEG) != 0)
			value = -value;

		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &value, sizeof(value));
	}
}
#endif	/* PG_INT128_TYPE */

/*
 * Date
 */
static inline void
__put_date_value_generic(SQLfield *attr, const char *addr, int pgsql_sz,
						 int64 adjustment, int arrow_sz)
{
	size_t		row_index = attr->nitems++;
	uint64		value;

	if (!addr)
		put_inline_null_value(attr, row_index, arrow_sz);
	else
	{
		assert(pgsql_sz == sizeof(DateADT));
		sql_buffer_setbit(&attr->nullmap, row_index);
		value = __ntoh32(*((const DateADT *)addr));
		value += (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
		/*
		 * PostgreSQL native is ArrowDateUnit__Day.
		 * Compiler optimization will remove the if-block below by constant
		 * 'adjustment' argument.
		 */
		if (adjustment > 0)
			value *= adjustment;
		else if (adjustment < 0)
			value /= adjustment;
		sql_buffer_append(&attr->values, &value, arrow_sz);
	}
}

static void
__put_date_day_value(SQLfield *attr, const char *addr, int sz)
{
	__put_date_value_generic(attr, addr, sz, 0, sizeof(int32));
}

static void
__put_date_ms_value(SQLfield *attr, const char *addr, int sz)
{
	__put_date_value_generic(attr, addr, sz, 86400000L, sizeof(int64));
}

static void
put_date_value(SQLfield *attr, const char *addr, int sz)
{
	/* validation checks only first call */
	switch (attr->arrow_type.Date.unit)
	{
		case ArrowDateUnit__Day:
			attr->put_value = __put_date_day_value;
			break;
		case ArrowDateUnit__MilliSecond:
			attr->put_value = __put_date_ms_value;
			break;
		default:
			Elog("ArrowTypeDate has unknown unit (%d)",
				 attr->arrow_type.Date.unit);
			break;
	}
	attr->put_value(attr, addr, sz);
}

/*
 * Time
 */
static inline void
__put_time_value_generic(SQLfield *attr, const char *addr, int pgsql_sz,
						 int64 adjustment, int arrow_sz)
{
	size_t		row_index = attr->nitems++;
	TimeADT		value;

	if (!addr)
		put_inline_null_value(attr, row_index, arrow_sz);
	else
	{
		assert(pgsql_sz == sizeof(TimeADT));
		value = __ntoh64(*((const TimeADT *)addr));
		/*
		 * PostgreSQL native is ArrowTimeUnit__MicroSecond
		 * Compiler optimization will remove the if-block below by constant
		 * 'adjustment' argument.
		 */
		if (adjustment > 0)
			value *= adjustment;
		else if (adjustment < 0)
			value /= -adjustment;
		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &value, arrow_sz);
	}
}

static void
__put_time_sec_value(SQLfield *attr, const char *addr, int sz)
{
	__put_time_value_generic(attr, addr, sz, -1000000L, sizeof(int32));
}

static void
__put_time_ms_value(SQLfield *attr, const char *addr, int sz)
{
	__put_time_value_generic(attr, addr, sz, -1000L, sizeof(int32));
}

static void
__put_time_us_value(SQLfield *attr, const char *addr, int sz)
{
	__put_time_value_generic(attr, addr, sz, 0L, sizeof(int64));
}

static void
__put_time_ns_value(SQLfield *attr, const char *addr, int sz)
{
	__put_time_value_generic(attr, addr, sz, 1000L, sizeof(int64));
}

static void
put_time_value(SQLfield *attr, const char *addr, int sz)
{
	switch (attr->arrow_type.Time.unit)
	{
		case ArrowTimeUnit__Second:
			if (attr->arrow_type.Time.bitWidth != 32)
				Elog("ArrowTypeTime has inconsistent bitWidth(%d) for [sec]",
					 attr->arrow_type.Time.bitWidth);
			attr->put_value = __put_time_sec_value;
			break;
		case ArrowTimeUnit__MilliSecond:
			if (attr->arrow_type.Time.bitWidth != 32)
				Elog("ArrowTypeTime has inconsistent bitWidth(%d) for [ms]",
					 attr->arrow_type.Time.bitWidth);
			attr->put_value = __put_time_ms_value;
			break;
		case ArrowTimeUnit__MicroSecond:
			if (attr->arrow_type.Time.bitWidth != 64)
				Elog("ArrowTypeTime has inconsistent bitWidth(%d) for [us]",
					 attr->arrow_type.Time.bitWidth);
			attr->put_value = __put_time_us_value;
			break;
		case ArrowTimeUnit__NanoSecond:
			if (attr->arrow_type.Time.bitWidth != 64)
				Elog("ArrowTypeTime has inconsistent bitWidth(%d) for [ns]",
					 attr->arrow_type.Time.bitWidth);
			attr->put_value = __put_time_ns_value;
		default:
			Elog("ArrowTypeTime has unknown unit (%d)",
				 attr->arrow_type.Time.unit);
			break;
	}
	attr->put_value(attr, addr, sz);
}

/*
 * Timestamp
 */
static inline void
__put_timestamp_value_generic(SQLfield *attr,
							  const char *addr, int pgsql_sz,
							  int64 adjustment, int arrow_sz)
{
	size_t		row_index = attr->nitems++;
	Timestamp	value;

	if (!addr)
		put_inline_null_value(attr, row_index, arrow_sz);
	else
	{
		assert(pgsql_sz == sizeof(Timestamp));
		value = __ntoh64(*((const Timestamp *)addr));
		/* convert PostgreSQL epoch to UNIX epoch */
		value += (POSTGRES_EPOCH_JDATE -
				  UNIX_EPOCH_JDATE) * USECS_PER_DAY;
		/*
		 * PostgreSQL native is ArrowTimeUnit__MicroSecond
		 * Compiler optimization will remove the if-block below by constant
		 * 'adjustment' argument.
		 */
		if (adjustment > 0)
			value *= adjustment;
		else if (adjustment < 0)
			value /= adjustment;
		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &value, arrow_sz);
	}
}

static void
__put_timestamp_sec_value(SQLfield *attr, const char *addr, int sz)
{
	__put_timestamp_value_generic(attr, addr, sz, -1000000L, sizeof(int64));
}

static void
__put_timestamp_ms_value(SQLfield *attr, const char *addr, int sz)
{
	__put_timestamp_value_generic(attr, addr, sz, -1000L, sizeof(int64));
}

static void
__put_timestamp_us_value(SQLfield *attr, const char *addr, int sz)
{
	__put_timestamp_value_generic(attr, addr, sz, 0L, sizeof(int64));
}

static void
__put_timestamp_ns_value(SQLfield *attr, const char *addr, int sz)
{
	__put_timestamp_value_generic(attr, addr, sz, -1000L, sizeof(int64));
}

static void
put_timestamp_value(SQLfield *attr, const char *addr, int sz)
{
	switch (attr->arrow_type.Timestamp.unit)
	{
		case ArrowTimeUnit__Second:
			attr->put_value = __put_timestamp_sec_value;
			break;
		case ArrowTimeUnit__MilliSecond:
			attr->put_value = __put_timestamp_ms_value;
			break;
		case ArrowTimeUnit__MicroSecond:
			attr->put_value = __put_timestamp_us_value;
			break;
		case ArrowTimeUnit__NanoSecond:
			attr->put_value = __put_timestamp_ns_value;
			break;
		default:
			Elog("ArrowTypeTimestamp has unknown unit (%d)",
				attr->arrow_type.Timestamp.unit);
			break;
	}
	attr->put_value(attr, addr, sz);
}

/*
 * Interval
 */
#define DAYS_PER_MONTH	30		/* assumes exactly 30 days per month */
#define HOURS_PER_DAY	24		/* assume no daylight savings time changes */

static void
__put_interval_year_month_value(SQLfield *attr, const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;

	if (!addr)
		put_inline_null_value(attr, row_index, sizeof(uint32));
	else
	{
		uint32	m;

		assert(sz == sizeof(Interval));
		m = __ntoh32(((const Interval *)addr)->month);
		sql_buffer_append(&attr->values, &m, sizeof(uint32));
	}
}

static void
__put_interval_day_time_value(SQLfield *attr, const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;

	if (!addr)
		put_inline_null_value(attr, row_index, 2 * sizeof(uint32));
	else
	{
		Interval	iv;
		uint32		value;

		assert(sz == sizeof(Interval));
		iv.time  = __ntoh64(((const Interval *)addr)->time);
		iv.day   = __ntoh32(((const Interval *)addr)->day);
		iv.month = __ntoh32(((const Interval *)addr)->month);

		/*
		 * Unit of PostgreSQL Interval is micro-seconds. Arrow Interval::time
		 * is represented as a pair of elapsed days and milli-seconds; needs
		 * to be adjusted.
		 */
		value = iv.month + DAYS_PER_MONTH * iv.day;
		sql_buffer_append(&attr->values, &value, sizeof(uint32));
		value = iv.time / 1000;
		sql_buffer_append(&attr->values, &value, sizeof(uint32));
	}
}

static void
put_interval_value(SQLfield *attr, const char *addr, int sz)
{
	switch (attr->arrow_type.Interval.unit)
	{
		case ArrowIntervalUnit__Year_Month:
			attr->put_value = __put_interval_year_month_value;
			break;
		case ArrowIntervalUnit__Day_Time:
			attr->put_value = __put_interval_day_time_value;
			break;
		default:
			Elog("attribute \"%s\" has unknown Arrow::Interval.unit(%d)",
				 attr->attname, attr->arrow_type.Interval.unit);
			break;
	}
	attr->put_value(attr, addr, sz);
}

/*
 * Utf8, Binary
 */
static void
put_variable_value(SQLfield *attr,
				   const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;

	if (row_index == 0)
		sql_buffer_append_zero(&attr->values, sizeof(uint32));
	if (!addr)
	{
		attr->nullcount++;
		sql_buffer_clrbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &attr->extra.usage, sizeof(uint32));
	}
	else
	{
		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->extra, addr, sz);
		sql_buffer_append(&attr->values, &attr->extra.usage, sizeof(uint32));
	}
}

/*
 * FixedSizeBinary
 */
static void
put_bpchar_value(SQLfield *attr,
				 const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;
	int			len = attr->atttypmod - VARHDRSZ;
	char	   *temp = alloca(len);

	assert(len > 0);
	memset(temp, ' ', len);
	if (!addr)
	{
		attr->nullcount++;
		sql_buffer_clrbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, temp, len);
	}
	else
	{
		memcpy(temp, addr, Min(sz, len));
		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, temp, len);
	}
}

/*
 * List::<element> type
 */
static void
put_array_value(SQLfield *attr,
				const char *addr, int sz)
{
	SQLfield *element = attr->element;
	size_t		row_index = attr->nitems++;

	if (row_index == 0)
		sql_buffer_append_zero(&attr->values, sizeof(uint32));
	if (!addr)
	{
		attr->nullcount++;
		sql_buffer_clrbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &element->nitems, sizeof(int32));
	}
	else
	{
#ifdef __PG2ARROW__
		struct {
			int32		ndim;
			int32		hasnull;
			int32		element_type;
			struct {
				int32	sz;
				int32	lb;
			} dim[FLEXIBLE_ARRAY_MEMBER];
		}  *rawdata = (void *) addr;
		int32		ndim = __ntoh32(rawdata->ndim);
		//int32		hasnull = __ntoh32(rawdata->hasnull);
		Oid			element_type = __ntoh32(rawdata->element_type);
		size_t		i, nitems = 1;
		int			item_sz;
		char	   *pos;

		if (element_type != element->atttypid)
			Elog("PostgreSQL array type mismatch");
		if (ndim < 1)
			Elog("Invalid dimension size of PostgreSQL Array (ndim=%d)", ndim);
		for (i=0; i < ndim; i++)
			nitems *= __ntoh32(rawdata->dim[i].sz);

		pos = (char *)&rawdata->dim[ndim];
		for (i=0; i < nitems; i++)
		{
			if (pos + sizeof(int32) > addr + sz)
				Elog("out of range - binary array has corruption");
			item_sz = __ntoh32(*((int32 *)pos));
			pos += sizeof(int32);
			if (item_sz < 0)
				element->put_value(element, NULL, 0);
			else
			{
				element->put_value(element, pos, item_sz);
				pos += item_sz;
			}
		}
#else	/* __PG2ARROW__ */
		Elog("Bug? server code must override put_array_value");
#endif	/* !__PG2ARROW__ */
		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &element->nitems, sizeof(int32));
	}
}

/*
 * Arrow::Struct
 */
static void
put_composite_value(SQLfield *attr,
					const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;
	int			j;

	if (!addr)
	{
		attr->nullcount++;
		sql_buffer_clrbit(&attr->nullmap, row_index);
		/* NULL for all the subtypes */
		for (j=0; j < attr->nfields; j++)
		{
			SQLfield *subattr = &attr->subfields[j];
			subattr->put_value(subattr, NULL, 0);
		}
	}
	else
	{
#ifdef __PG2ARROW__
		const char *pos = addr;
		int			j, nvalids;

		if (sz < sizeof(uint32))
			Elog("binary composite record corruption");
		nvalids = __ntoh32(*((const int *)pos));
		pos += sizeof(int);
		for (j=0; j < attr->nfields; j++)
		{
			SQLfield *subattr = &attr->subfields[j];
			Oid		atttypid;
			int		attlen;

			if (j >= nvalids)
			{
				subattr->put_value(subattr, NULL, 0);
				continue;
			}
			if ((pos - addr) + sizeof(Oid) + sizeof(int) > sz)
				Elog("binary composite record corruption");
			atttypid = __ntoh32(*((Oid *)pos));
			pos += sizeof(Oid);
			if (subattr->atttypid != atttypid)
				Elog("composite subtype mismatch");
			attlen = __ntoh32(*((int *)pos));
			pos += sizeof(int);
			if (attlen == -1)
			{
				subattr->put_value(subattr, NULL, 0);
			}
			else
			{
				if ((pos - addr) + attlen > sz)
					Elog("binary composite record corruption");
				subattr->put_value(subattr, pos, attlen);
				pos += attlen;
			}
			assert(attr->nitems == subattr->nitems);
		}
#else	/* __PG2ARROW__ */
		Elog("Bug? server code must override put_composite_value");
#endif	/* !__PG2ARROW__ */
		sql_buffer_setbit(&attr->nullmap, row_index);
	}
}

static void
put_dictionary_value(SQLfield *attr,
					 const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;

	if (!addr)
	{
		attr->nullcount++;
		sql_buffer_clrbit(&attr->nullmap, row_index);
		sql_buffer_append_zero(&attr->values, sizeof(uint32));
	}
	else
	{
		SQLdictionary *enumdict = attr->enumdict;
		hashItem   *hitem;
		uint32		hash;

		hash = hash_any((const unsigned char *)addr, sz);
		for (hitem = enumdict->hslots[hash % enumdict->nslots];
			 hitem != NULL;
			 hitem = hitem->next)
		{
			if (hitem->hash == hash &&
				hitem->label_len == sz &&
				memcmp(hitem->label, addr, sz) == 0)
				break;
		}
		if (!hitem)
			Elog("Enum label was not found in pg_enum result");

		sql_buffer_setbit(&attr->nullmap, row_index);
        sql_buffer_append(&attr->values,  &hitem->index, sizeof(int32));
	}
}

/*
 * Rewind the Arrow Type Buffer
 */
void
rewindArrowTypeBuffer(SQLfield *attr, size_t nitems)
{
	if (nitems > attr->nitems)
		Elog("Bug? tried to rewind the buffer beyond the tail");
	else if (nitems == attr->nitems)
	{
		/* special case, nothing to do */
		return;
	}
	else if (nitems == 0)
	{
		/* special case optimization */
		sql_buffer_clear(&attr->nullmap);
		sql_buffer_clear(&attr->values);
		sql_buffer_clear(&attr->extra);
		attr->nitems = 0;
		attr->nullcount = 0;
		return;
	}
	else if (attr->nullcount > 0)
	{
		long		nullcount = 0;
		uint32	   *nullmap = (uint32 *)attr->nullmap.data;
		uint32		i, n = nitems / 32;
		uint32		mask = (1UL << (nitems % 32)) - 1;

		for (i=0; i < n; i++)
			nullcount += __builtin_popcount(~nullmap[i]);
		if (mask != 0)
			nullcount += __builtin_popcount(~nullmap[n] & mask);
		attr->nullcount = nullcount;
	}
	attr->nitems = nitems;
	attr->nullmap.usage = (nitems + 7) >> 3;

	switch (attr->arrow_type.node.tag)
	{
		case ArrowNodeTag__Int:
			switch (attr->arrow_type.Int.bitWidth)
			{
				case 8:
					attr->values.usage = sizeof(int8) * nitems;
					break;
				case 16:
					attr->values.usage = sizeof(int16) * nitems;
					break;
				case 32:
					attr->values.usage = sizeof(int32) * nitems;
					break;
				case 64:
					attr->values.usage = sizeof(int64) * nitems;
					break;
				default:
					Elog("unknown width of Arrow::Int type: %d",
						 attr->arrow_type.Int.bitWidth);
					break;
			}
			break;
		case ArrowNodeTag__FloatingPoint:
			switch (attr->arrow_type.FloatingPoint.precision)
			{
				case ArrowPrecision__Half:
					attr->values.usage = sizeof(int16) * nitems;
					break;
				case ArrowPrecision__Single:
					attr->values.usage = sizeof(int32) * nitems;
					break;
				case ArrowPrecision__Double:
					attr->values.usage = sizeof(int64) * nitems;
					break;
				default:
					Elog("unknown precision of Arrow::FloatingPoint type: %d",
						 attr->arrow_type.FloatingPoint.precision);
					break;
			}
			break;
		case ArrowNodeTag__Utf8:
		case ArrowNodeTag__Binary:
			attr->values.usage = sizeof(int32) * (nitems + 1);
			attr->extra.usage = ((uint32 *)attr->values.data)[nitems];
			break;
		case ArrowNodeTag__Bool:
			attr->values.usage = (nitems + 7) >> 3;
			break;
		case ArrowNodeTag__Decimal:
			attr->values.usage = sizeof(int128) * nitems;
			break;
		case ArrowNodeTag__Date:
			switch (attr->arrow_type.Date.unit)
			{
				case ArrowDateUnit__Day:
					attr->values.usage = sizeof(int32) * nitems;
					break;
				case ArrowDateUnit__MilliSecond:
					attr->values.usage = sizeof(int64) * nitems;
					break;
				default:
					Elog("unknown unit of Arrow::Date type; %u",
						 attr->arrow_type.Date.unit);
					break;
			}
			break;
		case ArrowNodeTag__Time:
			switch (attr->arrow_type.Time.unit)
			{
				case ArrowTimeUnit__Second:
				case ArrowTimeUnit__MilliSecond:
					attr->values.usage = sizeof(int32) * nitems;
					break;
				case ArrowTimeUnit__MicroSecond:
				case ArrowTimeUnit__NanoSecond:
					attr->values.usage = sizeof(int64) * nitems;
					break;
				default:
					Elog("unknown unit of Arrow::Time type; %u",
						 attr->arrow_type.Time.unit);
			}
			break;
		case ArrowNodeTag__Timestamp:
			attr->values.usage = sizeof(int64) * nitems;
			break;
		case ArrowNodeTag__Interval:
			switch (attr->arrow_type.Interval.unit)
			{
				case ArrowIntervalUnit__Year_Month:
					attr->values.usage = sizeof(int32) * nitems;
					break;
				case ArrowIntervalUnit__Day_Time:
					attr->values.usage = sizeof(int64) * nitems;
					break;
				default:
					Elog("unknown unit of Arrow::Interval type; %u",
						 attr->arrow_type.Interval.unit);
			}
			break;
		case ArrowNodeTag__List:
			attr->values.usage = sizeof(int32) * (nitems + 1);
			rewindArrowTypeBuffer(attr->element, nitems);
			break;
		case ArrowNodeTag__Struct:
			{
				for (int j=0; j < attr->nfields; j++)
					rewindArrowTypeBuffer(&attr->subfields[j], nitems);

			}
			break;
		case ArrowNodeTag__FixedSizeBinary:
			{
				int32	unitsz = attr->arrow_type.FixedSizeBinary.byteWidth;

				attr->values.usage = unitsz * nitems;
			}
			break;
		default:
			Elog("unexpected ArrowType node tag: %s (%u)",
				 attr->arrow_type.node.tagName,
				 attr->arrow_type.node.tag);
	}
}

/* ----------------------------------------------------------------
 *
 * buffer_usage handler for each data types
 *
 * ---------------------------------------------------------------- */
static size_t
buffer_usage_inline_type(SQLfield *attr)
{
	size_t		usage;

	usage = ARROWALIGN(attr->values.usage);
	if (attr->nullcount > 0)
		usage += ARROWALIGN(attr->nullmap.usage);
	return usage;
}

static size_t
buffer_usage_varlena_type(SQLfield *attr)
{
	size_t		usage;

	usage = (ARROWALIGN(attr->values.usage) +
			 ARROWALIGN(attr->extra.usage));
	if (attr->nullcount > 0)
		usage += ARROWALIGN(attr->nullmap.usage);
	return usage;
}

static size_t
buffer_usage_array_type(SQLfield *attr)
{
	SQLfield   *element = attr->element;
	size_t			usage;

	usage = ARROWALIGN(attr->values.usage);
	if (attr->nullcount > 0)
		usage += ARROWALIGN(attr->nullmap.usage);
	usage += element->buffer_usage(element);

	return usage;
}

static size_t
buffer_usage_composite_type(SQLfield *attr)
{
	size_t		usage = 0;
	int			j;

	if (attr->nullcount > 0)
		usage += ARROWALIGN(attr->nullmap.usage);
	for (j=0; j < attr->nfields; j++)
	{
		SQLfield *subattr = &attr->subfields[j];
		usage += subattr->buffer_usage(subattr);
	}
	return usage;
}

/*
 * missing function in PG9.6
 */
#if PG_VERSION_NUM < 100000
Datum Float4GetDatum(float4 X)
{
	union
	{
		float4		value;
		int32		retval;
	}	myunion;
	myunion.value = X;
	return Int32GetDatum(myunion.retval);
}

Datum Float8GetDatum(float8 X)
{
	union
	{
		float8		value;
		int64		retval;
	}	myunion;
	myunion.value = X;
	return Int64GetDatum(myunion.retval);
}
#endif

/* ----------------------------------------------------------------
 *
 * setup_buffer handler for each data types
 *
 * ----------------------------------------------------------------
 */
static inline size_t
setup_arrow_buffer(ArrowBuffer *node, size_t offset, size_t length)
{
	memset(node, 0, sizeof(ArrowBuffer));
	INIT_ARROW_NODE(node, Buffer);
	node->offset = offset;
	node->length = ARROWALIGN(length);

	return node->length;
}

static int
setup_buffer_inline_type(SQLfield *attr,
						 ArrowBuffer *node, size_t *p_offset)
{
	size_t		offset = *p_offset;

	/* nullmap */
	if (attr->nullcount == 0)
		offset += setup_arrow_buffer(node, offset, 0);
	else
		offset += setup_arrow_buffer(node, offset, attr->nullmap.usage);
	/* inline values */
	offset += setup_arrow_buffer(node+1, offset, attr->values.usage);

	*p_offset = offset;
	return 2;	/* nullmap + values */
}

static int
setup_buffer_varlena_type(SQLfield *attr,
						  ArrowBuffer *node, size_t *p_offset)
{
	size_t		offset = *p_offset;

	/* nullmap */
	if (attr->nullcount == 0)
		offset += setup_arrow_buffer(node, offset, 0);
	else
		offset += setup_arrow_buffer(node, offset, attr->nullmap.usage);
	/* index values */
	offset += setup_arrow_buffer(node+1, offset, attr->values.usage);
	/* extra buffer */
	offset += setup_arrow_buffer(node+2, offset, attr->extra.usage);

	*p_offset = offset;
	return 3;	/* nullmap + values (index) + extra buffer */
}

static int
setup_buffer_array_type(SQLfield *attr,
						ArrowBuffer *node, size_t *p_offset)
{
	SQLfield *element = attr->element;
	int			count = 2;

	/* nullmap */
	if (attr->nullcount == 0)
		*p_offset += setup_arrow_buffer(node, *p_offset, 0);
	else
		*p_offset += setup_arrow_buffer(node, *p_offset,
										attr->nullmap.usage);
	/* index values */
	*p_offset += setup_arrow_buffer(node+1, *p_offset,
									attr->values.usage);
	/* element values */
	count += element->setup_buffer(element, node+2, p_offset);

	return count;
}

static int
setup_buffer_composite_type(SQLfield *attr,
							ArrowBuffer *node, size_t *p_offset)
{
	int			j, count = 1;

	/* nullmap */
	if (attr->nullcount == 0)
		*p_offset += setup_arrow_buffer(node, *p_offset, 0);
	else
		*p_offset += setup_arrow_buffer(node, *p_offset,
										attr->nullmap.usage);
	/* walk down the sub-types */
	for (j=0; j < attr->nfields; j++)
	{
		SQLfield   *subattr = &attr->subfields[j];

		count += subattr->setup_buffer(subattr, node+count, p_offset);
	}
	return count;	/* nullmap + subtypes */
}

/* ----------------------------------------------------------------
 *
 * write buffer handler for each data types
 *
 * ----------------------------------------------------------------
 */
static void
write_buffer_inline_type(SQLfield *attr, int fdesc)
{
	/* nullmap */
	if (attr->nullcount > 0)
		sql_buffer_write(fdesc, &attr->nullmap);
	/* fixed length values */
	sql_buffer_write(fdesc, &attr->values);
}

static void
write_buffer_varlena_type(SQLfield *attr, int fdesc)
{
	/* nullmap */
	if (attr->nullcount > 0)
		sql_buffer_write(fdesc, &attr->nullmap);
	/* index values */
	sql_buffer_write(fdesc, &attr->values);
	/* extra buffer */
	sql_buffer_write(fdesc, &attr->extra);
}

static void
write_buffer_array_type(SQLfield *attr, int fdesc)
{
	SQLfield *element = attr->element;

	/* nullmap */
	if (attr->nullcount > 0)
		sql_buffer_write(fdesc, &attr->nullmap);
	/* offset values */
	sql_buffer_write(fdesc, &attr->values);
	/* element values */
	element->write_buffer(element, fdesc);
}

static void
write_buffer_composite_type(SQLfield *attr, int fdesc)
{
	int			j;

	/* nullmap */
	if (attr->nullcount > 0)
		sql_buffer_write(fdesc, &attr->nullmap);
	/* sub-types */
	for (j=0; j < attr->nfields; j++)
	{
		SQLfield   *subattr = &attr->subfields[j];

		subattr->write_buffer(subattr, fdesc);
	}
}

/* ----------------------------------------------------------------
 *
 * setup handler for each data types
 *
 * ----------------------------------------------------------------
 */
static int
assignArrowTypeInt(SQLfield *attr, bool is_signed)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Int);
	attr->arrow_type.Int.is_signed = is_signed;
	switch (attr->attlen)
	{
		case sizeof(char):
			attr->arrow_type.Int.bitWidth = 8;
			attr->arrow_typename = (is_signed ? "Int8" : "Uint8");
			attr->put_value = put_int8_value;
			break;
		case sizeof(short):
			attr->arrow_type.Int.bitWidth = 16;
			attr->arrow_typename = (is_signed ? "Int16" : "Uint16");
			attr->put_value = put_int16_value;
			break;
		case sizeof(int):
			attr->arrow_type.Int.bitWidth = 32;
			attr->arrow_typename = (is_signed ? "Int32" : "Uint32");
			attr->put_value = put_int32_value;
			break;
		case sizeof(long):
			attr->arrow_type.Int.bitWidth = 64;
			attr->arrow_typename = (is_signed ? "Int64" : "Uint64");
			attr->put_value = put_int64_value;
			break;
		default:
			Elog("unsupported Int width: %d", attr->attlen);
			break;
	}
	attr->buffer_usage = buffer_usage_inline_type;
	attr->setup_buffer = setup_buffer_inline_type;
	attr->write_buffer = write_buffer_inline_type;

	return 2;		/* null map + values */
}

static int
assignArrowTypeFloatingPoint(SQLfield *attr)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, FloatingPoint);
	switch (attr->attlen)
	{
		case sizeof(short):		/* half */
			attr->arrow_type.FloatingPoint.precision = ArrowPrecision__Half;
			attr->arrow_typename = "Float16";
			attr->put_value = put_float16_value;
			break;
		case sizeof(float):
			attr->arrow_type.FloatingPoint.precision = ArrowPrecision__Single;
			attr->arrow_typename = "Float32";
			attr->put_value = put_float32_value;
			break;
		case sizeof(double):
			attr->arrow_type.FloatingPoint.precision = ArrowPrecision__Double;
			attr->arrow_typename = "Float64";
			attr->put_value = put_float64_value;
			break;
		default:
			Elog("unsupported floating point width: %d", attr->attlen);
			break;
	}
	attr->buffer_usage = buffer_usage_inline_type;
	attr->setup_buffer = setup_buffer_inline_type;
	attr->write_buffer = write_buffer_inline_type;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeBinary(SQLfield *attr)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Binary);
	attr->arrow_typename	= "Binary";
	attr->put_value			= put_variable_value;
	attr->buffer_usage		= buffer_usage_varlena_type;
	attr->setup_buffer		= setup_buffer_varlena_type;
	attr->write_buffer		= write_buffer_varlena_type;

	return 3;		/* nullmap + index + extra */
}

static int
assignArrowTypeUtf8(SQLfield *attr)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Utf8);
	attr->arrow_typename	= "Utf8";
	attr->put_value			= put_variable_value;
	attr->buffer_usage		= buffer_usage_varlena_type;
	attr->setup_buffer		= setup_buffer_varlena_type;
	attr->write_buffer		= write_buffer_varlena_type;

	return 3;		/* nullmap + index + extra */
}

static int
assignArrowTypeBpchar(SQLfield *attr)
{
	if (attr->atttypmod <= VARHDRSZ)
		Elog("unexpected Bpchar definition (typmod=%d)", attr->atttypmod);

	INIT_ARROW_TYPE_NODE(&attr->arrow_type, FixedSizeBinary);
	attr->arrow_type.FixedSizeBinary.byteWidth = attr->atttypmod - VARHDRSZ;
	attr->arrow_typename	= "FixedSizeBinary";
	attr->put_value			= put_bpchar_value;
	attr->buffer_usage		= buffer_usage_inline_type;
	attr->setup_buffer		= setup_buffer_inline_type;
	attr->write_buffer		= write_buffer_inline_type;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeBool(SQLfield *attr)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Bool);
	attr->arrow_typename	= "Bool";
	attr->put_value			= put_bool_value;
	attr->buffer_usage		= buffer_usage_inline_type;
	attr->setup_buffer		= setup_buffer_inline_type;
	attr->write_buffer		= write_buffer_inline_type;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeDecimal(SQLfield *attr)
{
#ifdef PG_INT128_TYPE
	int		typmod			= attr->atttypmod;
	int		precision		= 30;	/* default, if typmod == -1 */
	int		scale			=  8;	/* default, if typmod == -1 */

	if (typmod >= VARHDRSZ)
	{
		typmod -= VARHDRSZ;
		precision = (typmod >> 16) & 0xffff;
		scale = (typmod & 0xffff);
	}
	memset(&attr->arrow_type, 0, sizeof(ArrowType));
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Decimal);
	attr->arrow_type.Decimal.precision = precision;
	attr->arrow_type.Decimal.scale = scale;
	attr->arrow_typename	= "Decimal";
	attr->put_value			= put_decimal_value;
	attr->buffer_usage		= buffer_usage_inline_type;
	attr->setup_buffer		= setup_buffer_inline_type;
	attr->write_buffer		= write_buffer_inline_type;
#else
	/*
	 * MEMO: Numeric of PostgreSQL is mapped to Decimal128 in Apache Arrow.
	 * Due to implementation reason, we require int128 support by compiler.
	 */
	Elog("Numeric type of PostgreSQL is not supported in this build");
#endif
	return 2;		/* nullmap + values */
}

static int
assignArrowTypeDate(SQLfield *attr)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Date);
	attr->arrow_type.Date.unit = ArrowDateUnit__Day;
	attr->arrow_typename	= "Date";
	attr->put_value			= put_date_value;
	attr->buffer_usage		= buffer_usage_inline_type;
	attr->setup_buffer		= setup_buffer_inline_type;
	attr->write_buffer		= write_buffer_inline_type;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeTime(SQLfield *attr)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Time);
	attr->arrow_type.Time.unit = ArrowTimeUnit__MicroSecond;
	attr->arrow_type.Time.bitWidth = 64;
	attr->arrow_typename	= "Time";
	attr->put_value			= put_time_value;
	attr->buffer_usage		= buffer_usage_inline_type;
	attr->setup_buffer		= setup_buffer_inline_type;
	attr->write_buffer		= write_buffer_inline_type;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeTimestamp(SQLfield *attr)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Timestamp);
	attr->arrow_type.Timestamp.unit = ArrowTimeUnit__MicroSecond;
	attr->arrow_typename	= "Timestamp";
	attr->put_value			= put_timestamp_value;
	attr->buffer_usage		= buffer_usage_inline_type;
	attr->setup_buffer		= setup_buffer_inline_type;
	attr->write_buffer		= write_buffer_inline_type;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeInterval(SQLfield *attr)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Interval);
	attr->arrow_type.Interval.unit = ArrowIntervalUnit__Day_Time;
	attr->arrow_typename	= "Interval";
	attr->put_value         = put_interval_value;
	attr->buffer_usage		= buffer_usage_inline_type;
	attr->setup_buffer		= setup_buffer_inline_type;
	attr->write_buffer		= write_buffer_inline_type;

	return 2;		/* nullmap + values */
}

static int
assignArrowTypeList(SQLfield *attr)
{
	SQLfield *element = attr->element;

	INIT_ARROW_TYPE_NODE(&attr->arrow_type, List);
	attr->put_value			= put_array_value;
	attr->arrow_typename	= psprintf("List<%s>", element->arrow_typename);
	attr->buffer_usage		= buffer_usage_array_type;
	attr->setup_buffer		= setup_buffer_array_type;
	attr->write_buffer		= write_buffer_array_type;

	return 2;		/* nullmap + offset vector */
}

static int
assignArrowTypeStruct(SQLfield *attr)
{
	assert(attr->subfields != NULL);
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Struct);
	attr->arrow_typename	= "Struct";
	attr->put_value			= put_composite_value;
	attr->buffer_usage		= buffer_usage_composite_type;
	attr->setup_buffer		= setup_buffer_composite_type;
	attr->write_buffer		= write_buffer_composite_type;

	return 1;	/* only nullmap */
}

static int
assignArrowTypeDictionary(SQLfield *attr)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Utf8);
	attr->arrow_typename	= psprintf("Enum; dictionary=%u", attr->atttypid);
	attr->put_value			= put_dictionary_value;
	attr->buffer_usage		= buffer_usage_inline_type;
	attr->setup_buffer		= setup_buffer_inline_type;
	attr->write_buffer		= write_buffer_inline_type;

	return 2;	/* nullmap + values */
}

/*
 * assignArrowType
 */
int
assignArrowType(SQLfield *attr)
{
	memset(&attr->arrow_type, 0, sizeof(ArrowType));
	if (attr->subfields)
	{
		/* composite type */
		return assignArrowTypeStruct(attr);
	}
	else if (attr->element)
	{
		/* array type */
		return assignArrowTypeList(attr);
	}
	else if (attr->typtype == 'e')
	{
		/* enum type */
		return assignArrowTypeDictionary(attr);
	}
	else if (strcmp(attr->typnamespace, "pg_catalog") == 0)
	{
		/* well known built-in data types? */
		if (strcmp(attr->typname, "bool") == 0)
		{
			return assignArrowTypeBool(attr);
		}
		else if (strcmp(attr->typname, "int2") == 0 ||
				 strcmp(attr->typname, "int4") == 0 ||
				 strcmp(attr->typname, "int8") == 0)
		{
			return assignArrowTypeInt(attr, true);
		}
		else if (strcmp(attr->typname, "float2") == 0 ||	/* by PG-Strom */
				 strcmp(attr->typname, "float4") == 0 ||
				 strcmp(attr->typname, "float8") == 0)
		{
			return assignArrowTypeFloatingPoint(attr);
		}
		else if (strcmp(attr->typname, "date") == 0)
		{
			return assignArrowTypeDate(attr);
		}
		else if (strcmp(attr->typname, "time") == 0)
		{
			return assignArrowTypeTime(attr);
		}
		else if (strcmp(attr->typname, "timestamp") == 0 ||
				 strcmp(attr->typname, "timestamptz") == 0)
		{
			return assignArrowTypeTimestamp(attr);
		}
		else if (strcmp(attr->typname, "interval") == 0)
		{
			return assignArrowTypeInterval(attr);
		}
		else if (strcmp(attr->typname, "text") == 0 ||
				 strcmp(attr->typname, "varchar") == 0)
		{
			return assignArrowTypeUtf8(attr);
		}
		else if (strcmp(attr->typname, "bpchar") == 0)
		{
			return assignArrowTypeBpchar(attr);
		}
		else if (strcmp(attr->typname, "numeric") == 0)
		{
			return assignArrowTypeDecimal(attr);
		}
	}
	/* elsewhere, we save the column just a bunch of binary data */
	if (attr->attlen > 0)
	{
		if (attr->attlen == sizeof(char) ||
			attr->attlen == sizeof(short) ||
			attr->attlen == sizeof(int) ||
			attr->attlen == sizeof(long))
		{
			return assignArrowTypeInt(attr, false);
		}
		/*
		 * MEMO: Unfortunately, we have no portable way to pack user defined
		 * fixed-length binary data types, because their 'send' handler often
		 * manipulate its internal data representation.
		 * Please check box_send() for example. It sends four float8 (which
		 * is reordered to bit-endien) values in 32bytes. We cannot understand
		 * its binary format without proper knowledge.
		 */
	}
	else if (attr->attlen == -1)
	{
		return assignArrowTypeBinary(attr);
	}
	Elog("PostgreSQL type: '%s.%s' is not supported",
		 attr->typnamespace,
		 attr->typname);
}
