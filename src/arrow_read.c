/*
 * arrow_read.c - routines to parse apache arrow file
 *
 * Copyright 2018-2019 (C) KaiGai Kohei <kaigai@heterodb.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License. See the LICENSE file.
 */
#include "pg_strom.h"
#include "arrow_defs.h"

#ifdef PG_STROM_H
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
#define __dumpArrowTypeNull		__dumpArrowNodeSimple
#define __dumpArrowTypeUtf8		__dumpArrowNodeSimple
#define __dumpArrowTypeBinary	__dumpArrowNodeSimple
#define __dumpArrowTypeBool		__dumpArrowNodeSimple
#define __dumpArrowTypeList		__dumpArrowNodeSimple
#define __dumpArrowTypeStruct	__dumpArrowNodeSimple

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
		f->precision == ArrowPrecision__Half ? "16" :
		f->precision == ArrowPrecision__Single ? "32" :
		f->precision == ArrowPrecision__Double ? "64" : "??");
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
		d->unit == ArrowDateUnit__Day ? "day" :
		d->unit == ArrowDateUnit__MilliSecond ? "msec" : "???");
}

static void
__dumpArrowTypeTime(StringInfo str, ArrowNode *node)
{
	ArrowTypeTime *t = (ArrowTypeTime *)node;

	appendStringInfo(
		str,"{Time: unit=%s}",
		t->unit == ArrowTimeUnit__Second ? "sec" :
		t->unit == ArrowTimeUnit__MilliSecond ? "ms" :
		t->unit == ArrowTimeUnit__MicroSecond ? "us" :
		t->unit == ArrowTimeUnit__NanoSecond ? "ns" : "???");
}

static void
__dumpArrowTypeTimestamp(StringInfo str, ArrowNode *node)
{
	ArrowTypeTimestamp *t = (ArrowTypeTimestamp *)node;

	appendStringInfo(
		str,"{Timestamp: unit=%s}",
		t->unit == ArrowTimeUnit__Second ? "sec" :
		t->unit == ArrowTimeUnit__MilliSecond ? "ms" :
		t->unit == ArrowTimeUnit__MicroSecond ? "us" :
		t->unit == ArrowTimeUnit__NanoSecond ? "ns" : "???");
}

static void
__dumpArrowTypeInterval(StringInfo str, ArrowNode *node)
{
	ArrowTypeInterval *t = (ArrowTypeInterval *)node;

	appendStringInfo(
		str,"{Interval: unit=%s}",
		t->unit==ArrowIntervalUnit__Year_Month ? "Year/Month" :
		t->unit==ArrowIntervalUnit__Day_Time ? "Day/Time" : "???");
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
__dumpArrowKeyValue(StringInfo str, ArrowNode *node)
{
	ArrowKeyValue *kv = (ArrowKeyValue *)node;

	appendStringInfo(str,"{KeyValue: key=\"");
	if (kv->key)
		outToken(str, kv->key);
	appendStringInfo(str,"\", value=\"");
	if (kv->value)
		outToken(str, kv->value);
	appendStringInfo(str,"\"}");
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

	appendStringInfo(str, "{Field: name=\"");
	if (f->name)
		outToken(str, f->name);
	appendStringInfo(str, "\", nullable=%s, type=",
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
	appendStringInfo(str, ", recordBatches=[");
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
#endif

#ifdef PG_STROM_H
/*
 * Copy support of ArrowNode
 */
#define COPY_SCALAR(FIELD)								\
	(dest)->FIELD = (src)->FIELD
#define COPY_CSTRING(FIELD)								\
	do {												\
		(dest)->FIELD = pstrdup((src)->FIELD);			\
		(dest)->_##FIELD##_len = strlen((dest)->FIELD);	\
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
	COPY_SCALAR(nodeSz);
	COPY_SCALAR(dumpArrowNode);
	COPY_SCALAR(copyArrowNode);
}
#define __copyArrowTypeNull		__copyArrowNode
#define __copyArrowTypeUtf8		__copyArrowNode
#define __copyArrowTypeBinary	__copyArrowNode
#define __copyArrowTypeBool		__copyArrowNode
#define __copyArrowTypeList		__copyArrowNode
#define __copyArrowTypeStruct	__copyArrowNode

typedef void (*copyArrowNode_f)(ArrowNode *dest, const ArrowNode *src);

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
	COPY_SCALAR(indexType);
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
			elog(ERROR, "Bug? unknown ArrowMessageBody");
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
#endif

#ifdef PG_STROM_H
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
			{
				ArrowTypeFloatingPoint *f = &t->FloatingPoint;

				sz = snprintf(
					buf, len, "Float%s",
					f->precision == ArrowPrecision__Half ? "16" :
					f->precision == ArrowPrecision__Single ? "32" :
					f->precision == ArrowPrecision__Double ? "64" : "??");
			}
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
			{
				ArrowDateUnit	unit = t->Date.unit;

				sz = snprintf(
					buf, len, "Date[%s]",
					unit == ArrowDateUnit__Day ? "day" :
					unit == ArrowDateUnit__MilliSecond ? "msec" : "??");
			}
			break;
		case ArrowNodeTag__Time:
			{
				ArrowTimeUnit	unit = t->Time.unit;

				sz = snprintf(buf, len, "Time[%s]",
							  unit == ArrowTimeUnit__Second ? "sec" :
							  unit == ArrowTimeUnit__MilliSecond ? "ms" :
							  unit == ArrowTimeUnit__MicroSecond ? "us" :
							  unit == ArrowTimeUnit__NanoSecond ? "ns" : "??");
			}
			break;
		case ArrowNodeTag__Timestamp:
			{
				ArrowTimeUnit	unit = t->Timestamp.unit;

				sz = snprintf(buf, len, "Timestamp[%s]",
							  unit == ArrowTimeUnit__Second ? "sec" :
							  unit == ArrowTimeUnit__MilliSecond ? "ms" :
							  unit == ArrowTimeUnit__MicroSecond ? "us" :
							  unit == ArrowTimeUnit__NanoSecond ? "ns" : "??");
			}
			break;
		case ArrowNodeTag__Interval:
			switch (t->Interval.unit)
			{
				case ArrowIntervalUnit__Year_Month:
					sz = snprintf(buf, len, "Interval[Year/Month]");
					break;
				case ArrowIntervalUnit__Day_Time:
					sz = snprintf(buf, len, "Interval[Day/Time]");
					break;
				default:
					sz = snprintf(buf, len, "Interval[???]");
					break;
			}
			break;
		case ArrowNodeTag__List:
			if (field->_num_children != 1)
				elog(ERROR, "corrupted List data type");
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
		default:
			elog(ERROR, "unknown Arrow type");
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
#endif

/* ------------------------------------------------
 *
 * Routines to read Apache Arrow files
 *
 * ------------------------------------------------
 */
#ifdef PG_STROM_H
#define __INIT_ARROW_NODE(PTR,TYPENAME,NAME)					\
	do {														\
		((ArrowNode *)(PTR))->tag = ArrowNodeTag__##NAME;		\
		((ArrowNode *)(PTR))->tagName = #NAME;					\
		((ArrowNode *)(PTR))->nodeSz = sizeof(Arrow##TYPENAME);	\
		((ArrowNode *)(PTR))->dumpArrowNode = __dumpArrow##TYPENAME; \
		((ArrowNode *)(PTR))->copyArrowNode						\
			= (copyArrowNode_f)__copyArrow##TYPENAME;			\
	} while(0)
#else	/* PG_STROM_H */
#define __INIT_ARROW_NODE(PTR,TYPENAME,NAME)					\
	do {														\
		((ArrowNode *)(PTR))->tag = ArrowNodeTag__##NAME;		\
		((ArrowNode *)(PTR))->tagName = #NAME;					\
		((ArrowNode *)(PTR))->nodeSz = sizeof(Arrow##TYPENAME);	\
	} while(0)
#endif	/* PG_STROM_H */
#define INIT_ARROW_NODE(PTR,NAME)		__INIT_ARROW_NODE(PTR,NAME,NAME)
#define INIT_ARROW_TYPE_NODE(PTR,NAME)	__INIT_ARROW_NODE(PTR,Type##NAME,NAME)


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

typedef struct
{
	int32		metaLength;
	int32		headOffset;
} FBMetaData;

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
		int		offset = vtable->offset[index];
		if (offset)
		{
			void   *addr = (char *)t->table + offset;

			assert((char *)addr < (char *)t->table + vtable->tlen);
			return addr;
		}
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
		temp = pnstrdup((char *)ptr, len);
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
}

static void
readArrowTypeFloatingPoint(ArrowTypeFloatingPoint *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32 *) pos);

	node->precision = fetchShort(&t, 0);
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
}

static void
readArrowTypeTime(ArrowTypeTime *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32 *) pos);

	node->unit = fetchShort(&t, 0);
	node->bitWidth = fetchInt(&t, 1);
}

static void
readArrowTypeTimestamp(ArrowTypeTimestamp *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32 *) pos);

	node->unit = fetchInt(&t, 0);
	node->timezone = fetchString(&t, 1, &node->_timezone_len);
}

static void
readArrowTypeInterval(ArrowTypeInterval *node, const char *pos)
{
	FBTable		t = fetchFBTable((int32 *) pos);

	node->unit = fetchShort(&t, 0);
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
	dict->indexType.node.tag = ArrowNodeTag__Int;
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
				elog(ERROR, "ArrowField has NULL-element in children[]");
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
				elog(ERROR, "ArrowField has NULL-element in custom_metadata[]");
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
				elog(ERROR, "ArrowSchema has NULL-element in fields[]");
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
				elog(ERROR, "ArrowSchema has NULL-element in custom_metadata[]");
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
		elog(ERROR, "metadata version %d is not supported", message->version);

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
			elog(ERROR, "message type: Tensor is not implemented");
			break;
		case ArrowMessageHeader__SparseTensor:
			elog(ERROR, "message type: SparseTensor is not implemented");
			break;
		default:
			elog(ERROR, "unknown message header type: %d", mtype);
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
void
readArrowFileDesc(File filp, ArrowFileInfo *af_info)
{
	int				fdesc = FileGetRawDesc(filp);
	size_t			file_sz;
	size_t			mmap_sz;
	char		   *mmap_head = NULL;
	char		   *mmap_tail = NULL;
	const char	   *pos;
	FBMetaData	   *meta;
	cl_int			offset;
	cl_int			i, nitems;

	memset(af_info, 0, sizeof(ArrowFileInfo));
	if (fstat(fdesc, &af_info->stat_buf) != 0)
		elog(ERROR, "failed on fstat: %m");
	file_sz = af_info->stat_buf.st_size;
	mmap_sz = TYPEALIGN(sysconf(_SC_PAGESIZE), file_sz);
	mmap_head = mmap(NULL, mmap_sz, PROT_READ, MAP_SHARED, fdesc, 0);
	if (mmap_head == MAP_FAILED)
		elog(ERROR, "failed on mmap: %m");
	mmap_tail = mmap_head + file_sz;

	PG_TRY();
	{
		/* check signature */
		if (memcmp(mmap_head, "ARROW1\0\0", 8) != 0 ||
			memcmp(mmap_tail - 6, "ARROW1", 6) != 0)
			elog(ERROR, "Signature mismatch on Apache Arrow file");

		/* OK, read the Apache Arrow file */
		memset(af_info, 0, sizeof(ArrowFileInfo));
		/* Read Footer chunk */
		pos = mmap_tail - 6 - sizeof(int32);
		offset = *((int32 *)pos);
		pos -= offset;
		offset = *((int32 *)pos);
		readArrowFooter(&af_info->footer, pos + offset);

		/* Read DictionaryBatch chunks */
		nitems = af_info->footer._num_dictionaries;
		for (i=0; i < nitems; i++)
		{
			ArrowBlock	   *b = &af_info->footer.dictionaries[i];
			ArrowMessage   *m = palloc0(sizeof(ArrowMessage));

			meta = (FBMetaData *)(mmap_head + b->offset);
			if (b->metaDataLength != meta->metaLength + sizeof(int32))
				elog(ERROR, "metadata length mismatch");
			pos = (const char *)&meta->headOffset + meta->headOffset;
			readArrowMessage(m, pos);

			af_info->dictionaries = lappend(af_info->dictionaries,
											&m->body.dictionaryBatch);
		}

		/* Read RecordBatch chunks */
		nitems = af_info->footer._num_recordBatches;
		for (i=0; i < nitems; i++)
		{
			ArrowBlock	   *b = &af_info->footer.recordBatches[i];
			ArrowMessage   *m = palloc0(sizeof(ArrowMessage));

			meta = (FBMetaData *)(mmap_head + b->offset);
			if (b->metaDataLength != meta->metaLength + sizeof(int32))
				elog(ERROR, "metadata length mismatch");
			pos = (const char *)&meta->headOffset + meta->headOffset;
			readArrowMessage(m, pos);

			af_info->recordBatches = lappend(af_info->recordBatches,
											 &m->body.recordBatch);
		}
	}
	PG_CATCH();
	{
		if (munmap(mmap_head, mmap_sz) != 0)
			elog(LOG, "failed on munmap(2): %m");
		PG_RE_THROW();
	}
	PG_END_TRY();
	if (munmap(mmap_head, mmap_sz) != 0)
		elog(LOG, "failed on munmap(2): %m");
}

void
readArrowFile(char *pathname, ArrowFileInfo *af_info)
{
	File		filp = PathNameOpenFile(pathname, O_RDONLY | PG_BINARY);

	readArrowFileDesc(filp, af_info);

	FileClose(filp);
}
