/*
 * arrow_ipc.h
 *
 * Definitions for Apache Arrow IPC stuff.
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
#ifndef ARROW_IPC_H
#define ARROW_IPC_H
#include <assert.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

#include "arrow_defs.h"

#define	ARROWALIGN(LEN)		TYPEALIGN(64, (LEN))

typedef struct SQLbuffer		SQLbuffer;
typedef struct SQLtable			SQLtable;
typedef struct SQLfield			SQLfield;
typedef struct SQLdictionary	SQLdictionary;
typedef union  SQLtype			SQLtype;
typedef struct SQLtype__pgsql	SQLtype__pgsql;
typedef struct SQLtype__mysql	SQLtype__mysql;

struct SQLbuffer
{
	char	   *data;
	uint32		usage;
	uint32		length;
};

struct SQLtype__pgsql
{
	Oid			typeid;
	int			typmod;
	const char *typname;
	const char *typnamespace;
	short		typlen;
	bool		typbyval;
	char		typtype;
	uint8		typalign;
};

struct SQLtype__mysql
{
	const char *typname;
};

union SQLtype
{
	SQLtype__pgsql	pgsql;
	SQLtype__mysql	mysql;
};

struct SQLfield
{
	char	   *field_name;		/* name of the column, element or sub-field */
	SQLtype		sql_type;		/* attributes of SQL type */
	SQLfield   *element;		/* valid, if array type */
	int			nfields;		/* # of sub-fields of composite type */
	SQLfield   *subfields;	/* valid, if composite type */
	SQLdictionary *enumdict;	/* valid, if enum type */

	ArrowType	arrow_type;		/* type in apache arrow */
	const char *arrow_typename;	/* typename in apache arrow */
	/* data save as Apache Arrow datum */
	size_t	(*put_value)(SQLfield *attr, const char *addr, int sz);
	/* data buffers of the field */
	long		nitems;			/* number of rows */
	long		nullcount;		/* number of null values */
	SQLbuffer	nullmap;		/* null bitmap */
	SQLbuffer	values;			/* main storage of values */
	SQLbuffer	extra;			/* extra buffer for varlena */
	size_t		__curr_usage__;	/* current buffer usage */
	/* custom metadata(optional) */
	ArrowKeyValue *customMetadata;
	int			numCustomMetadata;
};
static inline size_t
sql_field_put_value(SQLfield *column, const char *addr, int sz)
{
	return (column->__curr_usage__ = column->put_value(column, addr, sz));
}

struct SQLtable
{
	const char *filename;		/* output filename */
	int			fdesc;			/* output file descriptor */
	ArrowBlock *recordBatches;	/* recordBatches written in the past */
	int			numRecordBatches;
	ArrowBlock *dictionaries;	/* dictionaryBatches written in the past */
	int			numDictionaries;
	int			numFieldNodes;	/* # of FieldNode vector elements */
	int			numBuffers;		/* # of Buffer vector elements */
	ArrowKeyValue *customMetadata; /* custom metadata, if any */
	int			numCustomMetadata;
	SQLdictionary *sql_dict_list; /* list of SQLdictionary */
	size_t		segment_sz;		/* threshold of the memory usage */
	int			nbatches;		/* number of buffered record-batches */
	int			nfields;		/* number of attributes */
	SQLfield columns[FLEXIBLE_ARRAY_MEMBER];
};
#define SQLtableGetColumns(table,index)								\
	((table)->columns + (table)->nfields * (index))
#define SQLtableLatestColumns(table)									\
	SQLtableGetColumns((table),(table)->nbatches - 1)

typedef struct hashItem		hashItem;
struct hashItem
{
	struct hashItem	*next;
	uint32		hash;
	uint32		index;
	uint32		label_len;
	char		label[FLEXIBLE_ARRAY_MEMBER];
};

struct SQLdictionary
{
	struct SQLdictionary *next;
	Oid			enum_typeid;
	int			dict_id;
	char		is_delta;
	SQLbuffer	values;
	SQLbuffer	extra;
	int			nitems;
	int			nslots;			/* width of hash slot */
	hashItem   *hslots[FLEXIBLE_ARRAY_MEMBER];
};

/*
 * Error message and exit
 */
#ifndef __PG2ARROW__
#define Elog(fmt, ...)		elog(ERROR,(fmt),##__VA_ARGS__)
#else
#define Elog(fmt, ...)								\
	do {											\
		fprintf(stderr,"%s:%d  " fmt "\n",			\
				__FILE__,__LINE__, ##__VA_ARGS__);	\
		exit(1);									\
	} while(0)
#endif

/* arrow_write.c */
extern ssize_t	writeArrowSchema(SQLtable *table);
extern void		writeArrowDictionaryBatches(SQLtable *table);
extern void		writeArrowRecordBatch(SQLtable *table,
									  SQLfield *attrs);
extern ssize_t	writeArrowFooter(SQLtable *table);
extern size_t	estimateArrowBufferLength(SQLfield *column, size_t nitems);

/* arrow_nodes.c */
extern void		__initArrowNode(ArrowNode *node, ArrowNodeTag tag);
#define initArrowNode(PTR,NAME)					\
	__initArrowNode((ArrowNode *)(PTR),ArrowNodeTag__##NAME)
extern char	   *dumpArrowNode(ArrowNode *node);
extern void		copyArrowNode(ArrowNode *dest, const ArrowNode *src);
extern void		readArrowFileDesc(int fdesc, ArrowFileInfo *af_info);
extern char	   *arrowTypeName(ArrowField *field);

/* arrow_buf.c */
extern void		sql_field_rewind(SQLfield *column, size_t nitems);
extern void		sql_table_rewind(SQLtable *table, int nbatches, size_t nitems);
extern SQLtable *sql_table_expand(SQLtable *table);
extern void		sql_buffer_init(SQLbuffer *buf);
extern void		sql_buffer_expand(SQLbuffer *buf, size_t required);
extern void		sql_buffer_append(SQLbuffer *buf, const void *src, size_t len);
extern void		sql_buffer_append_zero(SQLbuffer *buf, size_t len);
extern void		sql_buffer_setbit(SQLbuffer *buf, size_t __index);
extern void		sql_buffer_clrbit(SQLbuffer *buf, size_t __index);
extern void		sql_buffer_clear(SQLbuffer *buf);
extern void		sql_buffer_copy(SQLbuffer *dst, const SQLbuffer *src);
extern void		sql_buffer_write(int fdesc, SQLbuffer *buf);


extern void		rewindArrowTypeBuffer(SQLfield *column, size_t nitems);
extern void		duplicateArrowTypeBuffer(SQLfield *dest, SQLfield *source);

/* arrow_pgsql.c */
extern int		assignArrowTypePgSQL(SQLfield *column,
									 const char *field_name,
									 Oid typeid,
									 int typmod,
									 const char *typname,
									 const char *typnamespace,
									 short typlen,
									 bool typbyval,
									 char typtype,
									 char typalign,
									 Oid typelem,
									 Oid typrelid);
/*
 * Misc functions
 */
extern void initStringInfo(StringInfo buf);
extern void resetStringInfo(StringInfo buf);
extern void appendStringInfo(StringInfo buf,
							 const char *fmt,...) pg_attribute_printf(2, 3);
extern Datum hash_any(const unsigned char *k, int keylen);

#endif	/* ARROW_IPC_H */
