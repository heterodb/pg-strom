/*
 * arrow_ipc.h
 *
 * Definitions for Apache Arrow IPC stuff.
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef ARROW_IPC_H
#define ARROW_IPC_H
#ifdef __PGSTROM_MODULE__
#include "pg_strom.h"
#endif
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>
#include <time.h>

#include "arrow_defs.h"

/* several primitive definitions */
#ifndef offsetof
//#define offsetof(type,field)		((long) &((type *)0UL)->field)
#endif
#ifndef lengthof
//#define lengthof(array)				(sizeof(array) / sizeof((array)[0]))
#endif

#ifndef TYPEALIGN
#define TYPEALIGN(ALIGNVAL,LEN)					\
	(((uint64_t)(LEN) + ((ALIGNVAL) - 1)) & ~((uint64_t)((ALIGNVAL) - 1)))
#endif
#define ARROWALIGN(LEN)			TYPEALIGN(64,(LEN))

#ifndef Oid
typedef unsigned int	Oid;
#endif

#ifdef PG_INT128_TYPE
typedef PG_INT128_TYPE			int128_t;
typedef unsigned PG_INT128_TYPE	uint128_t;
#else
#ifndef HAVE_INT128_T
#define HAVE_INT128_T 1
typedef __int128				int128_t;
#endif
#ifndef HAVE_UINT128_T
#define HAVE_UINT128_T 1
typedef unsigned __int128		uint128_t;
#endif
#endif

typedef struct SQLbuffer		SQLbuffer;
typedef struct SQLtable			SQLtable;
typedef struct SQLfield			SQLfield;
typedef struct SQLdictionary	SQLdictionary;
typedef struct SQLstat			SQLstat;
typedef union  SQLstat__datum	SQLstat__datum;
typedef union  SQLtype			SQLtype;
typedef struct SQLtype__pgsql	SQLtype__pgsql;
typedef struct SQLtype__mysql	SQLtype__mysql;
typedef struct SQLtype__fluent	SQLtype__fluent;

struct SQLbuffer
{
	char	   *data;
	uint32_t	usage;
	uint32_t	length;
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
	uint8_t		typalign;
};

struct SQLtype__mysql
{
	int			typeid;
};

struct SQLtype__fluent
{
	bool		ts_column;		/* source is 'ts' (timestamp) */
	bool		tag_column;		/* source is 'tag' (string) */
};

union SQLtype
{
	SQLtype__pgsql	pgsql;
	SQLtype__mysql	mysql;
	SQLtype__fluent	fluent;
};

union SQLstat__datum
{
	int8_t			i8;
	uint8_t			u8;
	int16_t			i16;
	uint16_t		u16;
	int32_t			i32;
	int32_t			u32;
	int64_t			i64;
	int64_t			u64;
	int128_t		i128;
	float			f32;
	double			f64;
};

struct SQLstat
{
	SQLstat		   *next;
	int				rb_index;	/* record-batch index */
	bool			is_valid;	/* true, if min/max is not NULL */
	SQLstat__datum	min;
	SQLstat__datum	max;
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
	/* data save as Apache Arrow datum */
	size_t	(*put_value)(SQLfield *attr, const char *addr, int sz);
	int		(*write_stat)(SQLfield *attr, char *buf, size_t len,
						  const SQLstat__datum *stat_datum);
	/* data buffers of the field */
	long		nitems;			/* number of rows */
	long		nullcount;		/* number of null values */
	SQLbuffer	nullmap;		/* null bitmap */
	SQLbuffer	values;			/* main storage of values */
	SQLbuffer	extra;			/* extra buffer for varlena */
	size_t		__curr_usage__;	/* current buffer usage */
	/* min/max statistics */
	bool		stat_enabled;
	SQLstat		stat_datum;
	SQLstat	   *stat_list;
	/* custom metadata(optional) */
	ArrowKeyValue *customMetadata;
	int			numCustomMetadata;
};

static inline size_t
sql_field_put_value(SQLfield *column, const char *addr, int sz)
{
	return (column->__curr_usage__ = column->put_value(column, addr, sz));
}

#ifndef FLEXIBLE_ARRAY_MEMBER
#define FLEXIBLE_ARRAY_MEMBER
#endif

struct SQLtable
{
	const char *filename;		/* output filename */
	int			fdesc;			/* output file descriptor */
	off_t		f_pos;			/* current file position */
	int			__iov_len;		/* for internal use of pwritev support */
	int			__iov_cnt;
	struct iovec *__iov;

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
	size_t		usage;			/* current buffer usage */
	size_t		nitems;			/* number of items */
	int			nfields;		/* number of attributes */
	bool		has_statistics;	/* one or more columns enable min/max statistics */
	SQLfield columns[FLEXIBLE_ARRAY_MEMBER];
};

typedef struct hashItem		hashItem;
struct hashItem
{
	struct hashItem	*next;
	uint32_t	hash;
	uint32_t	index;
	uint32_t	label_sz;
	char		label[FLEXIBLE_ARRAY_MEMBER];
};

struct SQLdictionary
{
	struct SQLdictionary *next;
	int64_t		dict_id;
	SQLbuffer	values;
	SQLbuffer	extra;
	int			nloaded;	/* # of items loaded from existing file */
	int			nitems;
	int			nslots;		/* width of hash slot */
	hashItem   *hslots[FLEXIBLE_ARRAY_MEMBER];
};

/* arrow_write.c */
extern void		arrowFileWrite(SQLtable *table,
							   const char *buffer,
							   ssize_t length);
extern void		arrowFileWriteIOV(SQLtable *table);
extern void		writeArrowSchema(SQLtable *table);
extern void		writeArrowDictionaryBatches(SQLtable *table);
extern int		writeArrowRecordBatch(SQLtable *table);
extern void		writeArrowFooter(SQLtable *table);

extern size_t	setupArrowRecordBatchIOV(SQLtable *table);

/* arrow_nodes.c */
extern void		__initArrowNode(ArrowNode *node, ArrowNodeTag tag);
#define initArrowNode(PTR,NAME)					\
	__initArrowNode((ArrowNode *)(PTR),ArrowNodeTag__##NAME)
extern char	   *dumpArrowNode(ArrowNode *node);
extern void		copyArrowNode(ArrowNode *dest, const ArrowNode *src);
extern void		readArrowFileDesc(int fdesc, ArrowFileInfo *af_info);
extern bool		arrowFieldTypeIsEqual(ArrowField *a, ArrowField *b);
extern const char *arrowNodeName(ArrowNode *node);

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
									 Oid typrelid,
									 Oid typelem,
									 const char *tz_name,
									 const char *extname,
									 const char *extschema,
									 ArrowField *arrow_field);
/*
 * Error messages, and misc definitions for pg2arrow
 */
#ifndef Elog
#ifdef __PGSTROM_MODULE__
#define Elog(fmt, ...)			elog(ERROR,(fmt),##__VA_ARGS__)
#else /* __PGSTROM_MODULE__ */
#define Elog(fmt, ...)								\
	do {											\
		fprintf(stderr,"%s:%d  " fmt "\n",			\
				__FILE__,__LINE__, ##__VA_ARGS__);	\
		exit(1);									\
	} while(0)
#endif	/* __PGSTROM_MODULE__ */
#endif	/* Elog() */

/*
 * SQLbuffer related routines
 */
extern void	   *palloc(size_t sz);
extern void	   *palloc0(size_t sz);
extern char	   *pstrdup(const char *orig);
extern void	   *repalloc(void *ptr, size_t sz);
extern void		pfree(void *ptr);

static inline void
sql_buffer_init(SQLbuffer *buf)
{
	buf->data = NULL;
	buf->usage = 0;
	buf->length = 0;
}

/* nullmap + values */
static inline size_t
__buffer_usage_inline_type(SQLfield *column)
{
	size_t		usage;

	usage = ARROWALIGN(column->values.usage);
	if (column->nullcount > 0)
		usage += ARROWALIGN(column->nullmap.usage);
	return usage;
}

/* nullmap + index + values */
static inline size_t
__buffer_usage_varlena_type(SQLfield *column)
{
	size_t		usage;

	usage = (ARROWALIGN(column->values.usage) +
			 ARROWALIGN(column->extra.usage));
	if (column->nullcount > 0)
		usage += ARROWALIGN(column->nullmap.usage);
	return usage;
}

static inline void
sql_buffer_expand(SQLbuffer *buf, size_t required)
{
	if (buf->length < required)
	{
		void	   *data;
		size_t		length;

		if (buf->data == NULL)
		{
			length = (1UL << 20);	/* start from 1MB */
			while (length < required)
				length *= 2;
			data = palloc(length);
			if (!data)
				Elog("palloc: out of memory (sz=%zu)", length);
			buf->data   = data;
			buf->usage  = 0;
			buf->length = length;
		}
		else
		{
			length = buf->length;
			while (length < required)
				length *= 2;
			data = repalloc(buf->data, length);
			if (!data)
				Elog("repalloc: out of memory (sz=%zu)", length);
			buf->data = data;
			buf->length = length;
		}
	}
}

static inline void
sql_buffer_append(SQLbuffer *buf, const void *src, size_t len)
{
	sql_buffer_expand(buf, buf->usage + len);
	memcpy(buf->data + buf->usage, src, len);
	buf->usage += len;
	assert(buf->usage <= buf->length);
}

static inline void
sql_buffer_append_zero(SQLbuffer *buf, size_t len)
{
	if (len > 0)
	{
		sql_buffer_expand(buf, buf->usage + len);
		memset(buf->data + buf->usage, 0, len);
		buf->usage += len;
	}
	assert(buf->usage <= buf->length);
}

static inline void
sql_buffer_append_char(SQLbuffer *buf, int c, size_t len)
{
	if (len > 0)
	{
		sql_buffer_expand(buf, buf->usage + len);
		memset(buf->data + buf->usage, c, len);
		buf->usage += len;
	}
	assert(buf->usage <= buf->length);
}

static inline void
sql_buffer_setbit(SQLbuffer *buf, size_t __index)
{
	size_t		index = __index >> 3;
	int			mask  = (1 << (__index & 7));

	sql_buffer_expand(buf, index + 1);
	((uint8_t *)buf->data)[index] |= mask;
	if (buf->usage < index + 1)
		buf->usage = index + 1;
}

static inline void
sql_buffer_clrbit(SQLbuffer *buf, size_t __index)
{
	size_t		index = __index >> 3;
	int			mask  = (1 << (__index & 7));

	sql_buffer_expand(buf, index + 1);
	((uint8_t *)buf->data)[index] &= ~mask;
	if (buf->usage < index + 1)
		buf->usage = index + 1;
}

static inline void
sql_buffer_clear(SQLbuffer *buf)
{
	buf->usage = 0;
}

static inline void
sql_buffer_copy(SQLbuffer *dest, const SQLbuffer *orig)
{
	sql_buffer_init(dest);
	if (orig->data)
	{
		assert(orig->usage <= orig->length);
		sql_buffer_expand(dest, orig->length);
		memcpy(dest->data, orig->data, orig->usage);
		dest->usage = orig->usage;
	}
}

static inline void
sql_field_clear(SQLfield *column)
{
	int		j;

	column->nitems = 0;
	column->nullcount = 0;
	sql_buffer_clear(&column->nullmap);
	sql_buffer_clear(&column->values);
	sql_buffer_clear(&column->extra);
	column->__curr_usage__ = 0;

	if (column->element)
		sql_field_clear(column->element);
	if (column->nfields > 0)
	{
		for (j=0; j < column->nfields; j++)
			sql_field_clear(&column->subfields[j]);
	}
}

static inline void
sql_table_clear(SQLtable *table)
{
	int		j;

	for (j=0; j < table->nfields; j++)
		sql_field_clear(&table->columns[j]);
	table->nitems = 0;
	table->usage = 0;
}

static inline int
sql_table_append_record_batch(SQLtable *table, ArrowBlock *block)
{
	int			index = table->numRecordBatches++;

	if (!table->recordBatches)
		table->recordBatches = palloc(sizeof(ArrowBlock) * 32);
	else
	{
		size_t	sz = sizeof(ArrowBlock) * (index + 1);
		table->recordBatches = repalloc(table->recordBatches, sz);
	}
	memcpy(&table->recordBatches[index], block, sizeof(ArrowBlock));

	return index;
}
#endif	/* ARROW_IPC_H */
