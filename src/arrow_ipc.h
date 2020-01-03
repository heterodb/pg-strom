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
typedef struct SQLattribute		SQLattribute;
typedef struct SQLdictionary	SQLdictionary;

struct SQLbuffer
{
	char	   *data;
	uint32		usage;
	uint32		length;
};

struct SQLattribute
{
	char	   *attname;
	Oid			atttypid;
	int			atttypmod;
	short		attlen;
	bool		attbyval;
	uint8		attalign;		/* 1, 2, 4 or 8 */
	SQLtable   *subtypes;		/* valid, if composite type */
	SQLattribute *element;		/* valid, if array type */
	SQLdictionary *enumdict;	/* valid, if enum type */
	const char *typnamespace;	/* name of pg_type.typnamespace */
	const char *typname;		/* pg_type.typname */
	char		typtype;		/* pg_type.typtype */
	ArrowType	arrow_type;		/* type in apache arrow */
	const char *arrow_typename;	/* typename in apache arrow */
	/* data buffer and handler */
	void   (*put_value)(SQLattribute *attr,
						const char *addr, int sz);
	size_t (*buffer_usage)(SQLattribute *attr);
	int	   (*setup_buffer)(SQLattribute *attr,
						   ArrowBuffer *node,
						   size_t *p_offset);
	void   (*write_buffer)(SQLattribute *attr, int fdesc);

	long		nitems;			/* number of rows */
	long		nullcount;		/* number of null values */
	SQLbuffer	nullmap;		/* null bitmap */
	SQLbuffer	values;			/* main storage of values */
	SQLbuffer	extra;			/* extra buffer for varlena */
};

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
	SQLdictionary *sql_dict_list; /* list of SQLdictionary */
	size_t		segment_sz;		/* threshold of the memory usage */
	size_t		nitems;			/* current number of rows */
	int			nfields;		/* number of attributes */
	SQLattribute attrs[FLEXIBLE_ARRAY_MEMBER];
};

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

/* arrow_write.c */
extern ssize_t	writeArrowSchema(SQLtable *table);
extern void		writeArrowDictionaryBatches(SQLtable *table);
extern void		writeArrowRecordBatch(SQLtable *table);
extern ssize_t	writeArrowFooter(SQLtable *table);


/* arrow_nodes.c */
extern void		assignArrowType(SQLattribute *attr, int *p_numBuffers);
extern void		__initArrowNode(ArrowNode *node, ArrowNodeTag tag);
#define initArrowNode(PTR,NAME)					\
	__initArrowNode((ArrowNode *)(PTR),ArrowNodeTag__##NAME)
extern void		readArrowFileDesc(int fdesc, ArrowFileInfo *af_info);
extern char	   *dumpArrowNode(ArrowNode *node);

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
/*
 * SQLbuffer related routines
 */
static inline void
sql_buffer_init(SQLbuffer *buf)
{
	buf->data = NULL;
	buf->usage = 0;
	buf->length = 0;
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
			length = (1UL << 21);	/* start from 2MB */
			while (length < required)
				length *= 2;
			data = mmap(NULL, length, PROT_READ | PROT_WRITE,
						MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
			if (data == MAP_FAILED)
				Elog("failed on mmap(len=%zu): %m", length);
			buf->data   = data;
			buf->usage  = 0;
			buf->length = length;
		}
		else
		{
			length = 2 * buf->length;
			while (length < required)
				length *= 2;
			data = mremap(buf->data, buf->length, length, MREMAP_MAYMOVE);
			if (data == MAP_FAILED)
				Elog("failed on mremap(len=%zu): %m", length);
			buf->data   = data;
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
	sql_buffer_expand(buf, buf->usage + len);
	memset(buf->data + buf->usage, 0, len);
	buf->usage += len;
	assert(buf->usage <= buf->length);
}

static inline void
sql_buffer_setbit(SQLbuffer *buf, size_t __index)
{
	size_t		index = __index >> 3;
	int			mask  = (1 << (__index & 7));

	sql_buffer_expand(buf, index + 1);
	((uint8 *)buf->data)[index] |= mask;
	buf->usage = Max(buf->usage, index + 1);
}

static inline void
sql_buffer_clrbit(SQLbuffer *buf, size_t __index)
{
	size_t		index = __index >> 3;
	int			mask  = (1 << (__index & 7));

	sql_buffer_expand(buf, index + 1);
	((uint8 *)buf->data)[index] &= ~mask;
	buf->usage = Max(buf->usage, index + 1);
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
sql_buffer_write(int fdesc, SQLbuffer *buf)
{
	ssize_t		length = buf->usage;
	ssize_t		offset = 0;
	ssize_t		nbytes;

	while (offset < length)
	{
		nbytes = write(fdesc, buf->data + offset, length - offset);
		if (nbytes < 0)
		{
			if (errno == EINTR)
				continue;
			Elog("failed on write(2): %m");
		}
		offset += nbytes;
	}

	if (length != ARROWALIGN(length))
	{
		ssize_t	gap = ARROWALIGN(length) - length;
		char	zero[64];

		offset = 0;
		memset(zero, 0, sizeof(zero));
		while (offset < gap)
		{
			nbytes = write(fdesc, zero + offset, gap - offset);
			if (nbytes < 0)
			{
				if (errno == EINTR)
					continue;
				Elog("failed on write(2): %m");
			}
			offset += nbytes;
		}
	}
}

/*
 * Misc functions
 */
extern void initStringInfo(StringInfo buf);
extern void resetStringInfo(StringInfo buf);
extern void appendStringInfo(StringInfo buf,
							 const char *fmt,...) pg_attribute_printf(2, 3);
extern Datum hash_any(const unsigned char *k, int keylen);

#endif	/* ARROW_IPC_H */
