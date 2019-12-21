/*
 * pg2arrow.h
 *
 * common header file
 */
#ifndef PG2ARROW_H
#define PG2ARROW_H
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif	/* _GNU_SOURCE */

#include "postgres.h"
#include "access/htup_details.h"
#include "datatype/timestamp.h"
#include "utils/date.h"
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

#include <libpq-fe.h>

#define	ARROWALIGN(LEN)		TYPEALIGN(64, (LEN))

typedef struct SQLbuffer		SQLbuffer;
typedef struct SQLtable			SQLtable;
typedef struct SQLattribute		SQLattribute;
typedef struct SQLdictionary	SQLdictionary;
/* fake StringInfo definition */
typedef struct SQLbuffer		StringInfoData;
typedef struct SQLbuffer	   *StringInfo;
#include "arrow_defs.h"

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
	SQLbuffer	values;
	SQLbuffer	extra;
	int			nitems;
	int			nslots;			/* width of hash slot */
	hashItem   *hslots[FLEXIBLE_ARRAY_MEMBER];
};

/* pg2arrow.c */
extern int			shows_progress;
extern void			writeArrowRecordBatch(SQLtable *table,
										  size_t *p_metaLength,
										  size_t *p_bodyLength);
/* query.c */
extern SQLdictionary *pgsql_dictionary_list;
extern SQLtable	   *pgsql_create_buffer(PGconn *conn, PGresult *res,
								size_t segment_sz);
extern void			pgsql_append_results(SQLtable *table, PGresult *res);
extern void 		pgsql_writeout_buffer(SQLtable *table);
extern void			pgsql_dump_buffer(SQLtable *table);
/* arrow_write.c */
extern ssize_t		writeFlatBufferMessage(int fdesc, ArrowMessage *message);
extern ssize_t		writeFlatBufferFooter(int fdesc, ArrowFooter *footer);
/* arrow_types.c */
extern void			assignArrowType(SQLattribute *attr, int *p_numBuffers);
/* arrow_read.c */
extern void			readArrowFile(const char *pathname);
/* arrow_dump.c */
extern char		   *dumpArrowNode(ArrowNode *node);

/*
 * Checker of ArrowNode Tag
 */
#define ArrowNodeTag(PTR)			(((ArrowNode *)(PTR))->tag)
#define ArrowNodeIs(PTR,NAME)		(ArrowNodeTag(PTR) == ArrowNodeTag__##NAME)

/*
 * Error message and exit
 */
#define Elog(fmt, ...)								\
	do {											\
		fprintf(stderr,"%s:%d  " fmt "\n",			\
				__FILE__,__LINE__, ##__VA_ARGS__);	\
		exit(1);									\
	} while(0)

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

/*
 * Fake StringInfo operations
 */
static inline void
initStringInfo(StringInfo buf)
{
	sql_buffer_init(buf);
}

static inline void
resetStringInfo(StringInfo buf)
{
	buf->usage = 0;
}

static inline void
appendStringInfo(StringInfo buf, const char *fmt,...)
{
	if (!buf->data)
		sql_buffer_expand(buf, 0);
	for (;;)
	{
		char	   *pos = buf->data + buf->usage;
		size_t		len = buf->length - buf->usage;
		int			nbytes;
		va_list		args;

		va_start(args, fmt);
		nbytes = vsnprintf(pos, len, fmt, args);
		va_end(args);

		if (nbytes < len)
		{
			buf->usage += nbytes;
			break;
		}
		sql_buffer_expand(buf, 2 * nbytes + 1024);
	}
}

/*
 * Hash-function -- from the src/backend/access/hash/hashfunc.c
 */

/*
 * This hash function was written by Bob Jenkins
 * (bob_jenkins@burtleburtle.net), and superficially adapted
 * for PostgreSQL by Neil Conway. For more information on this
 * hash function, see http://burtleburtle.net/bob/hash/doobs.html,
 * or Bob's article in Dr. Dobb's Journal, Sept. 1997.
 *
 * In the current code, we have adopted Bob's 2006 update of his hash
 * function to fetch the data a word at a time when it is suitably aligned.
 * This makes for a useful speedup, at the cost of having to maintain
 * four code paths (aligned vs unaligned, and little-endian vs big-endian).
 * It also uses two separate mixing functions mix() and final(), instead
 * of a slower multi-purpose function.
 */

/* Get a bit mask of the bits set in non-uint32 aligned addresses */
#define UINT32_ALIGN_MASK (sizeof(uint32) - 1)

/* Rotate a uint32 value left by k bits - note multiple evaluation! */
#define rot(x,k) (((x)<<(k)) | ((x)>>(32-(k))))

/*----------
 * mix -- mix 3 32-bit values reversibly.
 *
 * This is reversible, so any information in (a,b,c) before mix() is
 * still in (a,b,c) after mix().
 *
 * If four pairs of (a,b,c) inputs are run through mix(), or through
 * mix() in reverse, there are at least 32 bits of the output that
 * are sometimes the same for one pair and different for another pair.
 * This was tested for:
 * * pairs that differed by one bit, by two bits, in any combination
 *	 of top bits of (a,b,c), or in any combination of bottom bits of
 *	 (a,b,c).
 * * "differ" is defined as +, -, ^, or ~^.  For + and -, I transformed
 *	 the output delta to a Gray code (a^(a>>1)) so a string of 1's (as
 *	 is commonly produced by subtraction) look like a single 1-bit
 *	 difference.
 * * the base values were pseudorandom, all zero but one bit set, or
 *	 all zero plus a counter that starts at zero.
 *
 * This does not achieve avalanche.  There are input bits of (a,b,c)
 * that fail to affect some output bits of (a,b,c), especially of a.  The
 * most thoroughly mixed value is c, but it doesn't really even achieve
 * avalanche in c.
 *
 * This allows some parallelism.  Read-after-writes are good at doubling
 * the number of bits affected, so the goal of mixing pulls in the opposite
 * direction from the goal of parallelism.  I did what I could.  Rotates
 * seem to cost as much as shifts on every machine I could lay my hands on,
 * and rotates are much kinder to the top and bottom bits, so I used rotates.
 *----------
 */
#define mix(a,b,c) \
{ \
  a -= c;  a ^= rot(c, 4);	c += b; \
  b -= a;  b ^= rot(a, 6);	a += c; \
  c -= b;  c ^= rot(b, 8);	b += a; \
  a -= c;  a ^= rot(c,16);	c += b; \
  b -= a;  b ^= rot(a,19);	a += c; \
  c -= b;  c ^= rot(b, 4);	b += a; \
}

/*----------
 * final -- final mixing of 3 32-bit values (a,b,c) into c
 *
 * Pairs of (a,b,c) values differing in only a few bits will usually
 * produce values of c that look totally different.  This was tested for
 * * pairs that differed by one bit, by two bits, in any combination
 *	 of top bits of (a,b,c), or in any combination of bottom bits of
 *	 (a,b,c).
 * * "differ" is defined as +, -, ^, or ~^.  For + and -, I transformed
 *	 the output delta to a Gray code (a^(a>>1)) so a string of 1's (as
 *	 is commonly produced by subtraction) look like a single 1-bit
 *	 difference.
 * * the base values were pseudorandom, all zero but one bit set, or
 *	 all zero plus a counter that starts at zero.
 *
 * The use of separate functions for mix() and final() allow for a
 * substantial performance increase since final() does not need to
 * do well in reverse, but is does need to affect all output bits.
 * mix(), on the other hand, does not need to affect all output
 * bits (affecting 32 bits is enough).  The original hash function had
 * a single mixing operation that had to satisfy both sets of requirements
 * and was slower as a result.
 *----------
 */
#define final(a,b,c) \
{ \
  c ^= b; c -= rot(b,14); \
  a ^= c; a -= rot(c,11); \
  b ^= a; b -= rot(a,25); \
  c ^= b; c -= rot(b,16); \
  a ^= c; a -= rot(c, 4); \
  b ^= a; b -= rot(a,14); \
  c ^= b; c -= rot(b,24); \
}

/*
 * hash_any() -- hash a variable-length key into a 32-bit value
 *		k		: the key (the unaligned variable-length array of bytes)
 *		len		: the length of the key, counting by bytes
 *
 * Returns a uint32 value.  Every bit of the key affects every bit of
 * the return value.  Every 1-bit and 2-bit delta achieves avalanche.
 * About 6*len+35 instructions. The best hash table sizes are powers
 * of 2.  There is no need to do mod a prime (mod is sooo slow!).
 * If you need less than 32 bits, use a bitmask.
 *
 * This procedure must never throw elog(ERROR); the ResourceOwner code
 * relies on this not to fail.
 *
 * Note: we could easily change this function to return a 64-bit hash value
 * by using the final values of both b and c.  b is perhaps a little less
 * well mixed than c, however.
 */
static inline uint32
hash_any(const unsigned char *k, int keylen)
{
	register uint32 a,
				b,
				c,
				len;

	/* Set up the internal state */
	len = keylen;
	a = b = c = 0x9e3779b9 + len + 3923095;

	/* If the source pointer is word-aligned, we use word-wide fetches */
	if (((uintptr_t) k & UINT32_ALIGN_MASK) == 0)
	{
		/* Code path for aligned source data */
		register const uint32 *ka = (const uint32 *) k;

		/* handle most of the key */
		while (len >= 12)
		{
			a += ka[0];
			b += ka[1];
			c += ka[2];
			mix(a, b, c);
			ka += 3;
			len -= 12;
		}

		/* handle the last 11 bytes */
		k = (const unsigned char *) ka;
#ifdef WORDS_BIGENDIAN
		switch (len)
		{
			case 11:
				c += ((uint32) k[10] << 8);
				/* fall through */
			case 10:
				c += ((uint32) k[9] << 16);
				/* fall through */
			case 9:
				c += ((uint32) k[8] << 24);
				/* fall through */
			case 8:
				/* the lowest byte of c is reserved for the length */
				b += ka[1];
				a += ka[0];
				break;
			case 7:
				b += ((uint32) k[6] << 8);
				/* fall through */
			case 6:
				b += ((uint32) k[5] << 16);
				/* fall through */
			case 5:
				b += ((uint32) k[4] << 24);
				/* fall through */
			case 4:
				a += ka[0];
				break;
			case 3:
				a += ((uint32) k[2] << 8);
				/* fall through */
			case 2:
				a += ((uint32) k[1] << 16);
				/* fall through */
			case 1:
				a += ((uint32) k[0] << 24);
				/* case 0: nothing left to add */
		}
#else							/* !WORDS_BIGENDIAN */
		switch (len)
		{
			case 11:
				c += ((uint32) k[10] << 24);
				/* fall through */
			case 10:
				c += ((uint32) k[9] << 16);
				/* fall through */
			case 9:
				c += ((uint32) k[8] << 8);
				/* fall through */
			case 8:
				/* the lowest byte of c is reserved for the length */
				b += ka[1];
				a += ka[0];
				break;
			case 7:
				b += ((uint32) k[6] << 16);
				/* fall through */
			case 6:
				b += ((uint32) k[5] << 8);
				/* fall through */
			case 5:
				b += k[4];
				/* fall through */
			case 4:
				a += ka[0];
				break;
			case 3:
				a += ((uint32) k[2] << 16);
				/* fall through */
			case 2:
				a += ((uint32) k[1] << 8);
				/* fall through */
			case 1:
				a += k[0];
				/* case 0: nothing left to add */
		}
#endif							/* WORDS_BIGENDIAN */
	}
	else
	{
		/* Code path for non-aligned source data */

		/* handle most of the key */
		while (len >= 12)
		{
#ifdef WORDS_BIGENDIAN
			a += (k[3] + ((uint32) k[2] << 8) + ((uint32) k[1] << 16) + ((uint32) k[0] << 24));
			b += (k[7] + ((uint32) k[6] << 8) + ((uint32) k[5] << 16) + ((uint32) k[4] << 24));
			c += (k[11] + ((uint32) k[10] << 8) + ((uint32) k[9] << 16) + ((uint32) k[8] << 24));
#else							/* !WORDS_BIGENDIAN */
			a += (k[0] + ((uint32) k[1] << 8) + ((uint32) k[2] << 16) + ((uint32) k[3] << 24));
			b += (k[4] + ((uint32) k[5] << 8) + ((uint32) k[6] << 16) + ((uint32) k[7] << 24));
			c += (k[8] + ((uint32) k[9] << 8) + ((uint32) k[10] << 16) + ((uint32) k[11] << 24));
#endif							/* WORDS_BIGENDIAN */
			mix(a, b, c);
			k += 12;
			len -= 12;
		}

		/* handle the last 11 bytes */
#ifdef WORDS_BIGENDIAN
		switch (len)
		{
			case 11:
				c += ((uint32) k[10] << 8);
				/* fall through */
			case 10:
				c += ((uint32) k[9] << 16);
				/* fall through */
			case 9:
				c += ((uint32) k[8] << 24);
				/* fall through */
			case 8:
				/* the lowest byte of c is reserved for the length */
				b += k[7];
				/* fall through */
			case 7:
				b += ((uint32) k[6] << 8);
				/* fall through */
			case 6:
				b += ((uint32) k[5] << 16);
				/* fall through */
			case 5:
				b += ((uint32) k[4] << 24);
				/* fall through */
			case 4:
				a += k[3];
				/* fall through */
			case 3:
				a += ((uint32) k[2] << 8);
				/* fall through */
			case 2:
				a += ((uint32) k[1] << 16);
				/* fall through */
			case 1:
				a += ((uint32) k[0] << 24);
				/* case 0: nothing left to add */
		}
#else							/* !WORDS_BIGENDIAN */
		switch (len)
		{
			case 11:
				c += ((uint32) k[10] << 24);
				/* fall through */
			case 10:
				c += ((uint32) k[9] << 16);
				/* fall through */
			case 9:
				c += ((uint32) k[8] << 8);
				/* fall through */
			case 8:
				/* the lowest byte of c is reserved for the length */
				b += ((uint32) k[7] << 24);
				/* fall through */
			case 7:
				b += ((uint32) k[6] << 16);
				/* fall through */
			case 6:
				b += ((uint32) k[5] << 8);
				/* fall through */
			case 5:
				b += k[4];
				/* fall through */
			case 4:
				a += ((uint32) k[3] << 24);
				/* fall through */
			case 3:
				a += ((uint32) k[2] << 16);
				/* fall through */
			case 2:
				a += ((uint32) k[1] << 8);
				/* fall through */
			case 1:
				a += k[0];
				/* case 0: nothing left to add */
		}
#endif							/* WORDS_BIGENDIAN */
	}

	final(a, b, c);

	/* report the result */
	return c;
}
#endif	/* PG2ARROW_H */
