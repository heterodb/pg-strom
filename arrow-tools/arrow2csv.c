/*
 * arrow_dump
 *
 * A tool to dump Apache Arrow files
 *
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <ctype.h>
#include <getopt.h>
#include <netinet/in.h>
#include <stdarg.h>
#include <unistd.h>
#include "arrow_ipc.h"
#include "float2.h"

/* columns to dump */
#define ARROW_PRINT_DATUM_ARGS	\
	struct arrowColumn *column, \
	const char *rb_chunk,		\
	ArrowBuffer *buffers,		\
	int64_t index,				\
	const char *quote

typedef struct arrowColumn
{
	ArrowType	arrow_type;
	const char *colname;
	const char *sqltype;
	bool	  (*print_datum)(ARROW_PRINT_DATUM_ARGS);
	bool		nullable;
	int			buffer_index;
	int			buffer_count;
	struct arrowColumn *children;
	int			num_children;
} arrowColumn;

/* static variables */
static ArrowFileInfo *arrow_files = NULL;
static int		   *arrow_fdescs = NULL;
static int			arrow_num_files = -1;
static arrowColumn *arrow_columns = NULL;
static int			arrow_num_columns = 0;
static const char  *output_filename = NULL;
static FILE		   *output_filp = NULL;
static bool			print_header = false;
static char			csv_delimiter = ',';
static char			current_context = 'n';
static int64_t		num_skip_rows = -1;			/* --offset */
static int64_t		num_dump_rows = -1;			/* --limit */
static const char  *create_table_name = NULL;	/* --create-table */
static const char  *tablespace_name = NULL;		/* --tablespace */
static const char  *partition_name = NULL;		/* --partition-of */

static void
usage(void)
{
	fputs("usage:  arrow_dump OPTIONS <file1> [<file2> ...]\n\n"
		  "OPTIONS:\n"
		  "  -o|--output=FILENAME specify the output filename\n"
		  "                 (default: stdout)\n"
		  "  --header       dump column names as csv header\n"
		  "  --offset NUM   skip first NUM rows\n"
		  "  --limit NUM    dump only NUM rows\n"
		  "\n"
		  "  --create-table=TABLE_NAME  dump with CREATE TABLE statement\n"
		  "  --tablespace=TABLESPACE    specify tablespace of the table, if any\n"
		  "  --partition-of=PARENT_NAME specify partition-parent of the table, if any\n"
		  "\n"
		  "  --help         print this message.\n"
		  "\n"
		  "Report bugs to <pgstrom@heterodb.com>.\n",
		  stderr);
	exit(1);
}

static inline void
sql_buffer_printf(SQLbuffer *buf, const char *fmt, ...)
{
	va_list		ap;
	int			sz;

	sql_buffer_expand(buf, 1024);	/* minimum */
	for (;;)
	{
		assert(buf->usage <= buf->length);
		va_start(ap, fmt);
		sz = vsnprintf(buf->data + buf->usage,
					   buf->length - buf->usage,
					   fmt, ap);
		va_end(ap);
		if (sz < 0)
			Elog("failed on vsnprintf: %m");
		if (buf->usage + sz <= buf->length)
			break;
		sql_buffer_expand(buf, buf->length + sz + 1024);
	}
	buf->usage += sz;
}

static const char *
__quote_ident(const char *ident)
{
	const char *r_pos;
	char	   *w_pos, *buf;
	int			nquotes = 0;

	for (r_pos = ident; *r_pos != '\0'; r_pos++)
	{
		if (*r_pos == '"')
			nquotes++;
	}
	buf = w_pos = palloc(strlen(ident) + nquotes + 3);
	*w_pos++ = '"';
	for (r_pos = ident; *r_pos != '\0'; r_pos++)
	{
		if (*r_pos == '"')
			*w_pos++ = '"';
		*w_pos++ = *r_pos;
	}
	*w_pos++ = '"';
	*w_pos++ = '\0';

	return buf;
}

static const char *
quote_ident(const char *ident)
{
	const char *pos;

	if (!isalpha(*ident) && *ident != '_')
		return __quote_ident(ident);
	for (pos = ident; *pos != '\0'; pos++)
	{
		if (!isalnum(*pos) && *pos != '_')
			return __quote_ident(ident);
	}
	return ident;
}

static void
printNullDatum(void)
{
	/* print "null" only if List elements */
	if (current_context == 'e')
		fprintf(output_filp, "null");
}

static void
printArrowDatum(arrowColumn *column,
				ArrowBuffer *buffers,
				const char *rb_chunk,
				int64_t index,
				const char *quote)
{
	/* null checks */
	if (column->nullable)
	{
		const char *nullmap = rb_chunk + buffers[0].offset;
		int64_t		k = (index >> 3);
		int32_t		mask = (1 << (index & 7));

		if (k < buffers[0].length && (nullmap[k] & mask) == 0)
		{
			printNullDatum();
			return;
		}
	}
	if (column->print_datum(column, rb_chunk, buffers, index, quote))
		return;
	printNullDatum();
}

static inline const void *
__print_arrow_common_varlena(ARROW_PRINT_DATUM_ARGS, size_t *p_sz)
{
	const uint32_t *__values;
	uint32_t		__head;
	uint32_t		__tail;

	if (sizeof(uint32_t) * (index+2) > buffers[1].length)
		return NULL;
	__values = (const uint32_t *)(rb_chunk + buffers[1].offset);
	__head = __values[index];
	__tail = __values[index+1];
	if (__tail < __head ||
		__tail > buffers[2].length)
		return NULL;
	*p_sz = __tail - __head;
	return rb_chunk + buffers[2].offset + __head;
}

#define ARROW_PRINT_DATUM_SETUP_INLINE(__TYPE)				\
	__TYPE		datum;										\
	if (sizeof(__TYPE) * (index+1) > buffers[1].length)		\
		return false;										\
	datum = ((const __TYPE *)(rb_chunk + buffers[1].offset))[index]

#define ARROW_PRINT_DATUM_SETUP_FIXEDSIZEBINARY(__WIDTH)	\
	const unsigned char *addr;								\
	if ((__WIDTH) * (index+1) > buffers[1].length)			\
		return false;										\
	addr = (const unsigned char *)							\
		(rb_chunk + buffers[1].offset + (__WIDTH) * index)

static inline const void *
__arrow_fetch_varlena32(const char *rb_chunk,
						ArrowBuffer *buffers,
						int64_t index,
						size_t *p_sz)
{
	const uint32_t *values;
	uint32_t		head, tail;

	if (sizeof(uint32_t) * (index+2) > buffers[1].length)
		return (const void *)NULL;
	values = (const uint32_t *)(rb_chunk + buffers[1].offset);
	head = values[index];
	tail = values[index+1];
	if (tail > buffers[2].length)
		return (const void *)NULL;
	*p_sz = (tail - head);
	return (rb_chunk + buffers[2].offset + head);
}

static inline const void *
__arrow_fetch_varlena64(const char *rb_chunk,
						ArrowBuffer *buffers,
						int64_t index,
						size_t *p_sz)
{
	const uint64_t *values;
	uint64_t		head, tail;

	if (sizeof(uint64_t) * (index+2) > buffers[1].length)
		return (const void *)NULL;
	values = (const uint64_t *)(rb_chunk + buffers[1].offset);
	head = values[index];
	tail = values[index+1];
	if (tail > buffers[2].length)
		return (const void *)NULL;
	*p_sz = (tail - head);
	return (rb_chunk + buffers[2].offset + head);
}

static bool
print_arrow_int8(ARROW_PRINT_DATUM_ARGS)
{
	ARROW_PRINT_DATUM_SETUP_INLINE(int8_t);
	fprintf(output_filp, "%d", (int)datum);
	return true;
}

static bool
print_arrow_uint8(ARROW_PRINT_DATUM_ARGS)
{
	ARROW_PRINT_DATUM_SETUP_INLINE(uint8_t);
	fprintf(output_filp, "%u", (unsigned int)datum);
	return true;
}

static bool
print_arrow_int16(ARROW_PRINT_DATUM_ARGS)
{
	ARROW_PRINT_DATUM_SETUP_INLINE(int16_t);
	fprintf(output_filp, "%d", (int)datum);
	return true;
}

static bool
print_arrow_uint16(ARROW_PRINT_DATUM_ARGS)
{
	ARROW_PRINT_DATUM_SETUP_INLINE(uint16_t);
	fprintf(output_filp, "%u", (unsigned int)datum);
	return true;
}

static bool
print_arrow_int32(ARROW_PRINT_DATUM_ARGS)
{
	ARROW_PRINT_DATUM_SETUP_INLINE(int32_t);
	fprintf(output_filp, "%d", datum);
	return true;
}

static bool
print_arrow_uint32(ARROW_PRINT_DATUM_ARGS)
{
	ARROW_PRINT_DATUM_SETUP_INLINE(uint32_t);
	fprintf(output_filp, "%u", datum);
	return true;
}

static bool
print_arrow_int64(ARROW_PRINT_DATUM_ARGS)
{
	ARROW_PRINT_DATUM_SETUP_INLINE(int64_t);
	fprintf(output_filp, "%ld", datum);
	return true;
}

static bool
print_arrow_uint64(ARROW_PRINT_DATUM_ARGS)
{
	ARROW_PRINT_DATUM_SETUP_INLINE(uint64_t);
	fprintf(output_filp, "%lu", datum);
	return true;
}

static bool
print_arrow_float2(ARROW_PRINT_DATUM_ARGS)
{
	ARROW_PRINT_DATUM_SETUP_INLINE(uint16_t);
	fprintf(output_filp, "%f", fp16_to_fp64(datum));
	return true;
}

static bool
print_arrow_float4(ARROW_PRINT_DATUM_ARGS)
{
	ARROW_PRINT_DATUM_SETUP_INLINE(float);
	fprintf(output_filp, "%f", (double)datum);
	return true;
}

static bool
print_arrow_float8(ARROW_PRINT_DATUM_ARGS)
{
	ARROW_PRINT_DATUM_SETUP_INLINE(double);
	fprintf(output_filp, "%f", datum);
	return true;
}

static bool
__print_arrow_utf8_common(const char *addr, size_t sz, const char *quote)
{
	size_t	i;

	fprintf(output_filp, "%s", quote);
	for (i=0; i < sz; i++)
	{
		int		c = (unsigned char)addr[i];

		if (c == '"')
			fputc('"', output_filp);
		fputc(c, output_filp);
	}
	fprintf(output_filp, "%s", quote);
	return true;
}

static bool
print_arrow_utf8(ARROW_PRINT_DATUM_ARGS)
{
	const char *addr;
	size_t		sz;

	addr = __arrow_fetch_varlena32(rb_chunk, buffers, index, &sz);
	if (!addr)
		return false;
	return __print_arrow_utf8_common(addr, sz, quote);
}

static bool
print_arrow_large_utf8(ARROW_PRINT_DATUM_ARGS)
{
	const char *addr;
	size_t		sz;

	addr = __arrow_fetch_varlena64(rb_chunk, buffers, index, &sz);
	if (!addr)
		return false;
	return __print_arrow_utf8_common(addr, sz, quote);
}

static bool
__print_arrow_binary_common(const char *addr, size_t sz, const char *quote)
{
	static const char hextbl[] = "0123456789abcdef";
	size_t	i;

	fprintf(output_filp, "%s\\x", quote);
	for (i=0; i < sz; i++)
	{
		int		c = (unsigned char)addr[i];

		fputc(hextbl[(c >> 4) & 0x0f], output_filp);
		fputc(hextbl[(c & 0x0f)], output_filp);
	}
	fprintf(output_filp, "%s", quote);
	return true;
}

static bool
print_arrow_binary(ARROW_PRINT_DATUM_ARGS)
{
	const char *addr;
	size_t		sz;

	addr = __arrow_fetch_varlena32(rb_chunk, buffers, index, &sz);
	if (!addr)
		return false;
	return __print_arrow_binary_common(addr, sz, quote);
}

static bool
print_arrow_large_binary(ARROW_PRINT_DATUM_ARGS)
{
	const char *addr;
	size_t		sz;

	addr = __arrow_fetch_varlena64(rb_chunk, buffers, index, &sz);
	if (!addr)
		return false;
	return __print_arrow_binary_common(addr, sz, quote);
}

static bool
print_arrow_bool(ARROW_PRINT_DATUM_ARGS)
{
	int64_t		k = (index >> 3);
	int32_t		mask = (1 << (index & 7));
	const char *bitmap = rb_chunk + buffers[1].offset;

	if (k >= buffers[1].length)
		return false;

	if ((bitmap[k] & mask) != 0)
		fprintf(output_filp, "true");
	else
		fprintf(output_filp, "false");
	return true;
}

static bool
print_arrow_decimal128(ARROW_PRINT_DATUM_ARGS)
{
	int32_t		scale = column->arrow_type.Decimal.scale;
	bool		negative = false;
	size_t		len = 200 + (scale < 0 ? -scale : 0);
	char	   *buf = alloca(len);
	char	   *pos = buf + len;
	const char *src;
	int128_t	datum;

	if (sizeof(int128_t) * (index+1) > buffers[1].length)
		return false;
	src = rb_chunk + buffers[1].offset + sizeof(int128_t) * index;
	memcpy(&datum, src, sizeof(int128_t));

	/* zero handling */
	if (datum == 0)
	{
		fputc('0', output_filp);
		if (scale > 0)
		{
			fputc('.', output_filp);
			while (scale-- > 0)
				fputc('0', output_filp);
		}
		return true;
	}
	/* negative handling */
	if (datum < 0)
	{
		datum = -datum;
		negative = true;
	}

	*--pos = '\0';
	while (scale < 0)
	{
		*--pos = '0';
		scale++;
	}
	assert(scale >= 0);

	while (datum != 0)
	{
		int		dig = datum % 10;

		*--pos = ('0' + dig);
		datum /= 10;
		if (scale > 0)
		{
			if (scale == 1)
			{
				*--pos = '.';
				if (datum == 0)
					*--pos = '0';
			}
			scale--;
		}
	}

	if (scale > 0)
	{
		while (scale-- > 0)
			*--pos = '0';
		*--pos = '.';
		*--pos = '0';
	}

	if (negative)
		*--pos = '-';
	fprintf(output_filp, "%s", pos);
	return true;
}

static bool
print_arrow_date_day(ARROW_PRINT_DATUM_ARGS)
{
	time_t		t;
	struct tm	tm;
	ARROW_PRINT_DATUM_SETUP_INLINE(uint32_t);
	/* to seconds from the epoch */
	t = (time_t)datum * 86400LL;
	gmtime_r(&t, &tm);
	fprintf(output_filp, "%s%04d-%02d-%02d%s",
			quote,
			tm.tm_year + 1900,
			tm.tm_mon + 1,
			tm.tm_mday,
			quote);
	return true;
}

static bool
print_arrow_date_ms(ARROW_PRINT_DATUM_ARGS)
{
	time_t		t;
	struct tm	tm;
	uint32_t	msec;
	ARROW_PRINT_DATUM_SETUP_INLINE(uint64_t);

	msec = datum % 1000;
	t = datum / 1000;
	gmtime_r(&t, &tm);
	fprintf(output_filp, "%s%04d-%02d-%02d %02d:%02d:%02d.%03d%s",
			quote,
			tm.tm_year + 1900,
			tm.tm_mon + 1,
			tm.tm_mday,
			tm.tm_hour,
			tm.tm_min,
			tm.tm_sec,
			msec,
			quote);
	return true;
}

static bool
print_arrow_time_sec(ARROW_PRINT_DATUM_ARGS)
{
	uint32_t	min, sec;
	ARROW_PRINT_DATUM_SETUP_INLINE(uint32_t);

	sec = datum % 60;
	datum /= 60;
	min = datum % 60;
	datum /= 60;
	fprintf(output_filp, "%s%02u:%02u:%02u%s",
			quote,
			datum, min, sec,
			quote);
	return true;
}

static bool
print_arrow_time_ms(ARROW_PRINT_DATUM_ARGS)
{
	uint32_t	min, sec, ms;
	ARROW_PRINT_DATUM_SETUP_INLINE(uint32_t);

	ms = datum % 1000;
	datum /= 1000;
	sec = datum % 60;
	datum /= 60;
	min = datum % 60;
	datum /= 60;
	fprintf(output_filp, "%s%02u:%02u:%02u.%03u%s",
			quote,
			datum, min, sec, ms,
			quote);
	return true;
}

static bool
print_arrow_time_us(ARROW_PRINT_DATUM_ARGS)
{
	uint32_t	min, sec, us;
	ARROW_PRINT_DATUM_SETUP_INLINE(uint64_t);

	us = datum % 1000000;
	datum /= 1000000;
	sec = datum % 60;
	datum /= 60;
	min = datum % 60;
	datum /= 60;
	fprintf(output_filp, "%s%02d:%02d:%02d.%06d%s",
			quote,
			(uint32_t)datum, min, sec, us,
			quote);
	return true;
}

static bool
print_arrow_time_ns(ARROW_PRINT_DATUM_ARGS)
{
	uint32_t	min, sec, ns;
	ARROW_PRINT_DATUM_SETUP_INLINE(uint64_t);

	ns = datum % 1000000000;
	datum /= 1000000000;
	sec = datum % 60;
	datum /= 60;
	min = datum % 60;
	datum /= 60;
	fprintf(output_filp, "%s%02d:%02d:%02d.%09d%s",
			quote,
			(uint32_t)datum, min, sec, ns,
			quote);
	return true;
}

static bool
__assign_timestamp_timezone(ArrowTypeTimestamp *timestamp)
{
	static const char  *current_tz_name = NULL;

	if (!timestamp->timezone)
		return false;
	if (!current_tz_name || strcmp(current_tz_name, timestamp->timezone) != 0)
	{
		if (setenv("TZ", timestamp->timezone, 1) != 0)
			Elog("failed on setenv('TZ'): %m");
        current_tz_name = timestamp->timezone;
	}
	return true;
}

static bool
print_arrow_timestamp_sec(ARROW_PRINT_DATUM_ARGS)
{
	time_t		t;
	struct tm	tm;
	ARROW_PRINT_DATUM_SETUP_INLINE(uint64_t);

	t = (time_t)datum;
	if (!__assign_timestamp_timezone(&column->arrow_type.Timestamp))
		gmtime_r(&t, &tm);
	else
		localtime_r(&t, &tm);
	fprintf(output_filp,
			"%s%04d-%02d-%02d %02d:%02d:%02d%s",
			quote,
			tm.tm_year + 1900,
			tm.tm_mon + 1,
			tm.tm_mday,
			tm.tm_hour,
			tm.tm_min,
			tm.tm_sec,
			quote);
	return true;
}

static bool
print_arrow_timestamp_ms(ARROW_PRINT_DATUM_ARGS)
{
	time_t		t;
	struct tm	tm;
	uint32_t	ms;
	ARROW_PRINT_DATUM_SETUP_INLINE(uint64_t);

	ms = datum % 1000;
	datum /= 1000;
	t = (time_t)datum;
	if (!__assign_timestamp_timezone(&column->arrow_type.Timestamp))
		gmtime_r(&t, &tm);
	else
		localtime_r(&t, &tm);
	fprintf(output_filp,
			"%s%04d-%02d-%02d %02d:%02d:%02d.%03u%s",
			quote,
			tm.tm_year + 1900,
			tm.tm_mon + 1,
			tm.tm_mday,
			tm.tm_hour,
			tm.tm_min,
			tm.tm_sec,
			ms,
			quote);
	return true;
}

static bool
print_arrow_timestamp_us(ARROW_PRINT_DATUM_ARGS)
{
	time_t		t;
	struct tm	tm;
	uint32_t	us;
	ARROW_PRINT_DATUM_SETUP_INLINE(uint64_t);

	us = datum % 1000000;
	datum /= 1000000;
	t = (time_t)datum;
	if (!__assign_timestamp_timezone(&column->arrow_type.Timestamp))
		gmtime_r(&t, &tm);
	else
		localtime_r(&t, &tm);
	fprintf(output_filp,
			"%s%04d-%02d-%02d %02d:%02d:%02d.%06u%s",
			quote,
			tm.tm_year + 1900,
			tm.tm_mon + 1,
			tm.tm_mday,
			tm.tm_hour,
			tm.tm_min,
			tm.tm_sec,
			us,
			quote);
	return true;
}

static bool
print_arrow_timestamp_ns(ARROW_PRINT_DATUM_ARGS)
{
	time_t		t;
	struct tm	tm;
	uint32_t	ns;
	ARROW_PRINT_DATUM_SETUP_INLINE(uint64_t);

	ns = datum % 1000000000;
	datum /= 1000000000;
	t = (time_t)datum;
	if (!__assign_timestamp_timezone(&column->arrow_type.Timestamp))
		gmtime_r(&t, &tm);
	else
		localtime_r(&t, &tm);
	fprintf(output_filp,
			"%s%04d-%02d-%02d %02d:%02d:%02d.%09u%s",
			quote,
			tm.tm_year + 1900,
			tm.tm_mon + 1,
			tm.tm_mday,
			tm.tm_hour,
			tm.tm_min,
			tm.tm_sec,
			ns,
			quote);
	return true;
}

static bool
print_arrow_interval_ym(ARROW_PRINT_DATUM_ARGS)
{
	int		year, mon;
	bool	negative = false;
	ARROW_PRINT_DATUM_SETUP_INLINE(int32_t);

	if (datum < 0)
	{
		datum = -datum;
		negative = true;
	}
	year = datum / 12;
	mon = datum % 12;
	fprintf(output_filp, "%s", quote);
	if (year != 0 && mon != 0)
		fprintf(output_filp, "%d %s %d %s",
				year, (year > 1 ? "years" : "year"),
				mon, (mon > 1 ? "months" : "month"));
	else if (year != 0)
		fprintf(output_filp, "%d %s",
				year, (year > 1 ? "years" : "year"));
	else
		fprintf(output_filp, "%d %s",
				mon, (mon > 1 ? "months" : "month"));
	if (negative)
		fprintf(output_filp, " ago");
	fprintf(output_filp, "%s", quote);
	return true;
}

static bool
print_arrow_interval_dt(ARROW_PRINT_DATUM_ARGS)
{
	int32_t		days;
	int			hour, min, sec, msec;
	bool		negative = false;
	ARROW_PRINT_DATUM_SETUP_INLINE(uint64_t);

	days = (int32_t)(datum & 0xffffffffU);
	if (days < 0)
	{
		negative = true;
		days = -days;
	}
	datum = (datum >> 32);
	msec = datum % 1000;
	datum /= 1000;
	sec = datum % 60;
	datum /= 60;
	min = datum % 60;
	datum /= 60;
	hour = datum;
	fprintf(output_filp, "%s", quote);
	if (days != 0)
	{
		if (hour != 0 || min != 0 || sec != 0)
			fprintf(output_filp, "%d %s %02d:%02d:%02d",
					days, (days > 1 ? "days" : "day"),
					hour, min, sec);
	}
	else
	{
		fprintf(output_filp, "%02d:%02d:%02d",
				hour, min, sec);
	}
	if (msec != 0)
		fprintf(output_filp, ".%03d", msec);
	if (negative)
		fprintf(output_filp, " ago");
	fprintf(output_filp, "%s", quote);
	return true;
}

static bool
print_arrow_fixedsizebinary(ARROW_PRINT_DATUM_ARGS)
{
	int32_t		i, width = column->arrow_type.FixedSizeBinary.byteWidth;
	ARROW_PRINT_DATUM_SETUP_FIXEDSIZEBINARY(width);

	fprintf(output_filp, "%s\\x", quote);
	for (i=0; i < width; i++)
	{
		static const char *hextbl = "0123456789abcdef";
		int		c = addr[i];

		fputc(hextbl[(c >> 4) & 0x0f], output_filp);
		fputc(hextbl[(c & 0x0f)], output_filp);
	}
	fprintf(output_filp, "%s", quote);
	return true;
}

static bool
print_arrow_macaddr(ARROW_PRINT_DATUM_ARGS)
{
	int32_t		width = column->arrow_type.FixedSizeBinary.byteWidth;
	ARROW_PRINT_DATUM_SETUP_FIXEDSIZEBINARY(width);
	assert(width == 6);
	fprintf(output_filp,
			"%s%02x:%02x:%02x:%02x:%02x:%02x%s",
			quote,
			(unsigned char)addr[0],
			(unsigned char)addr[1],
			(unsigned char)addr[2],
			(unsigned char)addr[3],
			(unsigned char)addr[4],
			(unsigned char)addr[5],
			quote);
	return true;
}

static bool
print_arrow_inet4(ARROW_PRINT_DATUM_ARGS)
{
	int32_t		width = column->arrow_type.FixedSizeBinary.byteWidth;
	ARROW_PRINT_DATUM_SETUP_FIXEDSIZEBINARY(width);
    assert(width == 4);
	fprintf(output_filp,
			"%s%u.%u.%u.%u%s",
			quote,
			(unsigned char)addr[0],
			(unsigned char)addr[1],
			(unsigned char)addr[2],
			(unsigned char)addr[3],
			quote);
	return true;
}

static bool
print_arrow_inet6(ARROW_PRINT_DATUM_ARGS)
{
	int32_t		width = column->arrow_type.FixedSizeBinary.byteWidth;
	uint16_t	words[8];
	int			zero_base = -1;
	int			zero_len = -1;
	int			i, j;
	ARROW_PRINT_DATUM_SETUP_FIXEDSIZEBINARY(width);
	assert(width == 16);

	/* copy IPv6 address */
	for (i=0; i < 8; i++)
		words[i] = ntohs(((uint16_t *)addr)[i]);
	/* lookup the longest run of 0x00 */
	for (i=0; i < 8; i++)
	{
		if (words[i] == 0)
		{
			int		count = 1;

			for (j=i+1; j < 8; j++)
			{
				if (words[j] != 0)
					break;
				count++;
			}
			if (count > 1 && count > zero_len)
			{
				zero_base = i;
				zero_len = count;
			}
		}
	}

	/* print out IPv6 */
	fprintf(output_filp, "%s", quote);
	for (i=0; i < 8; i++)
	{
		if (zero_base >= 0 &&
			i >= zero_base &&
			i <  zero_base + zero_len)
		{
			if (i == zero_base)
				fputc(':', output_filp);
			continue;
		}
		if (i > 0)
			fputc(':', output_filp);
		/* Is this address an encapsulated IPv4? */
		if (i == 6 && zero_base == 0 && ((zero_len == 6) ||
										 (zero_len == 7 && words[7] != 0x0001) ||
										 (zero_len == 5 && words[5] == 0xffff)))
		{
			fprintf(output_filp, "%u.%u.%u.%u",
					(unsigned char)addr[12],
					(unsigned char)addr[13],
					(unsigned char)addr[14],
					(unsigned char)addr[15]);
			break;
		}
		fprintf(output_filp, "%x", words[i]);
	}
	if (zero_base >= 0 && zero_base + zero_len == 8)
		fputc(':', output_filp);
	fprintf(output_filp, "%s", quote);
	return true;
}

static bool
print_arrow_list(ARROW_PRINT_DATUM_ARGS)
{
	char			saved_context = current_context;
	const uint32_t *values;
	uint32_t		i, head, tail;
	arrowColumn	   *child;
	ArrowBuffer	   *__buffers;

	if (sizeof(uint32_t) * (index+2) > buffers[1].length)
		return false;

	values = (const uint32_t *)(rb_chunk + buffers[1].offset);
	head = values[index];
	tail = values[index+1];
	if (head > tail)
		return false;

	child = &column->children[0];
	__buffers = buffers + (child->buffer_index -
						   column->buffer_index);
	fprintf(output_filp, "%s[", quote);
	current_context = 'e';
	for (i=head; i < tail; i++)
	{
		if (i > head)
			fprintf(output_filp, ",");
		printArrowDatum(child, __buffers, rb_chunk, i, "");
	}
	current_context = saved_context;
	fprintf(output_filp, "%s]", quote);
	return true;
}

static bool
print_arrow_struct(ARROW_PRINT_DATUM_ARGS)
{
	char		saved_context = current_context;
	int			j;

	fprintf(output_filp, "%s(", quote);
	current_context = 'e';
	for (j=0; j < column->num_children; j++)
	{
		arrowColumn *child = &column->children[j];
		ArrowBuffer *__buffers = buffers + (child->buffer_index -
											column->buffer_index);
		if (j > 0)
			fprintf(output_filp, "%c", csv_delimiter);
		printArrowDatum(child, __buffers, rb_chunk, index, "");
	}
	current_context = saved_context;
	fprintf(output_filp, "%s)", quote);
	return true;
}

static int
setupArrowColumn(arrowColumn *column, ArrowField *field, int buffer_index)
{
	const char *pg_type = NULL;
	char	   *__sqltype;
	int			buffer_count = 0;
	int			i, j;

	memcpy(&column->arrow_type, &field->type, sizeof(ArrowType));
	column->colname = pstrdup(field->name);
	column->buffer_index = buffer_index;
	column->nullable = field->nullable;
	/* fetch "pg_type" custom-metadata */
	for (i=0; i < field->_num_custom_metadata; i++)
	{
		ArrowKeyValue *kv = &field->custom_metadata[i];

		if (strcmp(kv->key, "pg_type") == 0)
		{
			if (strcmp(kv->value, "pg_catalog.macaddr") == 0 ||
				strcmp(kv->value, "macaddr") == 0)
			{
				if (field->type.node.tag != ArrowNodeTag__FixedSizeBinary)
					Elog("custom-metadata pg_type=%s is only supported with Arrow::FixedSizeBinary", kv->value);
				pg_type = "macaddr";
			}
			else if (strcmp(kv->value, "pg_catalog.inet") == 0 ||
					 strcmp(kv->value, "inet") == 0)
			{
				if (field->type.node.tag != ArrowNodeTag__FixedSizeBinary)
					Elog("custom-metadata pg_type=%s is only supported with Arrow::FixedSizeBinary", kv->value);
				pg_type = "inet";
			}
			else
				Elog("unknown 'pg_type' custom-metadata [%s]", kv->value);
		}
	}

	switch (field->type.node.tag)
	{
		case ArrowNodeTag__Int:
			switch (field->type.Int.bitWidth)
			{
				case 8:
					column->sqltype = "int1";
					column->print_datum = (field->type.Int.is_signed
										   ? print_arrow_int8
										   : print_arrow_uint8);
					break;
				case 16:
					column->sqltype = "int2";
					column->print_datum = (field->type.Int.is_signed
										   ? print_arrow_int16
										   : print_arrow_uint16);
					break;
				case 32:
					column->sqltype = "int4";
					column->print_datum = (field->type.Int.is_signed
										   ? print_arrow_int32
										   : print_arrow_uint32);
					break;
				case 64:
					column->sqltype = "int8";
					column->print_datum = (field->type.Int.is_signed
										   ? print_arrow_int64
										   : print_arrow_uint64);
					break;
				default:
					Elog("unsupported Arrow::%s bitWidwh(%d)",
						 field->type.Int.is_signed ? "Int" : "Uint",
						 field->type.Int.bitWidth);
			}
			buffer_count = 2;
			break;

		case ArrowNodeTag__FloatingPoint:
			switch (field->type.FloatingPoint.precision)
			{
				case ArrowPrecision__Half:
					column->sqltype = "float2";
					column->print_datum = print_arrow_float2;
					break;
				case ArrowPrecision__Single:
					column->sqltype = "float4";
					column->print_datum = print_arrow_float4;
					break;
				case ArrowPrecision__Double:
					column->sqltype = "float8";
					column->print_datum = print_arrow_float8;
					break;
				default:
					Elog("unsupported Arrow::FloatingPoint precision (%d)",
						 (int)field->type.FloatingPoint.precision);
			}
			buffer_count = 2;
			break;
			
		case ArrowNodeTag__Utf8:
		case ArrowNodeTag__LargeUtf8:
			column->sqltype = "text";
			if (field->type.node.tag == ArrowNodeTag__Utf8)
				column->print_datum = print_arrow_utf8;
			else
				column->print_datum = print_arrow_large_utf8;
			buffer_count = 3;
			break;

		case ArrowNodeTag__Binary:
		case ArrowNodeTag__LargeBinary:
			column->sqltype = "bytea";
			if (field->type.node.tag == ArrowNodeTag__Binary)
				column->print_datum = print_arrow_binary;
			else
				column->print_datum = print_arrow_large_binary;
			buffer_count = 3;
			break;

		case ArrowNodeTag__Bool:
			column->sqltype = "bool";
			column->print_datum = print_arrow_bool;
			buffer_count = 2;
			break;

		case ArrowNodeTag__Decimal:
			switch (field->type.Decimal.bitWidth)
			{
				case 128:
					column->sqltype = "numeric";
					column->print_datum = print_arrow_decimal128;
					break;
				default:
					Elog("unsupported Arrow::Decimal bitWidth (%d)",
						 field->type.Decimal.bitWidth);
			}
			buffer_count = 2;
			break;

		case ArrowNodeTag__Date:
			column->sqltype = "date";
			switch (field->type.Date.unit)
			{
				case ArrowDateUnit__Day:
					column->print_datum = print_arrow_date_day;
					break;
				case ArrowDateUnit__MilliSecond:
					column->print_datum = print_arrow_date_ms;
					break;
				default:
					Elog("unsupported Arrow::Date unit (%d)",
						 (int)field->type.Date.unit);
			}
			buffer_count = 2;
			break;

		case ArrowNodeTag__Time:
			column->sqltype = "time";
			switch (field->type.Time.unit)
			{
				case ArrowTimeUnit__Second:
					column->print_datum = print_arrow_time_sec;
					break;
				case ArrowTimeUnit__MilliSecond:
					column->print_datum = print_arrow_time_ms;
					break;
				case ArrowTimeUnit__MicroSecond:
					column->print_datum = print_arrow_time_us;
					break;
				case ArrowTimeUnit__NanoSecond:
					column->print_datum = print_arrow_time_ns;
					break;
				default:
					Elog("unsupported Arrow::Time unit (%d)",
						 (int)field->type.Time.unit);
			}
			buffer_count = 2;
			break;

		case ArrowNodeTag__Timestamp:
			column->sqltype = "timestamp";
			switch (field->type.Timestamp.unit)
			{
				case ArrowTimeUnit__Second:
					column->print_datum = print_arrow_timestamp_sec;
					break;
				case ArrowTimeUnit__MilliSecond:
					column->print_datum = print_arrow_timestamp_ms;
					break;
				case ArrowTimeUnit__MicroSecond:
					column->print_datum = print_arrow_timestamp_us;
					break;
				case ArrowTimeUnit__NanoSecond:
					column->print_datum = print_arrow_timestamp_ns;
					break;
				default:
					Elog("unsupported Arrow::Timestamp unit (%d)",
						 (int)field->type.Timestamp.unit);
			}
			buffer_count = 2;
			break;

		case ArrowNodeTag__Interval:
			column->sqltype = "interval";
			switch (field->type.Interval.unit)
			{
				case ArrowIntervalUnit__Year_Month:
					column->print_datum = print_arrow_interval_ym;
					break;
				case ArrowIntervalUnit__Day_Time:
					column->print_datum = print_arrow_interval_dt;
					break;
				default:
					Elog("unsupported Arrow::Interval unit (%d)",
						 (int)field->type.Interval.unit);
			}
			buffer_count = 2;
			break;

		case ArrowNodeTag__List:
			if (field->_num_children != 1)
				Elog("wrong metadata - Arrow::List must have a subtype");
			buffer_count = 2;
			column->print_datum = print_arrow_list;
			column->children = palloc0(sizeof(arrowColumn));
			column->num_children = 1;
			buffer_count += setupArrowColumn(column->children,
											 field->children,
											 buffer_index + 2);
			__sqltype = palloc(strlen(column->children->sqltype) + 3);
			sprintf(__sqltype, "%s[]", column->children->sqltype);
			column->sqltype = __sqltype;
			break;
			
		case ArrowNodeTag__Struct:
			if (field->_num_children == 0)
				Elog("wrong metadata - Arrow::Struct must have subtypes");
			column->sqltype = "__dummy__";		/* never referenced */
			column->print_datum = print_arrow_struct;
			column->num_children = field->_num_children;
			column->children = palloc0(sizeof(arrowColumn) * column->num_children);
			buffer_count = 1;
			for (j=0; j < column->num_children; j++)
			{
				buffer_count += setupArrowColumn(&column->children[j],
												 &field->children[j],
												 buffer_index + buffer_count);
			}

			if (create_table_name)
			{
				__sqltype = palloc(strlen(create_table_name) +
								   strlen(field->name) + 32);
				sprintf(__sqltype, "%s_%s_t",
						create_table_name,
						field->name);
				column->sqltype = __sqltype;
			}
			break;

		case ArrowNodeTag__FixedSizeBinary:
			if (!pg_type)
			{
				__sqltype = palloc(48);
				sprintf(__sqltype, "char(%d)", field->type.FixedSizeBinary.byteWidth);
				column->sqltype = __sqltype;
				column->print_datum = print_arrow_fixedsizebinary;
			}
			else if (strcmp(pg_type, "macaddr") == 0)
			{
				if (field->type.FixedSizeBinary.byteWidth != 6)
					Elog("'macaddr' must be FixedSizeBinary with byteWidth=6");
				column->sqltype = "macaddr";
				column->print_datum = print_arrow_macaddr;
			}
			else if (strcmp(pg_type, "inet") == 0)
			{
				column->sqltype = "inet";
				if (field->type.FixedSizeBinary.byteWidth == 4)
					column->print_datum = print_arrow_inet4;
				else if (field->type.FixedSizeBinary.byteWidth == 16)
					column->print_datum = print_arrow_inet6;
				else
					Elog("'inet' must be FixedSizeBinary with byteWidth=4 or 16");
			}
			else
			{
				Elog("unknown custom pg_type [%s]", pg_type);
			}
			buffer_count = 2;
			break;
			
		default:
			Elog("Arrow::%s is not a supported type",
				 field->type.node.tagName);
	}
	column->buffer_index = buffer_index;
	column->buffer_count = buffer_count;

	return buffer_count;
}

/*
 * printCreateTable - dump the schema definition
 */
static void
printCreateType(arrowColumn *col)
{
	int		j;

	for (j=0; j < col->num_children; j++)
	{
		arrowColumn *child = &col->children[j];

		if (child->arrow_type.node.tag == ArrowNodeTag__Struct)
			printCreateType(child);
	}

	fprintf(output_filp,
			"CREATE TYPE %s AS (\n",
			quote_ident(col->sqltype));
	for (j=0; j < col->num_children; j++)
	{
		arrowColumn *child = &col->children[j];

		if (j > 0)
			fprintf(output_filp, ",\n");
		fprintf(output_filp, "  %s %s",
				quote_ident(child->colname),
				quote_ident(child->sqltype));
	}
	fprintf(output_filp,
			"\n);\n");
}

static void
printCreateTable(void)
{
	int		j;

	/* type declarations */
	for (j=0; j < arrow_num_columns; j++)
	{
		arrowColumn *col = &arrow_columns[j];

		if (col->arrow_type.node.tag == ArrowNodeTag__Struct)
			printCreateType(col);
	}
	fprintf(output_filp,
			"CREATE TABLE %s (\n",
			quote_ident(create_table_name));
	for (j=0; j < arrow_num_columns; j++)
	{
		arrowColumn *col = &arrow_columns[j];

		if (j > 0)
			fprintf(output_filp, ",\n");
		fprintf(output_filp,
				"    %s %s",
				quote_ident(col->colname),
				quote_ident(col->sqltype));
	}
	fprintf(output_filp, "\n)");
	if (partition_name)
		fprintf(output_filp,
				"\n  PARTITION OF %s",
				quote_ident(partition_name));
	if (tablespace_name)
		fprintf(output_filp,
				"\n  TABLESPACE %s",
				quote_ident(tablespace_name));
	fprintf(output_filp, ";\n");
}

static void
printRecordBatch(ArrowRecordBatch *rbatch, const char *rb_chunk)
{
	int64_t		i, j;

	/* consider --offset */
	if (num_skip_rows < 0)
		i = 0;
	else
	{
		i = num_skip_rows;
		if (num_skip_rows > rbatch->length)
			num_skip_rows -= rbatch->length;
		else
			num_skip_rows = 0;
	}

	while (i < rbatch->length && num_dump_rows != 0)
	{
		for (j=0; j < arrow_num_columns; j++)
		{
			arrowColumn	   *column = &arrow_columns[j];
			ArrowBuffer	   *buffers;

			if (j > 0)
				fprintf(output_filp, "%c", csv_delimiter);
			if (j >= rbatch->_num_nodes)
				printNullDatum();
			else if (i >= rbatch->nodes[j].length)
				printNullDatum();
			else
			{
				assert(column->buffer_index +
					   column->buffer_count <= rbatch->_num_buffers);
				buffers = rbatch->buffers + column->buffer_index;
				printArrowDatum(column, buffers, rb_chunk, i, "\"");
			}
		}
		fprintf(output_filp, "\r\n");
		i++;
		if (num_dump_rows > 0)
			num_dump_rows--;
	}
}

static void
dumpArrowFile(ArrowFileInfo *af_info, int fdesc)
{
	static long	__PAGE_SIZE = -1;
	size_t		file_sz = af_info->stat_buf.st_size;
	size_t		mmap_sz;
	char	   *mmap_head;
	int			i;

	if (__PAGE_SIZE < 0)
		__PAGE_SIZE = sysconf(_SC_PAGESIZE);
	mmap_sz = (file_sz + __PAGE_SIZE - 1) & ~(__PAGE_SIZE - 1);
	mmap_head = mmap(NULL, mmap_sz, PROT_READ, MAP_SHARED, fdesc, 0);
	if (mmap_head == MAP_FAILED)
		Elog("failed on mmap: %m");
	for (i=0; i < af_info->footer._num_recordBatches; i++)
	{
		ArrowBlock *block = &af_info->footer.recordBatches[i];
		char	   *rb_chunk = (mmap_head + block->offset + block->metaDataLength);
		ArrowRecordBatch *rbatch = &af_info->recordBatches[i].body.recordBatch;

		printRecordBatch(rbatch, rb_chunk);
	}
	munmap(mmap_head, mmap_sz);
}

int
main(int argc, char * const argv[])
{
	static struct option long_options[] = {
		{"output",       required_argument, NULL, 'o'},
		{"header",       no_argument,       NULL, 1002},
		{"offset",       required_argument, NULL, 1004},
		{"limit",        required_argument, NULL, 1005},
		/* CREATE TABLE & COPY FROM */
		{"create-table", required_argument, NULL, 1200},
		{"tablespace",   required_argument, NULL, 1201},
		{"partition-of", required_argument, NULL, 1202},
		/* Other options */
		{"help",         no_argument,       NULL, 9999},
		{NULL, 0, NULL, 0},
	};
	int		i, j, c;

	while ((c = getopt_long(argc, argv, "o:h", long_options, NULL)) >= 0)
	{
		switch (c)
		{
			case 'o':	/* --output */
				if (output_filename)
					Elog("-o|--output was specified twice");
				output_filename = optarg;
				break;
			case 1002:
				if (print_header)
					Elog("--header was specified twice");
				print_header = true;
				break;
			case 1004:	/* --offset */
				if (num_skip_rows >= 0)
					Elog("--offset was specified twice");
				num_skip_rows = atol(optarg);
				if (num_skip_rows < 0)
					Elog("--offset=%s is not a numeric value", optarg);
				break;
			case 1005:	/* --limit */
				if (num_dump_rows >= 0)
					Elog("--limit was specified twice");
				num_dump_rows = atol(optarg);
				if (num_dump_rows < 0)
					Elog("--limit=%s is not a numeric value", optarg);
				break;
			case 1200:	/* --create-table */
				if (create_table_name)
					Elog("--create-table was specified twice");
				create_table_name = optarg;
				break;
			case 1201:	/* --tablespace */
				if (tablespace_name)
					Elog("--tablespace was specified twice");
				tablespace_name = optarg;
				break;
			case 1202:	/* --partition-of */
				if (partition_name)
					Elog("--partition-of was specified twice");
				partition_name = optarg;
				break;
			case 'h':	/* --help */
			default:
				usage();
				break;
		}
	}
	/* sanity check */
	if (tablespace_name && !create_table_name)
		Elog("--tablespace must be used with --create-table");
	if (partition_name && !create_table_name)
		Elog("--partition-of must be used with --create-table");
	
	/* arrow files */
	if (optind >= argc)
		Elog("no input arrow files given");
	arrow_num_files = argc - optind;
	arrow_files = palloc(sizeof(ArrowFileInfo) * arrow_num_files);
	arrow_fdescs = palloc(sizeof(int) * arrow_num_files);
	for (i=0; i < arrow_num_files; i++)
	{
		const char *filename = argv[optind + i];
		int			fdesc;

		fdesc = open(filename, O_RDONLY);
		if (fdesc < 0)
			Elog("failed on open('%s'): %m", filename);
		readArrowFileDesc(fdesc, &arrow_files[i]);
		arrow_files[i].filename = filename;
		arrow_fdescs[i] = fdesc;

		if (i == 0)
		{
			/* setup arrowColumn array */
			ArrowFileInfo  *af_info = &arrow_files[0];
			ArrowSchema	   *schema = &af_info->footer.schema;
			int				bindex = 0;

			arrow_num_columns = schema->_num_fields;
			arrow_columns = palloc0(sizeof(arrowColumn) * arrow_num_columns);
			for (j=0; j < schema->_num_fields; j++)
			{
				bindex += setupArrowColumn(&arrow_columns[j],
										   &schema->fields[j],
										   bindex);
			}
		}
		else
		{
			/* check schema compatibility, if multiple source files */
			ArrowFileInfo  *a = &arrow_files[0];
			ArrowFileInfo  *b = &arrow_files[i];

			if (a->footer.schema._num_fields != b->footer.schema._num_fields)
				Elog("Arrow file '%s' and '%s' has different number of the fields",
					 a->filename,
					 b->filename);
			for (j=0; j < a->footer.schema._num_fields; j++)
			{
				if (!arrowFieldTypeIsEqual(&a->footer.schema.fields[j],
										   &b->footer.schema.fields[j]))
					Elog("Arrow file '%s' and '%s' has incompatible column",
						 a->filename,
						 b->filename);

				if (strcmp(a->footer.schema.fields[j].name,
						   b->footer.schema.fields[j].name) != 0)
					fprintf(stderr, "warning: column name '%s' in '%s' is not identical '%s' of '%s'\n",
							a->footer.schema.fields[j].name,
							a->filename,
							b->footer.schema.fields[j].name,
							b->filename);
			}
		}
	}
	
	/* open the output file, if necessary */
	if (!output_filename)
		output_filp = stdout;
	else
	{
		output_filp = fopen(output_filename, "w");
		if (!output_filp)
			Elog("unable to open output file '%s': %m", output_filename);
	}

	/* CREATE TABLE / CREATE TYPE, if required */
	if (create_table_name)
	{
		printCreateTable();
		fprintf(output_filp, "COPY %s FROM stdin csv%s;\r\n",
				quote_ident(create_table_name),
				print_header ? " HEADER" : "");
	}
	/* Dump column names */
	if (print_header)
	{
		for (j=0; j < arrow_num_columns; j++)
		{
			const char *colname = arrow_columns[j].colname;
			if (j > 0)
				fprintf(output_filp, "%c", csv_delimiter);
			fprintf(output_filp, "%s", __quote_ident(colname));
		}
		fprintf(output_filp, "\r\n");
	}
	/* Dump Arrpw files */
	for (i=0; i < arrow_num_files; i++)
		dumpArrowFile(&arrow_files[i], arrow_fdescs[i]);
	if (create_table_name && num_dump_rows != 0)
		fprintf(output_filp, "\\.\r\n");
	return 0;
}

/*
 * memory allocation handlers
 */
void *
palloc(size_t sz)
{
    void   *ptr = malloc(sz);

    if (!ptr)
        Elog("out of memory");
    return ptr;
}

void *
palloc0(size_t sz)
{
    void   *ptr = malloc(sz);

    if (!ptr)
        Elog("out of memory");
    memset(ptr, 0, sz);
    return ptr;
}

char *
pstrdup(const char *str)
{
	char   *ptr = strdup(str);

	if (!ptr)
		Elog("out of memory");
	return ptr;
}

void *
repalloc(void *old, size_t sz)
{
	char   *ptr = realloc(old, sz);

	if (!ptr)
		Elog("out of memory");
	return ptr;
}

void
pfree(void *ptr)
{
	free(ptr);
}
