/*
 * arrow_buf.c
 *
 * Routines for local buffer of Apache Arrow
 * ----
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
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
#include "postgres.h"
typedef struct SQLbuffer    StringInfoData;
typedef struct SQLbuffer   *StringInfo;
#include "arrow_ipc.h"

/*
 * Rewind the local arrow buffer of SQLfield
 */
void
sql_field_rewind(SQLfield *column, size_t nitems)
{
	if (nitems > column->nitems)
		return;		/* nothing to do */
	else if (nitems == column->nitems)
	{
		/* special case, nothing to do */
		return;
	}
	else if (nitems == 0)
	{
		/* special case optimization */
		sql_buffer_clear(&column->nullmap);
		sql_buffer_clear(&column->values);
		sql_buffer_clear(&column->extra);
		column->nitems = 0;
		column->nullcount = 0;
		return;
	}
	else if (column->nullcount > 0)
	{
		long		nullcount = 0;
		uint32	   *nullmap = (uint32 *)column->nullmap.data;
		uint32		i, n = nitems / 32;
		uint32		mask = (1UL << (nitems % 32)) - 1;

		for (i=0; i < n; i++)
			nullcount += __builtin_popcount(~nullmap[i]);
		if (mask != 0)
			nullcount += __builtin_popcount(~nullmap[n] & mask);
		column->nullcount = nullcount;
	}
	column->nitems = nitems;
	column->nullmap.usage = (nitems + 7) >> 3;

	switch (column->arrow_type.node.tag)
	{
		case ArrowNodeTag__Int:
			switch (column->arrow_type.Int.bitWidth)
			{
				case 8:
					column->values.usage = sizeof(int8) * nitems;
					break;
				case 16:
					column->values.usage = sizeof(int16) * nitems;
					break;
				case 32:
					column->values.usage = sizeof(int32) * nitems;
					break;
				case 64:
					column->values.usage = sizeof(int64) * nitems;
					break;
				default:
					Elog("unknown width of Arrow::Int type: %d",
						 column->arrow_type.Int.bitWidth);
					break;
			}
			break;
		case ArrowNodeTag__FloatingPoint:
			switch (column->arrow_type.FloatingPoint.precision)
			{
				case ArrowPrecision__Half:
					column->values.usage = sizeof(int16) * nitems;
					break;
				case ArrowPrecision__Single:
					column->values.usage = sizeof(int32) * nitems;
					break;
				case ArrowPrecision__Double:
					column->values.usage = sizeof(int64) * nitems;
					break;
				default:
					Elog("unknown precision of Arrow::FloatingPoint type: %d",
						 column->arrow_type.FloatingPoint.precision);
					break;
			}
			break;
		case ArrowNodeTag__Utf8:
		case ArrowNodeTag__Binary:
			column->values.usage = sizeof(int32) * (nitems + 1);
			column->extra.usage = ((uint32 *)column->values.data)[nitems];
			break;
		case ArrowNodeTag__Bool:
			column->values.usage = (nitems + 7) >> 3;
			break;
		case ArrowNodeTag__Decimal:
			column->values.usage = sizeof(int128) * nitems;
			break;
		case ArrowNodeTag__Date:
			switch (column->arrow_type.Date.unit)
			{
				case ArrowDateUnit__Day:
					column->values.usage = sizeof(int32) * nitems;
					break;
				case ArrowDateUnit__MilliSecond:
					column->values.usage = sizeof(int64) * nitems;
					break;
				default:
					Elog("unknown unit of Arrow::Date type; %u",
						 column->arrow_type.Date.unit);
					break;
			}
			break;
		case ArrowNodeTag__Time:
			switch (column->arrow_type.Time.unit)
			{
				case ArrowTimeUnit__Second:
				case ArrowTimeUnit__MilliSecond:
					column->values.usage = sizeof(int32) * nitems;
					break;
				case ArrowTimeUnit__MicroSecond:
				case ArrowTimeUnit__NanoSecond:
					column->values.usage = sizeof(int64) * nitems;
					break;
				default:
					Elog("unknown unit of Arrow::Time type; %u",
						 column->arrow_type.Time.unit);
			}
			break;
		case ArrowNodeTag__Timestamp:
			column->values.usage = sizeof(int64) * nitems;
			break;
		case ArrowNodeTag__Interval:
			switch (column->arrow_type.Interval.unit)
			{
				case ArrowIntervalUnit__Year_Month:
					column->values.usage = sizeof(int32) * nitems;
					break;
				case ArrowIntervalUnit__Day_Time:
					column->values.usage = sizeof(int64) * nitems;
					break;
				default:
					Elog("unknown unit of Arrow::Interval type; %u",
						 column->arrow_type.Interval.unit);
			}
			break;
		case ArrowNodeTag__List:
			column->values.usage = sizeof(int32) * (nitems + 1);
			sql_field_rewind(column->element, nitems);
			break;
		case ArrowNodeTag__Struct:
			{
				int		j;

				for (j=0; j < column->nfields; j++)
					sql_field_rewind(&column->subfields[j], nitems);
			}
			break;
		case ArrowNodeTag__FixedSizeBinary:
			{
				int32	unitsz = column->arrow_type.FixedSizeBinary.byteWidth;

				column->values.usage = unitsz * nitems;
			}
			break;
		default:
			Elog("unexpected ArrowType node tag: %s (%u)",
				 column->arrow_type.node.tagName,
				 column->arrow_type.node.tag);
	}
}

/*
 * sql_table_rewind
 */
void
sql_table_rewind(SQLtable *table, int nbatches, size_t nitems)
{
	int			i, j;

	if (nbatches > table->nbatches)
		return;		/* nothing to do */
	for (i = table->nbatches; i >= nbatches; i--)
	{
		SQLfield   *columns = SQLtableGetColumns(table, i-1);

		for (j=0; j < table->nfields; j++)
		{
			assert(columns[0].nitems == columns[j].nitems);
			if (i > nbatches)
				sql_field_rewind(&columns[j], 0);
			else
				sql_field_rewind(&columns[j], nitems);
		}
	}
}



/*
 * sql_field_duplicate
 */
static void
sql_field_duplicate(SQLfield *dst, SQLfield *src)
{
	int		j;

	memset(dst, 0, sizeof(SQLfield));
	dst->field_name = pstrdup(src->field_name);
	memcpy(&dst->sql_type, &src->sql_type, sizeof(SQLtype));
	if (src->element)
	{
		dst->element = palloc0(sizeof(SQLfield));
		sql_field_duplicate(dst->element, src->element);
	}
	if (src->subfields)
	{
		assert(src->nfields > 0);
		dst->nfields   = src->nfields;
		dst->subfields = palloc0(sizeof(SQLfield) * src->nfields);
		for (j=0; j < dst->nfields; j++)
			sql_field_duplicate(&dst->subfields[j],
								&src->subfields[j]);
	}
	/* dictionary batch is shared by multiple batches */
	dst->enumdict = src->enumdict;
	copyArrowNode((ArrowNode *)&dst->arrow_type,
				  (ArrowNode *)&src->arrow_type);
	dst->arrow_typename = pstrdup(src->arrow_typename);
	dst->put_value = src->put_value;
}

/*
 * sql_table_expand - expand record batches of SQLtable buffer
 */
SQLtable *
sql_table_expand(SQLtable *table)
{
	SQLfield   *src, *dst;
	int			j, nfields = table->nfields;
	size_t		sz;

	sz = offsetof(SQLtable, columns[nfields * (table->nbatches + 1)]);
	table = repalloc(table, sz);
	src = SQLtableGetColumns(table, table->nbatches - 1);
	dst = SQLtableGetColumns(table, table->nbatches);
	for (j=0; j < nfields; j++)
		sql_field_duplicate(&dst[j], &src[j]);
	table->nbatches++;

	return table;
}

/*
 * SQLbuffer related routines
 */
void
sql_buffer_init(SQLbuffer *buf)
{
	buf->data = NULL;
	buf->usage = 0;
	buf->length = 0;
}

void
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

void
sql_buffer_append(SQLbuffer *buf, const void *src, size_t len)
{
	sql_buffer_expand(buf, buf->usage + len);
	memcpy(buf->data + buf->usage, src, len);
	buf->usage += len;
	assert(buf->usage <= buf->length);
}

void
sql_buffer_append_zero(SQLbuffer *buf, size_t len)
{
	sql_buffer_expand(buf, buf->usage + len);
	memset(buf->data + buf->usage, 0, len);
	buf->usage += len;
	assert(buf->usage <= buf->length);
}

void
sql_buffer_setbit(SQLbuffer *buf, size_t __index)
{
	size_t		index = __index >> 3;
	int			mask  = (1 << (__index & 7));

	sql_buffer_expand(buf, index + 1);
	((uint8 *)buf->data)[index] |= mask;
	buf->usage = Max(buf->usage, index + 1);
}

void
sql_buffer_clrbit(SQLbuffer *buf, size_t __index)
{
	size_t		index = __index >> 3;
	int			mask  = (1 << (__index & 7));

	sql_buffer_expand(buf, index + 1);
	((uint8 *)buf->data)[index] &= ~mask;
	buf->usage = Max(buf->usage, index + 1);
}

void
sql_buffer_clear(SQLbuffer *buf)
{
	buf->usage = 0;
}

void
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

void
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
