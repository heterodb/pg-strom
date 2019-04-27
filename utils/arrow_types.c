/*
 * arrow_types.c
 *
 * intermediation of PostgreSQL types <--> Apache Arrow types
 *
 * Copyright 2018-2019 (C) KaiGai Kohei <kaigai@heterodb.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License. See the LICENSE file.
 */
#include "pg2arrow.h"

/*
 * File operations
 */
static void
__write_buffer_common(int fdesc, const void *buffer, size_t length)
{
	ssize_t		nbytes;
	ssize_t		offset = 0;

	while (offset < length)
	{
		nbytes = write(fdesc, (const char *)buffer + offset, length - offset);
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
			nbytes = write(fdesc, (const char *)&zero + offset, gap - offset);
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

/* ----------------------------------------------------------------
 *
 * put_value handler for each data types (optional)
 *
 * ----------------------------------------------------------------
 */
static void
put_inline_bool_value(SQLattribute *attr,
					  const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;

	if (!addr)
	{
		attr->nullcount++;
		sql_buffer_clrbit(&attr->nullmap, row_index);
		sql_buffer_clrbit(&attr->values,  row_index);
	}
	else
	{
		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_setbit(&attr->values,  row_index);
	}
}

static void
put_inline_8b_value(SQLattribute *attr,
					const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;

	if (!addr)
	{
		attr->nullcount++;
		sql_buffer_clrbit(&attr->nullmap, row_index);
		sql_buffer_append_zero(&attr->values, sizeof(char));
	}
	else
	{
		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, addr, sz);
	}
}

static void
put_inline_16b_value(SQLattribute *attr,
					 const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;
	uint16		value;

	if (!addr)
	{
		attr->nullcount++;
		sql_buffer_clrbit(&attr->nullmap, row_index);
		sql_buffer_append_zero(&attr->values, sizeof(uint16));
	}
	else
	{
		value = ntohs(*((const uint16 *)addr));
		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &value, sz);
	}
}

static void
put_inline_32b_value(SQLattribute *attr,
					 const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;
	uint32		value;

	assert(attr->attlen == sizeof(uint32));
	if (!addr)
	{
		attr->nullcount++;
		sql_buffer_clrbit(&attr->nullmap, row_index);
		sql_buffer_append_zero(&attr->values, sizeof(uint32));
	}
	else
	{
		assert(sz == sizeof(uint32));
		value = ntohl(*((const uint32 *)addr));
		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &value, sz);
	}
}

static void
put_inline_64b_value(SQLattribute *attr,
					 const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;
	uint64		value;
	uint32		h, l;

	if (!addr)
	{
		attr->nullcount++;
		sql_buffer_clrbit(&attr->nullmap, row_index);
		sql_buffer_append_zero(&attr->values, sizeof(uint64));
	}
	else
	{
		h = ntohl(*((const uint32 *)(addr)));
		l = ntohl(*((const uint32 *)(addr + sizeof(uint32))));
		value = (uint64)h << 32 | (uint64)l;
		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &value, sz);
	}
}

#ifdef PG_INT128_TYPE
/* parameters of Numeric type */
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

static void
put_decimal_value(SQLattribute *attr,
				  const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;

	if (!addr)
	{
		attr->nullcount++;
		sql_buffer_clrbit(&attr->nullmap, row_index);
        sql_buffer_append_zero(&attr->values, sizeof(int128));
	}
	else
	{
		struct {
			int16		ndigits;
			int16		weight;		/* weight of first digit */
			int16		sign;		/* NUMERIC_(POS|NEG|NAN) */
			int16		dscale;		/* display scale */
			NumericDigit digits[FLEXIBLE_ARRAY_MEMBER];
		}	   *rawdata = (void *)addr;
		int		ndigits	= ntohs(rawdata->ndigits);
		int		weight	= ntohs(rawdata->weight);
		int		sign	= ntohs(rawdata->sign);
		int		precision = attr->arrow_type.Decimal.precision;
		int128	value = 0;
		int		d, dig;

		if ((sign & NUMERIC_SIGN_MASK) == NUMERIC_NAN)
			Elog("Decimal128 cannot map NaN in PostgreSQL Numeric");

		/* makes integer portion first */
		for (d=0; d <= weight; d++)
		{
			dig = (d < ndigits) ? ntohs(rawdata->digits[d]) : 0;
			if (dig < 0 || dig >= NBASE)
				Elog("Numeric digit is out of range: %d", (int)dig);
			value = NBASE * value + (int128)dig;
		}

		/* makes floating point portion if any */
		while (precision > 0)
		{
			dig = (d >= 0 && d < ndigits) ? ntohs(rawdata->digits[d]) : 0;
			if (dig < 0 || dig >= NBASE)
				Elog("Numeric digit is out of range: %d", (int)dig);

			if (precision >= DEC_DIGITS)
				value = NBASE * value + dig;
			else if (precision == 3)
				value = 1000L * value + dig / 10L;
			else if (precision == 2)
				value =  100L * value + dig / 100L;
			else if (precision == 1)
				value =   10L * value + dig / 1000L;
			else
				Elog("internal bug");
			precision -= DEC_DIGITS;
			d++;
		}
		/* is it a negative value? */
		if ((sign & NUMERIC_NEG) != 0)
			value = -value;

		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &value, sizeof(value));
	}
}
#endif

static void
put_date_value(SQLattribute *attr,
			   const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;

	if (!addr)
	{
		attr->nullcount++;
		sql_buffer_clrbit(&attr->nullmap, row_index);
		sql_buffer_append_zero(&attr->values, attr->attlen);
	}
    else
    {
		DateADT	value;

		assert(sz == sizeof(DateADT));
		sql_buffer_setbit(&attr->nullmap, row_index);
		value = ntohl(*((const DateADT *)addr));
		value += (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
		sql_buffer_append(&attr->values, &value, sz);
	}
}

static void
put_timestamp_value(SQLattribute *attr,
					const char *addr, int sz)
{
	size_t		row_index = attr->nitems++;

	if (!addr)
	{
		attr->nullcount++;
		sql_buffer_clrbit(&attr->nullmap, row_index);
		sql_buffer_append_zero(&attr->values, attr->attlen);
	}
	else
	{
		Timestamp	value;
		uint32		h, l;

		assert(sz == sizeof(Timestamp));
		sql_buffer_setbit(&attr->nullmap, row_index);
		h = ((const uint32 *)addr)[0];
		l = ((const uint32 *)addr)[1];
		value = (Timestamp)ntohl(h) << 32 | (Timestamp)ntohl(l);
		/* convert PostgreSQL epoch to UNIX epoch */
		value += (POSTGRES_EPOCH_JDATE -
				  UNIX_EPOCH_JDATE) * USECS_PER_DAY;
		sql_buffer_append(&attr->values, &value, sizeof(Timestamp));
	}
}

static void
put_variable_value(SQLattribute *attr,
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

static void
put_bpchar_value(SQLattribute *attr,
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

static void
put_array_value(SQLattribute *attr,
				const char *addr, int sz)
{
	SQLattribute *element = attr->element;
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
		static bool		notification_has_printed = false;
		struct {
			int32		ndim;
			int32		hasnull;
			int32		element_type;
			struct {
				int32	sz;
				int32	lb;
			} dim[FLEXIBLE_ARRAY_MEMBER];
		}  *rawdata = (void *) addr;
		int32		ndim = ntohl(rawdata->ndim);
		//int32		hasnull = ntohl(rawdata->hasnull);
		Oid			element_type = ntohl(rawdata->element_type);
		size_t		i, nitems = 1;
		int			item_sz;
		char	   *pos;

		if (element_type != element->atttypid)
			Elog("PostgreSQL array type mismatch");
		if (ndim < 1)
			Elog("Invalid dimension size of PostgreSQL Array (ndim=%d)", ndim);
		if (ndim > 1 && !notification_has_printed)
		{
			fprintf(stderr, "Notice: multi-dimensional PostgreSQL Array is extracted to List<%s> type in Apache Arrow, due to data type restrictions", attr->arrow_typename);
			notification_has_printed = true;
		}
		for (i=0; i < ndim; i++)
			nitems *= ntohl(rawdata->dim[i].sz);

		pos = (char *)&rawdata->dim[ndim];
		for (i=0; i < nitems; i++)
		{
			if (pos + sizeof(int32) > addr + sz)
				Elog("out of range - binary array has corruption");
			item_sz = ntohl(*((int32 *)pos));
			pos += sizeof(int32);
			if (item_sz < 0)
				element->put_value(element, NULL, 0);
			else
			{
				element->put_value(element, pos, item_sz);
				pos += item_sz;
			}
		}
		sql_buffer_setbit(&attr->nullmap, row_index);
		sql_buffer_append(&attr->values, &element->nitems, sizeof(int32));
	}
}

static void
put_composite_value(SQLattribute *attr,
					const char *addr, int sz)
{
	/* see record_send() */
	SQLtable   *subtypes = attr->subtypes;
	size_t		row_index = attr->nitems++;
	size_t		usage = 0;
	int			j, nvalids;

	if (!addr)
	{
		attr->nullcount++;
		sql_buffer_clrbit(&attr->nullmap, row_index);
		usage += ARROWALIGN(attr->nullmap.usage);
		/* NULL for all the subtypes */
		for (j=0; j < subtypes->nfields; j++)
		{
			SQLattribute *subattr = &subtypes->attrs[j];
			subattr->put_value(subattr, NULL, 0);
		}
	}
	else
	{
		const char *pos = addr;

		sql_buffer_setbit(&attr->nullmap, row_index);
		if (sz < sizeof(uint32))
			Elog("binary composite record corruption");
		if (attr->nullcount > 0)
			usage += ARROWALIGN(attr->nullmap.usage);
		nvalids = ntohl(*((const int *)pos));
		pos += sizeof(int);
		for (j=0; j < subtypes->nfields; j++)
		{
			SQLattribute *subattr = &subtypes->attrs[j];
			Oid		atttypid;
			int		attlen;

			assert(subattr->nitems == row_index);
			if (j >= nvalids)
			{
				subattr->put_value(subattr, NULL, 0);
				continue;
			}
			if ((pos - addr) + sizeof(Oid) + sizeof(int) > sz)
				Elog("binary composite record corruption");
			atttypid = ntohl(*((Oid *)pos));
			pos += sizeof(Oid);
			if (subattr->atttypid != atttypid)
				Elog("composite subtype mismatch");
			attlen = ntohl(*((int *)pos));
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
		}
	}
}

static void
put_dictionary_value(SQLattribute *attr,
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

/* ----------------------------------------------------------------
 *
 * buffer_usage handler for each data types
 *
 * ---------------------------------------------------------------- */
static size_t
buffer_usage_inline_type(SQLattribute *attr)
{
	size_t		usage;

	usage = ARROWALIGN(attr->values.usage);
	if (attr->nullcount > 0)
		usage += ARROWALIGN(attr->nullmap.usage);
	return usage;
}

static size_t
buffer_usage_varlena_type(SQLattribute *attr)
{
	size_t		usage;

	usage = (ARROWALIGN(attr->values.usage) +
			 ARROWALIGN(attr->extra.usage));
	if (attr->nullcount > 0)
		usage += ARROWALIGN(attr->nullmap.usage);
	return usage;
}

static size_t
buffer_usage_array_type(SQLattribute *attr)
{
	SQLattribute   *element = attr->element;
	size_t			usage;

	usage = ARROWALIGN(attr->values.usage);
	if (attr->nullcount > 0)
		usage += ARROWALIGN(attr->nullmap.usage);
	usage += element->buffer_usage(element);

	return usage;
}

static size_t
buffer_usage_composite_type(SQLattribute *attr)
{
	SQLtable   *subtypes = attr->subtypes;
	size_t		usage = 0;
	int			j;

	if (attr->nullcount > 0)
		usage += ARROWALIGN(attr->nullmap.usage);
	for (j=0; j < subtypes->nfields; j++)
	{
		SQLattribute *subattr = &subtypes->attrs[j];
		usage += subattr->buffer_usage(subattr);
	}
	return usage;
}

/* ----------------------------------------------------------------
 *
 * stat_update handler for each data types (optional)
 *
 * ---------------------------------------------------------------- */
#define STAT_UPDATE_INLINE_TEMPLATE(TYPENAME,TO_DATUM)			\
	static void													\
	stat_update_##TYPENAME##_value(SQLattribute *attr,			\
								   const char *addr, int sz)	\
	{															\
		TYPENAME		value;									\
																\
		if (!addr)												\
			return;												\
		value = *((const TYPENAME *)addr);						\
		if (attr->min_isnull)									\
		{														\
			attr->min_isnull = false;							\
			attr->min_value  = TO_DATUM(value);					\
		}														\
		else if (value < attr->min_value)						\
			attr->min_value = value;							\
																\
		if (attr->max_isnull)									\
		{														\
			attr->max_isnull = false;							\
			attr->max_value  = TO_DATUM(value);					\
		}														\
		else if (value > attr->max_value)						\
			attr->max_value = value;							\
	}

STAT_UPDATE_INLINE_TEMPLATE(int8,   Int8GetDatum)
STAT_UPDATE_INLINE_TEMPLATE(int16,  Int16GetDatum)
STAT_UPDATE_INLINE_TEMPLATE(int32,  Int32GetDatum)
STAT_UPDATE_INLINE_TEMPLATE(int64,  Int64GetDatum)
STAT_UPDATE_INLINE_TEMPLATE(float4, Float4GetDatum)
STAT_UPDATE_INLINE_TEMPLATE(float8, Float8GetDatum)

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
setup_buffer_inline_type(SQLattribute *attr,
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
setup_buffer_varlena_type(SQLattribute *attr,
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
setup_buffer_array_type(SQLattribute *attr,
						ArrowBuffer *node, size_t *p_offset)
{
	SQLattribute *element = attr->element;
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
setup_buffer_composite_type(SQLattribute *attr,
							ArrowBuffer *node, size_t *p_offset)
{
	SQLtable   *subtypes = attr->subtypes;
	int			i, count = 1;

	/* nullmap */
	if (attr->nullcount == 0)
		*p_offset += setup_arrow_buffer(node, *p_offset, 0);
	else
		*p_offset += setup_arrow_buffer(node, *p_offset,
										attr->nullmap.usage);
	/* walk down the sub-types */
	for (i=0; i < subtypes->nfields; i++)
	{
		SQLattribute   *subattr = &subtypes->attrs[i];

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
write_buffer_inline_type(SQLattribute *attr, int fdesc)
{
	/* nullmap */
	if (attr->nullcount > 0)
		__write_buffer_common(fdesc,
							  attr->nullmap.data,
							  attr->nullmap.usage);
	/* fixed length values */
	__write_buffer_common(fdesc,
						  attr->values.data,
						  attr->values.usage);
}

static void
write_buffer_varlena_type(SQLattribute *attr, int fdesc)
{
	/* nullmap */
	if (attr->nullcount > 0)
		__write_buffer_common(fdesc,
							  attr->nullmap.data,
							  attr->nullmap.usage);
	/* index values */
	__write_buffer_common(fdesc,
						  attr->values.data,
						  attr->values.usage);
	/* extra buffer */
	__write_buffer_common(fdesc,
						  attr->extra.data,
						  attr->extra.usage);
}

static void
write_buffer_array_type(SQLattribute *attr, int fdesc)
{
	SQLattribute *element = attr->element;

	/* nullmap */
	if (attr->nullcount > 0)
		__write_buffer_common(fdesc,
							  attr->nullmap.data,
							  attr->nullmap.usage);
	/* offset values */
	__write_buffer_common(fdesc,
						  attr->values.data,
						  attr->values.usage);
	/* element values */
	element->write_buffer(element, fdesc);
}

static void
write_buffer_composite_type(SQLattribute *attr, int fdesc)
{
	SQLtable   *subtypes = attr->subtypes;
	int			i;

	/* nullmap */
	if (attr->nullcount > 0)
		__write_buffer_common(fdesc,
							  attr->nullmap.data,
							  attr->nullmap.usage);
	/* sub-types */
	for (i=0; i < subtypes->nfields; i++)
	{
		SQLattribute   *subattr = &subtypes->attrs[i];

		subattr->write_buffer(subattr, fdesc);
	}
}

/* ----------------------------------------------------------------
 *
 * setup handler for each data types
 *
 * ----------------------------------------------------------------
 */
static void
assignArrowTypeInt(SQLattribute *attr, int *p_numBuffers, bool is_signed)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Int);
	attr->arrow_type.Int.is_signed = is_signed;
	switch (attr->attlen)
	{
		case sizeof(char):
			attr->arrow_type.Int.bitWidth = 8;
			attr->arrow_typename = (is_signed ? "Int8" : "Uint8");
			attr->put_value = put_inline_8b_value;
			attr->stat_update = stat_update_int8_value;
			break;
		case sizeof(short):
			attr->arrow_type.Int.bitWidth = 16;
			attr->arrow_typename = (is_signed ? "Int16" : "Uint16");
			attr->put_value = put_inline_16b_value;
			attr->stat_update = stat_update_int16_value;
			break;
		case sizeof(int):
			attr->arrow_type.Int.bitWidth = 32;
			attr->arrow_typename = (is_signed ? "Int32" : "Uint32");
			attr->put_value = put_inline_32b_value;
			attr->stat_update = stat_update_int32_value;
			break;
		case sizeof(long):
			attr->arrow_type.Int.bitWidth = 64;
			attr->arrow_typename = (is_signed ? "Int64" : "Uint64");
			attr->put_value = put_inline_64b_value;
			attr->stat_update = stat_update_int64_value;
			break;
		default:
			Elog("unsupported Int width: %d", attr->attlen);
			break;
	}
	attr->buffer_usage = buffer_usage_inline_type;
	attr->setup_buffer = setup_buffer_inline_type;
	attr->write_buffer = write_buffer_inline_type;

	*p_numBuffers += 2;		/* nullmap + values */
}

static void
assignArrowTypeFloatingPoint(SQLattribute *attr, int *p_numBuffers)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, FloatingPoint);
	switch (attr->attlen)
	{
		case sizeof(short):		/* half */
			attr->arrow_type.FloatingPoint.precision = ArrowPrecision__Half;
			attr->arrow_typename = "Float16";
			attr->put_value = put_inline_16b_value;
			break;
		case sizeof(float):
			attr->arrow_type.FloatingPoint.precision = ArrowPrecision__Single;
			attr->arrow_typename = "Float32";
			attr->put_value = put_inline_32b_value;
			attr->stat_update = stat_update_float4_value;
			break;
		case sizeof(double):
			attr->arrow_type.FloatingPoint.precision = ArrowPrecision__Double;
			attr->arrow_typename = "Float64";
			attr->put_value = put_inline_64b_value;
			attr->stat_update = stat_update_float8_value;
			break;
		default:
			Elog("unsupported floating point width: %d", attr->attlen);
			break;
	}
	attr->buffer_usage = buffer_usage_inline_type;
	attr->setup_buffer = setup_buffer_inline_type;
	attr->write_buffer = write_buffer_inline_type;

	*p_numBuffers += 2;		/* nullmap + values */
}

static void
assignArrowTypeBinary(SQLattribute *attr, int *p_numBuffers)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Binary);
	attr->arrow_typename	= "Binary";
	attr->put_value			= put_variable_value;
	attr->buffer_usage		= buffer_usage_varlena_type;
	attr->setup_buffer		= setup_buffer_varlena_type;
	attr->write_buffer		= write_buffer_varlena_type;

	*p_numBuffers += 3;		/* nullmap + index + extra */
}

static void
assignArrowTypeUtf8(SQLattribute *attr, int *p_numBuffers)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Utf8);
	attr->arrow_typename	= "Utf8";
	attr->put_value			= put_variable_value;
	attr->buffer_usage		= buffer_usage_varlena_type;
	attr->setup_buffer		= setup_buffer_varlena_type;
	attr->write_buffer		= write_buffer_varlena_type;

	*p_numBuffers += 3;		/* nullmap + index + extra */
}

static void
assignArrowTypeBpchar(SQLattribute *attr, int *p_numBuffers)
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

	*p_numBuffers += 2;		/* nullmap + values */
}

static void
assignArrowTypeBool(SQLattribute *attr, int *p_numBuffers)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Bool);
	attr->arrow_typename	= "Bool";
	attr->put_value			= put_inline_bool_value;
	attr->stat_update		= stat_update_int8_value;
	attr->buffer_usage		= buffer_usage_inline_type;
	attr->setup_buffer		= setup_buffer_inline_type;
	attr->write_buffer		= write_buffer_inline_type;

	*p_numBuffers += 2;		/* nullmap + values */
}

static void
assignArrowTypeDecimal(SQLattribute *attr, int *p_numBuffers)
{
#ifdef PG_INT128_TYPE
	int		typmod			= attr->atttypmod;
	int		precision		= 11;	/* default, if typmod == -1 */
	int		scale			= 30;	/* default, if typmod == -1 */

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

	*p_numBuffers += 2;		/* nullmap + values */
#else
	/*
	 * MEMO: Numeric of PostgreSQL is mapped to Decimal128 in Apache Arrow.
	 * Due to implementation reason, we require int128 support by compiler.
	 */
	Elog("Numeric type of PostgreSQL is not supported in this build");
#endif
}

static void
assignArrowTypeDate(SQLattribute *attr, int *p_numBuffers)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Date);
	attr->arrow_type.Date.unit = ArrowDateUnit__Day;
	attr->arrow_typename	= "Date";
	attr->put_value			= put_date_value;
	attr->stat_update		= stat_update_int32_value;
	attr->buffer_usage		= buffer_usage_inline_type;
	attr->setup_buffer		= setup_buffer_inline_type;
	attr->write_buffer		= write_buffer_inline_type;

	*p_numBuffers += 2;		/* nullmap + values */
}

static void
assignArrowTypeTime(SQLattribute *attr, int *p_numBuffers)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Time);
	attr->arrow_type.Time.unit = ArrowTimeUnit__MicroSecond;
	attr->arrow_type.Time.bitWidth = 64;
	attr->arrow_typename	= "Time";
	attr->put_value			= put_inline_64b_value;
	attr->stat_update		= stat_update_int64_value;
	attr->buffer_usage		= buffer_usage_inline_type;
	attr->setup_buffer		= setup_buffer_inline_type;
	attr->write_buffer		= write_buffer_inline_type;

	*p_numBuffers += 2;		/* nullmap + values */
}

static void
assignArrowTypeTimestamp(SQLattribute *attr, int *p_numBuffers)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Timestamp);
	attr->arrow_type.Timestamp.unit = ArrowTimeUnit__MicroSecond;
	attr->arrow_typename	= "Timestamp";
	attr->put_value			= put_timestamp_value;
	attr->stat_update		= stat_update_int64_value;
	attr->buffer_usage		= buffer_usage_inline_type;
	attr->setup_buffer		= setup_buffer_inline_type;
	attr->write_buffer		= write_buffer_inline_type;

	*p_numBuffers += 2;		/* nullmap + values */
}

#if 0
static void
assignArrowTypeInterval(SQLattribute *attr, int *p_numBuffers)
{
	usec + day + mon;
	Elog("Interval is not supported yet");
}
#endif

static void
assignArrowTypeList(SQLattribute *attr, int *p_numBuffers)
{
	SQLattribute *element = attr->element;

	INIT_ARROW_TYPE_NODE(&attr->arrow_type, List);
	attr->put_value			= put_array_value;
	attr->arrow_typename	= psprintf("List<%s>", element->arrow_typename);
	attr->buffer_usage		= buffer_usage_array_type;
	attr->setup_buffer		= setup_buffer_array_type;
	attr->write_buffer		= write_buffer_array_type;

	*p_numBuffers += 2;		/* nullmap + offset vector */
}

static void
assignArrowTypeStruct(SQLattribute *attr, int *p_numBuffers)
{
	assert(attr->subtypes != NULL);
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Struct);
	attr->arrow_typename	= "Struct";
	attr->put_value			= put_composite_value;
	attr->buffer_usage		= buffer_usage_composite_type;
	attr->setup_buffer		= setup_buffer_composite_type;
	attr->write_buffer		= write_buffer_composite_type;

	*p_numBuffers += 1;		/* only nullmap */
}

static void
assignArrowTypeDictionary(SQLattribute *attr, int *p_numBuffers)
{
	INIT_ARROW_TYPE_NODE(&attr->arrow_type, Utf8);
	attr->arrow_typename	= psprintf("Enum; dictionary=%u", attr->atttypid);
	attr->put_value			= put_dictionary_value;
	attr->buffer_usage		= buffer_usage_inline_type;
	attr->setup_buffer		= setup_buffer_inline_type;
	attr->write_buffer		= write_buffer_inline_type;

	*p_numBuffers += 2;		/* nullmap + values */
}

/*
 * assignArrowType
 */
void
assignArrowType(SQLattribute *attr, int *p_numBuffers)
{
	memset(&attr->arrow_type, 0, sizeof(ArrowType));
	if (attr->subtypes)
	{
		/* composite type */
		assignArrowTypeStruct(attr, p_numBuffers);
		return;
	}
	else if (attr->element)
	{
		/* array type */
		assignArrowTypeList(attr, p_numBuffers);
		return;
	}
	else if (attr->typtype == 'e')
	{
		/* enum type */
		assignArrowTypeDictionary(attr, p_numBuffers);
		return;
	}
	else if (strcmp(attr->typnamespace, "pg_catalog") == 0)
	{
		/* well known built-in data types? */
		if (strcmp(attr->typname, "bool") == 0)
		{
			assignArrowTypeBool(attr, p_numBuffers);
			return;
		}
		else if (strcmp(attr->typname, "int2") == 0 ||
				 strcmp(attr->typname, "int4") == 0 ||
				 strcmp(attr->typname, "int8") == 0)
		{
			assignArrowTypeInt(attr, p_numBuffers, true);
			return;
		}
		else if (strcmp(attr->typname, "float2") == 0 ||	/* by PG-Strom */
				 strcmp(attr->typname, "float4") == 0 ||
				 strcmp(attr->typname, "float8") == 0)
		{
			assignArrowTypeFloatingPoint(attr, p_numBuffers);
			return;
		}
		else if (strcmp(attr->typname, "date") == 0)
		{
			assignArrowTypeDate(attr, p_numBuffers);
			return;
		}
		else if (strcmp(attr->typname, "time") == 0)
		{
			assignArrowTypeTime(attr, p_numBuffers);
			return;
		}
		else if (strcmp(attr->typname, "timestamp") == 0 ||
				 strcmp(attr->typname, "timestamptz") == 0)
		{
			assignArrowTypeTimestamp(attr, p_numBuffers);
			return;
		}
		else if (strcmp(attr->typname, "text") == 0 ||
				 strcmp(attr->typname, "varchar") == 0)
		{
			assignArrowTypeUtf8(attr, p_numBuffers);
			return;
		}
		else if (strcmp(attr->typname, "bpchar") == 0)
		{
			assignArrowTypeBpchar(attr, p_numBuffers);
			return;
		}
		else if (strcmp(attr->typname, "numeric") == 0)
		{
			assignArrowTypeDecimal(attr, p_numBuffers);
			return;
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
			assignArrowTypeInt(attr, p_numBuffers, false);
			return;
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
		assignArrowTypeBinary(attr, p_numBuffers);
		return;
	}
	Elog("PostgreSQL type: '%s.%s' is not supported",
		 attr->typnamespace,
		 attr->typname);
}
