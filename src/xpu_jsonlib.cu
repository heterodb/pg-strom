/*
 * xpu_jsonlib.cu
 *
 * Collection of json functions for xPU device code
 * --
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "xpu_common.h"

typedef uint32_t				JEntry;

#define JENTRY_OFFLENMASK		0x0FFFFFFF
#define JENTRY_TYPEMASK			0x70000000
#define JENTRY_HAS_OFF			0x80000000

/* values stored in the type bits */
#define JENTRY_ISSTRING         0x00000000
#define JENTRY_ISNUMERIC        0x10000000
#define JENTRY_ISBOOL_FALSE     0x20000000
#define JENTRY_ISBOOL_TRUE      0x30000000
#define JENTRY_ISNULL           0x40000000
#define JENTRY_ISCONTAINER      0x50000000  /* array or object */

/* access macros */
#define JBE_OFFLENFLD(je)	((je) & JENTRY_OFFLENMASK)
#define JBE_HAS_OFF(je)		(((je) & JENTRY_HAS_OFF) != 0)
#define JBE_ISSTRING(je)	(((je) & JENTRY_TYPEMASK) == JENTRY_ISSTRING)
#define JBE_ISNUMERIC(je)	(((je) & JENTRY_TYPEMASK) == JENTRY_ISNUMERIC)
#define JBE_ISCONTAINER(je)	(((je) & JENTRY_TYPEMASK) == JENTRY_ISCONTAINER)
#define JBE_ISNULL(je)		(((je) & JENTRY_TYPEMASK) == JENTRY_ISNULL)
#define JBE_ISBOOL_TRUE(je)	(((je) & JENTRY_TYPEMASK) == JENTRY_ISBOOL_TRUE)
#define JBE_ISBOOL_FALSE(je) (((je) & JENTRY_TYPEMASK) == JENTRY_ISBOOL_FALSE)
#define JBE_ISBOOL(je)		(JBE_ISBOOL_TRUE(je_) || JBE_ISBOOL_FALSE(je_))

/*
 * We store an offset every JB_OFFSET_STRIDE children, regardless of the
 * sub-field type.
 */
#define JB_OFFSET_STRIDE		32

/* Jsonb array or object node */
typedef struct JsonbContainer
{
	uint32_t	header;
	JEntry		children[1];
} JsonbContainer;

/* flags for the header-field in JsonbContainer */
#define JB_CMASK			0x0FFFFFFF	/* mask for count field */
#define JB_FSCALAR			0x10000000  /* flag bits */
#define JB_FOBJECT			0x20000000
#define JB_FARRAY			0x40000000

/* convenience macros for accessing a JsonbContainer struct */
#define JsonContainerSize(jch)		((jch) & JB_CMASK)
#define JsonContainerIsScalar(jch)	(((jch) & JB_FSCALAR) != 0)
#define JsonContainerIsObject(jch)	(((jch) & JB_FOBJECT) != 0)
#define JsonContainerIsArray(jch)	(((jch) & JB_FARRAY) != 0)

/*
 * Basic Jsonb type handlers
 */
INLINE_FUNCTION(bool)
xpu_jsonb_is_valid(kern_context *kcxt, const xpu_jsonb_t *arg)
{
	if (arg->length < 0)
	{
		STROM_CPU_FALLBACK(kcxt, "jsonb datum is compressed or external");
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_jsonb_datum_ref(kern_context *kcxt,
					xpu_datum_t *__result,
					int vclass,
					const kern_variable *kvar)
{
	xpu_jsonb_t	   *result = (xpu_jsonb_t *)__result;
	const char	   *addr = (const char *)kvar->ptr;

	if (vclass == KVAR_CLASS__VARLENA)
	{
		if (VARATT_IS_EXTERNAL(addr) || VARATT_IS_COMPRESSED(addr))
		{
			result->value  = addr;
			result->length = -1;
		}
		else
		{
			result->value  = VARDATA_ANY(addr);
			result->length = VARSIZE_ANY_EXHDR(addr);
        }
	}
	else if (vclass >= 0)
	{
		result->value  = addr;
		result->length = vclass;
	}
	else
	{
		STROM_ELOG(kcxt, "unexpected vclass for device jsonb data type.");
		return false;
	}
	result->expr_ops = &xpu_jsonb_ops;
	return true;
}

STATIC_FUNCTION(bool)
xpu_jsonb_datum_store(kern_context *kcxt,
					  const xpu_datum_t *__arg,
					  int *p_vclass,
					  kern_variable *p_kvar)
{
	const xpu_jsonb_t  *arg = (const xpu_jsonb_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
	{
		*p_vclass = KVAR_CLASS__NULL;
	}
	else if (arg->length < 0)
	{
		*p_vclass   = KVAR_CLASS__VARLENA;
		p_kvar->ptr = (void *)arg->value;
	}
	else
	{
		*p_vclass   = arg->length;
		p_kvar->ptr = (void *)arg->value;
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_jsonb_datum_write(kern_context *kcxt,
					  char *buffer,
					  const kern_colmeta *cmeta,
					  const xpu_datum_t *__arg)
{
	const xpu_jsonb_t  *arg = (const xpu_jsonb_t *)__arg;
	int		nbytes;

	if (XPU_DATUM_ISNULL(arg))
		return 0;
	if (arg->length < 0)
	{
		nbytes = VARSIZE_ANY(arg->value);
		if (buffer)
			memcpy(buffer, arg->value, nbytes);
	}
	else
	{
		nbytes = VARHDRSZ + arg->length;
		if (buffer)
		{
			memcpy(buffer+VARHDRSZ, arg->value, arg->length);
			SET_VARSIZE(buffer, nbytes);
		}
	}
	return nbytes;
}

/*
 * xpu_jsonb_hash
 */
INLINE_FUNCTION(uint32_t)
getJsonbOffset(const JsonbContainer *jc,		/* may not be aligned */
			   int index)
{
	uint32_t		offset = 0;
	JEntry		entry;
	int			j;

	for (j=index-1; j >= 0; j--)
	{
		entry = __Fetch(&jc->children[j]);
		offset += JBE_OFFLENFLD(entry);
		if (JBE_HAS_OFF(entry))
			break;
	}
	return offset;
}

INLINE_FUNCTION(uint32_t)
getJsonbLength(const JsonbContainer *jc,		/* may not be aligned */
			   int index)
{
	JEntry		entry = __Fetch(&jc->children[index]);
	uint32_t		off;
	uint32_t		len;

	if (JBE_HAS_OFF(entry))
	{
		off = getJsonbOffset(jc, index);
		len = JBE_OFFLENFLD(entry) - off;
	}
	else
		len = JBE_OFFLENFLD(entry);
	return len;
}

/*
 * Jsonb specific hash function
 */
STATIC_FUNCTION(uint32_t)
__xpu_jsonb_comp_hash(kern_context *kcxt, JsonbContainer *jc)
{
	uint32_t	hash = 0;
	uint32_t	jheader = __Fetch(&jc->header);
	uint32_t	j, nitems = JsonContainerSize(jheader);
	char	   *base;
	char	   *data;
	uint32_t	datalen;

	if (!JsonContainerIsScalar(jheader))
	{
		if (JsonContainerIsObject(jheader))
		{
			base = (char *)(jc->children + 2 * nitems);
			hash ^= JB_FOBJECT;
		}
		else
		{
			base = (char *)(jc->children + nitems);
			hash ^= JB_FARRAY;
		}
	}

	for (j=0; j < nitems; j++)
	{
		uint32_t	index = j;
		uint32_t	temp;
		JEntry		entry;

		/* hash value for key */
		if (JsonContainerIsObject(jheader))
		{
			entry = __Fetch(&jc->children[index]);
			if (!JBE_ISSTRING(entry))
			{
				STROM_ELOG(kcxt, "corrupted jsonb entry");
				return 0;
			}
			data = base + getJsonbOffset(jc, index);
			datalen = getJsonbLength(jc, index);
			temp = pg_hash_any((uint8_t *)data, datalen);
			hash = ((hash << 1) | (hash >> 31)) ^ temp;

			index += nitems;
		}
		/* hash value for element */
		entry = __Fetch(&jc->children[index]);
		if (JBE_ISNULL(entry))
			temp = 0x01;
		else if (JBE_ISSTRING(entry))
		{
			data = base + getJsonbOffset(jc, index);
			datalen = getJsonbLength(jc, index);
			temp = pg_hash_any((uint8_t *)data, datalen);
		}
		else if (JBE_ISNUMERIC(entry))
		{
			xpu_numeric_t	num;
			const char	   *error_msg;

			data = base + INTALIGN(getJsonbOffset(jc, index));
			error_msg = __xpu_numeric_from_varlena(&num, (varlena *)data);
			if (error_msg)
			{
				STROM_ELOG(kcxt, error_msg);
				return false;
			}
			else if (!num.expr_ops)
				temp = 0;
			else
				num.expr_ops->xpu_datum_hash(kcxt, &temp, (xpu_datum_t *)&num);
		}
		else if (JBE_ISBOOL_TRUE(entry))
			temp = 0x02;
		else if (JBE_ISBOOL_FALSE(entry))
			temp = 0x04;
		else if (JBE_ISCONTAINER(entry))
		{
			data = base + INTALIGN(getJsonbOffset(jc, index));
			temp = __xpu_jsonb_comp_hash(kcxt, (JsonbContainer *)data);
		}
		else
		{
			STROM_ELOG(kcxt, "corrupted jsonb entry");
			return 0;
		}
		hash = ((hash << 1) | (hash >> 31)) ^ temp;
	}
	return hash;
}

STATIC_FUNCTION(bool)
xpu_jsonb_datum_hash(kern_context*kcxt,
					 uint32_t *p_hash,
					 const xpu_datum_t *__arg)
{
	const xpu_jsonb_t  *arg = (const xpu_jsonb_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_hash = 0;
	else if (xpu_jsonb_is_valid(kcxt, arg))
		*p_hash = __xpu_jsonb_comp_hash(kcxt, (JsonbContainer *)arg->value);
	else
		return false;

	return true;
}

STATIC_FUNCTION(bool)
xpu_jsonb_datum_comp(kern_context*kcxt,
					 int *p_comp,
					 const xpu_datum_t *__a,
					 const xpu_datum_t *__b)
{
	STROM_ELOG(kcxt, "device jsonb type has no compare handler");
	return false;
}

STATIC_FUNCTION(bool)
xpu_jsonb_datum_load_heap(kern_context *kcxt,
						  kvec_datum_t *__result,
						  int kvec_id,
						  const char *addr)
{
	kvec_jsonb_t *result = (kvec_jsonb_t *)__result;

	kvec_update_nullmask(&result->nullmask, kvec_id, addr);
	if (addr)
	{
		if (VARATT_IS_EXTERNAL(addr) || VARATT_IS_COMPRESSED(addr))
		{
			result->values[kvec_id] = addr;
			result->length[kvec_id] = -1;
		}
		else
		{
			result->values[kvec_id] = VARDATA_ANY(addr);
			result->length[kvec_id] = VARSIZE_ANY_EXHDR(addr);
		}
	}
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(jsonb,false,4,-1);

/* ----------------------------------------------------------------
 *
 * JSONB access operators
 *
 * ---------------------------------------------------------------- */
INLINE_FUNCTION(int32_t)
compareJsonbStringValue(const char *s1, int32_t len1,
						const char *s2, int32_t len2)
{
	if (len1 > len2)
		return 1;
	else if (len1 < len2)
		return -1;
	for (int i=0; i < len1; i++, s1++, s2++)
	{
		if (*s1 < *s2)
			return -1;
		if (*s1 > *s2)
			return 1;
	}
	return 0;
}

STATIC_FUNCTION(int32_t)
findJsonbIndexFromObject(JsonbContainer *jc,	/* may not be aligned */
						 const char *key, int32_t keylen)
{
	uint32_t		jheader = __Fetch(&jc->header);

	if (JsonContainerIsObject(jheader))
	{
		uint32_t	count = JsonContainerSize(jheader);
		uint32_t	stopLow = 0;
		uint32_t	stopHigh = count;
		char	   *base = (char *)(jc->children + 2 * count);

		/* Binary search on object/pair keys *only* */
		while (stopLow < stopHigh)
		{
			uint32_t	stopMiddle = stopLow + (stopHigh - stopLow) / 2;
			int32_t		diff;
			char	   *name;
			int32_t		namelen;

			name = base + getJsonbOffset(jc, stopMiddle);
			namelen = getJsonbLength(jc, stopMiddle);
			diff = compareJsonbStringValue(name, namelen, key, keylen);
			if (diff < 0)
				stopLow = stopMiddle + 1;
			else if (diff > 0)
				stopHigh = stopMiddle;
			else
				return stopMiddle;		/* found */
		}
	}
	return -1;
}

STATIC_FUNCTION(bool)
extractJsonbItemFromContainer(kern_context *kcxt,
							  xpu_jsonb_t *result,
							  JsonbContainer *jc,	/* may not be aligned */
							  int32_t index, char *base)
{
	uint32_t	jheader = __Fetch(&jc->header);
	JEntry		entry;

	assert(JsonContainerIsObject(jheader) ||
		   JsonContainerIsArray(jheader));
	/* extract jsonb object value */
	entry = __Fetch(&jc->children[index]);

	if (JBE_ISNULL(entry) ||
		JBE_ISSTRING(entry) ||
		JBE_ISNUMERIC(entry) ||
		JBE_ISBOOL_TRUE(entry) ||
		JBE_ISBOOL_FALSE(entry))
	{
		uint32_t	sz = offsetof(JsonbContainer, children[1]);
		char	   *data = NULL;
		uint32_t	datalen = 0;
		JsonbContainer *r;

		if (JBE_ISSTRING(entry))
		{
			data = base + getJsonbOffset(jc, index);
			datalen = getJsonbLength(jc, index);
			sz += datalen;
		}
		else if (JBE_ISNUMERIC(entry))
		{
			data = base + INTALIGN(getJsonbOffset(jc, index));
			datalen = VARSIZE_ANY(data);
			sz += datalen;
		}
		r = (JsonbContainer *)kcxt_alloc(kcxt, sz);
		if (!r)
		{
			STROM_CPU_FALLBACK(kcxt, "out of memory");
			return false;
		}
		r->header = JB_FARRAY | JB_FSCALAR | 1;
		if (!data)
		{
			entry = (entry & JENTRY_TYPEMASK);
		}
		else
		{
			memcpy(r->children, data, datalen);
			entry = (entry & JENTRY_TYPEMASK) | datalen;
		}
		r->children[0] = entry;

		result->expr_ops = &xpu_jsonb_ops;
		result->length   = sz;
		result->value    = (char *)r;
	}
	else if (JBE_ISCONTAINER(entry))
	{
		char	   *data = base + INTALIGN(getJsonbOffset(jc, index));
		uint32_t	datalen = getJsonbLength(jc, index);

		result->expr_ops = &xpu_jsonb_ops;
		result->length   = datalen;
		result->value    = data;
	}
	else
	{
		STROM_ELOG(kcxt, "corrupted jsonb entry");
		return false;
	}
	return true;
}

INLINE_FUNCTION(int32_t)
escape_json_cstring(const char *str, int32_t strlen, char *buf, char *endp)
{
	char   *pos = buf;
	int		i, c, bits;

	if (pos >= endp)
		return -1;
	*pos++ = '"';
	for (i=0; i < strlen; i++)
	{
		c = (unsigned char)str[i];

		switch (c)
		{
			case '\b':
				if (pos + 2 > endp)
					return -1;
				*pos++ = '\\';
				*pos++ = 'b';
				break;
			case '\f':
				if (pos + 2 > endp)
					return -1;
				*pos++ = '\\';
				*pos++ = 'f';
                break;
            case '\n':
				if (pos + 2 > endp)
					return -1;
				*pos++ = '\\';
				*pos++ = 'n';
				break;
            case '\r':
				if (pos + 2 > endp)
					return -1;
				*pos++ = '\\';
				*pos++ = 'r';
				break;
            case '\t':
				if (pos + 2 > endp)
					return -1;
				*pos++ = '\\';
				*pos++ = 't';
				break;
            case '"':
				if (pos + 2 > endp)
					return -1;
				*pos++ = '\\';
				*pos++ = '"';
                break;
            case '\\':
				if (pos + 2 > endp)
					return -1;
				*pos++ = '\\';
				*pos++ = '\\';
				break;
            default:
				if (c < ' ')
				{
					if (pos + 6 > endp)
						return -1;
					*pos++ = '\\';
					*pos++ = 'u';
					*pos++ = '0';
					*pos++ = '0';
					bits = (c & 0x00f0) >> 4;
					*pos++ = (bits >= 10 ? bits + 'a' - 10 : bits + '0');
					bits = (c & 0x000f);
					*pos++ = (bits >= 10 ? bits + 'a' - 10 : bits + '0');
				}
				else
				{
					if (pos >= endp)
						return -1;
					*pos++ = c;
				}
		}
	}
	if (pos >= endp)
		return -1;
	*pos++ = '"';

	return (int32_t)(pos - buf);
}

STATIC_FUNCTION(int32_t)
pg_jsonb_to_cstring(kern_context *kcxt,
					JsonbContainer *jc,		/* may not be aligned */
					char *buf, char *endp, int depth)
{
	uint32_t	jheader;
	uint32_t	j, count;
	JEntry		entry;
	char	   *base;
	char	   *pos = buf;
	char	   *data;
	int32_t		datalen, sz;

	if (depth > 8)
	{
		STROM_CPU_FALLBACK(kcxt,"too deep recursive function call");
		return -1;
	}

	jheader = __Fetch(&jc->header);
	count = JsonContainerSize(jheader);
	if (JsonContainerIsObject(jheader))
		base = (char *)(jc->children + 2 * count);
	else
		base = (char *)(jc->children + count);
	if (!JsonContainerIsScalar(jheader))
	{
		if (pos >= endp)
			return -1;
		*pos++ = (JsonContainerIsObject(jheader) ? '{' : '[');
	}

	for (j=0; j < count; j++)
	{
		int		index = j;

		if (j > 0)
		{
			if (pos + 2 > endp)
				return -1;
			*pos++ = ',';
			*pos++ = ' ';
		}
		/* key name */
		if (JsonContainerIsObject(jheader))
		{
			entry = __Fetch(&jc->children[index]);
			if (!JBE_ISSTRING(entry))
			{
				STROM_ELOG(kcxt, "corrupter jsonb entry");
				return -1;
			}
			data = base + getJsonbOffset(jc, index);
			datalen = getJsonbLength(jc, index);
			sz = escape_json_cstring(data, datalen, pos, endp);
			if (sz < 0)
				return -1;
			pos += sz;
			if (pos + 2 > endp)
				return -1;
			*pos++ = ':';
			*pos++ = ' ';

			index += count;
		}
		/* element value */
		entry = __Fetch(&jc->children[index]);
		if (JBE_ISSTRING(entry))
		{
			data = base + getJsonbOffset(jc, index);
			datalen = getJsonbLength(jc, index);
			sz = escape_json_cstring(data, datalen, pos, endp);
			if (sz < 0)
				return -1;
			pos += sz;
		}
		else if (JBE_ISNUMERIC(entry))
		{
			data = base + INTALIGN(getJsonbOffset(jc, index));

			sz = pg_numeric_to_cstring(kcxt, (varlena *)data, pos, endp);
			if (sz < 0)
				return -1;
			pos += sz;
		}
		else if (JBE_ISCONTAINER(entry))
		{
			data = base + INTALIGN(getJsonbOffset(jc, index));

			sz = pg_jsonb_to_cstring(kcxt, (JsonbContainer *)data,
									 pos, endp, depth+1);
			if (sz < 0)
				return -1;
			pos += sz;
		}
		else if (JBE_ISNULL(entry))
		{
			if (pos + 4 > endp)
				return -1;
			memcpy(pos, "null", 4);
			pos += 4;
		}
		else if (JBE_ISBOOL_TRUE(entry))
		{
			if (pos + 4 > endp)
				return -1;
			memcpy(pos, "true", 4);
			pos += 4;
		}
		else if (JBE_ISBOOL_FALSE(entry))
		{
			if (pos + 5 > endp)
				return -1;
			memcpy(pos, "false", 5);
			pos += 5;
		}
		else
		{
			STROM_ELOG(kcxt, "corrupted jsonb entry");
			return -1;
		}
	}

	if (!JsonContainerIsScalar(jheader))
	{
		if (pos >= endp)
			return -1;
		*pos++ = (JsonContainerIsObject(jheader) ? '}' : ']');
	}
	return (int)(pos - buf);
}

STATIC_FUNCTION(bool)
extractTextItemFromContainer(kern_context *kcxt,
							 xpu_text_t *result,
							 JsonbContainer *jc,	/* may not be aligned */
							 int32_t index, char *base)
{
	uint32_t	jheader = __Fetch(&jc->header);
	JEntry		entry;

	assert(JsonContainerIsObject(jheader) ||
		   JsonContainerIsArray(jheader));
	/* extract jsonb object value */
	entry = __Fetch(&jc->children[index]);
	if (JBE_ISNULL(entry))
		result->expr_ops = NULL;
	else if (JBE_ISSTRING(entry))
	{
		char	   *data = base + getJsonbOffset(jc, index);
		uint32_t	datalen = getJsonbLength(jc, index);

		result->expr_ops = &xpu_text_ops;
		result->length = datalen;
		result->value = data;
	}
	else if (JBE_ISNUMERIC(entry))
	{
		char	   *data = base + INTALIGN(getJsonbOffset(jc, index));
		char	   *vlpos = (char *)MAXALIGN(kcxt->vlpos);
		int32_t		sz;

		sz = pg_numeric_to_cstring(kcxt, (varlena *)data, vlpos, kcxt->vlend);
		if (sz < 0)
		{
			STROM_CPU_FALLBACK(kcxt, "out of memory");
			return false;
		}
		else
		{
			kcxt->vlpos = vlpos + sz;
			assert(kcxt->vlpos <= kcxt->vlend);

			result->expr_ops = &xpu_text_ops;
			result->length = sz;
			result->value = (char *)vlpos;
		}
	}
	else if (JBE_ISBOOL_TRUE(entry))
	{
		result->expr_ops = &xpu_text_ops;
		result->length = 4;
		result->value = "true";
	}
	else if (JBE_ISBOOL_FALSE(entry))
	{
		result->expr_ops = &xpu_text_ops;
		result->length = 5;
		result->value = "false";
	}
	else if (JBE_ISCONTAINER(entry))
	{
		char	   *data = base + INTALIGN(getJsonbOffset(jc, index));
		char	   *vlpos = (char *)MAXALIGN(kcxt->vlpos);
		int32_t		sz;

		sz = pg_jsonb_to_cstring(kcxt, (JsonbContainer *)data,
								 vlpos, kcxt->vlend, 0);
		if (sz < 0)
		{
			STROM_CPU_FALLBACK(kcxt, "out of memory");
			return false;
		}
		else
		{
			kcxt->vlpos = vlpos + sz;
			assert(kcxt->vlpos <= kcxt->vlend);

			result->expr_ops = &xpu_text_ops;
			result->length = sz;
			result->value = vlpos;
		}
	}
	else
	{
		STROM_ELOG(kcxt, "corrupted jsonb entry");
		return false;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_jsonb_object_field(XPU_PGFUNCTION_ARGS)
{
	xpu_jsonb_t	   *result = (xpu_jsonb_t *)__result;
	xpu_jsonb_t		json;
	xpu_text_t		key;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 2);
	assert(KEXP_IS_VALID(karg, jsonb));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &json) ||
		!xpu_jsonb_is_valid(kcxt, &json))
		return false;
	karg = KEXP_NEXT_ARG(karg);
	assert(KEXP_IS_VALID(karg, text));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &key) ||
		!xpu_text_is_valid(kcxt, &key))
		return false;

	if (XPU_DATUM_ISNULL(&json) || XPU_DATUM_ISNULL(&key))
	{
		result->expr_ops = NULL;
	}
	else
	{
		JsonbContainer *jc = (JsonbContainer *)json.value;
		uint32_t		jheader = __Fetch(&jc->header);

		if (!JsonContainerIsObject(jheader))
			result->expr_ops = NULL;
		else
		{
			uint32_t	count = JsonContainerSize(jheader);
			char	   *base = (char *)(jc->children + 2 * count);
			int32_t		index;

			index = findJsonbIndexFromObject(jc, key.value, key.length);
			if (index < 0 || index >= count)
				result->expr_ops = NULL;
			else
			{
				index += count;	/* index now points one of values, not keys */
				if (!extractJsonbItemFromContainer(kcxt, result,
												   jc, index, base))
					return false;
			}
		}
	}
	return result;
}

PUBLIC_FUNCTION(bool)
pgfn_jsonb_array_element(XPU_PGFUNCTION_ARGS)
{
	xpu_jsonb_t	   *result = (xpu_jsonb_t *)__result;
	xpu_jsonb_t		json;
	xpu_int4_t		key;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 2);
	assert(KEXP_IS_VALID(karg, jsonb));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &json) ||
		!xpu_jsonb_is_valid(kcxt, &json))
		return false;
	karg = KEXP_NEXT_ARG(karg);
	assert(KEXP_IS_VALID(karg, int4));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &key))
		return false;

	if (XPU_DATUM_ISNULL(&json) || XPU_DATUM_ISNULL(&key))
    {
        result->expr_ops = NULL;
    }
    else
    {
		JsonbContainer *jc = (JsonbContainer *)json.value;
		uint32_t		jheader = __Fetch(&jc->header);

		if (!JsonContainerIsArray(jheader))
			result->expr_ops = NULL;
		else
		{
			uint32_t	count = JsonContainerSize(jheader);
            char	   *base = (char *)(jc->children + count);	/* values */
			int32_t		index = key.value;

			if (index < 0)
				index += count;		/* index from the tail, if negative */
			if (index < 0 || index >= count)
				result->expr_ops = NULL;
			else if (!extractJsonbItemFromContainer(kcxt, result,
													jc, index, base))
				return false;
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_jsonb_object_field_text(XPU_PGFUNCTION_ARGS)
{
	xpu_text_t *result = (xpu_text_t *)__result;;
	xpu_jsonb_t	json;
	xpu_text_t	key;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 2);
	assert(KEXP_IS_VALID(karg, jsonb));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &json) ||
		!xpu_jsonb_is_valid(kcxt, &json))
		return false;
	karg = KEXP_NEXT_ARG(karg);
	assert(KEXP_IS_VALID(karg, text));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &key) ||
		!xpu_text_is_valid(kcxt, &key))
		return false;
	if (XPU_DATUM_ISNULL(&json) || XPU_DATUM_ISNULL(&key))
	{
		result->expr_ops = NULL;
	}
	else
	{
		JsonbContainer *jc = (JsonbContainer *)json.value;
		uint32_t		jheader = __Fetch(&jc->header);

		if (!JsonContainerIsObject(jheader))
			result->expr_ops = NULL;
		else
		{
			uint32_t	count = JsonContainerSize(jheader);
			char	   *base = (char *)(jc->children + 2 * count);
			int32_t		index;

			index = findJsonbIndexFromObject(jc, key.value, key.length);
			if (index < 0 || index >= count)
				result->expr_ops = NULL;
			else
			{
				index += count;	/* index now points one of values, not keys */
				if (!extractTextItemFromContainer(kcxt, result,
												  jc, index, base))
					return false;
			}
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_jsonb_array_element_text(XPU_PGFUNCTION_ARGS)
{
	xpu_text_t *result = (xpu_text_t *)__result;
	xpu_jsonb_t	json;
	xpu_int4_t	key;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 2);
	assert(KEXP_IS_VALID(karg, jsonb));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &json) ||
		!xpu_jsonb_is_valid(kcxt, &json))
		return false;
	karg = KEXP_NEXT_ARG(karg);
	assert(KEXP_IS_VALID(karg, int4));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &key))
		return false;
    if (XPU_DATUM_ISNULL(&json) || XPU_DATUM_ISNULL(&key))
	{
		result->expr_ops = NULL;
	}
	else
	{
		JsonbContainer *jc = (JsonbContainer *)json.value;
		uint32_t		jheader = __Fetch(&jc->header);

		if (!JsonContainerIsArray(jheader))
			result->expr_ops = NULL;
		else
		{
			uint32_t	count = JsonContainerSize(jheader);
            char	   *base = (char *)(jc->children + count);	/* values */
			int32_t		index = key.value;

			if (index < 0)
				index += count;		/* index from the tail, if negative */
			if (index < 0 || index >= count)
				result->expr_ops = NULL;
			else if (!extractTextItemFromContainer(kcxt, result,
												   jc, index, base))
				return false;
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_jsonb_exists(XPU_PGFUNCTION_ARGS)
{
	xpu_bool_t *result = (xpu_bool_t *)__result;
	xpu_jsonb_t	json;
	xpu_text_t	key;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 2);
	assert(KEXP_IS_VALID(karg, jsonb));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &json) ||
		!xpu_jsonb_is_valid(kcxt, &json))
		return false;
	karg = KEXP_NEXT_ARG(karg);
	assert(KEXP_IS_VALID(karg,text));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &key) ||
		!xpu_text_is_valid(kcxt, &key))
		return false;
	if (XPU_DATUM_ISNULL(&json) || XPU_DATUM_ISNULL(&key))
	{
		result->expr_ops = NULL;
	}
	else
	{
		int32_t	index;

		index = findJsonbIndexFromObject((JsonbContainer *)json.value,
										 key.value, key.length);
		result->expr_ops = &xpu_bool_ops;
		result->value = (index >= 0);
	}
	return true;
}

/*
 * Special shortcut for CoerceViaIO; fetch jsonb element as numeric values
 */
PUBLIC_FUNCTION(bool)
pgfn_jsonb_object_field_as_numeric(XPU_PGFUNCTION_ARGS)
{
	xpu_numeric_t  *result = (xpu_numeric_t *)__result;
	xpu_jsonb_t		json;
	xpu_text_t		key;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 2);
	assert(KEXP_IS_VALID(karg, jsonb));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &json) ||
		!xpu_jsonb_is_valid(kcxt, &json))
		return false;
	karg = KEXP_NEXT_ARG(karg);
	assert(KEXP_IS_VALID(karg,text));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &key) ||
		!xpu_text_is_valid(kcxt, &key))
		return false;
	if (!XPU_DATUM_ISNULL(&json) && !XPU_DATUM_ISNULL(&key))
	{
		JsonbContainer *jc = (JsonbContainer *)json.value;
		uint32_t		jheader = __Fetch(&jc->header);

		if (JsonContainerIsObject(jheader))
		{
			uint32_t	count = JsonContainerSize(jheader);
			char	   *base = (char *)(jc->children + 2 * count);
			int32_t		index;

			index = findJsonbIndexFromObject(jc, key.value, key.length);
			if (index >= 0 && index < count)
			{
				JEntry		entry;
				char	   *data;
				int32_t		datalen;
				const char *error_msg;

				index += count;		/* index now points values, not keys */
				entry = __Fetch(&jc->children[index]);
				if (JBE_ISNUMERIC(entry))
				{
					data = base + INTALIGN(getJsonbOffset(jc, index));
					datalen = getJsonbLength(jc, index);

					assert(VARSIZE_ANY(data) <= datalen);
					error_msg = __xpu_numeric_from_varlena(result, (varlena *)data);
					if (error_msg)
					{
						STROM_ELOG(kcxt, error_msg);
						return false;
					}
					return true;
				}
				else if (!JBE_ISNULL(entry))
				{
					/*
					 * Elsewhere, if item is neither numeric nor null,
					 * query eventually raises an error, because of value
					 * conversion problems.
					 */
					STROM_ELOG(kcxt, "referenced jsonb field is not numeric");
					return false;
				}
			}
		}
	}
	result->expr_ops = NULL;
	return true;
}

#define JSONB_OBJECT_FIELD_AS_INT_TEMPLATE(NAME,MIN_VALUE,MAX_VALUE)	\
	PUBLIC_FUNCTION(bool)												\
	pgfn_jsonb_object_field_as_##NAME(XPU_PGFUNCTION_ARGS)				\
	{																	\
		xpu_##NAME##_t *result = (xpu_##NAME##_t *)__result;			\
		xpu_numeric_t	num;											\
		int64_t			ival;											\
																		\
		if (!pgfn_jsonb_object_field_as_numeric(kcxt, kexp,				\
												(xpu_datum_t *)&num))	\
			return false;												\
		if (XPU_DATUM_ISNULL(&num))										\
			result->expr_ops = NULL;									\
		else															\
		{																\
			if (!__xpu_numeric_to_int64(kcxt, &ival, &num,				\
										MIN_VALUE, MAX_VALUE))			\
				return false;											\
			result->expr_ops = &xpu_##NAME##_ops;						\
			result->value = ival;										\
		}																\
		return true;													\
	}
JSONB_OBJECT_FIELD_AS_INT_TEMPLATE(int1, SCHAR_MIN,SCHAR_MAX)
JSONB_OBJECT_FIELD_AS_INT_TEMPLATE(int2, SHRT_MIN, SHRT_MAX)
JSONB_OBJECT_FIELD_AS_INT_TEMPLATE(int4, INT_MIN,  INT_MAX)
JSONB_OBJECT_FIELD_AS_INT_TEMPLATE(int8, LONG_MIN, LONG_MAX)

#define JSONB_OBJECT_FIELD_AS_FLOAT_TEMPLATE(NAME,__CAST)				\
	PUBLIC_FUNCTION(bool)												\
	pgfn_jsonb_object_field_as_##NAME(XPU_PGFUNCTION_ARGS)				\
	{																	\
		xpu_##NAME##_t *result = (xpu_##NAME##_t *)__result;			\
		xpu_numeric_t   num;											\
		float8_t		fval;											\
                                                                        \
		if (!pgfn_jsonb_object_field_as_numeric(kcxt, kexp,				\
												(xpu_datum_t *)&num))	\
			return false;                                               \
        if (XPU_DATUM_ISNULL(&num))                                     \
			result->expr_ops = NULL;                                    \
        else                                                            \
        {                                                               \
            if (!__xpu_numeric_to_fp64(kcxt, &fval, &num))				\
				return false;                                           \
            result->expr_ops = &xpu_##NAME##_ops;                       \
            result->value = __CAST(fval);								\
        }                                                               \
        return true;                                                    \
	}
JSONB_OBJECT_FIELD_AS_FLOAT_TEMPLATE(float2, __to_fp16)
JSONB_OBJECT_FIELD_AS_FLOAT_TEMPLATE(float4, __to_fp32)
JSONB_OBJECT_FIELD_AS_FLOAT_TEMPLATE(float8, __to_fp64)

PUBLIC_FUNCTION(bool)
pgfn_jsonb_array_element_as_numeric(XPU_PGFUNCTION_ARGS)
{
	xpu_numeric_t  *result = (xpu_numeric_t *)__result;
	xpu_jsonb_t		json;
	xpu_int4_t		key;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 2);
	assert(KEXP_IS_VALID(karg, jsonb));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &json) ||
		!xpu_jsonb_is_valid(kcxt, &json))
		return false;
	karg = KEXP_NEXT_ARG(karg);
	assert(KEXP_IS_VALID(karg, int4));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &key))
		return false;
	if (!XPU_DATUM_ISNULL(&json) && !XPU_DATUM_ISNULL(&key))
	{
		JsonbContainer *jc = (JsonbContainer *)json.value;
		uint32_t		jheader = __Fetch(&jc->header);

		if (JsonContainerIsArray(jheader))
		{
			uint32_t	count = JsonContainerSize(jheader);
			char	   *base = (char *)(jc->children + count);	/* values */
			int32_t		index = key.value;

			if (index < 0)
				index += count;		/* index from the tail, if negative */
			if (index >= 0 && index < count)
			{
				JEntry		entry;
				char	   *data;
				int32_t		datalen;
				const char *error_msg;

				entry = __Fetch(&jc->children[index]);
				if (JBE_ISNUMERIC(entry))
				{
					data = base + INTALIGN(getJsonbOffset(jc, index));
					datalen = getJsonbLength(jc, index);

					assert(VARSIZE_ANY(data) <= datalen);
					error_msg = __xpu_numeric_from_varlena(result, (varlena *)data);
					if (error_msg)
					{
						STROM_ELOG(kcxt, error_msg);
						return false;
					}
					return true;
				}
				else if (!JBE_ISNULL(entry))
				{
					/*
					 * Elsewhere, if item is neither numeric nor null,
					 * query eventually raises an error, because of value
					 * conversion problems.
					 */
					STROM_ELOG(kcxt, "referenced jsonb item is not numeric");
					return false;
				}
			}
		}
	}
	result->expr_ops = NULL;
	return true;
}

#define JSONB_ARRAY_ELEMENT_AS_INT_TEMPLATE(NAME,MIN_VALUE,MAX_VALUE)	\
	PUBLIC_FUNCTION(bool)												\
	pgfn_jsonb_array_element_as_##NAME(XPU_PGFUNCTION_ARGS)				\
	{																	\
		xpu_##NAME##_t *result = (xpu_##NAME##_t *)__result;			\
		xpu_numeric_t	num;											\
		int64_t			ival;											\
																		\
		if (!pgfn_jsonb_array_element_as_numeric(kcxt, kexp,			\
												 (xpu_datum_t *)&num))	\
			return false;												\
		if (XPU_DATUM_ISNULL(&num))										\
			result->expr_ops = NULL;									\
		else															\
		{																\
			if (!__xpu_numeric_to_int64(kcxt, &ival, &num,				\
										MIN_VALUE, MAX_VALUE))			\
				return false;											\
			result->expr_ops = &xpu_##NAME##_ops;						\
			result->value = ival;										\
		}																\
		return true;													\
	}
JSONB_ARRAY_ELEMENT_AS_INT_TEMPLATE(int1, SCHAR_MIN,SCHAR_MAX)
JSONB_ARRAY_ELEMENT_AS_INT_TEMPLATE(int2, SHRT_MIN, SHRT_MAX)
JSONB_ARRAY_ELEMENT_AS_INT_TEMPLATE(int4, INT_MIN,  INT_MAX)
JSONB_ARRAY_ELEMENT_AS_INT_TEMPLATE(int8, LONG_MIN, LONG_MAX)

#define JSONB_ARRAY_ELEMENT_AS_FLOAT_TEMPLATE(NAME,__CAST)				\
	PUBLIC_FUNCTION(bool)												\
	pgfn_jsonb_array_element_as_##NAME(XPU_PGFUNCTION_ARGS)				\
	{																	\
		xpu_##NAME##_t *result = (xpu_##NAME##_t *)__result;			\
		xpu_numeric_t   num;											\
		float8_t		fval;											\
                                                                        \
		if (!pgfn_jsonb_array_element_as_numeric(kcxt, kexp,			\
												 (xpu_datum_t *)&num))	\
			return false;                                               \
        if (XPU_DATUM_ISNULL(&num))                                     \
			result->expr_ops = NULL;                                    \
        else                                                            \
        {                                                               \
            if (!__xpu_numeric_to_fp64(kcxt, &fval, &num))				\
				return false;                                           \
            result->expr_ops = &xpu_##NAME##_ops;                       \
            result->value = __CAST(fval);								\
        }                                                               \
        return true;                                                    \
	}
JSONB_ARRAY_ELEMENT_AS_FLOAT_TEMPLATE(float2, __to_fp16)
JSONB_ARRAY_ELEMENT_AS_FLOAT_TEMPLATE(float4, __to_fp32)
JSONB_ARRAY_ELEMENT_AS_FLOAT_TEMPLATE(float8, __to_fp64)
