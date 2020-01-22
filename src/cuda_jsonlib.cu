/*
 * cuda_jsonlib.cu
 *
 * Collection of text functions for CUDA GPU devices
 * --
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
#include "cuda_common.h"
#include "cuda_jsonlib.h"

typedef cl_uint		JEntry;

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
	cl_uint		header;
	JEntry		children[FLEXIBLE_ARRAY_MEMBER];
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

STATIC_INLINE(cl_uint)
getJsonbOffset(const JsonbContainer *jc,		/* may not be aligned */
			   int index)
{
	cl_uint		offset = 0;
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

STATIC_INLINE(cl_uint)
getJsonbLength(const JsonbContainer *jc,		/* may not be aligned */
			   int index)
{
	JEntry		entry = __Fetch(&jc->children[index]);
	cl_uint		off;
	cl_uint		len;

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
STATIC_FUNCTION(cl_uint)
__pg_jsonb_comp_hash(kern_context *kcxt, JsonbContainer *jc)
{
	cl_uint		hash = 0;
	cl_uint		jheader = __Fetch(&jc->header);
	cl_uint		j, nitems = JsonContainerSize(jheader);
	char	   *base;
	char	   *data;
	cl_uint		datalen;

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
		cl_uint		index = j;
		cl_uint		temp;
		JEntry		entry;

		/* hash value for key */
		if (JsonContainerIsObject(jheader))
		{
			entry = __Fetch(&jc->children[index]);
			if (!JBE_ISSTRING(entry))
			{
				STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
							  "corrupted jsonb entry");
				return 0;
			}
			data = base + getJsonbOffset(jc, index);
			datalen = getJsonbLength(jc, index);
			temp = pg_hash_any((cl_uchar *)data, datalen);
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
			temp = pg_hash_any((cl_uchar *)data, datalen);
		}
		else if (JBE_ISNUMERIC(entry))
		{
			pg_numeric_t	numeric;

			data = base + INTALIGN(getJsonbOffset(jc, index));
			numeric = pg_numeric_from_varlena(kcxt, (varlena *) data);
			temp = pg_comp_hash(kcxt, numeric);
		}
		else if (JBE_ISBOOL_TRUE(entry))
			temp = 0x02;
		else if (JBE_ISBOOL_FALSE(entry))
			temp = 0x04;
		else if (JBE_ISCONTAINER(entry))
		{
			data = base + INTALIGN(getJsonbOffset(jc, index));
			temp = __pg_jsonb_comp_hash(kcxt, (JsonbContainer *)data);
		}
		else
		{
			STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
						  "corrupted jsonb entry");
			return 0;
		}
		hash = ((hash << 1) | (hash >> 31)) ^ temp;
	}
	return hash;
}

DEVICE_FUNCTION(cl_uint)
pg_comp_hash(kern_context *kcxt, pg_jsonb_t datum)
{
	char	   *jdata;
	cl_int		jlen;

	if (!pg_varlena_datum_extract(kcxt, datum, &jdata, &jlen))
		return 0;
	return __pg_jsonb_comp_hash(kcxt, (JsonbContainer *)jdata);
}

STATIC_INLINE(cl_int)
compareJsonbStringValue(const char *s1, cl_int len1,
						const char *s2, cl_int len2)
{
	if (len1 > len2)
		return 1;
	else if (len1 < len2)
		return -1;
	return __memcmp(s1, s2, len1);
}

STATIC_FUNCTION(cl_int)
findJsonbIndexFromObject(JsonbContainer *jc,	/* may not be aligned */
						 char *key, cl_int keylen)
{
	cl_uint		jheader = __Fetch(&jc->header);

	if (JsonContainerIsObject(jheader))
	{
		cl_uint		count = JsonContainerSize(jheader);
		cl_uint		stopLow = 0;
		cl_uint		stopHigh = count;
		char	   *base = (char *)(jc->children + 2 * count);

		/* Binary search on object/pair keys *only* */
		while (stopLow < stopHigh)
		{
			cl_uint		stopMiddle = stopLow + (stopHigh - stopLow) / 2;
			cl_int		diff;
			char	   *name;
			cl_int		namelen;

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

STATIC_FUNCTION(pg_jsonb_t)
extractJsonbItemFromContainer(kern_context *kcxt,
							  JsonbContainer *jc,	/* may not be aligned */
							  cl_int index, char *base)
{
	cl_uint		jheader = __Fetch(&jc->header);
	JEntry		entry;
	pg_jsonb_t	res;

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
		cl_uint		off = offsetof(JsonbContainer, children[1]);
		cl_uint		sz = VARHDRSZ + off;
		char	   *data = NULL;
		cl_uint		datalen = 0;
		varlena	   *vl;

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
		vl = (varlena *)kern_context_alloc(kcxt, sz);
		if (!vl)
		{
			STROM_CPU_FALLBACK(kcxt, ERRCODE_OUT_OF_MEMORY,
							   "out of memory");
			res.isnull = true;
		}
		else
		{
			JsonbContainer *r = (JsonbContainer *)vl->vl_dat;

			r->header = JB_FARRAY | JB_FSCALAR | 1;
			if (!data)
			{
				entry = (entry & JENTRY_TYPEMASK);
			}
			else
			{
				memcpy((char *)r + off, data, datalen);
				entry = (entry & JENTRY_TYPEMASK) | datalen;
			}
			r->children[0] = entry;
			SET_VARSIZE(vl, sz);

			res.isnull = false;
			res.length = -1;
			res.value  = (char *)vl;
		}
	}
	else if (JBE_ISCONTAINER(entry))
	{
		char	   *data = base + INTALIGN(getJsonbOffset(jc, index));
		cl_uint		datalen = getJsonbLength(jc, index);

		res.isnull = false;
		res.length = datalen;
		res.value  = data;
	}
	else
	{
		STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
					  "corrupted jsonb entry");
		res.isnull = true;
	}
	return res;
}

STATIC_INLINE(cl_int)
escape_json_cstring(const char *str, cl_int strlen, char *buf, char *endp)
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

	return (cl_int)(pos - buf);
}

STATIC_FUNCTION(cl_int)
pg_jsonb_to_cstring(kern_context *kcxt,
					JsonbContainer *jc,		/* may not be aligned */
					char *buf, char *endp, int depth)
{
	cl_uint		jheader;
	cl_uint		j, count;
	JEntry		entry;
	char	   *base;
	char	   *pos = buf;
	char	   *data;
	cl_int		datalen, sz;

	if (depth > 8)
	{
		STROM_CPU_FALLBACK(kcxt, ERRCODE_STROM_RECURSION_TOO_DEEP,
						   "too deep recursive function call");
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
				STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
							  "corrupter jsonb entry");
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
			STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
						  "corrupted jsonb entry");
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

STATIC_FUNCTION(pg_text_t)
extractTextItemFromContainer(kern_context *kcxt,
							 JsonbContainer *jc,	/* may not be aligned */
							 cl_int index, char *base)
{
	cl_uint		jheader = __Fetch(&jc->header);
	JEntry		entry;
	pg_text_t	res;

	assert(JsonContainerIsObject(jheader) ||
		   JsonContainerIsArray(jheader));
	/* extract jsonb object value */
	entry = __Fetch(&jc->children[index]);
	if (JBE_ISNULL(entry))
		res.isnull = true;
	else if (JBE_ISSTRING(entry))
	{
		char	   *data = base + getJsonbOffset(jc, index);
		cl_uint		datalen = getJsonbLength(jc, index);

		res.isnull = false;
		res.length = datalen;
		res.value  = data;
	}
	else if (JBE_ISNUMERIC(entry))
	{
		char	   *data = base + INTALIGN(getJsonbOffset(jc, index));
		varlena	   *vl;
		cl_int		sz;

		vl = (varlena *)INTALIGN(kcxt->vlpos);
		sz = pg_numeric_to_cstring(kcxt, (varlena *)data,
								   vl->vl_dat, kcxt->vlend);
		if (sz < 0)
		{
            res.isnull = true;
			STROM_CPU_FALLBACK(kcxt, ERRCODE_OUT_OF_MEMORY, "out of memory");
		}
		else
		{
			kcxt->vlpos = vl->vl_dat + sz;
			assert(kcxt->vlpos <= kcxt->vlend);
			SET_VARSIZE(vl, VARHDRSZ + sz);

			res.isnull = false;
			res.length = -1;
			res.value  = (char *)vl;
		}
	}
	else if (JBE_ISBOOL_TRUE(entry))
	{
		res.isnull = false;
		res.length = -1;
		res.value  = (char *)"\x0b" "true";
	}
	else if (JBE_ISBOOL_FALSE(entry))
	{
		res.isnull = false;
		res.length = -1;
		res.value  = (char *)"\x0d" "false";
	}
	else if (JBE_ISCONTAINER(entry))
	{
		char	   *data = base + INTALIGN(getJsonbOffset(jc, index));
		varlena	   *vl;
		cl_int		sz;

		vl = (varlena *)INTALIGN(kcxt->vlpos);
		sz = pg_jsonb_to_cstring(kcxt, (JsonbContainer *)data,
								 vl->vl_dat, kcxt->vlend, 0);
		if (sz < 0)
		{
			res.isnull = true;
			STROM_CPU_FALLBACK(kcxt, ERRCODE_OUT_OF_MEMORY, "out of memory");
		}
		else
		{
			kcxt->vlpos = vl->vl_dat + sz;
			assert(kcxt->vlpos <= kcxt->vlend);
			SET_VARSIZE(vl, VARHDRSZ + sz);

			res.isnull = false;
			res.length = -1;
			res.value  = (char *)vl;
		}
	}
	else
	{
		res.isnull = true;
		STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
					  "corrupted jsonb entry");
	}
	return res;
}

DEVICE_FUNCTION(pg_jsonb_t)
pgfn_jsonb_object_field(kern_context *kcxt,
						pg_jsonb_t arg1, pg_text_t arg2)
{
	pg_jsonb_t	result;
	char	   *jdata;		/* jsonb */
	char	   *kdata;		/* key text */
	cl_int		jlen, klen;

	if (!pg_varlena_datum_extract(kcxt, arg1, &jdata, &jlen) ||
		!pg_varlena_datum_extract(kcxt, arg2, &kdata, &klen))
	{
		result.isnull = true;
	}
	else
	{
		JsonbContainer *jc = (JsonbContainer *)jdata;
		cl_uint		jheader = __Fetch(&jc->header);

		if (!JsonContainerIsObject(jheader))
			result.isnull = true;
		else
		{
			cl_uint		count = JsonContainerSize(jheader);
			char	   *base = (char *)(jc->children + 2 * count);
			cl_int		index;

			index = findJsonbIndexFromObject(jc, kdata, klen);
			if (index < 0 || index >= count)
				result.isnull = true;
			else
			{
				index += count;	/* index now points one of values, not keys */
				result = extractJsonbItemFromContainer(kcxt, jc, index, base);
			}
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_jsonb_t)
pgfn_jsonb_array_element(kern_context *kcxt,
						 pg_jsonb_t arg1, pg_int4_t arg2)
{
	pg_jsonb_t	result;
	char	   *jdata;
	cl_int		jlen;

	if (!pg_varlena_datum_extract(kcxt, arg1, &jdata, &jlen) || arg2.isnull)
		result.isnull = true;
	else
	{
		JsonbContainer *jc = (JsonbContainer *)jdata;
		cl_uint		jheader = __Fetch(&jc->header);

		if (!JsonContainerIsArray(jheader))
			result.isnull = true;
		else
		{
			cl_uint		count = JsonContainerSize(jheader);
            char	   *base = (char *)(jc->children + count);	/* values */
			cl_int		index = arg2.value;

			if (index < 0)
				index += count;		/* index from the tail, if negative */
			if (index < 0 || index >= count)
				result.isnull = true;
			else
				result = extractJsonbItemFromContainer(kcxt, jc, index, base);
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_text_t)
pgfn_jsonb_object_field_text(kern_context *kcxt,
							 pg_jsonb_t arg1, pg_text_t arg2)
{
	pg_text_t	result;
	char	   *jdata;		/* jsonb */
	char	   *kdata;		/* key text */
	cl_int		jlen, klen;

	if (!pg_varlena_datum_extract(kcxt, arg1, &jdata, &jlen) ||
		!pg_varlena_datum_extract(kcxt, arg2, &kdata, &klen))
	{
		result.isnull = true;
	}
	else
	{
		JsonbContainer *jc = (JsonbContainer *)jdata;
		cl_uint		jheader = __Fetch(&jc->header);

		if (!JsonContainerIsObject(jheader))
			result.isnull = true;
		else
		{
			cl_uint		count = JsonContainerSize(jheader);
			char	   *base = (char *)(jc->children + 2 * count);
			cl_int		index;

			index = findJsonbIndexFromObject(jc, kdata, klen);
			if (index < 0 || index >= count)
				result.isnull = true;
			else
			{
				index += count;	/* index now points one of values, not keys */
				result = extractTextItemFromContainer(kcxt, jc, index, base);
			}
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_text_t)
pgfn_jsonb_array_element_text(kern_context *kcxt,
							  pg_jsonb_t arg1, pg_int4_t arg2)
{
	pg_text_t	result;
	char	   *jdata;
	cl_int		jlen;

	if (!pg_varlena_datum_extract(kcxt, arg1, &jdata, &jlen) || arg2.isnull)
		result.isnull = true;
	else
	{
		JsonbContainer *jc = (JsonbContainer *)jdata;
		cl_uint		jheader = __Fetch(&jc->header);

		if (!JsonContainerIsArray(jheader))
			result.isnull = true;
		else
		{
			cl_uint		count = JsonContainerSize(jheader);
            char	   *base = (char *)(jc->children + count);	/* values */
			cl_int		index = arg2.value;

			if (index < 0)
				index += count;		/* index from the tail, if negative */
			if (index < 0 || index >= count)
				result.isnull = true;
			else
                result = extractTextItemFromContainer(kcxt, jc, index, base);
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_jsonb_exists(kern_context *kcxt,
				  pg_jsonb_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;
	char	   *jdata;		/* jsonb */
	char	   *kdata;		/* key text */
	cl_int		jlen, klen;
	cl_int		index;

	if (!pg_varlena_datum_extract(kcxt, arg1, &jdata, &jlen) ||
		!pg_varlena_datum_extract(kcxt, arg2, &kdata, &klen))
	{
		result.isnull = true;
	}
	else
	{
		index = findJsonbIndexFromObject((JsonbContainer *)jdata,
										 kdata, klen);
		result.isnull = false;
		result.value  = (index >= 0);
	}
	return result;
}

/*
 * Special shortcut for CoerceViaIO; fetch jsonb element as numeric values
 */
DEVICE_FUNCTION(pg_numeric_t)
pgfn_jsonb_object_field_as_numeric(kern_context *kcxt,
								   pg_jsonb_t arg1, pg_text_t arg2)
{
	pg_numeric_t result;
	char	   *jdata;		/* jsonb */
	char	   *kdata;		/* key text */
	cl_int		jlen, klen;

	if (!pg_varlena_datum_extract(kcxt, arg1, &jdata, &jlen) ||
		!pg_varlena_datum_extract(kcxt, arg2, &kdata, &klen))
	{
		result.isnull = true;
	}
	else
	{
		JsonbContainer *jc = (JsonbContainer *)jdata;
		cl_uint		jheader = __Fetch(&jc->header);

		if (JsonContainerIsObject(jheader))
		{
			cl_uint		count = JsonContainerSize(jheader);
			char	   *base = (char *)(jc->children + 2 * count);
			cl_int		index;

			index = findJsonbIndexFromObject(jc, kdata, klen);
			if (index >= 0 && index < count)
			{
				JEntry		entry;
				char	   *data;
				cl_int		datalen;

				index += count;		/* index now points values, not keys */
				entry = __Fetch(&jc->children[index]);
				if (JBE_ISNUMERIC(entry))
				{
					data = base + INTALIGN(getJsonbOffset(jc, index));
					datalen = getJsonbLength(jc, index);

					assert(VARSIZE_ANY(data) <= datalen);
					return pg_numeric_from_varlena(kcxt, (varlena *)data);
				}
				else if (!JBE_ISNULL(entry))
				{
					/*
					 * Elsewhere, if item is neither numeric nor null,
					 * query eventually raises an error, because of value
					 * conversion problems.
					 */
					STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
								  "corrupted jsonb entry");
				}
			}
		}
	}
	result.isnull = true;
	return result;
}

DEVICE_FUNCTION(pg_int2_t)
pgfn_jsonb_object_field_as_int2(kern_context *kcxt,
								pg_jsonb_t arg1, pg_text_t arg2)
{
	pg_numeric_t	num;

	num = pgfn_jsonb_object_field_as_numeric(kcxt, arg1, arg2);
	return pgfn_numeric_int2(kcxt, num);
}

DEVICE_FUNCTION(pg_int4_t)
pgfn_jsonb_object_field_as_int4(kern_context *kcxt,
								pg_jsonb_t arg1, pg_text_t arg2)
{
	pg_numeric_t	num;

	num = pgfn_jsonb_object_field_as_numeric(kcxt, arg1, arg2);
	return pgfn_numeric_int4(kcxt, num);
}

DEVICE_FUNCTION(pg_int8_t)
pgfn_jsonb_object_field_as_int8(kern_context *kcxt,
								pg_jsonb_t arg1, pg_text_t arg2)
{
	pg_numeric_t	num;

	num = pgfn_jsonb_object_field_as_numeric(kcxt, arg1, arg2);
	return pgfn_numeric_int8(kcxt, num);
}

DEVICE_FUNCTION(pg_float4_t)
pgfn_jsonb_object_field_as_float4(kern_context *kcxt,
								  pg_jsonb_t arg1, pg_text_t arg2)
{
	pg_numeric_t	num;

	num = pgfn_jsonb_object_field_as_numeric(kcxt, arg1, arg2);
	return pgfn_numeric_float4(kcxt, num);
}

DEVICE_FUNCTION(pg_float8_t)
pgfn_jsonb_object_field_as_float8(kern_context *kcxt,
								  pg_jsonb_t arg1, pg_text_t arg2)
{
	pg_numeric_t	num;

	num = pgfn_jsonb_object_field_as_numeric(kcxt, arg1, arg2);
	return pgfn_numeric_float8(kcxt, num);
}

DEVICE_FUNCTION(pg_numeric_t)
pgfn_jsonb_array_element_as_numeric(kern_context *kcxt,
									pg_jsonb_t arg1, pg_int4_t arg2)
{
	pg_numeric_t result;
	char	   *jdata;
	cl_int		jlen;

	if (pg_varlena_datum_extract(kcxt, arg1, &jdata, &jlen) && !arg2.isnull)
	{
		JsonbContainer *jc = (JsonbContainer *)jdata;
		cl_uint		jheader = __Fetch(&jc->header);

		if (JsonContainerIsArray(jheader))
		{
			cl_uint		count = JsonContainerSize(jheader);
			char	   *base = (char *)(jc->children + count);	/* values */
			cl_int		index = arg2.value;

			if (index < 0)
				index += count;		/* index from the tail, if negative */
			if (index >= 0 && index < count)
			{
				JEntry		entry;
				char	   *data;
				cl_int		datalen;

				entry = __Fetch(&jc->children[index]);
				if (JBE_ISNUMERIC(entry))
				{
					data = base + INTALIGN(getJsonbOffset(jc, index));
					datalen = getJsonbLength(jc, index);

					assert(VARSIZE_ANY(data) <= datalen);
					return pg_numeric_from_varlena(kcxt, (varlena *)data);
				}
				else if (!JBE_ISNULL(entry))
				{
					/*
					 * Elsewhere, if item is neither numeric nor null,
					 * query eventually raises an error, because of value
					 * conversion problems.
					 */
					STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
								  "corrupted jsonb entry");
				}
			}
		}
	}
	result.isnull = true;
	return result;
}

DEVICE_FUNCTION(pg_int2_t)
pgfn_jsonb_array_element_as_int2(kern_context *kcxt,
								 pg_jsonb_t arg1, pg_int4_t arg2)
{
	pg_numeric_t	num;

	num = pgfn_jsonb_array_element_as_numeric(kcxt, arg1, arg2);
	return pgfn_numeric_int2(kcxt, num);
}

DEVICE_FUNCTION(pg_int4_t)
pgfn_jsonb_array_element_as_int4(kern_context *kcxt,
								 pg_jsonb_t arg1, pg_int4_t arg2)
{
	pg_numeric_t	num;

	num = pgfn_jsonb_array_element_as_numeric(kcxt, arg1, arg2);
	return pgfn_numeric_int4(kcxt, num);
}

DEVICE_FUNCTION(pg_int8_t)
pgfn_jsonb_array_element_as_int8(kern_context *kcxt,
								 pg_jsonb_t arg1, pg_int4_t arg2)
{
	pg_numeric_t	num;

	num = pgfn_jsonb_array_element_as_numeric(kcxt, arg1, arg2);
	return pgfn_numeric_int8(kcxt, num);
}

DEVICE_FUNCTION(pg_float4_t)
pgfn_jsonb_array_element_as_float4(kern_context *kcxt,
								   pg_jsonb_t arg1, pg_int4_t arg2)
{
	pg_numeric_t	num;

	num = pgfn_jsonb_array_element_as_numeric(kcxt, arg1, arg2);
	return pgfn_numeric_float4(kcxt, num);
}

DEVICE_FUNCTION(pg_float8_t)
pgfn_jsonb_array_element_as_float8(kern_context *kcxt,
								   pg_jsonb_t arg1, pg_int4_t arg2)
{
	pg_numeric_t	num;

	num = pgfn_jsonb_array_element_as_numeric(kcxt, arg1, arg2);
	return pgfn_numeric_float8(kcxt, num);
}
