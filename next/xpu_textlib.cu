/*
 * xpu_textlib.cu
 *
 * Collection of text functions and operators for both of GPU and DPU
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 *
 */
#include "xpu_common.h"

/*
 * bpchar type handlers
 */
INLINE_FUNCTION(int)
bpchar_truelen(const char *s, int len)
{
	int		i;

	for (i = len - 1; i >= 0; i--)
	{
		if (s[i] != ' ')
			break;
	}
	return i + 1;
}

INLINE_FUNCTION(bool)
xpu_bpchar_is_valid(kern_context *kcxt, const xpu_bpchar_t *arg)
{
	if (arg->length < 0)
	{
		STROM_CPU_FALLBACK(kcxt, "bpchar datum is compressed or external");
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_bpchar_datum_ref(kern_context *kcxt,
					 xpu_datum_t *__result,
					 const void *addr)
{
	xpu_bpchar_t *result = (xpu_bpchar_t *)__result;

	if (!addr)
		result->isnull = true;
	else if (VARATT_IS_EXTERNAL(addr) || VARATT_IS_COMPRESSED(addr))
	{
		result->isnull = false;
		result->value  = (const char *)addr;
		result->length = -1;
	}
	else
	{
		result->isnull = false;
		result->value  = VARDATA_ANY(addr);
		result->length = bpchar_truelen(result->value, VARSIZE_ANY_EXHDR(addr));
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_bpchar_arrow_ref(kern_context *kcxt,
					 xpu_datum_t *__result,
					 const kern_colmeta *cmeta,
					 const void *addr, int len)
{
	xpu_bpchar_t *result = (xpu_bpchar_t *)__result;

	if (!addr)
		result->isnull = true;
	else
	{
		result->isnull = false;
		result->value  = (const char *)addr;
		result->length = len;
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_bpchar_arrow_move(kern_context *kcxt,
					  char *buffer,
					  const kern_colmeta *cmeta,
					  const void *addr, int len)
{
	if (!addr)
		return 0;
	if (buffer)
	{
		memcpy(buffer + VARHDRSZ, addr, len);
		SET_VARSIZE(buffer, VARHDRSZ + len);
	}
	return VARHDRSZ + len;
}

STATIC_FUNCTION(int)
xpu_bpchar_datum_store(kern_context *kcxt,
					   char *buffer,
					   const xpu_datum_t *__arg)
{
	const xpu_bpchar_t *arg = (const xpu_bpchar_t *)__arg;
	int		sz;

	if (arg->isnull)
		return 0;
	if (arg->length < 0)
	{
		sz = VARSIZE_ANY(arg->value);
		if (buffer)
			memcpy(buffer, arg->value, sz);
	}
	else
	{
		sz = VARHDRSZ + arg->length;
		if (buffer)
		{
			memcpy(buffer + VARHDRSZ, arg->value, arg->length);
			SET_VARSIZE(buffer, sz);
		}
	}
	return sz;
}

STATIC_FUNCTION(bool)
xpu_bpchar_datum_hash(kern_context*kcxt,
					  uint32_t *p_hash,
					  const xpu_datum_t *__arg)
{
	const xpu_bpchar_t *arg = (const xpu_bpchar_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else if (xpu_bpchar_is_valid(kcxt, arg))
		*p_hash = pg_hash_any(arg->value, arg->length);
	else
		return false;
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(bpchar, false, 4, -1);

/*
 * xpu_text_t device type handler
 */
INLINE_FUNCTION(bool)
xpu_text_is_valid(kern_context *kcxt, const xpu_text_t *arg)
{
	if (arg->length < 0)
	{
		STROM_CPU_FALLBACK(kcxt, "text datum is compressed or external");
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_text_datum_ref(kern_context *kcxt,
				   xpu_datum_t *__result,
				   const void *addr)
{
	xpu_text_t *result = (xpu_text_t *)__result;

	if (!addr)
		result->isnull = true;
	else if (VARATT_IS_EXTERNAL(addr) || VARATT_IS_COMPRESSED(addr))
	{
		result->isnull = false;
		result->value  = (const char *)addr;
		result->length = -1;
	}
	else
	{
		result->isnull = false;
		result->value  = VARDATA_ANY(addr);
		result->length = VARSIZE_ANY(addr);
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_text_arrow_ref(kern_context *kcxt,
				   xpu_datum_t *__result,
				   const kern_colmeta *cmeta,
				   const void *addr, int len)
{
	xpu_text_t *result = (xpu_text_t *)__result;

	if (!addr)
		result->isnull = true;
	else
	{
		result->isnull = false;
		result->value  = (const char *)addr;
		result->length = len;
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_text_arrow_move(kern_context *kcxt,
					char *buffer,
					const kern_colmeta *cmeta,
					const void *addr, int len)
{
	if (!addr)
		return 0;
	if (buffer)
	{
		memcpy(buffer + VARHDRSZ, addr, len);
		SET_VARSIZE(buffer, VARHDRSZ + len);
	}
	return VARHDRSZ + len;
}

STATIC_FUNCTION(int)
xpu_text_datum_store(kern_context *kcxt,
					 char *buffer,
					 const xpu_datum_t *__arg)
{
	const xpu_text_t *arg = (const xpu_text_t *)__arg;
	int		sz;

	if (arg->isnull)
		return 0;
	if (arg->length < 0)
	{
		sz = VARSIZE_ANY(arg->value);
		if (buffer)
			memcpy(buffer, arg->value, sz);
	}
	else
	{
		sz = VARHDRSZ + arg->length;
		if (buffer)
		{
			memcpy(buffer + VARHDRSZ, arg->value, arg->length);
			SET_VARSIZE(buffer, sz);
		}
	}
	return sz;
}

STATIC_FUNCTION(bool)
xpu_text_datum_hash(kern_context *kcxt,
					uint32_t *p_hash,
					const xpu_datum_t *__arg)
{
	const xpu_text_t *arg = (const xpu_text_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else if (!xpu_text_is_valid(kcxt, arg))
		*p_hash = pg_hash_any(arg->value, arg->length);
	else
		return false;
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(text, false, 4, -1);

/*
 * xpu_bytea_t device type handler
 */
INLINE_FUNCTION(bool)
xpu_bytea_is_valid(kern_context *kcxt, const xpu_bytea_t *arg)
{
	if (arg->length < 0)
	{
		STROM_CPU_FALLBACK(kcxt, "bytea datum is compressed or external");
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_bytea_datum_ref(kern_context *kcxt,
				   xpu_datum_t *__result,
					const void *addr)
{
	xpu_bytea_t *result = (xpu_bytea_t *)__result;

	if (!addr)
		result->isnull = true;
	else if (VARATT_IS_EXTERNAL(addr) || VARATT_IS_COMPRESSED(addr))
	{
		result->isnull = false;
		result->value  = (const char *)addr;
		result->length = -1;
	}
	else
	{
		result->isnull = false;
		result->value  = VARDATA_ANY(addr);
		result->length = VARSIZE_ANY(addr);
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_bytea_arrow_ref(kern_context *kcxt,
					xpu_datum_t *__result,
					const kern_colmeta *cmeta,
					const void *addr, int len)
{
	xpu_bytea_t *result = (xpu_bytea_t *)__result;

	if (!addr)
		result->isnull = true;
	else
	{
		result->isnull = false;
		result->value  = (const char *)addr;
		result->length = len;
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_bytea_arrow_move(kern_context *kcxt,
					 char *buffer,
					 const kern_colmeta *cmeta,
					 const void *addr, int len)
{
	if (!addr)
		return 0;
	if (buffer)
	{
		memcpy(buffer + VARHDRSZ, addr, len);
		SET_VARSIZE(buffer, VARHDRSZ + len);
	}
	return VARHDRSZ + len;
}

STATIC_FUNCTION(int)
xpu_bytea_datum_store(kern_context *kcxt,
					  char *buffer,
					  const xpu_datum_t *__arg)
{
	const xpu_bytea_t *arg = (const xpu_bytea_t *)__arg;
	int		sz;

	if (arg->isnull)
		return 0;
	if (arg->length < 0)
	{
		sz = VARSIZE_ANY(arg->value);
		if (buffer)
			memcpy(buffer, arg->value, sz);
	}
	else
	{
		sz = VARHDRSZ + arg->length;
		if (buffer)
		{
			memcpy(buffer + VARHDRSZ, arg->value, arg->length);
			SET_VARSIZE(buffer, sz);
		}
	}
	return sz;
}

STATIC_FUNCTION(bool)
xpu_bytea_datum_hash(kern_context *kcxt,
					 uint32_t *p_hash,
					 const xpu_datum_t *__arg)
{
	const xpu_bytea_t *arg = (const xpu_bytea_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else if (!xpu_bytea_is_valid(kcxt, arg))
		*p_hash = pg_hash_any(arg->value, arg->length);
	else
		return false;
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(bytea, false, 4, -1);

/*
 * Bpchar functions
 */
STATIC_FUNCTION(bool)
__bpchar_compare(kern_context *kcxt,
				 int *p_status,
				 const xpu_bpchar_t *str1,
				 const xpu_bpchar_t *str2)
{
	const char *s1 = str1->value;
	const char *s2 = str2->value;
	int		len;

	if (!xpu_bpchar_is_valid(kcxt, str1) ||
		!xpu_bpchar_is_valid(kcxt, str2))
		return false;

	len = Min(str1->length, str2->length);
	while (len > 0)
	{
		if (*s1 < *s2)
		{
			*p_status = -1;
			return true;
		}
		if (*s1 > *s2)
		{
			*p_status = 1;
			return true;
		}
		s1++;
		s2++;
		len--;
	}
	if (str1->length == str2->length)
		*p_status = 0;
	else if (str1->length > str2->length)
		*p_status = 1;
	else
		*p_status = -1;
	return true;
}

#define PG_BPCHAR_COMPARE_TEMPLATE(NAME,OPER)						\
	PUBLIC_FUNCTION(bool)											\
	pgfn_bpchar##NAME(XPU_PGFUNCTION_ARGS)							\
	{																\
		xpu_bool_t	   *result = (xpu_bool_t *)__result;			\
		xpu_bpchar_t	datum_a;									\
		xpu_bpchar_t	datum_b;									\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);			\
																	\
		assert(kexp->nr_args == 2 &&								\
			   KEXP_IS_VALID(karg, bpchar));						\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))			\
			return false;											\
		karg = KEXP_NEXT_ARG(karg);									\
		assert(KEXP_IS_VALID(karg, bpchar));						\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))			\
			return false;											\
		result->isnull = (datum_a.isnull | datum_b.isnull);			\
		if (!result->isnull)										\
		{															\
			int		status;											\
																	\
			if (!__bpchar_compare(kcxt, &status,					\
								  &datum_a,							\
								  &datum_b))						\
				return false;										\
			result->value = (status OPER 0);						\
		}															\
		return true;												\
	}
PG_BPCHAR_COMPARE_TEMPLATE(eq, ==)
PG_BPCHAR_COMPARE_TEMPLATE(ne, !=)
PG_BPCHAR_COMPARE_TEMPLATE(lt, <)
PG_BPCHAR_COMPARE_TEMPLATE(le, <=)
PG_BPCHAR_COMPARE_TEMPLATE(gt, >)
PG_BPCHAR_COMPARE_TEMPLATE(ge, >=)

PUBLIC_FUNCTION(bool)
pgfn_bpcharlen(XPU_PGFUNCTION_ARGS)
{
	xpu_int4_t	   *result = (xpu_int4_t *)__result;
	xpu_bpchar_t	datum;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(karg->nr_args == 1 &&
		   KEXP_IS_VALID(karg, bpchar));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum))
		return false;
	memset(&result, 0, sizeof(xpu_int4_t));
	result->isnull = datum.isnull;
	if (!datum.isnull)
	{
		if (!xpu_bpchar_is_valid(kcxt, &datum))
			return false;
		result->value = datum.length;
	}
	return true;
}

/*
 * Text functions
 */
STATIC_FUNCTION(bool)
__text_compare(kern_context *kcxt,
			   int *p_status,
			   const xpu_text_t *str1,
			   const xpu_text_t *str2)
{
	const char *s1 = str1->value;
	const char *s2 = str2->value;
	int		len;

	if (!xpu_text_is_valid(kcxt, str1) ||
		!xpu_text_is_valid(kcxt, str2))
		return false;
	len = Min(str1->length, str2->length);
	while (len > 0)
	{
		if (*s1 < *s2)
		{
			*p_status = -1;
			return true;
		}
		if (*s1 > *s2)
		{
			*p_status = 1;
			return true;
		}
	}
	if (str1->length != str2->length)
		*p_status = (str1->length > str2->length ? 1 : -1);
	else
		*p_status = 0;
	return true;
}

#define PG_TEXT_COMPARE_TEMPLATE(NAME,OPER)							\
	PUBLIC_FUNCTION(bool)											\
	pgfn_text##NAME(XPU_PGFUNCTION_ARGS)							\
	{																\
		xpu_bool_t	   *result = (xpu_bool_t *)__result;			\
		xpu_text_t		datum_a;									\
		xpu_text_t		datum_b;									\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);			\
																	\
		assert(kexp->nr_args == 2 &&								\
			   KEXP_IS_VALID(karg, text));							\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))			\
			return false;											\
		karg = KEXP_NEXT_ARG(karg);									\
		assert(KEXP_IS_VALID(karg, text));							\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))			\
			return false;											\
		result->isnull = (datum_a.isnull | datum_b.isnull);			\
		if (!result->isnull)										\
		{															\
			int		status;											\
																	\
			if (!__text_compare(kcxt, &status,						\
								&datum_a,							\
								&datum_b))							\
				return false;										\
			result->value = (status OPER 0);						\
		}															\
		return true;												\
	}
PG_TEXT_COMPARE_TEMPLATE(eq, ==)
PG_TEXT_COMPARE_TEMPLATE(ne, !=)
PG_TEXT_COMPARE_TEMPLATE(_lt, <)
PG_TEXT_COMPARE_TEMPLATE(_le, <=)
PG_TEXT_COMPARE_TEMPLATE(_gt, >)
PG_TEXT_COMPARE_TEMPLATE(_ge, >=)

PUBLIC_FUNCTION(bool)
pgfn_textlen(XPU_PGFUNCTION_ARGS)
{
	xpu_int4_t	   *result = (xpu_int4_t *)__result;
	xpu_text_t		datum;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 1 &&
		   KEXP_IS_VALID(karg, text));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum))
		return false;
	memset(result, 0, sizeof(xpu_int4_t));
	result->isnull = datum.isnull;
	if (!datum.isnull)
	{
		xpu_encode_info *encode = SESSION_ENCODE(kcxt->session);
		int		len = datum.length;

		if (!encode)
		{
			STROM_ELOG(kcxt, "No encoding info was supplied");
			return false;
		}
		else if (encode->enc_maxlen == 1)
		{
			if (len < 0)
			{
				if (VARATT_IS_COMPRESSED(datum.value))
				{
					toast_compress_header	c_hdr;

					memcpy(&c_hdr, VARDATA(datum.value),
						   sizeof(toast_compress_header));
					len = TOAST_COMPRESS_EXTSIZE(&c_hdr);
				}
				else if (VARATT_IS_EXTERNAL(datum.value))
				{
					varatt_external			e_hdr;

					memcpy(&e_hdr, VARDATA_1B_E(datum.value),
						   sizeof(varatt_external));
					len = (e_hdr.va_extinfo & VARLENA_EXTSIZE_MASK);
				}
				else
				{
					STROM_ELOG(kcxt, "unknown varlena format");
					return false;
				}
			}
		}
		else if (len >= 0)
		{
			const char *mbstr = datum.value;
			int			sz = 0;

			while (len > 0 && *mbstr)
			{
				int		n = encode->enc_mblen(mbstr);

				len   -= n;
				mbstr += n;
				sz++;
			}
			len = sz;
		}
		else
		{
			STROM_ELOG(kcxt, "unable to count compressed/external text under multi-bytes encoding");
			return false;
		}
		result->value = len;
	}
	return true;
}

/* ----------------------------------------------------------------
 *
 * Fundamental Encoding Catalog
 *
 * ---------------------------------------------------------------- */
STATIC_FUNCTION(int)
pg_latin_mblen(const char *s)
{
	return 1;
}

STATIC_FUNCTION(int)
pg_euc_mblen(const char *s)
{
	unsigned char	c = *s;

	if (c == 0x8e)
		return 2;
	if (c == 0x8f)
		return 3;
	if (c & 0x80)
		return 2;
	return 1;
}

STATIC_FUNCTION(int)
pg_euc_cn_mblen(const char *s)
{
	unsigned char	c = *s;

	return ((c & 0x80) ? 2 : 1);
}

STATIC_FUNCTION(int)
pg_utf8_mblen(const char *s)
{
	unsigned char	c = *s;

	if ((c & 0x80) == 0)
		return 1;
	if ((c & 0xe0) == 0xc0)
		return 2;
	if ((c & 0xf0) == 0xe0)
		return 3;
	if ((c & 0xf8) == 0xf0)
		return 4;
	return 1;
}

STATIC_FUNCTION(int)
pg_mule_mblen(const char *s)
{
	unsigned char	c = *s;

	if (c >= 0x81 && c <= 0x8d)
		return 2;
	if (c == 0x9a || c == 0x9b)
		return 3;
	if (c >= 0x90 && c <= 0x99)
		return 2;
	if (c == 0x9c || c == 0x9d)
		return 4;
	return 1;
}

STATIC_FUNCTION(int)
pg_sjis_mblen(const char *s)
{
	unsigned char	c = *s;

	if (c >= 0xa1 && c <= 0xdf)
		return 1;	/* 1 byte kana? */
	if (c & 0x80)
		return 2;
	return 1;
}

STATIC_FUNCTION(int)
pg_big5_mblen(const char *s)
{
	unsigned char	c = *s;

	return ((c & 0x80) ? 2 : 1);
}

STATIC_FUNCTION(int)
pg_gbk_mblen(const char *s)
{
	unsigned char	c = *s;

	return ((c & 0x80) ? 2 : 1);
}

STATIC_FUNCTION(int)
pg_uhc_mblen(const char *s)
{
	unsigned char	c = *s;

	return ((c & 0x80) ? 2 : 1);
}

STATIC_FUNCTION(int)
pg_gb18030_mblen(const char *s)
{
	unsigned char	c1 = *s;
	unsigned char	c2;

	if ((c1 & 0x80) == 0)
		return 1;	/* ascii */
	c2 = s[1];
	if (c2 >= 0x30 && c2 <= 0x39)
		return 4;
	return 2;
}

#define DATABASE_SB_ENCODE(NAME)					{ #NAME, 1, pg_latin_mblen }
#define DATABASE_MB_ENCODE(NAME,MAXLEN,FN_MBLEN)	{ #NAME, MAXLEN, FN_MBLEN }

PUBLIC_DATA	xpu_encode_info	xpu_encode_catalog[] = {
	DATABASE_SB_ENCODE(SQL_ASCII),
	DATABASE_MB_ENCODE(EUC_JP, 3, pg_euc_mblen),
	DATABASE_MB_ENCODE(EUC_CN, 2, pg_euc_cn_mblen),
	DATABASE_MB_ENCODE(EUC_KR, 3, pg_euc_mblen),
	DATABASE_MB_ENCODE(EUC_TW, 4, pg_euc_mblen),
	DATABASE_MB_ENCODE(EUC_JIS_2004, 3, pg_euc_mblen),
	DATABASE_MB_ENCODE(UTF8, 4, pg_utf8_mblen),
	DATABASE_MB_ENCODE(MULE_INTERNAL, 4, pg_mule_mblen),
	DATABASE_SB_ENCODE(LATIN1),
	DATABASE_SB_ENCODE(LATIN2),
	DATABASE_SB_ENCODE(LATIN3),
	DATABASE_SB_ENCODE(LATIN4),
	DATABASE_SB_ENCODE(LATIN5),
	DATABASE_SB_ENCODE(LATIN6),
	DATABASE_SB_ENCODE(LATIN7),
	DATABASE_SB_ENCODE(LATIN8),
	DATABASE_SB_ENCODE(LATIN9),
	DATABASE_SB_ENCODE(LATIN10),
	DATABASE_SB_ENCODE(WIN1256),
	DATABASE_SB_ENCODE(WIN1258),
	DATABASE_SB_ENCODE(WIN866),
	DATABASE_SB_ENCODE(WIN874),
	DATABASE_SB_ENCODE(KOI8R),
	DATABASE_SB_ENCODE(WIN1251),
	DATABASE_SB_ENCODE(WIN1252),
	DATABASE_SB_ENCODE(ISO_8859_5),
	DATABASE_SB_ENCODE(ISO_8859_6),
	DATABASE_SB_ENCODE(ISO_8859_7),
	DATABASE_SB_ENCODE(ISO_8859_8),
	DATABASE_SB_ENCODE(WIN1250),
	DATABASE_SB_ENCODE(WIN1253),
	DATABASE_SB_ENCODE(WIN1254),
	DATABASE_SB_ENCODE(WIN1255),
	DATABASE_SB_ENCODE(WIN1257),
	DATABASE_SB_ENCODE(KOI8U),
	DATABASE_MB_ENCODE(SJIS, 2, pg_sjis_mblen),
	DATABASE_MB_ENCODE(BIG5, 2, pg_big5_mblen),
	DATABASE_MB_ENCODE(GBK, 2, pg_gbk_mblen),
	DATABASE_MB_ENCODE(UHC, 2, pg_uhc_mblen),
	DATABASE_MB_ENCODE(GB18030, 4, pg_gb18030_mblen),
	DATABASE_MB_ENCODE(JOHAB, 3, pg_euc_mblen),
	DATABASE_MB_ENCODE(SHIFT_JIS_2004, 2, pg_sjis_mblen),
	{ "__LAST__", -1, NULL },
};

/* ----------------------------------------------------------------
 *
 * Routines to support LIKE and ILIKE 
 *
 * ---------------------------------------------------------------- */
#define LIKE_TRUE			1
#define LIKE_FALSE			0
#define LIKE_ABORT			(-1)
#define LIKE_EXCEPTION		(-255)
#define NextByte(p, plen)							\
	do { (p)++; (plen)--; } while(0)
#define NextChar(p, plen)							\
	do { int __len = encode->enc_mblen(p);			\
		(p) += __len; (plen) -= __len; } while(0)
#define GetChar(c)			(c)
#define GetCharUpper(c)		(((c) >= 'a' && (c) <= 'z') ? c + ('A' - 'a') : c)

#define GENERIC_MATCH_TEXT_TEMPLATE(FUNCNAME, GETCHAR)					\
	STATIC_FUNCTION(int)												\
	FUNCNAME(kern_context *kcxt,										\
			 const char *t, int tlen,									\
			 const char *p, int plen,									\
			 int depth)													\
	{																	\
		xpu_encode_info	   *encode = SESSION_ENCODE(kcxt->session);		\
																		\
		/* Fast path for match-everything pattern */					\
		if (plen == 1 && *p == '%')										\
			return LIKE_TRUE;											\
		/* this function is recursive */								\
		if (depth > 10)													\
		{																\
			STROM_ELOG(kcxt, "like recursion too deep");				\
			return LIKE_EXCEPTION;										\
		}																\
		/*																\
		 * In this loop, we advance by char when matching wildcards		\
		 * (and thus on recursive entry to this function we are			\
		 * properly char-synced). On other occasions it is safe to		\
		 * advance by byte,as the text and pattern will be in lockstep. \
		 * This allows us to perform all comparisons between the		\
		 * text and pattern on a byte by byte basis, even for			\
		 * multi-byte encodings.										\
		 */																\
		while (tlen > 0 && plen > 0)									\
		{																\
			if (*p == '\\')												\
			{															\
				/* Next pattern byte must match literally,				\
				 * whatever it is */									\
				NextByte(p, plen);										\
				/* ... and there had better be one, per SQL standard */	\
				if (plen <= 0)											\
				{														\
					STROM_ELOG(kcxt, "invalid escape in LIKE pattern");	\
					return LIKE_EXCEPTION;								\
				}														\
				if (GETCHAR(*p) != GETCHAR(*t))							\
					return LIKE_FALSE;									\
			}															\
			else if (*p == '%')											\
			{															\
				char        firstpat;									\
																		\
				/*														\
				 * % processing is essentially a search for a text		\
				 *  position at which the remainder of the text matches	\
				 * the remainder of the pattern, using a recursive call \
				 * to check each potential match.						\
				 *														\
				 * If there are wildcards immediately following the %,	\
				 * we can skip over them first, using the idea that any \
				 * sequence of N _'s and one or more %'s is equivalent	\
				 * to N _'s and one % (ie, it will						\
				 * match any sequence of at least N text characters).	\
				 * In this way we will always run the recursive search	\
				 * loop using a pattern fragment that begins with		\
				 * a literal character-to-match, thereby not recursing	\
				 * more than we have to.								\
				 */														\
				NextByte(p, plen);										\
																		\
				while (plen > 0)										\
				{														\
					if (*p == '%')										\
						NextByte(p, plen);								\
					else if (*p == '_')									\
					{													\
						/* If not enough text left to match				\
						 *the pattern, ABORT */							\
						if (tlen <= 0)									\
							return LIKE_ABORT;							\
						NextChar(t, tlen);								\
						NextByte(p, plen);								\
					}													\
					else												\
						break;	/* A non-wildcard pattern char */		\
				}														\
				/*														\
				 * If we're at end of pattern, match: we have			\
				 * a trailing a trailing % which matches any remaining	\
				 * text string.											\
				 */														\
				if (plen <= 0)											\
					return LIKE_TRUE;									\
				/*														\
				 * Otherwise, scan for a text position at which we can	\
				 * match the rest of the pattern.  The first remaining	\
				 * pattern char is known to be a regular or escaped		\
				 * literal character, so we can compare the first		\
				 * pattern byte to each text byte to avoid recursing	\
				 * more than we have to.  This fact also guarantees		\
				 * that we don't have to consider a match to the zero-	\
				 * length substring at the end of the text.				\
				 */														\
				if (*p == '\\')											\
				{														\
					if (plen < 2)										\
					{													\
						STROM_ELOG(kcxt, "invalid escape in LIKE pattern");	\
						return LIKE_EXCEPTION;							\
					}													\
					firstpat = GETCHAR(p[1]);							\
				}														\
				else													\
					firstpat = GETCHAR(*p);								\
																		\
				while (tlen > 0)										\
				{														\
					if (GETCHAR(*t) == firstpat)						\
					{													\
						int		matched = FUNCNAME(kcxt,				\
												   t, tlen,				\
												   p, plen,				\
												   depth+1);			\
						if (matched != LIKE_FALSE)						\
							return matched; /* TRUE or ABORT */			\
					}													\
					NextChar(t, tlen);									\
				}														\
				/*														\
				 * End of text with no match, so no point in trying later places \
				 * to start matching this pattern.						\
				 */														\
				return LIKE_ABORT;										\
			}															\
			else if (*p == '_')											\
			{															\
				/* _ matches any single character, and					\
				 * we know there is one */								\
				NextChar(t, tlen);										\
				NextByte(p, plen);										\
				continue;												\
			}															\
			else if (GETCHAR(*p) != GETCHAR(*t))						\
			{															\
				/* non-wildcard pattern char fails to match */			\
				return LIKE_FALSE;										\
			}															\
			/*															\
			 * Pattern and text match, so advance.						\
			 *															\
			 * It is safe to use NextByte instead of NextChar here,		\
			 * even for multi-byte character sets, because we are not	\
			 * following immediately after a wildcard character.		\
			 * If we are in the middle of a multibyte character, we		\
			 * must already have matched at least one byte of the		\
			 * character from both text and pattern; so we cannot get	\
			 * out-of-sync on character boundaries.  And we know that	\
			 * no backend-legal encoding allows ASCII characters such	\
			 * as '%' to appear as non-first bytes of characters, so	\
			 * we won't mistakenly detect a new wildcard */				\
			NextByte(t, tlen);											\
			NextByte(p, plen);											\
		}																\
		if (tlen > 0)													\
			return LIKE_FALSE;	/* end of pattern, but not of text */	\
																		\
		/*																\
		 * End of text, but perhaps not of pattern. Match iff the		\
		 remaining pattern can match a zero-length string,				\
		 * ie, it's zero or more %'s.									\
		 */																\
		while (plen > 0 && *p == '%')									\
			NextByte(p, plen);											\
		if (plen <= 0)													\
			return LIKE_TRUE;											\
		/*																\
		 * End of text with no match, so no point in trying later		\
		 * places to start matching this pattern.						\
		 */																\
		return LIKE_ABORT;												\
	}
GENERIC_MATCH_TEXT_TEMPLATE(GenericMatchText, GetChar)
GENERIC_MATCH_TEXT_TEMPLATE(GenericCaseMatchText, GetCharUpper)

#define PG_TEXTLIKE_TEMPLATE(FN_NAME,FN_MATCH,OPER)					\
	PUBLIC_FUNCTION(bool)											\
	pgfn_##FN_NAME(XPU_PGFUNCTION_ARGS)								\
	{																\
		xpu_bool_t	   *result = (xpu_bool_t *)__result;			\
		xpu_text_t		datum_a;	/* string */					\
		xpu_text_t		datum_b;	/* pattern */					\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);			\
																	\
		assert(kexp->nr_args == 2 &&								\
			   KEXP_IS_VALID(karg, text));							\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))			\
			return false;											\
		karg = KEXP_NEXT_ARG(karg);									\
		assert(KEXP_IS_VALID(karg, text));							\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))			\
			return false;											\
		result->isnull = (datum_a.isnull | datum_b.isnull);			\
		if (!result->isnull)										\
		{															\
			int		status;											\
																	\
			if (!xpu_text_is_valid(kcxt, &datum_a) ||				\
				!xpu_text_is_valid(kcxt, &datum_b))					\
				return false;										\
			status = FN_MATCH(kcxt,									\
							  datum_a.value, datum_a.length,		\
							  datum_b.value, datum_b.length, 0);	\
			if (status == LIKE_EXCEPTION)							\
				return false;										\
			result->value = (status OPER LIKE_TRUE);				\
		}															\
		return true;												\
	}																\

PG_TEXTLIKE_TEMPLATE(like, GenericMatchText, ==)
PG_TEXTLIKE_TEMPLATE(textlike, GenericMatchText, ==)
PG_TEXTLIKE_TEMPLATE(notlike, GenericMatchText, !=)
PG_TEXTLIKE_TEMPLATE(textnlike, GenericMatchText, !=)
PG_TEXTLIKE_TEMPLATE(texticlike, GenericCaseMatchText, ==)
PG_TEXTLIKE_TEMPLATE(texticnlike, GenericCaseMatchText, !=)

#define PG_BPCHARLIKE_TEMPLATE(FN_NAME,FN_MATCH,OPER)				\
	PUBLIC_FUNCTION(bool)											\
	pgfn_##FN_NAME(XPU_PGFUNCTION_ARGS)								\
	{																\
		xpu_bool_t	   *result = (xpu_bool_t *)__result;			\
		xpu_bpchar_t	datum_a;	/* string */					\
		xpu_text_t		datum_b;	/* pattern */					\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);			\
																	\
		assert(kexp->nr_args == 2 &&								\
			   KEXP_IS_VALID(karg, bpchar));						\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))			\
			return false;											\
		karg = KEXP_NEXT_ARG(karg);									\
		assert(KEXP_IS_VALID(karg, text));							\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))			\
			return false;											\
		result->isnull = (datum_a.isnull | datum_b.isnull);			\
		if (!result->isnull)										\
		{															\
			int		status;											\
																	\
			if (!xpu_bpchar_is_valid(kcxt, &datum_a) ||				\
				!xpu_text_is_valid(kcxt, &datum_b))					\
				return false;										\
			status = FN_MATCH(kcxt,									\
							  datum_a.value, datum_a.length,		\
							  datum_b.value, datum_b.length, 0);	\
			if (status == LIKE_EXCEPTION)							\
				return false;										\
			result->value = (status OPER LIKE_TRUE);				\
		}															\
		return true;												\
	}
PG_BPCHARLIKE_TEMPLATE(bpcharlike, GenericMatchText, ==)
PG_BPCHARLIKE_TEMPLATE(bpcharnlike, GenericMatchText, !=)
PG_BPCHARLIKE_TEMPLATE(bpchariclike, GenericCaseMatchText, ==)
PG_BPCHARLIKE_TEMPLATE(bpcharicnlike, GenericCaseMatchText, !=)










	
