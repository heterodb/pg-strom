/*
 * cuda_textlib.h
 *
 * Collection of text functions for OpenCL devices
 * --
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
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
#ifndef CUDA_TEXTLIB_H
#define CUDA_TEXTLIB_H
#ifdef __CUDACC__

/* ----------------------------------------------------------------
 *
 * Basic Text comparison functions
 * 
 * ----------------------------------------------------------------
 */
STATIC_INLINE(cl_int)
bpchar_truelen(struct varlena *arg)
{
	cl_char	   *s = VARDATA_ANY(arg);
	cl_int		i, len;

	len = VARSIZE_ANY_EXHDR(arg);
	for (i = len - 1; i >= 0; i--)
	{
		if (s[i] != ' ')
			break;
	}
	return i + 1;
}

#ifndef PG_BPCHAR_TYPE_DEFINED
#define PG_BPCHAR_TYPE_DEFINED
STROMCL_VARLENA_DATATYPE_TEMPLATE(bpchar)
STROMCL_VARLENA_VARREF_TEMPLATE(bpchar)
STROMCL_VARLENA_VARSTORE_TEMPLATE(bpchar)
STROMCL_VARLENA_PARAMREF_TEMPLATE(bpchar)
STROMCL_VARLENA_NULLTEST_TEMPLATE(bpchar)
/* pg_bpchar_comp_crc32 has to be defined with own way */
STATIC_FUNCTION(cl_uint)
pg_bpchar_comp_crc32(const cl_uint *crc32_table,
					 cl_uint hash, pg_bpchar_t datum)
{
	if (!datum.isnull)
	{
		hash = pg_common_comp_crc32(crc32_table, hash,
									VARDATA_ANY(datum.value),
									bpchar_truelen(datum.value));
	}
	return hash;
}
#endif

STATIC_FUNCTION(cl_int)
bpchar_compare(kern_context *kcxt, varlena *arg1, varlena *arg2)
{
	cl_char	   *s1 = VARDATA_ANY(arg1);
	cl_char	   *s2 = VARDATA_ANY(arg2);
	cl_int		len1 = bpchar_truelen(arg1);
	cl_int		len2 = bpchar_truelen(arg2);
	cl_int		len = min(len1, len2);

	while (len > 0)
	{
		if (*s1 < *s2)
			return -1;
		if (*s1 > *s2)
			return 1;
		s1++;
		s2++;
		len--;
	}
	if (len1 != len2)
		return (len1 > len2 ? 1 : -1);
	return 0;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_bpchareq(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(bpchar_compare(kcxt,
												arg1.value,
												arg2.value) == 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_bpcharne(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(bpchar_compare(kcxt,
												arg1.value,
												arg2.value) != 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_bpcharlt(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(bpchar_compare(kcxt,
												arg1.value,
												arg2.value) < 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_bpcharle(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(bpchar_compare(kcxt,
												arg1.value,
												arg2.value) <= 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_bpchargt(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(bpchar_compare(kcxt,
												arg1.value,
												arg2.value) > 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_bpcharge(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(bpchar_compare(kcxt,
												arg1.value,
												arg2.value) >= 0);
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_bpcharcmp(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_int4_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = bpchar_compare(kcxt, arg1.value, arg2.value);
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_bpcharlen(kern_context *kcxt, pg_bpchar_t arg1)
{
	pg_int4_t	result;

	/* NOTE: At this moment, we don't support any special encodings,
	 * so no multibytes character is assumed.
	 */
	result.isnull = arg1.isnull;
	if (!result.isnull)
		result.value = bpchar_truelen(arg1.value);
	return result;
}

/* ----------------------------------------------------------------
 *
 * Basic Text comparison functions
 * 
 * ----------------------------------------------------------------
 */
#ifndef PG_TEXT_TYPE_DEFINED
#define PG_TEXT_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(text)
#endif

STATIC_FUNCTION(cl_int)
text_compare(kern_context *kcxt, varlena *arg1, varlena *arg2)
{
	cl_char	   *s1 = VARDATA_ANY(arg1);
	cl_char	   *s2 = VARDATA_ANY(arg2);
	cl_int		len1 = VARSIZE_ANY_EXHDR(arg1);
	cl_int		len2 = VARSIZE_ANY_EXHDR(arg2);
	cl_int		len = min(len1, len2);

	while (len > 0)
	{
		if (*s1 < *s2)
			return -1;
		if (*s1 > *s2)
			return 1;

		s1++;
		s2++;
		len--;
	}
	if (len1 != len2)
		return (len1 > len2 ? 1 : -1);
	return 0;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_texteq(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(text_compare(kcxt,
											  arg1.value,
											  arg2.value) == 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_textne(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(text_compare(kcxt,
											  arg1.value,
											  arg2.value) != 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_text_lt(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(text_compare(kcxt,
											  arg1.value,
											  arg2.value) < 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_text_le(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(text_compare(kcxt,
											  arg1.value,
											  arg2.value) <= 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_text_gt(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(text_compare(kcxt,
											  arg1.value,
											  arg2.value) > 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_text_ge(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(text_compare(kcxt,
											  arg1.value,
											  arg2.value) >= 0);
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_text_cmp(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_int4_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = text_compare(kcxt, arg1.value, arg2.value);
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_textlen(kern_context *kcxt, pg_text_t arg1)
{
	pg_int4_t	result;

	/* NOTE: At this moment, we don't support any special encodings,
	 * so no multibytes character is assumed.
	 */
	result.isnull = arg1.isnull;
	if (!result.isnull)
		result.value = toast_raw_datum_size(kcxt, arg1.value) - VARHDRSZ;
	return result;
}

/*
 * varchar(*) type definition
 */
#ifndef PG_VARCHAR_TYPE_DEFINED
#define PG_VARCHAR_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(varchar)
#endif

/*
 * Support for LIKE operator
 */
#define LIKE_TRUE				1
#define LIKE_FALSE				0
#define LIKE_ABORT				(-1)

#define GetCharNormal(t)		(t)
STATIC_INLINE(cl_char)
GetCharLowerCase(cl_char c)
{
	return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c;
}

#define NextByte(p, plen)		\
	do { (p)++; (plen)--; } while(0)
#define NextChar(p, plen)		\
	do { int __l = pg_wchar_mblen(p); (p) += __l; (plen) -= __l; } while(0)

#define RECURSIVE_RETURN(__retcode)			\
	do {									\
		if (depth == 0)						\
			return (__retcode);				\
		retcode = (__retcode);				\
		goto recursive_return;				\
	} while(0)
#define VIRTUAL_STACK_MAX_DEPTH		8


#define GENERIC_MATCH_TEXT_TEMPLATE(FUNCNAME, GETCHAR)					\
	STATIC_FUNCTION(cl_int)												\
	FUNCNAME(kern_context *kcxt,										\
			 char *t, int tlen,											\
			 char *p, int plen,											\
			 int depth)													\
	{																	\
		cl_int		retcode;											\
		struct {														\
			char   *__t;												\
			char   *__p;												\
			int		__tlen;												\
			int		__plen;												\
		} vstack[VIRTUAL_STACK_MAX_DEPTH];								\
																		\
	recursive_entry:													\
		/* Fast path for match-everything pattern */					\
		if (plen == 1 && *p == '%')										\
			RECURSIVE_RETURN(LIKE_TRUE);								\
																		\
		/*																\
		 * In this loop, we advance by char when matching wildcards		\
		 * (and thus on recursive entry to this function we are			\
		 * properly char-synced). On other occasions it is safe to		\
		 * advance by byte,as the text and pattern will be in lockstep.	\
		 * This allows us to perform all comparisons between the text	\
		 * and pattern on a byte by byte basis, even for multi-byte		\
		 * encodings.													\
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
					STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);	\
					return LIKE_ABORT;									\
				}														\
				if (GETCHAR(*p) != GETCHAR(*t))							\
					RECURSIVE_RETURN(LIKE_FALSE);						\
			}															\
			else if (*p == '%')											\
			{															\
				char		firstpat;									\
																		\
				/*														\
				 * % processing is essentially a search for a text		\
				 * position at which the remainder of the text matches	\
				 * the remainder of the pattern, using a recursive call	\
				 * to check each potential match.						\
				 *														\
				 * If there are wildcards immediately following the %,	\
				 * we can skip over them first, using the idea that any	\
				 * sequence of N _\'s and one or more %\'s is equivalent \
				 * to N\'s and one % (ie, it will match any sequence of	\
				 * at least N text characters).  In this way we will	\
				 * always run the recursive search loop using a pattern \
				 * fragment that begins with a literal character-to-match, \
				 * thereby not recursing more than we have to.			\
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
						 * the pattern ABORT */							\
						if (tlen <= 0)									\
							RECURSIVE_RETURN(LIKE_ABORT);				\
						NextChar(t, tlen);								\
						NextByte(p, plen);								\
					}													\
					else												\
						break;	/* Reached a non-wildcard pattern char */ \
				}														\
																		\
				/*														\
				 * If we are at end of pattern, match: we have			\
				 * a trailing % which matches any remaining text string. \
				 */														\
				if (plen <= 0)											\
					RECURSIVE_RETURN(LIKE_TRUE);						\
																		\
				/*														\
				 * Otherwise, scan for a text position at which we can	\
				 * match the rest of the pattern.  The first remaining	\
				 * pattern char is known to be a regular or escaped		\
				 * literal character, so we can compare the first		\
				 * pattern byte to each text byte to avoid recursing	\
				 * more than we have to.								\
				 * This fact also guarantees that we don\'t have to		\
				 * consider	a match to the zero-length substring at		\
				 * the end of the text.									\
				 * text.												\
				 */														\
				if (*p == '\\')											\
				{														\
					if (plen < 0)										\
					{													\
						STROM_SET_ERROR(&kcxt->e,						\
										StromError_CpuReCheck);			\
						return LIKE_ABORT;								\
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
						if (depth >= VIRTUAL_STACK_MAX_DEPTH)			\
						{												\
							STROM_SET_ERROR(&kcxt->e,					\
											StromError_CpuReCheck);		\
							return LIKE_ABORT;							\
						}												\
						/* push values */								\
						vstack[depth].__t = t;							\
						vstack[depth].__p = p;							\
						vstack[depth].__tlen = tlen;					\
						vstack[depth].__plen = plen;					\
						depth++;										\
						goto recursive_entry;							\
					recursive_return:									\
						depth--;										\
						if (retcode != LIKE_FALSE)						\
							RECURSIVE_RETURN(retcode);	/* TRUE or ABORT */	\
						/* pop values */								\
						t = vstack[depth].__t;							\
						p = vstack[depth].__p;							\
						tlen = vstack[depth].__tlen;					\
						plen = vstack[depth].__plen;					\
					}													\
					NextChar(t, tlen);									\
				}														\
																		\
				/*														\
				 * End of text with no match, so no point in trying \
				 * later places to start matching this pattern.				\
				 */														\
				RECURSIVE_RETURN(LIKE_ABORT);							\
			}															\
			else if (*p == '_')											\
			{															\
				/* _ matches any single character */					\
				NextChar(t, tlen);										\
				NextByte(p, plen);										\
				continue;												\
			}															\
			else if (GETCHAR(*p) != GETCHAR(*t))						\
			{															\
				RECURSIVE_RETURN(LIKE_FALSE);							\
			}															\
																		\
			/*															\
			 * Pattern and text match, so advance.						\
			 *															\
			 * It is safe to use NextByte instead of NextChar here,		\
			 * even for multi-byte character sets, because we are not	\
			 * following immediately after a wildcard character. If we	\
			 * are in the middle of a multibyte character, we must already \
			 * have matched at least one byte of the character from both \
			 * text and pattern; so we cannot get out-of-sync on character \
			 * boundaries.  And we know that no backend-legal encoding	\
			 * allows ASCII characters such as \'%\' to appear as non-first	\
			 bytes of characters, so we won\'t mistakenly detect a new	\
			 * wildcard.												\
			 */															\
			NextByte(t, tlen);											\
			NextByte(p, plen);											\
		}																\
																		\
		if (tlen > 0)													\
			RECURSIVE_RETURN(LIKE_FALSE);								\
																		\
		/*																\
		 * End of text, but perhaps not of pattern.  Match iff the		\
		 * remaining pattern can match a zero-length string, ie, it\'s	\
		 * zero or more %\'s.											\
		 */																\
		while (plen > 0 && *p == '%')									\
			NextByte(p, plen);											\
		if (plen <= 0)													\
			RECURSIVE_RETURN(LIKE_TRUE);								\
																		\
		/*																\
		 * End of text with no match, so no point in trying later		\
		 * places to start matching this pattern.						\
		 */																\
		RECURSIVE_RETURN(LIKE_ABORT);									\
	}

GENERIC_MATCH_TEXT_TEMPLATE(GenericMatchText, GetCharNormal)
GENERIC_MATCH_TEXT_TEMPLATE(GenericCaseMatchText, GetCharLowerCase)

#undef GetCharNormal
#undef GetCharLowerCase
#undef NextByte
#undef NextChar
#undef RECURSIVE_RETURN
#undef VIRTUAL_STACK_MAX_DEPTH

STATIC_FUNCTION(pg_bool_t)
pgfn_textlike(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		char	   *s = VARDATA_ANY(arg1.value);
		char	   *p = VARDATA_ANY(arg2.value);
		cl_uint		slen = VARSIZE_ANY_EXHDR(arg1.value);
		cl_uint		plen = VARSIZE_ANY_EXHDR(arg2.value);

		result.value = (GenericMatchText(kcxt,
										 s, slen,
										 p, plen, 0) == LIKE_TRUE);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_textnlike(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		char	   *s = VARDATA_ANY(arg1.value);
		char	   *p = VARDATA_ANY(arg2.value);
		cl_uint		slen = VARSIZE_ANY_EXHDR(arg1.value);
		cl_uint		plen = VARSIZE_ANY_EXHDR(arg2.value);

		result.value = (GenericMatchText(kcxt,
										 s, slen,
										 p, plen, 0) != LIKE_TRUE);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_texticlike(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		char	   *s = VARDATA_ANY(arg1.value);
		char	   *p = VARDATA_ANY(arg2.value);
		cl_uint		slen = VARSIZE_ANY_EXHDR(arg1.value);
		cl_uint		plen = VARSIZE_ANY_EXHDR(arg2.value);

		result.value = (GenericCaseMatchText(kcxt,
											 s, slen,
											 p, plen, 0) == LIKE_TRUE);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_texticnlike(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		char	   *s = VARDATA_ANY(arg1.value);
		char	   *p = VARDATA_ANY(arg2.value);
		cl_uint		slen = VARSIZE_ANY_EXHDR(arg1.value);
		cl_uint		plen = VARSIZE_ANY_EXHDR(arg2.value);

		result.value = (GenericCaseMatchText(kcxt,
											 s, slen,
											 p, plen, 0) != LIKE_TRUE);
	}
	return result;
}

#undef LIKE_TRUE
#undef LIKE_FALSE
#undef LIKE_ABORT



#else	/* __CUDACC__ */
#include "mb/pg_wchar.h"

STATIC_INLINE(void)
assign_textlib_session_info(StringInfo buf)
{
	/*
	 * Put encoding aware character length function
	 */
	appendStringInfoString(
		buf,
		"STATIC_INLINE(cl_int)\n"
		"pg_wchar_mblen(const char *str)\n"
		"{\n");

	switch (GetDatabaseEncoding())
	{
		case PG_EUC_JP:		/* logic in pg_euc_mblen() */
		case PG_EUC_KR:
		case PG_EUC_TW:		/* logic in pg_euctw_mblen(), but identical */
		case PG_EUC_JIS_2004:
		case PG_JOHAB:		/* logic in pg_johab_mblen(), but identical */
			appendStringInfoString(
				buf,
				"  cl_uchar c = *((const cl_uchar *)str);\n"
				"  if (c == 0x8e)\n"
				"    return 2;\n"
				"  else if (c == 0x8f)\n"
				"    return 3;\n"
				"  else if (c & 0x80)\n"
				"    return 2;\n"
				"  return 1;\n");
			break;

		case PG_EUC_CN:		/* logic in pg_euccn_mblen */
			appendStringInfoString(
				buf,
				"  cl_uchar c = *((const cl_uchar *)str);\n"
				"  if (c & 0x80)\n"
				"    return 2;\n"
				"  return 1;\n");
			break;

		case PG_UTF8:		/* logic in pg_utf_mblen */
			appendStringInfoString(
				buf,
				"  cl_uchar c = *((const cl_uchar *)str);\n"
				"  if ((c & 0x80) == 0)\n"
				"    return 1;\n"
				"  else if ((c & 0xe0) == 0xc0)\n"
				"    return 2;\n"
				"  else if ((c & 0xf0) == 0xe0)\n"
				"    return 3;\n"
				"  else if ((c & 0xf8) == 0xf0)\n"
				"    return 4;\n"
				"#ifdef NOT_USED\n"
				"  else if ((c & 0xfc) == 0xf8)\n"
				"    return 5;\n"
				"  else if ((c & 0xfe) == 0xfc)\n"
				"    return 6;\n"
				"#endif\n"
				"  return 1;\n");
			break;

		case PG_MULE_INTERNAL:	/* logic in pg_mule_mblen */
			appendStringInfoString(
				buf,
				"  cl_uchar c = *((const cl_uchar *)str);\n"
				"  if (c >= 0x81 && c <= 0x8d)\n"
				"    return 2;\n"
				"  else if (c == 0x9a || c == 0x9b)\n"
				"    return 3;\n"
				"  else if (c >= 0x90 && c <= 0x99)\n"
				"    return 2;\n"
				"  else if (c == 0x9c || c == 0x9d)\n"
				"    return 4;\n"
				"  return 1;\n");
			break;

		case PG_SJIS:	/* logic in pg_sjis_mblen */
		case PG_SHIFT_JIS_2004:
			appendStringInfoString(
				buf,
				"  cl_uchar c = *((const cl_uchar *)str);\n"
				"  if (c >= 0xa1 && c <= 0xdf)\n"
				"    return 1;	/* 1byte kana? */\n"
				"  else if (c & 0x80)\n"
				"    return 2;\n"
				"  return 1;\n");
			break;

		case PG_BIG5:	/* logic in pg_big5_mblen */
		case PG_GBK:	/* logic in pg_gbk_mblen, but identical */
		case PG_UHC:	/* logic in pg_uhc_mblen, but identical */
			appendStringInfoString(
				buf,
				"  cl_uchar c = *((const cl_uchar *)str);\n"
				"  if (c & 0x80)\n"
				"    return 2;\n"
				"  return 1;\n");
			break;

		case PG_GB18030:/* logic in pg_gb18030_mblen */
			appendStringInfoString(
				buf,
				"  cl_uchar c1 = *((const cl_uchar *)str);\n"
				"  cl_uchar c2;\n"
				"  if ((c & 0x80) == 0)\n"
				"    return 1; /* ASCII */\n"
				"  c2 = *((const cl_uchar *)(str + 1));\n"
				"  if (c2 >= 0x30 && c2 <= 0x39)\n"
				"    return 4;\n"
				"  return 2;\n");
			break;

		default:	/* encoding with maxlen==1 */
			if (pg_database_encoding_max_length() != 1)
				elog(ERROR, "Bug? unsupported database encoding: %s",
					 GetDatabaseEncodingName());
			appendStringInfoString(
				buf,
				"  return 1;\n");
			break;
	}
	appendStringInfoString(
		buf,
		"}\n\n");
}

#endif	/* __CUDACC__ */
#endif	/* CUDA_TEXTLIB_H */
