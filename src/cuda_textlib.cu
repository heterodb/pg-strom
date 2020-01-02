/*
 * libgputext.cu
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
#include "cuda_textlib.h"

#define CHECK_VARLENA_ARGS(kcxt,result,arg1,arg2)				\
	do {														\
		if (VARATT_IS_COMPRESSED(arg1) ||						\
			VARATT_IS_COMPRESSED(arg2) ||						\
			VARATT_IS_EXTERNAL(arg1) ||							\
			VARATT_IS_EXTERNAL(arg2))							\
		{														\
			STROM_CPU_FALLBACK(kcxt, ERRCODE_STROM_VARLENA_UNSUPPORTED,	\
							   "varlena datum is compressed or external"); \
			(result).isnull = true;								\
			return (result);									\
		}														\
	} while(0)

/*
 * definitions to be made by session info
 */
DEVICE_FUNCTION(cl_int) pg_database_encoding_max_length(void);
DEVICE_FUNCTION(cl_int) pg_wchar_mblen(const char *str);
/* ----------------------------------------------------------------
 *
 * Bpchar Text comparison functions
 * 
 * ----------------------------------------------------------------
 */
DEVICE_INLINE(cl_bool)
pg_bpchar_datum_extract(kern_context *kcxt, pg_bpchar_t arg,
						char **s, cl_int *len)
{
	if (arg.isnull)
		return false;
	if (arg.length < 0)
	{
		if (VARATT_IS_COMPRESSED(arg.value) ||
			VARATT_IS_EXTERNAL(arg.value))
		{
			STROM_CPU_FALLBACK(kcxt, ERRCODE_STROM_VARLENA_UNSUPPORTED,
							   "varlena datum is compressed or external");
			return false;
		}
		*s = VARDATA_ANY(arg.value);
		*len = bpchar_truelen(VARDATA_ANY(arg.value),
							  VARSIZE_ANY_EXHDR(arg.value));
	}
	else
	{
		*s = arg.value;
		*len = arg.length;
	}
	return true;
}

DEVICE_FUNCTION(cl_int)
bpchar_compare(kern_context *kcxt,
			   const char *s1, cl_int len1,
			   const char *s2, cl_int len2,
			   cl_bool *p_isnull)
{
	cl_int		len;

	if (len1 < 0)
	{
		if (VARATT_IS_COMPRESSED(s1) || VARATT_IS_EXTERNAL(s1))
		{
			*p_isnull = true;
			STROM_CPU_FALLBACK(kcxt, ERRCODE_STROM_VARLENA_UNSUPPORTED,
							   "varlena datum is compressed or external");
			return 0;
		}
		len1 = bpchar_truelen(VARDATA_ANY(s1),
							  VARSIZE_ANY_EXHDR(s1));
		s1 = VARDATA_ANY(s1);
	}
	if (len2 < 0)
	{
		if (VARATT_IS_COMPRESSED(s2) || VARATT_IS_EXTERNAL(s2))
		{
			*p_isnull = true;
			STROM_CPU_FALLBACK(kcxt, ERRCODE_STROM_VARLENA_UNSUPPORTED,
							   "varlena datum is compressed or external");
			return 0;
		}
		len2 = bpchar_truelen(VARDATA_ANY(s2),
							  VARSIZE_ANY_EXHDR(s2));
		s2 = VARDATA_ANY(s2);
	}
	len = min(len1, len2);
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

DEVICE_FUNCTION(pg_bool_t)
pgfn_bpchareq(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		result.value = (cl_bool)(bpchar_compare(kcxt,
												arg1.value, arg1.length,
												arg2.value, arg2.length,
												&result.isnull) == 0);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_bpcharne(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		result.value = (cl_bool)(bpchar_compare(kcxt,
												arg1.value, arg1.length,
												arg2.value, arg2.length,
												&result.isnull) != 0);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_bpcharlt(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		result.value = (cl_bool)(bpchar_compare(kcxt,
												arg1.value, arg1.length,
												arg2.value, arg2.length,
												&result.isnull) < 0);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_bpcharle(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		result.value = (cl_bool)(bpchar_compare(kcxt,
												arg1.value, arg1.length,
												arg2.value, arg2.length,
												&result.isnull) <= 0);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_bpchargt(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		result.value = (cl_bool)(bpchar_compare(kcxt,
												arg1.value, arg1.length,
												arg2.value, arg2.length,
												&result.isnull) > 0);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_bpcharge(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		result.value = (cl_bool)(bpchar_compare(kcxt,
												arg1.value, arg1.length,
												arg2.value, arg2.length,
												&result.isnull) >= 0);
	}
	return result;
}

DEVICE_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_int4_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		result.value = bpchar_compare(kcxt,
									  arg1.value, arg1.length,
									  arg2.value, arg2.length,
									  &result.isnull);
	}
	return result;
}

DEVICE_FUNCTION(pg_int4_t)
pgfn_bpcharlen(kern_context *kcxt, pg_bpchar_t arg1)
{
	pg_int4_t	result;

	/* NOTE: At this moment, we don't support any special encodings,
	 * so no multibytes character is assumed.
	 */
	result.isnull = arg1.isnull;
	if (!result.isnull)
	{
		if (arg1.length >= 0)
			result.value = arg1.length;
		else
			result.value = bpchar_truelen(VARDATA_ANY(arg1.value),
										  VARSIZE_ANY_EXHDR(arg1.value));
	}
	return result;
}

/* ----------------------------------------------------------------
 *
 * Basic Text comparison functions
 * 
 * ----------------------------------------------------------------
 */
DEVICE_FUNCTION(cl_int)
text_compare(kern_context *kcxt,
			 pg_text_t arg1,
			 pg_text_t arg2,
			 cl_bool *p_isnull)
{
	char	   *s1, *s2;
	cl_int		len1;
	cl_int		len2;
	cl_int		len;

	if (!pg_varlena_datum_extract(kcxt, arg1, &s1, &len1) ||
		!pg_varlena_datum_extract(kcxt, arg2, &s2, &len2))
	{
		*p_isnull = true;
		return 0;
	}
	len = min(len1, len2);
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

DEVICE_FUNCTION(pg_bool_t)
pgfn_texteq(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		result.value = (cl_bool)(text_compare(kcxt, arg1, arg2,
											  &result.isnull) == 0);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_textne(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		result.value = (cl_bool)(text_compare(kcxt, arg1, arg2,
											  &result.isnull) != 0);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_text_lt(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		result.value = (cl_bool)(text_compare(kcxt, arg1, arg2,
											  &result.isnull) < 0);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_text_le(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		result.value = (cl_bool)(text_compare(kcxt, arg1, arg2,
											  &result.isnull) <= 0);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_text_gt(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		result.value = (cl_bool)(text_compare(kcxt, arg1, arg2,
											  &result.isnull) > 0);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_text_ge(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		result.value = (cl_bool)(text_compare(kcxt, arg1, arg2,
											  &result.isnull) >= 0);
	}
	return result;
}

DEVICE_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_int4_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
	{
		result.value = text_compare(kcxt, arg1, arg2,
									&result.isnull);
	}
	return result;
}

DEVICE_FUNCTION(pg_int4_t)
pgfn_textlen(kern_context *kcxt, pg_text_t arg1)
{
	pg_int4_t	result;
	char	   *s, *end;
	cl_int		j, len;

	result.isnull = arg1.isnull;
	if (!result.isnull)
	{
		if (pg_database_encoding_max_length() == 1)
		{
			if (arg1.length >= 0)
				result.value = arg1.length;
			else
				result.value = toast_raw_datum_size(kcxt, (varlena *)
													arg1.value) - VARHDRSZ;
		}
		else if (pg_varlena_datum_extract(kcxt, arg1, &s, &len))
		{
			end = s + len;
			for (j=0; s < end; j++)
				s += pg_wchar_mblen(s);
			result.value = j;
		}
		else
		{
			result.isnull = true;
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_text_t)
pgfn_textcat(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	return pgfn_text_concat2(kcxt, arg1, arg2);
}

DEVICE_FUNCTION(pg_text_t)
pgfn_text_concat2(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	char	   *s1, *s2;
	cl_int		len1, len2;
	pg_text_t	result;
	char	   *pos;

	if (arg1.isnull || arg2.isnull)
	{
		result.isnull = true;
		return result;
	}
	if (!pg_varlena_datum_extract(kcxt, arg1, &s1, &len1) ||
		!pg_varlena_datum_extract(kcxt, arg2, &s2, &len2))
	{
		result.isnull = true;
		return result;
	}
	pos = (char *)kern_context_alloc(kcxt, VARHDRSZ + len1 + len2);
	if (!pos)
	{
		STROM_CPU_FALLBACK(kcxt, ERRCODE_OUT_OF_MEMORY,
						   "out of memory");
		result.isnull = true;
		return result;
	}
	result.isnull = false;
	result.value = pos;
	result.length = -1;
	SET_VARSIZE(pos, VARHDRSZ + len1 + len2);
	pos += VARHDRSZ;
	memcpy(pos, s1, len1);
	pos += len1;
	memcpy(pos, s2, len2);
	pos += len2;

	return result;
}

DEVICE_FUNCTION(pg_text_t)
pgfn_text_concat3(kern_context *kcxt,
				  pg_text_t arg1, pg_text_t arg2, pg_text_t arg3)
{
	char	   *s1, *s2, *s3;
	cl_int		len1, len2, len3;
	cl_int		sz;
	pg_text_t	result;
	char	   *pos;

	if (arg1.isnull || arg2.isnull || arg3.isnull)
	{
		result.isnull = true;
		return result;
	}
	if (!pg_varlena_datum_extract(kcxt, arg1, &s1, &len1) ||
		!pg_varlena_datum_extract(kcxt, arg2, &s2, &len2) ||
		!pg_varlena_datum_extract(kcxt, arg3, &s3, &len3))
	{
		result.isnull = true;
		return result;
	}
	sz = VARHDRSZ + len1 + len2 + len3;
	pos = (char *)kern_context_alloc(kcxt, sz);
	if (!pos)
	{
		STROM_CPU_FALLBACK(kcxt, ERRCODE_OUT_OF_MEMORY,
						   "out of memory");
		result.isnull = true;
		return result;
	}
	result.isnull = false;
	result.value = pos;
	result.length = -1;
	SET_VARSIZE(pos, sz);
	pos += VARHDRSZ;
	memcpy(pos, s1, len1);
	pos += len1;
	memcpy(pos, s2, len2);
	pos += len2;
	memcpy(pos, s3, len3);
	pos += len3;

	return result;
}

DEVICE_FUNCTION(pg_text_t)
pgfn_text_concat4(kern_context *kcxt,
				  pg_text_t arg1, pg_text_t arg2,
				  pg_text_t arg3, pg_text_t arg4)
{
	char	   *s1, *s2, *s3, *s4;
	cl_int		len1, len2, len3, len4;
	cl_int		sz;
	pg_text_t	result;
	char	   *pos;

	if (arg1.isnull || arg2.isnull || arg3.isnull || arg4.isnull)
	{
		result.isnull = true;
		return result;
	}
	if (!pg_varlena_datum_extract(kcxt, arg1, &s1, &len1) ||
		!pg_varlena_datum_extract(kcxt, arg2, &s2, &len2) ||
		!pg_varlena_datum_extract(kcxt, arg3, &s3, &len3) ||
		!pg_varlena_datum_extract(kcxt, arg4, &s4, &len4))
	{
		result.isnull = true;
		return result;
	}
	sz = VARHDRSZ + len1 + len2 + len3 + len4;
	pos = (char *)kern_context_alloc(kcxt, sz);
	if (!pos)
	{
		STROM_CPU_FALLBACK(kcxt, ERRCODE_OUT_OF_MEMORY,
						   "out of memory");
		result.isnull = true;
		return result;
	}
	result.isnull = false;
	result.value = pos;
	result.length = -1;
	SET_VARSIZE(pos, sz);
	pos += VARHDRSZ;
	memcpy(pos, s1, len1);
	pos += len1;
	memcpy(pos, s2, len2);
	pos += len2;
	memcpy(pos, s3, len3);
	pos += len3;
	memcpy(pos, s4, len4);
	pos += len4;

	return result;
}

/*
 * substring
 */
DEVICE_FUNCTION(pg_text_t)
text_substring(kern_context *kcxt,
			   char *str, cl_int strlen, cl_int start, cl_int length)
{
	pg_text_t	result;
	char	   *pos;

	if (pg_database_encoding_max_length() == 1)
	{
		cl_int	tail = start + length - 1;

		start = Max(start, 1) - 1;	/* 0-origin */
		if (start >= strlen)
			goto empty;
		pos = str + start;

		if (length < 0 || tail >= strlen)
			length = strlen - start;
		else
			length = tail - start;
		/* substring */
		result.isnull = false;
		result.value  = pos;
		result.length = length;
	}
	else
	{
		cl_int	tail = start + length - 1;
		char   *end = str + strlen;
		char   *mark;

		start = Max(start, 1) - 1;
		if (length < 0)
			length = INT_MAX;
		else
			length = tail - start;
		pos = str;
		while (start-- > 0 && pos < end)
			pos += pg_wchar_mblen(pos);
		if (pos >= end)
			goto empty;
		mark = pos;
		if (length < 0)
			length = INT_MAX;
		while (length-- > 0 && pos < end)
			pos += pg_wchar_mblen(pos);
		if (pos >= end)
			pos = end;
		result.isnull = false;
		result.value  = mark;
		result.length = (pos - mark);
	}
	return result;

empty:
	result.isnull = false;
	result.value  = str + strlen;
	result.length = 0;
	return result;
}

DEVICE_FUNCTION(pg_text_t)
pgfn_text_substring(kern_context *kcxt,
					pg_text_t arg1, pg_int4_t arg2, pg_int4_t arg3)
{
	pg_text_t	result;
	char	   *s1;
	cl_int		len1;

	if (arg1.isnull || arg2.isnull || arg3.isnull)
		result.isnull = true;
	else if (arg3.value < 0)
	{
		result.isnull = true;
		STROM_EREPORT(kcxt, ERRCODE_SUBSTRING_ERROR,
					  "negative substring length not allowed");
	}
	else if (!pg_varlena_datum_extract(kcxt, arg1, &s1, &len1))
		result.isnull = true;
	else
		result = text_substring(kcxt, s1, len1, arg2.value, arg3.value);

	return result;
}

DEVICE_FUNCTION(pg_text_t)
pgfn_text_substring_nolen(kern_context *kcxt,
						  pg_text_t arg1, pg_int4_t arg2)
{
	pg_text_t	result;
	char	   *s1;
	cl_int		len1;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else if (!pg_varlena_datum_extract(kcxt, arg1, &s1, &len1))
		result.isnull = true;
	else
		result = text_substring(kcxt, s1, len1, arg2.value, -1);

	return result;
}

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
					STROM_EREPORT(kcxt, ERRCODE_INVALID_ESCAPE_SEQUENCE,\
								  "invalid escape in LIKE pattern");	\
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
					if (plen < 2)										\
					{													\
						STROM_EREPORT(kcxt,								\
							ERRCODE_INVALID_ESCAPE_SEQUENCE,			\
							"invalid escape in LIKE pattern");			\
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
							STROM_CPU_FALLBACK(kcxt,					\
								ERRCODE_STROM_RECURSION_TOO_DEEP,		\
								"too deep recursive function call");	\
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

DEVICE_FUNCTION(pg_bool_t)
pgfn_textlike(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		char	   *s, *p;
		cl_int		slen;
		cl_int		plen;

		if (!pg_varlena_datum_extract(kcxt, arg1, &s, &slen) ||
			!pg_varlena_datum_extract(kcxt, arg2, &p, &plen))
		{
			result.isnull = true;
			return result;
		}
		result.value = (GenericMatchText(kcxt,
										 s, slen,
										 p, plen, 0) == LIKE_TRUE);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_textnlike(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		char	   *s, *p;
		cl_int		slen;
		cl_int		plen;

		if (!pg_varlena_datum_extract(kcxt, arg1, &s, &slen) ||
			!pg_varlena_datum_extract(kcxt, arg2, &p, &plen))
		{
			result.isnull = true;
			return result;
		}
		result.value = (GenericMatchText(kcxt,
										 s, slen,
										 p, plen, 0) != LIKE_TRUE);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_bpcharlike(kern_context *kcxt, pg_bpchar_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		char	   *s, *p;
		cl_int		slen;
		cl_int		plen;

		if (!pg_bpchar_datum_extract(kcxt, arg1, &s, &slen) ||
			!pg_varlena_datum_extract(kcxt, arg2, &p, &plen))
		{
			result.isnull = true;
			return result;
		}
		result.value = (GenericMatchText(kcxt,
										 s, slen,
										 p, plen, 0) == LIKE_TRUE);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_bpcharnlike(kern_context *kcxt, pg_bpchar_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		char	   *s, *p;
		cl_int		slen;
		cl_int		plen;

		if (!pg_bpchar_datum_extract(kcxt, arg1, &s, &slen) ||
			!pg_varlena_datum_extract(kcxt, arg2, &p, &plen))
		{
			result.isnull = true;
			return result;
		}
		result.value = (GenericMatchText(kcxt,
										 s, slen,
										 p, plen, 0) != LIKE_TRUE);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_texticlike(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		char	   *s, *p;
		cl_int		slen;
		cl_int		plen;

		if (!pg_varlena_datum_extract(kcxt, arg1, &s, &slen) ||
			!pg_varlena_datum_extract(kcxt, arg2, &p, &plen))
		{
			result.isnull = true;
			return result;
		}
		result.value = (GenericCaseMatchText(kcxt,
											 s, slen,
											 p, plen, 0) == LIKE_TRUE);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_texticnlike(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		char	   *s, *p;
		cl_int		slen;
		cl_int		plen;

		if (!pg_varlena_datum_extract(kcxt, arg1, &s, &slen) ||
			!pg_varlena_datum_extract(kcxt, arg2, &p, &plen))
		{
			result.isnull = true;
			return result;
		}
		result.value = (GenericCaseMatchText(kcxt,
											 s, slen,
											 p, plen, 0) != LIKE_TRUE);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_bpchariclike(kern_context *kcxt, pg_bpchar_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		char	   *s, *p;
		cl_int		slen;
		cl_int		plen;

		if (!pg_bpchar_datum_extract(kcxt, arg1, &s, &slen) ||
			!pg_varlena_datum_extract(kcxt, arg2, &p, &plen))
		{
			result.isnull = true;
			return result;
		}
		result.value = (GenericCaseMatchText(kcxt,
											 s, slen,
											 p, plen, 0) == LIKE_TRUE);
	}
	return result;
}

DEVICE_FUNCTION(pg_bool_t)
pgfn_bpcharicnlike(kern_context *kcxt, pg_bpchar_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		char	   *s, *p;
		cl_int		slen;
		cl_int		plen;

		if (!pg_bpchar_datum_extract(kcxt, arg1, &s, &slen) ||
			!pg_varlena_datum_extract(kcxt, arg2, &p, &plen))
		{
			result.isnull = true;
			return result;
		}
		result.value = (GenericCaseMatchText(kcxt,
											 s, slen,
											 p, plen, 0) != LIKE_TRUE);
	}
	return result;
}

#undef LIKE_TRUE
#undef LIKE_FALSE
#undef LIKE_ABORT
