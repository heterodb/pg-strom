/*
 * cuda_textlib.h
 *
 * Collection of text functions for OpenCL devices
 * --
 * Copyright 2011-2015 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2015 (C) The PG-Strom Development Team
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
#ifndef PG_BPCHAR_TYPE_DEFINED
#define PG_BPCHAR_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(bpchar)
#endif

STATIC_INLINE(cl_int)
bpchar_truelen(varlena *arg)
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

STATIC_FUNCTION(cl_int)
bpchar_compare(cl_int *errcode, varlena *arg1, varlena *arg2)
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
pgfn_bpchareq(cl_int *errcode, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(bpchar_compare(errcode,
												arg1.value,
												arg2.value) == 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_bpcharne(cl_int *errcode, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(bpchar_compare(errcode,
												arg1.value,
												arg2.value) != 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_bpcharlt(cl_int *errcode, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(bpchar_compare(errcode,
												arg1.value,
												arg2.value) < 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_bpcharle(cl_int *errcode, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(bpchar_compare(errcode,
												arg1.value,
												arg2.value) <= 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_bpchargt(cl_int *errcode, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(bpchar_compare(errcode,
												arg1.value,
												arg2.value) > 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_bpcharge(cl_int *errcode, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(bpchar_compare(errcode,
												arg1.value,
												arg2.value) >= 0);
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_bpcharcmp(cl_int *errcode, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_int4_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = bpchar_compare(errcode, arg1.value, arg2.value);
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_bpcharlen(cl_int *errcode, pg_bpchar_t arg1)
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
text_compare(cl_int *errcode, varlena *arg1, varlena *arg2)
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
pgfn_texteq(cl_int *errcode, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(text_compare(errcode,
											  arg1.value,
											  arg2.value) == 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_textne(cl_int *errcode, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(text_compare(errcode,
											  arg1.value,
											  arg2.value) != 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_text_lt(cl_int *errcode, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(text_compare(errcode,
											  arg1.value,
											  arg2.value) < 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_text_le(cl_int *errcode, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(text_compare(errcode,
											  arg1.value,
											  arg2.value) <= 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_text_gt(cl_int *errcode, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(text_compare(errcode,
											  arg1.value,
											  arg2.value) > 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_text_ge(cl_int *errcode, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (cl_bool)(text_compare(errcode,
											  arg1.value,
											  arg2.value) >= 0);
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_text_cmp(cl_int *errcode, pg_text_t arg1, pg_text_t arg2)
{
	pg_int4_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = text_compare(errcode, arg1.value, arg2.value);
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_textlen(cl_int *errcode, pg_text_t arg1)
{
	pg_int4_t	result;

	/* NOTE: At this moment, we don't support any special encodings,
	 * so no multibytes character is assumed.
	 */
	result.isnull = arg1.isnull;
	if (!result.isnull)
		result.value = toast_raw_datum_size(errcode, arg1.value);
	return result;
}

/*
 * varchar(*) type definition
 */
#ifndef PG_VARCHAR_TYPE_DEFINED
#define PG_VARCHAR_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(varchar)
#endif

#endif	/* __CUDACC__ */
#endif	/* CUDA_TEXTLIB_H */
