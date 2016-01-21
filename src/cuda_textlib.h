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

#endif	/* __CUDACC__ */
#endif	/* CUDA_TEXTLIB_H */
