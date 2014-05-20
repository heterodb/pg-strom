/*
 * opencl_textlib.h
 *
 * Collection of text functions for OpenCL devices
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef OPENCL_TEXTLIB_H
#define OPENCL_TEXTLIB_H
#ifdef OPENCL_DEVICE_CODE

static int
varlena_cmp(__private cl_int *errcode,
			__global varlena *arg1, __global varlena *arg2)
{
	__global cl_char *s1 = VARDATA_ANY(arg1);
	__global cl_char *s2 = VARDATA_ANY(arg2);
	cl_int		len1 = VARSIZE_ANY_EXHDR(arg1);
	cl_int		len2 = VARSIZE_ANY_EXHDR(arg2);
	cl_int		len = min(len1, len2);

	/*
	 * XXX - to be revised for more GPU/MIC confortable coding style.
	 * Once thing you need to pay attention is varlena variables may
	 * be unaligned if short format, thus it leads unaligned data
	 * access, then eventually leads kernel crash.
	 */
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

#ifndef PG_BPCHAR_TYPE_DEFINED
#define PG_BPCHAR_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(bpchar)
#endif

static pg_bool_t
pgfn_bpchareq(__private cl_int *errcode, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (varlena_cmp(errcode, arg1.value, arg2.value) == 0);
	return result;
}

static pg_bool_t
pgfn_bpcharne(__private cl_int *errcode, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (varlena_cmp(errcode, arg1.value, arg2.value) != 0);
	return result;
}

static pg_bool_t
pgfn_bpcharlt(__private cl_int *errcode, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (varlena_cmp(errcode, arg1.value, arg2.value) < 0);
	return result;
}

static pg_bool_t
pgfn_bpcharle(__private cl_int *errcode, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (varlena_cmp(errcode, arg1.value, arg2.value) <= 0);
	return result;
}

static pg_bool_t
pgfn_bpchargt(__private cl_int *errcode, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (varlena_cmp(errcode, arg1.value, arg2.value) > 0);
	return result;
}

static pg_bool_t
pgfn_bpcharge(__private cl_int *errcode, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (varlena_cmp(errcode, arg1.value, arg2.value) >= 0);
	return result;
}

static pg_int4_t
pgfn_bpcharcmp(__private cl_int *errcode, pg_bpchar_t arg1, pg_bpchar_t arg2)
{
	pg_int4_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = varlena_cmp(errcode, arg1.value, arg2.value);
	return result;
}

#ifndef PG_TEXT_TYPE_DEFINED
#define PG_TEXT_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(text)
#endif

static pg_bool_t
pgfn_texteq(__private cl_int *errcode, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (bool)(varlena_cmp(errcode,
										  arg1.value,
										  arg2.value) == 0);
	return result;
}

static pg_bool_t
pgfn_textne(__private cl_int *errcode, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (bool)(varlena_cmp(errcode,
										  arg1.value,
										  arg2.value) != 0);
	return result;
}

static pg_bool_t
pgfn_text_lt(__private cl_int *errcode, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (bool)(varlena_cmp(errcode,
										  arg1.value,
										  arg2.value) < 0);
	return result;
}

static pg_bool_t
pgfn_text_le(__private cl_int *errcode, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (bool)(varlena_cmp(errcode,
										  arg1.value,
										  arg2.value) <= 0);
	return result;
}

static pg_bool_t
pgfn_text_gt(__private cl_int *errcode, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (bool)(varlena_cmp(errcode,
										  arg1.value,
										  arg2.value) > 0);
	return result;
}

static pg_bool_t
pgfn_text_ge(__private cl_int *errcode, pg_text_t arg1, pg_text_t arg2)
{
	pg_bool_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = (bool)(varlena_cmp(errcode,
										  arg1.value,
										  arg2.value) >= 0);
	return result;
}

static pg_int4_t
pgfn_text_cmp(__private cl_int *errcode, pg_text_t arg1, pg_text_t arg2)
{
	pg_int4_t	result;

	result.isnull = (arg1.isnull | arg2.isnull);
	if (!result.isnull)
		result.value = varlena_cmp(errcode, arg1.value, arg2.value);
	return result;
}

#endif	/* OPENCL_DEVICE_CODE */
#endif	/* OPENCL_TEXTLIB_H */
