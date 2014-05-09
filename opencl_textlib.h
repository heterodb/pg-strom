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
	cl_char	   *s1 = VARDATA_ANY(arg1);
	cl_char	   *s2 = VARDATA_ANY(arg2);
	cl_int		len1 = VARSIZE_ANY(arg1);
	cl_int		len2 = VARSIZE_ANY(arg2);
	cl_int		len = min(len1, len2);
	cl_uint		v1;
	cl_uint		v2;
	cl_uint		mask;

	/* XXX - to be revised to GPU style */
	while (len > 0)
	{
		v1 = *((__global cl_uint *)s1);
		v2 = *((__global cl_uint *)s2);
		mask = v1 ^ v2;
		if (mask != 0)
		{
			if (len > 0 && (mask & 0x000000ffU) != 0)
				return ((v1 & 0x000000ffU) > (v2 & 0x000000ffU) ? 1 : -1);
			else if (len > 1 && (mask & 0x0000ff00U) != 0)
				return ((v1 & 0x0000ff00U) > (v2 & 0x0000ff00U) ? 1 : -1);
			else if (len > 2 && (mask & 0x00ff0000U) != 0)
				return ((v1 & 0x00ff0000U) > (v2 & 0x00ff0000U) ? 1 : -1);
			else if (len > 3)
				return ((v1 & 0xff000000U) > (v2 & 0xff000000U) ? 1 : -1);
		}
		s1 += sizeof(cl_uint);
		s2 += sizeof(cl_uint);
		len -= sizeof(cl_uint);
	}
	if (len1 != len2)
		return (len1 > len2 ? 1 : -1);
	return 0;
}

#ifndef PG_BPCHAR_TYPE_DEFINED
#define PG_BPCHAR_TYPE_DEFINED
STROMCL_VRALENA_TYPE_TEMPLATE(bpchar)
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

#ifndef PG_TEXT_TYPE_DEFINED
#define PG_TEXT_TYPE_DEFINED
STROMCL_VRALENA_TYPE_TEMPLATE(text)
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

#endif	/* OPENCL_DEVICE_CODE */
#endif	/* OPENCL_TEXTLIB_H */
