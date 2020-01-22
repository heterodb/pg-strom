/*
 * cuda_textlib.h
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
#ifndef CUDA_TEXTLIB_H
#define CUDA_TEXTLIB_H
#ifdef __CUDACC__
DEVICE_INLINE(cl_int)
bpchar_truelen(const char *s, cl_int len)
{
	cl_int		i;

	for (i = len - 1; i >= 0; i--)
	{
		if (s[i] != ' ')
			break;
	}
	return i + 1;
}
/* basic comparison of bpchar */
DEVICE_FUNCTION(pg_bool_t)
pgfn_bpchareq(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_bpcharne(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_bpcharlt(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_bpcharle(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_bpchargt(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_bpcharge(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2);
DEVICE_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt, pg_bpchar_t arg1, pg_bpchar_t arg2);
DEVICE_FUNCTION(pg_int4_t)
pgfn_bpcharlen(kern_context *kcxt, pg_bpchar_t arg1);
/* basic comparison of text */
DEVICE_FUNCTION(pg_bool_t)
pgfn_texteq(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_textne(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_text_lt(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_text_le(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_text_gt(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_text_ge(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2);

/* other primitive text functions */
DEVICE_FUNCTION(pg_int4_t)
pgfn_textlen(kern_context *kcxt, pg_text_t arg1);
DEVICE_FUNCTION(pg_text_t)
pgfn_textcat(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_text_t)
pgfn_text_concat2(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_text_t)
pgfn_text_concat3(kern_context *kcxt,
				  pg_text_t arg1, pg_text_t arg2, pg_text_t arg3);
DEVICE_FUNCTION(pg_text_t)
pgfn_text_concat4(kern_context *kcxt,
				  pg_text_t arg1, pg_text_t arg2,
				  pg_text_t arg3, pg_text_t arg4);
DEVICE_FUNCTION(pg_text_t)
pgfn_text_substring(kern_context *kcxt,
					pg_text_t arg1, pg_int4_t arg2, pg_int4_t arg3);
DEVICE_FUNCTION(pg_text_t)
pgfn_text_substring_nolen(kern_context *kcxt,
						  pg_text_t arg1, pg_int4_t arg2);
/* binary compatible type cast */
DEVICE_INLINE(pg_text_t)
to_text(pg_varchar_t arg)
{
	return arg;
}

DEVICE_INLINE(pg_bpchar_t)
to_bpchar(pg_text_t arg)
{
	pg_bpchar_t		r;
	r.isnull = arg.isnull;
	r.value  = arg.value;
	r.length = arg.length;
	return r;
}
DEVICE_INLINE(pg_varchar_t)
to_varchar(pg_text_t arg)
{
	return arg;
}
/* LIKE operator */
DEVICE_FUNCTION(pg_bool_t)
pgfn_textlike(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_textnlike(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_bpcharlike(kern_context *kcxt, pg_bpchar_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_bpcharnlike(kern_context *kcxt, pg_bpchar_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_texticlike(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_texticnlike(kern_context *kcxt, pg_text_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_bpchariclike(kern_context *kcxt, pg_bpchar_t arg1, pg_text_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_bpcharicnlike(kern_context *kcxt, pg_bpchar_t arg1, pg_text_t arg2);
#endif	/* __CUDACC__ */
#endif	/* CUDA_TEXTLIB_H */
