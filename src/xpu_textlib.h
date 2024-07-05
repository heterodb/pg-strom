/*
 * xpu_textlib.h
 *
 * Misc definitions for text routines for both of GPU and DPU
 * --
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef XPU_TEXTLIB_H
#define XPU_TEXTLIB_H

#ifndef PG_BYTEAOID
#define PG_BYTEAOID		17
#endif
#ifndef PG_TEXTOID
#define PG_TEXTOID		25
#endif
PGSTROM_SQLTYPE_VARLENA_DECLARATION(bytea);
PGSTROM_SQLTYPE_VARLENA_DECLARATION(text);
PGSTROM_SQLTYPE_VARLENA_DECLARATION(bpchar);

/*
 * Database Encoding Info
 */
struct xpu_encode_info {
	char	encname[16];
	int		enc_maxlen;
	int	  (*enc_mblen)(const char *s);
};
typedef struct xpu_encode_info	xpu_encode_info;

EXTERN_DATA xpu_encode_info		xpu_encode_catalog[];

/*
 * validation checkers
 */
INLINE_FUNCTION(bool)
xpu_bpchar_is_valid(kern_context *kcxt, const xpu_bpchar_t *arg)
{
	if (arg->length < 0)
	{
		SUSPEND_FALLBACK(kcxt, "bpchar datum is compressed or external");
		return false;
	}
	return true;
}

INLINE_FUNCTION(bool)
xpu_text_is_valid(kern_context *kcxt, const xpu_text_t *arg)
{
	if (arg->length < 0)
	{
		SUSPEND_FALLBACK(kcxt, "text datum is compressed or external");
		return false;
	}
	return true;
}

INLINE_FUNCTION(bool)
xpu_bytea_is_valid(kern_context *kcxt, const xpu_bytea_t *arg)
{
	if (arg->length < 0)
	{
		SUSPEND_FALLBACK(kcxt, "bytea datum is compressed or external");
		return false;
	}
	return true;
}

#endif  /* XPU_TEXTLIB_H */
