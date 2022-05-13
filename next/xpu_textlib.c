/*
 * xpu_textlib.c
 *
 * Collection of text functions and operators for xPU(GPU/DPU/SPU)
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 *
 */
#include "xpu_common.h"

PGSTROM_VARLENA_BASETYPE_TEMPLATE(bytea);
PGSTROM_VARLENA_BASETYPE_TEMPLATE(text);
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

STATIC_FUNCTION(bool)
xpu_bpchar_datum_ref(kern_context *kcxt,
					 xpu_datum_t *__result,
					 const void *addr)
{
	xpu_bpchar_t *result = (xpu_bpchar_t *)__result;
	memset(result, 0, sizeof(xpu_bpchar_t));
	if (!addr)
		result->isnull = true;
	else
	{
		result->length = -1;
		result->value = (char *)addr;
	}
	result->ops = &xpu_bpchar_ops;
	return true;
}

STATIC_FUNCTION(bool)
arrow_bpchar_datum_ref(kern_context *kcxt,
					   xpu_datum_t *__result,
					   kern_data_store *kds,
					   kern_colmeta *cmeta,
					   uint32_t rowidx)
{
	xpu_bpchar_t *result = (xpu_bpchar_t *)__result;
	int		unitsz = cmeta->attopts.fixed_size_binary.byteWidth;
	char   *addr = NULL;

	if (unitsz > 0)
		addr = (char *)KDS_ARROW_REF_SIMPLE_DATUM(kds, cmeta, rowidx, unitsz);
	memset(result, 0, sizeof(xpu_bpchar_t));
	if (!addr)
		result->isnull = true;
	else
	{
		result->value = (char *)addr;
		result->length = bpchar_truelen(addr, unitsz);
	}
	result->ops = &xpu_bpchar_ops;
	return true;
}

STATIC_FUNCTION(int)
xpu_bpchar_datum_store(kern_context *kcxt,
					   char *buffer,
					   xpu_datum_t *__arg)
{
	xpu_bpchar_t *arg = (xpu_bpchar_t *)__arg;
	char   *data;
	int		len;

	if (arg->isnull)
		return 0;
	if (arg->length < 0)
	{
		data = VARDATA_ANY(arg->value);
		len = VARSIZE_ANY_EXHDR(arg->value);
		if (!VARATT_IS_COMPRESSED(data) &&
			!VARATT_IS_EXTERNAL(data))
			len = bpchar_truelen(data, len);
	}
	else
	{
		data = arg->value;
		len = bpchar_truelen(data, arg->length);
	}
	if (buffer)
	{
		memcpy(buffer + VARHDRSZ, data, len);
		SET_VARSIZE(buffer, len + VARHDRSZ);
	}
	return len + VARHDRSZ;
}

STATIC_FUNCTION(bool)
xpu_bpchar_datum_hash(kern_context*kcxt,
					  uint32_t *p_hash,
					  xpu_datum_t *__arg)
{
	xpu_bpchar_t *arg = (xpu_bpchar_t *)__arg;
	char   *data;
	int		len;

	if (arg->isnull)
		*p_hash = 0;
	else
	{
		if (arg->length >= 0)
		{
			data = arg->value;
			len = arg->length;
		}
		else if (!VARATT_IS_COMPRESSED(arg->value) &&
				 !VARATT_IS_EXTERNAL(arg->value))
		{
			data = VARDATA_ANY(arg->value);
			len = VARSIZE_ANY_EXHDR(arg->value);
		}
		else
		{
			STROM_CPU_FALLBACK(kcxt, ERRCODE_STROM_VARLENA_UNSUPPORTED,
							   "bpchar datum is compressed or external");
			return false;
		}
		len = bpchar_truelen(data, len);
		*p_hash = pg_hash_any(data, len);
	}
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(bpchar);
