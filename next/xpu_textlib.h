/*
 * xpu_textlib.h
 *
 * Misc definitions for xPU(GPU/DPU/SPU).
 * --
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef XPU_TEXTLIB_H
#define XPU_TEXTLIB_H

#define PGSTROM_VARLENA_BASETYPE_TEMPLATE(NAME)							\
	STATIC_FUNCTION(bool)												\
	xpu_##NAME##_datum_ref(kern_context *kcxt,							\
						   xpu_datum_t *__result,						\
						   const void *addr)							\
	{																	\
		xpu_##NAME##_t *result = (xpu_##NAME##_t *)__result;			\
																		\
		memset(result, 0, sizeof(xpu_##NAME##_t));						\
		if (!addr)														\
			result->isnull = true;										\
		else															\
			result->value = (char *)addr;								\
		result->ops = &xpu_##NAME##_ops;								\
		return true;													\
	}																	\
	STATIC_FUNCTION(bool)												\
	arrow_##NAME##_datum_ref(kern_context *kcxt,						\
							 xpu_datum_t *__result,						\
							 kern_data_store *kds,						\
							 kern_colmeta *cmeta,						\
							 uint32_t rowidx)							\
	{																	\
		xpu_##NAME##_t *result = (xpu_##NAME##_t *)__result;			\
		void	   *addr;												\
		uint32_t	length;												\
																		\
		addr = KDS_ARROW_REF_VARLENA_DATUM(kds, cmeta, rowidx,			\
										  &length);						\
		memset(result, 0, sizeof(xpu_##NAME##_t));						\
		if (!addr)														\
			result->isnull = true;										\
		else															\
		{																\
			result->length = length;									\
			result->value = (char *)addr;								\
		}																\
		result->ops = &xpu_##NAME##_ops;								\
		return true;													\
	}																	\
	STATIC_FUNCTION(int)												\
	xpu_##NAME##_datum_store(kern_context *kcxt,						\
							 char *buffer,								\
							 xpu_datum_t *__arg)						\
	{																	\
		xpu_##NAME##_t *arg = (xpu_##NAME##_t *)__arg;					\
		char	   *data;												\
		uint32_t	len;												\
																		\
		if (arg->isnull)												\
			return 0;													\
		if (arg->length < 0)											\
		{																\
			data = VARDATA_ANY(arg->value);								\
			len = VARSIZE_ANY_EXHDR(arg->value);						\
		}																\
		else															\
		{																\
			data = arg->value;											\
			len = arg->length;											\
		}																\
		if (buffer)														\
		{																\
			memcpy(buffer + VARHDRSZ, data, len);						\
			SET_VARSIZE(buffer, VARHDRSZ + len);						\
		}																\
		return VARHDRSZ + len;											\
	}																	\
	STATIC_FUNCTION(bool)												\
	xpu_##NAME##_datum_hash(kern_context *kcxt,							\
							uint32_t *p_hash,							\
							xpu_datum_t *__arg)							\
	{																	\
		xpu_##NAME##_t *arg = (xpu_##NAME##_t *)__arg;					\
																		\
		if (arg->isnull)												\
			*p_hash = 0;												\
		else if (arg->length < 0)										\
			*p_hash = pg_hash_any(VARDATA_ANY(arg->value),				\
								  VARSIZE_ANY_EXHDR(arg->value));		\
		else															\
			*p_hash = pg_hash_any(arg->value, arg->length);				\
		return true;													\
	}																	\
	PGSTROM_SQLTYPE_OPERATORS(NAME)

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


#endif  /* XPU_TEXTLIB_H */
