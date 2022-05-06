/*
 * xpu_basetype.h
 *
 * Collection of base Int/Float functions for xPU (GPU/DPU/SPU)
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef XPU_BASETYPE_H
#define XPU_BASETYPE_H

#define PGSTROM_SIMPLE_BASETYPE_TEMPLATE(NAME,BASETYPE)					\
	STATIC_FUNCTION(bool)												\
	sql_##NAME##_datum_ref(kern_context *kcxt,							\
						   sql_datum_t *__result,						\
						   void *addr)									\
	{																	\
		sql_##NAME##_t *result = (sql_##NAME##_t *)__result;			\
																		\
		memset(result, 0, sizeof(sql_##NAME##_t));						\
		if (!addr)														\
			result->isnull = true;										\
		else															\
			result->value = *((BASETYPE *)addr);						\
		result->ops = &sql_##NAME##_ops;								\
		return true;													\
	}																	\
	STATIC_FUNCTION(bool)												\
	arrow_##NAME##_datum_ref(kern_context *kcxt,						\
							 sql_datum_t *__result,						\
							 kern_data_store *kds,						\
							 kern_colmeta *cmeta,						\
							 uint32_t rowidx)							\
	{																	\
		void	   *addr;												\
																		\
		addr = KDS_ARROW_REF_SIMPLE_DATUM(kds, cmeta, rowidx,			\
										  sizeof(BASETYPE));			\
		return sql_##NAME##_datum_ref(kcxt, __result, addr);			\
	}																	\
	STATIC_FUNCTION(int)												\
	sql_##NAME##_datum_store(kern_context *kcxt,						\
							 char *buffer,								\
							 sql_datum_t *__arg)						\
	{																	\
		sql_##NAME##_t *arg = (sql_##NAME##_t *)__arg;					\
																		\
		if (arg->isnull)												\
			return 0;													\
		*((BASETYPE *)buffer) = arg->value;								\
		return sizeof(BASETYPE);										\
	}																	\
	STATIC_FUNCTION(bool)												\
	sql_##NAME##_datum_hash(kern_context *kcxt,							\
							uint32_t *p_hash,							\
							sql_datum_t *__arg)							\
	{																	\
		sql_##NAME##_t *arg = (sql_##NAME##_t *)__arg;					\
																		\
		if (arg->isnull)												\
			*p_hash = 0;												\
		else															\
			*p_hash = pg_hash_any(&arg->value, sizeof(BASETYPE));		\
		return true;													\
	}																	\
	PGSTROM_SQLTYPE_OPERATORS(NAME)
		
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(bool, int8_t);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(int1, int8_t);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(int2, int16_t);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(int4, int32_t);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(int8, int64_t);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(float2, float2_t);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(float4, float4_t);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(float8, float8_t);

#endif	/* XPU_BASETYPE_H */
