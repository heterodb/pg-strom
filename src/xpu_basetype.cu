/*
 * xpu_basetype.cc
 *
 * Collection of primitive Int/Float type support for both of GPU and DPU
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 *
 */
#include "xpu_common.h"
#include <math.h>

/*
 * Bool type handlers
 */
STATIC_FUNCTION(bool)
xpu_bool_datum_heap_read(kern_context *kcxt,
						 const void *addr,
						 xpu_datum_t *__result)
{
	xpu_bool_t *result = (xpu_bool_t *)__result;

	if (!addr)
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_bool_ops;
		result->value = *((const bool *)addr);
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_bool_datum_arrow_read(kern_context *kcxt,
						  const kern_data_store *kds,
						  const kern_colmeta *cmeta,
						  uint32_t kds_index,
						  xpu_datum_t *__result)
{
	xpu_bool_t *result = (xpu_bool_t *)__result;

	if (KDS_ARROW_CHECK_ISNULL(kds, cmeta, kds_index))
		result->expr_ops = NULL;
	else
	{
		const uint8_t  *bitmap = (const uint8_t *)
			((const char *)kds + cmeta->values_offset);
		uint8_t			mask = (1<<(kds_index & 7));

		result->expr_ops = &xpu_bool_ops;
		result->value    = ((bitmap[kds_index>>3] & mask) != 0);
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_bool_datum_kvec_load(kern_context *kcxt,
						 const kvec_datum_t *__kvecs,
						 uint32_t kvecs_id,
						 xpu_datum_t *__result)
{
	const kvec_bool_t *kvecs = (const kvec_bool_t *)__kvecs;
	xpu_bool_t *result = (xpu_bool_t *)__result;

	result->expr_ops = &xpu_bool_ops;
	result->value = kvecs->values[kvecs_id];

	return true;
}

STATIC_FUNCTION(bool)
xpu_bool_datum_kvec_save(kern_context *kcxt,
						 const xpu_datum_t *__xdatum,
						 kvec_datum_t *__kvec_dst,
						 uint32_t kvec_dst_id)
{
	const xpu_bool_t *xdatum = (const xpu_bool_t *)__xdatum;
	kvec_bool_t *kvecs_dst = (kvec_bool_t *)__kvec_dst;

	assert(!XPU_DATUM_ISNULL(xdatum));
	kvecs_dst->values[kvec_dst_id] = xdatum->value;

	return true;
}

STATIC_FUNCTION(bool)
xpu_bool_datum_kvec_copy(kern_context *kcxt,
						 const kvec_datum_t *__kvecs_src,
						 uint32_t kvecs_src_id,
						 kvec_datum_t *__kvecs_dst,
						 uint32_t kvecs_dst_id)
{
	const kvec_bool_t *kvecs_src = (const kvec_bool_t *)__kvecs_src;
	kvec_bool_t *kvecs_dst = (kvec_bool_t *)__kvecs_dst;

	kvecs_dst->values[kvecs_dst_id] = kvecs_src->values[kvecs_src_id];
	return true;
}

STATIC_FUNCTION(int)
xpu_bool_datum_write(kern_context *kcxt,
					 char *buffer,
					 const kern_colmeta *cmeta,
					 const xpu_datum_t *__arg)
{
	const xpu_bool_t   *arg = (const xpu_bool_t *)__arg;

	if (buffer)
		*((bool *)buffer) = arg->value;
	return sizeof(bool);
}

STATIC_FUNCTION(bool)
xpu_bool_datum_hash(kern_context *kcxt,
					uint32_t *p_hash,
					xpu_datum_t *__arg)
{
	xpu_bool_t *arg = (xpu_bool_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(bool));
	return true;
}

STATIC_FUNCTION(bool)
xpu_bool_datum_comp(kern_context *kcxt,
					int *p_comp,
					xpu_datum_t *__a,
					xpu_datum_t *__b)
{
	xpu_bool_t *a = (xpu_bool_t *)__a;
	xpu_bool_t *b = (xpu_bool_t *)__b;

	assert(!XPU_DATUM_ISNULL(a) && !XPU_DATUM_ISNULL(b));
	*p_comp = ((int)a->value - (int)b->value);
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(bool,true,1,sizeof(bool));

/*
 * Int1/Int2/Int4/Int8 and Float2/Float4/Float8 handlers
 */
#define __PGSTROM_SIMPLE_INT_FLOAT_TEMPLATE(NAME,BASETYPE)				\
	STATIC_FUNCTION(bool)												\
	xpu_##NAME##_datum_heap_read(kern_context *kcxt,					\
								 const void *addr,						\
								 xpu_datum_t *__result)					\
	{																	\
		xpu_##NAME##_t *result = (xpu_##NAME##_t *)__result;			\
																		\
		result->expr_ops = &xpu_##NAME##_ops;							\
		__FetchStore(result->value, (const BASETYPE *)addr);			\
		return true;													\
	}																	\
	STATIC_FUNCTION(bool)												\
	xpu_##NAME##_datum_kvec_load(kern_context *kcxt,					\
								 const kvec_datum_t *__kvecs,			\
								 uint32_t kvecs_id,						\
								 xpu_datum_t *__result)					\
	{																	\
		const kvec_##NAME##_t *kvecs = (const kvec_##NAME##_t *)__kvecs; \
		xpu_##NAME##_t *result = (xpu_##NAME##_t *)__result;			\
																		\
		result->expr_ops = &xpu_##NAME##_ops;							\
		result->value = kvecs->values[kvecs_id];						\
		return true;													\
	}																	\
	STATIC_FUNCTION(bool)												\
	xpu_##NAME##_datum_kvec_save(kern_context *kcxt,					\
								 const xpu_datum_t *__xdatum,			\
								 kvec_datum_t *__kvecs,					\
								 uint32_t kvecs_id)						\
	{																	\
		const xpu_##NAME##_t *xdatum = (const xpu_##NAME##_t *)__xdatum; \
		kvec_##NAME##_t *kvecs = (kvec_##NAME##_t *)__kvecs;			\
																		\
		kvecs->values[kvecs_id] = xdatum->value;						\
		return true;													\
	}																	\
	STATIC_FUNCTION(bool)												\
	xpu_##NAME##_datum_kvec_copy(kern_context *kcxt,					\
								 const kvec_datum_t *__kvecs_src,		\
								 uint32_t kvecs_src_id,					\
								 kvec_datum_t *__kvecs_dst,				\
								 uint32_t kvecs_dst_id)					\
	{																	\
		const kvec_##NAME##_t *kvecs_src = (const kvec_##NAME##_t *)__kvecs_src; \
		kvec_##NAME##_t *kvecs_dst = (kvec_##NAME##_t *)__kvecs_dst;	\
																		\
		kvecs_dst->values[kvecs_dst_id] = kvecs_src->values[kvecs_src_id]; \
		return true;													\
	}																	\
	STATIC_FUNCTION(int)												\
	xpu_##NAME##_datum_write(kern_context *kcxt,						\
							 char *buffer,								\
							 const kern_colmeta *cmeta,					\
							 const xpu_datum_t *__arg)					\
	{																	\
		xpu_##NAME##_t *arg = (xpu_##NAME##_t *)__arg;					\
																		\
		if (buffer)														\
			*((BASETYPE *)buffer) = arg->value;							\
		return sizeof(BASETYPE);										\
	}																	\
	STATIC_FUNCTION(bool)												\
	xpu_##NAME##_datum_hash(kern_context *kcxt,							\
							uint32_t *p_hash,							\
							xpu_datum_t *__arg)							\
	{																	\
		xpu_##NAME##_t *arg = (xpu_##NAME##_t *)__arg;					\
																		\
		if (XPU_DATUM_ISNULL(arg))										\
			*p_hash = 0;												\
		else															\
			*p_hash = pg_hash_any(&arg->value, sizeof(BASETYPE));		\
		return true;													\
	}																	\
	STATIC_FUNCTION(bool)												\
	xpu_##NAME##_datum_comp(kern_context *kcxt,							\
						int *p_comp,									\
						xpu_datum_t *__a,								\
						xpu_datum_t *__b)								\
	{																	\
		xpu_##NAME##_t *a = (xpu_##NAME##_t *)__a;						\
		xpu_##NAME##_t *b = (xpu_##NAME##_t *)__b;						\
																		\
		assert(!XPU_DATUM_ISNULL(a) && !XPU_DATUM_ISNULL(b));			\
		if (a->value > b->value)										\
			*p_comp = 1;												\
		else if (a->value < b->value)									\
			*p_comp = -1;												\
		else															\
			*p_comp = 0;												\
		return true;													\
	}																	\
	PGSTROM_SQLTYPE_OPERATORS(NAME,true,sizeof(BASETYPE),sizeof(BASETYPE))

#define PGSTROM_SIMPLE_INTEGER_TEMPLATE(NAME,BASETYPE)					\
	STATIC_FUNCTION(bool)												\
	xpu_##NAME##_datum_arrow_read(kern_context *kcxt,					\
								  const kern_data_store *kds,			\
								  const kern_colmeta *cmeta,			\
								  uint32_t index,						\
								  xpu_datum_t *__result)				\
	{																	\
		xpu_##NAME##_t *result = (xpu_##NAME##_t *)__result;			\
																		\
		if (cmeta->attopts.tag != ArrowType__Int ||						\
			cmeta->attopts.integer.bitWidth != sizeof(BASETYPE) * 8)	\
		{																\
			STROM_ELOG(kcxt, "xpu_" #NAME "_t must be mapped on Arrow::Int"); \
		}																\
		else															\
		{																\
			const BASETYPE *addr = (const BASETYPE *)					\
				KDS_ARROW_REF_SIMPLE_DATUM(kds, cmeta, index, sizeof(BASETYPE)); \
			if (!addr)													\
				result->expr_ops = NULL;								\
			else if (!cmeta->attopts.integer.is_signed && *addr < 0)	\
			{															\
				STROM_ELOG(kcxt, "Arrow::Int is out of range");			\
				return false;											\
			}															\
			else														\
			{															\
				result->expr_ops = &xpu_##NAME##_ops;					\
				result->value = *addr;									\
			}															\
			return true;												\
		}																\
		return false;													\
	}																	\
	__PGSTROM_SIMPLE_INT_FLOAT_TEMPLATE(NAME,BASETYPE)

#define PGSTROM_SIMPLE_FLOAT_TEMPLATE(NAME,BASETYPE,PREC)				\
	STATIC_FUNCTION(bool)												\
	xpu_##NAME##_datum_arrow_read(kern_context *kcxt,					\
								  const kern_data_store *kds,			\
								  const kern_colmeta *cmeta,			\
								  uint32_t index,						\
								  xpu_datum_t *__result)				\
	{																	\
		xpu_##NAME##_t *result = (xpu_##NAME##_t *)__result;			\
																		\
		if (cmeta->attopts.tag != ArrowType__FloatingPoint ||			\
			cmeta->attopts.floating_point.precision != ArrowPrecision__##PREC) \
		{																\
			STROM_ELOG(kcxt, "xpu_" #NAME "_t must be mapped on Arrow::FloatingPoint<" #PREC ">"); \
		}																\
		else															\
		{																\
			const BASETYPE *addr = (const BASETYPE *)					\
				KDS_ARROW_REF_SIMPLE_DATUM(kds, cmeta, index, sizeof(BASETYPE)); \
			if (!addr)													\
				result->expr_ops = NULL;								\
			else														\
			{															\
				result->expr_ops = &xpu_##NAME##_ops;					\
				result->value = *addr;									\
			}															\
			return true;												\
		}																\
		return false;													\
	}																	\
	__PGSTROM_SIMPLE_INT_FLOAT_TEMPLATE(NAME,BASETYPE)
	
PGSTROM_SIMPLE_INTEGER_TEMPLATE(int1,  int8_t);
PGSTROM_SIMPLE_INTEGER_TEMPLATE(int2, int16_t);
PGSTROM_SIMPLE_INTEGER_TEMPLATE(int4, int32_t);
PGSTROM_SIMPLE_INTEGER_TEMPLATE(int8, int64_t);
PGSTROM_SIMPLE_FLOAT_TEMPLATE(float2, float2_t, Half);
PGSTROM_SIMPLE_FLOAT_TEMPLATE(float4, float4_t, Single);
PGSTROM_SIMPLE_FLOAT_TEMPLATE(float8, float8_t, Double);

/* special support functions for float2 */
INLINE_FUNCTION(bool) isinf(float2_t fval) { return isinf(fp16_to_fp32(fval)); }
INLINE_FUNCTION(bool) isnan(float2_t fval) { return isnan(fp16_to_fp32(fval)); }
INLINE_FUNCTION(int)  lrinth(float2_t fval) { return rintf(fp16_to_fp32(fval)); }
INLINE_FUNCTION(bool) __iszero(float2_t fval) { return fp16_to_fp32(fval) == 0.0; }
INLINE_FUNCTION(bool) __iszero(float4_t fval) { return fval == 0.0; }
INLINE_FUNCTION(bool) __iszero(float8_t fval) { return fval == 0.0; }

/*
 * XPU Type Cast functions
 */
#define PG_SIMPLE_TYPECAST_TEMPLATE(TARGET,SOURCE,CAST,CHECKER)			\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##SOURCE##_to_##TARGET(XPU_PGFUNCTION_ARGS)					\
	{																	\
		KEXP_PROCESS_ARGS1(TARGET, SOURCE, datum);						\
																		\
		if (XPU_DATUM_ISNULL(&datum))									\
			result->expr_ops = NULL;									\
		else															\
		{																\
			if (!CHECKER(datum.value))									\
			{															\
				STROM_ELOG(kcxt, #SOURCE " to " #TARGET ": out of range"); \
				return false;											\
			}															\
			result->expr_ops = &xpu_##TARGET##_ops;						\
			result->value = CAST(datum.value);							\
		}																\
		return true;													\
	}
#define __TYPECAST_NOCHECK(X)		(true)
#define __INTEGER_TO_BOOL(X)		((X) != 0 ? true : false)
PG_SIMPLE_TYPECAST_TEMPLATE(bool,int4,__INTEGER_TO_BOOL,__TYPECAST_NOCHECK)

#define __INTEGER_FITS_IN_INT1(X)	((X) >= SCHAR_MIN && (X) <= SCHAR_MAX)
#define __FLOAT2_FITS_IN_INT1(X)	(!isnan(X) &&						\
									 lrinth(X) >= SCHAR_MIN &&			\
									 lrinth(X) <= SCHAR_MAX)
#define __FLOAT4_FITS_IN_INT1(X)	(!isnan(X) &&						\
									 rintf(X) >= (float)SCHAR_MIN &&	\
									 rintf(X) <= (float)SCHAR_MAX)
#define __FLOAT8_FITS_IN_INT1(X)	(!isnan(X) &&						\
									 rint(X) >= (double)SCHAR_MIN &&	\
									 rint(X) <= (double)SCHAR_MAX)
PG_SIMPLE_TYPECAST_TEMPLATE(int1,int2,(int8_t), __INTEGER_FITS_IN_INT1)
PG_SIMPLE_TYPECAST_TEMPLATE(int1,int4,(int8_t), __INTEGER_FITS_IN_INT1)
PG_SIMPLE_TYPECAST_TEMPLATE(int1,int8,(int8_t), __INTEGER_FITS_IN_INT1)
PG_SIMPLE_TYPECAST_TEMPLATE(int1,float2,lrinth, __FLOAT2_FITS_IN_INT1)
PG_SIMPLE_TYPECAST_TEMPLATE(int1,float4,rintf, __FLOAT4_FITS_IN_INT1)
PG_SIMPLE_TYPECAST_TEMPLATE(int1,float8,rint, __FLOAT8_FITS_IN_INT1)

#define __INTEGER_FITS_IN_INT2(X)	((X) >= SHRT_MIN && (X) <= SHRT_MAX)
#define __FLOAT2_FITS_IN_INT2(X)	(!isnan(X) &&						\
									 lrinth(X) >= SHRT_MIN &&			\
									 lrinth(X) <= SHRT_MAX)
#define __FLOAT4_FITS_IN_INT2(X)	(!isnan(X) &&						\
									 rintf(X) >= (float)SHRT_MIN &&		\
									 rintf(X) <= (float)SHRT_MAX)
#define __FLOAT8_FITS_IN_INT2(X)	(!isnan(X) &&						\
									 rint(X) >= (double)SHRT_MIN &&		\
									 rint(X) <= (double)SHRT_MAX)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,int1,(int16_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,int4,(int16_t),__INTEGER_FITS_IN_INT2)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,int8,(int16_t),__INTEGER_FITS_IN_INT2)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,float2,lrinth, __FLOAT2_FITS_IN_INT2)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,float4,rintf, __FLOAT4_FITS_IN_INT2)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,float8,rint, __FLOAT8_FITS_IN_INT2)

#define __INTEGER_FITS_IN_INT4(X)	((X) >= INT_MIN && (X) <= INT_MAX)
#define __FLOAT4_FITS_IN_INT4(X)	(!isnan(X) &&						\
									 rintf(X) >= (float)INT_MIN &&		\
									 rintf(X) <= (float)INT_MAX)
#define __FLOAT8_FITS_IN_INT4(X)	(!isnan(X) &&						\
									 rint(X) >= (double)INT_MIN &&			\
									 rint(X) <= (double)INT_MAX)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,int1,(int32_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,int2,(int32_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,int8,(int32_t),__INTEGER_FITS_IN_INT4)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,float2,lrinth, __TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,float4,rintf, __FLOAT4_FITS_IN_INT4)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,float8,rint, __FLOAT8_FITS_IN_INT4)

#define __FLOAT4_FITS_IN_INT8(X)	(!isnan((float)(X)) &&				\
									 rintf(X) >= (float)LLONG_MIN &&	\
									 rintf(X) <= (float)LLONG_MAX)
#define __FLOAT8_FITS_IN_INT8(X)	(!isnan((double)(X)) &&				\
									 rint(X) >= (double)LLONG_MIN &&	\
									 rint(X) <= (double)LLONG_MAX)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,int1,(int64_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,int2,(int64_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,int4,(int64_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,float2,lrinth, __TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,float4,rintf, __FLOAT4_FITS_IN_INT8)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,float8,rint, __FLOAT8_FITS_IN_INT8)

#define __INTEGER_FITS_IN_FLOAT2(X)	((X) >= -65504 && (X) <= 65504)
#define __FLOAT_FITS_IN_FLOAT2(X)	(!isinf(X) && !isnan(X) &&	\
									 (X) >= -65504.0 &&			\
									 (X) <=  65504.0)
PG_SIMPLE_TYPECAST_TEMPLATE(float2,  int1,fp32_to_fp16,__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float2,  int2,fp32_to_fp16,__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float2,  int4,fp64_to_fp16,__INTEGER_FITS_IN_FLOAT2)
PG_SIMPLE_TYPECAST_TEMPLATE(float2,  int8,fp64_to_fp16,__INTEGER_FITS_IN_FLOAT2)
PG_SIMPLE_TYPECAST_TEMPLATE(float2,float4,fp32_to_fp16,__FLOAT_FITS_IN_FLOAT2)
PG_SIMPLE_TYPECAST_TEMPLATE(float2,float8,fp64_to_fp16,__FLOAT_FITS_IN_FLOAT2)

#define __FLOAT_FITS_IN_FLOAT4(X)	(!isinf(X) && !isnan(X) &&			\
									 (X) >= -FLT_MAX &&					\
									 (X) <=  FLT_MAX)
PG_SIMPLE_TYPECAST_TEMPLATE(float4,  int1,(float), __TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float4,  int2,(float), __TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float4,  int4,(float), __TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float4,  int8,(float), __TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float4,float2,fp16_to_fp32, __TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float4,float8,(float), __FLOAT_FITS_IN_FLOAT4)

PG_SIMPLE_TYPECAST_TEMPLATE(float8,  int1,(double), __TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float8,  int2,(double), __TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float8,  int4,(double), __TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float8,  int8,(double), __TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float8,float2,fp16_to_fp64, __TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float8,float4,(double), __TYPECAST_NOCHECK)
#undef PG_SIMPLE_TYPECAST_TEMPLATE

__PG_SIMPLE_COMPARE_TEMPLATE(bool,bool,bool,,==,eq)

PG_SIMPLE_COMPARE_TEMPLATE(int1,  int1, int1, )
PG_SIMPLE_COMPARE_TEMPLATE(int12, int1, int2, (int16_t))
PG_SIMPLE_COMPARE_TEMPLATE(int14, int1, int4, (int32_t))
PG_SIMPLE_COMPARE_TEMPLATE(int18, int1, int8, (int64_t))

PG_SIMPLE_COMPARE_TEMPLATE(int21, int2, int1, (int16_t))
PG_SIMPLE_COMPARE_TEMPLATE(int2,  int2, int2, )
PG_SIMPLE_COMPARE_TEMPLATE(int24, int2, int4, (int32_t))
PG_SIMPLE_COMPARE_TEMPLATE(int28, int2, int8, (int64_t))

PG_SIMPLE_COMPARE_TEMPLATE(int41, int4, int1, (int32_t))
PG_SIMPLE_COMPARE_TEMPLATE(int42, int4, int2, (int32_t))
PG_SIMPLE_COMPARE_TEMPLATE(int4,  int4, int4, )
PG_SIMPLE_COMPARE_TEMPLATE(int48, int4, int8, (int64_t))

PG_SIMPLE_COMPARE_TEMPLATE(int81, int8, int1, (int64_t))
PG_SIMPLE_COMPARE_TEMPLATE(int82, int8, int2, (int64_t))
PG_SIMPLE_COMPARE_TEMPLATE(int84, int8, int4, (int64_t))
PG_SIMPLE_COMPARE_TEMPLATE(int8,  int8, int8, )

#define __float_comp_eq(x,y,CAST)						\
	(isnan(x) ? isnan(y) : !isnan(y) && (CAST (x) == CAST (y)))
#define __float_comp_ne(x,y,CAST)						\
	(isnan(x) ? !isnan(y) : isnan(y) || (CAST (x) != CAST (y)))
#define __float_comp_lt(x,y,CAST)						\
	(!isnan(x) && (isnan(y) || (CAST (x) <  CAST (y))))
#define __float_comp_le(x,y,CAST)						\
	(isnan(y) || (!isnan(x) && (CAST (x) <= CAST (y))))
#define __float_comp_gt(x,y,CAST)						\
	(!isnan(y) && (isnan(x) || (CAST (x) >  CAST (y))))
#define __float_comp_ge(x,y,CAST)						\
	(isnan(x) || (!isnan(y) && (CAST (x) >= CAST (y))))

#define __PG_FLOAT_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,EXTRA)	\
	PUBLIC_FUNCTION(bool)											\
	pgfn_##FNAME##EXTRA(XPU_PGFUNCTION_ARGS)						\
	{																\
		KEXP_PROCESS_ARGS2(bool,LNAME,lval,RNAME,rval);				\
		if (XPU_DATUM_ISNULL(&lval) || XPU_DATUM_ISNULL(&rval))		\
		{															\
			__pg_simple_nullcomp_eq(&lval, &rval);					\
		}															\
		else														\
		{															\
			result->expr_ops = kexp->expr_ops;						\
			result->value = __float_comp_##EXTRA(lval.value,		\
												 rval.value, CAST);	\
		}															\
		return true;												\
	}
#define PG_FLOAT_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST)	\
	__PG_FLOAT_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,eq)	\
	__PG_FLOAT_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,ne)	\
	__PG_FLOAT_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,lt)	\
	__PG_FLOAT_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,le)	\
	__PG_FLOAT_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,gt)	\
	__PG_FLOAT_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,ge)	\

PG_FLOAT_COMPARE_TEMPLATE(float2,  float2, float2, )
PG_FLOAT_COMPARE_TEMPLATE(float24, float2, float4, (float4_t))
PG_FLOAT_COMPARE_TEMPLATE(float28, float2, float8, (float8_t))
PG_FLOAT_COMPARE_TEMPLATE(float42, float4, float2, (float4_t))
PG_FLOAT_COMPARE_TEMPLATE(float4,  float4, float4, )
PG_FLOAT_COMPARE_TEMPLATE(float48, float4, float8, (float8_t))
PG_FLOAT_COMPARE_TEMPLATE(float82, float8, float2, (float8_t))
PG_FLOAT_COMPARE_TEMPLATE(float84, float8, float4, (float8_t))
PG_FLOAT_COMPARE_TEMPLATE(float8,  float8, float8, )

#undef __float_comp_eq
#undef __float_comp_ne
#undef __float_comp_lt
#undef __float_comp_le
#undef __float_comp_gt
#undef __float_comp_ge

/*
 * Binary operator template
 */
#define PG_INT_BIN_OPERATOR_TEMPLATE(FNAME,RTYPE,XTYPE,YTYPE,OPER,		\
									 __TEMP,__MIN,__MAX)				\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##FNAME(XPU_PGFUNCTION_ARGS)									\
	{																	\
		KEXP_PROCESS_ARGS2(RTYPE, XTYPE, x_val, YTYPE, y_val);			\
																		\
		if (XPU_DATUM_ISNULL(&x_val) || XPU_DATUM_ISNULL(&y_val))		\
			result->expr_ops = NULL;									\
		else															\
		{																\
			__TEMP r = (__TEMP)x_val.value OPER (__TEMP)y_val.value;	\
																		\
			if (r < (__TEMP)__MIN || r > (__TEMP)__MAX)					\
			{															\
				STROM_ELOG(kcxt, #FNAME ": value out of range");		\
				return false;											\
			}															\
			result->expr_ops = &xpu_##RTYPE##_ops;						\
			result->value = r;											\
		}																\
		return true;													\
	}

#define PG_FLOAT_BIN_OPERATOR_TEMPLATE(FNAME,RTYPE,XTYPE,YTYPE,OPER,__CAST)	\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##FNAME(XPU_PGFUNCTION_ARGS)									\
	{																	\
		KEXP_PROCESS_ARGS2(RTYPE, XTYPE, x_val, YTYPE, y_val);			\
																		\
		if (XPU_DATUM_ISNULL(&x_val) || XPU_DATUM_ISNULL(&y_val))		\
			result->expr_ops = NULL;									\
		else															\
		{																\
			result->expr_ops = &xpu_##RTYPE##_ops;						\
			result->value = (__CAST(x_val.value) OPER					\
							 __CAST(y_val.value));						\
			if (isinf(result->value) &&									\
				!isinf(x_val.value) &&									\
				!isinf(y_val.value))									\
			{															\
				STROM_ELOG(kcxt, #FNAME ": value out of range");		\
				return false;											\
			}															\
		}																\
		return true;													\
	}

/* '+' : add operators */
PG_INT_BIN_OPERATOR_TEMPLATE(int1pl, int1,int1,int1,+, int16_t,SCHAR_MIN,SCHAR_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int12pl,int2,int1,int2,+, int32_t,SHRT_MIN,SHRT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int14pl,int4,int1,int4,+, int64_t,INT_MIN,INT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int18pl,int8,int1,int8,+,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int21pl,int2,int2,int1,+, int32_t,SHRT_MIN,SHRT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int2pl, int2,int2,int2,+, int32_t,SHRT_MIN,SHRT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int24pl,int4,int2,int4,+, int64_t,INT_MIN,INT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int28pl,int8,int2,int8,+,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int41pl,int4,int4,int1,+, int64_t,INT_MIN,INT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int42pl,int4,int4,int2,+, int64_t,INT_MIN,INT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int4pl, int4,int4,int4,+, int64_t,INT_MIN,INT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int48pl,int8,int4,int8,+,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int81pl,int8,int8,int1,+,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int82pl,int8,int8,int2,+,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int84pl,int8,int8,int4,+,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int8pl, int8,int8,int8,+,int128_t,LLONG_MIN,LLONG_MAX)

PG_FLOAT_BIN_OPERATOR_TEMPLATE(float2pl, float4,float2,float2,+,__to_fp32)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float24pl,float4,float2,float4,+,__to_fp32)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float28pl,float8,float2,float8,+,__to_fp64)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float42pl,float4,float4,float2,+,__to_fp32)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float4pl, float4,float4,float4,+,)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float48pl,float8,float4,float8,+,__to_fp64)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float82pl,float8,float8,float2,+,__to_fp64)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float84pl,float8,float8,float4,+,__to_fp64)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float8pl, float8,float8,float8,+,)

/* '-' : subtract operators */
PG_INT_BIN_OPERATOR_TEMPLATE(int1mi, int1,int1,int1,-, int16_t,SCHAR_MIN,SCHAR_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int12mi,int2,int1,int2,-, int32_t,SHRT_MIN,SHRT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int14mi,int4,int1,int4,-, int64_t,INT_MIN,INT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int18mi,int8,int1,int8,-,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int21mi,int2,int2,int1,-, int32_t,SHRT_MIN,SHRT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int2mi, int2,int2,int2,-, int32_t,SHRT_MIN,SHRT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int24mi,int4,int2,int4,-, int64_t,INT_MIN,INT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int28mi,int8,int2,int8,-,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int41mi,int4,int4,int1,-, int64_t,INT_MIN,INT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int42mi,int4,int4,int2,-, int64_t,INT_MIN,INT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int4mi, int4,int4,int4,-, int64_t,INT_MIN,INT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int48mi,int8,int4,int8,-,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int81mi,int8,int8,int1,-,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int82mi,int8,int8,int2,-,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int84mi,int8,int8,int4,-,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int8mi, int8,int8,int8,-,int128_t,LLONG_MIN,LLONG_MAX)

PG_FLOAT_BIN_OPERATOR_TEMPLATE(float2mi, float4,float2,float2,-,__to_fp32)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float24mi,float4,float2,float4,-,__to_fp32)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float28mi,float8,float2,float8,-,__to_fp64)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float42mi,float4,float4,float2,-,__to_fp32)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float4mi, float4,float4,float4,-,)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float48mi,float8,float4,float8,-,__to_fp64)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float82mi,float8,float8,float2,-,__to_fp64)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float84mi,float8,float8,float4,-,__to_fp64)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float8mi, float8,float8,float8,-,)

/* '*' : multiply operators */
PG_INT_BIN_OPERATOR_TEMPLATE(int1mul, int1,int1,int1,*, int16_t,SCHAR_MIN,SCHAR_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int12mul,int2,int1,int2,*, int32_t,SHRT_MIN,SHRT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int14mul,int4,int1,int4,*, int64_t,INT_MIN,INT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int18mul,int8,int1,int8,*,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int21mul,int2,int2,int1,*, int32_t,SHRT_MIN,SHRT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int2mul, int2,int2,int2,*, int32_t,SHRT_MIN,SHRT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int24mul,int4,int2,int4,*, int64_t,INT_MIN,INT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int28mul,int8,int2,int8,*,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int41mul,int4,int4,int1,*, int64_t,INT_MIN,INT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int42mul,int4,int4,int2,*, int64_t,INT_MIN,INT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int4mul, int4,int4,int4,*, int64_t,INT_MIN,INT_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int48mul,int8,int4,int8,*,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int81mul,int8,int8,int1,*,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int82mul,int8,int8,int2,*,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int84mul,int8,int8,int4,*,int128_t,LLONG_MIN,LLONG_MAX)
PG_INT_BIN_OPERATOR_TEMPLATE(int8mul, int8,int8,int8,*,int128_t,LLONG_MIN,LLONG_MAX)

PG_FLOAT_BIN_OPERATOR_TEMPLATE(float2mul, float4,float2,float2,*,__to_fp32)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float24mul,float4,float2,float4,*,__to_fp32)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float28mul,float8,float2,float8,*,__to_fp64)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float42mul,float4,float4,float2,*,__to_fp32)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float4mul, float4,float4,float4,*,)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float48mul,float8,float4,float8,*,__to_fp64)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float82mul,float8,float8,float2,*,__to_fp64)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float84mul,float8,float8,float4,*,__to_fp64)
PG_FLOAT_BIN_OPERATOR_TEMPLATE(float8mul, float8,float8,float8,*,)

/* '/' : divide operators */
#define PG_INT_DIV_OPERATOR_TEMPLATE(FNAME,RTYPE,XTYPE,YTYPE,__MIN)		\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##FNAME(XPU_PGFUNCTION_ARGS)									\
	{																	\
		KEXP_PROCESS_ARGS2(RTYPE, XTYPE, x_val, YTYPE, y_val);			\
																		\
		if (XPU_DATUM_ISNULL(&x_val) || XPU_DATUM_ISNULL(&y_val))		\
			result->expr_ops = NULL;									\
		else															\
		{																\
			if (y_val.value == 0)										\
			{															\
				STROM_ELOG(kcxt, #FNAME ": division by zero");			\
				return false;											\
			}															\
			/* if overflow may happen, __MIN should be set */			\
			if (__MIN != 0 && y_val.value == -1)						\
			{															\
				if (x_val.value == __MIN)								\
				{														\
					STROM_ELOG(kcxt, #FNAME ": value out of range");	\
					return false;										\
				}														\
				result->value = -x_val.value;							\
			}															\
			else														\
			{															\
				result->value = x_val.value / y_val.value;				\
			}															\
			result->expr_ops = &xpu_##RTYPE##_ops;						\
		}																\
		return true;													\
	}

#define PG_FLOAT_DIV_OPERATOR_TEMPLATE(FNAME,RTYPE,XTYPE,YTYPE,__CAST)	\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##FNAME(XPU_PGFUNCTION_ARGS)                                   \
    {                                                                   \
		KEXP_PROCESS_ARGS2(RTYPE, XTYPE, x_val, YTYPE, y_val);			\
																		\
		if (XPU_DATUM_ISNULL(&x_val) || XPU_DATUM_ISNULL(&y_val))		\
			result->expr_ops = NULL;									\
		else															\
		{																\
			if (__iszero(y_val.value))									\
			{															\
				STROM_ELOG(kcxt, #FNAME ": division by zero");			\
				return false;											\
			}															\
			result->value = __CAST(x_val.value) / __CAST(y_val.value);	\
			/* CHECKFLOATVAL */											\
			if ((isinf(result->value) && (!isinf(x_val.value) &&		\
										  !isinf(y_val.value))) ||		\
				(__iszero(result->value) && !__iszero(x_val.value)))	\
			{															\
				STROM_ELOG(kcxt, #FNAME ": value out of range");		\
				return false;											\
			}															\
			result->expr_ops = &xpu_##RTYPE##_ops;						\
		}																\
		return true;													\
	}
PG_INT_DIV_OPERATOR_TEMPLATE(int1div, int1,int1,int1,SCHAR_MIN)
PG_INT_DIV_OPERATOR_TEMPLATE(int12div,int2,int1,int2,0)
PG_INT_DIV_OPERATOR_TEMPLATE(int14div,int4,int1,int4,0)
PG_INT_DIV_OPERATOR_TEMPLATE(int18div,int8,int1,int8,0)
PG_INT_DIV_OPERATOR_TEMPLATE(int21div,int2,int2,int1,SHRT_MIN)
PG_INT_DIV_OPERATOR_TEMPLATE(int2div, int2,int2,int2,SHRT_MIN)
PG_INT_DIV_OPERATOR_TEMPLATE(int24div,int4,int2,int4,0)
PG_INT_DIV_OPERATOR_TEMPLATE(int28div,int8,int2,int8,0)
PG_INT_DIV_OPERATOR_TEMPLATE(int41div,int4,int4,int1,INT_MIN)
PG_INT_DIV_OPERATOR_TEMPLATE(int42div,int4,int4,int2,INT_MIN)
PG_INT_DIV_OPERATOR_TEMPLATE(int4div, int4,int4,int4,INT_MIN)
PG_INT_DIV_OPERATOR_TEMPLATE(int48div,int8,int4,int8,0)
PG_INT_DIV_OPERATOR_TEMPLATE(int81div,int8,int8,int1,LLONG_MIN)
PG_INT_DIV_OPERATOR_TEMPLATE(int82div,int8,int8,int2,LLONG_MIN)
PG_INT_DIV_OPERATOR_TEMPLATE(int84div,int8,int8,int4,LLONG_MIN)
PG_INT_DIV_OPERATOR_TEMPLATE(int8div, int8,int8,int8,LLONG_MIN)
PG_FLOAT_DIV_OPERATOR_TEMPLATE(float2div, float4,float2,float2,)
PG_FLOAT_DIV_OPERATOR_TEMPLATE(float24div,float4,float2,float4,__to_fp32)
PG_FLOAT_DIV_OPERATOR_TEMPLATE(float28div,float8,float2,float8,__to_fp64)
PG_FLOAT_DIV_OPERATOR_TEMPLATE(float42div,float4,float4,float2,__to_fp32)
PG_FLOAT_DIV_OPERATOR_TEMPLATE(float4div, float4,float4,float4,)
PG_FLOAT_DIV_OPERATOR_TEMPLATE(float48div,float8,float4,float8,__to_fp64)
PG_FLOAT_DIV_OPERATOR_TEMPLATE(float82div,float8,float8,float2,__to_fp64)
PG_FLOAT_DIV_OPERATOR_TEMPLATE(float84div,float8,float8,float4,__to_fp64)
PG_FLOAT_DIV_OPERATOR_TEMPLATE(float8div, float8,float8,float8,)

/*
 * Modulo operator: '%'
 */
#define PG_INT_MOD_OPERATOR_TEMPLATE(TYPE)								\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##TYPE##mod(XPU_PGFUNCTION_ARGS)								\
	{																	\
		KEXP_PROCESS_ARGS2(TYPE, TYPE, x_val, TYPE, y_val);				\
																		\
		if (XPU_DATUM_ISNULL(&x_val) || XPU_DATUM_ISNULL(&y_val))		\
			result->expr_ops = NULL;									\
		else															\
		{																\
			if (y_val.value == 0)										\
			{															\
				STROM_ELOG(kcxt, #TYPE "mod : division by zero");		\
				return false;											\
			}															\
			result->expr_ops = &xpu_##TYPE##_ops;                       \
			result->value = x_val.value % y_val.value;                  \
		}																\
		return true;													\
	}
PG_INT_MOD_OPERATOR_TEMPLATE(int1)
PG_INT_MOD_OPERATOR_TEMPLATE(int2)
PG_INT_MOD_OPERATOR_TEMPLATE(int4)
PG_INT_MOD_OPERATOR_TEMPLATE(int8)

/*
 * Bit operators: '&', '|', '#', '~', '>>', and '<<'
 */
#define PG_UNARY_OPERATOR_TEMPLATE(FNAME,XTYPE,OPER,CHECKER)			\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##FNAME(XPU_PGFUNCTION_ARGS)									\
	{																	\
		KEXP_PROCESS_ARGS1(XTYPE,XTYPE,x_val);							\
																		\
		if (XPU_DATUM_ISNULL(&x_val))									\
			result->expr_ops = NULL;									\
		else if (CHECKER)												\
		{																\
			result->expr_ops = &xpu_##XTYPE##_ops;						\
			result->value = OPER(x_val.value);							\
		}																\
		else															\
		{																\
			STROM_ELOG(kcxt, #FNAME ": value out of range");			\
			return false;												\
		}																\
		return true;													\
	}

#define PG_BITWISE_OPERATOR_TEMPLATE(FNAME,RTYPE,XTYPE,YTYPE,OPER)		\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##FNAME(XPU_PGFUNCTION_ARGS)									\
	{																	\
		KEXP_PROCESS_ARGS2(RTYPE,XTYPE,x_val,YTYPE,y_val);				\
																		\
		if (XPU_DATUM_ISNULL(&x_val) || XPU_DATUM_ISNULL(&y_val))		\
			result->expr_ops = NULL;									\
		else															\
		{																\
			result->expr_ops = &xpu_##RTYPE##_ops;						\
			result->value = x_val.value OPER y_val.value;				\
		}																\
		return true;													\
	}

PG_UNARY_OPERATOR_TEMPLATE(int1up,int1,,true)
PG_UNARY_OPERATOR_TEMPLATE(int2up,int2,,true)
PG_UNARY_OPERATOR_TEMPLATE(int4up,int4,,true)
PG_UNARY_OPERATOR_TEMPLATE(int8up,int8,,true)
PG_UNARY_OPERATOR_TEMPLATE(float2up,float2,,true)
PG_UNARY_OPERATOR_TEMPLATE(float4up,float4,,true)
PG_UNARY_OPERATOR_TEMPLATE(float8up,float8,,true)

PG_UNARY_OPERATOR_TEMPLATE(int1um,int1,-,x_val.value!=SCHAR_MIN)
PG_UNARY_OPERATOR_TEMPLATE(int2um,int2,-,x_val.value!=SHRT_MIN)
PG_UNARY_OPERATOR_TEMPLATE(int4um,int4,-,x_val.value!=INT_MIN)
PG_UNARY_OPERATOR_TEMPLATE(int8um,int8,-,x_val.value!=LONG_MIN)
PG_UNARY_OPERATOR_TEMPLATE(float2um,float2,__fp16_unary_minus,true)
PG_UNARY_OPERATOR_TEMPLATE(float4um,float4,-,true)
PG_UNARY_OPERATOR_TEMPLATE(float8um,float8,-,true)

PG_UNARY_OPERATOR_TEMPLATE(int1abs,int1,abs,x_val.value!=SCHAR_MIN)
PG_UNARY_OPERATOR_TEMPLATE(int2abs,int2,abs,x_val.value!=SHRT_MIN)
PG_UNARY_OPERATOR_TEMPLATE(int4abs,int4,abs,x_val.value!=INT_MIN)
PG_UNARY_OPERATOR_TEMPLATE(int8abs,int8,llabs,x_val.value!=LONG_MIN)
PG_UNARY_OPERATOR_TEMPLATE(float2abs,float2,__fp16_unary_abs,true)
PG_UNARY_OPERATOR_TEMPLATE(float4abs,float4,fabsf,true)
PG_UNARY_OPERATOR_TEMPLATE(float8abs,float8,fabs,true)

PG_BITWISE_OPERATOR_TEMPLATE(int1and,int1,int1,int1,&)
PG_BITWISE_OPERATOR_TEMPLATE(int2and,int2,int2,int2,&)
PG_BITWISE_OPERATOR_TEMPLATE(int4and,int4,int4,int4,&)
PG_BITWISE_OPERATOR_TEMPLATE(int8and,int8,int8,int8,&)

PG_BITWISE_OPERATOR_TEMPLATE(int1or,int1,int1,int1,|)
PG_BITWISE_OPERATOR_TEMPLATE(int2or,int2,int2,int2,|)
PG_BITWISE_OPERATOR_TEMPLATE(int4or,int4,int4,int4,|)
PG_BITWISE_OPERATOR_TEMPLATE(int8or,int8,int8,int8,|)

PG_BITWISE_OPERATOR_TEMPLATE(int1xor,int1,int1,int1,^)
PG_BITWISE_OPERATOR_TEMPLATE(int2xor,int2,int2,int2,^)
PG_BITWISE_OPERATOR_TEMPLATE(int4xor,int4,int4,int4,^)
PG_BITWISE_OPERATOR_TEMPLATE(int8xor,int8,int8,int8,^)

PG_UNARY_OPERATOR_TEMPLATE(int1not,int1,~,true)
PG_UNARY_OPERATOR_TEMPLATE(int2not,int2,~,true)
PG_UNARY_OPERATOR_TEMPLATE(int4not,int4,~,true)
PG_UNARY_OPERATOR_TEMPLATE(int8not,int8,~,true)

PG_BITWISE_OPERATOR_TEMPLATE(int1shr,int1,int1,int4,>>)
PG_BITWISE_OPERATOR_TEMPLATE(int2shr,int2,int2,int4,>>)
PG_BITWISE_OPERATOR_TEMPLATE(int4shr,int4,int4,int4,>>)
PG_BITWISE_OPERATOR_TEMPLATE(int8shr,int8,int8,int4,>>)

PG_BITWISE_OPERATOR_TEMPLATE(int1shl,int1,int1,int4,<<)
PG_BITWISE_OPERATOR_TEMPLATE(int2shl,int2,int2,int4,<<)
PG_BITWISE_OPERATOR_TEMPLATE(int4shl,int4,int4,int4,<<)
PG_BITWISE_OPERATOR_TEMPLATE(int8shl,int8,int8,int4,<<)

/*
 * Device only type cast functions instead of CoerceViaIO
 */
#define PG_DEVCAST_TEXT_TO_INT_TEMPLATE(XTYPE,BTYPE,__MIN,__MAX)		\
	PUBLIC_FUNCTION(bool)												\
	pgfn_devcast_text_to_##XTYPE(XPU_PGFUNCTION_ARGS)					\
	{																	\
		KEXP_PROCESS_ARGS1(XTYPE,text,arg);								\
																		\
		if (XPU_DATUM_ISNULL(&arg))										\
			result->expr_ops = NULL;									\
		else if (!xpu_text_is_valid(kcxt, &arg))						\
			return false;												\
		else															\
		{																\
			const char *str = arg.value;								\
			int			len = arg.length;								\
			BTYPE		ival = 0;										\
			bool		negative = false;								\
																		\
			if (*str == '-')											\
			{															\
				str++;													\
				len--;													\
				negative = true;										\
			}															\
			while (len-- > 0)											\
			{															\
				int		c = *str++;										\
																		\
				if (c < '0' || c > '9')									\
				{														\
					STROM_ELOG(kcxt, "invalid input for " #XTYPE);		\
					return false;										\
				}														\
				if (ival > (__MAX/10) ||								\
					(ival == (__MAX/10) && c > (negative ? '8' : '7')))	\
				{														\
					STROM_ELOG(kcxt, #XTYPE ": out of range");			\
					return false;										\
				}														\
				ival = 10 * ival + (c - '0');							\
			}															\
			if (negative)												\
				ival = -ival;											\
			assert(ival >= __MIN && ival <= __MAX);						\
			result->expr_ops = &xpu_##XTYPE##_ops;						\
			result->value = ival;										\
		}																\
		return true;													\
	 }
PG_DEVCAST_TEXT_TO_INT_TEMPLATE(int1,int32_t,SCHAR_MIN,SCHAR_MAX)
PG_DEVCAST_TEXT_TO_INT_TEMPLATE(int2,int32_t,SHRT_MIN,SHRT_MAX)
PG_DEVCAST_TEXT_TO_INT_TEMPLATE(int4,int32_t,INT_MIN,INT_MAX)
PG_DEVCAST_TEXT_TO_INT_TEMPLATE(int8,int64_t,LONG_MIN,LONG_MAX)

STATIC_FUNCTION(bool)
__strtof(const char *str, int len, double *p_fval)
{
	const char *pos;
	double		fval = 0.0;
	double		divisor = 1.0;
	char		sign = '+';
	bool		meet_period = false;

	/* skip leading whitespace */
	while (len > 0 && __isspace(*str))
	{
		str++;
		len--;
	}
	if (len == 0)
		return false;
	pos = str;
	/* sign (optional) */
	if (*pos == '+' || *pos == '-')
	{
		sign = *pos;
		pos++;
		len--;
	}
	/* Decimal */
	while (len > 0)
	{
		int		c = *pos;

		if (c == '.' && !meet_period)
			meet_period = true;
		else if (__isdigit(c))
		{
			fval = 10.0 * fval + (double)(c - '0');
			if (meet_period)
				divisor *= 10.0;
		}
		else
			break;
		pos++;
		len--;
	}
	/* Exponential */
	if (len > 0 && (*pos == 'e' || *pos == 'E'))
	{
		int		expo = 0;
		char	expo_sign = '+';

		pos++;
		len--;
		if (len == 0)
			goto out;
		if (*pos == '+' || *pos == '-')
		{
			expo_sign = *pos;
			pos++;
			len--;
		}
		while (len > 0)
		{
			int		c = *pos;

			if (!__isdigit(c))
				break;
			expo = 10 * expo + (c - '0');
			pos++;
			len--;
		}
		if (expo_sign == '+')
			divisor *= pow(1.0, expo);
		else
			fval *= pow(1.0, expo);
	}
	/* skip tailing whitespace */
	while (len > 0 && __isspace(*pos))
	{
		pos++;
		len--;
	}
	/* len == 0 if valid digits */
	if (len == 0)
	{
		fval /= divisor;
		if (sign == '-')
			fval = -fval;
		*p_fval = fval;
		return true;
	}
out:
	assert(pos >= str);
	len += (pos - str);
	if (len >= 3 && __strncasecmp(str, "NaN", 3) == 0)
	{
		fval = DBL_NAN;
		str += 3;
		len -= 3;
	}
	else if (len >= 8 && __strncasecmp(str, "Infinity", 8) == 0)
	{
		fval = DBL_INF;
		str += 8;
		len -= 8;
	}
	else if (len >= 9 && __strncasecmp(str, "+Infinity", 9) == 0)
	{
		fval = DBL_INF;
		str += 9;
		len -= 9;
	}
	else if (len >= 9 && __strncasecmp(str, "-Infinity", 9) == 0)
	{
		fval = -DBL_INF;
		str += 9;
		len -= 9;
	}
	else if (len >= 3 && __strncasecmp(str, "inf", 3) == 0)
	{
		fval = DBL_INF;
		str += 3;
		len -= 3;
	}
	else if (len >= 4 && __strncasecmp(str, "+inf", 4) == 0)
	{
		fval = DBL_INF;
		str += 4;
		len -= 4;
	}
	else if (len >= 4 && __strncasecmp(str, "-inf", 4) == 0)
	{
		fval = -DBL_INF;
		str += 4;
		len -= 4;
	}
    /* skip tailing whitespace */
    while (len > 0 && __isspace(*str))
    {
        str++;
        len--;
    }
	if (len == 0)
	{
		*p_fval = fval;
		return true;
	}
	return false;
}

PUBLIC_FUNCTION(bool)
pgfn_devcast_text_to_float2(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float2,text,arg);
	double		fval;

	if (XPU_DATUM_ISNULL(&arg))
		result->expr_ops = NULL;
	else if (!xpu_text_is_valid(kcxt, &arg))
		return false;
	else if (__strtof(arg.value, arg.length, &fval))
	{
		if (isfinite(fval) && (fval < -(double)HALF_MAX ||
							   fval >  (double)HALF_MAX))
		{
			STROM_ELOG(kcxt, "value is out of range for float2");
			return false;
		}
		result->expr_ops = &xpu_float2_ops;
		result->value = fval;
	}
	else
	{
		STROM_ELOG(kcxt, "invalid input for float2");
		return false;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_devcast_text_to_float4(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float4,text,arg);
	double		fval;

	if (XPU_DATUM_ISNULL(&arg))
		result->expr_ops = NULL;
	else if (!xpu_text_is_valid(kcxt, &arg))
		return false;
	else if (__strtof(arg.value, arg.length, &fval))
	{
		if (isfinite(fval) && (fval < -(double)FLT_MAX ||
							   fval >  (double)FLT_MAX))
		{
			STROM_ELOG(kcxt, "value is out of range for float4");
			return false;
		}
		result->expr_ops = &xpu_float4_ops;
		result->value = fval;
	}
	else
	{
		STROM_ELOG(kcxt, "invalid input for float4");
		return false;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_devcast_text_to_float8(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8,text,arg);
	double		fval;

	if (XPU_DATUM_ISNULL(&arg))
		result->expr_ops = NULL;
	else if (!xpu_text_is_valid(kcxt, &arg))
		return false;
	else if (__strtof(arg.value, arg.length, &fval))
	{
		result->expr_ops = &xpu_float8_ops;
		result->value = fval;
	}
	else
	{
		STROM_ELOG(kcxt, "invalid input for float8");
		return false;
	}
	return true;
}
