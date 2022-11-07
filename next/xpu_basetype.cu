/*
 * xpu_basetype.cc
 *
 * Collection of primitive Int/Float type support for both of GPU and DPU
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
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
xpu_bool_datum_ref(kern_context *kcxt,
				   xpu_datum_t *__result,
				   const void *addr)
{
	xpu_bool_t *result = (xpu_bool_t *)__result;

	if (!addr)
		result->isnull = true;
	else
	{
		result->isnull = false;
		result->value = *((const bool *)addr);
	}
	return true;
}
STATIC_FUNCTION(bool)
xpu_bool_arrow_ref(kern_context *kcxt,
				   xpu_datum_t *result,
				   const kern_colmeta *cmeta,
				   const void *addr, int len)
{
	STROM_ELOG(kcxt, "Bug? Arrow::Bool shall not be referenced by generic handler");
	return false;
}
STATIC_FUNCTION(int)
xpu_bool_arrow_move(kern_context *kcxt,
					char *buffer,
					const kern_colmeta *cmeta,
					const void *addr, int len)
{
	STROM_ELOG(kcxt, "Bug? Arrow::Bool shall not be moved by generic handler");
	return false;
}
STATIC_FUNCTION(int)
xpu_bool_datum_store(kern_context *kcxt,
					 char *buffer,
					 const xpu_datum_t *__arg)
{
	const xpu_bool_t *arg = (const xpu_bool_t *)__arg;

	if (arg->isnull)
		return 0;
	if (buffer)
		*((bool *)buffer) = arg->value;
	return sizeof(bool);
}
STATIC_FUNCTION(bool)
xpu_bool_datum_hash(kern_context *kcxt,
					uint32_t *p_hash,
					const xpu_datum_t *__arg)
{
	const xpu_bool_t *arg = (const xpu_bool_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(bool));
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(bool,true,1,sizeof(bool));

/*
 * Int1/Int2/Int4/Int8 handlers
 */
#define PGSTROM_SIMPLE_INTEGER_TEMPLATE(NAME,BASETYPE)					\
	STATIC_FUNCTION(bool)												\
	xpu_##NAME##_datum_ref(kern_context *kcxt,							\
						   xpu_datum_t *__result,						\
						   const void *addr)							\
	{																	\
		xpu_##NAME##_t *result = (xpu_##NAME##_t *)__result;			\
		if (!addr)														\
			result->isnull = true;										\
		else															\
		{																\
			result->isnull = false;										\
			result->value = *((const BASETYPE *)addr);					\
		}																\
		return true;													\
	}																	\
	STATIC_FUNCTION(bool)												\
	xpu_##NAME##_arrow_ref(kern_context *kcxt,							\
						   xpu_datum_t *__result,						\
						   const kern_colmeta *cmeta,					\
						   const void *addr, int len)					\
	{																	\
		xpu_##NAME##_t *result = (xpu_##NAME##_t *)__result;			\
		if (cmeta->attopts.common.tag != ArrowType__Int ||				\
			cmeta->attopts.integer.bitWidth != 8 * sizeof(BASETYPE))	\
		{																\
			STROM_ELOG(kcxt, "value is not compatible Arrow::Int");		\
			return false;												\
		}																\
		if (!addr)														\
			result->isnull = true;										\
		else															\
		{																\
			result->isnull = false;										\
			result->value = *((BASETYPE *)addr);						\
			if (!cmeta->attopts.integer.is_signed && result->value < 0)	\
			{															\
				STROM_ELOG(kcxt, "Arrow::Uint - out of range");			\
				return false;											\
			}															\
		}																\
		return true;													\
	}																	\
	STATIC_FUNCTION(int)												\
	xpu_##NAME##_arrow_move(kern_context *kcxt,							\
							char *buffer,								\
							const kern_colmeta *cmeta,					\
							const void *addr, int len)					\
	{																	\
		if (cmeta->attopts.common.tag != ArrowType__Int ||				\
			cmeta->attopts.integer.bitWidth != 8 * sizeof(BASETYPE))	\
		{																\
			STROM_ELOG(kcxt, "value is not compatible Arrow::Int");		\
			return -1;													\
		}																\
		if (!addr)														\
			return 0;													\
		if (buffer)														\
			*((BASETYPE *)buffer) = *((BASETYPE *)addr);				\
		return sizeof(BASETYPE);										\
	}																	\
	STATIC_FUNCTION(int)												\
	xpu_##NAME##_datum_store(kern_context *kcxt,						\
							 char *buffer,								\
							 const xpu_datum_t *__arg)					\
	{																	\
		xpu_##NAME##_t *arg = (xpu_##NAME##_t *)__arg;					\
																		\
		if (arg->isnull)												\
			return 0;													\
		if (buffer)														\
 			*((BASETYPE *)buffer) = arg->value;							\
		return sizeof(BASETYPE);										\
	}																	\
	STATIC_FUNCTION(bool)												\
	xpu_##NAME##_datum_hash(kern_context *kcxt,							\
							uint32_t *p_hash,							\
							const xpu_datum_t *__arg)					\
	{																	\
		xpu_##NAME##_t *arg = (xpu_##NAME##_t *)__arg;					\
																		\
		if (arg->isnull)												\
			*p_hash = 0;												\
		else															\
			*p_hash = pg_hash_any(&arg->value, sizeof(BASETYPE));		\
		return true;													\
	}																	\
	PGSTROM_SQLTYPE_OPERATORS(NAME,true,sizeof(BASETYPE),sizeof(BASETYPE))

PGSTROM_SIMPLE_INTEGER_TEMPLATE(int1,int8_t);
PGSTROM_SIMPLE_INTEGER_TEMPLATE(int2,int16_t);
PGSTROM_SIMPLE_INTEGER_TEMPLATE(int4,int32_t);
PGSTROM_SIMPLE_INTEGER_TEMPLATE(int8,int64_t);

/*
 * FloatingPoint handlers
 */
#define PGSTROM_SIMPLE_FLOAT_TEMPLATE(NAME,BASETYPE,PRECISION)			\
	STATIC_FUNCTION(bool)												\
	xpu_##NAME##_datum_ref(kern_context *kcxt,							\
						   xpu_datum_t *__result,						\
						   const void *addr)							\
	{																	\
		xpu_##NAME##_t *result = (xpu_##NAME##_t *)__result;			\
																		\
		if (!addr)														\
			result->isnull = true;										\
		else															\
		{																\
			result->isnull = false;										\
			result->value = *((BASETYPE *)addr);						\
		}																\
		return true;													\
	}																	\
	STATIC_FUNCTION(bool)												\
	xpu_##NAME##_arrow_ref(kern_context *kcxt,							\
						   xpu_datum_t *__result,						\
						   const kern_colmeta *cmeta,					\
						   const void *addr, int len)					\
	{																	\
		xpu_##NAME##_t *result = (xpu_##NAME##_t *)__result;			\
																		\
		if (cmeta->attopts.common.tag != ArrowType__FloatingPoint ||	\
			cmeta->attopts.floating_point.precision != ArrowPrecision__##PRECISION || \
			len != sizeof(BASETYPE))									\
		{																\
			STROM_ELOG(kcxt, "Arrow::FloatingPoint - not a compatible value"); \
			return false;												\
		}																\
		if (!addr)														\
			result->isnull = true;										\
		else															\
		{																\
			result->isnull = false;										\
			result->value  = *((BASETYPE *)addr);						\
		}																\
		return true;													\
	}																	\
	STATIC_FUNCTION(int)												\
	xpu_##NAME##_arrow_move(kern_context *kcxt,							\
							char *buffer,								\
							const kern_colmeta *cmeta,					\
							const void *addr, int len)					\
	{																	\
		if (cmeta->attopts.common.tag != ArrowType__FloatingPoint ||	\
			cmeta->attopts.floating_point.precision != ArrowPrecision__##PRECISION || \
			len != sizeof(BASETYPE))									\
		{																\
			STROM_ELOG(kcxt, "Arrow::FloatingPoint - not a compatible value"); \
			return -1;													\
		}																\
		if (!addr)														\
			return 0;													\
		if (buffer)														\
			*((BASETYPE *)buffer) = *((BASETYPE *)addr);				\
		return sizeof(BASETYPE);										\
	}																	\
	STATIC_FUNCTION(int)												\
	xpu_##NAME##_datum_store(kern_context *kcxt,						\
							 char *buffer,								\
							 const xpu_datum_t *__arg)					\
	{																	\
		const xpu_##NAME##_t *arg = (const xpu_##NAME##_t *)__arg;		\
																		\
		if (arg->isnull)												\
			return 0;													\
		if (buffer)														\
			memcpy(buffer, &arg->value, sizeof(BASETYPE));				\
		return sizeof(BASETYPE);										\
	}																	\
	STATIC_FUNCTION(bool)												\
	xpu_##NAME##_datum_hash(kern_context *kcxt,							\
							uint32_t *p_hash,							\
							const xpu_datum_t *__arg)					\
	{																	\
		xpu_##NAME##_t *arg = (xpu_##NAME##_t *)__arg;					\
																		\
		if (arg->isnull)												\
			*p_hash = 0;												\
		else															\
			*p_hash = pg_hash_any(&arg->value, sizeof(BASETYPE));		\
		return true;													\
	}																	\
	PGSTROM_SQLTYPE_OPERATORS(NAME,true,sizeof(BASETYPE),sizeof(BASETYPE))

PGSTROM_SIMPLE_FLOAT_TEMPLATE(float2, float2_t, Half);
PGSTROM_SIMPLE_FLOAT_TEMPLATE(float4, float4_t, Single);
PGSTROM_SIMPLE_FLOAT_TEMPLATE(float8, float8_t, Double);

/* special support functions for float2 */
INLINE_FUNCTION(bool) isinf(float2_t fval) { return isinf(fp16_to_fp32(fval)); }
INLINE_FUNCTION(bool) isnan(float2_t fval) { return isnan(fp16_to_fp32(fval)); }
INLINE_FUNCTION(int)  lrinth(float2_t fval) { return lrintf((float)fval); }
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
		xpu_##TARGET##_t *result = (xpu_##TARGET##_t *)__result;		\
		xpu_##SOURCE##_t  datum;										\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args == 1);										\
		assert(KEXP_IS_VALID(karg,SOURCE));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum))					\
			return false;												\
		result->isnull = datum.isnull;									\
		if (!result->isnull)											\
		{																\
			if (!CHECKER(datum.value))									\
			{															\
				STROM_ELOG(kcxt, #SOURCE " to " #TARGET ": out of range"); \
				return false;											\
			}															\
			result->value = CAST(datum.value);							\
		}																\
		return true;													\
	}
#define __TYPECAST_NOCHECK(X)		(true)
#define __INTEGER_TO_BOOL(X)		((X) != 0 ? true : false)
PG_SIMPLE_TYPECAST_TEMPLATE(bool,int4,__INTEGER_TO_BOOL,__TYPECAST_NOCHECK)

#define __INTEGER_FITS_IN_INT1(X)	((X) >= SCHAR_MIN && (X) <= SCHAR_MAX)
#define __FLOAT2_FITS_IN_INT1(X)	(!isnan(X) &&						\
									 fp16_to_fp32(lrinth(X)) >= (float)SCHAR_MIN && \
									 fp16_to_fp32(lrinth(X)) <= (float)SCHAR_MAX)
#define __FLOAT4_FITS_IN_INT1(X)	(!isnan(X) &&						\
									 lrintf(X) >= (float)SCHAR_MIN &&	\
									 lrintf(X) <= (float)SCHAR_MAX)
#define __FLOAT8_FITS_IN_INT1(X)	(!isnan(X) &&						\
									 llrintf(X) >= (double)SCHAR_MIN &&	\
									 llrintf(X) <= (double)SCHAR_MAX)
PG_SIMPLE_TYPECAST_TEMPLATE(int1,int2,(int8_t), __INTEGER_FITS_IN_INT1)
PG_SIMPLE_TYPECAST_TEMPLATE(int1,int4,(int8_t), __INTEGER_FITS_IN_INT1)
PG_SIMPLE_TYPECAST_TEMPLATE(int1,int8,(int8_t), __INTEGER_FITS_IN_INT1)
PG_SIMPLE_TYPECAST_TEMPLATE(int1,float2,lrinth, __FLOAT2_FITS_IN_INT1)
PG_SIMPLE_TYPECAST_TEMPLATE(int1,float4,lrintf, __FLOAT4_FITS_IN_INT1)
PG_SIMPLE_TYPECAST_TEMPLATE(int1,float8,llrintf,__FLOAT8_FITS_IN_INT1)

#define __INTEGER_FITS_IN_INT2(X)	((X) >= SHRT_MIN && (X) <= SHRT_MAX)
#define __FLOAT2_FITS_IN_INT2(X)	(!isnan(X) &&						\
									 fp16_to_fp32(lrinth(X)) >= (float)SHRT_MIN && \
									 fp16_to_fp32(lrinth(X)) <= (float)SHRT_MAX)
#define __FLOAT4_FITS_IN_INT2(X)	(!isnan(X) &&						\
									 lrintf(X) >= (float)SHRT_MIN &&	\
									 lrintf(X) <= (float)SHRT_MAX)
#define __FLOAT8_FITS_IN_INT2(X)	(!isnan(X) &&						\
									 llrintf(X) >= (double)SHRT_MIN &&	\
									 llrintf(X) <= (double)SHRT_MAX)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,int1,(int16_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,int4,(int16_t),__INTEGER_FITS_IN_INT2)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,int8,(int16_t),__INTEGER_FITS_IN_INT2)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,float2,lrinth, __FLOAT2_FITS_IN_INT2)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,float4,lrintf, __FLOAT4_FITS_IN_INT2)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,float8,llrintf,__FLOAT8_FITS_IN_INT2)

#define __INTEGER_FITS_IN_INT4(X)	((X) >= INT_MIN && (X) <= INT_MAX)
#define __FLOAT4_FITS_IN_INT4(X)	(!isnan(X) &&						\
									 lrintf(X) >= (float)INT_MIN &&	\
									 lrintf(X) <= (float)INT_MAX)
#define __FLOAT8_FITS_IN_INT4(X)	(!isnan(X) &&						\
									 llrintf(X) >= (double)INT_MIN &&	\
									 llrintf(X) <= (double)INT_MAX)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,int1,(int32_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,int2,(int32_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,int8,(int32_t),__INTEGER_FITS_IN_INT4)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,float2,lrinth, __TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,float4,lrintf, __FLOAT4_FITS_IN_INT4)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,float8,llrintf,__FLOAT8_FITS_IN_INT4)

#define __FLOAT4_FITS_IN_INT8(X)	(!isnan((float)(X)) &&				\
									 lrintf(X) >= (float)LLONG_MIN && \
									 lrintf(X) <= (float)LLONG_MAX)
#define __FLOAT8_FITS_IN_INT8(X)	(!isnan((double)(X)) &&				\
									 llrintf(X) >= (double)LLONG_MIN && \
									 llrintf(X) <= (double)LLONG_MAX)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,int1,(int64_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,int2,(int64_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,int4,(int64_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,float2,lrinth, __TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,float4,lrintf, __FLOAT4_FITS_IN_INT8)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,float8,llrintf,__FLOAT8_FITS_IN_INT8)

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
PG_SIMPLE_TYPECAST_TEMPLATE(float8,float2,(double), __TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float8,float4,(double), __TYPECAST_NOCHECK)
#undef PG_SIMPLE_TYPECAST_TEMPLATE

__PG_SIMPLE_COMPARE_TEMPLATE(bool,bool,bool,bool,==,eq)

PG_SIMPLE_COMPARE_TEMPLATE(int1,  int1, int1, int8_t)
PG_SIMPLE_COMPARE_TEMPLATE(int12, int1, int2, int16_t)
PG_SIMPLE_COMPARE_TEMPLATE(int14, int1, int4, int32_t)
PG_SIMPLE_COMPARE_TEMPLATE(int18, int1, int8, int64_t)

PG_SIMPLE_COMPARE_TEMPLATE(int21, int2, int1, int16_t)
PG_SIMPLE_COMPARE_TEMPLATE(int2,  int2, int2, int16_t)
PG_SIMPLE_COMPARE_TEMPLATE(int24, int2, int4, int32_t)
PG_SIMPLE_COMPARE_TEMPLATE(int28, int2, int8, int64_t)

PG_SIMPLE_COMPARE_TEMPLATE(int41, int4, int1, int32_t)
PG_SIMPLE_COMPARE_TEMPLATE(int42, int4, int2, int32_t)
PG_SIMPLE_COMPARE_TEMPLATE(int4,  int4, int4, int32_t)
PG_SIMPLE_COMPARE_TEMPLATE(int48, int4, int8, int64_t)

PG_SIMPLE_COMPARE_TEMPLATE(int81, int8, int1, int64_t)
PG_SIMPLE_COMPARE_TEMPLATE(int82, int8, int2, int64_t)
PG_SIMPLE_COMPARE_TEMPLATE(int84, int8, int4, int64_t)
PG_SIMPLE_COMPARE_TEMPLATE(int8,  int8, int8, int64_t)

PG_SIMPLE_COMPARE_TEMPLATE(float2,  float2, float2, float2_t)
PG_SIMPLE_COMPARE_TEMPLATE(float24, float2, float4, float4_t)
PG_SIMPLE_COMPARE_TEMPLATE(float28, float2, float8, float8_t)
PG_SIMPLE_COMPARE_TEMPLATE(float42, float4, float2, float4_t)
PG_SIMPLE_COMPARE_TEMPLATE(float4,  float4, float4, float4_t)
PG_SIMPLE_COMPARE_TEMPLATE(float48, float4, float8, float8_t)
PG_SIMPLE_COMPARE_TEMPLATE(float82, float8, float2, float8_t)
PG_SIMPLE_COMPARE_TEMPLATE(float84, float8, float4, float8_t)
PG_SIMPLE_COMPARE_TEMPLATE(float8,  float8, float8, float8_t)

/*
 * Binary operator template
 */
#define PG_INT_BIN_OPERATOR_TEMPLATE(FNAME,RTYPE,XTYPE,YTYPE,OPER,		\
									 __TEMP,__MIN,__MAX)				\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##FNAME(XPU_PGFUNCTION_ARGS)									\
	{																	\
		xpu_##RTYPE##_t *result = (xpu_##RTYPE##_t *)__result;			\
		xpu_##XTYPE##_t x_val;											\
		xpu_##YTYPE##_t y_val;											\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args == 2);										\
		assert(KEXP_IS_VALID(karg, XTYPE));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &x_val))					\
			return false;												\
		karg = KEXP_NEXT_ARG(karg);										\
		assert(KEXP_IS_VALID(karg, YTYPE));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &y_val))					\
			return false;												\
		result->isnull = (x_val.isnull | y_val.isnull);					\
		if (!result->isnull)											\
		{																\
			__TEMP r = (__TEMP)x_val.value OPER (__TEMP)y_val.value;	\
																		\
			if (r < (__TEMP)__MIN || r > (__TEMP)__MAX)					\
			{															\
				STROM_ELOG(kcxt, #FNAME ": value out of range");		\
				return false;											\
			}															\
			result->value = r;											\
		}																\
		return true;													\
	}

#define PG_FLOAT_BIN_OPERATOR_TEMPLATE(FNAME,RTYPE,XTYPE,YTYPE,OPER,__CAST)	\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##FNAME(XPU_PGFUNCTION_ARGS)									\
	{																	\
		xpu_##RTYPE##_t *result = (xpu_##RTYPE##_t *)__result;			\
		xpu_##XTYPE##_t x_val;											\
		xpu_##YTYPE##_t y_val;											\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args == 2);										\
		assert(KEXP_IS_VALID(karg,XTYPE));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &x_val))					\
			return false;												\
		karg = KEXP_NEXT_ARG(karg);										\
		assert(KEXP_IS_VALID(karg, YTYPE));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &y_val))					\
			return false;												\
		result->isnull = (x_val.isnull | y_val.isnull);					\
		if (!result->isnull)											\
		{																\
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
PG_INT_BIN_OPERATOR_TEMPLATE(int48pl,int8,int4,int8,+,int128_t,LLONG_MAX,LLONG_MIN)
PG_INT_BIN_OPERATOR_TEMPLATE(int81pl,int8,int8,int1,+,int128_t,LLONG_MAX,LLONG_MIN)
PG_INT_BIN_OPERATOR_TEMPLATE(int82pl,int8,int8,int2,+,int128_t,LLONG_MAX,LLONG_MIN)
PG_INT_BIN_OPERATOR_TEMPLATE(int84pl,int8,int8,int4,+,int128_t,LLONG_MAX,LLONG_MIN)
PG_INT_BIN_OPERATOR_TEMPLATE(int8pl, int8,int8,int8,+,int128_t,LLONG_MAX,LLONG_MIN)

PG_FLOAT_BIN_OPERATOR_TEMPLATE(float2pl, float2,float2,float2,+,)
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
PG_INT_BIN_OPERATOR_TEMPLATE(int48mi,int8,int4,int8,-,int128_t,LLONG_MAX,LLONG_MIN)
PG_INT_BIN_OPERATOR_TEMPLATE(int81mi,int8,int8,int1,-,int128_t,LLONG_MAX,LLONG_MIN)
PG_INT_BIN_OPERATOR_TEMPLATE(int82mi,int8,int8,int2,-,int128_t,LLONG_MAX,LLONG_MIN)
PG_INT_BIN_OPERATOR_TEMPLATE(int84mi,int8,int8,int4,-,int128_t,LLONG_MAX,LLONG_MIN)
PG_INT_BIN_OPERATOR_TEMPLATE(int8mi, int8,int8,int8,-,int128_t,LLONG_MAX,LLONG_MIN)

PG_FLOAT_BIN_OPERATOR_TEMPLATE(float2mi, float2,float2,float2,-,)
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
PG_INT_BIN_OPERATOR_TEMPLATE(int48mul,int8,int4,int8,*,int128_t,LLONG_MAX,LLONG_MIN)
PG_INT_BIN_OPERATOR_TEMPLATE(int81mul,int8,int8,int1,*,int128_t,LLONG_MAX,LLONG_MIN)
PG_INT_BIN_OPERATOR_TEMPLATE(int82mul,int8,int8,int2,*,int128_t,LLONG_MAX,LLONG_MIN)
PG_INT_BIN_OPERATOR_TEMPLATE(int84mul,int8,int8,int4,*,int128_t,LLONG_MAX,LLONG_MIN)
PG_INT_BIN_OPERATOR_TEMPLATE(int8mul, int8,int8,int8,*,int128_t,LLONG_MAX,LLONG_MIN)

PG_FLOAT_BIN_OPERATOR_TEMPLATE(float2mul, float2,float2,float2,*,)
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
		xpu_##RTYPE##_t *result = (xpu_##RTYPE##_t *)__result;			\
		xpu_##XTYPE##_t x_val;											\
		xpu_##YTYPE##_t y_val;											\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args == 2);										\
		assert(KEXP_IS_VALID(karg,XTYPE));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &x_val))					\
			return false;												\
		karg = KEXP_NEXT_ARG(karg);										\
		assert(KEXP_IS_VALID(karg,YTYPE));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &y_val))					\
			return false;												\
		result->isnull = (x_val.isnull | y_val.isnull);					\
		if (!result->isnull)											\
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
		}																\
		return true;													\
	}

#define PG_FLOAT_DIV_OPERATOR_TEMPLATE(FNAME,RTYPE,XTYPE,YTYPE,__CAST)	\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##FNAME(XPU_PGFUNCTION_ARGS)                                   \
    {                                                                   \
        xpu_##RTYPE##_t *result = (xpu_##RTYPE##_t *)__result;          \
        xpu_##XTYPE##_t x_val;                                          \
        xpu_##YTYPE##_t y_val;                                          \
        const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
                                                                        \
		assert(kexp->nr_args == 2);										\
		assert(KEXP_IS_VALID(karg,XTYPE));								\
        if (!EXEC_KERN_EXPRESSION(kcxt, karg, &x_val))					\
            return false;                                               \
        karg = KEXP_NEXT_ARG(karg);										\
		assert(KEXP_IS_VALID(karg,YTYPE));								\
        if (!EXEC_KERN_EXPRESSION(kcxt, karg, &y_val))					\
            return false;                                               \
        result->isnull = (x_val.isnull | y_val.isnull);                 \
        if (!result->isnull)                                            \
        {                                                               \
			if (__iszero(y_val.value))									\
			{                                                           \
				STROM_ELOG(kcxt, #FNAME ": division by zero");          \
				return false;                                           \
			}                                                           \
			result->value = __CAST(x_val.value) / __CAST(y_val.value);	\
			/* CHECKFLOATVAL */											\
			if ((isinf(result->value) && (!isinf(x_val.value) &&		\
										  !isinf(y_val.value))) ||		\
				(__iszero(result->value) && !__iszero(x_val.value)))	\
			{															\
				STROM_ELOG(kcxt, #FNAME ": value out of range");		\
				return false;											\
			}															\
		}																\
        return true;                                                    \
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
PG_FLOAT_DIV_OPERATOR_TEMPLATE(float2div, float2,float2,float2,)
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
		xpu_##TYPE##_t *result = (xpu_##TYPE##_t *)__result;			\
		xpu_##TYPE##_t x_val;											\
		xpu_##TYPE##_t y_val;											\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args == 2);										\
		assert(KEXP_IS_VALID(karg, TYPE));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &x_val))					\
			return false;												\
		karg = KEXP_NEXT_ARG(karg);										\
		assert(KEXP_IS_VALID(karg, TYPE));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &y_val))					\
			return false;												\
		result->isnull = (x_val.isnull | y_val.isnull);					\
		if (!result->isnull)											\
		{																\
			if (y_val.value == 0)										\
			{															\
				STROM_ELOG(kcxt, #TYPE "mod : division by zero");		\
				return false;											\
			}															\
			result->value = x_val.value % y_val.value;					\
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
#define PG_UNARY_OPERATOR_TEMPLATE(FNAME,XTYPE,OPER)					\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##FNAME(XPU_PGFUNCTION_ARGS)									\
	{																	\
		xpu_##XTYPE##_t *result = (xpu_##XTYPE##_t *)__result;			\
		xpu_##XTYPE##_t x_val;											\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args==1 && KEXP_IS_VALID(karg,XTYPE));			\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &x_val))					\
			return false;												\
		result->isnull = x_val.isnull;									\
		if (!result->isnull)											\
			result->value = OPER(x_val.value);							\
		return true;													\
	}

#define PG_BITWISE_OPERATOR_TEMPLATE(FNAME,RTYPE,XTYPE,YTYPE,OPER)		\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##FNAME(XPU_PGFUNCTION_ARGS)									\
	{																	\
		xpu_##RTYPE##_t *result = (xpu_##RTYPE##_t *)__result;			\
		xpu_##XTYPE##_t x_val;											\
		xpu_##YTYPE##_t y_val;											\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(karg->nr_args == 2);										\
		assert(KEXP_IS_VALID(karg,XTYPE));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &x_val))					\
			return false;												\
		karg = KEXP_NEXT_ARG(karg);										\
		assert(KEXP_IS_VALID(karg,YTYPE));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &y_val))					\
			return false;												\
		result->isnull = (x_val.isnull | y_val.isnull);					\
		if (!result->isnull)											\
			result->value = x_val.value OPER y_val.value;				\
		return true;													\
	}

PG_UNARY_OPERATOR_TEMPLATE(int1up,int1,)
PG_UNARY_OPERATOR_TEMPLATE(int2up,int2,)
PG_UNARY_OPERATOR_TEMPLATE(int4up,int4,)
PG_UNARY_OPERATOR_TEMPLATE(int8up,int8,)
PG_UNARY_OPERATOR_TEMPLATE(float2up,float2,)
PG_UNARY_OPERATOR_TEMPLATE(float4up,float4,)
PG_UNARY_OPERATOR_TEMPLATE(float8up,float8,)

PG_UNARY_OPERATOR_TEMPLATE(int1um,int1,-)
PG_UNARY_OPERATOR_TEMPLATE(int2um,int2,-)
PG_UNARY_OPERATOR_TEMPLATE(int4um,int4,-)
PG_UNARY_OPERATOR_TEMPLATE(int8um,int8,-)
PG_UNARY_OPERATOR_TEMPLATE(float2um,float2,__fp16_unary_minus)
PG_UNARY_OPERATOR_TEMPLATE(float4um,float4,-)
PG_UNARY_OPERATOR_TEMPLATE(float8um,float8,-)

PG_UNARY_OPERATOR_TEMPLATE(int1abs,int1,abs)
PG_UNARY_OPERATOR_TEMPLATE(int2abs,int2,abs)
PG_UNARY_OPERATOR_TEMPLATE(int4abs,int4,abs)
PG_UNARY_OPERATOR_TEMPLATE(int8abs,int8,llabs)
PG_UNARY_OPERATOR_TEMPLATE(float2abs,float2,__fp16_unary_abs)
PG_UNARY_OPERATOR_TEMPLATE(float4abs,float4,fabsf)
PG_UNARY_OPERATOR_TEMPLATE(float8abs,float8,fabs)

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

PG_UNARY_OPERATOR_TEMPLATE(int1not,int1,~)
PG_UNARY_OPERATOR_TEMPLATE(int2not,int2,~)
PG_UNARY_OPERATOR_TEMPLATE(int4not,int4,~)
PG_UNARY_OPERATOR_TEMPLATE(int8not,int8,~)

PG_BITWISE_OPERATOR_TEMPLATE(int1shr,int1,int1,int4,>>)
PG_BITWISE_OPERATOR_TEMPLATE(int2shr,int2,int2,int4,>>)
PG_BITWISE_OPERATOR_TEMPLATE(int4shr,int4,int4,int4,>>)
PG_BITWISE_OPERATOR_TEMPLATE(int8shr,int8,int8,int4,>>)

PG_BITWISE_OPERATOR_TEMPLATE(int1shl,int1,int1,int4,<<)
PG_BITWISE_OPERATOR_TEMPLATE(int2shl,int2,int2,int4,<<)
PG_BITWISE_OPERATOR_TEMPLATE(int4shl,int4,int4,int4,<<)
PG_BITWISE_OPERATOR_TEMPLATE(int8shl,int8,int8,int4,<<)
