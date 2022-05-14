/*
 * xpu_basetype.c
 *
 * Collection of primitive Int/Float type support on XPU(GPU/DPU/SPU)
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 *
 */
#include "xpu_common.h"

PGSTROM_SIMPLE_BASETYPE_TEMPLATE(bool, int8_t);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(int1, int8_t);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(int2, int16_t);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(int4, int32_t);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(int8, int64_t);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(float2, float2_t);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(float4, float4_t);
PGSTROM_SIMPLE_BASETYPE_TEMPLATE(float8, float8_t);

/*
 * XPU Type Cast functions
 */
#define PG_SIMPLE_TYPECAST_TEMPLATE(TARGET,SOURCE,CAST,CHECK)			\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##SOURCE##_to_##TARGET(XPU_PGFUNCTION_ARGS)					\
	{																	\
		xpu_##TARGET##_t *result = (xpu_##TARGET##_t *)__result;		\
		xpu_##SOURCE##_t  datum;										\
		const kern_expression *arg = KEXP_FIRST_ARG(1,SOURCE);			\
																		\
		if (!EXEC_KERN_EXPRESSION(kcxt, arg, &datum))					\
			return false;												\
		result->ops = &xpu_##TARGET##_ops;								\
		result->isnull = datum.isnull;									\
		if (!result->isnull)											\
		{																\
			if (!CHECK(datum.value))									\
			{															\
				STROM_ELOG(kcxt, #SOURCE " to " #TARGET ": out of range"); \
				return false;											\
			}															\
			result->value = CAST(datum.value);							\
		}																\
		return true;													\
	}

#define __TYPECAST_NOCHECK(X)		(true)
#define __INTEGER_FITS_IN_INT1(X)	((X) >= SCHAR_MIN && (X) <= SCHAR_MAX)
#define __FLOAT4_FITS_IN_INT1(X)	(!isnan((float)(X)) &&				\
									 rint((float)(X)) >= (float)SCHAR_MIN && \
									 rint((float)(X)) <= (float)SCHAR_MAX)
#define __FLOAT8_FITS_IN_INT1(X)	(!isnan((double)(X)) &&				\
									 rint((double)(X)) >= (double)SCHAR_MIN && \
									 rint((double)(X)) <= (double)SCHAR_MAX)
INLINE_FUNCTION(long int) lrinth(float2_t fval) { return lrintf((float)fval); }
PG_SIMPLE_TYPECAST_TEMPLATE(int1,int2,(int8_t),__INTEGER_FITS_IN_INT1)
PG_SIMPLE_TYPECAST_TEMPLATE(int1,int4,(int8_t),__INTEGER_FITS_IN_INT1)
PG_SIMPLE_TYPECAST_TEMPLATE(int1,int8,(int8_t),__INTEGER_FITS_IN_INT1)
PG_SIMPLE_TYPECAST_TEMPLATE(int1,float2,lrinth,__FLOAT4_FITS_IN_INT1)
PG_SIMPLE_TYPECAST_TEMPLATE(int1,float4,lrintf,__FLOAT4_FITS_IN_INT1)
PG_SIMPLE_TYPECAST_TEMPLATE(int1,float8,llrintf,__FLOAT8_FITS_IN_INT1)

#define __INTEGER_FITS_IN_INT2(X)	((X) >= SHRT_MIN && (X) <= SHRT_MAX)
#define __FLOAT4_FITS_IN_INT2(X)	(!isnan((float)(X)) &&				\
									 rint((float)(X)) >= (float)SHRT_MIN &&	\
									 rint((float)(X)) <= (float)SHRT_MAX)
#define __FLOAT8_FITS_IN_INT2(X)	(!isnan((double)(X)) &&				\
									 rint((double)(X)) >= (double)SHRT_MIN && \
									 rint((double)(X)) <= (double)SHRT_MAX)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,int1,(int16_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,int4,(int16_t),__INTEGER_FITS_IN_INT2)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,int8,(int16_t),__INTEGER_FITS_IN_INT2)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,float2,lrinth,__FLOAT4_FITS_IN_INT2)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,float4,lrintf,__FLOAT4_FITS_IN_INT2)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,float8,llrintf,__FLOAT8_FITS_IN_INT2)

#define __INTEGER_FITS_IN_INT4(X)	((X) >= INT_MIN && (X) <= INT_MAX)
#define __FLOAT4_FITS_IN_INT4(X)	(!isnan((float)(X)) &&				\
									 rint((float)(X)) >= (float)INT_MIN && \
									 rint((float)(X)) <= (float)INT_MAX)
#define __FLOAT8_FITS_IN_INT4(X)	(!isnan((double)(X)) &&				\
									 rint((double)(X)) >= (double)INT_MIN && \
									 rint((double)(X)) <= (double)INT_MAX)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,int1,(int32_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,int2,(int32_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,int8,(int32_t),__INTEGER_FITS_IN_INT4)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,float2,lrinth,__FLOAT4_FITS_IN_INT4)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,float4,lrintf,__FLOAT4_FITS_IN_INT4)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,float8,llrintf,__FLOAT8_FITS_IN_INT4)

#define __FLOAT4_FITS_IN_INT8(X)	(!isnan((float)(X)) &&				\
									 rint((float)(X)) >= (float)LLONG_MIN && \
									 rint((float)(X)) <= (float)LLONG_MAX)
#define __FLOAT8_FITS_IN_INT8(X)	(!isnan((double)(X)) &&				\
									 rint((double)(X)) >= (double)LLONG_MIN && \
									 rint((double)(X)) <= (double)LLONG_MAX)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,int1,(int64_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,int2,(int64_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,int4,(int64_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,float2,lrinth,__FLOAT4_FITS_IN_INT8)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,float4,lrintf,__FLOAT4_FITS_IN_INT8)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,float8,llrintf,__FLOAT8_FITS_IN_INT8)

#define __i2fp16(X)					((float2_t)((float)(X)))
#define __INTEGER_FITS_IN_FLOAT2(X)	!isinf((float)__i2fp16(X))
#define __FLOAT_FITS_IN_FLOAT2(X)	((!isinf((float)((float2_t)(X))) || isinf(X)) && \
									 ((float)((float2_t)(X))) != 0.0 || (X) == 0.0)
PG_SIMPLE_TYPECAST_TEMPLATE(float2,int1,__i2fp16,__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float2,int2,__i2fp16,__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float2,int4,__i2fp16,__INTEGER_FITS_IN_FLOAT2)
PG_SIMPLE_TYPECAST_TEMPLATE(float2,int8,__i2fp16,__INTEGER_FITS_IN_FLOAT2)
PG_SIMPLE_TYPECAST_TEMPLATE(float2,float4,(float2_t),__FLOAT_FITS_IN_FLOAT2)
PG_SIMPLE_TYPECAST_TEMPLATE(float2,float8,(float2_t),__FLOAT_FITS_IN_FLOAT2)
#undef __i2fp16

#define __FLOAT_FITS_IN_FLOAT4(X)	((!isinf((float4_t)(X)) || isinf(X)) && \
									 ((float4_t)(X) != 0.0 || (X) == 0.0))
PG_SIMPLE_TYPECAST_TEMPLATE(float4,int1,(float4_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float4,int2,(float4_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float4,int4,(float4_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float4,int8,(float4_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float4,float2,(float4_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float4,float8,(float4_t),__FLOAT_FITS_IN_FLOAT4)

PG_SIMPLE_TYPECAST_TEMPLATE(float8,int1,(float8_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float8,int2,(float8_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float8,int4,(float8_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float8,int8,(float8_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float8,float2,(float8_t),__TYPECAST_NOCHECK)
PG_SIMPLE_TYPECAST_TEMPLATE(float8,float4,(float8_t),__TYPECAST_NOCHECK)
#undef PG_SIMPLE_TYPECAST_TEMPLATE

#define __PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,OPER,EXTRA)	\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##FNAME##EXTRA(XPU_PGFUNCTION_ARGS)							\
	{																	\
		xpu_bool_t *result = (xpu_bool_t *)__result;					\
		xpu_##LNAME##_t lval;											\
		xpu_##RNAME##_t rval;											\
		const kern_expression *arg = KEXP_FIRST_ARG(2,LNAME);			\
																		\
		if (!EXEC_KERN_EXPRESSION(kcxt, arg, &lval))					\
			return false;												\
		arg = KEXP_NEXT_ARG(arg, RNAME);								\
		if (!EXEC_KERN_EXPRESSION(kcxt, arg, &rval))					\
			return false;												\
		result->ops = &xpu_bool_ops;									\
		result->isnull = (lval.isnull | rval.isnull);					\
		if (!result->isnull)											\
			result->value = ((CAST)lval.value OPER (CAST)rval.value);	\
		return true;													\
	}

#define PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST)		\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,==,eq)	\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,!=,ne)	\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,<,lt)	\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,<=,le)	\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,>,gt)	\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,>=,ge)

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

/* '+' : add operators */



/* '-' : subtract operators */


/* '*' : multiply operators */



/* '/' : divide operators */
