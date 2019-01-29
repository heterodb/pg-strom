/*
 * cuda_primitive.h
 *
 * Functions and operators for the primitive data types
 * --
 * Copyright 2011-2019 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2019 (C) The PG-Strom Development Team
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
#ifndef CUDA_PRIMITIVE_H
#define CUDA_PRIMITIVE_H

/*
 * type cast functions
 */
STATIC_INLINE(pg_bool_t)
pgfn_to_bool(kern_context *kcxt, pg_int4_t arg)
{
	pg_bool_t	result;

	result.isnull = arg.isnull;
	if (!result.isnull)
		result.value = (arg.value == 0 ? false : true);
	return result;
}

#define PG_SIMPLE_TYPECAST_TEMPLATE(TARGET,SOURCE,CAST)	\
	STATIC_INLINE(pg_##TARGET##_t)						\
	pgfn_to_##TARGET(kern_context *kcxt,				\
					 pg_##SOURCE##_t arg)				\
	{													\
		pg_##TARGET##_t result;							\
														\
		result.isnull = arg.isnull;						\
		if (!result.isnull)								\
			result.value = (CAST)arg.value;				\
		return result;									\
	}
PG_SIMPLE_TYPECAST_TEMPLATE(int2,int4,cl_short)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,int8,cl_short)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,float4,cl_short)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,float8,cl_short)

PG_SIMPLE_TYPECAST_TEMPLATE(int4,int2,cl_int)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,int8,cl_int)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,float4,cl_int)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,float8,cl_int)

PG_SIMPLE_TYPECAST_TEMPLATE(int8,int2,cl_int)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,int4,cl_int)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,float4,cl_int)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,float8,cl_int)

PG_SIMPLE_TYPECAST_TEMPLATE(float2,int2,cl_half)
PG_SIMPLE_TYPECAST_TEMPLATE(float2,int4,cl_half)
PG_SIMPLE_TYPECAST_TEMPLATE(float2,int8,cl_half)
PG_SIMPLE_TYPECAST_TEMPLATE(float2,float4,cl_half)
PG_SIMPLE_TYPECAST_TEMPLATE(float2,float8,cl_half)

PG_SIMPLE_TYPECAST_TEMPLATE(float4,int2,cl_float)
PG_SIMPLE_TYPECAST_TEMPLATE(float4,int4,cl_float)
PG_SIMPLE_TYPECAST_TEMPLATE(float4,int8,cl_float)
PG_SIMPLE_TYPECAST_TEMPLATE(float4,float2,cl_float)
PG_SIMPLE_TYPECAST_TEMPLATE(float4,float8,cl_float)

PG_SIMPLE_TYPECAST_TEMPLATE(float8,int2,cl_double)
PG_SIMPLE_TYPECAST_TEMPLATE(float8,int4,cl_double)
PG_SIMPLE_TYPECAST_TEMPLATE(float8,int8,cl_double)
PG_SIMPLE_TYPECAST_TEMPLATE(float8,float2,cl_double)
PG_SIMPLE_TYPECAST_TEMPLATE(float8,float4,cl_double)

#undef PG_SIMPLE_TYPECAST_TEMPLATE

/*
 * unary operators
 */
#define PG_UNARY_PLUS_TEMPLATE(NAME)			\
	STATIC_INLINE(pg_##NAME##_t)				\
	pgfn_##NAME##up(kern_context *kcxt,			\
					pg_##NAME##_t arg)			\
	{											\
		return arg;								\
	}
PG_UNARY_PLUS_TEMPLATE(int2)
PG_UNARY_PLUS_TEMPLATE(int4)
PG_UNARY_PLUS_TEMPLATE(int8)
PG_UNARY_PLUS_TEMPLATE(float2)
PG_UNARY_PLUS_TEMPLATE(float4)
PG_UNARY_PLUS_TEMPLATE(float8)
#undef PG_UNARY_PLUS_TEMPLATE

#define PG_UNARY_MINUS_TEMPLATE(NAME)			\
	STATIC_INLINE(pg_##NAME##_t)				\
	pgfn_##NAME##um(kern_context *kcxt,			\
					pg_##NAME##_t arg)			\
	{											\
		if (!arg.isnull)						\
			arg.value = -arg.value;				\
		return arg;								\
	}
PG_UNARY_MINUS_TEMPLATE(int2)
PG_UNARY_MINUS_TEMPLATE(int4)
PG_UNARY_MINUS_TEMPLATE(int8)
PG_UNARY_MINUS_TEMPLATE(float2)
PG_UNARY_MINUS_TEMPLATE(float4)
PG_UNARY_MINUS_TEMPLATE(float8)
#undef PG_UNARY_MINUS_TEMPLATE

#define PG_UNARY_NOT_TEMPLATE(NAME)				\
	STATIC_INLINE(pg_##NAME##_t)				\
	pgfn_##NAME##not(kern_context *kcxt,		\
					 pg_##NAME##_t arg)			\
	{											\
		if (!arg.isnull)						\
			arg.value = ~arg.value;				\
		return arg;								\
	}
PG_UNARY_NOT_TEMPLATE(int2)
PG_UNARY_NOT_TEMPLATE(int4)
PG_UNARY_NOT_TEMPLATE(int8)

#define PG_UNARY_ABS_TEMPLATE(NAME,CAST)		\
	STATIC_INLINE(pg_##NAME##_t)				\
	pgfn_##NAME##abs(kern_context *kcxt,		\
					 pg_##NAME##_t arg)			\
	{											\
		if (!arg.isnull)						\
			arg.value = abs((CAST)arg.value);	\
		return arg;								\
	}
PG_UNARY_ABS_TEMPLATE(int2, cl_int)
PG_UNARY_ABS_TEMPLATE(int4, cl_int)
PG_UNARY_ABS_TEMPLATE(int8, cl_long)
PG_UNARY_ABS_TEMPLATE(float2, cl_float)
PG_UNARY_ABS_TEMPLATE(float4, cl_float)
PG_UNARY_ABS_TEMPLATE(float8, cl_double)

/*
 * Simple comparison operators across data types
 */
__STROMCL_SIMPLE_COMPARE_TEMPLATE(bool,  bool,   bool,   cl_char, ==, eq)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int2,    int2,   int2,   cl_short)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int24,   int2,   int4,   cl_int)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int28,   int2,   int8,   cl_long)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int42,   int4,   int2,   cl_int)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int4,    int4,   int4,   cl_int)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int48,   int4,   int8,   cl_long)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int82,   int8,   int2,   cl_long)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int84,   int8,   int4,   cl_long)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int8,    int8,   int8,   cl_long)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float2,  float2, float2, cl_float)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float24, float2, float4, cl_float)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float28, float2, float8, cl_double)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float42, float4, float2, cl_float)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float4,  float4, float4, cl_float)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float48, float4, float8, cl_double)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float82, float8, float2, cl_double)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float84, float8, float4, cl_double)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float8,  float8, float8, cl_double)

/*
 * scalar comparison functions
 */
#define PG_SIMPLE_INT_COMPARE_TEMPLATE(LNAME,RNAME,CAST)		\
	STATIC_INLINE(pg_int4_t)									\
	pgfn_type_compare(kern_context *kcxt,						\
					  pg_##LNAME##_t arg1, pg_##RNAME##_t arg2)	\
	{															\
		pg_int4_t result;										\
																\
		result.isnull = arg1.isnull | arg2.isnull;				\
		if (!result.isnull)										\
		{														\
			if ((CAST)arg1.value < (CAST)arg2.value)			\
				result.value = -1;								\
			else if ((CAST)arg1.value > (CAST)arg2.value)		\
				result.value = 1;								\
			else												\
				result.value = 0;								\
		}														\
		return result;											\
	}

#define PG_SIMPLE_FLOAT_COMPARE_TEMPLATE(LNAME,RNAME,CAST)		\
	STATIC_INLINE(pg_int4_t)									\
	pgfn_type_compare(kern_context *kcxt,						\
					  pg_##LNAME##_t arg1, pg_##RNAME##_t arg2)	\
	{															\
		pg_int4_t result;										\
																\
		result.isnull = arg1.isnull | arg2.isnull;				\
		if (!result.isnull)										\
		{														\
			if (isnan((CAST)arg1.value))						\
			{													\
				if (isnan((CAST)arg2.value))					\
					result.value = 0;	/* NAN = NAN */			\
				else											\
					result.value = 1;	/* NAN > non-NAN */		\
			}													\
			else if (isnan((CAST)arg2.value))					\
			{													\
				result.value = -1;		/* non-NAN < NAN */		\
			}													\
			else												\
			{													\
				if ((CAST)arg1.value > (CAST)arg2.value)		\
					result.value = 1;							\
				else if ((CAST)arg1.value < (CAST)arg2.value)	\
					result.value = -1;							\
				else											\
					result.value = 0;							\
			}													\
		}														\
		return result;											\
	}

PG_SIMPLE_INT_COMPARE_TEMPLATE(bool, bool, cl_int)
PG_SIMPLE_INT_COMPARE_TEMPLATE(int2, int2, cl_int)
PG_SIMPLE_INT_COMPARE_TEMPLATE(int2, int4, cl_int)
PG_SIMPLE_INT_COMPARE_TEMPLATE(int2, int8, cl_long)
PG_SIMPLE_INT_COMPARE_TEMPLATE(int4, int2, cl_int)
PG_SIMPLE_INT_COMPARE_TEMPLATE(int4, int4, cl_int)
PG_SIMPLE_INT_COMPARE_TEMPLATE(int4, int8, cl_int)
PG_SIMPLE_INT_COMPARE_TEMPLATE(int8, int2, cl_long)
PG_SIMPLE_INT_COMPARE_TEMPLATE(int8, int4, cl_long)
PG_SIMPLE_INT_COMPARE_TEMPLATE(int8, int8, cl_long)
PG_SIMPLE_FLOAT_COMPARE_TEMPLATE(float2, float2, cl_float)
PG_SIMPLE_FLOAT_COMPARE_TEMPLATE(float2, float4, cl_float)
PG_SIMPLE_FLOAT_COMPARE_TEMPLATE(float2, float8, cl_double)
PG_SIMPLE_FLOAT_COMPARE_TEMPLATE(float4, float2, cl_float)
PG_SIMPLE_FLOAT_COMPARE_TEMPLATE(float4, float4, cl_float)
PG_SIMPLE_FLOAT_COMPARE_TEMPLATE(float4, float8, cl_double)
PG_SIMPLE_FLOAT_COMPARE_TEMPLATE(float8, float2, cl_double)
PG_SIMPLE_FLOAT_COMPARE_TEMPLATE(float8, float4, cl_double)
PG_SIMPLE_FLOAT_COMPARE_TEMPLATE(float8, float8, cl_double)

#undef PG_SIMPLE_INT_COMPARE_TEMPLATE
#undef PG_SIMPLE_FLOAT_COMPARE_TEMPLATE

/*
 * Larger/Smaller functions
 */
#define PG_SIMPLE_LARGER_SMALLER_TEMPLATE(NAME)				\
	STATIC_INLINE(pg_##NAME##_t)							\
	pgfn_larger(kern_context *kcxt,							\
				pg_##NAME##_t arg1, pg_##NAME##_t arg2)		\
	{														\
		pg_##NAME##_t result;								\
															\
		result.isnull = arg1.isnull | arg2.isnull;			\
		if (!result.isnull)									\
			result.value = Max(arg1.value, arg2.value);		\
		return result;										\
	}														\
															\
	STATIC_INLINE(pg_##NAME##_t)							\
	pgfn_smaller(kern_context *kcxt,						\
				 pg_##NAME##_t arg1, pg_##NAME##_t arg2)	\
	{														\
		pg_##NAME##_t result;								\
															\
		result.isnull = arg1.isnull | arg2.isnull;			\
		if (!result.isnull)									\
			result.value = Min(arg1.value, arg2.value);		\
		return result;										\
	}

PG_SIMPLE_LARGER_SMALLER_TEMPLATE(int2)
PG_SIMPLE_LARGER_SMALLER_TEMPLATE(int4)
PG_SIMPLE_LARGER_SMALLER_TEMPLATE(int8)
PG_SIMPLE_LARGER_SMALLER_TEMPLATE(float2)
PG_SIMPLE_LARGER_SMALLER_TEMPLATE(float4)
PG_SIMPLE_LARGER_SMALLER_TEMPLATE(float8)

#undef PG_SIMPLE_LARGER_SMALLER_TEMPLATE

/*
 * integer bitwise operations
 */
#define PG_INTEGER_BITWISE_OPER_TEMPLATE(NAME,OPER,EXTRA)	\
	STATIC_INLINE(pg_##NAME##_t)							\
	pgfn_##NAME##EXTRA(kern_context *kcxt,					\
					   pg_##NAME##_t arg1, pg_##NAME##_t arg2)	\
	{														\
		pg_##NAME##_t result;								\
															\
		result.isnull = arg1.isnull | arg2.isnull;			\
		if (!result.isnull)									\
			result.value = (arg1.value OPER arg2.value);	\
		return result;										\
	}
PG_INTEGER_BITWISE_OPER_TEMPLATE(int2,&,and)
PG_INTEGER_BITWISE_OPER_TEMPLATE(int4,&,and)
PG_INTEGER_BITWISE_OPER_TEMPLATE(int8,&,and)
PG_INTEGER_BITWISE_OPER_TEMPLATE(int2,|,or)
PG_INTEGER_BITWISE_OPER_TEMPLATE(int4,|,or)
PG_INTEGER_BITWISE_OPER_TEMPLATE(int8,|,or)
PG_INTEGER_BITWISE_OPER_TEMPLATE(int2,^,xor)
PG_INTEGER_BITWISE_OPER_TEMPLATE(int4,^,xor)
PG_INTEGER_BITWISE_OPER_TEMPLATE(int8,^,xor)
#undef PG_INTEGER_BITWISE_OPER_TEMPLATE

/*
 * integer bit shift operators
 */
#define PG_INTEGER_BITSHIFT_OPER_TEMPLATE(NAME,OPER,EXTRA)	\
	STATIC_INLINE(pg_##NAME##_t)							\
	pgfn_##NAME##EXTRA(kern_context *kcxt,					\
					   pg_##NAME##_t arg1, pg_int4_t arg2)	\
	{														\
		pg_##NAME##_t result;								\
															\
		result.isnull = arg1.isnull | arg2.isnull;			\
		if (!result.isnull)									\
			result.value = (arg1.value OPER arg2.value);	\
		return result;										\
	}
PG_INTEGER_BITSHIFT_OPER_TEMPLATE(int2,<<,shl)
PG_INTEGER_BITSHIFT_OPER_TEMPLATE(int2,>>,shr)
PG_INTEGER_BITSHIFT_OPER_TEMPLATE(int4,<<,shl)
PG_INTEGER_BITSHIFT_OPER_TEMPLATE(int4,>>,shr)
PG_INTEGER_BITSHIFT_OPER_TEMPLATE(int8,<<,shl)
PG_INTEGER_BITSHIFT_OPER_TEMPLATE(int8,>>,shr)

/*
 * type re-interpretation routines
 */
#define PG_TYPE_REINTERPRETE_TEMPLATE(TARGET,SOURCE,OPER)		\
	STATIC_INLINE(pg_##TARGET##_t)								\
	pgfn_as_##TARGET(kern_context *kcxt, pg_##SOURCE##_t arg)	\
	{															\
		pg_##TARGET##_t result;									\
																\
		result.isnull = arg.isnull;								\
		if (!result.isnull)										\
			result.value = OPER(arg.value);						\
		return result;											\
	}
PG_TYPE_REINTERPRETE_TEMPLATE(int8,float8,__double_as_longlong)
PG_TYPE_REINTERPRETE_TEMPLATE(int4,float4,__float_as_int)
PG_TYPE_REINTERPRETE_TEMPLATE(int2,float2,__half_as_short)
PG_TYPE_REINTERPRETE_TEMPLATE(float8,int8,__longlong_as_double)
PG_TYPE_REINTERPRETE_TEMPLATE(float4,int4,__int_as_float)
PG_TYPE_REINTERPRETE_TEMPLATE(float2,int2,__short_as_half)
#undef PG_TYPE_REINTERPRETE_TEMPLATE

#endif	/* CUDA_PRIMITIVE_H */
