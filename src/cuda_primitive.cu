/*
 * cuda_primitive.cu
 *
 * Collection of primitive functions for CUDA GPU devices
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
#include "cuda_common.h"
#include "cuda_primitive.h"

/*
 * Utility macros
 */
#define CHECKFLOATVAL(kerror, result, inf_is_valid, zero_is_valid)	\
	do {															\
		if (isinf((result).value) && !(inf_is_valid))				\
		{															\
			(result).isnull = true;									\
			STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,	\
						  "value out of range: overflow");			\
		}															\
		if ((result).value == 0.0 && !(zero_is_valid))				\
		{															\
			(result).isnull = true;									\
			STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,	\
						  "value out of range: underflow");			\
		}															\
	} while(0)

#ifndef SAMESIGN
#define SAMESIGN(a,b)	(((a) < 0) == ((b) < 0))
#endif

/*
 * Functions for addition operator on basic data types
 */
#define BASIC_INT_ADDFUNC_TEMPLATE(name,r_type,x_type,y_type)		\
	DEVICE_FUNCTION(pg_##r_type##_t)								\
	pgfn_##name(kern_context *kcxt,									\
				pg_##x_type##_t arg1, pg_##y_type##_t arg2)			\
	{																\
		pg_##r_type##_t	result;										\
																	\
		result.isnull = arg1.isnull | arg2.isnull;					\
		if (!result.isnull)											\
		{															\
			result.value = arg1.value + arg2.value;					\
			if (SAMESIGN(arg1.value, arg2.value) &&					\
				!SAMESIGN(result.value, arg1.value))				\
			{														\
				result.isnull = true;								\
				STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,\
							  "integer out of range");				\
			}														\
		}															\
		return result;												\
	}

#define BASIC_FLOAT_ADDFUNC_TEMPLATE(name,r_type,x_type,y_type,cast_t) \
	DEVICE_FUNCTION(pg_##r_type##_t)								\
	pgfn_##name(kern_context *kcxt,									\
				pg_##x_type##_t arg1, pg_##y_type##_t arg2)         \
    {																\
		pg_##r_type##_t	result;										\
																	\
		result.isnull = arg1.isnull | arg2.isnull;					\
		if (!result.isnull)											\
		{															\
			result.value = (cast_t)arg1.value + (cast_t)arg2.value;	\
			CHECKFLOATVAL(&kcxt->e, result,							\
						  isinf((cast_t)arg1.value) ||				\
						  isinf((cast_t)arg2.value), true);			\
		}                                                           \
		return result;												\
	}

BASIC_INT_ADDFUNC_TEMPLATE(int2pl, int2,int2,int2)
BASIC_INT_ADDFUNC_TEMPLATE(int24pl,int4,int2,int4)
BASIC_INT_ADDFUNC_TEMPLATE(int28pl,int8,int2,int8)

BASIC_INT_ADDFUNC_TEMPLATE(int42pl,int4,int4,int2)
BASIC_INT_ADDFUNC_TEMPLATE(int4pl, int4,int4,int4)
BASIC_INT_ADDFUNC_TEMPLATE(int48pl,int8,int4,int8)

BASIC_INT_ADDFUNC_TEMPLATE(int82pl,int8,int8,int2)
BASIC_INT_ADDFUNC_TEMPLATE(int84pl,int8,int8,int4)
BASIC_INT_ADDFUNC_TEMPLATE(int8pl, int8,int8,int8)

BASIC_FLOAT_ADDFUNC_TEMPLATE(float2pl, float4, float2, float2, cl_float)
BASIC_FLOAT_ADDFUNC_TEMPLATE(float24pl,float4, float2, float4, cl_float)
BASIC_FLOAT_ADDFUNC_TEMPLATE(float28pl,float8, float2, float8, cl_double)
BASIC_FLOAT_ADDFUNC_TEMPLATE(float42pl,float4, float4, float2, cl_float)
BASIC_FLOAT_ADDFUNC_TEMPLATE(float4pl, float4, float4, float4, cl_float)
BASIC_FLOAT_ADDFUNC_TEMPLATE(float48pl,float8, float4, float8, cl_double)
BASIC_FLOAT_ADDFUNC_TEMPLATE(float82pl,float8, float8, float2, cl_double)
BASIC_FLOAT_ADDFUNC_TEMPLATE(float84pl,float8, float8, float4, cl_double)
BASIC_FLOAT_ADDFUNC_TEMPLATE(float8pl, float8, float8, float8, cl_double)


#undef BASIC_INT_ADDFUNC_TEMPLATE
#undef BASIC_FLOAT_ADDFUNC_TEMPLATE

/*
 * Functions for addition operator on basic data types
 */
#define BASIC_INT_SUBFUNC_TEMPLATE(name,r_type,x_type,y_type)		\
	DEVICE_FUNCTION(pg_##r_type##_t)								\
	pgfn_##name(kern_context *kcxt,									\
				pg_##x_type##_t arg1, pg_##y_type##_t arg2)			\
	{																\
		pg_##r_type##_t	result;										\
																	\
		result.isnull = arg1.isnull | arg2.isnull;					\
		if (!result.isnull)											\
		{															\
			result.value = arg1.value - arg2.value;					\
			if (!SAMESIGN(arg1.value, arg2.value) &&				\
				!SAMESIGN(result.value, arg1.value))				\
			{														\
				result.isnull = true;								\
				STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,	\
							  "integer out of range");				\
			}														\
		}															\
		return result;												\
	}

#define BASIC_FLOAT_SUBFUNC_TEMPLATE(name,r_type,x_type,y_type,cast_t) \
	DEVICE_FUNCTION(pg_##r_type##_t)								\
	pgfn_##name(kern_context *kcxt,									\
				pg_##x_type##_t arg1, pg_##y_type##_t arg2)         \
    {																\
		pg_##r_type##_t	result;										\
																	\
		result.isnull = arg1.isnull | arg2.isnull;					\
		if (!result.isnull)											\
		{															\
			result.value = (cast_t)arg1.value - (cast_t)arg2.value;	\
			CHECKFLOATVAL(&kcxt->e, result,							\
						  isinf((cast_t)arg1.value) ||				\
						  isinf((cast_t)arg2.value), true);			\
		}                                                           \
		return result;												\
	}

BASIC_INT_SUBFUNC_TEMPLATE(int2mi,  int2, int2, int2)
BASIC_INT_SUBFUNC_TEMPLATE(int24mi, int4, int2, int4)
BASIC_INT_SUBFUNC_TEMPLATE(int28mi, int8, int2, int8)

BASIC_INT_SUBFUNC_TEMPLATE(int42mi, int4, int4, int2)
BASIC_INT_SUBFUNC_TEMPLATE(int4mi,  int4, int4, int4)
BASIC_INT_SUBFUNC_TEMPLATE(int48mi, int8, int4, int8)

BASIC_INT_SUBFUNC_TEMPLATE(int82mi, int8, int8, int2)
BASIC_INT_SUBFUNC_TEMPLATE(int84mi, int8, int8, int4)
BASIC_INT_SUBFUNC_TEMPLATE(int8mi,  int8, int8, int8)

BASIC_FLOAT_SUBFUNC_TEMPLATE(float2mi,  float4, float2, float2, cl_float)
BASIC_FLOAT_SUBFUNC_TEMPLATE(float24mi, float4, float2, float4, cl_float)
BASIC_FLOAT_SUBFUNC_TEMPLATE(float28mi, float8, float2, float8, cl_double)
BASIC_FLOAT_SUBFUNC_TEMPLATE(float42mi, float4, float4, float2, cl_float)
BASIC_FLOAT_SUBFUNC_TEMPLATE(float4mi,  float4, float4, float4, cl_float)
BASIC_FLOAT_SUBFUNC_TEMPLATE(float48mi, float8, float4, float8, cl_double)
BASIC_FLOAT_SUBFUNC_TEMPLATE(float82mi, float8, float8, float2, cl_double)
BASIC_FLOAT_SUBFUNC_TEMPLATE(float84mi, float8, float8, float4, cl_double)
BASIC_FLOAT_SUBFUNC_TEMPLATE(float8mi,  float8, float8, float8, cl_double)

#undef BASIC_INT_SUBFUNC_TEMPLATE
#undef BASIC_FLOAT_SUBFUNC_TEMPLATE


/*
 * Functions for multiplication operator on basic data types
 */
DEVICE_FUNCTION(pg_int2_t)
pgfn_int2mul(kern_context *kcxt, pg_int2_t arg1, pg_int2_t arg2)
{
	pg_int2_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		cl_int	temp = (cl_int)arg1.value * (cl_int)arg2.value;

		if (temp < SHRT_MIN || temp > SHRT_MAX)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
						  "smallint out of range");
		}
		else
			result.value = (cl_short) temp;
	}
	return result;
}

DEVICE_FUNCTION(pg_int4_t)
pgfn_int24mul(kern_context *kcxt, pg_int2_t arg1, pg_int4_t arg2)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = (cl_int)arg1.value * arg2.value;
		/* logic copied from int24mul() */
		if (!(arg2.value >= (cl_int) SHRT_MIN &&
			  arg2.value <= (cl_int) SHRT_MAX) &&
			result.value / arg2.value != arg1.value)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
						  "integer out of range");
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_int8_t)
pgfn_int28mul(kern_context *kcxt, pg_int2_t arg1, pg_int8_t arg2)
{
	pg_int8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = (cl_long)arg1.value * arg2.value;
		/* logic copied from int28mul() */
		if (arg2.value != (cl_long)((cl_int) arg2.value) &&
			result.value / arg2.value != arg1.value)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
						  "bigint out of range");
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_int4_t)
pgfn_int42mul(kern_context *kcxt, pg_int4_t arg1, pg_int2_t arg2)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * (cl_int)arg2.value;
		/* logic copied from int42mul() */
		if (!(arg1.value >= (cl_int)SHRT_MIN &&
			  arg1.value <= (cl_int) SHRT_MAX) &&
			result.value / arg1.value != arg2.value)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
						  "integer out of range");
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_int4_t)
pgfn_int4mul(kern_context *kcxt, pg_int4_t arg1, pg_int4_t arg2)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * arg2.value;
		/* logic copied from int4mul() */
		if (!(arg1.value >= (cl_int) SHRT_MIN &&
			  arg1.value <= (cl_int) SHRT_MAX &&
			  arg2.value >= (cl_int) SHRT_MIN &&
			  arg2.value <= (cl_int) SHRT_MAX) &&
			arg2.value != 0 &&
			((arg2.value == -1 && arg1.value < 0 && result.value < 0) ||
			 result.value / arg2.value != arg1.value))
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
						  "integer out of range");
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_int8_t)
pgfn_int48mul(kern_context *kcxt, pg_int4_t arg1, pg_int8_t arg2)
{
	pg_int8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * arg2.value;
		/* logic copied from int48mul() */
		if (arg2.value != (cl_long) ((cl_int) arg2.value) &&
			result.value / arg2.value != arg1.value)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
						  "bigint out of range");
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_int8_t)
pgfn_int82mul(kern_context *kcxt, pg_int8_t arg1, pg_int2_t arg2)
{
	pg_int8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * (cl_long) arg2.value;
		/* logic copied from int82mul() */
		if (arg1.value != (cl_long) ((cl_int) arg1.value) &&
			result.value / arg1.value != arg2.value)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
						  "bigint out of range");
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_int8_t)
pgfn_int84mul(kern_context *kcxt, pg_int8_t arg1, pg_int4_t arg2)
{
	pg_int8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * (cl_long) arg2.value;
		/* logic copied from int84mul() */
		if (arg1.value != (cl_long) ((cl_int) arg1.value) &&
			result.value / arg1.value != arg2.value)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
						  "bigint out of range");
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_int8_t)
pgfn_int8mul(kern_context *kcxt, pg_int8_t arg1, pg_int8_t arg2)
{
	pg_int8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * arg2.value;
		/* logic copied from int8mul() */
		if ((arg1.value != (cl_long) ((cl_int) arg1.value) ||
			 arg2.value != (cl_long) ((cl_int) arg2.value)) &&
			(arg2.value != 0 &&
			 ((arg2.value == -1 && arg1.value < 0 && result.value < 0) ||
			  result.value / arg2.value != arg1.value)))
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
						  "bigint out of range");
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_float4_t)
pgfn_float2mul(kern_context *kcxt, pg_float2_t arg1, pg_float2_t arg2)
{
	cl_float	value1 = arg1.value;
	cl_float	value2 = arg2.value;
	pg_float4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = value1 * value2;
		CHECKFLOATVAL(&kcxt->e, result,
					  isinf(value1) * isinf(value2),
					  value1 == 0.0 || value2 == 0.0);
	}
	return result;
}


DEVICE_FUNCTION(pg_float4_t)
pgfn_float24mul(kern_context *kcxt, pg_float2_t arg1, pg_float4_t arg2)
{
	cl_float	value1 = arg1.value;
	pg_float4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = value1 * arg2.value;
		CHECKFLOATVAL(&kcxt->e, result,
					  isinf(value1) || isinf(arg2.value),
					  value1 == 0.0 || arg2.value == 0.0);
	}
	return result;
}

DEVICE_FUNCTION(pg_float8_t)
pgfn_float28mul(kern_context *kcxt, pg_float2_t arg1, pg_float8_t arg2)
{
	cl_double	value1 = arg1.value;
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = value1 * arg2.value;
		CHECKFLOATVAL(&kcxt->e, result,
					  isinf(value1) || isinf(arg2.value),
					  value1 == 0.0 || arg2.value == 0.0);
	}
	return result;
}

DEVICE_FUNCTION(pg_float4_t)
pgfn_float42mul(kern_context *kcxt, pg_float4_t arg1, pg_float2_t arg2)
{
	cl_float	value2 = arg2.value;
	pg_float4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * (cl_float)arg2.value;
		CHECKFLOATVAL(&kcxt->e, result,
					  isinf(arg1.value) || isinf(value2),
					  arg1.value == 0.0 || value2 == 0.0);
	}
	return result;
}

DEVICE_FUNCTION(pg_float4_t)
pgfn_float4mul(kern_context *kcxt, pg_float4_t arg1, pg_float4_t arg2)
{
	pg_float4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * arg2.value;
		CHECKFLOATVAL(&kcxt->e, result,
					  isinf(arg1.value) || isinf(arg2.value),
					  arg1.value == 0.0 || arg2.value == 0.0);
	}
	return result;
}

DEVICE_FUNCTION(pg_float8_t)
pgfn_float48mul(kern_context *kcxt, pg_float4_t arg1, pg_float8_t arg2)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = (cl_double)arg1.value * arg2.value;
		CHECKFLOATVAL(&kcxt->e, result,
					  isinf(arg1.value) || isinf(arg2.value),
					  arg1.value == 0.0 || arg2.value == 0.0);
	}
	return result;
}

DEVICE_FUNCTION(pg_float8_t)
pgfn_float82mul(kern_context *kcxt, pg_float8_t arg1, pg_float2_t arg2)
{
	cl_double	value2 = arg2.value;
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * value2;
		CHECKFLOATVAL(&kcxt->e, result,
					  isinf(arg1.value) || isinf(value2),
					  arg1.value == 0.0 || value2 == 0.0);
	}
	return result;
}

DEVICE_FUNCTION(pg_float8_t)
pgfn_float84mul(kern_context *kcxt, pg_float8_t arg1, pg_float4_t arg2)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * (cl_double)arg2.value;
		CHECKFLOATVAL(&kcxt->e, result,
					  isinf(arg1.value) || isinf(arg2.value),
					  arg1.value == 0.0 || arg2.value == 0.0);
	}
	return result;
}

DEVICE_FUNCTION(pg_float8_t)
pgfn_float8mul(kern_context *kcxt, pg_float8_t arg1, pg_float8_t arg2)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * arg2.value;
		CHECKFLOATVAL(&kcxt->e, result,
					  isinf(arg1.value) || isinf(arg2.value),
					  arg1.value == 0.0 || arg2.value == 0.0);
	}
	return result;
}

/*
 * Functions for division operator on basic data types
 */
#define SAMESIGN(a,b)	(((a) < 0) == ((b) < 0))

DEVICE_FUNCTION(pg_int2_t)
pgfn_int2div(kern_context *kcxt, pg_int2_t arg1, pg_int2_t arg2)
{
	pg_int2_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
						  "division by zero");
		}
		else if (arg2.value == -1)
		{
			result.value = -arg1.value;
			if (arg1.value != 0 && SAMESIGN(result.value, arg1.value))
			{
				result.isnull = true;
				STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
							  "smallint out of range");
			}
		}
		else
			result.value = arg1.value / arg2.value;
	}
	return result;
}

DEVICE_FUNCTION(pg_int4_t)
pgfn_int24div(kern_context *kcxt, pg_int2_t arg1, pg_int4_t arg2)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
						  "division by zero");
		}
		else
			result.value = (cl_int) arg1.value / arg2.value;
	}
	return result;
}

DEVICE_FUNCTION(pg_int8_t)
pgfn_int28div(kern_context *kcxt, pg_int2_t arg1, pg_int8_t arg2)
{
	pg_int8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
                          "division by zero");
		}
		else
			result.value = (cl_long) arg1.value / arg2.value;
	}
	return result;
}

DEVICE_FUNCTION(pg_int4_t)
pgfn_int42div(kern_context *kcxt, pg_int4_t arg1, pg_int2_t arg2)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
    {
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
						  "division by zero");
		}
		else if (arg2.value == -1)
		{
			result.value = -arg1.value;
			if (arg1.value != 0 && SAMESIGN(result.value, arg1.value))
			{
				result.isnull = true;
				STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
							  "integer out of range");
			}
		}
		else
			result.value = arg1.value / arg2.value;
	}
	return result;
}

DEVICE_FUNCTION(pg_int4_t)
pgfn_int4div(kern_context *kcxt, pg_int4_t arg1, pg_int4_t arg2)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
						  "division by zero");
		}
		else if (arg2.value == -1)
		{
			result.value = -arg1.value;
			if (arg1.value != 0 && SAMESIGN(result.value, arg1.value))
			{
				result.isnull = true;
				STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
							  "integer out of range");
			}
		}
		else
			result.value = arg1.value / arg2.value;
	}
	return result;
}

DEVICE_FUNCTION(pg_int8_t)
pgfn_int48div(kern_context *kcxt, pg_int4_t arg1, pg_int8_t arg2)
{
	pg_int8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
                          "division by zero");
		}
		else
			result.value = (cl_long) arg1.value / arg2.value;
	}
	return result;
}

DEVICE_FUNCTION(pg_int8_t)
pgfn_int82div(kern_context *kcxt, pg_int8_t arg1, pg_int2_t arg2)
{
	pg_int8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
						  "division by zero");
		}
		else if (arg2.value == -1)
		{
			result.value = -arg1.value;
			if (arg1.value != 0 && SAMESIGN(result.value, arg1.value))
			{
				result.isnull = true;
				STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
							  "bigint out of range");
			}
		}
		else
			result.value = arg1.value / arg2.value;
	}
	return result;
}

DEVICE_FUNCTION(pg_int8_t)
pgfn_int84div(kern_context *kcxt, pg_int8_t arg1, pg_int4_t arg2)
{
	pg_int8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
						  "division by zero");
		}
		else if (arg2.value == -1)
		{
			result.value = -arg1.value;
			if (arg1.value != 0 && SAMESIGN(result.value, arg1.value))
			{
				result.isnull = true;
				STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
							  "bigint out of range");
			}
		}
		else
			result.value = arg1.value / arg2.value;
	}
	return result;
}

DEVICE_FUNCTION(pg_int8_t)
pgfn_int8div(kern_context *kcxt, pg_int8_t arg1, pg_int8_t arg2)
{
	pg_int8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
						  "division by zero");
		}
		else if (arg2.value == -1)
		{
			result.value = -arg1.value;
			if (arg1.value != 0 && SAMESIGN(result.value, arg1.value))
			{
				result.isnull = true;
				STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
							  "bigint out of range");
			}
		}
		else
			result.value = arg1.value / arg2.value;
	}
	return result;
}

DEVICE_FUNCTION(pg_float4_t)
pgfn_float2div(kern_context *kcxt, pg_float2_t arg1, pg_float2_t arg2)
{
	cl_float	value1 = arg1.value;
	cl_float	value2 = arg2.value;
	pg_float4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
    {
		if (value2 == 0.0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
						  "division by zero");
		}
		else
		{
			result.value = value1 / value2;
			CHECKFLOATVAL(&kcxt->e, result,
						  isinf(value1) || isinf(value2),
						  value1 == 0.0);
		}
	}
	return result;
}


DEVICE_FUNCTION(pg_float4_t)
pgfn_float24div(kern_context *kcxt, pg_float2_t arg1, pg_float4_t arg2)
{
	cl_float	value1 = arg1.value;
	pg_float4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
    {
		if (arg2.value == 0.0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
						  "division by zero");
		}
		else
		{
			result.value = value1 / arg2.value;
			CHECKFLOATVAL(&kcxt->e, result,
						  isinf(value1) || isinf(arg2.value),
						  value1 == 0.0);
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_float8_t)
pgfn_float28div(kern_context *kcxt, pg_float2_t arg1, pg_float8_t arg2)
{
	cl_double	value1 = arg1.value;
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
    {
		if (arg2.value == 0.0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
						  "division by zero");
		}
		else
		{
			result.value = value1 / arg2.value;
			CHECKFLOATVAL(&kcxt->e, result,
						  isinf(value1) || isinf(arg2.value),
						  value1 == 0.0);
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_float4_t)
pgfn_float42div(kern_context *kcxt, pg_float4_t arg1, pg_float2_t arg2)
{
	cl_float	value2 = arg2.value;
	pg_float4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
    {
		if (value2 == 0.0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
						  "division by zero");
		}
		else
		{
			result.value = arg1.value / value2;
			CHECKFLOATVAL(&kcxt->e, result,
						  isinf(arg1.value) || isinf(value2),
						  arg1.value == 0.0);
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_float4_t)
pgfn_float4div(kern_context *kcxt, pg_float4_t arg1, pg_float4_t arg2)
{
	pg_float4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
    {
		if (arg2.value == 0.0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
						  "division by zero");
		}
		else
		{
			result.value = arg1.value / arg2.value;
			CHECKFLOATVAL(&kcxt->e, result,
						  isinf(arg1.value) || isinf(arg2.value),
						  arg1.value == 0.0);
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_float8_t)
pgfn_float48div(kern_context *kcxt, pg_float4_t arg1, pg_float8_t arg2)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
    {
		if (arg2.value == 0.0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
						  "division by zero");
		}
		else
		{
			result.value = (cl_double)arg1.value / arg2.value;
			CHECKFLOATVAL(&kcxt->e, result,
						  isinf(arg1.value) || isinf(arg2.value),
						  arg1.value == 0.0);
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_float8_t)
pgfn_float82div(kern_context *kcxt, pg_float8_t arg1, pg_float2_t arg2)
{
	cl_double	value2 = arg2.value;
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
    {
		if (value2 == 0.0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
						  "division by zero");
		}
		else
		{
			result.value = arg1.value / value2;
			CHECKFLOATVAL(&kcxt->e, result,
						  isinf(arg1.value) || isinf(value2),
						  arg1.value == 0.0);
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_float8_t)
pgfn_float84div(kern_context *kcxt, pg_float8_t arg1, pg_float4_t arg2)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
    {
		if (arg2.value == 0.0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
						  "division by zero");
		}
		else
		{
			result.value = arg1.value / (cl_double)arg2.value;
			CHECKFLOATVAL(&kcxt->e, result,
						  isinf(arg1.value) || isinf(arg2.value),
						  arg1.value == 0.0);
		}
	}
	return result;
}

DEVICE_FUNCTION(pg_float8_t)
pgfn_float8div(kern_context *kcxt, pg_float8_t arg1, pg_float8_t arg2)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
    {
		if (arg2.value == 0.0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,
						  "division by zero");
		}
		else
		{
			result.value = arg1.value / arg2.value;
			CHECKFLOATVAL(&kcxt->e, result,
						  isinf(arg1.value) || isinf(arg2.value),
						  arg1.value == 0.0);
		}
	}
	return result;
}

/*
 * Functions for modulo operator on basic data types
 */
#define BASIC_INT_MODFUNC_TEMPLATE(name,d_type)						\
	DEVICE_FUNCTION(pg_##d_type##_t)								\
	pgfn_##name(kern_context *kcxt,									\
				pg_##d_type##_t arg1, pg_##d_type##_t arg2)			\
	{																\
		pg_##d_type##_t	result;										\
																	\
		result.isnull = arg1.isnull | arg2.isnull;					\
		if (!result.isnull)											\
		{															\
			if (arg2.value == 0)									\
			{														\
				result.isnull = true;								\
				STROM_EREPORT(kcxt, ERRCODE_DIVISION_BY_ZERO,		\
							  "division by zero");					\
			}														\
			else if (arg2.value == -1)								\
				result.value = 0;									\
			else													\
				result.value = arg1.value % arg2.value;				\
		}															\
		return result;												\
	}

BASIC_INT_MODFUNC_TEMPLATE(int2mod, int2)
BASIC_INT_MODFUNC_TEMPLATE(int4mod, int4)
BASIC_INT_MODFUNC_TEMPLATE(int8mod, int8)

#undef BASIC_INT_MODFUNC_TEMPLATE
