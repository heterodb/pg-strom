/*
 * opencl_math.h
 *
 * Collection of math functions for OpenCL devices
 * --
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
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
#ifndef CUDA_MATH_H
#define CUDA_MATH_H
#ifdef __CUDACC__

/*
 * Utility macros
 */
#define CHECKFLOATVAL(errcode, result, inf_is_valid, zero_is_valid)	\
	do {															\
		if ((isinf((result).value) && !(inf_is_valid)) ||			\
			((result).value == 0.0 && !(zero_is_valid)))			\
		{															\
			(result).isnull = true;									\
			STROM_SET_ERROR((errcode), StromError_CpuReCheck);		\
		}															\
	} while(0)

#define SAMESIGN(a,b)	(((a) < 0) == ((b) < 0))

/*
 * Functions for addition operator on basic data types
 */
#define BASIC_INT_ADDFUNC_TEMPLATE(name,r_type,x_type,y_type)		\
	STATIC_FUNCTION(pg_##r_type##_t)								\
	pgfn_##name(cl_int *errcode,									\
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
				STROM_SET_ERROR(errcode, StromError_CpuReCheck);	\
			}														\
		}															\
		return result;												\
	}

#define BASIC_FLOAT_ADDFUNC_TEMPLATE(name,r_type,x_type,y_type)		\
	STATIC_FUNCTION(pg_##r_type##_t)								\
	pgfn_##name(cl_int *errcode,									\
				pg_##x_type##_t arg1, pg_##y_type##_t arg2)         \
    {																\
		pg_##r_type##_t	result;										\
																	\
		result.isnull = arg1.isnull | arg2.isnull;					\
		if (!result.isnull)											\
		{															\
			result.value = arg1.value + arg2.value;					\
			CHECKFLOATVAL(errcode, result,							\
						  isinf(arg1.value) ||						\
						  isinf(arg2.value), true);					\
		}                                                           \
		return result;												\
	}

BASIC_INT_ADDFUNC_TEMPLATE(int2pl, int2,int2,int2)
BASIC_INT_ADDFUNC_TEMPLATE(int24pl,int4,int4,int2)
BASIC_INT_ADDFUNC_TEMPLATE(int28pl,int8,int8,int2)

BASIC_INT_ADDFUNC_TEMPLATE(int42pl,int4,int4,int2)
BASIC_INT_ADDFUNC_TEMPLATE(int4pl, int4,int4,int4)
BASIC_INT_ADDFUNC_TEMPLATE(int48pl,int4,int4,int8)

BASIC_INT_ADDFUNC_TEMPLATE(int82pl,int8,int8,int2)
BASIC_INT_ADDFUNC_TEMPLATE(int84pl,int8,int8,int4)
BASIC_INT_ADDFUNC_TEMPLATE(int8pl, int8,int8,int8)

BASIC_FLOAT_ADDFUNC_TEMPLATE(float4pl, float4, float4, float4)
BASIC_FLOAT_ADDFUNC_TEMPLATE(float48pl,float8, float4, float8)
BASIC_FLOAT_ADDFUNC_TEMPLATE(float84pl,float8, float8, float4)
BASIC_FLOAT_ADDFUNC_TEMPLATE(float8pl, float8, float8, float8)

#undef BASIC_INT_ADDFUNC_TEMPLATE
#undef BASIC_FLOAT_ADDFUNC_TEMPLATE

/*
 * Functions for addition operator on basic data types
 */
#define BASIC_INT_SUBFUNC_TEMPLATE(name,r_type,x_type,y_type)		\
	STATIC_FUNCTION(pg_##r_type##_t)								\
	pgfn_##name(cl_int *errcode,									\
				pg_##x_type##_t arg1, pg_##y_type##_t arg2)			\
	{																\
		pg_##r_type##_t	result;										\
																	\
		result.isnull = arg1.isnull | arg2.isnull;					\
		if (!result.isnull)											\
		{															\
			result.value = arg1.value - arg2.value;					\
			if (SAMESIGN(arg1.value, arg2.value) &&					\
				!SAMESIGN(result.value, arg1.value))				\
			{														\
				result.isnull = true;								\
				STROM_SET_ERROR(errcode, StromError_CpuReCheck);	\
			}														\
		}															\
		return result;												\
	}

#define BASIC_FLOAT_SUBFUNC_TEMPLATE(name,r_type,x_type,y_type)		\
	STATIC_FUNCTION(pg_##r_type##_t)								\
	pgfn_##name(cl_int *errcode,									\
				pg_##x_type##_t arg1, pg_##y_type##_t arg2)         \
    {																\
		pg_##r_type##_t	result;										\
																	\
		result.isnull = arg1.isnull | arg2.isnull;					\
		if (!result.isnull)											\
		{															\
			result.value = arg1.value - arg2.value;					\
			CHECKFLOATVAL(errcode, result,							\
						  isinf(arg1.value) ||						\
						  isinf(arg2.value), true);					\
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

BASIC_FLOAT_SUBFUNC_TEMPLATE(float4mi,  float4, float4, float4)
BASIC_FLOAT_SUBFUNC_TEMPLATE(float48mi, float8, float4, float8)
BASIC_FLOAT_SUBFUNC_TEMPLATE(float84mi, float8, float8, float4)
BASIC_FLOAT_SUBFUNC_TEMPLATE(float8mi,  float8, float8, float8)

#undef BASIC_INT_SUBFUNC_TEMPLATE
#undef BASIC_FLOAT_SUBFUNC_TEMPLATE


/*
 * Functions for multiplication operator on basic data types
 */
STATIC_FUNCTION(pg_int2_t)
pgfn_int2mul(cl_int *errcode, pg_int2_t arg1, pg_int2_t arg2)
{
	pg_int2_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		cl_int	temp = (cl_int)arg1.value * (cl_int)arg2.value;

		if (temp < SHRT_MIN || temp > SHRT_MAX)
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
		else
			result.value = (cl_short) temp;
	}
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_int24mul(cl_int *errcode, pg_int2_t arg1, pg_int4_t arg2)
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
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
	}
	return result;
}

STATIC_FUNCTION(pg_int8_t)
pgfn_int28mul(cl_int *errcode, pg_int2_t arg1, pg_int8_t arg2)
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
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
	}
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_int42mul(cl_int *errcode, pg_int4_t arg1, pg_int2_t arg2)
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
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
	}
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_int4mul(cl_int *errcode, pg_int4_t arg1, pg_int4_t arg2)
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
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
	}
	return result;
}

STATIC_FUNCTION(pg_int8_t)
pgfn_int48mul(cl_int *errcode, pg_int4_t arg1, pg_int8_t arg2)
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
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
	}
	return result;
}

STATIC_FUNCTION(pg_int8_t)
pgfn_int82mul(cl_int *errcode, pg_int8_t arg1, pg_int2_t arg2)
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
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
	}
	return result;
}

STATIC_FUNCTION(pg_int8_t)
pgfn_int84mul(cl_int *errcode, pg_int8_t arg1, pg_int4_t arg2)
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
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
	}
	return result;
}

STATIC_FUNCTION(pg_int8_t)
pgfn_int8mul(cl_int *errcode, pg_int8_t arg1, pg_int8_t arg2)
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
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
	}
	return result;
}

STATIC_FUNCTION(pg_float4_t)
pgfn_float4mul(cl_int *errcode, pg_float4_t arg1, pg_float4_t arg2)
{
	pg_float4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * arg2.value;
		CHECKFLOATVAL(errcode, result,
					  isinf(arg1.value) || isinf(arg2.value),
					  arg1.value == 0.0 || arg2.value == 0.0);
	}
	return result;
}

STATIC_FUNCTION(pg_float8_t)
pgfn_float48mul(cl_int *errcode, pg_float4_t arg1, pg_float8_t arg2)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * arg2.value;
		CHECKFLOATVAL(errcode, result,
					  isinf(arg1.value) || isinf(arg2.value),
					  arg1.value == 0.0 || arg2.value == 0.0);
	}
	return result;
}

STATIC_FUNCTION(pg_float8_t)
pgfn_float84mul(cl_int *errcode, pg_float8_t arg1, pg_float4_t arg2)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * arg2.value;
		CHECKFLOATVAL(errcode, result,
					  isinf(arg1.value) || isinf(arg2.value),
					  arg1.value == 0.0 || arg2.value == 0.0);
	}
	return result;
}

STATIC_FUNCTION(pg_float8_t)
pgfn_float8mul(cl_int *errcode, pg_float8_t arg1, pg_float8_t arg2)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * arg2.value;
		CHECKFLOATVAL(errcode, result,
					  isinf(arg1.value) || isinf(arg2.value),
					  arg1.value == 0.0 || arg2.value == 0.0);
	}
	return result;
}

/*
 * Functions for division operator on basic data types
 */
#define SAMESIGN(a,b)	(((a) < 0) == ((b) < 0))

STATIC_FUNCTION(pg_int2_t)
pgfn_int2div(cl_int *errcode, pg_int2_t arg1, pg_int2_t arg2)
{
	pg_int2_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
		else if (arg2.value == -1)
		{
			result.value = -arg1.value;
			if (arg1.value != 0 && SAMESIGN(result.value, arg1.value))
			{
				result.isnull = true;
				STROM_SET_ERROR(errcode, StromError_CpuReCheck);
			}
		}
		else
			result.value = arg1.value / arg2.value;
	}
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_int24div(cl_int *errcode, pg_int2_t arg1, pg_int4_t arg2)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
            STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
		else
			result.value = (cl_int) arg1.value / arg2.value;
	}
	return result;
}

STATIC_FUNCTION(pg_int8_t)
pgfn_int28div(cl_int *errcode, pg_int2_t arg1, pg_int8_t arg2)
{
	pg_int8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
            STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
		else
			result.value = (cl_long) arg1.value / arg2.value;
	}
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_int42div(cl_int *errcode, pg_int4_t arg1, pg_int2_t arg2)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
    {
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
		else if (arg2.value == -1)
		{
			result.value = -arg1.value;
			if (arg1.value != 0 && SAMESIGN(result.value, arg1.value))
			{
				result.isnull = true;
				STROM_SET_ERROR(errcode, StromError_CpuReCheck);
			}
		}
		else
			result.value = arg1.value / arg2.value;
	}
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_int4div(cl_int *errcode, pg_int4_t arg1, pg_int4_t arg2)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
		else if (arg2.value == -1)
		{
			result.value = -arg1.value;
			if (arg1.value != 0 && SAMESIGN(result.value, arg1.value))
			{
				result.isnull = true;
				STROM_SET_ERROR(errcode, StromError_CpuReCheck);
			}
		}
		else
			result.value = arg1.value / arg2.value;
	}
	return result;
}

STATIC_FUNCTION(pg_int8_t)
pgfn_int48div(cl_int *errcode, pg_int4_t arg1, pg_int8_t arg2)
{
	pg_int8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
		else
			result.value = (cl_long) arg1.value / arg2.value;
	}
	return result;
}

STATIC_FUNCTION(pg_int8_t)
pgfn_int82div(cl_int *errcode, pg_int8_t arg1, pg_int2_t arg2)
{
	pg_int8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
		else if (arg2.value == -1)
		{
			result.value = -arg1.value;
			if (arg1.value != 0 && SAMESIGN(result.value, arg1.value))
			{
				result.isnull = true;
				STROM_SET_ERROR(errcode, StromError_CpuReCheck);
			}
		}
		else
			result.value = arg1.value / arg2.value;
	}
	return result;
}

STATIC_FUNCTION(pg_int8_t)
pgfn_int84div(cl_int *errcode, pg_int8_t arg1, pg_int4_t arg2)
{
	pg_int8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
		else if (arg2.value == -1)
		{
			result.value = -arg1.value;
			if (arg1.value != 0 && SAMESIGN(result.value, arg1.value))
			{
				result.isnull = true;
				STROM_SET_ERROR(errcode, StromError_CpuReCheck);
			}
		}
		else
			result.value = arg1.value / arg2.value;
	}
	return result;
}

STATIC_FUNCTION(pg_int8_t)
pgfn_int8div(cl_int *errcode, pg_int8_t arg1, pg_int8_t arg2)
{
	pg_int8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
		else if (arg2.value == -1)
		{
			result.value = -arg1.value;
			if (arg1.value != 0 && SAMESIGN(result.value, arg1.value))
			{
				result.isnull = true;
				STROM_SET_ERROR(errcode, StromError_CpuReCheck);
			}
		}
		else
			result.value = arg1.value / arg2.value;
	}
	return result;
}

STATIC_FUNCTION(pg_float4_t)
pgfn_float4div(cl_int *errcode, pg_float4_t arg1, pg_float4_t arg2)
{
	pg_float4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
    {
		if (arg2.value == 0.0)
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
		else
		{
			result.value = arg1.value / arg2.value;
			CHECKFLOATVAL(errcode, result,
						  isinf(arg1.value) || isinf(arg2.value),
						  arg1.value == 0.0);
		}
	}
	return result;
}

STATIC_FUNCTION(pg_float8_t)
pgfn_float48div(cl_int *errcode, pg_float4_t arg1, pg_float8_t arg2)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
    {
		if (arg2.value == 0.0)
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
		else
		{
			result.value = arg1.value / arg2.value;
			CHECKFLOATVAL(errcode, result,
						  isinf(arg1.value) || isinf(arg2.value),
						  arg1.value == 0.0);
		}
	}
	return result;
}

STATIC_FUNCTION(pg_float8_t)
pgfn_float84div(cl_int *errcode, pg_float8_t arg1, pg_float4_t arg2)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
    {
		if (arg2.value == 0.0)
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
		else
		{
			result.value = arg1.value / arg2.value;
			CHECKFLOATVAL(errcode, result,
						  isinf(arg1.value) || isinf(arg2.value),
						  arg1.value == 0.0);
		}
	}
	return result;
}

STATIC_FUNCTION(pg_float8_t)
pgfn_float8div(cl_int *errcode, pg_float8_t arg1, pg_float8_t arg2)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
    if (!result.isnull)
    {
		if (arg2.value == 0.0)
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
		else
		{
			result.value = arg1.value / arg2.value;
			CHECKFLOATVAL(errcode, result,
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
	STATIC_FUNCTION(pg_##d_type##_t)								\
	pgfn_##name(cl_int *errcode,									\
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
				STROM_SET_ERROR(errcode, StromError_CpuReCheck);	\
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

/*
 * Misc mathematic functions
 */
STATIC_FUNCTION(pg_float8_t)
pgfn_dsqrt(cl_int *errcode, pg_float8_t arg1)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull;
	if (!arg1.isnull)
	{
		if (arg1.value < 0.0)
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
		else
		{
			result.value = sqrt(arg1.value);
			CHECKFLOATVAL(errcode, result,
						  isinf(arg1.value), arg1.value == 0);
		}
	}
	return result;
}

STATIC_FUNCTION(pg_float8_t)
pgfn_dpow(cl_int *errcode, pg_float8_t arg1, pg_float8_t arg2)
{
	pg_float8_t	result;

	if ((arg1.value == 0.0 && arg2.value < 0.0) ||
		(arg1.value < 0.0 && floor(arg2.value) != arg2.value))
	{
		result.isnull = true;
		STROM_SET_ERROR(errcode, StromError_CpuReCheck);
	}
	else
	{
		result.isnull = false;
		result.value = pow(arg1.value, arg2.value);
	}
	// TODO: needs to investigate which value shall be returned when
	// NVIDIA platform accept very large number
	return result;
}

STATIC_FUNCTION(pg_float8_t)
pgfn_dpi(cl_int *errcode)
{
	pg_float8_t	result;

	result.isnull = false;
	result.value = 3.141592653589793115998;

	return result;
}


#endif	/* __CUDACC__ */
#endif	/* CUDA_MATH_H */
