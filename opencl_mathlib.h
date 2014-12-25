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
#ifndef OPENCL_MATH_H
#define OPENCL_MATH_H
#ifdef OPENCL_DEVICE_CODE

/*
 * Utility macros
 */
#define CHECKFLOATVAL(val, inf_is_valid, zero_is_valid)         \
	((isinf(val) && !(inf_is_valid)) ||							\
	 (val) == 0.0 && !(zero_is_valid))

#define SAMESIGN(a,b)	(((a) < 0) == ((b) < 0))

/*
 * Functions for multiplication operator on basic data types
 */
static inline pg_int2_t
pgfn_int2mul(__private cl_int *errcode, pg_int2_t arg1, pg_int2_t arg2)
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

static inline pg_int4_t
pgfn_int24mul(__private cl_int *errcode, pg_int2_t arg1, pg_int4_t arg2)
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

static inline pg_int8_t
pgfn_int28mul(__private cl_int *errcode, pg_int2_t arg1, pg_int8_t arg2)
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

static inline pg_int4_t
pgfn_int42mul(__private cl_int *errcode, pg_int4_t arg1, pg_int2_t arg2)
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

static inline pg_int4_t
pgfn_int4mul(__private cl_int *errcode, pg_int4_t arg1, pg_int4_t arg2)
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

static inline pg_int8_t
pgfn_int48mul(__private cl_int *errcode, pg_int4_t arg1, pg_int8_t arg2)
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

static inline pg_int8_t
pgfn_int82mul(__private cl_int *errcode, pg_int8_t arg1, pg_int2_t arg2)
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

static inline pg_int8_t
pgfn_int84mul(__private cl_int *errcode, pg_int8_t arg1, pg_int4_t arg2)
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

static inline pg_int8_t
pgfn_int8mul(__private cl_int *errcode, pg_int8_t arg1, pg_int8_t arg2)
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

static inline pg_float4_t
pgfn_float4mul(__private cl_int *errcode, pg_float4_t arg1, pg_float4_t arg2)
{
	pg_float4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * arg2.value;
		/* logic copied from CHECKFLOATVAL */
		if (CHECKFLOATVAL(result.value,
						  isinf(arg1.value) || isinf(arg2.value),
						  arg1.value == 0.0 || arg2.value == 0.0))
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
	}
	return result;
}

static inline pg_float8_t
pgfn_float48mul(__private cl_int *errcode, pg_float4_t arg1, pg_float8_t arg2)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * arg2.value;
		/* logic copied from CHECKFLOATVAL */
		if (CHECKFLOATVAL(result.value,
						  isinf(arg1.value) || isinf(arg2.value),
						  arg1.value == 0.0 || arg2.value == 0.0))
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
	}
	return result;
}

static inline pg_float8_t
pgfn_float84mul(__private cl_int *errcode, pg_float8_t arg1, pg_float4_t arg2)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * arg2.value;
		/* logic copied from CHECKFLOATVAL */
		if (CHECKFLOATVAL(result.value,
						  isinf(arg1.value) || isinf(arg2.value),
						  arg1.value == 0.0 || arg2.value == 0.0))
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
	}
	return result;
}

static inline pg_float8_t
pgfn_float8mul(__private cl_int *errcode, pg_float8_t arg1, pg_float8_t arg2)
{
	pg_float8_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		result.value = arg1.value * arg2.value;
		/* logic copied from CHECKFLOATVAL */
		if (CHECKFLOATVAL(result.value,
						  isinf(arg1.value) || isinf(arg2.value),
						  arg1.value == 0.0 || arg2.value == 0.0))
		{
			result.isnull = true;
			STROM_SET_ERROR(errcode, StromError_CpuReCheck);
		}
	}
	return result;
}

/*
 * Functions for division operator on basic data types
 */
#define SAMESIGN(a,b)	(((a) < 0) == ((b) < 0))

static inline pg_int2_t
pgfn_int2div(__private cl_int *errcode, pg_int2_t arg1, pg_int2_t arg2)
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

static inline pg_int4_t
pgfn_int24div(__private cl_int *errcode, pg_int2_t arg1, pg_int4_t arg2)
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

static inline pg_int8_t
pgfn_int28div(__private cl_int *errcode, pg_int2_t arg1, pg_int8_t arg2)
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

static inline pg_int4_t
pgfn_int42div(__private cl_int *errcode, pg_int4_t arg1, pg_int2_t arg2)
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

static inline pg_int4_t
pgfn_int4div(__private cl_int *errcode, pg_int4_t arg1, pg_int4_t arg2)
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

static inline pg_int8_t
pgfn_int48div(__private cl_int *errcode, pg_int4_t arg1, pg_int8_t arg2)
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

static inline pg_int8_t
pgfn_int82div(__private cl_int *errcode, pg_int8_t arg1, pg_int2_t arg2)
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

static inline pg_int8_t
pgfn_int84div(__private cl_int *errcode, pg_int8_t arg1, pg_int4_t arg2)
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

static inline pg_int8_t
pgfn_int8div(__private cl_int *errcode, pg_int8_t arg1, pg_int8_t arg2)
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

static inline pg_float4_t
pgfn_float4div(__private cl_int *errcode, pg_float4_t arg1, pg_float4_t arg2)
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
			if (CHECKFLOATVAL(result.value,
							  isinf(arg1.value) || isinf(arg2.value),
							  arg1.value == 0.0))
			{
				result.isnull = true;
				STROM_SET_ERROR(errcode, StromError_CpuReCheck);
			}
		}
	}
	return result;
}

static inline pg_float8_t
pgfn_float48div(__private cl_int *errcode, pg_float4_t arg1, pg_float8_t arg2)
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
			if (CHECKFLOATVAL(result.value,
							  isinf(arg1.value) || isinf(arg2.value),
							  arg1.value == 0.0))
			{
				result.isnull = true;
				STROM_SET_ERROR(errcode, StromError_CpuReCheck);
			}
		}
	}
	return result;
}

static inline pg_float8_t
pgfn_float84div(__private cl_int *errcode, pg_float8_t arg1, pg_float4_t arg2)
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
			if (CHECKFLOATVAL(result.value,
							  isinf(arg1.value) || isinf(arg2.value),
							  arg1.value == 0.0))
			{
				result.isnull = true;
				STROM_SET_ERROR(errcode, StromError_CpuReCheck);
			}
		}
	}
	return result;
}

static inline pg_float8_t
pgfn_float8div(__private cl_int *errcode, pg_float8_t arg1, pg_float8_t arg2)
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
			if (CHECKFLOATVAL(result.value,
							  isinf(arg1.value) || isinf(arg2.value),
							  arg1.value == 0.0))
			{
				result.isnull = true;
				STROM_SET_ERROR(errcode, StromError_CpuReCheck);
			}
		}
	}
	return result;
}

#endif	/* OPENCL_DEVICE_CODE */
#endif	/* OPENCL_MATH_H */
