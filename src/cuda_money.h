/*
 * cuda_money.h
 *
 * Collection of currency functions for CUDA devices
 * --
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
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
#ifndef CUDA_MONEY_H
#define CUDA_MONEY_H
#ifdef __CUDACC__

/* pg_money_t */
#ifndef PG_MONEY_TYPE_DEFINED
#define PG_MONEY_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(money, cl_long);
STATIC_INLINE(Datum)
pg_money_to_datum(cl_long value)
{
	return pg_int8_to_datum(value);
}
#endif

/*
 * Cast function to currency data type
 */
#ifdef PG_NUMERIC_TYPE_DEFINED
STATIC_FUNCTION(pg_money_t)
pgfn_numeric_cash(kern_context *kcxt, pg_numeric_t arg1)
{

}
#endif

STATIC_FUNCTION(pg_money_t)
pgfn_int4_cash(kern_context *kcxt, pg_int4_t arg1)
{
	pg_money_t	result;

	if (arg1.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_long) arg1.value / (cl_long)PGLC_CURRENCY_SCALE;
	}
	return result;
}

STATIC_FUNCTION(pg_money_t)
pgfn_int8_cash(kern_context *kcxt, pg_int8_t arg1)
{
	pg_money_t	result;

	if (arg1.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (cl_long) arg1.value / (cl_long)PGLC_CURRENCY_SCALE;
	}
	return result;
}

/*
 * Currency operator functions
 */
STATIC_FUNCTION(pg_money_t)
pgfn_cash_pl(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2)
{
	pg_money_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = arg1.value + arg2.value;
	}
	return result;
}

STATIC_FUNCTION(pg_money_t)
pgfn_cash_mi(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2)
{
	pg_money_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = arg1.value + arg2.value;
	}
	return result;
}

STATIC_FUNCTION(pg_float8_t)
pgfn_cash_div_cash(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2)
{
	pg_float8_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		if (arg2.value == 0)
		{
			result.isnull = true;
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		}
		else
		{
			result.isnull = false;
			result.value = (cl_double)arg1.value / (cl_double)arg2.value;
		}
	}
	return result;
}

#define PGFN_MONEY_MULFUNC_TEMPLATE(name,d_type)					\
	STATIC_FUNCTION(pg_money_t)										\
	pgfn_cash_mul_##name(kern_context *kcxt,						\
						 pg_money_t arg1, pg_##d_type##_t arg2)		\
	{																\
		pg_money_t	result;											\
																	\
		if (arg1.isnull || arg2.isnull)								\
			result.isnull = true;									\
		else														\
		{															\
			result.isnull = false;									\
			result.value =  arg1.value * arg2.value;				\
		}															\
		return result;												\
	}

PGFN_MONEY_MULFUNC_TEMPLATE(int2, int2)
PGFN_MONEY_MULFUNC_TEMPLATE(int4, int4)
//PGFN_MONEY_MULFUNC_TEMPLATE(int8, int8)
PGFN_MONEY_MULFUNC_TEMPLATE(flt4, float4)
PGFN_MONEY_MULFUNC_TEMPLATE(flt8, float8)
#undef PGFN_MONEY_MULFUNC_TEMPLATE

#define PGFN_MONEY_DIVFUNC_TEMPLATE(name,d_type,zero)				\
	STATIC_FUNCTION(pg_money_t)										\
	pgfn_cash_div_##name(kern_context *kcxt,						\
						 pg_money_t arg1, pg_##d_type##_t arg2)		\
	{																\
		pg_money_t	result;											\
																	\
		if (arg1.isnull || arg2.isnull)								\
			result.isnull = true;									\
		else														\
		{															\
			if (arg2.value == (zero))								\
			{														\
				result.isnull = true;								\
				STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);	\
			}														\
			else													\
			{														\
				result.isnull = false;								\
				result.value =  arg1.value / arg2.value;			\
			}														\
		}															\
		return result;												\
	}

PGFN_MONEY_DIVFUNC_TEMPLATE(int2, int2, 0)
PGFN_MONEY_DIVFUNC_TEMPLATE(int4, int4, 0)
//PGFN_MONEY_DIVFUNC_TEMPLATE(int8, int8, 0)
PGFN_MONEY_DIVFUNC_TEMPLATE(flt4, float4, 0.0)
PGFN_MONEY_DIVFUNC_TEMPLATE(flt8, float8, 0.0)
#undef PGFN_MONEY_DIVFUNC_TEMPLATE

STATIC_FUNCTION(pg_money_t)
pgfn_int2_mul_cash(kern_context *kcxt, pg_int2_t arg1, pg_money_t arg2)
{
	return pgfn_cash_mul_int2(kcxt, arg2, arg1);
}

STATIC_FUNCTION(pg_money_t)
pgfn_int4_mul_cash(kern_context *kcxt, pg_int4_t arg1, pg_money_t arg2)
{
	return pgfn_cash_mul_int4(kcxt, arg2, arg1);
}

STATIC_FUNCTION(pg_money_t)
pgfn_flt4_mul_cash(kern_context *kcxt, pg_float4_t arg1, pg_money_t arg2)
{
	return pgfn_cash_mul_flt4(kcxt, arg2, arg1);
}

STATIC_FUNCTION(pg_money_t)
pgfn_flt8_mul_cash(kern_context *kcxt, pg_float8_t arg1, pg_money_t arg2)
{
	return pgfn_cash_mul_flt8(kcxt, arg2, arg1);
}

/*
 * Currency comparison functions
 */
STATIC_FUNCTION(pg_int4_t)
pgfn_cash_cmp(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2)
{
	pg_int4_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (arg1.value > arg2.value ? 1 :
						arg1.value < arg2.value ? -1 : 0);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_cash_eq(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (arg1.value == arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_cash_ne(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (arg1.value != arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_cash_lt(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (arg1.value < arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_cash_le(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (arg1.value <= arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_cash_gt(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (arg1.value > arg2.value);
	}
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_cash_ge(kern_context *kcxt, pg_money_t arg1, pg_money_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull || arg2.isnull)
		result.isnull = true;
	else
	{
		result.isnull = false;
		result.value = (arg1.value >= arg2.value);
	}
	return result;
}

#else	/* __CUDACC__ */
#include "utils/pg_locale.h"

STATIC_INLINE(void)
assign_moneylib_session_info(StringInfo buf)
{
	struct lconv *lconvert = PGLC_localeconv();
	cl_int		fpoint;
	cl_long		scale;
	cl_int		i;

	/* see comments about frac_digits in cash_in() */
	fpoint = lconvert->frac_digits;
	if (fpoint < 0 || fpoint > 10)
		fpoint = 2;

	/* compute required scale factor */
	scale = 1;
	for (i=0; i < fpoint; i++)
		scale *= 10;

	appendStringInfo(
		buf,
		"#ifdef __CUDACC__\n"
		"/* ================================================\n"
		" * session information for cuda_money.h\n"
		" * ================================================ */\n"
		"\n"
		"#define PGLC_CURRENCY_SCALE_LOG10  %d\n"
		"#define PGLC_CURRENCY_SCALE        %ld\n"
		"\n"
		"#endif /* __CUDACC__ */\n",
		fpoint,
		scale);
}
#endif	/* __CUDACC__ */
#endif	/* CUDA_MONEY_H */
