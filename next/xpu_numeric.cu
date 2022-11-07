/*
 * xpu_numeric.cu
 *
 * collection of numeric type support for both of GPU and DPU
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "xpu_common.h"
#include <math.h>

INLINE_FUNCTION(int)
xpu_numeric_sign(xpu_numeric_t *num)
{
	if (num->kind != XPU_NUMERIC_KIND__VALID)
	{
		Assert(num->kind != XPU_NUMERIC_KIND__NAN);
		/* Must be +Inf or -Inf */
		return num->kind == XPU_NUMERIC_KIND__POS_INF ? INT_MAX : INT_MIN;
	}
	if (num->value > 0)
		return 1;
	if (num->value < 0)
		return -1;
	return 0;
}

INLINE_FUNCTION(void)
set_normalized_numeric(xpu_numeric_t *result, int128_t value, int16_t weight)
{
	if (value == 0)
		weight = 0;
	else
	{
		while (value % 10 == 0)
		{
			value /= 10;
			weight--;
		}
	}
	result->isnull = false;
	result->kind = XPU_NUMERIC_KIND__VALID;
	result->weight = weight;
	result->value = value;
}

STATIC_FUNCTION(bool)
__numeric_from_varlena(kern_context *kcxt,
					   xpu_numeric_t *result,
					   const varlena *addr)
{
	uint32_t		len;

	len = VARSIZE_ANY_EXHDR(addr);
	if (len >= sizeof(uint16_t))
	{
		NumericChoice *nc = (NumericChoice *)VARDATA_ANY(addr);
		uint16_t	n_head = __Fetch(&nc->n_header);

		/* special case if NaN, +/-Inf */
		if (NUMERIC_IS_SPECIAL(n_head))
		{
			if (NUMERIC_IS_NAN(n_head))
				result->kind = XPU_NUMERIC_KIND__NAN;
			else if (NUMERIC_IS_PINF(n_head))
				result->kind = XPU_NUMERIC_KIND__POS_INF;
			else if (NUMERIC_IS_NINF(n_head))
				result->kind = XPU_NUMERIC_KIND__NEG_INF;
			else
				goto error;

			result->weight = 0;
			result->value = 0;
		}
		else
		{
			NumericDigit *digits = NUMERIC_DIGITS(nc, n_head);
			int			weight  = NUMERIC_WEIGHT(nc, n_head) + 1;
			int			i, ndigits = NUMERIC_NDIGITS(n_head, len);
			int128_t	value = 0;

			for (i=0; i < ndigits; i++)
			{
				NumericDigit dig = __Fetch(&digits[i]);

				/*
				 * Rough overflow check - PG_NBASE is 10000, therefore,
				 * we never touch the upper limit as long as the value's
				 * significant 14bits are all zero.
				 */
				if ((value >> 114) != 0)
				{
					STROM_ELOG(kcxt, "numeric value is out of range");
					return false;
				}
				value = value * PG_NBASE + dig;
			}
			if (NUMERIC_SIGN(n_head) == NUMERIC_NEG)
				value = -value;
			weight = PG_DEC_DIGITS * (ndigits - weight);

			set_normalized_numeric(result, value, weight);
		}
		return true;
	}
error:
	STROM_ELOG(kcxt, "corrupted numeric header");
	return false;
}

STATIC_FUNCTION(int)
__numeric_to_varlena(char *buffer, int16_t weight, int128_t value)
{
	NumericData	   *numData = (NumericData *)buffer;
	NumericLong	   *numBody = &numData->choice.n_long;
	NumericDigit	n_data[PG_MAX_DATA];
	int				ndigits;
	int				len;
	uint16_t		n_header = (Max(weight, 0) & NUMERIC_DSCALE_MASK);
	bool			is_negative = (value < 0);

	if (is_negative)
		value = -value;
	switch (weight % PG_DEC_DIGITS)
	{
		case 3:
		case -1:
			value *= 10;
			weight += 1;
			break;
		case 2:
		case -2:
			value *= 100;
			weight += 2;
			break;
		case 1:
		case -3:
			value *= 1000;
			weight += 3;
			break;
		default:
			/* ok */
			break;
	}
	Assert(weight % PG_DEC_DIGITS == 0);

	ndigits = 0;
	while (value != 0)
    {
		int		mod;

		mod = (value % PG_NBASE);
		value /= PG_NBASE;
		Assert(ndigits < PG_MAX_DATA);
		ndigits++;
		n_data[PG_MAX_DATA - ndigits] = mod;
	}
	len = (offsetof(NumericData, choice.n_long.n_data)
		   + sizeof(NumericDigit) * ndigits);
	if (buffer)
	{
		memcpy(numBody->n_data,
			   n_data + PG_MAX_DATA - ndigits,
			   sizeof(NumericDigit) * ndigits);
		if (is_negative)
			n_header |= NUMERIC_NEG;
		numBody->n_sign_dscale = n_header;
		numBody->n_weight = ndigits - (weight / PG_DEC_DIGITS) - 1;

		SET_VARSIZE(numData, len);
	}
	return len;
}

STATIC_FUNCTION(bool)
xpu_numeric_datum_ref(kern_context *kcxt,
					  xpu_datum_t *__result,
					  const void *addr)
{
	xpu_numeric_t  *result = (xpu_numeric_t *)__result;

	if (!addr)
		result->isnull = true;
	else if (!__numeric_from_varlena(kcxt, result,
									 (const varlena *)addr))
	{
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_numeric_arrow_ref(kern_context *kcxt,
					  xpu_datum_t *__result,
					  const kern_colmeta *cmeta,
					  const void *addr, int len)
{
	xpu_numeric_t  *result = (xpu_numeric_t *)__result;

	if (cmeta->attopts.common.tag != ArrowType__Decimal ||
		cmeta->attopts.decimal.bitWidth != 128)
	{
		STROM_ELOG(kcxt, "Not a convertible Arrow::Decimal value");
		return false;
	}
	if (!addr)
		result->isnull = true;
	else
		set_normalized_numeric(result, *((int128_t *)addr),
							   cmeta->attopts.decimal.scale);
	return true;
}

STATIC_FUNCTION(int)
xpu_numeric_arrow_move(kern_context *kcxt,
					   char *buffer,
					   const kern_colmeta *cmeta,
					   const void *addr, int len)
{
	if (cmeta->attopts.common.tag != ArrowType__Decimal ||
		cmeta->attopts.decimal.bitWidth != 128)
	{
		STROM_ELOG(kcxt, "Not a convertible Arrow::Decimal value");
		return -1;
	}
	if (!addr)
		return 0;
	return __numeric_to_varlena(buffer,
								cmeta->attopts.decimal.scale,
								*((int128_t *)addr));
}

PUBLIC_FUNCTION(int)
xpu_numeric_datum_store(kern_context *kcxt,
						char *buffer,
						const xpu_datum_t *__arg)
{
	const xpu_numeric_t *arg = (const xpu_numeric_t *)__arg;

	if (arg->isnull)
		return 0;
	if (arg->kind != XPU_NUMERIC_KIND__VALID)
	{
		int		sz = offsetof(NumericData, choice.n_header) + sizeof(uint16_t);

		if (buffer)
		{
			NumericChoice *nc = (NumericChoice *)buffer;

			if (arg->kind == XPU_NUMERIC_KIND__POS_INF)
				nc->n_header = NUMERIC_PINF;
			else if (arg->kind == XPU_NUMERIC_KIND__NEG_INF)
				nc->n_header = NUMERIC_NINF;
			else
				nc->n_header = NUMERIC_NAN;
			SET_VARSIZE(nc, sz);
		}
		return sz;
	}
	return __numeric_to_varlena(buffer, arg->weight, arg->value);
}

PUBLIC_FUNCTION(bool)
xpu_numeric_datum_hash(kern_context *kcxt,
					   uint32_t *p_hash,
					   const xpu_datum_t *__arg)
{
	const xpu_numeric_t *arg = (const xpu_numeric_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else if (arg->kind != XPU_NUMERIC_KIND__VALID)
		*p_hash = pg_hash_any(&arg->kind, sizeof(uint8_t));
	else
		*p_hash = (pg_hash_any(&arg->weight, sizeof(int16_t)) ^
				   pg_hash_any(&arg->value, sizeof(int128_t)));
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(numeric, false, 4, -1);

#define PG_NUMERIC_TO_INT_TEMPLATE(TARGET,MIN_VALUE,MAX_VALUE)		\
	PUBLIC_FUNCTION(bool)											\
	pgfn_numeric_to_##TARGET(XPU_PGFUNCTION_ARGS)					\
	{																\
		xpu_##TARGET##_t *result = (xpu_##TARGET##_t *)__result;	\
		xpu_numeric_t	datum;										\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);			\
																	\
		assert(kexp->nr_args == 1 &&								\
			   KEXP_IS_VALID(karg, numeric));						\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum))				\
			return false;											\
		result->isnull = datum.isnull;								\
		if (result->isnull)											\
			return true;											\
		if (datum.kind == XPU_NUMERIC_KIND__VALID)					\
		{															\
			int128_t	ival = datum.value;							\
			int16_t		weight = datum.weight;						\
																	\
			if (ival != 0)											\
			{														\
				while (weight > 0)									\
				{													\
					/* round of 0.x digit */						\
					if (weight == 1)								\
						ival += (ival > 0 ? 5 : -5);				\
					ival /= 10;										\
					weight--;										\
				}													\
				while (weight < 0)									\
				{													\
					ival *= 10;										\
					weight++;										\
					if (ival < MIN_VALUE || ival > MAX_VALUE)		\
						break;										\
				}													\
				if (ival < MIN_VALUE || ival > MAX_VALUE)			\
				{													\
					STROM_ELOG(kcxt, "integer out of range");		\
					return false;									\
				}													\
			}														\
			result->value = ival;									\
			return true;											\
		}															\
		STROM_ELOG(kcxt, "cannot convert NaN/Inf to integer");		\
		return false;												\
	}
PG_NUMERIC_TO_INT_TEMPLATE(int1,SCHAR_MIN,SCHAR_MAX)
PG_NUMERIC_TO_INT_TEMPLATE(int2,SHRT_MIN,SHRT_MAX)
PG_NUMERIC_TO_INT_TEMPLATE(int4,INT_MIN,INT_MAX)
PG_NUMERIC_TO_INT_TEMPLATE(int8,LLONG_MIN,LLONG_MAX)
PG_NUMERIC_TO_INT_TEMPLATE(money,LLONG_MIN,LLONG_MAX)

#define PG_NUMERIC_TO_FLOAT_TEMPLATE(TARGET,__TEMP,__CAST)			\
	PUBLIC_FUNCTION(bool)											\
	pgfn_numeric_to_##TARGET(XPU_PGFUNCTION_ARGS)					\
	{																\
		xpu_##TARGET##_t *result = (xpu_##TARGET##_t *)__result;	\
		xpu_numeric_t	datum;										\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);			\
																	\
		assert(kexp->nr_args == 1 &&								\
			   KEXP_IS_VALID(karg, numeric));						\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum))				\
			return false;											\
		result->isnull = datum.isnull;								\
		if (result->isnull)											\
			return true;											\
		if (datum.kind == XPU_NUMERIC_KIND__VALID)					\
		{															\
			__TEMP		fval = datum.value;							\
			int16_t		weight = datum.weight;						\
																	\
			if (fval != 0.0)										\
			{														\
				while (weight > 0)									\
				{													\
					fval *= 10.0;									\
					weight--;										\
				}													\
				while (weight < 0)									\
				{													\
					fval /= 10.0;									\
					weight++;										\
				}													\
				if (isnan(fval) || isinf(fval))						\
				{													\
					STROM_ELOG(kcxt,"integer out of range");		\
					return false;									\
				}													\
			}														\
			result->value = __CAST(fval);							\
		}															\
		else if (datum.kind == XPU_NUMERIC_KIND__POS_INF)			\
			result->value = __CAST(INFINITY);						\
		else if (datum.kind == XPU_NUMERIC_KIND__NEG_INF)			\
			result->value = __CAST(-INFINITY);						\
		else														\
			result->value = __CAST(NAN);							\
		return true;												\
	}
PG_NUMERIC_TO_FLOAT_TEMPLATE(float2, float,__to_fp16)
PG_NUMERIC_TO_FLOAT_TEMPLATE(float4, float,__to_fp32)
PG_NUMERIC_TO_FLOAT_TEMPLATE(float8,double,__to_fp64)

#define PG_INT_TO_NUMERIC_TEMPLATE(SOURCE)							\
	PUBLIC_FUNCTION(bool)											\
	pgfn_##SOURCE##_to_numeric(XPU_PGFUNCTION_ARGS)					\
	{																\
		xpu_numeric_t	   *result = (xpu_numeric_t *)__result;		\
		xpu_##SOURCE##_t	datum;									\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);			\
																	\
		assert(kexp->nr_args == 1 &&								\
			   KEXP_IS_VALID(karg,SOURCE));							\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum))				\
			return false;											\
		result->isnull = datum.isnull;								\
		if (!result->isnull)										\
			set_normalized_numeric(result, datum.value, 0);			\
		return true;												\
	}
PG_INT_TO_NUMERIC_TEMPLATE(int1)
PG_INT_TO_NUMERIC_TEMPLATE(int2)
PG_INT_TO_NUMERIC_TEMPLATE(int4)
PG_INT_TO_NUMERIC_TEMPLATE(int8)
PG_INT_TO_NUMERIC_TEMPLATE(money)

#define PG_FLOAT_TO_NUMERIC_TEMPLATE(SOURCE,__TYPE,__CAST,			\
									 __MODF,__RINTL)				\
	PUBLIC_FUNCTION(bool)											\
	pgfn_##SOURCE##_to_numeric(XPU_PGFUNCTION_ARGS)					\
	{																\
		xpu_numeric_t	   *result = (xpu_numeric_t *)__result;		\
		xpu_##SOURCE##_t	datum;									\
		__TYPE				fval;									\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);			\
																	\
		assert(kexp->nr_args == 1 &&								\
			   KEXP_IS_VALID(karg,SOURCE));							\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum))				\
			return false;											\
		result->isnull = datum.isnull;								\
		if (result->isnull)											\
			return true;											\
		fval = __CAST(datum.value);									\
		if (isinf(fval))											\
			result->kind = (fval > 0.0								\
							? XPU_NUMERIC_KIND__POS_INF				\
							: XPU_NUMERIC_KIND__NEG_INF);			\
		else if (isnan(fval))										\
			result->kind = XPU_NUMERIC_KIND__NAN;					\
		else														\
		{															\
			__TYPE		a,b = __MODF(fval, &a);						\
			int128_t	value = (int128_t)a;						\
			int16_t		weight = 0;									\
			bool		negative = (value < 0);						\
																	\
			if (negative)											\
				value = -value;										\
			while (b != 0.0 && (value>>124) == 0)					\
			{														\
				b = __MODF(b * 10.0, &a);							\
				value = 10 * value + (int128_t)a;					\
				weight++;											\
			}														\
			set_normalized_numeric(result,value,weight);			\
		}															\
		return true;												\
	}
PG_FLOAT_TO_NUMERIC_TEMPLATE(float2, float,__to_fp32,modff,rintf)
PG_FLOAT_TO_NUMERIC_TEMPLATE(float4, float,__to_fp32,modff,rintf)
PG_FLOAT_TO_NUMERIC_TEMPLATE(float8,double,__to_fp64,modf, rint)

STATIC_FUNCTION(int)
__numeric_compare(const xpu_numeric_t *a, const xpu_numeric_t *b)
{
	int128_t	a_val = a->value;
	int128_t	b_val = b->value;
	int16_t		a_weight = a->weight;
	int16_t		b_weight = b->weight;

	/* If any NaN or Inf */
	if (a->kind != XPU_NUMERIC_KIND__VALID)
	{
		if (a->kind == XPU_NUMERIC_KIND__NAN)
		{
			if (b->kind == XPU_NUMERIC_KIND__NAN)
				return 0;	/* NaN == Nan */
			return 1;		/* NaN > non-NaN */
		}
		else if (a->kind == XPU_NUMERIC_KIND__POS_INF)
		{
			if (b->kind == XPU_NUMERIC_KIND__NAN)
				return -1;	/* +Inf < NaN */
			if (b->kind == XPU_NUMERIC_KIND__POS_INF)
				return 0;	/* +Inf == +Inf */
			return 1;		/* +Inf > anything else */
		}
		else
		{
			if (b->kind == XPU_NUMERIC_KIND__NEG_INF)
				return 0;	/* -Inf == -Inf */
			return -1;		/* -Inf < anything else */
		}
	}
	else if (b->kind != XPU_NUMERIC_KIND__VALID)
	{
		if (b->kind == XPU_NUMERIC_KIND__NEG_INF)
			return 1;		/* normal > -Inf */
		else
			return -1;		/* normal < NaN or +Inf */
	}
	else if ((a_val > 0 && b_val <= 0) || (a_val == 0 && b_val < 0))
		return 1;
	else if ((b_val > 0 && a_val <= 0) || (b_val == 0 && a_val < 0))
		return -1;
	/* Ok, both side are same sign with valid values */
	while (a_weight > b_weight)
	{
		b_val *= 10;
		b_weight++;
	}
	while (a_weight < b_weight)
	{
		a_val *= 10;
		a_weight++;
	}
	if (a_val > b_val)
		return 1;
	if (a_val < b_val)
		return -1;
	return 0;
}

#define PG_NUMERIC_COMPARE_TEMPLATE(NAME,OPER)						\
	PUBLIC_FUNCTION(bool)											\
	pgfn_numeric_##NAME(XPU_PGFUNCTION_ARGS)						\
	{																\
		xpu_bool_t	   *result = (xpu_bool_t *)__result;			\
		xpu_numeric_t	datum_a, datum_b;							\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);			\
																	\
		assert(kexp->nr_args == 2 &&								\
			   KEXP_IS_VALID(karg,numeric));						\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))			\
			return false;											\
		karg = KEXP_NEXT_ARG(karg);									\
		assert(KEXP_IS_VALID(karg, numeric));						\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))			\
			return false;											\
		result->isnull = (datum_a.isnull | datum_b.isnull);			\
		if (!result->isnull)										\
			result->value = (__numeric_compare(&datum_a,			\
											   &datum_b) OPER 0);	\
		return true;												\
	}
PG_NUMERIC_COMPARE_TEMPLATE(eq, ==)
PG_NUMERIC_COMPARE_TEMPLATE(ne, !=)
PG_NUMERIC_COMPARE_TEMPLATE(lt, <)
PG_NUMERIC_COMPARE_TEMPLATE(le, <=)
PG_NUMERIC_COMPARE_TEMPLATE(gt, >)
PG_NUMERIC_COMPARE_TEMPLATE(ge, >=)

PUBLIC_FUNCTION(bool)
pgfn_numeric_add(XPU_PGFUNCTION_ARGS)
{
	xpu_numeric_t  *result = (xpu_numeric_t *)__result;
	xpu_numeric_t	datum_a;
	xpu_numeric_t	datum_b;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 2 &&
		   KEXP_IS_VALID(karg, numeric));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))
		return false;
	karg = KEXP_NEXT_ARG(karg);
	assert(KEXP_IS_VALID(karg, numeric));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))
		return false;
	result->isnull = (datum_a.isnull | datum_b.isnull);
	if (!result->isnull)
	{
		if (datum_a.kind != XPU_NUMERIC_KIND__VALID ||
			datum_b.kind != XPU_NUMERIC_KIND__VALID)
		{
			if (datum_a.kind == XPU_NUMERIC_KIND__NAN ||
				datum_b.kind == XPU_NUMERIC_KIND__NAN)
				result->kind = XPU_NUMERIC_KIND__NAN;
			else if (datum_a.kind == XPU_NUMERIC_KIND__POS_INF)
			{
				if (datum_b.kind == XPU_NUMERIC_KIND__NEG_INF)
					result->kind = XPU_NUMERIC_KIND__NAN;	/* Inf - Inf */
				else
					result->kind = XPU_NUMERIC_KIND__POS_INF;
			}
			else if (datum_a.kind == XPU_NUMERIC_KIND__NEG_INF)
			{
				if (datum_b.kind == XPU_NUMERIC_KIND__POS_INF)
					result->kind = XPU_NUMERIC_KIND__NAN;	/* -Inf + Inf */
				else
					result->kind = XPU_NUMERIC_KIND__NEG_INF;
			}
			else if (datum_b.kind == XPU_NUMERIC_KIND__POS_INF)
				result->kind = XPU_NUMERIC_KIND__POS_INF;
			else
				result->kind = XPU_NUMERIC_KIND__NEG_INF;
		}
		else
		{
			while (datum_a.weight > datum_b.weight)
			{
				datum_b.value *= 10;
				datum_b.weight++;
			}
			while (datum_a.weight < datum_b.weight)
			{
				datum_a.value *= 10;
				datum_a.weight++;
			}
			set_normalized_numeric(result,
								   datum_a.value + datum_b.value,
								   datum_a.weight);
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_numeric_sub(XPU_PGFUNCTION_ARGS)
{
	xpu_numeric_t  *result = (xpu_numeric_t *)__result;
	xpu_numeric_t	datum_a;
	xpu_numeric_t	datum_b;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 2 &&
		   KEXP_IS_VALID(karg, numeric));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))
		return false;
	karg = KEXP_NEXT_ARG(karg);
	assert(KEXP_IS_VALID(karg, numeric));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))
		return false;
	result->isnull = (datum_a.isnull | datum_b.isnull);
	if (!result->isnull)
	{
		if (datum_a.kind != XPU_NUMERIC_KIND__VALID ||
			datum_b.kind != XPU_NUMERIC_KIND__VALID)
		{
			if (datum_a.kind == XPU_NUMERIC_KIND__NAN ||
				datum_b.kind == XPU_NUMERIC_KIND__NAN)
				result->kind = XPU_NUMERIC_KIND__NAN;
			else if (datum_a.kind == XPU_NUMERIC_KIND__POS_INF)
			{
				if (datum_b.kind == XPU_NUMERIC_KIND__POS_INF)
					result->kind = XPU_NUMERIC_KIND__NAN;	/* Inf - Inf */
				else
					result->kind = XPU_NUMERIC_KIND__POS_INF;
			}
			else if (datum_a.kind == XPU_NUMERIC_KIND__NEG_INF)
			{
				if (datum_b.kind == XPU_NUMERIC_KIND__NEG_INF)
					result->kind = XPU_NUMERIC_KIND__NAN;	/* -Inf - -Inf*/
				else
					result->kind = XPU_NUMERIC_KIND__NEG_INF;
			}
			else if (datum_b.kind == XPU_NUMERIC_KIND__POS_INF)
				result->kind = XPU_NUMERIC_KIND__NEG_INF;
			else
				result->kind = XPU_NUMERIC_KIND__POS_INF;
		}
		else
		{
			while (datum_a.weight > datum_b.weight)
			{
				datum_b.value *= 10;
				datum_b.weight++;
			}
			while (datum_a.weight < datum_b.weight)
			{
				datum_a.value *= 10;
				datum_a.weight++;
			}
			set_normalized_numeric(result,
								   datum_a.value - datum_b.value,
								   datum_a.weight);
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_numeric_mul(XPU_PGFUNCTION_ARGS)
{
	xpu_numeric_t  *result = (xpu_numeric_t *)__result;
	xpu_numeric_t	datum_a;
	xpu_numeric_t	datum_b;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 2 &&
		   KEXP_IS_VALID(karg, numeric));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))
		return false;
	karg = KEXP_NEXT_ARG(karg);
	assert(KEXP_IS_VALID(karg, numeric));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))
		return false;
	result->isnull = (datum_a.isnull | datum_b.isnull);
	if (!result->isnull)
	{
		if (datum_a.kind != XPU_NUMERIC_KIND__VALID ||
			datum_b.kind != XPU_NUMERIC_KIND__VALID)
		{
			if (datum_a.kind == XPU_NUMERIC_KIND__NAN ||
				datum_a.kind == XPU_NUMERIC_KIND__NAN)
			{
				result->kind = XPU_NUMERIC_KIND__NAN;
			}
			else if (datum_a.kind == XPU_NUMERIC_KIND__POS_INF)
			{
				int		__sign = xpu_numeric_sign(&datum_b);

				if (__sign < 0)
					result->kind = XPU_NUMERIC_KIND__NEG_INF;
				else if (__sign > 0)
					result->kind = XPU_NUMERIC_KIND__POS_INF;
				else
					result->kind = XPU_NUMERIC_KIND__NAN;
			}
			else if (datum_a.kind == XPU_NUMERIC_KIND__NEG_INF)
			{
				int		__sign = xpu_numeric_sign(&datum_b);

				if (__sign < 0)
					result->kind = XPU_NUMERIC_KIND__POS_INF;
				else if (__sign > 0)
					result->kind = XPU_NUMERIC_KIND__NEG_INF;
				else
					result->kind = XPU_NUMERIC_KIND__NAN;
			}
			else if (datum_b.kind == XPU_NUMERIC_KIND__POS_INF)
			{
				int		__sign = xpu_numeric_sign(&datum_a);

				if (__sign < 0)
					result->kind = XPU_NUMERIC_KIND__NEG_INF;
				else if (__sign > 0)
					result->kind = XPU_NUMERIC_KIND__POS_INF;
				else
					result->kind = XPU_NUMERIC_KIND__NAN;
			}
			else
			{
				int		__sign = xpu_numeric_sign(&datum_a);

				if (__sign < 0)
					result->kind = XPU_NUMERIC_KIND__POS_INF;
				else if (__sign > 0)
					result->kind = XPU_NUMERIC_KIND__NEG_INF;
				else
					result->kind = XPU_NUMERIC_KIND__NAN;
			}
		}
		else
		{
			set_normalized_numeric(result,
								   datum_a.value * datum_b.value,
								   datum_a.weight + datum_b.weight);
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_numeric_div(XPU_PGFUNCTION_ARGS)
{
	xpu_numeric_t  *result = (xpu_numeric_t *)__result;
	xpu_numeric_t	datum_a;
	xpu_numeric_t	datum_b;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 2 &&
		   KEXP_IS_VALID(karg, numeric));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))
		return false;
	karg = KEXP_NEXT_ARG(karg);
	assert(KEXP_IS_VALID(karg, numeric));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))
		return false;
	result->isnull = (datum_a.isnull | datum_b.isnull);
	if (result->isnull)
		return true;
	if (datum_a.kind != XPU_NUMERIC_KIND__VALID ||
		datum_b.kind != XPU_NUMERIC_KIND__VALID)
	{
		if (datum_a.kind == XPU_NUMERIC_KIND__NAN ||
			datum_b.kind == XPU_NUMERIC_KIND__NAN)
		{
			result->kind = XPU_NUMERIC_KIND__NAN;
		}
		else if (datum_a.kind == XPU_NUMERIC_KIND__POS_INF)
		{
			int		__sign = xpu_numeric_sign(&datum_b);
			if (__sign == 1)
				result->kind = XPU_NUMERIC_KIND__NEG_INF;
			else if (__sign == -1)
				result->kind = XPU_NUMERIC_KIND__POS_INF;
			else if (__sign != 0)
				result->kind = XPU_NUMERIC_KIND__NAN;
			else
			{
				STROM_ELOG(kcxt, "division by zero");
				return false;
			}
		}
		else if (datum_a.kind == XPU_NUMERIC_KIND__NEG_INF)
		{
			int		__sign = xpu_numeric_sign(&datum_b);

			if (__sign == 1)
				result->kind = XPU_NUMERIC_KIND__NEG_INF;
			else if (__sign == -1)
				result->kind = XPU_NUMERIC_KIND__POS_INF;
			else if (__sign != 0)
				result->kind = XPU_NUMERIC_KIND__NAN;
			else
			{
				STROM_ELOG(kcxt, "division by zero");
				return false;
			}
		}
		else
		{
			/* by here, datum_a must be finite, so datum_b is not */
			set_normalized_numeric(result, 0, 0);
		}
	}
	else if (datum_b.value == 0)
	{
		STROM_ELOG(kcxt, "division by zero");
		return false;
	}
	else
	{
		int128_t	rem = datum_a.value;
		int128_t	div = datum_b.value;
		int128_t	x, ival = 0;
		int16_t		weight = datum_a.weight - datum_b.weight;
		bool		negative = false;

		if (rem < 0)
		{
			rem = -rem;
			if (div < 0)
				div = -div;
			else
				negative = true;
		}
		else if (div < 0)
		{
			negative = true;
			div = -div;
		}
		assert(rem >= 0 && div >= 0);

		for (;;)
		{
			x = rem / div;
			ival = 10 * ival + x;
			rem -= x * div;
			if (rem == 0)
				break;
			
			rem *= 10;
			weight++;
		}
		if (negative)
			ival = -ival;
		set_normalized_numeric(result, ival, weight);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_numeric_mod(XPU_PGFUNCTION_ARGS)
{
	xpu_numeric_t  *result = (xpu_numeric_t *)__result;
	xpu_numeric_t	datum_a;
	xpu_numeric_t	datum_b;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 2 &&
		   KEXP_IS_VALID(karg, numeric));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))
		return false;
	karg = KEXP_NEXT_ARG(karg);
	assert(KEXP_IS_VALID(karg, numeric));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))
		return false;
	result->isnull = (datum_a.isnull | datum_b.isnull);
	if (result->isnull)
		return true;
	if (datum_a.kind != XPU_NUMERIC_KIND__VALID ||
		datum_b.kind != XPU_NUMERIC_KIND__VALID)
	{
		if (datum_a.kind == XPU_NUMERIC_KIND__NAN ||
			datum_b.kind == XPU_NUMERIC_KIND__NAN)
		{
			result->kind = XPU_NUMERIC_KIND__NAN;
		}
		else if (datum_a.kind == XPU_NUMERIC_KIND__POS_INF ||
				 datum_a.kind == XPU_NUMERIC_KIND__NEG_INF)
		{
			if (datum_b.kind == XPU_NUMERIC_KIND__VALID &&
				datum_b.value == 0)
			{
				STROM_ELOG(kcxt, "division by zero");
				return false;
			}
			else
			{
				result->kind = XPU_NUMERIC_KIND__NAN;
			}
		}
		else
		{
			/* num2 must be [-]Inf; result is num1 regardless of sign of num2 */
			result->kind = datum_b.kind;
			result->value = datum_b.value;
		}
	}
	else if (datum_b.value == 0)
	{
		STROM_ELOG(kcxt, "division by zero");
		return false;
	}
	else
	{
		while (datum_a.weight > datum_b.value)
		{
			datum_b.value *= 10;
			datum_b.weight++;
		}
		while (datum_a.weight < datum_b.weight)
		{
			datum_a.value *= 10;
			datum_a.weight++;
		}
		set_normalized_numeric(result,
							   datum_a.value % datum_b.value,
							   datum_a.weight);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_numeric_uplus(XPU_PGFUNCTION_ARGS)
{
	xpu_numeric_t  *result = (xpu_numeric_t *)__result;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 1 &&
		   KEXP_IS_VALID(karg, numeric));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, result))
		return false;
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_numeric_uminus(XPU_PGFUNCTION_ARGS)
{
	xpu_numeric_t  *result = (xpu_numeric_t *)__result;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

		assert(kexp->nr_args == 1 &&
		   KEXP_IS_VALID(karg, numeric));

	if (!EXEC_KERN_EXPRESSION(kcxt, karg, result))
		return false;
	if (!result->isnull &&
		result->kind == XPU_NUMERIC_KIND__VALID)
		result->value = -result->value;
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_numeric_abs(XPU_PGFUNCTION_ARGS)
{
	xpu_numeric_t  *result = (xpu_numeric_t *)__result;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 1 &&
		   KEXP_IS_VALID(karg, numeric));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, result))
		return false;
	if (!result->isnull &&
		result->kind == XPU_NUMERIC_KIND__VALID &&
		result->value < 0)
		result->value = -result->value;
	return true;
}
