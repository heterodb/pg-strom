/*
 * xpu_misclib.cu
 *
 * Collection of misc functions and operators for both of GPU and DPU
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
 * Mathmatical functions
 */
#define CHECKFLOATVAL(kcxt, fp_value, inf_is_valid, zero_is_valid)	\
	do {															\
		if (!(inf_is_valid) && isinf(fp_value))						\
		{															\
			STROM_ELOG(kcxt, "value out of range: overflow");		\
			return false;											\
		}															\
		if (!(zero_is_valid) && (fp_value) == 0.0)					\
		{															\
			STROM_ELOG(kcxt, "value out of range: underflow");		\
			return false;											\
		}															\
	} while(0)

PUBLIC_FUNCTION(bool)
pgfn_cbrt(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_float8_ops;
		result->value = cbrt(fval.value);
		CHECKFLOATVAL(kcxt, result->value,
					  isinf(fval.value),
					  fval.value == 0.0);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_dcbrt(XPU_PGFUNCTION_ARGS)
{
	return pgfn_cbrt(kcxt, kexp, __result);
}

PUBLIC_FUNCTION(bool)
pgfn_ceil(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
        result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_float8_ops;
		result->value = ceil(fval.value);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_ceiling(XPU_PGFUNCTION_ARGS)
{
	return pgfn_ceil(kcxt, kexp, __result);
}

PUBLIC_FUNCTION(bool)
pgfn_exp(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_float8_ops;
		if (isnan(fval.value))
			result->value = fval.value;
		else if (isinf(fval.value))
			result->value = (fval.value > 0.0 ? fval.value : 0.0);
		else
		{
			result->value = exp(fval.value);
			CHECKFLOATVAL(kcxt, result->value, false, false);
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_dexp(XPU_PGFUNCTION_ARGS)
{
	return pgfn_exp(kcxt, kexp, __result);
}

PUBLIC_FUNCTION(bool)
pgfn_floor(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_float8_ops;
		result->value = floor(fval.value);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_ln(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else if (fval.value <= 0.0)
	{
		STROM_ELOG(kcxt, "cannot take logarithm of zero or negative number");
		return false;
	}
	else
	{
		result->expr_ops = &xpu_float8_ops;
		result->value = log(fval.value);
		CHECKFLOATVAL(kcxt, result->value,
					  isinf(fval.value),
					  fval.value == 1.0);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_dlog10(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else if (fval.value == 0.0)
	{
		STROM_ELOG(kcxt, "cannot take logarithm of zero");
		return false;
	}
	else if (fval.value < 0.0)
	{
		STROM_ELOG(kcxt, "cannot take logarithm of a negative number");
		return false;
	}
	else
	{
		result->expr_ops = &xpu_float8_ops;
		result->value = log10(fval.value);
		CHECKFLOATVAL(kcxt, result->value,
					  isinf(fval.value),
					  fval.value != 1.0);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_dlog1(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else if (fval.value == 0.0)
	{
		STROM_ELOG(kcxt, "cannot take logarithm of zero");
		return false;
	}
	else if (fval.value < 0.0)
	{
		STROM_ELOG(kcxt, "cannot take logarithm of a negative number");
		return false;
	}
	else
	{
		result->expr_ops = &xpu_float8_ops;
		result->value = log(fval.value);
		CHECKFLOATVAL(kcxt, result->value,
					  isinf(fval.value),
					  fval.value != 1.0);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_pi(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS0(float8);

	result->expr_ops = &xpu_float8_ops;
	result->value = 3.14159265358979323846;
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_dpow(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(float8, float8, fval, float8, pval);

	if (XPU_DATUM_ISNULL(&fval) || XPU_DATUM_ISNULL(&pval))
		result->expr_ops = NULL;
	else if (fval.value == 0.0 && pval.value < 0.0)
	{
		STROM_ELOG(kcxt, "zero raised to a negative power is undefined");
		return false;
	}
	else if (fval.value < 0.0 && floor(pval.value) != pval.value)
	{
		STROM_ELOG(kcxt, "a negative number raised to a non-integer power yields a complex result");
		return false;
	}
	else
	{
		result->expr_ops = &xpu_float8_ops;
		result->value = pow(fval.value, pval.value);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_dround(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_float8_ops;
		result->value = rint(fval.value);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_sign(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_float8_ops;
		if (fval.value > 0.0)
			result->value = 1.0;
		else if (fval.value < 0.0)
			result->value = -1.0;
		else
			result->value = 0;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_dsqrt(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else if (fval.value < 0.0)
	{
		STROM_ELOG(kcxt, "cannot take square root of a negative number");
		return false;
	}
	else
	{
		result->expr_ops = &xpu_float8_ops;
		result->value = sqrt(fval.value);
		CHECKFLOATVAL(kcxt, result->value,
					  isinf(fval.value),
					  fval.value == 0.0);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_dtrunc(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_float8_ops;
		if (fval.value >= 0)
			result->value = floor(fval.value);
		else
			result->value = -floor(-fval.value);
	}
	return true;
}

/*
 * Trigonometric function
 */
#define RADIANS_PER_DEGREE		0.0174532925199432957692

PUBLIC_FUNCTION(bool)
pgfn_degrees(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_float8_ops;
		result->value = fval.value / RADIANS_PER_DEGREE;
		CHECKFLOATVAL(kcxt, result->value,
					  isinf(fval.value),
					  fval.value == 0.0);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_radians(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_float8_ops;
		result->value = fval.value * RADIANS_PER_DEGREE;
		CHECKFLOATVAL(kcxt, result->value,
					  isinf(fval.value),
					  fval.value == 0.0);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_acos(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_float8_ops;
		if (isnan(fval.value))
			result->value = fval.value;
		else if (fval.value < -1.0 || fval.value > 1.0)
		{
			STROM_ELOG(kcxt, "input is out of range");
			return false;
		}
		else
		{
			result->value = acos(fval.value);
			CHECKFLOATVAL(kcxt, result->value, false, true);
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_asin(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_float8_ops;
		if (isnan(fval.value))
			result->value = fval.value;
		else if (fval.value < -1.0 || fval.value > 1.0)
		{
			STROM_ELOG(kcxt, "input is out of range");
			return false;
		}
		else
		{
			result->value = asin(fval.value);
			CHECKFLOATVAL(kcxt, result->value, false, true);
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_atan(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_float8_ops;
		if (isnan(fval.value))
			result->value = DBL_NAN;
		else
		{
			result->value = atan(fval.value);
			CHECKFLOATVAL(kcxt, result->value, false, true);
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_atan2(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(float8, float8, fval, float8, pval);

	if (XPU_DATUM_ISNULL(&fval) || XPU_DATUM_ISNULL(&pval))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_float8_ops;
		if (isnan(fval.value) || isnan(pval.value))
			result->value = DBL_NAN;
		else
		{
			result->value = atan2(fval.value, pval.value);
			CHECKFLOATVAL(kcxt, result->value, false, true);
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_cos(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_float8_ops;
		if (isnan(fval.value))
			result->value = DBL_NAN;
		else
		{
			result->value = cos(fval.value);
			CHECKFLOATVAL(kcxt, result->value, false, true);
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_cot(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else
	{
		CHECKFLOATVAL(kcxt, fval.value, false, true);

		result->expr_ops = &xpu_float8_ops;
		result->value = 1.0 / tan(fval.value);
		CHECKFLOATVAL(kcxt, result->value, true, true);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_sin(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_float8_ops;
		if (isnan(fval.value))
			result->value = DBL_NAN;
		else
		{
			result->value = sin(fval.value);
			CHECKFLOATVAL(kcxt, result->value, false, true);
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_tan(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(float8, float8, fval);

	if (XPU_DATUM_ISNULL(&fval))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_float8_ops;
		if (isnan(fval.value))
			result->value = DBL_NAN;
		else
		{
			result->value = tan(fval.value);
			CHECKFLOATVAL(kcxt, result->value, true, true);
		}
	}
	return true;
}

/*
 * Currency data type (xpu_money_t), functions and operators
 */
STATIC_FUNCTION(bool)
xpu_money_datum_heap_read(kern_context *kcxt,
						  const void *addr,
						  xpu_datum_t *__result)
{
	xpu_money_t *result = (xpu_money_t *)__result;

	result->expr_ops = &xpu_money_ops;
	__FetchStore(result->value, (Cash *)addr);
	return true;
}

STATIC_FUNCTION(bool)
xpu_money_datum_arrow_read(kern_context *kcxt,
						   const kern_data_store *kds,
						   const kern_colmeta *cmeta,
						   uint32_t kds_index,
						   xpu_datum_t *__result)
{
	xpu_money_t *result = (xpu_money_t *)__result;
	const void	*addr;

	if (cmeta->attopts.tag != ArrowType__Int)
	{
		STROM_ELOG(kcxt, "xpu_money_t must be mapped on Arrow::Int32 or Int64");
		return false;
	}
	switch (cmeta->attopts.integer.bitWidth)
	{
		case 32:
			addr = KDS_ARROW_REF_SIMPLE_DATUM(kds, cmeta,
											  kds_index,
											  sizeof(int32_t));
			if (!addr)
				result->expr_ops = NULL;
			else
			{
				if (cmeta->attopts.integer.is_signed)
					result->value = *((int32_t *)addr);
				else
					result->value = *((uint32_t *)addr);
				result->expr_ops = &xpu_money_ops;
			}
			break;
		case 64:
			addr = KDS_ARROW_REF_SIMPLE_DATUM(kds, cmeta, kds_index, sizeof(int64_t));
			if (!addr)
				result->expr_ops = NULL;
			else if (!cmeta->attopts.integer.is_signed && *((int64_t *)addr) < 0)
			{
				STROM_ELOG(kcxt, "Arrow::Int64 out of range");
				return false;
			}
			else
			{
				result->value = *((int64_t *)addr);
				result->expr_ops = &xpu_money_ops;
			}
			break;
		default:
			STROM_ELOG(kcxt, "xpu_money_t must be mapped on Arrow::Int32 or Int64");
			return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_money_datum_kvec_load(kern_context *kcxt,
						  const kvec_datum_t *__kvecs,
						  uint32_t kvecs_id,
						  xpu_datum_t *__result)
{
	const kvec_money_t *kvecs = (const kvec_money_t *)__kvecs;
	xpu_money_t *result = (xpu_money_t *)__result;

	result->expr_ops = &xpu_money_ops;
	result->value = kvecs->values[kvecs_id];
	return true;
}

STATIC_FUNCTION(bool)
xpu_money_datum_kvec_save(kern_context *kcxt,
						  const xpu_datum_t *__xdatum,
						  kvec_datum_t *__kvecs,
						  uint32_t kvecs_id)
{
	const xpu_money_t *xdatum = (const xpu_money_t *)__xdatum;
	kvec_money_t *kvecs = (kvec_money_t *)__kvecs;

	kvecs->values[kvecs_id] = xdatum->value;
	return true;
}

STATIC_FUNCTION(bool)
xpu_money_datum_kvec_copy(kern_context *kcxt,
                          const kvec_datum_t *__kvecs_src,
                          uint32_t kvecs_src_id,
                          kvec_datum_t *__kvecs_dst,
                          uint32_t kvecs_dst_id)
{
	const kvec_money_t *kvecs_src = (const kvec_money_t *)__kvecs_src;
	kvec_money_t *kvecs_dst = (kvec_money_t *)__kvecs_dst;

	kvecs_dst->values[kvecs_dst_id] = kvecs_src->values[kvecs_src_id];
	return true;
}

STATIC_FUNCTION(int)
xpu_money_datum_write(kern_context *kcxt,
					  char *buffer,
					  const kern_colmeta *cmeta,
					  const xpu_datum_t *__arg)
{
	const xpu_money_t *arg = (const xpu_money_t *)__arg;

	if (buffer)
		*((Cash *)buffer) = arg->value;
	return sizeof(Cash);
}

STATIC_FUNCTION(bool)
xpu_money_datum_hash(kern_context *kcxt,
					 uint32_t *p_hash,
					 xpu_datum_t *__arg)
{
	xpu_money_t *arg = (xpu_money_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(Cash));
	return true;
}

STATIC_FUNCTION(bool)
xpu_money_datum_comp(kern_context *kcxt,
					 int *p_comp,
					 xpu_datum_t *__a,
					 xpu_datum_t *__b)
{
	xpu_money_t *a = (xpu_money_t *)__a;
	xpu_money_t *b = (xpu_money_t *)__b;

	assert(!XPU_DATUM_ISNULL(a) && !XPU_DATUM_ISNULL(b));
	if (a->value < b->value)
		*p_comp = -1;
	else if (a->value > b->value)
		*p_comp = 1;
	else
		*p_comp = 0;
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(money, true, 8, sizeof(Cash));
PG_SIMPLE_COMPARE_TEMPLATE(cash_,money,money,)

/*
 * UUID data type (xpu_uuid_t), functions and operators
 */
STATIC_FUNCTION(bool)
xpu_uuid_datum_heap_read(kern_context *kcxt,
						 const void *addr,
                         xpu_datum_t *__result)
{
	xpu_uuid_t *result = (xpu_uuid_t *)__result;

	result->expr_ops = &xpu_uuid_ops;
	memcpy(result->value.data, addr, UUID_LEN);
	return true;
}

STATIC_FUNCTION(bool)
xpu_uuid_datum_arrow_read(kern_context *kcxt,
						  const kern_data_store *kds,
						  const kern_colmeta *cmeta,
						  uint32_t kds_index,
						  xpu_datum_t *__result)
{
	xpu_uuid_t *result = (xpu_uuid_t *)__result;
	const void *addr;

	if (cmeta->attopts.tag != ArrowType__FixedSizeBinary ||
		cmeta->attopts.fixed_size_binary.byteWidth != UUID_LEN)
	{
		STROM_ELOG(kcxt, "xpu_uuid_t must be mapped on Arrow::FixedSizeBinary");
		return false;
	}
	addr = KDS_ARROW_REF_SIMPLE_DATUM(kds, cmeta, kds_index, UUID_LEN);
	if (!addr)
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_uuid_ops;
		memcpy(result->value.data, addr, UUID_LEN);
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_uuid_datum_kvec_load(kern_context *kcxt,
						 const kvec_datum_t *__kvecs,
                         uint32_t kvecs_id,
                         xpu_datum_t *__result)
{
	const kvec_uuid_t *kvecs = (const kvec_uuid_t *)__kvecs;
	xpu_uuid_t *result = (xpu_uuid_t *)__result;

	result->expr_ops = &xpu_uuid_ops;
	memcpy(result->value.data, kvecs->values[kvecs_id].data, UUID_LEN);
	return true;
}

STATIC_FUNCTION(bool)
xpu_uuid_datum_kvec_save(kern_context *kcxt,
						 const xpu_datum_t *__xdatum,
						 kvec_datum_t *__kvecs,
						 uint32_t kvecs_id)
{
	const xpu_uuid_t *xdatum = (const xpu_uuid_t *)__xdatum;
	kvec_uuid_t *kvecs = (kvec_uuid_t *)__kvecs;

	memcpy(kvecs->values[kvecs_id].data, xdatum->value.data, UUID_LEN);
	return true;
}

STATIC_FUNCTION(bool)
xpu_uuid_datum_kvec_copy(kern_context *kcxt,
						 const kvec_datum_t *__kvecs_src,
						 uint32_t kvecs_src_id,
						 kvec_datum_t *__kvecs_dst,
						 uint32_t kvecs_dst_id)
{
	const kvec_uuid_t *kvecs_src = (const kvec_uuid_t *)__kvecs_src;
	kvec_uuid_t *kvecs_dst = (kvec_uuid_t *)__kvecs_dst;

	memcpy(kvecs_dst->values[kvecs_dst_id].data,
		   kvecs_src->values[kvecs_src_id].data, UUID_LEN);
	return true;
}

STATIC_FUNCTION(int)
xpu_uuid_datum_write(kern_context *kcxt,
					 char *buffer,
					 const kern_colmeta *cmeta,
					 const xpu_datum_t *__arg)
{
	const xpu_uuid_t *arg = (const xpu_uuid_t *)__arg;

	if (buffer)
		memcpy(buffer, arg->value.data, UUID_LEN);
	return sizeof(pg_uuid_t);
}

STATIC_FUNCTION(bool)
xpu_uuid_datum_hash(kern_context *kcxt,
					uint32_t *p_hash,
					xpu_datum_t *__arg)
{
	xpu_uuid_t *arg = (xpu_uuid_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(arg->value.data, UUID_LEN);
	return true;
}

INLINE_FUNCTION(int)
uuid_cmp_internal(const xpu_uuid_t *datum_a,
				  const xpu_uuid_t *datum_b)
{
	const uint8_t  *s1 = datum_a->value.data;
	const uint8_t  *s2 = datum_b->value.data;

	for (int i=0; i < UUID_LEN; i++, s1++, s2++)
	{
		if (*s1 < *s2)
			return -1;
		if (*s1 > *s2)
			return 1;
	}
	return 0;
}

STATIC_FUNCTION(bool)
xpu_uuid_datum_comp(kern_context *kcxt,
					int *p_comp,
					xpu_datum_t *__a,
					xpu_datum_t *__b)
{
	xpu_uuid_t *a = (xpu_uuid_t *)__a;
	xpu_uuid_t *b = (xpu_uuid_t *)__b;

	assert(!XPU_DATUM_ISNULL(a) && !XPU_DATUM_ISNULL(b));
	*p_comp = __memcmp(a->value.data,
					   b->value.data,
					   UUID_LEN);
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(uuid, false, 1, UUID_LEN);

#define PG_UUID_COMPARE_TEMPLATE(NAME,OPER)								\
	PUBLIC_FUNCTION(bool)												\
	pgfn_uuid_##NAME(XPU_PGFUNCTION_ARGS)								\
	{																	\
		xpu_bool_t *result = (xpu_bool_t *)__result;					\
		xpu_uuid_t	datum_a;											\
		xpu_uuid_t	datum_b;											\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args == 2 && KEXP_IS_VALID(karg, uuid));		\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))				\
			return false;												\
		karg = KEXP_NEXT_ARG(karg);										\
		assert(KEXP_IS_VALID(karg, uuid));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))				\
			return false;												\
																		\
		if (XPU_DATUM_ISNULL(&datum_a) || XPU_DATUM_ISNULL(&datum_b))	\
		{																\
			__pg_simple_nullcomp_##NAME(&datum_a, &datum_b);			\
		}																\
		else															\
		{																\
			result->expr_ops = &xpu_bool_ops;							\
			result->value = (__memcmp(datum_a.value.data,				\
									  datum_b.value.data,				\
									  UUID_LEN) OPER 0);				\
		}																\
		return true;													\
	}
PG_UUID_COMPARE_TEMPLATE(eq, ==)
PG_UUID_COMPARE_TEMPLATE(ne, != )
PG_UUID_COMPARE_TEMPLATE(lt, < )
PG_UUID_COMPARE_TEMPLATE(le, <=)
PG_UUID_COMPARE_TEMPLATE(gt, > )
PG_UUID_COMPARE_TEMPLATE(ge, >=)

/*
 * Macaddr data type (xpu_macaddr_t), functions and operators
 */
STATIC_FUNCTION(bool)
xpu_macaddr_datum_heap_read(kern_context *kcxt,
							const void *addr,
							xpu_datum_t *__result)
{
	xpu_macaddr_t *result = (xpu_macaddr_t *)__result;

	result->expr_ops = &xpu_macaddr_ops;
	memcpy(&result->value, addr, sizeof(macaddr));
	return true;
}

STATIC_FUNCTION(bool)
xpu_macaddr_datum_arrow_read(kern_context *kcxt,
							 const kern_data_store *kds,
							 const kern_colmeta *cmeta,
							 uint32_t kds_index,
							 xpu_datum_t *__result)
{
	xpu_macaddr_t  *result = (xpu_macaddr_t *)__result;
	const void	   *addr;

	if (cmeta->attopts.tag != ArrowType__FixedSizeBinary ||
		cmeta->attopts.fixed_size_binary.byteWidth != sizeof(macaddr))
	{
		STROM_ELOG(kcxt, "xpu_macaddr_t must be mapped on Arrow::FixedSizeBinary");
		return false;
	}
	addr = KDS_ARROW_REF_SIMPLE_DATUM(kds, cmeta, kds_index, sizeof(macaddr));
	if (!addr)
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_macaddr_ops;
		memcpy(&result->value, addr, sizeof(macaddr));
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_macaddr_datum_kvec_load(kern_context *kcxt,
							const kvec_datum_t *__kvecs,
							uint32_t kvecs_id,
							xpu_datum_t *__result)
{
	const kvec_macaddr_t *kvecs = (const kvec_macaddr_t *)__kvecs;
	xpu_macaddr_t *result = (xpu_macaddr_t *)__result;

	result->expr_ops = &xpu_macaddr_ops;
	memcpy(&result->value, &kvecs->values[kvecs_id], sizeof(macaddr));
	return true;
}

STATIC_FUNCTION(bool)
xpu_macaddr_datum_kvec_save(kern_context *kcxt,
							const xpu_datum_t *__xdatum,
							kvec_datum_t *__kvecs,
							uint32_t kvecs_id)
{
	const xpu_macaddr_t *xdatum = (const xpu_macaddr_t *)__xdatum;
	kvec_macaddr_t *kvecs = (kvec_macaddr_t *)__kvecs;

	memcpy(&kvecs->values[kvecs_id], &xdatum->value, sizeof(macaddr));
    return true;
}

STATIC_FUNCTION(bool)
xpu_macaddr_datum_kvec_copy(kern_context *kcxt,
							const kvec_datum_t *__kvecs_src,
							uint32_t kvecs_src_id,
							kvec_datum_t *__kvecs_dst,
							uint32_t kvecs_dst_id)
{
	const kvec_macaddr_t *kvecs_src = (const kvec_macaddr_t *)__kvecs_src;
	kvec_macaddr_t *kvecs_dst = (kvec_macaddr_t *)__kvecs_dst;

	memcpy(&kvecs_dst->values[kvecs_dst_id],
		   &kvecs_src->values[kvecs_src_id], sizeof(macaddr));
	return true;
}

STATIC_FUNCTION(int)
xpu_macaddr_datum_write(kern_context *kcxt,
						char *buffer,
						const kern_colmeta *cmeta,
						const xpu_datum_t *__arg)
{
	const xpu_macaddr_t  *arg = (xpu_macaddr_t *)__arg;

	if (buffer)
		memcpy(buffer, &arg->value, sizeof(macaddr));
	return sizeof(macaddr);
}

STATIC_FUNCTION(bool)
xpu_macaddr_datum_hash(kern_context *kcxt,
					   uint32_t *p_hash,
					   xpu_datum_t *__arg)
{
	xpu_macaddr_t *arg = (xpu_macaddr_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_hash = 0;
	else
		*p_hash = pg_hash_any(&arg->value, sizeof(macaddr));
	return true;
}

INLINE_FUNCTION(int)
macaddr_cmp_internal(const xpu_macaddr_t *datum_a, const xpu_macaddr_t *datum_b)
{
	uint32_t	bits_a;
	uint32_t	bits_b;

	/* high-bits */
	bits_a = ((uint32_t)datum_a->value.a << 16 |
			  (uint32_t)datum_a->value.b << 8  |
			  (uint32_t)datum_a->value.c);
	bits_b = ((uint32_t)datum_b->value.a << 16 |
			  (uint32_t)datum_b->value.b << 8  |
			  (uint32_t)datum_b->value.c);
	if (bits_a < bits_b)
		return -1;
	if (bits_a > bits_b)
		return 1;
	/* low-bits  */
	bits_a = ((uint32_t)datum_a->value.d << 16 |
			  (uint32_t)datum_a->value.e << 8  |
			  (uint32_t)datum_a->value.f);
	bits_b = ((uint32_t)datum_b->value.d << 16 |
			  (uint32_t)datum_b->value.e << 8  |
			  (uint32_t)datum_b->value.f);
	if (bits_a < bits_b)
		return -1;
	if (bits_a > bits_b)
		return 1;
	return 0;
}

STATIC_FUNCTION(bool)
xpu_macaddr_datum_comp(kern_context *kcxt,
					   int *p_comp,
					   xpu_datum_t *__a,
					   xpu_datum_t *__b)
{
	xpu_macaddr_t *a = (xpu_macaddr_t *)__a;
	xpu_macaddr_t *b = (xpu_macaddr_t *)__b;

	assert(!XPU_DATUM_ISNULL(a) && !XPU_DATUM_ISNULL(b));
	*p_comp = macaddr_cmp_internal(a, b);
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(macaddr, false, 4, sizeof(macaddr));

#define PG_MACADDR_COMPARE_TEMPLATE(NAME,OPER)							\
	PUBLIC_FUNCTION(bool)												\
	pgfn_macaddr_##NAME(XPU_PGFUNCTION_ARGS)							\
	{																	\
		xpu_bool_t	   *result = (xpu_bool_t *)__result;				\
		xpu_macaddr_t	datum_a;										\
		xpu_macaddr_t	datum_b;										\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args == 2 && KEXP_IS_VALID(karg, macaddr));		\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))				\
			return false;												\
		karg = KEXP_NEXT_ARG(karg);										\
		assert(KEXP_IS_VALID(karg, macaddr));							\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))				\
			return false;												\
																		\
		if (XPU_DATUM_ISNULL(&datum_a) || XPU_DATUM_ISNULL(&datum_b))	\
		{																\
			__pg_simple_nullcomp_##NAME(&datum_a, &datum_b);			\
		}																\
		else															\
		{																\
			result->expr_ops = &xpu_bool_ops;							\
			result->value = (macaddr_cmp_internal(&datum_a,				\
												  &datum_b) OPER 0);	\
		}																\
		return true;													\
	}
PG_MACADDR_COMPARE_TEMPLATE(eq, ==)
PG_MACADDR_COMPARE_TEMPLATE(ne, !=)
PG_MACADDR_COMPARE_TEMPLATE(lt, < )
PG_MACADDR_COMPARE_TEMPLATE(le, <=)
PG_MACADDR_COMPARE_TEMPLATE(gt, > )
PG_MACADDR_COMPARE_TEMPLATE(ge, >=)

PUBLIC_FUNCTION(bool)
pgfn_macaddr_trunc(XPU_PGFUNCTION_ARGS)
{
	xpu_macaddr_t  *result = (xpu_macaddr_t *)__result;
	xpu_macaddr_t	datum;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 1 && KEXP_IS_VALID(karg, macaddr));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum))
		return false;
	if (XPU_DATUM_ISNULL(&datum))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_macaddr_ops;
		result->value.a = datum.value.a;
		result->value.b = datum.value.b;
		result->value.c = datum.value.c;
		result->value.d = 0;
		result->value.e = 0;
		result->value.f = 0;
	}
	return true;
}

/*
 * Inet data type (xpu_iner_t), functions and operators
 */
STATIC_FUNCTION(bool)
xpu_inet_datum_heap_read(kern_context *kcxt,
                         const void *addr,
                         xpu_datum_t *__result)
{
	xpu_inet_t *result = (xpu_inet_t *)__result;
	int		sz;

	if (VARATT_IS_EXTERNAL(addr) || VARATT_IS_COMPRESSED(addr))
	{
		STROM_CPU_FALLBACK(kcxt, "inet value is compressed or toasted");
		return false;
	}
	sz = VARSIZE_ANY_EXHDR(addr);
	if (sz == offsetof(inet_struct, ipaddr[4]))
	{
		memcpy(&result->value, VARDATA_ANY(addr), sz);
		if (result->value.family != PGSQL_AF_INET)
		{
			STROM_ELOG(kcxt, "inet (ipv4) value corruption");
			return false;
		}
	}
	else if (sz == offsetof(inet_struct, ipaddr[16]))
	{
		memcpy(&result->value, VARDATA_ANY(addr), sz);
		if (result->value.family != PGSQL_AF_INET6)
		{
			STROM_ELOG(kcxt, "inet (ipv6) value corruption");
			return false;
		}
	}
	else
	{
		STROM_ELOG(kcxt, "Bug? inet value is corrupted");
		return false;
	}
	result->expr_ops = &xpu_inet_ops;
	return true;
}

STATIC_FUNCTION(bool)
xpu_inet_datum_arrow_read(kern_context *kcxt,
                          const kern_data_store *kds,
                          const kern_colmeta *cmeta,
                          uint32_t kds_index,
                          xpu_datum_t *__result)
{
	xpu_inet_t *result = (xpu_inet_t *)__result;
	const void *addr;

	if (cmeta->attopts.tag != ArrowType__FixedSizeBinary)
	{
		STROM_ELOG(kcxt, "xpu_inet_t must be mapped on Arrow::FixedSizeBinary");
		return false;
	}
	if (cmeta->attopts.fixed_size_binary.byteWidth == 4)
	{
		addr = KDS_ARROW_REF_SIMPLE_DATUM(kds, cmeta, kds_index, 4);
		if (!addr)
			result->expr_ops = NULL;
		else
		{
			result->expr_ops = &xpu_inet_ops;
			result->value.family = PGSQL_AF_INET;
			result->value.bits = 32;
			memcpy(result->value.ipaddr, addr, 4);
		}
	}
	else if (cmeta->attopts.fixed_size_binary.byteWidth == 16)
	{
		addr = KDS_ARROW_REF_SIMPLE_DATUM(kds, cmeta, kds_index, 4);
		if (!addr)
			result->expr_ops = NULL;
		else
		{
			result->expr_ops = &xpu_inet_ops;
			result->value.family = PGSQL_AF_INET6;
			result->value.bits = 128;
			memcpy(result->value.ipaddr, addr, 16);
		}
	}
	else
	{
		STROM_ELOG(kcxt, "xpu_inet_t must be mapped on Arrow::FixedSizeBinary<4> or <16>");
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_inet_datum_kvec_load(kern_context *kcxt,
                         const kvec_datum_t *__kvecs,
                         uint32_t kvecs_id,
                         xpu_datum_t *__result)
{
	const kvec_inet_t *kvecs = (const kvec_inet_t *)__kvecs;
	xpu_inet_t *result = (xpu_inet_t *)__result;
	uint8_t		family;

	result->expr_ops = &xpu_inet_ops;
	result->value.family = family = kvecs->family[kvecs_id];
	result->value.bits = kvecs->bits[kvecs_id];
	memcpy(result->value.ipaddr,
		   kvecs->ipaddr[kvecs_id].data,
		   (family == PGSQL_AF_INET ? 4 : 16));
	return true;
}

STATIC_FUNCTION(bool)
xpu_inet_datum_kvec_save(kern_context *kcxt,
						 const xpu_datum_t *__xdatum,
						 kvec_datum_t *__kvecs,
						 uint32_t kvecs_id)
{
    const xpu_inet_t *xdatum = (const xpu_inet_t *)__xdatum;
    kvec_inet_t *kvecs = (kvec_inet_t *)__kvecs;
	uint8_t		family;

	kvecs->family[kvecs_id] = family = xdatum->value.family;
	kvecs->bits[kvecs_id] = xdatum->value.bits;
	memcpy(kvecs->ipaddr[kvecs_id].data,
		   xdatum->value.ipaddr,
		   (family == PGSQL_AF_INET ? 4 : 16));
	return true;
}

STATIC_FUNCTION(bool)
xpu_inet_datum_kvec_copy(kern_context *kcxt,
                          const kvec_datum_t *__kvecs_src,
                          uint32_t kvecs_src_id,
                          kvec_datum_t *__kvecs_dst,
                          uint32_t kvecs_dst_id)
{
	const kvec_inet_t *kvecs_src = (const kvec_inet_t *)__kvecs_src;
	kvec_inet_t *kvecs_dst = (kvec_inet_t *)__kvecs_dst;
	uint8_t		family;

	family = kvecs_src->family[kvecs_src_id];
	kvecs_dst->family[kvecs_dst_id] = family;
	kvecs_dst->bits[kvecs_dst_id]   = kvecs_src->bits[kvecs_src_id];
	memcpy(kvecs_dst->ipaddr[kvecs_dst_id].data,
		   kvecs_src->ipaddr[kvecs_src_id].data,
		   (family == PGSQL_AF_INET ? 4 : 16));
	return true;
}

STATIC_FUNCTION(int)
xpu_inet_datum_write(kern_context *kcxt,
					 char *buffer,
					 const kern_colmeta *cmeta,
					 const xpu_datum_t *__arg)
{
	const xpu_inet_t  *arg = (xpu_inet_t *)__arg;
	int		sz;

	if (XPU_DATUM_ISNULL(arg))
		return 0;
	if (arg->value.family == PGSQL_AF_INET)
		sz = offsetof(inet_struct, ipaddr) + 4;
	else if (arg->value.family == PGSQL_AF_INET6)
		sz = offsetof(inet_struct, ipaddr) + 16;
	else
	{
		STROM_ELOG(kcxt, "Bug? inet value is corrupted");
		return -1;
	}
	if (buffer)
	{
		memcpy(buffer+VARHDRSZ, &arg->value, sz);
		SET_VARSIZE(buffer, VARHDRSZ + sz);
	}
	return VARHDRSZ + sz;
}

STATIC_FUNCTION(bool)
xpu_inet_datum_hash(kern_context *kcxt,
					uint32_t *p_hash,
					xpu_datum_t *__arg)
{
	xpu_inet_t *arg = (xpu_inet_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_hash = 0;
	else
	{
		int		sz;

		if (arg->value.family == PGSQL_AF_INET)
			sz = offsetof(inet_struct, ipaddr[4]);		/* IPv4 */
		else if (arg->value.family == PGSQL_AF_INET6)
			sz = offsetof(inet_struct, ipaddr[16]);	/* IPv6 */
		else
		{
			STROM_ELOG(kcxt, "corrupted inet datum");
			return false;
		}
		*p_hash = pg_hash_any(&arg->value, sz);
	}
	return true;
}
/* see utils/adt/network.c */
INLINE_FUNCTION(int)
bitncmp(const unsigned char *l, const unsigned char *r, int n)
{
	unsigned int lb, rb;
	int		b = n / 8;

	for (int i=0; i < b; i++)
	{
		if (l[i] < r[i])
			return -1;
		if (l[i] > r[i])
			return 1;
	}
	if ((n % 8) == 0)
		return 0;

	lb = l[b];
	rb = r[b];
	for (b = n % 8; b > 0; b--)
	{
		if ((lb & 0x80) != (rb & 0x80))
		{
			if ((lb & 0x80) != 0)
				return 1;
			return -1;
		}
		lb <<= 1;
		rb <<= 1;
	}
	return 0;
}

INLINE_FUNCTION(int)
inet_cmp_internal(const xpu_inet_t *datum_a,
				  const xpu_inet_t *datum_b)
{
	if (datum_a->value.family == datum_b->value.family)
	{
		int		order;

		order = bitncmp(datum_a->value.ipaddr,
						datum_b->value.ipaddr,
						Min(datum_a->value.bits,
							datum_a->value.bits));
		if (order != 0)
			return order;
		order = (int)datum_a->value.bits - (int)datum_a->value.bits;
		if (order != 0)
			return order;
		return bitncmp(datum_a->value.ipaddr,
					   datum_b->value.ipaddr,
					   datum_a->value.family == PGSQL_AF_INET ? 32 : 128);
	}
	return ((int)datum_a->value.family - (int)datum_b->value.family);
}

STATIC_FUNCTION(bool)
xpu_inet_datum_comp(kern_context *kcxt,
					int *p_comp,
					xpu_datum_t *__a,
					xpu_datum_t *__b)
{
	xpu_inet_t *a = (xpu_inet_t *)__a;
	xpu_inet_t *b = (xpu_inet_t *)__b;

	assert(!XPU_DATUM_ISNULL(a) && !XPU_DATUM_ISNULL(b));
	*p_comp = inet_cmp_internal(a, b);
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(inet, false, 4, -1);

#define PG_NETWORK_COMPARE_TEMPLATE(NAME,OPER)							\
	PUBLIC_FUNCTION(bool)												\
	pgfn_network_##NAME(XPU_PGFUNCTION_ARGS)							\
	{																	\
		xpu_bool_t *result = (xpu_bool_t *)__result;					\
		xpu_inet_t	datum_a;											\
		xpu_inet_t	datum_b;											\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args == 2 && KEXP_IS_VALID(karg, inet));		\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))				\
			return false;												\
		karg = KEXP_NEXT_ARG(karg);										\
		assert(KEXP_IS_VALID(karg, inet));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))				\
			return false;												\
																		\
		if (XPU_DATUM_ISNULL(&datum_a) || XPU_DATUM_ISNULL(&datum_b))	\
		{																\
			__pg_simple_nullcomp_##NAME(&datum_a, &datum_b);			\
		}																\
		else															\
		{																\
			result->expr_ops = &xpu_bool_ops;							\
			result->value = (inet_cmp_internal(&datum_a,				\
											   &datum_b) OPER 0);		\
		}																\
		return true;													\
	}
PG_NETWORK_COMPARE_TEMPLATE(eq, ==)
PG_NETWORK_COMPARE_TEMPLATE(ne, !=)
PG_NETWORK_COMPARE_TEMPLATE(lt, < )
PG_NETWORK_COMPARE_TEMPLATE(le, <=)
PG_NETWORK_COMPARE_TEMPLATE(gt, > )
PG_NETWORK_COMPARE_TEMPLATE(ge, >=)

#define PG_NETWORK_SUBSUP_TEMPLATE(NAME,OPER)							\
	PUBLIC_FUNCTION(bool)												\
	pgfn_network_##NAME(XPU_PGFUNCTION_ARGS)							\
	{																	\
		xpu_bool_t *result = (xpu_bool_t *)__result;					\
		xpu_inet_t	datum_a;											\
		xpu_inet_t	datum_b;											\
		const kern_expression *karg = KEXP_FIRST_ARG(kexp);				\
																		\
		assert(kexp->nr_args == 2 && KEXP_IS_VALID(karg, inet));		\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))				\
			return false;												\
		karg = KEXP_NEXT_ARG(karg);										\
		assert(KEXP_IS_VALID(karg, inet));								\
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))				\
			return false;												\
																		\
		if (XPU_DATUM_ISNULL(&datum_a) || XPU_DATUM_ISNULL(&datum_b))	\
		{																\
			result->expr_ops = NULL;									\
		}																\
		else if (datum_a.value.family == datum_b.value.family &&		\
				 datum_a.value.bits OPER datum_b.value.bits)			\
		{																\
			result->expr_ops = &xpu_bool_ops;							\
			result->value = (bitncmp(datum_a.value.ipaddr,				\
									 datum_b.value.ipaddr,				\
									 datum_a.value.bits) == 0);			\
		}																\
		else															\
		{																\
			result->expr_ops = &xpu_bool_ops;							\
			result->value = false;										\
		}																\
		return true;													\
	}

PG_NETWORK_SUBSUP_TEMPLATE(sub, <)
PG_NETWORK_SUBSUP_TEMPLATE(subeq, <=)
PG_NETWORK_SUBSUP_TEMPLATE(sup, >)
PG_NETWORK_SUBSUP_TEMPLATE(supeq, >=)

PUBLIC_FUNCTION(bool)
pgfn_network_overlap(XPU_PGFUNCTION_ARGS)
{
	xpu_bool_t *result = (xpu_bool_t *)__result;
	xpu_inet_t	datum_a;
	xpu_inet_t	datum_b;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 2 && KEXP_IS_VALID(karg, inet));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_a))
		return false;
	karg = KEXP_NEXT_ARG(karg);
	assert(KEXP_IS_VALID(karg, inet));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum_b))
		return false;

	if (XPU_DATUM_ISNULL(&datum_a) || XPU_DATUM_ISNULL(&datum_b))
	{
		result->expr_ops = NULL;
	}
	else if (datum_a.value.family == datum_b.value.family)
	{
		result->expr_ops = &xpu_bool_ops;
        result->value = (bitncmp(datum_a.value.ipaddr,
								 datum_b.value.ipaddr,
								 Min(datum_a.value.bits,
									 datum_b.value.bits)) == 0);
	}
	else
	{
		result->expr_ops = &xpu_bool_ops;
		result->value = false;
	}
	return true;
}

/* ----------------------------------------------------------------
 *
 * cube (alias of earthdistance) data type and functions
 *
 * ----------------------------------------------------------------
 */
INLINE_FUNCTION(bool)
IS_POINT(const __NDBOX *cube)
{
	return ((__Fetch(&cube->header) & POINT_BIT) != 0);
}

INLINE_FUNCTION(int)
DIM(const __NDBOX *cube)
{
	return (__Fetch(&cube->header) & DIM_MASK);
}

INLINE_FUNCTION(double)
LL_COORD(const __NDBOX *cube, int i)
{
	return __Fetch(cube->x + i);
}

INLINE_FUNCTION(double)
UR_COORD(const __NDBOX *cube, int i)
{
	return __Fetch(cube->x + (IS_POINT(cube) ? i : DIM(cube) + i));
}

INLINE_FUNCTION(bool)
xpu_cube_is_valid(kern_context *kcxt, const xpu_cube_t *arg)
{
	int		dim;

	if (arg->length < 0)
	{
		STROM_CPU_FALLBACK(kcxt, "cube datum is compressed or external");
		return false;
	}

	dim = DIM((const __NDBOX *)arg->value);
	if (arg->length < (offsetof(__NDBOX, x) +
					   IS_POINT((const __NDBOX *)arg->value)
					   ? sizeof(double) * dim
					   : 2 * sizeof(double) * dim))
	{
		STROM_ELOG(kcxt, "cube datum is corrupted");
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_cube_datum_heap_read(kern_context *kcxt,
						 const void *addr,
						 xpu_datum_t *__result)
{
	xpu_cube_t *result = (xpu_cube_t *)__result;

	if (VARATT_IS_EXTERNAL(addr) || VARATT_IS_COMPRESSED(addr))
	{
		result->value  = (const char *)addr;
		result->length = -1;
	}
	else
	{
		result->value  = VARDATA_ANY(addr);
		result->length = VARSIZE_ANY_EXHDR(addr);
	}
	result->expr_ops = &xpu_cube_ops;
	return true;
}

STATIC_FUNCTION(bool)
xpu_cube_datum_arrow_read(kern_context *kcxt,
						  const kern_data_store *kds,
						  const kern_colmeta *cmeta,
						  uint32_t kds_index,
						  xpu_datum_t *__result)
{
	xpu_cube_t *result = (xpu_cube_t *)__result;

	if (cmeta->attopts.tag == ArrowType__Binary)
	{
		result->value = (const char *)
			KDS_ARROW_REF_VARLENA32_DATUM(kds, cmeta, kds_index,
										  &result->length);
	}
	else if (cmeta->attopts.tag == ArrowType__LargeBinary)
	{
		result->value = (const char *)
			KDS_ARROW_REF_VARLENA64_DATUM(kds, cmeta, kds_index,
										  &result->length);
	}
	else
	{
		STROM_ELOG(kcxt, "not a mappable Arrow data type for cube");
		return false;
	}
	result->expr_ops = (result->value != NULL ? &xpu_cube_ops : NULL);
	return true;
}

STATIC_FUNCTION(bool)
xpu_cube_datum_kvec_load(kern_context *kcxt,
						 const kvec_datum_t *__kvecs,
						 uint32_t kvecs_id,
						 xpu_datum_t *__result)
{
	const kvec_cube_t *kvecs = (const kvec_cube_t *)__kvecs;
	xpu_cube_t *result = (xpu_cube_t *)__result;

	result->expr_ops = &xpu_cube_ops;
	result->length = kvecs->length[kvecs_id];
	result->value  = kvecs->values[kvecs_id];
	return true;
}

STATIC_FUNCTION(bool)
xpu_cube_datum_kvec_save(kern_context *kcxt,
						 const xpu_datum_t *__xdatum,
						 kvec_datum_t *__kvecs,
						 uint32_t kvecs_id)
{
	const xpu_cube_t *xdatum = (const xpu_cube_t *)__xdatum;
	kvec_cube_t *kvecs = (kvec_cube_t *)__kvecs;

	kvecs->length[kvecs_id] = xdatum->length;
	kvecs->values[kvecs_id] = xdatum->value;
	return true;
}

STATIC_FUNCTION(bool)
xpu_cube_datum_kvec_copy(kern_context *kcxt,
						 const kvec_datum_t *__kvecs_src,
						 uint32_t kvecs_src_id,
						 kvec_datum_t *__kvecs_dst,
						 uint32_t kvecs_dst_id)
{
	const kvec_cube_t *kvecs_src = (const kvec_cube_t *)__kvecs_src;
	kvec_cube_t *kvecs_dst = (kvec_cube_t *)__kvecs_dst;

	kvecs_dst->length[kvecs_dst_id] = kvecs_src->length[kvecs_src_id];
	kvecs_dst->values[kvecs_dst_id] = kvecs_src->values[kvecs_src_id];
    return true;
}

STATIC_FUNCTION(int)
xpu_cube_datum_write(kern_context *kcxt,
					 char *buffer,
					 const kern_colmeta *cmeta,
					 const xpu_datum_t *__arg)
{
	const xpu_cube_t *arg = (const xpu_cube_t *)__arg;
	int		nbytes;

	if (arg->length < 0)
	{
		nbytes = VARSIZE_ANY(arg->value);
		if (buffer)
			memcpy(buffer, arg->value, nbytes);
	}
	else
	{
		nbytes = VARHDRSZ + arg->length;
		if (buffer)
		{
			memcpy(buffer+VARHDRSZ, arg->value, arg->length);
			SET_VARSIZE(buffer, nbytes);
		}
	}
	return nbytes;
}

STATIC_FUNCTION(bool)
xpu_cube_datum_hash(kern_context *kcxt,
					uint32_t *p_hash,
					xpu_datum_t *__arg)
{
	xpu_cube_t *arg = (xpu_cube_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
		*p_hash = 0;
	else if (xpu_cube_is_valid(kcxt, arg))
		*p_hash = pg_hash_any(arg->value, arg->length);
	else
		return false;
	return true;
}

STATIC_FUNCTION(int)
pg_cube_cmp_v0(const __NDBOX *a, const __NDBOX *b)
{
	int		dim = Min(DIM(a), DIM(b));

	/* compare the common dimensions */
	for (int i = 0; i < dim; i++)
	{
		if (Min(LL_COORD(a,i), UR_COORD(a,i)) > Min(LL_COORD(b,i), UR_COORD(b,i)))
			return 1;
		if (Min(LL_COORD(a,i), UR_COORD(a,i)) < Min(LL_COORD(b,i), UR_COORD(b,i)))
			return -1;
	}
	for (int i = 0; i < dim; i++)
	{
		if (Max(LL_COORD(a,i), UR_COORD(a,i)) > Max(LL_COORD(b,i), UR_COORD(b,i)))
			return 1;
		if (Max(LL_COORD(a,i), UR_COORD(a,i)) < Max(LL_COORD(b,i), UR_COORD(b,i)))
			return -1;
	}

	/* compare extra dimensions to zero */
	if (DIM(a) > DIM(b))
	{
		for (int i = dim; i < DIM(a); i++)
		{
			if (Min(LL_COORD(a,i), UR_COORD(a,i)) > 0)
				return 1;
			if (Min(LL_COORD(a,i), UR_COORD(a,i)) < 0)
				return -1;
		}
		for (int i = dim; i < DIM(a); i++)
		{
			if (Max(LL_COORD(a, i), UR_COORD(a, i)) > 0)
				return 1;
			if (Max(LL_COORD(a, i), UR_COORD(a, i)) < 0)
				return -1;
		}

		/*
		 * if all common dimensions are equal, the cube with more dimensions
		 * wins
		 */
		return 1;
	}
	if (DIM(a) < DIM(b))
	{
		for (int i = dim; i < DIM(b); i++)
		{
			if (Min(LL_COORD(b,i), UR_COORD(b,i)) > 0)
				return -1;
			if (Min(LL_COORD(b,i), UR_COORD(b,i)) < 0)
				return 1;
		}
		for (int i = dim; i < DIM(b); i++)
		{
			if (Max(LL_COORD(b,i), UR_COORD(b,i)) > 0)
				return -1;
			if (Max(LL_COORD(b,i), UR_COORD(b,i)) < 0)
				return 1;
		}

		/*
		 * if all common dimensions are equal, the cube with more dimensions
		 * wins
		 */
		return -1;
	}
	/* They're really equal */
	return 0;
}

STATIC_FUNCTION(bool)
xpu_cube_datum_comp(kern_context *kcxt,
					int *p_comp,
					xpu_datum_t *__a,
					xpu_datum_t *__b)
{
	const xpu_cube_t *a = (const xpu_cube_t *)__a;
	const xpu_cube_t *b = (const xpu_cube_t *)__b;	

	if (!xpu_cube_is_valid(kcxt, a) ||
		!xpu_cube_is_valid(kcxt, b))
		return false;

	*p_comp = pg_cube_cmp_v0((const __NDBOX *)a->value,
							 (const __NDBOX *)b->value);
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(cube, false, 8, -1);

PUBLIC_FUNCTION(bool)
pgfn_cube_eq(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(bool, cube, arg1, cube, arg2);

	if (XPU_DATUM_ISNULL(&arg1) || XPU_DATUM_ISNULL(&arg2))
		result->expr_ops = NULL;
	else if (!xpu_cube_is_valid(kcxt, &arg1) ||
			 !xpu_cube_is_valid(kcxt, &arg2))
		return false;
	else
	{
		result->expr_ops = &xpu_bool_ops;
		result->value = (pg_cube_cmp_v0((const __NDBOX *)arg1.value,
										(const __NDBOX *)arg2.value) == 0);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_cube_ne(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(bool, cube, arg1, cube, arg2);

	if (XPU_DATUM_ISNULL(&arg1) || XPU_DATUM_ISNULL(&arg2))
		result->expr_ops = NULL;
	else if (!xpu_cube_is_valid(kcxt, &arg1) ||
			 !xpu_cube_is_valid(kcxt, &arg2))
		return false;
	else
	{
		result->expr_ops = &xpu_bool_ops;
		result->value = (pg_cube_cmp_v0((const __NDBOX *)arg1.value,
										(const __NDBOX *)arg2.value) != 0);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_cube_lt(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(bool, cube, arg1, cube, arg2);

	if (XPU_DATUM_ISNULL(&arg1) || XPU_DATUM_ISNULL(&arg2))
		result->expr_ops = NULL;
	else if (!xpu_cube_is_valid(kcxt, &arg1) ||
			 !xpu_cube_is_valid(kcxt, &arg2))
		return false;
	else
	{
		result->expr_ops = &xpu_bool_ops;
		result->value = (pg_cube_cmp_v0((const __NDBOX *)arg1.value,
										(const __NDBOX *)arg2.value) < 0);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_cube_gt(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(bool, cube, arg1, cube, arg2);

	if (XPU_DATUM_ISNULL(&arg1) || XPU_DATUM_ISNULL(&arg2))
		result->expr_ops = NULL;
	else if (!xpu_cube_is_valid(kcxt, &arg1) ||
			 !xpu_cube_is_valid(kcxt, &arg2))
		return false;
	else
	{
		result->expr_ops = &xpu_bool_ops;
		result->value = (pg_cube_cmp_v0((const __NDBOX *)arg1.value,
										(const __NDBOX *)arg2.value) > 0);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_cube_le(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(bool, cube, arg1, cube, arg2);

	if (XPU_DATUM_ISNULL(&arg1) || XPU_DATUM_ISNULL(&arg2))
		result->expr_ops = NULL;
	else if (!xpu_cube_is_valid(kcxt, &arg1) ||
			 !xpu_cube_is_valid(kcxt, &arg2))
		return false;
	else
	{
		result->expr_ops = &xpu_bool_ops;
		result->value = (pg_cube_cmp_v0((const __NDBOX *)arg1.value,
										(const __NDBOX *)arg2.value) <= 0);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_cube_ge(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(bool, cube, arg1, cube, arg2);

	if (XPU_DATUM_ISNULL(&arg1) || XPU_DATUM_ISNULL(&arg2))
		result->expr_ops = NULL;
	else if (!xpu_cube_is_valid(kcxt, &arg1) ||
			 !xpu_cube_is_valid(kcxt, &arg2))
		return false;
	else
	{
		result->expr_ops = &xpu_bool_ops;
		result->value = (pg_cube_cmp_v0((const __NDBOX *)arg1.value,
										(const __NDBOX *)arg2.value) >= 0);
	}
	return true;
}

STATIC_FUNCTION(bool)
pg_cube_contains_v0(const __NDBOX *a, const __NDBOX *b)
{
	if (DIM(a) < DIM(b))
	{
		/*
		 * the further comparisons will make sense if the excess dimensions of
		 * (b) were zeroes Since both UL and UR coordinates must be zero, we
		 * can check them all without worrying about which is which.
		 */
		for (int i = DIM(a); i < DIM(b); i++)
		{
			if (LL_COORD(b, i) != 0)
				return false;
			if (UR_COORD(b, i) != 0)
				return false;
        }
	}
	/* Can't care less about the excess dimensions of (a), if any */
	for (int i = 0; i < Min(DIM(a), DIM(b)); i++)
	{
		if (Min(LL_COORD(a, i), UR_COORD(a, i)) >
			Min(LL_COORD(b, i), UR_COORD(b, i)))
			return false;
		if (Max(LL_COORD(a, i), UR_COORD(a, i)) <
			Max(LL_COORD(b, i), UR_COORD(b, i)))
			return false;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_cube_contains(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(bool, cube, arg1, cube, arg2);

	if (XPU_DATUM_ISNULL(&arg1) || XPU_DATUM_ISNULL(&arg2))
		result->expr_ops = NULL;
	else if (!xpu_cube_is_valid(kcxt, &arg1) ||
			 !xpu_cube_is_valid(kcxt, &arg2))
		return false;
	else
	{
		result->expr_ops = &xpu_bool_ops;
		result->value = pg_cube_contains_v0((const __NDBOX *)arg1.value,
											(const __NDBOX *)arg2.value);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_cube_contained(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(bool, cube, arg1, cube, arg2);

	if (XPU_DATUM_ISNULL(&arg1) || XPU_DATUM_ISNULL(&arg2))
		result->expr_ops = NULL;
	else if (!xpu_cube_is_valid(kcxt, &arg1) ||
			 !xpu_cube_is_valid(kcxt, &arg2))
		return false;
	else
	{
		result->expr_ops = &xpu_bool_ops;
		result->value = pg_cube_contains_v0((const __NDBOX *)arg2.value,
											(const __NDBOX *)arg1.value);
	}
	return true;
}

PUBLIC_FUNCTION(bool)
pgfn_cube_ll_coord(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS2(float8, cube, cval, int4, ival);

	if (XPU_DATUM_ISNULL(&cval) || XPU_DATUM_ISNULL(&ival))
		result->expr_ops = NULL;
	else if (!xpu_cube_is_valid(kcxt, &cval))
		return false;
	else
	{
		const __NDBOX  *c = (const __NDBOX *)cval.value;
		int				n = ival.value;

		if (DIM(c) >= n && n > 0)
			result->value = Max(LL_COORD(c, n-1),
								UR_COORD(c, n-1));
		else
			result->value = 0.0;
		result->expr_ops = &xpu_float8_ops;
	}
	return true;
}
