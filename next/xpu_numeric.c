/*
 * xpu_numeric.c
 *
 * collection of numeric type support on xPU
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "xpu_common.h"

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
	result->weight = weight;
	result->value = value;
}

STATIC_FUNCTION(bool)
xpu_numeric_from_varlena(kern_context *kcxt,
						 xpu_numeric_t *result,
						 const varlena *addr)
{
	uint32_t		len;

	result->ops = &xpu_numeric_ops;
	if (!addr)
	{
		result->isnull = true;
		return true;
	}

	len = VARSIZE_ANY_EXHDR(addr);
	if (len >= sizeof(uint16_t))
	{
		NumericChoice *nc = (NumericChoice *)VARDATA_ANY(addr);
		NumericDigit *digits = NUMERIC_DIGITS(nc);
		int			weight  = NUMERIC_WEIGHT(nc) + 1;
		int			i, ndigits = NUMERIC_NDIGITS(nc, len);
		int128_t	value = 0;

		for (i=0; i < ndigits; i++)
		{
			NumericDigit dig = __Fetch(&digits[i]);

			value = value * PG_NBASE + dig;
			if (value < 0)
			{
				STROM_EREPORT(kcxt, ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE,
							  "numeric value is out of range");
				return false;
			}
		}
		if (NUMERIC_SIGN(nc) == NUMERIC_NEG)
			value = -value;
		weight = PG_DEC_DIGITS * (ndigits - weight);

		set_normalized_numeric(result, value, weight);
		return true;
	}
	STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,
				  "corrupted numeric header");
	return false;
}

STATIC_FUNCTION(int)
xpu_numeric_to_varlena(char *buffer, int16_t weight, int128_t value)
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
	len = offsetof(NumericData, choice.n_long.n_data[ndigits]);

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
	return xpu_numeric_from_varlena(kcxt, (xpu_numeric_t *)__result,
									(const varlena *)addr);
}

STATIC_FUNCTION(bool)
arrow_numeric_datum_ref(kern_context *kcxt,
						xpu_datum_t *__result,
						kern_data_store *kds,
						kern_colmeta *cmeta,
						uint32_t rowidx)
{
	xpu_numeric_t  *result = (xpu_numeric_t *)__result;
	void   *addr;

	addr = KDS_ARROW_REF_SIMPLE_DATUM(kds, cmeta, rowidx,
									  sizeof(int128_t));
	if (!addr)
		result->isnull = true;
	else
	{
		/*
		 * Note that Decimal::scale is equivalent to numeric::weight.
		 * It is the number of digits after the decimal point.
		 */
		set_normalized_numeric(result, *((int128_t *)addr),
							   cmeta->attopts.decimal.scale);
	}
	result->ops = &xpu_numeric_ops;
	return true;
}

PUBLIC_FUNCTION(int)
xpu_numeric_datum_store(kern_context *kcxt,
						char *buffer,
						xpu_datum_t *__arg)
{
	xpu_numeric_t *arg = (xpu_numeric_t *)__arg;

	if (arg->isnull)
		return 0;
	return xpu_numeric_to_varlena(buffer, arg->weight, arg->value);
}

PUBLIC_FUNCTION(bool)
xpu_numeric_datum_hash(kern_context *kcxt,
					   uint32_t *p_hash,
					   xpu_datum_t *__arg)
{
	xpu_numeric_t *arg = (xpu_numeric_t *)__arg;

	if (arg->isnull)
		*p_hash = 0;
	else
	{
		*p_hash = (pg_hash_any(&arg->weight, sizeof(int16_t)) ^
				   pg_hash_any(&arg->value, sizeof(int128_t)));
	}
	return true;
}
PGSTROM_SQLTYPE_OPERATORS(numeric);
