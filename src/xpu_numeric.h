/*
 * xpu_numeric.h
 *
 * Collection of numeric functions for both of GPU and DPU
 * --
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef XPU_NUMERIC_H
#define XPU_NUMERIC_H

typedef struct {
	KVEC_DATUM_COMMON_FIELD;
	uint8_t		kinds[KVEC_UNITSZ];
	int16_t		weights[KVEC_UNITSZ];
	struct {
		const varlena  *ptr;
		uint64_t		u64;
	}			values_lo[KVEC_UNITSZ];
	int64_t		values_hi[KVEC_UNITSZ];
} kvec_numeric_t;

typedef struct {
	XPU_DATUM_COMMON_FIELD;
	uint8_t		kind;			/* one of XPU_NUMERIC_KIND__* below */
	int16_t		weight;
	union {
		const varlena *vl_addr;	/* <= XPU_NUMERIC_KIND__VARLENA */
		int128_t	value;		/* <= XPU_NUMERIC_KIND__VALID */
	} u;
} xpu_numeric_t;
#define XPU_NUMERIC_KIND__VALID		0x00
#define XPU_NUMERIC_KIND__NAN		0x01
#define XPU_NUMERIC_KIND__POS_INF	0x02
#define XPU_NUMERIC_KIND__NEG_INF	0x03
#define XPU_NUMERIC_KIND__VARLENA	0xff		/* still in raw varlena format */
EXTERN_DATA(xpu_datum_operators, xpu_numeric_ops);

/*
 * PostgreSQL numeric data type
 */
#define PG_DEC_DIGITS	4
#define PG_NBASE		10000
typedef int16_t			NumericDigit;
#define PG_MAX_DIGITS	40	/* Max digits of 128bit integer */
#define PG_MAX_DATA		(PG_MAX_DIGITS / PG_DEC_DIGITS)

struct NumericShort
{
	uint16_t		n_header;				/* Sign + display scale + weight */
	NumericDigit	n_data[PG_MAX_DATA];	/* Digits */
};
typedef struct NumericShort	NumericShort;

struct NumericLong
{
	uint16_t		n_sign_dscale;			/* Sign + display scale */
	int16_t			n_weight;				/* Weight of 1st digit	*/
	NumericDigit	n_data[PG_MAX_DATA];	/* Digits */
};
typedef struct NumericLong	NumericLong;

typedef union
{
	uint16_t		n_header;			/* Header word */
	NumericLong		n_long;				/* Long form (4-byte header) */
	NumericShort	n_short;			/* Short form (2-byte header) */
} NumericChoice;

struct NumericData
{
	uint32_t		vl_len_;		/* varlena header */
	NumericChoice	choice;			/* payload */
};
typedef struct NumericData	NumericData;

#define NUMERIC_SIGN_MASK	0xC000
#define NUMERIC_POS			0x0000
#define NUMERIC_NEG			0x4000
#define NUMERIC_SHORT		0x8000
#define NUMERIC_SPECIAL		0xC000

#define NUMERIC_FLAGBITS(n_head)	((n_head) & NUMERIC_SIGN_MASK)
#define NUMERIC_IS_SHORT(n_head)	(NUMERIC_FLAGBITS(n_head) == NUMERIC_SHORT)
#define NUMERIC_IS_SPECIAL(n_head)	(NUMERIC_FLAGBITS(n_head) == NUMERIC_SPECIAL)

#define NUMERIC_EXT_SIGN_MASK	0xF000	/* high bits plus NaN/Inf flag bits */
#define NUMERIC_NAN				0xC000
#define NUMERIC_PINF			0xD000
#define NUMERIC_NINF			0xF000
#define NUMERIC_INF_SIGN_MASK	0x2000

#define NUMERIC_EXT_FLAGBITS(n_head) (n_head & NUMERIC_EXT_SIGN_MASK)
#define NUMERIC_IS_NAN(n_head)		(n_head == NUMERIC_NAN)
#define NUMERIC_IS_PINF(n_head)		(n_head == NUMERIC_PINF)
#define NUMERIC_IS_NINF(n_head)		(n_head == NUMERIC_NINF)
#define NUMERIC_IS_INF(n_head)		((n_head & ~NUMERIC_INF_SIGN_MASK) == NUMERIC_PINF)

#define NUMERIC_SHORT_SIGN_MASK		0x2000
#define NUMERIC_SHORT_DSCALE_MASK	0x1F80
#define NUMERIC_SHORT_DSCALE_SHIFT	7
#define NUMERIC_SHORT_DSCALE_MAX	(NUMERIC_SHORT_DSCALE_MASK >> \
									 NUMERIC_SHORT_DSCALE_SHIFT)
#define NUMERIC_SHORT_WEIGHT_SIGN_MASK 0x0040
#define NUMERIC_SHORT_WEIGHT_MASK	0x003F
#define NUMERIC_SHORT_WEIGHT_MAX	NUMERIC_SHORT_WEIGHT_MASK
#define NUMERIC_SHORT_WEIGHT_MIN	(-(NUMERIC_SHORT_WEIGHT_MASK+1))

#define NUMERIC_DSCALE_MASK         0x3FFF

INLINE_FUNCTION(uint32_t)
NUMERIC_NDIGITS(uint16_t n_head, uint32_t nc_len)
{
	return (NUMERIC_IS_SHORT(n_head)
			? (nc_len - offsetof(NumericChoice, n_short.n_data))
			: (nc_len - offsetof(NumericChoice, n_long.n_data)))
		/ sizeof(NumericDigit);
}

INLINE_FUNCTION(NumericDigit *)
NUMERIC_DIGITS(NumericChoice *nc, uint16_t n_head)
{
	return NUMERIC_IS_SHORT(n_head) ? nc->n_short.n_data : nc->n_long.n_data;
}

INLINE_FUNCTION(int)
NUMERIC_SIGN(uint16_t n_head)
{
	if (NUMERIC_IS_SHORT(n_head))
		return ((n_head & NUMERIC_SHORT_SIGN_MASK) ? NUMERIC_NEG : NUMERIC_POS);
	if (NUMERIC_IS_SPECIAL(n_head))
		return NUMERIC_EXT_FLAGBITS(n_head);
	return NUMERIC_FLAGBITS(n_head);
}

INLINE_FUNCTION(uint32_t)
NUMERIC_DSCALE(NumericChoice *nc, uint16_t n_head)
{
	if (NUMERIC_IS_SHORT(n_head))
		return ((n_head & NUMERIC_SHORT_DSCALE_MASK) >> NUMERIC_SHORT_DSCALE_SHIFT);
	return (__Fetch(&nc->n_long.n_sign_dscale) & NUMERIC_DSCALE_MASK);
}

INLINE_FUNCTION(int)
NUMERIC_WEIGHT(NumericChoice *nc, uint16_t n_head)
{
	int			weight;

	if (NUMERIC_IS_SHORT(n_head))
	{
		weight = (n_head) & NUMERIC_SHORT_WEIGHT_MASK;
		if (n_head & NUMERIC_SHORT_WEIGHT_SIGN_MASK)
			weight |= ~NUMERIC_SHORT_WEIGHT_MASK;
	}
	else
	{
		weight = __Fetch(&nc->n_long.n_weight);
	}
	return weight;
}

// __xpu_numeric_normalize
INLINE_FUNCTION(void)
NUMERIC_NORMALIZE(int16_t *p_weight, int128_t *p_value)
{
	int16_t		weight = *p_weight;
	int128_t	value  = *p_value;

	if (value == 0)
		weight = 0;
	else
	{
		while (weight > 0)
		{
			if (weight >= 5 && (value % 100000) == 0)
			{
				value  /= 100000;
				weight -= 5;
			}
			else if (weight >= 4 && (value % 10000) == 0)
			{
				value  /= 10000;
				weight -= 4;
			}
			else if (weight >= 3 && (value % 1000) == 0)
			{
				value  /= 1000;
				weight -= 3;
			}
			else if (weight >= 2 && (value % 100) == 0)
			{
				value  /= 100;
				weight -= 2;
			}
			else if (weight >= 1 && (value % 10) == 0)
			{
				value  /= 10;
				weight -= 1;
			}
			else
			{
				break;
			}
		}
	}
	*p_weight = weight;
	*p_value  = value;
}

/*
 * __decimal_from_varlena
 */
INLINE_FUNCTION(const char *)
__decimal_from_varlena(uint8_t *p_kind,
					   int16_t *p_weight,
					   int128_t *p_value,
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
				*p_kind = XPU_NUMERIC_KIND__NAN;
			else if (NUMERIC_IS_PINF(n_head))
				*p_kind = XPU_NUMERIC_KIND__POS_INF;
			else if (NUMERIC_IS_NINF(n_head))
				*p_kind = XPU_NUMERIC_KIND__NEG_INF;
			else
				return "unknown special numeric value";
			*p_weight = 0;
			*p_value  = 0;
		}
		else
		{
			NumericDigit *digits = NUMERIC_DIGITS(nc, n_head);
			int16_t		weight  = NUMERIC_WEIGHT(nc, n_head) + 1;
			int			ndigits = NUMERIC_NDIGITS(n_head, len);
			int128_t	value = 0;

			for (int i=0; i < ndigits; i++)
			{
				NumericDigit dig = __Fetch(&digits[i]);

				/*
				 * Rough overflow check - PG_NBASE is 10000, therefore,
				 * we never touch the upper limit as long as the value's
				 * significant 14bits are all zero.
				 */
				if ((value >> 114) != 0)
					return "numeric value is out of range";

				value = value * PG_NBASE + dig;
			}
			if (NUMERIC_SIGN(n_head) == NUMERIC_NEG)
				value = -value;
			weight = PG_DEC_DIGITS * (ndigits - weight);
			NUMERIC_NORMALIZE(&weight, &value);

			*p_kind   = XPU_NUMERIC_KIND__VALID;
			*p_weight = weight;
			*p_value  = value;
		}
		return NULL;
	}
	return "corrupted numeric header";
}
EXTERN_FUNCTION(const char *)
xpu_numeric_from_varlena(xpu_numeric_t *result, const varlena *addr);

INLINE_FUNCTION(int)
__decimal_to_varlena(char *buffer, uint8_t kind, int16_t weight, int128_t value)
{
	NumericData	   *numData = (NumericData *)buffer;
	NumericLong	   *numBody = &numData->choice.n_long;
	NumericDigit	n_data[PG_MAX_DATA];
	int				ndigits = 0;
	int				len;
	bool			is_negative = false;

	if (kind != XPU_NUMERIC_KIND__VALID)
	{
		len = offsetof(NumericData, choice.n_header) + sizeof(uint16_t);
		if (buffer)
		{
			NumericChoice *nc = (NumericChoice *)buffer;

			if (kind == XPU_NUMERIC_KIND__POS_INF)
				nc->n_header = NUMERIC_PINF;
			else if (kind == XPU_NUMERIC_KIND__POS_INF)
				nc->n_header = NUMERIC_NINF;
			else
				nc->n_header = NUMERIC_NAN;
			SET_VARSIZE(nc, len);
		}
		return len;
	}

	if (value < 0)
	{
		is_negative = true;
		value = -value;
	}
	NUMERIC_NORMALIZE(&weight, &value);

	/* special case handling for the least digits */
	if (value != 0)
	{
		int		mod = -1;

		switch (weight % PG_DEC_DIGITS)
		{
			case -1:
			case 3:
				mod = (value % 1000) * 10;
				value /= 1000;
				weight += 1;
				break;

			case -2:
			case 2:
				mod = (value % 100) * 100;
				value /= 100;
				weight += 2;
				break;

			case -3:
			case 1:
				mod = (value % 10) * 1000;
				value /= 10;
				weight += 3;
				break;
			default:
				/* well aligned */
				break;
		}
		if (mod >= 0)
		{
			ndigits++;
			n_data[PG_MAX_DATA - ndigits] = mod;
		}
	}
	else
	{
		/* value == 0 makes no sense on 'weight' */
		weight = 0;
	}

	while (value != 0)
    {
		int		mod;

		mod = (value % PG_NBASE);
		value /= PG_NBASE;
		ndigits++;
		n_data[PG_MAX_DATA - ndigits] = mod;
	}
	assert((weight % PG_DEC_DIGITS) == 0);
	len = (offsetof(NumericData, choice.n_long.n_data)
		   + sizeof(NumericDigit) * ndigits);
	if (weight < 0)
		len += sizeof(NumericDigit) * (-weight / PG_DEC_DIGITS);
	if (buffer)
	{
		uint16_t	n_header = Max(weight, 0);

		if (ndigits > 0)
			memcpy(numBody->n_data,
				   n_data + PG_MAX_DATA - ndigits,
				   sizeof(NumericDigit) * ndigits);
		if (weight < 0)
			memset(numBody->n_data + ndigits, 0,
				   sizeof(NumericDigit) * (-weight / PG_DEC_DIGITS));
		if (is_negative)
			n_header |= NUMERIC_NEG;
		numBody->n_sign_dscale = n_header;
		numBody->n_weight = ndigits - (weight / PG_DEC_DIGITS) - 1;

		SET_VARSIZE(numData, len);
	}
	return len;
}

INLINE_FUNCTION(bool)
xpu_numeric_validate(kern_context *kcxt, xpu_numeric_t *num)
{
	assert(num->expr_ops == &xpu_numeric_ops);
	if (num->kind == XPU_NUMERIC_KIND__VARLENA)
	{
		const varlena  *vl_addr = num->u.vl_addr;
		const char	   *errmsg;

		errmsg = xpu_numeric_from_varlena(num, vl_addr);
		if (errmsg)
		{
			STROM_ELOG(kcxt, errmsg);
			return false;
		}
		assert(num->kind != XPU_NUMERIC_KIND__VARLENA);
	}
	return true;
}

/*
 * for fixed-point numeric, the logic come from 'numeric_typmod_scale' in numeric.c
 */
INLINE_FUNCTION(int)
__numeric_typmod_weight(int32_t typmod)
{
	int		weight = __DBL_DIG__;	/* default if typmod < 0 */

	if (typmod >= 0)
	{
		weight = (((typmod - VARHDRSZ) & 0x7ff) ^ 1024) - 1024;
		weight = Max(weight, 0);			/* never negative */
		weight = Min(weight, __DBL_DIG__);	/* upper limit */
	}
	return weight;
}

/*
 * xpu_numeric_t related public functions
 */
EXTERN_FUNCTION(int)
pg_numeric_to_cstring(kern_context *kcxt,
					  varlena *numeric,
					  char *buf, char *endp);
EXTERN_FUNCTION(int128_t)
__normalize_numeric_int128(int16_t weight_d,
						   int16_t weight_s,
						   int128_t ival);
/*
 * utility functions to handle decimal form (int128)
 */
INLINE_FUNCTION(const char *)
__decimal_add(uint8_t *r_kind, int16_t *r_weight, int128_t *r_value,
			  uint8_t a_kind,  int16_t a_weight,  int128_t a_value,
			  uint8_t b_kind,  int16_t b_weight,  int128_t b_value)
{
	if (a_kind == XPU_NUMERIC_KIND__VALID ||
		b_kind == XPU_NUMERIC_KIND__VALID)
	{
		while (a_weight > b_weight)
		{
			b_value *= 10;
			b_weight++;
		}
		while (a_weight < b_weight)
		{
			a_value *= 10;
			a_weight++;
		}
		a_value += b_value;
		NUMERIC_NORMALIZE(&a_weight, &a_value);
		*r_kind = XPU_NUMERIC_KIND__VALID;
		*r_weight = a_weight;
		*r_value = a_value;
	}
	else
	{
		assert(a_kind != XPU_NUMERIC_KIND__VARLENA &&
			   b_kind != XPU_NUMERIC_KIND__VARLENA);
		if (a_kind == XPU_NUMERIC_KIND__NAN ||
			b_kind == XPU_NUMERIC_KIND__NAN)
			*r_kind = XPU_NUMERIC_KIND__NAN;
		else if (a_kind == XPU_NUMERIC_KIND__POS_INF)
		{
			if (b_kind == XPU_NUMERIC_KIND__NEG_INF)
				*r_kind = XPU_NUMERIC_KIND__NAN;	/* Inf - Inf */
			else
				*r_kind = XPU_NUMERIC_KIND__POS_INF;
		}
		else if (a_kind == XPU_NUMERIC_KIND__NEG_INF)
		{
			if (b_kind == XPU_NUMERIC_KIND__POS_INF)
				*r_kind = XPU_NUMERIC_KIND__NAN;	/* -Inf + Inf */
			else
				*r_kind = XPU_NUMERIC_KIND__NEG_INF;
		}
		else if (b_kind == XPU_NUMERIC_KIND__POS_INF)
			*r_kind = XPU_NUMERIC_KIND__POS_INF;
		else
			*r_kind = XPU_NUMERIC_KIND__NEG_INF;
	}
	return NULL;
}

INLINE_FUNCTION(const char *)
__decimal_sub(uint8_t *r_kind, int16_t *r_weight, int128_t *r_value,
			  uint8_t a_kind,  int16_t a_weight,  int128_t a_value,
			  uint8_t b_kind,  int16_t b_weight,  int128_t b_value)
{
	if (a_kind == XPU_NUMERIC_KIND__VALID ||
		b_kind == XPU_NUMERIC_KIND__VALID)
	{
		while (a_weight > b_weight)
		{
			b_value *= 10;
			b_weight++;
		}
		while (a_weight < b_weight)
		{
			a_value *= 10;
			a_weight++;
		}
		a_value -= b_value;
		NUMERIC_NORMALIZE(&a_weight, &a_value);
		*r_kind = XPU_NUMERIC_KIND__VALID;
		*r_weight = a_weight;
		*r_value = a_value;
	}
	else
	{
		assert(a_kind != XPU_NUMERIC_KIND__VARLENA &&
			   b_kind != XPU_NUMERIC_KIND__VARLENA);
		if (a_kind == XPU_NUMERIC_KIND__NAN ||
			b_kind == XPU_NUMERIC_KIND__NAN)
			*r_kind = XPU_NUMERIC_KIND__NAN;
		else if (a_kind == XPU_NUMERIC_KIND__POS_INF)
		{
			if (b_kind == XPU_NUMERIC_KIND__POS_INF)
				*r_kind = XPU_NUMERIC_KIND__NAN;	/* Inf - Inf */
			else
				*r_kind = XPU_NUMERIC_KIND__POS_INF;
		}
		else if (a_kind == XPU_NUMERIC_KIND__NEG_INF)
		{
			if (b_kind == XPU_NUMERIC_KIND__NEG_INF)
				*r_kind = XPU_NUMERIC_KIND__NAN;	/* -Inf - -Inf*/
			else
				*r_kind = XPU_NUMERIC_KIND__NEG_INF;
		}
		else if (b_kind == XPU_NUMERIC_KIND__POS_INF)
			*r_kind = XPU_NUMERIC_KIND__NEG_INF;
		else
			*r_kind = XPU_NUMERIC_KIND__POS_INF;
	}
	return NULL;
}
INLINE_FUNCTION(const char *)
__decimal_mul(uint8_t *r_kind, int16_t *r_weight, int128_t *r_value,
			  uint8_t a_kind,  int16_t a_weight,  int128_t a_value,
			  uint8_t b_kind,  int16_t b_weight,  int128_t b_value)
{
	if (a_kind == XPU_NUMERIC_KIND__VALID ||
		b_kind == XPU_NUMERIC_KIND__VALID)
	{
		*r_kind   = XPU_NUMERIC_KIND__VALID;
		*r_weight = a_weight + b_weight;
		*r_value  = a_value * b_value;
		NUMERIC_NORMALIZE(r_weight, r_value);
	}
	else
	{
		assert(a_kind != XPU_NUMERIC_KIND__VARLENA &&
			   b_kind != XPU_NUMERIC_KIND__VARLENA);
		if (a_kind == XPU_NUMERIC_KIND__POS_INF &&
			b_kind != XPU_NUMERIC_KIND__NAN)
		{
			if (b_kind == XPU_NUMERIC_KIND__NEG_INF ||
				(b_kind == XPU_NUMERIC_KIND__VALID && b_value < 0))
				*r_kind = XPU_NUMERIC_KIND__NEG_INF;
			else if (b_kind == XPU_NUMERIC_KIND__POS_INF ||
					 (b_kind == XPU_NUMERIC_KIND__VALID && b_value > 0))
				*r_kind = XPU_NUMERIC_KIND__POS_INF;
			else
				*r_kind = XPU_NUMERIC_KIND__NAN;
		}
		else if (a_kind == XPU_NUMERIC_KIND__NEG_INF &&
				 b_kind != XPU_NUMERIC_KIND__NAN)
		{
			if (b_kind == XPU_NUMERIC_KIND__NEG_INF ||
				(b_kind == XPU_NUMERIC_KIND__VALID && b_value < 0))
				*r_kind = XPU_NUMERIC_KIND__POS_INF;
			else if (b_kind == XPU_NUMERIC_KIND__POS_INF ||
					 (b_kind == XPU_NUMERIC_KIND__VALID && b_value > 0))
				*r_kind = XPU_NUMERIC_KIND__NEG_INF;
			else
				*r_kind = XPU_NUMERIC_KIND__NAN;
		}
		else if (a_kind != XPU_NUMERIC_KIND__NAN &&
				 b_kind == XPU_NUMERIC_KIND__POS_INF)
			
		{
			if (a_kind == XPU_NUMERIC_KIND__NEG_INF ||
				(a_kind == XPU_NUMERIC_KIND__VALID && a_value < 0))
				*r_kind = XPU_NUMERIC_KIND__NEG_INF;
			else if (a_kind == XPU_NUMERIC_KIND__POS_INF ||
					 (a_kind == XPU_NUMERIC_KIND__VALID && a_value > 0))
				*r_kind = XPU_NUMERIC_KIND__POS_INF;
			else
				*r_kind = XPU_NUMERIC_KIND__NAN;
		}
		else if (a_kind != XPU_NUMERIC_KIND__NAN &&
				 b_kind == XPU_NUMERIC_KIND__NEG_INF)
		{
			if (a_kind == XPU_NUMERIC_KIND__NEG_INF ||
				(a_kind == XPU_NUMERIC_KIND__VALID && a_value < 0))
				*r_kind = XPU_NUMERIC_KIND__POS_INF;
			else if (a_kind == XPU_NUMERIC_KIND__POS_INF ||
					 (a_kind == XPU_NUMERIC_KIND__VALID && a_value > 0))
				*r_kind = XPU_NUMERIC_KIND__NEG_INF;
			else
				*r_kind = XPU_NUMERIC_KIND__NAN;
		}
		else
		{
			*r_kind = XPU_NUMERIC_KIND__NAN;
		}
	}
	return NULL;
}

INLINE_FUNCTION(const char *)
__decimal_div(uint8_t *r_kind, int16_t *r_weight, int128_t *r_value,
			  uint8_t a_kind,  int16_t a_weight,  int128_t a_value,
			  uint8_t b_kind,  int16_t b_weight,  int128_t b_value)
{
	if (a_kind == XPU_NUMERIC_KIND__VALID &&
		b_kind == XPU_NUMERIC_KIND__VALID)
	{
		int128_t	rem = a_value;
		int128_t	div = b_value;
		int128_t	x, ival = 0;
		int16_t		weight = a_weight - b_weight;
		bool		negative = false;

		if (div == 0)
			return "division by zero";
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
			/*
			 * 999,999,999,999,999 = 0x03 8D7E A4C6 7FFF
			 */
			if (rem == 0 || (ival >> 50) != 0)
				break;
			rem *= 10;
			weight++;
		}
		if (negative)
			ival = -ival;
		NUMERIC_NORMALIZE(&weight, &ival);
		*r_kind   = XPU_NUMERIC_KIND__VALID;
		*r_weight = weight;
		*r_value  = ival;
	}
	else
	{
		assert(a_kind != XPU_NUMERIC_KIND__VARLENA &&
			   b_kind != XPU_NUMERIC_KIND__VARLENA);
		if (a_kind == XPU_NUMERIC_KIND__NAN ||
			b_kind == XPU_NUMERIC_KIND__NAN)
		{
			*r_kind = XPU_NUMERIC_KIND__NAN;
		}
		else if (a_kind == XPU_NUMERIC_KIND__POS_INF)
		{
			if (b_kind == XPU_NUMERIC_KIND__POS_INF)
				*r_kind = XPU_NUMERIC_KIND__POS_INF;
			else if (b_kind == XPU_NUMERIC_KIND__NEG_INF)
				*r_kind = XPU_NUMERIC_KIND__NEG_INF;
			else if (b_kind != XPU_NUMERIC_KIND__VALID)
				*r_kind = XPU_NUMERIC_KIND__NAN;
			else if (b_value > 0)
				*r_kind = XPU_NUMERIC_KIND__POS_INF;
			else if (b_value < 0)
				*r_kind = XPU_NUMERIC_KIND__NEG_INF;
			else
				return "division by zero";
		}
		else if (a_kind == XPU_NUMERIC_KIND__NEG_INF)
		{
			if (b_kind == XPU_NUMERIC_KIND__POS_INF)
				*r_kind = XPU_NUMERIC_KIND__NEG_INF;
			else if (b_kind == XPU_NUMERIC_KIND__NEG_INF)
				*r_kind = XPU_NUMERIC_KIND__POS_INF;
			else if (b_kind != XPU_NUMERIC_KIND__VALID)
				*r_kind = XPU_NUMERIC_KIND__NAN;
			else if (b_value > 0)
				*r_kind = XPU_NUMERIC_KIND__NEG_INF;
			else if (b_value < 0)
				*r_kind = XPU_NUMERIC_KIND__POS_INF;
			else
				return "division by zero";
		}
		else
		{
			/* by here, datum_a must be finite, so datum_b is not */
			*r_kind = XPU_NUMERIC_KIND__VALID;
			*r_weight = 0;
			*r_value  = 0;
		}
	}
	return NULL;
}

INLINE_FUNCTION(const char *)
__decimal_mod(uint8_t *r_kind, int16_t *r_weight, int128_t *r_value,
			  uint8_t a_kind,  int16_t a_weight,  int128_t a_value,
			  uint8_t b_kind,  int16_t b_weight,  int128_t b_value)
{

	assert(a_kind != XPU_NUMERIC_KIND__VARLENA &&
		   b_kind != XPU_NUMERIC_KIND__VARLENA);
	if (a_kind == XPU_NUMERIC_KIND__VALID &&
		b_kind == XPU_NUMERIC_KIND__VALID)
	{
		if (b_value == 0)
			return "division by zero";
		while (a_weight > b_weight)
		{
			b_value *= 10;
			b_weight++;
		}
		while (a_weight < b_weight)
		{
			a_value *= 10;
			a_weight++;
		}
		a_value = a_value / b_value;
		NUMERIC_NORMALIZE(&a_weight, &a_value);
		*r_kind = XPU_NUMERIC_KIND__VALID;
		*r_weight = a_weight;
		*r_value  = a_value;
	}
	else
	{
		if (a_kind == XPU_NUMERIC_KIND__NAN ||
			b_kind == XPU_NUMERIC_KIND__NAN)
		{
			*r_kind = XPU_NUMERIC_KIND__NAN;
		}
		else if (a_kind == XPU_NUMERIC_KIND__POS_INF ||
				 a_kind == XPU_NUMERIC_KIND__NEG_INF)
		{
			if (b_kind == XPU_NUMERIC_KIND__VALID &&
				b_value == 0)
				return "division by zero";
			*r_kind = XPU_NUMERIC_KIND__NAN;
		}
		else
		{
			/* num2 must be [-]Inf; result is num1 regardless of sign of num2 */
			*r_kind   = a_kind;
			*r_weight = a_weight;
			*r_value  = a_value;
		}
	}
	return NULL;
}

INLINE_FUNCTION(void)
__decimal_from_float8(float8_t __fval,
					  uint8_t *p_kind,
					  int16_t *p_weight,
					  int128_t *p_value)
{
	uint64_t	fval = __double_as_longlong__(__fval);
	uint64_t	frac = (fval & ((1UL<<FP64_FRAC_BITS)-1));
	int32_t		expo = (fval >> (FP64_FRAC_BITS)) & ((1UL<<FP64_EXPO_BITS)-1);
	bool		sign = (fval >> (FP64_FRAC_BITS + FP64_EXPO_BITS)) != 0;
	int			weight = 0;

	/* special cases */
	if (expo == 0x7ff)
	{
		if (fval != 0)
			*p_kind = XPU_NUMERIC_KIND__NAN;
		else if (sign)
			*p_kind = XPU_NUMERIC_KIND__NEG_INF;
		else
			*p_kind = XPU_NUMERIC_KIND__POS_INF;
		*p_weight = 0;
		*p_value = 0;
		return;
	}
	frac |= (1UL << FP64_FRAC_BITS);

	/*
	 * fraction must be adjusted by 10^prec / 2^(FP64_FRAC_BITS - expo)
	 * with keeping accuracy (52bit).
	 */
	while (expo > FP64_EXPO_BIAS + FP64_FRAC_BITS)
	{
		if (frac <= 0x1800000000000000UL)
		{
			frac *= 2;
			expo--;
		}
		else
		{
			frac /= 10;
			weight--;
		}
	}
	while (expo < FP64_EXPO_BIAS + FP64_FRAC_BITS)
	{
		if (frac >= 0x1800000000000000UL)
		{
			frac /= 2;
			expo++;
		}
		else
		{
			frac *= 10;
			weight++;
		}
	}
	/* only 15 digits are valid */
	while (frac >= 1000000000000000UL)
	{
		frac /= 10;
		weight--;
	}
	*p_kind = XPU_NUMERIC_KIND__VALID;
	*p_weight = weight;
	if (!sign)
		*p_value = (int128_t)frac;
	else
		*p_value = -(int128_t)frac;
}

EXTERN_FUNCTION(bool)
__xpu_numeric_to_int64(kern_context *kcxt,
					   int64_t *p_ival,
					   xpu_numeric_t *num,
					   int64_t min_value,
					   int64_t max_value);
EXTERN_FUNCTION(bool)
__xpu_numeric_to_fp64(kern_context *kcxt,
					  float8_t *p_ival,
					  xpu_numeric_t *num);
EXTERN_FUNCTION(void)
__xpu_fp64_to_numeric(xpu_numeric_t *result,
					  float8_t __fval);

#endif /* XPU_NUMERIC_H */
