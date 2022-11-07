/*
 * xpu_numeric.h
 *
 * Collection of numeric functions for both of GPU and DPU
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef XPU_NUMERIC_H
#define XPU_NUMERIC_H

typedef struct {
	XPU_DATUM_COMMON_FIELD;
	uint8_t		kind;		/* one of XPU_NUMERIC_KIND__* below */
	int16_t		weight;
	int128_t	value;
} xpu_numeric_t;
#define XPU_NUMERIC_KIND__VALID		0x00
#define XPU_NUMERIC_KIND__NAN		0x01
#define XPU_NUMERIC_KIND__POS_INF	0x02
#define XPU_NUMERIC_KIND__NEG_INF	0x03

EXTERN_DATA xpu_datum_operators xpu_numeric_ops;

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

#endif /* XPU_NUMERIC_H */
