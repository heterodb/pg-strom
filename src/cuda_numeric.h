/*
 * cuda_numeric.h
 *
 * Collection of numeric functions for CUDA GPU devices
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
#ifndef CUDA_NUMERIC_H
#define CUDA_NUMERIC_H

/* PostgreSQL numeric data type */
#if 0
#define PG_DEC_DIGITS		1
#define PG_NBASE			10
typedef cl_char		NumericDigit;
#endif

#if 0
#define PG_DEC_DIGITS		2
#define PG_NBASE			100
typedef cl_char		NumericDigit;
#endif

#if 1
#define PG_DEC_DIGITS		4
#define PG_NBASE			10000
typedef cl_short	NumericDigit;
#endif

#define PG_MAX_DIGITS		40	/* Max digits of 128bit integer */
#define PG_MAX_DATA			(PG_MAX_DIGITS / PG_DEC_DIGITS)

struct NumericShort
{
	cl_ushort		n_header;				/* Sign + display scale + weight */
	NumericDigit	n_data[PG_MAX_DATA];	/* Digits */
};
typedef struct NumericShort	NumericShort;

struct NumericLong
{
	cl_ushort		n_sign_dscale;			/* Sign + display scale */
	cl_short		n_weight;				/* Weight of 1st digit	*/
	NumericDigit	n_data[PG_MAX_DATA];	/* Digits */
};
typedef struct NumericLong	NumericLong;

typedef union
{
	cl_ushort		n_header;			/* Header word */
	NumericLong		n_long;				/* Long form (4-byte header) */
	NumericShort	n_short;			/* Short form (2-byte header) */
} NumericChoice;

struct NumericData
{
 	cl_int			vl_len_;		/* varlena header */
	NumericChoice	choice;			/* payload */
};
typedef struct NumericData	NumericData;

#define NUMERIC_SIGN_MASK	0xC000
#define NUMERIC_POS			0x0000
#define NUMERIC_NEG			0x4000
#define NUMERIC_SHORT		0x8000
#define NUMERIC_NAN			0xC000

#define NUMERIC_FLAGBITS(n_head)	((n_head) & NUMERIC_SIGN_MASK)
#define NUMERIC_IS_NAN(n_head)		(NUMERIC_FLAGBITS(n_head) == NUMERIC_NAN)
#define NUMERIC_IS_SHORT(n_head)	(NUMERIC_FLAGBITS(n_head) == NUMERIC_SHORT)

#define NUMERIC_SHORT_SIGN_MASK			0x2000
#define NUMERIC_SHORT_DSCALE_MASK		0x1F80
#define NUMERIC_SHORT_DSCALE_SHIFT		7
#define NUMERIC_SHORT_DSCALE_MAX		(NUMERIC_SHORT_DSCALE_MASK >> \
										 NUMERIC_SHORT_DSCALE_SHIFT)
#define NUMERIC_SHORT_WEIGHT_SIGN_MASK	0x0040
#define NUMERIC_SHORT_WEIGHT_MASK		0x003F
#define NUMERIC_SHORT_WEIGHT_MAX		NUMERIC_SHORT_WEIGHT_MASK
#define NUMERIC_SHORT_WEIGHT_MIN		(-(NUMERIC_SHORT_WEIGHT_MASK+1))

#define NUMERIC_DSCALE_MASK			0x3FFF

STATIC_INLINE(cl_uint)
NUMERIC_NDIGITS(varlena *numeric)
{
	NumericChoice  *nc = (NumericChoice *)VARDATA_ANY(numeric);
	cl_int			nc_len = VARSIZE_ANY_EXHDR(numeric);
	cl_ushort		n_head = __Fetch(&nc->n_header);

	return (NUMERIC_IS_SHORT(n_head)
			? (nc_len - offsetof(NumericChoice, n_short.n_data))
			: (nc_len - offsetof(NumericChoice, n_long.n_data)))
		/ sizeof(NumericDigit);
}

STATIC_INLINE(NumericDigit *)
NUMERIC_DIGITS(varlena *numeric)		/* may not be aligned */
{
	NumericChoice  *nc = (NumericChoice *)VARDATA_ANY(numeric);
	cl_ushort		n_head = __Fetch(&nc->n_header);

	return NUMERIC_IS_SHORT(n_head) ? nc->n_short.n_data : nc->n_long.n_data;
}

STATIC_INLINE(cl_int)
NUMERIC_SIGN(varlena *numeric)
{
	NumericChoice  *nc = (NumericChoice *)VARDATA_ANY(numeric);
	cl_ushort		n_head = __Fetch(&nc->n_header);

	if (NUMERIC_IS_SHORT(n_head))
		return (n_head & NUMERIC_SHORT_SIGN_MASK) ? NUMERIC_NEG : NUMERIC_POS;
	return NUMERIC_FLAGBITS(n_head);
}

STATIC_INLINE(cl_uint)
NUMERIC_DSCALE(varlena *numeric)
{
	NumericChoice  *nc = (NumericChoice *)VARDATA_ANY(numeric);
	cl_ushort		n_head = __Fetch(&nc->n_header);
	cl_uint			dscale;

	if (NUMERIC_IS_SHORT(n_head))
	{
		dscale = (n_head & NUMERIC_SHORT_DSCALE_MASK)
			>> NUMERIC_SHORT_DSCALE_SHIFT;
	}
	else
	{
		dscale = __Fetch(&nc->n_long.n_sign_dscale) & NUMERIC_DSCALE_MASK;
	}
	return dscale;
}

STATIC_INLINE(cl_int)
NUMERIC_WEIGHT(varlena *numeric)
{
	NumericChoice  *nc = (NumericChoice *)VARDATA_ANY(numeric);
	cl_ushort		n_head = __Fetch(&nc->n_header);
	cl_int			weight;

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

#define FP64_FRAC_MASK		0x000fffffffffffffUL
#define FP64_FRAC_BITS		52
#define FP64_FRAC_VALUE(x)	(((x) & FP64_FRAC_MASK) | (FP64_FRAC_MASK + 1UL))
#define FP64_EXPO_MASK		0x7ff0000000000000UL
#define FP64_EXPO_BITS		11
#define FP64_EXPO_BIAS		1023
#define FP64_EXPO_VALUE(x)	\
	((((x) & FP64_EXPO_MASK) >> FP64_FRAC_BITS) - FP64_EXPO_BIAS)
#define FP64_SIGN_MASK		0x8000000000000000UL
#define FP64_SIGN_VALUE(x)	(((x) & FP64_SIGN_MASK) != 0UL)

/*
 * operations of signed 128bit integer
 */
typedef struct
{
#ifdef HAVE_INT128
	int128		ival;
#else
	cl_ulong	lo;
	cl_long		hi;
#endif
} Int128_t;

STATIC_INLINE(Int128_t)
__Int128_lshift(Int128_t x, cl_uint shift)
{
	Int128_t	res;
#ifdef HAVE_INT128
	res.ival = (x.ival << shift);
#else
	if (shift >= 64)
	{
		if (shift >= 128)
			res.hi = 0;
		else
			res.hi = (x.lo << (shift - 64));
		res.lo = 0;
	}
	else
	{
		res.hi = (x.hi << shift) | (x.lo >> (64 - shift));
		res.lo = (x.lo << shift);
	}
#endif
	return res;
}

STATIC_INLINE(Int128_t)
__Int128_rshift(Int128_t x, cl_uint shift)
{
	Int128_t	res;
#ifdef HAVE_INT128
	res.ival = (x.ival >> shift);
#else
	if (shift >= 64)
	{
		if (shift >= 128)
			res.lo = 0;
		else
			res.lo = (x.hi >> (shift - 64));
		res.hi = 0;
	}
	else
	{
		res.lo = (x.hi >> (64 - shift)) | (x.lo >> shift);
		res.hi = (x.hi >> shift);
	}
#endif
	return res;
}

STATIC_INLINE(Int128_t)
__Int128_add(Int128_t x, cl_long a)
{
	Int128_t	res;
#ifdef HAVE_INT128
	res.ival = x.ival + a;
#else
	asm("add.cc.u64     %0, %2, %3;\n"
		"addc.cc.u64    %1, %4, %5;\n"
		: "=l" (res.lo), "=l" (res.hi)
		: "l" (x.lo), "l" (a),
		  "l" (x.hi), "l" (a < 0 ? ~0UL : 0));
#endif
	return res;
}

STATIC_INLINE(Int128_t)
__Int128_sub(Int128_t x, cl_long a)
{
	return __Int128_add(x, -a);
}

STATIC_INLINE(cl_int)
__Int128_sign(Int128_t x)
{
#ifdef HAVE_INT128
	if (x.ival < 0)
		return -1;
	else if (x.ival > 0)
		return 1;
#else
	if ((x.hi & (1UL<<63)) != 0)
		return -1;	/* negative */
	else if (x.hi != 0 || x.lo != 0)
		return 1;
#endif
	return 0;
}

STATIC_INLINE(Int128_t)
__Int128_inverse(Int128_t x)
{
#ifdef HAVE_INT128
	x.ival = -x.ival;
	return x;
#else
	x.hi = ~x.hi;
	x.lo = ~x.lo;
	return __Int128_add(x,1);
#endif
}

STATIC_INLINE(cl_int)
__Int128_compare(Int128_t x, Int128_t y)
{
#ifdef HAVE_INT128
	if (x.ival > y.ival)
		return 1;
	else if (x.ival < y.ival)
		return -1;
#else
	if (x.hi > y.hi)
		return 1;
	else if (x.hi < y.hi)
		return -1;
	else if (x.lo > y.lo)
		return 1;
	else if (x.lo < y.lo)
		return -1;
#endif
	return 0;
}

STATIC_INLINE(Int128_t)
__Int128_mul(Int128_t x, cl_long a)
{
	Int128_t	res;
#ifdef HAVE_INT128
	res.ival = a * x.ival;
#else
	res.lo = x.lo * a;
	res.hi = __umul64hi(x.lo, a);
	res.hi += x.hi * a;
#endif
	return res;
}

STATIC_INLINE(Int128_t)
__Int128_mad(Int128_t x, cl_long a, cl_long b)
{
	Int128_t	res;
#ifdef HAVE_INT128
	res.ival = a * x.ival + b;
#else
	asm volatile("mad.lo.cc.u64  %0, %2, %3, %4;\n"
				 "madc.hi.u64    %1, %2, %3, %5;\n"
				 : "=l" (res.lo), "=l" (res.hi)
				 : "l" (x.lo), "l" (a),
				 "l" (b), "l" (b < 0 ? ~0UL : 0));
	res.hi += x.hi * a;
#endif
	return res;
}

STATIC_INLINE(Int128_t)
__Int128_div(Int128_t x, cl_long a, cl_long *p_mod)
{
	Int128_t	res;
#ifdef HAVE_INT128
	res.ival = x.ival / a;
	*p_mod = x.ival % a;
#else
	assert(a != 0);
	if (a == 1)
	{
		res = x;
		*p_mod = 0;
	}
	else if (a == -1)
	{
		res = __Int128_inverse(x);
		*p_mod = 0;
	}
	else
	{
		cl_bool	is_negative = false;
		cl_int	remain = 64;

		memset(&res, 0, sizeof(res));
		if (__Int128_sign(x) < 0)
		{
			x = __Int128_inverse(x);
			is_negative = true;
		}

		for (;;)
		{
			cl_ulong	div = (cl_ulong)x.hi / a;
			cl_ulong	mod = (cl_ulong)x.hi % a;
			cl_uint		shift;

			res = __Int128_add(res, div);
			if (remain == 0)
			{
				*p_mod = mod;
				break;
			}
			shift = __clzll(mod);
			if (shift > remain)
				shift = remain;
			x.hi = (mod << shift) | (x.lo >> (64 - shift));
			x.lo <<= shift;
			res = __Int128_lshift(res, shift);
			remain -= shift;
		}

		if (is_negative)
			res = __Int128_inverse(res);
	}
#endif
	return res;
}

/*
 * PG-Strom internal representation of NUMERIC data type
 *
 * Even though the nature of NUMERIC data type is variable-length and error-
 * less mathmatical operation, we assume most of numeric usage can be stored
 * within 128bit fixed-point number; that is compatible to Decimal type in
 * Apache Arrow.
 * Internal data format (pg_numeric_t) has 128bit value and precision (16bit).
 * Function that handles NUMERIC data type may set StromError_CpuReCheck,
 * if it detects overflow during calculation.
 */
typedef struct {
	Int128_t	value;		/* 128bit value */
	cl_short	precision;
	cl_bool		isnull;
} pg_numeric_t;

STATIC_INLINE(pg_numeric_t)
pg_numeric_normalize(pg_numeric_t num)
{
	if (!num.isnull)
	{
#ifdef HAVE_INT128
		if (num.value.ival == 0)
			num.precision = 0;
		else
		{
			while (num.value.ival % 10 == 0)
			{
				num.value.ival /= 10;
				num.precision--;
			}
		}
#else
		/* special case if zero */
		if (num.value.hi == 0 &&
			num.value.lo == 0)
		{
			num.precision = 0;
		}
		else
		{
			Int128_t	temp;
			cl_long		mod;

			for (;;)
			{
				temp = __Int128_div(num.value, 10, &mod);
				if (mod != 0)
					break;
				num.precision--;
				num.value = temp;
			}
		}
#endif
	}
	return num;
}

STATIC_FUNCTION(pg_numeric_t)
pg_numeric_from_varlena(kern_context *kcxt, struct varlena *vl_datum)
{
	pg_numeric_t	result;
	cl_uint			len;

	memset(&result, 0, sizeof(pg_numeric_t));
	if (vl_datum == NULL)
	{
		result.isnull = true;
		return result;
	}
	len = VARSIZE_ANY_EXHDR(vl_datum);
	if (sizeof(NumericChoice) < len)
	{
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		result.isnull = true;
		return result;
	}
	/* construct pg_numeric_t value from PostgreSQL Numeric */
	{
		NumericDigit *digits = NUMERIC_DIGITS(vl_datum); //may be unaligned
		int		weight  = NUMERIC_WEIGHT(vl_datum) + 1;
		int		i, ndigits = NUMERIC_NDIGITS(vl_datum);

		/* Numeric value is 0, if ndigits is 0 */
        if (ndigits == 0)
            return result;
		for (i=0; i < ndigits; i++)
		{
			NumericDigit	dig = __Fetch(digits + i);
			Int128_t		temp;

			temp = __Int128_mad(result.value, PG_NBASE, dig);
			if (__Int128_sign(temp) < 0)
			{
				/* !overflow! */
				STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
                result.isnull = true;
                return result;
			}
			result.value = temp;
		}
		/* sign of the value */
		if (NUMERIC_SIGN(vl_datum) == NUMERIC_NEG)
			result.value = __Int128_inverse(result.value);
		/* precision */
		result.precision = PG_DEC_DIGITS * (ndigits - weight);
	}
	return pg_numeric_normalize(result);
}

/*
 * pg_numeric_to_varlena
 *
 * It transforms a pair of supplied precision and Int128_t value into
 * numeric of PostgreSQL representation; as a form varlena on the vl_buffer.
 * Caller is responsible vl_buffer has enough space more than
 * sizeof(NumericData). Once this function makes a varlena datum on the
 * buffer, it returns total length of the written data.
 */
STATIC_FUNCTION(cl_uint)
pg_numeric_to_varlena(char *vl_buffer, cl_short precision, Int128_t value)
{
	NumericData	   *numData = (NumericData *)vl_buffer;
	NumericLong	   *numBody = &numData->choice.n_long;
	NumericDigit	n_data[PG_MAX_DATA];
	int				ndigits;
	cl_uint			len;
	cl_ushort		n_header = 0;
	cl_bool			is_negative = (__Int128_sign(value) < 0);

	if (is_negative)
		value = __Int128_inverse(value);

	switch (precision % PG_DEC_DIGITS)
	{
		case 3:
		case -1:
			value = __Int128_mul(value, 10);
			precision += 1;
			break;
		case 2:
		case -2:
			value = __Int128_mul(value, 100);
			precision += 2;
			break;
		case 1:
		case -3:
			value = __Int128_mul(value, 1000);
			precision += 3;
			break;
		default:
			/* ok */
			break;
	}
	assert(precision % PG_DEC_DIGITS == 0);

	ndigits = 0;
	while (__Int128_sign(value) != 0)
	{
		cl_long		mod;

		value = __Int128_div(value, PG_NBASE, &mod);
		if (mod == 0)
			precision -= PG_DEC_DIGITS;
		else
		{
			assert(ndigits < PG_MAX_DATA);
			ndigits++;
			n_data[PG_MAX_DATA - ndigits] = mod;
		}
	}
	len = offsetof(NumericData, choice.n_long.n_data[ndigits]);

	if (vl_buffer)
	{
		memcpy(numBody->n_data,
			   n_data + PG_MAX_DATA - ndigits,
			   sizeof(NumericDigit) * ndigits);
		/* other metadata */
		n_header = (Max(precision, 0) & NUMERIC_DSCALE_MASK);
		if (is_negative)
			n_header |= NUMERIC_NEG;
		numBody->n_sign_dscale = n_header;
		numBody->n_weight = ndigits - (precision / PG_DEC_DIGITS) - 1;

		SET_VARSIZE(numData, len);
	}
	return len;
}

#ifdef __CUDACC__

/*
 * pg_numeric_datum_(ref|store)
 *
 * It contains special case handling due to internal numeric format.
 * If kds intends to have varlena format (PostgreSQL compatible), it tries
 * to reference varlena variable. Otherwise, in case when attlen > 0, it
 * tries to fetch fixed-length variable.
 */
STATIC_INLINE(pg_numeric_t)
pg_numeric_datum_ref(kern_context *kcxt, void *addr)
{
	pg_numeric_t	result;

	if (!addr)
		result.isnull = true;
	else
		result = pg_numeric_from_varlena(kcxt, (varlena *) addr);
	return result;
}

STATIC_INLINE(void)
pg_datum_ref(kern_context *kcxt,
			 pg_numeric_t &result, void *addr)
{
	result = pg_numeric_datum_ref(kcxt, addr);
}

STATIC_INLINE(void)
pg_datum_ref_slot(kern_context *kcxt,
				  pg_numeric_t &result,
				  cl_char dclass, Datum datum)
{
	if (dclass == DATUM_CLASS__NULL)
		result = pg_numeric_datum_ref(kcxt, NULL);
	else
	{
		assert(dclass == DATUM_CLASS__NORMAL);
		result = pg_numeric_datum_ref(kcxt, (char *)datum);
	}
}

/* usually, called via pg_datum_ref_arrow() */
STATIC_INLINE(void)
pg_datum_fetch_arrow(kern_context *kcxt,
					 pg_numeric_t &result,
					 kern_colmeta *cmeta,
					 char *base, cl_uint rowidx)
{
	pg_numeric_t	temp;
	void		   *addr;

	addr = kern_fetch_simple_datum_arrow(cmeta,
										 base,
										 rowidx,
										 sizeof(Int128_t));
	if (!addr)
		result.isnull = true;
	else
	{
		memcpy(&temp.value, addr, sizeof(Int128_t));
		temp.precision = cmeta->attopts.decimal.precision;
		temp.isnull = false;

		result = pg_numeric_normalize(temp);
	}
}

STATIC_INLINE(cl_uint)
pg_numeric_array_from_arrow(kern_context *kcxt,
							char *dest,
							kern_colmeta *cmeta,
							char *base, cl_uint start, cl_uint end)
{
	ArrayType	   *res = (ArrayType *)dest;
	cl_uint			nitems = end - start;
	cl_uint			i, sz;
	char		   *nullmap = NULL;
	pg_numeric_t	temp;

	Assert((cl_ulong)res == MAXALIGN(res));
	Assert(start <= end);
	if (cmeta->nullmap_offset == 0)
		sz = ARR_OVERHEAD_NONULLS(1);
	else
		sz = ARR_OVERHEAD_WITHNULLS(1, nitems);

	if (res)
	{
		res->ndim = 1;
		res->dataoffset = (cmeta->nullmap_offset == 0 ? 0 : sz);
		res->elemtype = cmeta->atttypid;
		ARR_DIMS(res)[0] = nitems;
		ARR_LBOUND(res)[0] = 1;

		nullmap = ARR_NULLBITMAP(res);
		Assert(dest + sz == ARR_DATA_PTR(res));
	}

	for (i=0; i < nitems; i++)
	{
		pg_datum_fetch_arrow(kcxt, temp, cmeta, base, start+i);
		if (temp.isnull)
		{
			if (nullmap)
				nullmap[i>>3] &= ~(1<<(i&7));
			else
				Assert(!dest);
		}
		else
		{
			if (nullmap)
				nullmap[i>>3] |= (1<<(i&7));

			sz = TYPEALIGN(cmeta->attalign, sz);
			if (dest)
				sz += pg_numeric_to_varlena(dest + sz,
											temp.precision,
											temp.value);
			else
				sz += pg_numeric_to_varlena(NULL,
											temp.precision,
											temp.value);
		}
	}
	return sz;
}

STATIC_INLINE(cl_int)
pg_datum_store(kern_context *kcxt,
			   pg_numeric_t datum,
			   cl_char &dclass,
			   Datum &value)
{
	char   *res;

	if (datum.isnull)
	{
		dclass = DATUM_CLASS__NULL;
		return 0;
	}
	res = (char *)kern_context_alloc(kcxt, sizeof(struct NumericData));
	if (!res)
	{
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		dclass = DATUM_CLASS__NULL;
		return 0;
	}
	pg_numeric_to_varlena(res,
						  datum.precision,
						  datum.value);
	dclass = DATUM_CLASS__NORMAL;
	value  = PointerGetDatum(res);
	return VARSIZE_ANY(res);
}

STATIC_FUNCTION(pg_numeric_t)
pg_numeric_param(kern_context *kcxt,
				 cl_uint param_id)
{
	kern_parambuf  *kparams = kcxt->kparams;
	varlena		   *vl_val;
	pg_numeric_t	result;

	if (param_id < kparams->nparams &&
		kparams->poffset[param_id] > 0)
	{
		vl_val = (varlena *)((char *)kparams + kparams->poffset[param_id]);
		/* only uncompressed & inline datum */
		if (VARATT_IS_4B_U(vl_val) || VARATT_IS_1B(vl_val))
			return pg_numeric_from_varlena(kcxt, vl_val);

		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
	}
	result.isnull = true;
	return result;
}

STATIC_FUNCTION(cl_uint)
pg_comp_hash(kern_context *kcxt, pg_numeric_t datum)
{
	if (datum.isnull)
		return 0;

	return pg_hash_any((cl_uchar *)&datum.value,
					   offsetof(pg_numeric_t, precision) + sizeof(cl_short));
}
/* to avoid conflicts with auto-generated data type */
#define PG_NUMERIC_TYPE_DEFINED

/*
 * Numeric format translation functions
 * ----------------------------------------------------------------
 */
STATIC_FUNCTION(cl_long)
numeric_to_integer(kern_context *kcxt, pg_numeric_t arg,
				   cl_ulong max_value, cl_bool *p_isnull)
{
	Int128_t	curr = arg.value;
	int			precision = arg.precision;
	bool		is_negative = false;
	cl_long		mod;

	if (__Int128_sign(curr) < 0)
	{
		is_negative = true;
		curr = __Int128_inverse(curr);
	}
	while (precision > 0)
	{
		curr = __Int128_div(curr, 10, &mod);
		precision--;
	}
	while (precision < 0)
	{
		curr = __Int128_mul(curr, 10);
		precision++;
	}
	/* overflow? */
	if (curr.hi != 0 || curr.lo > max_value)
	{
		*p_isnull = true;
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
	}
	return (!is_negative ? (cl_long)curr.lo : -((cl_long)curr.lo));
}

STATIC_FUNCTION(cl_double)
numeric_to_float(kern_context *kcxt, pg_numeric_t arg)
{
	Int128_t	curr = arg.value;
	cl_int		precision = arg.precision;
	bool		is_negative = (__Int128_sign(curr) < 0);
	cl_ulong	fp64_mask = (~FP64_FRAC_MASK << 1);
	cl_ulong	ival;
	cl_int		expo;

	if (is_negative)
		curr = __Int128_inverse(curr);
	while (curr.hi != 0)
	{
		cl_long		mod;

		curr = __Int128_div(curr, 10, &mod);
		precision--;
	}
	ival = curr.lo;
	expo = FP64_FRAC_BITS;
	while (precision > 0)
	{
		if ((ival & fp64_mask) != 0)
		{
			ival /= 10;
			precision--;
		}
		else
		{
			ival *= 2;
			expo--;
		}
	}
	while (precision < 0)
	{
		if ((ival & fp64_mask) == 0)
		{
			ival *= 10;
			precision++;
		}
		else
		{
			ival /= 2;
			expo++;
		}
	}
	/* special case if zero */
	if (ival == 0)
		return 0.0;

	while ((ival & fp64_mask) != 0)
	{
		ival /= 2;
		expo++;
	}
	while ((ival & (FP64_FRAC_MASK + 1)) == 0)
	{
		ival *= 2;
		expo--;
	}
	expo += FP64_EXPO_BIAS;
	if (expo < 0)			/* -infinity */
		return __longlong_as_double(FP64_SIGN_MASK | FP64_EXPO_MASK);
	else if (expo >= 2047)	/* +infinity */
		return __longlong_as_double(FP64_EXPO_MASK);
	else
		return __longlong_as_double((is_negative ? FP64_SIGN_MASK : 0) |
									((cl_ulong)expo << FP64_FRAC_BITS) |
									(ival & FP64_FRAC_MASK));
}

STATIC_FUNCTION(pg_int2_t)
pgfn_numeric_int2(kern_context *kcxt, pg_numeric_t arg)
{
	pg_int2_t	result;

	result.isnull = arg.isnull;
	if (!result.isnull)
	{
		result.value = (cl_short)
			numeric_to_integer(kcxt, arg, SHRT_MAX, &result.isnull);
	}
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_numeric_int4(kern_context *kcxt, pg_numeric_t arg)
{
	pg_int4_t	result;

	result.isnull = arg.isnull;
	if (!result.isnull)
	{
		result.value = (cl_int)
			numeric_to_integer(kcxt, arg, INT_MAX, &result.isnull);
	}
	return result;
}

STATIC_FUNCTION(pg_int8_t)
pgfn_numeric_int8(kern_context *kcxt, pg_numeric_t arg)
{
	pg_int8_t	result;

	result.isnull = arg.isnull;
	if (!result.isnull)
	{
		result.value = (cl_long)
			numeric_to_integer(kcxt, arg, LONG_MAX, &result.isnull);
	}
	return result;
}

STATIC_INLINE(pg_float2_t)
pgfn_numeric_float2(kern_context *kcxt, pg_numeric_t arg)
{
	pg_float2_t	result;

	result.isnull = arg.isnull;
	if (!result.isnull)
	{
		result.value = numeric_to_float(kcxt, arg);
		if (isinf((cl_float)result.value))
		{
			result.isnull = true;
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		}
	}
	return result;
}

STATIC_INLINE(pg_float4_t)
pgfn_numeric_float4(kern_context *kcxt, pg_numeric_t arg)
{
	pg_float4_t	result;

	result.isnull = arg.isnull;
	if (!result.isnull)
	{
		result.value = numeric_to_float(kcxt, arg);
		if (isinf(result.value))
		{
			result.isnull = true;
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		}
	}
	return result;
}

STATIC_INLINE(pg_float8_t)
pgfn_numeric_float8(kern_context *kcxt, pg_numeric_t arg)
{
	pg_float8_t	result;

	result.isnull = arg.isnull;
	if (!result.isnull)
	{
		result.value = numeric_to_float(kcxt, arg);
		if (isinf(result.value))
		{
			result.isnull = true;
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		}
	}
	return result;
}

STATIC_FUNCTION(pg_numeric_t)
integer_to_numeric(kern_context *kcxt, cl_long ival)
{
	pg_numeric_t	result;

	result.isnull = false;
	result.precision = 0;
	result.value.lo = ival;
	result.value.hi = (ival < 0 ? ~0UL : 0);

	return pg_numeric_normalize(result);
}

STATIC_FUNCTION(pg_numeric_t)
float_to_numeric(kern_context *kcxt, cl_double fval)
{
	pg_numeric_t result;
	cl_ulong	ival = __double_as_longlong(fval);
	cl_ulong	frac = FP64_FRAC_VALUE(ival);
	cl_int		expo = FP64_EXPO_VALUE(ival);
	cl_bool		sign = FP64_SIGN_VALUE(ival);
	cl_int		prec, x, y;

	/* special case if zero */
	if ((ival & (FP64_FRAC_MASK | FP64_EXPO_MASK)) == 0)
	{
		memset(&result, 0, sizeof(pg_numeric_t));
		return result;
	}
	frac = FP64_FRAC_VALUE(ival);
	expo = FP64_EXPO_VALUE(ival);
	sign = FP64_SIGN_VALUE(ival);

	/*
	 * fraction must be adjusted by 10^prec / 2^(FP64_FRAC_BITS - expo)
	 * with keeping accuracy (52bit).
	 */
	prec = log10(exp2((double)(FP64_FRAC_BITS - expo)));
	x = prec;
	y = FP64_FRAC_BITS - expo;
	while (y > 0)
	{
		int		width = 64 - __clzll(frac);

		if (width > FP64_FRAC_BITS)
		{
			frac /= 2;
			y--;
		}
		else
		{
			frac *= 10;
			x--;
		}
	}
	while (y < 0)
	{
		int		width = 64 - __clzll(frac);

		if (width > FP64_FRAC_BITS)
		{
			frac *= 2;
			y++;
		}
		else
		{
			frac /= 10;
			x++;
		}
	}
	/*  float64 is not valid on more than 15-digits */
	while (frac > 1000000000000000UL)
	{
		frac /= 10;
		x++;
	}
	result.value.hi = 0;
	result.value.lo = frac;
	result.precision = prec - x;
	if (sign)
		result.value = __Int128_inverse(result.value);
    return pg_numeric_normalize(result);
}

STATIC_FUNCTION(pg_numeric_t)
pgfn_int2_numeric(kern_context *kcxt, pg_int2_t arg)
{
	pg_numeric_t	result;

	if (arg.isnull)
		result.isnull = true;
	else
		result = integer_to_numeric(kcxt, arg.value);
	return result;
}

STATIC_FUNCTION(pg_numeric_t)
pgfn_int4_numeric(kern_context *kcxt, pg_int4_t arg)
{
	pg_numeric_t	result;

	if (arg.isnull)
		result.isnull = true;
	else
		result = integer_to_numeric(kcxt, arg.value);
	return result;
}

STATIC_FUNCTION(pg_numeric_t)
pgfn_int8_numeric(kern_context *kcxt, pg_int8_t arg)
{
	pg_numeric_t	result;

	if (arg.isnull)
		result.isnull = true;
	else
		result = integer_to_numeric(kcxt, arg.value);
	return result;
}

STATIC_INLINE(pg_numeric_t)
pgfn_float2_numeric(kern_context *kcxt, pg_float2_t arg)
{
	pg_numeric_t	result;
	if (arg.isnull)
		result.isnull = true;
	else
		result = float_to_numeric(kcxt, (cl_double)arg.value);
	return result;
}

STATIC_FUNCTION(pg_numeric_t)
pgfn_float4_numeric(kern_context *kcxt, pg_float4_t arg)
{
	pg_numeric_t	result;
	if (arg.isnull)
		result.isnull = true;
	else
		result = float_to_numeric(kcxt, (cl_double)arg.value);
	return result;
}

STATIC_FUNCTION(pg_numeric_t)
pgfn_float8_numeric(kern_context *kcxt, pg_float8_t arg)
{
	pg_numeric_t	result;
	if (arg.isnull)
		result.isnull = true;
	else
		result = float_to_numeric(kcxt, (cl_double)arg.value);
	return result;
}

/*
 * pg_numeric_to_cstring
 */
STATIC_FUNCTION(cl_int)
pg_numeric_to_cstring(kern_context *kcxt, varlena *numeric,
					  char *buf, char *endp)
{
	int			ndigits = NUMERIC_NDIGITS(numeric);
	int			weight  = NUMERIC_WEIGHT(numeric);
	int			sign    = NUMERIC_SIGN(numeric);
	int			dscale  = NUMERIC_DSCALE(numeric);
	int			d;
	char	   *cp = buf;
	NumericDigit *n_data = NUMERIC_DIGITS(numeric);
	NumericDigit  dig, d1 __attribute__ ((unused));

	if (sign == NUMERIC_NEG)
	{
		if (cp >= endp)
			return -1;
		*cp++ = '-';
	}
	/* Output all digits before the decimal point */
	if (weight < 0)
	{
		d = weight + 1;
		if (cp >= endp)
			return -1;
		*cp++ = '0';
	}
	else
	{
		for (d = 0; d <= weight; d++)
		{
			bool		putit __attribute__ ((unused)) = (d > 0);

			if (d < ndigits)
				dig = __Fetch(n_data + d);
			else
				dig = 0;
#if PG_DEC_DIGITS == 4
			d1 = dig / 1000;
			dig -= d1 * 1000;
			putit |= (d1 > 0);
			if (putit)
			{
				if (cp >= endp)
					return -1;
				*cp++ = d1 + '0';
			}
			d1 = dig / 100;
			dig -= d1 * 100;
			putit |= (d1 > 0);
			if (putit)
			{
				if (cp >= endp)
					return -1;
				*cp++ = d1 + '0';
			}
			d1 = dig / 10;
			dig -= d1 * 10;
			putit |= (d1 > 0);
			if (putit)
			{
				if (cp >= endp)
					return -1;
				*cp++ = d1 + '0';
			}
			*cp++ = dig + '0';
#elif PG_DEC_DIGITS == 2
			d1 = dig / 10;
			dig -= d1 * 10;
			if (d1 > 0 || d > 0)
			{
				if (cp >= endp)
					return -1;
				*cp++ = d1 + '0';
			}
			if (cp >= endp)
				return -1;
			*cp++ = dig + '0';
#elif PG_DEC_DIGITS == 1
			if (cp >= endp)
				return -1;
			*cp++ = dig + '0';
#else
#error unsupported NBASE
#endif
		}
	}

	if (dscale > 0)
	{
		if (cp >= endp)
			return -1;
		*cp++ = '.';
		for (int i = 0; i < dscale; d++, i += PG_DEC_DIGITS)
		{
#if PG_DEC_DIGITS == 4
			if (cp + 4 > endp)
				return -1;
			d1 = dig / 1000;
			dig -= d1 * 1000;
			*cp++ = d1 + '0';
			d1 = dig / 100;
			dig -= d1 * 100;
			*cp++ = d1 + '0';
			d1 = dig / 10;
			dig -= d1 * 10;
			*cp++ = d1 + '0';
			*cp++ = dig + '0';
#elif PG_DEC_DIGITS == 2
			if (cp + 2 > endp)
				return -1;
			d1 = dig / 10;
			dig -= d1 * 10;
			*cp++ = d1 + '0';
			*cp++ = dig + '0';
#elif PG_DEC_DIGITS == 1
			if (cp >= endp)
				return -1;
			*cp++ = dig + '0';
#else
#error unsupported NBASE
#endif
		}
	}
	return (cl_int)(cp - buf);
}

/*
 * Numeric operator functions
 * ----------------------------------------------------------------
 */
STATIC_FUNCTION(pg_numeric_t)
pgfn_numeric_uplus(kern_context *kcxt, pg_numeric_t arg)
{
	return arg;
}

STATIC_FUNCTION(pg_numeric_t)
pgfn_numeric_uminus(kern_context *kcxt, pg_numeric_t arg)
{
	if (!arg.isnull)
		arg.value = __Int128_inverse(arg.value);
	return arg;
}

STATIC_FUNCTION(pg_numeric_t)
pgfn_numeric_abs(kern_context *kcxt, pg_numeric_t arg)
{
	if (!arg.isnull)
	{
		if (__Int128_sign(arg.value) < 0)
			arg.value = __Int128_inverse(arg.value);
	}
	return arg;
}

STATIC_FUNCTION(pg_numeric_t)
pgfn_numeric_add(kern_context *kcxt,
				 pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_numeric_t result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (result.isnull)
		return result;

	while (arg1.precision > arg2.precision)
	{
		arg2.value = __Int128_mul(arg2.value, 10);
		arg2.precision++;
	}
	while (arg1.precision < arg2.precision)
	{
		arg1.value = __Int128_mul(arg1.value, 10);
		arg1.precision++;
	}
	asm volatile("add.cc.u64     %0, %2, %3;\n"
				 "addc.u64       %1, %4, %5;\n"
				 : "=l" (result.value.lo),
				   "=l" (result.value.hi)
				 : "l" (arg1.value.lo),
				   "l" (arg2.value.lo),
				   "l" (arg1.value.hi),
				   "l" (arg2.value.hi));
	result.precision = arg1.precision;

	return pg_numeric_normalize(result);
}

STATIC_FUNCTION(pg_numeric_t)
pgfn_numeric_sub(kern_context *kcxt,
				 pg_numeric_t arg1, pg_numeric_t arg2)
{
	arg2.value = __Int128_inverse(arg2.value);

	return pgfn_numeric_add(kcxt, arg1, arg2);
}

STATIC_FUNCTION(pg_numeric_t)
pgfn_numeric_mul(kern_context *kcxt,
				 pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_numeric_t result;
	cl_bool		is_negative_1 = false;
	cl_bool		is_negative_2 = false;

	result.isnull = arg1.isnull | arg2.isnull;
	if (result.isnull)
		return result;
	if (__Int128_sign(arg1.value) < 0)
	{
		is_negative_1 = true;
		arg1.value = __Int128_inverse(arg1.value);
	}
	if (__Int128_sign(arg2.value) < 0)
	{
		is_negative_2 = true;
		arg2.value = __Int128_inverse(arg2.value);
	}

	if ((arg1.value.hi != 0 && arg2.value.hi != 0) ||
		__umul64hi(arg1.value.hi, arg2.value.lo) != 0 ||
		__umul64hi(arg1.value.lo, arg2.value.hi) != 0)
	{
		STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		result.isnull = true;
		return result;
	}
	result.value.lo = arg1.value.lo * arg2.value.lo;
	result.value.hi = __umul64hi(arg1.value.lo, arg2.value.lo);
	result.value.hi += arg1.value.hi * arg2.value.lo;
	result.value.hi += arg1.value.lo * arg2.value.hi;

	if (is_negative_1 != is_negative_2)
		result.value = __Int128_inverse(result.value);
	result.precision = arg1.precision + arg2.precision;
	return pg_numeric_normalize(result);
}

/*
 * Numeric comparison functions
 * ----------------------------------------------------------------
 */
STATIC_FUNCTION(int)
numeric_cmp(kern_context *kcxt, pg_numeric_t arg1, pg_numeric_t arg2)
{
	int		sign1 = __Int128_sign(arg1.value);
	int		sign2 = __Int128_sign(arg2.value);

	/* shortcut for obvious cases */
	if (sign1 > sign2)
		return 1;
	else if (sign1 < sign2)
		return -1;
	/* ok, both of arg1 and arg2 is not zero, and have same sign */
	while (arg1.precision > arg2.precision)
	{
		arg2.value = __Int128_mul(arg2.value, 10);
		arg2.precision++;
	}
	while (arg1.precision < arg2.precision)
	{
		arg1.value = __Int128_mul(arg1.value, 10);
		arg1.precision++;
	}
	return __Int128_compare(arg1.value, arg2.value);
}

STATIC_FUNCTION(pg_bool_t)
pgfn_numeric_eq(kern_context *kcxt,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (numeric_cmp(kcxt, arg1, arg2) == 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_numeric_ne(kern_context *kcxt,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (numeric_cmp(kcxt, arg1, arg2) != 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_numeric_lt(kern_context *kcxt,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (numeric_cmp(kcxt, arg1, arg2) < 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_numeric_le(kern_context *kcxt,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (numeric_cmp(kcxt, arg1, arg2) <= 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_numeric_gt(kern_context *kcxt,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (numeric_cmp(kcxt, arg1, arg2) > 0);
	return result;
}

STATIC_FUNCTION(pg_bool_t)
pgfn_numeric_ge(kern_context *kcxt,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (numeric_cmp(kcxt, arg1, arg2) >= 0);
	return result;
}

STATIC_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt,
				  pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = numeric_cmp(kcxt, arg1, arg2);
	return result;
}

#endif /* __CUDACC__ */
#endif /* CUDA_NUMERIC_H */
