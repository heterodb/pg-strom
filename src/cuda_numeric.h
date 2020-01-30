/*
 * cuda_numeric.h
 *
 * Collection of numeric functions for CUDA GPU devices
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

/*
 * operations of signed 128bit integer
 *
 * host code can use int128 operators supported by compiler (gcc/nvcc),
 * however, device code needs to implement own int128 operations by itself.
 */
#ifndef HAVE_INT128
#ifndef __CUDA_ARCH__
#define HAVE_INT128		1
typedef __int128		int128;
#endif  /* __CUDA_ARCH__ */
#endif  /* HAVE_INT128 */

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
 * Internal data format (pg_numeric_t) has 128bit value and weight (16bit).
 * Function that handles NUMERIC data type can aise CPU-FALLBACKed error,
 * if it detects overflow during calculation.
 */
#ifndef PG_NUMERIC_TYPE_DEFINED
#define PG_NUMERIC_TYPE_DEFINED
typedef struct {
	Int128_t	value;		/* 128bit value */
	cl_short	weight;
	cl_bool		isnull;
} pg_numeric_t;
STROMCL_EXTERNAL_VARREF_TEMPLATE(numeric)
STROMCL_EXTERNAL_COMP_HASH_TEMPLATE(numeric)
STROMCL_EXTERNAL_ARROW_TEMPLATE(numeric)
#endif /* PG_NUMERIC_TYPE_DEFINED */

/* convert pg_numeric_t <-> convert */
PUBLIC_FUNCTION(cl_uint)
pg_numeric_to_varlena(char *vl_buffer, cl_short precision, Int128_t value);
PUBLIC_FUNCTION(pg_numeric_t)
pg_numeric_from_varlena(kern_context *kcxt, struct varlena *vl_datum);

#ifdef __CUDACC__
/* numeric cast functions */
DEVICE_FUNCTION(pg_int2_t)
pgfn_numeric_int2(kern_context *kcxt, pg_numeric_t arg);
DEVICE_FUNCTION(pg_int4_t)
pgfn_numeric_int4(kern_context *kcxt, pg_numeric_t arg);
DEVICE_FUNCTION(pg_int8_t)
pgfn_numeric_int8(kern_context *kcxt, pg_numeric_t arg);
DEVICE_FUNCTION(pg_float2_t)
pgfn_numeric_float2(kern_context *kcxt, pg_numeric_t arg);
DEVICE_FUNCTION(pg_float4_t)
pgfn_numeric_float4(kern_context *kcxt, pg_numeric_t arg);
DEVICE_FUNCTION(pg_float8_t)
pgfn_numeric_float8(kern_context *kcxt, pg_numeric_t arg);
DEVICE_FUNCTION(pg_numeric_t)
integer_to_numeric(kern_context *kcxt, cl_long ival);
DEVICE_FUNCTION(pg_numeric_t)
float_to_numeric(kern_context *kcxt, cl_double fval);
DEVICE_FUNCTION(pg_numeric_t)
pgfn_int2_numeric(kern_context *kcxt, pg_int2_t arg);
DEVICE_FUNCTION(pg_numeric_t)
pgfn_int4_numeric(kern_context *kcxt, pg_int4_t arg);
DEVICE_FUNCTION(pg_numeric_t)
pgfn_int8_numeric(kern_context *kcxt, pg_int8_t arg);
DEVICE_FUNCTION(pg_numeric_t)
pgfn_float2_numeric(kern_context *kcxt, pg_float2_t arg);
DEVICE_FUNCTION(pg_numeric_t)
pgfn_float4_numeric(kern_context *kcxt, pg_float4_t arg);
DEVICE_FUNCTION(pg_numeric_t)
pgfn_float8_numeric(kern_context *kcxt, pg_float8_t arg);
DEVICE_FUNCTION(cl_int)
pg_numeric_to_cstring(kern_context *kcxt, varlena *numeric,
					  char *buf, char *endp);
/*
 * Numeric operator functions
 */
DEVICE_FUNCTION(pg_numeric_t)
pgfn_numeric_uplus(kern_context *kcxt, pg_numeric_t arg);
DEVICE_FUNCTION(pg_numeric_t)
pgfn_numeric_uminus(kern_context *kcxt, pg_numeric_t arg);
DEVICE_FUNCTION(pg_numeric_t)
pgfn_numeric_abs(kern_context *kcxt, pg_numeric_t arg);
DEVICE_FUNCTION(pg_numeric_t)
pgfn_numeric_add(kern_context *kcxt,
				 pg_numeric_t arg1, pg_numeric_t arg2);
DEVICE_FUNCTION(pg_numeric_t)
pgfn_numeric_sub(kern_context *kcxt,
				 pg_numeric_t arg1, pg_numeric_t arg2);
DEVICE_FUNCTION(pg_numeric_t)
pgfn_numeric_mul(kern_context *kcxt,
				 pg_numeric_t arg1, pg_numeric_t arg2);
/*
 * Numeric comparison functions
 */
DEVICE_FUNCTION(pg_bool_t)
pgfn_numeric_eq(kern_context *kcxt,
				pg_numeric_t arg1, pg_numeric_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_numeric_ne(kern_context *kcxt,
				pg_numeric_t arg1, pg_numeric_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_numeric_lt(kern_context *kcxt,
				pg_numeric_t arg1, pg_numeric_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_numeric_le(kern_context *kcxt,
				pg_numeric_t arg1, pg_numeric_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_numeric_gt(kern_context *kcxt,
				pg_numeric_t arg1, pg_numeric_t arg2);
DEVICE_FUNCTION(pg_bool_t)
pgfn_numeric_ge(kern_context *kcxt,
				pg_numeric_t arg1, pg_numeric_t arg2);
DEVICE_FUNCTION(pg_int4_t)
pgfn_type_compare(kern_context *kcxt,
				  pg_numeric_t arg1, pg_numeric_t arg2);
#endif /* __CUDACC__ */
#endif /* CUDA_NUMERIC_H */
