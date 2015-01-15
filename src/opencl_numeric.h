/*
 * opencl_numeric.h
 *
 * Collection of numeric functions for OpenCL devices
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
#ifndef OPENCL_NUMERIC_H
#define OPENCL_NUMERIC_H
#ifdef OPENCL_DEVICE_CODE


#define CL_CHAR_BIT			8


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

#define PG_MAX_DIGITS		18	/* Max digits of 57 bit mantissa. */
#define PG_MAX_DATA			((PG_MAX_DIGITS + PG_DEC_DIGITS - 1) / \
							 PG_DEC_DIGITS)

struct NumericShort
{
	cl_ushort		n_header;				/* Sign + display scale + weight */
	NumericDigit	n_data[PG_MAX_DATA];	/* Digits */
};

struct NumericLong
{
	cl_ushort		n_sign_dscale;			/* Sign + display scale */
	cl_short		n_weight;				/* Weight of 1st digit	*/
	NumericDigit	n_data[PG_MAX_DATA];	/* Digits */
};

union NumericChoice
{
	cl_ushort			n_header;			/* Header word */
	struct NumericLong	n_long;				/* Long form (4-byte header) */
	struct NumericShort	n_short;			/* Short form (2-byte header) */
};

// struct NumericData
// {
// 	int32		vl_len_;		/* varlena header (do not touch directly!) */
// 	union NumericChoice choice; /* choice of format */
// };


#define NUMERIC_SIGN_MASK	0xC000
#define NUMERIC_POS			0x0000
#define NUMERIC_NEG			0x4000
#define NUMERIC_SHORT		0x8000
#define NUMERIC_NAN			0xC000

#define NUMERIC_FLAGBITS(n)		((n)->n_header & NUMERIC_SIGN_MASK)
#define NUMERIC_IS_NAN(n)		(NUMERIC_FLAGBITS(n) == NUMERIC_NAN)
#define NUMERIC_IS_SHORT(n)		(NUMERIC_FLAGBITS(n) == NUMERIC_SHORT)

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

#define NUMERIC_DIGITS(n)   (NUMERIC_IS_SHORT(n) ? (n)->n_short.n_data : (n)->n_long.n_data)
#define NUMERIC_SIGN(n) 	(NUMERIC_IS_SHORT(n) ? (((n)->n_short.n_header & NUMERIC_SHORT_SIGN_MASK) ? NUMERIC_NEG : NUMERIC_POS) : NUMERIC_FLAGBITS(n))
#define NUMERIC_DSCALE(n)	(NUMERIC_IS_SHORT(n) ? ((n)->n_short.n_header & NUMERIC_SHORT_DSCALE_MASK) >> NUMERIC_SHORT_DSCALE_SHIFT : ((n)->n_long.n_sign_dscale & NUMERIC_DSCALE_MASK))
#define NUMERIC_WEIGHT(n)	(NUMERIC_IS_SHORT(n) ? (((n)->n_short.n_header & NUMERIC_SHORT_WEIGHT_SIGN_MASK ? ~NUMERIC_SHORT_WEIGHT_MASK : 0) | ((n)->n_short.n_header & NUMERIC_SHORT_WEIGHT_MASK)) : ((n)->n_long.n_weight))


/* IEEE 754 FORMAT */
#if 0
#define PG_FLOAT_SIGN_POS	31
#define PG_FLOAT_SIGN_BITS	1
#define PG_FLOAT_EXPO_POS	23
#define PG_FLOAT_EXPO_BITS	8
#define PG_FLOAT_MANT_POS	0
#define PG_FLOAT_MANT_BITS	23

#define PG_DOUBLE_SIGN_POS	63
#define PG_DOUBLE_SIGN_BITS	1
#define PG_DOUBLE_EXPO_POS	52
#define PG_DOUBLE_EXPO_BITS	11
#define PG_DOUBLE_MANT_POS	0
#define PG_DOUBLE_MANT_BITS	52
#endif


/*
 * PG-Strom internal representation of NUMERIC data type
 *
 * Even though the nature of NUMERIC data type is variable-length and error-
 * less mathmatical operation, we assume most of numeric usage can be hosted
 * within 64bit variable. A small number anomaly can be calculated by CPU,
 * so we focus on the major portion of use-cases.
 * Internal data format of numeric is 64-bit integer that is separated to
 * (1) 6bit exponents based on 10, (2) 1bit sign bit, and (3) 57bit mantissa.
 * Function that can handle NUMERIC data type will set StromError_CpuReCheck,
 * if it detects overflow during calculation.
 */
typedef struct {
	cl_ulong	value;
	bool		isnull;
} pg_numeric_t;

#endif	/* OPENCL_DEVICE_CODE */

#define PG_NUMERIC_EXPONENT_BITS	6
#define PG_NUMERIC_EXPONENT_POS		58
#define PG_NUMERIC_EXPONENT_MASK	(((0x1UL << (PG_NUMERIC_EXPONENT_BITS)) - 1) << (PG_NUMERIC_EXPONENT_POS))
#define PG_NUMERIC_EXPONENT_MAX		((1 << ((PG_NUMERIC_EXPONENT_BITS) - 1)) - 1)
#define PG_NUMERIC_EXPONENT_MIN		(0 - (1 << ((PG_NUMERIC_EXPONENT_BITS) - 1)))

#define PG_NUMERIC_SIGN_BITS		1
#define PG_NUMERIC_SIGN_POS			57
#define PG_NUMERIC_SIGN_MASK		(((0x1UL << (PG_NUMERIC_SIGN_BITS)) - 1) << (PG_NUMERIC_SIGN_POS))

#define PG_NUMERIC_MANTISSA_BITS	57
#define PG_NUMERIC_MANTISSA_POS		0
#define PG_NUMERIC_MANTISSA_MASK	(((0x1UL << (PG_NUMERIC_MANTISSA_BITS)) - 1) << (PG_NUMERIC_MANTISSA_POS))
#define PG_NUMERIC_MANTISSA_MAX		((0x1UL << (PG_NUMERIC_MANTISSA_BITS)) - 1)

#define PG_NUMERIC_EXPONENT(num)	((cl_long)(num) >> 58)
#define PG_NUMERIC_SIGN(num)		(((num) & PG_NUMERIC_SIGN_MASK) != 0)
#define PG_NUMERIC_MANTISSA(num)	((num) & PG_NUMERIC_MANTISSA_MASK)
#define PG_NUMERIC_SET(expo,sign,mant)							\
	((cl_ulong)((cl_long)(expo) << 58) |						\
	 ((sign) != 0 ? PG_NUMERIC_SIGN_MASK : 0UL) |				\
	 ((mant) & PG_NUMERIC_MANTISSA_MASK))

#ifdef OPENCL_DEVICE_CODE

static pg_numeric_t
pg_numeric_from_varlena(__private int *errcode, __global varlena *vl_val)
{
	pg_numeric_t		result;
	union NumericChoice	numData;


	if (vl_val == NULL)
	{
		result.isnull = true;
		result.value  = 0;
		return result;
	}

	__global cl_char *pSrc = VARDATA_ANY(vl_val);
	cl_int len = VARSIZE_ANY_EXHDR(vl_val);

	if (sizeof(numData) < len) {
		// Numeric data is too large.
		// PG-Strom numeric type support 18 characters.
		result.isnull = true;
		result.value  = 0;
		*errcode      = StromError_CpuReCheck;
		return result;
	}

	// Once data copy to private memory for alignment.
    // memcpy(&numData, pSrc, len);
	{
		// OpenCL memcpy does not support private memory.
		__private cl_char *dst = (__private cl_char *) &numData;
		__global  cl_char *src = (__global  cl_char *) pSrc;
		for(int i=0; i<len; i++) {
			dst[i] = src[i];
		}
	}

	// Convert PG-Strom numeric type from PostgreSQL numeric type.
	{
		int		     sign	 = NUMERIC_SIGN(&numData);
		int		     expo;
		cl_ulong     mant;
		int 	     weight  = NUMERIC_WEIGHT(&numData);
//		int		     dscale  = NUMERIC_DSCALE(&numData);
		NumericDigit *digits = NUMERIC_DIGITS(&numData);
		int			 offset  = (unsigned long)digits - (unsigned long)&numData;
		int 	     ndigits = (len - offset) / sizeof(NumericDigit);

		int			 i, base;
		cl_ulong	 mantLast;


		// Numeric value is 0, if ndigits is 0. 
		if (ndigits == 0) {
			result.isnull = false;
			result.value  = PG_NUMERIC_SET(0, 0, 0);
			return result;
		}

		// Generate exponent.
		expo = (weight - (ndigits - 1)) * PG_DEC_DIGITS;

		// Generate mantissa.
		mant = 0;
		for (i=0; i<ndigits-1; i++) {
			mant = mant * PG_NBASE + digits[i];
		}

		base     = PG_NBASE;
		mantLast = digits[i];
		for (i=0; i<PG_DEC_DIGITS; i++) {
			if (mantLast % 10 == 0) {
				expo ++;
				base     /= 10;
				mantLast /= 10;
			} else {
				break;
			}
		}

		// overflow check
		if ((mant * base) / base != mant) {
			result.isnull = true;
			result.value  = 0;
			*errcode      = StromError_CpuReCheck;
			return result;
		}

		mant = mant * base + mantLast;


		// Normalize
		while (mant % 10 == 0  &&  expo < PG_NUMERIC_EXPONENT_MAX) {
			mant /= 10;
			expo ++;
		}

		if (PG_NUMERIC_EXPONENT_MAX < expo) {
			// Exponent is overflow.
			int expoDiff = expo - PG_NUMERIC_EXPONENT_MAX;
			int		i;
			ulong	mag;

			if (PG_MAX_DIGITS <= expoDiff) {
				// magnify is overflow
				result.isnull = true;
				result.value  = 0;
				*errcode = StromError_CpuReCheck;
				return result;
			}

			for (i=0, mag=1; i < expoDiff; i++) {
				mag *= 10;
			}

			if ((mant * mag) / mag != mant) {
				result.isnull = true;
				result.value  = 0;
				*errcode      = StromError_CpuReCheck;
				return result;
			}

			expo -= expoDiff;
			mant *= mag;
		}

		// Error check
		if (expo < PG_NUMERIC_EXPONENT_MIN || PG_NUMERIC_EXPONENT_MAX < expo ||
			(mant & ~PG_NUMERIC_MANTISSA_MASK)) {
			result.isnull = true;
			result.value  = 0;
			*errcode = StromError_CpuReCheck;
			return result;
		}

		// Set value to PG_Strom numeric type
		result.isnull = false;
		result.value  = PG_NUMERIC_SET(expo, sign, mant);
	}

	return result;
}

/*
 * pg_numeric_vref
 *
 * It contains special case handling due to internal numeric format.
 * If kds intends to have varlena format (PostgreSQL compatible), it tries
 * to reference varlena variable. Otherwise, in case when attlen > 0, it
 * tries to fetch fixed-length variable.
 */
pg_numeric_t
pg_numeric_vref(__global kern_data_store *kds,
				__global kern_data_store *ktoast,
				__private int *errcode,
				cl_uint colidx,
				cl_uint rowidx)
{
	__global void  *addr = kern_get_datum(kds,ktoast,colidx,rowidx);
	pg_numeric_t	result;

	if (!addr)
		result.isnull = true;
	else if (kds->colmeta[colidx].attlen < 0)
		result = pg_numeric_from_varlena(errcode, (__global varlena *)addr);
	else
	{
		result.isnull = false;
		result.value  = *((__global cl_ulong *) addr);
	}
	return result;
}

/* pg_numeric_vstore() is same as template */
STROMCL_SIMPLE_VARSTORE_TEMPLATE(numeric, cl_ulong)

pg_numeric_t
pg_numeric_param(__global kern_parambuf *kparams,
				 __private int *errcode,
				 cl_uint param_id)
{
	__global varlena *vl_val;
	pg_numeric_t	result;

	if (param_id < kparams->nparams &&
		kparams->poffset[param_id] > 0)
	{
		vl_val = (__global varlena *)
			((__global char *)kparams + kparams->poffset[param_id]);
		/* only uncompressed & inline datum */
		if (VARATT_IS_4B_U(vl_val) || VARATT_IS_1B(vl_val))
			return pg_numeric_from_varlena(errcode, vl_val);

		STROM_SET_ERROR(errcode, StromError_CpuReCheck);
	}
	result.isnull = true;
	return result;
}

pg_bool_t
pgfn_numeric_isnull(__private int *errcode,
					pg_numeric_t arg)
{
	pg_bool_t	result;

	result.isnull = false;
	result.value  = arg.isnull;
	return result;
}

pg_bool_t
pgfn_numeric_isnotnull(__private int *errcode,
					   pg_numeric_t arg)
{
	pg_bool_t	result;

	result.isnull = false;
	result.value = !arg.isnull;
	return result;
}


/* to avoid conflicts with auto-generated data type */
#define PG_NUMERIC_TYPE_DEFINED

/*
 * Numeric format translation functions
 * ----------------------------------------------------------------
 */
static pg_int8_t
numeric_to_integer(__private int *errcode, pg_numeric_t arg, cl_int size)
{
	pg_int8_t	v;


	if (arg.isnull == true) {
		v.isnull = true;
		v.value  = 0;
		return v;
	}

	int	     expo = PG_NUMERIC_EXPONENT(arg.value);
	int		 sign = PG_NUMERIC_SIGN(arg.value);
	cl_ulong mant = PG_NUMERIC_MANTISSA(arg.value);
	
	if (mant == 0) {
		v.isnull = false;
		v.value  = 0;
	}

	int  exp = abs(expo);
	long mag = 1;
	for(int i=0; i<exp; i++) {
		if((mag * 10) < mag) {
			v.isnull = true;
			v.value  = 0;
			return v;
		}
		mag *= 10;
	}

	if (expo < 0) {
		// Round off if exponent is minus.
		mant = (mant + mag/2) / mag;

	} else {
		// Overflow check
		if ((mant * mag) / mag != mant) {
			v.isnull = true;
			v.value  = 0;
			*errcode = StromError_CpuReCheck;
			return v;
		}

		mant *= mag;
	}

	// Overflow check
	int      nbits       = size * CL_CHAR_BIT;
	cl_ulong max_val     = (1UL << (nbits - 1)) - 1;
	cl_ulong abs_min_val = (1UL << (nbits - 1));
	if((sign == 0 && max_val < mant) || (sign != 0 && abs_min_val < mant)) {
		v.isnull = true;
		v.value  = 0;
		*errcode = StromError_CpuReCheck;
		return v;
	}

	v.isnull = false;
	v.value  = (sign == 0) ? mant : (-mant);

	return v;
}

static pg_float8_t
numeric_to_float(__private int *errcode, pg_numeric_t arg)
{
	pg_float8_t	v;


	if (arg.isnull == true) {
		v.isnull = true;
		v.value  = 0;
		return v;
	}


	int   expo = PG_NUMERIC_EXPONENT(arg.value);
	int	  sign = PG_NUMERIC_SIGN(arg.value);
	ulong mant = PG_NUMERIC_MANTISSA(arg.value);

	if (mant == 0) {
		v.isnull = false;
		v.value  = PG_NUMERIC_SET(0, 0, 0);
		return v;
	}


	double	fvalue = (double)mant * exp10((double)expo);

	if (isinf(fvalue) || isnan(fvalue)) {
		v.isnull = true;
		v.value  = 0;
		*errcode = StromError_CpuReCheck;
		return v;
	}

	v.isnull = false;
	v.value  = (sign == 0) ? fvalue : (-fvalue);

	return v;
}



pg_int2_t
pgfn_numeric_int2(__private int *errcode, pg_numeric_t arg)
{
	pg_int2_t v;
	pg_int8_t tmp = numeric_to_integer(errcode, arg, sizeof(v.value));

	v.isnull = tmp.isnull;
	v.value  = tmp.value;

	return v;
}

pg_int4_t
pgfn_numeric_int4(__private int *errcode, pg_numeric_t arg)
{
	pg_int4_t v;
	pg_int8_t tmp = numeric_to_integer(errcode, arg, sizeof(v.value));

	v.isnull = tmp.isnull;
	v.value  = tmp.value;

	return v;
}

pg_int8_t
pgfn_numeric_int8(__private int *errcode, pg_numeric_t arg)
{
	pg_int8_t v;
	return numeric_to_integer(errcode, arg, sizeof(v.value));
}

pg_float4_t
pgfn_numeric_float4(__private int *errcode, pg_numeric_t arg)
{

	pg_float8_t tmp = numeric_to_float(errcode, arg);
	pg_float4_t	v   = { (cl_float)tmp.value, tmp.isnull };

	if (v.isnull == false  &&  isinf(v.value)) {
		v.isnull	= true;
		v.value		= 0;
		*errcode	= StromError_CpuReCheck;
	}

	return v;
}

pg_float8_t
pgfn_numeric_float8(__private int *errcode, pg_numeric_t arg)
{
	return numeric_to_float(errcode, arg);
}



static pg_numeric_t
integer_to_numeric(__private int *errcode, pg_int8_t arg, cl_int size)
{
	pg_numeric_t	v;
	int				sign;
	int				expo;
	cl_ulong		mant;


	if (arg.isnull) {
		v.isnull = true;
		v.value  = 0;
		return v;
	}
		
	if (arg.value == 0) {
		v.isnull = false;
		v.value  = PG_NUMERIC_SET(0, 0, 0);
		return v;
	}

	if (0 <= arg.value) {
		sign = 0;
		mant = arg.value;
	} else {
		sign = 1;
		mant = -arg.value;
	}
	expo = 0;

	// Normalize
	while (mant % 10 == 0  &&  expo < PG_NUMERIC_EXPONENT_MAX) {
		mant /= 10;
		expo ++;
	}

	if(PG_NUMERIC_MANTISSA_BITS < size * CL_CHAR_BIT - 1) {
		// Error check
		if (mant & ~PG_NUMERIC_MANTISSA_MASK) {
			v.isnull = true;
			v.value  = 0;
			*errcode = StromError_CpuReCheck;
			return v;
		}
	}

	v.isnull = false;
	v.value  = PG_NUMERIC_SET(expo, sign, mant);

	return v;
}



static pg_numeric_t
float_to_numeric(__private int *errcode, pg_float8_t arg, int dig)
{
	pg_numeric_t	v;


	if (arg.isnull) {
		v.isnull = true;
		v.value  = 0;
		return v;
	}

	if (isnan(arg.value) || isinf(arg.value)) {
		v.isnull = true;
		v.value  = 0;
		*errcode = StromError_CpuReCheck;
		return v;
	}

	if (arg.value == 0.0) {
		v.isnull = false;
		v.value  = PG_NUMERIC_SET(0, 0, 0);
		return v;
	}


	int		sign;
	double	fval;

	if (0 <= arg.value) {
		sign = 0;
		fval = arg.value;
	} else {
		sign = 1;
		fval = -arg.value;
	}

	int		fexpo = ceil(log10(fval)) + 1;
	double  fmant = fval * (double)exp10((double)(dig - fexpo));
	if(isinf(fmant)) {
		v.isnull = true;
		v.value  = 0;
		*errcode = StromError_CpuReCheck;
		return v;
	}

	int		expo  = fexpo - dig;
	ulong	mant;

	double  thrMax = exp10((double)(dig));
	while(thrMax < fmant) {
		fmant /= 10;
		expo ++;
	}

	double  thrMin = thrMax / 10;
	while(fmant < thrMin) {
		fmant *= 10;
		expo --;
	}

	mant = fmant + 0.5;


	// normalize
	while (mant % 10 == 0  &&  expo < PG_NUMERIC_EXPONENT_MAX) {
		mant /= 10;
		expo ++;
	}

	if (PG_NUMERIC_EXPONENT_MAX < expo) {
		// Exponent is overflow.
		int expoDiff = expo - PG_NUMERIC_EXPONENT_MAX;
		int		i;
		ulong	mag;

		if (PG_MAX_DIGITS <= expoDiff) {
			// magnify is overflow
			v.isnull = true;
			v.value  = 0;
			*errcode = StromError_CpuReCheck;
			return v;
		}

		for (i=0, mag=1; i < expoDiff; i++) {
			mag *= 10;
		}

		if ((mant * mag) / mag != mant) {
			v.isnull = true;
			v.value  = 0;
			*errcode = StromError_CpuReCheck;
			return v;
		}

		expo -= expoDiff;
		mant *= mag;
	}

	// Error check
	if (expo < PG_NUMERIC_EXPONENT_MIN || PG_NUMERIC_EXPONENT_MAX < expo ||
		(mant & ~PG_NUMERIC_MANTISSA_MASK)) {
		v.isnull = true;
		v.value  = 0;
		*errcode = StromError_CpuReCheck;
		return v;
	}

	v.isnull = false;
	v.value  = PG_NUMERIC_SET(expo, sign, mant);

	return v;
}



pg_numeric_t
pgfn_int2_numeric(__private int *errcode, pg_int2_t arg)
{
	pg_int8_t tmp = { arg.value, arg.isnull };
	return integer_to_numeric(errcode, tmp, sizeof(arg.value));
}

pg_numeric_t
pgfn_int4_numeric(__private int *errcode, pg_int4_t arg)
{
	pg_int8_t tmp = { arg.value, arg.isnull };
	return integer_to_numeric(errcode, tmp, sizeof(arg.value));
}

pg_numeric_t
pgfn_int8_numeric(__private int *errcode, pg_int8_t arg)
{
	return integer_to_numeric(errcode, arg, sizeof(arg.value));
}

pg_numeric_t
pgfn_float4_numeric(__private int *errcode, pg_float4_t arg)
{
	pg_float8_t tmp = { (cl_double)arg.value, arg.isnull };
	return float_to_numeric(errcode, tmp, FLT_DIG);
}

pg_numeric_t
pgfn_float8_numeric(__private int *errcode, pg_float8_t arg)
{
	return float_to_numeric(errcode, arg, DBL_DIG);
}



/*
 * Numeric operator functions
 * ----------------------------------------------------------------
 */
pg_numeric_t
pgfn_numeric_uplus(__private int *errcode, pg_numeric_t arg)
{
	/* return the value as-is */
	return arg;
}

pg_numeric_t
pgfn_numeric_uminus(__private int *errcode, pg_numeric_t arg)
{
	/* reverse the sign bit */
	arg.value ^= PG_NUMERIC_SIGN_MASK;
	return arg;
}

pg_numeric_t
pgfn_numeric_abs(__private int *errcode, pg_numeric_t arg)
{
	/* clear the sign bit */
	arg.value &= ~PG_NUMERIC_SIGN_MASK;
	return arg;
}

pg_numeric_t
pgfn_numeric_add(__private int *errcode,
				 pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_numeric_t	v;


	if (arg1.isnull || arg2.isnull) {
		v.isnull = true;
		v.value  = 0;
		return v;
	}

	int			expo1 = PG_NUMERIC_EXPONENT(arg1.value);
	int			sign1 = PG_NUMERIC_SIGN(arg1.value);
	cl_ulong	mant1 = PG_NUMERIC_MANTISSA(arg1.value);

	int			expo2 = PG_NUMERIC_EXPONENT(arg2.value);
	int			sign2 = PG_NUMERIC_SIGN(arg2.value);
	cl_ulong	mant2 = PG_NUMERIC_MANTISSA(arg2.value);

	// Change the number of digits
	if (expo1 != expo2) {
		int 	 expoDiff = abs(expo1 - expo2);
		cl_ulong value	  = (expo1 < expo2) ? (mant2) : (mant1);
		ulong    mag;
		int      i;

		if (PG_MAX_DIGITS <= expoDiff) {
			// magnify is overflow
			v.isnull = true;
			v.value  = 0;
			*errcode = StromError_CpuReCheck;
			return v;
		}

		mag = 1;
		for (i=0; i < expoDiff; i++) {
			mag *= 10;
		}

		// Overflow check
		if ((value * mag) / mag != value) {
			v.isnull = true;
			v.value  = 0;
			*errcode = StromError_CpuReCheck;
			return v;
		}

		if (expo1 < expo2) {
			mant2 = value * mag;
			expo2 = expo1;
		} else {
			mant1 = value * mag;
			expo1 = expo2;
		}
	}

	// Add mantissa 
	if (sign1 != sign2) {
		if (mant1 < mant2) {
			sign1 = sign2;
			mant1 = mant2 - mant1;
		} else {
			mant1 -= mant2;
		}
	} else {
		if ((mant1 + mant2) < mant1) {
			// Overflow
			v.isnull = true;
			v.value  = 0;
			*errcode = StromError_CpuReCheck;
			return v;
		}
		mant1 += mant2;
	}

	// Set 0 if mantissa is 0
	if(mant1 == 0UL) {
		v.isnull = false;
		v.value  = PG_NUMERIC_SET(0, 0, 0);
		return v;
	}

	// Normalize
	while(mant1 % 10 == 0  &&  expo1 < PG_NUMERIC_EXPONENT_MAX) {
		mant1 /= 10;
		expo1 ++;
	}

	// Error check
	if (expo1 < PG_NUMERIC_EXPONENT_MIN || PG_NUMERIC_EXPONENT_MAX < expo1 ||
		(mant1 & ~PG_NUMERIC_MANTISSA_MASK)) {
		v.isnull = true;
		v.value  = 0;
		*errcode = StromError_CpuReCheck;
		return v;
	}

	// Set
	v.isnull = false;
	v.value  = PG_NUMERIC_SET(expo1, sign1, mant1);

	return v;
}


pg_numeric_t
pgfn_numeric_sub(__private int *errcode,
				 pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_numeric_t arg = pgfn_numeric_uminus(errcode, arg2);
	
	return pgfn_numeric_add(errcode, arg1, arg);
}


pg_numeric_t
pgfn_numeric_mul(__private int *errcode,
				 pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_numeric_t	v;

	if (arg1.isnull || arg2.isnull) {
		v.isnull = true;
		v.value  = 0;
		return v;
	}

	int			expo1 = PG_NUMERIC_EXPONENT(arg1.value);
	int			sign1 = PG_NUMERIC_SIGN(arg1.value);
	cl_ulong	mant1 = PG_NUMERIC_MANTISSA(arg1.value);

	int			expo2 = PG_NUMERIC_EXPONENT(arg2.value);
	int			sign2 = PG_NUMERIC_SIGN(arg2.value);
	cl_ulong	mant2 = PG_NUMERIC_MANTISSA(arg2.value);

	// Set 0, if mantissa is 0.
	if (mant1 == 0UL || mant2 == 0UL) {
		v.isnull = false;
		v.value  = PG_NUMERIC_SET(0, 0, 0);
		return v;
	}

	// Calculate exponential
	expo1 += expo2;

	// Calculate sign
	sign1 ^= sign2;
 
	// Calculate mantissa
	if ((mant1 * mant2) / mant2 != mant1) {
		v.isnull = true;
		v.value  = 0;
		*errcode = StromError_CpuReCheck;
		return v;
	}
	mant1 *= mant2;

	// Normalize
	while (mant1 % 10 == 0  &&  expo1 < PG_NUMERIC_EXPONENT_MAX) {
		mant1 /= 10;
		expo1 ++;
	}

	if (PG_NUMERIC_EXPONENT_MAX < expo1) {
		// Exponent is overflow.
		int expoDiff = expo1 - PG_NUMERIC_EXPONENT_MAX;

		ulong mag;
		int	i;

		if (PG_MAX_DIGITS <= expoDiff) {
			// magnify is overflow
			v.isnull = true;
			v.value  = 0;
			*errcode = StromError_CpuReCheck;
			return v;
		}

		for (i=0, mag=1; i < expoDiff; i++) {
			mag *= 10;
		}

		if ((mant1 * mag) / mag != mant1) {
			v.isnull = true;
			v.value  = 0;
			*errcode = StromError_CpuReCheck;
			return v;
		}

		expo1 -= expoDiff;
		mant1 *= mag;
	}

	// Error check
	if (expo1 < PG_NUMERIC_EXPONENT_MIN || PG_NUMERIC_EXPONENT_MAX < expo1 ||
		(mant1 & ~PG_NUMERIC_MANTISSA_MASK)) {
		v.isnull = true;
		v.value  = 0;
		*errcode = StromError_CpuReCheck;
		return v;
	}

	// set
	v.isnull = false;
	v.value  = PG_NUMERIC_SET(expo1, sign1, mant1);

	return v;
}



/*
 * Numeric comparison functions
 * ----------------------------------------------------------------
 */
static int
numeric_cmp(__private cl_int *errcode, pg_numeric_t arg1, pg_numeric_t arg2)
{
	int			expo1 = PG_NUMERIC_EXPONENT(arg1.value);
	int			sign1 = PG_NUMERIC_SIGN(arg1.value);
	cl_ulong	mant1 = PG_NUMERIC_MANTISSA(arg1.value);

	int			expo2 = PG_NUMERIC_EXPONENT(arg2.value);
	int			sign2 = PG_NUMERIC_SIGN(arg2.value);
	cl_ulong	mant2 = PG_NUMERIC_MANTISSA(arg2.value);


	// Ignore exponential and sign, if both mantissa is 0.
	if(mant1 == 0  &&  mant2 == 0) {
		return 0;
	}

	// Compair flag, If sign flag is different.
	if(sign1 != sign2) {
		return sign2 - sign1;
	}

	// Compair the exponential/matissa.
	int		 expoDiff = min(PG_MAX_DIGITS, (int)(abs(expo1 - expo2)));

	cl_ulong mantL, mantR;
	ulong	 mag;
	int		 i, ret;

	if (expo1 < expo2) {
		mantL = mant1;
		mantR = mant2;	// arg2's exponential is large.
	} else {
		mantL = mant2;
		mantR = mant1;	// arg1's exponential is large.
	}

	for (i=0, mag=1; i < expoDiff; i++) {
		mag *= 10;
	}

	if ((mantR * mag) / mag != mantR  ||  mantL < mantR * mag) {
		// mantR * mag is overflow, or larger than mantL
		ret = 1;
	} else if(mantL == mantR * mag) {
		ret = 0;
	} else {
		ret = -1;
	}

	if(expo1 < expo2) {
		ret *= -1;
	}

	if(sign1 != 0) {
		ret *= -1;
	}

	return ret;
}

pg_bool_t
pgfn_numeric_eq(__private cl_int *errcode,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull  ||  arg2.isnull) {
		result.isnull = true;
		result.value  = 0;

	} else {
		result.isnull = false;
		result.value = (numeric_cmp(errcode, arg1, arg2) == 0);
	}

	return result;
}

pg_bool_t
pgfn_numeric_ne(__private cl_int *errcode,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull  ||  arg2.isnull) {
		result.isnull = true;
		result.value  = 0;

	} else {
		result.isnull = false;
		result.value = (numeric_cmp(errcode, arg1, arg2) != 0);
	}

	return result;
}


pg_bool_t
pgfn_numeric_lt(__private cl_int *errcode,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull  ||  arg2.isnull) {
		result.isnull = true;
		result.value  = 0;

	} else {
		result.isnull = false;
		result.value = (numeric_cmp(errcode, arg1, arg2) < 0);
	}

	return result;
}


pg_bool_t
pgfn_numeric_le(__private cl_int *errcode,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull  ||  arg2.isnull) {
		result.isnull = true;
		result.value  = 0;

	} else {
		result.isnull = false;
		result.value = (numeric_cmp(errcode, arg1, arg2) <= 0);
	}

	return result;
}


pg_bool_t
pgfn_numeric_gt(__private cl_int *errcode,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull  ||  arg2.isnull) {
		result.isnull = true;
		result.value  = 0;

	} else {
		result.isnull = false;
		result.value = (numeric_cmp(errcode, arg1, arg2) > 0);
	}

	return result;
}


pg_bool_t
pgfn_numeric_ge(__private cl_int *errcode,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	if (arg1.isnull  ||  arg2.isnull) {
		result.isnull = true;
		result.value  = 0;

	} else {
		result.isnull = false;
		result.value = (numeric_cmp(errcode, arg1, arg2) >= 0);
	}

	return result;
}


pg_int4_t
pgfn_numeric_cmp(__private cl_int *errcode,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_int4_t	result;

	if (arg1.isnull  ||  arg2.isnull) {
		result.isnull = true;
		result.value  = 0;

	} else {
		result.isnull = false;
		result.value = numeric_cmp(errcode, arg1, arg2);
	}

	return result;
}

#endif /* OPENCL_DEVICE_CODE */
#endif /* OPENCL_NUMERIC_H */
