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

#define PG_NUMERIC_SIGN_MASK		0x0200000000000000UL
#define PG_NUMERIC_MANTISSA_MASK	0x01ffffffffffffffUL
#define PG_NUMERIC_EXPONENT_OFFSET	32

#define PG_NUMERIC_EXPONENT(num)	\
	((int)((num) >> 58) - PG_NUMERIC_EXPONENT_OFFSET)
#define PG_NUMERIC_SIGN(num)		(((num) & PG_NUMERIC_SIGN_MASK) != 0)
#define PG_NUMERIC_MANTISSA(num)	(((num) & PG_NUMERIC_MANTISSA_MASK)
#define PG_NUMERIC_SET(expo,sign,mant)							\
	(((cl_ulong)(expo + PG_NUMERIC_EXPONENT_OFFSET) << 58) |	\
	 ((sign) != 0 ? PG_NUMERIC_SIGN_MASK : 0UL) |				\
	 ((mant) & PG_NUMERIC_MANTISSA_MASK))

static pg_numeric_t
pg_numeric_from_varlena(__private int *errcode, __global varlena *vl_val)
{
	pg_numeric_t	result;

	if (!vl_val)
	{
		result.isnull = true;
		return result;
	}

	/* put here to translate PostgreSQL internal format to pg_numeric_t */

	return result;
}

static pg_numeric_t
pg_numeric_vref(__global kern_data_store *kds,
				__global kern_toastbuf *ktoast,
				__private int *errcode,
				cl_uint colidx,
				cl_uint rowidx)
{
	__global varlena *vl_val = kern_get_datum(kds,ktoast,colidx,rowidx);

	return pg_numeric_from_varlena(errcode, vl_val);
}

static pg_numeric_t
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

static pg_bool_t
pgfn_numeric_isnull(__private int *errcode,
					pg_numeric_t arg)
{
	pg_bool_t	result;

	result.isnull = false;
	result.value  = arg.isnull;
	return result;
}

static pg_bool_t
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

/* pg_int2_t */
#ifndef PG_INT2_TYPE_DEFINED
#define PG_INT2_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int2, cl_short)
#endif
/* pg_int4_t */
#ifndef PG_INT4_TYPE_DEFINED
#define PG_INT4_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int4, cl_int)
#endif
/* pg_int8_t */
#ifndef PG_INT8_TYPE_DEFINED
#define PG_INT8_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int8, cl_long)
#endif
/* pg_float4_t */
#ifndef PG_FLOAT4_TYPE_DEFINED
#define PG_FLOAT4_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(float4, cl_float)
#endif
/* pg_float8_t */
#ifndef PG_FLOAT8_TYPE_DEFINED
#define PG_FLOAT8_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(float8, cl_double)
#endif

static pg_int2_t
pgfn_numeric_int2(__private int *errcode, pg_numeric_t arg)
{}

static pg_int4_t
pgfn_numeric_int4(__private int *errcode, pg_numeric_t arg)
{}

static pg_int8_t
pgfn_numeric_int8(__private int *errcode, pg_numeric_t arg)
{}

static pg_float4_t
pgfn_numeric_float4(__private int *errcode, pg_numeric_t arg)
{}

static pg_float8_t
pgfn_numeric_float8(__private int *errcode, pg_numeric_t arg)
{}

static pg_numeric_t
pgfn_int2_numeric(__private int *errcode, pg_int2_t arg)
{}

static pg_numeric_t
pgfn_int4_numeric(__private int *errcode, pg_int4_t arg)
{}

static pg_numeric_t
pgfn_int8_numeric(__private int *errcode, pg_int8_t arg)
{}

static pg_numeric_t
pgfn_float4_numeric(__private int *errcode, pg_float4_t arg)
{}

static pg_numeric_t
pgfn_float8_numeric(__private int *errcode, pg_float8_t arg)
{}

/*
 * Numeric operator functions
 * ----------------------------------------------------------------
 */
static pg_numeric_t
pgfn_numeric_add(__private int *errcode,
				 pg_numeric_t arg1, pg_numeric_t arg2)
{}

static pg_numeric_t
pgfn_numeric_sub(__private int *errcode,
				 pg_numeric_t arg1, pg_numeric_t arg2)
{}

static pg_numeric_t
pgfn_numeric_mul(__private int *errcode,
				 pg_numeric_t arg1, pg_numeric_t arg2)
{}

static pg_numeric_t
pgfn_numeric_uplus(__private int *errcode, pg_numeric_t arg)
{
	/* return the value as-is */
	return arg;
}

static pg_numeric_t
pgfn_numeric_uminus(__private int *errcode, pg_numeric_t arg)
{
	/* reverse the sign bit */
	arg.value ^= PG_NUMERIC_SIGN_MASK;
	return arg;
}

static pg_numeric_t
pgfn_numeric_abs(__private int *errcode, pg_numeric_t arg)
{
	/* clear the sign bit */
	arg.value &= ~PG_NUMERIC_SIGN_MASK;
	return arg;
}

/*
 * Numeric comparison functions
 * ----------------------------------------------------------------
 */
static int
numeric_cmp(__private cl_int *errcode, pg_numeric_t arg1, pg_numeric_t arg2)
{
	/*
	 * here is the code to compare two numeric values
	 */
}

static pg_bool_t
pgfn_numeric_eq(__private cl_int *errcode,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	result.isnull = false;
	result.value = (numeric_cmp(errcode, arg1, arg2) == 0);
	return result;
}

static pg_bool_t
pgfn_numeric_ne(__private cl_int *errcode,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	result.isnull = false;
	result.value = (numeric_cmp(errcode, arg1, arg2) != 0);
	return result;
}

static pg_bool_t
pgfn_numeric_lt(__private cl_int *errcode,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	result.isnull = false;
	result.value = (numeric_cmp(errcode, arg1, arg2) < 0);
	return result;
}

static pg_bool_t
pgfn_numeric_le(__private cl_int *errcode,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	result.isnull = false;
	result.value = (numeric_cmp(errcode, arg1, arg2) <= 0);
	return result;
}

static pg_bool_t
pgfn_numeric_gt(__private cl_int *errcode,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	result.isnull = false;
	result.value = (numeric_cmp(errcode, arg1, arg2) > 0);
	return result;
}

static pg_bool_t
pgfn_numeric_ge(__private cl_int *errcode,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_bool_t	result;

	result.isnull = false;
	result.value = (numeric_cmp(errcode, arg1, arg2) >= 0);
	return result;
}

static pg_int4_t
pgfn_numeric_cmp(__private cl_int *errcode,
				pg_numeric_t arg1, pg_numeric_t arg2)
{
	pg_int4_t	result;

	result.isnull = false;
	result.value = numeric_cmp(errcode, arg1, arg2);
	return result;
}
#endif /* OPENCL_DEVICE_CODE */
#endif /* OPENCL_NUMERIC_H */
