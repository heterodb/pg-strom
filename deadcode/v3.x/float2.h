/*
 * float2.h
 *
 * Definition of half-precision floating-point
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef FLOAT2_H
#define FLOAT2_H
#include <stdint.h>

typedef uint16_t			half_t;

/* parameters of floating-point */
#define FP16_FRAC_BITS		(10)
#define FP16_EXPO_BITS		(5)
#define FP16_EXPO_MIN		(-14)
#define FP16_EXPO_MAX		(15)
#define FP16_EXPO_BIAS		(15)

#define FP32_FRAC_BITS		(23)
#define FP32_EXPO_BITS		(8)
#define FP32_EXPO_MIN		(-126)
#define FP32_EXPO_MAX		(127)
#define FP32_EXPO_BIAS		(127)

#define FP64_FRAC_BITS		(52)
#define FP64_EXPO_BITS		(11)
#define FP64_EXPO_MIN		(-1022)
#define FP64_EXPO_MAX		(1023)
#define FP64_EXPO_BIAS		(1023)

/* int/float reinterpret functions */
static inline double
long_as_double(uint64_t ival)
{
	union {
		uint64_t	ival;
		double		fval;
	} datum;
	datum.ival = ival;
	return datum.fval;
}

static inline uint64_t
double_as_long(double fval)
{
	union {
		uint64_t	ival;
		double		fval;
	} datum;
	datum.fval = fval;
	return datum.ival;
}

static inline float
int_as_float(uint32_t ival)
{
	union {
		uint32_t	ival;
		float		fval;
	} datum;
	datum.ival = ival;
	return datum.fval;
}

static inline uint32_t
float_as_int(float fval)
{
	union {
		uint32_t	ival;
		float		fval;
	} datum;
	datum.fval = fval;
	return datum.ival;
}

/*
 * cast functions across floating point
 */
static inline half_t
fp32_to_fp16(float value)
{
	uint32_t	x = float_as_int(value);
	uint32_t	u = (x & 0x7fffffffU);
	uint32_t	sign = ((x >> 16U) & 0x8000U);
	uint32_t	remainder;
	uint32_t	result = 0;

	if (u >= 0x7f800000U)
	{
		/* NaN/+Inf/-Inf */
		remainder = 0U;
		result = ((u == 0x7f800000U) ? (sign | 0x7c00U) : 0x7fffU);
    }
	else if (u > 0x477fefffU)
	{
		/* Overflows */
		remainder = 0x80000000U;
		result = (sign | 0x7bffU);
    }
	else if (u >= 0x38800000U)
	{
		/* Normal numbers */
		remainder = u << 19U;
		u -= 0x38000000U;
		result = (sign | (u >> 13U));
    }
	else if (u < 0x33000001U)
	{
		/* +0/-0 */
		remainder = u;
		result = sign;
    }
	else
	{
		/* Denormal numbers */
		const uint32_t	exponent = u >> 23U;
		const uint32_t	shift = 0x7eU - exponent;
		uint32_t		mantissa = (u & 0x7fffffU) | 0x800000U;

		remainder = mantissa << (32U - shift);
		result = (sign | (mantissa >> shift));
	}

	if ((remainder > 0x80000000U) ||
		((remainder == 0x80000000U) && ((result & 0x1U) != 0U)))
		result++;

	return result;
}

static inline half_t
fp64_to_fp16(double fval)
{
	return fp32_to_fp16((float)fval);
}

static inline float
fp16_to_fp32(half_t fp16val)
{
	uint32_t	sign = ((uint32_t)(fp16val & 0x8000) << 16);
	int32_t		expo = ((fp16val & 0x7c00) >> 10);
	int32_t		frac = ((fp16val & 0x03ff));
	uint32_t	result;

	if (expo == 0x1f)
	{
		if (frac == 0)
			result = (sign | 0x7f800000);	/* +/-Infinity */
		else
			result = 0xffffffff;			/* NaN */
	}
	else if (expo == 0 && frac == 0)
		result = sign;						/* +/-0.0 */
	else
	{
		if (expo == 0)
		{
			expo = FP16_EXPO_MIN;
			while ((frac & 0x400) == 0)
			{
				frac <<= 1;
				expo--;
			}
			frac &= 0x3ff;
		}
		else
			expo -= FP16_EXPO_BIAS;

		expo += FP32_EXPO_BIAS;

		result = (sign | (expo << FP32_FRAC_BITS) | (frac << 13));
	}
	return int_as_float(result);
}

static inline double
fp16_to_fp64(half_t fp16val)
{
	uint64_t	sign = ((uint64_t)(fp16val & 0x8000) << 48);
	int64_t		expo = ((fp16val & 0x7c00) >> 10);
	int64_t		frac = ((fp16val & 0x03ff));
	uint64_t	result;

	if (expo == 0x1f)
	{
		if (frac == 0)
			result = (sign | 0x7f800000);	/* +/-Infinity */
		else
			result = 0xffffffff;			/* NaN */
	}
	else if (expo == 0 && frac == 0)
		result = sign;						/* +/-0.0 */
	else
	{
		if (expo == 0)
		{
			expo = FP16_EXPO_MIN;
			while ((frac & 0x400) == 0)
			{
				frac <<= 1;
				expo--;
			}
			frac &= 0x3ff;
		}
		else
			expo -= FP16_EXPO_BIAS;

		expo += FP64_EXPO_BIAS;
		result = (sign | (expo << FP64_FRAC_BITS) | (frac << 42));
	}
	return long_as_double(result);
}

#endif	/* FLOAT2_H */
