/*
 * cuda_curand.h
 *
 * Switch to link cuRand library for random number generation in GPU kernel.
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
#ifndef CUDA_CURAND_H
#define CUDA_CURAND_H
#ifdef	PGSTROM_BUILD_WRAPPER
namespace __curand
{
#include <curand_kernel.h>
}
typedef __curand::curandStateXORWOW_t	curandState_t;
typedef __curand::curandStateXORWOW_t	curandState;
#else
/**
 * Use CURAND XORWOW as default
 */
struct curandStateXORWOW {
	unsigned int d, v[5];
	int boxmuller_flag;
	int boxmuller_flag_double;
	float boxmuller_extra;
	double boxmuller_extra_double;
};
typedef struct curandStateXORWOW		curandState_t;
typedef struct curandStateXORWOW		curandState;
#endif	/* PGSTROM_BUILD_WRAPPER */
#define DEVICE_FUNCTION(RET_TYPE)	extern "C" __device__ RET_TYPE

DEVICE_FUNCTION(void)
curand_init(unsigned long long seed,
			unsigned long long subsequence,
			unsigned long long offset,
			curandState_t *state)
#ifdef PGSTROM_BUILD_WRAPPER
{
	__curand::curand_init(seed, subsequence, offset, state);
}
#else
;
#endif

DEVICE_FUNCTION(unsigned int)
curand(curandState_t *state)
#ifdef PGSTROM_BUILD_WRAPPER
{
	return __curand::curand(state);
}
#else
;
#endif

DEVICE_FUNCTION(void)
skipahead(unsigned long long n, curandState_t *state)
#ifdef PGSTROM_BUILD_WRAPPER
{
	__curand::skipahead(n, state);
}
#else
;
#endif

DEVICE_FUNCTION(void)
skipahead_sequence(unsigned long long n, curandState_t *state)
#ifdef PGSTROM_BUILD_WRAPPER
{
	__curand::skipahead_sequence(n, state);
}
#else
;
#endif

/* uniform */
DEVICE_FUNCTION(float)
curand_uniform(curandState_t *state)
#ifdef PGSTROM_BUILD_WRAPPER
{
	return __curand::curand_uniform(state);
}
#else
;
#endif

DEVICE_FUNCTION(double)
curand_uniform_double(curandState_t *state)
#ifdef PGSTROM_BUILD_WRAPPER
{
	return __curand::curand_uniform_double(state);
}
#else
;
#endif

/* normal */
DEVICE_FUNCTION(float)
curand_normal(curandState_t *state)
#ifdef PGSTROM_BUILD_WRAPPER
{
	return __curand::curand_normal(state);
}
#else
;
#endif

DEVICE_FUNCTION(double)
curand_normal_double(curandState_t *state)
#ifdef PGSTROM_BUILD_WRAPPER
{
	return __curand::curand_normal_double(state);
}
#else
;
#endif

DEVICE_FUNCTION(float2)
curand_normal2(curandState_t *state)
#ifdef PGSTROM_BUILD_WRAPPER
{
	return __curand::curand_normal2(state);
}
#else
;
#endif

DEVICE_FUNCTION(double2)
curand_normal2_double(curandState_t *state)
#ifdef PGSTROM_BUILD_WRAPPER
{
	return __curand::curand_normal2_double(state);
}
#else
;
#endif

/* lognormal */
DEVICE_FUNCTION(float)
curand_log_normal(curandState_t *state, float mean, float stddev)
#ifdef PGSTROM_BUILD_WRAPPER
{
	return __curand::curand_log_normal(state, mean, stddev);
}
#else
;
#endif

DEVICE_FUNCTION(double)
curand_log_normal_double(curandState_t *state, float mean, float stddev)
#ifdef PGSTROM_BUILD_WRAPPER
{
	return __curand::curand_log_normal_double(state, mean, stddev);
}
#else
;
#endif

DEVICE_FUNCTION(float2)
curand_log_normal2(curandState_t *state, float mean, float stddev)
#ifdef PGSTROM_BUILD_WRAPPER
{
	return __curand::curand_log_normal2(state, mean, stddev);
}
#else
;
#endif

DEVICE_FUNCTION(double2)
curand_log_normal2_double(curandState_t *state, float mean, float stddev)
#ifdef PGSTROM_BUILD_WRAPPER
{
	return __curand::curand_log_normal2_double(state, mean, stddev);
}
#else
;
#endif

/* poisson */
DEVICE_FUNCTION(unsigned int)
curand_poisson(curandState_t *state, double lambda)
#ifdef PGSTROM_BUILD_WRAPPER
{
	return __curand::curand_poisson(state, lambda);
}
#else
;
#endif
#undef	DEVICE_FUNCTION
#endif	/* CUDA_CURAND_H */
