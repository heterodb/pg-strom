/*
 * select_into.c
 *
 * Routines related to SELECT INTO Direct mode
 * ----
 * Copyright 2011-2026 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2026 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

void
pgstrom_init_select_into(void)
{
}

/* ----------------------------------------------------------------
 *
 * select_into_form_heap_tuple
 *
 * ----------------------------------------------------------------
 */
#define __SELECT_INTO_FORM__FMINMAX_INT_TEMPLATE(LABEL,CHECKER_MACRO)	\
	INLINE_FUNCTION(int)												\
	__select_into_form__fminmax_##LABEL(gpuClient *gclient,				\
										const kern_colmeta *cmeta,		\
										char *dest,						\
										const char *source)				\
	{																	\
		kagg_state__pminmax_int64_packed *state							\
			= (kagg_state__pminmax_int64_packed *)source;				\
																		\
		assert(cmeta->attlen == sizeof(LABEL##_t) && cmeta->attbyval);	\
		if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)			\
		{																\
			int64_t		ival = state->value;							\
																		\
			if (CHECKER_MACRO)											\
			{															\
				gpuClientELog(gclient, "min/max(" #LABEL ") out of range");	\
				return -1;												\
			}															\
			if (dest)													\
				*((LABEL##_t *)dest) = (LABEL##_t)ival;					\
			return cmeta->attlen;										\
		}																\
		return 0;														\
	}
__SELECT_INTO_FORM__FMINMAX_INT_TEMPLATE(int8, ival < SCHAR_MIN || ival > SCHAR_MAX)
__SELECT_INTO_FORM__FMINMAX_INT_TEMPLATE(int16,ival < SHRT_MIN  || ival > SHRT_MAX)
__SELECT_INTO_FORM__FMINMAX_INT_TEMPLATE(int32,ival < INT_MIN   || ival > INT_MAX)
__SELECT_INTO_FORM__FMINMAX_INT_TEMPLATE(int64,false)
#undef __SELECT_INTO_FORM__FMINMAX_INT_TEMPLATE

#define __SELECT_INTO_FORM__FMINMAX_FP_TEMPLATE(LABEL,TYPE)				\
	INLINE_FUNCTION(int)												\
	__select_into_form__fminmax_##LABEL(gpuClient *gclient,				\
										const kern_colmeta *cmeta,		\
										char *dest,						\
										const char *source)				\
	{																	\
		kagg_state__pminmax_fp64_packed *state							\
			= (kagg_state__pminmax_fp64_packed *)source;				\
																		\
		assert(cmeta->attlen == sizeof(TYPE) && cmeta->attbyval);		\
		if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)			\
		{																\
			if (dest)													\
				*((TYPE *)dest) = (TYPE)state->value;					\
			return cmeta->attlen;										\
		}																\
		return 0;														\
	}
__SELECT_INTO_FORM__FMINMAX_FP_TEMPLATE(fp16, float2_t)
__SELECT_INTO_FORM__FMINMAX_FP_TEMPLATE(fp32, float4_t)
__SELECT_INTO_FORM__FMINMAX_FP_TEMPLATE(fp64, float8_t)
#undef __SELECT_INTO_FORM__FMINMAX_FP_TEMPLATE

INLINE_FUNCTION(int)
__select_into_form__fminmax_numeric(gpuClient *gclient,
									const kern_colmeta *cmeta,
									char *dest,
									const char *source)
{
	kagg_state__pminmax_fp64_packed *state
		= (kagg_state__pminmax_fp64_packed *)source;
	assert(cmeta->attlen < 0 && !cmeta->attbyval);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)
	{
		uint8_t		kind;
		int16_t		weight;
		int128_t	value;

		__decimal_from_float8(state->value,
							  &kind,
							  &weight,
							  &value);
		return __decimal_to_varlena(dest, kind, weight, value);
	}
	return 0;
}

INLINE_FUNCTION(int)
__select_into_form__fsum_int(gpuClient *gclient,
							 const kern_colmeta *cmeta,
							 char *dest,
							 const char *source)
{
	kagg_state__psum_int_packed *state
		= (kagg_state__psum_int_packed *)source;
	assert(cmeta->attlen == sizeof(int64_t));
	if (state->nitems == 0)
		return 0;
	if (dest)
		*((int64_t *)dest) = state->sum;
	return sizeof(int64_t);
}

INLINE_FUNCTION(int)
__select_into_form__fsum_int64(gpuClient *gclient,
							   const kern_colmeta *cmeta,
							   char *dest,
							   const char *source)
{
	kagg_state__psum_int_packed *state
        = (kagg_state__psum_int_packed *)source;
	assert(cmeta->attlen < 0 && !cmeta->attbyval);
	if (state->nitems == 0)
		return 0;
	return __decimal_to_varlena(dest, XPU_NUMERIC_KIND__VALID, 0, state->sum);
}

INLINE_FUNCTION(int)
__select_into_form__fsum_fp32(gpuClient *gclient,
							  const kern_colmeta *cmeta,
							  char *dest,
							  const char *source)
{
	kagg_state__psum_fp_packed *state
		= (kagg_state__psum_fp_packed *)source;
	assert(cmeta->attlen == sizeof(float4_t));
	if (state->nitems == 0)
		return 0;
	if (dest)
		*((float4_t *)dest) = state->sum;
	return sizeof(float4_t);
}

INLINE_FUNCTION(int)
__select_into_form__fsum_fp64(gpuClient *gclient,
							  const kern_colmeta *cmeta,
							  char *dest,
							  const char *source)
{
	kagg_state__psum_fp_packed *state
		= (kagg_state__psum_fp_packed *)source;
	assert(cmeta->attlen == sizeof(float4_t));
	if (state->nitems == 0)
		return 0;
	if (dest)
		*((float8_t *)dest) = state->sum;
	return sizeof(float8_t);
}

INLINE_FUNCTION(int)
__select_into_form__fsum_numeric(gpuClient *gclient,
								 const kern_colmeta *cmeta,
								 char *dest,
								 const char *source)
{
	kagg_state__psum_numeric_packed *state
		= (kagg_state__psum_numeric_packed *)source;
	uint32_t	special = (state->attrs & __PAGG_NUMERIC_ATTRS__MASK);
	uint8_t		kind;
	int16_t		weight;
	int128_t	value;

	assert(cmeta->attlen < 0);
	if (state->nitems == 0)
		return 0;
	switch (special)
	{
		case 0:
			kind = XPU_NUMERIC_KIND__VALID;
			weight = (state->attrs & __PAGG_NUMERIC_ATTRS__WEIGHT);
			value = __fetch_int128_packed(&state->sum);
			break;
		case __PAGG_NUMERIC_ATTRS__PINF:
			kind = XPU_NUMERIC_KIND__POS_INF;
			weight = 0;
			value = 0;
			break;
		case __PAGG_NUMERIC_ATTRS__NINF:
			kind = XPU_NUMERIC_KIND__NEG_INF;
			weight = 0;
			value = 0;
			break;
		default:	/* NaN */
			kind = XPU_NUMERIC_KIND__NAN;
			weight = 0;
			value = 0;
			break;
	}
	return __decimal_to_varlena(dest, kind, weight, value);
}

INLINE_FUNCTION(int)
__select_into_form__favg_int(gpuClient *gclient,
							 const kern_colmeta *cmeta,
							 char *dest,
							 const char *source)
{
	kagg_state__psum_int_packed *state
		= (kagg_state__psum_int_packed *)source;
	uint8_t		n_kind, sum_kind, avg_kind;
	int16_t		n_weight, sum_weight, avg_weight;
	int128_t	n_value, sum_value, avg_value;
	const char *emsg;

	if (state->nitems == 0)
		return 0;
	n_kind = XPU_NUMERIC_KIND__VALID;
	n_weight = 0;
	n_value = state->nitems;

	sum_kind = XPU_NUMERIC_KIND__VALID;
	sum_weight = 0;
	sum_value = state->sum;

	emsg = __decimal_div(&avg_kind, &avg_weight, &avg_value,
						 sum_kind, sum_weight, sum_value,
						 n_kind, n_weight, n_value);
	if (emsg)
	{
		gpuClientELog(gclient, "%s", emsg);
		return -1;
	}
	return __decimal_to_varlena(dest, avg_kind, avg_weight, avg_value);
}

INLINE_FUNCTION(int)
__select_into_form__favg_fp(gpuClient *gclient,
							const kern_colmeta *cmeta,
							char *dest,
							const char *source)
{
	kagg_state__psum_fp_packed *state
		= (kagg_state__psum_fp_packed *)source;
	if (state->nitems == 0)
		return 0;
	*((float8_t *)dest) = (double)state->sum / (double)state->nitems;
	return sizeof(float8_t);
}

INLINE_FUNCTION(int)
__select_into_form__favg_numeric(gpuClient *gclient,
								 const kern_colmeta *cmeta,
								 char *dest,
								 const char *source)
{
	kagg_state__psum_numeric_packed *state
		= (kagg_state__psum_numeric_packed *)source;
	uint32_t	special = (state->attrs & __PAGG_NUMERIC_ATTRS__MASK);
	uint8_t		n_kind, sum_kind, avg_kind;
	int16_t		n_weight, sum_weight, avg_weight;
	int128_t	n_value, sum_value, avg_value;
	const char *emsg;

	if (state->nitems == 0)
		return 0;
	if (special != 0)
	{
		if (special == __PAGG_NUMERIC_ATTRS__PINF)
			avg_kind = XPU_NUMERIC_KIND__POS_INF;
		else if (special == __PAGG_NUMERIC_ATTRS__NINF)
			avg_kind = XPU_NUMERIC_KIND__NEG_INF;
		else
			avg_kind = XPU_NUMERIC_KIND__NAN;
	}
	else
	{
		n_kind = XPU_NUMERIC_KIND__VALID;
		n_weight = 0;
		n_value = state->nitems;

		sum_kind = XPU_NUMERIC_KIND__VALID;
		sum_weight = (state->attrs & __PAGG_NUMERIC_ATTRS__WEIGHT);
		sum_value = __fetch_int128_packed(&state->sum);

		emsg = __decimal_div(&avg_kind, &avg_weight, &avg_value,
							 sum_kind, sum_weight, sum_value,
							 n_kind, n_weight, n_value);
		if (emsg)
		{
			gpuClientELog(gclient, "%s", emsg);
			return -1;
		}
	}
	return __decimal_to_varlena(dest, avg_kind, avg_weight, avg_value);
}

/*
 * VARIANCE/STDDEV
 */
INLINE_FUNCTION(int)
__select_into_form__fstddev_samp(gpuClient *gclient,
								 const kern_colmeta *cmeta,
								 char *dest,
								 const char *source)
{
	kagg_state__stddev_packed *state
		= (kagg_state__stddev_packed *)source;
	if (state->nitems > 1)
	{
		float8_t	N = (double)state->nitems;
		float8_t	fval = N * state->sum_x2 - state->sum_x * state->sum_x;
		uint8_t		kind;
		int16_t		weight;
		int128_t	value;

		__decimal_from_float8(sqrt(fval / (N * (N - 1.0))),
							  &kind,
							  &weight,
							  &value);
		return __decimal_to_varlena(dest, kind, weight, value);
	}
	return 0;
}

INLINE_FUNCTION(int)
__select_into_form__fstddev_sampf(gpuClient *gclient,
								  const kern_colmeta *cmeta,
								  char *dest,
								  const char *source)
{
	kagg_state__stddev_packed *state
		= (kagg_state__stddev_packed *)source;
	if (state->nitems > 1)
	{
		float8_t	N = (double)state->nitems;
		float8_t	fval = N * state->sum_x2 - state->sum_x * state->sum_x;

		if (dest)
			*((float8_t *)dest) = sqrt(fval / (N * (N - 1.0)));
		return sizeof(float8_t);
	}
	return 0;
}

INLINE_FUNCTION(int)
__select_into_form__fvar_samp(gpuClient *gclient,
							  const kern_colmeta *cmeta,
							  char *dest,
							  const char *source)
{
	kagg_state__stddev_packed *state
		= (kagg_state__stddev_packed *)source;
	if (state->nitems > 1)
	{
		float8_t	N = (double)state->nitems;
		float8_t	fval = N * state->sum_x2 - state->sum_x * state->sum_x;
		uint8_t		kind;
		int16_t		weight;
		int128_t	value;

		__decimal_from_float8(fval / (N * (N - 1.0)),
							  &kind,
							  &weight,
							  &value);
		return __decimal_to_varlena(dest, kind, weight, value);
	}
	return 0;
}

INLINE_FUNCTION(int)
__select_into_form__fvar_sampf(gpuClient *gclient,
							   const kern_colmeta *cmeta,
							   char *dest,
							   const char *source)
{
	kagg_state__stddev_packed *state
		= (kagg_state__stddev_packed *)source;
	if (state->nitems > 1)
	{
		float8_t	N = (double)state->nitems;
		float8_t	fval = N * state->sum_x2 - state->sum_x * state->sum_x;

		if (dest)
			*((float8_t *)dest) = (fval / (N * (N - 1.0)));
		return sizeof(float8_t);
	}
	return 0;
}

INLINE_FUNCTION(int)
__select_into_form__fstddev_pop(gpuClient *gclient,
								const kern_colmeta *cmeta,
								char *dest,
								const char *source)
{
	kagg_state__stddev_packed *state
		= (kagg_state__stddev_packed *)source;
	if (state->nitems > 0)
	{
		float8_t	N = (double)state->nitems;
		float8_t	fval = N * state->sum_x2 - state->sum_x * state->sum_x;
		uint8_t		kind;
		int16_t		weight;
		int128_t	value;

		__decimal_from_float8(sqrt(fval / (N * N)),
							  &kind,
							  &weight,
							  &value);
		return __decimal_to_varlena(dest, kind, weight, value);
	}
	return 0;
}

INLINE_FUNCTION(int)
__select_into_form__fstddev_popf(gpuClient *gclient,
								 const kern_colmeta *cmeta,
								 char *dest,
								 const char *source)
{
	kagg_state__stddev_packed *state
		= (kagg_state__stddev_packed *)source;
	if (state->nitems > 0)
	{
		float8_t	N = (double)state->nitems;
		float8_t	fval = N * state->sum_x2 - state->sum_x * state->sum_x;

		if (dest)
			*((float8_t *)dest) = sqrt(fval / (N * N));
		return sizeof(float8_t);
	}
	return 0;
}

INLINE_FUNCTION(int)
__select_into_form__fvar_pop(gpuClient *gclient,
							 const kern_colmeta *cmeta,
							 char *dest,
							 const char *source)
{
	kagg_state__stddev_packed *state
		= (kagg_state__stddev_packed *)source;
	if (state->nitems > 0)
	{
		float8_t	N = (double)state->nitems;
		float8_t	fval = N * state->sum_x2 - state->sum_x * state->sum_x;
		uint8_t		kind;
		int16_t		weight;
		int128_t	value;

		__decimal_from_float8(fval / (N * N),
							  &kind,
							  &weight,
							  &value);
		__decimal_to_varlena(dest, kind, weight, value);
	}
	return 0;
}

INLINE_FUNCTION(int)
__select_into_form__fvar_popf(gpuClient *gclient,
							  const kern_colmeta *cmeta,
							  char *dest,
							  const char *source)
{
	kagg_state__stddev_packed *state
		= (kagg_state__stddev_packed *)source;
	if (state->nitems > 0)
	{
		float8_t	N = (double)state->nitems;
		float8_t	fval = N * state->sum_x2 - state->sum_x * state->sum_x;

		if (dest)
			*((float8_t *)dest) = (fval / (N * N));
		return sizeof(float8_t);
	}
	return 0;
}

/*
 * CORELATION
 */
INLINE_FUNCTION(int)
__select_into_form__fcorr(gpuClient *gclient,
						  const kern_colmeta *cmeta,
						  char *dest,
						  const char *source)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)source;
	if (state->nitems < 1 ||
		state->sum_xx == 0.0 ||
		state->sum_yy == 0.0)
		return 0;
	if (dest)
		*((float8_t *)dest) = (state->sum_xy / sqrt(state->sum_xx * state->sum_yy));
	return sizeof(float8_t);
}

INLINE_FUNCTION(int)
__select_into_form__fcovar_samp(gpuClient *gclient,
								const kern_colmeta *cmeta,
								char *dest,
								const char *source)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)source;
	if (state->nitems < 2)
		return 0;
	if (dest)
		*((float8_t *)dest) = (state->sum_xy / (double)(state->nitems - 1));
	return sizeof(float8_t);
}

INLINE_FUNCTION(int)
__select_into_form__fcovar_pop(gpuClient *gclient,
							   const kern_colmeta *cmeta,
							   char *dest,
							   const char *source)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)source;
	if (state->nitems < 1)
		return 0;
	if (dest)
		*((float8_t *)dest) = (state->sum_xy / (double)state->nitems);
	return sizeof(float8_t);
}

INLINE_FUNCTION(int)
__select_into_form__fregr_avgx(gpuClient *gclient,
							   const kern_colmeta *cmeta,
							   char *dest,
							   const char *source)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)source;
	if (state->nitems < 1)
		return 0;
	if (dest)
		*((float8_t *)dest) = (state->sum_x / (double)state->nitems);
	return sizeof(float8_t);
}

INLINE_FUNCTION(int)
__select_into_form__fregr_avgy(gpuClient *gclient,
							   const kern_colmeta *cmeta,
							   char *dest,
							   const char *source)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)source;
	if (state->nitems < 1)
		return 0;
	if (dest)
		*((float8_t *)dest) = (state->sum_y / (double)state->nitems);
	return sizeof(float8_t);
}

INLINE_FUNCTION(int)
__select_into_form__fregr_count(gpuClient *gclient,
								const kern_colmeta *cmeta,
								char *dest,
								const char *source)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)source;
	if (dest)
		*((float8_t *)dest) = (double)state->nitems;
	return sizeof(float8_t);
}

INLINE_FUNCTION(int)
__select_into_form__fregr_intercept(gpuClient *gclient,
									const kern_colmeta *cmeta,
									char *dest,
									const char *source)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)source;
	if (state->nitems < 1 || state->sum_xx == 0.0)
		return 0;
	if (dest)
		*((float8_t *)dest) = ((state->sum_y -
								state->sum_x * state->sum_xy / state->sum_xx) / (double)state->nitems);
	return sizeof(float8_t);
}

INLINE_FUNCTION(int)
__select_into_form__fregr_r2(gpuClient *gclient,
							 const kern_colmeta *cmeta,
							 char *dest,
							 const char *source)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)source;
	if (state->nitems > 0 &&
		state->sum_xx != 0.0 &&
		state->sum_yy != 0.0)
	{
		if (dest)
			*((float8_t *)dest) = ((state->sum_xy * state->sum_xy) /
								   (state->sum_xx * state->sum_yy));
		return sizeof(float8_t);
	}
	return 0;
}

INLINE_FUNCTION(int)
__select_into_form__fregr_slope(gpuClient *gclient,
								const kern_colmeta *cmeta,
								char *dest,
								const char *source)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)source;
	if (state->nitems > 0 && state->sum_xx != 0.0)
	{
		if (dest)
			*((float8_t *)dest) = (state->sum_xy / state->sum_xx);
		return sizeof(float8_t);
	}
	return 0;
}

INLINE_FUNCTION(int)
__select_into_form__fregr_sxx(gpuClient *gclient,
							  const kern_colmeta *cmeta,
							  char *dest,
							  const char *source)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)source;
	if (state->nitems > 0)
	{
		if (dest)
			*((float8_t *)dest) = state->sum_xx;
		return sizeof(float8_t);
	}
	return 0;
}

INLINE_FUNCTION(int)
__select_into_form__fregr_sxy(gpuClient *gclient,
							  const kern_colmeta *cmeta,
							  char *dest,
							  const char *source)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)source;
	if (state->nitems > 0)
	{
		if (dest)
			*((float8_t *)dest) = state->sum_xy;
		return sizeof(float8_t);
	}
	return 0;
}

INLINE_FUNCTION(int)
__select_into_form__fregr_syy(gpuClient *gclient,
							  const kern_colmeta *cmeta,
							  char *dest,
							  const char *source)
{
	kagg_state__covar_packed *state
		= (kagg_state__covar_packed *)source;
	if (state->nitems > 0)
	{
		if (dest)
			*((float8_t *)dest) = state->sum_yy;
		return sizeof(float8_t);
	}
	return 0;
}

static int32_t
select_into_form_heap_tuple(gpuClient *gclient,
							const kern_data_store *kds_dst,
							HeapTupleHeaderData *htup,
							const kern_aggfinal_projection_desc *af_proj,
							const kern_data_store *kds_src,
							const kern_tupitem *titem,
							uint16_t *p_infomask)
{
	uint16_t	t_infomask = *p_infomask;
	uint8_t	   *nullmap = NULL;
	int			ncols = Min((titem->t_infomask2 & HEAP_NATTS_MASK), kds_src->ncols);
	uint32_t	t_hoff = (titem->t_hoff - MINIMAL_TUPLE_OFFSET);
	const void **attrs = alloca(sizeof(void *) * ncols);
	char	   *base = NULL;
	int32_t		head_sz;

	/* deform the source tuple */
	if ((titem->t_infomask & HEAP_HASNULL) != 0)
		nullmap = (uint8_t *)titem->t_bits;
	for (int j=0; j < ncols; j++)
	{
		const kern_colmeta *cmeta = &kds_src->colmeta[j];

		if (nullmap && att_isnull(j, nullmap))
			attrs[j] = NULL;
		else
		{
			if (cmeta->attlen > 0)
				t_hoff = TYPEALIGN(cmeta->attalign, t_hoff);
			else if (!VARATT_NOT_PAD_BYTE((const char *)titem + t_hoff))
				t_hoff = TYPEALIGN(cmeta->attalign, t_hoff);
			attrs[j] = (const char *)titem + t_hoff;
			if (cmeta->attlen > 0)
				t_hoff += cmeta->attlen;
			else
				t_hoff += VARSIZE_ANY((const char *)titem + t_hoff);
		}
	}
	/* estimate and form the result tuple */
	nullmap = NULL;
	t_hoff = 0;
	if (htup)
	{
		int		__off = offsetof(HeapTupleHeaderData, t_bits);

		if ((htup->t_infomask & HEAP_HASNULL) != 0)
		{
			nullmap = htup->t_bits;
			memset(nullmap, 0, BITMAPLEN(af_proj->nattrs));
			__off += BITMAPLEN(af_proj->nattrs);
		}
		__off = MAXALIGN(__off);
		htup->t_hoff = __off;
		base = (char *)htup + __off;
	}

	assert(kds_dst->ncols == af_proj->nattrs);
	for (int j=0; j < af_proj->nattrs; j++)
	{
		const kern_colmeta *cmeta = &kds_dst->colmeta[j];
		int16_t		action = af_proj->desc[j].action;
		int16_t		resno = af_proj->desc[j].resno;
		uint32_t	t_next;
		int			sz;
		char	   *dest;
		const char *source;

		if (resno < 1 || resno > ncols || (source = attrs[resno-1]) == NULL)
		{
			t_infomask |= HEAP_HASNULL;
			continue;	/* NULL */
		}
		t_next = TYPEALIGN(cmeta->attalign, t_hoff);
		dest = (base ? base + t_next : NULL);
		switch (action)
		{
			case KAGG_FINAL__SIMPLE_VREF:
				if (cmeta->attlen > 0)
					sz = cmeta->attlen;
				else
					sz = VARSIZE_ANY(source);
				if (dest)
					memcpy(dest, source, sz);
				break;
			case KAGG_FINAL__FMINMAX_INT8:
				sz = __select_into_form__fminmax_int8(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FMINMAX_INT16:
				sz = __select_into_form__fminmax_int16(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FMINMAX_INT32:
			case KAGG_FINAL__FMINMAX_DATE:
				sz = __select_into_form__fminmax_int32(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FMINMAX_INT64:
			case KAGG_FINAL__FMINMAX_CASH:			/* 64bit Int */
			case KAGG_FINAL__FMINMAX_TIME:			/* 64bit Int */
			case KAGG_FINAL__FMINMAX_TIMESTAMP:		/* 64bit Int */
			case KAGG_FINAL__FMINMAX_TIMESTAMPTZ:	/* 64bit Int */
				sz = __select_into_form__fminmax_int64(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FMINMAX_FP16:
				sz = __select_into_form__fminmax_fp16(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FMINMAX_FP32:
				sz = __select_into_form__fminmax_fp32(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FMINMAX_FP64:
				sz = __select_into_form__fminmax_fp64(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FMINMAX_NUMERIC:
				sz = __select_into_form__fminmax_numeric(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSUM_INT:
			case KAGG_FINAL__FSUM_CASH:
				sz = __select_into_form__fsum_int(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSUM_INT64:
				sz = __select_into_form__fsum_int64(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSUM_FP32:
				sz = __select_into_form__fsum_fp32(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSUM_FP64:
				sz = __select_into_form__fsum_fp64(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSUM_NUMERIC:
				sz = __select_into_form__fsum_numeric(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FAVG_INT:
			case KAGG_FINAL__FAVG_INT64:
				sz = __select_into_form__favg_int(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FAVG_FP64:
				sz = __select_into_form__favg_fp(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FAVG_NUMERIC:
				sz = __select_into_form__favg_numeric(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSTDDEV_SAMP:
				sz = __select_into_form__fstddev_samp(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSTDDEV_SAMPF:
				sz = __select_into_form__fstddev_sampf(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSTDDEV_POP:
				sz = __select_into_form__fstddev_pop(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSTDDEV_POPF:
				sz = __select_into_form__fstddev_popf(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FVAR_SAMP:
				sz = __select_into_form__fvar_samp(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FVAR_SAMPF:
				sz = __select_into_form__fvar_sampf(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FVAR_POP:
				sz = __select_into_form__fvar_pop(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FVAR_POPF:
				sz = __select_into_form__fvar_popf(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FCORR:
				sz = __select_into_form__fcorr(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FCOVAR_SAMP:
				sz = __select_into_form__fcovar_samp(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FCOVAR_POP:
				sz = __select_into_form__fcovar_pop(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_AVGX:
				sz = __select_into_form__fregr_avgx(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_AVGY:
				sz = __select_into_form__fregr_avgy(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_COUNT:
				sz = __select_into_form__fregr_count(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_INTERCEPT:
				sz = __select_into_form__fregr_intercept(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_R2:
				sz = __select_into_form__fregr_r2(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_SLOPE:
				sz = __select_into_form__fregr_slope(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_SXX:
				sz = __select_into_form__fregr_sxx(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_SXY:
				sz = __select_into_form__fregr_sxy(gclient, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_SYY:
				sz = __select_into_form__fregr_syy(gclient, cmeta, dest, source);
				break;
			default:
				return -1;
		}
		if (sz < 0)			/* errors */
			return -1;
		else if (sz == 0)	/* null */
			t_infomask |= HEAP_HASNULL;
		else
		{
			if (cmeta->attlen < 0)
				t_infomask |= HEAP_HASVARWIDTH;
			if (base && t_next > t_hoff)
				memset(base + t_hoff, 0, t_next - t_hoff);
			if (nullmap)
				nullmap[j>>3] |= (1<<(j&7));    /* not null */
			t_hoff = t_next + sz;
		}
	}
	/* infomask */
	if (htup)
	{
		htup->t_infomask2 = af_proj->nattrs;
		htup->t_infomask = t_infomask;
	}
	/* ok */
	*p_infomask = t_infomask;
	head_sz = offsetof(HeapTupleHeaderData, t_bits);
	if ((t_infomask & HEAP_HASNULL) != 0)
		head_sz += BITMAPLEN(af_proj->nattrs);
	return MAXALIGN(head_sz) + t_hoff;
}

INLINE_FUNCTION(void)
__initPageHeaderData(PageHeaderData *hpage)
{
	/*
	 * NOTE: PG-Strom's should write heap blocks only to a newly created empty table
	 * in SELECT INTO Direct mode.
	 * This table is not visible to other transactions until the current transaction is
	 * committed, and the table is deleted if any error occurs before the transaction is
	 * committed.
	 * So, when writing to an empty table with SELECT INTO, you can safely consider
	 * the tuples to be ALL_VISIBLE and XMIN_FROZEN.
	 */
    hpage->pd_checksum = 0;
    hpage->pd_flags = PD_ALL_VISIBLE;   /* see the comments above */
    hpage->pd_lower = offsetof(PageHeaderData, pd_linp);
    hpage->pd_upper = BLCKSZ;
    hpage->pd_special = BLCKSZ;
    hpage->pd_pagesize_version = (BLCKSZ | PG_PAGE_LAYOUT_VERSION);
    hpage->pd_prune_xid = 0;
}

INLINE_FUNCTION(kern_data_store *)
__allocSelectIntoKDS(const kern_data_store *kds_head)
{
	kern_data_store *kds_new;
	size_t		head_sz = KDS_HEAD_LENGTH(kds_head);

	kds_new = malloc(head_sz + BLCKSZ * RELSEG_SIZE + PAGE_SIZE);
	if (!kds_new)
		return NULL;
	memcpy(kds_new, kds_head, head_sz);
	kds_new->nitems = 0;
	kds_new->block_offset = (TYPEALIGN(PAGE_SIZE, (char *)kds_new + head_sz)
							 - (uintptr_t)kds_new);
	kds_new->length = kds_new->block_offset + BLCKSZ * RELSEG_SIZE;

	return kds_new;
}

bool
__initSelectIntoState(selectIntoState *si_state, const kern_session_info *session)
{
	const char *base_name = SESSION_SELECT_INTO_PATHNAME(session);
	const kern_data_store *kds_head = SESSION_SELECT_INTO_KDS_HEAD(session);

	if (!base_name ||
		!kds_head ||
		(session->xpu_task_flags & DEVTASK__SELECT_INTO_DIRECT) == 0)
		return true;		/* nothing to do */
	pthreadMutexInit(&si_state->kds_heap_lock);
	si_state->kds_heap = __allocSelectIntoKDS(kds_head);
	if (!si_state->kds_heap)
		goto error;
	si_state->kds_heap_segno = 0;

	pthreadMutexInit(&si_state->dpage_lock);
	si_state->dpage = malloc(BLCKSZ);
	__initPageHeaderData(si_state->dpage);

	si_state->base_name = strdup(base_name);
	if (!si_state->base_name)
		goto error;
	return true;
error:
	__cleanupSelectIntoState(si_state);
	return false;
}

void
__cleanupSelectIntoState(selectIntoState *si_state)
{
	if (si_state->kds_heap)
		free(si_state->kds_heap);
	if (si_state->dpage)
        free(si_state->dpage);
    if (si_state->base_name)
        free((void *)si_state->base_name);
}

/*
 * __selectIntoWriteOutOneKDS
 */
static bool
__selectIntoWriteOutOneKDS(gpuClient *gclient,
						   selectIntoState *si_state,
						   kern_data_store *kds_heap)
{
	uint32_t	segment_no = __atomic_add_uint32(&si_state->kds_heap_segno, 1);
	char	   *fname = alloca(strlen(si_state->base_name) + 20);
	int			fdesc;
	uint32_t	nblocks = Min(kds_heap->nitems, RELSEG_SIZE);
	size_t		remained = nblocks * BLCKSZ;
	const char *curr_pos = (const char *)kds_heap + kds_heap->block_offset;
	ssize_t		nbytes;

	if (segment_no == 0)
		strcpy(fname, si_state->base_name);
	else
		sprintf(fname, "%s.%u", si_state->base_name, segment_no);
	fdesc = open(fname, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0600);
	if (fdesc < 0)
	{
		gpuClientELog(gclient, "failed on open('%s'): %m", fname);
		return false;
	}
	while (remained > 0)
	{
		nbytes = write(fdesc, curr_pos, remained);
		if (nbytes <= 0)
		{
			if (errno == EINTR)
				continue;
			gpuClientELog(gclient, "failed on write('%s', %lu): %m",
						  fname, remained);
			close(fdesc);
			return false;
		}
		assert(nbytes <= remained);
		curr_pos += nbytes;
		remained -= nbytes;
	}
	close(fdesc);
	__atomic_add_uint64(&si_state->nblocks_written, nblocks);
	return true;
}

/*
 * __selectIntoWriteOutOneBlock
 */
static bool
__selectIntoWriteOutOneBlock(gpuClient *gclient,
							 selectIntoState *si_state,
							 PageHeaderData *hpage)
{
	kern_data_store *kds_heap;
	PageHeaderData *dpage;

	pthreadMutexLock(&si_state->kds_heap_lock);
	kds_heap = si_state->kds_heap;
	assert(kds_heap->nitems < RELSEG_SIZE);
	dpage = KDS_BLOCK_PGPAGE(kds_heap, kds_heap->nitems++);
	/* replace and write out the buffer, if last one */
	if (kds_heap->nitems < RELSEG_SIZE)
		__atomic_add_int64((int64_t *)&kds_heap->usage, 2);
	else
	{
		kern_data_store *kds_new = __allocSelectIntoKDS(kds_heap);

		if (!kds_new)
		{
			pthreadMutexUnlock(&si_state->kds_heap_lock);
			gpuClientELog(gclient, "out of memory");
			return false;
		}
		si_state->kds_heap = kds_new;
		/* mark writable */
		__atomic_add_int64((int64_t *)&kds_heap->usage, 3);
	}
	pthreadMutexUnlock(&si_state->kds_heap_lock);
	memcpy(dpage, hpage, BLCKSZ);
	if (__atomic_add_int64((int64_t *)&kds_heap->usage, -2) == 3)
	{
		__selectIntoWriteOutOneKDS(gclient, si_state, kds_heap);
		free(kds_heap);
	}
	return true;
}

/*
 * __selectIntoWriteOutHeapCommon
 */
static bool
__selectIntoWriteOutHeapCommon(gpuClient *gclient,
							   kern_session_info *session,
							   selectIntoState *si_state,
							   kern_data_store *kds_dst)
{
	const kern_aggfinal_projection_desc *af_proj = NULL;
	const kern_data_store *kds_head = NULL;
	PageHeaderData *hpage = (PageHeaderData *)alloca(BLCKSZ);
	uint32_t	lp_index = 0;
	uint32_t	lp_offset = BLCKSZ;

	assert(kds_dst->format == KDS_FORMAT_ROW ||
		   kds_dst->format == KDS_FORMAT_HASH);
	if (session)
	{
		af_proj = SESSION_SELECT_INTO_PROJDESC(session);
		kds_head = SESSION_SELECT_INTO_KDS_HEAD(session);
		assert(kds_head && kds_head->format == KDS_FORMAT_BLOCK);
	}
	__initPageHeaderData(hpage);
	for (uint32_t index=0; index < kds_dst->nitems; index++)
	{
		kern_tupitem   *titem = KDS_GET_TUPITEM(kds_dst, index);
		ItemIdData		lp_item;
		int32_t			tupsz;
		uint16_t		t_infomask = 0;
		HeapTupleHeaderData *htup;

		if (!titem)
			continue;
		if (!af_proj)
			tupsz = MINIMAL_TUPLE_OFFSET + titem->t_len;
		else
		{
			tupsz = select_into_form_heap_tuple(gclient,
												kds_head,
												NULL,
												af_proj,
												kds_dst,
												titem,
												&t_infomask);
			if (tupsz < 0)
				return false;
		}
		/*
		 * move the local page to the kds_heap because it cannot load
		 * any tuples here. 
		 */
	again:
		if (offsetof(PageHeaderData,
					 pd_linp[lp_index+1]) + MAXALIGN(tupsz) > lp_offset)
		{
			/* tuple is too large to write out heap blocks without toast */
			if (lp_index == 0)
			{
				gpuClientELog(gclient, "SELECT INTO: too large HeapTuple (tupsz=%u)",
							  tupsz);
				return false;
			}
			hpage->pd_lower = offsetof(PageHeaderData, pd_linp[lp_index]);
			hpage->pd_upper = lp_offset;
			assert(hpage->pd_lower <= hpage->pd_upper);
			if (!__selectIntoWriteOutOneBlock(gclient, si_state, hpage))
				return false;
			/* reset the local buffer usage */
			lp_index = 0;
			lp_offset = BLCKSZ;
			goto again;
		}
		/* ok, the tuple still fits the local block buffer */
		lp_offset -= MAXALIGN(tupsz);
		lp_item.lp_off = lp_offset;
		lp_item.lp_flags = LP_NORMAL;
		lp_item.lp_len = tupsz;
		hpage->pd_linp[lp_index++] = lp_item;
		htup = (HeapTupleHeaderData *)((char *)hpage + lp_offset);
		htup->t_choice.t_heap.t_xmin = FrozenTransactionId;
		htup->t_choice.t_heap.t_xmax = InvalidTransactionId;
		htup->t_choice.t_heap.t_field3.t_cid = InvalidCommandId;
		htup->t_choice.t_datum.datum_typeid = kds_dst->tdtypeid;
		htup->t_ctid.ip_blkid.bi_hi = 0xffff;
		htup->t_ctid.ip_blkid.bi_lo = 0xffff;
		htup->t_ctid.ip_posid = 0;
		if (!af_proj)
			memcpy(&htup->t_infomask2,
				   &titem->t_infomask2,
				   titem->t_len - offsetof(kern_tupitem, t_infomask2));
		else
			select_into_form_heap_tuple(gclient,
										kds_head,
										htup,
										af_proj,
										kds_dst,
										titem,
										&t_infomask);
		/* force tuple to all-visible */
		htup->t_infomask &= ~HEAP_XACT_MASK;
		htup->t_infomask |= (HEAP_XMIN_FROZEN | HEAP_XMAX_INVALID);
	}
	/* flush remained tuples using tuple-by-tuple (slow) mode */
	if (lp_index > 0)
	{
		HeapTupleHeaderData *htup;
		PageHeaderData *dpage;

		pthreadMutexLock(&si_state->dpage_lock);
		dpage = si_state->dpage;
		for (int index=0; index < lp_index; index++)
		{
			ItemIdData	lp_item = hpage->pd_linp[index];
			int32_t		tupsz = lp_item.lp_len;

			htup = (HeapTupleHeaderData *)((char *)hpage + lp_item.lp_off);
			if (dpage->pd_lower +
				sizeof(ItemIdData) +
				MAXALIGN(tupsz) > dpage->pd_upper)
			{
				if (!__selectIntoWriteOutOneBlock(gclient, si_state, dpage))
				{
					pthreadMutexUnlock(&si_state->dpage_lock);
					return false;
				}
				/* reset the buffer usage */
				dpage->pd_lower = offsetof(PageHeaderData, pd_linp);
				dpage->pd_upper = BLCKSZ;
			}
			assert(dpage->pd_lower +
				   sizeof(ItemIdData) +
				   MAXALIGN(tupsz) <= dpage->pd_upper);
			dpage->pd_upper -= MAXALIGN(tupsz);
			memcpy((char *)dpage + dpage->pd_upper, htup, tupsz);
			lp_item.lp_off = dpage->pd_upper;
			dpage->pd_linp[(dpage->pd_lower -
							offsetof(PageHeaderData, pd_linp)) / sizeof(ItemIdData)] = lp_item;
			dpage->pd_lower += sizeof(ItemIdData);
		}
		pthreadMutexUnlock(&si_state->dpage_lock);
	}
	return true;
}

/*
 * selectIntoWriteOutHeapNormal
 */
bool
selectIntoWriteOutHeapNormal(gpuClient *gclient,
							 selectIntoState *si_state,
							 kern_data_store *kds_dst)	/* managed memory */
{
	return __selectIntoWriteOutHeapCommon(gclient,
										  NULL,
										  si_state,
										  kds_dst);
}


/*
 * selectIntoWriteOutHeapFinal
 */
long
selectIntoWriteOutHeapFinal(gpuClient *gclient,
							kern_session_info *session,
							selectIntoState *si_state,
							kern_data_store *kds_dst)
{
	if (__selectIntoWriteOutHeapCommon(gclient,
									   session,
									   si_state,
									   kds_dst))
		return selectIntoFinalFlushBuffer(gclient, si_state);
	return -1;
}

/*
 * __selectIntoDirectWriteVM
 */
#define MAPSIZE					(BLCKSZ - MAXALIGN(SizeOfPageHeaderData))
#define HEAPBLOCKS_PER_BYTE		4		/* visibility map takes 2bits for each byte */
#define HEAPBLOCKS_PER_PAGE		(MAPSIZE * HEAPBLOCKS_PER_BYTE)

static long
__selectIntoDirectWriteVM(gpuClient *gclient,
						  selectIntoState *si_state)
{
	uint64_t	nblocks = si_state->nblocks_written;
	char	   *vm_fname = (char *)alloca(strlen(si_state->base_name) + 20);
	int			vm_fdesc;
	uint8_t		vm_buf[BLCKSZ];

	sprintf(vm_fname, "%s_vm", si_state->base_name);
	vm_fdesc = open(vm_fname, O_WRONLY | O_CREAT | O_TRUNC, 0600);
	if (vm_fdesc < 0)
	{
		gpuClientELog(gclient, "failed on open('%s'): %m", vm_fname);
		return -1;
	}
	while (nblocks > 0)
	{
		PageHeaderData *hpage = (PageHeaderData *)vm_buf;
		int		nwrites = Min(nblocks, HEAPBLOCKS_PER_PAGE);
		int		length = nwrites / HEAPBLOCKS_PER_BYTE;
		off_t	offset = MAXALIGN(SizeOfPageHeaderData);
		ssize_t	nbytes;

		PageInit((Page)hpage, BLCKSZ, 0);
		if (length > 0)
		{
			memset(vm_buf + offset, -1, length);
			offset += length;
		}
		switch (nwrites % 4)
		{
			case 1:
				vm_buf[offset++] = 0x03;
				break;
			case 2:
				vm_buf[offset++] = 0x0f;
				break;
			case 3:
				vm_buf[offset++] = 0x3f;
				break;
			default:
				break;
		}
		if (offset < BLCKSZ)
			memset(vm_buf + offset, 0, BLCKSZ - offset);
		offset = 0;
		while (offset < BLCKSZ)
		{
			nbytes = write(vm_fdesc, vm_buf + offset, BLCKSZ - offset);
			if (nbytes <= 0)
			{
				if (errno == EINTR)
					continue;
				gpuClientELog(gclient, "failed on write('%s'): %m", vm_fname);
				close(vm_fdesc);
				return -1;
			}
			offset += nbytes;
		}
		nblocks -= nwrites;
	}
	close(vm_fdesc);
	/* returns statistics */
	return si_state->nblocks_written;
}

/*
 * selectIntoFinalFlushBuffer
 */
long
selectIntoFinalFlushBuffer(gpuClient *gclient,
						   selectIntoState *si_state)
{
	PageHeaderData *hpage = si_state->dpage;
	kern_data_store	*kds_heap;

	if (hpage->pd_lower > offsetof(PageHeaderData, pd_linp))
	{
		if (!__selectIntoWriteOutOneBlock(gclient, si_state, hpage))
			return false;
	}
	kds_heap = si_state->kds_heap;
	if (kds_heap->nitems > 0)
	{
		if (!__selectIntoWriteOutOneKDS(gclient,
										si_state,
										kds_heap))
			return false;
	}
	/* visibility map */
	return __selectIntoDirectWriteVM(gclient, si_state);
}
