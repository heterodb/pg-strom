/*
 * cuda_gpupreagg.cu
 *
 * Device implementation of GpuScan
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "cuda_common.h"
#include "float2.h"

/*
 * __atomic_add_int128
 *
 * atomically increment packed int128 value using 64bit atomic operation.
 */
INLINE_FUNCTION(void)
__atomic_add_int128(int128_packed_t *ptr, int128_t ival)
{
	uint64_t	old_lo;
	uint64_t	new_hi;
	uint64_t	temp	__attribute__((unused));

	old_lo = atomicAdd((unsigned long long *)&ptr->u64_lo,
					   (uint64_t)(ival & ULONG_MAX));
	asm volatile("add.cc.u64 %0, %2, %3;\n"
				 "addc.u64   %1, %4, %5;\n"
				 : "=l"(temp),  "=l"(new_hi)
				 : "l"(old_lo), "l"((uint64_t)(ival & ULONG_MAX)),
				   "n"(0),      "l"((uint64_t)((ival>>64) & ULONG_MAX)));
	/* new_hi = ival_hi + carry bit of (old_lo + ival_lo) */
	if (new_hi != 0)
		atomicAdd((unsigned long long *)&ptr->u64_hi, new_hi);
}

/*
 * __normalize_numeric_int128
 */
STATIC_FUNCTION(int128_t)
__normalize_numeric_int128(int16_t weight_d, int16_t weight_s, int128_t ival)
{
	static uint64_t		__pow10[] = {
		1UL,						/* 10^0 */
		10UL,						/* 10^1 */
		100UL,						/* 10^2 */
		1000UL,						/* 10^3 */
		10000UL,					/* 10^4 */
		100000UL,					/* 10^5 */
		1000000UL,					/* 10^6 */
		10000000UL,					/* 10^7 */
		100000000UL,				/* 10^8 */
		1000000000UL,				/* 10^9 */
		10000000000UL,				/* 10^10 */
		100000000000UL,				/* 10^11 */
		1000000000000UL,			/* 10^12 */
		10000000000000UL,			/* 10^13 */
		100000000000000UL,			/* 10^14 */
		1000000000000000UL,			/* 10^15 */
		10000000000000000UL,		/* 10^16 */
		100000000000000000UL,		/* 10^17 */
		1000000000000000000UL,		/* 10^18 */
	};

	if (weight_d > weight_s)
	{
		int		shift = (weight_d - weight_s);

		while (shift > 0)
		{
			int		k = Min(shift, 18);

			ival *= (int128_t)__pow10[k];
			shift -= k;
		}
	}
	else if (weight_d < weight_s)
	{
		int		shift = (weight_s - weight_d);

		while (shift > 0)
		{
			int		k = Min(shift, 18);

			ival /= (int128_t)__pow10[k];
			shift -= k;
		}
	}
	return ival;
}

/*
 * __writeOutOneTuplePreAgg
 */
STATIC_FUNCTION(int32_t)
__writeOutOneTuplePreAgg(kern_context *kcxt,
						 kern_data_store *kds_final,
						 kern_tupitem *titem,
						 const kern_expression *kexp_actions)
{
	int			nattrs = Min(kds_final->ncols, kexp_actions->u.pagg.nattrs);
	uint32_t	t_hoff, t_next;
	uint16_t	t_infomask = HEAP_HASNULL;
	char	   *buffer = NULL;

	t_hoff = MAXALIGN(offsetof(kern_tupitem,
							   t_bits) + BITMAPLEN(nattrs));
	if (titem)
	{
		memset(titem, 0, t_hoff);
		titem->t_infomask2 = (nattrs & HEAP_NATTS_MASK);
		titem->t_hoff = t_hoff + MINIMAL_TUPLE_OFFSET;
	}
	/* walk on the columns */
	for (int j=0; j < nattrs; j++)
	{
		const kern_aggregate_desc *desc = &kexp_actions->u.pagg.desc[j];
		kern_colmeta   *cmeta = &kds_final->colmeta[j];
		xpu_datum_t	   *xdatum;
		int				nbytes;

		assert((char *)cmeta > (char *)kds_final &&
			   (char *)cmeta < (char *)kds_final + kds_final->length);
		assert(cmeta->attalign > 0 && cmeta->attalign <= 8);
		t_next = TYPEALIGN(cmeta->attalign, t_hoff);
		if (titem)
		{
			if (t_next > t_hoff)
				memset((char *)titem + t_hoff, 0, t_next - t_hoff);
			buffer = (char *)titem + t_next;
		}

		switch (desc->action)
		{
			case KAGG_ACTION__VREF:
				assert(desc->arg0_slot_id >= 0 &&
					   desc->arg0_slot_id < kcxt->kvars_nslots);
				xdatum = kcxt->kvars_slot[desc->arg0_slot_id];
				if (XPU_DATUM_ISNULL(xdatum))
					nbytes = 0;
				else
				{
					nbytes = xdatum->expr_ops->xpu_datum_write(kcxt,
															   buffer,
															   cmeta,
															   xdatum);
					if (nbytes < 0)
						return -1;
				}
				break;

			case KAGG_ACTION__NROWS_ANY:
			case KAGG_ACTION__NROWS_COND:
				assert(cmeta->attlen == sizeof(int64_t));
				nbytes = sizeof(int64_t);
				if (buffer)
					*((int64_t *)buffer) = 0;
				break;

			case KAGG_ACTION__PMIN_INT32:
			case KAGG_ACTION__PMIN_INT64:
				nbytes = sizeof(kagg_state__pminmax_int64_packed);
				if (buffer)
				{
					kagg_state__pminmax_int64_packed *r =
						(kagg_state__pminmax_int64_packed *)buffer;
					r->attrs = 0;
					r->value = LONG_MAX;
					SET_VARSIZE(r, sizeof(kagg_state__pminmax_int64_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__PMAX_INT32:
			case KAGG_ACTION__PMAX_INT64:
				nbytes = sizeof(kagg_state__pminmax_int64_packed);
				if (buffer)
				{
					kagg_state__pminmax_int64_packed *r =
						(kagg_state__pminmax_int64_packed *)buffer;
					r->attrs = 0;
					r->value = LONG_MIN;
					SET_VARSIZE(r, sizeof(kagg_state__pminmax_int64_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__PMIN_FP64:
				nbytes = sizeof(kagg_state__pminmax_fp64_packed);
				if (buffer)
				{
					kagg_state__pminmax_fp64_packed *r =
						(kagg_state__pminmax_fp64_packed *)buffer;
					r->attrs = 0;
					r->value = DBL_MAX;
					SET_VARSIZE(r, sizeof(kagg_state__pminmax_fp64_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__PMAX_FP64:
				nbytes = sizeof(kagg_state__pminmax_fp64_packed);
				if (buffer)
				{
					kagg_state__pminmax_fp64_packed *r =
						(kagg_state__pminmax_fp64_packed *)buffer;
					r->attrs = 0;
					r->value = -DBL_MAX;
					SET_VARSIZE(r, sizeof(kagg_state__pminmax_fp64_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__PSUM_INT:
			case KAGG_ACTION__PAVG_INT:
				nbytes = sizeof(kagg_state__psum_int_packed);
				if (buffer)
				{
					memset(buffer, 0, sizeof(kagg_state__psum_int_packed));
					SET_VARSIZE(buffer, sizeof(kagg_state__psum_int_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__PSUM_INT64:
			case KAGG_ACTION__PAVG_INT64:
				nbytes = sizeof(kagg_state__psum_numeric_packed);
				if (buffer)
				{
					kagg_state__psum_numeric_packed *r =
						(kagg_state__psum_numeric_packed *)buffer;
					memset(r, 0, sizeof(kagg_state__psum_numeric_packed));
					/* weight is always 0, unlike numeric */
                    SET_VARSIZE(buffer, sizeof(kagg_state__psum_numeric_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__PSUM_FP:
			case KAGG_ACTION__PAVG_FP:
				nbytes = sizeof(kagg_state__psum_fp_packed);
				if (buffer)
				{
					memset(buffer, 0, sizeof(kagg_state__psum_fp_packed));
					SET_VARSIZE(buffer, sizeof(kagg_state__psum_fp_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__PSUM_NUMERIC:
			case KAGG_ACTION__PAVG_NUMERIC:
				nbytes = sizeof(kagg_state__psum_numeric_packed);
				if (buffer)
				{
					kagg_state__psum_numeric_packed *r =
						(kagg_state__psum_numeric_packed *)buffer;
					memset(r, 0, sizeof(kagg_state__psum_numeric_packed));
					r->attrs = __numeric_typmod_weight(desc->typmod);
					SET_VARSIZE(buffer, sizeof(kagg_state__psum_numeric_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__STDDEV:
				nbytes = sizeof(kagg_state__stddev_packed);
				if (buffer)
				{
					memset(buffer, 0, sizeof(kagg_state__stddev_packed));
					SET_VARSIZE(buffer, sizeof(kagg_state__stddev_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__COVAR:
				nbytes = sizeof(kagg_state__covar_packed);
				if (buffer)
				{
					memset(buffer, 0, sizeof(kagg_state__covar_packed));
					SET_VARSIZE(buffer, sizeof(kagg_state__covar_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			default:
				STROM_ELOG(kcxt, "unknown xpuPreAgg action");
				return -1;
		}
		if (titem && nbytes > 0)
			titem->t_bits[j>>3] |= (1<<(j&7));
		t_hoff = t_next + nbytes;
	}
	t_hoff += ROWID_SIZE;
	
	if (titem)
	{
		titem->t_len = t_hoff;
		titem->t_infomask = t_infomask;
	}
	return t_hoff;
}

/*
 * __update_nogroups__nrows_any
 */
INLINE_FUNCTION(void)
__update_nogroups__nrows_any(kern_context *kcxt,
							 char *buffer,
							 kern_colmeta *cmeta,
							 kern_aggregate_desc *desc,
							 bool source_is_valid)
{
	int		count;

	count = __syncthreads_count(source_is_valid);
	if (get_local_id() == 0)
	{
		if (__isShared(buffer))
			*((uint64_t *)buffer) += count;
		else
			__atomic_add_uint64((uint64_t *)buffer, count);
	}
}

/*
 * __update_nogroups__nrows_cond
 */
INLINE_FUNCTION(void)
__update_nogroups__nrows_cond(kern_context *kcxt,
							  char *buffer,
							  kern_colmeta *cmeta,
							  kern_aggregate_desc *desc,
							  bool source_is_valid)
{
	int		count;

	if (source_is_valid)
	{
		xpu_datum_t	   *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];

		if (XPU_DATUM_ISNULL(xdatum))
			source_is_valid = false;
	}
	count = __syncthreads_count(source_is_valid);
	if (get_local_id() == 0)
	{
		if (__isShared(buffer))
			*((uint64_t *)buffer) += count;
		else
			__atomic_add_uint64((uint64_t *)buffer, count);
	}
}

/*
 * __update_nogroups__XXXX
 */

/*
 * __update_nogroups__pmin_int32
 */
INLINE_FUNCTION(void)
__update_nogroups__pmin_int32(kern_context *kcxt,
							  char *buffer,
							  kern_colmeta *cmeta,
							  kern_aggregate_desc *desc,
							  bool source_is_valid)
{
	int32_t		ival;

	if (source_is_valid)
	{
		const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];

		if (!__preagg_fetch_xdatum_as_int32(&ival, xdatum))
			source_is_valid = false;
	}

	if (__syncthreads_count(source_is_valid) > 0)
	{
		ival = pgstrom_local_min_int32(source_is_valid ? ival : INT_MAX);
		if (get_local_id() == 0)
		{
			kagg_state__pminmax_int64_packed *r =
				(kagg_state__pminmax_int64_packed *)buffer;
			if (__isShared(r))
			{
				r->attrs |= __PAGG_MINMAX_ATTRS__VALID;
				if (r->value > ival)
					r->value = ival;
			}
			else
			{
				__atomic_or_uint32(&r->attrs, __PAGG_MINMAX_ATTRS__VALID);
				__atomic_min_int64(&r->value, (int64_t)ival);
			}
		}
	}
}

/*
 * __update_nogroups__pmin_int64
 */
INLINE_FUNCTION(void)
__update_nogroups__pmin_int64(kern_context *kcxt,
							  char *buffer,
							  kern_colmeta *cmeta,
							  kern_aggregate_desc *desc,
							  bool source_is_valid)
{
	int64_t		ival;

	if (source_is_valid)
	{
		const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];

		if (!__preagg_fetch_xdatum_as_int64(&ival, xdatum))
			source_is_valid = false;
	}

	if (__syncthreads_count(source_is_valid) > 0)
	{
		ival = pgstrom_local_min_int64(source_is_valid ? ival : LONG_MAX);
		if (get_local_id() == 0)
		{
			kagg_state__pminmax_int64_packed *r =
				(kagg_state__pminmax_int64_packed *)buffer;
			if (__isShared(r))
			{
				r->attrs |= __PAGG_MINMAX_ATTRS__VALID;
				if (r->value > ival)
					r->value = ival;
			}
			else
			{
				__atomic_or_uint32(&r->attrs, __PAGG_MINMAX_ATTRS__VALID);
				__atomic_min_int64(&r->value, ival);
			}
		}
	}
}

/*
 * __update_nogroups__pmax_int32
 */
INLINE_FUNCTION(void)
__update_nogroups__pmax_int32(kern_context *kcxt,
							  char *buffer,
							  kern_colmeta *cmeta,
							  kern_aggregate_desc *desc,
							  bool source_is_valid)
{
	int32_t		ival;

	if (source_is_valid)
	{
		const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];

		if (!__preagg_fetch_xdatum_as_int32(&ival, xdatum))
			source_is_valid = false;
	}

	if (__syncthreads_count(source_is_valid) > 0)
	{
		ival = pgstrom_local_max_int64(source_is_valid ? ival : INT_MIN);
		if (get_local_id() == 0)
		{
			kagg_state__pminmax_int64_packed *r =
				(kagg_state__pminmax_int64_packed *)buffer;
			if (__isShared(r))
			{
				r->attrs |= __PAGG_MINMAX_ATTRS__VALID;
				if (r->value < ival)
					r->value = ival;
			}
			else
			{
				__atomic_or_uint32(&r->attrs, __PAGG_MINMAX_ATTRS__VALID);
				__atomic_max_int64(&r->value, (int64_t)ival);
			}
		}
	}
}

/*
 * __update_nogroups__pmax_int64
 */
INLINE_FUNCTION(void)
__update_nogroups__pmax_int64(kern_context *kcxt,
							  char *buffer,
							  kern_colmeta *cmeta,
							  kern_aggregate_desc *desc,
							  bool source_is_valid)
{
	int64_t		ival;

	if (source_is_valid)
	{
		const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];

		if (!__preagg_fetch_xdatum_as_int64(&ival, xdatum))
			source_is_valid = false;
	}

	if (__syncthreads_count(source_is_valid) > 0)
	{
		ival = pgstrom_local_max_int64(source_is_valid ? ival : LONG_MIN);
		if (get_local_id() == 0)
		{
			kagg_state__pminmax_int64_packed *r =
				(kagg_state__pminmax_int64_packed *)buffer;
			if (__isShared(r))
			{
				r->attrs |= __PAGG_MINMAX_ATTRS__VALID;
				if (r->value < ival)
					r->value = ival;
			}
			else
			{
				__atomic_or_uint32(&r->attrs, __PAGG_MINMAX_ATTRS__VALID);
				__atomic_max_int64(&r->value, ival);
			}
		}
	}
}

/*
 * __update_nogroups__pmin_fp64
 */
INLINE_FUNCTION(void)
__update_nogroups__pmin_fp64(kern_context *kcxt,
							 char *buffer,
							 kern_colmeta *cmeta,
							 kern_aggregate_desc *desc,
							 bool source_is_valid)
{
	float8_t	fval;

	if (source_is_valid)
	{
		const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];

		if (!__preagg_fetch_xdatum_as_float64(&fval, xdatum))
			source_is_valid = false;
	}

	if (__syncthreads_count(source_is_valid) > 0)
	{
		fval = pgstrom_local_min_fp64(source_is_valid ? fval : DBL_MAX);
		if (get_local_id() == 0)
		{
			kagg_state__pminmax_fp64_packed *r =
				(kagg_state__pminmax_fp64_packed *)buffer;
			if (__isShared(r))
			{
				r->attrs |= __PAGG_MINMAX_ATTRS__VALID;
				if (r->value > fval)
					r->value = fval;
			}
			else
			{
				__atomic_or_uint32(&r->attrs, __PAGG_MINMAX_ATTRS__VALID);
				__atomic_min_fp64(&r->value, fval);
			}
		}
	}
}

/*
 * __update_nogroups__pmax_fp64
 */
INLINE_FUNCTION(void)
__update_nogroups__pmax_fp64(kern_context *kcxt,
							 char *buffer,
							 kern_colmeta *cmeta,
							 kern_aggregate_desc *desc,
							 bool source_is_valid)
{
	float8_t	fval;

	if (source_is_valid)
	{
		const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];

		if (!__preagg_fetch_xdatum_as_float64(&fval, xdatum))
			source_is_valid = false;
	}

	if (__syncthreads_count(source_is_valid) > 0)
	{
		fval = pgstrom_local_max_fp64(source_is_valid ? fval : -DBL_MAX);
		if (get_local_id() == 0)
		{
			kagg_state__pminmax_fp64_packed *r =
				(kagg_state__pminmax_fp64_packed *)buffer;
			if (__isShared(r))
			{
				r->attrs |= __PAGG_MINMAX_ATTRS__VALID;
				if (r->value < fval)
					r->value = fval;
			}
			else
			{
				__atomic_or_uint32(&r->attrs, __PAGG_MINMAX_ATTRS__VALID);
				__atomic_min_fp64(&r->value, fval);
			}
		}
	}
}

/*
 * __update_nogroups__psum_int
 */
INLINE_FUNCTION(void)
__update_nogroups__psum_int(kern_context *kcxt,
							char *buffer,
							kern_colmeta *cmeta,
							kern_aggregate_desc *desc,
							bool source_is_valid)
{
	int64_t		sum, ival = 0;
	int			count;

	if (source_is_valid)
	{
		const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];

		if (!__preagg_fetch_xdatum_as_int64(&ival, xdatum))
			source_is_valid = false;
	}
	count = __syncthreads_count(source_is_valid);
	if (count > 0)
	{
		kagg_state__psum_int_packed *r =
			(kagg_state__psum_int_packed *)buffer;

		pgstrom_stair_sum_int64(ival, &sum);
		if (get_local_id() == 0)
		{
			if (__isShared(r))
			{
				r->nitems += count;
				r->sum    += sum;
			}
			else
			{
				__atomic_add_int64(&r->nitems, count);
				__atomic_add_int64(&r->sum, sum);
			}
		}
	}
}

/*
 * __update_nogroups__psum_int64
 */
INLINE_FUNCTION(void)
__update_nogroups__psum_int64(kern_context *kcxt,
							  char *buffer,
							  kern_colmeta *cmeta,
							  kern_aggregate_desc *desc,
							  bool source_is_valid)
{
	int64_t		ival = 0;
	int			count;

	if (source_is_valid)
	{
		const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];

		if (!__preagg_fetch_xdatum_as_int64(&ival, xdatum))
			source_is_valid = false;
	}
	count = __syncthreads_count(source_is_valid);
	if (count > 0)
	{
		kagg_state__psum_numeric_packed *r =
			(kagg_state__psum_numeric_packed *)buffer;
		int128_t	sum, __temp;

		pgstrom_stair_sum_int128(ival, &sum);
		if (get_local_id() == 0)
		{
			if (__isShared(r))
			{
				r->nitems += count;
				__temp = __fetch_int128_packed(&r->sum);
				__store_int128_packed(&r->sum, __temp + sum);
			}
			else
			{
				__atomic_add_uint64(&r->nitems, count);
				__atomic_add_int128(&r->sum, sum);
			}
		}
	}
}

/*
 * __update_nogroups__psum_fp
 */
INLINE_FUNCTION(void)
__update_nogroups__psum_fp(kern_context *kcxt,
						   char *buffer,
						   kern_colmeta *cmeta,
						   kern_aggregate_desc *desc,
						   bool source_is_valid)
{
	float8_t	sum, fval = 0.0;
	int			count;

	if (source_is_valid)
	{
		const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];

		if (!__preagg_fetch_xdatum_as_float64(&fval, xdatum))
			source_is_valid = false;
	}
	count = __syncthreads_count(source_is_valid);
	if (count > 0)
	{
		kagg_state__psum_fp_packed *r =
			(kagg_state__psum_fp_packed *)buffer;

		pgstrom_stair_sum_fp64(fval, &sum);
		if (get_local_id() == 0)
		{
			if (__isShared(r))
			{
				r->nitems += count;
				r->sum    += sum;
			}
			else
			{
				__atomic_add_int64(&r->nitems, count);
				__atomic_add_fp64(&r->sum, sum);
			}
		}
	}
}

/*
 * __update_nogroups__psum_numeric
 */
INLINE_FUNCTION(void)
__update_nogroups__psum_numeric(kern_context *kcxt,
								char *buffer,
								kern_colmeta *cmeta,
								kern_aggregate_desc *desc,
								bool source_is_valid)
{
	xpu_numeric_t  *xnum = NULL;
	int			count;

	if (source_is_valid)
	{
		xnum = (xpu_numeric_t *)kcxt->kvars_slot[desc->arg0_slot_id];

		if (xnum->expr_ops != &xpu_numeric_ops)
			xnum = NULL;
		else if (!xpu_numeric_validate(kcxt, xnum))
			xnum = NULL;
		//XXX - TODO: Error handling if we could not transform varlena numeric
		//            to int128 form
	}
	count = __syncthreads_count(xnum != NULL);
	if (count > 0)
	{
		kagg_state__psum_numeric_packed *r =
			(kagg_state__psum_numeric_packed *)buffer;
		int128_t	ival = 0;
		uint32_t	special = 0;

		if (xnum)
		{
			if (xnum->kind == XPU_NUMERIC_KIND__VALID)
			{
				int16_t	weight = (int16_t)(r->attrs & __PAGG_NUMERIC_ATTRS__WEIGHT);

				ival = __normalize_numeric_int128(weight, xnum->weight, xnum->u.value);
			}
			else if (xnum->kind == XPU_NUMERIC_KIND__POS_INF)
				special = __PAGG_NUMERIC_ATTRS__PINF;
			else if (xnum->kind == XPU_NUMERIC_KIND__NEG_INF)
				special = __PAGG_NUMERIC_ATTRS__NINF;
			else
				special = XPU_NUMERIC_KIND__NAN;
		}

		if (__syncthreads_count(special != 0) > 0)
		{
			/* Special case handling for Nan,+/-Inf */
			special = pgstrom_local_or_uint32(special);
			if (get_local_id() == 0)
				__atomic_or_uint32(&r->attrs, (special & __PAGG_NUMERIC_ATTRS__MASK));
		}
		else
		{
			/* Elsewhere, it is finite values */
			int128_t	sum, __temp;

			pgstrom_stair_sum_int128(ival, &sum);
			if (get_local_id() == 0)
			{
				if (__isShared(r))
				{
					r->nitems += count;
					__temp = __fetch_int128_packed(&r->sum);
					__store_int128_packed(&r->sum, __temp + sum);
				}
				else
				{
					__atomic_add_uint64(&r->nitems, count);
					__atomic_add_int128(&r->sum, sum);
				}
			}
		}
	}
}

/*
 * __update_nogroups__pstddev
 */
INLINE_FUNCTION(void)
__update_nogroups__pstddev(kern_context *kcxt,
						   char *buffer,
						   kern_colmeta *cmeta,
						   kern_aggregate_desc *desc,
						   bool source_is_valid)
{
	float8_t	xval = 0.0;
	int			count;

	if (source_is_valid)
	{
		const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];

		if (!__preagg_fetch_xdatum_as_float64(&xval, xdatum))
			source_is_valid = false;
	}
	count = __syncthreads_count(source_is_valid);
	if (count > 0)
	{
		float8_t	sum_x, sum_x2;

		pgstrom_stair_sum_fp64(xval, &sum_x);
		pgstrom_stair_sum_fp64(xval * xval, &sum_x2);

		if (get_local_id() == 0)
		{
			kagg_state__stddev_packed *r =
				(kagg_state__stddev_packed *)buffer;
			if (__isShared(r))
			{
				r->nitems += count;
				r->sum_x  += sum_x;
				r->sum_x2 += sum_x2;
			}
			else
			{
				__atomic_add_int64(&r->nitems, count);
				__atomic_add_fp64(&r->sum_x,  sum_x);
				__atomic_add_fp64(&r->sum_x2, sum_x2);
			}
		}
	}
}

/*
 * __update_nogroups__pavg_covar
 */
INLINE_FUNCTION(void)
__update_nogroups__pcovar(kern_context *kcxt,
						  char *buffer,
						  kern_colmeta *cmeta,
						  kern_aggregate_desc *desc,
						  bool source_is_valid)
{
	float8_t	xval = 0.0;
	float8_t	yval = 0.0;
	int			count;

	if (source_is_valid)
	{
		const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];
		const xpu_datum_t *ydatum = kcxt->kvars_slot[desc->arg1_slot_id];

		if (!__preagg_fetch_xdatum_as_float64(&xval, xdatum) ||
			!__preagg_fetch_xdatum_as_float64(&yval, ydatum))
			source_is_valid = false;
	}
	count = __syncthreads_count(source_is_valid);
	if (count > 0)
	{
		float8_t	sum_x, sum_y, sum_xx, sum_yy, sum_xy;

		pgstrom_stair_sum_fp64(xval, &sum_x);
		pgstrom_stair_sum_fp64(yval, &sum_y);
		pgstrom_stair_sum_fp64(xval * xval, &sum_xx);
		pgstrom_stair_sum_fp64(yval * yval, &sum_yy);
		pgstrom_stair_sum_fp64(xval * yval, &sum_xy);

		if (get_local_id() == 0)
		{
			kagg_state__covar_packed *r =
				(kagg_state__covar_packed *)buffer;
			if (__isShared(r))
			{
				r->nitems += count;
				r->sum_x  += sum_x;
				r->sum_xx += sum_xx;
				r->sum_y  += sum_y;
				r->sum_yy += sum_yy;
				r->sum_xy += sum_xy;
			}
			else
			{
				__atomic_add_int64(&r->nitems, count);
				__atomic_add_fp64(&r->sum_x,  sum_x);
				__atomic_add_fp64(&r->sum_xx, sum_xx);
				__atomic_add_fp64(&r->sum_y,  sum_y);
				__atomic_add_fp64(&r->sum_yy, sum_yy);
				__atomic_add_fp64(&r->sum_xy, sum_xy);
			}
		}
	}
}

/*
 * __checkAggFuncFilter
 */
INLINE_FUNCTION(bool)
__checkAggFuncFilter(kern_context *kcxt,
					 const kern_aggregate_desc *desc)
{
	if (desc->filter_slot_id >= 0)
	{
		xpu_bool_t *xdatum = (xpu_bool_t *)
			kcxt->kvars_slot[desc->filter_slot_id];
		assert(desc->filter_slot_id < kcxt->kvars_nslots);
		if (XPU_DATUM_ISNULL(xdatum))
			return false;
		assert(xdatum->expr_ops == &xpu_bool_ops);
		if (!xdatum->value)
			return false;
	}
	return true;
}

/*
 * __updateOneTupleNoGroups
 */
STATIC_FUNCTION(void)
__updateOneTupleNoGroups(kern_context *kcxt,
						 kern_data_store *kds_final,
						 bool source_is_valid,
						 kern_tupitem *tupitem,
						 kern_expression *kexp_groupby_actions)
{
	char   *buffer;
	char   *curr;

	assert(__KDS_TUPITEM_CHECK_VALID(kds_final, tupitem));
	if (kcxt->groupby_prepfn_buffer)
		buffer = kcxt->groupby_prepfn_buffer;
	else
		buffer = KERN_TUPITEM_GET_PAYLOAD(tupitem);
	assert((uintptr_t)buffer == MAXALIGN(buffer));

	curr = buffer;
	for (int j=0; j < kexp_groupby_actions->u.pagg.nattrs; j++)
	{
		kern_aggregate_desc *desc = &kexp_groupby_actions->u.pagg.desc[j];
		kern_colmeta   *cmeta = &kds_final->colmeta[j];
		bool			item_is_valid = source_is_valid;

		/* usually, buffer shall never be mis-aligned before grouping-keys */
		assert((uintptr_t)curr == TYPEALIGN(cmeta->attalign, curr));

		/* check aggregate functions filter, if any */
		if (item_is_valid && !__checkAggFuncFilter(kcxt, desc))
			item_is_valid = false;

		switch (desc->action)
		{
			case KAGG_ACTION__NROWS_ANY:
				__update_nogroups__nrows_any(kcxt, curr,
											 cmeta, desc,
											 item_is_valid);
				break;
			case KAGG_ACTION__NROWS_COND:
				__update_nogroups__nrows_cond(kcxt, curr,
											  cmeta, desc,
											  item_is_valid);
				break;
			case KAGG_ACTION__PMIN_INT32:
				__update_nogroups__pmin_int32(kcxt, curr,
											  cmeta, desc,
											  item_is_valid);
				break;
			case KAGG_ACTION__PMIN_INT64:
				__update_nogroups__pmin_int64(kcxt, curr,
											  cmeta, desc,
											  item_is_valid);
				break;
			case KAGG_ACTION__PMAX_INT32:
				__update_nogroups__pmax_int32(kcxt, curr,
											  cmeta, desc,
											  item_is_valid);
				break;
			case KAGG_ACTION__PMAX_INT64:
				__update_nogroups__pmax_int64(kcxt, curr,
											  cmeta, desc,
											  item_is_valid);
				break;
			case KAGG_ACTION__PMIN_FP64:
				__update_nogroups__pmin_fp64(kcxt, curr,
											 cmeta, desc,
											 item_is_valid);
				break;
			case KAGG_ACTION__PMAX_FP64:
				__update_nogroups__pmax_fp64(kcxt, curr,
											 cmeta, desc,
											 item_is_valid);
				break;
			case KAGG_ACTION__PSUM_INT:
			case KAGG_ACTION__PAVG_INT:
				__update_nogroups__psum_int(kcxt, curr,
											cmeta, desc,
											item_is_valid);
				break;
			case KAGG_ACTION__PSUM_INT64:
			case KAGG_ACTION__PAVG_INT64:
				__update_nogroups__psum_int64(kcxt, curr,
											  cmeta, desc,
											  item_is_valid);
				break;
			case KAGG_ACTION__PAVG_FP:
			case KAGG_ACTION__PSUM_FP:
				__update_nogroups__psum_fp(kcxt, curr,
										   cmeta, desc,
										   item_is_valid);
				break;
			case KAGG_ACTION__PAVG_NUMERIC:
			case KAGG_ACTION__PSUM_NUMERIC:
				assert((uintptr_t)curr == MAXALIGN(curr));
				__update_nogroups__psum_numeric(kcxt, curr,
												cmeta, desc,
												item_is_valid);
				break;
			case KAGG_ACTION__STDDEV:
				__update_nogroups__pstddev(kcxt, curr,
										   cmeta, desc,
										   item_is_valid);
				break;
			case KAGG_ACTION__COVAR:
				__update_nogroups__pcovar(kcxt, curr,
										  cmeta, desc,
										  item_is_valid);
				break;
			default:
				/*
				 * No more partial aggregation exists after
				 * the grouping-keys
				 */
				goto bailout;
		}
		/* move to the next */
		if (cmeta->attlen > 0)
			curr += cmeta->attlen;
        else
		{
			assert(cmeta->attlen == -1);
			curr += VARSIZE_ANY(curr);
		}
	}
bailout:
	assert(curr == buffer + kcxt->groupby_prepfn_bufsz);
}

/*
 * __insertOneTupleNoGroups
 */
STATIC_FUNCTION(kern_tupitem *)
__insertOneTupleNoGroups(kern_context *kcxt,
						 kern_data_store *kds_final,
						 const kern_expression *kexp_groupby_actions)
{
	kern_tupitem   *tupitem;
	int32_t			tupsz;
	uint32_t		required;
	uint64_t		offset;

	assert(kds_final->format == KDS_FORMAT_ROW &&
		   kds_final->hash_nslots == 0);
	/* estimate length */
	tupsz = __writeOutOneTuplePreAgg(kcxt, kds_final, NULL,
									 kexp_groupby_actions);
	assert(tupsz > 0);
	required = TYPEALIGN(CUDA_L1_CACHELINE_SZ,
						 tupsz + kcxt->session->gpusort_htup_margin);
	offset = __atomic_add_uint64(&kds_final->usage, required);
	if (!__KDS_CHECK_OVERFLOW(kds_final, 1, offset + required))
		return NULL;	/* out of memory */
	offset += required;
	tupitem = (kern_tupitem *)((char *)kds_final
							   + kds_final->length
							   - offset);
	__writeOutOneTuplePreAgg(kcxt, kds_final, tupitem,
							 kexp_groupby_actions);
	KERN_TUPITEM_SET_ROWID(tupitem, 0);
	assert(tupitem->t_len == tupsz);
	__threadfence();
	__atomic_write_uint64(KDS_GET_ROWINDEX(kds_final), offset);

	return tupitem;
}

STATIC_FUNCTION(kern_tupitem *)
__execGpuPreAggNoGroups(kern_context *kcxt,
						kern_data_store *kds_final,
						bool source_is_valid,
						int final_depth,
						kern_expression *kexp_groupby_actions)
{
	__shared__ kern_tupitem *tupitem;

	assert(kds_final->format == KDS_FORMAT_ROW);
	assert(kexp_groupby_actions->opcode == FuncOpCode__AggFuncs);
	for (;;)
	{
		if (get_local_id() == 0)
		{
			uint32_t	nitems = __volatileRead(&kds_final->nitems);
			uint32_t	oldval;

			if (nitems == 1)
			{
				/* normal case; destination tuple already exists */
				tupitem = KDS_GET_TUPITEM(kds_final, 0);
				assert(tupitem != NULL);
			}
			else if (nitems == 0)
			{
				oldval = __atomic_cas_uint32(&kds_final->nitems, 0, UINT_MAX);
				if (oldval == 0)
				{
					/* LOCKED */
					tupitem = __insertOneTupleNoGroups(kcxt, kds_final,
													   kexp_groupby_actions);
					if (!tupitem)
					{
						/* UNLOCK */
						oldval = __atomic_write_uint32(&kds_final->nitems, 0);
						assert(oldval == UINT_MAX);
						SUSPEND_NO_SPACE(kcxt, "GpuPreAgg(NoGroup) - no space to write");
					}
					else
					{
						/* UNLOCK */
						oldval = __atomic_write_uint32(&kds_final->nitems, 1);
						assert(oldval == UINT_MAX);
					}
				}
				else
				{
					assert(oldval == 1 || oldval == UINT_MAX);
					tupitem = NULL;
				}
			}
			else
			{
				assert(nitems == UINT_MAX);
				/* works in progress - someone setup the destination tuple */
				tupitem = NULL;
			}
		}
		/* error & suspend checks */
		if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
			return NULL;
		/* is the destination tuple ready? */
		if (tupitem)
			return tupitem;
	}
	return NULL;
}

/*
 * __insertOneTupleGroupBy
 */
STATIC_FUNCTION(kern_hashitem *)
__insertOneTupleGroupBy(kern_context *kcxt,
						kern_data_store *kds_final,
						kern_expression *kexp_groupby_actions)
{
	kern_hashitem  *hitem;
	int32_t			tupsz;
	uint64_t		__usage;
	uint32_t		__nitems;

	assert(kds_final->format == KDS_FORMAT_HASH &&
		   kds_final->hash_nslots > 0);
	/* estimate length */
	tupsz = __writeOutOneTuplePreAgg(kcxt, kds_final, NULL,
									 kexp_groupby_actions);
	assert(tupsz > 0);
	/*
	 * expand kds_final buffer
	 * ------------
	 * NOTE: A L1 Cache Line should not be shared by multiple tuples.
	 *
	 * When we try to add a new tuple on the same L1 cache line that is
	 * partially used by other tuples, this cache line may be already
	 * loaded to other SMs L1 cache.
	 * At CUDA 12.1, we observed the newer tuple that is written in this
	 * L1 cache line is not visible to other SMs. To avoid the problem,
	 * we ensure one L1 cache line (128B) will never store the multiple
	 * tuples. Only kds_final of GpuPreAgg will refer the tuple once
	 * written by other threads.
	 */
	tupsz = TYPEALIGN(CUDA_L1_CACHELINE_SZ,
					  offsetof(kern_hashitem, t) + tupsz
					  + kcxt->session->gpusort_htup_margin);
	for (;;)
	{
		__nitems = __volatileRead(&kds_final->nitems);
		if (__nitems != UINT_MAX &&
			__nitems == __atomic_cas_uint32(&kds_final->nitems,
											__nitems,
											UINT_MAX))	/* LOCK */
		{
			__usage = __volatileRead(&kds_final->usage);
			if (__KDS_CHECK_OVERFLOW(kds_final,
									 __nitems + 1,
									 __usage + tupsz))
			{
				__atomic_add_uint64(&kds_final->usage, tupsz);
				__atomic_write_uint32(&kds_final->nitems,
									  __nitems + 1);	/* UNLOCK */
				break;
			}
			else
			{
				__atomic_write_uint32(&kds_final->nitems,
									  __nitems);		/* UNLOCK */
				return NULL;	/* out of memory */
			}
		}
		//__nanosleep(10);	/* sleep 10ns */
	}
	/* ok, both nitems and usage are valid to write */
	hitem = (kern_hashitem *)((char *)kds_final
							  + kds_final->length
							  - __usage
							  - tupsz);
	__writeOutOneTuplePreAgg(kcxt, kds_final, &hitem->t,
							 kexp_groupby_actions);
	KERN_TUPITEM_SET_ROWID(&hitem->t, __nitems);
	assert(offsetof(kern_hashitem, t) + hitem->t.t_len <= tupsz);
	/*
	 * all setup stuff must be completed before its row-index is visible
	 * to other threads in the device.
	 */
	__threadfence();
	KDS_GET_ROWINDEX(kds_final)[__nitems]
		= ((char *)kds_final
		   + kds_final->length
		   - (char *)&hitem->t);
	return hitem;
}

/*
 * __update_groupby__nrows_any
 */
INLINE_FUNCTION(void)
__update_groupby__nrows_any(kern_context *kcxt,
							char *buffer,
							const kern_colmeta *cmeta,
							const kern_aggregate_desc *desc)
{
	__atomic_add_uint64((uint64_t *)buffer, 1);
	assert(cmeta->attlen == sizeof(uint64_t));
}

INLINE_FUNCTION(void)
__update_groupby__nrows_cond(kern_context *kcxt,
							 char *buffer,
							 const kern_colmeta *cmeta,
							 const kern_aggregate_desc *desc)
{
	const xpu_datum_t  *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];

	if (!XPU_DATUM_ISNULL(xdatum))
		__atomic_add_uint64((uint64_t *)buffer, 1);
	assert(cmeta->attlen == sizeof(uint64_t));
}

INLINE_FUNCTION(void)
__update_groupby__pmin_int32(kern_context *kcxt,
							 char *buffer,
							 const kern_colmeta *cmeta,
							 const kern_aggregate_desc *desc)
{
	const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];
	int32_t		ival;

	if (__preagg_fetch_xdatum_as_int32(&ival, xdatum))
	{
		kagg_state__pminmax_int64_packed *r =
			(kagg_state__pminmax_int64_packed *)buffer;

		__atomic_or_uint32(&r->attrs, __PAGG_MINMAX_ATTRS__VALID);
		__atomic_min_int64(&r->value, ival);
	}
	assert(cmeta->attlen == -1 &&
		   VARSIZE_ANY(buffer) >= sizeof(kagg_state__pminmax_int64_packed));
}

INLINE_FUNCTION(void)
__update_groupby__pmin_int64(kern_context *kcxt,
							 char *buffer,
							 const kern_colmeta *cmeta,
							 const kern_aggregate_desc *desc)
{
	const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];
	int64_t		ival;

	if (__preagg_fetch_xdatum_as_int64(&ival, xdatum))
	{
		kagg_state__pminmax_int64_packed *r =
			(kagg_state__pminmax_int64_packed *)buffer;

		__atomic_or_uint32(&r->attrs, __PAGG_MINMAX_ATTRS__VALID);
		__atomic_min_int64(&r->value, ival);
	}
	assert(cmeta->attlen == -1 &&
		   VARSIZE_ANY(buffer) >= sizeof(kagg_state__pminmax_int64_packed));
}

INLINE_FUNCTION(void)
__update_groupby__pmax_int32(kern_context *kcxt,
							 char *buffer,
							 const kern_colmeta *cmeta,
							 const kern_aggregate_desc *desc)
{
	const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];
	int32_t		ival;

	if (__preagg_fetch_xdatum_as_int32(&ival, xdatum))
	{
		kagg_state__pminmax_int64_packed *r =
			(kagg_state__pminmax_int64_packed *)buffer;

		__atomic_or_uint32(&r->attrs, __PAGG_MINMAX_ATTRS__VALID);
		__atomic_max_int64(&r->value, ival);
	}
	assert(cmeta->attlen == -1 &&
		   VARSIZE_ANY(buffer) >= sizeof(kagg_state__pminmax_int64_packed));
}

INLINE_FUNCTION(void)
__update_groupby__pmax_int64(kern_context *kcxt,
							 char *buffer,
							 const kern_colmeta *cmeta,
							 const kern_aggregate_desc *desc)
{
	const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];
	int64_t		ival;

	if (__preagg_fetch_xdatum_as_int64(&ival, xdatum))
	{
		kagg_state__pminmax_int64_packed *r =
			(kagg_state__pminmax_int64_packed *)buffer;

		__atomic_or_uint32(&r->attrs, __PAGG_MINMAX_ATTRS__VALID);
		__atomic_max_int64(&r->value, ival);
	}
	assert(cmeta->attlen == -1 &&
		   VARSIZE_ANY(buffer) >= sizeof(kagg_state__pminmax_int64_packed));
}

INLINE_FUNCTION(void)
__update_groupby__pmin_fp64(kern_context *kcxt,
							char *buffer,
							const kern_colmeta *cmeta,
							const kern_aggregate_desc *desc)
{
	const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];
	float8_t	fval;

	if (__preagg_fetch_xdatum_as_float64(&fval, xdatum))
	{
		kagg_state__pminmax_fp64_packed *r =
			(kagg_state__pminmax_fp64_packed *)buffer;

		__atomic_or_uint32(&r->attrs, __PAGG_MINMAX_ATTRS__VALID);
		__atomic_min_fp64(&r->value, fval);
	}
	assert(cmeta->attlen == -1 &&
		   VARSIZE_ANY(buffer) >= sizeof(kagg_state__pminmax_fp64_packed));
}

INLINE_FUNCTION(void)
__update_groupby__pmax_fp64(kern_context *kcxt,
							char *buffer,
							const kern_colmeta *cmeta,
							const kern_aggregate_desc *desc)
{
	const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];
	float8_t	fval;

	if (__preagg_fetch_xdatum_as_float64(&fval, xdatum))
	{
		kagg_state__pminmax_fp64_packed *r =
			(kagg_state__pminmax_fp64_packed *)buffer;

		__atomic_or_uint32(&r->attrs, __PAGG_MINMAX_ATTRS__VALID);
		__atomic_max_fp64(&r->value, fval);
	}
	assert(cmeta->attlen == -1 &&
		   VARSIZE_ANY(buffer) >= sizeof(kagg_state__pminmax_fp64_packed));
}

INLINE_FUNCTION(void)
__update_groupby__psum_int(kern_context *kcxt,
						   char *buffer,
						   const kern_colmeta *cmeta,
						   const kern_aggregate_desc *desc)
{
	const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];
	int64_t		ival;

	if (__preagg_fetch_xdatum_as_int64(&ival, xdatum))
	{
		kagg_state__psum_int_packed *r =
			(kagg_state__psum_int_packed *)buffer;

		__atomic_add_int64(&r->nitems, 1);
		__atomic_add_int64(&r->sum, ival);
	}
	assert(cmeta->attlen == -1 &&
		   VARSIZE_ANY(buffer) >= sizeof(kagg_state__psum_int_packed));
}

INLINE_FUNCTION(void)
__update_groupby__psum_int64(kern_context *kcxt,
							 char *buffer,
							 const kern_colmeta *cmeta,
							 const kern_aggregate_desc *desc)
{
	const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];
	int64_t		ival;

	if (__preagg_fetch_xdatum_as_int64(&ival, xdatum))
	{
		kagg_state__psum_numeric_packed *r =
			(kagg_state__psum_numeric_packed *)buffer;

		__atomic_add_uint64(&r->nitems, 1);
		__atomic_add_int128(&r->sum, ival);
	}
	assert(cmeta->attlen == -1 &&
		   VARSIZE_ANY(buffer) >= sizeof(kagg_state__psum_numeric_packed));
}

INLINE_FUNCTION(void)
__update_groupby__psum_fp(kern_context *kcxt,
						  char *buffer,
						  const kern_colmeta *cmeta,
						  const kern_aggregate_desc *desc)
{
	const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];
	float8_t	fval;

	if (__preagg_fetch_xdatum_as_float64(&fval, xdatum))
	{
		kagg_state__psum_fp_packed *r =
			(kagg_state__psum_fp_packed *)buffer;

		__atomic_add_int64(&r->nitems, 1);
		__atomic_add_fp64(&r->sum, fval);
	}
	assert(cmeta->attlen == -1 &&
		   VARSIZE_ANY(buffer) >= sizeof(kagg_state__psum_fp_packed));
}

INLINE_FUNCTION(void)
__update_groupby__psum_numeric(kern_context *kcxt,
							   char *buffer,
							   const kern_colmeta *cmeta,
							   const kern_aggregate_desc *desc)
{
	xpu_numeric_t *xnum = (xpu_numeric_t *)kcxt->kvars_slot[desc->arg0_slot_id];

	if (xnum->expr_ops == &xpu_numeric_ops &&
		xpu_numeric_validate(kcxt, xnum))
	{
		kagg_state__psum_numeric_packed *r =
			(kagg_state__psum_numeric_packed *)buffer;
		if (xnum->kind == XPU_NUMERIC_KIND__VALID)
		{
			int16_t		weight = (int16_t)(r->attrs & __PAGG_NUMERIC_ATTRS__WEIGHT);
			int128_t	ival = __normalize_numeric_int128(weight,
														  xnum->weight,
														  xnum->u.value);
			__atomic_add_uint64(&r->nitems, 1);
			__atomic_add_int128(&r->sum, ival);
		}
		else
		{
			uint32_t	special;

			if (xnum->kind == XPU_NUMERIC_KIND__POS_INF)
				special = __PAGG_NUMERIC_ATTRS__PINF;
			else if (xnum->kind == XPU_NUMERIC_KIND__NEG_INF)
				special = __PAGG_NUMERIC_ATTRS__NINF;
			else
				special = XPU_NUMERIC_KIND__NAN;
			__atomic_or_uint32(&r->attrs, special);
		}
	}
	assert(cmeta->attlen == -1 &&
		   VARSIZE_ANY(buffer) >= sizeof(kagg_state__psum_numeric_packed));
}

INLINE_FUNCTION(void)
__update_groupby__pstddev(kern_context *kcxt,
						  char *buffer,
						  const kern_colmeta *cmeta,
						  const kern_aggregate_desc *desc)
{
	const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];
	float8_t	fval;

	if (__preagg_fetch_xdatum_as_float64(&fval, xdatum))
	{
		kagg_state__stddev_packed *r =
			(kagg_state__stddev_packed *)buffer;

		__atomic_add_int64(&r->nitems, 1);
		__atomic_add_fp64(&r->sum_x,  fval);
		__atomic_add_fp64(&r->sum_x2, fval * fval);
	}
	assert(cmeta->attlen == -1 &&
		   VARSIZE_ANY(buffer) >= sizeof(kagg_state__stddev_packed));
}

INLINE_FUNCTION(void)
__update_groupby__pcovar(kern_context *kcxt,
						 char *buffer,
						 const kern_colmeta *cmeta,
						 const kern_aggregate_desc *desc)
{
	const xpu_datum_t *xdatum = kcxt->kvars_slot[desc->arg0_slot_id];
	const xpu_datum_t *ydatum = kcxt->kvars_slot[desc->arg1_slot_id];
	float8_t		xval, yval;

	if (__preagg_fetch_xdatum_as_float64(&xval, xdatum) &&
		__preagg_fetch_xdatum_as_float64(&yval, ydatum))
	{
		kagg_state__covar_packed *r =
			(kagg_state__covar_packed *)buffer;

		__atomic_add_int64(&r->nitems, 1);
		__atomic_add_fp64(&r->sum_x,  xval);
		__atomic_add_fp64(&r->sum_xx, xval * xval);
		__atomic_add_fp64(&r->sum_y,  yval);
		__atomic_add_fp64(&r->sum_yy, yval * yval);
		__atomic_add_fp64(&r->sum_xy, xval * yval);
	}
	assert(cmeta->attlen == -1 &&
		   VARSIZE_ANY(buffer) >= sizeof(kagg_state__covar_packed));
}

/*
 * __updateOneTupleGroupBy
 */
STATIC_FUNCTION(void)
__updateOneTupleGroupBy(kern_context *kcxt,
						kern_data_store *kds_final,
						kern_tupitem *tupitem,
						const kern_expression *kexp_groupby_actions)
{
	uint32_t	rowid = KERN_TUPITEM_GET_ROWID(tupitem);
	char	   *buffer;
	char	   *curr;

	if (kcxt->groupby_prepfn_buffer &&
		rowid < kcxt->groupby_prepfn_nbufs)
	{
		buffer = (kcxt->groupby_prepfn_buffer +
				  rowid * kcxt->groupby_prepfn_bufsz);
	}
	else
	{
		buffer = KERN_TUPITEM_GET_PAYLOAD(tupitem);
	}
	assert((uintptr_t)buffer == MAXALIGN(buffer));

	curr = buffer;
	for (int j=0; j < kexp_groupby_actions->u.pagg.nattrs; j++)
	{
		const kern_aggregate_desc *desc = &kexp_groupby_actions->u.pagg.desc[j];
		const kern_colmeta *cmeta = &kds_final->colmeta[j];

		/*
		 * PreAgg tuple has strictly aligned partial aggregate function on its earlier
		 * half, so we don't need to align t_hoff here.
		 * In case when 'groupby_prepfn_buffer' is on the kcxt->groupby_prepfn_buffer
		 * (SM's shared memory), its grouping-key portion is not cached. So, alignment
		 * towards varlena datum may cause overrun.
		 */
		assert((uintptr_t)curr == TYPEALIGN(cmeta->attalign, curr));

		/* check FILTER clause of aggregate functions, if any */
		if (__checkAggFuncFilter(kcxt, desc))
		{
			switch (desc->action)
			{
				case KAGG_ACTION__NROWS_ANY:
					__update_groupby__nrows_any(kcxt, curr, cmeta, desc);
					break;
				case KAGG_ACTION__NROWS_COND:
					__update_groupby__nrows_cond(kcxt, curr, cmeta, desc);
					break;
				case KAGG_ACTION__PMIN_INT32:
					__update_groupby__pmin_int32(kcxt, curr, cmeta, desc);
					break;
				case KAGG_ACTION__PMIN_INT64:
					__update_groupby__pmin_int64(kcxt, curr, cmeta, desc);
					break;
				case KAGG_ACTION__PMAX_INT32:
					__update_groupby__pmax_int32(kcxt, curr, cmeta, desc);
					break;
				case KAGG_ACTION__PMAX_INT64:
					__update_groupby__pmax_int64(kcxt, curr, cmeta, desc);
					break;
				case KAGG_ACTION__PMIN_FP64:
					__update_groupby__pmin_fp64(kcxt, curr, cmeta, desc);
					break;
				case KAGG_ACTION__PMAX_FP64:
					__update_groupby__pmax_fp64(kcxt, curr, cmeta, desc);
					break;
				case KAGG_ACTION__PAVG_INT:
				case KAGG_ACTION__PSUM_INT:
					__update_groupby__psum_int(kcxt, curr, cmeta, desc);
					break;
				case KAGG_ACTION__PAVG_INT64:
				case KAGG_ACTION__PSUM_INT64:
					__update_groupby__psum_int64(kcxt, curr, cmeta, desc);
					break;
				case KAGG_ACTION__PAVG_FP:
				case KAGG_ACTION__PSUM_FP:
					__update_groupby__psum_fp(kcxt, curr, cmeta, desc);
					break;
				case KAGG_ACTION__PAVG_NUMERIC:
				case KAGG_ACTION__PSUM_NUMERIC:
					__update_groupby__psum_numeric(kcxt, curr, cmeta, desc);
					break;
				case KAGG_ACTION__STDDEV:
					__update_groupby__pstddev(kcxt, curr, cmeta, desc);
					break;
				case KAGG_ACTION__COVAR:
					__update_groupby__pcovar(kcxt, curr, cmeta, desc);
					break;
				default:
					/*
					 * No partial aggregate function exits after grouping-keys
					 */
					goto bailout;
			}
		}
		/* move to the next */
		if (cmeta->attlen > 0)
			curr += cmeta->attlen;
		else
		{
			assert(cmeta->attlen == -1);
			curr += VARSIZE_ANY(curr);
		}
	}
bailout:
	assert(curr == buffer + kcxt->groupby_prepfn_bufsz);
}

STATIC_FUNCTION(kern_tupitem *)
__execGpuPreAggGroupByHash(kern_context *kcxt,
						   kern_data_store *kds_final,
						   xpu_int4_t *hash,
						   int final_depth,
						   kern_expression *kexp_groupby_keyhash,
						   kern_expression *kexp_groupby_keyload,
						   kern_expression *kexp_groupby_keycomp,
						   kern_expression *kexp_groupby_actions)
{
	kern_hashitem *hitem = NULL;

	/*
	 * lookup the destination grouping tuple. if not found, create a new one.
	 */
	assert(kds_final->format == KDS_FORMAT_HASH);
	do {
		if (!XPU_DATUM_ISNULL(hash) && !hitem)
		{
			uint64_t   *hslot = KDS_GET_HASHSLOT(kds_final, hash->value);
			uint64_t	hoffset;
			uint64_t	saved;
			bool		has_lock = false;
			xpu_bool_t	status;

			hoffset = __volatileRead(hslot);
		try_again:
			for (hitem = KDS_HASH_NEXT_ITEM(kds_final, hoffset);
				 hitem != NULL;
				 hitem = KDS_HASH_NEXT_ITEM(kds_final, hitem->next))
			{
				bool	saved_compare_nulls = kcxt->kmode_compare_nulls;

				if (hitem->t.hash != hash->value)
					continue;
				kcxt->kmode_compare_nulls = true;
				ExecLoadVarsMinimalTuple(kcxt, kexp_groupby_keyload,
										 -2,
										 kds_final,
										 &hitem->t);
				if (EXEC_KERN_EXPRESSION(kcxt, kexp_groupby_keycomp, &status))
				{
					kcxt->kmode_compare_nulls = saved_compare_nulls;
#if 0
					printf("[%d](%c) = [%d](%c) [%.12s](%c) <-> [%.12s](%c)\n",
						   ((xpu_int4_t *)kcxt->kvars_slot[0])->value,
						   XPU_DATUM_ISNULL(kcxt->kvars_slot[0]) ? '*' : '-',
						   ((xpu_int4_t *)kcxt->kvars_slot[2])->value,
						   XPU_DATUM_ISNULL(kcxt->kvars_slot[2]) ? '*' : '-',
						   ((xpu_bpchar_t *)kcxt->kvars_slot[1])->value,
						   XPU_DATUM_ISNULL(kcxt->kvars_slot[1]) ? '*' : '-',
						   ((xpu_bpchar_t *)kcxt->kvars_slot[3])->value,
						   XPU_DATUM_ISNULL(kcxt->kvars_slot[3]) ? '*' : '-');
#endif
					if (!XPU_DATUM_ISNULL(&status) && status.value)
						break;	/* found */
				}
				else
				{
					kcxt->kmode_compare_nulls = saved_compare_nulls;
					if (HandleErrorIfCpuFallback(kcxt, final_depth, 0, false))
						memset(hash, 0, sizeof(xpu_int4_t));
					else
						assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
					hitem = NULL;
					goto skip;
				}
			}

			/* if not found, insert a new empty one under the exclusive lock */
			if (!hitem)
			{
				if (!has_lock)
				{
					/* try lock */
					saved = __volatileRead(hslot);
					if (saved != ULONG_MAX &&
						__atomic_cas_uint64(hslot, saved, ULONG_MAX) == saved)
					{
						has_lock = true;
						hoffset = saved;
						goto try_again;
					}
				}
				else
				{
					hitem = __insertOneTupleGroupBy(kcxt, kds_final,
													kexp_groupby_actions);
					if (hitem)
					{
						/* insert and unlock */
						uint64_t	__offset;

						hitem->next = saved;
						hitem->t.hash = hash->value;
						__threadfence();
						__offset = ((char *)kds_final
									+ kds_final->length
									- (char *)hitem);
						__atomic_write_uint64(hslot, __offset);
					}
					else
					{
						/* out of the memory, and unlcok */
						__atomic_write_uint64(hslot, saved);
						SUSPEND_NO_SPACE(kcxt, "GpuPreAgg(GroupBy) - no space to write");
					}
					has_lock = false;
				}
			}
			else if (has_lock)
			{
				/* unlock */
				__atomic_write_uint64(hslot, saved);
			}
		}
	skip:
		/* error & suspend checks */
		if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
			return NULL;
		/* retry, if any threads are not ready yet */
	} while (__syncthreads_count(!XPU_DATUM_ISNULL(hash) && !hitem) > 0);

	return (hitem ? &hitem->t : NULL);
}

INLINE_FUNCTION(kern_tupitem *)
__execGpuPreAggGroupBy(kern_context *kcxt,
					   kern_data_store *kds_final,
					   bool source_is_valid,
					   int final_depth,
					   kern_expression *kexp_groupby_keyhash,
					   kern_expression *kexp_groupby_keyload,
					   kern_expression *kexp_groupby_keycomp,
					   kern_expression *kexp_groupby_actions)
{
	xpu_int4_t	hash;

	/*
	 * compute hash value of the grouping keys
	 */
	memset(&hash, 0, sizeof(hash));
	if (source_is_valid)
	{
		if (EXEC_KERN_EXPRESSION(kcxt, kexp_groupby_keyhash, &hash))
		{
			assert(!XPU_DATUM_ISNULL(&hash));
		}
		else if (HandleErrorIfCpuFallback(kcxt, final_depth, 0, false))
		{
			memset(&hash, 0, sizeof(hash));
		}
		else
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
		}
	}
	if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
		return NULL;
	return __execGpuPreAggGroupByHash(kcxt,
									  kds_final,
									  &hash,
									  final_depth,
									  kexp_groupby_keyhash,
									  kexp_groupby_keyload,
									  kexp_groupby_keycomp,
									  kexp_groupby_actions);
}

/*
 * setupGpuPreAggGroupByBuffer
 */
STATIC_FUNCTION(void)
__setupGpuPreAggGroupByBufferOne(kern_context *kcxt,
								 const kern_expression *kexp_actions,
								 char *prepfn_buffer)
{
	char   *pos = prepfn_buffer;

	for (int j=0; j < kexp_actions->u.pagg.nattrs; j++)
	{
		const kern_aggregate_desc *desc = &kexp_actions->u.pagg.desc[j];

		switch (desc->action)
		{
			case KAGG_ACTION__NROWS_ANY:
			case KAGG_ACTION__NROWS_COND:
				*((int64_t *)pos) = 0;
				pos += sizeof(int64_t);
				break;

			case KAGG_ACTION__PMIN_INT32:
			case KAGG_ACTION__PMIN_INT64:
				{
					kagg_state__pminmax_int64_packed *r =
						(kagg_state__pminmax_int64_packed *)pos;
					r->attrs = 0;
					r->value = LONG_MAX;
					SET_VARSIZE(r, sizeof(kagg_state__pminmax_int64_packed));
					pos += sizeof(kagg_state__pminmax_int64_packed);
				}
				break;

			case KAGG_ACTION__PMAX_INT32:
			case KAGG_ACTION__PMAX_INT64:
				{
					kagg_state__pminmax_int64_packed *r =
						(kagg_state__pminmax_int64_packed *)pos;
					r->attrs = 0;
					r->value = LONG_MIN;
					SET_VARSIZE(r, sizeof(kagg_state__pminmax_int64_packed));
					pos += sizeof(kagg_state__pminmax_int64_packed);
				}
				break;

			case KAGG_ACTION__PMIN_FP64:
				{
					kagg_state__pminmax_fp64_packed *r =
						(kagg_state__pminmax_fp64_packed *)pos;
					r->attrs = 0;
					r->value = DBL_MAX;
					SET_VARSIZE(r, sizeof(kagg_state__pminmax_fp64_packed));
					pos += sizeof(kagg_state__pminmax_fp64_packed);
				}
				break;

			case KAGG_ACTION__PMAX_FP64:
				{
					kagg_state__pminmax_fp64_packed *r =
                        (kagg_state__pminmax_fp64_packed *)pos;
                    r->attrs = 0;
                    r->value = -DBL_MAX;
                    SET_VARSIZE(r, sizeof(kagg_state__pminmax_fp64_packed));
					pos += sizeof(kagg_state__pminmax_fp64_packed);
				}
				break;

			case KAGG_ACTION__PSUM_INT:
			case KAGG_ACTION__PAVG_INT:
				memset(pos, 0, sizeof(kagg_state__psum_int_packed));
				SET_VARSIZE(pos, sizeof(kagg_state__psum_int_packed));
				pos += sizeof(kagg_state__psum_int_packed);
				break;

			case KAGG_ACTION__PSUM_INT64:
			case KAGG_ACTION__PAVG_INT64:
				memset(pos, 0, sizeof(kagg_state__psum_numeric_packed));
				SET_VARSIZE(pos, sizeof(kagg_state__psum_numeric_packed));
				pos += sizeof(kagg_state__psum_numeric_packed);
				break;

			case KAGG_ACTION__PSUM_FP:
			case KAGG_ACTION__PAVG_FP:
				memset(pos, 0, sizeof(kagg_state__psum_fp_packed));
				SET_VARSIZE(pos, sizeof(kagg_state__psum_fp_packed));
				pos += sizeof(kagg_state__psum_fp_packed);
				break;

			case KAGG_ACTION__PSUM_NUMERIC:
			case KAGG_ACTION__PAVG_NUMERIC:
				{
					kagg_state__psum_numeric_packed *r =
						(kagg_state__psum_numeric_packed *)pos;
					memset(r, 0, sizeof(kagg_state__psum_numeric_packed));
					r->attrs = __numeric_typmod_weight(desc->typmod);
					SET_VARSIZE(r, sizeof(kagg_state__psum_numeric_packed));
					pos += sizeof(kagg_state__psum_numeric_packed);
				}
				break;

			case KAGG_ACTION__STDDEV:
				memset(pos, 0, sizeof(kagg_state__stddev_packed));
				SET_VARSIZE(pos, sizeof(kagg_state__stddev_packed));
				pos += sizeof(kagg_state__stddev_packed);
				break;

			case KAGG_ACTION__COVAR:
				memset(pos, 0, sizeof(kagg_state__covar_packed));
				SET_VARSIZE(pos, sizeof(kagg_state__covar_packed));
				pos += sizeof(kagg_state__covar_packed);
				break;

			default:
				/* no more prep-function should exist after the keyref */
				goto bailout;
		}
	}
bailout:
	assert(pos == prepfn_buffer + kcxt->groupby_prepfn_bufsz);
}

PUBLIC_FUNCTION(void)
setupGpuPreAggGroupByBuffer(kern_context *kcxt,
							kern_gputask *kgtask,
							char *groupby_prepfn_buffer)
{
	const kern_session_info *session = kcxt->session;
	const kern_expression	*kexp_actions = SESSION_KEXP_GROUPBY_ACTIONS(session);

	if (kexp_actions != NULL &&
		kgtask->groupby_prepfn_nbufs > 0)
	{
		assert(kcxt->groupby_prepfn_bufsz == session->groupby_prepfn_bufsz);
		if (SESSION_KEXP_GROUPBY_KEYHASH(session) &&
			SESSION_KEXP_GROUPBY_KEYLOAD(session) &&
			SESSION_KEXP_GROUPBY_KEYCOMP(session))
		{
			uint32_t		index;

			kcxt->groupby_prepfn_nbufs = kgtask->groupby_prepfn_nbufs;
			kcxt->groupby_prepfn_buffer = groupby_prepfn_buffer;

			for (index = get_local_id();
				 index < kcxt->groupby_prepfn_nbufs;
				 index += get_local_size())
			{
				char   *pos = (kcxt->groupby_prepfn_buffer +
							   kcxt->groupby_prepfn_bufsz * index);
				__setupGpuPreAggGroupByBufferOne(kcxt, kexp_actions, pos);
			}
		}
		else
		{
			assert(kgtask->groupby_prepfn_nbufs == 1);
			kcxt->groupby_prepfn_nbufs = 1;
			kcxt->groupby_prepfn_buffer = groupby_prepfn_buffer;
			if (get_local_id() == 0)
				__setupGpuPreAggGroupByBufferOne(kcxt, kexp_actions,
												 kcxt->groupby_prepfn_buffer);
		}
	}
}

/*
 * mergeGpuPreAggGroupByBuffer
 */
STATIC_FUNCTION(void)
__mergeGpuPreAggGroupByBufferOne(kern_context *kcxt,
								 kern_data_store *kds_final,
								 kern_tupitem *tupitem,
								 const kern_expression *kexp_actions,
								 const char *prepfn_buffer)
{
	int			nattrs = Min(kds_final->ncols, kexp_actions->u.pagg.nattrs);
	int			t_hoff = 0;
	char	   *base = KERN_TUPITEM_GET_PAYLOAD(tupitem);
	const char *pos = prepfn_buffer;
	uint32_t	nbytes;

	/* walk on the columns */
	for (int j=0; j < nattrs; j++)
	{
		const kern_aggregate_desc *desc = &kexp_actions->u.pagg.desc[j];
		const kern_colmeta *cmeta = &kds_final->colmeta[j];

		assert(t_hoff == TYPEALIGN(cmeta->attalign, t_hoff));
		switch (desc->action)
		{
			case KAGG_ACTION__NROWS_ANY:
			case KAGG_ACTION__NROWS_COND:
				{
					uint64_t	ival = *((const uint64_t *)pos);

					if (ival > 0)
						__atomic_add_uint64((uint64_t *)(base + t_hoff), ival);
					nbytes = sizeof(int64_t);
				}
				break;

			case KAGG_ACTION__PMIN_INT32:
			case KAGG_ACTION__PMIN_INT64:
				{
					const kagg_state__pminmax_int64_packed *s =
						(const kagg_state__pminmax_int64_packed *)pos;
					kagg_state__pminmax_int64_packed *r =
						(kagg_state__pminmax_int64_packed *)(base + t_hoff);
					if ((s->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)
					{
						__atomic_or_uint32(&r->attrs, s->attrs);
						__atomic_min_int64(&r->value, s->value);
					}
					nbytes = sizeof(kagg_state__pminmax_int64_packed);
				}
				break;

			case KAGG_ACTION__PMAX_INT32:
			case KAGG_ACTION__PMAX_INT64:
				{
					const kagg_state__pminmax_int64_packed *s =
						(const kagg_state__pminmax_int64_packed *)pos;
					kagg_state__pminmax_int64_packed *r =
						(kagg_state__pminmax_int64_packed *)(base + t_hoff);
					if ((s->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)
					{
						__atomic_or_uint32(&r->attrs, s->attrs);
						__atomic_max_int64(&r->value, s->value);
					}
					nbytes = sizeof(kagg_state__pminmax_int64_packed);
				}
				break;

			case KAGG_ACTION__PMIN_FP64:
				{
					const kagg_state__pminmax_fp64_packed *s =
						(const kagg_state__pminmax_fp64_packed *)pos;
					kagg_state__pminmax_fp64_packed *r =
						(kagg_state__pminmax_fp64_packed *)(base + t_hoff);
					if ((s->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)
					{
						__atomic_or_uint32(&r->attrs, s->attrs);
						__atomic_min_fp64(&r->value, s->value);
					}
					nbytes = sizeof(kagg_state__pminmax_fp64_packed);
				}
				break;

			case KAGG_ACTION__PMAX_FP64:
				{
					const kagg_state__pminmax_fp64_packed *s =
						(const kagg_state__pminmax_fp64_packed *)pos;
					kagg_state__pminmax_fp64_packed *r =
						(kagg_state__pminmax_fp64_packed *)(base + t_hoff);
					if ((s->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)
					{
						__atomic_or_uint32(&r->attrs, s->attrs);
						__atomic_max_fp64(&r->value, s->value);
					}
					nbytes = sizeof(kagg_state__pminmax_fp64_packed);
				}
				break;

			case KAGG_ACTION__PSUM_INT:
			case KAGG_ACTION__PAVG_INT:
				{
					const kagg_state__psum_int_packed *s =
						(const kagg_state__psum_int_packed *)pos;
					kagg_state__psum_int_packed *r =
						(kagg_state__psum_int_packed *)(base + t_hoff);
					if (s->nitems > 0)
					{
						__atomic_add_int64(&r->nitems, s->nitems);
						__atomic_add_int64(&r->sum, s->sum);
					}
					nbytes = sizeof(kagg_state__psum_int_packed);
				}
				break;

			case KAGG_ACTION__PSUM_FP:
			case KAGG_ACTION__PAVG_FP:
				{
					const kagg_state__psum_fp_packed *s =
						(const kagg_state__psum_fp_packed *)pos;
					kagg_state__psum_fp_packed *r =
						(kagg_state__psum_fp_packed *)(base + t_hoff);
					if (s->nitems > 0)
					{
						__atomic_add_int64(&r->nitems, s->nitems);
						__atomic_add_fp64(&r->sum, s->sum);
					}
					nbytes = sizeof(kagg_state__psum_fp_packed);
				}
				break;

			case KAGG_ACTION__PSUM_INT64:
			case KAGG_ACTION__PAVG_INT64:
			case KAGG_ACTION__PSUM_NUMERIC:
			case KAGG_ACTION__PAVG_NUMERIC:
				{
					const kagg_state__psum_numeric_packed *s =
						(const kagg_state__psum_numeric_packed *)pos;
                    kagg_state__psum_numeric_packed *r =
                        (kagg_state__psum_numeric_packed *)(base + t_hoff);
					if (s->nitems > 0)
					{
						/* weight must be equal */
						uint32_t	special = (s->attrs & __PAGG_NUMERIC_ATTRS__MASK);

						assert((s->attrs & __PAGG_NUMERIC_ATTRS__WEIGHT) ==
							   (r->attrs & __PAGG_NUMERIC_ATTRS__WEIGHT));
						__atomic_or_uint32(&r->attrs, special);
						__atomic_add_uint64(&r->nitems, s->nitems);
						__atomic_add_int128(&r->sum, __fetch_int128_packed(&s->sum));
					}
					nbytes = sizeof(kagg_state__psum_numeric_packed);
				}
				break;

			case KAGG_ACTION__STDDEV:
				{
					const kagg_state__stddev_packed *s =
						(const kagg_state__stddev_packed *)pos;
					kagg_state__stddev_packed *r =
						(kagg_state__stddev_packed *)(base + t_hoff);
					if (s->nitems > 0)
					{
						__atomic_add_int64(&r->nitems, s->nitems);
						__atomic_add_fp64(&r->sum_x,  s->sum_x);
						__atomic_add_fp64(&r->sum_x2, s->sum_x2);
					}
					nbytes = sizeof(kagg_state__stddev_packed);
				}
				break;

			case KAGG_ACTION__COVAR:
				{
					const kagg_state__covar_packed *s =
						(const kagg_state__covar_packed *)pos;
					kagg_state__covar_packed *r =
						(kagg_state__covar_packed *)(base + t_hoff);
					if (s->nitems > 0)
					{
						__atomic_add_int64(&r->nitems, s->nitems);
						__atomic_add_fp64(&r->sum_x,  s->sum_x);
						__atomic_add_fp64(&r->sum_xx, s->sum_xx);
						__atomic_add_fp64(&r->sum_y,  s->sum_y);
						__atomic_add_fp64(&r->sum_yy, s->sum_yy);
						__atomic_add_fp64(&r->sum_xy, s->sum_xy);
					}
					nbytes = sizeof(kagg_state__covar_packed);
				}
				break;

			default:
				goto bailout;
		}
		t_hoff += nbytes;
		pos += nbytes;
	}
bailout:
	assert(pos == prepfn_buffer + kcxt->groupby_prepfn_bufsz);
}

PUBLIC_FUNCTION(void)
mergeGpuPreAggGroupByBuffer(kern_context *kcxt,
							kern_data_store *kds_final)
{
	const kern_session_info *session = kcxt->session;
	const kern_expression *kexp_groupby_actions = SESSION_KEXP_GROUPBY_ACTIONS(session);

	if (!kcxt->groupby_prepfn_buffer ||
		!kexp_groupby_actions)
		return;		/* nothing to do */

	if (SESSION_KEXP_GROUPBY_KEYHASH(session) &&
		SESSION_KEXP_GROUPBY_KEYLOAD(session) &&
		SESSION_KEXP_GROUPBY_KEYCOMP(session))
	{
		/*
		 * merge the group-by results
		 *
		 * NOTE: about availability of the tuples on kds_final.
		 *
		 * __execGpuPreAggGroupBy() builds result tuples on the kds_final
		 * as follows:
		 * 1. acquires the exclusive lock on the hash-slot.
		 * 2. expand nitems/usage of kds_final by atomicCAS
		 *   (at this timing, result tuple is not built yet)
		 * 3. set up an empty tuple using __writeOutOneTuplePreAgg()
		 * 4. injection of memory-barrier
		 * 5. assign row offset on the row-index of the kds_final.
		 *
		 * It means that @kds_final->nitems does not guarantee the existence of
		 * the result tuple, however, valid row-index of the kds_final guarantees
		 * existence of the result tuple successfully built.
		 * So, in case when row-index is less than @kds_final->nitem, and its
		 * @tupitem is valid, we can consider the result tuple has been already
		 * successfully built.
		 *
		 * In addition, merge operation touches the kds_final, only if buffered
		 * prep-function state has been updated by at least one row. It means
		 * this row is already built on the kds_final, thus ready to update.
		 */
		uint32_t	nitems = __volatileRead(&kds_final->nitems);
		uint32_t	index;

		nitems = Min(nitems, kcxt->groupby_prepfn_nbufs);
		for (index = get_local_id();
			 index < nitems;
			 index += get_local_size())
		{
			kern_tupitem   *tupitem = KDS_GET_TUPITEM(kds_final, index);
			char		   *pos = (kcxt->groupby_prepfn_buffer +
								   kcxt->groupby_prepfn_bufsz * index);
			if (!tupitem)
				continue;
			__mergeGpuPreAggGroupByBufferOne(kcxt, kds_final,
											 tupitem,
											 kexp_groupby_actions,
											 pos);
		}
	}
	else
	{
		/* merge the no-group results */
		uint32_t	nitems = __volatileRead(&kds_final->nitems);

		if (get_local_id() == 0 && nitems > 0)
		{
			kern_tupitem *tupitem = KDS_GET_TUPITEM(kds_final, 0);

			__mergeGpuPreAggGroupByBufferOne(kcxt, kds_final,
											 tupitem,
											 kexp_groupby_actions,
											 kcxt->groupby_prepfn_buffer);
		}
	}
}

PUBLIC_FUNCTION(int)
execGpuPreAggGroupBy(kern_context *kcxt,
					 kern_warp_context *wp,
					 int n_rels,
					 kern_data_store *kds_final,
					 char *src_kvecs_buffer)
{
	kern_session_info *session = kcxt->session;
	kern_expression *kexp_groupby_keyhash = SESSION_KEXP_GROUPBY_KEYHASH(session);
	kern_expression *kexp_groupby_keyload = SESSION_KEXP_GROUPBY_KEYLOAD(session);
	kern_expression *kexp_groupby_keycomp = SESSION_KEXP_GROUPBY_KEYCOMP(session);
	kern_expression *kexp_groupby_actions = SESSION_KEXP_GROUPBY_ACTIONS(session);
	kern_expression *karg;
	uint32_t	rd_pos = WARP_READ_POS(wp,n_rels);
	uint32_t	wr_pos = WARP_WRITE_POS(wp,n_rels);
	kern_tupitem *tupitem = NULL;

	/*
	 * The previous depth still may produce new tuples, and number of
	 * the current result tuples is not sufficient to run projection.
	 */
	if (wp->scan_done <= n_rels && rd_pos + get_local_size() > wr_pos)
		return n_rels;

	rd_pos += get_local_id();
	kcxt->kvecs_curr_id = (rd_pos % KVEC_UNITSZ);
	kcxt->kvecs_curr_buffer = src_kvecs_buffer;
	if (__syncthreads_count(rd_pos < wr_pos) == 0)
		goto skip_reduction;

	/*
	 * fillup the kvars_slot if it involves expressions
	 */
	if (rd_pos < wr_pos)
	{
		int		i;

		for (i=0, karg = KEXP_FIRST_ARG(kexp_groupby_actions);
			 i < kexp_groupby_actions->nr_args;
			 i++, karg = KEXP_NEXT_ARG(karg))
		{
			assert(karg->opcode == FuncOpCode__SaveExpr);
			if (!EXEC_KERN_EXPRESSION(kcxt, karg, NULL))
			{
				assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
				break;
			}
		}
	}
	if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
		return -1;

	/*
	 * main logic of GpuPreAgg
	 */
	assert(kexp_groupby_actions != NULL);
	if (kexp_groupby_keyhash &&
		kexp_groupby_keyload &&
		kexp_groupby_keycomp)
	{
		tupitem = __execGpuPreAggGroupBy(kcxt, kds_final,
										 (rd_pos < wr_pos),
										 n_rels + 1,
										 kexp_groupby_keyhash,
										 kexp_groupby_keyload,
										 kexp_groupby_keycomp,
										 kexp_groupby_actions);
		if (tupitem)
			__updateOneTupleGroupBy(kcxt, kds_final,
									tupitem,
									kexp_groupby_actions);
	}
	else
	{
		tupitem = __execGpuPreAggNoGroups(kcxt, kds_final,
										  (rd_pos < wr_pos),
										  n_rels + 1,
										  kexp_groupby_actions);
		if (tupitem)
			__updateOneTupleNoGroups(kcxt, kds_final,
									 (rd_pos < wr_pos),
									 tupitem,
									 kexp_groupby_actions);
	}
	if (__syncthreads_count(!tupitem) > 0)
		return -1;

	/*
	 * Update the read position
	 */
skip_reduction:
	if (get_local_id() == 0)
		WARP_READ_POS(wp,n_rels) = Min(WARP_WRITE_POS(wp,n_rels),
									   WARP_READ_POS(wp,n_rels) + get_local_size());
	__syncthreads();
	if (wp->scan_done <= n_rels)
	{
		if (WARP_WRITE_POS(wp,n_rels) < WARP_READ_POS(wp,n_rels) + get_local_size())
			return n_rels;	/* back to the previous depth */
    }
	else
	{
		if (WARP_READ_POS(wp,n_rels) >= WARP_WRITE_POS(wp,n_rels))
			return -1;		/* ok, end of GpuPreAgg */
	}
	return n_rels + 1;		/* elsewhere, try again? */
}

/*
 * kern_gpupreagg_final_merge
 */
KERNEL_FUNCTION(void)
kern_gpupreagg_final_merge(kern_session_info *session,
						   kern_gputask *kgtask,
						   kern_data_store *kds_src,
						   kern_data_store *kds_dst)
{
	kern_context    *kcxt;
	kern_expression *kexp_groupby_keyhash = SESSION_KEXP_GROUPBY_KEYHASH(session);
	kern_expression *kexp_groupby_keyload = SESSION_KEXP_GROUPBY_KEYLOAD(session);
	kern_expression *kexp_groupby_keycomp = SESSION_KEXP_GROUPBY_KEYCOMP(session);
	kern_expression *kexp_groupby_actions = SESSION_KEXP_GROUPBY_ACTIONS(session);
	int			final_depth = kgtask->n_rels + 1;
	uint32_t	base;

	/* sanity checks */
	assert(kgtask->right_outer_depth == 0 ? kds_src != NULL : kds_src == NULL);
	/* save the GPU-Task specific read-only properties */
	if (get_local_id() == 0)
	{
		stromTaskProp__cuda_dindex        = kgtask->cuda_dindex;
		stromTaskProp__cuda_stack_limit   = kgtask->cuda_stack_limit;
		stromTaskProp__partition_divisor  = kgtask->partition_divisor;
		stromTaskProp__partition_reminder = kgtask->partition_reminder;
	}
	/* setup execution context */
	INIT_KERNEL_CONTEXT(kcxt, session, NULL);

	for (base = get_global_base(); base < kds_src->nitems; base += get_global_size())
	{
		kern_tupitem *tupitem_src = NULL;
		kern_tupitem *tupitem_dst = NULL;
		uint32_t	index = base + get_local_id();

		/*
		 * Extract the source tuple to the current kvars_slot
		 */
		if (index < kds_src->nitems)
			tupitem_src = KDS_GET_TUPITEM(kds_src, index);
		ExecLoadKeysFromGroupByFinal(kcxt, kds_src, tupitem_src,
									 kexp_groupby_actions);
		if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
			break;
		/*
		 * Merge the final results on the destination buffer
		 */
		if (kexp_groupby_keyhash &&
			kexp_groupby_keyload &&
			kexp_groupby_keycomp)
		{
			xpu_int4_t	hash;

			if (!tupitem_src)
				memset(&hash, 0, sizeof(xpu_int4_t));	/* NULL */
			else
			{
				kern_hashitem *hitem = (kern_hashitem *)
					((char *)tupitem_src - offsetof(kern_hashitem, t));
				hash.expr_ops = &xpu_int4_ops;
				hash.value = hitem->t.hash;
			}
			tupitem_dst = __execGpuPreAggGroupByHash(kcxt, kds_dst,
													 &hash,
													 final_depth,
													 kexp_groupby_keyhash,
													 kexp_groupby_keyload,
													 kexp_groupby_keycomp,
													 kexp_groupby_actions);
			if (tupitem_src && tupitem_dst)
			{
				const char *prepfn_buffer
					= KERN_TUPITEM_GET_PAYLOAD(tupitem_src);
				__mergeGpuPreAggGroupByBufferOne(kcxt, kds_dst,
												 tupitem_dst,
												 kexp_groupby_actions,
												 prepfn_buffer);
				KDS_GET_ROWINDEX(kds_src)[index] = 0UL;	/* invalidate source */
			}
		}
		else
		{
			tupitem_dst = __execGpuPreAggNoGroups(kcxt, kds_dst,
												  (tupitem_src != NULL),
												  final_depth,
												  kexp_groupby_actions);
			/* merge the final results */
			if (tupitem_src && tupitem_dst)
			{
				const char *prepfn_buffer
					= KERN_TUPITEM_GET_PAYLOAD(tupitem_src);
				__mergeGpuPreAggGroupByBufferOne(kcxt, kds_dst,
												 tupitem_dst,
												 kexp_groupby_actions,
												 prepfn_buffer);
				KDS_GET_ROWINDEX(kds_src)[index] = 0UL;	/* invalidate source */
			}
		}
		/* error checks */
		if (__syncthreads_count(kcxt->errcode != ERRCODE_STROM_SUCCESS) > 0)
			break;
	}
	STROM_WRITEBACK_ERROR_STATUS(&kgtask->kerror, kcxt);
}
