/*
 * cuda_gpusort.cu
 *
 * Device implementation of GpuSort
 * ----
 * Copyright 2011-2025 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2025 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "cuda_common.h"
#include "float2.h"

INLINE_FUNCTION(int)
__gpusort_comp_rawkey(kern_context *kcxt,
					  const kern_sortkey_desc *sdesc,
					  const kern_data_store *kds_final,
					  const kern_tupitem *titem_x,
					  const kern_tupitem *titem_y)
{
	const void *addr_x = kern_fetch_minimal_tuple_attr(kds_final, titem_x,
													   sdesc->src_anum);
	const void *addr_y = kern_fetch_minimal_tuple_attr(kds_final, titem_y,
													   sdesc->src_anum);

	if (addr_x && addr_y)
	{
		const xpu_datum_operators *key_ops = sdesc->key_ops;
		xpu_datum_t	   *datum_x;
		xpu_datum_t	   *datum_y;
		int				sz, comp;

		/*
		 * !!!workaround for a bug!!!
		 *
		 * A couple of identical alloca() calls below were unintentionally
		 * optimized by the compiler.
		 * Probably, compiler considered that alloca() will return same
		 * value for the identical argument, thus datum_x and datum_y will
		 * have same value.
		 *
		 * datum_x = (xpu_datum_t *)alloca(key_ops->xpu_type_sizeof);
		 * datum_y = (xpu_datum_t *)alloca(key_ops->xpu_type_sizeof);
		 *
		 * If alloca() would be an immutable function, it is a right assumption,
		 * however, alloca() modified the current stack frame and allocates
		 * a temporary buffer. So, datum_x and datum_y should be different
		 * pointers.
		 */
		sz = TYPEALIGN(16, key_ops->xpu_type_sizeof);
		datum_x = (xpu_datum_t *)alloca(2 * sz);
		datum_y = (xpu_datum_t *)((char *)datum_x + sz);
		if (key_ops->xpu_datum_heap_read(kcxt, addr_x, datum_x) &&
			key_ops->xpu_datum_heap_read(kcxt, addr_y, datum_y) &&
			key_ops->xpu_datum_comp(kcxt, &comp, datum_x, datum_y))
			return sdesc->order_asc ? comp : -comp;
	}
	else if (addr_x && !addr_y)
		return (sdesc->nulls_first ? 1 : -1);	/* X is NOT NULL, Y is NULL */
	else if (!addr_x && addr_y)
		return (sdesc->nulls_first ? -1 : 1);	/* X is NULL, Y is NOT NULL */
	return 0;
}

INLINE_FUNCTION(int)
__gpusort_comp_pminmax_int64(kern_context *kcxt,
							 const kern_sortkey_desc *sdesc,
							 const kern_data_store *kds_final,
							 const kern_tupitem *titem_x,
							 const kern_tupitem *titem_y)
{
	const kagg_state__pminmax_int64_packed *x = (const kagg_state__pminmax_int64_packed *)
		kern_fetch_minimal_tuple_attr(kds_final, titem_x, sdesc->src_anum);
	const kagg_state__pminmax_int64_packed *y = (const kagg_state__pminmax_int64_packed *)
		kern_fetch_minimal_tuple_attr(kds_final, titem_y, sdesc->src_anum);
	if (x && (x->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)
	{
		if (y && (y->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)
		{
			if (x->value < y->value)
				return (sdesc->order_asc ? -1 : 1);
			if (x->value > y->value)
				return (sdesc->order_asc ? 1 : -1);
			return 0;
		}
		else
			return (sdesc->nulls_first ? 1 : -1);	/* X is NOT NULL, Y is NULL */
	}
	else if (y && (y->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)
		return (sdesc->nulls_first ? -1 : 1);		/* X is NULL, Y is NOT NULL */
	return 0;
}

INLINE_FUNCTION(int)
__gpusort_comp_pminmax_fp64(kern_context *kcxt,
							const kern_sortkey_desc *sdesc,
							const kern_data_store *kds_final,
							const kern_tupitem *titem_x,
							const kern_tupitem *titem_y)
{
	const kagg_state__pminmax_fp64_packed *x = (const kagg_state__pminmax_fp64_packed *)
		kern_fetch_minimal_tuple_attr(kds_final, titem_x, sdesc->src_anum);
	const kagg_state__pminmax_fp64_packed *y = (const kagg_state__pminmax_fp64_packed *)
		kern_fetch_minimal_tuple_attr(kds_final, titem_y, sdesc->src_anum);
	if (x && (x->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)
	{
		if (y && (y->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)
		{
			if (x->value < y->value)
				return (sdesc->order_asc ? -1 : 1);
			if (x->value > y->value)
				return (sdesc->order_asc ? 1 : -1);
			return 0;
		}
		else
			return (sdesc->nulls_first ? 1 : -1);	/* X is NOT NULL, Y is NULL */
	}
	else if (y && (y->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)
		return (sdesc->nulls_first ? -1 : 1);		/* X is NULL, Y is NOT NULL */
	return 0;
}

INLINE_FUNCTION(int)
__gpusort_comp_psum_int64(kern_context *kcxt,
						  const kern_sortkey_desc *sdesc,
						  const kern_data_store *kds_final,
						  const kern_tupitem *titem_x,
						  const kern_tupitem *titem_y)
{
	const kagg_state__psum_int_packed *x = (const kagg_state__psum_int_packed *)
		kern_fetch_minimal_tuple_attr(kds_final, titem_x, sdesc->src_anum);
	const kagg_state__psum_int_packed *y = (const kagg_state__psum_int_packed *)
		kern_fetch_minimal_tuple_attr(kds_final, titem_y, sdesc->src_anum);
	if (x && x->nitems > 0)
	{
		if (y && y->nitems > 0)
		{
			if (x->sum < y->sum)
				return (sdesc->order_asc ? -1 : 1);
			if (x->sum > y->sum)
				return (sdesc->order_asc ? 1 : -1);
			return 0;
		}
		else
			return (sdesc->nulls_first ? 1 : -1);	/* X!=NULL and Y==NULL */
	}
	else if (y && y->nitems > 0)
		return (sdesc->nulls_first ? -1 : 1);		/* X==NULL and Y!=NULL */
	return 0;
}

INLINE_FUNCTION(int)
__gpusort_comp_psum_fp64(kern_context *kcxt,
						  const kern_sortkey_desc *sdesc,
						  const kern_data_store *kds_final,
						  const kern_tupitem *titem_x,
						  const kern_tupitem *titem_y)
{
	const kagg_state__psum_fp_packed *x = (const kagg_state__psum_fp_packed *)
		kern_fetch_minimal_tuple_attr(kds_final, titem_x, sdesc->src_anum);
	const kagg_state__psum_fp_packed *y = (const kagg_state__psum_fp_packed *)
		kern_fetch_minimal_tuple_attr(kds_final, titem_y, sdesc->src_anum);
	if (x && x->nitems > 0)
	{
		if (y && y->nitems > 0)
		{
			if (x->sum < y->sum)
				return (sdesc->order_asc ? -1 : 1);
			if (x->sum > y->sum)
				return (sdesc->order_asc ? 1 : -1);
			return 0;
		}
		else
			return (sdesc->nulls_first ? 1 : -1);	/* X!=NULL and Y==NULL */
	}
	else if (y && y->nitems > 0)
		return (sdesc->nulls_first ? -1 : 1);		/* X==NULL and Y!=NULL */
	return 0;
}

INLINE_FUNCTION(int)
__gpusort_comp_psum_numeric(kern_context *kcxt,
							const kern_sortkey_desc *sdesc,
							const kern_data_store *kds_final,
							const kern_tupitem *titem_x,
							const kern_tupitem *titem_y)
{
	const kagg_state__psum_numeric_packed *x = (const kagg_state__psum_numeric_packed *)
		kern_fetch_minimal_tuple_attr(kds_final, titem_x, sdesc->src_anum);
	const kagg_state__psum_numeric_packed *y = (const kagg_state__psum_numeric_packed *)
		kern_fetch_minimal_tuple_attr(kds_final, titem_y, sdesc->src_anum);
	if (x && x->nitems > 0)
	{
		if (y && y->nitems > 0)
		{
			xpu_numeric_t x_datum;
			xpu_numeric_t y_datum;
			int		xspecial = (x->attrs & __PAGG_NUMERIC_ATTRS__MASK);
			int		yspecial = (y->attrs & __PAGG_NUMERIC_ATTRS__MASK);
			int		comp;

			if (xspecial == 0)
			{
				x_datum.kind = XPU_NUMERIC_KIND__VALID;
				x_datum.weight = (int16_t)(x->attrs & __PAGG_NUMERIC_ATTRS__WEIGHT);
				x_datum.u.value = __fetch_int128_packed(&x->sum);
			}
			else if (xspecial == __PAGG_NUMERIC_ATTRS__PINF)
				x_datum.kind = XPU_NUMERIC_KIND__POS_INF;
			else if (xspecial == __PAGG_NUMERIC_ATTRS__NINF)
				x_datum.kind = XPU_NUMERIC_KIND__NEG_INF;
			else
				x_datum.kind = XPU_NUMERIC_KIND__NAN;
			x_datum.expr_ops = &xpu_numeric_ops;

			if (yspecial == 0)
			{
				y_datum.kind = XPU_NUMERIC_KIND__VALID;
				y_datum.weight = (int16_t)(y->attrs & __PAGG_NUMERIC_ATTRS__WEIGHT);
				y_datum.u.value = __fetch_int128_packed(&y->sum);
			}
			else if (yspecial == __PAGG_NUMERIC_ATTRS__PINF)
				y_datum.kind = XPU_NUMERIC_KIND__POS_INF;
			else if (yspecial == __PAGG_NUMERIC_ATTRS__NINF)
				y_datum.kind = XPU_NUMERIC_KIND__NEG_INF;
			else
				y_datum.kind = XPU_NUMERIC_KIND__NAN;
			y_datum.expr_ops = &xpu_numeric_ops;

			sdesc->key_ops->xpu_datum_comp(kcxt,
										   &comp,
										   (xpu_datum_t *)&x_datum,
										   (xpu_datum_t *)&y_datum);
			return (sdesc->order_asc ? comp : -comp);
		}
		else
			return (sdesc->nulls_first ? 1 : -1)	/* X!=NULL, Y==NULL */;
	}
	else if (y && y->nitems > 0)
		return (sdesc->nulls_first ? -1 : 1);		/* X==NULL, Y!=NULL */
	return 0;
}

INLINE_FUNCTION(int)
__gpusort_comp_precomp_fp64(const kern_sortkey_desc *sdesc,
							const kern_tupitem *titem_x,
							const kern_tupitem *titem_y)
{
	const char *addr_x = ((char *)titem_x + titem_x->t_len + sdesc->buf_offset);
	const char *addr_y = ((char *)titem_y + titem_y->t_len + sdesc->buf_offset);
	bool		notnull_x = *addr_x++;
	bool		notnull_y = *addr_y++;
	float8_t	fval_x;
	float8_t	fval_y;

	if (notnull_x && notnull_y)
	{
		memcpy(&fval_x, addr_x, sizeof(float8_t));
		memcpy(&fval_y, addr_y, sizeof(float8_t));
		if (fval_x < fval_y)
			return (sdesc->order_asc ? -1 : 1);
		if (fval_x > fval_y)
			return (sdesc->order_asc ? 1 : -1);
	}
	else if (notnull_x && !notnull_y)
		return (sdesc->nulls_first ? 1 : -1);
	else if (!notnull_x && notnull_y)
		return (sdesc->nulls_first ? -1 : 1);
	return 0;
}

INLINE_FUNCTION(int)
__gpusort_comp_keys(kern_context *kcxt,
					const kern_expression *sort_kexp,
					const kern_data_store *kds_final,
					const kern_tupitem *titem_x,
					const kern_tupitem *titem_y)
{
	if (!titem_x)
		return (!titem_y ? 0 : 1);
	else if (!titem_y)
		return -1;

	for (int k=0; k < sort_kexp->u.sort.nkeys; k++)
	{
		const kern_sortkey_desc *sdesc = &sort_kexp->u.sort.desc[k];
		int		comp;

		switch (sdesc->kind)
		{
			case KSORT_KEY_KIND__VREF:
				comp = __gpusort_comp_rawkey(kcxt, sdesc, kds_final, titem_x, titem_y);
				break;
			case KSORT_KEY_KIND__PMINMAX_INT64:
				comp = __gpusort_comp_pminmax_int64(kcxt, sdesc, kds_final, titem_x, titem_y);
				break;
			case KSORT_KEY_KIND__PMINMAX_FP64:
				comp = __gpusort_comp_pminmax_fp64(kcxt, sdesc, kds_final, titem_x, titem_y);
				break;
			case KSORT_KEY_KIND__PSUM_INT64:
				comp = __gpusort_comp_psum_int64(kcxt, sdesc, kds_final, titem_x, titem_y);
				break;
			case KSORT_KEY_KIND__PSUM_FP64:
				comp = __gpusort_comp_psum_fp64(kcxt, sdesc, kds_final, titem_x, titem_y);
				break;
			case KSORT_KEY_KIND__PSUM_NUMERIC:
				comp = __gpusort_comp_psum_numeric(kcxt, sdesc, kds_final, titem_x, titem_y);
				break;
			case KSORT_KEY_KIND__PAVG_INT64:
			case KSORT_KEY_KIND__PAVG_FP64:
			case KSORT_KEY_KIND__PAVG_NUMERIC:
			case KSORT_KEY_KIND__PVARIANCE_SAMP:
			case KSORT_KEY_KIND__PVARIANCE_POP:
			case KSORT_KEY_KIND__PCOVAR_CORR:
			case KSORT_KEY_KIND__PCOVAR_SAMP:
			case KSORT_KEY_KIND__PCOVAR_POP:
			case KSORT_KEY_KIND__PCOVAR_AVGX:
			case KSORT_KEY_KIND__PCOVAR_AVGY:
			case KSORT_KEY_KIND__PCOVAR_COUNT:
			case KSORT_KEY_KIND__PCOVAR_INTERCEPT:
			case KSORT_KEY_KIND__PCOVAR_REGR_R2:
			case KSORT_KEY_KIND__PCOVAR_REGR_SLOPE:
			case KSORT_KEY_KIND__PCOVAR_REGR_SXX:
			case KSORT_KEY_KIND__PCOVAR_REGR_SXY:
			case KSORT_KEY_KIND__PCOVAR_REGR_SYY:
				/* pre-computed float8 values */
				comp = __gpusort_comp_precomp_fp64(sdesc, titem_x, titem_y);
				break;
			default:
				/* Bug? should not happen */
				comp = 0;
				break;
		}
		if (comp != 0)
			return comp;
	}
	return 0;
}

/*
 * kern_gpusort_exec_bitonic
 */
KERNEL_FUNCTION(void)
kern_gpusort_exec_bitonic(kern_session_info *session,
						  kern_gputask *kgtask,
						  kern_data_store *kds_final,
						  uint32_t	nr_threads,
						  uint64_t *row_index,
						  int scale, int step)
{
	const kern_expression *sort_kexp = SESSION_KEXP_GPUSORT_KEYDESC(session);
	const char	   *end = (const char *)kds_final + kds_final->length;
	kern_context   *kcxt;
	uint32_t		thread_id;

	/* sanity checks */
	assert(get_local_size() <= CUDA_MAXTHREADS_PER_BLOCK);
	assert((nr_threads & (nr_threads-1)) == 0);
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
	if (!row_index)
		row_index = KDS_GET_ROWINDEX(kds_final);
	for (thread_id = get_global_id();
		 thread_id < nr_threads;
		 thread_id += get_global_size())
	{
		uint32_t	base = ((thread_id >> scale) << (scale+1));
		uint32_t	m_bits = (thread_id & ((1U<<scale)-1)) >> step;
        uint32_t	l_bits = (thread_id & ((1U<<step)-1));
        uint32_t	index = base + (m_bits << (step+1)) + l_bits;
        uint32_t	buddy = index + (1U << step);
		bool		direction = (thread_id & (1U<<scale));
		int			comp;
		kern_tupitem *titem_x = NULL;
		kern_tupitem *titem_y = NULL;

		if (row_index[index] != 0)
			titem_x = (kern_tupitem *)(end - row_index[index]);
		if (row_index[buddy] != 0)
			titem_y = (kern_tupitem *)(end - row_index[buddy]);
		comp = __gpusort_comp_keys(kcxt, sort_kexp, kds_final, titem_x, titem_y);
		if (direction ? comp < 0 : comp > 0)
		{
			uint64_t	temp = row_index[index];

			row_index[index] = row_index[buddy];
			row_index[buddy] = temp;
		}
	}
}

INLINE_FUNCTION(void)
__gpusort_prep_pavg_int64(kern_context *kcxt,
						  const kern_data_store *kds_final,
						  const kern_tupitem *titem,
						  const kern_sortkey_desc *sdesc)
{
	const void *addr;
	char	   *dest;

	dest = ((char *)titem + titem->t_len + sdesc->buf_offset);
	addr = kern_fetch_minimal_tuple_attr(kds_final, titem, sdesc->src_anum);
	if (!addr)
		*dest++ = false;
	else
	{
		const kagg_state__psum_int_packed *r =
			(const kagg_state__psum_int_packed *)addr;

		if (r->nitems == 0)
			*dest++ = false;
		else
		{
			double	fval = (double)r->sum / (double)r->nitems;

			*dest++ = true;
			memcpy(dest, &fval, sizeof(double));
		}
	}
}

INLINE_FUNCTION(void)
__gpusort_prep_pavg_fp64(kern_context *kcxt,
						 const kern_data_store *kds_final,
						 const kern_tupitem *titem,
						 const kern_sortkey_desc *sdesc)
{
	const void *addr;
	char	   *dest;

	dest = ((char *)titem + titem->t_len + sdesc->buf_offset);
	addr = kern_fetch_minimal_tuple_attr(kds_final, titem, sdesc->src_anum);
	if (!addr)
		*dest++ = false;
	else
	{
		const kagg_state__psum_fp_packed *r =
			(const kagg_state__psum_fp_packed *)addr;

		if (r->nitems == 0)
			*dest++ = false;
		else
		{
			double	fval = r->sum / (double)r->nitems;

			*dest++ = true;
			memcpy(dest, &fval, sizeof(double));
		}
	}
}

INLINE_FUNCTION(void)
__gpusort_prep_pavg_numeric(kern_context *kcxt,
							const kern_data_store *kds_final,
							const kern_tupitem *titem,
							const kern_sortkey_desc *sdesc)
{
	const void *addr;
	char	   *dest;

	dest = ((char *)titem + titem->t_len + sdesc->buf_offset);
	addr = kern_fetch_minimal_tuple_attr(kds_final, titem, sdesc->src_anum);
	if (!addr)
		*dest++ = false;
	else
	{
		const kagg_state__psum_numeric_packed *r =
            (const kagg_state__psum_numeric_packed *)addr;

		if (r->nitems == 0)
			*dest++ = false;
		else
		{
			uint32_t	special = (r->attrs & __PAGG_NUMERIC_ATTRS__MASK);
			int16_t		weight  = (r->attrs & __PAGG_NUMERIC_ATTRS__WEIGHT);
			float8_t	fval;

			*dest++ = true;
			if (special == 0)
			{
				int128_t	x, rem = __fetch_int128_packed(&r->sum);
				int64_t		div = r->nitems;
				float8_t	base = 1.0;
				float8_t	prev;
				bool		negative = false;

				if (rem < 0)
				{
					rem = -rem;
					negative = true;
				}
				/* integer portion */
				x = rem / div;
				fval = (double)x;
				rem -= x * div;

				while (rem != 0)
				{
					base /= 2.0;
					rem *= 2;
					if (rem > div)
					{
						prev = fval;
						fval += base;
						if (fval == prev)
							break;
						rem -= div;
					}
				}
				while (weight < 0)
				{
					fval *= 10.0;
					weight++;
				}
				while (weight > 0)
				{
					fval /= 10.0;
					weight--;
				}
				if (negative)
					fval = -fval;
			}
			else if (special == __PAGG_NUMERIC_ATTRS__PINF)
				fval = INFINITY;
			else if (special == __PAGG_NUMERIC_ATTRS__NINF)
				fval = -INFINITY;
			else
				fval = NAN;
			memcpy(dest, &fval, sizeof(float8_t));
		}
	}
}

INLINE_FUNCTION(void)
__gpusort_prep_pvariance(kern_context *kcxt,
						 const kern_data_store *kds_final,
						 const kern_tupitem *titem,
						 const kern_sortkey_desc *sdesc)
{
	const void *addr;
	char	   *dest;

	dest = ((char *)titem + titem->t_len + sdesc->buf_offset);
	addr = kern_fetch_minimal_tuple_attr(kds_final, titem, sdesc->src_anum);
	if (!addr)
		*dest++ = false;
	else
	{
		const kagg_state__stddev_packed *r =
			(const kagg_state__stddev_packed *)addr;

		if (r->nitems == 0)
			*dest++ = false;
		else
		{
			float8_t	fval = 0.0;
			bool		isnull = false;

			switch (sdesc->kind)
			{
				case KSORT_KEY_KIND__PVARIANCE_SAMP:
					if (r->nitems < 2)
						isnull = true;
					else
					{
						double	N = (double)r->nitems;
						fval = (N * r->sum_x2 - r->sum_x * r->sum_x) / (N * (N - 1.0));
					}
				case KSORT_KEY_KIND__PVARIANCE_POP:
					if (r->nitems < 1)
						isnull = true;
					else
					{
						double	N = (double)r->nitems;
						fval = (N * r->sum_x2 - r->sum_x * r->sum_x) / (N * N);
					}
					break;
				default:
					isnull = true;
					break;
			}
			if (isnull)
				*dest++ = false;
			else
			{
				*dest++ = true;
				memcpy(dest, &fval, sizeof(float8_t));
			}
		}
	}
}

INLINE_FUNCTION(void)
__gpusort_prep_pcovariance(kern_context *kcxt,
						   const kern_data_store *kds_final,
						   const kern_tupitem *titem,
						   const kern_sortkey_desc *sdesc)
{
	const void *addr;
	char	   *dest;

	dest = ((char *)titem + titem->t_len + sdesc->buf_offset);
	addr = kern_fetch_minimal_tuple_attr(kds_final, titem, sdesc->src_anum);
	if (!addr)
		*dest++ = false;
	else
	{
		const kagg_state__covar_packed *r =
			(const kagg_state__covar_packed *)addr;

		if (r->nitems == 0)
			*dest++ = false;
		else
		{
			float8_t	fval = 0.0;
			bool		isnull = false;

			switch (sdesc->kind)
			{
				case KSORT_KEY_KIND__PCOVAR_CORR:
					if (r->nitems < 1 ||
						r->sum_xx == 0.0 ||
						r->sum_yy == 0.0)
						isnull = true;
					else
						fval = r->sum_xy / sqrt(r->sum_xx * r->sum_yy);
					break;
				case KSORT_KEY_KIND__PCOVAR_SAMP:
					if (r->nitems < 2)
						isnull = true;
					else
						fval = r->sum_xy / (double)(r->nitems - 1);
					break;
                case KSORT_KEY_KIND__PCOVAR_POP:
					if (r->nitems < 1)
						isnull = true;
					else
						fval = r->sum_xy / (double)r->nitems;
					break;
                case KSORT_KEY_KIND__PCOVAR_AVGX:
					if (r->nitems < 1)
						isnull = true;
					else
						fval = r->sum_x / (double)r->nitems;
					break;
                case KSORT_KEY_KIND__PCOVAR_AVGY:
					if (r->nitems < 1)
						isnull = true;
					else
						fval = r->sum_x / (double)r->nitems;
					break;
                case KSORT_KEY_KIND__PCOVAR_COUNT:
					fval = (double)r->nitems;
					break;
                case KSORT_KEY_KIND__PCOVAR_INTERCEPT:
					if (r->nitems < 1 || r->sum_xx == 0.0)
						isnull = true;
					else
						fval = (r->sum_y -
								r->sum_x * r->sum_xy / r->sum_xx) / (double)r->nitems;
					break;
                case KSORT_KEY_KIND__PCOVAR_REGR_R2:
					if (r->nitems < 1 || r->sum_xx == 0.0 || r->sum_yy == 0.0)
						isnull = true;
					else
						fval = (r->sum_xy * r->sum_xy) / (r->sum_xx * r->sum_yy);
					break;
                case KSORT_KEY_KIND__PCOVAR_REGR_SLOPE:
					if (r->nitems < 1 || r->sum_xx == 0.0)
						isnull = true;
					else
						fval = (r->sum_xy / r->sum_xx);
					break;
				case KSORT_KEY_KIND__PCOVAR_REGR_SXX:
					if (r->nitems < 1)
						isnull = true;
					else
						fval = r->sum_xx;
					break;
				case KSORT_KEY_KIND__PCOVAR_REGR_SXY:
					if (r->nitems < 1)
						isnull = true;
					else
						fval = r->sum_xy;
					break;
                case KSORT_KEY_KIND__PCOVAR_REGR_SYY:
					if (r->nitems < 1)
						isnull = true;
					else
						fval = r->sum_yy;
					break;
				default:
					isnull = true;
					break;
			}

			if (isnull)
				*dest++ = false;
			else
			{
				*dest++ = true;
				memcpy(dest, &fval, sizeof(float8_t));
			}
		}
	}
}

/*
 * per-tuple preparation on demand
 */
INLINE_FUNCTION(void)
__gpusort_prep_tupitem(kern_context *kcxt,
					   const kern_expression *sort_kexp,
					   const kern_data_store *kds_final,
					   uint32_t kds_index)
{
	const kern_tupitem *titem = KDS_GET_TUPITEM(kds_final, kds_index);

	for (int k=0; k < sort_kexp->u.sort.nkeys; k++)
	{
		const kern_sortkey_desc *sdesc = &sort_kexp->u.sort.desc[k];

		switch (sdesc->kind)
		{
			case KSORT_KEY_KIND__PAVG_INT64:
				__gpusort_prep_pavg_int64(kcxt, kds_final, titem, sdesc);
				break;
			case KSORT_KEY_KIND__PAVG_FP64:
				__gpusort_prep_pavg_fp64(kcxt, kds_final, titem, sdesc);
				break;
			case KSORT_KEY_KIND__PAVG_NUMERIC:
				__gpusort_prep_pavg_numeric(kcxt, kds_final, titem, sdesc);
				break;
			case KSORT_KEY_KIND__PVARIANCE_SAMP:
			case KSORT_KEY_KIND__PVARIANCE_POP:
				__gpusort_prep_pvariance(kcxt, kds_final, titem, sdesc);
				break;
			case KSORT_KEY_KIND__PCOVAR_CORR:
			case KSORT_KEY_KIND__PCOVAR_SAMP:
			case KSORT_KEY_KIND__PCOVAR_POP:
			case KSORT_KEY_KIND__PCOVAR_AVGX:
			case KSORT_KEY_KIND__PCOVAR_AVGY:
			case KSORT_KEY_KIND__PCOVAR_COUNT:
			case KSORT_KEY_KIND__PCOVAR_INTERCEPT:
			case KSORT_KEY_KIND__PCOVAR_REGR_R2:
			case KSORT_KEY_KIND__PCOVAR_REGR_SLOPE:
			case KSORT_KEY_KIND__PCOVAR_REGR_SXX:
			case KSORT_KEY_KIND__PCOVAR_REGR_SXY:
			case KSORT_KEY_KIND__PCOVAR_REGR_SYY:
				__gpusort_prep_pcovariance(kcxt, kds_final, titem, sdesc);
				break;
			default:
				/* nothing to do */
				break;
		}
	}
}

/*
 * kern_gpusort_prep_buffer
 */
KERNEL_FUNCTION(void)
kern_gpusort_prep_buffer(kern_session_info *session,
						 kern_gputask *kgtask,
						 kern_data_store *kds_final,
						 uint32_t nr_threads,
						 uint64_t *row_index)
{
	const kern_expression *sort_kexp = SESSION_KEXP_GPUSORT_KEYDESC(session);
	kern_context   *kcxt;
	uint32_t		nrooms = 2 * nr_threads;
	uint32_t		index;

	/* sanity checks */
	assert(get_local_size() <= CUDA_MAXTHREADS_PER_BLOCK);
	assert(kds_final->nitems >= nr_threads &&
		   kds_final->nitems <= nrooms);
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
	for (index=get_global_id(); index < nrooms; index += get_global_size())
	{
		if (index < kds_final->nitems)
		{
			if (sort_kexp->u.sort.needs_finalization)
				__gpusort_prep_tupitem(kcxt, sort_kexp, kds_final, index);
			if (row_index)
				row_index[index] = KDS_GET_ROWINDEX(kds_final)[index];
		}
		else if (row_index)
			row_index[index] = NULL;
		else
			KDS_GET_ROWINDEX(kds_final)[index] = NULL;
	}
}

/*
 * kern_windowrank_exec_row_number
 */
KERNEL_FUNCTION(void)
kern_windowrank_exec_row_number(kern_session_info *session,
								kern_data_store *kds_final,
								uint32_t *partition_hash_array,
								uint64_t *windowrank_row_index)
{
	const kern_expression *sort_kexp = SESSION_KEXP_GPUSORT_KEYDESC(session);
	uint32_t   *orderby_hash_array = partition_hash_array + kds_final->nitems;
	uint32_t   *results_array = orderby_hash_array + kds_final->nitems;
	uint32_t	index;

	assert(sort_kexp->u.sort.window_rank_func == KSORT_WINDOW_FUNC__ROW_NUMBER);
	for (index = get_global_id();
		 index < kds_final->nitems;
		 index += get_global_size())
	{
		uint32_t	start = 0;
		uint32_t	end = index;
		uint32_t	my_hash;

		my_hash = partition_hash_array[index];
		while (start != end)
		{
			uint32_t	curr = (start + end) / 2;

			if (partition_hash_array[curr] == my_hash)
				end = curr;
			else
				start = curr + 1;
		}
		assert(partition_hash_array[start] == my_hash);
		if (index - start < sort_kexp->u.sort.window_rank_limit - 1)
		{
			results_array[index] = 1;
			windowrank_row_index[index] = KDS_GET_ROWINDEX(kds_final)[index];
		}
		else
		{
			results_array[index] = 0;
			windowrank_row_index[index] = 0UL;
		}
	}
}

/*
 * kern_windowrank_exec_rank
 */
KERNEL_FUNCTION(void)
kern_windowrank_exec_rank(kern_session_info *session,
						  kern_data_store *kds_final,
						  uint32_t *partition_hash_array,
						  uint64_t *windowrank_row_index)
{
	const kern_expression *sort_kexp = SESSION_KEXP_GPUSORT_KEYDESC(session);
	uint32_t   *orderby_hash_array = partition_hash_array + kds_final->nitems;
	uint32_t   *results_array = orderby_hash_array + kds_final->nitems;
	uint32_t	index;

	assert(sort_kexp->u.sort.window_rank_func == KSORT_WINDOW_FUNC__RANK);
	for (index = get_global_id();
		 index < kds_final->nitems;
		 index += get_global_size())
	{
		uint32_t	start = 0;
		uint32_t	end = index;
		uint32_t	my_hash;
		uint32_t	part_leader;

		my_hash = partition_hash_array[index];
		while (start != end)
		{
			uint32_t	curr = (start + end) / 2;

			if (partition_hash_array[curr] == my_hash)
				end = curr;
			else
				start = curr + 1;
		}
		assert(partition_hash_array[start] == my_hash);
		part_leader = start;
		end = index;
		my_hash = orderby_hash_array[index];
		while (start != end)
		{
			uint32_t	curr = (start + end) / 2;

			if (orderby_hash_array[curr] == my_hash)
				end = curr;
			else
				start = curr + 1;
		}
		assert(orderby_hash_array[start] == my_hash);
		assert(part_leader <= start && start <= index);
		if (start - part_leader < sort_kexp->u.sort.window_rank_limit - 1)
		{
			//printf("RANK-FOUND (%u %u %u) delta=%u %u\n", part_leader, start, index, start - part_leader, index - start);
			results_array[index] = 1;
			windowrank_row_index[index] = KDS_GET_ROWINDEX(kds_final)[index];
		}
		else
		{
			results_array[index] = 0;
			windowrank_row_index[index] = 0UL;
		}
	}
}

/*
 * kern_windowrank_exec_dense_rank
 */
KERNEL_FUNCTION(void)
kern_windowrank_exec_dense_rank(kern_session_info *session,
								kern_data_store *kds_final,
								uint32_t *partition_hash_array,
								uint64_t *windowrank_row_index,
								int phase)
{
	const kern_expression *sort_kexp = SESSION_KEXP_GPUSORT_KEYDESC(session);
	uint32_t   *orderby_hash_array = partition_hash_array + kds_final->nitems;
	uint32_t   *results_array = orderby_hash_array + kds_final->nitems;
	uint32_t	index;

	assert(sort_kexp->u.sort.window_rank_func == KSORT_WINDOW_FUNC__DENSE_RANK);
	for (index = get_global_id();
		 index < kds_final->nitems;
		 index += get_global_size())
	{
		if (phase == 0)
		{
			if (index == 0 ||
				partition_hash_array[index] != partition_hash_array[index-1] ||
				orderby_hash_array[index] != orderby_hash_array[index-1])
			{
				results_array[index] = 1;
			}
			else
			{
				results_array[index] = 0;
			}
		}
		else if (phase == 1)
		{
			uint32_t	start = 0;
			uint32_t	end = index;
			uint32_t	my_hash = partition_hash_array[index];;

			while (start != end)
			{
				uint32_t	curr = (start + end) / 2;

				if (partition_hash_array[curr] == my_hash)
					end = curr;
				else
					start = curr + 1;
			}
			assert(start <= index);
			assert(partition_hash_array[start] == my_hash);
			if (results_array[index] -
				results_array[start] < sort_kexp->u.sort.window_rank_limit - 1)
			{
				windowrank_row_index[index] = KDS_GET_ROWINDEX(kds_final)[index];
			}
			else
			{
				windowrank_row_index[index] = 0UL;
			}
		}
		else if (phase == 2)
		{
			results_array[index] = (windowrank_row_index[index] != 0UL);
		}
		else
		{
			break;		/* should not happen */
		}
	}
}

/*
 * internal APIs to load sorting keys
 */
INLINE_FUNCTION(bool)
__gpusort_load_rawkey(kern_context *kcxt,
					  const kern_sortkey_desc *sdesc,
					  const kern_data_store *kds_final,
					  const kern_tupitem *titem,
					  xpu_datum_t *xdatum)
{
	const void *addr = kern_fetch_minimal_tuple_attr(kds_final, titem, sdesc->src_anum);
	if (addr)
	{
		const xpu_datum_operators *key_ops = sdesc->key_ops;

		if (key_ops->xpu_datum_heap_read(kcxt, addr, xdatum))
			return true;
	}
	xdatum->expr_ops = NULL;
	return true;
}

INLINE_FUNCTION(bool)
__gpusort_load_pminmax_int64(kern_context *kcxt,
							 const kern_sortkey_desc *sdesc,
							 const kern_data_store *kds_final,
							 const kern_tupitem *titem,
							 xpu_datum_t *__xdatum)
{
	const kagg_state__pminmax_int64_packed *x = (const kagg_state__pminmax_int64_packed *)
		kern_fetch_minimal_tuple_attr(kds_final, titem, sdesc->src_anum);
	assert(sdesc->key_ops == &xpu_int8_ops);
	if (x && (x->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)
	{
		xpu_int8_t *xdatum = (xpu_int8_t *)__xdatum;
		xdatum->expr_ops = &xpu_int8_ops;
		xdatum->value = x->value;
	}
	else
	{
		__xdatum->expr_ops = NULL;
	}
	return true;

}

INLINE_FUNCTION(bool)
__gpusort_load_pminmax_fp64(kern_context *kcxt,
							const kern_sortkey_desc *sdesc,
							const kern_data_store *kds_final,
							const kern_tupitem *titem,
							xpu_datum_t *__xdatum)
{
	const kagg_state__pminmax_fp64_packed *x = (const kagg_state__pminmax_fp64_packed *)
		kern_fetch_minimal_tuple_attr(kds_final, titem, sdesc->src_anum);
	assert(sdesc->key_ops == &xpu_float8_ops);
	if (x && (x->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)
	{
		xpu_float8_t *xdatum = (xpu_float8_t *)__xdatum;
		xdatum->expr_ops = &xpu_float8_ops;
		xdatum->value = x->value;
	}
	else
	{
		__xdatum->expr_ops = NULL;
	}
	return true;
}

INLINE_FUNCTION(bool)
__gpusort_load_psum_int64(kern_context *kcxt,
						  const kern_sortkey_desc *sdesc,
						  const kern_data_store *kds_final,
						  const kern_tupitem *titem,
						  xpu_datum_t *__xdatum)
{
	const kagg_state__psum_int_packed *x = (const kagg_state__psum_int_packed *)
		kern_fetch_minimal_tuple_attr(kds_final, titem, sdesc->src_anum);
	assert(sdesc->key_ops == &xpu_int8_ops);
	if (x && x->nitems > 0)
	{
		xpu_int8_t *xdatum = (xpu_int8_t *)__xdatum;
		xdatum->expr_ops = &xpu_int8_ops;
		xdatum->value = x->sum;
	}
	else
	{
		__xdatum->expr_ops = NULL;
	}
	return true;
}

INLINE_FUNCTION(bool)
__gpusort_load_psum_fp64(kern_context *kcxt,
						 const kern_sortkey_desc *sdesc,
						 const kern_data_store *kds_final,
						 const kern_tupitem *titem,
						 xpu_datum_t *__xdatum)
{
	const kagg_state__psum_fp_packed *x = (const kagg_state__psum_fp_packed *)
		kern_fetch_minimal_tuple_attr(kds_final, titem, sdesc->src_anum);
	assert(sdesc->key_ops == &xpu_float8_ops);
	if (x && x->nitems > 0)
	{
		xpu_float8_t *xdatum = (xpu_float8_t *)__xdatum;
		xdatum->expr_ops = &xpu_float8_ops;
		xdatum->value = x->sum;
	}
	else
	{
		__xdatum->expr_ops = NULL;
	}
	return true;
}

INLINE_FUNCTION(bool)
__gpusort_load_psum_numeric(kern_context *kcxt,
							const kern_sortkey_desc *sdesc,
							const kern_data_store *kds_final,
							const kern_tupitem *titem,
							xpu_datum_t *__xdatum)
{
	const kagg_state__psum_numeric_packed *x = (const kagg_state__psum_numeric_packed *)
        kern_fetch_minimal_tuple_attr(kds_final, titem, sdesc->src_anum);
	assert(sdesc->key_ops == &xpu_numeric_ops);
	if (x && x->nitems > 0)
	{
		xpu_numeric_t *xdatum = (xpu_numeric_t *)__xdatum;
		int		special = (x->attrs & __PAGG_NUMERIC_ATTRS__MASK);

		if (special == 0)
		{
			xdatum->kind = XPU_NUMERIC_KIND__VALID;
			xdatum->weight = (int16_t)(x->attrs & __PAGG_NUMERIC_ATTRS__WEIGHT);
			xdatum->u.value = __fetch_int128_packed(&x->sum);
		}
		else if (special == __PAGG_NUMERIC_ATTRS__PINF)
			xdatum->kind = XPU_NUMERIC_KIND__POS_INF;
		else if (special == __PAGG_NUMERIC_ATTRS__NINF)
			xdatum->kind = XPU_NUMERIC_KIND__NEG_INF;
		else
			xdatum->kind = XPU_NUMERIC_KIND__NAN;
		xdatum->expr_ops = &xpu_numeric_ops;
	}
	else
	{
		__xdatum->expr_ops = NULL;
	}
	return true;
}

INLINE_FUNCTION(bool)
__gpusort_load_precomp_fp64(kern_context *kcxt,
							const kern_sortkey_desc *sdesc,
							const kern_data_store *kds_final,
							const kern_tupitem *titem,
							xpu_datum_t *__xdatum)
{
	const char *addr = ((char *)titem + titem->t_len + sdesc->buf_offset);
	bool		notnull = *addr++;

	assert(sdesc->key_ops == &xpu_float8_ops);
	if (notnull)
	{
		xpu_float8_t *xdatum = (xpu_float8_t *)__xdatum;
		xdatum->expr_ops = &xpu_float8_ops;
		memcpy(&xdatum->value, addr, sizeof(float8_t));
	}
	else
	{
		__xdatum->expr_ops = NULL;
	}
	return true;
}

/*
 * kern_windowrank_prep_hash
 */
KERNEL_FUNCTION(void)
kern_windowrank_prep_hash(kern_session_info *session,
						  kern_gputask *kgtask,
						  kern_data_store *kds_final,
						  uint32_t *partition_hash_array)
{
	const kern_expression *sort_kexp = SESSION_KEXP_GPUSORT_KEYDESC(session);
	uint32_t	   *orderby_hash_array = partition_hash_array + kds_final->nitems;
	kern_context   *kcxt;
	uint32_t		index, sz = 0;
	xpu_datum_t	   *xdatum;

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
	/* allocation of xdatum */
	for (int j=0; j < sort_kexp->u.sort.nkeys; j++)
	{
		const kern_sortkey_desc *sdesc = &sort_kexp->u.sort.desc[j];

		sz = Max(sdesc->key_ops->xpu_type_sizeof, sz);
	}
	xdatum = (xpu_datum_t *)alloca(sz);

	assert(sort_kexp->u.sort.window_partby_nkeys > 0);
	assert(sort_kexp->u.sort.window_orderby_nkeys > 0);
	for (index = get_global_id();
		 index < kds_final->nitems;
		 index += get_global_size())
	{
		kern_tupitem *titem = KDS_GET_TUPITEM(kds_final, index);
		uint32_t	hash = 0;

		for (int anum=1; anum <= sort_kexp->u.sort.nkeys; anum++)
		{
			const kern_sortkey_desc *sdesc = &sort_kexp->u.sort.desc[anum-1];
			const xpu_datum_operators  *key_ops = sdesc->key_ops;
			uint32_t	__hash;

			switch (sdesc->kind)
			{
				case KSORT_KEY_KIND__VREF:
					if (!__gpusort_load_rawkey(kcxt,
											   sdesc,
											   kds_final,
											   titem,
											   xdatum))
						goto bailout;
					break;
				case KSORT_KEY_KIND__PMINMAX_INT64:
					if (!__gpusort_load_pminmax_int64(kcxt,
													  sdesc,
													  kds_final,
													  titem,
													  xdatum))
						goto bailout;
					break;
				case KSORT_KEY_KIND__PMINMAX_FP64:
					if (!__gpusort_load_pminmax_fp64(kcxt,
													 sdesc,
													 kds_final,
													 titem,
													 xdatum))
						goto bailout;
					break;
				case KSORT_KEY_KIND__PSUM_INT64:
					if (!__gpusort_load_psum_int64(kcxt,
												   sdesc,
												   kds_final,
												   titem,
												   xdatum))
						goto bailout;
					break;
				case KSORT_KEY_KIND__PSUM_FP64:
					if (!__gpusort_load_psum_fp64(kcxt,
												  sdesc,
												  kds_final,
												  titem,
												  xdatum))
						goto bailout;
					break;
				case KSORT_KEY_KIND__PSUM_NUMERIC:
					if (!__gpusort_load_psum_numeric(kcxt,
													 sdesc,
													 kds_final,
													 titem,
													 xdatum))
                        goto bailout;
					break;
				case KSORT_KEY_KIND__PAVG_INT64:
				case KSORT_KEY_KIND__PAVG_FP64:
				case KSORT_KEY_KIND__PAVG_NUMERIC:
				case KSORT_KEY_KIND__PVARIANCE_SAMP:
				case KSORT_KEY_KIND__PVARIANCE_POP:
				case KSORT_KEY_KIND__PCOVAR_CORR:
				case KSORT_KEY_KIND__PCOVAR_SAMP:
				case KSORT_KEY_KIND__PCOVAR_POP:
				case KSORT_KEY_KIND__PCOVAR_AVGX:
				case KSORT_KEY_KIND__PCOVAR_AVGY:
				case KSORT_KEY_KIND__PCOVAR_COUNT:
				case KSORT_KEY_KIND__PCOVAR_INTERCEPT:
				case KSORT_KEY_KIND__PCOVAR_REGR_R2:
				case KSORT_KEY_KIND__PCOVAR_REGR_SLOPE:
				case KSORT_KEY_KIND__PCOVAR_REGR_SXX:
				case KSORT_KEY_KIND__PCOVAR_REGR_SXY:
				case KSORT_KEY_KIND__PCOVAR_REGR_SYY:
					if (!__gpusort_load_precomp_fp64(kcxt,
													 sdesc,
													 kds_final,
													 titem,
													 xdatum))
						goto bailout;
					break;
				default:
					STROM_ELOG(kcxt, "unknown sorting key kind");
					goto bailout;
			}
			if (!key_ops->xpu_datum_hash(kcxt, &__hash, xdatum))
				goto bailout;
			hash = ((hash >> 27) | (hash << 27)) ^ __hash;
			if (anum == sort_kexp->u.sort.window_partby_nkeys)
				partition_hash_array[index] = hash;
			if (anum == (sort_kexp->u.sort.window_partby_nkeys +
						 sort_kexp->u.sort.window_orderby_nkeys))
			{
				orderby_hash_array[index] = hash;
				break;
			}
		}
	}
bailout:
	STROM_WRITEBACK_ERROR_STATUS(&kgtask->kerror, kcxt);
}

/*
 * kern_windowrank_finalize
 */
KERNEL_FUNCTION(void)
kern_windowrank_finalize(kern_data_store *kds_final,
						 uint64_t old_length,
						 uint32_t old_nitems,
						 const uint32_t *results_array,
						 const uint64_t *windowrank_row_index)
{
	uint64_t	__nrooms = GPUSORT_WINDOWRANK_RESULTS_NROOMS(old_nitems);
	uint32_t	new_nitems = (__nrooms > 0 ? results_array[__nrooms-1] : 0);
	uint32_t	base;
	__shared__ uint64_t base_usage;

	for (base = get_global_base();
		 base < old_nitems;
		 base += get_global_size())
	{
		const kern_tupitem *titem = NULL;
		uint32_t	index = base + get_local_id();
		uint32_t	tupsz = 0;
		uint64_t	offset;
		uint64_t	total_sz;

		if (index < old_nitems && windowrank_row_index[index] != 0)
		{
			titem = (const kern_tupitem *)
				((char *)kds_final
				 + old_length
				 - windowrank_row_index[index]);
			tupsz = MAXALIGN(titem->t_len);
		}
		/* allocation of the destination buffer */
		offset = pgstrom_stair_sum_uint64(tupsz, &total_sz);
		if (get_local_id() == 0)
			base_usage = __atomic_add_uint64(&kds_final->usage,  total_sz);
		__syncthreads();
		/* put tuples on the destination */
		offset += base_usage;
		if (tupsz > 0)
		{
			kern_tupitem   *__titem = (kern_tupitem *)
				((char *)kds_final + kds_final->length - offset);
			uint32_t		__index = results_array[index] - 1;

			assert(__index < new_nitems);
			memcpy(__titem, titem, titem->t_len);
			KERN_TUPITEM_SET_ROWID(__titem, __index);
			__threadfence();
			KDS_GET_ROWINDEX(kds_final)[__index] = ((char *)kds_final
													+ kds_final->length
													- (char *)__titem);
		}
		__syncthreads();
	}
	if (get_global_id() == 0)
		kds_final->nitems = new_nitems;
}

/*
 * Simple GPU-Sort + LIMIT clause
 */
KERNEL_FUNCTION(void)
kern_buffer_simple_limit(kern_data_store *kds_final, uint64_t old_length)
{
	uint64_t   *row_index = KDS_GET_ROWINDEX(kds_final);
	uint32_t	base;
	__shared__ uint64_t base_usage;

	assert(kds_final->format == KDS_FORMAT_ROW ||
		   kds_final->format == KDS_FORMAT_HASH);
	for (base = get_global_base();
		 base < kds_final->nitems;
		 base += get_global_size())
	{
		const kern_tupitem *titem = NULL;
		uint32_t	index = base + get_local_id();
		uint32_t	tupsz = 0;
		uint64_t	offset;
		uint64_t	total_sz;

		if (index < kds_final->nitems)
		{
			// XXX - must not use KDS_GET_TUPITEM() because kds_final->length
			//       is already truncated.
			assert(row_index[index] != 0);
			titem = (const kern_tupitem *)
				((char *)kds_final + old_length - row_index[index]);
			tupsz = MAXALIGN(titem->t_len);
		}
		/* allocation of the destination buffer */
		offset = pgstrom_stair_sum_uint64(tupsz, &total_sz);
		if (get_local_id() == 0)
			base_usage = __atomic_add_uint64(&kds_final->usage,  total_sz);
		__syncthreads();
		/* put tuples on the destination */
		offset += base_usage;
		if (tupsz > 0)
		{
			kern_tupitem   *__titem = (kern_tupitem *)
				((char *)kds_final + kds_final->length - offset);
			memcpy(__titem, titem, titem->t_len);
			KERN_TUPITEM_SET_ROWID(__titem, index);
			__threadfence();
			row_index[index] = ((char *)kds_final
								+ kds_final->length
								- (char *)__titem);
		}
		__syncthreads();
	}
}

/*
 * GPU Buffer simple reconstruction (ROW-format with PARTITION)
 */
KERNEL_FUNCTION(void)
kern_gpusort_partition_buffer(kern_session_info *session,
							  kern_gputask *kgtask,
							  kern_data_store *kds_dst,
							  kern_data_store *kds_src)
{
	const kern_expression *sort_kexp = SESSION_KEXP_GPUSORT_KEYDESC(session);
	kern_context   *kcxt;
	uint32_t		base;
	xpu_datum_t	   *xdatum = NULL;
	__shared__ uint32_t base_rowid;
	__shared__ uint64_t base_usage;

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
	__syncthreads();

	assert((kds_src->format == KDS_FORMAT_ROW ||
			kds_src->format == KDS_FORMAT_HASH) &&
		   (kds_dst->format == KDS_FORMAT_ROW));
	/* allocation of xdatum */
	if (sort_kexp && stromTaskProp__partition_divisor > 0)
	{
		int		sz = sizeof(xpu_datum_t);

		for (int j=0; j < sort_kexp->u.sort.nkeys; j++)
		{
			const kern_sortkey_desc *sdesc = &sort_kexp->u.sort.desc[j];

			sz = Max(sdesc->key_ops->xpu_type_sizeof, sz);
		}
		xdatum = (xpu_datum_t *)alloca(sz);
	}

	for (base = get_global_base();
		 base < kds_src->nitems;
		 base += get_global_size())
	{
		kern_tupitem *titem = NULL;
		uint32_t	index = base + get_local_id();
		uint32_t	tupsz = 0;
		uint32_t	row_id;
		uint32_t	count;
		uint64_t	offset;
		uint64_t	total_sz;

		if (index < kds_src->nitems)
		{
			titem = KDS_GET_TUPITEM(kds_src, index);
			if (stromTaskProp__partition_divisor == 0)
			{
				/*
				 * Although not currently used, if the divisor is zero, it is
				 * considered as a simple reconstruction operation without
				 * partitioning.
				 */
				tupsz = MAXALIGN(titem->t_len);
			}
			else if (stromTaskProp__partition_reminder == 0)
			{
				/*
				 * When partitioning the buffer, the GPU kernel is called in order,
				 * starting with the remainder 0 and ending with (divisor-1).
				 * At this time, the sort key used for the window function is hashed
				 * and partitioning is performed by this value, but this only needs
				 * to be calculated once at the beginning; in the case of remainders
				 * 1, 2, ..., the value is also written to the kds_src side so that
				 * it is sufficient to refer to kern_tupitem->hash.
				 */
				uint32_t	hash = 0;

				for (int anum=1; anum <= sort_kexp->u.sort.nkeys; anum++)
				{
					const kern_sortkey_desc *sdesc = &sort_kexp->u.sort.desc[anum-1];
					const xpu_datum_operators  *key_ops = sdesc->key_ops;
					uint32_t		__hash;

					assert(sdesc->kind == KSORT_KEY_KIND__VREF);
					if (sdesc->kind != KSORT_KEY_KIND__VREF)
					{
						STROM_ELOG(kcxt, "unexpected sorting key kind");
						break;
					}
					if (!__gpusort_load_rawkey(kcxt,
											   sdesc,
											   kds_src,
											   titem,
											   xdatum) ||
						!key_ops->xpu_datum_hash(kcxt,
												 &__hash,
												 xdatum))
						break;
					hash = ((hash >> 27) | (hash << 27)) ^ __hash;
					if (anum == sort_kexp->u.sort.window_partby_nkeys)
					{
						titem->hash = hash;
						if ((hash % stromTaskProp__partition_divisor) == stromTaskProp__partition_reminder)
							tupsz = MAXALIGN(titem->t_len);
						break;
					}
				}
			}
			else if ((titem->hash % stromTaskProp__partition_divisor) == stromTaskProp__partition_reminder)
			{
				tupsz = MAXALIGN(titem->t_len);
			}
		}
		/* allocation of the destination buffer */
		row_id = pgstrom_stair_sum_binary(tupsz > 0, &count);
		offset = pgstrom_stair_sum_uint64(tupsz, &total_sz);
		if (get_local_id() == 0)
		{
			base_rowid = __atomic_add_uint32(&kds_dst->nitems, count);
			base_usage = __atomic_add_uint64(&kds_dst->usage,  total_sz);
		}
		__syncthreads();
		/* put tuples on the destination */
		row_id += base_rowid;
		offset += base_usage;
		if (tupsz > 0)
		{
			kern_tupitem   *__titem = (kern_tupitem *)
				((char *)kds_dst + kds_dst->length - offset);
			memcpy(__titem, titem, titem->t_len);
			KERN_TUPITEM_SET_ROWID(__titem, row_id);
			__threadfence();
			KDS_GET_ROWINDEX(kds_dst)[row_id] = offset;
		}
		__syncthreads();
	}
}

/*
 * GPU Buffer simple reconstruction (ROW-format by simple consolidation)
 */
KERNEL_FUNCTION(void)
kern_gpusort_consolidate_buffer(kern_data_store *kds_dst,
								const kern_data_store *__restrict__ kds_src)
{
	__shared__ uint32_t base_rowid;
	__shared__ uint64_t base_usage;
	uint32_t	base;

	assert(kds_dst->format == KDS_FORMAT_ROW &&
		   kds_src->format == KDS_FORMAT_ROW);
	for (base = get_global_base();
		 base < kds_src->nitems;
		 base += get_global_size())
	{
		const kern_tupitem *titem = NULL;
		uint32_t	index = base + get_local_id();
		uint32_t	tupsz = 0;
		uint32_t	row_id;
		uint32_t	count;
		uint64_t	offset;
		uint64_t	total_sz;

		if (index < kds_src->nitems)
		{
			titem = KDS_GET_TUPITEM(kds_src, index);
			tupsz = MAXALIGN(titem->t_len);
		}
		/* allocation of the destination buffer */
		row_id = pgstrom_stair_sum_binary(tupsz > 0, &count);
		offset = pgstrom_stair_sum_uint64(tupsz, &total_sz);
		if (get_local_id() == 0)
		{
			base_rowid = __atomic_add_uint32(&kds_dst->nitems, count);
			base_usage = __atomic_add_uint64(&kds_dst->usage,  total_sz);
		}
		__syncthreads();
		/* put tuples on the destination */
		row_id += base_rowid;
		offset += base_usage;
		if (tupsz > 0)
		{
			kern_tupitem   *__titem = (kern_tupitem *)
				((char *)kds_dst + kds_dst->length - offset);
			memcpy(__titem, titem, titem->t_len);
			KERN_TUPITEM_SET_ROWID(__titem, row_id);
			__threadfence();
			KDS_GET_ROWINDEX(kds_dst)[row_id] = offset;
		}
		__syncthreads();
	}
}
