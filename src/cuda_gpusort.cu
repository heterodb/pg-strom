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

/* ----------------------------------------------------------------
 *
 * kern_gpu_projection_block
 *
 * ----------------------------------------------------------------
 */
INLINE_FUNCTION(int)
__simple_kagg_final__simple_vref(kern_context *kcxt,
								 const kern_colmeta *cmeta,
								 char *dest,
								 const char *source)
{
	int		sz;

	if (cmeta->attlen > 0)
		sz = cmeta->attlen;
	else
		sz = VARSIZE_ANY(source);
	if (dest)
		memcpy(dest, source, sz);
	return sz;
}

#define __SIMPLE_KAGG_FINAL__FMINMAX_INT_TEMPLATE(LABEL,CHECKER_MACRO)	\
	INLINE_FUNCTION(int)												\
	__simple_kagg_final__fminmax_##LABEL(kern_context *kcxt,			\
										 const kern_colmeta *cmeta,		\
										 char *dest,					\
										 const char *source)			\
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
				STROM_ELOG(kcxt, "min/max(" #LABEL ") out of range");	\
				return -1;												\
			}															\
			if (dest)													\
				*((LABEL##_t *)dest) = (LABEL##_t)ival;					\
			return cmeta->attlen;										\
		}																\
		return 0;														\
	}
__SIMPLE_KAGG_FINAL__FMINMAX_INT_TEMPLATE(int8, ival < SCHAR_MIN || ival > SCHAR_MAX)
__SIMPLE_KAGG_FINAL__FMINMAX_INT_TEMPLATE(int16,ival < SHRT_MIN  || ival > SHRT_MAX)
__SIMPLE_KAGG_FINAL__FMINMAX_INT_TEMPLATE(int32,ival < INT_MIN   || ival > INT_MAX)
__SIMPLE_KAGG_FINAL__FMINMAX_INT_TEMPLATE(int64,false)
#undef __SIMPLE_KAGG_FINAL__FMINMAX_INT_TEMPLATE

#define __SIMPLE_KAGG_FINAL__FMINMAX_FP_TEMPLATE(LABEL,TYPE)			\
	INLINE_FUNCTION(int)												\
	__simple_kagg_final__fminmax_##LABEL(kern_context *kcxt,			\
										 const kern_colmeta *cmeta,		\
										 char *dest,					\
										 const char *source)			\
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
__SIMPLE_KAGG_FINAL__FMINMAX_FP_TEMPLATE(fp16, float2_t)
__SIMPLE_KAGG_FINAL__FMINMAX_FP_TEMPLATE(fp32, float4_t)
__SIMPLE_KAGG_FINAL__FMINMAX_FP_TEMPLATE(fp64, float8_t)
#undef __SIMPLE_KAGG_FINAL__FMINMAX_FP_TEMPLATE

INLINE_FUNCTION(int)
__simple_kagg_final__fminmax_numeric(kern_context *kcxt,
									 const kern_colmeta *cmeta,
									 char *dest,
									 const char *source)
{
	kagg_state__pminmax_fp64_packed *state
		= (kagg_state__pminmax_fp64_packed *)source;
	assert(cmeta->attlen < 0 && !cmeta->attbyval);
	if ((state->attrs & __PAGG_MINMAX_ATTRS__VALID) != 0)
	{
		xpu_numeric_t num;

		__xpu_fp64_to_numeric(&num, state->value);
		return num.expr_ops->xpu_datum_write(kcxt, dest, cmeta,
											 (xpu_datum_t *)&num);
	}
	return 0;
}

INLINE_FUNCTION(int)
__simple_kagg_final__fsum_int(kern_context *kcxt,
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
__simple_kagg_final__fsum_int64(kern_context *kcxt,
								const kern_colmeta *cmeta,
								char *dest,
								const char *source)
{
	kagg_state__psum_int_packed *state
        = (kagg_state__psum_int_packed *)source;
	assert(cmeta->attlen < 0 && !cmeta->attbyval);
	if (state->nitems == 0)
		return 0;
	return __xpu_numeric_to_varlena(dest, 0, state->sum);
}

INLINE_FUNCTION(int)
__simple_kagg_final__fsum_fp32(kern_context *kcxt,
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
__simple_kagg_final__fsum_fp64(kern_context *kcxt,
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
__simple_kagg_final__fsum_numeric(kern_context *kcxt,
								  const kern_colmeta *cmeta,
								  char *dest,
								  const char *source)
{
	kagg_state__psum_numeric_packed *state
		= (kagg_state__psum_numeric_packed *)source;
	xpu_numeric_t num;
	uint32_t	special = (state->attrs & __PAGG_NUMERIC_ATTRS__MASK);

	assert(cmeta->attlen < 0);
	if (state->nitems == 0)
		return 0;
	num.expr_ops = &xpu_numeric_ops;
	switch (special)
	{
		case 0:
			num.kind = XPU_NUMERIC_KIND__VALID;
			num.weight = (state->attrs & __PAGG_NUMERIC_ATTRS__WEIGHT);
			num.u.value = __fetch_int128_packed(&state->sum);
			break;
		case __PAGG_NUMERIC_ATTRS__PINF:
			num.kind = XPU_NUMERIC_KIND__POS_INF;
			break;
		case __PAGG_NUMERIC_ATTRS__NINF:
			num.kind = XPU_NUMERIC_KIND__NEG_INF;
			break;
		default:	/* NaN */
			num.kind = XPU_NUMERIC_KIND__NAN;
			break;
	}
	return num.expr_ops->xpu_datum_write(kcxt, dest, cmeta,
										 (xpu_datum_t *)&num);
}

INLINE_FUNCTION(int)
__simple_kagg_final__favg_int(kern_context *kcxt,
							  const kern_colmeta *cmeta,
							  char *dest,
							  const char *source)
{
	kagg_state__psum_int_packed *state
		= (kagg_state__psum_int_packed *)source;
	xpu_numeric_t	n, sum, avg;

	if (state->nitems == 0)
		return 0;
	n.expr_ops = &xpu_numeric_ops;
	n.kind = XPU_NUMERIC_KIND__VALID;
	n.weight = 0;
	n.u.value = state->nitems;

	sum.expr_ops = &xpu_numeric_ops;
	sum.kind = XPU_NUMERIC_KIND__VALID;
	sum.weight = 0;
	sum.u.value = state->sum;

	if (!__xpu_numeric_div(kcxt, &avg, &sum, &n))
		return -1;
	return avg.expr_ops->xpu_datum_write(kcxt, dest, cmeta,
										 (xpu_datum_t *)&avg);
}

INLINE_FUNCTION(int)
__simple_kagg_final__favg_fp(kern_context *kcxt,
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
__simple_kagg_final__favg_numeric(kern_context *kcxt,
								  const kern_colmeta *cmeta,
								  char *dest,
								  const char *source)
{
	kagg_state__psum_numeric_packed *state
		= (kagg_state__psum_numeric_packed *)source;
	uint32_t	special = (state->attrs & __PAGG_NUMERIC_ATTRS__MASK);
	xpu_numeric_t	n, sum, avg;

	if (state->nitems == 0)
		return 0;
	if (special != 0)
	{
		if (special == __PAGG_NUMERIC_ATTRS__PINF)
			avg.kind = XPU_NUMERIC_KIND__POS_INF;
		else if (special == __PAGG_NUMERIC_ATTRS__NINF)
			avg.kind = XPU_NUMERIC_KIND__NEG_INF;
		else
			avg.kind = XPU_NUMERIC_KIND__NAN;
		avg.expr_ops = &xpu_numeric_ops;
	}
	else
	{
		n.expr_ops = &xpu_numeric_ops;
		n.kind = XPU_NUMERIC_KIND__VALID;
		n.weight = 0;
		n.u.value = state->nitems;

		sum.expr_ops = &xpu_numeric_ops;
		sum.kind = XPU_NUMERIC_KIND__VALID;
		sum.weight = (state->attrs & __PAGG_NUMERIC_ATTRS__WEIGHT);
		sum.u.value = __fetch_int128_packed(&state->sum);

		if (!__xpu_numeric_div(kcxt, &avg, &sum, &n))
			return -1;
	}
	return avg.expr_ops->xpu_datum_write(kcxt, dest, cmeta,
										 (xpu_datum_t *)&avg);
}

/*
 * VARIANCE/STDDEV
 */
INLINE_FUNCTION(int)
__simple_kagg_final__fstddev_samp(kern_context *kcxt,
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
		xpu_numeric_t num;

		__xpu_fp64_to_numeric(&num, sqrt(fval / (N * (N - 1.0))));
		return num.expr_ops->xpu_datum_write(kcxt, dest, cmeta,
											 (xpu_datum_t *)&num);
	}
	return 0;
}

INLINE_FUNCTION(int)
__simple_kagg_final__fstddev_sampf(kern_context *kcxt,
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
__simple_kagg_final__fvar_samp(kern_context *kcxt,
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
		xpu_numeric_t num;

		__xpu_fp64_to_numeric(&num, fval / (N * (N - 1.0)));
		return num.expr_ops->xpu_datum_write(kcxt, dest, cmeta,
											 (xpu_datum_t *)&num);
	}
	return 0;
}

INLINE_FUNCTION(int)
__simple_kagg_final__fvar_sampf(kern_context *kcxt,
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
__simple_kagg_final__fstddev_pop(kern_context *kcxt,
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
		xpu_numeric_t num;

		__xpu_fp64_to_numeric(&num, sqrt(fval / (N * N)));
		return num.expr_ops->xpu_datum_write(kcxt, dest, cmeta,
											 (xpu_datum_t *)&num);
	}
	return 0;
}

INLINE_FUNCTION(int)
__simple_kagg_final__fstddev_popf(kern_context *kcxt,
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
__simple_kagg_final__fvar_pop(kern_context *kcxt,
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
		xpu_numeric_t num;

		__xpu_fp64_to_numeric(&num, (fval / (N * N)));
		return num.expr_ops->xpu_datum_write(kcxt, dest, cmeta,
											 (xpu_datum_t *)&num);
	}
	return 0;
}

INLINE_FUNCTION(int)
__simple_kagg_final__fvar_popf(kern_context *kcxt,
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
__simple_kagg_final__fcorr(kern_context *kcxt,
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
__simple_kagg_final__fcovar_samp(kern_context *kcxt,
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
__simple_kagg_final__fcovar_pop(kern_context *kcxt,
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
__simple_kagg_final__fregr_avgx(kern_context *kcxt,
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
__simple_kagg_final__fregr_avgy(kern_context *kcxt,
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
__simple_kagg_final__fregr_count(kern_context *kcxt,
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
__simple_kagg_final__fregr_intercept(kern_context *kcxt,
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
__simple_kagg_final__fregr_r2(kern_context *kcxt,
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
__simple_kagg_final__fregr_slope(kern_context *kcxt,
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
__simple_kagg_final__fregr_sxx(kern_context *kcxt,
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
__simple_kagg_final__fregr_sxy(kern_context *kcxt,
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
__simple_kagg_final__fregr_syy(kern_context *kcxt,
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

STATIC_FUNCTION(int)
simple_form_heap_tuple(kern_context *kcxt,
					   const kern_data_store *kds_src,
					   const kern_tupitem *titem,	/* source tuple */
					   const kern_aggfinal_projection_desc *af_proj,
					   kern_data_store *kds_dst,	/* destination buffer */
					   HeapTupleHeaderData *htup,	/* destination buffer */
					   uint16_t *p_t_infomask)
{
	uint8_t	   *nullmap = NULL;
	uint32_t   *attrs;	/* offset */
	int			ncols = (titem->t_infomask2 & HEAP_NATTS_MASK);
	uint32_t	t_hoff = (titem->t_hoff - MINIMAL_TUPLE_OFFSET);
	uint16_t	t_infomask = 0;
	char	   *base = NULL;
	int			head_sz;

	/* deform the source tuple */
	if ((titem->t_infomask & HEAP_HASNULL) != 0)
		nullmap = (uint8_t *)titem->t_bits;
	assert(t_hoff == MAXALIGN(offsetof(kern_tupitem, t_bits)
							  + (nullmap ? BITMAPLEN(ncols) : 0)));
	ncols = Min(ncols, kds_src->ncols);
	attrs = (uint32_t *)alloca(sizeof(uint32_t) * ncols);
	for (int j=0; j < ncols; j++)
	{
		const kern_colmeta *cmeta = &kds_src->colmeta[j];

		if (nullmap && att_isnull(j, nullmap))
			attrs[j] = 0;
		else
		{
			if (cmeta->attlen > 0)
				t_hoff = TYPEALIGN(cmeta->attalign, t_hoff);
			else if (!VARATT_NOT_PAD_BYTE((const char *)titem + t_hoff))
				t_hoff = TYPEALIGN(cmeta->attalign, t_hoff);
			attrs[j] = t_hoff;
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
		int16_t		resno  = af_proj->desc[j].resno;
		uint32_t	t_next;
		int			sz;
		char	   *dest;
		const char *source;

		if (resno < 1 || resno > ncols || attrs[resno-1] == 0)
		{
			t_infomask |= HEAP_HASNULL;
			continue;	/* NULL */
		}
		source = (const char *)titem + attrs[resno-1];
		t_next = TYPEALIGN(cmeta->attalign, t_hoff);
		dest = (base ? base + t_next : NULL);
		switch (action)
		{
			case KAGG_FINAL__SIMPLE_VREF:
				sz = __simple_kagg_final__simple_vref(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FMINMAX_INT8:
				sz = __simple_kagg_final__fminmax_int8(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FMINMAX_INT16:
				sz = __simple_kagg_final__fminmax_int16(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FMINMAX_INT32:
			case KAGG_FINAL__FMINMAX_DATE:
				sz = __simple_kagg_final__fminmax_int32(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FMINMAX_INT64:
			case KAGG_FINAL__FMINMAX_CASH:			/* 64bit Int */
			case KAGG_FINAL__FMINMAX_TIME:			/* 64bit Int */
			case KAGG_FINAL__FMINMAX_TIMESTAMP:		/* 64bit Int */
			case KAGG_FINAL__FMINMAX_TIMESTAMPTZ:	/* 64bit Int */
				sz = __simple_kagg_final__fminmax_int64(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FMINMAX_FP16:
				sz = __simple_kagg_final__fminmax_fp16(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FMINMAX_FP32:
				sz = __simple_kagg_final__fminmax_fp32(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FMINMAX_FP64:
				sz = __simple_kagg_final__fminmax_fp64(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FMINMAX_NUMERIC:
				sz = __simple_kagg_final__fminmax_numeric(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSUM_INT:
			case KAGG_FINAL__FSUM_CASH:
				sz = __simple_kagg_final__fsum_int(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSUM_INT64:
				sz = __simple_kagg_final__fsum_int64(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSUM_FP32:
				sz = __simple_kagg_final__fsum_fp32(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSUM_FP64:
				sz = __simple_kagg_final__fsum_fp64(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSUM_NUMERIC:
				sz = __simple_kagg_final__fsum_numeric(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FAVG_INT:
			case KAGG_FINAL__FAVG_INT64:
				sz = __simple_kagg_final__favg_int(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FAVG_FP64:
				sz = __simple_kagg_final__favg_fp(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FAVG_NUMERIC:
				sz = __simple_kagg_final__favg_numeric(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSTDDEV_SAMP:
				sz = __simple_kagg_final__fstddev_samp(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSTDDEV_SAMPF:
				sz = __simple_kagg_final__fstddev_sampf(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSTDDEV_POP:
				sz = __simple_kagg_final__fstddev_pop(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FSTDDEV_POPF:
				sz = __simple_kagg_final__fstddev_popf(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FVAR_SAMP:
				sz = __simple_kagg_final__fvar_samp(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FVAR_SAMPF:
				sz = __simple_kagg_final__fvar_sampf(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FVAR_POP:
				sz = __simple_kagg_final__fvar_pop(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FVAR_POPF:
				sz = __simple_kagg_final__fvar_popf(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FCORR:
				sz = __simple_kagg_final__fcorr(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FCOVAR_SAMP:
				sz = __simple_kagg_final__fcovar_samp(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FCOVAR_POP:
				sz = __simple_kagg_final__fcovar_pop(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_AVGX:
				sz = __simple_kagg_final__fregr_avgx(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_AVGY:
				sz = __simple_kagg_final__fregr_avgy(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_COUNT:
				sz = __simple_kagg_final__fregr_count(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_INTERCEPT:
				sz = __simple_kagg_final__fregr_intercept(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_R2:
				sz = __simple_kagg_final__fregr_r2(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_SLOPE:
				sz = __simple_kagg_final__fregr_slope(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_SXX:
				sz = __simple_kagg_final__fregr_sxx(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_SXY:
				sz = __simple_kagg_final__fregr_sxy(kcxt, cmeta, dest, source);
				break;
			case KAGG_FINAL__FREGR_SYY:
				sz = __simple_kagg_final__fregr_syy(kcxt, cmeta, dest, source);
				break;
			default:
				STROM_ELOG(kcxt, "unexpected kagg-final projection");
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
				nullmap[j>>3] |= (1<<(j&7));	/* not null */
			t_hoff = t_next + sz;
		}
	}
	/* ok */
	if (p_t_infomask)
		*p_t_infomask = t_infomask;
	head_sz = offsetof(HeapTupleHeaderData, t_bits);
	if ((t_infomask & HEAP_HASNULL) != 0)
		head_sz += BITMAPLEN(af_proj->nattrs);
	return MAXALIGN(head_sz) + t_hoff;
}

KERNEL_FUNCTION(void)
kern_gpu_projection_block_phase1(kern_session_info *session,
								 kern_gputask *kgtask,
								 kern_data_store *kds_src,
								 kern_data_store *kds_dst)
{
	const kern_aggfinal_projection_desc *af_proj = SESSION_SELECT_INTO_PROJDESC(session);
	kern_select_into_tuple_desc *si_tuples_array = (kern_select_into_tuple_desc *)kgtask->stats;
	kern_context   *kcxt;

	/* sanity checks */
	assert(kds_src->format == KDS_FORMAT_ROW ||
		   kds_src->format == KDS_FORMAT_HASH);
	assert(kds_dst->format == KDS_FORMAT_BLOCK &&
		   kds_dst->block_offset >= KDS_HEAD_LENGTH(kds_dst));
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
	for (uint32_t index = get_global_id(); index < kds_src->nitems; index += get_global_size())
	{
		kern_tupitem *titem = KDS_GET_TUPITEM(kds_src, index);
		kern_select_into_tuple_desc si_tup;
		uint16_t	t_infomask = 0;
		int			tupsz;

		if (!af_proj)
		{
			tupsz = MINIMAL_TUPLE_OFFSET + titem->t_len;
			t_infomask = titem->t_infomask;
		}
		else
		{
			tupsz = simple_form_heap_tuple(kcxt,
										   kds_src,
										   titem,
										   af_proj,
										   kds_dst,
										   NULL,
										   &t_infomask);
		}
		if (tupsz < 0)
			break;
		si_tup.lp_len = tupsz;
		si_tup.lp_off = 0;		/* set by phase-2 */
		si_tup.heap_hasnull = ((t_infomask & HEAP_HASNULL) != 0);
		si_tup.heap_hasvarwidth = ((t_infomask & HEAP_HASVARWIDTH) != 0);
		si_tup.lp_index = 0;	/* set by phase-2 */
		si_tup.block_no = 0;	/* set by phase-2 */
		si_tup.init_hpage = false; /* set by phase-2 */
		si_tuples_array[index] = si_tup;
	}
	STROM_WRITEBACK_ERROR_STATUS(&kgtask->kerror, kcxt);
}

KERNEL_FUNCTION(void)
kern_gpu_projection_block_phase3(kern_session_info *session,
								 kern_gputask *kgtask,
								 kern_data_store *kds_src,
								 kern_data_store *kds_dst,
								 uint32_t start,
								 uint32_t end)
{
	const kern_aggfinal_projection_desc *af_proj = SESSION_SELECT_INTO_PROJDESC(session);
	kern_select_into_tuple_desc *si_tuples_array = (kern_select_into_tuple_desc *)kgtask->stats;
	kern_context   *kcxt;

	/* sanity checks */
	assert(kds_src->format == KDS_FORMAT_ROW ||
		   kds_src->format == KDS_FORMAT_HASH);
	assert(kds_dst->format == KDS_FORMAT_BLOCK &&
		   kds_dst->block_offset >= KDS_HEAD_LENGTH(kds_dst));
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
	for (uint32_t index = start + get_global_id(); index < end; index += get_global_size())
	{
		kern_select_into_tuple_desc si_tup = si_tuples_array[index];
		kern_tupitem   *titem = KDS_GET_TUPITEM(kds_src, index);
		PageHeaderData *hpage;
		ItemIdData		item;
		HeapTupleHeaderData *htup;

		assert(si_tup.block_no < kds_dst->nitems);
		hpage = KDS_BLOCK_PGPAGE(kds_dst, si_tup.block_no);
		if (si_tup.init_hpage)
		{
			__initKdsBlockHeapPage(hpage,
								   si_tup.lp_index+1,
								   si_tup.lp_off);
		}
		/* init heap-tuple */
		htup = (HeapTupleHeaderData *)((char *)hpage + si_tup.lp_off);
		htup->t_choice.t_heap.t_xmin = session->session_curr_xid;
		htup->t_choice.t_heap.t_xmax = InvalidTransactionId;
		htup->t_choice.t_heap.t_field3.t_cid = session->session_curr_cid;
		htup->t_choice.t_datum.datum_typeid = kds_dst->tdtypeid;
		htup->t_ctid.ip_blkid.bi_hi = 0xffff;
		htup->t_ctid.ip_blkid.bi_lo = 0xffff;
		htup->t_ctid.ip_posid = 0;
		if (!af_proj)
		{
			assert(si_tup.lp_len == MINIMAL_TUPLE_OFFSET + titem->t_len);
			memcpy(&htup->t_infomask2,
				   &titem->t_infomask2,
				   titem->t_len - offsetof(kern_tupitem, t_infomask2));
		}
		else
		{
			int			__sz			__attribute__((unused));
			uint16_t	__t_infomask	__attribute__((unused));

			htup->t_infomask = ((si_tup.heap_hasnull ? HEAP_HASNULL : 0) |
								(si_tup.heap_hasvarwidth ? HEAP_HASVARWIDTH : 0));
			htup->t_infomask2 = kds_dst->ncols;
			__sz = simple_form_heap_tuple(kcxt,
										  kds_src,
										  titem,
										  af_proj,
										  kds_dst,
										  htup,
										  &__t_infomask);
			assert(si_tup.lp_len == __sz);
			assert(htup->t_infomask == __t_infomask);
		}
		/* force tuple to all-visible */
		htup->t_infomask &= ~HEAP_XACT_MASK;
		htup->t_infomask |= (HEAP_XMIN_FROZEN | HEAP_XMAX_INVALID);
		/* put item */
		item.lp_off = si_tup.lp_off;
		item.lp_flags = LP_NORMAL;
		item.lp_len = si_tup.lp_len;
		hpage->pd_linp[si_tup.lp_index] = item;
	}
	STROM_WRITEBACK_ERROR_STATUS(&kgtask->kerror, kcxt);
}
