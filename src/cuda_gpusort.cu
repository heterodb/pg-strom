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


/*
 * kern_gpusort_exec_bitonic
 */
KERNEL_FUNCTION(void)
kern_gpusort_exec_bitonic(kern_session_info *session,
						  kern_gputask *kgtask,
						  kern_data_store *kds_final,
						  int step)
{

}

STATIC_FUNCTION(void)
__gpusort_finalize_pavg_int64(kern_context *kcxt,
							  kern_data_store *kds_final,
							  kern_tupitem *titem,
							  const kern_sortkey_desc *sdesc)
{
	const void *addr;
	char	   *dest;

	dest = ((char *)&titem->htup + titem->t_len + sdesc->buf_offset);
	addr = kern_fetch_heaptuple_attr(kcxt, kds_final, titem, sdesc->src_anum);
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

STATIC_FUNCTION(void)
__gpusort_finalize_pavg_fp64(kern_context *kcxt,
							 kern_data_store *kds_final,
							 kern_tupitem *titem,
							 const kern_sortkey_desc *sdesc)
{
	const void *addr;
	char	   *dest;

	dest = ((char *)&titem->htup + titem->t_len + sdesc->buf_offset);
	addr = kern_fetch_heaptuple_attr(kcxt, kds_final, titem, sdesc->src_anum);
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

STATIC_FUNCTION(void)
__gpusort_finalize_pavg_numeric(kern_context *kcxt,
								kern_data_store *kds_final,
								kern_tupitem *titem,
								const kern_sortkey_desc *sdesc)
{
	const void *addr;
	char	   *dest;

	dest = ((char *)&titem->htup + titem->t_len + sdesc->buf_offset);
	addr = kern_fetch_heaptuple_attr(kcxt, kds_final, titem, sdesc->src_anum);
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
			printf("fval = %f\n", fval);
			memcpy(dest, &fval, sizeof(float8_t));
		}
	}
}


STATIC_FUNCTION(void)
__gpusort_finalize_pvariance(kern_context *kcxt,
							 kern_data_store *kds_final,
							 kern_tupitem *titem,
							 const kern_sortkey_desc *sdesc)
{
	const void *addr;
	char	   *dest;

	dest = ((char *)&titem->htup + titem->t_len + sdesc->buf_offset);
	addr = kern_fetch_heaptuple_attr(kcxt, kds_final, titem, sdesc->src_anum);
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

STATIC_FUNCTION(void)
__gpusort_finalize_pcovariance(kern_context *kcxt,
							   kern_data_store *kds_final,
							   kern_tupitem *titem,
							   const kern_sortkey_desc *sdesc)
{
	const void *addr;
	char	   *dest;

	dest = ((char *)&titem->htup + titem->t_len + sdesc->buf_offset);
	addr = kern_fetch_heaptuple_attr(kcxt, kds_final, titem, sdesc->src_anum);
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
 * kern_gpusort_finalize_buffer
 */
KERNEL_FUNCTION(void)
kern_gpusort_finalize_buffer(kern_session_info *session,
							 kern_gputask *kgtask,
							 kern_data_store *kds_final)
{
	const kern_expression *sort_kexp = SESSION_KEXP_GPUSORT_KEYDESC(session);
	kern_context   *kcxt;
	uint32_t		index;

	/* sanity checks */
	assert(get_local_size() <= CUDA_MAXTHREADS_PER_BLOCK);
	assert(sort_kexp->u.sort.needs_finalization);
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
	for (index=get_global_id();
		 index < kds_final->nitems;
		 index += get_global_size())
	{
		kern_tupitem   *titem = KDS_GET_TUPITEM(kds_final, index);

		for (int k=0; k < sort_kexp->u.sort.nkeys; k++)
		{
			const kern_sortkey_desc *sdesc = &sort_kexp->u.sort.desc[k];

			switch (sdesc->kind)
			{
				case KSORT_KEY_KIND__PAVG_INT64:
					__gpusort_finalize_pavg_int64(kcxt, kds_final, titem, sdesc);
					break;
				case KSORT_KEY_KIND__PAVG_FP64:
					__gpusort_finalize_pavg_fp64(kcxt, kds_final, titem, sdesc);
					break;
				case KSORT_KEY_KIND__PAVG_NUMERIC:
					__gpusort_finalize_pavg_numeric(kcxt, kds_final, titem, sdesc);
					break;
				case KSORT_KEY_KIND__PVARIANCE_SAMP:
				case KSORT_KEY_KIND__PVARIANCE_POP:
					__gpusort_finalize_pvariance(kcxt, kds_final, titem, sdesc);
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
					__gpusort_finalize_pcovariance(kcxt, kds_final, titem, sdesc);
					break;
				default:
					/* nothing to do */
					break;
			}
		}
	}
}
