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
 * __writeOutOneTuplePreAgg
 */
STATIC_FUNCTION(int32_t)
__writeOutOneTupleGroupKey(kern_context *kcxt,
						   kern_colmeta *cmeta,
						   kern_aggregate_desc *desc,
						   char *buffer)
{
	kern_variable *kvar;
	int			vclass;
	int32_t		nbytes;

	assert(desc->action == KAGG_ACTION__VREF &&
		   desc->arg0_slot_id >= 0 &&
		   desc->arg0_slot_id < kcxt->kvars_nslots);
	vclass = kcxt->kvars_class[desc->arg0_slot_id];
	kvar = &kcxt->kvars_slot[desc->arg0_slot_id];
	switch (vclass)
	{
		case KVAR_CLASS__NULL:
			return 0;

		case KVAR_CLASS__INLINE:
			assert(cmeta->attlen >= 0 &&
				   cmeta->attlen <= sizeof(kern_variable));
			if (buffer)
				memcpy(buffer, kvar, cmeta->attlen);
			return cmeta->attlen;

		case KVAR_CLASS__VARLENA:
			assert(cmeta->attlen == -1);
			nbytes = VARSIZE_ANY(kvar->ptr);
			if (buffer)
				memcpy(buffer, kvar->ptr, nbytes);
			return nbytes;

		case KVAR_CLASS__XPU_DATUM:
			{
				xpu_datum_t *xdatum = (xpu_datum_t *)kvar->ptr;
				const xpu_datum_operators *expr_ops = xdatum->expr_ops;

				if (XPU_DATUM_ISNULL(xdatum))
					return 0;
				assert(xdatum->expr_ops == expr_ops);
				return expr_ops->xpu_datum_write(kcxt, buffer, cmeta, xdatum);
			}

		default:
			if (vclass < 0)
				return -1;
			if (cmeta->attlen >= 0)
			{
				if (buffer)
				{
					nbytes = Min(vclass, cmeta->attlen);
					memcpy(buffer, kvar->ptr, nbytes);
					if (nbytes < cmeta->attlen)
						memset(buffer + nbytes, 0, cmeta->attlen - nbytes);
				}
				return cmeta->attlen;
            }
            else if (cmeta->attlen == -1)
            {
				nbytes = VARHDRSZ + vclass;
				if (buffer)
				{
					memcpy(buffer+VARHDRSZ, kvar->ptr, vclass);
					SET_VARSIZE(buffer, nbytes);
				}
				return nbytes;
			}
	}
	return -1;
}

STATIC_FUNCTION(int32_t)
__writeOutOneTuplePreAgg(kern_context *kcxt,
						 kern_data_store *kds_final,
						 HeapTupleHeaderData *htup,
						 kern_expression *kexp_actions)
{
	int			nattrs = Min(kds_final->ncols, kexp_actions->u.pagg.nattrs);
	uint32_t	t_hoff, t_next;
	uint16_t	t_infomask = HEAP_HASNULL;
	char	   *buffer = NULL;

	t_hoff = MAXALIGN(offsetof(HeapTupleHeaderData,
							   t_bits) + BITMAPLEN(nattrs));
	if (htup)
	{
		memset(htup, 0, t_hoff);
		htup->t_choice.t_datum.datum_typmod = kds_final->tdtypmod;
		htup->t_choice.t_datum.datum_typeid = kds_final->tdtypeid;
		htup->t_ctid.ip_blkid.bi_hi = 0xffff;	/* InvalidBlockNumber */
		htup->t_ctid.ip_blkid.bi_lo = 0xffff;
		htup->t_ctid.ip_posid = 0;				/* InvalidOffsetNumber */
		htup->t_infomask2 = (nattrs & HEAP_NATTS_MASK);
		htup->t_hoff = t_hoff;
	}
	/* walk on the columns */
	for (int j=0; j < nattrs; j++)
	{
		kern_aggregate_desc *desc = &kexp_actions->u.pagg.desc[j];
		kern_colmeta   *cmeta = &kds_final->colmeta[j];
		int				nbytes;

		assert((char *)cmeta > (char *)kds_final &&
			   (char *)cmeta < (char *)kds_final + kds_final->length);
		assert(cmeta->attalign > 0 && cmeta->attalign <= 8);
		t_next = TYPEALIGN(cmeta->attalign, t_hoff);
		if (htup)
		{
			if (t_next > t_hoff)
				memset((char *)htup + t_hoff, 0, t_next - t_hoff);
			buffer = (char *)htup + t_next;
		}

		switch (desc->action)
		{
			case KAGG_ACTION__VREF:
				nbytes = __writeOutOneTupleGroupKey(kcxt, cmeta, desc, buffer);
				if (nbytes < 0)
					return -1;
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
					r->nitems = 0;
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
					r->nitems = 0;
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
					r->nitems = 0;
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
					r->nitems = 0;
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
		if (htup && nbytes > 0)
			htup->t_bits[j>>3] |= (1<<(j&7));
		t_hoff = t_next + nbytes;
	}

	if (htup)
		htup->t_infomask = t_infomask;
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
							 bool kvars_is_valid)
{
	uint32_t	mask;

	mask = __ballot_sync(__activemask(), kvars_is_valid);
	if (LaneId() == 0)
		__atomic_add_uint64((uint64_t *)buffer, __popc(mask));
}

/*
 * __update_nogroups__nrows_cond
 */
INLINE_FUNCTION(void)
__update_nogroups__nrows_cond(kern_context *kcxt,
							  char *buffer,
							  kern_colmeta *cmeta,
							  kern_aggregate_desc *desc,
							  bool kvars_is_valid)
{
	uint32_t	mask;

	if (kvars_is_valid)
	{
		if (kcxt->kvars_class[desc->arg0_slot_id] == KVAR_CLASS__NULL)
			kvars_is_valid = false;
	}
	mask = __ballot_sync(__activemask(), kvars_is_valid);
	if (LaneId() == 0)
		__atomic_add_uint64((uint64_t *)buffer, __popc(mask));
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
							  bool kvars_is_valid)
{
	int32_t		ival = INT_MAX;
	int32_t		temp;
	uint32_t	mask;

	if (kvars_is_valid)
	{
		int		slot_id = desc->arg0_slot_id;
		int		vclass = kcxt->kvars_class[slot_id];

		if (vclass == KVAR_CLASS__INLINE)
			ival = kcxt->kvars_slot[slot_id].i32;
		else
		{
			assert(vclass == KVAR_CLASS__NULL);
			kvars_is_valid = false;
		}
	}
	mask = __ballot_sync(__activemask(), kvars_is_valid);
	if (mask != 0)
	{
		kagg_state__pminmax_int64_packed *r =
			(kagg_state__pminmax_int64_packed *)buffer;

		temp = __shfl_xor_sync(__activemask(), ival, 0x0001);
		ival = Min(ival, temp);
		temp = __shfl_xor_sync(__activemask(), ival, 0x0002);
		ival = Min(ival, temp);
		temp = __shfl_xor_sync(__activemask(), ival, 0x0004);
		ival = Min(ival, temp);
		temp = __shfl_xor_sync(__activemask(), ival, 0x0008);
		ival = Min(ival, temp);
		temp = __shfl_xor_sync(__activemask(), ival, 0x0010);
		ival = Min(ival, temp);
		if (LaneId() == 0)
		{
			__atomic_add_uint32(&r->nitems, __popc(mask));
			__atomic_min_int64(&r->value, (int64_t)ival);
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
							  bool kvars_is_valid)
{
	int64_t		ival = LONG_MAX;
	int64_t		temp;
	uint32_t	mask;

	if (kvars_is_valid)
	{
		int		slot_id = desc->arg0_slot_id;
		int		vclass = kcxt->kvars_class[slot_id];

		if (vclass == KVAR_CLASS__INLINE)
			ival = kcxt->kvars_slot[slot_id].i64;
		else
		{
			assert(vclass == KVAR_CLASS__NULL);
			kvars_is_valid = false;
		}
	}
	mask = __ballot_sync(__activemask(), kvars_is_valid);
	if (mask != 0)
	{
		kagg_state__pminmax_int64_packed *r =
			(kagg_state__pminmax_int64_packed *)buffer;

		temp = __shfl_xor_sync(__activemask(), ival, 0x0001);
		ival = Min(ival, temp);
		temp = __shfl_xor_sync(__activemask(), ival, 0x0002);
		ival = Min(ival, temp);
		temp = __shfl_xor_sync(__activemask(), ival, 0x0004);
		ival = Min(ival, temp);
		temp = __shfl_xor_sync(__activemask(), ival, 0x0008);
		ival = Min(ival, temp);
		temp = __shfl_xor_sync(__activemask(), ival, 0x0010);
		ival = Min(ival, temp);
		if (LaneId() == 0)
		{
			__atomic_add_uint32(&r->nitems, __popc(mask));
			__atomic_min_int64(&r->value, ival);
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
							  bool kvars_is_valid)
{
	int32_t		ival = INT_MIN;
	int32_t		temp;
	uint32_t	mask;

	if (kvars_is_valid)
	{
		int		slot_id = desc->arg0_slot_id;
		int		vclass = kcxt->kvars_class[slot_id];

		if (vclass == KVAR_CLASS__INLINE)
		{
			ival = kcxt->kvars_slot[slot_id].i32;
		}
		else
		{
			assert(vclass == KVAR_CLASS__NULL);
			kvars_is_valid = false;
		}
	}
	mask = __ballot_sync(__activemask(), kvars_is_valid);
	if (mask != 0)
	{
		kagg_state__pminmax_int64_packed *r =
			(kagg_state__pminmax_int64_packed *)buffer;

		temp = __shfl_xor_sync(__activemask(), ival, 0x0001);
		ival = Max(ival, temp);
		temp = __shfl_xor_sync(__activemask(), ival, 0x0002);
		ival = Max(ival, temp);
		temp = __shfl_xor_sync(__activemask(), ival, 0x0004);
		ival = Max(ival, temp);
		temp = __shfl_xor_sync(__activemask(), ival, 0x0008);
		ival = Max(ival, temp);
		temp = __shfl_xor_sync(__activemask(), ival, 0x0010);
		ival = Max(ival, temp);

		if (LaneId() == 0)
		{
			__atomic_add_uint32(&r->nitems, __popc(mask));
			__atomic_max_int64(&r->value,  (int64_t)ival);
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
							  bool kvars_is_valid)
{
	int64_t		ival = LONG_MIN;
	int64_t		temp;
	uint32_t	mask;

	if (kvars_is_valid)
	{
		int		slot_id = desc->arg0_slot_id;
		int		vclass = kcxt->kvars_class[slot_id];

		if (vclass == KVAR_CLASS__INLINE)
		{
			ival = kcxt->kvars_slot[slot_id].i64;
		}
		else
		{
			assert(vclass == KVAR_CLASS__NULL);
			kvars_is_valid = false;
		}
	}
	mask = __ballot_sync(__activemask(), kvars_is_valid);
	if (mask != 0)
	{
		kagg_state__pminmax_int64_packed *r =
			(kagg_state__pminmax_int64_packed *)buffer;

		temp = __shfl_xor_sync(__activemask(), ival, 0x0001);
		ival = Max(ival, temp);
		temp = __shfl_xor_sync(__activemask(), ival, 0x0002);
		ival = Max(ival, temp);
		temp = __shfl_xor_sync(__activemask(), ival, 0x0004);
		ival = Max(ival, temp);
		temp = __shfl_xor_sync(__activemask(), ival, 0x0008);
		ival = Max(ival, temp);
		temp = __shfl_xor_sync(__activemask(), ival, 0x0010);
		ival = Max(ival, temp);

		if (LaneId() == 0)
		{
			__atomic_add_uint32(&r->nitems, __popc(mask));
			__atomic_max_int64(&r->value,  ival);
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
							 bool kvars_is_valid)
{
	float8_t	fval = DBL_MAX;
	float8_t	temp;
	uint32_t	mask;

	if (kvars_is_valid)
	{
		int		slot_id = desc->arg0_slot_id;
		int		vclass = kcxt->kvars_class[slot_id];

		if (vclass == KVAR_CLASS__INLINE)
			fval = kcxt->kvars_slot[slot_id].fp64;
		else
		{
			assert(vclass == KVAR_CLASS__NULL);
			kvars_is_valid = false;
		}
	}
	mask = __ballot_sync(__activemask(), kvars_is_valid);
	if (mask != 0)
	{
		kagg_state__pminmax_fp64_packed *r =
			(kagg_state__pminmax_fp64_packed *)buffer;

		temp = __shfl_xor_sync(__activemask(), fval, 0x0001);
		fval = Min(fval, temp);
		temp = __shfl_xor_sync(__activemask(), fval, 0x0002);
		fval = Min(fval, temp);
		temp = __shfl_xor_sync(__activemask(), fval, 0x0004);
		fval = Min(fval, temp);
		temp = __shfl_xor_sync(__activemask(), fval, 0x0008);
		fval = Min(fval, temp);
		temp = __shfl_xor_sync(__activemask(), fval, 0x0010);
		fval = Min(fval, temp);

		if (LaneId() == 0)
		{
			__atomic_add_uint32(&r->nitems, __popc(mask));
			__atomic_min_fp64(&r->value, fval);
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
							 bool kvars_is_valid)
{
	float8_t	fval = -DBL_MAX;
	float8_t	temp;
	uint32_t	mask;

	if (kvars_is_valid)
	{
		int		slot_id = desc->arg0_slot_id;
		int		vclass = kcxt->kvars_class[slot_id];

		if (vclass == KVAR_CLASS__INLINE)
			fval = kcxt->kvars_slot[slot_id].fp64;
		else
		{
			assert(vclass == KVAR_CLASS__NULL);
			kvars_is_valid = false;
		}
	}
	mask = __ballot_sync(__activemask(), kvars_is_valid);
	if (mask != 0)
	{
		kagg_state__pminmax_fp64_packed *r =
			(kagg_state__pminmax_fp64_packed *)buffer;

		temp = __shfl_xor_sync(__activemask(), fval, 0x0001);
		fval = Max(fval, temp);
		temp = __shfl_xor_sync(__activemask(), fval, 0x0002);
		fval = Max(fval, temp);
		temp = __shfl_xor_sync(__activemask(), fval, 0x0004);
		fval = Max(fval, temp);
		temp = __shfl_xor_sync(__activemask(), fval, 0x0008);
		fval = Max(fval, temp);
		temp = __shfl_xor_sync(__activemask(), fval, 0x0010);
		fval = Max(fval, temp);

		if (LaneId() == 0)
		{
			__atomic_add_uint32(&r->nitems, __popc(mask));
			__atomic_max_fp64(&r->value, fval);
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
							bool kvars_is_valid)
{
	int64_t		ival = 0;
	uint32_t	mask;

	if (kvars_is_valid)
	{
		int		slot_id = desc->arg0_slot_id;
		int		vclass = kcxt->kvars_class[slot_id];

		if (vclass == KVAR_CLASS__INLINE)
			ival = kcxt->kvars_slot[slot_id].i64;
		else
		{
			assert(vclass == KVAR_CLASS__NULL);
			kvars_is_valid = false;
		}
	}
	mask = __ballot_sync(__activemask(), kvars_is_valid);
	if (mask != 0)
	{
		ival += __shfl_xor_sync(__activemask(), ival, 0x0001);
		ival += __shfl_xor_sync(__activemask(), ival, 0x0002);
		ival += __shfl_xor_sync(__activemask(), ival, 0x0004);
		ival += __shfl_xor_sync(__activemask(), ival, 0x0008);
		ival += __shfl_xor_sync(__activemask(), ival, 0x0010);
		if (LaneId() == 0)
		{
			kagg_state__psum_int_packed *r =
				(kagg_state__psum_int_packed *)buffer;
			__atomic_add_uint32(&r->nitems, __popc(mask));
			__atomic_add_int64(&r->sum, ival);
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
						   bool kvars_is_valid)
{
	float8_t	fval = 0;
	uint32_t	mask;

	if (kvars_is_valid)
	{
		int		slot_id = desc->arg0_slot_id;
		int		vclass = kcxt->kvars_class[slot_id];

		if (vclass == KVAR_CLASS__INLINE)
			fval = kcxt->kvars_slot[slot_id].fp64;
		else
		{
			assert(vclass == KVAR_CLASS__NULL);
			kvars_is_valid = false;
		}
	}
	mask = __ballot_sync(__activemask(), kvars_is_valid);
	if (mask != 0)
	{
		fval += __shfl_xor_sync(__activemask(), fval, 0x0001);
		fval += __shfl_xor_sync(__activemask(), fval, 0x0002);
		fval += __shfl_xor_sync(__activemask(), fval, 0x0004);
		fval += __shfl_xor_sync(__activemask(), fval, 0x0008);
		fval += __shfl_xor_sync(__activemask(), fval, 0x0010);
		if (LaneId() == 0)
		{
			kagg_state__psum_fp_packed *r =
				(kagg_state__psum_fp_packed *)buffer;
			__atomic_add_uint32(&r->nitems, __popc(mask));
			__atomic_add_fp64(&r->sum, fval);
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
						   bool kvars_is_valid)
{
	float8_t	sum_x = 0.0;
	uint32_t	mask;

	if (kvars_is_valid)
	{
		int		slot_id = desc->arg0_slot_id;
		int		vclass = kcxt->kvars_class[slot_id];

		if (vclass == KVAR_CLASS__INLINE)
			sum_x = kcxt->kvars_slot[slot_id].fp64;
		else
		{
			assert(vclass == KVAR_CLASS__NULL);
			kvars_is_valid = false;
		}
	}
	mask = __ballot_sync(__activemask(), kvars_is_valid);
	if (mask != 0)
	{
		float8_t	sum_x2 = sum_x * sum_x;

		/* sum_x */
		sum_x += __shfl_xor_sync(__activemask(), sum_x, 0x0001);
		sum_x += __shfl_xor_sync(__activemask(), sum_x, 0x0002);
		sum_x += __shfl_xor_sync(__activemask(), sum_x, 0x0004);
		sum_x += __shfl_xor_sync(__activemask(), sum_x, 0x0008);
		sum_x += __shfl_xor_sync(__activemask(), sum_x, 0x0010);
		/* sum_x2 */
		sum_x2 += __shfl_xor_sync(__activemask(), sum_x2, 0x0001);
		sum_x2 += __shfl_xor_sync(__activemask(), sum_x2, 0x0002);
		sum_x2 += __shfl_xor_sync(__activemask(), sum_x2, 0x0004);
		sum_x2 += __shfl_xor_sync(__activemask(), sum_x2, 0x0008);
		sum_x2 += __shfl_xor_sync(__activemask(), sum_x2, 0x0010);

		if (LaneId() == 0)
		{
			kagg_state__stddev_packed *r =
				(kagg_state__stddev_packed *)buffer;
			__atomic_add_uint32(&r->nitems, __popc(mask));
			__atomic_add_fp64(&r->sum_x,  sum_x);
			__atomic_add_fp64(&r->sum_x2, sum_x2);
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
						  bool kvars_is_valid)
{
	float8_t	sum_x = 0.0;
	float8_t	sum_y = 0.0;
	uint32_t	mask;

	if (kvars_is_valid)
	{
		if (kcxt->kvars_class[desc->arg0_slot_id] == KVAR_CLASS__INLINE &&
			kcxt->kvars_class[desc->arg1_slot_id] == KVAR_CLASS__INLINE)
		{
			sum_x = kcxt->kvars_slot[desc->arg0_slot_id].fp64;
			sum_y = kcxt->kvars_slot[desc->arg0_slot_id].fp64;
		}
		else
		{
			assert(kcxt->kvars_class[desc->arg0_slot_id] == KVAR_CLASS__NULL ||
				   kcxt->kvars_class[desc->arg1_slot_id] == KVAR_CLASS__NULL);
			kvars_is_valid = false;
		}
	}
	mask = __ballot_sync(__activemask(), kvars_is_valid);
	if (mask != 0)
	{
		float8_t	sum_xx = sum_x * sum_x;
		float8_t	sum_xy = sum_x * sum_y;
		float8_t	sum_yy = sum_y * sum_y;

		/* sum_x */
		sum_x += __shfl_xor_sync(__activemask(), sum_x, 0x0001);
		sum_x += __shfl_xor_sync(__activemask(), sum_x, 0x0002);
		sum_x += __shfl_xor_sync(__activemask(), sum_x, 0x0004);
		sum_x += __shfl_xor_sync(__activemask(), sum_x, 0x0008);
		sum_x += __shfl_xor_sync(__activemask(), sum_x, 0x0010);

		/* sum_y */
		sum_y += __shfl_xor_sync(__activemask(), sum_y, 0x0001);
		sum_y += __shfl_xor_sync(__activemask(), sum_y, 0x0002);
		sum_y += __shfl_xor_sync(__activemask(), sum_y, 0x0004);
		sum_y += __shfl_xor_sync(__activemask(), sum_y, 0x0008);
		sum_y += __shfl_xor_sync(__activemask(), sum_y, 0x0010);

		/* sum_xx */
		sum_xx += __shfl_xor_sync(__activemask(), sum_xx, 0x0001);
		sum_xx += __shfl_xor_sync(__activemask(), sum_xx, 0x0002);
		sum_xx += __shfl_xor_sync(__activemask(), sum_xx, 0x0004);
		sum_xx += __shfl_xor_sync(__activemask(), sum_xx, 0x0008);
		sum_xx += __shfl_xor_sync(__activemask(), sum_xx, 0x0010);

		/* sum_xy */
		sum_xy += __shfl_xor_sync(__activemask(), sum_xy, 0x0001);
		sum_xy += __shfl_xor_sync(__activemask(), sum_xy, 0x0002);
		sum_xy += __shfl_xor_sync(__activemask(), sum_xy, 0x0004);
		sum_xy += __shfl_xor_sync(__activemask(), sum_xy, 0x0008);
		sum_xy += __shfl_xor_sync(__activemask(), sum_xy, 0x0010);

		/* sum_yy */
		sum_yy += __shfl_xor_sync(__activemask(), sum_yy, 0x0001);
		sum_yy += __shfl_xor_sync(__activemask(), sum_yy, 0x0002);
		sum_yy += __shfl_xor_sync(__activemask(), sum_yy, 0x0004);
		sum_yy += __shfl_xor_sync(__activemask(), sum_yy, 0x0008);
		sum_yy += __shfl_xor_sync(__activemask(), sum_yy, 0x0010);

		if (LaneId() == 0)
		{
			kagg_state__covar_packed *r =
				(kagg_state__covar_packed *)buffer;
			__atomic_add_uint32(&r->nitems, __popc(mask));
			__atomic_add_fp64(&r->sum_x,  sum_x);
			__atomic_add_fp64(&r->sum_xx, sum_xx);
			__atomic_add_fp64(&r->sum_y,  sum_y);
			__atomic_add_fp64(&r->sum_yy, sum_yy);
			__atomic_add_fp64(&r->sum_xy, sum_xy);
		}
	}
}

/*
 * __updateOneTupleNoGroups
 */
STATIC_FUNCTION(void)
__updateOneTupleNoGroups(kern_context *kcxt,
						 kern_data_store *kds_final,
						 bool kvars_is_valid,
						 HeapTupleHeaderData *htup,
						 kern_expression *kexp_groupby_actions)
{
	int			nattrs = (htup->t_infomask2 & HEAP_NATTS_MASK);
	bool		heap_hasnull = ((htup->t_infomask & HEAP_HASNULL) != 0);
	uint32_t	t_hoff;
	char	   *buffer = NULL;

	t_hoff = offsetof(HeapTupleHeaderData, t_bits);
	if (heap_hasnull)
		t_hoff += BITMAPLEN(nattrs);
	t_hoff = MAXALIGN(t_hoff);

	for (int j=0; j < nattrs; j++)
	{
		kern_aggregate_desc *desc = &kexp_groupby_actions->u.pagg.desc[j];
		kern_colmeta   *cmeta = &kds_final->colmeta[j];

		if (heap_hasnull && att_isnull(j, htup->t_bits))
		{
			/* only grouping-key may have NULL */
			assert(desc->action == KAGG_ACTION__VREF);
			continue;
		}

		if (cmeta->attlen > 0)
			t_hoff = TYPEALIGN(cmeta->attalign, t_hoff);
		else if (!VARATT_NOT_PAD_BYTE((char *)htup + t_hoff))
			t_hoff = TYPEALIGN(cmeta->attalign, t_hoff);
		buffer = ((char *)htup + t_hoff);
		if (cmeta->attlen > 0)
			t_hoff += cmeta->attlen;
		else
			t_hoff += VARSIZE_ANY(buffer);

		switch (desc->action)
		{
			case KAGG_ACTION__NROWS_ANY:
				__update_nogroups__nrows_any(kcxt, buffer,
											 cmeta, desc,
											 kvars_is_valid);
				break;
			case KAGG_ACTION__NROWS_COND:
				__update_nogroups__nrows_cond(kcxt, buffer,
											  cmeta, desc,
											  kvars_is_valid);
				break;
			case KAGG_ACTION__PMIN_INT32:
				__update_nogroups__pmin_int32(kcxt, buffer,
											  cmeta, desc,
											  kvars_is_valid);
				break;
			case KAGG_ACTION__PMIN_INT64:
				__update_nogroups__pmin_int64(kcxt, buffer,
											  cmeta, desc,
											  kvars_is_valid);
				break;
			case KAGG_ACTION__PMAX_INT32:
				__update_nogroups__pmax_int32(kcxt, buffer,
											  cmeta, desc,
											  kvars_is_valid);
				break;
			case KAGG_ACTION__PMAX_INT64:
				__update_nogroups__pmax_int64(kcxt, buffer,
											  cmeta, desc,
											  kvars_is_valid);
				break;
			case KAGG_ACTION__PMIN_FP64:
				__update_nogroups__pmin_fp64(kcxt, buffer,
											 cmeta, desc,
											 kvars_is_valid);
				break;
			case KAGG_ACTION__PMAX_FP64:
				__update_nogroups__pmax_fp64(kcxt, buffer,
											 cmeta, desc,
											 kvars_is_valid);
				break;
			case KAGG_ACTION__PSUM_INT:
			case KAGG_ACTION__PAVG_INT:
				__update_nogroups__psum_int(kcxt, buffer,
											cmeta, desc,
											kvars_is_valid);
				break;
			case KAGG_ACTION__PAVG_FP:
			case KAGG_ACTION__PSUM_FP:
				__update_nogroups__psum_fp(kcxt, buffer,
										   cmeta, desc,
										   kvars_is_valid);
				break;
			case KAGG_ACTION__STDDEV:
				__update_nogroups__pstddev(kcxt, buffer,
										   cmeta, desc,
										   kvars_is_valid);
				break;
			case KAGG_ACTION__COVAR:
				__update_nogroups__pcovar(kcxt, buffer,
										  cmeta, desc,
										  kvars_is_valid);
				break;
			default:
				/*
				 * No more partial aggregation exists after grouping-keys
				 */
				return;
		}
	}
}

/*
 * __insertOneTupleNoGroups
 */
STATIC_FUNCTION(kern_tupitem *)
__insertOneTupleNoGroups(kern_context *kcxt,
						 kern_data_store *kds_final,
						 kern_expression *kexp_groupby_actions)
{
	kern_tupitem   *tupitem;
	int32_t			tupsz;
	uint32_t		required;
	uint32_t		usage;
	size_t			total_sz;

	assert(kds_final->format == KDS_FORMAT_ROW &&
		   kds_final->hash_nslots == 0);
	/* estimate length */
	tupsz = __writeOutOneTuplePreAgg(kcxt, kds_final, NULL,
									 kexp_groupby_actions);
	assert(tupsz > 0);
	required = MAXALIGN(offsetof(kern_tupitem, htup) + tupsz);
	assert(required < 1000);
	total_sz = (KDS_HEAD_LENGTH(kds_final) +
				MAXALIGN(sizeof(uint32_t)) +
				required + __kds_unpack(kds_final->usage));
	if (total_sz > kds_final->length)
		return NULL;	/* out of memory */
	usage = __atomic_add_uint32(&kds_final->usage, __kds_packed(required));
	tupitem = (kern_tupitem *)((char *)kds_final
							   + kds_final->length
							   - __kds_unpack(usage)
							   - required);
	
	__writeOutOneTuplePreAgg(kcxt, kds_final,
							 &tupitem->htup,
							 kexp_groupby_actions);
	tupitem->t_len = tupsz;
	tupitem->rowid = 0;
	__atomic_write_uint32(KDS_GET_ROWINDEX(kds_final),
						  __kds_packed((char *)kds_final
									   + kds_final->length
									   - (char *)tupitem));
	return tupitem;
}

STATIC_FUNCTION(bool)
__execGpuPreAggNoGroups(kern_context *kcxt,
						kern_data_store *kds_final,
						bool kvars_is_valid,
						kern_expression *kexp_groupby_actions,
						bool *p_try_suspend)
{
	kern_tupitem *tupitem;
	bool		try_suspend = false;

	assert(kds_final->format == KDS_FORMAT_ROW);
	assert(kexp_groupby_actions->opcode == FuncOpCode__AggFuncs);
	for (;;)
	{
		if (LaneId() == 0)
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
						try_suspend = true;
						/* UNLOCK */
						oldval = __atomic_write_uint32(&kds_final->nitems, 0);
						assert(oldval == UINT_MAX);
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
					assert(oldval == 0 || oldval == UINT_MAX);
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
		/* out of memory? */
		try_suspend = __shfl_sync(__activemask(), try_suspend, 0);
		if (try_suspend)
		{
			*p_try_suspend = true;
			return false;
		}
		/* is the destination tuple ready?  */
		tupitem = (kern_tupitem *)__shfl_sync(__activemask(), (uintptr_t)tupitem, 0);
		if (tupitem != NULL)
			break;
	}
	/* update partial aggregation */
	__updateOneTupleNoGroups(kcxt, kds_final,
							 kvars_is_valid,
							 &tupitem->htup,
							 kexp_groupby_actions);
	return true;
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
	union {
		uint64_t	u64;
		struct {
			uint32_t nitems;
			uint32_t usage;
		} kds;
	} oldval, curval, newval;

	assert(kds_final->format == KDS_FORMAT_HASH &&
		   kds_final->hash_nslots > 0);
	/* estimate length */
	tupsz = __writeOutOneTuplePreAgg(kcxt, kds_final, NULL,
									 kexp_groupby_actions);
	assert(tupsz > 0);

	/* expand kds_final */
	curval.u64 = __volatileRead((uint64_t *)&kds_final->nitems);
	for (;;)
	{
		uintptr_t	tup_addr;
		size_t		total_sz;

		/*
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
		tup_addr = TYPEALIGN_DOWN(CUDA_L1_CACHELINE_SZ,
								  (uintptr_t)kds_final
								  + kds_final->length
								  - __kds_unpack(curval.kds.usage)
								  - (offsetof(kern_hashitem, t.htup) + tupsz));
		newval.kds.nitems = curval.kds.nitems + 1;
		newval.kds.usage  = __kds_packed((uintptr_t)kds_final
										 + kds_final->length
										 - tup_addr);
		total_sz = (KDS_HEAD_LENGTH(kds_final) +
					MAXALIGN(sizeof(uint32_t) * (kds_final->hash_nslots +
												 newval.kds.nitems)) +
					__kds_unpack(newval.kds.usage));
		if (total_sz > kds_final->length)
			return NULL;	/* out of memory */
		oldval.u64 = __atomic_cas_uint64((uint64_t *)&kds_final->nitems,
										 curval.u64,
										 newval.u64);
		if (oldval.u64 == curval.u64)
			break;
		curval.u64 = oldval.u64;
	}
	hitem = (kern_hashitem *)((char *)kds_final
							  + kds_final->length
							  - __kds_unpack(newval.kds.usage));
	__writeOutOneTuplePreAgg(kcxt, kds_final,
							 &hitem->t.htup,
							 kexp_groupby_actions);
	hitem->t.t_len = tupsz;
	hitem->t.rowid = newval.kds.nitems - 1;
	KDS_GET_ROWINDEX(kds_final)[hitem->t.rowid]
		= __kds_packed((char *)kds_final
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
							kern_colmeta *cmeta,
							kern_aggregate_desc *desc)
{
	__atomic_add_uint64((uint64_t *)buffer, 1);
}

INLINE_FUNCTION(void)
__update_groupby__nrows_cond(kern_context *kcxt,
							 char *buffer,
							 kern_colmeta *cmeta,
							 kern_aggregate_desc *desc)
{
	if (kcxt->kvars_class[desc->arg0_slot_id] != KVAR_CLASS__NULL)
		__atomic_add_uint64((uint64_t *)buffer, 1);
}

INLINE_FUNCTION(void)
__update_groupby__pmin_int32(kern_context *kcxt,
							 char *buffer,
							 kern_colmeta *cmeta,
							 kern_aggregate_desc *desc)
{
	int		vclass = kcxt->kvars_class[desc->arg0_slot_id];

	if (vclass == KVAR_CLASS__INLINE)
	{
		kagg_state__pminmax_int64_packed *r =
			(kagg_state__pminmax_int64_packed *)buffer;
		int32_t		ival = kcxt->kvars_slot[desc->arg0_slot_id].i32;

		__atomic_add_uint32(&r->nitems, 1);
		__atomic_min_int64(&r->value, ival);
	}
	else
	{
		assert(vclass == KVAR_CLASS__NULL);
	}
}

INLINE_FUNCTION(void)
__update_groupby__pmin_int64(kern_context *kcxt,
							 char *buffer,
							 kern_colmeta *cmeta,
							 kern_aggregate_desc *desc)
{
	int		vclass = kcxt->kvars_class[desc->arg0_slot_id];

	if (vclass == KVAR_CLASS__INLINE)
	{
		kagg_state__pminmax_int64_packed *r =
			(kagg_state__pminmax_int64_packed *)buffer;
		int64_t		ival = kcxt->kvars_slot[desc->arg0_slot_id].i64;

		__atomic_add_uint32(&r->nitems, 1);
		__atomic_min_int64(&r->value, ival);
	}
	else
	{
		assert(vclass == KVAR_CLASS__NULL);
	}
}

INLINE_FUNCTION(void)
__update_groupby__pmax_int32(kern_context *kcxt,
							 char *buffer,
							 kern_colmeta *cmeta,
							 kern_aggregate_desc *desc)
{
	int		vclass = kcxt->kvars_class[desc->arg0_slot_id];

	if (vclass == KVAR_CLASS__INLINE)
	{
		kagg_state__pminmax_int64_packed *r =
			(kagg_state__pminmax_int64_packed *)buffer;
		int32_t		ival = kcxt->kvars_slot[desc->arg0_slot_id].i32;

		__atomic_add_uint32(&r->nitems, 1);
		__atomic_max_int64(&r->value, ival);
	}
	else
	{
		assert(vclass == KVAR_CLASS__NULL);
	}
}

INLINE_FUNCTION(void)
__update_groupby__pmax_int64(kern_context *kcxt,
							 char *buffer,
							 kern_colmeta *cmeta,
							 kern_aggregate_desc *desc)
{
	int		vclass = kcxt->kvars_class[desc->arg0_slot_id];

	if (vclass == KVAR_CLASS__INLINE)
	{
		kagg_state__pminmax_int64_packed *r =
			(kagg_state__pminmax_int64_packed *)buffer;
		int64_t		ival = kcxt->kvars_slot[desc->arg0_slot_id].i64;

		__atomic_add_uint32(&r->nitems, 1);
		__atomic_max_int64(&r->value, ival);
	}
	else
	{
		assert(vclass == KVAR_CLASS__NULL);
	}
}

INLINE_FUNCTION(void)
__update_groupby__pmin_fp64(kern_context *kcxt,
							char *buffer,
							kern_colmeta *cmeta,
							kern_aggregate_desc *desc)
{
	int		vclass = kcxt->kvars_class[desc->arg0_slot_id];

	if (vclass == KVAR_CLASS__INLINE)
	{
		kagg_state__pminmax_fp64_packed *r =
			(kagg_state__pminmax_fp64_packed *)buffer;
		float8_t	fval = kcxt->kvars_slot[desc->arg0_slot_id].fp64;

		__atomic_add_uint32(&r->nitems, 1);
		__atomic_min_fp64(&r->value, fval);
	}
	else
	{
		assert(vclass == KVAR_CLASS__NULL);
	}
}

INLINE_FUNCTION(void)
__update_groupby__pmax_fp64(kern_context *kcxt,
							char *buffer,
							kern_colmeta *cmeta,
							kern_aggregate_desc *desc)
{
	int		vclass = kcxt->kvars_class[desc->arg0_slot_id];

	if (vclass == KVAR_CLASS__INLINE)
	{
		kagg_state__pminmax_fp64_packed *r =
			(kagg_state__pminmax_fp64_packed *)buffer;
		float8_t	fval = kcxt->kvars_slot[desc->arg0_slot_id].fp64;

		__atomic_add_uint32(&r->nitems, 1);
		__atomic_max_fp64(&r->value, fval);
	}
	else
	{
		assert(vclass == KVAR_CLASS__NULL);
	}
}

INLINE_FUNCTION(void)
__update_groupby__psum_int(kern_context *kcxt,
						   char *buffer,
						   kern_colmeta *cmeta,
						   kern_aggregate_desc *desc)
{
	int		vclass = kcxt->kvars_class[desc->arg0_slot_id];

	if (vclass == KVAR_CLASS__INLINE)
	{
		kagg_state__psum_int_packed *r =
			(kagg_state__psum_int_packed *)buffer;
		int64_t		ival = kcxt->kvars_slot[desc->arg0_slot_id].i64;

		__atomic_add_uint32(&r->nitems, 1);
		__atomic_add_int64(&r->sum, ival);
	}
	else
	{
		 assert(vclass == KVAR_CLASS__NULL);
	}
}

INLINE_FUNCTION(void)
__update_groupby__psum_fp(kern_context *kcxt,
						  char *buffer,
						  kern_colmeta *cmeta,
						  kern_aggregate_desc *desc)
{
	int		vclass = kcxt->kvars_class[desc->arg0_slot_id];

	if (vclass == KVAR_CLASS__INLINE)
	{
		kagg_state__psum_fp_packed *r =
			(kagg_state__psum_fp_packed *)buffer;
		float8_t	fval = kcxt->kvars_slot[desc->arg0_slot_id].fp64;

		__atomic_add_uint32(&r->nitems, 1);
		__atomic_add_fp64(&r->sum, fval);
	}
	else
	{
		assert(vclass == KVAR_CLASS__NULL);
	}
}

INLINE_FUNCTION(void)
__update_groupby__pstddev(kern_context *kcxt,
						  char *buffer,
						  kern_colmeta *cmeta,
						  kern_aggregate_desc *desc)
{
	int		vclass = kcxt->kvars_class[desc->arg0_slot_id];

	if (vclass == KVAR_CLASS__INLINE)
	{
		kagg_state__stddev_packed *r =
			(kagg_state__stddev_packed *)buffer;
		float8_t	fval = kcxt->kvars_slot[desc->arg0_slot_id].fp64;

		__atomic_add_uint32(&r->nitems, 1);
		__atomic_add_fp64(&r->sum_x,  fval);
		__atomic_add_fp64(&r->sum_x2, fval * fval);
	}
	else
	{
		assert(vclass == KVAR_CLASS__NULL);
	}
}

INLINE_FUNCTION(void)
__update_groupby__pcovar(kern_context *kcxt,
						 char *buffer,
						 kern_colmeta *cmeta,
						 kern_aggregate_desc *desc)
{
	if (kcxt->kvars_class[desc->arg0_slot_id] == KVAR_CLASS__INLINE &&
		kcxt->kvars_class[desc->arg1_slot_id] == KVAR_CLASS__INLINE)
	{
		kagg_state__covar_packed *r =
			(kagg_state__covar_packed *)buffer;
		float8_t	xval = kcxt->kvars_slot[desc->arg0_slot_id].fp64;
		float8_t	yval = kcxt->kvars_slot[desc->arg0_slot_id].fp64;

		__atomic_add_uint32(&r->nitems, 1);
		__atomic_add_fp64(&r->sum_x,  xval);
		__atomic_add_fp64(&r->sum_xx, xval * xval);
		__atomic_add_fp64(&r->sum_y,  yval);
		__atomic_add_fp64(&r->sum_yy, yval * yval);
		__atomic_add_fp64(&r->sum_xy, xval * yval);
	}
	else
	{
		assert(kcxt->kvars_class[desc->arg0_slot_id] == KVAR_CLASS__NULL ||
			   kcxt->kvars_class[desc->arg1_slot_id] == KVAR_CLASS__NULL);
	}
}

/*
 * __updateOneTupleGroupBy
 */
STATIC_FUNCTION(void)
__updateOneTupleGroupBy(kern_context *kcxt,
						kern_data_store *kds_final,
						HeapTupleHeaderData *htup,
						kern_expression *kexp_groupby_actions)
{
	int			nattrs = (htup->t_infomask2 & HEAP_NATTS_MASK);
	bool		heap_hasnull = ((htup->t_infomask & HEAP_HASNULL) != 0);
	uint32_t	t_hoff;
	char	   *buffer;

	t_hoff = offsetof(HeapTupleHeaderData, t_bits);
	if (heap_hasnull)
		t_hoff += BITMAPLEN(nattrs);
	t_hoff = MAXALIGN(t_hoff);

	for (int j=0; j < nattrs; j++)
	{
		kern_aggregate_desc *desc = &kexp_groupby_actions->u.pagg.desc[j];
		kern_colmeta   *cmeta = &kds_final->colmeta[j];

		if (heap_hasnull && att_isnull(j, htup->t_bits))
		{
			/* only grouping-key may have NULL */
			assert(desc->action == KAGG_ACTION__VREF);
			continue;
		}

		if (cmeta->attlen > 0)
			t_hoff = TYPEALIGN(cmeta->attalign, t_hoff);
		else if (!VARATT_NOT_PAD_BYTE((char *)htup + t_hoff))
			t_hoff = TYPEALIGN(cmeta->attalign, t_hoff);
		buffer = ((char *)htup + t_hoff);
		if (cmeta->attlen > 0)
			t_hoff += cmeta->attlen;
		else
			t_hoff += VARSIZE_ANY(buffer);

		switch (desc->action)
        {
			case KAGG_ACTION__NROWS_ANY:
				__update_groupby__nrows_any(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__NROWS_COND:
				__update_groupby__nrows_cond(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__PMIN_INT32:
				__update_groupby__pmin_int32(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__PMIN_INT64:
				__update_groupby__pmin_int64(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__PMAX_INT32:
				__update_groupby__pmax_int32(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__PMAX_INT64:
				__update_groupby__pmax_int64(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__PMIN_FP64:
				__update_groupby__pmin_fp64(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__PMAX_FP64:
				__update_groupby__pmax_fp64(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__PAVG_INT:
			case KAGG_ACTION__PSUM_INT:
				__update_groupby__psum_int(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__PAVG_FP:
			case KAGG_ACTION__PSUM_FP:
				__update_groupby__psum_fp(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__STDDEV:
				__update_groupby__pstddev(kcxt, buffer, cmeta, desc);
				break;
			case KAGG_ACTION__COVAR:
				__update_groupby__pcovar(kcxt, buffer, cmeta, desc);
				break;
			default:
				/*
				 * No more partial aggregation exists after grouping-keys
				 */
				return;
		}
	}
}

STATIC_FUNCTION(int)
__execGpuPreAggGroupBy(kern_context *kcxt,
					   kern_data_store *kds_final,
					   bool kvars_is_valid,
					   kern_expression *kexp_groupby_keyhash,
					   kern_expression *kexp_groupby_keyload,
					   kern_expression *kexp_groupby_keycomp,
					   kern_expression *kexp_groupby_actions,
					   bool *p_try_suspend)
{
	kern_hashitem *hitem = NULL;
	xpu_int4_t	hash;

	assert(kds_final->format == KDS_FORMAT_HASH);
	/*
	 * compute hash value of the grouping keys
	 */
	memset(&hash, 0, sizeof(hash));
	if (kvars_is_valid)
	{
		if (EXEC_KERN_EXPRESSION(kcxt, kexp_groupby_keyhash, &hash))
			assert(!XPU_DATUM_ISNULL(&hash));
	}
	if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
		return false;

	/*
	 * lookup the destination grouping tuple. if not found, create a new one.
	 */
	do {
		if (!XPU_DATUM_ISNULL(&hash) && !hitem)
		{
			uint32_t   *hslot = KDS_GET_HASHSLOT(kds_final, hash.value);
			uint32_t	hoffset;
			uint32_t	saved;
			bool		has_lock = false;
			xpu_bool_t	status;

			hoffset = __volatileRead(hslot);
		try_again:
			for (hitem = KDS_HASH_NEXT_ITEM(kds_final, hoffset);
				 hitem != NULL;
				 hitem = KDS_HASH_NEXT_ITEM(kds_final, hitem->next))
			{
				bool	saved_compare_nulls = kcxt->kmode_compare_nulls;

				if (hitem->hash != hash.value)
					continue;

				kcxt->kmode_compare_nulls = true;
				ExecLoadVarsHeapTuple(kcxt, kexp_groupby_keyload,
									  -2,
									  kds_final,
									  &hitem->t.htup);
				if (EXEC_KERN_EXPRESSION(kcxt, kexp_groupby_keycomp, &status))
				{
					kcxt->kmode_compare_nulls = saved_compare_nulls;
					if (!XPU_DATUM_ISNULL(&status) && status.value)
						break;
				}
				kcxt->kmode_compare_nulls = saved_compare_nulls;
			}

			if (!hitem)
			{
				if (!has_lock)
				{
					/* try lock */
					saved = __volatileRead(hslot);
					if (saved != UINT_MAX &&
						__atomic_cas_uint32(hslot, saved, UINT_MAX) == saved)
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
						size_t		__offset;

						hitem->hash = hash.value;
						hitem->next = saved;
						__threadfence();
						__offset = ((char *)kds_final
									+ kds_final->length
									- (char *)hitem);
						__atomic_write_uint32(hslot, __kds_packed(__offset));
					}
					else
					{
						/* out of the memory, and unlcok */
						__atomic_write_uint32(hslot, saved);
						*p_try_suspend = true;
					}
					has_lock = false;
				}
			}
			else if (has_lock)
			{
				/* unlock */
				__atomic_write_uint32(hslot, saved);
			}
		}
		/* suspend the kernel? */
		if (__any_sync(__activemask(), *p_try_suspend))
			return false;
		/* error checks */
		if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
			return false;
		/* retry, if any threads are not ready yet */
	} while (__any_sync(__activemask(), !XPU_DATUM_ISNULL(&hash) && !hitem));

	/*
	 * update the partial aggregation
	 */
	if (hitem)
		__updateOneTupleGroupBy(kcxt, kds_final,
								&hitem->t.htup,
								kexp_groupby_actions);
	return true;
}

PUBLIC_FUNCTION(int)
execGpuPreAggGroupBy(kern_context *kcxt,
					 kern_warp_context *wp,
					 int n_rels,
					 kern_data_store *kds_final,
					 char *kvars_addr_wp,
					 bool *p_try_suspend)
{
	kern_session_info *session = kcxt->session;
	kern_expression *kexp_groupby_keyhash = SESSION_KEXP_GROUPBY_KEYHASH(session);
	kern_expression *kexp_groupby_keyload = SESSION_KEXP_GROUPBY_KEYLOAD(session);
	kern_expression *kexp_groupby_keycomp = SESSION_KEXP_GROUPBY_KEYCOMP(session);
	kern_expression *kexp_groupby_actions = SESSION_KEXP_GROUPBY_ACTIONS(session);
	kern_expression *karg;
	uint32_t	write_pos = WARP_WRITE_POS(wp,n_rels);
	uint32_t	read_pos = WARP_READ_POS(wp,n_rels);
	uint32_t	i, mask;
	bool		status;
	
	/*
	 * The previous depth still may produce new tuples, and number of
	 * the current result tuples is not sufficient to run projection.
	 */
	if (wp->scan_done <= n_rels && read_pos + warpSize > write_pos)
		return n_rels;

	read_pos += LaneId();
	if (read_pos < write_pos)
	{
		int		index = (read_pos % UNIT_TUPLES_PER_DEPTH);

		kcxt->kvars_slot = (kern_variable *)
			(kvars_addr_wp + index * kcxt->kvars_nbytes);
		kcxt->kvars_class = (int *)(kcxt->kvars_slot + kcxt->kvars_nslots);
	}
	else
	{
		kcxt->kvars_slot = NULL;
		kcxt->kvars_class = NULL;
	}
	mask = __ballot_sync(__activemask(), kcxt->kvars_class != NULL);
	if (mask == 0)
		goto skip_reduction;

	/*
	 * fillup the kvars_slot if it involves expressions
	 */
	if (kcxt->kvars_slot != NULL)
	{
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
	if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
		return -1;

	/*
	 * main logic of GpuPreAgg
	 */
	assert(kexp_groupby_actions != NULL);
	if (kexp_groupby_keyhash &&
		kexp_groupby_keyload &&
		kexp_groupby_keycomp)
	{
		status = __execGpuPreAggGroupBy(kcxt, kds_final,
										kcxt->kvars_slot != NULL,
										kexp_groupby_keyhash,
										kexp_groupby_keyload,
										kexp_groupby_keycomp,
										kexp_groupby_actions,
										p_try_suspend);
	}
	else
	{
		status = __execGpuPreAggNoGroups(kcxt, kds_final,
										 kcxt->kvars_slot != NULL,
										 kexp_groupby_actions,
										 p_try_suspend);
	}
	if (__any_sync(__activemask(), !status))
		return -1;

	/*
	 * Update the read position
	 */
skip_reduction:
	if (LaneId() == 0)
	{
		WARP_READ_POS(wp,n_rels) += __popc(mask);
		assert(WARP_WRITE_POS(wp,n_rels) >= WARP_READ_POS(wp,n_rels));
	}
	__syncwarp();
	if (wp->scan_done <= n_rels)
	{
		if (WARP_WRITE_POS(wp,n_rels) < WARP_READ_POS(wp,n_rels) + warpSize)
			return n_rels;	/* back to the previous depth */
    }
	else
	{
		if (WARP_READ_POS(wp,n_rels) >= WARP_WRITE_POS(wp,n_rels))
			return -1;		/* ok, end of GpuPreAgg */
	}
	return n_rels + 1;		/* elsewhere, try again? */
}
