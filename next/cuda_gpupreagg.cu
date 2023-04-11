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










INLINE_FUNCTION(bool)
__spinlock_try_acquire(unsigned int *lock, uint32_t *p_saved)
{
	unsigned int	oldval;

	if (LaneId() == 0)
	{
		oldval = __volatileRead(lock);
		if (oldval != UINT_MAX)
		{
			if (atomicCAS(lock, oldval, UINT_MAX) != oldval)
				oldval = UINT_MAX;	/* failed on lock */
		}
	}
	oldval = __shfl_sync(__activemask(), oldval, 0);
	if (oldval == UINT_MAX)
		return false;
	*p_saved = oldval;
	return true;
}

INLINE_FUNCTION(void)
__spinlock_release(unsigned int *lock, uint32_t *p_saved)
{
	unsigned int	oldval;

	if (LaneId() == 0)
	{
		oldval = atomicExch(lock, *p_saved);
	}
	oldval = __shfl_sync(__activemask(), oldval, 0);
	assert(oldval == UINT_MAX);
}

/*
 * Atomic operations
 */
INLINE_FUNCTION(uint32_t)
__atomic_write_uint32(uint32_t *ptr, uint32_t ival)
{
	return atomicExch((unsigned int *)ptr, ival);
}

INLINE_FUNCTION(uint64_t)
__atomic_write_uint64(uint64_t *ptr, uint64_t ival)
{
	return atomicExch((unsigned long long int *)ptr, ival);
}

INLINE_FUNCTION(uint32_t)
__atomic_add_uint32(uint32_t *ptr, uint32_t ival)
{
	return atomicAdd((unsigned int *)ptr, (unsigned int)ival);
}

INLINE_FUNCTION(uint64_t)
__atomic_add_uint64(uint64_t *ptr, uint64_t ival)
{
	return atomicAdd((unsigned long long *)ptr, (unsigned long long)ival);
}

INLINE_FUNCTION(int64_t)
__atomic_add_int64(int64_t *ptr, int64_t ival)
{
	return atomicAdd((unsigned long long int *)ptr, (unsigned long long int)ival);
}

INLINE_FUNCTION(float8_t)
__atomic_add_fp64(float8_t *ptr, float8_t fval)
{
	return atomicAdd((double *)ptr, (double)fval);
}

INLINE_FUNCTION(int64_t)
__atomic_min_int64(int64_t *ptr, int64_t ival)
{
	return atomicMin((long long int *)ptr, (long long int)ival);
}

INLINE_FUNCTION(int64_t)
__atomic_max_int64(int64_t *ptr, int64_t ival)
{
	return atomicMax((long long int *)ptr, (long long int)ival);
}

INLINE_FUNCTION(float8_t)
__atomic_min_fp64(float8_t *ptr, float8_t fval)
{
	union {
		unsigned long long ival;
		float8_t	fval;
	} oldval, curval, newval;

	newval.fval = fval;
	curval.fval = __volatileRead(ptr);
	while (newval.fval < curval.fval)
	{
		oldval = curval;
		curval.ival = atomicCAS((unsigned long long *)ptr,
								oldval.ival,
								newval.ival);
		if (curval.ival == oldval.ival)
			break;
	}
	return curval.fval;
}

INLINE_FUNCTION(float8_t)
__atomic_max_fp64(float8_t *ptr, float8_t fval)
{
	union {
		unsigned long long ival;
		float8_t	fval;
	} oldval, curval, newval;

	newval.fval = fval;
	curval.fval = __volatileRead(ptr);
	while (newval.fval > curval.fval)
	{
		oldval = curval;
		curval.ival = atomicCAS((unsigned long long *)ptr,
								oldval.ival,
								newval.ival);
		if (curval.ival == oldval.ival)
			break;
	}
	return curval.fval;
}









/*
 * xpuPreAggWriteOutOneTuple
 */
STATIC_FUNCTION(int32_t)
__xpuPreAggWriteOutGroupKey(kern_context *kcxt,
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
				xpu_datum_t *xdatum = (xpu_datum_t *)
					((char *)kcxt->kvars_slot + kvar->xpu.offset);
				const xpu_datum_operators *expr_ops = xdatum->expr_ops;

				if (XPU_DATUM_ISNULL(xdatum))
					return 0;
				assert(expr_ops->xpu_type_code == kvar->xpu.type_code);
				return expr_ops->xpu_datum_write(kcxt, buffer, xdatum);
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
xpuPreAggWriteOutOneTuple(kern_context *kcxt,
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
		int		nbytes;

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
				nbytes = __xpuPreAggWriteOutGroupKey(kcxt, cmeta, desc, buffer);
				if (nbytes < 0)
					return -1;
				break;

			case KAGG_ACTION__NROWS_ANY:
			case KAGG_ACTION__NROWS_COND:
			case KAGG_ACTION__PSUM_INT:
				t_next += sizeof(int64_t);
				if (buffer)
					*((int64_t *)buffer) = 0;
				break;

			case KAGG_ACTION__PSUM_FP:
				t_next += sizeof(float8_t);
				if (buffer)
					*((float8_t *)buffer) = 0.0;
				break;

			case KAGG_ACTION__PMIN_INT:
				t_next += sizeof(kagg_state__pminmax_int64_packed);
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

			case KAGG_ACTION__PMAX_INT:
				t_next += sizeof(kagg_state__pminmax_int64_packed);
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

			case KAGG_ACTION__PMIN_FP:
				t_next += sizeof(kagg_state__pminmax_fp64_packed);
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

			case KAGG_ACTION__PMAX_FP:
				t_next += sizeof(kagg_state__pminmax_fp64_packed);
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

			case KAGG_ACTION__PAVG_INT:
				t_next += sizeof(kagg_state__pavg_int_packed);
				if (buffer)
				{
					memset(buffer, 0, sizeof(kagg_state__pavg_int_packed));
					SET_VARSIZE(buffer, sizeof(kagg_state__pavg_int_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__PAVG_FP:
				t_next += sizeof(kagg_state__pavg_fp_packed);
				if (buffer)
				{
					memset(buffer, 0, sizeof(kagg_state__pavg_fp_packed));
					SET_VARSIZE(buffer, sizeof(kagg_state__pavg_fp_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__STDDEV:
				t_next += sizeof(kagg_state__stddev_packed);
				if (buffer)
				{
					memset(buffer, 0, sizeof(kagg_state__stddev_packed));
					SET_VARSIZE(buffer, sizeof(kagg_state__stddev_packed));
				}
				t_infomask |= HEAP_HASVARWIDTH;
				break;

			case KAGG_ACTION__COVAR:
				t_next += sizeof(kagg_state__covar_packed);
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
		if (nbytes > 0)
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
INLINE_FUNCTION(void)
__update_nogroups__pmin_int(kern_context *kcxt,
							char *buffer,
							kern_colmeta *cmeta,
							kern_aggregate_desc *desc,
							bool kvars_is_valid)
{
	int64_t		ival = LONG_MAX;
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

		ival = Min(ival, __shfl_xor_sync(__activemask(), ival, 0x0001));
		ival = Min(ival, __shfl_xor_sync(__activemask(), ival, 0x0002));
		ival = Min(ival, __shfl_xor_sync(__activemask(), ival, 0x0004));
		ival = Min(ival, __shfl_xor_sync(__activemask(), ival, 0x0008));
		ival = Min(ival, __shfl_xor_sync(__activemask(), ival, 0x0010));

		if (LaneId() == 0)
		{
			__atomic_add_uint32(&r->nitems, __popc(mask));
			__atomic_min_int64(&r->value, ival);
		}
	}
}

/*
 * __update_nogroups__pmax_int
 */
INLINE_FUNCTION(void)
__update_nogroups__pmax_int(kern_context *kcxt,
							char *buffer,
							kern_colmeta *cmeta,
							kern_aggregate_desc *desc,
							bool kvars_is_valid)
{
	int64_t		ival = LONG_MIN;
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

		ival = Max(ival, __shfl_xor_sync(__activemask(), ival, 0x0001));
		ival = Max(ival, __shfl_xor_sync(__activemask(), ival, 0x0002));
		ival = Max(ival, __shfl_xor_sync(__activemask(), ival, 0x0004));
		ival = Max(ival, __shfl_xor_sync(__activemask(), ival, 0x0008));
		ival = Max(ival, __shfl_xor_sync(__activemask(), ival, 0x0010));

		if (LaneId() == 0)
		{
			__atomic_add_uint32(&r->nitems, __popc(mask));
			__atomic_max_int64(&r->value,  ival);
		}
	}
}

/*
 * __update_nogroups__pmin_fp
 */
INLINE_FUNCTION(void)
__update_nogroups__pmin_fp(kern_context *kcxt,
						   char *buffer,
						   kern_colmeta *cmeta,
						   kern_aggregate_desc *desc,
						   bool kvars_is_valid)
{
	float8_t	fval = DBL_MAX;
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

		fval = Min(fval, __shfl_xor_sync(__activemask(), fval, 0x0001));
		fval = Min(fval, __shfl_xor_sync(__activemask(), fval, 0x0002));
		fval = Min(fval, __shfl_xor_sync(__activemask(), fval, 0x0004));
		fval = Min(fval, __shfl_xor_sync(__activemask(), fval, 0x0008));
		fval = Min(fval, __shfl_xor_sync(__activemask(), fval, 0x0010));

		if (LaneId() == 0)
		{
			__atomic_add_uint32(&r->nitems, __popc(mask));
			__atomic_min_fp64(&r->value, fval);
		}
	}
}

/*
 * __update_nogroups__pmax_fp
 */
INLINE_FUNCTION(void)
__update_nogroups__pmax_fp(kern_context *kcxt,
						   char *buffer,
						   kern_colmeta *cmeta,
						   kern_aggregate_desc *desc,
						   bool kvars_is_valid)
{
	float8_t	fval = -DBL_MAX;
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

		fval = Min(fval, __shfl_xor_sync(__activemask(), fval, 0x0001));
		fval = Min(fval, __shfl_xor_sync(__activemask(), fval, 0x0002));
		fval = Min(fval, __shfl_xor_sync(__activemask(), fval, 0x0004));
		fval = Min(fval, __shfl_xor_sync(__activemask(), fval, 0x0008));
		fval = Min(fval, __shfl_xor_sync(__activemask(), fval, 0x0010));

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
			__atomic_add_int64((int64_t *)buffer, ival);
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
	float8_t	fval = 0.0;
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
			__atomic_add_fp64((float8_t *)buffer, fval);
	}
}

/*
 * __update_nogroups__pavg_int
 */
INLINE_FUNCTION(void)
__update_nogroups__pavg_int(kern_context *kcxt,
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
			kagg_state__pavg_int_packed *r =
				(kagg_state__pavg_int_packed *)buffer;
			__atomic_add_uint32(&r->nitems, __popc(mask));
			__atomic_add_int64(&r->sum, ival);
		}
	}
}

/*
 * __update_nogroups__pavg_fp
 */
INLINE_FUNCTION(void)
__update_nogroups__pavg_fp(kern_context *kcxt,
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
			kagg_state__pavg_fp_packed *r =
				(kagg_state__pavg_fp_packed *)buffer;
			__atomic_add_uint32(&r->nitems, __popc(mask));
			__atomic_add_fp64(&r->sum, fval);
		}
	}
}
/*
 * __update_nogroups__pavg_stddev
 */
INLINE_FUNCTION(void)
__update_nogroups__pavg_stddev(kern_context *kcxt,
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
__update_nogroups__pavg_covar(kern_context *kcxt,
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
STATIC_FUNCTION(bool)
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
			case KAGG_ACTION__PMIN_INT:
				__update_nogroups__pmin_int(kcxt, buffer,
											cmeta, desc,
											kvars_is_valid);
				break;
			case KAGG_ACTION__PMAX_INT:
				__update_nogroups__pmax_int(kcxt, buffer,
											cmeta, desc,
											kvars_is_valid);
				break;
			case KAGG_ACTION__PMIN_FP:
				__update_nogroups__pmin_fp(kcxt, buffer,
										   cmeta, desc,
										   kvars_is_valid);
				break;
			case KAGG_ACTION__PMAX_FP:
				__update_nogroups__pmax_fp(kcxt, buffer,
										   cmeta, desc,
										   kvars_is_valid);
				break;
			case KAGG_ACTION__PSUM_INT:
				__update_nogroups__psum_int(kcxt, buffer,
											cmeta, desc,
											kvars_is_valid);
				break;
			case KAGG_ACTION__PSUM_FP:
				__update_nogroups__psum_fp(kcxt, buffer,
										   cmeta, desc,
										   kvars_is_valid);
				break;
			case KAGG_ACTION__PAVG_INT:
				__update_nogroups__pavg_int(kcxt, buffer,
											cmeta, desc,
											kvars_is_valid);
				break;
			case KAGG_ACTION__PAVG_FP:
				__update_nogroups__pavg_fp(kcxt, buffer,
										   cmeta, desc,
										   kvars_is_valid);
				break;
			case KAGG_ACTION__STDDEV:
				__update_nogroups__pavg_stddev(kcxt, buffer,
											   cmeta, desc,
											   kvars_is_valid);
				break;
			case KAGG_ACTION__COVAR:
				__update_nogroups__pavg_covar(kcxt, buffer,
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
 * __insertOneTupleGroupBy
 */
STATIC_FUNCTION(bool)
__insertOneTupleGroupBy(kern_context *kcxt,
						kern_data_store *kds_final,
						kern_expression *kexp_groupby_actions)
{
	int32_t				tupsz;
	int32_t				item_sz;
	size_t				total_sz;
	union {
		uint64_t		u64;
		struct {
			uint32_t	nitems;
			uint32_t	usage;
		} kds;
	} u;
	assert(kds_final->format == KDS_FORMAT_ROW ||
		   kds_final->format == KDS_FORMAT_HASH);
	if (LaneId() == 0)
	{
		kern_tupitem   *tupitem;

		tupsz = xpuPreAggWriteOutOneTuple(kcxt, kds_final, NULL,
										  kexp_groupby_actions);
		assert(tupsz > 0);
		item_sz = MAXALIGN(offsetof(kern_tupitem, htup) + tupsz);
		total_sz = (KDS_HEAD_LENGTH(kds_final) +
					MAXALIGN(sizeof(uint32_t) * (kds_final->nitems + 1)) +
					item_sz + __kds_unpack(kds_final->usage));
		if (total_sz > kds_final->length)
			return false;
		tupitem = (kern_tupitem *)((char *)kds_final
								   + kds_final->length
								   - __kds_unpack(kds_final->usage)
								   - item_sz);
		xpuPreAggWriteOutOneTuple(kcxt, kds_final,
								  &tupitem->htup,
								  kexp_groupby_actions);
		tupitem->t_len = tupsz;
		tupitem->rowid = kds_final->nitems;
		u.kds.nitems = kds_final->nitems + 1;
		u.kds.usage  = kds_final->usage  + __kds_packed(item_sz);
		KDS_GET_ROWINDEX(kds_final)[kds_final->nitems] = u.kds.usage;
		/* !!visible to other threads!! */
		__atomic_write_uint64((uint64_t *)&kds_final->nitems, u.u64);
	}
	return true;
}

STATIC_FUNCTION(bool)
__execGpuPreAggNoGroups(kern_context *kcxt,
						kern_data_store *kds_final,
						bool kvars_is_valid,
						kern_expression *kexp_groupby_actions)
{
	kern_tupitem *tupitem;
	uint32_t	nitems;
	uint32_t	saved;
	bool		status;
	bool		has_lock = false;

	assert(kds_final->format == KDS_FORMAT_ROW);
	assert(kexp_groupby_actions->opcode == FuncOpCode__AggFuncs);
retry:
	if (LaneId() == 0)
		nitems = __volatileRead(&kds_final->nitems);
	nitems = __shfl_sync(__activemask(), nitems, 0);
	if (nitems == 0)
	{
		if (!has_lock)
		{
			if (__spinlock_try_acquire(&kds_final->lock, &saved))
				has_lock = true;
			goto retry;
		}
		status = __insertOneTupleGroupBy(kcxt, kds_final,
										 kexp_groupby_actions);
		if (__any_sync(__activemask(), !status))
			return -2;		/* informs out of memory */
	}
	else
	{
		assert(nitems == 1);
	}
	if (has_lock)
		__spinlock_release(&kds_final->lock, &saved);

	tupitem = KDS_GET_TUPITEM(kds_final, 0);
	status = __updateOneTupleNoGroups(kcxt, kds_final,
									  kvars_is_valid,
									  &tupitem->htup,
									  kexp_groupby_actions);
	if (__any_sync(__activemask(), status))
		return -1;

	return 0;	/* ok */
}

STATIC_FUNCTION(int)
__execGpuPreAggGroupBy(kern_context *kcxt,
					   kern_data_store *kds_final,
					   bool kvars_is_valid,
					   kern_expression *kexp_groupby_keyhash,
					   kern_expression *kexp_groupby_keycomp,
					   kern_expression *kexp_groupby_actions)
{
	return 0;
}

PUBLIC_FUNCTION(int)
execGpuPreAggGroupBy(kern_context *kcxt,
					 kern_warp_context *wp,
					 int n_rels,
					 kern_data_store *kds_final,
					 kern_expression *kexp_groupby_keyhash,
					 kern_expression *kexp_groupby_keycomp,
					 kern_expression *kexp_groupby_actions,
					 char *kvars_addr_wp)
{
	kern_expression *karg;
	uint32_t	write_pos = WARP_WRITE_POS(wp,n_rels);
	uint32_t	read_pos = WARP_READ_POS(wp,n_rels);
	uint32_t	mask;
	bool		kvars_is_valid = true;
	int			i, status;

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
		kvars_is_valid = false;
	}
	mask = __ballot_sync(__activemask(), kvars_is_valid);
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
	if (kexp_groupby_keyhash && kexp_groupby_keycomp)
	{
		status = __execGpuPreAggGroupBy(kcxt, kds_final,
										kvars_is_valid,
										kexp_groupby_keyhash,
										kexp_groupby_keycomp,
										kexp_groupby_actions);
	}
	else
	{
		status = __execGpuPreAggNoGroups(kcxt, kds_final,
										 kvars_is_valid,
										 kexp_groupby_actions);
	}

	if (status < 0)
	{
		assert(__activemask() == 0xffffffffU);
		return status;
	}
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
