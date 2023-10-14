/*
 * xpu_common.cu
 *
 * Core implementation of GPU/DPU device code
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "xpu_common.h"

/* ----------------------------------------------------------------
 *
 * LoadVars / Projection Routines
 *
 * ----------------------------------------------------------------
 */
INLINE_FUNCTION(bool)
__extract_heap_tuple_attr(kern_context *kcxt,
						  const kern_varload_desc *vl_desc,
						  const char *addr)
{
	uint16_t	slot_id = vl_desc->vl_slot_id;

	if (slot_id < kcxt->kvars_nslots)
	{
		xpu_datum_t	   *xdatum = kcxt->kvars_values[slot_id];

		assert(vl_desc->vl_type_code == kcxt->kvars_desc[slot_id].vs_type_code);
		return vl_desc->vl_ops->xpu_datum_heap_ref(kcxt, addr, xdatum);
	}
	STROM_ELOG(kcxt, "vl_desc::slot_id is out of range");
	return false;
}

INLINE_FUNCTION(bool)
__fillup_by_null_values(kern_context *kcxt,
						const kern_varload_desc *vl_desc,
						int kvload_nitems)
{
	while (kvload_nitems > 0)
	{
		if (!__extract_heap_tuple_attr(kcxt, vl_desc, NULL))
			return false;
		vl_desc++;
		kvload_nitems--;
	}
	return true;
}

INLINE_FUNCTION(bool)
__extract_heap_tuple_sysattr(kern_context *kcxt,
							 const kern_data_store *kds,
							 const HeapTupleHeaderData *htup,
							 const kern_varload_desc *vl_desc)
{
	uint16_t	slot_id = vl_desc->vl_slot_id;

	if (slot_id < kcxt->kvars_nslots)
	{
		xpu_datum_t	   *xdatum = kcxt->kvars_values[slot_id];
		const void	   *addr;

		switch (vl_desc->vl_resno)
		{
			case SelfItemPointerAttributeNumber:
				addr = &htup->t_ctid;
				break;
			case MinTransactionIdAttributeNumber:
				addr = &htup->t_choice.t_heap.t_xmin;
				break;
			case MaxTransactionIdAttributeNumber:
				addr = &htup->t_choice.t_heap.t_xmax;
				break;
			case MinCommandIdAttributeNumber:
			case MaxCommandIdAttributeNumber:
				addr = &htup->t_choice.t_heap.t_field3.t_cid;
				break;
			case TableOidAttributeNumber:
				addr = &kds->table_oid;
				break;
			default:
				STROM_ELOG(kcxt, "not a supported system attribute reference");
				return false;
		}
		return vl_desc->vl_ops->xpu_datum_heap_ref(kcxt, addr, xdatum);
	}
	STROM_ELOG(kcxt, "kvars slot-id: out of range");
	return false;
}

STATIC_FUNCTION(bool)
kern_extract_heap_tuple(kern_context *kcxt,
						const kern_data_store *kds,
						const HeapTupleHeaderData *htup,
						const kern_varload_desc *vl_desc,
						int kvload_nitems)
{
	uint32_t	offset = htup->t_hoff;
	int			resno = 1;
	int			kvload_count = 0;
	int			ncols = Min(htup->t_infomask2 & HEAP_NATTS_MASK, kds->ncols);
	bool		heap_hasnull = ((htup->t_infomask & HEAP_HASNULL) != 0);

	/* extract system attributes, if rquired */
	while (kvload_count < kvload_nitems &&
		   vl_desc->vl_resno < 0)
	{
		if (!__extract_heap_tuple_sysattr(kcxt, kds, htup, vl_desc))
			return;
		vl_desc++;
		kvload_count++;
	}
	/* try attcacheoff shortcut, if available. */
	if (!heap_hasnull)
	{
		while (kvload_count < kvload_nitems &&
			   vl_desc->vl_resno > 0 &&
			   vl_desc->vl_resno <= ncols)
		{
			const kern_colmeta *cmeta = &kds->colmeta[vl_desc->vl_resno-1];
			char	   *addr;

			if (cmeta->attcacheoff < 0)
				break;
			offset = htup->t_hoff + cmeta->attcacheoff;
			addr = (char *)htup + offset;
			if (!__extract_heap_tuple_attr(kcxt, vl_desc, addr))
				return false;
			/* next resno */
			resno = vl_desc->vl_resno + 1;
			if (cmeta->attlen > 0)
				offset += cmeta->attlen;
			else
				offset += VARSIZE_ANY(addr);
			vl_desc++;
			kvload_count++;
		}
	}

	/* extract slow path */
	while (resno <= ncols && kvload_count < kvload_nitems)
	{
		const kern_colmeta *cmeta = &kds->colmeta[resno-1];
		char   *addr;

		if (heap_hasnull && att_isnull(resno-1, htup->t_bits))
		{
			addr = NULL;
		}
		else
		{
			if (cmeta->attlen > 0)
				offset = TYPEALIGN(cmeta->attalign, offset);
			else if (!VARATT_NOT_PAD_BYTE((char *)htup + offset))
				offset = TYPEALIGN(cmeta->attalign, offset);

			addr = ((char *)htup + offset);
			if (cmeta->attlen > 0)
				offset += cmeta->attlen;
			else
				offset += VARSIZE_ANY(addr);
		}

		if (vl_desc->vl_resno == resno)
		{
			if (!__extract_heap_tuple_attr(kcxt, vl_desc, addr))
				return false;
			vl_desc++;
			kvload_count++;
		}
		resno++;
	}
	/* fill-up with NULLs for the remained slot */
	return __fillup_by_null_values(kcxt, vl_desc, kvload_nitems - kvload_count);
}

/*
 * Routines to extract Arrow data store
 */
INLINE_FUNCTION(bool)
arrow_bitmap_check(const kern_data_store *kds,
				   uint32_t kds_index,
				   uint32_t bitmap_offset,
				   uint32_t bitmap_length)
{
	uint8_t	   *bitmap;
	uint8_t		mask = (1<<(kds_index & 7));
	uint32_t	idx = (kds_index >> 3);

	if (bitmap_offset == 0 ||	/* no bitmap */
		bitmap_length == 0 ||	/* no bitmap */
		idx >= __kds_unpack(bitmap_length))		/* out of range */
		return false;
	bitmap = (uint8_t *)kds + __kds_unpack(bitmap_offset);

	return (bitmap[idx] & mask) != 0;
}

STATIC_FUNCTION(bool)
arrow_fetch_secondary_index(kern_context *kcxt,
							const kern_data_store *kds,
							uint32_t kds_index,
							uint32_t values_offset,
							uint32_t values_length,
							bool is_large_offset,
							uint64_t *p_start,
							uint64_t *p_end)
{
	if (!values_offset || !values_length)
	{
		STROM_ELOG(kcxt, "Arrow variable index/buffer is missing");
		return false;
	}

	if (is_large_offset)
	{
		uint64_t   *base = (uint64_t *)((char *)kds + __kds_unpack(values_offset));

		if (sizeof(uint64_t) * (kds_index+2) > __kds_unpack(values_length))
		{
			STROM_ELOG(kcxt, "Arrow variable index[64bit] out of range");
			return false;
		}
		*p_start = base[kds_index];
		*p_end = base[kds_index+1];
	}
	else
	{
		uint32_t   *base = (uint32_t *)((char *)kds + __kds_unpack(values_offset));

		if (sizeof(uint32_t) * (kds_index+2) > __kds_unpack(values_length))
		{
			STROM_ELOG(kcxt, "Arrow variable index[32bit] out of range");
			return false;
		}
		*p_start = base[kds_index];
		*p_end = base[kds_index+1];

	}
	return true;
}

STATIC_FUNCTION(bool)
__arrow_fetch_bool_datum(kern_context *kcxt,
						 const kern_data_store *kds,
						 const kern_colmeta *cmeta,
						 uint32_t kds_index,
						 kern_variable *kvar,
						 int *vclass)
{
	assert(cmeta->extra_offset == 0 &&
		   cmeta->extra_length == 0);
	kvar->i8 = arrow_bitmap_check(kds, kds_index,
								  cmeta->values_offset,
								  cmeta->values_length);
	*vclass = KVAR_CLASS__INLINE;
	return true;
}

STATIC_FUNCTION(bool)
__arrow_fetch_int_datum(kern_context *kcxt,
						const kern_data_store *kds,
						const kern_colmeta *cmeta,
						uint32_t kds_index,
						kern_variable *kvar,
						int *vclass)
{
	size_t		values_length = __kds_unpack(cmeta->values_length);

	assert(cmeta->extra_offset == 0 &&
		   cmeta->extra_length == 0);
	switch (cmeta->attopts.integer.bitWidth)
	{
		case 8:
			if (cmeta->values_offset &&
				sizeof(uint8_t) * (kds_index+1) <= values_length)
			{
				uint8_t	   *base = (uint8_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				kvar->u8 = base[kds_index];
				*vclass = KVAR_CLASS__INLINE;
				return true;
			}
			break;
		case 16:
			if (cmeta->values_offset &&
				sizeof(uint16_t) * (kds_index+1) <= values_length)
			{
				uint16_t   *base = (uint16_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				kvar->u16 = base[kds_index];
				*vclass = KVAR_CLASS__INLINE;
				return true;
			}
			break;
		case 32:
			if (cmeta->values_offset &&
				sizeof(uint32_t) * (kds_index+1) <= values_length)
			{
				uint32_t   *base = (uint32_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				kvar->u32 = base[kds_index];
				*vclass = KVAR_CLASS__INLINE;
				return true;
			}
			break;
		case 64:
			if (cmeta->values_offset &&
				sizeof(uint64_t) * (kds_index+1) <= values_length)
			{
				uint64_t   *base = (uint64_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				kvar->u64 = base[kds_index];
				*vclass = KVAR_CLASS__INLINE;
				return true;
			}
			break;
		default:
			STROM_ELOG(kcxt, "Arrow::Int unsupported bitWidth");
			return false;
	}
	*vclass = KVAR_CLASS__NULL;
    return true;
}

STATIC_FUNCTION(bool)
__arrow_fetch_float_datum(kern_context *kcxt,
						  const kern_data_store *kds,
						  const kern_colmeta *cmeta,
						  uint32_t kds_index,
						  kern_variable *kvar,
						  int *vclass)
{
	size_t		values_length = __kds_unpack(cmeta->values_length);

	assert(cmeta->extra_offset == 0 &&
		   cmeta->extra_length == 0);
	switch (cmeta->attopts.floating_point.precision)
	{
		case ArrowPrecision__Half:
			if (cmeta->values_offset &&
				sizeof(float2_t) * (kds_index+1) <= cmeta->values_length)
			{
				float2_t   *base = (float2_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				kvar->fp16 = base[kds_index];
				*vclass = KVAR_CLASS__INLINE;
				return true;
			}
			break;
		case ArrowPrecision__Single:
			if (cmeta->values_offset &&
				sizeof(float4_t) * (kds_index+1) <= values_length)
			{
				float4_t   *base = (float4_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				kvar->fp32 = base[kds_index];
				*vclass = KVAR_CLASS__INLINE;
				return true;
			}
			break;
		case ArrowPrecision__Double:
			if (cmeta->values_offset &&
				sizeof(float8_t) * (kds_index+1) <= values_length)
			{
				float8_t   *base = (float8_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				kvar->fp64 = base[kds_index];
				*vclass = KVAR_CLASS__INLINE;
				return true;
			}
			break;
		default:
			STROM_ELOG(kcxt, "Arrow::FloatingPoint unsupported precision");
			return false;
	}
	*vclass = KVAR_CLASS__NULL;
	return true;
}

STATIC_FUNCTION(bool)
__arrow_fetch_decimal_datum(kern_context *kcxt,
							const kern_data_store *kds,
							const kern_colmeta *cmeta,
							uint32_t kds_index,
							kern_variable *kvar,
							int *vclass,
							char *slot_buf)
{
	assert(cmeta->extra_offset == 0 &&
		   cmeta->extra_length == 0 &&
		   slot_buf != NULL);
	if (cmeta->attopts.decimal.bitWidth != 128)
	{
		STROM_ELOG(kcxt, "Arrow::Decimal unsupported bitWidth");
		return false;
	}
	if (cmeta->values_offset &&
		sizeof(int128_t) * (kds_index+1) <= __kds_unpack(cmeta->values_length))
	{
		xpu_numeric_t  *num = (xpu_numeric_t *)slot_buf;
		int128_t	   *base = (int128_t *)
			((char *)kds + __kds_unpack(cmeta->values_offset));

		assert((((uintptr_t)base) & (sizeof(int128_t)-1)) == 0);
		set_normalized_numeric(num, base[kds_index],
							   cmeta->attopts.decimal.scale);
		*vclass = KVAR_CLASS__XPU_DATUM;
		kvar->ptr = num;
	}
	else
	{
		*vclass = KVAR_CLASS__NULL;
	}
	return true;
}

STATIC_FUNCTION(bool)
__arrow_fetch_date_datum(kern_context *kcxt,
						 const kern_data_store *kds,
						 const kern_colmeta *cmeta,
						 uint32_t kds_index,
						 kern_variable *kvar,
						 int *vclass)
{
	size_t		values_length = __kds_unpack(cmeta->values_length);

	assert(cmeta->extra_offset == 0 &&
		   cmeta->extra_length == 0);
	switch (cmeta->attopts.date.unit)
	{
		case ArrowDateUnit__Day:
			if (cmeta->values_offset &&
				sizeof(uint32_t) * (kds_index+1) <= values_length)
			{
				uint32_t   *base = (uint32_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				kvar->u32 = base[kds_index]
					- (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
				*vclass = KVAR_CLASS__INLINE;
				return true;
			}
			break;

		case ArrowDateUnit__MilliSecond:
			if (cmeta->values_offset &&
				sizeof(uint64_t) * (kds_index+1) <= values_length)
			{
				uint64_t   *base = (uint64_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				kvar->u32 = base[kds_index] / (SECS_PER_DAY * 1000)
					- (POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE);
				*vclass = KVAR_CLASS__INLINE;
				return true;
			}
			break;

		default:
			STROM_ELOG(kcxt, "unknown unit size of Arrow::Date");
			return false;
	}
	*vclass = KVAR_CLASS__NULL;
	return true;
}

STATIC_FUNCTION(bool)
__arrow_fetch_time_datum(kern_context *kcxt,
						 const kern_data_store *kds,
						 const kern_colmeta *cmeta,
						 uint32_t kds_index,
						 kern_variable *kvar,
						 int *vclass)
{
	size_t		values_length = __kds_unpack(cmeta->values_length);

	assert(cmeta->extra_offset == 0 &&
		   cmeta->extra_length == 0);
	switch (cmeta->attopts.time.unit)
	{
		case ArrowTimeUnit__Second:
			if (cmeta->values_offset &&
				sizeof(int32_t) * (kds_index+1) <= values_length)
			{
				int32_t   *base = (int32_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				kvar->i64 = (int64_t)base[kds_index] * 1000000L;
				*vclass = KVAR_CLASS__INLINE;
				return true;
			}
			break;
			
		case ArrowTimeUnit__MilliSecond:
			if (cmeta->values_offset &&
				sizeof(int32_t) * (kds_index+1) <= values_length)
			{
				int32_t	   *base = (int32_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				kvar->i64 = (int64_t)base[kds_index] * 1000L;
				*vclass = KVAR_CLASS__INLINE;
				return true;
			}
			break;

		case ArrowTimeUnit__MicroSecond:
			if (cmeta->values_offset &&
				sizeof(int64_t) * (kds_index+1) <= values_length)
			{
				int64_t	   *base = (int64_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				kvar->i64 = base[kds_index];
				*vclass = KVAR_CLASS__INLINE;
				return true;
			}
			break;

		case ArrowTimeUnit__NanoSecond:
			if (cmeta->values_offset &&
				sizeof(int64_t) * (kds_index+1) <= values_length)
			{
				int64_t	   *base = (int64_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				kvar->i64 = base[kds_index] / 1000L;
				*vclass = KVAR_CLASS__INLINE;
				return true;
			}
			break;

		default:
			STROM_ELOG(kcxt, "unknown unit size of Arrow::Time");
			return false;
	}
	*vclass = KVAR_CLASS__NULL;
	return true;
}

STATIC_FUNCTION(bool)
__arrow_fetch_timestamp_datum(kern_context *kcxt,
							  const kern_data_store *kds,
							  const kern_colmeta *cmeta,
							  uint32_t kds_index,
							  kern_variable *kvar,
							  int *vclass)

{
	size_t		values_length = __kds_unpack(cmeta->values_length);

	assert(cmeta->extra_offset == 0 &&
		   cmeta->extra_length == 0);
	switch (cmeta->attopts.time.unit)
	{
		case ArrowTimeUnit__Second:
			if (cmeta->values_offset &&
				sizeof(uint64_t) * (kds_index+1) <= values_length)
			{
				uint64_t   *base = (uint64_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				kvar->u64 = base[kds_index] * 1000000L -
					(POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE) * USECS_PER_DAY;
				*vclass = KVAR_CLASS__INLINE;
				return true;
			}
			break;

		case ArrowTimeUnit__MilliSecond:
			if (cmeta->values_offset &&
				sizeof(uint64_t) * (kds_index+1) <= values_length)
			{
				uint64_t   *base = (uint64_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				kvar->u64 = base[kds_index] * 1000L -
					(POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE) * USECS_PER_DAY;
				*vclass = KVAR_CLASS__INLINE;
				return true;
			}
			break;

		case ArrowTimeUnit__MicroSecond:
			if (cmeta->values_offset &&
				sizeof(uint64_t) * (kds_index+1) <= values_length)
			{
				uint64_t   *base = (uint64_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				kvar->u64 = base[kds_index] -
					(POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE) * USECS_PER_DAY;
				*vclass = KVAR_CLASS__INLINE;
				return true;
			}
			break;

		case ArrowTimeUnit__NanoSecond:
			if (cmeta->values_offset &&
				sizeof(uint64_t) * (kds_index+1) <= values_length)
			{
				uint64_t   *base = (uint64_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				kvar->u64 = base[kds_index] / 1000L -
					(POSTGRES_EPOCH_JDATE - UNIX_EPOCH_JDATE) * USECS_PER_DAY;
				*vclass = KVAR_CLASS__INLINE;
				return true;
			}
			break;

		default:
			STROM_ELOG(kcxt, "unknown unit size of Arrow::Timestamp");
			return false;
	}
	*vclass = KVAR_CLASS__NULL;
	return true;
}

STATIC_FUNCTION(bool)
__arrow_fetch_interval_datum(kern_context *kcxt,
							 const kern_data_store *kds,
							 const kern_colmeta *cmeta,
							 uint32_t kds_index,
							 kern_variable *kvar,
							 int *vclass,
							 char *slot_buf)
{
	size_t		values_length = __kds_unpack(cmeta->values_length);

	assert(cmeta->extra_offset == 0 &&
		   cmeta->extra_length == 0 &&
		   slot_buf != NULL);
	switch (cmeta->attopts.interval.unit)
	{
		case ArrowIntervalUnit__Year_Month:
			if (cmeta->values_offset &&
				sizeof(uint32_t) * (kds_index+1) <= values_length)
			{
				xpu_interval_t *iv = (xpu_interval_t *)slot_buf;
				uint32_t   *base = (uint32_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				iv->value.month = base[kds_index];
				iv->value.day   = 0;
				iv->value.time  = 0;
				*vclass = KVAR_CLASS__XPU_DATUM;
				kvar->ptr = iv;
				return true;
			}
			break;
		case ArrowIntervalUnit__Day_Time:
			if (cmeta->values_offset &&
				sizeof(uint32_t) * 2 * (kds_index+1) <= values_length)
			{
				xpu_interval_t *iv = (xpu_interval_t *)slot_buf;
				uint32_t   *base = (uint32_t *)
					((char *)kds + __kds_unpack(cmeta->values_offset));
				iv->value.month = 0;
				iv->value.day   = base[2*kds_index];
				iv->value.time  = base[2*kds_index+1];
				*vclass = KVAR_CLASS__XPU_DATUM;
				kvar->ptr = iv;
				return true;
			}
			break;
		default:
			STROM_ELOG(kcxt, "unknown unit-size of Arrow::Interval");
			return false;
	}
	*vclass = KVAR_CLASS__NULL;
	return true;
}

STATIC_FUNCTION(bool)
__arrow_fetch_fixed_size_binary_datum(kern_context *kcxt,
									  const kern_data_store *kds,
									  const kern_colmeta *cmeta,
									  uint32_t kds_index,
									  kern_variable *kvar,
									  int *vclass)
{
	unsigned int	unitsz = cmeta->attopts.fixed_size_binary.byteWidth;

	assert(cmeta->extra_offset == 0 &&
		   cmeta->extra_length == 0);
	if (cmeta->values_offset &&
		unitsz * (kds_index+1) <= __kds_unpack(cmeta->values_length))
	{
		char	   *base = ((char *)kds + __kds_unpack(cmeta->values_offset));

		kvar->ptr = base + unitsz * kds_index;
		*vclass = unitsz;
		return true;
	}
	*vclass = KVAR_CLASS__NULL;
	return true;
}

STATIC_FUNCTION(bool)
__arrow_fetch_variable_datum(kern_context *kcxt,
							 const kern_data_store *kds,
							 const kern_colmeta *cmeta,
							 uint32_t kds_index,
							 bool is_large_offset,
							 kern_variable *kvar,
							 int *vclass)
{
	uint64_t	start, end;
	char	   *extra;

	if (arrow_fetch_secondary_index(kcxt, kds, kds_index,
									cmeta->values_offset,
									cmeta->values_length,
									is_large_offset,
									&start, &end))
	{
		/* sanity checks */
		if (start > end || end - start >= 0x40000000UL ||
			end > __kds_unpack(cmeta->extra_length))
		{
			STROM_ELOG(kcxt, "Arrow variable data corruption");
			return false;
		}
		extra = (char *)kds + __kds_unpack(cmeta->extra_offset);
		kvar->ptr = extra + start;
		*vclass = (end - start);
		return true;
	}
	return false;
}

STATIC_FUNCTION(bool)
__arrow_fetch_array_datum(kern_context *kcxt,
						  const kern_data_store *kds,
						  const kern_colmeta *cmeta,
						  uint32_t kds_index,
						  bool is_large_offset,
						  kern_variable *kvar,
						  int *vclass,
						  char *slot_buf)
{
	uint64_t	start, end;

	assert(cmeta->idx_subattrs < kds->nr_colmeta &&
		   cmeta->num_subattrs == 1 &&
		   slot_buf != NULL);
	if (arrow_fetch_secondary_index(kcxt, kds, kds_index,
									cmeta->values_offset,
									cmeta->values_length,
									is_large_offset,
									&start, &end))
	{
		xpu_array_t *array = (xpu_array_t *)slot_buf;

		/* sanity checks */
		if (start > end)
		{
			STROM_ELOG(kcxt, "Arrow::List secondary index corruption");
			return false;
		}
		array->expr_ops      = &xpu_array_ops;
		array->length        = end - start;
		array->u.arrow.start = start;
		array->u.arrow.smeta = &kds->colmeta[cmeta->idx_subattrs];
		assert(cmeta->num_subattrs == 1);
		*vclass = KVAR_CLASS__XPU_DATUM;
		kvar->ptr = array;
		return true;
	}
	return false;
}

STATIC_FUNCTION(bool)
__arrow_fetch_composite_datum(kern_context *kcxt,
							  const kern_data_store *kds,
							  const kern_colmeta *cmeta,
							  uint32_t kds_index,
							  kern_variable *kvar,
							  int *vclass,
							  char *slot_buf)
{
	xpu_composite_t	*comp = (xpu_composite_t *)slot_buf;

	comp->expr_ops      = &xpu_composite_ops;
	comp->comp_typid    = cmeta->atttypid;
	comp->comp_typmod   = cmeta->atttypmod;
	comp->smeta         = &kds->colmeta[cmeta->idx_subattrs];
	comp->nfields       = cmeta->num_subattrs;
	comp->is_arrow      = true;
	comp->u.arrow.rowidx = kds_index;
	*vclass = KVAR_CLASS__XPU_DATUM;
	kvar->ptr = comp;
	return true;
}

STATIC_FUNCTION(bool)
__kern_extract_arrow_field(kern_context *kcxt,
						   const kern_data_store *kds,
						   const kern_colmeta *cmeta,
						   uint32_t kds_index,
						   kern_variable *kvar,
						   int *vclass,
						   char *slot_buf)
{
	switch (cmeta->attopts.tag)
	{
		case ArrowType__Bool:
			if (!__arrow_fetch_bool_datum(kcxt, kds, cmeta,
										  kds_index,
										  kvar, vclass))
				return false;
			break;

		case ArrowType__Int:
			if (!__arrow_fetch_int_datum(kcxt, kds, cmeta,
										 kds_index,
										 kvar, vclass))
				return false;
			break;
				
		case ArrowType__FloatingPoint:
			if (!__arrow_fetch_float_datum(kcxt, kds, cmeta,
										   kds_index,
										   kvar, vclass))
				return false;
			break;

		case ArrowType__Decimal:
			if (!__arrow_fetch_decimal_datum(kcxt, kds, cmeta,
											 kds_index,
											 kvar, vclass,
											 slot_buf))
				return false;
			break;

		case ArrowType__Date:
			if (!__arrow_fetch_date_datum(kcxt, kds, cmeta,
										  kds_index,
										  kvar, vclass))
				return false;
			break;
					
		case ArrowType__Time:
			if (!__arrow_fetch_time_datum(kcxt, kds, cmeta,
										  kds_index,
										  kvar, vclass))
				return false;
			break;

		case ArrowType__Timestamp:
			if (!__arrow_fetch_timestamp_datum(kcxt, kds, cmeta,
											   kds_index,
											   kvar, vclass))
				return false;
			break;

		case ArrowType__Interval:
			if (!__arrow_fetch_interval_datum(kcxt, kds, cmeta,
											  kds_index,
											  kvar, vclass,
											  slot_buf))
				return false;
			break;

		case ArrowType__FixedSizeBinary:
			if (!__arrow_fetch_fixed_size_binary_datum(kcxt, kds, cmeta,
													   kds_index,
													   kvar, vclass))
				return false;
			break;

		case ArrowType__Utf8:
		case ArrowType__Binary:
			if (!__arrow_fetch_variable_datum(kcxt, kds, cmeta,
											  kds_index,
											  false,
											  kvar, vclass))
				return false;
			break;

		case ArrowType__LargeUtf8:
		case ArrowType__LargeBinary:
			if (!__arrow_fetch_variable_datum(kcxt, kds, cmeta,
											  kds_index,
											  true,
											  kvar, vclass))
				return false;
			break;

		case ArrowType__List:
			if (!__arrow_fetch_array_datum(kcxt, kds, cmeta,
										   kds_index,
										   false,
										   kvar, vclass,
										   slot_buf))
				return false;
			break;

		case ArrowType__LargeList:
			if (!__arrow_fetch_array_datum(kcxt, kds, cmeta,
										   kds_index,
										   true,
										   kvar, vclass,
										   slot_buf))
				return false;
			break;

		case ArrowType__Struct:
			if (!__arrow_fetch_composite_datum(kcxt, kds, cmeta,
											   kds_index,
											   kvar, vclass,
											   slot_buf))
				return false;
			break;
		default:
			STROM_ELOG(kcxt, "Unsupported Apache Arrow type");
			return false;
	}
	return true;
}

INLINE_FUNCTION(bool)
__extract_arrow_tuple_sysattr(kern_context *kcxt,
							  const kern_data_store *kds,
							  uint32_t kds_index,
							  const kern_varload_desc *vl_desc)
{
	uint64_t	dummy;

	switch (vl_desc->vl_resno)
	{
		case SelfItemPointerAttributeNumber:
			dummy = 0;
			break;
		case MinTransactionIdAttributeNumber:
			dummy = FrozenTransactionId;
			break;
		case MaxTransactionIdAttributeNumber:
			dummy = InvalidTransactionId;
			break;
		case  MinCommandIdAttributeNumber:
		case MaxCommandIdAttributeNumber:
			dummy = FirstCommandId;
			break;
		case TableOidAttributeNumber:
			dummy = kds->table_oid;
			break;
		default:
			STROM_ELOG(kcxt, "not a supported system attribute reference");
			return false;
	}
	return __extract_heap_tuple_attr(kcxt, vl_desc, (char *)&dummy);
}

STATIC_FUNCTION(bool)
kern_extract_arrow_tuple(kern_context *kcxt,
						 kern_data_store *kds,
						 uint32_t kds_index,
						 const kern_varload_desc *vl_desc,
						 int vload_nitems)
{
	int		vload_count = 0;

	assert(kds->format == KDS_FORMAT_ARROW);
	/* fillup invalid values for system attribute, if any */
	while (vload_count < vload_nitems &&
		   vl_desc->vl_resno < 0)
	{
		if (!__extract_arrow_tuple_sysattr(kcxt, kds, kds_index, vl_desc))
			return false;
		vl_desc++;
		vload_count++;
	}

	while (vload_count < vload_nitems &&
		   vl_desc->vl_resno <= kds->ncols)
	{
		kern_colmeta *cmeta = &kds->colmeta[vl_desc->vl_resno-1];
		uint32_t	slot_id = vl_desc->vl_slot_id;

		if (cmeta->nullmap_offset == 0 ||
			arrow_bitmap_check(kds, kds_index,
							   cmeta->nullmap_offset,
							   cmeta->nullmap_length))
		{
			if (!__kern_extract_arrow_field(kcxt,
											kds,
											cmeta,
											kds_index,
											&kcxt->kvars_slot[slot_id],
											&kcxt->kvars_class[slot_id],
											NULL)) //slot_buf))
				return false;
		}
		else
		{
			if (!__extract_heap_tuple_attr(kcxt, vl_desc, NULL))
				return false;
		}
		vl_desc++;
		vload_count++;
	}
	/* other fields, which refers out of range, are NULL */
	return __fillup_by_null_values(kcxt, vl_desc, vload_nitems - vload_count);
}

/* ----------------------------------------------------------------
 *
 * Device-side Expression Support Routines
 *
 * ----------------------------------------------------------------
 */

/*
 * Const Expression
 */
STATIC_FUNCTION(bool)
pgfn_ConstExpr(XPU_PGFUNCTION_ARGS)
{
	const xpu_datum_operators *expr_ops = kexp->expr_ops;
	const char	   *addr;

	addr = (kexp->u.c.const_isnull ? NULL : kexp->u.c.const_value);

	return expr_ops->xpu_datum_heap_ref(kcxt, addr, __result);
}

STATIC_FUNCTION(bool)
pgfn_ParamExpr(XPU_PGFUNCTION_ARGS)
{
	const xpu_datum_operators *expr_ops = kexp->expr_ops;
	kern_session_info *session = kcxt->session;
	uint32_t		param_id = kexp->u.p.param_id;
	const char	   *addr;

	if (param_id < session->nparams && session->poffset[param_id] != 0)
		addr = ((char *)session + session->poffset[param_id]);
	else
		addr = NULL;

	return expr_ops->xpu_datum_heap_ref(kcxt, addr, __result);
}

STATIC_FUNCTION(bool)
pgfn_VarExpr(XPU_PGFUNCTION_ARGS)
{
	const xpu_datum_operators *expr_ops = kexp->expr_ops;

	if (kexp->u.v.var_offset < 0)
	{
		uint16_t		slot_id = kexp->u.v.var_slot_id;
		xpu_datum_t	   *__xdatum = kcxt->kvars_values[slot_id];

		if (slot_id >= kcxt->kvars_nslots)
		{
			STROM_ELOG(kcxt, "Bug? slot_id is out of range");
			return false;
		}
		memcpy(__result, __xdatum, expr_ops->xpu_type_sizeof);
		assert(XPU_DATUM_ISNULL(__result) || __result->expr_ops == expr_ops);
	}
	else
	{
		const kvec_datum_t *kvecs = (kvec_datum_t *)(kcxt->kvecs_curr_buffer +
													 kexp->u.v.var_offset);
		return expr_ops->xpu_datum_kvec_ref(kcxt, kvecs,
											kcxt->kvecs_curr_id,
											__result);
	}
	return true;
}

STATIC_FUNCTION(bool)
pgfn_BoolExprAnd(XPU_PGFUNCTION_ARGS)
{
	xpu_bool_t *result = (xpu_bool_t *)__result;
	int			i;
	bool		anynull = false;
	const kern_expression *karg;

	assert(kexp->exptype == TypeOpCode__bool &&
		   kexp->nr_args >= 2);
	for (i=0, karg=KEXP_FIRST_ARG(kexp);
		 i < kexp->nr_args;
		 i++, karg=KEXP_NEXT_ARG(karg))
	{
		xpu_bool_t	status;

		assert(KEXP_IS_VALID(karg, bool));
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &status))
			return false;
		if (XPU_DATUM_ISNULL(&status))
			anynull = true;
		else if (!status.value)
		{
			result->expr_ops = &xpu_bool_ops;
			result->value  = false;
			return true;
		}
	}
	result->expr_ops = (anynull ? NULL : &xpu_bool_ops);
	result->value  = true;
	return true;
}

STATIC_FUNCTION(bool)
pgfn_BoolExprOr(XPU_PGFUNCTION_ARGS)
{
	xpu_bool_t *result = (xpu_bool_t *)__result;
	int			i;
	bool		anynull = false;
	const kern_expression *karg;

	assert(kexp->exptype == TypeOpCode__bool &&
		   kexp->nr_args >= 2);
	for (i=0, karg=KEXP_FIRST_ARG(kexp);
		 i < kexp->nr_args;
		 i++, karg=KEXP_NEXT_ARG(karg))
	{
		xpu_bool_t	status;

		assert(KEXP_IS_VALID(karg, bool));
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &status))
			return false;
		if (XPU_DATUM_ISNULL(&status))
			anynull = true;
		else if (status.value)
		{
			result->expr_ops = &xpu_bool_ops;
			result->value = true;
			return true;
		}
	}
	result->expr_ops = (anynull ? NULL : &xpu_bool_ops);
	result->value  = false;
	return true;
}

STATIC_FUNCTION(bool)
pgfn_BoolExprNot(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(bool, bool, status);

	if (XPU_DATUM_ISNULL(&status))
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = kexp->expr_ops;
		result->value = !status.value;
	}
	return true;
}

STATIC_FUNCTION(bool)
pgfn_NullTestExpr(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(bool, bool, status);

	result->expr_ops = &xpu_bool_ops;
	switch (kexp->opcode)
	{
		case FuncOpCode__NullTestExpr_IsNull:
			result->value = XPU_DATUM_ISNULL(&status);
			break;
		case FuncOpCode__NullTestExpr_IsNotNull:
			result->value = !XPU_DATUM_ISNULL(&status);
			break;
		default:
			STROM_ELOG(kcxt, "corrupted kernel expression");
			return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
pgfn_BoolTestExpr(XPU_PGFUNCTION_ARGS)
{
	KEXP_PROCESS_ARGS1(bool, bool, status);

	result->expr_ops = kexp->expr_ops;
	switch (kexp->opcode)
	{
		case FuncOpCode__BoolTestExpr_IsTrue:
			result->value = (!XPU_DATUM_ISNULL(&status) && status.value);
			break;
		case FuncOpCode__BoolTestExpr_IsNotTrue:
			result->value = (XPU_DATUM_ISNULL(&status) || !status.value);
			break;
		case FuncOpCode__BoolTestExpr_IsFalse:
			result->value = (!XPU_DATUM_ISNULL(&status) && !status.value);
			break;
		case FuncOpCode__BoolTestExpr_IsNotFalse:
			result->value = (XPU_DATUM_ISNULL(&status) || status.value);
			break;
		case FuncOpCode__BoolTestExpr_IsUnknown:
			result->value = XPU_DATUM_ISNULL(&status);
			break;
		case FuncOpCode__BoolTestExpr_IsNotUnknown:
			result->value = !XPU_DATUM_ISNULL(&status);
			break;
		default:
			STROM_ELOG(kcxt, "corrupted kernel expression");
			return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
pgfn_DistinctFrom(XPU_PGFUNCTION_ARGS)
{
	xpu_bool_t	   *result = (xpu_bool_t *)__result;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);
	const kern_expression *subarg1;
	const kern_expression *subarg2;
	xpu_datum_t	   *subbuf1;
	xpu_datum_t	   *subbuf2;

	assert(kexp->exptype == TypeOpCode__bool &&
		   kexp->nr_args == 1 &&
		   KEXP_IS_VALID(karg, bool) &&
		   karg->nr_args == 2);
	subarg1 = KEXP_FIRST_ARG(karg);
	assert(__KEXP_IS_VALID(karg, subarg1));
	subbuf1 = (xpu_datum_t *)alloca(subarg1->expr_ops->xpu_type_sizeof);
	if (!EXEC_KERN_EXPRESSION(kcxt, subarg1, subbuf1))
		return false;

	subarg2 = KEXP_NEXT_ARG(subarg1);
	assert(__KEXP_IS_VALID(karg, subarg2));
	subbuf2 = (xpu_datum_t *)alloca(subarg2->expr_ops->xpu_type_sizeof);
	if (!EXEC_KERN_EXPRESSION(kcxt, subarg2, subbuf2))
		return false;

	if (XPU_DATUM_ISNULL(subbuf1) && XPU_DATUM_ISNULL(subbuf2))
	{
		/* Both NULL? Then is not distinct... */
		result->expr_ops = &xpu_bool_ops;
		result->value = false;
	}
	else if (XPU_DATUM_ISNULL(subbuf1) || XPU_DATUM_ISNULL(subbuf2))
	{
		/* Only one is NULL? Then is distinct... */
		result->expr_ops = &xpu_bool_ops;
		result->value = true;
	}
	else if (EXEC_KERN_EXPRESSION(kcxt, karg, __result))
	{
		assert(result->expr_ops == &xpu_bool_ops);
		result->value = !result->value;
	}
	else
	{
		return false;
	}
	return true;
}

STATIC_FUNCTION(bool)
pgfn_CoalesceExpr(XPU_PGFUNCTION_ARGS)
{
	const kern_expression *karg;
	int		i;

	for (i=0, karg = KEXP_FIRST_ARG(kexp);
		 i < kexp->nr_args;
		 i++, karg = KEXP_NEXT_ARG(karg))
	{
		assert(__KEXP_IS_VALID(kexp, karg) && kexp->exptype == karg->exptype);
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, __result))
			return false;
		if (!XPU_DATUM_ISNULL(__result))
			return true;
	}
	__result->expr_ops = NULL;
	return true;
}

STATIC_FUNCTION(bool)
pgfn_LeastExpr(XPU_PGFUNCTION_ARGS)
{
	const xpu_datum_operators *kexp_ops = kexp->expr_ops;
	const kern_expression *karg;
	xpu_datum_t	   *temp;
	int				comp;
	int				i, sz = kexp_ops->xpu_type_sizeof;

	temp = (xpu_datum_t *)alloca(sz);
	memset(temp, 0, sz);
	__result->expr_ops = NULL;
	for (i=0,  karg = KEXP_FIRST_ARG(kexp);
		 i < kexp->nr_args;
		 i++, karg = KEXP_NEXT_ARG(karg))
	{
		assert(__KEXP_IS_VALID(kexp, karg) && kexp->exptype == karg->exptype);
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, temp))
			return false;
		if (XPU_DATUM_ISNULL(temp))
			continue;

		if (XPU_DATUM_ISNULL(__result))
		{
			memcpy(__result, temp, sz);
		}
		else
		{
			if (!kexp_ops->xpu_datum_comp(kcxt, &comp, __result, temp))
				return false;
			if (comp > 0)
				memcpy(__result, temp, sz);
		}
	}
	return true;
}

STATIC_FUNCTION(bool)
pgfn_GreatestExpr(XPU_PGFUNCTION_ARGS)
{
	const xpu_datum_operators *kexp_ops = kexp->expr_ops;
	const kern_expression *karg;
	xpu_datum_t	   *temp;
	int				comp;
	int				i, sz = kexp_ops->xpu_type_sizeof;

	temp = (xpu_datum_t *)alloca(sz);
	memset(temp, 0, sz);
	__result->expr_ops = NULL;
	for (i=0,  karg = KEXP_FIRST_ARG(kexp);
		 i < kexp->nr_args;
		 i++, karg = KEXP_NEXT_ARG(karg))
	{
		assert(__KEXP_IS_VALID(kexp, karg) && kexp->exptype == karg->exptype);
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, temp))
			return false;
		if (XPU_DATUM_ISNULL(temp))
			continue;

		if (XPU_DATUM_ISNULL(__result))
		{
			memcpy(__result, temp, sz);
		}
		else
		{
			if (!kexp_ops->xpu_datum_comp(kcxt, &comp, __result, temp))
				return false;
			if (comp < 0)
				memcpy(__result, temp, sz);
		}
	}
	return true;
}

STATIC_FUNCTION(bool)
pgfn_CaseWhenExpr(XPU_PGFUNCTION_ARGS)
{
	const kern_expression *karg;
	xpu_datum_t	   *comp = NULL;
	xpu_datum_t	   *temp = NULL;
	int				i, temp_sz = 0;

	/* CASE <key> expression, if any */
	if (kexp->u.casewhen.case_comp)
	{
		karg = (const kern_expression *)
			((char *)kexp + kexp->u.casewhen.case_comp);
		assert(__KEXP_IS_VALID(kexp, karg));
		comp = (xpu_datum_t *)alloca(karg->expr_ops->xpu_type_sizeof);
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, comp))
			return false;
	}

	/* evaluate each WHEN-clauses */
	assert((kexp->nr_args % 2) == 0);
	for (i = 0, karg=KEXP_FIRST_ARG(kexp);
		 i < kexp->nr_args;
		 i += 2, karg=KEXP_NEXT_ARG(karg))
	{
		bool		matched = false;

		assert(__KEXP_IS_VALID(kexp, karg));
		if (comp)
		{
			int			status;

			if (temp_sz < karg->expr_ops->xpu_type_sizeof)
			{
				temp_sz = karg->expr_ops->xpu_type_sizeof + 32;
				temp = (xpu_datum_t *)alloca(temp_sz);
			}
			if (!EXEC_KERN_EXPRESSION(kcxt, karg, temp))
				return false;
			if (!karg->expr_ops->xpu_datum_comp(kcxt, &status, comp, temp))
				return false;
			if (status == 0)
				matched = true;
		}
		else
		{
			xpu_bool_t	status;

			if (!EXEC_KERN_EXPRESSION(kcxt, karg, &status))
				return false;
			if (!XPU_DATUM_ISNULL(&status) && status.value)
				matched = true;
		}

		karg = KEXP_NEXT_ARG(karg);
		assert(__KEXP_IS_VALID(kexp, karg));
		if (matched)
		{
			assert(kexp->exptype == karg->exptype);
			if (!EXEC_KERN_EXPRESSION(kcxt, karg, __result))
				return false;
			/* OK */
			return true;
		}
	}

	/* ELSE clause, if any */
	if (kexp->u.casewhen.case_else == 0)
		__result->expr_ops = NULL;
	else
	{
		karg = (const kern_expression *)
			((char *)kexp + kexp->u.casewhen.case_else);
		assert(__KEXP_IS_VALID(kexp, karg) && kexp->exptype == karg->exptype);
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, __result))
			return false;
	}
	return true;
}

/*
 * ScalarArrayOpExpr
 */
STATIC_FUNCTION(bool)
__ScalarArrayOpHeap(kern_context *kcxt,
					xpu_bool_t *result,
					const kern_expression *kexp,
					const kern_expression *kcmp,
					xpu_array_t *aval)
{
	const __ArrayTypeData *ar = (const __ArrayTypeData *)VARDATA_ANY(aval->u.heap.value);
	const kern_varload_desc *vl_desc = &kexp->u.saop.elem_desc;
	uint8_t	   *nullmap = __pg_array_nullmap(ar);
	char	   *base = __pg_array_dataptr(ar);
	int			ndim = __pg_array_ndim(ar);
	uint32_t	nitems = 0;
	uint32_t	offset = 0;
	bool		use_any = (kexp->opcode == FuncOpCode__ScalarArrayOpAny);
	bool		meet_nulls = false;

	/* determine the number of items */
	if (ndim > 0)
	{
		nitems = __pg_array_dim(ar, 0);
		for (int k=1; k < ndim; k++)
			nitems *= __pg_array_dim(ar,k);
	}
	/* walk on the array */
	for (uint32_t i=0; i < nitems; i++)
	{
		xpu_bool_t	status;
		const char *addr;

		/* datum reference */
		if (nullmap && att_isnull(i, nullmap))
		{
			addr = NULL;
		}
		else
		{
			if (vl_desc->vl_typlen > 0)
				offset = TYPEALIGN(vl_desc->vl_typalign, offset);
			else if (!VARATT_NOT_PAD_BYTE(base + offset))
				offset = TYPEALIGN(vl_desc->vl_typalign, offset);
			addr = base + offset;

			if (vl_desc->vl_typlen > 0)
				offset += vl_desc->vl_typlen;
			else if (vl_desc->vl_typlen == -1)
				offset += VARSIZE_ANY(addr);
			else
			{
				STROM_ELOG(kcxt, "not a supported attribute length");
				return false;
			}
		}
		/* load the array element */
		if (!__extract_heap_tuple_attr(kcxt, vl_desc, addr))
			return false;
		/* call the comparator */
		if (!EXEC_KERN_EXPRESSION(kcxt, kcmp, &status))
			return false;
		if (!XPU_DATUM_ISNULL(&status))
		{
			if (use_any)
			{
                if (status.value)
                {
					result->expr_ops = &xpu_bool_ops;
                    result->value = true;
					return true;
                }
            }
            else
            {
                if (!status.value)
                {
					result->expr_ops = &xpu_bool_ops;
                    result->value = false;
                    break;
                }
            }
        }
		else
		{
			meet_nulls = true;
		}
	}

	if (meet_nulls)
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_bool_ops;
		result->value = !use_any;
	}
	return true;
}

STATIC_FUNCTION(bool)
__ScalarArrayOpArrow(kern_context *kcxt,
					 xpu_bool_t *result,
					 const kern_expression *kexp,
					 const kern_expression *kcmp,
					 xpu_array_t *aval)
{
	const kern_varload_desc *vl_desc = &kexp->u.saop.elem_desc;
	const kern_colmeta *smeta = aval->u.arrow.smeta;
	const kern_data_store *kds;
	bool		use_any = (kexp->opcode == FuncOpCode__ScalarArrayOpAny);
	bool		meet_nulls = false;

	result->value = !use_any;
	kds = (const kern_data_store *)
		((char *)smeta - smeta->kds_offset);
	for (int k=0; k < aval->length; k++)
	{
		uint32_t	index = aval->u.arrow.start + k;
		xpu_bool_t	status;

		/* load the element */
		if (smeta->nullmap_offset == 0 ||
			arrow_bitmap_check(kds, index,
							   smeta->nullmap_offset,
							   smeta->nullmap_length))
		{
			if (!__kern_extract_arrow_field(kcxt,
											kds,
											smeta,
											index,
											&kcxt->kvars_slot[12345], //FIXME
											&kcxt->kvars_class[12345], //FIXME
											NULL))
				return false;
		}
		else
		{
			if (!__extract_heap_tuple_attr(kcxt, vl_desc, NULL))
				return false;
		}
		/* call the comparator */
		if (!EXEC_KERN_EXPRESSION(kcxt, kcmp, &status))
			return false;
		if (!XPU_DATUM_ISNULL(&status))
		{
			if (use_any)
			{
				if (status.value)
				{
					result->expr_ops = &xpu_bool_ops;
					result->value = true;
					break;
				}
			}
			else
			{
				if (!status.value)
				{
					result->expr_ops = &xpu_bool_ops;
					result->value = false;
					break;
				}
			}
		}
		else
		{
			meet_nulls = true;
		}
	}

	if (meet_nulls)
		result->expr_ops = NULL;
	else
	{
		result->expr_ops = &xpu_bool_ops;
		result->value = !use_any;
	}
	return true;
}

STATIC_FUNCTION(bool)
pgfn_ScalarArrayOp(XPU_PGFUNCTION_ARGS)
{
	xpu_bool_t	   *result = (xpu_bool_t *)__result;
	xpu_array_t		aval;
	const kern_expression *karg;

	assert(kexp->exptype == TypeOpCode__bool &&
		   kexp->nr_args == 2 &&
		   kexp->u.saop.elem_desc.vl_slot_id < kcxt->kvars_nslots);
	memset(result, 0, sizeof(xpu_bool_t));

	/* fetch array value */
	karg = KEXP_FIRST_ARG(kexp);
	assert(KEXP_IS_VALID(karg, array));
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, &aval))
		return false;
	/* comparator expression */
	karg = KEXP_NEXT_ARG(karg);
	assert(KEXP_IS_VALID(karg, bool));
	if (aval.length < 0)
	{
		if (!__ScalarArrayOpHeap(kcxt, result, kexp, karg, &aval))
			return false;
	}
	else
	{
		if (!__ScalarArrayOpArrow(kcxt, result, kexp, karg, &aval))
			return false;
	}
	return true;
}

/* ----------------------------------------------------------------
 *
 * Routines to support Projection
 *
 * ----------------------------------------------------------------
 */
PUBLIC_FUNCTION(int)
kern_form_heaptuple(kern_context *kcxt,
					const kern_expression *kexp_proj,
					const kern_data_store *kds_dst,
					HeapTupleHeaderData *htup)
{
	/*
	 * NOTE: the caller must call kern_estimate_heaptuple() preliminary to ensure
	 *       kcxt->kvars_values[] are filled-up by the scalar values to be written.
	 */
	uint32_t	t_hoff;
	uint32_t	t_next;
	uint16_t	t_infomask = 0;
	bool		t_hasnull = false;
	int			nattrs = kexp_proj->u.proj.nattrs;

	if (kds_dst && kds_dst->ncols < nattrs)
		nattrs = kds_dst->ncols;
	/* has any NULL attributes? */
	for (int j=0; j < nattrs; j++)
	{
		uint32_t	slot_id = kexp_proj->u.proj.desc[j].proj_slot_id;

		assert(slot_id < kcxt->kvars_nslots);
		if (XPU_DATUM_ISNULL(kcxt->kvars_values[slot_id]))
		{
			t_infomask |= HEAP_HASNULL;
			t_hasnull = true;
			break;
		}
	}

	/* set up headers */
	t_hoff = offsetof(HeapTupleHeaderData, t_bits);
	if (t_hasnull)
		t_hoff += BITMAPLEN(nattrs);
	t_hoff = MAXALIGN(t_hoff);

	if (htup)
	{
		memset(htup, 0, t_hoff);
		htup->t_choice.t_datum.datum_typmod = kds_dst->tdtypmod;
		htup->t_choice.t_datum.datum_typeid = kds_dst->tdtypeid;
		htup->t_ctid.ip_blkid.bi_hi = 0xffff;	/* InvalidBlockNumber */
		htup->t_ctid.ip_blkid.bi_lo = 0xffff;
		htup->t_ctid.ip_posid = 0;				/* InvalidOffsetNumber */
		htup->t_infomask2 = (nattrs & HEAP_NATTS_MASK);
		htup->t_hoff = t_hoff;
	}

	/* walk on the columns */
	for (int j=0; j < nattrs; j++)
	{
		const kern_colmeta *cmeta = &kds_dst->colmeta[j];
		const kern_projection_desc *pj_desc = &kexp_proj->u.proj.desc[j];
		xpu_datum_t	   *xdatum;
		int				nbytes;
		char		   *buffer = NULL;

		assert(pj_desc->proj_slot_id < kcxt->kvars_nslots);
		xdatum = kcxt->kvars_values[pj_desc->proj_slot_id];
		if (XPU_DATUM_ISNULL(xdatum))
			continue;
		/* adjust alignment */
		t_next = TYPEALIGN(cmeta->attalign, t_hoff);
		if (htup)
		{
			if (t_next > t_hoff)
				memset((char *)htup + t_hoff, 0, t_next - t_hoff);
			buffer = (char *)htup + t_next;
		}
		/* write-out or estimate length */
		nbytes = xdatum->expr_ops->xpu_datum_write(kcxt,
												   buffer,
												   cmeta,
												   xdatum);
		if (nbytes < 0)
			return -1;
		/* update t_infomask if varlena */
		if (cmeta->attlen == -1)
		{
			if (buffer && VARATT_IS_EXTERNAL(buffer))
				t_infomask |= HEAP_HASEXTERNAL;
			t_infomask |= HEAP_HASVARWIDTH;
		}
		/* set not-null bit, if valid */
		if (htup && t_hasnull)
			htup->t_bits[j>>3] |= (1<<(j & 7));
		t_hoff = t_next + nbytes;
	}

	if (htup)
	{
		htup->t_infomask = t_infomask;
		SET_VARSIZE(&htup->t_choice.t_datum, t_hoff);
	}
	return t_hoff;	
}

EXTERN_FUNCTION(int)
kern_estimate_heaptuple(kern_context *kcxt,
                        const kern_expression *kexp_proj,
                        const kern_data_store *kds_dst)
{
	const kern_expression *karg;
	int		i, sz;

	/* Run SaveExpr expressions to warm up kcxt->kvars_values[] */
	for (i=0, karg=KEXP_FIRST_ARG(kexp_proj);
		 i < kexp_proj->nr_args;
		 i++, karg=KEXP_NEXT_ARG(karg))
	{
		xpu_datum_t	   *xdatum;

		assert(__KEXP_IS_VALID(kexp_proj, karg) &&
			   karg->opcode == FuncOpCode__SaveExpr &&
			   karg->u.save.sv_slot_id < kcxt->kvars_nslots);
		xdatum = kcxt->kvars_values[karg->u.save.sv_slot_id];
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, xdatum))
			return -1;
	}
	/* then, estimate the length */
	sz = kern_form_heaptuple(kcxt, kexp_proj, kds_dst, NULL);
	if (sz < 0)
		return -1;
	return MAXALIGN(offsetof(kern_tupitem, htup) + sz);
}

STATIC_FUNCTION(bool)
pgfn_Projection(XPU_PGFUNCTION_ARGS)
{
	/*
	 * FuncOpExpr_Projection should be handled by kern_estimate_heaptuple()
	 * and kern_form_heaptuple() by the caller.
	 */
	STROM_ELOG(kcxt, "pgfn_Projection is not implemented");
	return false;
}

STATIC_FUNCTION(bool)
pgfn_HashValue(XPU_PGFUNCTION_ARGS)
{
	const kern_expression *karg;
	xpu_int4_t	   *result = (xpu_int4_t *)__result;
	xpu_datum_t	   *datum = (xpu_datum_t *)alloca(64);
	int				i, datum_sz = 64;
	uint32_t		hash = 0xffffffffU;

	for (i=0, karg = KEXP_FIRST_ARG(kexp);
		 i < kexp->nr_args;
		 i++, karg = KEXP_NEXT_ARG(karg))
	{
		const xpu_datum_operators *expr_ops = karg->expr_ops;
		uint32_t	__hash;

		if (expr_ops->xpu_type_sizeof > datum_sz)
		{
			datum_sz = expr_ops->xpu_type_sizeof;
			datum = (xpu_datum_t *)alloca(datum_sz);
		}
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, datum))
			return false;
		if (!XPU_DATUM_ISNULL(datum))
		{
			if (!expr_ops->xpu_datum_hash(kcxt, &__hash, datum))
				return false;
			hash ^= __hash;
		}
	}
	hash ^= 0xffffffffU;

	result->expr_ops = &xpu_int4_ops;
	result->value = hash;
	return true;
}

STATIC_FUNCTION(bool)
pgfn_SaveExpr(XPU_PGFUNCTION_ARGS)
{
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);
	const xpu_datum_operators *expr_ops = kexp->expr_ops;
	uint16_t		slot_id = kexp->u.save.sv_slot_id;
	xpu_datum_t	   *xdatum = kcxt->kvars_values[slot_id];

	assert(kexp->nr_args == 1 &&
		   kexp->exptype == karg->exptype &&
		   slot_id < kcxt->kvars_nslots);
	/* Run the expression */
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, xdatum))
		return false;
	/*
	 * NOTE: SaveExpr accepts NULL result buffer, if caller just wants to fill-up
	 * the kvars-slot at the time.
	 */
	if (__result)
	{
		if (XPU_DATUM_ISNULL(xdatum))
			__result->expr_ops = NULL;
		else if (xdatum != __result)
			memcpy(__result, xdatum, expr_ops->xpu_type_sizeof);
	}
	return true;
}

STATIC_FUNCTION(bool)
pgfn_JoinQuals(XPU_PGFUNCTION_ARGS)
{
	const kern_expression *karg;
	xpu_int4_t *result = (xpu_int4_t *)__result;
	int			i, status = 1;

	assert(kexp->exptype == TypeOpCode__bool);
	for (i=0, karg = KEXP_FIRST_ARG(kexp);
		 i < kexp->nr_args;
		 i++, karg = KEXP_NEXT_ARG(karg))
	{
		xpu_bool_t	datum;

		if (status < 0 && (karg->expflags & KEXP_FLAG__IS_PUSHED_DOWN) != 0)
			continue;
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum))
			return false;
		if (XPU_DATUM_ISNULL(&datum) || !datum.value)
		{
			/*
			 * NOTE: Even if JoinQual returns 'unmatched' status, we need
			 * to check whether the pure JOIN ... ON clause is satisfied,
			 * or not, if OUTER JOIN case.
			 * '-1' means JoinQual is not matched, because of the pushed-
			 * down qualifiers from WHERE-clause, not JOIN ... ON.
			 */
			if ((karg->expflags & KEXP_FLAG__IS_PUSHED_DOWN) == 0)
			{
				status = 0;
				break;
			}
			status = -1;
		}
	}
	result->expr_ops = kexp->expr_ops;
	result->value = status;
	return true;
}

STATIC_FUNCTION(bool)
pgfn_GiSTEval(XPU_PGFUNCTION_ARGS)
{
	STROM_ELOG(kcxt, "pgfn_GiSTEval should not be called as a normal kernel expression");
	return false;
}

STATIC_FUNCTION(bool)
pgfn_Packed(XPU_PGFUNCTION_ARGS)
{
	STROM_ELOG(kcxt, "pgfn_Packed should not be called as a normal kernel expression");
	return false;
}

STATIC_FUNCTION(bool)
pgfn_AggFuncs(XPU_PGFUNCTION_ARGS)
{
	STROM_ELOG(kcxt, "pgfn_AggFuncs should not be called as a normal kernel expression");
	return false;
}

/* ------------------------------------------------------------
 *
 * Extract GpuCache tuples
 *
 * ------------------------------------------------------------
 */
INLINE_FUNCTION(bool)
__extract_gpucache_tuple_sysattr(kern_context *kcxt,
								 const kern_data_store *kds,
								 uint32_t kds_index,
								 const kern_varload_desc *vl_desc)
{
	const kern_colmeta *cmeta = &kds->colmeta[kds->nr_colmeta - 1];
	GpuCacheSysattr	   *sysatt;
	const char		   *addr;
	uint32_t			slot_id = vl_desc->vl_slot_id;
	uint32_t			temp;

	assert(!cmeta->attbyval &&
		   cmeta->attlen == sizeof(GpuCacheSysattr) &&
		   cmeta->nullmap_offset == 0);		/* NOT NULL */
	sysatt = (GpuCacheSysattr *)
		((char *)kds + __kds_unpack(cmeta->values_offset));
	switch (vl_desc->vl_resno)
	{
		case SelfItemPointerAttributeNumber:
			addr = (const char *)&sysatt[kds_index].ctid;
			break;
		case MinTransactionIdAttributeNumber:
			addr = (const char *)&sysatt[kds_index].xmin;
			break;
		case MaxTransactionIdAttributeNumber:
			addr = (const char *)&sysatt[kds_index].xmax;
			break;
		case MinCommandIdAttributeNumber:
		case MaxCommandIdAttributeNumber:
			temp = FirstCommandId;
			addr = (const char *)&temp;
			break;
		case TableOidAttributeNumber:
			addr = (const char *)&kds->table_oid;
			kcxt->kvars_class[slot_id] = KVAR_CLASS__INLINE;
			break;
		default:
			STROM_ELOG(kcxt, "not a supported system attribute reference");
			return false;
	}
	return __extract_heap_tuple_attr(kcxt, vl_desc, addr);
}

STATIC_FUNCTION(bool)
kern_extract_gpucache_tuple(kern_context *kcxt,
							const kern_data_store *kds,
							const kern_data_extra *extra,
							uint32_t kds_index,
							const kern_varload_desc *vl_desc,
							int vload_nitems)
{
	int		vload_count = 0;

	assert(kds->format == KDS_FORMAT_COLUMN);
	/* out of range? */
	if (kds_index >= kds->nitems)
		goto bailout;
	/* fillup values for system attribute, if any */
	while (vload_count < vload_nitems &&
		   vl_desc->vl_resno < 0)
	{
		if (!__extract_gpucache_tuple_sysattr(kcxt, kds, kds_index, vl_desc))
			return false;
		vl_desc++;
		vload_count++;
	}

	while (vload_count < vload_nitems &&
		   vl_desc->vl_resno <= kds->ncols)
	{
		const kern_colmeta *cmeta = &kds->colmeta[vl_desc->vl_resno-1];
		const char *addr;
		uint32_t	slot_id = vl_desc->vl_slot_id;

		assert(slot_id < kcxt->kvars_nslots);
		if (!KDS_COLUMN_ITEM_ISNULL(kds, cmeta, kds_index))
		{
			/* base pointer */
			addr = ((const char *)kds + __kds_unpack(cmeta->values_offset));

			if (cmeta->attlen > 0)
			{
				addr += cmeta->attlen * kds_index;
			}
			else
			{
				addr = ((char *)extra +
						__kds_unpack(((uint32_t *)addr)[kds_index]));
			}
		}
		else
		{
			addr = NULL;
		}

		if (!__extract_heap_tuple_attr(kcxt, vl_desc, addr))
			return false;
		vl_desc++;
		vload_count++;
	}
bailout:
	/* other fields, which refers out of range, are NULL */
	return __fillup_by_null_values(kcxt, vl_desc, vload_nitems - vload_count);
}

STATIC_FUNCTION(bool)
pgfn_LoadVars(XPU_PGFUNCTION_ARGS)
{
	STROM_ELOG(kcxt, "Bug? LoadVars shall not be called as a part of expression");
	return false;
}

PUBLIC_FUNCTION(bool)
ExecLoadVarsHeapTuple(kern_context *kcxt,
					  const kern_expression *kexp,
					  int depth,
					  const kern_data_store *kds,
					  const HeapTupleHeaderData *htup)	/* htup may be NULL */
{
	if (kexp)
	{
		assert(kexp->opcode == FuncOpCode__LoadVars &&
			   kexp->exptype == TypeOpCode__int4 &&
			   kexp->nr_args == 0 &&
			   kexp->u.load.depth == depth);
		if (htup)
		{
			if (!kern_extract_heap_tuple(kcxt,
										 kds,
										 htup,
										 kexp->u.load.desc,
										 kexp->u.load.nitems))
				return false;
		}
		else
		{
			if (!__fillup_by_null_values(kcxt,
										 kexp->u.load.desc,
										 kexp->u.load.nitems))
				return false;
		}
	}
	return true;
}

PUBLIC_FUNCTION(bool)
ExecLoadVarsOuterRow(kern_context *kcxt,
					 kern_expression *kexp_load_vars,
					 kern_expression *kexp_scan_quals,
					 kern_data_store *kds,
					 HeapTupleHeaderData *htup)
{
	/* load the one tuple */
	ExecLoadVarsHeapTuple(kcxt, kexp_load_vars, 0, kds, htup);
	/* check scan quals if given */
	if (kexp_scan_quals)
	{
		xpu_bool_t	retval;

		if (EXEC_KERN_EXPRESSION(kcxt, kexp_scan_quals, &retval))
		{
			if (!XPU_DATUM_ISNULL(&retval) && retval.value)
				return true;
		}
		else
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
		}
		return false;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
ExecLoadVarsOuterArrow(kern_context *kcxt,
					   kern_expression *kexp_load_vars,
					   kern_expression *kexp_scan_quals,
					   kern_data_store *kds,
					   uint32_t kds_index)
{
	if (kexp_load_vars)
	{
		assert(kexp_load_vars->opcode == FuncOpCode__LoadVars &&
			   kexp_load_vars->exptype == TypeOpCode__int4 &&
			   kexp_load_vars->nr_args == 0 &&
			   kexp_load_vars->u.load.depth == 0);
		if (!kern_extract_arrow_tuple(kcxt,
									  kds,
									  kds_index,
									  kexp_load_vars->u.load.desc,
									  kexp_load_vars->u.load.nitems))
			return false;
	}
	/* check scan quals if given */
	if (kexp_scan_quals)
	{
		xpu_bool_t	retval;

		if (EXEC_KERN_EXPRESSION(kcxt, kexp_scan_quals, &retval))
		{
			if (!XPU_DATUM_ISNULL(&retval) && retval.value)
				return true;
		}
		else
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
		}
		return false;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
ExecLoadVarsOuterColumn(kern_context *kcxt,
						kern_expression *kexp_load_vars,
						kern_expression *kexp_scan_quals,
						kern_data_store *kds,
						kern_data_extra *extra,
						uint32_t kds_index)
{
	if (kexp_load_vars)
	{
		assert(kexp_load_vars->opcode == FuncOpCode__LoadVars &&
			   kexp_load_vars->exptype == TypeOpCode__int4 &&
			   kexp_load_vars->nr_args == 0 &&
			   kexp_load_vars->u.load.depth == 0);
		if (!kern_extract_gpucache_tuple(kcxt,
										 kds,
										 extra,
										 kds_index,
										 kexp_load_vars->u.load.desc,
										 kexp_load_vars->u.load.nitems))
			return false;
	}
	/* check scan quals if given */
	if (kexp_scan_quals)
	{
		xpu_bool_t	retval;

		if (EXEC_KERN_EXPRESSION(kcxt, kexp_scan_quals, &retval))
		{
			if (!XPU_DATUM_ISNULL(&retval) && retval.value)
				return true;
		}
		else
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
		}
		return false;
	}
	return true;
}

/* ------------------------------------------------------------
 *
 * MoveVars - that moves values in kvars-slot or vectorized kernel variables buffer
 *
 * ------------------------------------------------------------
 */
STATIC_FUNCTION(bool)
pgfn_MoveVars(XPU_PGFUNCTION_ARGS)
{
	STROM_ELOG(kcxt, "Bug? MoveVars shall not be called as a part of expression");
	return false;
}

PUBLIC_FUNCTION(bool)
ExecMoveKernelVariables(kern_context *kcxt,
						const kern_expression *kexp_move_vars,
						int curr_depth,   /* just for assertion checks */
						const char *src_kvec_buffer,  /* NULL, if depth==0 */
						int src_kvec_id,
						char *dst_kvec_buffer,
						int dst_kvec_id)
{
	if (!kexp_move_vars)
		return true;		/* nothing to move */
	assert(kexp_move_vars->opcode == FuncOpCode__MoveVars &&
		   kexp_move_vars->exptype == TypeOpCode__int4 &&
		   kexp_move_vars->nr_args == 0 &&
		   kexp_move_vars->u.move.depth == curr_depth);
	for (int i=0; i < kexp_move_vars->u.move.nitems; i++)
	{
		const kern_varmove_desc *vm_desc = &kexp_move_vars->u.move.desc[i];
		const xpu_datum_operators *vm_ops = vm_desc->vm_ops;

		if (vm_desc->vm_from_xdatum)
		{
			/* xdatum -> kvec-buffer */
			xpu_datum_t	   *xdatum;
			kvec_datum_t   *dst_kvec;

			assert(vm_desc->vm_slot_id >= 0 &&
				   vm_desc->vm_slot_id <  kcxt->kvars_nslots);
			xdatum = kcxt->kvars_values[vm_desc->vm_slot_id];
			dst_kvec = (kvec_datum_t *)(dst_kvec_buffer + vm_desc->vm_offset);
			assert((char *)dst_kvec +
				   vm_ops->xpu_kvec_sizeof <= (dst_kvec_buffer + kcxt->kvecs_bufsz));
			assert(XPU_DATUM_ISNULL(xdatum) || xdatum->expr_ops == vm_ops);
			if (!vm_ops->xpu_datum_kvec_store(kcxt,
											  xdatum,
											  dst_kvec,
											  dst_kvec_id))
				return false;
		}
		else
		{
			/* kvec-buffer -> kvec-buffer */
			const kvec_datum_t *src_kvec;
			kvec_datum_t	   *dst_kvec;

			assert(src_kvec_buffer != NULL);
			src_kvec = (const kvec_datum_t *)(src_kvec_buffer +
											  vm_desc->vm_offset);
			assert((char *)src_kvec +
				   vm_ops->xpu_kvec_sizeof <= (src_kvec_buffer + kcxt->kvecs_bufsz));
			dst_kvec = (kvec_datum_t *)(dst_kvec_buffer +
										vm_desc->vm_offset);
			assert((char *)dst_kvec +
				   vm_ops->xpu_kvec_sizeof <= (dst_kvec_buffer + kcxt->kvecs_bufsz));
			if (!vm_ops->xpu_datum_kvec_copy(kcxt,
											 src_kvec, src_kvec_id,
											 dst_kvec, dst_kvec_id))
				return false;
		}
	}
	return true;
}

/* ------------------------------------------------------------
 *
 * Routines to support GiST-Index
 *
 * ------------------------------------------------------------
 */
STATIC_FUNCTION(bool)
kern_extract_gist_tuple(kern_context *kcxt,
						const kern_data_store *kds_gist,
						const IndexTupleData *itup,
						const kern_varload_desc *vl_desc)
{
	char	   *nullmap = NULL;
	uint32_t	i_off;

	assert(vl_desc->vl_resno > 0 &&
		   vl_desc->vl_resno <= kds_gist->ncols);
	if (IndexTupleHasNulls(itup))
	{
		nullmap = (char *)itup + sizeof(IndexTupleData);
		i_off =  MAXALIGN(offsetof(IndexTupleData, data) +
						  sizeof(IndexAttributeBitMapData));
	}
	else
	{
		const kern_colmeta *cmeta = &kds_gist->colmeta[vl_desc->vl_resno-1];

		i_off = MAXALIGN(offsetof(IndexTupleData, data));
		if (cmeta->attcacheoff >= 0)
		{
			char   *addr = (char *)itup + i_off + cmeta->attcacheoff;
			return __extract_heap_tuple_attr(kcxt, vl_desc, addr);
		}
	}
	/* extract the index-tuple by the slow path */
	for (int resno=1; resno <= kds_gist->ncols; resno++)
	{
		const kern_colmeta *cmeta = &kds_gist->colmeta[resno-1];
		char	   *addr;

		if (nullmap && att_isnull(resno-1, nullmap))
			addr = NULL;
		else
		{
			if (cmeta->attlen > 0)
				i_off = TYPEALIGN(cmeta->attalign, i_off);
			else if (!VARATT_NOT_PAD_BYTE((char *)itup + i_off))
				i_off = TYPEALIGN(cmeta->attalign, i_off);

			addr = (char *)itup + i_off;
			if (cmeta->attlen > 0)
				i_off += cmeta->attlen;
			else
				i_off += VARSIZE_ANY(addr);
		}
		if (vl_desc->vl_resno == resno)
			return __extract_heap_tuple_attr(kcxt, vl_desc, addr);
	}
	/* fill-up by NULL, if not found */
	return __extract_heap_tuple_attr(kcxt, vl_desc, NULL);
}

PUBLIC_FUNCTION(uint32_t)
ExecGiSTIndexGetNext(kern_context *kcxt,
					 const kern_data_store *kds_hash,
					 const kern_data_store *kds_gist,
					 const kern_expression *kexp_gist,
					 uint32_t l_state)
{
	PageHeaderData *gist_page;
	ItemIdData	   *lpp;
	IndexTupleData *itup;
	OffsetNumber	start;
	OffsetNumber	index;
	OffsetNumber	maxoff;
	const kern_expression *karg_gist;
	const kern_varload_desc *vl_desc;

	assert(kds_hash->format == KDS_FORMAT_HASH &&
		   kds_gist->format == KDS_FORMAT_BLOCK);
	assert(kexp_gist->opcode == FuncOpCode__GiSTEval &&
		   kexp_gist->exptype == TypeOpCode__bool);
	vl_desc = &kexp_gist->u.gist.ivar_desc;
	karg_gist = KEXP_FIRST_ARG(kexp_gist);
	assert(karg_gist->exptype ==  TypeOpCode__bool);

	if (l_state == 0)
	{
		gist_page = KDS_BLOCK_PGPAGE(kds_gist, GIST_ROOT_BLKNO);
		start = FirstOffsetNumber;
	}
	else
	{
		size_t		l_off = sizeof(ItemIdData) * l_state;
		size_t		diff;

		assert(l_off >= kds_gist->block_offset &&
			   l_off <  kds_gist->length);
		lpp = (ItemIdData *)((char *)kds_gist + l_off);
		diff = ((l_off - kds_gist->block_offset) & (BLCKSZ-1));
		gist_page = (PageHeaderData *)((char *)lpp - diff);
		assert((char *)lpp >= (char *)gist_page->pd_linp &&
			   (char *)lpp <  (char *)gist_page + BLCKSZ);
		start = (lpp - gist_page->pd_linp) + FirstOffsetNumber;
	}
restart:
	assert(KDS_BLOCK_CHECK_VALID(kds_gist, gist_page));

	if (GistPageIsDeleted(gist_page))
		maxoff = InvalidOffsetNumber;	/* skip any entries */
	else
		maxoff = PageGetMaxOffsetNumber(gist_page);

	for (index=start; index <= maxoff; index++)
	{
		xpu_bool_t	status;

		lpp = PageGetItemId(gist_page, index);
		if (!ItemIdIsNormal(lpp))
			continue;

		kcxt_reset(kcxt);
		/* extract the index tuple */
		itup = (IndexTupleData *)PageGetItem(gist_page, lpp);
		if (!kern_extract_gist_tuple(kcxt, kds_gist, itup, vl_desc))
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
			return UINT_MAX;
		}
		/* runs index-qualifier */
		if (!EXEC_KERN_EXPRESSION(kcxt, karg_gist, &status))
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
			return UINT_MAX;
		}
		/* check result */
		if (!XPU_DATUM_ISNULL(&status) && status.value)
		{
			BlockNumber		block_nr;

			if (GistPageIsLeaf(gist_page))
			{
				const kern_varslot_desc *vs_desc;
				uint32_t	slot_id = kexp_gist->u.gist.htup_slot_id;
				uint32_t	t_off;
				const char *addr;

				assert(itup->t_tid.ip_posid == InvalidOffsetNumber);
				t_off = ((uint32_t)itup->t_tid.ip_blkid.bi_hi << 16 |
						 (uint32_t)itup->t_tid.ip_blkid.bi_lo);
				addr = (const char *)kds_hash + __kds_unpack(t_off);
				assert(slot_id < kcxt->kvars_nslots);
				vs_desc = &kcxt->kvars_desc[slot_id];
				assert(vs_desc->vs_type_code == TypeOpCode__internal);				
				if (!vs_desc->vs_ops->xpu_datum_heap_ref(kcxt, addr,
														 kcxt->kvars_values[slot_id]))
				{
					assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
					return UINT_MAX;
				}
				/* returns the offset of the next line item pointer */
				assert((((uintptr_t)lpp) & (sizeof(ItemIdData)-1)) == 0);
				return ((char *)(lpp+1) - (char *)(kds_gist)) / sizeof(ItemIdData);
			}
			block_nr = ((BlockNumber)itup->t_tid.ip_blkid.bi_hi << 16 |
						(BlockNumber)itup->t_tid.ip_blkid.bi_lo);
			assert(block_nr < kds_gist->nitems);
			gist_page = KDS_BLOCK_PGPAGE(kds_gist, block_nr);
			start = FirstOffsetNumber;
			goto restart;
		}
	}

	if (!GistPageIsRoot(gist_page))
	{
		/* pop to the parent page if not found */
		start = gist_page->pd_parent_item + 1;
		gist_page = KDS_BLOCK_PGPAGE(kds_gist, gist_page->pd_parent_blkno);
		goto restart;
	}
	return UINT_MAX;	/* no more chance for this outer */
}

PUBLIC_FUNCTION(bool)
ExecGiSTIndexPostQuals(kern_context *kcxt,
					   int depth,
					   const kern_data_store *kds_hash,
					   const kern_expression *kexp_gist,
					   const kern_expression *kexp_load,
					   const kern_expression *kexp_join)
{
	HeapTupleHeaderData *htup;
	xpu_internal_t *xdatum;
	xpu_bool_t		status;
	uint32_t		slot_id;

	/* fetch the inner heap tuple */
	assert(kexp_gist->opcode == FuncOpCode__GiSTEval);
	slot_id = kexp_gist->u.gist.htup_slot_id;
	assert(slot_id < kcxt->kvars_nslots &&
		   kcxt->kvars_desc[slot_id].vs_type_code == TypeOpCode__internal);
	/* load the inner heap tuple */
	xdatum = (xpu_internal_t *)kcxt->kvars_values[slot_id];
	assert(!XPU_DATUM_ISNULL(xdatum) &&
		   xdatum->value >= (char *)kds_hash &&
		   xdatum->value <  (char *)kds_hash + kds_hash->length);
	htup = (HeapTupleHeaderData *)xdatum->value;
	if (!ExecLoadVarsHeapTuple(kcxt, kexp_load, depth, kds_hash, htup))
	{
		STROM_ELOG(kcxt, "Bug? GiST-Index prep fetched corrupted tuple");
		return false;
	}
	/* run the join quals */
	kcxt_reset(kcxt);
	if (!EXEC_KERN_EXPRESSION(kcxt, kexp_join, &status))
	{
		assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
		return false;
	}
	return (!XPU_DATUM_ISNULL(&status) && status.value);
}

/*
 * xpu_arrow_t type support routine
 */
STATIC_FUNCTION(bool)
xpu_array_datum_ref(kern_context *kcxt,
					xpu_datum_t *__result,
					int vclass,
					const kern_variable *kvar)
{
	xpu_array_t	   *result = (xpu_array_t *)__result;

	if (vclass == KVAR_CLASS__VARLENA)
	{
		result->expr_ops = &xpu_array_ops;
		result->length = -1;
		result->u.heap.value = (const varlena *)kvar->ptr;
	}
	else
	{
		 STROM_ELOG(kcxt, "unexpected vclass for device numeric data type.");
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_array_datum_store(kern_context *kcxt,
					  const xpu_datum_t *__arg,
					  int *p_vclass,
					  kern_variable *p_kvar)
{
	const xpu_array_t *arg = (const xpu_array_t *)__arg;

	if (XPU_DATUM_ISNULL(arg))
	{
		*p_vclass = KVAR_CLASS__NULL;
	}
	else if (arg->length < 0)
	{
		*p_vclass   = KVAR_CLASS__VARLENA;
		p_kvar->ptr = (void *)arg->u.heap.value;
	}
	else
	{
		STROM_ELOG(kcxt, "unable to use intermediation results of xpu_array_t");
		return false;
	}
	return true;
}

STATIC_FUNCTION(int)
xpu_array_datum_write(kern_context *kcxt,
					  char *buffer,
					  const kern_colmeta *cmeta,
					  const xpu_datum_t *__arg)
{
	const xpu_array_t *arg = (const xpu_array_t *)__arg;
	int		nbytes;

	if (XPU_DATUM_ISNULL(arg))
		return 0;
	if (arg->length < 0)
	{
		nbytes = VARSIZE_ANY(arg->u.heap.value);
		if (buffer)
			memcpy(buffer, arg->u.heap.value, nbytes);
	}
	else
	{
		const kern_colmeta *smeta = arg->u.arrow.smeta;
		const kern_data_store *kds;
		uint8_t		   *nullmap = NULL;
		kern_variable	kvar;
		int				vclass;
		char		   *slot_buf = NULL;

		kds = (const kern_data_store *)
			((char *)smeta - smeta->kds_offset);
		if (smeta->dtype_sizeof > 0)
			slot_buf = (char *)alloca(smeta->dtype_sizeof);
		nbytes = (VARHDRSZ +
				  offsetof(__ArrayTypeData, data[2]) +
				  MAXALIGN(BITMAPLEN(arg->length)));
		if (buffer)
		{
			__ArrayTypeData *arr = (__ArrayTypeData *)(buffer + VARHDRSZ);

			memset(arr, 0, nbytes - VARHDRSZ);
			arr->ndim = 1;
			arr->elemtype = smeta->atttypid;
			arr->data[0] = arg->length;
			arr->data[1] = 1;
			if (smeta->nullmap_offset != 0)
				nullmap = (uint8_t *)&arr->data[2];
		}

		for (int k=0; k < arg->length; k++)
		{
			uint32_t	index = arg->u.arrow.start + k;

			if (smeta->nullmap_offset != 0 &&
				!arrow_bitmap_check(kds, index,
									smeta->nullmap_offset,
									smeta->nullmap_length))
				continue;
			if (!__kern_extract_arrow_field(kcxt,
											kds,
											smeta,
											index,
											&kvar,
											&vclass,
											slot_buf))
				return false;
			if (vclass != KVAR_CLASS__NULL)
			{
				if (nullmap)
					nullmap[k>>3] |= (1<<(k & 7));

				nbytes = TYPEALIGN(smeta->attalign, nbytes);
				if (vclass == KVAR_CLASS__INLINE)
				{
					if (buffer)
						memcpy(buffer + nbytes, &kvar, smeta->attlen);
					nbytes += smeta->attlen;
				}
				else if (vclass == KVAR_CLASS__VARLENA)
				{
					int		sz = VARSIZE_ANY(kvar.ptr);

					if (buffer)
						memcpy(buffer + nbytes, kvar.ptr, sz);
					nbytes += sz;
				}
				else if (vclass == KVAR_CLASS__XPU_DATUM)
				{
					xpu_datum_t	*xdatum = (xpu_datum_t *)kvar.ptr;
					char   *dst = (buffer ? buffer + nbytes : NULL);
					int		sz;

					sz = xdatum->expr_ops->xpu_datum_write(kcxt, dst, smeta, xdatum);
					if (sz < 0)
						return false;
					nbytes += sz;
				}
				else
				{
					assert(vclass > 0);

					if (buffer)
						memcpy(buffer + nbytes, kvar.ptr, vclass);
					nbytes += vclass;
				}
			}
		}
	}
	return nbytes;
}

STATIC_FUNCTION(bool)
xpu_array_datum_hash(kern_context *kcxt,
					 uint32_t *p_hash,
					 const xpu_datum_t *arg)
{
	STROM_ELOG(kcxt, "xpu_array_datum_hash is not implemented");
	return false;
}

STATIC_FUNCTION(bool)
xpu_array_datum_comp(kern_context *kcxt,
					 int *p_comp,
					 const xpu_datum_t *__a,
					 const xpu_datum_t *__b)
{
	STROM_ELOG(kcxt, "xpu_array_datum_comp is not implemented");
	return false;
}
STATIC_FUNCTION(bool)
xpu_array_datum_load_heap(kern_context *kcxt,
						  kvec_datum_t *__result,
						  int kvec_id,
						  const char *addr)
{
	STROM_ELOG(kcxt, "xpu_composite_datum_heap_load is not implemented");
	return false;
}
//MEMO: some array type uses typalign=4. is it ok?
PGSTROM_SQLTYPE_OPERATORS(array,false,4,-1);

/*
 * xpu_composite_t type support routine
 */
STATIC_FUNCTION(bool)
xpu_composite_datum_ref(kern_context *kcxt,
						xpu_datum_t *result,
						int vclass,
						const kern_variable *kvar)
{
	STROM_ELOG(kcxt, "xpu_composite_datum_ref is not implemented");
	return false;
}

STATIC_FUNCTION(bool)
xpu_composite_datum_store(kern_context *kcxt,
						  const xpu_datum_t *arg,
						  int *p_vclass,
						  kern_variable *p_kvar)
{
	STROM_ELOG(kcxt, "xpu_composite_datum_store is not implemented");
	return false;
}

STATIC_FUNCTION(int)
xpu_composite_datum_write(kern_context *kcxt,
						  char *buffer,
						  const kern_colmeta *cmeta,
						  const xpu_datum_t *xdatum)
{
	STROM_ELOG(kcxt, "xpu_composite_datum_write is not implemented");
	return false;
}

STATIC_FUNCTION(bool)
xpu_composite_datum_hash(kern_context *kcxt,
						 uint32_t *p_hash,
						 const xpu_datum_t *arg)
{
	STROM_ELOG(kcxt, "xpu_composite_datum_hash is not implemented");
	return false;
}
STATIC_FUNCTION(bool)
xpu_composite_datum_comp(kern_context *kcxt,
						 int *p_comp,
						 const xpu_datum_t *__a,
						 const xpu_datum_t *__b)
{
	STROM_ELOG(kcxt, "xpu_composite_datum_comp is not implemented");
	return false;
}
STATIC_FUNCTION(bool)
xpu_composite_datum_load_heap(kern_context *kcxt,
							  kvec_datum_t *__result,
							  int kvec_id,
							  const char *addr)
{
	STROM_ELOG(kcxt, "xpu_composite_datum_heap_load is not implemented");
	return false;
}
PGSTROM_SQLTYPE_OPERATORS(composite,false,8,-1);

/*
 * Catalog of built-in device types
 */
/*
 * Built-in SQL type / function catalog
 */
#define TYPE_OPCODE(NAME,a,b)					\
	{ TypeOpCode__##NAME, &xpu_##NAME##_ops },
PUBLIC_DATA xpu_type_catalog_entry builtin_xpu_types_catalog[] = {
#include "xpu_opcodes.h"
	//{ TypeOpCode__composite, &xpu_composite_ops },
	{ TypeOpCode__array, &xpu_array_ops },
	{ TypeOpCode__Invalid, NULL }
};

/*
 * Catalog of built-in device functions
 */
#define FUNC_OPCODE(a,b,c,NAME,d,e)			\
	{FuncOpCode__##NAME, pgfn_##NAME},
#define DEVONLY_FUNC_OPCODE(a,NAME,b,c,d)	\
	{FuncOpCode__##NAME, pgfn_##NAME},
PUBLIC_DATA xpu_function_catalog_entry builtin_xpu_functions_catalog[] = {
	{FuncOpCode__ConstExpr, 				pgfn_ConstExpr },
	{FuncOpCode__ParamExpr, 				pgfn_ParamExpr },
	{FuncOpCode__VarExpr,					pgfn_VarExpr },
	{FuncOpCode__BoolExpr_And,				pgfn_BoolExprAnd },
	{FuncOpCode__BoolExpr_Or,				pgfn_BoolExprOr },
	{FuncOpCode__BoolExpr_Not,				pgfn_BoolExprNot },
	{FuncOpCode__NullTestExpr_IsNull,		pgfn_NullTestExpr },
	{FuncOpCode__NullTestExpr_IsNotNull,	pgfn_NullTestExpr },
	{FuncOpCode__BoolTestExpr_IsTrue,		pgfn_BoolTestExpr},
	{FuncOpCode__BoolTestExpr_IsNotTrue,	pgfn_BoolTestExpr},
	{FuncOpCode__BoolTestExpr_IsFalse,		pgfn_BoolTestExpr},
	{FuncOpCode__BoolTestExpr_IsNotFalse,	pgfn_BoolTestExpr},
	{FuncOpCode__BoolTestExpr_IsUnknown,	pgfn_BoolTestExpr},
	{FuncOpCode__BoolTestExpr_IsNotUnknown,	pgfn_BoolTestExpr},
	{FuncOpCode__DistinctFrom,              pgfn_DistinctFrom},
	{FuncOpCode__CoalesceExpr,				pgfn_CoalesceExpr},
	{FuncOpCode__LeastExpr,					pgfn_LeastExpr},
	{FuncOpCode__GreatestExpr,				pgfn_GreatestExpr},
	{FuncOpCode__CaseWhenExpr,				pgfn_CaseWhenExpr},
	{FuncOpCode__ScalarArrayOpAny,			pgfn_ScalarArrayOp},
	{FuncOpCode__ScalarArrayOpAll,			pgfn_ScalarArrayOp},
#include "xpu_opcodes.h"
	{FuncOpCode__Projection,                pgfn_Projection},
	{FuncOpCode__LoadVars,                  pgfn_LoadVars},
	{FuncOpCode__MoveVars,					pgfn_MoveVars},
	{FuncOpCode__HashValue,                 pgfn_HashValue},
	{FuncOpCode__GiSTEval,                  pgfn_GiSTEval},
	{FuncOpCode__SaveExpr,                  pgfn_SaveExpr},
	{FuncOpCode__AggFuncs,                  pgfn_AggFuncs},
	{FuncOpCode__JoinQuals,                 pgfn_JoinQuals},
	{FuncOpCode__Packed,                    pgfn_Packed},
	{FuncOpCode__Invalid, NULL},
};

/*
 * Device version of hash_any() in PG host code
 */
#define rot(x,k)		(((x)<<(k)) | ((x)>>(32-(k))))
#define mix(a,b,c)								\
	{											\
		a -= c;  a ^= rot(c, 4);  c += b;		\
		b -= a;  b ^= rot(a, 6);  a += c;		\
		c -= b;  c ^= rot(b, 8);  b += a;		\
		a -= c;  a ^= rot(c,16);  c += b;		\
		b -= a;  b ^= rot(a,19);  a += c;		\
		c -= b;  c ^= rot(b, 4);  b += a;		\
	}

#define final(a,b,c)							\
	{											\
		c ^= b; c -= rot(b,14);					\
		a ^= c; a -= rot(c,11);					\
		b ^= a; b -= rot(a,25);					\
		c ^= b; c -= rot(b,16);					\
		a ^= c; a -= rot(c, 4);					\
		b ^= a; b -= rot(a,14);					\
		c ^= b; c -= rot(b,24);					\
	}

PUBLIC_FUNCTION(uint32_t)
pg_hash_any(const void *ptr, int sz)
{
	const uint8_t  *k = (const uint8_t *)ptr;
	uint32_t		a, b, c;
	uint32_t		len = sz;

	/* Set up the internal state */
	a = b = c = 0x9e3779b9 + len + 3923095;

	/* If the source pointer is word-aligned, we use word-wide fetches */
	if (((uint64_t) k & (sizeof(uint32_t) - 1)) == 0)
	{
		/* Code path for aligned source data */
		const uint32_t	*ka = (const uint32_t *) k;

		/* handle most of the key */
		while (len >= 12)
		{
			a += ka[0];
			b += ka[1];
			c += ka[2];
			mix(a, b, c);
			ka += 3;
			len -= 12;
		}

		/* handle the last 11 bytes */
		k = (const unsigned char *) ka;
		switch (len)
		{
			case 11:
				c += ((uint32_t) k[10] << 24);
				/* fall through */
			case 10:
				c += ((uint32_t) k[9] << 16);
				/* fall through */
			case 9:
				c += ((uint32_t) k[8] << 8);
				/* the lowest byte of c is reserved for the length */
				/* fall through */
			case 8:
				b += ka[1];
				a += ka[0];
				break;
			case 7:
				b += ((uint32_t) k[6] << 16);
				/* fall through */
			case 6:
				b += ((uint32_t) k[5] << 8);
				/* fall through */
			case 5:
				b += k[4];
				/* fall through */
			case 4:
				a += ka[0];
				break;
			case 3:
				a += ((uint32_t) k[2] << 16);
				/* fall through */
			case 2:
				a += ((uint32_t) k[1] << 8);
				/* fall through */
			case 1:
				a += k[0];
				/* case 0: nothing left to add */
		}
	}
	else
	{
		/* Code path for non-aligned source data */

		/* handle most of the key */
		while (len >= 12)
		{
			a += k[0] + (((uint32_t) k[1] << 8) +
						 ((uint32_t) k[2] << 16) +
						 ((uint32_t) k[3] << 24));
			b += k[4] + (((uint32_t) k[5] << 8) +
						 ((uint32_t) k[6] << 16) +
						 ((uint32_t) k[7] << 24));
			c += k[8] + (((uint32_t) k[9] << 8) +
						 ((uint32_t) k[10] << 16) +
						 ((uint32_t) k[11] << 24));
			mix(a, b, c);
			k += 12;
			len -= 12;
		}

		/* handle the last 11 bytes */
		switch (len)            /* all the case statements fall through */
		{
			case 11:
				c += ((uint32_t) k[10] << 24);
			case 10:
				c += ((uint32_t) k[9] << 16);
			case 9:
				c += ((uint32_t) k[8] << 8);
				/* the lowest byte of c is reserved for the length */
			case 8:
				b += ((uint32_t) k[7] << 24);
			case 7:
				b += ((uint32_t) k[6] << 16);
			case 6:
				b += ((uint32_t) k[5] << 8);
			case 5:
				b += k[4];
			case 4:
				a += ((uint32_t) k[3] << 24);
			case 3:
				a += ((uint32_t) k[2] << 16);
			case 2:
				a += ((uint32_t) k[1] << 8);
			case 1:
				a += k[0];
				/* case 0: nothing left to add */
		}
	}
	final(a, b, c);

	return c;
}
#undef rot
#undef mix
#undef final
