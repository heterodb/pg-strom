/*
 * cuda_anytype.h
 *
 * Routines to be put after all the type definition, like pg_array_t or
 * pg_anytype_t types.
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
#ifndef CUDA_ANYTYPE_H
#define CUDA_ANYTYPE_H

#ifndef PG_ARRAY_TYPE_DEFINED
#define PG_ARRAY_TYPE_DEFINED
/*
 * NOTE: pg_array_t is designed to store both of PostgreSQL / Arrow array
 * values. If @length < 0, it means @value points a varlena based PostgreSQL
 * array values; which includes nitems, dimension, nullmap and so on.
 * Elsewhere, @length means number of elements, from @start of the array on
 * the columnar buffer by @smeta.
 */
typedef struct {
	char	   *value;
	cl_bool		isnull;
	cl_int		length;
	cl_int		start;
	kern_colmeta *smeta;
} pg_array_t;

#ifdef __CUDACC__
STATIC_INLINE(pg_array_t)
pg_array_datum_ref(kern_context *kcxt, void *addr)
{
	pg_array_t	result;

	if (!addr)
		result.isnull = true;
	else
	{
		result.value  = (char *)addr;
		result.isnull = false;
		result.length = -1;
		result.start  = -1;
		result.smeta  = NULL;
	}
	return result;
}

STATIC_INLINE(void)
pg_datum_ref(kern_context *kcxt,
			 pg_array_t &result, void *addr)
{
	result = pg_array_datum_ref(kcxt, addr);
}

STATIC_INLINE(void)
pg_datum_ref_slot(kern_context *kcxt,
				  pg_array_t &result,
				  cl_char dclass, Datum datum)
{
	if (dclass == DATUM_CLASS__NULL)
		result = pg_array_datum_ref(kcxt, NULL);
	else if (dclass == DATUM_CLASS__ARRAY)
		memcpy(&result, DatumGetPointer(datum), sizeof(pg_array_t));
	else
	{
		assert(dclass == DATUM_CLASS__NORMAL);
		result = pg_array_datum_ref(kcxt, (char *)datum);
	}
}

STATIC_INLINE(void)
pg_datum_ref_arrow(kern_context *kcxt,
				   pg_array_t &result,
				   kern_data_store *kds,
				   cl_uint colidx, cl_uint rowidx)
{
	kern_colmeta   *cmeta = &kds->colmeta[colidx];
	kern_colmeta   *smeta;
	cl_uint		   *offset;

	assert(kds->format == KDS_FORMAT_ARROW);
	assert(colidx < kds->ncols &&
		   rowidx < kds->nitems);
	smeta = &kds->colmeta[cmeta->idx_subattrs];
	assert(cmeta->num_subattrs == 1 &&
		   cmeta->idx_subattrs < kds->nr_colmeta);
	if (cmeta->nullmap_offset != 0)
	{
		cl_char	   *nullmap =
			(char *)kds + __kds_unpack(cmeta->nullmap_offset);
		if (att_isnull(rowidx, nullmap))
		{
			result.isnull = true;
			return;
		}
	}
	offset = (cl_uint *)((char *)kds + __kds_unpack(cmeta->values_offset));
	assert(offset[rowidx] <= offset[rowidx+1]);

	result.value  = (char *)kds;
	result.isnull = false;
	result.length = (offset[rowidx+1] - offset[rowidx]);
	result.start  = offset[rowidx];
	result.smeta  = smeta;
}

STATIC_INLINE(cl_int)
pg_datum_store(kern_context *kcxt,
			   pg_array_t datum,
			   cl_char &dclass,
			   Datum &value)
{
	if (datum.isnull)
		dclass = DATUM_CLASS__NULL;
	else if (datum.length < 0)
	{
		cl_uint		len = VARSIZE_ANY(datum.value);

		dclass = DATUM_CLASS__NORMAL;
		value  = PointerGetDatum(datum.value);
		if (PTR_ON_VLBUF(kcxt, datum.value, len))
			return len;
	}
	else
	{
		pg_array_t *temp;

		temp = (pg_array_t *)
			kern_context_alloc(kcxt, sizeof(pg_array_t));
		if (temp)
		{
			memcpy(temp, &datum, sizeof(pg_array_t));
			dclass = DATUM_CLASS__ARRAY;
			value  = PointerGetDatum(temp);
			return sizeof(pg_array_t);
		}
		STROM_SET_ERROR(&kcxt->e, StromError_OutOfMemory);
		dclass = DATUM_CLASS__NULL;
	}
	return 0;
}

STATIC_INLINE(pg_array_t)
pg_array_param(kern_context *kcxt, cl_uint param_id)
{
	kern_parambuf  *kparams = kcxt->kparams;
	pg_array_t		result;

	if (param_id < kparams->nparams &&
		kparams->poffset[param_id] > 0)
	{
		char   *addr = (char *)kparams + kparams->poffset[param_id];

		if (VARATT_IS_4B_U(addr) || VARATT_IS_1B(addr))
		{
			result.value  = addr;
			result.isnull = false;
			result.length = -1;
			result.start  = -1;
			result.smeta  = NULL;
		}
		else
		{
			result.isnull = true;
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		}
	}
	else
	{
		result.isnull = true;
	}
	return result;
}

STATIC_INLINE(cl_uint)
pg_comp_hash(kern_context *kcxt, pg_array_t datum)
{
	if (datum.isnull)
		return 0;
	if (datum.length < 0)
		return 0;
	else
		return 0;
}
#endif	/* __CUDACC__ */
#endif	/* PG_ARRAY_TYPE_DEFINED */

#ifdef __CUDACC__
/*
 * functions to write out Arrow::List<T> as an array of PostgreSQL
 *
 * only called if dclass == DATUM_CLASS__ARRAY
 */
STATIC_FUNCTION(cl_uint)
__pg_array_from_arrow(kern_context *kcxt, char *dest, Datum datum)
{
	pg_array_t	   *array = (pg_array_t *)DatumGetPointer(datum);
	kern_colmeta   *smeta = array->smeta;
	char		   *base  = array->value;
	cl_uint			start = array->start;
	cl_uint			end   = array->start + array->length;
	cl_uint			sz;

	assert(!array->isnull && array->length >= 0);
	assert(start <= end);
	switch (smeta->atttypid)
	{
		case PG_BOOLOID:
			sz = pg_bool_array_from_arrow(kcxt,dest,smeta,base,start,end);
			break;
		case PG_INT2OID:
			sz = pg_int2_array_from_arrow(kcxt,dest,smeta,base,start,end);
			break;
		case PG_INT4OID:
			sz = pg_int4_array_from_arrow(kcxt,dest,smeta,base,start,end);
			break;
		case PG_INT8OID:
			sz = pg_int8_array_from_arrow(kcxt,dest,smeta,base,start,end);
			break;
		case PG_FLOAT2OID:
			sz = pg_float2_array_from_arrow(kcxt,dest,smeta,base,start,end);
			break;
		case PG_FLOAT4OID:
			sz = pg_float4_array_from_arrow(kcxt,dest,smeta,base,start,end);
			break;
		case PG_FLOAT8OID:
			sz = pg_float8_array_from_arrow(kcxt,dest,smeta,base,start,end);
			break;
#ifdef PG_NUMERIC_TYPE_DEFINED
		case PG_NUMERICOID:
			sz = pg_numeric_array_from_arrow(kcxt,dest,smeta,base,start,end);
			break;
#endif
#ifdef PG_DATE_TYPE_DEFINED
		case PG_DATEOID:
			sz = pg_date_array_from_arrow(kcxt,dest,smeta,base,start,end);
			break;
#endif
#ifdef PG_TIME_TYPE_DEFINED
		case PG_TIMEOID:
			sz = pg_time_array_from_arrow(kcxt,dest,smeta,base,start,end);
			break;
#endif
#ifdef PG_TIMESTAMP_TYPE_DEFINED
		case PG_TIMESTAMPOID:
			sz = pg_timestamp_array_from_arrow(kcxt,dest,smeta,base,start,end);
			break;
#endif
#ifdef PG_INTERVAL_TYPE_DEFINED
		case PG_INTERVALOID:
			sz = pg_interval_array_from_arrow(kcxt,dest,smeta,base,start,end);
			break;
#endif
#ifdef PG_BPCHAR_TYPE_DEFINED
		case PG_BPCHAROID:
			sz = pg_bpchar_array_from_arrow(kcxt,dest,smeta,base,start,end);
			break;
#endif
#ifdef PG_TEXT_TYPE_DEFINED
		case PG_TEXTOID:
			sz = pg_text_array_from_arrow(kcxt,dest,smeta,base,start,end);
			break;
#endif
#ifdef PG_VARCHAR_TYPE_DEFINED
		case PG_VARCHAROID:
			sz = pg_varchar_array_from_arrow(kcxt,dest,smeta,base,start,end);
			break;
#endif
#ifdef PG_BYTEA_TYPE_DEFINED
		case PG_BYTEAOID:
			sz = pg_bytea_array_from_arrow(kcxt,dest,smeta,base,start,end);
			break;
#endif
		default:
			STROM_SET_ERROR(&kcxt->e, StromError_WrongCodeGeneration);
			return 0;
	}
	return sz;
}
#endif	/* __CUDACC__ */

/*
 * declaration of pg_anytype_t
 */
#ifdef __CUDACC__
typedef union {
/* cuda_common.h */
	pg_varlena_t		varlena_v;
	pg_bool_t			bool_v;
	pg_int2_t			int2_v;
	pg_int4_t			int4_v;
	pg_int8_t			int8_v;
	pg_float2_t			float2_v;
	pg_float4_t			float4_v;
	pg_float8_t			float8_v;
/* cuda_numeric.h */
#ifdef PG_NUMERIC_TYPE_DEFINED
	pg_numeric_t		numeric_v;
#endif
/* cuda_misc.h */
#ifdef PG_MONEY_TYPE_DEFINED
	pg_money_t			monery_v;
#endif
#ifdef PG_UUID_TYPE_DEFINED
	pg_uuid_t			uuid_v;
#endif
#ifdef PG_MACADDR_TYPE_DEFINED
	pg_macaddr_t		macaddr_v;
#endif
#ifdef PG_INET_TYPE_DEFINED
	pg_inet_t			inet_v;
#endif
#ifdef PG_CIDR_TYPE_DEFINED
	pg_cidr_t			cidr_v;
#endif
/* cuda_timelib.h */
#ifdef PG_DATE_TYPE_DEFINED
	pg_date_t			date_v;
#endif
#ifdef PG_TIME_TYPE_DEFINED
	pg_time_t			time_v;
#endif
#ifdef PG_TIMETZ_TYPE_DEFINED
	pg_timetz_t			timetz_v;
#endif
#ifdef PG_TIMESTAMP_TYPE_DEFINED
	pg_timestamp_t		timestamp_v;
#endif
#ifdef PG_TIMESTAMPTZ_TYPE_DEFINED
	pg_timestamptz_t	timestamptz_v;
#endif
#ifdef PG_INTERVAL_TYPE_DEFINED
	pg_interval_t		interval_v;
#endif
/* cuda_textlib.h */
#ifdef PG_BPCHAR_TYPE_DEFINED
	pg_bpchar_t			bpchar_v;
#endif
#ifdef PG_TEXT_TYPE_DEFINED
	pg_text_t			text_v;
#endif
#ifdef PG_VARCHAR_TYPE_DEFINED
	pg_varchar_t		varchar_v;
#endif
/* cuda_jsonlib.h */
#ifdef PG_JSONB_TYPE_DEFINED
	pg_jsonb_t			jsonb_v;
#endif
/* cuda_rangetype.h */
#ifdef PG_INT4RANGE_TYPE_DEFINED
	pg_int4range_t		int4range_v;
#endif
#ifdef PG_INT8RANGE_TYPE_DEFINED
	pg_int8range_t		int8range_v;
#endif
#ifdef PG_TSRANGE_TYPE_DEFINED
	pg_tsrange_t		tsrange_v;
#endif
#ifdef PG_TSTZRANGE_TYPE_DEFINED
	pg_tstzrange_t		tstzrange_v;
#endif
#ifdef PG_DATERANGE_TYPE_DEFINED
	pg_daterange_t		daterange_v
#endif
#ifdef PG_ARRAY_TYPE_DEFINED
	pg_array_t			array_v;
#endif
} pg_anytype_t;
#endif	/* __CUDACC__ */
#endif	/* CUDA_ANYTYPE_H */
