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

#ifdef PGSTROM_KERNEL_HAS_PGARRAY
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
pg_datum_fetch_arrow(kern_context *kcxt,
					 pg_array_t &result,
					 kern_colmeta *cmeta,
					 char *base, cl_uint rowidx)
{
	kern_data_store *kds = (kern_data_store *)base;
	kern_colmeta   *smeta;
	cl_uint		   *offset;

	assert(kds->format == KDS_FORMAT_ARROW);
	assert(cmeta->num_subattrs == 1 &&
		   cmeta->idx_subattrs < kds->nr_colmeta);
	assert(rowidx < kds->nitems);
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
	smeta = &kds->colmeta[cmeta->idx_subattrs];
	offset = (cl_uint *)((char *)kds + __kds_unpack(cmeta->values_offset));
	assert(offset[rowidx] <= offset[rowidx+1]);

	result.value  = base;
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
			result = pg_array_datum_ref(kcxt, addr);
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
	/* we don't support to use pg_array_t for JOIN/GROUP BY key */
	STROM_SET_ERROR(&kcxt->e, StromError_WrongCodeGeneration);
	return 0;
}
#endif	/* PGSTROM_KERNEL_HAS_PGARRAY */
#endif	/* PG_ARRAY_TYPE_DEFINED */

/*
 * composite data type support
 *
 * A composite value is stored as a varlena in PostgreSQL row-format, so
 * we don't need a special case handling unless we don't walk on the sub-
 * fields of the composite value. (Exactly, it is a future enhancement.)
 * On the other hands, Arrow::Struct requires to reference multiple buffers,
 * to reconstruct a composite value of PostgreSQL format.
 *
 * If @length < 0, it means @value points a varlena based PostgreSQL composite
 * value, and other fields are invalid. Elsewhere, @length means number of
 * sub-fields; its metadata is smeta[0 ... @length-1]. In this case, @value
 * points KDS; that is also base pointer for nullmap/values/extra offset.
 * @rowidx is as literal; must be less than ((kern_data_store *)@value)->nitems
 */
#ifndef PG_COMPOSITE_TYPE_DEFINED
#define PG_COMPOSITE_TYPE_DEFINED
typedef struct {
	char	   *value;
	cl_bool		isnull;
	cl_int		length;
	cl_int		rowidx;
	cl_uint		comp_typid;
	cl_int		comp_typmod;
	kern_colmeta *smeta;
} pg_composite_t;

#ifdef PGSTROM_KERNEL_HAS_PGCOMPOSITE

STATIC_INLINE(pg_composite_t)
pg_composite_datum_ref(kern_context *kcxt, void *addr)
{
	pg_composite_t	result;

	if (!addr)
		result.isnull = true;
	else
	{
		HeapTupleHeaderData *htup = (HeapTupleHeaderData *)addr;
		result.value  = (char *)htup;
		result.isnull = false;
		result.length = -1;
		result.rowidx = -1;
		result.comp_typid = __Fetch(&htup->t_choice.t_datum.datum_typeid);
		result.comp_typmod = __Fetch(&htup->t_choice.t_datum.datum_typmod);
		result.smeta  = NULL;
	}
	return result;
}

STATIC_INLINE(void)
pg_datum_ref(kern_context *kcxt,
			 pg_composite_t &result, void *addr)
{
	result = pg_composite_datum_ref(kcxt, addr);
}

STATIC_INLINE(void)
pg_datum_ref_slot(kern_context *kcxt,
				  pg_composite_t &result,
				  cl_char dclass, Datum datum)
{
	if (dclass == DATUM_CLASS__NULL)
		result = pg_composite_datum_ref(kcxt, NULL);
	else if (dclass == DATUM_CLASS__COMPOSITE)
		memcpy(&result, DatumGetPointer(datum), sizeof(pg_composite_t));
	else
	{
		assert(dclass == DATUM_CLASS__NORMAL);
		result = pg_composite_datum_ref(kcxt, DatumGetPointer(datum));
	}
}

STATIC_INLINE(void)
pg_datum_fetch_arrow(kern_context *kcxt,
					 pg_composite_t &result,
					 kern_colmeta *cmeta,
					 char *base, cl_uint rowidx)
{
	kern_data_store *kds = (kern_data_store *)base;

	assert(kds->format == KDS_FORMAT_ARROW);
	assert(rowidx < kds->nitems);
	assert(cmeta->idx_subattrs + cmeta->num_subattrs <= kds->nr_colmeta);
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
	result.value  = base;
	result.isnull = false;
	result.length = cmeta->num_subattrs;
	result.rowidx = rowidx;
	result.comp_typid = cmeta->atttypid;
	result.comp_typmod = cmeta->atttypmod;
	result.smeta  = &kds->colmeta[cmeta->idx_subattrs];
}

STATIC_INLINE(cl_int)
pg_datum_store(kern_context *kcxt,
               pg_composite_t datum,
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
		pg_composite_t *temp;

		temp = (pg_composite_t *)
			kern_context_alloc(kcxt, sizeof(pg_composite_t));
		if (temp)
		{
			memcpy(temp, &datum, sizeof(pg_composite_t));
			dclass = DATUM_CLASS__COMPOSITE;
			value  = PointerGetDatum(temp);
			return sizeof(pg_composite_t);
		}
		STROM_SET_ERROR(&kcxt->e, StromError_OutOfMemory);
		dclass = DATUM_CLASS__NULL;
	}
	return 0;
}

STATIC_INLINE(pg_composite_t)
pg_composite_param(kern_context *kcxt, cl_uint param_id)
{
	kern_parambuf  *kparams = kcxt->kparams;
	pg_composite_t	result;

	if (param_id < kparams->nparams &&
		kparams->poffset[param_id] > 0)
	{
		char   *addr = (char *)kparams + kparams->poffset[param_id];

		if (VARATT_IS_4B_U(addr) || VARATT_IS_1B(addr))
			result = pg_composite_datum_ref(kcxt, addr);
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
pg_comp_hash(kern_context *kcxt, pg_composite_t datum)
{
	/* we don't support to use pg_composite_t for JOIN/GROUP BY key */
	STROM_SET_ERROR(&kcxt->e, StromError_WrongCodeGeneration);
	return 0;
}
#endif	/* PGSTROM_KERNEL_HAS_PGCOMPOSITE */
#endif	/* PG_COMPOSITE_TYPE_DEFINED */

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
/* cuda_varchar.h */
#ifdef PG_BYTEA_TYPE_DEFINED
	pg_bytea_t			bytea_v;
#endif
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
#ifdef PG_COMPOSITE_TYPE_DEFINED
	pg_composite_t		composite_v;
#endif
} pg_anytype_t;
#endif	/* __CUDACC__ */
#endif	/* CUDA_ANYTYPE_H */

#ifdef PGSTROM_KERNEL_HAS_PGARRAY
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
#ifdef PG_BYTEA_TYPE_DEFINED
		case PG_BYTEAOID:
			sz = pg_bytea_array_from_arrow(kcxt,dest,smeta,base,start,end);
			break;
#endif
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
		default:
			STROM_SET_ERROR(&kcxt->e, StromError_WrongCodeGeneration);
			return 0;
	}
	return sz;
}

STATIC_INLINE(cl_uint)
pg_array_datum_length(kern_context *kcxt, Datum datum)
{
    return __pg_array_from_arrow(kcxt, NULL, datum);
}

STATIC_FUNCTION(cl_uint)
pg_array_datum_write(kern_context *kcxt, char *dest, Datum datum)
{
    return __pg_array_from_arrow(kcxt, dest, datum);
}
#endif	/* PGSTROM_KERNEL_HAS_PGARRAY */

#ifdef PGSTROM_KERNEL_HAS_PGCOMPOSITE
/*
 * functions to write out Arrow::Struct as a composite datum of PostgreSQL
 *
 * only called if dclass == DATUM_CLASS__COMPOSITE
 */
STATIC_FUNCTION(void)
__pg_composite_from_arrow(kern_context *kcxt,
						  pg_composite_t *comp,
						  cl_char *tup_dclass,
						  Datum *tup_values)
{
	char	   *base = comp->value;
	cl_uint		j, nfields = comp->length;
	cl_uint		rowidx = comp->rowidx;

	for (j=0; j < nfields; j++)
	{
		kern_colmeta   *smeta = comp->smeta + j;
		pg_anytype_t	temp;

		if (smeta->atttypkind == TYPE_KIND__COMPOSITE)
		{
			pg_datum_fetch_arrow(kcxt, temp.composite_v,
								 smeta, base, rowidx);
			pg_datum_store(kcxt, temp.composite_v,
						   tup_dclass[j],
						   tup_values[j]);
		}
#ifdef PGSTROM_KERNEL_HAS_PGARRAY
		else if (smeta->atttypkind == TYPE_KIND__ARRAY)
		{
			pg_datum_fetch_arrow(kcxt, temp.array_v,
								 smeta, base, rowidx);
			pg_datum_store(kcxt, temp.array_v,
						   tup_dclass[j],
						   tup_values[j]);
		}
#endif
		else if (smeta->atttypkind == TYPE_KIND__BASE)
		{
			switch (smeta->atttypid)
			{
				case PG_BOOLOID:
					pg_datum_fetch_arrow(kcxt, temp.bool_v,
										 smeta, base, rowidx);
					pg_datum_store(kcxt, temp.bool_v,
								   tup_dclass[j],
								   tup_values[j]);
					break;
				case PG_INT2OID:
					pg_datum_fetch_arrow(kcxt, temp.int2_v,
										 smeta, base, rowidx);
					pg_datum_store(kcxt, temp.int2_v,
								   tup_dclass[j],
								   tup_values[j]);
					break;
				case PG_INT4OID:
					pg_datum_fetch_arrow(kcxt, temp.int4_v,
										 smeta, base, rowidx);
					pg_datum_store(kcxt, temp.int4_v,
								   tup_dclass[j],
								   tup_values[j]);
					break;
				case PG_INT8OID:

					pg_datum_fetch_arrow(kcxt, temp.int8_v,
										 smeta, base, rowidx);
					pg_datum_store(kcxt, temp.int8_v,
								   tup_dclass[j],
								   tup_values[j]);
					break;
				case PG_FLOAT2OID:
					pg_datum_fetch_arrow(kcxt, temp.float2_v,
										 smeta, base, rowidx);
					pg_datum_store(kcxt, temp.float2_v,
								   tup_dclass[j],
								   tup_values[j]);
					break;
				case PG_FLOAT4OID:
					pg_datum_fetch_arrow(kcxt, temp.float4_v,
										 smeta, base, rowidx);
					pg_datum_store(kcxt, temp.float4_v,
								   tup_dclass[j],
								   tup_values[j]);
					break;
				case PG_FLOAT8OID:
					pg_datum_fetch_arrow(kcxt, temp.float8_v,
										 smeta, base, rowidx);
					pg_datum_store(kcxt, temp.float8_v,
								   tup_dclass[j],
								   tup_values[j]);
					break;
#ifdef PG_NUMERIC_TYPE_DEFINED
				case PG_NUMERICOID:
					pg_datum_fetch_arrow(kcxt, temp.numeric_v,
										 smeta, base, rowidx);
					pg_datum_store(kcxt, temp.numeric_v,
								   tup_dclass[j],
								   tup_values[j]);
					break;
#endif
#ifdef PG_DATE_TYPE_DEFINED
				case PG_DATEOID:
					pg_datum_fetch_arrow(kcxt, temp.date_v,
										 smeta, base, rowidx);
					pg_datum_store(kcxt, temp.date_v,
								   tup_dclass[j],
								   tup_values[j]);
					break;
#endif
#ifdef PG_TIME_TYPE_DEFINED
				case PG_TIMEOID:
					pg_datum_fetch_arrow(kcxt, temp.time_v,
										 smeta, base, rowidx);
					pg_datum_store(kcxt, temp.time_v,
								   tup_dclass[j],
								   tup_values[j]);
					break;
#endif
#ifdef PG_TIMESTAMP_TYPE_DEFINED
				case PG_TIMESTAMPOID:
					pg_datum_fetch_arrow(kcxt, temp.timestamp_v,
										 smeta, base, rowidx);
					pg_datum_store(kcxt, temp.timestamp_v,
								   tup_dclass[j],
								   tup_values[j]);
					break;
#endif
#ifdef PG_INTERVAL_TYPE_DEFINED
				case PG_INTERVALOID:
					pg_datum_fetch_arrow(kcxt, temp.interval_v,
										 smeta, base, rowidx);
					pg_datum_store(kcxt, temp.interval_v,
								   tup_dclass[j],
								   tup_values[j]);
					break;
#endif
#ifdef PG_BPCHAR_TYPE_DEFINED
				case PG_BPCHAROID:
					pg_datum_fetch_arrow(kcxt, temp.bpchar_v,
										 smeta, base, rowidx);
					pg_datum_store(kcxt, temp.bpchar_v,
								   tup_dclass[j],
								   tup_values[j]);
					break;
#endif
#ifdef PG_TEXT_TYPE_DEFINED
				case PG_TEXTOID:
					pg_datum_fetch_arrow(kcxt, temp.text_v,
										 smeta, base, rowidx);
					pg_datum_store(kcxt, temp.text_v,
								   tup_dclass[j],
								   tup_values[j]);
					break;
#endif
#ifdef PG_VARCHAR_TYPE_DEFINED
				case PG_VARCHAROID:
					pg_datum_fetch_arrow(kcxt, temp.varchar_v,
										 smeta, base, rowidx);
					pg_datum_store(kcxt, temp.varchar_v,
								   tup_dclass[j],
								   tup_values[j]);
					break;
#endif
#ifdef PG_BYTEA_TYPE_DEFINED
				case PG_BYTEAOID:
					pg_datum_fetch_arrow(kcxt, temp.bytea_v,
										 smeta, base, rowidx);
					pg_datum_store(kcxt, temp.bytea_v,
								   tup_dclass[j],
								   tup_values[j]);
					break;
#endif
				default:
					STROM_SET_ERROR(&kcxt->e, StromError_WrongCodeGeneration);
					tup_dclass[j] = DATUM_CLASS__NULL;
			}
		}
	}
}

STATIC_INLINE(cl_uint)
pg_composite_datum_length(kern_context *kcxt, Datum datum)
{
	pg_composite_t *comp = (pg_composite_t *)DatumGetPointer(datum);
	cl_uint		nfields = comp->length;
	cl_char	   *tup_dclass;
	Datum	   *tup_values;
	cl_char	   *vlpos_saved = kcxt->vlpos;
	cl_uint		sz;

	tup_dclass = (cl_char *)
		kern_context_alloc(kcxt, sizeof(cl_char) * nfields);
	tup_values = (Datum *)
		kern_context_alloc(kcxt, sizeof(Datum) * nfields);
	if (!tup_dclass || !tup_values)
	{
		STROM_SET_ERROR(&kcxt->e, StromError_OutOfMemory);
		kcxt->vlpos = vlpos_saved;
		return 0;
	}
	__pg_composite_from_arrow(kcxt, comp, tup_dclass, tup_values);
	sz = __compute_heaptuple_size(kcxt,
								  comp->smeta,
								  false,
								  comp->length,
								  tup_dclass,
								  tup_values);
	kcxt->vlpos = vlpos_saved;
	return sz;
}

STATIC_INLINE(cl_uint)
pg_composite_datum_write(kern_context *kcxt, char *dest, Datum datum)
{
	pg_composite_t *comp = (pg_composite_t *)DatumGetPointer(datum);
	cl_uint		nfields = comp->length;
	cl_char	   *tup_dclass;
	Datum	   *tup_values;
	cl_uint		sz;
	cl_char	   *vlpos_saved = kcxt->vlpos;

	tup_dclass = (cl_char *)
		kern_context_alloc(kcxt, sizeof(cl_char) * nfields);
	tup_values = (Datum *)
		kern_context_alloc(kcxt, sizeof(Datum) * nfields);
	if (!tup_dclass || !tup_values)
	{
		STROM_SET_ERROR(&kcxt->e, StromError_OutOfMemory);
		kcxt->vlpos = vlpos_saved;
		return 0;
	}
	__pg_composite_from_arrow(kcxt, comp, tup_dclass, tup_values);
	sz = form_kern_composite_type(kcxt,
								  dest,
								  comp->comp_typid,
								  comp->comp_typmod,
								  comp->length,
								  comp->smeta,
								  tup_dclass,
								  tup_values);
	kcxt->vlpos = vlpos_saved;
	return sz;
}
#endif	/* PGSTROM_KERNEL_HAS_PGCOMPOSITE */

#ifdef PGSTROM_KERNEL_HAS_PGARRAY
/*
 * ArrayOpExpr support routines
 */
STATIC_INLINE(cl_int)
ArrayGetNItems(kern_context *kcxt, char *array)
{
	cl_int		ndim = ARR_NDIM(array);
	cl_int	   *dims = ARR_DIMS(array);
	cl_int      i, ret;
    cl_long     prod;

	if (ndim <= 0)
		return 0;

	ret = 1;
	for (i=0; i < ndim; i++)
	{
		if (dims[i] < 0)
		{
			/* negative dimension implies an error... */
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
			return 0;
		}
		prod = (cl_long) ret * (cl_long) dims[i];
		ret = (cl_int) prod;
		if ((cl_long) ret != prod)
		{
			/* array size exceeds the maximum allowed... */
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
			return 0;
		}
	}
	assert(ret >= 0);

	return ret;
}

/*
 * Support routine for ScalarArrayOpExpr on fixed-length data type
 *
 * S = scalar type, E = element of array type
 */
template <typename S, typename E>
STATIC_FUNCTION(pg_bool_t)
PG_SIMPLE_SCALAR_ARRAY_OP(kern_context *kcxt,
						  pg_bool_t (*equal_fn)(kern_context *, S, E),
						  S scalar,
						  pg_array_t array,
						  cl_bool useOr)
{
	pg_bool_t	result;
	pg_bool_t	rv;
	cl_uint		i, nitems;
	char	   *pos = NULL;
	char	   *nullmap = NULL;

	/* NULL result for NULL array */
	if (array.isnull)
	{
		result.isnull = true;
		return result;
	}

	/* num of items in the array */
	result.isnull = false;
	result.value = (useOr ? false : true);
	if (array.length >= 0)
		nitems  = array.length;
	else
	{
		nitems  = ArrayGetNItems(kcxt, array.value);
		pos     = ARR_DATA_PTR(array.value);
		nullmap = ARR_NULLBITMAP(array.value);
	}
	if (nitems == 0)
		return result;

	for (i=0; i < nitems; i++)
	{
		E		element;

		if (array.length >= 0)
		{
			pg_datum_fetch_arrow(kcxt, element,
								 array.smeta,
								 array.value, array.start + i);
		}
		else if (nullmap && att_isnull(i, nullmap))
			pg_datum_ref(kcxt, element, NULL);
		else
		{
			pg_datum_ref(kcxt, element, pos);
			pos += sizeof(element.value);
		}
		/* call of the comparison function */
		rv = equal_fn(kcxt, scalar, element);
		if (rv.isnull)
			result.isnull = true;
		else if (useOr)
		{
			if (rv.value)
			{
				result.isnull = false;
				result.value  = true;
				break;
			}
		}
		else
		{
			if (!rv.value)
			{
				result.isnull = false;
				result.value  = false;
				break;
			}
		}
	}
	return result;
}

/*
 * Support routine for ScalarArrayOpExpr on variable-length data type
 *
 * S = scalar type, E = element of array type
 */
template <typename S, typename E>
STATIC_FUNCTION(pg_bool_t)
PG_VARLENA_SCALAR_ARRAY_OP(kern_context *kcxt,
						   pg_bool_t (*equal_fn)(kern_context *, S, E),
						   S scalar,
						   pg_array_t array,
						   cl_bool useOr)
{
	pg_bool_t	result;
	pg_bool_t	rv;
	cl_uint		i, nitems;
	char	   *pos = NULL;
	char	   *nullmap = NULL;

	/* NULL result for NULL array */
	if (array.isnull)
	{
		result.isnull = true;
		return result;
	}

	/* num of items in the array */
	result.isnull = false;
	result.value = (useOr ? false : true);
	if (array.length >= 0)
		nitems  = array.length;
	else
	{
		nitems  = ArrayGetNItems(kcxt, array.value);
		pos     = ARR_DATA_PTR(array.value);
		nullmap = ARR_NULLBITMAP(array.value);
	}
	if (nitems == 0)
		return result;

	for (i=0; i < nitems; i++)
	{
		E		element;

		/* bailout if any error */
		if (kcxt->e.errcode != StromError_Success)
		{
			result.isnull = true;
			return result;
		}

		if (array.length >= 0)
		{
			pg_datum_fetch_arrow(kcxt, element,
								 array.smeta,
								 array.value, array.start + i);
		}
		else if (nullmap && att_isnull(i, nullmap))
			pg_datum_ref(kcxt, element, NULL);
		else
		{
			pos = (char *)INTALIGN(pos);
			pg_datum_ref(kcxt, element, pos);
			if (!element.isnull)
				pos += VARSIZE_ANY(element.value);
		}
		/* call of the comparison function */
		rv = equal_fn(kcxt, scalar, element);
		if (rv.isnull)
			result.isnull = true;
		else if (useOr)
		{
			if (rv.value)
			{
				result.isnull = false;
				result.value  = true;
				break;
			}
		}
		else
		{
			if (!rv.value)
			{
				result.isnull = false;
				result.value  = false;
				break;
			}
		}
	}
	return result;
}
#endif	/* PGSTROM_KERNEL_HAS_PGARRAY */
