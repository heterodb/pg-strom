/*
 * cuda_basetype.h
 *
 * Definition of base device types. "BASE" type means ones required by
 * PG-Strom core, or Apache Arrow support.
 * --
 * Copyright 2011-2020 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2020 (C) The PG-Strom Development Team
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
#ifndef CUDA_BASETYPE_H
#define CUDA_BASETYPE_H

/* Type OID of PostgreSQL for base types */
#define PG_BOOLOID			16
#define PG_INT2OID			21
#define PG_INT4OID			23
#define PG_INT8OID			20
#define PG_FLOAT2OID		421
#define PG_FLOAT4OID		700
#define PG_FLOAT8OID		701
#define PG_NUMERICOID		1700
#define PG_DATEOID			1082
#define PG_TIMEOID			1083
#define PG_TIMESTAMPOID		1114
#define PG_TIMESTAMPTZOID	1184
#define PG_INTERVALOID		1186
#define PG_BPCHAROID		1042
#define PG_TEXTOID			25
#define PG_VARCHAROID		1043
#define PG_BYTEAOID			17

/* ---------------------------------------------------------------
 * Template of variable classes: fixed-length referenced by value
 * ---------------------------------------------------------------
 */
#define STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)					\
	typedef struct {												\
		BASE		value;											\
		cl_bool		isnull;											\
	} pg_##NAME##_t;

#define STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)					\
	typedef struct {											\
		char	   *value;										\
		cl_bool		isnull;										\
		cl_int		length;		/* -1, if PG varlena */			\
	} pg_##NAME##_t;

/* host code use sizeof(pg_varlena_t) */
STROMCL_VARLENA_DATATYPE_TEMPLATE(varlena);

#ifdef __CUDACC__
#define STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE,AS_DATUM)			\
	DEVICE_INLINE(pg_##NAME##_t)									\
	pg_##NAME##_datum_ref(kern_context *kcxt, void *addr)			\
	{																\
		pg_##NAME##_t	result;										\
																	\
		if (!addr)													\
			result.isnull = true;									\
		else														\
		{															\
			result.isnull = false;									\
			memcpy(&result.value, (BASE *)addr, sizeof(BASE));		\
		}															\
		return result;												\
	}																\
	DEVICE_INLINE(void)												\
	pg_datum_ref(kern_context *kcxt,								\
				 pg_##NAME##_t &result, void *addr)					\
	{																\
		result = pg_##NAME##_datum_ref(kcxt, addr);					\
	}																\
	DEVICE_INLINE(void)												\
	pg_datum_ref_slot(kern_context *kcxt,							\
					  pg_##NAME##_t &result,						\
					  cl_char dclass, Datum datum)					\
	{																\
		if (dclass == DATUM_CLASS__NULL)							\
			result = pg_##NAME##_datum_ref(kcxt, NULL);				\
		else														\
		{															\
			assert(dclass == DATUM_CLASS__NORMAL);					\
			result = pg_##NAME##_datum_ref(kcxt, &datum);			\
		}															\
	}																\
	DEVICE_INLINE(pg_##NAME##_t)									\
	pg_##NAME##_param(kern_context *kcxt,cl_uint param_id)			\
	{																\
		kern_parambuf *kparams = kcxt->kparams;						\
		pg_##NAME##_t result;										\
																	\
		if (param_id < kparams->nparams &&							\
			kparams->poffset[param_id] > 0)							\
		{															\
			void   *addr = ((char *)kparams +						\
							kparams->poffset[param_id]);			\
			result = pg_##NAME##_datum_ref(kcxt, addr);				\
		}															\
		else														\
			result.isnull = true;									\
		return result;												\
	}																\
	DEVICE_INLINE(cl_int)											\
	pg_datum_store(kern_context *kcxt,								\
				   pg_##NAME##_t datum,								\
				   cl_char &dclass,									\
				   Datum &value)									\
	{																\
		if (datum.isnull)											\
			dclass = DATUM_CLASS__NULL;								\
		else														\
		{															\
			dclass = DATUM_CLASS__NORMAL;							\
			value = AS_DATUM(datum.value);							\
		}															\
		return 0;													\
	}

/* ---------------------------------------------------------------
 * Template of variable classes: fixed-length referenced by pointer
 * ----------------------------------------------------------------
 */
#define STROMCL_INDIRECT_VARREF_TEMPLATE(NAME,BASE)					\
	DEVICE_INLINE(pg_##NAME##_t)									\
	pg_##NAME##_datum_ref(kern_context *kcxt, void *addr)			\
	{																\
		pg_##NAME##_t	result;										\
																	\
		if (!addr)													\
			result.isnull = true;									\
		else														\
		{															\
			result.isnull = false;									\
			memcpy(&result.value, (BASE *)addr, sizeof(BASE));		\
		}															\
		return result;												\
	}																\
	DEVICE_INLINE(void)												\
	pg_datum_ref(kern_context *kcxt,								\
				 pg_##NAME##_t &result, void *addr)					\
	{																\
		result = pg_##NAME##_datum_ref(kcxt, addr);					\
	}																\
	DEVICE_INLINE(void)												\
	pg_datum_ref_slot(kern_context *kcxt,							\
					  pg_##NAME##_t &result,						\
					  cl_char dclass, Datum datum)					\
	{																\
		if (dclass == DATUM_CLASS__NULL)							\
			result = pg_##NAME##_datum_ref(kcxt, NULL);				\
		else														\
		{															\
			assert(dclass == DATUM_CLASS__NORMAL);					\
			result = pg_##NAME##_datum_ref(kcxt, (char *)datum);	\
		}															\
	}																\
	DEVICE_INLINE(pg_##NAME##_t)									\
	pg_##NAME##_param(kern_context *kcxt,cl_uint param_id)			\
	{																\
		kern_parambuf *kparams = kcxt->kparams;						\
		pg_##NAME##_t result;										\
																	\
		if (param_id < kparams->nparams &&							\
			kparams->poffset[param_id] > 0)							\
		{															\
			void   *addr = ((char *)kparams +						\
							kparams->poffset[param_id]);			\
			result = pg_##NAME##_datum_ref(kcxt, addr);				\
		}															\
		else														\
			result.isnull = true;									\
		return result;												\
	}																\
	DEVICE_INLINE(cl_int)											\
	pg_datum_store(kern_context *kcxt,								\
				   pg_##NAME##_t datum,								\
				   cl_char &dclass,									\
				   Datum &value)									\
	{																\
		void	   *res;											\
																	\
		if (datum.isnull)											\
		{															\
			dclass = DATUM_CLASS__NULL;								\
			return 0;												\
		}															\
		res = kern_context_alloc(kcxt, sizeof(BASE));				\
		if (!res)													\
		{															\
			dclass = DATUM_CLASS__NULL;								\
			STROM_CPU_FALLBACK(kcxt, ERRCODE_OUT_OF_MEMORY,			\
							   "out of memory");					\
			return 0;												\
		}															\
		memcpy(res, &datum.value, sizeof(BASE));					\
		dclass = DATUM_CLASS__NORMAL;								\
		value = PointerGetDatum(res);								\
		return sizeof(BASE);										\
	}

/* ---------------------------------------------------------------
 * Template of variable classes: varlena of PostgreSQL
 * ----------------------------------------------------------------
 */
#define STROMCL_VARLENA_VARREF_TEMPLATE(NAME)					\
	DEVICE_INLINE(pg_##NAME##_t)								\
	pg_##NAME##_datum_ref(kern_context *kcxt,					\
						  void *addr)							\
	{															\
		pg_##NAME##_t result;									\
																\
		if (!addr)												\
			result.isnull = true;								\
		else													\
		{														\
			result.isnull = false;								\
			result.length = -1;									\
			result.value = (char *)addr;						\
		}														\
		return result;											\
	}															\
	DEVICE_INLINE(void)											\
	pg_datum_ref(kern_context *kcxt,							\
				 pg_##NAME##_t &result, void *addr)				\
	{															\
		result = pg_##NAME##_datum_ref(kcxt, addr);				\
	}															\
	DEVICE_INLINE(void)											\
	pg_datum_ref_slot(kern_context *kcxt,						\
					  pg_##NAME##_t &result,					\
					  cl_char dclass, Datum datum)				\
	{															\
		if (dclass == DATUM_CLASS__NULL)						\
			result = pg_##NAME##_datum_ref(kcxt, NULL);			\
		else if (dclass == DATUM_CLASS__VARLENA)				\
			memcpy(&result, DatumGetPointer(datum), sizeof(result));	\
		else													\
		{														\
			assert(dclass == DATUM_CLASS__NORMAL);				\
			result = pg_##NAME##_datum_ref(kcxt, (char *)datum); \
		}														\
	}															\
	DEVICE_INLINE(cl_int)										\
	pg_datum_store(kern_context *kcxt,							\
				   pg_##NAME##_t datum,							\
				   cl_char &dclass,								\
				   Datum &value)								\
	{															\
		if (datum.isnull)										\
			dclass = DATUM_CLASS__NULL;							\
		else if (datum.length < 0)								\
		{														\
			cl_uint		len = VARSIZE_ANY(datum.value);			\
																\
			dclass = DATUM_CLASS__NORMAL;						\
			value  = PointerGetDatum(datum.value);				\
			if (PTR_ON_VLBUF(kcxt, datum.value, len))			\
				return len;										\
		}														\
		else													\
		{														\
			pg_##NAME##_t  *vl_buf;								\
																\
			vl_buf = (pg_##NAME##_t *)							\
				kern_context_alloc(kcxt, sizeof(pg_##NAME##_t));\
			if (vl_buf)											\
			{													\
				memcpy(vl_buf, &datum, sizeof(pg_##NAME##_t));	\
				dclass = DATUM_CLASS__VARLENA;					\
				value  = PointerGetDatum(vl_buf);				\
				return sizeof(pg_##NAME##_t);					\
			}													\
			dclass = DATUM_CLASS__NULL;							\
			STROM_CPU_FALLBACK(kcxt, ERRCODE_OUT_OF_MEMORY,		\
							   "out of memory");				\
		}														\
		return 0;												\
	}															\
	DEVICE_INLINE(pg_##NAME##_t)								\
	pg_##NAME##_param(kern_context *kcxt, cl_uint param_id)		\
	{															\
		kern_parambuf  *kparams = kcxt->kparams;				\
		pg_##NAME##_t	result;									\
																\
		if (param_id < kparams->nparams &&						\
			kparams->poffset[param_id] > 0)						\
		{														\
			char	   *vl_val = ((char *)kparams +				\
								  kparams->poffset[param_id]);	\
			if (VARATT_IS_4B_U(vl_val) || VARATT_IS_1B(vl_val))	\
			{													\
				result.value = vl_val;							\
				result.length = -1;								\
				result.isnull = false;							\
			}													\
			else												\
			{													\
				result.isnull = true;							\
				STROM_CPU_FALLBACK(kcxt, ERRCODE_STROM_VARLENA_UNSUPPORTED,	\
							"varlena datum is compressed or external"); \
			}													\
		}														\
		else													\
			result.isnull = true;								\
																\
		return result;											\
	}

/*
 * pg_varlena_datum_extract - extract datum body and length from pg_varlena_t,
 * or its compatible types. PostgreSQL embeds length of variable-length field
 * at the header, however, Arrow has separated length field. This interface
 * enables to extract pointer and length regardless of the physical layout.
 */
template <typename T>
DEVICE_INLINE(cl_bool)
pg_varlena_datum_extract(kern_context *kcxt, T &arg,
						 char **s, cl_int *len)
{
	if (arg.isnull)
		return false;
	if (arg.length < 0)
	{
		if (VARATT_IS_COMPRESSED(arg.value) ||
			VARATT_IS_EXTERNAL(arg.value))
        {
			STROM_CPU_FALLBACK(kcxt, ERRCODE_STROM_VARLENA_UNSUPPORTED,
							   "compressed or external varlena on device");
			return false;
        }
		*s = VARDATA_ANY(arg.value);
		*len = VARSIZE_ANY_EXHDR(arg.value);
	}
	else
	{
		*s = arg.value;
		*len = arg.length;
	}
	return true;
}

/* ---------------------------------------------------------------
 * Template of other special data structure defined at library
 * ---------------------------------------------------------------- */
#define STROMCL_EXTERNAL_VARREF_TEMPLATE(NAME)					\
	DEVICE_FUNCTION(pg_##NAME##_t)								\
	pg_##NAME##_datum_ref(kern_context *kcxt, void *addr);		\
	DEVICE_FUNCTION(void)										\
	pg_datum_ref(kern_context *kcxt,							\
				 pg_##NAME##_t &result, void *addr);			\
	DEVICE_FUNCTION(void)										\
	pg_datum_ref_slot(kern_context *kcxt,						\
					  pg_##NAME##_t &result,					\
					  cl_char dclass, Datum datum);				\
	DEVICE_FUNCTION(pg_##NAME##_t)								\
	pg_##NAME##_param(kern_context *kcxt,						\
					  cl_uint param_id);						\
	DEVICE_FUNCTION(cl_int)										\
	pg_datum_store(kern_context *kcxt,							\
				   pg_##NAME##_t datum,							\
				   cl_char &dclass,								\
				   Datum &value);

/*
 * General purpose hash-function which is compatible to PG's hash_any()
 * These are basis of Hash-Join and Group-By reduction.
 */
#define STROMCL_SIMPLE_COMP_HASH_TEMPLATE(NAME,BASE)			\
	DEVICE_INLINE(cl_uint)										\
	pg_comp_hash(kern_context *kcxt, pg_##NAME##_t datum)		\
	{															\
		if (datum.isnull)										\
			return 0;											\
		return pg_hash_any((cl_uchar *)&datum.value, sizeof(BASE));	\
	}
#define STROMCL_VARLENA_COMP_HASH_TEMPLATE(NAME)				\
	DEVICE_INLINE(cl_uint)											\
	pg_comp_hash(kern_context *kcxt, pg_##NAME##_t datum)		\
	{                                                           \
		if (datum.isnull)										\
			return 0;											\
		if (datum.length >= 0)									\
			return pg_hash_any((cl_uchar *)datum.value,			\
							   datum.length);					\
		if (VARATT_IS_COMPRESSED(datum.value) ||                \
			VARATT_IS_EXTERNAL(datum.value))					\
		{														\
			STROM_CPU_FALLBACK(kcxt, ERRCODE_STROM_VARLENA_UNSUPPORTED,	\
							   "compressed or external varlena on device");	\
			return 0;											\
		}														\
		return pg_hash_any((cl_uchar *)VARDATA_ANY(datum.value), \
						   VARSIZE_ANY_EXHDR(datum.value));		\
	}
#define STROMCL_EXTERNAL_COMP_HASH_TEMPLATE(NAME)				\
	DEVICE_FUNCTION(cl_uint)									\
	pg_comp_hash(kern_context *kcxt, pg_##NAME##_t datum);

#define STROMCL_SIMPLE_TYPE_TEMPLATE(NAME,BASE,AS_DATUM)	\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)				\
	STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE,AS_DATUM)

#define STROMCL_INDIRECT_TYPE_TEMPLATE(NAME,BASE)			\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)				\
	STROMCL_INDIRECT_VARREF_TEMPLATE(NAME,BASE)

#define STROMCL_VARLENA_TYPE_TEMPLATE(NAME)					\
	STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)					\
	STROMCL_VARLENA_VARREF_TEMPLATE(NAME)

/*
 * Template of Reference to Arrow values without any transformation
 */
#define STROMCL_SIMPLE_ARROW_TEMPLATE(NAME,BASE)			\
	DEVICE_INLINE(void)										\
	pg_datum_fetch_arrow(kern_context *kcxt,				\
						 pg_##NAME##_t &result,				\
						 kern_colmeta *cmeta,				\
						 char *base, cl_uint rowidx)		\
	{														\
		void		   *addr;								\
															\
		addr = kern_fetch_simple_datum_arrow(cmeta,			\
											 base,			\
											 rowidx,		\
											 sizeof(BASE));	\
		if (!addr)											\
			result.isnull = true;							\
		else												\
		{													\
			result.isnull = false;							\
			result.value  = *((BASE *)addr);				\
		}													\
	}
#define STROMCL_VARLENA_ARROW_TEMPLATE(NAME)				\
	DEVICE_INLINE(void)										\
	pg_datum_fetch_arrow(kern_context *kcxt,				\
						 pg_##NAME##_t &result,             \
						 kern_colmeta *cmeta,				\
						 char *base, cl_uint rowidx)		\
	{														\
		void           *addr;                               \
		cl_uint			length;								\
                                                            \
		addr = kern_fetch_varlena_datum_arrow(cmeta,		\
											  base,			\
											  rowidx,		\
											  &length);		\
		if (!addr)											\
		{													\
			result.isnull = true;							\
			return;											\
		}													\
		result.isnull = false;								\
		result.value  = (char *)addr;						\
		result.length = length;								\
	}
#define STROMCL_EXTERNAL_ARROW_TEMPLATE(NAME)				\
	DEVICE_FUNCTION(void)									\
	pg_datum_fetch_arrow(kern_context *kcxt,				\
						 pg_##NAME##_t &result,				\
						 kern_colmeta *cmeta,				\
						 char *base, cl_uint rowidx);
#define STROMCL_UNSUPPORTED_ARROW_TEMPLATE(NAME)			\
	DEVICE_INLINE(void)										\
	pg_datum_fetch_arrow(kern_context *kcxt,				\
						 pg_##NAME##_t &result,				\
						 kern_colmeta *cmeta,				\
						 char *base, cl_uint rowidx)		\
	{														\
		result.isnull = true;								\
		STROM_EREPORT(kcxt, ERRCODE_STROM_WRONG_CODE_GENERATION,\
					  "wrong code generation");				\
	}

/*
 * A common interface to reference scalar value on KDS_FORMAT_ARROW
 */
template <typename T>
DEVICE_INLINE(void)
pg_datum_ref_arrow(kern_context *kcxt,
				   T &result,
				   kern_data_store *kds,
				   cl_uint colidx, cl_uint rowidx)
{
	kern_colmeta   *cmeta = &kds->colmeta[colidx];

	assert(kds->format == KDS_FORMAT_ARROW);
	assert(colidx < kds->nr_colmeta &&
		   rowidx < kds->nitems);
	pg_datum_fetch_arrow(kcxt, result, cmeta,
						 (char *)kds, rowidx);
}
#else  /* __CUDACC__ */
#define STROMCL_SIMPLE_VARREF_TEMPLATE(NAME,BASE,AS_DATUM)
#define STROMCL_INDIRECT_VARREF_TEMPLATE(NAME,BASE)
#define STROMCL_VARLENA_VARREF_TEMPLATE(NAME)
#define STROMCL_EXTERNAL_VARREF_TEMPLATE(NAME)

#define STROMCL_SIMPLE_COMP_HASH_TEMPLATE(NAME,BASE)
#define STROMCL_VARLENA_COMP_HASH_TEMPLATE(NAME)
#define STROMCL_EXTERNAL_COMP_HASH_TEMPLATE(NAME)

#define STROMCL_SIMPLE_TYPE_TEMPLATE(NAME,BASE,AS_DATUM) \
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,BASE)
#define STROMCL_INDIRECT_TYPE_TEMPLATE(NAME,BASE) \
	STROMCL_INDIRECT_DATATYPE_TEMPLATE(NAME,BASE)
#define STROMCL_VARLENA_TYPE_TEMPLATE(NAME) \
	STROMCL_VARLENA_DATATYPE_TEMPLATE(NAME)

#define STROMCL_SIMPLE_ARROW_TEMPLATE(NAME,BASE)
#define STROMCL_VARLENA_ARROW_TEMPLATE(NAME)
#define STROMCL_EXTERNAL_ARROW_TEMPLATE(NAME)
#define STROMCL_UNSUPPORTED_ARROW_TEMPLATE(NAME)
#endif /* !__CUDACC__ */

#ifdef __CUDACC__
#define __STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,OPER,EXTRA) \
	DEVICE_INLINE(pg_bool_t)								\
	pgfn_##FNAME##EXTRA(kern_context *kcxt,					\
						pg_##LNAME##_t arg1,				\
						pg_##RNAME##_t arg2)				\
	{														\
		pg_bool_t result;									\
															\
		result.isnull = arg1.isnull | arg2.isnull;			\
		if (!result.isnull)									\
			result.value = ((CAST)arg1.value OPER			\
							(CAST)arg2.value);				\
		return result;										\
	}
#define STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST)		\
	__STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,==,eq)	\
	__STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,!=,ne)	\
	__STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,<, lt)	\
	__STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,<=,le)	\
	__STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,>, gt)	\
	__STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,>=,ge)
#else  /* __CUDACC__ */
#define __STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,OPER,EXTRA)
#define STROMCL_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST)
#endif /* !__CUDACC__ */

/* pg_bool_t */
#ifndef PG_BOOL_TYPE_DEFINED
#define PG_BOOL_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(bool, cl_bool, )
STROMCL_EXTERNAL_ARROW_TEMPLATE(bool)
STROMCL_SIMPLE_COMP_HASH_TEMPLATE(bool, cl_bool)
#endif	/* PG_BOOL_TYPE_DEFINED */

/* pg_int2_t */
#ifndef PG_INT2_TYPE_DEFINED
#define PG_INT2_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int2, cl_short, )
STROMCL_EXTERNAL_COMP_HASH_TEMPLATE(int2)
STROMCL_SIMPLE_ARROW_TEMPLATE(int2, cl_short)
#endif	/* PG_INT2_TYPE_DEFINED */

/* pg_int4_t */
#ifndef PG_INT4_TYPE_DEFINED
#define PG_INT4_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int4, cl_int, )
STROMCL_EXTERNAL_COMP_HASH_TEMPLATE(int4)
STROMCL_SIMPLE_ARROW_TEMPLATE(int4, cl_int)
#endif	/* PG_INT4_TYPE_DEFINED */

/* pg_int8_t */
#ifndef PG_INT8_TYPE_DEFINED
#define PG_INT8_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(int8, cl_long, )
STROMCL_EXTERNAL_COMP_HASH_TEMPLATE(int8)
STROMCL_SIMPLE_ARROW_TEMPLATE(int8, cl_long)
#endif	/* PG_INT8_TYPE_DEFINED */

/* pg_float2_t */
#ifndef PG_FLOAT2_TYPE_DEFINED
#define PG_FLOAT2_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(float2, cl_half, __half_as_short)
STROMCL_EXTERNAL_COMP_HASH_TEMPLATE(float2)
STROMCL_SIMPLE_ARROW_TEMPLATE(float2, cl_half)
#endif	/* PG_FLOAT2_TYPE_DEFINED */

/* pg_float4_t */
#ifndef PG_FLOAT4_TYPE_DEFINED
#define PG_FLOAT4_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(float4, cl_float, __float_as_int)
STROMCL_EXTERNAL_COMP_HASH_TEMPLATE(float4)
STROMCL_SIMPLE_ARROW_TEMPLATE(float4, cl_float)
#endif	/* PG_FLOAT4_TYPE_DEFINED */

/* pg_float8_t */
#ifndef PG_FLOAT8_TYPE_DEFINED
#define PG_FLOAT8_TYPE_DEFINED
STROMCL_SIMPLE_TYPE_TEMPLATE(float8, cl_double, __double_as_longlong)
STROMCL_EXTERNAL_COMP_HASH_TEMPLATE(float8)
STROMCL_SIMPLE_ARROW_TEMPLATE(float8, cl_double)
#endif	/* PG_FLOAT8_TYPE_DEFINED */

#ifdef __CUDACC__
#define PG_SIMPLE_TYPECAST_TEMPLATE(TARGET,SOURCE,CAST)	\
	DEVICE_INLINE(pg_##TARGET##_t)						\
	pgfn_to_##TARGET(kern_context *kcxt,				\
					 pg_##SOURCE##_t arg)				\
	{													\
		pg_##TARGET##_t result;							\
														\
		result.isnull = arg.isnull;						\
		if (!result.isnull)								\
			result.value = CAST(arg.value);				\
		return result;									\
	}

DEVICE_INLINE(pg_bool_t)
pgfn_to_bool(kern_context *kcxt, pg_int4_t arg)
{
	pg_bool_t	result;

	result.isnull = arg.isnull;
	if (!result.isnull)
		result.value = (arg.value == 0 ? false : true);
	return result;
}
PG_SIMPLE_TYPECAST_TEMPLATE(int2,int4,(cl_short))
PG_SIMPLE_TYPECAST_TEMPLATE(int2,int8,(cl_short))
PG_SIMPLE_TYPECAST_TEMPLATE(int2,float2,lrintf)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,float4,lrintf)
PG_SIMPLE_TYPECAST_TEMPLATE(int2,float8,llrint)

PG_SIMPLE_TYPECAST_TEMPLATE(int4,int2,(cl_int))
PG_SIMPLE_TYPECAST_TEMPLATE(int4,int8,(cl_int))
PG_SIMPLE_TYPECAST_TEMPLATE(int4,float2,lrintf)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,float4,lrintf)
PG_SIMPLE_TYPECAST_TEMPLATE(int4,float8,llrint)

PG_SIMPLE_TYPECAST_TEMPLATE(int8,int2,(cl_long))
PG_SIMPLE_TYPECAST_TEMPLATE(int8,int4,(cl_long))
PG_SIMPLE_TYPECAST_TEMPLATE(int8,float2,lrintf)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,float4,lrintf)
PG_SIMPLE_TYPECAST_TEMPLATE(int8,float8,llrint)

PG_SIMPLE_TYPECAST_TEMPLATE(float2,int2,(cl_half))
PG_SIMPLE_TYPECAST_TEMPLATE(float2,int4,(cl_half))
PG_SIMPLE_TYPECAST_TEMPLATE(float2,int8,(cl_half))
PG_SIMPLE_TYPECAST_TEMPLATE(float2,float4,(cl_half))
PG_SIMPLE_TYPECAST_TEMPLATE(float2,float8,(cl_half))

PG_SIMPLE_TYPECAST_TEMPLATE(float4,int2,(cl_float))
PG_SIMPLE_TYPECAST_TEMPLATE(float4,int4,(cl_float))
PG_SIMPLE_TYPECAST_TEMPLATE(float4,int8,(cl_float))
PG_SIMPLE_TYPECAST_TEMPLATE(float4,float2,(cl_float))
PG_SIMPLE_TYPECAST_TEMPLATE(float4,float8,(cl_float))

PG_SIMPLE_TYPECAST_TEMPLATE(float8,int2,(cl_double))
PG_SIMPLE_TYPECAST_TEMPLATE(float8,int4,(cl_double))
PG_SIMPLE_TYPECAST_TEMPLATE(float8,int8,(cl_double))
PG_SIMPLE_TYPECAST_TEMPLATE(float8,float2,(cl_double))
PG_SIMPLE_TYPECAST_TEMPLATE(float8,float4,(cl_double))
#undef PG_SIMPLE_TYPECAST_TEMPLATE
#endif /* __CUDACC__ */

/*
 * Simple comparison operators between intX/floatX
 */
__STROMCL_SIMPLE_COMPARE_TEMPLATE(bool,  bool,   bool,   cl_char, ==, eq)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int2,    int2,   int2,   cl_short)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int24,   int2,   int4,   cl_int)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int28,   int2,   int8,   cl_long)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int42,   int4,   int2,   cl_int)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int4,    int4,   int4,   cl_int)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int48,   int4,   int8,   cl_long)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int82,   int8,   int2,   cl_long)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int84,   int8,   int4,   cl_long)
STROMCL_SIMPLE_COMPARE_TEMPLATE(int8,    int8,   int8,   cl_long)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float2,  float2, float2, cl_float)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float24, float2, float4, cl_float)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float28, float2, float8, cl_double)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float42, float4, float2, cl_float)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float4,  float4, float4, cl_float)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float48, float4, float8, cl_double)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float82, float8, float2, cl_double)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float84, float8, float4, cl_double)
STROMCL_SIMPLE_COMPARE_TEMPLATE(float8,  float8, float8, cl_double)

/* pg_bytea_t */
#ifndef PG_BYTEA_TYPE_DEFINED
#define PG_BYTEA_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(bytea)
STROMCL_VARLENA_COMP_HASH_TEMPLATE(bytea)
STROMCL_VARLENA_ARROW_TEMPLATE(bytea)
#endif	/* PG_BYTEA_TYPE_DEFINED */

/* pg_text_t */
#ifndef PG_TEXT_TYPE_DEFINED
#define PG_TEXT_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(text)
STROMCL_VARLENA_COMP_HASH_TEMPLATE(text)
STROMCL_VARLENA_ARROW_TEMPLATE(text)
#endif	/* PG_TEXT_TYPE_DEFINED */

/* pg_varchar_t */
#ifndef PG_VARCHAR_TYPE_DEFINED
#define PG_VARCHAR_TYPE_DEFINED
typedef pg_text_t						pg_varchar_t;
#define pg_varchar_param(a,b)			pg_text_param(a,b)
#endif

/* pg_bpchar_t */
#ifndef PG_BPCHAR_TYPE_DEFINED
#define PG_BPCHAR_TYPE_DEFINED
STROMCL_VARLENA_TYPE_TEMPLATE(bpchar)
STROMCL_EXTERNAL_COMP_HASH_TEMPLATE(bpchar)
STROMCL_EXTERNAL_ARROW_TEMPLATE(bpchar)
#endif	/* PG_BPCHAR_TYPE_DEFINED */

/* pg_array_t */
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
STROMCL_EXTERNAL_VARREF_TEMPLATE(array)
STROMCL_EXTERNAL_COMP_HASH_TEMPLATE(array)
STROMCL_EXTERNAL_ARROW_TEMPLATE(array)
#endif

/* pg_composite_t */
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
STROMCL_EXTERNAL_VARREF_TEMPLATE(composite)
STROMCL_EXTERNAL_COMP_HASH_TEMPLATE(composite)
STROMCL_EXTERNAL_ARROW_TEMPLATE(composite)
#endif

/*
 * handler functions for DATUM_CLASS__(VARLENA|ARRAY|COMPOSITE)
 */
#ifdef __CUDACC__
DEVICE_FUNCTION(cl_uint)
pg_varlena_datum_length(kern_context *kcxt, Datum datum);
DEVICE_FUNCTION(cl_uint)
pg_varlena_datum_write(kern_context *kcxt, char *dest, Datum datum);

DEVICE_FUNCTION(cl_uint)
pg_array_datum_length(kern_context *kcxt, Datum datum);
DEVICE_FUNCTION(cl_uint)
pg_array_datum_write(kern_context *kcxt, char *dest, Datum datum);

DEVICE_FUNCTION(cl_uint)
pg_composite_datum_length(kern_context *kcxt, Datum datum);
DEVICE_FUNCTION(cl_uint)
pg_composite_datum_write(kern_context *kcxt, char *dest, Datum datum);
#endif /* __CUDACC__ */
#endif /* CUDA_BASETYPE_H */
