/*
 * cuda_rangetype.cu
 *
 * Routines to support range data types for CUDA GPU devices
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "cuda_common.h"
#include "cuda_rangetype.h"

/* A range's flags byte contains these bits: */
#define RANGE_EMPTY			0x01	/* range is empty */
#define RANGE_LB_INC		0x02	/* lower bound is inclusive */
#define RANGE_UB_INC		0x04	/* upper bound is inclusive */
#define RANGE_LB_INF		0x08	/* lower bound is -infinity */
#define RANGE_UB_INF		0x10	/* upper bound is +infinity */
#define RANGE_LB_NULL		0x20	/* lower bound is null (NOT USED) */
#define RANGE_UB_NULL		0x40	/* upper bound is null (NOT USED) */
#define RANGE_CONTAIN_EMPTY 0x80	/* marks a GiST internal-page entry */

#define RANGE_HAS_LBOUND(flags) (!((flags) & (RANGE_EMPTY |		\
											  RANGE_LB_NULL |	\
											  RANGE_LB_INF)))

#define RANGE_HAS_UBOUND(flags) (!((flags) & (RANGE_EMPTY |		\
											  RANGE_UB_NULL |	\
											  RANGE_UB_INF)))

/*
 * Basic type reference/store APIs
 */
#define PG_RANGETYPE_TEMPLATE(NAME,BASE,PG_TYPEOID,AS_DATUM64)		\
	DEVICE_FUNCTION(pg_##NAME##_t)									\
	pg_##NAME##_datum_ref(kern_context *kcxt, void *addr)			\
	{																\
		pg_##NAME##_t result;										\
		char		flags;											\
		char	   *pos;											\
		cl_uint		type_oid;										\
																	\
		memset(&result, 0, sizeof(result));							\
		if (!addr)													\
		{															\
			result.isnull = true;									\
			return result;											\
		}															\
		if (VARATT_IS_EXTERNAL(addr) ||								\
			VARATT_IS_COMPRESSED(addr))								\
		{															\
			result.isnull = true;									\
			STROM_CPU_FALLBACK(kcxt,ERRCODE_STROM_VARLENA_UNSUPPORTED,\
							   "compressed or external varlena");	\
			return result;											\
		}															\
		flags = *((char *)addr + VARSIZE_ANY(addr) - sizeof(char));	\
		memcpy(&type_oid, VARDATA_ANY(addr), sizeof(cl_uint));		\
		if (type_oid != PG_TYPEOID)									\
		{															\
			result.isnull = true;									\
			STROM_EREPORT(kcxt, ERRCODE_DATA_CORRUPTED,				\
						  "corrupted range type");					\
			return result;											\
		}															\
		pos = VARDATA_ANY(addr) + sizeof(cl_uint);					\
		if (!RANGE_HAS_LBOUND(flags))								\
			result.value.l.val = 0;									\
		else														\
		{															\
			memcpy(&result.value.l.val, pos, sizeof(BASE));			\
			pos += sizeof(BASE);									\
		}															\
		if (!RANGE_HAS_UBOUND(flags))								\
			result.value.u.val = 0;									\
		else														\
		{															\
			memcpy(&result.value.u.val, pos, sizeof(BASE));			\
			pos += sizeof(BASE);									\
		}															\
		result.isnull = false;										\
		result.value.empty = ((flags & RANGE_EMPTY) != 0);			\
		result.value.l.infinite = ((flags & RANGE_LB_INF) != 0);	\
		result.value.l.inclusive = ((flags & RANGE_LB_INC) != 0);	\
		result.value.l.lower = true;								\
		result.value.u.infinite = ((flags & RANGE_UB_INF) != 0);	\
		result.value.u.inclusive = ((flags & RANGE_UB_INC) != 0);	\
		result.value.u.lower = false;								\
		return result;												\
	}																\
	DEVICE_FUNCTION(void)											\
	pg_datum_ref(kern_context *kcxt,								\
				 pg_##NAME##_t &result, void *addr)					\
	{																\
		result = pg_##NAME##_datum_ref(kcxt, addr);					\
	}																\
	DEVICE_FUNCTION(void)											\
	pg_datum_ref_slot(kern_context *kcxt,							\
					  pg_##NAME##_t &result,						\
					  cl_char dclass, Datum datum)					\
	{																\
		if (dclass == DATUM_CLASS__NULL)							\
			result = pg_##NAME##_datum_ref(kcxt, NULL);				\
		else														\
		{															\
			assert(dclass == DATUM_CLASS__NORMAL);					\
			result = pg_##NAME##_datum_ref(kcxt, (void *)datum);	\
		}															\
	}																\
	DEVICE_FUNCTION(cl_int)											\
	pg_datum_store(kern_context *kcxt,								\
				   pg_##NAME##_t datum,								\
				   cl_char &dclass,									\
                   Datum &value)									\
	{																\
		char			flags = 0;									\
		struct range_raw {											\
			cl_uint		vl_len_;									\
			cl_uint		rangetypid;									\
			BASE		l_val;										\
			BASE		u_val;										\
			cl_char		flags;										\
		} *res;														\
																	\
		if (datum.isnull)											\
		{															\
			dclass = DATUM_CLASS__NULL;								\
			return 0;												\
		}															\
		res = (struct range_raw *)									\
			kern_context_alloc(kcxt, sizeof(*res));					\
		if (!res)													\
		{															\
			dclass = DATUM_CLASS__NULL;								\
			STROM_CPU_FALLBACK(kcxt, ERRCODE_OUT_OF_MEMORY,			\
							   "out of memory");					\
			return 0;												\
		}															\
																	\
		flags = ((datum.value.empty ? RANGE_EMPTY : 0) |			\
				 (datum.value.l.infinite  ? RANGE_LB_INF : 0) |		\
				 (datum.value.l.inclusive ? RANGE_LB_INC : 0) |		\
				 (datum.value.u.infinite  ? RANGE_UB_INF : 0) |		\
				 (datum.value.u.inclusive ? RANGE_UB_INC : 0) |		\
				 (datum.value.empty       ? RANGE_EMPTY : 0));		\
		res->rangetypid = PG_TYPEOID;								\
		res->l_val = datum.value.l.val;								\
		res->u_val = datum.value.u.val;								\
		res->flags = flags;											\
		SET_VARSIZE(res, offsetof(struct range_raw,					\
								  flags) + sizeof(cl_char));		\
		dclass = DATUM_CLASS__NORMAL;								\
		value  = PointerGetDatum(res);								\
		return sizeof(*res);										\
	}																\
	DEVICE_FUNCTION(pg_##NAME##_t)									\
	pg_##NAME##_param(kern_context *kcxt,cl_uint param_id)			\
	{																\
		kern_parambuf  *kparams = kcxt->kparams;					\
		void		   *paddr;										\
																	\
		if (param_id < kparams->nparams &&							\
			kparams->poffset[param_id] > 0)							\
			paddr = ((char *)kparams + kparams->poffset[param_id]);	\
		else														\
			paddr = NULL;											\
																	\
		return pg_##NAME##_datum_ref(kcxt,paddr);					\
	}																\
	DEVICE_FUNCTION(cl_uint)										\
	pg_comp_hash(kern_context *kcxt, pg_##NAME##_t datum)			\
	{																\
		cl_char		flags;											\
		struct {													\
			Datum	l_val;											\
			Datum	u_val;											\
			char	flags;											\
		} temp;														\
																	\
		if (datum.isnull)											\
			return 0;												\
		flags = (datum.value.empty ? RANGE_EMPTY : 0)				\
			| (datum.value.l.infinite ? RANGE_LB_INF  : 0)			\
			| (datum.value.l.inclusive ? RANGE_LB_INC : 0)			\
			| (datum.value.u.infinite ? RANGE_UB_INF  : 0)			\
			| (datum.value.u.inclusive ? RANGE_UB_INC : 0);			\
		if (RANGE_HAS_LBOUND(flags))								\
			temp.l_val = AS_DATUM64(datum.value.l.val);				\
		else														\
			temp.l_val = 0;											\
		if (RANGE_HAS_UBOUND(flags))								\
			temp.u_val = AS_DATUM64(datum.value.u.val);				\
		else														\
			temp.u_val = 0;											\
		return pg_hash_any((unsigned char *)&temp,					\
						   2*sizeof(Datum)+sizeof(char));			\
	}

#ifdef PG_INT4RANGE_TYPE_DEFINED
PG_RANGETYPE_TEMPLATE(int4range,cl_int,PG_INT4RANGEOID,(cl_long))
#endif	/* PG_INT4RANGE_TYPE_DEFINED */
#ifdef PG_INT8RANGE_TYPE_DEFINED
PG_RANGETYPE_TEMPLATE(int8range,cl_long,PG_INT8RANGEOID,)
#endif	/* PG_INT4RANGE_TYPE_DEFINED */
#ifdef PG_TSRANGE_TYPE_DEFINED
PG_RANGETYPE_TEMPLATE(tsrange,Timestamp,PG_TSRANGEOID,)
#endif	/* PG_TSRANGE_TYPE_DEFINED */
#ifdef PG_TSTZRANGE_TYPE_DEFINED
PG_RANGETYPE_TEMPLATE(tstzrange,TimestampTz,PG_TSTZRANGEOID,)
#endif	/* PG_TSTZRANGE_TYPE_DEFINED */
#ifdef PG_DATERANGE_TYPE_DEFINED
PG_RANGETYPE_TEMPLATE(daterange,DateADT,PG_DATERANGEOID,(cl_long))
#endif

/*
 * template for various range functions
 */
#define __GENERIC_RANGE_LOWER(ELEMENT,RANGE)				\
	DEVICE_FUNCTION(pg_##ELEMENT##_t)						\
	pgfn_##RANGE##_lower(kern_context *kcxt,				\
						 const pg_##RANGE##_t &arg1)		\
	{														\
		pg_##ELEMENT##_t result;							\
															\
		result.isnull = arg1.isnull;						\
		if (!result.isnull)									\
		{													\
			if (arg1.value.empty || arg1.value.l.infinite)	\
				result.isnull = true;						\
			else											\
				result.value = arg1.value.l.val;			\
		}													\
		return result;										\
	}

#define __GENERIC_RANGE_UPPER(ELEMENT,RANGE)				\
	DEVICE_FUNCTION(pg_##ELEMENT##_t)						\
	pgfn_##RANGE##_upper(kern_context *kcxt,				\
						 const pg_##RANGE##_t &arg1)		\
	{														\
		pg_##ELEMENT##_t result;							\
															\
		result.isnull = arg1.isnull;						\
		if (!result.isnull)									\
		{													\
			if (arg1.value.empty || arg1.value.u.infinite)	\
				result.isnull = true;						\
			else											\
				result.value = arg1.value.u.val;			\
		}													\
		return result;										\
	}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_isempty(kern_context *kcxt, const R &arg1)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull;
	if (!result.isnull)
		result.value = arg1.value.empty;
	return result;
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_lower_inc(kern_context *kcxt, const R &arg1)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull;
	if (!result.isnull)
		result.value = arg1.value.l.inclusive;
	return result;
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_upper_inc(kern_context *kcxt, const R &arg1)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull;
	if (!result.isnull)
		result.value = arg1.value.u.inclusive;
	return result;
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_lower_inf(kern_context *kcxt, const R &arg1)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull;
	if (!result.isnull)
		result.value = arg1.value.l.infinite;
	return result;
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_upper_inf(kern_context *kcxt, const R &arg1)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull;
	if (!result.isnull)
		result.value = arg1.value.u.infinite;
	return result;
}

template <typename RangeBound>
STATIC_FUNCTION(int)
__range_cmp_bound_values(const RangeBound *b1, const RangeBound *b2)
{
	if (b1->infinite && b2->infinite)
	{
		if (b1->lower == b2->lower)
			return 0;
		else
			return b1->lower ? -1 : 1;
	}
	else if (b1->infinite)
		return b1->lower ? -1 : 1;
	else if (b2->infinite)
		return b2->lower ? 1 : -1;

	return Compare(b1->val, b2->val);
}

template <typename RangeBound>
STATIC_FUNCTION(int)
__range_cmp_bounds(const RangeBound *b1, const RangeBound *b2)
{
	int		cmp = __range_cmp_bound_values(b1, b2);

	if (cmp == 0)
	{
		if (!b1->inclusive && !b2->inclusive)
		{
			if (b1->lower == b2->lower)
				cmp = 0;
			else
				cmp = (b1->lower ? 1 : -1);
		}
		else if (!b1->inclusive)
			cmp = (b1->lower ? 1 : -1);
        else if (!b2->inclusive)
            cmp = (b2->lower ? -1 : 1);
	}
	return cmp;
}

template <typename RangeType>
STATIC_FUNCTION(int)
__range_cmp(const RangeType *r1, const RangeType *r2)
{
	int		cmp;

	if (r1->empty && r2->empty)
		cmp = 0;
	else if (r1->empty)
		cmp = -1;
	else if (r2->empty)
		cmp = 1;
	else
	{
		cmp = __range_cmp_bounds(&r1->l, &r2->l);
		if (cmp == 0)
			cmp = __range_cmp_bounds(&r1->u, &r2->u);
	}
	return cmp;
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_eq(kern_context *kcxt,
				   const R &arg1, const R &arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg1.value.empty && arg2.value.empty)
			result.value = true;
		else if (arg1.value.empty != arg2.value.empty)
			result.value = false;
		else if (__range_cmp_bounds(&arg1.value.l, &arg2.value.l) != 0)
			result.value = false;
		else if (__range_cmp_bounds(&arg1.value.u, &arg2.value.u) != 0)
			result.value = false;
		else
			result.value = true;
	}
	return result;
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_ne(kern_context *kcxt,
				   const R &arg1, const R &arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg1.value.empty && arg2.value.empty)
			result.value = false;
		else if (arg1.value.empty != arg2.value.empty)
			result.value = true;
		else if (__range_cmp_bounds(&arg1.value.l, &arg2.value.l) != 0)
			result.value = true;
		else if (__range_cmp_bounds(&arg1.value.u, &arg2.value.u) != 0)
			result.value = true;
		else
			result.value = false;
	}
	return result;
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_gt(kern_context *kcxt,
				   const R &arg1, const R &arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (__range_cmp(&arg1.value, &arg2.value) > 0);
	return result;
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_ge(kern_context *kcxt,
				   const R &arg1, const R &arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (__range_cmp(&arg1.value, &arg2.value) >= 0);
	return result;
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_lt(kern_context *kcxt,
				   const R &arg1, const R &arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (__range_cmp(&arg1.value, &arg2.value) < 0);
	return result;
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_le(kern_context *kcxt,
				   const R &arg1, const R &arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = (__range_cmp(&arg1.value, &arg2.value) <= 0);
	return result;
}

template <typename R>
STATIC_FUNCTION(pg_int4_t)
__generic_range_cmp(kern_context *kcxt,
					const R &arg1, const R &arg2)
{
	pg_int4_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = __range_cmp(&arg1.value, &arg2.value);
	return result;
}

template <typename RangeType>
STATIC_FUNCTION(cl_bool)
__range_overlaps_internal(RangeType *r1, RangeType *r2)
{
	if (r1->empty || r2->empty)
		return false;
	if (__range_cmp_bounds(&r1->l, &r2->l) >= 0 &&
		__range_cmp_bounds(&r1->l, &r2->u) <= 0)
		return true;
	if (__range_cmp_bounds(&r2->l, &r1->l) >= 0 &&
		__range_cmp_bounds(&r2->l, &r1->u) <= 0)
		return true;
	return false;
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_overlaps(kern_context *kcxt,
						 const R &arg1, const R &arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = __range_overlaps_internal(&arg1.value,
												 &arg2.value);
	return result;
}

template <typename RangeType, typename ElementType>
STATIC_FUNCTION(cl_bool)
__range_contains_elem_internal(RangeType *r, ElementType *val)
{
	int		cmp;

	if (r->empty)
		return false;
	if (!r->l.infinite)
	{
		cmp = Compare(r->l.val, *val);
		if (cmp > 0)
			return false;
		if (cmp == 0 && !r->l.inclusive)
			return false;
	}

	if (!r->u.infinite)
	{
		cmp = Compare(r->u.val, *val);
		if (cmp < 0)
			return false;
		if (cmp == 0 && !r->u.inclusive)
			return false;
	}
	return true;
}

template <typename R, typename E>
STATIC_FUNCTION(pg_bool_t)
__generic_range_contains_elem(kern_context *kcxt,
							  const R &arg1, const E &arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = __range_contains_elem_internal(&arg1.value,
													  &arg2.value);
	return result;
}

template <typename RangeType>
STATIC_FUNCTION(cl_bool)
__range_contains_internal(RangeType *r1, RangeType *r2)
{
	if (r2->empty)
		return true;
	else if (r1->empty)
		return false;
	/* else we must have lower1 <= lower2 and upper1 >= upper2 */
	if (__range_cmp_bounds(&r1->l, &r2->l) > 0)
		return false;
	if (__range_cmp_bounds(&r1->u, &r2->u) < 0)
		return false;

	return true;
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_contains(kern_context *kcxt,
						 const R &arg1, const R &arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = __range_contains_internal(&arg1.value,
												 &arg2.value);
	return result;
}

template <typename R, typename E>
STATIC_FUNCTION(pg_bool_t)
__generic_elem_contained_by_range(kern_context *kcxt,
								  const E &arg1, const R &arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = __range_contains_elem_internal(&arg2.value,
													  &arg1.value);
	return result;
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_contained_by(kern_context *kcxt,
							 const R &arg1, const R &arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = __range_contains_internal(&arg2.value,
												 &arg1.value);
	return result;
}

template <typename RangeBound>
STATIC_FUNCTION(cl_bool)
__bounds_adjacent(RangeBound b1, RangeBound b2)
{
	int		cmp;

	assert(!b1.lower && b2.lower);
	cmp = __range_cmp_bound_values(&b1, &b2);
	if (cmp < 0)
	{
		/* The bounds are of a discrete range type */
		b1.inclusive = !b1.inclusive;
		b2.inclusive = !b2.inclusive;
		b1.lower = true;
		b2.lower = false;
		cmp = __range_cmp_bound_values(&b1, &b2);
		if (cmp == 0 && !(b1.inclusive && b2.inclusive))
			return true;
	}
	else if (cmp == 0)
		return b1.inclusive != b2.inclusive;
	return false;	/* bounds overlap */
}

template <typename RangeType>
STATIC_INLINE(cl_bool)
__range_adjacent_internal(RangeType *r1, RangeType *r2)
{
	if (r1->empty || r2->empty)
		return false;
	return (__bounds_adjacent(r1->u, r2->l) ||
			__bounds_adjacent(r2->u, r1->l));
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_adjacent(kern_context *kcxt,
						 const R &arg1, const R &arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
		result.value = __range_adjacent_internal(&arg1.value,
												 &arg2.value);
	return result;
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_before(kern_context *kcxt,
					   const R &arg1, const R &arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg1.value.empty || arg2.value.empty)
			result.value = false;
		else
			result.value = (__range_cmp_bounds(&arg1.value.u,
											   &arg2.value.l) < 0);
	}
	return result;
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_after(kern_context *kcxt,
					  const R &arg1, const R &arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg1.value.empty || arg2.value.empty)
			result.value = false;
		else
			result.value = (__range_cmp_bounds(&arg1.value.l,
											   &arg2.value.u) > 0);
	}
	return result;
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_overright(kern_context *kcxt,
						  const R &arg1, const R &arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg1.value.empty || arg2.value.empty)
			result.value = false;
		else
			result.value = (__range_cmp_bounds(&arg1.value.l,
											   &arg2.value.l) >= 0);
	}
	return result;
}

template <typename R>
STATIC_FUNCTION(pg_bool_t)
__generic_range_overleft(kern_context *kcxt,
						 const R &arg1, const R &arg2)
{
	pg_bool_t	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg1.value.empty || arg2.value.empty)
			result.value = false;
		else
			result.value = (__range_cmp_bounds(&arg1.value.u,
											   &arg2.value.u) <= 0);
	}
	return result;
}

template <typename R>
DEVICE_FUNCTION(R)
__generic_range_union(kern_context *kcxt,
					  const R &arg1, const R &arg2, bool strict)
{
	R	result;

	memset(&result, 0, sizeof(result));
	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg1.value.empty)
			return arg2;
		if (arg2.value.empty)
			return arg1;
		if (strict &&
			!__range_overlaps_internal(&arg1.value, &arg2.value) &&
			!__range_adjacent_internal(&arg1.value, &arg2.value))
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DATA_EXCEPTION,
						  "result of range union would not be contiguous");
		}
		else
		{
			if (__range_cmp_bounds(&arg1.value.l, &arg2.value.l) < 0)
				result.value.l = arg1.value.l;
			else
				result.value.l = arg2.value.l;

			if (__range_cmp_bounds(&arg1.value.u, &arg2.value.u) > 0)
				result.value.u = arg1.value.u;
			else
				result.value.u = arg2.value.u;

			result.value.empty = false;
		}
	}
	return result;
}

template <typename R>
STATIC_FUNCTION(R)
__generic_range_intersect(kern_context *kcxt, const R &arg1, const R &arg2)
{
	R	result;

	memset(&result, 0, sizeof(result));
	result.isnull = arg1.isnull | arg2.isnull;
	if (!result.isnull)
	{
		if (arg1.value.empty ||
			arg2.value.empty ||
			!__range_overlaps_internal(&arg1.value, &arg2.value))
		{
			result.value.empty = true;
		}
		else
		{
			if (__range_cmp_bounds(&arg1.value.l, &arg2.value.l) >= 0)
				result.value.l = arg1.value.l;
			else
				result.value.l = arg2.value.l;

			if (__range_cmp_bounds(&arg1.value.u, &arg2.value.u) <= 0)
				result.value.u = arg1.value.u;
			else
				result.value.u = arg2.value.u;

			result.value.empty = false;
		}
	}
	return result;
}

template <typename R>
STATIC_FUNCTION(R)
__generic_range_minus(kern_context *kcxt, const R &arg1, const R &arg2)
{
	R	result;

	result.isnull = arg1.isnull | arg2.isnull;
	if (result.isnull)
	{
		memset(&result.value, 0, sizeof(result.value));
	}
	else if (arg1.value.empty || arg2.value.empty)
	{
		/* if either is empty, arg1 is the correct answer */
		result.value = arg1.value;
	}
	else
	{
		int		cmp_l1l2 = __range_cmp_bounds(&arg1.value.l, &arg2.value.l);
		int		cmp_l1u2 = __range_cmp_bounds(&arg1.value.l, &arg2.value.u);
		int		cmp_u1l2 = __range_cmp_bounds(&arg1.value.u, &arg2.value.l);
		int		cmp_u1u2 = __range_cmp_bounds(&arg1.value.u, &arg2.value.u);

		memset(&result.value, 0, sizeof(result.value));
		if (cmp_l1l2 < 0 && cmp_u1u2 > 0)
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_DATA_EXCEPTION,
						"result of range difference would not be contiguous");
		}
		else if (cmp_l1u2 > 0 || cmp_u1l2 < 0)
		{
			result.value = arg1.value;
		}
		else if (cmp_l1l2 >= 0 && cmp_u1u2 <= 0)
		{
			result.value.empty = true;
		}
		else if (cmp_l1l2 <= 0 && cmp_u1l2 >= 0 && cmp_u1u2 <= 0)
		{
			result.value.l = arg1.value.l;
			result.value.u = arg2.value.l;
			result.value.u.inclusive = !result.value.u.inclusive;
			result.value.u.lower = false;
		}
		else if (cmp_l1l2 >= 0 && cmp_u1u2 >= 0 && cmp_l1u2 <= 0)
		{
			result.value.l = arg2.value.u;
			result.value.l.inclusive = !result.value.l.inclusive;
			result.value.l.lower = true;
			result.value.u = arg1.value.u;
		}
		else
		{
			result.isnull = true;
			STROM_EREPORT(kcxt, ERRCODE_INTERNAL_ERROR,
						  "unexpected case in range_minus");
		}
	}
	return result;
}

#define PG_RANGETYPE_FUNCTION_TEMPLATE(ELEMENT,RANGE)			\
	__GENERIC_RANGE_LOWER(ELEMENT,RANGE)						\
	__GENERIC_RANGE_UPPER(ELEMENT,RANGE)						\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_isempty(kern_context *kcxt,					\
						   const pg_##RANGE##_t &arg1)			\
	{															\
		return __generic_range_isempty(kcxt,arg1);				\
	}															\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_lower_inc(kern_context *kcxt,				\
							 const pg_##RANGE##_t &arg1)		\
	{															\
		return __generic_range_lower_inc(kcxt,arg1);			\
	}															\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_upper_inc(kern_context *kcxt,				\
							 const pg_##RANGE##_t &arg1)		\
	{															\
		return __generic_range_upper_inc(kcxt,arg1);			\
	}															\
	DEVICE_FUNCTION(pg_bool_t)									\
    pgfn_##RANGE##_lower_inf(kern_context *kcxt,				\
							 const pg_##RANGE##_t &arg1)		\
	{															\
		return __generic_range_lower_inf(kcxt,arg1);			\
	}															\
    DEVICE_FUNCTION(pg_bool_t)									\
    pgfn_##RANGE##_upper_inf(kern_context *kcxt,				\
                             const pg_##RANGE##_t &arg1)		\
	{															\
		return __generic_range_upper_inf(kcxt,arg1);			\
	}															\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_eq(kern_context *kcxt,						\
					  const pg_##RANGE##_t &arg1,				\
					  const pg_##RANGE##_t &arg2)				\
	{															\
		return __generic_range_eq(kcxt,arg1,arg2);				\
	}															\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_ne(kern_context *kcxt,						\
					  const pg_##RANGE##_t &arg1,				\
					  const pg_##RANGE##_t &arg2)				\
	{															\
		return __generic_range_ne(kcxt,arg1,arg2);				\
	}															\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_gt(kern_context *kcxt,						\
					  const pg_##RANGE##_t &arg1,				\
					  const pg_##RANGE##_t &arg2)				\
	{															\
		return __generic_range_gt(kcxt,arg1,arg2);				\
	}															\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_ge(kern_context *kcxt,						\
					  const pg_##RANGE##_t &arg1,				\
					  const pg_##RANGE##_t &arg2)				\
	{															\
		return __generic_range_ge(kcxt,arg1,arg2);				\
	}															\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_lt(kern_context *kcxt,						\
					  const pg_##RANGE##_t &arg1,				\
					  const pg_##RANGE##_t &arg2)				\
	{															\
		return __generic_range_lt(kcxt,arg1,arg2);				\
	}															\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_le(kern_context *kcxt,						\
					  const pg_##RANGE##_t &arg1,				\
					  const pg_##RANGE##_t &arg2)				\
	{															\
		return __generic_range_le(kcxt,arg1,arg2);				\
	}															\
	DEVICE_FUNCTION(pg_int4_t)									\
	pgfn_type_compare(kern_context *kcxt,						\
					  const pg_##RANGE##_t &arg1,				\
					  const pg_##RANGE##_t &arg2)				\
	{															\
		return __generic_range_cmp(kcxt,arg1,arg2);				\
	}															\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_overlaps(kern_context *kcxt,					\
							const pg_##RANGE##_t &arg1,			\
							const pg_##RANGE##_t &arg2)			\
	{															\
		return __generic_range_overlaps(kcxt, arg1, arg2);		\
	}															\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_contains_elem(kern_context *kcxt,			\
								 const pg_##RANGE##_t &arg1,	\
								 const pg_##ELEMENT##_t &arg2)	\
	{															\
		return __generic_range_contains_elem(kcxt,arg1,arg2);	\
	}															\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_contains(kern_context *kcxt,					\
							const pg_##RANGE##_t &arg1,			\
							const pg_##RANGE##_t &arg2)			\
	{															\
		return __generic_range_contains(kcxt,arg1,arg2);		\
	}															\
    DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_elem_contained_by_##RANGE(kern_context *kcxt,			\
								   const pg_##ELEMENT##_t &arg1,\
								   const pg_##RANGE##_t &arg2)	\
	{															\
		return __generic_elem_contained_by_range(kcxt,arg1,arg2); \
	}															\
    DEVICE_FUNCTION(pg_bool_t)									\
    pgfn_##RANGE##_contained_by(kern_context *kcxt,				\
                                const pg_##RANGE##_t &arg1,		\
                                const pg_##RANGE##_t &arg2)		\
	{															\
		return __generic_range_contained_by(kcxt,arg1,arg2);	\
	}															\
    DEVICE_FUNCTION(pg_bool_t)									\
    pgfn_##RANGE##_adjacent(kern_context *kcxt,					\
                            const pg_##RANGE##_t &arg1,			\
                            const pg_##RANGE##_t &arg2)			\
	{															\
		return __generic_range_adjacent(kcxt,arg1,arg2);		\
	}															\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_before(kern_context *kcxt,					\
						  const pg_##RANGE##_t &arg1,			\
						  const pg_##RANGE##_t &arg2)			\
	{															\
		return __generic_range_before(kcxt,arg1,arg2);			\
	}															\
    DEVICE_FUNCTION(pg_bool_t)									\
    pgfn_##RANGE##_after(kern_context *kcxt,					\
                         const pg_##RANGE##_t &arg1,			\
                         const pg_##RANGE##_t &arg2)			\
	{															\
		return __generic_range_after(kcxt,arg1,arg2);			\
	}															\
    DEVICE_FUNCTION(pg_bool_t)									\
    pgfn_##RANGE##_overright(kern_context *kcxt,				\
							 const pg_##RANGE##_t &arg1,		\
							 const pg_##RANGE##_t &arg2)		\
	{															\
		return __generic_range_overright(kcxt,arg1,arg2);		\
	}															\
    DEVICE_FUNCTION(pg_bool_t)									\
    pgfn_##RANGE##_overleft(kern_context *kcxt,					\
                            const pg_##RANGE##_t &arg1,			\
                            const pg_##RANGE##_t &arg2)			\
	{															\
		return __generic_range_overleft(kcxt,arg1,arg2);		\
	}															\
    DEVICE_FUNCTION(pg_##RANGE##_t)								\
	pgfn_##RANGE##_union(kern_context *kcxt,					\
						 const pg_##RANGE##_t &arg1,			\
						 const pg_##RANGE##_t &arg2)			\
	{															\
		return __generic_range_union(kcxt,arg1,arg2,true);		\
	}															\
	DEVICE_FUNCTION(pg_##RANGE##_t)								\
	pgfn_##RANGE##_merge(kern_context *kcxt,					\
						 const pg_##RANGE##_t &arg1,			\
						 const pg_##RANGE##_t &arg2)			\
	{															\
		return __generic_range_union(kcxt,arg1,arg2,false);		\
	}															\
	DEVICE_FUNCTION(pg_##RANGE##_t)								\
    pgfn_##RANGE##_intersect(kern_context *kcxt,				\
                             const pg_##RANGE##_t &arg1,		\
                             const pg_##RANGE##_t &arg2)		\
	{															\
		return __generic_range_intersect(kcxt,arg1,arg2);		\
	}															\
	DEVICE_FUNCTION(pg_##RANGE##_t)								\
	pgfn_##RANGE##_minus(kern_context *kcxt,					\
						 const pg_##RANGE##_t &arg1,			\
						 const pg_##RANGE##_t &arg2)			\
	{															\
		return __generic_range_minus(kcxt,arg1,arg2);			\
	}

PG_RANGETYPE_FUNCTION_TEMPLATE(int4,int4range)
PG_RANGETYPE_FUNCTION_TEMPLATE(int8,int8range)
PG_RANGETYPE_FUNCTION_TEMPLATE(timestamp,tsrange)
PG_RANGETYPE_FUNCTION_TEMPLATE(timestamptz,tstzrange)
PG_RANGETYPE_FUNCTION_TEMPLATE(date,daterange)
