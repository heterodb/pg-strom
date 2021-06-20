/*
 * cuda_rangetypes.h
 *
 * Collection of range-types support routeins for CUDA GPU devices
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef CUDA_RANGETYPES_H
#define CUDA_RANGETYPES_H
#include "cuda_timelib.h"

#define PG_RANGETYPE_TEMPLATE(NAME,BASE)							\
	typedef struct													\
	{																\
		BASE	val;												\
		cl_bool	infinite;		/* bound is +/- infinity */			\
		cl_bool	inclusive;		/* bound is inclusive */			\
		cl_bool	lower;			/* bound is lower */				\
	} __##NAME##_bound;												\
																	\
	typedef struct													\
	{																\
		__##NAME##_bound	l;	/* lower bound */					\
		__##NAME##_bound	u;	/* upper bound */					\
		bool		empty;											\
	} __##NAME;														\
																	\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,__##NAME)					\
	STROMCL_EXTERNAL_VARREF_TEMPLATE(NAME)							\
	STROMCL_EXTERNAL_COMP_HASH_TEMPLATE(NAME)						\
	STROMCL_UNSUPPORTED_ARROW_TEMPLATE(NAME)

#ifndef PG_INT4RANGE_TYPE_DEFINED
#define PG_INT4RANGE_TYPE_DEFINED
#define PG_INT4RANGEOID				3904
PG_RANGETYPE_TEMPLATE(int4range,cl_int)
#endif	/* PG_INT4RANGE_TYPE_DEFINED */

#ifndef PG_INT8RANGE_TYPE_DEFINED
#define PG_INT8RANGE_TYPE_DEFINED
#define PG_INT8RANGEOID				3926
PG_RANGETYPE_TEMPLATE(int8range,cl_long)
#endif	/* PG_INT8RANGE_TYPE_DEFINED */

#ifdef CUDA_TIMELIB_H
#ifndef PG_TSRANGE_TYPE_DEFINED
#define PG_TSRANGE_TYPE_DEFINED
#define PG_TSRANGEOID				3908
PG_RANGETYPE_TEMPLATE(tsrange,Timestamp)
#endif	/* PG_TSRANGE_TYPE_DEFINED */

#ifndef PG_TSTZRANGE_TYPE_DEFINED
#define PG_TSTZRANGE_TYPE_DEFINED
#define PG_TSTZRANGEOID				3910
PG_RANGETYPE_TEMPLATE(tstzrange,TimestampTz)
#endif	/* PG_TSTZRANGE_TYPE_DEFINED */

#ifndef PG_DATERANGE_TYPE_DEFINED
#define PG_DATERANGE_TYPE_DEFINED
#define PG_DATERANGEOID				3912
PG_RANGETYPE_TEMPLATE(daterange,DateADT)
#endif	/* PG_DATERANGE_TYPE_DEFINED */
#endif	/* CUDA_TIMELIB_H */

#undef PG_RANGETYPE_TEMPLATE

#define PG_RANGETYPE_DECLARATION_TEMPLATE(ELEMENT,RANGE)		\
	DEVICE_FUNCTION(pg_##ELEMENT##_t)							\
	pgfn_##RANGE##_lower(kern_context *kcxt,					\
						 const pg_##RANGE##_t &arg1);			\
	DEVICE_FUNCTION(pg_##ELEMENT##_t)							\
	pgfn_##RANGE##_upper(kern_context *kcxt,					\
						 const pg_##RANGE##_t &arg1);			\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_isempty(kern_context *kcxt,					\
						   const pg_##RANGE##_t &arg1);			\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_lower_inc(kern_context *kcxt,				\
							 const pg_##RANGE##_t &arg1);		\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_upper_inc(kern_context *kcxt,				\
							 const pg_##RANGE##_t &arg1);		\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_lower_inf(kern_context *kcxt,				\
							 const pg_##RANGE##_t &arg1);		\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_upper_inf(kern_context *kcxt,				\
							 const pg_##RANGE##_t &arg1);		\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_eq(kern_context *kcxt,						\
					  const pg_##RANGE##_t &arg1,				\
					  const pg_##RANGE##_t &arg2);				\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_ne(kern_context *kcxt,						\
					  const pg_##RANGE##_t &arg1,				\
					  const pg_##RANGE##_t &arg2);				\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_lt(kern_context *kcxt,						\
					  const pg_##RANGE##_t &arg1,				\
					  const pg_##RANGE##_t &arg2);				\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_le(kern_context *kcxt,						\
					  const pg_##RANGE##_t &arg1,				\
					  const pg_##RANGE##_t &arg2);				\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_gt(kern_context *kcxt,						\
					  const pg_##RANGE##_t &arg1,				\
					  const pg_##RANGE##_t &arg2);				\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_ge(kern_context *kcxt,						\
					  const pg_##RANGE##_t &arg1,				\
					  const pg_##RANGE##_t &arg2);				\
	DEVICE_INLINE(pg_int4_t)									\
	pgfn_type_compare(kern_context *kcxt,						\
					  const pg_##RANGE##_t &arg1,				\
					  const pg_##RANGE##_t &arg2);				\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_overlaps(kern_context *kcxt,					\
							const pg_##RANGE##_t &arg1,			\
							const pg_##RANGE##_t &arg2);		\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_contains_elem(kern_context *kcxt,			\
								 const pg_##RANGE##_t &arg1,	\
								 const pg_##ELEMENT##_t &arg2);	\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_contains(kern_context *kcxt,					\
							const pg_##RANGE##_t &arg1,			\
							const pg_##RANGE##_t &arg2);		\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_elem_contained_by_##RANGE(kern_context *kcxt,			\
								   const pg_##ELEMENT##_t &arg1,\
								   const pg_##RANGE##_t &arg2);	\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_contained_by(kern_context *kcxt,				\
								const pg_##RANGE##_t &arg1,		\
								const pg_##RANGE##_t &arg2);	\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_adjacent(kern_context *kcxt,					\
							const pg_##RANGE##_t &arg1,			\
							const pg_##RANGE##_t &arg2);		\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_before(kern_context *kcxt,					\
						  const pg_##RANGE##_t &arg1,			\
						  const pg_##RANGE##_t &arg2);			\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_after(kern_context *kcxt,					\
						 const pg_##RANGE##_t &arg1,			\
						 const pg_##RANGE##_t &arg2);			\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_overright(kern_context *kcxt,				\
							 const pg_##RANGE##_t &arg1,		\
							 const pg_##RANGE##_t &arg2);		\
	DEVICE_FUNCTION(pg_bool_t)									\
	pgfn_##RANGE##_overleft(kern_context *kcxt,					\
							const pg_##RANGE##_t &arg1,			\
							const pg_##RANGE##_t &arg2);		\
	DEVICE_FUNCTION(pg_##RANGE##_t)								\
	pgfn_##RANGE##_union(kern_context *kcxt,					\
						 const pg_##RANGE##_t &arg1,			\
						 const pg_##RANGE##_t &arg2);			\
	DEVICE_FUNCTION(pg_##RANGE##_t)								\
	pgfn_##RANGE##_merge(kern_context *kcxt,					\
						 const pg_##RANGE##_t &arg1,			\
						 const pg_##RANGE##_t &arg2);			\
	DEVICE_FUNCTION(pg_##RANGE##_t)								\
	pgfn_##RANGE##_intersect(kern_context *kcxt,				\
							 const pg_##RANGE##_t &arg1,		\
							 const pg_##RANGE##_t &arg2);		\
	DEVICE_FUNCTION(pg_##RANGE##_t)								\
	pgfn_##RANGE##_minus(kern_context *kcxt,					\
						 const pg_##RANGE##_t &arg1,			\
						 const pg_##RANGE##_t &arg2);

PG_RANGETYPE_DECLARATION_TEMPLATE(int4,int4range)
PG_RANGETYPE_DECLARATION_TEMPLATE(int8,int8range)
PG_RANGETYPE_DECLARATION_TEMPLATE(timestamp,tsrange)
PG_RANGETYPE_DECLARATION_TEMPLATE(timestamptz,tstzrange)
PG_RANGETYPE_DECLARATION_TEMPLATE(date,daterange)
#undef PG_RANGETYPE_DECLARATION_TEMPLATE

#endif	/* CUDA_RANGETYPES_H */
