/*
 * cuda_rangetypes.h
 *
 * Collection of range-types support routeins for CUDA GPU devices
 * --
 * Copyright 2011-2018 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2018 (C) The PG-Strom Development Team
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
#ifdef CUDA_RANGETYPES_H
#define CUDA_RANGETYPES_H

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
 * !!NOTE!!
 * This template assumes BASE type is aligned to sizeof(BASE)
 */
template<typename T>
STATIC_INLINE(cl_bool)
__rangetype_datum_ref(void *datum,
					  T &lbound, cl_bool &l_infinite, cl_bool &l_inclusive,
					  T &ubound, cl_bool &u_infinite, cl_bool &u_inclusive,
					  cl_bool &is_empty)
{
	char	vl_buf[MAXALIGN(VARHDRSZ + sizeof(cl_uint) +
							s * sizeof(T) + 1)];
	char	flags;
	char   *pos;

	if (VARATT_IS_EXTERNAL(datum))
		return false;
	else if (VARATT_IS_COMPRESSED(datum))
	{
		if (!toast_decompress_datum(vl_buf, sizeof(vl_buf),
									(struct varlena *)datum))
			return false;
		datum = vl_buf;
	}
	flags = *((char *)datum + VARSIZE_ANY(datum) - 1);
	pos = VARDATA_ANY(datum) + sizeof(cl_uint);
	if (!RANGE_HAS_LBOUND(flags))
		lbound = 0;
	else
	{
		memcpy(&lbound, pos, sizeof(T));
		pos += sizeof(BASE);
	}
	if (!RANGE_HAS_UBOUND(flags))
		ubound = 0;
	else
	{
		memcpy(&ubound, pos, sizeof(T));
		pos += sizeof(BASE);
	}
	is_empty = ((flags & RANGE_EMPTY) != 0);
	l_infinite = ((flags & RANGE_LB_INF) != 0);
	l_inclusive = ((flags & RANGE_LB_INC) != 0);
	u_infinite = ((flags & RANGE_UB_INF) != 0);
	u_inclusive = ((flags & RANGE_UB_INC) != 0);

	return true;
}

template<typename T>
STATIC_INLINE(void)
__rangetype_datum_store(char *extra_buf,
						cl_uint type_oid,
						T lbound, cl_bool l_infinite, cl_bool l_inclusive,
						T ubound, cl_bool u_infinite, cl_bool u_inclusive,
						cl_bool is_empty)
{
	char   *pos = extra_buf + VARHDRSZ;
	char	flags = ((l_infinite ? RANGE_LB_INF : 0)  |
					 (l_inclusive ? RANGE_LB_INC : 0) |
					 (u_infinite ? RANGE_UB_INF : 0)  |
					 (u_inclusive ? RANGE_UB_INC : 0) |
					 (is_empty ? RANGE_EMPTY : 0));
	*((cl_uint *)pos) = type_oid;
	pos += sizeof(cl_uint);
	*((T *)pos) = lbound;
	pos += sizeof(T);
	*((T *)pos) = ubound;
	pos += sizeof(T);
	*((char *)pos) = flags;
	SET_VARSIZE(extra_buf, pos - extra_buf);
}

template<typename T>
STATIC_INLINE(cl_uint)
__rangetype_comp_crc32(const cl_uint *crc32_table,
					   cl_uint hash,
					   cl_uint type_oid,
					   T lbound, cl_bool l_infinite, cl_bool l_inclusive,
					   T ubound, cl_bool u_infinite, cl_bool u_inclusive,
					   cl_bool is_empty)
{
	char	flags = ((l_infinite ? RANGE_LB_INF : 0)  |
					 (l_inclusive ? RANGE_LB_INC : 0) |
					 (u_infinite ? RANGE_UB_INF : 0)  |
					 (u_inclusive ? RANGE_UB_INC : 0) |
					 (is_empty ? RANGE_EMPTY : 0));

	pg_common_comp_crc32(crc32_table, hash,
						 (char *)&type_oid,
						 sizeof(cl_uint));
	pg_common_comp_crc32(crc32_table, hash,
						 (char *)&lbound,
						 sizeof(lbound));
	pg_common_comp_crc32(crc32_table, hash,
						 (char *)&ubound,
						 sizeof(ubound));
	pg_common_comp_crc32(crc32_table, hash,
						 (char *)&flags,
						 sizeof(char));
	return hash;
}

#define PG_RANGETYPE_TEMPLATE(NAME,BASE,TYPEOID)					\
	typedef struct													\
	{																\
		struct {													\
			BASE	val;											\
			cl_bool	infinite;										\
			cl_bool	inclusive;										\
		} l;	/* lower bound */									\
		struct {													\
			BASE	val;											\
			cl_bool	infinite;										\
			cl_bool	inclusive;										\
		} u;	/* upper bound */									\
		bool		empty;											\
	} __##NAME;														\
																	\
	STROMCL_SIMPLE_DATATYPE_TEMPLATE(NAME,__##NAME)					\
																	\
	STATIC_FUNCTION(pg_##NAME##_t)									\
	pg_##NAME##_datum_ref(kern_context *kcxt, void *datum)			\
	{																\
		pg_##NAME##_t	result;										\
																	\
		if (__rangetype_datum_ref(datum,							\
								  result.l.val,						\
								  result.l.infinite,				\
								  result.l.inclusive,				\
								  result.u.val,						\
								  result.u.infinite,				\
								  result.u.inclusive,				\
								  result.empty))					\
			result.isnull = false;									\
		else														\
		{															\
			result.isnull = true;									\
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);		\
		}															\
	}																\
																	\
	STATIC_FUNCTION(cl_uint)										\
	pg_##NAME##_datum_store(kern_context *kcxt,						\
							void *extra_buf,						\
							pg_##NAME##_t datum)					\
	{																\
		if (datum.isnull)											\
			return 0;												\
		if (extra_buf)												\
			__rangetype_datum_store(extra_buf,						\
									TYPEOID,						\
									datum.l.val,					\
									datum.l.infinite,				\
									datum.l.inclusive,				\
									datum.u.val,					\
									datum.u.infinite,				\
									datum.u.inclusive,				\
									datum.empty);					\
		return VARHDRSZ + sizeof(cl_uint) + 2 * sizeof(BASE) + 1;	\
	}																\
																	\
	STATIC_FUNCTION(pg_##NAME##_t)									\
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
																	\
	}																\
																	\
	STROMCL_SIMPLE_NULLTEST_TEMPLATE(NAME)							\
																	\
	STATIC_FUNCTION(cl_uint)										\
	pg_##NAME##_comp_crc32(const cl_uint *crc32_table,				\
						   cl_uint hash, pg_##NAME##_t datum)		\
	{																\
		if (!datum.isnull)											\
			hash = __rangetype_comp_crc32(crc32_table,				\
										  hash,						\
										  TYPEOID,					\
										  datum.l.val,				\
										  datum.l.infinite,			\
										  datum.l.inclusive,		\
										  datum.u.val,				\
										  datum.u.infinite,			\
										  datum.u.inclusive,		\
										  datum.empty);				\
		return hash;												\
	}																\
																	\
	STATIC_INLINE(Datum)											\
	pg_##NAME##_as_datum(void *addr)								\
	{																\
		return PointerGetDatum(addr);								\
	}

#ifndef PG_INT4RANGE_TYPE_DEFINED
#define PG_INT4RANGE_TYPE_DEFINED
PG_RANGETYPE_TEMPLATE(int4range,cl_int,PG_INT4OID)
#endif	/* PG_INT4RANGE_TYPE_DEFINED */

#ifndef PG_INT8RANGE_TYPE_DEFINED
#define PG_INT8RANGE_TYPE_DEFINED
PG_RANGETYPE_TEMPLATE(int8range,cl_long,PG_INT8OID)
#endif	/* PG_INT4RANGE_TYPE_DEFINED */

#ifdef CUDA_TIMELIB_H
#ifndef PG_TSRANGE_TYPE_DEFINED
#define PG_TSRANGE_TYPE_DEFINED
PG_RANGETYPE_TEMPLATE(tsrange,Timestamp,PG_TIMESTAMPOID)
#endif	/* PG_TSRANGE_TYPE_DEFINED */

#ifndef PG_TSTZRANGE_TYPE_DEFINED
#define PG_TSTZRANGE_TYPE_DEFINED
PG_RANGETYPE_TEMPLATE(tstzrange,TimestampTz,PG_TIMESTAMPTZOID)
#endif	/* PG_TSTZRANGE_TYPE_DEFINED */

#ifndef PG_DATERANGE_TYPE_DEFINED
#define PG_DATERANGE_TYPE_DEFINED
PG_RANGETYPE_TEMPLATE(daterange,DateADT,PG_DATEOID)
#endif	/* PG_DATERANGE_TYPE_DEFINED */
#endif	/* CUDA_TIMELIB_H */

template<typename R,typename T>
STATIC_INLINE(R)
pgfn_range_lower(kern_context *kcxt, T arg1)
{
	R	result;

	result.isnull = arg1.isnull;
	if (!result.isnull)
	{
		if (arg1.value.empty || arg1.value.l.infinite)
			result.isnull = true;
		else
			result.value = arg1.value.l.val;
	}
	return result;
}

template<typename R,typename T>
STATIC_INLINE(R)
pgfn_range_upper(kern_context *kcxt, T arg1)
{
	R	result;

	result.isnull = arg1.isnull;
	if (!result.isnull)
	{
		if (arg1.value.empty || arg1.value.u.infinite)
			result.isnull = true;
		else
			result.value = arg1.value.u.val;
	}
	return result;
}






#endif	/* CUDA_RANGETYPES_H */
