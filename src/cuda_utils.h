/*
 * cuda_utils.h
 *
 * Collection of CUDA inline functions for device code
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
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#ifdef __CUDACC__
/*
 * pgstromTotalSum
 *
 * A utility routine to calculate total sum of the supplied array which are
 * consists of primitive types.
 * Unlike pgstromStairLikeSum, it accepts larger length of the array than
 * size of thread block, and unused threads shall be relaxed earlier.
 *
 * Restrictions:
 * - array must be a primitive types, like int, double.
 * - array must be on the shared memory.
 * - all the threads must call the function simultaneously.
 *   (Unacceptable to call the function in if-block)
 */
template <typename T>
DEVICE_ONLY_FUNCTION(T)
pgstromTotalSum(T *values, cl_uint nitems)
{
	cl_uint		nsteps = get_next_log2(nitems);
	cl_uint		nthreads;
	cl_uint		step;
	cl_uint		loop;
	T			retval;

	if (nitems == 0)
		return (T)(0);
	__syncthreads();
	for (step=1; step <= nsteps; step++)
	{
		nthreads = ((nitems - 1) >> step) + 1;

		for (loop=get_local_id(); loop < nthreads; loop += get_local_size())
		{
			cl_uint		dst = (loop << step);
			cl_uint		src = dst + (1U << (step - 1));

			if (src < nitems)
				values[dst] += values[src];
		}
		__syncthreads();
	}
	retval = values[0];
	__syncthreads();

	return retval;
}
#endif /* __CUDACC__ */

#ifdef __CUDACC__
/*
 * Utility functions to reference system columns
 *   (except for ctid and table_oid)
 */
STATIC_INLINE(cl_int)
pg_sysattr_ctid_store(kern_context *kcxt,
					  kern_data_store *kds,
					  HeapTupleHeaderData *htup,
					  ItemPointerData *t_self,
					  cl_char &dclass,
					  Datum   &value)
{
	void	   *temp;

	if (!t_self)
		dclass = DATUM_CLASS__NULL;
	else
	{
		dclass = DATUM_CLASS__NORMAL;
		if (kds && ((char *)t_self >= (char *)kds &&
					(char *)t_self <  (char *)kds + kds->length))
		{
			value = PointerGetDatum(t_self);
		}
		else
		{
			temp = kern_context_alloc(kcxt, sizeof(ItemPointerData));
			if (temp)
			{
				memcpy(temp, t_self, sizeof(ItemPointerData));
				value = PointerGetDatum(temp);
				return sizeof(ItemPointerData);
			}
			dclass = DATUM_CLASS__NULL;
			STROM_SET_ERROR(&kcxt->e, StromError_CpuReCheck);
		}
	}
	return 0;
}

STATIC_INLINE(cl_int)
pg_sysattr_oid_store(kern_context *kcxt,
					 kern_data_store *kds,
					 HeapTupleHeaderData *htup,
					 ItemPointerData *t_self,
					 cl_char &dclass,
					 Datum   &value)
{
	if (!htup)
		dclass = DATUM_CLASS__NULL;
	else
	{
		dclass = DATUM_CLASS__NORMAL;
		if ((htup->t_infomask & HEAP_HASOID) == 0)
			value = 0;
		else
			value = *((cl_uint *)((char *) htup
								  + htup->t_hoff
								  - sizeof(cl_uint)));
	}
	return 0;
}

STATIC_INLINE(cl_int)
pg_sysattr_xmin_store(kern_context *kcxt,
					  kern_data_store *kds,
					  HeapTupleHeaderData *htup,
					  ItemPointerData *t_self,
					  cl_char &dclass,
					  Datum   &value)
{
	if (!htup)
		dclass = DATUM_CLASS__NULL;
	else
	{
		dclass = DATUM_CLASS__NORMAL;
		value  = htup->t_choice.t_heap.t_xmin;
	}
	return 0;
}

STATIC_INLINE(cl_int)
pg_sysattr_cmin_store(kern_context *kcxt,
					  kern_data_store *kds,
					  HeapTupleHeaderData *htup,
					  ItemPointerData *t_self,
					  cl_char &dclass,
					  Datum   &value)
{
	if (!htup)
		dclass = DATUM_CLASS__NULL;
	else
	{
		dclass = DATUM_CLASS__NORMAL;
		value  = htup->t_choice.t_heap.t_field3.t_cid;
	}
	return 0;
}

STATIC_INLINE(cl_int)
pg_sysattr_xmax_store(kern_context *kcxt,
					  kern_data_store *kds,
					  HeapTupleHeaderData *htup,
					  ItemPointerData *t_self,
					  cl_char &dclass,
					  Datum   &value)
{
	if (!htup)
		dclass = DATUM_CLASS__NULL;
	else
	{
		dclass = DATUM_CLASS__NORMAL;
		value  = htup->t_choice.t_heap.t_xmax;
	}
	return 0;
}

STATIC_INLINE(cl_int)
pg_sysattr_cmax_store(kern_context *kcxt,
					  kern_data_store *kds,
					  HeapTupleHeaderData *htup,
					  ItemPointerData *t_self,
					  cl_char &dclass,
					  Datum   &value)
{
	if (!htup)
		dclass = DATUM_CLASS__NULL;
	else
	{
		dclass = DATUM_CLASS__NORMAL;
		value  = htup->t_choice.t_heap.t_field3.t_cid;
	}
	return 0;
}

STATIC_INLINE(cl_int)
pg_sysattr_tableoid_store(kern_context *kcxt,
						  kern_data_store *kds,
						  HeapTupleHeaderData *htup,
						  ItemPointerData *t_self,
						  cl_char &dclass,
						  Datum   &value)
{
	if (!kds)
		dclass = DATUM_CLASS__NULL;
	else
	{
		dclass = DATUM_CLASS__NORMAL;
		value  = kds->table_oid;
	}
	return 0;
}
#endif	/* __CUDACC__ */

#ifdef __CUDACC__
/*
 * inline functions to form/deform HeapTuple
 */
STATIC_INLINE(cl_uint)
compute_heaptuple_size(kern_context *kcxt,
					   kern_data_store *kds,
					   cl_char *tup_dclass,
					   Datum   *tup_values)
{
	return __compute_heaptuple_size(kcxt,
									kds->colmeta,
									kds->tdhasoid,
									kds->ncols,
									tup_dclass,
									tup_values);
}

/*
 * form_kern_heaptuple
 *
 * A utility routine to build a kern_tupitem on the destination buffer
 * already allocated.
 *
 * tupitem      ... kern_tupitem allocated on the kds
 * kds_dst      ... destination data store
 * tup_self     ... item pointer of the tuple, if any
 * htup         ... tuple-header of the original tuple, if any
 * tup_dclass   ... any of DATUM_CLASS__*
 * tup_values   ... array of values to be written
 */
STATIC_INLINE(cl_uint)
form_kern_heaptuple(kern_context    *kcxt,
					kern_tupitem    *tupitem,		/* out */
					kern_data_store	*kds_dst,		/* in */
					ItemPointerData *tup_self,		/* in, optional */
					HeapTupleHeaderData *htup,		/* in, optional */
					cl_char         *tup_dclass,	/* in */
					Datum           *tup_values)	/* in */
{
	cl_uint		htuple_oid = 0;

	assert((uintptr_t)tupitem == MAXALIGN(tupitem));
	assert((char *)tupitem >= (char *)kds_dst &&
		   (char *)tupitem <  (char *)kds_dst + kds_dst->length);
	/* setup kern_tupitem */
	if (tup_self)
		tupitem->t_self = *tup_self;
	else
	{
		tupitem->t_self.ip_blkid.bi_hi = 0xffff;	/* InvalidBlockNumber */
		tupitem->t_self.ip_blkid.bi_lo = 0xffff;
		tupitem->t_self.ip_posid = 0;				/* InvalidOffsetNumber */
	}
	/* OID of tuple; deprecated at PG12 */
	if (kds_dst->tdhasoid &&
		htup && (htup->t_infomask & HEAP_HASOID) != 0)
	{
		htuple_oid = *((cl_uint *)((char *)htup
								   + htup->t_hoff
								   - sizeof(cl_uint)));
	}
	tupitem->t_len = __form_kern_heaptuple(kcxt,
										   &tupitem->htup,
										   kds_dst->ncols,
										   kds_dst->colmeta,
										   htup,
										   0,	/* not a composite type */
										   0,	/* not a composite type */
										   htuple_oid,
										   tup_dclass,
										   tup_values);
	return tupitem->t_len;
}

/*
 * form_kern_composite_type
 *
 * A utility routine to set up a composite type data structure
 * on the supplied pre-allocated region. It in
 *
 * @buffer     ... pointer to global memory where caller wants to construct
 *                 a composite datum. It must have enough length and also
 *                 must be aligned to DWORD.
 * @typeoid    ... type OID of the composite type 
 * @typemod    ... type modifier of the composite type
 * @nfields    ... number of sub-fields of the composite type
 * @colmeta    ... array of kern_colmeta for sub-field types
 * @tup_dclass ... any of DATUM_CLASS__*
 * @tup_values ... values of the sub-fields
 */
STATIC_INLINE(cl_uint)
form_kern_composite_type(kern_context *kcxt,
						 void      *buffer,      /* out */
						 cl_uint    comp_typeid, /* in: type OID */
						 cl_int		comp_typmod, /* in: type modifier */
						 cl_int		nfields,     /* in: # of attributes */
						 kern_colmeta *colmeta,  /* in: sub-type attributes */
						 cl_char   *tup_dclass,  /* in: */
						 Datum	   *tup_values)	 /* in: */
{
	return __form_kern_heaptuple(kcxt,
								 buffer,
								 nfields,
								 colmeta,
								 NULL,
								 comp_typmod,
								 comp_typeid,
								 0,	/* composite type never have OID */
								 tup_dclass,
								 tup_values);
}
#endif /* __CUDACC__ */

#ifdef __CUDACC__
/*
 * A utility function to evaluate pg_bool_t value as if built-in
 * bool variable.
 */
STATIC_INLINE(cl_bool)
EVAL(pg_bool_t arg)
{
	if (!arg.isnull && arg.value != 0)
		return true;
	return false;
}

STATIC_INLINE(pg_bool_t)
NOT(pg_bool_t arg)
{
	arg.value = !arg.value;
	return arg;
}

STATIC_INLINE(pg_bool_t)
to_bool(cl_bool value)
{
	pg_bool_t	result;

	result.isnull = false;
	result.value  = value;

	return result;
}

/*
 * Support routine for CASE x WHEN y then ... else ... end
 */
template <typename E, typename T>
STATIC_INLINE(T)
PG_CASEWHEN_ELSE(kern_context *kcxt,
				 const E& case_val,
				 const T& else_val)
{
	return else_val;
}

template <typename E, typename T, typename ...R>
STATIC_INLINE(T)
PG_CASEWHEN_ELSE(kern_context *kcxt,
				 const E& case_val,
				 const T& else_val,
				 const E& test_val,
				 const T& then_val,
				 const R&... args_rest)
{
	pg_int4_t	cmp;

	if (!case_val.isnull && !test_val.isnull)
	{
		cmp = pgfn_type_compare(kcxt, case_val, test_val);
		if (!cmp.isnull && cmp.value == 0)
			return then_val;
	}
	return PG_CASEWHEN_ELSE(kcxt, case_val, else_val, args_rest...);
}

template <typename E, typename T, typename ...R>
STATIC_INLINE(T)
PG_CASEWHEN_EXPR(kern_context *kcxt,
				 const E& case_val,
				 const E& test_val,
				 const T& then_val,
				 const R&... args_rest)
{
	pg_int4_t	cmp;
	E			else_val;

	if (!case_val.isnull && !test_val.isnull)
	{
		cmp = pgfn_type_compare(kcxt, case_val, test_val);
		if (!cmp.isnull && cmp.value == 0)
			return then_val;
	}
	else_val.isnull = true;
	return PG_CASEWHEN_ELSE(kcxt, case_val, else_val, args_rest...);
}

/*
 * Support routine for COALESCE / GREATEST / LEAST
 */
template <typename T>
STATIC_INLINE(T)
PG_COALESCE(kern_context *kcxt, const T& arg)
{
	return arg;
}

template <typename T, typename ...R>
STATIC_INLINE(T)
PG_COALESCE(kern_context *kcxt, const T& arg1, const R&... args_rest)
{
	if (!arg1.isnull)
		return arg1;
	return PG_COALESCE(kcxt, args_rest...);
}

template <typename T>
STATIC_INLINE(T)
PG_GREATEST(kern_context *kcxt, const T& arg)
{
	return arg;
}

template <typename T, typename ...R>
STATIC_INLINE(T)
PG_GREATEST(kern_context *kcxt, const T& arg1, const R&... args_rest)
{
	if (arg1.isnull)
		return PG_GREATEST(kcxt, args_rest...);
	else
	{
		T			arg2 = PG_GREATEST(kcxt, args_rest...);
		pg_int4_t	cmp;

		cmp = pgfn_type_compare(kcxt, arg1, arg2);
		if (cmp.isnull)
			return arg1;
		else if (cmp.value > 0)
			return arg1;
		else
			return arg2;
	}
}

template <typename T>
STATIC_INLINE(T)
PG_LEAST(kern_context *kcxt, const T& arg)
{
	return arg;
}

template <typename T, typename... R>
STATIC_INLINE(T)
PG_LEAST(kern_context *kcxt, const T& arg1, const R&... args_rest)
{
	if (arg1.isnull)
		return PG_LEAST(kcxt, args_rest...);
	else
	{
		T			arg2 = PG_LEAST(kcxt, args_rest...);
		pg_int4_t	cmp;

		cmp = pgfn_type_compare(kcxt, arg1, arg2);
		if (cmp.isnull)
			return arg1;
		else if (cmp.value > 0)
			return arg2;
		else
			return arg1;
	}
}

/*
 * Support routine for NullTest
 */
template <typename T>
STATIC_INLINE(pg_bool_t)
PG_ISNULL(kern_context *kcxt, T arg)
{
	pg_bool_t	result;

	result.isnull = false;
	result.value = arg.isnull;

	return result;
}

template <typename T>
STATIC_INLINE(pg_bool_t)
PG_ISNOTNULL(kern_context *kcxt, T arg)
{
	pg_bool_t	result;

	result.isnull = false;
	result.value = !arg.isnull;

	return result;
}

/*
 * Functions for BooleanTest
 */
STATIC_INLINE(pg_bool_t)
pgfn_bool_is_true(kern_context *kcxt, pg_bool_t result)
{
	result.value = (!result.isnull && result.value);
	result.isnull = false;
	return result;
}

STATIC_INLINE(pg_bool_t)
pgfn_bool_is_not_true(kern_context *kcxt, pg_bool_t result)
{
	result.value = (result.isnull || !result.value);
	result.isnull = false;
	return result;
}

STATIC_INLINE(pg_bool_t)
pgfn_bool_is_false(kern_context *kcxt, pg_bool_t result)
{
	result.value = (!result.isnull && !result.value);
	result.isnull = false;
	return result;
}

STATIC_INLINE(pg_bool_t)
pgfn_bool_is_not_false(kern_context *kcxt, pg_bool_t result)
{
	result.value = (result.isnull || result.value);
	result.isnull = false;
	return result;
}

STATIC_INLINE(pg_bool_t)
pgfn_bool_is_unknown(kern_context *kcxt, pg_bool_t result)
{
	result.value = result.isnull;
	result.isnull = false;
	return result;
}

STATIC_INLINE(pg_bool_t)
pgfn_bool_is_not_unknown(kern_context *kcxt, pg_bool_t result)
{
	result.value = !result.isnull;
	result.isnull = false;
	return result;
}
#endif /* __CUDACC__ */

#endif  /* CUDA_UTILS_H */
