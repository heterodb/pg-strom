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
						  uint32_t slot_id,
						  const char *addr)
{
	if (slot_id < kcxt->kvars_nslots)
	{
		const kern_varslot_desc *vs_desc = &kcxt->kvars_desc[slot_id];
		xpu_datum_t	   *xdatum = kcxt->kvars_slot[slot_id];

		if (!addr)
			xdatum->expr_ops = NULL;
		else
		{
			if (!vs_desc->vs_ops->xpu_datum_heap_read(kcxt, addr, xdatum))
				return false;
			assert(xdatum->expr_ops == vs_desc->vs_ops);
		}
		return true;
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
		uint16_t	slot_id = vl_desc->vl_slot_id;

		kcxt->kvars_slot[slot_id]->expr_ops = NULL;
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
		const kern_varslot_desc *vs_desc = &kcxt->kvars_desc[slot_id];
		xpu_datum_t	   *xdatum = kcxt->kvars_slot[slot_id];
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
		if (vs_desc->vs_ops->xpu_datum_heap_read(kcxt, addr, xdatum))
		{
			assert(XPU_DATUM_ISNULL(xdatum) || xdatum->expr_ops == vs_desc->vs_ops);
			return true;
		}
		return false;
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
			return false;
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
			if (!__extract_heap_tuple_attr(kcxt, vl_desc->vl_slot_id, addr))
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
			if (!__extract_heap_tuple_attr(kcxt, vl_desc->vl_slot_id, addr))
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
STATIC_FUNCTION(bool)
__kern_extract_arrow_field(kern_context *kcxt,
						   const kern_data_store *kds,
						   const kern_colmeta *cmeta,
						   uint32_t kds_index,
						   const kern_varslot_desc *vs_desc,
						   xpu_datum_t *result)
{
	if (!KDS_ARROW_CHECK_ISNULL(kds, cmeta, kds_index))
	{
		if (!vs_desc->vs_ops->xpu_datum_arrow_read(kcxt,
												   kds,
												   cmeta,
												   kds_index,
												   result))
			return false;
		/* If Arrow file is sane, NULL is not welcome */
		assert(vs_desc->vs_ops == result->expr_ops);
		if (result->expr_ops == &xpu_array_ops)
		{
			uint32_t	slot_id = (vs_desc - kcxt->kvars_desc);
			assert(slot_id < kcxt->kvars_nrooms);
			((xpu_array_t *)result)->u.arrow.slot_id = slot_id;
		}
		else if (result->expr_ops == &xpu_composite_ops)
		{
			uint32_t	slot_id = (vs_desc - kcxt->kvars_desc);
			assert(slot_id < kcxt->kvars_nrooms);
			((xpu_composite_t *)result)->u.arrow.slot_id = slot_id;
		}
	}
	else
	{
		result->expr_ops = NULL;
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
	return __extract_heap_tuple_attr(kcxt, vl_desc->vl_slot_id, (char *)&dummy);
}

STATIC_FUNCTION(bool)
kern_extract_arrow_tuple(kern_context *kcxt,
						 const kern_data_store *kds,
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
		const kern_colmeta *cmeta = &kds->colmeta[vl_desc->vl_resno-1];
		uint16_t	slot_id = vl_desc->vl_slot_id;

		if (cmeta->virtual_offset != 0)
		{
			/* virtual column is immutable for each KDS */
			const char *addr = NULL;

			if (cmeta->virtual_offset > 0)
				addr = ((char *)kds + cmeta->virtual_offset);
			if (!__extract_heap_tuple_attr(kcxt, slot_id, addr))
				return false;
		}
		else if (!__kern_extract_arrow_field(kcxt,
											 kds,
											 cmeta,
											 kds_index,
											 &kcxt->kvars_desc[slot_id],
											 kcxt->kvars_slot[slot_id]))
		{
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

	if (kexp->u.c.const_isnull)
		__result->expr_ops = NULL;
	else
	{
		if (!expr_ops->xpu_datum_heap_read(kcxt, kexp->u.c.const_value, __result))
			return false;
		assert(expr_ops == __result->expr_ops);
	}
	return true;
}

STATIC_FUNCTION(bool)
pgfn_ParamExpr(XPU_PGFUNCTION_ARGS)
{
	const xpu_datum_operators *expr_ops = kexp->expr_ops;
	kern_session_info *session = kcxt->session;
	uint32_t		param_id = kexp->u.p.param_id;

	if (param_id < session->nparams && session->poffset[param_id] != 0)
	{
		const char *addr = ((char *)session + session->poffset[param_id]);

		if (!expr_ops->xpu_datum_heap_read(kcxt, addr, __result))
			return false;
		assert(expr_ops == __result->expr_ops);
	}
	else
	{
		__result->expr_ops = NULL;
	}
	return true;
}

STATIC_FUNCTION(bool)
pgfn_VarExpr(XPU_PGFUNCTION_ARGS)
{
	const xpu_datum_operators *expr_ops = kexp->expr_ops;

	if (kexp->u.v.var_offset < 0)
	{
		uint16_t		slot_id = kexp->u.v.var_slot_id;
		xpu_datum_t	   *__xdatum = kcxt->kvars_slot[slot_id];

		if (slot_id >= kcxt->kvars_nslots)
		{
			STROM_ELOG(kcxt, "Bug? slot_id is out of range");
			return false;
		}
		if (__result != __xdatum)
			memcpy(__result, __xdatum, expr_ops->xpu_type_sizeof);
		assert(XPU_DATUM_ISNULL(__result) || __result->expr_ops == expr_ops);
	}
	else
	{
		const kvec_datum_t *kvecs = (kvec_datum_t *)(kcxt->kvecs_curr_buffer +
													 kexp->u.v.var_offset);
		assert(kcxt->kvecs_curr_id < KVEC_UNITSZ);
		if (kvecs->isnull[kcxt->kvecs_curr_id])
			__result->expr_ops = NULL;
		else if (!expr_ops->xpu_datum_kvec_load(kcxt, kvecs,
												kcxt->kvecs_curr_id,
												__result))
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
			return false;
		}
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
	xpu_bool_t	   *result = (xpu_bool_t *)__result;
	xpu_datum_t	   *xdatum;
	const kern_expression *karg = KEXP_FIRST_ARG(kexp);

	assert(kexp->nr_args == 1);
	xdatum = (xpu_datum_t *)alloca(karg->expr_ops->xpu_type_sizeof);
	if (!EXEC_KERN_EXPRESSION(kcxt, karg, xdatum))
		return false;
	result->expr_ops = &xpu_bool_ops;
	switch (kexp->opcode)
	{
		case FuncOpCode__NullTestExpr_IsNull:
			result->value = XPU_DATUM_ISNULL(xdatum);
			break;
		case FuncOpCode__NullTestExpr_IsNotNull:
			result->value = !XPU_DATUM_ISNULL(xdatum);
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
	int			i;

	/* CASE <key> expression, if any */
	if (kexp->u.casewhen.case_comp)
	{
		karg = (const kern_expression *)
			((char *)kexp + kexp->u.casewhen.case_comp);
		assert(__KEXP_IS_VALID(kexp, karg) &&
			   karg->opcode == FuncOpCode__SaveExpr);
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, NULL))
			return false;
	}

	/* evaluate each WHEN-clauses */
	assert((kexp->nr_args % 2) == 0);
	for (i = 0, karg=KEXP_FIRST_ARG(kexp);
		 i < kexp->nr_args;
		 i += 2, karg=KEXP_NEXT_ARG(karg))
	{
		xpu_bool_t		status;

		assert(__KEXP_IS_VALID(kexp, karg) &&
			   karg->exptype == TypeOpCode__bool);
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &status))
			return false;

		karg = KEXP_NEXT_ARG(karg);
		assert(__KEXP_IS_VALID(kexp, karg));
		if (!XPU_DATUM_ISNULL(&status) && status.value)
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
	const kern_varslot_desc *vs_desc = &kcxt->kvars_desc[kexp->u.saop.elem_slot_id];
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
			if (vs_desc->vs_typlen > 0)
				offset = TYPEALIGN(vs_desc->vs_typalign, offset);
			else if (!VARATT_NOT_PAD_BYTE(base + offset))
				offset = TYPEALIGN(vs_desc->vs_typalign, offset);
			addr = base + offset;

			if (vs_desc->vs_typlen > 0)
				offset += vs_desc->vs_typlen;
			else if (vs_desc->vs_typlen == -1)
				offset += VARSIZE_ANY(addr);
			else
			{
				STROM_ELOG(kcxt, "not a supported attribute length");
				return false;
			}
		}
		/* load the array element */
		if (!__extract_heap_tuple_attr(kcxt, kexp->u.saop.elem_slot_id, addr))
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
	const kern_colmeta *cmeta = aval->u.arrow.cmeta;
	const kern_colmeta *smeta;
	const kern_data_store *kds;
	uint32_t	slot_id = kexp->u.saop.elem_slot_id;
	bool		use_any = (kexp->opcode == FuncOpCode__ScalarArrayOpAny);
	bool		meet_nulls = false;

	result->value = !use_any;
	kds = (const kern_data_store *)
		((char *)cmeta - cmeta->kds_offset);
	smeta = &kds->colmeta[cmeta->idx_subattrs];
	assert(cmeta->num_subattrs == 1);
	assert(slot_id < kcxt->kvars_nslots);
	for (int k=0; k < aval->length; k++)
	{
		uint32_t	index = aval->u.arrow.start + k;
		xpu_bool_t	status;

		/* load the element */
		if (!__kern_extract_arrow_field(kcxt, kds, smeta, index,
										&kcxt->kvars_desc[slot_id],
										kcxt->kvars_slot[slot_id]))
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
		   kexp->u.saop.elem_slot_id < kcxt->kvars_nslots);
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
__kern_form_heaptuple(kern_context *kcxt,
					  int proj_nattrs,
					  const uint16_t *proj_slot_id,
					  const kern_data_store *kds_dst,
					  HeapTupleHeaderData *htup)
{
	/*
	 * NOTE: the caller must call kern_estimate_heaptuple() preliminary to ensure
	 *       kcxt->kvars_slot[] are filled-up by the scalar values to be written.
	 */
	uint32_t	t_hoff;
	uint16_t	t_infomask = 0;
	bool		t_hasnull = false;

	if (kds_dst && kds_dst->ncols < proj_nattrs)
		proj_nattrs = kds_dst->ncols;
	/* has any NULL attributes? */
	for (int j=0; j < proj_nattrs; j++)
	{
		uint16_t		slot_id = proj_slot_id[j];

		if (slot_id >= kcxt->kvars_nslots ||
			XPU_DATUM_ISNULL(kcxt->kvars_slot[slot_id]))
		{
			t_infomask |= HEAP_HASNULL;
			t_hasnull = true;
			break;
		}
	}
	/* set up headers */
	t_hoff = offsetof(HeapTupleHeaderData, t_bits);
	if (t_hasnull)
		t_hoff += BITMAPLEN(proj_nattrs);
	t_hoff = MAXALIGN(t_hoff);

	if (htup)
	{
		memset(htup, 0, t_hoff);
		htup->t_choice.t_datum.datum_typmod = kds_dst->tdtypmod;
		htup->t_choice.t_datum.datum_typeid = kds_dst->tdtypeid;
		htup->t_ctid.ip_blkid.bi_hi = 0xffff;	/* InvalidBlockNumber */
		htup->t_ctid.ip_blkid.bi_lo = 0xffff;
		htup->t_ctid.ip_posid = 0;				/* InvalidOffsetNumber */
		htup->t_infomask2 = (proj_nattrs & HEAP_NATTS_MASK);
		htup->t_hoff = t_hoff;
	}

	/* walks on the source attributes */
	for (int j=0; j < proj_nattrs; j++)
	{
		const kern_colmeta *cmeta_dst = &kds_dst->colmeta[j];
		uint16_t		slot_id = proj_slot_id[j];
		xpu_datum_t	   *xdatum;
		int				sz, t_next;
		char		   *buffer = NULL;

		if (slot_id >= kcxt->kvars_nslots)
			continue;	/* considered as NULL */
		xdatum = kcxt->kvars_slot[slot_id];
		if (!XPU_DATUM_ISNULL(xdatum))
		{
			/* adjust alignment */
			t_next = TYPEALIGN(cmeta_dst->attalign, t_hoff);
			if (htup)
			{
				if (t_next > t_hoff)
					memset((char *)htup + t_hoff, 0, t_next - t_hoff);
				buffer = (char *)htup + t_next;
			}
			/* write-out or estimate length */
			sz = xdatum->expr_ops->xpu_datum_write(kcxt,
												   buffer,
												   cmeta_dst,
												   xdatum);
			if (sz < 0)
				return -1;
			/* update t_infomask if varlena */
			if (cmeta_dst->attlen == -1)
			{
				if (buffer && VARATT_IS_EXTERNAL(buffer))
					t_infomask |= HEAP_HASEXTERNAL;
				t_infomask |= HEAP_HASVARWIDTH;
			}
			/* set not-null bit, if valid */
			if (htup && t_hasnull)
				htup->t_bits[j>>3] |= (1<<(j & 7));
			t_hoff = t_next + sz;
		}
	}

	if (htup)
	{
		htup->t_infomask = t_infomask;
		SET_VARSIZE(&htup->t_choice.t_datum, t_hoff);
	}
	return t_hoff;	
}

PUBLIC_FUNCTION(int)
kern_form_heaptuple(kern_context *kcxt,
					const kern_expression *kexp_proj,
					const kern_data_store *kds_dst,
					HeapTupleHeaderData *htup)
{
	return __kern_form_heaptuple(kcxt,
								 kexp_proj->u.proj.nattrs,
								 kexp_proj->u.proj.slot_id,
								 kds_dst,
								 htup);
}

EXTERN_FUNCTION(int)
kern_estimate_heaptuple(kern_context *kcxt,
                        const kern_expression *kexp_proj,
                        const kern_data_store *kds_dst)
{
	const kern_expression *karg;
	int		i, sz;

	/* Run SaveExpr expressions to warm up kcxt->kvars_slot[] */
	for (i=0, karg=KEXP_FIRST_ARG(kexp_proj);
		 i < kexp_proj->nr_args;
		 i++, karg=KEXP_NEXT_ARG(karg))
	{
		xpu_datum_t	   *xdatum;
		uint16_t		slot_id;

		assert(__KEXP_IS_VALID(kexp_proj, karg));
		if (karg->opcode == FuncOpCode__VarExpr)
			slot_id = karg->u.v.var_slot_id;
		else
		{
			assert(karg->opcode == FuncOpCode__SaveExpr);
			slot_id = karg->u.save.sv_slot_id;
		}
		assert(slot_id < kcxt->kvars_nslots);
		xdatum = kcxt->kvars_slot[slot_id];
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
			hash = pg_hash_merge(hash, __hash);
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
	xpu_datum_t	   *xdatum = kcxt->kvars_slot[slot_id];

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
	STROM_ELOG(kcxt, "pgfn_JoinQuals should not be called as a normal kernel expression");
	return false;
}

/*
 * ExecGpuJoinQuals - runs JoinQuals operator.
 *   result =  1 : matched to JoinQuals
 *          =  0 : unmatched to JoinQuals
 *          = -1 : matched to JoinQuals, but unmatched to any of other quals
 *                 --> don't generate inner join row, but set outer-join-map.
 */
PUBLIC_FUNCTION(bool)
ExecGpuJoinQuals(kern_context *kcxt,
				 const kern_expression *kexp,
				 int *p_status)
{
	const kern_expression *karg;
	int		i, status = 1;

	assert(kexp->opcode  == FuncOpCode__JoinQuals &&
		   kexp->exptype == TypeOpCode__bool);
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
	*p_status = status;
	return true;
}

/*
 * ExecGpuJoinOtherQuals - runs OtherQuals in the JoinQuals if any.
 *   it is usually used in validation of RIGHT OUTER JOIN row.
 */
PUBLIC_FUNCTION(bool)
ExecGpuJoinOtherQuals(kern_context *kcxt,
					  const kern_expression *kexp,
					  bool *p_status)
{
	const kern_expression *karg;
	bool	status = true;
	int		i;

	assert(kexp->opcode  == FuncOpCode__JoinQuals &&
		   kexp->exptype == TypeOpCode__bool);
	for (i=0, karg = KEXP_FIRST_ARG(kexp);
		 i < kexp->nr_args;
		 i++, karg = KEXP_NEXT_ARG(karg))
	{
		xpu_bool_t	datum;

		if ((karg->expflags & KEXP_FLAG__IS_PUSHED_DOWN) == 0)
			continue;
		if (!EXEC_KERN_EXPRESSION(kcxt, karg, &datum))
			return false;
		if (XPU_DATUM_ISNULL(&datum) || !datum.value)
		{
			status = false;
			break;
		}
	}
	*p_status = status;
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
	uint32_t			temp;

	assert(!cmeta->attbyval &&
		   cmeta->attlen == sizeof(GpuCacheSysattr) &&
		   cmeta->nullmap_offset == 0);		/* NOT NULL */
	sysatt = (GpuCacheSysattr *)
		((char *)kds + cmeta->values_offset);
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
			break;
		default:
			STROM_ELOG(kcxt, "not a supported system attribute reference");
			return false;
	}
	return __extract_heap_tuple_attr(kcxt, vl_desc->vl_slot_id, addr);
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
			addr = ((const char *)kds + cmeta->values_offset);

			if (cmeta->attlen > 0)
			{
				addr += cmeta->attlen * kds_index;
			}
			else
			{
				addr = ((char *)extra + ((uint64_t *)addr)[kds_index]);
			}
		}
		else
		{
			addr = NULL;
		}

		if (!__extract_heap_tuple_attr(kcxt, vl_desc->vl_slot_id, addr))
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
					 const kern_expression *kexp_load_vars,
					 const kern_expression *kexp_scan_quals,
					 const kern_data_store *kds,
					 const HeapTupleHeaderData *htup)
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
		else if (!HandleErrorIfCpuFallback(kcxt, 0, 0, false))
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
		}
		return false;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
ExecLoadVarsOuterArrow(kern_context *kcxt,
					   const kern_expression *kexp_load_vars,
					   const kern_expression *kexp_scan_quals,
					   const kern_data_store *kds,
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
		else if (!HandleErrorIfCpuFallback(kcxt, 0, 0, false))
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
		}
		return false;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
ExecLoadVarsOuterColumn(kern_context *kcxt,
						const kern_expression *kexp_load_vars,
						const kern_expression *kexp_scan_quals,
						const kern_data_store *kds,
						const kern_data_extra *extra,
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
		else if (!HandleErrorIfCpuFallback(kcxt, 0, 0, false))
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
		}
		return false;
	}
	return true;
}

PUBLIC_FUNCTION(bool)
ExecLoadKeysFromGroupByFinal(kern_context *kcxt,
							 const kern_data_store *kds_final,
							 const kern_tupitem *tupitem,
							 const kern_expression *kexp_groupby_actions)
{
	const char *pos = NULL;
	const uint8_t *nullmap = NULL;
	int			ncols = 0;

	if (tupitem)
	{
		if ((tupitem->htup.t_infomask & HEAP_HASNULL) != 0)
			nullmap = tupitem->htup.t_bits;
		ncols = (tupitem->htup.t_infomask2 & HEAP_NATTS_MASK);
		pos = (const char *)&tupitem->htup + tupitem->htup.t_hoff;
	}
	ncols = Min(kds_final->ncols, ncols);
	for (int j=0; j < kexp_groupby_actions->u.pagg.nattrs; j++)
	{
		const kern_aggregate_desc *desc = &kexp_groupby_actions->u.pagg.desc[j];
		const kern_colmeta *cmeta = &kds_final->colmeta[j];

		if (j >= ncols || (nullmap && att_isnull(j, nullmap)))
		{
			if (desc->action == KAGG_ACTION__VREF)
			{
				if (!__extract_heap_tuple_attr(kcxt, desc->arg0_slot_id, NULL))
					return false;
			}
		}
		else
		{
			assert(pos != NULL);
			pos = (char *)TYPEALIGN(cmeta->attalign, pos);
			if (desc->action == KAGG_ACTION__VREF)
			{
				if (!__extract_heap_tuple_attr(kcxt, desc->arg0_slot_id, pos))
					return false;
			}
			if (cmeta->attlen > 0)
				pos += cmeta->attlen;
			else if (cmeta->attlen == -1)
				pos += VARSIZE_ANY(pos);
			else
			{
				STROM_ELOG(kcxt, "unknown attribute length");
				return false;
			}
		}
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
						char *dst_kvec_buffer,
						int dst_kvec_id)
{
	if (!kexp_move_vars)
		return true;		/* nothing to move */
	assert(kexp_move_vars->opcode == FuncOpCode__MoveVars &&
		   kexp_move_vars->exptype == TypeOpCode__int4 &&
		   kexp_move_vars->nr_args == 0);
	assert(dst_kvec_id >= 0 && dst_kvec_id < KVEC_UNITSZ);
	for (int i=0; i < kexp_move_vars->u.move.nitems; i++)
	{
		const kern_varmove_desc *vm_desc = &kexp_move_vars->u.move.desc[i];
		const kern_varslot_desc *vs_desc = &kcxt->kvars_desc[vm_desc->vm_slot_id];

		assert(vm_desc->vm_slot_id >= 0 &&
			   vm_desc->vm_slot_id <  kcxt->kvars_nslots);
		assert(vm_desc->vm_offset +
			   vs_desc->vs_ops->xpu_kvec_sizeof <= kcxt->kvecs_bufsz);
		if (vm_desc->vm_from_xdatum)
		{
			/* xdatum -> kvec-buffer */
			xpu_datum_t	   *xdatum = kcxt->kvars_slot[vm_desc->vm_slot_id];
			kvec_datum_t   *dst_kvec = (kvec_datum_t *)(dst_kvec_buffer +
														vm_desc->vm_offset);
			if (XPU_DATUM_ISNULL(xdatum))
				dst_kvec->isnull[dst_kvec_id] = true;
			else
			{
				dst_kvec->isnull[dst_kvec_id] = false;
				assert(xdatum->expr_ops == vs_desc->vs_ops);
				if (!xdatum->expr_ops->xpu_datum_kvec_save(kcxt,
														   xdatum,
														   dst_kvec,
														   dst_kvec_id))
				{
					assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
					return false;
				}
			}
		}
		else
		{
			/* kvec-buffer -> kvec-buffer */
			const kvec_datum_t *src_kvec;
			kvec_datum_t	   *dst_kvec;
			uint32_t			src_kvec_id = kcxt->kvecs_curr_id;

			assert(kcxt->kvecs_curr_buffer != NULL);
			assert(kcxt->kvecs_curr_id < KVEC_UNITSZ);
			src_kvec = (const kvec_datum_t *)(kcxt->kvecs_curr_buffer +
											  vm_desc->vm_offset);
			dst_kvec = (kvec_datum_t *)(dst_kvec_buffer +
										vm_desc->vm_offset);
			if (src_kvec->isnull[src_kvec_id])
				dst_kvec->isnull[dst_kvec_id] = true;
			else
			{
				dst_kvec->isnull[dst_kvec_id] = false;
				if (!vs_desc->vs_ops->xpu_datum_kvec_copy(kcxt,
														  src_kvec, src_kvec_id,
														  dst_kvec, dst_kvec_id))
				{
					assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
					return false;
				}
			}
		}
	}
	return true;
}


/* ------------------------------------------------------------
 *
 * Routines to support CPU Fallback
 *
 * ------------------------------------------------------------
 */
STATIC_FUNCTION(bool)
__writeOutCpuFallbackTuple(kern_context *kcxt,
						   int fallback_nattrs,
						   uint16_t *fallback_slots,
						   int depth,
						   uint64_t l_state,
						   bool matched)
{
	kern_data_store *kds_fallback = kcxt->kds_fallback;
	kern_fallbackitem *fbitem;
	uint64_t	__usage;
	uint32_t	__nitems_old;
	uint32_t	__nitems_cur;
	uint32_t	__nitems_new;
	int			reqsz;
	int			tupsz;

	/* estimate length of the fallback tuple */
	tupsz = __kern_form_heaptuple(kcxt,
								  fallback_nattrs,
								  fallback_slots,
								  kds_fallback,
								  NULL);
	if (tupsz < 0)
	{
		STROM_ELOG(kcxt, "fallback: unable to compute tuple size");
		return false;
	}
	reqsz = MAXALIGN(offsetof(kern_fallbackitem, htup) + tupsz);
	/* allocation of fallback buffer */
	__usage = __atomic_add_uint64(&kds_fallback->usage, reqsz);
	__usage += reqsz;
	__nitems_cur = __volatileRead(&kds_fallback->nitems);
	do {
		__nitems_old = __nitems_cur;
		__nitems_new = __nitems_cur + 1;
		if (!__KDS_CHECK_OVERFLOW(kds_fallback,
								  __nitems_new,
								  __usage))
		{
			/* needs expand the fallback buffer */
			return false;
		}
	} while ((__nitems_cur = __atomic_cas_uint32(&kds_fallback->nitems,
												 __nitems_old,
												 __nitems_new)) != __nitems_old);
	/* write out the fallback tuple */
	assert(depth >= 0 && depth <= USHRT_MAX);
	fbitem = (kern_fallbackitem *)((char *)kds_fallback
								   + kds_fallback->length
								   - __usage);
	fbitem->t_len = tupsz;
	fbitem->depth = depth;
	fbitem->matched = matched;
	fbitem->l_state = l_state;
	__kern_form_heaptuple(kcxt,
						  fallback_nattrs,
						  fallback_slots,
						  kcxt->kds_fallback,
						  &fbitem->htup);
	KDS_GET_ROWINDEX(kds_fallback)[__nitems_old] = __usage;

	return true;
}

PUBLIC_FUNCTION(bool)
HandleErrorIfCpuFallback(kern_context *kcxt,
						 int depth,
						 uint64_t l_state,
						 bool matched)
{
	if (kcxt->errcode == ERRCODE_SUSPEND_FALLBACK &&
		kcxt->kds_fallback != NULL)
	{
		kern_session_info *session = kcxt->session;
		kern_data_store   *kds_fallback = kcxt->kds_fallback;
		int			fallback_ncols = kds_fallback->ncols;
		kern_fallback_desc *fb_desc_array = (kern_fallback_desc *)
			((char *)session + session->fallback_desc_defs);
		int			fallback_nitems = session->fallback_desc_nitems;
		uint16_t   *fallback_slots = (uint16_t *)
			alloca(sizeof(uint16_t) * fallback_ncols);
		assert(session->fallback_desc_defs > 0);

		/*
		 * Load variables from the kvec-buffer (if depth>0)
		 */
		memset(fallback_slots, -1, sizeof(uint16_t) * fallback_ncols);
		for (int j=0; j < fallback_nitems; j++)
		{
			kern_fallback_desc *fb_desc = &fb_desc_array[j];
			int		slot_id = fb_desc->fb_slot_id;

			assert(slot_id >= 0 && slot_id < kcxt->kvars_nslots);
			assert(fb_desc->fb_dst_resno > 0 &&
				   fb_desc->fb_dst_resno <= kds_fallback->ncols);
			if (depth >  fb_desc->fb_src_depth &&
				depth <= fb_desc->fb_max_depth)
			{
				const kern_varslot_desc *vs_desc = &kcxt->kvars_desc[slot_id];
				const kvec_datum_t *kvecs = (const kvec_datum_t *)
					((char *)kcxt->kvecs_curr_buffer + vs_desc->vs_offset);

				assert(vs_desc->vs_offset >= 0 &&
					   vs_desc->vs_offset < kcxt->kvecs_bufsz);
				if (!vs_desc->vs_ops->xpu_datum_kvec_load(kcxt,
														  kvecs,
														  kcxt->kvecs_curr_id,
														  kcxt->kvars_slot[slot_id]))
				{
					STROM_ELOG(kcxt, "fallback: unable to load variables");
					return false;
				}
			}
			else if (depth != 0 || fb_desc->fb_src_depth != 0)
			{
				/* depth==0 is already loaded by the caller */
				kcxt->kvars_slot[slot_id]->expr_ops = NULL;
			}
			fallback_slots[fb_desc->fb_dst_resno - 1] = slot_id;
		}
		/* try write out a fallback tuple */
		if (__writeOutCpuFallbackTuple(kcxt,
									   fallback_ncols,
									   fallback_slots,
									   depth, l_state, matched))
		{
			/* successfull fallbacked, clear the error code */
			kcxt->errcode = ERRCODE_STROM_SUCCESS;
			return true;
		}
	}
	return false;
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
			return __extract_heap_tuple_attr(kcxt, vl_desc->vl_slot_id, addr);
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
			return __extract_heap_tuple_attr(kcxt, vl_desc->vl_slot_id, addr);
	}
	/* fill-up by NULL, if not found */
	return __extract_heap_tuple_attr(kcxt, vl_desc->vl_slot_id, NULL);
}

PUBLIC_FUNCTION(uint64_t)
ExecGiSTIndexGetNext(kern_context *kcxt,
					 const kern_data_store *kds_hash,
					 const kern_data_store *kds_gist,
					 const kern_expression *kexp_gist,
					 uint64_t l_state)
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
		size_t		l_off = l_state;
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
			return ULONG_MAX;
		}
		/* runs index-qualifier */
		if (!EXEC_KERN_EXPRESSION(kcxt, karg_gist, &status))
		{
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
			/*
			 * XXX - right now, we don't support CPU fallback in the GiST-Join
			 *       index qualifiers. So, we just rewrite error code to stop
			 *       execution.
			 */
			if (kcxt->errcode == ERRCODE_SUSPEND_FALLBACK)
				kcxt->errcode = ERRCODE_DEVICE_ERROR;
			return ULONG_MAX;
		}
		/* check result */
		if (!XPU_DATUM_ISNULL(&status) && status.value)
		{
			BlockNumber		block_nr;

			if (GistPageIsLeaf(gist_page))
			{
				const kern_varslot_desc *vs_desc;
				uint32_t	slot_id = kexp_gist->u.gist.htup_slot_id;
				uint32_t	rowid;
				const kern_tupitem *tupitem;

				assert(itup->t_tid.ip_posid == InvalidOffsetNumber);
				rowid = ((uint32_t)itup->t_tid.ip_blkid.bi_hi << 16 |
						 (uint32_t)itup->t_tid.ip_blkid.bi_lo);
				tupitem = KDS_GET_TUPITEM(kds_hash, rowid);
				assert(slot_id < kcxt->kvars_nslots);
				vs_desc = &kcxt->kvars_desc[slot_id];
				assert(vs_desc->vs_ops == &xpu_internal_ops);
				if (!vs_desc->vs_ops->xpu_datum_heap_read(kcxt, &tupitem->htup,
														  kcxt->kvars_slot[slot_id]))
				{
					assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
					return ULONG_MAX;
				}
				/* returns the offset of the next line item pointer */
				return ((char *)(lpp+1) - (char *)(kds_gist));
			}
			block_nr = ((BlockNumber)itup->t_tid.ip_blkid.bi_hi << 16 |
						(BlockNumber)itup->t_tid.ip_blkid.bi_lo);
			assert(block_nr < kds_gist->nitems);
			gist_page = KDS_BLOCK_PGPAGE(kds_gist, block_nr);
			start = FirstOffsetNumber;
			goto restart;
		}
	}

	if (GistFollowRight(gist_page))
	{
		/* move to the next chain, if any */
		uint32_t	rightlink = GistPageGetOpaque(gist_page)->rightlink;

		gist_page = KDS_BLOCK_PGPAGE(kds_gist, rightlink);
		start = FirstOffsetNumber;
		goto restart;
	}
	else if (!GistPageIsRoot(gist_page))
	{
		/* pop to the parent page if not found */
		start = gist_page->pd_parent_item + 1;
		gist_page = KDS_BLOCK_PGPAGE(kds_gist, gist_page->pd_parent_blkno);
		goto restart;
	}
	return ULONG_MAX;	/* no more chance for this outer */
}

PUBLIC_FUNCTION(bool)
ExecGiSTIndexPostQuals(kern_context *kcxt,
					   int depth,
					   const kern_data_store *kds_hash,
					   const kern_expression *kexp_gist,
					   const kern_expression *kexp_load,
					   const kern_expression *kexp_join)
{
	const kvec_internal_t *kvecs;
	HeapTupleHeaderData *htup;
	uint32_t		slot_id;
	int				status;

	/* fetch the inner heap tuple */
	assert(kexp_gist->opcode == FuncOpCode__GiSTEval);
	slot_id = kexp_gist->u.gist.htup_slot_id;
	assert(slot_id < kcxt->kvars_nslots);
	/* load the inner heap tuple */
	assert(kcxt->kvars_desc[slot_id].vs_type_code == TypeOpCode__internal);
	kvecs = (const kvec_internal_t *)(kcxt->kvecs_curr_buffer +
									  kexp_gist->u.gist.htup_offset);
	assert(kcxt->kvecs_curr_id < KVEC_UNITSZ);
	assert(!kvecs->isnull[kcxt->kvecs_curr_id]);
	htup = (HeapTupleHeaderData *)kvecs->values[kcxt->kvecs_curr_id];
	assert((char *)htup >= (char *)kds_hash &&
		   (char *)htup <  (char *)kds_hash + kds_hash->length);
	if (!ExecLoadVarsHeapTuple(kcxt, kexp_load, depth, kds_hash, htup))
	{
		STROM_ELOG(kcxt, "Bug? GiST-Index prep fetched corrupted tuple");
		return false;
	}
	/* run the join quals */
	kcxt_reset(kcxt);
	if (!ExecGpuJoinQuals(kcxt, kexp_join, &status))
	{
		if (!HandleErrorIfCpuFallback(kcxt, depth, 0, false))
			assert(kcxt->errcode != ERRCODE_STROM_SUCCESS);
		return false;
	}
	return (status > 0);
}

/* ----------------------------------------------------------------
 *
 * xpu_arrow_t device type support routine
 *
 * ----------------------------------------------------------------
 */
STATIC_FUNCTION(bool)
xpu_array_datum_heap_read(kern_context *kcxt,
						  const void *addr,
						  xpu_datum_t *__result)
{
	xpu_array_t *result = (xpu_array_t *)__result;

	result->expr_ops = &xpu_array_ops;
	result->length = -1;
	result->u.heap.value = (const varlena *)addr;
	return true;
}

STATIC_FUNCTION(bool)
xpu_array_datum_arrow_read(kern_context *kcxt,
						   const kern_data_store *kds,
						   const kern_colmeta *cmeta,
						   uint32_t kds_index,
						   xpu_datum_t *__result)
{
	xpu_array_t *result = (xpu_array_t *)__result;
	uint32_t	start;
	uint32_t	length;

	switch (cmeta->attopts.tag)
	{
		case ArrowType__List:
			{
				const uint32_t *base = (const uint32_t *)
					((const char *)kds + cmeta->values_offset);
				if (sizeof(uint32_t) * (kds_index+2) > cmeta->values_length)
				{
					STROM_ELOG(kcxt, "Arrow::List reference out of range");
					return false;
				}
				if (base[kds_index] > base[kds_index+1])
				{
					STROM_ELOG(kcxt, "Arrow::List looks corrupted");
					return false;
				}
				start  = base[kds_index];
				length = base[kds_index+1] - start;
			}
			break;
		case ArrowType__LargeList:
			{
				const uint64_t   *base = (const uint64_t *)
					((const char *)kds + cmeta->values_offset);
				if (sizeof(uint64_t) * (kds_index+2) > cmeta->values_length)
				{
					STROM_ELOG(kcxt, "Arrow::LargeList reference out of range");
					return false;
				}
				if (base[kds_index] > base[kds_index+1] ||
					base[kds_index+1] > INT_MAX)
				{
					STROM_ELOG(kcxt, "Arrow::LargeList looks corrupted");
					return false;
				}
				start  = base[kds_index];
				length = base[kds_index+1] - start;
			}
			break;
		default:
			STROM_ELOG(kcxt, "xpu_array_t must be mapped on Arrow::List or LargeList");
			return false;
	}
	result->expr_ops = &xpu_array_ops;
	result->length = length;
	result->u.arrow.cmeta = cmeta;
	result->u.arrow.start = start;
	result->u.arrow.slot_id = UINT_MAX;	/* should be set by the caller */
	return true;
}

STATIC_FUNCTION(bool)
xpu_array_datum_kvec_load(kern_context *kcxt,
						  const kvec_datum_t *__kvecs,
						  uint32_t kvecs_id,
						  xpu_datum_t *__result)
{
	const kvec_array_t *kvecs = (const kvec_array_t *)__kvecs;
	xpu_array_t *result = (xpu_array_t *)__result;

	result->expr_ops = &xpu_array_ops;
	result->length = kvecs->length[kvecs_id];
	if (result->length < 0)
		result->u.heap.value = kvecs->u.heap.values[kvecs_id];
	else
	{
		result->u.arrow.cmeta  = kvecs->cmeta;
		result->u.arrow.start  = kvecs->u.arrow.start[kvecs_id];
		result->u.arrow.slot_id = kvecs->slot_id;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_array_datum_kvec_save(kern_context *kcxt,
						  const xpu_datum_t *__xdatum,
						  kvec_datum_t *__kvecs,
						  uint32_t kvecs_id)
{
	const xpu_array_t *xdatum = (const xpu_array_t *)__xdatum;
	kvec_array_t *kvecs = (kvec_array_t *)__kvecs;

	if (xdatum->length < 0)
	{
		kvecs->length[kvecs_id] = xdatum->length;
		kvecs->u.heap.values[kvecs_id] = xdatum->u.heap.value;
	}
	else
	{
		kvecs->length[kvecs_id]        = xdatum->length;
		kvecs->u.arrow.start[kvecs_id] = xdatum->u.arrow.start;
		kvecs->cmeta                   = xdatum->u.arrow.cmeta;
		kvecs->slot_id                 = xdatum->u.arrow.slot_id;
	}
	return true;
}

STATIC_FUNCTION(bool)
xpu_array_datum_kvec_copy(kern_context *kcxt,
						  const kvec_datum_t *__kvecs_src,
						  uint32_t kvecs_src_id,
						  kvec_datum_t *__kvecs_dst,
						  uint32_t kvecs_dst_id)
{
	const kvec_array_t *kvecs_src = (const kvec_array_t *)__kvecs_src;
	kvec_array_t *kvecs_dst = (kvec_array_t *)__kvecs_dst;
	int32_t		length;

	length = kvecs_src->length[kvecs_src_id];
	if (length < 0)
	{
		kvecs_dst->length[kvecs_dst_id] = length;
		kvecs_dst->u.heap.values[kvecs_dst_id] = kvecs_src->u.heap.values[kvecs_src_id];
	}
	else
	{
		kvecs_dst->cmeta = kvecs_src->cmeta;
		kvecs_dst->slot_id = kvecs_src->slot_id;
		kvecs_dst->length[kvecs_dst_id] = length;
		kvecs_dst->u.arrow.start[kvecs_dst_id] = kvecs_src->u.arrow.start[kvecs_src_id];
	}
	return true;
}

STATIC_FUNCTION(int)
__xpu_array_arrow_write(kern_context *kcxt,
						char *buffer,
						const kern_colmeta *__cmeta_dst,	/* unused */
						const xpu_array_t *arg)
{
	const kern_varslot_desc *vs_desc;
	const kern_colmeta *cmeta = arg->u.arrow.cmeta;
	const kern_colmeta *emeta;	/* array element (source) */
	const kern_data_store *kds;
	xpu_datum_t	   *xdatum;
	uint8_t		   *nullmap = NULL;
	int32_t			sz, nbytes;

	/* setup source emeta/vs_desc */
	kds = (const kern_data_store *)
		((const char *)cmeta - cmeta->kds_offset);
	assert(cmeta->num_subattrs == 1);
	emeta = &kds->colmeta[cmeta->idx_subattrs];
	vs_desc = &kcxt->kvars_desc[arg->u.arrow.slot_id];
	assert(vs_desc->num_subfield == 1);
	vs_desc = &kcxt->kvars_desc[vs_desc->idx_subfield];
	xdatum = (xpu_datum_t *)alloca(vs_desc->vs_ops->xpu_type_sizeof);

	nbytes = (VARHDRSZ +
			  offsetof(__ArrayTypeData, data[2]) +
			  MAXALIGN(BITMAPLEN(arg->length)));
	if (buffer)
	{
		__ArrayTypeData *arr = (__ArrayTypeData *)(buffer + VARHDRSZ);

		memset(arr, 0, nbytes - VARHDRSZ);
		arr->ndim = 1;
		arr->elemtype = emeta->atttypid;
		arr->data[0] = arg->length;
		arr->data[1] = 1;
		if (emeta->nullmap_offset != 0)
			nullmap = (uint8_t *)&arr->data[2];
	}

	for (int k=0; k < arg->length; k++)
	{
		uint32_t	index = arg->u.arrow.start + k;

		if (!__kern_extract_arrow_field(kcxt,
										kds,
										emeta,
										index,
										vs_desc,
										xdatum))
			return false;
		if (!XPU_DATUM_ISNULL(xdatum))
		{
			int		__nbytes = nbytes;
			char   *pos;

			if (nullmap)
				nullmap[k>>3] |= (1<<(k & 7));
			/* alignment */
			nbytes = TYPEALIGN(emeta->attalign, nbytes);
			if (nbytes > __nbytes)
			{
				if (buffer)
					memset(buffer + __nbytes, 0, (nbytes - __nbytes));
			}
			pos = (buffer ? buffer + nbytes : NULL);
			sz = xdatum->expr_ops->xpu_datum_write(kcxt,
												   pos,
												   emeta,
												   xdatum);
			if (sz < 0)
				return -1;
			nbytes += sz;
		}
	}
	return nbytes;
}

STATIC_FUNCTION(int)
xpu_array_datum_write(kern_context *kcxt,
					  char *buffer,
					  const kern_colmeta *cmeta_dst,
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
		return nbytes;
	}
	return __xpu_array_arrow_write(kcxt, buffer, cmeta_dst, arg);
}

STATIC_FUNCTION(bool)
xpu_array_datum_hash(kern_context *kcxt,
					 uint32_t *p_hash,
					 xpu_datum_t *arg)
{
	STROM_ELOG(kcxt, "xpu_array_datum_hash is not implemented");
	return false;
}

STATIC_FUNCTION(bool)
xpu_array_datum_comp(kern_context *kcxt,
					 int *p_comp,
					 xpu_datum_t *__a,
					 xpu_datum_t *__b)
{
	STROM_ELOG(kcxt, "xpu_array_datum_comp is not implemented");
	return false;
}
//MEMO: some array type uses typalign=4. is it ok?
PGSTROM_SQLTYPE_OPERATORS(array,false,4,-1);

/* ----------------------------------------------------------------
 *
 * xpu_composite_t type support routine
 *
 * ----------------------------------------------------------------
 */
STATIC_FUNCTION(bool)
xpu_composite_datum_heap_read(kern_context *kcxt,
							  const void *addr,
							  xpu_datum_t *__result)
{
	xpu_composite_t *result = (xpu_composite_t *)__result;

	result->expr_ops = &xpu_composite_ops;
	result->cmeta = NULL;
	result->u.heap.value = (const varlena *)addr;
	return true;
}

STATIC_FUNCTION(bool)
xpu_composite_datum_arrow_read(kern_context *kcxt,
							  const kern_data_store *kds,
							   const kern_colmeta *cmeta,
							   uint32_t kds_index,
							   xpu_datum_t *__result)
{
	xpu_composite_t *result = (xpu_composite_t *)__result;

	result->expr_ops = &xpu_composite_ops;
	result->cmeta = cmeta;
	result->u.arrow.rowidx = kds_index;
	result->u.arrow.slot_id = UINT_MAX;

	return true;
}

STATIC_FUNCTION(bool)
xpu_composite_datum_kvec_load(kern_context *kcxt,
							  const kvec_datum_t *__kvecs,
							  uint32_t kvecs_id,
							  xpu_datum_t *__result)
{
	const kvec_composite_t *kvecs = (const kvec_composite_t *)__kvecs;
	xpu_composite_t *result = (xpu_composite_t *)__result;

	result->expr_ops = &xpu_composite_ops;
	if (!kvecs->cmeta)
	{
		result->cmeta = NULL;
		result->u.heap.value = kvecs->u.heap.values[kvecs_id];
	}
	else
	{
		result->cmeta = kvecs->cmeta;
		result->u.arrow.rowidx = kvecs->u.arrow.rowidx[kvecs_id];
		result->u.arrow.slot_id = kvecs->u.arrow.slot_id;
	}
    return true;
}

STATIC_FUNCTION(bool)
xpu_composite_datum_kvec_save(kern_context *kcxt,
							  const xpu_datum_t *__xdatum,
							  kvec_datum_t *__kvecs,
							  uint32_t kvecs_id)
{
	const xpu_composite_t *xdatum = (const xpu_composite_t *)__xdatum;
	kvec_composite_t *kvecs = (kvec_composite_t *)__kvecs;

	if (!xdatum->cmeta)
	{
		kvecs->cmeta = NULL;
		kvecs->u.heap.values[kvecs_id] = xdatum->u.heap.value;
	}
	else
	{
		kvecs->cmeta = xdatum->cmeta;
		kvecs->u.arrow.slot_id = xdatum->u.arrow.slot_id;
		kvecs->u.arrow.rowidx[kvecs_id] = xdatum->u.arrow.rowidx;
	}
    return true;
}

STATIC_FUNCTION(bool)
xpu_composite_datum_kvec_copy(kern_context *kcxt,
                          const kvec_datum_t *__kvecs_src,
                          uint32_t kvecs_src_id,
                          kvec_datum_t *__kvecs_dst,
                          uint32_t kvecs_dst_id)
{
	const kvec_composite_t *kvecs_src = (const kvec_composite_t *)__kvecs_src;
	kvec_composite_t *kvecs_dst = (kvec_composite_t *)__kvecs_dst;

	if (!kvecs_src->cmeta)
	{
		kvecs_dst->cmeta = NULL;
		kvecs_dst->u.heap.values[kvecs_dst_id] = kvecs_src->u.heap.values[kvecs_src_id];
	}
	else
	{
		kvecs_dst->cmeta = kvecs_src->cmeta;
		kvecs_dst->u.arrow.slot_id = kvecs_src->u.arrow.slot_id;
		kvecs_dst->u.arrow.rowidx[kvecs_dst_id] = kvecs_src->u.arrow.rowidx[kvecs_src_id];
	}
	return true;
}

STATIC_FUNCTION(int)
__xpu_composite_arrow_write(kern_context *kcxt,
							char *buffer,
							const kern_colmeta *cmeta_dst,
							const xpu_composite_t *arg)
{
	HeapTupleHeaderData *htup = (HeapTupleHeaderData *)buffer;
	const kern_colmeta *cmeta_src = arg->cmeta;
	const kern_colmeta *smeta_src;
	const kern_colmeta *smeta_dst;
	const kern_data_store *kds;
	const kern_varslot_desc *vs_desc;
	uint32_t		nattrs = cmeta_src->num_subattrs;
	uint32_t		row_index = arg->u.arrow.rowidx;
	uint32_t		t_hoff;
	uint16_t		t_infomask = HEAP_HASNULL;
	uint32_t		xdatum_sz = 0;
	xpu_datum_t	   *xdatum = NULL;
	uint8_t		   *nullmap = NULL;

	/* setup destination cmeta */
	kds = (const kern_data_store *)
		((const char *)cmeta_dst - cmeta_dst->kds_offset);
	smeta_dst = &kds->colmeta[cmeta_dst->idx_subattrs];
	/* setup source cmeta/vs_desc */
	kds = (const kern_data_store *)
		((const char *)cmeta_src - cmeta_src->kds_offset);
	smeta_src = &kds->colmeta[cmeta_src->idx_subattrs];
	vs_desc = &kcxt->kvars_desc[arg->u.arrow.slot_id];
	assert(vs_desc->idx_subfield < kcxt->kvars_nrooms);
	vs_desc = &kcxt->kvars_desc[vs_desc->idx_subfield];
	
	t_hoff = MAXALIGN(offsetof(HeapTupleHeaderData, t_bits) +
					  BITMAPLEN(nattrs));
	if (htup)
	{
		memset(htup, 0, t_hoff);
		htup->t_choice.t_datum.datum_typmod = cmeta_src->atttypid;
		htup->t_choice.t_datum.datum_typeid = cmeta_src->atttypmod;
		htup->t_ctid.ip_blkid.bi_hi = 0xffff;	/* InvalidBlockNumber */
		htup->t_ctid.ip_blkid.bi_lo = 0xffff;
		htup->t_ctid.ip_posid = 0;				/* InvalidOffsetNumber */
		htup->t_infomask2 = (nattrs & HEAP_NATTS_MASK);
		htup->t_hoff = t_hoff;
		nullmap = htup->t_bits;
	}

	for (int j=0; j < nattrs; j++, smeta_dst++, smeta_src++, vs_desc++)
	{
		uint32_t	t_next;
		int32_t		nbytes;

		if (vs_desc->vs_ops->xpu_type_sizeof > xdatum_sz)
		{
			xdatum_sz = vs_desc->vs_ops->xpu_type_sizeof + 32;
			xdatum = (xpu_datum_t *)alloca(xdatum_sz);
		}
		if (!__kern_extract_arrow_field(kcxt,
										kds,
										smeta_src,
										row_index,
										vs_desc,
										xdatum))
			return -1;

		if (!XPU_DATUM_ISNULL(xdatum))
		{
			if (nullmap)
				nullmap[j>>3] |= (1<<(j & 7));
			/* alignment */
			t_next = TYPEALIGN(smeta_dst->attalign, t_hoff);
			if (htup)
			{
				if (t_next > t_hoff)
					memset((char *)htup + t_hoff, 0, t_next - t_hoff);
				buffer = (char *)htup + t_next;
			}
			nbytes =  xdatum->expr_ops->xpu_datum_write(kcxt,
														buffer,
														smeta_dst,
														xdatum);
			if (nbytes < 0)
				return -1;
			if (smeta_dst->attlen == -1)
			{
				if (buffer && VARATT_IS_EXTERNAL(buffer))
					t_infomask |= HEAP_HASEXTERNAL;
				t_infomask |= HEAP_HASVARWIDTH;
			}
			t_hoff = t_next + nbytes;
		}
	}

	if (htup)
	{
		htup->t_infomask = t_infomask;
		SET_VARSIZE(&htup->t_choice.t_datum, t_hoff);
	}
	return t_hoff;
}

STATIC_FUNCTION(int)
xpu_composite_datum_write(kern_context *kcxt,
						  char *buffer,
						  const kern_colmeta *cmeta_dst,	/* destination */
						  const xpu_datum_t *__arg)
{
	const xpu_composite_t *arg = (const xpu_composite_t *)__arg;
	int		nbytes;

	if (XPU_DATUM_ISNULL(arg))
		return 0;
	if (!arg->cmeta)
	{
		nbytes = VARSIZE_ANY(arg->u.heap.value);
		if (buffer)
			memcpy(buffer, arg->u.heap.value, nbytes);
		return nbytes;
	}
	return __xpu_composite_arrow_write(kcxt, buffer, cmeta_dst, arg);
}

STATIC_FUNCTION(bool)
xpu_composite_datum_hash(kern_context *kcxt,
						 uint32_t *p_hash,
						 xpu_datum_t *arg)
{
	STROM_ELOG(kcxt, "xpu_composite_datum_hash is not implemented");
	return false;
}
STATIC_FUNCTION(bool)
xpu_composite_datum_comp(kern_context *kcxt,
						 int *p_comp,
						 xpu_datum_t *__a,
						 xpu_datum_t *__b)
{
	STROM_ELOG(kcxt, "xpu_composite_datum_comp is not implemented");
	return false;
}
PGSTROM_SQLTYPE_OPERATORS(composite,false,8,-1);

/* ----------------------------------------------------------------
 *
 * xpu_internal_t type support routine
 *
 * ----------------------------------------------------------------
 */
STATIC_FUNCTION(bool)
xpu_internal_datum_heap_read(kern_context *kcxt,
							 const void *addr,
							 xpu_datum_t *__result)
{
	xpu_internal_t *result = (xpu_internal_t *)__result;

	result->expr_ops = &xpu_internal_ops;
	result->value = addr;
	return true;
}

STATIC_FUNCTION(bool)
xpu_internal_datum_arrow_read(kern_context *kcxt,
							 const kern_data_store *kds,
							  const kern_colmeta *cmeta,
							  uint32_t kds_index,
							  xpu_datum_t *__result)
{
	STROM_ELOG(kcxt, "xpu_internal_t cannot map any Apache Arrow type");
	return false;
}

STATIC_FUNCTION(bool)
xpu_internal_datum_kvec_load(kern_context *kcxt,
							 const kvec_datum_t *__kvecs,
							 uint32_t kvecs_id,
							 xpu_datum_t *__result)
{
	const kvec_internal_t *kvecs = (const kvec_internal_t *)__kvecs;
	xpu_internal_t *result = (xpu_internal_t *)__result;

	result->expr_ops = &xpu_internal_ops;
	result->value = kvecs->values[kvecs_id];
	return true;
}

STATIC_FUNCTION(bool)
xpu_internal_datum_kvec_save(kern_context *kcxt,
							 const xpu_datum_t *__xdatum,
							 kvec_datum_t *__kvecs,
							 uint32_t kvecs_id)
{
	const xpu_internal_t *xdatum = (const xpu_internal_t *)__xdatum;
	kvec_internal_t *kvecs = (kvec_internal_t *)__kvecs;

	kvecs->values[kvecs_id] = xdatum->value;
	return true;
}

STATIC_FUNCTION(bool)
xpu_internal_datum_kvec_copy(kern_context *kcxt,
							 const kvec_datum_t *__kvecs_src,
							 uint32_t kvecs_src_id,
							 kvec_datum_t *__kvecs_dst,
							 uint32_t kvecs_dst_id)
{
    const kvec_internal_t *kvecs_src = (const kvec_internal_t *)__kvecs_src;
    kvec_internal_t *kvecs_dst = (kvec_internal_t *)__kvecs_dst;

	kvecs_dst->values[kvecs_dst_id] = kvecs_src->values[kvecs_src_id];
	return true;
}

STATIC_FUNCTION(int)
xpu_internal_datum_write(kern_context *kcxt,
						 char *buffer,
						 const kern_colmeta *cmeta_dst,	/* destination */
						 const xpu_datum_t *__arg)
{
	const xpu_internal_t *arg = (const xpu_internal_t *)__arg;

	if (cmeta_dst->attlen < 0)
	{
		STROM_ELOG(kcxt, "unable to write out xpu_internal_t with negative attlen");
		return false;
	}
	if (buffer && cmeta_dst->attlen > 0)
		memcpy(buffer, arg->value, cmeta_dst->attlen);
	return cmeta_dst->attlen;
}

STATIC_FUNCTION(bool)
xpu_internal_datum_hash(kern_context *kcxt,
					    uint32_t *p_hash,
						xpu_datum_t *arg)
{
	STROM_ELOG(kcxt, "xpu_internal_datum_hash is not supported");
	return false;
}
STATIC_FUNCTION(bool)
xpu_internal_datum_comp(kern_context *kcxt,
						int *p_comp,
						xpu_datum_t *__a,
						xpu_datum_t *__b)
{
	STROM_ELOG(kcxt, "xpu_internal_datum_comp is not supported");
	return false;
}
PGSTROM_SQLTYPE_OPERATORS(internal,true,8,8);

/*
 * Catalog of built-in device types
 */
/*
 * Built-in SQL type / function catalog
 */
#define TYPE_OPCODE(NAME,a,b)					\
	{ TypeOpCode__##NAME, &xpu_##NAME##_ops },
PUBLIC_DATA(xpu_type_catalog_entry, builtin_xpu_types_catalog[]) = {
#include "xpu_opcodes.h"
	//{ TypeOpCode__composite, &xpu_composite_ops },
	{ TypeOpCode__array, &xpu_array_ops },
	{ TypeOpCode__internal, &xpu_internal_ops },
	{ TypeOpCode__Invalid, NULL }
};

/*
 * Catalog of built-in device functions
 */
#define FUNC_OPCODE(a,b,c,NAME,d,e)			\
	{FuncOpCode__##NAME, pgfn_##NAME},
#define DEVONLY_FUNC_OPCODE(a,NAME,b,c,d)	\
	{FuncOpCode__##NAME, pgfn_##NAME},
PUBLIC_DATA(xpu_function_catalog_entry, builtin_xpu_functions_catalog[]) = {
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
