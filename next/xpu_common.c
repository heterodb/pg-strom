/*
 * xpu_common.c
 *
 * Core implementation of xPU device code
 * ----
 * Copyright 2011-2022 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2022 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "xpu_common.h"

#if 0
/*
 * kern_get_datum_xxx
 *
 * Reference to a particular datum on the supplied kernel data store.
 * It returns NULL, if it is a really null-value in context of SQL,
 * or in case when out of range with error code
 *
 * NOTE: We are paranoia for validation of the data being fetched from
 * the kern_data_store in heap-format because we may see a phantom page
 * if the source transaction that required this kernel execution was
 * aborted during execution.
 * Once a transaction gets aborted, shared buffers being pinned are
 * released, even if DMA send request on the buffers are already
 * enqueued. In this case, the calculation result shall be discarded,
 * so no need to worry about correctness of the calculation, however,
 * needs to be care about address of the variables being referenced.
 */
STATIC_FUNCTION(void *)
kern_get_datum_tuple(kern_colmeta *colmeta,
					 kern_tupitem *tupitem,
					 uint32_t colidx)
{
	HeapTupleHeaderData *htup = &tupitem->htup;
	uint32_t	offset = htup->t_hoff;
	uint32_t	j, ncols = (htup->t_infomask2 & HEAP_NATTS_MASK);
	bool		heap_hasnull = ((htup->t_infomask & HEAP_HASNULL) != 0);

	/* shortcut if colidx is obviously out of range */
	if (colidx >= ncols)
		return NULL;
	/* shortcut if tuple contains no NULL values */
	if (!heap_hasnull)
	{
		kern_colmeta   *cmeta = &colmeta[colidx];

		if (cmeta->attcacheoff >= 0)
			return (char *)htup + cmeta->attcacheoff;
	}
	/* regular path that walks on heap-tuple from the head */
	for (j=0; j < ncols; j++)
	{
		if (heap_hasnull && att_isnull(j, htup->t_bits))
		{
			if (j == colidx)
				return NULL;
		}
		else
		{
			kern_colmeta   *cmeta = &colmeta[j];
			char		   *addr;

			if (cmeta->attlen > 0)
				offset = TYPEALIGN(cmeta->attalign, offset);
			else if (!VARATT_NOT_PAD_BYTE((char *)htup + offset))
				offset = TYPEALIGN(cmeta->attalign, offset);

			/* TODO: overrun checks here */
			addr = ((char *) htup + offset);
			if (j == colidx)
				return addr;
			if (cmeta->attlen > 0)
				offset += cmeta->attlen;
			else
				offset += VARSIZE_ANY(addr);
		}
	}
	return NULL;
}

STATIC_FUNCTION(void *)
kern_get_datum_column(kern_data_store *kds,
					  kern_data_extra *extra,
					  uint32_t colidx, uint32_t rowidx)
{
	kern_colmeta   *cmeta = &kds->colmeta[colidx];
	char		   *addr;

	if (rowidx >= kds->nitems)
		return NULL;	/* out of range */
	if (cmeta->nullmap_offset != 0)
	{
		uint32_t   *nullmap = (uint32_t *)
			((char *)kds + __kds_unpack(cmeta->nullmap_offset));
		if ((nullmap[rowidx>>5] & (1U << (rowidx & 0x1f))) == 0)
			return NULL;
	}

	addr = (char *)kds + __kds_unpack(cmeta->values_offset);
	if (cmeta->attlen > 0)
	{
		addr += TYPEALIGN(cmeta->attalign,
						  cmeta->attlen) * rowidx;
		return addr;
	}
	else if (cmeta->attlen == -1)
	{
		assert(extra != NULL);
		return (char *)extra + __kds_unpack(((uint32_t *)addr)[rowidx]);
	}
	return NULL;
}
#endif

/*
 * Const Expression
 */
PUBLIC_FUNCTION(bool)
pgfn_ExecExpression(XPU_PGFUNCTION_ARGS)
{

	
	
	return false;
}

STATIC_FUNCTION(bool)
pgfn_ConstExpr(XPU_PGFUNCTION_ARGS)
{
	const void *addr;

	if (kexp->u.c.const_isnull)
		addr = NULL;
	else
		addr = kexp->u.c.const_value;
	return kexp->exptype_ops->xpu_datum_ref(kcxt, __result, NULL, addr, -1);
}

STATIC_FUNCTION(bool)
pgfn_ParamExpr(XPU_PGFUNCTION_ARGS)
{
	kern_session_info *session = kcxt->session;
	uint32_t	param_id = kexp->u.p.param_id;
	void	   *addr = NULL;

	if (param_id < session->nparams && session->poffset[param_id] != 0)
		addr = (char *)session + session->poffset[param_id];
	return kexp->exptype_ops->xpu_datum_ref(kcxt, __result, NULL, addr, -1);
}

STATIC_FUNCTION(bool)
pgfn_VarExpr(XPU_PGFUNCTION_ARGS)
{
	uint32_t	slot_id = kexp->u.v.var_slot_id;
	const kern_colmeta *cmeta;
	const void *addr;
	int			len;

	if (slot_id < kcxt->kvars_nslots)
	{
		cmeta = kcxt->kvars_cmeta[slot_id];
		addr = kcxt->kvars_addr[slot_id];
		len = kcxt->kvars_len[slot_id];
	}
	else
	{
		cmeta = NULL;
		addr = NULL;
		len = -1;
	}
	return kexp->exptype_ops->xpu_datum_ref(kcxt, __result, cmeta, addr, len);
}

STATIC_FUNCTION(bool)
pgfn_BoolExprAnd(XPU_PGFUNCTION_ARGS)
{
	xpu_bool_t *result = (xpu_bool_t *)__result;
	int			i;
	bool		anynull = false;
	const kern_expression *arg = KEXP_FIRST_ARG(-1,bool);

	result->ops = &xpu_bool_ops;
	for (i=0; i < kexp->nargs; i++)
	{
		xpu_bool_t	status;

		if (!EXEC_KERN_EXPRESSION(kcxt, arg, &status))
			return false;
		if (status.isnull)
			anynull = true;
		else if (!status.value)
		{
			result->value = false;
			return true;
		}
		arg = KEXP_NEXT_ARG(arg, bool);
	}
	result->isnull = anynull;
	result->value  = true;
	return true;
}

STATIC_FUNCTION(bool)
pgfn_BoolExprOr(XPU_PGFUNCTION_ARGS)
{
	xpu_bool_t *result = (xpu_bool_t *)__result;
	int			i;
	bool		anynull = false;
	const kern_expression *arg = KEXP_FIRST_ARG(-1,bool);

	result->ops = &xpu_bool_ops;
	for (i=0; i < kexp->nargs; i++)
	{
		xpu_bool_t	status;

		if (!EXEC_KERN_EXPRESSION(kcxt, arg, &status))
			return false;
		if (status.isnull)
			anynull = true;
		else if (status.value)
		{
			result->value = true;
			return true;
		}
		arg = KEXP_NEXT_ARG(arg, bool);
	}
	result->isnull = anynull;
	result->value  = false;
	return true;
}

STATIC_FUNCTION(bool)
pgfn_BoolExprNot(XPU_PGFUNCTION_ARGS)
{
	xpu_bool_t *result = (xpu_bool_t *)__result;
	xpu_bool_t	status;
	const kern_expression *arg = KEXP_FIRST_ARG(1,bool);

	if (!EXEC_KERN_EXPRESSION(kcxt, arg, &status))
		return false;
	result->ops = &xpu_bool_ops;
	if (status.isnull)
		result->isnull = true;
	else
	{
		result->isnull = false;
		result->value = !result->value;
	}
	return true;
}

STATIC_FUNCTION(bool)
pgfn_NullTestExpr(XPU_PGFUNCTION_ARGS)
{
	xpu_bool_t	   *result = (xpu_bool_t *)__result;
	xpu_datum_t	   *status;
	const kern_expression *arg = KEXP_FIRST_ARG(1,Invalid);

	status = (xpu_datum_t *)alloca(arg->exptype_ops->xpu_type_sizeof);
	if (!EXEC_KERN_EXPRESSION(kcxt, arg, status))
		return false;
	result->ops = &xpu_bool_ops;
	result->isnull = false;
	switch (kexp->opcode)
	{
		case FuncOpCode__NullTestExpr_IsNull:
			result->value = status->isnull;
			break;
		case FuncOpCode__NullTestExpr_IsNotNull:
			result->value = !status->isnull;
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
	xpu_bool_t	   *result = (xpu_bool_t *)__result;
	xpu_bool_t		status;
	const kern_expression *arg = KEXP_FIRST_ARG(1,bool);

	if (!EXEC_KERN_EXPRESSION(kcxt, arg, &status))
		return false;
	result->ops = &xpu_bool_ops;
	result->isnull = false;
	switch (kexp->opcode)
	{
		case FuncOpCode__BoolTestExpr_IsTrue:
			result->value = (!status.isnull && status.value);
			break;
		case FuncOpCode__BoolTestExpr_IsNotTrue:
			result->value = (status.isnull || !status.value);
			break;
		case FuncOpCode__BoolTestExpr_IsFalse:
			result->value = (!status.isnull && !status.value);
			break;
		case FuncOpCode__BoolTestExpr_IsNotFalse:
			result->value = (status.isnull || status.value);
			break;
		case FuncOpCode__BoolTestExpr_IsUnknown:
			result->value = status.isnull;
			break;
		case FuncOpCode__BoolTestExpr_IsNotUnknown:
			result->value = !status.isnull;
			break;
		default:
			STROM_ELOG(kcxt, "corrupted kernel expression");
			return false;
	}
	return true;
}

/*
 * Projection
 */
STATIC_FUNCTION(bool)
pgfn_Projection(XPU_PGFUNCTION_ARGS)
{
	STROM_ELOG(kcxt, "Projection is not implemented");
	return false;
}

/* ----------------------------------------------------------------
 *
 * LoadVars / Projection
 *
 * ----------------------------------------------------------------
 */
STATIC_FUNCTION(int)
kern_extract_heap_tuple(kern_context *kcxt,
						kern_data_store *kds,
						kern_tupitem *tupitem,
						int curr_depth,
						const kern_preload_vars_item *kvars_items,
						int kvars_nloads)
{
	const kern_preload_vars_item *kvars = kvars_items;
	HeapTupleHeaderData *htup = &tupitem->htup;
	uint32_t	offset = htup->t_hoff;
	int			kvars_nloads_saved = kvars_nloads;
	int			resno = 1;
	int			slot_id;
	int			ncols = (htup->t_infomask2 & HEAP_NATTS_MASK);
	bool		heap_hasnull = ((htup->t_infomask & HEAP_HASNULL) != 0);

	/* shortcut if no columns in this depth */
	if (kvars->var_depth != curr_depth)
		return 0;

	if (ncols > kds->ncols)
		ncols = kds->ncols;
	/* try attcacheoff shortcut, if available. */
	if (!heap_hasnull)
	{
		while (kvars_nloads > 0 &&
			   kvars->var_depth == curr_depth &&
			   kvars->var_resno <= ncols)
		{
			kern_colmeta   *cmeta = &kds->colmeta[kvars->var_resno - 1];

			if (cmeta->attcacheoff < 0)
				break;
			slot_id = kvars->var_slot_id;
			resno   = kvars->var_resno;
			offset  = htup->t_hoff + cmeta->attcacheoff;
			assert(slot_id < kcxt->kvars_nslots);

			kcxt->kvars_cmeta[slot_id] = cmeta;
			kcxt->kvars_addr[slot_id]  = (char *)htup + offset;
			kcxt->kvars_len[slot_id]   = -1;

			kvars++;
			kvars_nloads--;
		}
	}

	/* move to the slow heap-tuple extract */
	while (kvars_nloads > 0 &&
		   kvars->var_depth == curr_depth &&
		   kvars->var_resno >= resno &&
		   kvars->var_resno <= ncols)
	{
		while (resno <= ncols)
		{
			kern_colmeta   *cmeta = &kds->colmeta[resno-1];
			char		   *addr;

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

			if (kvars->var_resno == resno)
			{
				slot_id = kvars->var_slot_id;
				assert(slot_id < kcxt->kvars_nslots);

				kcxt->kvars_cmeta[slot_id] = cmeta;
				kcxt->kvars_addr[slot_id]  = addr;
				kcxt->kvars_len[slot_id]   = -1;

				kvars++;
				kvars_nloads--;
				break;
			}
		}
	}

	/* other fields, which refers out of ranges, are NULL */
	while (kvars_nloads > 0 &&
		   kvars->var_depth == curr_depth)
	{
		kern_colmeta *cmeta = &kds->colmeta[kvars->var_resno-1];
		int		slot_id = kvars->var_slot_id;

		assert(slot_id < kcxt->kvars_nslots);
		kcxt->kvars_cmeta[slot_id] = cmeta;
		kcxt->kvars_addr[slot_id]  = NULL;
		kcxt->kvars_len[slot_id]   = -1;

		kvars++;
		kvars_nloads--;
	}
	return (kvars_nloads_saved - kvars_nloads);
}

STATIC_FUNCTION(bool)
pgfn_LoadVars(XPU_PGFUNCTION_ARGS)
{
	STROM_ELOG(kcxt, "Bug? LoadVars shall not be called as a part of expression");
	return false;
}

PUBLIC_FUNCTION(bool)
ExecLoadVarsOuterRow(XPU_PGFUNCTION_ARGS,
					 kern_data_store *kds_outer,
					 kern_tupitem *tupitem_outer,
					 int num_inners,
					 kern_data_store **kds_inners,
					 kern_tupitem **tupitem_inners)
{
	const kern_preload_vars *preload;
	const kern_expression *karg;
	int			index;
	int			depth;

	assert(kexp->opcode == FuncOpCode__LoadVars &&
		   kexp->nargs == 1);
	karg = (const kern_expression *)kexp->u.data;
	assert(kexp->exptype == karg->exptype);
	preload = (const kern_preload_vars *)((const char *)karg +
										  MAXALIGN(VARSIZE(karg)));
	/*
	 * Walking on the outer/inner tuples
	 */
	for (depth=0, index=0;
		 depth <= num_inners && index < preload->nloads;
		 depth++)
	{
		kern_data_store *kds;
		kern_tupitem *tupitem;

		if (depth == 0)
		{
			kds     = kds_outer;
			tupitem = tupitem_outer;
		}
		else
		{
			kds     = kds_inners[depth - 1];
			tupitem = tupitem_inners[depth - 1];
		}
		index += kern_extract_heap_tuple(kcxt,
										 kds,
										 tupitem,
										 depth,
										 preload->kvars + index,
										 preload->nloads - index);
	}
	/*
	 * Call the argument
	 */
	return EXEC_KERN_EXPRESSION(kcxt, karg, __result);
}

/*
 * __form_kern_heaptuple
 */
STATIC_FUNCTION(uint32_t)
__form_kern_heaptuple(kern_context    *kcxt,
					  kern_expression *kproj,
					  kern_data_store *kds,
					  HeapTupleHeaderData *htup)
{
	kern_projection_map *proj_map;
	uint32_t   *proj_slot;
	bool		t_hasnull = false;
	uint16_t	t_infomask = 0;
	uint32_t	t_hoff;
	uint32_t	sz;
	int			j, ncols = 0;

	/*
	 * Fetch 'kern_projection_map' from the tail of 'kern_expression'
	 */
	assert(kproj->opcode == FuncOpCode__Projection);
	sz = *((uint32_t *)((char *)kproj + VARSIZE(kproj) - sizeof(uint32_t)));
	proj_map = (kern_projection_map *)((char *)kproj + VARSIZE(kproj) - sz);
	proj_slot = proj_map->slot_id + proj_map->nexprs;

	/* has any NULL attributes? */
	for (j = proj_map->nattrs; j > 0; j--)
	{
		uint32_t	slot_id = proj_slot[j-1];

		assert(slot_id < kcxt->kvars_nslots);
		if (kcxt->kvars_addr[slot_id])
		{
			if (ncols == 0)
				ncols = j;
		}
		else if (ncols > 0)
		{
			t_infomask |= HEAP_HASNULL;
			t_hasnull = true;
			break;
		}
	}

	/* set up headers */
	t_hoff = offsetof(HeapTupleHeaderData, t_bits);
	if (t_hasnull)
		t_hoff = BITMAPLEN(ncols);
	t_hoff = MAXALIGN(t_hoff);

	memset(htup, 0, t_hoff);
	htup->t_choice.t_datum.datum_typmod = kds->tdtypmod;
	htup->t_choice.t_datum.datum_typeid = kds->tdtypeid;
	htup->t_ctid.ip_blkid.bi_hi = 0xffff;	/* InvalidBlockNumber */
	htup->t_ctid.ip_blkid.bi_lo = 0xffff;
	htup->t_ctid.ip_posid = 0;				/* InvalidOffsetNumber */
	htup->t_infomask2 = (ncols & HEAP_NATTS_MASK);
	htup->t_hoff = t_hoff;

	/* walk on the columns */
	for (j=0; j < ncols; j++)
	{
		uint32_t		slot_id = proj_slot[j];
		kern_colmeta   *cmeta = kcxt->kvars_cmeta[slot_id];
		void		   *vaddr = kcxt->kvars_addr[slot_id];
		int				vlen  = kcxt->kvars_len[slot_id];

		assert(slot_id < kcxt->kvars_nslots);
		if (!vaddr)
		{
			assert(t_hasnull);
			continue;
		}
		if (t_hasnull)
			htup->t_bits[j>>3] |= (1<<(j & 7));
		//put datum


		



	}
	htup->t_infomask = t_infomask;
	SET_VARSIZE(&htup->t_choice.t_datum, t_hoff);

	return t_hoff;
}

/*
 * __execProjectionCommon
 */
STATIC_FUNCTION(bool)
__execProjectionCommon(kern_context *kcxt,
					   kern_expression *kexp,
					   kern_data_store *kds_dst,
					   int nvalids,
					   uint32_t tupsz)
{
	size_t		offset;
	uint32_t	total_sz;
	union {
		struct {
			uint32_t	nitems;
			uint32_t	usage;
		} i;
		uint64_t		v64;
	} oldval, curval, newval;

	/* allocation of the destination buffer */
	assert(kds_dst->format == KDS_FORMAT_ROW);
	offset = __reduce_stair_add_sync(tupsz, &total_sz);
	if (LaneId() == 0)
	{
		curval.i.nitems = kds_dst->nitems;
		curval.i.usage  = kds_dst->usage;
		do {
			newval = oldval = curval;
			newval.i.nitems += nvalids;
			newval.i.usage  += __kds_packed(total_sz);

			if (KDS_HEAD_LENGTH(kds_dst) +
				MAXALIGN(sizeof(uint32_t) * newval.i.nitems) +
				__kds_unpack(newval.i.usage) > kds_dst->length)
			{
				STROM_EREPORT(kcxt, ERRCODE_STROM_DATASTORE_NOSPACE,
							  "No space left on the destination buffer");
				break;
			}
		} while ((curval.v64 = atomicCAS((uint64_t *)&kds_dst->nitems,
										 oldval.v64,
										 newval.v64)) != oldval.v64);
	}
	oldval.v64 = __shfl_sync(__activemask(), oldval.v64, 0);
	/* error checks */
	if (__any_sync(__activemask(), kcxt->errcode != ERRCODE_STROM_SUCCESS))
		return false;

	/* write out the tuple */
	if (LaneId() < nvalids)
	{
		kern_expression *kproj = (kern_expression *)kexp->u.data;
		uint32_t	row_id = oldval.i.nitems + LaneId();
		kern_tupitem *tupitem;

		offset += __kds_unpack(oldval.i.usage);
		KDS_GET_ROWINDEX(kds_dst)[row_id] = __kds_packed(offset);
		tupitem = (kern_tupitem *)
			((char *)kds_dst + kds_dst->length - offset);
		tupitem->rowid = row_id;
		tupitem->t_len = __form_kern_heaptuple(kcxt, kproj,
											   kds_dst,
											   &tupitem->htup);
	}
}

/*
 * ExecKernProjection
 */
PUBLIC_FUNCTION(bool)
ExecProjectionOuterRow(kern_context *kcxt,
					   kern_expression *kexp,	/* LoadVars + Projection */
					   kern_data_store *kds_dst,
					   kern_data_store *kds_outer,
					   kern_tupitem *tupitem_outer,
					   int num_inners,
					   kern_data_store **kds_inners,
					   kern_tupitem **tupitem_inners)
{
	uint32_t	nvalids;
	uint32_t	mask;
	uint32_t	tupsz = 0;

	assert(__activemask() == 0xffffffffU);
	mask = __ballot_sync(__activemask(), tupitem_outer != NULL);
	nvalids = __popc(mask);
	assert(tupitem_outer != NULL
		   ? LaneId() <  nvalids
		   : LaneId() >= nvalids);
	/*
	 * First, extract the variables from outer/inner tuples, and
	 * calculate expressions, if any.
	 */
	assert(kexp->opcode == FuncOpCode__LoadVars &&
		   kexp->exptype == TypeOpCode__int4);
	if (LaneId() < nvalids)
	{
		xpu_int4_t	__tupsz;

		if (ExecLoadVarsOuterRow(kcxt,
								 kexp,
								 (xpu_datum_t *)&tupsz,
								 kds_outer,
								 tupitem_outer,
								 num_inners,
								 kds_inners,
								 tupitem_inners))
		{
			if (!__tupsz.isnull)
			{
				tupsz = MAXALIGN(__tupsz.value);
				assert(tupsz > 0);
			}
		}
	}
	return __execProjectionCommon(kcxt, kexp, kds_dst, unitsz);
}

/*
 * Catalog of built-in device types
 */
/*
 * Built-in SQL type / function catalog
 */
#define TYPE_OPCODE(NAME,a,b)							\
	{ TypeOpCode__##NAME, &xpu_##NAME##_ops },
PUBLIC_DATA xpu_type_catalog_entry builtin_xpu_types_catalog[] = {
#include "xpu_opcodes.h"
	{ TypeOpCode__unsupported, &xpu_unsupported_ops },
	{ TypeOpCode__Invalid, NULL }
};

/*
 * Catalog of built-in device functions
 */
#define FUNC_OPCODE(a,b,c,NAME,d,e)				\
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
#include "xpu_opcodes.h"
	{FuncOpCode__Projection,                pgfn_Projection},
	{FuncOpCode__LoadVars,                  pgfn_LoadVars},
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
