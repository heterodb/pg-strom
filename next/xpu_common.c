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
	return kexp->rettype_ops->xpu_datum_ref(kcxt, __result, NULL, addr, -1);
}

STATIC_FUNCTION(bool)
pgfn_ParamExpr(XPU_PGFUNCTION_ARGS)
{
	kern_session_info *session = kcxt->session;
	uint32_t	param_id = kexp->u.p.param_id;
	void	   *addr = NULL;

	if (param_id < session->nparams && session->poffset[param_id] != 0)
		addr = (char *)session + session->poffset[param_id];
	return kexp->rettype_ops->xpu_datum_ref(kcxt, __result, NULL, addr, -1);
}

STATIC_FUNCTION(bool)
pgfn_VarExpr(XPU_PGFUNCTION_ARGS)
{
	uint32_t	slot_id = kexp->u.v.var_slot_id;
	const kern_colmeta *cmeta;
	const void *addr = NULL;
	int			len = -1;

	if (slot_id < kcxt->kvars_num)
	{
		cmeta = kcxt->kvars_cmeta[slot_id];
		addr = kcxt->kvars_addr[slot_id];
		len = kcxt->kvars_len[slot_id];
	}
	return kexp->rettype_ops->xpu_datum_ref(kcxt, __result, cmeta, addr, len);
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

	status = (xpu_datum_t *)alloca(arg->rettype_ops->xpu_type_sizeof);
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
 * Catalog of built-in device types
 */
/*
 * Built-in SQL type / function catalog
 */
#define TYPE_OPCODE(NAME,a,b)							\
	{ TypeOpCode__##NAME, &xpu_##NAME##_ops },
PUBLIC_DATA xpu_type_catalog_entry builtin_xpu_types_catalog[] = {
#include "xpu_opcodes.h"
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
