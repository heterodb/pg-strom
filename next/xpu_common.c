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
 * Const Expression
 */
PUBLIC_FUNCTION(bool)
pgfn_ExecExpression(XPU_PGFUNCTION_ARGS)
{

	
	
	
}

STATIC_FUNCTION(bool)
pgfn_ConstExpr(XPU_PGFUNCTION_ARGS)
{
	kern_const_expression *con = (kern_const_expression *)expr;
	void	   *addr;

	addr = (con->const_isnull ? NULL : con->const_datum);
	return con->rettype_ops->sql_datum_ref(kcxt, __result, addr);
}

STATIC_FUNCTION(bool)
pgfn_ParamExpr(XPU_PGFUNCTION_ARGS)
{
	kern_param_expression *prm = (kern_param_expression *)expr;
	void	   *addr;

	addr = kparam_get_value(kcxt->kparams, prm->param_id);
	return prm->rettype_ops->sql_datum_ref(kcxt, __result, addr);
}

STATIC_FUNCTION(bool)
pgfn_VarExpr(XPU_PGFUNCTION_ARGS)
{
	return true;
}

STATIC_FUNCTION(bool)
pgfn_BoolExprAnd(XPU_PGFUNCTION_ARGS)
{
	sql_bool_t *result = (sql_bool_t *)__result;
	int			i, off = 0;
	bool		anynull = false;

	memset(result, 0, sizeof(sql_bool_t));
	result->ops = &sql_bool_ops;
	for (i=0; i < expr->nargs; i++)
	{
		const kern_expression *arg;
		sql_bool_t	status;

		arg = (const kern_expression *)(expr->data + off);
		EXPR_OVERRUN_CHECKS(arg);
		if (arg->rettype != TypeOpCode__bool)
		{
			STROM_ELOG(kcxt, "corrupted kernel expression");
			return false;
		}
		if (!EXEC_KERN_EXPRESSION(kcxt, arg, &status))
			return false;
		if (status.isnull)
			anynull = true;
		else if (!status.value)
		{
			result->value = false;
			return true;
		}
		off = MAXALIGN(off + VARSIZE(arg));
	}
	result->isnull = anynull;
	result->value  = true;
	return true;
}

STATIC_FUNCTION(bool)
pgfn_BoolExprOr(XPU_PGFUNCTION_ARGS)
{
	sql_bool_t *result = (sql_bool_t *)__result;
	int			i, off = 0;
	bool		anynull = false;

	memset(result, 0, sizeof(sql_bool_t));
	result->ops = &sql_bool_ops;
	for (i=0; i < expr->nargs; i++)
	{
		const kern_expression *arg;
		sql_bool_t	status;

		arg = (const kern_expression *)(expr->data + off);
		EXPR_OVERRUN_CHECKS(arg);
		if (arg->rettype != TypeOpCode__bool)
		{
			STROM_ELOG(kcxt, "corrupted kernel expression");
			return false;
		}
		if (!EXEC_KERN_EXPRESSION(kcxt, arg, &status))
			return false;
		if (status.isnull)
			anynull = true;
		else if (status.value)
		{
			result->value = true;
			return true;
		}
		off = MAXALIGN(off + VARSIZE(arg));
	}
	result->isnull = anynull;
	result->value  = false;
	return true;
}

STATIC_FUNCTION(bool)
pgfn_BoolExprNot(XPU_PGFUNCTION_ARGS)
{
	sql_bool_t *result = (sql_bool_t *)__result;
	sql_bool_t	status;
	const kern_expression *arg = (const kern_expression *)expr->data;

	memset(result, 0, sizeof(sql_bool_t));
	result->ops = &sql_bool_ops;
	assert(expr->nargs == 1);

	EXPR_OVERRUN_CHECKS(arg);
	if (arg->rettype != TypeOpCode__bool)
	{
		STROM_ELOG(kcxt, "corrupted kernel expression");
		return false;
	}
	if (!EXEC_KERN_EXPRESSION(kcxt, arg, &status))
		return false;
	if (status.isnull)
		result->isnull = true;
	else
		result->value = !result->value;
	return true;
}

STATIC_FUNCTION(bool)
pgfn_NullTestExpr(XPU_PGFUNCTION_ARGS)
{
	sql_bool_t	   *result = (sql_bool_t *)__result;
	sql_datum_t	   *status;
	const kern_expression *arg = (const kern_expression *)expr->data;

	assert(expr->nargs == 1);
	EXPR_OVERRUN_CHECKS(arg);
	status = (sql_datum_t *)alloca(arg->rettype_ops->sql_type_sizeof);
	if (!EXEC_KERN_EXPRESSION(kcxt, arg, status))
		return false;
	memset(result, 0, sizeof(sql_bool_t));
	result->ops = &sql_bool_ops;
	switch (expr->opcode)
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
	sql_bool_t	   *result = (sql_bool_t *)__result;
	sql_bool_t		status;
	const kern_expression *arg = (const kern_expression *)expr->data;

	assert(expr->nargs == 1);
	EXPR_OVERRUN_CHECKS(arg);
	if (!EXEC_KERN_EXPRESSION(kcxt, arg, &status))
		return false;
	memset(result, 0, sizeof(sql_bool_t));
	result->ops = &sql_bool_ops;
	switch (expr->opcode)
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
PUBLIC_DATA sql_type_catalog_entry builtin_sql_types_catalog[] = {
	{TypeOpCode__int1, &sql_int1_ops},
//#include "xpu_opcodes.h"
	{TypeOpCode__Invalid, NULL},
};

/*
 * Catalog of built-in device functions
 */
PUBLIC_DATA sql_function_catalog_entry builtin_sql_functions_catalog[] = {
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
	{FuncOpCode__Invalid, NULL},
//#include "xpu_opcodes.h"

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
