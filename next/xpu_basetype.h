/*
 * xpu_basetype.h
 *
 * Collection of base Int/Float functions for xPU (GPU/DPU/SPU)
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef XPU_BASETYPE_H
#define XPU_BASETYPE_H


#ifndef PG_BOOLOID
#define PG_BOOLOID		16
#endif	/* PG_BOOLOID */
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(bool, int8_t);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(int1, int8_t);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(int2, int16_t);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(int4, int32_t);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(int8, int64_t);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(float2, float2_t);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(float4, float4_t);
PGSTROM_SQLTYPE_SIMPLE_DECLARATION(float8, float8_t);

/*
 * Template for simple comparison
 */
#define __PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,OPER,EXTRA)	\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##FNAME##EXTRA(XPU_PGFUNCTION_ARGS)							\
	{																	\
		xpu_bool_t *result = (xpu_bool_t *)__result;					\
		xpu_##LNAME##_t lval;											\
		xpu_##RNAME##_t rval;											\
		const kern_expression *arg = KEXP_FIRST_ARG(2,LNAME);			\
																		\
		if (!EXEC_KERN_EXPRESSION(kcxt, arg, &lval))					\
			return false;												\
		arg = KEXP_NEXT_ARG(arg, RNAME);								\
		if (!EXEC_KERN_EXPRESSION(kcxt, arg, &rval))					\
			return false;												\
		result->ops = &xpu_bool_ops;									\
		result->isnull = (lval.isnull | rval.isnull);					\
		if (!result->isnull)											\
			result->value = ((CAST)lval.value OPER (CAST)rval.value);	\
		return true;													\
	}

#define PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST)		\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,==,eq)	\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,!=,ne)	\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,<,lt)	\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,<=,le)	\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,>,gt)	\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,>=,ge)

#endif	/* XPU_BASETYPE_H */
