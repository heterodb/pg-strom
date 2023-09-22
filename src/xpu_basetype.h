/*
 * xpu_basetype.h
 *
 * Collection of base Int/Float functions for both of GPU and DPU
 * --
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
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
#define __pg_simple_nullcomp_eq(a,b)						\
	do {													\
		if (kcxt->kmode_compare_nulls)						\
		{													\
			result->expr_ops = &xpu_bool_ops;				\
			if (XPU_DATUM_ISNULL(a) && XPU_DATUM_ISNULL(b))	\
				result->value = true;						\
			else											\
				result->value = false;						\
		}													\
		else												\
		{													\
			result->expr_ops = NULL;						\
		}													\
	} while(0)

#define __pg_simple_nullcomp_ne(a,b)						\
	do {													\
		if (kcxt->kmode_compare_nulls)						\
		{													\
			result->expr_ops = &xpu_bool_ops;				\
			if (XPU_DATUM_ISNULL(a) && XPU_DATUM_ISNULL(b))	\
				result->value = false;						\
			else											\
				result->value = true;						\
		}													\
		else												\
		{													\
			result->expr_ops = NULL;						\
		}													\
	} while(0)

#define __pg_simple_nullcomp_lt(a,b)	result->expr_ops = NULL
#define __pg_simple_nullcomp_le(a,b)	result->expr_ops = NULL
#define __pg_simple_nullcomp_gt(a,b)	result->expr_ops = NULL
#define __pg_simple_nullcomp_ge(a,b)	result->expr_ops = NULL

#define __PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,OPER,EXTRA)	\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##FNAME##EXTRA(XPU_PGFUNCTION_ARGS)							\
	{																	\
		KEXP_PROCESS_ARGS2(bool,LNAME,lval,RNAME,rval);					\
		if (XPU_DATUM_ISNULL(&lval) || XPU_DATUM_ISNULL(&rval))			\
		{																\
			__pg_simple_nullcomp_##EXTRA(&lval,&rval);					\
		}																\
		else															\
		{																\
			result->expr_ops = kexp->expr_ops;							\
			result->value = ((CAST lval.value) OPER (CAST rval.value));	\
		}																\
		return true;													\
	}

#define PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST)		\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,==,eq)	\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,!=,ne)	\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,< ,lt)	\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,<=,le)	\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,> ,gt)	\
	__PG_SIMPLE_COMPARE_TEMPLATE(FNAME,LNAME,RNAME,CAST,>=,ge)

#define __PG_FLEXIBLE1_COMPARE_TEMPLATE(TNAME,COMP,COND,EXTRA)			\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##TNAME##_##EXTRA(XPU_PGFUNCTION_ARGS)							\
	{																	\
		KEXP_PROCESS_ARGS2(bool,TNAME,lval,TNAME,rval);					\
		if (XPU_DATUM_ISNULL(&lval) || XPU_DATUM_ISNULL(&rval))         \
		{                                                               \
			__pg_simple_nullcomp_##EXTRA(&lval,&rval);                  \
		}                                                               \
		else                                                            \
		{                                                               \
			result->value = (COMP(kcxt, lval.value, rval.value) COND 0); \
			result->expr_ops = kexp->expr_ops;							\
        }                                                               \
        return true;                                                    \
    }
#define PG_FLEXIBLE1_COMPARE_TEMPLATE(TNAME,COMP)		\
	__PG_FLEXIBLE1_COMPARE_TEMPLATE(TNAME,COMP,==,eq)	\
	__PG_FLEXIBLE1_COMPARE_TEMPLATE(TNAME,COMP,!=,ne)	\
	__PG_FLEXIBLE1_COMPARE_TEMPLATE(TNAME,COMP,< ,lt)	\
	__PG_FLEXIBLE1_COMPARE_TEMPLATE(TNAME,COMP,<=,le)	\
	__PG_FLEXIBLE1_COMPARE_TEMPLATE(TNAME,COMP,> ,gt)	\
	__PG_FLEXIBLE1_COMPARE_TEMPLATE(TNAME,COMP,>=,ge)

#define __PG_FLEXIBLE2_COMPARE_TEMPLATE(LNAME,RNAME,COMP,COND,EXTRA)	\
	PUBLIC_FUNCTION(bool)												\
	pgfn_##LNAME##_##EXTRA##_##RNAME(XPU_PGFUNCTION_ARGS)				\
	{																	\
		KEXP_PROCESS_ARGS2(bool,LNAME,lval,RNAME,rval);					\
		if (XPU_DATUM_ISNULL(&lval) || XPU_DATUM_ISNULL(&rval))         \
		{                                                               \
			__pg_simple_nullcomp_##EXTRA(&lval,&rval);                  \
		}                                                               \
		else                                                            \
		{                                                               \
			result->value = (COMP(kcxt, lval.value, rval.value) COND 0); \
			result->expr_ops = kexp->expr_ops;							\
        }                                                               \
        return true;                                                    \
    }

#define PG_FLEXIBLE2_COMPARE_TEMPLATE(LNAME,RNAME,COMP)		\
	__PG_FLEXIBLE2_COMPARE_TEMPLATE(LNAME,RNAME,COMP,==,eq)	\
	__PG_FLEXIBLE2_COMPARE_TEMPLATE(LNAME,RNAME,COMP,!=,ne)	\
	__PG_FLEXIBLE2_COMPARE_TEMPLATE(LNAME,RNAME,COMP,< ,lt)	\
	__PG_FLEXIBLE2_COMPARE_TEMPLATE(LNAME,RNAME,COMP,<=,le)	\
	__PG_FLEXIBLE2_COMPARE_TEMPLATE(LNAME,RNAME,COMP,> ,gt)	\
	__PG_FLEXIBLE2_COMPARE_TEMPLATE(LNAME,RNAME,COMP,>=,ge)

#endif	/* XPU_BASETYPE_H */
