/*
 * pg_compat.h
 *
 * Macro definitions to keep compatibility of PostgreSQL internal APIs.
 * --
 * Copyright 2011-2018 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2018 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef PG_COMPAT_H
#define PG_COMPAT_H

/*
 * MEMO: tupleDesc was re-defined at PG11. Newer version has flexible-
 * length of FormData_pg_attribute on the tail
 */
#if PG_VERSION_NUM < 110000
#define tupleDescAttr(tdesc,colidx)		((tdesc)->attrs[(colidx)])
#else
#define tupleDescAttr(tdesc,colidx)		(&(tdesc)->attrs[(colidx)])
#endif

/*
 * MEMO: Naming convension of data access macro on some data types
 * were confused before PG11
 */
#if PG_VERSION_NUM < 110000
#define DatumGetRangeTypeP(x)		DatumGetRangeType(x)
#define PG_GETARG_RANGE_P(x)		PG_GETARG_RANGE(x)
#define PG_RETURN_RANGE_P(x)		PG_RETURN_RANGE(x)

#define PG_GETARG_ANY_ARRAY_P(x)	PG_GETARG_ANY_ARRAY(x)
#endif

/*
 * MEMO: PG10 adds the 4th argument for WaitLatch(), to inform the event
 * type which blocks PostgreSQL backend context. In our case, it is always
 * PG_WAIT_EXTENSION.
 */
#if PG_VERSION_NUM < 100000
#define WaitLatch(a,b,c,d)				WaitLatch((a),(b),(c))
#define WaitLatchOrSocket(a,b,c,d,e)	WaitLatchOrSocket((a),(b),(c),(d))
#endif

/*
 * MEMO: PG11 prohibits to replace only tupdesc of TupleTableSlot (it hits
 * Assert condition), so we have to use ExecInitScanTupleSlot() instead of
 * ExecAssignScanType().
 */
#if PG_VERSION_NUM < 110000
#define ExecInitScanTupleSlot(es, ss, tupdesc)							\
	do {																\
		(ss)->ss_ScanTupleSlot = ExecAllocTableSlot(&(es)->es_tupleTable); \
		ExecAssignScanType((ss),(tupdesc));								\
	} while(0)
#endif

/*
 * MEMO: PG10 adds PlanState argument to ExecBuildProjectionInfo
 */
#if PG_VERSION_NUM < 100000
#define ExecBuildProjectionInfo(a,b,c,d,e)\
	ExecBuildProjectionInfo(a,b,c,e)
#endif

/*
 * MEMO: PG11 allows to display unit of numerical values if text-format
 * Just omit 'unit' if PG10 or older
 */
#if PG_VERSION_NUM < 110000
#define ExplainPropertyInteger(qlabel,unit,value,es)		\
	ExplainPropertyLong((qlabel),(value),(es))
#define ExplainPropertyFloat(qlabel,unit,value,ndigits,es)	\
	ExplainPropertyFloat((qlabel),(value),(ndigits),(es))
#endif

/*
 * MEMO: Bugfix at 10.4, 9.6.9 changed internal interface of
 * extract_actual_join_clauses(). Newer version requires 'relids' bitmap
 * of the join-relation.
 *
 * PG96: 0c141fcaa7dd806752986401b25de8f665ceb3fe
 * PG10: 68fab04f7c2a07c5308e3d2957198ccd7a80ebc5
 */
#if ((PG_MAJOR_VERSION == 906 && PG_MINOR_VERSION < 9) ||	\
	 (PG_MAJOR_VERSION == 1000 && PG_MINOR_VERSION < 4))
#define extract_actual_join_clauses(a,b,c,d)	\
	extract_actual_join_clauses((a),(c),(d))
#endif

/*
 * MEMO: PG11 adds PathNameOpenFilePerm and removed creation permission
 * flags from the PathNameOpenFile.
 */
#if PG_VERSION_NUM < 110000
#define PathNameOpenFile(a,b)	PathNameOpenFile((a),(b),0600)
#endif

/*
 * MEMO: PG11 adds 'missing_ok' flag on the get_attname(), however,
 * it deprecates get_relid_attribute_name() that raises an ereport
 * if attname is missing.
 */
#if PG_VERSION_NUM < 110000
#define get_attname(a,b,missing_ok)				\
	((missing_ok) ? get_attname((a),(b)) : get_relid_attribute_name((a),(b)))
#endif

#endif	/* PG_COMPAT_H */
