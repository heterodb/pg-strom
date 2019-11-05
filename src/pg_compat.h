/*
 * pg_compat.h
 *
 * Macro definitions to keep compatibility of PostgreSQL internal APIs.
 * --
 * Copyright 2011-2019 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2019 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#ifndef PG_COMPAT_H
#define PG_COMPAT_H

/*
 * MEMO: PostgreSQL v11 declares all the type OIDs, but some of types
 * has no label definition in the older version.
 */
#if PG_VERSION_NUM < 110000
//#define INT4RANGEOID	3904
#define INT8RANGEOID	3926
#define NUMRANGEOID		3906
#define TSRANGEOID		3908
#define TSTZRANGEOID	3910
#define DATERANGEOID	3912
#endif

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
#define DatumGetRangeTypePCopy(x)	DatumGetRangeTypeCopy(x)
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
 * MEMO: PG10 add this function (also look at misc.c), then PG11 adds
 * argument of 'max_workers' argument.
 */
#if PG_VERSION_NUM < 100000
#define compute_parallel_worker(a,b,c,d)		\
	__compute_parallel_worker((a),(b),(c))
#elif PG_VERSION_NUM < 110000
#define compute_parallel_worker(a,b,c,d)		\
	compute_parallel_worker((a),(b),(c))
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

/*
 * MEMO: PG11.2, 10.7, and 9.6.12 changed ABI of GenerateTypeDependencies()
 *
 * PG11: 1b55acb2cf48341822261bf9c36785be5ee275db
 * PG10: 2d83863ea2739dc559ed490c284f5c1817db4752
 * PG96: d431dff1af8c220490b84dd978aa3a508f71d415
 */
#if ((PG_MAJOR_VERSION ==  906 && PG_MINOR_VERSION < 12) || \
	 (PG_MAJOR_VERSION == 1000 && PG_MINOR_VERSION < 7)  ||	\
	 (PG_MAJOR_VERSION == 1100 && PG_MINOR_VERSION < 2))
#define GenerateTypeDependencies(a,b,c,d,e,f,g,h)						\
	GenerateTypeDependencies((b)->typnamespace,	/* typeNamespace */		\
							 (a),				/* typeObjectId */		\
							 (b)->typrelid,		/* relationOid */		\
							 (e),				/* relationKind */		\
							 (b)->typowner,		/* owner */				\
							 (b)->typinput,		/* inputProcedure  */	\
							 (b)->typoutput,	/* outputProcedure */	\
							 (b)->typreceive,	/* receiveProcedure */	\
							 (b)->typsend,		/* sendProcedure */		\
							 (b)->typmodin,		/* typmodinProcedure */	\
							 (b)->typmodout,	/* typmodoutProcedure */ \
							 (b)->typanalyze,	/* analyzeProcedure */	\
							 (b)->typelem,		/* elementType */		\
							 (f),				/* isImplicitArray */	\
							 (b)->typbasetype,	/* baseType */			\
							 (b)->typcollation,	/* typeCollation */		\
							 (c),				/* defaultExpr */		\
							 (h))				/* rebuild */
#endif

/*
 * MEMO: PG9.6 does not define macros below
 */
#if PG_MAJOR_VERSION < 1000
/* convenience macros for accessing a JsonbContainer struct */
#define JsonContainerSize(jc)       ((jc)->header & JB_CMASK)
#define JsonContainerIsScalar(jc)   (((jc)->header & JB_FSCALAR) != 0)
#define JsonContainerIsObject(jc)   (((jc)->header & JB_FOBJECT) != 0)
#define JsonContainerIsArray(jc)    (((jc)->header & JB_FARRAY) != 0)

#define IS_SIMPLE_REL(rel)							\
	((rel)->reloptkind == RELOPT_BASEREL ||			\
	 (rel)->reloptkind == RELOPT_OTHER_MEMBER_REL)

static inline Oid
CatalogTupleInsert(Relation heapRel, HeapTuple tup)
{
	CatalogIndexState indstate;
	Oid         oid;

	indstate = CatalogOpenIndexes(heapRel);

	oid = simple_heap_insert(heapRel, tup);

	CatalogIndexInsert(indstate, tup);
	CatalogCloseIndexes(indstate);

	return oid;
}
#endif

/*
 * MEMO: PG11.3 and PG10.8 added is_dummy_rel() instead of old IS_DUMMY_REL().
 *
 * PG11: 925f46ffb82f0b25c94e7997caff732eaf14367d
 * PG10: 19ff710aaa5f131a15da97484da5a669a3448864
 */
#if ((PG_MAJOR_VERSION == 1100 && PG_MINOR_VERSION < 3) || \
	 (PG_MAJOR_VERSION == 1000 && PG_MINOR_VERSION < 8) || \
	 (PG_MAJOR_VERSION <  1000))
#define is_dummy_rel(r)			IS_DUMMY_REL(r)
#endif

/*
 * MEMO: PG12 adopted storage access method (a.k.a pluggable storage layer).
 */
#if PG_VERSION_NUM < 120000
typedef ParallelHeapScanDesc			ParallelTableScanDesc;
typedef ParallelHeapScanDescData		ParallelTableScanDescData;

#define table_open(a,b)					heap_open(a,b)
#define table_close(a,b)				heap_close(a,b)
#define table_beginscan(a,b,c,d)		heap_beginscan(a,b,c,d)
#define table_beginscan_parallel(a,b)	heap_beginscan_parallel(a,b)

#define table_endscan(a)				heap_endscan(a)
#define table_rescan(a,b)				heap_rescan(a,b)

#define table_parallelscan_estimate(a,b)		\
	heap_parallelscan_estimate(b)
#define table_parallelscan_initialize(a,b,c)	\
	heap_parallelscan_initialize(b,a,c)
#define table_parallelscan_reinitialize(a,b)	\
	heap_parallelscan_reinitialize(b)

#endif	/* < PG12 */





#endif	/* PG_COMPAT_H */
