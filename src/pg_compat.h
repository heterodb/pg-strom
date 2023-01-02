/*
 * pg_compat.h
 *
 * Macro definitions to keep compatibility of PostgreSQL internal APIs.
 * --
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
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
 * MEMO: PG12 re-defines 'oid' as a regular column, not a system column.
 * Thus, we don't need to have special treatment for OID, on the other
 * hand, widespread APIs were affected by this change.
 */
#if PG_VERSION_NUM < 120000
#define tupleDescHasOid(tdesc)		((tdesc)->tdhasoid)
#define PgClassTupleGetOid(tuple)	HeapTupleGetOid(tuple)
#define PgProcTupleGetOid(tuple)	HeapTupleGetOid(tuple)
#define PgTypeTupleGetOid(tuple)	HeapTupleGetOid(tuple)
#define PgTriggerTupleGetOid(tuple)	HeapTupleGetOid(tuple)
#define PgExtensionTupleGetOid(tuple) HeapTupleGetOid(tuple)
#define CreateTemplateTupleDesc(a)	CreateTemplateTupleDesc((a), false)
#define ExecCleanTypeFromTL(a)		ExecCleanTypeFromTL((a),false)
#define SystemAttributeDefinition(a)			\
	SystemAttributeDefinition((a),true)
/* also, 'oid' attribute numbers */
#define Anum_pg_class_oid			ObjectIdAttributeNumber
#define Anum_pg_proc_oid			ObjectIdAttributeNumber
#define Anum_pg_type_oid			ObjectIdAttributeNumber
#define Anum_pg_trigger_oid			ObjectIdAttributeNumber
#define Anum_pg_extension_oid		ObjectIdAttributeNumber
#else
#define tupleDescHasOid(tdesc)		(false)
#define PgClassTupleGetOid(tuple)	(((Form_pg_class)GETSTRUCT(tuple))->oid)
#define PgProcTupleGetOid(tuple)	(((Form_pg_proc)GETSTRUCT(tuple))->oid)
#define PgTypeTupleGetOid(tuple)	(((Form_pg_type)GETSTRUCT(tuple))->oid)
#define PgTriggerTupleGetOid(tuple)	(((Form_pg_trigger)GETSTRUCT(tuple))->oid)
#define PgExtensionTupleGetOid(tuple) (((Form_pg_extension)GETSTRUCT(tuple))->oid)
#endif

/*
 * MEMO: PG13 added 'Datum attoptions' argument to InsertPgAttributeTuple
 */
#if PG_VERSION_NUM < 130000
#define InsertPgAttributeTuple(a,b,c,d)	InsertPgAttributeTuple((a),(b),(d))
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
 * MEMO: PG11 adds 'max_workers' argument to the compute_parallel_worker().
 */
#if PG_VERSION_NUM < 110000
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
#define PathNameOpenFilePerm(a,b,c)	PathNameOpenFile((FileName)(a),(b),(c))
#define PathNameOpenFile(a,b)		PathNameOpenFile((FileName)(a),(b),0600)
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
 * MEMO: PG11.2, changed ABI of GenerateTypeDependencies()
 *
 * PG11: 1b55acb2cf48341822261bf9c36785be5ee275db
 * PG10: 2d83863ea2739dc559ed490c284f5c1817db4752
 * PG96: d431dff1af8c220490b84dd978aa3a508f71d415
 *
 * then, PG13 modified ABI of GenerateTypeDependencies
 *
 * then, PG13.5 added 'makeExtensionDep' argument.
 */
#if PG_VERSION_NUM < 110002
#define GenerateTypeDependencies(tup,cat,a,b,c,d,e,f,g)					\
	do {																\
		Form_pg_type __type = (Form_pg_type)GETSTRUCT(tup);				\
																		\
		GenerateTypeDependencies(__type->typnamespace,	/* typeNamespace */	\
								 PgTypeTupleGetOid(tup),/* typeObjectId */ \
								 __type->typrelid,		/* relationOid */ \
								 (c),					/* relationKind */ \
								 __type->typowner,		/* owner */		\
								 __type->typinput,		/* inputProcedure  */ \
								 __type->typoutput,		/* outputProcedure */ \
								 __type->typreceive,	/* receiveProcedure */ \
								 __type->typsend,		/* sendProcedure */	\
								 __type->typmodin,		/* typmodinProcedure */	\
								 __type->typmodout,		/* typmodoutProcedure */ \
								 __type->typanalyze,	/* analyzeProcedure */ \
								 __type->typelem,		/* elementType */ \
								 (d),					/* isImplicitArray */ \
								 __type->typbasetype,	/* baseType */	\
								 __type->typcollation,	/* typeCollation */	\
								 (a),					/* defaultExpr */ \
								 (g));					/* rebuild */	\
	} while(0)
#elif PG_VERSION_NUM < 130000
#define GenerateTypeDependencies(tup,cat,a,b,c,d,e,f,g)		\
	GenerateTypeDependencies(PgTypeTupleGetOid(tup),		\
							 (Form_pg_type)GETSTRUCT(tup),	\
							 (a),(b),(c),(d),(e),(g))
#elif PG_VERSION_NUM < 130005
#define GenerateTypeDependencies(tup,cat,a,b,c,d,e,f,g)		\
	GenerateTypeDependencies((tup),(cat),(a),(b),(c),(d),(e),(g))
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
 * It affects widespread APIs we had used in PG11 or older.
 */
#if PG_VERSION_NUM < 120000
typedef HeapScanDesc					TableScanDesc;
typedef HeapScanDescData				TableScanDescData;
typedef ParallelHeapScanDesc			ParallelTableScanDesc;
typedef ParallelHeapScanDescData		ParallelTableScanDescData;

#define table_open(a,b)					heap_open(a,b)
#define table_openrv(a,b)				heap_openrv(a,b)
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

/*
 * PG12 and newer required TupleTableSlot to have TupleTableSlotOps,
 * for better support of pluggable storage engines. It affects to
 * the widespread relevant APIs.
 *
 * PG10 or older didn't assign TupleDesc at ExecInitScanTupleSlot(),
 * so we had to call ExecAssignScanType() additionally.
 */
#if PG_VERSION_NUM < 110000
#define ExecInitScanTupleSlot(estate,ss,tdesc,tts_ops)	\
	do {												\
		ExecInitScanTupleSlot((estate),(ss));			\
		ExecAssignScanType((ss),(tdesc));				\
	} while(0)
#else
#define ExecInitScanTupleSlot(estate,ss,tdesc,tts_ops)	\
	ExecInitScanTupleSlot((estate),(ss),(tdesc))
#endif

#define MakeSingleTupleTableSlot(tdesc,tts_ops)			\
	MakeSingleTupleTableSlot((tdesc))
#define ExecForceStoreHeapTuple(tuple,slot,shouldFree)	\
	ExecStoreTuple((tuple),(slot),InvalidBuffer,(shouldFree))
static inline HeapTuple
ExecFetchSlotHeapTuple(TupleTableSlot *slot,
					   bool materialize, bool *shouldFree)
{
	Assert(!materialize && !shouldFree);
	return ExecFetchSlotTuple(slot);
}
#endif	/* < PG12 */

/*
 * PG12 adds RELKIND_HAS_STORAGE macro to determine the relation kind
 * that usually has physical storage.
 */
#if PG_VERSION_NUM < 120000
#define RELATION_HAS_STORAGE(rel)							\
	(RelationGetForm(rel)->relkind == RELKIND_RELATION ||	\
	 RelationGetForm(rel)->relkind == RELKIND_INDEX ||		\
	 RelationGetForm(rel)->relkind == RELKIND_SEQUENCE ||	\
	 RelationGetForm(rel)->relkind == RELKIND_TOASTVALUE || \
	 RelationGetForm(rel)->relkind == RELKIND_MATVIEW)
#else
#define RELATION_HAS_STORAGE(rel)				\
	RELKIND_HAS_STORAGE(RelationGetForm(rel)->relkind)
#endif

/*
 * At PG13, 6f38d4dac381b5b8bead302a0b4f81761042cd25 changed
 * declaration of CheckForSerializableConflictOut(), and its role
 * was inherited to HeapCheckForSerializableConflictOut().
 */
#if PG_VERSION_NUM < 130000
#define HeapCheckForSerializableConflictOut(a,b,c,d,e)	\
	CheckForSerializableConflictOut(a,b,c,d,e)
#endif

/*
 * PG12 (commit: 1ef6bd2954c4ec63ff8a2c9c4ebc38251d7ef5c5) don't
 * require return slots for nodes without projection.
 * Instead of the ps_ResultTupleSlot->tts_tupleDescriptor,
 * ps_ResultTupleDesc is now reliable source to determine the tuple
 * definition. For the compatibility to PG11 or older, we use the
 * access macro below.
 */
#if PG_VERSION_NUM < 120000
#define planStateResultTupleDesc(ps)			\
	((ps)->ps_ResultTupleSlot->tts_tupleDescriptor)
#else
#define planStateResultTupleDesc(ps)	((ps)->ps_ResultTupleDesc)
#endif

/*
 * PG12 added 'pathkey' argument of create_append_path().
 * It shall be ignored on the older versions.
 */
#if PG_VERSION_NUM < 120000
#define create_append_path(a,b,c,d,e,f,g,h,i,j)	\
	create_append_path((a),(b),(c),(d),(f),(g),(h),(i),(j))
#endif

/*
 * PG11 added 'flags' argument for BackgroundWorkerInitializeConnection
 */
#if PG_VERSION_NUM < 110000
#define BackgroundWorkerInitializeConnection(dbname,username,flags)	\
	BackgroundWorkerInitializeConnection((dbname),(username))
#endif

/*
 * PG12 deprecated get_func_cost(), instead of add_function_cost()
 */
#if PG_VERSION_NUM < 120000
static inline void
add_function_cost(PlannerInfo *root, Oid funcid, Node *node,
				  QualCost *cost)
{
	cost->per_tuple += get_func_cost(funcid) * cpu_operator_cost;
}
#endif

/*
 * PG13 changed varnoold/varoattno field names of Var-node,
 * to varnosyn/varattnosyn. Be careful to use these names.
 * 9ce77d75c5ab094637cc4a446296dc3be6e3c221
 */
#if PG_VERSION_NUM < 130000
#define varnosyn		varnoold
#define varattnosyn		varoattno
#endif

/*
 * MEMO: EventTriggerData->tag was declared as 'const char *' in PG12 or older.
 * Then, PG13 re-defined this field as CommandTag enum.
 * This field is used to check which context call the event-trigger function,
 * and usually not performance intensive. So, we continue to use cstring
 * comparion for the tag identification, like:
 *
 * if (strcmp(GetCommandTagName(trigdata->tag), "CREATE FOREIGN TABLE") == 0)
 *            :
 */
#if PG_VERSION_NUM < 130000
#define GetCommandTagName(tag)		(tag)
#endif

/*
 * PG13 replaced set_deparse_context_planstate by set_deparse_context_plan.
 * As literal, it allows to construct a deparse context by Plan-node,
 * however, we have no way to reference PlanState from Plan of course.
 * So, exceptionally, we continue to use the old API here, because PlanState
 * has its Plan node.
 */
#if PG_VERSION_NUM >= 130000
#define set_deparse_context_planstate(deparse_cxt,planstate,ancestors)	\
	set_deparse_context_plan((deparse_cxt),((PlanState *)(planstate))->plan,ancestors)
#endif

/*
 * PG14 changed API to pick up var-nodes from the expression.
 */
#if PG_VERSION_NUM < 140000
#define pull_varnos_of_level(a,b,c)		pull_varnos_of_level((b),(c))
#define pull_varnos(a,b)				pull_varnos(b)
#endif

/*
 * PG15 enforced RequestAddinShmemSpace() must be called on shmem_request_hook
 */
#if PG_VERSION_NUM < 150000
typedef void (*shmem_request_hook_type) (void);
extern shmem_request_hook_type	shmem_request_hook;
#endif

/*
 * PG15 requires CustomPath CUSTOMPATH_SUPPORT_PROJECTION if it supports
 * projection.
 */
#if PG_VERSION_NUM < 150000
#define CUSTOMPATH_SUPPORT_PROJECTION		0x0004
#endif

#endif	/* PG_COMPAT_H */
