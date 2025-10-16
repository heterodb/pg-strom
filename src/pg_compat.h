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
 * MEMO: PostgreSQL v16 adds escontext arguments on the various internal
 * functions. it should be ignored at the PostgreSQL v15
 */
#if  PG_VERSION_NUM < 160000
#define make_range(a,b,c,d,e)		make_range(a,b,c,d)
#endif

/*
 * MEMO: PostgreSQL v16 changed layout and field name of SMgrRelationData
 * So, it affects to some internal APIs.
 */
#if PG_VERSION_NUM < 160000
#define	smgr_relpath(smgr,forknum)		relpath((smgr)->smgr_rnode,(forknum))
#define smgr_init_buffer_tag(tag,smgr,fork_num,block_num)	\
	INIT_BUFFERTAG((*tag),(smgr)->smgr_rnode.node,(fork_num),(block_num))
#else
#if PG_VERSION_NUM < 180000
#define smgr_relpath(smgr,forknum)		relpath((smgr)->smgr_rlocator,(forknum))
#else
#define smgr_relpath(smgr,forknum)		pstrdup(relpath((smgr)->smgr_rlocator,(forknum)).str)
#endif
#define smgr_init_buffer_tag(tag,smgr,fork_num,block_num)				\
	InitBufferTag((tag),&(smgr)->smgr_rlocator.locator,(fork_num),(block_num))
#endif

/*
 * MEMO: PostgreSQL v16 replaced pg_XXXX_aclcheck() APIs by object_aclcheck.
 * We put a thin compatibility layer here.
 */
#if PG_VERSION_NUM >= 160000
#define pg_type_aclcheck(a,b,c)		object_aclcheck(TypeRelationId,(a),(b),(c))
#define pg_proc_aclcheck(a,b,c)		object_aclcheck(ProcedureRelationId,(a),(b),(c))
#endif

/*
 * MEMO: PostgreSQL v16 removed the 7th 'jointype' argument that has been
 * redundant because same value is also stored in the SpecialJoinInfo.
 *
 * MEMO: PostgreSQL v18 changed build_child_join_rel() prototype for better
 * memory usage.
 *
 * git: 513f4472a4a0d294ca64123627ce7b48ce0ee7c1
 */
#if PG_VERSION_NUM < 160000
#define build_child_join_rel(a,b,c,d,e,f,g,h)	\
	build_child_join_rel((a),(b),(c),(d),(e),(f),(f)->jointype)
#elif PG_VERSION_NUM < 180000
#define build_child_join_rel(a,b,c,d,e,f,g,h)	\
	build_child_join_rel((a),(b),(c),(d),(e),(f))
#endif

/*
 * MEMO: PostgreSQL v16 removed 'IsBackgroundWorker' variable, and
 * AmBackgroundWorkerProcess() is used instead.
 */
#if PG_VERSION_NUM < 170000
#define AmBackgroundWorkerProcess()		(IsBackgroundWorker)
#endif

/*
 * MEMO: PostgreSQL v16 defined function pointer type for expression
 * tree walker/mutator
 */
#if PG_VERSION_NUM < 160000
typedef bool (*tree_walker_callback) (Node *node, void *context);
typedef Node *(*tree_mutator_callback) (Node *node, void *context);
#endif

/*
 * MEMO: PostgreSQL v17 removed 'snapshot' argument from the
 * brinRevmapInitialize().
 */
#if PG_VERSION_NUM < 170000
#define brinRevmapInitialize(a,b)				\
	brinRevmapInitialize((a),(b), estate->es_snapshot)
#define brinGetTupleForHeapBlock(a,b,c,d,e,f)	\
	brinGetTupleForHeapBlock((a),(b),(c),(d),(e),(f), estate->es_snapshot)
#endif

/*
 * MEMO: PostgreSQL v17 adds 'runCondition' to the create_windowagg_path().
 * It had been attached from WindowClause later on the Path -> Plan
 * tranformation phase. (So, WindowAggPath does not have 'runCondition').
 */
#if PG_VERSION_NUM < 170000
#define create_windowagg_path(a,b,c,d,e,f,g,h,i)	\
	create_windowagg_path((a),(b),(c),(d),(e),(g),(h),(i))
#define __windowAggPathGetRunCondition(wpath)	((wpath)->winclause->runCondition)
#else
#define __windowAggPathGetRunCondition(wpath)	((wpath)->runCondition)
#endif

/*
 * MEMO: PostgreSQL v17 adds 'fdw_restrictinfo' but we don't use it.
 *
 * MEMO: PostgreSQL v18 adds 'disabled_nodes' to shows number of disabled node
 * in the plan tree, and it affects to create_xxxx() constructor APIs.
 *
 * git: 161320b4b960ee4fe918959be6529ae9b106ea5a
 */
#if PG_VERSION_NUM < 170000
#define create_foreignscan_path(a,b,c,d,e,f,g,h,i,j,k,l)					\
	create_foreignscan_path((a),(b),(c),(d),(f),(g),(h),(i),(j),(l))
#elif PG_VERSION_NUM < 180000
#define create_foreignscan_path(a,b,c,d,e,f,g,h,i,j,k,l)					\
	create_foreignscan_path((a),(b),(c),(d),(f),(g),(h),(i),(j),(k),(l))
#endif

/*
 * MEMO: PostgreSQL v18 adds the 4th argument 'extra' to inform extra bytes
 * for heap_form_minimal_tuple(), but PG-Strom does not use this option.
 */
#if PG_VERSION_NUM < 180000
#define heap_form_minimal_tuple(a,b,c,d)	heap_form_minimal_tuple((a),(b),(c))
#endif

/*
 * MEMO: PostgreSQL v18 removed lc_collate_is_c() that is a checker function
 * to determine the collation is simple enough.
 *
 * git: 06421b08436414b42cd169501005f15adee986f1
 */
#if PG_VERSION_NUM < 180000
#define __collate_is_c(coll_oid)	lc_collate_is_c(coll_oid)
#else
#define __collate_is_c(coll_oid)	(pg_newlocale_from_collation(coll_oid)->collate_is_c)
#endif

/*
 * MEMO: PostgreSQL v18 changed PathKey's field layout to use CompareType
 * to record the sort direction instead of hardcoding btree strategy numbers.
 *
 * git: 8123e91f5aeb26c6e4cf583bb61c99281485af83
 */
#if PG_VERSION_NUM < 180000
#define __PathKeyUseNullsFirstCompare(pk)	((pk)->pk_nulls_first)
#define __PathKeyUseLessThanCompare(pk)		((pk)->pk_strategy == BTLessStrategyNumber)
#define __PathKeyUseGreaterThanCompare(pk)	((pk)->pk_strategy == BTGreaterStrategyNumber)
#else
#define __PathKeyUseNullsFirstCompare(pk)	((pk)->pk_nulls_first)
#define __PathKeyUseLessThanCompare(pk)		((pk)->pk_cmptype == COMPARE_LT)
#define __PathKeyUseGreaterThanCompare(pk)	((pk)->pk_cmptype == COMPARE_GT)
#endif

/*
 * MEMO: CUDA 13.0 changed the following APIs. The existing _v2 APIs were renamed
 * to the primary one, and legacy interfaces were deprecated.
 * - cuMemPrefetchAsync
 * - cuMemAdvise
 */
#if CUDA_VERSION < 13000
#define cuMemPrefetchAsync(a,b,c,d,e)	cuMemPrefetchAsync_v2((a),(b),(c),(d),(e))
#define cuMemAdvise(a,b,c,d)			cuMemAdvise_v2((a),(b),(c),(d))
#endif

#endif	/* PG_COMPAT_H */
