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
#define smgr_relpath(smgr,forknum)		relpath((smgr)->smgr_rlocator,(forknum))
#define smgr_init_buffer_tag(tag,smgr,fork_num,block_num)	\
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
 */
#if PG_VERSION_NUM < 160000
#define build_child_join_rel(a,b,c,d,e,f)	\
	build_child_join_rel((a),(b),(c),(d),(e),(f),(f)->jointype)
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
 * MEMO: PostgreSQL v17 adds 'orstronger' flag for LockHeldByMe()
 * to control whether the caller accept stronger lock mode.
 */
#if PG_VERSION_NUM < 170000
INLINE_FUNCTION(bool)
__LockHeldByMe(const LOCKTAG *locktag,
			   LOCKMODE lockmode,
			   bool orstronger)
{
	while (lockmode <= MaxLockMode)
	{
		if (LockHeldByMe(locktag, lockmode))
			return true;
		if (!orstronger)
			break;
		lockmode++;
	}
	return false;
}
#define LockHeldByMe(a,b,c)		__LockHeldByMe((a),(b),(c))
#endif

#endif	/* PG_COMPAT_H */
