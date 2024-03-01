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

#endif	/* PG_COMPAT_H */
