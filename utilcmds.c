/*
 * utilcmds.c
 *
 * Routines to handle utility command queries
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
#include "access/xact.h"
#include "catalog/dependency.h"
#include "catalog/heap.h"
#include "catalog/index.h"
#include "catalog/namespace.h"
#include "catalog/pg_am.h"
#include "catalog/pg_authid.h"
#include "catalog/pg_class.h"
#include "catalog/pg_namespace.h"
#include "catalog/pg_type.h"
#include "commands/defrem.h"
#include "commands/sequence.h"
#include "foreign/fdwapi.h"
#include "foreign/foreign.h"
#include "nodes/makefuncs.h"
#include "nodes/pg_list.h"
#include "miscadmin.h"
#include "tcop/utility.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "pg_strom.h"

/*
 * Saved hook entries
 */
static ProcessUtility_hook_type next_process_utility_hook = NULL;

/*
 * Utility functions to open shadow tables/indexes
 */
static Relation
pgstrom_open_shadow_relation(Relation base_rel, AttrNumber attnum,
							 LOCKMODE lockmode, bool is_index)
{
	char	   *nsp_name;
	char		rel_name[NAMEDATALEN * 2 + 20];
	RangeVar   *range;
	Relation	relation;

	/*
	 * The base relation must be a foreign table being managed by
	 * pg_strom foreign data wrapper.
	 */
	if (RelationGetForm(base_rel)->relkind != RELKIND_FOREIGN_TABLE)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not a foreign table",
						RelationGetRelationName(base_rel))));
	else
	{
		ForeignTable	   *ft = GetForeignTable(RelationGetRelid(base_rel));
		ForeignServer	   *fs = GetForeignServer(ft->serverid);
		ForeignDataWrapper *fdw = GetForeignDataWrapper(fs->fdwid);

		if (GetFdwRoutine(fdw->fdwhandler) != &pgstromFdwHandlerData)
			ereport(ERROR,
					(errcode(ERRCODE_WRONG_OBJECT_TYPE),
					 errmsg("\"%s\" is not managed by pg_strom",
							RelationGetRelationName(base_rel))));
	}

	nsp_name = get_namespace_name(RelationGetForm(base_rel)->relnamespace);
	if (attnum == InvalidAttrNumber)
		snprintf(rel_name, sizeof(rel_name), "%s.%s.%s",
				 nsp_name, RelationGetRelationName(base_rel),
				 (!is_index ? "rowid" : "idx"));
	else
		snprintf(rel_name, sizeof(rel_name), "%s.%s.%s.%s",
				 nsp_name, RelationGetRelationName(base_rel),
				 NameStr(RelationGetDescr(base_rel)->attrs[attnum-1]->attname),
				 (!is_index ? "cs" : "idx"));
	if (strlen(rel_name) >= NAMEDATALEN)
		ereport(ERROR,
				(errcode(ERRCODE_NAME_TOO_LONG),
				 errmsg("Name of shadow index: \"%s\" too long", rel_name)));
	pfree(nsp_name);

	range = makeRangeVar(PGSTROM_SCHEMA_NAME, rel_name, -1);
	relation = relation_openrv(range, lockmode);
	pfree(range);

	return relation;
}

Relation
pgstrom_open_shadow_table(Relation base_rel,
						  AttrNumber attnum,
						  LOCKMODE lockmode)
{
	Relation	relation;

	relation = pgstrom_open_shadow_relation(base_rel, attnum, lockmode, false);
	if (RelationGetForm(relation)->relkind != RELKIND_RELATION)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not a table",
						RelationGetRelationName(relation))));
	return relation;
}

Relation
pgstrom_open_shadow_index(Relation base_rel,
						  AttrNumber attnum,
						  LOCKMODE lockmode)
{
	Relation	relation;

	relation = pgstrom_open_shadow_relation(base_rel, attnum, lockmode, true);
	if (RelationGetForm(relation)->relkind != RELKIND_INDEX)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an index",
						RelationGetRelationName(relation))));
	return relation;
}

RangeVar *
pgstrom_lookup_shadow_sequence(Relation base_rel)
{
	char	   *nsp_name;
	char		seq_name[NAMEDATALEN * 2 + 20];
	RangeVar   *range;

	nsp_name = get_namespace_name(RelationGetForm(base_rel)->relnamespace);
	snprintf(seq_name, sizeof(seq_name), "%s.%s.seq",
			 nsp_name, RelationGetRelationName(base_rel));
	range = makeRangeVar(PGSTROM_SCHEMA_NAME, pstrdup(seq_name), -1);

	pfree(nsp_name);

	return range;
}

static void
pgstrom_create_shadow_index(Relation base_rel, Relation cs_rel,
							Form_pg_attribute attr)
{
	char	   *nsp_name;
	char		idx_name[NAMEDATALEN * 2 + 20];
	IndexInfo  *idx_info;
	Oid			collationOid[1];
    Oid         opclassOid[1];
    int16       colOptions[1];

	nsp_name = get_namespace_name(RelationGetForm(base_rel)->relnamespace);
	if (!attr)
		snprintf(idx_name, sizeof(idx_name), "%s.%s.idx",
				 nsp_name, RelationGetRelationName(base_rel));
	else
		snprintf(idx_name, sizeof(idx_name), "%s.%s.%s.idx",
				 nsp_name, RelationGetRelationName(base_rel),
				 NameStr(attr->attname));

	if (strlen(idx_name) >= NAMEDATALEN)
		ereport(ERROR,
                (errcode(ERRCODE_NAME_TOO_LONG),
                 errmsg("Name of shadow index: \"%s\" too long", idx_name)));

	idx_info = makeNode(IndexInfo);
	idx_info->ii_NumIndexAttrs = 1;
	idx_info->ii_KeyAttrNumbers[0] = Anum_pg_strom_rowid;
	idx_info->ii_Expressions = NIL;
	idx_info->ii_ExpressionsState = NIL;
	idx_info->ii_Predicate = NIL;
	idx_info->ii_PredicateState = NIL;
	idx_info->ii_ExclusionOps = NULL;
	idx_info->ii_ExclusionProcs = NULL;
	idx_info->ii_ExclusionStrats = NULL;
	idx_info->ii_Unique = true;
	idx_info->ii_ReadyForInserts = true;
	idx_info->ii_Concurrent = false;
	idx_info->ii_BrokenHotChain = false;

	collationOid[0] = InvalidOid;
    opclassOid[0] = GetDefaultOpClass(INT8OID, BTREE_AM_OID);
    if (!OidIsValid(opclassOid[0]))
		elog(ERROR, "no default operator class found on (int8, btree)");
	colOptions[0] = 0;

	index_create(cs_rel,			/* heapRelation */
				 idx_name,			/* indexRelationName */
				 InvalidOid,		/* indexRelationId */
				 InvalidOid,		/* relFileNode */
				 idx_info,			/* indexInfo */
				 list_make1("rowid"),	/* indexColNames */
				 BTREE_AM_OID,		/* accessMethodObjectId */
				 cs_rel->rd_rel->reltablespace,	/* tableSpaceId */
				 collationOid,		/* collationObjectId */
				 opclassOid,		/* OpClassObjectId */
				 colOptions,		/* coloptions */
				 (Datum) 0,			/* reloptions */
				 false,				/* isprimary */
				 false,				/* isconstraint */
				 false,				/* deferrable */
				 false,				/* initdeferred */
				 false,				/* allow_system_table_mods */
				 false,				/* skip_build */
				 false);			/* concurrent */
}

static void
pgstrom_create_shadow_table(Oid namespaceId, Relation base_rel,
							Form_pg_attribute attr)
{
	char	   *nsp_name;
	char		cs_name[NAMEDATALEN * 2 + 20];
	TupleDesc	tupdesc;
	Oid			cs_relid;
	Relation	cs_rel;
	ObjectAddress	base_address;
	ObjectAddress	cs_address;

	nsp_name = get_namespace_name(RelationGetForm(base_rel)->relnamespace);
	if (!attr)
		snprintf(cs_name, sizeof(cs_name), "%s.%s.rowid",
				 nsp_name, RelationGetRelationName(base_rel));
	else
		snprintf(cs_name, sizeof(cs_name), "%s.%s.%s.cs",
				 nsp_name, RelationGetRelationName(base_rel),
				 NameStr(attr->attname));

	if (strlen(cs_name) >= NAMEDATALEN)
		ereport(ERROR,
				(errcode(ERRCODE_NAME_TOO_LONG),
				 errmsg("Name of shadow table: \"%s\" too long", cs_name)));
	pfree(nsp_name);

	if (!attr)
	{
		tupdesc = CreateTemplateTupleDesc(Natts_pg_strom - 1, false);
		TupleDescInitEntry(tupdesc, Anum_pg_strom_rowid,
						   "rowid",  INT8OID,  -1, 0);
		TupleDescInitEntry(tupdesc, Anum_pg_strom_nitems,
						   "nitems", INT4OID,  -1, 0);
		TupleDescInitEntry(tupdesc, Anum_pg_strom_isnull,
						   "isnull", BYTEAOID, -1, 0);
		tupdesc->attrs[Anum_pg_strom_isnull - 1]->attstorage = 'p';
	}
	else
	{
		tupdesc = CreateTemplateTupleDesc(Natts_pg_strom, false);
		TupleDescInitEntry(tupdesc, Anum_pg_strom_rowid,
						   "rowid",  INT8OID,  -1, 0);
		TupleDescInitEntry(tupdesc, Anum_pg_strom_nitems,
						   "nitems", INT4OID,  -1, 0);
		TupleDescInitEntry(tupdesc, Anum_pg_strom_isnull,
						   "isnull", BYTEAOID, -1, 0);
		TupleDescInitEntry(tupdesc, Anum_pg_strom_values,
						   "valued", BYTEAOID, -1, 0);
		/*
		 * PG-Strom wants to keep varlena value being inlined, and never
		 * uses external toast relation due to the performance reason.
		 * So, we override the default setting of type definitions.
		 */
		tupdesc->attrs[Anum_pg_strom_isnull - 1]->attstorage = 'p';
		tupdesc->attrs[Anum_pg_strom_values - 1]->attstorage = 'p';
	}

	cs_relid = heap_create_with_catalog(cs_name,
										namespaceId,
										InvalidOid,
										InvalidOid,
										InvalidOid,
										InvalidOid,
										base_rel->rd_rel->relowner,
										tupdesc,
										NIL,
										RELKIND_RELATION,
										base_rel->rd_rel->relpersistence,
										false,
										false,
										true,
										0,
										ONCOMMIT_NOOP,
										(Datum) 0,
										false,
										false);
    Assert(OidIsValid(cs_relid));

	/* make the new shadow table visible */
	CommandCounterIncrement();

	/* ShareLock is not really needed here, but take it anyway */
	cs_rel = heap_open(cs_relid, ShareLock);

	/* Create a unique index on the rowid column */
	pgstrom_create_shadow_index(base_rel, cs_rel, attr);

	heap_close(cs_rel, NoLock);

	/* Register dependency between base and shadow tables */
	cs_address.classId = RelationRelationId;
	cs_address.objectId = cs_relid;
	cs_address.objectSubId = 0;
	base_address.classId  = RelationRelationId;
	base_address.objectId = RelationGetRelid(base_rel);
	base_address.objectSubId = (!attr ? 0 : attr->attnum);

	recordDependencyOn(&cs_address, &base_address, DEPENDENCY_INTERNAL);

    /* Make changes visible */
    CommandCounterIncrement();
}


/*
 * pgstrom_create_rowid_seq
 *
 * create "<base_schema>.<base_rel>.seq" sequence of pg_strom schema
 * that enables to generate unique number between 0 to 2^48-1 by
 * PGSTROM_CHUNK_SIZE.
 */
static void
pgstrom_create_shadow_sequence(Oid namespaceId, Relation base_rel)
{
	CreateSeqStmt  *seq_stmt;
	char		   *nsp_name;
	char			seq_name[2*NAMEDATALEN + 20];
	char			rel_name[2*NAMEDATALEN + 20];
	List		   *rowid_namelist;

	nsp_name = get_namespace_name(RelationGetForm(base_rel)->relnamespace);
	snprintf(rel_name, sizeof(rel_name), "%s.%s.rowid",
			 nsp_name, RelationGetRelationName(base_rel));
	snprintf(seq_name, sizeof(seq_name), "%s.%s.seq",
			 nsp_name, RelationGetRelationName(base_rel));
	Assert(strlen(rel_name) < NAMEDATALEN);

	seq_stmt = makeNode(CreateSeqStmt);
	seq_stmt->sequence = makeRangeVar(PGSTROM_SCHEMA_NAME, seq_name, -1);
	rowid_namelist = list_make3(makeString(PGSTROM_SCHEMA_NAME),
								makeString(rel_name),
								makeString("rowid"));
	seq_stmt->options = list_make3(
		makeDefElem("minvalue", (Node *)makeInteger(0)),
		makeDefElem("maxvalue", (Node *)makeInteger((1UL<<48) - 1)),
		makeDefElem("owned_by", (Node *)rowid_namelist));
	seq_stmt->ownerId = RelationGetForm(base_rel)->relowner;

	DefineSequence(seq_stmt);
}

static void
pgstrom_process_post_create(RangeVar *base_range)
{
	Relation	base_rel;
	Oid			namespaceId;
	AttrNumber	attnum;
	Oid			save_userid;
	int			save_sec_context;

	/* Ensure the base relation being visible */
	CommandCounterIncrement();

	/*
	 * Ensure existence of the schema that shall stores all the
	 * corresponding stuff. If not found, create it anyway.
	 */
	namespaceId = get_namespace_oid(PGSTROM_SCHEMA_NAME, true);
	if (!OidIsValid(namespaceId))
	{
		namespaceId = NamespaceCreate(PGSTROM_SCHEMA_NAME,
									  BOOTSTRAP_SUPERUSERID);
		CommandCounterIncrement();
	}

	/*
	 * Open the base relation; exclusive lock should be already
	 * acquired, so we use NoLock instead.
	 */
	base_rel = heap_openrv(base_range, NoLock);

	/* switch current credential of database users */
	GetUserIdAndSecContext(&save_userid, &save_sec_context);
	SetUserIdAndSecContext(BOOTSTRAP_SUPERUSERID, save_sec_context);

	/* create shadow table and corresponding index */
	pgstrom_create_shadow_table(namespaceId, base_rel, NULL);
	for (attnum=0; attnum < RelationGetNumberOfAttributes(base_rel); attnum++)
	{
		Form_pg_attribute	attr
			= RelationGetDescr(base_rel)->attrs[attnum];
		pgstrom_create_shadow_table(namespaceId, base_rel, attr);
	}
	/* create a sequence to generate rowid */
	pgstrom_create_shadow_sequence(namespaceId, base_rel);

	/* restore security setting and close the base relation */
	SetUserIdAndSecContext(save_userid, save_sec_context);

	heap_close(base_rel, NoLock);
}

static void
pgstrom_process_post_alter_schema(void)
{}

static void
pgstrom_process_post_alter_rename(void)
{}

static void
pgstrom_process_post_alter_owner(void)
{}

/*
 * pgstrom_process_utility_command
 *
 * Entrypoint of the ProcessUtility hook; that handles post DDL operations.
 */
static void
pgstrom_process_utility_command(Node *stmt,
								const char *queryString,
								ParamListInfo params,
								bool isTopLevel,
								DestReceiver *dest,
								char *completionTag)
{
	if (next_process_utility_hook)
		(*next_process_utility_hook)(stmt, queryString, params,
									 isTopLevel, dest, completionTag);
	else
		standard_ProcessUtility(stmt, queryString, params,
								isTopLevel, dest, completionTag);
	/*
	 * Do we need post ddl works?
	 */
	if (IsA(stmt, CreateForeignTableStmt))
	{
		CreateForeignTableStmt *cfts = (CreateForeignTableStmt *)stmt;
		ForeignServer	   *fs;
		ForeignDataWrapper *fdw;

		fs = GetForeignServerByName(cfts->servername, false);
		fdw = GetForeignDataWrapper(fs->fdwid);
		if (GetFdwRoutine(fdw->fdwhandler) == &pgstromFdwHandlerData)
		   	pgstrom_process_post_create(cfts->base.relation);
	}
	else if (IsA(stmt, AlterObjectSchemaStmt))
	{
		pgstrom_process_post_alter_schema();
	}
	else if (IsA(stmt, RenameStmt))
	{
		pgstrom_process_post_alter_rename();
	}
	else if (IsA(stmt, AlterOwnerStmt))
	{
		pgstrom_process_post_alter_owner();
	}
}

/*
 * pgstrom_utilcmds_init
 *
 * Registers ProcessUtility hook
 */
void
pgstrom_utilcmds_init(void)
{
	next_process_utility_hook = ProcessUtility_hook;
	ProcessUtility_hook = pgstrom_process_utility_command;
}
