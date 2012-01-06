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
 * pgstrom_open_relation_set
 *
 * Open the relation set with/without related indexes
 */
RelationSet
pgstrom_open_relation_set(Relation base_rel,
						  LOCKMODE lockmode, bool with_index)
{
	RelationSet	relset;
	AttrNumber	i, nattrs;
	RangeVar   *range;
	char	   *base_schema;
	char		namebuf[NAMEDATALEN * 3 + 20];

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

	/*
	 * Setting up RelationSet
	 */
	nattrs = RelationGetNumberOfAttributes(base_rel);
	relset = palloc0(sizeof(RelationSetData));
	relset->cs_rel = palloc0(sizeof(Relation) * nattrs);
	relset->cs_idx = palloc0(sizeof(Relation) * nattrs);
	relset->base_rel = base_rel;

	/*
	 * Open the underlying tables and corresponding indexes
	 */
	range = makeRangeVar(PGSTROM_SCHEMA_NAME, namebuf, -1);
	base_schema = get_namespace_name(RelationGetForm(base_rel)->relnamespace);

	snprintf(namebuf, sizeof(namebuf), "%s.%s.rowid",
			 base_schema, RelationGetRelationName(base_rel));
	relset->rowid_rel = relation_openrv(range, lockmode);
	if (RelationGetForm(relset->rowid_rel)->relkind != RELKIND_RELATION)
		ereport(ERROR,
                (errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not a regular table",
						RelationGetRelationName(relset->rowid_rel))));
	if (with_index)
	{
		snprintf(namebuf, sizeof(namebuf), "%s.%s.idx",
				 base_schema, RelationGetRelationName(base_rel));
		relset->rowid_idx = relation_openrv(range, lockmode);
		if (RelationGetForm(relset->rowid_idx)->relkind != RELKIND_INDEX)
			ereport(ERROR,
					(errcode(ERRCODE_WRONG_OBJECT_TYPE),
					 errmsg("\"%s\" is not an index",
							RelationGetRelationName(relset->rowid_idx))));
	}

	for (i = 0; i < nattrs; i++)
	{
		Form_pg_attribute attr = RelationGetDescr(base_rel)->attrs[i];

		if (attr->attisdropped)
			continue;

		snprintf(namebuf, sizeof(namebuf), "%s.%s.%s.cs",
				 base_schema,
                 RelationGetRelationName(base_rel),
                 NameStr(attr->attname));
		relset->cs_rel[i] = relation_openrv(range, lockmode);
		if (RelationGetForm(relset->cs_rel[i])->relkind != RELKIND_RELATION)
			ereport(ERROR,
					(errcode(ERRCODE_WRONG_OBJECT_TYPE),
					 errmsg("\"%s\" is not a regular table",
							RelationGetRelationName(relset->cs_rel[i]))));
		if (with_index)
		{
			snprintf(namebuf, sizeof(namebuf), "%s.%s.%s.idx",
					 base_schema,
					 RelationGetRelationName(base_rel),
					 NameStr(attr->attname));
			relset->cs_idx[i] = relation_openrv(range, lockmode);
			if (RelationGetForm(relset->cs_idx[i])->relkind != RELKIND_INDEX)
				ereport(ERROR,
						(errcode(ERRCODE_WRONG_OBJECT_TYPE),
						 errmsg("\"%s\" is not an index",
								RelationGetRelationName(relset->cs_idx[i]))));
		}
	}

	/*
	 * Also, solve the sequence name
	 */
	snprintf(namebuf, sizeof(namebuf), "%s.%s.seq",
			 base_schema, RelationGetRelationName(base_rel));
	relset->rowid_seqid = RangeVarGetRelid(range, NoLock, false);

	return relset;
}

/*
 * pgstrom_close_relation_set
 *
 * Close the set of relations
 */
void
pgstrom_close_relation_set(RelationSet relset, LOCKMODE lockmode)
{
	AttrNumber	i, nattrs = RelationGetNumberOfAttributes(relset->base_rel);

	relation_close(relset->rowid_rel, lockmode);
	if (relset->rowid_idx)
		relation_close(relset->rowid_idx, lockmode);

	for (i=0; i < nattrs; i++)
	{
		if (relset->cs_rel[i])
			relation_close(relset->cs_rel[i], lockmode);
		if (relset->cs_idx[i])
			relation_close(relset->cs_idx[i], lockmode);
	}
	pfree(relset->cs_rel);
	pfree(relset->cs_idx);
	pfree(relset);
}

/*
 * pgstrom_create_rowid_index
 *
 * create "<base_schema>.<base_rel>.(<column>.)idx" index of pg_strom
 * schema to find-up a tuple that contains a particular rowid.
 */
static void
pgstrom_create_rowid_index(Relation base_rel, const char *attname,
						   Relation store_rel, AttrNumber indexed_anum)
{
	char	   *nsp_name;
	char		index_name[3 * NAMEDATALEN + 20];
	char	   *indexed_attname;
	IndexInfo  *index_info;
	Oid			collationObjectId[1];
	Oid			classObjectId[1];
	int16		coloptions[1];

	nsp_name = get_namespace_name(RelationGetForm(base_rel)->relnamespace);
	if (!attname)
		snprintf(index_name, sizeof(index_name), "%s.%s.idx",
				 nsp_name, RelationGetRelationName(base_rel));
	else
		snprintf(index_name, sizeof(index_name), "%s.%s.%s.idx",
				 nsp_name, RelationGetRelationName(base_rel), attname);

	if (strlen(index_name) >= NAMEDATALEN - 1)
		ereport(ERROR,
				(errcode(ERRCODE_NAME_TOO_LONG),
				 errmsg("Name of shadow index: \"%s\" too long", index_name)));

	indexed_attname =
		NameStr(store_rel->rd_att->attrs[indexed_anum - 1]->attname);

	index_info = makeNode(IndexInfo);
	index_info->ii_NumIndexAttrs = 1;
	index_info->ii_KeyAttrNumbers[0] = indexed_anum;
	index_info->ii_Expressions = NIL;
	index_info->ii_ExpressionsState = NIL;
	index_info->ii_Predicate = NIL;
	index_info->ii_PredicateState = NIL;
	index_info->ii_ExclusionOps = NULL;
	index_info->ii_ExclusionProcs = NULL;
	index_info->ii_ExclusionStrats = NULL;
	index_info->ii_Unique = true;
	index_info->ii_ReadyForInserts = true;
	index_info->ii_Concurrent = false;
	index_info->ii_BrokenHotChain = false;

	collationObjectId[0] = InvalidOid;
	coloptions[0] = 0;
	classObjectId[0] = GetDefaultOpClass(INT8OID, BTREE_AM_OID);
	if (!OidIsValid(classObjectId[0]))
		elog(ERROR, "defaule operator class was not found: {int8, btree}");

	index_create(store_rel,			/* heapRelation */
				 index_name,		/* indexRelationName */
				 InvalidOid,		/* indexRelationId */
				 InvalidOid,		/* relFileNode */
				 index_info,		/* indexInfo */
				 list_make1(indexed_attname),	/* indexColNames */
				 BTREE_AM_OID,		/* accessMethodObjectId */
				 store_rel->rd_rel->reltablespace, /* tableSpaceId */
				 collationObjectId,	/* collationObjectId */
				 classObjectId,		/* OpClassObjectId */
				 coloptions,		/* coloptions */
				 (Datum) 0,			/* reloptions */
				 false,				/* isprimary */
				 false,				/* isconstraint */
				 false,				/* deferrable */
				 false,				/* initdeferred */
				 false,				/* allow_system_table_mods */
				 false,				/* skip_build */
				 false);			/* concurrent */

	elog(NOTICE, "pg_strom implicitly created a shadow index: \"%s.%s\"",
		 PGSTROM_SCHEMA_NAME, index_name);
}

/*
 * pgstrom_create_rowid_map
 *
 * create "<base_schema>.<base_rel>.rowid" table of pg_strom schema
 * that has "rowid(int8)" and "rowid_map(VarBit)" to track which rowid
 * has been in use.
 */
static void
pgstrom_create_rowid_map(Oid namespaceId, Relation base_rel)
{
	char		   *nsp_name;
	char			store_name[NAMEDATALEN * 2 + 20];
	Oid				store_oid;
	Relation		store_rel;
	TupleDesc		tupdesc;
	ObjectAddress	base_address;
	ObjectAddress	store_address;

	nsp_name = get_namespace_name(RelationGetForm(base_rel)->relnamespace);
	snprintf(store_name, sizeof(store_name), "%s.%s.rowid",
			 nsp_name, RelationGetRelationName(base_rel));
	if (strlen(store_name) >= NAMEDATALEN - 1)
		ereport(ERROR,
				(errcode(ERRCODE_NAME_TOO_LONG),
				 errmsg("Name of shadow table: \"%s\" too long", store_name)));

	tupdesc = CreateTemplateTupleDesc(2, false);
	TupleDescInitEntry(tupdesc,
					   (AttrNumber) 1,
					   "rowid",
					   INT8OID,
					   -1, 0);
	TupleDescInitEntry(tupdesc,
					   (AttrNumber) 2,
					   "rowid_map",
					   VARBITOID,
					   -1, 0);
	/*
	 * Pg_strom want to keep varlena data being inlined; never uses external
	 * toast relation due to the performance reason. So, we override the
	 * default setting of pg_type definitions.
	 */
	tupdesc->attrs[0]->attstorage = 'p';
	tupdesc->attrs[1]->attstorage = 'm';

	store_oid = heap_create_with_catalog(store_name,
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
	Assert(OidIsValid(store_oid));

	elog(NOTICE, "pg_strom implicitly created a shadow table: \"%s.%s\"",
		 PGSTROM_SCHEMA_NAME, store_name);

	/* make the shadow table visible */
	CommandCounterIncrement();

	/* ShareLock is not really needed here, but take it anyway */
    store_rel = heap_open(store_oid, ShareLock);

	/* Create a unique index on the rowid */
	pgstrom_create_rowid_index(base_rel, NULL, store_rel, (AttrNumber) 1);

	heap_close(store_rel, NoLock);

	/* Register dependency between base and shadow tables */
	base_address.classId  = RelationRelationId;
	base_address.objectId =	RelationGetRelid(base_rel);
	base_address.objectSubId = 0;
	store_address.classId = RelationRelationId;
	store_address.objectId = store_oid;
	store_address.objectSubId = 0;

	recordDependencyOn(&store_address, &base_address, DEPENDENCY_INTERNAL);

	/* Make changes visible */
	CommandCounterIncrement();
}

/*
 * pgstrom_create_column_store
 *
 * create "<base_schema>.<base_rel>.<column>.cs" table of pg_strom
 * schema that has "rowid(int8)", "nulls(VarBit)" and "values(bytes)"
 * to store the values of original columns. The maximum number of
 * values being stored in a tuple depends on length of unit data.
 */
static void
pgstrom_create_column_store(Oid namespaceId, Relation base_rel,
							Form_pg_attribute attform)
{
	char		   *nsp_name;
	char			store_name[NAMEDATALEN * 3 + 20];
	Oid				store_oid;
	Relation		store_rel;
	TupleDesc		tupdesc;
	Oid				array_oid;
	ObjectAddress	base_address;
	ObjectAddress	store_address;

	nsp_name = get_namespace_name(RelationGetForm(base_rel)->relnamespace);
	snprintf(store_name, sizeof(store_name), "%s.%s.%s.cs",
			 nsp_name,
			 RelationGetRelationName(base_rel),
			 NameStr(attform->attname));
	if (strlen(store_name) >= NAMEDATALEN - 1)
		ereport(ERROR,
				(errcode(ERRCODE_NAME_TOO_LONG),
				 errmsg("Name of shadow table: \"%s\" too long", store_name)));

	if (attform->attndims == 0)
		array_oid = get_array_type(attform->atttypid);
	else
		array_oid = get_array_type(BYTEAOID);
	if (!OidIsValid(array_oid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("could not find array type for data type %s",
						format_type_be(attform->atttypid))));
	tupdesc = CreateTemplateTupleDesc(2, false);
	TupleDescInitEntry(tupdesc,
					   (AttrNumber) 1,
					   "rowid",
					   INT8OID,
					   -1, 0);
	TupleDescInitEntry(tupdesc,
					   (AttrNumber) 2,
					   "values",
					   array_oid,
					   -1, 1);
	/*
	 * Pg_strom want to keep varlena data being inlined; never uses external
	 * toast relation due to the performance reason. So, we override the
	 * default setting of pg_type definitions.
	 */
	tupdesc->attrs[0]->attstorage = 'p';
	tupdesc->attrs[1]->attstorage = 'm';

	store_oid = heap_create_with_catalog(store_name,
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
	Assert(OidIsValid(store_oid));

	elog(NOTICE, "pg_strom implicitly created a shadow table: \"%s.%s\"",
		 PGSTROM_SCHEMA_NAME, store_name);

	/* make the shadow table visible */
	CommandCounterIncrement();

	/* ShareLock is not really needed here, but take it anyway */
    store_rel = heap_open(store_oid, ShareLock);

	/* Create a unique index on the rowid */
	pgstrom_create_rowid_index(base_rel, NameStr(attform->attname),
							   store_rel, (AttrNumber) 1);

	heap_close(store_rel, NoLock);

	/* Register dependency between base and shadow tables */
	base_address.classId  = RelationRelationId;
	base_address.objectId =	RelationGetRelid(base_rel);
	base_address.objectSubId = attform->attnum;
	store_address.classId = RelationRelationId;
	store_address.objectId = store_oid;
	store_address.objectSubId = 0;

	recordDependencyOn(&store_address, &base_address, DEPENDENCY_INTERNAL);

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
pgstrom_create_rowid_seq(Oid namespaceId, Relation base_rel)
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

	elog(NOTICE, "pg_strom implicitly created a shadow table: \"%s.%s\"",
		 PGSTROM_SCHEMA_NAME, seq_name);
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

	/* create pg_strom.<base_schema>.<base_rel>.rowid */
	pgstrom_create_rowid_map(namespaceId, base_rel);

	/* create pg_strom.<base_schema>.<base_rel>.<column> */
	for (attnum = 0;
		 attnum < RelationGetNumberOfAttributes(base_rel);
		 attnum++)
	{
		pgstrom_create_column_store(namespaceId, base_rel,
									RelationGetDescr(base_rel)->attrs[attnum]);
	}

	/* create pg_strom.<base_schema>.<base_rel>.seq */
	pgstrom_create_rowid_seq(namespaceId, base_rel);


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
