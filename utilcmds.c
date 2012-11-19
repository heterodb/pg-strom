/*
 * utilcmds.c
 *
 * Routines to handle utility command queries
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "access/htup_details.h"
#include "access/xact.h"
#include "catalog/dependency.h"
#include "catalog/heap.h"
#include "catalog/index.h"
#include "catalog/namespace.h"
#include "catalog/pg_am.h"
#include "catalog/pg_attribute.h"
#include "catalog/pg_authid.h"
#include "catalog/pg_class.h"
#include "catalog/pg_namespace.h"
#include "catalog/pg_type.h"
#include "commands/defrem.h"
#include "commands/sequence.h"
#include "commands/tablecmds.h"
#include "foreign/fdwapi.h"
#include "foreign/foreign.h"
#include "nodes/makefuncs.h"
#include "nodes/pg_list.h"
#include "miscadmin.h"
#include "tcop/utility.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/syscache.h"
#include "utils/tqual.h"
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

		if (GetFdwRoutine(fdw->fdwhandler) != &PgStromFdwHandlerData)
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
pgstrom_open_rowid_map(Relation base_rel, LOCKMODE lockmode)
{
	Relation	relation;

	relation = pgstrom_open_shadow_relation(base_rel, InvalidAttrNumber,
											lockmode, false);
	if (RelationGetForm(relation)->relkind != RELKIND_RELATION)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not a table",
						RelationGetRelationName(relation))));
	return relation;
}

Relation
pgstrom_open_cs_table(Relation base_rel, AttrNumber attno, LOCKMODE lockmode)
{
	Relation	relation;

	relation = pgstrom_open_shadow_relation(base_rel, attno,
											lockmode, false);
	if (RelationGetForm(relation)->relkind != RELKIND_RELATION)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not a table",
						RelationGetRelationName(relation))));
	return relation;
}

Relation
pgstrom_open_cs_index(Relation base, AttrNumber attno, LOCKMODE lockmode)
{
	Relation	relation;

	relation = pgstrom_open_shadow_relation(base, attno,
											lockmode, true);
	if (RelationGetForm(relation)->relkind != RELKIND_INDEX)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an index",
						RelationGetRelationName(relation))));
	return relation;
}

RangeVar *
pgstrom_lookup_sequence(Oid base_relid)
{
	Oid			nsp_oid;
	char	   *nsp_name;
	char	   *rel_name;
	char		seq_name[NAMEDATALEN * 2 + 20];
	RangeVar   *range;

	nsp_oid = get_rel_namespace(base_relid);
	nsp_name = get_namespace_name(nsp_oid);
	rel_name = get_rel_name(base_relid);

	snprintf(seq_name, sizeof(seq_name), "%s.%s.seq",
			 nsp_name, rel_name);
	range = makeRangeVar(PGSTROM_SCHEMA_NAME, pstrdup(seq_name), -1);

	pfree(nsp_name);
	pfree(rel_name);

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
	idx_info->ii_KeyAttrNumbers[0]
		= (!attr ? Anum_pg_strom_rmap_rowid : Anum_pg_strom_cs_rowid);
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
				 false,				/* concurrent */
				 true);				/* is_internal */
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
		tupdesc = CreateTemplateTupleDesc(Natts_pg_strom_rmap, false);
		TupleDescInitEntry(tupdesc, Anum_pg_strom_rmap_rowid,
						   "rowid",  INT8OID,  -1, 0);
		TupleDescInitEntry(tupdesc, Anum_pg_strom_rmap_nitems,
						   "nitems", INT4OID,  -1, 0);
		TupleDescInitEntry(tupdesc, Anum_pg_strom_rmap_rowmap,
						   "rowmap", BYTEAOID, -1, 0);
		tupdesc->attrs[Anum_pg_strom_rmap_rowmap - 1]->attstorage = 'p';
	}
	else
	{
		tupdesc = CreateTemplateTupleDesc(Natts_pg_strom_cs, false);
		TupleDescInitEntry(tupdesc, Anum_pg_strom_cs_rowid,
						   "rowid",  INT8OID,  -1, 0);
		TupleDescInitEntry(tupdesc, Anum_pg_strom_cs_nitems,
						   "nitems", INT4OID,  -1, 0);
		TupleDescInitEntry(tupdesc, Anum_pg_strom_cs_isnull,
						   "isnull", BYTEAOID, -1, 0);
		TupleDescInitEntry(tupdesc, Anum_pg_strom_cs_values,
						   "valued", BYTEAOID, -1, 0);
		/*
		 * PG-Strom wants to keep varlena value being inlined, and never
		 * uses external toast relation due to the performance reason.
		 * So, we override the default setting of type definitions.
		 */
		tupdesc->attrs[Anum_pg_strom_cs_isnull - 1]->attstorage = 'p';
		tupdesc->attrs[Anum_pg_strom_cs_values - 1]->attstorage = 'p';
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
										false,
										true);
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
	seq_stmt->options = list_make4(
		makeDefElem("increment", (Node *)makeInteger(PGSTROM_CHUNK_SIZE)),
		makeDefElem("minvalue",  (Node *)makeInteger(0)),
		makeDefElem("maxvalue",  (Node *)makeInteger((1UL<<48) - 1)),
		makeDefElem("owned_by",  (Node *)rowid_namelist));
	seq_stmt->ownerId = RelationGetForm(base_rel)->relowner;

	DefineSequence(seq_stmt);
}

static void
pgstrom_post_create_foreign_table(CreateForeignTableStmt *stmt)
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
									  BOOTSTRAP_SUPERUSERID, false);
		CommandCounterIncrement();
	}

	/*
	 * Open the base relation; exclusive lock should be already
	 * acquired, so we use NoLock instead.
	 */
	base_rel = heap_openrv(stmt->base.relation, NoLock);

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

/*
 * pgstrom_shadow_relation_rename
 *
 * It is internal utility function to rename shadow table, index and
 * sequence according to changes on base relation.
 */
static void
pgstrom_shadow_relation_rename(const char *old_nspname,
							   const char *old_relname,
							   const char *old_attname,
							   const char *new_nspname,
							   const char *new_relname,
							   const char *new_attname,
							   char relkind)
{
	Oid		namespace_id;
	Oid		shadow_relid;
	char	old_name[NAMEDATALEN * 3 + 20];
	char	new_name[NAMEDATALEN * 3 + 20];

	Assert((old_attname != NULL && new_attname != NULL) ||
		   (old_attname == NULL && new_attname == NULL));
	Assert(old_attname == NULL || relkind != RELKIND_SEQUENCE);
	Assert(new_attname == NULL || relkind != RELKIND_SEQUENCE);

	namespace_id = get_namespace_oid(PGSTROM_SCHEMA_NAME, false);

	if (old_attname)
		snprintf(old_name, sizeof(old_name), "%s.%s.%s.%s",
				 old_nspname, old_relname, old_attname,
				 (relkind == RELKIND_RELATION ? "cs" : "idx"));
	else
		snprintf(old_name, sizeof(old_name), "%s.%s.%s",
				 old_nspname, old_relname,
				 (relkind == RELKIND_RELATION ? "rowid" :
				  (relkind == RELKIND_INDEX ? "idx" : "seq")));

	shadow_relid = get_relname_relid(old_name, namespace_id);
	if (!OidIsValid(shadow_relid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_TABLE),
				 errmsg("relation \"%s.%s\" does not exist",
						PGSTROM_SCHEMA_NAME, old_name)));

	if (new_attname)
		snprintf(new_name, sizeof(new_name), "%s.%s.%s.%s",
				 new_nspname, new_relname, new_attname,
				 (relkind == RELKIND_RELATION ? "cs" : "idx"));
	else
		snprintf(new_name, sizeof(new_name), "%s.%s.%s",
				 new_nspname, new_relname,
				 (relkind == RELKIND_RELATION ? "rowid" :
				  (relkind == RELKIND_INDEX ? "idx" : "seq")));
	if (strlen(new_name) >= NAMEDATALEN)
		ereport(ERROR,
				(errcode(ERRCODE_NAME_TOO_LONG),
				 errmsg("Name of shadow relation \"%s\" too long",
						new_name)));

	RenameRelationInternal(shadow_relid, new_name);
}

/*
 * pgstrom_post_alter_schema
 *
 * It is a post-fixup routine to handle ALTER TABLE ... SET SCHEMA
 * on the foreign table managed by PG-Strom. It replaces the portion
 * of all the underlying shadow relations.
 */
static void
pgstrom_post_alter_schema(AlterObjectSchemaStmt *stmt,
						  Oid base_nspid, Oid base_relid)
{
	char	   *old_nspname;
	char	   *cur_relname;
	Relation	pg_attr;
	ScanKeyData	skey[2];
	SysScanDesc	scan;
	HeapTuple	tuple;

	/* Ensure the changes to base relation being visible */
	CommandCounterIncrement();

	old_nspname = get_namespace_name(base_nspid);
	if (!old_nspname)
		elog(ERROR, "cache lookup failed for schema %u", base_nspid);
	cur_relname = get_rel_name(base_relid);
	if (!cur_relname)
		elog(ERROR, "cache lookup failed for relation %u", base_relid);

	/* Rename rowid table and its index */
	pgstrom_shadow_relation_rename(old_nspname, cur_relname, NULL,
								   stmt->newschema, cur_relname, NULL,
								   RELKIND_RELATION);
	pgstrom_shadow_relation_rename(old_nspname, cur_relname, NULL,
								   stmt->newschema, cur_relname, NULL,
								   RELKIND_INDEX);

	/* Rename column-store and its index */
	pg_attr =  heap_open(AttributeRelationId, AccessShareLock);

	ScanKeyInit(&skey[0],
				Anum_pg_attribute_attrelid,
				BTEqualStrategyNumber, F_OIDEQ,
				ObjectIdGetDatum(base_relid));
	ScanKeyInit(&skey[1],
				Anum_pg_attribute_attnum,
				BTGreaterEqualStrategyNumber, F_INT2GT,
				Int16GetDatum(InvalidAttrNumber));

	scan = systable_beginscan(pg_attr, AttributeRelidNumIndexId, true,
							  SnapshotNow, 2, skey);
	while (HeapTupleIsValid(tuple = systable_getnext(scan)))
	{
		Form_pg_attribute	attr = (Form_pg_attribute) GETSTRUCT(tuple);

		if (attr->attisdropped)
			continue;

		pgstrom_shadow_relation_rename(old_nspname, cur_relname,
									   NameStr(attr->attname),
									   stmt->newschema, cur_relname,
									   NameStr(attr->attname),
									   RELKIND_RELATION);
		pgstrom_shadow_relation_rename(old_nspname, cur_relname,
									   NameStr(attr->attname),
									   stmt->newschema, cur_relname,
									   NameStr(attr->attname),
									   RELKIND_INDEX);
	}
	systable_endscan(scan);
	heap_close(pg_attr, AccessShareLock);

	/* Rename rowid generator sequence */
	pgstrom_shadow_relation_rename(old_nspname, cur_relname, NULL,
								   stmt->newschema, cur_relname, NULL,
								   RELKIND_SEQUENCE);
	/* Make changes visible */
	CommandCounterIncrement();

	pfree(old_nspname);
}

/*
 * pgstrom_post_rename_schema
 *
 * It is a post-fixup routine to handle ALTER SCHEMA ... RENAME TO
 * on the namespace that owning foreign tables managed by PG-Strom.
 * It replaces the portion of all the underlying shadow relations.
 */
static void
pgstrom_post_rename_schema(RenameStmt *stmt, Oid namespaceId)
{
	Relation		pg_class;
	ScanKeyData		skey[2];
	HeapScanDesc	scan;
	HeapTuple		tuple;

	Assert(stmt->renameType == OBJECT_SCHEMA);
	Assert(OidIsValid(namespaceId));

	pg_class = heap_open(RelationRelationId, RowExclusiveLock);

	ScanKeyInit(&skey[0],
				Anum_pg_class_relnamespace,
				BTEqualStrategyNumber, F_OIDEQ,
				ObjectIdGetDatum(namespaceId));
	ScanKeyInit(&skey[1],
				Anum_pg_class_relkind,
				BTEqualStrategyNumber, F_CHAREQ,
				CharGetDatum(RELKIND_FOREIGN_TABLE));

	scan = heap_beginscan(pg_class, SnapshotNow, 2, skey);

	while (HeapTupleIsValid(tuple = heap_getnext(scan, ForwardScanDirection)))
	{
		ForeignTable	   *ft = GetForeignTable(HeapTupleGetOid(tuple));
		ForeignServer	   *fs = GetForeignServer(ft->serverid);
		ForeignDataWrapper *fdw = GetForeignDataWrapper(fs->fdwid);
		const char	   *cur_relname;
		Relation		pg_attr;
		ScanKeyData		akey[2];
		SysScanDesc		ascan;
		HeapTuple		atup;

		if (GetFdwRoutine(fdw->fdwhandler) != &PgStromFdwHandlerData)
			continue;

		cur_relname = NameStr(((Form_pg_class) GETSTRUCT(tuple))->relname);

		/* Rename rowid table and its index */
		pgstrom_shadow_relation_rename(stmt->subname, cur_relname, NULL,
									   stmt->newname, cur_relname, NULL,
									   RELKIND_RELATION);
		pgstrom_shadow_relation_rename(stmt->subname, cur_relname, NULL,
									   stmt->newname, cur_relname, NULL,
									   RELKIND_INDEX);

		/* Rename column-store and its index */
		pg_attr = heap_open(AttributeRelationId, AccessShareLock);

		ScanKeyInit(&akey[0],
					Anum_pg_attribute_attrelid,
					BTEqualStrategyNumber, F_OIDEQ,
					ObjectIdGetDatum(HeapTupleGetOid(tuple)));
		ScanKeyInit(&akey[1],
					Anum_pg_attribute_attnum,
					BTGreaterEqualStrategyNumber, F_INT2GT,
					Int16GetDatum(InvalidAttrNumber));

		ascan = systable_beginscan(pg_attr, AttributeRelidNumIndexId, true,
								   SnapshotSelf, 2, akey);
		while (HeapTupleIsValid(atup = systable_getnext(ascan)))
		{
			Form_pg_attribute	attr = (Form_pg_attribute) GETSTRUCT(atup);

			if (attr->attisdropped)
				continue;

			pgstrom_shadow_relation_rename(stmt->subname, cur_relname,
										   NameStr(attr->attname),
										   stmt->newname, cur_relname,
										   NameStr(attr->attname),
										   RELKIND_RELATION);
			pgstrom_shadow_relation_rename(stmt->subname, cur_relname,
										   NameStr(attr->attname),
										   stmt->newname, cur_relname,
										   NameStr(attr->attname),
										   RELKIND_INDEX);
		}
		systable_endscan(ascan);
		heap_close(pg_attr, AccessShareLock);

		/* Rename rowid generator sequence */
		pgstrom_shadow_relation_rename(stmt->subname, cur_relname, NULL,
									   stmt->newname, cur_relname, NULL,
									   RELKIND_SEQUENCE);
	}
	heap_endscan(scan);
	heap_close(pg_class, RowExclusiveLock);

	/* Make changes visible */
	CommandCounterIncrement();
}

/*
 * pgstrom_post_rename_table
 *
 * It is a post-fixup routine to handle ALTER TABLE ... RENAME TO on
 * the foreign table managed by PG-Strom. It replaces portion of the
 * relation name on all the underlying shadow relations.
 */
static void
pgstrom_post_rename_table(RenameStmt *stmt, Oid base_relid)
{
	char	   *cur_nspname;
	char	   *old_relname;
	Relation	pg_attr;
	ScanKeyData skey[2];
	SysScanDesc scan;
	HeapTuple   tuple;

	Assert(stmt->renameType == OBJECT_FOREIGN_TABLE);
	Assert(stmt->relation->relname != NULL);

	/* Ensure the changes to base relation being visible */
	CommandCounterIncrement();

	cur_nspname = get_namespace_name(get_rel_namespace(base_relid));
	if (!cur_nspname)
		elog(ERROR, "cache lookup failed for schema of table %u", base_relid);
	old_relname = stmt->relation->relname;

	/* Rename rowid table and its index */
	pgstrom_shadow_relation_rename(cur_nspname, old_relname, NULL,
								   cur_nspname, stmt->newname, NULL,
								   RELKIND_RELATION);
	pgstrom_shadow_relation_rename(cur_nspname, old_relname, NULL,
								   cur_nspname, stmt->newname, NULL,
								   RELKIND_INDEX);

	/* Rename column-store and its index */
	pg_attr = heap_open(AttributeRelationId, AccessShareLock);

	ScanKeyInit(&skey[0],
				Anum_pg_attribute_attrelid,
				BTEqualStrategyNumber, F_OIDEQ,
				ObjectIdGetDatum(base_relid));
	ScanKeyInit(&skey[1],
				Anum_pg_attribute_attnum,
				BTGreaterEqualStrategyNumber, F_INT2GT,
				Int16GetDatum(InvalidAttrNumber));

	scan = systable_beginscan(pg_attr, AttributeRelidNumIndexId, true,
							  SnapshotNow, 2, skey);

	while (HeapTupleIsValid(tuple = systable_getnext(scan)))
	{
		Form_pg_attribute	attr = (Form_pg_attribute) GETSTRUCT(tuple);

		if (attr->attisdropped)
			continue;

		pgstrom_shadow_relation_rename(cur_nspname, old_relname,
									   NameStr(attr->attname),
									   cur_nspname, stmt->newname,
									   NameStr(attr->attname),
									   RELKIND_RELATION);
		pgstrom_shadow_relation_rename(cur_nspname, old_relname,
									   NameStr(attr->attname),
									   cur_nspname, stmt->newname,
									   NameStr(attr->attname),
									   RELKIND_INDEX);
	}
	systable_endscan(scan);
	heap_close(pg_attr, AccessShareLock);

	/* Rename rowid generator sequence */
	pgstrom_shadow_relation_rename(cur_nspname, old_relname, NULL,
								   cur_nspname, stmt->newname, NULL,
								   RELKIND_SEQUENCE);
	/* Make changes visible */
	CommandCounterIncrement();

	pfree(cur_nspname);
}

/*
 * pgstrom_post_rename_table
 *
 * It is a post-fixup routine to handle ALTER TABLE ... RENAME TO on
 * column of the foreign table managed by PG-Strom. It replaces portion
 * of the relation name on all the underlying shadow relations.
 */
static void
pgstrom_post_rename_column(RenameStmt *stmt, Oid base_relid)
{
	char	   *cur_nspname;
	char	   *cur_relname;

	Assert(stmt->renameType == OBJECT_COLUMN);
	Assert(stmt->relation->relname != NULL);
	Assert(stmt->subname != NULL);

	/* Ensure the changes to base relation being visible */
	CommandCounterIncrement();

	cur_nspname = get_namespace_name(get_rel_namespace(base_relid));
	if (!cur_nspname)
		elog(ERROR, "cache lookup failed for schema of table %u", base_relid);
	cur_relname = stmt->relation->relname;

	/* Rename column-store and its index */
    pgstrom_shadow_relation_rename(cur_nspname, cur_relname, stmt->subname,
								   cur_nspname, cur_relname, stmt->newname,
								   RELKIND_RELATION);
    pgstrom_shadow_relation_rename(cur_nspname, cur_relname, stmt->subname,
								   cur_nspname, cur_relname, stmt->newname,
								   RELKIND_INDEX);
	/* Make changes visible */
	CommandCounterIncrement();

	pfree(cur_nspname);
}

/*
 * pgstrom_post_change_owner
 *
 * It is a post-fixup routine to handle ALTER TABLE ... OWNER TO on
 * the foreign table managed by PG-Strom. It also changes ownership
 * of the shadow relations according to the new owner of base relation.
 */
static void
pgstrom_post_change_owner(Oid base_relid, AlterTableCmd *cmd,
						  LOCKMODE lockmode)
{
	Oid			namespace_id;
	char	   *base_nspname;
	char	   *base_relname;
	Oid			base_owner;
	char		namebuf[NAMEDATALEN * 3 + 20];
	Oid			shadow_relid;
	Relation	pg_attr;
	ScanKeyData	skey[2];
	SysScanDesc	scan;
	HeapTuple	tuple;
	Form_pg_class	classForm;

	namespace_id = get_namespace_oid(PGSTROM_SCHEMA_NAME, false);

	tuple =  SearchSysCache1(RELOID, ObjectIdGetDatum(base_relid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for relation %u", base_relid);
	classForm = (Form_pg_class) GETSTRUCT(tuple);

	base_nspname = get_namespace_name(classForm->relnamespace);
	base_relname = pstrdup(NameStr(classForm->relname));
	base_owner = classForm->relowner;

	ReleaseSysCache(tuple);

	/*
	 * change owner of rowid table, index and sequence
	 */
	snprintf(namebuf, sizeof(namebuf), "%s.%s.rowid",
			 base_nspname, base_relname);
	shadow_relid = get_relname_relid(namebuf, namespace_id);
	if (!OidIsValid(shadow_relid))
		elog(ERROR, "cache lookup failed for relation \"%s\"", namebuf);
	ATExecChangeOwner(shadow_relid, base_owner, true, lockmode);

	/*
	 * change owner of column-store table and index
	 */
	pg_attr = heap_open(AttributeRelationId, AccessShareLock);

	ScanKeyInit(&skey[0],
				Anum_pg_attribute_attrelid,
				BTEqualStrategyNumber, F_OIDEQ,
				ObjectIdGetDatum(base_relid));
    ScanKeyInit(&skey[1],
                Anum_pg_attribute_attnum,
				BTGreaterEqualStrategyNumber, F_INT2GT,
                Int16GetDatum(InvalidAttrNumber));

	scan = systable_beginscan(pg_attr, AttributeRelidNumIndexId, true,
							  SnapshotNow, 2, skey);
	while (HeapTupleIsValid(tuple = systable_getnext(scan)))
	{
		Form_pg_attribute   attr = (Form_pg_attribute) GETSTRUCT(tuple);

		if (attr->attisdropped)
			continue;

		snprintf(namebuf, sizeof(namebuf), "%s.%s.%s.cs",
				 base_nspname, base_relname, NameStr(attr->attname));
		shadow_relid = get_relname_relid(namebuf, namespace_id);
		if (!OidIsValid(shadow_relid))
			elog(ERROR, "cache lookup failed for relation \"%s\"", namebuf);

		ATExecChangeOwner(shadow_relid, base_owner, true, lockmode);
	}
	systable_endscan(scan);
    heap_close(pg_attr, AccessShareLock);
}

/*
 * pgstrom_post_add_column
 *
 * It is a post-fixup routine to handle ALTER TABLE ... ADD COLUMN on
 * the foreign table managed by PG-Strom. It adds a new shadow column-
 * store and its index.
 */
static void
pgstrom_post_add_column(Oid base_relid, AlterTableCmd *cmd, LOCKMODE lockmode)
{
	Relation	base_rel;
	HeapTuple	tuple;

	base_rel = heap_open(base_relid, lockmode);

	tuple =  SearchSysCache2(ATTNAME,
							 ObjectIdGetDatum(base_relid),
							 CStringGetDatum(cmd->name));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for column \"%s\" of relation \"%s\"",
			 cmd->name, RelationGetRelationName(base_rel));

	pgstrom_create_shadow_table(RelationGetForm(base_rel)->relnamespace,
								base_rel,
								(Form_pg_attribute) GETSTRUCT(tuple));
	ReleaseSysCache(tuple);
	heap_close(base_rel, NoLock);
}

/*
 * pgstrom_post_drop_column
 *
 * It is a post-fixup routine to handle ALTER TABLE ... DROP COLUMN on
 * the foreign table managed by PG-Strom. It also drops the shadow column-
 * store and its index associated with the removed column.
 */
static void
pgstrom_post_drop_column(Oid base_relid, AlterTableCmd *cmd, LOCKMODE lockmode)
{
	Relation	base_rel;
	Oid			base_nspid;
	Oid			namespace_id;
	Oid			shadow_relid;
	char		namebuf[NAMEDATALEN * 2 + 20];

	namespace_id = get_namespace_oid(PGSTROM_SCHEMA_NAME, false);

	base_rel = heap_open(base_relid, lockmode);

	base_nspid = RelationGetForm(base_rel)->relnamespace;

	snprintf(namebuf, sizeof(namebuf), "%s.%s.%s.cs",
			 get_namespace_name(base_nspid),
			 RelationGetRelationName(base_rel),
			 cmd->name);

	shadow_relid = get_relname_relid(namebuf, namespace_id);
	heap_drop_with_catalog(shadow_relid);

	heap_close(base_rel, NoLock);
}

/*
 * pgstrom_post_set_notnull
 *
 * It is a post-fixup routine to handle ALTER TABLE ... SET NOT NULL on
 * column of the foreign table managed by PG-Strom. It checks whether the
 * shadow column-store contains a NULL value, or not.
 * If found, this routine prevent this operation.
 */
static void
pgstrom_post_set_notnull(Oid base_relid, AlterTableCmd *cmd, LOCKMODE lockmode)
{
	elog(ERROR, "Not supported yet");
}

/*
 * pgstrom_post_alter_column_type
 *
 * It is a post-fixup routine to handle ALTER TABLE ... ALTER COLUMN TYPE
 * on column of the foreign table managed by PG-Strom. It re-construct
 * data array of the shadow column-store performing behaind of the modified
 * column.
 */
static void
pgstrom_post_alter_column_type(Oid base_relid, AlterTableCmd *cmd,
							   LOCKMODE lockmode)
{
	elog(ERROR, "Not supported yet");
}

/*
 * pgstrom_process_utility_command
 *
 * Entrypoint of the ProcessUtility hook; that handles post DDL operations.
 */
static void
pgstrom_process_utility_command(Node *parsetree,
								const char *queryString,
								ParamListInfo params,
								DestReceiver *dest,
								char *completionTag,
								ProcessUtilityContext context)
{
	ForeignTable	   *ft;
	ForeignServer	   *fs;
	ForeignDataWrapper *fdw;
	Oid			base_nspid = InvalidOid;
	Oid			base_relid = InvalidOid;
	LOCKMODE	lockmode = NoLock;

	/*
	 * XXX - Preparation of ProcessUtility; Some of operation will changes
	 * name of the foreign table being altered and managed by PG-Strom, so
	 * we open the relation prior to update of system catalog.
	 */
	switch (nodeTag(parsetree))
	{
		case T_AlterObjectSchemaStmt:
			{
				AlterObjectSchemaStmt *stmt
					= (AlterObjectSchemaStmt *)parsetree;
				if (stmt->objectType == OBJECT_FOREIGN_TABLE)
				{
					base_relid = RangeVarGetRelid(stmt->relation,
												  AccessExclusiveLock,
												  true);
					base_nspid = get_rel_namespace(base_relid);
				}
			}
			break;

		case T_RenameStmt:
			{
				RenameStmt *stmt = (RenameStmt *)parsetree;
				if (stmt->renameType == OBJECT_SCHEMA)
					base_nspid = get_namespace_oid(stmt->subname, true);
				else if (stmt->renameType == OBJECT_FOREIGN_TABLE ||
						 stmt->renameType == OBJECT_COLUMN)
					base_relid = RangeVarGetRelid(stmt->relation,
												  AccessExclusiveLock,
												  true);
			}
			break;

		case T_AlterTableStmt:
			{
				AlterTableStmt *stmt = (AlterTableStmt *)parsetree;

				lockmode = AlterTableGetLockLevel(stmt->cmds);
				base_relid = RangeVarGetRelid(stmt->relation, lockmode, true);
			}
			break;

		default:
			/* no preparations for other commands */
			break;
	}

	/*
	 * Call the original ProcessUtility
	 */
	if (next_process_utility_hook)
		(*next_process_utility_hook)(parsetree, queryString, params,
									 dest, completionTag, context);
	else
		standard_ProcessUtility(parsetree, queryString, params,
								dest, completionTag, context);

	/*
	 * Post ProcessUtility Stuffs
	 */
	switch (nodeTag(parsetree))
	{
		case T_CreateForeignTableStmt:
			{
				CreateForeignTableStmt *stmt
					= (CreateForeignTableStmt *)parsetree;

				/* Is a foreign table managed by PG-Strom? */
				fs = GetForeignServerByName(stmt->servername, false);
				fdw = GetForeignDataWrapper(fs->fdwid);
				if (GetFdwRoutine(fdw->fdwhandler) == &PgStromFdwHandlerData)
					pgstrom_post_create_foreign_table(stmt);
			}
			break;

		case T_AlterObjectSchemaStmt:
			{
				AlterObjectSchemaStmt *stmt
					= (AlterObjectSchemaStmt *)parsetree;

				if (!OidIsValid(base_relid) ||
					get_rel_relkind(base_relid) != RELKIND_FOREIGN_TABLE)
					break;

				ft = GetForeignTable(base_relid);
				fs = GetForeignServer(ft->serverid);
				fdw = GetForeignDataWrapper(fs->fdwid);
				if (GetFdwRoutine(fdw->fdwhandler) == &PgStromFdwHandlerData)
					pgstrom_post_alter_schema(stmt, base_nspid, base_relid);
			}
			break;

		case T_RenameStmt:
			{
				RenameStmt *stmt = (RenameStmt *)parsetree;

				if (OidIsValid(base_nspid))
					pgstrom_post_rename_schema(stmt, base_nspid);
				else if (OidIsValid(base_relid) &&
						 get_rel_relkind(base_relid) == RELKIND_FOREIGN_TABLE)
				{
					ft = GetForeignTable(base_relid);
					fs = GetForeignServer(ft->serverid);
					fdw = GetForeignDataWrapper(fs->fdwid);
					if (GetFdwRoutine(fdw->fdwhandler)==&PgStromFdwHandlerData)
					{
						if (stmt->renameType == OBJECT_FOREIGN_TABLE)
							pgstrom_post_rename_table(stmt, base_relid);
						else
							pgstrom_post_rename_column(stmt, base_relid);
					}
				}
			}
			break;

		case T_AlterTableStmt:
			{
				AlterTableStmt *stmt = (AlterTableStmt *)parsetree;
				ListCell	   *cell;

				if (!OidIsValid(base_relid) ||
					get_rel_relkind(base_relid) != RELKIND_FOREIGN_TABLE)
					break;

				ft = GetForeignTable(base_relid);
				fs = GetForeignServer(ft->serverid);
				fdw = GetForeignDataWrapper(fs->fdwid);
				if (GetFdwRoutine(fdw->fdwhandler) != &PgStromFdwHandlerData)
					break;

				foreach (cell, stmt->cmds)
				{
					AlterTableCmd *cmd = lfirst(cell);

					switch (cmd->subtype)
					{
						case AT_ChangeOwner:
							pgstrom_post_change_owner(base_relid, cmd,
													  lockmode);
							break;

						case AT_AddColumn:
							pgstrom_post_add_column(base_relid, cmd,
													lockmode);
							break;

						case AT_DropColumn:
							pgstrom_post_drop_column(base_relid, cmd,
													 lockmode);
							break;

						case AT_SetNotNull:
							pgstrom_post_set_notnull(base_relid, cmd,
													 lockmode);
							break;

						case AT_AlterColumnType:
							pgstrom_post_alter_column_type(base_relid,
														   cmd, lockmode);
							break;
						default:
							/* do nothing elsewhere */
							break;
					}
				}
			}
			break;

		default:
			/* do nothing */
			break;
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
