/*
 * scan.c
 *
 * Routines to scan column based data store with stream processing
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
#include "catalog/namespace.h"
#include "catalog/pg_class.h"
#include "foreign/foreign.h"
#include "nodes/makefuncs.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "pg_strom.h"

RelationSet
pgstrom_open_relation_set(Oid base_relid, LOCKMODE lockmode)
{
	RelationSet	relset;
	Relation	base_rel;
	AttrNumber	i, nattrs;
	RangeVar   *range;
	char	   *base_schema;
	char		namebuf[NAMEDATALEN * 3 + 20];

	/* Open the base relation */
	base_rel = relation_open(base_relid, lockmode);

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
		ForeignTable	   *ft = GetForeignTable(base_relid);
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
	relset->base_rel = base_rel;
	relset->column_rel = palloc0(sizeof(Relation) * nattrs);
	relset->column_idx = palloc0(sizeof(CatalogIndexState) * nattrs);

	/*
	 * Open the underlying tables and corresponding indexes
	 */
	range = makeRangeVar(PGSTROM_SCHEMA_NAME, namebuf, -1);
	base_schema = get_namespace_name(RelationGetForm(base_rel)->relnamespace);

	snprintf(namebuf, sizeof(namebuf), "%s.%s.usemap",
			 base_schema, RelationGetRelationName(base_rel));
	relset->usemap_rel = relation_openrv(range, lockmode);
	relset->usemap_idx = CatalogOpenIndexes(relset->usemap_rel);

	for (i = 0; i < nattrs; i++)
	{
		Form_pg_attribute attr = RelationGetDescr(base_rel)->attrs[i];

		if (attr->attisdropped)
			continue;

		snprintf(namebuf, sizeof(namebuf), "%s.%s.%s.col",
				 base_schema,
				 RelationGetRelationName(base_rel),
				 NameStr(attr->attname));

		relset->column_rel[i] = relation_openrv(range, lockmode);
		relset->column_idx[i] = CatalogOpenIndexes(relset->column_rel[i]);
	}

	/*
	 * Also, solve the sequence name
	 */
	snprintf(namebuf, sizeof(namebuf), "%s.%s.seq",
			 base_schema, RelationGetRelationName(base_rel));
	relset->sequence_id = RangeVarGetRelid(range, NoLock, false, false);

	return relset;
}

void
pgstrom_close_relation_set(RelationSet relset, LOCKMODE lockmode)
{
	AttrNumber	i, nattrs = RelationGetNumberOfAttributes(relset->base_rel);

	CatalogCloseIndexes(relset->usemap_idx);
	relation_close(relset->usemap_rel, lockmode);

	for (i=0; i < nattrs; i++)
	{
		if (!relset->column_rel[i])
			continue;

		CatalogCloseIndexes(relset->column_idx[i]);
		relation_close(relset->column_rel[i], lockmode);
	}
	relation_close(relset->base_rel, lockmode);

	pfree(relset->column_rel);
	pfree(relset->column_idx);
	pfree(relset);
}

void
pgstrom_begin_foreign_scan(ForeignScanState *fss, int eflags)
{
}

TupleTableSlot*
pgstrom_iterate_foreign_scan(ForeignScanState *fss)
{
	return NULL;
}

void
pgboost_rescan_foreign_scan(ForeignScanState *fss)
{
}

void
pgboost_end_foreign_scan(ForeignScanState *fss)
{
}
