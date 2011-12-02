/*
 * blkload.c
 *
 * Routines to load data from regular table to pg_strom store
 *
 * --
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the 'LICENSE' included within
 * this package.
 */
#include "postgres.h"
#include "access/sysattr.h"
#include "access/tuptoaster.h"
#include "access/xact.h"
#include "catalog/indexing.h"
#include "catalog/namespace.h"
#include "catalog/pg_attribute.h"
#include "catalog/pg_authid.h"
#include "catalog/pg_class.h"
#include "commands/sequence.h"
#include "foreign/foreign.h"
#include "miscadmin.h"
#include "nodes/makefuncs.h"
#include "utils/errcodes.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/syscache.h"
#include "utils/tqual.h"
#include "utils/varbit.h"

#include "pg_strom.h"

typedef struct
{
	Relation	base_rel;
	Relation	usemap_rel;
	Relation   *column_rel;
	CatalogIndexState	usemap_idx;
	CatalogIndexState  *column_idx;
	Oid			rowid_seq;
} relation_bunch_t;

relation_bunch_t *
pgstrom_relation_bunch_open(Oid base_relid, LOCKMODE lockmode)
{
	relation_bunch_t   *relbunch;
	Relation			base_rel;
	ForeignTable	   *ft;
	ForeignServer	   *fs;
	ForeignDataWrapper *fdw;
	AttrNumber			i, nattrs;
	RangeVar		   *range;
	char			   *base_schema;
	char				namebuf[NAMEDATALEN * 3 + 20];

	/*
	 * Open the base relation
	 */
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

	ft = GetForeignTable(RelationGetRelid(base_rel));
	fs = GetForeignServer(ft->serverid);
	fdw = GetForeignDataWrapper(fs->fdwid);
	if (GetFdwRoutine(fdw->fdwhandler) != &pgstromFdwHandlerData)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not a foreign table managed by pg_strom",
						RelationGetRelationName(base_rel))));

	/*
	 * Setting up relation_bunch_t object
	 */
	relbunch = palloc(sizeof(relation_bunch_t));
	relbunch->base_rel = base_rel;
	nattrs = RelationGetNumberOfAttributes(relbunch->base_rel);
	relbunch->column_rel = palloc0(sizeof(Relation) * nattrs);
	relbunch->column_idx = palloc0(sizeof(CatalogIndexState) * nattrs);

	range = makeRangeVar(PGSTROM_SCHEMA_NAME, namebuf, -1);
	base_schema = get_namespace_name(RelationGetForm(base_rel)->relnamespace);

	/*
	 * Open the underlying tables and corresponding indexes.
	 */
	snprintf(namebuf, sizeof(namebuf), "%s.%s.usemap",
			 base_schema, RelationGetRelationName(base_rel));
	relbunch->usemap_rel = relation_openrv(range, lockmode);
	relbunch->usemap_idx = CatalogOpenIndexes(relbunch->usemap_rel);

	for (i = 0; i < nattrs; i++)
	{
		Form_pg_attribute	attr = RelationGetDescr(base_rel)->attrs[i];

		if (attr->attisdropped)
			continue;

		snprintf(namebuf, sizeof(namebuf), "%s.%s.%s.col",
				 base_schema,
				 RelationGetRelationName(base_rel),
				 NameStr(attr->attname));

		relbunch->column_rel[i] = relation_openrv(range, lockmode);
		relbunch->column_idx[i] = CatalogOpenIndexes(relbunch->column_rel[i]);
	}

	/*
	 * Also, solve the sequence name
	 */
	snprintf(namebuf, sizeof(namebuf), "%s.%s.seq",
			 base_schema, RelationGetRelationName(base_rel));
	relbunch->rowid_seq = RangeVarGetRelid(range, NoLock, false, false);

	return relbunch;
}

void
pgstrom_relation_bunch_close(relation_bunch_t *relbunch, LOCKMODE lockmode)
{
	AttrNumber	i, nattrs = RelationGetNumberOfAttributes(relbunch->base_rel);

	CatalogCloseIndexes(relbunch->usemap_idx);
	relation_close(relbunch->usemap_rel, lockmode);

	for (i = 0; i < nattrs; i++)
	{
		if (!relbunch->column_rel[i])
			continue;

		CatalogCloseIndexes(relbunch->column_idx[i]);
		relation_close(relbunch->column_rel[i], lockmode);
	}
	relation_close(relbunch->base_rel, lockmode);

	pfree(relbunch->column_rel);
	pfree(relbunch->column_idx);
	pfree(relbunch);
}

static void
pgstrom_relation_bunch_reset(relation_bunch_t *relbunch)
{
	HeapScanDesc	scan;
	HeapTuple		tup;
	AttrNumber		i, nattrs;
	Relation		seq_rel;

	/* reset any contents of usemap table */
	scan = heap_beginscan(relbunch->usemap_rel,
						  SnapshotNow, 0, NULL);
	while (HeapTupleIsValid(tup = heap_getnext(scan,
											   ForwardScanDirection)))
	{
		simple_heap_delete(relbunch->usemap_rel, &tup->t_self);
		CatalogIndexInsert(relbunch->usemap_idx, tup);
	}
	heap_endscan(scan);

	/* reset any contents of column store */
	nattrs = RelationGetNumberOfAttributes(relbunch->base_rel);
	for (i=0; i < nattrs; i++)
	{
		scan = heap_beginscan(relbunch->column_rel[i],
							  SnapshotNow, 0, NULL);
		while (HeapTupleIsValid(tup = heap_getnext(scan,
												   ForwardScanDirection)))
		{
			if (!relbunch->column_rel[i])
				continue;

			simple_heap_delete(relbunch->column_rel[i], &tup->t_self);
			CatalogIndexInsert(relbunch->column_idx[i], tup);
		}
		heap_endscan(scan);
	}

	/* reset sequence generator of rowid */
	seq_rel = relation_open(relbunch->rowid_seq, AccessExclusiveLock);
	ResetSequence(RelationGetRelid(seq_rel));
	relation_close(seq_rel, NoLock);

	/* make changes visible */
	CommandCounterIncrement();
}

static void
pgstrom_cs_write_usemap(Relation usemap_rel,
						CatalogIndexState usemap_idx,
						int nitems,
						int64 rowid, VarBit *cs_usemap)
{
	HeapTuple	tuple;
	Datum		values[2];
	bool		nulls[2];
	Datum		temp;

	memset(values, 0, sizeof(values));
	memset(nulls, false, sizeof(nulls));

	values[0] = Int64GetDatum(rowid);

	VARBITLEN(cs_usemap) = nitems;
	SET_VARSIZE(cs_usemap, VARBITTOTALLEN(nitems));
	values[1] = PointerGetDatum(cs_usemap);
	temp = toast_compress_datum(values[1]);
	if (DatumGetPointer(temp))
		values[1] = temp;

	tuple = heap_form_tuple(RelationGetDescr(usemap_rel), values, nulls);
	simple_heap_insert(usemap_rel, tuple);
	CatalogIndexInsert(usemap_idx, tuple);

	heap_freetuple(tuple);
	if (DatumGetPointer(temp))
		pfree(DatumGetPointer(temp));
}

static void
pgstrom_cs_write_varlena(Relation column_rel,
						 CatalogIndexState column_idx,
						 int64 cs_rowid, Datum cs_value)
{
	HeapTuple	tuple;
	Datum		values[3];
	bool		nulls[3];

	memset(values, false, sizeof(values));
	memset(nulls, 0, sizeof(nulls));

	values[0] = Int64GetDatum(cs_rowid);
	nulls[1] = true;
	values[2] = cs_value;

	tuple = heap_form_tuple(RelationGetDescr(column_rel), values, nulls);

	simple_heap_insert(column_rel, tuple);
	CatalogIndexInsert(column_idx, tuple);

	heap_freetuple(tuple);
}

static void
pgstrom_cs_write_chunk(Relation column_rel,
					   CatalogIndexState column_idx,
					   int nitems, int16 unitsz,
					   int64 cs_rowid,
					   VarBit *cs_nulls, bytea *cs_values)
{
	HeapTuple	tuple;
	Datum		values[3];
	bool		nulls[3];
	Datum		temp1;
	Datum		temp2;

	memset(values, 0, sizeof(cs_values));
	memset(nulls, false, sizeof(cs_nulls));

	/* "rowid" of column store  */
	values[0] = Int64GetDatum(cs_rowid);

	/* "nulls" of column store */
	VARBITLEN(cs_nulls) = nitems;
	SET_VARSIZE(cs_nulls, VARBITTOTALLEN(nitems));
	values[1] = PointerGetDatum(cs_nulls);
	temp1 = toast_compress_datum(values[1]);
	if (PointerGetDatum(temp1))
		values[1] = temp1;

	/* "values" of column store */
	SET_VARSIZE(cs_values, VARHDRSZ + nitems * unitsz);
	values[2] = PointerGetDatum(cs_values);
	temp2 = toast_compress_datum(values[2]);
	if (PointerGetDatum(temp2))
		values[2] = temp2;

	tuple = heap_form_tuple(RelationGetDescr(column_rel), values, nulls);

	simple_heap_insert(column_rel, tuple);
	CatalogUpdateIndexes(column_rel, tuple);

	heap_freetuple(tuple);
	if (DatumGetPointer(temp1))
		pfree(DatumGetPointer(temp1));
	if (DatumGetPointer(temp2))
		pfree(DatumGetPointer(temp2));
}

static void
pgstrom_relation_bunch_blkload(relation_bunch_t *relbunch,
							   Relation source_rel, AttrNumber *references)
{
	VarBit		   *cs_usemap;
	bytea		  **cs_values;
	VarBit		  **cs_nulls;
	int16		   *cs_unitsz;
	int			   *cs_nitems;
	bool		   *cs_attbyval;
	bool		   *cs_buffered;
	AttrNumber		i, j, nattrs;
	HeapScanDesc	scan;
	HeapTuple		tuple;
	Datum		   *values;
	bool		   *nulls;
	int64			rowid;
	int				index;

	/*
	 * Set up buffer of column store.
	 */
	nattrs = RelationGetNumberOfAttributes(relbunch->base_rel);
	cs_usemap   = palloc0(VARBITTOTALLEN(PGSTROM_CHUNK_SIZE));
	cs_values   = palloc0(nattrs * sizeof(bytea *));
	cs_nulls    = palloc0(nattrs * sizeof(VarBit *));
	cs_unitsz   = palloc0(nattrs * sizeof(int16));
	cs_nitems   = palloc0(nattrs * sizeof(int));
	cs_attbyval = palloc0(nattrs * sizeof(bool));
	cs_buffered = palloc0(nattrs * sizeof(bool));

	for (j=0; j < nattrs; j++)
	{
		Form_pg_attribute	attr
			= RelationGetDescr(relbunch->base_rel)->attrs[j];

		cs_unitsz[j] = attr->attlen;
		if (cs_unitsz[j] > 0)
		{
			cs_nitems[j] = PGSTROM_CHUNK_SIZE / cs_unitsz[j];
			cs_values[j] = palloc0(VARHDRSZ + PGSTROM_CHUNK_SIZE);
			cs_nulls[j] = palloc0(VARBITTOTALLEN(cs_nitems[j]));
			cs_attbyval[j] = attr->attbyval;
			cs_buffered[j] = false;
		}
	}

	/*
	 * Scan all the tuples from source relation
	 */
	nattrs = RelationGetNumberOfAttributes(source_rel);
	values = palloc(sizeof(Datum) * nattrs);
	nulls = palloc(sizeof(bool) * nattrs);

	scan = heap_beginscan(source_rel, SnapshotNow, 0, NULL);
	for (rowid = 0;
		 HeapTupleIsValid(tuple = heap_getnext(scan, ForwardScanDirection));
		 rowid++)
	{
		/*
		 * If rowid exceeds boundary of the chunk, we reference sequence
		 * object to generate rowid again.
		 */
		if ((rowid % PGSTROM_CHUNK_SIZE) == 0)
		{
			Oid		save_userid;
			int		save_sec_context;
			Datum	temp;

			GetUserIdAndSecContext(&save_userid, &save_sec_context);
			SetUserIdAndSecContext(BOOTSTRAP_SUPERUSERID,
					save_sec_context | SECURITY_LOCAL_USERID_CHANGE);

			temp = DirectFunctionCall1(nextval_oid,
							ObjectIdGetDatum(relbunch->rowid_seq));

			SetUserIdAndSecContext(save_userid, save_sec_context);

			rowid = DatumGetInt64(temp);
		}

		/*
		 * Mark this rowid is in-use.
		 */
		index = (rowid % PGSTROM_CHUNK_SIZE);
		VARBITS(cs_usemap)[index >> 3] |= (1 << (index & 7));

		heap_deform_tuple(tuple, RelationGetDescr(source_rel), values, nulls);

		for (i=0; i < RelationGetNumberOfAttributes(source_rel); i++)
		{
			Assert(references[i] > 0 && references[i] <= nattrs);
			j = references[i] - 1;

			/*
			 * varlena data shall be stored for each rows.
			 */
			if (cs_unitsz[j] < 1)
			{
				if (!nulls[i])
					pgstrom_cs_write_varlena(relbunch->column_rel[j],
											 relbunch->column_idx[j],
											 rowid, values[i]);
				continue;
			}

			/*
			 * Put fetched value on the buffer of column store as an element
			 * of array tentatively.
			 */
			index = (rowid % PGSTROM_CHUNK_SIZE) % cs_nitems[j];
			if (!nulls[i])
			{
				memcpy(VARDATA(cs_values[j]) + index * cs_unitsz[j],
					   (cs_attbyval[j] ?
						(Pointer)&values[i] :
						DatumGetPointer(values[i])),
					   cs_unitsz[j]);
			}
			else
			{
				VARBITS(cs_nulls[j])[index >> 3] |= (1 << (index & 7));
			}
			cs_buffered[j] = true;

			/*
			 * The current buffer shall be written out to the column store
			 * when the number of items exceeds limitation of array, or
			 * number of chunks, because we don't allow a particular chunk
			 * goes across boundary of the chunk.
			 */
			if (index + 1 == cs_nitems[j] ||
				rowid % PGSTROM_CHUNK_SIZE == PGSTROM_CHUNK_SIZE - 1)
			{
				pgstrom_cs_write_chunk(relbunch->column_rel[j],
									   relbunch->column_idx[j],
									   index + 1, cs_unitsz[j],
									   rowid - index,
									   cs_nulls[j], cs_values[j]);
				memset(cs_values[j], 0, VARHDRSZ + PGSTROM_CHUNK_SIZE);
				memset(cs_nulls[j], 0, VARBITTOTALLEN(cs_nitems[j]));
				cs_buffered[j] = false;
			}
		}

		/*
		 * Also write back to the usemap on the boundary of this chunk.
		 */
		if (rowid % PGSTROM_CHUNK_SIZE == PGSTROM_CHUNK_SIZE - 1)
		{
			pgstrom_cs_write_usemap(relbunch->usemap_rel,
									relbunch->usemap_idx,
									PGSTROM_CHUNK_SIZE,
									rowid - (PGSTROM_CHUNK_SIZE - 1),
									cs_usemap);
			memset(cs_usemap, 0, VARBITTOTALLEN(PGSTROM_CHUNK_SIZE));
		}
	}

	/*
	 * If some values are remained on the buffer, we weite out them.
	 */
	if (rowid % PGSTROM_CHUNK_SIZE != 0)
	{
		nattrs = RelationGetNumberOfAttributes(relbunch->base_rel);
		for (j=0; j < nattrs; j++)
		{
			if (!cs_buffered[j])
				continue;

			index = (rowid % PGSTROM_CHUNK_SIZE) % cs_nitems[j];
			if (index > 0)
				pgstrom_cs_write_chunk(relbunch->column_rel[j],
									   relbunch->column_idx[j],
									   index, cs_unitsz[j],
									   rowid - index,
									   cs_nulls[j], cs_values[j]);
		}
		index = (rowid % PGSTROM_CHUNK_SIZE);
		pgstrom_cs_write_usemap(relbunch->usemap_rel,
								relbunch->usemap_idx,
								index,
								rowid - index, cs_usemap);
	}
	heap_endscan(scan);
}

Datum
pgstrom_blkload(PG_FUNCTION_ARGS)
{
	Oid		source_relid = PG_GETARG_OID(0);
	Oid		dest_relid = PG_GETARG_OID(1);
	Relation			source_rel;
	relation_bunch_t   *relbunch;
	RangeTblEntry	   *srte;
	RangeTblEntry	   *drte;
	AttrNumber			i, nattrs;
	AttrNumber		   *references;

	/*
	 * Open and check adequacy of the source relation
	 */
	source_rel = relation_open(source_relid, AccessShareLock);
	if (RelationGetForm(source_rel)->relkind != RELKIND_RELATION)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("%s is not a regular table",
						RelationGetRelationName(source_rel))));

	/*
	 * Open the bunch of destination relation
	 */
	relbunch = pgstrom_relation_bunch_open(dest_relid, ExclusiveLock);

	/*
	 * Set up RangeTblEntry for permission checks
	 */
	srte = makeNode(RangeTblEntry);
	srte->rtekind = RTE_RELATION;
	srte->relid = RelationGetRelid(source_rel);
	srte->relkind = RelationGetForm(source_rel)->relkind;
	srte->requiredPerms = ACL_SELECT;

	drte = makeNode(RangeTblEntry);
	drte->rtekind = RTE_RELATION;
	drte->relid = RelationGetRelid(relbunch->base_rel);
	drte->relkind = RelationGetForm(relbunch->base_rel)->relkind;
	drte->requiredPerms = ACL_INSERT | ACL_DELETE;

	/*
	 * Any columns of the source relation must exist on the destination
	 * relation with same data type.
	 *
	 * TODO: we should allow implicit cast.
	 */
	nattrs = RelationGetNumberOfAttributes(source_rel);
	references = palloc0(sizeof(AttrNumber) * nattrs);
	for (i=0; i < nattrs; i++)
	{
		Form_pg_attribute	attr1 = RelationGetDescr(source_rel)->attrs[i];
		Form_pg_attribute	attr2;
		HeapTuple			tuple;

		if (attr1->attisdropped)
			continue;

		tuple = SearchSysCacheAttName(RelationGetRelid(source_rel),
									  NameStr(attr1->attname));
		if (!HeapTupleIsValid(tuple))
			ereport(ERROR,
					(errcode(ERRCODE_FDW_INVALID_COLUMN_NAME),
					 errmsg("column \"%s\" of relation \"%s\" did not exist on the destination relation \"%s\"",
							NameStr(attr1->attname),
							RelationGetRelationName(source_rel),
							RelationGetRelationName(relbunch->base_rel))));

		attr2 = (Form_pg_attribute) GETSTRUCT(tuple);
		if (attr1->atttypid != attr2->atttypid ||
			attr1->attlen != attr2->attlen ||
			attr1->attndims != attr2->attndims ||
			attr1->attbyval != attr2->attbyval)
			ereport(ERROR,
					(errcode(ERRCODE_FDW_INVALID_DATA_TYPE),
					 errmsg("column \"%s\" of relation \"%s\" does not have compatible data type with column \"%s\" of relation \"%s\"",
							NameStr(attr1->attname),
							RelationGetRelationName(source_rel),
							NameStr(attr2->attname),
							RelationGetRelationName(relbunch->base_rel))));

		references[i] = attr2->attnum;
		srte->selectedCols = bms_add_member(srte->selectedCols,
				attr1->attnum - FirstLowInvalidHeapAttributeNumber);
		drte->modifiedCols = bms_add_member(drte->modifiedCols,
				attr2->attnum - FirstLowInvalidHeapAttributeNumber);
	}

	/*
	 * Permission checks
	 */
	ExecCheckRTPerms(list_make2(srte, drte), true);

	/*
	 * Remove existing data
	 */
	pgstrom_relation_bunch_reset(relbunch);

	/*
	 * Load data
	 */
	pgstrom_relation_bunch_blkload(relbunch, source_rel, references);

	/*
	 * Close the source and destination relations. Table locks on
	 * the destination side shall be remained by end of the transaction.
	 */
	pgstrom_relation_bunch_close(relbunch, NoLock);

	relation_close(source_rel, AccessShareLock);

	PG_RETURN_BOOL(true);
}
