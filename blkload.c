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
#include "catalog/indexing.h"
#include "catalog/pg_attribute.h"
#include "catalog/pg_authid.h"
#include "catalog/pg_class.h"
#include "commands/sequence.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "utils/array.h"
#include "utils/errcodes.h"
#include "utils/rel.h"
#include "utils/syscache.h"
#include "utils/tqual.h"
#include "utils/varbit.h"

#include "pg_strom.h"


static void
pgstrom_cschunk_insert(RelationSet relset, VarBit *cs_usemap,
					   Datum **cs_values, bool **cs_nulls, int nitems)
{
	int64		rowid;
	TupleDesc	tupdesc;
	HeapTuple	tuple;
	Datum		temp;
	Datum		values[2];
	bool		nulls[2];
	Oid			save_userid;
	int			save_sec_context;
	AttrNumber	csidx, nattrs;

	/*
	 * Acquire a row-id of the head of this chunk
	 */
	GetUserIdAndSecContext(&save_userid, &save_sec_context);
	SetUserIdAndSecContext(BOOTSTRAP_SUPERUSERID,
						   save_sec_context | SECURITY_LOCAL_USERID_CHANGE);
	temp = DirectFunctionCall1(nextval_oid,
							   ObjectIdGetDatum(relset->rowid_seqid));
	rowid = DatumGetInt64(temp);
	SetUserIdAndSecContext(save_userid, save_sec_context);

	/*
	 * Insert usemap of this chunk
	 */
	memset(values, 0, sizeof(values));
	memset(nulls, false, sizeof(nulls));

	values[0] = Int64GetDatum(rowid);
	VARBITLEN(cs_usemap) = PGSTROM_CHUNK_SIZE;
	SET_VARSIZE(cs_usemap, VARBITTOTALLEN(PGSTROM_CHUNK_SIZE));
	values[1] = PointerGetDatum(cs_usemap);
	temp = toast_compress_datum(values[1]);
	if (DatumGetPointer(temp) != NULL)
		values[1] = temp;

	tupdesc = RelationGetDescr(relset->rowid_rel);
	tuple = heap_form_tuple(tupdesc, values, nulls);
	simple_heap_insert(relset->rowid_rel, tuple);
	CatalogUpdateIndexes(relset->rowid_rel, tuple);

	heap_freetuple(tuple);
	if (DatumGetPointer(temp) != NULL)
		pfree(DatumGetPointer(temp));

	/*
	 * Insert into column store
	 */
	nattrs = RelationGetNumberOfAttributes(relset->base_rel);
	for (csidx=0; csidx < nattrs; csidx++)
	{
		Form_pg_attribute	attr
			= RelationGetDescr(relset->base_rel)->attrs[csidx];
		int			rs_base;
		int			rs_unitsz;
		ArrayType  *cs_array;
		int			cs_dims[1];
		int			cs_lbound[1];

		for (rs_base=0; rs_base < nitems; rs_base += rs_unitsz)
		{
			int		index;

			/*
			 * compute length of array; fixed length shall be applied
			 * in the case when attlen > 0; that can be handled by GPU.
			 * otherwise, we try to put varlena data into a single chunk
			 * as much as possible we can.
			 */
			if (attr->attlen > 0)
				rs_unitsz = PGSTROM_CHUNK_SIZE / attr->attlen;
			else
			{
				int		total_length = 0;

				for (rs_unitsz=0; rs_base+rs_unitsz < nitems; rs_unitsz++)
				{
					index = rs_base+rs_unitsz;

					/*
					 * XXX - we may need to pay attention to null-bitmap,
					 * if and when one varlena is null.
					 */
					if (cs_nulls[csidx][index])
						continue;

					/* make sure being uncompressed */
					temp = (Datum)PG_DETOAST_DATUM(cs_values[csidx][index]);
					cs_values[csidx][index] = temp;

					if (total_length + VARSIZE(temp) > PGSTROM_CHUNK_SIZE)
						break;

					total_length += VARSIZE(temp);
				}

				/*
				 * Even if a varlena has its length larger than chunk-size,
				 * data compression may enables to store it into a single
				 * disk block.
				 */
				if (rs_unitsz == 0)
					rs_unitsz = 1;
			}
			if (rs_base + rs_unitsz > nitems)
				rs_unitsz = nitems - rs_base;

			/*
			 * In the case when column-store does not contain any valid
			 * values between [rs_base] and [rs_base + rs_unitsz - 1],
			 * no need to insert an array. Scanner considers these values
			 * are all NULLs.
			 */
			for (index=rs_base; index < rs_base + rs_unitsz; index++)
			{
				if (!cs_nulls[csidx][index])
					break;
			}
			if (index == rs_base + rs_unitsz)
				continue;

			/*
			 * Set up an array
			 */
			Assert(rs_unitsz > 0 && rs_unitsz <= PGSTROM_CHUNK_SIZE);
			cs_dims[0] = rs_unitsz;
			cs_lbound[0] = 0;
			cs_array = construct_md_array(&cs_values[csidx][rs_base],
										  &cs_nulls[csidx][rs_base],
										  1,
										  cs_dims,
										  cs_lbound,
										  attr->atttypid,
										  attr->attlen,
										  attr->attbyval,
										  attr->attalign);
			/*
			 * Insert a tuple with compression
			 */
			memset(nulls, false, sizeof(nulls));
			values[0] = Int64GetDatum(rowid + rs_base);
			values[1] = PointerGetDatum(cs_array);
			temp = toast_compress_datum(values[1]);
			if (DatumGetPointer(temp) != NULL)
				values[1] = temp;

			tupdesc = RelationGetDescr(relset->cs_rel[csidx]);
			tuple = heap_form_tuple(tupdesc, values, nulls);
			simple_heap_insert(relset->cs_rel[csidx], tuple);
			CatalogUpdateIndexes(relset->cs_rel[csidx], tuple);
			heap_freetuple(tuple);

			if (DatumGetPointer(temp) != NULL)
				pfree(DatumGetPointer(temp));
			pfree(cs_array);
		}
	}
}

static void
pgstrom_data_load_internal(RelationSet relset,
						   Relation source, AttrNumber *attmap)
{
	TupleDesc		tupdesc;
	HeapScanDesc	scan;
	HeapTuple		tuple;
	HeapTuple	   *rs_tuples;
	Datum		   *rs_values;
	bool		   *rs_nulls;
	VarBit		   *cs_usemap;
	Datum		  **cs_values;
	bool		  **cs_nulls;
	MemoryContext	cs_memcxt;
	MemoryContext	oldcxt;
	AttrNumber		i, j, nattrs;
	int				index = 0;

	tupdesc = RelationGetDescr(source);
	rs_values = palloc(sizeof(Datum) * tupdesc->natts);
	rs_nulls  = palloc(sizeof(bool)  * tupdesc->natts);

	nattrs = RelationGetNumberOfAttributes(relset->base_rel);
	cs_usemap = palloc0(VARBITTOTALLEN(PGSTROM_CHUNK_SIZE));
	cs_values = palloc(sizeof(Datum *) * nattrs);
	cs_nulls = palloc(sizeof(bool *) * nattrs);
	for (j=0; j < nattrs; j++)
	{
		cs_values[j] = palloc(sizeof(Datum) * PGSTROM_CHUNK_SIZE);
		cs_nulls[j] = palloc(sizeof(bool) * PGSTROM_CHUNK_SIZE);
	}
	rs_tuples = palloc(sizeof(HeapTuple) * PGSTROM_CHUNK_SIZE);


	/*
	 * Create a temp memory context to prevent possible memory leak
	 * on load of a huge table.
	 */
	cs_memcxt = AllocSetContextCreate(CurrentMemoryContext,
									  "Per-chunk memory context",
									  ALLOCSET_DEFAULT_MINSIZE,
									  ALLOCSET_DEFAULT_INITSIZE,
									  ALLOCSET_DEFAULT_MAXSIZE);
	/*
	 * Scan the source relation
	 */
	scan = heap_beginscan(source, SnapshotNow, 0, NULL);

	oldcxt = MemoryContextSwitchTo(cs_memcxt);
	while (HeapTupleIsValid(tuple = heap_getnext(scan, ForwardScanDirection)))
	{
		rs_tuples[index] = heap_copytuple(tuple);
		heap_deform_tuple(rs_tuples[index], tupdesc, rs_values, rs_nulls);

		/* set usemap */
		VARBITS(cs_usemap)[index / BITS_PER_BYTE]
			|= (1 << (index % BITS_PER_BYTE));

		for (i=0; i < tupdesc->natts; i++)
		{
			if ((j = attmap[i] - 1) < 0)
				continue;

			if (rs_nulls[i])
				cs_nulls[j][index] = true;
			else
			{
				cs_nulls[j][index] = false;
				cs_values[j][index] = rs_values[i];
			}
		}
		if (++index == PGSTROM_CHUNK_SIZE)
		{
			pgstrom_cschunk_insert(relset, cs_usemap,
								   cs_values, cs_nulls, index);
			/* reset to head of the chunk */
			index = 0;
			MemoryContextReset(cs_memcxt);
		}
	}
	if (index > 0)
	{
		pgstrom_cschunk_insert(relset, cs_usemap,
							   cs_values, cs_nulls, index);
	}
	MemoryContextSwitchTo(oldcxt);

	heap_endscan(scan);

	MemoryContextDelete(cs_memcxt);

	/* release chunk memory */
	for (j=0; j < nattrs; j++)
	{
		pfree(cs_nulls[j]);
		pfree(cs_values[j]);
	}
	pfree(cs_nulls);
	pfree(cs_values);
	pfree(cs_usemap);
	pfree(rs_tuples);
	pfree(rs_nulls);
	pfree(rs_values);
}

static void
pgstrom_data_clear_internal(RelationSet relset)
{
	HeapScanDesc	scan;
	HeapTuple		tuple;
	AttrNumber		i, nattrs;

	/* clear the usemap table */
	scan = heap_beginscan(relset->rowid_rel,
						  SnapshotNow, 0, NULL);
	while (HeapTupleIsValid(tuple = heap_getnext(scan, ForwardScanDirection)))
	{
		simple_heap_delete(relset->rowid_rel, &tuple->t_self);
		CatalogUpdateIndexes(relset->rowid_rel, tuple);
	}
	heap_endscan(scan);

	/* clear the column stores */
	nattrs = RelationGetNumberOfAttributes(relset->base_rel);
	for (i=0; i < nattrs; i++)
	{
		scan = heap_beginscan(relset->cs_rel[i],
							  SnapshotNow, 0, NULL);
		while (HeapTupleIsValid(tuple = heap_getnext(scan,
													 ForwardScanDirection)))
		{
			if (!relset->cs_rel[i])
				continue;
			simple_heap_delete(relset->cs_rel[i], &tuple->t_self);
			CatalogUpdateIndexes(relset->cs_rel[i], tuple);
		}
		heap_endscan(scan);
	}
}

/*
 * bool pgstrom_data_clear(regclass)
 *
 *
 *
 *
 */
Datum
pgstrom_data_clear(PG_FUNCTION_ARGS)
{
	Relation		base_rel;
	RelationSet		relset;
	RangeTblEntry  *rte;

	/*
	 * Open the destination relation set
	 */
	base_rel = relation_open(PG_GETARG_OID(1), RowExclusiveLock);
	relset = pgstrom_open_relation_set(base_rel, RowExclusiveLock, false);

	/*
	 * Set up RangeTblEntry for permission checks
	 */
	rte = makeNode(RangeTblEntry);
	rte->rtekind = RTE_RELATION;
	rte->relid = RelationGetRelid(relset->base_rel);
	rte->relkind = RelationGetForm(relset->base_rel)->relkind;
	rte->requiredPerms = ACL_DELETE;

	ExecCheckRTPerms(list_make1(rte), true);

	/*
	 * Clear data
	 */
	pgstrom_data_clear_internal(relset);

	/*
	 * Close the relation
	 */
	pgstrom_close_relation_set(relset, NoLock);
	relation_close(base_rel, NoLock);

	PG_RETURN_BOOL(true);
}

/*
 * bool pgstrom_data_load(regclass source_rel,
 *                        regclass dest_rel);
 *
 */
Datum
pgstrom_data_load(PG_FUNCTION_ARGS)
{
	Relation		srel;
	Relation		base_rel;
	RelationSet		drelset;
	RangeTblEntry  *srte;
	RangeTblEntry  *drte;
	AttrNumber		i, nattrs;
	AttrNumber	   *attmap;

	/*
	 * Open the source relation
	 */
	srel = relation_open(PG_GETARG_OID(0), AccessShareLock);
	if (RelationGetForm(srel)->relkind != RELKIND_RELATION)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("%s is not a regular table",
						RelationGetRelationName(srel))));
	/*
	 * Open the destination relation set
	 */
	base_rel = relation_open(PG_GETARG_OID(1), RowExclusiveLock);
	drelset = pgstrom_open_relation_set(base_rel, RowExclusiveLock, false);

	/*
	 * Set up RangeTblEntry for permission checks
	 */
	srte = makeNode(RangeTblEntry);
	srte->rtekind = RTE_RELATION;
	srte->relid = RelationGetRelid(srel);
	srte->relkind = RelationGetForm(srel)->relkind;
	srte->requiredPerms = ACL_SELECT;

	drte = makeNode(RangeTblEntry);
	drte->rtekind = RTE_RELATION;
	drte->relid = RelationGetRelid(drelset->base_rel);
	drte->relkind = RelationGetForm(drelset->base_rel)->relkind;
	drte->requiredPerms = ACL_INSERT;

	/*
	 * Any columns of the source relation must exist on the destination
	 * relation with same data type.
	 *
	 * TODO: we should allow implicit cast.
	 */
	nattrs = RelationGetNumberOfAttributes(srel);
	attmap = palloc0(sizeof(AttrNumber) * nattrs);
	for (i=0; i < nattrs; i++)
	{
		Form_pg_attribute	attr1 = RelationGetDescr(srel)->attrs[i];
		Form_pg_attribute	attr2;
		HeapTuple			tuple;

		if (attr1->attisdropped)
			continue;

		tuple = SearchSysCacheAttName(RelationGetRelid(drelset->base_rel),
									  NameStr(attr1->attname));
		if (!HeapTupleIsValid(tuple))
			ereport(ERROR,
					(errcode(ERRCODE_FDW_INVALID_COLUMN_NAME),
					 errmsg("column \"%s\" of relation \"%s\" did not exist"
							" on the foreign table \"%s\"",
							NameStr(attr1->attname),
							RelationGetRelationName(srel),
							RelationGetRelationName(drelset->base_rel))));

		attr2 = (Form_pg_attribute) GETSTRUCT(tuple);
		if (attr1->atttypid != attr2->atttypid ||
			attr1->attlen != attr2->attlen ||
			attr1->attndims != attr2->attndims ||
			attr1->attbyval != attr2->attbyval)
			ereport(ERROR,
					(errcode(ERRCODE_FDW_INVALID_DATA_TYPE),
					 errmsg("column \"%s\" of relation \"%s\" isn't compatible"
							" with column \"%s\" of the foreign table \"%s\"",
							NameStr(attr1->attname),
							RelationGetRelationName(srel),
							NameStr(attr2->attname),
							RelationGetRelationName(drelset->base_rel))));
		attmap[i] = attr2->attnum;
		srte->selectedCols = bms_add_member(srte->selectedCols,
				attr1->attnum - FirstLowInvalidHeapAttributeNumber);
		drte->modifiedCols = bms_add_member(drte->modifiedCols,
				attr2->attnum - FirstLowInvalidHeapAttributeNumber);

		ReleaseSysCache(tuple);
	}

	/*
	 * Permission checks
	 */
	ExecCheckRTPerms(list_make2(srte, drte), true);

	/*
	 * Loada data
	 */
	pgstrom_data_load_internal(drelset, srel, attmap);

	/*
	 * Close the relation
	 */
	pgstrom_close_relation_set(drelset, NoLock);
	relation_close(base_rel, NoLock);
	relation_close(srel, AccessShareLock);

	PG_RETURN_BOOL(true);
}

Datum
pgstrom_data_compaction(PG_FUNCTION_ARGS)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("%s is not supported yet", __FUNCTION__)));
	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(pgstrom_data_load);
PG_FUNCTION_INFO_V1(pgstrom_data_clear);
PG_FUNCTION_INFO_V1(pgstrom_data_compaction);
