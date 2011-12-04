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
#include "utils/errcodes.h"
#include "utils/rel.h"
#include "utils/syscache.h"
#include "utils/tqual.h"
#include "utils/varbit.h"

#include "pg_strom.h"

typedef struct {
	int64		rowid;
	VarBit	   *usemap;
	int16		nattrs;
	int16	   *attlen;
	bool	   *attbyval;
	bool	  **nulls;
	void	  **values;
} CStoreChunkData;
typedef CStoreChunkData *CStoreChunk;

static CStoreChunk
pgstrom_cschunk_alloc(Relation base_rel, Bitmapset *valid_cols)
{
	CStoreChunk	chunk;
	AttrNumber	i;

	chunk = palloc0(sizeof(CStoreChunkData));
	chunk->usemap = palloc0(VARBITTOTALLEN(PGSTROM_CHUNK_SIZE));
	chunk->nattrs = RelationGetNumberOfAttributes(base_rel);
	chunk->attlen = palloc0(sizeof(int16) * chunk->nattrs);
	chunk->attbyval = palloc0(sizeof(bool) * chunk->nattrs);
	chunk->nulls = palloc0(sizeof(bool *) * chunk->nattrs);
	chunk->values = palloc0(sizeof(void *) * chunk->nattrs);

	for (i=0; i < chunk->nattrs; i++)
	{
		Form_pg_attribute	attr;

		if (valid_cols && !bms_is_member(i, valid_cols))
			continue;

		attr = RelationGetDescr(base_rel)->attrs[i];
		chunk->attlen[i] = attr->attlen;
		chunk->attbyval[i] = attr->attbyval;
		chunk->nulls[i] = palloc0(sizeof(bool) * PGSTROM_CHUNK_SIZE);
		if (attr->attlen > 0)
			chunk->values[i] = palloc0(attr->attlen * PGSTROM_CHUNK_SIZE);
		else
			chunk->values[i] = palloc0(sizeof(void *) * PGSTROM_CHUNK_SIZE);
	}
	return chunk;
}

static void
pgstrom_cschunk_free(CStoreChunk chunk)
{
	AttrNumber	i;

	for (i=0; i < chunk->nattrs; i++)
	{
		if (chunk->nulls[i])
			pfree(chunk->nulls[i]);
		if (chunk->values[i])
			pfree(chunk->values[i]);
	}
	pfree(chunk->values);
	pfree(chunk->nulls);
	pfree(chunk->attbyval);
	pfree(chunk->attlen);
	pfree(chunk->usemap);
	pfree(chunk);
}

/*
 * do_cschunk_insert_one
 *
 *
 *
 */
static void
do_cschunk_insert_one(Relation csrel, CatalogIndexState csidx,
					  int attlen, int nitems,
					  int64 cs_rowid, VarBit *cs_nulls, bytea *cs_values)
{
	HeapTuple	tuple;
	Datum		values[3];
	bool		nulls[3];
	Datum		temp1;
	Datum		temp2;

	memset(values, 0, sizeof(values));
	memset(nulls, false, sizeof(values));

	/* "rowid" of column store  */
	values[0] = Int64GetDatum(cs_rowid);

	/* "nulls" of column store */
	VARBITLEN(cs_nulls) = nitems;
	SET_VARSIZE(cs_nulls, VARBITTOTALLEN(attlen));
	values[1] = PointerGetDatum(cs_nulls);
	temp1 = toast_compress_datum(values[1]);
	if (DatumGetPointer(temp1) != NULL)
		values[1] = temp1;

	/* "values" of column store */
	SET_VARSIZE(cs_values, VARHDRSZ + nitems * attlen);
	values[2] = PointerGetDatum(cs_values);
	temp2 = toast_compress_datum(values[2]);
	if (DatumGetPointer(temp2) != NULL)
		values[2] = temp2;

	tuple = heap_form_tuple(RelationGetDescr(csrel), values, nulls);

	simple_heap_insert(csrel, tuple);
	CatalogIndexInsert(csidx, tuple);

	heap_freetuple(tuple);
	if (DatumGetPointer(temp1))
		pfree(DatumGetPointer(temp1));
	if (DatumGetPointer(temp2))
		pfree(DatumGetPointer(temp2));
}

static void
pgstrom_cschunk_insert(RelationSet relset, CStoreChunk chunk, int nitems)
{
	int64		rowid;
	TupleDesc	tupdesc;
	HeapTuple	tuple;
	Datum		temp;
	Datum		values[3];
	bool		nulls[3];
	int			j, index;
	Oid			save_userid;
	int			save_sec_context;

	/*
	 * Acquire a row-id of the head of this chunk
	 */
	GetUserIdAndSecContext(&save_userid, &save_sec_context);
	SetUserIdAndSecContext(BOOTSTRAP_SUPERUSERID,
						   save_sec_context | SECURITY_LOCAL_USERID_CHANGE);
	temp = DirectFunctionCall1(nextval_oid,
							   ObjectIdGetDatum(relset->sequence_id));
	rowid = DatumGetInt64(temp);
	SetUserIdAndSecContext(save_userid, save_sec_context);

	/*
	 * Insert a usemap of the rowid
	 */
	memset(values, 0, sizeof(values));
	memset(nulls, false, sizeof(nulls));

	values[0] = Int64GetDatum(rowid);
	VARBITLEN(chunk->usemap) = nitems;
	SET_VARSIZE(chunk->usemap, VARBITTOTALLEN(nitems));
	values[1] = PointerGetDatum(chunk->usemap);
	temp = toast_compress_datum(values[1]);
	if (DatumGetPointer(temp) != NULL)
		values[1] = temp;

	tupdesc = RelationGetDescr(relset->usemap_rel);
	tuple = heap_form_tuple(tupdesc, values, nulls);
	simple_heap_insert(relset->usemap_rel, tuple);
	CatalogIndexInsert(relset->usemap_idx, tuple);

	heap_freetuple(tuple);
	if (DatumGetPointer(temp) != NULL)
		pfree(DatumGetPointer(temp));

	/*
	 * Insert into column store
	 */
	Assert(chunk->nattrs == RelationGetNumberOfAttributes(relset->base_rel));
	for (j=0; j < chunk->nattrs; j++)
	{
		if (!chunk->values)
			continue;

		if (chunk->attlen[j] > 0)
		{
			VarBit *cs_nulls;
			bytea  *cs_values;
			int		cs_unitsz = PGSTROM_CHUNK_SIZE / chunk->attlen[j];
			int		cs_base;
			int		cs_offset;

			cs_nulls = palloc0(VARBITTOTALLEN(cs_unitsz));
			cs_values = palloc0(VARHDRSZ + PGSTROM_CHUNK_SIZE);

			for (index=0, cs_base=0, cs_offset=0;
				 index < nitems;
				 index++, cs_offset = index - cs_base)
			{
				if (cs_offset == cs_unitsz)
				{
					do_cschunk_insert_one(relset->column_rel[j],
										  relset->column_idx[j],
										  chunk->attlen[j],
										  cs_offset,
										  rowid + cs_base,
										  cs_nulls, cs_values);
					cs_base += cs_unitsz;
					cs_offset = 0;
					memset(cs_nulls, 0, VARBITTOTALLEN(cs_unitsz));
					memset(cs_values, 0, VARHDRSZ + PGSTROM_CHUNK_SIZE);
				}

				if (chunk->nulls[j][index])
					VARBITS(cs_nulls)[cs_offset >> 3] |= (1<<(cs_offset & 7));
				else
				{
					void   *psrc = ((char *)chunk->values[j] +
									index * chunk->attlen[j]);
					void   *pdst = (VARDATA(cs_values) +
									cs_offset * chunk->attlen[j]);
					memcpy(pdst, psrc, chunk->attlen[j]);
				}
			}
			if (cs_offset > 0)
				do_cschunk_insert_one(relset->column_rel[j],
									  relset->column_idx[j],
									  chunk->attlen[j],
									  cs_offset,
									  rowid + cs_base,
									  cs_nulls, cs_values);
			pfree(cs_nulls);
			pfree(cs_values);
		}
		else
		{
			tupdesc = RelationGetDescr(relset->column_rel[j]);
			for (index=0; index < nitems; index++)
			{
				if (chunk->nulls[j][index])
					continue;

				values[0] = Int64GetDatum(rowid + index);
				nulls[1] = true;
				values[2] = ((Datum *)chunk->values[j])[index];
				temp = toast_compress_datum(values[2]);
				if (DatumGetPointer(temp) != NULL)
					values[2] = temp;

				tuple = heap_form_tuple(tupdesc, values, nulls);
				simple_heap_insert(relset->column_rel[j], tuple);
				CatalogIndexInsert(relset->column_idx[j], tuple);
				heap_freetuple(tuple);

				if (DatumGetPointer(temp) != NULL)
					pfree(DatumGetPointer(temp));
			}
		}
	}
}

static void
pgstrom_data_load_internal(RelationSet relset,
						   Relation source, AttrNumber *attmap)
{
	CStoreChunk		chunk;
	HeapScanDesc	scan;
	HeapTuple		tuple;
	TupleDesc		tupdesc;
	Datum		   *values;
	bool		   *nulls;
	AttrNumber		i, j;
	int				index = 0;
	Bitmapset	   *valid_cols;

	tupdesc = RelationGetDescr(source);
	values = palloc(sizeof(Datum) * tupdesc->natts);
	nulls  = palloc(sizeof(bool)  * tupdesc->natts);

	for (i=0, valid_cols = NULL; tupdesc->natts; i++)
	{
		if ((j = attmap[i]) >= 0)
			valid_cols = bms_add_member(valid_cols, j);
	}
	chunk = pgstrom_cschunk_alloc(relset->base_rel, valid_cols);

	/*
	 * Scan the source relation
	 */
	scan = heap_beginscan(source, SnapshotNow, 0, NULL);
	while (HeapTupleIsValid(tuple = heap_getnext(scan, ForwardScanDirection)))
	{
		heap_deformtuple(tuple, tupdesc, values, nulls);

		/* set usemap */
		chunk->usemap->bit_dat[index >> 3] |= (1 << (index & 0x07));

		for (i=0; i < tupdesc->natts; i++)
		{
			if ((j = attmap[i] - 1) < 0)
				continue;

			if (nulls[i])
				chunk->nulls[j][index] = true;
			else
			{
				chunk->nulls[j][index] = false;

				if (chunk->attlen[j] > 0)
				{
					void   *pdst = ((char *)chunk->values[j] +
									index * chunk->attlen[j]);
					void   *psrc = (chunk->attbyval[j] ?
									DatumGetPointer(&values[i]) :
									DatumGetPointer(values[i]));
					memcpy(pdst, psrc, chunk->attlen[j]);
				}
				else
				{
					struct varlena *vl
						= pg_detoast_datum_copy((struct varlena *)(values[i]));
					((Datum *)chunk->values[j])[index] = PointerGetDatum(vl);
				}
			}
		}
		if (++index == PGSTROM_CHUNK_SIZE)
		{
			pgstrom_cschunk_insert(relset, chunk, index);
			for (j=0; j < chunk->nattrs; j++)
			{
				if (chunk->nulls[j])
					memset(chunk->nulls[j], false, PGSTROM_CHUNK_SIZE);
				if (chunk->values[j])
					memset(chunk->values[j], 0,
						   chunk->attlen[j] > 0 ?
						   chunk->attlen[j] * PGSTROM_CHUNK_SIZE :
						   sizeof(void *) * PGSTROM_CHUNK_SIZE);
			}
			index = 0;
		}
	}
	if (index > 0)
		pgstrom_cschunk_insert(relset, chunk, index);

	heap_endscan(scan);

	pgstrom_cschunk_free(chunk);
}

static void
pgstrom_data_clear_internal(RelationSet relset)
{
	HeapScanDesc	scan;
	HeapTuple		tuple;
	AttrNumber		i, nattrs;

	/* clear the usemap table */
	scan = heap_beginscan(relset->usemap_rel,
						  SnapshotNow, 0, NULL);
	while (HeapTupleIsValid(tuple = heap_getnext(scan, ForwardScanDirection)))
	{
		simple_heap_delete(relset->usemap_rel, &tuple->t_self);
		CatalogIndexInsert(relset->usemap_idx, tuple);
	}
	heap_endscan(scan);

	/* clear the column stores */
	nattrs = RelationGetNumberOfAttributes(relset->base_rel);
	for (i=0; i < nattrs; i++)
	{
		scan = heap_beginscan(relset->column_rel[i],
							  SnapshotNow, 0, NULL);
		while (HeapTupleIsValid(tuple = heap_getnext(scan,
													 ForwardScanDirection)))
		{
			if (!relset->column_rel[i])
				continue;
			simple_heap_delete(relset->column_rel[i], &tuple->t_self);
			CatalogIndexInsert(relset->column_idx[i], tuple);
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
	RelationSet		relset;
	RangeTblEntry  *rte;

	/*
	 * Open the destination relation set
	 */
	relset = pgstrom_open_relation_set(PG_GETARG_OID(1), RowExclusiveLock);

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
	drelset = pgstrom_open_relation_set(PG_GETARG_OID(1), RowExclusiveLock);

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

		tuple = SearchSysCacheAttName(RelationGetRelid(srel),
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
