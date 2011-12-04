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

typedef struct {
	AttrNumber	nattrs;
	int64		rowid;
	VarBit	   *usemap;
	bool	  **nulls;
	void	  **values;
} CStoreChunkData;
typedef CStoreChunkData *CStoreChunk;

static CStoreChunk
pgstrom_cschunk_alloc(Relation base_rel)
{
	CStoreChunk	chunk;
	AttrNumber	i;

	chunk = palloc0(sizeof(CStoreChunkData));
	chunk->nattrs = RelationGetNumberOfAttributes(base_rel);
	chunk->usemap = palloc0(VARBITTOTALLEN(PGSTROM_CHUNK_SIZE));
	chunk->nulls = palloc0(sizeof(bool *) * chunk->nattrs);
	chunk->values = palloc0(sizeof(void *) * chunk->nattrs);

	for (i=0; i < chunk->nattrs; i++)
	{
		Form_pg_attribute	attr
			= RelationGetDescr(base_rel)->attrs[i];

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
		pfree(chunk->nulls[i]);
		pfree(chunk->values[i]);
	}
	pfree(chunk->values);
	pfree(chunk->nulls);
	pfree(chunk->usemap);
	pfree(chunk);
}

static void
pgstrom_cschunk_insert(RelationSet relset, CStoreChunk chunk, int nitems)
{
	int64		rowid;
	TupleDesc	tupdesc;
	HeapTuple	tuple;
	Datum		temp;
	Datum		values[2];
	bool		nulls[2];
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
		Form_pg_attribute	attr
			= RelationGetDescr(relset->base_rel)->attrs[j];

		if (attr->attlen > 0)
		{
			int			cs_unitsz = PGSTROM_CHUNK_SIZE / attr->attlen;
			ArrayType  *cs_value;
			int			cs_base;
			int			cs_limit;
			int			cs_count;
			bool		cs_hasnull;
			bool		cs_hasvalid;

			cs_value = palloc0(ARR_OVERHEAD_WITHNULLS(1, cs_unitsz) +
							   PGSTROM_CHUNK_SIZE);

			for (cs_base = 0; cs_base < nitems; cs_base += cs_unitsz)
			{
				bits8  *nullmap;

				cs_limit = cs_base + cs_unitsz;
				if (cs_limit > nitems)
					cs_limit = nitems;
				cs_count = (cs_limit - cs_base);

				/* set up an array */
				cs_hasnull = false;
				cs_hasvalid = false;
				cs_value->ndim = 1;
				cs_value->dataoffset = ARR_OVERHEAD_WITHNULLS(1, cs_count);
				cs_value->elemtype = attr->atttypid;
				ARR_DIMS(cs_value)[0] = cs_count;
				ARR_LBOUND(cs_value)[0] = 0;

				/* set up nullmap, if needed */
				nullmap = ARR_NULLBITMAP(cs_value);
				memset(nullmap, 0, (cs_count + 7) / 8);
				for (index = cs_base; index < cs_limit; index++)
				{
					if (chunk->nulls[j][index])
					{
						nullmap[(index - cs_base) >> 3]
							|= (1 << ((index - cs_base) & 7));
						cs_hasnull = true;
					}
					else
						cs_hasvalid = true;
				}
				/* if no valid items, no need to write this array */
				if (!cs_hasvalid)
					continue;
				/* if no null, we remove nullbitmap from array */
				if (!cs_hasnull)
					cs_value->dataoffset = 0;

				/* data copy */
				memcpy(ARR_DATA_PTR(cs_value),
					   ((char *)chunk->values[j] + cs_base * attr->attlen),
					   cs_count * attr->attlen);
				SET_VARSIZE(cs_value,
							(ARR_HASNULL(cs_value) ?
							 ARR_OVERHEAD_WITHNULLS(1, cs_count) :
							 ARR_OVERHEAD_NONULLS(1)) +
							cs_count * attr->attlen);

				memset(nulls, false, sizeof(nulls));
				values[0] = Int64GetDatum(rowid + cs_base);
				values[1] = PointerGetDatum(cs_value);
				temp = toast_compress_datum(values[1]);
				if (DatumGetPointer(temp) != NULL)
					values[1] = temp;

				tuple = heap_form_tuple(tupdesc, values, nulls);
				simple_heap_insert(relset->column_rel[j], tuple);
				CatalogIndexInsert(relset->column_idx[j], tuple);
				heap_freetuple(tuple);

				if (DatumGetPointer(temp) != NULL)
					pfree(DatumGetPointer(temp));
			}
			pfree(cs_value);
		}
		else
		{
			tupdesc = RelationGetDescr(relset->column_rel[j]);
			for (index=0; index < nitems; index++)
			{
				if (chunk->nulls[j][index])
					continue;

				values[0] = Int64GetDatum(rowid + index);
				values[1] = ((Datum *)chunk->values[j])[index];
				temp = toast_compress_datum(values[2]);
				if (DatumGetPointer(temp) != NULL)
					values[1] = temp;

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

	tupdesc = RelationGetDescr(source);
	values = palloc(sizeof(Datum) * tupdesc->natts);
	nulls  = palloc(sizeof(bool)  * tupdesc->natts);
	chunk = pgstrom_cschunk_alloc(relset->base_rel);

	/*
	 * Scan the source relation
	 */
	scan = heap_beginscan(source, SnapshotNow, 0, NULL);
	while (HeapTupleIsValid(tuple = heap_getnext(scan, ForwardScanDirection)))
	{
		heap_deform_tuple(tuple, tupdesc, values, nulls);

		/* set usemap */
		VARBITS(chunk->usemap)[index >> 3] |= (1 << (index & 0x07));

		for (i=0; i < tupdesc->natts; i++)
		{
			if ((j = attmap[i] - 1) < 0)
				continue;

			if (nulls[i])
				chunk->nulls[j][index] = true;
			else
			{
				Form_pg_attribute	attr
					= RelationGetDescr(relset->base_rel)->attrs[j];

				chunk->nulls[j][index] = false;

				if (attr->attlen > 0)
				{
					void   *pdst = ((char *)chunk->values[j] +
									index * attr->attlen);
					void   *psrc = (attr->attbyval ?
									DatumGetPointer(&values[i]) :
									DatumGetPointer(values[i]));
					memcpy(pdst, psrc, attr->attlen);
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
				Form_pg_attribute	attr
					= RelationGetDescr(relset->base_rel)->attrs[j];

				memset(chunk->nulls[j], false, PGSTROM_CHUNK_SIZE);
				memset(chunk->values[j], 0,
					   attr->attlen > 0 ?
					   attr->attlen * PGSTROM_CHUNK_SIZE :
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
