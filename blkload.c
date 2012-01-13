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
#include "catalog/namespace.h"
#include "catalog/pg_attribute.h"
#include "catalog/pg_authid.h"
#include "catalog/pg_class.h"
#include "commands/sequence.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "nodes/makefuncs.h"
#include "utils/array.h"
#include "utils/errcodes.h"
#include "utils/rel.h"
#include "utils/lsyscache.h"
#include "utils/syscache.h"
#include "utils/tqual.h"
#include "utils/varbit.h"

#include "pg_strom.h"

static Datum
construct_cs_isnull(bool cs_isnull[], int nitems)
{
	bytea  *result;
	size_t	length;
	int		index;
	Datum	temp;

	length = VARHDRSZ + (nitems + (BITS_PER_BYTE - 1)) / BITS_PER_BYTE;
	result = palloc0(length);

	for (index = 0; index < nitems; index++)
	{
		if (cs_isnull[index])
			((uint8 *)VARDATA(result))[index / BITS_PER_BYTE]
				|= (1 << (index % BITS_PER_BYTE));
	}
	SET_VARSIZE(result, length);

	temp = toast_compress_datum(PointerGetDatum(result));
	if (DatumGetPointer(temp) == NULL)
		return PointerGetDatum(result);

	pfree(result);

	return temp;
}

static Datum
construct_cs_values(Datum cs_values[], bool cs_isnull[], int nitems,
					Form_pg_attribute cs_attr)
{
	bytea  *result;
	size_t	length;
	size_t	offset;
	int		index;
	Datum	temp;

	if (cs_attr->attlen > 0)
	{
		length = VARHDRSZ + cs_attr->attlen * nitems;
		offset = 0;
	}
	else
	{
		length = VARHDRSZ;
		for (index = 0; index < nitems; index++)
		{
			length += sizeof(uint16);
			if (!cs_isnull[index])
				length += MAXALIGN(VARSIZE(cs_values[index]));
		}
		offset = sizeof(uint16) * nitems;
	}

	result = palloc0(length);
	SET_VARSIZE(result, length);

	for (index=0; index < nitems; index++)
	{
		if (cs_attr->attlen > 0)
		{
			if (cs_attr->attbyval)
				store_att_byval(VARDATA(result) + offset,
								cs_values[index],
								cs_attr->attlen);
			else
				memmove(VARDATA(result) + offset,
						DatumGetPointer(cs_values[index]),
						cs_attr->attlen);
			offset += att_align_nominal(cs_attr->attlen,
										cs_attr->attalign);
		}
		else
		{
			if (cs_isnull[index])
				((uint16 *)VARDATA(result))[index] = 0;
			else
			{
				((uint16 *)VARDATA(result))[index] = offset;

				memcpy(VARDATA(result) + offset,
					   DatumGetPointer(cs_values[index]),
					   VARSIZE(cs_values[index]));
				offset += MAXALIGN(VARSIZE(cs_values[index]));
			}
		}
	}
	Assert(offset + VARHDRSZ == length);

	temp = toast_compress_datum(PointerGetDatum(result));
	if (DatumGetPointer(temp) != NULL)
	{
		pfree(result);
		return temp;
	}
	return PointerGetDatum(result);
}

static void
pgstrom_one_chunk_insert(Relation srel,
						 Oid    seqid,
						 uint32	chunk_size,
						 uint32 nitems,
						 Relation id_rel,
						 Relation cs_rels[],
						 Form_pg_attribute cs_attrs[],
						 bool  *cs_rowid,
						 bool  *cs_isnull[],
						 Datum *cs_values[])
{
	int64		rowid;
	TupleDesc	tupdesc;
	HeapTuple	tuple;
	Datum		temp;
	Datum		values[Natts_pg_strom];
	bool		isnull[Natts_pg_strom];
	Oid			save_userid;
	int			save_sec_context;
	AttrNumber	attno, nattrs;

	Assert(chunk_size % BITS_PER_BYTE == 0);

	/*
	 * Acquire a row-id of the head of this chunk
	 */
	GetUserIdAndSecContext(&save_userid, &save_sec_context);
	SetUserIdAndSecContext(BOOTSTRAP_SUPERUSERID,
						   save_sec_context | SECURITY_LOCAL_USERID_CHANGE);
	temp = DirectFunctionCall1(nextval_oid, ObjectIdGetDatum(seqid));
	rowid = DatumGetInt64(temp);
	SetUserIdAndSecContext(save_userid, save_sec_context);

	/*
	 * Insert rowid-map of this chunk.
	 *
	 * XXX - note that its 'nitems' is always chunk_size to make clear
	 * between rowid and (rowid + nitems - 1) are occupied.
	 */
	memset(values, 0, sizeof(values));
	memset(isnull, 0, sizeof(isnull));

	values[Anum_pg_strom_rowid - 1] = Int64GetDatum(rowid);
	values[Anum_pg_strom_nitems - 1] = Int32GetDatum(chunk_size);
	values[Anum_pg_strom_isnull - 1] =
		construct_cs_isnull(cs_rowid, chunk_size);

	tupdesc = RelationGetDescr(id_rel);
	tuple = heap_form_tuple(tupdesc, values, isnull);
	simple_heap_insert(id_rel, tuple);
	CatalogUpdateIndexes(id_rel, tuple);
	heap_freetuple(tuple);

	/*
	 * Insert into column store
	 */
	nattrs = RelationGetNumberOfAttributes(srel);
	for (attno = 0; attno < nattrs; attno++)
	{
		Form_pg_attribute	attr = cs_attrs[attno];
		int			rs_base;
		int			rs_unitsz;
		int			num_nulls;
		int			index;

		for (rs_base=0; rs_base < nitems; rs_base += rs_unitsz)
		{
			/*
			 * XXX - We assume 50% of BLCKSZ is a watermark to put data
			 * into an array of values. It is just a heuristics, so it
			 * may be changed in the future release.
			 */
			if (attr->attlen > 0)
				rs_unitsz = (BLCKSZ / 2) / attr->attlen;
			else
			{
				size_t	total_length = 0;

				for (rs_unitsz=0; rs_base+rs_unitsz < nitems; rs_unitsz++)
				{
					index = rs_base+rs_unitsz;

					/*
					 * XXX - we may need to pay attention to null-bitmap,
					 * if and when one varlena is null.
					 */
					if (cs_isnull[attno][index])
						continue;

					/* make sure being uncompressed */
					temp = (Datum)PG_DETOAST_DATUM(cs_values[attno][index]);
					cs_values[attno][index] = temp;

					if (total_length + VARSIZE(temp) > (BLCKSZ / 2))
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
			 * In the case of either all the values or no values were NULL,
			 * we treat them as a special case to optimize storage usage.
			 * If all the values were NULL, we skip to insert this record.
			 * If no values were NULL, 'isnull' attribute will have NULL.
			 */
			num_nulls = 0;
			for (index = 0; index < rs_unitsz; index++)
			{
				if (cs_isnull[attno][rs_base + index])
					num_nulls++;
			}
			if (num_nulls == rs_unitsz)
				continue;

			/*
			 * Set up a tuple of column store
			 */
			memset(isnull, false, sizeof(isnull));
			values[Anum_pg_strom_rowid - 1] = Int64GetDatum(rowid + rs_base);
			values[Anum_pg_strom_nitems - 1] = Int32GetDatum(rs_unitsz);
			if (num_nulls == 0)
				isnull[Anum_pg_strom_isnull - 1] = true;
			else
				values[Anum_pg_strom_isnull - 1]
					= construct_cs_isnull(&cs_isnull[attno][rs_base],
										  rs_unitsz);

			values[Anum_pg_strom_values - 1]
				= construct_cs_values(&cs_values[attno][rs_base],
									  &cs_isnull[attno][rs_base],
									  rs_unitsz, attr);

			tupdesc = RelationGetDescr(cs_rels[attno]);
			tuple = heap_form_tuple(tupdesc, values, isnull);
			simple_heap_insert(cs_rels[attno], tuple);
			CatalogUpdateIndexes(cs_rels[attno], tuple);
			heap_freetuple(tuple);
		}
	}
}

static void
pgstrom_data_load_internal(Relation srel,
						   Oid		seqid,
						   Relation id_rel,
						   Relation cs_rels[],
						   Form_pg_attribute cs_attrs[],
						   uint32 chunk_size)
{
	TupleDesc		tupdesc;
	HeapScanDesc	scan;
	HeapTuple		tuple;
	Datum		   *rs_values;
	bool		   *rs_isnull;
	bool		   *cs_rowmap;
	Datum		  **cs_values;
	bool		  **cs_isnull;
	MemoryContext	cs_memcxt;
	MemoryContext	oldcxt;
	AttrNumber		attno;
	int				index = 0;

	tupdesc = RelationGetDescr(srel);
	rs_values = palloc0(sizeof(Datum) * tupdesc->natts);
	rs_isnull = palloc0(sizeof(bool)  * tupdesc->natts);

	cs_rowmap = palloc0(sizeof(bool) * chunk_size);
	cs_values = palloc0(sizeof(Datum *) * tupdesc->natts);
	cs_isnull = palloc0(sizeof(bool *) * tupdesc->natts);
	for (attno = 0; attno < tupdesc->natts; attno++)
	{
		if (!cs_rels[attno])
			continue;
		cs_values[attno] = palloc(sizeof(Datum) * chunk_size);
		cs_isnull[attno] = palloc(sizeof(bool) * chunk_size);
	}

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
	scan = heap_beginscan(srel, SnapshotNow, 0, NULL);

	oldcxt = MemoryContextSwitchTo(cs_memcxt);

	index = 0;
	memset(cs_rowmap, -1, sizeof(bool) * chunk_size);

	while (HeapTupleIsValid(tuple = heap_getnext(scan, ForwardScanDirection)))
	{
		HeapTuple	rs_tuple = heap_copytuple(tuple);

		heap_deform_tuple(rs_tuple, tupdesc, rs_values, rs_isnull);

		cs_rowmap[index] = false;

		for (attno = 0; attno < tupdesc->natts; attno++)
		{
			if (!cs_rels[attno])
				continue;

			if (rs_isnull[attno])
			{
				if (cs_attrs[attno]->attnotnull)
					ereport(ERROR,
							(errcode(ERRCODE_NOT_NULL_VIOLATION),
							 errmsg("null in column \"%s\" violates NOT NULL constraint",
									NameStr(cs_attrs[attno]->attname))));
				cs_isnull[attno][index] = true;
				cs_values[attno][index] = (Datum) 0;
			}
			else
			{
				cs_isnull[attno][index] = false;
				cs_values[attno][index] = rs_values[attno];
			}
		}

		if (++index == chunk_size)
		{
			pgstrom_one_chunk_insert(srel, seqid, chunk_size, index,
									 id_rel, cs_rels, cs_attrs,
									 cs_rowmap, cs_isnull, cs_values);
			/*
			 * Rewind the index to the head, and release all
			 * the per-chunk memory
			 */
			index = 0;
			memset(cs_rowmap, -1, sizeof(bool) * chunk_size);
			MemoryContextReset(cs_memcxt);
		}
	}
	if (index > 0)
	{
		/* index should be round up to multiple number of 8 */
		index = (index + BITS_PER_BYTE - 1) & ~(BITS_PER_BYTE - 1);
		Assert(index <= chunk_size);

		pgstrom_one_chunk_insert(srel, seqid, chunk_size, index,
								 id_rel, cs_rels, cs_attrs,
								 cs_rowmap, cs_isnull, cs_values);
	}
	heap_endscan(scan);

	MemoryContextSwitchTo(oldcxt);

	MemoryContextDelete(cs_memcxt);
}

/*
 * bool
 * pgstrom_data_load(regclass dest, regclass source, uint32 chunk_size)
 *
 * This function loads the contents of source table into the destination
 * foreign table managed by PG-Strom; with the supplied chunk_size.
 */
Datum
pgstrom_data_load(PG_FUNCTION_ARGS)
{
	Relation		srel;
	Relation		drel;
	Relation		id_rel;
	Relation	   *cs_rels;
	Form_pg_attribute *cs_attrs;
	Bitmapset	   *scols = NULL;
	Bitmapset	   *dcols = NULL;
	RangeTblEntry  *srte;
	RangeTblEntry  *drte;
	HeapScanDesc	scan;
	HeapTuple		tuple;
	RangeVar	   *range;
	Oid				nspid;
	Oid				seqid;
	uint32			chunk_size = PG_GETARG_UINT32(2);
	AttrNumber		i, nattrs;

	/*
	 * Open the source relation
	 */
	srel = relation_open(PG_GETARG_OID(1), AccessShareLock);
	if (RelationGetForm(srel)->relkind != RELKIND_RELATION)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("%s is not a regular table",
						RelationGetRelationName(srel))));
	/*
	 * Open the destination relation and rowid store
	 */
	drel = relation_open(PG_GETARG_OID(0), RowExclusiveLock);
	id_rel = pgstrom_open_shadow_table(drel, InvalidAttrNumber,
									   RowExclusiveLock);
	cs_rels = palloc0(sizeof(Relation) *
					  RelationGetNumberOfAttributes(srel));
	cs_attrs = palloc0(sizeof(Form_pg_attribute) *
					   RelationGetNumberOfAttributes(srel));

	/*
	 * Lookup the Oid of rowid sequencial generator
	 */
	range = pgstrom_lookup_shadow_sequence(drel);
	nspid = get_namespace_oid(range->schemaname, false);
	seqid = get_relname_relid(range->relname, nspid);
	if (!OidIsValid(seqid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("PG-Strom: shadow sequence \"%s.%s\" not found",
						range->schemaname, range->relname)));

	/*
	 * Open required shadow tables
	 */
	nattrs = RelationGetNumberOfAttributes(srel);
	for (i=0; i < nattrs; i++)
	{
		Form_pg_attribute	attr1 = RelationGetDescr(srel)->attrs[i];
		Form_pg_attribute	attr2;

		if (attr1->attisdropped)
			continue;

		tuple = SearchSysCacheAttName(RelationGetRelid(drel),
									  NameStr(attr1->attname));
		if (!HeapTupleIsValid(tuple))
			ereport(ERROR,
					(errcode(ERRCODE_FDW_INVALID_COLUMN_NAME),
					 errmsg("column \"%s\" of relation \"%s\" did not exist on the foreign table \"%s\"",
							NameStr(attr1->attname),
							RelationGetRelationName(srel),
							RelationGetRelationName(drel))));

		attr2 = (Form_pg_attribute) GETSTRUCT(tuple);
		if (attr1->atttypid != attr2->atttypid ||
			attr1->attlen != attr2->attlen ||
			attr1->attndims != attr2->attndims ||
			attr1->attbyval != attr2->attbyval)
			ereport(ERROR,
					(errcode(ERRCODE_FDW_INVALID_DATA_TYPE),
					 errmsg("column \"%s\" of relation \"%s\" does not have compatible layout with column \"%s\" of the foreign table \"%s\"",
							NameStr(attr1->attname),
							RelationGetRelationName(srel),
							NameStr(attr2->attname),
							RelationGetRelationName(drel))));

		cs_attrs[i] = attr2;
		cs_rels[i]  = pgstrom_open_shadow_table(drel, attr2->attnum,
												RowExclusiveLock);
		scols = bms_add_member(scols,
					attr1->attnum - FirstLowInvalidHeapAttributeNumber);
		dcols = bms_add_member(dcols,
					attr2->attnum - FirstLowInvalidHeapAttributeNumber);

		ReleaseSysCache(tuple);
	}

	/*
	 * Check correctness of the chunk-size
	 */
	scan = heap_beginscan(id_rel, SnapshotNow, 0, NULL);
	tuple = heap_getnext(scan, ForwardScanDirection);
	if (HeapTupleIsValid(tuple))
	{
		TupleDesc	tupdesc = RelationGetDescr(id_rel);
		Datum		values[Natts_pg_strom - 1];
		bool		isnull[Natts_pg_strom - 1];
		uint32		nitems;

		heap_deform_tuple(tuple, tupdesc, values, isnull);
		Assert(!isnull[0] && !isnull[1]);

		nitems = DatumGetInt32(values[Anum_pg_strom_nitems - 1]);
		if (chunk_size == 0)
			chunk_size = nitems;
		else if (chunk_size != nitems)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_OBJECT_DEFINITION),
					 errmsg("chunk size must be same with existing data")));
	}
	else
	{
		AlterSeqStmt   *stmt;
		Oid		save_userid;
		int		save_sec_context;

		if (chunk_size == 0)
			chunk_size = BLCKSZ * BITS_PER_BYTE / 2; /* default */
		else if ((chunk_size & (chunk_size - 1)) != 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_OBJECT_DEFINITION),
					 errmsg("chunk size must be a power of 2")));
		else if (chunk_size < 4096 || chunk_size > BLCKSZ * BITS_PER_BYTE)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_OBJECT_DEFINITION),
					 errmsg("chunk size is out of range")));

		/* Reset sequence object to fit the supplied chunk size */
		stmt = makeNode(AlterSeqStmt);
		stmt->sequence = pgstrom_lookup_shadow_sequence(drel);
		stmt->options = list_make2(
			makeDefElem("increment", (Node *)makeInteger(chunk_size)),
			makeDefElem("restart",   (Node *)makeInteger(0)));

		GetUserIdAndSecContext(&save_userid, &save_sec_context);
		SetUserIdAndSecContext(BOOTSTRAP_SUPERUSERID, save_sec_context);

		AlterSequence(stmt);

		SetUserIdAndSecContext(save_userid, save_sec_context);
	}
	heap_endscan(scan);

	/*
	 * Set up RangeTblEntry, then Permission checks
	 */
	srte = makeNode(RangeTblEntry);
	srte->rtekind = RTE_RELATION;
	srte->relid = RelationGetRelid(srel);
	srte->relkind = RelationGetForm(srel)->relkind;
	srte->requiredPerms = ACL_SELECT;
	srte->selectedCols = scols;

	drte = makeNode(RangeTblEntry);
	drte->rtekind = RTE_RELATION;
	drte->relid = RelationGetRelid(drel);
	drte->relkind = RelationGetForm(drel)->relkind;
	drte->requiredPerms = ACL_INSERT;
	drte->modifiedCols = dcols;

	ExecCheckRTPerms(list_make2(srte, drte), true);

	/*
	 * Load data
	 */
	pgstrom_data_load_internal(srel, seqid, id_rel,
							   cs_rels, cs_attrs,
							   chunk_size);

	/*
	 * Close the relation
	 */
	for (i=0; i < nattrs; i++)
	{
		if (cs_rels[i])
			relation_close(cs_rels[i], NoLock);
	}
	relation_close(id_rel, NoLock);
	relation_close(drel, NoLock);
	relation_close(srel, AccessShareLock);

	PG_RETURN_BOOL(true);
}
PG_FUNCTION_INFO_V1(pgstrom_data_load);

/*
 * bool pgstrom_data_clear(regclass)
 *
 * This function clears contents of the foreign table managed by PG-Strom.
 */
Datum
pgstrom_data_clear(PG_FUNCTION_ARGS)
{
	Relation		base_rel;
	Relation		id_rel;
	Relation	   *cs_rels;
	RangeTblEntry  *rte;
    HeapScanDesc	scan;
    HeapTuple		tuple;	
	AttrNumber		csidx, nattrs;

	/*
	 * Open the destination relation set
	 */
	base_rel = relation_open(PG_GETARG_OID(1), RowExclusiveLock);
	id_rel = pgstrom_open_shadow_table(base_rel, InvalidAttrNumber,
									   RowExclusiveLock);
	nattrs = RelationGetNumberOfAttributes(base_rel);
	cs_rels = palloc0(sizeof(Relation) * nattrs);
	for (csidx = 0; csidx < nattrs; csidx++)
	{
		Form_pg_attribute	attr
			= RelationGetDescr(base_rel)->attrs[csidx];

		if (attr->attisdropped)
			continue;

		cs_rels[csidx] = pgstrom_open_shadow_table(base_rel, attr->attnum,
												   RowExclusiveLock);
	}

	/*
	 * Set up RangeTblEntry for permission checks
	 */
	rte = makeNode(RangeTblEntry);
	rte->rtekind = RTE_RELATION;
	rte->relid = RelationGetRelid(base_rel);
	rte->relkind = RelationGetForm(base_rel)->relkind;
	rte->requiredPerms = ACL_DELETE;

	ExecCheckRTPerms(list_make1(rte), true);

	/*
	 * Clear the rowid map
	 */
	scan = heap_beginscan(id_rel, SnapshotNow, 0, NULL);
	while (HeapTupleIsValid(tuple = heap_getnext(scan, ForwardScanDirection)))
	{
		simple_heap_delete(id_rel, &tuple->t_self);
		CatalogUpdateIndexes(id_rel, tuple);
	}
	heap_endscan(scan);

	/*
	 * Clear the column stores
	 */
	for (csidx = 0; csidx < nattrs; csidx++)
	{
		scan = heap_beginscan(cs_rels[csidx], SnapshotNow, 0, NULL);
		while (HeapTupleIsValid(tuple = heap_getnext(scan,
													 ForwardScanDirection)))
		{
			if (!cs_rels[csidx])
				continue;
			simple_heap_delete(cs_rels[csidx], &tuple->t_self);
			CatalogUpdateIndexes(cs_rels[csidx], tuple);
		}
		heap_endscan(scan);
	}

	/*
	 * Close the relation
	 */
	for (csidx = 0; csidx < nattrs; csidx++)
	{
		if (cs_rels[csidx])
			relation_close(cs_rels[csidx], NoLock);
	}
	relation_close(id_rel, NoLock);
	relation_close(base_rel, NoLock);

	PG_RETURN_BOOL(true);
}
PG_FUNCTION_INFO_V1(pgstrom_data_clear);

Datum
pgstrom_data_compaction(PG_FUNCTION_ARGS)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("%s is not supported yet", __FUNCTION__)));
	PG_RETURN_BOOL(true);
}
PG_FUNCTION_INFO_V1(pgstrom_data_compaction);
