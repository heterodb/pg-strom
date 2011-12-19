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
#include "access/relscan.h"
#include "catalog/namespace.h"
#include "catalog/pg_class.h"
#include "catalog/pg_type.h"
#include "foreign/foreign.h"
#include "nodes/makefuncs.h"
#include "utils/array.h"
#include "utils/fmgroids.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/resowner.h"
#include "utils/varbit.h"
#include "pg_strom.h"

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


typedef struct {
	int			nattrs;
	int64		rowid;
	VarBit	   *rowmap;
	bits8	  **cs_nulls;
	void	  **cs_values;
	bool	   *cs_pinned;	/* T, if pinned buffer */
} PgStromChunkBuf;

typedef struct {
	RelationSet		relset;
	Bitmapset	   *cols_needed;
	bool			with_syscols;

	Snapshot		es_snapshot;
	HeapScanDesc	es_scan;
	MemoryContext	es_context;

	List		   *chunk_list;
	ListCell	   *chunk_curr;
	int				chunk_index;
} PgStromExecState;

static void
pgstrom_release_chunk_buffer(PgStromChunkBuf *chunk)
{
	int		i;

	pfree(chunk->rowmap);
	for (i=0; i < chunk->nattrs; i++)
	{
		if (chunk->cs_nulls[i])
			pfree(chunk->cs_nulls[i]);
		if (chunk->cs_values[i])
			pfree(chunk->cs_values[i]);
	}
	pfree(chunk->cs_nulls);
	pfree(chunk->cs_values);
	pfree(chunk);
}

static void
pgstrom_read_chunk_buffer_cs(PgStromExecState *sestate,
							 PgStromChunkBuf *chunk,
							 int64 rowid, AttrNumber csidx)
{
	Form_pg_attribute	attr
		= RelationGetDescr(sestate->relset->base_rel)->attrs[csidx];
	IndexScanDesc	iscan;
	ScanKeyData		skeys[2];
	HeapTuple		tup;
	bool			found = false;

	/*
	 * XXX - should be pinned memory for async memory transfer
	 */
	chunk->cs_values[csidx]
		= MemoryContextAllocZero(sestate->es_context,
								 PGSTROM_CHUNK_SIZE *
								 (attr->attlen > 0 ?
								  attr->attlen : sizeof(bytea *)));
	/*
	 * Try to scan column store with cs_rowid betweem rowid and
	 * (rowid + PGSTROM_CHUNK_SIZE)
	 */
	ScanKeyInit(&skeys[0],
				(AttrNumber) 1,
				BTGreaterEqualStrategyNumber, F_INT8GE,
				Int64GetDatum(rowid));
	ScanKeyInit(&skeys[1],
				(AttrNumber) 1,
				BTLessStrategyNumber, F_INT8LT,
				Int64GetDatum(rowid + PGSTROM_CHUNK_SIZE));

	iscan = index_beginscan(sestate->relset->cs_rel[csidx],
							sestate->relset->cs_idx[csidx],
							sestate->es_snapshot, 2, 0);
	index_rescan(iscan, skeys, 2, NULL, 0);

	while (HeapTupleIsValid(tup = index_getnext(iscan, ForwardScanDirection)))
	{
		TupleDesc	tupdesc;
		Datum		values[2];
		bool		nulls[2];
		int64		cs_rowid;
		ArrayType  *cs_array;
		bits8	   *nullbitmap;
		int			i, offset;
		int			nitems;

		found = true;

		tupdesc = RelationGetDescr(sestate->relset->cs_rel[csidx]);
		heap_deform_tuple(tup, tupdesc, values, nulls);
		Assert(!nulls[0] && !nulls[1]);

		cs_rowid = Int64GetDatum(values[0]);
		cs_array = DatumGetArrayTypeP(values[1]);

		offset = cs_rowid - rowid;
		Assert(offset >= 0 && offset < PGSTROM_CHUNK_SIZE);
		Assert((offset & (BITS_PER_BYTE - 1)) == 0);
		Assert(ARR_NDIM(cs_array) == 1);
		Assert(ARR_LBOUND(cs_array)[0] == 0);
		Assert(attr->atttypid == ARR_ELEMTYPE(cs_array));

		/*
		 * XXX - nullbit map shall be acquired on demand
		 * Also note that it needs pinned buffer
		 */
		nullbitmap = ARR_NULLBITMAP(cs_array);
		if (nullbitmap && !chunk->cs_nulls[csidx])
			chunk->cs_nulls[csidx]
				= MemoryContextAllocZero(sestate->es_context,
										 PGSTROM_CHUNK_SIZE / BITS_PER_BYTE);
		nitems = ARR_DIMS(cs_array)[0];
		if (attr->attlen > 0)
		{
			memcpy(((char *)chunk->cs_values[csidx]) + offset * attr->attlen,
				   ARR_DATA_PTR(cs_array),
				   nitems * attr->attlen);
			if (nullbitmap)
				memcpy(chunk->cs_nulls[csidx] + (offset>>3),
					   nullbitmap,
					   (nitems + BITS_PER_BYTE - 1) >> 3);
		}
		else
		{
			char		   *vlptr = ARR_DATA_PTR(cs_array);
			MemoryContext	oldctx
				= MemoryContextSwitchTo(sestate->es_context);

			for (i=0; i < nitems; i++)
			{
				if (!nullbitmap ||
					(nullbitmap[i>>3] && (1 << (i & (BITS_PER_BYTE-1)))) == 0)
				{
					((bytea **)chunk->cs_values[csidx])[offset + i]
						= PG_DETOAST_DATUM_COPY((struct varlena *) vlptr);
					vlptr = att_addlength_pointer(vlptr, attr->attlen, vlptr);
					vlptr = (char *)att_align_nominal(vlptr, attr->attalign);
				}
				else
				{
					chunk->cs_nulls[csidx][offset + i]
						|= (1 << ((offset + i) & (BITS_PER_BYTE - 1)));
				}
			}
			MemoryContextSwitchTo(oldctx);
		}
	}

	/*
	 * In a case when no tuples are not found with cs_rowid between rowid
	 * and (rowid + PGSTROM_CHUNK_SIZE), we initialize all the items as null.
	 */
	if (!found)
	{
		chunk->cs_nulls[csidx] = palloc(PGSTROM_CHUNK_SIZE / BITS_PER_BYTE);
		memset(chunk->cs_nulls[csidx], -1,
			   PGSTROM_CHUNK_SIZE / BITS_PER_BYTE);
	}
	index_endscan(iscan);
}

static PgStromChunkBuf *
pgstrom_read_chunk_buffer(PgStromExecState *sestate,
						  int64 rowid, VarBit *rowmap)
{
	PgStromChunkBuf	   *chunk;
	MemoryContext		oldcxt;
	int		csidx;

	oldcxt = MemoryContextSwitchTo(sestate->es_context);

	chunk = palloc(sizeof(PgStromChunkBuf));
	chunk->nattrs = RelationGetNumberOfAttributes(sestate->relset->base_rel);
	chunk->rowid = rowid;
	chunk->rowmap = rowmap;

	chunk->cs_nulls = palloc0(sizeof(bits8 *) * chunk->nattrs);
	chunk->cs_values = palloc0(sizeof(void *) * chunk->nattrs);
	chunk->cs_pinned = palloc0(sizeof(bool) * chunk->nattrs);

	MemoryContextSwitchTo(oldcxt);

	for (csidx = 0; csidx < chunk->nattrs; csidx++)
	{
		if (!bms_is_member(csidx, sestate->cols_needed))
			continue;

		pgstrom_read_chunk_buffer_cs(sestate, chunk, rowid, csidx);
	}
	return chunk;
}

static bool
pgstrom_load_next_chunk_buffer(PgStromExecState *sestate)
{
	TupleDesc	tupdesc;
	HeapTuple	tuple;
	Datum		values[2];
	bool		nulls[2];
	int64		cs_rowid;
	VarBit	   *cs_rowmap;
	MemoryContext oldctx;
	PgStromChunkBuf *chunk;

	tuple = heap_getnext(sestate->es_scan, ForwardScanDirection);
	if (!HeapTupleIsValid(tuple))
		return false;

	tupdesc = RelationGetDescr(sestate->relset->rowid_rel);
	heap_deform_tuple(tuple, tupdesc, values, nulls);
	Assert(!nulls[0] && !nulls[1]);

	oldctx = MemoryContextSwitchTo(sestate->es_context);

	cs_rowid = DatumGetInt64(values[0]);
	cs_rowmap = DatumGetVarBitPCopy(values[1]);

	MemoryContextSwitchTo(oldctx);

	chunk = pgstrom_read_chunk_buffer(sestate, cs_rowid, cs_rowmap);

	oldctx = MemoryContextSwitchTo(sestate->es_context);
	sestate->chunk_list = lappend(sestate->chunk_list, chunk);
	MemoryContextSwitchTo(oldctx);

	return true;
}

static PgStromExecState *
pgstrom_init_exec_state(ForeignScanState *fss)
{
	ForeignScan		   *fscan = (ForeignScan *) fss->ss.ps.plan;
	PgStromExecState   *sestate;
	ListCell		   *l;

	sestate = palloc0(sizeof(PgStromExecState));
	sestate->cols_needed = NULL;
	sestate->with_syscols = fscan->fsSystemCol;

	foreach (l, fscan->fdwplan->fdw_private)
	{
		DefElem	   *defel = (DefElem *)lfirst(l);

		if (strcmp(defel->defname, "cols_needed") == 0)
		{
			int		csidx = (intVal(defel->arg) - 1);

			if (csidx < 0)
			{
				Assert(fscan->fsSystemCol);
				continue;
			}
			sestate->cols_needed = bms_add_member(sestate->cols_needed, csidx);
		}
		else
			elog(ERROR, "pg_strom: unexpected private plan information: %s",
				 defel->defname);
	}

	sestate->chunk_list = NIL;
	sestate->chunk_curr = NULL;
	sestate->chunk_index = 0;

	return sestate;
}

void
pgstrom_begin_foreign_scan(ForeignScanState *fss, int eflags)
{
	PgStromExecState   *sestate;

	/*
	 * Do nothing for EXPLAIN or ANALYZE cases
	 */
	if (eflags & EXEC_FLAG_EXPLAIN_ONLY)
		return;

	sestate = pgstrom_init_exec_state(fss);
	sestate->relset = pgstrom_open_relation_set(fss->ss.ss_currentRelation,
												AccessShareLock, true);

	/*
	 * Begin the rowid scan 
	 */
	sestate->es_snapshot = fss->ss.ps.state->es_snapshot;
	sestate->es_context = fss->ss.ps.state->es_query_cxt;
	sestate->es_scan = heap_beginscan(sestate->relset->rowid_rel,
									  sestate->es_snapshot,
									  0, NULL);
	fss->fdw_state = sestate;
}

TupleTableSlot*
pgstrom_iterate_foreign_scan(ForeignScanState *fss)
{
	PgStromExecState   *sestate = (PgStromExecState *)fss->fdw_state;
	PgStromChunkBuf	   *chunk;
	TupleTableSlot	   *slot = fss->ss.ss_ScanTupleSlot;
	int					index, csidx;

	ExecClearTuple(slot);

retry:
	if (sestate->chunk_list == NIL)
	{
		if (!pgstrom_load_next_chunk_buffer(sestate))
			return slot;
		sestate->chunk_curr = list_head(sestate->chunk_list);
		sestate->chunk_index = 0;
	}
	else
	{
		chunk = lfirst(sestate->chunk_curr);
		if (sestate->chunk_index == VARBITLEN(chunk->rowmap))
		{
			if (!lnext(sestate->chunk_curr))
			{
				if (!pgstrom_load_next_chunk_buffer(sestate))
					return slot;
			}
			sestate->chunk_curr = lnext(sestate->chunk_curr);
			sestate->chunk_index = 0;
		}
	}

	chunk = lfirst(sestate->chunk_curr);
	index = sestate->chunk_index++;
	Assert(index < VARBITLEN(chunk->rowmap));
	if (VARBITS(chunk->rowmap)[index >> 3] & (1<<(index & (BITS_PER_BYTE-1))))
	{
		for (csidx=0; csidx < chunk->nattrs; csidx++)
		{
			if (chunk->cs_values[csidx] &&
				(!chunk->cs_nulls[csidx] ||
				 (chunk->cs_nulls[csidx][index>>3] &
				  (1<<(index & (BITS_PER_BYTE-1)))) == 0))
			{
				Form_pg_attribute	attr
					= slot->tts_tupleDescriptor->attrs[csidx];
				slot->tts_isnull[csidx] = false;
				if (attr->attlen > 0)
					slot->tts_values[csidx] =
						fetchatt(attr, ((char *)chunk->cs_values[csidx] +
										index * attr->attlen));
				else
					slot->tts_values[csidx] =
						(Datum)(((bytea **)chunk->cs_values[csidx])[index]);
			}
			else
			{
				slot->tts_isnull[csidx] = true;
				slot->tts_values[csidx] = (Datum) 0;
			}
		}
		ExecStoreVirtualTuple(slot);
	}
	else
		goto retry;

	return slot;
}

void
pgboost_rescan_foreign_scan(ForeignScanState *fss)
{
	PgStromExecState   *sestate = (PgStromExecState *)fss->fdw_state;

	if (sestate->chunk_list)
	{
		sestate->chunk_curr = list_head(sestate->chunk_list);
		sestate->chunk_index = 0;
	}
}

void
pgboost_end_foreign_scan(ForeignScanState *fss)
{
	PgStromExecState   *sestate = (PgStromExecState *)fss->fdw_state;
	PgStromChunkBuf	   *chunk;
	ListCell		   *cell;

	/* if sestate is NULL, we are in EXPLAIN; nothing to do */
	if (!sestate)
		return;

	/*
	 * End the rowid scan
	 */
	heap_endscan(sestate->es_scan);

	foreach (cell, sestate->chunk_list)
	{
		chunk = (PgStromChunkBuf *) lfirst(cell);

		pgstrom_release_chunk_buffer(chunk);
	}
	pgstrom_close_relation_set(sestate->relset, AccessShareLock);
}
