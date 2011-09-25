/*
 * scan.c
 *
 * routines to scan shared memory buffer
 *
 * Copyright (C) 2011 - 2012 KaiGai Kohei <kaigai@kaigai.gr.jp>
 */
#include "postgres.h"
#include "pg_boost.h"



static void
pgboost_release_chunk(scan_state_t *scan_state, scan_chunk_t *scan_chunk)
{
	ListCell   *cell;

	foreach (cell, scan_chunk->buffer_list)
	{
		ReleaseBuffer((Buffer) lfirst(cell));
	}
	scan_chunk->buffer_list = NIL;
}

static inline Oid
pgboost_get_attribute_typeid(Relation relation, AttrNumber attnum)
{
	switch (attnum)
	{
		case TableOidAttributeNumber:
			return OIDOID;
		case MaxCommandIdAttributeNumber:
		case MinCommandIdAttributeNumber:
			return CIDOID;
		case MinTransactionIdAttributeNumber:
		case MaxTransactionIdAttributeNumber:
			return XIDOID;
		case ObjectIdAttributeNumber:
			return OIDOID;
		case SelfItemPointerAttributeNumber:
			return TIDOID;
		default:
			if (attnum > 0 &&
				attnum <= RelationGetNumberOfAttributes(relation))
			{
				Form_pg_attribute	attr
					= RelationGetDescr(relation)->attrs[attnum - 1];

				return attr->atttypid;
			}
			break;
	}
	elog(ERROR, "pgboost: attribute %d of \"%s\" is out of range",
		 attnum, RelationGetRelationName(relation));
	return InvalidOid;	/* be compiler quiet */
}


static scan_chunk_t *
pgboost_alloc_chunk(Relation foreign_rel, scan_state_t *scan_state)
{
	scan_chunk_t   *chunk_curr = scan_state->chunk_curr;
	scan_chunk_t   *chunk_next;
	int32			chunk_size = scan_state->chunk_size;
	int				nattrs = (scan_state->max_attr - scan_state->min_attr + 1);

	if (!chunk_curr)
	{
		/*
		 * A corner case: if the underlying table was empty, we don't need
		 * to assign any chunks, and returns NULL immediately.
		 */
		if (scan_state->scan->rs_nblocks == 0)
			return NULL;

		blkno_begin = scan_state->scan->rs_startblock;
	}
	else
	{
		blkno_begin = chunk_curr->blkno_end + 1;

		if (blkno_begin >= scan_state->scan->rs_nblocks)
			blkno_begin = 0;

		if (blkno_begin == scan_state->scan->rs_startblock)
			return NULL;
	}

	/*
	 * Construct a new chunk
	 */
	chunk_next = palloc(sizeof(scan_chunk_t));
	chunk_next->blkno_begin = blkno_begin;
	chunk_next->blkno_end = 0;		/* to be set later */
	chunk_next->buffer_list = NIL;
	chunk_next->num_tuples = 0;
	chunk_next->num_valid = 0;
	chunk_next->valid = pgboost_vector_alloc(BOOLOID, chunk_size);
	chunk_next->nulls = palloc(sizeof(vector_t *) * nattrs);
	chunk_next->attrs = palloc(sizeof(vector_t *) * nattrs);

	for (i=0; i < nattrs; i++)
	{
		Oid			typeId;

		if (!bms_is_member(i, scan_state->referenced))
		{
			chunk_next->nulls[i] = NULL;
			chunk_next->attrs[i] = NULL;
			continue;
		}
		typeId = pgboost_get_attribute_typeid(foreign_rel,
											  i + scan_state->min_attr);
		chunk_next->nulls[i] = pgboost_vector_alloc(BOOLOID, chunk_size);
		chunk_next->attrs[i] = pgboost_vector_alloc(typeId, chunk_size);
	}
	return chunk_next;
}

static inline void
pgboost_fetch_tuple(scan_state_t *scan_state, HeapTuple tuple)
{
	scan_chunk_t   *chunk = scan_state->chunk_curr;
	AttrNumber		min_attr = scan_state->min_attr;
	AttrNumber		max_attr = scan_state->max_attr;
	AttrNumber		fattno;
	AttrNumber		uattno;
	int32			index = chunk->num_tuples;
	vector_t	   *vec;
	Datum			datum;
	bool			isnull;

	for (fattno = min_attr; fattno <= max_attr; fattno++)
	{
		if (!bms_is_member(fattno, scan_state->referenced))
		{
			Assert(chunk->nulls[fattno - min_attr] == NULL);
			Assert(chunk->attrs[fattno - min_attr] == NULL);
			continue;
		}
		uattno = scan_state->map_attr[fattno - min_attr];

		datum = heap_getattr(tuple, attno,
							 RelationGetDescr(scan_state->scan->rs_rd),
							 &isnull);
		if (isnull)
		{
			vec = chunk->nulls[fattno - min_attr];
			datum = BoolGetDatum(true);
		}
		else
			vec = chunk->attrs[fattno - min_attr];

		pgboost_vector_setval(vec, chunk->num_tuples, datum);
	}
	chunk->num_tuples++;
}

static bool
pgboost_scan_chunk(Relation foreign_rel, scan_state_t *scan_state)
{
	HeapScanDesc   *scan_desc = scan_state->scan;
	scan_chunk_t   *scan_chunk = scan_state->chunk_curr;
	Relation		relation = scan_state->scan->rs_rd;
	Snapshot		snapshot = scan_desc->rs_snapshot;
	BlockNumber		blkno_curr;
	Buffer			buffer;
	Page			page;
	OffsetNumber	lineoff;
	ItemId			lpp;
	int				lines;
	int				ntuples;
	bool			all_visible;

	blkno_curr = scan_chunk->blkno_begin;
	Assert(blkno_curr < scan_desc->rs_nblocks);

	while (true)
	{
		buffer = ReadBufferExtended(scan_desc->rs_rd,
									MAIN_FORKNUM,
									blkno_curr,
									RBM_NORMAL,
									scan_desc->rs_strategy);
		LockBuffer(buffer, BUFFER_LOCK_SHARE);

		page = (Page) BufferGetPage(buffer);

		/*
		 * No more slots on this chunk, so stop to scan anymore.
		 */
		lines = PageGetMaxOffsetNumber(page);
		if (scan_chunk->num_tuples + lines > scan_state->chunk_size)
		{
			LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
			ReleaseBuffer(buffer);
			break;
		}

		/*
		 * Read buffer to chunk
		 */
		scan_chunk->blkno_end = chunk_curr;
		scan_chunk->buffer_list = lappend(scan_chunk->buffer_list, buffer);

		all_visible = (PageIsAllVisible(page) &&
					   !snapshot->takenDuringRecovery);

		for (lineoff = FirstOffsetNumber, lpp = PageGetItemId(page, lineoff);
			 lineoff <= lines;
			 lineoff++, lpp++)
		{
			if (ItemIdIsNormal(lpp))
			{
				HeapTupleData	tup;
				bool			valid;

				tup.t_data = (HeapTupleHeader) PageGetItem((Page) page, lpp);
				tup.t_len = ItemIdGetLength(lpp);
				ItemPointerSet(&(tup.t_self), page, lineoff);

				if (all_visible)
					valid = true;
				else
					valid = HeapTupleSatisfiesVisibility(&tup,
														 snapshot,
														 buffer);

				CheckForSerializableConflictOut(valid, relation, &tup,
												buffer, snapshot);
				if (valid)
					pgboost_fetch_tuple();
			}
		}
		LockBuffer(buffer, BUFFER_LOCK_UNLOCK);

		/*
		 * Pick up next page
		 */
		blkno_curr++;
		if (blkno_curr >= scan_desc->rs_nblocks)
			blkno_curr = 0;
		if (blkno_curr == scan_desc->rs_startblock)
			break;
	}





}

static Oid
pgboost_parse_options(Oid foreigntableId,
					  int32 *p_scan_chunk_size)
{
	ForeignTable   *ft = GetForeignTable(foreigntableId);
	ListCell	   *cell;
	Oid				relation_id = InvalidOid;
	int32			scan_chunk_size = 48000;

	foreach (cell, ft->options)
	{
		DefElem	   *def = (DefElem *) lfirst(cell);

		if (strcmp(def->defname, "relation_id") == 0)
		{
			relation_id = defGetInt64(def);
		}
		else if (strcmp(def->defname, "scan_chunk_size") == 0)
		{
			scan_chunk_size = defGetInt64(def);
		}
		else
			elog(ERROR,
				 "pg_boost: \"%s\" is not supported option", def->defname);
	}
	if (!OidIsValid(relation_id))
		elog(ERROR, "pg_boost: \"relation_id\" options was not found");

	*p_scan_chunk_size = scan_chunk_size;

	return relation_id;
}

static void
pgboost_verify_underlying_relation(Relation foreign_rel,
								   scan_state_t *scan_state)
{
	Relation	relation = scan_state->scan->rs_rd;
	AttrNumber *map_attr;
	AttrNumber	i, j;

	map_attr = palloc(sizeof(AttrNumber) *
					  RelationGetNumberOfAttributes(foreign_rel));
	for (i = 0; i < RelationGetNumberOfAttributes(foreign_rel); i++)
	{
		Form_pg_attribute	fattr = RelationGetDescr(foreign_rel)->attrs[i];

		for (j = 0; j < RelationGetNumberOfAttributes(relation); j++)
		{
			Form_pg_attribute	uattr
				= RelationGetDescr(relation)->attrs[j];

			if (uattr->attisdropped)
				continue;

			if (strcmp(NameStr(fattr->attname), NameStr(uattr->attname)) == 0)
			{
				if (fattr->atttypid != uattr->atttypid)
					ereport(ERROR,
							(errcode(ERRCODE_FDW_INVALID_DATA_TYPE),
							 errmsg("pgboost: table \"%s\" is not compatible",
									RelationGetRelationName(relation))));
				map_attr[i] = j;
				break;
			}
		}
		if (j == RelationGetNumberOfAttributes(relation))
			ereport(ERROR,
					(errcode(ERRCODE_FDW_INVALID_COLUMN_NAME),
					 errmsg("pgboost: table \"%s\" is not compatible",
							RelationGetRelationName(relation))));
	}
	scan_state->map_attr = map_attr;
}

void
pgboost_begin_foreign_scan(ForeignScanState *fss, int eflags)
{
	List		   *plan_info = ((FdwPlan *)(fss->ss.ps.plan))->fdw_private;
	scan_state_t   *scan_state = palloc(sizeof(scan_state_t));
	EState		   *estate = fss->ss.ps.state;
	Oid				relOid;
	int32			scan_chunk_size;
	Relation		foreign_rel;
	Relation		relation;
	AttrNumber		attnum;

	foreign_rel = node->ss.ss_currentRelation;

	relOid = pgboost_parse_options(RelationGetRelid(foreign_rel),
								   &scan_chunk_size);
	/*
	 * XXX - Permission checks on the underlying relation should be
	 * added here. Just call ExecCheckRTPerms() with list of RTE.
	 */

	/*
	 * Open the underlying relation for read-only scan
	 */
	relation = heap_open(relOid, AccessShareLock);

	scan_state->scan = heap_beginscan(relation, estate->es_snapshot, 0, NULL);

	/*
	 * Extract plan info from the private member
	 */
	pgboost_extract_plan_info(scan_state, plan_info);

	/*
	 * check whether the underlying relation is compatible structure
	 * with supported data type, and set up scan_state->map_attr, if OK.
	 */
	pgboost_verify_underlying_relation(foreign_rel, scan_state);

	scan_state->scan_chunk_size = scan_chunk_size;
	scan_state->chunk_list = NIL;
	scan_state->chunk_curr = NULL;
	scan_state->index_curr = 0;

	ExecAssignScanType((ScanState *)node, RelationGetDescr(foreign_rel));

	fss->fdw_state = scan_state;
}

TupleTableSlot*
pgboost_iterate_foreign_scan(ForeignScanState *fss)
{
	scan_state_t   *scan_state = fss->fdw_state;
	scan_chunk_t   *chunk_curr = scan_state->chunk_curr;

retry:
	if (!curr_chunk || scan_state->index_curr == chunk_curr->num_tuples)
	{
		Relation		foreign_rel = fss->ss.ss_currentRelation;
		scan_chunk_t   *chunk_next;

		if (chunk_curr)
			pgboost_release_chunk(scan_state, chunk_curr);

		chunk_next = pgboost_alloc_chunk(foreign_rel, scan_state);
		if (!chunk_next)
			goto not_found;
		scan_state->chunk_list = lappend(scan_state->chunk_list, chunk_next);
		scan_state->chunk_curr = chunk_next;

		pgboost_scan_chunk(foreign_rel, scan_state);

		/*
		 * Exec vectorilized qualifiers
		 */
		//if (scan_state->vquals)
		//	;
		scan_state->chunk_curr = chunk_next;
		scan_state->index_curr = 0;
	}

not_found:

}

void
pgboost_rescan_foreign_scan(ForeignScanState *fss)
{


}

void
pgboost_end_foreign_scan(ForeignScanState *fss)
{


}
