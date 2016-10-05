/*
 * datastore.c
 *
 * Routines to manage data store; row-store, column-store, toast-buffer,
 * and param-buffer.
 * ----
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */
#include "postgres.h"
#include "access/relscan.h"
#include "catalog/catalog.h"
#include "catalog/pg_tablespace.h"
#include "catalog/pg_type.h"
#include "optimizer/cost.h"
#include "storage/bufmgr.h"
#include "storage/fd.h"
#include "storage/predicate.h"
#include "utils/builtins.h"
#include "utils/bytea.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/tqual.h"
#include "pg_strom.h"
#include "cuda_numeric.h"
#include <float.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

/*
 * GUC variables
 */
static int		pgstrom_chunk_size_kb;
static int		pgstrom_chunk_limit_kb = INT_MAX;

/*
 * pgstrom_chunk_size - configured chunk size
 */
Size
pgstrom_chunk_size(void)
{
	return ((Size)pgstrom_chunk_size_kb) << 10;
}

static bool
check_guc_chunk_size(int *newval, void **extra, GucSource source)
{
	if (*newval > pgstrom_chunk_limit_kb)
	{
		GUC_check_errdetail("pg_strom.chunk_size = %d, is larger than "
							"pg_strom.chunk_limit = %d",
							*newval, pgstrom_chunk_limit_kb);
		return false;
	}
	return true;
}

/*
 * pgstrom_chunk_size_limit
 */
Size
pgstrom_chunk_size_limit(void)
{
	return ((Size)pgstrom_chunk_limit_kb) << 10;
}

static bool
check_guc_chunk_limit(int *newval, void **extra, GucSource source)
{
	if (*newval < pgstrom_chunk_size_kb)
	{
		GUC_check_errdetail("pg_strom.chunk_limit = %d, is less than "
							"pg_strom.chunk_size = %d",
							*newval, pgstrom_chunk_size_kb);
	}
	return true;
}

/*
 * pgstrom_bulk_exec_supported - returns true, if supplied planstate
 * supports bulk execution mode.
 */
bool
pgstrom_bulk_exec_supported(const PlanState *planstate)
{
	if (pgstrom_plan_is_gpuscan(planstate->plan) ||
        pgstrom_plan_is_gpujoin(planstate->plan) ||
        pgstrom_plan_is_gpupreagg(planstate->plan) ||
        pgstrom_plan_is_gpusort(planstate->plan))
	{
		GpuTaskState   *gts = (GpuTaskState *) planstate;

		if (gts->cb_bulk_exec != NULL)
			return true;
	}
	return false;
}

/*
 * estimate_num_chunks
 *
 * it estimates number of chunks to be fetched from the supplied Path
 */
cl_uint
estimate_num_chunks(Path *pathnode)
{
	RelOptInfo *rel = pathnode->parent;
	int			ncols = list_length(rel->reltargetlist);
    Size        htup_size;
	cl_uint		num_chunks;

	htup_size = MAXALIGN(offsetof(HeapTupleHeaderData,
								  t_bits[BITMAPLEN(ncols)]));
	if (rel->reloptkind != RELOPT_BASEREL)
		htup_size += MAXALIGN(rel->width);
	else
	{
		double      heap_size = (double)
			(BLCKSZ - SizeOfPageHeaderData) * rel->pages;

		htup_size += MAXALIGN(heap_size / Max(rel->tuples, 1.0) -
							  sizeof(ItemIdData) - SizeofHeapTupleHeader);
	}
	num_chunks = (cl_uint)
		((double)(htup_size + sizeof(cl_int)) * pathnode->rows /
		 (double)(pgstrom_chunk_size() -
				  STROMALIGN(offsetof(kern_data_store, colmeta[ncols]))));
	num_chunks = Max(num_chunks, 1);

	return num_chunks;
}

/*
 * BulkExecProcNode
 *
 * It runs the underlying sub-plan managed by PG-Strom in bulk-execution
 * mode. Caller can expect the data-store shall be filled up by the rows
 * read from the sub-plan.
 */
pgstrom_data_store *
BulkExecProcNode(GpuTaskState *gts, size_t chunk_size)
{
	PlanState		   *plannode = &gts->css.ss.ps;
	pgstrom_data_store *pds;

	CHECK_FOR_INTERRUPTS();

	if (plannode->chgParam != NULL)			/* If something changed, */
		ExecReScan(&gts->css.ss.ps);		/* let ReScan handle this */

	Assert(IsA(gts, CustomScanState));		/* rough checks */
	if (gts->cb_bulk_exec)
	{
		/* must provide our own instrumentation support */
		if (plannode->instrument)
			InstrStartNode(plannode->instrument);
		/* execution per chunk */
		pds = gts->cb_bulk_exec(gts, chunk_size);

		/* must provide our own instrumentation support */
		if (plannode->instrument)
			InstrStopNode(plannode->instrument,
						  !pds ? 0.0 : (double)pds->kds->nitems);
		Assert(!pds || pds->kds->nitems > 0);
		return pds;
	}
	elog(ERROR, "Bug? exec_chunk callback was not implemented");
}

bool
kern_fetch_data_store(TupleTableSlot *slot,
					  kern_data_store *kds,
					  size_t row_index,
					  HeapTuple tuple)
{
	if (row_index >= kds->nitems)
		return false;	/* out of range */

	/* in case of KDS_FORMAT_ROW */
	if (kds->format == KDS_FORMAT_ROW)
	{
		kern_tupitem   *tup_item = KERN_DATA_STORE_TUPITEM(kds, row_index);

		ExecClearTuple(slot);
		tuple->t_len = tup_item->t_len;
		tuple->t_self = tup_item->t_self;
		//tuple->t_tableOid = InvalidOid;
		tuple->t_data = &tup_item->htup;

		ExecStoreTuple(tuple, slot, InvalidBuffer, false);

		return true;
	}
	/* in case of KDS_FORMAT_SLOT */
	if (kds->format == KDS_FORMAT_SLOT)
	{
		Datum  *tts_values = (Datum *)KERN_DATA_STORE_VALUES(kds, row_index);
		bool   *tts_isnull = (bool *)KERN_DATA_STORE_ISNULL(kds, row_index);
		int		natts = slot->tts_tupleDescriptor->natts;

		memcpy(slot->tts_values, tts_values, sizeof(Datum) * natts);
		memcpy(slot->tts_isnull, tts_isnull, sizeof(bool) * natts);
#ifdef NOT_USED
		/*
		 * XXX - pointer reference is better than memcpy from performance
		 * perspectives, however, we need to ensure tts_values/tts_isnull
		 * shall be restored when pgstrom-data-store is released.
		 * It will be cause of complicated / invisible bugs.
		 */
		slot->tts_values = tts_values;
		slot->tts_isnull = tts_isnull;
#endif
		ExecStoreVirtualTuple(slot);
		return true;
	}
	elog(ERROR, "Bug? unexpected data-store format: %d", kds->format);
	return false;
}

bool
pgstrom_fetch_data_store(TupleTableSlot *slot,
						 pgstrom_data_store *pds,
						 size_t row_index,
						 HeapTuple tuple)
{
	return kern_fetch_data_store(slot, pds->kds, row_index, tuple);
}

pgstrom_data_store *
PDS_retain(pgstrom_data_store *pds)
{
	Assert(pds->refcnt > 0);

	pds->refcnt++;

	return pds;
}

void
PDS_release(pgstrom_data_store *pds)
{
	Assert(pds->refcnt > 0);
	if (--pds->refcnt == 0)
	{
		/* detach from the GpuContext */
		if (pds->pds_chain.prev && pds->pds_chain.next)
		{
			dlist_delete(&pds->pds_chain);
			memset(&pds->pds_chain, 0, sizeof(dlist_node));
		}
		/* release body of the data store */
		pfree(pds->kds);
		pfree(pds);
	}
}

void
init_kernel_data_store(kern_data_store *kds,
					   TupleDesc tupdesc,
					   Size length,
					   int format,
					   uint nrooms,
					   bool use_internal)
{
	int		i, attcacheoff;

	memset(kds, 0, offsetof(kern_data_store, colmeta));
	kds->hostptr = (hostptr_t) &kds->hostptr;
	kds->length = length;
	kds->usage = 0;
	kds->ncols = tupdesc->natts;
	kds->nitems = 0;
	kds->nrooms = nrooms;
	kds->format = format;
	kds->tdhasoid = tupdesc->tdhasoid;
	kds->tdtypeid = tupdesc->tdtypeid;
	kds->tdtypmod = tupdesc->tdtypmod;
	kds->table_oid = InvalidOid;	/* caller shall set */
	kds->nslots = 0;				/* caller shall set, if any */
	kds->hash_min = 0;
	kds->hash_max = UINT_MAX;

	attcacheoff = offsetof(HeapTupleHeaderData, t_bits);
	if (tupdesc->tdhasoid)
		attcacheoff += sizeof(Oid);
	attcacheoff = MAXALIGN(attcacheoff);

	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupdesc->attrs[i];
		int		attalign = typealign_get_width(attr->attalign);
		bool	attbyval = attr->attbyval;
		int		attlen = attr->attlen;

		if (!attr->attbyval)
			kds->has_notbyval = true;
		if (attr->atttypid == NUMERICOID)
		{
			kds->has_numeric = true;
			if (use_internal)
			{
				attbyval = true;
				attlen = sizeof(cl_long);
			}
		}

		if (attcacheoff > 0)
		{
			if (attlen > 0)
				attcacheoff = TYPEALIGN(attalign, attcacheoff);
			else
				attcacheoff = -1;	/* no more shortcut any more */
		}
		kds->colmeta[i].attbyval = attbyval;
		kds->colmeta[i].attalign = attalign;
		kds->colmeta[i].attlen = attlen;
		kds->colmeta[i].attnum = attr->attnum;
		kds->colmeta[i].attcacheoff = attcacheoff;
		kds->colmeta[i].atttypid = (cl_uint)attr->atttypid;
		kds->colmeta[i].atttypmod = (cl_int)attr->atttypmod;
		if (attcacheoff >= 0)
			attcacheoff += attr->attlen;
		/*
		 * !!don't forget to update pl_cuda.c if kern_colmeta layout would
		 * be updated !!
		 */
	}
}

void
PDS_expand_size(GpuContext *gcontext,
				pgstrom_data_store *pds,
				Size kds_length_new)
{
	kern_data_store	   *kds_old = pds->kds;
	kern_data_store	   *kds_new;
	Size				kds_length_old = kds_old->length;
	Size				kds_usage = kds_old->usage;
	cl_uint				i, nitems = kds_old->nitems;

	/* sanity checks */
	Assert(pds->kds_length == kds_length_old);
	Assert(kds_old->format == KDS_FORMAT_ROW ||
		   kds_old->format == KDS_FORMAT_HASH);
	Assert(kds_old->nslots == 0);

	/* no need to expand? */
	if (kds_length_old >= kds_length_new)
		return;

	kds_new = MemoryContextAllocHuge(gcontext->memcxt,
									 kds_length_new);
	memcpy(kds_new, kds_old, KERN_DATA_STORE_HEAD_LENGTH(kds_old));
	kds_new->hostptr = (hostptr_t)&kds_new->hostptr;
	kds_new->length = kds_length_new;

	/*
	 * Move the contents to new buffer from the old one
	 */
	if (kds_new->format == KDS_FORMAT_ROW ||
		kds_new->format == KDS_FORMAT_HASH)
	{
		cl_uint	   *row_index_old = KERN_DATA_STORE_ROWINDEX(kds_old);
		cl_uint	   *row_index_new = KERN_DATA_STORE_ROWINDEX(kds_new);
		size_t		shift = STROMALIGN_DOWN(kds_length_new - kds_length_old);
		size_t		offset = kds_length_old - kds_usage;

		/*
		 * If supplied new nslots is too big, larger than the expanded,
		 * it does not make sense to expand the buffer.
		 */
		if ((kds_new->format == KDS_FORMAT_HASH
			 ? KDS_CALCULATE_HASH_LENGTH(kds_new->ncols,
										 kds_new->nitems,
										 kds_new->usage)
			 : KDS_CALCULATE_ROW_LENGTH(kds_new->ncols,
										kds_new->nitems,
										kds_new->usage)) >= kds_new->length)
			elog(ERROR, "New nslots consumed larger than expanded");

		memcpy((char *)kds_new + offset + shift,
			   (char *)kds_old + offset,
			   kds_length_old - offset);
		for (i = 0; i < nitems; i++)
			row_index_new[i] = row_index_old[i] + shift;
	}
	else if (kds_new->format == KDS_FORMAT_SLOT)
	{
		/*
		 * We cannot expand KDS_FORMAT_SLOT with extra area because we don't
		 * know the way to fix pointers that reference the extra area.
		 */
		if (kds_new->usage > 0)
			elog(ERROR, "cannot expand KDS_FORMAT_SLOT with extra area");
		/* copy the values/isnull pair */
		memcpy(KERN_DATA_STORE_BODY(kds_new),
			   KERN_DATA_STORE_BODY(kds_old),
			   KERN_DATA_STORE_SLOT_LENGTH(kds_old, kds_old->nitems));
	}
	else
		elog(ERROR, "unexpected KDS format: %d", kds_new->format);
	/* old KDS is no longer referenced */
	pfree(kds_old);

	/* update length and kernel buffer */
	pds->kds_length = kds_length_new;
	pds->kds = kds_new;
}

void
PDS_shrink_size(pgstrom_data_store *pds)
{
	kern_data_store	   *kds = pds->kds;
	size_t				new_length;

	if (kds->format == KDS_FORMAT_ROW ||
		kds->format == KDS_FORMAT_HASH)
	{
		cl_uint	   *hash_slot = KERN_DATA_STORE_HASHSLOT(kds);
		cl_uint	   *row_index = KERN_DATA_STORE_ROWINDEX(kds);
		cl_uint		i, nslots = kds->nslots;
		size_t		shift;
		size_t		headsz;
		char	   *baseptr;

		/* small shift has less advantage than CPU cycle consumption */
		shift = kds->length - (kds->format == KDS_FORMAT_HASH
							   ? KDS_CALCULATE_HASH_LENGTH(kds->ncols,
														   kds->nitems,
														   kds->usage)
							   : KDS_CALCULATE_ROW_LENGTH(kds->ncols,
														  kds->nitems,
														  kds->usage));
		shift = STROMALIGN_DOWN(shift);

		if (shift < BLCKSZ || shift < sizeof(Datum) * kds->nitems)
			return;

		/* move the kern_tupitem / kern_hashitem */
		headsz = (kds->format == KDS_FORMAT_HASH
				  ? KDS_CALCULATE_HASH_FRONTLEN(kds->ncols, kds->nitems)
				  : KDS_CALCULATE_ROW_FRONTLEN(kds->ncols, kds->nitems));
		baseptr = (char *)kds + headsz;
		memmove(baseptr, baseptr + shift, kds->length - (headsz + shift));

		/* clear the hash slot once */
		if (nslots > 0)
		{
			Assert(kds->format == KDS_FORMAT_HASH);
			memset(hash_slot, 0, sizeof(cl_uint) * nslots);
		}

		/* adjust row_index and hash_slot */
		for (i=0; i < kds->nitems; i++)
		{
			row_index[i] -= shift;
			if (nslots > 0)
			{
				kern_hashitem  *khitem = KERN_DATA_STORE_HASHITEM(kds, i);
				cl_uint			khindex;

				Assert(khitem->rowid == i);
				khindex = khitem->hash % nslots;
                khitem->next = hash_slot[khindex];
                hash_slot[khindex] = (uintptr_t)khitem - (uintptr_t)kds;
			}
		}
		new_length = kds->length - shift;
	}
	else if (kds->format == KDS_FORMAT_SLOT)
	{
		new_length = KERN_DATA_STORE_SLOT_LENGTH(kds, kds->nitems);

		/*
		 * We cannot know which datum references the extra area with
		 * reasonable cost. So, prohibit it simply. We don't use SLOT
		 * format for data source, so usually no matter.
		 */
		if (kds->usage > 0)
			elog(ERROR, "cannot shirink KDS_SLOT with extra region");
	}
	else
		elog(ERROR, "Bug? unexpected PDS to be shrinked");

	Assert(new_length <= kds->length);
	kds->length = new_length;
	pds->kds_length = new_length;
}

pgstrom_data_store *
PDS_create_row(GpuContext *gcontext, TupleDesc tupdesc, Size length)
{
	pgstrom_data_store *pds;
	MemoryContext	gmcxt = gcontext->memcxt;

	/* allocation of pds */
	pds = MemoryContextAllocZero(gmcxt, sizeof(pgstrom_data_store));
	pds->refcnt = 1;	/* owned by the caller at least */

	/* allocation of kds */
	pds->kds_length = STROMALIGN_DOWN(length);
	pds->kds = MemoryContextAllocHuge(gmcxt, pds->kds_length);

	/*
	 * initialize common part of kds. Note that row-format cannot
	 * determine 'nrooms' preliminary, so INT_MAX instead.
	 */
	init_kernel_data_store(pds->kds, tupdesc, pds->kds_length,
						   KDS_FORMAT_ROW, INT_MAX, false);

	/* OK, it is now tracked by GpuContext */
	dlist_push_tail(&gcontext->pds_list, &pds->pds_chain);

	return pds;
}

pgstrom_data_store *
PDS_create_slot(GpuContext *gcontext,
				TupleDesc tupdesc,
				cl_uint nrooms,
				Size extra_length,
				bool use_internal)
{
	pgstrom_data_store *pds;
	size_t			kds_length;
	MemoryContext	gmcxt = gcontext->memcxt;

	/* allocation of pds */
	pds = MemoryContextAllocZero(gmcxt, sizeof(pgstrom_data_store));
	pds->refcnt = 1;	/* owned by the caller at least */

	/* allocation of kds */
	kds_length = (STROMALIGN(offsetof(kern_data_store,
									  colmeta[tupdesc->natts])) +
				  STROMALIGN(LONGALIGN((sizeof(Datum) + sizeof(char)) *
									   tupdesc->natts) * nrooms));
	kds_length += STROMALIGN(extra_length);

	pds->kds_length = kds_length;
	pds->kds = MemoryContextAllocHuge(gmcxt, pds->kds_length);

	init_kernel_data_store(pds->kds, tupdesc, pds->kds_length,
						   KDS_FORMAT_SLOT, nrooms, use_internal);

	/* OK, now it is tracked by GpuContext */
	dlist_push_tail(&gcontext->pds_list, &pds->pds_chain);

	return pds;
}

pgstrom_data_store *
PDS_create_hash(GpuContext *gcontext,
				TupleDesc tupdesc,
				Size length)
{
	pgstrom_data_store *pds;

	if (KDS_CALCULATE_HEAD_LENGTH(tupdesc->natts) > length)
		elog(ERROR, "Required length for KDS-Hash is too short");

	/*
	 * KDS_FORMAT_HASH has almost same initialization to KDS_FORMAT_ROW,
	 * so we once create it as _row format, then fixup the pds/kds.
	 */
	pds = PDS_create_row(gcontext, tupdesc, length);
	pds->kds->format = KDS_FORMAT_HASH;
	Assert(pds->kds->nslots == 0);	/* to be set later */

	return pds;
}

int
PDS_insert_block(pgstrom_data_store *pds,
				 Relation rel, BlockNumber blknum,
				 Snapshot snapshot,
				 BufferAccessStrategy strategy)
{
	kern_data_store	*kds = pds->kds;
	Buffer			buffer;
	Page			page;
	int				lines;
	int				ntup;
	OffsetNumber	lineoff;
	ItemId			lpp;
	uint		   *tup_index;
	kern_tupitem   *tup_item;
	bool			all_visible;
	Size			max_consume;

	/* only row-store can block read */
	Assert(kds->format == KDS_FORMAT_ROW && kds->nslots == 0);

	CHECK_FOR_INTERRUPTS();

	/* Load the target buffer */
	//buffer = ReadBuffer(rel, blknum);
	buffer = ReadBufferExtended(rel, MAIN_FORKNUM, blknum,
								RBM_NORMAL, strategy);

#if 1
	/* Just like heapgetpage(), however, jobs we focus on is OLAP
	 * workload, so it's uncertain whether we should vacuum the page
	 * here.
	 */
	heap_page_prune_opt(rel, buffer);
#endif

	/* we will check tuple's visibility under the shared lock */
	LockBuffer(buffer, BUFFER_LOCK_SHARE);
	page = (Page) BufferGetPage(buffer);
	lines = PageGetMaxOffsetNumber(page);
	ntup = 0;

	/*
	 * Check whether we have enough rooms to store expected number of
	 * tuples on the remaining space. If it is hopeless to load all
	 * the items in a block, we inform the caller this block shall be
	 * loaded on the next data store.
	 */
	max_consume = KDS_CALCULATE_HASH_LENGTH(kds->ncols,
											kds->nitems + lines,
											offsetof(kern_tupitem,
													 htup) * lines +
											BLCKSZ + kds->usage);
	if (max_consume > kds->length)
	{
		UnlockReleaseBuffer(buffer);
		return -1;
	}

	/*
	 * Logic is almost same as heapgetpage() doing.
	 */
	all_visible = PageIsAllVisible(page) && !snapshot->takenDuringRecovery;

	/* TODO: make SerializationNeededForRead() an external function
	 * on the core side. It kills necessity of setting up HeapTupleData
	 * when all_visible and non-serialized transaction.
	 */
	tup_index = KERN_DATA_STORE_ROWINDEX(kds) + kds->nitems;
	for (lineoff = FirstOffsetNumber, lpp = PageGetItemId(page, lineoff);
		 lineoff <= lines;
		 lineoff++, lpp++)
	{
		HeapTupleData	tup;
		bool			valid;

		if (!ItemIdIsNormal(lpp))
			continue;

		tup.t_tableOid = RelationGetRelid(rel);
		tup.t_data = (HeapTupleHeader) PageGetItem((Page) page, lpp);
		tup.t_len = ItemIdGetLength(lpp);
		ItemPointerSet(&tup.t_self, blknum, lineoff);

		if (all_visible)
			valid = true;
		else
			valid = HeapTupleSatisfiesVisibility(&tup, snapshot, buffer);

		CheckForSerializableConflictOut(valid, rel, &tup, buffer, snapshot);
		if (!valid)
			continue;

		/* put tuple */
		kds->usage += LONGALIGN(offsetof(kern_tupitem, htup) + tup.t_len);
		tup_item = (kern_tupitem *)((char *)kds + kds->length - kds->usage);
		tup_index[ntup] = (uintptr_t)tup_item - (uintptr_t)kds;
		tup_item->t_len = tup.t_len;
		tup_item->t_self = tup.t_self;
		memcpy(&tup_item->htup, tup.t_data, tup.t_len);

		ntup++;
	}
	UnlockReleaseBuffer(buffer);
	Assert(ntup <= MaxHeapTuplesPerPage);
	Assert(kds->nitems + ntup <= kds->nrooms);
	kds->nitems += ntup;

	return ntup;
}

/*
 * PDS_insert_tuple
 *
 * It inserts a tuple on the data store. Unlike block read mode, we can use
 * this interface for both of row and column data store.
 */
bool
PDS_insert_tuple(pgstrom_data_store *pds, TupleTableSlot *slot)
{
	kern_data_store	   *kds = pds->kds;
	size_t				required;
	HeapTuple			tuple;
	cl_uint			   *tup_index;
	kern_tupitem	   *tup_item;

	Assert(pds->kds_length == kds->length);

	/* No room to store a new kern_rowitem? */
	if (kds->nitems >= kds->nrooms)
		return false;
	Assert(kds->ncols == slot->tts_tupleDescriptor->natts);

	if (kds->format != KDS_FORMAT_ROW)
		elog(ERROR, "Bug? unexpected data-store format: %d", kds->format);

	/* OK, put a record */
	tup_index = KERN_DATA_STORE_ROWINDEX(kds);

	/* reference a HeapTuple in TupleTableSlot */
	tuple = ExecFetchSlotTuple(slot);

	/* check whether we have room for this tuple */
	required = LONGALIGN(offsetof(kern_tupitem, htup) + tuple->t_len);
	if (KDS_CALCULATE_ROW_LENGTH(kds->ncols,
								 kds->nitems + 1,
								 required + kds->usage) > kds->length)
		return false;

	kds->usage += required;
	tup_item = (kern_tupitem *)((char *)kds + kds->length - kds->usage);
	tup_item->t_len = tuple->t_len;
	tup_item->t_self = tuple->t_self;
	memcpy(&tup_item->htup, tuple->t_data, tuple->t_len);
	tup_index[kds->nitems++] = (uintptr_t)tup_item - (uintptr_t)kds;

	return true;
}


/*
 * PDS_insert_hashitem
 *
 * It inserts a tuple to the data store of hash format.
 */
bool
PDS_insert_hashitem(pgstrom_data_store *pds,
					TupleTableSlot *slot,
					cl_uint hash_value)
{
	kern_data_store	   *kds = pds->kds;
	cl_uint			   *row_index = KERN_DATA_STORE_ROWINDEX(kds);
	Size				required;
	HeapTuple			tuple;
	kern_hashitem	   *khitem;

	Assert(pds->kds_length == kds->length);

	/* No room to store a new kern_hashitem? */
	if (kds->nitems >= kds->nrooms)
		return false;
	Assert(kds->ncols == slot->tts_tupleDescriptor->natts);

	/* KDS has to be KDS_FORMAT_HASH */
	if (kds->format != KDS_FORMAT_HASH)
		elog(ERROR, "Bug? unexpected data-store format: %d", kds->format);

	/* compute required length */
	tuple = ExecFetchSlotTuple(slot);
	required = MAXALIGN(offsetof(kern_hashitem, t.htup) + tuple->t_len);

	Assert(kds->usage == MAXALIGN(kds->usage));
	if (KDS_CALCULATE_HASH_LENGTH(kds->ncols,
								  kds->nitems + 1,
								  required + kds->usage) > pds->kds_length)
		return false;	/* no more space to put */

	/* OK, put a tuple */
	Assert(kds->usage == MAXALIGN(kds->usage));
	khitem = (kern_hashitem *)((char *)kds + kds->length
							   - (kds->usage + required));
	kds->usage += required;
	khitem->hash = hash_value;
	khitem->next = 0x7f7f7f7f;	/* to be set later */
	khitem->rowid = kds->nitems++;
	khitem->t.t_len = tuple->t_len;
	khitem->t.t_self = tuple->t_self;
	memcpy(&khitem->t.htup, tuple->t_data, tuple->t_len);

	row_index[khitem->rowid] = (cl_uint)((uintptr_t)&khitem->t.t_len -
										 (uintptr_t)kds);
	return true;
}

/*
 * PDS_build_hashtable
 *
 * construct hash table according to the current contents
 */
void
PDS_build_hashtable(pgstrom_data_store *pds)
{
	kern_data_store *kds = pds->kds;
	cl_uint		   *row_index = KERN_DATA_STORE_ROWINDEX(kds);
	cl_uint		   *hash_slot = KERN_DATA_STORE_HASHSLOT(kds);
	cl_uint			i, j, nslots = __KDS_NSLOTS(kds->nitems);

	if (kds->format != KDS_FORMAT_HASH)
		elog(ERROR, "Bug? Only KDS_FORMAT_HASH can build a hash table");
	if (kds->nslots > 0)
		elog(ERROR, "Bug? hash table is already built");

	memset(hash_slot, 0, sizeof(cl_uint) * nslots);
	for (i = 0; i < kds->nitems; i++)
	{
		kern_hashitem  *khitem = (kern_hashitem *)
			((char *)kds + row_index[i] - offsetof(kern_hashitem, t));

		Assert(khitem->rowid == i);
		j = khitem->hash % nslots;
		khitem->next = hash_slot[j];
		hash_slot[j] = (uintptr_t)khitem - (uintptr_t)kds;
	}
	kds->nslots = nslots;
}

void
pgstrom_init_datastore(void)
{
	DefineCustomIntVariable("pg_strom.chunk_size",
							"default size of pgstrom_data_store",
							NULL,
							&pgstrom_chunk_size_kb,
							15872,
							4096,
							MAX_KILOBYTES,
							PGC_USERSET,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							check_guc_chunk_size, NULL, NULL);
	DefineCustomIntVariable("pg_strom.chunk_limit",
							"limit size of pgstrom_data_store",
							NULL,
							&pgstrom_chunk_limit_kb,
							5 * pgstrom_chunk_size_kb,
							4096,
							MAX_KILOBYTES,
							PGC_USERSET,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							check_guc_chunk_limit, NULL, NULL);
}
