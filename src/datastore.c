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
#include "access/visibilitymap.h"
#include "catalog/catalog.h"
#include "catalog/pg_tablespace.h"
#include "catalog/pg_type.h"
#include "optimizer/cost.h"
#include "storage/bufmgr.h"
#include "storage/buf_internals.h"
#include "storage/fd.h"
#include "storage/predicate.h"
#include "storage/smgr.h"
#include "utils/builtins.h"
#include "utils/bytea.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/tqual.h"
#include "pg_strom.h"
#include "cuda_numeric.h"
#include "nvme_strom.h"
#include <float.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

/*
 * static variables
 */
static int		pgstrom_chunk_size_kb;
static int		pgstrom_chunk_limit_kb = INT_MAX;
static bool		debug_force_nvme_strom = false;
static long		sysconf_pagesize;		/* _SC_PAGESIZE */
static long		sysconf_phys_pages;		/* _SC_PHYS_PAGES */
static long		nvme_strom_threshold;

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
		pgstrom_plan_is_gpupreagg(planstate->plan))
//        pgstrom_plan_is_gpusort(planstate->plan))
	{
		GpuTaskState_v2	   *gts = (GpuTaskState_v2 *) planstate;

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
	int			ncols = list_length(rel->reltarget->exprs);
    Size        htup_size;
	cl_uint		num_chunks;

	htup_size = MAXALIGN(offsetof(HeapTupleHeaderData,
								  t_bits[BITMAPLEN(ncols)]));
	if (rel->reloptkind != RELOPT_BASEREL)
		htup_size += MAXALIGN(rel->reltarget->width);
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

#ifdef NOT_USED
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
	/* in case of KDS_FORMAT_BLOCK */
	if (kds->format == KDS_FORMAT_BLOCK)
	{
		/* upper 16bits are block index */
		/* lower 16bits are lineitem */



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
	return kern_fetch_data_store(slot, &pds->kds, row_index, tuple);
}
#endif

/*
 * PDS_fetch_tuple - fetch a tuple from the PDS
 */
static inline bool
KDS_fetch_tuple_row(TupleTableSlot *slot,
					kern_data_store *kds,
					GpuTaskState_v2 *gts)
{
	if (gts->curr_index < kds->nitems)
	{
		size_t			row_index = gts->curr_index++;
		Relation		rel = gts->css.ss.ss_currentRelation;
		kern_tupitem   *tup_item;
		HeapTuple		tuple = &gts->curr_tuple;

		tup_item = KERN_DATA_STORE_TUPITEM(kds, row_index);
		ExecClearTuple(slot);
		tuple->t_len  = tup_item->t_len;
		tuple->t_self = tup_item->t_self;
		tuple->t_tableOid = (rel ? RelationGetRelid(rel) : InvalidOid);
		tuple->t_data = &tup_item->htup;

		ExecStoreTuple(tuple, slot, InvalidBuffer, false);

		return true;
	}
	return false;
}

static inline bool
KDS_fetch_tuple_slot(TupleTableSlot *slot,
					 kern_data_store *kds,
					 GpuTaskState_v2 *gts)
{
	if (gts->curr_index < kds->nitems)
	{
		size_t	row_index = gts->curr_index++;
		Datum  *tts_values = KERN_DATA_STORE_VALUES(kds, row_index);
		bool   *tts_isnull = KERN_DATA_STORE_ISNULL(kds, row_index);
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
	return false;
}

static inline bool
KDS_fetch_tuple_block(TupleTableSlot *slot,
					  kern_data_store *kds,
					  GpuTaskState_v2 *gts)
{
	Relation	rel = gts->css.ss.ss_currentRelation;
	HeapTuple	tuple = &gts->curr_tuple;
	BlockNumber	block_nr;
	PageHeader	hpage;
	cl_uint		max_lp_index;
	ItemId		lpp;

	while (gts->curr_index < kds->nitems)
	{
		block_nr = KERN_DATA_STORE_BLOCK_BLCKNR(kds, gts->curr_index);
		hpage = KERN_DATA_STORE_BLOCK_PGPAGE(kds, gts->curr_index);
		Assert(PageIsAllVisible(hpage));
		max_lp_index = PageGetMaxOffsetNumber(hpage);
		while (gts->curr_lp_index < max_lp_index)
		{
			cl_uint		lp_index = gts->curr_lp_index++;

			lpp = &hpage->pd_linp[lp_index];
			if (!ItemIdIsNormal(lpp))
				continue;

			tuple->t_len = ItemIdGetLength(lpp);
			BlockIdSet(&tuple->t_self.ip_blkid, block_nr);
			tuple->t_self.ip_posid = lp_index;
			tuple->t_tableOid = (rel ? RelationGetRelid(rel) : InvalidOid);
			tuple->t_data = (HeapTupleHeader)((char *)hpage +
											  ItemIdGetOffset(lpp));
			ExecStoreTuple(tuple, slot, InvalidBuffer, false);
			return true;
		}
		/* move to the next block */
		gts->curr_index++;
		gts->curr_lp_index = 0;
	}
	return false;	/* end of the PDS */
}

bool
PDS_fetch_tuple(TupleTableSlot *slot,
				pgstrom_data_store *pds,
				GpuTaskState_v2 *gts)
{
	switch (pds->kds.format)
	{
		case KDS_FORMAT_ROW:
		case KDS_FORMAT_HASH:
			return KDS_fetch_tuple_row(slot, &pds->kds, gts);
		case KDS_FORMAT_SLOT:
			return KDS_fetch_tuple_slot(slot, &pds->kds, gts);
		case KDS_FORMAT_BLOCK:
			return KDS_fetch_tuple_block(slot, &pds->kds, gts);
		default:
			elog(ERROR, "Bug? unsupported data store format: %d",
				pds->kds.format);
	}
}

/*
 * PDS_retain
 */
pgstrom_data_store *
PDS_retain(pgstrom_data_store *pds)
{
	int32		refcnt_old	__attribute__((unused));

	refcnt_old = (int32)pg_atomic_fetch_add_u32(&pds->refcnt, 1);

	Assert(refcnt_old > 0);

	return pds;
}

/*
 * PDS_release
 */
void
PDS_release(pgstrom_data_store *pds)
{
	int32		refcnt_new;

	refcnt_new = (int32)pg_atomic_sub_fetch_u32(&pds->refcnt, 1);
	Assert(refcnt_new >= 0);
	if (refcnt_new == 0)
		dmaBufferFree(pds);
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
	kds->nrows_per_block = 0;

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

pgstrom_data_store *
PDS_expand_size(GpuContext_v2 *gcontext,
				pgstrom_data_store *pds_old,
				Size kds_length_new)
{
	pgstrom_data_store *pds_new;
	Size		kds_length_old = pds_old->kds.length;
	Size		kds_usage = pds_old->kds.usage;
	cl_uint		i, nitems = pds_old->kds.nitems;

	/* sanity checks */
	Assert(pds_old->kds.format == KDS_FORMAT_ROW ||
		   pds_old->kds.format == KDS_FORMAT_HASH);
	Assert(pds_old->kds.nslots == 0);

	/* no need to expand? */
	kds_length_new = STROMALIGN_DOWN(kds_length_new);
	if (pds_old->kds.length >= kds_length_new)
		return pds_old;

	pds_new = dmaBufferAlloc(gcontext, offsetof(pgstrom_data_store,
												kds) + kds_length_new);
	memcpy(pds_new, pds_old, (offsetof(pgstrom_data_store, kds) +
							  KERN_DATA_STORE_HEAD_LENGTH(&pds_old->kds)));
	pds_new->kds.hostptr = (hostptr_t)&pds_new->kds.hostptr;
	pds_new->kds.length  = kds_length_new;

	/*
	 * Move the contents to new buffer from the old one
	 */
	if (pds_new->kds.format == KDS_FORMAT_ROW ||
		pds_new->kds.format == KDS_FORMAT_HASH)
	{
		cl_uint	   *row_index_old = KERN_DATA_STORE_ROWINDEX(&pds_old->kds);
		cl_uint	   *row_index_new = KERN_DATA_STORE_ROWINDEX(&pds_new->kds);
		size_t		shift = STROMALIGN_DOWN(kds_length_new - kds_length_old);
		size_t		offset = kds_length_old - kds_usage;

		/*
		 * If supplied new nslots is too big, larger than the expanded,
		 * it does not make sense to expand the buffer.
		 */
		if ((pds_new->kds.format == KDS_FORMAT_HASH
			 ? KDS_CALCULATE_HASH_LENGTH(pds_new->kds.ncols,
										 pds_new->kds.nitems,
										 pds_new->kds.usage)
			 : KDS_CALCULATE_ROW_LENGTH(pds_new->kds.ncols,
										pds_new->kds.nitems,
										pds_new->kds.usage)) >= kds_length_new)
			elog(ERROR, "New nslots consumed larger than expanded");

		memcpy((char *)&pds_new->kds + offset + shift,
			   (char *)&pds_old->kds + offset,
			   kds_length_old - offset);
		for (i = 0; i < nitems; i++)
			row_index_new[i] = row_index_old[i] + shift;
	}
	else if (pds_new->kds.format == KDS_FORMAT_SLOT)
	{
		/*
		 * We cannot expand KDS_FORMAT_SLOT with extra area because we don't
		 * know the way to fix pointers that reference the extra area.
		 */
		if (pds_new->kds.usage > 0)
			elog(ERROR, "cannot expand KDS_FORMAT_SLOT with extra area");
		/* copy the values/isnull pair */
		memcpy(KERN_DATA_STORE_BODY(&pds_new->kds),
			   KERN_DATA_STORE_BODY(&pds_old->kds),
			   KERN_DATA_STORE_SLOT_LENGTH(&pds_old->kds,
										   pds_old->kds.nitems));
	}
	else
		elog(ERROR, "unexpected KDS format: %d", pds_new->kds.format);

	/* release the old PDS, and return the new one */
	dmaBufferFree(pds_old);
	return pds_new;
}

void
PDS_shrink_size(pgstrom_data_store *pds)
{
	kern_data_store	   *kds = &pds->kds;
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
}

pgstrom_data_store *
PDS_create_row(GpuContext_v2 *gcontext, TupleDesc tupdesc, Size length)
{
	pgstrom_data_store *pds;
	Size		kds_length = STROMALIGN_DOWN(length);

	pds = dmaBufferAlloc(gcontext, offsetof(pgstrom_data_store,
											kds) + kds_length);
	/* owned by the caller at least */
	pg_atomic_init_u32(&pds->refcnt, 1);

	/*
	 * initialize common part of KDS. Note that row-format cannot
	 * determine 'nrooms' preliminary, so INT_MAX instead.
	 */
	init_kernel_data_store(&pds->kds, tupdesc, kds_length,
						   KDS_FORMAT_ROW, INT_MAX, false);
	pds->nblocks_uncached = 0;
	pds->ntasks_running = 0;

	return pds;
}

pgstrom_data_store *
PDS_create_slot(GpuContext_v2 *gcontext,
				TupleDesc tupdesc,
				cl_uint nrooms,
				Size extra_length,
				bool use_internal)
{
	pgstrom_data_store *pds;
	size_t			kds_length;

	kds_length = (STROMALIGN(offsetof(kern_data_store,
									  colmeta[tupdesc->natts])) +
				  STROMALIGN(LONGALIGN((sizeof(Datum) + sizeof(char)) *
									   tupdesc->natts) * nrooms) +
				  STROMALIGN(extra_length));
	pds = dmaBufferAlloc(gcontext, offsetof(pgstrom_data_store,
											kds) + kds_length);
	/* owned by the caller at least */
	pg_atomic_init_u32(&pds->refcnt, 1);

	init_kernel_data_store(&pds->kds, tupdesc, kds_length,
						   KDS_FORMAT_SLOT, nrooms, use_internal);
	pds->nblocks_uncached = 0;
	pds->ntasks_running = 0;

	return pds;
}

pgstrom_data_store *
PDS_duplicate_slot(GpuContext_v2 *gcontext,
				   kern_data_store *kds_head,
				   cl_uint nrooms,
				   cl_uint extra_unitsz)
{
	pgstrom_data_store *pds;
	size_t			required;

	required = (STROMALIGN(offsetof(kern_data_store,
									colmeta[kds_head->ncols])) +
				STROMALIGN((sizeof(Datum) + sizeof(char)) *
						   kds_head->ncols) * nrooms +
				STROMALIGN(extra_unitsz) * nrooms);

	pds = dmaBufferAlloc(gcontext, offsetof(pgstrom_data_store,
											kds) + required);
	/* owned by the caller at least */
	pg_atomic_init_u32(&pds->refcnt, 1);
	pds->ntasks_running = 0;

	/* setup KDS using the template */
	memcpy(&pds->kds, kds_head,
		   offsetof(kern_data_store,
					colmeta[kds_head->ncols]));
	pds->kds.hostptr = (hostptr_t)&pds->kds.hostptr;
	pds->kds.length  = required;
	pds->kds.usage   = 0;
	pds->kds.nrooms  = nrooms;
	pds->kds.nitems  = 0;

	return pds;
}

pgstrom_data_store *
PDS_create_hash(GpuContext_v2 *gcontext,
				TupleDesc tupdesc,
				Size length)
{
	pgstrom_data_store *pds;
	Size		kds_length = STROMALIGN_DOWN(length);

	if (KDS_CALCULATE_HEAD_LENGTH(tupdesc->natts) > kds_length)
		elog(ERROR, "Required length for KDS-Hash is too short");

	pds = dmaBufferAlloc(gcontext, offsetof(pgstrom_data_store,
											kds) + kds_length);
	/* owned by the caller at least */
	pg_atomic_init_u32(&pds->refcnt, 1);

	init_kernel_data_store(&pds->kds, tupdesc, kds_length,
						   KDS_FORMAT_HASH, INT_MAX, false);
	pds->nblocks_uncached = 0;
	pds->ntasks_running = 0;

	return pds;
}

pgstrom_data_store *
PDS_create_block(GpuContext_v2 *gcontext,
				 TupleDesc tupdesc,
				 NVMEScanState *nvme_sstate)
{
	pgstrom_data_store *pds;
	cl_uint		nrooms = nvme_sstate->nblocks_per_chunk;
	Size		kds_length;

	kds_length = KDS_CALCULATE_HEAD_LENGTH(tupdesc->natts)
		+ STROMALIGN(sizeof(BlockNumber) * nrooms)
		+ BLCKSZ * nrooms;
	if (offsetof(pgstrom_data_store, kds) + kds_length > pgstrom_chunk_size())
		elog(WARNING,
			 "Bug? PDS length (%zu) is larger than pg_strom.chunk_size(%zu)",
			 offsetof(pgstrom_data_store, kds) + kds_length,
			 pgstrom_chunk_size());

	/* allocation */
	pds = dmaBufferAlloc(gcontext, offsetof(pgstrom_data_store,
											kds) + kds_length);
	/* owned by the caller at least */
	pg_atomic_init_u32(&pds->refcnt, 1);

	init_kernel_data_store(&pds->kds, tupdesc, kds_length,
						   KDS_FORMAT_BLOCK, nrooms, false);
	pds->kds.nrows_per_block = nvme_sstate->nrows_per_block;
	pds->nblocks_uncached = 0;
	pds->ntasks_running = 0;

	return pds;
}

/*
 * support for bulkload onto ROW/BLOCK format
 */
/* see storage/smgr/md.c */
typedef struct _MdfdVec
{
	File			mdfd_vfd;		/* fd number in fd.c's pool */
	BlockNumber		mdfd_segno;		/* segment number, from 0 */
	struct _MdfdVec *mdfd_chain;	/* next segment, or NULL */
} MdfdVec;

/*
 * PDS_init_heapscan_state - construct a per-query state for heap-scan
 * with KDS_FORMAT_BLOCK / NVMe-Strom.
 */
void
PDS_init_heapscan_state(GpuTaskState_v2 *gts,
						cl_uint nrows_per_block)
{
	Relation		relation = gts->css.ss.ss_currentRelation;
	TupleDesc		tupdesc = RelationGetDescr(relation);
	EState		   *estate = gts->css.ss.ps.state;
	BlockNumber		nr_blocks;
	BlockNumber		nr_segs;
	MdfdVec		   *vec;
	NVMEScanState  *nvme_sstate;
	cl_uint			nrooms_max;
	cl_uint			nchunks;
	cl_uint			nblocks_per_chunk;
	cl_uint			i;

	/*
	 * Raw-block scan is valuable only when NVMe-Strom is configured,
	 * except for debugging.
	 */
	if (!debug_force_nvme_strom &&
		!RelationCanUseNvmeStrom(relation))
		return;

	/*
	 * NOTE: RelationGetNumberOfBlocks() has a significant side-effect.
	 * It opens all the underlying files of MAIN_FORKNUM, then set @rd_smgr
	 * of the relation.
	 * It allows extension to touch file descriptors without invocation of
	 * ReadBuffer().
	 */
	nr_blocks = RelationGetNumberOfBlocks(relation);
	if (!debug_force_nvme_strom &&
		nr_blocks < nvme_strom_threshold)
		return;

	/*
	 * Calculation of an optimal number of data-blocks for each PDS.
	 *
	 * We intend to load maximum available blocks onto the PDS which has
	 * configured chunk size, however, it will lead unbalanced smaller
	 * chunk around the bound of storage file segment.
	 * (Typically, 32 of 4091blocks + 1 of 160 blocks)
	 * So, we will adjust @nblocks_per_chunk to balance chunk size all
	 * around the relation scan.
	 */
	nrooms_max = (pgstrom_chunk_size() -
				  KDS_CALCULATE_HEAD_LENGTH(tupdesc->natts))
		/ (sizeof(BlockNumber) + BLCKSZ);
	while (KDS_CALCULATE_HEAD_LENGTH(tupdesc->natts) +
		   STROMALIGN(sizeof(BlockNumber) * nrooms_max) +
		   BLCKSZ * nrooms_max > pgstrom_chunk_size())
		nrooms_max--;
	if (nrooms_max < 1)
		return;

	nchunks = (RELSEG_SIZE + nrooms_max - 1) / nrooms_max;
	nblocks_per_chunk = (RELSEG_SIZE + nchunks - 1) / nchunks;

	/* allocation of NVMEScanState structure */
	nr_segs = (nr_blocks + (BlockNumber) RELSEG_SIZE - 1) / RELSEG_SIZE;
	nvme_sstate = MemoryContextAlloc(estate->es_query_cxt,
									 offsetof(NVMEScanState, mdfd[nr_segs]));
	memset(nvme_sstate, -1, offsetof(NVMEScanState, mdfd[nr_segs]));
	nvme_sstate->nrows_per_block = nrows_per_block;
	nvme_sstate->nblocks_per_chunk = nblocks_per_chunk;
	nvme_sstate->curr_segno = InvalidBlockNumber;
	nvme_sstate->curr_vmbuffer = InvalidBuffer;
	nvme_sstate->nr_segs = nr_segs;

	vec = relation->rd_smgr->md_fd[MAIN_FORKNUM];
	while (vec)
	{
		if (vec->mdfd_vfd < 0 ||
			vec->mdfd_segno >= nr_segs)
			elog(ERROR, "Bug? MdfdVec {vfd=%d segno=%u} is out of range",
				 vec->mdfd_vfd, vec->mdfd_segno);
		nvme_sstate->mdfd[vec->mdfd_segno].segno = vec->mdfd_segno;
		nvme_sstate->mdfd[vec->mdfd_segno].vfd   = vec->mdfd_vfd;
		vec = vec->mdfd_chain;
	}

	/* sanity checks */
	for (i=0; i < nr_segs; i++)
	{
		if (nvme_sstate->mdfd[i].segno >= nr_segs ||
			nvme_sstate->mdfd[i].vfd < 0)
			elog(ERROR, "Bug? Here is a hole segment which was not open");
	}
	gts->nvme_sstate = nvme_sstate;
}

/*
 * PDS_end_heapscan_state
 */
void
PDS_end_heapscan_state(GpuTaskState_v2 *gts)
{
	NVMEScanState   *nvme_sstate = gts->nvme_sstate;

	if (nvme_sstate)
	{
		/* release visibility map, if any */
		if (nvme_sstate->curr_vmbuffer != InvalidBuffer)
		{
			ReleaseBuffer(nvme_sstate->curr_vmbuffer);
			nvme_sstate->curr_vmbuffer = InvalidBuffer;
		}
		pfree(nvme_sstate);
		gts->nvme_sstate = NULL;
	}
}

/*
 * PDS_exec_heapscan_block - PDS scan for KDS_FORMAT_BLOCK format
 */
static bool
PDS_exec_heapscan_block(pgstrom_data_store *pds,
						Relation relation,
						HeapScanDesc hscan,
						NVMEScanState *nvme_sstate,
						int *p_filedesc)
{
	BlockNumber		blknum = hscan->rs_cblock;
	BlockNumber	   *block_nums;
	Snapshot		snapshot = hscan->rs_snapshot;
	BufferAccessStrategy strategy = hscan->rs_strategy;
	SMgrRelation	smgr = relation->rd_smgr;
	Buffer			buffer;
	Page			spage;
	Page			dpage;
	cl_uint			nr_loaded;
	bool			all_visible;

	/* PDS cannot eat any blocks more, obviously */
	if (pds->kds.nitems >= pds->kds.nrooms)
		return false;

	/* array of block numbers */
	block_nums = (BlockNumber *)KERN_DATA_STORE_BODY(&pds->kds);

	/*
	 * NVMe-Strom can be applied only when filesystem supports the feature,
	 * and the current source block is all-visible.
	 * Elsewhere, we will go fallback with synchronized buffer scan.
	 */
	if (RelationCanUseNvmeStrom(relation) &&
		VM_ALL_VISIBLE(relation, blknum,
					   &nvme_sstate->curr_vmbuffer))
	{
		BufferTag	newTag;
		uint32		newHash;
		LWLock	   *newPartitionLock = NULL;
		bool		retval;
		int			buf_id;

		/* create a tag so we can lookup the buffer */
		INIT_BUFFERTAG(newTag, smgr->smgr_rnode.node, MAIN_FORKNUM, blknum);
		/* determine its hash code and partition lock ID */
		newHash = BufTableHashCode(&newTag);
		newPartitionLock = BufMappingPartitionLock(newHash);

		/* check whether the block exists on the shared buffer? */
		LWLockAcquire(newPartitionLock, LW_SHARED);
		buf_id = BufTableLookup(&newTag, newHash);
		if (buf_id < 0)
		{
			BlockNumber	segno = blknum / RELSEG_SIZE;
			int			filedesc;

			Assert(segno < nvme_sstate->nr_segs);
			filedesc = FileGetRawDesc(nvme_sstate->mdfd[segno].vfd);

			/*
			 * We cannot mix up multiple source files in a single PDS chunk.
			 * If heapscan_block comes across segment boundary, rest of the
			 * blocks must be read on the next PDS chunk.
			 */
			if (*p_filedesc >= 0 && *p_filedesc != filedesc)
				retval = false;
			else
			{
				if (*p_filedesc < 0)
					*p_filedesc = filedesc;
				/* add uncached block for direct load */
				pds->nblocks_uncached++;
				pds->kds.nitems++;
				block_nums[pds->kds.nrooms - pds->nblocks_uncached] = blknum;

				retval = true;
			}
			LWLockRelease(newPartitionLock);
			return retval;
		}
		LWLockRelease(newPartitionLock);
	}
	/*
	 * Load the source buffer with synchronous read
	 */
	buffer = ReadBufferExtended(relation, MAIN_FORKNUM, blknum,
								RBM_NORMAL, strategy);
#if 1
	/* Just like heapgetpage(), however, jobs we focus on is OLAP
	 * workload, so it's uncertain whether we should vacuum the page
	 * here.
	 */
	heap_page_prune_opt(relation, buffer);
#endif
	/* we will check tuple's visibility under the shared lock */
	LockBuffer(buffer, BUFFER_LOCK_SHARE);
	nr_loaded = pds->kds.nitems - pds->nblocks_uncached;
	spage = (Page) BufferGetPage(buffer);
	dpage = (Page) KERN_DATA_STORE_BLOCK_PGPAGE(&pds->kds, nr_loaded);
	memcpy(dpage, spage, BLCKSZ);
	block_nums[nr_loaded] = blknum;

	/*
	 * Logic is almost same as heapgetpage() doing. We have to invalidate
	 * invisible tuples prior to GPU kernel execution, if not all-visible.
	 */
	all_visible = PageIsAllVisible(dpage) && !snapshot->takenDuringRecovery;
	if (!all_visible)
	{
		int				lines = PageGetMaxOffsetNumber(dpage);
		OffsetNumber	lineoff;
		ItemId			lpp;

		for (lineoff = FirstOffsetNumber, lpp = PageGetItemId(dpage, lineoff);
			 lineoff <= lines;
			 lineoff++, lpp++)
		{
			HeapTupleData	tup;
			bool			valid;

			if (!ItemIdIsNormal(lpp))
				continue;

			tup.t_tableOid = RelationGetRelid(relation);
			tup.t_data = (HeapTupleHeader) PageGetItem((Page) dpage, lpp);
			tup.t_len = ItemIdGetLength(lpp);
			ItemPointerSet(&tup.t_self, blknum, lineoff);

			valid = HeapTupleSatisfiesVisibility(&tup, snapshot, buffer);
			CheckForSerializableConflictOut(valid, relation, &tup,
											buffer, snapshot);
			if (!valid)
				ItemIdSetUnused(lpp);
		}
	}
	UnlockReleaseBuffer(buffer);
	/* dpage became all-visible also */
	PageSetAllVisible(dpage);
	pds->kds.nitems++;

	return true;
}

/*
 * PDS_exec_heapscan_row - PDS scan for KDS_FORMAT_ROW format
 */
static bool
PDS_exec_heapscan_row(pgstrom_data_store *pds,
					  Relation relation,
					  HeapScanDesc hscan)
{
	BlockNumber		blknum = hscan->rs_cblock;
	Snapshot		snapshot = hscan->rs_snapshot;
	BufferAccessStrategy strategy = hscan->rs_strategy;
	kern_data_store	*kds = &pds->kds;
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

	/* Load the target buffer */
	buffer = ReadBufferExtended(relation, MAIN_FORKNUM, blknum,
								RBM_NORMAL, strategy);

#if 1
	/* Just like heapgetpage(), however, jobs we focus on is OLAP
	 * workload, so it's uncertain whether we should vacuum the page
	 * here.
	 */
	heap_page_prune_opt(relation, buffer);
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
		return false;
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

		tup.t_tableOid = RelationGetRelid(relation);
		tup.t_data = (HeapTupleHeader) PageGetItem((Page) page, lpp);
		tup.t_len = ItemIdGetLength(lpp);
		ItemPointerSet(&tup.t_self, blknum, lineoff);

		if (all_visible)
			valid = true;
		else
			valid = HeapTupleSatisfiesVisibility(&tup, snapshot, buffer);

		CheckForSerializableConflictOut(valid, relation,
										&tup, buffer, snapshot);
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

	return true;
}

/*
 * PDS_exec_heapscan - PDS scan entrypoint
 */
bool
PDS_exec_heapscan(GpuTaskState_v2 *gts,
				  pgstrom_data_store *pds, int *p_filedesc)
{
	Relation		relation = gts->css.ss.ss_currentRelation;
	HeapScanDesc	hscan = gts->css.ss.ss_currentScanDesc;
	bool			retval;

	CHECK_FOR_INTERRUPTS();

	if (pds->kds.format == KDS_FORMAT_ROW)
		retval = PDS_exec_heapscan_row(pds, relation, hscan);
	else if (pds->kds.format == KDS_FORMAT_BLOCK)
	{
		Assert(gts->nvme_sstate);
		retval = PDS_exec_heapscan_block(pds, relation, hscan,
										 gts->nvme_sstate, p_filedesc);
	}
	else
		elog(ERROR, "Bug? unexpected PDS format: %d", pds->kds.format);

	return retval;
}

/*
 * PDS_insert_tuple
 *
 * It inserts a tuple onto the data store. Unlike block read mode, we cannot
 * use this API only for row-format.
 */
bool
PDS_insert_tuple(pgstrom_data_store *pds, TupleTableSlot *slot)
{
	kern_data_store	   *kds = &pds->kds;
	size_t				required;
	HeapTuple			tuple;
	cl_uint			   *tup_index;
	kern_tupitem	   *tup_item;

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
	kern_data_store	   *kds = &pds->kds;
	cl_uint			   *row_index = KERN_DATA_STORE_ROWINDEX(kds);
	Size				required;
	HeapTuple			tuple;
	kern_hashitem	   *khitem;

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
								  required + kds->usage) > pds->kds.length)
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
 * PDS_fillup_blocks
 *
 * It fills up uncached blocks using synchronous read APIs.
 */
void
PDS_fillup_blocks(pgstrom_data_store *pds, int file_desc)
{
	cl_int			i, nr_loaded;
	ssize_t			nbytes;
	char		   *dest_addr;
	loff_t			curr_fpos;
	size_t			curr_size;
	BlockNumber	   *block_nums;

	if (pds->kds.format != KDS_FORMAT_BLOCK)
		elog(ERROR, "Bug? only KDS_FORMAT_BLOCK can be filled up");

	if (pds->nblocks_uncached == 0)
		return;		/* already filled up */

	Assert(pds->nblocks_uncached <= pds->kds.nitems);
	nr_loaded = pds->kds.nitems - pds->nblocks_uncached;
	block_nums = (BlockNumber *)KERN_DATA_STORE_BODY(&pds->kds);
	dest_addr = (char *)KERN_DATA_STORE_BLOCK_PGPAGE(&pds->kds, nr_loaded);
	curr_fpos = 0;
	curr_size = 0;
	for (i=pds->nblocks_uncached-1; i >=0; i--)
	{
		loff_t	file_pos = (block_nums[i] & (RELSEG_SIZE - 1)) * BLCKSZ;

		if (curr_size > 0 &&
			curr_fpos + curr_size == file_pos)
		{
			/* merge with the pending i/o */
			curr_size += BLCKSZ;
		}
		else
		{
			while (curr_size > 0)
			{
				nbytes = pread(file_desc, dest_addr, curr_size, curr_fpos);
				Assert(nbytes <= curr_size);
				if (nbytes < 0 || (nbytes == 0 && errno != EINTR))
					elog(ERROR, "failed on pread(2): %m");
				dest_addr += nbytes;
				curr_fpos += nbytes;
				curr_size -= nbytes;
			}
			curr_fpos = file_pos;
			curr_size = BLCKSZ;
		}
	}

	while (curr_size > 0)
	{
		nbytes = pread(file_desc, dest_addr, curr_size, curr_fpos);
		Assert(nbytes <= curr_size);
		if (nbytes < 0 || (nbytes == 0 && errno != EINTR))
			elog(ERROR, "failed on pread(2): %m");
		dest_addr += nbytes;
		curr_fpos += nbytes;
		curr_size -= nbytes;
	}
	Assert(dest_addr == (char *)KERN_DATA_STORE_BLOCK_PGPAGE(&pds->kds,
															 pds->kds.nitems));
	pds->nblocks_uncached = 0;
}

/*
 * PDS_build_hashtable
 *
 * construct hash table according to the current contents
 */
void
PDS_build_hashtable(pgstrom_data_store *pds)
{
	kern_data_store *kds = &pds->kds;
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
	Size	shared_buffer_size = (Size)NBuffers * (Size)BLCKSZ;

	/* get system configuration */
	sysconf_pagesize = sysconf(_SC_PAGESIZE);
	if (sysconf_pagesize < 0)
		elog(ERROR, "failed on sysconf(_SC_PAGESIZE): %m");
	sysconf_phys_pages = sysconf(_SC_PHYS_PAGES);
	if (sysconf_phys_pages < 0)
		elog(ERROR, "failed on sysconf(_SC_PHYS_PAGES): %m");
	if (sysconf_pagesize * sysconf_phys_pages < shared_buffer_size)
		elog(ERROR, "Bug? shared_buffer is larger than system RAM");

	/*
	 * MEMO: Threshold of table's physical size to use NVMe-Strom:
	 *   ((System RAM size) -
	 *    (shared_buffer size)) * 0.67 + (shared_buffer size)
	 *
	 * If table size is enough large to issue real i/o, NVMe-Strom will
	 * make advantage by higher i/o performance.
	 */
	nvme_strom_threshold = ((sysconf_pagesize * sysconf_phys_pages -
							 shared_buffer_size) * 2 / 3 +
							shared_buffer_size) / BLCKSZ;

	/* init GUC variables */
	DefineCustomIntVariable("pg_strom.chunk_size",
							"default size of pgstrom_data_store",
							NULL,
							&pgstrom_chunk_size_kb,
							32768 - (2 * BLCKSZ / 1024),	/* almost 32MB */
							4096,
							MAX_KILOBYTES,
							PGC_INTERNAL,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							check_guc_chunk_size, NULL, NULL);
	DefineCustomIntVariable("pg_strom.chunk_limit",
							"limit size of pgstrom_data_store",
							NULL,
							&pgstrom_chunk_limit_kb,
							5 * pgstrom_chunk_size_kb,
							4096,
							MAX_KILOBYTES,
							PGC_INTERNAL,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_KB,
							check_guc_chunk_limit, NULL, NULL);
#if 1
	DefineCustomBoolVariable("pg_strom.debug_force_nvme_strom",
							 "(DEBUG) force to use raw block scan mode",
							 NULL,
							 &debug_force_nvme_strom,
							 false,
							 PGC_SUSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
#endif
}
