/*
 * datastore.c
 *
 * Routines to manage data store; row-store, column-store, toast-buffer,
 * and param-buffer.
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "access/relscan.h"
#include "access/sysattr.h"
#include "miscadmin.h"
#include "port.h"
#include "storage/bufmgr.h"
#include "storage/predicate.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/tqual.h"
#include "pg_strom.h"

/*
 * pgstrom_create_param_buffer
 *
 * It construct a param-buffer on the shared memory segment, according to
 * the supplied Const/Param list. Its initial reference counter is 1, so
 * this buffer can be released using pgstrom_put_param_buffer().
 */
kern_parambuf *
pgstrom_create_kern_parambuf(List *used_params,
							 ExprContext *econtext)
{
	StringInfoData	str;
	kern_parambuf  *kpbuf;
	char		padding[STROMALIGN_LEN];
	ListCell   *cell;
	Size		offset;
	int			index = 0;
	int			nparams = list_length(used_params);

	/* seek to the head of variable length field */
	offset = STROMALIGN(offsetof(kern_parambuf, poffset[nparams]));
	initStringInfo(&str);
	enlargeStringInfo(&str, offset);
	memset(str.data, 0, offset);
	str.len = offset;
	/* walks on the Para/Const list */
	foreach (cell, used_params)
	{
		Node   *node = lfirst(cell);

		if (IsA(node, Const))
		{
			Const  *con = (Const *) node;

			kpbuf = (kern_parambuf *)str.data;
			if (con->constisnull)
				kpbuf->poffset[index] = 0;	/* null */
			else
			{
				kpbuf->poffset[index] = str.len;
				if (con->constlen > 0)
					appendBinaryStringInfo(&str,
										   (char *)&con->constvalue,
										   con->constlen);
				else
					appendBinaryStringInfo(&str,
										   DatumGetPointer(con->constvalue),
										   VARSIZE(con->constvalue));
			}
		}
		else if (IsA(node, Param))
		{
			ParamListInfo param_info = econtext->ecxt_param_list_info;
			Param  *param = (Param *) node;

			if (param_info &&
				param->paramid > 0 && param->paramid <= param_info->numParams)
			{
				ParamExternData	*prm = &param_info->params[param->paramid - 1];

				/* give hook a chance in case parameter is dynamic */
				if (!OidIsValid(prm->ptype) && param_info->paramFetch != NULL)
					(*param_info->paramFetch) (param_info, param->paramid);

				kpbuf = (kern_parambuf *)str.data;
				if (!OidIsValid(prm->ptype))
				{
					elog(INFO, "debug: Param has no particular data type");
					kpbuf->poffset[index++] = 0;	/* null */
					continue;
				}
				/* safety check in case hook did something unexpected */
				if (prm->ptype != param->paramtype)
					ereport(ERROR,
							(errcode(ERRCODE_DATATYPE_MISMATCH),
							 errmsg("type of parameter %d (%s) does not match that when preparing the plan (%s)",
									param->paramid,
									format_type_be(prm->ptype),
									format_type_be(param->paramtype))));
				if (prm->isnull)
					kpbuf->poffset[index] = 0;	/* null */
				else
				{
					int		typlen = get_typlen(prm->ptype);

					if (typlen == 0)
						elog(ERROR, "cache lookup failed for type %u",
							 prm->ptype);
					if (typlen > 0)
						appendBinaryStringInfo(&str,
											   (char *)&prm->value,
											   typlen);
					else
						appendBinaryStringInfo(&str,
											   DatumGetPointer(prm->value),
											   VARSIZE(prm->value));
				}
			}
		}
		else
			elog(ERROR, "unexpected node: %s", nodeToString(node));

		/* alignment */
		if (STROMALIGN(str.len) != str.len)
			appendBinaryStringInfo(&str, padding,
								   STROMALIGN(str.len) - str.len);
		index++;
	}
	Assert(STROMALIGN(str.len) == str.len);
	kpbuf = (kern_parambuf *)str.data;
	kpbuf->length = str.len;
	kpbuf->nparams = nparams;

	return kpbuf;
}

/*
 * pgstrom_plan_can_multi_exec
 *
 * It gives a hint whether subplan support bulk-exec mode, or not.
 */
bool
pgstrom_planstate_can_bulkload(const PlanState *ps)
{
	if (IsA(ps, CustomPlanState))
	{
		const CustomPlanState  *cps = (const CustomPlanState *) ps;

		if (pgstrom_gpuscan_can_bulkload(cps))
			return true;
	}
	return false;
}

#if 0
/*
 * it makes a template of header portion of kern_data_store according
 * to the supplied tupdesc and bitmapset of referenced columns.
 */
bytea *
kparam_make_kds_head(TupleDesc tupdesc,
					 Bitmapset *referenced,
					 cl_uint nsyscols)
{
	kern_data_store	*kds_head;
	bytea	   *result;
	Size		length;
	int			i, j, ncols;

	/* allocation */
	ncols = tupdesc->natts + nsyscols;
	length = STROMALIGN(offsetof(kern_data_store, colmeta[ncols]));
	result = palloc0(VARHDRSZ + length);
	SET_VARSIZE(result, VARHDRSZ + length);

	kds_head = (kern_data_store *) VARDATA(result);
	kds_head->ncols = ncols;
	kds_head->nitems = (cl_uint)(-1);	/* to be set later */
	kds_head->nrooms = (cl_uint)(-1);	/* to be set later */

	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupdesc->attrs[i];

		j = attr->attnum - FirstLowInvalidHeapAttributeNumber;

		kds_head->colmeta[i].attnotnull = attr->attnotnull;
		kds_head->colmeta[i].attalign = typealign_get_width(attr->attalign);
		kds_head->colmeta[i].attlen = pgstrom_try_varlena_inline(attr);
		if (!bms_is_member(j, referenced))
			kds_head->colmeta[i].attvalid = 0;
		else
			kds_head->colmeta[i].attvalid = (cl_uint)(-1);
		/* rest of fields shall be set later */
	}
	return result;
}

void
kparam_refresh_kds_head(kern_parambuf *kparams,
						StromObject *rcstore,
						cl_uint nitems)
{
	kern_data_store *kds_head = KPARAM_GET_KDS_HEAD(kparams);
	Size		length;
	int			i, ncols = kds_head->ncols;

	length = STROMALIGN(offsetof(kern_data_store, colmeta[ncols]));
	kds_head->nitems = nitems;	/* usually, nitems and nrroms are same, */
	kds_head->nrooms = nitems;	/* if kds is filled in the host side */
	if (StromTagIs(rcstore, TCacheRowStore))
	{
		tcache_row_store *trs = (tcache_row_store *) rcstore;

		kds_head->column_form = false;
		for (i=0; i < ncols; i++)
		{
			/* put attribute number to reference row-data */
			if (!kds_head->colmeta[i].attvalid)
				continue;
			kds_head->colmeta[i].rs_attnum = i + 1;
		}
		length += STROMALIGN(trs->kern.length);
	}
	else if (StromTagIs(rcstore, TCacheColumnStore))
	{
		kds_head->column_form = true;
		for (i=0; i < ncols; i++)
		{
			if (!kds_head->colmeta[i].attvalid)
				continue;
			kds_head->colmeta[i].cs_offset = length;
			if (!kds_head->colmeta[i].attnotnull)
				length += STROMALIGN(BITMAPLEN(kds_head->nrooms));
			length += STROMALIGN((kds_head->colmeta[i].attlen > 0
								  ? kds_head->colmeta[i].attlen
								  : sizeof(cl_uint)) * kds_head->nrooms);
		}
	}
	else
		elog(ERROR, "bug? neither row- nor column-store");

	kds_head->length = length;
}

/*
 * kparam_make_ktoast_head
 *
 * it also makes header portion of kern_toastbuf according to the tupdesc.
 */
bytea *
kparam_make_ktoast_head(TupleDesc tupdesc, cl_uint nsyscols)
{
	kern_toastbuf *ktoast_head;
	bytea	   *result;
	Size		length;
	int			ncols;

	ncols = tupdesc->natts + nsyscols;
	length = STROMALIGN(offsetof(kern_toastbuf, coldir[ncols]));
	result = palloc0(VARHDRSZ + length);
	SET_VARSIZE(result, VARHDRSZ + length);

	ktoast_head = (kern_toastbuf *) VARDATA(result);
	ktoast_head->length = TOASTBUF_MAGIC;
	ktoast_head->ncols = ncols;
	/* individual coldir[] shall be set later */

	return result;
}

void
kparam_refresh_ktoast_head(kern_parambuf *kparams,
						   StromObject *rcstore)
{
	kern_data_store *kds_head = KPARAM_GET_KDS_HEAD(kparams);
	kern_toastbuf *ktoast_head = KPARAM_GET_KTOAST_HEAD(kparams);
	int			i;
	Size		offset;
	bool		has_toast = false;

	Assert(ktoast_head->length == TOASTBUF_MAGIC);
	Assert(ktoast_head->ncols == kds_head->ncols);
	offset = STROMALIGN(offsetof(kern_toastbuf,
								 coldir[ktoast_head->ncols]));
	for (i=0; i < ktoast_head->ncols; i++)
	{
		ktoast_head->coldir[i] = (cl_uint)(-1);

		/* column is not referenced */
		if (!kds_head->colmeta[i].attvalid)
			continue;
		/* fixed-length variables (incl. inlined varlena) */
		if (kds_head->colmeta[i].attlen > 0)
			continue;
		/* only column-store needs individual toast buffer */
		if (StromTagIs(rcstore, TCacheColumnStore))
		{
			tcache_column_store *tcs = (tcache_column_store *) rcstore;

			if (tcs->cdata[i].toast)
			{
				has_toast = true;
				ktoast_head->coldir[i] = offset;
                offset += STROMALIGN(tcs->cdata[i].toast->tbuf_length);
			}
		}
	}
	/* make KPARAM_1 as NULL, if no toast requirement */
	if (!has_toast)
		kparams->poffset[1] = 0;	/* mark it as null */
}
#endif



bool
pgstrom_fetch_data_store(TupleTableSlot *slot,
						 pgstrom_data_store *pds,
						 size_t row_index,
						 HeapTuple tuple)
{
	kern_data_store *kds = pds->kds;
	TupleDesc	tupdesc;
	cl_uint		cs_offset;
	char	   *cs_values;
	int			i, attlen;

	if (row_index >= kds->nitems)
		return false;	/* out of range */

	/* in case of row-store */
	if (!kds->is_column)
	{
		kern_rowitem   *ritem = KERN_DATA_STORE_ROWITEM(kds, row_index);
		Buffer			buffer;
		Page			page;
		ItemId			lpp;

		Assert(ritem->block_ofs < kds->nblocks);
		buffer = pds->blocks[ritem->block_ofs].buffer;
		page   = pds->blocks[ritem->block_ofs].page;
		lpp = PageGetItemId(page, ritem->item_ofs);
		Assert(ItemIdIsNormal(lpp));

		tuple->t_data = (HeapTupleHeader) PageGetItem(page, lpp);
		tuple->t_len = ItemIdGetLength(lpp);
		ItemPointerSet(&tuple->t_self, buffer, ritem->item_ofs);

		ExecStoreTuple(tuple, slot, buffer, false);

		return true;
	}

	/* otherwise, column store */
	ExecStoreAllNullTuple(slot);
	tupdesc = slot->tts_tupleDescriptor;
	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute	attr = tupdesc->attrs[i];

		if (i < kds->ncols && !kds->colmeta[i].attvalid)
		{
			cs_offset = kds->colmeta[i].cs_offset;
			if (!kds->colmeta[i].attnotnull)
			{
				char   *nullmap = (char *)kds + cs_offset;

				if (att_isnull(row_index, nullmap))
					continue;
				cs_offset += STROMALIGN(BITMAPLEN(kds->nrooms));
			}
			slot->tts_isnull[i] = false;
			cs_values = (char *)kds + cs_offset;

			attlen = kds->colmeta[i].attlen;
			if (attlen > 0)
			{
				cs_values += attlen * row_index;
				slot->tts_values[i] = fetch_att(cs_values,
												attr->attbyval,
												attr->attlen);
			}
			else
			{
				cl_uint		vl_offset = ((cl_uint *)cs_values)[row_index];
				char	   *vl_ptr = (char *)pds->ktoast + vl_offset;

				slot->tts_values[i] = PointerGetDatum(vl_ptr);
			}
		}
	}
	return true;
}

void
pgstrom_release_data_store(pgstrom_data_store *pds)
{
	kern_data_store	   *kds = pds->kds;
	int		i;

	Assert(kds->nblocks < pds->max_blocks);
	for (i=0; i < kds->nblocks; i++)
	{
		Page	page = pds->blocks[i].page;
		Buffer	buffer = pds->blocks[i].buffer;

		/*
		 * NOTE: A page buffer is assigned on the PG-Strom's shared memory,
		 * if referenced table is local / temporary relation (because its
		 * buffer is originally assigned on the private memory; invisible
		 * from OpenCL server process).
		 *
		 * Also note that, we don't need to call ReleaseBuffer() in case
		 * when the current context is OpenCL server or cleanup callback
		 * of resource-tracker.
		 * If pgstrom_release_data_store() is called on the OpenCL server
		 * context, it means the source transaction was already aborted
		 * thus the shared-buffers being mapped were already released.
		 * (It leads probability to send invalid region by DMA; so
		 * kern_get_datum_rs() takes careful data validation.)
		 * If pgstrom_release_data_store() is called under the cleanup
		 * callback context of resource-tracker, it means shared-buffers
		 * being pinned by the current transaction were already released
		 * by the built-in code prior to PG-Strom's cleanup. So, no need
		 * to do anything by ourselves.
		 */
		if (BufferIsLocal(buffer))
			pgstrom_shmem_free(page);
		else if (!pgstrom_i_am_clserv &&
				 !pgstrom_restrack_cleanup_context())
			ReleaseBuffer(buffer);
	}
	if (pds->ktoast)
		pgstrom_shmem_free(pds->ktoast);
	pgstrom_shmem_free(pds->kds);
	pgstrom_shmem_free(pds);
}

pgstrom_data_store *
pgstrom_create_data_store_row(TupleDesc tupdesc,
							  Size dstore_sz,
							  double ntup_per_block)
{
	pgstrom_data_store *pds;
	kern_data_store	   *kds;
	Size		baselen;
	Size		required;
	Size		allocation;
	cl_uint		max_blocks;
	cl_uint		nrooms;
	int			i;

	/* size of data-store has to be aligned to BLCKSZ */
	dstore_sz = TYPEALIGN(BLCKSZ, dstore_sz);
	max_blocks = dstore_sz / BLCKSZ;
	nrooms = (cl_uint)(ntup_per_block * (double)max_blocks * 1.1);

	/* allocation of kern_data_store; all we need to allocate here
	 * is base portion + array of rowitems, but no shared buffer page
	 * itself because we kick DMA on the pages directly.
	 */
	baselen = STROMALIGN(offsetof(kern_data_store,
								  colmeta[tupdesc->natts]));
	required = baselen + sizeof(kern_rowitem) * nrooms;
	kds = pgstrom_shmem_alloc_alap(required, &allocation);
	if (!kds)
		elog(ERROR, "out of shared memory");
	/* update exact number of rooms available */
	nrooms = (allocation - baselen) / sizeof(kern_rowitem);

	kds->hostptr = (hostptr_t) kds;
	kds->length = 0;	/* 0 for row-store */
	kds->ncols = tupdesc->natts;
	kds->nitems = 0;
	kds->nrooms = nrooms;
	kds->nblocks = 0;
	kds->is_column = false;
	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute	attr = tupdesc->attrs[i];

		kds->colmeta[i].attnotnull = attr->attnotnull;
		kds->colmeta[i].attalign = typealign_get_width(attr->attalign);
		kds->colmeta[i].attlen = attr->attlen;
		kds->colmeta[i].rs_attnum = attr->attnum;
	}
	Assert((uintptr_t)KERN_DATA_STORE_ROWITEM(kds, nrooms) <=
		   (uintptr_t)(kds) + allocation);

	/* allocation of pgstrom_data_store */
	required = offsetof(pgstrom_data_store, blocks[max_blocks]);
	pds = pgstrom_shmem_alloc(required);
	if (!pds)
	{
		pgstrom_shmem_free(kds);
		elog(ERROR, "out of shared memory");
	}
	pds->sobj.stag = StromTag_DataStore;
	SpinLockInit(&pds->lock);
	pds->refcnt = 1;
	pds->kds = kds;
	pds->ktoast = NULL;	/* never used */
	pds->max_blocks = max_blocks;

	return pds;
}

pgstrom_data_store *
pgstrom_create_data_store_column(TupleDesc tupdesc,
								 Size dstore_sz,
								 Bitmapset *attr_refs)
{
	pgstrom_data_store *pds;
	kern_data_store	   *kds;
	Size		required;
	Size		cs_offset;
	cl_uint		unitsz = 0;
	cl_uint		nrooms;
	int			i, j;

	/* calculate how many rows can be stored in a data store.
	 * note that unitsz here means memory consumption per
	 * (STROMALIGN_LEN * BITS_PER_BYTE) rows to simplify the
	 * calculation because of alignment and null-bitmap
	 */
	cs_offset = STROMALIGN(offsetof(kern_data_store,
									colmeta[tupdesc->natts]));
	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute	attr = tupdesc->attrs[i];

		j = i - FirstLowInvalidHeapAttributeNumber;
		if (!bms_is_member(j, attr_refs))
			continue;
		if (!attr->attnotnull)
			unitsz += STROMALIGN_LEN;
		if (attr->attlen > 0)
			unitsz += attr->attlen * (STROMALIGN_LEN * BITS_PER_BYTE);
		else
			unitsz += sizeof(cl_uint) * (STROMALIGN_LEN * BITS_PER_BYTE);
	}
	/* note that unitsz is size per STROMALIGN_LEN * BITS_PER_BYTE rows! */
	nrooms = ((dstore_sz - cs_offset) / unitsz);
	required = cs_offset + nrooms * unitsz;
	nrooms *= STROMALIGN_LEN * BITS_PER_BYTE;
	Assert(required <= dstore_sz);

	kds = pgstrom_shmem_alloc(required);
	if (!kds)
		elog(ERROR, "out of shared memory");
	kds->hostptr = (hostptr_t) kds;
	kds->length = required;
	kds->ncols = tupdesc->natts;
	kds->nitems = 0;
	kds->nrooms = nrooms;
	kds->nblocks = 0;	/* column format never uses blocks */
	kds->is_column = true;
	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute	attr = tupdesc->attrs[i];

		j = i - FirstLowInvalidHeapAttributeNumber;
		kds->colmeta[i].attnotnull = attr->attnotnull;
		kds->colmeta[i].attalign = typealign_get_width(attr->attalign);
		kds->colmeta[i].attlen = attr->attlen;
		if (!bms_is_member(j, attr_refs))
			kds->colmeta[i].attvalid = 0;
		else
		{
			kds->colmeta[i].cs_offset = cs_offset;
			if (!attr->attnotnull)
				cs_offset += STROMALIGN(nrooms / BITS_PER_BYTE);
			if (attr->attlen > 0)
				cs_offset += STROMALIGN(attr->attlen * nrooms);
			else
				cs_offset += STROMALIGN(sizeof(cl_uint) * nrooms);
		}
	}
	Assert(cs_offset == required);

	/* allocation of pgstrom_data_store also */
	pds = pgstrom_shmem_alloc(sizeof(pgstrom_data_store));
	if (!pds)
	{
		pgstrom_shmem_free(kds);
		elog(ERROR, "out of shared memory");
	}
	pds->sobj.stag = StromTag_DataStore;
	SpinLockInit(&pds->lock);
	pds->refcnt = 1;
	pds->kds = kds;
	pds->ktoast = NULL;	/* expand on demand */
	pds->max_blocks = 0;

	return pds;
}

pgstrom_data_store *
pgstrom_get_data_store(pgstrom_data_store *pds)
{
	SpinLockAcquire(&pds->lock);
	Assert(pds->refcnt > 0);
	pds->refcnt++;
	SpinLockRelease(&pds->lock);

	return pds;
}

void
pgstrom_put_data_store(pgstrom_data_store *pds)
{
	bool	do_release = false;

	SpinLockAcquire(&pds->lock);
    Assert(pds->refcnt > 0);
	if (--pds->refcnt == 0)
		do_release = true;
	SpinLockRelease(&pds->lock);
	if (do_release)
		pgstrom_release_data_store(pds);
}

int
pgstrom_data_store_insert_block(pgstrom_data_store *pds,
								Relation rel, BlockNumber blknum,
								Snapshot snapshot, bool page_prune)
{
	kern_data_store	*kds = pds->kds;
	kern_rowitem *kri;
	Buffer		buffer;
	Page		page;
	int			lines;
	int			ntup;
	OffsetNumber lineoff;
	ItemId		lpp;
	bool		all_visible;

	/* only row-store can block read */
	Assert(!pds->kds->is_column);
	/* we never use all the block slots */
	Assert(kds->nblocks < pds->max_blocks);

	CHECK_FOR_INTERRUPTS();

	buffer = ReadBuffer(rel, blknum);

	/* Just like heapgetpage(), however, jobs we focus on is OLAP workload,
	 * so it's uncertain whether we should vacuum the page here.
	 */
	if (page_prune)
		heap_page_prune_opt(rel, buffer);

	/* we will check tuple's visibility under the shared lock */
	LockBuffer(buffer, BUFFER_LOCK_SHARE);
	page = (Page) BufferGetPage(buffer);
	lines = PageGetMaxOffsetNumber(page);
	ntup = 0;

	/* Check whether we have enough rooms to store expected rowitems
	 * and shared buffer pages, any mode.
	 * If not, we have to inform the caller this block shall be loaded
	 * on the next data-store.
	 */
	if (kds->nitems + lines > kds->nrooms ||
		STROMALIGN(offsetof(kern_data_store, colmeta[kds->ncols])) +
		STROMALIGN(sizeof(kern_rowitem) * (kds->nitems + lines)) +
		BLCKSZ * kds->nblocks >= BLCKSZ * pds->max_blocks)
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
	kri = KERN_DATA_STORE_ROWITEM(kds, kds->nitems);
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

		CheckForSerializableConflictOut(valid, rel, &tup,
										buffer, snapshot);
		if (!valid)
			continue;

		kri->block_ofs = kds->nblocks;
		kri->item_ofs = lineoff;
		kri++;
		ntup++;
	}
	LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
	Assert(ntup <= MaxHeapTuplesPerPage);
	Assert(kds->nitems + ntup <= kds->nrooms);
	kds->nitems += ntup;

	/*
	 * NOTE: Local buffers are allocated on the private address space,
	 * it is not visible to opencl server. so, we make a duplication
	 * instead. Shared buffer can be referenced with zero-copy.
	 */
	pds->blocks[kds->nblocks].buffer = buffer;
	if (!BufferIsLocal(buffer))
		pds->blocks[kds->nblocks].page = page;
	else
	{
		Page	dup_page = pgstrom_shmem_alloc(BLCKSZ);

		if (!dup_page)
			elog(ERROR, "out of memory");
		memcpy(dup_page, page, BLCKSZ);
		pds->blocks[kds->nblocks].page = dup_page;
	}
	kds->nblocks++;

	return ntup;
}

/*
 * pgstrom_data_store_insert_tuple
 *
 * It inserts a tuple on the data store. Unlike block read mode, we can use
 * this interface for both of row and column data store.
 */
static bool
row_data_store_insert_tuple(pgstrom_data_store *pds,
							kern_data_store *kds,
							TupleTableSlot *slot)
{}

static bool
column_data_store_insert_tuple(pgstrom_data_store *pds,
							   kern_data_store *kds,
							   TupleTableSlot *slot)
{
	kern_data_store *kds = pds->kds;
	TupleDesc	tupdesc = slot->tts_tupleDescriptor;
	cl_uint		cs_offset;
	int			i, j;

	Assert(kds->is_column);
	Assert(kds->ncols == tupdesc->natts);

	/* no more rooms to store tuples */
	if (kds->nitems == kds->nrooms)
		return false;
	j = kds->nitems++;

	/* makes tts_values/tts_isnull available */
	slot_getallattrs(slot);

	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute	attr = tupdesc->attrs[i];

		Assert(attr->attlen > 0
			   ? attr->attlen == kds->colmeta[i].attlen
			   : kds->colmeta[i].attlen < 0);

		if (!kds->colmeta[i].attvalid)
			continue;

		cs_offset = kds->colmeta[i].cs_offset;
		if (!kds->colmeta[i].attnotnull)
		{
			cl_char   *nullmap = (cl_char *)kds + cs_offset;

			if (slot->tts_isnull[i])
			{
				nullmap[j>>3] &= ~(1 << (j & 7));
				continue;
			}
			nullmap[j>>3] |=  (1 << (j & 7));
            cs_offset += STROMALIGN(BITMAPLEN(kds->nrooms));
		}
		else if (slot->tts_isnull[i])
			elog(ERROR, "unable to put NULL on not-null attribute");

		if (attr->attlen > 0)
		{
			char   *cs_values = (char *)kds + cs_offset + attr->attlen * j;

			if (!attr->attbyval)
			{
				Pointer	ptr = DatumGetPointer(slot->tts_values[i]);

				memcpy(cs_values, ptr, attr->attlen);
			}						   
			else
			{
				switch (attr->attlen)
				{
					case sizeof(char):
						*((char *)cs_values)
							= DatumGetChar(slot->tts_values[i]);
						break;
					case sizeof(uint16):
						*((int16 *)cs_values)
							= DatumGetInt16(slot->tts_values[i]);
						break;
					case sizeof(uint32):
						*((int32 *)cs_values)
							= DatumGetInt32(slot->tts_values[i]);
						break;
					case sizeof(int64):
						*((int64 *)cs_values)
							= DatumGetInt64(slot->tts_values[i]);
						break;
					default:
						memcpy(cs_values, &slot->tts_values[i], attr->attlen);
						break;
				}
			}
		}
		else
		{
			struct varlena *vl_data = (struct varlena *) slot->tts_values[i];
			cl_uint			vl_len = VARSIZE_ANY(vl_data);
			cl_uint		   *vl_ofs = (cl_uint *)((char *)kds + cs_offset);
			kern_toastbuf  *ktoast;

			Assert(!attr->attbyval);
			/* expand toast buffer on demand */
			while (!pds->ktoast || (pds->ktoast->length -
									pds->ktoast->usage) < INTALIGN(vl_len))
			{
				Size	required;

				if (!pds->ktoast)
					required = TOASTBUF_UNITSZ;
				else
					required = 2 * pds->ktoast->length;

				ktoast = pgstrom_shmem_alloc(required);
				if (!ktoast)
					elog(ERROR, "out of shared memory");
				ktoast->hostptr = (hostptr_t) ktoast;
				ktoast->length = required;
				ktoast->usage = (!pds->ktoast
								 ? offsetof(kern_toastbuf, data[0])
								 : pds->ktoast->usage);
				if (pds->ktoast)
				{
					memcpy(ktoast->data,
						   pds->ktoast->data,
						   pds->ktoast->usage);
					pgstrom_shmem_free(pds->ktoast);
				}
				pds->ktoast = ktoast;
			}
			ktoast = pds->ktoast;
			Assert(ktoast->usage + INTALIGN(vl_len) <= ktoast->length);
			vl_ofs[j] = ktoast->usage;
			memcpy((char *)ktoast + ktoast->usage, vl_data, vl_len);
			ktoast->usage += INTALIGN(vl_len);
		}
	}
	return false;
}

bool
pgstrom_data_store_insert_tuple(pgstrom_data_store *pds,
								TupleTableSlot *slot)
{
	kern_data_store *kds = pds->kds;

	return (!kds->is_column
			? row_data_store_insert_tuple(pds, kds, slot)
			: column_data_store_insert_tuple(pds, kds, slot));
}

/*
 * clserv_dmasend_data_store
 *
 * It enqueues DMA send request of the supplied data-store.
 * Note that we expect this routine is called under the OpenCL server
 * context.
 */
cl_int
clserv_dmasend_data_store(pgstrom_data_store *pds,
						  cl_command_queue kcmdq,
						  cl_mem kds_buffer,
						  cl_mem ktoast_buffer,
						  cl_uint num_blockers,
						  const cl_event *blockers,
						  cl_uint *ev_index,
						  cl_event *events,
						  pgstrom_perfmon *pfm)
{
	kern_data_store *kds = pds->kds;
	size_t		length;
	size_t		offset;
	cl_int		i, n, rc;

#ifdef USE_ASSERT_CHECKING
	Assert(pgstrom_i_am_clserv);
	rc = clGetMemObjectInfo(kds_buffer,
							CL_MEM_SIZE,
							sizeof(length),
							&length,
							NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clGetMemObjectInfo (%s)",
				   opencl_strerror(rc));
		return rc;
	}
	Assert(length <= KERN_DATA_STORE_LENGTH(kds));
#endif

	if (kds->is_column)
	{
		rc = clEnqueueWriteBuffer(kcmdq,
								  kds_buffer,
								  CL_FALSE,
								  0,
								  kds->length,
								  kds,
								  num_blockers,
								  blockers,
								  events + *ev_index);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clEnqueueWriteBuffer: %s",
					   opencl_strerror(rc));
			return rc;
		}
		(*ev_index)++;
		pfm->bytes_dma_send += kds->length;
		pfm->num_dma_send++;

		if (!pds->ktoast)
			Assert(!ktoast_buffer);
		else
		{
			kern_toastbuf  *ktoast = pds->ktoast;

			Assert(ktoast_buffer);

			rc = clEnqueueWriteBuffer(kcmdq,
									  ktoast_buffer,
									  CL_FALSE,
									  0,
									  ktoast->usage,
									  ktoast,
									  num_blockers,
									  blockers,
									  events + *ev_index);
			if (rc != CL_SUCCESS)
			{
				clserv_log("failed on clEnqueueWriteBuffer: %s",
						   opencl_strerror(rc));
				return rc;
			}
			(*ev_index)++;
			pfm->bytes_dma_send += ktoast->usage;
			pfm->num_dma_send++;
		}
	}
	else
	{
		length = ((uintptr_t)KERN_DATA_STORE_ROWITEM(kds, kds->nitems) -
				  (uintptr_t)(kds));
		rc = clEnqueueWriteBuffer(kcmdq,
								  kds_buffer,
								  CL_FALSE,
								  0,
								  length,
								  kds,
								  num_blockers,
								  blockers,
								  events + *ev_index);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clEnqueueWriteBuffer: %s",
					   opencl_strerror(rc));
			return rc;
		}
		(*ev_index)++;
		pfm->bytes_dma_send += kds->length;
		pfm->num_dma_send++;

		offset = ((uintptr_t)KERN_DATA_STORE_ROWBLOCK(kds, 0) -
				  (uintptr_t)(kds));
		length = BLCKSZ;
		for (i=0, n=0; i < kds->nblocks; i++)
		{
			/*
			 * NOTE: A micro optimization; if next page is located
			 * on the continuous region, the upcoming DMA request
			 * can be merged to reduce DMA management cost
			 */
			if (i+1 < kds->nblocks &&
				(uintptr_t)pds->blocks[i].page + BLCKSZ ==
				(uintptr_t)pds->blocks[i+1].page)
			{
				n++;
				continue;
			}
			rc = clEnqueueWriteBuffer(kcmdq,
									  kds_buffer,
									  CL_FALSE,
									  offset,
									  BLCKSZ * (n+1),
									  pds->blocks[i-n].page,
									  num_blockers,
									  blockers,
									  events + *ev_index);
			if (rc != CL_SUCCESS)
			{
				clserv_log("failed on clEnqueueWriteBuffer: %s",
						   opencl_strerror(rc));
				return rc;
			}
			(*ev_index)++;
			pfm->bytes_dma_send += BLCKSZ * (n+1);
			pfm->num_dma_send++;
			offset += BLCKSZ * (n+1);
			n = 0;
		}
		Assert(!pds->ktoast);
	}
	return CL_SUCCESS;
}

/*
 * pgstrom_dump_data_store
 *
 * A utility routine that dumps properties of data store
 */
void
pgstrom_dump_data_store(pgstrom_data_store *pds)
{
	kern_data_store	   *kds = pds->kds;
	int					i;

#define PDS_DUMP(...)							\
	do {										\
		if (pgstrom_i_am_clserv)				\
			clserv_log(__VA_ARGS__);			\
		else									\
			elog(INFO, __VA_ARGS__);			\
	} while (0)

	PDS_DUMP("pds {refcnt=%d kds=%p ktoast=%p}",
			 pds->refcnt, pds->kds, pds->ktoast);
	PDS_DUMP("kds (%s) {length=%u ncols=%u nitems=%u nrooms=%u nblocks=%u}",
			 kds->is_column ? "column-store" : "row-store",
			 kds->length, kds->ncols, kds->nitems, kds->nrooms, kds->nblocks);
	for (i=0; i < kds->ncols; i++)
	{
		if (!kds->colmeta[i].attvalid)
		{
			PDS_DUMP("attr[%d] {attnotnull=%d attalign=%d attlen=%d; invalid}",
					 i,
					 kds->colmeta[i].attnotnull,
					 kds->colmeta[i].attalign,
					 kds->colmeta[i].attlen);
		}
		else
		{
			PDS_DUMP("attr[%d] {attnotnull=%d attalign=%d attlen=%d %s=%u}",
					 i,
					 kds->colmeta[i].attnotnull,
					 kds->colmeta[i].attalign,
					 kds->colmeta[i].attlen,
					 kds->is_column ? "cs_offset" : "rs_attnum",
					 kds->colmeta[i].cs_offset);
		}
	}

	for (i=0; i < kds->nblocks; i++)
	{
		PDS_DUMP("block[%d] {buffer=%u page=%p}",
				 i,
				 pds->blocks[i].buffer,
				 pds->blocks[i].page);
	}
#undef PDS_DUMP
}
