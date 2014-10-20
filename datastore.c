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
 * pgstrom_plan_can_multi_exec (obsolete; should not be used any more)
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

/*
 * it makes a template of header portion of kern_data_store according
 * to the supplied tupdesc and bitmapset of referenced columns.
 */
bytea *
kparam_make_kds_head(TupleDesc tupdesc,
					 int kds_format,
					 Bitmapset *referenced)
{
	kern_data_store	*kds_head;
	bytea	   *result;
	Size		length;
	int			i, j, ncols;

	/* allocation */
	ncols = tupdesc->natts;
	length = STROMALIGN(offsetof(kern_data_store, colmeta[ncols]));
	result = palloc0(VARHDRSZ + length);
	SET_VARSIZE(result, VARHDRSZ + length);

	kds_head = (kern_data_store *) VARDATA(result);
	kds_head->length = (cl_uint)(-1);	/* to be set later */
	kds_head->ncols = ncols;
	kds_head->nitems = (cl_uint)(-1);	/* to be set later */
	kds_head->nrooms = (cl_uint)(-1);	/* to be set later */
	kds_head->format = kds_format;

	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupdesc->attrs[i];

		j = attr->attnum - FirstLowInvalidHeapAttributeNumber;

		kds_head->colmeta[i].attnotnull = attr->attnotnull;
		kds_head->colmeta[i].attalign = typealign_get_width(attr->attalign);
		kds_head->colmeta[i].attlen = (attr->attlen > 0 ? attr->attlen : -1);
		if (!bms_is_member(j, referenced))
			kds_head->colmeta[i].attvalid = 0;
		else
			kds_head->colmeta[i].attvalid = (cl_uint)(-1);
		/* rest of fields shall be set later */
	}
	return result;
}

void
kparam_refresh_kds_head(kern_data_store *kds_head,
						cl_uint nitems, cl_uint nrooms)
{
	/*
	 * Only column- or tupslot- format is writable by device kernel
	 */
	Assert(kds_head->format == KDS_FORMAT_COLUMN ||
		   kds_head->format == KDS_FORMAT_TUPSLOT);
	kds_head->nitems = nitems;
	kds_head->nrooms = nrooms;

	if (kds_head->format == KDS_FORMAT_COLUMN)
	{
		Size		length;
		cl_uint		i, ncols = kds_head->ncols;

		length = STROMALIGN(offsetof(kern_data_store, colmeta[ncols]));
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
		kds_head->length = STROMALIGN(length);
	}
	else if (kds_head->format == KDS_FORMAT_TUPSLOT)
	{
		Size		length =
			((uintptr_t)KERN_DATA_STORE_VALUES(kds_head, nrooms) -
			 (uintptr_t)(kds_head));
		kds_head->length = STROMALIGN(length);
	}
	else
		elog(ERROR, "Bug? unexpected kds format: %d", kds_head->format);
}

#if 0
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

	if (row_index >= kds->nitems)
		return false;	/* out of range */

	/* in case of row-store */
	if (kds->format == KDS_FORMAT_ROW)
	{
		kern_rowitem   *ritem = KERN_DATA_STORE_ROWITEM(kds, row_index);
		kern_blkitem   *bitem;
		ItemId			lpp;

		Assert(ritem->blk_index < kds->nblocks);
		bitem = KERN_DATA_STORE_BLKITEM(kds, ritem->blk_index);
		lpp = PageGetItemId(bitem->page, ritem->item_offset);
		Assert(ItemIdIsNormal(lpp));

		tuple->t_data = (HeapTupleHeader) PageGetItem(bitem->page, lpp);
		tuple->t_len = ItemIdGetLength(lpp);
		ItemPointerSet(&tuple->t_self, bitem->buffer, ritem->item_offset);

		ExecStoreTuple(tuple, slot, bitem->buffer, false);

		return true;
	}
	/* in case of row-flat-store */
	if (kds->format == KDS_FORMAT_ROW_FLAT)
	{
		TupleDesc		tupdesc = slot->tts_tupleDescriptor;
		kern_rowitem   *ritem = KERN_DATA_STORE_ROWITEM(kds, row_index);
		HeapTupleHeader	htup;

		Assert(ritem->htup_offset < kds->length);
		htup = (HeapTupleHeader)((char *)kds + ritem->htup_offset);

		memset(tuple, 0, sizeof(HeapTupleData));
		tuple->t_data = htup;
		heap_deform_tuple(tuple, tupdesc,
						  slot->tts_values,
						  slot->tts_isnull);
		ExecStoreVirtualTuple(slot);

		return true;
	}
	/* in case of tuple-slot format */
	if (kds->format == KDS_FORMAT_TUPSLOT)
	{
		Datum	   *tts_values = KERN_DATA_STORE_VALUES(kds, row_index);
		cl_char	   *tts_isnull = KERN_DATA_STORE_ISNULL(kds, row_index);
		int			i;

		for (i=0; i < kds->ncols; i++)
		{
			elog(INFO, "(%u,%zu) isnull=%d value=%016lx", i, row_index, tts_isnull[i], tts_values[i]);
		}

		ExecClearTuple(slot);
		memcpy(slot->tts_values, tts_values, sizeof(Datum) * kds->ncols);
		memcpy(slot->tts_isnull, tts_isnull, sizeof(bool) * kds->ncols);
		ExecStoreVirtualTuple(slot);
		/*
		 * MEMO: Is it really needed to take memcpy() here? If we can 
		 * do same job with pointer opertion, it makes more sense from
		 * performance standpoint.
		 * (NOTE: hash-join materialization is a hot point)
		 */
		return true;
	}
	elog(ERROR, "Bug? unexpected data-store format: %d", kds->format);
	return false;
#if 0
	/* otherwise, column store */
	Assert(kds->format == KDS_FORMAT_COLUMN);
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
#endif
}

void
pgstrom_release_data_store(pgstrom_data_store *pds)
{
	ResourceOwner		saved_owner;
	int					i;

	saved_owner = CurrentResourceOwner;
	PG_TRY();
	{
		kern_data_store	   *kds = pds->kds;

		CurrentResourceOwner = pds->resowner;

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
		if (kds)
		{
			Assert(kds->nblocks <= kds->maxblocks);
			for (i = kds->nblocks - 1; i >= 0; i--)
			{
				kern_blkitem   *bitem = KERN_DATA_STORE_BLKITEM(kds, i);

				if (BufferIsLocal(bitem->buffer) ||
					BufferIsInvalid(bitem->buffer))
					continue;
				if (!pgstrom_i_am_clserv &&
					!pgstrom_restrack_cleanup_context())
					ReleaseBuffer(bitem->buffer);
			}
			pgstrom_shmem_free(kds);
		}
	}
	PG_CATCH();
	{
		CurrentResourceOwner = saved_owner;
		PG_RE_THROW();
	}
	PG_END_TRY();
	CurrentResourceOwner = saved_owner;
	if (pds->resowner &&
		!pgstrom_i_am_clserv &&
		!pgstrom_restrack_cleanup_context())
		ResourceOwnerDelete(pds->resowner);
	if (pds->ktoast)
		pgstrom_release_data_store(pds->ktoast);
	if (pds->local_pages)
		pgstrom_shmem_free(pds->local_pages);
	pgstrom_shmem_free(pds);
}

static void
init_kern_data_store(kern_data_store *kds,
					 TupleDesc tupdesc,
					 Size length,
					 int format,
					 cl_uint maxblocks,
					 cl_uint nrooms)
{
	int		i;

	kds->hostptr = (hostptr_t) &kds->hostptr;
	kds->length = length;
	kds->usage = 0;
	kds->ncols = tupdesc->natts;
	kds->nitems = 0;
	kds->nrooms = nrooms;
	kds->nblocks = 0;
	kds->maxblocks = maxblocks;
	kds->format = format;
	kds->tdhasoid = tupdesc->tdhasoid;
	kds->tdtypeid = tupdesc->tdtypeid;
	kds->tdtypmod = tupdesc->tdtypmod;
	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute	attr = tupdesc->attrs[i];

		kds->colmeta[i].attnotnull = attr->attnotnull;
		kds->colmeta[i].attalign = typealign_get_width(attr->attalign);
		kds->colmeta[i].attlen = attr->attlen;
		kds->colmeta[i].rs_attnum = attr->attnum;
	}
}

pgstrom_data_store *
pgstrom_create_data_store_row(TupleDesc tupdesc,
							  Size dstore_sz,
							  double ntup_per_block)
{
	pgstrom_data_store *pds;
	kern_data_store	   *kds;
	Size		required;
	cl_uint		maxblocks;
	cl_uint		nrooms;

	/* size of data-store has to be aligned to BLCKSZ */
	dstore_sz = TYPEALIGN(BLCKSZ, dstore_sz);
	maxblocks = dstore_sz / BLCKSZ;
	nrooms = (cl_uint)(ntup_per_block * (double)maxblocks * 1.25);

	/* allocation of kern_data_store */
	required = (STROMALIGN(offsetof(kern_data_store,
									colmeta[tupdesc->natts])) +
				STROMALIGN(sizeof(kern_blkitem) * maxblocks) +
				STROMALIGN(sizeof(kern_rowitem) * nrooms));
	kds = pgstrom_shmem_alloc(required);
	if (!kds)
		elog(ERROR, "out of shared memory");
	init_kern_data_store(kds, tupdesc, required,
						 KDS_FORMAT_ROW, maxblocks, nrooms);
	/* allocation of pgstrom_data_store */
	pds = pgstrom_shmem_alloc(sizeof(pgstrom_data_store));
	if (!pds)
	{
		pgstrom_shmem_free(kds);
		elog(ERROR, "out of shared memory");
	}
	/* ResourceOwnerCreate() may raise an error */
	PG_TRY();
	{
		pds->sobj.stag = StromTag_DataStore;
		SpinLockInit(&pds->lock);
		pds->refcnt = 1;
		pds->kds = kds;
		pds->ktoast = NULL;	/* never used */
		pds->resowner = ResourceOwnerCreate(CurrentResourceOwner,
											"pgstrom_data_store");
		pds->local_pages = NULL;	/* allocation on demand */
	}
	PG_CATCH();
	{
		pgstrom_shmem_free(kds);
		pgstrom_shmem_free(pds);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return pds;
}

pgstrom_data_store *
pgstrom_create_data_store_row_flat(TupleDesc tupdesc, Size length)
{
	pgstrom_data_store *pds;
	kern_data_store	   *kds;
	Size		allocated;
	cl_uint		nrooms;

	/* allocation of kern_data_store */
	kds = pgstrom_shmem_alloc_alap(length, &allocated);
	if (!kds)
		elog(ERROR, "out of shared memory");
	/*
	 * NOTE: length of row-flat format has to be strictly aligned
	 * because location of heaptuple is calculated using offset from
	 * the buffer tail!
	 */
	allocated = STROMALIGN_DOWN(allocated);

	/* max number of rooms according to the allocated buffer length */
	nrooms = (STROMALIGN_DOWN(length) -
			  STROMALIGN(offsetof(kern_data_store,
								  colmeta[tupdesc->natts])))
		/ sizeof(kern_rowitem);
	init_kern_data_store(kds, tupdesc, allocated,
						 KDS_FORMAT_ROW_FLAT, 0, nrooms);
	/* allocation of pgstrom_data_store */
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
	pds->ktoast = NULL;		/* never used */
	pds->resowner = NULL;	/* never used */
	pds->local_pages = NULL;/* never used */

	return pds;
}

pgstrom_data_store *
pgstrom_create_data_store_tupslot(TupleDesc tupdesc, cl_uint nrooms)
{
	pgstrom_data_store *pds;
	kern_data_store	   *kds;
	Size				required;

	/* kern_data_store */
	required = (STROMALIGN(offsetof(kern_data_store,
									colmeta[tupdesc->natts])) +
				(LONGALIGN(sizeof(bool) * tupdesc->natts) +
				 LONGALIGN(sizeof(Datum) * tupdesc->natts)) * nrooms);
	kds = pgstrom_shmem_alloc(STROMALIGN(required));
	if (!kds)
		elog(ERROR, "out of shared memory");
	init_kern_data_store(kds, tupdesc, required,
						 KDS_FORMAT_TUPSLOT, 0, nrooms);

	/* pgstrom_data_store */
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
	pds->ktoast = NULL;		/* assigned on demand */
	pds->resowner = NULL;	/* never used for tuple-slot */
	pds->local_pages = NULL;/* never used for tuple-slot */

	return pds;
}

#if 0
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
	kds->hostptr = (hostptr_t) &kds->hostptr;
	kds->length = required;
	kds->ncols = tupdesc->natts;
	kds->nitems = 0;
	kds->nrooms = nrooms;
	kds->nblocks = 0;	/* column format never */
	kds->maxblocks = 0;	/* uses buffer blocks */
	kds->format = KDS_FORMAT_COLUMN;
	kds->tdhasoid = tupdesc->tdhasoid;
	kds->tdtypeid = tupdesc->tdtypeid;
	kds->tdtypmod = tupdesc->tdtypmod;
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
	pds->ktoast = NULL;		/* expand on demand */
	pds->resowner = NULL;	/* should be never used to column-store */
	pds->local_pages = NULL;/* should be never used to column-store */

	return pds;
}
#endif

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
	kern_rowitem   *ritem;
	kern_blkitem   *bitem;
	Buffer			buffer;
	Page			page;
	int				lines;
	int				ntup;
	OffsetNumber	lineoff;
	ItemId			lpp;
	bool			all_visible;
	ResourceOwner	saved_owner;

	/* only row-store can block read */
	Assert(kds->format == KDS_FORMAT_ROW);
	/* we never use all the block slots */
	Assert(kds->nblocks < kds->maxblocks);
	/* we need a resource owner to track shared buffer */
	Assert(pds->resowner != NULL);

	CHECK_FOR_INTERRUPTS();

	saved_owner = CurrentResourceOwner;
	PG_TRY();
	{
		CurrentResourceOwner = pds->resowner;

		/* load the target buffer */
		buffer = ReadBuffer(rel, blknum);

		/* Just like heapgetpage(), however, jobs we focus on is OLAP
		 * workload, so it's uncertain whether we should vacuum the page
		 * here.
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
			STROMALIGN(sizeof(kern_blkitem) * (kds->maxblocks)) +
			STROMALIGN(sizeof(kern_rowitem) * (kds->nitems + lines)) +
			BLCKSZ * kds->nblocks >= BLCKSZ * kds->maxblocks)
		{
			UnlockReleaseBuffer(buffer);
			CurrentResourceOwner = saved_owner;
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
		ritem = KERN_DATA_STORE_ROWITEM(kds, kds->nitems);
		bitem = KERN_DATA_STORE_BLKITEM(kds, kds->nblocks);
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

			ritem->blk_index = kds->nblocks;
			ritem->item_offset = lineoff;
			ritem++;
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
		bitem->buffer = buffer;
		if (!BufferIsLocal(buffer))
			bitem->page = page;
		else
		{
			Page	dup_page;

			/*
			 * NOTE: We expect seldom cases requires to mix shared buffers
			 * and private buffers in a single data-chunk. So, we allocate
			 * all the expected local-pages at once. It has two another
			 * benefit; 1. duplicated pages tend to have continuous address
			 * that may reduce number of DMA call, 2. memory allocation
			 * request to BLCKSZ actually consumes 2*BLCKSZ because of
			 * additional fields that does not fit 2^N manner.
			 */
			if (!pds->local_pages)
			{
				Size	required = BLCKSZ * kds->maxblocks;

				pds->local_pages = pgstrom_shmem_alloc(required);
				if (!pds->local_pages)
					elog(ERROR, "out of memory");
			}
			dup_page = (Page)(pds->local_pages + BLCKSZ * kds->nblocks);
			memcpy(dup_page, page, BLCKSZ);
            bitem->page = dup_page;
            ReleaseBuffer(buffer);
		}
		kds->nblocks++;
	}
	PG_CATCH();
	{
		CurrentResourceOwner = saved_owner;
		PG_RE_THROW();
	}
	PG_END_TRY();
	CurrentResourceOwner = saved_owner;

	return ntup;
}

/*
 * pgstrom_data_store_insert_tuple
 *
 * It inserts a tuple on the data store. Unlike block read mode, we can use
 * this interface for both of row and column data store.
 */
bool
pgstrom_data_store_insert_tuple(pgstrom_data_store *pds,
								TupleTableSlot *slot)
{
	kern_data_store	   *kds = pds->kds;
	TupleDesc			tupdesc = slot->tts_tupleDescriptor;

	/* No room to store a new kern_rowitem? */
	if (kds->nitems >= kds->nrooms)
		return false;
	Assert(kds->ncols == tupdesc->natts);

	if (kds->format == KDS_FORMAT_ROW)
	{
		HeapTuple		tuple;
		OffsetNumber	offnum;
		kern_rowitem   *ritem = KERN_DATA_STORE_ROWITEM(kds, kds->nitems);
		kern_blkitem   *bitem;
		Page			page = NULL;

		/* reference a HeapTuple in TupleTableSlot */
		tuple = ExecFetchSlotTuple(slot);

		if (kds->nblocks == 0)
			bitem = NULL;
		else
		{
			bitem = KERN_DATA_STORE_BLKITEM(kds, kds->nblocks - 1);
			page = bitem->page;
		}

		if (!bitem ||
			!BufferIsInvalid(bitem->buffer) ||
			PageGetFreeSpace(bitem->page) < MAXALIGN(tuple->t_len))
		{
			/*
			 * Expand blocks if the last one is associated with a particular
			 * shared or private buffer, or free space is not available to
			 * put this new item.
			 */

			/* No rooms to expand blocks? */
			if (kds->nblocks >= kds->maxblocks)
				return false;

			/*
			 * Allocation of anonymous pages at once. See the comments at
			 * pgstrom_data_store_insert_block().
			 */
			if (!pds->local_pages)
			{
				Size	required = kds->maxblocks * BLCKSZ;

				pds->local_pages = pgstrom_shmem_alloc(required);
				if (!pds->local_pages)
					elog(ERROR, "out of memory");
			}
			page = (Page)(pds->local_pages + kds->nblocks * BLCKSZ);
			PageInit(page, BLCKSZ, 0);
			if (PageGetFreeSpace(page) < MAXALIGN(tuple->t_len))
			{
				pgstrom_shmem_free(page);
				elog(ERROR, "tuple too large (%zu)",
					 (Size)MAXALIGN(tuple->t_len));
			}
			bitem = KERN_DATA_STORE_BLKITEM(kds, kds->nblocks);
			bitem->buffer = InvalidBuffer;
			bitem->page = page;
			kds->nblocks++;
		}
		Assert(kds->nblocks > 0);
		offnum = PageAddItem(page, (Item) tuple->t_data, tuple->t_len,
							 InvalidOffsetNumber, false, true);
		if (offnum == InvalidOffsetNumber)
			elog(ERROR, "failed to add tuple");

		ritem->blk_index = kds->nblocks - 1;
		ritem->item_offset = offnum;
		kds->nitems++;

		return true;
	}
	else if (kds->format == KDS_FORMAT_ROW_FLAT)
	{
		HeapTuple		tuple;
		kern_rowitem   *ritem = KERN_DATA_STORE_ROWITEM(kds, kds->nitems);
		uintptr_t		usage;
		char		   *dest_addr;

		/* reference a HeapTuple in TupleTableSlot */
		tuple = ExecFetchSlotTuple(slot);

		/* check whether the tuple touches the watermark */
		usage = ((uintptr_t)(ritem + 1) -
				 (uintptr_t)(kds) +			/* from head */
				 kds->usage +				/* from tail */
				 LONGALIGN(tuple->t_len));	/* newly added */
		if (usage > kds->length)
			return false;

		dest_addr = ((char *)kds + kds->length -
					 kds->usage - LONGALIGN(tuple->t_len));
		memcpy(dest_addr, tuple, tuple->t_len);
		ritem->htup_offset = (hostptr_t)((char *)dest_addr - (char *)kds);
		kds->nitems++;

		return true;
	}
	Assert(kds->format == KDS_FORMAT_COLUMN);
	/*
	 * No longer column format is in-use
	 */
#if 0
	/* makes tts_values/tts_isnull available */
	slot_getallattrs(slot);

	/* row-index to store a new item */
	j = kds->nitems++;

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
				ktoast->hostptr = (hostptr_t) &ktoast->hostptr;
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
#endif
	return false;
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
	kern_data_store	   *kds = pds->kds;
	kern_blkitem	   *bitem;
	size_t				length;
	size_t				offset;
	cl_int				i, n, rc;

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
	Assert(length >= KERN_DATA_STORE_LENGTH(kds));
#endif
	if (kds->format == KDS_FORMAT_ROW_FLAT ||
		kds->format == KDS_FORMAT_TUPSLOT ||
		kds->format == KDS_FORMAT_COLUMN)
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

		if (pds->ktoast)
		{
			pgstrom_data_store	*ktoast = pds->ktoast;

			Assert(!ktoast->ktoast);

			rc = clserv_dmasend_data_store(ktoast,
										   kcmdq,
										   ktoast_buffer,
										   NULL,
										   num_blockers,
										   blockers,
										   ev_index,
										   events,
										   pfm);
		}
		return rc;
	}
	Assert(kds->format == KDS_FORMAT_ROW);
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
							  events + (*ev_index));
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
	bitem = KERN_DATA_STORE_BLKITEM(kds, 0);
	for (i=0, n=0; i < kds->nblocks; i++)
	{
		/*
		 * NOTE: A micro optimization; if next page is located
		 * on the continuous region, the upcoming DMA request
		 * can be merged to reduce DMA management cost
		 */
		if (i+1 < kds->nblocks &&
			(uintptr_t)bitem[i].page + BLCKSZ == (uintptr_t)bitem[i+1].page)
		{
			n++;
			continue;
		}
		rc = clEnqueueWriteBuffer(kcmdq,
								  kds_buffer,
								  CL_FALSE,
								  offset,
								  BLCKSZ * (n+1),
								  bitem[i-n].page,
								  num_blockers,
								  blockers,
								  events + (*ev_index));
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
	StringInfoData		buf;
	int					i, j, k;

#define PDS_DUMP(...)							\
	do {										\
		if (pgstrom_i_am_clserv)				\
			clserv_log(__VA_ARGS__);			\
		else									\
			elog(INFO, __VA_ARGS__);			\
	} while (0)

	initStringInfo(&buf);
	PDS_DUMP("pds {refcnt=%d kds=%p ktoast=%p}",
			 pds->refcnt, pds->kds, pds->ktoast);
	PDS_DUMP("kds (%s) {length=%u ncols=%u nitems=%u nrooms=%u "
			 "nblocks=%u maxblocks=%u}",
			 kds->format == KDS_FORMAT_ROW ? "row-store" :
			 kds->format == KDS_FORMAT_ROW_FLAT ? "row-flat" :
			 kds->format == KDS_FORMAT_COLUMN ? "column-store" :
			 kds->format == KDS_FORMAT_TUPSLOT ? "tuple-slot" : "unknown",
			 kds->length, kds->ncols, kds->nitems, kds->nrooms,
			 kds->nblocks, kds->maxblocks);
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
					 kds->format == KDS_FORMAT_COLUMN
					 ? "cs_offset"
					 : "rs_attnum",
					 kds->colmeta[i].cs_offset);
		}
	}

	if (kds->format == KDS_FORMAT_ROW_FLAT)
	{
		for (i=0; i < kds->nitems; i++)
		{
			kern_rowitem *ritem = KERN_DATA_STORE_ROWITEM(kds, i);
			HeapTupleHeaderData *htup =
				(HeapTupleHeaderData *)((char *)kds + ritem->htup_offset);
			cl_int		natts = (htup->t_infomask2 & HEAP_NATTS_MASK);
			cl_int		curr = htup->t_hoff;
			cl_int		vl_len;
			char	   *datum;

			resetStringInfo(&buf);
			appendStringInfo(&buf, "htup[%d] @%u natts=%u {",
							 i, ritem->htup_offset, natts);
			for (j=0; j < kds->ncols; j++)
			{
				if (j > 0)
					appendStringInfo(&buf, ", ");
				if ((htup->t_infomask & HEAP_HASNULL) != 0 &&
					att_isnull(j, htup->t_bits))
					appendStringInfo(&buf, "null");
				else if (kds->colmeta[j].attlen > 0)
				{
					curr = TYPEALIGN(kds->colmeta[j].attalign, curr);
					datum = (char *)htup + curr;

					if (kds->colmeta[j].attlen == sizeof(cl_char))
						appendStringInfo(&buf, "%02x", *((cl_char *)datum));
					else if (kds->colmeta[j].attlen == sizeof(cl_short))
						appendStringInfo(&buf, "%04x", *((cl_short *)datum));
					else if (kds->colmeta[j].attlen == sizeof(cl_int))
						appendStringInfo(&buf, "%08x", *((cl_int *)datum));
					else if (kds->colmeta[j].attlen == sizeof(cl_long))
						appendStringInfo(&buf, "%016lx", *((cl_long *)datum));
					else
					{
						for (k=0; k < kds->colmeta[j].attlen; k++)
						{
							if (k > 0)
								appendStringInfoChar(&buf, ' ');
							appendStringInfo(&buf, "%02x", datum[k]);
						}
					}
					curr += kds->colmeta[j].attlen;
				}
				else
				{
					if (!VARATT_NOT_PAD_BYTE((char *)htup + curr))
						curr = TYPEALIGN(kds->colmeta[j].attalign, curr);
					datum = (char *)htup + curr;
					vl_len = VARSIZE_ANY_EXHDR(datum);
					appendBinaryStringInfo(&buf, VARDATA_ANY(datum), vl_len);
					curr += vl_len;
				}
			}
			appendStringInfo(&buf, "}");
			PDS_DUMP("%s", buf.data);
		}
		pfree(buf.data);
	}

	if (false)
	{
		kern_blkitem	   *bitem = KERN_DATA_STORE_BLKITEM(kds, 0);

		for (i=0; i < kds->nblocks; i++)
		{
			PDS_DUMP("block[%d] {buffer=%u page=%p}",
					 i, bitem[i].buffer, bitem[i].page);
		}
	}
#undef PDS_DUMP
}
