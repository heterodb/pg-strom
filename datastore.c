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
 * pgstrom_try_varlena_inline
 *
 * It tried to inline varlena variables if it has an explicit
 * maximum length that is enough small than the threthold.
 * It enables to reduce number of DMA send and also allows
 * reduce waste of RAM by offset pointer (as long as user
 * designed database schema well).
 */
int
pgstrom_try_varlena_inline(Form_pg_attribute attr)
{
	if (attr->attlen < 0 &&
		attr->atttypmod > 0 &&
		attr->atttypmod <= pgstrom_max_inline_varlena)
		return INTALIGN(attr->atttypmod);
	return attr->attlen;
}

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
 * pgstrom_make_bulk_attmap
 *
 * It checks whether the supplied target-list has something except from
 * Var nodes. In case of simple var-node reference only, it is available
 * to skip expensive projection per row.
 */
List *
pgstrom_make_bulk_attmap(List *targetlist, Index varno)
{
	List	   *attmap = NIL;
	ListCell   *cell;

	foreach (cell, targetlist)
	{
		TargetEntry	   *tle = lfirst(cell);
		Var			   *var;

		if (!IsA(tle->expr, Var))
		{
			elog(INFO, "tlist contains things except for Var (%d)",
				 (int)nodeTag(tle->expr));
			return NIL;
		}
		var = (Var *) tle->expr;
		if (var->varno != varno)
		{
			elog(INFO, "var->varno = %d, but %d is expected",
				 var->varno, varno);
			return NIL;
		}

		/*
		 * FIXME: right now, we don't support to reference system columns
		 * using attmap of bulk-slot. Is it a reasonable restriction?
		 */
		if (var->varattno < 1)
			return NIL;
		attmap = lappend_int(attmap, var->varattno);
	}
	return attmap;
}

/*
 * pgstrom_plan_can_multi_exec
 *
 * It gives a hint whether subplan support bulk-exec mode, or not.
 */
bool
pgstrom_plan_can_multi_exec(const PlanState *ps)
{
	if (!IsA(ps, CustomPlanState))
		return false;
#if 0
	if (gpuscan_support_multi_exec((const CustomPlanState *) ps) ||
		gpusort_support_multi_exec((const CustomPlanState *) ps) ||
		gpuhashjoin_support_multi_exec((const CustomPlanState *) ps))
		return true;
#endif
	if (gpuscan_support_multi_exec((const CustomPlanState *) ps))
		return true;

	return false;
}

#if 0
/*
 * kparam_make_kds_head
 *
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
#if 0
/*
 * pgstrom_get_row_store
 *
 * increments reference counter of row-store
 */
tcache_row_store *
pgstrom_get_row_store(tcache_row_store *trs)
{
	SpinLockAcquire(&trs->refcnt_lock);
	Assert(trs->refcnt > 0);
	trs->refcnt++;
	SpinLockRelease(&trs->refcnt_lock);

	return trs;
}

/*
 * pgstrom_put_row_store
 *
 * decrements reference counter of row-store, then release it if no longer
 * referenced.
 */
void
pgstrom_put_row_store(tcache_row_store *trs)
{
	bool	do_release = false;

	SpinLockAcquire(&trs->refcnt_lock);
	Assert(trs->refcnt > 0);
	if (--trs->refcnt == 0)
		do_release = true;
	SpinLockRelease(&trs->refcnt_lock);

	if (do_release)
		pgstrom_shmem_free(trs);
}

/*
 * pgstrom_create_row_store
 *
 * create a row-store with refcnt=1
 */
tcache_row_store *
pgstrom_create_row_store(TupleDesc tupdesc)
{
	tcache_row_store *trs;
	int		i;

	trs = pgstrom_shmem_alloc(ROWSTORE_DEFAULT_SIZE);
	if (!trs)
		elog(ERROR, "out of shared memory");

	memset(trs, 0, sizeof(StromObject));
	trs->sobj.stag = StromTag_TCacheRowStore;
	SpinLockInit(&trs->refcnt_lock);
	trs->refcnt = 1;
	memset(&trs->chain, 0, sizeof(dlist_node));
	trs->usage
		= STROMALIGN_DOWN(ROWSTORE_DEFAULT_SIZE -
						  offsetof(tcache_row_store, kern));
	trs->blkno_max = 0;
	trs->blkno_min = MaxBlockNumber;
	trs->kern.length = trs->usage;
	trs->kern.ncols = tupdesc->natts;
	trs->kern.nrows = 0;

	/* construct colmeta structure for this row-store */
	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupdesc->attrs[i];

		trs->kern.colmeta[i].attnotnull = attr->attnotnull;
		trs->kern.colmeta[i].attalign = typealign_get_width(attr->attalign);
		trs->kern.colmeta[i].attlen = attr->attlen;
	}
	return trs;
}

/*
 * pgstrom_create_toast_buffer
 *
 * creata a toast-buffer to be attached on a particular column-store with
 * initial length, but TCACHE_TOASTBUF_INITSIZE at least.
 */
tcache_toastbuf *
pgstrom_create_toast_buffer(Size required)
{
	tcache_toastbuf *tbuf;
	Size		allocated;

	required = Max(required, TCACHE_TOASTBUF_INITSIZE);

	tbuf = pgstrom_shmem_alloc_alap(required, &allocated);
	if (!tbuf)
		return NULL;

	SpinLockInit(&tbuf->refcnt_lock);
	tbuf->refcnt = 1;
	tbuf->tbuf_length = allocated;
	tbuf->tbuf_usage = offsetof(tcache_toastbuf, data[0]);
	tbuf->tbuf_junk = 0;

	return tbuf;
}

/*
 * pgstrom_expand_toast_buffer
 *
 * it expand length of the toast buffer into twice.
 */
tcache_toastbuf *
pgstrom_expand_toast_buffer(tcache_toastbuf *tbuf_old)
{
	tcache_toastbuf *tbuf_new;
	Size	required = 2 * tbuf_old->tbuf_length;

	tbuf_new = pgstrom_create_toast_buffer(required);
	if (!tbuf_new)
		return NULL;
	memcpy(tbuf_new->data,
		   tbuf_old->data,
		   tbuf_old->tbuf_usage - offsetof(tcache_toastbuf, data[0]));
	tbuf_new->tbuf_usage = tbuf_old->tbuf_usage;
	tbuf_new->tbuf_junk = tbuf_old->tbuf_junk;

	return tbuf_new;
}

/*
 * pgstrom_get_toast_buffer
 *
 * It increments reference counter of the toast buffer.
 */
tcache_toastbuf *
pgstrom_get_toast_buffer(tcache_toastbuf *tbuf)
{
	SpinLockAcquire(&tbuf->refcnt_lock);
	Assert(tbuf->refcnt > 0);
	tbuf->refcnt++;
	SpinLockRelease(&tbuf->refcnt_lock);

	return tbuf;
}

/*
 * pgstrom_put_toast_buffer
 *
 * It decrements rerefence counter of the toast buffer, then release
 * shared memory region, if needed.
 */
void
pgstrom_put_toast_buffer(tcache_toastbuf *tbuf)
{
    bool    do_release = false;

    SpinLockAcquire(&tbuf->refcnt_lock);
    Assert(tbuf->refcnt > 0);
    if (--tbuf->refcnt == 0)
        do_release = true;
    SpinLockRelease(&tbuf->refcnt_lock);

    if (do_release)
        pgstrom_shmem_free(tbuf);
}

/*
 * pgstrom_get_column_store
 *
 * it increments reference counter of column-store
 */
tcache_column_store *
pgstrom_get_column_store(tcache_column_store *pcs)
{
	SpinLockAcquire(&pcs->refcnt_lock);
	Assert(pcs->refcnt > 0);
	pcs->refcnt++;
	SpinLockRelease(&pcs->refcnt_lock);

	return pcs;
}

/*
 * pgstrom_put_column_store
 *
 * it decrements reference counter of column-store, then release shared-
 * memory buffers if no longer referenced
 */
void
pgstrom_put_column_store(tcache_column_store *pcs)
{
	bool	do_release = false;
	int		i;

	SpinLockAcquire(&pcs->refcnt_lock);
	Assert(pcs->refcnt > 0);
	if (--pcs->refcnt == 0)
		do_release = true;
	SpinLockRelease(&pcs->refcnt_lock);

	if (!do_release)
		return;
	/* release resource */
	for (i=0; i < pcs->ncols; i++)
	{
		if (pcs->cdata[i].toast)
			pgstrom_put_toast_buffer(pcs->cdata[i].toast);
	}
	pgstrom_shmem_free(pcs);
}

/*
 * pgstrom_rcstore_fetch_slot
 *
 * A utility routine to fetch a record in row-/colum-store.
 */
TupleTableSlot *
pgstrom_rcstore_fetch_slot(TupleTableSlot *slot,
						   StromObject *rcstore,
						   int rowidx,
						   bool use_copy)
{
	if (StromTagIs(rcstore, TCacheRowStore))
	{
		tcache_row_store   *trs = (tcache_row_store *) rcstore;
		rs_tuple		   *rs_tup;
		HeapTuple			tuple;	

		rs_tup = kern_rowstore_get_tuple(&trs->kern, rowidx);
		if (!rs_tup)
			elog(ERROR, "Bug? rowid (%d) is out of range", rowidx);
		if (use_copy)
			tuple = heap_copytuple(&rs_tup->htup);
		else
			tuple = &rs_tup->htup;

		slot = ExecStoreTuple(tuple, slot, InvalidBuffer, use_copy);
	}
	else if (StromTagIs(rcstore, TCacheColumnStore))
	{
		tcache_column_store *tcs = (tcache_column_store *) rcstore;
		TupleDesc	tupdesc = slot->tts_tupleDescriptor;
		int			i;

		Assert(tcs->ncols == tupdesc->natts);
		Assert(rowidx < tcs->nrows);
		ExecStoreAllNullTuple(slot);
		for (i=0; i < tupdesc->natts; i++)
		{
			Form_pg_attribute	attr = tupdesc->attrs[i];

			/* uncached columns are dealt as null */
			if (!tcs->cdata[i].values)
				continue;
			/* null items are also considered as null */
			if (tcs->cdata[i].isnull &&
				att_isnull(rowidx, tcs->cdata[i].isnull))
				continue;

			/* otherwise, non-null values are stored in */
			slot->tts_isnull[i] = false;
			if (attr->attlen > 0)
			{
				char   *source = tcs->cdata[i].values + attr->attlen * rowidx;

				if (attr->attbyval)
					memcpy(&slot->tts_values[i], source, attr->attlen);
				else if (!use_copy)
					slot->tts_values[i] = PointerGetDatum(source);
				else
					slot->tts_values[i] =
						PointerGetDatum(pmemcpy(source, attr->attlen));
			}
			else if (!tcs->cdata[i].toast)
			{
				/* MEMO: In case of inlined varlena variables, we can have
				 * no toast buffer, instead of inline varlena.
				 */
				int		attlen = pgstrom_try_varlena_inline(attr);
				char   *vl_ptr = (tcs->cdata[i].values + attlen * rowidx);

				Assert(attlen > 0);
				if (!use_copy)
					slot->tts_values[i] = PointerGetDatum(vl_ptr);
				else
					slot->tts_values[i] =
						PointerGetDatum(pmemcpy(vl_ptr, VARSIZE_ANY(vl_ptr)));
			}
			else
			{
				cl_uint		vl_ofs = *((cl_uint *)(tcs->cdata[i].values +
												   sizeof(cl_uint) * rowidx));
				char	   *vl_ptr = ((char *)tcs->cdata[i].toast + vl_ofs);

				if (!use_copy)
					slot->tts_values[i] = PointerGetDatum(vl_ptr);
				else
					slot->tts_values[i] =
						PointerGetDatum(pmemcpy(vl_ptr, VARSIZE_ANY(vl_ptr)));
			}
		}
	}
	else
		elog(ERROR, "Bug? neither row- nor column-store");
	return slot;
}
#endif



#define IsSharedBuffer(PTR)						\
	((uintptr_t)(PTR) >= (uintptr_t)(BufferBlocks) &&	\
	 (uintptr_t)(PTR) < (uintptr_t)(BufferBlocks) + NBuffers * (Size) BLCKSZ)

bool
pgstrom_fetch_data_store(TupleTableSlot *slot,
						 pgstrom_data_store *pds,
						 int rowidx, bool use_copy)
{
	kern_data_store	   *kds = pds->kds;

	if (rowidx >= kds->nitems)
		return false;	/* out of range */

	if (!kds->is_column)
	{
		kern_rowitem   *ritem = KERN_DATA_STORE_ROWITEMS(kds) + rowidx;
		Buffer			buffer;
		Page			page;
		ItemId			lpp;

		Assert(ritem->block_id < kds->nblocks);
		page = pds->blocks[ritem->block_id];
		lpp = PageGetItemId(page, ritem->item_id);
		Assert(ItemIdIsNormal(lpp));

		tuple->t_data = (HeapTupleHeader) PageGetItem((Page) dp, lpp);
		tuple->t_len = ItemIdGetLength(lpp);
		ItemPointerSet(&(tuple->t_self), page, lineoff);


		if (IsSharedBuffer(page))

		if (!IsSharedBuffer(page))
			buffer = InvalidBuffer;
		else
			buffer = ((uintptr_t)(page) -
					  (uintptr_t)(BufferBlocks)) / BLCKSZ + 1;
		/* lock buffer and  */



	}
	else
	{

	}
	return false;
}

void
pgstrom_release_data_store(pgstrom_data_store *pds)
{
	kern_data_store	   *kds = pds->kds;
	int		i;

	Assert(kds->nblocks < pds->max_blocks);
	for (i=0; i < kds->nblocks; i++)
	{
		Page	block = pds->blocks[i];
		Buffer	buffer;

		if (block >= BufferBlocks &&
			block <  BufferBlocks + NBuffers * (Size) BLCKSZ)
		{
			buffer = ((uintptr_t)block -
					  (uintptr_t)BufferBlocks) / BLCKSZ + 1;
			Assert(BufferIsValid(buffer));
			ReleaseBuffer(buffer);
		}
		else
		{
			pgstrom_shmem_free(block);
		}
	}
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

	kds->length = baselen;
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
	Assert((uintptr_t)(KERN_DATA_STORE_ROWITEMS(kds) + nrooms) <=
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
	dlist_init(&pds->ktoast);
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
	dlist_init(&pds->ktoast);
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

	/*
	 * NOTE: Unlike other object types, pgstrom_data_store must be released
	 * by the backend process, not opencl server, because it touches some
	 * internal state of PostgreSQL core.
	 */
	if (do_release)
	{
		Assert(!pgstrom_i_am_clserv);
		pgstrom_release_data_store(pds);
	}
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
	kri = KERN_DATA_STORE_ROWITEMS(kds) + kds->nitems;
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

		kri->block_id = kds->nblocks;
		kri->item_id = lineoff;
		kri++;
		ntup++;
	}
	LockBuffer(buffer, BUFFER_LOCK_UNLOCK);
	Assert(ntup <= MaxHeapTuplesPerPage);
	Assert(kds->nitems + ntup <= kds->nrooms);
	kds->nitems += ntup;

	/* TODO: we have to copy the contents of shared-buffer if it is
	 * private buffer, thus assigned of the private memory area
	 */
	pds->blocks[kds->nblocks++] = page;

	return ntup;
}

bool
pgstrom_data_store_insert_tuple(pgstrom_data_store *pds,
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
			if (dlist_is_empty(&pds->ktoast))
			{
				ktoast = pgstrom_shmem_alloc(TOASTBUF_UNITSZ);
				if (!ktoast)
					elog(ERROR, "out of shared memory");
				ktoast->length = TOASTBUF_UNITSZ;
				ktoast->usage = offsetof(kern_toastbuf, data[0]);
				dlist_push_tail(&pds->ktoast, &ktoast->dnode);
			}
			else
			{
				ktoast = dlist_container(kern_toastbuf, dnode,
										 dlist_tail_node(&pds->ktoast));
				if (ktoast->usage + INTALIGN(vl_len) > ktoast->length)
				{
					ktoast = pgstrom_shmem_alloc(TOASTBUF_UNITSZ);
					if (!ktoast)
						elog(ERROR, "out of shared memory");
					ktoast->length = TOASTBUF_UNITSZ;
					ktoast->usage = offsetof(kern_toastbuf, data[0]);
					dlist_push_tail(&pds->ktoast, &ktoast->dnode);
				}
			}
			Assert(ktoast->usage + INTALIGN(vl_len) <= ktoast->length);
			vl_ofs[j] = ktoast->usage;
			memcpy((cl_char *)ktoast + ktoast->usage, vl_data, vl_len);
			ktoast->usage += INTALIGN(vl_len);
		}
	}
	return false;
}
