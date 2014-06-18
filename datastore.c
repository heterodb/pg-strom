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
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
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
					kpbuf->poffset[index] = 0;	/* null */
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
 * pgstrom_release_bulk_slot
 *
 * It releases the supplied pgstrom_bulk_slot object once constructed.
 */
void
pgstrom_release_bulk_slot(pgstrom_bulk_slot *bulk_slot)
{
	/* unlink the referenced row or column store */
	pgstrom_untrack_object(bulk_slot->rc_store);
	pgstrom_put_rcstore(bulk_slot->rc_store);
	pfree(bulk_slot);
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

	if (gpuscan_support_multi_exec((const CustomPlanState *) ps) ||
		gpusort_support_multi_exec((const CustomPlanState *) ps) ||
		gpuhashjoin_support_multi_exec((const CustomPlanState *) ps))
		return true;

	return false;
}


/*
 * kparam_make_attrefs_by_resnums
 *
 * makes an array to inform which columns (in row format) are referenced.
 * usually it is informed as kparam_0 constant
 */
bytea *
kparam_make_attrefs_by_resnums(TupleDesc tupdesc, List *attnums_list)
{
	bytea	   *result;
	cl_char	   *refatts;
	AttrNumber	anum;
	AttrNumber	anum_last = 0;
	ListCell   *lc;

	result = palloc0(VARHDRSZ + sizeof(cl_char) * tupdesc->natts);
	SET_VARSIZE(result, VARHDRSZ + sizeof(cl_char) * tupdesc->natts);
	refatts = (cl_char *)VARDATA(result);
	foreach (lc, attnums_list)
	{
		anum = lfirst_int(lc);
		Assert(anum > 0 && anum <= tupdesc->natts);
		refatts[anum - 1] = 1;
		anum_last = anum;
	}
	if (anum_last > 0)
		refatts[anum_last - 1] = -1;	/* end of reference marker */

	return result;
}

/*
 * kparam_make_attrefs
 *
 * same as kparam_make_attrefs_by_resnums, but extract varattno from
 * used_vars list;
 */
bytea *
kparam_make_attrefs(TupleDesc tupdesc,
					List *used_vars, Index varno)
{
	List	   *resnums = NIL;
	ListCell   *cell;
	bytea	   *result;

	foreach (cell, used_vars)
	{
		Var	   *var = lfirst(cell);

		if (var->varno != varno)
			continue;
		Assert(var->varattno > 0 && var->varattno <= tupdesc->natts);
		resnums = lappend_int(resnums, var->varattno);
	}
	result = kparam_make_attrefs_by_resnums(tupdesc, resnums);
	list_free(resnums);
	return result;
}

bytea *
kparam_make_kcs_head(TupleDesc tupdesc,
					 cl_char *attrefs,
					 cl_uint nsyscols,
					 cl_uint nrooms)
{
	kern_column_store *kcs_head;
	bytea	   *result;
	Size		length;
	int			i, j, ncols = 0;

	/* count-up number of referenced columns */
	for (i=0; i < tupdesc->natts; i++)
	{
		if (attrefs[i])
			ncols++;
	}
	ncols += nsyscols;

	length = STROMALIGN(offsetof(kern_column_store, colmeta[ncols]));
	result = palloc0(VARHDRSZ + length);
	SET_VARSIZE(result, VARHDRSZ + length);

	kcs_head = (kern_column_store *) VARDATA(result);
	kcs_head->ncols = ncols;
	kcs_head->nrows = 0;
	kcs_head->nrooms = nrooms;

	for (i=0, j=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupdesc->attrs[i];

		if (!attrefs[i])
			continue;
		kcs_head->colmeta[j].attnotnull = attr->attnotnull;
		kcs_head->colmeta[j].attalign = typealign_get_width(attr->attalign);
		kcs_head->colmeta[j].attlen = attr->attlen;
		kcs_head->colmeta[j].cs_ofs = length;
		if (!attr->attnotnull)
			length += STROMALIGN((nrooms + BITS_PER_BYTE - 1) / BITS_PER_BYTE);
		length += STROMALIGN((attr->attlen > 0
							  ? attr->attlen
							  : sizeof(cl_uint)) * nrooms);
		j++;
	}
	Assert(j + nsyscols == ncols);
	kcs_head->length = length;

	return result;
}

void
kparam_refresh_kcs_head(kern_parambuf *kparams, cl_uint nrows, cl_uint nrooms)
{
	bytea  *kparam_1 = kparam_get_value(kparams, 1);
	kern_column_store *kcs_head = (kern_column_store *)VARDATA_ANY(kparam_1);
	Size	length;
	int		i;

	length = STROMALIGN(offsetof(kern_column_store,
								 colmeta[kcs_head->ncols]));
	kcs_head->nrows = nrows;
	kcs_head->nrooms = nrooms;
	for (i=0; i < kcs_head->ncols; i++)
	{
		kcs_head->colmeta[i].cs_ofs = length;
		if (!kcs_head->colmeta[i].attnotnull)
			length += STROMALIGN((nrooms + BITS_PER_BYTE - 1) / BITS_PER_BYTE);
		length += STROMALIGN((kcs_head->colmeta[i].attlen > 0
							  ? kcs_head->colmeta[i].attlen
							  : sizeof(cl_uint)) * nrooms);
	}
	kcs_head->length = length;
}

bytea *
kparam_make_ktoast_head(TupleDesc tupdesc,
						cl_char *attrefs,
						cl_uint nsyscols)
{
	kern_toastbuf *ktoast_head;
	bytea	   *result;
	Size		length;
	int			i, ncols = 0;

	/* count-up number of referenced columns */
	for (i=0; i < tupdesc->natts; i++)
	{
		if (attrefs[i])
			ncols++;
	}
	ncols += nsyscols;

	length = STROMALIGN(offsetof(kern_toastbuf, coldir[ncols]));
	result = palloc0(VARHDRSZ + length);
	SET_VARSIZE(result, VARHDRSZ + length);

	ktoast_head = (kern_toastbuf *) VARDATA(result);
	memset(ktoast_head, 0, length);
	ktoast_head->length = TOASTBUF_MAGIC;
	ktoast_head->ncols = ncols;
	return result;
}

void
kparam_refresh_ktoast_head(kern_parambuf *kparams,
						   StromObject *rcstore)
{
	if (!StromTagIs(rcstore, TCacheColumnStore))
		kparams->poffset[2] = 0;
	else
	{
		tcache_column_store *tcs = (tcache_column_store *)rcstore;
		bytea	   *kparam_0 = kparam_get_value(kparams, 0);
		bytea	   *kparam_1 = kparam_get_value(kparams, 1);
		bytea	   *kparam_2 = kparam_get_value(kparams, 2);
		cl_char	   *attrefs = (cl_char *)VARDATA_ANY(kparam_0);
		kern_column_store *kcs_head ;
		kern_toastbuf *ktoast_head;
		Size		offset;
		int			i, j;
		bool		has_toast = false;

		kcs_head = (kern_column_store *) VARDATA_ANY(kparam_1);
		ktoast_head = (kern_toastbuf *) VARDATA_ANY(kparam_2);
		ktoast_head->length = TOASTBUF_MAGIC;
		ktoast_head->ncols = kcs_head->ncols;
		offset = STROMALIGN(offsetof(kern_toastbuf,
									 coldir[ktoast_head->ncols]));
		for (i=0, j=0; i < tcs->ncols; i++)
		{
			if (!attrefs[i])
				continue;
			if (!tcs->cdata[i].toast)
				ktoast_head->coldir[j] = (cl_uint)(-1);
			else
			{
				has_toast = true;
				ktoast_head->coldir[j] = offset;
				offset += STROMALIGN(tcs->cdata[i].toast->tbuf_length);
			}
			j++;
		}
		if (!has_toast)
			kparams->poffset[2] = 0;
	}
}


bytea *
kparam_make_kprojection(List *target_list)
{
	bytea	   *result;
	kern_projection *kproj;
	Size		length;
	int			ncols = list_length(target_list);
	int			i_col;
	ListCell   *cell;

	length = VARHDRSZ + offsetof(kern_projection, origins[ncols]);
	result = palloc0(length);
	SET_VARSIZE(result, length);
	kproj = (kern_projection *)VARDATA(result);
	kproj->length = length;
	kproj->ncols = ncols;
	kproj->dprog_key = 0;	/* to be set later, by caller */

	i_col = 0;
	foreach (cell, target_list)
	{
		TargetEntry	*tle = lfirst(cell);
		Var	   *var;
		int16	typlen;
		bool	typbyval;
		char	typalign;

		Assert(IsA(tle, TargetEntry));
		if (!IsA(tle->expr, Var))
			goto out_unavailable;
		var = (Var *)tle->expr;

		/*
		 * NOTE: we don't care about collation on projection because
		 * pseudo-tlist shall be actually handled on backend-side
		 * projection.
		 */
		if (var->varlevelsup > 0)
			goto out_unavailable;
		if (var->varno != INNER_VAR && var->varno != OUTER_VAR)
			goto out_unavailable;

		get_typlenbyvalalign(var->vartype,
							 &typlen,
							 &typbyval,
							 &typalign);

		kproj->origins[i_col].colmeta.attnotnull = false;
		kproj->origins[i_col].colmeta.attalign = typealign_get_width(typalign);
		kproj->origins[i_col].colmeta.attlen = typlen;
		kproj->origins[i_col].colmeta.cs_ofs = -1;	/* to be set later */
		kproj->origins[i_col].resjunk = tle->resjunk;
		if (var->varno == INNER_VAR)
			kproj->origins[i_col].is_outer = false;
		else
			kproj->origins[i_col].is_outer = true;
		kproj->origins[i_col].resno = var->varattno;

		i_col++;
	}
	return result;

out_unavailable:
	pfree(result);
	return NULL;
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
 * pgstrom_create_column_store
 *
 * creates a column-store on shared memory segment, but not linked to
 * a particular tcache structure.
 */
tcache_column_store *
pgstrom_create_column_store_with_projection(kern_projection *kproj,
											cl_uint nitems,
											bool with_syscols)
{
	tcache_column_store	*tcs;
	Size		length;
	Size		offset;
	int			i;

	length = MAXALIGN(offsetof(tcache_column_store, cdata[kproj->ncols]));
	if (with_syscols)
	{
		length += MAXALIGN(sizeof(ItemPointerData) * nitems);
		length += MAXALIGN(sizeof(HeapTupleHeaderData) * nitems);
	}

	for (i=0; i < kproj->ncols; i++)
	{
		if (!kproj->origins[i].colmeta.attnotnull)
			length += MAXALIGN((nitems + BITS_PER_BYTE - 1) / BITS_PER_BYTE);
		length += MAXALIGN((kproj->origins[i].colmeta.attlen > 0
							? kproj->origins[i].colmeta.attlen
							: sizeof(cl_uint)) * nitems);
	}
	tcs = pgstrom_shmem_alloc(length);
	if (!tcs)
		return NULL;	/* out of shared memory! */
	offset = MAXALIGN(offsetof(tcache_column_store, cdata[kproj->ncols]));

	memset(tcs, 0, offset);
	tcs->sobj.stag = StromTag_TCacheColumnStore;
	SpinLockInit(&tcs->refcnt_lock);
	tcs->refcnt = 1;
	tcs->ncols = kproj->ncols;
	if (with_syscols)
	{
		/* array of item-pointers */
		tcs->ctids = (ItemPointerData *)((char *)tcs + offset);
		offset += MAXALIGN(sizeof(ItemPointerData) * nitems);

		/* array of other system columns */
		tcs->theads = (HeapTupleHeaderData *)((char *)tcs + offset);
		offset += MAXALIGN(sizeof(HeapTupleHeaderData) * nitems);
	}

	for (i=0; i < kproj->ncols; i++)
	{
		if (kproj->origins[i].colmeta.attnotnull)
			tcs->cdata[i].isnull = NULL;
		else
		{
			tcs->cdata[i].isnull = (uint8 *)((char *)tcs + offset);
			offset += MAXALIGN((nitems + BITS_PER_BYTE - 1) / BITS_PER_BYTE);
		}
		tcs->cdata[i].values = ((char *)tcs + offset);
		if (kproj->origins[i].colmeta.attlen > 0)
		{
			offset += MAXALIGN(kproj->origins[i].colmeta.attlen * nitems);
			tcs->cdata[i].toast = NULL;
		}
		else
		{
			offset += MAXALIGN(sizeof(cl_uint) * nitems);
			tcs->cdata[i].toast = pgstrom_create_toast_buffer(0);
			if (!tcs->cdata[i].toast)
			{
				pgstrom_put_column_store(tcs);
				return NULL;
			}
		}
	}
	Assert(offset == length);

	return tcs;
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
