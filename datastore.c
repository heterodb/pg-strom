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
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "pg_strom.h"

/*
 * pgstrom_get_vrelation()
 *
 * it increments reference counter of the vrelation
 */
pgstrom_vrelation *
pgstrom_get_vrelation(pgstrom_vrelation *vrel)
{
	SpinLockAcquire(&vrel->lock);
	Assert(vrel->refcnt > 0);
	vrel->refcnt++;
	SpinLockRelease(&vrel->lock);
	return vrel;
}

/*
 * pgstrom_put_vrelation()
 *
 */
void
pgstrom_put_vrelation(pgstrom_vrelation *vrel)
{
	bool	do_release = false;
	int		i;

	SpinLockAcquire(&vrel->lock);
	Assert(vrel->refcnt > 0);
	if (--vrel->refcnt == 0)
		do_release = true;
	SpinLockRelease(&vrel->lock);

	if (do_release)
	{
		for (i=0; i < vrel->rcsnums; i++)
			pgstrom_put_rcstore(vrel->rcstore[i]);
		pgstrom_shmem_free(vrel);
	}
}

/*
 * pgstrom_create_vrelation_head
 *
 * It constructs header portion of virtual-relation object according to
 * the supplied tuple-descriptor and pair of referenced relidx/attidx.
 * Note that it just allocate header portion on private memory, so caller
 * has to populate this template using pgstrom_create_vrelation to create
 * actual vrelation object on the shared memory region.
 */
pgstrom_vrelation *
pgstrom_create_vrelation_head(TupleDesc tupdesc,
							  List *vtlist_relidx,
							  List *vtlist_attidx)
{
	pgstrom_vrelation *vrel;
	int			vt_index;
	ListCell   *lc1;
	ListCell   *lc2;

	Assert(list_length(vtlist_relidx) == tupdesc->natts &&
		   list_length(vtlist_attidx) == tupdesc->natts);
	vrel = palloc0(MAXALIGN(offsetof(pgstrom_vrelation,
									 vtlist[tupdesc->natts])));
	vrel->sobj.stag = StromTag_VirtRelation;
	SpinLockInit(&vrel->lock);
	vrel->refcnt = 1;
	vrel->ncols = tupdesc->natts;
	vrel->rcsnums = 0;		/* to be set later */
	vrel->rcstore = NULL;	/* to be set later */
	vrel->kern = NULL;		/* to be set later */

	vt_index = 0;
	forboth (lc1, vtlist_relidx,
			 lc2, vtlist_attidx)
	{
		Form_pg_attribute attr = tupdesc->attrs[vt_index];
		int		vt_relidx = lfirst_int(lc1);
		int		vt_attidx = lfirst_int(lc2);

		vrel->vtlist[vt_index].attnotnull = attr->attnotnull;
		vrel->vtlist[vt_index].attalign = typealign_get_width(attr->attalign);
		vrel->vtlist[vt_index].attlen = attr->attlen;
		vrel->vtlist[vt_index].vrelidx = vt_relidx;
		vrel->vtlist[vt_index].vattidx = vt_attidx;
		vt_index++;
	}
	return vrel;
}

/*
 * pgstrom_populate_vrelation
 *
 * it construct a virtual-relation object according to the prepared
 * header portion of virtual-relation object.
 */
pgstrom_vrelation *
pgstrom_populate_vrelation(pgstrom_vrelation *vrel_head,
						   int rcsnums, StromObject **rcstore,
						   cl_uint nitems, cl_uint nrooms)
{
	pgstrom_vrelation *vrel;
	Size		length;
	Size		offset;
	int			i, ncols;

	ncols = vrel_head->ncols;
	length = (MAXALIGN(offsetof(pgstrom_vrelation, vtlist[ncols])) +
			  MAXALIGN(sizeof(StromObject *) * rcsnums) +
			  STROMALIGN(offsetof(kern_vrelation, rindex[rcsnums * nrooms])));
	vrel = pgstrom_shmem_alloc(length);
	if (!vrel)
		elog(ERROR, "out of shared memory");

	offset = MAXALIGN(offsetof(pgstrom_vrelation, vtlist[ncols]));
	memcpy(vrel, vrel_head, offset);
	Assert(vrel->sobj.stag == StromTag_VirtRelation);
	Assert(SpinLockFree(&vrel->lock));
	Assert(vrel->refcnt == 1);
	vrel->rcsnums = rcsnums;
	vrel->rcstore = (StromObject **)((char *)vrel + offset);
	offset += MAXALIGN(sizeof(StromObject *) * rcsnums);
	for (i=0; i < rcsnums; i++)
		vrel->rcstore[i] = pgstrom_get_rcstore(rcstore[i]);
	vrel->kern = (kern_vrelation *)((char *)vrel + offset);

	for (i=0; i < ncols; i++)
	{
		int		vt_relidx = vrel->vtlist[i].vrelidx;
		int		vt_attidx = vrel->vtlist[i].vattidx;

		if (vt_relidx < 0 || vt_relidx >= vrel->rcsnums)
			elog(ERROR, "vt_relidx %d is out of range", vt_relidx);

		if (StromTagIs(rcstore[vt_relidx], TCacheRowStore))
		{
			tcache_row_store *trs
				= (tcache_row_store *)rcstore[vt_relidx];
			if (vt_attidx < 0 || vt_attidx >= trs->kern.ncols)
				elog(ERROR, "vt_attidx %d is out of range", vt_relidx);
		}
		else if (StromTagIs(rcstore[vt_relidx], TCacheColumnStore))
		{
			tcache_column_store *tcs
				= (tcache_column_store *)rcstore[vt_relidx];
			if (vt_attidx < 0 || vt_attidx >= tcs->ncols)
				elog(ERROR, "vt_attidx %d is out of range", vt_relidx);
		}
		else
			elog(ERROR, "bug? neither row- nor column- store: %s",
				 StromTagGetLabel(rcstore[vt_relidx]));
	}
	/* also set fields of kern_vrelation */
	memset(vrel->kern, 0, sizeof(kern_vrelation));
	length = offsetof(kern_vrelation, rindex[rcsnums * nrooms]);
	vrel->kern->length = STROMALIGN(length);
	vrel->kern->nrels  = rcsnums;
	vrel->kern->nitems = nitems;
	vrel->kern->nrooms = nrooms;

	return vrel;
}

/*
 * pgstrom_can_vrelation_projection
 *
 * It checks whether the supplied tlist can be handled by simple projection.
 * If available, it returns simple-projection list.
 */
List *
pgstrom_can_vrelation_projection(List *targetlist)
{
	List	   *result = NIL;
	ListCell   *cell;

	foreach (cell, targetlist)
	{
		TargetEntry	   *tle = lfirst(cell);
		Var			   *var;

		if (!IsA(tle->expr, Var))
			return NIL;
		var = (Var *) tle->expr;
		Assert(var->varno == INDEX_VAR || !IS_SPECIAL_VARNO(var->varno));
		/*
		 * FIXME: we don't support to reference system columns using
		 * vrelation structure. It should be fixed up later.
		 */
		if (var->varattno < 1)
			return NIL;
		result = lappend_int(result, var->varattno);
	}
	return result;
}

/*
 * pgstrom_apply_vrelation_projection
 *
 * It applies simple projection on the supplied vrelation.
 */
pgstrom_vrelation *
pgstrom_apply_vrelation_projection(pgstrom_vrelation *vrel, List *vrel_proj)
{
	pgstrom_vrelation  *temp;
	ListCell   *cell;
	int			i, j;

	Assert(vrel_proj != NIL);

	Assert(list_length(vrel_proj) <= vrel->ncols);
	temp = palloc0(offsetof(pgstrom_vrelation, vtlist[vrel->ncols]));
	i = 0;
	foreach (cell, vrel_proj)
	{
		j = lfirst_int(cell) - 1;

		Assert(j < vrel->ncols);
		temp->vtlist[i++] = vrel->vtlist[j];
	}
	memcpy(vrel->vtlist, temp->vtlist,
		   offsetof(pgstrom_vrelation, vtlist[j]) -
		   offsetof(pgstrom_vrelation, vtlist[0]));
	vrel->ncols = j;
	pfree(temp);

	return vrel;
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
kparam_make_kds_head(TupleDesc tupdesc,
					 Bitmapset *referenced,
					 cl_uint nsyscols,
					 cl_uint nrooms)
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
	kds_head->nitems = 0;
	kds_head->nrooms = nrooms;

	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupdesc->attrs[i];

		j = attr->attnum - FirstLowInvalidHeapAttributeNumber;

		kds_head->colmeta[i].attnotnull = attr->attnotnull;
		kds_head->colmeta[i].attalign = typealign_get_width(attr->attalign);
		kds_head->colmeta[i].attlen = attr->attlen;
		if (!bms_is_member(j, referenced))
			kds_head->colmeta[i].attvalid = false;
		else
			kds_head->colmeta[i].attvalid = true;
		/* rest of fields shall be set later */
	}

	return result;
}

void
kparam_refresh_kds_head(kern_parambuf *kparams,
						pgstrom_vrelation *vrel,
						cl_uint nitems)
{
	kern_data_store *kds_head = KPARAM_GET_KDS_HEAD(kparams);
	Size	length;
	Size   *rs_ofs = palloc0(sizeof(Size) * vrel->rcsnums);
	int		i, ncols = kds_head->ncols;

	Assert(ncols == vrel->ncols);
	length = STROMALIGN(offsetof(kern_data_store, colmeta[ncols]));
	kds_head->nitems = nitems;
	kds_head->nrooms = nitems;

	for (i=0; i < ncols; i++)
	{
		StromObject	*rcs;
		int			vt_relidx;
		int			vt_attidx;

		if (!kds_head->colmeta[i].attvalid)
			continue;

		vt_relidx = vrel->vtlist[i].vrelidx;
		vt_attidx = vrel->vtlist[i].vattidx;
		Assert(vt_relidx < vrel->rcsnums);
		rcs = vrel->rcstore[vt_relidx];

		if (StromTagIs(rcs, TCacheRowStore))
		{
			tcache_row_store *trs = (tcache_row_store *)rcs;

			kds_head->colmeta[i].rs_attnum = vt_attidx;
			/*
			 * In case when multiple columns in a particular row-store
			 * is referenced, we don't need to transfer the row-store
			 * twice. So, rs_ofs[] will point same region.
			 */
			if (rs_ofs[vt_relidx] > 0)
				kds_head->colmeta[i].ds_offset = rs_ofs[vt_relidx];
			else
			{
				kds_head->colmeta[i].ds_offset = length;
				rs_ofs[vt_relidx] = length;
				length += STROMALIGN(trs->kern.length);
			}
		}
		else
		{
			Assert(StromTagIs(rcs, TCacheColumnStore));

			kds_head->colmeta[i].rs_attnum = 0;
			kds_head->colmeta[i].ds_offset = length;
			if (!kds_head->colmeta[i].attnotnull)
				length += STROMALIGN((kds_head->nrooms +
									  BITS_PER_BYTE - 1) / BITS_PER_BYTE);
			length += STROMALIGN((kds_head->colmeta[i].attlen > 0
								  ? kds_head->colmeta[i].attlen
								  : sizeof(cl_uint)) * kds_head->nrooms);
		}
	}
	kds_head->length = length;
	pfree(rs_ofs);
}

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
						   pgstrom_vrelation *vrel)
{
	kern_data_store *kds_head = KPARAM_GET_KDS_HEAD(kparams);
	kern_toastbuf *ktoast_head = KPARAM_GET_KTOAST_HEAD(kparams);
	StromObject	*rcs;
	int			vt_relidx;
	int			vt_attidx;
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

		/* column is not referenced or fixed-length variable */
		if (!kds_head->colmeta[i].attvalid ||
			kds_head->colmeta[i].attlen > 0)
			continue;

		/* row-store does not need individula toast buffer */
		vt_relidx = vrel->vtlist[i].vrelidx;
		vt_attidx = vrel->vtlist[i].vattidx;
		Assert(vt_relidx < vrel->rcsnums);
		rcs = vrel->rcstore[vt_relidx];

		if (StromTagIs(rcs, TCacheColumnStore))
		{
			tcache_column_store *tcs = (tcache_column_store *)rcs;

			Assert(vt_attidx >= 0 && vt_attidx < tcs->ncols);
			if (tcs->cdata[vt_attidx].toast)
			{
				has_toast = true;
				ktoast_head->coldir[i] = offset;
				offset += STROMALIGN(tcs->cdata[i].toast->tbuf_length);
			}
		}
	}
	/* mark KPARAM_1 as null */
	if (!has_toast)
		kparams->poffset[1] = 0;	/* mark it as null */
}

#if 0
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
#endif

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
