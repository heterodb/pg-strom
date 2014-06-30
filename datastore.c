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
#include "port.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
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

#if 0
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

	SpinLockAcquire(&vrel->lock);
	Assert(vrel->refcnt > 0);
	if (--vrel->refcnt == 0)
		do_release = true;
	SpinLockRelease(&vrel->lock);

	if (do_release)
	{
		if (vrel->rcstore)
			pgstrom_put_rcstore(vrel->rcstore);
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
static int
vrelation_vtlist_sortcomp(const void *a, const void *b, void *arg)
{
	pgstrom_vrelation  *vrel = arg;
	AttrNumber i = *((AttrNumber *)a);
	AttrNumber j = *((AttrNumber *)b);

	if (vrel->vtlist[i].vrelidx < vrel->vtlist[j].vrelidx)
		return -1;
	if (vrel->vtlist[i].vrelidx > vrel->vtlist[j].vrelidx)
		return 1;
	if (vrel->vtlist[i].vrelidx == vrel->vtlist[j].vrelidx &&
		vrel->vtlist[i].vattsrc < vrel->vtlist[j].vattsrc)
		return -1;
	if (vrel->vtlist[i].vrelidx == vrel->vtlist[j].vrelidx &&
		vrel->vtlist[i].vattsrc > vrel->vtlist[j].vattsrc)
		return 1;
	return 0;
}

pgstrom_vrelation *
pgstrom_create_vrelation_head(TupleDesc tupdesc,
							  List *vtlist_relidx,
							  List *vtlist_attidx)
{
	pgstrom_vrelation *vrel;
	ListCell   *lc1;
	ListCell   *lc2;
	int			i, j;

	Assert(list_length(vtlist_relidx) == tupdesc->natts &&
		   list_length(vtlist_attidx) == tupdesc->natts);
	vrel = palloc0(MAXALIGN(offsetof(pgstrom_vrelation,
									 vtlist[tupdesc->natts + 1])) +
				   MAXALIGN(sizeof(AttrNumber) * (tupdesc->natts)));
	vrel->sobj.stag = StromTag_VirtRelation;
	SpinLockInit(&vrel->lock);
	vrel->refcnt = 1;
	vrel->ncols = tupdesc->natts;
	vrel->rcstore = NULL;	/* to be set later */
	vrel->kern = NULL;		/* to be set later */
	vrel->vtsources = (AttrNumber *)
		((char *)vrel + MAXALIGN(offsetof(pgstrom_vrelation,
										  vtlist[tupdesc->natts])));
	i = 0;
	forboth (lc1, vtlist_relidx,
			 lc2, vtlist_attidx)
	{
		Form_pg_attribute attr = tupdesc->attrs[i];
		int		vt_relidx = lfirst_int(lc1);
		int		vt_attsrc = lfirst_int(lc2);
		int		vattwidth;

		vrel->vtlist[i].attnotnull = attr->attnotnull;
		vrel->vtlist[i].attalign = typealign_get_width(attr->attalign);
		vrel->vtlist[i].attlen = attr->attlen;
		vrel->vtlist[i].vrelidx = vt_relidx;
		vrel->vtlist[i].vattsrc = vt_attsrc;
		vrel->vtlist[i].vattdst = i;
		vattwidth = get_attavgwidth(attr->attrelid, attr->attnum);
		if (vattwidth == 0)
			vattwidth = get_typavgwidth(attr->atttypid, attr->atttypmod);
		vrel->vtlist[i].vattwidth = vattwidth;

		vrel->vtsources[i] = i;
		i++;
	}
	Assert(i == tupdesc->natts);
	/* watch of loop */
	memset(&vrel->vtlist[i], -1, sizeof(vrelation_colmeta));
	vrel->vtsources[i] = i;
	qsort_arg(vrel->vtsources, tupdesc->natts, sizeof(AttrNumber),
			  vrelation_vtlist_sortcomp, vrel);

	for (i=0; i <= tupdesc->natts; i++)
	{
		j = vrel->vtsources[i];
		elog(INFO, "vtlist {attnotnull=%d attalign=%d attlen=%d vrelidx=%d vattsrc=%d vattdst=%d vattwidth=%d}", vrel->vtlist[j].attnotnull, vrel->vtlist[j].attalign, vrel->vtlist[j].attlen, vrel->vtlist[j].vrelidx, vrel->vtlist[j].vattsrc, vrel->vtlist[j].vattdst, vrel->vtlist[j].vattwidth);
	}
	return vrel;
}

/*
 * pgstrom_populate_vrelation
 *
 * it construct a virtual-relation object according to the prepared
 * header portion of virtual-relation object.
 *
 * NOTE: if nrooms == 0, it means all visible vrelation w/o rindex
 */
pgstrom_vrelation *
pgstrom_populate_vrelation(pgstrom_vrelation *vrel_head,
						   StromObject *rcstore,	/* row-/column-store */
						   cl_uint nrels,	/* number of source relations */
						   cl_uint nitems,	/* number of items in use */
						   cl_uint nrooms)	/* number of capable rooms */
{
	pgstrom_vrelation *vrel;
	Size		length;
	Size		offset;
	int			ncols;

	ncols = vrel_head->ncols;
	length = (MAXALIGN(offsetof(pgstrom_vrelation, vtlist[ncols + 1])) +
			  MAXALIGN(sizeof(AttrNumber) * ncols) +
			  STROMALIGN(offsetof(kern_vrelation, rindex[nrels * nrooms])));
	vrel = pgstrom_shmem_alloc(length);
	if (!vrel)
		return NULL;	/* out of shared memory */

	offset = (MAXALIGN(offsetof(pgstrom_vrelation, vtlist[ncols + 1])) +
			  MAXALIGN(sizeof(AttrNumber) * ncols));
	memcpy(vrel, vrel_head, offset);
	Assert(vrel->sobj.stag == StromTag_VirtRelation);
	SpinLockInit(&vrel->lock);
	vrel->refcnt = 1;
	vrel->rcstore = NULL;	/* result row-/column-store to be set later */

	/* acquire an row-/column-store, if given */
	if (rcstore)
		vrel->rcstore = pgstrom_get_rcstore(rcstore);
	/* copy the projection hints */
	vrel->vtsources = (AttrNumber *)
		((char *)vrel + STROMALIGN(offsetof(pgstrom_vrelation,
											vtlist[ncols + 1])));
	memcpy(vrel->vtsources, vrel_head->vtsources, sizeof(AttrNumber) * ncols);

	/* also set up fields of kern_vrelation */
	vrel->kern = (kern_vrelation *)((char *)vrel + offset);
	memset(vrel->kern, 0, sizeof(kern_vrelation));
	vrel->kern->nrels  = nrels;		/* number of source relations */
	vrel->kern->nitems = nitems;	/* number of items in use */
	vrel->kern->nrooms = nrooms;	/* number of capable rooms */
	vrel->kern->has_rechecks = false;
	vrel->kern->all_visible  = (nrooms == 0 ? true : false);

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

	/*
	 * FIXME: row-store needs to be materialized, if incompatible
	 *
	 *
	 */


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
#endif
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
 * pgstrom_create_bulkslot
 *
 * construction of a new bulkslot according to the parameters
 */
pgstrom_bulkslot *
pgstrom_create_bulkslot(StromObject *rc_store,
						List *bulk_attmap,
						cl_uint nitems,
						cl_uint nrooms)
{
	pgstrom_bulkslot *bulk = palloc(offsetof(pgstrom_bulkslot,
											 rindex[nrooms]));
	Assert(StromTagIs(rc_store, TCacheRowStore) ||
		   StromTagIs(rc_store, TCacheColumnStore));
	bulk->rc_store = pgstrom_get_rcstore(rc_store);
	pgstrom_track_object(bulk->rc_store, 0);
	bulk->nitems = nitems;
	bulk->attmap = list_copy(bulk_attmap);

	return bulk;
}

/*
 * pgstrom_release_bulk_slot
 *
 * It releases the supplied pgstrom_bulk_slot object once constructed.
 */
void
pgstrom_release_bulkslot(pgstrom_bulkslot *bulk)
{
	/* unlink referenced row- or column-store */
	if (bulk->rc_store)
	{
		pgstrom_untrack_object(bulk->rc_store);
		pgstrom_put_rcstore(bulk->rc_store);
	}
	pfree(bulk);
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
	kds_head->nitems = nitems;
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
				length += STROMALIGN((kds_head->nitems +
									  BITS_PER_BYTE - 1) / BITS_PER_BYTE);
			length += STROMALIGN((kds_head->colmeta[i].attlen > 0
								  ? kds_head->colmeta[i].attlen
								  : sizeof(cl_uint)) * kds_head->nitems);
		}
	}
	else
		elog(ERROR, "bug? neither row- nor column-store");

	kds_head->length = length;
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

#if 0
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
#endif

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

#if 0
/*
 * extract_rcstore
 *
 *
 *
 *
 *
 */
static int
extract_rcstore(pgstrom_vrelation *vrel, int vtsrc_index,
				cl_int rel_index, StromObject **rcstore,
				cl_uint row_index,
				void **dest_values, bool *dest_has_null)
{
	cl_uint			first_vtsrc_index = vtsrc_index;
	vrelation_colmeta *vtlist;

	Assert(vrel->vtsources[vtsrc_index] < vrel->ncols);
	vtlist = &vrel->vtlist[vrel->vtsources[vtsrc_index++]];
	if (vtlist->vrelidx != rel_index)
		return 0;	/* no column should be extracted from this row */

	if (StromTagIs(rcstore[rel_index], TCacheRowStore))
	{
		tcache_row_store *trs = (tcache_row_store *) rcstore[rel_index];
		rs_tuple	   *rs_tup;
		HeapTupleHeader	htup;
		cl_uint			offset;
		cl_uint			nattrs;
		cl_int			anum;
		bool			has_null;

		rs_tup = kern_rowstore_get_tuple(&trs->kern, row_index);
		if (!rs_tup)
			goto all_nulls;

		htup = &rs_tup->data;
		nattrs = HeapTupleHeaderGetNatts(htup);
		has_null = ((htup->t_infomask & HEAP_HASNULL) != 0);

		offset = htup->t_hoff;
		for (anum = 0; anum < nattrs; anum++)
		{
			int		attlen   = trs->kern.colmeta[anum].attlen;
			int		attalign = trs->kern.colmeta[anum].attalign;
			char   *addr;

			/* no need to scan this tuple any more */
			if (vtlist->vrelidx != rel_index)
				break;

			if (has_null && att_isnull(anum, htup->t_bits))
				addr = NULL;
			else
			{
				if (attlen > 0)
					offset = TYPEALIGN(attalign, offset);
				else if (!VARATT_NOT_PAD_BYTE((uintptr_t)htup + offset))
					offset = TYPEALIGN(attalign, offset);
				addr = (char *)htup + offset;
			}

			if (vtlist->vattsrc == anum)
			{
				Assert(vtlist->vattdst < vrel->ncols);
				dest_values[vtlist->vattdst] = addr;
				if (!addr)
					*dest_has_null = true;
				Assert(vrel->vtsources[vtsrc_index] < vrel->ncols);
				vtlist = &vrel->vtlist[vrel->vtsources[vtsrc_index++]];
			}
			if (addr)
				offset += (attlen > 0 ? attlen : VARSIZE_ANY(addr));
		}
	}
	else
	{
		tcache_column_store *tcs = (tcache_column_store *)rcstore[rel_index];

		if (row_index >= tcs->nrows)
			goto all_nulls;

		for (vtlist = &vrel->vtlist[vrel->vtsources[vtsrc_index]];
			 vtlist->vrelidx == rel_index;
			 vtlist = &vrel->vtlist[vrel->vtsources[vtsrc_index++]])
		{
			int		vattsrc = vtlist->vattsrc;
			int		vattdst = vtlist->vattdst;
			int		vattlen = vtlist->attlen;

			if (tcs->cdata[vattsrc].isnull &&
				att_isnull(row_index, tcs->cdata[vattsrc].isnull))
			{
				dest_values[vattdst] = NULL;
				*dest_has_null = true;
			}
			else
			{
				char   *cs_values = tcs->cdata[vattsrc].values;
				Assert(cs_values);

				if (vattlen > 0)
					addr = tcs->cdata[vattsrc].values + vattlen * row_index;
				else
				{
					cl_uint		vl_ofs = ((cl_uint *)cs_values)[row_index];

					addr = tcs->cdata[vattsrc].toast + vl_ofs;
				}
				dest_values[vattdst] = addr;
			}
		}
	}
	return vtsrc_index - first_vtsrc_index;

all_nulls:
	for (vtlist = &vrel->vtlist[vrel->vtsources[vtsrc_index]];
		 vtlist->vrelidx == rel_index;
		 vtlist = &vrel->vtlist[vrel->vtsources[vtsrc_index++]])
	{
		dest_values[vtlist->vattdst] = NULL;
		*dest_has_null = true;
	}
	return vtsrc_index - first_vtsrc_index;
}

static bool
do_materialize_column_store(pgstrom_vrelation *vrel, cl_uint row_index,
							int nrels, cl_uint *rindex, StromObject **rcstore)
{
	tcache_column_store *tcs = (tcache_column_store *) vrel->rcstore;






}






pgstrom_vrelation *
pgstrom_materialize_column_store(pgstrom_vrelation *vrel_tmpl,
								 StromObjet **rcstore)
{
	pgstrom_vrelation *vrel;
	tcache_column_store *tcs;
	cl_uint		nrels = vrel_tmpl->kern->nrels;
	cl_uint		nitems = vrel_tmpl->kern->nitems;
	cl_uint		ncols = vrel_tmpl->ncols;
	cl_uint		width;

	/*
     * TODO: if nrels == 1 and vrelation is compatible, no need to make
     * a projection. Just increment reference counter of source rcs.
     */

	/*
	 * allocation of vrelation with rindex[nitems] length
	 */
	length = (MAXALIGN(offsetof(pgstrom_vrelation, vtlist[ncols])) +
			  MAXALIGN(offsetof(kern_vrelation, rindex[nitems])));
	vrel = pgstrom_shmem_alloc(length);
	if (!vrel)
		return NULL;
	vrel->sobj.stag = StromTag_VirtRelation;
	SpinLockInit(&vrel->lock);
	vrel->refcnt = 1;
	vrel->ncols = ncols;
	vrel->rcstore = NULL;
	vrel->kern = (char *)vrel + MAXALIGN(offsetof(pgstrom_vrelation,
												  vtlist[ncols]));
	memcpy(vrel->vtlist, vrel_tmpl->vtlist,
		   offsetof(pgstrom_vrelation, vtlist[ncols]) -
		   offsetof(pgstrom_vrelation, vtlist[0]));
	vrel->kern->nrels = 1;  /* materialized to one row-store */
	vrel->kern->nitems = nitems;
	vrel->kern->nrooms = nitems;
	vrel->kern->errcode = vrel_tmpl->kern->errcode;

	/*
     * allocation of column-store of materialized relation
     */
	length = STROMALIGN(offsetof(tcache_column_store, cdata[ncols]));
	/* we cannot have system columns for materialized columns */
	for (i=0; i < vrel->ncols; i++)
	{
		if (!vrel->vtlist[i].attnotnull)
			length += STROMALIGN((nitems + BITS_PER_BYTE - 1) / BITS_PER_BYTE);
		length += STROMALIGN((vrel->vtlist[i].attlen > 0
							  ? vrel->vtlist[i].attlen
							  : sizeof(cl_uint)) * nitems);
	}

	tcs = pgstrom_shmem_alloc(length);
	if (!tcs)
	{
		pgstrom_shmem_free(vrel);
		return NULL;	/* out of shared memory */
	}
	offset = MAXALIGN(offsetof(tcache_column_store, cdata[ncols]));
	memset(tcs, 0, offset);
	tcs->sobj.stag = StromTag_TCacheColumnStore;
    SpinLockInit(&tcs->refcnt_lock);
    tcs->refcnt = 1;
    tcs->ncols = ncols;
	tcs->ctids = NULL;
	tcs->theads = NULL;

	for (i=0; i < ncols; i++)
	{
		if (!vrel->vtlist[i].attnotnull)
		{
			tcs->cdata[i].isnull = (uint8 *)((char *)tcs + offset);
			offset += STROMALIGN((nitems + BITS_PER_BYTE - 1) / BITS_PER_BYTE);
		}
		tcs->cdata[i].values = ((char *)tcs + offset);
		if (vrel->vtlist[i].attlen > 0)
		{
			offset += STROMALIGN(vrel->vtlist[i].attlen * nitems);
			tcs->cdata[i].toast = NULL;
		}
		else
		{
			offset += STROMALIGN(sizeof(cl_uint) * nitems);
			tcs->cdata[i].toast = pgstrom_create_toast_buffer(0);
			if (!tcs->cdata[i].toast)
			{
				pgstrom_put_column_store(tcs);
				pgstrom_shmem_free(vrel);
				return NULL;
			}
		}
	}
	Assert(offset == length);

	for (i=0; i < nitems; i++)
	{
		if (do_materialize_column_store())
			vrel->kern->rindex[i] = i + 1;
		else
			vrel->kern->rindex[i] = -(i + 1);
	}

	return vrel;
}

static bool
do_materialize_row_store(pgstrom_vrelation *vrel, cl_uint row_index,
						 int nrels, cl_uint *rindex, StromObject **rcstore)
{
	tcache_row_store *trs = (tcache_row_store *) vrel->rcstore;
	HeapTupleHeader td;
	Size		t_length;
	Size		t_hoff;
	void	  **dest_values = alloca(sizeof(void *) * vrel->ncols);
	bool		dest_has_null = false;

#ifdef USE_ASSERT_CHECKING
	/*
	 * NOTE: we assume vrel->vtsources is set up on creation time
	 * for fast column extraction from row-store
	 */
	for (i=1; i < vrel->ncols; i++)
	{
		vrelation_colmeta  *vtent1 = vrel->vtlist[vrel->vtsources[i-1]];
		vrelation_colmeta  *vtent2 = vrel->vtlist[vrel->vtsources[i]];

		Assert(vtent1->vrelidx < vtent2->vrelidx ||
			   (vtent1->vrelidx == vtent2->vrelidx &&
				vtent1->vattsrc == vtent2->vattsrc));
	}
#endif /* USE_ASSERT_CHECKING */

	/*
	 * extract source row/column stores
	 */
	memset(dest_values, 0, sizeof(void *) * vrel->ncols);
	for (i=0, j=0; i < nrels; i++)
	{
		j += extract_rcstore(vrel, j, i, rcstore, rindex[i],
							 dest_values, &dest_has_null);
	}
	Assert(j == vrel->ncols);

	/*
	 * compute length of tuple, and expand row-store if needed
	 */
	t_hoff = offsetof(HeapTupleHeaderData, t_bits);
	if (dest_has_null)
		t_hoff += BITMAPLEN(vrel->ncols);
	t_hoff = MAXALIGN(t_length);

	t_length = 0;
	for (i=0; i < vrel->ncols; i++)
	{
		if (dest_values[i])
		{
			int		attlen   = vrel->vtlist[i].attlen;
			int		attalign = vrel->vtlist[i].attalign;

			if (attlen > 0)
			{
				t_length = TYPEALIGN(attalign, t_length);
				t_length += attlen;
			}
			else if (VARATT_IS_1B(dest_values[i]))
				t_length += VARSIZE_ANY(dest_values[i]);
			else
			{
				t_length = TYPEALIGN(attalign, t_length);
				t_length += VARSIZE_ANY(dest_values[i]);
			}
		}
	}
	t_length += t_hoff;

	/* expand row-store, if not sufficient */
	if (trs->usage + offsetof(rs_tuple, data) + t_length > trs->kern.length)
	{
		tcache_row_store *trs_new;

		trs->kern.length += trs->kern.length;
		trs_new = pgstrom_shmem_realloc(trs, trs->kern.length);
		if (!trs_new)
		{
			vrel->rcstore = NULL;
			return false;
		}
		trs = trs_new;
	}

	/*
	 * OK, put a rs_tuple on the tuple store
	 */
	Assert(trs->usage +
		   offsetof(rs_tuple, data) +
		   t_length <= trs->kern.length);
	rs_tup = (rs_tuple *)((char *)&trs->kern + trs->usage);
	trs->usage += offsetof(rs_tuple, data) + t_length;

	memset(rs_tup, 0, sizeof(rs_tuple));
	rs_tup->htup.t_len = t_length;
	ItemPointerSetInvalid(&rs_tup->htup.t_self);
	rs_tup->htup.t_tableOid = InvalidOid;
	rs_tup->htup.t_data = &rs_tup->data;

	td = rs_tup->htup.t_data;
	HeapTupleHeaderSetDatumLength(td, );
	HeapTupleHeaderSetTypeId(td, RECORDOID);
	HeapTupleHeaderSetTypMod(td, -1);

	HeapTupleHeaderSetNatts(td, vrel->ncols);
	td->t_hoff = t_hoff;
	for (i=0; i < vrel->ncols; i++)
	{
		if (dest_values[i])
		{
			int		attlen   = vrel->vtlist[i].attlen;
			int		attalign = vrel->vtlist[i].attalign;

			if (dest_has_null)
				td->t_bits[i / BITS_PER_BYTE] |= (1 << (i % BITS_PER_BYTE));
			if (attlen > 0)
			{
				t_hoff = TYPEALIGN(attalign, t_hoff);
				memcpy((char *)td + t_hoff, dest_values[i], attlen);
				t_hoff += attlen;
			}
			else if (VARATT_IS_1B(dest_values[i]))
			{
				Size	vl_size = VARSIZE_ANY(dest_values[i]);

				memcpy((char *)td + t_hoff, dest_values[i], vl_size);
				t_hoff += vl_size;
			}
			else
			{
				Size	vl_size = VARSIZE_ANY(dest_values[i]);

				t_hoff = TYPEALIGN(attalign, t_hoff);
				memcpy((char *)td + t_hoff, dest_values[i], vl_size);
				t_hoff += vl_size;
			}
		}
		else
		{
			Assert(dest_has_null);
			td->t_bits[i / BITS_PER_BYTE] |= (1 << (i % BITS_PER_BYTE));
		}
	}
	return result;
}

pgstrom_vrelation *
pgstrom_materialize_row_store(pgstrom_vrelation *vrel_tmpl,
							  StromObjet **rcstore)
{
	pgstrom_vrelation *vrel;
	tcache_row_store *trs;
	cl_uint		nrels = vrel_tmpl->kern->nrels;
	cl_uint		nitems = vrel_tmpl->kern->nitems;
	cl_uint		ncols = vrel_tmpl->ncols;
	cl_uint		width;
	cl_uint	   *rindex;
	Size		length;
	Size		allocated;
	bool		has_rechecks;
	bool		all_visible;

	/*
	 * TODO: if nrels == 1 and vrelation is compatible, no need to make
	 * a projection. Just increment reference counter of source rcs.
	 */

	/*
	 * allocation of vrelation with rindex[nitems] length
	 */
	length = (MAXALIGN(offsetof(pgstrom_vrelation, vtlist[ncols])) +
			  MAXALIGN(offsetof(kern_vrelation, rindex[nitems])));
	vrel = pgstrom_shmem_alloc(length);
	if (!vrel)
		return NULL;
	vrel->sobj.stag = StromTag_VirtRelation;
	SpinLockInit(&vrel->lock);
	vrel->refcnt = 1;
	vrel->ncols = ncols;
	vrel->rcstore = NULL;	/* set below */
	vrel->kern = (char *)vrel + MAXALIGN(offsetof(pgstrom_vrelation,
												  vtlist[ncols]));
	memcpy(vrel->vtlist, vrel_tmpl->vtlist,
		   offsetof(pgstrom_vrelation, vtlist[ncols]) -
		   offsetof(pgstrom_vrelation, vtlist[0]));
	vrel->kern->nrels = 1;	/* materialized to one row-store */
	vrel->kern->nitems = nitems;
	vrel->kern->nrooms = nitems;
	vrel->kern->errcode = vrel_tmpl->kern->errcode;

	/*
	 * allocation of row-store according to the width and nitems
	 */
	width = offsetof(HeapTupleHeaderData, t_bits);
	width += BITMAPLEN(ncols);	/* nullmap */
	width = MAXALIGN(width);
	for (i=0; i < ncols; i++)
	{
		width = TYPEALIGN(vrel->vtlist[i].attalign, width);
		width += vrel->vtlist[i].vattwidth;
	}
	width = MAXALIGN(width);
	trs = pgstrom_shmem_alloc_alap(offsetof(tcache_row_store,
											kern.colmeta[ncols]) +
								   width * nitems,
								   &allocated);
	if (!trs)
	{
		pgstrom_shmem_free(vrel);
		return NULL;	/* out of shared memory */
	}

	trs->sobj.stag = StromTag_TCacheRowStore;
	SpinLockInit(&trs->refcnt_lock);
	trs->refcnt = 1;
	memset(&trs->chain, 0, sizeof(dlist_node));
	/* in case of materialization, we consume row-store from head to tail
	 * because we already know number of rows to be materialized, but don't
	 * know how much size is exactly needed.
	 */
	trs->usage = (offsetof(kern_row_store, colmeta[ncols]) +
				  sizeof(cl_uint) * nitems);
	trs->blkno_max = 0;
	trs->blkno_min = MaxBlockNumber;
	trs->kern.length = STROMALIGN_DOWN(allocated -
									   offsetof(tcache_row_store, kern));
	trs->kern.ncols = ncols;
	trs->kern.nrows = 0;
	for (i=0; i < ncols; i++)
	{
		trs->kern.colmeta[i].attnotnul = vrel->vtlist[i].attnotnull;
		trs->kern.colmeta[i].attalign = vrel->vtlist[i].attalign;
		trs->kern.colmeta[i].attlen = vrel->vtlist[i].attlen;
	}

	rindex = vrel_tmpl->kern->rindex;
	for (i=0; i < nitems; i++)
	{
		if (do_materialize_row_store(vrel, i, rcstore, rindex + i * nrels))
			vrel->kern->rindex[i] = i + 1;
		else
			vrel->kern->rindex[i] = -(i + 1);
	}
	vrel->rcstore = &trs->sobj;

	return vrel;
}
#endif
