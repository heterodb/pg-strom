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
	int			i, j, nkeys = 0;

	/* count-up number of referenced columns */
	for (i=0; i < tupdesc->natts; i++)
	{
		if (attrefs[i])
			nkeys++;
	}
	nkeys += nsyscols;

	length = STROMALIGN(offsetof(kern_column_store, colmeta[nkeys]));
	result = palloc0(VARHDRSZ + length);
	SET_VARSIZE(result, VARHDRSZ + length);
	kcs_head = (kern_column_store *) VARDATA(result);
	kcs_head->ncols = nkeys;
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
	Assert(j + nsyscols == nkeys);
	kcs_head->length = length;

	return result;
}

void
kparam_refresh_kcs_head(kern_column_store *kcs_head, cl_uint nrooms)
{
	Size	length;
	int		i;

	length = STROMALIGN(offsetof(kern_column_store,
								 colmeta[kcs_head->ncols]));
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

		if (OidIsValid(var->varcollid) || var->varlevelsup > 0)
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
