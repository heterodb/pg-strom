/*
 * misc.c
 *
 * miscellaneous and uncategorized routines but usefull for multiple subsystems
 * of PG-Strom.
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"

/* ----------------------------------------------------------------
 *
 * form/deform/copy pgstromPlanInfo
 *
 * ----------------------------------------------------------------
 */
static void
__form_codegen_kvar_defitem(codegen_kvar_defitem *kvdef,
							List **p__kv_privs,
							List **p__kv_exprs)
{
	List	   *__kv_privs = NIL;
	List	   *__kv_exprs = NIL;
	List	   *__sub_privs = NIL;
	List	   *__sub_exprs = NIL;
	ListCell   *lc;

	__kv_privs = lappend(__kv_privs, makeInteger(kvdef->kv_slot_id));
	__kv_privs = lappend(__kv_privs, makeInteger(kvdef->kv_depth));
	__kv_privs = lappend(__kv_privs, makeInteger(kvdef->kv_resno));
	__kv_privs = lappend(__kv_privs, makeInteger(kvdef->kv_maxref));
	__kv_privs = lappend(__kv_privs, makeInteger(kvdef->kv_offset));

	__kv_privs = lappend(__kv_privs, makeInteger(kvdef->kv_type_oid));
	__kv_privs = lappend(__kv_privs, makeInteger(kvdef->kv_type_code));
	__kv_privs = lappend(__kv_privs, makeBoolean(kvdef->kv_typbyval));
	__kv_privs = lappend(__kv_privs, makeInteger(kvdef->kv_typalign));
	__kv_privs = lappend(__kv_privs, makeInteger(kvdef->kv_typlen));
	__kv_privs = lappend(__kv_privs, makeInteger(kvdef->kv_xdatum_sizeof));
	__kv_privs = lappend(__kv_privs, makeInteger(kvdef->kv_kvec_sizeof));
	__kv_exprs = lappend(__kv_exprs, kvdef->kv_expr);
	foreach (lc, kvdef->kv_subfields)
	{
		codegen_kvar_defitem *__kvdef = lfirst(lc);
		List   *__privs = NIL;
		List   *__exprs = NIL;

		__form_codegen_kvar_defitem(__kvdef, &__privs, &__exprs);
		__sub_privs = lappend(__sub_privs, __privs);
		__sub_exprs = lappend(__sub_exprs, __exprs);
	}
	__kv_privs = lappend(__kv_privs, __sub_privs);
	__kv_exprs = lappend(__kv_exprs, __sub_exprs);

	*p__kv_privs = __kv_privs;
	*p__kv_exprs = __kv_exprs;
}

static codegen_kvar_defitem *
__deform_codegen_kvar_defitem(List *__kv_privs, List *__kv_exprs)
{
	codegen_kvar_defitem *kvdef = palloc0(sizeof(codegen_kvar_defitem));
	List	   *__sub_privs = NIL;
	List	   *__sub_exprs = NIL;
	ListCell   *lc1, *lc2;
	int			pindex = 0;
	int			eindex = 0;

	kvdef->kv_slot_id   = intVal(list_nth(__kv_privs, pindex++));
	kvdef->kv_depth     = intVal(list_nth(__kv_privs, pindex++));
	kvdef->kv_resno     = intVal(list_nth(__kv_privs, pindex++));
	kvdef->kv_maxref    = intVal(list_nth(__kv_privs, pindex++));
	kvdef->kv_offset    = intVal(list_nth(__kv_privs, pindex++));

	kvdef->kv_type_oid  = intVal(list_nth(__kv_privs, pindex++));
	kvdef->kv_type_code = intVal(list_nth(__kv_privs, pindex++));
	kvdef->kv_typbyval  = boolVal(list_nth(__kv_privs, pindex++));
	kvdef->kv_typalign  = intVal(list_nth(__kv_privs, pindex++));
	kvdef->kv_typlen    = intVal(list_nth(__kv_privs, pindex++));
	kvdef->kv_xdatum_sizeof = intVal(list_nth(__kv_privs, pindex++));
	kvdef->kv_kvec_sizeof = intVal(list_nth(__kv_privs, pindex++));
	kvdef->kv_expr      = list_nth(__kv_exprs, eindex++);
	__sub_privs         = list_nth(__kv_privs, pindex++);
	__sub_exprs         = list_nth(__kv_exprs, eindex++);
	forboth (lc1, __sub_privs,
			 lc2, __sub_exprs)
	{
		codegen_kvar_defitem *__kvdef;

		__kvdef = __deform_codegen_kvar_defitem(lfirst(lc1),
												lfirst(lc2));
		kvdef->kv_subfields = lappend(kvdef->kv_subfields, __kvdef);
	}
	return kvdef;
}

void
form_pgstrom_plan_info(CustomScan *cscan, pgstromPlanInfo *pp_info)
{
	List	   *privs = NIL;
	List	   *exprs = NIL;
	List	   *kvars_deflist_privs = NIL;
	List	   *kvars_deflist_exprs = NIL;
	ListCell   *lc;
	int			endpoint_id;

	privs = lappend(privs, makeInteger(pp_info->xpu_task_flags));
	privs = lappend(privs, makeInteger(pp_info->gpu_cache_dindex));
	privs = lappend(privs, bms_to_pglist(pp_info->gpu_direct_devs));
	endpoint_id = DpuStorageEntryGetEndpointId(pp_info->ds_entry);
	privs = lappend(privs, makeInteger(endpoint_id));
	/* plan information */
	privs = lappend(privs, bms_to_pglist(pp_info->outer_refs));
	exprs = lappend(exprs, pp_info->used_params);
	privs = lappend(privs, pp_info->host_quals);
	privs = lappend(privs, makeInteger(pp_info->scan_relid));
	privs = lappend(privs, pp_info->scan_quals_fallback);
	exprs = lappend(exprs, pp_info->scan_quals_explain);
	privs = lappend(privs, __makeFloat(pp_info->scan_tuples));
	privs = lappend(privs, __makeFloat(pp_info->scan_nrows));
	privs = lappend(privs, makeInteger(pp_info->parallel_nworkers));
	privs = lappend(privs, __makeFloat(pp_info->parallel_divisor));
	privs = lappend(privs, __makeFloat(pp_info->startup_cost));
	privs = lappend(privs, __makeFloat(pp_info->inner_cost));
	privs = lappend(privs, __makeFloat(pp_info->run_cost));
	privs = lappend(privs, __makeFloat(pp_info->final_cost));
	/* bin-index support */
	privs = lappend(privs, makeInteger(pp_info->brin_index_oid));
	privs = lappend(privs, pp_info->brin_index_conds);
	privs = lappend(privs, pp_info->brin_index_quals);
	/* XPU code */
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_load_vars_packed));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_move_vars_packed));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_scan_quals));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_join_quals_packed));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_hash_keys_packed));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_gist_evals_packed));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_projection));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_groupby_keyhash));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_groupby_keyload));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_groupby_keycomp));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_groupby_actions));
	/* Kvars definitions */
	foreach (lc, pp_info->kvars_deflist)
	{
		codegen_kvar_defitem *kvdef = lfirst(lc);
		List   *__kv_privs = NIL;
		List   *__kv_exprs = NIL;

		__form_codegen_kvar_defitem(kvdef, &__kv_privs, &__kv_exprs);
		kvars_deflist_privs = lappend(kvars_deflist_privs, __kv_privs);
		kvars_deflist_exprs = lappend(kvars_deflist_exprs, __kv_exprs);
	}
	/* other planner fields */
	privs = lappend(privs, kvars_deflist_privs);
	exprs = lappend(exprs, kvars_deflist_exprs);
	privs = lappend(privs, makeInteger(pp_info->kvecs_bufsz));
	privs = lappend(privs, makeInteger(pp_info->kvecs_ndims));
	privs = lappend(privs, makeInteger(pp_info->extra_bufsz));
	privs = lappend(privs, makeInteger(pp_info->cuda_stack_size));
	privs = lappend(privs, pp_info->fallback_tlist);
	privs = lappend(privs, pp_info->groupby_actions);
	privs = lappend(privs, makeInteger(pp_info->groupby_prepfn_bufsz));
	/* inner relations */
	privs = lappend(privs, makeInteger(pp_info->sibling_param_id));
	privs = lappend(privs, makeInteger(pp_info->num_rels));
	for (int i=0; i < pp_info->num_rels; i++)
	{
		pgstromPlanInnerInfo *pp_inner = &pp_info->inners[i];
		List   *__exprs = NIL;
		List   *__privs = NIL;

		__privs = lappend(__privs, makeInteger(pp_inner->join_type));
		__privs = lappend(__privs, __makeFloat(pp_inner->join_nrows));
		__exprs = lappend(__exprs, pp_inner->hash_outer_keys_original);
		__privs = lappend(__privs, pp_inner->hash_outer_keys_fallback);
		__exprs = lappend(__exprs, pp_inner->hash_inner_keys_original);
		__privs = lappend(__privs, pp_inner->hash_inner_keys_fallback);
		__exprs = lappend(__exprs, pp_inner->join_quals_original);
		__privs = lappend(__privs, pp_inner->join_quals_fallback);
		__exprs = lappend(__exprs, pp_inner->other_quals_original);
		__privs = lappend(__privs, pp_inner->other_quals_fallback);
		__privs = lappend(__privs, makeInteger(pp_inner->gist_index_oid));
		__privs = lappend(__privs, makeInteger(pp_inner->gist_index_col));
		__privs = lappend(__privs, makeInteger(pp_inner->gist_ctid_resno));
		__privs = lappend(__privs, makeInteger(pp_inner->gist_func_oid));
		__privs = lappend(__privs, makeInteger(pp_inner->gist_slot_id));
		__privs = lappend(__privs, pp_inner->gist_clause);
		__privs = lappend(__privs, __makeFloat(pp_inner->gist_selectivity));
		__privs = lappend(__privs, __makeFloat(pp_inner->gist_npages));
		__privs = lappend(__privs, makeInteger(pp_inner->gist_height));

		exprs = lappend(exprs, __exprs);
		privs = lappend(privs, __privs);
	}
	cscan->custom_exprs = exprs;
	cscan->custom_private = privs;
}

/*
 * deform_pgstrom_plan_info
 */
pgstromPlanInfo *
deform_pgstrom_plan_info(CustomScan *cscan)
{
	pgstromPlanInfo *pp_info;
	pgstromPlanInfo	pp_data;
	List	   *privs = cscan->custom_private;
	List	   *exprs = cscan->custom_exprs;
	int			pindex = 0;
	int			eindex = 0;
	List	   *kvars_deflist_privs;
	List	   *kvars_deflist_exprs;
	ListCell   *lc1, *lc2;
	int			endpoint_id;

	memset(&pp_data, 0, sizeof(pgstromPlanInfo));
	/* device identifiers */
	pp_data.xpu_task_flags = intVal(list_nth(privs, pindex++));
	pp_data.gpu_cache_dindex = intVal(list_nth(privs, pindex++));
	pp_data.gpu_direct_devs = bms_from_pglist(list_nth(privs, pindex++));
	endpoint_id = intVal(list_nth(privs, pindex++));
	pp_data.ds_entry = DpuStorageEntryByEndpointId(endpoint_id);
	/* plan information */
	pp_data.outer_refs   = bms_from_pglist(list_nth(privs, pindex++));
	pp_data.used_params  = list_nth(exprs, eindex++);
	pp_data.host_quals   = list_nth(privs, pindex++);
	pp_data.scan_relid   = intVal(list_nth(privs, pindex++));
	pp_data.scan_quals_fallback = list_nth(privs, pindex++);
	pp_data.scan_quals_explain = list_nth(exprs, eindex++);
	pp_data.scan_tuples  = floatVal(list_nth(privs, pindex++));
	pp_data.scan_nrows   = floatVal(list_nth(privs, pindex++));
	pp_data.parallel_nworkers = intVal(list_nth(privs, pindex++));
	pp_data.parallel_divisor = floatVal(list_nth(privs, pindex++));
	pp_data.startup_cost = floatVal(list_nth(privs, pindex++));
	pp_data.inner_cost   = floatVal(list_nth(privs, pindex++));
	pp_data.run_cost     = floatVal(list_nth(privs, pindex++));
	pp_data.final_cost   = floatVal(list_nth(privs, pindex++));
	/* brin-index support */
	pp_data.brin_index_oid = intVal(list_nth(privs, pindex++));
	pp_data.brin_index_conds = list_nth(privs, pindex++);
	pp_data.brin_index_quals = list_nth(privs, pindex++);
	/* XPU code */
	pp_data.kexp_load_vars_packed  = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_move_vars_packed  = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_scan_quals        = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_join_quals_packed = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_hash_keys_packed  = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_gist_evals_packed = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_projection        = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_groupby_keyhash   = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_groupby_keyload   = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_groupby_keycomp   = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_groupby_actions   = __getByteaConst(list_nth(privs, pindex++));
	/* Kvars definitions */
	kvars_deflist_privs = list_nth(privs, pindex++);
	kvars_deflist_exprs = list_nth(exprs, eindex++);
	Assert(list_length(kvars_deflist_privs) == list_length(kvars_deflist_exprs));
	forboth (lc1, kvars_deflist_privs,
			 lc2, kvars_deflist_exprs)
	{
		codegen_kvar_defitem *kvdef;

		kvdef = __deform_codegen_kvar_defitem(lfirst(lc1),
											  lfirst(lc2));
		pp_data.kvars_deflist = lappend(pp_data.kvars_deflist, kvdef);
	}
	pp_data.kvecs_bufsz = intVal(list_nth(privs, pindex++));
	pp_data.kvecs_ndims = intVal(list_nth(privs, pindex++));
	pp_data.extra_bufsz = intVal(list_nth(privs, pindex++));
	pp_data.cuda_stack_size = intVal(list_nth(privs, pindex++));
	pp_data.fallback_tlist = list_nth(privs, pindex++);
	pp_data.groupby_actions = list_nth(privs, pindex++);
	pp_data.groupby_prepfn_bufsz  = intVal(list_nth(privs, pindex++));
	/* inner relations */
	pp_data.sibling_param_id = intVal(list_nth(privs, pindex++));
	pp_data.num_rels = intVal(list_nth(privs, pindex++));
	pp_info = palloc0(offsetof(pgstromPlanInfo, inners[pp_data.num_rels]));
	memcpy(pp_info, &pp_data, offsetof(pgstromPlanInfo, inners));
	for (int i=0; i < pp_info->num_rels; i++)
	{
		pgstromPlanInnerInfo *pp_inner = &pp_info->inners[i];
		List   *__exprs = list_nth(exprs, eindex++);
		List   *__privs = list_nth(privs, pindex++);
		int		__eindex = 0;
		int		__pindex = 0;

		pp_inner->join_type       = intVal(list_nth(__privs, __pindex++));
		pp_inner->join_nrows      = floatVal(list_nth(__privs, __pindex++));
		pp_inner->hash_outer_keys_original = list_nth(__exprs, __eindex++);
		pp_inner->hash_outer_keys_fallback = list_nth(__privs, __pindex++);
		pp_inner->hash_inner_keys_original = list_nth(__exprs, __eindex++);
		pp_inner->hash_inner_keys_fallback = list_nth(__privs, __pindex++);
		pp_inner->join_quals_original = list_nth(__exprs, __eindex++);
		pp_inner->join_quals_fallback = list_nth(__privs, __pindex++);
		pp_inner->other_quals_original = list_nth(__exprs, __eindex++);
		pp_inner->other_quals_fallback = list_nth(__privs, __pindex++);
		pp_inner->gist_index_oid  = intVal(list_nth(__privs, __pindex++));
		pp_inner->gist_index_col  = intVal(list_nth(__privs, __pindex++));
		pp_inner->gist_ctid_resno = intVal(list_nth(__privs, __pindex++));
		pp_inner->gist_func_oid   = intVal(list_nth(__privs, __pindex++));
		pp_inner->gist_slot_id    = intVal(list_nth(__privs, __pindex++));
		pp_inner->gist_clause     = list_nth(__privs, __pindex++);
		pp_inner->gist_selectivity = floatVal(list_nth(__privs, __pindex++));
		pp_inner->gist_npages     = floatVal(list_nth(__privs, __pindex++));
		pp_inner->gist_height     = intVal(list_nth(__privs, __pindex++));
	}
	return pp_info;
}

/*
 * copy_pgstrom_plan_info
 */
pgstromPlanInfo *
copy_pgstrom_plan_info(const pgstromPlanInfo *pp_orig)
{
	pgstromPlanInfo *pp_dest;
	List	   *kvars_deflist = NIL;
	ListCell   *lc;

	/*
	 * NOTE: we add one pgstromPlanInnerInfo margin to be used for GpuJoin.
	 */
	pp_dest = palloc0(offsetof(pgstromPlanInfo, inners[pp_orig->num_rels+1]));
	memcpy(pp_dest, pp_orig, offsetof(pgstromPlanInfo,
									  inners[pp_orig->num_rels]));
	pp_dest->used_params      = list_copy(pp_dest->used_params);
	pp_dest->host_quals       = copyObject(pp_dest->host_quals);
	pp_dest->scan_quals_fallback = copyObject(pp_dest->scan_quals_fallback);
	pp_dest->scan_quals_explain = copyObject(pp_dest->scan_quals_explain);
	pp_dest->brin_index_conds = copyObject(pp_dest->brin_index_conds);
	pp_dest->brin_index_quals = copyObject(pp_dest->brin_index_quals);
	foreach (lc, pp_orig->kvars_deflist)
	{
		codegen_kvar_defitem *kvdef_orig = lfirst(lc);
		codegen_kvar_defitem *kvdef_dest;

		kvdef_dest = pmemdup(kvdef_orig, sizeof(codegen_kvar_defitem));
		kvdef_dest->kv_expr = copyObject(kvdef_dest->kv_expr);
		kvars_deflist = lappend(kvars_deflist, kvdef_dest);
	}
	pp_dest->kvars_deflist    = kvars_deflist;
	pp_dest->fallback_tlist   = copyObject(pp_dest->fallback_tlist);
	pp_dest->groupby_actions  = list_copy(pp_dest->groupby_actions);
	for (int j=0; j < pp_orig->num_rels; j++)
	{
		pgstromPlanInnerInfo *pp_inner = &pp_dest->inners[j];
#define __COPY(FIELD)	pp_inner->FIELD = copyObject(pp_inner->FIELD)
		__COPY(hash_outer_keys_original);
		__COPY(hash_outer_keys_fallback);
		__COPY(hash_inner_keys_original);
		__COPY(hash_inner_keys_fallback);
		__COPY(join_quals_original);
		__COPY(join_quals_fallback);
		__COPY(other_quals_original);
		__COPY(other_quals_fallback);
		__COPY(gist_clause);
#undef __COPY
	}
	return pp_dest;
}

/*
 * fixup_expression_by_partition_leaf
 */
List *
fixup_expression_by_partition_leaf(PlannerInfo *root,
								   Relids leaf_relids,
								   List *clauses)
{
	AppendRelInfo **appinfos;
	Relids		next_relids = NULL;
	int			i, nitems = 0;

	if (!root->append_rel_array)
		return clauses;		/* shortcut */

	appinfos = alloca(sizeof(AppendRelInfo *) * root->simple_rel_array_size);
	for (i = bms_next_member(leaf_relids, -1);
		 i >= 0;
		 i = bms_next_member(leaf_relids, i))
	{
		AppendRelInfo *appinfo = root->append_rel_array[i];

		if (appinfo)
		{
			Assert(appinfo->child_relid == i);
			appinfos[nitems++] = appinfo;
			next_relids = bms_add_member(next_relids, appinfo->parent_relid);
		}
	}

	if (nitems > 0)
	{
		clauses = fixup_expression_by_partition_leaf(root,
													 next_relids,
													 clauses);
		clauses = (List *)adjust_appendrel_attrs(root,
												 (Node *)clauses,
												 nitems,
												 appinfos);
	}
	return clauses;
}

/*
 * fixup_relids_by_partition_leaf
 */
Relids
fixup_relids_by_partition_leaf(PlannerInfo *root,
							   const Relids leaf_relids,
							   const Relids parent_relids)
{
	Relids	results = NULL;
	int		curr;

	for (curr = bms_next_member(leaf_relids, -1);
		 curr >= 0;
		 curr = bms_next_member(leaf_relids, curr))
	{
		int		relid = curr;
	again:
		for (int k=0; k < root->simple_rel_array_size; k++)
		{
			AppendRelInfo *ap_info = root->append_rel_array[k];

			if (ap_info && ap_info->child_relid == relid)
			{
				if (ap_info->parent_relid == relid)
					break;
				relid = ap_info->parent_relid;
				goto again;
			}
		}
		results = bms_add_member(results, relid);
	}
	return results;
}

/*
 * append a binary chunk at the aligned block
 */
int
__appendBinaryStringInfo(StringInfo buf, const void *data, int datalen)
{
	static uint64_t __zero = 0;
	int		padding = (MAXALIGN(buf->len) - buf->len);
	int		pos;

	if (padding > 0)
		appendBinaryStringInfo(buf, (char *)&__zero, padding);
	pos = buf->len;
	appendBinaryStringInfo(buf, data, datalen);
	return pos;
}

int
__appendZeroStringInfo(StringInfo buf, int nbytes)
{
	static uint64_t __zero = 0;
	int		padding = (MAXALIGN(buf->len) - buf->len);
	int		pos;

	if (padding > 0)
		appendBinaryStringInfo(buf, (char *)&__zero, padding);
	pos = buf->len;
	if (nbytes > 0)
	{
		enlargeStringInfo(buf, nbytes);
		memset(buf->data + pos, 0, nbytes);
		buf->len += nbytes;
	}
	return pos;
}

/*
 * get_type_name
 */
char *
get_type_name(Oid type_oid, bool missing_ok)
{
	HeapTuple	tup;
	char	   *retval;

	tup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
	if (!HeapTupleIsValid(tup))
	{
		if (!missing_ok)
			elog(ERROR, "cache lookup failed for type %u", type_oid);
		return NULL;
	}
	retval = pstrdup(NameStr(((Form_pg_type) GETSTRUCT(tup))->typname));
	ReleaseSysCache(tup);

	return retval;
}

/*
 * get_type_namespace
 */
Oid
get_type_namespace(Oid type_oid)
{
	HeapTuple	tup;
	Oid			namespace_oid = InvalidOid;

	tup = SearchSysCache1(TYPEOID, ObjectIdGetDatum(type_oid));
	if (HeapTupleIsValid(tup))
	{
		namespace_oid = ((Form_pg_type) GETSTRUCT(tup))->typnamespace;
		ReleaseSysCache(tup);
	}
	return namespace_oid;
}

/*
 * get_type_extension_name
 */
char *
get_type_extension_name(Oid type_oid)
{
	Oid		ext_oid = getExtensionOfObject(TypeRelationId, type_oid);

	if (OidIsValid(ext_oid))
		return get_extension_name(ext_oid);
	return NULL;
}

/*
 * get_func_extension_name
 */
char *
get_func_extension_name(Oid func_oid)
{
	Oid		ext_oid = getExtensionOfObject(ProcedureRelationId, func_oid);

	if (OidIsValid(ext_oid))
		return get_extension_name(ext_oid);
	return NULL;
}

/*
 * get_relation_am
 */
Oid
get_relation_am(Oid rel_oid, bool missing_ok)
{
	HeapTuple	tup;
	Oid			relam;

	tup = SearchSysCache1(RELOID, ObjectIdGetDatum(rel_oid));
	if (!HeapTupleIsValid(tup))
	{
		if (!missing_ok)
			elog(ERROR, "cache lookup failed for relation %u", rel_oid);
		return InvalidOid;
	}
	relam = ((Form_pg_class) GETSTRUCT(tup))->relam;
	ReleaseSysCache(tup);

	return relam;
}

/*
 * Bitmapset <-> numeric List transition
 */
List *
bms_to_pglist(const Bitmapset *bms)
{
	List   *pglist = NIL;
	int		k;

	for (k = bms_next_member(bms, -1);
		 k >= 0;
		 k = bms_next_member(bms, k))
	{
		pglist = lappend_int(pglist, k);
	}
	return pglist;
}

Bitmapset *
bms_from_pglist(List *pglist)
{
	Bitmapset  *bms = NULL;
	ListCell   *lc;

	foreach (lc, pglist)
	{
		bms = bms_add_member(bms, lfirst_int(lc));
	}
	return bms;
}

Float *
__makeFloat(double fval)
{
	return makeFloat(psprintf("%e", fval));
}

Const *
__makeByteaConst(bytea *data)
{
	return makeConst(BYTEAOID,
					 -1,
					 InvalidOid,
					 -1,
					 PointerGetDatum(data),
					 data == NULL,
					 false);
}

bytea *
__getByteaConst(Const *con)
{
	Assert(IsA(con, Const) && con->consttype == BYTEAOID);

	return (con->constisnull ? NULL : DatumGetByteaP(con->constvalue));
}

/*
 * pgstrom_copy_pathnode
 *
 * add_path() / add_partial_path() may reject path-nodes that are already
 * registered and referenced by upper path nodes, like GpuJoin.
 * To avoid the problem, we use copy of path-nodes that are (potentially)
 * released by another ones. However, it is not a full-copy. add_path() will
 * never release fields of individual path-nodes, so this function tries
 * to make a copy of path-node itself and child path-nodes only.
 */
static Path *
__pgstrom_copy_joinpath(const JoinPath *a, size_t sz)
{
	JoinPath   *b = pmemdup(a, sz);

	b->outerjoinpath = pgstrom_copy_pathnode(a->outerjoinpath);
	b->innerjoinpath = pgstrom_copy_pathnode(a->innerjoinpath);

	return &b->path;
}

Path *
pgstrom_copy_pathnode(const Path *pathnode)
{
	if (!pathnode)
		return NULL;

	switch (nodeTag(pathnode))
	{
		case T_Path:
			return pmemdup(pathnode, sizeof(Path));
		case T_IndexPath:
			return pmemdup(pathnode, sizeof(IndexPath));
		case T_BitmapHeapPath:
			return pmemdup(pathnode, sizeof(BitmapHeapPath));
		case T_BitmapAndPath:
			return pmemdup(pathnode, sizeof(BitmapAndPath));
		case T_BitmapOrPath:
			return pmemdup(pathnode, sizeof(BitmapOrPath));
		case T_TidPath:
			return pmemdup(pathnode, sizeof(TidPath));
		case T_TidRangePath:
			return pmemdup(pathnode, sizeof(TidRangePath));
		case T_SubqueryScanPath:
			{
				SubqueryScanPath *a = (SubqueryScanPath *)pathnode;
				SubqueryScanPath *b = pmemdup(a, sizeof(SubqueryScanPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_ForeignPath:
			{
				ForeignPath	   *a = (ForeignPath *)pathnode;
				ForeignPath	   *b = pmemdup(a, sizeof(ForeignPath));
				b->fdw_outerpath = pgstrom_copy_pathnode(a->fdw_outerpath);
				return &b->path;
			}
		case T_CustomPath:
			{
				CustomPath	   *a = (CustomPath *)pathnode;
				CustomPath	   *b = pmemdup(a, sizeof(CustomPath));
				List		   *subpaths = NIL;
				ListCell	   *lc;

				foreach (lc, a->custom_paths)
				{
					Path	   *sp = pgstrom_copy_pathnode(lfirst(lc));
					subpaths = lappend(subpaths, sp);
				}
				b->custom_paths = subpaths;
				return &b->path;
			}
		case T_NestPath:
			return __pgstrom_copy_joinpath((JoinPath *)pathnode, sizeof(NestPath));
		case T_MergePath:
			return __pgstrom_copy_joinpath((JoinPath *)pathnode, sizeof(MergePath));
		case T_HashPath:
			return __pgstrom_copy_joinpath((JoinPath *)pathnode, sizeof(HashPath));
		case T_AppendPath:
			{
				AppendPath *a = (AppendPath *)pathnode;
				AppendPath *b = pmemdup(a, sizeof(AppendPath));
				List	   *subpaths = NIL;
				ListCell   *lc;

				foreach (lc, a->subpaths)
					subpaths = lappend(subpaths,
									   pgstrom_copy_pathnode(lfirst(lc)));
				b->subpaths = subpaths;
				return &b->path;
			}
		case T_MergeAppendPath:
			{
				MergeAppendPath *a = (MergeAppendPath *)pathnode;
				MergeAppendPath *b = pmemdup(a, sizeof(MergeAppendPath));
				List	   *subpaths = NIL;
				ListCell   *lc;

				foreach (lc, a->subpaths)
					subpaths = lappend(subpaths,
									   pgstrom_copy_pathnode(lfirst(lc)));
				b->subpaths = subpaths;
				return &b->path;
			}
		case T_GroupResultPath:
			return pmemdup(pathnode, sizeof(GroupResultPath));
		case T_MaterialPath:
			{
				MaterialPath   *a = (MaterialPath *)pathnode;
				MaterialPath   *b = pmemdup(a, sizeof(MaterialPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_MemoizePath:
			{
				MemoizePath	   *a = (MemoizePath *)pathnode;
				MemoizePath	   *b = pmemdup(a, sizeof(MemoizePath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_UniquePath:
			{
				UniquePath	   *a = (UniquePath *)pathnode;
				UniquePath	   *b = pmemdup(a, sizeof(UniquePath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_GatherPath:
			{
				GatherPath	   *a = (GatherPath *)pathnode;
				GatherPath	   *b = pmemdup(a, sizeof(GatherPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_GatherMergePath:
			{
				GatherMergePath *a = (GatherMergePath *)pathnode;
				GatherMergePath *b = pmemdup(a, sizeof(GatherMergePath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_ProjectionPath:
			{
				ProjectionPath *a = (ProjectionPath *)pathnode;
				ProjectionPath *b = pmemdup(a, sizeof(ProjectionPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_ProjectSetPath:
			{
				ProjectSetPath *a = (ProjectSetPath *)pathnode;
				ProjectSetPath *b = pmemdup(a, sizeof(ProjectSetPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_SortPath:
			{
				SortPath	   *a = (SortPath *)pathnode;
				SortPath	   *b = pmemdup(a, sizeof(SortPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_IncrementalSortPath:
			{
				IncrementalSortPath *a = (IncrementalSortPath *)pathnode;
				IncrementalSortPath *b = pmemdup(a, sizeof(IncrementalSortPath));
				b->spath.subpath = pgstrom_copy_pathnode(a->spath.subpath);
				return &b->spath.path;
			}
		case T_GroupPath:
			{
				GroupPath	   *a = (GroupPath *)pathnode;
				GroupPath	   *b = pmemdup(a, sizeof(GroupPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_UpperUniquePath:
			{
				UpperUniquePath *a = (UpperUniquePath *)pathnode;
				UpperUniquePath *b = pmemdup(a, sizeof(UpperUniquePath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_AggPath:
			{
				AggPath		   *a = (AggPath *)pathnode;
				AggPath		   *b = pmemdup(a, sizeof(AggPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_GroupingSetsPath:
			{
				GroupingSetsPath *a = (GroupingSetsPath *)pathnode;
				GroupingSetsPath *b = pmemdup(a, sizeof(GroupingSetsPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
                return &b->path;
			}
		case T_MinMaxAggPath:
			return pmemdup(pathnode, sizeof(MinMaxAggPath));
		case T_WindowAggPath:
			{
				WindowAggPath  *a = (WindowAggPath *)pathnode;
				WindowAggPath  *b = pmemdup(a, sizeof(WindowAggPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_SetOpPath:
			{
				SetOpPath	   *a = (SetOpPath *)pathnode;
				SetOpPath	   *b = pmemdup(a, sizeof(SetOpPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_RecursiveUnionPath:
			{
				RecursiveUnionPath *a = (RecursiveUnionPath *)pathnode;
				RecursiveUnionPath *b = pmemdup(a, sizeof(RecursiveUnionPath));
				b->leftpath = pgstrom_copy_pathnode(a->leftpath);
				b->rightpath = pgstrom_copy_pathnode(a->rightpath);
				return &b->path;
			}
		case T_LockRowsPath:
			{
				LockRowsPath   *a = (LockRowsPath *)pathnode;
				LockRowsPath   *b = pmemdup(a, sizeof(LockRowsPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_ModifyTablePath:
			{
				ModifyTablePath *a = (ModifyTablePath *)pathnode;
				ModifyTablePath *b = pmemdup(a, sizeof(ModifyTablePath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		case T_LimitPath:
			{
				LimitPath  *a = (LimitPath *)pathnode;
				LimitPath  *b = pmemdup(a, sizeof(LimitPath));
				b->subpath = pgstrom_copy_pathnode(a->subpath);
				return &b->path;
			}
		default:
			elog(ERROR, "Bug? unknown path-node: %s", nodeToString(pathnode));
	}
	return NULL;
}

/*
 * ----------------------------------------------------------------
 *
 * SQL functions to support regression test
 *
 * ----------------------------------------------------------------
 */
static unsigned int		pgstrom_random_seed = 0;
static bool				pgstrom_random_seed_set = false;

PG_FUNCTION_INFO_V1(pgstrom_random_setseed);
PUBLIC_FUNCTION(Datum)
pgstrom_random_setseed(PG_FUNCTION_ARGS)
{
	unsigned int	seed = PG_GETARG_UINT32(0);

	pgstrom_random_seed = seed ^ 0xdeadbeafU;
	pgstrom_random_seed_set = true;

	PG_RETURN_VOID();
}

static int64_t
__random(void)
{
	if (!pgstrom_random_seed_set)
	{
		pgstrom_random_seed = (unsigned int)MyProcPid ^ 0xdeadbeafU;
		pgstrom_random_seed_set = true;
	}
	return (uint64_t)rand_r(&pgstrom_random_seed);
}

static inline double
__drand48(void)
{
	return (double)__random() / (double)RAND_MAX;
}

static inline bool
generate_null(double ratio)
{
	if (ratio <= 0.0)
		return false;
	if (100.0 * __drand48() < ratio)
		return true;
	return false;
}

PG_FUNCTION_INFO_V1(pgstrom_random_int);
PUBLIC_FUNCTION(Datum)
pgstrom_random_int(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	int64		lower = (!PG_ARGISNULL(1) ? PG_GETARG_INT64(1) : 0);
	int64		upper = (!PG_ARGISNULL(2) ? PG_GETARG_INT64(2) : INT_MAX);
	uint64_t	v;

	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		PG_RETURN_INT64(lower);
	v = (__random() << 31) | __random();

	PG_RETURN_INT64(lower + v % (upper - lower));
}

PG_FUNCTION_INFO_V1(pgstrom_random_float);
PUBLIC_FUNCTION(Datum)
pgstrom_random_float(PG_FUNCTION_ARGS)
{
	float8	ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	float8	lower = (!PG_ARGISNULL(1) ? PG_GETARG_FLOAT8(1) : 0.0);
	float8	upper = (!PG_ARGISNULL(2) ? PG_GETARG_FLOAT8(2) : 1.0);

	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		PG_RETURN_FLOAT8(lower);

	PG_RETURN_FLOAT8((upper - lower) * __drand48() + lower);
}

PG_FUNCTION_INFO_V1(pgstrom_random_date);
PUBLIC_FUNCTION(Datum)
pgstrom_random_date(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	DateADT		lower;
	DateADT		upper;
	uint64_t	v;

	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_DATEADT(1);
	else
		lower = date2j(2015, 1, 1) - POSTGRES_EPOCH_JDATE;
	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_DATEADT(2);
	else
		upper = date2j(2025, 12, 31) - POSTGRES_EPOCH_JDATE;

	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		PG_RETURN_DATEADT(lower);
	v = (__random() << 31) | __random();

	PG_RETURN_DATEADT(lower + v % (upper - lower));
}

PG_FUNCTION_INFO_V1(pgstrom_random_time);
PUBLIC_FUNCTION(Datum)
pgstrom_random_time(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	TimeADT		lower = 0;
	TimeADT		upper = HOURS_PER_DAY * USECS_PER_HOUR - 1;
	uint64_t	v;

	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_TIMEADT(1);
	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_TIMEADT(2);
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		PG_RETURN_TIMEADT(lower);
	v = (__random() << 31) | __random();

	PG_RETURN_TIMEADT(lower + v % (upper - lower));
}

PG_FUNCTION_INFO_V1(pgstrom_random_timetz);
PUBLIC_FUNCTION(Datum)
pgstrom_random_timetz(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	TimeADT		lower = 0;
	TimeADT		upper = HOURS_PER_DAY * USECS_PER_HOUR - 1;
	TimeTzADT  *temp;
	uint64_t	v;

	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_TIMEADT(1);
	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_TIMEADT(2);
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	temp = palloc(sizeof(TimeTzADT));
	temp->zone = (__random() % 23 - 11) * USECS_PER_HOUR;
	if (upper == lower)
		temp->time = lower;
	else
	{
		v = (__random() << 31) | __random();
		temp->time = lower + v % (upper - lower);
	}
	PG_RETURN_TIMETZADT_P(temp);
}

PG_FUNCTION_INFO_V1(pgstrom_random_timestamp);
PUBLIC_FUNCTION(Datum)
pgstrom_random_timestamp(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	Timestamp	lower;
	Timestamp	upper;
	uint64_t	v;
	struct pg_tm tm;

	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_TIMESTAMP(1);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 45;	/* '2015-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &lower) != 0)
			elog(ERROR, "timestamp out of range");
	}

	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_TIMESTAMP(2);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 55;	/* '2025-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &upper) != 0)
			elog(ERROR, "timestamp out of range");
	}
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		PG_RETURN_TIMEADT(lower);
	v = (__random() << 31) | __random();

	PG_RETURN_TIMESTAMP(lower + v % (upper - lower));
}

PG_FUNCTION_INFO_V1(pgstrom_random_macaddr);
PUBLIC_FUNCTION(Datum)
pgstrom_random_macaddr(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	macaddr	   *temp;
	uint64_t	lower;
	uint64_t	upper;
	uint64_t	v, x;

	if (PG_ARGISNULL(1))
		lower = 0xabcd00000000UL;
	else
	{
		temp = PG_GETARG_MACADDR_P(1);
		lower = (((uint64_t)temp->a << 40) | ((uint64_t)temp->b << 32) |
				 ((uint64_t)temp->c << 24) | ((uint64_t)temp->d << 16) |
				 ((uint64_t)temp->e <<  8) | ((uint64_t)temp->f));
	}

	if (PG_ARGISNULL(2))
		upper = 0xabcdffffffffUL;
	else
	{
		temp = PG_GETARG_MACADDR_P(2);
		upper = (((uint64_t)temp->a << 40) | ((uint64_t)temp->b << 32) |
				 ((uint64_t)temp->c << 24) | ((uint64_t)temp->d << 16) |
				 ((uint64_t)temp->e <<  8) | ((uint64_t)temp->f));
	}

	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (upper == lower)
		x = lower;
	else
	{
		v = (__random() << 31) | __random();
		x = lower + v % (upper - lower);
	}
	temp = palloc(sizeof(macaddr));
	temp->a = (x >> 40) & 0x00ff;
	temp->b = (x >> 32) & 0x00ff;
	temp->c = (x >> 24) & 0x00ff;
	temp->d = (x >> 16) & 0x00ff;
	temp->e = (x >>  8) & 0x00ff;
	temp->f = (x      ) & 0x00ff;
	PG_RETURN_MACADDR_P(temp);
}

PG_FUNCTION_INFO_V1(pgstrom_random_inet);
PUBLIC_FUNCTION(Datum)
pgstrom_random_inet(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	inet	   *temp;
	int			i, j, bits;
	uint64_t	v;

	if (generate_null(ratio))
		PG_RETURN_NULL();

	if (!PG_ARGISNULL(1))
		temp = (inet *)PG_DETOAST_DATUM_COPY(PG_GETARG_DATUM(1));
	else
	{
		/* default template 192.168.0.1/16 */
		temp = palloc(sizeof(inet));
		temp->inet_data.family = PGSQL_AF_INET;
		temp->inet_data.bits = 16;
		temp->inet_data.ipaddr[0] = 0xc0;
		temp->inet_data.ipaddr[1] = 0xa8;
		temp->inet_data.ipaddr[2] = 0x01;
		temp->inet_data.ipaddr[3] = 0x00;
		SET_VARSIZE(temp, sizeof(inet));
	}
	bits = ip_bits(temp);
	i = ip_maxbits(temp) / 8 - 1;
	j = v = 0;
	while (bits > 0)
	{
		if (j < 8)
		{
			v |= __random() << j;
			j += 31;	/* note: only 31b of random() are valid */
		}
		if (bits >= 8)
			temp->inet_data.ipaddr[i--] = (v & 0xff);
		else
		{
			uint32_t		mask = (1 << bits) - 1;

			temp->inet_data.ipaddr[i] &= ~(mask);
			temp->inet_data.ipaddr[i] |= (v & mask);
			i--;
		}
		bits -= 8;
		v >>= 8;
	}
	ip_bits(temp) = ip_maxbits(temp);
	PG_RETURN_INET_P(temp);
}

PG_FUNCTION_INFO_V1(pgstrom_random_text);
PUBLIC_FUNCTION(Datum)
pgstrom_random_text(PG_FUNCTION_ARGS)
{
	static const char *base32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	text	   *temp;
	char	   *pos;
	int			i, j, n;
	uint64_t	v;

	if (generate_null(ratio))
		PG_RETURN_NULL();

	if (PG_ARGISNULL(1))
		temp = cstring_to_text("test_**");
	else
		temp = PG_GETARG_TEXT_P_COPY(1);

	n = VARSIZE_ANY_EXHDR(temp);
	pos = VARDATA_ANY(temp);
	for (i=0, j=0, v=0; i < n; i++, pos++)
	{
		if (*pos == '*')
		{
			if (j < 5)
			{
				v |= __random() << j;
				j += 31;
			}
			*pos = base32[v & 0x1f];
			v >>= 5;
			j -= 5;
		}
	}
	PG_RETURN_TEXT_P(temp);
}

PG_FUNCTION_INFO_V1(pgstrom_random_text_length);
PUBLIC_FUNCTION(Datum)
pgstrom_random_text_length(PG_FUNCTION_ARGS)
{
	static const char *base64 =
		"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		"abcdefghijklmnopqrstuvwxyz"
		"0123456789+/";
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	int32_t		maxlen;
	text	   *temp;
	char	   *pos;
	int			i, j, n;
	uint64_t	v = 0;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	maxlen = (PG_ARGISNULL(1) ? 10 : PG_GETARG_INT32(1));
	if (maxlen < 1 || maxlen > BLCKSZ)
		elog(ERROR, "%s: max length too much", __FUNCTION__);
	n = 1 + __random() % maxlen;

	temp = palloc(VARHDRSZ + n);
	SET_VARSIZE(temp, VARHDRSZ + n);
	pos = VARDATA(temp);
	for (i=0, j=0; i < n; i++, pos++)
	{
		if (j < 6)
		{
			v |= __random() << j;
			j += 31;
		}
		*pos = base64[v & 0x3f];
		v >>= 6;
		j -= 6;
	}
	PG_RETURN_TEXT_P(temp);
}

static Datum
simple_make_range(PG_FUNCTION_ARGS,
				  TypeCacheEntry *typcache, Datum x_val, Datum y_val)
{
	RangeBound	x, y;
	RangeType  *range;

	memset(&x, 0, sizeof(RangeBound));
	x.val = x_val;
	x.infinite = generate_null(0.5);
	x.inclusive = generate_null(25.0);
	x.lower = true;

	memset(&y, 0, sizeof(RangeBound));
	y.val = y_val;
	y.infinite = generate_null(0.5);
	y.inclusive = generate_null(25.0);
	y.lower = false;

	range = make_range(typcache, &x, &y, false,
					   fcinfo->context);
	return PointerGetDatum(range);
}

PG_FUNCTION_INFO_V1(pgstrom_random_int4range);
PUBLIC_FUNCTION(Datum)
pgstrom_random_int4range(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	int32		lower = (!PG_ARGISNULL(1) ? PG_GETARG_INT32(1) : 0);
	int32		upper = (!PG_ARGISNULL(2) ? PG_GETARG_INT32(2) : INT_MAX);
	int32		x, y;
	Oid			type_oid;
	TypeCacheEntry *typcache;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	type_oid = GetSysCacheOid2(TYPENAMENSP,
							   Anum_pg_type_oid,
							   CStringGetDatum("int4range"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	if (!OidIsValid(type_oid))
		elog(ERROR, "type 'int4range' is not defined");
	typcache = range_get_typcache(fcinfo, type_oid);
	x = lower + __random() % (upper - lower);
	y = lower + __random() % (upper - lower);
	return simple_make_range(fcinfo, typcache,
							 Int32GetDatum(Min(x,y)),
							 Int32GetDatum(Max(x,y)));
}

PG_FUNCTION_INFO_V1(pgstrom_random_int8range);
PUBLIC_FUNCTION(Datum)
pgstrom_random_int8range(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	int64		lower = (!PG_ARGISNULL(1) ? PG_GETARG_INT64(1) : 0);
	int64		upper = (!PG_ARGISNULL(2) ? PG_GETARG_INT64(2) : LONG_MAX);
	TypeCacheEntry *typcache;
	Oid			type_oid;
	int64		x, y, v;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	type_oid = GetSysCacheOid2(TYPENAMENSP,
							   Anum_pg_type_oid,
							   CStringGetDatum("int8range"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	if (!OidIsValid(type_oid))
		elog(ERROR, "type 'int8range' is not defined");
	typcache = range_get_typcache(fcinfo, type_oid);
	v = (__random() << 31) | __random();
	x = lower + v % (upper - lower);
	v = (__random() << 31) | __random();
	y = lower + v % (upper - lower);
	return simple_make_range(fcinfo, typcache,
							 Int64GetDatum(Min(x,y)),
							 Int64GetDatum(Max(x,y)));
}

PG_FUNCTION_INFO_V1(pgstrom_random_tsrange);
PUBLIC_FUNCTION(Datum)
pgstrom_random_tsrange(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	Timestamp	lower;
	Timestamp	upper;
	struct pg_tm tm;
	TypeCacheEntry *typcache;
	Oid			type_oid;
	Timestamp	x, y;
	uint64_t	v;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_TIMESTAMP(1);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 45;	/* '2015-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &lower) != 0)
			elog(ERROR, "timestamp out of range");
	}

	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_TIMESTAMP(2);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 55;	/* '2025-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &upper) != 0)
			elog(ERROR, "timestamp out of range");
	}
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	type_oid = GetSysCacheOid2(TYPENAMENSP,
							   Anum_pg_type_oid,
							   CStringGetDatum("tsrange"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	if (!OidIsValid(type_oid))
		elog(ERROR, "type 'tsrange' is not defined");
	typcache = range_get_typcache(fcinfo, type_oid);
	v = (__random() << 31) | __random();
	x = lower + v % (upper - lower);
	v = (__random() << 31) | __random();
	y = lower + v % (upper - lower);
	return simple_make_range(fcinfo, typcache,
							 TimestampGetDatum(Min(x,y)),
							 TimestampGetDatum(Max(x,y)));	
}

PG_FUNCTION_INFO_V1(pgstrom_random_tstzrange);
PUBLIC_FUNCTION(Datum)
pgstrom_random_tstzrange(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	Timestamp	lower;
	Timestamp	upper;
	struct pg_tm tm;
	TypeCacheEntry *typcache;
	Oid			type_oid;
	Timestamp	x, y;
	uint64_t	v;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_TIMESTAMP(1);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 45;	/* '2015-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &lower) != 0)
			elog(ERROR, "timestamp out of range");
	}

	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_TIMESTAMP(2);
	else
	{
		GetEpochTime(&tm);
		tm.tm_year += 55;	/* '2025-01-01' */
		if (tm2timestamp(&tm, 0, NULL, &upper) != 0)
			elog(ERROR, "timestamp out of range");
	}
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);
	type_oid = GetSysCacheOid2(TYPENAMENSP,
							   Anum_pg_type_oid,
							   CStringGetDatum("tstzrange"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	if (!OidIsValid(type_oid))
		elog(ERROR, "type 'tstzrange' is not defined");
	typcache = range_get_typcache(fcinfo, type_oid);
	v = (__random() << 31) | __random();
	x = lower + v % (upper - lower);
	v = (__random() << 31) | __random();
	y = lower + v % (upper - lower);
	return simple_make_range(fcinfo, typcache,
							 TimestampTzGetDatum(Min(x,y)),
							 TimestampTzGetDatum(Max(x,y)));	
}

PG_FUNCTION_INFO_V1(pgstrom_random_daterange);
PUBLIC_FUNCTION(Datum)
pgstrom_random_daterange(PG_FUNCTION_ARGS)
{
	float8		ratio = (!PG_ARGISNULL(0) ? PG_GETARG_FLOAT8(0) : 0.0);
	DateADT		lower;
	DateADT		upper;
	DateADT		x, y;
	TypeCacheEntry *typcache;
	Oid			type_oid;

	if (generate_null(ratio))
		PG_RETURN_NULL();
	if (!PG_ARGISNULL(1))
		lower = PG_GETARG_DATEADT(1);
	else
		lower = date2j(2015, 1, 1) - POSTGRES_EPOCH_JDATE;
	if (!PG_ARGISNULL(2))
		upper = PG_GETARG_DATEADT(2);
	else
		upper = date2j(2025, 12, 31) - POSTGRES_EPOCH_JDATE;
	if (upper < lower)
		elog(ERROR, "%s: lower bound is larger than upper", __FUNCTION__);

	type_oid = GetSysCacheOid2(TYPENAMENSP,
							   Anum_pg_type_oid,
							   CStringGetDatum("daterange"),
							   ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	if (!OidIsValid(type_oid))
		elog(ERROR, "type 'daterange' is not defined");
	typcache = range_get_typcache(fcinfo, type_oid);
	x = lower + __random() % (upper - lower);
	y = lower + __random() % (upper - lower);
	return simple_make_range(fcinfo, typcache,
							 DateADTGetDatum(Min(x,y)),
							 DateADTGetDatum(Max(x,y)));
}

PG_FUNCTION_INFO_V1(pgstrom_abort_if);
PUBLIC_FUNCTION(Datum)
pgstrom_abort_if(PG_FUNCTION_ARGS)
{
	bool		cond = PG_GETARG_BOOL(0);

	if (cond)
		elog(ERROR, "abort transaction");

	PG_RETURN_VOID();
}

/*
 * Simple wrapper for read(2) and write(2) to ensure full-buffer read and
 * write, regardless of i/o-size and signal interrupts.
 */
ssize_t
__readFile(int fdesc, void *buffer, size_t nbytes)
{
	ssize_t		rv, count = 0;

	do {
		rv = read(fdesc, (char *)buffer + count, nbytes - count);
		if (rv > 0)
			count += rv;
		else if (rv == 0)
			break;
		else if (errno == EINTR)
			CHECK_FOR_INTERRUPTS();
		else
			return rv;
	} while (count < nbytes);

	return count;
}

ssize_t
__preadFile(int fdesc, void *buffer, size_t nbytes, off_t f_pos)
{
	ssize_t		rv, count = 0;

	do {
		rv = pread(fdesc, (char *)buffer + count, nbytes - count, f_pos + count);
		if (rv > 0)
			count += rv;
		else if (rv == 0)
			break;
		else if (errno == EINTR)
			CHECK_FOR_INTERRUPTS();
		else
			return rv;
	} while (count < nbytes);

	return count;
}

ssize_t
__writeFile(int fdesc, const void *buffer, size_t nbytes)
{
	ssize_t		rv, count = 0;

	do {
		rv = write(fdesc, (const char *)buffer + count, nbytes - count);
		if (rv > 0)
			count += rv;
		else if (rv == 0)
			break;
		else if (errno == EINTR)
			CHECK_FOR_INTERRUPTS();
		else
			return rv;
	} while (count < nbytes);

	return count;
}

ssize_t
__pwriteFile(int fdesc, const void *buffer, size_t nbytes, off_t f_pos)
{
	ssize_t		rv, count = 0;

	do {
		rv = pwrite(fdesc, (const char *)buffer + count, nbytes - count, f_pos + count);
		if (rv > 0)
			count += rv;
		else if (rv == 0)
			break;
		else if (errno == EINTR)
			CHECK_FOR_INTERRUPTS();
		else
			return rv;
	} while (count < nbytes);

	return count;
}

/* ----------------------------------------------------------------
 *
 * shared memory and mmap/munmap routines
 *
 * ----------------------------------------------------------------
 */
#define IS_POSIX_SHMEM		0x80000000U
typedef struct
{
	uint32_t	shmem_handle;
	int			shmem_fdesc;
	char		shmem_name[MAXPGPATH];
	ResourceOwner owner;
} shmemEntry;

typedef struct
{
	void	   *mmap_addr;
	size_t		mmap_size;
	int			mmap_prot;
	int			mmap_flags;
	ResourceOwner owner;
} mmapEntry;

static HTAB	   *shmem_tracker_htab = NULL;
static HTAB	   *mmap_tracker_htab = NULL;

static void
cleanup_shmem_chunks(ResourceReleasePhase phase,
					 bool isCommit,
					 bool isTopLevel,
					 void *arg)
{
	if (phase == RESOURCE_RELEASE_AFTER_LOCKS &&
		shmem_tracker_htab &&
		hash_get_num_entries(shmem_tracker_htab) > 0)
	{
		HASH_SEQ_STATUS	seq;
		shmemEntry	   *entry;

		hash_seq_init(&seq, shmem_tracker_htab);
		while ((entry = hash_seq_search(&seq)) != NULL)
		{
			if (entry->owner != CurrentResourceOwner)
				continue;
			if (isCommit)
				elog(WARNING, "shared-memory '%s' leaks, and still alive",
					 entry->shmem_name);
			if (unlink(entry->shmem_name) != 0)
				elog(WARNING, "failed on unlink('%s'): %m", entry->shmem_name);
			if (close(entry->shmem_fdesc) != 0)
				elog(WARNING, "failed on close('%s'): %m", entry->shmem_name);
			hash_search(shmem_tracker_htab,
						&entry->shmem_handle,
						HASH_REMOVE,
						NULL);
		}
	}
}

static void
cleanup_mmap_chunks(ResourceReleasePhase phase,
					bool isCommit,
					bool isTopLevel,
					void *arg)
{
	if (phase == RESOURCE_RELEASE_AFTER_LOCKS &&
		mmap_tracker_htab &&
		hash_get_num_entries(mmap_tracker_htab) > 0)
	{
		HASH_SEQ_STATUS seq;
		mmapEntry	   *entry;

		hash_seq_init(&seq, mmap_tracker_htab);
		while ((entry = hash_seq_search(&seq)) != NULL)
		{
			if (entry->owner != CurrentResourceOwner)
				continue;
			if (isCommit)
				elog(WARNING, "mmap (%p-%p; sz=%zu) leaks, and still mapped",
					 (char *)entry->mmap_addr,
					 (char *)entry->mmap_addr + entry->mmap_size,
					 entry->mmap_size);
			if (munmap(entry->mmap_addr, entry->mmap_size) != 0)
				elog(WARNING, "failed on munmap(%p, %zu): %m",
					 entry->mmap_addr, entry->mmap_size);
			hash_search(mmap_tracker_htab,
						&entry->mmap_addr,
						HASH_REMOVE,
						NULL);
		}
	}
}

uint32_t
__shmemCreate(const DpuStorageEntry *ds_entry)
{
	static uint	my_random_seed = 0;
	const char *shmem_dir = "/dev/shm";
	int			fdesc;
	uint32_t	handle;
	char		namebuf[MAXPGPATH];
	size_t		off = 0;

	if (!shmem_tracker_htab)
	{
		HASHCTL		hctl;

		my_random_seed = (uint)MyProcPid ^ 0xcafebabeU;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize   = sizeof(uint32_t);
		hctl.entrysize = sizeof(shmemEntry);
		shmem_tracker_htab = hash_create("shmem_tracker_htab",
										 256,
										 &hctl,
										 HASH_ELEM | HASH_BLOBS);
		RegisterResourceReleaseCallback(cleanup_shmem_chunks, 0);
	}

	if (ds_entry)
		shmem_dir = DpuStorageEntryBaseDir(ds_entry);
	off = snprintf(namebuf, sizeof(namebuf), "%s/", shmem_dir);
	do {
		handle = rand_r(&my_random_seed);
		if (handle == 0)
			continue;
		/* to avoid hash conflict */
		if (!shmem_dir)
			handle |= IS_POSIX_SHMEM;
		else
			handle &= ~IS_POSIX_SHMEM;

		snprintf(namebuf + off, sizeof(namebuf) - off,
				 ".pgstrom_shmbuf_%u_%d",
				 PostPortNumber, handle);
		fdesc = open(namebuf, O_RDWR | O_CREAT | O_EXCL, 0600);
		if (fdesc < 0 && errno != EEXIST)
			elog(ERROR, "failed on open('%s'): %m", namebuf);
	} while (fdesc < 0);

	PG_TRY();
	{
		shmemEntry *entry;
		bool		found;

		entry = hash_search(shmem_tracker_htab,
							&handle,
							HASH_ENTER,
							&found);
		if (found)
			elog(ERROR, "Bug? duplicated shmem entry");
		entry->shmem_handle = handle;
		entry->shmem_fdesc  = fdesc;
		strcpy(entry->shmem_name, namebuf);
		entry->owner = CurrentResourceOwner;
	}
	PG_CATCH();
	{
		if (close(fdesc) != 0)
			elog(WARNING, "failed on close('%s'): %m", namebuf);
		if (unlink(namebuf) != 0)
			elog(WARNING, "failed on unlink('%s'): %m", namebuf);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return handle;
}

void
__shmemDrop(uint32_t shmem_handle)
{
	if (shmem_tracker_htab)
	{
		shmemEntry *entry;

		entry = hash_search(shmem_tracker_htab,
							&shmem_handle,
							HASH_REMOVE,
							NULL);
		if (entry)
		{
			if (unlink(entry->shmem_name) != 0)
				elog(WARNING, "failed on unlink('%s'): %m", entry->shmem_name);
			if (close(entry->shmem_fdesc) != 0)
				elog(WARNING, "failed on close('%s'): %m", entry->shmem_name);
			return;
		}
	}
	elog(ERROR, "failed on __shmemDrop - no such segment (%u)", shmem_handle);
}

void *
__mmapShmem(uint32_t shmem_handle,
			size_t   shmem_length,
			const DpuStorageEntry *ds_entry)
{
	void	   *mmap_addr = MAP_FAILED;
	size_t		mmap_size = TYPEALIGN(PAGE_SIZE, shmem_length);
	int			mmap_prot = PROT_READ | PROT_WRITE;
	int			mmap_flags = MAP_SHARED;
	mmapEntry  *mmap_entry = NULL;
	shmemEntry *shmem_entry = NULL;
	int			fdesc = -1;
	const char *shmem_dir = "/dev/shm";
	const char *fname = NULL;
	struct stat	stat_buf;
	bool		found;
	char		namebuf[MAXPGPATH];

	if (ds_entry)
		shmem_dir = DpuStorageEntryBaseDir(ds_entry);
	if (!mmap_tracker_htab)
	{
		HASHCTL		hctl;

		memset(&hctl, 0, sizeof(HASHCTL));
		hctl.keysize = sizeof(void *);
		hctl.entrysize = sizeof(mmapEntry);
		mmap_tracker_htab = hash_create("mmap_tracker_htab",
										256,
										&hctl,
										HASH_ELEM | HASH_BLOBS);
		RegisterResourceReleaseCallback(cleanup_mmap_chunks, 0);
	}

	if (shmem_tracker_htab)
	{
		shmem_entry = hash_search(shmem_tracker_htab,
								  &shmem_handle,
								  HASH_FIND,
								  NULL);
		if (shmem_entry)
		{
			size_t		len = strlen(shmem_dir);

			if (strncmp(shmem_entry->shmem_name, shmem_dir, len) != 0 ||
				shmem_entry->shmem_name[len] != '/')
				elog(ERROR, "Bug? shmem_dir mismatch '%s'", shmem_dir);
			fdesc = shmem_entry->shmem_fdesc;
			fname = shmem_entry->shmem_name;
		}
	}
	if (fdesc < 0)
	{
		snprintf(namebuf, sizeof(namebuf),
				 "%s/.pgstrom_shmbuf_%u_%d",
				 shmem_dir, PostPortNumber, shmem_handle);
		fdesc = open(namebuf, O_RDWR, 0600);
		if (fdesc < 0)
			elog(ERROR, "failed on open('%s'): %m", namebuf);
		fname = namebuf;
	}

	PG_TRY();
	{
		if (fstat(fdesc, &stat_buf) != 0)
			elog(ERROR, "failed on fstat('%s'): %m", fname);
		if (stat_buf.st_size < mmap_size)
		{
			while (fallocate(fdesc, 0, 0, mmap_size) != 0)
			{
				if (errno != EINTR)
					elog(ERROR, "failed on fallocate('%s', %lu): %m",
						 fname, mmap_size);
			}
		}
		mmap_addr = mmap(NULL, mmap_size, mmap_prot, mmap_flags, fdesc, 0);
		if (mmap_addr == MAP_FAILED)
			elog(ERROR, "failed on mmap(2): %m");

		mmap_entry = hash_search(mmap_tracker_htab,
								 &mmap_addr,
								 HASH_ENTER,
								 &found);
		if (found)
			elog(ERROR, "Bug? duplicated mmap entry");
		Assert(mmap_entry->mmap_addr == mmap_addr);
		mmap_entry->mmap_size  = mmap_size;
		mmap_entry->mmap_prot  = mmap_prot;
		mmap_entry->mmap_flags = mmap_flags;
		mmap_entry->owner      = CurrentResourceOwner;

		if (!shmem_entry)
			close(fdesc);
	}
	PG_CATCH();
	{
		if (mmap_addr != MAP_FAILED)
		{
			if (munmap(mmap_addr, mmap_size) != 0)
				elog(WARNING, "failed on munmap(%p, %zu) of '%s': %m",
					 mmap_addr, mmap_size, fname);
		}
		if (!shmem_entry && close(fdesc) != 0)
			elog(WARNING, "failed on close('%s'): %m", fname);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return mmap_addr;
}

bool
__munmapShmem(void *mmap_addr)
{
	if (mmap_tracker_htab)
	{
		mmapEntry  *entry
			= hash_search(mmap_tracker_htab,
						  &mmap_addr,
						  HASH_REMOVE,
						  NULL);
		if (entry)
		{
			if (munmap(entry->mmap_addr,
					   entry->mmap_size) != 0)
				elog(WARNING, "failed on munmap(%p, %zu): %m",
					 entry->mmap_addr,
					 entry->mmap_size);
			return true;
		}
	}
	elog(ERROR, "it looks addr=%p not memory-mapped", mmap_addr);
	return false;
}
