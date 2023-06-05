/*
 * gpu_join.c
 *
 * Multi-relations join accelerated with GPU processors
 * ----
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "cuda_common.h"


/* static variables */
static set_join_pathlist_hook_type	set_join_pathlist_next = NULL;
static CustomPathMethods	gpujoin_path_methods;
static CustomScanMethods	gpujoin_plan_methods;
static CustomExecMethods	gpujoin_exec_methods;
static bool					pgstrom_enable_gpujoin;			/* GUC */
static bool					pgstrom_enable_gpuhashjoin;		/* GUC */
static bool					pgstrom_enable_gpugistindex;	/* GUC */

/*
 * form_pgstrom_plan_info
 *
 * pgstromPlanInfo --> custom_private/custom_exprs
 */
void
form_pgstrom_plan_info(CustomScan *cscan, pgstromPlanInfo *pp_info)
{
	List   *privs = NIL;
	List   *exprs = NIL;
	int		endpoint_id;

	privs = lappend(privs, makeInteger(pp_info->xpu_task_flags));
	privs = lappend(privs, bms_to_pglist(pp_info->gpu_cache_devs));
	privs = lappend(privs, bms_to_pglist(pp_info->gpu_direct_devs));
	endpoint_id = DpuStorageEntryGetEndpointId(pp_info->ds_entry);
	privs = lappend(privs, makeInteger(endpoint_id));
	/* plan information */
	privs = lappend(privs, bms_to_pglist(pp_info->outer_refs));
	exprs = lappend(exprs, pp_info->used_params);
	exprs = lappend(exprs, pp_info->host_quals);
	privs = lappend(privs, makeInteger(pp_info->scan_relid));
	exprs = lappend(exprs, pp_info->scan_quals);
	privs = lappend(privs, pp_info->scan_quals_fallback);
	privs = lappend(privs, __makeFloat(pp_info->scan_tuples));
	privs = lappend(privs, __makeFloat(pp_info->scan_rows));
	privs = lappend(privs, __makeFloat(pp_info->parallel_divisor));
	privs = lappend(privs, __makeFloat(pp_info->final_cost));
	/* bin-index support */
	privs = lappend(privs, makeInteger(pp_info->brin_index_oid));
	exprs = lappend(exprs, pp_info->brin_index_conds);
	exprs = lappend(exprs, pp_info->brin_index_quals);
	/* XPU code */
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_scan_kvars_load));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_scan_quals));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_join_kvars_load_packed));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_join_quals_packed));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_hash_keys_packed));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_gist_evals_packed));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_projection));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_groupby_keyhash));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_groupby_keyload));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_groupby_keycomp));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_groupby_actions));
	privs = lappend(privs, pp_info->kvars_depth);
	privs = lappend(privs, pp_info->kvars_resno);
	privs = lappend(privs, pp_info->kvars_types);
	exprs = lappend(exprs, pp_info->kvars_exprs);
	privs = lappend(privs, makeInteger(pp_info->extra_flags));
	privs = lappend(privs, makeInteger(pp_info->extra_bufsz));
	privs = lappend(privs, pp_info->fallback_tlist);
	privs = lappend(privs, pp_info->groupby_actions);
	privs = lappend(privs, pp_info->groupby_keys);
	/* inner relations */
	privs = lappend(privs, makeInteger(pp_info->num_rels));
	for (int i=0; i < pp_info->num_rels; i++)
	{
		pgstromPlanInnerInfo *pp_inner = &pp_info->inners[i];
		List   *__privs = NIL;
		List   *__exprs = NIL;

		__privs = lappend(__privs, makeInteger(pp_inner->join_type));
		__privs = lappend(__privs, __makeFloat(pp_inner->join_nrows));
		__exprs = lappend(__exprs, pp_inner->hash_outer_keys);
		__privs = lappend(__privs, pp_inner->hash_outer_keys_fallback);
		__exprs = lappend(__exprs, pp_inner->hash_inner_keys);
		__privs = lappend(__privs, pp_inner->hash_inner_keys_fallback);
		__exprs = lappend(__exprs, pp_inner->join_quals);
		__privs = lappend(__privs, pp_inner->join_quals_fallback);
		__exprs = lappend(__exprs, pp_inner->other_quals);
		__privs = lappend(__privs, pp_inner->other_quals_fallback);
		__privs = lappend(__privs, makeInteger(pp_inner->gist_index_oid));
		__privs = lappend(__privs, makeInteger(pp_inner->gist_index_col));
		__privs = lappend(__privs, makeInteger(pp_inner->gist_func_oid));
		__privs = lappend(__privs, makeInteger(pp_inner->gist_slot_id));
		__exprs = lappend(__exprs, pp_inner->gist_clause);
		__privs = lappend(__privs, __makeFloat(pp_inner->gist_selectivity));
		__privs = lappend(__privs, __makeFloat(pp_inner->gist_npages));
		__privs = lappend(__privs, makeInteger(pp_inner->gist_height));

		privs = lappend(privs, __privs);
		exprs = lappend(exprs, __exprs);
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
	int			endpoint_id;

	memset(&pp_data, 0, sizeof(pgstromPlanInfo));
	/* device identifiers */
	pp_data.xpu_task_flags = intVal(list_nth(privs, pindex++));
	pp_data.gpu_cache_devs = bms_from_pglist(list_nth(privs, pindex++));
	pp_data.gpu_direct_devs = bms_from_pglist(list_nth(privs, pindex++));
	endpoint_id = intVal(list_nth(privs, pindex++));
	pp_data.ds_entry = DpuStorageEntryByEndpointId(endpoint_id);
	/* plan information */
	pp_data.outer_refs = bms_from_pglist(list_nth(privs, pindex++));
	pp_data.used_params = list_nth(exprs, eindex++);
	pp_data.host_quals = list_nth(exprs, eindex++);
	pp_data.scan_relid = intVal(list_nth(privs, pindex++));
	pp_data.scan_quals = list_nth(exprs, eindex++);
	pp_data.scan_quals_fallback = list_nth(privs, pindex++);
	pp_data.scan_tuples = floatVal(list_nth(privs, pindex++));
	pp_data.scan_rows = floatVal(list_nth(privs, pindex++));
	pp_data.parallel_divisor = floatVal(list_nth(privs, pindex++));
	pp_data.final_cost = floatVal(list_nth(privs, pindex++));
	/* brin-index support */
	pp_data.brin_index_oid = intVal(list_nth(privs, pindex++));
	pp_data.brin_index_conds = list_nth(exprs, eindex++);
	pp_data.brin_index_quals = list_nth(exprs, eindex++);
	/* XPU code */
	pp_data.kexp_scan_kvars_load = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_scan_quals = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_join_kvars_load_packed = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_join_quals_packed = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_hash_keys_packed = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_gist_evals_packed = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_projection = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_groupby_keyhash = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_groupby_keyload = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_groupby_keycomp = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_groupby_actions = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kvars_depth = list_nth(privs, pindex++);
	pp_data.kvars_resno = list_nth(privs, pindex++);
	pp_data.kvars_types = list_nth(privs, pindex++);
	pp_data.kvars_exprs = list_nth(exprs, eindex++);
	pp_data.extra_flags = intVal(list_nth(privs, pindex++));
	pp_data.extra_bufsz = intVal(list_nth(privs, pindex++));
	pp_data.fallback_tlist = list_nth(privs, pindex++);
	pp_data.groupby_actions = list_nth(privs, pindex++);
	pp_data.groupby_keys = list_nth(privs, pindex++);
	/* inner relations */
	pp_data.num_rels = intVal(list_nth(privs, pindex++));
	pp_info = palloc0(offsetof(pgstromPlanInfo, inners[pp_data.num_rels]));
	memcpy(pp_info, &pp_data, offsetof(pgstromPlanInfo, inners));
	for (int i=0; i < pp_info->num_rels; i++)
	{
		pgstromPlanInnerInfo *pp_inner = &pp_info->inners[i];
		List   *__privs = list_nth(privs, pindex++);
		List   *__exprs = list_nth(exprs, eindex++);
		int		__pindex = 0;
		int		__eindex = 0;

		pp_inner->join_type       = intVal(list_nth(__privs, __pindex++));
		pp_inner->join_nrows      = floatVal(list_nth(__privs, __pindex++));
		pp_inner->hash_outer_keys = list_nth(__exprs, __eindex++);
		pp_inner->hash_outer_keys_fallback = list_nth(__privs, __pindex++);
		pp_inner->hash_inner_keys = list_nth(__exprs, __eindex++);
		pp_inner->hash_inner_keys_fallback = list_nth(__privs, __pindex++);
		pp_inner->join_quals      = list_nth(__exprs, __eindex++);
		pp_inner->join_quals_fallback = list_nth(__privs, __pindex++);
		pp_inner->other_quals     = list_nth(__exprs, __eindex++);
		pp_inner->other_quals_fallback = list_nth(__privs, __pindex++);
		pp_inner->gist_index_oid  = intVal(list_nth(__privs, __pindex++));
		pp_inner->gist_index_col  = intVal(list_nth(__privs, __pindex++));
		pp_inner->gist_func_oid   = intVal(list_nth(__privs, __pindex++));
		pp_inner->gist_slot_id    = intVal(list_nth(__privs, __pindex++));
		pp_inner->gist_clause     = list_nth(__exprs, __eindex++);
		pp_inner->gist_selectivity = floatVal(list_nth(__privs, __pindex++));
		pp_inner->gist_npages     = floatVal(list_nth(__privs, __pindex++));
		pp_inner->gist_height     = intVal(list_nth(__privs, __pindex++));
	}
	return pp_info;
}

/*
 * copy_pgstrom_plan_info
 */
static pgstromPlanInfo *
copy_pgstrom_plan_info(pgstromPlanInfo *pp_orig)
{
	pgstromPlanInfo *pp_dest;

	/*
	 * NOTE: we add one pgstromPlanInnerInfo margin to be used for GpuJoin.
	 */
	pp_dest = palloc0(offsetof(pgstromPlanInfo, inners[pp_orig->num_rels+1]));
	memcpy(pp_dest, pp_orig, offsetof(pgstromPlanInfo,
									  inners[pp_orig->num_rels]));
	pp_dest->used_params      = list_copy(pp_dest->used_params);
	pp_dest->host_quals       = copyObject(pp_dest->host_quals);
	pp_dest->scan_quals       = copyObject(pp_dest->scan_quals);
	pp_dest->scan_quals_fallback = copyObject(pp_dest->scan_quals_fallback);
	pp_dest->brin_index_conds = copyObject(pp_dest->brin_index_conds);
	pp_dest->brin_index_quals = copyObject(pp_dest->brin_index_quals);
	pp_dest->kvars_depth      = list_copy(pp_dest->kvars_depth);
	pp_dest->kvars_resno      = list_copy(pp_dest->kvars_resno);
	pp_dest->kvars_types      = list_copy(pp_dest->kvars_types);
	pp_dest->kvars_exprs      = copyObject(pp_dest->kvars_exprs);
	pp_dest->fallback_tlist   = copyObject(pp_dest->fallback_tlist);
	pp_dest->groupby_actions  = list_copy(pp_dest->groupby_actions);
	pp_dest->groupby_keys     = list_copy(pp_dest->groupby_keys);
	for (int j=0; j < pp_orig->num_rels; j++)
	{
		pgstromPlanInnerInfo *pp_inner = &pp_dest->inners[j];
#define __COPY(FIELD)	pp_inner->FIELD = copyObject(pp_inner->FIELD)
		__COPY(hash_outer_keys);
		__COPY(hash_outer_keys_fallback);
		__COPY(hash_inner_keys);
		__COPY(hash_inner_keys_fallback);
		__COPY(join_quals);
		__COPY(join_quals_fallback);
		__COPY(other_quals);
		__COPY(other_quals_fallback);
		__COPY(gist_clause);
#undef __COPY
	}
	return pp_dest;
}

/*
 * extract_input_path_params
 *
 * centralized point to extract the information from the input path 
 */
void
extract_input_path_params(const Path *input_path,
						  const Path *inner_path,   /* optional */
						  pgstromPlanInfo **p_pp_info,
						  List **p_input_rels_tlist,
						  List **p_inner_paths_list)
{
	const CustomPath *input_cpath = (const CustomPath *)input_path;
	pgstromPlanInfo *pp_info;
	List	   *input_rels_tlist;
	List	   *inner_paths_list;
	ListCell   *lc;

	Assert(IsA(input_cpath, CustomPath));
	pp_info = linitial(input_cpath->custom_private);
	input_rels_tlist = list_make1(makeInteger(pp_info->scan_relid));
	inner_paths_list = list_copy(input_cpath->custom_paths);
	foreach (lc, inner_paths_list)
	{
		Path   *i_path = lfirst(lc);
		input_rels_tlist = lappend(input_rels_tlist, i_path->pathtarget);
	}
	if (inner_path)
		input_rels_tlist = lappend(input_rels_tlist, inner_path->pathtarget);

	if (p_pp_info)
		*p_pp_info = copy_pgstrom_plan_info(pp_info);
	if (p_input_rels_tlist)
		*p_input_rels_tlist = input_rels_tlist;
	if (p_inner_paths_list)
		*p_inner_paths_list = inner_paths_list;
}

/*
 * try_add_simple_xpujoin_path
 */
static bool
try_add_simple_xpujoin_path(PlannerInfo *root,
							RelOptInfo *joinrel,
							RelOptInfo *outer_rel,
                            Path *inner_path,
                            JoinType join_type,
                            JoinPathExtraData *extra,
							bool try_parallel_path,
							uint32_t xpu_task_flags,
							const CustomPathMethods *xpujoin_path_methods)
{
	Path		   *outer_path;
	RelOptInfo	   *inner_rel = inner_path->parent;
	List		   *inner_paths_list = NIL;
	List		   *restrict_clauses = extra->restrictlist;
	Relids			required_outer = NULL;
	ParamPathInfo  *param_info;
	CustomPath	   *cpath;
	pgstromPlanInfo	*pp_prev;
	pgstromPlanInfo	*pp_info;
	pgstromPlanInnerInfo *pp_inner;
	List		   *join_quals = NIL;
	List		   *other_quals = NIL;
	List		   *hash_outer_keys = NIL;
	List		   *hash_inner_keys = NIL;
	List		   *input_rels_tlist = NIL;
	bool			enable_xpuhashjoin;
	bool			enable_xpugistindex;
	double			xpu_ratio;
	Cost			xpu_tuple_cost;
	Cost			startup_cost = 0.0;
	Cost			run_cost = 0.0;
	Cost			comp_cost = 0.0;
	Cost			final_cost = 0.0;
	QualCost		join_quals_cost;
	ListCell	   *lc;

	/* sanity checks */
	Assert(join_type == JOIN_INNER || join_type == JOIN_FULL ||
		   join_type == JOIN_LEFT  || join_type == JOIN_RIGHT);
	/*
	 * Parameters related to devices
	 */
	if ((xpu_task_flags & DEVKIND__ANY) == DEVKIND__NVIDIA_GPU)
	{
		enable_xpuhashjoin  = pgstrom_enable_gpuhashjoin;
		enable_xpugistindex = pgstrom_enable_gpugistindex;
		xpu_tuple_cost      = pgstrom_gpu_tuple_cost;
		xpu_ratio           = pgstrom_gpu_operator_ratio();
	}
	else if ((xpu_task_flags & DEVKIND__ANY) == DEVKIND__NVIDIA_DPU)
	{
		enable_xpuhashjoin  = pgstrom_enable_dpuhashjoin;
		enable_xpugistindex = pgstrom_enable_dpugistindex;
        xpu_tuple_cost      = pgstrom_dpu_tuple_cost;
		xpu_ratio           = pgstrom_dpu_operator_ratio();
	}
	else
	{
		elog(ERROR, "Bug? unexpected xpu_task_flags: %08x", xpu_task_flags);
	}

	/*
	 * Setup Outer Path
	 */
	if (IS_SIMPLE_REL(outer_rel))
	{
		outer_path = (Path *) buildXpuScanPath(root,
											   outer_rel,
											   try_parallel_path,
											   false,
											   true,
											   xpu_task_flags);
		if (!outer_path)
			return false;
	}
	else
	{
		outer_path = (Path *) custom_path_find_cheapest(root,
														outer_rel,
														try_parallel_path,
														xpu_task_flags);
		if (!outer_path)
			return false;
	}
	if (bms_overlap(PATH_REQ_OUTER(outer_path), inner_rel->relids))
		return false;
	/* extract the parameters of outer_path */
	extract_input_path_params(outer_path,
							  inner_path,
							  &pp_prev,
							  &input_rels_tlist,
							  &inner_paths_list);
	startup_cost = outer_path->startup_cost;
	run_cost = (outer_path->total_cost -
				outer_path->startup_cost - pp_prev->final_cost);

	/*
	 * Check to see if proposed path is still parameterized, and reject
	 * if the parameterization wouldn't be sensible.
	 * Note that GpuNestLoop does not support parameterized nest-loop,
	 * only cross-join or non-symmetric join are supported, therefore,
	 * calc_non_nestloop_required_outer() is sufficient.
	 */
	required_outer = calc_non_nestloop_required_outer(outer_path,
													  inner_path);
	if (required_outer && !bms_overlap(required_outer,
									   extra->param_source_rels))
	{
		bms_free(required_outer);
		return false;
	}

	/*
	 * Get param info
	 */
	param_info = get_joinrel_parampathinfo(root,
										   joinrel,
										   outer_path,
										   inner_path,
										   extra->sjinfo,
										   required_outer,
										   &restrict_clauses);
	if (!restrict_clauses)
		return false;		/* cross join is not welcome */

	/*
	 * Setup pgstromPlanInfo
	 */
	pp_info = palloc0(offsetof(pgstromPlanInfo, inners[pp_prev->num_rels+1]));
	memcpy(pp_info, pp_prev, offsetof(pgstromPlanInfo, inners[pp_prev->num_rels]));
	pp_info->xpu_task_flags = xpu_task_flags;
	pp_info->num_rels = pp_prev->num_rels + 1;
	pp_inner = &pp_info->inners[pp_prev->num_rels];

	/*
	 * All the join-clauses must be executable on GPU device.
	 * Even though older version supports HostQuals to be
	 * applied post device join, it leads undesirable (often
	 * unacceptable) growth of the result rows in device join.
	 * So, we simply reject any join that contains host-only
	 * qualifiers.
	 */
	foreach (lc, restrict_clauses)
	{
		RestrictInfo   *rinfo = lfirst(lc);

		if (!pgstrom_xpu_expression(rinfo->clause,
									xpu_task_flags,
									input_rels_tlist,
									NULL))
		{
			return false;
		}

		/*
		 * See the logic in extract_actual_join_clauses()
		 * We need to distinguish between the join-qualifiers explicitly described
		 * in the OUTER JOIN ... ON clause from other qualifiers come from WHERE-
		 * clause then pushed down to the joinrel.
		 */
		if (IS_OUTER_JOIN(join_type) &&
			RINFO_IS_PUSHED_DOWN(rinfo, joinrel->relids))
		{
			if (!rinfo->pseudoconstant)
				other_quals = lappend(other_quals, rinfo->clause);
			continue;
		}
		else
		{
			Assert(!rinfo->pseudoconstant);
			join_quals = lappend(join_quals, rinfo->clause);
		}
		/* Is the hash-join enabled? */
		if (!enable_xpuhashjoin)
			continue;
		/* Is it hash-joinable clause? */
		if (!rinfo->can_join || !OidIsValid(rinfo->hashjoinoperator))
			continue;
		Assert(is_opclause(rinfo->clause));
		/*
		 * Check if clause has the form "outer op inner" or
		 * "inner op outer". If suitable, we may be able to choose
		 * GpuHashJoin logic. See clause_sides_match_join also.
		 */
		if ((bms_is_subset(rinfo->left_relids,  outer_rel->relids) &&
			 bms_is_subset(rinfo->right_relids, inner_rel->relids)) ||
			(bms_is_subset(rinfo->left_relids,  inner_rel->relids) &&
			 bms_is_subset(rinfo->right_relids, outer_rel->relids)))
		{
			OpExpr *op = (OpExpr *)rinfo->clause;
			Node   *arg1 = linitial(op->args);
			Node   *arg2 = lsecond(op->args);
			Relids	relids1 = pull_varnos(root, arg1);
			Relids	relids2 = pull_varnos(root, arg2);
			devtype_info *dtype;

			/* hash-join key must support device hash function */
			dtype = pgstrom_devtype_lookup(exprType(arg1));
			if (!dtype || !dtype->type_hashfunc)
				continue;
			dtype = pgstrom_devtype_lookup(exprType(arg2));
			if (!dtype || !dtype->type_hashfunc)
				continue;
			if (bms_is_subset(relids1, outer_rel->relids) &&
				bms_is_subset(relids2, inner_rel->relids))
			{
				hash_outer_keys = lappend(hash_outer_keys, arg1);
				hash_inner_keys = lappend(hash_inner_keys, arg2);
			}
			else if (bms_is_subset(relids1, inner_rel->relids) &&
					 bms_is_subset(relids2, outer_rel->relids))
			{
				hash_inner_keys = lappend(hash_inner_keys, arg1);
				hash_outer_keys = lappend(hash_outer_keys, arg2);
			}
			bms_free(relids1);
			bms_free(relids2);
		}
	}
	pp_inner->join_type = join_type;
	pp_inner->join_nrows = joinrel->rows;
	pp_inner->hash_outer_keys = hash_outer_keys;
	pp_inner->hash_inner_keys = hash_inner_keys;
	pp_inner->join_quals = join_quals;
	pp_inner->other_quals = other_quals;
	if (enable_xpugistindex &&
		pp_inner->hash_outer_keys == NIL &&
		pp_inner->hash_inner_keys == NIL)
		pgstromTryFindGistIndex(root,
								inner_path,
								restrict_clauses,
								xpu_task_flags,
								input_rels_tlist,
								pp_inner);
	/*
	 * Cost estimation
	 */
	startup_cost += (inner_path->total_cost +
					 inner_path->rows * cpu_tuple_cost);

	/* cost for join_quals */
	cost_qual_eval(&join_quals_cost, join_quals, root);
	startup_cost += join_quals_cost.startup;
	if (hash_outer_keys != NIL && hash_inner_keys != NIL)
	{
		/*
		 * GpuHashJoin - It computes hash-value of inner tuples by CPU,
		 * but outer tuples by GPU, then it evaluates join-qualifiers
		 * for each items on inner hash table by GPU.
		 */
		int		num_hashkeys = list_length(hash_outer_keys);

		/* cost to compute inner hash value by CPU */
		startup_cost += cpu_operator_cost * num_hashkeys * inner_path->rows;
		/* cost to comput hash value by GPU */
		comp_cost += (cpu_operator_cost * xpu_ratio *
					  num_hashkeys *
					  outer_path->rows);
		/* cost to evaluate join qualifiers */
		comp_cost += join_quals_cost.per_tuple * xpu_ratio * outer_path->rows;
	}
	else if (OidIsValid(pp_inner->gist_index_oid))
	{
		/*
		 * GpuNestLoop+GiST-Index
		 */
		Expr	   *gist_clause = pp_inner->gist_clause;
		double		gist_selectivity = pp_inner->gist_selectivity;
		QualCost	gist_clause_cost;

		/* cost to preload inner heap tuples by CPU */
		startup_cost += cpu_tuple_cost * inner_path->rows;
		/* cost to preload the entire index pages once */
		startup_cost += seq_page_cost * pp_inner->gist_npages;
		/* cost to evaluate GiST index by GPU */
		cost_qual_eval_node(&gist_clause_cost, (Node *)gist_clause, root);
		comp_cost += gist_clause_cost.per_tuple * xpu_ratio * outer_path->rows;
		/* cost to evaluate join qualifiers by GPU */
		comp_cost += (join_quals_cost.per_tuple * xpu_ratio *
					  outer_path->rows *
					  gist_selectivity *
					  inner_path->rows);
	}
	else
	{
		/*
		 * GpuNestLoop - It evaluates join-qual for each pair of outer
		 * and inner tuples. So, its run_cost is usually higher than
		 * GpuHashJoin.
		 */

		/* cost to preload inner heap tuples by CPU */
		startup_cost += cpu_tuple_cost * inner_path->rows;

		/* cost to evaluate join qualifiers by GPU */
		run_cost += (join_quals_cost.per_tuple * xpu_ratio *
					 inner_path->rows *
					 outer_path->rows);
	}
	/* discount if CPU parallel is enabled */
	run_cost += (comp_cost / pp_info->parallel_divisor);

	/* cost for DMA receive (xPU --> Host) */
	final_cost += xpu_tuple_cost * joinrel->rows;

	/* cost for host projection */
	final_cost += joinrel->reltarget->cost.per_tuple * joinrel->rows;

	pp_info->final_cost = final_cost;

	/*
	 * Build the CustomPath
	 */
	cpath = makeNode(CustomPath);
	cpath->path.pathtype = T_CustomScan;
	cpath->path.parent = joinrel;
	cpath->path.pathtarget = joinrel->reltarget;
	cpath->path.param_info = param_info;
	cpath->path.parallel_aware = try_parallel_path;
	cpath->path.parallel_safe = joinrel->consider_parallel;
	cpath->path.parallel_workers = outer_path->parallel_workers;
	cpath->path.pathkeys = NIL;
	cpath->path.rows = joinrel->rows;
	cpath->path.startup_cost = startup_cost;
	cpath->path.total_cost = startup_cost + run_cost + final_cost;
	cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
	cpath->methods = xpujoin_path_methods;
	cpath->custom_paths = lappend(inner_paths_list, inner_path);
	cpath->custom_private = list_make1(pp_info);
	
	if (custom_path_remember(root,
							 joinrel,
							 try_parallel_path,
							 xpu_task_flags,
							 cpath))
	{
		if (!try_parallel_path)
			add_path(joinrel, &cpath->path);
		else
			add_partial_path(joinrel, &cpath->path);
	}
	return true;
}

/*
 * xpujoin_add_custompath
 */
void
xpujoin_add_custompath(PlannerInfo *root,
                       RelOptInfo *joinrel,
                       RelOptInfo *outerrel,
                       RelOptInfo *innerrel,
                       JoinType join_type,
                       JoinPathExtraData *extra,
					   uint32_t xpu_task_flags,
					   const CustomPathMethods *xpujoin_path_methods)
{
	List	   *inner_pathlist;
	ListCell   *lc;

	/* quick bailout if unsupported join type */
	if (join_type != JOIN_INNER &&
		join_type != JOIN_FULL &&
		join_type != JOIN_RIGHT &&
		join_type != JOIN_LEFT)
		return;
	//TODO: JOIN_SEMI and JOIN_ANTI

	inner_pathlist = innerrel->pathlist;
	for (int try_parallel=0; try_parallel < 2; try_parallel++)
	{
		Path	   *inner_path = NULL;

		/* pickup the cheapest inner path */
		foreach (lc, inner_pathlist)
		{
			Path   *path = lfirst(lc);

			if (bms_overlap(PATH_REQ_OUTER(path), outerrel->relids))
				continue;
			if (try_parallel > 0 && !path->parallel_safe)
				continue;
			if (!inner_path || inner_path->total_cost > path->total_cost)
				inner_path = path;
		}

		if (inner_path)
			try_add_simple_xpujoin_path(root,
										joinrel,
										outerrel,
										inner_path,
										join_type,
										extra,
										try_parallel > 0,
										xpu_task_flags,
										xpujoin_path_methods);
		/* 2nd trial uses the partial paths */
		inner_pathlist = innerrel->partial_pathlist;
	}
}

/*
 * gpujoin_add_custompath
 */
static void
gpujoin_add_custompath(PlannerInfo *root,
					   RelOptInfo *joinrel,
					   RelOptInfo *outerrel,
					   RelOptInfo *innerrel,
					   JoinType join_type,
					   JoinPathExtraData *extra)
{
	/* calls secondary module if exists */
	if (set_join_pathlist_next)
		set_join_pathlist_next(root,
							   joinrel,
							   outerrel,
							   innerrel,
							   join_type,
							   extra);
	/* quick bailout if PG-Strom/GpuJoin is not enabled */
	if (!pgstrom_enabled || !pgstrom_enable_gpujoin)
		return;
	/* common portion to add custom-paths for xPU-Join */
	xpujoin_add_custompath(root,
						   joinrel,
						   outerrel,
						   innerrel,
						   join_type,
						   extra,
						   TASK_KIND__GPUJOIN,
						   &gpujoin_path_methods);
}

/*
 * build_fallback_exprs_scan
 */
typedef struct
{
	Index		scan_relid;
	List	   *kvars_depth;
	List	   *kvars_resno;
	List	   *kvars_exprs;
	PathTarget *inner_tlist;
} build_fallback_exprs_context;

static Node *
__build_fallback_exprs_scan_walker(Node *node, void *data)
{
	build_fallback_exprs_context *con = data;

	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		Var	   *var = (Var *)node;

		if (var->varno == con->scan_relid)
		{
			return (Node *)makeVar(OUTER_VAR,
								   var->varattno,
								   var->vartype,
								   var->vartypmod,
								   var->varcollid,
								   var->varlevelsup);
		}
		elog(ERROR, "Var-node does not reference the base relation (%d): %s",
			 con->scan_relid, nodeToString(var));
	}
	return expression_tree_mutator(node, __build_fallback_exprs_scan_walker, data);
}

List *
build_fallback_exprs_scan(Index scan_relid, List *scan_exprs)
{
	build_fallback_exprs_context con;

	//TODO: fixup partitioned var-node references
	memset(&con, 0, sizeof(con));
	con.scan_relid = scan_relid;
	return (List *)__build_fallback_exprs_scan_walker((Node *)scan_exprs, &con);
}

/*
 * build_fallback_exprs_join
 */
static Node *
__build_fallback_exprs_join_walker(Node *node, void *data)
{
	build_fallback_exprs_context *con = data;
	ListCell   *lc;
	int			slot_id = 0;

	if (!node)
		return NULL;
	foreach (lc, con->kvars_exprs)
	{
		Node   *curr = lfirst(lc);

		if (equal(node, curr))
		{
			return (Node *)makeVar(INDEX_VAR,
								   slot_id + 1,
								   exprType(node),
								   exprTypmod(node),
								   exprCollation(node),
								   0);
		}
		slot_id++;
	}
	if (IsA(node, Var))
		elog(ERROR, "Bug? Var-node (%s) is missing at the kvars_exprs list: %s",
			 nodeToString(node), nodeToString(con->kvars_exprs));
	return expression_tree_mutator(node, __build_fallback_exprs_join_walker, con);
}

List *
build_fallback_exprs_join(codegen_context *context, List *join_exprs)
{
	build_fallback_exprs_context con;

	memset(&con, 0, sizeof(con));
	con.kvars_depth = context->kvars_depth;
	con.kvars_resno = context->kvars_resno;
	con.kvars_exprs = context->kvars_exprs;
	return (List *)__build_fallback_exprs_join_walker((Node *)join_exprs, &con);
}


/*
 * build_fallback_exprs_inner
 */
static Node *
__build_fallback_exprs_inner_walker(Node *node, void *data)
{
	build_fallback_exprs_context *con = data;
	PathTarget *inner_tlist = con->inner_tlist;
	ListCell   *lc;
	int			slot_id = 0;

	if (!node)
		return NULL;
	foreach (lc, inner_tlist->exprs)
	{
		Node   *curr = lfirst(lc);

		if (equal(node, curr))
		{
			return (Node *)makeVar(INNER_VAR,
								   slot_id + 1,
								   exprType(node),
                                   exprTypmod(node),
                                   exprCollation(node),
                                   0);
		}
		slot_id++;
	}
	if (IsA(node, Var))
		 elog(ERROR, "Bug? Var-node (%s) is missing at the inner tlist: %s",
			  nodeToString(node), nodeToString(con->inner_tlist));
	return expression_tree_mutator(node, __build_fallback_exprs_inner_walker, con);
}

static List *
build_fallback_exprs_inner(codegen_context *context, int depth, List *inner_hash_keys)
{
	build_fallback_exprs_context con;

	memset(&con, 0, sizeof(con));
	con.inner_tlist = list_nth(context->input_rels_tlist, depth);
	return (List *)__build_fallback_exprs_inner_walker((Node *)inner_hash_keys, &con);
}

/*
 * pgstrom_build_tlist_dev
 */
typedef struct
{
	PlannerInfo *root;
	List	   *tlist_dev;
	List	   *input_rels_tlist;
	uint32_t	xpu_task_flags;
	bool		only_vars;
	bool		resjunk;
} build_tlist_dev_context;

static bool
__pgstrom_build_tlist_dev_walker(Node *node, void *__priv)
{
	build_tlist_dev_context *context = __priv;
	ListCell   *lc;
	int			depth;
	int			resno;

	if (!node)
		return false;

	/* check whether the node is already on the tlist_dev */
	foreach (lc, context->tlist_dev)
	{
		TargetEntry	*tle = lfirst(lc);

		if (equal(node, tle->expr))
			return false;
	}

	/* check whether the node is identical with any of input */
	depth = 0;
	foreach (lc, context->input_rels_tlist)
	{
		Node   *curr = lfirst(lc);

		if (IsA(curr, Integer))
		{
			Index	varno = intVal(curr);
			Var	   *var = (Var *)node;

			if (IsA(var, Var) && var->varno == varno)
			{
				resno = var->varattno;
				goto found;
			}
		}
		else if (IsA(curr, PathTarget))
		{
			PathTarget *reltarget = (PathTarget *)curr;
			ListCell   *cell;

			resno = 1;
			foreach (cell, reltarget->exprs)
			{
				if (equal(node, lfirst(cell)))
					goto found;
				resno++;
			}
		}
		else
		{
			elog(ERROR, "Bug? unexpected input_rels_tlist entry");
		}
		depth++;
	}
	depth = -1;
	resno = -1;
found:
	if (IsA(node, Var) ||
		(!context->only_vars &&
		 pgstrom_xpu_expression((Expr *)node,
								context->xpu_task_flags,
								context->input_rels_tlist,
								NULL)))
	{
		TargetEntry *tle;

		tle = makeTargetEntry((Expr *)node,
							  list_length(context->tlist_dev) + 1,
							  NULL,
							  context->resjunk);
		tle->resorigtbl = (depth < 0 ? UINT_MAX : depth);
		tle->resorigcol = resno;
		context->tlist_dev = lappend(context->tlist_dev, tle);

		return false;
	}
	return expression_tree_walker(node, __pgstrom_build_tlist_dev_walker, __priv);
}

static void
__build_explain_tlist_junks(build_tlist_dev_context *context)
{
	PlannerInfo *root = context->root;
	ListCell   *cell;

	foreach (cell, context->input_rels_tlist)
	{
		Node   *node = lfirst(cell);

		if (IsA(node, Integer))
		{
			Index		relid = intVal(node);
			RelOptInfo *baserel = root->simple_rel_array[relid];
			RangeTblEntry *rte = root->simple_rte_array[relid];

			Assert(IS_SIMPLE_REL(baserel) && rte->rtekind == RTE_RELATION);
			for (int j=baserel->min_attr; j <= baserel->max_attr; j++)
			{
				Form_pg_attribute attr;
				HeapTuple	htup;
				Var		   *var;
				ListCell   *lc;

				if (bms_is_empty(baserel->attr_needed[j-baserel->min_attr]))
					continue;
				htup = SearchSysCache2(ATTNUM,
									   ObjectIdGetDatum(rte->relid),
									   Int16GetDatum(j));
				if (!HeapTupleIsValid(htup))
					elog(ERROR, "cache lookup failed for attribute %d of relation %u",
						 j, rte->relid);
				attr = (Form_pg_attribute) GETSTRUCT(htup);
				var = makeVar(baserel->relid,
							  attr->attnum,
							  attr->atttypid,
							  attr->atttypmod,
							  attr->attcollation,
							  0);

				foreach (lc, context->tlist_dev)
				{
					TargetEntry *tle = lfirst(lc);

					if (equal(tle->expr, var))
						break;
				}
				if (lc)
				{
					/* found */
					pfree(var);
				}
				else
				{
					/* not found, append a junk */
					TargetEntry *tle
						= makeTargetEntry((Expr *)var,
										  list_length(context->tlist_dev)+1,
										  pstrdup(NameStr(attr->attname)),
										  true);
					context->tlist_dev = lappend(context->tlist_dev, tle);
				}
				ReleaseSysCache(htup);
			}
		}
		else if (IsA(node, PathTarget))
		{
			PathTarget *target = (PathTarget *)node;
			ListCell   *lc1, *lc2;

			foreach (lc1, target->exprs)
			{
				Node   *curr = lfirst(lc1);

				foreach (lc2, context->tlist_dev)
				{
					TargetEntry *tle = lfirst(lc2);

					if (equal(tle->expr, curr))
						break;
				}
				if (!lc2)
				{
					/* not found, append a junk */
					TargetEntry *tle
						=  makeTargetEntry((Expr *)curr,
										   list_length(context->tlist_dev)+1,
										   NULL,
										   true);
					context->tlist_dev = lappend(context->tlist_dev, tle);
				}
			}
		}
		else
		{
			elog(ERROR, "Bug? invalid item in the input_rels_tlist: %s",
				 nodeToString(node));
		}
	}
}

static List *
pgstrom_build_tlist_dev(PlannerInfo *root,
						PathTarget *reltarget,
						uint32_t xpu_task_flags,
						List *tlist,		/* must be backed to CPU */
						List *host_quals,	/* must be backed to CPU */
						List *input_rels_tlist)
{
	build_tlist_dev_context context;
	ListCell   *lc;

	memset(&context, 0, sizeof(build_tlist_dev_context));
	context.root = root;
	context.input_rels_tlist = input_rels_tlist;
	context.xpu_task_flags = xpu_task_flags;
	if (tlist != NIL)
	{
		foreach (lc, tlist)
		{
			TargetEntry *tle = lfirst(lc);

			if (contain_var_clause((Node *)tle->expr))
				__pgstrom_build_tlist_dev_walker((Node *)tle->expr, &context);
		}
	}
	else
	{
		/*
		 * When ProjectionPath is on the CustomPath, it always assigns
		 * the result of build_path_tlist() and calls PlanCustomPath method
		 * with tlist == NIL.
		 * So, if xPU projection wants to make something valuable, we need
		 * to check path-target.
		 * Also don't forget all the Var-nodes to be added must exist at
		 * the custom_scan_tlist because setrefs.c references this list.
		 */
		foreach (lc, reltarget->exprs)
		{
			Node   *node = lfirst(lc);

			if (contain_var_clause(node))
				 __pgstrom_build_tlist_dev_walker(node, &context);
		}
	}
	context.only_vars = true;
	__pgstrom_build_tlist_dev_walker((Node *)host_quals, &context);

	context.resjunk = true;
	__build_explain_tlist_junks(&context);
	return context.tlist_dev;
}

/*
 * pgstrom_build_groupby_dev
 */
static List *
pgstrom_build_groupby_dev(PlannerInfo *root,
						  List *tlist,
						  List *host_quals,
						  List *input_rels_tlist)
{
	build_tlist_dev_context context;
	ListCell   *lc1, *lc2;

	memset(&context, 0, sizeof(build_tlist_dev_context));
	context.root = root;
	context.input_rels_tlist = input_rels_tlist;
	context.tlist_dev = copyObject(tlist);
	context.only_vars = true;
	__pgstrom_build_tlist_dev_walker((Node *)host_quals, &context);
	/* just for explain output */
	context.resjunk = true;
	foreach (lc1, tlist)
	{
		TargetEntry *tle = lfirst(lc1);

		if (IsA(tle->expr, FuncExpr))
		{
			FuncExpr   *f = (FuncExpr *)tle->expr;

			foreach (lc2, f->args)
			{
				Expr   *arg = lfirst(lc2);
				int		resno = list_length(context.tlist_dev) + 1;

				context.tlist_dev =
					lappend(context.tlist_dev,
							makeTargetEntry(arg, resno, NULL, true));
			}
		}
	}
	__build_explain_tlist_junks(&context);
	return context.tlist_dev;
}

/*
 * PlanXpuJoinPathCommon
 */
CustomScan *
PlanXpuJoinPathCommon(PlannerInfo *root,
					  RelOptInfo *joinrel,
					  CustomPath *cpath,
					  List *tlist,
					  List *custom_plans,
					  pgstromPlanInfo *pp_info,
					  const CustomScanMethods *xpujoin_plan_methods)
{
	codegen_context context;
	CustomScan *cscan;
	Bitmapset  *outer_refs = NULL;
	List	   *join_quals_stacked = NIL;
	List	   *other_quals_stacked = NIL;
	List	   *hash_keys_stacked = NIL;
	List	   *gist_quals_stacked = NIL;
	List	   *input_rels_tlist;
	List	   *fallback_tlist = NIL;
	ListCell   *lc;
	
	Assert(pp_info->num_rels == list_length(custom_plans));
	codegen_context_init(&context, pp_info->xpu_task_flags);
	input_rels_tlist = list_make1(makeInteger(pp_info->scan_relid));
	foreach (lc, cpath->custom_paths)
	{
		Path	   *__ipath = lfirst(lc);
		input_rels_tlist = lappend(input_rels_tlist, __ipath->pathtarget);
	}
	context.input_rels_tlist = input_rels_tlist;

	/* codegen for outer scan, if any */
	if (pp_info->scan_quals)
	{
		pp_info->kexp_scan_quals
			= codegen_build_scan_quals(&context, pp_info->scan_quals);
		pp_info->scan_quals_fallback
			= build_fallback_exprs_scan(pp_info->scan_relid,
										pp_info->scan_quals);
		pull_varattnos((Node *)pp_info->scan_quals,
					   pp_info->scan_relid,
					   &outer_refs);
	}

	/*
	 * codegen for hashing, join-quals, and gist-quals
	 */
	for (int i=0; i < pp_info->num_rels; i++)
	{
		pgstromPlanInnerInfo *pp_inner = &pp_info->inners[i];

		/* xpu code to generate outer hash-value */
		if (pp_inner->hash_outer_keys != NIL &&
			pp_inner->hash_inner_keys != NIL)
		{
			hash_keys_stacked = lappend(hash_keys_stacked,
										pp_inner->hash_outer_keys);
			pull_varattnos((Node *)pp_inner->hash_outer_keys,
						   pp_info->scan_relid,
						   &outer_refs);
		}
		else
		{
			Assert(pp_inner->hash_outer_keys == NIL &&
				   pp_inner->hash_inner_keys == NIL);
			hash_keys_stacked = lappend(hash_keys_stacked, NIL);
		}
		
		/* xpu code to evaluate join qualifiers */
		join_quals_stacked = lappend(join_quals_stacked, pp_inner->join_quals);
		pull_varattnos((Node *)pp_inner->join_quals,
					   pp_info->scan_relid,
					   &outer_refs);
		other_quals_stacked = lappend(other_quals_stacked, pp_inner->other_quals);
		pull_varattnos((Node *)pp_inner->other_quals,
					   pp_info->scan_relid,
					   &outer_refs);

		/* xpu code to evaluate gist qualifiers */
		gist_quals_stacked = lappend(gist_quals_stacked, pp_inner->gist_clause);
		pull_varattnos((Node *)pp_inner->gist_clause,
					   pp_info->scan_relid,
					   &outer_refs);
	}

	/*
	 * final depth shall be device projection (Scan/Join) or partial
	 * aggregation (GroupBy).
	 */
	if ((pp_info->xpu_task_flags & DEVTASK__MASK) == DEVTASK__PREAGG)
	{
		context.tlist_dev =  pgstrom_build_groupby_dev(root,
													   tlist,
													   NIL,
													   input_rels_tlist);
		codegen_build_groupby_actions(&context, pp_info);
	}
	else
	{
		/* build device projection */
		context.tlist_dev = pgstrom_build_tlist_dev(root,
													joinrel->reltarget,
													pp_info->xpu_task_flags,
													tlist,
													NIL,
													input_rels_tlist);
		pp_info->kexp_projection = codegen_build_projection(&context);
	}
	pull_varattnos((Node *)context.tlist_dev,
				   pp_info->scan_relid,
				   &outer_refs);
	/* assign remaining PlanInfo members */
	pp_info->kexp_join_quals_packed
		= codegen_build_packed_joinquals(&context,
										 join_quals_stacked,
										 other_quals_stacked);
	pp_info->kexp_hash_keys_packed
		= codegen_build_packed_hashkeys(&context,
										hash_keys_stacked);
	pp_info->kexp_scan_kvars_load = codegen_build_scan_loadvars(&context);
	pp_info->kexp_join_kvars_load_packed = codegen_build_join_loadvars(&context);
	codegen_build_packed_gistevals(&context, pp_info);
	pp_info->kvars_depth  = context.kvars_depth;
	pp_info->kvars_resno  = context.kvars_resno;
	pp_info->kvars_types  = context.kvars_types;
	pp_info->kvars_exprs  = context.kvars_exprs;
	pp_info->extra_flags  = context.extra_flags;
	pp_info->extra_bufsz  = context.extra_bufsz;
	pp_info->used_params  = context.used_params;
	pp_info->outer_refs   = outer_refs;
	/*
	 * fixup fallback expressions
	 */
	for (int i=0; i < pp_info->num_rels; i++)
	{
		pgstromPlanInnerInfo *pp_inner = &pp_info->inners[i];

		pp_inner->hash_outer_keys_fallback
			= build_fallback_exprs_join(&context, pp_inner->hash_outer_keys);
		pp_inner->hash_inner_keys_fallback
			= build_fallback_exprs_inner(&context, i+1, pp_inner->hash_inner_keys);
		pp_inner->join_quals_fallback
			= build_fallback_exprs_join(&context, pp_inner->join_quals);
		pp_inner->other_quals_fallback
			= build_fallback_exprs_join(&context, pp_inner->other_quals);
	}
	foreach (lc, context.tlist_dev)
	{
		TargetEntry *tle = lfirst(lc);

		if (tle->resjunk)
			continue;
		tle = makeTargetEntry(tle->expr,
							  list_length(fallback_tlist) + 1,
							  tle->resname,
							  false);
		fallback_tlist = lappend(fallback_tlist, tle);
	}
	pp_info->fallback_tlist =
		(pp_info->num_rels == 0
		 ? build_fallback_exprs_scan(pp_info->scan_relid, fallback_tlist)
		 : build_fallback_exprs_join(&context, fallback_tlist));

	cscan = makeNode(CustomScan);
	cscan->scan.plan.targetlist = tlist;
	cscan->scan.scanrelid = pp_info->scan_relid;
	cscan->flags = cpath->flags;
	cscan->custom_plans = custom_plans;
	cscan->custom_scan_tlist = context.tlist_dev;
	cscan->methods = xpujoin_plan_methods;

	return cscan;
}

/*
 * PlanGpuJoinPath
 */
static Plan *
PlanGpuJoinPath(PlannerInfo *root,
				RelOptInfo *joinrel,
				CustomPath *cpath,
				List *tlist,
				List *clauses,
				List *custom_plans)
{
	pgstromPlanInfo	*pp_info = linitial(cpath->custom_private);
	CustomScan	   *cscan;

	cscan = PlanXpuJoinPathCommon(root,
								  joinrel,
								  cpath,
								  tlist,
								  custom_plans,
								  pp_info,
								  &gpujoin_plan_methods);
	form_pgstrom_plan_info(cscan, pp_info);
	return &cscan->scan.plan;
}

/* ----------------------------------------------------------------
 *
 * Executor Routines
 *
 * ----------------------------------------------------------------
 */

/*
 * CreateGpuJoinState
 */
static Node *
CreateGpuJoinState(CustomScan *cscan)
{
	pgstromTaskState *pts;
	pgstromPlanInfo  *pp_info = deform_pgstrom_plan_info(cscan);
	int			num_rels = list_length(cscan->custom_plans);

	Assert(cscan->methods == &gpujoin_plan_methods);
	pts = palloc0(offsetof(pgstromTaskState, inners[num_rels]));
	NodeSetTag(pts, T_CustomScanState);
	pts->css.flags = cscan->flags;
	pts->css.methods = &gpujoin_exec_methods;
	pts->xpu_task_flags = pp_info->xpu_task_flags;
	pts->pp_info = pp_info;
	Assert((pts->xpu_task_flags & TASK_KIND__MASK) == TASK_KIND__GPUJOIN &&
		   pp_info->num_rels == num_rels);
	pts->num_rels = num_rels;

	return (Node *)pts;
}

/* ---------------------------------------------------------------- *
 *
 * Routines for inner-preloading
 *
 * ---------------------------------------------------------------- *
 */
static uint32_t
get_tuple_hashvalue(pgstromTaskInnerState *istate,
					TupleTableSlot *slot)
{
	ExprContext *econtext = istate->econtext;
	uint32_t	hash = 0xffffffffU;
	ListCell   *lc1, *lc2;

	/* calculation of a hash value of this entry */
	econtext->ecxt_innertuple = slot;
	forboth (lc1, istate->hash_inner_keys,
			 lc2, istate->hash_inner_dtypes)
	{
		ExprState	   *es = lfirst(lc1);
		devtype_info   *dtype = lfirst(lc2);
		Datum			datum;
		bool			isnull;

		datum = ExecEvalExpr(es, econtext, &isnull);
		hash ^= dtype->type_hashfunc(isnull, datum);
	}
	hash ^= 0xffffffffU;

	return hash;
}

/*
 * execInnerPreloadOneDepth
 */
static void
execInnerPreloadOneDepth(pgstromTaskInnerState *istate,
						 MemoryContext memcxt)
{
	PlanState	   *ps = istate->ps;
	MemoryContext	oldcxt;

	for (;;)
	{
		TupleTableSlot *slot;
		TupleDesc		tupdesc;
		HeapTuple		htup;

		CHECK_FOR_INTERRUPTS();

		slot = ExecProcNode(ps);
		if (TupIsNull(slot))
			break;

		/*
		 * NOTE: If varlena datum is compressed / toasted, obviously,
		 * GPU kernel cannot handle operators that reference these
		 * values. Even though we cannot prevent this kind of datum
		 * in OUTER side, we can fix up preloaded values on INNER-side.
		 */
		slot_getallattrs(slot);
		tupdesc = slot->tts_tupleDescriptor;
		for (int j=0; j < tupdesc->natts; j++)
		{
			Form_pg_attribute attr = TupleDescAttr(tupdesc, j);

			if (!slot->tts_isnull[j] && attr->attlen == -1)
				slot->tts_values[j] = (Datum)PG_DETOAST_DATUM(slot->tts_values[j]);
		}
		oldcxt = MemoryContextSwitchTo(memcxt);
		htup = heap_form_tuple(slot->tts_tupleDescriptor,
							   slot->tts_values,
							   slot->tts_isnull);
		if (istate->hash_inner_keys != NIL)
		{
			uint32_t	hash = get_tuple_hashvalue(istate, slot);

			istate->preload_tuples = lappend(istate->preload_tuples, htup);
			istate->preload_hashes = lappend_int(istate->preload_hashes, hash);
			istate->preload_usage += MAXALIGN(offsetof(kern_hashitem,
													   t.htup) + htup->t_len);
		}
		else if (istate->gist_irel)
		{
			uint32_t	hash;

			hash = hash_any((unsigned char *)&slot->tts_tid, sizeof(ItemPointerData));
			istate->preload_tuples = lappend(istate->preload_tuples, htup);
			istate->preload_hashes = lappend_int(istate->preload_hashes, hash);
			istate->preload_usage += MAXALIGN(offsetof(kern_hashitem,
													   t.htup) + htup->t_len);
		}
		else
		{
			istate->preload_tuples = lappend(istate->preload_tuples, htup);
			istate->preload_usage += MAXALIGN(offsetof(kern_tupitem,
													   htup) + htup->t_len);
		}
		MemoryContextSwitchTo(oldcxt);
	}
}

/*
 * innerPreloadAllocHostBuffer
 *
 * NOTE: This function is called with preload_mutex locked
 */
static void
innerPreloadAllocHostBuffer(pgstromTaskState *pts)
{
	pgstromSharedState *ps_state = pts->ps_state;
	kern_multirels	   *h_kmrels = NULL;
	kern_data_store	   *kds = NULL;
	size_t				offset;

	/* other backend already setup the buffer metadata */
	if (ps_state->preload_shmem_length > 0)
		return;
	
	/*
	 * 1st pass: calculation of the buffer length
	 * 2nd pass: initialization of buffer metadata
	 */
again:
	offset = MAXALIGN(offsetof(kern_multirels, chunks[pts->num_rels]));
	for (int i=0; i < pts->num_rels; i++)
	{
		pgstromTaskInnerState *istate = &pts->inners[i];
		TupleDesc	tupdesc = istate->ps->ps_ResultTupleDesc;
		uint64_t	nrooms;
		uint64_t	usage;
		size_t		nbytes;

		nrooms = pg_atomic_read_u64(&ps_state->inners[i].inner_nitems);
		usage  = pg_atomic_read_u64(&ps_state->inners[i].inner_usage);
		if (h_kmrels)
		{
			kds = (kern_data_store *)((char *)h_kmrels + offset);
			h_kmrels->chunks[i].kds_offset = offset;
		}

		nbytes = estimate_kern_data_store(tupdesc);
		if (istate->hash_inner_keys != NIL &&
			istate->hash_outer_keys != NIL)
		{
			/* Hash-Join */
			uint32_t	nslots = Max(320, nrooms);

			nbytes += (MAXALIGN(sizeof(uint32_t) * nrooms) +
					   MAXALIGN(sizeof(uint32_t) * nslots) +
					   MAXALIGN(usage));
			if (h_kmrels)
			{
				setup_kern_data_store(kds, tupdesc, nbytes,
									  KDS_FORMAT_HASH);
				kds->hash_nslots = nslots;
				memset(KDS_GET_HASHSLOT_BASE(kds), 0, sizeof(uint32_t) * nslots);
			}
			offset += nbytes;
		}
		else if (istate->gist_irel != NULL)
		{
			/* GiST-Join */
			Relation	i_rel = istate->gist_irel;
			TupleDesc	i_tupdesc = RelationGetDescr(i_rel);
			BlockNumber	nblocks = RelationGetNumberOfBlocks(i_rel);
			uint32_t	block_offset;
			uint32_t	nslots = Max(320, nrooms);

			/* 1st part - inner tuples indexed by ctid */
			nbytes += (MAXALIGN(sizeof(uint32_t) * nrooms) +
					   MAXALIGN(sizeof(uint32_t) * nslots) +
					   MAXALIGN(usage));
			if (h_kmrels)
			{
				setup_kern_data_store(kds, tupdesc, nbytes,
									  KDS_FORMAT_HASH);
				kds->hash_nslots = nslots;
			}
			offset += nbytes;

			/* 2nd part - GiST index blocks */
			Assert(i_rel->rd_amhandler == F_GISTHANDLER);
			block_offset = (estimate_kern_data_store(i_tupdesc) +
							MAXALIGN(sizeof(uint32_t) * nblocks));
			if (h_kmrels)
			{
				kds = (kern_data_store *)((char *)h_kmrels + offset);
				h_kmrels->chunks[i].gist_offset = offset;

				setup_kern_data_store(kds, i_tupdesc, nbytes,
									  KDS_FORMAT_BLOCK);
				kds->block_offset = block_offset;
				for (int k=0; k < nblocks; k++)
				{
					Buffer		buffer;
					Page		page;
					PageHeader	hpage;

					buffer = ReadBuffer(i_rel, k);
					LockBuffer(buffer, BUFFER_LOCK_SHARE);
					page = BufferGetPage(buffer);
					hpage = KDS_BLOCK_PGPAGE(kds, k);

					memcpy(hpage, page, BLCKSZ);
					hpage->pd_lsn.xlogid = InvalidBlockNumber;
					hpage->pd_lsn.xrecoff = InvalidOffsetNumber;
					KDS_BLOCK_BLCKNR(kds, k) = k;

					UnlockReleaseBuffer(buffer);
				}
				kds->block_nloaded = nblocks;
			}
			offset += (block_offset + BLCKSZ * nblocks);
		}
		else
		{
			/* Nested-Loop */
			nbytes += (MAXALIGN(sizeof(uint32_t) * nrooms) +
					   MAXALIGN(usage));
			if (h_kmrels)
			{
				setup_kern_data_store(kds, tupdesc, nbytes,
									  KDS_FORMAT_ROW);
				h_kmrels->chunks[i].is_nestloop = true;
			}
			offset += nbytes;
		}

		if (istate->join_type == JOIN_RIGHT ||
			istate->join_type == JOIN_FULL)
		{
			nbytes = MAXALIGN(sizeof(bool) * nrooms);
			if (h_kmrels)
			{
				h_kmrels->chunks[i].right_outer = true;
				h_kmrels->chunks[i].ojmap_offset = offset;
				memset((char *)h_kmrels + offset, 0, nbytes);
			}
			offset += nbytes;
		}
		if (istate->join_type == JOIN_LEFT ||
			istate->join_type == JOIN_FULL)
		{
			if (h_kmrels)
				h_kmrels->chunks[i].left_outer = true;
		}
	}

	/*
	 * allocation of the host inner-buffer
	 */
	if (!h_kmrels)
	{
		size_t		shmem_length = PAGE_ALIGN(offset);

		Assert(ps_state->preload_shmem_handle != 0);
		h_kmrels = __mmapShmem(ps_state->preload_shmem_handle,
							   shmem_length, pts->ds_entry);
		memset(h_kmrels, 0, offsetof(kern_multirels,
									 chunks[pts->num_rels]));
		h_kmrels->length = offset;
		h_kmrels->num_rels = pts->num_rels;
		ps_state->preload_shmem_length = shmem_length;
		goto again;
	}
	pts->h_kmrels = h_kmrels;
}

/*
 * __innerPreloadSetupHeapBuffer
 */
static void
__innerPreloadSetupHeapBuffer(kern_data_store *kds,
							  pgstromTaskInnerState *istate,
							  uint32_t base_nitems,
							  uint32_t base_usage)
{
	uint32_t   *row_index = KDS_GET_ROWINDEX(kds);
	uint32_t	rowid = base_nitems;
	char	   *tail_pos = (char *)kds + kds->length;
	char	   *curr_pos = tail_pos - __kds_unpack(base_usage);
	ListCell   *lc;

	Assert(istate->preload_hashes == NIL);
	foreach (lc, istate->preload_tuples)
	{
		HeapTuple	htup = lfirst(lc);
		size_t		sz;
		kern_tupitem *titem;

		sz = MAXALIGN(offsetof(kern_tupitem, htup) + htup->t_len);
		curr_pos -= sz;
		titem = (kern_tupitem *)curr_pos;
		titem->t_len = htup->t_len;
		titem->rowid = rowid;
		memcpy(&titem->htup, htup->t_data, htup->t_len);

		row_index[rowid++] = __kds_packed(tail_pos - curr_pos);
	}
}

/*
 * __innerPreloadSetupHashBuffer
 */
static void
__innerPreloadSetupHashBuffer(kern_data_store *kds,
							  pgstromTaskInnerState *istate,
							  uint32_t base_nitems,
							  uint32_t base_usage)
{
	uint32_t   *row_index = KDS_GET_ROWINDEX(kds);
	uint32_t   *hash_slot = KDS_GET_HASHSLOT_BASE(kds);
	uint32_t	rowid = base_nitems;
	char	   *tail_pos = (char *)kds + kds->length;
	char	   *curr_pos = tail_pos - __kds_unpack(base_usage);
	ListCell   *lc1, *lc2;

	forboth (lc1, istate->preload_tuples,
			 lc2, istate->preload_hashes)
	{
		HeapTuple	htup = lfirst(lc1);
		uint32_t	hash = lfirst_int(lc2);
		uint32_t	hindex = hash % kds->hash_nslots;
		uint32_t	next, self;
		size_t		sz;
		kern_hashitem *hitem;

		sz = MAXALIGN(offsetof(kern_hashitem, t.htup) + htup->t_len);
		curr_pos -= sz;
		self = __kds_packed(tail_pos - curr_pos);
		__atomic_exchange(&hash_slot[hindex], &self, &next,
						  __ATOMIC_SEQ_CST);
		hitem = (kern_hashitem *)curr_pos;
		hitem->hash = hash;
		hitem->next = next;
		hitem->t.t_len = htup->t_len;
		hitem->t.rowid = rowid;
		memcpy(&hitem->t.htup, htup->t_data, htup->t_len);

		row_index[rowid++] = __kds_packed(tail_pos - (char *)&hitem->t);
	}
}

#define INNER_PHASE__SCAN_RELATIONS		0
#define INNER_PHASE__SETUP_BUFFERS		1
#define INNER_PHASE__GPUJOIN_EXEC		2

uint32_t
GpuJoinInnerPreload(pgstromTaskState *pts)
{
	pgstromTaskState   *leader = pts;
	pgstromSharedState *ps_state;
	MemoryContext		memcxt;

	//pick up leader's ps_state if partitionwise plan
	//if (sibling is exist)
	// leader = pts->sibling->leader;
	ps_state = leader->ps_state;

	/* memory context for temporary store  */
	memcxt = AllocSetContextCreate(CurrentMemoryContext,
								   "inner preloading working memory",
								   ALLOCSET_DEFAULT_SIZES);
	/*
	 * Inner PreLoad State Machine
	 */
	SpinLockAcquire(&ps_state->preload_mutex);
	switch (ps_state->preload_phase)
	{
		case INNER_PHASE__SCAN_RELATIONS:
			ps_state->preload_nr_scanning++;
			SpinLockRelease(&ps_state->preload_mutex);
			/*
			 * Scan inner relations, often in parallel
			 */
			for (int i=0; i < pts->num_rels; i++)
			{
				pgstromTaskInnerState *istate = &leader->inners[i];

				execInnerPreloadOneDepth(istate, memcxt);
				pg_atomic_fetch_add_u64(&ps_state->inners[i].inner_nitems,
										list_length(istate->preload_tuples));
				pg_atomic_fetch_add_u64(&ps_state->inners[i].inner_usage,
										istate->preload_usage);
			}

			/*
			 * Once (parallel) scan completed, no other concurrent
			 * workers will not be able to fetch any records from
			 * the inner relations.
			 * So, 'phase' shall be switched to WAIT_FOR_SCANNING
			 * to prevent other worker try to start inner scan.
			 */
			SpinLockAcquire(&ps_state->preload_mutex);
			if (ps_state->preload_phase == INNER_PHASE__SCAN_RELATIONS)
				ps_state->preload_phase = INNER_PHASE__SETUP_BUFFERS;
			else
				Assert(ps_state->preload_phase ==INNER_PHASE__SETUP_BUFFERS);
			/*
			 * Wake up any other concurrent workers, if current
			 * process is the last guy who tried to scan inner
			 * relations.
			 */
			if (--ps_state->preload_nr_scanning == 0)
				ConditionVariableBroadcast(&ps_state->preload_cond);
			/* Falls through. */
		case INNER_PHASE__SETUP_BUFFERS:
			/*
			 * Wait for completion of other workers that still scan
			 * the inner relations.
			 */
			ps_state->preload_nr_setup++;
			if (ps_state->preload_phase == INNER_PHASE__SCAN_RELATIONS ||
				ps_state->preload_nr_scanning > 0)
			{
				ConditionVariablePrepareToSleep(&ps_state->preload_cond);
				while (ps_state->preload_phase == INNER_PHASE__SCAN_RELATIONS ||
					   ps_state->preload_nr_scanning > 0)
				{
					SpinLockRelease(&ps_state->preload_mutex);
					ConditionVariableSleep(&ps_state->preload_cond,
										   PG_WAIT_EXTENSION);
					SpinLockAcquire(&ps_state->preload_mutex);
				}
				ConditionVariableCancelSleep();
			}

			/*
			 * Allocation of the host inner buffer, if not yet
			 */
			PG_TRY();
			{
				innerPreloadAllocHostBuffer(leader);
			}
			PG_CATCH();
			{
				SpinLockRelease(&ps_state->preload_mutex);
				PG_RE_THROW();
			}
			PG_END_TRY();
			SpinLockRelease(&ps_state->preload_mutex);

			/*
			 * Setup the host inner buffer
			 */
			if (!pts->h_kmrels)
			{
				pts->h_kmrels = __mmapShmem(ps_state->preload_shmem_handle,
											ps_state->preload_shmem_length,
											pts->ds_entry);
			}

			for (int i=0; i < leader->num_rels; i++)
			{
				pgstromTaskInnerState *istate = &leader->inners[i];
				kern_data_store *kds = KERN_MULTIRELS_INNER_KDS(pts->h_kmrels, i);
                uint32_t		base_nitems;
				uint32_t		base_usage;

				SpinLockAcquire(&ps_state->preload_mutex);
				base_nitems  = kds->nitems;
				kds->nitems += list_length(istate->preload_tuples);
				base_usage   = kds->usage;
				kds->usage  += __kds_packed(istate->preload_usage);
				SpinLockRelease(&ps_state->preload_mutex);

				if (kds->format == KDS_FORMAT_ROW)
					__innerPreloadSetupHeapBuffer(kds, istate,
												  base_nitems,
                                                  base_usage);
                else if (kds->format == KDS_FORMAT_HASH)
                    __innerPreloadSetupHashBuffer(kds, istate,
                                                  base_nitems,
                                                  base_usage);
                else
					elog(ERROR, "unexpected inner-KDS format");

				/* reset local buffer */
				istate->preload_tuples = NIL;
				istate->preload_hashes = NIL;
				istate->preload_usage  = 0;
			}

			/*
			 * Wait for completion of the host buffer setup
			 * by other concurrent workers
			 */
			SpinLockAcquire(&ps_state->preload_mutex);
			ps_state->preload_nr_setup--;
			if (ps_state->preload_nr_scanning == 0 &&
				ps_state->preload_nr_setup == 0)
			{
				Assert(ps_state->preload_phase == INNER_PHASE__SETUP_BUFFERS);
				ps_state->preload_phase = INNER_PHASE__GPUJOIN_EXEC;
				ConditionVariableBroadcast(&ps_state->preload_cond);
            }
            else
            {
                ConditionVariablePrepareToSleep(&ps_state->preload_cond);
				while (ps_state->preload_nr_scanning > 0 ||
					   ps_state->preload_nr_setup > 0)
				{
                    SpinLockRelease(&ps_state->preload_mutex);
                    ConditionVariableSleep(&ps_state->preload_cond,
                                           PG_WAIT_EXTENSION);
                    SpinLockAcquire(&ps_state->preload_mutex);
                }
                ConditionVariableCancelSleep();
            }
            /* Falls through. */
        case INNER_PHASE__GPUJOIN_EXEC:
			/*
			 * If some worker processes comes into the inner-preload routine
			 * after all the setup jobs finished, the kern_multirels buffer
			 * is already built. So, all we need to do is just map the host
			 * inner buffer here.
			 */
			if (!pts->h_kmrels)
			{
				pts->h_kmrels = __mmapShmem(ps_state->preload_shmem_handle,
											ps_state->preload_shmem_length,
											pts->ds_entry);
			}

			//TODO: send the shmem handle to the GPU server or DPU server

			break;

		default:
			SpinLockRelease(&ps_state->preload_mutex);
			elog(ERROR, "GpuJoin: unexpected inner buffer phase");
			break;
	}
	SpinLockRelease(&ps_state->preload_mutex);
	/* release working memory */
	MemoryContextDelete(memcxt);
	Assert(pts->h_kmrels != NULL);

	return ps_state->preload_shmem_handle;
}

/*
 * CPU Fallback for JOIN
 */
static void
__execFallbackCpuJoinOneDepth(pgstromTaskState *pts, int depth);

static void
__execFallbackLoadVarsSlot(TupleTableSlot *fallback_slot,
						   TupleTableSlot *input_slot,
						   const kern_expression *kexp_vloads)
{
	Assert(kexp_vloads->opcode == FuncOpCode__LoadVars);
	for (int i=0; i < kexp_vloads->u.load.nloads; i++)
	{
		const kern_vars_defitem *kvdef = &kexp_vloads->u.load.kvars[i];
		int		resno = kvdef->var_resno;
		int		slot_id = kvdef->var_slot_id;

		if (slot_id >= 0 && slot_id < fallback_slot->tts_nvalid)
		{
			if (resno > 0 && resno <= input_slot->tts_nvalid)
			{
				fallback_slot->tts_isnull[slot_id] = input_slot->tts_isnull[resno-1];
				fallback_slot->tts_values[slot_id] = input_slot->tts_values[resno-1];
			}
			else
			{
				fallback_slot->tts_isnull[slot_id] = true;
				fallback_slot->tts_values[slot_id] = 0;
			}
		}
	}
}

static void
__execFallbackCpuNestLoop(pgstromTaskState *pts,
						  kern_data_store *kds_in,
						  bool *oj_map, int depth)
{
	pgstromTaskInnerState *istate = &pts->inners[depth-1];
	pgstromPlanInfo *pp_info = pts->pp_info;
	ExprContext    *econtext = pts->css.ss.ps.ps_ExprContext;
	TupleTableSlot *i_slot = istate->ps->ps_ResultTupleSlot;
	kern_expression *kexp_join_kvars_load = NULL;

	if (pp_info->kexp_join_kvars_load_packed)
	{
		const kern_expression *temp = (const kern_expression *)
			VARDATA(pp_info->kexp_join_kvars_load_packed);
		kexp_join_kvars_load = __PICKUP_PACKED_KEXP(temp, depth-1);
	}
	Assert(kds_in->format == KDS_FORMAT_ROW);

	for (uint32_t index=0; index < kds_in->nitems; index++)
	{
		kern_tupitem   *tupitem = KDS_GET_TUPITEM(kds_in, index);

		if (!tupitem)
			continue;
		ResetExprContext(econtext);
		/* load inner variable */
		if (kexp_join_kvars_load)
		{
			HeapTupleData	tuple;

			tuple.t_len = tupitem->t_len;
			ItemPointerSetInvalid(&tuple.t_self);
			tuple.t_tableOid = InvalidOid;
			tuple.t_data = &tupitem->htup;
			ExecForceStoreHeapTuple(&tuple, i_slot, false);
			slot_getallattrs(i_slot);
			Assert(kexp_join_kvars_load->u.load.depth == depth);
			__execFallbackLoadVarsSlot(pts->fallback_slot,
									   i_slot,
									   kexp_join_kvars_load);
			ExecClearTuple(i_slot);
		}
		/* check JOIN-clause */
		if (istate->join_quals != NULL ||
			ExecQual(istate->join_quals, econtext))
		{
			if (istate->other_quals != NULL ||
				ExecQual(istate->other_quals, econtext))
			{
				/* Ok, go to the next depth */
				__execFallbackCpuJoinOneDepth(pts, depth+1);
			}
			/* mark outer-join map, if any */
			if (oj_map)
				oj_map[index] = true;
		}
	}
}

static void
__execFallbackCpuHashJoin(pgstromTaskState *pts,
						  kern_data_store *kds_in,
						  bool *oj_map, int depth)
{
	pgstromTaskInnerState *istate = &pts->inners[depth-1];
	pgstromPlanInfo *pp_info = pts->pp_info;
	ExprContext    *econtext = pts->css.ss.ps.ps_ExprContext;
	TupleTableSlot *i_slot = istate->ps->ps_ResultTupleSlot;
	kern_expression *kexp_join_kvars_load = NULL;
	kern_hashitem  *hitem;
	uint32_t		hash;
	ListCell	   *lc1, *lc2;

	if (pp_info->kexp_join_kvars_load_packed)
	{
		const kern_expression *temp = (const kern_expression *)
			VARDATA(pp_info->kexp_join_kvars_load_packed);
		kexp_join_kvars_load = __PICKUP_PACKED_KEXP(temp, depth-1);
	}
	Assert(kds_in->format == KDS_FORMAT_HASH);

	/*
	 * Compute that hash-value
	 */
	hash = 0xffffffffU;
	forboth (lc1, istate->hash_outer_keys,
			 lc2, istate->hash_outer_dtypes)
	{
		ExprState	   *h_key = lfirst(lc1);
		devtype_info   *dtype = lfirst(lc2);
		Datum			datum;
		bool			isnull;

		datum = ExecEvalExprSwitchContext(h_key, econtext, &isnull);
		hash ^= dtype->type_hashfunc(isnull, datum);
	}
	hash ^= 0xffffffffU;

	/*
	 * walks on the hash-join-table
	 */
	for (hitem = KDS_HASH_FIRST_ITEM(kds_in, hash);
		 hitem != NULL;
		 hitem = KDS_HASH_NEXT_ITEM(kds_in, hitem->next))
	{
		if (hitem->hash != hash)
			continue;
		if (kexp_join_kvars_load)
		{
			HeapTupleData	tuple;

			tuple.t_len  = hitem->t.t_len;
			ItemPointerSetInvalid(&tuple.t_self);
			tuple.t_tableOid = InvalidOid;
			tuple.t_data = &hitem->t.htup;
			ExecForceStoreHeapTuple(&tuple, i_slot, false);
			slot_getallattrs(i_slot);
			Assert(kexp_join_kvars_load->u.load.depth == depth);
			__execFallbackLoadVarsSlot(pts->fallback_slot,
									   i_slot,
									   kexp_join_kvars_load);
			ExecClearTuple(i_slot);
		}
		/* check JOIN-clause */
		if (istate->join_quals == NULL ||
			ExecQual(istate->join_quals, econtext))
		{
			if (istate->other_quals == NULL ||
				ExecQual(istate->other_quals, econtext))
			{
				/* Ok, go to the next depth */
				__execFallbackCpuJoinOneDepth(pts, depth+1);
			}
			/* mark outer-join map, if any */
			if (oj_map)
				oj_map[hitem->t.rowid] = true;
		}
	}
}

static void
__execFallbackCpuJoinOneDepth(pgstromTaskState *pts, int depth)
{
	kern_multirels	   *h_kmrels = pts->h_kmrels;
	kern_data_store	   *kds_in;
	bool			   *oj_map;

	if (depth > h_kmrels->num_rels)
	{
		/* apply projection if any */
		HeapTuple		tuple;
		bool			should_free;

		if (pts->fallback_proj)
		{
			TupleTableSlot *proj_slot = ExecProject(pts->fallback_proj);

			tuple = ExecFetchSlotHeapTuple(proj_slot, false, &should_free);
		}
		else
		{
			tuple = ExecFetchSlotHeapTuple(pts->fallback_slot, false, &should_free);
		}
		/* save the tuple on the fallback buffer */
		pgstromStoreFallbackTuple(pts, tuple);
		if (should_free)
			pfree(tuple);
	}
	else
	{
		kds_in = KERN_MULTIRELS_INNER_KDS(h_kmrels, depth-1);
		oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(h_kmrels, depth-1);

		if (h_kmrels->chunks[depth-1].is_nestloop)
		{
			__execFallbackCpuNestLoop(pts, kds_in, oj_map, depth);
		}
		else
		{
			__execFallbackCpuHashJoin(pts, kds_in, oj_map, depth);
		}
	}
}

void
ExecFallbackCpuJoin(pgstromTaskState *pts,
					kern_data_store *kds,
					HeapTuple tuple)
{
	pgstromPlanInfo *pp_info = pts->pp_info;
	ExprContext    *econtext = pts->css.ss.ps.ps_ExprContext;
	TupleTableSlot *base_slot = pts->base_slot;
	TupleTableSlot *fallback_slot = pts->fallback_slot;
	const kern_expression *kexp_scan_kvars_load = NULL;

	ExecForceStoreHeapTuple(tuple, base_slot, false);
	/* check WHERE-clause if any */
	if (pts->base_quals)
	{
		econtext->ecxt_scantuple = base_slot;
		ResetExprContext(econtext);
		if (!ExecQual(pts->base_quals, econtext))
			return;
	}
	slot_getallattrs(base_slot);

	/* Load the base tuple (depth-0) to the fallback slot */
	ExecStoreAllNullTuple(fallback_slot);
	econtext->ecxt_scantuple = fallback_slot;
	if (pp_info->kexp_scan_kvars_load)
		kexp_scan_kvars_load = (const kern_expression *)
			VARDATA(pp_info->kexp_scan_kvars_load);
	Assert(kexp_scan_kvars_load->u.load.depth == 0);
	__execFallbackLoadVarsSlot(fallback_slot, base_slot, kexp_scan_kvars_load);
	/* Run JOIN, if any */
	Assert(pts->h_kmrels);
	__execFallbackCpuJoinOneDepth(pts, 1);
}

static void
__execFallbackCpuJoinRightOuterOneDepth(pgstromTaskState *pts, int depth)
{
	pgstromTaskInnerState *istate = &pts->inners[depth-1];
	pgstromPlanInfo	   *pp_info = pts->pp_info;
	ExprContext		   *econtext = pts->css.ss.ps.ps_ExprContext;
	TupleTableSlot	   *fallback_slot = pts->fallback_slot;
	TupleTableSlot	   *i_slot = istate->ps->ps_ResultTupleSlot;
	kern_expression	   *kexp_join_kvars_load = NULL;
	kern_multirels	   *h_kmrels = pts->h_kmrels;
	kern_data_store	   *kds_in = KERN_MULTIRELS_INNER_KDS(h_kmrels, depth-1);
	bool			   *oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(h_kmrels, depth-1);

	if (pp_info->kexp_join_kvars_load_packed)
	{
		const kern_expression *temp = (const kern_expression *)
			VARDATA(pp_info->kexp_join_kvars_load_packed);
		kexp_join_kvars_load = __PICKUP_PACKED_KEXP(temp, depth-1);
	}
	Assert(oj_map != NULL);

	ExecStoreAllNullTuple(fallback_slot);
	econtext->ecxt_scantuple = fallback_slot;
	for (uint32_t i=0; i < kds_in->nitems; i++)
	{
		if (oj_map[i])
			continue;

		if (kexp_join_kvars_load)
		{
			kern_tupitem   *titem = KDS_GET_TUPITEM(kds_in, i);
			HeapTupleData	tuple;

			if (!titem)
				continue;
			tuple.t_len  = titem->t_len;
			ItemPointerSetInvalid(&tuple.t_self);
			tuple.t_tableOid = InvalidOid;
			tuple.t_data = &titem->htup;
			ExecForceStoreHeapTuple(&tuple, i_slot, false);
			slot_getallattrs(i_slot);
			Assert(kexp_join_kvars_load->u.load.depth == depth);
			__execFallbackLoadVarsSlot(fallback_slot,
									   i_slot,
									   kexp_join_kvars_load);
			ExecClearTuple(i_slot);
		}
		__execFallbackCpuJoinOneDepth(pts, depth+1);
	}
}

void
ExecFallbackCpuJoinRightOuter(pgstromTaskState *pts)
{
	uint32_t	count;

	count = pg_atomic_add_fetch_u32(pts->rjoin_exit_count, 1);
	//TODO: use sibling count if partitioned join
	if (count == 1)
	{
		for (int depth=1; depth <= pts->num_rels; depth++)
		{
			JoinType	join_type = pts->inners[depth-1].join_type;

			if (join_type == JOIN_RIGHT || join_type == JOIN_FULL)
				__execFallbackCpuJoinRightOuterOneDepth(pts, depth);
		}
	}
}

void
ExecFallbackCpuJoinOuterJoinMap(pgstromTaskState *pts, XpuCommand *resp)
{
	kern_multirels *h_kmrels = pts->h_kmrels;
	bool	   *ojmap_resp = (bool *)((char *)resp + resp->u.results.ojmap_offset);

	Assert(resp->u.results.ojmap_offset +
		   resp->u.results.ojmap_length <= resp->length);
	for (int depth=1; depth <= pts->num_rels; depth++)
	{
		kern_data_store *kds_in = KERN_MULTIRELS_INNER_KDS(h_kmrels, depth-1);
		bool   *ojmap_curr = KERN_MULTIRELS_OUTER_JOIN_MAP(h_kmrels, depth-1);

		if (!ojmap_curr)
			continue;

		for (uint32_t i=0; i < kds_in->nitems; i++)
		{
			ojmap_curr[i] |= ojmap_resp[i];
		}
		ojmap_resp += MAXALIGN(sizeof(bool) * kds_in->nitems);
	}
}

/*
 * pgstrom_init_gpu_join
 */
void
pgstrom_init_gpu_join(void)
{
	/* turn on/off gpujoin */
	DefineCustomBoolVariable("pg_strom.enable_gpujoin",
							 "Enables the use of GpuJoin logic",
							 NULL,
							 &pgstrom_enable_gpujoin,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* turn on/off gpuhashjoin */
	DefineCustomBoolVariable("pg_strom.enable_gpuhashjoin",
							 "Enables the use of GpuHashJoin logic",
							 NULL,
							 &pgstrom_enable_gpuhashjoin,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* tuan on/off gpugistindex */
	DefineCustomBoolVariable("pg_strom.enable_gpugistindex",
							 "Enables the use of GpuGistIndex logic",
							 NULL,
							 &pgstrom_enable_gpugistindex,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* setup path methods */
	memset(&gpujoin_path_methods, 0, sizeof(CustomPathMethods));
	gpujoin_path_methods.CustomName				= "GpuJoin";
	gpujoin_path_methods.PlanCustomPath			= PlanGpuJoinPath;

	/* setup plan methods */
	memset(&gpujoin_plan_methods, 0, sizeof(CustomScanMethods));
	gpujoin_plan_methods.CustomName				= "GpuJoin";
	gpujoin_plan_methods.CreateCustomScanState  = CreateGpuJoinState;
	RegisterCustomScanMethods(&gpujoin_plan_methods);

	/* setup exec methods */
	memset(&gpujoin_exec_methods, 0, sizeof(CustomExecMethods));
	gpujoin_exec_methods.CustomName				= "GpuJoin";
	gpujoin_exec_methods.BeginCustomScan		= pgstromExecInitTaskState;
	gpujoin_exec_methods.ExecCustomScan			= pgstromExecTaskState;
	gpujoin_exec_methods.EndCustomScan			= pgstromExecEndTaskState;
	gpujoin_exec_methods.ReScanCustomScan		= pgstromExecResetTaskState;
	gpujoin_exec_methods.EstimateDSMCustomScan	= pgstromSharedStateEstimateDSM;
	gpujoin_exec_methods.InitializeDSMCustomScan = pgstromSharedStateInitDSM;
	gpujoin_exec_methods.InitializeWorkerCustomScan = pgstromSharedStateAttachDSM;
	gpujoin_exec_methods.ShutdownCustomScan		= pgstromSharedStateShutdownDSM;
	gpujoin_exec_methods.ExplainCustomScan		= pgstromExplainTaskState;

	/* hook registration */
	set_join_pathlist_next = set_join_pathlist_hook;
	set_join_pathlist_hook = gpujoin_add_custompath;
}
