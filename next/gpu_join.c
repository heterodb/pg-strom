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
static bool					enable_gpunestloop;		/* GUC */
static bool					enable_gpuhashjoin;		/* GUC */
static bool					enable_gpugistindex;	/* GUC */

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

	privs = lappend(privs, bms_to_pglist(pp_info->gpu_cache_devs));
	privs = lappend(privs, bms_to_pglist(pp_info->gpu_direct_devs));
	endpoint_id = DpuStorageEntryGetEndpointId(pp_info->ds_entry);
	privs = lappend(privs, makeInteger(endpoint_id));
	/* plan information */
	privs = lappend(privs, bms_to_pglist(pp_info->outer_refs));
	exprs = lappend(exprs, pp_info->used_params);
	privs = lappend(privs, makeInteger(pp_info->scan_relid));
	exprs = lappend(exprs, pp_info->scan_quals);
	privs = lappend(privs, __makeFloat(pp_info->scan_tuples));
	privs = lappend(privs, __makeFloat(pp_info->scan_rows));
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
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_gist_quals_packed));
	privs = lappend(privs, __makeByteaConst(pp_info->kexp_projection));
	privs = lappend(privs, pp_info->kvars_depth);
	privs = lappend(privs, pp_info->kvars_resno);
	privs = lappend(privs, makeInteger(pp_info->extra_flags));
	privs = lappend(privs, makeInteger(pp_info->extra_bufsz));
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
		__exprs = lappend(__exprs, pp_inner->hash_inner_keys);
		__exprs = lappend(__exprs, pp_inner->join_quals);
		__exprs = lappend(__exprs, pp_inner->other_quals);
		__privs = lappend(__privs, makeInteger(pp_inner->gist_index_oid));
		__privs = lappend(__privs, makeInteger(pp_inner->gist_index_col));
		__exprs = lappend(__exprs, pp_inner->gist_clause);
		__privs = lappend(__privs, __makeFloat(pp_inner->gist_selectivity));

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
	pp_data.gpu_cache_devs = bms_from_pglist(list_nth(privs, pindex++));
	pp_data.gpu_direct_devs = bms_from_pglist(list_nth(privs, pindex++));
	endpoint_id = intVal(list_nth(privs, pindex++));
	pp_data.ds_entry = DpuStorageEntryByEndpointId(endpoint_id);
	/* plan information */
	pp_data.outer_refs = bms_from_pglist(list_nth(privs, pindex++));
	pp_data.used_params = list_nth(exprs, eindex++);
	pp_data.scan_relid = intVal(list_nth(privs, pindex++));
	pp_data.scan_quals = list_nth(exprs, eindex++);
	pp_data.scan_tuples = floatVal(list_nth(privs, pindex++));
	pp_data.scan_rows = floatVal(list_nth(privs, pindex++));
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
	pp_data.kexp_gist_quals_packed = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kexp_projection = __getByteaConst(list_nth(privs, pindex++));
	pp_data.kvars_depth = list_nth(privs, pindex++);
	pp_data.kvars_resno = list_nth(privs, pindex++);
	pp_data.extra_flags = intVal(list_nth(privs, pindex++));
	pp_data.extra_bufsz = intVal(list_nth(privs, pindex++));
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

		pp_inner->join_type = intVal(list_nth(__privs, __pindex++));
		pp_inner->join_nrows = floatVal(list_nth(__privs, __pindex++));
		pp_inner->hash_outer_keys = list_nth(__exprs, __eindex++);
		pp_inner->hash_inner_keys = list_nth(__exprs, __eindex++);
		pp_inner->join_quals = list_nth(__exprs, __eindex++);
		pp_inner->other_quals = list_nth(__exprs, __eindex++);
		pp_inner->gist_index_oid = intVal(list_nth(__privs, __pindex++));
		pp_inner->gist_index_col = intVal(list_nth(__privs, __pindex++));
		pp_inner->gist_clause = list_nth(__exprs, __eindex++);
		pp_inner->gist_selectivity = floatVal(list_nth(__privs, __pindex++));
	}
	return pp_info;
}

/*
 * IsGpuJoinPath
 */
static inline bool
IsGpuJoinPath(const Path *path)
{
	return (IsA(path, CustomPath) &&
			((CustomPath *)path)->methods == &gpujoin_path_methods);
}

/*
 * match_clause_to_gist_index
 */
static Node *
match_clause_to_gist_index(PlannerInfo *root,
						   IndexOptInfo *index,
						   AttrNumber indexcol,
						   List *restrict_clauses,
						   Selectivity *p_selectivity)
{
	//to be implemented later
	return NULL;
}

/*
 * try_find_gpu_gist_index
 */
static void
try_find_gpu_gist_index(PlannerInfo *root,
						pgstromPlanInnerInfo *pp_inner,
						JoinType jointype,
						Path *inner_path,
						List *restrict_clauses)
{
	RelOptInfo	   *inner_rel = inner_path->parent;
	//AttrNumber	gist_ctid_resno = SelfItemPointerAttributeNumber;
	IndexOptInfo   *gist_index = NULL;
	AttrNumber		gist_index_col = InvalidAttrNumber;
	Node		   *gist_clause = NULL;
	Selectivity		gist_selectivity = 1.0;
	ListCell	   *lc;

	if (!enable_gpugistindex)
		return;
	/*
	 * Not only GiST, index should be built on normal relations.
	 * And, IndexOnlyScan may not contain CTID, so not supported.
	 */
	Assert(pp_inner->hash_outer_keys == NIL &&
		   pp_inner->hash_inner_keys == NIL);
	if (!IS_SIMPLE_REL(inner_rel) && inner_path->pathtype != T_IndexOnlyScan)
		return;
	/* see the logic in create_index_paths */
	foreach (lc, inner_rel->indexlist)
	{
		IndexOptInfo *curr_index = (IndexOptInfo *) lfirst(lc);
		int		nkeycolumns = curr_index->nkeycolumns;

		Assert(curr_index->rel == inner_rel);
		/* only GiST index is supported  */
		if (curr_index->relam != GIST_AM_OID)
			continue;
		/* ignore partial indexes that do not match the query. */
		if (curr_index->indpred != NIL && !curr_index->predOK)
			continue;
		for (int indexcol=0; indexcol < nkeycolumns; indexcol++)
		{
			Selectivity curr_selectivity = 1.0;
			Node	   *clause;

			clause = match_clause_to_gist_index(root,
												curr_index,
												indexcol,
												restrict_clauses,
												&curr_selectivity);
			if (clause && (!gist_index || gist_selectivity > curr_selectivity))
			{
				gist_index       = curr_index;
				gist_index_col   = indexcol;
				gist_clause      = clause;
				gist_selectivity = curr_selectivity;
			}
		}
	}
#if 0
	// MEMO: InnerPreload stores inner-tuples as Heap format
	// Is the gist_ctid_resno always SelfItemPointerAttributeNumber, isn't it?
	//
	if (gist_index)
	{
		AttrNumber	resno = 1;

		foreach (lc, inner_path->pathtarget->exprs)
		{
			Var	   *var = (Var *) lfirst(lc);

			if (IsA(var, Var) &&
				var->varno == inner_rel->relid &&
				var->varattno == SelfItemPointerAttributeNumber)
			{
				Assert(var->vartype == TIDOID &&
					   var->vartypmod == -1 &&
					   var->varcollid == InvalidOid);
				gist_ctid_resno = resno;
				break;
			}
			resno++;
		}
		/*
		 * Add projection for the ctid
		 */
		if (!lc)
		{
			Path	   *new_path = pgstrom_copy_pathnode(inner_path);
			PathTarget *new_target = copy_pathtarget(inner_path->pathtarget);
			Var		   *var;

			var = makeVar(inner_rel->relid,
						  SelfItemPointerAttributeNumber,
						  TIDOID, -1, InvalidOid, 0);
			new_target->exprs = lappend(new_target->exprs, var);
			gist_ctid_resno = list_length(new_target->exprs);
			new_path->pathtarget = new_target;
			pp_inner->inner_path = new_path;
		}
	}
#endif
	pp_inner->gist_index_oid = gist_index->indexoid;
	pp_inner->gist_index_col = gist_index_col;
//	pp_inner->gist_ctid_resno = SelfItemPointerAttributeNumber;
	pp_inner->gist_clause = gist_clause;
	pp_inner->gist_selectivity = gist_selectivity;
}

/*
 * try_add_simple_gpujoin_path
 */
static bool
try_add_simple_gpujoin_path(PlannerInfo *root,
							RelOptInfo *joinrel,
							Path *outer_path,
                            Path *inner_path,
                            JoinType join_type,
                            JoinPathExtraData *extra,
                            bool try_parallel_path)
{
	RelOptInfo	   *outer_rel = outer_path->parent;
	RelOptInfo	   *inner_rel = inner_path->parent;
	List		   *inner_paths_list = NIL;
	List		   *restrict_clauses = extra->restrictlist;
	Relids			required_outer;
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
	int				parallel_nworkers = 0;
	Cost			startup_cost = 0.0;
	Cost			run_cost = 0.0;
	Cost			final_cost = 0.0;
	QualCost		join_quals_cost;
	IndexOptInfo   *gist_index = NULL;
	double			xpu_ratio = xpuOperatorCostRatio(DEVKIND__NVIDIA_GPU);
	Cost			xpu_tuple_cost = xpuTupleCost(DEVKIND__NVIDIA_GPU);
	ListCell	   *lc;

	/* sanity checks */
	Assert(join_type == JOIN_INNER || join_type == JOIN_FULL ||
		   join_type == JOIN_LEFT  || join_type == JOIN_RIGHT);
	/*
	 * GpuJoin does not support JOIN in case when either side is parameterized
	 * by the other side.
	 */
	if (bms_overlap(PATH_REQ_OUTER(outer_path), inner_path->parent->relids) ||
		bms_overlap(PATH_REQ_OUTER(inner_path), outer_path->parent->relids))
		return false;

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
	 * Build input_rels_tlist for codegen.c
	 */
	if (IS_SIMPLE_REL(outer_rel))
	{
		List	   *scan_quals = NIL;
		List	   *scan_costs = NIL;
		int			devcost;

		pp_prev = alloca(offsetof(pgstromPlanInfo, inners));
		memset(pp_prev, 0, offsetof(pgstromPlanInfo, inners));

		input_rels_tlist = list_make1(makeInteger(outer_rel->relid));
		foreach (lc, outer_rel->baserestrictinfo)
		{
			RestrictInfo *rinfo = lfirst(lc);

			if (pgstrom_gpu_expression(rinfo->clause,
									   input_rels_tlist,
									   &devcost))
			{
				scan_quals = lappend(scan_quals, rinfo->clause);
				scan_costs = lappend_int(scan_costs, devcost);
			}
			else
				return false;	/* all qualifiers must be device executable */
		}
		if (outer_path->param_info)
		{
			foreach (lc, outer_path->param_info->ppi_clauses)
			{
				RestrictInfo *rinfo = lfirst(lc);

				if (pgstrom_gpu_expression(rinfo->clause,
										   input_rels_tlist,
										   &devcost))
					scan_quals = lappend(scan_quals, rinfo->clause);
				else
					return false;
			}
		}
		sort_device_qualifiers(scan_quals, scan_costs);
		pp_prev->scan_quals = scan_quals;
		pp_prev->scan_relid = outer_rel->relid;
		pp_prev->scan_tuples = outer_rel->tuples;
		pp_prev->scan_rows = outer_rel->rows;
		if (!considerXpuScanPathParams(root,
									   outer_rel,
									   DEVKIND__NVIDIA_GPU,
									   try_parallel_path,
									   pp_prev->scan_quals,
									   NIL,		/* host_quals */
									   &parallel_nworkers,
									   &pp_prev->brin_index_oid,
									   &pp_prev->brin_index_conds,
									   &pp_prev->brin_index_quals,
									   &startup_cost,
									   &run_cost,
									   NULL,
									   &pp_prev->gpu_cache_devs,
									   &pp_prev->gpu_direct_devs,
									   NULL))	/* ds_entry (DPU) */
			return false;
	}
	else if (IsGpuJoinPath(outer_path))
	{
		CustomPath	   *__cpath = (CustomPath *)outer_path;

		pp_prev = linitial(__cpath->custom_private);
		startup_cost = __cpath->path.startup_cost;
		run_cost = (__cpath->path.total_cost -
					__cpath->path.startup_cost - pp_prev ->final_cost);
		inner_paths_list = list_copy(__cpath->custom_paths);
		input_rels_tlist = list_make1(makeInteger(pp_prev->scan_relid));
		foreach (lc, inner_paths_list)
		{
			Path   *__ipath = lfirst(lc);
			input_rels_tlist = lappend(input_rels_tlist, __ipath->pathtarget);
		}
	}
	else
	{
		return false;	/* should not happen */
	}
	input_rels_tlist = lappend(input_rels_tlist, inner_path->pathtarget);

	/*
	 * Setup pgstromPlanInfo
	 */
	pp_info = palloc0(offsetof(pgstromPlanInfo, inners[pp_prev->num_rels+1]));
	memcpy(pp_info, pp_prev, offsetof(pgstromPlanInfo, inners[pp_prev->num_rels]));
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

		if (!pgstrom_gpu_expression(rinfo->clause,
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
	if (hash_outer_keys == NIL && hash_inner_keys == NIL)
		try_find_gpu_gist_index(root,
								pp_inner,
								join_type,
								inner_path,
								restrict_clauses);

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
		run_cost += cpu_operator_cost * xpu_ratio * num_hashkeys * outer_path->rows;
		/* cost to evaluate join qualifiers */
		run_cost += join_quals_cost.per_tuple * xpu_ratio * outer_path->rows;
	}
	else if (OidIsValid(pp_inner->gist_index_oid))
	{
		/*
		 * GpuNestLoop+GiST-Index
		 */
		Node	   *gist_clause = pp_inner->gist_clause;
		double		gist_selectivity = pp_inner->gist_selectivity;
		QualCost	gist_clause_cost;

		/* cost to preload inner heap tuples by CPU */
		startup_cost += cpu_tuple_cost * inner_path->rows;
		/* cost to preload the entire index pages once */
		startup_cost += seq_page_cost * (double)gist_index->pages;
		/* cost to evaluate GiST index by GPU */
		cost_qual_eval_node(&gist_clause_cost, gist_clause, root);
		run_cost += gist_clause_cost.per_tuple * xpu_ratio * outer_path->rows;
		/* cost to evaluate join qualifiers by GPU */
		run_cost += (join_quals_cost.per_tuple * xpu_ratio *
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
	cpath->path.parallel_safe = true;
	cpath->path.parallel_workers = parallel_nworkers;
	cpath->path.pathkeys = NIL;
	cpath->path.rows = joinrel->rows;
	cpath->path.startup_cost = startup_cost;
	cpath->path.total_cost = startup_cost + run_cost + final_cost;
	cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
	cpath->methods = &gpujoin_path_methods;
	cpath->custom_paths = lappend(inner_paths_list, inner_path);
	cpath->custom_private = list_make1(pp_info);

	if (custom_path_remember(root, joinrel, try_parallel_path, cpath))
	{
		if (!try_parallel_path)
			add_path(joinrel, &cpath->path);
		else
			add_partial_path(joinrel, &cpath->path);
	}
	return true;
}

/*
 * gpujoin_add_join_path
 */
static void
gpujoin_add_join_path(PlannerInfo *root,
					  RelOptInfo *joinrel,
					  RelOptInfo *outerrel,
					  RelOptInfo *innerrel,
					  JoinType join_type,
					  JoinPathExtraData *extra)
{
	List	   *outer_pathlist = outerrel->pathlist;
	List	   *inner_pathlist = innerrel->pathlist;
	ListCell   *lc1, *lc2;

	/* calls secondary module if exists */
	if (set_join_pathlist_next)
		set_join_pathlist_next(root,
							   joinrel,
							   outerrel,
							   innerrel,
							   join_type,
							   extra);
	/* quick bailout if PG-Strom/GpuJoin is not enabled */
	if (!pgstrom_enabled || (!enable_gpunestloop &&
							 !enable_gpuhashjoin))
		return;

	/* quick bailout if unsupported join type */
	if (join_type != JOIN_INNER &&
		join_type != JOIN_FULL &&
		join_type != JOIN_RIGHT &&
		join_type != JOIN_LEFT)
		return;
	/*
	 * make a GpuJoin path
	 */
	for (int try_parallel=0; try_parallel < 2; try_parallel++)
	{
		foreach (lc1, outer_pathlist)
		{
			Path	   *outer_path = lfirst(lc1);

			if (bms_overlap(PATH_REQ_OUTER(outer_path),
							innerrel->relids))
				continue;
			if (IS_SIMPLE_REL(outerrel) || IsGpuJoinPath(outer_path))
			{
				foreach (lc2, inner_pathlist)
				{
					Path   *inner_path = lfirst(lc2);

					if (bms_overlap(PATH_REQ_OUTER(inner_path),
									outerrel->relids))
						continue;
					if (try_add_simple_gpujoin_path(root,
													joinrel,
													outer_path,
													inner_path,
													join_type,
													extra,
													try_parallel > 0))
						break;
				}
			}
			break;
		}

		if (!joinrel->consider_parallel)
			break;
		outer_pathlist = outerrel->partial_pathlist;
		inner_pathlist = innerrel->partial_pathlist;
	}
}

/*
 * pgstrom_build_tlist_dev
 */
typedef struct
{
	List	   *tlist_dev;
	List	   *input_rels_tlist;
	uint32_t	devkind;
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
								context->devkind,
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

List *
pgstrom_build_tlist_dev(RelOptInfo *rel,
						List *tlist,		/* must be backed to CPU */
						List *host_quals,	/* must be backed to CPU */
						List *misc_exprs,
						List *input_rels_tlist)
{
	build_tlist_dev_context context;
	ListCell   *lc;

	memset(&context, 0, sizeof(build_tlist_dev_context));
	context.input_rels_tlist = input_rels_tlist;

	if (tlist != NIL)
	{
		foreach (lc, tlist)
		{
			TargetEntry *tle = lfirst(lc);

			if (IsA(tle->expr, Const) || IsA(tle->expr, Param))
				continue;
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
		foreach (lc, rel->reltarget->exprs)
		{
			Node   *node = lfirst(lc);

			if (IsA(node, Const) || IsA(node, Param))
				continue;
			__pgstrom_build_tlist_dev_walker(node, &context);
		}
	}
	context.only_vars = true;
	__pgstrom_build_tlist_dev_walker((Node *)host_quals, &context);

	context.resjunk = true;
	__pgstrom_build_tlist_dev_walker((Node *)misc_exprs, &context);

	return context.tlist_dev;
}

/*
 * PlanXpuJoinPathCommon
 */
static CustomScan *
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
	List	   *misc_exprs = NIL;
	List	   *input_rels_tlist;
	List	   *tlist_dev;
	ListCell   *lc;

	Assert(pp_info->num_rels == list_length(custom_plans));
	memset(&context, 0, sizeof(codegen_context));
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
		pull_varattnos((Node *)pp_info->scan_quals,
					   pp_info->scan_relid,
					   &outer_refs);
		misc_exprs = list_concat(misc_exprs, pp_info->scan_quals);
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
			misc_exprs = list_concat(misc_exprs, pp_inner->hash_outer_keys);
			misc_exprs = list_concat(misc_exprs, pp_inner->hash_inner_keys);
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
		misc_exprs = list_concat(misc_exprs, pp_inner->join_quals);
		misc_exprs = list_concat(misc_exprs, pp_inner->other_quals);

		/* xpu code to evaluate gist qualifiers */
		if (OidIsValid(pp_inner->gist_index_oid))
		{
			Assert(pp_inner->gist_clause != NIL);
			//TODO: XPU code generation
			misc_exprs = lappend(misc_exprs, pp_inner->gist_clause);
		}
		gist_quals_stacked = lappend(gist_quals_stacked, NIL);
	}
	/* build device projection */
	tlist_dev = pgstrom_build_tlist_dev(joinrel,
										tlist,
										NIL,
										misc_exprs,
										input_rels_tlist);
	pp_info->kexp_projection = codegen_build_projection(&context, tlist_dev);
	pp_info->kexp_join_quals_packed
		= codegen_build_packed_joinquals(&context,
										 join_quals_stacked,
										 other_quals_stacked);
	pp_info->kexp_hash_keys_packed
		= codegen_build_packed_hashkeys(&context,
										hash_keys_stacked);
	pp_info->kexp_gist_quals_packed = NULL;
	pp_info->kexp_scan_kvars_load = codegen_build_scan_loadvars(&context);
	pp_info->kexp_join_kvars_load_packed = codegen_build_join_loadvars(&context);
	elog(INFO, "%p %p %p %p %p %p",
		 pp_info->kexp_projection,
		 pp_info->kexp_join_quals_packed,
		 pp_info->kexp_hash_keys_packed,
		 pp_info->kexp_gist_quals_packed,
		 pp_info->kexp_scan_kvars_load,
		 pp_info->kexp_join_kvars_load_packed);

	pp_info->kvars_depth  = context.kvars_depth;
	pp_info->kvars_resno  = context.kvars_resno;
	pp_info->extra_flags  = context.extra_flags;
	pp_info->extra_bufsz  = context.extra_bufsz;
	pp_info->used_params  = context.used_params;
	pp_info->outer_refs   = outer_refs;

	cscan = makeNode(CustomScan);
	cscan->scan.plan.targetlist = tlist;
	cscan->scan.scanrelid = pp_info->scan_relid;
	cscan->flags = cpath->flags;
	cscan->custom_plans = custom_plans;
	cscan->custom_scan_tlist = tlist_dev;
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
#if 0
typedef struct
{
	PlanState	   *ps;
	ExprContext	   *econtext;
	/*
	 * inner preload buffer
	 */
	List		   *preload_tuples;
	List		   *preload_hashes;		/* if hash-join or gist-join */
	size_t			preload_usage;

	/*
	 * join properties (common)
	 */
	int				depth;
	JoinType		join_type;
	ExprState	   *join_quals;
	ExprState	   *other_quals;

	/*
	 * join properties (hash-join)
	 */
	List		   *hash_outer_keys;	/* list of ExprState */
	List		   *hash_inner_keys;	/* list of ExprState */
	List		   *hash_outer_dtypes;	/* list of devtype_info */
	List		   *hash_inner_dtypes;	/* list of devtype_info */

	/*
	 * join properties (gist-join)
	 */
	Relation		gist_irel;
	ExprState	   *gist_clause;
} GpuJoinInnerState;

typedef struct GpuJoinState
{
	pgstromTaskState pts;
	pgstromPlanInfo	*gj_info;
	XpuCommand	   *xcmd_req;
	size_t			xcmd_len;
	/* inner relations */
	int				num_rels;
	GpuJoinInnerState inners[FLEXIBLE_ARRAY_MEMBER];
} GpuJoinState;
#endif

static uint32_t		GpuJoinInnerPreload(pgstromTaskState *pts);

/*
 * fixup_inner_varnode
 *
 * Any var-nodes are rewritten at setrefs.c to indicate a particular item
 * on the cscan->custom_scan_tlist. However, inner expression must reference
 * the inner relation, so we need to fix up it again.
 */
typedef struct
{
	CustomScan *cscan;
	Plan	   *inner_plan;
} fixup_inner_varnode_context;

static Node *
__fixup_inner_varnode_walker(Node *node, void *data)
{
	fixup_inner_varnode_context *con = data;

	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		Var		   *var = (Var *)node;
		List	   *tlist_dev = con->cscan->custom_scan_tlist;
		TargetEntry *tle;

		Assert(var->varno == INDEX_VAR &&
			   var->varattno >= 1 &&
			   var->varattno <= list_length(tlist_dev));
		tle = list_nth(tlist_dev, var->varattno - 1);
		return (Node *)makeVar(INNER_VAR,
							   tle->resorigcol,
							   var->vartype,
							   var->vartypmod,
							   var->varcollid,
							   0);
	}
	return expression_tree_mutator(node, __fixup_inner_varnode_walker, con);
}

static List *
fixup_inner_varnode(List *exprs, CustomScan *cscan, Plan *inner_plan)
{
	fixup_inner_varnode_context con;

	memset(&con, 0, sizeof(con));
	con.cscan = cscan;
	con.inner_plan = inner_plan;

	return (List *)__fixup_inner_varnode_walker((Node *)exprs, &con);
}

/*
 * CreateGpuJoinState
 */
static Node *
CreateGpuJoinState(CustomScan *cscan)
{
	pgstromTaskState *pts;
	int		num_rels = list_length(cscan->custom_plans);

	Assert(cscan->methods == &gpujoin_plan_methods);
	pts = palloc0(offsetof(pgstromTaskState, inners[num_rels]));
	NodeSetTag(pts, T_CustomScanState);
	pts->css.flags = cscan->flags;
	pts->css.methods = &gpujoin_exec_methods;
	pts->devkind = DEVKIND__NVIDIA_GPU;
	pts->pp_info = deform_pgstrom_plan_info(cscan);
	Assert(pts->pp_info->num_rels == num_rels);
	pts->num_rels = num_rels;

	return (Node *)pts;
}

/*
 * ExecInitGpuJoin
 */
static void
ExecInitGpuJoin(CustomScanState *node,
				EState *estate,
				int eflags)
{
	pgstromTaskState *pts = (pgstromTaskState *) node;
	pgstromPlanInfo	*pp_info = pts->pp_info;
	CustomScan	   *cscan = (CustomScan *)node->ss.ps.plan;
	ListCell	   *lc;
	int				depth = 1;

	Assert(node->ss.ss_currentRelation != NULL &&
		   outerPlanState(node) == NULL &&
		   innerPlanState(node) == NULL);
	Assert(pp_info->num_rels == list_length(cscan->custom_plans));
	foreach (lc, cscan->custom_plans)
	{
		pgstromTaskInnerState *istate = &pts->inners[depth-1];
		pgstromPlanInnerInfo *pp_inner = &pp_info->inners[depth-1];
		Plan	   *plan = lfirst(lc);
		PlanState  *ps = ExecInitNode(plan, estate, eflags);
		List	   *hash_inner_keys;
		ListCell   *cell;
		devtype_info *dtype;

		istate->ps = ps;
		istate->econtext = CreateExprContext(estate);
		istate->depth = depth;
		istate->join_type = pp_inner->join_type;
		istate->join_quals = ExecInitQual(pp_inner->join_quals,
										  &pts->css.ss.ps);
		istate->other_quals = ExecInitQual(pp_inner->other_quals,
										   &pts->css.ss.ps);
		foreach (cell, pp_inner->hash_outer_keys)
		{
			ExprState  *es = ExecInitExpr((Expr *)lfirst(cell),
										  &pts->css.ss.ps);
			dtype = pgstrom_devtype_lookup(exprType((Node *)es->expr));
			if (!dtype)
				elog(ERROR, "failed on lookup device type of %s",
					 nodeToString(es->expr));
			istate->hash_outer_keys = lappend(istate->hash_outer_keys, es);
			istate->hash_outer_dtypes = lappend(istate->hash_outer_dtypes, dtype);
		}
		/* inner hash-keys references the result of inner-slot */
		hash_inner_keys = fixup_inner_varnode(pp_inner->hash_inner_keys,
											  cscan, plan);
		foreach (cell, hash_inner_keys)
		{
			ExprState  *es = ExecInitExpr((Expr *)lfirst(cell),
										  &pts->css.ss.ps);
			dtype = pgstrom_devtype_lookup(exprType((Node *)es->expr));
			if (!dtype)
				elog(ERROR, "failed on lookup device type of %s",
					 nodeToString(es->expr));
			istate->hash_inner_keys = lappend(istate->hash_inner_keys, es);
			istate->hash_inner_dtypes = lappend(istate->hash_inner_dtypes, dtype);
		}

		if (OidIsValid(pp_inner->gist_index_oid))
		{
			istate->gist_irel = index_open(pp_inner->gist_index_oid,
										   AccessShareLock);
			istate->gist_clause = ExecInitExpr((Expr *)pp_inner->gist_clause,
											   &pts->css.ss.ps);
		}
		pts->css.custom_ps = lappend(pts->css.custom_ps, ps);
		depth++;
	}
	pgstromExecInitTaskState(pts, DEVKIND__NVIDIA_GPU);
	pts->cb_cpu_fallback = ExecFallbackCpuScan;
}

/*
 * GpuJoinReCheckTuple
 */
static bool
GpuJoinReCheckTuple(pgstromTaskState *pts, TupleTableSlot *epq_slot)
{
	/*
	 * NOTE: Only immutable operators/functions are executable
	 * on the GPU devices, so its decision will never changed.
	 */
	return true;
}

/*
 * ExecGpuJoin
 */
static TupleTableSlot *
ExecGpuJoin(CustomScanState *node)
{
	pgstromTaskState *pts = (pgstromTaskState *) node;

	if (!pts->h_kmrels)
	{
		const XpuCommand *session;
		uint32_t	inner_handle;

		/* attach pgstromSharedState, if none */
		if (!pts->ps_state)
			pgstromSharedStateInitDSM(pts, NULL, NULL);
		/* preload inner buffer */
		inner_handle = GpuJoinInnerPreload(pts);
		if (inner_handle == 0)
			return NULL;
		/* open the GpuJoin session */
		session = pgstromBuildSessionInfo(pts, inner_handle);
		gpuClientOpenSession(pts, pts->optimal_gpus, session);
	}
	return ExecScan(&node->ss,
					(ExecScanAccessMtd) pgstromExecTaskState,
					(ExecScanRecheckMtd) GpuJoinReCheckTuple);
}

/*
 * ExecEndGpuJoin
 */
static void
ExecEndGpuJoin(CustomScanState *node)
{
	pgstromTaskState *pts = (pgstromTaskState *) node;

	pgstromExecEndTaskState(pts);
}

/*
 * ExecReScanGpuJoin
 */
static void
ExecReScanGpuJoin(CustomScanState *node)
{
	pgstromTaskState *pts = (pgstromTaskState *) node;

	pgstromExecResetTaskState(pts);
}

/*
 * ExecGpuJoinEstimateDSM
 */
static Size
ExecGpuJoinEstimateDSM(CustomScanState *node,
					   ParallelContext *pcxt)
{
	pgstromTaskState *pts = (pgstromTaskState *) node;

	return pgstromSharedStateEstimateDSM(pts);
}

/*
 * ExecGpuJoinInitDSM
 */
static void
ExecGpuJoinInitDSM(CustomScanState *node,
				   ParallelContext *pcxt,
				   void *dsm_addr)
{
	pgstromTaskState *pts = (pgstromTaskState *) node;

	pgstromSharedStateInitDSM(pts, pcxt, dsm_addr);
}

/*
 * ExecGpuJoinInitWorker
 */
static void
ExecGpuJoinInitWorker(CustomScanState *node,
					  shm_toc *toc,
					  void *dsm_addr)
{
	pgstromTaskState *pts = (pgstromTaskState *) node;

	pgstromSharedStateAttachDSM(pts, dsm_addr);
}

/*
 * ExecShutdownGpuJoin
 */
static void
ExecShutdownGpuJoin(CustomScanState *node)
{
	pgstromTaskState *pts = (pgstromTaskState *) node;

	pgstromSharedStateShutdownDSM(pts);
}

/*
 * ExplainGpuJoin
 */
static void
ExplainGpuJoin(CustomScanState *node,
			   List *ancestors,
			   ExplainState *es)
{
	pgstromTaskState *pts = (pgstromTaskState *)node;
	List	   *dcontext;

	dcontext = set_deparse_context_plan(es->deparse_cxt,
										node->ss.ps.plan,
										ancestors);
	pgstromTaskStateExplain(pts, es, dcontext, "GPU");
}

/* ---------------------------------------------------------------- *
 *
 * Routines for inner-preloading
 *
 * ---------------------------------------------------------------- *
 */
static uint32_t
get_tuple_hashvalue(pgstromTaskInnerState *istate,
					bool is_inner_hashkeys,
					TupleTableSlot *slot)
{
	ExprContext *econtext = istate->econtext;
	uint32_t	hash = 0xffffffffU;
	List	   *hash_keys_list;
	List	   *hash_dtypes_list;
	ListCell   *lc1, *lc2;

	if (is_inner_hashkeys)
	{
		hash_keys_list = istate->hash_inner_keys;
		hash_dtypes_list = istate->hash_inner_dtypes;
		econtext->ecxt_innertuple = slot;
	}
	else
	{
		hash_keys_list = istate->hash_outer_keys;
		hash_dtypes_list = istate->hash_outer_dtypes;
		econtext->ecxt_scantuple = slot;
	}

	/* calculation of a hash value of this entry */
	forboth (lc1, hash_keys_list,
			 lc2, hash_dtypes_list)
	{
		ExprState	   *es = lfirst(lc1);
		devtype_info   *dtype = lfirst(lc2);
		Datum			datum;
		bool			isnull;

		datum = ExecEvalExpr(es, econtext, &isnull);
		if (isnull)
			continue;
		hash ^= dtype->type_hashfunc(dtype, datum);
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
			uint32_t	hash = get_tuple_hashvalue(istate, true, slot);

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
			}
			memset((char *)h_kmrels + offset, 0, nbytes);
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
							   shmem_length, NULL);
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
	uint32_t   *hash_slot = KDS_GET_HASHSLOT(kds);
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

static uint32_t
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
											NULL);
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
											NULL);
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
 * pgstrom_init_gpu_join
 */
void
pgstrom_init_gpu_join(void)
{
	/* turn on/off gpunestloop */
	DefineCustomBoolVariable("pg_strom.enable_gpunestloop",
							 "Enables the use of GpuNestLoop logic",
							 NULL,
							 &enable_gpunestloop,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* turn on/off gpuhashjoin */
	DefineCustomBoolVariable("pg_strom.enable_gpuhashjoin",
							 "Enables the use of GpuHashJoin logic",
							 NULL,
							 &enable_gpuhashjoin,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* tuan on/off gpugistindex */
	DefineCustomBoolVariable("pg_strom.enable_gpugistindex",
							 "Enables the use of GpuGistIndex logic",
							 NULL,
							 &enable_gpugistindex,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* setup path methods */
	gpujoin_path_methods.CustomName				= "GpuJoin";
	gpujoin_path_methods.PlanCustomPath			= PlanGpuJoinPath;

	/* setup plan methods */
	gpujoin_plan_methods.CustomName				= "GpuJoin";
	gpujoin_plan_methods.CreateCustomScanState  = CreateGpuJoinState;
	RegisterCustomScanMethods(&gpujoin_plan_methods);

	/* setup exec methods */
	gpujoin_exec_methods.CustomName				= "GpuJoin";
	gpujoin_exec_methods.BeginCustomScan		= ExecInitGpuJoin;
	gpujoin_exec_methods.ExecCustomScan			= ExecGpuJoin;
	gpujoin_exec_methods.EndCustomScan			= ExecEndGpuJoin;
	gpujoin_exec_methods.ReScanCustomScan		= ExecReScanGpuJoin;
	gpujoin_exec_methods.MarkPosCustomScan		= NULL;
	gpujoin_exec_methods.RestrPosCustomScan		= NULL;
	gpujoin_exec_methods.EstimateDSMCustomScan	= ExecGpuJoinEstimateDSM;
	gpujoin_exec_methods.InitializeDSMCustomScan = ExecGpuJoinInitDSM;
	gpujoin_exec_methods.InitializeWorkerCustomScan = ExecGpuJoinInitWorker;
	gpujoin_exec_methods.ShutdownCustomScan		= ExecShutdownGpuJoin;
	gpujoin_exec_methods.ExplainCustomScan		= ExplainGpuJoin;

	/* hook registration */
	set_join_pathlist_next = set_join_pathlist_hook;
	set_join_pathlist_hook = gpujoin_add_join_path;
}
