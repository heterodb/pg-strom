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
 * form_gpujoin_info
 *
 * GpuJoinInfo --> custom_private/custom_exprs
 */
static void
form_gpujoin_info(CustomScan *cscan, GpuJoinInfo *gj_info)
{
	List	   *privs = NIL;
	List	   *exprs = NIL;
	int			endpoint_id;

	privs = lappend(privs, bms_to_pglist(gj_info->gpu_cache_devs));
	privs = lappend(privs, bms_to_pglist(gj_info->gpu_direct_devs));
	endpoint_id = DpuStorageEntryGetEndpointId(gj_info->ds_entry);
	privs = lappend(privs, makeInteger(endpoint_id));
	privs = lappend(privs, __makeByteaConst(gj_info->kern_projs));
	privs = lappend(privs, makeInteger(gj_info->extra_flags));
	privs = lappend(privs, makeInteger(gj_info->extra_bufsz));
	privs = lappend(privs, makeInteger(gj_info->kvars_nslots));
	privs = lappend(privs, bms_to_pglist(gj_info->outer_refs));
	exprs = lappend(exprs, gj_info->used_params);
	privs = lappend(privs, makeInteger(gj_info->scan_relid));
	exprs = lappend(exprs, gj_info->scan_quals);
	privs = lappend(privs, __makeByteaConst(gj_info->kern_scan_quals));
	privs = lappend(privs, __makeFloat(gj_info->scan_tuples));
	privs = lappend(privs, __makeFloat(gj_info->scan_rows));
	privs = lappend(privs, __makeFloat(gj_info->final_cost));
	privs = lappend(privs, makeInteger(gj_info->brin_index_oid));
	exprs = lappend(exprs, gj_info->brin_index_conds);
	exprs = lappend(exprs, gj_info->brin_index_quals);
	privs = lappend(privs, makeInteger(gj_info->num_rels));

	for (int i=0; i < gj_info->num_rels; i++)
	{
		GpuJoinInnerInfo *gj_inner = &gj_info->inners[i];
		List   *__privs = NIL;
		List   *__exprs = NIL;

		__privs = lappend(__privs, makeInteger(gj_inner->join_type));
		__privs = lappend(__privs, __makeFloat(gj_inner->join_nrows));
		__exprs = lappend(__exprs, gj_inner->hash_outer);
		__exprs = lappend(__exprs, gj_inner->hash_inner);
		__exprs = lappend(__exprs, gj_inner->join_quals);
		__privs = lappend(__privs, __makeByteaConst(gj_inner->kern_hash_value));
		__privs = lappend(__privs, __makeByteaConst(gj_inner->kern_join_quals));
		__privs = lappend(__privs, makeInteger(gj_inner->gist_index_oid));
		__privs = lappend(__privs, makeInteger(gj_inner->gist_index_col));
		__exprs = lappend(__exprs, gj_inner->gist_clause);
		__privs = lappend(__privs, __makeFloat(gj_inner->gist_selectivity));

		privs = lappend(privs, __privs);
		exprs = lappend(exprs, __exprs);
	}
	cscan->custom_exprs = exprs;
	cscan->custom_private = privs;
}

/*
 * deform_gpujoin_info
 *
 * custom_private/custom_exprs -> GpuJoinInfo
 */
static GpuJoinInfo *
deform_gpujoin_info(CustomScan *cscan)
{
	GpuJoinInfo	gj_data;
	GpuJoinInfo *gj_info;
	List	   *privs = cscan->custom_private;
	List	   *exprs = cscan->custom_exprs;
	int			pindex = 0;
	int			eindex = 0;
	int			endpoint_id;

	memset(&gj_data, 0, sizeof(GpuJoinInfo));
	gj_data.gpu_cache_devs = bms_from_pglist(list_nth(privs, pindex++));
	gj_data.gpu_direct_devs = bms_from_pglist(list_nth(privs, pindex++));
	endpoint_id = intVal(list_nth(privs, pindex++));
	gj_data.ds_entry = DpuStorageEntryByEndpointId(endpoint_id);
	gj_data.kern_projs = __getByteaConst(list_nth(privs, pindex++));
	gj_data.extra_flags = intVal(list_nth(privs, pindex++));
	gj_data.extra_bufsz = intVal(list_nth(privs, pindex++));
	gj_data.kvars_nslots = intVal(list_nth(privs, pindex++));
	gj_data.outer_refs = bms_from_pglist(list_nth(privs, pindex++));
	gj_data.used_params = list_nth(exprs, eindex++);
	gj_data.scan_relid = intVal(list_nth(privs, pindex++));
	gj_data.scan_quals = list_nth(exprs, eindex++);
	gj_data.kern_scan_quals = __getByteaConst(list_nth(privs, pindex++));
	gj_data.scan_tuples = floatVal(list_nth(privs, pindex++));
	gj_data.scan_rows = floatVal(list_nth(privs, pindex++));
	gj_data.final_cost = floatVal(list_nth(privs, pindex++));
	gj_data.brin_index_oid = intVal(list_nth(privs, pindex++));
	gj_data.brin_index_conds = list_nth(exprs, eindex++);
	gj_data.brin_index_quals = list_nth(exprs, eindex++);
	gj_data.num_rels = intVal(list_nth(privs, pindex++));

	gj_info = palloc0(offsetof(GpuJoinInfo, inners[gj_data.num_rels]));
	memcpy(gj_info, &gj_data, offsetof(GpuJoinInfo, inners));
	for (int i=0; i < gj_info->num_rels; i++)
	{
		GpuJoinInnerInfo *gj_inner = &gj_info->inners[i];
		List   *__privs = list_nth(privs, pindex++);
		List   *__exprs = list_nth(exprs, eindex++);
		int		__pindex = 0;
		int		__eindex = 0;

		gj_inner->join_type = intVal(list_nth(__privs, __pindex++));
		gj_inner->join_nrows = floatVal(list_nth(__privs, __pindex++));
		gj_inner->hash_outer = list_nth(__exprs, __eindex++);
		gj_inner->hash_inner = list_nth(__exprs, __eindex++);
		gj_inner->join_quals = list_nth(__exprs, __eindex++);
		gj_inner->kern_hash_value = __getByteaConst(list_nth(__privs, __pindex++));
		gj_inner->kern_join_quals = __getByteaConst(list_nth(__privs, __pindex++));
		gj_inner->gist_index_oid = intVal(list_nth(__privs, __pindex++));
		gj_inner->gist_index_col = intVal(list_nth(__privs, __pindex++));
		gj_inner->gist_clause = list_nth(__exprs, __eindex++);
		gj_inner->gist_selectivity = floatVal(list_nth(__privs, __pindex++));
	}
	return gj_info;
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
						GpuJoinInnerInfo *gj_inner,
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
	Assert(gj_inner->hash_outer == NIL && gj_inner->hash_inner == NIL);
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
			gj_inner->inner_path = new_path;
		}
	}
#endif
	gj_inner->gist_index_oid = gist_index->indexoid;
	gj_inner->gist_index_col = gist_index_col;
//	gj_inner->gist_ctid_resno = SelfItemPointerAttributeNumber;
	gj_inner->gist_clause = gist_clause;
	gj_inner->gist_selectivity = gist_selectivity;
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
	GpuJoinInfo	   *gj_prev;
	GpuJoinInfo	   *gj_info;
	GpuJoinInnerInfo *gj_inner;
	List		   *join_quals = NIL;
	List		   *hash_outer = NIL;
	List		   *hash_inner = NIL;
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

		gj_prev = alloca(offsetof(GpuJoinInfo, inners));
		memset(gj_prev, 0, offsetof(GpuJoinInfo, inners));

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
		gj_prev->scan_quals = scan_quals;
		gj_prev->scan_relid = outer_rel->relid;
		gj_prev->scan_tuples = outer_rel->tuples;
		gj_prev->scan_rows = outer_rel->rows;
		if (!considerXpuScanPathParams(root,
									   outer_rel,
									   DEVKIND__NVIDIA_GPU,
									   try_parallel_path,
									   gj_prev->scan_quals,
									   NIL,		/* host_quals */
									   &parallel_nworkers,
									   &gj_prev->brin_index_oid,
									   &gj_prev->brin_index_conds,
									   &gj_prev->brin_index_quals,
									   &startup_cost,
									   &run_cost,
									   NULL,
									   &gj_prev->gpu_cache_devs,
									   &gj_prev->gpu_direct_devs,
									   NULL))	/* ds_entry (DPU) */
			return false;
	}
	else if (IsGpuJoinPath(outer_path))
	{
		CustomPath	   *__cpath = (CustomPath *)outer_path;

		gj_prev = linitial(__cpath->custom_private);
		startup_cost = __cpath->path.startup_cost;
		run_cost = (__cpath->path.total_cost -
					__cpath->path.startup_cost - gj_prev ->final_cost);
		inner_paths_list = list_copy(__cpath->custom_paths);
		input_rels_tlist = list_make1(makeInteger(gj_prev->scan_relid));
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
	 * Setup GpuJoinInfo
	 */
	gj_info = palloc0(offsetof(GpuJoinInfo, inners[gj_prev->num_rels+1]));
	memcpy(gj_info, gj_prev, offsetof(GpuJoinInfo, inners[gj_prev->num_rels]));
	gj_info->num_rels = gj_prev->num_rels + 1;
	gj_inner = &gj_info->inners[gj_prev->num_rels];

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

		join_quals = lappend(join_quals, rinfo->clause);
		/*
		 * If processing an outer join, only use its own join clauses
		 * for hashing.  For inner joins we need not be so picky.
		 */
		if (IS_OUTER_JOIN(join_type) && rinfo->is_pushed_down)
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
				hash_outer = lappend(hash_outer, arg1);
				hash_inner = lappend(hash_inner, arg2);
			}
			else if (bms_is_subset(relids1, inner_rel->relids) &&
					 bms_is_subset(relids2, outer_rel->relids))
			{
				hash_inner = lappend(hash_inner, arg1);
				hash_outer = lappend(hash_outer, arg2);
			}
			bms_free(relids1);
			bms_free(relids2);
		}
	}
	gj_inner->join_type = join_type;
	gj_inner->join_nrows = joinrel->rows;
	gj_inner->hash_outer = hash_outer;
	gj_inner->hash_inner = hash_inner;
	gj_inner->join_quals = join_quals;
	if (hash_outer == NIL && hash_inner == NIL)
		try_find_gpu_gist_index(root,
								gj_inner,
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
	if (hash_outer != NIL && hash_inner != NIL)
	{
		/*
		 * GpuHashJoin - It computes hash-value of inner tuples by CPU,
		 * but outer tuples by GPU, then it evaluates join-qualifiers
		 * for each items on inner hash table by GPU.
		 */
		int		num_hashkeys = list_length(hash_outer);

		/* cost to compute inner hash value by CPU */
		startup_cost += cpu_operator_cost * num_hashkeys * inner_path->rows;
		/* cost to comput hash value by GPU */
		run_cost += cpu_operator_cost * xpu_ratio * num_hashkeys * outer_path->rows;
		/* cost to evaluate join qualifiers */
		run_cost += join_quals_cost.per_tuple * xpu_ratio * outer_path->rows;
	}
	else if (OidIsValid(gj_inner->gist_index_oid))
	{
		/*
		 * GpuNestLoop+GiST-Index
		 */
		Node	   *gist_clause = gj_inner->gist_clause;
		double		gist_selectivity = gj_inner->gist_selectivity;
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

	gj_info->final_cost = final_cost;

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
	cpath->custom_private = list_make1(gj_info);

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

	if (!node)
		return false;
	foreach (lc, context->tlist_dev)
	{
		TargetEntry	   *tle = lfirst(lc);

		if (equal(node, tle->expr))
			return false;
	}
	if (IsA(node, Var) ||
		(!context->only_vars &&
		 pgstrom_xpu_expression((Expr *)node,
								context->devkind,
								context->input_rels_tlist,
								NULL)))
	{
		AttrNumber	resno = list_length(context->tlist_dev) + 1;
		TargetEntry *tle;

		tle = makeTargetEntry((Expr *)node,
							  resno,
							  NULL,
							  context->resjunk);
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
					  GpuJoinInfo *gj_info,
					  const CustomScanMethods *xpujoin_plan_methods)
{
	CustomScan *cscan;
	Bitmapset  *outer_refs = NULL;
	List	   *misc_exprs = NIL;
	List	   *input_rels_tlist;
	List	   *tlist_dev;
	ListCell   *lc;

	/* sanity checks */
	Assert(gj_info->num_rels == list_length(custom_plans));
	input_rels_tlist = list_make1(makeInteger(gj_info->scan_relid));
	foreach (lc, cpath->custom_paths)
	{
		Path	   *__ipath = lfirst(lc);
		input_rels_tlist = lappend(input_rels_tlist, __ipath->pathtarget);
	}
	/* codegen for outer scan, if any */
	if (gj_info->scan_quals)
	{
		pgstrom_build_xpucode(&gj_info->kern_scan_quals,
							  (Expr *)gj_info->scan_quals,
							  input_rels_tlist,
							  &gj_info->extra_flags,
							  &gj_info->extra_bufsz,
							  &gj_info->kvars_nslots,
							  &gj_info->used_params);
		pull_varattnos((Node *)gj_info->scan_quals,
					   gj_info->scan_relid,
					   &outer_refs);
		misc_exprs = list_concat(misc_exprs, gj_info->scan_quals);
	}

	/*
	 * codegen for hashing, join-quals, and gist-quals
	 */
	for (int i=0; i < gj_info->num_rels; i++)
	{
		GpuJoinInnerInfo *gj_inner = &gj_info->inners[i];

		/* xpu code to generate outer hash-value */
		if (gj_inner->hash_outer != NIL && gj_inner->hash_inner != NIL)
		{
			codegen_build_hashvalue(&gj_inner->kern_hash_value,
									gj_inner->hash_outer,
									input_rels_tlist,
									&gj_info->extra_flags,
									&gj_info->extra_bufsz,
									&gj_info->kvars_nslots,
									&gj_info->used_params);
			pull_varattnos((Node *)gj_inner->hash_outer,
						   gj_info->scan_relid,
						   &outer_refs);
			misc_exprs = list_concat(misc_exprs, gj_inner->hash_outer);
			misc_exprs = list_concat(misc_exprs, gj_inner->hash_inner);
		}
		else
		{
			Assert(gj_inner->hash_outer == NIL &&
				   gj_inner->hash_inner == NIL);
		}
		
		/* xpu code to evaluate join qualifiers */
		pgstrom_build_xpucode(&gj_inner->kern_join_quals,
							  (Expr *)gj_inner->join_quals,
							  input_rels_tlist,
							  &gj_info->extra_flags,
							  &gj_info->extra_bufsz,
							  &gj_info->kvars_nslots,
							  &gj_info->used_params);
		pull_varattnos((Node *)gj_inner->join_quals,
					   gj_info->scan_relid,
					   &outer_refs);
		misc_exprs = list_concat(misc_exprs, gj_inner->join_quals);

		/* xpu code to evaluate gist qualifiers */
		if (OidIsValid(gj_inner->gist_index_oid))
		{
			Assert(gj_inner->gist_clause != NIL);
			//TODO: XPU code generation
			misc_exprs = lappend(misc_exprs, gj_inner->gist_clause);
		}
	}
	/* build device projection */
	tlist_dev = pgstrom_build_tlist_dev(joinrel,
										tlist,
										NIL,
										misc_exprs,
										input_rels_tlist);
	codegen_build_projection(&gj_info->kern_projs,
							 tlist_dev,
							 input_rels_tlist,
							 &gj_info->extra_flags,
							 &gj_info->extra_bufsz,
							 &gj_info->kvars_nslots,
							 &gj_info->used_params);

	cscan = makeNode(CustomScan);
	cscan->scan.plan.targetlist = tlist;
	cscan->scan.scanrelid = gj_info->scan_relid;
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
	GpuJoinInfo	   *gj_info = linitial(cpath->custom_private);
	CustomScan	   *cscan;

	cscan = PlanXpuJoinPathCommon(root,
								  joinrel,
								  cpath,
								  tlist,
								  custom_plans,
								  gj_info,
								  &gpujoin_plan_methods);
	form_gpujoin_info(cscan, gj_info);
	return &cscan->scan.plan;
}

/* ----------------------------------------------------------------
 *
 * Executor Routines
 *
 * ----------------------------------------------------------------
 */
typedef struct
{
	pgstromTaskState pts;
	GpuJoinInfo	   *gj_info;
	void		   *inner_buffer;
	XpuCommand	   *xcmd_req;
	size_t			xcmd_len;
} GpuJoinState;

/*
 * CreateGpuJoinState
 */
static Node *
CreateGpuJoinState(CustomScan *cscan)
{
	GpuJoinState   *gjs = palloc0(sizeof(GpuJoinState));

    Assert(cscan->methods == &gpujoin_plan_methods);
	/* Set tag and executor callbacks */
	NodeSetTag(gjs, T_CustomScanState);
    gjs->pts.css.flags = cscan->flags;
    gjs->pts.css.methods = &gpujoin_exec_methods;
	gjs->pts.devkind = DEVKIND__NVIDIA_GPU;
	gjs->gj_info = deform_gpujoin_info(cscan);

	return (Node *)gjs;
}

/*
 * ExecInitGpuJoin
 */
static void
ExecInitGpuJoin(CustomScanState *node,
				EState *estate,
				int eflags)
{
	GpuJoinState   *gjs = (GpuJoinState *) node;
	CustomScan	   *cscan = (CustomScan *)node->ss.ps.plan;
	ListCell	   *lc;

	foreach (lc, cscan->custom_plans)
	{
		Plan	   *plan = lfirst(lc);
		PlanState  *ps = ExecInitNode(plan, estate, eflags);

		gjs->pts.css.custom_ps = lappend(gjs->pts.css.custom_ps, ps);
	}
}

/*
 * ExecGpuJoin
 */
static TupleTableSlot *
ExecGpuJoin(CustomScanState *node)
{
	return NULL;
}

/*
 * ExecEndGpuJoin
 */
static void
ExecEndGpuJoin(CustomScanState *node)
{
	GpuJoinState   *gjs = (GpuJoinState *)node;
	ListCell	   *lc;

	foreach (lc, gjs->pts.css.custom_ps)
	{
		PlanState  *ps = lfirst(lc);

		ExecEndNode(ps);
	}
}

/*
 * ExecReScanGpuJoin
 */
static void
ExecReScanGpuJoin(CustomScanState *node)
{}

/*
 * ExecGpuJoinEstimateDSM
 */
static Size
ExecGpuJoinEstimateDSM(CustomScanState *node,
					   ParallelContext *pcxt)
{
	return 0;
}

/*
 * ExecGpuJoinInitDSM
 */
static void
ExecGpuJoinInitDSM(CustomScanState *node,
				   ParallelContext *pcxt,
				   void *coordinate)
{}

/*
 * ExecGpuJoinInitWorker
 */
static void
ExecGpuJoinInitWorker(CustomScanState *node,
					  shm_toc *toc,
					  void *coordinate)
{}

/*
 * ExecShutdownGpuJoin
 */
static void
ExecShutdownGpuJoin(CustomScanState *node)
{}

/*
 * ExplainGpuJoin
 */
static void
ExplainGpuJoin(CustomScanState *node,
			   List *ancestors,
			   ExplainState *es)
{
	GpuJoinState   *gjs = (GpuJoinState *)node;
	GpuJoinInfo	   *gj_info = gjs->gj_info;
	CustomScan	   *cscan = (CustomScan *)node->ss.ps.plan;
	const char	   *devkind = DevKindLabel(gjs->pts.devkind, true);
	List		   *dcontext;
	ListCell	   *lc;
	double			ntuples;
	StringInfoData	buf;

	dcontext = set_deparse_context_plan(es->deparse_cxt,
										node->ss.ps.plan,
										ancestors);
	pgstromExplainScanState(&gjs->pts, es,
							gj_info->scan_quals,
							gj_info->kern_scan_quals,
							cscan->custom_scan_tlist,
							gj_info->kern_projs,
							gj_info->scan_tuples,
							gj_info->scan_rows,
							dcontext);	
	initStringInfo(&buf);
	ntuples = gj_info->scan_rows;
	for (int depth=1; depth <= gj_info->num_rels; depth++)
	{
		GpuJoinInnerInfo *gj_inner = &gj_info->inners[depth-1];
		JoinType	join_type = gj_inner->join_type;
		Node	   *expr;
		char	   *str;
		char		label[100];
		int			off;

		resetStringInfo(&buf);
		if (list_length(gj_inner->join_quals) > 1)
			expr = (Node *)make_andclause(gj_inner->join_quals);
		else
			expr = linitial(gj_inner->join_quals);

		if (gj_inner->hash_outer != NIL &&
			gj_inner->hash_inner != NIL)
		{
			off = snprintf(label, sizeof(label), "%sHash%sJoin-%d",
						   devkind,
						   join_type == JOIN_FULL ? "Full" :
						   join_type == JOIN_LEFT ? "Left" :
						   join_type == JOIN_RIGHT ? "Right" : "",
						   depth);
		}
		else if (OidIsValid(gj_inner->gist_index_oid))
		{
			off = snprintf(label, sizeof(label), "%sGiST%sJoin-%d",
						   devkind,
						   join_type == JOIN_FULL ? "Full" :
						   join_type == JOIN_LEFT ? "Left" :
						   join_type == JOIN_RIGHT ? "Right" : "",
						   depth);
		}
		else
		{
			off = snprintf(label, sizeof(label), "%sNestLook%s-%d",
						   devkind,
						   join_type == JOIN_FULL ? "Full" :
						   join_type == JOIN_LEFT ? "Left" :
						   join_type == JOIN_RIGHT ? "Right" : "",
						   depth);
		}
		str = deparse_expression(expr, dcontext, false, true);
		appendStringInfo(&buf, "%s [rows: %.0f -> %.0f]",
						 str, ntuples, gj_inner->join_nrows);
		ExplainPropertyText(label, buf.data, es);

		if (es->verbose && gj_inner->kern_join_quals)
		{
			snprintf(label+off, sizeof(label)-off, " Code");

			resetStringInfo(&buf);
			pgstrom_explain_xpucode(&buf,
									gj_inner->kern_join_quals,
									&gjs->pts.css,
									es, dcontext);
			ExplainPropertyText(label, buf.data, es);
		}

		if (gj_inner->hash_outer != NIL &&
			gj_inner->hash_inner != NIL)
		{
			off = snprintf(label, sizeof(label), "%sHashKeys-%d", devkind, depth);

			resetStringInfo(&buf);
			appendStringInfo(&buf, "inner [");
			foreach (lc, gj_inner->hash_inner)
			{
				expr = lfirst(lc);
				if (lc != list_head(gj_inner->hash_inner))
					appendStringInfoString(&buf, ", ");
				appendStringInfoString(&buf, deparse_expression(expr,
																dcontext,
																false,
																true));
			}
			appendStringInfo(&buf, "], outer [");
			foreach (lc, gj_inner->hash_outer)
			{
				expr = lfirst(lc);
				if (lc != list_head(gj_inner->hash_outer))
					appendStringInfoString(&buf, ", ");
				appendStringInfoString(&buf, deparse_expression(expr,
																dcontext,
																false,
																true));
			}
			appendStringInfo(&buf, "]");

			ExplainPropertyText(label, buf.data, es);

			if (es->verbose && gj_inner->kern_hash_value)
			{
				snprintf(label+off, sizeof(label)-off, " Code");

				resetStringInfo(&buf);
				pgstrom_explain_xpucode(&buf,
										gj_inner->kern_hash_value,
										&gjs->pts.css,
										es, dcontext);
				ExplainPropertyText(label, buf.data, es);
			}
		}
		else if (OidIsValid(gj_inner->gist_index_oid))
		{
			resetStringInfo(&buf);
			snprintf(label, sizeof(label), "%sGiSTIndex-%d", devkind, depth);

			appendStringInfoString(&buf, deparse_expression(gj_inner->gist_clause,
															dcontext,
															false,
															true));
			appendStringInfo(&buf, " on %s (%s)",
							 get_rel_name(gj_inner->gist_index_oid),
							 get_attname(gj_inner->gist_index_oid,
										 gj_inner->gist_index_col, true));
			ExplainPropertyText(label, buf.data, es);
		}
		ntuples = gj_inner->join_nrows;
	}
	pgstromExplainTaskState(&gjs->pts, es, dcontext);
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
