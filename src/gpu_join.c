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
static bool					pgstrom_enable_gpujoin = false;		/* GUC */
static bool					pgstrom_enable_gpuhashjoin = false;	/* GUC */
static bool					pgstrom_enable_gpugistindex = false;/* GUC */
static bool					pgstrom_enable_partitionwise_gpujoin = false;

static CustomPathMethods	dpujoin_path_methods;
static CustomScanMethods	dpujoin_plan_methods;
static CustomExecMethods	dpujoin_exec_methods;
static bool					pgstrom_enable_dpujoin = false;		/* GUC */
static bool					pgstrom_enable_dpuhashjoin = false;	/* GUC */
static bool					pgstrom_enable_dpugistindex = false;/* GUC */
static bool					pgstrom_enable_partitionwise_dpujoin = false;

/*
 * try_fetch_xpujoin_planinfo
 */
pgstromPlanInfo *
try_fetch_xpujoin_planinfo(const Path *path)
{
	const CustomPath *cpath = (const CustomPath *)path;

	if (IsA(cpath, CustomPath) &&
		(cpath->methods == &gpujoin_path_methods ||
		 cpath->methods == &dpujoin_path_methods))
		return (pgstromPlanInfo *)linitial(cpath->custom_private);
	return NULL;
}

/*
 * __buildXpuJoinPlanInfo
 */
static pgstromPlanInfo *
__buildXpuJoinPlanInfo(PlannerInfo *root,
					   RelOptInfo *joinrel,
					   JoinType join_type,
					   List *restrict_clauses,
					   pgstromOuterPathLeafInfo *op_prev,
					   Path **p_inner_path)
{
	pgstromPlanInfo *pp_prev = op_prev->pp_info;
	pgstromPlanInfo *pp_info;
	pgstromPlanInnerInfo *pp_inner;
	Path		   *inner_path = *p_inner_path;
	RelOptInfo	   *inner_rel = inner_path->parent;
	RelOptInfo	   *outer_rel = op_prev->leaf_rel;
	Cardinality		outer_nrows;
	Cost			startup_cost;
	Cost			run_cost;
	bool			enable_xpuhashjoin;
	bool			enable_xpugistindex;
	double			xpu_tuple_cost;
	Cost			xpu_ratio;
	Cost			comp_cost = 0.0;
	Cost			final_cost = 0.0;
	QualCost		join_quals_cost;
	List		   *join_quals = NIL;
	List		   *other_quals = NIL;
	List		   *hash_outer_keys = NIL;
	List		   *hash_inner_keys = NIL;
	List		   *inner_target_list = NIL;
	ListCell	   *lc;
	bool			clauses_are_immutable = true;

	/* cross join is not welcome */
	if (!restrict_clauses)
		return NULL;

	/*
	 * device specific parameters
	 */
	if ((pp_prev->xpu_task_flags & DEVKIND__ANY) == DEVKIND__NVIDIA_GPU)
	{
		enable_xpuhashjoin  = pgstrom_enable_gpuhashjoin;
		enable_xpugistindex = pgstrom_enable_gpugistindex;
		xpu_tuple_cost      = pgstrom_gpu_tuple_cost;
		xpu_ratio           = pgstrom_gpu_operator_ratio();
	}
	else if ((pp_prev->xpu_task_flags & DEVKIND__ANY) == DEVKIND__NVIDIA_DPU)
	{
		enable_xpuhashjoin  = pgstrom_enable_dpuhashjoin;
		enable_xpugistindex = pgstrom_enable_dpugistindex;
		xpu_tuple_cost      = pgstrom_dpu_tuple_cost;
		xpu_ratio           = pgstrom_dpu_operator_ratio();
	}
	else
	{
		elog(ERROR, "Bug? unexpected xpu_task_flags: %08x",
			 pp_prev->xpu_task_flags);
	}

	/* setup inner_targets */
	foreach (lc, op_prev->inner_paths_list)
	{
		Path   *i_path = lfirst(lc);

		inner_target_list = lappend(inner_target_list, i_path->pathtarget);
	}
	inner_target_list = lappend(inner_target_list, inner_path->pathtarget);

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

		/*
		 * In case when neither outer-vars nor inner-vars are not referenced,
		 * this join always works as a cross-join or an empty-join.
		 * They are not welcome for xPU-Join workloads regardless of the cost
		 * estimation.
		 */
		if (contain_var_clause((Node *)rinfo->clause))
			clauses_are_immutable = false;

		/*
		 * Is the JOIN-clause executable on the target device?
		 */
		if (!pgstrom_xpu_expression(rinfo->clause,
									pp_prev->xpu_task_flags,
									pp_prev->scan_relid,
									inner_target_list,
									NULL))
		{
			return NULL;
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

	if (clauses_are_immutable)
	{
		elog(DEBUG2, "immutable join-clause is not supported: %s",
			 nodeToString(restrict_clauses));
		return NULL;
	}

	/*
	 * Setup pgstromPlanInfo
	 */
	pp_info = copy_pgstrom_plan_info(pp_prev);
	pp_inner = &pp_info->inners[pp_info->num_rels++];
	pp_inner->join_type = join_type;
	pp_inner->join_nrows = joinrel->rows;
	pp_inner->hash_outer_keys_original = hash_outer_keys;
	pp_inner->hash_inner_keys_original = hash_inner_keys;
	pp_inner->join_quals_original = join_quals;
	pp_inner->other_quals_original = other_quals;
	/* GiST-Index availability checks */
	if (enable_xpugistindex &&
		pp_inner->hash_outer_keys_original == NIL &&
		pp_inner->hash_inner_keys_original == NIL)
	{
		Path   *gist_inner_path
			= pgstromTryFindGistIndex(root,
									  inner_path,
									  restrict_clauses,
									  pp_info->xpu_task_flags,
									  pp_info->scan_relid,
									  inner_target_list,
									  pp_inner);
		if (gist_inner_path)
			*p_inner_path = inner_path = gist_inner_path;
	}
	/*
	 * Cost estimation
	 */
	if (pp_prev->num_rels == 0)
	{
		outer_nrows  = pp_prev->scan_rows;
		startup_cost = pp_prev->scan_startup_cost;
		run_cost     = pp_prev->scan_run_cost;
	}
	else
	{
		const pgstromPlanInnerInfo *__pp_inner = &pp_prev->inners[pp_prev->num_rels-1];

		outer_nrows  = __pp_inner->join_nrows;
		startup_cost = __pp_inner->join_startup_cost;
		run_cost     = __pp_inner->join_run_cost;
	}
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
					  outer_nrows);
		/* cost to evaluate join qualifiers */
		comp_cost += join_quals_cost.per_tuple * xpu_ratio * outer_nrows;
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
		comp_cost += gist_clause_cost.per_tuple * xpu_ratio * outer_nrows;
		/* cost to evaluate join qualifiers by GPU */
		comp_cost += (join_quals_cost.per_tuple * xpu_ratio *
					  outer_nrows *
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
					 outer_nrows);
	}
	/* discount if CPU parallel is enabled */
	run_cost += (comp_cost / pp_info->parallel_divisor);
	/* cost for DMA receive (xPU --> Host) */
	final_cost += xpu_tuple_cost * joinrel->rows;
	/* cost for host projection */
	final_cost += (joinrel->reltarget->cost.per_tuple *
				   joinrel->rows / pp_info->parallel_divisor);

	pp_info->final_cost = final_cost;
	pp_inner->join_nrows = (joinrel->rows / pp_info->parallel_divisor);
	pp_inner->join_startup_cost = startup_cost;
	pp_inner->join_run_cost = run_cost;

	return pp_info;
}

/*
 * __build_simple_xpujoin_path
 */
static CustomPath *
__build_simple_xpujoin_path(PlannerInfo *root,
							RelOptInfo *join_rel,
							Path *outer_path,	/* may be pseudo outer-path */
							Path *inner_path,
							JoinType join_type,
							pgstromOuterPathLeafInfo *op_prev,
							pgstromOuterPathLeafInfo **p_op_leaf,
							List *restrict_clauses,
							SpecialJoinInfo *sjinfo,
							Relids param_source_rels,
							bool try_parallel_path,
							uint32_t xpu_task_flags,
							const CustomPathMethods *xpujoin_path_methods)
{
	pgstromPlanInfo *pp_info;
	pgstromPlanInnerInfo *pp_inner;
	Relids			required_outer;
	ParamPathInfo  *param_info;
	CustomPath	   *cpath;

	required_outer = calc_non_nestloop_required_outer(outer_path,
													  inner_path);
	if (required_outer &&
		!bms_overlap(required_outer, param_source_rels))
		return NULL;

	/*
	 * Get param info
	 */
	param_info = get_joinrel_parampathinfo(root,
										   join_rel,
										   outer_path,
										   inner_path,
										   sjinfo,
										   required_outer,
										   &restrict_clauses);
	if (restrict_clauses == NIL)
		return NULL;		/* cross join is not welcome */

	/*
	 * Build a new pgstromPlanInfo
	 */
	pp_info = __buildXpuJoinPlanInfo(root,
									 join_rel,
									 join_type,
									 restrict_clauses,
									 op_prev,
									 &inner_path);
	if (!pp_info)
		return NULL;
	pp_info->xpu_task_flags &= ~DEVTASK__MASK;
	pp_info->xpu_task_flags |= DEVTASK__JOIN;
	pp_inner = &pp_info->inners[pp_info->num_rels-1];

	/*
	 * Build a new CustomPath
	 */
	cpath = makeNode(CustomPath);
	cpath->path.pathtype = T_CustomScan;
	cpath->path.parent = join_rel;
	cpath->path.pathtarget = join_rel->reltarget;
	cpath->path.param_info = param_info;
	cpath->path.parallel_aware = try_parallel_path;
	cpath->path.parallel_safe = join_rel->consider_parallel;
	cpath->path.parallel_workers = pp_info->parallel_nworkers;
	cpath->path.pathkeys = NIL;
	cpath->path.rows = pp_inner->join_nrows;
	cpath->path.startup_cost = pp_inner->join_startup_cost;
	cpath->path.total_cost = (pp_inner->join_startup_cost +
							  pp_inner->join_run_cost +
							  pp_info->final_cost);
	cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
	cpath->methods = xpujoin_path_methods;
	cpath->custom_paths = lappend(list_copy(op_prev->inner_paths_list), inner_path);
	cpath->custom_private = list_make1(pp_info);
	if (p_op_leaf)
	{
		pgstromOuterPathLeafInfo *op_leaf;

		op_leaf = palloc0(sizeof(pgstromOuterPathLeafInfo));
		op_leaf->pp_info    = pp_info;
		op_leaf->leaf_rel   = join_rel;
		op_leaf->leaf_param = param_info;
		op_leaf->leaf_nrows = cpath->path.rows;
		op_leaf->leaf_cost  = cpath->path.total_cost;
		op_leaf->inner_paths_list = cpath->custom_paths;

		*p_op_leaf = op_leaf;
	}
	return cpath;
}

/*
 * try_add_xpujoin_simple_path
 */
static void
try_add_xpujoin_simple_path(PlannerInfo *root,
							RelOptInfo *join_rel,
							RelOptInfo *outer_rel,
                            Path *inner_path,
                            JoinType join_type,
                            JoinPathExtraData *extra,
							bool try_parallel_path,
							uint32_t xpu_task_flags,
							const CustomPathMethods *xpujoin_path_methods)
{
	pgstromOuterPathLeafInfo *op_prev;
	pgstromOuterPathLeafInfo *op_leaf;
	Path			__outer_path;	/* pseudo outer-path */
	CustomPath	   *cpath;

	op_prev = pgstrom_find_op_normal(outer_rel, try_parallel_path);
	if (!op_prev)
		return;
	memset(&__outer_path, 0, sizeof(Path));
	__outer_path.parent = outer_rel;
	__outer_path.param_info = op_prev->leaf_param;
	__outer_path.rows       = op_prev->leaf_nrows;

	cpath = __build_simple_xpujoin_path(root,
										join_rel,
										&__outer_path,
										inner_path,
										join_type,
										op_prev,
										&op_leaf,
										extra->restrictlist,
										extra->sjinfo,
										extra->param_source_rels,
										try_parallel_path,
										xpu_task_flags,
										xpujoin_path_methods);
	if (!cpath)
		return;
	/* register the XpuJoinPath */
	pgstrom_remember_op_normal(join_rel, op_leaf,
							   try_parallel_path);
	if (!try_parallel_path)
		add_path(join_rel, &cpath->path);
	else
		add_partial_path(join_rel, &cpath->path);
}

/*
 * __build_child_join_sjinfo
 */
static SpecialJoinInfo *
__build_child_join_sjinfo(PlannerInfo *root,
						  Relids leaf_join_relids,
						  SpecialJoinInfo *parent_sjinfo)
{
	SpecialJoinInfo *sjinfo = makeNode(SpecialJoinInfo);

	memcpy(sjinfo, parent_sjinfo, sizeof(SpecialJoinInfo));
	sjinfo->min_lefthand = fixup_relids_by_partition_leaf(root,
														  leaf_join_relids,
														  sjinfo->min_lefthand);
	sjinfo->min_righthand = fixup_relids_by_partition_leaf(root,
														   leaf_join_relids,
														   sjinfo->min_righthand);
	sjinfo->syn_lefthand = fixup_relids_by_partition_leaf(root,
														  leaf_join_relids,
														  sjinfo->syn_lefthand);
	sjinfo->syn_righthand = fixup_relids_by_partition_leaf(root,
														   leaf_join_relids,
														   sjinfo->syn_righthand);
	return sjinfo;
}

/*
 * __lookup_or_build_leaf_joinrel
 */
static AppendRelInfo **
__make_fake_apinfo_array(PlannerInfo *root,
						 RelOptInfo *outer_rel,
						 RelOptInfo *inner_rel)
{
	AppendRelInfo **ap_info_array;
	Relids	__relids;
	int		i;

	ap_info_array = palloc(sizeof(AppendRelInfo *) * root->simple_rel_array_size);
	memcpy(ap_info_array, root->append_rel_array,
		   sizeof(AppendRelInfo *) * root->simple_rel_array_size);
	__relids = bms_union(outer_rel->relids,
						 inner_rel->relids);
	for (i = bms_next_member(__relids, -1);
		 i >= 0;
		 i = bms_next_member(__relids, i))
	{
		RangeTblEntry  *rte = root->simple_rte_array[i];

		if (!ap_info_array[i])
		{
			Relation	rel = relation_open(rte->relid, NoLock);

			ap_info_array[i] = make_append_rel_info(rel, rel, i, i);

			relation_close(rel, NoLock);
		}
	}
	return ap_info_array;
}

static RelOptInfo *
__lookup_or_build_leaf_joinrel(PlannerInfo *root,
							   RelOptInfo *parent_joinrel,
							   RelOptInfo *outer_rel,
							   RelOptInfo *inner_rel,
							   List *restrictlist,
							   SpecialJoinInfo *sjinfo,
							   JoinType jointype)
{
	RelOptInfo *leaf_joinrel;
	Relids		relids;

	relids = bms_union(outer_rel->relids,
					   inner_rel->relids);
	leaf_joinrel = find_join_rel(root, relids);
	if (!leaf_joinrel)
	{
		RelOptKind	reloptkind_saved = inner_rel->reloptkind;
		bool		partitionwise_saved = parent_joinrel->consider_partitionwise_join;
		AppendRelInfo **ap_array_saved = root->append_rel_array;

		/* a small hack to avoid assert() */
		inner_rel->reloptkind = RELOPT_OTHER_MEMBER_REL;
		PG_TRY();
		{
			inner_rel->reloptkind = RELOPT_OTHER_MEMBER_REL;
			parent_joinrel->consider_partitionwise_join = true;
			root->append_rel_array = __make_fake_apinfo_array(root,
															  outer_rel,
															  inner_rel);
			leaf_joinrel = build_child_join_rel(root,
												outer_rel,
												inner_rel,
												parent_joinrel,
												restrictlist,
												sjinfo);
//												jointype);
		}
		PG_CATCH();
		{
			inner_rel->reloptkind = reloptkind_saved;
			parent_joinrel->consider_partitionwise_join = partitionwise_saved;
			root->append_rel_array = ap_array_saved;
			PG_RE_THROW();
		}
		PG_END_TRY();
		inner_rel->reloptkind = reloptkind_saved;
		parent_joinrel->consider_partitionwise_join = partitionwise_saved;
		root->append_rel_array = ap_array_saved;
	}
	return leaf_joinrel;
}

/*
 * try_add_xpujoin_partition_path
 */
static void
try_add_xpujoin_partition_path(PlannerInfo *root,
							   RelOptInfo *join_rel,
							   RelOptInfo *outer_rel,
							   Path *inner_path,
							   JoinType join_type,
							   JoinPathExtraData *extra,
							   bool try_parallel_path,
							   uint32_t xpu_task_flags,
							   const CustomPathMethods *xpujoin_path_methods)
{
	List	   *op_prev_list = NIL;
	List	   *op_leaf_list = NIL;
	List	   *cpaths_list = NIL;
	PathTarget *append_target = NULL;
	Path	   *append_path;
	Relids		required_outer = NULL;
	int			parallel_nworkers = 0;
	double		total_nrows = 0.0;
	ListCell   *lc;

	op_prev_list = pgstrom_find_op_leafs(outer_rel, try_parallel_path);
	foreach (lc, op_prev_list)
	{
		pgstromOuterPathLeafInfo *op_prev = lfirst(lc);
		pgstromOuterPathLeafInfo *op_leaf;
		Path			__outer_path;	//pseudo outer-path
		Path		   *__inner_path = inner_path;
		Relids			__join_relids;
		Relids			__required_outer;
		List		   *__restrict_clauses = NIL;
		SpecialJoinInfo *__sjinfo;
		RelOptInfo	   *__join_rel;
		CustomPath	   *cpath;

		/* setup pseudo outer-path */
		memset(&__outer_path, 0, sizeof(Path));
		__outer_path.parent     = op_prev->leaf_rel;
		__outer_path.param_info = op_prev->leaf_param;
		__outer_path.rows       = op_prev->leaf_nrows;

		__required_outer = calc_non_nestloop_required_outer(&__outer_path,
															__inner_path);
		if (__required_outer &&
			!bms_overlap(__required_outer, extra->param_source_rels))
			return;

		/*
		 * setup SpecialJoinInfo for inner-leaf join
		 */
		__join_relids = bms_union(op_prev->leaf_rel->relids,
								  __inner_path->parent->relids);
		__sjinfo = __build_child_join_sjinfo(root,
											 __join_relids,
											 extra->sjinfo);
		/*
		 * fixup restrict_clauses
		 */
		__restrict_clauses =
			fixup_expression_by_partition_leaf(root,
											   op_prev->leaf_rel->relids,
											   extra->restrictlist);
		/*
		 * lookup or construct a new join_rel
		 */
		__join_rel = __lookup_or_build_leaf_joinrel(root,
													join_rel,
													op_prev->leaf_rel,
													inner_path->parent,
													__restrict_clauses,
													__sjinfo,
													join_type);
		/*
		 * build XpuJoinPath
		 */
		cpath = __build_simple_xpujoin_path(root,
											__join_rel,
											&__outer_path,
											inner_path,
											join_type,
											op_prev,
											&op_leaf,
											__restrict_clauses,
											__sjinfo,
											extra->param_source_rels,
											try_parallel_path,
											xpu_task_flags,
											xpujoin_path_methods);
		if (!cpath)
			return;
		parallel_nworkers += cpath->path.parallel_workers;
		total_nrows += cpath->path.rows;

		cpaths_list  = lappend(cpaths_list, cpath);
		op_leaf_list = lappend(op_leaf_list, op_leaf);
	}

	if (list_length(cpaths_list) == 0)
		return;

	if (try_parallel_path)
	{
		if (parallel_nworkers > max_parallel_workers_per_gather)
			parallel_nworkers = max_parallel_workers_per_gather;
		if (parallel_nworkers == 0)
			return;
	}
	append_path = (Path *)
		create_append_path(root,
						   join_rel,
						   (try_parallel_path ? NIL : cpaths_list),
						   (try_parallel_path ? cpaths_list : NIL),
						   NIL,
						   required_outer,
						   (try_parallel_path ? parallel_nworkers : 0),
						   try_parallel_path,
						   total_nrows);
	//append_path->target = append_target; //FIXME
	pgstrom_remember_op_leafs(join_rel, op_leaf_list,
							  try_parallel_path);
	if (!try_parallel_path)
		add_path(join_rel, append_path);
	else
		add_partial_path(join_rel, append_path);
}


/*
 * __xpuJoinAddCustomPathCommon
 */
static void
__xpuJoinAddCustomPathCommon(PlannerInfo *root,
							 RelOptInfo *joinrel,
							 RelOptInfo *outerrel,
							 RelOptInfo *innerrel,
							 JoinType join_type,
							 JoinPathExtraData *extra,
							 uint32_t xpu_task_flags,
							 const CustomPathMethods *xpujoin_path_methods,
							 bool consider_partition)
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
		{
			try_add_xpujoin_simple_path(root,
										joinrel,
										outerrel,
										inner_path,
										join_type,
										extra,
										try_parallel > 0,
										xpu_task_flags,
										xpujoin_path_methods);
			if (consider_partition)
				try_add_xpujoin_partition_path(root,
											   joinrel,
											   outerrel,
											   inner_path,
											   join_type,
											   extra,
											   try_parallel > 0,
											   xpu_task_flags,
											   xpujoin_path_methods);
		}
		/* 2nd trial uses the partial paths */
		inner_pathlist = innerrel->partial_pathlist;
	}
}

/*
 * __xpuJoinTryAddPartitionLeafs
 */
static void
__xpuJoinTryAddPartitionLeafs(PlannerInfo *root,
							  RelOptInfo *joinrel,
							  bool be_parallel)
{
	List	   *op_leaf_list = NIL;

	for (int k=0; k < joinrel->nparts; k++)
	{
		RelOptInfo *leaf_rel = joinrel->part_rels[k];
		pgstromOuterPathLeafInfo *op_leaf;

		op_leaf = pgstrom_find_op_normal(leaf_rel, be_parallel);
		if (!op_leaf)
			return;
		op_leaf_list = lappend(op_leaf_list, op_leaf);
	}
	pgstrom_remember_op_leafs(joinrel, op_leaf_list, be_parallel);

	if (joinrel->parent)
	{
		RelOptInfo *parent = joinrel->parent;

		if (parent->nparts > 0 &&
			parent->part_rels[parent->nparts-1] == joinrel)
		{
			__xpuJoinTryAddPartitionLeafs(root, parent, be_parallel);
		}
	}
}

/*
 * XpuJoinAddCustomPath
 */
static void
XpuJoinAddCustomPath(PlannerInfo *root,
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
	/* quick bailout if PG-Strom is not enabled */
	if (pgstrom_enabled())
	{
		if (pgstrom_enable_gpujoin)
			__xpuJoinAddCustomPathCommon(root,
										 joinrel,
										 outerrel,
										 innerrel,
										 join_type,
										 extra,
										 TASK_KIND__GPUJOIN,
										 &gpujoin_path_methods,
										 pgstrom_enable_partitionwise_gpujoin);
		if (pgstrom_enable_dpujoin)
			__xpuJoinAddCustomPathCommon(root,
										 joinrel,
										 outerrel,
										 innerrel,
										 join_type,
										 extra,
										 TASK_KIND__DPUJOIN,
										 &dpujoin_path_methods,
										 pgstrom_enable_partitionwise_dpujoin);
		if (joinrel->parent)
		{
			RelOptInfo *parent = joinrel->parent;

			if (parent->nparts > 0 &&
				parent->part_rels[parent->nparts-1] == joinrel)
			{
				__xpuJoinTryAddPartitionLeafs(root, parent, false);
				__xpuJoinTryAddPartitionLeafs(root, parent, true);
			}
		}
	}
}

/*
 * build_fallback_exprs_scan
 */
static Node *
__build_fallback_exprs_scan_walker(Node *node, void *data)
{
	codegen_context *context = (codegen_context *)data;

	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		Var	   *var = (Var *)node;

		if (var->varno == context->scan_relid)
		{
			return (Node *)makeVar(OUTER_VAR,
								   var->varattno,
								   var->vartype,
								   var->vartypmod,
								   var->varcollid,
								   var->varlevelsup);
		}
		elog(ERROR, "Var-node does not reference the base relation (%d): %s",
			 context->scan_relid, nodeToString(var));
	}
	return expression_tree_mutator(node, __build_fallback_exprs_scan_walker, data);
}

static List *
build_fallback_exprs_scan(codegen_context *context, List *scan_exprs)
{
	return (List *)__build_fallback_exprs_scan_walker((Node *)scan_exprs, context);
}

/*
 * build_fallback_exprs_join
 */
static Node *
__build_fallback_exprs_join_walker(Node *node, void *data)
{
	codegen_context *context = (codegen_context *)data;
	ListCell   *lc;

	if (!node)
		return NULL;
	foreach (lc, context->kvars_deflist)
	{
		codegen_kvar_defitem *kvar = lfirst(lc);

		if (codegen_expression_equals(node, kvar->kv_expr))
		{
			return (Node *)makeVar(INDEX_VAR,
								   kvar->kv_slot_id + 1,
								   exprType(node),
								   exprTypmod(node),
								   exprCollation(node),
								   0);
		}
	}
	if (IsA(node, Var))
		elog(ERROR, "Bug? Var-node (%s) is missing at the kvars_exprs list",
			 nodeToString(node));

	return expression_tree_mutator(node, __build_fallback_exprs_join_walker, data);
}

static List *
build_fallback_exprs_join(codegen_context *context, List *join_exprs)
{
	return (List *)__build_fallback_exprs_join_walker((Node *)join_exprs, context);
}

static Node *
__build_fallback_exprs_inner_walker(Node *node, void *data)
{
	codegen_context *context = (codegen_context *)data;
	ListCell   *lc;

	if (!node)
		return NULL;
	foreach (lc, context->kvars_deflist)
	{
		codegen_kvar_defitem *kvdef = lfirst(lc);

		if (codegen_expression_equals(node, kvdef->kv_expr))
		{
			return (Node *)makeVar(INNER_VAR,
								   kvdef->kv_resno,
								   exprType(node),
								   exprTypmod(node),
								   exprCollation(node),
								   0);
		}
	}
	if (IsA(node, Var))
		elog(ERROR, "Bug? Var-node (%s) is missing at the kvars_exprs list",
			 nodeToString(node));

	return expression_tree_mutator(node, __build_fallback_exprs_inner_walker, data);
}

static List *
build_fallback_exprs_inner(codegen_context *context, List *inner_keys)
{
	return (List *)__build_fallback_exprs_inner_walker((Node *)inner_keys, context);
}

/*
 * pgstrom_build_tlist_dev
 */
static List *
__pgstrom_build_tlist_dev_expr(List *tlist_dev,
							   Node *node,
							   uint32_t xpu_task_flags,
							   Index scan_relid,
							   List *inner_target_list)
{
	int			depth;
	int			resno;
	ListCell   *lc1, *lc2;

	if (!node)
		return tlist_dev;

	/* check whether the node is already on the tlist_dev */
	foreach (lc1, tlist_dev)
	{
		TargetEntry	*tle = lfirst(lc1);

		if (codegen_expression_equals(node, tle->expr))
			return tlist_dev;
	}

	/* check whether the node is identical with any of input */
	if (IsA(node, Var))
	{
		Var	   *var = (Var *)node;

		if (var->varno == scan_relid)
		{
			depth = 0;
			resno = var->varattno;
			goto found;
		}
	}
	depth = 1;
	foreach (lc1, inner_target_list)
	{
		PathTarget *reltarget = lfirst(lc1);

		resno = 1;
		foreach (lc2, reltarget->exprs)
		{
			if (codegen_expression_equals(node, lfirst(lc2)))
				goto found;
			resno++;
		}
	}
	depth = -1;
	resno = -1;
found:
	/*
	 * NOTE: Even if the expression is not supported by the device,
	 * Var-node must be added because it is a simple projection that
	 * is never touched during xPU kernel execution.
	 * All the xPU kernel doing is simple copy.
	 */
	if (IsA(node, Var) ||
		pgstrom_xpu_expression((Expr *)node,
							   xpu_task_flags,
							   scan_relid,
							   inner_target_list,
							   NULL))
	{
		AttrNumber		tleno = list_length(tlist_dev) + 1;
		TargetEntry	   *tle = makeTargetEntry((Expr *)node,
											  tleno,
											  NULL,
											  false);
		tle->resorigtbl = (depth < 0 ? UINT_MAX : depth);
		tle->resorigcol  = resno;
		tlist_dev = lappend(tlist_dev, tle);
	}
	else
	{
		List	   *vars_list = pull_vars_of_level(node, 0);

		foreach (lc1, vars_list)
		{
			tlist_dev = __pgstrom_build_tlist_dev_expr(tlist_dev,
													   lfirst(lc1),
													   xpu_task_flags,
													   scan_relid,
													   inner_target_list);
		}
	}
	return tlist_dev;
}

/*
 * __build_explain_tlist_junks
 *
 * it builds junk TLEs for EXPLAIN output only
 */
static void
__build_explain_tlist_junks(codegen_context *context,
							PlannerInfo *root,
							const Bitmapset *outer_refs)
{
	Index		scan_relid = context->scan_relid;
	RelOptInfo *base_rel = root->simple_rel_array[scan_relid];
	RangeTblEntry *rte = root->simple_rte_array[scan_relid];
	int			j, k;

	Assert(IS_SIMPLE_REL(base_rel) && rte->rtekind == RTE_RELATION);
	/* depth==0 */
	for (j = bms_next_member(outer_refs, -1);
		 j >= 0;
		 j = bms_next_member(outer_refs, j))
	{
		Var	   *var;
		char   *attname;

		k = j + FirstLowInvalidHeapAttributeNumber;
		if (k != InvalidAttrNumber)
		{
			HeapTuple	htup;
			Form_pg_attribute attr;

			htup = SearchSysCache2(ATTNUM,
								   ObjectIdGetDatum(rte->relid),
								   Int16GetDatum(k));
			if (!HeapTupleIsValid(htup))
				elog(ERROR,"cache lookup failed for attriubte %d of relation %u",
					 k, rte->relid);
			attr = (Form_pg_attribute) GETSTRUCT(htup);
			var = makeVar(base_rel->relid,
						  attr->attnum,
						  attr->atttypid,
						  attr->atttypmod,
						  attr->attcollation,
						  0);
			attname = pstrdup(NameStr(attr->attname));
			ReleaseSysCache(htup);
		}
		else
		{
			/* special case handling if whole row reference */
			var = makeWholeRowVar(rte,
								  base_rel->relid,
								  0, false);
			attname = get_rel_name(rte->relid);
		}

		if (tlist_member((Expr *)var, context->tlist_dev) == NULL)
		{
			TargetEntry *tle;
			int		resno = list_length(context->tlist_dev) + 1;

			tle = makeTargetEntry((Expr *)var, resno, attname, true);
			context->tlist_dev = lappend(context->tlist_dev, tle);
		}
	}
	/* depth > 0 */
	for (int depth=1; depth <= context->num_rels; depth++)
	{
		PathTarget *target = context->pd[depth].inner_target;
		ListCell   *lc;

		foreach (lc, target->exprs)
		{
			Expr   *expr = lfirst(lc);

			if (tlist_member(expr, context->tlist_dev) == NULL)
			{
				TargetEntry *tle;
				int		resno = list_length(context->tlist_dev) + 1;

				tle = makeTargetEntry(expr, resno, NULL, true);
				context->tlist_dev = lappend(context->tlist_dev, tle);
			}
		}
	}
}

static void
pgstrom_build_join_tlist_dev(codegen_context *context,
							 PlannerInfo *root,
							 RelOptInfo *joinrel,
							 List *tlist)
{
	List	   *tlist_dev = NIL;
	List	   *inner_target_list = NIL;
	ListCell   *lc;

	for (int depth=1; depth <= context->num_rels; depth++)
	{
		PathTarget *target = context->pd[depth].inner_target;

		inner_target_list = lappend(inner_target_list, target);
	}

	if (tlist != NIL)
	{
		foreach (lc, tlist)
		{
			TargetEntry *tle = lfirst(lc);

			context->top_expr = tle->expr;
			if (contain_var_clause((Node *)tle->expr))
				tlist_dev = __pgstrom_build_tlist_dev_expr(tlist_dev,
														   (Node *)tle->expr,
														   context->required_flags,
														   context->scan_relid,
														   inner_target_list);
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
		PathTarget *reltarget = joinrel->reltarget;

		foreach (lc, reltarget->exprs)
		{
			Node   *node = lfirst(lc);

			context->top_expr = (Expr *)node;
			if (contain_var_clause(node))
				tlist_dev = __pgstrom_build_tlist_dev_expr(tlist_dev,
														   node,
														   context->required_flags,
														   context->scan_relid,
														   inner_target_list);
		}
	}
	//add junk?
	context->tlist_dev = tlist_dev;
}

/*
 * pgstrom_build_groupby_tlist_dev
 */
static void
pgstrom_build_groupby_tlist_dev(codegen_context *context,
								PlannerInfo *root,
								List *tlist)
{
	ListCell   *lc1, *lc2;

	context->tlist_dev = copyObject(tlist);
	foreach (lc1, tlist)
	{
		TargetEntry *tle = lfirst(lc1);

		if (IsA(tle->expr, FuncExpr))
		{
			FuncExpr   *f = (FuncExpr *)tle->expr;

			foreach (lc2, f->args)
			{
				Expr   *arg = lfirst(lc2);
				int		resno = list_length(context->tlist_dev) + 1;

				if (!tlist_member(arg, context->tlist_dev))
				{
					TargetEntry *__tle = makeTargetEntry(arg, resno, NULL, true);

					context->tlist_dev = lappend(context->tlist_dev, __tle);
				}
			}
		}
		else
		{
			Assert(tlist_member(tle->expr, context->tlist_dev));
		}
	}
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
	codegen_context *context;
	CustomScan *cscan;
	Bitmapset  *outer_refs = NULL;
	List	   *join_quals_stacked = NIL;
	List	   *other_quals_stacked = NIL;
	List	   *hash_keys_stacked = NIL;
	List	   *gist_quals_stacked = NIL;
	List	   *fallback_tlist = NIL;
	ListCell   *lc;

	Assert(pp_info->num_rels == list_length(custom_plans));
	context = create_codegen_context(root, cpath, pp_info);

	/* codegen for outer scan, if any */
	if (pp_info->scan_quals)
	{
		pp_info->scan_quals = pp_info->scan_quals;
		pp_info->kexp_scan_quals
			= codegen_build_scan_quals(context, pp_info->scan_quals);
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
		if (pp_inner->hash_outer_keys_original != NIL &&
			pp_inner->hash_inner_keys_original != NIL)
		{
			hash_keys_stacked = lappend(hash_keys_stacked,
										pp_inner->hash_outer_keys_original);
			pull_varattnos((Node *)pp_inner->hash_outer_keys_original,
						   pp_info->scan_relid,
						   &outer_refs);
		}
		else
		{
			Assert(pp_inner->hash_outer_keys_original == NIL &&
				   pp_inner->hash_inner_keys_original == NIL);
			hash_keys_stacked = lappend(hash_keys_stacked, NIL);
		}
		
		/* xpu code to evaluate join qualifiers */
		join_quals_stacked = lappend(join_quals_stacked,
									 pp_inner->join_quals_original);
		pull_varattnos((Node *)pp_inner->join_quals_original,
					   pp_info->scan_relid,
					   &outer_refs);
		other_quals_stacked = lappend(other_quals_stacked,
									  pp_inner->other_quals_original);
		pull_varattnos((Node *)pp_inner->other_quals_original,
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
		Relids	leaf_relids = cpath->path.parent->relids;
		tlist = fixup_expression_by_partition_leaf(root, leaf_relids, tlist);
		pgstrom_build_groupby_tlist_dev(context, root, tlist);
		codegen_build_groupby_actions(context, pp_info);
	}
	else
	{
		/* build device projection */
		pgstrom_build_join_tlist_dev(context, root, joinrel, tlist);
		pp_info->kexp_projection = codegen_build_projection(context);
	}
	pull_varattnos((Node *)context->tlist_dev,
				   pp_info->scan_relid,
				   &outer_refs);
	__build_explain_tlist_junks(context, root, outer_refs);

	/* assign remaining PlanInfo members */
	pp_info->kexp_join_quals_packed
		= codegen_build_packed_joinquals(context,
										 join_quals_stacked,
										 other_quals_stacked);
	pp_info->kexp_hash_keys_packed
		= codegen_build_packed_hashkeys(context,
										hash_keys_stacked);
	codegen_build_packed_gistevals(context, pp_info);
	codegen_build_packed_kvars_load(context, pp_info);
	codegen_build_packed_kvars_move(context, pp_info);

	pp_info->kvars_deflist = context->kvars_deflist;
	pp_info->kvecs_bufsz = KVEC_ALIGN(context->kvecs_usage);
	pp_info->kvecs_ndims = context->kvecs_ndims;
	pp_info->extra_flags = context->extra_flags;
	pp_info->extra_bufsz = context->extra_bufsz;
	pp_info->used_params = context->used_params;
	pp_info->outer_refs  = outer_refs;
	/*
	 * fixup fallback expressions
	 */
	for (int i=0; i < pp_info->num_rels; i++)
	{
		pgstromPlanInnerInfo *pp_inner = &pp_info->inners[i];

		pp_inner->hash_outer_keys_fallback
			= build_fallback_exprs_join(context, pp_inner->hash_outer_keys_original);
		pp_inner->hash_inner_keys_fallback
			= build_fallback_exprs_inner(context, pp_inner->hash_inner_keys_original);
		pp_inner->join_quals_fallback
			= build_fallback_exprs_join(context, pp_inner->join_quals_original);
		pp_inner->other_quals_fallback
			= build_fallback_exprs_join(context, pp_inner->other_quals_original);
	}

	foreach (lc, context->tlist_dev)
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
		 ? build_fallback_exprs_scan(context, fallback_tlist)
		 : build_fallback_exprs_join(context, fallback_tlist));

	cscan = makeNode(CustomScan);
	cscan->scan.plan.targetlist = tlist;
	cscan->scan.scanrelid = pp_info->scan_relid;
	cscan->flags = cpath->flags;
	cscan->custom_plans = custom_plans;
	cscan->custom_scan_tlist = context->tlist_dev;
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

/*
 * PlanDpuJoinPath
 */
static Plan *
PlanDpuJoinPath(PlannerInfo *root,
				RelOptInfo *joinrel,
				CustomPath *cpath,
				List *tlist,
				List *clauses,
				List *custom_plans)
{
	pgstromPlanInfo *pp_info = linitial(cpath->custom_private);
	CustomScan *cscan;

	cscan = PlanXpuJoinPathCommon(root,
								  joinrel,
								  cpath,
								  tlist,
								  custom_plans,
								  pp_info,
								  &dpujoin_plan_methods);
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

/*
 * CreateDpuJoinState
 */
static Node *
CreateDpuJoinState(CustomScan *cscan)
{
	pgstromTaskState *pts;
	pgstromPlanInfo  *pp_info = deform_pgstrom_plan_info(cscan);
	int			num_rels = list_length(cscan->custom_plans);

	Assert(cscan->methods == &dpujoin_plan_methods);
	pts = palloc0(offsetof(pgstromTaskState, inners[num_rels]));
	NodeSetTag(pts, T_CustomScanState);
	pts->css.flags = cscan->flags;
	pts->css.methods = &dpujoin_exec_methods;
	pts->xpu_task_flags = pp_info->xpu_task_flags;
	pts->pp_info = pp_info;
	Assert((pts->xpu_task_flags & TASK_KIND__MASK) == TASK_KIND__DPUJOIN &&
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
typedef struct
{
	uint32_t		nitems;
	uint32_t		nrooms;
	size_t			usage;
	struct {
		HeapTuple	htup;
		uint32_t	hash;		/* if hash-join or gist-join */
	} rows[1];
} inner_preload_buffer;

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
			 lc2, istate->hash_inner_funcs)
	{
		ExprState	   *es = lfirst(lc1);
		devtype_hashfunc_f h_func = lfirst(lc2);
		Datum			datum;
		bool			isnull;

		datum = ExecEvalExpr(es, econtext, &isnull);
		hash ^= h_func(isnull, datum);
	}
	hash ^= 0xffffffffU;

	return hash;
}

/*
 * execInnerPreloadOneDepth
 */
static void
execInnerPreloadOneDepth(MemoryContext memcxt,
						 pgstromTaskInnerState *istate,
						 pg_atomic_uint64 *p_shared_inner_nitems,
						 pg_atomic_uint64 *p_shared_inner_usage)
{
	PlanState	   *ps = istate->ps;
	MemoryContext	oldcxt;
	inner_preload_buffer *preload_buf;

	/* initial alloc of inner_preload_buffer */
	preload_buf = MemoryContextAlloc(memcxt, offsetof(inner_preload_buffer,
													  rows[12000]));
	memset(preload_buf, 0, offsetof(inner_preload_buffer, rows));
	preload_buf->nrooms = 12000;

	for (;;)
	{
		TupleTableSlot *slot;
		TupleDesc		tupdesc;
		HeapTuple		htup;
		uint32_t		index;

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
		if (preload_buf->nitems >= preload_buf->nrooms)
		{
			uint32_t	nrooms_new = 2 * preload_buf->nrooms + 4000;

			preload_buf = repalloc_huge(preload_buf, offsetof(inner_preload_buffer,
															  rows[nrooms_new]));
			preload_buf->nrooms = nrooms_new;
		}
		index = preload_buf->nitems++;

		if (istate->hash_inner_keys != NIL)
		{
			uint32_t	hash = get_tuple_hashvalue(istate, slot);

			preload_buf->rows[index].htup = htup;
			preload_buf->rows[index].hash = hash;
			preload_buf->usage += MAXALIGN(offsetof(kern_hashitem,
													t.htup) + htup->t_len);
		}
		else if (istate->gist_irel)
		{
			ItemPointer	ctid;
			uint32_t	hash;
			bool		isnull;

			ctid = (ItemPointer)
				slot_getattr(slot, istate->gist_ctid_resno, &isnull);
			if (isnull)
				elog(ERROR, "Unable to build GiST-index buffer with NULL-ctid");
			ItemPointerCopy(ctid, &htup->t_self);
			hash = hash_any((unsigned char *)ctid, sizeof(ItemPointerData));

			preload_buf->rows[index].htup = htup;
			preload_buf->rows[index].hash = hash;
			preload_buf->usage += MAXALIGN(offsetof(kern_hashitem,
													t.htup) + htup->t_len);
		}
		else
		{
			preload_buf->rows[index].htup = htup;
			preload_buf->rows[index].hash = 0;
			preload_buf->usage += MAXALIGN(offsetof(kern_tupitem,
													htup) + htup->t_len);
		}
		MemoryContextSwitchTo(oldcxt);
	}
	istate->preload_buffer = preload_buf;
	pg_atomic_fetch_add_u64(p_shared_inner_nitems, preload_buf->nitems);
	pg_atomic_fetch_add_u64(p_shared_inner_usage,  preload_buf->usage);
}

/*
 * innerPreloadSetupGiSTIndex
 */
static void
__innerPreloadSetupGiSTIndexWalker(char *base,
								   BlockNumber blkno,
								   BlockNumber nblocks,
								   BlockNumber parent_blkno,
								   OffsetNumber parent_offno)
{
	Page			page = (Page)(base + BLCKSZ * blkno);
	PageHeader		hpage = (PageHeader) page;
	GISTPageOpaque	op = GistPageGetOpaque(page);
	OffsetNumber	i, maxoff;

	Assert(hpage->pd_lsn.xlogid == InvalidBlockNumber &&
		   hpage->pd_lsn.xrecoff == InvalidOffsetNumber);
	hpage->pd_lsn.xlogid = parent_blkno;
	hpage->pd_lsn.xrecoff = parent_offno;
	if ((op->flags & F_LEAF) != 0)
		return;
	maxoff = PageGetMaxOffsetNumber(page);
	for (i=FirstOffsetNumber; i <= maxoff; i = OffsetNumberNext(i))
    {
		ItemId		iid = PageGetItemId(page, i);
		IndexTuple	it;
		BlockNumber	child;

		if (ItemIdIsDead(iid))
			continue;
		it = (IndexTuple) PageGetItem(page, iid);
		child = BlockIdGetBlockNumber(&it->t_tid.ip_blkid);
		if (child < nblocks)
			__innerPreloadSetupGiSTIndexWalker(base, child, nblocks, blkno, i);
	}
}

static void
innerPreloadSetupGiSTIndex(kern_data_store *kds_gist)
{
	__innerPreloadSetupGiSTIndexWalker((char *)KDS_BLOCK_PGPAGE(kds_gist, 0),
									   0, kds_gist->nitems,
									   InvalidBlockNumber,
									   InvalidOffsetNumber);
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
				kds->length = block_offset + BLCKSZ * nblocks;
				kds->nitems = nblocks;
				kds->block_nloaded = nblocks;
				innerPreloadSetupGiSTIndex(kds);
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
	inner_preload_buffer *preload_buf = istate->preload_buffer;

	for (uint32_t index=0; index < preload_buf->nitems; index++)
	{
		HeapTuple	htup = preload_buf->rows[index].htup;
		size_t		sz;
		kern_tupitem *titem;

		sz = MAXALIGN(offsetof(kern_tupitem, htup) + htup->t_len);
		curr_pos -= sz;
		titem = (kern_tupitem *)curr_pos;
		titem->t_len = htup->t_len;
		titem->rowid = rowid;
		memcpy(&titem->htup, htup->t_data, htup->t_len);
		memcpy(&titem->htup.t_ctid, &htup->t_self, sizeof(ItemPointerData));

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
	inner_preload_buffer *preload_buf = istate->preload_buffer;

	for (uint32_t index=0; index < preload_buf->nitems; index++)
	{
		HeapTuple	htup = preload_buf->rows[index].htup;
		uint32_t	hash = preload_buf->rows[index].hash;
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
		memcpy(&hitem->t.htup.t_ctid, &htup->t_self, sizeof(ItemPointerData));

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

				execInnerPreloadOneDepth(memcxt, istate,
										 &ps_state->inners[i].inner_nitems,
										 &ps_state->inners[i].inner_usage);
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
				inner_preload_buffer *preload_buf = istate->preload_buffer;
				kern_data_store *kds = KERN_MULTIRELS_INNER_KDS(pts->h_kmrels, i);
                uint32_t		base_nitems;
				uint32_t		base_usage;

				SpinLockAcquire(&ps_state->preload_mutex);
				base_nitems  = kds->nitems;
				kds->nitems += preload_buf->nitems;
				base_usage   = kds->usage;
				kds->usage  += __kds_packed(preload_buf->usage);
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
						   const kern_expression *kexp_vloads,
						   const kern_data_store *kds,
						   const ItemPointer t_self,
						   const HeapTupleHeaderData *htup)
{
	const kern_varload_desc *vl_desc = kexp_vloads->u.load.desc;
	uint32_t	offset = htup->t_hoff;
	uint32_t	kvcnt = 0;
	uint32_t	resno;
	uint32_t	ncols = Min(htup->t_infomask2 & HEAP_NATTS_MASK, kds->ncols);
	bool		heap_hasnull = ((htup->t_infomask & HEAP_HASNULL) != 0);

	Assert(kexp_vloads->opcode == FuncOpCode__LoadVars);
	/* extract system attributes, if rquired */
	while (kvcnt < kexp_vloads->u.load.nitems &&
		   vl_desc->vl_resno < 0)
	{
		int		slot_id = vl_desc->vl_slot_id;
		Datum	datum;

		switch (vl_desc->vl_resno)
		{
			case SelfItemPointerAttributeNumber:
				datum = PointerGetDatum(t_self);
				break;
			case MinTransactionIdAttributeNumber:
				datum = TransactionIdGetDatum(HeapTupleHeaderGetRawXmin(htup));
				break;
			case MaxTransactionIdAttributeNumber:
				datum = TransactionIdGetDatum(HeapTupleHeaderGetRawXmax(htup));
				break;
			case MinCommandIdAttributeNumber:
			case MaxCommandIdAttributeNumber:
				datum = CommandIdGetDatum(HeapTupleHeaderGetRawCommandId(htup));
				break;
			case TableOidAttributeNumber:
				datum = ObjectIdGetDatum(kds->table_oid);
				break;
			default:
				elog(ERROR, "invalid attnum: %d", vl_desc->vl_resno);
		}
		fallback_slot->tts_isnull[slot_id] = false;
		fallback_slot->tts_values[slot_id] = datum;
		vl_desc++;
		kvcnt++;
	}
	/* extract the user data */
	resno = 1;
	while (kvcnt < kexp_vloads->u.load.nitems && resno <= ncols)
	{
		const kern_colmeta *cmeta = &kds->colmeta[resno-1];
		const char	   *addr;

		if (heap_hasnull && att_isnull(resno-1, htup->t_bits))
		{
			addr = NULL;
		}
		else
		{
			if (cmeta->attlen > 0)
				offset = TYPEALIGN(cmeta->attalign, offset);
			else if (!VARATT_NOT_PAD_BYTE((char *)htup + offset))
				offset = TYPEALIGN(cmeta->attalign, offset);
			addr = ((char *)htup + offset);
			if (cmeta->attlen > 0)
				offset += cmeta->attlen;
			else
				offset += VARSIZE_ANY(addr);
		}

		if (vl_desc->vl_resno == resno)
		{
			int		slot_id = vl_desc->vl_slot_id;
			Datum	datum;

			if (!addr)
				datum = 0;
			else if (cmeta->attbyval)
			{
				switch (cmeta->attlen)
				{
					case 1:
						datum = *((uint8_t *)addr);
						break;
					case 2:
						datum = *((uint16_t *)addr);
						break;
					case 4:
						datum = *((uint32_t *)addr);
						break;
					case 8:
						datum = *((uint64_t *)addr);
						break;
					default:
						elog(ERROR, "invalid attlen: %d", cmeta->attlen);
				}
			}
			else
			{
				datum = PointerGetDatum(addr);
			}
			fallback_slot->tts_isnull[slot_id] = !addr;
			fallback_slot->tts_values[slot_id] = datum;
			vl_desc++;
			kvcnt++;
		}
		resno++;
	}
	/* fill-up by NULL for the remaining fields */
	while (kvcnt < kexp_vloads->u.load.nitems)
	{
		int		slot_id = vl_desc->vl_slot_id;

		fallback_slot->tts_isnull[slot_id] = true;
		fallback_slot->tts_values[slot_id] = 0;
		vl_desc++;
		kvcnt++;
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
	kern_expression *kexp_join_kvars_load = NULL;

	if (pp_info->kexp_load_vars_packed)
	{
		const kern_expression *temp = (const kern_expression *)
			VARDATA(pp_info->kexp_load_vars_packed);
		kexp_join_kvars_load = __PICKUP_PACKED_KEXP(temp, depth);
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
			ItemPointerData	t_self;

			ItemPointerSetInvalid(&t_self);
			Assert(kexp_join_kvars_load->u.load.depth == depth);
			__execFallbackLoadVarsSlot(pts->fallback_slot,
									   kexp_join_kvars_load,
									   kds_in,
									   &t_self,
									   &tupitem->htup);
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
	kern_expression *kexp_join_kvars_load = NULL;
	kern_hashitem  *hitem;
	uint32_t		hash;
	ListCell	   *lc1, *lc2;

	if (pp_info->kexp_load_vars_packed)
	{
		const kern_expression *temp = (const kern_expression *)
			VARDATA(pp_info->kexp_load_vars_packed);
		kexp_join_kvars_load = __PICKUP_PACKED_KEXP(temp, depth);
	}
	Assert(kds_in->format == KDS_FORMAT_HASH);

	/*
	 * Compute that hash-value
	 */
	hash = 0xffffffffU;
	forboth (lc1, istate->hash_outer_keys,
			 lc2, istate->hash_outer_funcs)
	{
		ExprState	   *h_key = lfirst(lc1);
		devtype_hashfunc_f h_func = lfirst(lc2);
		Datum			datum;
		bool			isnull;

		datum = ExecEvalExprSwitchContext(h_key, econtext, &isnull);
		hash ^= h_func(isnull, datum);
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
			ItemPointerData	t_self;

			ItemPointerSetInvalid(&t_self);
			__execFallbackLoadVarsSlot(pts->fallback_slot,
									   kexp_join_kvars_load,
									   kds_in,
									   &t_self,
									   &hitem->t.htup);
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

bool
ExecFallbackCpuJoin(pgstromTaskState *pts, HeapTuple tuple)
{
	pgstromPlanInfo *pp_info = pts->pp_info;
	ExprContext    *econtext = pts->css.ss.ps.ps_ExprContext;
	TupleTableSlot *base_slot = pts->base_slot;
	TupleTableSlot *fallback_slot = pts->fallback_slot;
	size_t			fallback_index_saved = pts->fallback_index;
	ListCell	   *lc;

	ExecForceStoreHeapTuple(tuple, base_slot, false);
	econtext->ecxt_scantuple = base_slot;
	/* check WHERE-clause if any */
	if (pts->base_quals)
	{
		ResetExprContext(econtext);
		if (!ExecQual(pts->base_quals, econtext))
			return 0;
	}

	/*
	 * Shortcut, if GpuJoin is not involved. (GpuScan or GpuPreAgg + GpuScan).
	 * This case does not have fallback_slot, and the fallback_proj directly
	 * transforms the base-tuple to the ss_ScanTupleSlot.
	 */
	if (pts->num_rels == 0)
	{
		TupleTableSlot *proj_slot;
		HeapTuple	proj_htup;
		bool		should_free;

		Assert(pts->fallback_slot == 0);
		proj_slot = ExecProject(pts->fallback_proj);
		proj_htup = ExecFetchSlotHeapTuple(proj_slot, false, &should_free);
		pgstromStoreFallbackTuple(pts, proj_htup);
		if (should_free)
			pfree(proj_htup);
		return 1;
	}

	/* Load the base tuple (depth-0) to the fallback slot */
	slot_getallattrs(base_slot);
	Assert(fallback_slot != NULL);
    ExecStoreAllNullTuple(fallback_slot);
	foreach (lc, pp_info->kvars_deflist)
	{
		codegen_kvar_defitem *kvdef = lfirst(lc);

		if (kvdef->kv_depth == 0 &&
			kvdef->kv_resno >= 1 &&
			kvdef->kv_resno <= base_slot->tts_nvalid)
		{
			int		dst = kvdef->kv_slot_id;
			int		src = kvdef->kv_resno - 1;

			fallback_slot->tts_isnull[dst] = base_slot->tts_isnull[src];
			fallback_slot->tts_values[dst] = base_slot->tts_values[src];
		}
	}
	econtext->ecxt_scantuple = fallback_slot;
	/* Run JOIN, if any */
	Assert(pts->h_kmrels);
	__execFallbackCpuJoinOneDepth(pts, 1);
	return (pts->fallback_index -  fallback_index_saved > 0);
}

static void
__execFallbackCpuJoinRightOuterOneDepth(pgstromTaskState *pts, int depth)
{
	pgstromTaskInnerState *istate = &pts->inners[depth-1];
	pgstromPlanInfo	   *pp_info = pts->pp_info;
	ExprContext		   *econtext = pts->css.ss.ps.ps_ExprContext;
	TupleTableSlot	   *fallback_slot = pts->fallback_slot;
	kern_expression	   *kexp_join_kvars_load = NULL;
	kern_multirels	   *h_kmrels = pts->h_kmrels;
	kern_data_store	   *kds_in = KERN_MULTIRELS_INNER_KDS(h_kmrels, depth-1);
	bool			   *oj_map = KERN_MULTIRELS_OUTER_JOIN_MAP(h_kmrels, depth-1);

	if (pp_info->kexp_load_vars_packed)
	{
		const kern_expression *temp = (const kern_expression *)
			VARDATA(pp_info->kexp_load_vars_packed);
		kexp_join_kvars_load = __PICKUP_PACKED_KEXP(temp, depth);
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
			ItemPointerData	t_self;

			if (!titem)
				continue;
			ItemPointerSetInvalid(&t_self);
			__execFallbackLoadVarsSlot(fallback_slot,
									   kexp_join_kvars_load,
									   kds_in,
									   &t_self,
									   &titem->htup);
		}
		if (istate->other_quals && !ExecQual(istate->other_quals, econtext))
			continue;
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
	/* turn on/off partition-wise gpujoin */
	DefineCustomBoolVariable("pg_strom.enable_partitionwise_gpujoin",
							 "Enables the use of partition-wise GpuJoin",
							 NULL,
							 &pgstrom_enable_partitionwise_gpujoin,
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
	if (!set_join_pathlist_next)
	{
		set_join_pathlist_next = set_join_pathlist_hook;
		set_join_pathlist_hook = XpuJoinAddCustomPath;
	}
}


/*
 * pgstrom_init_dpu_join
 */
void
pgstrom_init_dpu_join(void)
{
	/* turn on/off dpujoin */
	DefineCustomBoolVariable("pg_strom.enable_dpujoin",
							 "Enables the use of DpuJoin logic",
							 NULL,
							 &pgstrom_enable_dpujoin,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* turn on/off dpuhashjoin */
	DefineCustomBoolVariable("pg_strom.enable_dpuhashjoin",
							 "Enables the use of DpuHashJoin logic",
							 NULL,
							 &pgstrom_enable_dpuhashjoin,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* turn on/off dpugistindex */
	DefineCustomBoolVariable("pg_strom.enable_dpugistindex",
							 "Enables the use of DpuGistIndex logic",
							 NULL,
							 &pgstrom_enable_dpugistindex,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* turn on/off partition-wise dpujoin */
	DefineCustomBoolVariable("pg_strom.enable_partitionwise_dpujoin",
							 "Enables the use of partition-wise DpuJoin",
							 NULL,
							 &pgstrom_enable_partitionwise_dpujoin,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* setup path methods */
	memset(&dpujoin_path_methods, 0, sizeof(CustomPathMethods));
	dpujoin_path_methods.CustomName             = "DpuJoin";
	dpujoin_path_methods.PlanCustomPath         = PlanDpuJoinPath;

	/* setup plan methods */
	memset(&dpujoin_plan_methods, 0, sizeof(CustomScanMethods));
	dpujoin_plan_methods.CustomName             = "DpuJoin";
	dpujoin_plan_methods.CreateCustomScanState  = CreateDpuJoinState;
	RegisterCustomScanMethods(&dpujoin_plan_methods);

	/* setup exec methods */
	memset(&dpujoin_exec_methods, 0, sizeof(CustomExecMethods));
	dpujoin_exec_methods.CustomName             = "DpuJoin";
	dpujoin_exec_methods.BeginCustomScan        = pgstromExecInitTaskState;
	dpujoin_exec_methods.ExecCustomScan         = pgstromExecTaskState;
	dpujoin_exec_methods.EndCustomScan          = pgstromExecEndTaskState;
	dpujoin_exec_methods.ReScanCustomScan       = pgstromExecResetTaskState;
	dpujoin_exec_methods.EstimateDSMCustomScan  = pgstromSharedStateEstimateDSM;
	dpujoin_exec_methods.InitializeDSMCustomScan = pgstromSharedStateInitDSM;
	dpujoin_exec_methods.InitializeWorkerCustomScan = pgstromSharedStateAttachDSM;
	dpujoin_exec_methods.ShutdownCustomScan     = pgstromSharedStateShutdownDSM;
	dpujoin_exec_methods.ExplainCustomScan      = pgstromExplainTaskState;

	/* hook registration */
	if (!set_join_pathlist_next)
	{
		set_join_pathlist_next = set_join_pathlist_hook;
		set_join_pathlist_hook = XpuJoinAddCustomPath;
	}
}
