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
static int					__pinned_inner_buffer_threshold_mb = 0; /* GUC */
static int					__pinned_inner_buffer_partition_size_mb = 0; /* GUC */
static CustomPathMethods	dpujoin_path_methods;
static CustomScanMethods	dpujoin_plan_methods;
static CustomExecMethods	dpujoin_exec_methods;
static bool					pgstrom_enable_dpujoin = false;		/* GUC */
static bool					pgstrom_enable_dpuhashjoin = false;	/* GUC */
static bool					pgstrom_enable_dpugistindex = false;/* GUC */
static bool					pgstrom_enable_partitionwise_dpujoin = false;

static bool					pgstrom_debug_xpujoinpath = false;

/*
 * DEBUG_XpuJoinPath
 */
static inline void
__appendRangeTblEntry(StringInfo buf,
					  Index rtindex,
					  RangeTblEntry *rte)
{
	if (rte->rtekind == RTE_RELATION)
	{
		char   *relname = get_rel_name(rte->relid);

		appendStringInfo(buf, "%s", relname);
		if (rte->eref &&
			rte->eref->aliasname &&
			strcmp(relname, rte->eref->aliasname) != 0)
			appendStringInfo(buf, "[%s]", rte->eref->aliasname);
	}
	else
	{
		const char *label;

		switch (rte->rtekind)
		{
			case RTE_SUBQUERY:  label = "subquery";       break;
			case RTE_JOIN:      label = "join";           break;
			case RTE_FUNCTION:  label = "function";       break;
			case RTE_TABLEFUNC: label = "table-function"; break;
			case RTE_VALUES:    label = "values-list";    break;
			case RTE_CTE:       label = "cte";            break;
			case RTE_NAMEDTUPLESTORE: label = "tuplestore"; break;
			case RTE_RESULT:    label = "result";         break;
			default:            label = "unknown";        break;
		}
		if (rte->eref &&
			rte->eref->aliasname)
			appendStringInfo(buf, "[%s:%s]", label, rte->eref->aliasname);
		else
			appendStringInfo(buf, "[%s:%u]", label, rtindex);
	}
}

static void
DEBUG_XpuJoinPathPrint(PlannerInfo *root,
					   const char *custom_name,
					   const Path *path,
					   RelOptInfo *outer_rel,
					   RelOptInfo *inner_rel)
{
	if (pgstrom_debug_xpujoinpath)
	{
		StringInfoData buf;
		int		i, count;

		initStringInfo(&buf);
		appendStringInfo(&buf, "%s: outer=(", custom_name);
		for (i = bms_next_member(outer_rel->relids, -1), count=0;
			 i >= 0;
			 i = bms_next_member(outer_rel->relids, i), count++)
		{
			RangeTblEntry *rte = root->simple_rte_array[i];
			if (count > 0)
				appendStringInfo(&buf, ", ");
			__appendRangeTblEntry(&buf, i, rte);
		}
		appendStringInfo(&buf, ") inner=(");
		for (i = bms_next_member(outer_rel->relids, -1), count=0;
			 i >= 0;
			 i = bms_next_member(outer_rel->relids, i), count++)
		{
			RangeTblEntry *rte = root->simple_rte_array[i];
			if (count > 0)
				appendStringInfo(&buf, ", ");
			__appendRangeTblEntry(&buf, i, rte);
		}
		appendStringInfo(&buf, ") parallel=%d cost=%.2f nrows=%.0f",
						 (int)path->parallel_aware,
						 path->total_cost,
						 path->rows);
		elog(NOTICE, "%s", buf.data);
		pfree(buf.data);
	}
}

/*
 * pgstrom_is_gpujoin_path
 */
bool
pgstrom_is_gpujoin_path(const Path *path)
{
	if (IsA(path, CustomPath))
	{
		const CustomPath *cpath = (const CustomPath *)path;

		if (cpath->methods == &gpujoin_path_methods)
			return true;
	}
	return false;
}

/*
 * pgstrom_is_gpujoin_plan
 */
bool
pgstrom_is_gpujoin_plan(const Plan *plan)
{
	if (IsA(plan, CustomScan))
	{
		const CustomScan *cscan = (const CustomScan *)plan;

		if (cscan->methods == &gpujoin_plan_methods)
			return true;
	}
	return false;
}

/*
 * pgstrom_is_gpujoin_state
 */
bool
pgstrom_is_gpujoin_state(const PlanState *ps)
{
	if (IsA(ps, CustomScanState))
	{
		const CustomScanState *css = (const CustomScanState *)ps;

		if (css->methods == &gpujoin_exec_methods)
			return true;
	}
	return false;
}

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
 * tryPinnedInnerJoinBufferPath
 */
static Path *
tryPinnedInnerJoinBufferPath(pgstromPlanInfo *pp_info,
							 pgstromPlanInnerInfo *pp_inner,
							 Path *inner_path,
							 Cost *p_inner_final_cost)
{
	PathTarget *inner_target;
	size_t		inner_threshold_sz;
	int			nattrs;
	int			unitsz;
	int			projection_hash_divisor = 0;
	double		bufsz;

	/*
	 * should not have RIGHT/FULL OUTER JOIN before pinned inner buffer
	 * (including the current-depth itself)
	 */
	for (int j=0; j < pp_info->num_rels; j++)
	{
		pgstromPlanInnerInfo *__pp_inner = &pp_info->inners[j];

		if (__pp_inner->join_type == JOIN_RIGHT ||
			__pp_inner->join_type == JOIN_FULL)
			return NULL;
		Assert(__pp_inner->join_type == JOIN_INNER ||
			   __pp_inner->join_type == JOIN_LEFT);
	}
	/*
	 * GiST-index buffer must be built by CPU
	 */
	if (OidIsValid(pp_inner->gist_index_oid))
		return NULL;
	/*
	 * Check expected pinned inner buffer size
	 */
	if (__pinned_inner_buffer_threshold_mb <= 0)
		return NULL;
	inner_threshold_sz = (size_t)__pinned_inner_buffer_threshold_mb << 20;

	inner_target = (inner_path->pathtarget
					? inner_path->pathtarget
					: inner_path->parent->reltarget);
	nattrs = list_length(inner_target->exprs);
	unitsz = ((pp_inner->hash_inner_keys != NIL
			   ? offsetof(kern_hashitem, t.htup)
			   : offsetof(kern_tupitem, htup)) +
			  MAXALIGN(offsetof(HeapTupleHeaderData,
								t_bits) + BITMAPLEN(nattrs)) +
			  MAXALIGN(inner_target->width));
	bufsz = MAXALIGN(offsetof(kern_data_store, colmeta[nattrs]));
	if (pp_inner->hash_inner_keys != NIL)
		bufsz += sizeof(uint64_t) * Max(inner_path->rows, 320.0);
	bufsz += sizeof(uint64_t) * inner_path->rows;
	bufsz += unitsz * inner_path->rows;

	if (bufsz < inner_threshold_sz)
		return NULL;
	/* Ok, this inner path can use pinned-buffer */
	if (pgstrom_is_gpuscan_path(inner_path) ||
		pgstrom_is_gpujoin_path(inner_path))
	{
		CustomPath *cpath = (CustomPath *)pgstrom_copy_pathnode(inner_path);
		pgstromPlanInfo *pp_temp = linitial(cpath->custom_private);

		pp_temp = copy_pgstrom_plan_info(pp_temp);
		pp_temp->projection_hashkeys = pp_inner->hash_inner_keys;
		cpath->custom_private = list_make1(pp_temp);

		/* turn on inner_pinned_buffer */
		pp_inner->inner_pinned_buffer = true;
		pp_inner->inner_partitions_divisor = projection_hash_divisor;

		*p_inner_final_cost = pp_temp->final_cost;
		return (Path *)cpath;
	}
#if 0
	else if (IsA(inner_path, Path) &&
			 inner_path->pathtype == T_SeqScan &&
			 inner_path->pathkeys == NIL)
	{
		//TODO: Try to replace a simple SeqScan to GpuScan if possible
	}
#endif
	return NULL;
}

/*
 * fixup_join_varnullingrels
 *
 * MEMO: PG16 added the Var::varnullingrels field to track potentially
 * NULL-able columns for OUTER JOINs. It it setup by the core optimizer
 * for each joinrel, so expression pulled-up from the prior level join
 * or scan must be adjusted as if Var-nodes are used in this Join.
 * Without this fixup, setrefs.c shall raise an error due to mismatch
 * of equal() that checks varnullingrels field also.
 */
#if PG_VERSION_NUM < 160000
#define fixup_join_varnullingrels(joinrel,pp_info)		(pp_info)
#else
static Node *
__fixup_join_varnullingrels_walker(Node *node, void *__data)
{
	RelOptInfo *joinrel = __data;
	PathTarget *reltarget = joinrel->reltarget;
	ListCell   *lc;

	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		Var	   *var = copyObject((Var *)node);

		foreach (lc, reltarget->exprs)
		{
			Var	   *__var = lfirst(lc);

			if (var->varno == __var->varno &&
				var->varattno == __var->varattno)
			{
				Assert(var->vartype   == __var->vartype &&
					   var->vartypmod == __var->vartypmod &&
					   var->varcollid == __var->varcollid);
				var->varnullingrels = bms_copy(__var->varnullingrels);
				return (Node *)var;
			}
		}
		var->varnullingrels = NULL;
		return (Node *)var;
	}
	return expression_tree_mutator(node, __fixup_join_varnullingrels_walker, __data);
}

static pgstromPlanInfo *
fixup_join_varnullingrels(RelOptInfo *joinrel, pgstromPlanInfo *pp_info)
{
#define __FIXUP_FIELD(VAL)												\
	VAL = (void *)__fixup_join_varnullingrels_walker((Node *)(VAL), joinrel)

	__FIXUP_FIELD(pp_info->used_params);
	__FIXUP_FIELD(pp_info->host_quals);
	__FIXUP_FIELD(pp_info->scan_quals);
	__FIXUP_FIELD(pp_info->brin_index_conds);
	__FIXUP_FIELD(pp_info->brin_index_quals);
	for (int i=0; i < pp_info->num_rels; i++)
	{
		pgstromPlanInnerInfo *pp_inner = &pp_info->inners[i];

		__FIXUP_FIELD(pp_inner->hash_outer_keys);
		__FIXUP_FIELD(pp_inner->hash_inner_keys);
		__FIXUP_FIELD(pp_inner->join_quals);
		__FIXUP_FIELD(pp_inner->other_quals);
		__FIXUP_FIELD(pp_inner->gist_clause);
	}
#undef __FIXUP_FIELD
	return pp_info;
}
#endif

/*
 * __buildXpuJoinPlanInfo
 */
static pgstromPlanInfo *
__buildXpuJoinPlanInfo(PlannerInfo *root,
					   RelOptInfo *joinrel,
					   JoinType join_type,
					   List *restrict_clauses,
					   pgstromOuterPathLeafInfo *op_prev,
					   List *inner_paths_list,
					   int sibling_param_id,
					   double inner_discount_ratio)
{
	pgstromPlanInfo *pp_prev = op_prev->pp_info;
	pgstromPlanInfo *pp_info;
	pgstromPlanInnerInfo *pp_inner;
	Path		   *inner_path = llast(inner_paths_list);
	Path		   *temp_path;
	RelOptInfo	   *inner_rel = inner_path->parent;
	RelOptInfo	   *outer_rel = op_prev->leaf_rel;
	Cardinality		outer_nrows;
	Cardinality		inner_nrows;
	Cost			startup_cost;
	Cost			inner_cost;
	Cost			run_cost;
	Cost			final_cost;
	Cost			comp_cost = 0.0;
	Cost			inner_final_cost = 0.0;
	bool			enable_xpuhashjoin;
	bool			enable_xpugistindex;
	double			xpu_tuple_cost;
	Cost			xpu_operator_cost;
	Cost			xpu_ratio;
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
		xpu_operator_cost   = cpu_operator_cost * pgstrom_gpu_operator_ratio();
		xpu_ratio           = pgstrom_gpu_operator_ratio();
	}
	else if ((pp_prev->xpu_task_flags & DEVKIND__ANY) == DEVKIND__NVIDIA_DPU)
	{
		enable_xpuhashjoin  = pgstrom_enable_dpuhashjoin;
		enable_xpugistindex = pgstrom_enable_dpugistindex;
		xpu_tuple_cost      = pgstrom_dpu_tuple_cost;
		xpu_operator_cost   = cpu_operator_cost * pgstrom_dpu_operator_ratio();
		xpu_ratio           = pgstrom_dpu_operator_ratio();
	}
	else
	{
		elog(ERROR, "Bug? unexpected xpu_task_flags: %08x",
			 pp_prev->xpu_task_flags);
	}

	/* setup inner_target_list */
	foreach (lc, inner_paths_list)
	{
		Path   *i_path = lfirst(lc);
		inner_target_list = lappend(inner_target_list, i_path->pathtarget);
	}

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
	pp_info->xpu_task_flags &= ~DEVTASK__MASK;
	pp_info->xpu_task_flags |= DEVTASK__JOIN;
	pp_info->sibling_param_id = sibling_param_id;

	pp_inner = &pp_info->inners[pp_info->num_rels++];
	pp_inner->join_type = join_type;
	pp_inner->hash_outer_keys = hash_outer_keys;
	pp_inner->hash_inner_keys = hash_inner_keys;
	pp_inner->join_quals = join_quals;
	pp_inner->other_quals = other_quals;
	/* GiST-Index availability checks */
	if (enable_xpugistindex &&
		hash_outer_keys == NIL &&
		hash_inner_keys == NIL)
	{
		Path   *orig_inner_path = llast(inner_paths_list);
		Path   *gist_inner_path
			= pgstromTryFindGistIndex(root,
									  orig_inner_path,
									  restrict_clauses,
									  pp_prev->xpu_task_flags,
									  pp_prev->scan_relid,
									  inner_target_list,
									  pp_inner);
		if (gist_inner_path)
			llast(inner_paths_list) = inner_path = gist_inner_path;
	}
	/*
	 * Try pinned inner buffer
	 */
	temp_path = tryPinnedInnerJoinBufferPath(pp_info,
											 pp_inner,
											 inner_path,
											 &inner_final_cost);
	if (temp_path)
		llast(inner_paths_list) = inner_path = temp_path;

	/*
	 * Cost estimation
	 */
	startup_cost = pp_prev->startup_cost;
	inner_cost   = pp_prev->inner_cost;
	run_cost     = pp_prev->run_cost;
	final_cost   = 0.0;
	outer_nrows  = PP_INFO_NUM_ROWS(pp_prev);

	/*
	 * Cost for inner-setup
	 */
	inner_nrows = inner_path->rows;
	if (inner_path->parallel_aware)
	{
		double		divisor = inner_path->parallel_workers;
		double		leader_contribution;

		if (parallel_leader_participation)
		{
			leader_contribution = 1.0 - (0.3 * inner_path->parallel_workers);
			if (leader_contribution > 0.0)
				divisor += leader_contribution;
		}
		inner_nrows *= divisor;
	}
	inner_cost += inner_path->total_cost;
	if (pp_inner->inner_pinned_buffer)
		inner_cost -= inner_final_cost * inner_discount_ratio;
	else
		inner_cost += cpu_tuple_cost * inner_nrows * inner_discount_ratio;

	/*
	 * Cost for join_quals
	 */
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
		startup_cost += (pp_inner->inner_pinned_buffer
						 ? xpu_operator_cost
						 : cpu_operator_cost) * num_hashkeys * inner_path->rows;
		/* cost to comput hash value by GPU */
		comp_cost += xpu_operator_cost * num_hashkeys * outer_nrows;
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
		if (!pp_inner->inner_pinned_buffer)
			startup_cost += cpu_tuple_cost * inner_path->rows;

		/* cost to evaluate join qualifiers by GPU */
		comp_cost += (join_quals_cost.per_tuple * xpu_ratio *
					  inner_path->rows *
					  outer_nrows);
	}
	/* discount if CPU parallel is enabled */
	run_cost += (comp_cost / pp_info->parallel_divisor);
	/* cost for DMA receive (xPU --> Host) */
	final_cost += (xpu_tuple_cost * joinrel->rows) / pp_info->parallel_divisor;
	/* cost for host projection */
	final_cost += (joinrel->reltarget->cost.per_tuple *
				   joinrel->rows / pp_info->parallel_divisor);

	pp_info->startup_cost = startup_cost;
	pp_info->inner_cost = inner_cost;
	pp_info->run_cost = run_cost;
	pp_info->final_cost = final_cost;
	pp_info->final_nrows = joinrel->rows;
	pp_inner->join_nrows = clamp_row_est(joinrel->rows / pp_info->parallel_divisor);

	return fixup_join_varnullingrels(joinrel, pp_info);
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
							int sibling_param_id,
							double inner_discount_ratio,
							uint32_t xpu_task_flags,
							const CustomPathMethods *xpujoin_path_methods)
{
	pgstromPlanInfo *pp_info;
	Relids			required_outer;
	ParamPathInfo  *param_info;
	CustomPath	   *cpath;
	List		   *inner_paths_list;

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
	inner_paths_list = list_copy(op_prev->inner_paths_list);
	inner_paths_list = lappend(inner_paths_list, inner_path);
	pp_info = __buildXpuJoinPlanInfo(root,
									 join_rel,
									 join_type,
									 restrict_clauses,
									 op_prev,
									 inner_paths_list,
									 sibling_param_id,
									 inner_discount_ratio);
	if (!pp_info)
		return NULL;

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
	cpath->path.rows = PP_INFO_NUM_ROWS(pp_info);
	cpath->path.startup_cost = (pp_info->startup_cost +
								pp_info->inner_cost);
	cpath->path.total_cost = (pp_info->startup_cost +
							  pp_info->inner_cost +
							  pp_info->run_cost +
							  pp_info->final_cost);
	cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
	cpath->methods = xpujoin_path_methods;
	Assert(list_length(inner_paths_list) == pp_info->num_rels);
	cpath->custom_paths = inner_paths_list;
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

	op_prev = pgstrom_find_op_normal(root, outer_rel, try_parallel_path);
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
										-1,		/* sibling_param_id */
										1.0,	/* inner discount ratio */
										xpu_task_flags,
										xpujoin_path_methods);
	if (!cpath)
		return;
	/* register the XpuJoinPath */
	DEBUG_XpuJoinPathPrint(root,
						   xpujoin_path_methods->CustomName,
						   &cpath->path,
						   outer_rel,
						   inner_path->parent);
	pgstrom_remember_op_normal(root,
							   join_rel,
							   op_leaf,
							   try_parallel_path);
	if (!try_parallel_path)
		add_path(join_rel, &cpath->path);
	else
		add_partial_path(join_rel, &cpath->path);
}

/*
 * try_add_sorted_gpujoin_path
 */
#define LOG2(x)		(log(x) / 0.693147180559945)

void
try_add_sorted_gpujoin_path(PlannerInfo *root,
							RelOptInfo *join_rel,
							CustomPath *cpath,
							bool be_parallel)
{
	pgstromPlanInfo *pp_info = linitial(cpath->custom_private);
	PathTarget *final_target = cpath->path.pathtarget;
	List	   *sortkeys_upper = NIL;
	List	   *sortkeys_expr = NIL;
	List	   *sortkeys_kind = NIL;
	List	   *inner_target_list = NIL;
	ListCell   *lc1, *lc2;
	size_t		unitsz, buffer_sz, devmem_sz;
	int			nattrs;
	Cost		gpusort_cost;

	if (!pgstrom_enable_gpusort)
	{
		elog(DEBUG1, "gpusort: disabled by pg_strom.enable_gpusort");
		return;
	}
	if (pgstrom_cpu_fallback_elevel < ERROR)
	{
		elog(DEBUG1, "gpusort: disabled by pgstrom.cpu_fallback");
		return;
	}
	if ((pp_info->xpu_task_flags & DEVKIND__NVIDIA_GPU) == 0)
	{
		elog(DEBUG1, "gpusort: disabled, because only GPUs are supported (flags: %08x)",
			 pp_info->xpu_task_flags);
		return;		/* feture available on GPU only */
	}
	/* pick up upper sortkeys */
	if (root->window_pathkeys != NIL)
        sortkeys_upper = root->window_pathkeys;
    else if (root->distinct_pathkeys != NIL)
        sortkeys_upper = root->distinct_pathkeys;
    else if (root->sort_pathkeys != NIL)
        sortkeys_upper = root->sort_pathkeys;
    else if (root->query_pathkeys != NIL)
        sortkeys_upper = root->query_pathkeys;
	else
	{
		elog(DEBUG1, "gpusort: disabled because no sortable pathkeys");
		return;		/* no upper sortkeys */
	}

	/*
	 * buffer size estimation, because GPU-Sort needs kds_final buffer to save
	 * the result of GPU-Projection until Bitonic-sorting.
	 */
	nattrs = list_length(final_target->exprs);
	unitsz = offsetof(kern_tupitem, htup) +
		MAXALIGN(offsetof(HeapTupleHeaderData,
						  t_bits) + BITMAPLEN(nattrs)) +
		MAXALIGN(final_target->width);
	buffer_sz = MAXALIGN(offsetof(kern_data_store, colmeta[nattrs])) +
		sizeof(uint64_t) * pp_info->final_nrows +
		unitsz * pp_info->final_nrows;
	devmem_sz = GetGpuMinimalDeviceMemorySize();
	if (buffer_sz > devmem_sz)
	{
		elog(DEBUG1, "gpusort: disabled by too large final buffer (expected: %s, physical: %s)",
			 format_bytesz(buffer_sz),
			 format_bytesz(devmem_sz));
		return;		/* too large */
	}
	/* preparation for pgstrom_xpu_expression */
	foreach (lc1, cpath->custom_paths)
	{
		inner_target_list = lappend(inner_target_list,
									((Path *)lfirst(lc1))->pathtarget);
	}
	/* check whether the sorting key is supported */
	foreach (lc1, sortkeys_upper)
	{
		PathKey	   *pk = lfirst(lc1);
		EquivalenceClass *ec = pk->pk_eclass;
		EquivalenceMember *em;
		Expr	   *em_expr;
		devtype_info *dtype;
		bool		found = false;

		if (list_length(ec->ec_members) != 1 ||
			ec->ec_sources != NIL ||
			ec->ec_derives != NIL)
			return;		/* not supported */

		/* strip Relabel for equal() comparison */
		em = (EquivalenceMember *)linitial(ec->ec_members);
		for (em_expr = em->em_expr;
			 IsA(em_expr, RelabelType);
			 em_expr = ((RelabelType *)em_expr)->arg);
		/* check whether the em_expr is fully executable on device */
		if (!pgstrom_xpu_expression(em_expr,
									pp_info->xpu_task_flags,
									pp_info->scan_relid,
									inner_target_list, NULL))
			return;		/* not supported */
		dtype = pgstrom_devtype_lookup(exprType((Node *)em_expr));
		if (!dtype || (dtype->type_flags & DEVTYPE__HAS_COMPARE) == 0)
			return;		/* not supported */
		/* lookup the sorting keys */
		foreach (lc2, final_target->exprs)
		{
			Node   *f_expr = lfirst(lc2);

			if (equal(f_expr, em_expr))
			{
				int		kind = KSORT_KEY_KIND__VREF;

				if (pk->pk_nulls_first)
					kind |= KSORT_KEY_ATTR__NULLS_FIRST;
				if (pk->pk_strategy == BTLessStrategyNumber)
					kind |= KSORT_KEY_ATTR__ORDER_ASC;
				else if (pk->pk_strategy != BTLessStrategyNumber)
					return;		/* should not happen */
				sortkeys_expr = lappend(sortkeys_expr, f_expr);
				sortkeys_kind = lappend_int(sortkeys_kind, kind);
				found = true;
				break;
			}
		}
		if (!found)
			return;		/* not found */
	}
	/* duplicate GpuScan/GpuJoin path and attach GPU-Sort */
	gpusort_cost = (2.0 * pgstrom_gpu_operator_cost *
					cpath->path.rows *
					LOG2(cpath->path.rows));
	cpath = (CustomPath *)pgstrom_copy_pathnode(&cpath->path);
	pp_info = copy_pgstrom_plan_info(pp_info);
	pp_info->gpusort_keys_expr = sortkeys_expr;
	pp_info->gpusort_keys_kind = sortkeys_kind;
	linitial(cpath->custom_private) = pp_info;
	cpath->path.startup_cost += gpusort_cost;
	cpath->path.total_cost   += gpusort_cost;

	/* add path */
	if (!be_parallel)
		add_path(join_rel, &cpath->path);
	else
	{
		GatherPath *gpath = create_gather_path(root,
											   cpath->path.parent,
											   &cpath->path,
											   cpath->path.pathtarget,
											   NULL,
											   &cpath->path.rows);
		gpath->path.pathkeys = sortkeys_upper;

		add_path(join_rel, &gpath->path);
	}
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
static AppendRelInfo *
__build_fake_apinfo_non_relations(PlannerInfo *root, Index rtindex)
{
	AppendRelInfo *apinfo = makeNode(AppendRelInfo);
	RelOptInfo *rel = root->simple_rel_array[rtindex];
	PathTarget *reltarget = rel->reltarget;
	List	   *trans_vars = NIL;
	ListCell   *lc;
	AttrNumber	attno = 1;
	AttrNumber *colnos
		= palloc0(sizeof(AttrNumber) * list_length(reltarget->exprs));

	foreach (lc, reltarget->exprs)
	{
		Node   *node = lfirst(lc);
		Var	   *var;

		var = makeVar(rtindex,
					  attno++,
					  exprType(node),
					  exprTypmod(node),
					  exprCollation(node),
					  0);
		trans_vars = lappend(trans_vars, var);
		colnos[attno-1] = attno;
	}
	apinfo->translated_vars = trans_vars;
	apinfo->num_child_cols = attno;
	apinfo->parent_colnos = colnos;

	return apinfo;
}

static AppendRelInfo **
__make_fake_apinfo_array(PlannerInfo *root,
						 RelOptInfo *parent_joinrel,
						 RelOptInfo *outer_rel,
						 RelOptInfo *inner_rel)
{
	AppendRelInfo **ap_info_array;
	Relids	__relids;
	int		i;

	ap_info_array = palloc0(sizeof(AppendRelInfo *) *
							root->simple_rel_array_size);
	__relids = bms_union(outer_rel->relids,
						 inner_rel->relids);
	for (i = bms_next_member(__relids, -1);
		 i >= 0;
		 i = bms_next_member(__relids, i))
	{
		AppendRelInfo  *ap_info = root->append_rel_array[i];

		if (ap_info)
		{
			bool	rebuild = false;

			while (!bms_is_member(ap_info->parent_relid,
								  parent_joinrel->relids))
			{
				Index	curr_child = ap_info->parent_relid;

				ap_info = NULL;
				for (int j=0; j < root->simple_rel_array_size; j++)
				{
					AppendRelInfo *__ap_info = root->append_rel_array[j];

					if (__ap_info &&
						__ap_info->child_relid == curr_child)
					{
						ap_info = root->append_rel_array[j];
						break;
					}
				}
				if (!ap_info)
					elog(ERROR, "Bug? AppendRelInfo chain is not linked");
				rebuild = true;
			}

			if (rebuild)
			{
				Index			parent_relid = ap_info->parent_relid;
				RangeTblEntry  *rte_child = root->simple_rte_array[i];
				RangeTblEntry  *rte_parent = root->simple_rte_array[parent_relid];
				Relation		rel_child;
				Relation		rel_parent;

				if (rte_child->rtekind != RTE_RELATION ||
					rte_parent->rtekind != RTE_RELATION)
					elog(ERROR, "Bug? not a relation has partition leaf");
				rel_child = relation_open(rte_child->relid, NoLock);
				rel_parent = relation_open(rte_parent->relid, NoLock);

				ap_info = make_append_rel_info(rel_parent,
											   rel_child,
											   parent_relid, i);
				relation_close(rel_child, NoLock);
				relation_close(rel_parent, NoLock);
			}
		}
		else
		{
			RangeTblEntry  *rte = root->simple_rte_array[i];

			if (rte->rtekind != RTE_RELATION)
				ap_info = __build_fake_apinfo_non_relations(root, i);
			else
			{
				Relation	rel = relation_open(rte->relid, NoLock);

				ap_info = make_append_rel_info(rel, rel, i, i);

				relation_close(rel, NoLock);
			}
		}
		ap_info_array[i] = ap_info;
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
															  parent_joinrel,
															  outer_rel,
															  inner_rel);
			leaf_joinrel = build_child_join_rel(root,
												outer_rel,
												inner_rel,
												parent_joinrel,
												restrictlist,
												sjinfo);
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
	Path	   *append_path;
	Relids		required_outer = NULL;
	int			parallel_nworkers = 0;
	double		total_nrows = 0.0;
	bool		identical_inners;
	int			sibling_param_id = -1;
	double		inner_discount_ratio = 1.0;
	ListCell   *lc;

	op_prev_list = pgstrom_find_op_leafs(root,
										 outer_rel,
										 try_parallel_path,
										 &identical_inners);
	if (identical_inners)
	{
		PlannerGlobal  *glob = root->glob;

		sibling_param_id = list_length(glob->paramExecTypes);
		glob->paramExecTypes = lappend_oid(glob->paramExecTypes,
										   INTERNALOID);
		if (list_length(op_prev_list) > 1)
			inner_discount_ratio = 1.0 / (double)list_length(op_prev_list);
	}
	Assert(inner_discount_ratio >= 0.0 && inner_discount_ratio <= 1.0);

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
											sibling_param_id,
											inner_discount_ratio,
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
	DEBUG_XpuJoinPathPrint(root,
						   xpujoin_path_methods->CustomName,
						   append_path,
						   outer_rel,
						   inner_path->parent);
	pgstrom_remember_op_leafs(root,
							  join_rel,
							  op_leaf_list,
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
		if (!inner_path && IS_SIMPLE_REL(innerrel) && innerrel->rtekind == RTE_RELATION)
		{
			RangeTblEntry  *rte = root->simple_rte_array[innerrel->relid];

			/*
			 * In case when inner relation is very small, PostgreSQL may
			 * skip to generate partial scan paths because it may calculate
			 * the number of parallel workers zero due to small size.
			 * Only if the innerrel is base relation, we add a partial
			 * SeqScan path to use parallel inner path.
			 */
			Assert(innerrel->relid < root->simple_rel_array_size);
			if (rte->relkind == RELKIND_RELATION)
			{
				inner_path = (Path *)
					create_seqscan_path(root,
										innerrel,
										innerrel->lateral_relids,
										try_parallel);
			}
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
		if (!joinrel->consider_parallel)
			break;
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
	RelOptInfo *parent;
	List	   *op_leaf_list = NIL;

	parent = find_join_rel(root, joinrel->top_parent_relids);
	if (parent == NULL ||
		parent->nparts == 0 ||
		parent->part_rels[parent->nparts-1] != joinrel)
		return;
	for (int k=0; k < parent->nparts; k++)
	{
		RelOptInfo *leaf_rel = parent->part_rels[k];
		pgstromOuterPathLeafInfo *op_leaf;

		op_leaf = pgstrom_find_op_normal(root, leaf_rel, be_parallel);
		if (!op_leaf)
			return;
		op_leaf_list = lappend(op_leaf_list, op_leaf);
	}
	pgstrom_remember_op_leafs(root,
							  parent,
							  op_leaf_list,
							  be_parallel);
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
		if (pgstrom_enable_gpujoin && gpuserv_ready_accept())
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
		if (joinrel->reloptkind == RELOPT_OTHER_JOINREL)
		{
			__xpuJoinTryAddPartitionLeafs(root, joinrel, false);
			__xpuJoinTryAddPartitionLeafs(root, joinrel, true);
		}
	}
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

	if (!node || tlist_member((Expr *)node, tlist_dev))
		return tlist_dev;

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
			if (equal(node, lfirst(lc2)))
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
														   context->xpu_task_flags,
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
														   context->xpu_task_flags,
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
								List *tlist,
								List *groupby_actions)
{
	ListCell   *lc1, *lc2;

	context->tlist_dev = copyObject(tlist);
	forboth (lc1, tlist,
			 lc2, groupby_actions)
	{
		TargetEntry *tle = lfirst(lc1);
		int		action = lfirst_int(lc2);

		if (action == KAGG_ACTION__VREF ||
			action == KAGG_ACTION__VREF_NOKEY ||
			!IsA(tle->expr, FuncExpr))
		{
			Assert(tlist_member(tle->expr, context->tlist_dev));
		}
		else
		{
			FuncExpr   *f = (FuncExpr *)tle->expr;
			ListCell   *cell;

			foreach (cell, f->args)
			{
				Expr   *arg = lfirst(cell);
				int		resno = list_length(context->tlist_dev) + 1;

				if (!tlist_member(arg, context->tlist_dev))
				{
					TargetEntry *__tle = makeTargetEntry(arg, resno, NULL, true);

					context->tlist_dev = lappend(context->tlist_dev, __tle);
				}
			}
		}
	}
}


/*
 * build_explain_tlist_junks
 *
 * it builds junk TLEs for EXPLAIN output only
 */
static bool
__build_explain_tlist_junks_walker(Node *node, void *__priv)
{
	codegen_context *context = __priv;

	if (!node)
		return false;

	if (tlist_member((Expr *)node, context->tlist_dev) != NULL)
		return false;
	if (IsA(node, Var))
	{
		TargetEntry *tle;

		tle = makeTargetEntry((Expr *)node,
							  list_length(context->tlist_dev) + 1,
							  NULL,
							  true);
		context->tlist_dev = lappend(context->tlist_dev, tle);
		return false;
	}
	return expression_tree_walker(node, __build_explain_tlist_junks_walker, __priv);
}

static void
build_explain_tlist_junks(codegen_context *context,
						  pgstromPlanInfo *pp_info,
						  const Bitmapset *outer_refs)
{
	List   *vars_in_exprs = pull_vars_of_level((Node *)context->tlist_dev, 0);

	__build_explain_tlist_junks_walker((Node *)pp_info->used_params, context);
	__build_explain_tlist_junks_walker((Node *)pp_info->host_quals, context);
	__build_explain_tlist_junks_walker((Node *)pp_info->scan_quals, context);

	for (int i=0; i < pp_info->num_rels; i++)
	{
		pgstromPlanInnerInfo *pp_inner = &pp_info->inners[i];

		__build_explain_tlist_junks_walker((Node *)pp_inner->hash_outer_keys,
										   context);
		__build_explain_tlist_junks_walker((Node *)pp_inner->hash_inner_keys,
										   context);
		__build_explain_tlist_junks_walker((Node *)pp_inner->join_quals,
										   context);
		__build_explain_tlist_junks_walker((Node *)pp_inner->other_quals,
										   context);
		__build_explain_tlist_junks_walker((Node *)pp_inner->gist_clause,
										   context);
	}
	__build_explain_tlist_junks_walker((Node *)vars_in_exprs, context);
	
	/* sanity checks for outer references */
#ifdef USE_ASSERT_CHECKING
	for (int j = bms_next_member(outer_refs, -1);
		 j >= 0;
		 j = bms_next_member(outer_refs, j))
	{
		int			anum = j + FirstLowInvalidHeapAttributeNumber;
		ListCell   *lc;

		foreach (lc, context->tlist_dev)
		{
			TargetEntry *tle = lfirst(lc);

			if (IsA(tle->expr, Var))
			{
				Var	   *var = (Var *)tle->expr;

				if (var->varno == pp_info->scan_relid &&
					var->varattno == anum)
					break;
			}
		}
		if (lc == NULL)
			elog(INFO, "scan_relid=%d anum=%d tlist_dev=%s", pp_info->scan_relid, anum, nodeToString(context->tlist_dev));
		Assert(lc != NULL);
	}
#endif
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

	Assert(pp_info->num_rels == list_length(custom_plans));
	context = create_codegen_context(root, cpath, pp_info);

	/* codegen for outer scan, if any */
	if (pp_info->scan_quals)
	{
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
		join_quals_stacked = lappend(join_quals_stacked,
									 pp_inner->join_quals);
		pull_varattnos((Node *)pp_inner->join_quals,
					   pp_info->scan_relid,
					   &outer_refs);
		other_quals_stacked = lappend(other_quals_stacked,
									  pp_inner->other_quals);
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
		Relids	leaf_relids = cpath->path.parent->relids;
		tlist = fixup_expression_by_partition_leaf(root, leaf_relids, tlist);
		pgstrom_build_groupby_tlist_dev(context, root, tlist,
										pp_info->groupby_actions);
		codegen_build_groupby_actions(context, pp_info);
	}
	else
	{
		/* build device projection */
		List   *proj_hash = pp_info->projection_hashkeys;

		pgstrom_build_join_tlist_dev(context, root, joinrel, tlist);
		pp_info->kexp_projection = codegen_build_projection(context,
															proj_hash);
	}
	pull_varattnos((Node *)context->tlist_dev,
				   pp_info->scan_relid,
				   &outer_refs);
	build_explain_tlist_junks(context, pp_info, outer_refs);
	/* attach GPU-Sort key definitions, if any */
	pp_info->kexp_gpusort_keydesc = codegen_build_gpusort_keydesc(context, pp_info);
	
	/* assign remaining PlanInfo members */
	pp_info->kexp_join_quals_packed
		= codegen_build_packed_joinquals(context,
										 join_quals_stacked,
										 other_quals_stacked);
	pp_info->kexp_hash_keys_packed
		= codegen_build_packed_hashkeys(context,
										hash_keys_stacked);
	codegen_build_packed_gistevals(context, pp_info);
	/* LoadVars for each depth */
	codegen_build_packed_kvars_load(context, pp_info);
	/* MoveVars for each depth (only GPUs) */
	codegen_build_packed_kvars_move(context, pp_info);
	/* xpu_task_flags should not be cleared in codege.c */
	Assert((context->xpu_task_flags &
			pp_info->xpu_task_flags) == pp_info->xpu_task_flags);
	pp_info->kvars_deflist = context->kvars_deflist;
	pp_info->xpu_task_flags = context->xpu_task_flags;
	pp_info->extra_bufsz = context->extra_bufsz;
	pp_info->used_params = context->used_params;
	pp_info->outer_refs  = outer_refs;
	pp_info->cuda_stack_size = estimate_cuda_stack_size(context);
	/*
	 * build CustomScan
	 */
	cscan = makeNode(CustomScan);
	cscan->scan.plan.targetlist = tlist;
	cscan->scan.scanrelid = pp_info->scan_relid;
	cscan->flags = cpath->flags;
	cscan->methods = xpujoin_plan_methods;
	cscan->custom_plans = custom_plans;
	cscan->custom_scan_tlist = assign_custom_cscan_tlist(context->tlist_dev,
														 pp_info);
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
	Assert(cscan->methods == &gpujoin_plan_methods);
	return pgstromCreateTaskState(cscan, &gpujoin_exec_methods);
}

/*
 * CreateDpuJoinState
 */
static Node *
CreateDpuJoinState(CustomScan *cscan)
{
	Assert(cscan->methods == &dpujoin_plan_methods);
	return pgstromCreateTaskState(cscan, &dpujoin_exec_methods);
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
get_tuple_hashvalue(pgstromTaskState *pts,
					pgstromTaskInnerState *istate,
					TupleTableSlot *inner_slot)
{
	ExprContext	   *econtext = pts->css.ss.ps.ps_ExprContext;
	TupleTableSlot *scan_slot = pts->css.ss.ss_ScanTupleSlot;
	uint32_t		hash = 0xffffffffU;
	ListCell	   *lc1, *lc2;

	/* move to scan_slot from inner_slot */
	forboth (lc1, istate->inner_load_src,
			 lc2, istate->inner_load_dst)
	{
		int		src = lfirst_int(lc1) - 1;
		int		dst = lfirst_int(lc2) - 1;

		scan_slot->tts_isnull[dst] = inner_slot->tts_isnull[src];
		scan_slot->tts_values[dst] = inner_slot->tts_values[src];
	}
	/* calculation of a hash value of this entry */
	econtext->ecxt_scantuple = scan_slot;
	forboth (lc1, istate->hash_inner_keys,
			 lc2, istate->hash_inner_funcs)
	{
		ExprState	   *es = lfirst(lc1);
		devtype_hashfunc_f h_func = lfirst(lc2);
		Datum			datum;
		bool			isnull;

		datum = ExecEvalExpr(es, econtext, &isnull);
		hash = pg_hash_merge(hash, h_func(isnull, datum));
	}
	hash ^= 0xffffffffU;

	return hash;
}

/*
 * execInnerPreloadOneDepth
 */
static void
execInnerPreloadOneDepth(MemoryContext memcxt,
						 pgstromTaskState *pts,
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

	ExecStoreAllNullTuple(pts->css.ss.ss_ScanTupleSlot);
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
			uint32_t	hash = get_tuple_hashvalue(pts, istate, slot);

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
__innerPreloadSetupGiSTIndexWalker(Relation i_rel,
								   char *base,
								   BlockNumber blkno,
								   BlockNumber nblocks,
								   BlockNumber parent_blkno,
								   OffsetNumber parent_offno)
{
	while (blkno < nblocks)
	{
		Page			page = (Page)(base + BLCKSZ * blkno);
		PageHeader		hpage = (PageHeader) page;
		OffsetNumber	i, maxoff;

		Assert(hpage->pd_lsn.xlogid == InvalidBlockNumber &&
			   hpage->pd_lsn.xrecoff == InvalidOffsetNumber);
		hpage->pd_lsn.xlogid = parent_blkno;
		hpage->pd_lsn.xrecoff = parent_offno;
		if (!GistPageIsLeaf(page))
		{
			maxoff = PageGetMaxOffsetNumber(page);
			for (i = FirstOffsetNumber;
				 i <= maxoff;
				 i = OffsetNumberNext(i))
			{
				ItemId		iid = PageGetItemId(page, i);
				IndexTuple	it;
				BlockNumber	child;

				if (!ItemIdIsNormal(iid))
					continue;
				it = (IndexTuple) PageGetItem(page, iid);
				child = BlockIdGetBlockNumber(&it->t_tid.ip_blkid);
				if (child < nblocks)
					__innerPreloadSetupGiSTIndexWalker(i_rel,
													   base,
													   child,
													   nblocks,
													   blkno, i);
				else
					elog(ERROR, "GiST-Index '%s' may be corrupted: index-node %u of block %u dives into %u but out of the relation (nblocks=%u)",
						 RelationGetRelationName(i_rel),
						 i, blkno, child, nblocks);
			}
		}

		if (GistFollowRight(page))
			blkno = GistPageGetOpaque(page)->rightlink;
		else
			break;
	}
}

static void
innerPreloadSetupGiSTIndex(Relation i_rel, kern_data_store *kds_gist)
{
	char   *base = (char *)KDS_BLOCK_PGPAGE(kds_gist, 0);

	__innerPreloadSetupGiSTIndexWalker(i_rel, base,
									   0, kds_gist->nitems,
									   InvalidBlockNumber,
									   InvalidOffsetNumber);
}

/*
 * innerPreloadSetupPinnedInnerBufferPartitions
 */
static size_t
innerPreloadSetupPinnedInnerBufferPartitions(kern_multirels *h_kmrels,
											 pgstromTaskState *pts,
											 size_t offset)
{
	pgstromSharedState *ps_state = pts->ps_state;
	size_t		partition_sz = (size_t)__pinned_inner_buffer_partition_size_mb << 20;
	size_t		largest_sz = 0;
	int			largest_depth = -1;

	for (int depth=1; depth <= pts->num_rels; depth++)
	{
		if (pts->inners[depth-1].inner_pinned_buffer)
		{
			size_t	sz = pg_atomic_read_u64(&ps_state->inners[depth-1].inner_total);

			if (largest_depth < 0 || sz > largest_sz)
			{
				largest_sz = sz;
				largest_depth = depth;
			}
		}
	}

	if (largest_depth > 0 && largest_sz > partition_sz)
	{
		int		divisor = (largest_sz + partition_sz - 1) / partition_sz;
		size_t	kbuf_parts_sz = MAXALIGN(offsetof(kern_buffer_partitions,
												  parts[divisor]));
		if (h_kmrels)
		{
			kern_buffer_partitions *kbuf_parts = (kern_buffer_partitions *)
				((char *)h_kmrels + offset);

			memset(kbuf_parts, 0, kbuf_parts_sz);
			kbuf_parts->inner_depth  = largest_depth;
			kbuf_parts->hash_divisor = divisor;
			/* assign GPUs for each partition */
			for (int base=0; base < divisor; base += numGpuDevAttrs)
			{
				gpumask_t	optimal_gpus = pts->optimal_gpus;
				gpumask_t	other_gpus = (GetSystemAvailableGpus() & ~optimal_gpus);
				int			count = 0;
				int			unitsz = Min(divisor-base, numGpuDevAttrs);

				while ((optimal_gpus | other_gpus) != 0)
				{
					int			__part_id = (count++ % unitsz) + base;
					gpumask_t	__mask = 1UL;

					if (optimal_gpus != 0)
					{
						/* optimal GPUs first */
						while ((optimal_gpus & __mask) == 0)
							__mask <<= 1;
						kbuf_parts->parts[__part_id].available_gpus |= __mask;
						optimal_gpus &= ~__mask;
					}
					else if (other_gpus != 0)
					{
						/* elsewhere, other GPUs */
						while ((other_gpus & __mask) == 0)
							__mask <<= 1;
						kbuf_parts->parts[__part_id].available_gpus |= __mask;
						other_gpus &= ~__mask;
					}
					else
					{
						elog(ERROR, "Bug? pinned inner-buffer partitions tries to distribute tuples more GPUs than the installed devices");
					}
				}
			}
			elog(NOTICE, "pinned inner-buffer partitions (depth=%d, divisor=%d)",
				 kbuf_parts->inner_depth,
				 kbuf_parts->hash_divisor);
			for (int k=0; k < kbuf_parts->hash_divisor; k++)
				elog(NOTICE, "partition-%d (GPUs: %08lx)", k, kbuf_parts->parts[k].available_gpus);
			/* offset to the partition descriptor */
			h_kmrels->kbuf_part_offset = offset;
		}
		return kbuf_parts_sz;
	}
	return 0;
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
	size_t				ojmap_sz;

	/* other backend already setup the buffer metadata */
	if (ps_state->preload_shmem_length > 0)
		return;

	/*
	 * 1st pass: calculation of the buffer length
	 * 2nd pass: initialization of buffer metadata
	 */
again:
	offset = MAXALIGN(offsetof(kern_multirels, chunks[pts->num_rels]));
	offset += innerPreloadSetupPinnedInnerBufferPartitions(h_kmrels, pts, offset);
	for (int depth=1; depth <= pts->num_rels; depth++)
	{
		pgstromTaskInnerState *istate = &pts->inners[depth-1];
		TupleDesc	tupdesc = istate->ps->ps_ResultTupleDesc;
		uint64_t	nrooms;
		uint64_t	usage;
		size_t		nbytes;

		nrooms = pg_atomic_read_u64(&ps_state->inners[depth-1].inner_nitems);
		usage  = pg_atomic_read_u64(&ps_state->inners[depth-1].inner_usage);
		if (nrooms >= UINT_MAX)
			elog(ERROR, "GpuJoin: Inner Relation[%d] has %lu tuples, too large",
				 depth, nrooms);

		if (istate->inner_pinned_buffer)
		{
			if (h_kmrels)
			{
				//TODO: buffer partitioning
				h_kmrels->chunks[depth-1].pinned_buffer = true;
				h_kmrels->chunks[depth-1].buffer_id = istate->inner_buffer_id;
			}
		}
		else if (istate->hash_inner_keys != NIL &&
				 istate->hash_outer_keys != NIL)
		{
			/* Hash-Join */
			uint32_t	nslots = Max(320, nrooms);

			nbytes = (estimate_kern_data_store(tupdesc) +
					  MAXALIGN(sizeof(uint64_t) * nrooms) +
					  MAXALIGN(sizeof(uint64_t) * nslots) +
					  MAXALIGN(usage));
			if (h_kmrels)
			{
				kds = (kern_data_store *)((char *)h_kmrels + offset);
				h_kmrels->chunks[depth-1].kds_offset = offset;

				setup_kern_data_store(kds, tupdesc, nbytes,
									  KDS_FORMAT_HASH);
				kds->hash_nslots = nslots;
				memset(KDS_GET_HASHSLOT_BASE(kds), 0, sizeof(uint64_t) * nslots);
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
			nbytes = (estimate_kern_data_store(tupdesc) +
					  MAXALIGN(sizeof(uint64_t) * nrooms) +
					  MAXALIGN(sizeof(uint64_t) * nslots) +
					  MAXALIGN(usage));
			if (h_kmrels)
			{
				kds = (kern_data_store *)((char *)h_kmrels + offset);
				h_kmrels->chunks[depth-1].kds_offset = offset;

				setup_kern_data_store(kds, tupdesc, nbytes,
									  KDS_FORMAT_HASH);
				kds->hash_nslots = nslots;
				memset(KDS_GET_HASHSLOT_BASE(kds), 0, sizeof(uint64_t) * nslots);
			}
			offset += nbytes;

			/* 2nd part - GiST index blocks */
			Assert(i_rel->rd_amhandler == F_GISTHANDLER);
			block_offset = (estimate_kern_data_store(i_tupdesc) +
							MAXALIGN(sizeof(uint32_t) * nblocks));
			if (h_kmrels)
			{
				kds = (kern_data_store *)((char *)h_kmrels + offset);
				h_kmrels->chunks[depth-1].gist_offset = offset;
				nbytes = block_offset + BLCKSZ * nblocks;

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
				kds->length = nbytes;
				kds->nitems = nblocks;
				kds->block_nloaded = nblocks;
				innerPreloadSetupGiSTIndex(i_rel, kds);
			}
			offset += nbytes;
		}
		else
		{
			/* Nested-Loop */
			nbytes = (estimate_kern_data_store(tupdesc) +
					  MAXALIGN(sizeof(uint64_t) * nrooms) +
					  MAXALIGN(usage));
			if (h_kmrels)
			{
				kds = (kern_data_store *)((char *)h_kmrels + offset);
				h_kmrels->chunks[depth-1].kds_offset = offset;

				setup_kern_data_store(kds, tupdesc, nbytes,
									  KDS_FORMAT_ROW);
				h_kmrels->chunks[depth-1].is_nestloop = true;
			}
			offset += nbytes;
		}
	}

	/*
	 * Parameters for OUTER JOIN
	 */
	offset = PAGE_ALIGN(offset);
	ojmap_sz = 0;
	for (int depth=1; depth <= pts->num_rels; depth++)
	{
		pgstromTaskInnerState *istate = &pts->inners[depth-1];
		uint64_t	nrooms;
		size_t		nbytes;

		if (istate->join_type == JOIN_RIGHT ||
			istate->join_type == JOIN_FULL)
        {
			nrooms = pg_atomic_read_u64(&ps_state->inners[depth-1].inner_nitems);
			nbytes = MAXALIGN(sizeof(bool) * nrooms);
			if (h_kmrels)
			{
				h_kmrels->chunks[depth-1].right_outer = true;
				h_kmrels->chunks[depth-1].ojmap_offset = offset;
				memset((char *)h_kmrels + offset, 0, nbytes);
			}
			offset += nbytes;
			ojmap_sz += nbytes;
		}
		if (istate->join_type == JOIN_LEFT ||
			istate->join_type == JOIN_FULL)
		{
			if (h_kmrels)
				h_kmrels->chunks[depth-1].left_outer = true;
        }
	}
	offset = PAGE_ALIGN(offset);

	/*
	 * allocation of the host inner-buffer
	 */
	if (!h_kmrels)
	{
		Assert(ps_state->preload_shmem_handle != 0);
		h_kmrels = __mmapShmem(ps_state->preload_shmem_handle,
							   offset, pts->ds_entry);
		memset(h_kmrels, 0, offsetof(kern_multirels,
									 chunks[pts->num_rels]));
		h_kmrels->length = offset;
		h_kmrels->ojmap_sz = PAGE_ALIGN(ojmap_sz);
		h_kmrels->num_rels = pts->num_rels;
		ps_state->preload_shmem_length = offset;
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
							  uint64_t base_usage)
{
	uint64_t   *row_index = KDS_GET_ROWINDEX(kds);
	uint32_t	rowid = base_nitems;
	char	   *tail_pos = (char *)kds + kds->length;
	char	   *curr_pos = (tail_pos - base_usage);
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

		row_index[rowid++] = (tail_pos - curr_pos);
	}
}

/*
 * __innerPreloadSetupHashBuffer
 */
static void
__innerPreloadSetupHashBuffer(kern_data_store *kds,
							  pgstromTaskInnerState *istate,
							  uint32_t base_nitems,
							  uint64_t base_usage)
{
	uint64_t   *row_index = KDS_GET_ROWINDEX(kds);
	uint64_t   *hash_slot = KDS_GET_HASHSLOT_BASE(kds);
	uint32_t	rowid = base_nitems;
	char	   *tail_pos = (char *)kds + kds->length;
	char	   *curr_pos = (tail_pos - base_usage);
	inner_preload_buffer *preload_buf = istate->preload_buffer;

	for (uint32_t index=0; index < preload_buf->nitems; index++)
	{
		HeapTuple	htup = preload_buf->rows[index].htup;
		uint32_t	hash = preload_buf->rows[index].hash;
		uint32_t	hindex = hash % kds->hash_nslots;
		uint64_t	next, self;
		size_t		sz;
		kern_hashitem *hitem;

		sz = MAXALIGN(offsetof(kern_hashitem, t.htup) + htup->t_len);
		curr_pos -= sz;
		self = (tail_pos - curr_pos);

		next = __atomic_exchange_uint64(&hash_slot[hindex], self);
		hitem = (kern_hashitem *)curr_pos;
		hitem->next = next;
		hitem->hash = hash;
		hitem->__padding__ = 0;
		hitem->t.t_len = htup->t_len;
		hitem->t.rowid = rowid;
		memcpy(&hitem->t.htup, htup->t_data, htup->t_len);
		memcpy(&hitem->t.htup.t_ctid, &htup->t_self, sizeof(ItemPointerData));

		row_index[rowid++] = (tail_pos - (char *)&hitem->t);
		Assert(curr_pos >= ((char *)kds
							+ KDS_HEAD_LENGTH(kds)
							+ sizeof(uint64_t) * (kds->hash_nslots + rowid)));
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
	kern_buffer_partitions *kbuf_parts;
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

				if (istate->inner_pinned_buffer)
				{
					/*
					 * Pinned Inner Buffer
					 *
					 * For large inner relations by GpuScan/GpuJoin, it is waste
					 * of shared memory consumption and data transfer over the
					 * IPC connection. We allow to retain GpuScan/GpuJoin results
					 * on the GPU device memory and reuse it on the next GpuJoin.
					 */
					Assert(pgstrom_is_gpuscan_state(istate->ps) ||
						   pgstrom_is_gpujoin_state(istate->ps));
					execInnerPreLoadPinnedOneDepth((pgstromTaskState *)istate->ps,
												   &ps_state->inners[i].inner_nitems,
												   &ps_state->inners[i].inner_usage,
												   &ps_state->inners[i].inner_total,
												   &pts->inners[i].inner_buffer_id);
				}
				else
				{
					execInnerPreloadOneDepth(memcxt, pts, istate,
											 &ps_state->inners[i].inner_nitems,
											 &ps_state->inners[i].inner_usage);
				}
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

			for (int depth=1; depth <= leader->num_rels; depth++)
			{
				pgstromTaskInnerState *istate = &leader->inners[depth-1];
				inner_preload_buffer *preload_buf = istate->preload_buffer;
				kern_data_store *kds = KERN_MULTIRELS_INNER_KDS(pts->h_kmrels, depth);
				uint64_t	base_nitems;
				uint64_t	base_usage;

				/*
				 * If this backend/worker process called GpuJoinInnerPreload()
				 * after the INNER_PHASE__SCAN_RELATIONS completed, it has no
				 * preload_buf, thus, no tuples should be added.
				 */
				if (!preload_buf)
					continue;

				Assert(kds != NULL);
				SpinLockAcquire(&ps_state->preload_mutex);
				base_nitems  = kds->nitems;
				kds->nitems += preload_buf->nitems;
				base_usage   = kds->usage;
				kds->usage  += preload_buf->usage;
				SpinLockRelease(&ps_state->preload_mutex);

				/* sanity checks */
				if (base_nitems + preload_buf->nitems >= UINT_MAX)
					elog(ERROR, "GpuJoin: inner relation has %lu tuples, too large",
						 base_nitems + preload_buf->nitems);
				Assert(KDS_HEAD_LENGTH(kds) +
					   MAXALIGN(sizeof(uint64_t) * (kds->hash_nslots +
													base_nitems +
													preload_buf->nitems)) +
					   base_usage +
					   preload_buf->usage <= kds->length);

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

			/*
			 * Inner-buffer partitioning often requires multiple outer-scan,
			 * if number of partitions is larger than the number of GPU devices.
			 */
			kbuf_parts = KERN_MULTIRELS_PARTITION_DESC(pts->h_kmrels, -1);
			if (kbuf_parts)
			{
				pts->num_scan_repeats = (kbuf_parts->hash_divisor +
										 numGpuDevAttrs - 1) / numGpuDevAttrs;
				assert(pts->num_scan_repeats > 0);
			}
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
 * GpuJoinInnerPreload
 */
void
GpuJoinInnerPreloadAfterWorks(pgstromTaskState *pts)
{
	for (int i=0; i < pts->num_rels; i++)
	{
		pgstromTaskInnerState *istate = &pts->inners[i];

		/*
		 * Once inner hash/heap join buffer was built, we no longer need
		 * the final buffer of the inner child GpuScan/GpuJoin, because
		 * it is already reconstructed as a part of partitioned inner-buffer,
		 * or parent GpuJoin acquired gpuQueryBuffer if zero-copy mode.
		 *
		 * Even though the final buffer is allocated as CUDA managed memory,
		 * some portion still occupies device memory, and eviction consumes
		 * unnecessary host memory and PCI-E bandwidth, so early release will
		 * reduce host/device memory pressure.
		 *
		 * But here is one exception. When divisor of inner-buffer partitions
		 * is larger than the number of GPU devices, this final buffer shall
		 * be reused for the inner buffer reconstruction.
		 */
		if (istate->inner_pinned_buffer)
		{
			pgstromTaskState   *i_pts = (pgstromTaskState *)istate->ps;

			Assert(pgstrom_is_gpuscan_state(istate->ps) ||
				   pgstrom_is_gpujoin_state(istate->ps));
			if (i_pts->conn)
			{
				xpuClientCloseSession(i_pts->conn);
				i_pts->conn = NULL;
			}
		}
	}
}

/*
 * __pgstrom_init_xpujoin_common
 */
static void
__pgstrom_init_xpujoin_common(void)
{
	static bool	__initialized = false;

	if (!__initialized)
	{
		/* pg_strom.debug_xpujoinpath */
		DefineCustomBoolVariable("pg_strom.debug_xpujoinpath",
								 "Turn on/off debug output for XpuJoin paths",
								 NULL,
								 &pgstrom_debug_xpujoinpath,
								 false,
								 PGC_USERSET,
								 GUC_NOT_IN_SAMPLE,
								 NULL, NULL, NULL);
		/* hook registration */
		set_join_pathlist_next = set_join_pathlist_hook;
		set_join_pathlist_hook = XpuJoinAddCustomPath;

		__initialized = true;
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
	/* threshold of pinned inner buffer of GpuJoin */
	DefineCustomIntVariable("pg_strom.pinned_inner_buffer_threshold",
							"Threshold of pinned inner buffer of GpuJoin",
							NULL,
							&__pinned_inner_buffer_threshold_mb,
							0,		/* disabled */
							0,		/* 0 means disabled */
							INT_MAX,
							PGC_SUSET,
							GUC_NOT_IN_SAMPLE | GUC_UNIT_MB,
							NULL, NULL, NULL);
	/* unit size of partitioned pinned inner buffer of GpuJoin
	 * default: 90% of usable DRAM for high-end GPUs,
	 *          80% of usable DRAM for middle-end GPUs.
	 */
	for (int i=0; i < numGpuDevAttrs; i++)
	{
		size_t	dram_sz = gpuDevAttrs[i].DEV_TOTAL_MEMSZ;
		size_t	part_sz;

		if (dram_sz >= (32UL<<30))
			part_sz = ((dram_sz - (2UL<<30)) * 9 / 10) >> 20;
		else
			part_sz = ((dram_sz - (1UL<<30)) * 8 / 10) >> 20;
		if (i==0 || part_sz < __pinned_inner_buffer_partition_size_mb)
			__pinned_inner_buffer_partition_size_mb = part_sz;
	}
	DefineCustomIntVariable("pg_strom.pinned_inner_buffer_partition_size",
							"Unit size of partitioned pinned inner buffer of GpuJoin",
							NULL,
							&__pinned_inner_buffer_partition_size_mb,
							__pinned_inner_buffer_partition_size_mb,
							1024,	/* 1GB */
							INT_MAX,
							PGC_SUSET,
							GUC_NO_SHOW_ALL | GUC_NOT_IN_SAMPLE | GUC_UNIT_MB,
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
	/* common portion */
	__pgstrom_init_xpujoin_common();
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
	/* common portion */
	__pgstrom_init_xpujoin_common();
}
