/*
 * gpunestloop.c
 *
 * GPU accelerated nested-loop implementation
  * ----
 * Copyright 2011-2015 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2015 (C) The PG-Strom Development Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */
#include "postgres.h"
#include "pg_strom.h"
#include "cuda_nestloop.h"



/* forward declaration of the GTS callbacks */
static bool	gpujoin_task_process(GpuTask *gtask);
static bool	gpujoin_task_complete(GpuTask *gtask);
static void	gpujoin_task_release(GpuTask *gtask);
static GpuTask *gpujoin_next_chunk(GpuTaskState *gts);
static TupleTableSlot *gpujoin_next_tuple(GpuTaskState *gts);

/* static variables */
static set_join_pathlist_hook_type set_join_pathlist_next;
static CustomPathMethods	gpujoin_path_methods;
static CustomScanMethods	gpujoin_plan_methods;
static PGStromExecMethods	gpujoin_exec_methods;
static bool					enable_gpunestloop;

/*
 * GpuJoinPath
 *
 *
 *
 */
typedef struct
{
	CustomPath		cpath;
	Path		   *outer_path;
	Size			total_length;
	double			kresults_ratio;
	int				num_rels;
	List		   *host_quals;
	struct {
		double		rows;			/* rows to be generated in this depth */
		Cost		startup_cost;	/* outer scan cost + something */
		Cost		total_cost;		/* outer scan cost + something */
		Path	   *scan_path;
		JoinType	join_type;
		List	   *hash_quals;		/* valid quals, if hash-join */
		List	   *join_quals;		/* all the device quals, incl hash_quals */
		double		row_growth_ratio;
		double		buffer_portion;
		Size		chunk_size;
		int			nbatches;
	} inners[FLEXIBLE_ARRAY_MEMBER];
} GpuJoinPath;

/*
 * GpuJoinInfo - private state object of CustomScan(GpuJoin)
 */
typedef struct
{
	char	   *kern_source;
	int			extra_flags;
	/* supplemental information of ps_tlist */
	List	   *ps_src_depth;	/* source depth of the ps_tlist entry */
	List	   *ps_src_resno;	/* source resno of the ps_tlist entry */

	int			num_rels;


} GpuJoinInfo;

static inline void
form_gpujoin_info(CustomScan *cscan, GpuNestLoopInfo *gnl_info)
{}

static inline GpuNestLoopInfo *
deform_gpujoin_info(CustomScan *cscan)
{}

/*
 * GpuJoinState - execution state object of GpuJoin
 */
typedef struct
{
	GpuTaskState	gts;
	List		   *join_types;
	ExprState	   *outer_quals;
	List		   *join_clauses;

} GpuNestLoopState;

/*****/
static inline bool
path_is_gpujoin(Path *path)
{
	CustpmPath *cpath = (CustpmPath *) path;

	if (!IsA(cpath, CustomPath))
		return false;
	if (cpath->methods != &gpujoin_path_methods)
		return false;
	return true;
}

static bool
path_is_mergeable_gpujoin(Path *pathnode)
{
	RelOptInfo	   *joinrel = pathnode->parent;
	GpuJoinPath	   *gpath = (GpuJoinPath *) pathnode;
	ListCell	   *lc;

	if (!path_is_gpujoin(pathnode))
		return false;

	/*
	 * Only last depth can have host only clauses
	 */
	if (gpath->host_quals != NIL)
		return false;

	/*
	 * Target-list must be simple var-nodes only
	 */
	foreach (lc, joinrel->reltargetlist)
	{
		Expr   *expr = lfirst(lc);

		if (!IsA(expr, Var))
			return false;
	}

	/*
	 * TODO: Any other condition to be checked?
	 */
	return true;
}

/*
 * cost_gpujoin
 *
 * estimation of GpuJoin cost
 */
static bool
cost_gpujoin(PlannerInfo *root,
			 GpuJoinPath *gpath,
			 Relids required_outer)
{
	Path	   *outer_path = gpath->outer_path;
	Cost		startup_cost;
	Cost		run_cost;
	Cost		gpu_startup_cost;
	Cost		gpu_run_cost;
	QualCost	host_cost;
	QualCost   *join_cost;
	double		gpu_cpu_ratio;
	double		kresults_ratio;
	double		kresults_ratio_max;
	double		outer_ntuples;
	double		inner_ntuples;
	Size		total_length;
	int			i, num_rels = gpath->num_rels;

	/*
	 * Buffer size estimation
	 */
	kresults_ratio = kresults_ratio_max = 1.0;
	for (i=0; i < num_rels; i++)
	{
		kresults_ratio = ((double)(i + 2) *
						  gpath->inners[i].row_growth_ratio *
						  kresults_ratio);
		kresults_ratio_max = Max(kresults_ratio, kresults_ratio_max);
	}
	gpath->kresults_ratio = kresults_ratio_max;

	/*
	 * Cost of per-tuple evaluation
	 */
	gpu_cpu_ratio = pgstrom_gpu_operator_cost / cpu_operator_cost;
	join_cost = palloc0(sizeof(QualCost) * num_rels);
	for (i=0; i < num_rels; i++)
	{
		cost_qual_eval(join_cost + i, join_quals, root);
		join_cost[i].per_tuple *= gpu_cpu_ratio;
	}
	cost_qual_eval(&host_cost, host_quals, root);

	/*
	 * Estimation of multi-relations buffer size
	 */
retry:
	gpu_startup_cost = 0.0;
	gpu_run_cost = 0.0;
	total_length = STROMALIGN(offsetof(kern_multirels,
									   krels[num_rels]));
	num_batches = 1;
	largest_size = 0;
	largest_index = -1;
	outer_ntuples = outer_path->rows;
	for (i=0; i < num_rels; i++)
	{
		Path	   *inner_path = gpath->inners[i].scan_path;
		RelOptInfo *inner_rel = inner_path->parent;
		cl_uint		ncols = list_length(inner_rel->reltargetlist);
		cl_uint		nslots;

		/* force a plausible relation size if no information.
		 * It expects 15% of margin to avoid unnecessary hash-
		 * table split
		 */
		inner_ntuples = (Max(1.15 * inner_path->rows, 1000.0)
						 / gpath->inners[i].nbatches);

		if (gpath->inners[i].hash_quals != NIL)
		{
			entry_size = (offsetof(kern_hashentry, htup) +
						  MAXALIGN(offsetof(HeapTupleHeaderData,
											t_bits[BITMAPLEN(ncols)])) +
						  MAXALIGN(inner_rel->width));
			/* header portion of kern_hashtable */
			chunk_size = STROMALIGN(offsetof(kern_hashtable,
											 colmeta[ncols]));
			/* hash entry slot */
			chunk_size += STROMALIGN(sizeof(cl_uint) * (Size)inner_ntuples);
			/* kern_hashentry body */
			chunk_size += STROMALIGN(entry_size * (Size)inner_ntuples);
			/* row-index of the tuples */
			chunk_size += STROMALIGN(sizeof(cl_uint) * (Size)inner_ntuples);
		}
		else
		{
			entry_size = (offsetof(kern_tupitem, htup) +
						  MAXALIGN(offsetof(HeapTupleHeaderData,
											t_bits[BITMAPLEN(ncols)])) +
						  MAXALIGN(inner_rel->width));
			/* header portion of kern_data_store */
			chunk_size += STROMALIGN(offsetof(kern_data_store,
											  colmeta[ncols]));
			/* row-index of the tuples */
			chunk_size += STROMALIGN(sizeof(cl_uint) * (Size)inner_ntuples);
			/* kern_tupitem body */
			chunk_size += STROMALIGN(entry_size * (Size)inner_ntuples);
		}
		gpath->inners[i].chunk_size = chunk_size;

		if (largest_index < 0 || largest_size < chunk_size)
		{
			largest_size = chunk_size;
			largest_index = i;
		}
		total_length += chunk_size;

		/*
		 * Cost calculation in this depth
		 */

		/* cost to load tuples onto the buffer */
		startup_cost = (inner_path->total_cost +
						cpu_tuple_cost * inner_path->rows);
		/* cost to compute hash value, if any */
		if (gpath->inners[i].hash_quals != NIL)
			startup_cost += (cpu_operator_cost * inner_path->rows *
							 list_length(gpath->inners[i].hash_quals));
		/* fixed cost to initialize/setup/use GPU device */
		startup_cost += pgstrom_gpu_setup_cost;

		/* cost to execute previous stage */
		if (i == 0)
			startup_cost += (outer_path->total_cost +
							 cpu_tuple_cost * outer_path->rows);
		else
			startup_cost += (gpath->inners[i-1].startup_cost +
							 gpath->inners[i-1].run_cost);
		/* iteration of outer scan/join */
		startup_cost *= (double) gpath->inners[i].nbatches;

		/* cost to evaluate join qualifiers */
		if (gpath->inners[i].hash_quals != NIL)
			run_cost = (join_cost[i].per_tuple
						* outer_ntuples
						* clamp_row_est(inner_ntuples) * 0.5
						* (double) gpath->inners[i].nbatches);
		else
			run_cost = (join_cost[i].per_tuple
						* outer_ntuples
						* clamp_row_est(inner_ntuples)
						* (double) gpath->inners[i].nbatches);

		if (gpath->inners[i].hash_quals != NIL)
		{
			/* In case of Hash-Join logic */
			gpath->inners[i].startup_cost += join_cost[i].startup;
			gpath->inners[i].run_cost = join_cost[i].per_tuple
				* outer_ntuples
				* (double) gpath->inners[i].nbatches;
		}
		else
		{
			/* In case of Nest-Loop logic */
			gpath->inners[i].startup_cost += join_cost[i].startup;
			gpath->inners[i].run_cost = join_cost[i].per_tuple
				* outer_ntuples
				* clamp_row_est(inner_ntuples)
				* (double) gpath->inners[i].nbatches;
		}
		outer_ntuples = gpath->inners[i].rows;
	}
	/* put cost value on the gpath */
	gpath->startup_cost = gpath->inners[num_rels - 1].startup_cost;
	gpath->total_cost = (gpath->inners[num_rels - 1].startup_cost +
						 gpath->inners[num_rels - 1].run_cost);

    /*
     * NOTE: In case when extreme number of rows are expected,
     * it does not make sense to split hash-tables because
     * increasion of numbatches also increases the total cost
     * by iteration of outer scan. In this case, the best
     * strategy is to give up this path, instead of incredible
     * number of numbatches!
     */
	if (!add_path_precheck(joinrel,
						   startup_cost + gpu_startup_cost,
						   startup_cost + gpu_startup_cost +
						   run_cost + gpu_run_cost,
						   NULL, required_outer))
		return false;

	/*
	 * If size of inner multi-relations buffer is still larger than
	 * device allocatable limitation, we try to split the largest
	 * relation then retry the estimation.
	 */
	if (total_length > gpuMemMaxAllocSize())
	{
		gpath->inners[largest_index].nbatches++;
		goto retry;
	}

	/*
	 * Update estimated multi-relations buffer length and portion
	 * of the 
	 */
	gpath->total_length = total_length;
	for (i=0; i < num_rels; i++)
	{
		gpath->inners[i].buffer_portion =
			((double)gpath->inners[i].chunk_size / (double)total_length);
	}
	return true;
}

static GpuJoinPath *
create_gpujoin_path(PlannerInfo *root,
					RelOptInfo *joinrel,
					JoinType jointype,
					Path *outer_path,
					Path *inner_path,					
					ParamPathInfo *param_info,
					List *hash_quals,
					List *join_quals,
					List *host_quals,
					bool can_bulkload,
					bool try_merge)
{
	GpuJoinPath	   *result;
	GpuJoinPath	   *source;
	double			row_growth_ratio;
	int				num_rels;

	/*
	 * 'row_growth_ratio' is used to compute size of result buffer 
	 * for GPU kernel execution. If joinrel contains host-only
	 * qualifiers, we need to estimate number of rows at the time
	 * of host-only qualifiers.
	 */
	if (host_quals == NIL)
		row_growth_ratio = joinrel->rows / outer_path->rows;
	else
	{
		RelOptInfo		dummy;

		set_joinrel_size_estimates(root, &dummy,
								   outer_path->parent,
								   inner_path->parent,
								   sjinfo,
								   join_quals);
		row_growth_ratio = dummy.rows / outer_path->rows;
	}


	if (!try_merge)
		num_rels = 1;
	else
	{
		Assert(path_is_mergeable_gpujoin(outer_path));
		source = (GpuJoinPath *) outer_path;
		outer_path = source->outer_path;
		num_rels = source->num_rels + 1;
		/* source path also has to support bulkload */
		if ((source->cpath.flags & CUSTOMPATH_SUPPORT_BULKLOAD) == 0)
			can_bulkload = false;
	}

	result = palloc0(offsetof(GpuJoinPath, inners[num_rels]));
	NodeSetTag(result, T_CustomPath);
	result->cpath.path.pathtype = T_CustomScan;
	result->cpath.path.parent = joinrel;
	result->cpath.path.param_info = param_info;
	result->cpath.path.pathkeys = NIL;
	result->cpath.flags = (can_bulkload ? CUSTOMPATH_SUPPORT_BULKLOAD : 0);
	result->cpath.methods = &gpujoin_path_mathods;
	result->outer_path = outer_path;
	result->total_length = 0;
	result->num_rels = num_rels;
	result->host_quals = host_quals;
	if (source)
	{
		memcpy(result->inners, source->inners,
			   offsetof(GpuHashJoinPath, inners[num_rels]) -
               offsetof(GpuHashJoinPath, inners[0]));
	}
	result->inners[num_rels - 1].rows = joinrel->rows;
	result->inners[num_rels - 1].startup_cost = ;
	result->inners[num_rels - 1].total_cost = ;
	result->inners[num_rels - 1].scan_path = inner_path;
	result->inners[num_rels - 1].join_type = join_type;
	result->inners[num_rels - 1].hash_quals = hash_quals;
	result->inners[num_rels - 1].join_quals = join_quals;
	result->inners[num_rels - 1].row_growth_ratio = row_growth_ratio;
	result->inners[num_rels - 1].buffer_portion = 0.0;	/* to be set later */
	result->inners[num_rels - 1].nbatches = 1;			/* to be set later */

	return result;
}

static void
try_gpujoin_path(PlannerInfo *root,
				 RelOptInfo *joinrel,
				 JoinType jointype,
				 SpecialJoinInfo *sjinfo,
				 SemiAntiJoinFactors *semifactors,
				 Relids param_source_rels,
				 Relids extra_lateral_rels,
				 Path *outer_path,
				 Path *inner_path,
				 ParamPathInfo *param_info,
				 List *hash_quals,
				 List *join_quals,
				 List *host_quals)
{
	GpuJoinPath	   *gpath;
	ParamPathInfo  *param_info;
	Relids			required_outer;
	cl_uint			can_bulkload = 0;

	required_outer = calc_non_nestloop_required_outer(outer_path,
													  inner_path);
	if (required_outer && !bms_overlap(required_outer, param_source_rels))
	{
		bms_free(required_outer);
		return;
	}

	/*
	 * Independently of that, add parameterization needed for any
	 * PlaceHolderVars that need to be computed at the join.
	 */
	required_outer = bms_add_members(required_outer, extra_lateral_rels);

	/*
	 * Check availability of bulkload in this joinrel. If child GpuJoin
	 * is merginable, both of nodes have to support bulkload.
	 */
	if (host_quals == NIL)
	{
		foreach (lc, joinrel->reltargetlist)
		{
			Expr   *expr = lfirst(lc);

			if (!IsA(expr, Var) &&
				!pgstrom_codegen_available_expression(expr))
				break;
		}
		if (lc == NULL)
			can_bulkload = CUSTOMPATH_SUPPORT_BULKLOAD;
	}

	/*
	 * ParamPathInfo of this join
	 */
	param_info = get_joinrel_parampathinfo(root,
										   joinrel,
										   outer_path,
										   inner_path,
										   sjinfo,
										   required_outer,
										   &restrict_clauses);

	/*
	 * Try GpuHashJoin logic
	 */
	if (hash_qual != NIL)
	{
		gpath = create_gpujoin_path(root, joinrel, jointype,
									outer_path, inner_path, param_info,
									hash_quals, join_quals, host_quals,
									can_bulkload, false);
		if (cost_gpujoin(root, gpath, required_outer))
			add_path(joinrel, &gpath->cpath.path);

		if (path_is_merginable_gpujoin(outer_path))
		{
			gpath = create_gpujoin_path(root, joinrel, jointype,
										outer_path, inner_path, param_info,
										hash_quals, join_quals, host_quals,
										can_bulkload, true);
			if (cost_gpujoin(root, gpath, required_outer))
				add_path(joinrel, &gpath->cpath.path);
		}
	}

	/*
	 * Try GpuNestLoop logic
	 */
	if (jointype == JOIN_INNER || jointype == JOIN_RIGHT)
	{
		gpath = create_gpujoin_path(root, joinrel, jointype,
									outer_path, inner_path, param_info,
									NIL, join_quals, host_quals,
									can_bulkload, false);
		if (cost_gpujoin(root, gpath, required_outer))
			add_path(joinrel, &gpath->cpath.path);

		if (path_is_merginable_gpujoin(outer_path))
		{
			gpath = create_gpujoin_path(root, joinrel, jointype,
										outer_path, inner_path, param_info,
										NIL, join_quals, host_quals,
										can_bulkload, true);
			if (cost_gpujoin(root, gpath, required_outer))
				add_path(joinrel, &gpath->cpath.path);
		}
	}
	return;
}

/*
 * gpujoin_add_join_path
 *
 * entrypoint of the GpuJoin logic
 */
static void
gpujoin_add_join_path(PlannerInfo *root,
					  RelOptInfo *joinrel,
					  RelOptInfo *outerrel,
					  RelOptInfo *innerrel,
					  List *restrictlist,
					  JoinType jointype,
					  SpecialJoinInfo *sjinfo,
					  SemiAntiJoinFactors *semifactors,
					  Relids param_source_rels,
					  Relids extra_lateral_rels)
{
	Path	   *cheapest_startup_outer = outerrel->cheapest_startup_path;
	Path	   *cheapest_total_outer = outerrel->cheapest_total_path;
	Path	   *cheapest_total_inner = innerrel->cheapest_total_path;
	List	   *host_quals = NIL;
	List	   *hash_quals = NIL;
	List	   *join_quals = NIL;
	ListCell   *lc;

	/* calls secondary module if exists */
	if (set_join_pathlist_next)
		set_join_pathlist_next(root,
							   joinrel,
							   outerrel,
							   innerrel,
							   restrictlist,
							   jointype,
							   sjinfo,
							   semifactors,
							   param_source_rels,
							   extra_lateral_rels);
	/* nothing to do, if PG-Strom is not enabled */
	if (!pgstrom_enabled())
		return;

	/* quick exit, if unsupported join type */
	if (jointype != JOIN_INNER && jointype != JOIN_FULL &&
		jointype != JOIN_RIGHT && jointype != JOIN_LEFT)
		return;

	/*
	 * 
	 *
	 *
	 */
	foreach (lc, restrictlist)
	{
		RestrictInfo   *rinfo = (RestrictInfo *) lfirst(lc);

		/* Even if clause is hash-joinable, here is no benefit
		 * in case when clause is not runnable on CUDA device.
		 * So, we drop them from the candidate of the join-key.
		 */
		if (!pgstrom_codegen_available_expression(rinfo->clause))
		{
			host_quals = lappend(host_quals, rinfo);
			continue;
		}
		/* otherwise, device executable expression */
		join_quals = lappend(join_quals, rinfo);

		/*
		 * If processing an outer join, only use its own join clauses
		 * for hashing.  For inner joins we need not be so picky.
		 */
		if (IS_OUTER_JOIN(jointype) && rinfo->is_pushed_down)
			continue;

		/* Is it hash-joinable clause? */
		if (!rinfo->can_join || !OidIsValid(rinfo->hashjoinoperator))
			continue;

		/*
		 * Check if clause has the form "outer op inner" or "inner op outer".
		 * If suitable, we may be able to choose GpuHashJoin logic.
		 *
		 * See clause_sides_match_join also.
		 */
		if ((bms_is_subset(rinfo->left_relids, outerrel->relids) &&
			 bms_is_subset(rinfo->right_relids, innerrel->relids)) ||
			(bms_is_subset(rinfo->left_relids, innerrel->relids) &&
			 bms_is_subset(rinfo->right_relids, outerrel->relids)))
		{
			/* OK, it is hash-joinable qualifier */
			hash_quals = lappend(hash_quals, rinfo);
		}
	}

	/*
	 * If either cheapest-total path is parameterized by the other rel, we
	 * can't use a hashjoin.  (There's no use looking for alternative
	 * input paths, since these should already be the least-parameterized
	 * available paths.)
	 */
	if (PATH_PARAM_BY_REL(cheapest_total_outer, innerrel) ||
		PATH_PARAM_BY_REL(cheapest_total_inner, outerrel))
		return;

	if (cheapest_startup_outer)
	{
		/* GpuHashJoin logic, if possible */
		if (hash_quals != NIL)
			try_gpujoin_path(root,
							 joinrel,
							 jointype,
							 cheapest_startup_outer,
							 cheapest_total_inner,
							 hash_quals,
							 join_quals,
							 host_quals);

		/* GpuNestLoop logic, if possible */
		if (jointype == JOIN_INNER || jointype == JOIN_RIGHT)
			try_gpujoin_path(root,
							 joinrel,
							 jointype,
							 cheapest_startup_outer,
							 cheapest_total_inner,
							 NIL,
							 join_quals,
							 host_quals);
	}

	if (cheapest_startup_outer != cheapest_total_outer)
	{
		/* GpuHashJoin logic, if possible */
		if (hash_quals != NIL)
			try_gpujoin_path(root,
							 joinrel,
							 jointype,
							 cheapest_startup_outer,
							 cheapest_total_inner,
							 hash_quals,
							 join_quals,
							 host_quals);

		/* GpuNestLoop logic, if possible */
		if (jointype == JOIN_INNER || jointype == JOIN_RIGHT)
			try_gpujoin_path(root,
							 joinrel,
							 jointype,
							 cheapest_startup_outer,
							 cheapest_total_inner,
							 NIL,
							 join_quals,
							 host_quals);
	}
}

/*
 * create_gpujoin_plan
 *
 *
 *
 *
 */
static Plan *
create_gpujoin_plan(PlannerInfo *root,
					RelOptInfo *rel,
					CustomPath *best_path,
					List *tlist,
					List *clauses)
{
	GpuJoinPath	   *gpath = (GpuJoinPath *) best_path;
	GpuJoinInfo	   *ginfo;
	CustomScan	   *gjoin;
	int				i;

	gjoin = makeNode(CustomScan);
	gjoin->scan.plan.targetlist = tlist;
	gjoin->scan.plan.qual = gpath->host_quals;
	gjoin->flags = best_path->flags;
	gjoin->methods = &gpujoin_plan_methods;

	for (i=0; i < gpath->num_rels; i++)
	{
		CustomScan	   *mrels
			= pgstrom_create_multirels_plan(&kern_source, &extra_flags);

		/* chain it under the GpuJoin */
		if (prev_plan)
			innerPlan(prev_plan) = &mrels->scan.plan;
		else
			innerPlan(gjoin) = &mrels->scan.plan;
		prev_plan = &mrels->scan.plan;
	}


	for (i=0; i < gpath->num_rels; i++)
	{
		MultiRelStore	mrs_info;
		CustomScan	   *mrs_scan;
		Plan		   *scan_plan;
		JoinType		join_type = gpath->inners[i].join_type;
		List		   *join_clause = gpath->inners[i].join_clause;

		/* make a plan node of the child inner path */
		scan_plan = create_plan_recurse(root, gpath->inners[i].scan_path);

		mrs_scan = makeNode(CustomScan);
		mrs_scan->scan.plan.startup_cost = scan_plan->total_cost;
		mrs_scan->scan.plan.total_cost = scan_plan->total_cost;
		mrs_scan->scan.plan.plan_rows = scan_plan->plan_rows;
		mrs_scan->scan.plan.plan_width = scan_plan->plan_width;
		mrs_scan->scan.plan.targetlist = scan_plan->targetlist;
		mrs_scan->scan.plan.qual = NIL;
		mrs_scan->scan.scanrelid = 0;
        mrs_scan->flags = 0;
        mrs_scan->custom_ps_tlist = scan_plan->targetlist;
        mrs_scan->custom_relids = NULL;
        mrs_scan->methods = &multi_rel_store_plan_methods;

		memset(&mrs_info, 0, sizeof(MultiRelStore));
		mrs_info.depth = i + 1;
		mrs_info.nloops = gpath->inners[i].nloops;
		mrs_info.threshold = gpath->inners[i].threshold;
		mrs_info.mrs_size = gpatg->mrs_size;

		form_multi_rel_store_info(mrs_scan, &mrs_info);

		/* also add properties of gnl_info */
		gnl_info.join_types = lappend_int(gnl_info.join_types,
										  (int) gpath->inners[i].join_type);
		gnl_info.join_clauses = lappend(gnl_info.join_clauses,
										build_flatten_qualifier(join_clause));

		/* chain it under the GpuNestLoop */
		outerPlan(mrs_scan) = scan_plan;
		if (prev_plan)
			innerPlan(prev_plan) = &mrs_scan->scan.plan;
		else
			innerPlan(gnl_scan) = &mrs_scan->scan.plan;
		prev_plan = &mrs_scan->scan.plan;
	}

	/*
	 * Creation of the underlying outer Plan node. In case of SeqScan,
	 * it may make sense to replace it with GpuScan for bulk-loading.
	 */
	outer_plan = create_plan_recurse(root, gpath->outer_path);
	if (IsA(outer_plan, SeqScan) || IsA(outer_plan, CustomScan))
	{
		Query	   *parse = root->parse;
		List	   *outer_quals = NULL;
		Plan	   *alter_plan;

		alter_plan = pgstrom_try_replace_plannode(outer_plan,
												  parse->rtable,
												  &outer_quals);
		if (alter_plan)
		{
			mrs_info.outer_quals = build_flatten_qualifier(outer_quals);
			outer_plan = alter_plan;
		}
	}

	/* check bulkload availability */
	if (IsA(outer_plan, CustomScan))
	{
		int		custom_flags = ((CustomScan *) outer_plan)->flags;

		if ((custom_flags & CUSTOMPATH_SUPPORT_BULKLOAD) != 0)
            gnl_info.outer_bulkload = true;
	}
	outerPlan(gnl_scan) = outer_plan;

	/*
	 * Build a pseudo-scan targetlist
	 */



	/*
	 * Kernel code generation
	 */


	form_gpunestloop_info(gnl_scan, &gnl_info);

	return &gnl_scan->scan.plan;
}

static void
gpujoin_textout_path(StringInfo str, const CustomPath *node)
{
	GpuJoinPath *gjoin = (GpuJoin *) node;
	int		i;

	/* outer_path */
	appendStringInfo(str, " :outer_path %s",
					 nodeToString(gjoin->outer_path));
	/* total_length */
	appendStringInfo(str, " :total_length %zu",
					 gjoin->total_length);

	/* kresults_ratio */
	appendStringInfo(str, " :kresults_ratio %.2f",
					 gjoin->kresults_ratio);
	/* num_rels */
	appendStringInfo(str, " :num_rels %d", gjoin->num_rels);

	/* host_quals */
	appendStringInfo(str, " :host_quals %s",
					 nodeToString(gjoin->host_quals));

	/* inners */
	appendStringInfo(str, " :inners (");
	for (i=0; i < gpath->num_rels; i++)
	{
		appendStringInfo(str, "{");
		/* rows, startup_cost, total_cost */
		appendStringInfo(str, " :rows %.2f", gjoin->rows);
		appendStringInfo(str, " :startup_cost %.2f", gjoin->startup_cost);
		appendStringInfo(str, " :total_cost %.2f", gjoin->total_cost);

		/* scan_path */
		appendStringInfo(str, " :scan_path %s",
						 nodeToString(gpath->inners[i].scan_path));
		/* join_type */
		appendStringInfo(str, " :join_type %d",
						 (int)gpath->inners[i].join_type);
		/* hash_quals */
		appendStringInfo(str, " :hash_quals %s",
						 nodeToString(gpath->inners[i].hash_quals));
		/* join_quals */
		appendStringInfo(str, " :join_clause %s",
						 nodeToString(gpath->inners[i].join_quals));
		/* row_growth_ratio */
		appendStringInfo(str, " :row_growth_ratio %.2f",
						 gpath->inners[i].row_growth_ratio);
		/* buffer_portion */
		appendStringInfo(str, " :buffer_portion %.2f",
						 gpath->inners[i].buffer_portion);
		/* chunk_size */
		appendStringInfo(str, " :chunk_size %zu",
						 gjoin->chunk_size);
		/* nbatches */
		appendStringInfo(str, " :nbatches %d",
						 gpath->inners[i].nbatches);
		appendStringInfo(str, "}");
	}
	appendStringInfo(str, ")");
}

static Node *
gpunestedloop_create_scan_state(CustomScan *node)
{
	GpuNestLoop	   *gnl = (GpuNestLoop *) node;
	GpuNestLoopState   *gnls;


	return (Node *) gnls;
}

static void
gpunestloop_begin(CustomScanState *node, EState *estate, int eflags)
{}

static TupleTableSlot *
gpunestloop_exec(CustomScanState *node)
{}

static void
gpunestloop_end(CustomScanState *node)
{}

static void
gpunestloop_rescan(CustomScanState *node)
{}

static void
gpunestloop_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{}




/*
 *
 *
 */
void
pgstrom_init_gpunestloop(void)
{
	/* turn on/off gpunestloop */
	DefineCustomBoolVariable("pg_strom.enable_gpunestloop",
							 "Enables the use of GPU accelerated nested-loop",
							 NULL,
							 &enable_gpunestloop,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* setup path methods */
	gpujoin_path_methods.CustomName				= "GpuJoin";
	gpujoin_path_methods.PlanCustomPath			= create_gpunestloop_plan;
	gpujoin_path_methods.TextOutCustomPath		= gpunestloop_textout_path;

	/* setup plan methods */
	gpujoin_methods.CustomName					= "GpuJoin";
	gpujoin_plan_methods.CreateCustomScanState	= gpujoin_create_scan_state;
	gpujoin_plan_methods.TextOutCustomScan		= NULL;

	/* setup exec methods */
	gpujoin_exec_methods.c.CustomName			= "GpuNestedLoop";
	gpujoin_exec_methods.c.BeginCustomScan		= gpunestloop_begin;
	gpujoin_exec_methods.c.ExecCustomScan		= gpunestloop_exec;
	gpujoin_exec_methods.c.EndCustomScan		= gpunestloop_end;
	gpujoin_exec_methods.c.ReScanCustomScan		= gpunestloop_rescan;
	gpujoin_exec_methods.c.MarkPosCustomScan	= NULL;
	gpujoin_exec_methods.c.RestrPosCustomScan	= NULL;
	gpujoin_exec_methods.c.ExplainCustomScan	= gpunestloop_explain;
	gpujoin_exec_methods.ExecCustomBulk			= gpunestloop_exec_bulk;

	/* hook registration */
	set_join_pathlist_next = set_join_pathlist_hook;
	set_join_pathlist_hook = gpujoin_add_join_path;
}
