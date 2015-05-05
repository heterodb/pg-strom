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
static bool					enable_gpuhashjoin;

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
	Size			kmrels_length;
	double			kresults_ratio;
	int				num_rels;
	List		   *host_quals;
	struct {
		double		rows;			/* rows to be generated in this depth */
		Cost		startup_cost;	/* outer scan cost + materialize */
		Cost		total_cost;		/* outer scan cost + materialize */
		JoinType	join_type;		/* one of JOIN_* */
		Path	   *scan_path;		/* outer scan path */
		List	   *hash_quals;		/* valid quals, if hash-join */
		List	   *join_quals;		/* all the device quals, incl hash_quals */
		double		nrows_ratio;
		double		kmrels_rate;
		Size		chunk_size;		/* kmrels_length * kmrels_ratio */
		int			nbatches;		/* expected iteration in this depth */
	} inners[FLEXIBLE_ARRAY_MEMBER];
} GpuJoinPath;

/*
 * GpuJoinInfo - private state object of CustomScan(GpuJoin)
 */
typedef struct
{
	int			num_rels;
	char	   *kern_source;
	int			extra_flags;
	List	   *used_params;
	bool		outer_bulkload;
	List	   *outer_quals;
	List	   *host_quals;
	/* for each depth */
	List	   *hash_outer_keys;
	List	   *join_quals;
	/* supplemental information of ps_tlist */
	List	   *ps_src_depth;	/* source depth of the ps_tlist entry */
	List	   *ps_src_resno;	/* source resno of the ps_tlist entry */
} GpuJoinInfo;

static inline void
form_gpujoin_info(CustomScan *cscan, GpuJoinInfo *gj_info)
{
	List   *privs = NIL;
	List   *exprs = NIL;

	privs = lappend(privs, makeInteger(gj_info->num_rels));
	privs = lappend(privs, makeString(gj_info->kern_source));
	privs = lappend(privs, makeInteger(gj_info->extra_flags));
	exprs = lappend(exprs, gj_info->used_params);
	privs = lappend(privs, makeInteger(gj_info->outer_bulkload));
	exprs = lappend(exprs, gj_info->outer_quals);
	exprs = lappend(exprs, gj_info->host_quals);
	exprs = lappend(exprs, gj_info->hash_outer_keys);
	exprs = lappend(exprs, gj_info->join_quals);
	privs = lappend(privs, gj_info->ps_src_depth);
	privs = lappend(privs, gj_info->ps_src_resno);

	cscan->custom_private = privs;
	cscan->custom_exprs = expr;
}

static inline GpuJoinInfo *
deform_gpujoin_info(CustomScan *cscan)
{
	GpuJoinInfo *gj_info = palloc0(sizeof(GpuJoinInfo));
	List	   *privs = cscan->custom_private;
	List	   *exprs = cscan->custom_exprs;
	int			pindex = 0;
	int			eindex = 0;

	gj_info->num_rels = intVal(list_nth(privs, pindex++));
	gj_info->kern_source = strVal(list_nth(privs, pindex++));
	gj_info->extra_flags = intVal(list_nth(privs, pindex++));
	gj_info->used_params = list_nth(exprs, eindex++);
	gj_info->outer_bulkload = intVal(list_nth(privs, pindex++));
	gj_info->outer_quals = list_nth(exprs, eindex++);
	gj_info->host_quals = list_nth(exprs, eindex++);
	gj_info->hash_outer_keys = list_nth(exprs, eindex++);
	gj_info->join_quals = list_nth(exprs, eindex++);
	gj_info->ps_src_depth = list_nth(privs, pindex++);
	gj_info->ps_src_resno = list_nth(privs, pindex++);

	return gj_info;
}

/*
 * GpuJoinState - execution state object of GpuJoin
 */
typedef struct
{
	GpuTaskState	gts;
	kern_parambuf  *kparams;
	


	List		   *join_types;
	ExprState	   *outer_quals;
	List		   *hash_outer_keys;
	List		   *join_quals;
	/*  */
	List		   *ps_src_depth;
	List		   *ps_src_resno;



} GpuNestLoopState;

/*
 * static function declaration
 */
static char *gpujoin_codegen(PlannerInfo *root,
							 CustomScan *gjoin,
							 GpuJoinInfo *gj_info,
							 codegen_context *context);


/*****/
static inline bool
path_is_gpujoin(Path *pathnode)
{
	CustpmPath *cpath = (CustpmPath *) pathnode;

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
	if (enable_gpuhashjoin && hash_qual != NIL)
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
	if (enable_gpunestloop &&
		(jointype == JOIN_INNER || jointype == JOIN_RIGHT))
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
 * build_flatten_qualifier
 *
 * It makes a flat AND expression that is equivalent to the given list.
 */
static Expr *
build_flatten_qualifier(List *clauses)
{
	List	   *args = NIL;
	ListCell   *lc;

	foreach (cell, clauses)
	{
		Expr   *expr = lfirst(lc);

		if (!expr)
			continue;
		Assert(exprType(expr) == BOOLOID);
		if (IsA(expr, BoolExpr) &&
			((BoolExpr *) expr)->boolop == AND_EXPR)
			args = list_concat(args, ((BoolExpr *) expr)->args);
		else
			args = lappend(args, expr);
	}
	if (list_length(args) == 0)
		return NULL;
	if (list_length(args) == 1)
		return linitial(args);
	return make_andclause(args);
}

/*
 * build_pseudo_targetlist
 *
 * constructor of pseudo-targetlist according to the expression tree
 * to be evaluated or returned. Usually, all we need to consider are
 * columns referenced by host-qualifiers and target-list. However,
 * we may need to execute device-qualifiers on CPU when device code
 * raised CpuReCheck error, so we also append columns (that is
 * referenced by device qualifiers only) in addition to the columns
 * referenced by host qualifiers. It has another benefit, because
 * it can share the data-structure regardless of CpuReCheck error.
 * Device code will generate full pseudo-scan data chunk, then we
 * can cut off the columns within scope of host references, if no
 * error was reported.
 */
typedef struct
{
	List		   *ps_tlist;
	List		   *ps_depth;
	List		   *ps_resno;
	GpuJoinPath	   *gpath;
	bool			resjunk;
} build_ps_tlist_context;

static bool
build_pseudo_targetlist_walker(Node *node, build_ps_tlist_context *context)
{
	GpuJoinPath	   *gpath = context->gpath;
	RelOptInfo	   *rel;
	ListCell	   *cell;

	if (!node)
		return false;
	if (IsA(node, Var))
	{
		Var	   *varnode = (Var *) node;
		Var	   *ps_node;
		int		ps_depth;
		Plan   *plan;

		foreach (cell, context->ps_tlist)
		{
			TargetEntry	   *tle = lfirst(cell);

			if (!IsA(tle->expr, Var))
				continue;

			ps_node = (Var *) tle->expr;
			if (ps_node->varno == varnode->varno &&
				ps_node->varattno == varnode->varattno &&
				ps_node->varlevelsup == varnode->varlevelsup)
			{
				/* sanity checks */
				Assert(ps_node->vartype == varnode->vartype &&
					   ps_node->vartypmod == varnode->vartypmod &&
					   ps_node->varcollid == varnode->varcollid);
				return false;
			}
		}
		/* not in the pseudo-scan targetlist, so append this one */
		rel = gpath->outer_path->parent;
		plan = gpath->outer_plan;
		if (bms_is_member(varnode->varno, rel->relids))
			ps_depth = 0;
		else
		{
			int		i;

			for (i=0; i < gpath->num_rels; i++)
			{
				rel = gpath->inners[i].scan_path->parent;
				plan = gpath->inners[i].scan_plan;
				if (bms_is_member(varnode->varno, rel->relids))
					break;
			}
			if (i == gpath->num_rels)
				elog(ERROR, "Bug? uncertain origin of Var-node: %s",
					 nodeToString(varnode));
			ps_depth = i + 1;
		}

		foreach (cell, plan->targetlist)
		{
			TargetEntry	   *tle = lfirst(cell);
			TargetEntry	   *tle_new;

			if (equal(varnode, tle->expr))
			{
				tle_new = makeTargetEntry((Expr *) copyObject(varnode),
										  list_length(context->ps_tlist) + 1,
										  NULL,
										  context->resjunk);
				context->ps_tlist = lappend(context->ps_tlist, tle_new);
				context->ps_depth = lappend_int(context->ps_depth, ps_depth);
				context->ps_resno = lappend_int(context->ps_resno, tle->resno);

				return false;
			}
		}
		elog(ERROR, "Bug? uncertain origin of Var-node: %s",
			 nodeToString(varnode));
	}
	return expression_tree_walker(node, build_pseudo_targetlist_walker,
								  (void *) context);
}

static List *
build_pseudo_targetlist(GpuJoinPath *gpath,
						GpuJoinInfo *gj_info,
						List *targetlist)
{
	build_ps_tlist_context context;

	memset(&context, 0, sizeof(build_ps_tlist_context));
	context.gpath   = gpath;
	context.resjunk = false;

	build_pseudo_targetlist_walker((Node *) targetlist, &context);
	build_pseudo_targetlist_walker((Node *) gj_info->host_quals, &context);

	/*
	 * Above are host referenced columns. On the other hands, the columns
	 * newly added below are device-only columns, so it will never
	 * referenced by the host-side. We mark it resjunk=true.
	 */
	context.resjunk = true;
	build_pseudo_targetlist_walker((Node *) gj_info->hash_quals, &context);
	build_pseudo_targetlist_walker((Node *) gj_info->join_quals, &context);
	build_pseudo_targetlist_walker((Node *) gj_info->outer_quals, &context);

    Assert(list_length(context.ps_tlist) == list_length(context.ps_depth) &&
           list_length(context.ps_tlist) == list_length(context.ps_resno));

	gj_info->ps_src_depth = context.ps_depth;
	gj_info->ps_src_resno = context.ps_resno;

	return context.ps_tlist;
}

#if 0
/*
 * fixup_device_expression
 *
 * Unlike host executable qualifiers, device qualifiers need to reference
 * relations before join, so varno of varnode needs to have either INNER
 * or OUTER_VAR, instead of pseudo-scan relation.
 */
typedef struct
{
	GpuJoinPath	   *gpath;
	CustomScan	   *gjoin;
} fixup_device_expr_context;

static Node *
fixup_device_expr_mutator(Node *node, fixup_device_expr_context *context)
{
	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		GpuJoinPath *gpath = context->gpath;
		Var		   *varnode = (Var *) node;
		Var		   *newnode;
		RelOptInfo *rel;
		Plan	   *plan;
		ListCell   *cell;
		int			i, depth;

		rel = gpath->outer_path->parent;
		plan = gpath->outer_plan;
		if (bms_is_member(varnode->varno, rel->relids))
			depth = 0;	/* outer relation */
		else
		{
			for (i=0; i < gpath->num_rels; i++)
			{
				rel = gpath->inners[i].scan_path->parent;
				plan = gpath->inners[i].scan_plan;
				if (bms_is_member(varnode->varno, rel->relids))
					goto found;
			}
			if (i == gpath->num_rels)
				elog(ERROR, "Bug? uncertain origin of Var-node: %s",
					 nodeToString(varnode));
			depth = i + 1;
		}

		foreach (cell, plan->targetlist)
		{
			TargetEntry	   *tle = lfirst(cell);

			if (equal(tle->expr, varnode))
			{
				newnode = copyObject(varnode);
				newnode->varnoold = varnode->varno;
				newnode->varoattno = varnode->varattno;
				newnode->varno = depth;
				newnode->varattno = tle->resno;

				return (Node *) newnode;
			}
		}
		elog(ERROR, "Bug? uncertain origin of Var-node: %s",
			 nodeToString(varnode));
	}
	return expression_tree_mutator(node, fixup_device_expr_mutator,
								   (void *) context);
}

static Node *
fixup_device_expr(CustomScan *gjoin, GpuJoinPath *gpath, Node *node)
{
	fixup_device_expr_context context;
	List	   *results = NIL;
	ListCell   *lc;

	if (!node)
		return NULL;

	memset(&context, 0, sizeof(fixup_device_expr_context));
	context.gpath = gpath;
	context.gjoin = gjoin;

	if (!IsA(node, List))
		return fixup_device_expression_mutator(node, &context);

	foreach (lc, (List *) node)
	{
		Node   *newnode
			= fixup_device_expr_mutator((Node *) lfirst(lc), &context);
		results = lappend(results, newnode);
	}
	return results;
}
#endif















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
	GpuJoinInfo		gj_info;
	CustomScan	   *gjoin;
	int				i;

	gjoin = makeNode(CustomScan);
	gjoin->scan.plan.targetlist = tlist;
	gjoin->scan.plan.qual = gpath->host_quals;
	gjoin->flags = best_path->flags;
	gjoin->methods = &gpujoin_plan_methods;

	memset(&gj_info, 0, sizeof(GpuJoinInfo));
	gj_info.num_rels = gpath->num_rels;
	gj_info.host_quals = gpath->host_quals;
	for (i=0; i < gpath->num_rels; i++)
	{
		CustomScan	   *mplan;
		List		   *hash_inner_keys = NIL;
		List		   *hash_outer_keys = NIL;

		foreach (lc, gpath->inners[i].hash_quals)
		{
			RestrictInfo   *rinfo = lfirst(lc);
			RelOptInfo	   *scan_rel = scan_path->parent;

			if (!rinfo->left_em || !rinfo->right_em)
				elog(ERROR, "Bug? EquivalenceMember was not set on %s",
					 nodeToString(rinfo));

			if (!bms_is_empty(rinfo->left_em->em_relids) &&
				bms_is_subset(rinfo->left_em->em_relids,
							  scan_rel->relids))
			{
				hash_inner_keys = lappend(hash_inner_keys,
										  rinfo->left_em->em_expr);
				hash_outer_keys = lappend(hash_outer_keys,
										  rinfo->right_em->em_expr);
			}
			else if (!bms_is_empty(rinfo->right_em->em_relids) &&
					 bms_is_subset(rinfo->right_em->em_relids,
								   scan_rel->relids))
			{
				hash_inner_keys = lappend(hash_inner_keys,
										  rinfo->right_em->em_expr);
				hash_outer_keys = lappend(hash_outer_keys,
										  rinfo->left_em->em_expr);
			}
			else
				elog(ERROR, "Bug? EquivalenceMember didn't fit GpuHashJoin");
		}
		mplan = pgstrom_create_multirels_plan(root,
											  i + 1,	/* depth */
											  gpath->inners[i].scan_path,
											  gpath->inners[i].join_type,
											  gpath->inners[i].buffer_size,
											  gpath->inners[i].buffer_portion,
											  gpath->inners[i].nbatches,
											  gpath->inners[i].nslots,
											  hash_inner_keys);
		/* add properties of GpuJoinInfo */
		gj_info.join_types = lappend(gj_info.join_types,
									 gpath->inners[i].join_type);
		gj_info.hash_outer_keys = lappend(gj_info.hash_outer_keys,
										  hash_outer_keys);
		gj_info.join_quals = lappend(gj_info.join_quals,
									 gpath->inners[i].join_quals);

		/* chain it under the GpuJoin */
		if (prev_plan)
			innerPlan(prev_plan) = &mplan->scan.plan;
		else
			innerPlan(gjoin) = &mplan->scan.plan;
		prev_plan = &mplan->scan.plan;
	}

	/*
	 * Creation of the underlying outer Plan node. In case of SeqScan,
	 * it may make sense to replace it with GpuScan for bulk-loading.
	 */
	outer_plan = create_plan_recurse(root, gpath->outer_path);
	if (IsA(outer_plan, SeqScan) || IsA(outer_plan, CustomScan))
	{
		Query	   *parse = root->parse;
		List	   *outer_quals = NIL;
		Plan	   *alter_plan;

		alter_plan = pgstrom_try_replace_plannode(outer_plan,
												  parse->rtable,
												  &outer_quals);
		if (alter_plan)
		{
			gj_info.outer_quals = build_flatten_qualifier(outer_quals);
			outer_plan = alter_plan;
		}
	}

	/* check bulkload availability */
	if (IsA(outer_plan, CustomScan))
	{
		int		custom_flags = ((CustomScan *) outer_plan)->flags;

		if ((custom_flags & CUSTOMPATH_SUPPORT_BULKLOAD) != 0)
			gj_info.outer_bulkload = true;
	}
	outerPlan(gjoin) = outer_plan;

	/*
	 * Build a pseudo-scan targetlist
	 */
	gjoin->custom_ps_tlist = build_pseudo_targetlist(gpath, &gj_info, tlist);

#if 0
	/*
	 * Fixup device execution 
	 *
	 * Is it really needed?
	 *
	 * TODO: We have to make host executable expression. Probably, it may
	 * reference INNER_VAR/OUTER_VAR according to context of the fallback
	 * routine.
	 */
	gj_info.outer_quals = (Expr *)
		fixup_device_expression(gjoin, gpath, (Node *)gj_info.outer_quals);
	gj_info.hash_outer_keys = (List *)
		fixup_device_expression(gjoin, gpath, (Node *)gj_info.hash_outer_keys);
	gj_info.hash_quals = (List *)
		fixup_device_expression(gjoin, gpath, (Node *)gj_info.hash_quals);
	gj_info.join_quals = (List *)
		fixup_device_expression(gjoin, gpath, (Node *)gj_info.join_quals);
#endif

	/*
	 * construct kernel code
	 */
	pgstrom_init_codegen_context(&context);

	gj_info.kern_source = gpujoin_codegen(root, gjoin, gj_info, &context);
	gj_info.extra_flags = context.extra_flags;
	gj_info.used_params = context.used_params;

	form_gpujoin_info(gjoin, &gj_info);

	return &gj_info->scan.plan;
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
gpujoin_create_scan_state(CustomScan *node)
{
	GpuContext	   *gcontext = pgstrom_get_gpucontext();
	GpuJoinState   *gjs;

	gjs = MemoryContextAllocZero(gcontext->memcxt, sizeof(GpuJoinState));
	NodeSetTag(gjs, T_CustomScanState);
    gjs->gts.css.flags = cscan->flags;
	gjs->gts.css.methods = &gpujoin_exec_methods.c;
	/* GpuTaskState setup */
	pgstrom_init_gputaskstate(gcontext, &gjs->gts);
	gjs->gts.cb_task_process = gpujoin_task_process;
	gjs->gts.cb_task_complete = gpujoin_task_complete;
	gjs->gts.cb_task_release = gpujoin_task_release;
	gjs->gts.cb_next_chunk = gpujoin_next_chunk;
	gjs->gts.cb_next_tuple = gpujoin_next_tuple;
	gjs->gts.cb_cleanup = NULL;

	return (Node *) gjs;
}

static void
gpujoin_begin(CustomScanState *node, EState *estate, int eflags)
{
	GpuJoinState   *gjs = (GpuJoinState *) node;
	PlanState	   *ps = &gjs->gts.css.ss.ps;
	CustomScan	   *gjoin = (CustomScan *) node->ss.ps.plan;
	GpuJoinInfo	   *gj_info = deform_gpujoin_info(cscan);

	/*
	 * NOTE: outer_quals, hash_outer_keys and join_quals are intended
	 * to use fallback routine if GPU kernel required host-side to
	 * retry a series of hash-join/nest-loop operation. So, we need to
	 * pay attention which slot is actually referenced.
	 * Right now, ExecEvalScalarVar can reference only three slots
	 * simultaneously (scan, inner and outer). So, varno of varnodes
	 * has to be initialized according to depth of the expression.
	 *
	 * TODO: we have to initialize above expressions carefully for
	 * CPU fallback implementation.
	 */
	gjs->join_types = gj_info->join_types;
	gjs->outer_quals = ExecInitExpr(gj_info->outer_quals, ps);
	gjs->hash_outer_keys = ExecInitExpr(gj_info->hash_outer_keys, ps);
	gjs->join_quals = ExecInitExpr(gj_info->join_quals, ps);
	gjs->gts.css.ss.ps.qual = ExecInitExpr(gj_info->host_quals, ps);

	/* needs to track corresponding columns */
	gjs->ps_src_depth = gj_info->ps_src_depth;
	gjs->ps_src_resno = gj_info->ps_src_resno;

	/*
	 * initialization of child nodes
	 */
	outerPlanState(gjs) = ExecInitNode(outerPlan(gjoin), estate, eflags);
	innerPlanState(gjs) = ExecInitNode(innerPlan(gjoin), estate, eflags);

	/*
	 * initialize kernel execution parameter
	 */
	gjs->gts.kern_source = gj_info->kern_source;
	gjs->gts.extra_flags = gj_info->extra_flags;
	gjs->kparams = pgstrom_create_kern_parambuf(gj_info->used_params,
												ps->ps_ExprContext);
	if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
		pgstrom_preload_cuda_program(&gjs->gts);

	/*
	 * initialize misc stuff
	 */
	if ((ghjs->gts.css.flags & CUSTOMPATH_PREFERE_ROW_FORMAT) != 0)
		gjs->result_format = KDS_FORMAT_ROW;
	else
		gjs->result_format = KDS_FORMAT_SLOT;
}

static TupleTableSlot *
gpujoin_exec(CustomScanState *node)
{
	return ExecScan(&node->ss,
					(ExecScanAccessMtd) pgstrom_exec_gputask,
					(ExecScanRecheckMtd) pgstrom_recheck_gputask);
}

static void
gpujoin_end(CustomScanState *node)
{
	GpuJoinState   *gjs = (GpuJoinState *) node;

	/*
	 * clean up subtree
	 */
	ExecEndNode(outerPlanState(node));
	ExecEndNode(innerPlanState(node));

	pgstrom_release_gputaskstate(&gjs->gts);
}

static void
gpujoin_rescan(CustomScanState *node)
{
	GpuJoinState   *gjs = (GpuJoinState *) node;

	/* clean-up and release any concurrent tasks */
	pgstrom_cleanup_gputaskstate(&gjs->gts);

	/* rewind the outer relation, also */
	gjs->gts.scan_done = false;
	gjs->outer_overflow = NULL;
	ExecReScan(outerPlanState(gjs));

	/*
	 * we reuse the inner hash table if it is flat (that means mhtables
	 * is not divided into multiple portions) and no parameter changed.
	 */
	if ((pmrels && pmrels->is_divided) ||
		innerPlanState(ghjs)->chgParam != NULL)
	{

		gjs->pmrels = NULL;
	}
}

static void
gpujoin_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
	GpuJoinState   *gjs = (GpuJoinState *) node;
	CustomScan	   *cscan = (CustomScan *) node->ss.ps.plan;
	GpuJoinInfo	   *gj_info = deform_gpujoin_info(cscan);
	List		   *context;
	ListCell	   *cell;
	char		   *temp;
	char			qlabel[128];

	initStringInfo(&str);

	/* name lookup context */
	context =  set_deparse_context_planstate(es->deparse_cxt,
											 (Node *) node,
											 ancestors);
	/* pseudo scan tlist if verbose */
	if (es->verbose)
	{
		ListCell   *cell;

		resetStringInfo(&str);
		foreach (cell, cscan->custom_ps_tlist)
		{
			TargetEntry	   *tle = lfirst(cell);

			temp = deparse_expression((Node *)tle->expr,
									  context, true, false);
			if (cell != list_head(cscan->custom_ps_tlist))
				appendStringInfo(&str, ", ");
			if (!tle->resjunk)
				appendStringInfo(&str, "%s", temp);
			else
				appendStringInfo(&str, "(%s)", temp);
		}
		ExplainPropertyText("Pseudo Scan", str.data, es);
	}

	/* outer bulkload */
	ExplainPropertyText("Bulkload", ghjs->outer_bulkload ? "On" : "Off", es);

	/* outer qualifier if any */
	if (gj_info->outer_quals)
	{
		temp = deparse_expression(gj_info->outer_quals,
								  context, es->verbose, false);
		ExplainPropertyText("OuterQual", temp, es);
	}

	/* join-qualifiers */
	depth = 1;
	forboth (lc1, gj_info->hash_outer_keys,
			 lc2, gj_info->join_quals)
	{
		Expr   *hash_outer_key = lfirst(lc1);
		Expr   *join_qual = lfirst(lc2);

		temp = deparse_expression(join_qual, context, es->verbose, false);
		snprintf(qlabel, sizeof(qlabel), "%s (depth %d)",
				 hash_outer_key != NIL ? "GpuHashJoin" : "GpuNestLoop",
				 depth);
		ExplainPropertyText(qlabel, temp, es);

		if (hash_outer_key)
		{
			temp = deparse_expression(hash_outer_key,
									  context, es->verbose, false);
			snprintf(qlabel, sizeof(qlabel), "HashKey (depth %d)", depth);
			ExplainPropertyText(qlabel, temp, es);
		}
	}
	/* host qualifier if any */
	if (gj_info->host_quals)
	{
		temp = deparse_expression(gj_info->host_quals,
								  context, es->verbose, false);
		snprintf(qlabel, sizeof(qlabel), "HostQual (depth %d)",
				 gj_info->num_rels);
		ExplainPropertyText(qlabel, temp, es);
	}
	/* other common field */
	pgstrom_explain_gputaskstate(&gjs->gts, es);
}

/*
 * codegen for:
 * STATIC_FUNCTION(cl_bool)
 * gpujoin_outer_quals(cl_int *errcode,
 *                     kern_parambuf *kparams,
 *                     kern_data_store *kds,
 *                     size_t kds_index)
 */
static void
gpujoin_codegen_outer_quals(StringInfo source,
							GpuJoinInfo *gj_info,
							codegen_context *context)
{
	appendStringInfo(
		source,
		"STATIC_FUNCTION(cl_bool)\n"
		"gpujoin_outer_quals(cl_int *errcode,\n"
		"                    kern_parambuf *kparams,\n"
		"                    kern_data_store *kds,\n"
		"                    size_t kds_index)\n"
		"{\n");
	if (!gj_info->outer_quals)
	{
		appendStringInfo(
			source,
			"  return true;\n");
	}
	else
	{
		List   *pseudo_tlist_saved = context->pseudo_tlist;
		Node   *outer_quals = (Node *) gj_info->outer_quals;
		char   *expr_text;

		context->pseudo_tlist = NIL;
		expr_text = pgstrom_codegen_expression(outer_quals, context);
		appendStringInfo(
			source,
			"%s%s\n"
			"  return EVAL(%s);\n",
			pgstrom_codegen_param_declarations(context),
			pgstrom_codegen_var_declarations(context),
			expr_text);
		context->pseudo_tlist = pseudo_tlist_saved;
	}
	appendStringInfo(
		source,
		"}\n\n");
}

/*
 * gpujoin_codegen_var_decl
 *
 * declaration of the variables in 'used_var' list
 */
static void
gpujoin_codegen_var_param_decl(StringInfo source,
							   GpuJoinInfo *gj_info,
							   int cur_depth,
							   codegen_context *context)
{
	bool		is_nestloop;
	bool		needs_kds_in = false;
	bool		needs_khtabls = false;
	List	   *kern_vars = NIL;
	ListCell   *cell;
	char	   *param_decl;
	int			cur_depth;

	Assert(cur_depth <= gj_info->num_rels);
	is_nestloop = (!list_nth(gj_info->hash_outer_keys, cur_depth - 1));

	/*
	 * Pick up variables in-use and append its properties in the order
	 * corresponding to depth/resno.
	 */
	foreach (cell, context->used_vars)
	{
		Var		   *varnode = lfirst(cell);
		Var		   *kernode = NULL;
		ListCell   *lc1;
		ListCell   *lc2;
		ListCell   *lc3;

		Assert(IsA(varnode, Var));
		forthree (lc1, context->pseudo_tlist,
				  lc2, gj_info->ps_src_depth,
				  lc3, gj_info->ps_src_resno)
		{
			TargetEntry	*tle = lfirst(lc1);
			int		src_depth = lfirst_int(lc2);
			int		src_resno = lfirst_int(lc3);

			if (equal(tle->expr, varnode))
			{
				kernode = copyObject(varnode);
				kernode->varno = src_depth;			/* save the source depth */
				kernode->varattno = src_resno;		/* save the source resno */
				kernode->varoattno = tle->resno;	/* resno on the ps_tlist */
				if (src_depth > 0 && src_depth < cur_depth)
				{
					if (list_nth(gj_info->hash_outer_keys, src_depth - 1))
						needs_khtable = true;
					else
						needs_kds_in = true;
				}
				break;
			}
		}
		if (!kernode)
			elog(ERROR, "Bug? device varnode was not is ps_tlist: %s",
				 nodeToString(varnode));

		/*
		 * attach 'kernode' in the order corresponding to depth/resno.
		 */
		lc2 = NULL;
		foreach (lc1, kern_vars)
		{
			Var	   *varnode = lfirst(lc1);

			if (varnode->varno > kernode->varno ||
				(varnode->varno == kernode->varno &&
				 varnode->varattno > kernode->varattno))
			{
				if (lc2 != NULL)
					lappend_cell(kern_vars, lc2, kernode);
				else
					kern_vars = lcons(kernode, kern_vars);
				break;
			}
			lc2 = lc1;
		}
	}

	/*
	 * variable declarations
	 */
	if (needs_kds_in)
		appendStrinfInfo(source, "  kern_data_store *kds_in;\n");
	if (needs_khtable)
		appendStrinfInfo(source, "  kern_hashtable *khtable;\n");
	appendStrinfInfo(source, "  void *datum;\n");

	foreach (cell, kern_vars)
	{
		Var			   *kernode = lfirst(cell);
		devtype_info   *dtype;

		dtype = pgstrom_devtype_lookup(kernode->vartype);
		if (!dtype)
			elog(ERROR, "device type \"%s\" not found",
				 format_type_be(kernode->vartype));

		appendStringInfo(
			source,
			"  pg_%s_t KVAR_%u;\n",
			dtype->type_name,
			kernode->varoattno);
	}

	/*
	 * parameter declaration
	 */
	param_decl = pgstrom_codegen_param_declarations(context);
	appendStrinfInfo(source, "%s", param_decl);

	/*
	 * variable initialization
	 */
	cur_depth = -1;
	foreach (cell, kern_vars)
	{
		Var			   *kernode = lfirst(cell);
		devtype_info   *dtype;

		dtype = pgstrom_devtype_lookup(kernode->vartype);
		if (!dtype)
			elog(ERROR, "device type \"%s\" not found",
				 format_type_be(kernode->vartype));

		if (cur_depth != kernode->varno)
		{
			if (kernode->varno == 0)
			{
				/* htup from KDS */
				appendStringInfo(
					"  /* variable load in depth-0 (outer KDS) */\n"
					"  colmeta = kds->colmeta;\n"
					"  htup = (rbuffer[0] == 0\n"
					"    ? NULL\n"
					"    : &((kern_tupitem *)\n"
					"        ((char *)kds + rbuffer[0]))->htup);\n");
			}
			else if (list_nth(context->hash_outer_keys, kernode->varno - 1))
			{
				/* in case of inner hash table */
				appendStringInfo(
					source,
					"  /* variables load in depth-%u (hash table) */\n"
					"  khtable = KERN_MULTIRELS_INNER_HASH(kmrels, %u);\n"
					"  assert(khtable != NULL);\n"
					"  colmeta = khtable->colmeta;\n",
					kernode->varno,
					kernode->varno);
				if (kernode->varno < depth)
					appendStringInfo(
						source,
						"  htup = (rbuffer[%d] == 0\n"
						"    ? NULL\n"
						"    : &((kern_tupitem *)\n"
						"        ((char *)khtable + rbuffer[%d]))->htup);\n",
						kernode->varno,
						kernode->varno);
				else if (kernode->varno == depth)
					appendStringInfo(
						source,
						"  htup = (!inner_tupitem\n"
						"    ? NULL\n"
						"    : &inner_tupitem->htup);\n"
						);
				else
					elog(ERROR, "Bug? too deeper varnode reference");
			}
			else
			{
				/* in case of inner data store */
				appendStringInfo(
					source,
					"  /* variable load in depth-%u (data store) */\n"
					"  kds_in = KERN_MULTIRELS_INNER_KDS(kmrels, %u);\n"
					"  assert(kds_in != NULL);\n"
					"  colmeta = kds_in->colmeta;\n",
					kernode->varno,
					kernode->varno);
				if (kernode->varno < depth)
					appendStringInfo(
						source,
						"  htup = (rbuffer[%d] == 0\n"
						"    ? NULL\n"
						"    : &((kern_tupitem *)\n"
						"        ((char *)kds_in + rbuffer[%d]))->htup);\n",
						kernode->varno,
                        kernode->varno);
				else if (kernode->varno == depth)
					appendStringInfo(
						source,
						"  htup = (!inner_tupitem\n"
						"    ? NULL\n"
						"    : &inner_tupitem->htup);\n"
						);
				else
					elog(ERROR, "Bug? too deeper varnode reference");
			}
			cur_depth = devnode->varno;
		}

		if (is_nestloop)
		{
			appendStringInfo(
				source,
				"  if (get_local_%s() == 0)\n"
				"  {\n"
				"    datum = (!htup\n"
				"      ? NULL\n"
				"      : kern_get_datum_tuple(colmeta, htup, %u));\n"
				"    SHARED_WORKMEM(pg_%s_t)[get_local_%s()] =\n"
				"      pg_%s_datum_ref(errcode, datum, false);\n"
				"  }\n"
				"  __syncthreads();\n"
				"  KVAR_%u = SHARED_WORKMEM(pg_%s_t)[get_local_%s()];\n",
				kernode->varno == cur_depth ? "xid" : "yid",
				kernode->varattno - 1,
				dtype->type_name,
				kernode->varno == cur_depth ? "yid" : "xid",
				dtype->type_name,
				kernode->varoattno,
				dtype->type_name,
				kernode->varno == cur_depth ? "yid" : "xid");
		}
		else
		{
			appendStringInfo(
				source,
				"  datum = (!htup\n"
				"    ? NULL\n"
				"    : kern_get_datum_tuple(colmeta, htup, %u));\n"
				"  KVAR_%u = pg_%s_datum_ref(errcode,datum,false);\n",
				kernode->varattno - 1,
				kernode->varoattno,
				dtype->type_name);
		}
	}
	appendStringInfoChar(source, '\n');
}

/*
 * codegen for:
 * STATIC_FUNCTION(cl_bool)
 * gpujoin_join_quals_depth%u(cl_int *errcode,
 *                            kern_parambuf *kparams,
 *                            kern_data_store *kds,
 *                            kern_multi_relstore *kmrels,
 *                            cl_int *outer_index,
 *                            kern_tupitem *tupitem);
 */
static void
gpujoin_codegen_join_quals(StringInfo source,
						   GpuJoinInfo *gj_info,
						   int cur_depth,
						   codegen_context *context)
{
	List	   *join_qual;
	char	   *join_code;

	Assert(cur_depth > 0 && cur_depth <= gj_info->num_rels);
	join_qual = list_nth(gj_info->join_quals, cur_depth);

	/*
	 * make a text representation of join_qual
	 */
	context->used_vars = NIL;
	context->param_refs = NULL;
	join_code = pgstrom_codegen_expression((Node *) join_qual, context);

	/*
	 * function declaration
	 */
	appendStrinfInfo(
		source,
		"STATIC_FUNCTION(cl_bool)\n"
		"gpujoin_join_quals_depth%d(cl_int *errcode,\n"
		"                           kern_parambuf *kparams,\n"
		"                           kern_data_store *kds,\n"
        "                           kern_multi_relstore *kmrels,\n"
		"                           cl_int *outer_index,\n"
		"                           kern_tupitem *inner_tupitem)\n"
		"{\n"
		"  kern_colmeta *colmeta;\n"
		"  HeapTupleHeaderData *htup;\n",
		cur_depth);
	/*
	 * variable/params declaration & initialization
	 */
	gpujoin_codegen_var_param_decl(source, gj_info, cur_depth, context);

	/*
	 * evaluate join qualifier
	 */
	appendStrinfInfo(
		source,
		"  return EVAL(%s);\n"
		"}\n\n",
		join_code);
}

/*
 * codegen for:
 * STATIC_FUNCTION(cl_uint)
 * gpujoin_hash_value_depth%u(cl_int *errcode,
 *                            kern_parambuf *kparams,
 *                            kern_data_store *kds,
 *                            kern_multi_relstore *kmrels,
 *                            cl_int *outer_index);
 */
static void
gpujoin_codegen_hash_value(StringInfo source,
						   GpuJoinInfo *gj_info,
						   int cur_depth,
						   codegen_context *context)
{
	StringInfoData	body;
	List		   *hash_outer_keys;

	Assert(cur_depth > 0 && cur_depth <= gj_info->num_rels);
	hash_outer_keys = list_nth(gj_info->hash_outer_keys, cur_depth - 1);
	Assert(hash_outer_keys != NIL);

	appendStringInfo(
		source,
		"STATIC_FUNCTION(cl_uint)\n"
		"gpujoin_hash_value_depth%u(cl_int *errcode,\n"
		"                           kern_parambuf *kparams,\n"
		"                           kern_data_store *kds,\n"
		"                           kern_multi_relstore *kmrels,\n"
		"                           cl_int *outer_index)\n"
		"{\n"
		"  cl_uint hash;\n"
		cur_depth);

	contect->used_vars = NIL;
	context->param_refs = NULL;

	initStringInfo(&body);
	appendStringInfo(&body, "INIT_CRC32C(hash);\n");
	foreach (cell, hash_outer_keys)
	{
		Node	   *key_expr = lfirst(cell);
		Oid			key_type;
		devtype_info *dtype;
		char	   *temp;

		temp = pgstrom_codegen_expression(key_expr, context);
		appendStringInfo(
			&body,
			"  hash = pg_%s_comp_crc32(pg_crc32_table, hash, %s);\n",
			dtype->type_name,
			temp);
		pfree(temp);
	}
	appendStringInfo(body, "  FIN_CRC32C(hash);\n");

	/*
	 * variable/params declaration & initialization
	 */
	gpujoin_codegen_var_param_decl(source, gj_info, cur_depth, context);

	appendStringInfo(
		source,
		"%s"
		"  return hash;\n"
		"}\n",
		body.data);
	pfree(body.data);
}

/*
 * codegen for:
 * STATIC_FUNCTION(cl_int)
 * gpujoin_projection_forward(cl_int src_depth,
 *                            cl_int src_colidx);
 */
static void
gpujoin_codegen_projection_forward(StringInfo source,
								   GpuJoinInfo *gj_info)
{
	ListCell   *lc1;
	ListCell   *lc2;
	ListCell   *lc3;
	int			depth;

	/* forward */
	appendStringInfo(
		source,
		"STATIC_FUNCTION(cl_int)\n"
		"gpujoin_projection_forward(cl_int src_depth,\n"
		"                           cl_int src_colidx)\n"
		"{\n"
		"  switch (src_depth)\n"
		"  {\n");
	for (depth=0; depth <= gj_info->num_rels; depth++)
	{
		appendStringInfo(
			source,
			"  case %d:\n"
			"    switch (src_colidx)\n"
			"    {\n"
			depth);

		forthree(lc1, context->pseudo_tlist,
				 lc2, gj_info->ps_src_depth,
				 lc3, gj_info->ps_src_resno)
		{
			TargetEntry *tle = lfirst(lc1);
			int		src_depth = lfirst_int(lc2);
			int		src_resno = lfirst_int(lc3);

			if (src_depth == depth)
			{
				appendStringInfo(
					source,
					"    case %d:\n"
					"      return %d;\n",
					src_resno - 1,
					tle->resno - 1);
			}
		}
		appendStringInfo(
			source,
			"    default:\n"
			"      break;\n"
			"    }\n");
	}
	appendStringInfo(
		source,
		"  default:\n"
		"    break;\n"
		"  }\n"
		"  return -1;\n"
		"}\n\n");
}

/*
 * codegen for:
 * STSTIC_FUNCTION(void)
 * gpujoin_projection_mapping(cl_int dest_resno,
 *                            cl_int *src_depth,
 *                            cl_int *src_colidx);
 */
static void
gpujoin_codegen_projection_mapping(StringInfo source,
								   GpuJoinInfo *gj_info)
{
	ListCell   *lc1;
	ListCell   *lc2;
	ListCell   *lc3;

	appendStringInfo(
		source,
		"STSTIC_FUNCTION(void)\n"
		"gpujoin_projection_mapping(cl_int dest_colidx,\n"
		"                           cl_int *src_depth,\n"
		"                           cl_int *src_colidx)\n"
		"{\n"
		"  switch (dest_colidx)\n"
		"  {\n");

   	forthree(lc1, context->pseudo_tlist,
   			 lc2, gj_info->ps_src_depth,
			 lc3, gj_info->ps_src_resno)
	{
		TargetEntry *tle = lfirst(lc1);
		int		src_depth = lfirst_int(lc2);
		int		src_resno = lfirst_int(lc3);

		appendStringInfo(
			source,
			"  case %d:\n"
			"    *src_depth = %d;\n"
			"    *src_colidx = %d;\n"
			"    break;\n",
			tle->resno - 1,
			src_depth,
			src_resno - 1);
	}
	appendStringInfo(
		source,

		"  }\n"
		"}\n\n");
}

static char *
gpujoin_codegen(PlannerInfo *root,
				CustomScan *gjoin,
				GpuJoinInfo *gj_info,
				codegen_context *context)
{
	StringInfoData source;
	const char *args;
	int			depth;
	ListCell   *cell;

	initStringInfo(&source);

	/* gpujoin_outer_quals  */
	gpujoin_codegen_outer_quals(&source, gj_info, context);

	/* gpujoin_join_quals */
	for (depth=1; depth <= gj_info->num_rels; depth++)
		gpujoin_codegen_join_quals(&source, gj_info, depth, context);
	appendStrinfInfo(
		&source,
		"STATIC_FUNCTION(cl_bool)\n"
		"gpujoin_join_quals(cl_int *errcode,\n"
		"                   kern_parambuf *kparams,\n"
		"                   kern_data_store *kds,\n"
		"                   kern_multi_relstore *kmrels,\n"
		"                   int depth,\n"
		"                   cl_int *outer_index,\n"
		"                   HeapTupleHeaderData *inner_htup)\n"
		"{\n"
		"  switch (depth)\n"
		"  {\n");

	args = "errcode, kparams, kds, kmrels, outer_index, inner_htup";
	for (depth=1; depth <= gj_info->num_rels; depth++)
	{
		appendStrinfInfo(
			&source,
			"  case %d:\n"
			"    return gpujoin_join_quals_depth%d(%s);\n",
			depth, depth, args);
	}
	appendStrinfInfo(
		&source,
		"  default:\n"
		"    STROM_SET_ERROR(errcode, StromError_SanityCheckViolation);\n"
		"    break;\n"
		"  }\n"
		"  return false;\n"
		"}\n\n");

	/* gpujoin_hash_value */
	appendStrinfInfo(
		&source,
		"STATIC_FUNCTION(cl_uint)\n"
		"gpujoin_hash_value(cl_int *errcode,\n"
		"                   kern_parambuf *kparams,\n"
		"                   kern_data_store *kds,\n"
		"                   kern_multi_relstore *kmrels,\n"
		"                   cl_int depth,\n"
		"                   cl_int *outer_index)\n"
		"{\n"
		"  switch (depth)\n"
		"  {\n");

	depth = 1;
	foreach (cell, gj_info->hash_outer_keys)
	{
		if (lfirst(cell) != NULL)
			gpujoin_codegen_hash_value(&source, gj_info, depth, context);
		depth++;
	}

	appendStrinfInfo(
		&source,
		"  default:\n"
		"    STROM_SET_ERROR(errcode, StromError_SanityCheckViolation);\n"
		"    break;\n"
		"  }\n"
		"  return (cl_uint)(-1);\n"
		"}\n");

	/* gpujoin_projection_mapping */
	gpujoin_codegen_projection_mapping(&source, gj_info);

	return source.data;
}

static GpuTask *
gpujoin_next_chunk(GpuTaskState *gts)
{
	GpuJoinState   *gjs = (GpuJoinState *) gts;
	PlanState	   *outer_node = outerPlanState(gjs);
	TupleDesc		tupdesc = ExecGetResultType(outer_node);
	pgstrom_data_store *pds = NULL;
	int				result_format = gjs->result_format;
	struct timeval	tv1, tv2, tv3;

	/*
     * Logic to fetch inner multi-relations looks like nested-loop.
     * If all the underlying inner scan already scaned its outer
	 * relation, current depth makes advance its scan pointer with
	 * reset of underlying scan pointer, or returns NULL if it is
	 * already reached end of scan.
     */
retry:
	PERFMON_BEGIN(&gts->pfm_accum, &tv1);

	if (gjs->gts.scan_done || !gjs->pmrels)
	{
		PlanState  *mrs = innerPlanState(gjs);
		void	   *pmrels;

		/* unlink previous inner multi-relations */
		if (gjs->pmrels)
		{
			Assert(gjs->gts.scan_done);
			multirels_put_buffer(gjs->gts.gcontext, gjs->pmrels);
			gjs->pmrels = NULL;
		}

		/* load an inner multi-relations buffer */
		pmrels = BulkExecMultiRels(mrs);
		if (!pmrels)
		{
			PERFMON_END(&gts->pfm_accum,
						time_inner_load, &tv1, &tv2);
			return NULL;	/* end of inner multi-relations */
		}
		gjs->pmrels = pmrels;

		/*
		 * Rewind the outer scan pointer, if it is not first time
		 */
		if (gjs->gts.scan_done)
		{
			ExecReScan(outerPlanState(gjs));
			gjs->gts.scan_done = false;
		}
	}
	PERFMON_BEGIN(&gts->pfm_accum, &tv2);

	if (!gjs->outer_bulkload)
	{
		while (true)
		{
			TupleTableSlot *slot;

			if (gjs->gts.scan_overflow)
			{
				slot = gjs->gts.scan_overflow;
				gjs->gts.scan_overflow = slot;
			}
			else
			{
				slot = ExecProcNode(outer_node);
				if (TupIsNull(slot))
				{
					gjs->gts.scan_done = true;
					break;
				}
			}

			/* create a new data-store if not constructed yet */
			if (!pds)
			{
				pds = pgstrom_create_data_store_row(gjs->gts.gcontext,
													tupdesc,
													pgstrom_chunk_size(),
													false);
			}

			/* insert the tuple on the data-store */
			if (!pgstrom_data_store_insert_tuple(pds, slot))
			{
				gjs->gts.scan_overflow = slot;
				break;
			}
		}
	}
	else
	{
		pds = BulkExecProcNode(outer_node);
		if (!pds)
			gjs->gts.scan_done = true;
	}
	PERFMON_END(&gjs->gts.pfm_accum, time_inner_load, &tv1, &tv2);
	PERFMON_END(&gjs->gts.pfm_accum, time_outer_load, &tv2, &tv3);

	/*
	 * We also need to check existence of next inner hash-chunks, even if
	 * here is no more outer records, In case of multi-relations splited-out,
	 * we have to rewind the outer relation scan, then makes relations
	 * join with the next inner hash chunks.
	 */
    if (!pds)
        goto retry;

	return gpuhashjoin_create_task(gjs, pds, result_format);
}

static TupleTableSlot *
gpujoin_next_tuple(GpuTaskState *gts)
{
	GpuJoinState	   *gjs = (GpuJoinState *) gts;
	TupleTableSlot	   *slot = gjs->gts.css.ss.ss_ScanTupleSlot;
	pgstrom_gpujoin	   *gjoin = (pgstrom_gpujoin *)gjs->gts.curr_chunk;
	pgstrom_data_store *pds_dst = gpuhashjoin->pds_dst;
	kern_data_store	   *kds_dst = pds_dst->kds;
	struct timeval		tv1, tv2;

	PERFMON_BEGIN(&gjs->gts.pfm_accum, &tv1);

	if (gjs->gts.curr_index < kds_dst->nitems)
	{
		int		index = gjs->gts.curr_index++;

		/* fetch a result tuple */
		pgstrom_fetch_data_store(slot,
								 pds_dst,
								 index,
								 &gjs->curr_tuple);
		/*
		 * NOTE: host-only qualifiers are checked during ExecScan(),
		 * so we don't check it here by itself.
		 */
	}
	else
		ExecClearTuple(slot);

	PERFMON_END(&ghjs->gts.pfm_accum, time_materialize, &tv1, &tv2);
	return slot;
}






static void
gpujoin_task_release(GpuTask *gtask)
{}


static bool
gpujoin_task_complete(GpuTask *gtask)
{}

static bool
gpujoin_task_process(GpuTask *gtask)
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
	/* setup path methods */
	gpujoin_path_methods.CustomName				= "GpuJoin";
	gpujoin_path_methods.PlanCustomPath			= create_gpujoin_plan;
	gpujoin_path_methods.TextOutCustomPath		= gpujoin_textout_path;

	/* setup plan methods */
	gpujoin_methods.CustomName					= "GpuJoin";
	gpujoin_plan_methods.CreateCustomScanState	= gpujoin_create_scan_state;
	gpujoin_plan_methods.TextOutCustomScan		= NULL;

	/* setup exec methods */
	gpujoin_exec_methods.c.CustomName			= "GpuNestedLoop";
	gpujoin_exec_methods.c.BeginCustomScan		= gpujoin_begin;
	gpujoin_exec_methods.c.ExecCustomScan		= gpujoin_exec;
	gpujoin_exec_methods.c.EndCustomScan		= gpujoin_end;
	gpujoin_exec_methods.c.ReScanCustomScan		= gpujoin_rescan;
	gpujoin_exec_methods.c.MarkPosCustomScan	= NULL;
	gpujoin_exec_methods.c.RestrPosCustomScan	= NULL;
	gpujoin_exec_methods.c.ExplainCustomScan	= gpujoin_explain;
	gpujoin_exec_methods.ExecCustomBulk			= gpujoin_exec_bulk;

	/* hook registration */
	set_join_pathlist_next = set_join_pathlist_hook;
	set_join_pathlist_hook = gpujoin_add_join_path;
}
