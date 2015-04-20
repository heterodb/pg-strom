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
static bool	gpunestloop_task_process(GpuTask *gtask);
static bool	gpunestloop_task_complete(GpuTask *gtask);
static void	gpunestloop_task_release(GpuTask *gtask);
static GpuTask *gpunestloop_next_chunk(GpuTaskState *gts);
static TupleTableSlot *gpunestloop_next_tuple(GpuTaskState *gts);

/* static variables */
static set_join_pathlist_hook_type set_join_pathlist_next;
static CustomPathMethods	gpunestloop_path_methods;
static CustomScanMethods	gpunestloop_plan_methods;
static PGStromExecMethods	gpunestloop_exec_methods;
static CustomScanMethods	multitables_plan_methods;
static PGStromExecMethods	multitables_exec_methods;
static bool					enable_gpunestloop;

/*
 * GpuNestLoopPath
 *
 * 
 *
 */
typedef struct
{
	CustomPath		cpath;
	Path		   *outer_path;
	Size			inner_relsize;
	double			row_population_ratio;
	int				num_rels;
	struct {
		Path	   *scan_path;
		JoinType	join_type;
		List	   *join_clause;
		double		threshold;
		int			nloops;
	} inners[FLEXIBLE_ARRAY_MEMBER];
} GpuNestLoopPath;

/*
 * GpuNestLoopInfo - state object of CustomScan(GpuNestedLoop)
 */
typedef struct
{
	int			num_rels;
	Size		inner_relsize;
	double		row_population_ratio;
	char	   *kern_source;
	int			extra_flags;
	bool		outer_bulkload;	/* is outer can bulk loadable? */
	Expr	   *outer_quals;	/* qualifier of outer scan, if any */
	List	   *join_types;		/* list of join types */
	List	   *join_clauses;		/* list of join quals */
	List	   *used_params;	/* template for kparams */
	/* supplemental information for ps_tlist */
	List	   *ps_src_depth;	/* source depth of the ps_tlist entry */
	List	   *ps_src_resno;	/* source resno of the ps_tlist entry */
} GpuNestLoopInfo;

static inline void
form_gpunestloop_info(CustomScan *cscan, GpuNestLoopInfo *gnl_info)
{}

static inline GpuNestLoopInfo *
deform_gpunestloop_info(CustomScan *cscan)
{}

/*
 * GNLInnerInfo - state object of CustomScan(GNLInner)
 */
typedef struct
{

} GNLInnerInfo;

static inline void
form_gnl_inner_info(CustomScan *cscan, GNLInnerInfo *ginner_info)
{}

static inline GNLInnerInfo *
deform_gnl_inner_info(CustomScan *cscan)
{}

/*
 * GpuNestLoopState - execution state object of GpuNestLoop
 */
typedef struct
{
	GpuTaskState	gts;
	List		   *join_types;
	ExprState	   *outer_quals;
	List		   *join_clauses;

} GpuNestLoopState;

/*
 * GNLInnerState - execution state object of GpuNestedLoopInner
 */
typedef struct
{
	CustomScanState	css;
	GpuContext	   *gcontext;
	int				depth;
	double			threshold;
} GNLInnerState;





/*
 * estimate_innertables_size
 *
 * It estimates size of the inner multi-tables buffer
 */
static bool
estimate_innertables_size(PlannerInfo *root,
						  GpuNestedLoopPath *gpath,
						  Relids required_outer,
						  JoinCostWorkspace *workspace)
{
	Size		inner_relsize;
	int			num_batches;
	int			num_batches_init = -1;
	Cost		startup_cost;
	Cost		total_cost;

retry:
	inner_relsize = LONGALIGN(offsetof(kern_multitables,
									   tables[gpath->num_rels]));
	numbatches = 1;
	largest_size = 0;
	i_largest = -1;
	for (i=0; i < gpath->num_rels; i++)
	{
		Path	   *inner_path = gpath->inners[i].scan_path;
		RelOptInfo *inner_rel = inner_path->parent;
		Size		table_size;
		Size		tuple_size;

		numbatches *= gpath->inners[i].nloops;

		/* force a plausible relation size if no information.
		 * It expects 15% of margin to avoid unnecessary multi-
		 * tables split
		 */
		ntuples = Max(1.15 * inner_path->rows, 1000.0);
		if (gpath->inners[i].nloops > 0)
			ntuples /= (double) gpath->inners[i].nloops;

		/*
		 * estimate length of each table entry
		 */
		tuple_size = (MAXALIGN(offsetof(kern_tupitem, htup)) +
					  MAXALIGN(offsetof(HeapTupleHeaderData,
										t_bits[BITMAPLEN(ncols)])) +
					  MAXALIGN(inner_rel->width));
		table_size = (MAXALIGN(offsetof(kern_data_store, colmeta[ncols])) +
					  MAXALIGN(sizeof(cl_uint) * ntuples) +
					  MAXALIGN(tuple_size * ntuples));

		if (largest_size < table_size)
		{
			largest_size = table_size;
			i_largest = i;
		}
		gpath->inners[i].table_size = table_size;

		/* expand estimated inner multi-tables size */
		inner_relsize += table_size;
	}
	if (num_batches_init < 0)
		num_batches_init = num_batches;

	/* also, compute threshold of each chunk */
	threshold_size = 0;
	for (i = gpath->num_rels - 1; i >= 0; i--)
	{
		threshold_size += gpath->inners[i].table_size;
		gpath->inners[i].threshold
			= (double) threshold_size / (double) inner_relsize;
	}

	/*
	 * NOTE: In case when extreme number of rows are expected,
	 * it does not make sense to split inner multi-tables because
	 * increasion of numbatches also increases the total cost by
	 * iteration of outer scan. In this case, the best strategy
	 *is to give up this path, instead of incredible number of
	 * numbatches!
	 */
	startup_cost = workspace->startup_cost;
	total_cost = (workspace->startup_cost +
				  workspace->run_cost * ((double) num_batches /
										 (double) num_batches_init));
	if (!add_path_precheck(gpath->cpath.path.parent,
						   startup_cost, total_cost,
						   NULL, required_outer))
		return false;

	/*
	 * If size of inner multi-tables is still larger than device
	 * allocatable limitation, we try to split the largest table
	 * then retry the estimation.
	 */
	if (inner_relsize > gpuMemMaxAllocSize())
	{
		gpath->inners[i_largest].nloops++;
		goto retry;
	}

	/*
	 * Update estimated hashtable_size, but ensure hashtable_size
	 * shall be allocated at least
	 */
	gpath->inner_relsize = Max(inner_relsize, pgstrom_chunk_size());

	/*
     * Update JoinCostWorkspace according to numbatches
     */
	workspace->run_cost *= ((double) num_batches /
							(double) num_batches_init);
	workspace->total_cost = workspace->startup_cost + workspace->run_cost;

	return true;	/* ok */
}

/*
 * initial_cost_gpunestloop
 *
 * routine of initial rough cost estimation for GpuNestedLoop
 */
static bool
initial_cost_gpunestloop(PlannerInfo *root,
						 GpuNestedLoopPath *gpath,
						 Relids required_outer,
						 JoinCostWorkspace *workspace)
{
	Path	   *outer_path;
	Path	   *inner_path;
	Cost		startup_cost;
	Cost		run_cost;
	int			num_rels;
	List	   *join_clause;
	double		row_population_ratio;

	num_rels = gpath->num_rels;
	join_clause = gpath->inners[num_rels - 1].join_clause;
	outer_path = gpath->outer_path;
	inner_path = gpath->inners[num_rels - 1].scan_path;

	/*
	 * Cost to construct inner multi-tables; to be processed by CPU
	 */
	startup_cost = outer_path->startup_cost + inner_path->startup_cost;
	if (num_rels == 1)
		startup_cost += pgstrom_gpu_setup_cost;
	startup_cost += cpu_tuple_cost * inner_path->rows;

	/*
	 * Cost to run GpuNestedLoop logic
	 */
	run_cost = outer_path->total_cost - outer_path->startup_cost;
	run_cost += (pgstrom_gpu_operator_cost *
				 list_length(join_clause) *
				 outer_path->rows *
				 inner_path->rows);
	/*
	 * Setup join cost workspace
	 */
	workspace->startup_cost = startup_cost;
	workspace->run_cost = run_cost;
	workspace->total_cost = startup_cost + run_cost;
	workspace->numbatches = 1;

	/*
	 * estimation of row-population ratio; that is a ratio between
	 * output rows and input rows come from outer relation stream.
	 * We use approx_tuple_count here because we need an estimation
	 * based JOIN_INNER semantics.
	 */
	row_population_ratio = gpath->cpath.path.rows / outer_path->rows;
	if (row_population_ratio > pgstrom_row_population_max)
	{
		elog(DEBUG1, "row population ratio (%.2f) too large, give up",
			 row_population_ratio);
		return false;
	}
	else if (row_population_ratio > pgstrom_row_population_max / 2.0)
	{
		elog(NOTICE, "row population ratio (%.2f) too large, rounded to %.2f",
			 row_population_ratio, pgstrom_row_population_max / 2.0);
		row_population_ratio = pgstrom_row_population_max / 2.0;
	}
	gpath->row_population_ratio = row_population_ratio;

	/*
	 * Estimation of inner multitable size and number of outer loops
	 * according to the split of kern_data_store.
	 * In case of estimated plan cost is too large to win the existing
	 * paths, it breaks to find out this path.
	 */
	if (!estimate_innertables_size(root, gpath, required_outer, workspace))
		return false;

	return true;
}

/*
 * final_cost_gpunestloop
 *
 * routine of final detailed cost estimation for GpuNestedLoop
 */
static void
final_cost_gpunestloop(PlannerInfo *root,
					   GpuNestedLoopPath *gpath,
					   JoinCostWorkspace *workspace)
{
	
	
	
}

static void
try_gpunestloop_path(PlannerInfo *root,
					 RelOptInfo *joinrel,
					 JoinType jointype,
					 SpecialJoinInfo *sjinfo,
					 Relids param_source_rels,
					 Relids extra_lateral_rels,
					 Path *outer_path,
					 Path *inner_path,
					 List *restrict_clauses)
{
	GpuNestLoop		   *gpath;
	ParamPathInfo	   *param_info;
	JoinCostWorkspace	workspace;
	Relids				required_outer;
	bool				support_bulkload = true;

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
	 * Check availability of bulkload in this joinrel. If child GpuNestedLoop
	 * is merginable, both of nodes have to support bulkload.
	 */
	foreach (lc, joinrel->reltargetlist)
	{
		Expr	   *expr = lfirst(lc);

		if (!IsA(expr, Var) &&
			!pgstrom_codegen_available_expression(expr))
		{
			support_bulkload = false;
			break;
		}
	}

	/*
	 * Creation of GpuNestedLoop Path, without merging child GpuNestedLoop
	 */
	gpath = palloc0(offsetof(GpuNestedLoopPath, inners[1]));
	NodeSetTag(gpath, T_CustomPath);
	gpath->cpath.path.pathtype = T_CustomScan;
	gpath->cpath.path.parent = joinrel;
	gpath->cpath.path.param_info = param_info =
		get_joinrel_parampathinfo(root,
								  joinrel,
								  outer_path,
								  inner_path,
								  sjinfo,
								  bms_copy(required_outer),
								  &restrict_clauses);
	gpath->cpath.path.rows = (param_info
							  ? param_info->ppi_rows
							  : joinrel->rows);
	gpath->cpath.path.pathkeys = NIL;
	gpath->cpath.flags = (support_bulkload ? CUSTOMPATH_SUPPORT_BULKLOAD : 0);
	gpath->cpath.methods = &gpunestloop_path_methods;
	gpath->outer_path = outer_path;
	gpath->num_rels = 1;
	gpath->inners[0].scan_path = inner_path;
	gpath->inners[0].jointype = jointype;
	gpath->inners[0].join_clause = restrict_clauses;
	gpath->inners[0].nloops = 1;

	/* cost estimation and check availability */
	if (initial_cost_gpunestloop(root, gpath, required_outer, &workspace))
	{
		if (add_path_precheck(joinrel,
							  workspace.startup_cost,
                              workspace.total_cost,
                              NULL, required_outer))
		{
			final_cost_gpunestloop(root, gpath, &workspace);
			add_path(joinrel, &gpath->cpath.path);
		}
	}

	/*
	 * creation of more efficient path, if underlying outer-path is also
	 * GpuNestedLoop that shall be merginable
	 */
	if (path_is_mergeable_gpunestloop(outer_path))
	{
		GpuNestedLoopPath  *outer_gnl = (GpuNestedLoopPath *) outer_path;
		int		num_rels = outer_gnl->num_rels;

		if (support_bulkload)
		{
			if ((outer_ghj->cpath.flags & CUSTOMPATH_SUPPORT_BULKLOAD) == 0)
				support_bulkload = false;
		}
		Assert(num_rels > 0);
		gpath = palloc0(offsetof(GpuHashJoinPath, inners[num_rels + 1]));
		gpath->cpath.path.pathtype = T_CustomScan;
		gpath->cpath.path.parent = joinrel;
		gpath->cpath.path.param_info = param_info =
			get_joinrel_parampathinfo(root,
									  joinrel,
									  outer_path,
									  inner_path,
									  sjinfo,
									  bms_copy(required_outer),
									  &restrict_clauses);
		gpath->cpath.path.rows = (param_info
								  ? param_info->ppi_rows
								  : joinrel->rows);
		gpath->cpath.path.pathkeys = NIL;
		gpath->cpath.flags
			= (support_bulkload ? CUSTOMPATH_SUPPORT_BULKLOAD : 0);
		gpath->num_rels = num_rels + 1;
		gpath->outer_path = outer_gnl->outer_path;
		memcpy(gpath->inners,
			   outer_gnl->inners,
			   offsetof(GpuNestedLoopPath, inners[num_rels]) -
			   offsetof(GpuNestedLoopPath, inners[0]));
		gpath->inners[num_rels].scan_path = inner_path;
        gpath->inners[num_rels].jointype = jointype;
		gpath->inners[num_rels].join_clause = restrict_clauses;
		gpath->inners[num_rels].nloops = 1;

		/* cost estimation and check availability */
		if (initial_cost_gpunestloop(root, gpath, required_outer, &workspace))
		{
			if (add_path_precheck(joinrel,
								  workspace.startup_cost,
								  workspace.total_cost,
								  NULL, required_outer))
			{
				final_cost_gpunestloop(root, gpath, &workspace);
				add_path(joinrel, &gpath->cpath.path);
			}
		}
	}
	bms_free(required_outer);
}

static void
gpunestloop_add_join_path(PlannerInfo *root,
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

	/* Nothing to do, if either PG-Strom or GpuHashJoin is not enabled */
	if (!pgstrom_enabled() || !enable_gpuhashjoin)
		return;

	/* Is this join-type supported? */
	if (jointype != JOIN_INNER && jointype != JOIN_LEFT)
		return;

	/* All the join-clauses of GpuNestedLoop has to be executable on
	 * the device.
	 */
	foreach (lc, restrictlist)
	{
		RestrictInfo   *rinfo = (RestrictInfo *) lfirst(lc);
		/*
		 * TODO: It may be possible to implement if only last inner
		 * relation takes host-only clauses, as long as row_population_ratio
		 * is reasonable enough.
		 */
		if (!pgstrom_codegen_available_expression(rinfo->clause))
			return;
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

	/*
	 * Let's try to make GpuNestLoop paths
	 */
	if (cheapest_startup_outer)
	{
		try_gpunestloop_path(root,
							 joinrel,
							 jointype,
							 cheapest_startup_outer,
							 cheapest_total_inner,
							 restrictlist);
	}

	try_gpunestloop_path(root,
						 joinrel,
						 jointype,
						 cheapest_total_outer,
						 cheapest_total_inner,
						 restrictlist);
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
	gpunestloop_path_methods.CustomName				= "GpuNestedLoop";
	gpunestloop_path_methods.PlanCustomPath			= create_gpunestloop_plan;
	gpunestloop_path_methods.TextOutCustomPath		= gpunestloop_textout_path;

	/* setup plan methods */
	gpunestloop_plan_methods.CustomName				= "GpuNestedLoop";
	gpunestloop_plan_methods.CreateCustomScanState
		= gpunestedloop_create_scan_state;
	gpunestloop_plan_methods.TextOutCustomScan		= NULL;

	/* setup exec methods */
	gpunestloop_exec_methods.c.CustomName			= "GpuNestedLoop";
	gpunestloop_exec_methods.c.BeginCustomScan		= gpunestloop_begin;
	gpunestloop_exec_methods.c.ExecCustomScan		= gpunestloop_exec;
	gpunestloop_exec_methods.c.EndCustomScan		= gpunestloop_end;
	gpunestloop_exec_methods.c.ReScanCustomScan		= gpunestloop_rescan;
	gpunestloop_exec_methods.c.MarkPosCustomScan	= NULL;
	gpunestloop_exec_methods.c.RestrPosCustomScan	= NULL;
	gpunestloop_exec_methods.c.ExplainCustomScan	= gpunestloop_explain;
	gpunestloop_exec_methods.ExecCustomBulk			= gpunestloop_exec_bulk;

	/* hook registration */
	set_join_pathlist_next = set_join_pathlist_hook;
	set_join_pathlist_hook = gpunestloop_add_join_path;
}
