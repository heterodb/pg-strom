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
 * GpuNestLoopPath
 *
 * 
 *
 */
typedef struct
{
	CustomPath		cpath;
	Path		   *outer_path;
	Size			mrs_size;	/* size of multi-rel-store */
	double			row_population_ratio;
	List		   *host_quals;
	int				num_rels;
	struct {
		Path	   *scan_path;
		JoinType	join_type;
		List	   *hash_quals;
		List	   *join_quals;
		double		threshold;
		int			nloops;
	} inners[FLEXIBLE_ARRAY_MEMBER];
} GpuJoinPath;

/*
 * GpuNestLoopInfo - state object of CustomScan(GpuNestedLoop)
 */
typedef struct
{
	int			num_rels;
	Size		mrs_size;
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
 * MultiRelStoreInfo - state object of CustomScan(MultiRelStore)
 */
typedef struct
{

} MultiRelStoreInfo;

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
 * MultiRelStoreState - execution state object of MultiRelStore
 */
typedef struct
{
	CustomScanState	css;
	GpuContext	   *gcontext;
	int				depth;
	double			threshold;
} MultiRelStoreState;





/*
 * estimate_multi_relation_store_size
 *
 * It estimates size of inner multi relation store
 */
static bool
estimate_multi_relation_store_size(PlannerInfo *root,
								   GpuNestedLoopPath *gpath,
								   Relids required_outer,
								   Cost startup_cost,
								   Cost total_cost)
{
	Cost		startup_cost = gpath->cpath.path.startup_cost;
	Cost		total_cost = gpath->cpath.path.total_cost;
	Size		mrs_size;
	Size		largest_size;
	long		num_batches;
	long		num_batches_init = -1;
	int			i_largest;

retry:
	mrs_size = LONGALIGN(offsetof(kern_multi_relstore,
								  rels[gpath->num_rels]));
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
		mrs_size += table_size;
	}
	if (num_batches_init < 0)
		num_batches_init = num_batches;

	/* also, compute threshold of each chunk */
	threshold_size = 0;
	for (i = gpath->num_rels - 1; i >= 0; i--)
	{
		threshold_size += gpath->inners[i].table_size;
		gpath->inners[i].threshold
			= (double) threshold_size / (double) mrs_size;
	}

	/*
	 * NOTE: In case when extreme number of rows are expected,
	 * it does not make sense to split inner multi-tables because
	 * increasion of numbatches also increases the total cost by
	 * iteration of outer scan. In this case, the best strategy
	 *is to give up this path, instead of incredible number of
	 * numbatches!
	 */
	total_cost = startup_cost +
		(total_cost - startup_cost) * ((double) num_batches /
									   (double) num_batches_init);
	if (!add_path_precheck(gpath->cpath.path.parent,
						   startup_cost, total_cost,
						   NULL, required_outer))
		return false;

	/*
	 * If size of inner multi-tables is still larger than device
	 * allocatable limitation, we try to split the largest table
	 * then retry the estimation.
	 */
	if (mrs_size > gpuMemMaxAllocSize())
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
	 * Update cost estimation value
	 */
	gpath->cpath.path.startup_cost = startup_cost;
	gpath->cpath.path.total_cost = startup_cost + run_cost;

	return true;	/* ok */
}

/*
 * cost_gpunestloop
 *
 * estimation of GpuNestedLoop cost
 */
static bool
cost_gpunestloop(PlannerInfo *root,
				 GpuNestedLoopPath *gpath,
				 Relids required_outer)
{
	Path	   *outer_path;
	Path	   *inner_path;
	Cost		startup_cost;
	Cost		run_cost;
	QualCost	qual_cost;
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
	cost_qual_eval(&qual_cost, join_clause, root);
	qual_cost.per_tuple *= (pgstrom_gpu_operator_cost / cpu_operator_cost);

	run_cost = (outer_path->total_cost - outer_path->startup_cost +
				inner_path->total_cost - inner_path->startup_cost);
	run_cost += (pgstrom_gpu_operator_cost *
				 qual_cost.per_tuple *
				 outer_path->rows *
				 inner_path->rows);
	run_cost += cpu_tuple_cost * gpath->cpath.path.rows;
	if (num_rels > 1)
		run_cost -= cpu_tuple_cost * outer_path->rows;

	gpath->cpath.path.startup_cost = startup_cost;
	gpath->cpath.path.total_cost = startup_cost + run_cost;

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

	/* make a param-info */
	param_info = get_joinrel_parampathinfo(root,
										   joinrel,
										   outer_path,
										   inner_path,
										   sjinfo,
										   bms_copy(required_outer),
										   &restrict_clauses);
	/*
	 * Creation of GpuNestedLoop Path, without merging child GpuNestedLoop
	 */
	gpath = palloc0(offsetof(GpuNestedLoopPath, inners[1]));
	NodeSetTag(gpath, T_CustomPath);
	gpath->cpath.path.pathtype = T_CustomScan;
	gpath->cpath.path.parent = joinrel;
	gpath->cpath.path.param_info = param_info;
	gpath->cpath.path.rows = (param_info
							  ? param_info->ppi_rows
							  : joinrel->rows);
	gpath->cpath.path.pathkeys = NIL;
	gpath->cpath.flags = (support_bulkload ? CUSTOMPATH_SUPPORT_BULKLOAD : 0);
	gpath->cpath.methods = &gpujoin_path_methods;
	gpath->outer_path = outer_path;
	gpath->num_rels = 1;
	gpath->inners[0].scan_path = inner_path;
	gpath->inners[0].jointype = jointype;
	gpath->inners[0].join_clause = restrict_clauses;
	gpath->inners[0].nloops = 1;

	/* cost estimation and check availability */
	if (cost_gpunestloop(root, gpath, required_outer))
		add_path(joinrel, &gpath->cpath.path);

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
		gpath->cpath.path.param_info = param_info;
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
		if (cost_gpunestloop(root, gpath, required_outer))
			add_path(joinrel, &gpath->cpath.path);
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

static Plan *
create_gpunestloop_plan(PlannerInfo *root,
						RelOptInfo *rel,
						CustomPath *best_path,
						List *tlist,
						List *clauses)
{
	GpuNestLoopPath	   *gpath = (GpuNestLoopPath *) best_path;
	GpuNestLoopInfo		gnl_info;
	CustomScan		   *gnl_scan;

	gnl_scan = makeNode(CustomScan);
	gnl_scan->scan.plan.targetlist = tlist;
	gnl_scan->scan.plan.qual = NIL;
	gnl_scan->flags = best_path->flags;
	gnl_scan->methods = &gpujoin_plan_methods;

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
gpunestloop_textout_path(StringInfo str, const CustomPath *node)
{
	GpuNestLoopPath *gpath = (GpuNestLoopPath *) node;
	int			i;

	/* outer_path */
	appendStringInfo(str, " :outer_path %s",
					 nodeToString(gpath->outer_path));
	/* mrs_size */
	appendStringInfo(str, " :mrs_size %zu",
					 gpath->mrs_size);

	/* row_population_ratio */
	appendStringInfo(str, " :row_population_ratio %.2f",
					 gpath->row_population_ratio);
	/* num_rels */
	appendStringInfo(str, " :num_rels %d", gpath->num_rels);

	/* inners */
	appendStringInfo(str, " :inners (");
	for (i=0; i < gpath->num_rels; i++)
	{
		appendStringInfo(str, "{");
		/* scan_path */
		appendStringInfo(str, " :scan_path %s",
						 nodeToString(gpath->inners[i].scan_path));
		/* join_type */
		appendStringInfo(str, " :join_type %d",
						 (int)gpath->inners[i].join_type);
		/* join_clause */
		appendStringInfo(str, " :join_clause %s",
						 nodeToString(gpath->inners[i].join_clause));
		/* threshold */
		appendStringInfo(str, " :threshold %.2f",
						 gpath->inners[i].threshold);
		/* nloops */
		appendStringInfo(str, " :nslots %d",
						 gpath->inners[i].nloops);
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
	set_join_pathlist_hook = gpunestloop_add_join_path;
}
