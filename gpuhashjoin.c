/*
 * gpuhashjoin.c
 *
 * Hash-Join acceleration by GPU processors
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"

#include "nodes/nodeFuncs.h"
#include "nodes/relation.h"
#include "nodes/plannodes.h"
#include "optimizer/cost.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "utils/lsyscache.h"
#include "pg_strom.h"
#include "opencl_hashjoin.h"

/* static variables */
static add_hashjoin_path_hook_type	add_hashjoin_path_next;
static CustomPathMethods		gpuhashjoin_path_methods;
static CustomPlanMethods		gpuhashjoin_plan_methods;

typedef struct
{
	CustomPath		cpath;
	JoinType		jointype;
	Path		   *outerjoinpath;
	Path		   *innerjoinpath;
	List		   *gpu_clauses;
	List		   *cpu_clauses;
} GpuHashJoinPath;

typedef struct
{
	CustomPlan		cplan;
	JoinType		jointype;
	List		   *gpu_clauses;
	List		   *cpu_clauses;
} GpuHashJoin;

typedef struct
{
	CustomPlanState	cps;

} GpuHashJoinState;

/*
 * estimate_hashitem_size
 *
 * It estimates size of hashitem for GpuHashJoin
 */
typedef struct
{
	Relids	relids;	/* relids located on the inner loop */
	int		natts;	/* roughly estimated number of variables */
	int		width;	/* roughly estimated width of referenced variables */
} estimate_hashtable_size_context;

static bool
estimate_hashtable_size_walker(Node *node,
							   estimate_hashtable_size_context *context)
{
	if (node == NULL)
		return false;
	if (IsA(node, Var))
	{
		Var	   *var = (Var *) node;

		if (bms_is_member(var->varno, context->relids))
		{
			int16	typlen;
			bool	typbyval;
			char	typalign;

			/* number of attributes affects NULL bitmap */
			context->natts++;

			get_typlenbyvalalign(var->vartype, &typlen, &typbyval, &typalign);
			if (typlen > 0)
			{
				if (typalign == 'd')
					context->width = TYPEALIGN(sizeof(cl_ulong),
											   context->width) + typlen;
				else if (typalign == 'i')
					context->width = TYPEALIGN(sizeof(cl_uint),
											   context->width) + typlen;
				else if (typalign == 's')
					context->width = TYPEALIGN(sizeof(cl_ushort),
											   context->width) + typlen;
				else
				{
					Assert(typalign == 'c');
					context->width += typlen;
				}
			}
			else
			{
				context->width = TYPEALIGN(sizeof(cl_uint),
										   context->width) + sizeof(cl_uint);
				context->width += INTALIGN(get_typavgwidth(var->vartype,
														   var->vartypmod));
			}
		}
		return false;
	}
	else if (IsA(node, RestrictInfo))
	{
		RestrictInfo   *rinfo = (RestrictInfo *)node;
		Relids			relids_saved = context->relids;
		bool			rc;

		context->relids = rinfo->left_relids;
		rc = estimate_hashtable_size_walker((Node *)rinfo->clause, context);
		context->relids = relids_saved;

		return rc;
	}
	/* Should not find an unplanned subquery */
	Assert(!IsA(node, Query));
	return expression_tree_walker(node,
								  estimate_hashtable_size_walker,
								  context);
}

static Size
estimate_hashtable_size(PlannerInfo *root, List *gpu_clauses, double ntuples)
{
	estimate_hashtable_size_context context;
	Size		entry_size;

	/* Force a plausible relation size if no info */
	if (ntuples <= 0.0)
		ntuples = 1000.0;

	/* walks on the join clauses to ensure var's width */
	memset(&context, 0, sizeof(estimate_hashtable_size_context));
	estimate_hashtable_size_walker((Node *)gpu_clauses, &context);

	entry_size = INTALIGN(offsetof(kern_hash_entry, keydata[0]) +
						  Max(sizeof(cl_ushort),
							  SHORTALIGN(context.natts / BITS_PER_BYTE)) +
						  context.width);

	return MAXALIGN(offsetof(kern_hash_table,
							 colmeta[context.natts])
					+ entry_size * (Size)ntuples);
}

/*
 * cost_gpuhashjoin
 *
 * cost estimation for GpuHashJoin
 */
static bool
cost_gpuhashjoin(PlannerInfo *root,
				 JoinType jointype,
				 Path *outer_path,
				 Path *inner_path,
				 List *gpu_clauses,
				 List *cpu_clauses,
				 JoinCostWorkspace *workspace)
{
	Cost		startup_cost = 0.0;
	Cost		run_cost = 0.0;
	double		outer_path_rows = outer_path->rows;
	double		inner_path_rows = inner_path->rows;
	int			num_gpu_clauses = list_length(gpu_clauses);
	int			num_cpu_clauses = list_length(cpu_clauses);
	Size		hashtable_size;

	/* cost of source data */
	startup_cost += outer_path->startup_cost;
	run_cost += outer_path->total_cost - outer_path->startup_cost;
	startup_cost += inner_path->total_cost;

	/*
	 * Cost of computing hash function: it is done by CPU right now,
	 * so we follow the logic in initial_cost_hashjoin().
	 */
	startup_cost += (cpu_operator_cost * (num_gpu_clauses + num_cpu_clauses)
					 + cpu_tuple_cost) * inner_path_rows;

	/* in addition, it takes setting up cost for GPU/MIC devices  */
	startup_cost += pgstrom_gpu_setup_cost;

	/*
	 * However, its cost to run outer scan for joinning is much less
	 * than usual CPU join.
	 */
	run_cost += ((pgstrom_gpu_operator_cost * num_gpu_clauses) +
				 (cpu_operator_cost * num_cpu_clauses)) * outer_path_rows;

	/*
	 * TODO: we need to pay attention towards joinkey length to copy
	 * data from host to device, to prevent massive amount of DMA
	 * request for wider keys, like text comparison.
	 */

	/*
	 * Estimation of hash table size - we want to keep it less than
	 * device restricted memory allocation size.
	 */
	hashtable_size = estimate_hashtable_size(root, gpu_clauses,
											 inner_path_rows);
	/*
	 * For safety, half of shmem zone size is considered as a hard
	 * restriction. If table size would be actually bigger, right
	 * now, we simply give it up.
	 */
	if (hashtable_size > pgstrom_shmem_zone_length() / 2)
		return false;
	/*
	 * FIXME: Right now, we pay attention on the memory consumption of
	 * kernel hash-table only, because host system mounts much larger
	 * amount of memory than GPU/MIC device. Of course, work_mem
	 * configuration should be considered, but not now.
	 */
	workspace->startup_cost = startup_cost;
	workspace->run_cost = run_cost;
	workspace->total_cost = startup_cost + run_cost;

	elog(INFO, "startup_cost = %f, total_cost = %f, hashtable_size = %zu",
		 startup_cost, run_cost, hashtable_size);

	return true;
}
#if 0
static void
final_cost_gpuhashjoin(PlannerInfo *root, GpuHashJoinPath *gpath)
{


}
#endif
static CustomPath *
gpuhashjoin_create_path(PlannerInfo *root,
						RelOptInfo *joinrel,
						JoinType jointype,
						SpecialJoinInfo *sjinfo,
						Path *outer_path,
						Path *inner_path,
						Relids required_outer,
						List *gpu_clauses,
						List *cpu_clauses,
						JoinCostWorkspace *workspace)
{
	GpuHashJoinPath	   *gpath = palloc0(sizeof(GpuHashJoinPath));

	NodeSetTag(gpath, T_CustomPath);
	gpath->cpath.methods = &gpuhashjoin_path_methods;
	gpath->cpath.path.parent = joinrel;
	gpath->cpath.path.param_info =
		get_joinrel_parampathinfo(root,
								  joinrel,
								  outer_path,
								  inner_path,
								  sjinfo,
								  required_outer,
								  &cpu_clauses);
	gpath->cpath.path.pathkeys = NIL;
	gpath->jointype = jointype;
	gpath->outerjoinpath = outer_path;
	gpath->innerjoinpath = inner_path;
	gpath->gpu_clauses = gpu_clauses;
	gpath->cpu_clauses = cpu_clauses;

    //final_cost_hashjoin(root, gpath, workspace, sjinfo, semifactors);
	
	return &gpath->cpath;
}

static void
gpuhashjoin_add_path(PlannerInfo *root,
					 RelOptInfo *joinrel,
					 JoinType jointype,
					 JoinCostWorkspace *core_workspace,
					 SpecialJoinInfo *sjinfo,
					 SemiAntiJoinFactors *semifactors,
					 Path *outer_path,
					 Path *inner_path,
					 List *restrict_clauses,
					 Relids required_outer,
					 List *hashclauses)
{
	//RelOptInfo	   *outer_rel = outer_path->parent;
	//RelOptInfo	   *inner_rel = inner_path->parent;
	List		   *gpu_clauses = NIL;
	List		   *cpu_clauses = NIL;
	ListCell	   *cell;
	JoinCostWorkspace gpu_workspace;

	/* calls secondary module if exists */
	if (add_hashjoin_path_next)
		add_hashjoin_path_next(root,
							   joinrel,
							   jointype,
							   core_workspace,
							   sjinfo,
							   semifactors,
							   outer_path,
							   inner_path,
							   restrict_clauses,
							   required_outer,
							   hashclauses);
	/*
	 * right now, only inner join is supported!
	 */
	if (jointype != JOIN_INNER)
		return;

	/* reasonable portion of hash-clauses can be runnable on GPU */
	foreach (cell, hashclauses)
	{
		RestrictInfo   *rinfo = lfirst(cell);

		if (pgstrom_codegen_available_expression(rinfo->clause))
			gpu_clauses = lappend(gpu_clauses, rinfo);
		else
			cpu_clauses = lappend(cpu_clauses, rinfo);
	}
	if (gpu_clauses == NIL)
		return;	/* no need to run it on GPU */

	/* cost estimation by gpuhashjoin */
	if (!cost_gpuhashjoin(root, jointype,
						  outer_path, inner_path,
						  gpu_clauses, cpu_clauses,
						  &gpu_workspace))
		return;	/* obviously unavailable to run it on GPU */

	if (add_path_precheck(joinrel,
						  gpu_workspace.startup_cost,
						  gpu_workspace.total_cost,
						  NULL, required_outer))
	{
		CustomPath *pathnode = gpuhashjoin_create_path(root,
													   joinrel,
													   jointype,
													   sjinfo,
													   outer_path,
													   inner_path,
													   required_outer,
													   gpu_clauses,
													   cpu_clauses,
													   &gpu_workspace);
		if (pathnode)
			add_path(joinrel, &pathnode->path);
	}
}

static CustomPlan *
gpuhashjoin_create_plan(PlannerInfo *root, CustomPath *best_path)
{
	return NULL;
}

static void
gpuhashjoin_textout_path(StringInfo str, Node *node)
{}

static void
gpuhashjoin_set_plan_ref(PlannerInfo *root,
						 CustomPlan *custom_plan,
						 int rtoffset)
{}

static void
gpuhashjoin_finalize_plan(PlannerInfo *root,
						  CustomPlan *custom_plan,
						  Bitmapset **paramids,
						  Bitmapset **valid_params,
						  Bitmapset **scan_params)
{}


/*
 * gpuhashjoin_support_multi_exec
 *
 * It gives a hint whether the supplied plan-state support bulk-exec mode,
 * or not. If it is GpuHashJooin provided by PG-Strom, it does not allow
 * bulk- exec mode right now.
 */
bool
gpuhashjoin_support_multi_exec(const CustomPlanState *cps)
{
    return false;
}


static CustomPlanState *
gpuhashjoin_begin(CustomPlan *custom_plan, EState *estate, int eflags)
{
	return NULL;
}

static TupleTableSlot *
gpuhashjoin_exec(CustomPlanState *node)
{
	return NULL;
}

static Node *
gpuhashjoin_exec_multi(CustomPlanState *node)
{
	elog(ERROR, "not implemented yet");
	return NULL;
}

static void
gpuhashjoin_end(CustomPlanState *node)
{}

static void
gpuhashjoin_rescan(CustomPlanState *node)
{}

static void
gpuhashjoin_explain_rel(CustomPlanState *node, ExplainState *es)
{}

static void
gpuhashjoin_explain(CustomPlanState *node, List *ancestors, ExplainState *es)
{}

static Bitmapset *
gpuhashjoin_get_relids(CustomPlanState *node)
{
	return NULL;
}

static Node *
gpuhashjoin_get_special_var(CustomPlanState *node, Var *varnode)
{
	return NULL;
}

static void
gpuhashjoin_textout_plan(StringInfo str, const CustomPlan *node)
{}

static CustomPlan *
gpuhashjoin_copy_plan(const CustomPlan *from)
{
	GpuHashJoin		   *newnode = palloc(sizeof(GpuHashJoin));

	CopyCustomPlanCommon((Node *)from, (Node *)newnode);

	return &newnode->cplan;
}

void
pgstrom_init_gpuhashjoin(void)
{
	/* setup path methods */
	gpuhashjoin_path_methods.CustomName = "GpuHashJoin";
	gpuhashjoin_path_methods.CreateCustomPlan	= gpuhashjoin_create_plan;
	gpuhashjoin_path_methods.TextOutCustomPath	= gpuhashjoin_textout_path;

	/* setup plan methods */
	gpuhashjoin_plan_methods.CustomName = "GpuHashJoin";
	gpuhashjoin_plan_methods.SetCustomPlanRef	= gpuhashjoin_set_plan_ref;
	gpuhashjoin_plan_methods.SupportBackwardScan= NULL;
	gpuhashjoin_plan_methods.FinalizeCustomPlan	= gpuhashjoin_finalize_plan;
	gpuhashjoin_plan_methods.BeginCustomPlan	= gpuhashjoin_begin;
	gpuhashjoin_plan_methods.ExecCustomPlan		= gpuhashjoin_exec;
	gpuhashjoin_plan_methods.MultiExecCustomPlan= gpuhashjoin_exec_multi;
	gpuhashjoin_plan_methods.EndCustomPlan		= gpuhashjoin_end;
	gpuhashjoin_plan_methods.ReScanCustomPlan	= gpuhashjoin_rescan;
	gpuhashjoin_plan_methods.ExplainCustomPlanTargetRel
		= gpuhashjoin_explain_rel;
	gpuhashjoin_plan_methods.ExplainCustomPlan	= gpuhashjoin_explain;
	gpuhashjoin_plan_methods.GetRelidsCustomPlan= gpuhashjoin_get_relids;
	gpuhashjoin_plan_methods.GetSpecialCustomVar= gpuhashjoin_get_special_var;
	gpuhashjoin_plan_methods.TextOutCustomPlan	= gpuhashjoin_textout_plan;
	gpuhashjoin_plan_methods.CopyCustomPlan		= gpuhashjoin_copy_plan;

	/* hook registration */
	add_hashjoin_path_next = add_hashjoin_path_hook;
	add_hashjoin_path_hook = gpuhashjoin_add_path;
}
