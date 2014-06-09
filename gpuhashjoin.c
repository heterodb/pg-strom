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

#include "catalog/pg_type.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "nodes/relation.h"
#include "nodes/plannodes.h"
#include "optimizer/cost.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/planmain.h"
#include "optimizer/restrictinfo.h"
#include "optimizer/tlist.h"
#include "optimizer/var.h"
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
	List		   *hash_clauses;
	List		   *qual_clauses;
	List		   *host_clauses;
} GpuHashJoinPath;

typedef struct
{
	int			num_vars;		/* current number of entries */
	int			max_vars;		/* max number of entries */
	Index		special_varno;	/* either of INDEX/INNER/OUTER_VAR */
	bool		has_ph_vars;	/* has any PlaceHolderVar? */
	bool		has_non_vars;	/* has any non Var expression? */
	List	   *tlist;			/* original tlist */
	struct {
		Index		varno;		/* RT index of Var */
		AttrNumber	varattno;	/* attribute number of Var */
		AttrNumber	resno;		/* TLE position of Var */
	} trans[FLEXIBLE_ARRAY_MEMBER];
} vartrans_info;

typedef struct
{
	CustomPlan	cplan;
	JoinType	jointype;
	const char *kernel_source;
	int			extra_flags;
	vartrans_info  *result_map;	/* vartrans of result slot */
	vartrans_info  *inner_map;	/* vartrans of device inner slot */
	vartrans_info  *outer_map;	/* vartrans of device outer slot */

	List	   *used_params;	/* template for kparams */
	List	   *used_vars;		/* var used in hash/qual clauses */
	List	   *hash_clauses;	/* expression form of hash_clauses */
	List	   *qual_clauses;	/* expression form of qual_clauses */
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
estimate_hashtable_size(PlannerInfo *root, List *hash_clauses, double ntuples)
{
	estimate_hashtable_size_context context;
	Size		entry_size;

	/* Force a plausible relation size if no info */
	if (ntuples <= 0.0)
		ntuples = 1000.0;

	/* walks on the join clauses to ensure var's width */
	memset(&context, 0, sizeof(estimate_hashtable_size_context));
	estimate_hashtable_size_walker((Node *)hash_clauses, &context);

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
				 List *hash_clauses,
				 List *qual_clauses,
				 List *host_clauses,
				 JoinCostWorkspace *workspace)
{
	Cost		startup_cost = 0.0;
	Cost		run_cost = 0.0;
	double		outer_path_rows = outer_path->rows;
	double		inner_path_rows = inner_path->rows;
	int			num_gpu_clauses;
	int			num_cpu_clauses;
	Size		hashtable_size;

	/* cost of source data */
	startup_cost += outer_path->startup_cost;
	run_cost += outer_path->total_cost - outer_path->startup_cost;
	startup_cost += inner_path->total_cost;

	/*
	 * Cost of computing hash function: it is done by CPU right now,
	 * so we follow the logic in initial_cost_hashjoin().
	 */
	num_gpu_clauses = list_length(hash_clauses) + list_length(qual_clauses);
	num_cpu_clauses = list_length(host_clauses);
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
	hashtable_size = estimate_hashtable_size(root, hash_clauses,
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

	return true;
}

/*
 * approx_tuple_count - copied from costsize.c but arguments are adjusted
 * according to GpuHashJoinPath.
 */
static double
approx_tuple_count(PlannerInfo *root, Path *outer_path, Path *inner_path,
				   List *hash_clauses, List *qual_clauses)
{
	double		tuples;
	double		outer_tuples = outer_path->rows;
	double		inner_tuples = inner_path->rows;
	SpecialJoinInfo sjinfo;
	Selectivity selec = 1.0;
	ListCell	*l;

	/*
	 * Make up a SpecialJoinInfo for JOIN_INNER semantics.
	 */
	sjinfo.type = T_SpecialJoinInfo;
	sjinfo.min_lefthand  = outer_path->parent->relids;
	sjinfo.min_righthand = inner_path->parent->relids;
	sjinfo.syn_lefthand  = outer_path->parent->relids;
	sjinfo.syn_righthand = inner_path->parent->relids;
	sjinfo.jointype = JOIN_INNER;
	/* we don't bother trying to make the remaining fields valid */
	sjinfo.lhs_strict = false;
	sjinfo.delay_upper_joins = false;
	sjinfo.join_quals = NIL;

	/* Get the approximate selectivity */
	foreach (l, hash_clauses)
	{
		Node	   *qual = (Node *) lfirst(l);

		/* Note that clause_selectivity will be able to cache its result */
		selec *= clause_selectivity(root, qual, 0, JOIN_INNER, &sjinfo);
	}

	foreach (l, qual_clauses)
	{
		Node	   *qual = (Node *) lfirst(l);

		/* Note that clause_selectivity will be able to cache its result */
		selec *= clause_selectivity(root, qual, 0, JOIN_INNER, &sjinfo);
	}
	/* Apply it to the input relation sizes */
	tuples = selec * outer_tuples * inner_tuples;

	return clamp_row_est(tuples);
}



static void
final_cost_gpuhashjoin(PlannerInfo *root, GpuHashJoinPath *gpath,
					   JoinCostWorkspace *workspace)
{
	Cost		startup_cost = workspace->startup_cost;
	Cost		run_cost = workspace->run_cost;
	List	   *hash_clauses = gpath->hash_clauses;
	List	   *qual_clauses = gpath->qual_clauses;
	List	   *host_clauses = gpath->host_clauses;
	double		outer_path_rows = gpath->outerjoinpath->rows;
	double		inner_path_rows = gpath->innerjoinpath->rows;
	QualCost	hash_cost;
	QualCost	qual_cost;
	QualCost	host_cost;
	double		hashjointuples;

	/* Mark the path with the correct row estimate */
	if (gpath->cpath.path.param_info)
        gpath->cpath.path.rows = gpath->cpath.path.param_info->ppi_rows;
    else
        gpath->cpath.path.rows = gpath->cpath.path.parent->rows;

	/* add disable_cost, if hash_join is not prefered */
	if (!enable_hashjoin)
		startup_cost += disable_cost;

	/*
	 * Compute cost of the hash, qual and host clauses separately.
	 */
	cost_qual_eval(&hash_cost, hash_clauses, root);
	cost_qual_eval(&qual_cost, qual_clauses, root);
	cost_qual_eval(&host_cost, host_clauses, root);

	/* adjust cost according to GPU/CPU ratio */
	hash_cost.per_tuple *= (pgstrom_gpu_operator_cost / cpu_operator_cost);
	qual_cost.per_tuple *= (pgstrom_gpu_operator_cost / cpu_operator_cost);

	/*
	 * The number of comparison according to hash_clauses and qual_clauses
	 * are the number of outer tuples, but right now PG-Strom does not
	 * support to divide hash table
	 */
	startup_cost += hash_cost.startup + qual_cost.startup;
	run_cost += ((hash_cost.per_tuple + qual_cost.per_tuple)
				 * outer_path_rows
				 * clamp_row_est(inner_path_rows) * 0.5);
	/*
	 * Get approx # tuples passing the hashquals.  We use
	 * approx_tuple_count here because we need an estimate done with
	 * JOIN_INNER semantics.
	 */
	hashjointuples = approx_tuple_count(root,
										gpath->outerjoinpath,
										gpath->innerjoinpath,
										hash_clauses,
										qual_clauses);
	/*
	 * Also add cost for qualifiers to be run on host
	 */
	startup_cost += host_cost.startup;
	run_cost += (cpu_tuple_cost + host_cost.per_tuple) * hashjointuples;

	gpath->cpath.path.startup_cost = startup_cost;
	gpath->cpath.path.total_cost = startup_cost + run_cost;
}

static CustomPath *
gpuhashjoin_create_path(PlannerInfo *root,
						RelOptInfo *joinrel,
						JoinType jointype,
						SpecialJoinInfo *sjinfo,
						Path *outer_path,
						Path *inner_path,
						Relids required_outer,
						List *hash_clauses,
						List *qual_clauses,
						List *host_clauses,
						JoinCostWorkspace *workspace)
{
	GpuHashJoinPath	   *gpath = palloc0(sizeof(GpuHashJoinPath));

	NodeSetTag(gpath, T_CustomPath);
	gpath->cpath.methods = &gpuhashjoin_path_methods;
	gpath->cpath.path.pathtype = T_CustomPlan;
	gpath->cpath.path.parent = joinrel;
	gpath->cpath.path.param_info =
		get_joinrel_parampathinfo(root,
								  joinrel,
								  outer_path,
								  inner_path,
								  sjinfo,
								  required_outer,
								  &host_clauses);
	gpath->cpath.path.pathkeys = NIL;
	gpath->jointype = jointype;
	gpath->outerjoinpath = outer_path;
	gpath->innerjoinpath = inner_path;
	gpath->hash_clauses = hash_clauses;
	gpath->qual_clauses = qual_clauses;
	gpath->host_clauses = host_clauses;

	final_cost_gpuhashjoin(root, gpath, workspace);

	elog(INFO, "cost {startup: %f, total: %f} inner-rows: %f, outer-rows: %f",
		 gpath->cpath.path.startup_cost,
		 gpath->cpath.path.total_cost,
		 inner_path->rows,
		 outer_path->rows);
	//elog(INFO, "hash_clauses = %s", nodeToString(hash_clauses));
	//elog(INFO, "qual_clauses = %s", nodeToString(qual_clauses));
	//elog(INFO, "host_clauses = %s", nodeToString(host_clauses));

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
	List	   *hash_clauses = NIL;
	List	   *qual_clauses = NIL;
	List	   *host_clauses = NIL;
	ListCell   *cell;
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
	/* nothing to do, if PG-Strom is not enabled */
	if (!pgstrom_enabled)
		return;

	/*
	 * right now, only inner join is supported!
	 */
	if (jointype != JOIN_INNER)
		return;

	/* reasonable portion of hash-clauses can be runnable on GPU */
	foreach (cell, restrict_clauses)
	{
		RestrictInfo   *rinfo = lfirst(cell);

		if (pgstrom_codegen_available_expression(rinfo->clause))
		{
			if (list_member_ptr(hashclauses, rinfo))
				hash_clauses = lappend(hash_clauses, rinfo);
			else
				qual_clauses = lappend(qual_clauses, rinfo);
		}
		else
			host_clauses = lappend(host_clauses, rinfo);
	}
	if (hash_clauses == NIL)
		return;	/* no need to run it on GPU */

	/* cost estimation by gpuhashjoin */
	if (!cost_gpuhashjoin(root, jointype, outer_path, inner_path,
						  hash_clauses, qual_clauses, host_clauses,
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
													   hash_clauses,
													   qual_clauses,
													   host_clauses,
													   &gpu_workspace);
		if (pathnode)
			add_path(joinrel, &pathnode->path);
	}
}

#if 0
/*
 * pull_joinrel_resnums
 *
 *
 *
 *
 *
 *
 *
 */
typedef struct
{
	Bitmapset  *inner_resnums;
	Bitmapset  *outer_resnums;
	Relids		inner_relids;
	Relids		outer_relids;
} pull_join_resnums_context;

static bool
pull_joinrel_resnums_walker(Node *node, pull_join_resnums_context *context)
{
	if (!node)
		return false;
	if (IsA(node, Var))
	{
		Var	   *var = (Var *) node;

		if (bms_is_member(var->varno, context->inner_relids))
		{
			//elog(INFO, "inner var: %s", nodeToString(var));
			
			
		}
		else if (bms_is_member(var->varno, context->outer_relids))
		{
			//elog(INFO, "outer var: %s", nodeToString(var));
		}
		else
			elog(ERROR, "varnode referenced neither inner nor outer relation");



	}
	/* Should not find an unplanned subquery */
	Assert(!IsA(node, Query));
	return expression_tree_walker(node, pull_joinrel_resnums_walker,
								  (void *) context);
}

static List *
pull_joinrel_resnums(Node *node, Path *inner_path, Path *outer_path)
{
	pull_join_resnums_context	context;

	memset(&context, 0, sizeof(pull_join_resnums_context));
	context.inner_relids = inner_path->parent->relids;
	context.outer_relids = outer_path->parent->relids;

	pull_joinrel_resnums_walker(node, &context);

	return NIL;
}
#endif

static CustomPlan *
gpuhashjoin_create_plan(PlannerInfo *root, CustomPath *best_path)
{
	GpuHashJoinPath *gpath = (GpuHashJoinPath *)best_path;
	GpuHashJoin	*ghj;
	List	   *tlist = build_path_tlist(root, &best_path->path);
	Path	   *inner_path = gpath->innerjoinpath;
	Path	   *outer_path = gpath->outerjoinpath;
	Plan	   *inner_plan = create_plan_recurse(root, inner_path);
	Plan	   *outer_plan = create_plan_recurse(root, outer_path);
	List	   *hash_clauses;
	List	   *qual_clauses;
	List	   *host_clauses;

	/*
	 * Sort clauses into best execution order, even though it's
	 * uncertain whether it makes sense in GPU execution...
	 */
	hash_clauses = order_qual_clauses(root, gpath->hash_clauses);
	qual_clauses = order_qual_clauses(root, gpath->qual_clauses);
	host_clauses = order_qual_clauses(root, gpath->host_clauses);

	/*
	 * Get plan expression form
	 */
	hash_clauses = extract_actual_clauses(hash_clauses, false);
	qual_clauses = extract_actual_clauses(qual_clauses, false);
	host_clauses = extract_actual_clauses(host_clauses, false);

	/*
	 * Replace any outer-relation variables with nestloop params.
	 * There should not be any in the hash_ or qual_clauses
	 */
	if (best_path->path.param_info)
	{
		host_clauses = (List *)
			replace_nestloop_params(root, (Node *)host_clauses);
	}

	/*
	 * Create a GpuHashJoin node; inherited from CustomPlan
	 */
	ghj = palloc0(sizeof(GpuHashJoin));
	NodeSetTag(ghj, T_CustomPlan);
	ghj->cplan.methods = &gpuhashjoin_plan_methods;
	ghj->cplan.plan.targetlist = tlist;
	ghj->cplan.plan.qual = host_clauses;
	outerPlan(ghj) = outer_plan;
	innerPlan(ghj) = inner_plan;
	ghj->jointype = gpath->jointype;
	ghj->hash_clauses = hash_clauses;
	ghj->qual_clauses = qual_clauses;
	/* rest of fields are set later */

	return &ghj->cplan;
}

static void
gpuhashjoin_textout_path(StringInfo str, Node *node)
{
	GpuHashJoinPath *pathnode = (GpuHashJoinPath *) node;
	char	   *temp;

	/* jointype */
	appendStringInfo(str, " :jointype %d", (int)pathnode->jointype);

	/* outerjoinpath */
	temp = nodeToString(pathnode->outerjoinpath);
	appendStringInfo(str, " :outerjoinpath %s", temp);

	/* innerjoinpath */
	temp = nodeToString(pathnode->innerjoinpath);
	appendStringInfo(str, " :innerjoinpath %s", temp);

	/* hash_clauses */
	temp = nodeToString(pathnode->hash_clauses);
	appendStringInfo(str, " :hash_clauses %s", temp);

	/* qual_clauses */
	temp = nodeToString(pathnode->qual_clauses);
	appendStringInfo(str, " :qual_clauses %s", temp);

	/* host_clauses */
	temp = nodeToString(pathnode->host_clauses);
	appendStringInfo(str, " :host_clauses %s", temp);
}

/*
 * pull_one_varno
 *
 * It returns one varno of Var-nodes in the supplied clause.
 * If multiple varno is exist, it shall raise an error.
 */
static bool
pull_one_varno_walker(Node *node, Index *curr_varno)
{
	if (!node)
		return false;
	if (IsA(node, Var))
	{
		Var	   *var = (Var *) node;

		if (*curr_varno == 0)
			*curr_varno = var->varno;
		else if (*curr_varno != var->varno)
			elog(ERROR, "multiple varno appeared in %s", __FUNCTION__);
		return false;
	}
	/* Should not find an unplanned subquery */
	Assert(!IsA(node, Query));
	return expression_tree_walker(node,
								  pull_one_varno_walker,
								  (void *)curr_varno);
}

static Index
pull_one_varno(Node *node)
{
	Index	curr_varno = 0;

	(void) pull_one_varno_walker(node, &curr_varno);

	if (curr_varno == 0)
		elog(ERROR, "Bug? no valid varno not found");

	return curr_varno;
}

/*
 * gpuhashjoin_codegen_hashkey
 *
 * code generator of gpuhashjoin_get_hash() - that computes a hash value
 * according to the hash-clause on outer relation
 */
static void
gpuhashjoin_codegen_hashkey(PlannerInfo *root,
							StringInfo str,
							List *hash_clauses,
							codegen_context *context)
{
	StringInfoData	calc;
	devtype_info   *dtype;
	ListCell	   *cell;
	int				key_index;
	int				var_index;
	int				outer_index;

	initStringInfo(&calc);

	/*
	 * Generate a function to calculate hash value
	 */
	appendStringInfo(str,
					 "static cl_uint\n"
					 "gpuhashjoin_hashkey(__private cl_int *errcode,\n"
					 "                    __global kern_parambuf *kparam,\n"
					 "                    __global kern_column_store *kcs,\n"
					 "                    __global kern_toastbuf *ktoast,\n"
					 "                    size_t row_index)\n"
					 "{\n");
	/*
	 * note that context->used_vars are already constructed
	 * on the preliminary call of gpuhashjoin_codegen_compare()
	 */
	var_index = 0;
	outer_index = 0;
	foreach (cell, context->used_vars)
	{
		Var	   *var = lfirst(cell);

		if (var->varno == OUTER_VAR)
		{
			dtype = pgstrom_devtype_lookup(var->vartype);
			if (!dtype)
				elog(ERROR, "cache lookup failed for type %u", var->vartype);
			appendStringInfo(str,
							 "  pg_%s_t KVAR_%u = pg_%s_vref(kcs%s, errcode, "
							 "%u, row_index);\n",
							 dtype->type_name,
							 var_index,
							 dtype->type_name,
							 dtype->type_length > 0 ? "" : ", toast",
							 outer_index);
			outer_index++;
		}
		var_index++;
	}
	appendStringInfo(str, "\n");

	key_index = 0;
	foreach (cell, hash_clauses)
	{
		OpExpr	   *oper = lfirst(cell);
		Expr	   *lefthand;
		char		varname[80];

		if (!IsA(oper, OpExpr) || list_length(oper->args) != 2)
			elog(ERROR, "Binary OpExpr is expected in hash_clauses: %s",
				 nodeToString(oper));

		if (pull_one_varno(linitial(oper->args)) == OUTER_VAR)
			lefthand = linitial(oper->args);
		else if (pull_one_varno(lsecond(oper->args)) == INNER_VAR)
			lefthand = lsecond(oper->args);
		else
			elog(ERROR, "neither left- nor right-hand is part of outer plan");

		if (IsA(lefthand, Var))
		{
			Var		   *var = (Var *)lefthand;
			ListCell   *lp;

			dtype = pgstrom_devtype_lookup(var->vartype);
			if (!dtype)
				elog(ERROR, "cache lookup failed for type: %u", var->vartype);

			var_index = 0;
			foreach (lp, context->used_vars)
			{
				if (equal(var, lfirst(lp)))
				{
					snprintf(varname, sizeof(varname), "KVAR_%u", var_index);
					break;
				}
				var_index++;
			}
			if (!lp)
				elog(ERROR, "bug? reference Var not in used_vars");
		}
		else
		{
			Oid		type_oid = exprType((Node *)lefthand);
			char   *temp;

			dtype = pgstrom_devtype_lookup(type_oid);
			if (!dtype)
				elog(ERROR, "cache lookup failed for type %u", type_oid);

			temp = pgstrom_codegen_expression((Node *)lefthand, context);
			appendStringInfo(str, "  pg_%s_t keyval_%u = %s;\n",
							 dtype->type_name, key_index, temp);
		}

		if (dtype->type_length > 0)
			appendStringInfo(&calc,
							 "  if (!%s.isnull)\n"
							 "    COMP_CRC32(hash, &%s.value,\n"
							 "               sizeof(%s.value);\n",
							 varname, varname, varname);
		else
			appendStringInfo(&calc,
							 "  if (!%s.isnull)\n"
							 "    COMP_CRC32(hash,\n"
							 "               VARDATA_ANY(%s.value),\n"
							 "               VARSIZE_ANY(%s.value));\n",
							 varname, varname, varname);
		key_index++;
	}
	appendStringInfo(str,
					 "  cl_uint hash;\n"
					 "\n"
					 "  INIT_CRC32(hash);\n"
					 "%s"
					 "  FIN_CRC32(hash);\n"
					 "\n"
					 "  return hash;\n"
					 "}\n", calc.data);
}

/*
 * gpuhashjoin_codegen_compare
 *
 * code generator of gpuhashjoin_compare(); that compares a particular
 * row in kern_column_store with the supplied hash_entry.
 */
static void
gpuhashjoin_codegen_compare(PlannerInfo *root,
							StringInfo str,
							List *hash_clauses,
							List *qual_clauses,
							codegen_context *context)
{
	StringInfoData	tmpl;
	StringInfoData	decl;
	devtype_info   *dtype;
	char		   *eval_formula;
	List		   *clauses;
	ListCell	   *cell;
	cl_uint			var_index = 0;
	cl_uint			outer_index = 0;
	cl_uint			inner_index = 0;
	cl_uint			inner_nums = 0;
	cl_uint			inner_offset;

	initStringInfo(&tmpl);
	initStringInfo(&decl);

	clauses = list_concat(list_copy(hash_clauses),
						  list_copy(qual_clauses));
	eval_formula = pgstrom_codegen_expression((Node *)clauses, context);

	foreach (cell, context->used_vars)
	{
		if (((Var *)lfirst(cell))->varno == INNER_VAR)
			inner_nums++;
	}
	inner_offset = INTALIGN(offsetof(kern_hash_entry,
									 keydata[inner_nums >> 3]));
	foreach (cell, context->used_vars)
	{
		Var	   *var = lfirst(cell);

		if (var->varno == INNER_VAR)
		{
			dtype = pgstrom_devtype_lookup(var->vartype);
			if (!dtype)
				elog(ERROR, "cache lookup failed for type: %u", var->vartype);

			if (dtype->type_length > 0)
				appendStringInfo(&tmpl,
								 "STROMCL_SIMPLE_HASHREF_TEMPLATE(%s,%s)\n",
								 dtype->type_name, dtype->type_base);
			else
				appendStringInfo(&tmpl,
								 "STROMCL_VARLENA_HASHREF_TEMPLATE(%s)\n",
								 dtype->type_name);
			appendStringInfo(&decl,
			"  pg_%s_t KVAR_%u = pg_%s_hashref(entry, errcode, %u, %u);\n",
							 dtype->type_name, var_index, dtype->type_name,
							 inner_index, inner_offset);
		}
		else if (var->varno == OUTER_VAR)
		{
			dtype = pgstrom_devtype_lookup(var->vartype);
			if (!dtype)
				elog(ERROR, "cache lookup failed for type: %u", var->vartype);

			appendStringInfo(&decl,
			"  pg_%s_t KVAR_%u = pg_%s_vref(kcs%s,errcode,%u,row_index);\n",
							 dtype->type_name,
							 var_index,
							 dtype->type_name,
							 dtype->type_length > 0 ? "" : ", toast",
							 outer_index);
			outer_index++;
		}
		else
			elog(ERROR, "Bug? var-node in neither inner nor outer relations");
		var_index++;
	}

	appendStringInfo(str,
					 "%s\n"
					 "static cl_bool\n"
					 "gpuhashjoin_keycomp(__private cl_int *errcode,\n"
					 "                    __global kern_parambuf *kparams,\n"
					 "                    __global kern_hash_entry *entry,\n"
					 "                    __global kern_column_store *kcs,\n"
					 "                    __global kern_toastbuf *ktoast,\n"
					 "                    size_t row_index,\n"
					 "                    cl_uint hash)\n"
					 "{\n"
					 "%s"
					 "  pg_bool_t rc;\n"
					 "\n"
					 "  if (entry->hash != hash)\n"
					 "    return false;\n"
					 "\n"
					 "  rc = %s;\n"
					 "\n"
					 "  return (!rc.isnull && rc_value ? true : false);\n"
					 "}\n",
					 tmpl.data,
					 decl.data,
					 eval_formula);
}

static char *
gpuhashjoin_codegen(PlannerInfo *root,
					List *hash_clauses,
					List *qual_clauses,
					codegen_context *context)
{
	StringInfoData	str;

	memset(context, 0, sizeof(codegen_context));
	initStringInfo(&str);

	/*
	 * A dummy constant
	 * KPARAM_0 is an array of bool to inform referenced columns
	 * in the outer relation, in GpuHashJoin.
	 * Just a placeholder here. Set up it later.
	 */
	context->used_params = list_make1(makeConst(BYTEAOID,
												-1,
												InvalidOid,
												-1,
												PointerGetDatum(NULL),
												true,
												false));
	context->type_defs = list_make1(pgstrom_devtype_lookup(BYTEAOID));

	/*
	 * definition of gpuhashjoin_keycomp()
	 */
	gpuhashjoin_codegen_compare(root, &str,
								hash_clauses,
								qual_clauses,
								context);
	/*
	 * definition of gpuhashjoin_hashkey()
	 */
	gpuhashjoin_codegen_hashkey(root, &str,
								hash_clauses,
								context);
	/*
	 * to include opencl_hashjoin.h
	 */
	context->extra_flags |= DEVKERNEL_NEEDS_HASHJOIN;

	return str.data;
}

/*
 * build_gpuhashjoin_vartrans
 *
 * It builds translation map of varno/varattno from inner/outer relations
 * to result and device inner/outer relations.
 * Once varno/varattno mappings are constructed, we can generate a kernel
 * source that references device inner/outer relations.
 *
 * most of logic was copied from build_tlist_index
 */
static vartrans_info *
build_gpuhashjoin_vartrans(List *tlist, Index special_varno)
{
	vartrans_info  *vartrans;
	ListCell	   *cell;

	vartrans = palloc0(offsetof(vartrans_info, trans[64]));
	vartrans->max_vars = 64;
	vartrans->special_varno = special_varno;
	vartrans->tlist = tlist;

	foreach (cell, tlist)
	{
		TargetEntry	   *tle = lfirst(cell);

		if (tle->expr && IsA(tle->expr, Var))
		{
			Var	   *var = (Var *) tle->expr;
			int		i;

			if (vartrans->num_vars == vartrans->max_vars)
			{
				vartrans += vartrans->max_vars;
				vartrans = repalloc(vartrans,
									offsetof(vartrans_info,
											 trans[vartrans->max_vars]));
			}
			i = vartrans->num_vars++;
			vartrans->trans[i].varno = var->varno;
			vartrans->trans[i].varattno = var->varattno;
			vartrans->trans[i].resno = tle->resno;
		}
		else if (tle->expr && IsA(tle->expr, PlaceHolderVar))
			vartrans->has_ph_vars = true;
		else
			vartrans->has_non_vars = true;
	}
	return vartrans;
}

/*
 * fix_gpuhashjoin_expr
 *
 *
 *
 *
 */
typedef struct
{
	PlannerInfo	   *root;
	vartrans_info  *vartrans_maps[3];
	int				rtoffset;
	bool			expand_on_demand;
} fix_gpuhashjoin_expr_context;

static Node *
fix_gpuhashjoin_expr_mutator(Node *node, fix_gpuhashjoin_expr_context *context)
{
	Var	   *newnode;

	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		Var			   *var = (Var *) node;
		int				i, j;

		for (i=0; i < lengthof(context->vartrans_maps); i++)
		{
			vartrans_info  *vartrans;

			if (!context->vartrans_maps[i])
				continue;
		retry:
			vartrans = context->vartrans_maps[i];
			for (j=0; j < vartrans->num_vars; j++)
			{
				if (vartrans->trans[j].varno != var->varno ||
					vartrans->trans[j].varattno != var->varattno)
					continue;

				newnode = copyObject(var);
				newnode->varno = vartrans->special_varno;
				newnode->varattno = vartrans->trans[i].resno;
				if (newnode->varnoold > 0)
					newnode->varnoold += context->rtoffset;
				return (Node *)newnode;
			}
			if (context->expand_on_demand)
			{
				if (vartrans->num_vars == vartrans->max_vars)
				{
					vartrans->max_vars += vartrans->max_vars;
					vartrans = repalloc(vartrans,
										offsetof(vartrans_info,
												 trans[vartrans->max_vars]));
					context->vartrans_maps[i] = vartrans;
				}
				j = vartrans->num_vars++;
				vartrans->trans[j].varno = var->varno;
				vartrans->trans[j].varattno = var->varattno;
				vartrans->trans[j].resno = 0;	/* not in tlist! */
				goto retry;
			}
		}
		elog(ERROR, "variable not found in subplan target lists");
	}
	if (IsA(node, PlaceHolderVar))
	{
		PlaceHolderVar *phv = (PlaceHolderVar *) node;
		int				i, j;

		for (i=0; i < lengthof(context->vartrans_maps); i++)
		{
			vartrans_info  *vartrans = context->vartrans_maps[i];

			if (!vartrans || !vartrans->has_ph_vars)
				continue;

			for (j=0; j < vartrans->num_vars; j++)
			{
				TargetEntry	*tle = tlist_member(node, vartrans->tlist);

				if (tle)
				{
					Var	   *newnode =
						makeVarFromTargetEntry(vartrans->special_varno, tle);
					newnode->varnoold = 0;
					newnode->varoattno = 0;

					return (Node *)newnode;
				}
			}
			/* If not supplied by input plans, evaluate the contained expr */
			return fix_gpuhashjoin_expr_mutator((Node *) phv->phexpr,
												(void *) context);
		}
	}
	fix_expr_common(context->root, node);
	return expression_tree_mutator(node,
								   fix_gpuhashjoin_expr_mutator,
								   (void *) context);
}

static Node *
fix_gpuhashjoin_expr(PlannerInfo *root,
					 List *clauses,
					 vartrans_info **vartrans_result,
					 vartrans_info **vartrans_outer,
					 vartrans_info **vartrans_inner,
					 int rtoffset,
					 bool expand_on_demand)
{
	fix_gpuhashjoin_expr_context context;
	Node	   *result;

	memset(&context, 0, sizeof(fix_gpuhashjoin_expr_context));
	context.root = root;
	if (vartrans_result)
		context.vartrans_maps[0] = *vartrans_result;
	if (vartrans_outer)
		context.vartrans_maps[1] = *vartrans_outer;
	if (vartrans_inner)
		context.vartrans_maps[2] = *vartrans_inner;
	context.rtoffset = rtoffset;
	context.expand_on_demand = expand_on_demand;

	result = fix_gpuhashjoin_expr_mutator((Node *) clauses, &context);

	if (vartrans_result)
		*vartrans_result = context.vartrans_maps[0];
	if (vartrans_outer)
		*vartrans_outer = context.vartrans_maps[1];
	if (vartrans_inner)
		*vartrans_inner = context.vartrans_maps[2];

	return result;
}

/*
 * gpuhashjoin_set_plan_ref
 *
 * It fixes up varno and varattno according to the data format being
 * visible to targetlist or host_clauses. Unlike built-in join logics,
 * GpuHashJoin looks like a scan on a pseudo relation even though its
 * contents are actually consist of two different input streams.
 * So, note that it looks like all the columns are in outer relation,
 * however, GpuHashJoin manages the mapping which column come from
 * which column of what relation.
 */
static void
gpuhashjoin_set_plan_ref(PlannerInfo *root,
						 CustomPlan *custom_plan,
						 int rtoffset)
{
	GpuHashJoin	   *ghj = (GpuHashJoin *) custom_plan;
	Plan		   *outer_plan = outerPlan(ghj);
	Plan		   *inner_plan = innerPlan(ghj);
	char		   *kernel_source;
	codegen_context context;

	ghj->result_map =
		build_gpuhashjoin_vartrans(ghj->cplan.plan.targetlist, INDEX_VAR);
	ghj->outer_map =
		build_gpuhashjoin_vartrans(outer_plan->targetlist, OUTER_VAR);
	ghj->inner_map =
		build_gpuhashjoin_vartrans(inner_plan->targetlist, INNER_VAR);

	ghj->cplan.plan.targetlist = (List *)
		fix_gpuhashjoin_expr(root,
							 ghj->cplan.plan.targetlist,
							 &ghj->result_map,
							 NULL,
							 NULL,
							 rtoffset,
							 false);
	ghj->cplan.plan.qual = (List *)
		fix_gpuhashjoin_expr(root,
							 ghj->cplan.plan.qual,
							 &ghj->result_map,
							 NULL,
							 NULL,
							 rtoffset,
							 true);
	ghj->hash_clauses = (List *)
		fix_gpuhashjoin_expr(root,
							 ghj->hash_clauses,
							 NULL,
							 &ghj->outer_map,
							 &ghj->inner_map,
							 rtoffset,
							 false);
	ghj->qual_clauses = (List *)
		fix_gpuhashjoin_expr(root,
							 ghj->qual_clauses,
							 NULL,
							 &ghj->outer_map,
                             &ghj->inner_map,
                             rtoffset,
                             false);


	/* OK, let's general kernel source code */
	kernel_source = gpuhashjoin_codegen(root,
										ghj->hash_clauses,
										ghj->qual_clauses,
										&context);
	ghj->kernel_source = kernel_source;
	ghj->extra_flags = context.extra_flags;
    ghj->used_params = context.used_params;
    ghj->used_vars = context.used_vars;
	






	/* host qual has to be revised to reference scan slot! */

}

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
