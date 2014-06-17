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
#include "optimizer/subselect.h"
#include "optimizer/tlist.h"
#include "optimizer/var.h"
#include "parser/parsetree.h"
#include "storage/ipc.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/pg_crc.h"
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
	List	   *used_params;	/* template for kparams */
	List	   *used_vars;		/* var used in hash/qual clauses */
	List	   *pscan_tlist;	/* pseudo-scan target-list */
	List	   *pscan_resnums;	/* source resno of pseudo-scan tlist.
								 * positive number, if outer source */
	List	   *inner_resnums;	/* resource number to be fetched */
	List	   *inner_offsets;	/* offset to be placed on hash-entry */
	Size		inner_fixlen;	/* fixed length portion of hash-entry */
	TupleTableSlot *pscan_slot;	/* tuple-slot of pscan */
	List	   *hash_clauses;	/* expression form of hash_clauses */
	List	   *qual_clauses;	/* expression form of qual_clauses */
	Size		entry_width;	/* average width of hash_entry */
} GpuHashJoin;

typedef struct
{
	CustomPlanState	cps;
	JoinType		jointype;
	TupleTableSlot *pscan_slot;
	AttrNumber	   *pscan_resnums;
	List		   *inner_resnums;
	List		   *inner_offsets;
	Size			inner_fixlen;
	bool			outer_done;
	bool			outer_bulk;
	HeapTuple		outer_overflow;

	pgstrom_queue  *mqueue;
	Datum			dprog_key;
	kern_parambuf  *kparams;

	pgstrom_hashjoin_table *hash_table;

	/* chunk currently fetched in row-by-row mode */
	pgstrom_gpuhashjoin *curr_ghjoin;
	cl_uint			curr_index;
	cl_int			num_running;
	dlist_head		ready_pscans;

	List		   *hash_clauses;
	List		   *qual_clauses;

	pgstrom_perfmon	pfm;
} GpuHashJoinState;

/* variable to be placed in shared memory segment */
static shmem_startup_hook_type shmem_startup_hook_next;
static struct {
	slock_t		lock;
	dlist_head	free_list;	/* list of inactive gpuhashjoin slab */
	uint32		num_free;
	uint32		num_active;
} *gpuhashjoin_shm_values;

/* declaration of static functions */
static void clserv_process_gpuhashjoin(pgstrom_message *message);

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

	entry_size = INTALIGN(offsetof(kern_hashentry, keydata[0]) +
						  Max(sizeof(cl_ushort),
							  SHORTALIGN(context.natts / BITS_PER_BYTE)) +
						  context.width);

	return MAXALIGN(offsetof(kern_hashtable,
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
			appendStringInfo(str, "  pg_%s_t KVAR_%u = ",
							 dtype->type_name,
							 var_index);
			appendStringInfo(str,
							 "pg_%s_vref(kcs%s, errcode, %u, row_index);\n",
							 dtype->type_name,
							 dtype->type_length > 0 ? "" : ", ktoast",
							 outer_index);
			outer_index++;
		}
		var_index++;
	}

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
							 "               sizeof(%s.value));\n",
							 varname, varname, varname);
		else
			appendStringInfo(&calc,
							 "  if (!%s.isnull)\n"
							 "    COMP_CRC32(hash, VARDATA_ANY(%s.value),\n"
							 "                     VARSIZE_ANY(%s.value));\n",
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
							codegen_context *context,
							List **p_inner_resnums,
							List **p_inner_offsets,
							Size *p_inner_fixlen)
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
	inner_offset = INTALIGN(offsetof(kern_hashentry,
									 keydata[(inner_nums >> 3) + 1]));
	foreach (cell, context->used_vars)
	{
		Var	   *var = lfirst(cell);

		if (var->varno == INNER_VAR)
		{
			dtype = pgstrom_devtype_lookup(var->vartype);
			if (!dtype)
				elog(ERROR, "cache lookup failed for type: %u", var->vartype);

			if (dtype->type_length > 0)
			{
				inner_offset = TYPEALIGN(dtype->type_align, inner_offset);
				appendStringInfo(&tmpl,
								 "STROMCL_SIMPLE_HASHREF_TEMPLATE(%s,%s)\n",
								 dtype->type_name, dtype->type_base);
			}
			else
			{
				inner_offset = INTALIGN(inner_offset);
				appendStringInfo(&tmpl,
								 "STROMCL_VARLENA_HASHREF_TEMPLATE(%s)\n",
								 dtype->type_name);
			}
			appendStringInfo(&decl,
			"  pg_%s_t KVAR_%u = pg_%s_hashref(entry, errcode, %u, %u);\n",
							 dtype->type_name, var_index, dtype->type_name,
							 inner_index, inner_offset);
			/* offset to be remembered */
			*p_inner_resnums = lappend_int(*p_inner_resnums, var->varattno);
			*p_inner_offsets = lappend_int(*p_inner_offsets, inner_offset);

			inner_offset += (dtype->type_length > 0
							 ? dtype->type_length
							 : sizeof(cl_uint));
		}
		else if (var->varno == OUTER_VAR)
		{
			dtype = pgstrom_devtype_lookup(var->vartype);
			if (!dtype)
				elog(ERROR, "cache lookup failed for type: %u", var->vartype);

			appendStringInfo(&decl, "  pg_%s_t KVAR_%u = ",
							 dtype->type_name,
							 var_index);
			appendStringInfo(&decl,
							 "pg_%s_vref(kcs%s,errcode,%u,row_index);\n",
							 dtype->type_name,
							 dtype->type_length > 0 ? "" : ", ktoast",
							 outer_index);
			outer_index++;
		}
		else
			elog(ERROR, "Bug? var-node in neither inner nor outer relations");
		var_index++;
	}
	/* also, width of fixed length portion shall be remembered */
	*p_inner_fixlen = INTALIGN(inner_offset);

	appendStringInfo(str,
					 "%s\n"
					 "static cl_bool\n"
					 "gpuhashjoin_keycomp(__private cl_int *errcode,\n"
					 "                    __global kern_parambuf *kparams,\n"
					 "                    __global kern_hashentry *entry,\n"
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
					 "  return (!rc.isnull && rc.value ? true : false);\n"
					 "}\n",
					 tmpl.data,
					 decl.data,
					 eval_formula);
}

static char *
gpuhashjoin_codegen(PlannerInfo *root,
					List *hash_clauses,
					List *qual_clauses,
					codegen_context *context,
					List **p_inner_resnums,
					List **p_inner_offsets,
					Size *p_inner_fixlen)
{
	StringInfoData	str;
	StringInfoData	decl;

	memset(context, 0, sizeof(codegen_context));
	initStringInfo(&str);
	initStringInfo(&decl);

	/*
	 * placeholder of system constant - attrefs, kcs_head, ktoast_head
	 * and simple projection
	 */
	context->used_params = list_make4(makeConst(BYTEAOID,
												-1,
												InvalidOid,
												-1,
												PointerGetDatum(NULL),
												true,
												false),
									  makeConst(BYTEAOID,
												-1,
												InvalidOid,
												-1,
												PointerGetDatum(NULL),
												true,
												false),
									  makeConst(BYTEAOID,
												-1,
												InvalidOid,
												-1,
												PointerGetDatum(NULL),
												true,
												false),
									  makeConst(BYTEAOID,
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
	gpuhashjoin_codegen_compare(root, &decl,
								hash_clauses,
								qual_clauses,
								context,
								p_inner_resnums,
								p_inner_offsets,
								p_inner_fixlen);
	/*
	 * definition of gpuhashjoin_hashkey()
	 */
	gpuhashjoin_codegen_hashkey(root, &decl,
								hash_clauses,
								context);

	/*
	 * put declarations of type/func/param
	 */
	appendStringInfo(&str,
					 "%s%s%s%s",
					 pgstrom_codegen_type_declarations(context),
					 pgstrom_codegen_func_declarations(context),
					 pgstrom_codegen_param_declarations(context),
					 decl.data);
	/*
	 * to include opencl_hashjoin.h
	 */
	context->extra_flags |= DEVKERNEL_NEEDS_HASHJOIN;

	return str.data;
}

/*
 * estimate_gpuhashjoin_keywidth
 *
 * It estimates average width of hashjoin entries; sum of column width
 * referenced by both of hash-clauses and qual-clauses, according to
 * the list of used vars in actual.
 */
static Size
estimate_gpuhashjoin_keywidth(PlannerInfo *root,
							  List *used_vars,
							  List *inner_tlist)
{
	Size		width = offsetof(kern_hashentry, keydata);
	ListCell   *cell;

	foreach (cell, used_vars)
	{
		Var	   *var = lfirst(cell);
		int16	typlen;
		bool	typbyval;
		char	typalign;

		if (var->varno != INNER_VAR)
			continue;
		get_typlenbyvalalign(var->vartype, &typlen, &typbyval, &typalign);

		/* consider alignment */
		width = TYPEALIGN(typealign_get_width(typalign), width);

		/* add width of variables */
		if (typlen > 0)
			width += typlen;
		else
		{
			TargetEntry	   *tle;
			RangeTblEntry  *rte;
			Size			vl_len = 0;

			width += sizeof(cl_uint);	/* offset to variable field */

			/* average width of variable-length values */
			tle = get_tle_by_resno(inner_tlist, var->varattno);
			if (tle && IsA(tle->expr, Var))
			{
				Var	   *tlevar = (Var *) tle->expr;

				rte = rt_fetch(tlevar->varno, root->parse->rtable);
				if (rte && rte->rtekind == RTE_RELATION)
					vl_len = get_attavgwidth(rte->relid, tlevar->varattno);
			}
			if (vl_len == 0)
				vl_len = get_typavgwidth(var->vartype, var->vartypmod);
			width += INTALIGN(vl_len);
		}
	}
	return LONGALIGN(width);
}

/*
 * build_pseudoscan_tlist
 *
 * GpuHashJoin performs like a scan-node that run on pseudo relation being
 * constructed with two source relations. Any (pseudo) columns in this
 * relation are, of course, reference to either inner or outer relation.
 */
typedef struct
{
	List   *pscan_tlist;
	List   *pscan_resnums;
	List   *outer_tlist;
	List   *inner_tlist;
} build_pseudoscan_context;

static bool
build_pseudoscan_tlist_walker(Node *node, build_pseudoscan_context *context)
{
	if (!node)
		return false;
	if (IsA(node, Var) || IsA(node, PlaceHolderVar))
	{
		TargetEntry	   *tle = tlist_member(node, context->pscan_tlist);
		bool			is_outer;

		if (tle)
			return false;
		/*
		 * Not found in the current pseudo-scan tlist, so expand it
		 */
		tle = tlist_member(node, context->outer_tlist);
		if (tle)
			is_outer = true;
		else
		{
			tle = tlist_member(node, context->inner_tlist);
			is_outer = false;
		}
		if (tle)
		{
			TargetEntry	   *newtle
				= makeTargetEntry((Expr *) node,
								  list_length(context->pscan_tlist) + 1,
								  !tle->resname ? NULL : pstrdup(tle->resname),
								  tle->resjunk);
			context->pscan_tlist = lappend(context->pscan_tlist, newtle);
			context->pscan_resnums = lappend_int(context->pscan_resnums,
												 is_outer
												 ? tle->resno
												 : -tle->resno);
		}
		else if (IsA(node, PlaceHolderVar))
		{
			/*
			 * If referenced PlaceHolderVar is not on the underlying
			 * target-list directly, try to walk down its expression
			 * tree.
			 */
			PlaceHolderVar *phv = (PlaceHolderVar *) node;

			build_pseudoscan_tlist_walker((Node *)phv->phexpr, context);
		}
		else
			elog(ERROR, "bug? referenced var-node not in underlying tlist");

		return false;
	}
	return expression_tree_walker(node,
								  build_pseudoscan_tlist_walker,
								  (void *) context);
}

static void
build_pseudoscan_tlist(GpuHashJoin *ghashjoin)
{
	Plan	   *outer_plan = outerPlan(ghashjoin);
	Plan	   *inner_plan = innerPlan(ghashjoin);
	build_pseudoscan_context context;

	context.pscan_tlist = NIL;
	context.pscan_resnums = NIL;
	context.outer_tlist = outer_plan->targetlist;
	context.inner_tlist = inner_plan->targetlist;

	build_pseudoscan_tlist_walker((Node *) ghashjoin->cplan.plan.targetlist,
								  &context);
	build_pseudoscan_tlist_walker((Node *) ghashjoin->cplan.plan.qual,
								  &context);
	ghashjoin->pscan_tlist = context.pscan_tlist;
	ghashjoin->pscan_resnums = context.pscan_resnums;
}

/*
 * fix_gpuhashjoin_expr
 *
 * It mutate expression node to reference either INDEX, OUTER or INNER_VAR
 * during query execution.
 */
typedef struct
{
	PlannerInfo	   *root;
	List		   *outer_tlist;
	Index			outer_varno;
	List		   *inner_tlist;
	Index			inner_varno;
	int				rtoffset;
} fix_gpuhashjoin_expr_context;

static Var *
search_tlist_for_var(Var *varnode, List *tlist, Index newvarno, int rtoffset)
{
	ListCell   *cell;

	foreach (cell, tlist)
	{
		TargetEntry	   *tle = lfirst(cell);
		Var			   *tlevar;
		Var			   *newvar;

		if (!IsA(tle->expr, Var))
			continue;
		tlevar = (Var *) tle->expr;

		if (varnode->varno == tlevar->varno &&
			varnode->varattno == tlevar->varattno)
		{
			newvar = copyObject(varnode);
			newvar->varno = newvarno;
			newvar->varattno = tle->resno;
			if (newvar->varnoold > 0)
				newvar->varnoold += rtoffset;
			return newvar;
		}
	}
	return NULL;	/* not found */
}

static Var *
search_tlist_for_non_var(Node *node, List *tlist, Index newvarno, int rtoffset)
{
	TargetEntry	   *tle = tlist_member(node, tlist);

	if (tle)
	{
		Var	   *newvar;

		newvar = makeVarFromTargetEntry(newvarno, tle);
		newvar->varnoold = 0;   /* wasn't ever a plain Var */
		newvar->varoattno = 0;
		return newvar;
	}
	return NULL;
}

static Node *
fix_gpuhashjoin_expr_mutator(Node *node, fix_gpuhashjoin_expr_context *context)
{
	Var	   *newnode;

	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		newnode = search_tlist_for_var((Var *)node,
									   context->outer_tlist,
									   context->outer_varno,
									   context->rtoffset);
		if (newnode)
			return (Node *) newnode;
		if (context->inner_tlist)
		{
			newnode = search_tlist_for_var((Var *)node,
										   context->inner_tlist,
										   context->inner_varno,
										   context->rtoffset);
			if (newnode)
				return (Node *) newnode;
		}
		/* No referent found for Var */
        elog(ERROR, "variable not found in subplan target lists");
	}
	else if (IsA(node, PlaceHolderVar))
	{
		PlaceHolderVar *phv = (PlaceHolderVar *) node;

		newnode = search_tlist_for_non_var(node,
										   context->outer_tlist,
										   context->outer_varno,
										   context->rtoffset);
		if (newnode)
			return (Node *) newnode;
		if (context->inner_tlist)
		{
			newnode = search_tlist_for_non_var(node,
											   context->inner_tlist,
											   context->inner_varno,
											   context->rtoffset);
			if (newnode)
				return (Node *) newnode;
		}
		/* If not supplied by input plans, evaluate the contained expr */
		return fix_gpuhashjoin_expr_mutator((Node *)phv->phexpr, context);
	}
	else
	{
		/* Try matching more complex expressions too */
		newnode = search_tlist_for_non_var(node,
										   context->outer_tlist,
										   context->outer_varno,
										   context->rtoffset);
		if (newnode)
			return (Node *) newnode;
		if (context->inner_tlist)
		{
			newnode = search_tlist_for_non_var(node,
											   context->inner_tlist,
											   context->inner_varno,
											   context->rtoffset);
			if (newnode)
				return (Node *) newnode;
		}
	}
	fix_expr_common(context->root, node);
	return expression_tree_mutator(node,
								   fix_gpuhashjoin_expr_mutator,
								   (void *) context);
}

static List *
fix_gpuhashjoin_expr(PlannerInfo *root,
					 List *clauses,
					 List *outer_tlist, Index outer_varno,
					 List *inner_tlist, Index inner_varno,
					 int rtoffset)
{
	fix_gpuhashjoin_expr_context context;

	memset(&context, 0, sizeof(fix_gpuhashjoin_expr_context));
	context.root = root;
	context.outer_tlist = outer_tlist;
	context.outer_varno = outer_varno;
	context.inner_tlist = inner_tlist;
	context.inner_varno = inner_varno;
	context.rtoffset    = rtoffset;

	return (List *) fix_gpuhashjoin_expr_mutator((Node *) clauses,
												 &context);
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
	Plan		   *outer_plan = outerPlan(custom_plan);
	Plan		   *inner_plan = innerPlan(custom_plan);
	char		   *kernel_source;
	codegen_context context;

	/* build a pseudo scan target-list */
	build_pseudoscan_tlist(ghj);

	/* fixup tlist and qual according to the pseudo scan tlist */
	ghj->cplan.plan.targetlist =
		fix_gpuhashjoin_expr(root,
							 ghj->cplan.plan.targetlist,
							 ghj->pscan_tlist, INDEX_VAR,
							 NIL, (Index) 0,
							 rtoffset);
	ghj->cplan.plan.qual =
		fix_gpuhashjoin_expr(root,
							 ghj->cplan.plan.qual,
							 ghj->pscan_tlist, INDEX_VAR,
                             NIL, (Index) 0,
                             rtoffset);
	/* pseudo scan tlist is also fixed up according to inner/outer */
	ghj->pscan_tlist =
		fix_gpuhashjoin_expr(root,
							 ghj->pscan_tlist,
							 outer_plan->targetlist, OUTER_VAR,
							 inner_plan->targetlist, INNER_VAR,
							 rtoffset);
	/* hash_clauses and qual_clauses also see inner/outer */
	ghj->hash_clauses =
		fix_gpuhashjoin_expr(root,
							 ghj->hash_clauses,
							 outer_plan->targetlist, OUTER_VAR,
							 inner_plan->targetlist, INNER_VAR,
							 rtoffset);
	ghj->qual_clauses =
		fix_gpuhashjoin_expr(root,
							 ghj->qual_clauses,
							 outer_plan->targetlist, OUTER_VAR,
							 inner_plan->targetlist, INNER_VAR,
							 rtoffset);

	/* OK, let's general kernel source code */
	kernel_source = gpuhashjoin_codegen(root,
										ghj->hash_clauses,
										ghj->qual_clauses,
										&context,
										&ghj->inner_resnums,
										&ghj->inner_offsets,
										&ghj->inner_fixlen);
	ghj->kernel_source = kernel_source;
	ghj->extra_flags = context.extra_flags;
	ghj->used_params = context.used_params;
	ghj->used_vars = context.used_vars;
	ghj->entry_width = estimate_gpuhashjoin_keywidth(root,
													 ghj->used_vars,
													 inner_plan->targetlist);
}

static void
gpuhashjoin_finalize_plan(PlannerInfo *root,
						  CustomPlan *custom_plan,
						  Bitmapset **paramids,
						  Bitmapset **valid_params,
						  Bitmapset **scan_params)
{
	GpuHashJoin	   *ghj = (GpuHashJoin *)custom_plan;

	finalize_primnode(root, (Node *)ghj->hash_clauses, *paramids);
	finalize_primnode(root, (Node *)ghj->qual_clauses, *paramids);
}


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
	return false;	/* not supported yet */
	/* we can issue bulk-exec mode if no projection */
	if (cps->ps.ps_ProjInfo == NULL)
		return true;
	return false;
}


static CustomPlanState *
gpuhashjoin_begin(CustomPlan *node, EState *estate, int eflags)
{
	GpuHashJoin		   *ghashjoin = (GpuHashJoin *) node;
	GpuHashJoinState   *ghjs;
	TupleDesc			tupdesc;
	TupleDesc			tupdesc_outer;
	bytea			   *pdatum;
	Const			   *kparam_0;	/* bitmap of outer references */
	Const			   *kparam_1;	/* template of outer kcs_head */
	Const			   *kparam_2;	/* template of outer ktoast_head */
	Const			   *kparam_3;	/* simple projection info */
	cl_char			   *attrefs;
	ListCell		   *cell;
	bool				has_oid;
	int					index;

	/*
	 * create a state structure
	 */
	ghjs = palloc0(sizeof(GpuHashJoinState));
	NodeSetTag(ghjs, T_CustomPlanState);
	ghjs->cps.ps.plan = &node->plan;
	ghjs->cps.ps.state = estate;
	ghjs->cps.methods = &gpuhashjoin_plan_methods;
	ghjs->jointype = ghashjoin->jointype;

	/*
	 * create expression context
	 */
	ExecAssignExprContext(estate, &ghjs->cps.ps);

	/*
	 * initialize child expression
	 */
	ghjs->cps.ps.targetlist = (List *)
		ExecInitExpr((Expr *) node->plan.targetlist, &ghjs->cps.ps);
	ghjs->cps.ps.qual = (List *)
		ExecInitExpr((Expr *) node->plan.qual, &ghjs->cps.ps);
	ghjs->hash_clauses = (List *)
		ExecInitExpr((Expr *) ghashjoin->hash_clauses, &ghjs->cps.ps);
	ghjs->qual_clauses = (List *)
		ExecInitExpr((Expr *) ghashjoin->qual_clauses, &ghjs->cps.ps);

	/*
	 * initialize child nodes
	 */
	outerPlanState(ghjs) = ExecInitNode(outerPlan(ghashjoin), estate, eflags);
	innerPlanState(ghjs) = ExecInitNode(innerPlan(ghashjoin), estate, eflags);

	/*
	 * initialize "pseudo" scan slot
	 */
	if (!ExecContextForcesOids(&ghjs->cps.ps, &has_oid))
		has_oid = false;
	tupdesc = ExecTypeFromTL(ghashjoin->pscan_tlist, has_oid);

	ghjs->pscan_slot = ExecAllocTableSlot(&estate->es_tupleTable);
	ExecSetSlotDescriptor(ghjs->pscan_slot, tupdesc);

	Assert(tupdesc->natts == list_length(ghashjoin->pscan_resnums));
	ghjs->pscan_resnums = palloc0(sizeof(AttrNumber) * tupdesc->natts);
	index = 0;
	foreach (cell, ghashjoin->pscan_resnums)
		ghjs->pscan_resnums[index++] = lfirst_int(cell);

	/* parameters to build hash-table */
	ghjs->inner_resnums = list_copy(ghashjoin->inner_resnums);
	ghjs->inner_offsets = list_copy(ghashjoin->inner_offsets);
	ghjs->inner_fixlen  = ghashjoin->inner_fixlen;

	/* is bulk-scan available on outer node? */
	ghjs->outer_bulk = pgstrom_plan_can_multi_exec(outerPlanState(ghjs));
	ghjs->outer_done = false;
	ghjs->outer_overflow = NULL;

	/*
	 * initialize result tuple type and projection info
	 */
	ExecInitResultTupleSlot(estate, &ghjs->cps.ps);
	ExecAssignResultTypeFromTL(&ghjs->cps.ps);
	if (tlist_matches_tupdesc(&ghjs->cps.ps,
							  node->plan.targetlist,
							  INDEX_VAR,
							  tupdesc))
		ghjs->cps.ps.ps_ProjInfo = NULL;
	else
		ExecAssignProjectionInfo(&ghjs->cps.ps, tupdesc);

	/*
	 * setting up system kparams_0 - flags array of referenced attributes
	 */
	Assert(list_length(ghashjoin->used_params) >= 3);
	kparam_0 = (Const *) linitial(ghashjoin->used_params);
	Assert(IsA(kparam_0, Const) &&
		   kparam_0->consttype == BYTEAOID &&
		   kparam_0->constisnull);
	tupdesc_outer = ExecGetResultType(outerPlanState(ghjs));
	pdatum = kparam_make_attrefs(tupdesc_outer,
								 ghashjoin->used_vars,
								 OUTER_VAR);
	attrefs = (cl_char *)VARDATA(pdatum);
	kparam_0->constvalue = PointerGetDatum(pdatum);
	kparam_0->constisnull = false;

	/*
	 * setting up system kparam_1 - template of kcs_head
	 */
	kparam_1 = (Const *) lsecond(ghashjoin->used_params);
	Assert(IsA(kparam_1, Const) &&
		   kparam_1->consttype == BYTEAOID &&
		   kparam_1->constisnull);
	pdatum = kparam_make_kcs_head(tupdesc_outer,
								  attrefs,
								  0,	/* no additional syscols */
								  100);
	kparam_1->constvalue = PointerGetDatum(pdatum);
	kparam_1->constisnull = false;

	/*
	 * setting up system kparam_2 - template of ktoast_head
	 */
	kparam_2 = (Const *) lthird(ghashjoin->used_params);
	Assert(IsA(kparam_2, Const) &&
		   kparam_2->consttype == BYTEAOID &&
		   kparam_2->constisnull);
	pdatum = kparam_make_ktoast_head(tupdesc_outer,
									 attrefs,
									 0);	/* no additional syscols */
	kparam_2->constvalue = PointerGetDatum(pdatum);
	kparam_2->constisnull = false;

	/*
	 * Setting up system kparam_3 - simple projection info
	 */
	kparam_3 = (Const *) lfourth(ghashjoin->used_params);
	Assert(IsA(kparam_3, Const) &&
		   kparam_3->consttype == BYTEAOID &&
		   kparam_3->constisnull);
	pdatum = kparam_make_kprojection(ghashjoin->pscan_tlist);
	kparam_3->constvalue =  PointerGetDatum(pdatum);
	kparam_3->constisnull = false;

	ghjs->kparams = pgstrom_create_kern_parambuf(ghashjoin->used_params,
												 ghjs->cps.ps.ps_ExprContext);
	/*
	 * Setting up a kernel program and message queue
	 */
	Assert(ghashjoin->kernel_source != NULL);
	ghjs->dprog_key = pgstrom_get_devprog_key(ghashjoin->kernel_source,
											  ghashjoin->extra_flags);
	pgstrom_track_object((StromObject *)ghjs->dprog_key, 0);

	ghjs->mqueue = pgstrom_create_queue();
	pgstrom_track_object(&ghjs->mqueue->sobj, 0);

	/* Is perfmon needed? */
	ghjs->pfm.enabled = pgstrom_perfmon_enabled;

	return &ghjs->cps;
}

pgstrom_hashjoin_table *
gpuhashjoin_get_hash_table(pgstrom_hashjoin_table *hash_table)
{
	SpinLockAcquire(&hash_table->lock);
	Assert(hash_table->refcnt > 0);
	hash_table->refcnt++;
	SpinLockRelease(&hash_table->lock);

	return hash_table;
}

void
gpuhashjoin_put_hash_table(pgstrom_hashjoin_table *hash_table)
{
	bool	do_release = false;
	int		i;

	SpinLockAcquire(&hash_table->lock);
	Assert(hash_table->refcnt > 0);
	if (--hash_table->refcnt == 0)
	{
		Assert(hash_table->n_kernel == 0 && hash_table->m_hash == NULL);
		do_release = true;
	}
	SpinLockRelease(&hash_table->lock);
	if (do_release)
	{
		for (i=0; i < hash_table->num_rcs; i++)
			pgstrom_put_rcstore(hash_table->rcstore[i]);
		pgstrom_shmem_free(hash_table->rcstore);
		pgstrom_shmem_free(hash_table);
	}
}

static pgstrom_hashjoin_table *
gpuhashjoin_create_hash_table(GpuHashJoinState *ghjs)
{
	pgstrom_hashjoin_table *hash_table;
	GpuHashJoin	   *ghashjoin = (GpuHashJoin *) ghjs->cps.ps.plan;
	ListCell	   *cell;
	Size			allocated;
	Size			length;
	int				nkeys = 0;
	cl_ulong		nslots;
	double			nrows;

	/*
	 * estimate length of hash-table
	 */
	foreach (cell, ghashjoin->used_vars)
	{
		if (((Var *) lfirst(cell))->varno == INNER_VAR)
			nkeys++;
	}
	/* if we don't have enough information, assume 10K rows */
	nrows = ghashjoin->cplan.plan.plan_rows;
	if (nrows < 1000.0)
		nrows = 1000.0;
	nslots = (cl_ulong)(nrows * 1.25);

	length = (LONGALIGN(offsetof(pgstrom_hashjoin_table,
								 kern.colmeta[nkeys])) +
			  LONGALIGN(sizeof(cl_uint) * nslots) +
			  LONGALIGN(ghashjoin->entry_width * (nrows * 1.05)));

	/* allocate a shared memory segment */
	hash_table = pgstrom_shmem_alloc_alap(length, &allocated);
	if (!hash_table)
		elog(ERROR, "out of shared memory");
	memset(&hash_table->kern, 0, (LONGALIGN(offsetof(pgstrom_hashjoin_table,
													 kern.colmeta[nkeys])) +
								  LONGALIGN(sizeof(cl_uint) * nslots)));

	hash_table->rcstore = pgstrom_shmem_alloc(SHMEM_BLOCKSZ -
											  SHMEM_ALLOC_COST);
	if (!hash_table->rcstore)
	{
		pgstrom_shmem_free(hash_table);
		elog(ERROR, "out of shared memory");
	}

	/* initialize the fields */
	hash_table->sobj.stag = StromTag_HashJoinTable;
	SpinLockInit(&hash_table->lock);
	hash_table->refcnt = 1;
	hash_table->n_kernel = 0;	/* set by opencl-server */
	hash_table->m_hash = NULL;	/* set by opencl-server */

	hash_table->maxlen = (allocated - offsetof(pgstrom_hashjoin_table, kern));
	hash_table->kern.length = (LONGALIGN(offsetof(kern_hashtable,
												  colmeta[nkeys])) +
							   LONGALIGN(sizeof(cl_uint) * nslots));
	hash_table->kern.nslots = nslots;
	hash_table->kern.nkeys = 0;	/* to be incremented later */
	hash_table->num_rcs = 0;
	hash_table->max_rcs = (SHMEM_BLOCKSZ - SHMEM_ALLOC_COST) / sizeof(cl_uint);

	return hash_table;
}



static pgstrom_hashjoin_table *
gpuhashjoin_preload_hash_table_rs(GpuHashJoinState *ghjs,
								  pgstrom_hashjoin_table *hash_table,
								  TupleDesc tupdesc,
								  tcache_row_store *trs, cl_uint rcs_index)
{
	kern_hashentry	 *kentry;
	Size		kentry_sz;
	cl_uint		nitems = trs->kern.nrows;
	cl_uint	   *hash_slot = KERN_HASHTABLE_SLOT(&hash_table->kern);
	int			i, j;

	for (i=0; i < nitems; i++)
	{
		rs_tuple   *rs_tup = kern_rowstore_get_tuple(&trs->kern, i);
		HeapTuple	tuple = &rs_tup->htup;
		int			i_key = 0;
		pg_crc32	hash;
		ListCell   *lp1, *lp2;

		/*
		 * Expand hash table on demand - usually, should not happen
		 * as long as table statistics is enough fresh
		 */
		if (hash_table->kern.length + rs_tup->htup.t_len > hash_table->maxlen)
		{
			pgstrom_hashjoin_table *new_table;

			pgstrom_untrack_object(&hash_table->sobj);
			new_table = pgstrom_shmem_realloc(hash_table,
											  2 * hash_table->maxlen);
			if (!new_table)
			{
				pgstrom_shmem_free(hash_table);
				elog(ERROR, "out of shared memory");
			}
			new_table->maxlen += new_table->maxlen;
			elog(INFO, "hashjoin table expanded %u => %u",
				 hash_table->maxlen, new_table->maxlen);
			pgstrom_shmem_free(hash_table);
			pgstrom_track_object(&new_table->sobj, 0);
			hash_table = new_table;
		}

		kentry = (kern_hashentry *)((char *)&hash_table->kern +
									 hash_table->kern.length);
		kentry_sz = ghjs->inner_fixlen;
		INIT_CRC32(hash);
		forboth(lp1, ghjs->inner_resnums,
				lp2, ghjs->inner_offsets)
		{
			Form_pg_attribute attr;
			AttrNumber	resno = lfirst_int(lp1);
			Size		offset = lfirst_int(lp2);
			Datum		value;
			bool		isnull;

			attr = tupdesc->attrs[resno - 1];

			value = heap_getattr(tuple, resno, tupdesc, &isnull);
			if (isnull)
				kentry->keydata[i_key >> 3] &= ~(1 << (i_key & 7));
			else
			{
				kentry->keydata[i_key >> 3] |=  (1 << (i_key & 7));

				if (attr->attlen > 0)
				{
					if (attr->attbyval)
						memcpy((char *)kentry + offset,
							   &value,
							   attr->attlen);
					else
						memcpy((char *)kentry + offset,
							   DatumGetPointer(value),
							   attr->attlen);
					COMP_CRC32(hash, (char *)kentry + offset, attr->attlen);
				}
				else
				{
					*((cl_uint *)((char *)kentry + offset)) = kentry_sz;
					memcpy((char *)kentry + kentry_sz,
						   DatumGetPointer(value),
						   VARSIZE_ANY(value));
					COMP_CRC32(hash,
							   (char *)kentry + kentry_sz,
							   VARSIZE_ANY(value));
					kentry_sz += INTALIGN(VARSIZE_ANY(value));
				}
			}
			i_key++;
		}
		FIN_CRC32(hash);
		kentry->hash = hash;
		kentry->rowid = (((cl_ulong)rcs_index << 32) | (cl_ulong) i);

		/* insert this new entry */
		j = hash % hash_table->kern.nslots;

		kentry->next = hash_slot[j];
		hash_slot[j] = ((uintptr_t)kentry - (uintptr_t)&hash_table->kern);

		/* increment usage counter */
		hash_table->kern.length += LONGALIGN(kentry_sz);
	}
	return hash_table;
}

static pgstrom_hashjoin_table *
gpuhashjoin_preload_hash_table_cs(GpuHashJoinState *ghjs,
								  pgstrom_hashjoin_table *hash_table,
								  TupleDesc tupdesc,
								  tcache_column_store *tcs,
								  cl_uint rcs_index,
								  cl_uint nitems, cl_uint *rindex,
								  List *toast_resnums)
{
	kern_hashentry *kentry;
	cl_uint	   *hash_slot = KERN_HASHTABLE_SLOT(&hash_table->kern);
	Size		kentry_sz;
	ListCell   *lp1, *lp2;
	ListCell   *cell;
	int			i, j;


	for (i=0; i < nitems; i++)
	{
		int			i_key = 0;
		pg_crc32	hash;
		Size		required;

		j = (!rindex ? i : rindex[i]);

		/*
		 * precheck length of hash-entry. it's a little bit expensive
		 * if hash-key contains toast variable.
		 */
		required = ghjs->inner_fixlen;
		foreach (cell, toast_resnums)
		{
			AttrNumber	resno = lfirst_int(cell);
			cl_uint		vl_ofs;
			char	   *vl_ptr;

			if (!tcs->cdata[resno-1].values)
				elog(ERROR, "bug? referenced column is not columnized");
			if (!tcs->cdata[resno-1].toast)
				elog(ERROR, "but? referenced column has no toast buffer");
			if (tcs->cdata[resno-1].isnull &&
				att_isnull(j, tcs->cdata[resno-1].isnull))
				continue;	/* no need to count NULL datum */
			vl_ofs = ((cl_uint *)(tcs->cdata[resno-1].values))[j];
			vl_ptr = ((char *)tcs->cdata[resno-1].toast + vl_ofs);
			required += INTALIGN(VARSIZE_ANY(vl_ptr));
		}

		/* expand the hash-table if not available to store any more */
		if (hash_table->kern.length + required > hash_table->maxlen)
		{
			pgstrom_hashjoin_table *new_table;

			pgstrom_untrack_object(&hash_table->sobj);
			new_table = pgstrom_shmem_realloc(hash_table,
											  2 * hash_table->maxlen);
			if (!new_table)
			{
				pgstrom_shmem_free(hash_table);
				elog(ERROR, "out of shared memory");
			}
			new_table->maxlen += new_table->maxlen;
			elog(INFO, "hashjoin table expanded %u => %u",
				 hash_table->maxlen, new_table->maxlen);
			pgstrom_shmem_free(hash_table);
			pgstrom_track_object(&new_table->sobj, 0);
			hash_table = new_table;
		}
		kentry = (kern_hashentry *)((char *)&hash_table->kern +
									hash_table->kern.length);
		kentry_sz = ghjs->inner_fixlen;
		INIT_CRC32(hash);

		forboth(lp1, ghjs->inner_resnums,
				lp2, ghjs->inner_offsets)
		{
			Form_pg_attribute attr;
			AttrNumber	resno = lfirst_int(lp1);
			Size		offset = lfirst_int(lp2);

			attr = tupdesc->attrs[resno-1];
			if (tcs->cdata[resno-1].isnull &&
				att_isnull(j, tcs->cdata[resno-1].isnull))
				kentry->keydata[i_key >> 3] &= ~(1 << (i_key & 7));
			else
			{
				kentry->keydata[i_key >> 3] |=  (1 << (i_key & 7));

				if (attr->attlen > 0)
				{
					memcpy((char *)kentry + offset,
						   tcs->cdata[resno-1].values + j * attr->attlen,
						   attr->attlen);
					COMP_CRC32(hash, (char *)kentry + offset, attr->attlen);
				}
				else
				{
					cl_uint		vl_ofs;
					char	   *vl_ptr;

					*((cl_uint *)((char *)kentry + offset)) = kentry_sz;

					vl_ofs = ((cl_uint *)tcs->cdata[resno-1].values)[j];
					vl_ptr = ((char *)tcs->cdata[resno-1].toast) + vl_ofs;
					memcpy((char *)kentry + kentry_sz,
						   vl_ptr,
						   VARSIZE_ANY(vl_ptr));
					COMP_CRC32(hash, vl_ptr, VARSIZE_ANY(vl_ptr));
					kentry_sz += INTALIGN(VARSIZE_ANY(vl_ptr));
				}
			}
			i_key++;
		}
		FIN_CRC32(hash);
		kentry->hash = hash;
		kentry->rowid = (((cl_ulong)rcs_index << 32) | (cl_ulong) j);

		/* insert this new entry */
		j = hash % hash_table->kern.nslots;
		kentry->next = hash_slot[j];
		hash_slot[j] = ((uintptr_t)kentry - (uintptr_t)&hash_table->kern);

		/* increment usage counter */
		hash_table->kern.length += LONGALIGN(kentry_sz);
	}
	return hash_table;
}

static pgstrom_hashjoin_table *
gpuhashjoin_preload_hash_table(GpuHashJoinState *ghjs)
{
	pgstrom_hashjoin_table *hash_table;
	PlanState	   *subnode = innerPlanState(ghjs);
	TupleDesc		tupdesc = ExecGetResultType(subnode);
	bool			bulk_scan = pgstrom_plan_can_multi_exec(subnode);
	bool			end_of_scan = false;
	List		   *toast_resnums = NIL;
	ListCell	   *cell;
	StromObject	   *rcstore;
	HeapTuple		overflow = NULL;
	pgstrom_bulk_slot *bulk = NULL;
	struct timeval tv1, tv2;

	if (ghjs->pfm.enabled)
		gettimeofday(&tv1, NULL);

	/* pulls resource number with toast buffer */
	foreach (cell, ghjs->inner_resnums)
	{
		Form_pg_attribute attr = tupdesc->attrs[lfirst_int(cell) - 1];

		if (attr->attlen < 1)
			toast_resnums = lappend_int(toast_resnums, lfirst_int(cell));
	}

	hash_table = gpuhashjoin_create_hash_table(ghjs);
	while (!end_of_scan)
	{
		cl_uint		rcs_index;
		cl_uint		nitems;
		cl_uint	   *rindex;

		/*
		 * Fetch a row/column store from the inner subplan. If subplan
		 * does not support bulk-exec mode, we construct a row-store
		 * on the fly.
		 */
		if (bulk_scan)
		{
			bulk = (pgstrom_bulk_slot *)MultiExecProcNode(subnode);
			if (!bulk)
			{
				end_of_scan = true;
				break;
			}
			rcstore = bulk->rc_store;
			nitems = bulk->nitems;
			rindex = bulk->rindex;
		}
		else
		{
			tcache_row_store   *trs = NULL;
			TupleTableSlot	   *slot;
			HeapTuple			tuple;

			while (true)
			{
				if (HeapTupleIsValid(overflow))
				{
					tuple = overflow;
					overflow = NULL;
				}
				else
				{
					slot = ExecProcNode(subnode);
					if (TupIsNull(slot))
					{
						end_of_scan = true;
						break;
					}
					tuple = ExecFetchSlotTuple(slot);
				}
				if (!trs)
					trs = tcache_create_row_store(tupdesc);
				if (!tcache_row_store_insert_tuple(trs, tuple))
				{
					overflow = tuple;
					break;
				}
			}
			if (!trs)
				break;	/* no more inner tuples to be hashed */
			if (trs && trs->kern.nrows == 0)
				elog(ERROR, "bug? tcache_row_store can store no tuple");
			rcstore = (!trs ? NULL : &trs->sobj);
			nitems = trs->kern.nrows;
			rindex = NULL;	/* all rows should be visible */
		}
		/*
		 * Expand the array of row/column-store on demand
		 */
		if (hash_table->num_rcs == hash_table->max_rcs)
		{
			StromObject	   *new_array;

			elog(INFO, "max_rcs = %u", hash_table->max_rcs);
			hash_table->max_rcs += hash_table->max_rcs;
			new_array = pgstrom_shmem_realloc(hash_table,
											  sizeof(StromObject *) *
											  hash_table->max_rcs);
			if (!new_array)
			{
				pgstrom_put_rcstore(rcstore);
				elog(ERROR, "out of shared memory");
			}
		}
		hash_table->rcstore[hash_table->num_rcs] = rcstore;
		rcs_index = hash_table->num_rcs++;

		/*
		 * Move hash join keys into the hashjoin_table
		 */
		if (StromTagIs(rcstore, TCacheRowStore))
		{
			tcache_row_store *trs = (tcache_row_store *) rcstore;
			Assert(nitems == trs->kern.nrows);
			Assert(!rindex);
			hash_table =
				gpuhashjoin_preload_hash_table_rs(ghjs, hash_table,
												  tupdesc,
												  trs, rcs_index);
		}
		else if (StromTagIs(rcstore, TCacheColumnStore))
		{
			tcache_column_store *tcs = (tcache_column_store *) rcstore;
			hash_table =
				gpuhashjoin_preload_hash_table_cs(ghjs, hash_table,
												  tupdesc,
												  tcs, rcs_index,
												  nitems, rindex,
												  toast_resnums);
		}
		else
			elog(ERROR, "bug? neither row nor column store");

		if (bulk)
			pfree(bulk);
	}
	if (ghjs->pfm.enabled)
	{
		gettimeofday(&tv2, NULL);
		ghjs->pfm.time_to_load_inner += timeval_diff(&tv1, &tv2);
	}

	/*
	 * NOTE: if num_rcs == 0, it means no tuples were not preloaded.
	 * In this case, we don't need to return any tuples for inner join,
	 * or null attached outer scan results.
	 */
	return hash_table;
}

static inline void
gpuhashjoin_dump_hash_table(GpuHashJoinState *ghjs)
{
	PlanState		   *subnode = innerPlanState(ghjs);
	TupleDesc			tupdesc = ExecGetResultType(subnode);
	StringInfoData		str;
	pgstrom_hashjoin_table *phash;
	kern_hashtable	   *khash;
	kern_hashentry	   *kentry;
	cl_uint			   *kslots;
	cl_int				i, i_key;
	ListCell		   *lp1, *lp2;

	initStringInfo(&str);
	phash = ghjs->hash_table;
	khash = &phash->kern;
	kslots = KERN_HASHTABLE_SLOT(khash);
	elog(INFO,
		 "pgstrom_hashjoin_table {maxlen=%u, refcnt=%d, n_kernel=%d, "
		 "m_hash=%p, num_rcs=%d, max_rcs=%d}",
		 phash->maxlen, phash->refcnt, phash->n_kernel,
		 phash->m_hash, phash->num_rcs, phash->max_rcs);

	for (i=0; i < khash->nslots; i++)
	{
		if (kslots[i] == 0)
			continue;
		kentry = (kern_hashentry *)((char *)khash + kslots[i]);
	next:
		resetStringInfo(&str);

		i_key = 0;
		forboth(lp1, ghjs->inner_resnums,
				lp2, ghjs->inner_offsets)
		{
			AttrNumber	resno = lfirst_int(lp1);
			Size		offset = lfirst_int(lp2);
			Form_pg_attribute attr = tupdesc->attrs[resno-1];

			if (att_isnull(i_key, kentry->keydata))
			{
				appendStringInfo(&str, "%snull",
								 i_key == 0 ? "" : ", ");
			}
			else
			{
				Oid		typoutput;
				bool	is_varlena;
				Datum	value;

				getTypeOutputInfo(attr->atttypid, &typoutput, &is_varlena);
				if (attr->attlen > 0)
				{
					value = fetch_att((char *)kentry + offset,
									  attr->attbyval,
									  attr->attlen);
				}
				else
				{
					cl_uint		vl_ofs = *((cl_uint *)(char *)kentry + offset);
					value = PointerGetDatum((char *)kentry + vl_ofs);
				}
				appendStringInfo(&str, "%s%s",
								 i_key == 0 ? "" : ", ",
								 OidOutputFunctionCall(typoutput, value));
			}
			appendStringInfo(&str, "::%s", format_type_be(attr->atttypid));
			i_key++;
		}
		elog(INFO, "[%d] kentry (%p) rowid=%016lx hash=%08x { %s }",
			 i, kentry, kentry->rowid, kentry->hash, str.data);

		if (kentry->next != 0)
		{
			kentry = (kern_hashentry *)((char *)khash + kentry->next);
			goto next;
		}
	}
}

static void
pgstrom_release_gpuhashjoin(pgstrom_message *message)
{
	pgstrom_gpuhashjoin *gpuhashjoin = (pgstrom_gpuhashjoin *) message;

	/* unlink message queue and device program */
	pgstrom_put_queue(gpuhashjoin->msg.respq);
    pgstrom_put_devprog_key(gpuhashjoin->dprog_key);

	/* unlink hashjoin-table */
	gpuhashjoin_put_hash_table(gpuhashjoin->hjtable);

	/* unlink row/column store */
	pgstrom_put_rcstore(gpuhashjoin->rcs_src);
	if (gpuhashjoin->rcs_dst)
		pgstrom_put_rcstore(gpuhashjoin->rcs_dst);

	/* release kern_hashjoin */
	if (gpuhashjoin->kern)
		pgstrom_shmem_free(gpuhashjoin->kern);

	/* release a pgstrom_gpu_hashjoin slab */
	SpinLockAcquire(&gpuhashjoin_shm_values->lock);
	memset(gpuhashjoin, 0, sizeof(pgstrom_gpuhashjoin));
	gpuhashjoin_shm_values->num_active--;
	dlist_push_tail(&gpuhashjoin_shm_values->free_list,
					&gpuhashjoin->chain);
	gpuhashjoin_shm_values->num_free++;
	SpinLockRelease(&gpuhashjoin_shm_values->lock);
}

static pgstrom_gpuhashjoin *
pgstrom_create_gpuhashjoin(GpuHashJoinState *ghjs, StromObject *rcstore,
						   cl_uint nvalids, cl_uint *rindex)
{
	pgstrom_gpuhashjoin	*gpuhashjoin;
	dlist_node	   *dnode;
	Size			length;
	cl_uint			nrows;
	cl_uint			nrooms;
	kern_hashjoin  *khashjoin;
	kern_parambuf  *kparams;
	kern_resultbuf *kresults;

	/* acquire a pgstrom_gpu_hashjoin from the slab */
	SpinLockAcquire(&gpuhashjoin_shm_values->lock);
	if (dlist_is_empty(&gpuhashjoin_shm_values->free_list))
	{
		Size		allocated;
		uintptr_t	tailaddr;

		gpuhashjoin = pgstrom_shmem_alloc_alap(sizeof(pgstrom_gpuhashjoin),
											   &allocated);
		tailaddr = (uintptr_t)gpuhashjoin + allocated;
		while (((uintptr_t)gpuhashjoin +
				sizeof(pgstrom_gpuhashjoin)) <= tailaddr)
		{
			dlist_push_tail(&gpuhashjoin_shm_values->free_list,
							&gpuhashjoin->chain);
			gpuhashjoin++;
			gpuhashjoin_shm_values->num_free++;
		}
	}
	Assert(!dlist_is_empty(&gpuhashjoin_shm_values->free_list));
	gpuhashjoin_shm_values->num_free--;
	dnode = dlist_pop_head_node(&gpuhashjoin_shm_values->free_list);
	gpuhashjoin = dlist_container(pgstrom_gpuhashjoin, chain, dnode);
	gpuhashjoin_shm_values->num_active++;
	SpinLockRelease(&gpuhashjoin_shm_values->lock);

	/* initialize the common message field */
	memset(gpuhashjoin, 0, sizeof(pgstrom_gpuhashjoin));
	gpuhashjoin->msg.sobj.stag = StromTag_GpuHashJoin;
	SpinLockInit(&gpuhashjoin->msg.lock);
	gpuhashjoin->msg.refcnt = 1;
	gpuhashjoin->msg.respq = pgstrom_get_queue(ghjs->mqueue);
	gpuhashjoin->msg.cb_process = clserv_process_gpuhashjoin;
	gpuhashjoin->msg.cb_release = pgstrom_release_gpuhashjoin;
	gpuhashjoin->msg.pfm.enabled = ghjs->pfm.enabled;
	/* other fields also */
	gpuhashjoin->dprog_key = pgstrom_retain_devprog_key(ghjs->dprog_key);
	gpuhashjoin->hjtable = gpuhashjoin_get_hash_table(ghjs->hash_table);
	gpuhashjoin->rcs_src = pgstrom_get_rcstore(rcstore);
	gpuhashjoin->rcs_dst = NULL;	/* result pscan-slot */

	/* setting up kern_hashjoin (pair of kparams & kresults) */
	length = STROMALIGN(ghjs->kparams->length);
	length += STROMALIGN(offsetof(kern_resultbuf,
						 results[3 * (Size)(1.5 * (double) nvalids)]));
	khashjoin = pgstrom_shmem_alloc_alap(length, &length);
	if (!khashjoin)
	{
		pgstrom_put_message(&gpuhashjoin->msg);
		elog(ERROR, "out of shared memory");
	}
	kparams = KERN_HASHJOIN_PARAMBUF(khashjoin);
	memcpy(kparams, ghjs->kparams, ghjs->kparams->length);
	kresults = KERN_HASHJOIN_RESULTBUF(khashjoin);

	length -= ((uintptr_t)&kresults->results[0] - (uintptr_t)khashjoin);
	kresults->nrooms = length / sizeof(cl_uint);
	kresults->nitems = 0;
	kresults->errcode = StromError_Success;

	/*
	 * update kcs_head and ktoast_head according to column references
	 */
	if (StromTagIs(rcstore, TCacheColumnStore))
		nrows = nrooms = ((tcache_column_store *)rcstore)->nrows;
	else
	{
		nrows = 0;
		nrooms = nvalids;
	}
	kparam_refresh_kcs_head(kparams, nrows, nrooms);
	kparam_refresh_ktoast_head(kparams, rcstore);

	/*
	 * NOTE: kern_resultbuf is not used unless DMA writeback happen,
	 * so we can use this buffer to save rindex; that shall be sent
	 * to the device prior to kernel execution then never referenced.
	 */
	if (!rindex)
		gpuhashjoin->src_nitems = -1;
	else
	{
		gpuhashjoin->src_nitems = nvalids;
		memcpy(kresults->results, rindex, sizeof(cl_int) * nvalids);
	}
	gpuhashjoin->kern = khashjoin;

	return gpuhashjoin;
}

static pgstrom_gpuhashjoin *
gpuhashjoin_load_next_outer(GpuHashJoinState *ghjs)
{
	PlanState	   *subnode = outerPlanState(ghjs);
	TupleDesc		tupdesc = ExecGetResultType(subnode);
	StromObject	   *rcstore = NULL;
	pgstrom_bulk_slot *bulk = NULL;
	cl_uint			nitems;
	cl_uint		   *rindex = NULL;

	if (ghjs->outer_done)
		return NULL;

	if (!ghjs->outer_bulk)
	{
		/* Scan the outer relation using row-by-row mode */
		tcache_row_store *trs = NULL;
		HeapTuple	tuple;

		while (true)
		{
			if (HeapTupleIsValid(ghjs->outer_overflow))
			{
				tuple = ghjs->outer_overflow;
				ghjs->outer_overflow = NULL;
			}
			else
			{
				TupleTableSlot *slot = ExecProcNode(subnode);
				if (TupIsNull(slot))
				{
					ghjs->outer_done = true;
					break;
				}
				tuple = ExecFetchSlotTuple(slot);
			}
			if (!trs)
				trs = tcache_create_row_store(tupdesc);
			if (!tcache_row_store_insert_tuple(trs, tuple))
			{
				ghjs->outer_overflow = tuple;
				break;
			}
		}
		if (trs)
		{
			nitems = trs->kern.nrows;
			rindex = NULL;
			rcstore = &trs->sobj;
		}
	}
	else
	{
		/* Scan the outer relation using bulk-scan mode */
		bulk = (pgstrom_bulk_slot *)MultiExecProcNode(subnode);
		if (!bulk)
			ghjs->outer_done = true;
		else
		{
			nitems = bulk->nitems;
			rindex = bulk->rindex;
			rcstore = bulk->rc_store;
		}
	}
	/* Is there tuples to return? */
	if (!rcstore)
		return NULL;

	return pgstrom_create_gpuhashjoin(ghjs, rcstore, nitems, rindex);
}

static bool
gpuhashjoin_next_tuple(GpuHashJoinState *ghjs, TupleTableSlot *slot)
{
	pgstrom_gpuhashjoin *ghjoin = ghjs->curr_ghjoin;
	tcache_column_store *tcs = (tcache_column_store *) ghjoin->rcs_dst;
	TupleDesc			tupdesc = slot->tts_tupleDescriptor;

	Assert(tcs->ncols == tupdesc->natts);
	while (ghjs->curr_index < tcs->nrows)
	{
		cl_int	index = ghjs->curr_index++;
		int		i;

		slot = ExecStoreAllNullTuple(slot);
		for (i=0; i < tupdesc->natts; i++)
		{
			Form_pg_attribute attr = tupdesc->attrs[i];
			Datum	value;

			if (!attr->attnotnull &&
				att_isnull(index, tcs->cdata[i].isnull))
				continue;

			if (attr->attlen > 0)
			{
				Assert(tcs->cdata[i].values);

				value = fetch_att(tcs->cdata[i].values +
								  attr->attlen * index,
								  attr->attbyval,
								  attr->attlen);
			}
			else
			{
				cl_uint	vl_ofs;

				Assert(tcs->cdata[i].values && tcs->cdata[i].toast);
				vl_ofs = ((cl_uint *)tcs->cdata[i].values)[index];
				value = PointerGetDatum((char *)tcs->cdata[i].toast + vl_ofs);
			}
			slot->tts_values[i] = value;
			slot->tts_isnull[i] = false;
		}

		/*
		 * check host clauses, if any
		 */
		if (ghjs->cps.ps.qual &&
			!ExecQual(ghjs->cps.ps.qual,
					  ghjs->cps.ps.ps_ExprContext,
					  false))
			continue;	/* ...try to next tuple */

		return true;
	}
	return false;
}

static TupleTableSlot *
gpuhashjoin_exec(CustomPlanState *node)
{
	GpuHashJoinState   *ghjs = (GpuHashJoinState *) node;
	TupleTableSlot	   *slot = ghjs->pscan_slot;
	pgstrom_gpuhashjoin *ghjoin;

	if (!ghjs->hash_table)
	{
		ghjs->hash_table = gpuhashjoin_preload_hash_table(ghjs);
		/*
		 * outer join is not supported right now, so an empty inner relation
		 * stream will lead empty result without outer relation scan.
		 */
		if (ghjs->hash_table->num_rcs == 0)
			return NULL;
	}

	ExecClearTuple(slot);
	while (!ghjs->curr_ghjoin || gpuhashjoin_next_tuple(ghjs, slot))
	{
		pgstrom_message	   *msg;
		dlist_node		   *dnode;

		/* release the current hashjoin chunk, being already fetched */
		if (ghjs->curr_ghjoin)
		{
			msg = &ghjs->curr_ghjoin->msg;
			if (msg->pfm.enabled)
				pgstrom_perfmon_add(&ghjs->pfm, &msg->pfm);
			Assert(msg->refcnt == 1);
			pgstrom_untrack_object(&msg->sobj);
			pgstrom_put_message(msg);
			ghjs->curr_ghjoin = NULL;
			ghjs->curr_index = 0;
		}

		/*
		 * dequeue the running gpuhashjoin chunk being already processed
		 */
		while ((msg = pgstrom_try_dequeue_message(ghjs->mqueue)) != NULL)
		{
			Assert(ghjs->num_running > 0);
			ghjs->num_running--;
			dlist_push_tail(&ghjs->ready_pscans, &msg->chain);
		}

		/*
		 * Keep number of asynchronous hashjoin request a particular level,
		 * unless it does not exceed pgstrom_max_async_chunks and any new
		 * response is not replied during the loading.
		 */
		while (!ghjs->outer_done &&
			   ghjs->num_running <= pgstrom_max_async_chunks)
		{
			pgstrom_gpuhashjoin *ghjoin = gpuhashjoin_load_next_outer(ghjs);

			if (!ghjoin)
				break;	/* outer scan reached to end of the relation */
			if (!pgstrom_enqueue_message(&ghjoin->msg))
			{
				pgstrom_put_message(&ghjoin->msg);
				elog(ERROR, "failed to enqueue pgstrom_gpuhashjoin message");
			}
			ghjs->num_running++;

			msg = pgstrom_try_dequeue_message(ghjs->mqueue);
			if (msg)
			{
				ghjs->num_running--;
				dlist_push_tail(&ghjs->ready_pscans, &msg->chain);
				break;
			}
		}

		/*
		 * wait for server's response if no available chunks were replied
		 */
		if (dlist_is_empty(&ghjs->ready_pscans))
		{
			/* OK, no more request should be fetched */
			if (ghjs->num_running == 0)
				break;

			msg = pgstrom_dequeue_message(ghjs->mqueue);
			if (!msg)
				elog(ERROR, "message queue wait timeout");
			ghjs->num_running--;
			dlist_push_tail(&ghjs->ready_pscans, &msg->chain);
		}

		/*
		 * picks up next available chunks, if any
		 */
		Assert(!dlist_is_empty(&ghjs->ready_pscans));
		dnode = dlist_pop_head_node(&ghjs->ready_pscans);
		ghjoin = dlist_container(pgstrom_gpuhashjoin, msg.chain, dnode);

		/*
		 * Raise an error, if significan error was reported
		 */
		if (ghjoin->msg.errcode != StromError_Success)
		{
			if (ghjoin->msg.errcode == CL_BUILD_PROGRAM_FAILURE)
			{
				const char *buildlog
					= pgstrom_get_devprog_errmsg(ghjoin->dprog_key);
				const char *kern_source
					= ((GpuHashJoin *)node->ps.plan)->kernel_source;

				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("PG-Strom: OpenCL execution error (%s)\n%s",
								pgstrom_strerror(ghjoin->msg.errcode),
								kern_source),
						 errdetail("%s", buildlog)));
			}
			else
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("PG-Strom: OpenCL execution error (%s)",
								pgstrom_strerror(ghjoin->msg.errcode))));
			}
		}
		/*
		 * NOTE: right now, we assume device can run simple projection
		 * if all the device clauses were runnable. So, we put assertion
		 * here, however, we need to put special case handing if ghjoin
		 * returned recheck-by-cpu.
		 */
		Assert(ghjoin->rcs_dst != NULL);
		ghjs->curr_ghjoin = ghjoin;
		ghjs->curr_index = 0;
	}
	/* can valid tuple be fetched? */
	if (TupIsNull(slot))
		return slot;

	/* applies host-side projection, if any */
	if (ghjs->cps.ps.ps_ProjInfo)
	{
		ProjectionInfo *pj_info = ghjs->cps.ps.ps_ProjInfo;
		ExprContext	   *econtext = ghjs->cps.ps.ps_ExprContext;
		ExprDoneCond	is_done;

		/*
		 * FIXME: we may need to revise the code according to ExecScan.
		 */
		econtext->ecxt_scantuple = slot;
		return ExecProject(pj_info, &is_done);
	}
	return slot;
}

static Node *
gpuhashjoin_exec_multi(CustomPlanState *node)
{
	// we can use bulk-scan mode if no projection, no host quals


	elog(ERROR, "not implemented yet");
	return NULL;
}

static void
gpuhashjoin_end(CustomPlanState *node)
{
	GpuHashJoinState   *ghjs = (GpuHashJoinState *) node;


	/*
	 *  Free the exprcontext
	 */
	ExecFreeExprContext(&node->ps);

	/*
	 * clean out hash-table
	 */
	if (ghjs->hash_table)
		gpuhashjoin_put_hash_table(ghjs->hash_table);

	/*
	 * clean out kernel source and message queue
	 */
	Assert(ghjs->dprog_key);
	pgstrom_untrack_object((StromObject *)ghjs->dprog_key);
	pgstrom_put_devprog_key(ghjs->dprog_key);

	Assert(ghjs->mqueue);
	pgstrom_untrack_object(&ghjs->mqueue->sobj);
	pgstrom_close_queue(ghjs->mqueue);

	/*
	 * clean up subtrees
	 */
	ExecEndNode(outerPlanState(node));
	ExecEndNode(innerPlanState(node));
}

static void
gpuhashjoin_rescan(CustomPlanState *node)
{
	elog(ERROR, "not implemented yet");
}

static void
gpuhashjoin_explain_rel(CustomPlanState *node, ExplainState *es)
{
	GpuHashJoinState   *ghjs = (GpuHashJoinState *) node;
	const char		   *jointype;

	switch (ghjs->jointype)
	{
		case JOIN_INNER:
			jointype = "Inner";
			break;
		case JOIN_LEFT:
			jointype = "Left";
			break;
		case JOIN_FULL:
			jointype = "Full";
			break;
		case JOIN_RIGHT:
			jointype = "Right";
			break;
		case JOIN_SEMI:
			jointype = "Semi";
			break;
		case JOIN_ANTI:
			jointype = "Anti";
			break;
		default:
			jointype = "???";
			break;
	}

	if (es->format == EXPLAIN_FORMAT_TEXT)
		appendStringInfo(es->str, " using %s Join", jointype);
	else
		ExplainPropertyText("Join Type", jointype, es);
}

static void
gpuhashjoin_explain(CustomPlanState *node, List *ancestors, ExplainState *es)
{
	GpuHashJoinState *ghjs = (GpuHashJoinState *) node;
	GpuHashJoin	   *ghashjoin = (GpuHashJoin *) node->ps.plan;
	StringInfoData	str;
	List		   *context;
	ListCell	   *cell;
	bool			useprefix;
	bool			is_first;

	initStringInfo(&str);

	/* Set up deparsing context */
	context = deparse_context_for_planstate((Node *) &node->ps,
											ancestors,
											es->rtable,
											es->rtable_names);
	/* device referenced columns */
	useprefix = es->verbose;
	is_first = false;
	foreach (cell, ghashjoin->used_vars)
	{
		if (is_first)
			appendStringInfo(&str, ", ");
		appendStringInfo(&str, "%s",
						 deparse_expression(lfirst(cell),
											context,
											useprefix,
											false));
		is_first = true;
	}
	ExplainPropertyText("Device references", str.data, es);

	if (ghashjoin->hash_clauses)
		show_scan_qual(ghashjoin->hash_clauses,
					   "hash clauses", &node->ps, ancestors, es);
	if (ghashjoin->qual_clauses)
		show_scan_qual(ghashjoin->qual_clauses,
					   "qual clauses", &node->ps, ancestors, es);
	if (ghashjoin->cplan.plan.qual)
		show_scan_qual(ghashjoin->cplan.plan.qual,
					   "host clauses", &node->ps, ancestors, es);
	show_device_kernel(ghjs->dprog_key, es);

	if (es->analyze && ghjs->pfm.enabled)
		pgstrom_perfmon_explain(&ghjs->pfm, es);
}

static Bitmapset *
gpuhashjoin_get_relids(CustomPlanState *node)
{
	/* nothing to do because core backend walks down inner/outer subtree */
	return NULL;
}

static Node *
gpuhashjoin_get_special_var(CustomPlanState *node, Var *varnode)
{
	GpuHashJoin	   *ghashjoin = (GpuHashJoin *)node->ps.plan;
	TargetEntry	   *tle;

	if (varnode->varno == INDEX_VAR)
	{
		tle = get_tle_by_resno(ghashjoin->pscan_tlist, varnode->varattno);
		if (tle)
			return (Node *)tle->expr;
	}
	else if (varnode->varno == OUTER_VAR)
	{
		Plan   *outer_plan = outerPlan(ghashjoin);

		if (outer_plan)
		{
			tle = get_tle_by_resno(outer_plan->targetlist, varnode->varattno);
			if (tle)
				return (Node *) tle->expr;
		}
	}
	else if (varnode->varno == INNER_VAR)
	{
		Plan   *inner_plan = innerPlan(ghashjoin);

		if (inner_plan)
		{
			tle = get_tle_by_resno(inner_plan->targetlist, varnode->varattno);
			if (tle)
				return (Node *) tle->expr;
		}
	}
	elog(ERROR, "variable (varno=%u,varattno=%d) is not relevant tlist",
		 varnode->varno, varnode->varattno);
	return NULL;	/* be compiler quiet */
}


static void
gpuhashjoin_textout_plan(StringInfo str, const CustomPlan *node)
{
	GpuHashJoin	   *plannode = (GpuHashJoin *) node;

	appendStringInfo(str, " :jointype %d", (int)plannode->jointype);

	appendStringInfo(str, " :kernel_source ");
	_outToken(str, plannode->kernel_source);

	appendStringInfo(str, " :extra_flags %u", plannode->extra_flags);

	appendStringInfo(str, " :used_params %s",
					 nodeToString(plannode->used_params));
	appendStringInfo(str, " :used_vars %s",
					 nodeToString(plannode->used_vars));
	appendStringInfo(str, " :pscan_tlist %s",
					 nodeToString(plannode->pscan_tlist));
	appendStringInfo(str, " :pscan_resnums %s",
					 nodeToString(plannode->pscan_resnums));
	appendStringInfo(str, " :hash_clauses %s",
					 nodeToString(plannode->hash_clauses));
	appendStringInfo(str, " :qual_clauses %s",
					 nodeToString(plannode->qual_clauses));
}

static CustomPlan *
gpuhashjoin_copy_plan(const CustomPlan *from)
{
	GpuHashJoin	   *oldnode = (GpuHashJoin *) from;
	GpuHashJoin	   *newnode = palloc0(sizeof(GpuHashJoin));

	CopyCustomPlanCommon((Node *)from, (Node *)newnode);
	newnode->jointype = oldnode->jointype;
	if (oldnode->kernel_source)
		newnode->kernel_source = pstrdup(oldnode->kernel_source);
	newnode->extra_flags   = oldnode->extra_flags;
	newnode->used_params   = copyObject(oldnode->used_params);
	newnode->used_vars     = copyObject(oldnode->used_vars);
	newnode->pscan_tlist   = copyObject(oldnode->pscan_tlist);
	newnode->pscan_resnums = copyObject(oldnode->pscan_resnums);
	newnode->hash_clauses  = copyObject(oldnode->hash_clauses);
	newnode->qual_clauses  = copyObject(oldnode->qual_clauses);

	return &newnode->cplan;
}

static void
pgstrom_startup_gpuhashjoin(void)
{
	bool	found;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	gpuhashjoin_shm_values =
		ShmemInitStruct("gpuhashjoin_shm_values",
						MAXALIGN(sizeof(*gpuhashjoin_shm_values)),
						&found);
	Assert(!found);
	memset(gpuhashjoin_shm_values, 0, sizeof(*gpuhashjoin_shm_values));
	SpinLockInit(&gpuhashjoin_shm_values->lock);
	dlist_init(&gpuhashjoin_shm_values->free_list);
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

	/* shared memory allocation */
	RequestAddinShmemSpace(MAXALIGN(sizeof(*gpuhashjoin_shm_values)));
	shmem_startup_hook_next = shmem_startup_hook;
	shmem_startup_hook = pgstrom_startup_gpuhashjoin;
}

/* ----------------------------------------------------------------
 *
 * NOTE: below is the code being run on OpenCL server context
 *
 * ---------------------------------------------------------------- */
typedef struct 
{
	pgstrom_message *msg;
	cl_command_queue kcmdq;
	cl_program	program;
	cl_kernel	kernel;
	cl_mem		m_hash;		/* kern_hashtable */
	cl_mem		m_join;		/* kern_hashjoin */
	cl_mem		m_rstore;	/* kern_row_store */
	cl_mem		m_cstore;	/* kern_column_store */
	cl_mem		m_toast;	/* kern_toastbuf */
	cl_int		dindex;
	cl_int		ev_index;
	cl_event	events[30];
} clstate_gpuhashjoin;

static void
clserv_respond_hashjoin(cl_event event, cl_int ev_status, void *private)
{
	clstate_gpuhashjoin	*clghj = (clstate_gpuhashjoin *) private;
	pgstrom_gpuhashjoin *ghjoin = (pgstrom_gpuhashjoin *) clghj->msg;
	pgstrom_hashjoin_table *hjtable = ghjoin->hjtable;
	kern_resultbuf		*kresult = KERN_HASHJOIN_RESULTBUF(ghjoin->kern);

	if (ev_status != CL_COMPLETE)
	{
		clserv_log("unexpected CL_EVENT_COMMAND_EXECUTION_STATUS: %d",
				   ev_status);
		ghjoin->msg.errcode = StromError_OpenCLInternal;
	}
	else if (ghjoin->msg.errcode == StromError_Success)
	{
		ghjoin->msg.errcode = kresult->errcode;
	}

	/* collect performance statistics */
	if (ghjoin->msg.pfm.enabled)
	{
		/* implement it later */
	}

	/* release opencl resources */
	while (clghj->ev_index > 0)
		clReleaseEvent(clghj->events[--clghj->ev_index]);
	if (clghj->m_hash)
		clReleaseMemObject(clghj->m_hash);
	if (clghj->m_join)
		clReleaseMemObject(clghj->m_join);
	if (clghj->m_rstore)
		clReleaseMemObject(clghj->m_rstore);
	if (clghj->m_cstore)
		clReleaseMemObject(clghj->m_cstore);
	if (clghj->m_toast)
		clReleaseMemObject(clghj->m_toast);
	if (clghj->kernel)
		clReleaseKernel(clghj->kernel);
	if (clghj->program)
		clReleaseProgram(clghj->program);

	/* remove hashjoin-table, if no longer referenced */
	SpinLockAcquire(&hjtable->lock);
	Assert(hjtable->n_kernel > 0);
	if (--hjtable->n_kernel == 0)
	{
		hjtable->m_hash = NULL;
		hjtable->ev_hash = NULL;
	}
	SpinLockRelease(&hjtable->lock);	
	free(clghj);

	/*
	 * Post-process of gpuhashjoin. If successfully done, it moves to post-
	 * hashjoin process to generate pseudo relation scan according to the
	 * result of join. If result buffer does not have enough capacity, we
	 * expand it and try to run hashjoin again. Elsewhere, error should be
	 * informed to the backend.
	 */
	if (ghjoin->msg.errcode == StromError_Success)
	{
		/* OK, move to the post inner-join process */
		ghjoin->hashjoin_done = true;
		pgstrom_enqueue_message(&ghjoin->msg);
	}
	else if (ghjoin->msg.errcode == StromError_DataStoreNoSpace)
	{
		/* expand the result buffer then retry, if rough estimation didn't
		 * offer enough space to store. */
		kern_hashjoin  *old_kern = ghjoin->kern;
		kern_hashjoin  *new_kern;
		cl_uint			nitems;
		Size			length;

		Assert(kresult->nitems > kresult->nrooms);
		nitems = kresult->nitems;
		length = (KERN_HASHJOIN_PARAMBUF_LENGTH(old_kern) +
				  STROMALIGN(offsetof(kern_resultbuf, results[3 * nitems])));
		new_kern = pgstrom_shmem_alloc(length);
		if (!new_kern)
		{
			ghjoin->msg.errcode = StromError_OutOfSharedMemory;
			pgstrom_reply_message(&ghjoin->msg);
			return;
		}
		memcpy(KERN_HASHJOIN_PARAMBUF(new_kern),
			   KERN_HASHJOIN_PARAMBUF(old_kern),
			   KERN_HASHJOIN_PARAMBUF_LENGTH(old_kern));
		kresult = KERN_HASHJOIN_RESULTBUF(new_kern);
		kresult->nrooms = nitems;
		kresult->nitems = 0;
		kresult->errcode = StromError_Success;

		ghjoin->kern = new_kern;
		pgstrom_shmem_free(old_kern);

		/* retry gpuhashjoin with wider result buffer */
		pgstrom_enqueue_message(&ghjoin->msg);
	}
	else
	{
		/* other errors */
		pgstrom_reply_message(&ghjoin->msg);
	}
}

static clstate_gpuhashjoin *
clserv_process_gpuhashjoin_common(pgstrom_gpuhashjoin *ghjoin)
{
	pgstrom_hashjoin_table *hjtable = ghjoin->hjtable;
	clstate_gpuhashjoin *clghj;
	cl_int		rc;

	/* state object of gpuhashjoin with row-store */
	clghj = calloc(1, sizeof(clstate_gpuhashjoin));
	if (!clghj)
	{
		rc = CL_OUT_OF_HOST_MEMORY;
		goto error;
	}
	clghj->msg = &ghjoin->msg;

	/*
	 * First of all, it looks up a program object to be run on
	 * the supplied row-store. We may have three cases.
	 * 1) NULL; it means the required program is under asynchronous
	 *    build, and the message is kept on its internal structure
	 *    to be enqueued again. In this case, we have nothing to do
	 *    any more on the invocation.
	 * 2) BAD_OPENCL_PROGRAM; it means previous compile was failed
	 *    and unavailable to run this program anyway. So, we need
	 *    to reply StromError_ProgramCompile error to inform the
	 *    backend this program.
	 * 3) valid cl_program object; it is an ideal result. pre-compiled
	 *    program object was on the program cache, and cl_program
	 *    object is ready to use.
	 */
    clghj->program = clserv_lookup_device_program(ghjoin->dprog_key,
												  &ghjoin->msg);
	if (!clghj->program)
	{
		free(clghj);
		return NULL;	/* message is in waitq, being retried later */
	}
	if (clghj->program == BAD_OPENCL_PROGRAM)
	{
		rc = CL_BUILD_PROGRAM_FAILURE;
		goto error;
	}

	/*
	 * Allocation of kernel memory for hash table. If someone already
	 * allocated it, we can reuse it.
	 */
	SpinLockAcquire(&hjtable->lock);
	if (hjtable->n_kernel == 0)
	{
		Assert(!hjtable->m_hash && !hjtable->ev_hash);

		hjtable->dindex = pgstrom_opencl_device_schedule(&ghjoin->msg);
		clghj->dindex = hjtable->dindex;
		clghj->kcmdq = opencl_cmdq[clghj->dindex];
		clghj->m_hash = clCreateBuffer(opencl_context,
									   CL_MEM_READ_WRITE,
									   hjtable->kern.length,
									   NULL,
									   &rc);
		if (rc != CL_SUCCESS)
		{
			SpinLockRelease(&hjtable->lock);
			goto error;
		}

		rc = clEnqueueWriteBuffer(clghj->kcmdq,
								  clghj->m_hash,
								  CL_FALSE,
								  0,
								  hjtable->kern.length,
								  &hjtable->kern,
								  0,
								  NULL,
								  &clghj->events[0]);
		if (rc != CL_SUCCESS)
		{
			rc = clReleaseMemObject(clghj->m_hash);
			SpinLockRelease(&hjtable->lock);
			goto error;
		}
		clghj->ev_index++;
		hjtable->m_hash = clghj->m_hash;
		hjtable->ev_hash = clghj->events[0];
	}
	else
	{
		Assert(hjtable->m_hash && hjtable->ev_hash);
		rc = clRetainMemObject(hjtable->m_hash);
		Assert(rc == CL_SUCCESS);
		rc = clRetainEvent(hjtable->ev_hash);
		Assert(rc == CL_SUCCESS);

		clghj->dindex = hjtable->dindex;
		clghj->kcmdq = opencl_cmdq[clghj->dindex];
		clghj->m_hash = hjtable->m_hash;
		clghj->events[0] = hjtable->ev_hash;
	}
	hjtable->n_kernel++;
	SpinLockRelease(&hjtable->lock);

	return clghj;

error:
	if (clghj)
	{
		if (clghj->program && clghj->program != BAD_OPENCL_PROGRAM)
			clReleaseProgram(clghj->program);
		free(clghj);
	}
	ghjoin->msg.errcode = rc;
	pgstrom_reply_message(&ghjoin->msg);
	return NULL;
}


static void
clserv_process_gpuhashjoin_column(pgstrom_gpuhashjoin *ghjoin,
								  tcache_column_store *tcs)
{
	clstate_gpuhashjoin *clghj;
	kern_parambuf  *kparams;
	kern_resultbuf *kresults;
	kern_column_store *kcs_head;
	kern_toastbuf *ktoast_head;
	cl_int		nitems;
	cl_int	   *rindex;
	size_t		offset;
	size_t		length;
	size_t		gwork_sz;
	size_t		lwork_sz;
	cl_char	   *attrefs;
	cl_int		i, k;
	cl_int		rc;

	clghj = clserv_process_gpuhashjoin_common(ghjoin);
	if (!clghj)
		return;

	/*
	 * find out param-buffer and result-buffer
	 */
	kparams = KERN_HASHJOIN_PARAMBUF(ghjoin->kern);
	kresults = KERN_HASHJOIN_RESULTBUF(ghjoin->kern);
	attrefs = KPARAM_GET_ATTREFS(kparams);
	kcs_head = KPARAM_GET_KCS_HEAD(kparams);
	ktoast_head = KPARAM_GET_KTOAST_HEAD(kparams);

#if 0
	clserv_log("kcs_head {length=%u ncols=%u nrows=%u nrooms=%u}",
			   kcs_head->length, kcs_head->ncols,
			   kcs_head->nrows, kcs_head->nrooms);
	for (i=0; i < kcs_head->ncols; i++)
	{
		kern_colmeta colmeta = kcs_head->colmeta[i];

		clserv_log("meta[%d] {attnotnull=%d attalign=%d attlen=%d cs_ofs=%u}",
				   i, colmeta.attnotnull, colmeta.attalign,
				   colmeta.attlen, colmeta.cs_ofs);
	}
#endif

	/*
	 * number of rows to be processed. If rindex is not given,
	 * we assume all the items on column-store are valid.
	 * Elsewhere, each item on rindex point valid rows.
	 */
	if (ghjoin->src_nitems < 0)
	{
		nitems = tcs->nrows;
		rindex = NULL;
	}
	else
	{
		nitems = ghjoin->src_nitems;
		rindex = kresults->results;
	}

	/*
	 * __kernel void
	 * gpuhashjoin_inner_cs(__global kern_hashjoin *khashjoin,
	 *                      __global kern_hashtable *khashtbl,
	 *                      __global kern_column_store *kcs,
	 *                      __global kern_toastbuf *toast,
	 *                      __local void *local_workmem)
	 */
	clghj->kernel = clCreateKernel(clghj->program,
								   "gpuhashjoin_inner_cs",
								   &rc);
	if (rc != CL_SUCCESS)
		goto error_sync;

	/*
	 * also, compute an optimal workgroup-size of this kernel
	 */
	if (!clserv_compute_workgroup_size(&gwork_sz, &lwork_sz,
									   clghj->kernel,
									   clghj->dindex,
									   nitems,
									   sizeof(cl_uint)))
		goto error_sync;

	/* buffer object of __global kern_hashjoin *khashjoin */
	length = (KERN_HASHJOIN_PARAMBUF_LENGTH(ghjoin->kern) +
			  KERN_HASHJOIN_RESULTBUF_LENGTH(ghjoin->kern));
	if (rindex)
		length += sizeof(cl_uint) * nitems;
	clghj->m_join = clCreateBuffer(opencl_context,
								   CL_MEM_READ_WRITE,
								   length,
								   NULL,
								   &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}

	/* buffer object of __global kern_column_store *kcs */
	kcs_head = KPARAM_GET_KCS_HEAD(kparams);
	length = offsetof(kern_column_store, colmeta[kcs_head->ncols]);
	clghj->m_cstore = clCreateBuffer(opencl_context,
									 CL_MEM_READ_WRITE,
									 kcs_head->length,
									 NULL,
									 &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error_sync;
    }

	/* buffer object of __global kern_toastbuf *ktoast */
	ktoast_head = KPARAM_GET_KTOAST_HEAD(kparams);
	if (ktoast_head)
	{
		clghj->m_toast = clCreateBuffer(opencl_context,
										CL_MEM_READ_WRITE,
										ktoast_head->length,
										NULL,
										&rc);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
			goto error_sync;
		}
	}

	/*
	 * OK, all the device memory and kernel objects are successfully
	 * constructed. Let's enqueue kernel invocation.
	 */
	rc = clSetKernelArg(clghj->kernel,
						0,	/* __global kern_hashjoin *khashjoin */
						sizeof(cl_mem),
						&clghj->m_join);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_sync;
	}

	rc = clSetKernelArg(clghj->kernel,
						1,	/* __global kern_hashtable *khashtbl */
						sizeof(cl_mem),
						&clghj->m_hash);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_sync;
	}

	rc = clSetKernelArg(clghj->kernel,
						2,	/* __global kern_column_store *kcs */
						sizeof(cl_mem),
						&clghj->m_cstore);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_sync;
	}

	rc = clSetKernelArg(clghj->kernel,
						3,	/* __global kern_toastbuf *toast */
						sizeof(cl_mem),
						&clghj->m_toast);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_sync;
	}

	/* negative nitems means all the rows are valid without rindex,
	 * elsewhere, it means number of rindex being processed. */
	if (!rindex)
		nitems = -1;
	rc = clSetKernelArg(clghj->kernel,
						4,	/* cl_int kcs_nitems */
						sizeof(cl_int),
						&nitems);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_sync;
	}

	rc = clSetKernelArg(clghj->kernel,
						5,	/* __local void *local_workmem */
						sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_sync;
	}

	/* DMA send of kern_hashjoin */
	length = KERN_HASHJOIN_DMA_SENDLEN(ghjoin->kern);
	rc = clEnqueueWriteBuffer(clghj->kcmdq,
							  clghj->m_join,
							  CL_FALSE,
							  0,
							  length,
							  KERN_HASHJOIN_PARAMBUF(ghjoin->kern),
							  0,
							  NULL,
							  &clghj->events[clghj->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}
	clghj->ev_index++;
	ghjoin->msg.pfm.bytes_dma_send += length;
	ghjoin->msg.pfm.num_dma_send++;

	/* DMA send of kern_column_store head */
	length = offsetof(kern_column_store, colmeta[kcs_head->ncols]);
	rc = clEnqueueWriteBuffer(clghj->kcmdq,
							  clghj->m_cstore,
							  CL_FALSE,
							  0,
							  length,
							  kcs_head,
							  0,
							  NULL,
							  &clghj->events[clghj->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}
	clghj->ev_index++;
	ghjoin->msg.pfm.bytes_dma_send += length;
	ghjoin->msg.pfm.num_dma_send++;

	/* DMA send of kern_toastbuf head */
	if (ktoast_head)
	{
		length = offsetof(kern_toastbuf, coldir[ktoast_head->ncols]);
		rc = clEnqueueWriteBuffer(clghj->kcmdq,
								  clghj->m_toast,
								  CL_FALSE,
								  0,
								  length,
								  ktoast_head,
								  0,
								  NULL,
								  &clghj->events[clghj->ev_index]);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clEnqueueWriteBuffer: %s",
					   opencl_strerror(rc));
			goto error_sync;
		}
		clghj->ev_index++;
		ghjoin->msg.pfm.bytes_dma_send += length;
		ghjoin->msg.pfm.num_dma_send++;
	}

	/*
	 * DMA send of kern_column_store / kern_toastbuf contents
	 */
	attrefs = KPARAM_GET_ATTREFS(kparams);
	for (i=0, k=0; i < tcs->ncols; i++)
	{
		if (!attrefs[i])
			continue;
		offset = kcs_head->colmeta[k].cs_ofs;
		if (!kcs_head->colmeta[k].attnotnull)
		{
			length = (kcs_head->nrooms + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
			rc = clEnqueueWriteBuffer(clghj->kcmdq,
									  clghj->m_cstore,
									  CL_FALSE,
									  offset,
									  length,
									  tcs->cdata[i].isnull,
									  0,
									  NULL,
									  &clghj->events[clghj->ev_index]);
			if (rc != CL_SUCCESS)
			{
				clserv_log("failed on clEnqueueWriteBuffer: %s",
						   opencl_strerror(rc));
				Assert(false);
				goto error_sync;
			}
			clghj->ev_index++;
			ghjoin->msg.pfm.bytes_dma_send += length;
			ghjoin->msg.pfm.num_dma_send++;

			offset += length;
		}
		length = (kcs_head->colmeta[k].attlen > 0
				  ? kcs_head->colmeta[k].attlen
				  : sizeof(cl_uint)) * kcs_head->nrooms;
		rc = clEnqueueWriteBuffer(clghj->kcmdq,
								  clghj->m_cstore,
								  CL_FALSE,
								  offset,
								  length,
								  tcs->cdata[i].values,
								  0,
								  NULL,
								  &clghj->events[clghj->ev_index]);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clEnqueueWriteBuffer: %s",
					   opencl_strerror(rc));
			goto error_sync;
		}
		clghj->ev_index++;
		ghjoin->msg.pfm.bytes_dma_send += length;
		ghjoin->msg.pfm.num_dma_send++;

		if (tcs->cdata[i].toast)
		{
			Assert(ktoast_head != NULL);
			rc = clEnqueueWriteBuffer(clghj->kcmdq,
									  clghj->m_toast,
									  CL_FALSE,
									  ktoast_head->coldir[k],
									  tcs->cdata[i].toast->tbuf_usage,
									  tcs->cdata[i].toast,
									  0,
									  NULL,
									  &clghj->events[clghj->ev_index]);
			if (rc != CL_SUCCESS)
			{
				clserv_log("failed on clEnqueueWriteBuffer: %s",
						   opencl_strerror(rc));
				Assert(false);
				goto error_sync;
			}
			clghj->ev_index++;
			ghjoin->msg.pfm.bytes_dma_send += length;
			ghjoin->msg.pfm.num_dma_send++;
		}
		k++;
	}

	/*
	 * kick gpuhashjoin_inner_cs()
	 */
	rc = clEnqueueNDRangeKernel(clghj->kcmdq,
								clghj->kernel,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								clghj->ev_index,
								&clghj->events[0],
								&clghj->events[clghj->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		goto error_sync;
	}
	clghj->ev_index++;

	/*
	 * write back the result-buffer
	 */
	offset = KERN_HASHJOIN_DMA_RECVOFS(ghjoin->kern);
	length = KERN_HASHJOIN_DMA_RECVLEN(ghjoin->kern);
	rc = clEnqueueReadBuffer(clghj->kcmdq,
							 clghj->m_join,
							 CL_FALSE,
							 offset,
							 length,
							 kresults,
							 1,
							 &clghj->events[clghj->ev_index - 1],
							 &clghj->events[clghj->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueReadBuffer: %s",
				   opencl_strerror(rc));
		goto error_sync;
	}
	clghj->ev_index++;
	ghjoin->msg.pfm.bytes_dma_recv += length;
	ghjoin->msg.pfm.num_dma_recv++;

	/*
	 * Last, registers a callback to handle post join process; that generate
	 * a pseudo scan relation
	 */
	rc = clSetEventCallback(clghj->events[clghj->ev_index - 1],
							CL_COMPLETE,
							clserv_respond_hashjoin,
							clghj);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetEventCallback: %s", opencl_strerror(rc));
		goto error_sync;
	}
	return;

error_sync:
	if (clghj->ev_index > 0)
		clWaitForEvents(clghj->ev_index, clghj->events);
	ghjoin->msg.errcode = rc;
	clserv_respond_hashjoin(NULL, CL_COMPLETE, clghj);
}

static void
clserv_process_gpuhashjoin_row(pgstrom_gpuhashjoin *ghjoin,
							   tcache_row_store *trs)
{
	clstate_gpuhashjoin *clghj;
	kern_resultbuf *kresults;
	kern_parambuf *kparams;
	kern_column_store *kcs_head;
	cl_int		nitems;
	cl_int	   *rindex;
	size_t		offset;
	size_t		length;
	size_t		gwork_sz;
	size_t		lwork_sz;
	cl_int		rc;

	clghj = clserv_process_gpuhashjoin_common(ghjoin);
	if (!clghj)
		return;

	kparams = KERN_HASHJOIN_PARAMBUF(ghjoin->kern);
	kresults = KERN_HASHJOIN_RESULTBUF(ghjoin->kern);
	kcs_head = KPARAM_GET_KCS_HEAD(kparams);

	/*
	 * number of rows to be processed. If rindex is not given,
	 * we assume all the items on column-store are valid.
	 * Elsewhere, each item on rindex point valid rows.
	 */
	if (ghjoin->src_nitems < 0)
	{
		nitems = trs->kern.nrows;
		rindex = NULL;
	}
	else
	{
		nitems = ghjoin->src_nitems;
		rindex = kresults->results;
	}

	/*
	 * __kernel void
	 * gpuhashjoin_inner_rs(__global kern_hashjoin *khashjoin,
	 *                      __global kern_hashtable *khashtbl,
	 *                      __global kern_row_store *krs,
	 *                      __global kern_column_store *kcs,
	 *                      cl_int   krs_nitems,
	 *                      __local void *local_workmem)
	 */
	clghj->kernel = clCreateKernel(clghj->program,
								   "gpuhashjoin_inner_cs",
								   &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		goto error_sync;
	}

	/*
	 * also, compute an optimal workgroup-size of this kernel
	 */
	if (!clserv_compute_workgroup_size(&gwork_sz, &lwork_sz,
									   clghj->kernel,
									   clghj->dindex,
									   nitems,
									   sizeof(cl_uint)))
		goto error_sync;

	/* buffer object of __global kern_hashjoin *khashjoin */
	length = (KERN_HASHJOIN_PARAMBUF_LENGTH(ghjoin->kern) +
			  KERN_HASHJOIN_RESULTBUF_LENGTH(ghjoin->kern));
	if (rindex)
		length += sizeof(cl_uint) * nitems;
	clghj->m_join = clCreateBuffer(opencl_context,
								   CL_MEM_READ_WRITE,
								   length,
								   NULL,
								   &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}

	/* buffer object of __global kern_row_store *krs */
	clghj->m_rstore = clCreateBuffer(opencl_context,
									 CL_MEM_READ_WRITE,
									 trs->kern.length,
									 NULL,
									 &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}

	/* buffer object of __global kern_column_store *kcs */
	clghj->m_cstore = clCreateBuffer(opencl_context,
									 CL_MEM_READ_WRITE,
									 kcs_head->length,
									 NULL,
									 &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}

	/*
	 * OK, all the device memory and kernel objects are successfully
	 * constructed. Let's enqueue kernel invocation.
	 */
	rc = clSetKernelArg(clghj->kernel,
						0,	/* __global kern_hashjoin *khashjoin */
						sizeof(cl_mem),
						&clghj->m_join);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_sync;
	}

	rc = clSetKernelArg(clghj->kernel,
						1,	/* __global kern_hashtable *khashtbl */
						sizeof(cl_mem),
						&clghj->m_hash);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_sync;
	}

	rc = clSetKernelArg(clghj->kernel,
						2,	/* __global kern_row_store *krs */
						sizeof(cl_mem),
						&clghj->m_rstore);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_sync;
	}

	rc = clSetKernelArg(clghj->kernel,
						3,	/* __global kern_column_store *kcs */
						sizeof(cl_mem),
						&clghj->m_cstore);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_sync;
	}

	if (!rindex)
		nitems = -1;	/* informs rindex is not valid */
	rc = clSetKernelArg(clghj->kernel,
						4,	/* cl_int krs_nitems */
						sizeof(cl_int),
						&nitems);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_sync;
	}

	rc = clSetKernelArg(clghj->kernel,
						5,	/* __local void *local_workmem */
						sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_sync;
	}

	/* DMA kick of kern_hashjoin */
	offset = KERN_HASHJOIN_DMA_SENDOFS(ghjoin->kern);
	length = KERN_HASHJOIN_DMA_SENDLEN(ghjoin->kern);
	rc = clEnqueueWriteBuffer(clghj->kcmdq,
							  clghj->m_join,
							  CL_FALSE,
							  offset,
							  length,
							  KERN_HASHJOIN_PARAMBUF(ghjoin->kern),
							  0,
							  NULL,
							  &clghj->events[clghj->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}
	clghj->ev_index++;
	ghjoin->msg.pfm.bytes_dma_send += length;
	ghjoin->msg.pfm.num_dma_send++;

	/* DMA kick of kern_row_store */
	rc = clEnqueueWriteBuffer(clghj->kcmdq,
							  clghj->m_rstore,
							  CL_FALSE,
							  0,
							  trs->kern.length,
							  &trs->kern,
							  0,
							  NULL,
							  &clghj->events[clghj->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}
	clghj->ev_index++;
	ghjoin->msg.pfm.bytes_dma_send += trs->kern.length;
	ghjoin->msg.pfm.num_dma_send++;

	/* DMA send of head of kern_column_store */
	length = offsetof(kern_column_store, colmeta[kcs_head->ncols]);
	rc = clEnqueueWriteBuffer(clghj->kcmdq,
							  clghj->m_cstore,
							  CL_FALSE,
							  0,
							  length,
							  kcs_head,
							  0,
							  NULL,
							  &clghj->events[clghj->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}
	clghj->ev_index++;
	ghjoin->msg.pfm.bytes_dma_send += length;
	ghjoin->msg.pfm.num_dma_send++;

	/*
	 * kick gpuhashjoin_inner_rs()
	 */
	rc = clEnqueueNDRangeKernel(clghj->kcmdq,
								clghj->kernel,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								clghj->ev_index,
								&clghj->events[0],
								&clghj->events[clghj->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		goto error_sync;
	}
	clghj->ev_index++;

	/*
	 * write back the result-buffer
	 */
	kresults = KERN_HASHJOIN_RESULTBUF(ghjoin->kern);
	offset = ((uintptr_t)KERN_HASHJOIN_RESULTBUF(ghjoin->kern) -
			  (uintptr_t)KERN_HASHJOIN_PARAMBUF(ghjoin->kern));
	length = KERN_HASHJOIN_DMA_RECVLEN(ghjoin->kern);
	rc = clEnqueueReadBuffer(clghj->kcmdq,
							 clghj->m_join,
							 CL_FALSE,
							 offset,
							 length,
							 kresults,
							 1,
							 &clghj->events[clghj->ev_index - 1],
							 &clghj->events[clghj->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueReadBuffer: %s",
				   opencl_strerror(rc));
		goto error_sync;
	}
	clghj->ev_index++;
	ghjoin->msg.pfm.bytes_dma_recv += length;
	ghjoin->msg.pfm.num_dma_recv++;

	/*
	 * Last, registers a callback to handle post join process; that generate
	 * a pseudo scan relation
	 */
	rc = clSetEventCallback(clghj->events[clghj->ev_index - 1],
							CL_COMPLETE,
							clserv_respond_hashjoin,
							clghj);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetEventCallback: %s", opencl_strerror(rc));
		goto error_sync;
	}
	return;

error_sync:
	if (clghj->ev_index > 0)
		clWaitForEvents(clghj->ev_index, clghj->events);
	ghjoin->msg.errcode = rc;
	clserv_respond_hashjoin(NULL, CL_COMPLETE, clghj);
}

static cl_int
clserv_projection_from_column(tcache_column_store *dst_tcs, cl_uint dindex,
							  tcache_column_store *src_tcs, cl_uint sindex,
							  kern_projection *kproj, bool is_outer)
{
	AttrNumber	resno;
	int			i;
	char	   *datum;

	for (i=0; i < kproj->ncols; i++)
	{
		if ((!is_outer && kproj->origins[i].is_outer) ||
			(is_outer && !kproj->origins[i].is_outer))
			continue;

		/* fetch value from the source column-store */
		resno = kproj->origins[i].resno;
		if (resno >= src_tcs->ncols)
			datum = NULL;	/* attribute is out of column-store */
		else if (!src_tcs->cdata[resno-1].values)
			datum = NULL;	/* attribute is not cached (bug?) */
		else if (src_tcs->cdata[resno-1].isnull &&
				 att_isnull(sindex, src_tcs->cdata[resno-1].isnull))
			datum = NULL;	/* it's a null value */
		else if (kproj->origins[i].colmeta.attlen > 0)
		{
			Assert(!src_tcs->cdata[resno-1].toast);
			datum = ((char *) src_tcs->cdata[resno-1].values +
					 kproj->origins[i].colmeta.attlen * dindex);
		}
		else
		{
			cl_uint	vl_ofs;

			Assert(src_tcs->cdata[resno-1].toast);
			vl_ofs = ((cl_uint *)src_tcs->cdata[resno-1].values)[sindex];
			Assert(vl_ofs > 0);
			datum = ((char *) src_tcs->cdata[resno-1].toast + vl_ofs);
		}

		/* put value on the destination column-store */
		Assert(i < dst_tcs->ncols);
		if (!datum)
		{
			Assert(dst_tcs->cdata[i].isnull != NULL);
			dst_tcs->cdata[i].isnull[sindex >> 3] &= ~(1 << (sindex & 7));
		}
		else
		{
			if (dst_tcs->cdata[i].isnull)
				dst_tcs->cdata[i].isnull[sindex >> 3] |= (1 << (sindex & 7));
			if (kproj->origins[i].colmeta.attlen > 0)
			{
				int		attlen = kproj->origins[i].colmeta.attlen;

				memcpy(dst_tcs->cdata[i].values + attlen * dindex,
					   datum,
					   attlen);
			}
			else
			{
				Size    vl_len = VARSIZE_ANY(datum);
				tcache_toastbuf *tbuf = dst_tcs->cdata[i].toast;

				Assert(tbuf);
				/* expand toast buffer on demand */
				if (tbuf->tbuf_length < tbuf->tbuf_usage + INTALIGN(vl_len))
				{
					tcache_toastbuf *new_tbuf
						= pgstrom_expand_toast_buffer(tbuf);
					if (!new_tbuf)
						return StromError_OutOfSharedMemory;
					pgstrom_put_toast_buffer(tbuf);
					dst_tcs->cdata[i].toast = tbuf = new_tbuf;
				}
				((cl_uint *)dst_tcs->cdata[i].values)[dindex]
					= tbuf->tbuf_usage;
				memcpy((char *)tbuf + tbuf->tbuf_usage,
					   datum,
					   vl_len);
				tbuf->tbuf_usage += INTALIGN(vl_len);
			}
		}
	}
	return StromError_Success;
}

static cl_int
clserv_projection_from_row(tcache_column_store *dst_tcs, cl_uint dindex,
						   tcache_row_store *src_trs, cl_uint sindex,
						   kern_projection *kproj, bool is_outer)
{
	rs_tuple   *rs_tup;
	char	  **values;
	int			i, natts;
	Size		offset;

	Assert(sindex < src_trs->kern.nrows);
	rs_tup = kern_rowstore_get_tuple(&src_trs->kern, sindex);
	if (!rs_tup)
		return StromError_Success;	/* fill by NULL; nothing to do */

	natts = HeapTupleHeaderGetNatts(&rs_tup->data);
	values = calloc(natts, sizeof(char *));
	if (!values)
		return StromError_OutOfMemory;

	/* extract row-format according to the column-metadata */
	offset = rs_tup->data.t_hoff;
	for (i=0; i < natts; i++)
	{
		kern_colmeta	rcmeta = src_trs->kern.colmeta[i];

		if ((rs_tup->data.t_infomask & HEAP_HASNULL) != 0 &&
			att_isnull(i, rs_tup->data.t_bits))
		{
			values[i] = NULL;
			continue;
		}

		if (rcmeta.attlen > 0)
			offset = TYPEALIGN(rcmeta.attalign, offset);
		else if (!VARATT_NOT_PAD_BYTE((uintptr_t)&rs_tup->data + offset))
			offset = TYPEALIGN(rcmeta.attalign, offset);
		values[i] = ((char *)&rs_tup->data) + offset;

		offset += (rcmeta.attlen > 0
				   ? rcmeta.attlen
				   : VARSIZE_ANY(values[i]));
	}

	/* do simple projection */
	for (i=0; i < kproj->ncols; i++)
	{
		AttrNumber	resno;
		char	   *datum;

		if ((!is_outer && kproj->origins[i].is_outer) ||
			(is_outer && !kproj->origins[i].is_outer))
			continue;

		resno = kproj->origins[i].resno;
		if (resno > natts)
			datum = NULL;
		else
			datum = values[resno - 1];

		if (i >= dst_tcs->ncols || !dst_tcs->cdata[i].values)
		{
			free(values);
			clserv_log("destination column is out of range");
			return StromError_DataStoreCorruption;
		}
		if (!datum)
		{
			if (!dst_tcs->cdata[i].isnull)
			{
				free(values);
				clserv_log("null value appeared in not-null column");
				return StromError_DataStoreCorruption;
			}
			dst_tcs->cdata[i].isnull[dindex >> 3] &= ~(1 << (dindex & 7));
		}
		else
		{
			int		attlen = kproj->origins[i].colmeta.attlen;

			dst_tcs->cdata[i].isnull[dindex >> 3] |= (1 << (dindex & 7));

			if (kproj->origins[i].colmeta.attlen > 0)
			{
				memcpy(dst_tcs->cdata[i].values + attlen * dindex,
					   datum,
					   attlen);
			}
			else
			{
				Size	vl_len = VARSIZE_ANY(datum);
				tcache_toastbuf *tbuf = dst_tcs->cdata[i].toast;

				Assert(tbuf);
				/* expand toast buffer on demand */
				if (tbuf->tbuf_length < tbuf->tbuf_usage + INTALIGN(vl_len))
				{
					tcache_toastbuf *new_tbuf
						= pgstrom_expand_toast_buffer(tbuf);
					if (!new_tbuf)
					{
						free(values);
						return StromError_OutOfSharedMemory;
					}
					pgstrom_put_toast_buffer(tbuf);
					dst_tcs->cdata[i].toast = tbuf = new_tbuf;
				}
				((cl_uint *)dst_tcs->cdata[i].values)[dindex]
					= tbuf->tbuf_usage;
				memcpy((char *)tbuf + tbuf->tbuf_usage,
					   datum,
					   vl_len);
				tbuf->tbuf_usage += INTALIGN(vl_len);
			}
		}
	}
	free(values);
	return StromError_Success;
}

static void
clserv_process_post_gpuhashjoin(pgstrom_gpuhashjoin *ghjoin)
{
	pgstrom_hashjoin_table *hjtable = ghjoin->hjtable;
	kern_parambuf	   *kparams = KERN_HASHJOIN_PARAMBUF(ghjoin->kern);
	kern_resultbuf	   *kresults = KERN_HASHJOIN_RESULTBUF(ghjoin->kern);
	kern_projection	   *kproj = KPARAM_GET_KPROJECTION(kparams);
	StromObject		   *outer_rcs = ghjoin->rcs_src;
	StromObject		   *inner_rcs;
	tcache_column_store *dst_tcs;
	cl_int				rc = StromError_Success;
	int					i;

	dst_tcs = pgstrom_create_column_store_with_projection(kproj,
														  kresults->nitems,
														  false);
	if (!dst_tcs)
	{
		rc = StromError_OutOfSharedMemory;
		goto error;
	}

	for (i=0; i < kresults->nitems; i++)
	{
		cl_uint	rcs_index = kresults->results[3 * i];
		cl_uint	inner_idx = kresults->results[3 * i + 1];
		cl_uint	outer_idx = kresults->results[3 * i + 2];

		/* fill up inner columns */
		Assert(rcs_index < hjtable->num_rcs);
		inner_rcs = hjtable->rcstore[rcs_index];

		if (StromTagIs(inner_rcs, TCacheRowStore))
		{
			tcache_row_store *src_trs = (tcache_row_store *) inner_rcs;
			rc = clserv_projection_from_row(dst_tcs, i,
											src_trs, inner_idx,
											kproj, false);
			if (rc != StromError_Success)
				goto error;
		}
		else if (StromTagIs(inner_rcs, TCacheColumnStore))
		{
			tcache_column_store *src_tcs = (tcache_column_store *) inner_rcs;
			rc = clserv_projection_from_column(dst_tcs, i,
											   src_tcs, inner_idx,
											   kproj, false);
			if (rc != StromError_Success)
				goto error;
		}
		else
		{
			rc = StromError_DataStoreCorruption;
			goto error;
		}

		/* fill up outer columns */
		if (StromTagIs(outer_rcs, TCacheRowStore))
		{
			tcache_row_store *src_trs = (tcache_row_store *) outer_rcs;
			rc = clserv_projection_from_row(dst_tcs, i,
											src_trs, outer_idx,
											kproj, true);
			if (rc != StromError_Success)
				goto error;
		}
		else if (StromTagIs(outer_rcs, TCacheColumnStore))
		{
			tcache_column_store *src_tcs = (tcache_column_store *) outer_rcs;
			rc = clserv_projection_from_column(dst_tcs, i,
											   src_tcs, outer_idx,
											   kproj, true);
			if (rc != StromError_Success)
				goto error;
		}
		else
		{
			rc = StromError_DataStoreCorruption;
			goto error;
		}
	}
	ghjoin->rcs_dst = &dst_tcs->sobj;

error:
	if (rc != StromError_Success && dst_tcs != NULL)
		pgstrom_put_column_store(dst_tcs);
	ghjoin->msg.errcode = rc;
	pgstrom_reply_message(&ghjoin->msg);
}

static void
clserv_process_gpuhashjoin(pgstrom_message *message)
{
	pgstrom_gpuhashjoin *ghjoin = (pgstrom_gpuhashjoin *) message;

	if (ghjoin->hashjoin_done)
	{
		clserv_process_post_gpuhashjoin(ghjoin);
	}
	else if (StromTagIs(ghjoin->rcs_src, TCacheRowStore))
	{
		tcache_row_store *trs = (tcache_row_store *) ghjoin->rcs_src;
		clserv_process_gpuhashjoin_row(ghjoin, trs);
	}
	else if (StromTagIs(ghjoin->rcs_src, TCacheColumnStore))
	{
		tcache_column_store *tcs = (tcache_column_store *) ghjoin->rcs_src;
		clserv_process_gpuhashjoin_column(ghjoin, tcs);
	}
	else
	{
		message->errcode = StromError_BadRequestMessage;
		pgstrom_reply_message(message);
	}
}
