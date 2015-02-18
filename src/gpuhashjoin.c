/*
 * gpuhashjoin.c
 *
 * Hash-Join acceleration by GPU processors
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
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

#include "access/sysattr.h"
#include "catalog/pg_type.h"
#include "common/pg_crc.h"
#include "miscadmin.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "nodes/relation.h"
#include "nodes/plannodes.h"
#include "optimizer/clauses.h"
#include "optimizer/cost.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/planmain.h"
#include "optimizer/restrictinfo.h"
#include "optimizer/subselect.h"
#include "optimizer/tlist.h"
#include "optimizer/var.h"
#include "parser/parsetree.h"
#include "parser/parse_coerce.h"
#include "storage/ipc.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/ruleutils.h"
#include "utils/selfuncs.h"
#include "pg_strom.h"
#include "opencl_hashjoin.h"

/* static variables */
static set_join_pathlist_hook_type set_join_pathlist_next;
static CustomPathMethods		gpuhashjoin_path_methods;
static CustomScanMethods		gpuhashjoin_plan_methods;
static CustomScanMethods		multihash_plan_methods;
static PGStromExecMethods		gpuhashjoin_exec_methods;
static PGStromExecMethods		multihash_exec_methods;
static bool						enable_gpuhashjoin;

/*
 *                              (depth=0)
 * [GpuHashJoin] ---<outer>--- [relation scan to be joined]
 *    |
 * <inner>
 *    |    (depth=1)
 *    +-- [MultiHash] ---<outer>--- [relation scan to be hashed]
 *           |
 *        <inner>
 *           |    (depth=2)
 *           +-- [MultiHash] ---<outer>--- [relation scan to be hashed]
 *
 * The diagram above shows structure of GpuHashJoin which can have a hash-
 * table that contains multiple inner scans. GpuHashJoin always takes a
 * MultiHash node as inner relation to join it with outer relation, then
 * materialize them into a single pseudo relation view. A MultiHash node
 * has an outer relation to be hashed, and can optionally have another
 * MultiHash node to put multiple inner (small) relations on a hash-table.
 * A smallest set of GpuHashJoin is consists of an outer relation and
 * an inner MultiHash node. When third relation is added, it inject the
 * third relation on the inner-tree of GpuHashJoin. So, it means the
 * deepest MultiHash is the first relation to be joined with outer
 * relation, then second deepest one shall be joined, in case when order
 * of join needs to be paid attention.
 */
typedef struct
{
	CustomPath		cpath;
	Path		   *outer_path;	/* outer path (always one) */
	Plan		   *outer_plan;
	Size			hashtable_size;	/* estimated hashtable size */
	double			row_population_ratio;	/* estimated population ratio */
	int				num_rels;	/* number of inner relations */
	struct {
		Path	   *scan_path;
		Plan	   *scan_plan;
		JoinType	jointype;
		List	   *hash_clauses;
		List	   *qual_clauses;
		List	   *host_clauses;
		double		threshold;
		Size		chunk_size;
		cl_uint		nslots;
		int			nloops;
	} inners[FLEXIBLE_ARRAY_MEMBER];
} GpuHashJoinPath;

/*
 * GpuHashJoinInfo - state object of CustomScan(GpuHashJoin)
 */
typedef struct
{
	int			num_rels;		/* number of underlying MultiHash */
	Size		hashtable_size;	/* estimated hashtable size */
	double		row_population_ratio;	/* estimated population ratio */
	char	   *kernel_source;
	int			extra_flags;
	bool		outer_bulkload;	/* is outer can bulk loading? */
	Expr	   *outer_quals;	/* Var has raw depth/resno pair */
	List	   *join_types;		/* list of join types */
	List	   *hash_keys;		/* Var has raw depth/resno pair */
	List	   *hash_clauses;	/* Var has raw depth/resno pair */
	List	   *qual_clauses;	/* Var has raw depth/resno pair */
	List	   *host_clauses;	/* Var is mapped on custom_ps_tlist */
	List	   *used_params;	/* template for kparams */
	/* supplemental information for ps_tlist */
	List	   *ps_src_depth;	/* source depth of the pseudo target entry */
	List	   *ps_src_resno;	/* source resno of the pseudo target entry */
} GpuHashJoinInfo;

static inline void
form_gpuhashjoin_info(CustomScan *cscan, GpuHashJoinInfo *ghj_info)
{
	List   *privs = NIL;
	List   *exprs = NIL;
	union {
        long    ival;
        double  fval;
    } datum;

	privs = lappend(privs, makeInteger(ghj_info->num_rels));
	privs = lappend(privs, makeInteger(ghj_info->hashtable_size));
	datum.fval = ghj_info->row_population_ratio;
	privs = lappend(privs, makeInteger(datum.ival));
	privs = lappend(privs, makeString(ghj_info->kernel_source));
	privs = lappend(privs, makeInteger(ghj_info->extra_flags));
	privs = lappend(privs, makeInteger(ghj_info->outer_bulkload));
	privs = lappend(privs, ghj_info->outer_quals);
	privs = lappend(privs, ghj_info->join_types);
	privs = lappend(privs, ghj_info->hash_keys);
	privs = lappend(privs, ghj_info->hash_clauses);
	privs = lappend(privs, ghj_info->qual_clauses);
	exprs = lappend(exprs, ghj_info->host_clauses);
	exprs = lappend(exprs, ghj_info->used_params);
	privs = lappend(privs, ghj_info->ps_src_depth);
	privs = lappend(privs, ghj_info->ps_src_resno);

	cscan->custom_private = privs;
	cscan->custom_exprs = exprs;
}

static inline GpuHashJoinInfo *
deform_gpuhashjoin_info(CustomScan *cscan)
{
	GpuHashJoinInfo *ghj_info = palloc0(sizeof(GpuHashJoinInfo));
	List	   *privs = cscan->custom_private;
	List	   *exprs = cscan->custom_exprs;
	int			pindex = 0;
	int			eindex = 0;
	union {
        long    ival;
        double  fval;
    } datum;

	ghj_info->num_rels        = intVal(list_nth(privs, pindex++));
	ghj_info->hashtable_size  = intVal(list_nth(privs, pindex++));
	datum.ival = intVal(list_nth(privs, pindex++));
	ghj_info->row_population_ratio = datum.fval;
	ghj_info->kernel_source   = strVal(list_nth(privs, pindex++));
	ghj_info->extra_flags     = intVal(list_nth(privs, pindex++));
	ghj_info->outer_bulkload  = intVal(list_nth(privs, pindex++));
	ghj_info->outer_quals     = list_nth(privs, pindex++);
	ghj_info->join_types      = list_nth(privs, pindex++);
	ghj_info->hash_keys       = list_nth(privs, pindex++);
	ghj_info->hash_clauses    = list_nth(privs, pindex++);
	ghj_info->qual_clauses    = list_nth(privs, pindex++);
	ghj_info->host_clauses    = list_nth(exprs, eindex++);
	ghj_info->used_params     = list_nth(exprs, eindex++);
	ghj_info->ps_src_depth    = list_nth(privs, pindex++);
	ghj_info->ps_src_resno    = list_nth(privs, pindex++);

	return ghj_info;
}

/*
 * MultiHashInfo - state object of CustomScan(MultiHash)
 */
typedef struct
{
	/*
	 * outerPlan ... relation to be hashed
	 * innerPlan ... one another MultiHash, if any
	 */
	int			depth;		/* depth of this hash table */

	cl_uint		nslots;		/* width of hash slots */
	cl_uint		nloops;		/* expected number of batches */
	Size		hashtable_size;
	double		threshold;

	/*
	 * NOTE: setrefs.c adjusts varnode reference on hash_keys because
	 * of custom-scan interface contract. It shall be redirected to
	 * INDEX_VAR reference, to reference pseudo-scan tlist.
	 * MultiHash has idential pseudo-scan tlist with its outer scan-
	 * path, it always reference correct attribute as long as tuple-
	 * slot is stored on ecxt_scantuple of ExprContext.
	 */
	List	   *hash_keys;
} MultiHashInfo;

static inline void
form_multihash_info(CustomScan *cscan, MultiHashInfo *mh_info)
{
	List   *exprs = NIL;
	List   *privs = NIL;
	union {
		long	ival;
		double	fval;
	} datum;

	privs = lappend(privs, makeInteger(mh_info->depth));
	privs = lappend(privs, makeInteger(mh_info->nslots));
	privs = lappend(privs, makeInteger(mh_info->nloops));
	privs = lappend(privs, makeInteger(mh_info->hashtable_size));
	datum.fval = mh_info->threshold;
	privs = lappend(privs, makeInteger(datum.ival));
	exprs = lappend(exprs, mh_info->hash_keys);

	cscan->custom_private = privs;
	cscan->custom_exprs = exprs;
}

static MultiHashInfo *
deform_multihash_info(CustomScan *cscan)
{
	MultiHashInfo *mh_info = palloc0(sizeof(MultiHashInfo));
	List   *privs = cscan->custom_private;
	List   *exprs = cscan->custom_exprs;
	int		pindex = 0;
	int		eindex = 0;
	union {
        long    ival;
        double  fval;
    } datum;

	mh_info->depth     = intVal(list_nth(privs, pindex++));
	mh_info->nslots    = intVal(list_nth(privs, pindex++));
	mh_info->nloops    = intVal(list_nth(privs, pindex++));
	mh_info->hashtable_size = intVal(list_nth(privs, pindex++));
	datum.ival         = intVal(list_nth(privs, pindex++));
	mh_info->threshold = datum.fval;
	mh_info->hash_keys = list_nth(exprs, eindex++);

	return mh_info;
}















/*
 * MultiHashNode - a data structure to be returned from MultiHash node;
 * that contains a pgstrom_multihash_tables object on shared memory
 * region and related tuplestore/tupleslot for each inner relations.
 */
typedef struct {
	Node		type;	/* T_Invalid */
	pgstrom_multihash_tables *mhtables;
	int				nrels;
} MultiHashNode;

typedef struct
{
	CustomScanState css;
	List		   *join_types;
	ExprState	   *outer_quals;
	List		   *hash_clauses;
	List		   *qual_clauses;
	List		   *host_clauses;	/* to be run on pseudo-scan slot */

	pgstrom_multihash_tables *mhtables;

	/* Source relation/columns of pseudo-scan */
	List	   *ps_src_depth;
	List	   *ps_src_resno;

	/* average ratio to popurate result row */
	double			row_population_ratio;
	/* average number of tuples per page */
	double			ntups_per_page;

	/* state for outer scan */
	bool			outer_done;
	bool			outer_bulkload;
	TupleTableSlot *outer_overflow;

	kern_parambuf  *kparams;
	const char	   *kernel_source;
	Datum			dprog_key;
	pgstrom_queue  *mqueue;

	pgstrom_gpuhashjoin *curr_ghjoin;
	cl_uint			curr_index;
	bool			curr_recheck;
	cl_int			num_running;
	dlist_head		ready_pscans;

	pgstrom_perfmon	pfm;
} GpuHashJoinState;

typedef struct {
	CustomScanState css;
	int				depth;
	cl_uint			nslots;
	cl_int			nbatches_plan;
	cl_int			nbatches_exec;
	double			threshold;
	Size			hashtable_size;
	TupleTableSlot *outer_overflow;
	bool			outer_done;
	kern_hashtable *curr_chunk;
	List		   *hash_keys;
	List		   *hash_keylen;
	List		   *hash_keybyval;
} MultiHashState;

/* declaration of static functions */
static void clserv_process_gpuhashjoin(pgstrom_message *message);

/*
 * path_is_gpuhashjoin - returns true, if supplied pathnode is gpuhashjoin
 */
static bool
path_is_gpuhashjoin(Path *pathnode)
{
	CustomPath *cpath = (CustomPath *) pathnode;

	if (!IsA(cpath, CustomPath))
		return false;
	if (cpath->methods != &gpuhashjoin_path_methods)
		return false;
	return true;
}

/*
 * path_is_mergeable_gpuhashjoin - returns true, if supplied pathnode is
 * gpuhashjoin that can be merged with one more inner scan.
 */
static bool
path_is_mergeable_gpuhashjoin(Path *pathnode)
{
	RelOptInfo		*rel = pathnode->parent;
	GpuHashJoinPath	*gpath;
	List	   *host_clause;
	ListCell   *cell;
	int			last;

	if (!path_is_gpuhashjoin(pathnode))
		return false;

	gpath = (GpuHashJoinPath *) pathnode;
	last = gpath->num_rels - 1;

	/*
	 * target-list must be simple var-nodes only
	 */
	foreach (cell, rel->reltargetlist)
	{
		Expr   *expr = lfirst(cell);

		if (!IsA(expr, Var))
			return false;
	}

	/*
	 * Only INNER JOIN is supported right now
	 */
	if (gpath->inners[last].jointype != JOIN_INNER)
		return false;

	/*
	 * Host qual should not contain volatile function except for
	 * the last inner relation
	 */
	host_clause = gpath->inners[last].host_clauses;
	foreach (cell, host_clause)
	{
		RestrictInfo   *rinfo = lfirst(cell);

		Assert(IsA(rinfo, RestrictInfo));
		if (contain_volatile_functions((Node *)rinfo->clause))
			return false;
	}

	/*
	 * TODO: Is any other condition to be checked?
	 */
	return true;
}

/*
 * pgstrom_plan_is_multihash - returns true, if supplied plan is multihash
 */
bool
pgstrom_plan_is_multihash(const Plan *plan)
{
	CustomScan *cscan = (CustomScan *) plan;

	if (IsA(cscan, CustomScan) &&
		cscan->methods == &multihash_plan_methods)
		return true;
	return false;
}

/*
 * estimate_hashitem_size
 *
 * It estimates size of hashitem for GpuHashJoin
 */
static bool
estimate_hashtable_size(PlannerInfo *root,
						GpuHashJoinPath *gpath,
						Relids required_outer,
						JoinCostWorkspace *workspace)
{
	Size		hashtable_size;
	Size		largest_size;
	Size		threshold_size;
	cl_int		i, i_largest;
	cl_int		numbatches;

	for (i=0; i < gpath->num_rels; i++)
		gpath->inners[i].nloops = 1;
retry:
	hashtable_size = LONGALIGN(offsetof(kern_multihash,
										htable_offset[gpath->num_rels]));
	numbatches = 1;
	largest_size = 0;
	i_largest = -1;
	for (i=0; i < gpath->num_rels; i++)
	{
		Path	   *inner_path = gpath->inners[i].scan_path;
		RelOptInfo *inner_rel = inner_path->parent;
		cl_uint		ncols = list_length(inner_rel->reltargetlist);
		cl_uint		nslots;
		double		ntuples;
		Size		entry_size;
		Size		chunk_size;

		numbatches *= gpath->inners[i].nloops;

		/* force a plausible relation size if no information.
		 * It expects 15% of margin to avoid unnecessary hash-
		 * table split
		 */
		ntuples = Max(1.15 * inner_path->rows, 1000.0);
		if (gpath->inners[i].nloops > 0)
			ntuples /= (double) gpath->inners[i].nloops;

		/* estimate length of each hash entry */
		entry_size = (offsetof(kern_hashentry, htup) +
					  MAXALIGN(offsetof(HeapTupleHeaderData,
										t_bits[BITMAPLEN(ncols)])) +
					  MAXALIGN(inner_rel->width));
		/* estimate length of hash slot */
		nslots = (Size)ntuples / gpath->inners[i].nloops;
		gpath->inners[i].nslots = nslots;
		/* estimate length of this chunk */
		chunk_size = (LONGALIGN(offsetof(kern_hashtable,
										 colmeta[ncols])) +
					  LONGALIGN(sizeof(cl_uint) * nslots) +
					  LONGALIGN(entry_size * (Size)ntuples));
		chunk_size = STROMALIGN(chunk_size);
		if (largest_size < chunk_size)
		{
			largest_size = chunk_size;
			i_largest = i;
		}
		gpath->inners[i].chunk_size = chunk_size;

		/* expand estimated hashtable-size */
		hashtable_size += chunk_size;
	}
	/* also, compute threshold of each chunk */
	threshold_size = 0;
	for (i = gpath->num_rels - 1; i >= 0; i--)
	{
		threshold_size += gpath->inners[i].chunk_size;
		gpath->inners[i].threshold
			= (double) threshold_size / (double) hashtable_size;
	}

	/*
	 * NOTE: In case when extreme number of rows are expected,
	 * it does not make sense to split hash-tables because
	 * increasion of numbatches also increases the total cost
	 * by iteration of outer scan. In this case, the best
	 * strategy is to give up this path, instead of incredible
	 * number of numbatches!
	 */
	if (!add_path_precheck(gpath->cpath.path.parent,
						   workspace->startup_cost,
						   workspace->startup_cost +
						   workspace->run_cost * numbatches,
						   NULL, required_outer))
		return false;

	/*
	 * If hashtable-size is still larger than device limitation,
	 * we try to split the largest chunk then retry the size
	 * estimation.
	 */
	if (hashtable_size > pgstrom_shmem_maxalloc())
	{
		gpath->inners[i_largest].nloops++;
		goto retry;
	}

	/*
	 * Update estimated hashtable_size, but ensure hashtable_size
	 * shall be allocated at least
	 */
	gpath->hashtable_size = Max(hashtable_size, pgstrom_chunk_size());

	/*
	 * Update JoinCostWorkspace according to numbatches
	 */
	workspace->run_cost *= numbatches;
	workspace->total_cost = workspace->startup_cost + workspace->run_cost;

	return true;	/* ok */
}

/*
 * initial_cost_gpuhashjoin
 *
 * cost estimation for GpuHashJoin
 */
static bool
initial_cost_gpuhashjoin(PlannerInfo *root,
						 GpuHashJoinPath *gpath,
						 Relids required_outer,
						 JoinCostWorkspace *workspace)
{
	Path	   *outer_path = gpath->outer_path;
	Cost		startup_cost;
	Cost		run_cost;
	Cost		row_cost;
	double		hashjointuples;
	double		row_population_ratio;
	int			num_hash_clauses = 0;
	int			i;

	/*
	 * cost to construct inner hash-table; that consists of two portions.
	 * 1. total cost to run underlying scan node.
	 * 2. cost to evaluate hash-functions to make hash-table.
	 *
	 * It has to be executed by CPU, so we follow the overall logic
	 * in initial_cost_hashjoin().
	 */
	startup_cost = outer_path->startup_cost;
	run_cost = outer_path->total_cost - outer_path->startup_cost;

	for (i=0; i < gpath->num_rels; i++)
	{
		Path   *inner_path = gpath->inners[i].scan_path;
		List   *hash_clauses = gpath->inners[i].hash_clauses;

		startup_cost += inner_path->total_cost;

		num_hash_clauses += list_length(hash_clauses);
		startup_cost += (cpu_operator_cost *
						 list_length(hash_clauses) +
						 cpu_tuple_cost) * inner_path->rows;
	}

	/* in addition, it takes cost to set up OpenCL device/program */
	startup_cost += pgstrom_gpu_setup_cost;

	/* on the other hands, its cost to run outer scan for joinning
	 * is much less than usual GPU hash join.
	 */
	row_cost = pgstrom_gpu_operator_cost * num_hash_clauses;
	run_cost += row_cost * outer_path->rows;

	/* setup join-cost-workspace */
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
	hashjointuples = gpath->cpath.path.parent->rows;
	row_population_ratio = Max(1.0, hashjointuples / gpath->outer_path->rows);
	if (row_population_ratio > 10.0)
	{
		elog(DEBUG1, "row population ratio (%.2f) too large, give up",
			 row_population_ratio);
		return false;
	}
	else if (row_population_ratio > 5.0)
	{
		elog(NOTICE, "row population ratio (%.2f) too large, rounded to 5.0",
			 row_population_ratio);
		row_population_ratio = 5.0;
	}
	gpath->row_population_ratio = row_population_ratio;

	/*
	 * Estimation of hash table size and number of outer loops
	 * according to the split of hash tables.
	 * In case of estimated plan cost is too large to win the
	 * existing paths, it breaks to find out this path.
	 */
	if (!estimate_hashtable_size(root, gpath, required_outer, workspace))
		return false;

	return true;
}

static void
final_cost_gpuhashjoin(PlannerInfo *root, GpuHashJoinPath *gpath,
					   JoinCostWorkspace *workspace)
{
	Path	   *path = &gpath->cpath.path;
	Cost		startup_cost = workspace->startup_cost;
	Cost		run_cost = workspace->run_cost;
	double		hashjointuples = gpath->cpath.path.parent->rows;
	QualCost	hash_cost;
	QualCost	qual_cost;
	QualCost	host_cost;
	int			i;

	/* Mark the path with correct row estimation */
	if (path->param_info)
		path->rows = path->param_info->ppi_rows;
	else
		path->rows = path->parent->rows;

	/* Compute cost of the hash, qual and host clauses */
	for (i=0; i < gpath->num_rels; i++)
	{
		List	   *hash_clauses = gpath->inners[i].hash_clauses;
		List	   *qual_clauses = gpath->inners[i].qual_clauses;
		List	   *host_clauses = gpath->inners[i].host_clauses;
		double		outer_path_rows = gpath->outer_path->rows;
		double		inner_path_rows = gpath->inners[i].scan_path->rows;
		Relids		inner_relids = gpath->inners[i].scan_path->parent->relids;
		Selectivity	innerbucketsize = 1.0;
		ListCell   *cell;

		/*
		 * Determine bucketsize fraction for inner relation.
		 * We use the smallest bucketsize estimated for any individual
		 * hashclause; this is undoubtedly conservative.
		 */
		foreach (cell, hash_clauses)
		{
			RestrictInfo   *restrictinfo = (RestrictInfo *) lfirst(cell);
			Selectivity		thisbucketsize;
			double			virtualbuckets;
			Node		   *op_expr;

			Assert(IsA(restrictinfo, RestrictInfo));

			/* Right now, GpuHashJoin assumes all the inner record can
			 * be loaded into a single "multihash_tables" structure,
			 * so hash table is never divided and outer relation is
			 * rescanned.
			 * This assumption may change in the future implementation
			 */
			if (inner_path_rows < 1000.0)
				virtualbuckets = 1000.0;
			else
				virtualbuckets = inner_path_rows;

			/*
			 * First we have to figure out which side of the hashjoin clause
			 * is the inner side.
			 *
			 * Since we tend to visit the same clauses over and over when
			 * planning a large query, we cache the bucketsize estimate in the
			 * RestrictInfo node to avoid repeated lookups of statistics.
			 */
			if (bms_is_subset(restrictinfo->right_relids, inner_relids))
				op_expr = get_rightop(restrictinfo->clause);
			else
				op_expr = get_leftop(restrictinfo->clause);

			thisbucketsize = estimate_hash_bucketsize(root, op_expr,
													  virtualbuckets);
			if (innerbucketsize > thisbucketsize)
				innerbucketsize = thisbucketsize;
		}

		/*
		 * Pulls function cost of individual clauses
		 */
		cost_qual_eval(&hash_cost, hash_clauses, root);
		cost_qual_eval(&qual_cost, qual_clauses, root);
		cost_qual_eval(&host_cost, host_clauses, root);
		/*
		 * Because cost_qual_eval returns cost value that assumes CPU
		 * execution, we need to adjust its ratio according to the score
		 * of GPU execution to CPU.
		 */
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
					 * clamp_row_est(inner_path_rows * innerbucketsize) * 0.5);
	}

	/*
	 * Also add cost for qualifiers to be run on host
	 */
	startup_cost += host_cost.startup;
	run_cost += (cpu_tuple_cost + host_cost.per_tuple) * hashjointuples;

	gpath->cpath.path.startup_cost = startup_cost;
	gpath->cpath.path.total_cost = startup_cost + run_cost;
}

/*
 * try_gpuhashjoin_path
 *
 *
 *
 *
 *
 */
static void
try_gpuhashjoin_path(PlannerInfo *root,
					 RelOptInfo *joinrel,
					 JoinType jointype,
					 SpecialJoinInfo *sjinfo,
					 SemiAntiJoinFactors *semifactors,
					 Relids param_source_rels,
					 Relids extra_lateral_rels,
					 Path *outer_path,
					 Path *inner_path,
					 List *restrict_clauses,
					 List *hash_clauses,
					 List *qual_clauses,
					 List *host_clauses)
{
	GpuHashJoinPath	   *gpath;
	Relids				required_outer;
	JoinCostWorkspace	workspace;

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
	 * creation of gpuhashjoin path, without merging underlying gpuhashjoin.
	 */
	gpath = palloc0(offsetof(GpuHashJoinPath, inners[1]));
	NodeSetTag(gpath, T_CustomPath);
	gpath->cpath.path.pathtype = T_CustomScan;
	gpath->cpath.path.parent = joinrel;
	gpath->cpath.path.param_info =
		get_joinrel_parampathinfo(root,
								  joinrel,
								  outer_path,
								  inner_path,
								  sjinfo,
								  bms_copy(required_outer),
								  &restrict_clauses);
	gpath->cpath.path.pathkeys = NIL;
	gpath->cpath.flags = 0;
	gpath->cpath.methods = &gpuhashjoin_path_methods;
	gpath->outer_path = outer_path;
	gpath->num_rels = 1;
	gpath->inners[0].scan_path = inner_path;
	gpath->inners[0].jointype = jointype;
	gpath->inners[0].hash_clauses = hash_clauses;
    gpath->inners[0].qual_clauses = qual_clauses;
	gpath->inners[0].host_clauses = host_clauses;

	/* cost estimation and check availability */
	if (initial_cost_gpuhashjoin(root, gpath, required_outer, &workspace))
	{
		if (add_path_precheck(joinrel,
							  workspace.startup_cost,
							  workspace.total_cost,
							  NULL, required_outer))
		{
			final_cost_gpuhashjoin(root, gpath, &workspace);
			add_path(joinrel, &gpath->cpath.path);
		}
	}

	/*
	 * creation of alternative path, if underlying outer-path is
	 * merginable gpuhashjoin.
	 */
	if (path_is_mergeable_gpuhashjoin(outer_path))
	{
		GpuHashJoinPath	*outer_ghj = (GpuHashJoinPath *) outer_path;
		int		num_rels = outer_ghj->num_rels;

		Assert(num_rels > 0);
		gpath = palloc0(offsetof(GpuHashJoinPath, inners[num_rels + 1]));
		NodeSetTag(gpath, T_CustomPath);
		gpath->cpath.path.pathtype = T_CustomScan;
		gpath->cpath.path.parent = joinrel;
		gpath->cpath.path.param_info =
			get_joinrel_parampathinfo(root,
									  joinrel,
									  outer_path,
									  inner_path,
									  sjinfo,
									  bms_copy(required_outer),
									  &restrict_clauses);
		gpath->cpath.path.pathkeys = NIL;
		gpath->cpath.flags = 0;
		gpath->cpath.methods = &gpuhashjoin_path_methods;
		gpath->num_rels = num_rels + 1;
		gpath->outer_path = outer_ghj->outer_path;
		memcpy(gpath->inners,
			   outer_ghj->inners,
			   offsetof(GpuHashJoinPath, inners[num_rels]) -
			   offsetof(GpuHashJoinPath, inners[0]));
		gpath->inners[num_rels].scan_path = inner_path;
        gpath->inners[num_rels].jointype = jointype;
        gpath->inners[num_rels].hash_clauses = hash_clauses;
        gpath->inners[num_rels].qual_clauses = qual_clauses;
        gpath->inners[num_rels].host_clauses = host_clauses;

		/* cost estimation and check availability */
		if (initial_cost_gpuhashjoin(root, gpath, required_outer, &workspace))
		{
			if (add_path_precheck(joinrel,
								  workspace.startup_cost,
								  workspace.total_cost,
								  NULL, required_outer))
			{
				final_cost_gpuhashjoin(root, gpath, &workspace);
				add_path(joinrel, &gpath->cpath.path);
			}
		}
	}
	bms_free(required_outer);
}

/*
 * gpuhashjoin_add_path
 *
 * callback function invoked to check up GpuHashJoinPath.
 */
#define PATH_PARAM_BY_REL(path, rel)	\
	((path)->param_info && bms_overlap(PATH_REQ_OUTER(path), (rel)->relids))

static void
gpuhashjoin_add_join_path(PlannerInfo *root,
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
	List	   *hash_clauses = NIL;
	List	   *qual_clauses = NIL;
	List	   *host_clauses = NIL;
	ListCell   *cell;

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

	/* nothing to do, if either PG-Strom or GpuHashJoin is not enabled */
	if (!pgstrom_enabled() || !enable_gpuhashjoin)
		return;

	/* Is this join supported by GpuHashJoin?
	 *
	 * Logic came from hash_inner_and_outer()
	 *
	 *
	 *
	 *
	 *
	 */
	if (jointype != JOIN_INNER)
		return;		/* right now, only inner join is supported */
	foreach (cell, restrictlist)
	{
		RestrictInfo   *rinfo = (RestrictInfo *) lfirst(cell);

		/* Even if clause is hash-joinable, here is no benefit
		 * in case when clause is not runnable on OpenCL device.
		 * So, we drop them from the candidate of join-key.
		 */
		if (!pgstrom_codegen_available_expression(rinfo->clause))
		{
			host_clauses = lappend(host_clauses, rinfo);
			continue;
		}

        /*
         * If processing an outer join, only use its own join clauses
         * for hashing.  For inner joins we need not be so picky.
         */
		if (IS_OUTER_JOIN(jointype) && rinfo->is_pushed_down)
		{
			qual_clauses = lappend(qual_clauses, rinfo);
			continue;
		}

		/* Is it hash-joinable clause? */
		if (!rinfo->can_join || !OidIsValid(rinfo->hashjoinoperator))
		{
			qual_clauses = lappend(qual_clauses, rinfo);
			continue;
		}

        /*
         * Check if clause has the form "outer op inner" or "inner op outer".
		 *
		 * Logic is copied from clause_sides_match_join
         */
		if (bms_is_subset(rinfo->left_relids, outerrel->relids) &&
			bms_is_subset(rinfo->right_relids, innerrel->relids))
			rinfo->outer_is_left = true;	/* lefthand side is outer */
		else if (bms_is_subset(rinfo->left_relids, innerrel->relids) &&
				 bms_is_subset(rinfo->right_relids, outerrel->relids))
			rinfo->outer_is_left = false;	/* righthand side is outer */
		else
		{
			/* no good for these input relations */
			qual_clauses = lappend(qual_clauses, rinfo);
			continue;
		}
		hash_clauses = lappend(hash_clauses, rinfo);
	}

	/* If we found any usable hashclauses, make paths */
	if (hash_clauses != NIL)
	{
		/* overall logic comes from hash_inner_and_outer.
		 *
		 * We consider both the cheapest-total-cost and cheapest-startup-cost
		 * outer paths.  There's no need to consider any but the
		 * cheapest-total-cost inner path, however.
		 */
		Path	   *cheapest_startup_outer = outerrel->cheapest_startup_path;
		Path	   *cheapest_total_outer = outerrel->cheapest_total_path;
		Path	   *cheapest_total_inner = innerrel->cheapest_total_path;
		ListCell   *lc1;
		ListCell   *lc2;

		/*
		 * If either cheapest-total path is parameterized by the other rel, we
		 * can't use a hashjoin.  (There's no use looking for alternative
		 * input paths, since these should already be the least-parameterized
		 * available paths.)
		 */
		if (PATH_PARAM_BY_REL(cheapest_total_outer, innerrel) ||
			PATH_PARAM_BY_REL(cheapest_total_inner, outerrel))
			return;

		if (cheapest_startup_outer != NULL)
			try_gpuhashjoin_path(root,
								 joinrel,
								 jointype,
								 sjinfo,
								 semifactors,
								 param_source_rels,
								 extra_lateral_rels,
								 cheapest_startup_outer,
								 cheapest_total_inner,
								 restrictlist,
								 hash_clauses,
								 qual_clauses,
								 host_clauses);

		foreach (lc1, outerrel->cheapest_parameterized_paths)
		{
			Path   *outerpath = (Path *) lfirst(lc1);

			/*
			 * We cannot use an outer path that is parameterized by the
			 * inner rel.
			 */
			if (PATH_PARAM_BY_REL(outerpath, innerrel))
				continue;

			foreach (lc2, innerrel->cheapest_parameterized_paths)
			{
				Path   *innerpath = (Path *) lfirst(lc2);

				/*
				 * We cannot use an inner path that is parameterized by
				 * the outer rel, either.
				 */
				if (PATH_PARAM_BY_REL(innerpath, outerrel))
					continue;

				if (outerpath == cheapest_startup_outer &&
					innerpath == cheapest_total_inner)
					continue;		/* already tried it */

				try_gpuhashjoin_path(root,
									 joinrel,
									 jointype,
									 sjinfo,
									 semifactors,
									 param_source_rels,
									 extra_lateral_rels,
									 outerpath,
									 innerpath,
									 restrictlist,
									 hash_clauses,
									 qual_clauses,
									 host_clauses);
			}
		}
	}
}

static void
gpuhashjoin_codegen_qual(StringInfo body,
						 CustomScan *ghjoin,
						 GpuHashJoinInfo *ghj_info,
						 codegen_context *context)
{
	appendStringInfo(
        body,
		"static bool\n"
		"gpuhashjoin_qual_eval(__private cl_int *errcode,\n"
		"                      __global kern_parambuf *kparams,\n"
		"                      __global kern_data_store *kds,\n"
		"                      __global kern_data_store *ktoast,\n"
		"                      size_t kds_index)\n");
	if (!ghj_info->outer_quals)
	{
		appendStringInfo(
			body,
			"{\n"
			"  return true;\n"
			"}\n");
	}
	else
	{
		List   *save_pseudo_tlist = context->pseudo_tlist;
		Node   *outer_quals = (Node *) ghj_info->outer_quals;
		char   *expr_code;

		context->pseudo_tlist = NIL;
		expr_code = pgstrom_codegen_expression(outer_quals, context);
		appendStringInfo(
			body,
			"{\n"
			"%s%s\n"
			"  return EVAL(%s);\n"
			"}\n",
			pgstrom_codegen_param_declarations(context),
			pgstrom_codegen_var_declarations(context),
			expr_code);
		context->pseudo_tlist = save_pseudo_tlist;
	}
}

static void
gpuhashjoin_codegen_projection(StringInfo body,
							   CustomScan *ghjoin,
							   GpuHashJoinInfo *ghj_info,
							   codegen_context *context)
{
	Plan	   *plan = (Plan *) ghjoin;
	int			depth = 0;
	ListCell   *lc1, *lc2, *lc3;

	/* materialize-mapping function */
	appendStringInfo(
		body,
		"\n"
		"static void\n"
		"gpuhashjoin_projection_mapping(cl_int dest_colidx,\n"
		"                               __private cl_uint *src_depth,\n"
		"                               __private cl_uint *src_colidx)\n"
		"{\n"
		"  switch (dest_colidx)\n"
		"  {\n");
	forthree (lc1, ghjoin->custom_ps_tlist,
              lc2, ghj_info->ps_src_depth,
              lc3, ghj_info->ps_src_resno)
	{
		TargetEntry    *tle = lfirst(lc1);
		int				src_depth = lfirst_int(lc2);
		int				src_resno = lfirst_int(lc3);

		appendStringInfo(
			body,
			"  case %d:\n"
			"    *src_depth = %d;\n"
			"    *src_colidx = %d;\n"
			"    break;\n",
			tle->resno - 1,
			src_depth,
			src_resno - 1);
	}
	appendStringInfo(
        body,
		"  default:\n"
		"    /* should not run here */\n"
		"    break;\n"
		"  }\n"
		"}\n"
		"\n");

	/* projection-datum function */
	appendStringInfo(
        body,
		"static void\n"
		"gpuhashjoin_projection_datum(__private cl_int *errcode,\n"
		"                             __global Datum *slot_values,\n"
		"                             __global cl_char *slot_isnull,\n"
		"                             cl_int depth,\n"
		"                             cl_int colidx,\n"
		"                             hostptr_t hostaddr,\n"
		"                             __global void *datum)\n"
		"{\n"
		"  switch (depth)\n"
		"  {\n");
	depth = 0;
    do {
        appendStringInfo(
            body,
            "  case %d:\n"
            "    switch (colidx)\n"
            "    {\n", depth);
		forthree (lc1, ghjoin->custom_ps_tlist,
				  lc2, ghj_info->ps_src_depth,
				  lc3, ghj_info->ps_src_resno)
		{
			TargetEntry	   *tle = lfirst(lc1);
			int				src_depth = lfirst_int(lc2);
			int				src_resno = lfirst_int(lc3);
			int16			typlen;
			bool			typbyval;

			if (tle->resjunk)
				break;	/* no valid tle should appear later */
			if (src_depth != depth)
				continue;

			get_typlenbyval(exprType((Node *)tle->expr),
							&typlen, &typbyval);
			if (typbyval)
			{
				char   *cl_type = NULL;

				appendStringInfo(
					body,
					"    case %d:\n"
					"      if (!datum)\n"
					"        slot_isnull[%d] = (cl_char) 1;\n"
					"      else\n"
					"      {\n"
					"        slot_isnull[%d] = (cl_char) 0;\n",
					src_resno - 1,
                    tle->resno - 1,
                    tle->resno - 1);
				if (typlen == sizeof(cl_char))
					cl_type = "cl_char";
				else if (typlen == sizeof(cl_short))
					cl_type = "cl_short";
				else if (typlen == sizeof(cl_int))
					cl_type = "cl_int";
				else if (typlen == sizeof(cl_long))
					cl_type = "cl_long";

				if (cl_type)
				{
					appendStringInfo(
						body,
						"        slot_values[%d]"
						" = (Datum)(*((__global %s *) datum));\n",
						tle->resno - 1,
						cl_type);
				}
				else if (typlen < sizeof(Datum))
				{
					appendStringInfo(
						body,
						"        memcpy(&slot_values[%d], datum, %d);\n",
						tle->resno - 1,
						typlen);
				}
				else
					elog(ERROR, "Bug? unexpected type length (%d)", typlen);
				appendStringInfo(
					body,
					"      }\n"
					"      break;\n");
			}
			else
			{
				appendStringInfo(
					body,
					"    case %d:\n"
					"      if (!datum)\n"
					"        slot_isnull[%d] = (cl_char) 1;\n"
					"      else\n"
					"      {\n"
					"        slot_isnull[%d] = (cl_char) 0;\n"
					"        slot_values[%d] = (Datum) hostaddr;\n"
					"      }\n"
					"      break;\n",
					src_resno - 1,
					tle->resno - 1,
					tle->resno - 1,
					tle->resno - 1);
			}
		}
        appendStringInfo(
            body,
            "    default: /* do nothing */ break;\n"
            "    }\n"
			"    break;\n");
		plan = innerPlan(plan);
		depth++;
	} while (plan);
	appendStringInfo(
		body,
		"  default: /* do nothing */ break;\n"
		"  }\n"
		"}\n");
}

static void
gpuhashjoin_codegen_recurse(StringInfo body,
							CustomScan *ghjoin,
							GpuHashJoinInfo *ghj_info,
							CustomScan *mhash, int depth,
							codegen_context *context)
{
	List	   *hash_keys = list_nth(ghj_info->hash_keys, depth - 1);
	Expr	   *hash_clause = list_nth(ghj_info->hash_clauses, depth - 1);
	Expr	   *qual_clause = list_nth(ghj_info->qual_clauses, depth - 1);
	Bitmapset  *attrs_ref = NULL;
	ListCell   *cell;
	char	   *clause;
	int			x;

	/*
	 * construct a hash-key in this nest-level
	 */
	appendStringInfo(body, "cl_uint hash_%u;\n\n", depth);
	appendStringInfo(body, "INIT_CRC32C(hash_%u);\n", depth);
	foreach (cell, hash_keys)
	{
		Node		   *expr = lfirst(cell);
		Oid				type_oid = exprType(expr);
		devtype_info   *dtype;
		char		   *temp;

		dtype = pgstrom_devtype_lookup(type_oid);
		if (!dtype)
			elog(ERROR, "device type \"%s\" not found",
				 format_type_be(type_oid));

		temp = pgstrom_codegen_expression(expr, context);
		appendStringInfo(
			body,
			"hash_%u = pg_%s_comp_crc32(pg_crc32_table, hash_%u, %s);\n",
			depth, dtype->type_name, depth, temp);
		pfree(temp);
	}
	appendStringInfo(body, "FIN_CRC32C(hash_%u);\n", depth);

	/*
	 * construct hash-table walking according to the hash-value
	 * calculated above
	 */
	appendStringInfo(
		body,
		"for (kentry_%d = KERN_HASH_FIRST_ENTRY(khtable_%d, hash_%d);\n"
		"     kentry_%d != NULL;\n"
		"     kentry_%d = KERN_HASH_NEXT_ENTRY(khtable_%d, kentry_%d))\n"
		"{\n",
		depth, depth, depth,
		depth,
		depth, depth, depth);

	/*
	 * construct variables that reference individual entries.
	 * (its value depends on the current entry, so it needs to be
	 * referenced within the loop)
	 */
	pull_varattnos((Node *)ghj_info->hash_clauses, depth, &attrs_ref);
	pull_varattnos((Node *)ghj_info->qual_clauses, depth, &attrs_ref);

	while ((x = bms_first_member(attrs_ref)) >= 0)
	{
		int		resno = x + FirstLowInvalidHeapAttributeNumber;

		foreach (cell, context->pseudo_tlist)
		{
			TargetEntry	   *tle = lfirst(cell);
			Var			   *var = (Var *) tle->expr;
			devtype_info   *dtype;

			if (!IsA(tle->expr, Var) ||
				var->varno != depth ||
				var->varattno != resno)
				continue;

			dtype = pgstrom_devtype_lookup(var->vartype);
			if (!dtype)
				elog(ERROR, "cache lookup failed for device type: %s",
					 format_type_be(var->vartype));

			appendStringInfo(
				body,
				"pg_%s_t KVAR_%u = "
				"pg_%s_hashref(khtable_%d,kentry_%u,errcode,%u);\n",
				dtype->type_name,
				tle->resno,
				dtype->type_name,
				depth,
				depth,
				var->varattno - 1);
			break;
		}
		if (!cell)
			elog(ERROR, "pseudo targetlist lookup failed");
	}
	bms_free(attrs_ref);

	/*
	 * construct hash-key (and other qualifiers) comparison
	 */
	appendStringInfo(body,
					 "if (EQ_CRC32C(kentry_%d->hash, hash_%d)",
					 depth, depth);
	if (hash_clause)
	{
		clause = pgstrom_codegen_expression((Node *)hash_clause, context);
		appendStringInfo(body, " &&\n    EVAL(%s)", clause);
		pfree(clause);
	}
	if (qual_clause)
	{
		clause = pgstrom_codegen_expression((Node *)qual_clause, context);
		appendStringInfo(body, " &&\n    EVAL(%s)", clause);
		pfree(clause);
	}
	appendStringInfo(body, ")\n{\n");

	/*
	 * If we have one more deeper hash-table, one nest level shall be added.
	 * Elsewhere, a code to put hash-join result and to increment the counter
	 * of matched items.
	 */
	if (innerPlan(mhash))
		gpuhashjoin_codegen_recurse(body, ghjoin, ghj_info,
									(CustomScan *) innerPlan(mhash),
									depth + 1,
									context);
	else
	{
		int		i;

		/*
		 * FIXME: needs to set negative value if host-recheck is needed
		 * (errcode: StromError_CpuReCheck)
		 */
		appendStringInfo(
			body,
			"n_matches++;\n"
			"if (rbuffer)\n"
			"{\n"
			"  rbuffer[0] = (cl_int)kds_index + 1;\n");	/* outer relation */
		for (i=1; i <= ghj_info->num_rels; i++)
			appendStringInfo(
				body,
				"  rbuffer[%d] = (cl_int)"
				"((uintptr_t)kentry_%d - (uintptr_t)khtable_%d);\n",
				i, i, i);
		appendStringInfo(
            body,
			"  rbuffer += %d;\n"
			"}\n",
			ghj_info->num_rels + 1);
	}
	appendStringInfo(body, "}\n");
	appendStringInfo(body, "}\n");
}

static char *
gpuhashjoin_codegen_type_declarations(codegen_context *context)
{
	StringInfoData	str;
	ListCell	   *cell;

	initStringInfo(&str);
	foreach (cell, context->type_defs)
	{
		devtype_info   *dtype = lfirst(cell);

		appendStringInfo(&str, "STROMCL_ANYTYPE_HASHREF_TEMPLATE(%s)\n",
						 dtype->type_name);
	}
    appendStringInfoChar(&str, '\n');

    return str.data;
}

static char *
gpuhashjoin_codegen(PlannerInfo *root,
					CustomScan *ghjoin,
					GpuHashJoinInfo *ghj_info,
					codegen_context *context)
{
	StringInfoData	str;
	StringInfoData	decl;
	StringInfoData	body;
	Bitmapset	   *attrs_ref = NULL;
	ListCell	   *cell;
	int				depth;
	int				x;

	initStringInfo(&str);
	initStringInfo(&body);
	initStringInfo(&decl);

	/* declaration of gpuhashjoin_execute */
	appendStringInfo(
		&decl,
		"static cl_uint\n"
		"gpuhashjoin_execute(__private cl_int *errcode,\n"
		"                    __global kern_parambuf *kparams,\n"
		"                    __global kern_multihash *kmhash,\n"
		"                    __local cl_uint *pg_crc32_table,\n"
		"                    __global kern_data_store *kds,\n"
		"                    __global kern_data_store *ktoast,\n"
		"                    size_t kds_index,\n"
		"                    __global cl_int *rbuffer)\n"
		"{\n"
		);
	/* reference to each hash table */
	for (depth=1; depth <= ghj_info->num_rels; depth++)
	{
		appendStringInfo(
			&decl,
			"__global kern_hashtable *khtable_%d"
			" = KERN_HASHTABLE(kmhash,%d);\n",
			depth, depth);
	}
	/* variable for individual hash entries */
	for (depth=1; depth <= ghj_info->num_rels; depth++)
	{
		appendStringInfo(
			&decl,
			"__global kern_hashentry *kentry_%d;\n",
			depth);
	}

	/*
	 * declaration of variables that reference outer relations
	 */
	pull_varattnos((Node *)ghj_info->hash_clauses, 0, &attrs_ref);
	pull_varattnos((Node *)ghj_info->qual_clauses, 0, &attrs_ref);

	while ((x = bms_first_member(attrs_ref)) >= 0)
	{
		int		resno = x + FirstLowInvalidHeapAttributeNumber;

		foreach (cell, context->pseudo_tlist)
		{
			TargetEntry	   *tle = lfirst(cell);
			Var			   *var = (Var *) tle->expr;
			devtype_info   *dtype;

			if (!IsA(tle->expr, Var) ||
				var->varno != 0 ||
				var->varattno != resno)
				continue;

			dtype = pgstrom_devtype_lookup(var->vartype);
			if (!dtype)
				elog(ERROR, "cache lookup failed for device type \"%s\"",
					 format_type_be(var->vartype));

			appendStringInfo(&body,
							 "pg_%s_t KVAR_%u = "
							 "pg_%s_vref(kds,ktoast,errcode,%u,kds_index);\n",
							 dtype->type_name,
							 tle->resno,
							 dtype->type_name,
							 var->varattno - 1);
			break;
		}
		if (!cell)
			elog(ERROR, "pseudo targetlist lookup failed");
	}
	bms_free(attrs_ref);

	/* misc variable definitions */
	appendStringInfo(&body,
					 "cl_int n_matches = 0;\n");

	/* nested loop for hash tables */
	gpuhashjoin_codegen_recurse(&body,
								ghjoin,
								ghj_info,
								(CustomScan *) innerPlan(ghjoin),
								1,
								context);

	/* end of gpuhashjoin_execute function */
	appendStringInfo(&body,
					 "return n_matches;\n"
					 "}\n");

	/* reference to kern_params */
	appendStringInfo(&decl, "%s",
					 pgstrom_codegen_param_declarations(context));
	context->param_refs = NULL;

	/* integrate decl and body */
	appendStringInfo(&decl, "%s", body.data);

	/* also, gpuhashjoin_projection_datum() */
	context->used_vars = NIL;
	context->param_refs = NULL;
	gpuhashjoin_codegen_projection(&decl, ghjoin, ghj_info, context);

	/* also, gpuhashjoin_qual_eval */
	context->used_vars = NIL;
	context->param_refs = NULL;
	gpuhashjoin_codegen_qual(&decl, ghjoin, ghj_info, context);

	/* put declarations of types/funcs/params */
	appendStringInfo(&str, "%s%s%s",
					 gpuhashjoin_codegen_type_declarations(context),
					 pgstrom_codegen_func_declarations(context),
					 decl.data);

	/* include opencl_hashjoin.h */
	context->extra_flags |= DEVKERNEL_NEEDS_HASHJOIN;

	return str.data;
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
typedef struct {
	List	   *ps_tlist;
	List	   *ps_depth;
	List	   *ps_resno;
	GpuHashJoinPath	*gpath;
	bool		resjunk;
} build_pseudo_targetlist_context;

static bool
build_pseudo_targetlist_walker(Node *node,
							   build_pseudo_targetlist_context *context)
{
	GpuHashJoinPath *gpath = context->gpath;
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
				if (ps_node->vartype != varnode->vartype ||
					ps_node->vartypmod != varnode->vartypmod ||
					ps_node->varcollid != varnode->varcollid)
					elog(ERROR, "inconsistent Var node on ps_tlist");
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
build_pseudo_targetlist(GpuHashJoinPath *gpath,
						GpuHashJoinInfo *ghj_info,
						List *targetlist)
{
	build_pseudo_targetlist_context	context;

	context.ps_tlist = NIL;
	context.ps_depth = NIL;
	context.ps_resno = NIL;
	context.gpath    = gpath;
	context.resjunk  = false;

	build_pseudo_targetlist_walker((Node *) targetlist, &context);
	build_pseudo_targetlist_walker((Node *) ghj_info->host_clauses,
								   &context);
	/*
	 * Above are host referenced columns. On the other hands, the columns
	 * newly added below are device-only columns, so it will never
	 * referenced by the host-side. We mark it resjunk=true.
	 */
	context.resjunk = true;
	build_pseudo_targetlist_walker((Node *) ghj_info->hash_clauses,
								   &context);
	build_pseudo_targetlist_walker((Node *) ghj_info->qual_clauses,
								   &context);
	build_pseudo_targetlist_walker((Node *) ghj_info->outer_quals,
								   &context);

	Assert(list_length(context.ps_tlist) == list_length(context.ps_depth) &&
		   list_length(context.ps_tlist) == list_length(context.ps_resno));

	ghj_info->ps_src_depth = context.ps_depth;
	ghj_info->ps_src_resno = context.ps_resno;

	return context.ps_tlist;
}

static Expr *
build_flatten_qualifier(List *clauses)
{
	List	   *args = NIL;
	ListCell   *cell;

	foreach (cell, clauses)
	{
		Expr   *expr = lfirst(cell);

		if (!expr)
			continue;
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
 * fixup_device_expression
 *
 * Unlike host executable qualifiers, device qualifiers need to reference
 * tables before the joinning. So, we need to apply special initialization
 * to reference OUTER/INNER relation, instead of pseudo-scan relation.
 */
typedef struct {
	GpuHashJoinPath *gpath;
	CustomScan *cscan;
} fixup_device_expression_context;

static Node *
fixup_device_expression_mutator(Node *node,
								fixup_device_expression_context *context)
{
	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		GpuHashJoinPath *gpath = context->gpath;
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
					break;
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
	return expression_tree_mutator(node, fixup_device_expression_mutator,
								   (void *) context);
}

static Node *
fixup_device_expression(CustomScan *cscan,
						GpuHashJoinPath *gpath,
						Node *node, bool is_single)
{
	fixup_device_expression_context context;
	List	   *result = NIL;
	ListCell   *cell;

	context.gpath = gpath;
	context.cscan = cscan;
	if (is_single)
		return fixup_device_expression_mutator(node, &context);

	Assert(IsA(node, List) &&
		   list_length((List *) node) == gpath->num_rels);

	foreach (cell, (List *) node)
	{
		Node   *node = lfirst(cell);

		node = fixup_device_expression_mutator(node, &context);
		result = lappend(result, node);
	}
	return (Node *)result;
}

static Plan *
create_gpuhashjoin_plan(PlannerInfo *root,
						RelOptInfo *rel,
						CustomPath *best_path,
						List *tlist,
						List *clauses)
{
	GpuHashJoinPath *gpath = (GpuHashJoinPath *)best_path;
	GpuHashJoinInfo  ghj_info;
	CustomScan *ghjoin;
	Plan	   *outer_plan;
	Plan	   *prev_plan = NULL;
	codegen_context context;
	int			i;





	ghjoin = makeNode(CustomScan);
	ghjoin->scan.plan.targetlist = tlist;
	ghjoin->scan.plan.qual = NIL;
	ghjoin->scan.scanrelid = 0;	/* not related to any relation */
	ghjoin->flags = 0;
	ghjoin->methods = &gpuhashjoin_plan_methods;

	memset(&ghj_info, 0, sizeof(GpuHashJoinInfo));
	ghj_info.num_rels = gpath->num_rels;
	ghj_info.hashtable_size = gpath->hashtable_size;
	ghj_info.row_population_ratio = gpath->row_population_ratio;

	for (i=0; i < gpath->num_rels; i++)
	{
		CustomScan	   *mhash;
		MultiHashInfo	mh_info;
		List		   *hash_clause = gpath->inners[i].hash_clauses;
		List		   *qual_clause = gpath->inners[i].qual_clauses;
		List		   *host_clause = gpath->inners[i].host_clauses;
		List		   *hash_inner_keys = NIL;
		List		   *hash_outer_keys = NIL;
		Path		   *scan_path = gpath->inners[i].scan_path;
		Plan		   *scan_plan = create_plan_recurse(root, scan_path);
		ListCell	   *cell;

		/* for convenience of later stage */
		gpath->inners[i].scan_plan = scan_plan;

		mhash = makeNode(CustomScan);
		mhash->scan.plan.startup_cost = scan_plan->total_cost;
		mhash->scan.plan.total_cost = scan_plan->total_cost;
		mhash->scan.plan.plan_rows = scan_plan->plan_rows;
		mhash->scan.plan.plan_width = scan_plan->plan_width;
		mhash->scan.plan.targetlist = scan_plan->targetlist;
		mhash->scan.plan.qual = NIL;
		mhash->scan.scanrelid = 0;
		mhash->flags = 0;
		mhash->custom_ps_tlist = scan_plan->targetlist;
		mhash->methods = &multihash_plan_methods;

		memset(&mh_info, 0, sizeof(MultiHashInfo));
		mh_info.depth = i + 1;
		mh_info.nslots = gpath->inners[i].nslots;
		mh_info.nloops = gpath->inners[i].nloops;
		mh_info.hashtable_size = gpath->hashtable_size;
		mh_info.threshold = gpath->inners[i].threshold;
		foreach (cell, hash_clause)
		{
			RestrictInfo   *rinfo = lfirst(cell);
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
		Assert(list_length(hash_inner_keys) == list_length(hash_outer_keys));
		mh_info.hash_keys = hash_inner_keys;
		form_multihash_info(mhash, &mh_info);

		/* also add keys and clauses to ghj_info */
		ghj_info.hash_keys = lappend(ghj_info.hash_keys, hash_outer_keys);
		ghj_info.join_types = lappend_int(ghj_info.join_types,
										  gpath->inners[i].jointype);
		hash_clause = extract_actual_clauses(hash_clause, false);
		ghj_info.hash_clauses = lappend(ghj_info.hash_clauses,
										build_flatten_qualifier(hash_clause));

		qual_clause = extract_actual_clauses(qual_clause, false);
		ghj_info.qual_clauses = lappend(ghj_info.qual_clauses,
										build_flatten_qualifier(qual_clause));

		host_clause = extract_actual_clauses(host_clause, false);
		ghj_info.host_clauses = lappend(ghj_info.host_clauses,
										build_flatten_qualifier(host_clause));

		/* chain it under the GpuHashJoin */
		outerPlan(mhash) = scan_plan;
		if (prev_plan)
			innerPlan(prev_plan) = &mhash->scan.plan;
		else
			innerPlan(ghjoin) = &mhash->scan.plan;

		prev_plan = &mhash->scan.plan;
	}

	/*
	 * Creation of the underlying outer Plan node. In case of SeqScan,
	 * it may make sense to replace it with GpuScan for bulk-loading.
	 */
	outer_plan = create_plan_recurse(root, gpath->outer_path);
	if (IsA(outer_plan, SeqScan) || IsA(outer_plan, CustomScan))
	{
		Query	   *parse = root->parse;
		Index		outer_scanrelid = ((Scan *) outer_plan)->scanrelid;
		Bitmapset  *outer_attrefs = NULL;
		List	   *outer_quals = NULL;
		Plan	   *alter_plan;

		pull_varattnos((Node *)ghjoin->scan.plan.targetlist,
					   outer_scanrelid,
					   &outer_attrefs);
		pull_varattnos((Node *)ghj_info.hash_clauses,
					   outer_scanrelid,
					   &outer_attrefs);
		pull_varattnos((Node *)ghj_info.qual_clauses,
					   outer_scanrelid,
					   &outer_attrefs);
		pull_varattnos((Node *)ghj_info.host_clauses,
					   outer_scanrelid,
					   &outer_attrefs);
		alter_plan = gpuscan_try_replace_relscan(outer_plan,
												 parse->rtable,
												 outer_attrefs,
												 &outer_quals);
		if (alter_plan)
		{
			ghj_info.outer_quals = build_flatten_qualifier(outer_quals);
			ghj_info.outer_bulkload = true;
			outerPlan(ghjoin) = alter_plan;
		}
		else
			outerPlan(ghjoin) = outer_plan;

		bms_free(outer_attrefs);
	}
	else
		outerPlan(ghjoin) = outer_plan;

	gpath->outer_plan = outer_plan;	/* for convenience below */

	/*
	 * Build a pseudo-scan targetlist.
	 */
	ghjoin->custom_ps_tlist = build_pseudo_targetlist(gpath, &ghj_info, tlist);
	ghj_info.outer_quals = (Expr *)
		fixup_device_expression(ghjoin, gpath,
								(Node *)ghj_info.outer_quals, true);
	ghj_info.hash_clauses = (List *)
		fixup_device_expression(ghjoin, gpath,
								(Node *)ghj_info.hash_clauses, false);
	ghj_info.qual_clauses = (List *)
		fixup_device_expression(ghjoin, gpath,
								(Node *)ghj_info.qual_clauses, false);
	ghj_info.hash_keys = (List *)
		fixup_device_expression(ghjoin, gpath,
								(Node *)ghj_info.hash_keys,false);

	pgstrom_init_codegen_context(&context);
	context.pseudo_tlist = (List *)
		fixup_device_expression(ghjoin, gpath,
								(Node *)ghjoin->custom_ps_tlist, true);
	ghj_info.kernel_source =
		gpuhashjoin_codegen(root, ghjoin, &ghj_info, &context);
	ghj_info.extra_flags = context.extra_flags |
		(!devprog_enable_optimize ? DEVKERNEL_DISABLE_OPTIMIZE : 0);
	ghj_info.used_params = context.used_params;

	form_gpuhashjoin_info(ghjoin, &ghj_info);

	return &ghjoin->scan.plan;
}

static void
gpuhashjoin_textout_path(StringInfo str, const CustomPath *node)
{
	GpuHashJoinPath *gpath = (GpuHashJoinPath *) node;
	char	   *temp;
	int			i;

	/* outerpath */
	temp = nodeToString(gpath->outer_path);
	appendStringInfo(str, " :outer_path %s", temp);

	/* hashtable_size */
	appendStringInfo(str, " :hashtable_size %zu", gpath->hashtable_size);

	/* num_rels */
	appendStringInfo(str, " :num_rels %d", gpath->num_rels);

	/* inners */
	appendStringInfo(str, " :num_rels (");
	for (i=0; i < gpath->num_rels; i++)
	{
		appendStringInfo(str, "{");
		/* path */
		temp = nodeToString(gpath->inners[i].scan_path);
		appendStringInfo(str, " :scan_path %s", temp);

		/* jointype */
		appendStringInfo(str, " :jointype %d",
						 (int)gpath->inners[i].jointype);

		/* hash_clause */
		temp = nodeToString(gpath->inners[i].hash_clauses);
		appendStringInfo(str, " :hash_clause %s", temp);

		/* qual_clause */
		temp = nodeToString(gpath->inners[i].qual_clauses);
		appendStringInfo(str, " :qual_clause %s", temp);

		/* host_clause */
		temp = nodeToString(gpath->inners[i].host_clauses);
		appendStringInfo(str, " :host_clause %s", temp);

		/* threshold */
		appendStringInfo(str, " :threshold %f",
						 gpath->inners[i].threshold);
		/* chunk_size */
		appendStringInfo(str, " :chunk_size %zu",
						 gpath->inners[i].chunk_size);
		/* nslots */
		appendStringInfo(str, " :nslots %u",
						 gpath->inners[i].nslots);
		/* nloops */
		appendStringInfo(str, " :nslots %d",
                         gpath->inners[i].nloops);
		appendStringInfo(str, "}");		
	}
	appendStringInfo(str, ")");
}

/*
 * gpuhashjoin_create_scan_state
 *
 * callback to create GpuHashJoinState object.
 */
static Node *
gpuhashjoin_create_scan_state(CustomScan *cscan)
{
	GpuHashJoinState   *ghjs = palloc0(sizeof(GpuHashJoinState));

	NodeSetTag(ghjs, T_CustomScanState);
	ghjs->css.flags = cscan->flags;
	ghjs->css.methods = &gpuhashjoin_exec_methods.c;

	return (Node *) ghjs;
}

/*
 * pgstrom_plan_is_gpuhashjoin
 *
 * It returns true, if supplied plan node is gpuhashjoin.
 */
bool
pgstrom_plan_is_gpuhashjoin(const Plan *plan)
{
	CustomScan	   *cscan = (CustomScan *) plan;

	if (IsA(cscan, CustomScan) &&
		cscan->methods == &gpuhashjoin_plan_methods)
		return true;
	return false;
}

/*
 * multihash_dump_tables
 *
 * For debugging, it dumps contents of multihash-tables
 */
static inline void
multihash_dump_tables(pgstrom_multihash_tables *mhtables)
{
	StringInfoData	str;
	int		i, j;

	initStringInfo(&str);
	for (i=1; i <= mhtables->kern.ntables; i++)
	{
		kern_hashtable *khash = KERN_HASHTABLE(&mhtables->kern, i);

		elog(INFO, "----hashtable[%d] {nslots=%u ncols=%u} ------------",
			 i, khash->nslots, khash->ncols);
		for (j=0; j < khash->ncols; j++)
		{
			elog(INFO, "colmeta {attbyval=%d attalign=%d attlen=%d "
				 "attnum=%d attcacheoff=%d}",
				 khash->colmeta[j].attbyval,
				 khash->colmeta[j].attalign,
				 khash->colmeta[j].attlen,
				 khash->colmeta[j].attnum,
				 khash->colmeta[j].attcacheoff);
		}

		for (j=0; j < khash->nslots; j++)
		{
			kern_hashentry *kentry;

			for (kentry = KERN_HASH_FIRST_ENTRY(khash, j);
				 kentry;
				 kentry = KERN_HASH_NEXT_ENTRY(khash, kentry))
			{
				elog(INFO, "entry[%d] hash=%08x rowid=%u t_len=%u",
					 j, kentry->hash, kentry->rowid, kentry->t_len);
			}
		}
	}
}

static void
gpuhashjoin_begin(CustomScanState *node, EState *estate, int eflags)
{
	PlanState		   *ps = &node->ss.ps;
	GpuHashJoinState   *ghjs = (GpuHashJoinState *) node;
	CustomScan		   *ghjoin = (CustomScan *)node->ss.ps.plan;
	GpuHashJoinInfo	   *ghj_info = deform_gpuhashjoin_info(ghjoin);
	TupleDesc			tupdesc;
	ListCell		   *cell;

	ghjs->join_types = copyObject(ghj_info->join_types);

	/*
	 * NOTE: outer_quals, hash_clauses and qual_clauses assume
	 * input stream that is not joinned yet, so its expression
	 * state must be initialized based on the device qualifiers'
	 * perspective. On the other hands, host qualifiers can
	 * assume pseudo-scan targetlist already constructed.
	 */
	ghjs->outer_quals = ExecInitExpr(ghj_info->outer_quals, ps);
	foreach (cell, ghj_info->hash_clauses)
		ghjs->hash_clauses = lappend(ghjs->hash_clauses,
									 ExecInitExpr(lfirst(cell), ps));
	foreach (cell, ghj_info->qual_clauses)
		ghjs->qual_clauses = lappend(ghjs->qual_clauses,
									 ExecInitExpr(lfirst(cell), ps));
	foreach (cell, ghj_info->host_clauses)
		ghjs->host_clauses = lappend(ghjs->host_clauses,
									 ExecInitExpr(lfirst(cell), ps));
	/* for ExecScan */
	foreach (cell, ghj_info->host_clauses)
	{
		if (!lfirst(cell))
			continue;
		ghjs->css.ss.ps.qual = lappend(ghjs->css.ss.ps.qual,
									   ExecInitExpr(lfirst(cell), ps));
	}

	/* XXX - it may need to sort by depth/resno */
	ghjs->ps_src_depth = ghj_info->ps_src_depth;
	ghjs->ps_src_resno = ghj_info->ps_src_resno;

	/*
	 * initialize child nodes
	 */
	outerPlanState(ghjs) = ExecInitNode(outerPlan(ghjoin), estate, eflags);
	innerPlanState(ghjs) = ExecInitNode(innerPlan(ghjoin), estate, eflags);

	/*
	 * rough estimation for expected amount of resource comsumption
	 */
	ghjs->row_population_ratio = ghj_info->row_population_ratio;
	ghjs->ntups_per_page =
		((double)(BLCKSZ - MAXALIGN(SizeOfPageHeaderData))) /
		((double)(sizeof(ItemIdData) +
				  sizeof(HeapTupleHeaderData) +
				  outerPlanState(ghjs)->plan->plan_width));
	/*
	 * fix-up definition of the pseudo scan tuple-desc, because our
	 * ps_tlist contains TLE with resjust=true; that means columns
	 * referenced by device only, so it leads unnecessary projection.
	 */
	tupdesc = ExecCleanTypeFromTL(ghjoin->custom_ps_tlist, false);
	ExecAssignScanType(&ghjs->css.ss, tupdesc);
	ExecAssignScanProjectionInfo(&ghjs->css.ss);

	/*
	 * initialize outer scan state
	 */
	ghjs->outer_done = false;
	ghjs->outer_bulkload =
		(!pgstrom_debug_bulkload_enabled ? false : ghj_info->outer_bulkload);
	ghjs->outer_overflow = NULL;

	/*
	 * initialize kernel execution parameter
	 */
	ghjs->kparams = pgstrom_create_kern_parambuf(ghj_info->used_params,
												 ps->ps_ExprContext);
	Assert(ghj_info->kernel_source != NULL);
	ghjs->kernel_source = ghj_info->kernel_source;
	ghjs->dprog_key = pgstrom_get_devprog_key(ghj_info->kernel_source,
											  ghj_info->extra_flags);
	pgstrom_track_object((StromObject *)ghjs->dprog_key, 0);

	ghjs->mqueue = pgstrom_create_queue();
	pgstrom_track_object(&ghjs->mqueue->sobj, 0);

	/*
	 * initialize misc stuff
	 */
	ghjs->curr_ghjoin = NULL;
	ghjs->curr_index = 0;
	ghjs->curr_recheck = false;
	ghjs->num_running = 0;
	dlist_init(&ghjs->ready_pscans);

	/* Is perfmon needed? */
	ghjs->pfm.enabled = pgstrom_perfmon_enabled;
}

static void
pgstrom_release_gpuhashjoin(pgstrom_message *message)
{
	pgstrom_gpuhashjoin *gpuhashjoin = (pgstrom_gpuhashjoin *) message;

	/* unlink message queue and device program */
	pgstrom_put_queue(gpuhashjoin->msg.respq);
    pgstrom_put_devprog_key(gpuhashjoin->dprog_key);

	/* unlink hashjoin-table */
	multihash_put_tables(gpuhashjoin->mhtables);

	/* unlink outer data store */
	if (gpuhashjoin->pds)
		pgstrom_put_data_store(gpuhashjoin->pds);

	/* unlink destination data store */
	if (gpuhashjoin->pds_dest)
		pgstrom_put_data_store(gpuhashjoin->pds_dest);

	/* release this message itself */
	pgstrom_shmem_free(gpuhashjoin);
}

static pgstrom_gpuhashjoin *
pgstrom_create_gpuhashjoin(GpuHashJoinState *ghjs,
						   pgstrom_bulkslot *bulk,
						   int result_format)
{
	pgstrom_multihash_tables *mhtables = ghjs->mhtables;
	pgstrom_gpuhashjoin	*gpuhashjoin;
	pgstrom_data_store *pds_dest;
	pgstrom_data_store *pds = bulk->pds;
	kern_data_store	   *kds = pds->kds;
	cl_int				nvalids = bulk->nvalids;
	cl_int				nrels = mhtables->kern.ntables;
	cl_uint				nrooms;
	Size				required;
	TupleDesc			tupdesc;
	kern_hashjoin	   *khashjoin;
	kern_resultbuf	   *kresults;
	kern_parambuf	   *kparams;
	kern_row_map	   *krowmap;

	/*
	 * Allocation of pgstrom_gpuhashjoin message object
	 */
	required = (offsetof(pgstrom_gpuhashjoin, khashjoin) +
				STROMALIGN(ghjs->kparams->length) +
				STROMALIGN(sizeof(kern_resultbuf)) +
				(nvalids < 0 ?
				 STROMALIGN(offsetof(kern_row_map, rindex[0])) :
				 STROMALIGN(offsetof(kern_row_map, rindex[nvalids]))));
	gpuhashjoin = pgstrom_shmem_alloc(required);
	if (!gpuhashjoin)
		elog(ERROR, "out of shared memory");

	/* initialization of the common message field */
	pgstrom_init_message(&gpuhashjoin->msg,
						 StromTag_GpuHashJoin,
						 ghjs->mqueue,
						 clserv_process_gpuhashjoin,
						 pgstrom_release_gpuhashjoin,
						 ghjs->pfm.enabled);
	/* initialization of other fields also */
	gpuhashjoin->dprog_key = pgstrom_retain_devprog_key(ghjs->dprog_key);
	gpuhashjoin->mhtables = multihash_get_tables(mhtables);
	gpuhashjoin->pds = pds;
	gpuhashjoin->pds_dest = NULL;		/* to be set below */
	khashjoin = &gpuhashjoin->khashjoin;

	/* setup kern_parambuf */
	kparams = KERN_HASHJOIN_PARAMBUF(khashjoin);
	memcpy(kparams, ghjs->kparams, ghjs->kparams->length);

	/* setup kern_resultbuf */
	nrooms = (cl_uint)((double)(nvalids < 0 ? kds->nitems : nvalids) *
					   ghjs->row_population_ratio * 1.1);
	kresults = KERN_HASHJOIN_RESULTBUF(khashjoin);
    memset(kresults, 0, sizeof(kern_resultbuf));
	kresults->nrels = nrels + 1;
	kresults->nrooms = nrooms;
	kresults->nitems = 0;
	kresults->errcode = StromError_Success;

	/* setup kern_row_map */
	krowmap = KERN_HASHJOIN_ROWMAP(khashjoin);
	if (nvalids < 0)
		krowmap->nvalids = -1;
	else
	{
		krowmap->nvalids = nvalids;
		memcpy(krowmap->rindex, bulk->rindex, sizeof(cl_int) * nvalids);
	}

	/*
	 * Once a pgstrom_data_store connected to the pgstrom_gpuhashjoin
	 * structure, it becomes pgstrom_release_gpuhashjoin's role to
	 * unlink this data-store. So, we don't need to track individual
	 * data-store no longer.
	 */
	pgstrom_untrack_object(&pds->sobj);
	pgstrom_track_object(&gpuhashjoin->msg.sobj, 0);

	/*
	 * allocation of the destination data-store
	 */
	tupdesc = ghjs->css.ss.ss_ScanTupleSlot->tts_tupleDescriptor;
	if (result_format == KDS_FORMAT_TUPSLOT)
		pds_dest = pgstrom_create_data_store_tupslot(tupdesc, nrooms, false);
	else if (result_format == KDS_FORMAT_ROW_FLAT)
	{
		int		plan_width = ghjs->css.ss.ps.plan->plan_width;
		Size	length;

		length = (STROMALIGN(offsetof(kern_data_store,
									  colmeta[tupdesc->natts])) +
				  STROMALIGN(sizeof(kern_rowitem) * nrooms) +
				  (MAXALIGN(offsetof(HeapTupleHeaderData,
									 t_bits[BITMAPLEN(tupdesc->natts)]) +
							(tupdesc->tdhasoid ? sizeof(Oid) : 0)) +
				   MAXALIGN(plan_width)) * nrooms);
		pds_dest = pgstrom_create_data_store_row_flat(tupdesc, length);
	}
	else
		elog(ERROR, "Bug? unexpected result format: %d", result_format);
	gpuhashjoin->pds_dest = pds_dest;

	return gpuhashjoin;
}

static pgstrom_gpuhashjoin *
gpuhashjoin_load_next_chunk(GpuHashJoinState *ghjs, int result_format)
{
	PlanState		   *subnode = outerPlanState(ghjs);
	TupleDesc			tupdesc = ExecGetResultType(subnode);
	pgstrom_bulkslot	bulkdata;
	pgstrom_bulkslot   *bulk = NULL;
	struct timeval		tv1, tv2, tv3;

	/*
	 * Logic to fetch inner multihash-table looks like nested-loop.
	 * If all the underlying inner scan already scaned its outer relation,
	 * current depth makes advance its scan pointer with reset of underlying
	 * scan pointer, or returns NULL if it is already reached end of scan.
	 */
retry:
	if (ghjs->pfm.enabled)
		gettimeofday(&tv1, NULL);

	if (ghjs->outer_done || !ghjs->mhtables)
	{
		PlanState	   *inner_ps = innerPlanState(ghjs);
		MultiHashNode  *mhnode;

		/* load an inner hash-table */
		mhnode = (MultiHashNode *) BulkExecProcNode(inner_ps);
		if (!mhnode)
		{
			if (ghjs->pfm.enabled)
			{
				gettimeofday(&tv2, NULL);
				ghjs->pfm.time_inner_load += timeval_diff(&tv1, &tv2);
			}
			return NULL;	/* end of inner multi-hashtable */
		}

		/*
		 * unlink the previous pgstrom_multihash_tables
		 *
		 * NOTE: we shouldn't release the multihash-tables immediately, even
		 * if outer scan already touched end of the scan, because GpuHashJoin
		 * may be required to rewind the position later. In multihash-table
		 * is flat (not divided to multiple portion) and no parameter changes,
		 * we can omit inner scan again.
		 */
		if (ghjs->mhtables)
		{
			pgstrom_multihash_tables *mhtables = ghjs->mhtables;

			Assert(ghjs->outer_done);	/* should not be the first call */
			pgstrom_untrack_object(&mhtables->sobj);
			multihash_put_tables(mhtables);
			ghjs->mhtables = NULL;
		}
		ghjs->mhtables = mhnode->mhtables;
		pfree(mhnode);

		/*
		 * Rewind the outer scan pointer, if it is not first time.
		 */
		if (ghjs->outer_done)
		{
			ExecReScan(outerPlanState(ghjs));
			ghjs->outer_done = false;
		}
	}

	if (ghjs->pfm.enabled)
		gettimeofday(&tv2, NULL);

	if (!ghjs->outer_bulkload)
	{
		/* Scan the outer relation using row-by-row mode */
		pgstrom_data_store *pds = NULL;

		while (true)
		{
			TupleTableSlot *slot;

			if (ghjs->outer_overflow)
			{
				slot = ghjs->outer_overflow;
				ghjs->outer_overflow = NULL;
			}
			else
			{
				slot = ExecProcNode(subnode);
				if (TupIsNull(slot))
				{
					ghjs->outer_done = true;
					break;
				}
			}
			/* create a new data-store if not constructed yet */
			if (!pds)
			{
				pds = pgstrom_create_data_store_row(tupdesc,
													pgstrom_chunk_size(),
													ghjs->ntups_per_page);
				pgstrom_track_object(&pds->sobj, 0);
			}
			/* insert the tuple on the data-store */
			if (!pgstrom_data_store_insert_tuple(pds, slot))
			{
				ghjs->outer_overflow = slot;
				break;
			}
		}

		if (pds)
		{
			memset(&bulkdata, 0, sizeof(pgstrom_bulkslot));
			bulkdata.pds = pds;
			bulkdata.nvalids = -1;	/* all valid */
			bulk = &bulkdata;
		}
	}
	else
	{
		/*
		 * FIXME: Right now, bulk-loading is supported only when target-list
		 * of the underlyin relation has compatible layout.
		 * It reduces the cases when we can apply bulk loding, however, it
		 * can be revised later.
		 * An idea is to fix-up target list on planner stage to fit bulk-
		 * loading.
		 */

		/* load a bunch of records at once */
		bulk = (pgstrom_bulkslot *) BulkExecProcNode(subnode);
		if (!bulk)
			ghjs->outer_done = true;
	}
	if (ghjs->pfm.enabled)
	{
		gettimeofday(&tv3, NULL);
		ghjs->pfm.time_inner_load += timeval_diff(&tv1, &tv2);
		ghjs->pfm.time_outer_load += timeval_diff(&tv2, &tv3);
	}

	/*
	 * We also need to check existence of next inner hash-chunks, even if
	 * here is no more outer records, In case of hash-table splited-out,
	 * we have to rewind the outer relation scan, then makes relations
	 * join with the next inner hash chunks.
	 */
	if (!bulk)
		goto retry;

	/*
	 * Older krowmap is no longer supported.
	 */
	if (bulk->nvalids >= 0)
		elog(ERROR, "Bulk-load with rowmap no longer supported");

	return pgstrom_create_gpuhashjoin(ghjs, bulk, result_format);
}

static bool
gpuhashjoin_next_tuple(GpuHashJoinState *ghjs)
{
	TupleTableSlot		   *ps_slot = ghjs->css.ss.ss_ScanTupleSlot;
	TupleDesc				tupdesc = ps_slot->tts_tupleDescriptor;
	pgstrom_gpuhashjoin	   *gpuhashjoin = ghjs->curr_ghjoin;
	pgstrom_data_store	   *pds_dest = gpuhashjoin->pds_dest;
	kern_data_store		   *kds_dest = pds_dest->kds;
	struct timeval			tv1, tv2;

	/*
	 * TODO: All fallback code here
	 */
	Assert(kds_dest->format == KDS_FORMAT_TUPSLOT);

	if (ghjs->pfm.enabled)
		gettimeofday(&tv1, NULL);

	while (ghjs->curr_index < kds_dest->nitems)
	{
		Datum		   *tts_values;
		cl_char		   *tts_isnull;
		int				index = ghjs->curr_index++;

		/* fetch a result tuple */
		ExecClearTuple(ps_slot);
		tts_values = KERN_DATA_STORE_VALUES(kds_dest, index);
		tts_isnull = KERN_DATA_STORE_ISNULL(kds_dest, index);
		Assert(tts_values != NULL && tts_isnull != NULL);
		memcpy(ps_slot->tts_values, tts_values,
			   sizeof(Datum) * tupdesc->natts);
		memcpy(ps_slot->tts_isnull, tts_isnull,
			   sizeof(bool) * tupdesc->natts);
		ExecStoreVirtualTuple(ps_slot);

		if (ghjs->css.ss.ps.qual != NIL)
		{
			ExprContext	   *econtext = ghjs->css.ss.ps.ps_ExprContext;

			econtext->ecxt_scantuple = ps_slot;
			if (!ExecQual(ghjs->css.ss.ps.qual, econtext, false))
				continue;	/* try to fetch next tuple */
		}

		if (ghjs->pfm.enabled)
		{
			gettimeofday(&tv2, NULL);
			ghjs->pfm.time_materialize += timeval_diff(&tv1, &tv2);
		}
		return true;
	}

	if (ghjs->pfm.enabled)
	{
		gettimeofday(&tv2, NULL);
		ghjs->pfm.time_materialize += timeval_diff(&tv1, &tv2);
	}
	ExecClearTuple(ps_slot);
	return false;
}

static pgstrom_gpuhashjoin *
pgstrom_fetch_gpuhashjoin(GpuHashJoinState *ghjs,
						  bool *needs_recheck,
						  int result_format)
{
	pgstrom_message	   *msg;
	pgstrom_gpuhashjoin *ghjoin;
	dlist_node			*dnode;

	/*
	 * Keep number of asynchronous hashjoin request a particular level,
	 * unless it does not exceed pgstrom_max_async_chunks and any new
	 * response is not replied during the loading.
	 */
	while (ghjs->num_running <= pgstrom_max_async_chunks)
	{
		pgstrom_gpuhashjoin *ghjoin
			= gpuhashjoin_load_next_chunk(ghjs, result_format);
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
		if (ghjs->num_running == 0)
			return NULL;
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
#if 0
		/* FIXME: Go to fallback case if CPUReCheck or OutOfSharedMemory */
		if (ghjoin->msg.errcode == StromError_CpuReCheck ||
			ghjoin->msg.errcode == StromError_OutOfSharedMemory)
			*needs_recheck = true;
		else
#endif
		if (ghjoin->msg.errcode == CL_BUILD_PROGRAM_FAILURE)
		{
			const char *buildlog
				= pgstrom_get_devprog_errmsg(ghjoin->dprog_key);

			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("PG-Strom: OpenCL execution error (%s)\n%s",
							pgstrom_strerror(ghjoin->msg.errcode),
							ghjs->kernel_source),
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
	else
		*needs_recheck = false;
	return ghjoin;
}

static TupleTableSlot *
gpuhashjoin_exec(CustomScanState *node)
{
	GpuHashJoinState   *ghjs = (GpuHashJoinState *) node;
	TupleTableSlot	   *ps_slot = ghjs->css.ss.ss_ScanTupleSlot;
	ProjectionInfo	   *ps_proj = ghjs->css.ss.ps.ps_ProjInfo;
	pgstrom_gpuhashjoin *ghjoin;

	while (!ghjs->curr_ghjoin || !gpuhashjoin_next_tuple(ghjs))
	{
		pgstrom_message	   *msg;

		/*
		 * Release previous hashjoin chunk that
		 * should be already fetched.
		 */
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
		 * Fetch a next hashjoin chunk already processed
		 */
		ghjoin = pgstrom_fetch_gpuhashjoin(ghjs, &ghjs->curr_recheck,
										   KDS_FORMAT_TUPSLOT);
		if (!ghjoin)
		{
			ExecClearTuple(ps_slot);
			break;
		}
		ghjs->curr_ghjoin = ghjoin;
		ghjs->curr_index = 0;
	}
	/* can valid tuple be fetched? */
	if (TupIsNull(ps_slot))
		return ps_slot;

	/* needs to apply projection? */
	if (ps_proj)
	{
		ExprContext	   *econtext = ghjs->css.ss.ps.ps_ExprContext;
		ExprDoneCond	is_done;

		econtext->ecxt_scantuple = ps_slot;
		return ExecProject(ps_proj, &is_done);
	}
	return ps_slot;
}

static void *
gpuhashjoin_exec_bulk(CustomScanState *node)
{
	GpuHashJoinState	   *ghjs = (GpuHashJoinState *) node;
	pgstrom_gpuhashjoin	   *ghjoin;
	pgstrom_data_store	   *pds;
	pgstrom_data_store	   *pds_dest;
	pgstrom_bulkslot	   *bulk = NULL;

	/* must provide our own instrumentation support */
	if (node->ss.ps.instrument)
		InstrStartNode(node->ss.ps.instrument);

	while (true)
	{
		bool		needs_rechecks;
		cl_uint		nitems;
		cl_uint		i, j;

		ghjoin = pgstrom_fetch_gpuhashjoin(ghjs, &needs_rechecks,
										   KDS_FORMAT_ROW_FLAT);
		if (!ghjoin)
			break;
		if (needs_rechecks)
		{
			/* fill up kds_dest by CPU */
			elog(ERROR, "CPU Recheck not implemented yet");
		}

		/* source kds performs as ktoast of pds_dest */
		pds = ghjoin->pds;
		pds_dest = ghjoin->pds_dest;
		Assert(pds->kds->format == KDS_FORMAT_ROW ||
			   pds->kds->format == KDS_FORMAT_ROW_FLAT);
		Assert(pds_dest->kds->format == KDS_FORMAT_ROW_FLAT);

		/* update perfmon info */
		if (ghjoin->msg.pfm.enabled)
			pgstrom_perfmon_add(&ghjs->pfm, &ghjoin->msg.pfm);

		/*
		 * Make a bulk-slot according to the result
		 */
		nitems = pds_dest->kds->nitems;
		bulk = palloc0(offsetof(pgstrom_bulkslot, rindex[nitems]));
		bulk->pds = pgstrom_get_data_store(pds_dest);
		bulk->nvalids = -1;
		pgstrom_track_object(&pds_dest->sobj, 0);

		/* No longer gpuhashjoin is referenced any more. Its pds_dest
		 * shall not be actually released because its refcnt is already
		 * incremented above
		 */
		pgstrom_untrack_object(&ghjoin->msg.sobj);
        pgstrom_put_message(&ghjoin->msg);

		/*
		 * Reduce results if host-only qualifiers
		 */
		if (node->ss.ps.qual)
		{
			ExprContext	   *econtext = ghjs->css.ss.ps.ps_ExprContext;
			TupleTableSlot *slot = ghjs->css.ss.ss_ScanTupleSlot;
			HeapTupleData	tuple;

			for (i=0, j=0; i < nitems; i++)
			{
				if (!pgstrom_fetch_data_store(slot, bulk->pds, i, &tuple))
					elog(ERROR, "Bug? unable to fetch a result slot");
				econtext->ecxt_scantuple = slot;

				if (!ghjs->css.ss.ps.qual ||
					ExecQual(ghjs->css.ss.ps.qual, econtext, false))
					bulk->rindex[j++] = i;
			}
			bulk->nvalids = j;
		}

		if ((bulk->nvalids < 0 ? nitems : bulk->nvalids) > 0)
			break;

		/* If this chunk has no valid items, it does not make sense to
		 * return upper level this chunk. So, release this data-store
		 * and tries to fetch next one.
		 */
		pgstrom_untrack_object(&bulk->pds->sobj);
		pgstrom_put_data_store(bulk->pds);
		pfree(bulk);
		bulk = NULL;
	}

	/* must provide our own instrumentation support */
    if (node->ss.ps.instrument)
	{
		if (!bulk)
			InstrStopNode(node->ss.ps.instrument, 0.0);
		else
			InstrStopNode(node->ss.ps.instrument,
						  bulk->nvalids < 0 ?
						  (double) bulk->pds->kds->nitems :
						  (double) bulk->nvalids);
	}
	return bulk;
}

static void
gpuhashjoin_end(CustomScanState *node)
{
	GpuHashJoinState	   *ghjs = (GpuHashJoinState *) node;
	pgstrom_gpuhashjoin	   *ghjoin;

	/* release asynchronous jobs */
	if (ghjs->curr_ghjoin)
	{
		ghjoin = ghjs->curr_ghjoin;
		pgstrom_untrack_object(&ghjoin->msg.sobj);
        pgstrom_put_message(&ghjoin->msg);
	}

	while (ghjs->num_running > 0)
	{
		ghjoin = (pgstrom_gpuhashjoin *)pgstrom_dequeue_message(ghjs->mqueue);
		if (!ghjoin)
			elog(ERROR, "message queue wait timeout");
		pgstrom_untrack_object(&ghjoin->msg.sobj);
        pgstrom_put_message(&ghjoin->msg);
		ghjs->num_running--;
	}

	/*
	 * clean out multiple hash tables on the portion of shared memory
	 * regison (because private memory stuff shall be released in-auto.
	 */
	if (ghjs->mhtables)
	{
		pgstrom_multihash_tables   *mhtables = ghjs->mhtables;
		pgstrom_untrack_object(&mhtables->sobj);
		multihash_put_tables(mhtables);
	}

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
gpuhashjoin_rescan(CustomScanState *node)
{
	GpuHashJoinState	   *ghjs = (GpuHashJoinState *) node;
	pgstrom_gpuhashjoin	   *ghjoin;
	pgstrom_multihash_tables *mhtables = ghjs->mhtables;

	/* release asynchronous jobs, if any */
	if (ghjs->curr_ghjoin)
	{
		ghjoin = ghjs->curr_ghjoin;
		pgstrom_untrack_object(&ghjoin->msg.sobj);
		pgstrom_put_message(&ghjoin->msg);
		ghjs->curr_ghjoin = NULL;
		ghjs->curr_index = 0;
		ghjs->curr_recheck = false;
	}

	while (ghjs->num_running > 0)
	{
		ghjoin = (pgstrom_gpuhashjoin *)pgstrom_dequeue_message(ghjs->mqueue);
		if (!ghjoin)
			elog(ERROR, "message queue wait timeout");
		pgstrom_untrack_object(&ghjoin->msg.sobj);
		pgstrom_put_message(&ghjoin->msg);
		ghjs->num_running--;
	}

	/*
	 * TODO: we may reuse inner hash table, if flat hash table (that is not
	 * divided to multiple portions) and no parameter changes.
	 * However, gpuhashjoin_load_next_chunk() releases the hash-table when
	 * our scan reached end of the scan... needs to fix up.
	 */
	if (!mhtables || mhtables->is_divided ||
		innerPlanState(ghjs)->chgParam != NULL)
	{
		ExecReScan(innerPlanState(ghjs));

		/* also, rewind the outer relation */
		ghjs->outer_done = false;
		ghjs->outer_overflow = NULL;
		ExecReScan(outerPlanState(ghjs));

		/* release the previous one */
		if (mhtables)
		{
			pgstrom_untrack_object(&mhtables->sobj);
			multihash_put_tables(mhtables);
			ghjs->mhtables = NULL;
		}
	}
	/* elsewhere, we can reuse pre-built inner multihash-tables */
}

/*
 * gpuhashjoin_remap_raw_expression
 *
 * It translate Var-nodes with raw format (pair of depth/resno) into
 * an entry on pseudo-targetlist. A pair of depth/resno is used to
 * device executable expressions (thus, not chained to custom_expr),
 * so we have to remap them on one of custom_ps_tlist.
 */
typedef struct
{
	List   *ps_tlist;
	List   *ps_depth;
	List   *ps_resno;
} gpuhashjoin_remap_raw_expression_context;

static Node *
gpuhashjoin_remap_raw_expression_mutator(Node *node, void *__context)
{
	gpuhashjoin_remap_raw_expression_context *context = __context;

	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		Var		   *varnode = (Var *) node;
		ListCell   *lc1, *lc2, *lc3;

		forthree (lc1, context->ps_tlist,
				  lc2, context->ps_depth,
				  lc3, context->ps_resno)
		{
			TargetEntry	*tle = lfirst(lc1);
			int		depth = lfirst_int(lc2);
			int		resno = lfirst_int(lc3);

			Assert(IsA(tle->expr, Var));
			if (varnode->varno == depth &&
				varnode->varattno == resno)
				return copyObject(tle->expr);
		}
		elog(ERROR, "pseudo targetlist lookup failed");
	}
	return expression_tree_mutator(node,
								   gpuhashjoin_remap_raw_expression_mutator,
								   __context);
}

static Node *
gpuhashjoin_remap_raw_expression(GpuHashJoinState *ghjs, Node *node)
{
	gpuhashjoin_remap_raw_expression_context context;
	CustomScan *cscan = (CustomScan *) ghjs->css.ss.ps.plan;

	context.ps_tlist = cscan->custom_ps_tlist;
	context.ps_depth = ghjs->ps_src_depth;
	context.ps_resno = ghjs->ps_src_resno;

	return gpuhashjoin_remap_raw_expression_mutator(node, &context);
}

static void
gpuhashjoin_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
	GpuHashJoinState   *ghjs = (GpuHashJoinState *) node;
	CustomScan		   *cscan = (CustomScan *) node->ss.ps.plan;
	GpuHashJoinInfo	   *ghj_info = deform_gpuhashjoin_info(cscan);
	StringInfoData		str;
	List			   *context;
	ListCell		   *lc1, *lc2, *lc3;
	Node			   *expr;
	char			   *temp;
	int					depth;

	initStringInfo(&str);

	/* name lookup context */
	context = set_deparse_context_planstate(es->deparse_cxt,
											(Node *) node,
											ancestors);
	/* pseudo scan tlist if verbose mode */
	if (es->verbose)
	{
		resetStringInfo(&str);
		foreach (lc1, cscan->custom_ps_tlist)
		{
			TargetEntry	   *tle = lfirst(lc1);

			temp = deparse_expression((Node *)tle->expr,
									  context, true, false);
			if (lc1 != list_head(cscan->custom_ps_tlist))
				appendStringInfo(&str, ", ");
			if (!tle->resjunk)
				appendStringInfo(&str, "%s", temp);
			else
				appendStringInfo(&str, "(%s)", temp);
		}
		ExplainPropertyText("Pseudo scan", str.data, es);
	}
	/* hash, qual and host clauses */
	depth = 1;
	forthree (lc1, ghj_info->hash_clauses,
			  lc2, ghj_info->qual_clauses,
			  lc3, ghj_info->host_clauses)
	{
		char	qlabel[80];

		/* hash clause */
		if (lfirst(lc1))
		{
			expr = gpuhashjoin_remap_raw_expression(ghjs, lfirst(lc1));
			temp = deparse_expression(expr, context, es->verbose, false);
			snprintf(qlabel, sizeof(qlabel), "Hash clause %d", depth);
			ExplainPropertyText(qlabel, temp, es);
		}
		/* qual clause */
		if (lfirst(lc2))
		{
			expr = gpuhashjoin_remap_raw_expression(ghjs, lfirst(lc2));
			temp = deparse_expression(expr, context, es->verbose, false);
			snprintf(qlabel, sizeof(qlabel), "Qual clause %d", depth);
			ExplainPropertyText(qlabel, temp, es);
		}
		/* host clause */
		if (lfirst(lc3))
		{
			expr = lfirst(lc3);
			temp = deparse_expression(expr, context, es->verbose, false);
			snprintf(qlabel, sizeof(qlabel), "Host clause %d", depth);
			ExplainPropertyText(qlabel, temp, es);
		}
		depth++;
	}
	/* outer clause if any */
	if (ghj_info->outer_quals)
	{
		expr = (Node *) ghj_info->outer_quals;
		expr = gpuhashjoin_remap_raw_expression(ghjs, expr);
		temp = deparse_expression(expr, context, es->verbose, false);
		ExplainPropertyText("Outer clause", temp, es);
	}
	ExplainPropertyText("Bulkload", ghjs->outer_bulkload ? "On" : "Off", es);

	show_device_kernel(ghjs->dprog_key, es);

	if (es->analyze && ghjs->pfm.enabled)
		pgstrom_perfmon_explain(&ghjs->pfm, es);
}

/* ----------------------------------------------------------------
 *
 * Callback routines for MultiHash node
 *
 * ---------------------------------------------------------------- */
pgstrom_multihash_tables *
multihash_get_tables(pgstrom_multihash_tables *mhtables)
{
	SpinLockAcquire(&mhtables->lock);
	Assert(mhtables->refcnt > 0);
	mhtables->refcnt++;
	SpinLockRelease(&mhtables->lock);

	return mhtables;
}

void
multihash_put_tables(pgstrom_multihash_tables *mhtables)
{
	bool	do_release = false;

	SpinLockAcquire(&mhtables->lock);
	Assert(mhtables->refcnt > 0);
	if (--mhtables->refcnt == 0)
	{
		Assert(mhtables->n_kernel == 0 && mhtables->m_hash == NULL);
		do_release = true;
	}
	SpinLockRelease(&mhtables->lock);
	if (do_release)
		pgstrom_shmem_free(mhtables);
}

static Node *
multihash_create_scan_state(CustomScan *cscan)
{
	MultiHashState *mhs = palloc0(sizeof(MultiHashState));

	NodeSetTag(mhs, T_CustomScanState);
	mhs->css.flags = cscan->flags;
	mhs->css.methods = &multihash_exec_methods.c;

	return (Node *) mhs;
}

static void
multihash_begin(CustomScanState *node, EState *estate, int eflags)
{
	MultiHashState *mhs = (MultiHashState *) node;
	CustomScan	   *cscan = (CustomScan *) node->ss.ps.plan;
	MultiHashInfo  *mh_info = deform_multihash_info(cscan);
	List		   *hash_keys = NIL;
	List		   *hash_keylen = NIL;
	List		   *hash_keybyval = NIL;
	ListCell	   *cell;

	/* check for unsupported flags */
	Assert(!(eflags & (EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK)));
	/* ensure the plan is MultiHash */
	Assert(pgstrom_plan_is_multihash((Plan *) cscan));

	mhs->depth = mh_info->depth;
	mhs->nslots = mh_info->nslots;
	mhs->nbatches_plan = mh_info->nloops;
	mhs->nbatches_exec = ((eflags & EXEC_FLAG_EXPLAIN_ONLY) != 0 ? -1 : 0);
	mhs->threshold = mh_info->threshold;
	mhs->hashtable_size = mh_info->hashtable_size;

	mhs->outer_overflow = NULL;
	mhs->outer_done = false;
	mhs->curr_chunk = NULL;

	/*
	 * initialize the expression state of hash-keys
	 */
	foreach (cell, mh_info->hash_keys)
	{
		int16		typlen;
		bool		typbyval;

		get_typlenbyval(exprType(lfirst(cell)), &typlen, &typbyval);

		hash_keys = lappend(hash_keys,
							ExecInitExpr(lfirst(cell), &mhs->css.ss.ps));
		hash_keylen = lappend_int(hash_keylen, typlen);
		hash_keybyval = lappend_int(hash_keybyval, typbyval);
	}
	mhs->hash_keys = hash_keys;
	mhs->hash_keylen = hash_keylen;
	mhs->hash_keybyval = hash_keybyval;

	/*
	 * initialize child nodes
	 */
	outerPlanState(mhs) = ExecInitNode(outerPlan(cscan), estate, eflags);
	innerPlanState(mhs) = ExecInitNode(innerPlan(cscan), estate, eflags);
}

static TupleTableSlot *
multihash_exec(CustomScanState *node)
{
	elog(ERROR, "MultiHash does not support ExecProcNode call convention");
	return NULL;
}

static bool
expand_multihash_tables(MultiHashState *mhs,
						pgstrom_multihash_tables **p_mhtables, Size consumed)
{

	pgstrom_multihash_tables *mhtables_old = *p_mhtables;
	pgstrom_multihash_tables *mhtables_new;
	Size	length_old = mhtables_old->length;
	Size	allocated;

	mhtables_new = pgstrom_shmem_alloc_alap(2 * length_old, &allocated);
	if (!mhtables_new)
		return false;	/* out of shmem, or too large to allocate */
	memcpy(mhtables_new, mhtables_old,
		   offsetof(pgstrom_multihash_tables, kern) +
		   mhtables_old->usage + consumed);

	mhtables_new->length =
		allocated - offsetof(pgstrom_multihash_tables, kern);
	mhtables_new->kern.hostptr = (hostptr_t)&mhtables_new->kern.hostptr;
	Assert(mhtables_new->length > mhtables_old->length);
	elog(INFO, "pgstrom_multihash_tables was expanded %zu (%p) => %zu (%p)",
		 mhtables_old->length, mhtables_old,
		 mhtables_new->length, mhtables_new);
	pgstrom_track_object(&mhtables_new->sobj, 0);

	pgstrom_untrack_object(&mhtables_old->sobj);
	multihash_put_tables(mhtables_old);

	/* update hashtable_size of MultiHashState */
	do {
		Assert(IsA(mhs, CustomScanState) &&
			   mhs->css.methods == &multihash_exec_methods.c);
		mhs->hashtable_size = allocated;
		mhs = (MultiHashState *) innerPlanState(mhs);
	} while (mhs);

	*p_mhtables = mhtables_new;

	return true;
}

static void
multihash_preload_khashtable(MultiHashState *mhs,
							 pgstrom_multihash_tables **p_mhtables,
							 bool scan_forward)
{
	TupleDesc		tupdesc = ExecGetResultType(outerPlanState(mhs));
	ExprContext	   *econtext = mhs->css.ss.ps.ps_ExprContext;
	int				depth = mhs->depth;
	pgstrom_multihash_tables *mhtables = *p_mhtables;
	kern_hashtable *khtable;
	kern_hashentry *hentry;
	Size			required;
	Size			consumed;
	cl_uint		   *hash_slots;
	cl_uint			ntuples = 0;
	int				attcacheoff;
	int				attalign;
	int				i;

	/* preload should be done under the MultiExec context */
	Assert(CurrentMemoryContext == mhs->css.ss.ps.state->es_query_cxt);

	/*
	 * First of all, construct a kern_hashtable on the tail of current
	 * usage pointer of mhtables.
	 */
	Assert(StromTagIs(mhtables, HashJoinTable));
	Assert(mhtables->kern.htable_offset[depth] == 0);
	Assert(mhtables->usage == LONGALIGN(mhtables->usage));
	mhtables->kern.htable_offset[depth] = mhtables->usage;

	if (!scan_forward)
	{
		Assert(mhs->curr_chunk);
		required = mhtables->usage + mhs->curr_chunk->length;
		while (required > mhs->threshold * mhtables->length)
		{
			if (!expand_multihash_tables(mhs, p_mhtables, 0))
				elog(ERROR, "No multi-hashtables expandable any more");
			mhtables = *p_mhtables;
		}
		memcpy((char *)&mhtables->kern + mhtables->usage,
			   mhs->curr_chunk,
			   mhs->curr_chunk->length);
		mhtables->usage += mhs->curr_chunk->length;
		Assert(mhtables->usage < mhtables->length);
		if (!mhs->outer_done)
			mhtables->is_divided = true;
		return;
	}

	/*
	 * Below is the case when we need to make the scan pointer advanced
	 */
	required = (mhtables->usage +
				LONGALIGN(offsetof(kern_hashtable,
								   colmeta[tupdesc->natts])) +
				LONGALIGN(sizeof(cl_uint) * mhs->nslots));
	while (required > mhs->threshold * mhtables->length)
	{
		if (!expand_multihash_tables(mhs, &mhtables, 0))
			elog(ERROR, "No multi-hashtables expandable any more");
		mhtables = *p_mhtables;
	}
	khtable = (kern_hashtable *)((char *)&mhtables->kern + mhtables->usage);
	khtable->ncols = tupdesc->natts;
	khtable->nslots = mhs->nslots;
	khtable->is_outer = false;	/* Only INNER is supported right now */

	attcacheoff = offsetof(HeapTupleHeaderData, t_bits);
	if (tupdesc->tdhasoid)
		attcacheoff += sizeof(Oid);
	attcacheoff = MAXALIGN(attcacheoff);

	for (i=0; i < tupdesc->natts; i++)
	{
		Form_pg_attribute attr = tupdesc->attrs[i];

		attalign = typealign_get_width(attr->attalign);
        if (attcacheoff > 0)
		{
			if (attr->attlen > 0)
				attcacheoff = TYPEALIGN(attalign, attcacheoff);
			else
				attcacheoff = -1;	/* no more shortcut any more */
		}
		khtable->colmeta[i].attbyval = attr->attbyval;
		khtable->colmeta[i].attalign = attalign;
		khtable->colmeta[i].attlen = attr->attlen;
		khtable->colmeta[i].attnum = attr->attnum;
		khtable->colmeta[i].attcacheoff = attcacheoff;
		if (attcacheoff >= 0)
			attcacheoff += attr->attlen;
	}
	hash_slots = KERN_HASHTABLE_SLOT(khtable);
	memset(hash_slots, 0, sizeof(cl_uint) * khtable->nslots);
	consumed = LONGALIGN((uintptr_t)&hash_slots[khtable->nslots] -
						 (uintptr_t)khtable);

	/*
	 * Nest, fill up tuples fetched from the outer relation into
	 * the hash-table in this level
	 */
	while (true)
	{
		TupleTableSlot *scan_slot;
		HeapTuple		scan_tuple;
		Size			entry_size;
		pg_crc32		hash;
		ListCell	   *lc1;
		ListCell	   *lc2;
		ListCell	   *lc3;

		if (!mhs->outer_overflow)
			scan_slot = ExecProcNode(outerPlanState(mhs));
		else
		{
			scan_slot = mhs->outer_overflow;
			mhs->outer_overflow = NULL;
		}
		if (TupIsNull(scan_slot))
		{
			mhs->outer_done = true;
			break;
		}
		scan_tuple = ExecFetchSlotTuple(scan_slot);

		/* acquire the space on buffer */
		entry_size = LONGALIGN(offsetof(kern_hashentry, htup) +
							   scan_tuple->t_len);
		required = mhtables->usage + consumed + entry_size;
		while (required > mhs->threshold * mhtables->length)
		{
			if (!expand_multihash_tables(mhs, p_mhtables, consumed))
			{
				mhs->outer_overflow = scan_slot;
				goto out;
			}
			mhtables = *p_mhtables;
			khtable = (kern_hashtable *)
				((char *)&mhtables->kern + mhtables->usage);
			hash_slots = KERN_HASHTABLE_SLOT(khtable);
		}

		/* calculation of a hash value of this entry */
		INIT_CRC32C(hash);
		econtext->ecxt_scantuple = scan_slot;
		forthree (lc1, mhs->hash_keys,
				  lc2, mhs->hash_keylen,
				  lc3, mhs->hash_keybyval)
		{
			ExprState  *clause = lfirst(lc1);
			int			keylen = lfirst_int(lc2);
			bool		keybyval = lfirst_int(lc3);
			Datum		value;
			bool		isnull;

			value = ExecEvalExpr(clause, econtext, &isnull, NULL);
			if (isnull)
				continue;
			if (keylen > 0)
			{
				if (keybyval)
					COMP_CRC32C(hash, &value, keylen);
				else
					COMP_CRC32C(hash, DatumGetPointer(value), keylen);
			}
			else
			{
				COMP_CRC32C(hash,
							VARDATA_ANY(value),
							VARSIZE_ANY_EXHDR(value));
			}
		}
		FIN_CRC32C(hash);

		/* allocation of hash entry and insert it */
		hentry = (kern_hashentry *)((char *)khtable + consumed);
		hentry->hash = hash;
		hentry->rowid = ntuples;	/* actually not used... */
		hentry->t_len = scan_tuple->t_len;
		memcpy(&hentry->htup, scan_tuple->t_data, scan_tuple->t_len);

		i = hash % khtable->nslots;
		hentry->next = hash_slots[i];
		hash_slots[i] = (cl_uint)consumed;

		/* increment buffer consumption */
		consumed += entry_size;
		/* increment number of tuples read */
		ntuples++;
	}
out:
	mhtables->ntuples += (double) ntuples;
	mhtables->usage += consumed;
	Assert(mhtables->usage < mhtables->length);
	khtable->length = consumed;
	if (mhs->curr_chunk || !mhs->outer_done)
		mhtables->is_divided = true;
	if (mhs->curr_chunk)
		pfree(mhs->curr_chunk);
	mhs->curr_chunk = pmemcpy(khtable, khtable->length);
}

static void *
multihash_exec_bulk(CustomScanState *node)
{
	MultiHashState *mhs = (MultiHashState *) node;
	MultiHashNode  *mhnode = NULL;
	PlanState	   *inner_ps;	/* underlying MultiHash, if any */
	bool			scan_forward = false;
	int				depth = mhs->depth;

	/* must provide our own instrumentation support */
	if (node->ss.ps.instrument)
		InstrStartNode(node->ss.ps.instrument);

	inner_ps = innerPlanState(mhs);
	if (inner_ps)
	{
		mhnode = (MultiHashNode *) BulkExecProcNode(inner_ps);
		if (!mhnode)
		{
			if (mhs->outer_done)
				goto out;
			ExecReScan(inner_ps);
			mhnode = (MultiHashNode *) BulkExecProcNode(inner_ps);
			if (!mhnode)
				goto out;
			scan_forward = true;
		}
		else if (!mhs->curr_chunk)
			scan_forward = true;
		Assert(mhnode);
	}
	else
	{
		/* no more deep hash-table, so create a MultiHashNode */
		pgstrom_multihash_tables *mhtables;
		int			nrels = depth;
		Size		usage;
		Size		allocated;

		if (mhs->outer_done)
			goto out;
		scan_forward = true;

		mhnode = palloc0(sizeof(MultiHashNode));
		NodeSetTag(mhnode, T_Invalid);
		mhnode->nrels = nrels;

		/* allocation of multihash_tables on shared memory */
		mhtables = pgstrom_shmem_alloc_alap(mhs->hashtable_size, &allocated);
		if (!mhtables)
			elog(ERROR, "out of shared memory");

		/* initialize multihash_tables */
		usage = STROMALIGN(offsetof(kern_multihash,
									htable_offset[nrels + 1]));
		memset(mhtables, 0, usage);

		mhtables->sobj.stag = StromTag_HashJoinTable;
		mhtables->length =
			(allocated - offsetof(pgstrom_multihash_tables, kern));
		mhtables->usage = usage;
		mhtables->ntuples = 0.0;
		SpinLockInit(&mhtables->lock);
		mhtables->refcnt = 1;
		mhtables->dindex = -1;		/* set by opencl-server */
		mhtables->n_kernel = 0;		/* set by opencl-server */
		mhtables->m_hash = NULL;	/* set by opencl-server */
		mhtables->ev_hash = NULL;	/* set by opencl-server */

		memcpy(mhtables->kern.pg_crc32_table,
			   pg_crc32c_table,
			   sizeof(uint32) * 256);
		mhtables->kern.hostptr = (hostptr_t) &mhtables->kern.hostptr;
		mhtables->kern.ntables = nrels;
		memset(mhtables->kern.htable_offset, 0, sizeof(cl_uint) * (nrels + 1));
		pgstrom_track_object(&mhtables->sobj, 0);

		mhnode->mhtables = mhtables;
	}
	/*
	 * construct a kernel hash-table that stores all the inner-keys
	 * in this level, being loaded from the outer relation
	 */
	multihash_preload_khashtable(mhs, &mhnode->mhtables, scan_forward);
out:
	/* must provide our own instrumentation support */
	if (node->ss.ps.instrument)
		InstrStopNode(node->ss.ps.instrument,
					  !mhnode ? 0.0 : mhnode->mhtables->ntuples);
	if (mhnode)
		mhs->nbatches_exec++;

	return mhnode;
}

static void
multihash_end(CustomScanState *node)
{
	/*
	 * shut down the subplan
	 */
	ExecEndNode(outerPlanState(node));
	ExecEndNode(innerPlanState(node));
}

static void
multihash_rescan(CustomScanState *node)
{
	MultiHashState *mhs = (MultiHashState *) node;

	if (innerPlanState(node))
		ExecReScan(innerPlanState(node));
	ExecReScan(outerPlanState(node));

	if (mhs->curr_chunk)
		pfree(mhs->curr_chunk);
	mhs->curr_chunk = NULL;
	mhs->outer_done = false;
	mhs->outer_overflow = NULL;
}

static void
multihash_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
	CustomScan	   *cscan = (CustomScan *)node->ss.ps.plan;
	MultiHashState *mhs = (MultiHashState *) node;
	MultiHashInfo  *mh_info = deform_multihash_info(cscan);
	StringInfoData	str;
	List		   *context;
	ListCell	   *cell;

	/* set up deparsing context */
	context = set_deparse_context_planstate(es->deparse_cxt,
											(Node *) node,
											ancestors);
	/* shows hash keys */
	initStringInfo(&str);
	foreach (cell, mh_info->hash_keys)
	{
		char   *exprstr;

		if (cell != list_head(mh_info->hash_keys))
			appendStringInfo(&str, ", ");

		exprstr = deparse_expression(lfirst(cell),
									 context,
									 es->verbose,
									 false);
		appendStringInfo(&str, "%s", exprstr);
		pfree(exprstr);
	}
    ExplainPropertyText("hash keys", str.data, es);

	/* shows hash parameters */
	if (es->format != EXPLAIN_FORMAT_TEXT)
	{
		resetStringInfo(&str);
		if (mhs->nbatches_exec >= 0)
			ExplainPropertyInteger("nBatches", mhs->nbatches_exec, es);
		else
			ExplainPropertyInteger("nBatches", mhs->nbatches_plan, es);
		ExplainPropertyInteger("Buckets", mh_info->nslots, es);
		appendStringInfo(&str, "%.2f%%", 100.0 * mh_info->threshold);
		ExplainPropertyText("Memory Usage", str.data, es);
	}
	else
	{
		appendStringInfoSpaces(es->str, es->indent * 2);
		appendStringInfo(es->str,
						 "nBatches: %u  Buckets: %u  Memory Usage: %.2f%%\n",
						 mhs->nbatches_exec >= 0
						 ? mhs->nbatches_exec
						 : mhs->nbatches_plan,
						 mh_info->nslots,
						 100.0 * mh_info->threshold);
	}
}

/*
 * pgstrom_init_gpuhashjoin
 *
 * a startup routine to initialize gpuhashjoin.c
 */
void
pgstrom_init_gpuhashjoin(void)
{
	/* enable_gpuhashjoin parameter */
	DefineCustomBoolVariable("enable_gpuhashjoin",
							 "Enables the use of GPU accelerated hash-join",
							 NULL,
							 &enable_gpuhashjoin,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);

	/* setup path methods */
	gpuhashjoin_path_methods.CustomName = "GpuHashJoin";
	gpuhashjoin_path_methods.PlanCustomPath     = create_gpuhashjoin_plan;
	gpuhashjoin_path_methods.TextOutCustomPath	= gpuhashjoin_textout_path;

	/* setup plan methods */
	gpuhashjoin_plan_methods.CustomName = "GpuHashJoin";
	gpuhashjoin_plan_methods.CreateCustomScanState
		= gpuhashjoin_create_scan_state;
	gpuhashjoin_plan_methods.TextOutCustomScan  = NULL;

	/* setup exec methods */
	gpuhashjoin_exec_methods.c.CustomName = "GpuHashJoin";
	gpuhashjoin_exec_methods.c.BeginCustomScan    = gpuhashjoin_begin;
	gpuhashjoin_exec_methods.c.ExecCustomScan     = gpuhashjoin_exec;
	gpuhashjoin_exec_methods.c.EndCustomScan      = gpuhashjoin_end;
	gpuhashjoin_exec_methods.c.ReScanCustomScan   = gpuhashjoin_rescan;
	gpuhashjoin_exec_methods.c.MarkPosCustomScan  = NULL;
	gpuhashjoin_exec_methods.c.RestrPosCustomScan = NULL;
	gpuhashjoin_exec_methods.c.ExplainCustomScan  = gpuhashjoin_explain;
	gpuhashjoin_exec_methods.ExecCustomBulk       = gpuhashjoin_exec_bulk;

	/* setup plan methods of MultiHash */
	multihash_plan_methods.CustomName          = "MultiHash";
	multihash_plan_methods.CreateCustomScanState
		= multihash_create_scan_state;
	multihash_plan_methods.TextOutCustomScan   = NULL;

	/* setup exec methods of MultiHash */
	multihash_exec_methods.c.CustomName        = "MultiHash";
	multihash_exec_methods.c.BeginCustomScan   = multihash_begin;
	multihash_exec_methods.c.ExecCustomScan    = multihash_exec;
	multihash_exec_methods.c.EndCustomScan     = multihash_end;
	multihash_exec_methods.c.ReScanCustomScan  = multihash_rescan;
	multihash_exec_methods.c.MarkPosCustomScan = NULL;
	multihash_exec_methods.c.RestrPosCustomScan= NULL;
	multihash_exec_methods.c.ExplainCustomScan = multihash_explain;
	multihash_exec_methods.ExecCustomBulk      = multihash_exec_bulk;

	/* hook registration */
	set_join_pathlist_next = set_join_pathlist_hook;
	set_join_pathlist_hook = gpuhashjoin_add_join_path;
}

/* ----------------------------------------------------------------
 *
 * NOTE: below is the code being run on OpenCL server context
 *
 * ---------------------------------------------------------------- */

typedef struct
{
	pgstrom_gpuhashjoin *gpuhashjoin;
	cl_command_queue kcmdq;
	cl_program		program;
	cl_kernel		kern_main;
	cl_kernel		kern_proj;
	cl_mem			m_join;
	cl_mem			m_hash;
	cl_mem			m_dstore;
	cl_mem			m_ktoast;
	cl_mem			m_rowmap;
	cl_mem			m_kresult;
	cl_int			dindex;
	bool			hash_loader;/* true, if this context loads hash table */
	cl_uint			ev_kern_main;	/* event index of kern_main */
	cl_uint			ev_kern_proj;	/* event index of kern_proj */
	cl_uint			ev_index;
	cl_event		events[30];
} clstate_gpuhashjoin;

static void
clserv_respond_hashjoin(cl_event event, cl_int ev_status, void *private)
{
	clstate_gpuhashjoin	*clghj = (clstate_gpuhashjoin *) private;
	pgstrom_gpuhashjoin *gpuhashjoin = clghj->gpuhashjoin;
	pgstrom_multihash_tables *mhtables = gpuhashjoin->mhtables;
	kern_resultbuf		*kresults
		= KERN_HASHJOIN_RESULTBUF(&gpuhashjoin->khashjoin);

	if (ev_status == CL_COMPLETE)
		gpuhashjoin->msg.errcode = kresults->errcode;
	else
	{
		clserv_log("unexpected CL_EVENT_COMMAND_EXECUTION_STATUS: %d",
				   ev_status);
		gpuhashjoin->msg.errcode = StromError_OpenCLInternal;
	}

	/* collect performance statistics */
	if (gpuhashjoin->msg.pfm.enabled)
	{
		pgstrom_perfmon *pfm = &gpuhashjoin->msg.pfm;
		cl_ulong	tv_start;
		cl_ulong	tv_end;
		cl_ulong	temp;
		cl_int		i, rc;

		/* Time to load hash-tables should be counted on the context that
		 * actually kicked DMA send request only.
		 */
		if (clghj->hash_loader)
		{
			rc = clGetEventProfilingInfo(clghj->events[0],
										 CL_PROFILING_COMMAND_START,
										 sizeof(cl_ulong),
										 &tv_start,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			rc = clGetEventProfilingInfo(clghj->events[0],
										 CL_PROFILING_COMMAND_END,
										 sizeof(cl_ulong),
										 &tv_end,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			pfm->time_dma_send += (tv_end - tv_start) / 1000;
		}

		/*
		 * DMA send time of hashjoin headers and row-/column-store
		 */
		tv_start = ~0;
		tv_end = 0;
		for (i=1; i < clghj->ev_kern_main; i++)
		{
			rc = clGetEventProfilingInfo(clghj->events[i],
                                         CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong),
                                         &temp,
                                         NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			tv_start = Min(tv_start, temp);

			rc = clGetEventProfilingInfo(clghj->events[i],
										 CL_PROFILING_COMMAND_END,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			tv_end = Max(tv_end, temp);
		}
		pfm->time_dma_send += (tv_end - tv_start) / 1000;

		/*
		 * Main kernel execution time
		 */
		rc = clGetEventProfilingInfo(clghj->events[clghj->ev_kern_main],
									 CL_PROFILING_COMMAND_START,
									 sizeof(cl_ulong),
									 &tv_start,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;

		rc = clGetEventProfilingInfo(clghj->events[clghj->ev_kern_main],
									 CL_PROFILING_COMMAND_END,
									 sizeof(cl_ulong),
									 &tv_end,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;
		pfm->time_kern_exec += (tv_end - tv_start) / 1000;

		/*
		 * Projection kernel execution time
		 */
		rc = clGetEventProfilingInfo(clghj->events[clghj->ev_kern_proj],
									 CL_PROFILING_COMMAND_START,
									 sizeof(cl_ulong),
									 &tv_start,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;

		rc = clGetEventProfilingInfo(clghj->events[clghj->ev_kern_proj],
									 CL_PROFILING_COMMAND_END,
									 sizeof(cl_ulong),
									 &tv_end,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;
		pfm->time_kern_proj += (tv_end - tv_start) / 1000;

		/*
		 * DMA recv time
		 */
		tv_start = ~0;
		tv_end = 0;
		for (i=clghj->ev_kern_proj + 1; i < clghj->ev_index; i++)
		{
			rc = clGetEventProfilingInfo(clghj->events[i],
										 CL_PROFILING_COMMAND_START,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			tv_start = Min(tv_start, temp);

			rc = clGetEventProfilingInfo(clghj->events[i],
										 CL_PROFILING_COMMAND_END,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			tv_end = Max(tv_end, temp);
		}
		pfm->time_dma_recv += (tv_end - tv_start) / 1000;

	skip_perfmon:
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clGetEventProfilingInfo (%s)",
					   opencl_strerror(rc));
			pfm->enabled = false;	/* turn off profiling */
		}
	}

	/*
	 * release opencl resources
	 *
	 * NOTE: The first event object (a.k.a hjtable->ev_hash) and memory
	 * object of hash table (a.k.a hjtable->m_hash) has to be released
	 * under the hjtable->lock
	 */
	while (clghj->ev_index > 1)
		clReleaseEvent(clghj->events[--clghj->ev_index]);
	if (clghj->m_kresult)
		clReleaseMemObject(clghj->m_kresult);
	if (clghj->m_rowmap)
		clReleaseMemObject(clghj->m_rowmap);
	if (clghj->m_ktoast)
		clReleaseMemObject(clghj->m_ktoast);
	if (clghj->m_dstore)
		clReleaseMemObject(clghj->m_dstore);
	if (clghj->m_join)
		clReleaseMemObject(clghj->m_join);
	if (clghj->kern_main)
		clReleaseKernel(clghj->kern_main);
	if (clghj->kern_proj)
		clReleaseKernel(clghj->kern_proj);
	if (clghj->program)
		clReleaseProgram(clghj->program);

	/* Unload hashjoin-table, if no longer referenced */
	SpinLockAcquire(&mhtables->lock);
	Assert(mhtables->n_kernel > 0);
	clReleaseMemObject(mhtables->m_hash);
	clReleaseEvent(mhtables->ev_hash);
	if (--mhtables->n_kernel == 0)
	{
		mhtables->m_hash = NULL;
		mhtables->ev_hash = NULL;
	}
	SpinLockRelease(&mhtables->lock);	
	free(clghj);

	/*
	 * A hash-join operation may produce unpredicated number of rows;
	 * larger than capability of kern_resultbuf being allocated in-
	 * advance. In this case, kernel code returns the error code of
	 * StromError_DataStoreNoSpace, so we try again with larger result-
	 * buffer.
	 */
	if (gpuhashjoin->msg.errcode == StromError_DataStoreNoSpace)
	{
		/*
		 * Expand the result buffer then retry, if rough estimation didn't
		 * give enough space to store the result
		 */
		pgstrom_data_store *old_pds = gpuhashjoin->pds_dest;
		pgstrom_data_store *new_pds;
		kern_data_store	   *old_kds = old_pds->kds;
		kern_data_store	   *new_kds;
		kern_resultbuf	   *kresults;
		cl_uint				ncols = old_kds->ncols;
		cl_uint				nitems = old_kds->nitems;
		Size				head_len;
		Size				required;

		/* adjust kern_resultbuf */
		kresults = KERN_HASHJOIN_RESULTBUF(&gpuhashjoin->khashjoin);
		clserv_log("GHJ input kresults (%u=>%u)", kresults->nrooms, nitems);
		kresults->nrooms = nitems;
		kresults->nitems = 0;
		kresults->errcode = StromError_Success;

		head_len = STROMALIGN(offsetof(kern_data_store, colmeta[ncols]));
		if (old_kds->format == KDS_FORMAT_TUPSLOT)
		{
			clserv_log("GHJ input again (nrooms: %u => %u)",
					   old_kds->nrooms, nitems);
			required = STROMALIGN(head_len +
								  (LONGALIGN(sizeof(Datum) * ncols) +
								   LONGALIGN(sizeof(bool) * ncols)) * nitems);
			new_kds = pgstrom_shmem_alloc(required);
			if (!new_kds)
			{
				gpuhashjoin->msg.errcode = StromError_OutOfSharedMemory;
				pgstrom_reply_message(&gpuhashjoin->msg);
				return;
			}
			memcpy(new_kds, old_kds, head_len);
			new_kds->hostptr = (hostptr_t) &new_kds->hostptr;
			new_kds->length = required;
			new_kds->usage = 0;
			new_kds->nrooms = nitems;
			new_kds->nitems = 0;
		}
		else if (old_kds->format == KDS_FORMAT_ROW_FLAT)
		{
			clserv_log("GHJ input again (length: %u => %u)",
					   old_kds->length, old_kds->usage);
			required = head_len +
				STROMALIGN(sizeof(kern_blkitem) * old_kds->maxblocks) +
				STROMALIGN(sizeof(kern_rowitem) * nitems) +
				STROMALIGN(old_kds->usage);
			new_kds = pgstrom_shmem_alloc(required);
			if (!new_kds)
			{
				gpuhashjoin->msg.errcode = StromError_OutOfSharedMemory;
				pgstrom_reply_message(&gpuhashjoin->msg);
				return;
			}
			memcpy(new_kds, old_kds, head_len);
			new_kds->hostptr = (hostptr_t) &new_kds->hostptr;
			new_kds->length = required;
			new_kds->usage = 0;
			new_kds->nrooms = (required - head_len) / sizeof(kern_rowitem);
			new_kds->nitems = 0;
		}
		else
		{
			gpuhashjoin->msg.errcode = StromError_DataStoreCorruption;
			pgstrom_reply_message(&gpuhashjoin->msg);
			return;
		}
		/* allocate a new pgstrom_data_store */
		new_pds = pgstrom_shmem_alloc(sizeof(pgstrom_data_store));
		if (!new_pds)
		{
			pgstrom_shmem_free(new_kds);
			gpuhashjoin->msg.errcode = StromError_OutOfSharedMemory;
            pgstrom_reply_message(&gpuhashjoin->msg);
            return;
		}
		memset(new_pds, 0, sizeof(pgstrom_data_store));
		new_pds->sobj.stag = StromTag_DataStore;
		SpinLockInit(&new_pds->lock);
		new_pds->refcnt = 1;
		new_pds->kds = new_kds;

		/* replace an old pds by new pds */
		gpuhashjoin->pds_dest = new_pds;
		pgstrom_put_data_store(old_pds);

		/* retry gpuhashjoin with larger result buffer */
		pgstrom_enqueue_message(&gpuhashjoin->msg);
		return;
	}
	/* otherwise, hash-join is successfully done */
	pgstrom_reply_message(&gpuhashjoin->msg);
}

static void
clserv_process_gpuhashjoin(pgstrom_message *message)
{
	pgstrom_gpuhashjoin *gpuhashjoin = (pgstrom_gpuhashjoin *) message;
	pgstrom_multihash_tables *mhtables = gpuhashjoin->mhtables;
	pgstrom_data_store *pds = gpuhashjoin->pds;
	pgstrom_data_store *pds_dest = gpuhashjoin->pds_dest;
	kern_data_store	   *kds = pds->kds;
	kern_data_store	   *kds_dest = pds_dest->kds;
	clstate_gpuhashjoin	*clghj = NULL;
	kern_row_map	   *krowmap;
	kern_resultbuf	   *kresults;
	size_t				nitems;
	size_t				gwork_sz;
	size_t				lwork_sz;
	Size				offset;
	Size				length;
	void			   *dmaptr;
	cl_int				rc;

	Assert(StromTagIs(gpuhashjoin, GpuHashJoin));
	Assert(StromTagIs(gpuhashjoin->mhtables, HashJoinTable));
	Assert(StromTagIs(gpuhashjoin->pds, DataStore));
	krowmap = KERN_HASHJOIN_ROWMAP(&gpuhashjoin->khashjoin);
	kresults = KERN_HASHJOIN_RESULTBUF(&gpuhashjoin->khashjoin);

	/* state object of gpuhashjoin */
	clghj = calloc(1, (sizeof(clstate_gpuhashjoin) +
					   sizeof(cl_event) * kds->nblocks));
	if (!clghj)
	{
		rc = CL_OUT_OF_HOST_MEMORY;
		goto error;
	}
	clghj->gpuhashjoin = gpuhashjoin;

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
	clghj->program = clserv_lookup_device_program(gpuhashjoin->dprog_key,
                                                  &gpuhashjoin->msg);
    if (!clghj->program)
    {
        free(clghj);
		return;	/* message is in waitq, being retried later */
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
	SpinLockAcquire(&mhtables->lock);
	if (mhtables->n_kernel == 0)
	{
		int		dindex;

		Assert(!mhtables->m_hash && !mhtables->ev_hash);

		dindex = pgstrom_opencl_device_schedule(&gpuhashjoin->msg);
		mhtables->dindex = dindex;
		clghj->dindex = dindex;
		clghj->kcmdq = opencl_cmdq[dindex];
		clghj->m_hash = clCreateBuffer(opencl_context,
                                       CL_MEM_READ_WRITE,
									   mhtables->length,
									   NULL,
									   &rc);
		if (rc != CL_SUCCESS)
		{
			SpinLockRelease(&mhtables->lock);
			goto error;
		}

		rc = clEnqueueWriteBuffer(clghj->kcmdq,
								  clghj->m_hash,
								  CL_FALSE,
								  0,
                                  mhtables->length,
								  &mhtables->kern,
								  0,
								  NULL,
								  &clghj->events[clghj->ev_index]);
		if (rc != CL_SUCCESS)
		{
			clReleaseMemObject(clghj->m_hash);
			clghj->m_hash = NULL;
			SpinLockRelease(&mhtables->lock);
			goto error;
        }
		mhtables->m_hash = clghj->m_hash;
		mhtables->ev_hash = clghj->events[clghj->ev_index];
		clghj->ev_index++;
		clghj->hash_loader = true;
		gpuhashjoin->msg.pfm.bytes_dma_send += mhtables->length;
		gpuhashjoin->msg.pfm.num_dma_send++;
	}
	else
	{
		Assert(mhtables->m_hash && mhtables->ev_hash);
		rc = clRetainMemObject(mhtables->m_hash);
		Assert(rc == CL_SUCCESS);
		rc = clRetainEvent(mhtables->ev_hash);
		Assert(rc == CL_SUCCESS);

		clghj->dindex = mhtables->dindex;
		clghj->kcmdq = opencl_cmdq[clghj->dindex];
		clghj->m_hash = mhtables->m_hash;
		clghj->events[clghj->ev_index++] = mhtables->ev_hash;
	}
	mhtables->n_kernel++;
	SpinLockRelease(&mhtables->lock);

	/*
	 * __kernel void
	 * kern_gpuhashjoin_main(__global kern_hashjoin *khashjoin,
	 *                        __global kern_multihash *kmhash,
	 *                        __global kern_data_store *kds,
	 *                        __global kern_data_store *ktoast,
	 *                        KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
	 */
	clghj->kern_main = clCreateKernel(clghj->program,
									  "kern_gpuhashjoin_main",
									  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		goto error;
	}

	/*
	 * __kernel void
	 * kern_gpuhashjoin_projection(__global kern_hashjoin *khashjoin,
	 *                             __global kern_multihash *kmhash,
	 *                             __global kern_data_store *kds,
	 *                             __global kern_data_store *ktoast,
	 *                             __global kern_data_store *kds_dest,
	 *                             KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
	 */
	if (pds_dest->kds->format == KDS_FORMAT_TUPSLOT)
		clghj->kern_proj = clCreateKernel(clghj->program,
										  "kern_gpuhashjoin_projection_slot",
										  &rc);
	else if (pds_dest->kds->format == KDS_FORMAT_ROW_FLAT)
		clghj->kern_proj = clCreateKernel(clghj->program,
										  "kern_gpuhashjoin_projection_row",
										  &rc);
	else
	{
		clserv_log("pds_dest has unexpected format");
		goto error;
	}
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		goto error;
	}

	/* buffer object of __global kern_hashjoin *khashjoin */
	length = (KERN_HASHJOIN_PARAMBUF_LENGTH(&gpuhashjoin->khashjoin) +
			  KERN_HASHJOIN_RESULTBUF_LENGTH(&gpuhashjoin->khashjoin) +
			  sizeof(cl_int) * kresults->nrels * kresults->nrooms);
	clghj->m_join = clCreateBuffer(opencl_context,
								   CL_MEM_READ_WRITE,
								   length,
								   NULL,
								   &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
        goto error;
	}

	/* buffer object of __global kern_data_store *kds */
	clghj->m_dstore = clCreateBuffer(opencl_context,
									 CL_MEM_READ_WRITE,
									 KERN_DATA_STORE_LENGTH(kds),
									 NULL,
									 &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error;
	}

	/* buffer object of __global kern_data_store *ktoast, if needed */
	if (!pds->ktoast)
		clghj->m_ktoast = NULL;
	else
	{
		pgstrom_data_store *ktoast = pds->ktoast;

		clghj->m_ktoast = clCreateBuffer(opencl_context,
										 CL_MEM_READ_WRITE,
										 KERN_DATA_STORE_LENGTH(ktoast->kds),
										 NULL,
										 &rc);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
			goto error;
		}
	}

	/* buffer object of __global kern_row_map *krowmap */
	if (krowmap->nvalids < 0)
		clghj->m_rowmap = NULL;
	else
	{
		length = STROMALIGN(offsetof(kern_row_map,
									 rindex[krowmap->nvalids]));
		clghj->m_rowmap = clCreateBuffer(opencl_context,
                                         CL_MEM_READ_WRITE,
										 length,
										 NULL,
										 &rc);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
			goto error;
		}
	}

	/* buffer object of __global kern_data_store *kds_dest */
	clghj->m_kresult = clCreateBuffer(opencl_context,
									  CL_MEM_READ_WRITE,
									  STROMALIGN(kds_dest->length),
									  NULL,
									  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error;
	}

	/*
	 * OK, all the device memory and kernel objects are successfully
	 * constructed. Let's enqueue DMA send/recv and kernel invocations.
	 */

	/* Enqueue DMA send of kern_hashjoin */
	dmaptr = KERN_HASHJOIN_DMA_SENDPTR(&gpuhashjoin->khashjoin);
	offset = KERN_HASHJOIN_DMA_SENDOFS(&gpuhashjoin->khashjoin);
	length = KERN_HASHJOIN_DMA_SENDLEN(&gpuhashjoin->khashjoin);
	rc = clEnqueueWriteBuffer(clghj->kcmdq,
							  clghj->m_join,
							  CL_FALSE,
							  offset,
							  length,
							  dmaptr,
							  0,
							  NULL,
							  &clghj->events[clghj->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
        goto error;
	}
	clghj->ev_index++;
    gpuhashjoin->msg.pfm.bytes_dma_send += length;
    gpuhashjoin->msg.pfm.num_dma_send++;

	/*
	 * Enqueue DMA send of kern_rowmap, if any
	 */
	if (clghj->m_rowmap)
	{
		length = STROMALIGN(offsetof(kern_row_map,
									 rindex[krowmap->nvalids]));
		rc = clEnqueueWriteBuffer(clghj->kcmdq,
								  clghj->m_rowmap,
								  CL_FALSE,
								  0,
								  length,
								  krowmap,
								  0,
								  NULL,
								  &clghj->events[clghj->ev_index]);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
			goto error;
		}
		clghj->ev_index++;
		gpuhashjoin->msg.pfm.bytes_dma_send += length;
		gpuhashjoin->msg.pfm.num_dma_send++;
	}

	/*
	 * Enqueue DMA send of kern_data_store
	 * according to the type of data store
	 */
	rc = clserv_dmasend_data_store(pds,
								   clghj->kcmdq,
								   clghj->m_dstore,
								   clghj->m_ktoast,
								   0,
								   NULL,
								   &clghj->ev_index,
								   clghj->events,
								   &gpuhashjoin->msg.pfm);
	if (rc != CL_SUCCESS)
		goto error;

	/*
	 * Enqueue DMA send of destination kern_data_store
	 */
	length = STROMALIGN(offsetof(kern_data_store, colmeta[kds_dest->ncols]));
	rc = clEnqueueWriteBuffer(clghj->kcmdq,
                              clghj->m_kresult,
							  CL_FALSE,
							  0,
							  length,
							  kds_dest,
							  0,
							  NULL,
							  &clghj->events[clghj->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error;
	}
	clghj->ev_index++;
	gpuhashjoin->msg.pfm.bytes_dma_send += length;
	gpuhashjoin->msg.pfm.num_dma_send++;

	/*
	 * __kernel void
	 * kern_gpuhashjoin_main(__global kern_hashjoin *khashjoin,
	 *                       __global kern_multihash *kmhash,
	 *                       __global kern_data_store *kds,
	 *                       __global kern_data_store *ktoast,
	 *                       __global kern_row_map   *krowmap,
	 *                       KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
	 */

	/* Get an optimal workgroup-size of this kernel */
	nitems = (krowmap->nvalids < 0 ? kds->nitems : krowmap->nvalids);
	if (!clserv_compute_workgroup_size(&gwork_sz, &lwork_sz,
									   clghj->kern_main,
									   clghj->dindex,
									   true,	/* larger is better? */
									   nitems,
									   sizeof(cl_uint)))
		goto error;

	rc = clSetKernelArg(clghj->kern_main,
						0,	/* __global kern_hashjoin *khashjoin */
						sizeof(cl_mem),
						&clghj->m_join);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clSetKernelArg(clghj->kern_main,
						1,	/* __global kern_multihash *kmhash */
						sizeof(cl_mem),
						&clghj->m_hash);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clSetKernelArg(clghj->kern_main,
						2,	/* __global kern_data_store *kds */
						sizeof(cl_mem),
						&clghj->m_dstore);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clSetKernelArg(clghj->kern_main,
						3,	/*  __global kern_data_store *ktoast */
						sizeof(cl_mem),
						&clghj->m_ktoast);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clSetKernelArg(clghj->kern_main,
						4,	/*  __global kern_row_map *krowmap */
						sizeof(cl_mem),
						&clghj->m_rowmap);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clSetKernelArg(clghj->kern_main,
						5,	/* KERN_DYNAMIC_LOCAL_WORKMEM_ARG */
						sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clEnqueueNDRangeKernel(clghj->kcmdq,
								clghj->kern_main,
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
		clserv_log("gwork_sz=%zu lwork_sz=%zu", gwork_sz, lwork_sz);
		goto error;
	}
	clghj->ev_kern_main = clghj->ev_index;
	clghj->ev_index++;
	gpuhashjoin->msg.pfm.num_kern_exec++;

	/*
	 * __kernel void
	 * kern_gpuhashjoin_projection(__global kern_hashjoin *khashjoin,
	 *                             __global kern_multihash *kmhash,
	 *                             __global kern_data_store *kds,
	 *                             __global kern_data_store *ktoast,
	 *                             __global kern_data_store *kds_dest,
	 *                             KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
	 */

	/* Get an optimal workgroup-size of this kernel */
	if (!clserv_compute_workgroup_size(&gwork_sz, &lwork_sz,
									   clghj->kern_proj,
									   clghj->dindex,
									   false,   /* smaller is better */
									   kds_dest->nrooms,
									   sizeof(cl_uint)))
		goto error;

	rc = clSetKernelArg(clghj->kern_proj,
						0,	/* __global kern_hashjoin *khashjoin */
						sizeof(cl_mem),
						&clghj->m_join);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clSetKernelArg(clghj->kern_proj,
						1,	/* __global kern_multihash *kmhash */
						sizeof(cl_mem),
						&clghj->m_hash);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clSetKernelArg(clghj->kern_proj,
						2,	/* __global kern_data_store *kds */
						sizeof(cl_mem),
						&clghj->m_dstore);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clSetKernelArg(clghj->kern_proj,
						3,	/*  __global kern_data_store *ktoast */
						sizeof(cl_mem),
						&clghj->m_ktoast);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clSetKernelArg(clghj->kern_proj,
						4,	/* __global kern_data_store *kds_dest */
						sizeof(cl_mem),
						&clghj->m_kresult);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clSetKernelArg(clghj->kern_proj,
						5,	/* __local void *local_workbuf */
						sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error;
	}

	rc = clEnqueueNDRangeKernel(clghj->kcmdq,
								clghj->kern_proj,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								1,
								&clghj->events[clghj->ev_index - 1],
								&clghj->events[clghj->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		goto error;
	}
	clghj->ev_kern_proj = clghj->ev_index;
	clghj->ev_index++;
	gpuhashjoin->msg.pfm.num_kern_proj++;

	/*
	 * Write back result status
	 */
	dmaptr = KERN_HASHJOIN_DMA_RECVPTR(&gpuhashjoin->khashjoin);
	offset = KERN_HASHJOIN_DMA_RECVOFS(&gpuhashjoin->khashjoin);
	length = KERN_HASHJOIN_DMA_RECVLEN(&gpuhashjoin->khashjoin);
	rc = clEnqueueReadBuffer(clghj->kcmdq,
							 clghj->m_join,
							 CL_FALSE,
							 offset,
							 length,
							 dmaptr,
							 1,
							 &clghj->events[clghj->ev_index - 1],
							 &clghj->events[clghj->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueReadBuffer: %s",
				   opencl_strerror(rc));
		goto error;
	}
	clghj->ev_index++;
    gpuhashjoin->msg.pfm.bytes_dma_recv += length;
    gpuhashjoin->msg.pfm.num_dma_recv++;

	/*
	 * Write back projection data-store
	 */
	rc = clEnqueueReadBuffer(clghj->kcmdq,
							 clghj->m_kresult,
							 CL_FALSE,
							 0,
							 kds_dest->length,
							 kds_dest,
							 1,
							 &clghj->events[clghj->ev_index - 1],
							 &clghj->events[clghj->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueReadBuffer: %s",
				   opencl_strerror(rc));
		goto error;
	}
	clghj->ev_index++;
	gpuhashjoin->msg.pfm.bytes_dma_recv += kds_dest->length;
	gpuhashjoin->msg.pfm.num_dma_recv++;

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
		goto error;
	}
	return;

error:
	if (clghj)
	{
		if (clghj->ev_index > 0)
		{
			clWaitForEvents(clghj->ev_index, clghj->events);
			/* NOTE: first event has to be released under mhtables->lock */
			while (clghj->ev_index > 1)
				clReleaseEvent(clghj->events[--clghj->ev_index]);
		}
		if (clghj->m_kresult)
			clReleaseMemObject(clghj->m_kresult);
		if (clghj->m_ktoast)
			clReleaseMemObject(clghj->m_ktoast);
		if (clghj->m_dstore)
			clReleaseMemObject(clghj->m_dstore);
		if (clghj->m_join)
			clReleaseMemObject(clghj->m_join);
		if (clghj->m_hash)
		{
			SpinLockAcquire(&mhtables->lock);
			Assert(mhtables->n_kernel > 0);
			clReleaseMemObject(mhtables->m_hash);
			clReleaseEvent(mhtables->ev_hash);
			if (--mhtables->n_kernel == 0)
			{
				mhtables->m_hash = NULL;
				mhtables->ev_hash = NULL;
			}
			SpinLockRelease(&mhtables->lock);
		}
		if (clghj->kern_main)
			clReleaseKernel(clghj->kern_main);
		if (clghj->kern_proj)
			clReleaseKernel(clghj->kern_proj);
		if (clghj->program && clghj->program != BAD_OPENCL_PROGRAM)
			clReleaseProgram(clghj->program);
		free(clghj);
	}
	gpuhashjoin->msg.errcode = rc;
	pgstrom_reply_message(&gpuhashjoin->msg);
}
