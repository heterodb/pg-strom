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
#include "storage/latch.h"
#include "storage/proc.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/pg_crc.h"
#include "utils/ruleutils.h"
#include "utils/selfuncs.h"
#include "pg_strom.h"
#include "cuda_hashjoin.h"
#include "cuda_numeric.h"

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
 * multiple-hashatables
 */
typedef struct
{
	Size			length;		/* max available length of this mhash */
	Size			usage;		/* current usage of this mhash */
	cl_ulong		ntuples;	/* number of tuples in this mhash */
	bool			is_divided;	/* true, if not whole of the inner relation */
	cl_int			refcnt;		/* reference counter */
	CUdeviceptr	   *m_hash;		/* Gpu memory for each cuda context */
	CUevent		   *ev_loaded;	/* Sync object for each cuda context *
								 * NULL means DMA already kicked */
	kern_multihash	kern;		/* in-kernel structure; to be sent */
} pgstrom_multihash_tables;

/*
 * task for each gpuhashjoin request
 */
typedef struct
{
	GpuTask			task;
	CUfunction		kern_main;
	void		   *kern_main_args[5];
	CUfunction		kern_proj;
	void		   *kern_proj_args[5];
	CUdeviceptr		m_join;
	CUdeviceptr		m_kds_src;
	CUdeviceptr		m_kds_dst;
	bool			hash_loader; /* true, if this context loads hash table */
	CUevent			ev_dma_send_start;
	CUevent			ev_dma_send_stop;
	CUevent			ev_kern_main_end;
	CUevent			ev_dma_recv_start;
	CUevent			ev_dma_recv_stop;
	pgstrom_multihash_tables   *mhtables;	/* inner hashjoin tables */
	pgstrom_data_store *pds_src;	/* data store of outer relation */
	pgstrom_data_store *pds_dst;	/* data store of result buffer */
	kern_hashjoin	kern;	/* kern_hashjoin of this request */
} pgstrom_gpuhashjoin;

typedef struct
{
	GpuTaskState	gts;
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

	int				result_format;
	HeapTupleData	curr_tuple;

	pgstrom_perfmon	pfm;
} GpuHashJoinState;

typedef struct {
	CustomScanState	css;
	GpuContext	   *gcontext;
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
	List		   *hash_keytype;
} MultiHashState;

/*
 * static functions
 */
static bool pgstrom_process_gpuhashjoin(GpuTask *gtask);
static bool pgstrom_complete_gpuhashjoin(GpuTask *gtask);
static void pgstrom_release_gpuhashjoin(GpuTask *gtask);
static GpuTask *gpuhashjoin_next_chunk(GpuTaskState *gts);
static TupleTableSlot *gpuhashjoin_next_tuple(GpuTaskState *gts);

static pgstrom_multihash_tables *
multihash_get_tables(pgstrom_multihash_tables *mhtables);
static void
multihash_put_tables(GpuContext *gcontext, pgstrom_multihash_tables *mhtables);

/*
 * BulkExecMultiHashTables
 *
 * Unlike BulkExecProcNode, it assumes the plannode returns MultiHashNode
 * object, without any sanity checks.
 */
static pgstrom_multihash_tables *
BulkExecMultiHashTables(PlanState *plannode)
{
	CHECK_FOR_INTERRUPTS();

	if (plannode->chgParam != NULL)		/* something changed */
		ExecReScan(plannode);			/* let ReScan handle this */

	/* rough check, not sufficient... */
	if (IsA(plannode, CustomScanState))
	{
		CustomScanState	   *css = (CustomScanState *) plannode;
		PGStromExecMethods *methods = (PGStromExecMethods *) css->methods;
		Assert(methods->ExecCustomBulk != NULL);
		return methods->ExecCustomBulk(css);
	}
	elog(ERROR, "unrecognized node type: %d", (int) nodeTag(plannode));
}

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
	if (hashtable_size > gpuMemMaxAllocSize())
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
	ListCell		   *cell;
	bool				support_bulkload;

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
	 * Check availability of bulkload in this joinrel. If child
	 * GpuHashJoin is merginable, both of nodes have to support
	 * bulkload.
	 */
	if (host_clauses != NIL)
		support_bulkload = false;
	else
	{
		support_bulkload = true;

		foreach (cell, joinrel->reltargetlist)
		{
			Expr   *expr = lfirst(cell);

			if (!IsA(expr, Var) &&
				!pgstrom_codegen_available_expression(expr))
			{
				support_bulkload = false;
				break;
			}
		}
	}

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
	gpath->cpath.flags = (support_bulkload ? CUSTOMPATH_SUPPORT_BULKLOAD : 0);
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

		if (support_bulkload)
		{
			if ((outer_ghj->cpath.flags & CUSTOMPATH_SUPPORT_BULKLOAD) == 0)
				support_bulkload = false;
		}

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
		gpath->cpath.flags
			= (support_bulkload ? CUSTOMPATH_SUPPORT_BULKLOAD : 0);
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
		"STATIC_FUNCTION(bool)\n"
		"gpuhashjoin_qual_eval(cl_int *errcode,\n"
		"                      kern_parambuf *kparams,\n"
		"                      kern_data_store *kds,\n"
		"                      kern_data_store *ktoast,\n"
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
		"STATIC_FUNCTION(void)\n"
		"gpuhashjoin_projection_mapping(cl_int dest_colidx,\n"
		"                               cl_uint *src_depth,\n"
		"                               cl_uint *src_colidx)\n"
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
		"STATIC_FUNCTION(void)\n"
		"gpuhashjoin_projection_datum(cl_int *errcode,\n"
		"                             Datum *slot_values,\n"
		"                             cl_char *slot_isnull,\n"
		"                             cl_int depth,\n"
		"                             cl_int colidx,\n"
		"                             hostptr_t hostaddr,\n"
		"                             void *datum)\n"
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
						"        slot_values[%d] = (Datum)(*((%s *)datum));\n",
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
	appendStringInfo(body, "INIT_LEGACY_CRC32(hash_%u);\n", depth);
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
	appendStringInfo(body, "FIN_LEGACY_CRC32(hash_%u);\n", depth);

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
					 "if (EQ_LEGACY_CRC32(kentry_%d->hash, hash_%d)",
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
				"((char *)kentry_%d - (char *)khtable_%d);\n",
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
		"STATIC_FUNCTION(cl_uint)\n"
		"gpuhashjoin_execute(cl_int *errcode,\n"
		"                    kern_parambuf *kparams,\n"
		"                    kern_multihash *kmhash,\n"
		"                    cl_uint *pg_crc32_table,\n"
		"                    kern_data_store *kds,\n"
		"                    kern_data_store *ktoast,\n"
		"                    size_t kds_index,\n"
		"                    cl_int *rbuffer)\n"
		"{\n"
		);
	/* reference to each hash table */
	for (depth=1; depth <= ghj_info->num_rels; depth++)
	{
		appendStringInfo(
			&decl,
			"kern_hashtable *khtable_%d = KERN_HASHTABLE(kmhash,%d);\n",
			depth, depth);
	}
	/* variable for individual hash entries */
	for (depth=1; depth <= ghj_info->num_rels; depth++)
	{
		appendStringInfo(
			&decl,
			"kern_hashentry *kentry_%d;\n",
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
							 "pg_%s_vref(kds,errcode,%u,kds_index);\n",
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
	ghjoin->flags = best_path->flags;
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
		mhash->custom_relids = NULL;
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
		List	   *outer_quals = NULL;
		Plan	   *alter_plan;

		alter_plan = pgstrom_try_replace_plannode(outer_plan,
												  parse->rtable,
												  &outer_quals);
		if (alter_plan)
		{
			ghj_info.outer_quals = build_flatten_qualifier(outer_quals);
			outer_plan = alter_plan;
		}
	}
	/* check bulkload availability */
	if (IsA(outer_plan, CustomScan))
	{
		int		custom_flags = ((CustomScan *) outer_plan)->flags;

		if ((custom_flags & CUSTOMPATH_SUPPORT_BULKLOAD) != 0)
			ghj_info.outer_bulkload = true;
	}
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
	ghj_info.extra_flags = context.extra_flags;
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
	GpuContext		   *gcontext = pgstrom_get_gpucontext();
	GpuHashJoinState   *ghjs =
		MemoryContextAllocZero(gcontext->memcxt, sizeof(GpuHashJoinState));

	NodeSetTag(ghjs, T_CustomScanState);
	ghjs->gts.css.flags = cscan->flags;
	ghjs->gts.css.methods = &gpuhashjoin_exec_methods.c;
	/* GpuTaskState setup */
	pgstrom_init_gputaskstate(gcontext, &ghjs->gts);
	ghjs->gts.cb_task_process = pgstrom_process_gpuhashjoin;
	ghjs->gts.cb_task_complete = pgstrom_complete_gpuhashjoin;
	ghjs->gts.cb_task_release = pgstrom_release_gpuhashjoin;
	ghjs->gts.cb_next_chunk = gpuhashjoin_next_chunk;
	ghjs->gts.cb_next_tuple = gpuhashjoin_next_tuple;
	ghjs->gts.cb_cleanup = NULL;

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
		ghjs->gts.css.ss.ps.qual = lappend(ghjs->gts.css.ss.ps.qual,
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
	ExecAssignScanType(&ghjs->gts.css.ss, tupdesc);
	ExecAssignScanProjectionInfo(&ghjs->gts.css.ss);

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
	ghjs->gts.kern_source = ghj_info->kernel_source;
	ghjs->gts.extra_flags = ghj_info->extra_flags;
	if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
		pgstrom_preload_cuda_program(&ghjs->gts);

	/*
	 * initialize misc stuff
	 */
	if ((ghjs->gts.css.flags & CUSTOMPATH_PREFERE_ROW_FORMAT) == 0)
		ghjs->result_format = KDS_FORMAT_ROW;
	else
		ghjs->result_format = KDS_FORMAT_SLOT;

	/* Is perfmon needed? */
	ghjs->pfm.enabled = pgstrom_perfmon_enabled;
}

static void
pgstrom_release_gpuhashjoin(GpuTask *gtask)
{
	pgstrom_gpuhashjoin *gpuhashjoin = (pgstrom_gpuhashjoin *) gtask;

	/* unlink hashjoin-table, if remained */
	if (gpuhashjoin->mhtables)
		multihash_put_tables(gtask->gts->gcontext,
							 gpuhashjoin->mhtables);
	/* unlink source data store */
	if (gpuhashjoin->pds_src)
		pgstrom_release_data_store(gpuhashjoin->pds_src);
	/* unlink destination data store */
	if (gpuhashjoin->pds_dst)
		pgstrom_release_data_store(gpuhashjoin->pds_dst);
	/* release this gpu-task itself */
	pfree(gtask);
}

static GpuTask *
pgstrom_create_gpuhashjoin(GpuHashJoinState *ghjs,
						   pgstrom_data_store *pds_src,
						   int result_format)
{
	GpuContext		   *gcontext = ghjs->gts.gcontext;
	pgstrom_gpuhashjoin	*gpuhashjoin;
	pgstrom_data_store *pds_dst;
	kern_data_store	   *kds = pds_src->kds;
	cl_uint				nrooms;
	Size				required;
	TupleDesc			tupdesc;
	kern_hashjoin	   *khashjoin;
	kern_resultbuf	   *kresults;
	kern_parambuf	   *kparams;

	/*
	 * Allocation of pgstrom_gpuhashjoin gputask object
	 */
	required = (offsetof(pgstrom_gpuhashjoin, kern) +
				STROMALIGN(ghjs->kparams->length) +
				STROMALIGN(offsetof(kern_resultbuf, results[0])));
	gpuhashjoin = MemoryContextAllocZero(gcontext->memcxt, required);
	pgstrom_init_gputask(&ghjs->gts, &gpuhashjoin->task);
	gpuhashjoin->mhtables = multihash_get_tables(ghjs->mhtables);
	gpuhashjoin->pds_src = pds_src;
	gpuhashjoin->pds_dst = NULL;

	/*
	 * Set up kern_hashjoin
	 */
	khashjoin = &gpuhashjoin->kern;

	kparams = KERN_HASHJOIN_PARAMBUF(khashjoin);
	memcpy(kparams, ghjs->kparams, ghjs->kparams->length);
	kresults = KERN_HASHJOIN_RESULTBUF(khashjoin);
	nrooms = (cl_uint)((double)kds->nitems *
					   (ghjs->row_population_ratio *
						(1.0 + pgstrom_row_population_margin)));
	memset(kresults, 0, sizeof(kern_resultbuf));
	kresults->nrels = gpuhashjoin->mhtables->kern.ntables + 1;
	kresults->nrooms = nrooms;
	kresults->nitems = 0;
	kresults->errcode = StromError_Success;

	/*
	 * allocation of the destination data-store
	 */
	tupdesc = ghjs->gts.css.ss.ss_ScanTupleSlot->tts_tupleDescriptor;
	if (result_format == KDS_FORMAT_SLOT)
		pds_dst = pgstrom_create_data_store_slot(gcontext, tupdesc,
												 nrooms, false, pds_src);
	else if (result_format == KDS_FORMAT_ROW)
	{
		Size	tuplen;
		Size	length;

		length = (STROMALIGN(offsetof(kern_data_store,
									  colmeta[tupdesc->natts])) +
				  STROMALIGN(sizeof(cl_uint) * nrooms));
		tuplen = MAXALIGN(offsetof(HeapTupleHeaderData,
								   t_bits[BITMAPLEN(tupdesc->natts)]) +
						  (tupdesc->tdhasoid ? sizeof(Oid) : 0));
		tuplen += MAXALIGN(ghjs->gts.css.ss.ps.plan->plan_width);
		length += tuplen * nrooms;

		pds_dst = pgstrom_create_data_store_row(gcontext, tupdesc,
												length, false);
	}
	else
		elog(ERROR, "Bug? unexpected result format: %d", result_format);

	gpuhashjoin->pds_dst = pds_dst;

	return &gpuhashjoin->task;
}

static GpuTask *
gpuhashjoin_next_chunk(GpuTaskState *gts)
{
	GpuHashJoinState   *ghjs = (GpuHashJoinState *) gts;
	PlanState		   *subnode = outerPlanState(ghjs);
	TupleDesc			tupdesc = ExecGetResultType(subnode);
	pgstrom_data_store *pds = NULL;
	int					result_format = ghjs->result_format;
	struct timeval		tv1, tv2, tv3;

	/*
     * Logic to fetch inner multihash-table looks like nested-loop.
     * If all the underlying inner scan already scaned its outer relation,
     * current depth makes advance its scan pointer with reset of underlying
     * scan pointer, or returns NULL if it is already reached end of scan.
     */
retry:
	PERFMON_BEGIN(&gts->pfm_accum, &tv1);

	if (ghjs->gts.scan_done || !ghjs->mhtables)
	{
		PlanState	   *mhstate = innerPlanState(ghjs);
		pgstrom_multihash_tables *mhtables;

		/* load an inner hash-table */
		mhtables = BulkExecMultiHashTables(mhstate);
		if (!mhtables)
		{
			PERFMON_END(&gts->pfm_accum,
						time_inner_load, &tv1, &tv2);
			return NULL;	/* end of inner multi-hashtable  */
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
			Assert(ghjs->gts.scan_done);
			multihash_put_tables(ghjs->gts.gcontext,
								 ghjs->mhtables);
			ghjs->mhtables = NULL;
		}
		ghjs->mhtables = mhtables;

		/*
		 * Rewind the outer scan pointer, if it is not first time.
		 */
		if (ghjs->gts.scan_done)
		{
			ExecReScan(outerPlanState(ghjs));
			ghjs->gts.scan_done = false;
		}
	}
	PERFMON_BEGIN(&gts->pfm_accum, &tv2);

	if (!ghjs->outer_bulkload)
	{
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
					ghjs->gts.scan_done = true;
					break;
				}
			}
			/* create a new data-store if not constructed yet */
			if (!pds)
			{
				pds = pgstrom_create_data_store_row(ghjs->gts.gcontext,
													tupdesc,
													pgstrom_chunk_size(),
													false);
			}
			/* insert the tuple on the data-store */
			if (!pgstrom_data_store_insert_tuple(pds, slot))
			{
				ghjs->outer_overflow = slot;
				break;
			}
		}
	}
	else
	{
		pds = BulkExecProcNode(subnode);
		if (!pds)
			ghjs->gts.scan_done = true;
	}
	PERFMON_END(&gts->pfm_accum, time_inner_load, &tv1, &tv2);
	PERFMON_END(&gts->pfm_accum, time_outer_load, &tv2, &tv3);

	/*
	 * We also need to check existence of next inner hash-chunks, even if
	 * here is no more outer records, In case of hash-table splited-out,
	 * we have to rewind the outer relation scan, then makes relations
	 * join with the next inner hash chunks.
	 */
	if (!pds)
		goto retry;

	return pgstrom_create_gpuhashjoin(ghjs, pds, result_format);
}

static TupleTableSlot *
gpuhashjoin_next_tuple(GpuTaskState *gts)
{
	GpuHashJoinState	   *ghjs = (GpuHashJoinState *) gts;
	TupleTableSlot		   *ps_slot = gts->css.ss.ss_ScanTupleSlot;
	pgstrom_gpuhashjoin	   *gpuhashjoin =
		(pgstrom_gpuhashjoin *)ghjs->gts.curr_task;
	pgstrom_data_store	   *pds_dst = gpuhashjoin->pds_dst;
	kern_data_store		   *kds_dst = pds_dst->kds;
	struct timeval			tv1, tv2;

	PERFMON_BEGIN(&ghjs->gts.pfm_accum, &tv1);

	while (ghjs->gts.curr_index < kds_dst->nitems)
	{
		int			index = ghjs->gts.curr_index++;

		/* fetch a result tuple */
		pgstrom_fetch_data_store(ps_slot,
								 pds_dst,
								 index,
								 &ghjs->curr_tuple);
		if (ghjs->gts.css.ss.ps.qual != NIL)
		{
			ExprContext	   *econtext = ghjs->gts.css.ss.ps.ps_ExprContext;

			econtext->ecxt_scantuple = ps_slot;
			if (!ExecQual(ghjs->gts.css.ss.ps.qual, econtext, false))
				continue;	/* try to fetch next tuple */
		}
		PERFMON_END(&ghjs->gts.pfm_accum, time_materialize, &tv1, &tv2);
		return ps_slot;
	}
	PERFMON_END(&ghjs->gts.pfm_accum, time_materialize, &tv1, &tv2);
	ExecClearTuple(ps_slot);
	return NULL;
}

#if 0
static TupleTableSlot *
gpuhashjoin_fetch_tuple(CustomScanState *node)
{
	GpuHashJoinState   *ghjs = (GpuHashJoinState *) node;
	TupleTableSlot	   *slot = ghjs->gts.css.ss.ss_ScanTupleSlot;

	ExecClearTuple(slot);

	while (!ghjs->gts.curr_task || !gpuhashjoin_next_tuple(ghjs))
	{
		pgstrom_gpuhashjoin *ghjoin
			= (pgstrom_gpuhashjoin *) ghjs->gts.curr_task;

		/* release the current gpuhashjoin chunk if any */
		if (ghjoin)
		{
			SpinLockAcquire(&ghjs->gts.lock);
			dlist_delete(&ghjoin->task.tracker);
			ghjs->gts.curr_task = NULL;
			ghjs->gts.curr_index = 0;
			SpinLockRelease(&ghjs->gts.lock);
			pgstrom_release_gpuhashjoin(&ghjoin->task);
		}
		/* fetch next chunk to be scanned */
		ghjoin = (pgstrom_gpuhashjoin *) pgstrom_fetch_gputask(&ghjs->gts);
		if (!ghjoin)
			break;
		SpinLockAcquire(&ghjs->gts.lock);
		ghjs->gts.curr_task = &ghjoin->task;
		ghjs->gts.curr_index = 0;
		SpinLockRelease(&ghjs->gts.lock);
	}
	return slot;
}

static bool
gpuhashjoin_recheck(CustomScanState *node, TupleTableSlot *slot)
{
	/* There are no access-method-specific conditions to recheck. */
	return true;
}
#endif

static TupleTableSlot *
gpuhashjoin_exec(CustomScanState *node)
{
	return ExecScan(&node->ss,
					(ExecScanAccessMtd) pgstrom_exec_gputask,
					(ExecScanRecheckMtd) pgstrom_recheck_gputask);
}

static void *
gpuhashjoin_exec_bulk(CustomScanState *node)
{
	GpuHashJoinState	   *ghjs = (GpuHashJoinState *) node;
	pgstrom_gpuhashjoin	   *ghjoin;
	pgstrom_data_store	   *pds;

	/* force to return row-format */
	ghjs->result_format = KDS_FORMAT_ROW;

	/* fetch next chunk to be processed */
	ghjoin = (pgstrom_gpuhashjoin *) pgstrom_fetch_gputask(&ghjs->gts);
	if (!ghjoin)
		return NULL;

	/* extrace pgstrom_data_store */
	pds = ghjoin->pds_dst;
	ghjoin->pds_dst = NULL;
	pgstrom_release_gpuhashjoin(&ghjoin->task);

	return pds;
}

static void
gpuhashjoin_end(CustomScanState *node)
{
	GpuHashJoinState	   *ghjs = (GpuHashJoinState *) node;

	/*
	 * Clean up subtrees
	 */
	ExecEndNode(outerPlanState(node));
	ExecEndNode(innerPlanState(node));

	pgstrom_release_gputaskstate(&ghjs->gts);
}

static void
gpuhashjoin_rescan(CustomScanState *node)
{
	GpuHashJoinState		 *ghjs = (GpuHashJoinState *) node;
	pgstrom_multihash_tables *mhtables = ghjs->mhtables;

	/* clean-up and release any concurrent tasks */
	pgstrom_cleanup_gputaskstate(&ghjs->gts);

	/* rewind the outer relation, also */
	ghjs->gts.scan_done = false;
	ghjs->outer_overflow = NULL;
	ExecReScan(outerPlanState(ghjs));

	/*
	 * we reuse the inner hash table if it is flat (that means mhtables
	 * is not divided into multiple portions) and no parameter changed.
	 */
	if (mhtables->is_divided ||
		innerPlanState(ghjs)->chgParam != NULL)
	{
		pfree(mhtables);
		ghjs->mhtables = NULL;
	}
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
	CustomScan *cscan = (CustomScan *) ghjs->gts.css.ss.ps.plan;

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

	pgstrom_explain_gputaskstate(&ghjs->gts, es);
}

/* ----------------------------------------------------------------
 *
 * Callback routines for MultiHash node
 *
 * ---------------------------------------------------------------- */
static pgstrom_multihash_tables *
multihash_get_tables(pgstrom_multihash_tables *mhtables)
{
	Assert(mhtables->refcnt > 0);
	mhtables->refcnt++;

	return mhtables;
}

static void
multihash_put_tables(GpuContext *gcontext, pgstrom_multihash_tables *mhtables)
{
	if (--mhtables->refcnt == 0)
	{
		CUresult	rc;
		int			i;

		for (i=0; i < gcontext->num_context; i++)
		{
			rc = cuCtxPushCurrent(gcontext->gpu[i].cuda_context);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));

			if (mhtables->m_hash[i])
				__gpuMemFree(gcontext, i, mhtables->m_hash[i]);

			if (mhtables->ev_loaded[i])
			{
				rc = cuEventDestroy(mhtables->ev_loaded[i]);
				if (rc != CUDA_SUCCESS)
					elog(ERROR, "failed on cuEventDestroy: %s", errorText(rc));
			}

			rc = cuCtxPopCurrent(NULL);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuCtxPopCurrent: %s", errorText(rc));
		}
		pfree(mhtables->m_hash);
		pfree(mhtables->ev_loaded);
		pfree(mhtables);
	}
}

static Node *
multihash_create_scan_state(CustomScan *cscan)
{
	MultiHashState *mhs = palloc0(sizeof(MultiHashState));

	NodeSetTag(mhs, T_CustomScanState);
	mhs->css.flags = cscan->flags;
	mhs->css.methods = &multihash_exec_methods.c;
	mhs->gcontext = pgstrom_get_gpucontext();

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
	List		   *hash_keytype = NIL;
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
		Oid			type_oid = exprType(lfirst(cell));
		int16		typlen;
		bool		typbyval;

		get_typlenbyval(type_oid, &typlen, &typbyval);

		hash_keys = lappend(hash_keys,
							ExecInitExpr(lfirst(cell), &mhs->css.ss.ps));
		hash_keylen = lappend_int(hash_keylen, typlen);
		hash_keybyval = lappend_int(hash_keybyval, typbyval);
		hash_keytype = lappend_oid(hash_keytype, type_oid);
	}
	mhs->hash_keys = hash_keys;
	mhs->hash_keylen = hash_keylen;
	mhs->hash_keybyval = hash_keybyval;
	mhs->hash_keytype = hash_keytype;

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

static pgstrom_multihash_tables *
expand_multihash_tables(MultiHashState *mhs,
						pgstrom_multihash_tables *mhtables_old,
						Size consumed, Size required)
{
	pgstrom_multihash_tables *mhtables_new;
	GpuContext *gcontext = mhs->gcontext;
	Size		length_new;

	/* sanity check */
	Assert(consumed <= required);

	/* estimate new length */
	length_new = mhtables_old->length;
	while (required > mhs->threshold * length_new)
		length_new += length_new;
	Assert(length_new > mhtables_old->length);

	/* allocate a new one */
	mhtables_new = MemoryContextAlloc(gcontext->memcxt, length_new);
	memcpy(mhtables_new, mhtables_old, consumed);
	mhtables_new->length = length_new;
	mhtables_new->kern.hostptr = (hostptr_t)&mhtables_new->kern.hostptr;
	Assert(mhtables_new->length > mhtables_old->length);

	elog(INFO, "pgstrom_multihash_tables was expanded %zu (%p) => %zu (%p)",
		 mhtables_old->length, mhtables_old,
		 mhtables_new->length, mhtables_new);

	/* NOTE: must not use multihash_put_tables() here, because it releases
	 * m_hash[] and ev_loaded[] array. Also, we don't need to pay attention
	 * for concurrent task during inner preloading.
	 */
	pfree(mhtables_old);

	/* update hashtable_size of MultiHashState */
	do {
		Assert(IsA(mhs, CustomScanState) &&
			   mhs->css.methods == &multihash_exec_methods.c);
		mhs->hashtable_size = length_new;
		mhs = (MultiHashState *) innerPlanState(mhs);
	} while (mhs);

	return mhtables_new;
}

static void
multihash_preload_khashtable(MultiHashState *mhs,
							 pgstrom_multihash_tables *mhtables,
							 bool scan_forward)
{
	TupleDesc		tupdesc = ExecGetResultType(outerPlanState(mhs));
	ExprContext	   *econtext = mhs->css.ss.ps.ps_ExprContext;
	int				depth = mhs->depth;
	kern_hashtable *khtable;
	kern_hashentry *hentry;
	Size			required;
	Size			consumed;
	cl_uint		   *hash_slots;
	int				attcacheoff;
	int				attalign;
	int				i;

	/* preload should be done under the MultiExec context */
	Assert(CurrentMemoryContext == mhs->css.ss.ps.state->es_query_cxt);

	/*
	 * First of all, construct a kern_hashtable on the tail of current
	 * usage pointer of mhtables.
	 */
	Assert(mhtables->kern.htable_offset[depth] == 0);
	Assert(mhtables->usage == LONGALIGN(mhtables->usage));
	mhtables->kern.htable_offset[depth] =
		mhtables->usage - offsetof(pgstrom_multihash_tables, kern);

	if (!scan_forward)
	{
		Assert(mhs->curr_chunk);
		required = mhtables->usage + mhs->curr_chunk->length;
		if (required > mhs->threshold * mhtables->length)
		{
			mhtables = expand_multihash_tables(mhs, mhtables,
											   mhtables->usage, required);
		}
		memcpy((char *)mhtables + mhtables->usage,
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
	if (required > mhs->threshold * mhtables->length)
	{
		mhtables = expand_multihash_tables(mhs, mhtables,
										   mhtables->usage, required);
	}
	khtable = (kern_hashtable *)((char *)mhtables + mhtables->usage);
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
	Assert(mhtables->usage + consumed == required);

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
		ListCell	   *lc4;

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
		if (required > mhs->threshold * mhtables->length)
		{
			mhtables = expand_multihash_tables(mhs, mhtables,
											   mhtables->usage + consumed,
											   required);
			khtable = (kern_hashtable *)((char *)mhtables + mhtables->usage);
			hash_slots = KERN_HASHTABLE_SLOT(khtable);
		}

		/* calculation of a hash value of this entry */
		INIT_LEGACY_CRC32(hash);
		econtext->ecxt_scantuple = scan_slot;
		forfour(lc1, mhs->hash_keys,
				lc2, mhs->hash_keylen,
				lc3, mhs->hash_keybyval,
				lc4, mhs->hash_keytype)
		{
			ExprState  *clause = lfirst(lc1);
			int			keylen = lfirst_int(lc2);
			bool		keybyval = lfirst_int(lc3);
			Oid			keytype = lfirst_oid(lc4);
			int			errcode;
			Datum		value;
			bool		isnull;

			value = ExecEvalExpr(clause, econtext, &isnull, NULL);
			if (isnull)
				continue;

			/* fixup host representation to special internal format. */
			if (keytype == NUMERICOID)
			{
				pg_numeric_t	temp
					= pg_numeric_from_varlena(&errcode, (struct varlena *)
											  DatumGetPointer(value));
				keylen = sizeof(temp.value);
				keybyval = true;
				value = temp.value;
			}

			if (keylen > 0)
			{
				if (keybyval)
					COMP_LEGACY_CRC32(hash, &value, keylen);
				else
					COMP_LEGACY_CRC32(hash, DatumGetPointer(value), keylen);
			}
			else
			{
				COMP_LEGACY_CRC32(hash,
								  VARDATA_ANY(value),
								  VARSIZE_ANY_EXHDR(value));
			}
		}
		FIN_LEGACY_CRC32(hash);

		/* allocation of hash entry and insert it */
		hentry = (kern_hashentry *)((char *)khtable + consumed);
		hentry->hash = hash;
		hentry->rowid = (cl_uint) mhtables->ntuples;	/* not in use */
		hentry->t_len = scan_tuple->t_len;
		memcpy(&hentry->htup, scan_tuple->t_data, scan_tuple->t_len);

		i = hash % khtable->nslots;
		hentry->next = hash_slots[i];
		hash_slots[i] = (cl_uint)consumed;

		/* increment buffer consumption */
		consumed += entry_size;
		/* increment number of tuples read */
		mhtables->ntuples++;
	}
	Assert(mhtables->usage + consumed <= mhtables->length);
	mhtables->usage += consumed;
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
	GpuContext	   *gcontext = mhs->gcontext;
	pgstrom_multihash_tables *mhtables = NULL;
	PlanState	   *mhstate;	/* underlying MultiHash, if any */
	bool			scan_forward = false;
	int				depth = mhs->depth;

	/* must provide our own instrumentation support */
	if (node->ss.ps.instrument)
		InstrStartNode(node->ss.ps.instrument);

	mhstate = innerPlanState(mhs);
	if (mhstate)
	{
		mhtables = BulkExecMultiHashTables(mhstate);
		if (!mhtables)
		{
			if (mhs->outer_done)
				goto out;
			ExecReScan(mhstate);
			mhtables = BulkExecMultiHashTables(mhstate);
			if (!mhtables)
				goto out;
			scan_forward = true;
		}
		else if (!mhs->curr_chunk)
			scan_forward = true;
	}
	else
	{
		/* no more deep hash-table, so create a new pgstrom_multihash_tables */
		int			ntables = depth;
		Size		usage;

		if (mhs->outer_done)
			goto out;
		scan_forward = true;

		/* allocation of multihash_tables on pinned memory */
		mhtables = MemoryContextAlloc(gcontext->memcxt,
									  mhs->hashtable_size);
		/* initialize multihash_tables */
		usage = offsetof(pgstrom_multihash_tables, kern) +
			STROMALIGN(offsetof(kern_multihash, htable_offset[ntables + 1]));
		memset(mhtables, 0, usage);
		mhtables->length = mhs->hashtable_size;
		mhtables->usage = usage;
		mhtables->ntuples = 0.0;
		mhtables->refcnt = 1;
		mhtables->is_divided = false;
		mhtables->m_hash = MemoryContextAllocZero(gcontext->memcxt,
												  sizeof(CUdeviceptr) *
												  gcontext->num_context);
		mhtables->ev_loaded = MemoryContextAllocZero(gcontext->memcxt,
													 sizeof(CUevent) *
													 gcontext->num_context);
		memcpy(mhtables->kern.pg_crc32_table,
			   pg_crc32_table,
			   sizeof(uint32) * 256);
		mhtables->kern.hostptr = (hostptr_t) &mhtables->kern.hostptr;
		mhtables->kern.ntables = ntables;
	}
	Assert(mhtables != NULL);

	/*
	 * construct a kernel hash-table that stores all the inner-keys
	 * in this level, being loaded from the outer relation
	 */
	multihash_preload_khashtable(mhs, mhtables, scan_forward);
out:
	/* must provide our own instrumentation support */
	if (node->ss.ps.instrument)
		InstrStopNode(node->ss.ps.instrument,
					  !mhtables ? 0.0 : mhtables->ntuples);
	if (mhtables)
		mhs->nbatches_exec++;

	return mhtables;
}

static void
multihash_end(CustomScanState *node)
{
	MultiHashState *mhs = (MultiHashState *) node;

	/*
	 * shut down the subplan
	 */
	ExecEndNode(outerPlanState(node));
	ExecEndNode(innerPlanState(node));

	/*
	 * release GpuContext
	 */
	pgstrom_put_gpucontext(mhs->gcontext);
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

/* ----------------------------------------------------------------
 *
 * GpuTask handlers of GpuHashJoin
 *
 * ----------------------------------------------------------------
 */
static void
gpuhashjoin_cleanup_cuda_resources(pgstrom_gpuhashjoin *ghjoin)
{
	CUDA_EVENT_DESTROY(ghjoin, ev_dma_send_start);
	CUDA_EVENT_DESTROY(ghjoin, ev_dma_send_stop);
	CUDA_EVENT_DESTROY(ghjoin, ev_kern_main_end);
	CUDA_EVENT_DESTROY(ghjoin, ev_dma_recv_start);
	CUDA_EVENT_DESTROY(ghjoin, ev_dma_recv_stop);

	if (ghjoin->m_join)
		gpuMemFree(&ghjoin->task, ghjoin->m_join);
	if (ghjoin->m_kds_src)
		gpuMemFree(&ghjoin->task, ghjoin->m_kds_src);
	if (ghjoin->m_kds_dst)
		gpuMemFree(&ghjoin->task, ghjoin->m_kds_dst);

	/* clear the pointers */
	ghjoin->kern_main = NULL;
	memset(ghjoin->kern_main_args, 0, sizeof(ghjoin->kern_main_args));
	ghjoin->kern_proj = NULL;
	memset(ghjoin->kern_proj_args, 0, sizeof(ghjoin->kern_proj_args));
	ghjoin->m_join = 0UL;
	ghjoin->m_kds_src = 0UL;
	ghjoin->m_kds_dst = 0UL;
	ghjoin->hash_loader = false;
	ghjoin->ev_dma_send_start = NULL;
	ghjoin->ev_dma_send_stop = NULL;
	ghjoin->ev_kern_main_end = NULL;
	ghjoin->ev_dma_recv_start = NULL;
	ghjoin->ev_dma_recv_stop = NULL;
}

static bool
pgstrom_complete_gpuhashjoin(GpuTask *gtask)
{
	pgstrom_gpuhashjoin *ghjoin = (pgstrom_gpuhashjoin *) gtask;
	GpuTaskState   *gts = gtask->gts;

	if (gts->pfm_accum.enabled)
	{
		CUDA_EVENT_ELAPSED(ghjoin, time_dma_send,
						   ev_dma_send_start,
						   ev_dma_send_stop);
		CUDA_EVENT_ELAPSED(ghjoin, time_kern_join,
						   ev_dma_send_stop,
						   ev_kern_main_end);
		CUDA_EVENT_ELAPSED(ghjoin, time_kern_proj,
						   ev_kern_main_end,
						   ev_dma_recv_start);
		CUDA_EVENT_ELAPSED(ghjoin, time_dma_recv,
						   ev_dma_recv_start,
						   ev_dma_recv_stop);
		pgstrom_accum_perfmon(&gts->pfm_accum, &ghjoin->task.pfm);
	}
	gpuhashjoin_cleanup_cuda_resources(ghjoin);

	/*
	 * StromError_DataStoreNoSpace indicates pds_dst was smaller than
	 * what GpuHashJoin required. So, we expand the buffer and kick
	 * this gputask again.
	 */
	if (ghjoin->task.errcode == StromError_DataStoreNoSpace)
	{
		GpuContext		   *gcontext = gts->gcontext;
		kern_resultbuf	   *kresults = KERN_HASHJOIN_RESULTBUF(&ghjoin->kern);
		pgstrom_data_store *pds = ghjoin->pds_dst;
		kern_data_store	   *old_kds = pds->kds;
		kern_data_store	   *new_kds;
		cl_uint				ncols = old_kds->ncols;
		cl_uint				nitems = old_kds->nitems;
		Size				required;

		/* GpuHashJoin should not take file-mapped data store */
		Assert(pds->kds_fname == NULL);

		/* adjust kern_resultbuf */
		kresults->nrooms = nitems;
		kresults->nitems = 0;
		kresults->errcode = StromError_Success;

		/* re-allocation of pgstrom_data_store */
		if (old_kds->format == KDS_FORMAT_ROW)
		{
			elog(INFO, "GpuHashJoin input again (length: %u => %u)",
				 old_kds->length, old_kds->usage);
			required = (KERN_DATA_STORE_HEAD_LENGTH(old_kds) +
						STROMALIGN(sizeof(cl_uint) * nitems) +
						STROMALIGN(old_kds->usage));
			new_kds = MemoryContextAlloc(gcontext->memcxt, required);
			memcpy(new_kds, old_kds, KERN_DATA_STORE_HEAD_LENGTH(old_kds));
			new_kds->hostptr = (hostptr_t) &new_kds->hostptr;
			new_kds->length = required;
			new_kds->usage = 0;
			new_kds->nrooms = nitems;
			new_kds->nitems = 0;
		}
		else if (old_kds->format == KDS_FORMAT_SLOT)
		{
			elog(INFO, "GpuHashJoin input again (nrooms: %u => %u)",
				 old_kds->nrooms, nitems);
			required = STROMALIGN(KERN_DATA_STORE_HEAD_LENGTH(old_kds) +
								  (LONGALIGN(sizeof(Datum) * ncols) +
								   LONGALIGN(sizeof(bool) * ncols)) * nitems);
			new_kds = MemoryContextAlloc(gcontext->memcxt, required);
			memcpy(new_kds, old_kds, KERN_DATA_STORE_HEAD_LENGTH(old_kds));
			new_kds->hostptr = (hostptr_t) &new_kds->hostptr;
            new_kds->length = required;
            new_kds->usage = 0;
            new_kds->nrooms = nitems;
            new_kds->nitems = 0;
		}
		else
			elog(ERROR, "Bug? unexpected KDS format: %d", old_kds->format);

		pds->kds = new_kds;
		pfree(old_kds);

		/*
		 * OK, chain this task on the pending_tasks queue again
		 */
		SpinLockAcquire(&gts->lock);
		dlist_push_head(&gts->pending_tasks, &ghjoin->task.chain);
		gts->num_pending_tasks++;
		SpinLockRelease(&gts->lock);

		return false;	/* exceptional path! */
	}
	return true;
}

static void
pgstrom_respond_gpuhashjoin(CUstream stream, CUresult status, void *private)
{
	pgstrom_gpuhashjoin	*ghjoin = private;
	GpuTaskState		*gts = ghjoin->task.gts;
	kern_resultbuf		*kresults = KERN_HASHJOIN_RESULTBUF(&ghjoin->kern);

	SpinLockAcquire(&gts->lock);
	if (status != CUDA_SUCCESS)
		ghjoin->task.errcode = status;
	else
		ghjoin->task.errcode = kresults->errcode;

	/* remove from the running_tasks list */
	dlist_delete(&ghjoin->task.chain);
	gts->num_running_tasks--;
	/* then, attach it on the completed_tasks list */
	if (ghjoin->task.errcode == StromError_Success)
		dlist_push_tail(&gts->completed_tasks, &ghjoin->task.chain);
	else
		dlist_push_head(&gts->completed_tasks, &ghjoin->task.chain);
	gts->num_completed_tasks++;
	SpinLockRelease(&gts->lock);

	SetLatch(&MyProc->procLatch);
}

static bool
__pgstrom_process_gpuhashjoin(pgstrom_gpuhashjoin *ghjoin)
{
	pgstrom_multihash_tables *mhtables = ghjoin->mhtables;
	pgstrom_data_store *pds_src = ghjoin->pds_src;
	pgstrom_data_store *pds_dst = ghjoin->pds_dst;
	kern_resultbuf	   *kresults = KERN_HASHJOIN_RESULTBUF(&ghjoin->kern);
	int			cuda_index = ghjoin->task.cuda_index;
	CUdeviceptr	m_hash;
	Size		offset;
	Size		length;
	size_t		nitems;
	size_t		grid_size;
	size_t		block_size;
	CUevent		ev_loaded;
	CUresult	rc;

	/*
	 * sanity checks
	 */
	Assert(pds_src->kds->format == KDS_FORMAT_ROW);
	Assert(!pds_src->ptoast);
	Assert(pds_dst->kds->format == KDS_FORMAT_ROW ||
		   pds_dst->kds->format == KDS_FORMAT_SLOT);

	/*
	 * kernel function lookup
	 */
	rc = cuModuleGetFunction(&ghjoin->kern_main,
							 ghjoin->task.cuda_module,
							 "kern_gpuhashjoin_main");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	rc = cuModuleGetFunction(&ghjoin->kern_proj,
							 ghjoin->task.cuda_module,
							 pds_dst->kds->format == KDS_FORMAT_ROW
							 ? "kern_gpuhashjoin_projection_row"
							 : "kern_gpuhashjoin_projection_slot");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	/*
	 * Allocation of device memory for hash table. If someone already
	 * allocated it, we shall reuse this segment.
	 */
	if (mhtables->m_hash[cuda_index])
		m_hash = mhtables->m_hash[cuda_index];
	else
	{
		m_hash = gpuMemAlloc(&ghjoin->task, mhtables->length);
		if (!m_hash)
			goto out_of_resource;
		mhtables->m_hash[cuda_index] = m_hash;
	}
	Assert(m_hash != 0UL);

	/*
	 * Allocation of device memory for each chunks
	 */

	/* __global kern_hashjoin *khashjoin */
	length = (KERN_HASHJOIN_PARAMBUF_LENGTH(&ghjoin->kern) +
			  KERN_HASHJOIN_RESULTBUF_LENGTH(&ghjoin->kern) +
			  sizeof(cl_int) * kresults->nrels * kresults->nrooms);
	ghjoin->m_join = gpuMemAlloc(&ghjoin->task, length);
	if (!ghjoin->m_join)
		goto out_of_resource;

	/* __global kern_data_store *kds_src */
	length = KERN_DATA_STORE_LENGTH(pds_src->kds);
	ghjoin->m_kds_src = gpuMemAlloc(&ghjoin->task, length);
	if (!ghjoin->m_kds_src)
		goto out_of_resource;

	/* __global kern_data_store *kds_dst */
	length = KERN_DATA_STORE_LENGTH(pds_dst->kds);
	ghjoin->m_kds_dst = gpuMemAlloc(&ghjoin->task, length);
	if (!ghjoin->m_kds_dst)
		goto out_of_resource;

	/* Creation of event objects, if needed */
	if (ghjoin->task.pfm.enabled)
	{
		rc = cuEventCreate(&ghjoin->ev_dma_send_start, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

		rc = cuEventCreate(&ghjoin->ev_dma_send_stop, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

		rc = cuEventCreate(&ghjoin->ev_kern_main_end, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

		rc = cuEventCreate(&ghjoin->ev_dma_recv_start, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

		rc = cuEventCreate(&ghjoin->ev_dma_recv_stop, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));
	}

	/*
	 * OK, all the device memory and kernel objects are successfully
	 * constructed. Let's enqueue DMA send/recv and kernel invocations.
	 */
	CUDA_EVENT_RECORD(ghjoin, ev_dma_send_start);

	if (!mhtables->ev_loaded[cuda_index])
	{
		rc = cuEventCreate(&ev_loaded, CU_EVENT_DISABLE_TIMING);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

		rc = cuMemcpyHtoDAsync(m_hash,
							   &mhtables->kern,
							   length,
							   ghjoin->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));

		rc = cuEventRecord(ev_loaded, ghjoin->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventRecord: %s", errorText(rc));

		mhtables->ev_loaded[cuda_index] = ev_loaded;
		ghjoin->hash_loader = true;
	}
	else
	{
		ev_loaded = mhtables->ev_loaded[cuda_index];
		rc = cuStreamWaitEvent(ghjoin->task.cuda_stream, ev_loaded, 0);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuStreamWaitEvent: %s", errorText(rc));
		ghjoin->hash_loader = false;
	}

	/* DMA Send: __global kern_hashjoin *khashjoin */
	length = KERN_HASHJOIN_DMA_SENDLEN(&ghjoin->kern);
	rc = cuMemcpyHtoDAsync(ghjoin->m_join,
						   KERN_HASHJOIN_DMA_SENDPTR(&ghjoin->kern),
						   length,
						   ghjoin->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	ghjoin->task.pfm.bytes_dma_send += length;
	ghjoin->task.pfm.num_dma_send++;

	/* DMA Send: __global kern_data_store *kds_src */
	length = KERN_DATA_STORE_LENGTH(pds_src->kds);
	rc = cuMemcpyHtoDAsync(ghjoin->m_kds_src,
						   pds_src->kds,
						   length,
						   ghjoin->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	ghjoin->task.pfm.bytes_dma_send += length;
	ghjoin->task.pfm.num_dma_send++;

	/* DMA Send: __global kern_data_store *kds_dst */
	length = KERN_DATA_STORE_HEAD_LENGTH(pds_dst->kds);
	rc = cuMemcpyHtoDAsync(ghjoin->m_kds_dst,
						   pds_dst->kds,
						   length,
						   ghjoin->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	ghjoin->task.pfm.bytes_dma_send += length;
	ghjoin->task.pfm.num_dma_send++;

	/*
	 * OK, enqueue a series of requests
	 */
	CUDA_EVENT_RECORD(ghjoin, ev_dma_send_stop);

	/*
	 * Launch: __global void
	 * kern_gpuhashjoin_main(kern_hashjoin *khashjoin,
	 *                       kern_multihash *kmhash,
	 *                       kern_data_store *kds)
	 */
	nitems = pds_src->kds->nitems;
	pgstrom_compute_workgroup_size(&grid_size,
								   &block_size,
								   ghjoin->kern_main,
								   ghjoin->task.cuda_device,
								   false,
								   nitems,
								   sizeof(cl_uint));

	ghjoin->kern_main_args[0] = &ghjoin->m_join;
	ghjoin->kern_main_args[1] = &m_hash;
	ghjoin->kern_main_args[2] = &ghjoin->m_kds_src;

	rc = cuLaunchKernel(ghjoin->kern_main,
						grid_size, 1, 1,
						block_size, 1, 1,
						sizeof(uint) * block_size,
						ghjoin->task.cuda_stream,
						ghjoin->kern_main_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
	ghjoin->task.pfm.num_kern_join++;

	CUDA_EVENT_RECORD(ghjoin, ev_kern_main_end);

	/*
	 * Launch: __global__ void
	 * kern_gpuhashjoin_projection_row(kern_hashjoin *khashjoin,
	 * 								   kern_multihash *kmhash,
	 *                                 kern_data_store *kds_src,
	 *                                 kern_data_store *kds_dst)
	 */
	pgstrom_compute_workgroup_size(&grid_size,
								   &block_size,
								   ghjoin->kern_proj,
								   ghjoin->task.cuda_device,
								   false,
								   nitems,
								   sizeof(cl_uint));
	ghjoin->kern_proj_args[0] = &ghjoin->m_join;
	ghjoin->kern_proj_args[1] = &m_hash;
	ghjoin->kern_proj_args[2] = &ghjoin->m_kds_src;
	ghjoin->kern_proj_args[3] = &ghjoin->m_kds_dst;

	rc = cuLaunchKernel(ghjoin->kern_proj,
						grid_size, 1, 1,
						block_size, 1, 1,
						sizeof(uint) * block_size,
						ghjoin->task.cuda_stream,
						ghjoin->kern_proj_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
	ghjoin->task.pfm.num_kern_proj++;

	CUDA_EVENT_RECORD(ghjoin, ev_dma_recv_start);

	/* DMA Recv: __global kern_hashjoin *khashjoin */
	offset = KERN_HASHJOIN_DMA_RECVOFS(&ghjoin->kern);
	length = KERN_HASHJOIN_DMA_RECVLEN(&ghjoin->kern);
	rc = cuMemcpyDtoHAsync(KERN_HASHJOIN_DMA_RECVPTR(&ghjoin->kern),
						   ghjoin->m_join + offset,
						   length,
						   ghjoin->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuMemcpyDtoHAsync: %s", errorText(rc));
	ghjoin->task.pfm.bytes_dma_recv += length;
	ghjoin->task.pfm.num_dma_recv++;

	/* DMA Recv: __global kern_data_store *kds_dst */
	length = KERN_DATA_STORE_LENGTH(pds_dst->kds);
	rc = cuMemcpyDtoHAsync(pds_dst->kds,
						   ghjoin->m_kds_dst,
						   length,
						   ghjoin->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuMemcpyDtoHAsync: %s", errorText(rc));
	ghjoin->task.pfm.bytes_dma_recv += length;
	ghjoin->task.pfm.num_dma_recv++;

	CUDA_EVENT_RECORD(ghjoin, ev_dma_recv_stop);

	/*
	 * Register callback
	 */
	rc = cuStreamAddCallback(ghjoin->task.cuda_stream,
							 pgstrom_respond_gpuhashjoin,
							 ghjoin, 0);
    if (rc != CUDA_SUCCESS)
        elog(ERROR, "cuStreamAddCallback: %s", errorText(rc));

	return true;

out_of_resource:
	gpuhashjoin_cleanup_cuda_resources(ghjoin);
	return false;
}

static bool
pgstrom_process_gpuhashjoin(GpuTask *gtask)
{
	pgstrom_gpuhashjoin	*ghjoin = (pgstrom_gpuhashjoin *) gtask;
	bool		status;
	CUresult	rc;

	/* Switch CUDA Context */
	rc = cuCtxPushCurrent(gtask->cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));
	PG_TRY();
	{
		status = __pgstrom_process_gpuhashjoin(ghjoin);
	}
	PG_CATCH();
	{
		rc = cuCtxPopCurrent(NULL);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));
		gpuhashjoin_cleanup_cuda_resources(ghjoin);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Reset CUDA Context */
	rc = cuCtxPopCurrent(NULL);
	if (rc != CUDA_SUCCESS)
		elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));

	return status;
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
