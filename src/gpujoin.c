/*
 * gpujoin.c
 *
 * GPU accelerated relations join, based on nested-loop or hash-join
 * algorithm.
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
#include "catalog/pg_type.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/clauses.h"
#include "optimizer/cost.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/planmain.h"
#include "optimizer/restrictinfo.h"
#include "optimizer/var.h"
#include "parser/parsetree.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/ruleutils.h"
#include <math.h>
#include "pg_strom.h"
#include "cuda_gpujoin.h"



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
	Plan		   *outer_plan;		/* for create_gpujoin_plan convenience */
	Size			kmrels_length;
	double			kresults_ratio;	/* expected total-items ratio */
	int				num_rels;
	List		   *host_quals;
	struct {
		Cost		startup_cost;	/* outer scan cost + materialize */
		Cost		run_cost;		/* outer scan cost + materialize */
		double		nrows_ratio;	/* nrows ratio towards outer rows */
		JoinType	join_type;		/* one of JOIN_* */
		Path	   *scan_path;		/* outer scan path */
		Plan	   *scan_plan;		/* for create_gpujoin_plan convenience */
		List	   *hash_quals;		/* valid quals, if hash-join */
		List	   *join_quals;		/* all the device quals, incl hash_quals */
		double		kmrels_rate;
		Size		chunk_size;		/* kmrels_length * kmrels_ratio */
		int			nbatches;		/* expected iteration in this depth */
		int			nslots;			/* expected hashjoin slots width, if any */
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
	double		kresults_ratio;
	List	   *nrows_ratio;
	bool		outer_bulkload;
	Expr	   *outer_quals;
	List	   *host_quals;
	/* for each depth */
	List	   *join_types;
	List	   *join_quals;
	List	   *hash_outer_keys;
	List	   *hash_nslots;
	/* supplemental information of ps_tlist */
	List	   *ps_src_depth;	/* source depth of the ps_tlist entry */
	List	   *ps_src_resno;	/* source resno of the ps_tlist entry */
} GpuJoinInfo;

static inline void
form_gpujoin_info(CustomScan *cscan, GpuJoinInfo *gj_info)
{
	List	   *privs = NIL;
	List	   *exprs = NIL;
	long		kresults_ratio;

	privs = lappend(privs, makeInteger(gj_info->num_rels));
	privs = lappend(privs, makeString(gj_info->kern_source));
	privs = lappend(privs, makeInteger(gj_info->extra_flags));
	exprs = lappend(exprs, gj_info->used_params);
	kresults_ratio = (long)(gj_info->kresults_ratio * 1000000.0);
	privs = lappend(privs, makeInteger(kresults_ratio));
	privs = lappend(privs, gj_info->nrows_ratio);
	privs = lappend(privs, makeInteger(gj_info->outer_bulkload));
	exprs = lappend(exprs, gj_info->outer_quals);
	exprs = lappend(exprs, gj_info->host_quals);
	privs = lappend(privs, gj_info->join_types);
	exprs = lappend(exprs, gj_info->join_quals);
	exprs = lappend(exprs, gj_info->hash_outer_keys);
	privs = lappend(privs, gj_info->hash_nslots);
	privs = lappend(privs, gj_info->ps_src_depth);
	privs = lappend(privs, gj_info->ps_src_resno);

	cscan->custom_private = privs;
	cscan->custom_exprs = exprs;
}

static inline GpuJoinInfo *
deform_gpujoin_info(CustomScan *cscan)
{
	GpuJoinInfo *gj_info = palloc0(sizeof(GpuJoinInfo));
	List	   *privs = cscan->custom_private;
	List	   *exprs = cscan->custom_exprs;
	int			pindex = 0;
	int			eindex = 0;
	long		kresults_ratio;

	gj_info->num_rels = intVal(list_nth(privs, pindex++));
	gj_info->kern_source = strVal(list_nth(privs, pindex++));
	gj_info->extra_flags = intVal(list_nth(privs, pindex++));
	gj_info->used_params = list_nth(exprs, eindex++);
	kresults_ratio = intVal(list_nth(privs, pindex++));
	gj_info->kresults_ratio = (double)kresults_ratio / 1000000.0;
	gj_info->nrows_ratio = list_nth(privs, pindex++);
	gj_info->outer_bulkload = intVal(list_nth(privs, pindex++));
	gj_info->outer_quals = list_nth(exprs, eindex++);
	gj_info->host_quals = list_nth(exprs, eindex++);
	gj_info->join_types = list_nth(privs, pindex++);
	gj_info->join_quals = list_nth(exprs, eindex++);
	gj_info->hash_outer_keys = list_nth(exprs, eindex++);
	gj_info->hash_nslots = list_nth(privs, pindex++);
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
	int				num_rels;
	/* expressions to be used in fallback path */
	List		   *join_types;
	ExprState	   *outer_quals;
	List		   *hash_outer_keys;
	List		   *join_quals;
	/* current window of inner relations */
	pgstrom_multirels *curr_pmrels;
	/* format of destination store */
	int				result_format;
	/* buffer population ratio */
	int				result_width;	/* result width for buffer length calc */
	double			kresults_ratio;	/* estimated number of rows to outer */
	List		   *nrows_ratio;
	/* supplemental information to ps_tlist  */
	List		   *ps_src_depth;
	List		   *ps_src_resno;
	/* buffer for row materialization  */
	HeapTupleData	curr_tuple;
	/* buffer for PDS on bulk-loading mode */
	pgstrom_data_store *next_pds;
} GpuJoinState;

/*
 * pgstrom_gpujoin - task object of GpuJoin
 */
typedef struct
{
	GpuTask			task;
	CUfunction		kern_prep;
	CUfunction		kern_exec_nl;	/* gpujoin_exec_nestloop */
	CUfunction		kern_exec_hj;	/* gpujoin_exec_hashjoin */
	CUfunction		kern_outer_nl;	/* gpujoin_leftouter_nestloop */
	CUfunction		kern_outer_hj;	/* gpujoin_leftouter_hashjoin */
	CUfunction		kern_proj;
	CUdeviceptr		m_kgjoin;
	CUdeviceptr		m_kmrels;
	CUdeviceptr		m_kds_src;
	CUdeviceptr		m_kds_dst;
	CUdeviceptr		m_ojmaps;
	bool			is_last_chunk;	/* true, if this chunk is the last */
	bool			inner_loader;	/* true, if this task is inner loader */
	CUevent			ev_dma_send_start;
	CUevent			ev_dma_send_stop;
	CUevent			ev_kern_join_end;
	CUevent			ev_dma_recv_start;
	CUevent			ev_dma_recv_stop;
	void		   *pmrels;			/* inner multi relations */
	pgstrom_data_store *pds_src;	/* data store of outer relation */
	pgstrom_data_store *pds_dst;	/* data store of result buffer */
	kern_gpujoin	kern;			/* kern_gpujoin of this request */
} pgstrom_gpujoin;


/*
 * static function declaration
 */
static char *gpujoin_codegen(PlannerInfo *root,
							 CustomScan *cscan,
							 GpuJoinInfo *gj_info,
							 codegen_context *context);

/*
 * misc declarations
 */

/* copied from joinpath.c */
#define PATH_PARAM_BY_REL(path, rel)  \
	((path)->param_info && bms_overlap(PATH_REQ_OUTER(path), (rel)->relids))



/*****/
static inline bool
path_is_gpujoin(Path *pathnode)
{
	CustomPath *cpath = (CustomPath *) pathnode;

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
 * dump_gpujoin_path
 *
 * Dumps candidate GpuJoinPath for debugging
 */
static void
dump_gpujoin_path(PlannerInfo *root, GpuJoinPath *gpath)
{
	Relids		outer_relids;
	Relids		inner_relids;
	StringInfoData buf;
	bool		is_nestloop;
	JoinType	join_type;
	int			i, rtindex;
	bool		is_first;
	List	   *range_tables = root->parse->rtable;

	if (client_min_messages > DEBUG1)
		return;

	/* outer relids */
	outer_relids = gpath->outer_path->parent->relids;

	/* inner relids */
	inner_relids = NULL;
	for (i=0; i < gpath->num_rels; i++)
	{
		RelOptInfo *rel = gpath->inners[i].scan_path->parent;

		inner_relids = bms_union(inner_relids, rel->relids);
	}

	/* make a result */
	initStringInfo(&buf);
	is_nestloop = (gpath->inners[gpath->num_rels - 1].hash_quals == NIL);
	join_type = gpath->inners[gpath->num_rels - 1].join_type;

	appendStringInfo(&buf, "(");
	is_first = true;
	rtindex = -1;
	while ((rtindex = bms_next_member(outer_relids, rtindex)) >= 0)
	{
		RangeTblEntry  *rte = rt_fetch(rtindex, range_tables);
		Alias  *eref = rte->eref;

		appendStringInfo(&buf, "%s%s",
						 is_first ? "" : ", ",
						 eref->aliasname);
		is_first = false;
	}
	appendStringInfo(&buf, ") %s (",
					 is_nestloop
					 ? (join_type == JOIN_FULL ? "FNL" :
						join_type == JOIN_LEFT ? "LNL" :
						join_type == JOIN_RIGHT ? "RNL" : "INL")
					 : (join_type == JOIN_FULL ? "FHJ" :
						join_type == JOIN_LEFT ? "LHJ" :
                        join_type == JOIN_RIGHT ? "RHJ" : "IHJ"));
	is_first = true;
	rtindex = -1;
	while ((rtindex = bms_next_member(inner_relids, rtindex)) >= 0)
	{
		RangeTblEntry  *rte = rt_fetch(rtindex, range_tables);
		Alias  *eref = rte->eref;

		appendStringInfo(&buf, "%s%s",
						 is_first ? "" : ", ",
						 eref->aliasname);
		is_first = false;
	}
	appendStringInfo(&buf, ")");

	elog(DEBUG1, "%s Cost=%.2f..%.2f",
		 buf.data,
		 gpath->cpath.path.startup_cost,
		 gpath->cpath.path.total_cost);
	pfree(buf.data);
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
	QualCost	host_cost;
	QualCost   *join_cost;
	double		gpu_cpu_ratio;
	double		kresults_ratio;
	double		outer_ntuples;
	double		inner_ntuples;
	Size		kmrels_length;
	Size		largest_size;
	int			largest_index;
	int			i, num_rels = gpath->num_rels;

	/*
	 * Buffer size estimation
	 */
	kresults_ratio = 1.0;
	for (i=0; i < num_rels; i++)
	{
		double	temp = (double)(i+2) * gpath->inners[i].nrows_ratio;

		kresults_ratio = Max(kresults_ratio, temp);
	}
	gpath->kresults_ratio = kresults_ratio;

	/*
	 * Cost of per-tuple evaluation
	 */
	gpu_cpu_ratio = pgstrom_gpu_operator_cost / cpu_operator_cost;
	join_cost = palloc0(sizeof(QualCost) * num_rels);
	for (i=0; i < num_rels; i++)
	{
		cost_qual_eval(&join_cost[i], gpath->inners[i].join_quals, root);
		join_cost[i].per_tuple *= gpu_cpu_ratio;
	}
	cost_qual_eval(&host_cost, gpath->host_quals, root);

	/*
	 * Estimation of multi-relations buffer size
	 */
retry:
	kmrels_length = STROMALIGN(offsetof(kern_multirels,
										chunks[num_rels]));
	largest_size = 0;
	largest_index = -1;
	outer_ntuples = outer_path->rows;

	/* fixed cost to initialize/setup/use GPU device */
	startup_cost = pgstrom_gpu_setup_cost;

	for (i=0; i < num_rels; i++)
	{
		Path	   *inner_path = gpath->inners[i].scan_path;
		RelOptInfo *inner_rel = inner_path->parent;
		cl_uint		ncols = list_length(inner_rel->reltargetlist);
		Size		chunk_size;
		Size		entry_size;
		Size		htup_size;
		Size		nslots = 0;

		/* force a plausible relation size if no information.
		 * It expects 15% of margin to avoid unnecessary hash-
		 * table split
		 */
		inner_ntuples = (Max(1.15 * inner_path->rows, 1000.0)
						 / gpath->inners[i].nbatches);

		/*
		 * NOTE: RelOptInfo->width is not reliable for base relations 
		 * because this fields shows the length of attributes which
		 * are actually referenced, however, we once load physical
		 * tuple on the KDS/KHash buffer if base relation.
		 */
		htup_size = MAXALIGN(offsetof(HeapTupleHeaderData,
									  t_bits[BITMAPLEN(ncols)]));
		if (inner_rel->reloptkind != RELOPT_BASEREL)
			htup_size += MAXALIGN(inner_rel->width);
		else
		{
			double		heap_size = (double)
				(BLCKSZ - SizeOfPageHeaderData) * inner_rel->pages;

			htup_size += MAXALIGN(heap_size / Max(inner_rel->tuples, 1.0) -
								  sizeof(ItemIdData) - SizeofHeapTupleHeader);
		}

		if (gpath->inners[i].hash_quals != NIL)
		{
			/* header portion of kern_hashtable */
			chunk_size = STROMALIGN(offsetof(kern_hashtable,
											 colmeta[ncols]));
			/* hash entry slot */
			nslots = Max((Size)inner_ntuples, 1024);
			nslots = Min(nslots, gpuMemMaxAllocSize() / sizeof(void *));
			chunk_size += STROMALIGN(sizeof(cl_uint) * (Size)nslots);
			/* kern_hashentry body */
			entry_size = offsetof(kern_hashentry, htup) + htup_size;
			chunk_size += STROMALIGN(entry_size * (Size)inner_ntuples);
		}
		else
		{
			/* header portion of kern_data_store */
			chunk_size = STROMALIGN(offsetof(kern_data_store,
											 colmeta[ncols]));
			/* row-index of the tuples */
			chunk_size += STROMALIGN(sizeof(cl_uint) * (Size)inner_ntuples);
			/* kern_tupitem body */
			entry_size = offsetof(kern_tupitem, htup) + htup_size;
			chunk_size += STROMALIGN(entry_size * (Size)inner_ntuples);
		}
		gpath->inners[i].chunk_size = chunk_size;
		gpath->inners[i].nslots = nslots;

		if (largest_index < 0 || largest_size < chunk_size)
		{
			largest_size = chunk_size;
			largest_index = i;
		}
		kmrels_length += chunk_size;

		/*
		 * Cost calculation in this depth
		 */

		/* cost to load tuples onto the buffer */
		startup_cost = (inner_path->total_cost +
						cpu_tuple_cost * inner_path->rows);
		/* cost to compute hash value, if hash-join */
		if (gpath->inners[i].hash_quals != NIL)
			startup_cost += (cpu_operator_cost * inner_path->rows *
							 list_length(gpath->inners[i].hash_quals));
		/* cost to execute previous stage */
		if (i == 0)
			run_cost = (outer_path->total_cost +
						cpu_tuple_cost * outer_path->rows);
		else
			run_cost = (gpath->inners[i-1].startup_cost +
						gpath->inners[i-1].run_cost);
		/* iteration of outer scan/join */
		run_cost *= (double) gpath->inners[i].nbatches;

		/* cost to evaluate join qualifiers */
		if (gpath->inners[i].hash_quals != NIL)
			run_cost += (join_cost[i].per_tuple
						 * outer_ntuples
						 * (inner_ntuples / (double)gpath->inners[i].nslots)
						 * (double) gpath->inners[i].nbatches);
		else
			run_cost += (join_cost[i].per_tuple
						 * outer_ntuples
						 * clamp_row_est(inner_ntuples)
						 * (double) gpath->inners[i].nbatches);
		/* save the startup/run_cost in this depth */
		gpath->inners[i].startup_cost = startup_cost;
		gpath->inners[i].run_cost = run_cost;

		/* number of outer items on the next depth */
		outer_ntuples = gpath->inners[i].nrows_ratio * outer_path->rows;
	}
	/* put cost value on the gpath */
	gpath->cpath.path.startup_cost
		= gpath->inners[num_rels - 1].startup_cost;
	gpath->cpath.path.total_cost
		= (gpath->inners[num_rels - 1].startup_cost +
		   gpath->inners[num_rels - 1].run_cost);
	/*
	 * NOTE: In case when extreme number of rows are expected,
	 * it does not make sense to split hash-tables because
	 * increasion of numbatches also increases the total cost
	 * by iteration of outer scan. In this case, the best
	 * strategy is to give up this path, instead of incredible
	 * number of numbatches!
	 */
	if (!add_path_precheck(gpath->cpath.path.parent,
						   gpath->cpath.path.startup_cost,
						   gpath->cpath.path.total_cost,
						   NULL, required_outer))
		return false;

	/*
	 * If size of inner multi-relations buffer is still larger than
	 * device allocatable limitation, we try to split the largest
	 * relation then retry the estimation.
	 */
	if (kmrels_length > gpuMemMaxAllocSize())
	{
		gpath->inners[largest_index].nbatches++;
		goto retry;
	}

	/*
	 * Update estimated multi-relations buffer length and portion
	 * of the 
	 */
	gpath->kmrels_length = kmrels_length;
	for (i=0; i < num_rels; i++)
	{
		gpath->inners[i].kmrels_rate =
			((double)gpath->inners[i].chunk_size / (double)kmrels_length);
	}
	dump_gpujoin_path(root, gpath);
	return true;
}

static GpuJoinPath *
create_gpujoin_path(PlannerInfo *root,
					RelOptInfo *joinrel,
					JoinType jointype,
					Path *outer_path,
					Path *inner_path,
					SpecialJoinInfo *sjinfo,
					ParamPathInfo *param_info,
					List *hash_quals,
					List *join_quals,
					List *host_quals,
					bool can_bulkload,
					bool try_merge)
{
	GpuJoinPath	   *result;
	GpuJoinPath	   *source = NULL;
	double			nrows;
	double			nrows_ratio;
	int				num_rels;

	/*
	 * 'nrows_ratio' is used to estimate size of result buffer 
	 * for GPU kernel execution. If joinrel contains host-only
	 * qualifiers, we need to estimate number of rows at the time
	 * of host-only qualifiers.
	 */
	if (host_quals == NIL)
		nrows = joinrel->rows;
	else
	{
		RelOptInfo		dummy;

		set_joinrel_size_estimates(root, &dummy,
								   outer_path->parent,
								   inner_path->parent,
								   sjinfo,
								   join_quals);
		nrows = dummy.rows;
	}
	nrows_ratio = nrows / outer_path->rows;

	/*
	 * If expected results generated by GPU looks too large, we immediately
	 * give up to calculate this path.
	 */
	if (nrows_ratio > pgstrom_row_population_max)
	{
		elog(DEBUG1, "row population ratio (%.2f) too large, give up",
			 nrows_ratio);
		return NULL;
	}
	else if (nrows_ratio > pgstrom_row_population_max / 2.0)
	{
		elog(DEBUG1,
			 "row population ratio (%.2f) looks too large, rounded to %.2f",
			 nrows_ratio, pgstrom_row_population_max / 2.0);
		nrows_ratio = pgstrom_row_population_max / 2.0;
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
	result->cpath.path.rows = joinrel->rows;
	result->cpath.flags = (can_bulkload ? CUSTOMPATH_SUPPORT_BULKLOAD : 0);
	result->cpath.methods = &gpujoin_path_methods;
	result->outer_path = outer_path;
	result->kmrels_length = 0;		/* to be set later */
	result->kresults_ratio = 0.0;	/* to be set later */
	result->num_rels = num_rels;
	result->host_quals = host_quals;
	if (source && num_rels > 1)
	{
		memcpy(result->inners, source->inners,
			   offsetof(GpuJoinPath, inners[num_rels - 1]) -
			   offsetof(GpuJoinPath, inners[0]));
	}
	result->inners[num_rels - 1].startup_cost = 0.0;	/* to be set later */
	result->inners[num_rels - 1].run_cost = 0.0;		/* to be set later */
	result->inners[num_rels - 1].nrows_ratio = nrows_ratio;
	result->inners[num_rels - 1].scan_path = inner_path;
	result->inners[num_rels - 1].join_type = jointype;
	result->inners[num_rels - 1].hash_quals = hash_quals;
	result->inners[num_rels - 1].join_quals = join_quals;
	result->inners[num_rels - 1].kmrels_rate = 0.0;		/* to be set later */
	result->inners[num_rels - 1].nbatches = 1;			/* to be set later */
	result->inners[num_rels - 1].nslots = 0;			/* to be set later */

	return result;
}

static void
try_gpujoin_path(PlannerInfo *root,
				 RelOptInfo *joinrel,
				 Path *outer_path,
                 Path *inner_path,
				 List *restrictlist,	/* host + join quals */
				 JoinType jointype,
				 SpecialJoinInfo *sjinfo,
				 Relids param_source_rels,
				 Relids extra_lateral_rels,
				 List *hash_quals,
				 List *join_quals,
				 List *host_quals)
{
	GpuJoinPath	   *gpath;
	ParamPathInfo  *param_info;
	Relids			required_outer;
	ListCell	   *lc;
	bool			can_bulkload = false;

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
			can_bulkload = true;
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
										   &restrictlist);

	/*
	 * Try GpuHashJoin logic
	 */
	if (enable_gpuhashjoin && hash_quals != NIL)
	{
		gpath = create_gpujoin_path(root, joinrel, jointype,
									outer_path, inner_path,
									sjinfo, param_info,
									hash_quals, join_quals, host_quals,
									can_bulkload, false);
		if (gpath != NULL &&
			cost_gpujoin(root, gpath, required_outer))
			add_path(joinrel, &gpath->cpath.path);

		if (path_is_mergeable_gpujoin(outer_path))
		{
			gpath = create_gpujoin_path(root, joinrel, jointype,
										outer_path, inner_path,
										sjinfo, param_info,
										hash_quals, join_quals, host_quals,
										can_bulkload, true);
			if (gpath != NULL &&
				cost_gpujoin(root, gpath, required_outer))
				add_path(joinrel, &gpath->cpath.path);
		}
	}

	/*
	 * Try GpuNestLoop logic
	 */
	if (enable_gpunestloop &&
		(jointype == JOIN_INNER || jointype == JOIN_LEFT))
	{
		gpath = create_gpujoin_path(root, joinrel, jointype,
									outer_path, inner_path,
									sjinfo, param_info,
									NIL, join_quals, host_quals,
									can_bulkload, false);
		if (gpath != NULL &&
			cost_gpujoin(root, gpath, required_outer))
			add_path(joinrel, &gpath->cpath.path);

		if (path_is_mergeable_gpujoin(outer_path))
		{
			gpath = create_gpujoin_path(root, joinrel, jointype,
										outer_path, inner_path,
										sjinfo, param_info,
										NIL, join_quals, host_quals,
										can_bulkload, true);
			if (gpath != NULL &&
				cost_gpujoin(root, gpath, required_outer))
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
	 * If either cheapest-total path is parameterized by the other rel, we
	 * can't use a hashjoin.  (There's no use looking for alternative
	 * input paths, since these should already be the least-parameterized
	 * available paths.)
	 */
	if (PATH_PARAM_BY_REL(cheapest_total_outer, innerrel) ||
		PATH_PARAM_BY_REL(cheapest_total_inner, outerrel))
		return;

	/*
	 * Check restrictions of joinrel.
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
	 * If no qualifiers are executable on GPU device, it does not make
	 * sense to run with GpuJoin node. So, we add no paths here.
	 */
	if (!join_quals)
		return;

	if (cheapest_startup_outer)
	{
		/* GpuHashJoin logic, if possible */
		if (hash_quals != NIL)
			try_gpujoin_path(root,
							 joinrel,
							 cheapest_startup_outer,
							 cheapest_total_inner,
							 restrictlist,
							 jointype,
							 sjinfo,
							 param_source_rels,
							 extra_lateral_rels,
							 hash_quals,
							 join_quals,
							 host_quals);
		/* GpuNestLoop logic, if possible */
		if (jointype == JOIN_INNER || jointype == JOIN_RIGHT)
			try_gpujoin_path(root,
							 joinrel,
							 cheapest_startup_outer,
							 cheapest_total_inner,
							 restrictlist,
							 jointype,
							 sjinfo,
							 param_source_rels,
							 extra_lateral_rels,
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
							 cheapest_total_outer,
							 cheapest_total_inner,
							 restrictlist,
							 jointype,
							 sjinfo,
							 param_source_rels,
							 extra_lateral_rels,
							 hash_quals,
							 join_quals,
							 host_quals);
		/* GpuNestLoop logic, if possible */
		if (jointype == JOIN_INNER || jointype == JOIN_RIGHT)
			try_gpujoin_path(root,
							 joinrel,
							 cheapest_total_outer,
							 cheapest_total_inner,
							 restrictlist,
							 jointype,
							 sjinfo,
							 param_source_rels,
							 extra_lateral_rels,
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

	foreach (lc, clauses)
	{
		Node   *expr = lfirst(lc);

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

	build_pseudo_targetlist_walker((Node *)targetlist, &context);
	build_pseudo_targetlist_walker((Node *)gj_info->host_quals, &context);

	/*
	 * Above are host referenced columns. On the other hands, the columns
	 * newly added below are device-only columns, so it will never
	 * referenced by the host-side. We mark it resjunk=true.
	 */
	context.resjunk = true;
	build_pseudo_targetlist_walker((Node *)gj_info->hash_outer_keys, &context);
	build_pseudo_targetlist_walker((Node *)gj_info->join_quals, &context);
	build_pseudo_targetlist_walker((Node *)gj_info->outer_quals, &context);

    Assert(list_length(context.ps_tlist) == list_length(context.ps_depth) &&
           list_length(context.ps_tlist) == list_length(context.ps_resno));

	gj_info->ps_src_depth = context.ps_depth;
	gj_info->ps_src_resno = context.ps_resno;

	return context.ps_tlist;
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
	GpuJoinInfo		gj_info;
	CustomScan	   *cscan;
	Plan		   *prev_plan = NULL;
	Plan		   *outer_plan;
	codegen_context	context;
	ListCell	   *lc;
	int				i;

	cscan = makeNode(CustomScan);
	cscan->scan.plan.targetlist = tlist;
	cscan->scan.plan.qual = gpath->host_quals;
	cscan->flags = best_path->flags;
	cscan->methods = &gpujoin_plan_methods;

	memset(&gj_info, 0, sizeof(GpuJoinInfo));
	gj_info.num_rels = gpath->num_rels;
	gj_info.kresults_ratio = gpath->kresults_ratio;
	gj_info.host_quals = extract_actual_clauses(gpath->host_quals, false);
	for (i=0; i < gpath->num_rels; i++)
	{
		CustomScan	   *mplan;
		List		   *hash_inner_keys = NIL;
		List		   *hash_outer_keys = NIL;
		List		   *clauses;
		int				nrows_ratio;

		foreach (lc, gpath->inners[i].hash_quals)
		{
			Path		   *scan_path = gpath->inners[i].scan_path;
			RelOptInfo	   *scan_rel = scan_path->parent;
			RestrictInfo   *rinfo = lfirst(lc);
			OpExpr		   *op_clause = (OpExpr *) rinfo->clause;
			Relids			relids1;
			Relids			relids2;
			Node		   *arg1;
			Node		   *arg2;

			Assert(is_opclause(op_clause));
			arg1 = (Node *) linitial(op_clause->args);
			arg2 = (Node *) lsecond(op_clause->args);
			relids1 = pull_varnos(arg1);
			relids2 = pull_varnos(arg2);
			if (bms_is_subset(relids1, scan_rel->relids) &&
				!bms_is_subset(relids2, scan_rel->relids))
			{
				hash_inner_keys = lappend(hash_inner_keys, arg1);
				hash_outer_keys = lappend(hash_outer_keys, arg2);
			}
			else if (bms_is_subset(relids2, scan_rel->relids) &&
					 !bms_is_subset(relids1, scan_rel->relids))
			{
				hash_inner_keys = lappend(hash_inner_keys, arg2);
				hash_outer_keys = lappend(hash_outer_keys, arg1);
			}
			else
				elog(ERROR, "Bug? hash-clause reference bogus varnos");
		}
		mplan = multirels_create_plan(root,
									  i + 1,	/* depth */
									  gpath->inners[i].startup_cost,
									  gpath->inners[i].startup_cost +
									  gpath->inners[i].run_cost,
									  gpath->inners[i].join_type,
									  gpath->inners[i].scan_path,
									  gpath->kmrels_length,
									  gpath->inners[i].kmrels_rate,
									  gpath->inners[i].nbatches,
									  gpath->inners[i].nslots,
									  hash_inner_keys);
		gpath->inners[i].scan_plan = (Plan *) mplan;
		/* add properties of GpuJoinInfo */
		gj_info.join_types = lappend_int(gj_info.join_types,
										 gpath->inners[i].join_type);
		clauses = extract_actual_clauses(gpath->inners[i].join_quals, false);
		gj_info.join_quals =
			lappend(gj_info.join_quals, build_flatten_qualifier(clauses));
		gj_info.hash_outer_keys = lappend(gj_info.hash_outer_keys,
										  hash_outer_keys);
		nrows_ratio = (int)(gpath->inners[i].nrows_ratio * 1000000.0);
		gj_info.nrows_ratio = lappend_int(gj_info.nrows_ratio, nrows_ratio);
		/* chain it under the GpuJoin */
		if (prev_plan)
			innerPlan(prev_plan) = &mplan->scan.plan;
		else
			innerPlan(cscan) = &mplan->scan.plan;
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
	outerPlan(cscan) = outer_plan;
	gpath->outer_plan = outer_plan;		/* for convenience */

	/*
	 * Build a pseudo-scan targetlist
	 */
	cscan->custom_ps_tlist = build_pseudo_targetlist(gpath, &gj_info, tlist);

	/*
	 * construct kernel code
	 */
	pgstrom_init_codegen_context(&context);
	context.pseudo_tlist = cscan->custom_ps_tlist;

	gj_info.kern_source = gpujoin_codegen(root, cscan, &gj_info, &context);
	gj_info.extra_flags = DEVKERNEL_NEEDS_GPUJOIN | context.extra_flags;
	gj_info.used_params = context.used_params;

	form_gpujoin_info(cscan, &gj_info);

	return &cscan->scan.plan;
}

static void
gpujoin_textout_path(StringInfo str, const CustomPath *node)
{
	GpuJoinPath *gpath = (GpuJoinPath *) node;
	int		i;

	/* outer_path */
	appendStringInfo(str, " :outer_path %s",
					 nodeToString(gpath->outer_path));
	/* outer_plan */
	appendStringInfo(str, " :outer_plan %s",
					 nodeToString(gpath->outer_plan));
	/* total_length */
	appendStringInfo(str, " :kmrels_length %zu", gpath->kmrels_length);
	/* kresults_ratio */
	appendStringInfo(str, " :kresults_ratio %.2f", gpath->kresults_ratio);
	/* num_rels */
	appendStringInfo(str, " :num_rels %d", gpath->num_rels);
	/* host_quals */
	appendStringInfo(str, " :host_quals %s", nodeToString(gpath->host_quals));
	/* inner relations */
	appendStringInfo(str, " :inners (");
	for (i=0; i < gpath->num_rels; i++)
	{
		appendStringInfo(str, "{");
		/* startup_cost, run_cost */
		appendStringInfo(str, " :startup_cost %.2f",
						 gpath->inners[i].startup_cost);
		appendStringInfo(str, " :run_cost %.2f",
						 gpath->inners[i].run_cost);
		/* join_type */
		appendStringInfo(str, " :join_type %d",
						 (int)gpath->inners[i].join_type);
		/* scan_path */
		appendStringInfo(str, " :scan_path %s",
						 nodeToString(gpath->inners[i].scan_path));
		/* scan_plan */
		appendStringInfo(str, " :scan_plan %s",
						 nodeToString(gpath->inners[i].scan_plan));
		/* hash_quals */
		appendStringInfo(str, " :hash_quals %s",
						 nodeToString(gpath->inners[i].hash_quals));
		/* join_quals */
		appendStringInfo(str, " :join_clause %s",
						 nodeToString(gpath->inners[i].join_quals));
		/* nrows_ratio */
		appendStringInfo(str, " :nrows_ratio %.2f",
						 gpath->inners[i].nrows_ratio);
		/* kmrels_rate */
		appendStringInfo(str, " :kmrels_rate %.2f",
						 gpath->inners[i].kmrels_rate);
		/* chunk_size */
		appendStringInfo(str, " :chunk_size %zu",
						 gpath->inners[i].chunk_size);
		/* nbatches */
		appendStringInfo(str, " :nbatches %d",
						 gpath->inners[i].nbatches);
		/* nslots */
		appendStringInfo(str, " :nslots %d",
						 gpath->inners[i].nslots);
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
    gjs->gts.css.flags = node->flags;
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
	CustomScan	   *cscan = (CustomScan *) node->ss.ps.plan;
	GpuJoinInfo	   *gj_info = deform_gpujoin_info(cscan);
	TupleDesc		tupdesc = GTS_GET_RESULT_TUPDESC(gjs);

	gjs->num_rels = gj_info->num_rels;

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
	gjs->outer_quals = ExecInitExpr((Expr *)gj_info->outer_quals, ps);
	gjs->hash_outer_keys = (List *)
		ExecInitExpr((Expr *)gj_info->hash_outer_keys, ps);
	gjs->join_quals = (List *)
		ExecInitExpr((Expr *)gj_info->join_quals, ps);
	gjs->gts.css.ss.ps.qual = (List *)
		ExecInitExpr((Expr *)gj_info->host_quals, ps);

	/* needs to track corresponding columns */
	gjs->ps_src_depth = gj_info->ps_src_depth;
	gjs->ps_src_resno = gj_info->ps_src_resno;

	/*
	 * initialization of child nodes
	 */
	outerPlanState(gjs) = ExecInitNode(outerPlan(cscan), estate, eflags);
	innerPlanState(gjs) = ExecInitNode(innerPlan(cscan), estate, eflags);

	/*
	 * Is bulkload available?
	 */
	gjs->gts.scan_bulk =
		(!pgstrom_debug_bulkload_enabled ? false : gj_info->outer_bulkload);

	/*
	 * initialize kernel execution parameter
	 */
	pgstrom_assign_cuda_program(&gjs->gts,
								gj_info->kern_source,
								gj_info->extra_flags);
	gjs->kparams = pgstrom_create_kern_parambuf(gj_info->used_params,
												ps->ps_ExprContext);
	if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
		pgstrom_preload_cuda_program(&gjs->gts);

	/*
	 * initialize misc stuff
	 */
	if ((gjs->gts.css.flags & CUSTOMPATH_PREFERE_ROW_FORMAT) != 0)
		gjs->result_format = KDS_FORMAT_ROW;
	else
		gjs->result_format = KDS_FORMAT_SLOT;

	/* expected kresults buffer expand rate */
	gjs->result_width =
		MAXALIGN(offsetof(HeapTupleHeaderData,
						  t_bits[BITMAPLEN(tupdesc->natts)]) +
				 (tupdesc->tdhasoid ? sizeof(Oid) : 0)) +
		MAXALIGN(cscan->scan.plan.plan_width);	/* average width */
	gjs->kresults_ratio = gj_info->kresults_ratio;
	gjs->nrows_ratio = gj_info->nrows_ratio;
}

static TupleTableSlot *
gpujoin_exec(CustomScanState *node)
{
	return ExecScan(&node->ss,
					(ExecScanAccessMtd) pgstrom_exec_gputask,
					(ExecScanRecheckMtd) pgstrom_recheck_gputask);
}

static void *
gpujoin_exec_bulk(CustomScanState *node)
{
	GpuJoinState	   *gjs = (GpuJoinState *) node;
	pgstrom_gpujoin	   *pgjoin;
	pgstrom_data_store *pds_dst;

	/* force to return row-format */
	gjs->result_format = KDS_FORMAT_ROW;

	/* fetch next chunk to be processed */
	pgjoin = (pgstrom_gpujoin *) pgstrom_fetch_gputask(&gjs->gts);
	if (!pgjoin)
		return NULL;

	/* extract its destination data-store */
	pds_dst = pgjoin->pds_dst;
	pgjoin->pds_dst = NULL;
	/* release this pgstrom_gpujoin */
	pgstrom_release_gputask(&pgjoin->task);

	return pds_dst;
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
	gjs->gts.scan_overflow = NULL;
	ExecReScan(outerPlanState(gjs));

	/*
	 * we reuse the inner hash table if it is flat (that means mhtables
	 * is not divided into multiple portions) and no parameter changed.
	 */

	/*
	 * FIXME: need to consider how to detach multi_relations chunk
	 * if concurrent tasks may be still working on.
	 */
	if (gjs->curr_pmrels)
	{
		// release curr_pmrels

		/*
		 * if chgParam of subnode is not null then plan will be re-scanned
		 * by first ExecProcNode.
		 */
		if (innerPlanState(gjs)->chgParam == NULL)
			ExecReScan(innerPlanState(gjs));
	}

#if 0
	if ((pmrels && pmrels->is_divided) ||
		innerPlanState(gjs)->chgParam != NULL)
	{

		gjs->pmrels = NULL;
	}
#endif
}

static void
gpujoin_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
	GpuJoinState   *gjs = (GpuJoinState *) node;
	CustomScan	   *cscan = (CustomScan *) node->ss.ps.plan;
	GpuJoinInfo	   *gj_info = deform_gpujoin_info(cscan);
	List		   *context;
	ListCell	   *lc1;
	ListCell	   *lc2;
	ListCell	   *lc3;
	char		   *temp;
	char			qlabel[128];
	int				depth;
	StringInfoData	str;

	initStringInfo(&str);

	/* name lookup context */
	context =  set_deparse_context_planstate(es->deparse_cxt,
											 (Node *) node,
											 ancestors);
	/* pseudo scan tlist if verbose */
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
		ExplainPropertyText("Pseudo Scan", str.data, es);
	}

	/* outer bulkload */
	ExplainPropertyText("Bulkload", gjs->gts.scan_bulk ? "On" : "Off", es);

	/* outer qualifier if any */
	if (gj_info->outer_quals)
	{
		temp = deparse_expression((Node *)gj_info->outer_quals,
								  context, es->verbose, false);
		ExplainPropertyText("OuterQual", temp, es);
	}

	/* join-qualifiers */
	depth = 1;
	forthree (lc1, gj_info->join_types,
			  lc2, gj_info->join_quals,
			  lc3, gj_info->hash_outer_keys)
	{
		JoinType	join_type = (JoinType) lfirst_int(lc1);
		Expr	   *join_qual = lfirst(lc2);
		Expr	   *hash_outer_key = lfirst(lc3);

		resetStringInfo(&str);
		if (hash_outer_key != NULL)
		{
			appendStringInfo(&str, "Logic: GpuHash%sJoin",
							 join_type == JOIN_FULL ? "Full" :
							 join_type == JOIN_LEFT ? "Left" :
							 join_type == JOIN_RIGHT ? "Right" : "");
		}
		else
		{
			appendStringInfo(&str, "Logic: GpuNestLoop%s",
							 join_type == JOIN_FULL ? "Full" :
							 join_type == JOIN_LEFT ? "Left" :
							 join_type == JOIN_RIGHT ? "Right" : "");
		}

		if (hash_outer_key)
		{
			temp = deparse_expression((Node *)hash_outer_key,
                                      context, es->verbose, false);
			appendStringInfo(&str, ", HashKeys: (%s)", temp);
		}
		temp = deparse_expression((Node *)join_qual, context,
								  es->verbose, false);
		appendStringInfo(&str, ", JoinQual: %s", temp);

		snprintf(qlabel, sizeof(qlabel), "Depth %d", depth);
		ExplainPropertyText(qlabel, str.data, es);

		depth++;
	}
	/* host qualifier if any */
	if (gj_info->host_quals)
	{
		temp = deparse_expression((Node *)gj_info->host_quals,
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
	bool		needs_khtable = false;
	List	   *kern_vars = NIL;
	ListCell   *cell;
	int			depth;
	char	   *param_decl;

	Assert(cur_depth > 0 && cur_depth <= gj_info->num_rels);
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
				if (src_depth < 0 || src_depth > cur_depth)
					elog(ERROR, "Bug? device varnode out of range");
				else if (src_depth > 0)
				{
					if (list_nth(gj_info->hash_outer_keys, src_depth - 1))
						needs_khtable = true;	/* inner hashtable reference */
					else
						needs_kds_in = true;	/* inner datastore reference */
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
		if (kern_vars == NIL)
			kern_vars = list_make1(kernode);
		else
		{
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
			if (lc1 == NULL)
				kern_vars = lappend(kern_vars, kernode);
		}
	}

	/*
	 * variable declarations
	 */
	appendStringInfo(
		source,
		"  HeapTupleHeaderData *htup;\n"
		"  kern_colmeta *colmeta;\n");
	if (needs_kds_in)
		appendStringInfo(source, "  kern_data_store *kds_in;\n");
	if (needs_khtable)
		appendStringInfo(source, "  kern_hashtable *khtable;\n");
	appendStringInfo(source, "  void *datum;\n");

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
	appendStringInfo(source, "%s\n", param_decl);

	/*
	 * variable initialization
	 */
	depth = -1;
	foreach (cell, kern_vars)
	{
		Var			   *keynode = lfirst(cell);
		devtype_info   *dtype;

		dtype = pgstrom_devtype_lookup(keynode->vartype);
		if (!dtype)
			elog(ERROR, "device type \"%s\" not found",
				 format_type_be(keynode->vartype));

		if (depth != keynode->varno)
		{
			if (keynode->varno == 0)
			{
				/* htup from KDS */
				appendStringInfo(
					source,
					"  /* variable load in depth-0 (outer KDS) */\n"
					"  colmeta = kds->colmeta;\n"
					"  htup = (!o_buffer ? NULL :\n"
					"          GPUJOIN_REF_HTUP(kds,o_buffer[0]));\n"
					);
			}
			else if (list_nth(gj_info->hash_outer_keys, keynode->varno - 1))
			{
				/* in case of inner hash table */
				appendStringInfo(
					source,
					"  /* variables load in depth-%u (hash table) */\n"
					"  khtable = KERN_MULTIRELS_INNER_HASH(kmrels, %u);\n"
					"  assert(khtable != NULL);\n"
					"  colmeta = khtable->colmeta;\n",
					keynode->varno,
					keynode->varno);
				if (keynode->varno < cur_depth)
					appendStringInfo(
						source,
						"  htup = (!o_buffer ? NULL :\n"
						"          GPUJOIN_REF_HTUP(khtable,o_buffer[%d]));\n",
						keynode->varno);
				else if (keynode->varno == cur_depth)
					appendStringInfo(
						source,
						"  htup = i_htup;\n"
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
					keynode->varno,
					keynode->varno);
				if (keynode->varno < cur_depth)
					appendStringInfo(
						source,
						"  htup = (!o_buffer ? NULL :\n"
						"          GPUJOIN_REF_HTUP(kds_in,o_buffer[%d]));\n",
						keynode->varno);
				else if (keynode->varno == cur_depth)
					appendStringInfo(
						source,
						"  htup = i_htup;\n"
						);
				else
					elog(ERROR, "Bug? too deeper varnode reference");
			}
			depth = keynode->varno;
		}

		if (is_nestloop)
		{
			appendStringInfo(
				source,
				"  if (get_local_%s() == 0)\n"
				"  {\n"
				"    datum = GPUJOIN_REF_DATUM(colmeta,htup,%u);\n"
				"    SHARED_WORKMEM(pg_%s_t)[get_local_%s()]\n"
				"      = pg_%s_datum_ref(errcode, datum, false);\n"
				"  }\n"
				"  __syncthreads();\n"
				"  KVAR_%u = SHARED_WORKMEM(pg_%s_t)[get_local_%s()];\n"
				"  __syncthreads();\n"
				"\n",
				keynode->varno == cur_depth ? "xid" : "yid",
				keynode->varattno - 1,
				dtype->type_name,
				keynode->varno == cur_depth ? "yid" : "xid",
				dtype->type_name,
				keynode->varoattno,
				dtype->type_name,
				keynode->varno == cur_depth ? "yid" : "xid");
		}
		else
		{
			appendStringInfo(
				source,
				"  datum = GPUJOIN_REF_DATUM(colmeta,htup,%u);\n"
				"  KVAR_%u = pg_%s_datum_ref(errcode,datum,false);\n"
				"\n",
				keynode->varattno - 1,
				keynode->varoattno,
				dtype->type_name);
		}
	}
}

/*
 * codegen for:
 * STATIC_FUNCTION(cl_bool)
 * gpujoin_join_quals_depth%u(cl_int *errcode,
 *                            kern_parambuf *kparams,
 *                            kern_data_store *kds,
 *                            kern_multirels *kmrels,
 *                            cl_int *o_buffer,
 *                            HeapTupleHeaderData *i_htup)
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
	join_qual = list_nth(gj_info->join_quals, cur_depth - 1);

	/*
	 * make a text representation of join_qual
	 */
	context->used_vars = NIL;
	context->param_refs = NULL;
	join_code = pgstrom_codegen_expression((Node *) join_qual, context);

	/*
	 * function declaration
	 */
	appendStringInfo(
		source,
		"STATIC_FUNCTION(cl_bool)\n"
		"gpujoin_join_quals_depth%d(cl_int *errcode,\n"
		"                           kern_parambuf *kparams,\n"
		"                           kern_data_store *kds,\n"
        "                           kern_multirels *kmrels,\n"
		"                           cl_int *o_buffer,\n"
		"                           HeapTupleHeaderData *i_htup)\n"
		"{\n",
		cur_depth);
	/*
	 * variable/params declaration & initialization
	 */
	gpujoin_codegen_var_param_decl(source, gj_info, cur_depth, context);

	/*
	 * evaluate join qualifier
	 */
	appendStringInfo(
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
 *                            cl_uint *pg_crc32_table,
 *                            kern_data_store *kds,
 *                            kern_multirels *kmrels,
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
	ListCell	   *lc;

	Assert(cur_depth > 0 && cur_depth <= gj_info->num_rels);
	hash_outer_keys = list_nth(gj_info->hash_outer_keys, cur_depth - 1);
	Assert(hash_outer_keys != NIL);

	appendStringInfo(
		source,
		"STATIC_FUNCTION(cl_uint)\n"
		"gpujoin_hash_value_depth%u(cl_int *errcode,\n"
		"                           kern_parambuf *kparams,\n"
		"                           cl_uint *pg_crc32_table,\n"
		"                           kern_data_store *kds,\n"
		"                           kern_multirels *kmrels,\n"
		"                           cl_int *o_buffer)\n"
		"{\n"
		"  cl_uint hash;\n",
		cur_depth);

	context->used_vars = NIL;
	context->param_refs = NULL;

	initStringInfo(&body);
	appendStringInfo(
		&body,
		"  /* Hash-value calculation */\n"
		"  INIT_LEGACY_CRC32(hash);\n");
	foreach (lc, hash_outer_keys)
	{
		Node	   *key_expr = lfirst(lc);
		Oid			key_type = exprType(key_expr);
		devtype_info *dtype;
		char	   *temp;

		dtype = pgstrom_devtype_lookup(key_type);
		if (!dtype)
			elog(ERROR, "Bug? device type \"%s\" not found",
                 format_type_be(key_type));
		temp = pgstrom_codegen_expression(key_expr, context);
		appendStringInfo(
			&body,
			"  hash = pg_%s_comp_crc32(pg_crc32_table, hash, %s);\n",
			dtype->type_name,
			temp);
		pfree(temp);
	}
	appendStringInfo(&body, "  FIN_LEGACY_CRC32(hash);\n");

	/*
	 * variable/params declaration & initialization
	 */
	gpujoin_codegen_var_param_decl(source, gj_info, cur_depth, context);

	appendStringInfo(
		source,
		"%s"
		"  return hash;\n"
		"}\n"
		"\n",
		body.data);
	pfree(body.data);
}

/*
 * codegen for:
 * STATIC_FUNCTION(void)
 * gpujoin_projection_mapping(cl_int dest_resno,
 *                            cl_int *src_depth,
 *                            cl_int *src_colidx);
 */
static void
gpujoin_codegen_projection_mapping(StringInfo source,
								   GpuJoinInfo *gj_info,
								   codegen_context *context)
{
	ListCell   *lc1;
	ListCell   *lc2;
	ListCell   *lc3;

	appendStringInfo(
		source,
		"STATIC_FUNCTION(void)\n"
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
				CustomScan *cscan,
				GpuJoinInfo *gj_info,
				codegen_context *context)
{
	StringInfoData decl;
	StringInfoData source;
	const char *args;
	int			depth;
	ListCell   *cell;

	initStringInfo(&decl);
	initStringInfo(&source);

	/* gpujoin_outer_quals  */
	gpujoin_codegen_outer_quals(&source, gj_info, context);

	/* gpujoin_join_quals */
	for (depth=1; depth <= gj_info->num_rels; depth++)
		gpujoin_codegen_join_quals(&source, gj_info, depth, context);
	appendStringInfo(
		&source,
		"STATIC_FUNCTION(cl_bool)\n"
		"gpujoin_join_quals(cl_int *errcode,\n"
		"                   kern_parambuf *kparams,\n"
		"                   kern_data_store *kds,\n"
		"                   kern_multirels *kmrels,\n"
		"                   int depth,\n"
		"                   cl_int *outer_index,\n"
		"                   HeapTupleHeaderData *i_htup)\n"
		"{\n"
		"  switch (depth)\n"
		"  {\n");

	args = "errcode, kparams, kds, kmrels, outer_index, i_htup";
	for (depth=1; depth <= gj_info->num_rels; depth++)
	{
		appendStringInfo(
			&source,
			"  case %d:\n"
			"    return gpujoin_join_quals_depth%d(%s);\n",
			depth, depth, args);
	}
	appendStringInfo(
		&source,
		"  default:\n"
		"    STROM_SET_ERROR(errcode, StromError_SanityCheckViolation);\n"
		"    break;\n"
		"  }\n"
		"  return false;\n"
		"}\n\n");


	depth = 1;
	foreach (cell, gj_info->hash_outer_keys)
	{
		if (lfirst(cell) != NULL)
			gpujoin_codegen_hash_value(&source, gj_info, depth, context);
		depth++;
	}

	/* gpujoin_hash_value */
	appendStringInfo(
		&source,
		"STATIC_FUNCTION(cl_uint)\n"
		"gpujoin_hash_value(cl_int *errcode,\n"
		"                   kern_parambuf *kparams,\n"
		"                   cl_uint *pg_crc32_table,\n"
		"                   kern_data_store *kds,\n"
		"                   kern_multirels *kmrels,\n"
		"                   cl_int depth,\n"
		"                   cl_int *o_buffer)\n"
		"{\n"
		"  switch (depth)\n"
		"  {\n");
	args = "errcode,kparams,pg_crc32_table,kds,kmrels,o_buffer";
	depth = 1;
	foreach (cell, gj_info->hash_outer_keys)
	{
		if (lfirst(cell) != NULL)
		{
			appendStringInfo(
				&source,
				"  case %u:\n"
				"    return gpujoin_hash_value_depth%u(%s);\n",
				depth, depth, args);
		}
		depth++;
	}
	appendStringInfo(
		&source,
		"  default:\n"
		"    STROM_SET_ERROR(errcode, StromError_SanityCheckViolation);\n"
		"    break;\n"
		"  }\n"
		"  return (cl_uint)(-1);\n"
		"}\n"
		"\n");

	/* gpujoin_projection_mapping */
	gpujoin_codegen_projection_mapping(&source, gj_info, context);

	/* */
	appendStringInfo(&decl, "%s\n%s",
					 pgstrom_codegen_func_declarations(context),
					 source.data);
	pfree(source.data);

	return decl.data;
}

static GpuTask *
gpujoin_create_task(GpuJoinState *gjs, pgstrom_data_store *pds_src)
{
	GpuContext		   *gcontext = gjs->gts.gcontext;
	pgstrom_gpujoin	   *pgjoin;
	kern_gpujoin	   *kgjoin;
	kern_parambuf	   *kparams;
	TupleDesc			tupdesc;
	cl_uint				nrooms;
	cl_uint				total_items;
	Size				kgjoin_head;
	Size				required;
	pgstrom_data_store *pds_dst;

	/*
	 * Allocation of pgstrom_gpujoin task object
	 */
	kgjoin_head = (offsetof(pgstrom_gpujoin, kern) +
				   offsetof(kern_gpujoin, kparams) +
				   STROMALIGN(gjs->kparams->length));
	required = (kgjoin_head +
				STROMALIGN(offsetof(kern_resultbuf, results[0])));
	pgjoin = MemoryContextAllocZero(gcontext->memcxt, required);
	pgstrom_init_gputask(&gjs->gts, &pgjoin->task);
	pgjoin->pmrels = multirels_attach_buffer(gjs->curr_pmrels);
	pgjoin->pds_src = pds_src;

	/*
	 * Last chunk checks - this information is needed to handle left outer
	 * join case because last chunk also kicks special kernel to generate
	 * half-null tuples on GPU.
	 */
	if (!gjs->gts.scan_bulk)
		pgjoin->is_last_chunk = (gjs->gts.scan_overflow == NULL);
	else
		pgjoin->is_last_chunk = (gjs->next_pds == NULL);

	/*
	 * Setup kern_gpujoin
	 */
	total_items = (cl_uint)((double)pds_src->kds->nitems *
							gjs->kresults_ratio *
							(1.0 + pgstrom_row_population_margin));
	kgjoin = &pgjoin->kern;
	kgjoin->kresults_1_offset = kgjoin_head;
	kgjoin->kresults_2_offset = kgjoin_head +
		STROMALIGN(offsetof(kern_resultbuf, results[total_items]));
	kgjoin->kresults_total_items = total_items;
	kgjoin->kresults_max_items = 0;
	kgjoin->max_depth = gjs->num_rels;
	kgjoin->errcode = StromError_Success;

	kparams = KERN_GPUJOIN_PARAMBUF(kgjoin);
	memcpy(kparams, gjs->kparams, gjs->kparams->length);

	/*
	 * Allocation of the destination data-store
	 */
	nrooms = (cl_uint)((double) pds_src->kds->nitems *
					   (double) llast_int(gjs->nrows_ratio) / 1000000.0 *
					   (1.0 + pgstrom_row_population_margin));
	tupdesc = gjs->gts.css.ss.ss_ScanTupleSlot->tts_tupleDescriptor;

	if (gjs->result_format == KDS_FORMAT_SLOT)
	{
		pds_dst = pgstrom_create_data_store_slot(gcontext, tupdesc,
												 nrooms, false, NULL);
	}
	else if (gjs->result_format == KDS_FORMAT_ROW)
	{
		Size		length;

		length = (STROMALIGN(offsetof(kern_data_store,
									  colmeta[tupdesc->natts])) +
				  STROMALIGN(sizeof(cl_uint) * nrooms) +
				  gjs->result_width * nrooms);
		pds_dst = pgstrom_create_data_store_row(gcontext, tupdesc,
												length, false);
	}
	else
		elog(ERROR, "Bug? unexpected result format: %d", gjs->result_format);

	pgjoin->pds_dst = pds_dst;

	return &pgjoin->task;
}


static GpuTask *
gpujoin_next_chunk(GpuTaskState *gts)
{
	GpuJoinState   *gjs = (GpuJoinState *) gts;
	PlanState	   *outer_node = outerPlanState(gjs);
	TupleDesc		tupdesc = ExecGetResultType(outer_node);
	pgstrom_data_store *pds = NULL;
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

	if (gjs->gts.scan_done || !gjs->curr_pmrels)
	{
		PlanState  *mrs = innerPlanState(gjs);
		pgstrom_multirels *pmrels;

		/* unlink previous inner multi-relations */
		if (gjs->curr_pmrels)
		{
			Assert(gjs->gts.scan_done);
			multirels_detach_buffer(gjs->curr_pmrels);
			gjs->curr_pmrels = NULL;
		}

		/* load an inner multi-relations buffer */
		pmrels = pgstrom_multirels_exec_bulk(mrs);
		if (!pmrels)
		{
			PERFMON_END(&gts->pfm_accum,
						time_inner_load, &tv1, &tv2);
			return NULL;	/* end of inner multi-relations */
		}
		gjs->curr_pmrels = multirels_attach_buffer(pmrels);

		/*
		 * Rewind the outer scan pointer, if it is not first time
		 */
		if (gjs->gts.scan_done)
		{
			ExecReScan(outerPlanState(gjs));
			gjs->gts.scan_done = false;
		}
	}
	PERFMON_END(&gjs->gts.pfm_accum, time_inner_load, &tv1, &tv2);

	if (!gjs->gts.scan_bulk)
	{
		while (true)
		{
			TupleTableSlot *slot;

			if (gjs->gts.scan_overflow)
			{
				slot = gjs->gts.scan_overflow;
				gjs->gts.scan_overflow = NULL;
				
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
		if (!gjs->next_pds)
			gjs->next_pds = BulkExecProcNode(outer_node);

		pds = gjs->next_pds;
		if (pds)
			gjs->next_pds = BulkExecProcNode(outer_node);
		else
		{
			gjs->next_pds = NULL;
			gjs->gts.scan_done = true;
		}
	}
	PERFMON_END(&gjs->gts.pfm_accum, time_outer_load, &tv2, &tv3);

	/*
	 * We also need to check existence of next inner hash-chunks, even if
	 * here is no more outer records, In case of multi-relations splited-out,
	 * we have to rewind the outer relation scan, then makes relations
	 * join with the next inner hash chunks.
	 */
    if (!pds)
        goto retry;

	return gpujoin_create_task(gjs, pds);
}

static TupleTableSlot *
gpujoin_next_tuple(GpuTaskState *gts)
{
	GpuJoinState	   *gjs = (GpuJoinState *) gts;
	TupleTableSlot	   *slot = gjs->gts.css.ss.ss_ScanTupleSlot;
	pgstrom_gpujoin	   *gjoin = (pgstrom_gpujoin *)gjs->gts.curr_task;
	pgstrom_data_store *pds_dst = gjoin->pds_dst;
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
		slot = NULL;	/* try next chunk */

	PERFMON_END(&gjs->gts.pfm_accum, time_materialize, &tv1, &tv2);
	return slot;
}

/* ----------------------------------------------------------------
 *
 * GpuTask handlers of GpuJoin
 *
 * ----------------------------------------------------------------
 */
static void
gpujoin_cleanup_cuda_resources(pgstrom_gpujoin *pgjoin)
{
	CUDA_EVENT_DESTROY(pgjoin, ev_dma_send_start);
	CUDA_EVENT_DESTROY(pgjoin, ev_dma_send_stop);
	CUDA_EVENT_DESTROY(pgjoin, ev_kern_join_end);
	CUDA_EVENT_DESTROY(pgjoin, ev_dma_recv_start);
	CUDA_EVENT_DESTROY(pgjoin, ev_dma_recv_stop);

	if (pgjoin->m_kgjoin)
		gpuMemFree(&pgjoin->task, pgjoin->m_kgjoin);
	if (pgjoin->m_kds_src)
		gpuMemFree(&pgjoin->task, pgjoin->m_kds_src);
	if (pgjoin->m_kds_dst)
		gpuMemFree(&pgjoin->task, pgjoin->m_kds_dst);
	if (pgjoin->m_kmrels)
		multirels_put_buffer(pgjoin->pmrels, &pgjoin->task);

	/* clear the pointers */
	pgjoin->kern_prep = NULL;
	pgjoin->kern_exec_nl = NULL;
	pgjoin->kern_exec_hj = NULL;
	pgjoin->kern_outer_nl = NULL;
	pgjoin->kern_outer_hj = NULL;
	pgjoin->kern_proj = NULL;
	pgjoin->m_kgjoin = 0UL;
	pgjoin->m_kds_src = 0UL;
	pgjoin->m_kds_dst = 0UL;
	pgjoin->m_kmrels = 0UL;
	pgjoin->ev_dma_send_start = NULL;
	pgjoin->ev_dma_send_stop = NULL;
	pgjoin->ev_kern_join_end = NULL;
	pgjoin->ev_dma_recv_start = NULL;
	pgjoin->ev_dma_recv_stop = NULL;
}

static void
gpujoin_task_release(GpuTask *gtask)
{
	pgstrom_gpujoin	   *pgjoin = (pgstrom_gpujoin *) gtask;

	/* release all the cuda resources, if any */
	gpujoin_cleanup_cuda_resources(pgjoin);
	/* detach multi-relations buffer, if any */
	if (pgjoin->pmrels)
		multirels_detach_buffer(pgjoin->pmrels);
	/* unlink source data store */
	if (pgjoin->pds_src)
		pgstrom_release_data_store(pgjoin->pds_src);
	/* unlink destination data store */
	if (pgjoin->pds_dst)
		pgstrom_release_data_store(pgjoin->pds_dst);
	/* release this gpu-task itself */
	pfree(pgjoin);
}

static bool
gpujoin_task_complete(GpuTask *gtask)
{
	pgstrom_gpujoin	   *gjoin = (pgstrom_gpujoin *) gtask;
	kern_gpujoin	   *kgjoin = &gjoin->kern;
	GpuTaskState	   *gts = gtask->gts;

	if (gts->pfm_accum.enabled)
	{
		CUDA_EVENT_ELAPSED(gjoin, time_dma_send,
						   ev_dma_send_start,
						   ev_dma_send_stop);
		CUDA_EVENT_ELAPSED(gjoin, time_kern_join,
						   ev_dma_send_stop,
						   ev_kern_join_end);
		CUDA_EVENT_ELAPSED(gjoin, time_kern_proj,
						   ev_kern_join_end,
						   ev_dma_recv_start);
		CUDA_EVENT_ELAPSED(gjoin, time_dma_recv,
						   ev_dma_recv_start,
						   ev_dma_recv_stop);
		pgstrom_accum_perfmon(&gts->pfm_accum, &gjoin->task.pfm);
	}
	gpujoin_cleanup_cuda_resources(gjoin);

	/*
	 * StromError_DataStoreNoSpace indicates pds_dst was smaller than
	 * what GpuHashJoin required. So, we expand the buffer and kick
	 * this gputask again.
	 */
	if (gjoin->task.errcode == StromError_DataStoreNoSpace)
	{
		GpuContext		   *gcontext = gts->gcontext;
		GpuJoinState	   *gjs = (GpuJoinState *) gts;
		pgstrom_data_store *pds_src = gjoin->pds_src;
		pgstrom_data_store *pds_dst = gjoin->pds_dst;
		kern_data_store	   *kds_old = pds_dst->kds;
		kern_data_store	   *kds_new;
		Size				kds_length;
		cl_uint				ncols = kds_old->ncols;
		cl_uint				nrooms;
		cl_uint				total_items;

		/* GpuJoin should not take file-mapped data store */
		Assert(!pds_dst->kds_fname);

		/*
		 * NOTE: StromError_DataStoreNoSpace may happen in two cases.
		 * First, kern_resultbuf, that stores intermediate results, does
		 * not have enough space, thus kernel code cannot generate join
		 * result. Second, kern_data_store of destination didn't have
		 * enough space, thus kernel projection got failed.
		 * The kern_gpujoin->kresults_max_items tell us which is the
		 * cause of this error.
		 */
		if (kgjoin->kresults_total_items < kgjoin->kresults_max_items)
		{
			/*
			 * In the first scenario, GPU projection didn't work at all.
			 * So, we expand the kern_resultbuf first, then expand the
			 * destination data-store according to the estimation number
			 * of result items.
			 */
			double	kresults_ratio_new;

			/* kresults_max_items tells how many items are needed */
			total_items = kgjoin->kresults_max_items *
				(1 + pgstrom_row_population_margin);
			nrooms = total_items / (gjs->num_rels + 1);
			/* adjust estimation */
			kresults_ratio_new = ((double)kgjoin->kresults_max_items /
								  (double)pds_src->kds->nitems);

			elog(NOTICE, "kresults was small, rate expanded %.2f => %.2f",
                 gjs->kresults_ratio, kresults_ratio_new);

			gjs->kresults_ratio = kresults_ratio_new;

			/* reset kern_gpujoin */
			kgjoin->kresults_2_offset = kgjoin->kresults_1_offset +
				STROMALIGN(offsetof(kern_resultbuf, results[total_items]));
			kgjoin->kresults_total_items = total_items;
			kgjoin->kresults_max_items = 0;
			kgjoin->errcode = StromError_Success;
		}
		else
		{
			/*
			 * In the second scenario, kds_old->nitems (that shall be
			 * larger than kds_old->nrooms) and kds_usage will tell us
			 * exact usage of the buffer
			 */
			if (kds_old->format == KDS_FORMAT_ROW)
			{
				Size	result_width;

				result_width = ((Size)(kds_old->usage -
									   KERN_DATA_STORE_HEAD_LENGTH(kds_old) -
									   sizeof(cl_uint) * kds_old->nitems) /
								(Size) kds_old->nitems) + 1;

				elog(NOTICE, "Destination KDS was small, "
					 "size expanded: width %u => %zu, nrooms %u => %u",
					 gjs->result_width, MAXALIGN(result_width),
					 kds_old->nrooms, kds_old->nitems);
			}
			else
			{
				elog(NOTICE, "Destination KDS was small, "
					 "size expanded: nrooms %u => %u",
					 kds_old->nrooms, kds_old->nitems);
			}
			nrooms = kds_old->nitems;
		}

		/*
		 * Expand kern_data_store according to the hint on previous
		 * execution.
		 */
		if (kds_old->format == KDS_FORMAT_SLOT)
		{
			kds_length = STROMALIGN(offsetof(kern_data_store,
											 colmeta[ncols])) +
				(LONGALIGN(sizeof(bool) * ncols) +
				 LONGALIGN(sizeof(Datum) * ncols)) * nrooms;
		}
		else if (kds_old->format == KDS_FORMAT_ROW)
		{
			kds_length = (STROMALIGN(offsetof(kern_data_store,
											  colmeta[ncols])) +
						  STROMALIGN(sizeof(cl_uint) * nrooms) +
						  gjs->result_width * nrooms);
		}
		else
			elog(ERROR, "Bug? unexpected result format: %d", kds_old->format);

		if (kds_length <= kds_old->length)
		{
			/* no need to alloc again, just reset usage */
			kds_old->usage = 0;
			kds_old->nitems = 0;
			kds_old->nrooms = nrooms;
		}
		else
		{
			kds_new = MemoryContextAllocZero(gcontext->memcxt, kds_length);
			memcpy(kds_new, kds_old, KERN_DATA_STORE_HEAD_LENGTH(kds_old));
			kds_new->hostptr = (hostptr_t) &kds_new->hostptr;
			kds_new->length = kds_length;
			kds_new->usage = 0;
			kds_new->nitems = 0;
			kds_new->nrooms = nrooms;
			pds_dst->kds = kds_new;
			pds_dst->kds_length = kds_new->length;
			pfree(kds_old);
		}

		/*
		 * OK, chain this task on the pending_tasks queue again
		 *
		 * NOTE: 'false' indicates cuda_control.c that this cb_complete
		 * callback handled this request by itself - we re-entered the
		 * GpuTask on the pending_task queue to execute again.
		 */
		SpinLockAcquire(&gts->lock);
		dlist_push_head(&gts->pending_tasks, &gjoin->task.chain);
		gts->num_pending_tasks++;
		SpinLockRelease(&gts->lock);

		return false;
	}
	return true;
}

static void
gpujoin_task_respond(CUstream stream, CUresult status, void *private)
{
	pgstrom_gpujoin	   *pgjoin = private;
	GpuTaskState	   *gts = pgjoin->task.gts;

	SpinLockAcquire(&gts->lock);
	if (status != CUDA_SUCCESS)
		pgjoin->task.errcode = status;
	else
		pgjoin->task.errcode = pgjoin->kern.errcode;

	/* remove from the running_tasks list */
	dlist_delete(&pgjoin->task.chain);
	gts->num_running_tasks--;

	/* then, attach it on the completed_tasks list */
	if (pgjoin->task.errcode == StromError_Success)
		dlist_push_tail(&gts->completed_tasks, &pgjoin->task.chain);
	else
		dlist_push_head(&gts->completed_tasks, &pgjoin->task.chain);
	gts->num_completed_tasks++;
	SpinLockRelease(&gts->lock);

	SetLatch(&MyProc->procLatch);
}

static bool
__gpujoin_task_process(pgstrom_gpujoin *pgjoin)
{
	pgstrom_data_store *pds_src = pgjoin->pds_src;
	pgstrom_data_store *pds_dst = pgjoin->pds_dst;
	GpuJoinState   *gjs = (GpuJoinState *) pgjoin->task.gts;
	const char	   *kern_proj_name;
	Size			length;
	size_t			total_items;
	size_t			outer_ntuples;
	size_t			grid_xsize;
	size_t			grid_ysize;
	size_t			block_xsize;
	size_t			block_ysize;
	CUresult		rc;
	ListCell	   *lc1;
	ListCell	   *lc2;
	ListCell	   *lc3;
	void		   *kern_args[10];
	int				depth;

	/*
	 * sanity checks
	 */
	Assert(pds_src->kds->format == KDS_FORMAT_ROW);
	Assert(pds_dst->kds->format == KDS_FORMAT_ROW ||
		   pds_dst->kds->format == KDS_FORMAT_SLOT);

	/*
	 * GPU kernel function lookup
	 */
	rc = cuModuleGetFunction(&pgjoin->kern_prep,
							 pgjoin->task.cuda_module,
							 "gpujoin_preparation");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	rc = cuModuleGetFunction(&pgjoin->kern_exec_nl,
							 pgjoin->task.cuda_module,
							 "gpujoin_exec_nestloop");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	rc = cuModuleGetFunction(&pgjoin->kern_exec_hj,
							 pgjoin->task.cuda_module,
							 "gpujoin_exec_hashjoin");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	rc = cuModuleGetFunction(&pgjoin->kern_outer_nl,
							 pgjoin->task.cuda_module,
							 "gpujoin_outer_nestloop");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	rc = cuModuleGetFunction(&pgjoin->kern_outer_hj,
							 pgjoin->task.cuda_module,
							 "gpujoin_outer_hashjoin");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	kern_proj_name = (pds_dst->kds->format == KDS_FORMAT_ROW
					  ? "gpujoin_projection_row"
					  : "gpujoin_projection_slot");
	rc = cuModuleGetFunction(&pgjoin->kern_proj,
							 pgjoin->task.cuda_module,
							 kern_proj_name);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	/*
	 * Allocation of device memory for each chunks
	 */

	/* kern_gpujoin *kgjoin (includes 2x kern_resultbuf) */
	total_items = (size_t)(gjs->kresults_ratio *
						   (double) pds_src->kds->nitems *
						   (1.0 + pgstrom_row_population_margin));
	length = STROMALIGN(offsetof(kern_resultbuf, results[total_items]));

	pgjoin->kern.kresults_1_offset =
		STROMALIGN(offsetof(kern_gpujoin, kparams) +
				   KERN_GPUJOIN_PARAMBUF_LENGTH(&pgjoin->kern));
	pgjoin->kern.kresults_2_offset =
		pgjoin->kern.kresults_1_offset + length;
	pgjoin->kern.kresults_total_items = total_items;
	pgjoin->kern.kresults_max_items = 0;
	pgjoin->kern.errcode = StromError_Success;

	length = pgjoin->kern.kresults_1_offset + 2 * length;
	pgjoin->m_kgjoin = gpuMemAlloc(&pgjoin->task, length);
	if (!pgjoin->m_kgjoin)
		goto out_of_resource;

	/* kern_data_store *kds_src */
	length = KERN_DATA_STORE_LENGTH(pds_src->kds);
	pgjoin->m_kds_src = gpuMemAlloc(&pgjoin->task, length);
	if (!pgjoin->m_kds_src)
		goto out_of_resource;

	/* kern_data_store *kds_dst */
	length = KERN_DATA_STORE_LENGTH(pds_dst->kds);
	pgjoin->m_kds_dst = gpuMemAlloc(&pgjoin->task, length);
	if (!pgjoin->m_kds_dst)
		goto out_of_resource;

	/*
	 * Creation of event objects, if needed
	 */
	if (pgjoin->task.pfm.enabled)
	{
		rc = cuEventCreate(&pgjoin->ev_dma_send_start, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

		rc = cuEventCreate(&pgjoin->ev_dma_send_stop, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

		rc = cuEventCreate(&pgjoin->ev_kern_join_end, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

		rc = cuEventCreate(&pgjoin->ev_dma_recv_start, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

		rc = cuEventCreate(&pgjoin->ev_dma_recv_stop, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));
    }

	/*
	 * OK, all the device memory and kernel objects are successfully
	 * constructed. Let's enqueue DMA send/recv and kernel invocations.
	 */
	CUDA_EVENT_RECORD(pgjoin, ev_dma_send_start);

	/* inner multi relations */
	multirels_send_buffer(pgjoin->pmrels, &pgjoin->task);
	/* kern_gpujoin */
	length = KERN_GPUJOIN_HEAD_LENGTH(&pgjoin->kern);
	rc = cuMemcpyHtoDAsync(pgjoin->m_kgjoin,
						   &pgjoin->kern,
						   length,
						   pgjoin->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	pgjoin->task.pfm.bytes_dma_send += length;
	pgjoin->task.pfm.num_dma_send++;

	/* kern_data_store (src) */
	length = KERN_DATA_STORE_LENGTH(pds_src->kds);
	rc = cuMemcpyHtoDAsync(pgjoin->m_kds_src,
						   pds_src->kds,
						   length,
						   pgjoin->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	pgjoin->task.pfm.bytes_dma_send += length;
	pgjoin->task.pfm.num_dma_send++;

	/* kern_data_store (dst of head) */
	length = KERN_DATA_STORE_HEAD_LENGTH(pds_dst->kds);
	rc = cuMemcpyHtoDAsync(pgjoin->m_kds_dst,
						   pds_dst->kds,
						   length,
						   pgjoin->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	pgjoin->task.pfm.bytes_dma_send += length;
	pgjoin->task.pfm.num_dma_send++;

	CUDA_EVENT_RECORD(pgjoin, ev_dma_send_stop);

	/*
	 * OK, enqueue a series of requests
	 */
	depth = 1;
	outer_ntuples = pds_src->kds->nitems;
	forthree (lc1, gjs->join_types,
			  lc2, gjs->hash_outer_keys,
			  lc3, gjs->nrows_ratio)
	{
		JoinType	join_type = (JoinType) lfirst_int(lc1);
		bool		is_nestloop = (lfirst(lc2) == NIL);
		double		nrows_ratio = (double)lfirst_int(lc3) / 1000000.0;
		size_t		num_threads;

		/*
		 * Launch:
		 * KERNEL_FUNCTION(void)
		 * gpujoin_preparation(kern_gpujoin *kgjoin,
		 *                     kern_data_store *kds,
		 *                     kern_multirels *kmrels,
		 *                     cl_int depth)
		 */
		num_threads = (depth > 1 ? 1 : pds_src->kds->nitems);
		pgstrom_compute_workgroup_size(&grid_xsize,
									   &block_xsize,
									   pgjoin->kern_prep,
									   pgjoin->task.cuda_device,
									   false,
									   num_threads,
									   sizeof(cl_uint));
		kern_args[0] = &pgjoin->m_kgjoin;
		kern_args[1] = &pgjoin->m_kds_src;
		kern_args[2] = &pgjoin->m_kmrels;
		kern_args[3] = &depth;

		rc = cuLaunchKernel(pgjoin->kern_prep,
							grid_xsize, 1, 1,
							block_xsize, 1, 1,
							sizeof(cl_uint) * block_xsize,
							pgjoin->task.cuda_stream,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
		pgjoin->task.pfm.num_kern_join++;

		/*
		 * Main logic of GpuHashJoin or GpuNestLoop
		 */
		if (is_nestloop)
		{
			size_t	shmem_size;
			size_t	inner_ntuples
				= multirels_get_nitems(pgjoin->pmrels, depth);

			/* NestLoop logic cannot run LEFT JOIN */
			Assert(join_type != JOIN_LEFT && join_type != JOIN_FULL);

			/*
			 * Launch:
			 * KERNEL_FUNCTION_MAXTHREADS(void)
			 * gpujoin_exec_nestloop(kern_gpujoin *kgjoin,
			 *                       kern_data_store *kds,
			 *                       kern_multirels *kmrels,
			 *                       cl_int depth,
			 *                       cl_uint cuda_index,
			 *                       cl_bool *outer_join_map)
			 */
			pgstrom_compute_workgroup_size_2d(&grid_xsize, &block_xsize,
											  &grid_ysize, &block_ysize,
											  pgjoin->kern_exec_nl,
                                              pgjoin->task.cuda_device,
											  outer_ntuples,
											  inner_ntuples,
											  2 * sizeof(Datum),
											  2 * sizeof(Datum),
											  sizeof(cl_uint));
			kern_args[0] = &pgjoin->m_kgjoin;
			kern_args[1] = &pgjoin->m_kds_src;
			kern_args[2] = &pgjoin->m_kmrels;
			kern_args[3] = &depth;
			kern_args[4] = &pgjoin->task.cuda_index;
			kern_args[5] = &pgjoin->m_ojmaps;

			shmem_size = Max(2 * sizeof(Datum) * Max(block_xsize,
													 block_ysize),
							 sizeof(cl_uint) * block_xsize * block_ysize);

			rc = cuLaunchKernel(pgjoin->kern_exec_nl,
								grid_xsize, grid_ysize, 1,
								block_xsize, block_ysize, 1,
								shmem_size,
								pgjoin->task.cuda_stream,
								kern_args,
								NULL);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
			pgjoin->task.pfm.num_kern_join++;

			/*
			 * Launch:
			 * KERNEL_FUNCTION(void)
			 * gpujoin_leftouter_nestloop(kern_gpujoin *kgjoin,
			 *                            kern_data_store *kds,
			 *                            kern_multirels *kmrels,
			 *                            cl_int depth,
			 *                            cl_uint cuda_index,
			 *                            cl_bool *outer_join_maps)
			 */
			if ((join_type == JOIN_RIGHT ||
				 join_type == JOIN_FULL) && pgjoin->is_last_chunk)
			{
				/* gather the outer join map, if multi-GPUs environment */
				multirels_gather_ojmaps(pgjoin->pmrels, &pgjoin->task, depth);

				pgstrom_compute_workgroup_size(&grid_xsize,
											   &block_xsize,
											   pgjoin->kern_outer_nl,
											   pgjoin->task.cuda_device,
											   false,
											   inner_ntuples,
											   sizeof(cl_uint));
				kern_args[0] = &pgjoin->m_kgjoin;
				kern_args[1] = &pgjoin->m_kds_src;
				kern_args[2] = &pgjoin->m_kmrels;
				kern_args[3] = &depth;
				kern_args[4] = &pgjoin->task.cuda_index;
				kern_args[5] = &pgjoin->m_ojmaps;

				rc = cuLaunchKernel(pgjoin->kern_outer_nl,
									grid_xsize, 1, 1,
									block_xsize, 1, 1,
									sizeof(cl_uint) * block_xsize,
									pgjoin->task.cuda_stream,
									kern_args,
									NULL);
				if (rc != CUDA_SUCCESS)
					elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
				pgjoin->task.pfm.num_kern_join++;
			}
		}
		else
		{
			size_t	inner_nslots
				= multirels_get_nslots(pgjoin->pmrels, depth);

			/*
			 * Launch:
			 * KERNEL_FUNCTION(void)
			 * gpujoin_exec_hashjoin(kern_gpujoin *kgjoin,
			 *                       kern_data_store *kds,
			 *                       kern_multirels *kmrels,
			 *                       cl_int depth,
			 *                       cl_uint cuda_index,
			 *                       cl_bool *outer_join_map)
			 */
			pgstrom_compute_workgroup_size(&grid_xsize,
										   &block_xsize,
										   pgjoin->kern_exec_hj,
										   pgjoin->task.cuda_device,
										   false,
										   outer_ntuples,
										   sizeof(cl_uint));
			kern_args[0] = &pgjoin->m_kgjoin;
			kern_args[1] = &pgjoin->m_kds_src;
			kern_args[2] = &pgjoin->m_kmrels;
			kern_args[3] = &depth;
			kern_args[4] = &pgjoin->task.cuda_index;
			kern_args[5] = &pgjoin->m_ojmaps;

			rc = cuLaunchKernel(pgjoin->kern_exec_hj,
								grid_xsize, 1, 1,
								block_xsize, 1, 1,
								sizeof(cl_uint) * block_xsize,
								pgjoin->task.cuda_stream,
								kern_args,
								NULL);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
			pgjoin->task.pfm.num_kern_join++;

			/*
			 * Launch:
			 * KERNEL_FUNCTION(void)
			 * gpujoin_leftouter_hashjoin(kern_gpujoin *kgjoin,
			 *                            kern_data_store *kds,
			 *                            kern_multirels *kmrels,
			 *                            cl_int depth,
			 *                            cl_uint cuda_index,
			 *                            cl_bool *outer_join_maps)
			 */
			if ((join_type == JOIN_RIGHT ||
				 join_type == JOIN_FULL) && pgjoin->is_last_chunk)
			{
				/* gather the outer join map, if multi-GPUs environment */
				multirels_gather_ojmaps(pgjoin->pmrels, &pgjoin->task, depth);

				pgstrom_compute_workgroup_size(&grid_xsize,
											   &block_xsize,
											   pgjoin->kern_outer_hj,
											   pgjoin->task.cuda_device,
											   false,
											   inner_nslots,
											   sizeof(cl_uint));
				kern_args[0] = &pgjoin->m_kgjoin;
				kern_args[1] = &pgjoin->m_kds_src;
				kern_args[2] = &pgjoin->m_kmrels;
				kern_args[3] = &depth;
				kern_args[4] = &pgjoin->task.cuda_index;
				kern_args[5] = &pgjoin->m_ojmaps;

				rc = cuLaunchKernel(pgjoin->kern_outer_hj,
									grid_xsize, 1, 1,
									block_xsize, 1, 1,
									sizeof(cl_uint) * block_xsize,
									pgjoin->task.cuda_stream,
									kern_args,
									NULL);
				if (rc != CUDA_SUCCESS)
					elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
				pgjoin->task.pfm.num_kern_join++;
			}
		}
		outer_ntuples = (size_t)((double)outer_ntuples * nrows_ratio);
		depth++;
	}
	Assert(pgjoin->kern.max_depth == depth - 1);
	CUDA_EVENT_RECORD(pgjoin, ev_kern_join_end);

	/*
	 * Launch:
	 * KERNEL_FUNCTION(void)
	 * gpujoin_projection_(row|slot)(kern_gpujoin *kgjoin,
	 *                               kern_multirels *kmrels,
	 *                               kern_data_store *kds_src,
	 *                               kern_data_store *kds_dst)
	 */
	pgstrom_compute_workgroup_size(&grid_xsize,
								   &block_xsize,
								   pgjoin->kern_proj,
								   pgjoin->task.cuda_device,
								   false,
								   outer_ntuples,
								   sizeof(cl_uint));
	kern_args[0] = &pgjoin->m_kgjoin;
	kern_args[1] = &pgjoin->m_kmrels;
	kern_args[2] = &pgjoin->m_kds_src;
	kern_args[3] = &pgjoin->m_kds_dst;

	rc = cuLaunchKernel(pgjoin->kern_proj,
						grid_xsize, 1, 1,
						block_xsize, 1, 1,
						sizeof(cl_uint) * block_xsize,
						pgjoin->task.cuda_stream,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
	pgjoin->task.pfm.num_kern_proj++;

	CUDA_EVENT_RECORD(pgjoin, ev_dma_recv_start);

	/* DMA Recv: kern_gpujoin *kgjoin */
	length = offsetof(kern_gpujoin, kparams);
	rc = cuMemcpyDtoHAsync(&pgjoin->kern,
						   pgjoin->m_kgjoin,
						   length,
						   pgjoin->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuMemcpyDtoHAsync: %s", errorText(rc));
	pgjoin->task.pfm.bytes_dma_recv += length;
	pgjoin->task.pfm.num_dma_recv++;

	/* DMA Recv: kern_data_store *kds_dst */
	length = KERN_DATA_STORE_LENGTH(pds_dst->kds);
	rc = cuMemcpyDtoHAsync(pds_dst->kds,
						   pgjoin->m_kds_dst,
						   length,
						   pgjoin->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuMemcpyDtoHAsync: %s", errorText(rc));
	pgjoin->task.pfm.bytes_dma_recv += length;
	pgjoin->task.pfm.num_dma_recv++;

	CUDA_EVENT_RECORD(pgjoin, ev_dma_recv_stop);

	/*
	 * Register the callback
	 */
	rc = cuStreamAddCallback(pgjoin->task.cuda_stream,
							 gpujoin_task_respond,
							 pgjoin, 0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuStreamAddCallback: %s", errorText(rc));

	return true;

out_of_resource:
	gpujoin_cleanup_cuda_resources(pgjoin);
	return false;
}

static bool
gpujoin_task_process(GpuTask *gtask)
{
	pgstrom_gpujoin *pgjoin = (pgstrom_gpujoin *) gtask;
	bool		status = false;
	CUresult	rc;

	/* switch CUDA context */
	rc = cuCtxPushCurrent(gtask->cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));
	PG_TRY();
	{
		if (multirels_get_buffer(pgjoin->pmrels, &pgjoin->task,
								 &pgjoin->m_kmrels,
								 &pgjoin->m_ojmaps))
			status = __gpujoin_task_process(pgjoin);
		else
			status = false;
	}
	PG_CATCH();
	{
		rc = cuCtxPopCurrent(NULL);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));
		gpujoin_cleanup_cuda_resources(pgjoin);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* reset CUDA context */
	rc = cuCtxPopCurrent(NULL);
	if (rc != CUDA_SUCCESS)
		elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));

	return status;
}

/*
 * pgstrom_init_gpujoin
 *
 * Entrypoint of GpuJoin
 */
void
pgstrom_init_gpujoin(void)
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
	gpujoin_plan_methods.CustomName				= "GpuJoin";
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
