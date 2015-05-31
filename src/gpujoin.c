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
#include "access/xact.h"
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
#include "utils/lsyscache.h"
#include "utils/pg_crc.h"
#include "utils/ruleutils.h"
#include <math.h>
#include "pg_strom.h"
#include "cuda_numeric.h"
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
	int				num_rels;
	Path		   *outer_path;
	Size			kmrels_length;	/* expected inner buffer size */
	double			kresults_ratio;	/* expected total-items ratio */
	List		   *host_quals;
	struct {
		double		nrows_ratio;	/* nrows ratio towards outer rows */
		JoinType	join_type;		/* one of JOIN_* */
		Path	   *scan_path;		/* outer scan path */
		List	   *hash_quals;		/* valid quals, if hash-join */
		List	   *join_quals;		/* all the device quals, incl hash_quals */
		double		kmrels_ratio;
		Size		chunk_size;		/* kmrels_length * kmrels_ratio */
		int			nbatches;		/* expected iteration in this depth */
		int			hash_nslots;	/* expected hashjoin slots width, if any */
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
	Size		kmrels_length;
	double		kresults_ratio;
	bool		outer_bulkload;
	double		bulkload_density;
	Expr	   *outer_quals;
	List	   *host_quals;
	/* for each depth */
	List	   *nrows_ratio;
	List	   *kmrels_ratio;
	List	   *join_types;
	List	   *join_quals;
	List	   *nbatches;
	List	   *hash_inner_keys;	/* if hash-join */
	List	   *hash_outer_keys;	/* if hash-join */
	List	   *hash_nslots;		/* if hash-join */
	/* supplemental information of ps_tlist */
	List	   *ps_src_depth;	/* source depth of the ps_tlist entry */
	List	   *ps_src_resno;	/* source resno of the ps_tlist entry */
} GpuJoinInfo;

static inline void
form_gpujoin_info(CustomScan *cscan, GpuJoinInfo *gj_info)
{
	List	   *privs = NIL;
	List	   *exprs = NIL;

	privs = lappend(privs, makeInteger(gj_info->num_rels));
	privs = lappend(privs, makeString(gj_info->kern_source));
	privs = lappend(privs, makeInteger(gj_info->extra_flags));
	exprs = lappend(exprs, gj_info->used_params);
	privs = lappend(privs, makeInteger(gj_info->kmrels_length));
	privs = lappend(privs,
					makeInteger(double_as_long(gj_info->kresults_ratio)));
	privs = lappend(privs, makeInteger(gj_info->outer_bulkload));
	privs = lappend(privs,
					makeInteger(double_as_long(gj_info->bulkload_density)));
	exprs = lappend(exprs, gj_info->outer_quals);
	exprs = lappend(exprs, gj_info->host_quals);
	/* for each depth */
	privs = lappend(privs, gj_info->nrows_ratio);
	privs = lappend(privs, gj_info->kmrels_ratio);
	privs = lappend(privs, gj_info->join_types);
	exprs = lappend(exprs, gj_info->join_quals);
	privs = lappend(privs, gj_info->nbatches);
	exprs = lappend(exprs, gj_info->hash_inner_keys);
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

	gj_info->num_rels = intVal(list_nth(privs, pindex++));
	gj_info->kern_source = strVal(list_nth(privs, pindex++));
	gj_info->extra_flags = intVal(list_nth(privs, pindex++));
	gj_info->used_params = list_nth(exprs, eindex++);
	gj_info->kmrels_length = intVal(list_nth(privs, pindex++));
	gj_info->kresults_ratio =
		long_as_double(intVal(list_nth(privs, pindex++)));
	gj_info->outer_bulkload = intVal(list_nth(privs, pindex++));
	gj_info->bulkload_density =
		long_as_double(intVal(list_nth(privs, pindex++)));
	gj_info->outer_quals = list_nth(exprs, eindex++);
	gj_info->host_quals = list_nth(exprs, eindex++);
	/* for each depth */
	gj_info->nrows_ratio = list_nth(privs, pindex++);
	gj_info->kmrels_ratio = list_nth(privs, pindex++);
	gj_info->join_types = list_nth(privs, pindex++);
    gj_info->join_quals = list_nth(exprs, eindex++);
	gj_info->nbatches = list_nth(privs, pindex++);
	gj_info->hash_inner_keys = list_nth(exprs, eindex++);
    gj_info->hash_outer_keys = list_nth(exprs, eindex++);
	gj_info->hash_nslots = list_nth(privs, pindex++);
	gj_info->ps_src_depth = list_nth(privs, pindex++);
	gj_info->ps_src_resno = list_nth(privs, pindex++);
	Assert(pindex == list_length(privs));
	Assert(eindex == list_length(exprs));

	return gj_info;
}

/*
 * GpuJoinState - execution state object of GpuJoin
 */
struct pgstrom_multirels;

typedef struct
{
	/*
	 * Execution status
	 */
	PlanState		   *state;
	ExprContext		   *econtext;
	pgstrom_data_store *curr_chunk;
	TupleTableSlot	   *scan_overflow;
	bool				scan_done;
	Size				ntuples;
	/* temp store, if KDS-hash overflow */
	Tuplestorestate	   *tupstore;

	/*
	 * Join properties; both nest-loop and hash-join
	 */
	int					depth;
	JoinType			join_type;
	int					nbatches_plan;
	int					nbatches_exec;
	double				nrows_ratio;
	double				kmrels_ratio;
	ExprState		   *join_quals;

	/*
	 * Join properties; only hash-join
	 */
	cl_uint				hash_nslots;
	cl_uint				hgram_shift;
	cl_uint				hgram_curr;
	Size			   *hgram_size;
	List			   *hash_outer_keys;
	List			   *hash_inner_keys;
	List			   *hash_keylen;
	List			   *hash_keybyval;
	List			   *hash_keytype;
} innerState;

typedef struct
{
	GpuTaskState	gts;
	/* expressions to be used in fallback path */
	List		   *join_types;
	ExprState	   *outer_quals;
	List		   *hash_outer_keys;
	List		   *join_quals;
	/* current window of inner relations */
	struct pgstrom_multirels *curr_pmrels;
	/* format of destination store */
	int				result_format;
	/* buffer population ratio */
	int				result_width;	/* result width for buffer length calc */
	Size			kmrels_length;	/* length of inner buffer */
	double			kresults_ratio;	/* estimated number of rows to outer */
	/* supplemental information to ps_tlist  */
	List		   *ps_src_depth;
	List		   *ps_src_resno;
	/* buffer for row materialization  */
	HeapTupleData	curr_tuple;
	/* buffer for PDS on bulk-loading mode */
	pgstrom_data_store *next_pds;

	/*
	 * Properties of underlying inner relations
	 *
	 */
	int				num_rels;
	innerState		inners[FLEXIBLE_ARRAY_MEMBER];
} GpuJoinState;

/*
 * pgstrom_multirels - inner buffer of multiple PDS/KDSs
 */
typedef struct pgstrom_multirels
{
	GpuJoinState   *gjs;		/* GpuJoinState of this buffer */
	Size			kmrels_length;	/* total length of the kern_multirels */
	Size			head_length;	/* length of the header portion */
	Size			usage_length;	/* length actually in use */
	Size			ojmap_length;	/* length of outer-join map */
	pgstrom_data_store **inner_chunks;	/* array of inner PDS */
	cl_int			n_attached;	/* Number of attached tasks */
	cl_int		   *refcnt;		/* Reference counter for each context */
	CUdeviceptr	   *m_kmrels;	/* GPU memory for each CUDA context */
	CUevent		   *ev_loaded;	/* Sync object for each CUDA context */
	CUdeviceptr	   *m_ojmaps;	/* GPU memory for outer join maps */
	kern_multirels	kern;
} pgstrom_multirels;

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
	pgstrom_multirels  *pmrels;		/* inner multi relations (heap or hash) */
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


static pgstrom_multirels *gpujoin_inner_preload(GpuJoinState *gjs);
static pgstrom_multirels *multirels_attach_buffer(pgstrom_multirels *pmrels);
static bool multirels_get_buffer(pgstrom_multirels *pmrels, GpuTask *gtask,
								 CUdeviceptr *p_kmrels,
								 CUdeviceptr *p_ojmaps);
static void multirels_put_buffer(pgstrom_multirels *pmrels, GpuTask *gtask);
static void multirels_send_buffer(pgstrom_multirels *pmrels, GpuTask *gtask);
static void multirels_colocate_outer_join_maps(pgstrom_multirels *pmrels,
											   GpuTask *gtask, int depth);
static void multirels_detach_buffer(pgstrom_multirels *pmrels);

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

static Path *
lookup_mergeable_gpujoin(PlannerInfo *root, RelOptInfo *outer_rel)
{
	CustomPath *cpath = pgstrom_find_path(root, outer_rel);

	if (cpath)
	{
		Assert(cpath->path.parent == outer_rel);
		if (path_is_mergeable_gpujoin(&cpath->path))
			return &cpath->path;
	}
	return NULL;
}

/*
 * returns true, if plannode is GpuJoin
 */
bool
pgstrom_plan_is_gpujoin(Plan *plannode)
{
	CustomScan *cscan = (CustomScan *) plannode;

	if (IsA(cscan, CustomScan) &&
		cscan->methods == &gpujoin_plan_methods)
		return true;
	return false;
}

/*
 * returns true, if plannode is GpuJoin and takes bulk-input
 */
bool
pgstrom_plan_is_gpujoin_bulkinput(Plan *plannode)
{
	if (pgstrom_plan_is_gpujoin(plannode))
	{
		GpuJoinInfo	   *gj_info = deform_gpujoin_info((CustomScan *) plannode);

		return gj_info->outer_bulkload;
	}
	return false;
}

/*
 * dump_gpujoin_path
 *
 * Dumps candidate GpuJoinPath for debugging
 */
static void
__dump_gpujoin_path(StringInfo buf, PlannerInfo *root,
					RelOptInfo *outer_rel, RelOptInfo *inner_rel,
					JoinType join_type, const char *join_label)
{
	Relids		outer_relids = outer_rel->relids;
	Relids		inner_relids = inner_rel->relids;
	List	   *range_tables = root->parse->rtable;
	int			rtindex;
	bool		is_first;

	/* outer relations */
	appendStringInfo(buf, "(");
	is_first = true;
	rtindex = -1;
	while ((rtindex = bms_next_member(outer_relids, rtindex)) >= 0)
	{
		RangeTblEntry  *rte = rt_fetch(rtindex, range_tables);
		Alias		   *eref = rte->eref;

		appendStringInfo(buf, "%s%s",
						 is_first ? "" : ", ",
						 eref->aliasname);
		is_first = false;
	}

	/* join logic */
	appendStringInfo(buf, ") %s%s (",
					 join_type == JOIN_FULL ? "F" :
					 join_type == JOIN_LEFT ? "L" :
					 join_type == JOIN_RIGHT ? "R" : "I",
					 join_label);

	/* inner relations */
	is_first = true;
	rtindex = -1;
	while ((rtindex = bms_next_member(inner_relids, rtindex)) >= 0)
	{
		RangeTblEntry  *rte = rt_fetch(rtindex, range_tables);
		Alias		   *eref = rte->eref;

		appendStringInfo(buf, "%s%s",
						 is_first ? "" : ", ",
						 eref->aliasname);
		is_first = false;
	}
	appendStringInfo(buf, ")");
}

/*
 * check_nrows_growth_ratio
 *
 * compute expected nrows growth ratio - too large destination buffer may
 * give up adoption of GpuJoin.
 */
static double
check_nrows_growth_ratio(PlannerInfo *root,
						 RelOptInfo *joinrel,
						 RelOptInfo *outer_rel,
						 RelOptInfo *inner_rel,
						 JoinType jointype,
						 SpecialJoinInfo *sjinfo,
						 List *join_quals,
						 List *host_quals)
{
	StringInfoData	buf;
	double			nrows;
	double			nrows_ratio;

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
								   outer_rel,
								   inner_rel,
								   sjinfo,
								   join_quals);
		nrows = dummy.rows;
	}
	nrows_ratio = nrows / outer_rel->rows;

	/*
	 * If expected results generated by GPU looks too large, we immediately
	 * give up to calculate this path.
	 */
	if (nrows_ratio > pgstrom_nrows_growth_ratio_limit)
	{
		if (client_min_messages <= DEBUG1)
		{
			initStringInfo(&buf);
			__dump_gpujoin_path(&buf, root, outer_rel,
								inner_rel, jointype, "J");
			elog(DEBUG1, "Nrows growth ratio %.2f on %s too large, give up",
				 nrows_ratio, buf.data);
			pfree(buf.data);
		}
		return -1.0;
	}
	else if (nrows_ratio > pgstrom_nrows_growth_ratio_limit / 2.0)
	{
		if (client_min_messages <= DEBUG1)
		{
			initStringInfo(&buf);
			__dump_gpujoin_path(&buf, root, outer_rel,
								inner_rel, jointype, "J");
			elog(DEBUG1,
				 "Nrows growth ratio %.2f on %s looks large, rounded to %.2f",
				 nrows_ratio, buf.data,
				 pgstrom_nrows_growth_ratio_limit / 2.0);
			pfree(buf.data);
		}
		nrows_ratio = pgstrom_nrows_growth_ratio_limit / 2.0;
	}
	return nrows_ratio;
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
	startup_cost = pgstrom_gpu_setup_cost;
	run_cost = outer_path->total_cost + cpu_tuple_cost * outer_path->rows;

	kmrels_length = STROMALIGN(offsetof(kern_multirels,
										chunks[num_rels]));
	largest_size = 0;
	largest_index = -1;
	outer_ntuples = outer_path->rows;
	for (i=0; i < num_rels; i++)
	{
		Path	   *inner_path = gpath->inners[i].scan_path;
		RelOptInfo *inner_rel = inner_path->parent;
		cl_uint		ncols = list_length(inner_rel->reltargetlist);
		Size		chunk_size;
		Size		entry_size;
		Size		htup_size;
		Size		hash_nslots = 0;

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

		/* chunk_size estimation */
		chunk_size = STROMALIGN(offsetof(kern_data_store,
										 colmeta[ncols]));
		if (gpath->inners[i].hash_quals != NIL)
		{
			/* KDS_FORMAT_HASH */
			/* hash slots */
			hash_nslots = Max((Size)inner_ntuples, 1024);
			hash_nslots = Min(hash_nslots,
							  gpuMemMaxAllocSize() / sizeof(void *));
			chunk_size += STROMALIGN(sizeof(cl_uint) * (Size)hash_nslots);
			/* kern_hashitem body */
			entry_size = offsetof(kern_hashitem, htup) + htup_size;
			chunk_size += STROMALIGN(entry_size * (Size)inner_ntuples);
		}
		else
		{
			/* KDS_FORMAT_ROW */
			/* row-index to kern_tupitem */
			chunk_size += STROMALIGN(sizeof(cl_uint) * (Size)inner_ntuples);
			/* kern_tupitem body */
			entry_size = offsetof(kern_tupitem, htup) + htup_size;
			chunk_size += STROMALIGN(entry_size * (Size)inner_ntuples);
		}
		gpath->inners[i].chunk_size = chunk_size;
		gpath->inners[i].hash_nslots = hash_nslots;

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
		startup_cost += (inner_path->total_cost +
						 cpu_tuple_cost * inner_path->rows);
		/* cost to compute hash value, if hash-join */
		if (gpath->inners[i].hash_quals != NIL)
			startup_cost += (cpu_operator_cost * inner_path->rows *
							 list_length(gpath->inners[i].hash_quals));
		/* cost to evaluate join qualifiers */
		if (gpath->inners[i].hash_quals != NIL)
            run_cost += (join_cost[i].per_tuple
						 * outer_ntuples
                         * (inner_ntuples /
							(double) gpath->inners[i].hash_nslots));
		else
			run_cost += (join_cost[i].per_tuple
                         * outer_ntuples
						 * clamp_row_est(inner_ntuples));
		/* iteration if nbatches > 1 */
		if (gpath->inners[i].nbatches > 1)
			run_cost *= (double) gpath->inners[i].nbatches;

		/* number of outer items on the next depth */
		outer_ntuples = gpath->inners[i].nrows_ratio * outer_path->rows;
	}
	/* put cost value on the gpath */
	gpath->cpath.path.startup_cost = startup_cost;
	gpath->cpath.path.total_cost = startup_cost + run_cost;

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
		gpath->inners[i].kmrels_ratio =
			((double)gpath->inners[i].chunk_size / (double)kmrels_length);
	}

	/* Dumps candidate GpuJoinPath for debugging */
	if (client_min_messages <= DEBUG1)
	{
		StringInfoData buf;
		int			num_rels = gpath->num_rels;
		RelOptInfo *outer_rel = gpath->outer_path->parent;
		RelOptInfo *inner_rel = gpath->inners[num_rels - 1].scan_path->parent;
		JoinType	join_type = gpath->inners[num_rels - 1].join_type;
		bool		is_nestloop
			= (gpath->inners[num_rels - 1].hash_quals == NIL);

		initStringInfo(&buf);
		__dump_gpujoin_path(&buf, root, outer_rel, inner_rel,
							join_type, is_nestloop ? "NL" : "HJ");

		elog(DEBUG1, "%s Cost=%.2f..%.2f",
			 buf.data,
			 gpath->cpath.path.startup_cost,
			 gpath->cpath.path.total_cost);
		pfree(buf.data);
	}
	return true;
}

static void
create_gpujoin_path(PlannerInfo *root,
					RelOptInfo *joinrel,
					JoinType jointype,
					Path *outer_path,
					Path *inner_path,
					SpecialJoinInfo *sjinfo,
					ParamPathInfo *param_info,
					Relids required_outer,
					List *hash_quals,
					List *join_quals,
					List *host_quals,
					bool support_bulkload,
					bool outer_merge,
					double nrows_ratio)
{
	GpuJoinPath	   *result;
	GpuJoinPath	   *source = NULL;
	Size			length;
	int				i, num_rels;

	if (!outer_merge)
		num_rels = 1;
	else
	{
		Assert(path_is_mergeable_gpujoin(outer_path));
		source = (GpuJoinPath *) outer_path;
		outer_path = source->outer_path;
		num_rels = source->num_rels + 1;
		/* source path also has to support bulkload */
		if ((source->cpath.flags & CUSTOMPATH_SUPPORT_BULKLOAD) == 0)
			support_bulkload = false;
	}

	length = offsetof(GpuJoinPath, inners[num_rels]);
	result = palloc0(length);
	NodeSetTag(result, T_CustomPath);
	result->cpath.path.pathtype = T_CustomScan;
	result->cpath.path.parent = joinrel;
	result->cpath.path.param_info = param_info;
	result->cpath.path.pathkeys = NIL;
	result->cpath.path.rows = joinrel->rows;
	result->cpath.flags = (support_bulkload ? CUSTOMPATH_SUPPORT_BULKLOAD : 0);
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
	result->inners[num_rels - 1].nrows_ratio = nrows_ratio;
	result->inners[num_rels - 1].scan_path = inner_path;
	result->inners[num_rels - 1].join_type = jointype;
	result->inners[num_rels - 1].hash_quals = hash_quals;
	result->inners[num_rels - 1].join_quals = join_quals;
	result->inners[num_rels - 1].kmrels_ratio = 0.0;	/* to be set later */
	result->inners[num_rels - 1].nbatches = 1;			/* to be set later */
	result->inners[num_rels - 1].hash_nslots = 0;		/* to be set later */

	/*
	 * cost calculation of GpuJoin, then, add this path to the joinrel,
	 * unless its cost is not obviously huge.
	 */
	if (cost_gpujoin(root, result, required_outer))
	{
		List   *custom_children = list_make1(result->outer_path);

		/* informs planner a list of child pathnodes */
		for (i=0; i < num_rels; i++)
			custom_children = lappend(custom_children,
									  result->inners[i].scan_path);
		result->cpath.custom_children = custom_children;
		/* add GpuJoin path */
		pgstrom_add_path(root, joinrel, &result->cpath, length);
	}
	else
		pfree(result);
}

static void
try_gpujoin_path(PlannerInfo *root,
				 RelOptInfo *joinrel,
				 JoinType jointype,
				 Path *outer_path,
                 Path *inner_path,
				 JoinPathExtraData *extra,
				 List *hash_quals,
				 List *join_quals,
				 List *host_quals,
				 double nrows_ratio)
{
	ParamPathInfo  *param_info;
	Relids			required_outer;
	List		   *restrictlist = extra->restrictlist;
	ListCell	   *lc;
	bool			support_bulkload = false;

	required_outer = calc_non_nestloop_required_outer(outer_path,
													  inner_path);
	if (required_outer &&
		!bms_overlap(required_outer, extra->param_source_rels))
	{
		bms_free(required_outer);
		return;
	}

	/*
	 * Independently of that, add parameterization needed for any
	 * PlaceHolderVars that need to be computed at the join.
	 */
	required_outer = bms_add_members(required_outer,
									 extra->extra_lateral_rels);

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
			support_bulkload = true;
	}

	/*
	 * ParamPathInfo of this join
	 */
	param_info = get_joinrel_parampathinfo(root,
										   joinrel,
										   outer_path,
										   inner_path,
										   extra->sjinfo,
										   required_outer,
										   &restrictlist);

	/*
	 * Try GpuHashJoin logic
	 */
	if (enable_gpuhashjoin && hash_quals != NIL)
	{
		create_gpujoin_path(root, joinrel, jointype,
							outer_path, inner_path,
							extra->sjinfo, param_info, required_outer,
							hash_quals, join_quals, host_quals,
							support_bulkload, false, nrows_ratio);

		if (path_is_mergeable_gpujoin(outer_path))
		{
			create_gpujoin_path(root, joinrel, jointype,
								outer_path, inner_path,
								extra->sjinfo, param_info, required_outer,
								hash_quals, join_quals, host_quals,
								support_bulkload, true, nrows_ratio);
		}
	}

	/*
	 * Try GpuNestLoop logic
	 */
	if (enable_gpunestloop &&
		(jointype == JOIN_INNER || jointype == JOIN_LEFT))
	{
	    create_gpujoin_path(root, joinrel, jointype,
							outer_path, inner_path,
							extra->sjinfo, param_info, required_outer,
							NIL, join_quals, host_quals,
							support_bulkload, false, nrows_ratio);

		if (path_is_mergeable_gpujoin(outer_path))
		{
			create_gpujoin_path(root, joinrel, jointype,
								outer_path, inner_path,
								extra->sjinfo, param_info, required_outer,
								NIL, join_quals, host_quals,
								support_bulkload, true, nrows_ratio);
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
					  JoinType jointype,
					  JoinPathExtraData *extra)
{
	Path	   *cheapest_total_inner = innerrel->cheapest_total_path;
	Path	   *cheapest_total_outer = outerrel->cheapest_total_path;
	Path	   *mergeable_gpujoin_outer;
	List	   *host_quals = NIL;
	List	   *hash_quals = NIL;
	List	   *join_quals = NIL;
	ListCell   *lc;
	double		nrows_ratio;

	/* calls secondary module if exists */
	if (set_join_pathlist_next)
		set_join_pathlist_next(root,
							   joinrel,
							   outerrel,
							   innerrel,
							   jointype,
							   extra);

	/* nothing to do, if PG-Strom is not enabled */
	if (!pgstrom_enabled)
		return;

	/* quick exit, if unsupported join type */
	if (jointype != JOIN_INNER && jointype != JOIN_FULL &&
		jointype != JOIN_RIGHT && jointype != JOIN_LEFT)
		return;

	/*
	 * Check restrictions of joinrel.
	 */
	foreach (lc, extra->restrictlist)
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

	/*
	 * Check nrows growth ratio. If too large PDS buffer is required,
	 * we will give up GpuJoin at all.
	 */
	nrows_ratio = check_nrows_growth_ratio(root, joinrel, outerrel, innerrel,
										   jointype, extra->sjinfo,
										   join_quals, host_quals);
	if (nrows_ratio < 0.0)
		return;

	/*
	 * Find out an inner path with cheapest total cost, but not parameterized
	 * by outer relation.
	 */
	if (PATH_PARAM_BY_REL(cheapest_total_inner, outerrel))
	{
		cheapest_total_inner = NULL;
		foreach (lc, innerrel->pathlist)
		{
			Path   *curr_path = lfirst(lc);

			if (!cheapest_total_inner ||
				cheapest_total_inner->total_cost > curr_path->total_cost)
				cheapest_total_inner = curr_path;
		}
		if (!cheapest_total_inner)
			return;
	}

	/*
	 * Find out an outer path with cheapest startup / total cost, but not
	 * parameterized by inner relation.
	 */
	if (PATH_PARAM_BY_REL(cheapest_total_outer, innerrel))
	{
		cheapest_total_outer = NULL;
		foreach (lc, outerrel->pathlist)
		{
			Path   *curr_path = lfirst(lc);

			if (!cheapest_total_outer ||
				cheapest_total_outer->total_cost > curr_path->total_cost)
				cheapest_total_outer = curr_path;
		}
		if (!cheapest_total_outer)
			return;
	}

	/*
	 * Find out mergeable outer GpuJoin Path, if any
	 */
	mergeable_gpujoin_outer = lookup_mergeable_gpujoin(root, outerrel);

	/*
	 * Try, cheapest_total_inner + cheapest_total_outer
	 */
	try_gpujoin_path(root,
					 joinrel,
					 jointype,
					 cheapest_total_outer,
					 cheapest_total_inner,
					 extra,
					 hash_quals,
					 join_quals,
					 host_quals,
					 nrows_ratio);

	/*
	 * Try, cheapest_total_inner + mergeable_gpujoin_outer
	 */
	if (mergeable_gpujoin_outer != NULL &&
		mergeable_gpujoin_outer != cheapest_total_outer)
	{
		try_gpujoin_path(root,
						 joinrel,
						 jointype,
						 mergeable_gpujoin_outer,
						 cheapest_total_inner,
						 extra,
						 hash_quals,
						 join_quals,
						 host_quals,
						 nrows_ratio);
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
		if (bms_is_member(varnode->varno, rel->relids))
			ps_depth = 0;
		else
		{
			int		i;

			for (i=0; i < gpath->num_rels; i++)
			{
				rel = gpath->inners[i].scan_path->parent;
				if (bms_is_member(varnode->varno, rel->relids))
					break;
			}
			if (i == gpath->num_rels)
				elog(ERROR, "Bug? uncertain origin of Var-node: %s",
					 nodeToString(varnode));
			ps_depth = i + 1;
		}

		foreach (cell, rel->reltargetlist)
		{
			Node	   *expr = lfirst(cell);

			if (equal(varnode, expr))
			{
				TargetEntry	   *ps_tle
					= makeTargetEntry((Expr *) copyObject(varnode),
									  list_length(context->ps_tlist) + 1,
									  NULL,
									  context->resjunk);
				context->ps_tlist =
					lappend(context->ps_tlist, ps_tle);
				context->ps_depth =
					lappend_int(context->ps_depth, ps_depth);
				context->ps_resno =
					lappend_int(context->ps_resno, varnode->varattno);
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
					List *clauses,
					List *custom_children)
{
	GpuJoinPath	   *gpath = (GpuJoinPath *) best_path;
	GpuJoinInfo		gj_info;
	CustomScan	   *cscan;
	codegen_context	context;
	Plan		   *outer_plan;
	ListCell	   *lc;
	int				i;

	Assert(gpath->num_rels + 1 == list_length(custom_children));
	outer_plan = linitial(custom_children);

	cscan = makeNode(CustomScan);
	cscan->scan.plan.targetlist = tlist;
	cscan->scan.plan.qual = gpath->host_quals;
	cscan->flags = best_path->flags;
	cscan->methods = &gpujoin_plan_methods;
	cscan->custom_children = list_copy_tail(custom_children, 1);

	memset(&gj_info, 0, sizeof(GpuJoinInfo));
	gj_info.kmrels_length = gpath->kmrels_length;
	gj_info.kresults_ratio = gpath->kresults_ratio;
	gj_info.host_quals = extract_actual_clauses(gpath->host_quals, false);
	gj_info.num_rels = gpath->num_rels;

	for (i=0; i < gpath->num_rels; i++)
	{
		List	   *hash_inner_keys = NIL;
		List	   *hash_outer_keys = NIL;
		List	   *clauses;

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

		/*
		 * Add properties of GpuJoinInfo
		 */
		gj_info.nrows_ratio = lappend_int(gj_info.nrows_ratio,
								float_as_int(gpath->inners[i].nrows_ratio));
		gj_info.kmrels_ratio = lappend_int(gj_info.kmrels_ratio,
								float_as_int(gpath->inners[i].kmrels_ratio));
		gj_info.join_types = lappend_int(gj_info.join_types,
										 gpath->inners[i].join_type);
		clauses = extract_actual_clauses(gpath->inners[i].join_quals, false);
		gj_info.join_quals = lappend(gj_info.join_quals,
									 build_flatten_qualifier(clauses));
		gj_info.nbatches = lappend_int(gj_info.nbatches,
									   gpath->inners[i].nbatches);
		gj_info.hash_inner_keys = lappend(gj_info.hash_inner_keys,
										  hash_inner_keys);
		gj_info.hash_outer_keys = lappend(gj_info.hash_outer_keys,
										  hash_outer_keys);
		gj_info.hash_nslots = lappend_int(gj_info.hash_nslots,
										  gpath->inners[i].hash_nslots);
	}

	/*
	 * Creation of the underlying outer Plan node. In case of SeqScan,
	 * it may make sense to replace it with GpuScan for bulk-loading.
	 */
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
		double	outer_density = pgstrom_get_bulkload_density(outer_plan);

		if ((custom_flags & CUSTOMPATH_SUPPORT_BULKLOAD) != 0 &&
			outer_density >= (1.0 - pgstrom_bulkload_density) &&
			outer_density <= (1.0 + pgstrom_bulkload_density))
		{
			gj_info.outer_bulkload = true;
			gj_info.bulkload_density = outer_density;
		}
	}
	outerPlan(cscan) = outer_plan;

	/*
	 * Build a pseudo-scan targetlist
	 */
	cscan->custom_scan_tlist = build_pseudo_targetlist(gpath, &gj_info, tlist);

	/*
	 * construct kernel code
	 */
	pgstrom_init_codegen_context(&context);
	context.pseudo_tlist = cscan->custom_scan_tlist;

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
		/* join_type */
		appendStringInfo(str, " :join_type %d",
						 (int)gpath->inners[i].join_type);
		/* scan_path */
		appendStringInfo(str, " :scan_path %s",
						 nodeToString(gpath->inners[i].scan_path));
		/* hash_quals */
		appendStringInfo(str, " :hash_quals %s",
						 nodeToString(gpath->inners[i].hash_quals));
		/* join_quals */
		appendStringInfo(str, " :join_clause %s",
						 nodeToString(gpath->inners[i].join_quals));
		/* nrows_ratio */
		appendStringInfo(str, " :nrows_ratio %.2f",
						 gpath->inners[i].nrows_ratio);
		/* kmrels_ratio */
		appendStringInfo(str, " :kmrels_ratio %.2f",
						 gpath->inners[i].kmrels_ratio);
		/* chunk_size */
		appendStringInfo(str, " :chunk_size %zu",
						 gpath->inners[i].chunk_size);
		/* nbatches */
		appendStringInfo(str, " :nbatches %d",
						 gpath->inners[i].nbatches);
		/* nslots */
		appendStringInfo(str, " :nslots %d",
						 gpath->inners[i].hash_nslots);
		appendStringInfo(str, "}");
	}
	appendStringInfo(str, ")");
}



typedef struct
{
	int		depth;
	List   *ps_src_depth;
	List   *ps_src_resno;
} fixup_varnode_to_origin_context;

static Node *
fixup_varnode_to_origin_mutator(Node *node,
								fixup_varnode_to_origin_context *context)
{
	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		Var	   *varnode = (Var *) node;
		int		varattno = varnode->varattno;
		int		src_depth;

		Assert(varnode->varno == INDEX_VAR);
		src_depth = list_nth_int(context->ps_src_depth,
								 varnode->varattno - 1);
		if (src_depth == context->depth)
		{
			Var	   *newnode = copyObject(varnode);

			newnode->varno = INNER_VAR;
			newnode->varattno = list_nth_int(context->ps_src_resno,
											 varattno - 1);
			return (Node *) newnode;
		}
		else if (src_depth > context->depth)
			elog(ERROR, "Expression reference deeper than current depth");
	}
	return expression_tree_mutator(node, fixup_varnode_to_origin_mutator,
								   (void *) context);
}

static List *
fixup_varnode_to_origin(GpuJoinState *gjs, int depth, List *expr_list)
{
	fixup_varnode_to_origin_context	context;

	Assert(IsA(expr_list, List));
	context.depth = depth;
	context.ps_src_depth = gjs->ps_src_depth;
	context.ps_src_resno = gjs->ps_src_resno;

	return (List *) fixup_varnode_to_origin_mutator((Node *) expr_list,
													&context);
}



static Node *
gpujoin_create_scan_state(CustomScan *node)
{
	GpuJoinState   *gjs;
	GpuJoinInfo	   *gj_info = deform_gpujoin_info(node);

	Assert(gj_info->num_rels == list_length(node->custom_children));
	gjs = palloc0(offsetof(GpuJoinState, inners[gj_info->num_rels]));

	/* Set tag and executor callbacks */
	NodeSetTag(gjs, T_CustomScanState);
	gjs->gts.css.flags = node->flags;
	gjs->gts.css.methods = &gpujoin_exec_methods.c;

	return (Node *) gjs;
}

static void
gpujoin_begin(CustomScanState *node, EState *estate, int eflags)
{
	GpuContext	   *gcontext = NULL;
	GpuJoinState   *gjs = (GpuJoinState *) node;
	PlanState	   *ps = &gjs->gts.css.ss.ps;
	CustomScan	   *cscan = (CustomScan *) node->ss.ps.plan;
	GpuJoinInfo	   *gj_info = deform_gpujoin_info(cscan);
	TupleDesc		tupdesc = GTS_GET_RESULT_TUPDESC(gjs);
	int				i;

	/* activate GpuContext for device execution */
	if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
		gcontext = pgstrom_get_gpucontext();

	/* Setup common GpuTaskState fields */
	pgstrom_init_gputaskstate(gcontext, &gjs->gts);
	gjs->gts.cb_task_process = gpujoin_task_process;
	gjs->gts.cb_task_complete = gpujoin_task_complete;
	gjs->gts.cb_task_release = gpujoin_task_release;
	gjs->gts.cb_next_chunk = gpujoin_next_chunk;
	gjs->gts.cb_next_tuple = gpujoin_next_tuple;

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
	gjs->num_rels = gj_info->num_rels;
	gjs->join_types = gj_info->join_types;
	gjs->outer_quals = ExecInitExpr((Expr *)gj_info->outer_quals, ps);
	gjs->gts.css.ss.ps.qual = (List *)
		ExecInitExpr((Expr *)gj_info->host_quals, ps);

	/* needs to track corresponding columns */
	gjs->ps_src_depth = gj_info->ps_src_depth;
	gjs->ps_src_resno = gj_info->ps_src_resno;

	/*
	 * initialization of child nodes
	 */
	outerPlanState(gjs) = ExecInitNode(outerPlan(cscan), estate, eflags);
	for (i=0; i < gj_info->num_rels; i++)
	{
		Plan	   *inner_plan = list_nth(cscan->custom_children, i);
		innerState *istate = &gjs->inners[i];
		List	   *hash_inner_keys;
		List	   *hash_outer_keys;
		ListCell   *lc;

		istate->state = ExecInitNode(inner_plan, estate, eflags);
		istate->econtext = CreateExprContext(estate);
		istate->depth = i + 1;
		istate->nbatches_plan = list_nth_int(gj_info->nbatches, i);
		istate->nbatches_exec =
			((eflags & EXEC_FLAG_EXPLAIN_ONLY) != 0 ? -1 : 0);
		istate->nrows_ratio =
			int_as_float(list_nth_int(gj_info->nrows_ratio, i));
		istate->kmrels_ratio =
			int_as_float(list_nth_int(gj_info->kmrels_ratio, i));
		istate->join_type = (JoinType)list_nth_int(gj_info->join_types, i);

		/*
		 * NOTE: We need to deal with Var-node references carefully,
		 * because varno/varattno pair depends on the context when
		 * ExecQual() is called.
		 * - join_quals and hash_outer_keys are only called for
		 * fallback process when CpuReCheck error was returned.
		 * So, we can expect values are stored in ecxt_scantuple
		 * according to the pseudo-scan-tlist.
		 *- hash_inner_keys are only called to construct hash-table
		 * prior to GPU execution, so, we can expect input values
		 * are deployed according to the result of child plans.
		 */
		istate->join_quals =
			ExecInitExpr(list_nth(gj_info->join_quals, i), ps);

		hash_inner_keys = list_nth(gj_info->hash_inner_keys, i);
		if (hash_inner_keys != NIL)
		{
			cl_uint		shift;

			hash_inner_keys = fixup_varnode_to_origin(gjs, i+1,
													  hash_inner_keys);
			foreach (lc, hash_inner_keys)
			{
				Expr	   *expr = lfirst(lc);
				ExprState  *expr_state = ExecInitExpr(expr, ps);
				Oid			type_oid = exprType((Node *)expr);
				int16		typlen;
				bool		typbyval;

				istate->hash_inner_keys =
					lappend(istate->hash_inner_keys, expr_state);

				get_typlenbyval(type_oid, &typlen, &typbyval);
				istate->hash_keytype =
					lappend_oid(istate->hash_keytype, type_oid);
				istate->hash_keylen =
					lappend_int(istate->hash_keylen, typlen);
				istate->hash_keybyval =
					lappend_int(istate->hash_keybyval, typbyval);
			}
			/* outer keys also */
			hash_outer_keys = list_nth(gj_info->hash_outer_keys, i);
			Assert(hash_outer_keys != NIL);
			istate->hash_outer_keys = (List *)
				ExecInitExpr((Expr *)hash_outer_keys, ps);

			Assert(IsA(istate->hash_outer_keys, List) &&
				   list_length(istate->hash_inner_keys) ==
				   list_length(istate->hash_outer_keys));

			/* hash slot width */
			istate->hash_nslots = list_nth_int(gj_info->hash_nslots, i);

			/* usage histgram */
			shift = get_next_log2(gjs->inners[i].nbatches_plan) + 4;
			Assert(shift < sizeof(cl_uint) * BITS_PER_BYTE);
			istate->hgram_size = palloc0(sizeof(Size) * (1U << shift));
			istate->hgram_shift = sizeof(cl_uint) * BITS_PER_BYTE - shift;
			istate->hgram_curr = 0;
		}
		gjs->gts.css.custom_children =
			lappend(gjs->gts.css.custom_children, gjs->inners[i].state);
	}

	/*
	 * Is bulkload available?
	 */
	gjs->gts.scan_bulk =
		(!pgstrom_bulkload_enabled ? false : gj_info->outer_bulkload);
	gjs->gts.scan_bulk_density = gj_info->bulkload_density;

	/*
	 * initialize kernel execution parameter
	 */
	pgstrom_assign_cuda_program(&gjs->gts,
								gj_info->used_params,
								gj_info->kern_source,
								gj_info->extra_flags);
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
	gjs->kmrels_length = gj_info->kmrels_length;
	gjs->kresults_ratio = gj_info->kresults_ratio;
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

retry:
	/* fetch next chunk to be processed */
	pgjoin = (pgstrom_gpujoin *) pgstrom_fetch_gputask(&gjs->gts);
	if (!pgjoin)
		return NULL;

	pds_dst = pgjoin->pds_dst;
	/* retry, if no valid rows are contained */
	if (pds_dst->kds->nitems == 0)
	{
		pgstrom_release_gputask(&pgjoin->task);
		goto retry;
	}
	/* release this pgstrom_gpujoin, except for pds_dst */
	pgjoin->pds_dst = NULL;
	pgstrom_release_gputask(&pgjoin->task);

	return pds_dst;
}

static void
gpujoin_end(CustomScanState *node)
{
	GpuJoinState   *gjs = (GpuJoinState *) node;
	int				i;

	/*
	 * clean up subtree
	 */
	ExecEndNode(outerPlanState(node));
	for (i=0; i < gjs->num_rels; i++)
		ExecEndNode(gjs->inners[i].state);

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
		foreach (lc1, cscan->custom_scan_tlist)
		{
			TargetEntry	   *tle = lfirst(lc1);

			temp = deparse_expression((Node *)tle->expr,
									  context, true, false);
			if (lc1 != list_head(cscan->custom_scan_tlist))
				appendStringInfo(&str, ", ");
			if (!tle->resjunk)
				appendStringInfo(&str, "%s", temp);
			else
				appendStringInfo(&str, "(%s)", temp);
		}
		ExplainPropertyText("Pseudo Scan", str.data, es);
	}

	/* outer bulkload */
	if (!gjs->gts.scan_bulk)
		ExplainPropertyText("Bulkload", "Off", es);
	else
	{
		char   *temp = psprintf("On (density: %.2f%%)",
								100.0 * gjs->gts.scan_bulk_density);
		ExplainPropertyText("Bulkload", temp, es);
	}

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
		"  kern_data_store *kds_in;\n"
		"  kern_colmeta *colmeta;\n"
		"  void *datum;\n");

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
			else
			{
				/* in case of inner data store */
				appendStringInfo(
					source,
					"  /* variable load in depth-%u (data store) */\n"
					"  kds_in = KERN_MULTIRELS_INNER_KDS(kmrels, %u);\n"
					"  assert(kds_in->format == %s);\n"
					"  colmeta = kds_in->colmeta;\n",
					keynode->varno,
					keynode->varno,
					list_nth(gj_info->hash_outer_keys,
							 keynode->varno - 1) == NIL
					? "KDS_FORMAT_ROW"
					: "KDS_FORMAT_HASH");

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
				   STROMALIGN(gjs->gts.kern_params->length));
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
							(1.0 + pgstrom_nrows_growth_margin));
	kgjoin = &pgjoin->kern;
	kgjoin->kresults_1_offset = kgjoin_head;
	kgjoin->kresults_2_offset = kgjoin_head +
		STROMALIGN(offsetof(kern_resultbuf, results[total_items]));
	kgjoin->kresults_total_items = total_items;
	kgjoin->kresults_max_items = 0;
	kgjoin->max_depth = gjs->num_rels;
	kgjoin->errcode = StromError_Success;

	memcpy(KERN_GPUJOIN_PARAMBUF(kgjoin),
		   gjs->gts.kern_params,
		   gjs->gts.kern_params->length);

	/*
	 * Allocation of the destination data-store
	 */
	nrooms = (cl_uint)((double) pds_src->kds->nitems *
					   gjs->inners[gjs->num_rels - 1].nrows_ratio *
					   (1.0 + pgstrom_nrows_growth_margin));
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
	struct timeval	tv1, tv2;

	/*
     * Logic to fetch inner multi-relations looks like nested-loop.
     * If all the underlying inner scan already scaned its outer
	 * relation, current depth makes advance its scan pointer with
	 * reset of underlying scan pointer, or returns NULL if it is
	 * already reached end of scan.
     */
retry:
	if (gjs->gts.scan_done || !gjs->curr_pmrels)
	{
		pgstrom_multirels *pmrels;

		/* unlink previous inner multi-relations */
		if (gjs->curr_pmrels)
		{
			Assert(gjs->gts.scan_done);
			multirels_detach_buffer(gjs->curr_pmrels);
			gjs->curr_pmrels = NULL;
		}

		/* load an inner multi-relations buffer */
		pmrels = gpujoin_inner_preload(gjs);
		if (!pmrels)
			return NULL;	/* end of inner multi-relations */
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

	PERFMON_BEGIN(&gts->pfm_accum, &tv1);
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
	PERFMON_END(&gjs->gts.pfm_accum, time_outer_load, &tv1, &tv2);

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
				(1 + pgstrom_nrows_growth_margin);
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

	/* See comments in pgstrom_respond_gpuscan() */
	if (status == CUDA_ERROR_INVALID_CONTEXT || !IsTransactionState())
		return;

	if (status != CUDA_SUCCESS)
		pgjoin->task.errcode = status;
	else
		pgjoin->task.errcode = pgjoin->kern.errcode;

	/*
	 * Remove from the running_tasks list, then attach it
	 * on the completed_tasks list
	 */
	SpinLockAcquire(&gts->lock);
	dlist_delete(&pgjoin->task.chain);
	gts->num_running_tasks--;

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
						   (1.0 + pgstrom_nrows_growth_margin));
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
	for (depth = 1; depth <= gjs->num_rels; depth++)
	{
		JoinType	join_type = gjs->inners[depth - 1].join_type;
		bool		is_nestloop = (!gjs->inners[depth - 1].hash_outer_keys);
		double		nrows_ratio = gjs->inners[depth - 1].nrows_ratio;
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
			pgstrom_data_store *pds = pgjoin->pmrels->inner_chunks[depth - 1];
			size_t	inner_ntuples = pds->kds->nitems;
			size_t	shmem_size;

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
				multirels_colocate_outer_join_maps(pgjoin->pmrels,
												   &pgjoin->task, depth);
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
			pgstrom_data_store *pds = pgjoin->pmrels->inner_chunks[depth - 1];
			size_t	inner_nslots = pds->kds->nslots;

			Assert(inner_nslots > 0);
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
				multirels_colocate_outer_join_maps(pgjoin->pmrels,
												   &pgjoin->task, depth);
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
		outer_ntuples = (size_t)((double)outer_ntuples *
								 Max(nrows_ratio, 1.0));
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

/* ================================================================
 *
 * Routines to preload inner relations (heap/hash)
 *
 * ================================================================
 */

static bool
multirels_expand_length(GpuJoinState *gjs, pgstrom_multirels *pmrels)
{
	Size		new_length = 2 * pmrels->kmrels_length;

	/* No more physical space to expand, we will give up */
	if (new_length > gpuMemMaxAllocSize())
		return false;

	pmrels->kmrels_length = new_length;
	gjs->kmrels_length = new_length;

	return true;
}

/*****/
static pg_crc32
get_tuple_hashvalue(innerState *istate, TupleTableSlot *slot)
{
	ExprContext	   *econtext = istate->econtext;
	pg_crc32		hash;
	ListCell	   *lc1;
	ListCell	   *lc2;
	ListCell	   *lc3;
	ListCell	   *lc4;

	/* calculation of a hash value of this entry */
	econtext->ecxt_innertuple = slot;
	INIT_LEGACY_CRC32(hash);
	forfour (lc1, istate->hash_inner_keys,
			 lc2, istate->hash_keylen,
			 lc3, istate->hash_keybyval,
			 lc4, istate->hash_keytype)
	{
		ExprState  *clause = lfirst(lc1);
		int			keylen = lfirst_int(lc2);
		bool		keybyval = lfirst_int(lc3);
		Oid			keytype = lfirst_oid(lc4);
		int			errcode;
		Datum		value;
		bool		isnull;

		value = ExecEvalExpr(clause, istate->econtext, &isnull, NULL);
		if (isnull)
			continue;

		/* fixup host representation to special internal format. */
		if (keytype == NUMERICOID)
		{
			pg_numeric_t	temp;

			temp = pg_numeric_from_varlena(&errcode, (struct varlena *)
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

	return hash;
}

/*
 * gpujoin_inner_hash_preload_partial
 *
 * It preloads a part of inner relation, within a particular range of
 * hash-values, to the data store with hash-format, for hash-join
 * execution. Its source is preliminary materialized within tuple-store
 * of PostgreSQL.
 */
static bool
gpujoin_inner_hash_preload_partial(innerState *istate,
								   pgstrom_data_store *pds_hash)
{
	kern_data_store	*kds_hash = pds_hash->kds;
	TupleTableSlot	*tupslot = istate->state->ps_ResultTupleSlot;
	Tuplestorestate *tupstore = istate->tupstore;
	cl_uint	   *hash_slots;
	cl_uint		limit;
	Size		required = 0;

	/* tuplestore must be already built */
	Assert(tupslot != NULL);

	/* reset hash-table usage */
	Assert(kds_hash->hostptr == (uintptr_t)&kds_hash->hostptr);
	Assert(kds_hash->format == KDS_FORMAT_HASH);
	kds_hash->usage = KERN_DATA_STORE_HEAD_LENGTH(kds_hash) +
		STROMALIGN(sizeof(cl_uint) * kds_hash->nslots);
	kds_hash->nitems = 0;
	hash_slots = KERN_DATA_STORE_HASHSLOT(kds_hash);
	memset(hash_slots, 0, STROMALIGN(sizeof(cl_uint) * kds_hash->nslots));

	/*
	 * Find a suitable range of hash_min/hash_max
	 */
	limit = (1U << (sizeof(cl_uint) * BITS_PER_BYTE - istate->hgram_shift));
	if (istate->hgram_curr >= limit)
		return false;	/* no more records to read */
	kds_hash->hash_min = istate->hgram_curr;
	kds_hash->hash_max = UINT_MAX;
	while (istate->hgram_curr < limit)
	{
		Size	next_size = istate->hgram_size[istate->hgram_curr];

		if (kds_hash->usage + required + next_size > kds_hash->length)
		{
			if (required == 0)
				elog(ERROR, "hash-key didn't distribute tuples enough");
			kds_hash->hash_max =
				istate->hgram_curr * (1U << istate->hgram_shift);
			break;
		}
		required += istate->hgram_size[istate->hgram_curr];
		istate->hgram_curr++;
	}

	/* No more records to read? */
	if (required == 0)
		return false;

	/*
	 * Load from the tuplestore
	 */
	while (tuplestore_gettupleslot(tupstore, true, false, tupslot))
	{
		pg_crc32	hash = get_tuple_hashvalue(istate, tupslot);

		if (hash >= kds_hash->hash_min && hash <= kds_hash->hash_max)
		{
			HeapTuple		tuple = ExecFetchSlotTuple(tupslot);
			kern_hashitem  *khitem;
			Size			entry_size;
			cl_int			index;

			entry_size = MAXALIGN(offsetof(kern_hashitem, htup) +
								  tuple->t_len);
			khitem = (kern_hashitem *)((char *)kds_hash + kds_hash->usage);
			khitem->hash = hash;
			khitem->rowid = kds_hash->nitems++;
			khitem->t_len = tuple->t_len;
			memcpy(&khitem->htup, tuple->t_data, tuple->t_len);

			index = hash % kds_hash->nslots;
			khitem->next = hash_slots[index];
			hash_slots[index] = kds_hash->usage;

			/* usage increment */
			kds_hash->usage += entry_size;
		}
	}
	Assert(kds_hash->usage <= kds_hash->length);

	return true;
}

/*
 * gpujoin_inner_hash_preload
 *
 * Preload inner relation to the data store with hash-format, for hash-
 * join execution.
 */
static void
gpujoin_inner_hash_preload(GpuJoinState *gjs,
						   innerState *istate,
						   pgstrom_multirels *pmrels)
{
	PlanState		   *scan_ps = istate->state;
	TupleTableSlot	   *scan_slot = scan_ps->ps_ResultTupleSlot;
	TupleDesc			scan_desc = scan_slot->tts_tupleDescriptor;
	Size				chunk_size;
	pgstrom_data_store *pds_hash;
	kern_data_store	   *kds_hash;
	cl_uint			   *hash_slots;

	/*
	 * Make a pgstrom_data_store for materialization
	 */
	chunk_size = (Size)(istate->kmrels_ratio *
						(double)(pmrels->kmrels_length -
								 pmrels->head_length));
	pds_hash = pgstrom_create_data_store_hash(gjs->gts.gcontext,
											  scan_desc,
											  chunk_size,
											  istate->hash_nslots,
											  false);
	kds_hash = pds_hash->kds;
	hash_slots = KERN_DATA_STORE_HASHSLOT(kds_hash);

	while (!istate->scan_done)
	{
		HeapTuple	tuple;
		pg_crc32	hash;
		Size		item_size;
		int			index;

		if (!istate->scan_overflow)
			scan_slot = ExecProcNode(istate->state);
		else
		{
			scan_slot = istate->scan_overflow;
			istate->scan_overflow = NULL;
		}

		if (TupIsNull(scan_slot))
		{
			istate->scan_done = true;
			break;
		}
		tuple = ExecFetchSlotTuple(scan_slot);
		hash = get_tuple_hashvalue(istate, scan_slot);
		item_size = MAXALIGN(offsetof(kern_hashitem, htup) + tuple->t_len);

		/*
		 * Once we switched to the Tuplestore instead of kern_data_store,
		 * we try to materialize the inner relation once, then split it
		 * to the suitable scale.
		 */
		if (istate->tupstore)
		{
			tuplestore_puttuple(istate->tupstore, tuple);
			istate->hgram_size[hash >> istate->hgram_shift] += item_size;
			continue;
		}

		/* do we have enough space to store? */
		if (kds_hash->usage + item_size <= kds_hash->length)
		{
			kern_hashitem  *khitem = (kern_hashitem *)
				((char *)kds_hash + kds_hash->usage);
			khitem->hash = hash;
			khitem->rowid = kds_hash->nitems++;
			khitem->t_len = tuple->t_len;
			memcpy(&khitem->htup, tuple->t_data, tuple->t_len);

			index = hash % kds_hash->nslots;
			khitem->next = hash_slots[index];
			hash_slots[index] = kds_hash->usage;

			/* usage increment */
			kds_hash->usage += item_size;
			/* histgram update */
			istate->hgram_size[hash >> istate->hgram_shift] += item_size;
		}
		else
		{
			Assert(istate->scan_overflow == NULL);
			istate->scan_overflow = scan_slot;

			if (multirels_expand_length(gjs, pmrels))
			{
				Size    chunk_size_new = (Size)
					(istate->kmrels_ratio *(double)(pmrels->kmrels_length -
												   pmrels->head_length));
                elog(DEBUG1, "KDS-Hash (depth=%d) expanded %zu => %zu",
                     istate->depth, (Size)kds_hash->length, chunk_size_new);
				pgstrom_expand_data_store(gjs->gts.gcontext,
										  pds_hash, chunk_size_new);
				kds_hash = pds_hash->kds;
				hash_slots = KERN_DATA_STORE_HASHSLOT(kds_hash);
			}
			else if (istate->join_type == JOIN_INNER ||
					 istate->join_type == JOIN_RIGHT)
			{
				/*
				 * In case of INNER or RIGHT join, we don't need to
				 * materialize the underlying relation once, because
				 * its logic don't care about range of hash-value.
				 */
				break;
			}
			else
			{
				/*
				 * If join logic is one of outer, and we cannot expand
				 * a single kern_data_store chunk any more, we switch to
				 * use tuple-store to materialize the underlying relation
				 * once. Then, we split tuples according to the hash range.
				 */
				kern_hashitem  *khitem;
				HeapTupleData	tupData;

				istate->tupstore = tuplestore_begin_heap(false, false,
														 work_mem);
				for (index = 0; index < kds_hash->nslots; index++)
				{
					for (khitem = KERN_HASH_FIRST_ITEM(kds_hash, index);
						 khitem != NULL;
						 khitem = KERN_HASH_NEXT_ITEM(kds_hash, khitem))
					{
						tupData.t_len = khitem->t_len;
						tupData.t_data = &khitem->htup;
						tuplestore_puttuple(istate->tupstore, &tupData);
					}
				}
			}
		}
	}

	/*
	 * Try to preload the kern_data_store if inner relation was too big
	 * to load into a single chunk, thus we materialized them on the
	 * tuple-store once.
	 */
	if (istate->tupstore &&
		!gpujoin_inner_hash_preload_partial(istate, pds_hash))
	{
		Assert(istate->scan_done);
		pgstrom_release_data_store(pds_hash);
		return;
	}

	/* release data store, if no tuples were loaded */
	if (kds_hash && kds_hash->nitems == 0)
	{
		pgstrom_release_data_store(pds_hash);
		return;
	}

	/* OK, successfully preloaded */
	istate->curr_chunk = pds_hash;


}

/*
 * gpujoin_inner_heap_preload
 *
 * Preload inner relation to the data store with row-format, for nested-
 * loop execution.
 */
static void
gpujoin_inner_heap_preload(GpuJoinState *gjs,
						   innerState *istate,
						   pgstrom_multirels *pmrels)
{
	PlanState		   *scan_ps = istate->state;
	TupleTableSlot	   *scan_slot = scan_ps->ps_ResultTupleSlot;
	TupleDesc			scan_desc = scan_slot->tts_tupleDescriptor;
	Size				chunk_size;
	pgstrom_data_store *pds_heap;

	/*
	 * Make a pgstrom_data_store for materialization
	 */
	chunk_size = (Size)(istate->kmrels_ratio *
						(double)(pmrels->kmrels_length -
								 pmrels->head_length));
	pds_heap = pgstrom_create_data_store_row(gjs->gts.gcontext,
											 scan_desc, chunk_size, false);
	while (true)
	{
		if (!istate->scan_overflow)
			scan_slot = ExecProcNode(scan_ps);
		else
		{
			scan_slot = istate->scan_overflow;
			istate->scan_overflow = NULL;
		}

		if (TupIsNull(scan_slot))
		{
			istate->scan_done = true;
			break;
		}

		if (!pgstrom_data_store_insert_tuple(pds_heap, scan_slot))
		{
			/* to be inserted on the next try */
			Assert(istate->scan_overflow = NULL);
			istate->scan_overflow = scan_slot;

			/*
             * We try to expand total length of pgstrom_multirels buffer,
             * as long as it can be acquired on the device memory.
             * If no more physical space is expected, we give up to preload
             * entire relation on this store.
             */
            if (!multirels_expand_length(gjs, pmrels))
                break;

			/*
			 * Once total length of the buffer got expanded, current store
			 * also can have wider space.
			 */
			chunk_size = (Size)(istate->kmrels_ratio *
								(double)(pmrels->kmrels_length -
										 pmrels->head_length));
			chunk_size = STROMALIGN_DOWN(chunk_size);
			pgstrom_expand_data_store(gjs->gts.gcontext, pds_heap, chunk_size);
		}
	}

	/* How many tuples read? */
	if (pds_heap->kds->nitems > 0)
	{
		istate->curr_chunk = pds_heap;
		istate->ntuples += pds_heap->kds->nitems;
	}
	else
	{
		Assert(istate->scan_done);
		pgstrom_release_data_store(pds_heap);
	}
	return;
}

/*
 * gpujoin_create_multirels
 *
 * It construct an empty pgstrom_multirels
 */
static pgstrom_multirels *
gpujoin_create_multirels(GpuJoinState *gjs)
{
	GpuContext *gcontext = gjs->gts.gcontext;
	pgstrom_multirels  *pmrels;
	int			num_rels = gjs->num_rels;
	Size		head_length;
	Size		alloc_length;
	char	   *pos;

	head_length = STROMALIGN(offsetof(pgstrom_multirels,
									  kern.chunks[num_rels]));
	alloc_length = head_length +
		STROMALIGN(sizeof(pgstrom_data_store *) * num_rels) +
		STROMALIGN(sizeof(cl_int) * gcontext->num_context) +
		STROMALIGN(sizeof(CUdeviceptr) * gcontext->num_context) +
		STROMALIGN(sizeof(CUevent) * gcontext->num_context) +
		STROMALIGN(sizeof(CUdeviceptr) * gcontext->num_context);

	pmrels = MemoryContextAllocZero(gcontext->memcxt, alloc_length);
	pmrels->gjs = gjs;
	pmrels->kmrels_length = gjs->kmrels_length;
	pmrels->head_length = head_length;
	pmrels->usage_length = head_length;
	pmrels->ojmap_length = 0;

	pos = (char *)pmrels + head_length;
	pmrels->inner_chunks = (pgstrom_data_store **) pos;
	pos += STROMALIGN(sizeof(pgstrom_data_store *) * num_rels);
	pmrels->refcnt = (cl_int *) pos;
	pos += STROMALIGN(sizeof(cl_int) * gcontext->num_context);
	pmrels->m_kmrels = (CUdeviceptr *) pos;
	pos += STROMALIGN(sizeof(CUdeviceptr) * gcontext->num_context);
	pmrels->ev_loaded = (CUevent *) pos;
	pos += STROMALIGN(sizeof(CUevent) * gcontext->num_context);
	pmrels->m_ojmaps = (CUdeviceptr *) pos;
	pos += STROMALIGN(sizeof(CUdeviceptr) * gcontext->num_context);

	memcpy(pmrels->kern.pg_crc32_table,
		   pg_crc32_table,
		   sizeof(cl_uint) * 256);
	pmrels->kern.nrels = num_rels;
	pmrels->kern.ndevs = gcontext->num_context;
	memset(pmrels->kern.chunks,
		   0,
		   offsetof(pgstrom_multirels, kern.chunks[num_rels]) -
		   offsetof(pgstrom_multirels, kern.chunks[0]));

	return pmrels;
}

/*****/
static pgstrom_multirels *
gpujoin_inner_preload(GpuJoinState *gjs)
{
	pgstrom_multirels  *pmrels;
	int					i, depth;
	struct timeval		tv1, tv2;

	PERFMON_BEGIN(&gjs->gts.pfm_accum, &tv1);

	for (depth = gjs->num_rels; depth > 1; depth--)
	{
		if (!gjs->inners[depth - 1].scan_done)
		{
			/* rescan deeper inner relations, if any */
			for (i = depth; i < gjs->num_rels; i++)
			{
				ExecReScan(gjs->inners[i].state);
				gjs->inners[i].scan_done = false;
			}
			break;
		}
	}

	/*
	 * All the inner relations are already done. Nothing to read any more.
	 */
	if (depth < 1)
		return NULL;

	/*
	 * OK, make a pgstrom_multirels buffer
	 */
	pmrels = gpujoin_create_multirels(gjs);

	for (i = 0; i < gjs->num_rels; i++)
	{
		pgstrom_data_store *pds;
		innerState *istate = &gjs->inners[i];
		bool		scan_forward = false;

		if (!istate->scan_done)
		{
			if (istate->curr_chunk)
			{
				pgstrom_release_data_store(istate->curr_chunk);
				istate->curr_chunk = NULL;
			}
			if (istate->hash_inner_keys != NIL)
				gpujoin_inner_hash_preload(gjs, istate, pmrels);
			else
				gpujoin_inner_heap_preload(gjs, istate, pmrels);
			scan_forward = true;
		}
		Assert(istate->curr_chunk != NULL);
		pds = istate->curr_chunk;

		/* make advanced the usage counter */
		pmrels->inner_chunks[i] = pds;
		pmrels->kern.chunks[i].chunk_offset = pmrels->usage_length;
		pmrels->usage_length += STROMALIGN(pds->kds->length);
		Assert(pmrels->usage_length <= pmrels->kmrels_length);

		if (istate->join_type == JOIN_RIGHT ||
			istate->join_type == JOIN_FULL)
		{
			pmrels->kern.chunks[i].right_outer = true;
			pmrels->kern.chunks[i].ojmap_offset = pmrels->ojmap_length;
			pmrels->ojmap_length += (STROMALIGN(sizeof(cl_bool) *
												pds->kds->nitems) *
									 pmrels->kern.ndevs);
		}
		if (istate->join_type == JOIN_LEFT ||
			istate->join_type == JOIN_FULL)
			pmrels->kern.chunks[i].left_outer = true;

		if (scan_forward)
			gjs->inners[i].ntuples += pds->kds->nitems;
	}
	PERFMON_END(&gjs->gts.pfm_accum, time_inner_load, &tv1, &tv2);

	return pmrels;
}

/*
 * multirels_attach_buffer
 *
 * It attache multirels buffer on a particular gpujoin task.
 */
static pgstrom_multirels *
multirels_attach_buffer(pgstrom_multirels *pmrels)
{
	Assert(pmrels->n_attached >= 0);

	pmrels->n_attached++;

	return pmrels;
}

/*****/
static bool
multirels_get_buffer(pgstrom_multirels *pmrels, GpuTask *gtask,
					 CUdeviceptr *p_kmrels,		/* inner relations */
					 CUdeviceptr *p_ojmaps)		/* left-outer map */
{
	cl_int		cuda_index = gtask->cuda_index;
	CUresult	rc;

	Assert(&pmrels->gjs->gts == gtask->gts);

	if (pmrels->refcnt[cuda_index] == 0)
	{
		CUdeviceptr	m_kmrels = 0UL;
		CUdeviceptr	m_ojmaps = 0UL;

		/* buffer for the inner multi-relations */
		m_kmrels = gpuMemAlloc(gtask, pmrels->kmrels_length);
		if (!m_kmrels)
			return false;

		if (pmrels->ojmap_length > 0 && !pmrels->m_ojmaps[cuda_index])
		{
			m_ojmaps = gpuMemAlloc(gtask, pmrels->ojmap_length);
			if (!m_ojmaps)
			{
				gpuMemFree(gtask, m_kmrels);
				return false;
			}
			/*
			 * Zero clear the left-outer map in sync manner
			 */
			rc = cuMemsetD32(m_ojmaps, 0, pmrels->ojmap_length / sizeof(int));
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuMemsetD32: %s", errorText(rc));
			Assert(!pmrels->m_ojmaps[cuda_index]);
			pmrels->m_ojmaps[cuda_index] = m_ojmaps;
		}
		Assert(!pmrels->m_kmrels[cuda_index]);
		Assert(!pmrels->ev_loaded[cuda_index]);
		pmrels->m_kmrels[cuda_index] = m_kmrels;
	}
	pmrels->refcnt[cuda_index]++;
	*p_kmrels = pmrels->m_kmrels[cuda_index];
	*p_ojmaps = pmrels->m_ojmaps[cuda_index];

	return true;
}

static void
multirels_put_buffer(pgstrom_multirels *pmrels, GpuTask *gtask)
{
	cl_int		cuda_index = gtask->cuda_index;
	CUresult	rc;

	Assert(&pmrels->gjs->gts == gtask->gts);
	Assert(pmrels->refcnt[cuda_index] > 0);
	if (--pmrels->refcnt[cuda_index] == 0)
	{
		/*
		 * OK, no concurrent tasks did not reference the inner-relations
		 * buffer any more, so release it and mark the pointer as NULL.
		 */
		Assert(pmrels->m_kmrels[cuda_index] != 0UL);
		gpuMemFree(gtask, pmrels->m_kmrels[cuda_index]);
		pmrels->m_kmrels[cuda_index] = 0UL;

		/*
		 * Also, event object if any
		 */
		if (pmrels->ev_loaded[cuda_index])
		{
			rc = cuEventDestroy(pmrels->ev_loaded[cuda_index]);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuEventDestroy: %s", errorText(rc));
			pmrels->ev_loaded[cuda_index] = NULL;
		}
		/* should not be dettached prior to device memory release */
		Assert(pmrels->n_attached > 0);
	}
}

static void
multirels_send_buffer(pgstrom_multirels *pmrels, GpuTask *gtask)
{
	cl_int		cuda_index = gtask->cuda_index;
	CUstream	cuda_stream = gtask->cuda_stream;
	CUevent		ev_loaded;
	CUresult	rc;

	Assert(&pmrels->gjs->gts == gtask->gts);
	if (!pmrels->ev_loaded[cuda_index])
	{
		CUdeviceptr	m_kmrels = pmrels->m_kmrels[cuda_index];
		Size		length;
		cl_int		i;

		rc = cuEventCreate(&ev_loaded, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

		/* DMA send to the kern_multirels buffer */
		length = offsetof(kern_multirels, chunks[pmrels->kern.nrels]);
		rc = cuMemcpyHtoDAsync(m_kmrels, &pmrels->kern, length, cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));

		for (i=0; i < pmrels->kern.nrels; i++)
		{
			pgstrom_data_store *pds = pmrels->inner_chunks[i];
			kern_data_store	   *kds = pds->kds;
			Size				offset = pmrels->kern.chunks[i].chunk_offset;

			rc = cuMemcpyHtoDAsync(m_kmrels + offset, kds, kds->length,
								   cuda_stream);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		}
		/* DMA Send synchronization */
		rc = cuEventRecord(ev_loaded, cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventRecord: %s", errorText(rc));
		/* save the event */
		pmrels->ev_loaded[cuda_index] = ev_loaded;
	}
	else
	{
		/* DMA Send synchronization, kicked by other task */
		ev_loaded = pmrels->ev_loaded[cuda_index];
		rc = cuStreamWaitEvent(cuda_stream, ev_loaded, 0);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuStreamWaitEvent: %s", errorText(rc));
	}
}

static void
multirels_colocate_outer_join_maps(pgstrom_multirels *pmrels,
								   GpuTask *gtask, int depth)
{
	GpuContext	   *gcontext = pmrels->gjs->gts.gcontext;
	cl_int			cuda_index = gtask->cuda_index;
	CUstream		cuda_stream = gtask->cuda_stream;
	CUcontext		dst_context = gtask->cuda_context;
	CUcontext		src_context;
	cl_bool		   *dst_lomap;
	cl_bool		   *src_lomap;
	pgstrom_data_store *chunk;
	size_t			nitems;
	cl_int			i;
	CUresult		rc;

	Assert(pmrels->m_ojmaps[cuda_index] != 0UL);
	Assert(gcontext->gpu[cuda_index].cuda_context == gtask->cuda_context);
	chunk = pmrels->inner_chunks[depth - 1];
	nitems = chunk->kds->nitems;
	dst_lomap = KERN_MULTIRELS_OUTER_JOIN_MAP(&pmrels->kern, depth, nitems,
											  cuda_index,
											  pmrels->m_ojmaps[cuda_index]);
	for (i=0; i < gcontext->num_context; i++)
	{
		/* no need to copy from the destination device */
		if (i == cuda_index)
			continue;
		/* never executed on this device */
		if (!pmrels->m_ojmaps[i])
			continue;

		src_context = gcontext->gpu[i].cuda_context;
		src_lomap = KERN_MULTIRELS_OUTER_JOIN_MAP(&pmrels->kern, depth, nitems,
												  i, pmrels->m_ojmaps[i]);
		rc = cuMemcpyPeerAsync((CUdeviceptr)dst_lomap, dst_context,
							   (CUdeviceptr)src_lomap, src_context,
							   STROMALIGN(sizeof(cl_bool) * (nitems)),
							   cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyPeerAsync: %s", errorText(rc));
	}
}

static void
multirels_detach_buffer(pgstrom_multirels *pmrels)
{
	Assert(pmrels->n_attached > 0);
	if (--pmrels->n_attached == 0)
	{
		GpuContext *gcontext = pmrels->gjs->gts.gcontext;
		int			index;

		for (index=0; index < gcontext->num_context; index++)
		{
			Assert(pmrels->refcnt[index] == 0);
			if (pmrels->m_ojmaps[index] != 0UL)
				__gpuMemFree(gcontext, index, pmrels->m_ojmaps[index]);
		}
		pfree(pmrels);
	}
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
