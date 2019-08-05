/*
 * gpujoin.c
 *
 * GPU accelerated relations join, based on nested-loop or hash-join
 * algorithm.
 * ----
 * Copyright 2011-2019 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2019 (C) The PG-Strom Development Team
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
#include "pg_strom.h"
#include "cuda_gpuscan.h"
#include "cuda_gpujoin.h"

/*
 * GpuJoinPath
 */
typedef struct
{
	CustomPath		cpath;
	int				num_rels;
	int				optimal_gpu;
	Index			outer_relid;	/* valid, if outer scan pull-up */
	List		   *outer_quals;	/* qualifier of outer scan */
	cl_uint			outer_nrows_per_block;
	IndexOptInfo   *index_opt;		/* BRIN-index if any */
	List		   *index_conds;
	List		   *index_quals;
	cl_long			index_nblocks;
	Cost			inner_cost;		/* cost to setup inner heap/hash */
	cl_int		   *sibling_param_id; /* only if partition-wise join child */
	struct {
		JoinType	join_type;		/* one of JOIN_* */
		double		join_nrows;		/* intermediate nrows in this depth */
		Path	   *scan_path;		/* outer scan path */
		List	   *hash_quals;		/* valid quals, if hash-join */
		List	   *join_quals;		/* all the device quals, incl hash_quals */
		Size		ichunk_size;	/* expected inner chunk size */
	} inners[FLEXIBLE_ARRAY_MEMBER];
} GpuJoinPath;

/*
 * GpuJoinInfo - private state object of CustomScan(GpuJoin)
 */
typedef struct
{
	int			num_rels;
	int			optimal_gpu;
	char	   *kern_source;
	cl_uint		extra_flags;
	cl_uint		varlena_bufsz;
	List	   *used_params;
	List	   *outer_quals;
	List	   *outer_refs;
	double		outer_ratio;
	double		outer_nrows;		/* number of estimated outer nrows*/
	int			outer_width;		/* copy of @plan_width in outer path */
	Cost		outer_startup_cost;	/* copy of @startup_cost in outer path */
	Cost		outer_total_cost;	/* copy of @total_cost in outer path */
	cl_uint		outer_nrows_per_block;
	cl_int		sibling_param_id;
	Oid			index_oid;			/* OID of BRIN-index, if any */
	List	   *index_conds;		/* BRIN-index key conditions */
	List	   *index_quals;		/* original BRIN-index qualifiers */
	/* for each depth */
	List	   *plan_nrows_in;	/* list of floatVal for planned nrows_in */
	List	   *plan_nrows_out;	/* list of floatVal for planned nrows_out */
	List	   *ichunk_size;
	List	   *join_types;
	List	   *join_quals;
	List	   *other_quals;
	List	   *hash_inner_keys;	/* if hash-join */
	List	   *hash_outer_keys;	/* if hash-join */
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
	privs = lappend(privs, makeInteger(gj_info->optimal_gpu));
	privs = lappend(privs, makeString(pstrdup(gj_info->kern_source)));
	privs = lappend(privs, makeInteger(gj_info->extra_flags));
	privs = lappend(privs, makeInteger(gj_info->varlena_bufsz));
	exprs = lappend(exprs, gj_info->used_params);
	exprs = lappend(exprs, gj_info->outer_quals);
	privs = lappend(privs, gj_info->outer_refs);
	privs = lappend(privs, pmakeFloat(gj_info->outer_ratio));
	privs = lappend(privs, pmakeFloat(gj_info->outer_nrows));
	privs = lappend(privs, makeInteger(gj_info->outer_width));
	privs = lappend(privs, pmakeFloat(gj_info->outer_startup_cost));
	privs = lappend(privs, pmakeFloat(gj_info->outer_total_cost));
	privs = lappend(privs, makeInteger(gj_info->outer_nrows_per_block));
	privs = lappend(privs, makeInteger(gj_info->sibling_param_id));
	privs = lappend(privs, makeInteger(gj_info->index_oid));
	privs = lappend(privs, gj_info->index_conds);
	exprs = lappend(exprs, gj_info->index_quals);
	/* for each depth */
	privs = lappend(privs, gj_info->plan_nrows_in);
	privs = lappend(privs, gj_info->plan_nrows_out);
	privs = lappend(privs, gj_info->ichunk_size);
	privs = lappend(privs, gj_info->join_types);
	exprs = lappend(exprs, gj_info->join_quals);
	exprs = lappend(exprs, gj_info->other_quals);
	exprs = lappend(exprs, gj_info->hash_inner_keys);
	exprs = lappend(exprs, gj_info->hash_outer_keys);

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
	gj_info->optimal_gpu = intVal(list_nth(privs, pindex++));
	gj_info->kern_source = strVal(list_nth(privs, pindex++));
	gj_info->extra_flags = intVal(list_nth(privs, pindex++));
	gj_info->varlena_bufsz = intVal(list_nth(privs, pindex++));
	gj_info->used_params = list_nth(exprs, eindex++);
	gj_info->outer_quals = list_nth(exprs, eindex++);
	gj_info->outer_refs = list_nth(privs, pindex++);
	gj_info->outer_ratio = floatVal(list_nth(privs, pindex++));
	gj_info->outer_nrows = floatVal(list_nth(privs, pindex++));
	gj_info->outer_width = intVal(list_nth(privs, pindex++));
	gj_info->outer_startup_cost = floatVal(list_nth(privs, pindex++));
	gj_info->outer_total_cost = floatVal(list_nth(privs, pindex++));
	gj_info->outer_nrows_per_block = intVal(list_nth(privs, pindex++));
	gj_info->sibling_param_id = intVal(list_nth(privs, pindex++));
	gj_info->index_oid = intVal(list_nth(privs, pindex++));
	gj_info->index_conds = list_nth(privs, pindex++);
	gj_info->index_quals = list_nth(exprs, eindex++);
	/* for each depth */
	gj_info->plan_nrows_in = list_nth(privs, pindex++);
	gj_info->plan_nrows_out = list_nth(privs, pindex++);
	gj_info->ichunk_size = list_nth(privs, pindex++);
	gj_info->join_types = list_nth(privs, pindex++);
    gj_info->join_quals = list_nth(exprs, eindex++);
	gj_info->other_quals = list_nth(exprs, eindex++);
	gj_info->hash_inner_keys = list_nth(exprs, eindex++);
    gj_info->hash_outer_keys = list_nth(exprs, eindex++);

	gj_info->ps_src_depth = list_nth(privs, pindex++);
	gj_info->ps_src_resno = list_nth(privs, pindex++);
	Assert(pindex == list_length(privs));
	Assert(eindex == list_length(exprs));

	return gj_info;
}

/*
 * GpuJoinState - execution state object of GpuJoin
 */
typedef struct
{
	/*
	 * Execution status
	 */
	PlanState		   *state;
	ExprContext		   *econtext;

	/*
	 * Join properties; both nest-loop and hash-join
	 */
	int					depth;
	JoinType			join_type;
	double				nrows_ratio;
	cl_uint				ichunk_size;
#if PG_VERSION_NUM < 100000	
	List			   *join_quals;		/* single element list of ExprState */
	List			   *other_quals;	/* single element list of ExprState */
#else
	ExprState		   *join_quals;
	ExprState		   *other_quals;
#endif

	/*
	 * Join properties; only hash-join
	 */
	List			   *hash_outer_keys;
	List			   *hash_inner_keys;
//	List			   *hash_keylen;
//	List			   *hash_keybyval;
//	List			   *hash_keytype;

	/* CPU Fallback related */
	AttrNumber		   *inner_dst_resno;
	AttrNumber			inner_src_anum_min;
	AttrNumber			inner_src_anum_max;
	cl_long				fallback_inner_index;
	pg_crc32			fallback_inner_hash;
	cl_bool				fallback_inner_matched;
} innerState;

typedef struct
{
	GpuTaskState	gts;
	struct GpuJoinSharedState *gj_sstate;	/* may be on DSM */
	struct GpuJoinRuntimeStat *gj_rtstat;	/* valid only PG10 or later */

	/* Inner Buffers */
	struct GpuJoinSiblingState *sibling;	/* only partition-wise join */
	CUdeviceptr		m_kmrels;
	CUdeviceptr	   *m_kmrels_array;		/* only master process */
	GpuContext	  **m_kmrels_gcontext;	/* only master process */
	dsm_segment	   *seg_kmrels;
	cl_int			curr_outer_depth;

	/*
	 * Expressions to be used in the CPU fallback path
	 */
	List		   *join_types;
#if PG_VERSION_NUM < 100000
	List		   *outer_quals;	/* list of ExprState */
#else
	ExprState	   *outer_quals;
#endif
	double			outer_ratio;
	double			outer_nrows;
	List		   *hash_outer_keys;
	List		   *join_quals;
	/* result width per tuple for buffer length calculation */
	int				result_width;

	/*
	 * CPU Fallback
	 */
	TupleTableSlot *slot_fallback;
	ProjectionInfo *proj_fallback;		/* slot_fallback -> scan_slot */
	AttrNumber	   *outer_dst_resno;	/* destination attribute number to */
	AttrNumber		outer_src_anum_min;	/* be mapped on the slot_fallback */
	AttrNumber		outer_src_anum_max;
	cl_int			fallback_resume_depth;
	cl_long			fallback_thread_count;
	cl_long			fallback_outer_index;

	/*
	 * Properties of underlying inner relations
	 */
	int				num_rels;
	innerState		inners[FLEXIBLE_ARRAY_MEMBER];
} GpuJoinState;

/*
 * GpuJoinSharedState - shared inner hash/heap buffer
 */
struct GpuJoinSharedState
{
	dsm_handle		ss_handle;		/* DSM handle of the SharedState */
	cl_uint			ss_length;		/* Length of the SharedState */
	size_t			offset_runtime_stat; /* offset to the runtime statistics */
	Latch		   *masterLatch;	/* Latch of the master process */
	dsm_handle		kmrels_handle;	/* DSM of kern_multirels */
	pg_atomic_uint32 needs_colocation; /* non-zero, if colocation is needed */
	pg_atomic_uint32 preload_done;	/* non-zero, if preload is done */
	pg_atomic_uint32 pg_nworkers;	/* # of active PG workers */
	struct {
		pg_atomic_uint32 pg_nworkers; /* # of PG workers per GPU device */
		CUipcMemHandle	m_handle;	/* IPC handle for PG workers */
	} pergpu[FLEXIBLE_ARRAY_MEMBER];
};
typedef struct GpuJoinSharedState	GpuJoinSharedState;

/*
 * GpuJoinSiblingState - shared state if GpuJoin works as partition leaf.
 *
 * In case when GpuJoin is pushed-down across Append-node that represents
 * a partition table, its inner hash/heap table is shared to all the
 * partition-leafs. So, we have to be careful to detach DSM segment of the
 * inner table, because other sibling GpuJoin may still need inner buffer
 * even if one GpuJoin exits.
 */
struct GpuJoinSiblingState
{
	cl_int			n_actives;	/* number of active siblings */
	GpuJoinSharedState *gj_sstate;
	dsm_segment	   *seg_kmrels;
	struct {
		CUdeviceptr	m_kmrels;	/* device memory of inner buffer */
		GpuContext *gcontext;	/* GpuContext which allocates/opens m_kmrels */
	} pergpu[FLEXIBLE_ARRAY_MEMBER];
};
typedef struct GpuJoinSiblingState	GpuJoinSiblingState;

/*
 * GpuJoinRuntimeStat - shared runtime statistics
 */
struct GpuJoinRuntimeStat
{
	GpuTaskRuntimeStat	c;		/* common statistics */
	struct {
		pg_atomic_uint64 inner_nitems;
		pg_atomic_uint64 right_nitems;
	} jstat[FLEXIBLE_ARRAY_MEMBER];
};
typedef struct GpuJoinRuntimeStat	GpuJoinRuntimeStat;

#define GPUJOIN_RUNTIME_STAT(gj_sstate)							\
	((GpuJoinRuntimeStat *)((char *)(gj_sstate) +				\
							(gj_sstate)->offset_runtime_stat))

/*
 * GpuJoinTask - task object of GpuJoin
 */
typedef struct
{
	GpuTask			task;
	cl_bool			with_nvme_strom;	/* true, if NVMe-Strom */
	cl_int			outer_depth;		/* base depth, if RIGHT OUTER */
	/* DMA buffers */
	pgstrom_data_store *pds_src;	/* data store of outer relation */
	pgstrom_data_store *pds_dst;	/* data store of result buffer */
	kern_gpujoin	kern;		/* kern_gpujoin of this request */
} GpuJoinTask;

/* static variables */
static set_join_pathlist_hook_type set_join_pathlist_next;
static CustomPathMethods	gpujoin_path_methods;
static CustomScanMethods	gpujoin_plan_methods;
static CustomExecMethods	gpujoin_exec_methods;
static bool					enable_gpunestloop;				/* GUC */
static bool					enable_gpuhashjoin;				/* GUC */
static bool					enable_partitionwise_gpujoin;	/* GUC */

/* static functions */
static void gpujoin_switch_task(GpuTaskState *gts, GpuTask *gtask);
static GpuTask *gpujoin_next_task(GpuTaskState *gts);
static GpuTask *gpujoin_terminator_task(GpuTaskState *gts,
										cl_bool *task_is_ready);
static TupleTableSlot *gpujoin_next_tuple(GpuTaskState *gts);
static cl_uint get_tuple_hashvalue(innerState *istate,
								   bool is_inner_hashkeys,
								   TupleTableSlot *slot,
								   bool *p_is_null_keys);

static char *gpujoin_codegen(PlannerInfo *root,
							 CustomScan *cscan,
							 GpuJoinInfo *gj_info,
							 List *tlist,
							 codegen_context *context);

static void createGpuJoinSharedState(GpuJoinState *gjs,
									 ParallelContext *pcxt,
									 void *coordinate);
static void gpujoinColocateOuterJoinMapsToHost(GpuJoinState *gjs);

/*
 * misc declarations
 */

/* copied from joinpath.c */
#define PATH_PARAM_BY_REL(path, rel)  \
	((path)->param_info && bms_overlap(PATH_REQ_OUTER(path), (rel)->relids))

/*
 * returns true, if pathnode is GpuJoin
 */
bool
pgstrom_path_is_gpujoin(const Path *pathnode)
{
	CustomPath *cpath = (CustomPath *) pathnode;

	if (IsA(cpath, CustomPath) &&
		cpath->methods == &gpujoin_path_methods)
		return true;
	return false;
}

/*
 * returns true, if plannode is GpuJoin
 */
bool
pgstrom_plan_is_gpujoin(const Plan *plannode)
{
	if (IsA(plannode, CustomScan) &&
		((CustomScan *) plannode)->methods == &gpujoin_plan_methods)
		return true;
	return false;
}

/*
 * returns true, if planstate node is GpuJoin
 */
bool
pgstrom_planstate_is_gpujoin(const PlanState *ps)
{
	if (IsA(ps, CustomScanState) &&
		((CustomScanState *) ps)->methods == &gpujoin_exec_methods)
		return true;
	return false;
}

/*
 * gpujoin_get_optimal_gpu
 */
cl_int
gpujoin_get_optimal_gpu(const Path *pathnode)
{
	if (pgstrom_path_is_gpujoin(pathnode))
		return ((GpuJoinPath *)pathnode)->optimal_gpu;
	return -1;
}

/*
 * pgstrom_copy_gpujoin_path
 *
 * Note that this function shall never copies individual fields recursively.
 */
Path *
pgstrom_copy_gpujoin_path(const Path *pathnode)
{
	GpuJoinPath	   *gjpath_old = (GpuJoinPath *) pathnode;
	GpuJoinPath	   *gjpath_new;
	Size			length;

	Assert(pgstrom_path_is_gpujoin(pathnode));
	length = offsetof(GpuJoinPath, inners[gjpath_old->num_rels]);
	gjpath_new = palloc0(length);
	memcpy(gjpath_new, gjpath_old, length);

	return &gjpath_new->cpath.path;
}

/*
 * dump_gpujoin_rel
 *
 * Dumps candidate GpuJoinPath for debugging
 */
static void
__dump_gpujoin_rel(StringInfo buf, PlannerInfo *root, RelOptInfo *rel)
{
	Relids		relids = rel->relids;
	List	   *range_tables = root->parse->rtable;
	int			rtindex = -1;
	bool		is_first = true;


	if (rel->reloptkind != RELOPT_BASEREL)
		appendStringInfo(buf, "(");

	while ((rtindex = bms_next_member(relids, rtindex)) >= 0)
	{
		RangeTblEntry  *rte = rt_fetch(rtindex, range_tables);
		Alias		   *eref = rte->eref;

		if (!is_first)
			appendStringInfoString(buf, ", ");
		appendStringInfo(buf, "%s", eref->aliasname);
		if (rte->rtekind == RTE_RELATION)
		{
			char   *relname = get_rel_name(rte->relid);

			if (relname && strcmp(relname, eref->aliasname) != 0)
				appendStringInfo(buf, " [%s]", relname);
			pfree(relname);
		}
		is_first = false;
	}

	if (rel->reloptkind != RELOPT_BASEREL)
		appendStringInfo(buf, ")");
}

/*
 * estimate_inner_buffersize
 */
static Size
estimate_inner_buffersize(PlannerInfo *root,
						  RelOptInfo *joinrel,
						  Path *outer_path,
						  GpuJoinPath *gpath,
						  double num_chunks)
{
	Size		inner_total_sz;
	cl_int		ncols;
	cl_int		i, num_rels = gpath->num_rels;

	/*
	 * Estimation: size of inner hash/heap buffer
	 */
	inner_total_sz = STROMALIGN(offsetof(kern_multirels,
										 chunks[num_rels]));
	for (i=0; i < num_rels; i++)
	{
		Path	   *inner_path = gpath->inners[i].scan_path;
		RelOptInfo *inner_rel = inner_path->parent;
		PathTarget *inner_reltarget = inner_rel->reltarget;
		Size		inner_nrows = (Size)inner_path->rows;
		Size		chunk_size;
		Size		htup_size;

		/*
		 * NOTE: PathTarget->width is not reliable for base relations 
		 * because this fields shows the length of attributes which
		 * are actually referenced, however, we usually load physical
		 * tuples on the KDS/KHash buffer if base relation.
		 */
		ncols = list_length(inner_reltarget->exprs);

		htup_size = MAXALIGN(offsetof(HeapTupleHeaderData,
									  t_bits[BITMAPLEN(ncols)]));
		if (inner_rel->reloptkind != RELOPT_BASEREL)
			htup_size += MAXALIGN(inner_reltarget->width);
		else
		{
			htup_size += MAXALIGN(((double)(BLCKSZ -
											SizeOfPageHeaderData)
								   * inner_rel->pages
								   / Max(inner_rel->tuples, 1.0))
								  - sizeof(ItemIdData)
								  - SizeofHeapTupleHeader);
		}

		/*
		 * estimation of the inner chunk in this depth
		 */
		if (gpath->inners[i].hash_quals != NIL)
			chunk_size = KDS_ESTIMATE_HASH_LENGTH(ncols,inner_nrows,htup_size);
		else
			chunk_size = KDS_ESTIMATE_ROW_LENGTH(ncols,inner_nrows,htup_size);
		gpath->inners[i].ichunk_size = chunk_size;
		inner_total_sz += chunk_size;
	}
	return inner_total_sz;
}

/*
 * cost_gpujoin
 *
 * estimation of GpuJoin cost
 */
static bool
cost_gpujoin(PlannerInfo *root,
			 GpuJoinPath *gpath,
			 RelOptInfo *joinrel,
			 Path *outer_path,
			 Relids required_outer,
			 int parallel_nworkers)
{
	Cost		inner_cost = 0.0;
	Cost		startup_cost = 0.0;
	Cost		run_cost = 0.0;
	Cost		run_cost_per_chunk = 0.0;
	Cost		startup_delay;
	Size		inner_buffer_sz = 0;
	double		gpu_ratio = pgstrom_gpu_operator_cost / cpu_operator_cost;
	double		parallel_divisor = 1.0;
	double		num_chunks;
	double		outer_ntuples = outer_path->rows;
	int			i, num_rels = gpath->num_rels;
	bool		retval = false;

	/*
	 * Cost comes from the outer-path
	 */
	if (gpath->outer_relid > 0)
	{
		double		dummy;

		pgstrom_common_relscan_cost(root,
									outer_path->parent,
									gpath->outer_quals,
									parallel_nworkers,	/* parallel scan */
									gpath->index_opt,
									gpath->index_quals,
									gpath->index_nblocks,
									&parallel_divisor,
									&dummy,
									&num_chunks,
									&gpath->outer_nrows_per_block,
									&startup_cost,
									&run_cost);
		outer_ntuples /= parallel_divisor;
	}
	else
	{
		if (pathtree_has_gpupath(outer_path))
			startup_cost = pgstrom_gpu_setup_cost / 2;
		else
			startup_cost = pgstrom_gpu_setup_cost;
		startup_cost = outer_path->startup_cost;
		run_cost = outer_path->total_cost - outer_path->startup_cost;
		num_chunks = estimate_num_chunks(outer_path);

		parallel_divisor = get_parallel_divisor(&gpath->cpath.path);
	}

	/*
	 * Estimation of inner hash/heap buffer, and number of internal loop
	 * to process in-kernel Join logic
	 */
	inner_buffer_sz = estimate_inner_buffersize(root,
												joinrel,
												outer_path,
												gpath,
												num_chunks);
	/*
	 * Cost for each depth
	 */
	for (i=0; i < num_rels; i++)
	{
		Path	   *scan_path = gpath->inners[i].scan_path;
		List	   *hash_quals = gpath->inners[i].hash_quals;
		List	   *join_quals = gpath->inners[i].join_quals;
		double		join_nrows = gpath->inners[i].join_nrows;
		Size		ichunk_size = gpath->inners[i].ichunk_size;
		Cost		inner_run_cost;
		QualCost	join_quals_cost;

		/*
		 * FIXME: Right now, KDS_FORMAT_ROW/HASH does not support KDS size
		 * larger than 4GB because of 32bit index from row_index[] or
		 * hash_slot[]. So, tentatively, we prohibit to construct GpuJoin
		 * path which contains large tables (expected 1.5GB, with safety
		 * margin) in the inner buffer.
		 * In the future version, up to 32GB chunk will be supported using
		 * least 3bit because row-/hash-item shall be always put on 64bit
		 * aligned location.
		 */
		if (ichunk_size >= 0x60000000UL)
		{
			if (client_min_messages <= DEBUG1 || log_min_messages <= DEBUG1)
			{
				StringInfoData buf;

				initStringInfo(&buf);
				__dump_gpujoin_rel(&buf, root, scan_path->parent);
				elog(DEBUG1, "expected inner size (%zu) on %s is too large",
					 ichunk_size, buf.data);
				pfree(buf.data);
			}
			return false;
		}

		/*
		 * cost to load all the tuples from inner-path
		 *
		 * FIXME: This is an ad-hoc workaround not to breal multi-level
		 * GpuJoin chain. PostgreSQL v11 begins to support parallel build
		 * of inner hash-table, and its cost tends to be discounted by
		 * num of parallel workers. It is exactly a rapid option to process
		 * inner side, however, cannot use combined GpuJoin+GpuPreAgg here.
		 * So, we tentatively discount the inner cost by parallel_nworkers,
		 * as if it is processed in parallel.
		 * It shall be supported on the later PG-Strom version.
		 */
		inner_run_cost = (scan_path->total_cost -
						  scan_path->startup_cost) / parallel_divisor;
		inner_cost += scan_path->startup_cost + inner_run_cost;

		/* cost for join_qual startup */
		cost_qual_eval(&join_quals_cost, join_quals, root);
		join_quals_cost.per_tuple *= gpu_ratio;
		startup_cost += join_quals_cost.startup;

		/*
		 * cost to evaluate join qualifiers according to
		 * the GpuJoin logic
		 */
		if (hash_quals != NIL)
		{
			/*
			 * GpuHashJoin - It computes hash-value of inner tuples by CPU,
			 * but outer tuples by GPU, then it evaluates join-qualifiers
			 * for each items on inner hash table by GPU.
			 */
			cl_uint		num_hashkeys = list_length(hash_quals);
			double		hash_nsteps = scan_path->rows /
				(double)__KDS_NSLOTS((Size)scan_path->rows);

			/* cost to compute inner hash value by CPU */
			inner_cost += cpu_operator_cost * num_hashkeys * scan_path->rows;

			/* cost to comput hash value by GPU */
			run_cost += (pgstrom_gpu_operator_cost *
						 num_hashkeys *
						 outer_ntuples);
			/* cost to evaluate join qualifiers */
			run_cost += (join_quals_cost.per_tuple *
						 Max(hash_nsteps, 1.0) *
						 outer_ntuples);
		}
		else
		{
			/*
			 * GpuNestLoop - It evaluates join-qual for each pair of outer
			 * and inner tuples. So, its run_cost is usually higher than
			 * GpuHashJoin.
			 */
			double		inner_ntuples = scan_path->rows;

			/* cost to preload inner heap tuples by CPU */
			inner_cost += cpu_tuple_cost * inner_ntuples;

			/* cost to evaluate join qualifiers */
			run_cost_per_chunk += (join_quals_cost.per_tuple *
								   outer_ntuples *
								   inner_ntuples);
		}
		/* number of outer items on the next depth */
		outer_ntuples = join_nrows / parallel_divisor;
	}
	/* outer DMA send cost */
	run_cost += (double)num_chunks * pgstrom_gpu_dma_cost;

	/* inner DMA send cost */
	inner_cost += ((double)inner_buffer_sz /
				   (double)pgstrom_chunk_size()) * pgstrom_gpu_dma_cost;

	/* cost for GPU projection */
	startup_cost += joinrel->reltarget->cost.startup;
	run_cost += (joinrel->reltarget->cost.per_tuple +
				 cpu_tuple_cost) * gpu_ratio * gpath->cpath.path.rows;
	/* cost for DMA receive (GPU-->host) */
	run_cost += cost_for_dma_receive(joinrel, -1.0);

	/* cost to exchange tuples */
	run_cost += cpu_tuple_cost * gpath->cpath.path.rows;

	/*
	 * delay to fetch the first tuple
	 */
	startup_delay = run_cost * (1.0 / num_chunks);

	/*
	 * Put cost value on the gpath.
	 */
	gpath->cpath.path.rows /= parallel_divisor;
	gpath->cpath.path.startup_cost = startup_cost + inner_cost + startup_delay;
	gpath->cpath.path.total_cost = startup_cost + inner_cost + run_cost;
	gpath->inner_cost = inner_cost;

	/*
	 * NOTE: If very large number of rows are estimated, it may cause
	 * overflow of variables, then makes nearly negative infinite cost
	 * even though the plan is very bad.
	 * At this moment, we put assertion to detect it.
	 */
	Assert(gpath->cpath.path.startup_cost >= 0.0 &&
		   gpath->cpath.path.total_cost >= 0.0);
	retval = add_path_precheck(gpath->cpath.path.parent,
							   gpath->cpath.path.startup_cost,
							   gpath->cpath.path.total_cost,
							   NULL, required_outer);
	/* Dumps candidate GpuJoinPath for debugging */
	if (client_min_messages <= DEBUG1 || log_min_messages <= DEBUG1)
	{
		StringInfoData buf;

		initStringInfo(&buf);
		for (i=gpath->num_rels-1; i >= 0; i--)
		{
			JoinType	join_type = gpath->inners[i].join_type;
			Path	   *inner_path = gpath->inners[i].scan_path;
			bool		is_nestloop = (gpath->inners[i].hash_quals == NIL);

			__dump_gpujoin_rel(&buf, root, inner_path->parent);
			appendStringInfo(&buf, " %s%s ",
							 join_type == JOIN_FULL ? "F" :
							 join_type == JOIN_LEFT ? "L" :
							 join_type == JOIN_RIGHT ? "R" : "I",
							 is_nestloop ? "NL" : "HJ");
		}
		__dump_gpujoin_rel(&buf, root, outer_path->parent);
		elog(DEBUG1, "GpuJoin: %s Cost=%.2f..%.2f%s",
			 buf.data,
			 gpath->cpath.path.startup_cost,
			 gpath->cpath.path.total_cost,
			 !retval ? " rejected" : "");
		pfree(buf.data);
	}
	return retval;
}

typedef struct
{
	JoinType	join_type;
	Path	   *inner_path;
	List	   *join_quals;
	List	   *hash_quals;
	double		join_nrows;
} inner_path_item;

static GpuJoinPath *
create_gpujoin_path(PlannerInfo *root,
					RelOptInfo *joinrel,
					Path *outer_path,
					List *inner_path_items_list,
					ParamPathInfo *param_info,
					Relids required_outer,
					bool try_parallel_path)
{
	GpuJoinPath *gjpath;
	cl_int		num_rels = list_length(inner_path_items_list);
	ListCell   *lc;
	int			parallel_nworkers = 0;
	bool		inner_parallel_safe = true;
	int			i;

	/* parallel path must have parallel_safe sub-paths */
	if (try_parallel_path)
	{
		if (!outer_path->parallel_safe)
			return NULL;
		foreach (lc, inner_path_items_list)
		{
			inner_path_item *ip_item = lfirst(lc);

			if (!ip_item->inner_path->parallel_safe)
				return NULL;
		}
		parallel_nworkers = outer_path->parallel_workers;
	}

	gjpath = palloc0(offsetof(GpuJoinPath, inners[num_rels + 1]));
	NodeSetTag(gjpath, T_CustomPath);
	gjpath->cpath.path.pathtype = T_CustomScan;
	gjpath->cpath.path.parent = joinrel;
	gjpath->cpath.path.pathtarget = joinrel->reltarget;
	gjpath->cpath.path.param_info = param_info;	// XXXXXX
	gjpath->cpath.path.parallel_safe = try_parallel_path;
	gjpath->cpath.path.parallel_workers = parallel_nworkers;
	gjpath->cpath.path.pathkeys = NIL;
	gjpath->cpath.path.rows = joinrel->rows;
	gjpath->cpath.flags = 0;
	gjpath->cpath.methods = &gpujoin_path_methods;
	gjpath->outer_relid = 0;
	gjpath->outer_quals = NULL;
	gjpath->num_rels = num_rels;

	i = 0;
	foreach (lc, inner_path_items_list)
	{
		inner_path_item *ip_item = lfirst(lc);
		List	   *hash_quals;

		if (enable_gpuhashjoin && ip_item->hash_quals != NIL)
			hash_quals = ip_item->hash_quals;
		else if (enable_gpunestloop &&
				 (ip_item->join_type == JOIN_INNER ||
				  ip_item->join_type == JOIN_LEFT))
			hash_quals = NIL;
		else
		{
			pfree(gjpath);
			return NULL;
		}
		if (!ip_item->inner_path->parallel_safe)
			inner_parallel_safe = false;
		gjpath->inners[i].join_type = ip_item->join_type;
		gjpath->inners[i].join_nrows = ip_item->join_nrows;
		gjpath->inners[i].scan_path = ip_item->inner_path;
		gjpath->inners[i].hash_quals = hash_quals;
		gjpath->inners[i].join_quals = ip_item->join_quals;
		gjpath->inners[i].ichunk_size = 0;		/* to be set later */
		i++;
	}
	Assert(i == num_rels);

	/* Try to pull up outer scan if enough simple */
	pgstrom_pullup_outer_scan(root, outer_path,
							  &gjpath->outer_relid,
							  &gjpath->outer_quals,
							  &gjpath->optimal_gpu,
							  &gjpath->index_opt,
							  &gjpath->index_conds,
							  &gjpath->index_quals,
							  &gjpath->index_nblocks);
	if (try_parallel_path && gjpath->outer_relid != 0)
		gjpath->cpath.path.parallel_aware = true;

	/*
	 * cost calculation of GpuJoin, then, add this path to the joinrel,
	 * unless its cost is not obviously huge.
	 */
	if (cost_gpujoin(root,
					 gjpath,
					 joinrel,
					 outer_path,
					 required_outer,
					 parallel_nworkers))
	{
		List   *custom_paths = list_make1(outer_path);

		/* informs planner a list of child pathnodes */
		for (i=0; i < num_rels; i++)
			custom_paths = lappend(custom_paths, gjpath->inners[i].scan_path);
		gjpath->cpath.custom_paths = custom_paths;
		gjpath->cpath.path.parallel_safe = (joinrel->consider_parallel &&
											outer_path->parallel_safe &&
											inner_parallel_safe);
		return gjpath;
	}
	pfree(gjpath);
	return NULL;
}

/*
 * gpujoin_find_cheapest_path
 *
 * finds the cheapest path-node but not parameralized by other relations
 * involved in this GpuJoin.
 */
static Path *
gpujoin_find_cheapest_path(PlannerInfo *root,
						   RelOptInfo *joinrel,
						   RelOptInfo *inputrel,
						   bool only_parallel_safe)
{
	Path	   *input_path = inputrel->cheapest_total_path;
	Relids		other_relids;

	ListCell   *lc;

	other_relids = bms_difference(joinrel->relids, inputrel->relids);
	if ((only_parallel_safe && !input_path->parallel_safe) ||
		bms_overlap(PATH_REQ_OUTER(input_path), other_relids))
	{
		/*
		 * We try to find out the second best path if cheapest path is not
		 * sufficient for the requiement of GpuJoin
		 */
		foreach (lc, inputrel->pathlist)
		{
			Path   *curr_path = lfirst(lc);

			if (only_parallel_safe && !curr_path->parallel_safe)
				continue;
			if (bms_overlap(PATH_REQ_OUTER(curr_path), other_relids))
				continue;
			if (input_path == NULL ||
				input_path->total_cost > curr_path->total_cost)
				input_path = curr_path;
		}
	}
	bms_free(other_relids);

	return input_path;
}

/*
 * extract_gpuhashjoin_quals - pick up qualifiers usable for GpuHashJoin
 */
static List *
extract_gpuhashjoin_quals(PlannerInfo *root,
						  RelOptInfo *outer_rel,
						  RelOptInfo *inner_rel,
						  JoinType jointype,
						  List *restrict_clauses)
{
	List	   *hash_quals = NIL;
	ListCell   *lc;

	foreach (lc, restrict_clauses)
	{
		RestrictInfo   *rinfo = (RestrictInfo *) lfirst(lc);

		/*
		 * If processing an outer join, only use its own join clauses
		 * for hashing.  For inner joins we need not be so picky.
		 */
		if (IS_OUTER_JOIN(jointype) && rinfo->is_pushed_down)
			continue;

		/* Is it hash-joinable clause? */
		if (!rinfo->can_join || !OidIsValid(rinfo->hashjoinoperator))
			continue;
		Assert(is_opclause(rinfo->clause));

		/*
		 * Check if clause has the form "outer op inner" or
		 * "inner op outer". If suitable, we may be able to choose
		 * GpuHashJoin logic. See clause_sides_match_join also.
		 */
		if ((bms_is_subset(rinfo->left_relids,  outer_rel->relids) &&
			 bms_is_subset(rinfo->right_relids, inner_rel->relids)) ||
			(bms_is_subset(rinfo->left_relids,  inner_rel->relids) &&
			 bms_is_subset(rinfo->right_relids, outer_rel->relids)))
		{
			OpExpr	   *op = (OpExpr *)rinfo->clause;
			Node	   *arg1 = linitial(op->args);
			Node	   *arg2 = lsecond(op->args);
			devtype_info *dtype;

			/* hash-join key must support device hash function */
			dtype = pgstrom_devtype_lookup(exprType(arg1));
			if (!dtype || !dtype->hash_func)
				continue;
			dtype = pgstrom_devtype_lookup(exprType(arg2));
			if (!dtype || !dtype->hash_func)
				continue;

			/* OK, it is hash-joinable qualifier */
			hash_quals = lappend(hash_quals, rinfo);
		}
	}
	return hash_quals;
}

#if PG_VERSION_NUM >= 100000
/*
 * Partition support for GPU-aware custom-plans (GpuJoin, GpuPreAgg)
 *
 * In case when GPU-aware custom-plans try to pick up input path which
 * already scanned and unified the partitioned child tables, it may be
 * valuable to push down GpuJoin/GpuPreAgg before the Append.
 * Unlike CPU-only tasks, GPU-aware custom-plans need to send/receive
 * data to/from the device, so ping-pong between CPU and GPU are usually
 * inefficient.
 *
 * For example, we may have the expected query execution plan below:
 *   Final-Aggregate
 *    + GpuPreAgg
 *       + GpuJoin
 *       |  + Append
 *       |     + Scan on t_1
 *       |     + Scan on t_2
 *       |     + Scan on t_3
 *       + Scan on t_x
 *
 * In this case, all the records in t_1/t_2/t_3 must be loaded to RAM
 * once, then moved to GPU.
 *
 * What we want to run is below:
 *
 *   Final-Aggregation
 *    + Append
 *       + GpuPreAgg
 *       |  + GpuJoin on t_1
 *       |     + Scan on t_x
 *       + GpuPreAgg
 *       |  + GpuJoin on t_2
 *       |     + Scan on t_x
 *       + GpuPreAgg
 *          + GpuJoin on t_3
 *             + Scan on t_x
 *
 * The above form delivers much smaller data to Append, because most of
 * the data stream is preliminary aggregated on GpuPreAgg stage.
 */

/*
 * adjust_appendrel_attr_needed
 *
 * 'attr_needed' of child relations of partition/inheritance are not
 * initialized (probably, because no code path references the attribute
 * on the PG side), then it leads wrong construction of joinrel with
 * missing target-vars.
 * So, we fix up 'attr_needed' of the base relation here.
 *
 * See adjust_appendrel_attrs() also.
 */
static bool
adjust_appendrel_attr_needed(PlannerInfo *root,
							 List *part_tree_list,
							 RelOptInfo *parent,
							 RelOptInfo *subrel)
{
	List	   *part_tree;
	ListCell   *lc;
	int			i, j, k;

	foreach (lc, part_tree_list)
	{
		List		   *temp = lfirst(lc);
		AppendRelInfo  *head = (AppendRelInfo *)linitial(temp);
		AppendRelInfo  *tail = (AppendRelInfo *)llast(temp);

		if (bms_is_member(head->parent_relid, parent->relids) &&
			bms_is_member(tail->child_relid, subrel->relids))
		{
			part_tree = temp;
			goto found;
		}
	}
	return false;
found:

	for (i=parent->min_attr, j=0; i <= parent->max_attr; i++, j++)
	{
		Relids		needed = parent->attr_needed[j];

		if (!needed)
			continue;
		if (i <= 0)
			k = i;
		else
		{
			Var			__var;
			Var		   *var = &__var;
			ListCell   *lc;

			memset(var, 0, sizeof(Var));
			var->varattno = i;
			foreach (lc, part_tree)
			{
				AppendRelInfo *apinfo = lfirst(lc);

				var = list_nth(apinfo->translated_vars, var->varattno - 1);
				if (!var)
					break;
			}
			if (!var)
				continue;		/* dropped column */
			if (!IsA(var, Var))
				return false;	/* UNION ALL? */
			if (var->varno != subrel->relid)
				return false;
			k = var->varattno;
		}

		if (k >= subrel->min_attr && k <= subrel->max_attr)
		{
			k -= subrel->min_attr;

			if (!subrel->attr_needed[k])
				subrel->attr_needed[k] = bms_copy(needed);
			else
				Assert(bms_equal(subrel->attr_needed[k], needed));
		}
	}
	return true;
}

/*
 * make_pseudo_special_join_info
 */
static SpecialJoinInfo *
make_pseudo_special_join_info(PlannerInfo *root,
							  List *part_tree_list,
							  SpecialJoinInfo *__sj)
{
	SpecialJoinInfo *sj = copyObject(__sj);
	ListCell   *lc;
	Bitmapset  *temp;

	foreach (lc, part_tree_list)
	{
		List   *part_tree = lfirst(lc);
		Index	part, leaf;

		part = ((AppendRelInfo *)linitial(part_tree))->parent_relid;
		leaf = ((AppendRelInfo *)llast(part_tree))->child_relid;

		if (bms_is_member(part, sj->min_lefthand))
		{
			Assert(!bms_is_member(leaf, sj->min_lefthand));
			temp = bms_del_member(sj->min_lefthand, part);
			sj->min_lefthand = bms_add_member(temp, leaf);
		}

		if (bms_is_member(part, sj->min_righthand))
		{
			Assert(!bms_is_member(leaf, sj->min_righthand));
			temp = bms_del_member(sj->min_righthand, part);
			sj->min_righthand = bms_add_member(temp, leaf);
		}

		if (bms_is_member(part, sj->syn_lefthand))
		{
			Assert(!bms_is_member(leaf, sj->syn_lefthand));
			temp = bms_del_member(sj->syn_lefthand, part);
			sj->syn_lefthand = bms_add_member(temp, leaf);
		}

		if (bms_is_member(part, sj->syn_righthand))
		{
			Assert(!bms_is_member(leaf, sj->syn_righthand));
			temp = bms_del_member(sj->syn_righthand, part);
			sj->syn_righthand = bms_add_member(temp, leaf);
		}
	}
	return sj;
}

/*
 * pickup_partition_tree - makes List of AppendRelInfo
 */
static List *
__pickup_partition_tree_walker(PlannerInfo *root,
							   RelOptInfo *part_rel,
							   Index leaf_relid)
{
	List	   *temp;
	ListCell   *lc;

	foreach (lc, root->append_rel_list)
	{
		AppendRelInfo *apinfo = lfirst(lc);

		if (apinfo->child_relid == leaf_relid)
		{
			if (bms_is_member(apinfo->parent_relid, part_rel->relids))
				return list_make1(apinfo);
			temp = __pickup_partition_tree_walker(root, part_rel,
												  apinfo->parent_relid);
			if (temp != NIL)
				return lappend(temp, apinfo);
		}
	}
	return NIL;
}

static List *
pickup_partition_tree(PlannerInfo *root,
					  RelOptInfo *part_rel,
					  RelOptInfo *leaf_rel)
{
	List	   *results = NIL;
	List	   *temp;
	ListCell   *lc;

	foreach (lc, root->append_rel_list)
	{
		AppendRelInfo *apinfo = lfirst(lc);

		if (bms_is_member(apinfo->child_relid, leaf_rel->relids))
		{
			if (bms_is_member(apinfo->parent_relid, part_rel->relids))
				results = lappend(results, list_make1(apinfo));
			else
			{
				temp = __pickup_partition_tree_walker(root, part_rel,
													  apinfo->parent_relid);
				if (temp != NIL)
					results = lappend(results, lappend(temp, apinfo));
			}
		}
	}
	return results;
}

/*
 * fixup_appendrel_child_relids
 */
static Relids
__fixup_appendrel_child_relids(Relids relids_orig,
							   RelOptInfo *append_rel,
							   List *part_tree_list)
{
	Relids		result = NULL;
	int			index = -1;
	ListCell   *lc;

	while ((index = bms_next_member(relids_orig, index)) >= 0)
	{
		if (bms_is_member(index, append_rel->relids))
		{
			foreach (lc, part_tree_list)
			{
				List   *part_tree = lfirst(lc);
				Index	part, leaf;

				part = ((AppendRelInfo *)linitial(part_tree))->parent_relid;
				leaf = ((AppendRelInfo *)llast(part_tree))->child_relid;

				if (index == part)
				{
					result = bms_add_member(result, leaf);
					goto found;
				}
			}
		}
		result = bms_add_member(result, index);
	found:
		;
	}
	return result;
}

/*
 * fixup_appendrel_child_varnode
 */
typedef struct
{
	PlannerInfo *root;
	RelOptInfo	*append_rel;
	List		*part_tree_list;
} fixup_appendrel_child_varnode_context;

static Node *
__fixup_appendrel_child_varnode(Node *node, void *__context)
{
	fixup_appendrel_child_varnode_context *con = __context;

	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		Var			   *var = (Var *) node;
		ListCell	   *lc1, *lc2;
		AppendRelInfo  *apinfo;

		foreach (lc1, con->part_tree_list)
		{
			List	   *part_tree = lfirst(lc1);
			int			part, leaf;

			part = ((AppendRelInfo *)linitial(part_tree))->parent_relid;
			leaf = ((AppendRelInfo *)llast(part_tree))->child_relid;
			if (var->varno != part)
				continue;
			/* system column? */
			if (var->varattno <= 0)
				return (Node *)makeVar(leaf,
									   var->varattno,
									   var->vartype,
									   var->vartypmod,
									   var->varcollid, 0);
			/* elsewhere, walk down the translated_vars */
			foreach (lc2, part_tree)
			{
				apinfo = (AppendRelInfo *)lfirst(lc2);

				var = list_nth(apinfo->translated_vars,
							   var->varattno - 1);
			}
			Assert(var->varno == leaf);
			return copyObject((Node *)var);
		}
		/* likely, reference to inner (non-partitioned) columns */
		return copyObject((Node *)var);
	}
	return expression_tree_mutator(node, __fixup_appendrel_child_varnode, con);
}

/*
 * fixup_appendrel_child_varnode
 */
List *
fixup_appendrel_child_varnode(List *exprs_list,
							  PlannerInfo *root,
							  RelOptInfo *append_rel,
							  RelOptInfo *leaf_rel)
{
	fixup_appendrel_child_varnode_context con;
	List   *part_tree_list;

	if (exprs_list == NIL)
		return NIL;
	Assert(IsA(exprs_list, List));

	part_tree_list = pickup_partition_tree(root, append_rel, leaf_rel);
	if (part_tree_list == NIL)
		elog(ERROR, "Bug? could not pick up partition tree");

	con.root = root;
	con.append_rel = append_rel;
	con.part_tree_list = part_tree_list;
	return (List *)__fixup_appendrel_child_varnode((Node *)exprs_list, &con);
}

static Path *
setup_append_child_path(PlannerInfo *root,
						RelOptInfo *append_rel,
						Path *subpath)
{
	Path	   *newpath;
	PathTarget *new_target;
	List	   *part_tree_list;

	part_tree_list = pickup_partition_tree(root, append_rel, subpath->parent);
	if (part_tree_list == NIL)
		return NULL;

	if (pgstrom_path_is_gpujoin(subpath))
	{
		newpath = pgstrom_copy_gpujoin_path(subpath);
	}
	else if (pgstrom_path_is_gpuscan(subpath))
	{
		newpath = pgstrom_copy_gpuscan_path(subpath);
	}
	else
	{
		size_t	sz;

		switch (nodeTag(subpath))
		{
			case T_Path:			sz = sizeof(Path); break;
			case T_IndexPath:		sz = sizeof(IndexPath);	break;
			case T_BitmapHeapPath:	sz = sizeof(BitmapHeapPath); break;
			case T_BitmapAndPath:	sz = sizeof(BitmapAndPath); break;
			case T_BitmapOrPath:	sz = sizeof(BitmapOrPath); break;
			case T_TidPath:			sz = sizeof(TidPath); break;
			case T_SubqueryScanPath:sz = sizeof(SubqueryScanPath); break;
			case T_ForeignPath:		sz = sizeof(ForeignPath); break;
			case T_CustomPath:		sz = sizeof(CustomPath); break;
			case T_NestPath:		sz = sizeof(NestPath); break;
			case T_MergePath:		sz = sizeof(MergePath); break;
			case T_HashPath:		sz = sizeof(HashPath); break;
			case T_ProjectionPath:	sz = sizeof(ProjectionPath); break;
			default:
				return NULL;	/* not expected */
		}
		newpath = palloc(sz);
		memcpy(newpath, subpath, sz);
	}
	new_target = copy_pathtarget(subpath->pathtarget);
	if (new_target->exprs != NIL)
	{
		fixup_appendrel_child_varnode_context con;

		con.root      = root;
		con.part_tree_list = part_tree_list;
		new_target->exprs = (List *)
			__fixup_appendrel_child_varnode((Node *)new_target->exprs, &con);
	}
	newpath->pathtarget = new_target;

	return newpath;
}

/*
 * build_partitionwise_gpujoin_final
 */
static bool
build_partitionwise_gpujoin_final(PlannerInfo *root,
								  RelOptInfo *join_rel,
								  RelOptInfo *outer_rel,
								  RelOptInfo *inner_rel,
								  JoinType join_type,
								  JoinPathExtraData *extra,
								  RelOptInfo *append_rel,
								  List *part_tree_list)
{
	SpecialJoinInfo *sjinfo;
	List	   *restrictlist = NIL;
	ListCell   *lc;

	sjinfo = make_pseudo_special_join_info(root, part_tree_list,
										   extra->sjinfo);
	foreach (lc, extra->restrictlist)
	{
		RestrictInfo *rinfo = copyObject(lfirst(lc));
		fixup_appendrel_child_varnode_context con;

		con.root = root;
		con.append_rel = append_rel;
		con.part_tree_list = part_tree_list;
		rinfo->clause = (Expr *)
			__fixup_appendrel_child_varnode((Node *)rinfo->clause, &con);

		rinfo->clause_relids =
			__fixup_appendrel_child_relids(rinfo->clause_relids,
										   append_rel,
										   part_tree_list);
		rinfo->required_relids =
			__fixup_appendrel_child_relids(rinfo->required_relids,
										   append_rel,
										   part_tree_list);
		rinfo->outer_relids =
			__fixup_appendrel_child_relids(rinfo->outer_relids,
										   append_rel,
										   part_tree_list);
		rinfo->nullable_relids =
			__fixup_appendrel_child_relids(rinfo->nullable_relids,
										   append_rel,
										   part_tree_list);
		rinfo->left_relids =
			__fixup_appendrel_child_relids(rinfo->left_relids,
										   append_rel,
										   part_tree_list);
		rinfo->right_relids =
			__fixup_appendrel_child_relids(rinfo->right_relids,
										   append_rel,
										   part_tree_list);

		restrictlist = lappend(restrictlist, rinfo);
	}
	add_paths_to_joinrel(root,
						 join_rel,
						 outer_rel,
						 inner_rel,
						 join_type,
						 sjinfo,
						 restrictlist);
	return true;
}

/*
 * build_partitionwise_gpujoin_leafs
 */
static List *
build_partitionwise_gpujoin_leafs(PlannerInfo *root,
								  AppendPath *append_path,
								  List *inner_rels_list,
								  JoinType join_type,		/* only GpuJoin */
								  JoinPathExtraData *extra,	/* only GpuJoin */
								  bool try_parallel_path,
								  List *join_info_list_saved,
								  int *p_parallel_nworkers,
								  AppendPath **p_append_path,
								  Cost *p_discount_cost)
{
	RelOptInfo *append_rel = append_path->path.parent;
	Cost		discount_cost = 0.0;
	List	   *new_append_subpaths = NIL;
	ListCell   *lc1, *lc2;
	ListCell   *cell;
	int			parallel_nworkers = 0;
	bool		all_gpujoin = true;

	foreach (lc1, append_path->subpaths)
	{
		Path	   *curr_path = lfirst(lc1);
		RelOptInfo *curr_rel = curr_path->parent;
		RelOptInfo *leaf_rel = curr_path->parent;
		Path	   *child_path = NULL;
		List	   *child_path_list = NIL;
		List	   *part_tree_list;

		part_tree_list = pickup_partition_tree(root, append_rel, leaf_rel);
		if (!part_tree_list)
			return NIL;

		if (leaf_rel->reloptkind == RELOPT_OTHER_MEMBER_REL)
		{
			if (!adjust_appendrel_attr_needed(root,
											  part_tree_list,
											  append_rel,
											  leaf_rel))
				return NIL;
		}

		/*
		 * MEMO: make_join_rel() makes OUTER JOIN decision based on
		 * SpecialJoinInfo, so we have to fixup relid of the parent
		 * relation as if child relation is referenced.
		 */
		root->join_info_list = NIL;
		foreach (lc2, join_info_list_saved)
		{
			SpecialJoinInfo *sj = lfirst(lc2);

			sj = make_pseudo_special_join_info(root, part_tree_list, sj);
			root->join_info_list = lappend(root->join_info_list, sj);
		}

		foreach (lc2, inner_rels_list)
		{
			RelOptInfo *inner_rel = lfirst(lc2);
			RelOptInfo *join_rel;

			join_rel = find_join_rel(root, bms_union(curr_rel->relids,
													 inner_rel->relids));
			if (!join_rel)
			{
				RelOptKind	reloptkind_saved = leaf_rel->reloptkind;

				leaf_rel->reloptkind = RELOPT_BASEREL;
				PG_TRY();
				{
					join_rel = make_join_rel(root,
											 curr_rel,
											 inner_rel);
				}
				PG_CATCH();
				{
					leaf_rel->reloptkind = reloptkind_saved;
					PG_RE_THROW();
				}
				PG_END_TRY();
				leaf_rel->reloptkind = reloptkind_saved;

				if (!join_rel)
					return NIL;
			}
			/* add various JOIN path; including GpuJoin */
			if (extra && lc2 == list_tail(inner_rels_list))
			{
				if (!build_partitionwise_gpujoin_final(root,
													   join_rel,
													   curr_rel,
													   inner_rel,
													   join_type,
													   extra,
													   append_rel,
													   part_tree_list))
					return NIL;
			}
			set_cheapest(join_rel);
			curr_rel = join_rel;
		}

		if (try_parallel_path)
			child_path_list = curr_rel->partial_pathlist;
		else
			child_path_list = curr_rel->pathlist;
		foreach (cell, child_path_list)
		{
			Path   *temppath = lfirst(cell);
			Relids	paramrels = PATH_REQ_OUTER(temppath);

			/* XXX - Is this restriction reasonable? */
			if (!bms_is_empty(paramrels))
				continue;
			if (child_path == NULL ||
				child_path->total_cost > temppath->total_cost)
				child_path = temppath;
		}
		if (!child_path)
			return NIL;

		parallel_nworkers = Max(parallel_nworkers,
								child_path->parallel_workers);
		if (!pgstrom_path_is_gpujoin(child_path))
			all_gpujoin = false;
		new_append_subpaths = lappend(new_append_subpaths, child_path);
	}
	/* see add_paths_to_append_rel() */
	parallel_nworkers = Max(parallel_nworkers,
							fls(list_length(append_path->subpaths)));
	parallel_nworkers = Min(parallel_nworkers,
							max_parallel_workers_per_gather);
	/*
	 * Special optimization -- If multiple append subpaths are GpuJoin
	 * which have identical inner set, we can skip re-read of inner
	 * tables.
	 */
	if (all_gpujoin && list_length(new_append_subpaths) > 1)
	{
		Relids	   *inner_relids = NULL;
		int			j, num_rels = -1;
		cl_int	   *sibling_param_id;

		foreach (cell, new_append_subpaths)
		{
			GpuJoinPath *gpath = lfirst(cell);

			if (cell == list_head(new_append_subpaths))
			{
				num_rels = gpath->num_rels;
				inner_relids = alloca(sizeof(Relids) * num_rels);
				for (j=0; j < num_rels; j++)
				{
					Path   *ipath = gpath->inners[j].scan_path;
					inner_relids[j] = ipath->parent->relids;
				}
			}
			else
			{
				if (gpath->num_rels != num_rels)
					goto gpujoin_not_compatible;
				for (j=0; j < gpath->num_rels; j++)
				{
					Path   *ipath = gpath->inners[j].scan_path;

					if (!bms_equal(inner_relids[j],
								   ipath->parent->relids))
						goto gpujoin_not_compatible;
				}
			}
		}
		/*
		 * OK, all the sibling GpuJoin paths have identical inner
		 * relations. We can skip re-read of same contents for each.
		 */
		sibling_param_id = palloc(sizeof(cl_int) * num_rels);
		memset(sibling_param_id, -1, sizeof(cl_int) * num_rels);

		foreach (cell, new_append_subpaths)
		{
			GpuJoinPath *gpath = lfirst(cell);

			gpath->sibling_param_id = sibling_param_id;
			if (cell != list_head(new_append_subpaths))
				discount_cost += gpath->inner_cost;
		}
	}
gpujoin_not_compatible:
	/* make a debug message */
	if (client_min_messages <= DEBUG2 || log_min_error_statement <= DEBUG2)
	{
		StringInfoData	buf;
		ListCell	   *cell;
		int				count = 1;

		initStringInfo(&buf);
		foreach (cell, inner_rels_list)
		{
			RelOptInfo *irel= lfirst(cell);

			if (cell == list_head(inner_rels_list))
				appendStringInfoString(&buf, "inner: ");
			__dump_gpujoin_rel(&buf, root, irel);
			appendStringInfoString(&buf, " x ");
		}
		appendStringInfoString(&buf, "outer: ");
		__dump_gpujoin_rel(&buf, root, append_rel);
		appendStringInfoString(&buf, " was expanded to:");

		foreach (cell, new_append_subpaths)
		{
			Path   *subpath = lfirst(cell);

			appendStringInfo(&buf, "\n        %d: ", count++);
			if (!pgstrom_path_is_gpujoin(subpath))
			{
				__dump_gpujoin_rel(&buf, root, subpath->parent);
			}
			else
			{
				GpuJoinPath *gp = (GpuJoinPath *)subpath;
				Path   *op = linitial(gp->cpath.custom_paths);
				int		j;

				for (j=gp->num_rels-1; j >= 0; j--)
				{
					Path	*ip = gp->inners[j].scan_path;

					__dump_gpujoin_rel(&buf, root, ip->parent);
					appendStringInfo(&buf, " x ");
				}
				__dump_gpujoin_rel(&buf, root, op->parent);
			}
			appendStringInfo(&buf, " Cost=%.2f..%.2f",
							 subpath->startup_cost,
							 subpath->total_cost);

			appendStringInfo(&buf, " parallel (%s %s %d)",
							 subpath->parallel_aware ? "t" : "f",
							 subpath->parallel_safe ? "t" : "f",
							 subpath->parallel_workers);
		}
		if (discount_cost > 0.0)
			appendStringInfo(&buf, "\n        inner discount cost=%.2f",
							 discount_cost);
		elog(DEBUG2, "GpuJoin: %s", buf.data);
		pfree(buf.data);
	}
	*p_parallel_nworkers = parallel_nworkers;
	*p_append_path       = append_path;
	*p_discount_cost     = discount_cost;

	return new_append_subpaths;
}

/*
 * extract_partitionwise_pathlist
 */
static List *
__extract_partitionwise_pathlist(PlannerInfo *root,
								 Path *outer_path,
								 List *inner_rels_list,		/* only GpuJoin */
								 JoinType join_type,		/* only GpuJoin */
								 JoinPathExtraData *extra,	/* only GpuJoin */
								 bool try_parallel_path,
								 AppendPath **p_append_path,
								 int *p_parallel_nworkers,
								 Cost *p_discount_cost)
{
	List	   *result = NIL;
	List	   *subpaths_list;
	ListCell   *lc;

	if (IsA(outer_path, AppendPath))
	{
		AppendPath *append_path = (AppendPath *) outer_path;
		bool		enable_gpunestloop_saved;
		bool		enable_gpuhashjoin_saved;
		List	   *join_info_list_saved;
		List	  **join_rel_level_saved;

		if (append_path->partitioned_rels == NIL)
			return NIL;		/* not a partition */

		join_rel_level_saved = root->join_rel_level;
		join_info_list_saved = root->join_info_list;
		enable_gpunestloop_saved = enable_gpunestloop;
		enable_gpuhashjoin_saved = enable_gpuhashjoin;
		if (!enable_partitionwise_gpujoin)
		{
			enable_gpunestloop = false;
			enable_gpuhashjoin = false;
		}
		PG_TRY();
		{
			/* temporary disables "dynamic programming" algorithm */
			root->join_rel_level = NULL;
			subpaths_list =
				build_partitionwise_gpujoin_leafs(root,
												  append_path,
												  inner_rels_list,
												  join_type,
												  extra,
												  try_parallel_path,
												  join_info_list_saved,
												  p_parallel_nworkers,
												  p_append_path,
												  p_discount_cost);
		}
		PG_CATCH();
		{
			/* restore */
			root->join_rel_level = join_rel_level_saved;
			root->join_info_list = join_info_list_saved;
			enable_gpunestloop = enable_gpunestloop_saved;
			enable_gpuhashjoin = enable_gpuhashjoin_saved;
			PG_RE_THROW();
		}
		PG_END_TRY();
		/* restore */
		root->join_rel_level = join_rel_level_saved;
		root->join_info_list = join_info_list_saved;
		enable_gpunestloop = enable_gpunestloop_saved;
		enable_gpuhashjoin = enable_gpuhashjoin_saved;
	}
	else if (IsA(outer_path, NestPath) ||
			 IsA(outer_path, MergePath) ||
			 IsA(outer_path, HashPath))
	{
		JoinPath   *jpath = (JoinPath *) outer_path;
		Path	   *ipath = jpath->innerjoinpath;

		if (jpath->jointype != JOIN_INNER &&
			jpath->jointype != JOIN_LEFT)
			return NIL;		/* not a supported join type */
		inner_rels_list = lcons(ipath->parent, inner_rels_list);
		subpaths_list =
			__extract_partitionwise_pathlist(root,
											 jpath->outerjoinpath,
											 inner_rels_list,
											 join_type,
											 extra,
											 try_parallel_path,
											 p_append_path,
											 p_parallel_nworkers,
											 p_discount_cost);
	}
	else if (pgstrom_path_is_gpujoin(outer_path))
	{
		GpuJoinPath *gjpath = (GpuJoinPath *) outer_path;
		Path   *gjouter_path = linitial(gjpath->cpath.custom_paths);
		int		i;

		for (i=gjpath->num_rels-1; i >= 0; i--)
		{
			Path   *ipath = gjpath->inners[i].scan_path;

			if (gjpath->inners[i].join_type != JOIN_INNER &&
				gjpath->inners[i].join_type != JOIN_LEFT)
				return NIL;	/* not a supported join type */
			inner_rels_list = lcons(ipath->parent, inner_rels_list);
		}
		subpaths_list =
			__extract_partitionwise_pathlist(root,
											 gjouter_path,
											 inner_rels_list,
											 join_type,
											 extra,
											 try_parallel_path,
											 p_append_path,
											 p_parallel_nworkers,
											 p_discount_cost);
	}
	else if (IsA(outer_path, ProjectionPath))
	{
		ProjectionPath *projection_path = (ProjectionPath *)outer_path;
		List	   *temp;
		ListCell   *lc;

		temp = __extract_partitionwise_pathlist(root,
												projection_path->subpath,
												inner_rels_list,
												join_type,
												extra,
												try_parallel_path,
												p_append_path,
												p_parallel_nworkers,
												p_discount_cost);
		foreach (lc, temp)
		{
			Path	   *subpath = lfirst(lc);
			ProjectionPath *newpath;
		
			newpath = create_projection_path(root,
                                             subpath->parent,
                                             subpath,
											 projection_path->path.pathtarget);
			subpaths_list = lappend(subpaths_list, newpath);
		}
	}
	else if (IsA(outer_path, GatherPath))
	{
		Path   *subpath = ((GatherPath *) outer_path)->subpath;

		subpaths_list = __extract_partitionwise_pathlist(root,
														 subpath,
														 inner_rels_list,
														 join_type,
														 extra,
														 try_parallel_path,
														 p_append_path,
														 p_parallel_nworkers,
														 p_discount_cost);
	}
	else
	{
		return NIL;
	}

	foreach (lc, subpaths_list)
	{
		RelOptInfo *append_rel = (*p_append_path)->path.parent;
		Path	   *oldpath = lfirst(lc);
		Path	   *newpath;

		newpath = setup_append_child_path(root, append_rel, oldpath);
		if (!newpath)
			return NIL;
		result = lappend(result, newpath);
	}
	return result;
}

List *
extract_partitionwise_pathlist(PlannerInfo *root,
							   Path *outer_path,
							   bool try_parallel_path,
							   AppendPath **p_append_path,
							   int *p_parallel_nworkers,
							   Cost *p_discount_cost)
{
	return __extract_partitionwise_pathlist(root,
											outer_path,
											NIL,		/* only GpuJoin */
											JOIN_INNER,	/* only GpuJoin */
											NULL,		/* only GpuJoin */
											try_parallel_path,
											p_append_path,
											p_parallel_nworkers,
											p_discount_cost);
}

#endif		/* >=PG10; partition support */

/*
 * try_add_gpujoin_append_paths
 */
static void
try_add_gpujoin_append_paths(PlannerInfo *root,
							 RelOptInfo *joinrel,
							 Path *outer_path,
							 Path *inner_path,
							 JoinType join_type,
							 JoinPathExtraData *extra,
							 Relids required_outer,
							 bool try_parallel_path)
{
#if PG_VERSION_NUM >= 100000
	List	   *inner_rels_list = list_make1(inner_path->parent);
	List	   *subpaths_list = NIL;
	List	   *partitioned_rels;
	AppendPath *append_path;

	int			parallel_nworkers;

	Cost		discount_cost;

	bool		all_gpujoin = true;

	if (join_type != JOIN_INNER &&
		join_type != JOIN_LEFT)
		return;

#if PG_VERSION_NUM < 110000
	if (try_parallel_path)
		return;		/* no parallel append support at PG10 */
#endif
	Assert(extra != NULL);
	subpaths_list = __extract_partitionwise_pathlist(root,
													 outer_path,
													 inner_rels_list,
													 join_type,
													 extra,
													 try_parallel_path,
													 &append_path,
													 &parallel_nworkers,
													 &discount_cost);
	if (subpaths_list == NIL)
		return;
	/*
	 * Now inner_path X outer_path is distributed to all the leaf-pathnodes.
	 * Then, create a new AppendPath.
	 */
	partitioned_rels = copyObject(append_path->partitioned_rels);
#if PG_VERSION_NUM < 110000
	/* PG10 has no support of Parallel Append */
	append_path = create_append_path(joinrel,
									 subpaths_list,
									 required_outer,
									 parallel_nworkers,
									 partitioned_rels);
	/* discount inner heap/hash cost if shared by siblings */
	if (all_gpujoin)
		append_path->path.total_cost -= discount_cost;
	add_path(joinrel, (Path *) append_path);
#else
	if (try_parallel_path)
	{
		append_path = create_append_path(root, joinrel,
										 NIL, subpaths_list,
										 required_outer,
										 parallel_nworkers, true,
										 partitioned_rels, -1.0);
		if (all_gpujoin)
			append_path->path.total_cost -= discount_cost;
		add_partial_path(joinrel, (Path *) append_path);
	}
	else
	{
		append_path = create_append_path(root, joinrel,
										 subpaths_list, NIL,
										 required_outer,
										 0, false,
										 partitioned_rels, -1.0);
		if (all_gpujoin)
			append_path->path.total_cost -= discount_cost;
		add_path(joinrel, (Path *) append_path);
	}
#endif
#endif
}

/*
 * try_add_gpujoin_paths
 */
static void
try_add_gpujoin_paths(PlannerInfo *root,
					  RelOptInfo *joinrel,
					  Path *outer_path,
					  Path *inner_path,
					  JoinType join_type,
					  JoinPathExtraData *extra,
					  bool try_parallel_path)
{
	Relids			required_outer;
	ParamPathInfo  *param_info;
	inner_path_item *ip_item;
	List		   *ip_items_list;
	List		   *restrict_clauses = extra->restrictlist;
	ListCell	   *lc;

	/* Quick exit if unsupported join type */
	if (join_type != JOIN_INNER &&
		join_type != JOIN_FULL &&
		join_type != JOIN_RIGHT &&
		join_type != JOIN_LEFT)
		return;

	/*
	 * GpuJoin does not support JOIN in case when either side is parameterized
	 * by the other side.
	 */
	if (bms_overlap(PATH_REQ_OUTER(outer_path), inner_path->parent->relids) ||
		bms_overlap(PATH_REQ_OUTER(inner_path), outer_path->parent->relids))
		return;

	/*
	 * Check to see if proposed path is still parameterized, and reject
	 * if the parameterization wouldn't be sensible.
	 * Note that GpuNestLoop does not support parameterized nest-loop,
	 * only cross-join or non-symmetric join are supported, therefore,
	 * calc_non_nestloop_required_outer() is sufficient.
	 */
	required_outer = calc_non_nestloop_required_outer(outer_path,
													  inner_path);
	if (required_outer &&
		!bms_overlap(required_outer, extra->param_source_rels))
	{
		bms_free(required_outer);
		return;
	}

	/*
	 * Get param info
	 */
	param_info = get_joinrel_parampathinfo(root,
										   joinrel,
										   outer_path,
										   inner_path,
										   extra->sjinfo,
										   required_outer,
										   &restrict_clauses);
	/*
	 * It makes no sense to run cross join on GPU devices
	 */
	if (!restrict_clauses)
		return;

	/*
	 * All the join-clauses must be executable on GPU device.
	 * Even though older version supports HostQuals to be
	 * applied post device join, it leads undesirable (often
	 * unacceptable) growth of the result rows in device join.
	 * So, we simply reject any join that contains host-only
	 * qualifiers.
	 */
	foreach (lc, restrict_clauses)
	{
		RestrictInfo   *rinfo = lfirst(lc);

		if (!pgstrom_device_expression(root, joinrel, rinfo->clause))
			return;
	}

	/*
	 * setup inner_path_item
	 */
	ip_item = palloc0(sizeof(inner_path_item));
	ip_item->join_type = join_type;
	ip_item->inner_path = inner_path;
	ip_item->join_quals = restrict_clauses;
	ip_item->hash_quals = extract_gpuhashjoin_quals(root,
													outer_path->parent,
													inner_path->parent,
													join_type,
													restrict_clauses);
	ip_item->join_nrows = joinrel->rows;
	ip_items_list = list_make1(ip_item);

	for (;;)
	{
		GpuJoinPath	   *gjpath = create_gpujoin_path(root,
													 joinrel,
													 outer_path,
													 ip_items_list,
													 param_info,
													 required_outer,
													 try_parallel_path);
		if (gjpath)
		{
			gjpath->cpath.path.parallel_aware = try_parallel_path;
			if (try_parallel_path)
				add_partial_path(joinrel, (Path *)gjpath);
			else
				add_path(joinrel, (Path *)gjpath);
		}

		/*
		 * pull up outer GpuJoin, and merge if possible
		 */
		if (pgstrom_path_is_gpujoin(outer_path))
		{
			GpuJoinPath	   *gjpath = (GpuJoinPath *) outer_path;
			int				i;

			for (i=gjpath->num_rels-1; i>=0; i--)
			{
				inner_path_item *ip_temp = palloc0(sizeof(inner_path_item));

				ip_temp->join_type  = gjpath->inners[i].join_type;
				ip_temp->inner_path = gjpath->inners[i].scan_path;
				ip_temp->join_quals = gjpath->inners[i].join_quals;
				ip_temp->hash_quals = gjpath->inners[i].hash_quals;
				ip_temp->join_nrows = gjpath->inners[i].join_nrows;

				ip_items_list = lcons(ip_temp, ip_items_list);
			}
			outer_path = linitial(gjpath->cpath.custom_paths);
		}
#ifdef NOT_USED
		/*
		 * MEMO: We are not 100% certain whether it is safe operation to
		 * pull up built-in JOIN as a part of GpuJoin. So, this code block
		 * is disabled right now.
		 */
		else if (outer_path->pathtype == T_NestLoop ||
				 outer_path->pathtype == T_HashJoin ||
				 outer_path->pathtype == T_MergeJoin)
		{
			JoinPath   *join_path = (JoinPath *) outer_path;

			/*
			 * We cannot pull-up outer join path if its inner/outer paths
			 * are mutually parameterized.
			 */
			if (bms_overlap(PATH_REQ_OUTER(join_path->innerjoinpath),
							join_path->outerjoinpath->parent->relids) ||
				bms_overlap(PATH_REQ_OUTER(join_path->outerjoinpath),
							join_path->innerjoinpath->parent->relids))
				return;

			foreach (lc, join_path->joinrestrictinfo)
			{
				RestrictInfo *rinfo = lfirst(lc);

				if (!pgstrom_device_expression(root, joinrel, rinfo->clause))
					return;
			}
			ip_item = palloc0(sizeof(inner_path_item));
			ip_item->join_type = join_path->jointype;
			ip_item->inner_path = join_path->innerjoinpath;
			ip_item->join_quals = join_path->joinrestrictinfo;
			ip_item->hash_quals = extract_gpuhashjoin_quals(
										root,
										join_path->outerjoinpath->parent,
										join_path->innerjoinpath->parent,
										join_path->jointype,
										join_path->joinrestrictinfo);
			ip_item->join_nrows = join_path->path.parent->rows;
			ip_items_list = lcons(ip_item, ip_items_list);
			outer_path = join_path->outerjoinpath;
		}
#endif
		else
		{
			break;
		}
	}

	/* Add partition-wise GpuJoin path */
	if (enable_partitionwise_gpujoin)
		try_add_gpujoin_append_paths(root,
									 joinrel,
									 outer_path,
									 inner_path,
									 join_type,
									 extra,
									 required_outer,
									 try_parallel_path);
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
	Path	   *outer_path;
	Path	   *inner_path;
	ListCell   *lc1, *lc2;

	/* calls secondary module if exists */
	if (set_join_pathlist_next)
		set_join_pathlist_next(root,
							   joinrel,
							   outerrel,
							   innerrel,
							   jointype,
							   extra);

	/* nothing to do, if PG-Strom is not enabled */
	if (!pgstrom_enabled || (!enable_gpunestloop && !enable_gpuhashjoin))
		return;

	/*
	 * make a traditional sequential path
	 */
	outer_path = gpujoin_find_cheapest_path(root,
											joinrel,
											outerrel,
											false);
	if (outer_path)
	{
		inner_path = gpujoin_find_cheapest_path(root,
												joinrel,
												innerrel,
												false);
		if (inner_path)
			try_add_gpujoin_paths(root,
								  joinrel,
								  outer_path,
								  inner_path,
								  jointype, extra, false);
	}

	/*
	 * consider partial paths if any partial outers
	 */
	if (!joinrel->consider_parallel)
		return;

	foreach (lc1, outerrel->partial_pathlist)
	{
		outer_path = lfirst(lc1);

		if (!outer_path->parallel_safe ||
			outer_path->parallel_workers == 0 ||
			bms_overlap(PATH_REQ_OUTER(outer_path), innerrel->relids))
			continue;

		foreach (lc2, innerrel->pathlist)
		{
			inner_path = lfirst(lc2);

			if (!inner_path->parallel_safe ||
				bms_overlap(PATH_REQ_OUTER(inner_path), outerrel->relids))
				continue;

			try_add_gpujoin_paths(root, joinrel,
								  outer_path, inner_path,
								  jointype, extra, true);
		}
	}
}

/*
 * build_device_targetlist
 *
 * It constructs a tentative custom_scan_tlist, according to
 * the expression to be evaluated, returned or shown in EXPLAIN.
 * Usually, all we need to pay attention is columns referenced by host-
 * qualifiers and target-list. However, we may need to execute entire
 * JOIN operations on CPU if GPU raised CpuReCheck error. So, we also
 * adds columns which are also referenced by device qualifiers.
 * (EXPLAIN command has to solve the name, so we have to have these
 * Var nodes in the custom_scan_tlist.)
 *
 * pgstrom_post_planner_gpujoin() may update the custom_scan_tlist
 * to push-down CPU projection. In this case, custom_scan_tlist will
 * have complicated expression not only simple Var-nodes, to simplify
 * targetlist of the CustomScan to reduce cost for CPU projection as
 * small as possible we can.
 */
typedef struct
{
	PlannerInfo	   *root;
	List		   *ps_tlist;
	List		   *ps_depth;
	List		   *ps_resno;
	GpuJoinPath	   *gpath;
	List		   *custom_plans;
	Index			outer_scanrelid;
	bool			resjunk;
} build_device_tlist_context;

static void
build_device_tlist_walker(Node *node, build_device_tlist_context *context)
{
	GpuJoinPath	   *gpath = context->gpath;
	RelOptInfo	   *gjrel = gpath->cpath.path.parent;
	RelOptInfo	   *rel;
	ListCell	   *cell;
	int				i;

	if (!node)
		return;
	if (IsA(node, List))
	{
		List   *temp = (List *)node;

		foreach (cell, temp)
			build_device_tlist_walker(lfirst(cell), context);
	}
	else if (IsA(node, TargetEntry))
	{
		TargetEntry *tle = (TargetEntry *)node;

		build_device_tlist_walker((Node *)tle->expr, context);
	}
	else if (IsA(node, Var))
	{
		Var	   *varnode = (Var *) node;
		Var	   *ps_node;

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
				return;
			}
		}

		/*
		 * Not in the pseudo-scan targetlist, so append this one
		 */
		for (i=0; i <= gpath->num_rels; i++)
		{
			if (i == 0)
			{
				Path   *outer_path = linitial(gpath->cpath.custom_paths);

				rel = outer_path->parent;
				/* special case if outer scan was pulled up */
				if (varnode->varno == context->outer_scanrelid)
				{
					TargetEntry	   *ps_tle =
						makeTargetEntry((Expr *) copyObject(varnode),
										list_length(context->ps_tlist) + 1,
										NULL,
										context->resjunk);
					context->ps_tlist = lappend(context->ps_tlist, ps_tle);
					context->ps_depth = lappend_int(context->ps_depth, i);
					context->ps_resno = lappend_int(context->ps_resno,
													varnode->varattno);
					Assert(bms_is_member(varnode->varno, rel->relids));
					Assert(varnode->varno == rel->relid);
					return;
				}
			}
			else
				rel = gpath->inners[i-1].scan_path->parent;

			if (bms_is_member(varnode->varno, rel->relids))
			{
				Plan   *plan = list_nth(context->custom_plans, i);

				foreach (cell, plan->targetlist)
				{
					TargetEntry *tle = lfirst(cell);

					if (equal(varnode, tle->expr))
					{
						TargetEntry	   *ps_tle =
							makeTargetEntry((Expr *) copyObject(varnode),
											list_length(context->ps_tlist) + 1,
											NULL,
											context->resjunk);
						context->ps_tlist = lappend(context->ps_tlist, ps_tle);
						context->ps_depth = lappend_int(context->ps_depth, i);
						context->ps_resno = lappend_int(context->ps_resno,
														tle->resno);
						return;
					}
				}
				break;
			}
		}
		elog(ERROR, "Bug? uncertain origin of Var-node: %s",
			 nodeToString(varnode));
	}
	else if (IsA(node, PlaceHolderVar))
	{
		PlaceHolderVar *phvnode = (PlaceHolderVar *) node;

		foreach (cell, context->ps_tlist)
		{
			TargetEntry	   *tle = lfirst(cell);

			if (equal(phvnode, tle->expr))
				return;
		}

		/* Not in the pseudo-scan target-list, so append a new one */
		for (i=0; i <= gpath->num_rels; i++)
		{
			if (i == 0)
			{
				/*
				 * NOTE: We don't assume PlaceHolderVar that references the
				 * outer-path which was pulled-up, because only simple scan
				 * paths (SeqScan or GpuScan with no host-only qualifiers)
				 * can be pulled-up, thus, no chance for SubQuery paths.
				 */
				Index	outer_scanrelid = context->outer_scanrelid;
				Path   *outer_path = linitial(gpath->cpath.custom_paths);

				if (outer_scanrelid != 0 &&
					bms_is_member(outer_scanrelid, phvnode->phrels))
					elog(ERROR, "Bug? PlaceHolderVar referenced simple scan outer-path, not expected: %s", nodeToString(phvnode));

				rel = outer_path->parent;
			}
			else
				rel = gpath->inners[i-1].scan_path->parent;

			if (bms_is_subset(phvnode->phrels, rel->relids))
			{
				Plan   *plan = list_nth(context->custom_plans, i);

				foreach (cell, plan->targetlist)
				{
					TargetEntry	   *tle = lfirst(cell);
					TargetEntry	   *ps_tle;
					AttrNumber		ps_resno;

					if (!equal(phvnode, tle->expr))
						continue;

					ps_resno = list_length(context->ps_tlist) + 1;
					ps_tle = makeTargetEntry((Expr *) copyObject(phvnode),
											 ps_resno,
											 NULL,
											 context->resjunk);
					context->ps_tlist = lappend(context->ps_tlist, ps_tle);
					context->ps_depth = lappend_int(context->ps_depth, i);
					context->ps_resno = lappend_int(context->ps_resno,
													tle->resno);
					return;
				}
			}
		}
		elog(ERROR, "Bug? uncertain origin of PlaceHolderVar-node: %s",
			 nodeToString(phvnode));
	}
	else if (!context->resjunk &&
			 pgstrom_device_expression(context->root, gjrel,
									   (Expr *)node))
	{
		TargetEntry	   *ps_tle;

		foreach (cell, context->ps_tlist)
		{
			TargetEntry	   *tle = lfirst(cell);

			if (equal(node, tle->expr))
				return;
		}

		ps_tle = makeTargetEntry((Expr *) copyObject(node),
								 list_length(context->ps_tlist) + 1,
								 NULL,
								 context->resjunk);
		context->ps_tlist = lappend(context->ps_tlist, ps_tle);
		context->ps_depth = lappend_int(context->ps_depth, -1);	/* dummy */
		context->ps_resno = lappend_int(context->ps_resno, -1);	/* dummy */
	}
	else
	{
		List   *temp = pull_var_clause(node, PVC_RECURSE_PLACEHOLDERS);

		foreach (cell, temp)
			build_device_tlist_walker(lfirst(cell), context);
	}
}

static void
build_device_targetlist(PlannerInfo *root,
						GpuJoinPath *gpath,
						CustomScan *cscan,
						GpuJoinInfo *gj_info,
						List *targetlist,
						List *custom_plans)
{
	build_device_tlist_context context;

	Assert(outerPlan(cscan)
		   ? cscan->scan.scanrelid == 0
		   : cscan->scan.scanrelid != 0);

	memset(&context, 0, sizeof(build_device_tlist_context));
	context.root = root;
	context.gpath = gpath;
	context.custom_plans = custom_plans;
	context.outer_scanrelid = cscan->scan.scanrelid;
	context.resjunk = false;
	if (targetlist != NIL)
		build_device_tlist_walker((Node *)targetlist, &context);
	else
	{
		/*
		 * MEMO: If GpuJoinPath is located under ProjectionPath,
		 * create_plan_recurse delivers invalid tlist (=NIL).
		 * So, we picks up referenced Var nodes from the PathTarget,
		 * instead of the tlist.
		 */
		PathTarget *path_target = gpath->cpath.path.pathtarget;
		ListCell   *lc;

		foreach (lc, path_target->exprs)
			build_device_tlist_walker((Node *)lfirst(lc), &context);
	}

	/*
	 * Above are host referenced columns. On the other hands, the columns
	 * newly added below are device-only columns, so it will never
	 * referenced by the host-side. We mark it resjunk=true.
	 *
	 * Also note that any Var nodes in the device executable expression
	 * must be added with resjunk=true to solve the variable name.
	 */
	context.resjunk = true;
	build_device_tlist_walker((Node *)gj_info->outer_quals, &context);
	build_device_tlist_walker((Node *)gj_info->join_quals, &context);
	build_device_tlist_walker((Node *)gj_info->other_quals, &context);
	build_device_tlist_walker((Node *)gj_info->hash_inner_keys, &context);
	build_device_tlist_walker((Node *)gj_info->hash_outer_keys, &context);
	build_device_tlist_walker((Node *)targetlist, &context);

	Assert(list_length(context.ps_tlist) == list_length(context.ps_depth) &&
		   list_length(context.ps_tlist) == list_length(context.ps_resno));

	gj_info->ps_src_depth = context.ps_depth;
	gj_info->ps_src_resno = context.ps_resno;
	cscan->custom_scan_tlist = context.ps_tlist;
}

/*
 * PlanGpuJoinPath
 *
 * Entrypoint to create CustomScan(GpuJoin) node
 */
static Plan *
PlanGpuJoinPath(PlannerInfo *root,
				RelOptInfo *rel,
				CustomPath *best_path,
				List *tlist,
				List *clauses,
				List *custom_plans)
{
	GpuJoinPath	   *gjpath = (GpuJoinPath *) best_path;
	RelOptInfo	   *joinrel = gjpath->cpath.path.parent;
	Index			outer_relid = gjpath->outer_relid;
	GpuJoinInfo		gj_info;
	CustomScan	   *cscan;
	codegen_context	context;
	Plan		   *outer_plan;
	ListCell	   *lc;
	double			outer_nrows;
	int				i, j, k;

	Assert(gjpath->num_rels + 1 == list_length(custom_plans));
	outer_plan = linitial(custom_plans);

	cscan = makeNode(CustomScan);
	cscan->scan.plan.targetlist = tlist;
	cscan->scan.plan.qual = NIL;
	cscan->scan.scanrelid = outer_relid;
	cscan->flags = best_path->flags;
	cscan->methods = &gpujoin_plan_methods;
	cscan->custom_plans = list_copy_tail(custom_plans, 1);

	memset(&gj_info, 0, sizeof(GpuJoinInfo));
	gj_info.outer_ratio = 1.0;
	gj_info.outer_nrows = outer_plan->plan_rows;
	gj_info.outer_width = outer_plan->plan_width;
	gj_info.outer_startup_cost = outer_plan->startup_cost;
	gj_info.outer_total_cost = outer_plan->total_cost;
	gj_info.num_rels = gjpath->num_rels;

	if (!gjpath->sibling_param_id)
		gj_info.sibling_param_id = -1;
	else
	{
		cl_int	param_id = *gjpath->sibling_param_id;

		if (param_id < 0)
		{
			PlannerGlobal  *glob = root->glob;
#if PG_VERSION_NUM < 110000
			param_id = glob->nParamExec++;
#else
			param_id = list_length(glob->paramExecTypes);
			glob->paramExecTypes = lappend_oid(glob->paramExecTypes,
											   INTERNALOID);
#endif
			*gjpath->sibling_param_id = param_id;
		}
		gj_info.sibling_param_id = param_id;
	}

	outer_nrows = outer_plan->plan_rows;
	for (i=0; i < gjpath->num_rels; i++)
	{
		List	   *hash_inner_keys = NIL;
		List	   *hash_outer_keys = NIL;
		List	   *join_quals = NIL;
		List	   *other_quals = NIL;

		foreach (lc, gjpath->inners[i].hash_quals)
		{
			Path		   *scan_path = gjpath->inners[i].scan_path;
			RelOptInfo	   *scan_rel = scan_path->parent;
			RestrictInfo   *rinfo = lfirst(lc);
			OpExpr		   *op_clause = (OpExpr *) rinfo->clause;
			Node		   *arg1 = (Node *)linitial(op_clause->args);
			Node		   *arg2 = (Node *)lsecond(op_clause->args);
			Relids			relids1 = pull_varnos(arg1);
			Relids			relids2 = pull_varnos(arg2);

			/*
			 * NOTE: Both sides of hash-join operator may have different
			 * types if cross-type operators, like int48eq(x,y), are used.
			 * Hash functions are designed to generate same hash-value
			 * regardless of the data types, so we can run hash-join
			 * without implicit type casts.
			 */
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
		gj_info.plan_nrows_in = lappend(gj_info.plan_nrows_in,
										pmakeFloat(outer_nrows));
		gj_info.plan_nrows_out = lappend(gj_info.plan_nrows_out,
									pmakeFloat(gjpath->inners[i].join_nrows));
		gj_info.ichunk_size = lappend_int(gj_info.ichunk_size,
										  gjpath->inners[i].ichunk_size);
		gj_info.join_types = lappend_int(gj_info.join_types,
										 gjpath->inners[i].join_type);

		if (IS_OUTER_JOIN(gjpath->inners[i].join_type))
		{
			extract_actual_join_clauses(gjpath->inners[i].join_quals,
										best_path->path.parent->relids,
										&join_quals, &other_quals);
		}
		else
		{
			join_quals = extract_actual_clauses(gjpath->inners[i].join_quals,
												false);
			other_quals = NIL;
		}
		gj_info.join_quals = lappend(gj_info.join_quals, join_quals);
		gj_info.other_quals = lappend(gj_info.other_quals, other_quals);
		gj_info.hash_inner_keys = lappend(gj_info.hash_inner_keys,
										  hash_inner_keys);
		gj_info.hash_outer_keys = lappend(gj_info.hash_outer_keys,
										  hash_outer_keys);
		outer_nrows = gjpath->inners[i].join_nrows;
	}

	/*
	 * If outer-plan node is simple relation scan; SeqScan or GpuScan with
	 * device executable qualifiers, GpuJoin can handle the relation scan
	 * for better i/o performance. Elsewhere, call the child outer node.
	 */
	if (outer_relid)
	{
		RelOptInfo *baserel = root->simple_rel_array[outer_relid];
		Bitmapset  *referenced = NULL;
		List	   *outer_quals = gjpath->outer_quals;
		List	   *outer_refs = NULL;

		/* pick up outer referenced columns */
		pull_varattnos((Node *)outer_quals, outer_relid, &referenced);
		for (i=baserel->min_attr, j=0; i <= baserel->max_attr; i++, j++)
		{
			if (i < 0 || baserel->attr_needed[j] == NULL)
				continue;
			k = i - FirstLowInvalidHeapAttributeNumber;
			referenced = bms_add_member(referenced, k);
		}
		for (k = bms_next_member(referenced, -1);
			 k >= 0;
			 k = bms_next_member(referenced, k))
		{
			i = k + FirstLowInvalidHeapAttributeNumber;
			if (i >= 0)
				outer_refs = lappend_int(outer_refs, i);
		}

		/* BRIN-index stuff */
		if (gjpath->index_opt)
		{
			gj_info.index_oid = gjpath->index_opt->indexoid;
			gj_info.index_conds = gjpath->index_conds;
			gj_info.index_quals
				= extract_actual_clauses(gjpath->index_quals, false);
		}
		gj_info.outer_quals = outer_quals;
		gj_info.outer_refs = outer_refs;
	}
	else
	{
		outerPlan(cscan) = outer_plan;
		Assert(gjpath->outer_quals == NIL);
		Assert(gjpath->index_opt == NULL);
	}
	gj_info.outer_nrows_per_block = gjpath->outer_nrows_per_block;

	/*
	 * Build a tentative pseudo-scan targetlist. At this point, we cannot
	 * know which expression shall be applied on the final results, thus,
	 * all we can construct is a pseudo-scan targetlist that is consists
	 * of Var-nodes only.
	 */
	build_device_targetlist(root, gjpath, cscan, &gj_info,
							tlist, custom_plans);

	/*
	 * construct kernel code
	 */
	pgstrom_init_codegen_context(&context, root, joinrel);
	gj_info.optimal_gpu = gjpath->optimal_gpu;
	gj_info.kern_source = gpujoin_codegen(root,
										  cscan,
										  &gj_info,
										  tlist,
										  &context);
	gj_info.extra_flags = context.extra_flags | DEVKERNEL_NEEDS_GPUJOIN;
	gj_info.used_params = context.used_params;

	form_gpujoin_info(cscan, &gj_info);

	return &cscan->scan.plan;
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
fixup_varnode_to_origin(int depth, List *ps_src_depth, List *ps_src_resno,
						List *expr_list)
{
	fixup_varnode_to_origin_context	context;

	Assert(IsA(expr_list, List));
	context.depth = depth;
	context.ps_src_depth = ps_src_depth;
	context.ps_src_resno = ps_src_resno;

	return (List *) fixup_varnode_to_origin_mutator((Node *) expr_list,
													&context);
}

/*
 * assign_gpujoin_session_info
 *
 * Gives some definitions to the static portion of GpuJoin implementation
 */
void
assign_gpujoin_session_info(StringInfo buf, GpuTaskState *gts)
{
	GpuJoinState   *gjs = (GpuJoinState *) gts;

	Assert(gts->css.methods == &gpujoin_exec_methods);
	appendStringInfo(
		buf,
		"#define GPUJOIN_MAX_DEPTH %u\n",
		gjs->num_rels);
}

static Node *
gpujoin_create_scan_state(CustomScan *node)
{
	GpuJoinState   *gjs;
	cl_int			num_rels = list_length(node->custom_plans);

	gjs = MemoryContextAllocZero(CurTransactionContext,
								 offsetof(GpuJoinState,
										  inners[num_rels]));
	NodeSetTag(gjs, T_CustomScanState);
	gjs->gts.css.flags = node->flags;
	gjs->gts.css.methods = &gpujoin_exec_methods;

	return (Node *) gjs;
}

static void
ExecInitGpuJoin(CustomScanState *node, EState *estate, int eflags)
{
	GpuJoinState   *gjs = (GpuJoinState *) node;
	ScanState	   *ss = &gjs->gts.css.ss;
	CustomScan	   *cscan = (CustomScan *) node->ss.ps.plan;
	GpuJoinInfo	   *gj_info = deform_gpujoin_info(cscan);
	TupleTableSlot *result_slot = gjs->gts.css.ss.ps.ps_ResultTupleSlot;
	TupleDesc		result_tupdesc = result_slot->tts_tupleDescriptor;
	TupleDesc		scan_tupdesc;
	TupleDesc		junk_tupdesc;
	List		   *tlist_fallback = NIL;
	bool			fallback_needs_projection = false;
	bool			fallback_meets_resjunk = false;
	bool			explain_only = ((eflags & EXEC_FLAG_EXPLAIN_ONLY) != 0);
	ListCell	   *lc1;
	ListCell	   *lc2;
	cl_int			i, j, nattrs;
	StringInfoData	kern_define;
	ProgramId		program_id;

	/* activate a GpuContext for CUDA kernel execution */
	gjs->gts.gcontext = AllocGpuContext(gj_info->optimal_gpu,
										false, false, false);
	/*
	 * Re-initialization of scan tuple-descriptor and projection-info,
	 * because commit 1a8a4e5cde2b7755e11bde2ea7897bd650622d3e of
	 * PostgreSQL makes to assign result of ExecTypeFromTL() instead
	 * of ExecCleanTypeFromTL; that leads unnecessary projection.
	 * So, we try to remove junk attributes from the scan-descriptor.
	 *
	 * Also note that the supplied TupleDesc that contains junk attributes
	 * are still useful to run CPU fallback code. So, we keep this tuple-
	 * descriptor to initialize the related stuff.
	 */
	junk_tupdesc = gjs->gts.css.ss.ss_ScanTupleSlot->tts_tupleDescriptor;
	scan_tupdesc = ExecCleanTypeFromTL(cscan->custom_scan_tlist, false);
	ExecInitScanTupleSlot(estate, &gjs->gts.css.ss, scan_tupdesc);
	ExecAssignScanProjectionInfoWithVarno(&gjs->gts.css.ss, INDEX_VAR);

	/* Setup common GpuTaskState fields */
	pgstromInitGpuTaskState(&gjs->gts,
							gjs->gts.gcontext,
							GpuTaskKind_GpuJoin,
							gj_info->outer_refs,
							gj_info->used_params,
							gj_info->optimal_gpu,
							gj_info->outer_nrows_per_block,
							estate);
	gjs->gts.cb_next_tuple		= gpujoin_next_tuple;
	gjs->gts.cb_next_task		= gpujoin_next_task;
	gjs->gts.cb_terminator_task	= gpujoin_terminator_task;
	gjs->gts.cb_switch_task		= gpujoin_switch_task;
	gjs->gts.cb_process_task	= gpujoin_process_task;
	gjs->gts.cb_release_task	= gpujoin_release_task;

	/* DSM & GPU memory of inner buffer */
	gjs->m_kmrels = 0UL;
	gjs->m_kmrels_array = NULL;
	gjs->seg_kmrels = NULL;
	gjs->curr_outer_depth = -1;
	if (gj_info->sibling_param_id >= 0)
	{
		ParamExecData  *param
			= &(estate->es_param_exec_vals[gj_info->sibling_param_id]);
		if (param->value == 0UL)
		{
			GpuJoinSiblingState *sibling
				= palloc0(offsetof(GpuJoinSiblingState,
								   pergpu[numDevAttrs]));
			param->isnull = false;
			param->value = PointerGetDatum(sibling);
		}
		gjs->sibling = (GpuJoinSiblingState *)DatumGetPointer(param->value);
	}

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
	if (gj_info->outer_quals)
	{
#if PG_VERSION_NUM < 100000
		gjs->outer_quals = (List *)
			ExecInitExpr((Expr *)gj_info->outer_quals, &ss->ps);
#else
		gjs->outer_quals = ExecInitQual(gj_info->outer_quals, &ss->ps);
#endif
	}
	gjs->outer_ratio = gj_info->outer_ratio;
	gjs->outer_nrows = gj_info->outer_nrows;
	Assert(!cscan->scan.plan.qual);

	/*
	 * Init OUTER child node
	 */
	if (gjs->gts.css.ss.ss_currentRelation)
	{
		Relation	scan_rel = gjs->gts.css.ss.ss_currentRelation;

		pgstromExecInitBrinIndexMap(&gjs->gts,
									gj_info->index_oid,
									gj_info->index_conds,
									gj_info->index_quals);
		nattrs = RelationGetNumberOfAttributes(scan_rel);
	}
	else
	{
		TupleTableSlot *outer_slot;

		outerPlanState(gjs) = ExecInitNode(outerPlan(cscan), estate, eflags);
		outer_slot = outerPlanState(gjs)->ps_ResultTupleSlot;
		nattrs = outer_slot->tts_tupleDescriptor->natts;
		Assert(!OidIsValid(gj_info->index_oid));
	}

	/*
	 * Init CPU fallback stuff
	 */
	foreach (lc1, cscan->custom_scan_tlist)
	{
		TargetEntry	   *tle = lfirst(lc1);
		Var			   *var;

		/*
		 * NOTE: Var node inside of general expression shall reference
		 * the custom_scan_tlist recursively. Thus, we don't need to
		 * care about varno/varattno fixup here.
		 */
		Assert(IsA(tle, TargetEntry));

		/*
		 * Because ss_ScanTupleSlot does not contain junk attribute,
		 * we have to remove junk attribute by projection, if any of
		 * target-entry in custom_scan_tlist (that is tuple format to
		 * be constructed by CPU fallback) are junk.
		 */
		if (tle->resjunk)
		{
			fallback_needs_projection = true;
			fallback_meets_resjunk = true;
		}
		else
		{
			/* no valid attribute after junk attribute */
			if (fallback_meets_resjunk)
				elog(ERROR, "Bug? a valid attribute appear after junk ones");

			Assert(!fallback_meets_resjunk);

			if (IsA(tle->expr, Var))
			{
				tle = copyObject(tle);
				var = (Var *) tle->expr;
				var->varnoold	= var->varno;
				var->varoattno	= var->varattno;
				var->varno		= INDEX_VAR;
				var->varattno	= tle->resno;
			}
			else
			{
				/* also, non-simple Var node needs projection */
				fallback_needs_projection = true;
			}
#if PG_VERSION_NUM < 100000
			tlist_fallback = lappend(tlist_fallback,
									 ExecInitExpr((Expr *)tle, &ss->ps));
#else
			tlist_fallback = lappend(tlist_fallback, tle);
#endif
		}
	}

	if (fallback_needs_projection)
	{
		gjs->slot_fallback = MakeSingleTupleTableSlot(junk_tupdesc);
		gjs->proj_fallback = ExecBuildProjectionInfo(tlist_fallback,
													 ss->ps.ps_ExprContext,
													 ss->ss_ScanTupleSlot,
													 &ss->ps,
													 junk_tupdesc);
	}
	else
	{
		gjs->slot_fallback = ss->ss_ScanTupleSlot;
		gjs->proj_fallback = NULL;
	}
	ExecStoreAllNullTuple(gjs->slot_fallback);

	gjs->outer_src_anum_min = nattrs;
	gjs->outer_src_anum_max = FirstLowInvalidHeapAttributeNumber;
	nattrs -= FirstLowInvalidHeapAttributeNumber;
	gjs->outer_dst_resno = palloc0(sizeof(AttrNumber) * nattrs);
	j = 1;
	forboth (lc1, gj_info->ps_src_depth,
			 lc2, gj_info->ps_src_resno)
	{
		int		depth = lfirst_int(lc1);
		int		resno = lfirst_int(lc2);

		if (depth == 0)
		{
			if (gjs->outer_src_anum_min > resno)
				gjs->outer_src_anum_min = resno;
			if (gjs->outer_src_anum_max < resno)
				gjs->outer_src_anum_max = resno;
			resno -= FirstLowInvalidHeapAttributeNumber;
			Assert(resno > 0 && resno <= nattrs);
			gjs->outer_dst_resno[resno - 1] = j;
		}
		j++;
	}
	gjs->fallback_outer_index = -1;

	/*
	 * Init INNER child nodes for each depth
	 */
	for (i=0; i < gj_info->num_rels; i++)
	{
		Plan	   *inner_plan = list_nth(cscan->custom_plans, i);
		innerState *istate = &gjs->inners[i];
		List	   *join_quals;
		List	   *other_quals;
		List	   *hash_inner_keys;
		List	   *hash_outer_keys;
		TupleTableSlot *inner_slot;
		double		plan_nrows_in;
		double		plan_nrows_out;

		istate->state = ExecInitNode(inner_plan, estate, eflags);
		istate->econtext = CreateExprContext(estate);
		istate->depth = i + 1;
		plan_nrows_in = floatVal(list_nth(gj_info->plan_nrows_in, i));
		plan_nrows_out = floatVal(list_nth(gj_info->plan_nrows_out, i));
		istate->nrows_ratio = plan_nrows_out / Max(plan_nrows_in, 1.0);
		istate->ichunk_size = list_nth_int(gj_info->ichunk_size, i);
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
		join_quals = list_nth(gj_info->join_quals, i);
		if (join_quals)
		{
			Assert(IsA(join_quals, List));
#if PG_VERSION_NUM < 100000
			istate->join_quals = (List *)
				ExecInitExpr((Expr *)join_quals, &ss->ps);
#else
			istate->join_quals = ExecInitQual(join_quals, &ss->ps);
#endif
		}

		other_quals = list_nth(gj_info->other_quals, i);
		if (other_quals)
		{
#if PG_VERSION_NUM < 100000
			istate->other_quals = (List *)
				ExecInitExpr((Expr *)other_quals, &ss->ps);
#else
			istate->other_quals = ExecInitQual(other_quals, &ss->ps);
#endif
		}

		hash_inner_keys = list_nth(gj_info->hash_inner_keys, i);
		hash_outer_keys = list_nth(gj_info->hash_outer_keys, i);
		Assert(list_length(hash_inner_keys) == list_length(hash_outer_keys));
		if (hash_inner_keys != NIL && hash_outer_keys != NIL)
		{
			hash_inner_keys = fixup_varnode_to_origin(i+1,
													  gj_info->ps_src_depth,
													  gj_info->ps_src_resno,
													  hash_inner_keys);
			forboth (lc1, hash_inner_keys,
					 lc2, hash_outer_keys)
			{
				Expr	   *i_expr = lfirst(lc1);
				Expr	   *o_expr = lfirst(lc2);
				ExprState  *i_expr_state = ExecInitExpr(i_expr, &ss->ps);
				ExprState  *o_expr_state = ExecInitExpr(o_expr, &ss->ps);

				istate->hash_inner_keys =
					lappend(istate->hash_inner_keys, i_expr_state);
				istate->hash_outer_keys =
					lappend(istate->hash_outer_keys, o_expr_state);
			}
		}

		/*
		 * CPU fallback setup for INNER reference
		 */
		inner_slot = istate->state->ps_ResultTupleSlot;
		nattrs = inner_slot->tts_tupleDescriptor->natts;
		istate->inner_src_anum_min = nattrs;
		istate->inner_src_anum_max = FirstLowInvalidHeapAttributeNumber;
		nattrs -= FirstLowInvalidHeapAttributeNumber;
		istate->inner_dst_resno = palloc0(sizeof(AttrNumber) * nattrs);

		j = 1;
		forboth (lc1, gj_info->ps_src_depth,
				 lc2, gj_info->ps_src_resno)
		{
			int		depth = lfirst_int(lc1);
			int		resno = lfirst_int(lc2);

			if (depth == istate->depth)
			{
				if (istate->inner_src_anum_min > resno)
					istate->inner_src_anum_min = resno;
				if (istate->inner_src_anum_max < resno)
					istate->inner_src_anum_max = resno;
				resno -= FirstLowInvalidHeapAttributeNumber;
				Assert(resno > 0 && resno <= nattrs);
				istate->inner_dst_resno[resno - 1] = j;
			}
			j++;
		}
		/* add inner state as children of this custom-scan */
		gjs->gts.css.custom_ps = lappend(gjs->gts.css.custom_ps,
										 istate->state);
	}
	initStringInfo(&kern_define);
	pgstrom_build_session_info(&kern_define,
							   &gjs->gts,
							   gj_info->extra_flags);
	program_id = pgstrom_create_cuda_program(gjs->gts.gcontext,
											 gj_info->extra_flags,
											 gj_info->varlena_bufsz,
											 gj_info->kern_source,
											 kern_define.data,
											 false,
											 explain_only);
	gjs->gts.program_id = program_id;
	pfree(kern_define.data);

	/* expected kresults buffer expand rate */
	gjs->result_width =
		MAXALIGN(offsetof(HeapTupleHeaderData,
						  t_bits[BITMAPLEN(result_tupdesc->natts)]) +
				 (result_tupdesc->tdhasoid ? sizeof(Oid) : 0)) +
		MAXALIGN(cscan->scan.plan.plan_width);	/* average width */
}

/*
 * ExecReCheckGpuJoin
 *
 * Routine of EPQ recheck on GpuJoin. Join condition shall be checked on
 * the EPQ tuples.
 */
static bool
ExecReCheckGpuJoin(CustomScanState *node, TupleTableSlot *slot)
{
	/*
	 * TODO: Extract EPQ tuples on CPU fallback slot, then check
	 * join condition by CPU
	 */
	return true;
}

/*
 * ExecGpuJoin
 */
static TupleTableSlot *
ExecGpuJoin(CustomScanState *node)
{
	GpuJoinState *gjs = (GpuJoinState *) node;

	ActivateGpuContext(gjs->gts.gcontext);
	if (!GpuJoinInnerPreload(&gjs->gts, NULL))
		return NULL;
	return ExecScan(&node->ss,
					(ExecScanAccessMtd) pgstromExecGpuTaskState,
					(ExecScanRecheckMtd) ExecReCheckGpuJoin);
}

static void
ExecEndGpuJoin(CustomScanState *node)
{
	GpuJoinState *gjs = (GpuJoinState *) node;
	GpuTaskRuntimeStat *gt_rtstat = (GpuTaskRuntimeStat *)gjs->gj_rtstat;
	int		i;

	/* wait for completion of any asynchronous GpuTask */
	SynchronizeGpuContext(gjs->gts.gcontext);
	/* close index related stuff if any */
	pgstromExecEndBrinIndexMap(&gjs->gts);
	/* shutdown inner/outer subtree */
	ExecEndNode(outerPlanState(node));
	for (i=0; i < gjs->num_rels; i++)
		ExecEndNode(gjs->inners[i].state);
	/* then other private resources */
	GpuJoinInnerUnload(&gjs->gts, false);
	/* shutdown the common portion */
	pgstromReleaseGpuTaskState(&gjs->gts, gt_rtstat);
}

static void
ExecReScanGpuJoin(CustomScanState *node)
{
	GpuJoinState   *gjs = (GpuJoinState *) node;
	cl_int			i;

	/* wait for completion of any asynchronous GpuTask */
	SynchronizeGpuContext(gjs->gts.gcontext);
	/* rescan the outer sub-plan */
	if (outerPlanState(gjs))
		ExecReScan(outerPlanState(gjs));
	gjs->gts.scan_overflow = NULL;

	/*
	 * NOTE: ExecReScan() does not pay attention on the PlanState within
	 * custom_ps, so we need to assign its chgParam by ourself.
	 */
	if (gjs->gts.css.ss.ps.chgParam != NULL)
	{
		for (i=0; i < gjs->num_rels; i++)
		{
			innerState *istate = &gjs->inners[i];

			UpdateChangedParamSet(gjs->inners[i].state,
								  gjs->gts.css.ss.ps.chgParam);
			if (istate->state->chgParam != NULL)
				ExecReScan(istate->state);
		}
		/* rewind the inner hash/heap buffer */
		GpuJoinInnerUnload(&gjs->gts, true);
	}
	/* common rescan handling */
	pgstromRescanGpuTaskState(&gjs->gts);
}

static void
ExplainGpuJoin(CustomScanState *node, List *ancestors, ExplainState *es)
{
	GpuJoinState   *gjs = (GpuJoinState *) node;
	CustomScan	   *cscan = (CustomScan *) node->ss.ps.plan;
	GpuJoinInfo	   *gj_info = deform_gpujoin_info(cscan);
	GpuJoinRuntimeStat *gj_rtstat = gjs->gj_rtstat;
	List		   *dcontext;
	ListCell	   *lc1;
	ListCell	   *lc2;
	ListCell	   *lc3;
	ListCell	   *lc4;
	char		   *temp;
	char			qlabel[128];
	int				depth;
	StringInfoData	str;

	initStringInfo(&str);

	if (gj_rtstat)
		mergeGpuTaskRuntimeStat(&gjs->gts, &gj_rtstat->c);
	if (gjs->gts.css.ss.ps.instrument)
		memcpy(&gjs->gts.css.ss.ps.instrument->bufusage,
			   &gjs->gts.outer_instrument.bufusage,
			   sizeof(BufferUsage));

	/* deparse context */
	dcontext =  set_deparse_context_planstate(es->deparse_cxt,
											  (Node *) node,
											  ancestors);
	/* Device projection (verbose only) */
	if (es->verbose)
	{
		resetStringInfo(&str);
		foreach (lc1, cscan->custom_scan_tlist)
		{
			TargetEntry	   *tle = lfirst(lc1);

			if (tle->resjunk)
				continue;

			if (lc1 != list_head(cscan->custom_scan_tlist))
				appendStringInfo(&str, ", ");
			if (tle->resjunk)
				appendStringInfoChar(&str, '[');
			temp = deparse_expression((Node *)tle->expr,
									  dcontext, true, false);
			appendStringInfo(&str, "%s", temp);
			if (es->verbose)
			{
				temp = format_type_with_typemod(exprType((Node *)tle->expr),
												exprTypmod((Node *)tle->expr));
				appendStringInfo(&str, "::%s", temp);
			}
			if (tle->resjunk)
				appendStringInfoChar(&str, ']');
		}
		ExplainPropertyText("GPU Projection", str.data, es);
	}

	/* statistics for outer scan, if any */
	pgstromExplainOuterScan(&gjs->gts, dcontext, ancestors, es,
							gj_info->outer_quals,
                            gj_info->outer_startup_cost,
                            gj_info->outer_total_cost,
                            gj_info->outer_nrows,
                            gj_info->outer_width);
	/* join-qualifiers */
	depth = 1;
	forfour (lc1, gj_info->join_types,
			 lc2, gj_info->join_quals,
			 lc3, gj_info->other_quals,
			 lc4, gj_info->hash_outer_keys)
	{
		JoinType	join_type = (JoinType) lfirst_int(lc1);
		Expr	   *join_quals = lfirst(lc2);
		Expr	   *other_quals = lfirst(lc3);
		Expr	   *hash_outer_key = lfirst(lc4);
		innerState *istate = &gjs->inners[depth-1];
		kern_data_store *kds_in = NULL;
		int			indent_width;
		double		plan_nrows_in;
		double		plan_nrows_out;
		double		exec_nrows_in = 0.0;
		double		exec_nrows_out1 = 0.0;	/* by INNER JOIN */
		double		exec_nrows_out2 = 0.0;	/* by OUTER JOIN */

		if (gjs->seg_kmrels)
		{
			kern_multirels *kmrels = dsm_segment_address(gjs->seg_kmrels);
			kds_in = KERN_MULTIRELS_INNER_KDS(kmrels, depth);
		}

		/* fetch number of rows */
		plan_nrows_in = floatVal(list_nth(gj_info->plan_nrows_in, depth-1));
		plan_nrows_out = floatVal(list_nth(gj_info->plan_nrows_out, depth-1));
		if (gj_rtstat)
		{
			exec_nrows_in = (double)
				(pg_atomic_read_u64(&gj_rtstat->jstat[depth-1].inner_nitems) +
				 pg_atomic_read_u64(&gj_rtstat->jstat[depth-1].right_nitems));
			exec_nrows_out1 = (double)
				pg_atomic_read_u64(&gj_rtstat->jstat[depth].inner_nitems);
			exec_nrows_out2 = (double)
				pg_atomic_read_u64(&gj_rtstat->jstat[depth].right_nitems);
		}

		resetStringInfo(&str);
		if (hash_outer_key != NULL)
		{
			appendStringInfo(&str, "GpuHash%sJoin",
							 join_type == JOIN_FULL ? "Full" :
							 join_type == JOIN_LEFT ? "Left" :
							 join_type == JOIN_RIGHT ? "Right" : "");
		}
		else
		{
			appendStringInfo(&str, "GpuNestLoop%s",
							 join_type == JOIN_FULL ? "Full" :
							 join_type == JOIN_LEFT ? "Left" :
							 join_type == JOIN_RIGHT ? "Right" : "");
		}
		snprintf(qlabel, sizeof(qlabel), "Depth% 2d", depth);
		indent_width = es->indent * 2 + strlen(qlabel) + 2;

		if (es->format == EXPLAIN_FORMAT_TEXT)
		{
			if (es->costs)
			{
				if (!es->analyze)
				{
					appendStringInfo(&str, "  (%s-size: %s",
									 hash_outer_key ? "hash" : "heap",
									 format_bytesz(istate->ichunk_size));
				}
				else
				{
					Size	exec_sz = (kds_in ? kds_in->length : 0);

					appendStringInfo(&str, "  (%s-size: %s actual-size: %s",
									 hash_outer_key ? "hash" : "heap",
									 format_bytesz(istate->ichunk_size),
									 format_bytesz(exec_sz));
				}

				if (!gj_rtstat)
					appendStringInfo(&str, ", nrows %.0f...%.0f)",
									 plan_nrows_in,
									 plan_nrows_out);
				else if (exec_nrows_out2 > 0.0)
					appendStringInfo(&str, ", plan nrows: %.0f...%.0f,"
									 " actual nrows: %.0f...%.0f+%.0f)",
									 plan_nrows_in,
									 plan_nrows_out,
									 exec_nrows_in,
									 exec_nrows_out1,
									 exec_nrows_out2);
				else
					appendStringInfo(&str, ", plan nrows: %.0f...%.0f,"
									 " actual nrows: %.0f...%.0f)",
									 plan_nrows_in,
									 plan_nrows_out,
									 exec_nrows_in,
									 exec_nrows_out1);
			}
			ExplainPropertyText(qlabel, str.data, es);
		}
		else
		{
			ExplainPropertyText(qlabel, str.data, es);

			if (es->costs)
			{
				snprintf(qlabel, sizeof(qlabel),
						 "Depth % 2d KDS Plan Size", depth);
				ExplainPropertyInteger(qlabel, NULL, istate->ichunk_size, es);
				if (es->analyze)
				{
					Size	len = (kds_in ? kds_in->length : 0);

					snprintf(qlabel, sizeof(qlabel),
							 "Depth % 2d KDS Exec Size", depth);
					ExplainPropertyInteger(qlabel, NULL, len, es);
				}

				snprintf(qlabel, sizeof(qlabel),
						 "Depth% 2d Plan Rows-in", depth);
				ExplainPropertyFloat(qlabel, NULL, plan_nrows_in, 0, es);

				snprintf(qlabel, sizeof(qlabel),
						 "Depth% 2d Plan Rows-out", depth);
				ExplainPropertyFloat(qlabel, NULL, plan_nrows_out, 0, es);

				if (gj_rtstat)
				{
					snprintf(qlabel, sizeof(qlabel),
							 "Depth% 2d Actual Rows-in", depth);
					ExplainPropertyFloat(qlabel, NULL, exec_nrows_in, 0, es);

					snprintf(qlabel, sizeof(qlabel),
							 "Depth% 2d Actual Rows-out by inner join", depth);
					ExplainPropertyFloat(qlabel, NULL, exec_nrows_out1, 0, es);

					snprintf(qlabel, sizeof(qlabel),
							 "Depth% 2d Actual Rows-out by outer join", depth);
					ExplainPropertyFloat(qlabel, NULL, exec_nrows_out2, 0, es);
				}
			}
		}

		/*
		 * HashJoinKeys, if any
		 */
		if (hash_outer_key)
		{
			temp = deparse_expression((Node *)hash_outer_key,
                                      dcontext, true, false);
			if (es->format == EXPLAIN_FORMAT_TEXT)
			{
				appendStringInfoSpaces(es->str, indent_width);
				appendStringInfo(es->str, "HashKeys: %s\n", temp);
			}
			else
			{
				snprintf(qlabel, sizeof(qlabel),
						 "Depth% 2d HashKeys", depth);
				ExplainPropertyText(qlabel, temp, es);
			}
		}

		/*
		 * JoinQuals, if any
		 */
		if (join_quals)
		{
			temp = deparse_expression((Node *)join_quals, dcontext,
									  true, false);
			if (es->format == EXPLAIN_FORMAT_TEXT)
			{
				appendStringInfoSpaces(es->str, indent_width);
				appendStringInfo(es->str, "JoinQuals: %s\n", temp);
			}
			else
			{
				snprintf(qlabel, sizeof(qlabel),
						 "Depth% 2d JoinQuals", depth);
				ExplainPropertyText(qlabel, temp, es);
			}
		}

		/*
		 * OtherQuals, if any
		 */
		if (other_quals)
		{
			temp = deparse_expression((Node *)other_quals, dcontext,
									  es->verbose, false);
			if (es->format == EXPLAIN_FORMAT_TEXT)
			{
				appendStringInfoSpaces(es->str, indent_width);
				appendStringInfo(es->str, "JoinFilter: %s\n", temp);
			}
			else
			{
				snprintf(qlabel, sizeof(qlabel), "Depth %02d-Filter", depth);
				ExplainPropertyText(qlabel, str.data, es);
			}
		}
		depth++;
	}
	/* other common field */
	pgstromExplainGpuTaskState(&gjs->gts, es);
}

/*
 * OverWriteSharedStateIfPartitionLeaf
 */
static void
OverWriteSharedStateIfPartitionLeaf(GpuJoinState *gjs)
{
	GpuJoinSiblingState *sibling = gjs->sibling;

	if (sibling)
	{
		if (!sibling->gj_sstate)
			sibling->gj_sstate = gjs->gj_sstate;
		else
			gjs->gj_sstate = sibling->gj_sstate;
	}
}

/*
 * ExecGpuJoinEstimateDSM
 */
static Size
ExecGpuJoinEstimateDSM(CustomScanState *node,
					   ParallelContext *pcxt)
{
	GpuJoinState   *gjs = (GpuJoinState *) node;

	return MAXALIGN(offsetof(GpuJoinSharedState,
							 pergpu[numDevAttrs]))
		+ MAXALIGN(offsetof(GpuJoinRuntimeStat,
							jstat[gjs->num_rels + 1]))
		+ pgstromSizeOfBrinIndexMap((GpuTaskState *) node)
		+ pgstromEstimateDSMGpuTaskState((GpuTaskState *)node, pcxt);
}

/*
 * ExecGpuJoinInitDSM
 */
static void
ExecGpuJoinInitDSM(CustomScanState *node,
				   ParallelContext *pcxt,
				   void *coordinate)
{
	GpuJoinState   *gjs = (GpuJoinState *) node;

	/* save the ParallelContext */
	gjs->gts.pcxt = pcxt;
	/* setup shared-state and runtime-statistics */
	createGpuJoinSharedState(gjs, pcxt, coordinate);
	on_dsm_detach(pcxt->seg,
				  SynchronizeGpuContextOnDSMDetach,
				  PointerGetDatum(gjs->gts.gcontext));
	/* allocation of an empty multirel buffer */
	coordinate = (char *)coordinate + gjs->gj_sstate->ss_length;
	if (gjs->gts.outer_index_state)
	{
		gjs->gts.outer_index_map = (Bitmapset *)coordinate;
		gjs->gts.outer_index_map->nwords = -1;		/* uninitialized */
		coordinate = ((char *)coordinate +
					  pgstromSizeOfBrinIndexMap(&gjs->gts));
	}
	pgstromInitDSMGpuTaskState(&gjs->gts, pcxt, coordinate);
	OverWriteSharedStateIfPartitionLeaf(gjs);
}

/*
 * ExecGpuJoinInitWorker
 */
static void
ExecGpuJoinInitWorker(CustomScanState *node,
					  shm_toc *toc,
					  void *coordinate)
{
	GpuJoinState	   *gjs = (GpuJoinState *) node;
	GpuJoinSharedState *gj_sstate = (GpuJoinSharedState *) coordinate;

	gjs->gj_sstate = gj_sstate;
	gjs->gj_rtstat = GPUJOIN_RUNTIME_STAT(gjs->gj_sstate);
	/* ensure to stop workers prior to detach DSM */
	on_dsm_detach(dsm_find_mapping(gj_sstate->ss_handle),
				  SynchronizeGpuContextOnDSMDetach,
				  PointerGetDatum(gjs->gts.gcontext));
	coordinate = (char *)coordinate + gj_sstate->ss_length;
	if (gjs->gts.outer_index_state)
	{
		gjs->gts.outer_index_map = (Bitmapset *)coordinate;
		coordinate = ((char *)coordinate +
					  pgstromSizeOfBrinIndexMap(&gjs->gts));
	}
	pgstromInitWorkerGpuTaskState(&gjs->gts, coordinate);
	OverWriteSharedStateIfPartitionLeaf(gjs);
}

#if PG_VERSION_NUM >= 100000
/*
 * ExecGpuJoinReInitializeDSM
 */
static void
ExecGpuJoinReInitializeDSM(CustomScanState *node,
						   ParallelContext *pcxt, void *coordinate)
{
	pgstromReInitializeDSMGpuTaskState((GpuTaskState *) node);
}

/*
 * ExecShutdownGpuJoin
 *
 * DSM shall be released prior to Explain callback, so we have to save the
 * run-time statistics on the shutdown timing.
 */
static void
ExecShutdownGpuJoin(CustomScanState *node)
{
	GpuJoinState	   *gjs = (GpuJoinState *) node;
	GpuJoinRuntimeStat *gj_rtstat_old = gjs->gj_rtstat;
	GpuJoinRuntimeStat *gj_rtstat_new;

	/*
	 * If this GpuJoin node is located under the inner side of another
	 * GpuJoin, it should not be called under the background worker
	 * context, however, ExecShutdown walks down the node.
	 */
	if (!gj_rtstat_old)
		return;

	/* parallel worker put runtime-stat on ExecEnd handler */
	if (IsParallelWorker())
		mergeGpuTaskRuntimeStatParallelWorker(&gjs->gts, &gj_rtstat_old->c);
	else
	{
		EState	   *estate = gjs->gts.css.ss.ps.state;
		size_t		length = MAXALIGN(offsetof(GpuJoinRuntimeStat,
											   jstat[gjs->num_rels + 1]));
		gj_rtstat_new = MemoryContextAlloc(estate->es_query_cxt,
										   MAXALIGN(length));
		memcpy(gj_rtstat_new, gj_rtstat_old, length);
		gjs->gj_rtstat = gj_rtstat_new;
	}
}
#endif

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
	List	   *kern_vars = NIL;
	ListCell   *cell;
	int			depth;
	StringInfoData row;
	StringInfoData column;

	Assert(cur_depth > 0 && cur_depth <= gj_info->num_rels);
	initStringInfo(&row);
	initStringInfo(&column);

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
	 * parameter declaration
	 */
	pgstrom_codegen_param_declarations(source, context);

	/*
	 * variable declarations
	 */
	appendStringInfoString(
		source,
		"  HeapTupleHeaderData *htup  __attribute__((unused));\n"
		"  kern_data_store *kds_in    __attribute__((unused));\n"
		"  void *datum                __attribute__((unused));\n"
		"  cl_uint offset             __attribute__((unused));\n");

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
	appendStringInfoChar(source, '\n');

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
			/* close the previous block */
			if (depth >= 0)
				appendStringInfo(
					source,
					"%s%s  }\n",
					row.data,
					column.data);
			resetStringInfo(&row);
			resetStringInfo(&column);

			depth = keynode->varno;
			if (depth == 0)
			{
				appendStringInfoString(
					source,
					"  /* variable load in depth-0 (outer KDS) */\n"
					"  offset = (!o_buffer ? 0 : o_buffer[0]);\n"
					"  if (!kds)\n"
					"  {\n");
				appendStringInfoString(
					&row,
					"  }\n"
					"  else if (__ldg(&kds->format) != KDS_FORMAT_ARROW)\n"
					"  {\n"
					"    if (offset == 0)\n"
					"      htup = NULL;\n"
					"    else if (__ldg(&kds->format) == KDS_FORMAT_ROW)\n"
					"      htup = KDS_ROW_REF_HTUP(kds,offset,NULL,NULL);\n"
					"    else if (__ldg(&kds->format) == KDS_FORMAT_BLOCK)\n"
					"      htup = KDS_BLOCK_REF_HTUP(kds,offset,NULL,NULL);\n"
					"    else\n"
					"      htup = NULL; /* bug */\n");
				appendStringInfoString(
					&column,
					"  }\n"
					"  else\n"
					"  {\n"
					);
			}
			else if (depth < cur_depth)
			{
				appendStringInfo(
					&row,
					"  /* variable load in depth-%u (inner KDS) */\n"
					"  {\n"
					"    kds_in = KERN_MULTIRELS_INNER_KDS(kmrels, %u);\n"
					"    assert(__ldg(&kds_in->format) == %s);\n"
					"    if (!o_buffer)\n"
					"      htup = NULL;\n"
					"    else\n"
					"      htup = KDS_ROW_REF_HTUP(kds_in,o_buffer[%d],\n"
					"                              NULL, NULL);\n",
					depth,
					depth,
					list_nth(gj_info->hash_outer_keys,
							 keynode->varno - 1) == NIL
					? "KDS_FORMAT_ROW"
					: "KDS_FORMAT_HASH",
					depth);
			}
			else if (depth == cur_depth)
			{
				appendStringInfo(
					&row,
					"  /* variable load in depth-%u (inner KDS) */\n"
					"  {\n"
					"    kds_in = KERN_MULTIRELS_INNER_KDS(kmrels, %u);\n"
					"    assert(__ldg(&kds_in->format) == %s);\n"
					"    htup = i_htup;\n",
					depth,
					depth,
					list_nth(gj_info->hash_outer_keys,
							 keynode->varno - 1) == NIL
					? "KDS_FORMAT_ROW"
					: "KDS_FORMAT_HASH");
			}
			else
				elog(ERROR, "Bug? variables reference too deep");
		}
		/* RIGHT OUTER may have kds==NULL */
		if (depth == 0)
			appendStringInfo(
				source,
				"    pg_datum_ref(kcxt,KVAR_%u,NULL); //pg_%s_t\n",
				keynode->varoattno,
				dtype->type_name);

		appendStringInfo(
			&row,
			"    datum = GPUJOIN_REF_DATUM(%s->colmeta,htup,%u);\n"
			"    pg_datum_ref(kcxt,KVAR_%u,datum); //pg_%s_t\n",
			(depth == 0 ? "kds" : "kds_in"),
			keynode->varattno - 1,
			keynode->varoattno, dtype->type_name);

		/* KDS_FORMAT_COLUMN only if depth == 0 */
		if (depth == 0)
			appendStringInfo(
				&column,
				"    if (offset > 0)\n"
				"      pg_datum_ref_arrow(kcxt,KVAR_%u,kds,%u,offset-1);\n"
				"    else\n"
				"      pg_datum_ref(kcxt,KVAR_%u,NULL);\n",
				keynode->varoattno, keynode->varattno - 1,
				keynode->varoattno);
	}

	/* close the previous block */
	if (depth >= 0)
	{
		appendStringInfo(
			source,
			"%s%s  }\n",
			row.data,
			column.data);
	}
	pfree(row.data);
	pfree(column.data);

	appendStringInfo(source, "\n");
}

/*
 * codegen for:
 * STATIC_FUNCTION(cl_bool)
 * gpujoin_join_quals_depth%u(kern_context *kcxt,
 *                            kern_data_store *kds,
 *                            kern_multirels *kmrels,
 *                            cl_int *o_buffer,
 *                            HeapTupleHeaderData *i_htup,
 *                            cl_bool *joinquals_matched)
 */
static void
gpujoin_codegen_join_quals(StringInfo source,
						   GpuJoinInfo *gj_info,
						   int cur_depth,
						   codegen_context *context)
{
	List	   *join_quals;
	List	   *other_quals;
	char	   *join_quals_code = NULL;
	char	   *other_quals_code = NULL;

	Assert(cur_depth > 0 && cur_depth <= gj_info->num_rels);
	join_quals = list_nth(gj_info->join_quals, cur_depth - 1);
	other_quals = list_nth(gj_info->other_quals, cur_depth - 1);

	/*
	 * make a text representation of join_qual
	 */
	context->used_vars = NIL;
	context->param_refs = NULL;
	resetStringInfo(&context->decl_temp);
	if (join_quals != NIL)
		join_quals_code = pgstrom_codegen_expression((Node *)join_quals,
													 context);
	if (other_quals != NIL)
		other_quals_code = pgstrom_codegen_expression((Node *)other_quals,
													  context);
	/*
	 * function declaration
	 */
	appendStringInfo(
		source,
		"DEVICE_FUNCTION(cl_bool)\n"
		"gpujoin_join_quals_depth%d(kern_context *kcxt,\n"
		"                          kern_data_store *kds,\n"
        "                          kern_multirels *kmrels,\n"
		"                          cl_uint *o_buffer,\n"
		"                          HeapTupleHeaderData *i_htup,\n"
		"                          cl_bool *joinquals_matched)\n"
		"{\n%s",
		cur_depth, context->decl_temp.data);

	/*
	 * variable/params declaration & initialization
	 */
	gpujoin_codegen_var_param_decl(source, gj_info,
								   cur_depth, context);

	/*
	 * evaluation of other-quals and join-quals
	 */
	if (join_quals_code != NULL)
	{
		appendStringInfo(
			source,
			"  if (i_htup && o_buffer && !EVAL(%s))\n"
			"  {\n"
			"    if (joinquals_matched)\n"
			"      *joinquals_matched = false;\n"
			"    return false;\n"
			"  }\n",
			join_quals_code);
	}
	appendStringInfo(
		source,
		"  if (joinquals_matched)\n"
		"    *joinquals_matched = true;\n");
	if (other_quals_code != NULL)
	{
		appendStringInfo(
			source,
			"  if (!EVAL(%s))\n"
			"    return false;\n",
			other_quals_code);
	}
	appendStringInfo(
		source,
		"  return true;\n"
		"}\n"
		"\n");
}

/*
 * codegen for:
 * STATIC_FUNCTION(cl_uint)
 * gpujoin_hash_value_depth%u(kern_context *kcxt,
 *                            kern_data_store *kds,
 *                            kern_multirels *kmrels,
 *                            cl_int *o_buffer,
 *                            cl_bool *is_null_keys)
 */
static void
gpujoin_codegen_hash_value(StringInfo source,
						   GpuJoinInfo *gj_info,
						   int cur_depth,
						   codegen_context *context)
{
	StringInfoData	decl;
	StringInfoData	body;
	List		   *hash_outer_keys;
	List		   *type_oid_list = NIL;
	ListCell	   *lc;

	Assert(cur_depth > 0 && cur_depth <= gj_info->num_rels);
	hash_outer_keys = list_nth(gj_info->hash_outer_keys, cur_depth - 1);
	Assert(hash_outer_keys != NIL);

	initStringInfo(&decl);
	initStringInfo(&body);

	appendStringInfo(
		&decl,
		"  cl_uint hash = 0xffffffffU;\n"
		"  cl_bool is_null_keys = true;\n");

	context->used_vars = NIL;
	context->param_refs = NULL;
	resetStringInfo(&context->decl_temp);
	foreach (lc, hash_outer_keys)
	{
		Node	   *key_expr = lfirst(lc);
		Oid			key_type = exprType(key_expr);
		devtype_info *dtype;

		dtype = pgstrom_devtype_lookup(key_type);
		if (!dtype)
			elog(ERROR, "Bug? device type \"%s\" not found",
                 format_type_be(key_type));
		appendStringInfo(
			&body,
			"  temp.%s_v = %s;\n"
			"  if (!temp.%s_v.isnull)\n"
			"    is_null_keys = false;\n"
			"  hash ^= pg_comp_hash(kcxt, temp.%s_v);\n",
			dtype->type_name,
			pgstrom_codegen_expression(key_expr, context),
			dtype->type_name,
			dtype->type_name);
		type_oid_list = list_append_unique_oid(type_oid_list,
											   dtype->type_oid);
	}

	/*
	 * variable/params declaration & initialization
	 */
	pgstrom_union_type_declarations(&decl, "temp", type_oid_list);
	gpujoin_codegen_var_param_decl(&decl, gj_info,
								   cur_depth, context);
	appendStringInfo(
		source,
		"STATIC_FUNCTION(cl_uint)\n"
		"gpujoin_hash_value_depth%u(kern_context *kcxt,\n"
		"                          kern_data_store *kds,\n"
		"                          kern_multirels *kmrels,\n"
		"                          cl_uint *o_buffer,\n"
		"                          cl_bool *p_is_null_keys)\n"
		"{\n"
		"%s%s%s"
		"  *p_is_null_keys = is_null_keys;\n"
		"  hash ^= 0xffffffff;\n"
		"  return hash;\n"
		"}\n"
		"\n",
		cur_depth,
		decl.data,
		context->decl_temp.data,
		body.data);
	pfree(decl.data);
	pfree(body.data);
}

/*
 * gpujoin_codegen_projection
 *
 * It makes a device function for device projection.
 */
static void
gpujoin_codegen_projection(StringInfo source,
						   CustomScan *cscan,
						   GpuJoinInfo *gj_info,
						   codegen_context *context)
{
	List		   *tlist_dev = cscan->custom_scan_tlist;
	List		   *ps_src_depth = gj_info->ps_src_depth;
	List		   *ps_src_resno = gj_info->ps_src_resno;
	ListCell	   *lc1;
	ListCell	   *lc2;
	ListCell	   *lc3;
	AttrNumber	   *varattmaps;
	Bitmapset	   *refs_by_vars = NULL;
	Bitmapset	   *refs_by_expr = NULL;
	List		   *type_oid_list = NIL;
	devtype_info   *dtype;
	StringInfoData	decl;
	StringInfoData	body;
	StringInfoData	temp;
	StringInfoData	row;
	StringInfoData	column;
	StringInfoData	outer;
	cl_int			nfields = list_length(tlist_dev);
	cl_int			depth;
	cl_bool			is_first;

	varattmaps = palloc(sizeof(AttrNumber) * list_length(tlist_dev));
	initStringInfo(&decl);
	initStringInfo(&body);
	initStringInfo(&temp);
	initStringInfo(&row);
	initStringInfo(&column);
	initStringInfo(&outer);

	context->used_vars = NIL;
	context->param_refs = NULL;
	resetStringInfo(&context->decl_temp);

	/* expand varlena_bufsz for tup_dclass/values/extra array */
	context->varlena_bufsz += (MAXALIGN(sizeof(cl_char) * nfields) +
							   MAXALIGN(sizeof(Datum)   * nfields) +
							   MAXALIGN(sizeof(cl_uint) * nfields));

	/*
	 * Pick up all the var-node referenced directly or indirectly by
	 * device expressions; which are resjunk==false.
	 */
	forthree (lc1, tlist_dev,
			  lc2, ps_src_depth,
			  lc3, ps_src_resno)
	{
		TargetEntry	*tle = lfirst(lc1);
		cl_int		src_depth = lfirst_int(lc2);

		if (tle->resjunk)
			continue;
		if (src_depth >= 0)
		{
			refs_by_vars = bms_add_member(refs_by_vars, tle->resno -
										  FirstLowInvalidHeapAttributeNumber);
		}
		else
		{
			List	   *expr_vars = pull_vars_of_level((Node *)tle->expr, 0);
			ListCell   *cell;

			foreach (cell, expr_vars)
			{
				TargetEntry	   *__tle = tlist_member(lfirst(cell), tlist_dev);

				if (!__tle)
					elog(ERROR, "Bug? no indirectly referenced Var-node exists in custom_scan_tlist");
				refs_by_expr = bms_add_member(refs_by_expr, __tle->resno -
										FirstLowInvalidHeapAttributeNumber);
			}
			list_free(expr_vars);
		}
	}

	appendStringInfoString(
		&body,
		"  if (tup_extras)\n"
		"    memset(tup_extras, 0, sizeof(cl_uint) * kds_dst->ncols);\n");

	for (depth=0; depth <= gj_info->num_rels; depth++)
	{
		List	   *kvars_srcnum = NIL;
		List	   *kvars_dstnum = NIL;
		const char *kds_label;
		cl_int		i, nattrs = -1;
		bool		sysattr_refs = false;

		resetStringInfo(&row);
		resetStringInfo(&column);
		resetStringInfo(&outer);

		/* collect information in this depth */
		memset(varattmaps, 0, sizeof(AttrNumber) * list_length(tlist_dev));

		forthree (lc1, tlist_dev,
				  lc2, ps_src_depth,
				  lc3, ps_src_resno)
		{
			TargetEntry *tle = lfirst(lc1);
			cl_int		src_depth = lfirst_int(lc2);
			cl_int		src_resno = lfirst_int(lc3);
			cl_int		k = tle->resno - FirstLowInvalidHeapAttributeNumber;

			if (depth != src_depth)
				continue;
			if (src_resno < 0)
			{
				if (depth != 0)
					elog(ERROR, "Bug? sysattr reference at inner table");
				sysattr_refs = true;
			}
			if (bms_is_member(k, refs_by_vars))
				varattmaps[tle->resno - 1] = src_resno;
			if (bms_is_member(k, refs_by_expr))
			{
				kvars_srcnum = lappend_int(kvars_srcnum, src_resno);
				kvars_dstnum = lappend_int(kvars_dstnum, tle->resno);
			}
			if (bms_is_member(k, refs_by_vars) ||
				bms_is_member(k, refs_by_expr))
				nattrs = Max(nattrs, src_resno);
		}

		/* no need to extract inner/outer tuple in this depth */
		if (nattrs < 1 && !sysattr_refs)
			continue;

		if (depth == 0)
			kds_label = "kds_src";
		else
			kds_label = "kds_in";

		/* System column reference if any */
		foreach (lc1, tlist_dev)
		{
			TargetEntry		   *tle = lfirst(lc1);
			Form_pg_attribute	attr;

			if (varattmaps[tle->resno-1] >= 0)
				continue;
			attr = SystemAttributeDefinition(varattmaps[tle->resno-1], true);
			appendStringInfo(
				&row,
				"  sz = pg_sysattr_%s_store(kcxt,%s,htup,&t_self,\n"
				"                           tup_dclass[%d],\n"
				"                           tup_values[%d]);\n"
				"  if (tup_extras)\n"
				"    tup_extras[%d] = sz;\n",
				NameStr(attr->attname),
				kds_label,
				tle->resno - 1,
				tle->resno - 1,
				tle->resno - 1);
			appendStringInfo(
				&column,
				"    sz = pg_sysattr_%s_store(kcxt,(offset == 0 ? NULL : %s),\n"
				"                             NULL, NULL,\n"
				"                             tup_dclass[%d],\n"
				"                             tup_values[%d]);\n"
				"    if (tup_extras)\n"
				"      tup_extras[%d] = sz;\n",
				NameStr(attr->attname),
				kds_label,
				tle->resno - 1,
				tle->resno - 1,
				tle->resno - 1);
			/* NULL-initialization for LEFT OUTER JOIN */
			appendStringInfo(
				&outer,
				"  tup_dclass[%d] = DATUM_CLASS__NULL;\n",
				tle->resno - 1);
		}
		/* begin to walk on the tuple */
		if (nattrs > 0)
			appendStringInfo(
				&row,
				"    EXTRACT_HEAP_TUPLE_BEGIN(addr, %s, htup);\n",
				kds_label);
		resetStringInfo(&temp);
		for (i=1; i <= nattrs; i++)
		{
			TargetEntry	   *tle;
			Oid				type_oid;
			int16			typelen;
			bool			typebyval;
			cl_bool			referenced = false;

			foreach (lc1, tlist_dev)
			{
				tle = lfirst(lc1);

				if (varattmaps[tle->resno - 1] != i)
					continue;
				/*
				 * attributes can be directly copied regardless of device-
				 * type support (if row format).
				 */
				type_oid = exprType((Node *)tle->expr);
				get_typlenbyval(type_oid, &typelen, &typebyval);
				dtype = pgstrom_devtype_lookup_and_track(type_oid, context);

				/* row */
				if (typebyval)
				{
					appendStringInfo(
						&temp,
						"    EXTRACT_HEAP_READ_%dBIT(addr,tup_dclass[%d],tup_values[%d]);\n",
						8 * typelen, tle->resno - 1, tle->resno - 1);
				}
				else
				{
					appendStringInfo(
						&temp,
						"    EXTRACT_HEAP_READ_POINTER(addr,tup_dclass[%d],tup_values[%d]);\n",
						tle->resno - 1, tle->resno - 1);
				}
				/* column */
				if (!dtype)
				{
					appendStringInfo(
						&column,
						"    tup_dclass[%d] = DATUM_CLASS__NULL;\n",
						tle->resno - 1);
				}
				else
				{
					appendStringInfo(
						&column,
						"    if (offset > 0)\n"
						"      pg_datum_ref_arrow(kcxt,temp.%s_v,%s,%d,offset-1);\n"
						"    else\n"
						"      pg_datum_ref(kcxt,temp.%s_v,NULL);\n"
						"    sz = pg_datum_store(kcxt, temp.%s_v,\n"
						"                        tup_dclass[%d],\n"
						"                        tup_values[%d]);\n"
						"    if (tup_extras)\n"
						"      tup_extras[%d] = sz;\n"
						"    extra_sum += MAXALIGN(sz);\n",
						dtype->type_name, kds_label, i-1,
						dtype->type_name,
						dtype->type_name,
						tle->resno - 1,
						tle->resno - 1,
						tle->resno - 1);
					context->varlena_bufsz += MAXALIGN(dtype->extra_sz);
					type_oid_list = list_append_unique_oid(type_oid_list,
														   dtype->type_oid);
				}
				/* NULL-initialization for LEFT OUTER JOIN */
				appendStringInfo(
					&outer,
					"    tup_dclass[%d] = DATUM_CLASS__NULL;\n",
					tle->resno - 1);

				referenced = true;
			}

			forboth (lc1, kvars_srcnum,
					 lc2, kvars_dstnum)
			{
				devtype_info   *dtype;
				cl_int			src_num = lfirst_int(lc1);
				cl_int			dst_num = lfirst_int(lc2);
				Oid				type_oid;

				if (src_num != i)
					continue;
				/* add KVAR_%u declarations */
				tle = list_nth(tlist_dev, dst_num - 1);
				type_oid = exprType((Node *)tle->expr);
				dtype = pgstrom_devtype_lookup(type_oid);
				if (!dtype)
					elog(ERROR, "cache lookup failed for device type: %s",
						 format_type_be(type_oid));

				appendStringInfo(
					&decl,
					"  pg_%s_t KVAR_%u;\n",
					dtype->type_name,
					dst_num);
				appendStringInfo(
					&temp,
					"    pg_datum_ref(kcxt, KVAR_%u, addr);\n", dst_num);
				if (!referenced)
				{
					appendStringInfo(
						&column,
						"    if (offset > 0)\n"
						"      pg_datum_ref_arrow(kcxt,temp.%s_v,%s,%d,offset-1);\n"
						"    else\n"
						"      pg_datum_ref(kcxt,temp.%s_v,NULL);\n",
						dtype->type_name, kds_label, i-1,
						dtype->type_name);
					context->varlena_bufsz += MAXALIGN(dtype->extra_sz);
					type_oid_list = list_append_unique_oid(type_oid_list,
														   dtype->type_oid);
				}
				else
				{
					appendStringInfo(
						&column,
						"    KVAR_%u = temp.%s_v;\n",
						dst_num, dtype->type_name);
				}
				referenced = true;
			}

			/* flush to the main buffer */
			if (referenced)
			{
				appendStringInfoString(&row, temp.data);
				resetStringInfo(&temp);
			}
			appendStringInfoString(
				&temp,
				"    EXTRACT_HEAP_TUPLE_NEXT(addr);\n");
		}
		if (nattrs > 0)
			appendStringInfoString(
				&row,
				"    EXTRACT_HEAP_TUPLE_END();\n");

		if (depth == 0)
		{
			appendStringInfo(
				&body,
				"  /* ------ extract outer relation ------ */\n"
				"  if (!kds_src)\n"
				"  {\n"
				"%s"
				"  }\n"
				"  else if (kds_src->format == KDS_FORMAT_ROW ||\n"
				"           kds_src->format == KDS_FORMAT_BLOCK)\n"
				"  {\n"
				"    offset = r_buffer[0];\n"
				"    if (offset == 0)\n"
				"      htup = NULL;\n"
				"    else if (kds_src->format == KDS_FORMAT_ROW)\n"
				"      htup = KDS_ROW_REF_HTUP(kds_src,offset,&t_self,NULL);\n"
				"    else\n"
				"      htup = KDS_BLOCK_REF_HTUP(kds_src,offset,&t_self,NULL);\n"
				"%s"
				"  }\n"
				"  else\n"
				"  {\n"
				"    offset = r_buffer[0];\n"
				"%s"
				"  }\n",
				outer.data,
				row.data,
				column.data);
		}
		else
		{
			appendStringInfo(
				&body,
				"  /* ---- extract inner relation (depth=%d) */\n"
				"  kds_in = KERN_MULTIRELS_INNER_KDS(kmrels, %d);\n"
				"  assert(kds_in->format == KDS_FORMAT_ROW ||\n"
				"         kds_in->format == KDS_FORMAT_HASH);\n"
				"  offset = r_buffer[%d];\n"
				"  if (offset == 0)\n"
				"  {\n"
				"%s"
				"  }\n"
				"  else\n"
				"  {\n"
				"    htup = KDS_ROW_REF_HTUP(kds_in,offset,&t_self,NULL);\n"
				"%s"
				"  }\n",
				depth, depth, depth,
				outer.data,
				row.data);
		}
	}

	/*
	 * Execution of the expression
	 */
	is_first = true;
	forboth (lc1, tlist_dev,
			 lc2, ps_src_depth)
	{
		TargetEntry	   *tle = lfirst(lc1);
		cl_int			src_depth = lfirst_int(lc2);
		devtype_info   *dtype;

		if (tle->resjunk || src_depth >= 0)
			continue;

		if (is_first)
		{
			appendStringInfoString(
				&body,
				"\n"
				"  /* calculation of expressions */\n");
			is_first = false;
		}

		dtype = pgstrom_devtype_lookup(exprType((Node *) tle->expr));
		if (!dtype)
			elog(ERROR, "cache lookup failed for device type: %s",
				 format_type_be(exprType((Node *) tle->expr)));
		appendStringInfo(
			&body,
			"  temp.%s_v = %s;\n"
			"  sz = pg_datum_store(kcxt, temp.%s_v,\n"
			"                      tup_dclass[%d],\n"
			"                      tup_values[%d]);\n"
			"  if (tup_extras)\n"
			"    tup_extras[%d] = sz;\n"
			"  extra_sum += MAXALIGN(sz);\n\n",
			dtype->type_name,
			pgstrom_codegen_expression((Node *)tle->expr, context),
			dtype->type_name,
			tle->resno - 1,
			tle->resno - 1,
			tle->resno - 1);
		context->varlena_bufsz += MAXALIGN(dtype->extra_sz);
		type_oid_list = list_append_unique_oid(type_oid_list,
											   dtype->type_oid);
	}
	/* add const/param and temporary declarations */
	pgstrom_codegen_param_declarations(&decl, context);
	pgstrom_union_type_declarations(&decl, "temp", type_oid_list);
	/* merge declarations and function body */
	appendStringInfo(
		source,
		"DEVICE_FUNCTION(cl_uint)\n"
		"gpujoin_projection(kern_context *kcxt,\n"
		"                   kern_data_store *kds_src,\n"
		"                   kern_multirels *kmrels,\n"
		"                   cl_uint *r_buffer,\n"
		"                   kern_data_store *kds_dst,\n"
		"                   cl_char *tup_dclass,\n"
		"                   Datum   *tup_values,\n"
		"                   cl_uint *tup_extras)\n"
		"{\n"
		"  HeapTupleHeaderData *htup    __attribute__((unused));\n"
		"  kern_data_store *kds_in      __attribute__((unused));\n"
		"  ItemPointerData  t_self      __attribute__((unused));\n"
		"  cl_uint          offset      __attribute__((unused));\n"
		"  cl_uint          sz          __attribute__((unused));\n"
		"  cl_uint          extra_sum = 0;\n"
		"  void            *addr        __attribute__((unused)) = NULL;\n"
		"%s%s\n%s"
		"  return extra_sum;\n"
		"}\n",
		decl.data, context->decl_temp.data,
		body.data);

	pfree(decl.data);
	pfree(body.data);
	pfree(temp.data);
	pfree(row.data);
	pfree(column.data);
}

static char *
gpujoin_codegen(PlannerInfo *root,
				CustomScan *cscan,
				GpuJoinInfo *gj_info,
				List *tlist,
				codegen_context *context)
{
	StringInfoData source;
	int			depth;
	size_t		varlena_bufsz;
	ListCell   *cell;

	initStringInfo(&source);

	/*
	 * gpuscan_quals_eval
	 */
	codegen_gpuscan_quals(&source,
						  context,
						  "gpujoin",
						  cscan->scan.scanrelid,
						  gj_info->outer_quals);
	varlena_bufsz = context->varlena_bufsz;

	/*
	 * gpujoin_join_quals
	 */
	context->pseudo_tlist = cscan->custom_scan_tlist;
	for (depth=1; depth <= gj_info->num_rels; depth++)
	{
		context->varlena_bufsz = 0;
		gpujoin_codegen_join_quals(&source, gj_info, depth, context);
		varlena_bufsz = Max(varlena_bufsz, context->varlena_bufsz);
	}

	appendStringInfo(
		&source,
		"DEVICE_FUNCTION(cl_bool)\n"
		"gpujoin_join_quals(kern_context *kcxt,\n"
		"                   kern_data_store *kds,\n"
		"                   kern_multirels *kmrels,\n"
		"                   int depth,\n"
		"                   cl_uint *o_buffer,\n"
		"                   HeapTupleHeaderData *i_htup,\n"
		"                   cl_bool *needs_outer_row)\n"
		"{\n"
		"  switch (depth)\n"
		"  {\n");

	for (depth=1; depth <= gj_info->num_rels; depth++)
	{
		appendStringInfo(
			&source,
			"  case %d:\n"
			"    return gpujoin_join_quals_depth%d(kcxt, kds, kmrels,\n"
			"                                     o_buffer, i_htup,\n"
			"                                     needs_outer_row);\n",
			depth, depth);
	}
	appendStringInfo(
		&source,
		"  default:\n"
		"    STROM_EREPORT(kcxt, ERRCODE_STROM_WRONG_CODE_GENERATION,\n"
		"                  \"GpuJoin: wrong code generation\");\n"
		"    break;\n"
		"  }\n"
		"  return false;\n"
		"}\n\n");


	depth = 1;
	foreach (cell, gj_info->hash_outer_keys)
	{
		if (lfirst(cell) != NULL)
		{
			context->varlena_bufsz = 0;
			gpujoin_codegen_hash_value(&source, gj_info, depth, context);
			varlena_bufsz = Max(varlena_bufsz, context->varlena_bufsz);
		}
		depth++;
	}

	/*
	 * gpujoin_hash_value
	 */
	appendStringInfo(
		&source,
		"DEVICE_FUNCTION(cl_uint)\n"
		"gpujoin_hash_value(kern_context *kcxt,\n"
		"                   kern_data_store *kds,\n"
		"                   kern_multirels *kmrels,\n"
		"                   cl_int depth,\n"
		"                   cl_uint *o_buffer,\n"
		"                   cl_bool *is_null_keys)\n"
		"{\n"
		"  switch (depth)\n"
		"  {\n");
	depth = 1;
	foreach (cell, gj_info->hash_outer_keys)
	{
		if (lfirst(cell) != NULL)
		{
			appendStringInfo(
				&source,
				"  case %u:\n"
				"    return gpujoin_hash_value_depth%u(kcxt,kds,kmrels,\n"
				"                                      o_buffer,is_null_keys);\n",
				depth, depth);
		}
		depth++;
	}
	appendStringInfo(
		&source,
		"  default:\n"
		"    STROM_EREPORT(kcxt, ERRCODE_STROM_WRONG_CODE_GENERATION,\n"
		"                  \"GpuJoin: wrong code generation\");\n"
		"    break;\n"
		"  }\n"
		"  return (cl_uint)(-1);\n"
		"}\n"
		"\n");

	/*
	 * gpujoin_projection
	 */
	context->varlena_bufsz = 0;
	gpujoin_codegen_projection(&source, cscan, gj_info, context);
	varlena_bufsz = Max(varlena_bufsz, context->varlena_bufsz);

	/* required varlena buffer size */
	gj_info->varlena_bufsz = varlena_bufsz;

	return source.data;
}

/*
 * GpuJoinSetupTask
 */
Size
GpuJoinSetupTask(struct kern_gpujoin *kgjoin, GpuTaskState *gts,
				 pgstrom_data_store *pds_src)
{
	GpuJoinState *gjs = (GpuJoinState *) gts;
	GpuContext *gcontext = gjs->gts.gcontext;
	cl_int		nrels = gjs->num_rels;
	size_t		head_sz;
	size_t		param_sz;
	size_t		pstack_sz;
	size_t		pstack_nrooms;
	size_t		suspend_sz;
	int			mp_count;

	mp_count = (GPUKERNEL_MAX_SM_MULTIPLICITY *
				devAttrs[gcontext->cuda_dindex].MULTIPROCESSOR_COUNT);
	head_sz = STROMALIGN(offsetof(kern_gpujoin,
								  stat_nitems[gjs->num_rels + 1]));
	param_sz = STROMALIGN(gjs->gts.kern_params->length);
	pstack_nrooms = 2048;
	pstack_sz = MAXALIGN(sizeof(cl_uint) *
						 pstack_nrooms * ((nrels+1) * (nrels+2)) / 2);
	suspend_sz = STROMALIGN(offsetof(gpujoinSuspendContext,
									 pd[nrels + 1]));
	if (kgjoin)
	{
		memset(kgjoin, 0, head_sz);
		kgjoin->kparams_offset = head_sz;
		kgjoin->pstack_offset = head_sz + param_sz;
		kgjoin->pstack_nrooms = pstack_nrooms;
		kgjoin->suspend_offset = head_sz + param_sz + mp_count * pstack_sz;
		kgjoin->suspend_size = mp_count * suspend_sz;
		kgjoin->num_rels = gjs->num_rels;
		kgjoin->src_read_pos = 0;

		/* kern_parambuf */
		memcpy(KERN_GPUJOIN_PARAMBUF(kgjoin),
			   gjs->gts.kern_params,
			   gjs->gts.kern_params->length);
	}
	return (head_sz + param_sz +
			mp_count * pstack_sz +
			mp_count * suspend_sz);
}

/*
 * gpujoin_create_task
 */
static GpuTask *
gpujoin_create_task(GpuJoinState *gjs,
					pgstrom_data_store *pds_src,
					int outer_depth)
{
	TupleTableSlot *scan_slot = gjs->gts.css.ss.ss_ScanTupleSlot;
	TupleDesc		scan_tupdesc = scan_slot->tts_tupleDescriptor;
	GpuContext	   *gcontext = gjs->gts.gcontext;
	GpuJoinTask	   *pgjoin;
	Size			required;
	CUdeviceptr		m_deviceptr;
	CUresult		rc;

	Assert(pds_src || (outer_depth > 0 && outer_depth <= gjs->num_rels));

	required = GpuJoinSetupTask(NULL, &gjs->gts, pds_src);
	rc = gpuMemAllocManaged(gcontext,
							&m_deviceptr,
							offsetof(GpuJoinTask,
									 kern) + required,
							CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on gpuMemAllocManaged: %s", errorText(rc));
	pgjoin = (GpuJoinTask *)m_deviceptr;

	memset(pgjoin, 0, offsetof(GpuJoinTask, kern));
	pgstromInitGpuTask(&gjs->gts, &pgjoin->task);
	pgjoin->pds_src = pds_src;
	pgjoin->pds_dst = PDS_create_row(gcontext,
									 scan_tupdesc,
									 pgstrom_chunk_size());
	pgjoin->outer_depth = outer_depth;

	/* Is NVMe-Strom available to run this GpuJoin? */
	if (pds_src && ((pds_src->kds.format == KDS_FORMAT_BLOCK &&
					 pds_src->nblocks_uncached > 0) ||
					(pds_src->kds.format == KDS_FORMAT_ARROW &&
					 pds_src->iovec != NULL)))
	{
		pgjoin->with_nvme_strom = true;
	}
	GpuJoinSetupTask(&pgjoin->kern, &gjs->gts, pds_src);

	return &pgjoin->task;
}

/*
 * gpujoinExecOuterScanChunk
 */
pgstrom_data_store *
GpuJoinExecOuterScanChunk(GpuTaskState *gts)
{
	GpuJoinState   *gjs = (GpuJoinState *) gts;
	pgstrom_data_store *pds = NULL;

	if (gjs->gts.css.ss.ss_currentRelation)
	{
		if (gjs->gts.af_state)
			pds = ExecScanChunkArrowFdw(gts);
		else
			pds = pgstromExecScanChunk(gts);
	}
	else
	{
		PlanState	   *outer_node = outerPlanState(gjs);
		TupleTableSlot *slot;

		for (;;)
		{
			if (gjs->gts.scan_overflow)
			{
				if (gjs->gts.scan_overflow == (void *)(~0UL))
					break;
				slot = gjs->gts.scan_overflow;
				gjs->gts.scan_overflow = NULL;
			}
			else
			{
				slot = ExecProcNode(outer_node);
				if (TupIsNull(slot))
				{
					/*
					 * FIXME: Why not just scan_done = true here?
					 */
					gjs->gts.scan_overflow = (void *)(~0UL);
					break;
				}
			}

			/* creation of a new data-store on demand */
			if (!pds)
			{
				pds = PDS_create_row(gjs->gts.gcontext,
									 ExecGetResultType(outer_node),
									 pgstrom_chunk_size());
			}
			/* insert the tuple on the data-store */
			if (!PDS_insert_tuple(pds, slot))
			{
				gjs->gts.scan_overflow = slot;
				break;
			}
		}
	}
	return pds;
}

/*
 * gpujoin_switch_task
 */
static void
gpujoin_switch_task(GpuTaskState *gts, GpuTask *gtask)
{
}

/*
 * gpujoin_next_task
 */
static GpuTask *
gpujoin_next_task(GpuTaskState *gts)
{
	GpuJoinState   *gjs = (GpuJoinState *) gts;
	GpuTask		   *gtask = NULL;
	pgstrom_data_store *pds;

	if (gjs->gts.af_state)
		pds = ExecScanChunkArrowFdw(gts);
	else
		pds = GpuJoinExecOuterScanChunk(gts);
	if (pds)
		gtask = gpujoin_create_task(gjs, pds, -1);
	return gtask;
}

/*
 * gpujoinNextRightOuterJoin
 */
int
gpujoinNextRightOuterJoin(GpuTaskState *gts)
{
	GpuJoinState   *gjs = (GpuJoinState *) gts;
	kern_multirels *h_kmrels = dsm_segment_address(gjs->seg_kmrels);
	int				outer_depth = -1;

	for (outer_depth = Max(gjs->curr_outer_depth + 1, 1);
		 outer_depth <= gjs->num_rels;
		 outer_depth++)
	{
		if (h_kmrels->chunks[outer_depth - 1].right_outer)
		{
			gjs->curr_outer_depth = outer_depth;
			return outer_depth;
		}
	}
	return -1;	/* no more RIGHT/FULL OUTER JOIN task */
}

/*
 * gpujoinSyncRightOuterJoin
 */
void
gpujoinSyncRightOuterJoin(GpuTaskState *gts)
{
	GpuJoinState   *gjs = (GpuJoinState *) gts;
	GpuJoinSharedState *gj_sstate = gjs->gj_sstate;
	GpuContext	   *gcontext = gjs->gts.gcontext;

	/*
	 * wait for completion of PG workers, if any
	 */
	if (!IsParallelWorker())
	{
		ResetLatch(MyLatch);
		while (pg_atomic_read_u32(&gj_sstate->pg_nworkers) > 1)
		{
			int		ev;

			CHECK_FOR_GPUCONTEXT(gcontext);

			ev = WaitLatch(MyLatch,
						   WL_LATCH_SET |
						   WL_TIMEOUT |
						   WL_POSTMASTER_DEATH,
						   1000L,
						   PG_WAIT_EXTENSION);
			if (ev & WL_POSTMASTER_DEATH)
				elog(FATAL, "Unexpected Postmaster Dead");
			ResetLatch(MyLatch);
		}
		/* OK, no PG workers run GpuJoin at this point */
	}
	else
	{
		/*
		 * In case of PG workers, the last process per GPU needs to write back
		 * OUTER JOIN map to the DSM area. (only happen on multi-GPU mode)
		 */
		kern_multirels *h_kmrels = dsm_segment_address(gjs->seg_kmrels);
		cl_int			dindex = gcontext->cuda_dindex;
		uint32			pg_nworkers_pergpu =
			pg_atomic_sub_fetch_u32(&gj_sstate->pergpu[dindex].pg_nworkers, 1);

		if (pg_nworkers_pergpu == 0)
		{
			CUresult	rc;
			size_t		offset = (h_kmrels->kmrels_length +
								  h_kmrels->ojmaps_length * dindex);

			rc = cuCtxPushCurrent(gcontext->cuda_context);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));

			rc = cuMemcpyDtoH((char *)h_kmrels + offset,
							  gjs->m_kmrels + offset,
							  h_kmrels->ojmaps_length);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuMemcpyDtoH: %s", errorText(rc));

			rc = cuCtxPopCurrent(NULL);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));
		}
		pg_atomic_fetch_sub_u32(&gj_sstate->pg_nworkers, 1);
		SetLatch(gj_sstate->masterLatch);
	}
}

/*
 * gpujoin_terminator_task
 */
static GpuTask *
gpujoin_terminator_task(GpuTaskState *gts, cl_bool *task_is_ready)
{
	GpuJoinState   *gjs = (GpuJoinState *) gts;
	GpuTask		   *gtask = NULL;
	cl_int			outer_depth;

	/* Has RIGHT/FULL OUTER JOIN? */
	if (gpujoinHasRightOuterJoin(&gjs->gts))
	{
		gpujoinSyncRightOuterJoin(&gjs->gts);
		if (!IsParallelWorker() &&
			(outer_depth = gpujoinNextRightOuterJoin(&gjs->gts)) > 0)
			gtask = gpujoin_create_task(gjs, NULL, outer_depth);
	}
	return gtask;
}

static TupleTableSlot *
gpujoin_next_tuple(GpuTaskState *gts)
{
	GpuJoinState   *gjs = (GpuJoinState *) gts;
	GpuJoinTask	   *pgjoin = (GpuJoinTask *)gjs->gts.curr_task;
	TupleTableSlot *slot;

	if (pgjoin->task.cpu_fallback)
	{
		/*
		 * MEMO: We may reuse tts_values[]/tts_isnull[] of the previous
		 * tuple, to avoid same part of tuple extraction. For example,
		 * portion by depth < 2 will not be changed during iteration in
		 * depth == 3. You may need to pay attention on the core changes
		 * in the future version.
		 */
		slot = gpujoinNextTupleFallback(&gjs->gts,
										&pgjoin->kern,
										pgjoin->pds_src,
										Max(pgjoin->outer_depth,0));
	}
	else
	{
		slot = gjs->gts.css.ss.ss_ScanTupleSlot;
		ExecClearTuple(slot);
		if (!PDS_fetch_tuple(slot, pgjoin->pds_dst, &gjs->gts))
			slot = NULL;
	}
	return slot;
}

/* ----------------------------------------------------------------
 *
 * Routines for CPU fallback, if kernel code returned CpuReCheck
 * error code.
 *
 * ----------------------------------------------------------------
 */
static void
gpujoin_fallback_tuple_extract(TupleTableSlot *slot_fallback,
							   kern_data_store *kds,
							   ItemPointer t_self,
							   HeapTupleHeader htup,
							   AttrNumber *tuple_dst_resno,
							   AttrNumber src_anum_min,
							   AttrNumber src_anum_max)
{
	bool		hasnulls;
	TupleDesc	tts_tupdesc __attribute__((unused))
		= slot_fallback->tts_tupleDescriptor;
	Datum	   *tts_values = slot_fallback->tts_values;
	bool	   *tts_isnull = slot_fallback->tts_isnull;
	cl_uint		offset;
	int			i, nattrs;
	AttrNumber	resnum;

	Assert(kds->format == KDS_FORMAT_ROW ||
		   kds->format == KDS_FORMAT_HASH ||
		   kds->format == KDS_FORMAT_BLOCK);
	Assert(src_anum_min > FirstLowInvalidHeapAttributeNumber);
	Assert(src_anum_max <= kds->ncols);

	/* fill up the destination with NULL, if no tuple was supplied. */
	if (!htup)
	{
		for (i = src_anum_min; i <= src_anum_max; i++)
		{
			resnum = tuple_dst_resno[i-FirstLowInvalidHeapAttributeNumber-1];
			if (resnum)
			{
				Assert(resnum > 0 && resnum <= tts_tupdesc->natts);
				tts_values[resnum - 1] = (Datum) 0;
				tts_isnull[resnum - 1] = true;
			}
		}
		return;
	}
	hasnulls = ((htup->t_infomask & HEAP_HASNULL) != 0);

	/* Extract system columns if any */
	for (i = src_anum_min; i < 0; i++)
	{
		ItemPointer	temp;
		Datum		datum;

		resnum = tuple_dst_resno[i - FirstLowInvalidHeapAttributeNumber - 1];
		if (!resnum)
			continue;
		Assert(resnum > 0 && resnum <= tts_tupdesc->natts);
		switch (i)
		{
			case SelfItemPointerAttributeNumber:
				temp = palloc(sizeof(ItemPointerData));
				ItemPointerCopy(t_self, temp);
				datum = PointerGetDatum(temp);
				break;
			case MaxCommandIdAttributeNumber:
				datum = CommandIdGetDatum(HeapTupleHeaderGetRawCommandId(htup));
				break;
			case MaxTransactionIdAttributeNumber:
				datum = TransactionIdGetDatum(HeapTupleHeaderGetRawXmax(htup));
				break;
			case MinCommandIdAttributeNumber:
				datum = CommandIdGetDatum(HeapTupleHeaderGetRawCommandId(htup));
				break;
			case MinTransactionIdAttributeNumber:
				datum = TransactionIdGetDatum(HeapTupleHeaderGetRawXmin(htup));
				break;
			case ObjectIdAttributeNumber:
				datum = ObjectIdGetDatum(HeapTupleHeaderGetOid(htup));
				break;
			case TableOidAttributeNumber:
				datum = ObjectIdGetDatum(kds->table_oid);
				break;
			default:
				elog(ERROR, "Bug? unknown system attribute: %d", i);
		}
		tts_isnull[resnum - 1] = false;
		tts_values[resnum - 1] = datum;
	}

	/*
	 * Extract user defined columns, according to the logic in
	 * heap_deform_tuple(), but implemented by ourselves for performance.
	 */
	nattrs = HeapTupleHeaderGetNatts(htup);
	nattrs = Min3(nattrs, kds->ncols, src_anum_max);

	offset = htup->t_hoff;
	for (i=0; i < nattrs; i++)
	{
		resnum = tuple_dst_resno[i - FirstLowInvalidHeapAttributeNumber];
		if (hasnulls && att_isnull(i, htup->t_bits))
		{
			if (resnum > 0)
			{
				Assert(resnum <= tts_tupdesc->natts);
				tts_values[resnum - 1] = (Datum) 0;
				tts_isnull[resnum - 1] = true;
			}
		}
		else
		{
			kern_colmeta   *cmeta = &kds->colmeta[i];

			if (cmeta->attlen > 0)
				offset = TYPEALIGN(cmeta->attalign, offset);
			else if (!VARATT_NOT_PAD_BYTE((char *)htup + offset))
				offset = TYPEALIGN(cmeta->attalign, offset);
			if (resnum > 0)
			{
				void	   *addr = ((char *)htup + offset);

				Assert(resnum <= tts_tupdesc->natts);
				tts_isnull[resnum - 1] = false;
				if (cmeta->attbyval)
				{
					Datum	datum = 0;

					if (cmeta->attlen == sizeof(cl_char))
                        datum = *((cl_char *)addr);
                    else if (cmeta->attlen == sizeof(cl_short))
                        datum = *((cl_short *)addr);
                    else if (cmeta->attlen == sizeof(cl_int))
						datum = *((cl_int *)addr);
					else if (cmeta->attlen == sizeof(cl_long))
						datum = *((cl_long *)addr);
					else
					{
						Assert(cmeta->attlen <= sizeof(Datum));
						memcpy(&datum, addr, cmeta->attlen);
					}
					tts_values[resnum - 1] = datum;
					offset += cmeta->attlen;
				}
				else
				{
					tts_values[resnum - 1] = PointerGetDatum(addr);
				    offset += (cmeta->attlen < 0
							   ? VARSIZE_ANY(addr)
							   : cmeta->attlen);
				}
			}
		}
	}

	/*
     * If tuple doesn't have all the atts indicated by src_anum_max,
	 * read the rest as null
	 */
	for (; i < src_anum_max; i++)
	{
		resnum = tuple_dst_resno[i - FirstLowInvalidHeapAttributeNumber];
		if (resnum > 0)
		{
			Assert(resnum <= tts_tupdesc->natts);
			tts_values[resnum - 1] = (Datum) 0;
			tts_isnull[resnum - 1] = true;
		}
	}
}

/*
 * Hash-Join for CPU fallback
 */
static int
gpujoinFallbackHashJoin(int depth, GpuJoinState *gjs)
{
	ExprContext	   *econtext = gjs->gts.css.ss.ps.ps_ExprContext;
	innerState	   *istate = &gjs->inners[depth-1];
	kern_multirels *h_kmrels = dsm_segment_address(gjs->seg_kmrels);
	kern_data_store *kds_in = KERN_MULTIRELS_INNER_KDS(h_kmrels, depth);
	cl_bool		   *ojmaps = KERN_MULTIRELS_OUTER_JOIN_MAP(h_kmrels, depth);
	kern_hashitem  *khitem;
	cl_uint			hash;
	bool			retval;

	do {
		if (istate->fallback_inner_index == 0)
		{
			bool	is_nullkeys;

			hash = get_tuple_hashvalue(istate,
									   false,
									   gjs->slot_fallback,
									   &is_nullkeys);
			/* all-null keys never match to inner rows */
			if (is_nullkeys)
				goto end;
			istate->fallback_inner_hash = hash;
			for (khitem = KERN_HASH_FIRST_ITEM(kds_in, hash);
				 khitem && khitem->hash != hash;
				 khitem = KERN_HASH_NEXT_ITEM(kds_in, khitem));
			if (!khitem)
				goto end;
		}
		else
		{
			hash = istate->fallback_inner_hash;
			khitem = (kern_hashitem *)
				((char *)kds_in + istate->fallback_inner_index);
			for (khitem = KERN_HASH_NEXT_ITEM(kds_in, khitem);
				 khitem && khitem->hash != hash;
				 khitem = KERN_HASH_NEXT_ITEM(kds_in, khitem));
			if (!khitem)
				goto end;
		}
		istate->fallback_inner_index =
			(cl_uint)((char *)khitem - (char *)kds_in);

		gpujoin_fallback_tuple_extract(gjs->slot_fallback,
									   kds_in,
									   &khitem->t.t_self,
									   &khitem->t.htup,
									   istate->inner_dst_resno,
									   istate->inner_src_anum_min,
									   istate->inner_src_anum_max);
#if PG_VERSION_NUM < 100000
		retval = ExecQual(istate->other_quals, econtext, false);
#else
		retval = ExecQual(istate->other_quals, econtext);
#endif
	} while (!retval);

	/* update outer join map */
	if (ojmaps)
		ojmaps[khitem->rowid] = 1;
	/* rewind the next depth */
	if (depth < gjs->num_rels)
	{
		istate++;
		istate->fallback_inner_index = 0;
		istate->fallback_inner_matched = false;
	}
	return depth+1;

end:
	if (!istate->fallback_inner_matched &&
		(istate->join_type == JOIN_LEFT ||
		 istate->join_type == JOIN_FULL))
	{
		istate->fallback_inner_matched = true;
		gpujoin_fallback_tuple_extract(gjs->slot_fallback,
									   kds_in,
                                       NULL,
                                       NULL,
									   istate->inner_dst_resno,
									   istate->inner_src_anum_min,
									   istate->inner_src_anum_max);
		if (depth < gjs->num_rels)
		{
			istate++;
			istate->fallback_inner_index = 0;
			istate->fallback_inner_matched = false;
		}
		return depth+1;
	}
	/* pop up one level */
	return depth-1;
}

/*
 * Nest-Loop for CPU fallback
 */
static int
gpujoinFallbackNestLoop(int depth, GpuJoinState *gjs)
{
	ExprContext	   *econtext = gjs->gts.css.ss.ps.ps_ExprContext;
	innerState	   *istate = &gjs->inners[depth-1];
	kern_multirels *h_kmrels = dsm_segment_address(gjs->seg_kmrels);
	kern_data_store *kds_in = KERN_MULTIRELS_INNER_KDS(h_kmrels, depth);
	cl_bool		   *ojmaps = KERN_MULTIRELS_OUTER_JOIN_MAP(h_kmrels, depth);
	cl_uint			index;

	for (index = istate->fallback_inner_index;
		 index < kds_in->nitems;
		 index++)
	{
		kern_tupitem   *tupitem = KERN_DATA_STORE_TUPITEM(kds_in, index);
		bool			retval;

		gpujoin_fallback_tuple_extract(gjs->slot_fallback,
									   kds_in,
									   &tupitem->t_self,
									   &tupitem->htup,
									   istate->inner_dst_resno,
									   istate->inner_src_anum_min,
									   istate->inner_src_anum_max);
#if PG_VERSION_NUM < 100000
		retval = ExecQual(istate->join_quals, econtext, false);
#else
		retval = ExecQual(istate->join_quals, econtext);
#endif
		if (retval)
		{
			istate->fallback_inner_index = index + 1;
			/* update outer join map */
			if (ojmaps)
				ojmaps[index] = 1;
			/* rewind the next depth */
			if (depth < gjs->num_rels)
			{
				istate++;
				istate->fallback_inner_index = 0;
				istate->fallback_inner_matched = false;
			}
			return depth+1;
		}
	}

	if (!istate->fallback_inner_matched &&
		(istate->join_type == JOIN_LEFT ||
		 istate->join_type == JOIN_FULL))
	{
		istate->fallback_inner_index = kds_in->nitems;
		istate->fallback_inner_matched = true;

		gpujoin_fallback_tuple_extract(gjs->slot_fallback,
									   kds_in,
									   NULL,
									   NULL,
									   istate->inner_dst_resno,
									   istate->inner_src_anum_min,
									   istate->inner_src_anum_max);
		/* rewind the next depth */
		if (depth < gjs->num_rels)
		{
			istate++;
			istate->fallback_inner_index = 0;
			istate->fallback_inner_matched = false;
		}
		return depth+1;
	}
	/* pop up one level */
	return depth-1;
}

static int
gpujoinFallbackLoadOuter(int depth, GpuJoinState *gjs)
{
	kern_multirels *h_kmrels = dsm_segment_address(gjs->seg_kmrels);
	kern_data_store *kds_in = KERN_MULTIRELS_INNER_KDS(h_kmrels, depth);
	cl_bool		   *ojmaps = KERN_MULTIRELS_OUTER_JOIN_MAP(h_kmrels, depth);
	cl_uint			index;

	for (index = gjs->inners[depth-1].fallback_inner_index;
		 index < kds_in->nitems;
		 index++)
	{
		if (!ojmaps[index])
		{
			innerState	   *istate = &gjs->inners[depth-1];
			kern_tupitem   *tupitem = KERN_DATA_STORE_TUPITEM(kds_in, index);

			gpujoin_fallback_tuple_extract(gjs->slot_fallback,
										   kds_in,
										   &tupitem->t_self,
										   &tupitem->htup,
										   istate->inner_dst_resno,
										   istate->inner_src_anum_min,
										   istate->inner_src_anum_max);
			istate->fallback_inner_index = index + 1;
			/* rewind the next depth */
			if (depth < gjs->num_rels)
			{
				istate++;
				istate->fallback_inner_index = 0;
				istate->fallback_inner_matched = false;
			}
			return depth + 1;
		}
	}
	return -1;
}

static int
gpujoinFallbackLoadSource(int depth, GpuJoinState *gjs,
						  pgstrom_data_store *pds_src)
{
	kern_data_store *kds_src = &pds_src->kds;
	ExprContext	   *econtext = gjs->gts.css.ss.ps.ps_ExprContext;
	bool			retval;

	Assert(depth == 0);
	do {
		if (kds_src->format == KDS_FORMAT_ROW)
		{
			cl_uint			index = gjs->fallback_outer_index++;
			kern_tupitem   *tupitem;

			if (index >= kds_src->nitems)
				return -1;
			/* fills up fallback_slot with outer columns */
			tupitem = KERN_DATA_STORE_TUPITEM(kds_src, index);
			gpujoin_fallback_tuple_extract(gjs->slot_fallback,
										   kds_src,
										   &tupitem->t_self,
										   &tupitem->htup,
										   gjs->outer_dst_resno,
										   gjs->outer_src_anum_min,
										   gjs->outer_src_anum_max);
		}
		else if (kds_src->format == KDS_FORMAT_BLOCK)
		{
			HeapTupleHeader	htup;
			ItemPointerData	t_self;
			PageHeaderData *pg_page;
			BlockNumber		block_nr;
			cl_uint			line_nr;
			cl_uint			index;
			ItemIdData	   *lpp;

			index = (gjs->fallback_outer_index >> 16);
			line_nr = (gjs->fallback_outer_index++ & 0xffff);
			if (index >= kds_src->nitems)
				return -1;
			pg_page = KERN_DATA_STORE_BLOCK_PGPAGE(kds_src, index);
			block_nr = KERN_DATA_STORE_BLOCK_BLCKNR(kds_src, index);
			if (line_nr >= PageGetMaxOffsetNumber(pg_page))
			{
				gjs->fallback_outer_index = (cl_ulong)(index + 1) << 16;
				continue;
			}
			lpp = PageGetItemId(pg_page, line_nr + 1);
			if (!ItemIdIsNormal(lpp))
				continue;
			htup = (HeapTupleHeader)PageGetItem(pg_page, lpp);
			t_self.ip_blkid.bi_hi = block_nr >> 16;
			t_self.ip_blkid.bi_lo = block_nr & 0xffff;
			t_self.ip_posid = line_nr + 1;

			gpujoin_fallback_tuple_extract(gjs->slot_fallback,
										   kds_src,
										   &t_self,
										   htup,
										   gjs->outer_dst_resno,
										   gjs->outer_src_anum_min,
										   gjs->outer_src_anum_max);
		}
		else
			elog(ERROR, "Bug? unexpected KDS format: %d", pds_src->kds.format);
#if PG_VERSION_NUM < 100000
		retval = ExecQual(gjs->outer_quals, econtext, false);
#else
		retval = ExecQual(gjs->outer_quals, econtext);
#endif
	} while (!retval);

	/* rewind the next depth */
	gjs->inners[0].fallback_inner_index = 0;
	return 1;
}

/*
 * gpujoinFallbackLoadFromSuspend
 */
static int
gpujoinFallbackLoadFromSuspend(GpuJoinState *gjs,
							   kern_gpujoin *kgjoin,
							   pgstrom_data_store *pds_src,
							   int outer_depth)
{
	kern_multirels *h_kmrels = dsm_segment_address(gjs->seg_kmrels);
	cl_int		num_rels = gjs->num_rels;
	cl_uint		block_sz = kgjoin->block_sz;
	cl_uint		grid_sz = kgjoin->grid_sz;
	cl_uint		global_sz = block_sz * grid_sz;
	cl_long		thread_index;
	cl_long		thread_loops;
	cl_int		depth;
	cl_uint		global_id;
	cl_uint		group_id;
	cl_uint		local_id;
	cl_uint		nrooms = kgjoin->pstack_nrooms;
	cl_uint		write_pos;
	cl_uint		read_pos;
	cl_uint		row_index;
	cl_uint	   *pstack;
	cl_int		j;
	gpujoinSuspendContext *sb;
	HeapTupleHeaderData *htup;
	ItemPointerData t_self;

lnext:
	/* setup pseudo thread-id based on fallback_thread_count */
	thread_index = (gjs->fallback_thread_count >> 10);
	thread_loops = (gjs->fallback_thread_count & 0x03ff);
	depth = thread_index / global_sz + outer_depth;
	global_id = thread_index % global_sz;
	group_id = global_id / block_sz;
	local_id = global_id % block_sz;

	/* no more pending rows in the suspend context */
	if (depth > num_rels)
	{
		gjs->fallback_outer_index = kgjoin->src_read_pos;
		kgjoin->resume_context = false;
		return 0;
	}
	gjs->fallback_resume_depth = depth;

	/* suspend context and pseudo stack */
	pstack = (cl_uint *)((char *)kgjoin + kgjoin->pstack_offset)
		+ group_id * nrooms * ((num_rels + 1) * 
							   (num_rels + 2) / 2)
		+ nrooms * (depth * (depth + 1)) / 2;
	sb = KERN_GPUJOIN_SUSPEND_CONTEXT(kgjoin, group_id);
	if (sb->depth < 0)
	{
		/*
		 * This threads-group successfull finished.
		 * So, move to the next threads-groups.
		 */
		gjs->fallback_thread_count =
			((thread_index / block_sz + 1) * block_sz) << 10;
		goto lnext;
	}
	else if (sb->depth != num_rels + 1)
		elog(ERROR, "Bug? unexpected point for GpuJoin kernel suspend");

	write_pos = sb->pd[depth].write_pos;
	read_pos = sb->pd[depth].read_pos;
	row_index = block_sz * thread_loops + local_id;
	if (row_index >= write_pos)
	{
		if (local_id < block_sz)
		{
			/* move to the next thread */
			gjs->fallback_thread_count = (thread_index + 1) << 10;
		}
		else
		{
			/* move to the next threads group */
			gjs->fallback_thread_count =
				((thread_index / block_sz + 1) * block_sz) << 10;
		}
		goto lnext;
	}
	gjs->fallback_thread_count++;

	/* extract partially joined tuples */
	pstack += row_index * (depth + 1);
	for (j=outer_depth; j <= depth; j++)
	{
		if (j == 0)
		{
			/* load from the outer source buffer */
			if (pds_src->kds.format == KDS_FORMAT_ROW)
			{
				htup = KDS_ROW_REF_HTUP(&pds_src->kds,
										pstack[0],
										&t_self, NULL);
				gpujoin_fallback_tuple_extract(gjs->slot_fallback,
											   &pds_src->kds,
											   &t_self,
											   htup,
											   gjs->outer_dst_resno,
											   gjs->outer_src_anum_min,
											   gjs->outer_src_anum_max);
			}
			else if (pds_src->kds.format == KDS_FORMAT_BLOCK)
			{
				HeapTupleHeader	htup;
				ItemPointerData	t_self;

				htup = KDS_BLOCK_REF_HTUP(&pds_src->kds,
										  pstack[0],
										  &t_self, NULL);
				gpujoin_fallback_tuple_extract(gjs->slot_fallback,
											   &pds_src->kds,
											   &t_self,
											   htup,
											   gjs->outer_dst_resno,
											   gjs->outer_src_anum_min,
											   gjs->outer_src_anum_max);
			}
			else
			{
				elog(ERROR, "Bug? unexpected PDS format: %d",
					 pds_src->kds.format);
			}
		}
		else
		{
			innerState	   *istate = &gjs->inners[j-1];
			kern_data_store *kds_in = KERN_MULTIRELS_INNER_KDS(h_kmrels, j);

			htup = KDS_ROW_REF_HTUP(kds_in,pstack[j],&t_self,NULL);
			gpujoin_fallback_tuple_extract(gjs->slot_fallback,
										   kds_in,
										   &t_self,
										   htup,
										   istate->inner_dst_resno,
										   istate->inner_src_anum_min,
										   istate->inner_src_anum_max);
		}
	}

	/* assign starting point of the next depth */
	if (depth < num_rels)
	{
		innerState	   *istate = &gjs->inners[depth];

		if (row_index < read_pos)
		{
			/*
			 * This partially joined row is already processed by the deeper
			 * level, so no need to move deeper level any more.
			 */
			goto lnext;
		}
		else if (row_index < read_pos + kgjoin->block_sz)
		{
			/*
			 * This partially joined row is now processed by the deeper
			 * level, so we must restart from the next position.
			 */
			kern_data_store *kds_in;
			cl_uint		l_state = sb->pd[depth+1].l_state[local_id];
			cl_bool		matched = sb->pd[depth+1].matched[local_id];

			kds_in = KERN_MULTIRELS_INNER_KDS(h_kmrels, depth+1);
			if (kds_in->format == KDS_FORMAT_HASH)
			{
				if (l_state == 0)
				{
					/* restart from the head */
					gjs->inners[depth].fallback_inner_index = 0;
				}
				else if (l_state == UINT_MAX)
				{
					/* already reached end of the hash-chain */
					gjs->fallback_thread_count = (thread_index + 1) << 10;
					goto lnext;
				}
				else
				{
					kern_hashitem  *khitem = (kern_hashitem *)
						((char *)kds_in
						 + __kds_unpack(l_state)
						 - offsetof(kern_hashitem, t.htup));
					istate->fallback_inner_index =
						((char *)khitem - (char *)kds_in);
					istate->fallback_inner_hash = khitem->hash;
					istate->fallback_inner_matched = matched;
				}
			}
			else if (kds_in->format == KDS_FORMAT_ROW)
			{
				cl_uint		x_unitsz = Min(write_pos - read_pos,
										   kgjoin->grid_sz);
				cl_uint		y_unitsz = kgjoin->grid_sz / x_unitsz;

				istate->fallback_inner_index = l_state + y_unitsz;
			}
			else
				elog(ERROR, "Bug? unexpected inner buffer format: %d",
					 kds_in->format);
		}
		else
		{
			/*
			 * This partially joined row is not processed in the deeper
			 * level, so we shall suspend join from the head.
			 */
			istate->fallback_inner_index = 0;
		}
	}
	else
	{
		/*
		 * This completely joined row is already written to the destination
		 * buffer, thus should be preliminary fetched.
		 */
		if (row_index < read_pos)
			goto lnext;
	}
	/* make the fallback_thread_count advanced */
	return depth + 1;
}

/*
 * gpujoinNextTupleFallback - CPU Fallback
 */
TupleTableSlot *
gpujoinNextTupleFallback(GpuTaskState *gts,
						 kern_gpujoin *kgjoin,
						 pgstrom_data_store *pds_src,
						 cl_int outer_depth)
{
	GpuJoinState   *gjs = (GpuJoinState *) gts;
	GpuJoinSharedState *gj_sstate = gjs->gj_sstate;
	ExprContext	   *econtext = gjs->gts.css.ss.ps.ps_ExprContext;
	cl_int			depth;

	econtext->ecxt_scantuple = gjs->slot_fallback;
	ResetExprContext(econtext);

	if (gjs->fallback_outer_index < 0)
	{
		cl_int		i, num_rels = gjs->num_rels;

		/* init cpu fallback state for each GpuTask */
		pg_atomic_write_u32(&gj_sstate->needs_colocation, 1);
		if (pds_src)
			Assert(outer_depth == 0);
		else
		{
			Assert(outer_depth > 0 && outer_depth <= num_rels);
			gpujoinColocateOuterJoinMapsToHost(gjs);
		}
		gjs->fallback_resume_depth = outer_depth;
		gjs->fallback_thread_count = 0;
		gjs->fallback_outer_index = 0;
		for (i=0; i < num_rels; i++)
			gjs->inners[i].fallback_inner_index = 0;

		depth = outer_depth;
	}
	else
	{
		depth = gjs->num_rels;
	}

	while (depth >= 0)
	{
		Assert(depth >= outer_depth);
		if (depth == (kgjoin->resume_context
					  ? gjs->fallback_resume_depth
					  : outer_depth))
		{
			ExecStoreAllNullTuple(gjs->slot_fallback);
			if (kgjoin->resume_context)
				depth = gpujoinFallbackLoadFromSuspend(gjs, kgjoin, pds_src,
													   outer_depth);
			else if (pds_src)
				depth = gpujoinFallbackLoadSource(depth, gjs, pds_src);
			else
				depth = gpujoinFallbackLoadOuter(depth, gjs);
		}
		else if (depth <= gjs->num_rels)
		{
			if (gjs->inners[depth-1].hash_outer_keys != NIL)
				depth = gpujoinFallbackHashJoin(depth, gjs);
			else
				depth = gpujoinFallbackNestLoop(depth, gjs);
		}
		else
		{
			TupleTableSlot *slot = gjs->slot_fallback;

			/* projection? */
			if (gjs->proj_fallback)
			{
#if PG_VERSION_NUM < 100000
				ExprDoneCond	is_done;

				slot = ExecProject(gjs->proj_fallback, &is_done);
#else
				slot = ExecProject(gjs->proj_fallback);
#endif
			}
			Assert(slot == gjs->gts.css.ss.ss_ScanTupleSlot);
			return slot;
		}
	}
	/* rewind the fallback status for the further GpuJoinTask */
	gjs->fallback_outer_index = -1;
	return NULL;
}

/* ----------------------------------------------------------------
 *
 * Routines to support combined GpuPreAgg + GpuJoin
 *
 * ----------------------------------------------------------------
 */
ProgramId
GpuJoinCreateCombinedProgram(PlanState *node,
							 GpuTaskState *gpa_gts,
							 cl_uint gpa_extra_flags,
							 cl_uint gpa_varlena_bufsz,
							 const char *gpa_kern_source,
							 bool explain_only)
{
	GpuJoinState   *gjs = (GpuJoinState *) node;
	GpuJoinInfo	   *gj_info;
	StringInfoData	kern_define;
	StringInfoData	kern_source;
	cl_uint			extra_flags;
	ProgramId		program_id;

	initStringInfo(&kern_define);
	initStringInfo(&kern_source);

	gj_info = deform_gpujoin_info((CustomScan *) gjs->gts.css.ss.ps.plan);
	extra_flags = (gpa_extra_flags | gj_info->extra_flags);
	pgstrom_build_session_info(&kern_define,
							   gpa_gts,
							   extra_flags & ~DEVKERNEL_NEEDS_GPUJOIN);
	assign_gpujoin_session_info(&kern_define, &gjs->gts);

	appendStringInfoString(
		&kern_source,
		"\n/* ====== BEGIN GpuJoin Portion ====== */\n\n");
	appendStringInfoString(
		&kern_source,
		gj_info->kern_source);
	appendStringInfoString(
		&kern_source,
		"\n/* ====== BEGIN GpuPreAgg Portion ====== */\n\n");
	appendStringInfoString(&kern_source, gpa_kern_source);

	program_id = pgstrom_create_cuda_program(gpa_gts->gcontext,
											 extra_flags,
											 Max(gj_info->varlena_bufsz,
												 gpa_varlena_bufsz),
											 kern_source.data,
											 kern_define.data,
											 false,
											 explain_only);
	pfree(kern_source.data);
	pfree(kern_define.data);

	return program_id;
}

/* ----------------------------------------------------------------
 *
 * GpuTask handlers of GpuJoin
 *
 * ----------------------------------------------------------------
 */
void
gpujoin_release_task(GpuTask *gtask)
{
	GpuJoinTask	   *pgjoin = (GpuJoinTask *) gtask;
	GpuTaskState   *gts = (GpuTaskState *) gtask->gts;

	if (pgjoin->pds_src)
		PDS_release(pgjoin->pds_src);
	if (pgjoin->pds_dst)
		PDS_release(pgjoin->pds_dst);
	/* release this gpu-task itself */
	gpuMemFree(gts->gcontext, (CUdeviceptr)pgjoin);
}

void
gpujoinUpdateRunTimeStat(GpuTaskState *gts, kern_gpujoin *kgjoin)
{
	GpuJoinState	   *gjs = (GpuJoinState *)gts;
	GpuJoinRuntimeStat *gj_rtstat = gjs->gj_rtstat;
	cl_int		i;

	pg_atomic_fetch_add_u64(&gj_rtstat->c.source_nitems,
							kgjoin->source_nitems);
	pg_atomic_fetch_add_u64(&gj_rtstat->c.nitems_filtered,
							kgjoin->source_nitems -
							kgjoin->outer_nitems);
	pg_atomic_fetch_add_u64(&gj_rtstat->jstat[0].inner_nitems,
							kgjoin->outer_nitems);
	for (i=0; i < gjs->num_rels; i++)
	{
		pg_atomic_fetch_add_u64(&gj_rtstat->jstat[i+1].inner_nitems,
								kgjoin->stat_nitems[i]);
	}
	/* reset counters (may be reused by the resumed kernel) */
	kgjoin->source_nitems = 0;
	kgjoin->outer_nitems  = 0;
	for (i=0; i < gjs->num_rels; i++)
		kgjoin->stat_nitems[i] = 0;
}

/*
 * gpujoin_throw_partial_result
 */
static void
gpujoin_throw_partial_result(GpuJoinTask *pgjoin)
{
	GpuContext	   *gcontext = GpuWorkerCurrentContext;
	GpuTaskState   *gts = pgjoin->task.gts;
	pgstrom_data_store *pds_dst = pgjoin->pds_dst;
	pgstrom_data_store *pds_new = PDS_clone(pds_dst);
	cl_int			num_rels = pgjoin->kern.num_rels;
	GpuJoinTask	   *gresp;
	size_t			head_sz;
	size_t			param_sz;
	CUresult		rc;

	/* async prefetch kds_dst; which should be on the device memory */
	rc = cuMemPrefetchAsync((CUdeviceptr) &pds_dst->kds,
							pds_dst->kds.length,
							CU_DEVICE_CPU,
							CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuMemPrefetchAsync: %s", errorText(rc));

	/* setup responder task with supplied @kds_dst */
	head_sz = STROMALIGN(offsetof(GpuJoinTask, kern) +
						 offsetof(kern_gpujoin,
								  stat_nitems[num_rels + 1]));
	param_sz = KERN_GPUJOIN_PARAMBUF_LENGTH(&pgjoin->kern);
	/* pstack/suspend buffer is not necessary */
	rc = gpuMemAllocManaged(gcontext,
							(CUdeviceptr *)&gresp,
							head_sz + param_sz,
							CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
		werror("failed on gpuMemAllocManaged: %s", errorText(rc));

	memset(gresp, 0, head_sz);
	gresp->task.task_kind	= pgjoin->task.task_kind;
	gresp->task.program_id	= pgjoin->task.program_id;
	gresp->task.cpu_fallback= false;
	gresp->task.gts			= gts;
	gresp->pds_src			= PDS_retain(pgjoin->pds_src);
	gresp->pds_dst			= pds_dst;
	gresp->outer_depth		= pgjoin->outer_depth;

	gresp->kern.num_rels	= num_rels;
	memcpy((char *)gresp + head_sz,
		   KERN_GPUJOIN_PARAMBUF(&pgjoin->kern),
		   KERN_GPUJOIN_PARAMBUF_LENGTH(&pgjoin->kern));
	/* assign a new empty buffer */
	pgjoin->pds_dst			= pds_new;

	/* Back GpuTask to GTS */
	pthreadMutexLock(gcontext->mutex);
	dlist_push_tail(&gts->ready_tasks,
					&gresp->task.chain);
	gts->num_ready_tasks++;
	pthreadMutexUnlock(gcontext->mutex);

	SetLatch(MyLatch);
}

/*
 * gpujoinColocateOuterJoinMapsToHost
 *
 * It moves outer-join-map on the device memory to the host memory prior to
 * CPU fallback of RIGHT/FULL OUTER JOIN. When this function is called,
 * no GPU kernel shall not be working, so just cuMemcpyDtoH() works.
 */
static void
gpujoinColocateOuterJoinMapsToHost(GpuJoinState *gjs)
{
	GpuContext	   *gcontext = gjs->gts.gcontext;
	kern_multirels *h_kmrels = dsm_segment_address(gjs->seg_kmrels);
	cl_bool		   *h_ojmaps;
	CUdeviceptr		m_ojmaps;
	CUresult		rc;
	size_t			ojmaps_length = h_kmrels->ojmaps_length;
	cl_uint			i, j, n;

	rc = cuCtxPushCurrent(gcontext->cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));

	h_ojmaps = ((char *)h_kmrels + h_kmrels->kmrels_length);
	m_ojmaps = gjs->m_kmrels + h_kmrels->kmrels_length;
	rc = cuMemcpyDtoH(h_ojmaps,
					  m_ojmaps,
					  ojmaps_length * numDevAttrs);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoD: %s", errorText(rc));

	rc = cuCtxPopCurrent(NULL);
	if (rc != CUDA_SUCCESS)
		elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));

	/* merge OJMaps */
	n = ojmaps_length / sizeof(cl_ulong);
	for (i=0; i < n; i++)
	{
		cl_ulong	mask = 0;
		for (j=0; j < numDevAttrs; j++)
			mask |= ((cl_ulong *)(h_ojmaps + j * ojmaps_length))[i];
		((cl_ulong *)(h_ojmaps + j * ojmaps_length))[i] |= mask;
	}
}

/*
 * gpujoinColocateOuterJoinMaps
 */
void
gpujoinColocateOuterJoinMaps(GpuTaskState *gts, CUmodule cuda_module)
{
	GpuJoinState   *gjs = (GpuJoinState *) gts;
	GpuJoinSharedState *gj_sstate = gjs->gj_sstate;
	kern_multirels *h_kmrels = dsm_segment_address(gjs->seg_kmrels);
	cl_bool		   *h_ojmaps;
	CUdeviceptr		m_ojmaps;
	CUfunction		kern_colocate;
	CUresult		rc;
	size_t			ojmaps_sz = h_kmrels->ojmaps_length;
	cl_int			grid_sz;
	cl_int			block_sz;
	void		   *kern_args[4];

	/* Is the co-location of OUTER JOIN Map needed? */
	if (pg_atomic_read_u32(&gj_sstate->needs_colocation) == 0)
		return;

	rc = cuModuleGetFunction(&kern_colocate,
							 cuda_module,
							 "gpujoin_colocate_outer_join_map");
	if (rc != CUDA_SUCCESS)
		werror("failed on cuModuleGetFunction: %s", errorText(rc));

	h_ojmaps = ((char *)h_kmrels + h_kmrels->kmrels_length);
	m_ojmaps = gjs->m_kmrels + h_kmrels->kmrels_length;
	if (CU_DINDEX_PER_THREAD > 0)
	{
		rc = cuMemcpyHtoD(m_ojmaps,
						  h_ojmaps,
						  ojmaps_sz * CU_DINDEX_PER_THREAD);
		if (rc != CUDA_SUCCESS)
			werror("failed on cuMemcpyHtoD: %s", errorText(rc));
	}
	rc = cuMemcpyHtoD(m_ojmaps + ojmaps_sz * (CU_DINDEX_PER_THREAD + 1),
					  h_ojmaps + ojmaps_sz * (CU_DINDEX_PER_THREAD + 1),
					  (numDevAttrs - CU_DINDEX_PER_THREAD) * ojmaps_sz);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuMemcpyHtoD: %s", errorText(rc));

	/*
	 * Launch)
	 * KERNEL_FUNCTION(void)
	 * gpujoin_colocate_outer_join_map(kern_multirels *kmrels,
	 *                                 cl_uint num_devices)
	 */
	rc = gpuOptimalBlockSize(&grid_sz,
							 &block_sz,
							 kern_colocate,
							 CU_DEVICE_PER_THREAD,
							 0, 0);
	if (rc != CUDA_SUCCESS)
		werror("failed on gpuOptimalBlockSize: %s", errorText(rc));
	grid_sz = Min(grid_sz, (ojmaps_sz / sizeof(cl_uint) +
							block_sz - 1) / block_sz);
	kern_args[0] = &gjs->m_kmrels;
	kern_args[1] = &numDevAttrs;

	rc = cuLaunchKernel(kern_colocate,
						block_sz, 1, 1,
						grid_sz, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuLaunchKernel: %s", errorText(rc));
	pg_atomic_write_u32(&gj_sstate->needs_colocation, 0);
}

static cl_int
gpujoin_process_inner_join(GpuJoinTask *pgjoin, CUmodule cuda_module)
{
	GpuContext		   *gcontext = GpuWorkerCurrentContext;
	GpuJoinState	   *gjs = (GpuJoinState *) pgjoin->task.gts;
	pgstrom_data_store *pds_src = pgjoin->pds_src;
	pgstrom_data_store *pds_dst = pgjoin->pds_dst;
	CUfunction			kern_gpujoin_main;
	CUdeviceptr			m_kgjoin = (CUdeviceptr)&pgjoin->kern;
	CUdeviceptr			m_kds_src = 0UL;
	CUdeviceptr			m_kds_dst;
	CUdeviceptr			m_nullptr = 0UL;
	CUresult			rc;
	bool				m_kds_src_release = false;
	cl_int				grid_sz;
	cl_int				block_sz;
	cl_int				retval = 10001;
	void			   *kern_args[10];
	void			   *last_suspend = NULL;

	/* sanity checks */
	Assert(!pds_src || (pds_src->kds.format == KDS_FORMAT_ROW ||
						pds_src->kds.format == KDS_FORMAT_BLOCK ||
						pds_src->kds.format == KDS_FORMAT_ARROW));
	Assert(pds_dst->kds.format == KDS_FORMAT_ROW);

	/* Lookup GPU kernel function */
	rc = cuModuleGetFunction(&kern_gpujoin_main,
							 cuda_module,
							 "kern_gpujoin_main");
	if (rc != CUDA_SUCCESS)
		werror("failed on cuModuleGetFunction: %s", errorText(rc));

	/*
	 * Device memory allocation
	 */
	if (pgjoin->with_nvme_strom)
	{
		Size	required = GPUMEMALIGN(pds_src->kds.length);

		Assert(pds_src->kds.format == KDS_FORMAT_BLOCK ||
			   pds_src->kds.format == KDS_FORMAT_ARROW);
		rc = gpuMemAllocIOMap(gcontext,
							  &m_kds_src,
							  required);
		if (rc == CUDA_SUCCESS)
			m_kds_src_release = true;
		else if (rc == CUDA_ERROR_OUT_OF_MEMORY)
		{
			pgjoin->with_nvme_strom = false;
			if (pds_src->kds.format == KDS_FORMAT_BLOCK)
			{
				PDS_fillup_blocks(pds_src);

				rc = gpuMemAlloc(gcontext,
								 &m_kds_src,
                                 required);
				if (rc == CUDA_SUCCESS)
					m_kds_src_release = true;
				else if (rc == CUDA_ERROR_OUT_OF_MEMORY)
					goto out_of_resource;
				else
					werror("failed on gpuMemAlloc: %s", errorText(rc));
			}
			else
			{
				pds_src = PDS_fillup_arrow(pgjoin->pds_src);
				PDS_release(pgjoin->pds_src);
				pgjoin->pds_src = pds_src;
				Assert(!pds_src->iovec);
			}
		}
		else
			werror("failed on gpuMemAllocIOMap: %s", errorText(rc));
	}
	else
	{
		m_kds_src = (CUdeviceptr)&pds_src->kds;
	}

	/*
	 * OK, kick a series of GpuJoin invocations
	 */
	if (pgjoin->with_nvme_strom)
	{
		gpuMemCopyFromSSD(m_kds_src, pds_src);
	}
	else if (pds_src->kds.format == KDS_FORMAT_BLOCK)
	{
		rc = cuMemcpyHtoDAsync(m_kds_src,
							   &pds_src->kds,
							   pds_src->kds.length,
							   CU_STREAM_PER_THREAD);
		if (rc != CUDA_SUCCESS)
			werror("failed on cuMemcpyHtoD: %s", errorText(rc));

	}
	else
	{
		rc = cuMemPrefetchAsync(m_kds_src,
								pds_src->kds.length,
								CU_DEVICE_PER_THREAD,
								CU_STREAM_PER_THREAD);
		if (rc != CUDA_SUCCESS)
			werror("failed on cuMemPrefetchAsync: %s", errorText(rc));
	}

	/* Launch:
	 * KERNEL_FUNCTION(void)
	 * gpujoin_main(kern_gpujoin *kgjoin,
	 *              kern_multirels *kmrels,
	 *              kern_data_store *kds_src,
	 *              kern_data_store *kds_dst,
	 *              kern_parambuf *kparams_gpreagg)
	 */
	rc = gpuOptimalBlockSize(&grid_sz,
							 &block_sz,
							 kern_gpujoin_main,
							 CU_DEVICE_PER_THREAD,
							 0, sizeof(cl_int));
	if (rc != CUDA_SUCCESS)
		werror("failed on gpuOptimalBlockSize: %s", errorText(rc));
	pgjoin->kern.grid_sz	= grid_sz;
	pgjoin->kern.block_sz	= block_sz;

resume_kernel:
	m_kds_dst = (CUdeviceptr)&pds_dst->kds;
	kern_args[0] = &m_kgjoin;
	kern_args[1] = &gjs->m_kmrels;
	kern_args[2] = &m_kds_src;
	kern_args[3] = &m_kds_dst;
	kern_args[4] = &m_nullptr;

	rc = cuLaunchKernel(kern_gpujoin_main,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						sizeof(cl_int) * block_sz,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuLaunchKernel: %s", errorText(rc));

	rc = cuEventRecord(CU_EVENT0_PER_THREAD, CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuEventRecord: %s", errorText(rc));

	/* Point of synchronization */
	rc = cuEventSynchronize(CU_EVENT0_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuEventSynchronize: %s", errorText(rc));

	memcpy(&pgjoin->task.kerror,
		   &pgjoin->kern.kerror, sizeof(kern_errorbuf));
	if (pgjoin->task.kerror.errcode == ERRCODE_STROM_SUCCESS)
	{
		if (pgjoin->kern.suspend_count > 0)
		{
			CHECK_WORKER_TERMINATION();
			gpujoin_throw_partial_result(pgjoin);

			pgjoin->kern.suspend_count = 0;
			pgjoin->kern.resume_context = true;
			if (!last_suspend)
				last_suspend = alloca(pgjoin->kern.suspend_size);
			memcpy(last_suspend,
				   KERN_GPUJOIN_SUSPEND_CONTEXT(&pgjoin->kern, 0),
				   pgjoin->kern.suspend_size);
			fprintf(stderr, "suspend / resume\n");
			/* renew buffer and restart */
			pds_dst = pgjoin->pds_dst;
			goto resume_kernel;
		}
		gpujoinUpdateRunTimeStat(&gjs->gts, &pgjoin->kern);
		/* return task if any result rows */
		retval = (pds_dst->kds.nitems > 0 ? 0 : -1);
	}
	else if (pgstrom_cpu_fallback_enabled &&
			 (pgjoin->kern.kerror.errcode & ERRCODE_FLAGS_CPU_FALLBACK) != 0)
	{
		/*
		 * In case of KDS_FORMAT_BLOCK, we have to write back the kernel
		 * buffer to the host-side, because SSD2GPU mode bypass the host
		 * memory, thus, CPU fallback routine will miss the data.
		 */
		if (pds_src->kds.format == KDS_FORMAT_BLOCK)
		{
			rc = cuMemcpyDtoH(&pds_src->kds,
							  m_kds_src,
							  pds_src->kds.length);
			if (rc != CUDA_SUCCESS)
				werror("failed on cuMemcpyDtoH: %s", errorText(rc));
			pds_src->nblocks_uncached = 0;
		}
		else if (pds_src->kds.format == KDS_FORMAT_ARROW &&
				 pds_src->iovec != NULL)
		{
			pgjoin->pds_src = PDS_writeback_arrow(pds_src, m_kds_src);
		}
		memset(&pgjoin->task.kerror, 0, sizeof(kern_errorbuf));
		pgjoin->task.cpu_fallback = true;
		pgjoin->kern.resume_context = (last_suspend != NULL);
		if (last_suspend)
		{
			memcpy(KERN_GPUJOIN_SUSPEND_CONTEXT(&pgjoin->kern, 0),
				   last_suspend,
				   pgjoin->kern.suspend_size);
		}
		retval = 0;
	}
	else
	{
		/* raise an error */
		retval = 0;
	}
out_of_resource:
	if (m_kds_src_release)
		gpuMemFree(gcontext, m_kds_src);
	return retval;
}

static cl_int
gpujoin_process_right_outer(GpuJoinTask *pgjoin, CUmodule cuda_module)
{
	GpuJoinState	   *gjs = (GpuJoinState *) pgjoin->task.gts;
	pgstrom_data_store *pds_dst = pgjoin->pds_dst;
	CUfunction			kern_gpujoin_main;
	CUdeviceptr			m_kgjoin = (CUdeviceptr)&pgjoin->kern;
	CUdeviceptr			m_kds_dst;
	CUdeviceptr			m_nullptr = 0UL;
	CUresult			rc;
	cl_int				outer_depth = pgjoin->outer_depth;
	cl_int				grid_sz;
	cl_int				block_sz;
	void			   *kern_args[5];
	void			   *last_suspend = NULL;
	cl_int				retval;

	/* sanity checks */
	Assert(!pgjoin->pds_src);
	Assert(pds_dst->kds.format == KDS_FORMAT_ROW);
	Assert(outer_depth > 0 && outer_depth <= gjs->num_rels);

	/* Co-location of the outer join map */
	gpujoinColocateOuterJoinMaps(&gjs->gts, cuda_module);

	/* Lookup GPU kernel function */
	rc = cuModuleGetFunction(&kern_gpujoin_main,
							 cuda_module,
							 "kern_gpujoin_right_outer");
	if (rc != CUDA_SUCCESS)
		werror("failed on cuModuleGetFunction: %s", errorText(rc));

	/*
	 * KERNEL_FUNCTION(void)
	 * gpujoin_right_outer(kern_gpujoin *kgjoin,
	 *                     kern_multirels *kmrels,
	 *                     cl_int outer_depth,
	 *                     kern_data_store *kds_dst,
	 *                     kern_parambuf *kparams_gpreagg)
	 */
	rc = gpuOptimalBlockSize(&grid_sz,
							 &block_sz,
							 kern_gpujoin_main,
							 CU_DEVICE_PER_THREAD,
							 0, sizeof(cl_int));
	if (rc != CUDA_SUCCESS)
		werror("failed on gpuOptimalBlockSize: %s", errorText(rc));
resume_kernel:
	m_kds_dst = (CUdeviceptr)&pds_dst->kds;
	kern_args[0] = &m_kgjoin;
	kern_args[1] = &gjs->m_kmrels;
	kern_args[2] = &outer_depth;
	kern_args[3] = &m_kds_dst;
	kern_args[4] = &m_nullptr;

	rc = cuLaunchKernel(kern_gpujoin_main,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						sizeof(cl_int) * block_sz,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuLaunchKernel: %s", errorText(rc));

	rc = cuEventRecord(CU_EVENT0_PER_THREAD, CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuEventRecord: %s", errorText(rc));

	/* Point of synchronization */
	rc = cuEventSynchronize(CU_EVENT0_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuEventSynchronize: %s", errorText(rc));

	memcpy(&pgjoin->task.kerror,
		   &pgjoin->kern.kerror, sizeof(kern_errorbuf));
	if (pgjoin->task.kerror.errcode == ERRCODE_STROM_SUCCESS)
	{
		if (pgjoin->kern.suspend_count > 0)
		{
			CHECK_WORKER_TERMINATION();

			gpujoin_throw_partial_result(pgjoin);
			pds_dst = pgjoin->pds_dst;	/* buffer renew */

			pgjoin->kern.suspend_count = 0;
			pgjoin->kern.resume_context = true;
			if (!last_suspend)
				last_suspend = alloca(pgjoin->kern.suspend_size);
			memcpy(last_suspend,
				   KERN_GPUJOIN_SUSPEND_CONTEXT(&pgjoin->kern, 0),
				   pgjoin->kern.suspend_size);
			goto resume_kernel;
		}
		gpujoinUpdateRunTimeStat(&gjs->gts, &pgjoin->kern);
		/* return task if any result rows */
		retval = (pds_dst->kds.nitems > 0 ? 0 : -1);
	}
	else if (pgstrom_cpu_fallback_enabled &&
			 (pgjoin->task.kerror.errcode & ERRCODE_FLAGS_CPU_FALLBACK) != 0)
	{
		memset(&pgjoin->task.kerror, 0, sizeof(kern_errorbuf));
		pgjoin->task.cpu_fallback = true;
		pgjoin->kern.resume_context = (last_suspend != NULL);
		if (last_suspend)
		{
			memcpy(KERN_GPUJOIN_SUSPEND_CONTEXT(&pgjoin->kern, 0),
				   last_suspend,
				   pgjoin->kern.suspend_size);
		}
		retval = 0;
	}
	else
	{
		/* raise an error */
		retval = 0;
	}
	return retval;
}

int
gpujoin_process_task(GpuTask *gtask, CUmodule cuda_module)
{
	GpuJoinTask *pgjoin = (GpuJoinTask *) gtask;
	int		retval;

	if (pgjoin->pds_src)
		retval = gpujoin_process_inner_join(pgjoin, cuda_module);
	else
		retval = gpujoin_process_right_outer(pgjoin, cuda_module);

	return retval;
}

/* ================================================================
 *
 * Routines to preload inner relations (heap/hash)
 *
 * ================================================================
 */

/*
 * calculation of the hash-value
 */
static cl_uint
get_tuple_hashvalue(innerState *istate,
					bool is_inner_hashkeys,
					TupleTableSlot *slot,
					bool *p_is_null_keys)
{
	ExprContext	   *econtext = istate->econtext;
	cl_uint			hash;
	List		   *hash_keys_list;
	ListCell	   *lc;
	bool			is_null_keys = true;

	if (is_inner_hashkeys)
	{
		hash_keys_list = istate->hash_inner_keys;
		econtext->ecxt_innertuple = slot;
	}
	else
	{
		hash_keys_list = istate->hash_outer_keys;
		econtext->ecxt_scantuple = slot;
	}

	/* calculation of a hash value of this entry */
	hash = 0xffffffffU;
	foreach (lc, hash_keys_list)
	{
		ExprState	   *clause = lfirst(lc);
		devtype_info   *dtype;
		Datum			datum;
		bool			isnull;

#if PG_VERSION_NUM < 100000
		datum = ExecEvalExpr(clause, istate->econtext, &isnull, NULL);
#else
	    datum = ExecEvalExpr(clause, istate->econtext, &isnull);
#endif
		if (isnull)
			continue;
		is_null_keys = false;	/* key contains at least a valid value */

		dtype = pgstrom_devtype_lookup(exprType((Node *)clause->expr));
		Assert(dtype != NULL);

		hash ^= dtype->hash_func(dtype, datum);
	}
	hash ^= 0xffffffffU;

	*p_is_null_keys = is_null_keys;

	return hash;
}

/*
 * gpujoin_expand_inner_kds
 */
static kern_data_store *
gpujoin_expand_inner_kds(dsm_segment *seg, size_t kds_offset)
{
	kern_data_store *kds;
	size_t		new_dsmlen;
	size_t		kds_length;
	size_t		shift;
	char	   *new_kmrels;
	cl_uint	   *row_index;
	cl_int		i;

	/* check current size */
	kds = (kern_data_store *)
		((char *)dsm_segment_address(seg) + kds_offset);
	if (kds->length >= 0x100000000)
		elog(ERROR, "GpuJoin: inner hash table larger than 4GB is not supported right now (nitems=%u, usage=%u)", kds->nitems, kds->usage);

	new_dsmlen = TYPEALIGN(BLCKSZ, (3 * dsm_segment_map_length(seg)) / 2);
	new_kmrels = dsm_resize(seg, new_dsmlen);
	kds = (kern_data_store *)(new_kmrels + kds_offset);
	row_index = KERN_DATA_STORE_ROWINDEX(kds);
	kds_length = Min(new_dsmlen - kds_offset, 0x400000000);	/* up to 16GB */
	shift = kds_length - kds->length;
	Assert(shift == MAXALIGN(shift));
	if (kds->nitems > 0)
	{
		size_t	curr_usage = __kds_unpack(kds->usage);

		Assert(curr_usage > 0);
		memmove((char *)kds + kds_length  - curr_usage,	/* new pos */
				(char *)kds + kds->length - curr_usage,	/* old pos */
				curr_usage);
		for (i=0; i < kds->nitems; i++)
			row_index[i] += __kds_packed(shift);
	}
	kds->length = kds_length;
	return kds;
}

/*
 * gpujoin_compaction_inner_kds - close the hole between row-index/hash-slots
 * and heap-tuples in the KDS tail.
 */
static void
gpujoin_compaction_inner_kds(kern_data_store *kds_in)
{
	size_t		curr_usage = __kds_unpack(kds_in->usage);
	size_t		front_sz;
	size_t		shift;
	cl_uint	   *row_index;
	cl_uint		i;

	Assert(kds_in->format == KDS_FORMAT_HASH ||
		   kds_in->nslots == 0);
	front_sz = KERN_DATA_STORE_HEAD_LENGTH(kds_in) +
		STROMALIGN(sizeof(cl_uint) * kds_in->nitems) +
		STROMALIGN(sizeof(cl_uint) * kds_in->nslots);
	Assert(front_sz == MAXALIGN(front_sz));
	Assert(front_sz + curr_usage <= kds_in->length);
	shift = kds_in->length - (front_sz + curr_usage);
	if (shift > 32 * 1024)	/* close the hole larger than 32KB */
	{
		memmove((char *)kds_in + front_sz,
				(char *)kds_in + kds_in->length - curr_usage,
				curr_usage);
		row_index = KERN_DATA_STORE_ROWINDEX(kds_in);
		for (i=0; i < kds_in->nitems; i++)
			row_index[i] -= __kds_packed(shift);
		kds_in->length = front_sz + curr_usage;
	}
}

/*
 * gpujoin_inner_hash_preload
 *
 * Preload inner relation to the data store with hash-format, for hash-
 * join execution.
 */
static void
gpujoin_inner_hash_preload(innerState *istate,
						   dsm_segment *seg,
						   kern_data_store *kds_hash,
						   size_t kds_offset)
{
	TupleTableSlot *scan_slot;
	cl_uint		   *row_index;
	cl_uint		   *hash_slot;
	cl_uint			i, j;
	pg_crc32		hash;
	bool			is_null_keys;

	for (;;)
	{
		scan_slot = ExecProcNode(istate->state);
		if (TupIsNull(scan_slot))
			break;

		(void)ExecFetchSlotTuple(scan_slot);
		hash = get_tuple_hashvalue(istate, true, scan_slot,
								   &is_null_keys);
		/*
		 * If join keys are NULLs, it is obvious that inner tuple shall not
		 * match with outer tuples. Unless it is not referenced in outer join,
		 * we don't need to keep this tuple in the 
		 */
		if (is_null_keys && (istate->join_type == JOIN_INNER ||
							 istate->join_type == JOIN_LEFT))
			continue;

		while (!KDS_insert_hashitem(kds_hash, scan_slot, hash))
			kds_hash = gpujoin_expand_inner_kds(seg, kds_offset);
	}
	kds_hash->nslots = __KDS_NSLOTS(kds_hash->nitems);
	gpujoin_compaction_inner_kds(kds_hash);
	/* construction of the hash table */
	row_index = KERN_DATA_STORE_ROWINDEX(kds_hash);
	hash_slot = KERN_DATA_STORE_HASHSLOT(kds_hash);
	memset(hash_slot, 0, sizeof(cl_uint) * kds_hash->nslots);
	for (i=0; i < kds_hash->nitems; i++)
	{
		kern_hashitem  *khitem = (kern_hashitem *)
			((char *)kds_hash
			 + __kds_unpack(row_index[i])
			 - offsetof(kern_hashitem, t));
		Assert(khitem->rowid == i);
		j = khitem->hash % kds_hash->nslots;
		khitem->next = hash_slot[j];
		hash_slot[j] = __kds_packed((char *)khitem -
									(char *)kds_hash);
	}
}

/*
 * gpujoin_inner_heap_preload
 *
 * Preload inner relation to the data store with row-format, for nested-
 * loop execution.
 */
static void
gpujoin_inner_heap_preload(innerState *istate,
						   dsm_segment *seg,
						   kern_data_store *kds_heap,
						   size_t kds_offset)
{
	PlanState	   *scan_ps = istate->state;
	TupleTableSlot *scan_slot;

	for (;;)
	{
		scan_slot = ExecProcNode(scan_ps);
		if (TupIsNull(scan_slot))
			break;
		(void)ExecFetchSlotTuple(scan_slot);
		while (!KDS_insert_tuple(kds_heap, scan_slot))
			kds_heap = gpujoin_expand_inner_kds(seg, kds_offset);
	}
	Assert(kds_heap->nslots == 0);
	gpujoin_compaction_inner_kds(kds_heap);
	if (kds_heap->length > (size_t)UINT_MAX)
		elog(ERROR, "GpuJoin: inner heap table larger than 4GB is not supported right now (%zu bytes)", kds_heap->length);		
}

/*
 * gpujoin_inner_preload
 *
 * It preload inner relation to the DSM buffer once.
 */
static bool
__gpujoin_inner_preload(GpuJoinState *gjs, bool preload_multi_gpu)
{
	GpuContext	   *gcontext = gjs->gts.gcontext;
	GpuJoinSharedState *gj_sstate = gjs->gj_sstate;
	kern_multirels *h_kmrels;
	kern_data_store *kds;
	dsm_segment	   *seg;
	int				i, num_rels = gjs->num_rels;
	int				dindex_min;
	int				dindex_max;
	size_t			ojmaps_usage = 0;
	size_t			kmrels_usage = 0;
	size_t			required;
	bool			result = true;

	Assert(!IsParallelWorker());
	gjs->m_kmrels_array = MemoryContextAllocZero(CurTransactionContext,
										numDevAttrs * sizeof(CUdeviceptr));
	gjs->m_kmrels_gcontext = MemoryContextAllocZero(CurTransactionContext,
										numDevAttrs * sizeof(GpuContext *));
	seg = dsm_create(pgstrom_chunk_size(), 0);
	h_kmrels = dsm_segment_address(seg);
	kmrels_usage = STROMALIGN(offsetof(kern_multirels, chunks[num_rels]));
	memset(h_kmrels, 0, kmrels_usage);

	/*
	 * Load inner relations
	 */
	for (i=0; i < num_rels; i++)
	{
		innerState	   *istate = &gjs->inners[i];
		PlanState	   *scan_ps = istate->state;
		TupleTableSlot *ps_slot = scan_ps->ps_ResultTupleSlot;
		TupleDesc		ps_desc = ps_slot->tts_tupleDescriptor;
		kern_data_store *kds;
		size_t			dsm_length;
		size_t			kds_length;
		size_t			kds_head_sz;

		/* expand DSM on demand */
		dsm_length = dsm_segment_map_length(seg);
		kds_head_sz = KDS_calculateHeadSize(ps_desc, false);
		while (kmrels_usage + kds_head_sz > dsm_length)
		{
			h_kmrels = dsm_resize(seg, TYPEALIGN(BLCKSZ, (3*dsm_length)/2));
			dsm_length = dsm_segment_map_length(seg);
		}
		kds = (kern_data_store *)((char *)h_kmrels + kmrels_usage);
		kds_length = Min(dsm_length - kmrels_usage, 0x100000000L);
		init_kernel_data_store(kds,
							   ps_desc,
							   kds_length,
							   (istate->hash_inner_keys != NIL
								? KDS_FORMAT_HASH
								: KDS_FORMAT_ROW),
							   UINT_MAX,
							   false);
		h_kmrels->chunks[i].chunk_offset = kmrels_usage;
		if (istate->hash_inner_keys != NIL)
			gpujoin_inner_hash_preload(istate, seg, kds, kmrels_usage);
		else
			gpujoin_inner_heap_preload(istate, seg, kds, kmrels_usage);

		/* NOTE: gpujoin_inner_xxxx_preload may expand and remap segment */
		h_kmrels = dsm_segment_address(seg);
		kds = (kern_data_store *)((char *)h_kmrels + kmrels_usage);

		if (!istate->hash_outer_keys)
			h_kmrels->chunks[i].is_nestloop = true;
		if (istate->join_type == JOIN_RIGHT ||
			istate->join_type == JOIN_FULL)
		{
			h_kmrels->chunks[i].right_outer = true;
			h_kmrels->chunks[i].ojmap_offset = ojmaps_usage;
			ojmaps_usage += STROMALIGN(kds->nitems);
		}
		if (istate->join_type == JOIN_LEFT ||
			istate->join_type == JOIN_FULL)
		{
			h_kmrels->chunks[i].left_outer = true;
		}
		kmrels_usage += STROMALIGN(kds->length);
	}
	Assert(kmrels_usage <= dsm_segment_map_length(seg));
	h_kmrels->kmrels_length = kmrels_usage;
	h_kmrels->ojmaps_length = ojmaps_usage;
	h_kmrels->cuda_dindex = numDevAttrs;	/* host side */
	h_kmrels->nrels = num_rels;

	/*
	 * NOTE: Special optimization case. In case when any chunk has no items,
	 * and all deeper level is inner join, it is obvious no tuples shall be
	 * produced in this GpuJoin. We can omit outer relation load that shall
	 * be eventually dropped.
	 */
	for (i=num_rels; i > 0; i--)
	{
		kds = KERN_MULTIRELS_INNER_KDS(h_kmrels, i);
		/* outer join can produce something from empty */
		if (gjs->inners[i-1].join_type != JOIN_INNER)
			break;
		if (kds->nitems == 0)
		{
			result = false;
			goto skip_device_malloc;
		}
	}
	required = h_kmrels->kmrels_length + ojmaps_usage * (numDevAttrs + 1);

	/* expand DSM if outer-join map requires more */
	if (required > dsm_segment_map_length(seg))
		h_kmrels = dsm_resize(seg, required);
	memset((char *)h_kmrels + h_kmrels->kmrels_length,
		   0,
		   ojmaps_usage * (numDevAttrs + 1));
	/*
	 * Allocation of GPU device memory for each device if needed.
	 */
	dindex_min = (!preload_multi_gpu ? gcontext->cuda_dindex : 0);
	dindex_max = (!preload_multi_gpu ? gcontext->cuda_dindex : numDevAttrs-1);
	for (i = dindex_min; i <= dindex_max; i++)
	{
		GpuContext *__gcontext;
		CUdeviceptr	m_deviceptr;
		CUresult	rc;

		if (i == gcontext->cuda_dindex)
			__gcontext = GetGpuContext(gcontext);
		else
			__gcontext = AllocGpuContext(i, false, true, false);

		rc = gpuMemAllocDev(__gcontext,
							&m_deviceptr,
							required,
							&gj_sstate->pergpu[i].m_handle);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on gpuMemAllocDev: %s", errorText(rc));
		if (i == gcontext->cuda_dindex)
			gjs->m_kmrels = m_deviceptr;
		gjs->m_kmrels_array[i] = m_deviceptr;
		gjs->m_kmrels_gcontext[i] = __gcontext;

		GPUCONTEXT_PUSH(__gcontext);
		rc = cuMemcpyHtoD(m_deviceptr, h_kmrels, required);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoD: %s", errorText(rc));
		rc = cuMemsetD32(m_deviceptr + offsetof(kern_multirels,
												cuda_dindex),
						 (unsigned int)i,
						 1);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemsetD32: %s", errorText(rc));
		GPUCONTEXT_POP(__gcontext);
	}
skip_device_malloc:
	gj_sstate->kmrels_handle = dsm_segment_handle(seg);
	gjs->seg_kmrels = seg;

	return result;
}

bool
GpuJoinInnerPreload(GpuTaskState *gts, CUdeviceptr *p_m_kmrels)
{
	GpuJoinState   *gjs = (GpuJoinState *) gts;
	GpuContext	   *gcontext = gjs->gts.gcontext;
	GpuJoinSharedState *gj_sstate;
	GpuJoinSiblingState *sibling = gjs->sibling;
	int				i, dindex = gcontext->cuda_dindex;
	bool			preload_multi_gpu = false;
	uint32			preload_done;
	CUdeviceptr		m_deviceptr;
	CUresult		rc;

	Assert(pgstrom_planstate_is_gpujoin(&gts->css.ss.ps));
	if (gjs->gj_sstate)
		preload_multi_gpu = true;
	else
	{
		createGpuJoinSharedState(gjs, NULL, NULL);
		OverWriteSharedStateIfPartitionLeaf(gjs);
		preload_multi_gpu = (sibling != NULL);
	}
	gj_sstate = gjs->gj_sstate;

	/*
	 * Quick exit, if inner-preload already done
	 */
	if (gjs->seg_kmrels)
	{
		if (p_m_kmrels)
			*p_m_kmrels = gjs->m_kmrels;
		return (gjs->seg_kmrels != (CUdeviceptr) 0UL);
	}
	pg_atomic_add_fetch_u32(&gj_sstate->pergpu[dindex].pg_nworkers, 1);
	pg_atomic_add_fetch_u32(&gj_sstate->pg_nworkers, 1);

	/*
	 * In case of partition-wise GpuJoin, other sibling GpuJoin might already
	 * load or attach inner hash/heap buffer. If so, we don't need to load
	 * same buffer again. Just re-use the inner buffer.
	 */
	if (sibling && sibling->seg_kmrels)
	{
		if (sibling->pergpu[dindex].m_kmrels == (CUdeviceptr) 0UL)
		{
			Assert(sibling->pergpu[dindex].gcontext == NULL);
			rc = gpuIpcOpenMemHandle(gcontext,
									 &m_deviceptr,
									 gj_sstate->pergpu[dindex].m_handle,
									 CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on gpuIpcOpenMemHandle: %s",
					 errorText(rc));
			sibling->pergpu[dindex].m_kmrels = m_deviceptr;
			sibling->pergpu[dindex].gcontext = GetGpuContext(gcontext);
		}
		gjs->seg_kmrels = sibling->seg_kmrels;
		gjs->m_kmrels = sibling->pergpu[dindex].m_kmrels;
		sibling->n_actives++;
		if (p_m_kmrels)
			*p_m_kmrels = gjs->m_kmrels;
		return (gjs->seg_kmrels != (CUdeviceptr) 0UL);
	}

	/*
	 * Preload inner hash/heap buffer, or wait for completion
	 */
	ResetLatch(&MyProc->procLatch);
	while ((preload_done = pg_atomic_read_u32(&gj_sstate->preload_done)) == 0)
	{
		if (!IsParallelWorker())
		{
			/* master process is responsible for inner preloading */
			if (__gpujoin_inner_preload(gjs, preload_multi_gpu))
			{
				preload_done = 1;	/* valid inner buffer was loaded */
				if (sibling)
				{
					Assert(sibling->n_actives == 0);
					sibling->seg_kmrels = gjs->seg_kmrels;
					for (i=0; i < numDevAttrs; i++)
					{
						sibling->pergpu[i].m_kmrels
							= gjs->m_kmrels_array[i];
						sibling->pergpu[i].gcontext
							= gjs->m_kmrels_gcontext[i];
					}
					sibling->n_actives = 1;
				}
			}
			else
				preload_done = INT_MAX;	/* no need to run GpuJoin */
			pg_atomic_write_u32(&gj_sstate->preload_done, preload_done);

			/* wake up parallel workers, if any */
			if (gjs->gts.pcxt)
			{
				ParallelContext *pcxt = gjs->gts.pcxt;
				pid_t		pid;
				int			i;

				for (i=0; i < pcxt->nworkers_launched; i++)
				{
					if (GetBackgroundWorkerPid(pcxt->worker[i].bgwhandle,
											   &pid) == BGWH_STARTED)
						ProcSendSignal(pid);
				}
			}

			if (preload_done == 1)
			{
				if (p_m_kmrels)
					*p_m_kmrels = gjs->m_kmrels;
				return true;
			}
			return false;
		}
		else
		{
			/* wait for the completion of inner preload by the master */
			CHECK_FOR_INTERRUPTS();

			WaitLatch(&MyProc->procLatch,
					  WL_LATCH_SET,
					  -1,
					  PG_WAIT_EXTENSION);
			ResetLatch(&MyProc->procLatch);
		}
	}
	/* the inner buffer ready? */
	if (preload_done != 1)
	{
		gjs->m_kmrels = 0UL;
		if (p_m_kmrels)
			*p_m_kmrels = 0UL;
		return false;
	}

	/* inner DSM segment */
	if (sibling)
	{
		if (sibling->seg_kmrels)
			gjs->seg_kmrels = sibling->seg_kmrels;
		else
		{
			gjs->seg_kmrels = dsm_attach(gj_sstate->kmrels_handle);
			if (!gjs->seg_kmrels)
				elog(ERROR, "could not map dynamic shared memory segment");
			sibling->seg_kmrels = gjs->seg_kmrels;
		}
		sibling->n_actives++;

		if (sibling->pergpu[dindex].m_kmrels != (CUdeviceptr) 0UL)
		{
			gjs->m_kmrels = sibling->pergpu[dindex].m_kmrels;
			Assert(sibling->pergpu[dindex].gcontext != NULL);
		}
		else
		{
			rc = gpuIpcOpenMemHandle(gcontext,
									 &m_deviceptr,
									 gj_sstate->pergpu[dindex].m_handle,
									 CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on gpuIpcOpenMemHandle: %s",
					 errorText(rc));
			gjs->m_kmrels = m_deviceptr;
			sibling->pergpu[dindex].m_kmrels = m_deviceptr;
			sibling->pergpu[dindex].gcontext = GetGpuContext(gcontext);
		}
	}
	else
	{
		gjs->seg_kmrels = dsm_attach(gj_sstate->kmrels_handle);
		if (!gjs->seg_kmrels)
			elog(ERROR, "could not map dynamic shared memory segment");

		rc = gpuIpcOpenMemHandle(gcontext,
								 &m_deviceptr,
								 gj_sstate->pergpu[dindex].m_handle,
								 CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on gpuIpcOpenMemHandle: %s", errorText(rc));
		gjs->m_kmrels = m_deviceptr;
	}

	if (p_m_kmrels)
		*p_m_kmrels = m_deviceptr;
	return true;
}

/*
 * GpuJoinInnerUnload
 */
void
GpuJoinInnerUnload(GpuTaskState *gts, bool is_rescan)
{
	GpuJoinState   *gjs = (GpuJoinState *) gts;
	GpuJoinSharedState *gj_sstate = gjs->gj_sstate;
	GpuContext	   *gcontext = gjs->gts.gcontext;
	GpuJoinSiblingState *sibling = gjs->sibling;
	cl_int			i;
	CUresult		rc;

	if (!gj_sstate || !gjs->seg_kmrels)
		return;

	if (IsParallelWorker())
	{
		if (!sibling)
		{
			rc = gpuIpcCloseMemHandle(gcontext, gjs->m_kmrels);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on gpuIpcCloseMemHandle: %s",
					 errorText(rc));
			dsm_detach(gjs->seg_kmrels);
		}
		else if (--sibling->n_actives == 0)
		{
			for (i=0; i < numDevAttrs; i++)
			{
				if (sibling->pergpu[i].m_kmrels == (CUdeviceptr) 0UL)
				{
					Assert(sibling->pergpu[i].gcontext == NULL);
					continue;
				}
				rc = gpuIpcCloseMemHandle(sibling->pergpu[i].gcontext,
										  sibling->pergpu[i].m_kmrels);
				if (rc != CUDA_SUCCESS)
					elog(ERROR, "failed on gpuIpcCloseMemHandle: %s",
						 errorText(rc));
				PutGpuContext(sibling->pergpu[i].gcontext);
				sibling->pergpu[i].gcontext = NULL;
				sibling->pergpu[i].m_kmrels = (CUdeviceptr) 0UL;
			}
			Assert(gjs->seg_kmrels == sibling->seg_kmrels);
			dsm_detach(sibling->seg_kmrels);
			gjs->seg_kmrels = NULL;
			sibling->seg_kmrels = 0;
		}
		else
		{
			Assert(gjs->seg_kmrels == sibling->seg_kmrels);
			gjs->seg_kmrels = NULL;
		}
	}
	else if (!sibling || --sibling->n_actives == 0)
	{
		/* Reset GpuJoinSharedState, if rescan */
		if (is_rescan)
		{
			gj_sstate->kmrels_handle	= UINT_MAX;	/* DSM */
			pg_atomic_init_u32(&gj_sstate->needs_colocation,
							   numDevAttrs > 1 ? 1 : 0);
			pg_atomic_init_u32(&gj_sstate->preload_done, 0);
			pg_atomic_init_u32(&gj_sstate->pg_nworkers, 0);
			memset(gj_sstate->pergpu, 0,
				   offsetof(GpuJoinSharedState, pergpu[numDevAttrs]) -
				   offsetof(GpuJoinSharedState, pergpu[0]));
		}

		/* Release device memory */
		if (gjs->m_kmrels_array)
		{
			for (i=0; i < numDevAttrs; i++)
			{
				GpuContext *__gcontext = gjs->m_kmrels_gcontext[i];

				if (gjs->m_kmrels_array[i] == 0UL)
					continue;

				rc = gpuMemFree(__gcontext, gjs->m_kmrels_array[i]);
				if (rc != CUDA_SUCCESS)
					elog(ERROR, "failed on gpuMemFree: %s", errorText(rc));
				PutGpuContext(__gcontext);
			}
			pfree(gjs->m_kmrels_array);
			pfree(gjs->m_kmrels_gcontext);
			gjs->m_kmrels_array = NULL;
			gjs->m_kmrels_gcontext = NULL;
		}
		/* Also release DSM */
		dsm_detach(gjs->seg_kmrels);
		/* Clear sibling, if any */
		if (sibling)
		{
			sibling->seg_kmrels = NULL;
			memset(sibling->pergpu, 0,
				   offsetof(GpuJoinSiblingState, pergpu[numDevAttrs]) -
				   offsetof(GpuJoinSiblingState, pergpu[0]));
		}
	}
	gjs->curr_outer_depth = -1;
	gjs->m_kmrels = 0UL;
	gjs->seg_kmrels = NULL;
}

/*
 * createGpuJoinSharedState
 *
 * It construct an empty inner multi-relations buffer. It can be shared with
 * multiple backends, and referenced by CPU/GPU.
 */
static void
createGpuJoinSharedState(GpuJoinState *gjs,
						 ParallelContext *pcxt,
						 void *dsm_addr)
{
	EState	   *estate = gjs->gts.css.ss.ps.state;
	GpuJoinSharedState *gj_sstate;
	GpuJoinRuntimeStat *gj_rtstat;
	size_t		sstate_len;
	size_t		rtstat_len;
	size_t		ss_length;

	Assert(!IsParallelWorker());
	sstate_len = MAXALIGN(offsetof(GpuJoinSharedState,
								   pergpu[numDevAttrs]));
	rtstat_len = MAXALIGN(offsetof(GpuJoinRuntimeStat,
								   jstat[gjs->num_rels + 1]));
	ss_length = sstate_len + rtstat_len;
	if (dsm_addr)
		gj_sstate = dsm_addr;
	else
		gj_sstate = MemoryContextAlloc(estate->es_query_cxt, ss_length);
	memset(gj_sstate, 0, ss_length);
	gj_sstate->ss_handle = (pcxt ? dsm_segment_handle(pcxt->seg) : UINT_MAX);
	gj_sstate->ss_length = ss_length;
	gj_sstate->offset_runtime_stat = sstate_len;
	gj_sstate->masterLatch = MyLatch;
	gj_sstate->kmrels_handle = UINT_MAX;	/* to be set later */
	pg_atomic_init_u32(&gj_sstate->needs_colocation,
					   numDevAttrs > 1 ? 1 : 0);
	pg_atomic_init_u32(&gj_sstate->preload_done, 0);
	pg_atomic_init_u32(&gj_sstate->pg_nworkers, 0);

	gj_rtstat = GPUJOIN_RUNTIME_STAT(gj_sstate);
	SpinLockInit(&gj_rtstat->c.lock);
#if PG_VERSION_NUM < 100000
	/* Because of restrictions at PG9.6, see createGpuScanSharedState */
	if (dsm_addr)
	{
		gj_rtstat = MemoryContextAllocZero(estate->es_query_cxt, rtstat_len);
		SpinLockInit(&gj_rtstat->c.lock);
	}
#endif
	gjs->gj_sstate = gj_sstate;
	gjs->gj_rtstat = gj_rtstat;
}

/*
 * gpujoinHasRightOuterJoin
 */
bool
gpujoinHasRightOuterJoin(GpuTaskState *gts)
{
	GpuJoinState   *gjs = (GpuJoinState *) gts;
	kern_multirels *kmrels = dsm_segment_address(gjs->seg_kmrels);

	return (kmrels->ojmaps_length > 0);
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
#if PG_VERSION_NUM >= 100000
	/* turn on/off partition wise gpujoin */
	DefineCustomBoolVariable("pg_strom.enable_partitionwise_gpujoin",
							 "(EXPERIMENTAL) Enables partition wise GpuJoin",
							 NULL,
							 &enable_partitionwise_gpujoin,
							 true,
							 PGC_USERSET,
                             GUC_NOT_IN_SAMPLE,
                             NULL, NULL, NULL);
#else
	enable_partitionwise_gpujoin = false;
#endif
	/* setup path methods */
	gpujoin_path_methods.CustomName				= "GpuJoin";
	gpujoin_path_methods.PlanCustomPath			= PlanGpuJoinPath;

	/* setup plan methods */
	gpujoin_plan_methods.CustomName				= "GpuJoin";
	gpujoin_plan_methods.CreateCustomScanState	= gpujoin_create_scan_state;
	RegisterCustomScanMethods(&gpujoin_plan_methods);

	/* setup exec methods */
	gpujoin_exec_methods.CustomName				= "GpuJoin";
	gpujoin_exec_methods.BeginCustomScan		= ExecInitGpuJoin;
	gpujoin_exec_methods.ExecCustomScan			= ExecGpuJoin;
	gpujoin_exec_methods.EndCustomScan			= ExecEndGpuJoin;
	gpujoin_exec_methods.ReScanCustomScan		= ExecReScanGpuJoin;
	gpujoin_exec_methods.MarkPosCustomScan		= NULL;
	gpujoin_exec_methods.RestrPosCustomScan		= NULL;
	gpujoin_exec_methods.EstimateDSMCustomScan  = ExecGpuJoinEstimateDSM;
	gpujoin_exec_methods.InitializeDSMCustomScan = ExecGpuJoinInitDSM;
	gpujoin_exec_methods.InitializeWorkerCustomScan = ExecGpuJoinInitWorker;
#if PG_VERSION_NUM >= 100000
	gpujoin_exec_methods.ReInitializeDSMCustomScan = ExecGpuJoinReInitializeDSM;
	gpujoin_exec_methods.ShutdownCustomScan		= ExecShutdownGpuJoin;
#endif
	gpujoin_exec_methods.ExplainCustomScan		= ExplainGpuJoin;

	/* hook registration */
	set_join_pathlist_next = set_join_pathlist_hook;
	set_join_pathlist_hook = gpujoin_add_join_path;
}
