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
	List		   *host_quals;
	struct {
		JoinType	join_type;		/* one of JOIN_* */
		double		join_nrows;		/* intermediate nrows in this depth */
		Path	   *scan_path;		/* outer scan path */
		List	   *hash_quals;		/* valid quals, if hash-join */
		List	   *join_quals;		/* all the device quals, incl hash_quals */
		Size		ichunk_size;	/* expected inner chunk size */
		int			nloops_minor;	/* # of virtual segment of inner buffer */
		int			nloops_major;	/* # of physical split of inner buffer */
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
	bool		outer_bulkload;
	double		bulkload_density;
	Expr	   *outer_quals;
	double		outer_ratio;
	/* for each depth */
	List	   *nrows_ratio;
	List	   *ichunk_size;
	List	   *join_types;
	List	   *join_quals;
	List	   *nloops_minor;
	List	   *nloops_major;
	List	   *hash_inner_keys;	/* if hash-join */
	List	   *hash_outer_keys;	/* if hash-join */
	List	   *hash_nslots;		/* if hash-join */
	List	   *gnl_shmem_xsize;	/* if nest-loop */
	List	   *gnl_shmem_ysize;	/* if nest-loop */
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
	privs = lappend(privs, makeString(pstrdup(gj_info->kern_source)));
	privs = lappend(privs, makeInteger(gj_info->extra_flags));
	exprs = lappend(exprs, gj_info->used_params);
	privs = lappend(privs, makeInteger(gj_info->outer_bulkload));
	privs = lappend(privs,
					makeInteger(double_as_long(gj_info->bulkload_density)));
	exprs = lappend(exprs, gj_info->outer_quals);
	privs = lappend(privs, makeInteger(double_as_long(gj_info->outer_ratio)));
	/* for each depth */
	privs = lappend(privs, gj_info->nrows_ratio);
	privs = lappend(privs, gj_info->ichunk_size);
	privs = lappend(privs, gj_info->join_types);
	exprs = lappend(exprs, gj_info->join_quals);
	privs = lappend(privs, gj_info->nloops_minor);
	privs = lappend(privs, gj_info->nloops_major);
	exprs = lappend(exprs, gj_info->hash_inner_keys);
	exprs = lappend(exprs, gj_info->hash_outer_keys);
	privs = lappend(privs, gj_info->hash_nslots);
	privs = lappend(privs, gj_info->gnl_shmem_xsize);
	privs = lappend(privs, gj_info->gnl_shmem_ysize);

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
	gj_info->outer_bulkload = intVal(list_nth(privs, pindex++));
	gj_info->bulkload_density =
		long_as_double(intVal(list_nth(privs, pindex++)));
	gj_info->outer_quals = list_nth(exprs, eindex++);
	gj_info->outer_ratio = long_as_double(intVal(list_nth(privs, pindex++)));
	/* for each depth */
	gj_info->nrows_ratio = list_nth(privs, pindex++);
	gj_info->ichunk_size = list_nth(privs, pindex++);
	gj_info->join_types = list_nth(privs, pindex++);
    gj_info->join_quals = list_nth(exprs, eindex++);
	gj_info->nloops_minor = list_nth(privs, pindex++);
	gj_info->nloops_major = list_nth(privs, pindex++);
	gj_info->hash_inner_keys = list_nth(exprs, eindex++);
    gj_info->hash_outer_keys = list_nth(exprs, eindex++);
	gj_info->hash_nslots = list_nth(privs, pindex++);
	gj_info->gnl_shmem_xsize = list_nth(privs, pindex++);
	gj_info->gnl_shmem_ysize = list_nth(privs, pindex++);

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

	List			   *pds_list;
	cl_int				pds_index;
	Size				pds_limit;
	Size				consumed;
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
	cl_uint				ichunk_size;
	ExprState		   *join_quals;

	/*
	 * Join properties; only hash-join
	 */
	cl_uint				hash_nslots;
	cl_uint				hgram_shift;
	cl_uint				hgram_curr;
	cl_uint				hgram_width;
	Size			   *hgram_size;
	Size			   *hgram_nitems;
	List			   *hash_outer_keys;
	List			   *hash_inner_keys;
	List			   *hash_keylen;
	List			   *hash_keybyval;
	List			   *hash_keytype;

	/*
	 * Join properties; only nest-loop
	 */
	cl_uint				gnl_shmem_xsize;
	cl_uint				gnl_shmem_ysize;
} innerState;

typedef struct
{
	GpuTaskState	gts;
	/* expressions to be used in fallback path */
	List		   *join_types;
	ExprState	   *outer_quals;
	double			outer_ratio;
	List		   *hash_outer_keys;
	List		   *join_quals;
	/* current window of inner relations */
	struct pgstrom_multirels *curr_pmrels;
	/* format of destination store */
	int				result_format;
	/* buffer population ratio */
	int				result_width;	/* result width for buffer length calc */
	/* supplemental information to ps_tlist  */
	List		   *ps_src_depth;
	List		   *ps_src_resno;
	/* buffer for row materialization  */
	HeapTupleData	curr_tuple;

	/*
	 * The least depth to process RIGHT/FULL OUTER JOIN if any. We shall
	 * generate zero tuples for earlier depths, obviously, so we can omit.
	 * If no OUTER JOIN cases, it shall be initialized to 1.
	 */
	cl_int			outer_join_start_depth;

	/*
	 * Runtime statistics
	 */
	int				num_rels;
	size_t			source_ntasks;	/* number of sampled tasks */
	size_t			source_nitems;	/* number of sampled source items */
	size_t			result_nitems[GPUJOIN_MAX_DEPTH + 1];
	size_t			inner_dma_nums;	/* number of inner DMA calls */
	size_t			inner_dma_size;	/* total length of inner DMA calls */

	/*
	 * Properties of underlying inner relations
	 */
	innerState		inners[FLEXIBLE_ARRAY_MEMBER];
} GpuJoinState;

/*
 * pgstrom_multirels - inner buffer of multiple PDS/KDSs
 */
typedef struct pgstrom_multirels
{
	GpuJoinState   *gjs;		/* GpuJoinState of this buffer */
	Size			head_length;	/* length of the header portion */
	Size			usage_length;	/* length actually in use */
	Size			ojmap_length;	/* length of outer-join map */
	pgstrom_data_store **inner_chunks;	/* array of inner PDS */
	cl_bool			needs_outer_join;	/* true, if OJ is needed */
	cl_int			n_attached;	/* Number of attached tasks */
	cl_int		   *refcnt;		/* Reference counter of each GpuContext */
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
	CUevent			ev_dma_send_start;
	CUevent			ev_dma_send_stop;
	CUevent			ev_kern_join_end;
	CUevent			ev_dma_recv_start;
	CUevent			ev_dma_recv_stop;

	/* number of task retry because of DataStoreNoSpace */
	cl_int			retry_count;

	/*
	 * NOTE: If expected size of the kds_dst is too large (that exceeds
	 * pg_strom.chunk_max_inout_ratio), we split GpuJoin steps multiple
	 * times. In this case, all we reference is kds_src[oitems_base] ...
	 * kds_src[oitems_base + oitems_nums - 1] on the next invocation,
	 * then this GpuJoinTask shall be reused with new oitems_base and
	 * oitems_nums after the result is processed at CPU side.
	 */
	cl_uint			num_threads[GPUJOIN_MAX_DEPTH + 1];
	cl_uint			inner_base[GPUJOIN_MAX_DEPTH + 1];
	cl_uint			inner_size[GPUJOIN_MAX_DEPTH + 1];
	double			inner_ratio;

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
static void multirels_detach_buffer(pgstrom_multirels *pmrels,
									bool may_kick_outer_join,
									const char *caller);
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
pgstrom_path_is_gpujoin(Path *pathnode)
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
__dump_gpujoin_path(StringInfo buf, PlannerInfo *root, Path *pathnode)
{
	RelOptInfo *rel = pathnode->parent;
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

		appendStringInfo(buf, "%s%s",
						 is_first ? "" : ", ",
						 eref->aliasname);
		is_first = false;
	}

	if (rel->reloptkind != RELOPT_BASEREL)
		appendStringInfo(buf, ")");
}

/*
 * estimate_buffersize_gpujoin 
 *
 * Top half of cost_gpujoin - we determine expected buffer consumption.
 * If inner relations buffer is too large, we must split pmrels on
 * preloading. If result is too large, we must split range of inner
 * chunks logically.
 */
static bool
estimate_buffersize_gpujoin(PlannerInfo *root,
							RelOptInfo *joinrel,
							GpuJoinPath *gpath,
							int num_chunks)
{
	Size		inner_limit_sz;
	Size		inner_total_sz;
	Size		prev_nloops_minor;
	Size		curr_nloops_minor;
	Size		largest_chunk_size = 0;
	cl_int		largest_chunk_index = -1;
	Size		largest_growth_ntuples = 0.0;
	cl_int		largest_growth_index = -1;
	Size		buffer_size;
	double		inner_ntuples;
	double		join_ntuples;
	double		prev_ntuples;
	cl_int		ncols;
	cl_int		i, num_rels = gpath->num_rels;

	/* init number of loops */
	for (i=0; i < num_rels; i++)
	{
		gpath->inners[i].nloops_minor = 1;
		gpath->inners[i].nloops_major = 1;
	}

	/*
	 * Estimation: size of multi relational inner buffer
	 */
retry_major:
	prev_nloops_minor = 1;
	largest_chunk_size = 0;
	largest_chunk_index = -1;
	largest_growth_ntuples = 0.0;
	largest_growth_index = -1;

	inner_total_sz = STROMALIGN(offsetof(kern_multirels,
										 chunks[num_rels]));
	prev_ntuples = gpath->outer_path->rows / (double) num_chunks;
	for (i=0; i < num_rels; i++)
	{
		Path	   *inner_path = gpath->inners[i].scan_path;
		RelOptInfo *inner_rel = inner_path->parent;
		Size		chunk_size;
		Size		entry_size;
		Size		hash_nslots;
		Size		num_items;

	retry_minor:
		/* total number of inner nloops until this depth */
		curr_nloops_minor = (prev_nloops_minor *
							 gpath->inners[i].nloops_minor);

		/* force a plausible relation size if no information. */
		inner_ntuples = Max(inner_path->rows *
							pgstrom_chunk_size_margin /
							(double)gpath->inners[i].nloops_major,
							100.0);

		/*
		 * NOTE: RelOptInfo->width is not reliable for base relations 
		 * because this fields shows the length of attributes which
		 * are actually referenced, however, we usually load physical
		 * tuples on the KDS/KHash buffer if base relation.
		 */
		ncols = list_length(inner_rel->reltargetlist);
		entry_size = MAXALIGN(offsetof(HeapTupleHeaderData,
									   t_bits[BITMAPLEN(ncols)]));
		if (inner_rel->reloptkind != RELOPT_BASEREL)
			entry_size += MAXALIGN(inner_rel->width);
		else
		{
			entry_size += MAXALIGN(((double)(BLCKSZ -
											 SizeOfPageHeaderData)
									* inner_rel->pages
									/ Max(inner_rel->tuples, 1.0))
								   - sizeof(ItemIdData)
								   - SizeofHeapTupleHeader);
		}

		if (gpath->inners[i].hash_quals != NIL)
		{
			entry_size += offsetof(kern_hashitem, htup);
			hash_nslots = (Size)(inner_ntuples *
								 pgstrom_chunk_size_margin);
		}
		else
		{
			entry_size += offsetof(kern_tupitem, htup);
			hash_nslots = 0;
		}

		/*
		 * inner chunk size estimation
		 */
		chunk_size = STROMALIGN(offsetof(kern_data_store, colmeta[ncols]))
			+ STROMALIGN(gpath->inners[i].hash_quals != NIL
						 ? sizeof(cl_uint) * hash_nslots
						 : sizeof(cl_uint) * (Size)(inner_ntuples))
			+ STROMALIGN(entry_size * (Size)(inner_ntuples));

		gpath->inners[i].ichunk_size = chunk_size;
		gpath->inners[i].hash_nslots = hash_nslots;

		if (largest_chunk_index < 0 || largest_chunk_size < chunk_size)
		{
			largest_chunk_size = chunk_size;
			largest_chunk_index = i;
		}
		inner_total_sz += chunk_size;

		/*
		 * NOTE: The number of intermediation result of GpuJoin has to
		 * fit pgstrom_chunk_size(). If too large number of rows are
		 * expected, we try to run same chunk multiple times with
		 * smaller inner_size[].
		 */
		join_ntuples = (gpath->inners[i].join_nrows /
						(double)(num_chunks * curr_nloops_minor));
		num_items = (Size)((double)(i+2) * join_ntuples *
						   pgstrom_chunk_size_margin);
		buffer_size = offsetof(kern_gpujoin, kparams)
			+ BLCKSZ	/* alternative of kern_parambuf */
			+ STROMALIGN(offsetof(kern_resultbuf, results[num_items]))
			+ STROMALIGN(offsetof(kern_resultbuf, results[num_items]));
		if (buffer_size > pgstrom_chunk_size())
		{
			Size	nsplit_minor = (buffer_size / pgstrom_chunk_size()) + 1;

			if (nsplit_minor > INT_MAX)
			{
				elog(DEBUG1, "Too large kgjoin {nitems=%zu size=%zu}",
					 num_items, buffer_size);
				/*
				 * NOTE: Heuristically, it is not a reasonable plan to
				 * expect massive amount of intermediation result items.
				 * It will lead very large ammount of minor iteration
				 * for GpuJoin kernel invocations. So, we bail out this
				 * plan immediately.
				 */
				return false;
			}
			gpath->inners[i].nloops_minor *= nsplit_minor;
			goto retry_minor;
		}

		if (largest_growth_index < 0 ||
			join_ntuples - prev_ntuples > largest_growth_ntuples)
		{
			largest_growth_index = i;
			largest_growth_ntuples = join_ntuples - prev_ntuples;
		}
		prev_nloops_minor = curr_nloops_minor;
		prev_ntuples = join_ntuples;
	}

	/*
	 * NOTE:If expected consumption of destination buffer exceeds the
	 * limitation, we logically divide an inner chunk (with largest
	 * growth ratio) and run GpuJoin task multiple times towards same
	 * data set.
	 * At this moment, we cannot determine which result format shall
	 * be used (KDS_FORMAT_ROW or KDS_FORMAT_SLOT), so we adopt the
	 * larger one, for safety.
	 */
	Assert(gpath->inners[num_rels-1].join_nrows == gpath->cpath.path.rows);
	join_ntuples = gpath->cpath.path.rows / (double)(num_chunks *
													 prev_nloops_minor);
	ncols = list_length(joinrel->reltargetlist);
	buffer_size = STROMALIGN(offsetof(kern_data_store, colmeta[ncols]));
	buffer_size += (Max(LONGALIGN((sizeof(Datum) + sizeof(char)) * ncols),
						MAXALIGN(offsetof(kern_tupitem, htup) +
								 joinrel->width) + sizeof(cl_uint))
					* (Size) join_ntuples);
	if (buffer_size > pgstrom_chunk_size_limit())
	{
		Size	nloops_minor = (buffer_size / pgstrom_chunk_size_limit()) + 1;

		if (nloops_minor > INT_MAX)
		{
			elog(DEBUG1, "Too large KDS-Dest {nrooms=%zu size=%zu}",
				 (Size) join_ntuples, (Size) buffer_size);
			return false;
		}
		Assert(largest_growth_index >= 0 &&
			   largest_growth_index < num_rels);
		gpath->inners[largest_growth_index].nloops_minor *= nloops_minor;
		goto retry_major;
	}

	/*
	 * NOTE: If total size of inner multi-relations buffer is out of
	 * range, we have to split inner buffer multiple portions to fit
	 * GPU RAMs. It is a restriction come from H/W capability.
	 */
	inner_limit_sz = gpuMemMaxAllocSize() / 2 - BLCKSZ * num_rels;
	if (inner_total_sz > inner_limit_sz)
	{
		Size	nloops_major = (inner_total_sz / inner_limit_sz) + 1;

		if (nloops_major > INT_MAX)
		{
			elog(DEBUG1, "Too large Inner multirel buffer {size=%zu}",
				 (Size) inner_total_sz);
			return false;
		}
		Assert(largest_chunk_index >= 0 &&
			   largest_chunk_index < num_rels);
		gpath->inners[largest_chunk_index].nloops_major *= nloops_major;
		goto retry_major;
	}
	return true;	/* probably, reasonable plan for buffer usage */
}

/*
 * cost_gpujoin
 *
 * estimation of GpuJoin cost
 */
static bool
cost_gpujoin(PlannerInfo *root,
			 RelOptInfo *joinrel,
			 GpuJoinPath *gpath,
			 Relids required_outer,
			 bool support_bulkload)
{
	Path	   *outer_path = gpath->outer_path;
	cl_uint		num_chunks = estimate_num_chunks(outer_path);
	Cost		startup_cost;
	Cost		run_cost;
	Cost		run_cost_per_chunk;
	Cost		startup_delay;
	QualCost   *join_cost;
	Size		inner_total_sz = 0;
	double		chunk_ntuples;
	double		total_nloops_minor = 1.0;	/* loops by kds_dst overflow */
	double		total_nloops_major = 1.0;	/* loops by pmrels overflow */
	int			i, num_rels = gpath->num_rels;

	/*
	 * Estimation of inner / destination buffer consumption
	 */
	if (!estimate_buffersize_gpujoin(root, joinrel, gpath, num_chunks))
		return false;

	for (i=0; i < num_rels; i++)
	{
		total_nloops_major *= (double)gpath->inners[i].nloops_major;
		total_nloops_minor *= (double)gpath->inners[i].nloops_minor;
	}

	/*
	 * Minimum cost comes from outer-path
	 */
	startup_cost = pgstrom_gpu_setup_cost + outer_path->startup_cost;
	run_cost = outer_path->total_cost - outer_path->startup_cost;
	run_cost_per_chunk = 0.0;
	subtract_tuplecost_if_bulkload(&run_cost, outer_path);

	/*
	 * Cost of per-tuple evaluation
	 */
	join_cost = palloc0(sizeof(QualCost) * num_rels);
	for (i=0; i < num_rels; i++)
	{
		cost_qual_eval(&join_cost[i], gpath->inners[i].join_quals, root);
		join_cost[i].per_tuple *= (pgstrom_gpu_operator_cost /
								   cpu_operator_cost);
	}

	/*
	 * Cost for each depth
	 */
	chunk_ntuples = gpath->outer_path->rows / (double) num_chunks;
	for (i=0; i < num_rels; i++)
	{
		Path	   *scan_path = gpath->inners[i].scan_path;

		/* cost to load all the tuples from inner-path */
		startup_cost += scan_path->total_cost;

		/* cost for join_qual startup */
		startup_cost += join_cost[i].startup;

		/*
		 * cost to evaluate join qualifiers according to
		 * the GpuJoin logic
		 */
		if (gpath->inners[i].hash_quals != NIL)
		{
			/*
			 * GpuHashJoin - It computes hash-value of inner tuples by CPU,
			 * but outer tuples by GPU, then it evaluates join-qualifiers
			 * for each items on inner hash table by GPU.
			 */
			List	   *hash_quals = gpath->inners[i].hash_quals;
			cl_uint		num_hashkeys = list_length(hash_quals);
			cl_uint		hash_nslots = gpath->inners[i].hash_nslots;
			double		hash_nsteps = scan_path->rows / (double) hash_nslots;

			/* cost to compute inner hash value by CPU */
			startup_cost += (cpu_operator_cost * num_hashkeys *
							 scan_path->rows);
			/* cost to compute hash value by GPU */
			run_cost_per_chunk = (pgstrom_gpu_operator_cost *
								  num_hashkeys *
								  chunk_ntuples);
			/* cost to evaluate join qualifiers */
			run_cost_per_chunk = (join_cost[i].per_tuple *
								  chunk_ntuples *
								  Max(hash_nsteps, 1.0));
		}
		else
		{
			/*
			 * GpuNestLoop - It evaluates join-qual for each pair of outer
			 * and inner tuples. So, its run_cost is usually higher than
			 * GpuHashJoin.
			 */
			double		inner_ntuples = scan_path->rows
				/ ((double) gpath->inners[i].nloops_major *
				   (double) gpath->inners[i].nloops_minor);

			/* cost to load inner heap tuples by CPU */
			startup_cost += cpu_tuple_cost * scan_path->rows;

			/* cost to evaluate join qualifiers */
			run_cost_per_chunk += (join_cost[i].per_tuple *
								   chunk_ntuples *
								   clamp_row_est(inner_ntuples));
		}
		/* number of outer items on the next depth */
		chunk_ntuples = (gpath->inners[i].join_nrows /
						 ((double) num_chunks *
						  (double) gpath->inners[i].nloops_minor));

		/* consider inner chunk size to be sent over DMA */
		inner_total_sz += gpath->inners[i].ichunk_size;
	}
	/* total GPU execution cost */
	run_cost += (run_cost_per_chunk *
				 (double) num_chunks *
				 (double) total_nloops_minor);
	run_cost *= total_nloops_major;

	/* cost to send inner buffer; assume 25% of kernel kick will take DMA */
	run_cost += ((double)(inner_total_sz / pgstrom_chunk_size())
				 * pgstrom_gpu_dma_cost
				 * ((double)num_chunks *
					total_nloops_minor *
					total_nloops_major) * 0.25);
	/*
	 * delay to fetch the first tuple
	 */
	startup_delay = run_cost * (1.0 / (double)(num_chunks));

	/*
	 * cost of final materialization, but GPU does projection
	 */
	run_cost += cpu_tuple_cost * gpath->cpath.path.rows;

	/*
	 * Put cost value on the gpath.
	 */
	gpath->cpath.path.startup_cost = startup_cost + startup_delay;
	gpath->cpath.path.total_cost = startup_cost + run_cost;

	/*
	 * NOTE: If very large number of rows are estimated, it may cause
	 * overflow of variables, then makes nearly negative infinite cost
	 * even though the plan is very bad.
	 * At this moment, we put assertion to detect it.
	 */
	Assert(gpath->cpath.path.startup_cost >= 0.0 &&
		   gpath->cpath.path.total_cost >= 0.0);

	if (add_path_precheck(gpath->cpath.path.parent,
						  gpath->cpath.path.startup_cost,
						  gpath->cpath.path.total_cost,
						  NULL, required_outer))
	{
		/* Dumps candidate GpuJoinPath for debugging */
		if (client_min_messages <= DEBUG1)
		{
			StringInfoData buf;

			initStringInfo(&buf);
			__dump_gpujoin_path(&buf, root, gpath->outer_path);
			for (i=0; i < gpath->num_rels; i++)
			{
				JoinType	join_type = gpath->inners[i].join_type;
				Path	   *inner_path = gpath->inners[i].scan_path;
				bool		is_nestloop = (gpath->inners[i].hash_quals == NIL);

				appendStringInfo(&buf, " %s%s ",
								 join_type == JOIN_FULL ? "F" :
								 join_type == JOIN_LEFT ? "L" :
								 join_type == JOIN_RIGHT ? "R" : "I",
								 is_nestloop ? "NL" : "HJ");

				__dump_gpujoin_path(&buf, root, inner_path);
			}
			elog(DEBUG1, "GpuJoin: %s Cost=%.2f..%.2f",
				 buf.data,
				 gpath->cpath.path.startup_cost,
				 gpath->cpath.path.total_cost);
			pfree(buf.data);
		}
		return true;
	}
	return false;
}

typedef struct
{
	JoinType	join_type;
	Path	   *inner_path;
	List	   *join_quals;
	List	   *hash_quals;
	double		join_nrows;
} inner_path_item;

static void
create_gpujoin_path(PlannerInfo *root,
					RelOptInfo *joinrel,
					Path *outer_path,
					List *inner_path_list,
					ParamPathInfo *param_info,
					Relids required_outer,
					bool support_bulkload)
{
	GpuJoinPath	   *result;
	cl_int			num_rels = list_length(inner_path_list);
	ListCell	   *lc;
	int				i;

	/*
	 * FIXME: TPC-DS Q4 and Q76 got failed when we allocate gpath
	 * using offsetof(GpuJoinPath, inners[num_rels]), but not happen
	 * if inners[num_rels + 1]. It looks like someone's memory write
	 * violation, however, I cannot find out who does it.
	 * As a workaround we extended length of gpath.
	 * (03-Sep-2015)
	 */
	result = palloc0(offsetof(GpuJoinPath, inners[num_rels + 1]));
	NodeSetTag(result, T_CustomPath);
	result->cpath.path.pathtype = T_CustomScan;
	result->cpath.path.parent = joinrel;
	result->cpath.path.param_info = param_info;	// XXXXXX
	result->cpath.path.pathkeys = NIL;
	result->cpath.path.rows = joinrel->rows;	// XXXXXX
	result->cpath.flags = (support_bulkload ? CUSTOMPATH_SUPPORT_BULKLOAD : 0);
	result->cpath.methods = &gpujoin_path_methods;
	result->outer_path = outer_path;
	result->num_rels = num_rels;
	result->host_quals = NIL;	/* host_quals are no longer supported */

	i = 0;
	foreach (lc, inner_path_list)
	{
		inner_path_item	   *ip_item = lfirst(lc);

		result->inners[i].join_type = ip_item->join_type;
		result->inners[i].join_nrows = ip_item->join_nrows;
		result->inners[i].scan_path = ip_item->inner_path;
		result->inners[i].hash_quals = ip_item->hash_quals;
		result->inners[i].join_quals = ip_item->join_quals;
		result->inners[i].ichunk_size = 0;		/* to be set later */
		result->inners[i].nloops_minor = 1;		/* to be set later */
		result->inners[i].nloops_major = 1;		/* to be set later */
		result->inners[i].hash_nslots = 0;		/* to be set later */
		i++;
	}
	Assert(i == num_rels);

	/*
	 * cost calculation of GpuJoin, then, add this path to the joinrel,
	 * unless its cost is not obviously huge.
	 */
	if (cost_gpujoin(root, joinrel, result, required_outer,
					 support_bulkload))
	{
		List   *custom_paths = list_make1(result->outer_path);

		/* informs planner a list of child pathnodes */
		for (i=0; i < num_rels; i++)
			custom_paths = lappend(custom_paths,
								   result->inners[i].scan_path);
		result->cpath.custom_paths = custom_paths;
		/* add GpuJoin path */
		add_path(joinrel, &result->cpath.path);
	}
	else
		pfree(result);
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
						   RelOptInfo *inputrel)
{
	Path	   *input_path = inputrel->cheapest_total_path;
	Relids		other_relids;
	ListCell   *lc;

	other_relids = bms_difference(joinrel->relids, inputrel->relids);
	if (bms_overlap(PATH_REQ_OUTER(input_path), other_relids))
	{
		input_path = NULL;
		foreach (lc, inputrel->pathlist)
		{
			Path   *curr_path = lfirst(lc);

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
 * gpujoin_calc_required_outer
 *
 * calculation and validation of required_outer for this GpuJoin.
 * Entire logic is described in calc_non_nestloop_required_outer()
 */
static bool
gpujoin_calc_required_outer(PlannerInfo *root,
							RelOptInfo *joinrel,
							Path *outer_path,
							List *inner_path_list,
							Relids param_source_rels,
							Relids *p_required_outer)
{
	Relids		outer_paramrels = PATH_REQ_OUTER(outer_path);
	Relids		required_outer = NULL;
	Relids		extra_lateral_rels = NULL;
	ListCell   *lc1;
	ListCell   *lc2;

	/*
	 * NOTE: Path-nodes that require relations being involved this
	 * GpuJoin shall be dropped at gpujoin_find_cheapest_path
	 */
	Assert(!bms_overlap(outer_paramrels, joinrel->relids));
	required_outer = bms_copy(outer_paramrels);

	/* also, for each inner path-nodes */
	foreach (lc1, inner_path_list)
	{
		inner_path_item	*ip_item = (inner_path_item *) lfirst(lc1);
		Relids	inner_paramrels = PATH_REQ_OUTER(ip_item->inner_path);

		Assert(!bms_overlap(inner_paramrels, joinrel->relids));
		required_outer = bms_add_members(required_outer, inner_paramrels);
	}

	/*
	 * Check extra lateral references by PlaceHolderVars
	 */
	foreach (lc1, root->placeholder_list)
	{
		PlaceHolderInfo *phinfo = (PlaceHolderInfo *) lfirst(lc1);

		/* PHVs without lateral refs can be skipped over quickly */
		if (phinfo->ph_lateral == NULL)
			continue;
		/* PHVs selection that shall be evaluated in this GpuJoin */
		if (!bms_is_subset(phinfo->ph_eval_at, joinrel->relids))
			continue;
		if (bms_is_subset(phinfo->ph_eval_at, outer_path->parent->relids))
			continue;
		foreach (lc2, inner_path_list)
		{
			inner_path_item *ip_item = (inner_path_item *) lfirst(lc2);
			Relids	inner_relids = ip_item->inner_path->parent->relids;

			if (bms_is_subset(phinfo->ph_eval_at, inner_relids))
				break;
		}
		/* Yes, remember its lateral rels */
		if (lc2 == NULL)
			extra_lateral_rels = bms_add_members(extra_lateral_rels,
												 phinfo->ph_lateral);
	}

	/*
	 * Validation checks
	 */
	if (required_outer && !bms_overlap(required_outer, param_source_rels))
		return false;

	*p_required_outer = bms_add_members(required_outer, extra_lateral_rels);

	return true;
}

/*
 * gpujoin_pullup_outer_path
 *
 * Pick up a path-node that shall be pulled up to the next depth
 */
static Path *
gpujoin_pullup_outer_path(RelOptInfo *joinrel, Path *outer_path)
{
	if (IsA(outer_path, NestPath) ||
		IsA(outer_path, HashPath) ||
		IsA(outer_path, MergePath))
	{
		RelOptInfo *outerrel = outer_path->parent;
		JoinPath   *join_path = (JoinPath *) outer_path;
		ListCell   *lc;

		if (!bms_overlap(PATH_REQ_OUTER(join_path->innerjoinpath),
						 join_path->outerjoinpath->parent->relids) &&
			!bms_overlap(PATH_REQ_OUTER(join_path->outerjoinpath),
						 join_path->innerjoinpath->parent->relids))
			return outer_path;
		/*
		 * If supplied outer_path has underlying inner and outer pathnodes
		 * and they are mutually parameralized, it is not a suitable path
		 * to make flatten by GpuJoin
		 */
		outer_path = NULL;
		foreach (lc, outerrel->pathlist)
		{
			Path   *curr_path = (Path *) lfirst(lc);

			if (pgstrom_path_is_gpujoin(curr_path) &&
				(outer_path == NULL ||
				 outer_path->total_cost > curr_path->total_cost))
			{
				outer_path = curr_path;
			}
			else if (IsA(curr_path, NestPath) ||
					 IsA(curr_path, HashPath) ||
					 IsA(curr_path, MergePath))
			{
				JoinPath   *jpath = (JoinPath *) curr_path;

				if (bms_overlap(PATH_REQ_OUTER(jpath->innerjoinpath),
								jpath->outerjoinpath->parent->relids) &&
					bms_overlap(PATH_REQ_OUTER(jpath->outerjoinpath),
								jpath->innerjoinpath->parent->relids) &&
					(outer_path == NULL ||
					 outer_path->total_cost > curr_path->total_cost))
				{
					outer_path = curr_path;
				}
			}
		}
	}
	else if (!pgstrom_path_is_gpujoin(outer_path))
		return NULL;

	return outer_path;
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
	List	   *inner_path_list;
	List	   *restrict_clauses;
	ListCell   *lc;
	Relids		required_outer;
	ParamPathInfo *param_info;
	bool		support_bulkload = true;
	inner_path_item *ip_item;

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

	/*
	 * check bulk-load capability around targetlist of joinrel.
	 * it may be turned off according to the host_quals if any.
	 */
	foreach (lc, joinrel->reltargetlist)
	{
		Expr   *expr = lfirst(lc);

		if (!IsA(expr, Var) && !pgstrom_codegen_available_expression(expr))
		{
			support_bulkload = false;
			break;
		}
	}

	/*
	 * Find out the cheapest inner and outer path from the standpoint
	 * of total_cost, but not be parametalized by other relations in
	 * this GpuJoin
	 */
	outer_path = gpujoin_find_cheapest_path(root, joinrel, outerrel);
	inner_path = gpujoin_find_cheapest_path(root, joinrel, innerrel);
	restrict_clauses = extra->restrictlist;
	ip_item = palloc0(sizeof(inner_path_item));
	ip_item->join_type = jointype;
	ip_item->inner_path = inner_path;
	ip_item->join_quals = NIL;		/* to be set later */
	ip_item->hash_quals = NIL;		/* to be set later */
	ip_item->join_nrows = joinrel->rows;
	inner_path_list = list_make1(ip_item);

	if (!gpujoin_calc_required_outer(root, joinrel,
									 outer_path, inner_path_list,
									 extra->param_source_rels,
									 &required_outer))
		return;

	param_info = get_joinrel_parampathinfo(root,
										   joinrel,
										   outer_path,
										   inner_path,
										   extra->sjinfo,
										   required_outer,
										   &restrict_clauses);
	for (;;)
	{
		List	   *hash_quals = NIL;
		ListCell   *lc;

		/*
		 * Quick exit if number of inner relations out of range
		 */
		if (list_length(inner_path_list) >= GPUJOIN_MAX_DEPTH)
			break;

		/*
		 * Quick exit if unsupported join type
		 */
		if (ip_item->join_type != JOIN_INNER &&
			ip_item->join_type != JOIN_FULL &&
			ip_item->join_type != JOIN_RIGHT &&
			ip_item->join_type != JOIN_LEFT)
			break;

		Assert(outerrel == outer_path->parent);
		Assert(innerrel == ip_item->inner_path->parent);

		/* no benefit to run cross join in GPU device */
		if (!restrict_clauses)
			return;

		/*
		 * Check restrictions of joinrel in this level
		 */
		foreach (lc, restrict_clauses)
		{
			RestrictInfo   *rinfo = (RestrictInfo *) lfirst(lc);

			/*
			 * All the join-clauses must be executable on GPU device.
			 * Even though older version supports HostQuals to be
			 * applied post device join, it leads undesirable (often
			 * unacceptable) growth of the result rows in device join.
			 * So, we simply reject any join that contains host-only
			 * qualifiers.
			 */
			if (!pgstrom_codegen_available_expression(rinfo->clause))
				return;

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
			 * Check if clause has the form "outer op inner" or
			 * "inner op outer". If suitable, we may be able to choose
			 * GpuHashJoin logic. See clause_sides_match_join also.
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
		ip_item->join_quals = restrict_clauses;

		/*
		 * OK, try GpuNestLoop logic
		 */
		if (enable_gpunestloop &&
			(ip_item->join_type == JOIN_INNER ||
			 ip_item->join_type == JOIN_LEFT))
		{
			create_gpujoin_path(root,
								joinrel,
								outer_path,
								inner_path_list,
								param_info,
                                required_outer,
                                support_bulkload);
		}

		/*
		 * OK, let's try GpuHashJoin logic
		 */
		ip_item->hash_quals = hash_quals;
		if (enable_gpuhashjoin && hash_quals != NIL)
		{
			create_gpujoin_path(root,
								joinrel,
								outer_path,
								inner_path_list,
								param_info,
								required_outer,
								support_bulkload);
		}

		/*
		 * Try to pull up outer pathnode if (known) join pathnode
		 * for more relations join on the GPU device at once.
		 */
		outer_path = gpujoin_pullup_outer_path(joinrel, outer_path);
		if (outer_path == NULL)
			break;

		if (pgstrom_path_is_gpujoin(outer_path))
		{
			GpuJoinPath	   *gpath = (GpuJoinPath *) outer_path;
			List		   *inner_path_temp = NIL;
			int				i;

			/* host_quals are no longer supported */
			Assert(gpath->host_quals == NIL);

			for (i = 0; i < gpath->num_rels; i++)
			{
				ip_item = palloc0(sizeof(inner_path_item));
				ip_item->join_type = gpath->inners[i].join_type;
				ip_item->inner_path = gpath->inners[i].scan_path;
				ip_item->join_quals = gpath->inners[i].join_quals;
				ip_item->hash_quals = gpath->inners[i].hash_quals;
				ip_item->join_nrows = gpath->inners[i].join_nrows;

				inner_path_temp = lappend(inner_path_temp, ip_item);
			}
			inner_path_list = list_concat(inner_path_temp, inner_path_list);
			ip_item = linitial(inner_path_list);

			outer_path = gpath->outer_path;
			outerrel = outer_path->parent;
			inner_path = ip_item->inner_path;
			innerrel = inner_path->parent;
			restrict_clauses = ip_item->join_quals;
		}
		else if (IsA(outer_path, NestPath) ||
				 IsA(outer_path, HashPath) ||
				 IsA(outer_path, MergePath))
		{
			JoinPath   *joinpath = (JoinPath *) outer_path;

			outer_path = joinpath->outerjoinpath;
			outerrel = outer_path->parent;
			inner_path = joinpath->innerjoinpath;
			innerrel = inner_path->parent;
			restrict_clauses = joinpath->joinrestrictinfo;

			ip_item = palloc0(sizeof(inner_path_item));
			ip_item->join_type = joinpath->jointype;
			ip_item->inner_path = inner_path;
			ip_item->join_quals = NIL;	/* to be set later */
			ip_item->hash_quals = NIL;	/* to be set later */
			ip_item->join_nrows = outer_path->parent->rows;
			inner_path_list = lcons(ip_item, inner_path_list);
		}
		else
			break;	/* elsewhere, not capable to pull-up */

		/*
		 * XXX - we may need to adjust param_info if new pair of inner and
		 * outer want to reference another external relations.
		 */

		/*
		 * Calculation of required_outer again but suitable to N-way join,
		 * then may give up immediately if unacceptable external references.
		 */
		if (!gpujoin_calc_required_outer(root, joinrel,
										 outer_path, inner_path_list,
										 extra->param_source_rels,
										 &required_outer))
			break;
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
	List		   *custom_plans;
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
		int		i;

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

		/*
		 * Not in the pseudo-scan targetlist, so append this one
		 */
		for (i=0; i <= gpath->num_rels; i++)
		{
			if (i == 0)
				rel = gpath->outer_path->parent;
			else
				rel = gpath->inners[i-1].scan_path->parent;

			if (bms_is_member(varnode->varno, rel->relids))
			{
				Plan   *plan = list_nth(context->custom_plans, i);

				foreach (cell, plan->targetlist)
				{
					TargetEntry *tle = lfirst(cell);

					if (!IsA(tle->expr, Var))
						elog(ERROR, "Bug? unexpected node in tlist: %s",
							 nodeToString(tle->expr));

					if (equal(varnode, tle->expr))
					{
						TargetEntry	   *ps_tle =
							makeTargetEntry((Expr *) copyObject(varnode),
											list_length(context->ps_tlist) + 1,
											NULL,
											context->resjunk);
						context->ps_tlist =
							lappend(context->ps_tlist, ps_tle);
						context->ps_depth =
							lappend_int(context->ps_depth, i);
						context->ps_resno =
							lappend_int(context->ps_resno, tle->resno);

						return false;
					}
				}
				break;
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
						List *targetlist,
						List *host_quals,
						List *custom_plans)
{
	build_ps_tlist_context context;

	memset(&context, 0, sizeof(build_ps_tlist_context));
	context.gpath   = gpath;
	context.custom_plans = custom_plans;
	context.resjunk = false;

	build_pseudo_targetlist_walker((Node *)targetlist, &context);
	build_pseudo_targetlist_walker((Node *)host_quals, &context);

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
					List *custom_plans)
{
	GpuJoinPath	   *gpath = (GpuJoinPath *) best_path;
	GpuJoinInfo		gj_info;
	CustomScan	   *cscan;
	codegen_context	context;
	Plan		   *outer_plan;
	List		   *host_quals;
	ListCell	   *lc;
	double			outer_nrows;
	int				i;

	Assert(gpath->num_rels + 1 == list_length(custom_plans));
	outer_plan = linitial(custom_plans);
	host_quals = extract_actual_clauses(gpath->host_quals, false);

	cscan = makeNode(CustomScan);
	cscan->scan.plan.targetlist = tlist;
	cscan->scan.plan.qual = host_quals;
	cscan->flags = best_path->flags;
	cscan->methods = &gpujoin_plan_methods;
	cscan->custom_plans = list_copy_tail(custom_plans, 1);

	memset(&gj_info, 0, sizeof(GpuJoinInfo));
	gj_info.outer_ratio = 1.0;
	gj_info.num_rels = gpath->num_rels;

	outer_nrows = outer_plan->plan_rows;
	for (i=0; i < gpath->num_rels; i++)
	{
		List	   *hash_inner_keys = NIL;
		List	   *hash_outer_keys = NIL;
		List	   *clauses;
		float		nrows_ratio;

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
		nrows_ratio = gpath->inners[i].join_nrows / outer_nrows;
		gj_info.nrows_ratio = lappend_int(gj_info.nrows_ratio,
										  float_as_int(nrows_ratio));
		gj_info.ichunk_size = lappend_int(gj_info.ichunk_size,
										  gpath->inners[i].ichunk_size);
		gj_info.join_types = lappend_int(gj_info.join_types,
										 gpath->inners[i].join_type);
		clauses = extract_actual_clauses(gpath->inners[i].join_quals, false);
		gj_info.join_quals = lappend(gj_info.join_quals,
									 build_flatten_qualifier(clauses));
		gj_info.nloops_minor = lappend_int(gj_info.nloops_minor,
										   gpath->inners[i].nloops_minor);
		gj_info.nloops_major = lappend_int(gj_info.nloops_major,
										   gpath->inners[i].nloops_major);
		gj_info.hash_inner_keys = lappend(gj_info.hash_inner_keys,
										  hash_inner_keys);
		gj_info.hash_outer_keys = lappend(gj_info.hash_outer_keys,
										  hash_outer_keys);
		gj_info.hash_nslots = lappend_int(gj_info.hash_nslots,
										  gpath->inners[i].hash_nslots);
		outer_nrows = gpath->inners[i].join_nrows;
	}

	/*
	 * Creation of the underlying outer Plan node. In case of SeqScan,
	 * it may make sense to replace it with GpuScan for bulk-loading.
	 */
	if (IsA(outer_plan, SeqScan) || IsA(outer_plan, CustomScan))
	{
		Query	   *parse = root->parse;
		List	   *outer_quals = NIL;
		double		outer_ratio = 1.0;
		Plan	   *alter_plan;

		alter_plan = pgstrom_try_replace_plannode(outer_plan,
												  parse->rtable,
												  &outer_quals,
												  &outer_ratio);
		if (alter_plan)
		{
			gj_info.outer_quals = build_flatten_qualifier(outer_quals);
			gj_info.outer_ratio = outer_ratio;
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
	cscan->custom_scan_tlist = build_pseudo_targetlist(gpath, &gj_info,
													   tlist, host_quals,
													   custom_plans);

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
		/* join nrows */
		appendStringInfo(str, " :join_nrows %zu",
						 (Size)gpath->inners[i].join_nrows);
		/* scan_path */
		appendStringInfo(str, " :scan_path %s",
						 nodeToString(gpath->inners[i].scan_path));
		/* hash_quals */
		appendStringInfo(str, " :hash_quals %s",
						 nodeToString(gpath->inners[i].hash_quals));
		/* join_quals */
		appendStringInfo(str, " :join_quals %s",
						 nodeToString(gpath->inners[i].join_quals));
		/* ichunk_size */
		appendStringInfo(str, " :ichunk_size %zu",
						 gpath->inners[i].ichunk_size);
		/* nloops_minor */
		appendStringInfo(str, " :nloops_minor %d",
						 gpath->inners[i].nloops_minor);
		/* nloops_major */
		appendStringInfo(str, " :nloops_major %d",
						 gpath->inners[i].nloops_major);
		/* hash_nslots */
		appendStringInfo(str, " :hash_nslots %d",
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

	Assert(gj_info->num_rels == list_length(node->custom_plans));
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
	TupleDesc		result_tupdesc = GTS_GET_RESULT_TUPDESC(gjs);
	TupleDesc		scan_tupdesc;
	cl_int			outer_join_start_depth = -1;
	cl_int			i;

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
	 * Re-initialization of scan tuple-descriptor and projection-info,
	 * because commit 1a8a4e5cde2b7755e11bde2ea7897bd650622d3e of
	 * PostgreSQL makes to assign result of ExecTypeFromTL() instead
	 * of ExecCleanTypeFromTL; that leads unnecessary projection.
	 * So, we try to remove junk attributes from the scan-descriptor.
	 */
	scan_tupdesc = ExecCleanTypeFromTL(cscan->custom_scan_tlist, false);
	ExecAssignScanType(&gjs->gts.css.ss, scan_tupdesc);
	ExecAssignScanProjectionInfoWithVarno(&gjs->gts.css.ss, INDEX_VAR);

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
	gjs->outer_ratio = gj_info->outer_ratio;
	gjs->gts.css.ss.ps.qual = (List *)
		ExecInitExpr((Expr *)cscan->scan.plan.qual, ps);

	/* needs to track corresponding columns */
	gjs->ps_src_depth = gj_info->ps_src_depth;
	gjs->ps_src_resno = gj_info->ps_src_resno;

	/*
	 * initialization of child nodes
	 */
	outerPlanState(gjs) = ExecInitNode(outerPlan(cscan), estate, eflags);
	for (i=0; i < gj_info->num_rels; i++)
	{
		Plan	   *inner_plan = list_nth(cscan->custom_plans, i);
		innerState *istate = &gjs->inners[i];
		List	   *hash_inner_keys;
		List	   *hash_outer_keys;
		ListCell   *lc;

		istate->state = ExecInitNode(inner_plan, estate, eflags);
		istate->econtext = CreateExprContext(estate);
		istate->depth = i + 1;
		istate->nbatches_plan = list_nth_int(gj_info->nloops_major, i);
		istate->nbatches_exec =
			((eflags & EXEC_FLAG_EXPLAIN_ONLY) != 0 ? -1 : 0);
		istate->nrows_ratio =
			int_as_float(list_nth_int(gj_info->nrows_ratio, i));
		istate->ichunk_size = list_nth_int(gj_info->ichunk_size, i);
		istate->join_type = (JoinType)list_nth_int(gj_info->join_types, i);

		if (outer_join_start_depth < 0 &&
			(istate->join_type == JOIN_RIGHT ||
			 istate->join_type == JOIN_FULL))
			outer_join_start_depth = istate->depth;

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
			istate->hgram_width = (1U << shift);
			istate->hgram_size = palloc0(sizeof(Size) * istate->hgram_width);
			istate->hgram_nitems = palloc0(sizeof(Size) * istate->hgram_width);
			istate->hgram_shift = sizeof(cl_uint) * BITS_PER_BYTE - shift;
			istate->hgram_curr = 0;
		}
		else
		{
			istate->gnl_shmem_xsize
				= list_nth_int(gj_info->gnl_shmem_xsize, i);
			istate->gnl_shmem_ysize
				= list_nth_int(gj_info->gnl_shmem_ysize, i);
		}
		gjs->gts.css.custom_ps = lappend(gjs->gts.css.custom_ps,
										 gjs->inners[i].state);
	}

	/*
	 * Is bulkload available?
	 */
	gjs->gts.scan_bulk =
		(!pgstrom_bulkload_enabled ? false : gj_info->outer_bulkload);
	gjs->gts.scan_bulk_density = gj_info->bulkload_density;

	/*
	 * Is OUTER RIGHT/FULL JOIN needed?
	 */
	gjs->outer_join_start_depth = Max(outer_join_start_depth, 1);

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
						  t_bits[BITMAPLEN(result_tupdesc->natts)]) +
				 (result_tupdesc->tdhasoid ? sizeof(Oid) : 0)) +
		MAXALIGN(cscan->scan.plan.plan_width);	/* average width */
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

	/*
	 * clean up GpuJoin specific resources
	 */
	if (gjs->curr_pmrels)
		multirels_detach_buffer(gjs->curr_pmrels, false, __FUNCTION__);

	/* then other generic resources */
	pgstrom_release_gputaskstate(&gjs->gts);
}

static void
gpujoin_rescan(CustomScanState *node)
{
	GpuJoinState   *gjs = (GpuJoinState *) node;
	bool			keep_pmrels = true;
	ListCell	   *lc;
	cl_int			i;

	/* clean-up and release any concurrent tasks */
	pgstrom_cleanup_gputaskstate(&gjs->gts);

	/*
	 * NOTE: ExecReScan() does not pay attention on the PlanState within
	 * custom_ps, so we need to assign its chgParam by ourself.
	 */
	if (gjs->gts.css.ss.ps.chgParam != NULL)
	{
		for (i=0; i < gjs->num_rels; i++)
		{
			UpdateChangedParamSet(gjs->inners[i].state,
								  gjs->gts.css.ss.ps.chgParam);
			if (gjs->inners[i].state->chgParam != NULL)
				keep_pmrels = false;
		}
	}

	/*
	 * Rewind the outer relation
	 */
	gjs->gts.scan_done = false;
	gjs->gts.scan_overflow = NULL;
	ExecReScan(outerPlanState(gjs));

	/*
	 * Rewind the inner relation
	 */
	if (!keep_pmrels)
	{
		/* detach previous inner relations buffer */
		if (gjs->curr_pmrels)
		{
			multirels_detach_buffer(gjs->curr_pmrels, false, __FUNCTION__);
			gjs->curr_pmrels = NULL;
		}

		for (i=0; i < gjs->num_rels; i++)
		{
			innerState *istate = &gjs->inners[i];

			/*
			 * If chgParam of subnode is not null then plan will be
			 * re-scanned by next ExecProcNode.
			 */
			if (istate->state->chgParam == NULL)
				ExecReScan(istate->state);

			foreach (lc, istate->pds_list)
				pgstrom_release_data_store((pgstrom_data_store *) lfirst(lc));
			istate->pds_list = NIL;
			istate->pds_index = 0;
			istate->pds_limit = 0;
			istate->consumed = 0;
			istate->ntuples = 0;
			istate->tupstore = NULL;
		}
	}
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

			temp = format_type_with_typemod(exprType((Node *)tle->expr),
											exprTypmod((Node *)tle->expr));
			appendStringInfo(&str, "::%s", temp);
		}
		ExplainPropertyText("Pseudo Scan", str.data, es);
	}

	/* outer bulkload */
	if (!gjs->gts.scan_bulk)
		ExplainPropertyText("Bulkload", "Off", es);
	else
	{
		temp = psprintf("On (density: %.2f%%)",
						100.0 * gjs->gts.scan_bulk_density);
		ExplainPropertyText("Bulkload", temp, es);
	}

	/* outer qualifier if any */
	if (gj_info->outer_quals)
	{
		temp = deparse_expression((Node *)gj_info->outer_quals,
								  context, es->verbose, false);
		if (es->analyze)
			temp = psprintf("%s (%.2f%%, expected %.2f%%)",
							temp,
							100.0 * ((double) gjs->result_nitems[0] /
									 (double) gjs->source_nitems),
							100.0 * gj_info->outer_ratio);
		else
			temp = psprintf("%s (%.2f%%)",
							temp,
							100.0 * gj_info->outer_ratio);
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

		if (hash_outer_key)
		{
			temp = deparse_expression((Node *)hash_outer_key,
                                      context, es->verbose, false);
			appendStringInfo(&str, ", HashKeys: (%s)", temp);
		}
		temp = deparse_expression((Node *)join_qual, context,
								  es->verbose, false);
		appendStringInfo(&str, ", JoinQual: %s", temp);

		snprintf(qlabel, sizeof(qlabel), "Depth% 2d", depth);
		ExplainPropertyText(qlabel, str.data, es);
		resetStringInfo(&str);

		if (es->analyze)
		{
			innerState *istate = &gjs->inners[depth-1];
			size_t		nrows_in  = gjs->result_nitems[depth-1];
			size_t		nrows_out = gjs->result_nitems[depth];
			cl_float	nrows_ratio
				= int_as_float(list_nth_int(gj_info->nrows_ratio, depth - 1));

			appendStringInfo(&str,
							 "Nrows (in:%zu out:%zu, %.2f%% planned %.2f%%)",
							 nrows_in,
							 nrows_out,
							 100.0 * ((double) nrows_out /
									  (double) nrows_in),
							 100.0 * nrows_ratio);
			appendStringInfo(&str,
							 ", KDS-%s (size: %s planned %s, "
							 "nbatches: %u planned %u)",
							 hash_outer_key ? "Hash" : "Heap",
							 bytesz_unitary_format(istate->pds_limit),
							 bytesz_unitary_format(istate->ichunk_size),
							 istate->nbatches_exec,
							 istate->nbatches_plan);
		}
		else
		{
			innerState *istate = &gjs->inners[depth-1];
			cl_float	nrows_ratio
				= int_as_float(list_nth_int(gj_info->nrows_ratio, depth - 1));

			appendStringInfo(&str, "Nrows (in/out: %.2f%%)",
							 100.0 * nrows_ratio);
			appendStringInfo(&str,
							 ", KDS-%s (size: %s, nbatches: %u)",
							 hash_outer_key ? "Hash" : "Heap",
							 bytesz_unitary_format((Size)istate->ichunk_size),
							 istate->nbatches_plan);
		}

		if (es->format == EXPLAIN_FORMAT_TEXT)
		{
			appendStringInfoSpaces(es->str, es->indent * 2);
			appendStringInfo(es->str, "         %s\n", str.data);
		}
		else
		{
			snprintf(qlabel, sizeof(qlabel), "Depth %02d-Ext", depth);
			ExplainPropertyText(qlabel, str.data, es);
		}
		depth++;
	}
	/* inner multirels buffer statistics */
	if (es->analyze)
	{
		if (es->format == EXPLAIN_FORMAT_TEXT)
		{
			resetStringInfo(&str);
			for (depth=1; depth <= gjs->num_rels; depth++)
			{
				innerState *istate = &gjs->inners[depth-1];

				appendStringInfo(&str, "%s(", depth > 1 ? "x" : "");
				foreach (lc1, istate->pds_list)
				{
					pgstrom_data_store *pds = lfirst(lc1);

					if (lc1 != list_head(istate->pds_list))
						appendStringInfo(&str, ", ");
					appendStringInfo(&str, "%s",
									 bytesz_unitary_format(pds->kds->length));
				}
				appendStringInfo(&str, ")");
			}
			appendStringInfo(&str, ", DMA nums: %zu, size: %s",
							 gjs->inner_dma_nums,
							 bytesz_unitary_format(gjs->inner_dma_size));
			ExplainPropertyText("Inner Buffer", str.data, es);
   		}
		else
		{
			ExplainPropertyLong("Num of Inner-DMA", gjs->inner_dma_nums, es);
			ExplainPropertyLong("Size of Inner-DMA", gjs->inner_dma_size, es);
		}
	}
	/* other common field */
	pgstrom_explain_gputaskstate(&gjs->gts, es);
}

/*
 * codegen for:
 * STATIC_FUNCTION(cl_bool)
 * gpujoin_outer_quals(kern_context *kcxt,
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
		"gpujoin_outer_quals(kern_context *kcxt,\n"
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
							   StringInfo v_unaliases,
							   codegen_context *context)
{
	bool		is_nestloop;
	List	   *kern_vars = NIL;
	ListCell   *cell;
	int			depth;
	char	   *param_decl;
	cl_int		gnl_shmem_xsize = 0;
	cl_int		gnl_shmem_ysize = 0;

	Assert(cur_depth > 0 && cur_depth <= gj_info->num_rels);
	is_nestloop = (!list_nth(gj_info->hash_outer_keys, cur_depth - 1));
	Assert(!is_nestloop || v_unaliases != NULL);

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
	param_decl = pgstrom_codegen_param_declarations(context);
	appendStringInfo(source, "%s\n", param_decl);

	/*
	 * variable declarations
	 */
	appendStringInfo(
		source,
		"  HeapTupleHeaderData *htup;\n"
		"  kern_data_store *kds_in;\n"
		"  kern_colmeta *colmeta;\n"
		"  void *datum;\n");

	if (is_nestloop)
	{
		StringInfoData	i_struct;
		StringInfoData	o_struct;

		initStringInfo(&i_struct);
		initStringInfo(&o_struct);

		appendStringInfo(&i_struct, "  struct inner_struct {\n");
		appendStringInfo(&o_struct, "  struct outer_struct {\n");

		foreach (cell, kern_vars)
		{
			Var			   *kernode = lfirst(cell);
			devtype_info   *dtype;
			size_t			field_size;

			dtype = pgstrom_devtype_lookup(kernode->vartype);
			if (!dtype)
				elog(ERROR, "device type \"%s\" not found",
					 format_type_be(kernode->vartype));

			if (dtype->type_byval &&
				dtype->type_length < sizeof(cl_ulong))
				field_size = sizeof(cl_ulong);
			else
				field_size = 2 * sizeof(cl_ulong);

			if (kernode->varno == cur_depth)
				gnl_shmem_xsize += field_size;
			else
				gnl_shmem_ysize += field_size;

			appendStringInfo(
				kernode->varno == cur_depth
				? &i_struct
				: &o_struct,
				"    pg_%s_t KVAR_%u;\n"
				"#define KVAR_%u\t(%s->KVAR_%u)\n",
				/* for var decl */
				dtype->type_name,
				kernode->varoattno,
				/* for alias */
				kernode->varoattno,
				kernode->varno == cur_depth
				? "inner_values"
				: "outer_values",
				kernode->varoattno);
			appendStringInfo(
				v_unaliases,
				"#undef KVAR_%u\n",
				kernode->varoattno);



		}
		appendStringInfo(
			&i_struct,
			"  } *inner_values = (SHARED_WORKMEM(struct inner_struct) +\n"
			"                     get_local_yid());\n");
		appendStringInfo(
			&o_struct,
			"  } *outer_values = ((struct outer_struct *)\n"
			"                     (SHARED_WORKMEM(struct inner_struct) +\n"
			"                      get_local_ysize())) +\n"
			"                     get_local_xid();\n");
		appendStringInfo(source, "%s%s\n",
						 i_struct.data,
						 o_struct.data);
		pfree(i_struct.data);
		pfree(o_struct.data);
	}
	else
	{
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
	}


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
			if (depth >= 0 && is_nestloop)
				appendStringInfo(source, "  }\n\n");

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

			if (is_nestloop)
			{
				appendStringInfo(
					source,
					"  if (get_local_%s() == 0)\n"
					"  {\n",
					depth == cur_depth ? "xid" : "yid");
			}
		}
		appendStringInfo(
			source,
			"  datum = GPUJOIN_REF_DATUM(colmeta,htup,%u);\n"
			"  KVAR_%u = pg_%s_datum_ref(kcxt,datum,false);\n",
			keynode->varattno - 1,
			keynode->varoattno,
			dtype->type_name);
	}
	if (is_nestloop)
		appendStringInfo(source,
						 "  }\n"
						 "  __syncthreads();\n");

	/*
	 * FIXME: We want to add gnl_shmem_?size only when this function
	 * was called to construct gpujoin_join_quals_depth%u().
	 * Is there more graceful way to do?
	 */
	if (v_unaliases != NULL)
	{
		Assert(list_length(gj_info->gnl_shmem_xsize) == cur_depth - 1);
		gj_info->gnl_shmem_xsize = lappend_int(gj_info->gnl_shmem_xsize,
											   gnl_shmem_xsize);
		Assert(list_length(gj_info->gnl_shmem_ysize) == cur_depth - 1);
		gj_info->gnl_shmem_ysize = lappend_int(gj_info->gnl_shmem_ysize,
											   gnl_shmem_ysize);
	}
}

/*
 * codegen for:
 * STATIC_FUNCTION(cl_bool)
 * gpujoin_join_quals_depth%u(kern_context *kcxt,
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
	bool		is_nestloop;
	StringInfoData	v_unaliases;

	is_nestloop = (!list_nth(gj_info->hash_outer_keys, cur_depth - 1));

	initStringInfo(&v_unaliases);

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
		"gpujoin_join_quals_depth%d(kern_context *kcxt,\n"
		"                           kern_data_store *kds,\n"
        "                           kern_multirels *kmrels,\n"
		"                           cl_int *o_buffer,\n"
		"                           HeapTupleHeaderData *i_htup)\n"
		"{\n"
		"  cl_bool result = false;\n",
		cur_depth);
	/*
	 * variable/params declaration & initialization
	 */
	gpujoin_codegen_var_param_decl(source, gj_info, cur_depth,
								   &v_unaliases, context);

	/*
	 * evaluate join qualifier
	 */
	appendStringInfo(
		source,
		"\n"
		"  if (o_buffer != NULL && i_htup != NULL)\n"
		"    result = EVAL(%s);\n"
		"%s"
		"  return result;\n"
		"%s"
		"}\n\n",
		join_code,
		is_nestloop ? "  __syncthreads();\n" : "",
		v_unaliases.data);

	pfree(v_unaliases.data);
}

/*
 * codegen for:
 * STATIC_FUNCTION(cl_uint)
 * gpujoin_hash_value_depth%u(kern_context *kcxt,
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
		"gpujoin_hash_value_depth%u(kern_context *kcxt,\n"
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
	gpujoin_codegen_var_param_decl(source, gj_info, cur_depth, NULL, context);

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
		"  default:\n"
		"    *src_depth = INT_MAX;\n"
		"    *src_colidx = INT_MAX;\n"
		"    break;\n"
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
		"gpujoin_join_quals(kern_context *kcxt,\n"
		"                   kern_data_store *kds,\n"
		"                   kern_multirels *kmrels,\n"
		"                   int depth,\n"
		"                   cl_int *outer_index,\n"
		"                   HeapTupleHeaderData *i_htup)\n"
		"{\n"
		"  switch (depth)\n"
		"  {\n");

	for (depth=1; depth <= gj_info->num_rels; depth++)
	{
		appendStringInfo(
			&source,
			"  case %d:\n"
			"    return gpujoin_join_quals_depth%d(kcxt, kds, kmrels, outer_index, i_htup);\n",
			depth, depth);
	}
	appendStringInfo(
		&source,
		"  default:\n"
		"    STROM_SET_ERROR(&kcxt->e, StromError_SanityCheckViolation);\n"
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
		"gpujoin_hash_value(kern_context *kcxt,\n"
		"                   cl_uint *pg_crc32_table,\n"
		"                   kern_data_store *kds,\n"
		"                   kern_multirels *kmrels,\n"
		"                   cl_int depth,\n"
		"                   cl_int *o_buffer)\n"
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
				"    return gpujoin_hash_value_depth%u(kcxt,pg_crc32_table,kds,kmrels,o_buffer);\n",
				depth, depth);
		}
		depth++;
	}
	appendStringInfo(
		&source,
		"  default:\n"
		"    STROM_SET_ERROR(&kcxt->e, StromError_SanityCheckViolation);\n"
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

/*
 * gpujoin_attach_result_buffer 
 *
 * 
 *
 *
 */
static void
gpujoin_attach_result_buffer(GpuJoinState *gjs,
							 pgstrom_gpujoin *pgjoin,
							 cl_uint *inner_base)
{
	GpuContext		   *gcontext = gjs->gts.gcontext;
	kern_gpujoin	   *kgjoin = &pgjoin->kern;
	pgstrom_multirels  *pmrels = pgjoin->pmrels;
	pgstrom_data_store *pds_src = pgjoin->pds_src;
	pgstrom_data_store *pds_dst = pgjoin->pds_dst;
	kern_resultbuf	   *kresults_in;
	TupleTableSlot	   *tupslot = gjs->gts.css.ss.ss_ScanTupleSlot;
	TupleDesc			tupdesc = tupslot->tts_tupleDescriptor;
	cl_int				ncols = tupdesc->natts;
	cl_uint				num_threads[GPUJOIN_MAX_DEPTH + 1];
	cl_uint				inner_size[GPUJOIN_MAX_DEPTH + 1];
	double				outer_plan_nrows;
	double				ntuples;
	double				ntuples_next;
	Size				kgjoin_length;
	Size				total_items;
	Size				max_items;
	Size				dst_nrooms;
	cl_int				start_depth;
	cl_int				depth;
	cl_int				largest_depth;
	double				largest_growth;
	double				plan_ratio;
	double				exec_ratio;
	double				merge_ratio;
	double				inner_ratio;
	double				inner_ratio_total;

	outer_plan_nrows = outerPlanState(gjs)->plan->plan_rows;

	/*
	 * If valid inner_base[] is given, we set starting position here.
	 * Elsewhere, default starting position will be set.
	 */
	if (inner_base)
		memcpy(pgjoin->inner_base, inner_base, sizeof(pgjoin->inner_base));
	else
		memset(pgjoin->inner_base, 0, sizeof(pgjoin->inner_base));

	/*
	 * If NoDataSpace error retry, we use the previous size as starting
	 * point of inner reduction. Elsewhere, we try to assign maximum
	 * available inner size.
	 */
	if (pds_dst)
		memcpy(inner_size, pgjoin->inner_size, sizeof(pgjoin->inner_size));
	else
	{
		for (depth=1; depth <= gjs->num_rels; depth++)
		{
			pgstrom_data_store *pds_in = pmrels->inner_chunks[depth - 1];
			inner_size[depth-1] = (pds_in->kds->nitems -
								   pgjoin->inner_base[depth-1]);
		}
	}

retry:
	max_items = 0;
	largest_depth = 1;
	largest_growth = -1.0;
	inner_ratio_total = 1.0;

	/* num_threads[] depends on ntuples of intermediation result */
	memset(num_threads, 0, sizeof(num_threads));

	/* Estimate number of input rows that servive outer_quals */
	if (pds_src != NULL)
	{
		if (pds_dst)
		{
			/* If DataStoreNoSpace error, last result the most exact
			 * information. So just pick up result of depth==0.
			 */
			ntuples = (double) pgjoin->kern.result_nitems[0];
		}
		else if (!gjs->outer_quals)
		{
			/* If no outer quals, ntuples should not be filtered.
			 * It is obviously same as kds_src->nitems.
			 */
			ntuples = (double) pds_src->kds->nitems;
		}
		else if (gjs->source_nitems == 0)
		{
			/* If no run-time information, all we can rely on is
			 * plan estimation.
			 */
			ntuples = gjs->outer_ratio * (double) pds_src->kds->nitems;
		}
		else
		{
			/* Elsewhere, we estimate number of input rows using plan
			 * estimation and run-time statistics information. Until
			 * outer scan progress is less than 30%, we merge both of
			 * estimation according to the marge ration.
			 * Once progress goes beyond 30%, we rely on run-time
			 * statistics rather than plan estimation.
			 */
			plan_ratio = gjs->outer_ratio;
			exec_ratio = ((double) gjs->result_nitems[0] /
						  (double) gjs->source_nitems);
			if (gjs->source_nitems >= (size_t)(0.30 * outer_plan_nrows) ||
				gjs->source_ntasks > 20)
				ntuples = exec_ratio * (double) pds_src->kds->nitems;
			else
			{
				merge_ratio = ((double) gjs->source_nitems /
							   (double)(0.30 * outer_plan_nrows));
				ntuples = ((exec_ratio * merge_ratio +
							plan_ratio * (1.0 - merge_ratio)) *
						   (double) pds_src->kds->nitems);
			}
		}
		max_items = (Size)(ntuples * pgstrom_chunk_size_margin);

		/*
		 * NOTE: ntuples is number of input rows at depth==1, never grow up
		 * than original, thus, it never larger than chunk size.
		 */
		Assert((Size) ntuples <= pds_src->kds->nitems);
		start_depth = 1;
	}
	else
	{
		ntuples = 0.0;
		start_depth = gjs->outer_join_start_depth;
	}

	/*
	 * Estimate number of rows generated for each depth.
	 */
	depth = start_depth;
	while (depth <= gjs->num_rels)
	{
		pgstrom_data_store *pds_in = pmrels->inner_chunks[depth - 1];
		innerState		   *istate = &gjs->inners[depth - 1];

		num_threads[depth-1] = Max((cl_uint) ntuples, 1);

		inner_ratio = ((double) inner_size[depth-1] /
					   (double) pds_in->kds->nitems);

		if (pds_dst && depth <= pgjoin->kern.result_valid_until)
		{
			/*
			 * In case of DataStoreNoSpace and retry, the last execution
			 * result tells us exact number of tuples to be generated.
			 */
			ntuples_next = ((double) inner_size[depth - 1] /
							(double) pgjoin->inner_size[depth - 1]) *
				(double) pgjoin->kern.result_nitems[depth];
		}
		else
		{
			if (gjs->result_nitems[depth - 1] == 0)
				ntuples_next = istate->nrows_ratio * inner_ratio * ntuples;
			else
			{
				plan_ratio = istate->nrows_ratio;
				exec_ratio = ((double) gjs->result_nitems[depth] /
							  (double) gjs->result_nitems[depth - 1]);
				if (gjs->source_nitems >= (size_t)(0.30 * outer_plan_nrows) ||
					gjs->source_ntasks > 20)
					ntuples_next = exec_ratio * inner_ratio * ntuples;
				else
				{
					merge_ratio = ((double) gjs->source_nitems /
								   (double)(0.30 * outer_plan_nrows));
					ntuples_next = (exec_ratio * merge_ratio +
									plan_ratio * (1.0 - merge_ratio)) *
						inner_ratio * ntuples;
				}
			}

			/*
			 * OUTER JOIN will add ntuples in this depth
			 */
			if (!pds_src && (istate->join_type == JOIN_RIGHT ||
							 istate->join_type == JOIN_FULL))
			{
				pgstrom_data_store *pds_in = pmrels->inner_chunks[depth - 1];
				double		selectivity;
				double		match_ratio;

				if (pds_in->kds->nitems == 0)
					selectivity = 0.0;	/* obviously, no OUTER JOIN rows */
				else
				{
					/*
					 * XXX - we assume number of unmatched row ratio using:
					 *   1.0 - SQRT(# of result rows) / (# of inner rows)
					 */
					match_ratio = (sqrt((double) gjs->result_nitems[depth])
								   / (double) pds_in->kds->nitems);
					selectivity = 1.0 - Min(1.0, match_ratio);
				}
				selectivity = Max(0.05, selectivity);	/* XXX - at least 5% */
				ntuples_next += selectivity * inner_size[depth-1];
			}
		}

		/*
		 * Check kern_resultbuf[] overflow
		 */
		total_items = (Size)((double)(depth + 1) * ntuples_next *
							 pgstrom_chunk_size_margin);
		kgjoin_length = offsetof(kern_gpujoin, kparams) +
			STROMALIGN(gjs->gts.kern_params->length) +
			STROMALIGN(offsetof(kern_resultbuf, results[total_items])) +
			STROMALIGN(offsetof(kern_resultbuf, results[total_items]));

		/*
		 * If too large, split this inner chunk and retry
		 */
		if (kgjoin_length > pgstrom_chunk_size())
		{
			inner_size[depth-1] /= (kgjoin_length / pgstrom_chunk_size()) + 1;
			continue;
		}
		max_items = Max(max_items, total_items);

		/* save the depth with largest row growth ratio */
		if (ntuples_next - ntuples > largest_growth)
		{
			largest_growth = ntuples_next - ntuples;
			largest_depth = depth;
		}
		inner_ratio_total *= inner_ratio;

		ntuples = ntuples_next;
		depth++;
	}
	Assert(depth == gjs->num_rels + 1);
	num_threads[gjs->num_rels] = Max((cl_uint) ntuples, 1);

	/*
	 * Calculation of the pds_dst length - If we have no run-time information,
	 * all we can do is statistic based estimation. Elsewhere, kds->nitems
	 * will tell us maximum number of row-slot consumption last time.
	 * If StromError_DataStoreNoSpace happen due to lack of kern_resultbuf,
	 * previous kds->nitems may shorter than estimation. So, for safety,
	 * we adopts the larger one.
	 */
	dst_nrooms = (Size)(ntuples * (double) pgstrom_chunk_size_margin);

	if (gjs->result_format == KDS_FORMAT_SLOT)
	{
		Size	length;

		length = (STROMALIGN(offsetof(kern_data_store,
									  colmeta[ncols])) +
				  LONGALIGN((sizeof(Datum) +
							 sizeof(char)) * ncols) * dst_nrooms);

		/* Adjustment if too short or too large */
		if (ncols == 0)
		{
			/* MEMO: Typical usage of ncols == 0 is GpuJoin underlying
			 * COUNT(*) because it does not need to put any contents in
			 * the slot. So, we can allow to increment nitems as long as
			 * 32bit width. :-)
			 */
			dst_nrooms = INT_MAX;
		}
		else if (length < pgstrom_chunk_size() / 4)
		{
			/*
			 * MEMO: If destination buffer size is too small, we doubt
			 * incorrect estimation by planner, so we try to prepare at
			 * least 25% of pgstrom_chunk_size().
			 */
			dst_nrooms = (pgstrom_chunk_size() / 4 -
						  STROMALIGN(offsetof(kern_data_store,
											  colmeta[ncols])))
				/ LONGALIGN((sizeof(Datum) + sizeof(char)) * ncols);
		}
		else if (length > pgstrom_chunk_size_limit())
		{
			cl_int	nsplit = 1 + length / pgstrom_chunk_size_limit();

			Assert(largest_depth > 0 && largest_depth <= gjs->num_rels);
			inner_size[largest_depth-1] /= nsplit;

			if (inner_size[largest_depth-1] < 1)
				elog(ERROR, "Too much growth of result rows");
			goto retry;
		}

		if (!pds_dst)
			pgjoin->pds_dst = pgstrom_create_data_store_slot(gcontext, tupdesc,
															 dst_nrooms,
															 false, 0, NULL);
		else
		{
			/* in case of StromError_DataStoreNoSpace */
			kern_data_store	   *kds_dst = pds_dst->kds;
			Size				new_length;

			new_length = STROMALIGN(offsetof(kern_data_store,
											 colmeta[ncols])) +
				LONGALIGN((sizeof(Datum) +
						   sizeof(char)) * ncols) * dst_nrooms;

			/* needs to allocate KDS again? */
			if (new_length <= kds_dst->length)
			{
				kds_dst->usage = 0;
				kds_dst->nitems = 0;
				kds_dst->nrooms = dst_nrooms;
			}
			else
			{
				kern_data_store *kds_new
					= MemoryContextAlloc(gcontext->memcxt, new_length);
				memcpy(kds_new, kds_dst, KERN_DATA_STORE_HEAD_LENGTH(kds_dst));
				kds_new->hostptr = (hostptr_t) &kds_new->hostptr;
				kds_new->length = new_length;
				kds_new->usage = 0;
				kds_new->nitems = 0;
				kds_new->nrooms = dst_nrooms;
				pds_dst->kds = kds_new;
				pds_dst->kds_length = new_length;
				pfree(kds_dst);
			}
		}
	}
	else if (gjs->result_format == KDS_FORMAT_ROW)
	{
		Size		result_width;
		Size		new_length;

		/*
		 * average length of the result tuple
		 * of course, last execution knows exact length of tuple width
		 */
		if (!pds_dst || pds_dst->kds->usage == 0)
			result_width = gjs->result_width;
		else
		{
			kern_data_store	   *kds_dst = pds_dst->kds;

			Assert(kds_dst->nitems > 0);
			result_width =
				MAXALIGN((Size)(kds_dst->usage -
								KERN_DATA_STORE_HEAD_LENGTH(kds_dst) -
								sizeof(cl_uint) * kds_dst->nitems) /
						 (Size) kds_dst->nitems);
		}

		/* expected buffer length */
		new_length = (STROMALIGN(offsetof(kern_data_store,
										  colmeta[ncols])) +
					  STROMALIGN(sizeof(cl_uint) * dst_nrooms) +
					  MAXALIGN(offsetof(kern_tupitem, htup) +
							   result_width) * dst_nrooms);
		/*
		 * Adjustment if too large or too short
		 */
		if (new_length < pgstrom_chunk_size() / 4)
			new_length = pgstrom_chunk_size() / 4;
		else if (new_length > pgstrom_chunk_size_limit())
		{
			Size		small_nrooms;
			cl_int		nsplit;

			/* maximum number of tuples we can store */
			small_nrooms = (pgstrom_chunk_size_limit() -
							STROMALIGN(offsetof(kern_data_store,
												colmeta[ncols])))
				/ (sizeof(cl_uint) + MAXALIGN(offsetof(kern_tupitem, htup) +
											  result_width));
			Assert(dst_nrooms > small_nrooms);
			nsplit = 1 + dst_nrooms / small_nrooms;

			inner_size[largest_depth-1] /= nsplit;
			if (inner_size[largest_depth-1] < 1)
				elog(ERROR, "Too much growth of result rows");
			goto retry;
		}

		if (!pds_dst)
			pgjoin->pds_dst = pgstrom_create_data_store_row(gcontext, tupdesc,
															new_length, false);
		else
		{
			/* in case of StromError_DataStoreNoSpace */
			kern_data_store	   *kds_dst = pds_dst->kds;

			/* needs to allocate KDS again? */
			if (new_length <= kds_dst->length)
			{
				kds_dst->usage = 0;
				kds_dst->nitems = 0;
				kds_dst->nrooms = INT_MAX;
			}
			else
			{
				kern_data_store	   *kds_new
					= MemoryContextAlloc(gcontext->memcxt, new_length);
				memcpy(kds_new, kds_dst, KERN_DATA_STORE_HEAD_LENGTH(kds_dst));
				kds_new->hostptr = (hostptr_t) &kds_new->hostptr;
				kds_new->length = new_length;
				kds_new->usage = 0;
				kds_new->nitems = 0;
				kds_new->nrooms = INT_MAX;
				pds_dst->kds = kds_new;
				pds_dst->kds_length = new_length;
				pfree(kds_dst);
			}
		}
	}
	else
		elog(ERROR, "Bug? unexpected result format: %d", gjs->result_format);

	/*
	 * Setup kern_gpujoin structure
	 */
	kgjoin_length = offsetof(kern_gpujoin, kparams)
		+ STROMALIGN(gjs->gts.kern_params->length)
		+ STROMALIGN(offsetof(kern_resultbuf, results[max_items]))
		+ STROMALIGN(offsetof(kern_resultbuf, results[max_items]));
	Assert(kgjoin_length <= pgstrom_chunk_size());
	/*
	 * Minimum guarantee of the kern_gpujoin buffer.
	 *
	 * NOTE: we usually have large volatility when GpuJoin tries to filter
	 * many rows, especially hwne row growth ratio is less than 5%, then
	 * it leads unnecessary retry of GpuJoin task.
	 * As long as it is several megabytes, larger kern_gpujoin buffer is
	 * almost harmless because its relevant kern_resultbuf is never sent
	 * or received over DMA.
	 */
	if (kgjoin_length < pgstrom_chunk_size() / 8)
	{
		Size	alt_items
			= ((pgstrom_chunk_size() / 8
				- offsetof(kern_gpujoin, kparams) 
				- STROMALIGN(gjs->gts.kern_params->length)
				- STROMALIGN(offsetof(kern_resultbuf, results[0]))
				- STROMALIGN(offsetof(kern_resultbuf, results[0])))
			   / (2 * sizeof(cl_uint)));
		Assert(alt_items >= max_items);
		kgjoin_length = pgstrom_chunk_size() / 8;
		max_items = alt_items;
	}
	memset(kgjoin, 0, offsetof(kern_gpujoin, kparams));
	kgjoin->kresults_1_offset = (offsetof(kern_gpujoin, kparams) +
								 STROMALIGN(gjs->gts.kern_params->length));
	kgjoin->kresults_2_offset = kgjoin->kresults_1_offset
		+ STROMALIGN(offsetof(kern_resultbuf, results[max_items]));
	kgjoin->kresults_max_space = max_items;
	kgjoin->num_rels = gjs->num_rels;
	kgjoin->start_depth = start_depth;

	/* copies the constant/parameter buffer */
	memcpy(KERN_GPUJOIN_PARAMBUF(kgjoin),
		   gjs->gts.kern_params,
		   gjs->gts.kern_params->length);
	/*
	 * Also, kresults_in of depth==1 has to be initialized preliminary
	 */
	kresults_in = KERN_GPUJOIN_IN_RESULTS(kgjoin, 1);
	memset(kresults_in, 0, offsetof(kern_resultbuf, results[0]));
	kresults_in->nrels = 1;
	kresults_in->nrooms = max_items;
	kresults_in->nitems = 0;

	/* copy inner_size[] and num_threads[] */
	memcpy(pgjoin->inner_size, inner_size, sizeof(pgjoin->inner_size));
	memcpy(pgjoin->num_threads, num_threads, sizeof(pgjoin->num_threads));
	pgjoin->inner_ratio = inner_ratio_total;
}

static GpuTask *
gpujoin_create_task(GpuJoinState *gjs,
					pgstrom_multirels *pmrels,
					pgstrom_data_store *pds_src,
					cl_uint *inner_base)
{
	GpuContext		   *gcontext = gjs->gts.gcontext;
	pgstrom_gpujoin	   *pgjoin;
	Size				pgjoin_head;
	Size				required;

	/*
	 * Allocation of pgstrom_gpujoin task object
	 */
	pgjoin_head = (offsetof(pgstrom_gpujoin, kern) +
				   offsetof(kern_gpujoin, kparams) +
				   STROMALIGN(gjs->gts.kern_params->length));
	required = pgjoin_head + STROMALIGN(offsetof(kern_resultbuf, results[0]));
	pgjoin = MemoryContextAllocZero(gcontext->memcxt, required);
	pgstrom_init_gputask(&gjs->gts, &pgjoin->task);
	pgjoin->pmrels = multirels_attach_buffer(pmrels);
	pgjoin->pds_src = pds_src;
	pgjoin->pds_dst = NULL;		/* to be set later */
	/* inner_base[], inner_size[] and num_threads[] will be set below */
	gpujoin_attach_result_buffer(gjs, pgjoin, inner_base);

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
		pgstrom_multirels *pmrels_new;

		/*
		 * NOTE: gpujoin_inner_preload() has to be called prior to
		 * multirels_detach_buffer() because some inner chunk (PDS)
		 * may be reused on the next loop, thus, refcnt of the PDS
		 * should not be touched to zero.
		 */
		pmrels_new = gpujoin_inner_preload(gjs);
		if (gjs->curr_pmrels)
		{
			Assert(gjs->gts.scan_done);
			multirels_detach_buffer(gjs->curr_pmrels, true, __FUNCTION__);
			gjs->curr_pmrels = NULL;
		}
		if (!pmrels_new)
			return NULL;	/* end of inner multi-relations */
		gjs->curr_pmrels = pmrels_new;

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
		pds = BulkExecProcNode(outer_node);
		if (!pds)
			gjs->gts.scan_done = true;
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

	return gpujoin_create_task(gjs, gjs->curr_pmrels, pds, NULL);
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
		multirels_detach_buffer(pgjoin->pmrels, false, __FUNCTION__);
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
	pgstrom_gpujoin	   *pgjoin = (pgstrom_gpujoin *) gtask;
	GpuJoinState	   *gjs = (GpuJoinState *) gtask->gts;

	if (gjs->gts.pfm_accum.enabled)
	{
		CUDA_EVENT_ELAPSED(pgjoin, time_dma_send,
						   ev_dma_send_start,
						   ev_dma_send_stop);
		CUDA_EVENT_ELAPSED(pgjoin, time_kern_join,
						   ev_dma_send_stop,
						   ev_kern_join_end);
		CUDA_EVENT_ELAPSED(pgjoin, time_kern_proj,
						   ev_kern_join_end,
						   ev_dma_recv_start);
		CUDA_EVENT_ELAPSED(pgjoin, time_dma_recv,
						   ev_dma_recv_start,
						   ev_dma_recv_stop);
		pgstrom_accum_perfmon(&gjs->gts.pfm_accum, &pgjoin->task.pfm);
	}
	gpujoin_cleanup_cuda_resources(pgjoin);

	if (pgjoin->task.kerror.errcode == StromError_Success)
	{
		pgstrom_data_store *pds_src = pgjoin->pds_src;
		pgstrom_multirels  *pmrels = pgjoin->pmrels;
		pgstrom_gpujoin	   *pgjoin_new;
		cl_uint				inner_base[GPUJOIN_MAX_DEPTH + 1];
		cl_int				i;

		/*
		 * Update run-time statistics information according to the number
		 * of rows actually processed by this GpuJoin task.
		 * In case of OUTER JOIN task, we don't count source items because
		 * it is generated as result of unmatched tuples.
		 */
		gjs->source_ntasks++;
		if (pds_src)
			gjs->source_nitems += (Size)
				(pgjoin->inner_ratio * (double) pds_src->kds->nitems);
		for (i=0; i <= pgjoin->kern.num_rels; i++)
			gjs->result_nitems[i] += pgjoin->kern.result_nitems[i];

		/*
		 * Enqueue another GpuJoin taks if completed one was run towards
		 * a piece of inner chunk, and we have to make inner_base[] and
		 * inner_size[] advanced.
		 */
		for (i=gjs->num_rels-1; i >= 0; i--)
		{
			pgstrom_data_store *pds_in = pmrels->inner_chunks[i];
			cl_uint				inner_limit
				= (pds_in->kds->format != KDS_FORMAT_ROW
				   ? pds_in->kds->nitems
				   : pds_in->kds->nslots);

			inner_base[i] = (pgjoin->inner_base[i] + pgjoin->inner_size[i]);
			if (inner_base[i] < inner_limit)
			{
				while (++i <= gjs->num_rels)
					inner_base[i] = 0;
				break;
			}
		}

		if (i > 0)
		{
			/*
			 * XXX - do we need to acquire the source PDS, rather than
			 * explicit detach from pgjoin?
			 */
			pgjoin->pds_src = NULL;

			/*
			 * NOTE: The inner buffer (pmrels) shall be attached within
			 * gpujoin_create_task, so don't need to attach it here.
			 */
			pgjoin_new = (pgstrom_gpujoin *)
				gpujoin_create_task(gjs,
									pgjoin->pmrels,
									pds_src,
									inner_base);
			/* add this new task to the pending list */
			SpinLockAcquire(&gjs->gts.lock);
			dlist_push_tail(&gjs->gts.pending_tasks, &pgjoin_new->task.chain);
			gjs->gts.num_pending_tasks++;
			SpinLockRelease(&gjs->gts.lock);
		}
		/*
		 * NOTE: We have to detach inner chunks here, because it may kick
		 * OUTER JOIN task if this context is the last holder of inner
		 * buffer.
		 */
		Assert(pgjoin->pmrels != NULL);
		multirels_detach_buffer(pgjoin->pmrels, true, __FUNCTION__);
		pgjoin->pmrels = NULL;
	}
	else if (pgjoin->task.kerror.errcode == StromError_DataStoreNoSpace)
	{
#ifdef PGSTROM_DEBUG
		/* For debug output */
		kern_data_store	   *kds_dst = pgjoin->pds_dst->kds;
		cl_uint		num_threads[GPUJOIN_MAX_DEPTH + 1];
		cl_uint		inner_base[GPUJOIN_MAX_DEPTH + 1];
		cl_uint		inner_size[GPUJOIN_MAX_DEPTH + 1];
		cl_uint		result_nitems[GPUJOIN_MAX_DEPTH + 1];
		cl_uint		result_valid_until;
		cl_uint		nrooms_old;
		cl_uint		length_old;
		cl_uint		max_space_old;
		cl_int		i;
		double		progress;
		bool		has_resized = false;
		StringInfoData	str;

		memcpy(num_threads, pgjoin->num_threads,
			   sizeof(num_threads));
		memcpy(inner_base, pgjoin->inner_base,
			   sizeof(inner_base));
		memcpy(inner_size, pgjoin->inner_size,
			   sizeof(inner_size));
		memcpy(result_nitems, pgjoin->kern.result_nitems,
			   sizeof(result_nitems));
		result_valid_until = pgjoin->kern.result_valid_until;
		length_old = kds_dst->length;
		nrooms_old = kds_dst->nrooms;
		max_space_old = pgjoin->kern.kresults_max_space;
		progress = 100.0 * ((double) gjs->source_nitems /
							(double) outerPlanState(gjs)->plan->plan_rows);
#endif
		/*
		 * StromError_DataStoreNoSpace indicates either/both of buffers
		 * were smaller than required. So, we expand the buffer or reduce
		 * number of outer tuples, then kick this gputask again.
		 */
		gpujoin_attach_result_buffer(gjs, pgjoin, pgjoin->inner_base);
		pgjoin->retry_count++;

#ifdef PGSTROM_DEBUG
		/*
		 * DEBUG OUTPUT if GpuJoinTask retry happen - to track how many
		 * items are required and actually required.
		 */

		/* kds_dst might be replaced */
		kds_dst = pgjoin->pds_dst->kds;
		initStringInfo(&str);
		if (pgjoin->pds_src)
			appendStringInfo(&str, "src_nitems: %u",
							 pgjoin->pds_src->kds->nitems);
		else
		{
			pgstrom_multirels  *pmrels = pgjoin->pmrels;
			pgstrom_data_store *pds_in =
				pmrels->inner_chunks[gjs->outer_join_start_depth];
			appendStringInfo(&str, "in_nitems: %u", pds_in->kds->nitems);
		}

		if (max_space_old == pgjoin->kern.kresults_max_space)
			appendStringInfo(&str, " max_space: %u ", max_space_old);
		else
		{
			appendStringInfo(&str, " max_space: %u=>%u ",
							 max_space_old, pgjoin->kern.kresults_max_space);
			has_resized = true;
		}

		if (kds_dst->format == KDS_FORMAT_ROW)
		{
			if (length_old == kds_dst->length)
				appendStringInfo(&str, "length: %u", length_old);
			else
			{
				appendStringInfo(&str, "length: %u=>%u",
								 length_old, kds_dst->length);
				has_resized = true;
			}
		}
		else
		{
			if (nrooms_old == kds_dst->nrooms)
				appendStringInfo(&str, "nrooms: %u", nrooms_old);
			else
			{
				appendStringInfo(&str, "nrooms: %u=>%u",
								 nrooms_old, kds_dst->nrooms);
				has_resized = true;
			}
		}
		appendStringInfo(&str, " Nthreads: (");
		for (i=0; i <= gjs->num_rels; i++)
		{
			if (num_threads[i] == pgjoin->num_threads[i])
				appendStringInfo(&str, "%s%u", i > 0 ? ", " : "",
								 pgjoin->num_threads[i]);
			else
			{
				appendStringInfo(&str, "%s%u=>%u", i > 0 ? ", " : "",
								 num_threads[i], pgjoin->num_threads[i]);
				has_resized = true;
			}
		}
		appendStringInfo(&str, ")");

		appendStringInfo(&str, " inners: ");
		for (i=0; i < gjs->num_rels; i++)
		{
			Assert(inner_base[i] == pgjoin->inner_base[i]);
			if (inner_size[i] == pgjoin->inner_size[i])
				appendStringInfo(&str, "%s(%u, %u)", i > 0 ? ", " : "",
								 inner_base[i], inner_size[i]);
			else
			{
				appendStringInfo(&str, "%s(%u, %u=>%u)", i > 0 ? ", " : "",
								 inner_base[i], inner_size[i],
								 pgjoin->inner_size[i]);
				has_resized = true;
			}
		}
		appendStringInfo(&str, " results: [");
		for (i=0; i <= gjs->num_rels; i++)
		{
			if (i <= result_valid_until)
				appendStringInfo(&str, "%s%u", i > 0 ? ", " : "",
								 result_nitems[i]);
			else
				appendStringInfo(&str, "%s*", i > 0 ? ", " : "");
		}
		appendStringInfo(&str, "]");

		elog(has_resized ? NOTICE : ERROR,
			 "GpuJoin(%p) DataStoreNoSpace retry=%d [%.2f%%] %s%s",
			 pgjoin, pgjoin->retry_count, progress, str.data,
			 has_resized ? "" : ", but not resized actually");
		pfree(str.data);
#endif
		/*
		 * OK, chain this task on the pending_tasks queue again
		 *
		 * NOTE: 'false' indicates cuda_control.c that this cb_complete
		 * callback handled this request by itself - we re-entered the
		 * GpuTask on the pending_task queue to execute again.
		 */
		SpinLockAcquire(&gjs->gts.lock);
		dlist_push_head(&gjs->gts.pending_tasks, &pgjoin->task.chain);
		gjs->gts.num_pending_tasks++;
		SpinLockRelease(&gjs->gts.lock);

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

	if (status == CUDA_SUCCESS)
		pgjoin->task.kerror = pgjoin->kern.kerror;
	else
	{
		pgjoin->task.kerror.errcode = status;
		pgjoin->task.kerror.kernel = StromKernel_CudaRuntime;
		pgjoin->task.kerror.lineno = 0;
	}

	/*
	 * Remove from the running_tasks list, then attach it
	 * on the completed_tasks list
	 */
	SpinLockAcquire(&gts->lock);
	dlist_delete(&pgjoin->task.chain);
	gts->num_running_tasks--;

	if (pgjoin->task.kerror.errcode == StromError_Success)
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
	pgstrom_multirels  *pmrels = pgjoin->pmrels;
	pgstrom_data_store *pds_src = pgjoin->pds_src;
	pgstrom_data_store *pds_dst = pgjoin->pds_dst;
	GpuJoinState   *gjs = (GpuJoinState *) pgjoin->task.gts;
	const char	   *kern_proj_name;
	Size			length;
	Size			total_length;
	Size			outer_ntuples;
	size_t			grid_xsize;
	size_t			grid_ysize;
	size_t			block_xsize;
	size_t			block_ysize;
	CUresult		rc;
	void		   *kern_args[10];
	int				depth;
	int				start_depth;

	/*
	 * sanity checks
	 */
	Assert(pds_src == NULL || pds_src->kds->format == KDS_FORMAT_ROW);
	Assert(gjs->outer_join_start_depth >= 1 &&
		   gjs->outer_join_start_depth <= gjs->num_rels);
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
	length = (pgjoin->kern.kresults_2_offset +
			  pgjoin->kern.kresults_2_offset - pgjoin->kern.kresults_1_offset);
	total_length = GPUMEMALIGN(length);
	if (pds_src)
		total_length += GPUMEMALIGN(KERN_DATA_STORE_LENGTH(pds_src->kds));
	total_length += GPUMEMALIGN(KERN_DATA_STORE_LENGTH(pds_dst->kds));

	pgjoin->m_kgjoin = gpuMemAlloc(&pgjoin->task, total_length);
	if (!pgjoin->m_kgjoin)
		goto out_of_resource;

	/*
	 * m_kds_src may be NULL, if OUTER JOIN
	 */
	if (pds_src)
	{
		pgjoin->m_kds_src = pgjoin->m_kgjoin + GPUMEMALIGN(length);
		pgjoin->m_kds_dst = pgjoin->m_kds_src +
			GPUMEMALIGN(KERN_DATA_STORE_LENGTH(pds_src->kds));
	}
	else
	{
		pgjoin->m_kds_src = 0UL;
		pgjoin->m_kds_dst = pgjoin->m_kgjoin + GPUMEMALIGN(length);
	}

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
	multirels_send_buffer(pmrels, &pgjoin->task);
	/* kern_gpujoin + static portion of kern_resultbuf */
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
	if (pds_src)
	{
		length = KERN_DATA_STORE_LENGTH(pds_src->kds);
		rc = cuMemcpyHtoDAsync(pgjoin->m_kds_src,
							   pds_src->kds,
							   length,
							   pgjoin->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		pgjoin->task.pfm.bytes_dma_send += length;
		pgjoin->task.pfm.num_dma_send++;
	}

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
	start_depth = gjs->outer_join_start_depth;
	for (depth = start_depth; depth <= gjs->num_rels; depth++)
	{
		innerState *istate = &gjs->inners[depth - 1];
		JoinType	join_type = istate->join_type;
		bool		is_nestloop = (!istate->hash_outer_keys);
		cl_uint		inner_base = pgjoin->inner_base[depth-1];
		cl_uint		inner_size = pgjoin->inner_size[depth-1];
		size_t		num_threads;

		/*
		 * Launch:
		 * KERNEL_FUNCTION(void)
		 * gpujoin_preparation(kern_gpujoin *kgjoin,
		 *                     kern_data_store *kds,
		 *                     kern_multirels *kmrels,
		 *                     cl_int depth)
		 */
		num_threads = ((depth > 1 || !pds_src) ? 1 : pds_src->kds->nitems);
		pgstrom_compute_workgroup_size(&grid_xsize,
									   &block_xsize,
									   pgjoin->kern_prep,
									   pgjoin->task.cuda_device,
									   false,
									   num_threads,
									   sizeof(kern_errorbuf));
		kern_args[0] = &pgjoin->m_kgjoin;
		kern_args[1] = &pgjoin->m_kds_src;
		kern_args[2] = &pgjoin->m_kmrels;
		kern_args[3] = &depth;

		rc = cuLaunchKernel(pgjoin->kern_prep,
							grid_xsize, 1, 1,
							block_xsize, 1, 1,
							sizeof(kern_errorbuf) * block_xsize,
							pgjoin->task.cuda_stream,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
		pgjoin->task.pfm.num_kern_join++;
		elog(DEBUG2, "CUDA launch %s grid:{%u,1,1}, block:{%u,1,1}",
			 "gpujoin_preparation",
			 (cl_uint)grid_xsize,
			 (cl_uint)block_xsize);
		/*
		 * Main logic of GpuHashJoin or GpuNestLoop
		 */
		if (is_nestloop)
		{
			pgstrom_data_store *pds_in = pmrels->inner_chunks[depth - 1];
			cl_uint		inner_ntuples = pds_in->kds->nitems;
			cl_uint		shmem_size;

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
			 *                       cl_bool *outer_join_map,
			 *                       cl_uint inner_base,
			 *                       cl_uint inner_size)
			 */
			if (pds_src != NULL || depth > gjs->outer_join_start_depth)
			{
				outer_ntuples = pgjoin->num_threads[depth-1];
				pgstrom_compute_workgroup_size_2d(&grid_xsize, &block_xsize,
												  &grid_ysize, &block_ysize,
												  pgjoin->kern_exec_nl,
												  pgjoin->task.cuda_device,
												  outer_ntuples,
												  inner_ntuples,
												  istate->gnl_shmem_xsize,
												  istate->gnl_shmem_ysize,
												  sizeof(kern_errorbuf));
				kern_args[0] = &pgjoin->m_kgjoin;
				kern_args[1] = &pgjoin->m_kds_src;
				kern_args[2] = &pgjoin->m_kmrels;
				kern_args[3] = &depth;
				kern_args[4] = &pgjoin->task.cuda_index;
				kern_args[5] = &pgjoin->m_ojmaps;
				kern_args[6] = &inner_base;
				kern_args[7] = &inner_size;

				shmem_size = Max(sizeof(kern_errorbuf) * (block_xsize *
														  block_ysize),
								 istate->gnl_shmem_xsize * block_xsize +
								 istate->gnl_shmem_ysize * block_ysize);

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
				elog(DEBUG2, "CUDA launch %s grid:{%u,%u,1}, block:{%u,%u,1}",
					 "gpujoin_exec_nestloop",
					 (cl_uint)grid_xsize, (cl_uint)grid_ysize,
					 (cl_uint)block_xsize, (cl_uint)block_ysize);
			}

			/*
			 * Launch:
			 * KERNEL_FUNCTION(void)
			 * gpujoin_leftouter_nestloop(kern_gpujoin *kgjoin,
			 *                            kern_data_store *kds,
			 *                            kern_multirels *kmrels,
			 *                            cl_int depth,
			 *                            cl_uint cuda_index,
			 *                            cl_bool *outer_join_maps,
			 *                            cl_uint inner_base,
			 *                            cl_uint inner_size)
			 */
			if (pds_src == NULL &&
				(join_type == JOIN_RIGHT || join_type == JOIN_FULL))
			{
				Assert(depth >= gjs->outer_join_start_depth);
				/* gather the outer join map, if multi-GPUs environment */
				multirels_colocate_outer_join_maps(pmrels,
												   &pgjoin->task,
												   depth);
				pgstrom_compute_workgroup_size(&grid_xsize,
											   &block_xsize,
											   pgjoin->kern_outer_nl,
											   pgjoin->task.cuda_device,
											   false,
											   inner_size,
											   sizeof(kern_errorbuf));
				kern_args[0] = &pgjoin->m_kgjoin;
				kern_args[1] = &pgjoin->m_kds_src;
				kern_args[2] = &pgjoin->m_kmrels;
				kern_args[3] = &depth;
				kern_args[4] = &pgjoin->task.cuda_index;
				kern_args[5] = &pgjoin->m_ojmaps;
				kern_args[6] = &inner_base;
				kern_args[7] = &inner_size;

				rc = cuLaunchKernel(pgjoin->kern_outer_nl,
									grid_xsize, 1, 1,
									block_xsize, 1, 1,
									sizeof(kern_errorbuf) * block_xsize,
									pgjoin->task.cuda_stream,
									kern_args,
									NULL);
				if (rc != CUDA_SUCCESS)
					elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
				pgjoin->task.pfm.num_kern_join++;

				elog(DEBUG2, "CUDA launch %s grid:{%u,1,1}, block:{%u,1,1}",
					 "gpujoin_leftouter_nestloop",
					 (cl_uint)grid_xsize,
					 (cl_uint)block_xsize);
			}
		}
		else
		{
			/*
			 * Launch:
			 * KERNEL_FUNCTION(void)
			 * gpujoin_exec_hashjoin(kern_gpujoin *kgjoin,
			 *                       kern_data_store *kds,
			 *                       kern_multirels *kmrels,
			 *                       cl_int depth,
			 *                       cl_uint cuda_index,
			 *                       cl_bool *outer_join_map,
			 *                       cl_uint inner_base,
			 *                       cl_uint inner_size)
			 */
			if (pds_src != NULL || depth > gjs->outer_join_start_depth)
			{
				outer_ntuples = pgjoin->num_threads[depth-1];
				Assert(outer_ntuples > 0);
				pgstrom_compute_workgroup_size(&grid_xsize,
											   &block_xsize,
											   pgjoin->kern_exec_hj,
											   pgjoin->task.cuda_device,
											   false,
											   outer_ntuples,
											   sizeof(kern_errorbuf));
				kern_args[0] = &pgjoin->m_kgjoin;
				kern_args[1] = &pgjoin->m_kds_src;
				kern_args[2] = &pgjoin->m_kmrels;
				kern_args[3] = &depth;
				kern_args[4] = &pgjoin->task.cuda_index;
				kern_args[5] = &pgjoin->m_ojmaps;
				kern_args[6] = &inner_base;
				kern_args[7] = &inner_size;

				rc = cuLaunchKernel(pgjoin->kern_exec_hj,
									grid_xsize, 1, 1,
									block_xsize, 1, 1,
									sizeof(kern_errorbuf) * block_xsize,
									pgjoin->task.cuda_stream,
									kern_args,
									NULL);
				if (rc != CUDA_SUCCESS)
					elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
				pgjoin->task.pfm.num_kern_join++;
				elog(DEBUG2, "CUDA launch %s grid:{%u,1,1}, block:{%u,1,1}",
					 "gpujoin_exec_hashjoin",
					 (cl_uint)grid_xsize,
					 (cl_uint)block_xsize);
			}

			/*
			 * Launch:
			 * KERNEL_FUNCTION(void)
			 * gpujoin_leftouter_hashjoin(kern_gpujoin *kgjoin,
			 *                            kern_data_store *kds,
			 *                            kern_multirels *kmrels,
			 *                            cl_int depth,
			 *                            cl_uint cuda_index,
			 *                            cl_bool *outer_join_maps
			 *                            cl_uint inner_base,
			 *                            cl_uint inner_size)
			 */
			if (pds_src == NULL &&
				(join_type == JOIN_RIGHT || join_type == JOIN_FULL))
			{
				pgstrom_data_store *pds_in = pmrels->inner_chunks[depth - 1];

				Assert(depth >= gjs->outer_join_start_depth);
				/* gather the outer join map, if multi-GPUs environment */
				multirels_colocate_outer_join_maps(pgjoin->pmrels,
												   &pgjoin->task, depth);
				pgstrom_compute_workgroup_size(&grid_xsize,
											   &block_xsize,
											   pgjoin->kern_outer_hj,
											   pgjoin->task.cuda_device,
											   false,
											   pds_in->kds->nslots,
											   sizeof(kern_errorbuf));
				kern_args[0] = &pgjoin->m_kgjoin;
				kern_args[1] = &pgjoin->m_kds_src;
				kern_args[2] = &pgjoin->m_kmrels;
				kern_args[3] = &depth;
				kern_args[4] = &pgjoin->task.cuda_index;
				kern_args[5] = &pgjoin->m_ojmaps;
				kern_args[6] = &inner_base;
				kern_args[7] = &inner_size;

				rc = cuLaunchKernel(pgjoin->kern_outer_hj,
									grid_xsize, 1, 1,
									block_xsize, 1, 1,
									sizeof(kern_errorbuf) * block_xsize,
									pgjoin->task.cuda_stream,
									kern_args,
									NULL);
				if (rc != CUDA_SUCCESS)
					elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
				pgjoin->task.pfm.num_kern_join++;

				elog(DEBUG2, "CUDA launch %s grid:{%u,1,1}, block:{%u,1,1}",
					 "gpujoin_leftouter_hashjoin",
					 (cl_uint)grid_xsize,
					 (cl_uint)block_xsize);
			}
		}
	}
	Assert(pgjoin->kern.num_rels == depth - 1);
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
								   pgjoin->num_threads[gjs->num_rels],
								   sizeof(kern_errorbuf));
	kern_args[0] = &pgjoin->m_kgjoin;
	kern_args[1] = &pgjoin->m_kmrels;
	kern_args[2] = &pgjoin->m_kds_src;
	kern_args[3] = &pgjoin->m_kds_dst;

	rc = cuLaunchKernel(pgjoin->kern_proj,
						grid_xsize, 1, 1,
						block_xsize, 1, 1,
						sizeof(kern_errorbuf) * block_xsize,
						pgjoin->task.cuda_stream,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
	pgjoin->task.pfm.num_kern_proj++;

	elog(DEBUG2, "CUDA launch %s grid:{%u,1,1}, block:{%u,1,1}",
		 (pds_dst->kds->format == KDS_FORMAT_ROW
		  ? "gpujoin_projection_row"
		  : "gpujoin_projection_slot"),
		 (cl_uint)grid_xsize,
		 (cl_uint)block_xsize);

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

/*
 * calculation of the hash-value
 */
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
		Datum		value;
		bool		isnull;

		value = ExecEvalExpr(clause, istate->econtext, &isnull, NULL);
		if (isnull)
			continue;

		/* fixup host representation to special internal format. */
		if (keytype == NUMERICOID)
		{
			kern_context	dummy;
			pg_numeric_t	temp;

			/*
			 * FIXME: If NUMERIC value is out of range, we cannot execute
			 * GpuJoin in the kernel space, so needs a fallback routine.
			 */
			temp = pg_numeric_from_varlena(&dummy, (struct varlena *)
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
 * gpujoin_inner_hash_preload_TC
 *
 * It preloads a part of inner relation, within a particular range of
 * hash-values, to the data store with hash-format, for hash-join
 * execution. Its source is preliminary materialized within tuple-store
 * of PostgreSQL.
 */
static void
gpujoin_inner_hash_preload_TS(GpuJoinState *gjs,
							  innerState *istate)
{
	PlanState		   *scan_ps = istate->state;
	TupleTableSlot	   *scan_slot = scan_ps->ps_ResultTupleSlot;
	TupleDesc			scan_desc = scan_slot->tts_tupleDescriptor;
	Tuplestorestate	   *tupstore = istate->tupstore;
	pgstrom_data_store *pds_hash;
	List			   *pds_list = NIL;
	List			   *hash_max_list = NIL;
	Size				curr_size = 0;
	Size				curr_nitems = 0;
	Size				kds_length;
	cl_uint				nslots;
	pg_crc32			hash_min;
	pg_crc32			hash_max;
	ListCell		   *lc1;
	ListCell		   *lc2;
	cl_uint				i;

	/* tuplestore must be built */
	Assert(tupstore != NULL);

	hash_min = 0;
	for (i=0; i < istate->hgram_width; i++)
	{
		Size	next_size = istate->hgram_size[i];
		Size	next_nitems = istate->hgram_nitems[i];

		if (curr_size + next_size > istate->pds_limit)
		{
			if (curr_size == 0)
				elog(ERROR, "Too extreme hash-key distribution");

			nslots = (cl_uint)((double) curr_nitems *
							   pgstrom_chunk_size_margin);
			kds_length = (STROMALIGN(offsetof(kern_data_store,
											 colmeta[scan_desc->natts])) +
						  STROMALIGN(sizeof(cl_uint) * nslots) +
						  curr_size);

			hash_max = i * (1U << istate->hgram_shift) - 1;
			pds_hash = pgstrom_create_data_store_hash(gjs->gts.gcontext,
													  scan_desc,
													  kds_length,
													  nslots,
													  false);
			pds_hash->kds->hash_min = hash_min;
			pds_hash->kds->hash_max = hash_max;

			pds_list = lappend(pds_list, pds_hash);
			hash_max_list = lappend_int(hash_max_list, (int) hash_max);
			/* reset counter */
			hash_min = hash_max + 1;
			curr_size = 0;
			curr_nitems = 0;
		}
		curr_size += next_size;
		curr_nitems += next_nitems;
	}
	/*
	 * The last partitioned chunk
	 */
	nslots = (cl_uint)((double) curr_nitems *
					   pgstrom_chunk_size_margin);
	nslots = Max(nslots, 128);
	kds_length = (STROMALIGN(offsetof(kern_data_store,
									  colmeta[scan_desc->natts])) +
				  STROMALIGN(sizeof(cl_uint) * nslots) +
				  curr_size + BLCKSZ);
	pds_hash = pgstrom_create_data_store_hash(gjs->gts.gcontext,
											  scan_desc,
											  kds_length,
											  nslots,
											  false);
	pds_hash->kds->hash_min = hash_min;
	pds_hash->kds->hash_max = UINT_MAX;
	pds_list = lappend(pds_list, pds_hash);
	hash_max_list = lappend_int(hash_max_list, (int) UINT_MAX);

	/*
	 * Load from the tuplestore
	 */
	while (tuplestore_gettupleslot(tupstore, true, false, scan_slot))
	{
		pg_crc32	hash = get_tuple_hashvalue(istate, scan_slot);

		forboth (lc1, pds_list,
				 lc2, hash_max_list)
		{
			pgstrom_data_store *pds = lfirst(lc1);
			pg_crc32			hash_max = (pg_crc32)lfirst_int(lc2);

			if (hash <= hash_max)
			{
				if (pgstrom_data_store_insert_hashitem(pds, scan_slot, hash))
					break;
				Assert(false);
				elog(ERROR, "Bug? GpuHashJoin Histgram was not correct");
			}
		}
	}

	foreach (lc1, pds_list)
	{
		pgstrom_data_store *pds_in = lfirst(lc1);
		pgstrom_shrink_data_store(pds_in);
	}
	Assert(istate->pds_list == NIL);
	istate->pds_list = pds_list;

	/* no longer tuple-store is needed */
	tuplestore_end(istate->tupstore);
	istate->tupstore = NULL;
}

/*
 * gpujoin_inner_hash_preload
 *
 * Preload inner relation to the data store with hash-format, for hash-
 * join execution.
 */
static bool
gpujoin_inner_hash_preload(GpuJoinState *gjs,
						   innerState *istate,
						   Size *p_total_usage)
{
	PlanState		   *scan_ps = istate->state;
	TupleTableSlot	   *scan_slot;
	TupleDesc			scan_desc;
	HeapTuple			tuple;
	Size				consumption;
	pgstrom_data_store *pds_hash = NULL;
	pg_crc32			hash;
	cl_int				index;

	scan_slot = ExecProcNode(istate->state);
	if (TupIsNull(scan_slot))
	{
		if (istate->tupstore)
			gpujoin_inner_hash_preload_TS(gjs, istate);
		/* put an empty hash table if no rows read */
		if (istate->pds_list == NIL)
		{
			Size	empty_len;

			scan_slot = scan_ps->ps_ResultTupleSlot;
			scan_desc = scan_slot->tts_tupleDescriptor;
			empty_len = (STROMALIGN(offsetof(kern_data_store,
											 colmeta[scan_desc->natts])) +
						 STROMALIGN(sizeof(cl_uint) * 4));
			pds_hash = pgstrom_create_data_store_hash(gjs->gts.gcontext,
													  scan_desc,
													  empty_len,
													  4,
													  false);
			istate->pds_list = list_make1(pds_hash);
		}
		return false;
	}

	scan_desc = scan_slot->tts_tupleDescriptor;
	if (istate->pds_list != NIL)
		pds_hash = (pgstrom_data_store *) llast(istate->pds_list);
	else if (!istate->tupstore)
	{
		Size	ichunk_size = Max(istate->ichunk_size,
								  pgstrom_chunk_size() / 4);
		pds_hash = pgstrom_create_data_store_hash(gjs->gts.gcontext,
												  scan_desc,
												  ichunk_size,
												  istate->hash_nslots,
												  false);
		istate->pds_list = list_make1(pds_hash);
		istate->consumed = KERN_DATA_STORE_HEAD_LENGTH(pds_hash->kds);
	}

	tuple = ExecFetchSlotTuple(scan_slot);
	hash = get_tuple_hashvalue(istate, scan_slot);
	consumption = sizeof(cl_uint) +	/* for hash_slot */
		MAXALIGN(offsetof(kern_hashitem, htup) + tuple->t_len);
	/*
	 * Update Histgram
	 */
	index = (hash >> istate->hgram_shift);
	istate->hgram_size[index] += consumption;
	istate->hgram_nitems[index]++;

	/*
	 * XXX - If join type is LEFT or FULL OUTER, each PDS has to be
	 * strictly partitioned by the hash-value, thus, we saves entire
	 * relation on the tuple-store, then reconstruct PDS later.
	 */
retry:
	if (istate->tupstore)
	{
		tuplestore_puttuple(istate->tupstore, tuple);
		istate->ntuples++;
		istate->consumed += consumption;
		*p_total_usage += consumption;
		return true;
	}

	if (istate->pds_limit > 0 &&
		istate->pds_limit <= istate->consumed + consumption)
	{
		if (istate->join_type == JOIN_INNER ||
			istate->join_type == JOIN_LEFT)
		{
			cl_uint		hash_nslots;

			pgstrom_shrink_data_store(pds_hash);

			hash_nslots = (cl_uint)((double) pds_hash->kds->nitems *
									pgstrom_chunk_size_margin);
			pds_hash = pgstrom_create_data_store_hash(gjs->gts.gcontext,
													  scan_desc,
													  istate->pds_limit,
													  hash_nslots,
													  false);
			istate->pds_list = lappend(istate->pds_list, pds_hash);
			istate->consumed = pds_hash->kds->usage;
		}
		else
		{
			/*
			 * NOTE: If join type requires inner-side is well partitioned
			 * by hash-value, we once needs to move all the entries to
			 * the tuple-store, then reconstruct them as PDS.
			 */
			kern_data_store	   *kds_hash = pds_hash->kds;
			kern_hashitem	   *khitem;
			HeapTupleData		tupData;

			istate->tupstore = tuplestore_begin_heap(false, false, work_mem);
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
			Assert(list_length(istate->pds_list) == 1);
			pgstrom_release_data_store(pds_hash);
			istate->pds_list = NULL;
			goto retry;
		}
	}

	if (!pgstrom_data_store_insert_hashitem(pds_hash, scan_slot, hash))
	{
		cl_uint	nitems_old = pds_hash->kds->nitems;
		cl_uint	nslots_new = (cl_uint)(pgstrom_chunk_size_margin *
									   (double)(2 * nitems_old));
		pgstrom_expand_data_store(gjs->gts.gcontext,
								  pds_hash,
								  2 * pds_hash->kds_length,
								  nslots_new);
		goto retry;
	}
	istate->ntuples++;
	istate->consumed += consumption;
	*p_total_usage += consumption;

	return true;
}

/*
 * gpujoin_inner_heap_preload
 *
 * Preload inner relation to the data store with row-format, for nested-
 * loop execution.
 */
static bool
gpujoin_inner_heap_preload(GpuJoinState *gjs,
						   innerState *istate,
						   Size *p_total_usage)
{
	PlanState	   *scan_ps = istate->state;
	TupleTableSlot *scan_slot;
	TupleDesc		scan_desc;
	HeapTuple		tuple;
	Size			consumption;
	pgstrom_data_store *pds_heap;

	/* fetch next tuple from inner relation */
	scan_slot = ExecProcNode(scan_ps);
	if (TupIsNull(scan_slot))
	{
		/* put an empty heap table if no rows read */
		if (istate->pds_list == NIL)
		{
			Size	empty_len;

			scan_slot = scan_ps->ps_ResultTupleSlot;
			scan_desc = scan_slot->tts_tupleDescriptor;
			empty_len = STROMALIGN(offsetof(kern_data_store,
											colmeta[scan_desc->natts]));
			pds_heap = pgstrom_create_data_store_row(gjs->gts.gcontext,
													 scan_desc,
													 empty_len,
													 false);
			istate->pds_list = list_make1(pds_heap);
		}
		return false;
	}
	scan_desc = scan_slot->tts_tupleDescriptor;

	if (istate->pds_list != NIL)
		pds_heap = (pgstrom_data_store *) llast(istate->pds_list);
	else
	{
		Size	ichunk_size = Max(istate->ichunk_size,
								  pgstrom_chunk_size() / 4);
		pds_heap = pgstrom_create_data_store_row(gjs->gts.gcontext,
												 scan_desc,
												 ichunk_size,
												 false);
		istate->pds_list = list_make1(pds_heap);
		istate->consumed = KERN_DATA_STORE_HEAD_LENGTH(pds_heap->kds);
	}

	tuple = ExecFetchSlotTuple(scan_slot);
	consumption = sizeof(cl_uint) +		/* for offset table */
		LONGALIGN(offsetof(kern_tupitem, htup) + tuple->t_len);

	/*
	 * Switch to the new chunk, if current one exceeds the limitation
	 */
	if (istate->pds_limit > 0 &&
		istate->pds_limit <= istate->consumed + consumption)
	{
		pds_heap = pgstrom_create_data_store_row(gjs->gts.gcontext,
												 scan_desc,
												 pds_heap->kds_length,
												 false);
		istate->pds_list = lappend(istate->pds_list, pds_heap);
		istate->consumed = STROMALIGN(offsetof(kern_data_store,
											   colmeta[scan_desc->natts]));
	}
	istate->consumed += consumption;
	*p_total_usage += consumption;

retry:
	if (!pgstrom_data_store_insert_tuple(pds_heap, scan_slot))
	{
		pgstrom_expand_data_store(gjs->gts.gcontext, pds_heap,
								  2 * pds_heap->kds_length, 0);
		goto retry;
	}
	istate->ntuples++;

	return true;
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
	pgstrom_multirels  *pmrels = NULL;
	int					i, j;
	struct timeval		tv1, tv2;

	PERFMON_BEGIN(&gjs->gts.pfm_accum, &tv1);

	if (!gjs->curr_pmrels)
	{
		innerState	  **istate_buf;
		cl_int			istate_nums = gjs->num_rels;
		Size			total_limit;
		Size			total_usage;
		bool			kmrels_size_fixed = false;

		/*
		 * Half of the max allocatable GPU memory (and minus some margin) is
		 * the current hard limit of the inner relations buffer.
		 */
		total_limit = gpuMemMaxAllocSize() / 2 - BLCKSZ * gjs->num_rels;
		total_usage = STROMALIGN(offsetof(kern_multirels,
										  chunks[gjs->num_rels]));
		istate_buf = palloc0(sizeof(innerState *) * gjs->num_rels);
		for (i=0; i < istate_nums; i++)
			istate_buf[i] = &gjs->inners[i];

		while (istate_nums > 0)
		{
			for (i=0; i < istate_nums; i++)
			{
				innerState *istate = istate_buf[i];

				if (!(istate->hash_inner_keys != NIL
					  ? gpujoin_inner_hash_preload(gjs, istate, &total_usage)
					  : gpujoin_inner_heap_preload(gjs, istate, &total_usage)))
				{
					memmove(istate_buf + i,
							istate_buf + i + 1,
							sizeof(innerState *) * (istate_nums - (i+1)));
					istate_nums--;
					i--;
				}
			}

			if (!kmrels_size_fixed && total_usage >= total_limit)
			{
				/*
				 * XXX - current usage became a limitation, so next call
				 * of gpujoin_inner_XXXX_preload makes second chunk.
				 */
				for (i=0; i < gjs->num_rels; i++)
					gjs->inners[i].pds_limit = gjs->inners[i].consumed;
				kmrels_size_fixed = true;
			}
		}

		/*
		 * XXX - It is ideal case; all the inner chunk can be loaded to
		 * a single multi-relations buffer.
		 */
		if (!kmrels_size_fixed)
		{
			for (i=0; i < gjs->num_rels; i++)
				gjs->inners[i].pds_limit = gjs->inners[i].consumed;
		}
		pfree(istate_buf);

		/*
		 * TODO: we may omit some depths if nitems==0 and JOIN_INNER
		 *
		 * needs to clarify its condition!!
		 */

		/* set up initial pds_index */
		for (i=0; i < gjs->num_rels; i++)
		{
			int		nbatches_exec = list_length(gjs->inners[i].pds_list);

			Assert(nbatches_exec > 0);
			gjs->inners[i].pds_index = 1;
			/* also record actual nbatches */
			gjs->inners[i].nbatches_exec = nbatches_exec;
		}
	}
	else
	{
		for (i=gjs->num_rels - 1; i >= 0; i--)
		{
			int		n = list_length(gjs->inners[i].pds_list);

			if (gjs->inners[i].pds_index < n)
			{
				gjs->inners[i].pds_index++;
				for (j=i+1; j < gjs->num_rels; j++)
					gjs->inners[i].pds_index = 1;
				break;
			}
		}
		/* end of the inner scan */
		if (i < 0)
		{
			PERFMON_END(&gjs->gts.pfm_accum, time_inner_load, &tv1, &tv2);
			return NULL;
		}
	}

	/* make a first pmrels */
	pmrels = gpujoin_create_multirels(gjs);
	for (i=0; i < gjs->num_rels; i++)
	{
		innerState		   *istate = &gjs->inners[i];
		pgstrom_data_store *pds = list_nth(istate->pds_list,
										   gjs->inners[i].pds_index - 1);

		pmrels->inner_chunks[i] = pgstrom_acquire_data_store(pds);
		pmrels->kern.chunks[i].chunk_offset = pmrels->usage_length;
		pmrels->usage_length += STROMALIGN(pds->kds->length);

		if (istate->join_type == JOIN_RIGHT ||
			istate->join_type == JOIN_FULL)
		{
			pmrels->kern.chunks[i].right_outer = true;
			pmrels->kern.chunks[i].ojmap_offset = pmrels->ojmap_length;
			pmrels->ojmap_length += (STROMALIGN(sizeof(cl_bool) *
												pds->kds->nitems) *
									 pmrels->kern.ndevs);
			pmrels->needs_outer_join = true;
		}
		if (istate->join_type == JOIN_LEFT ||
			istate->join_type == JOIN_FULL)
			pmrels->kern.chunks[i].left_outer = true;
	}
	/* already attached on the caller's context */
	pmrels->n_attached = 1;
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
	int		i, num_rels = pmrels->kern.nrels;

	/* attach this pmrels */
	Assert(pmrels->n_attached > 0);
	pmrels->n_attached++;
	/* also, data store */
	for (i=0; i < num_rels; i++)
		pgstrom_acquire_data_store(pmrels->inner_chunks[i]);

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
		m_kmrels = gpuMemAlloc(gtask, pmrels->usage_length);
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
		GpuJoinState   *gjs = pmrels->gjs;
		CUdeviceptr		m_kmrels = pmrels->m_kmrels[cuda_index];
		Size			total_length = 0;
		Size			length;
		cl_int			i;

		rc = cuEventCreate(&ev_loaded, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

		/* DMA send to the kern_multirels buffer */
		length = offsetof(kern_multirels, chunks[pmrels->kern.nrels]);
		rc = cuMemcpyHtoDAsync(m_kmrels, &pmrels->kern, length, cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		total_length += length;

		for (i=0; i < pmrels->kern.nrels; i++)
		{
			pgstrom_data_store *pds = pmrels->inner_chunks[i];
			kern_data_store	   *kds = pds->kds;
			Size				offset = pmrels->kern.chunks[i].chunk_offset;

			rc = cuMemcpyHtoDAsync(m_kmrels + offset, kds, kds->length,
								   cuda_stream);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
			total_length += kds->length;
		}
		/* DMA Send synchronization */
		rc = cuEventRecord(ev_loaded, cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventRecord: %s", errorText(rc));
		/* save the event */
		pmrels->ev_loaded[cuda_index] = ev_loaded;

		/* update statistics */
		gjs->inner_dma_nums++;
		gjs->inner_dma_size += total_length;
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
multirels_detach_buffer(pgstrom_multirels *pmrels, bool may_kick_outer_join,
						const char *caller)
{
	int		i, num_rels = pmrels->kern.nrels;

	Assert(pmrels->n_attached > 0);

	/*
	 * NOTE: Invocation of multirels_detach_buffer with n_attached==1 means
	 * release of pgstrom_multirels buffer. If GpuJoin contains RIGHT or
	 * FULL OUTER JOIN, we need to kick OUTER JOIN task prior on the last.
	 * pgstrom_gpujoin task with pds_src==NULL means OUTER JOIN launch.
	 */
	if (may_kick_outer_join &&
		pmrels->n_attached == 1 &&
		pmrels->needs_outer_join)
	{
		GpuJoinState	   *gjs = pmrels->gjs;
		pgstrom_gpujoin	   *pgjoin_new = (pgstrom_gpujoin *)
			gpujoin_create_task(gjs, pmrels, NULL, NULL);

		/* Enqueue OUTER JOIN task here */
		SpinLockAcquire(&gjs->gts.lock);
		dlist_push_tail(&gjs->gts.pending_tasks, &pgjoin_new->task.chain);
		gjs->gts.num_pending_tasks++;
		SpinLockRelease(&gjs->gts.lock);

		/* no need to kick outer join task twice */
		pmrels->needs_outer_join = false;
	}

	/* release data store */
	for (i=0; i < num_rels; i++)
		pgstrom_release_data_store(pmrels->inner_chunks[i]);

	/* Also, this pmrels */
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
	gpujoin_exec_methods.c.CustomName			= "GpuJoin";
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
