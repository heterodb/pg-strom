/*
 * pg_strom.h
 *
 * Header file of pg_strom module
 * --
 * Copyright 2011-2023 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2023 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef PG_STROM_H
#define PG_STROM_H

#include "postgres.h"
#if PG_VERSION_NUM < 150000
#error Base PostgreSQL version must be v15 or later
#endif
#define PG_MAJOR_VERSION		(PG_VERSION_NUM / 100)
#define PG_MINOR_VERSION		(PG_VERSION_NUM % 100)

#include "access/brin.h"
#include "access/heapam.h"
#include "access/genam.h"
#include "access/reloptions.h"
#include "access/relscan.h"
#include "access/syncscan.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/visibilitymap.h"
#include "access/xact.h"
#include "catalog/binary_upgrade.h"
#include "catalog/dependency.h"
#include "catalog/heap.h"
#include "catalog/indexing.h"
#include "catalog/namespace.h"
#include "catalog/objectaccess.h"
#include "catalog/pg_aggregate.h"
#include "catalog/pg_am.h"
#include "catalog/pg_amop.h"
#include "catalog/pg_cast.h"
#include "catalog/pg_database.h"
#include "catalog/pg_depend.h"
#include "catalog/pg_foreign_table.h"
#include "catalog/pg_foreign_data_wrapper.h"
#include "catalog/pg_foreign_server.h"
#include "catalog/pg_user_mapping.h"
#include "catalog/pg_extension.h"
#include "catalog/pg_namespace.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_statistic.h"
#include "catalog/pg_tablespace_d.h"
#include "catalog/pg_trigger.h"
#include "catalog/pg_type.h"
#include "commands/dbcommands.h"
#include "commands/defrem.h"
#include "commands/event_trigger.h"
#include "commands/extension.h"
#include "commands/tablecmds.h"
#include "commands/tablespace.h"
#include "commands/trigger.h"
#include "commands/typecmds.h"
#include "common/hashfn.h"
#include "common/int.h"
#include "executor/nodeSubplan.h"
#include "foreign/fdwapi.h"
#include "foreign/foreign.h"
#include "funcapi.h"
#include "libpq/pqformat.h"
#include "libpq/pqsignal.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "nodes/extensible.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "nodes/pathnodes.h"
#include "optimizer/clauses.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/plancat.h"
#include "optimizer/planner.h"
#include "optimizer/planmain.h"
#include "optimizer/restrictinfo.h"
#include "optimizer/tlist.h"
#include "parser/parse_func.h"
#include "postmaster/bgworker.h"
#include "postmaster/postmaster.h"
#include "storage/bufmgr.h"
#include "storage/buf_internals.h"
#include "storage/ipc.h"
#include "storage/fd.h"
#include "storage/latch.h"
#include "storage/pmsignal.h"
#include "storage/procarray.h"
#include "storage/shmem.h"
#include "storage/smgr.h"
#include "utils/builtins.h"
#include "utils/cash.h"
#include "utils/catcache.h"
#include "utils/date.h"
#include "utils/datetime.h"
#include "utils/float.h"
#include "utils/fmgroids.h"
#include "utils/guc.h"
#include "utils/inet.h"
#include "utils/inval.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "utils/pg_locale.h"
#include "utils/rangetypes.h"
#include "utils/regproc.h"
#include "utils/rel.h"
#include "utils/resowner.h"
#include "utils/ruleutils.h"
#include "utils/selfuncs.h"
#include "utils/spccache.h"
#include "utils/syscache.h"
#include "utils/timestamp.h"
#include "utils/tuplestore.h"
#include "utils/typcache.h"
#include "utils/uuid.h"
#include "utils/wait_event.h"
#include <assert.h>
#define CUDA_API_PER_THREAD_DEFAULT_STREAM		1
#include <cuda.h>
#include <cufile.h>
#include <ctype.h>
#include <float.h>
#include <libgen.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>
#include "xpu_common.h"
#include "cuda_common.h"
#include "pg_utils.h"
#include "pg_compat.h"
#include "heterodb_extra.h"

/* ------------------------------------------------
 *
 * Global Type Definitions
 *
 * ------------------------------------------------
 */
typedef struct GpuDevAttributes
{
	int32		NUMA_NODE_ID;
	int32		DEV_ID;
	char		DEV_NAME[256];
	char		DEV_UUID[sizeof(CUuuid)];
	size_t		DEV_TOTAL_MEMSZ;
	size_t		DEV_BAR1_MEMSZ;
	bool		DEV_SUPPORT_GPUDIRECTSQL;
#define DEV_ATTR(LABEL,DESC)	\
	int32		LABEL;
#include "gpu_devattrs.h"
#undef DEV_ATTR
} GpuDevAttributes;

extern GpuDevAttributes *gpuDevAttrs;
extern int		numGpuDevAttrs;
#define GPUKERNEL_MAX_SM_MULTIPLICITY	4

/*
 * devtype/devfunc/devcast definitions
 */
struct devtype_info;
struct devfunc_info;
struct devcast_info;

typedef uint32_t (*devtype_hashfunc_f)(bool isnull, Datum value);

typedef struct devtype_info
{
	uint32_t	hash;
	TypeOpCode	type_code;
	Oid			type_oid;
	uint64_t	type_flags;
	int16		type_length;
	int16		type_align;
	bool		type_byval;
	bool		type_is_negative;
	const char *type_extension;
	const char *type_name;
	Oid			type_namespace;
	int			type_sizeof;
	int			type_alignof;
	devtype_hashfunc_f type_hashfunc;
	/* oid of type related functions */
	Oid			type_eqfunc;
	Oid			type_cmpfunc;
	/* alias type, if any */
	struct devtype_info *type_alias;
	/* element type of array, if type is array */
	struct devtype_info *type_element;
	/* attribute of sub-fields, if type is composite */
	int			comp_nfields;
	struct devtype_info *comp_subtypes[1];
} devtype_info;

typedef struct devfunc_info
{
	dlist_node	chain;
	uint32_t	hash;
	FuncOpCode	func_code;
	const char *func_extension;
	const char *func_name;
	Oid			func_oid;
	struct devtype_info *func_rettype;
	uint64_t	func_flags;
	int			func_cost;
	bool		func_is_negative;
	int			func_nargs;
	struct devtype_info *func_argtypes[1];
} devfunc_info;

typedef struct XpuConnection	XpuConnection;
typedef struct GpuCacheDesc		GpuCacheDesc;
typedef struct DpuStorageEntry	DpuStorageEntry;
typedef struct ArrowFdwState	ArrowFdwState;
typedef struct BrinIndexState	BrinIndexState;

/*
 * pgstromPlanInfo
 */
typedef struct
{
	JoinType		join_type;      /* one of JOIN_* */
	double			join_nrows;     /* estimated nrows in this depth */
	Cost			join_startup_cost; /* estimated startup cost */
	Cost			join_run_cost;	/* estimated run cost (incl final_cost) */
	List		   *hash_outer_keys;/* hash-keys for outer-side */
	List		   *hash_outer_keys_fallback;
	List		   *hash_inner_keys;/* hash-keys for inner-side */
	List		   *hash_inner_keys_fallback;
	List		   *join_quals;     /* join quals */
	List		   *join_quals_fallback;
	List		   *other_quals;    /* other quals */
	List		   *other_quals_fallback;
	/* gist index properties */
	Oid				gist_index_oid; /* GiST index oid */
	int				gist_index_col; /* GiST index column number */
	int				gist_ctid_resno;/* resno to reference ctid */
	Oid				gist_func_oid;	/* device function to evaluate GiST clause */
	int				gist_slot_id;	/* slot-id to store the index key */
	Expr		   *gist_clause;    /* GiST index clause */
	Selectivity		gist_selectivity; /* GiST selectivity */
	double			gist_npages;	/* number of disk pages */
	int				gist_height;	/* index tree height, or -1 if unknown */
} pgstromPlanInnerInfo;

typedef struct
{
	uint32_t	xpu_task_flags;		/* mask of device flags */
	int			gpu_cache_dindex;	/* device for GpuCache, if any */
	const Bitmapset *gpu_direct_devs;	/* device for GPU-Direct SQL, if any */
	const DpuStorageEntry *ds_entry;	/* target DPU if DpuJoin */
	/* Plan information */
	const Bitmapset *outer_refs;	/* referenced columns */
	List	   *used_params;		/* param list in use */
	List	   *host_quals;			/* host qualifiers to scan the outer */
	Index		scan_relid;			/* relid of the outer relation to scan */
	List	   *scan_quals;			/* device qualifiers to scan the outer */
	List	   *scan_quals_fallback;/* 'scan_quals' for CPU fallback */
	double		scan_tuples;		/* copy of baserel->tuples */
	double		scan_rows;			/* copy of baserel->rows */
	Cost		scan_startup_cost;	/* estimated startup cost to scan baserel */
	Cost		scan_run_cost;		/* estimated run cost to scan baserel */
	int			parallel_nworkers;	/* # of parallel workers */
	double		parallel_divisor;	/* parallel divisor */
	Cost		final_cost;			/* cost for sendback and host-side tasks */
	/* BRIN-index support */
	Oid			brin_index_oid;		/* OID of BRIN-index, if any */
	List	   *brin_index_conds;	/* BRIN-index key conditions */
	List	   *brin_index_quals;	/* Original BRIN-index qualifier */
	/* XPU code for JOIN */
	bytea	   *kexp_scan_kvars_load;	/* VarLoads at depth=0 */
	bytea	   *kexp_scan_quals;
	bytea	   *kexp_join_kvars_load_packed; /* VarLoads at depth>0 */
	bytea	   *kexp_join_quals_packed;
	bytea	   *kexp_hash_keys_packed;
	bytea	   *kexp_gist_evals_packed;
	bytea	   *kexp_projection;
	bytea	   *kexp_groupby_keyhash;
	bytea	   *kexp_groupby_keyload;
	bytea	   *kexp_groupby_keycomp;
	bytea	   *kexp_groupby_actions;
	List	   *kvars_depth;
	List	   *kvars_resno;
	List	   *kvars_types;	/* type-oid, if it needs extra buffer on kvars-slot */
	List	   *kvars_exprs;
	uint32_t	extra_flags;
	uint32_t	extra_bufsz;
	/* fallback projection */
	List	   *fallback_tlist;	/* fallback_slot -> custom_scan_tlist if JOIN/PREAGG */
	/* group-by parameters */
	List	   *groupby_actions;	/* list of KAGG_ACTION__* on the kds_final */
	/* inner relations */
	int			num_rels;
	pgstromPlanInnerInfo inners[FLEXIBLE_ARRAY_MEMBER];
} pgstromPlanInfo;

#define PP_INFO_NUM_ROWS(pp_info)								\
	((pp_info)->num_rels == 0									\
	 ? (pp_info)->scan_rows										\
	 : (pp_info)->inners[(pp_info)->num_rels - 1].join_nrows)
#define PP_INFO_STARTUP_COST(pp_info)								\
	((pp_info)->num_rels == 0										\
	 ? (pp_info)->scan_startup_cost									\
	 : (pp_info)->inners[(pp_info)->num_rels - 1].join_startup_cost)
#define PP_INFO_RUN_COST(pp_info)									\
	((pp_info)->num_rels == 0										\
	 ? (pp_info)->scan_run_cost										\
	 : (pp_info)->inners[(pp_info)->num_rels - 1].join_run_cost)

/*
 * pgstromSharedState
 */
typedef struct
{
	pg_atomic_uint64	inner_nitems;
	pg_atomic_uint64	inner_usage;
	pg_atomic_uint64	stats_gist;			/* only GiST-index */
	pg_atomic_uint64	stats_join;			/* # of tuples by this join */
} pgstromSharedInnerState;

typedef struct
{
	dsm_handle			ss_handle;			/* DSM handle of the SharedState */
	uint32_t			ss_length;			/* length of the SharedState */
	/* pg-strom's unique plan-id */
	uint64_t			query_plan_id;
	/* control variables to detect the last plan-node at parallel execution */
	//pg_atomic_uint32	scan_task_control;
	pg_atomic_uint32	parallel_task_control;
	pg_atomic_uint32	__rjoin_exit_count;
	/* statistics */
	pg_atomic_uint64	source_ntuples;		/* only KDS_FORMAT_BLOCK */
	pg_atomic_uint64	source_nvalids;		/* # of tuples scan'ed */
	pg_atomic_uint64	result_ntuples;		/* # of tuples generated */
	/* for arrow_fdw */
	pg_atomic_uint32	arrow_rbatch_index;
	pg_atomic_uint32	arrow_rbatch_nload;	/* # of loaded record-batches */
	pg_atomic_uint32	arrow_rbatch_nskip;	/* # of skipped record-batches */
	/* for gpu-cache */
	pg_atomic_uint32	__gcache_fetch_count_data;
	/* for gpu/dpu-direct */
	pg_atomic_uint32	heap_normal_nblocks;
	pg_atomic_uint32	heap_direct_nblocks;
	pg_atomic_uint32	heap_fallback_nblocks;
	/* for brin-index */
	pg_atomic_uint32	brin_index_fetched;
	pg_atomic_uint32	brin_index_skipped;
	/* for join-inner-preload */
	ConditionVariable	preload_cond;		/* sync object */
	slock_t				preload_mutex;		/* mutex for inner-preloading */
	int					preload_phase;		/* one of INNER_PHASE__* in gpu_join.c */
	int					preload_nr_scanning;/* # of scanning process */
	int					preload_nr_setup;	/* # of setup process */
	uint32_t			preload_shmem_handle; /* host buffer handle */
	uint64_t			preload_shmem_length; /* host buffer length */
	/* for join-inner relations */
	uint32_t			num_rels;			/* if xPU-JOIN involved */
	pgstromSharedInnerState inners[FLEXIBLE_ARRAY_MEMBER];
	/*
	 * MEMO: ...and ParallelBlockTableScanDescData should be allocated
	 *       next to the inners[nmum_rels] array
	 */
} pgstromSharedState;

typedef struct
{
	PlanState	   *ps;
	ExprContext	   *econtext;
	/*
	 * inner preload buffer
	 */
	List		   *preload_tuples;
	List		   *preload_hashes;     /* if hash-join or gist-join */
	size_t			preload_usage;
	/*
	 * join properties (common)
	 */
	int				depth;
	JoinType		join_type;
	ExprState	   *join_quals;
	ExprState	   *other_quals;
	/*
	 * join properties (hash-join)
	 */
	List		   *hash_outer_keys;    /* list of ExprState */
	List		   *hash_inner_keys;    /* list of ExprState */
	List		   *hash_outer_dtypes;  /* list of devtype_info */
	List		   *hash_inner_dtypes;  /* list of devtype_info */
	/*
	 * join properties (gist-join)
	 */
	Relation		gist_irel;
	ExprState	   *gist_clause;
	AttrNumber		gist_ctid_resno;
} pgstromTaskInnerState;

struct pgstromTaskState
{
	CustomScanState		css;
//	uint32_t			task_kind;		/* one of TASK_KIND__* */
	uint32_t			xpu_task_flags;	/* mask of device flags */
	const Bitmapset	   *optimal_gpus;	/* candidate GPUs to connect */
	const DpuStorageEntry *ds_entry;	/* candidate DPUs to connect */
	XpuConnection	   *conn;
	pgstromSharedState *ps_state;		/* on the shared-memory segment */
	pgstromPlanInfo	   *pp_info;
	ArrowFdwState	   *arrow_state;
	BrinIndexState	   *br_state;
	GpuCacheDesc	   *gcache_desc;
	pg_atomic_uint32   *gcache_fetch_count;
	kern_multirels	   *h_kmrels;		/* host inner buffer (if JOIN) */
	const char		   *kds_pathname;	/* pathname to be used for KDS setup */
	/* current chunk (already processed by the device) */
	XpuCommand		   *curr_resp;
	HeapTupleData		curr_htup;
	kern_data_store	   *curr_kds;
	int					curr_chunk;
	int64_t				curr_index;
	bool				scan_done;
	bool				final_done;
	/*
	 * control variables to fire the end-of-task event
	 * for RIGHT OUTER JOIN and PRE-AGG
	 */
	pg_atomic_uint32   *rjoin_devs_count;
	pg_atomic_uint32   *rjoin_exit_count;
	/* base relation scan, if any */
	TupleTableSlot	   *base_slot;
	ExprState		   *base_quals;	/* equivalent to device quals */
	/* CPU fallback support */
	off_t			   *fallback_tuples;
	size_t				fallback_index;
	size_t				fallback_nitems;
	size_t				fallback_nrooms;
	size_t				fallback_usage;
	size_t				fallback_bufsz;
	char			   *fallback_buffer;
	TupleTableSlot	   *fallback_slot;	/* host-side kvars-slot */
	ProjectionInfo	   *fallback_proj;	/* base or fallback slot -> custom_tlist */
	/* request command buffer (+ status for table scan) */
	TBMIterateResult   *curr_tbm;
	Buffer				curr_vm_buffer;		/* for visibility-map */
	BlockNumber			curr_block_num;		/* for KDS_FORMAT_BLOCK */
	BlockNumber			curr_block_tail;	/* for KDS_FORMAT_BLOCK */
	StringInfoData		xcmd_buf;
	/* callbacks */
	TupleTableSlot	 *(*cb_next_tuple)(struct pgstromTaskState *pts);
	XpuCommand		 *(*cb_next_chunk)(struct pgstromTaskState *pts,
									   struct iovec *xcmd_iov, int *xcmd_iovcnt);
	XpuCommand		 *(*cb_final_chunk)(struct pgstromTaskState *pts,
										kern_final_task *fin,
										struct iovec *xcmd_iov, int *xcmd_iovcnt);
	void			  (*cb_cpu_fallback)(struct pgstromTaskState *pts,
										 struct kern_data_store *kds,
										 HeapTuple htuple);
	/* inner relations state (if JOIN) */
	int					num_rels;
	pgstromTaskInnerState inners[FLEXIBLE_ARRAY_MEMBER];
};
typedef struct pgstromTaskState		pgstromTaskState;

/*
 * Global variables
 */
extern long		PAGE_SIZE;
extern long		PAGE_MASK;
extern int		PAGE_SHIFT;
extern long		PHYS_PAGES;
extern long		PAGES_PER_BLOCK;	/* (BLCKSZ / PAGE_SIZE) */
#define PAGE_ALIGN(x)			TYPEALIGN(PAGE_SIZE,(x))
#define PAGE_ALIGN_DOWN(x)		TYPEALIGN_DOWN(PAGE_SIZE,(x))
#define PGSTROM_CHUNK_SIZE		((size_t)(65534UL << 10))

/*
 * extra.c
 */
extern void		pgstrom_init_extra(void);
extern bool		heterodbValidateDevice(int gpu_device_id,
									   const char *gpu_device_name,
									   const char *gpu_device_uuid);
extern void		gpuDirectOpenDriver(void);
extern void		gpuDirectCloseDriver(void);
extern bool		gpuDirectMapGpuMemory(CUdeviceptr m_segment,
									  size_t segment_sz);
extern bool		gpuDirectUnmapGpuMemory(CUdeviceptr m_segment);
extern bool		gpuDirectFileReadIOV(const char *pathname,
									 CUdeviceptr m_segment,
									 off_t m_offset,
									 const strom_io_vector *iovec);
extern char	   *gpuDirectGetProperty(void);
extern void		gpuDirectSetProperty(const char *key, const char *value);
extern bool		gpuDirectIsAvailable(void);

/*
 * codegen.c
 */
typedef struct
{
	int			elevel;			/* ERROR or DEBUG2 */
	Expr	   *top_expr;
	List	   *used_params;
	uint32_t	required_flags;
	uint32_t	extra_flags;
	uint32_t	extra_bufsz;
	uint32_t	device_cost;
	uint32_t	kexp_flags;
	List	   *kvars_depth;
	List	   *kvars_resno;
	List	   *kvars_types;
	List	   *kvars_exprs;
	List	   *tlist_dev;
	uint32_t	kvars_nslots;
	List	   *input_rels_tlist;
} codegen_context;

extern devtype_info *pgstrom_devtype_lookup(Oid type_oid);
extern devfunc_info *pgstrom_devfunc_lookup(Oid func_oid,
											List *func_args,
											Oid func_collid);

extern devfunc_info *devtype_lookup_equal_func(devtype_info *dtype, Oid coll_id);
extern devfunc_info *devtype_lookup_compare_func(devtype_info *dtype, Oid coll_id);

extern void		codegen_context_init(codegen_context *context,
									 uint32_t xpu_task_flags);
extern bytea   *codegen_build_qualifiers(codegen_context *context,
										 List *dev_quals);
extern bytea   *codegen_build_scan_loadvars(codegen_context *context);
extern bytea   *codegen_build_scan_quals(codegen_context *context,
										 List *dev_quals);
extern bytea   *codegen_build_join_loadvars(codegen_context *context);
extern bytea   *codegen_build_packed_joinquals(codegen_context *context,
											   List *stacked_join_quals,
											   List *stacked_other_quals);
extern bytea   *codegen_build_packed_hashkeys(codegen_context *context,
											  List *stacked_hash_values);
extern void		codegen_build_packed_gistevals(codegen_context *context,
											   pgstromPlanInfo *pp_info);
extern bytea   *codegen_build_projection(codegen_context *context);
extern void		codegen_build_groupby_actions(codegen_context *context,
											  pgstromPlanInfo *pp_info);
extern void		codegen_build_packed_xpucode(bytea **p_xpucode,
											 List *exprs_list,
											 bool inject_hash_value,
											 List *input_rels_tlist,
											 uint32_t *p_extra_flags,
											 uint32_t *p_extra_bufsz,
											 uint32_t *p_kvars_nslots,
											 List **p_used_params);
extern bool		pgstrom_xpu_expression(Expr *expr,
									   uint32_t required_xpu_flags,
									   List *input_rels_tlist,
									   int *p_devcost);
extern bool		pgstrom_gpu_expression(Expr *expr,
									   List *input_rels_tlist,
									   int *p_devcost);
extern bool		pgstrom_dpu_expression(Expr *expr,
									   List *input_rels_tlist,
									   int *p_devcost);
extern void		pgstrom_explain_xpucode(const CustomScanState *css,
										ExplainState *es,
										List *dcontext,
										const char *label,
										bytea *xpucode);
extern char	   *pgstrom_xpucode_to_string(bytea *xpu_code);
extern void		pgstrom_init_codegen(void);

/*
 * brin.c
 */
extern IndexOptInfo *pgstromTryFindBrinIndex(PlannerInfo *root,
											 RelOptInfo *baserel,
											 List **p_indexConds,
											 List **p_indexQuals,
											 int64_t *p_indexNBlocks);
extern Cost		cost_brin_bitmap_build(PlannerInfo *root,
									   RelOptInfo *baserel,
									   IndexOptInfo *indexOpt,
									   List *indexQuals);

extern void		pgstromBrinIndexExecBegin(pgstromTaskState *pts,
										  Oid index_oid,
										  List *index_conds,
										  List *index_quals);
extern bool		pgstromBrinIndexNextChunk(pgstromTaskState *pts);
extern TBMIterateResult *pgstromBrinIndexNextBlock(pgstromTaskState *pts);
extern void		pgstromBrinIndexExecEnd(pgstromTaskState *pts);
extern void		pgstromBrinIndexExecReset(pgstromTaskState *pts);
extern Size		pgstromBrinIndexEstimateDSM(pgstromTaskState *pts);
extern Size		pgstromBrinIndexInitDSM(pgstromTaskState *pts, char *dsm_addr);
extern Size		pgstromBrinIndexAttachDSM(pgstromTaskState *pts, char *dsm_addr);
extern void		pgstromBrinIndexShutdownDSM(pgstromTaskState *pts);
extern void		pgstromBrinIndexExplain(pgstromTaskState *pts,
										List *dcontext,
										ExplainState *es);
extern void		pgstrom_init_brin(void);

/*
 * gist.c
 */
extern Path	   *pgstromTryFindGistIndex(PlannerInfo *root,
										Path *inner_path,
										List *restrict_clauses,
										uint32_t xpu_task_flags,
										List *input_rels_tlist,
										pgstromPlanInnerInfo *pp_inner);

/*
 * relscan.c
 */
extern Bitmapset *pickup_outer_referenced(PlannerInfo *root,
										  RelOptInfo *base_rel,
										  Bitmapset *referenced);
extern size_t	estimate_kern_data_store(TupleDesc tupdesc);
extern size_t	setup_kern_data_store(kern_data_store *kds,
									  TupleDesc tupdesc,
									  size_t length,
									  char format);
extern XpuCommand *pgstromRelScanChunkDirect(pgstromTaskState *pts,
											 struct iovec *xcmd_iov,
											 int *xcmd_iovcnt);
extern XpuCommand *pgstromRelScanChunkNormal(pgstromTaskState *pts,
											 struct iovec *xcmd_iov,
											 int *xcmd_iovcnt);
extern void		pgstromStoreFallbackTuple(pgstromTaskState *pts, HeapTuple tuple);
extern TupleTableSlot *pgstromFetchFallbackTuple(pgstromTaskState *pts);
extern void		pgstrom_init_relscan(void);

/*
 * optimizer.c
 */



/*
 * executor.c
 */
extern void		__xpuClientOpenSession(pgstromTaskState *pts,
									   const XpuCommand *session,
									   pgsocket sockfd,
									   const char *devname,
									   int dev_index);
extern int
xpuConnectReceiveCommands(pgsocket sockfd,
						  void *(*alloc_f)(void *priv, size_t sz),
						  void  (*attach_f)(void *priv, XpuCommand *xcmd),
						  void *priv,
						  const char *error_label);
extern void		xpuClientCloseSession(XpuConnection *conn);
extern void		xpuClientSendCommand(XpuConnection *conn, const XpuCommand *xcmd);
extern void		xpuClientPutResponse(XpuCommand *xcmd);
extern const XpuCommand *pgstromBuildSessionInfo(pgstromTaskState *pts,
												 uint32_t join_inner_handle,
												 TupleDesc tdesc_final);
extern void		pgstromExecInitTaskState(CustomScanState *node,
										  EState *estate,
										 int eflags);
extern TupleTableSlot *pgstromExecTaskState(CustomScanState *node);
extern void		pgstromExecEndTaskState(CustomScanState *node);
extern void		pgstromExecResetTaskState(CustomScanState *node);
extern Size		pgstromSharedStateEstimateDSM(CustomScanState *node,
											  ParallelContext *pcxt);
extern void		pgstromSharedStateInitDSM(CustomScanState *node,
										  ParallelContext *pcxt,
										  void *coordinate);
extern void		pgstromSharedStateAttachDSM(CustomScanState *node,
											shm_toc *toc,
											void *coordinate);
extern void		pgstromSharedStateShutdownDSM(CustomScanState *node);
extern void		pgstromExplainTaskState(CustomScanState *node,
										List *ancestors,
										ExplainState *es);
extern void		pgstrom_init_executor(void);

/*
 * pcie.c
 */
extern const Bitmapset *GetOptimalGpuForFile(const char *pathname);
extern const Bitmapset *GetOptimalGpuForRelation(Relation relation);
extern const Bitmapset *GetOptimalGpuForBaseRel(PlannerInfo *root,
												RelOptInfo *baserel);
extern void		pgstrom_init_pcie(void);

/*
 * gpu_device.c
 */
extern double	pgstrom_gpu_setup_cost;		/* GUC */
extern double	pgstrom_gpu_tuple_cost;		/* GUC */
extern double	pgstrom_gpu_operator_cost;	/* GUC */
extern double	pgstrom_gpu_direct_seq_page_cost; /* GUC */
extern double	pgstrom_gpu_operator_ratio(void);
extern void		gpuClientOpenSession(pgstromTaskState *pts,
									 const XpuCommand *session);
extern CUresult	gpuOptimalBlockSize(int *p_grid_sz,
									int *p_block_sz,
									unsigned int *p_shmem_sz,
									CUfunction kern_function,
									size_t dynamic_shmem_per_block,
									size_t dynamic_shmem_per_warp);
extern bool		pgstrom_init_gpu_device(void);

/*
 * gpu_service.c
 */
typedef struct gpuContext	gpuContext;
typedef struct gpuClient	gpuClient;

extern int		pgstrom_max_async_tasks(void);
extern const char *cuStrError(CUresult rc);
extern bool		gpuServiceGoingTerminate(void);
extern void		gpuservBgWorkerMain(Datum arg);
extern void		pgstrom_init_gpu_service(void);

/*
 * gpu_cache.c
 */
extern void		pgstrom_init_gpu_cache(void);
extern int		baseRelHasGpuCache(PlannerInfo *root,
								   RelOptInfo *baserel);
extern bool		RelationHasGpuCache(Relation rel);
extern const GpuCacheIdent *getGpuCacheDescIdent(const GpuCacheDesc *gc_desc);
extern GpuCacheDesc *pgstromGpuCacheExecInit(pgstromTaskState *pts);
extern XpuCommand *pgstromScanChunkGpuCache(pgstromTaskState *pts,
											struct iovec *xcmd_iov,
											int *xcmd_iovcnt);
extern void		pgstromGpuCacheExecEnd(pgstromTaskState *pts);
extern void		pgstromGpuCacheExecReset(pgstromTaskState *pts);
extern void		pgstromGpuCacheInitDSM(pgstromTaskState *pts,
									   pgstromSharedState *ps_state);
extern void		pgstromGpuCacheAttachDSM(pgstromTaskState *pts,
										 pgstromSharedState *ps_state);
extern void		pgstromGpuCacheShutdown(pgstromTaskState *pts);
extern void		pgstromGpuCacheExplain(pgstromTaskState *pts,
									   ExplainState *es,
									   List *dcontext);
extern void		gpucacheManagerEventLoop(int cuda_dindex,
										 CUcontext cuda_context,
										 CUmodule cuda_module);
extern void		gpucacheManagerWakeUp(int cuda_dindex);

extern void	   *gpuCacheGetDeviceBuffer(const GpuCacheIdent *ident,
										CUdeviceptr *p_gcache_main_devptr,
										CUdeviceptr *p_gcache_extra_devptr,
										char *errbuf, size_t errbuf_sz);
extern void		gpuCachePutDeviceBuffer(void *gc_lmap);

/*
 * gpu_scan.c
 */
extern void		sort_device_qualifiers(List *dev_quals_list,
									   List *dev_costs_list);
extern pgstromPlanInfo *try_fetch_xpuscan_planinfo(const Path *path);
extern pgstromPlanInfo *buildOuterScanPlanInfo(PlannerInfo *root,
											   RelOptInfo *baserel,
											   uint32_t xpu_task_flags,
											   bool parallel_path,
											   bool allow_host_quals,
											   bool allow_no_device_quals,
											   ParamPathInfo **p_param_info);
extern void		ExecFallbackCpuScan(pgstromTaskState *pts,
									kern_data_store *kds,
									HeapTuple tuple);
extern void		gpuservHandleGpuScanExec(gpuClient *gclient, XpuCommand *xcmd);
extern void		pgstrom_init_gpu_scan(void);
extern void		pgstrom_init_dpu_scan(void);

/*
 * gpu_join.c
 */
extern void		form_pgstrom_plan_info(CustomScan *cscan,
									   pgstromPlanInfo *pp_info);
extern pgstromPlanInfo *deform_pgstrom_plan_info(CustomScan *cscan);
extern pgstromPlanInfo *try_fetch_xpujoin_planinfo(const Path *path);
extern pgstromPlanInfo *buildOuterJoinPlanInfo(PlannerInfo *root,
											   RelOptInfo *outer_rel,
											   uint32_t xpu_task_flags,
											   bool try_parallel_path,
											   ParamPathInfo **p_param_info,
											   List **p_inner_paths_list);
extern List *build_fallback_exprs_scan(Index scan_relid, List *scan_exprs);
extern List *build_fallback_exprs_join(codegen_context *context,
									   List *join_exprs);
extern CustomScan *PlanXpuJoinPathCommon(PlannerInfo *root,
										 RelOptInfo *joinrel,
										 CustomPath *cpath,
										 List *tlist,
										 List *custom_plans,
										 pgstromPlanInfo *pp_info,
										 const CustomScanMethods *methods);
extern uint32_t	GpuJoinInnerPreload(pgstromTaskState *pts);
extern void		ExecFallbackCpuJoin(pgstromTaskState *pts,
									kern_data_store *kds,
									HeapTuple tuple);
extern void		ExecFallbackCpuJoinRightOuter(pgstromTaskState *pts);
extern void		ExecFallbackCpuJoinOuterJoinMap(pgstromTaskState *pts,
												XpuCommand *resp);
extern void		pgstrom_init_gpu_join(void);
extern void		pgstrom_init_dpu_join(void);

/*
 * gpu_preagg.c
 */
extern int		pgstrom_hll_register_bits;
extern void		xpupreagg_add_custompath(PlannerInfo *root,
										 RelOptInfo *input_rel,
										 RelOptInfo *group_rel,
										 void *extra,
										 uint32_t task_kind,
										 const CustomPathMethods *methods);
extern void		ExecFallbackCpuPreAgg(pgstromTaskState *pts,
									  kern_data_store *kds,
									  HeapTuple tuple);
extern void		pgstrom_init_gpu_preagg(void);
extern void		pgstrom_init_dpu_preagg(void);

/*
 * arrow_fdw.c and arrow_read.c
 */
extern bool		baseRelIsArrowFdw(RelOptInfo *baserel);
extern bool 	RelationIsArrowFdw(Relation frel);
extern const Bitmapset *GetOptimalGpusForArrowFdw(PlannerInfo *root,
												  RelOptInfo *baserel);
extern const DpuStorageEntry *GetOptimalDpuForArrowFdw(PlannerInfo *root,
													   RelOptInfo *baserel);
extern bool		pgstromArrowFdwExecInit(pgstromTaskState *pts,
										List *outer_quals,
										const Bitmapset *outer_refs);
extern XpuCommand *pgstromScanChunkArrowFdw(pgstromTaskState *pts,
											struct iovec *xcmd_iov,
											int *xcmd_iovcnt);
extern void		pgstromArrowFdwExecEnd(ArrowFdwState *arrow_state);
extern void		pgstromArrowFdwExecReset(ArrowFdwState *arrow_state);
extern void		pgstromArrowFdwInitDSM(ArrowFdwState *arrow_state,
									   pgstromSharedState *ps_state);
extern void		pgstromArrowFdwAttachDSM(ArrowFdwState *arrow_state,
										 pgstromSharedState *ps_state);
extern void		pgstromArrowFdwShutdown(ArrowFdwState *arrow_state);
extern void		pgstromArrowFdwExplain(ArrowFdwState *arrow_state,
									   Relation frel,
									   ExplainState *es,
									   List *dcontext);
extern bool		kds_arrow_fetch_tuple(TupleTableSlot *slot,
									  kern_data_store *kds,
									  size_t index,
									  const Bitmapset *referenced);
extern void pgstrom_init_arrow_fdw(void);

/*
 * dpu_device.c
 */
extern double	pgstrom_dpu_setup_cost;
extern double	pgstrom_dpu_operator_cost;
extern double	pgstrom_dpu_seq_page_cost;
extern double	pgstrom_dpu_tuple_cost;
extern bool		pgstrom_dpu_handle_cached_pages;
extern double	pgstrom_dpu_operator_ratio(void);

extern const DpuStorageEntry *GetOptimalDpuForFile(const char *filename,
												   const char **p_dpu_pathname);
extern const DpuStorageEntry *GetOptimalDpuForBaseRel(PlannerInfo *root,
													  RelOptInfo *baserel);
extern const DpuStorageEntry *GetOptimalDpuForRelation(Relation relation,
													   const char **p_dpu_pathname);
extern const char *DpuStorageEntryBaseDir(const DpuStorageEntry *ds_entry);
extern bool		DpuStorageEntryIsEqual(const DpuStorageEntry *ds_entry1,
									   const DpuStorageEntry *ds_entry2);
extern int		DpuStorageEntryGetEndpointId(const DpuStorageEntry *ds_entry);
extern const DpuStorageEntry *DpuStorageEntryByEndpointId(int endpoint_id);
extern int		DpuStorageEntryCount(void);
extern void		DpuClientOpenSession(pgstromTaskState *pts,
									 const XpuCommand *session);
extern void		explainDpuStorageEntry(const DpuStorageEntry *ds_entry,
									   ExplainState *es);
extern bool		pgstrom_init_dpu_device(void);

/*
 * misc.c
 */
extern Node	   *fixup_varnode_to_origin(Node *node, List *cscan_tlist);
extern int		__appendBinaryStringInfo(StringInfo buf,
										 const void *data, int datalen);
extern int		__appendZeroStringInfo(StringInfo buf, int nbytes);
extern char	   *get_type_name(Oid type_oid, bool missing_ok);
extern Oid		get_type_namespace(Oid type_oid);
extern char	   *get_type_extension_name(Oid type_oid);
extern char	   *get_func_extension_name(Oid func_oid);
extern Oid		get_relation_am(Oid rel_oid, bool missing_ok);
extern List	   *bms_to_pglist(const Bitmapset *bms);
extern Bitmapset *bms_from_pglist(List *pglist);
extern Float   *__makeFloat(double fval);
extern Const   *__makeByteaConst(bytea *data);
extern bytea   *__getByteaConst(Const *con);
extern ssize_t	__readFile(int fdesc, void *buffer, size_t nbytes);
extern ssize_t	__preadFile(int fdesc, void *buffer, size_t nbytes, off_t f_pos);
extern ssize_t	__writeFile(int fdesc, const void *buffer, size_t nbytes);
extern ssize_t	__pwriteFile(int fdesc, const void *buffer, size_t nbytes, off_t f_pos);

extern uint32_t	__shmemCreate(const DpuStorageEntry *ds_entry);
extern void		__shmemDrop(uint32_t shmem_handle);
extern void	   *__mmapShmem(uint32_t shmem_handle,
							size_t shmem_length,
							const DpuStorageEntry *ds_entry);
extern bool		__munmapShmem(void *mmap_addr);

extern Path	   *pgstrom_copy_pathnode(const Path *pathnode);

/*
 * githash.c (auto-generated)
 */
extern const char *pgstrom_githash_cstring;

/*
 * main.c
 */
extern bool		pgstrom_enabled;
extern bool		pgstrom_cpu_fallback_enabled;
extern bool		pgstrom_regression_test_mode;
extern Path	   *pgstrom_create_dummy_path(PlannerInfo *root, Path *subpath);
extern void		_PG_init(void);

#endif	/* PG_STROM_H */
