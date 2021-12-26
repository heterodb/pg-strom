/*
 * gpupreagg.c
 *
 * Aggregate Pre-processing with GPU acceleration
 * ----
 * Copyright 2011-2021 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2021 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include "pg_strom.h"
#include "cuda_gpuscan.h"
#include "cuda_gpujoin.h"
#include "cuda_gpupreagg.h"

static create_upper_paths_hook_type create_upper_paths_next;
static CustomPathMethods	gpupreagg_path_methods;
static CustomScanMethods	gpupreagg_scan_methods;
static CustomExecMethods	gpupreagg_exec_methods;
static bool					enable_gpupreagg;				/* GUC */
static bool					enable_pullup_outer_join;		/* GUC */
static bool					enable_partitionwise_gpupreagg;	/* GUC */
static bool					enable_numeric_aggfuncs; 		/* GUC */
static double				gpupreagg_reduction_threshold;	/* GUC */
int							pgstrom_hll_register_bits;		/* GUC */


typedef struct
{
	cl_int			num_group_keys;	/* number of grouping keys */
	cl_int			num_accum_values; /* number of accumulation */
	cl_int			accum_extra_bufsz;/* size of accumulation extra buffer */
	double			plan_ngroups;	/* planned number of groups */
	cl_int			plan_nchunks;	/* planned number of chunks */
	Cost			outer_startup_cost; /* copy of @startup_cost in outer */
	Cost			outer_total_cost; /* copy of @total_cost in outer path */
	double			outer_nrows;	/* number of estimated outer nrows */
	int				outer_width;	/* copy of @plan_width in outer path */
	cl_uint			outer_nrows_per_block;
	Index			outer_scanrelid;/* RTI, if outer path pulled up */
	List		   *outer_quals;	/* device executable quals of outer-scan */
	List		   *outer_refs;		/* referenced columns */
	Oid				index_oid;		/* OID of BRIN-index, if any */
	List		   *index_conds;	/* BRIN-index key conditions */
	List		   *index_quals;	/* Original BRIN-index qualifiers */
	List		   *tlist_part;		/* template of kds_final */
	List		   *tlist_prep;		/* template of kds_slot */
	const Bitmapset *optimal_gpus;
	char		   *kern_source;
	cl_uint			extra_flags;
	cl_uint			extra_bufsz;
	List		   *used_params;	/* referenced Const/Param */
} GpuPreAggInfo;

static inline void
form_gpupreagg_info(CustomScan *cscan, GpuPreAggInfo *gpa_info)
{
	List	   *privs = NIL;
	List	   *exprs = NIL;

	privs = lappend(privs, makeInteger(gpa_info->num_group_keys));
	privs = lappend(privs, makeInteger(gpa_info->num_accum_values));
	privs = lappend(privs, makeInteger(gpa_info->accum_extra_bufsz));
	privs = lappend(privs, pmakeFloat(gpa_info->plan_ngroups));
	privs = lappend(privs, makeInteger(gpa_info->plan_nchunks));
	privs = lappend(privs, pmakeFloat(gpa_info->outer_startup_cost));
	privs = lappend(privs, pmakeFloat(gpa_info->outer_total_cost));
	privs = lappend(privs, pmakeFloat(gpa_info->outer_nrows));
	privs = lappend(privs, makeInteger(gpa_info->outer_width));
	privs = lappend(privs, makeInteger(gpa_info->outer_nrows_per_block));
	privs = lappend(privs, makeInteger(gpa_info->outer_scanrelid));
	exprs = lappend(exprs, gpa_info->outer_quals);
	privs = lappend(privs, gpa_info->outer_refs);
	privs = lappend(privs, makeInteger(gpa_info->index_oid));
	privs = lappend(privs, gpa_info->index_conds);
	exprs = lappend(exprs, gpa_info->index_quals);
	exprs = lappend(exprs, gpa_info->tlist_part);
	exprs = lappend(exprs, gpa_info->tlist_prep);
	privs = lappend(privs, bms_to_pglist(gpa_info->optimal_gpus));
	privs = lappend(privs, makeString(gpa_info->kern_source));
	privs = lappend(privs, makeInteger(gpa_info->extra_flags));
	privs = lappend(privs, makeInteger(gpa_info->extra_bufsz));
	exprs = lappend(exprs, gpa_info->used_params);

	cscan->custom_private = privs;
	cscan->custom_exprs = exprs;
}

static inline GpuPreAggInfo *
deform_gpupreagg_info(CustomScan *cscan)
{
	GpuPreAggInfo *gpa_info = palloc0(sizeof(GpuPreAggInfo));
	List	   *privs = cscan->custom_private;
	List	   *exprs = cscan->custom_exprs;
	int			pindex = 0;
	int			eindex = 0;

	gpa_info->num_group_keys = intVal(list_nth(privs, pindex++));
	gpa_info->num_accum_values = intVal(list_nth(privs, pindex++));
	gpa_info->accum_extra_bufsz = intVal(list_nth(privs, pindex++));
	gpa_info->plan_ngroups = floatVal(list_nth(privs, pindex++));
	gpa_info->plan_nchunks = intVal(list_nth(privs, pindex++));
	gpa_info->outer_startup_cost = floatVal(list_nth(privs, pindex++));
	gpa_info->outer_total_cost = floatVal(list_nth(privs, pindex++));
	gpa_info->outer_nrows = floatVal(list_nth(privs, pindex++));
	gpa_info->outer_width = intVal(list_nth(privs, pindex++));
	gpa_info->outer_nrows_per_block = intVal(list_nth(privs, pindex++));
	gpa_info->outer_scanrelid = intVal(list_nth(privs, pindex++));
	gpa_info->outer_quals = list_nth(exprs, eindex++);
	gpa_info->outer_refs = list_nth(privs, pindex++);
	gpa_info->index_oid = intVal(list_nth(privs, pindex++));
	gpa_info->index_conds = list_nth(privs, pindex++);
	gpa_info->index_quals = list_nth(exprs, eindex++);
	gpa_info->tlist_part = list_nth(exprs, eindex++);
	gpa_info->tlist_prep = list_nth(exprs, eindex++);
	gpa_info->optimal_gpus = bms_from_pglist(list_nth(privs, pindex++));
	gpa_info->kern_source = strVal(list_nth(privs, pindex++));
	gpa_info->extra_flags = intVal(list_nth(privs, pindex++));
	gpa_info->extra_bufsz = intVal(list_nth(privs, pindex++));
	gpa_info->used_params = list_nth(exprs, eindex++);
	Assert(pindex == list_length(privs));
	Assert(eindex == list_length(exprs));

	return gpa_info;
}

/*
 * GpuPreAggSharedState - to be allocated on DSM
 */
typedef struct
{
	GpuTaskState	gts;
	struct GpuPreAggSharedState *gpa_sstate;
	struct GpuPreAggRuntimeStat *gpa_rtstat;
	cl_bool			combined_gpujoin;
	cl_bool			terminator_done;
	cl_int			num_group_keys;
	cl_int			num_accum_values;	/* __GPUPREAGG_NUM_ACCUM_VALUES */
	cl_int			accum_extra_bufsz;	/* __GPUPREAGG_ACCUM_EXTRA_BUFSZ */
	cl_int			local_hash_nrooms;	/* __GPUPREAGG_LOCAL_HASH_NROOMS */
	TupleTableSlot *part_slot;	/* slot reflects tlist_part (kds_final) */
	TupleTableSlot *prep_slot;	/* slot reflects tlist_prep (kds_slot) */
	//TupleTableSlot *gpreagg_slot;	/* Slot reflects tlist_dev (w/o junks) */
	ExprState	   *outer_quals;
	TupleTableSlot *outer_slot;
	ProjectionInfo *fallback_proj;

	kern_data_store *kds_slot_head;
	pgstrom_data_store *pds_final;
	CUdeviceptr		m_fhash;
	CUevent			ev_init_fhash;
	size_t			f_hash_nslots;
	size_t			f_hash_length;
	pthread_mutex_t	f_mutex;

	size_t			plan_nrows_per_chunk;	/* planned nrows/chunk */
	size_t			plan_nrows_in;	/* num of outer rows planned */
	size_t			plan_ngroups;	/* num of groups planned */
} GpuPreAggState;

struct GpuPreAggRuntimeStat
{
	GpuTaskRuntimeStat	c;		/* common statistics */
};
typedef struct GpuPreAggRuntimeStat	GpuPreAggRuntimeStat;

struct GpuPreAggSharedState
{
	dsm_handle		ss_handle;	/* DSM handle of the SharedState */
	cl_uint			ss_length;	/* Length of the SharedState */
	GpuPreAggRuntimeStat gpa_rtstat;	/* Run-time statistics */
};
typedef struct GpuPreAggSharedState	GpuPreAggSharedState;

/*
 * GpuPreAggTask
 *
 * Host side representation of kern_gpupreagg. It can perform as a message
 * object of PG-Strom, has key of OpenCL device program, a source row/column
 * store and a destination kern_data_store.
 */
typedef struct GpuJoinSharedState	GpuJoinSharedState;

typedef enum {
	GpuPreAggTaskRole__Normal,
	GpuPreAggTaskRole__Dummy,
	GpuPreAggTaskRole__LocalTerminator,
	GpuPreAggTaskRole__GlobalTerminator,
} GpuPreAggTaskRole;

typedef struct
{
	GpuTask				task;
	bool				with_nvme_strom;/* true, if NVMe-Strom */
	pgstrom_data_store *pds_src;	/* source row/block buffer */
	kern_data_store	   *kds_slot;	/* partial results generated by combined
									 * GpuJoin, and should be kept for CPU
									 * fallback jobs */
	size_t				kds_slot_nrooms; /* for kds_slot */
	size_t				kds_slot_length; /* for kds_slot */
	/* <-- properties of combined mode --> */
	kern_gpujoin	   *kgjoin;		/* kern_gpujoin, if combined mode */
	CUdeviceptr			m_kmrels;	/* kern_multirels, if combined mode */
	CUdeviceptr			m_kparams;	/* kern_params, if combined mode */
	cl_int				outer_depth;/* RIGHT OUTER depth, if combined mode */
	kern_gpupreagg		kern;
} GpuPreAggTask;

/* declaration of static functions */
static bool		gpupreagg_build_path_target(PlannerInfo *root,
											PathTarget *target_upper,
											PathTarget *target_final,
											PathTarget *target_partial,
											Path       *input_path,
											Node      **p_havingQual,
											bool       *p_can_pullup_outerscan,
											AggClauseCosts *p_final_clause_costs);
static void		gpupreagg_codegen(PlannerInfo *root,
								  RelOptInfo *baserel,
								  CustomScan *cscan,
								  List *outer_tlist,
								  GpuPreAggInfo *gpa_info);
static size_t createGpuPreAggSharedState(GpuPreAggState *gpas,
										 ParallelContext *pcxt,
										 void *dsm_addr);
static void releaseGpuPreAggSharedState(GpuPreAggState *gpas);
static void resetGpuPreAggSharedState(GpuPreAggState *gpas);

static GpuTask *gpupreagg_next_task(GpuTaskState *gts);
static GpuTask *gpupreagg_terminator_task(GpuTaskState *gts,
										  cl_bool *task_is_ready);
static int  gpupreagg_process_task(GpuTask *gtask, CUmodule cuda_module);
static void gpupreagg_release_task(GpuTask *gtask);
static TupleTableSlot *gpupreagg_next_tuple(GpuTaskState *gts);

/*
 * Arguments of alternative functions.
 */
#define ALTFUNC_EXPR_NROWS			101	/* NROWS(X) */
#define ALTFUNC_EXPR_PMIN			102	/* PMIN(X) */
#define ALTFUNC_EXPR_PMAX			103	/* PMAX(X) */
#define ALTFUNC_EXPR_PSUM			104	/* PSUM(X) */
#define ALTFUNC_EXPR_PSUM_X2		105	/* PSUM_X2(X) = PSUM(X^2) */
#define ALTFUNC_EXPR_PCOV_X			106	/* PCOV_X(X,Y) */
#define ALTFUNC_EXPR_PCOV_Y			107	/* PCOV_Y(X,Y) */
#define ALTFUNC_EXPR_PCOV_X2		108	/* PCOV_X2(X,Y) */
#define ALTFUNC_EXPR_PCOV_Y2		109	/* PCOV_Y2(X,Y) */
#define ALTFUNC_EXPR_PCOV_XY		110	/* PCOV_XY(X,Y) */
#define ALTFUNC_EXPR_HLL_HASH		111	/* HLL_HASH(X) */

/*
 * XXX - GpuPreAgg with Numeric arguments are problematic because
 * it is implemented with normal function call and iteration of
 * cmpxchg. Thus, larger reduction ratio (usually works better)
 * will increase atomic contension. So, at this moment we turned
 * off GpuPreAgg + Numeric
 */
#define GPUPREAGG_SUPPORT_NUMERIC			1

#ifndef INT8ARRAYOID
#define INT8ARRAYOID		1016	/* see pg_type.h */
#endif
#ifndef FLOAT8ARRAYOID
#define FLOAT8ARRAYOID		1022	/* see pg_type.h */
#endif
#ifndef NUMERICARRAYOID
#define NUMERICARRAYOID		1231	/* see pg_type.h */
#endif

static size_t
__aggfunc_property_extra_sz__hll_count(void)
{
	return MAXALIGN(VARHDRSZ + (1U << pgstrom_hll_register_bits));
}

/*
 * List of supported aggregate functions
 */
typedef struct {
	/* aggregate function can be preprocessed */
	const char *aggfn_name;
	int			aggfn_nargs;
	Oid			aggfn_argtypes[4];
	/*
	 * A pair of final/partial function will generate same result.
	 * Its prefix indicates the schema that stores these functions.
	 * c: pg_catalog ... the system default
	 * s: pgstrom    ... PG-Strom's special ones
	 */
	const char *finalfn_name;
	Oid			finalfn_argtype;
	const char *partfn_name;
	int			partfn_nargs;
	Oid			partfn_argtypes[8];
	int			partfn_argexprs[8];
	size_t	  (*partfn_extra_sz)(void);
	int			extra_flags;
	bool		numeric_aware;	/* ignored, if !enable_numeric_aggfuncs */
} aggfunc_catalog_t;

static aggfunc_catalog_t  aggfunc_catalog[] = {
	/* AVG(X) = EX_AVG(NROWS(), PSUM(X)) */
	{ "avg",    1, {INT2OID},
	  "s:favg",     INT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  NULL, 0, false
	},
	{ "avg",    1, {INT4OID},
	  "s:favg",     INT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  NULL, 0, false
	},
	{ "avg",    1, {INT8OID},
	  "s:favg",     INT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  NULL, 0, false
	},
	{ "avg",    1, {FLOAT4OID},
	  "s:favg",     FLOAT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  NULL, 0, false
	},
	{ "avg",    1, {FLOAT8OID},
	  "s:favg",     FLOAT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  NULL, 0, false
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "avg",	1, {NUMERICOID},
	  "s:favg_numeric", FLOAT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  NULL, 0, true
	},
#endif
	/* COUNT(*) = SUM(NROWS(*|X)) */
	{ "count",  0, {},
	  "s:sum",      INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_NROWS},
	  NULL, 0, false
	},
	{ "count",  1, {ANYOID},
	  "s:sum",      INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_NROWS},
	  NULL, 0, false
	},
	/* HLL_COUNT(KEY) */
	{ "hll_count", 1, {INT1OID},
	  "c:hll_merge", BYTEAOID,
	  "s:hll_sketch_new", 1, {INT8OID},
	  {ALTFUNC_EXPR_HLL_HASH},
	  __aggfunc_property_extra_sz__hll_count,
	  0, false
	},
	{ "hll_count", 1, {INT2OID},
	  "c:hll_merge", BYTEAOID,
	  "s:hll_sketch_new", 1, {INT8OID},
	  {ALTFUNC_EXPR_HLL_HASH},
	  __aggfunc_property_extra_sz__hll_count,
	  0, false
	},
	{ "hll_count", 1, {INT4OID},
	  "c:hll_merge", BYTEAOID,
	  "s:hll_sketch_new", 1, {INT8OID},
	  {ALTFUNC_EXPR_HLL_HASH},
	  __aggfunc_property_extra_sz__hll_count,
	  0, false
	},
	{ "hll_count", 1, {INT8OID},
	  "c:hll_merge", BYTEAOID,
	  "s:hll_sketch_new", 1, {INT8OID},
	  {ALTFUNC_EXPR_HLL_HASH},
	  __aggfunc_property_extra_sz__hll_count,
	  0, false
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "hll_count", 1, {NUMERICOID},
	  "c:hll_merge", BYTEAOID,
	  "s:hll_sketch_new", 1, {INT8OID},
	  {ALTFUNC_EXPR_HLL_HASH},
	  __aggfunc_property_extra_sz__hll_count,
	  0, true
	},
#endif
 	{ "hll_count", 1, {DATEOID},
	  "c:hll_merge", BYTEAOID,
	  "s:hll_sketch_new", 1, {INT8OID},
	  {ALTFUNC_EXPR_HLL_HASH},
	  __aggfunc_property_extra_sz__hll_count,
	  0, false
	},
 	{ "hll_count", 1, {TIMEOID},
	  "c:hll_merge", BYTEAOID,
	  "s:hll_sketch_new", 1, {INT8OID},
	  {ALTFUNC_EXPR_HLL_HASH},
	  __aggfunc_property_extra_sz__hll_count,
	  0, false
	},
 	{ "hll_count", 1, {TIMETZOID},
	  "c:hll_merge", BYTEAOID,
	  "s:hll_sketch_new", 1, {INT8OID},
	  {ALTFUNC_EXPR_HLL_HASH},
	  __aggfunc_property_extra_sz__hll_count,
	  0, false
	},
 	{ "hll_count", 1, {TIMESTAMPOID},
	  "c:hll_merge", BYTEAOID,
	  "s:hll_sketch_new", 1, {INT8OID},
	  {ALTFUNC_EXPR_HLL_HASH},
	  __aggfunc_property_extra_sz__hll_count,
	  0, false
	},
 	{ "hll_count", 1, {TIMESTAMPTZOID},
	  "c:hll_merge", BYTEAOID,
	  "s:hll_sketch_new", 1, {INT8OID},
	  {ALTFUNC_EXPR_HLL_HASH},
	  __aggfunc_property_extra_sz__hll_count,
	  0, false
	},
 	{ "hll_count", 1, {BPCHAROID},
	  "c:hll_merge", BYTEAOID,
	  "s:hll_sketch_new", 1, {INT8OID},
	  {ALTFUNC_EXPR_HLL_HASH},
	  __aggfunc_property_extra_sz__hll_count,
	  0, false
	},
 	{ "hll_count", 1, {TEXTOID},
	  "c:hll_merge", BYTEAOID,
	  "s:hll_sketch_new", 1, {INT8OID},
	  {ALTFUNC_EXPR_HLL_HASH},
	  __aggfunc_property_extra_sz__hll_count,
	  0, false
	},
	{ "hll_count", 1, {UUIDOID},
	  "c:hll_merge", BYTEAOID,
	  "s:hll_sketch_new", 1, {INT8OID},
	  {ALTFUNC_EXPR_HLL_HASH},
	  __aggfunc_property_extra_sz__hll_count,
	  0, false
	},
	/* HLL_SKETCH(KEY) */
	{ "hll_sketch", 1, {INT1OID},
      "c:hll_combine", BYTEAOID,
      "s:hll_sketch_new", 1, {INT8OID},
      {ALTFUNC_EXPR_HLL_HASH},
      __aggfunc_property_extra_sz__hll_count,
      0, false
    },
	{ "hll_sketch", 1, {INT2OID},
      "c:hll_combine", BYTEAOID,
      "s:hll_sketch_new", 1, {INT8OID},
      {ALTFUNC_EXPR_HLL_HASH},
      __aggfunc_property_extra_sz__hll_count,
      0, false
    },
	{ "hll_sketch", 1, {INT4OID},
      "c:hll_combine", BYTEAOID,
      "s:hll_sketch_new", 1, {INT8OID},
      {ALTFUNC_EXPR_HLL_HASH},
      __aggfunc_property_extra_sz__hll_count,
      0, false
    },
	{ "hll_sketch", 1, {INT8OID},
      "c:hll_combine", BYTEAOID,
      "s:hll_sketch_new", 1, {INT8OID},
      {ALTFUNC_EXPR_HLL_HASH},
      __aggfunc_property_extra_sz__hll_count,
      0, false
    },
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "hll_sketch", 1, {NUMERICOID},
      "c:hll_combine", BYTEAOID,
      "s:hll_sketch_new", 1, {INT8OID},
      {ALTFUNC_EXPR_HLL_HASH},
      __aggfunc_property_extra_sz__hll_count,
      0, true
    },
#endif
	{ "hll_sketch", 1, {DATEOID},
	  "c:hll_combine", BYTEAOID,
	  "s:hll_sketch_new", 1, {INT8OID},
	  {ALTFUNC_EXPR_HLL_HASH},
	  __aggfunc_property_extra_sz__hll_count,
	  0, false
    },
	{ "hll_sketch", 1, {TIMEOID},
      "c:hll_combine", BYTEAOID,
      "s:hll_sketch_new", 1, {INT8OID},
      {ALTFUNC_EXPR_HLL_HASH},
      __aggfunc_property_extra_sz__hll_count,
      0, false
    },
	{ "hll_sketch", 1, {TIMETZOID},
      "c:hll_combine", BYTEAOID,
      "s:hll_sketch_new", 1, {INT8OID},
      {ALTFUNC_EXPR_HLL_HASH},
      __aggfunc_property_extra_sz__hll_count,
      0, false
    },
	{ "hll_sketch", 1, {TIMESTAMPOID},
      "c:hll_combine", BYTEAOID,
      "s:hll_sketch_new", 1, {INT8OID},
      {ALTFUNC_EXPR_HLL_HASH},
      __aggfunc_property_extra_sz__hll_count,
      0, false
    },
	{ "hll_sketch", 1, {TIMESTAMPTZOID},
      "c:hll_combine", BYTEAOID,
      "s:hll_sketch_new", 1, {INT8OID},
      {ALTFUNC_EXPR_HLL_HASH},
      __aggfunc_property_extra_sz__hll_count,
      0, false
    },
	{ "hll_sketch", 1, {BPCHAROID},
      "c:hll_combine", BYTEAOID,
      "s:hll_sketch_new", 1, {INT8OID},
      {ALTFUNC_EXPR_HLL_HASH},
      __aggfunc_property_extra_sz__hll_count,
      0, false
    },
	{ "hll_sketch", 1, {TEXTOID},
      "c:hll_combine", BYTEAOID,
      "s:hll_sketch_new", 1, {INT8OID},
      {ALTFUNC_EXPR_HLL_HASH},
      __aggfunc_property_extra_sz__hll_count,
      0, false
    },
	{ "hll_sketch", 1, {UUIDOID},
      "c:hll_combine", BYTEAOID,
      "s:hll_sketch_new", 1, {INT8OID},
      {ALTFUNC_EXPR_HLL_HASH},
      __aggfunc_property_extra_sz__hll_count,
      0, false
    },
	/* MAX(X) = MAX(PMAX(X)) */
	{ "max",    1, {INT2OID},
	  "s:fmax_int2", INT4OID,
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMAX},
	  NULL, 0, false
	},
	{ "max",    1, {INT4OID},
	  "c:max",      INT4OID,
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMAX},
	  NULL, 0, false
	},
	{ "max",    1, {INT8OID},
	  "c:max",      INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PMAX},
	  NULL, 0, false
	},
	{ "max",    1, {FLOAT4OID},
	  "c:max",      FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PMAX},
	  NULL, 0, false
	},
	{ "max",    1, {FLOAT8OID},
	  "c:max",      FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMAX},
	  NULL, 0, false
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "max",    1, {NUMERICOID},
	  "s:fmax_numeric", FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMAX},
	  NULL, 0, true
	},
#endif
	{ "max",    1, {CASHOID},
	  "c:max",      CASHOID,
	  "varref", 1, {CASHOID},
	  {ALTFUNC_EXPR_PMAX},
	  NULL, DEVKERNEL_NEEDS_MISCLIB, false
	},
	{ "max",    1, {DATEOID},
	  "c:max",      DATEOID,
	  "varref", 1, {DATEOID},
	  {ALTFUNC_EXPR_PMAX},
	  NULL, 0, false
	},
	{ "max",    1, {TIMEOID},
	  "c:max",      TIMEOID,
	  "varref", 1, {TIMEOID},
	  {ALTFUNC_EXPR_PMAX},
	  NULL, 0, false
	},
	{ "max",    1, {TIMESTAMPOID},
	  "c:max",      TIMESTAMPOID,
	  "varref", 1, {TIMESTAMPOID},
	  {ALTFUNC_EXPR_PMAX},
	  NULL, 0, false
	},
	{ "max",    1, {TIMESTAMPTZOID},
	  "c:max",      TIMESTAMPTZOID,
	  "varref", 1, {TIMESTAMPTZOID},
	  {ALTFUNC_EXPR_PMAX},
	  NULL, 0, false
	},
	/* MIX(X) = MIN(PMIN(X)) */
	{ "min",    1, {INT2OID},
	  "s:fmin_int2", INT4OID,
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMIN},
	  NULL, 0, false
	},
	{ "min",    1, {INT4OID},
	  "c:min",      INT4OID,
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMIN},
	  NULL, 0, false
	},
	{ "min",    1, {INT8OID},
	  "c:min",      INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PMIN},
	  NULL, 0, false
	},
	{ "min",    1, {FLOAT4OID},
	  "c:min",      FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PMIN},
	  NULL, 0, false
	},
	{ "min",    1, {FLOAT8OID},
	  "c:min",      FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMIN},
	  NULL, 0, false
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "min",    1, {NUMERICOID},
	  "s:fmin_numeric", FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMIN},
	  NULL, 0, true
	},
#endif
	{ "min",    1, {CASHOID},
	  "c:min",      CASHOID,
	  "varref", 1, {CASHOID},
	  {ALTFUNC_EXPR_PMAX},
	  NULL, DEVKERNEL_NEEDS_MISCLIB, false
	},
	{ "min",    1, {DATEOID},
	  "c:min",      DATEOID,
	  "varref", 1, {DATEOID},
	  {ALTFUNC_EXPR_PMIN},
	  NULL, 0, false
	},
	{ "min",    1, {TIMEOID},
	  "c:min",      TIMEOID,
	  "varref", 1, {TIMEOID},
	  {ALTFUNC_EXPR_PMIN},
	  NULL, 0, false
	},
	{ "min",    1, {TIMESTAMPOID},
	  "c:min",      TIMESTAMPOID,
	  "varref", 1, {TIMESTAMPOID},
	  {ALTFUNC_EXPR_PMIN},
	  NULL, 0, false
	},
	{ "min",    1, {TIMESTAMPTZOID},
	  "c:min",      TIMESTAMPTZOID,
	  "varref", 1, {TIMESTAMPTZOID},
	  {ALTFUNC_EXPR_PMIN},
	  NULL, 0, false
	},

	/* SUM(X) = SUM(PSUM(X)) */
	{ "sum",    1, {INT2OID},
	  "s:sum",      INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM},
	  NULL, 0, false
	},
	{ "sum",    1, {INT4OID},
	  "s:sum",      INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM},
	  NULL, 0, false
	},
	{ "sum",    1, {INT8OID},
	  "c:sum",      INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM},
	  NULL, 0, false
	},
	{ "sum",    1, {FLOAT4OID},
	  "c:sum",      FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PSUM},
	  NULL, 0, false
	},
	{ "sum",    1, {FLOAT8OID},
	  "c:sum",      FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PSUM},
	  NULL, 0, false
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "sum",    1, {NUMERICOID},
	  "s:fsum_numeric", FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PSUM},
	  NULL, 0, true
	},
#endif
	{ "sum",    1, {CASHOID},
	  "c:sum",      CASHOID,
	  "varref", 1, {CASHOID},
	  {ALTFUNC_EXPR_PSUM},
	  NULL, DEVKERNEL_NEEDS_MISCLIB, false
	},
	/* STDDEV(X) = EX_STDDEV(NROWS(),PSUM(X),PSUM(X*X)) */
	{ "stddev",      1, {INT2OID},
	  "s:stddev_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "stddev",      1, {INT4OID},
	  "s:stddev_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "stddev",      1, {INT8OID},
	  "s:stddev_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "stddev",      1, {FLOAT4OID},
	  "s:stddev",        FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "stddev",      1, {FLOAT8OID},
	  "s:stddev",        FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "stddev",      1, {NUMERICOID},
	  "s:stddev_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
#endif
	/* STDDEV_POP(X) = EX_STDDEV(NROWS(),PSUM(X),PSUM(X*X)) */
	{ "stddev_pop",  1, {INT2OID},
	  "s:stddev_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "stddev_pop",  1, {INT4OID},
	  "s:stddev_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "stddev_pop",  1, {INT8OID},
	  "s:stddev_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "stddev_pop",  1, {FLOAT4OID},
	  "s:stddev_pop",    FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "stddev_pop",  1, {FLOAT8OID},
	  "s:stddev_pop",    FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "stddev_pop",  1, {NUMERICOID},
	  "s:stddev_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
#endif
	/* STDDEV_POP(X) = EX_STDDEV(NROWS(),PSUM(X),PSUM(X*X)) */
	{ "stddev_samp", 1, {INT2OID},
	  "s:stddev_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "stddev_samp", 1, {INT4OID},
	  "s:stddev_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "stddev_samp", 1, {INT8OID},
	  "s:stddev_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "stddev_samp", 1, {FLOAT4OID},
	  "s:stddev_samp",   FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "stddev_samp", 1, {FLOAT8OID},
	  "s:stddev_samp",   FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "stddev_samp", 1, {NUMERICOID},
	  "s:stddev_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
#endif
	/* VARIANCE(X) = PGSTROM.VARIANCE(NROWS(), PSUM(X),PSUM(X^2)) */
	{ "variance",    1, {INT2OID},
	  "s:variance_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "variance",    1, {INT4OID},
	  "s:variance_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "variance",    1, {INT8OID},
	  "s:variance_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "variance",    1, {FLOAT4OID},
	  "s:variance",      FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "variance",    1, {FLOAT8OID},
	  "s:variance",      FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "variance",    1, {NUMERICOID},
	  "s:variance_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
#endif
	/* VAR_POP(X) = PGSTROM.VAR_POP(NROWS(), PSUM(X),PSUM(X^2)) */
	{ "var_pop",     1, {INT2OID},
	  "s:var_pop_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "var_pop",     1, {INT4OID},
	  "s:var_pop_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "var_pop",     1, {INT8OID},
	  "s:var_pop_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "var_pop",     1, {FLOAT4OID},
	  "s:var_pop",       FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "var_pop",     1, {FLOAT8OID},
	  "s:var_pop",       FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "var_pop",     1, {NUMERICOID},
	  "s:var_pop_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
#endif
	/* VAR_SAMP(X) = PGSTROM.VAR_SAMP(NROWS(), PSUM(X),PSUM(X^2)) */
	{ "var_samp",    1, {INT2OID},
	  "s:var_samp_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "var_samp",    1, {INT4OID},
	  "s:var_samp_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "var_samp",    1, {INT8OID},
	  "s:var_samp_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "var_samp",    1, {FLOAT4OID},
	  "s:var_samp",      FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
	{ "var_samp",    1, {FLOAT8OID},
	  "s:var_samp",      FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "var_samp",    1, {NUMERICOID},
	  "s:var_samp_numeric", FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2},
	  NULL, 0, false
	},
#endif
	/*
	 * CORR(X,Y) = PGSTROM.CORR(NROWS(X,Y),
	 *                          PCOV_X(X,Y),  PCOV_Y(X,Y)
	 *                          PCOV_X2(X,Y), PCOV_Y2(X,Y),
	 *                          PCOV_XY(X,Y))
	 */
	{ "corr",     2, {FLOAT8OID, FLOAT8OID},
	  "s:corr",       FLOAT8ARRAYOID,
	  "s:pcovar", 6,
	  {INT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  NULL, 0, false
	},
	{ "covar_pop", 2, {FLOAT8OID, FLOAT8OID},
	  "s:covar_pop",   FLOAT8ARRAYOID,
	  "s:pcovar", 6,
	  {INT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  NULL, 0, false
	},
	{ "covar_samp", 2, {FLOAT8OID, FLOAT8OID},
	  "s:covar_samp",   FLOAT8ARRAYOID,
	  "s:pcovar", 6,
	  {INT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  NULL, 0, false
	},
	/*
	 * Aggregation to support least squares method
	 *
	 * That takes PSUM_X, PSUM_Y, PSUM_X2, PSUM_Y2, PSUM_XY according
	 * to the function
	 */
	{ "regr_avgx", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_avgx",   FLOAT8ARRAYOID,
	  "s:pcovar", 6,
	  {INT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PCOV_X,
       ALTFUNC_EXPR_PCOV_X2,
       ALTFUNC_EXPR_PCOV_Y,
       ALTFUNC_EXPR_PCOV_Y2,
       ALTFUNC_EXPR_PCOV_XY},
	  NULL, 0, false
	},
	{ "regr_avgy", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_avgy",   FLOAT8ARRAYOID,
	  "s:pcovar", 6,
	  {INT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  NULL, 0, false
	},
	{ "regr_count", 2, {FLOAT8OID, FLOAT8OID},
	  "s:sum",      INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_NROWS}, 0
	},
	{ "regr_intercept", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_intercept",   FLOAT8ARRAYOID,
	  "s:pcovar", 6,
	  {INT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  NULL, 0, false
	},
	{ "regr_r2", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_r2",   FLOAT8ARRAYOID,
	  "s:pcovar", 6,
	  {INT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  NULL, 0, false
	},
	{ "regr_slope", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_slope",   FLOAT8ARRAYOID,
	  "s:pcovar", 6,
	  {INT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  NULL, 0, false
	},
	{ "regr_sxx", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_sxx",   FLOAT8ARRAYOID,
	  "s:pcovar", 6,
	  {INT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  NULL, 0, false
	},
	{ "regr_sxy", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_sxy",   FLOAT8ARRAYOID,
	  "s:pcovar", 6,
	  {INT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  NULL, 0, false
	},
	{ "regr_syy", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_syy",   FLOAT8ARRAYOID,
	  "s:pcovar", 6,
	  {INT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY},
	  NULL, 0, false
	},
};

static const aggfunc_catalog_t *
aggfunc_lookup_by_oid(Oid aggfnoid)
{
	Form_pg_proc	proc;
	HeapTuple		htup;
	int				i;

	htup = SearchSysCache1(PROCOID, ObjectIdGetDatum(aggfnoid));
	if (!HeapTupleIsValid(htup))
		elog(ERROR, "cache lookup failed for function %u", aggfnoid);
	proc = (Form_pg_proc) GETSTRUCT(htup);
	if (proc->pronamespace == PG_CATALOG_NAMESPACE)
	{
		for (i=0; i < lengthof(aggfunc_catalog); i++)
		{
			aggfunc_catalog_t  *catalog = &aggfunc_catalog[i];

			if (strcmp(catalog->aggfn_name, NameStr(proc->proname)) == 0 &&
				catalog->aggfn_nargs == proc->pronargs &&
				memcmp(catalog->aggfn_argtypes,
					   proc->proargtypes.values,
					   sizeof(Oid) * catalog->aggfn_nargs) == 0)
			{
				/* Is NUMERIC with GpuPreAgg acceptable? */
				if (catalog->numeric_aware && !enable_numeric_aggfuncs)
					continue;
				/* all ok */
				ReleaseSysCache(htup);
				return catalog;
			}
		}
	}
	ReleaseSysCache(htup);
	return NULL;
}

/*
 * cost_gpupreagg
 *
 * cost estimation for GpuPreAgg node
 */
static bool
cost_gpupreagg(PlannerInfo *root,
			   CustomPath *cpath,
			   GpuPreAggInfo *gpa_info,
			   PathTarget *target_partial,
			   Path *input_path,
			   int parallel_nworkers,
			   double num_groups,
			   IndexOptInfo *index_opt,
			   List *index_quals,
			   cl_long index_nblocks)
{
	double		gpu_cpu_ratio = pgstrom_gpu_operator_cost / cpu_operator_cost;
	Cost		startup_cost;
	Cost		run_cost;
	int			num_group_keys = 0;
	int			j, ncols;

	/* Cost come from the underlying path */
	if (gpa_info->outer_scanrelid == 0)
	{
		RelOptInfo *outer_rel = input_path->parent;
		Cost	outer_startup = input_path->startup_cost;
		Cost	outer_total   = input_path->total_cost;
		List   *outer_tlist   = input_path->pathtarget->exprs;
		Cost	discount = 0.0;

		/*
		 * Discount cost for DMA-receive if GpuPreAgg can pull-up
		 * outer GpuJoin.
		 */
		if (enable_pullup_outer_join &&
			pgstrom_path_is_gpujoin(input_path) &&
			pgstrom_device_expression(root, outer_rel, (Expr *)outer_tlist))
		{
			discount = (cost_for_dma_receive(input_path->parent, -1.0) +
						cpu_tuple_cost * input_path->rows);
		}
		else if (pathtree_has_gpupath(input_path))
			outer_total += pgstrom_gpu_setup_cost / 2;
		else
			outer_total += pgstrom_gpu_setup_cost;

		gpa_info->outer_startup_cost = Max(outer_startup - discount, 0.0);
		gpa_info->outer_total_cost   = Max(outer_total - discount, 0.0);

		startup_cost = outer_total;
		run_cost = 0.0;
	}
	else
	{
		double		parallel_divisor;
		double		ntuples;
		double		nchunks;

		pgstrom_common_relscan_cost(root,
									input_path->parent,
									gpa_info->outer_quals,
									parallel_nworkers,	/* parallel scan */
									index_opt,			/* BRIN-index */
									index_quals,		/* BRIN-index */
									index_nblocks,		/* BRIN-index */
									&ntuples,
									&nchunks,
									&parallel_divisor,
									&gpa_info->outer_nrows_per_block,
									&startup_cost,
									&run_cost);
		run_cost -= cpu_tuple_cost * ntuples;
		if (run_cost < 0.0)
			run_cost = 0.0;
		gpa_info->outer_startup_cost = startup_cost;
		gpa_info->outer_total_cost	= startup_cost + run_cost;

		startup_cost += run_cost;
		run_cost = 0.0;
	}
	gpa_info->outer_nrows		= input_path->rows;
	gpa_info->outer_width		= input_path->pathtarget->width;

	/*
	 * Estimation of the result buffer. It must fit to the target GPU device
	 * memory size.
	 */
	ncols = list_length(target_partial->exprs);
	for (j=0; j < ncols; j++)
	{
		if (get_pathtarget_sortgroupref(target_partial, j))
			num_group_keys++;
	}
	if (num_group_keys == 0)
		num_groups = 1.0;	/* AGG_PLAIN */

	/* Cost estimation for grouping */
	startup_cost += (pgstrom_gpu_operator_cost *
					 num_group_keys *
					 input_path->rows);
	/* Cost estimation for aggregate function */
	startup_cost += (target_partial->cost.per_tuple * input_path->rows +
					 target_partial->cost.startup) * gpu_cpu_ratio;

	/* Cost estimation to fetch results */
	run_cost += cpu_tuple_cost * num_groups;

	cpath->path.rows			= num_groups;
	cpath->path.startup_cost	= startup_cost;
	cpath->path.total_cost		= startup_cost + run_cost;

	gpa_info->num_group_keys    = num_group_keys;
	gpa_info->plan_ngroups		= num_groups;
	gpa_info->plan_nchunks		= estimate_num_chunks(input_path);

	return true;
}

#if PG_VERSION_NUM < 120000
/*
 * estimate_hashagg_tablesize - had been declared as a static function
 * until PG12 at the optimizer/plan/planner.c.
 */
static double
estimate_hashagg_tablesize(PlannerInfo *root, Path *path,
						   const AggClauseCosts *agg_costs,
                           double dNumGroups)
{
	Size		hashentrysize;

	/* Estimate per-hash-entry space at tuple width... */
	hashentrysize = MAXALIGN(path->pathtarget->width) +
		MAXALIGN(SizeofMinimalTupleHeader);

	/* plus space for pass-by-ref transition values... */
	hashentrysize += agg_costs->transitionSpace;
	/* plus the per-hash-entry overhead */
	hashentrysize += hash_agg_entry_size(agg_costs->numAggs);

	return hashentrysize * dNumGroups;
}
#elif PG_VERSION_NUM < 140000
#define estimate_hashagg_tablesize(a,b,c,d)		\
	estimate_hashagg_tablesize((b),(c),(d))
#endif

/*
 * make_gpupreagg_path
 *
 * constructor of the GpuPreAgg path node
 */
static CustomPath *
make_gpupreagg_path(PlannerInfo *root,
					RelOptInfo *group_rel,
					PathTarget *target_partial,
					Path *input_path,
					double num_groups,
					bool can_pullup_outerscan)
{
	CustomPath	   *cpath = makeNode(CustomPath);
	GpuPreAggInfo  *gpa_info = palloc0(sizeof(GpuPreAggInfo));
	List		   *custom_paths = NIL;
	int				parallel_nworkers = 0;
	IndexOptInfo   *index_opt = NULL;
	List		   *index_conds = NIL;
	List		   *index_quals = NIL;
	cl_long			index_nblocks;

	/* obviously, not suitable for GpuPreAgg */
	if (num_groups < 1.0 || num_groups > (double)INT_MAX)
		return NULL;

	/* Try to pull up input_path if simple relation scan */
	gpa_info->optimal_gpus = NULL;
	if (!can_pullup_outerscan ||
		!pgstrom_pullup_outer_scan(root, input_path,
								   &gpa_info->outer_scanrelid,
								   &gpa_info->outer_quals,
								   &gpa_info->optimal_gpus,
								   &index_opt,
								   &index_conds,
								   &index_quals,
								   &index_nblocks))
	{
		if (pgstrom_path_is_gpujoin(input_path))
			gpa_info->optimal_gpus = gpujoin_get_optimal_gpus(input_path);
		custom_paths = list_make1(input_path);
	}

	/* Number of workers if parallel */
	if (group_rel->consider_parallel &&
		input_path->parallel_safe)
		parallel_nworkers = input_path->parallel_workers;

	/* cost estimation */
	if (!cost_gpupreagg(root,
						cpath,
						gpa_info,
						target_partial,
						input_path,
						parallel_nworkers,
						num_groups,
						index_opt,
						index_quals,
						index_nblocks))
	{
		pfree(cpath);
		return NULL;
	}
	/* BRIN-index options */
	gpa_info->index_oid = (index_opt ? index_opt->indexoid : InvalidOid);
	gpa_info->index_conds = index_conds;
	gpa_info->index_quals = extract_actual_clauses(index_quals, false);

	/* Setup CustomPath */
	cpath->path.pathtype = T_CustomScan;
	cpath->path.parent = input_path->parent;
	cpath->path.pathtarget = target_partial;
	cpath->path.param_info = NULL;
	cpath->path.parallel_safe = (group_rel->consider_parallel &&
								 input_path->parallel_safe);
	cpath->path.parallel_workers = parallel_nworkers;
	cpath->path.pathkeys = NIL;
	cpath->custom_paths = custom_paths;
	cpath->custom_private = list_make1(gpa_info);
	cpath->methods = &gpupreagg_path_methods;

	return cpath;
}

/*
 * prepend_gpupreagg_path
 */
static Path *
prepend_gpupreagg_path(PlannerInfo *root,
					   RelOptInfo *group_rel,
					   PathTarget *target_partial,
					   Path *input_path,
					   double num_groups,
					   bool can_pullup_outerscan,
					   bool with_gather_node)
{
	CustomPath *cpath;
	Path	   *partial_path;

	/*
	 * MEMO: See grouping_planner() where it calls create_projection_path()
	 * on the partial Path-nodes. It forcibly injects ProjectionPath to
	 * have individual PathTarget on top of the scan/join paths, because
	 * these scan/join paths may be referenced by other path-trees, thus
	 * it is unable to set PathTarget in-place.
	 * In fact, create_projection_plan() pulls up sub-plan and attach its
	 * target-list if it can be compatible, however, this ProjectionPath
	 * prevents GpuJoin to have compatible host_tlist and dev_tlist, and
	 * leads host-side projection. It works as a blocker of combined-GpuJoin
	 * which is one of the performance key of reporting queries.
	 * So, we duplicate GpuJoinPath by ourself, and set PathTarget of
	 * the ProjectionPath here.
	 */
	if (IsA(input_path, ProjectionPath))
	{
		ProjectionPath *pjpath = (ProjectionPath *)input_path;
		RelOptInfo	   *pjrel = pjpath->path.parent;
		PathTarget	   *pathtarget = pjpath->path.pathtarget;

		if (pjpath->dummypp &&
			pgstrom_path_is_gpujoin(pjpath->subpath) &&
			pgstrom_device_expression(root, pjrel, (Expr *)pathtarget->exprs))
		{
			input_path = pgstrom_copy_gpujoin_path(pjpath->subpath);
			input_path->pathtarget = pathtarget;
		}
	}

	/*
	 * construction of GpuPreAgg pathnode on top of the cheapest total
	 * cost pathnode (partial aggregation)
	 */
	cpath = make_gpupreagg_path(root, group_rel,
								target_partial,
								input_path,
								num_groups,
								can_pullup_outerscan);
	if (!cpath)
		return NULL;

	/* Is it parallel capable? */
	if (cpath->path.parallel_safe &&
		cpath->path.parallel_workers > 0)
	{
		cpath->path.parallel_aware = true;

		if (with_gather_node)
		{
			double	total_groups = (cpath->path.rows *
									cpath->path.parallel_workers);
			partial_path = (Path *)create_gather_path(root,
													  group_rel,
													  &cpath->path,
													  target_partial,
													  NULL,
													  &total_groups);
		}
		else
			partial_path = &cpath->path;
	}
	else if (with_gather_node)
		partial_path = NULL;
	else
		partial_path = &cpath->path;

	return partial_path;
}

/*
 * try_add_final_aggregation_paths
 */
static void
try_add_final_aggregation_paths(PlannerInfo *root,
								RelOptInfo *group_rel,
								PathTarget *target_final,
								Path *partial_path,
								List *havingQuals,
								double num_groups,
								AggClauseCosts *final_clause_costs)
{
	Query	   *parse = root->parse;
	Path	   *sort_path;
	Path	   *final_path;
	bool		can_sort;
	bool		can_hash;

	/* strategy of the final aggregation */
	can_sort = grouping_is_sortable(parse->groupClause);
	can_hash = (parse->groupClause != NIL &&
				parse->groupingSets == NIL &&
				grouping_is_hashable(parse->groupClause));

	/* make a final grouping path (nogroup) */
	if (!parse->groupClause)
	{
		final_path = (Path *)create_agg_path(root,
											 group_rel,
											 partial_path,
											 target_final,
											 AGG_PLAIN,
											 AGGSPLIT_SIMPLE,
											 parse->groupClause,
											 havingQuals,
											 final_clause_costs,
											 num_groups);
		final_path = pgstrom_create_dummy_path(root, final_path);
		add_path(group_rel, final_path);
	}
	else
	{
		/* make a final grouping path (sort) */
		if (can_sort)
		{
			PathTarget *target_orig __attribute__((unused));

			sort_path = (Path *)
				create_sort_path(root,
								 group_rel,
								 partial_path,
								 root->group_pathkeys,
								 -1.0);
			if (parse->groupingSets)
			{
				AggStrategy	rollup_strategy = AGG_PLAIN;
				List	   *rollup_data_list = NIL;
				ListCell   *lc;
				/*
				 * TODO: In this version, we expect group_rel->pathlist have
				 * a GroupingSetsPath constructed by the built-in code.
				 * It may not be right, if multiple CSP/FDW is installed and
				 * cheaper path already eliminated the standard path.
				 * However, it is a corner case now, and we don't support
				 * this scenario _right now_.
				 */
				foreach (lc, group_rel->pathlist)
				{
					GroupingSetsPath   *pathnode = lfirst(lc);

					if (IsA(pathnode, GroupingSetsPath))
					{
						rollup_strategy = pathnode->aggstrategy;
						rollup_data_list = pathnode->rollups;
						break;
					}
				}
				if (!lc)
					return;		/* give up */
				final_path = (Path *)
					create_groupingsets_path(root,
											 group_rel,
											 sort_path,
											 (List *) parse->havingQual,
											 rollup_strategy,
											 rollup_data_list,
											 final_clause_costs,
											 num_groups);
				/* adjust cost and overwrite PathTarget */
				target_orig = final_path->pathtarget;
				final_path->startup_cost += (target_final->cost.startup -
											 target_orig->cost.startup);
				final_path->total_cost += (target_final->cost.startup -
										   target_orig->cost.startup) +
					(target_final->cost.per_tuple -
					 target_orig->cost.per_tuple) * final_path->rows;
				final_path->pathtarget = target_final;
			}
			else if (parse->hasAggs)
				final_path = (Path *)
					create_agg_path(root,
									group_rel,
									sort_path,
									target_final,
									AGG_SORTED,
									AGGSPLIT_SIMPLE,
									parse->groupClause,
								    havingQuals,
									final_clause_costs,
									num_groups);
			else if (parse->groupClause)
			{
				final_path = (Path *)
					create_group_path(root,
									  group_rel,
									  sort_path,
									  parse->groupClause,
									  havingQuals,
									  num_groups);
				/* adjust cost and overwrite PathTarget */
				target_orig = final_path->pathtarget;
				final_path->startup_cost += (target_final->cost.startup -
											 target_orig->cost.startup);
				final_path->total_cost += (target_final->cost.startup -
										   target_orig->cost.startup) +
					(target_final->cost.per_tuple -
					 target_orig->cost.per_tuple) * final_path->rows;
				final_path->pathtarget = target_final;
			}
			else
				elog(ERROR, "Bug? unexpected AGG/GROUP BY requirement");

			final_path = pgstrom_create_dummy_path(root, final_path);
			add_path(group_rel, final_path);
		}

		/* make a final grouping path (hash) */
		if (can_hash)
		{
			double	hashaggtablesize
				= estimate_hashagg_tablesize(root,
											 partial_path,
											 final_clause_costs,
											 num_groups);
			if (hashaggtablesize < work_mem * 1024L)
			{
				final_path = (Path *)
					create_agg_path(root,
									group_rel,
									partial_path,
									target_final,
									AGG_HASHED,
									AGGSPLIT_SIMPLE,
									parse->groupClause,
									havingQuals,
									final_clause_costs,
									num_groups);
				final_path = pgstrom_create_dummy_path(root, final_path);
				add_path(group_rel, final_path);
			}
		}
	}
}

/*
 * try_add_gpupreagg_append_paths
 */
static void
try_add_gpupreagg_append_paths(PlannerInfo *root,
							   RelOptInfo *group_rel,
							   PathTarget *target_final,
							   PathTarget *target_partial,
							   Path *input_path,
							   List *havingQual,
							   double num_groups,
							   AggClauseCosts *agg_final_costs,
							   bool can_pullup_outerscan,
							   bool try_outer_parallel)
{
#if PG_VERSION_NUM >= 110000
	bool		try_inner_parallel = false;
	List	   *append_paths_list = NIL;
	List	   *sub_paths_list;
	int			parallel_nworkers;
	AppendPath *append_path;
	Cost		discount_cost;
	Path	   *partial_path;
	ListCell   *lc;

retry:
	sub_paths_list = extract_partitionwise_pathlist(root,
													input_path,
													try_outer_parallel,
													false,
													&append_path,
													&parallel_nworkers,
													&discount_cost);
	if (sub_paths_list == NIL)
		return;
	if (list_length(sub_paths_list) > 1)
		discount_cost /= (Cost)(list_length(sub_paths_list) - 1);
	else
		discount_cost = 0.0;

	foreach (lc, sub_paths_list)
	{
		Path	   *sub_path = (Path *) lfirst(lc);
		RelOptInfo *sub_rel = sub_path->parent;
		PathTarget *curr_partial = copy_pathtarget(target_partial);
		Path	   *partial_path;
		AppendRelInfo **appinfos;
		int				nappinfos;

		appinfos = find_appinfos_by_relids_nofail(root, sub_rel->relids,
												  &nappinfos);
		/* fixup varno */
		curr_partial->exprs = (List *)
			adjust_appendrel_attrs(root, (Node *)curr_partial->exprs,
								   nappinfos, appinfos);

		partial_path = prepend_gpupreagg_path(root,
											  group_rel,
											  curr_partial,
											  sub_path,
											  num_groups,
											  can_pullup_outerscan,
											  false);
		if (!partial_path)
			return;
		partial_path->total_cost -= discount_cost;
		append_paths_list = lappend(append_paths_list, partial_path);
	}
	/* also see create_append_path(), some fields must be fixed up */
	if (try_outer_parallel)
		append_path = create_append_path(root, input_path->parent,
										 NIL, append_paths_list,
										 NIL, NULL,
										 parallel_nworkers, true,
#if PG_VERSION_NUM < 140000
										 append_path->partitioned_rels,
#endif
										 -1.0);
	else
		append_path = create_append_path(root, input_path->parent,
										 append_paths_list, NIL,
										 NIL, NULL,
										 parallel_nworkers, false,
#if PG_VERSION_NUM < 140000
										 append_path->partitioned_rels,
#endif
										 -1.0);
	append_path->path.pathtarget = target_partial;
	append_path->path.total_cost -= discount_cost;

	/* prepend Gather on demand */
	if (try_outer_parallel &&
		append_path->path.parallel_safe &&
		append_path->path.parallel_workers > 0)
	{
		double		total_groups = (append_path->path.rows *
									append_path->path.parallel_workers);
		append_path->path.parallel_aware = true;
		partial_path = (Path *)create_gather_path(root,
												  group_rel,
												  &append_path->path,
												  target_partial,
												  NULL,
												  &total_groups);
	}
	else
	{
		partial_path = &append_path->path;
		try_outer_parallel = false;		/* never retry again */
	}

	try_add_final_aggregation_paths(root,
									group_rel,
									target_final,
									partial_path,
									(List *) havingQual,
									num_groups,
									agg_final_costs);
	if (try_outer_parallel && !try_inner_parallel)
	{
		try_inner_parallel = true;
		goto retry;
	}
#endif	/* PG_VERSION_NUM >= 110000 */
}

/*
 * try_add_gpupreagg_paths
 */
static void
try_add_gpupreagg_paths(PlannerInfo *root,
						RelOptInfo *group_rel,
						Path *input_path,
						bool try_parallel_path)
{
	Query		   *parse = root->parse;
	PathTarget	   *target_upper	= root->upper_targets[UPPERREL_GROUP_AGG];
	PathTarget	   *target_final	= create_empty_pathtarget();
	PathTarget	   *target_partial	= create_empty_pathtarget();
	Path		   *partial_path;
	Node		   *havingQual;
	double			num_groups;
	double			reduction_ratio;
	bool			can_pullup_outerscan = true;
	AggClauseCosts	final_clause_costs;

	/*
	 * MEMO: The 'pg_strom.gpupreagg_reduction_threshold' is a tentative
	 * solution to avoid overflow of GPU device memory for GpuPreAgg.
	 * In case of large table scan with small reduction ratio is almost
	 * equivalent to cache most of input records in GPU device memory.
	 * Then, it shall be aggregated on CPU-side again, it is usually waste
	 * of computing power and data transfer.
	 * Of course, a threshold is not a perfect solution. We may need to
	 * switch to bypass GPU once reduction ratio (or absolute data size) is
	 * worse than the estimation at the planning stage.
	 */
	if (!parse->groupClause)
		num_groups = 1.0;
	else
	{
		Path   *pathnode = linitial(group_rel->pathlist);

		num_groups = Max(pathnode->rows, 1.0);
	}
	reduction_ratio = input_path->rows / num_groups;
	if (reduction_ratio < gpupreagg_reduction_threshold)
	{
		elog(DEBUG2, "GpuPreAgg: %.0f -> %.0f reduction ratio (%.2f) is bad",
			 input_path->rows, num_groups, reduction_ratio);
		return;
	}

	/* construction of the target-list for each level */
	if (!gpupreagg_build_path_target(root,
									 target_upper,
									 target_final,
									 target_partial,
									 input_path,
									 &havingQual,
									 &can_pullup_outerscan,
									 &final_clause_costs))
		return;

	if (enable_partitionwise_gpupreagg)
		try_add_gpupreagg_append_paths(root,
									   group_rel,
									   target_final,
									   target_partial,
									   input_path,
									   (List *) havingQual,
									   num_groups,
									   &final_clause_costs,
									   can_pullup_outerscan,
									   try_parallel_path);

	partial_path = prepend_gpupreagg_path(root,
										  group_rel,
										  target_partial,
										  input_path,
										  num_groups,
										  can_pullup_outerscan,
										  try_parallel_path);
	if (!partial_path ||
		(try_parallel_path && !IsA(partial_path, GatherPath)))
		return;

	try_add_final_aggregation_paths(root,
									group_rel,
									target_final,
									partial_path,
									(List *) havingQual,
									num_groups,
									&final_clause_costs);
}

/*
 * gpupreagg_add_grouping_paths
 *
 * entrypoint to add grouping path by GpuPreAgg logic
 */
static void
gpupreagg_add_grouping_paths(PlannerInfo *root,
							 UpperRelationKind stage,
							 RelOptInfo *input_rel,
							 RelOptInfo *group_rel,
							 void *extra)
{
	Path	   *input_path;
	ListCell   *lc;

	if (create_upper_paths_next)
		(*create_upper_paths_next)(root, stage, input_rel, group_rel, extra);

	if (stage != UPPERREL_GROUP_AGG)
		return;

	if (!pgstrom_enabled || !enable_gpupreagg)
		return;

	/* CREATE EXTENSION pg_strom; was not executed */
	if (get_namespace_oid("pgstrom", true) == InvalidOid)
		return;

	/* traditional GpuPreAgg + Agg path consideration */
	input_path = input_rel->cheapest_total_path;
	try_add_gpupreagg_paths(root, group_rel, input_path, false);

	/*
	 * add GpuPreAgg + Gather + Agg path for CPU+GPU hybrid parallel
	 */
	if (group_rel->consider_parallel)
	{
		foreach (lc, input_rel->partial_pathlist)
		{
			input_path = lfirst(lc);
			try_add_gpupreagg_paths(root, group_rel, input_path, true);
		}
	}
}

/*
 * replace_expression_by_outerref
 *
 * It transforms expression into the form of execution time.
 * Even if expression contains device non-executable portion, it is possible
 * to run as long as it is calculated on the sub-plan.
 */
static Node *
replace_expression_by_outerref(Node *node, PathTarget *target_input)
{
	ListCell   *lc;
	cl_int		resno = 1;

	if (!node)
		return NULL;
	foreach (lc, target_input->exprs)
	{
		if (equal(node, lfirst(lc)))
		{
			Var	   *newnode;

			newnode = makeVar(INDEX_VAR,
							  resno,
							  exprType(node),
							  exprTypmod(node),
							  exprCollation(node),
							  0);
			if (IsA(node, Var))
			{
				Var	   *varnode = (Var *) node;

				newnode->varnosyn  = varnode->varno;
				newnode->varattnosyn = varnode->varattno;
			}
			return (Node *) newnode;
		}
		resno++;
	}

	if (IsA(node, Var))
		elog(ERROR, "Bug? Var-node didn'd appear on the input targetlist: %s",
			 nodeToString(node));

	return expression_tree_mutator(node,
								   replace_expression_by_outerref,
								   target_input);
}

/*
 * is_altfunc_expression - true, if expression derives ALTFUNC_EXPR_*
 */
static bool
is_altfunc_expression(Node *node, int *p_extra_sz)
{
	FuncExpr	   *f;
	HeapTuple		tuple;
	Form_pg_proc	form_proc;
	int				extra_sz = 0;
	bool			retval = false;

	if (!IsA(node, FuncExpr))
		return false;
	f = (FuncExpr *) node;

	tuple = SearchSysCache1(PROCOID, ObjectIdGetDatum(f->funcid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for function %u", f->funcid);
	form_proc = (Form_pg_proc) GETSTRUCT(tuple);

	if (form_proc->pronamespace == get_namespace_oid("pgstrom", false))
	{
		if (strcmp(NameStr(form_proc->proname), "nrows") == 0 ||
			strcmp(NameStr(form_proc->proname), "pmin") == 0 ||
			strcmp(NameStr(form_proc->proname), "pmax") == 0 ||
			strcmp(NameStr(form_proc->proname), "psum") == 0 ||
			strcmp(NameStr(form_proc->proname), "psum_x2") == 0 ||
			strcmp(NameStr(form_proc->proname), "pcov_x") == 0 ||
			strcmp(NameStr(form_proc->proname), "pcov_y") == 0 ||
			strcmp(NameStr(form_proc->proname), "pcov_x2") == 0 ||
			strcmp(NameStr(form_proc->proname), "pcov_y2") == 0 ||
			strcmp(NameStr(form_proc->proname), "pcov_xy") == 0)
		{
			retval = true;
		}
		else if (strcmp(NameStr(form_proc->proname), "hll_sketch_new") == 0)
		{
			extra_sz = __aggfunc_property_extra_sz__hll_count();
			retval = true;
		}
	}
	ReleaseSysCache(tuple);

	if (p_extra_sz)
		*p_extra_sz = extra_sz;

	return retval;
}

/*
 * make_expr_typecast - constructor of type cast
 */
static Expr *
make_expr_typecast(Expr *expr, Oid target_type)
{
	Oid			source_type = exprType((Node *) expr);
	HeapTuple	tup;
	Form_pg_cast cast;

	/*
	 * NOTE: Var->vano shall be replaced to INDEX_VAR on the following
	 * make_altfunc_expr(), so we keep the expression as-is, at this
	 * moment.
	 */
	if (source_type == target_type)
		return expr;

	tup = SearchSysCache2(CASTSOURCETARGET,
						  ObjectIdGetDatum(source_type),
						  ObjectIdGetDatum(target_type));
	Assert(HeapTupleIsValid(tup));
	if (!HeapTupleIsValid(tup))
		elog(ERROR, "could not find tuple for cast (%u,%u)",
			 source_type, target_type);
	cast = (Form_pg_cast) GETSTRUCT(tup);
	if (cast->castmethod == COERCION_METHOD_FUNCTION)
	{
		FuncExpr	   *func;

		Assert(OidIsValid(cast->castfunc));
		func = makeFuncExpr(cast->castfunc,
							target_type,
							list_make1(expr),
							InvalidOid,	/* always right? */
							exprCollation((Node *) expr),
							COERCE_EXPLICIT_CAST);
		expr = (Expr *) func;
	}
	else if (cast->castmethod == COERCION_METHOD_BINARY)
	{
		RelabelType	   *relabel = makeNode(RelabelType);

		relabel->arg = expr;
		relabel->resulttype = target_type;
		relabel->resulttypmod = exprTypmod((Node *) expr);
		relabel->resultcollid = exprCollation((Node *) expr);
		relabel->relabelformat = COERCE_EXPLICIT_CAST;
		relabel->location = -1;

		expr = (Expr *) relabel;
	}
	else
	{
		elog(ERROR, "cast-method '%c' is not supported in opencl kernel",
			 cast->castmethod);
	}
	ReleaseSysCache(tup);

	return expr;
}

/*
 * make_expr_conditional - constructor of CASE ... WHEN ... END expression
 * which returns the supplied expression if condition is valid.
 */
static Expr *
make_expr_conditional(Expr *expr, Expr *filter, bool zero_if_unmatched)
{
	Oid			expr_typeoid = exprType((Node *)expr);
	int32		expr_typemod = exprTypmod((Node *)expr);
	Oid			expr_collid = exprCollation((Node *)expr);
	CaseWhen   *case_when;
	CaseExpr   *case_expr;
	Expr	   *defresult;

	if (!filter)
		return expr;

	if (!zero_if_unmatched)
		defresult = (Expr *) makeNullConst(expr_typeoid,
										   expr_typemod,
										   expr_collid);
	else
	{
		int16	typlen;
		bool	typbyval;

		get_typlenbyval(expr_typeoid, &typlen, &typbyval);
		defresult = (Expr *) makeConst(expr_typeoid,
									   expr_typemod,
									   expr_collid,
									   (int) typlen,
									   (Datum) 0,
									   false,
									   typbyval);
	}

	/* in case when the 'filter' is matched */
	case_when = makeNode(CaseWhen);
	case_when->expr = filter;
	case_when->result = expr;
	case_when->location = -1;

	/* case body */
	case_expr = makeNode(CaseExpr);
	case_expr->casetype = exprType((Node *) expr);
	case_expr->arg = NULL;
	case_expr->args = list_make1(case_when);
	case_expr->defresult = defresult;
	case_expr->location = -1;

	return (Expr *) case_expr;
}

/*
 * make_altfunc_simple_expr - constructor of simple function call
 */
static FuncExpr *
make_altfunc_simple_expr(const char *func_name, Expr *func_arg)
{
	Oid			namespace_oid = get_namespace_oid("pgstrom", false);
	Oid			argtype_oid = InvalidOid;
	oidvector  *func_argtypes;
	HeapTuple	tuple;
	Form_pg_proc proc_form;
	FuncExpr   *func_expr;

	if (func_arg)
	{
		argtype_oid = exprType((Node *)func_arg);
		func_argtypes = buildoidvector(&argtype_oid, 1);
	}
	else
		func_argtypes = buildoidvector(NULL, 0);

	/* find an alternative partial function */
	tuple = SearchSysCache3(PROCNAMEARGSNSP,
							PointerGetDatum(func_name),
							PointerGetDatum(func_argtypes),
							ObjectIdGetDatum(namespace_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "alternative function not found: %s",
			 func_arg != NULL
			 ? funcname_signature_string(func_name, 1, NIL, &argtype_oid)
			 : funcname_signature_string(func_name, 0, NIL, NULL));

	proc_form = (Form_pg_proc) GETSTRUCT(tuple);
	func_expr = makeFuncExpr(PgProcTupleGetOid(tuple),
							 proc_form->prorettype,
							 func_arg ? list_make1(func_arg) : NIL,
							 InvalidOid,
							 InvalidOid,
							 COERCE_EXPLICIT_CALL);
	ReleaseSysCache(tuple);

	return func_expr;
}

/*
 * make_altfunc_nrows_expr - constructor of the partial number of rows
 */
static FuncExpr *
make_altfunc_nrows_expr(Aggref *aggref)
{
	List	   *nrows_args = NIL;
	ListCell   *lc;
	Expr	   *expr;

	foreach (lc, aggref->args)
	{
		TargetEntry *tle = lfirst(lc);
		NullTest	*ntest = makeNode(NullTest);

		Assert(IsA(tle, TargetEntry));
		ntest->arg = copyObject(tle->expr);
		ntest->nulltesttype = IS_NOT_NULL;
		ntest->argisrow = false;

		nrows_args = lappend(nrows_args, ntest);
	}

	if (aggref->aggfilter)
	{
		Assert(exprType((Node *)aggref->aggfilter) == BOOLOID);
		nrows_args = lappend(nrows_args, copyObject(aggref->aggfilter));
	}

	if (nrows_args == NIL)
		expr = NULL;
	else if (list_length(nrows_args) == 1)
		expr = linitial(nrows_args);
	else
		expr = make_andclause(nrows_args);

	return make_altfunc_simple_expr("nrows", expr);
}

/*
 * make_altfunc_minmax_expr
 */
static FuncExpr *
make_altfunc_minmax_expr(Aggref *aggref, const char *func_name,
						 Oid pminmax_typeoid)
{
	TargetEntry	   *tle;
	Expr		   *expr;

	Assert(list_length(aggref->args) == 1);
    tle = linitial(aggref->args);
    Assert(IsA(tle, TargetEntry));
	/* cast to pminmax_typeoid, if mismatch */
	expr = make_expr_typecast(tle->expr, pminmax_typeoid);
	/* make conditional if aggref has any filter */
	expr = make_expr_conditional(expr, aggref->aggfilter, false);

	return make_altfunc_simple_expr(func_name, expr);
}

/*
 * make_altfunc_psum - constructor of a SUM/SUM_X2 reference
 */
static FuncExpr *
make_altfunc_psum_expr(Aggref *aggref, const char *func_name, Oid psum_typeoid)
{
	TargetEntry	   *tle;
	Expr		   *expr;

	Assert(list_length(aggref->args) == 1);
	tle = linitial(aggref->args);
	Assert(IsA(tle, TargetEntry));

	/* cast to psum_typeoid, if mismatch */
	expr = make_expr_typecast(tle->expr, psum_typeoid);
	/* make conditional if aggref has any filter */
	expr = make_expr_conditional(expr, aggref->aggfilter, true);

	return make_altfunc_simple_expr(func_name, expr);
}

/*
 * make_altfunc_pcov_xy - constructor of a co-variance arguments
 */
static FuncExpr *
make_altfunc_pcov_xy(Aggref *aggref, const char *func_name)
{
	Oid				namespace_oid = get_namespace_oid("pgstrom", false);
	oidvector	   *func_argtypes;
	Oid				func_argtypes_oid[3];
	Oid				func_oid;
	TargetEntry	   *tle_x;
	TargetEntry	   *tle_y;
	Expr		   *filter_expr;

	Assert(list_length(aggref->args) == 2);
	tle_x = linitial(aggref->args);
	tle_y = lsecond(aggref->args);
	if (exprType((Node *)tle_x->expr) != FLOAT8OID ||
		exprType((Node *)tle_y->expr) != FLOAT8OID)
		elog(ERROR, "Bug? unexpected argument type for co-variance");

	/* lookup pcov_XXX functions */
	func_argtypes_oid[0] = BOOLOID;
	func_argtypes_oid[1] = FLOAT8OID;
	func_argtypes_oid[2] = FLOAT8OID;
	func_argtypes = buildoidvector(func_argtypes_oid, 3);
	func_oid = get_function_oid(func_name,
								func_argtypes,
								namespace_oid, false);
	/* filter if any */
	if (aggref->aggfilter)
		filter_expr = aggref->aggfilter;
	else
		filter_expr = (Expr *)makeBoolConst(true, false);

	return makeFuncExpr(func_oid,
						FLOAT8OID,
						list_make3(filter_expr,
								   tle_x->expr,
								   tle_y->expr),
						InvalidOid,
						InvalidOid,
						COERCE_EXPLICIT_CALL);
}

/*
 * make_altfunc_hll_hash - Hyper-Log-Log Hash function
 */
static FuncExpr *
make_altfunc_hll_hash(Aggref *aggref)
{
	Oid				namespace_oid = get_namespace_oid("pgstrom", false);
	TargetEntry	   *tle;
	Oid				type_oid;
	oidvector	   *func_argtypes;
	HeapTuple		tuple;
	Form_pg_proc	proc;
	FuncExpr	   *func = NULL;

	/*
	 * lookup suitable pgstrom.hll_hash() function, and checks
	 * whether it is actually device executable.
	 */
	Assert(list_length(aggref->args) == 1);
	tle = linitial(aggref->args);
	Assert(IsA(tle, TargetEntry));
	type_oid = exprType((Node *)tle->expr);
	func_argtypes = buildoidvector(&type_oid, 1);

	tuple = SearchSysCache3(PROCNAMEARGSNSP,
							PointerGetDatum("hll_hash"),
							PointerGetDatum(func_argtypes),
							ObjectIdGetDatum(namespace_oid));
	if (!HeapTupleIsValid(tuple))
	{
		elog(DEBUG2, "no such function: %s",
			 funcname_signature_string("hll_hash", 1, NIL, &type_oid));
		return NULL;
	}
	proc = (Form_pg_proc)GETSTRUCT(tuple);
	if (pgstrom_devfunc_lookup(PgProcTupleGetOid(tuple),
							   proc->prorettype,
							   list_make1(tle->expr),
							   InvalidOid))
	{
		func = makeFuncExpr(PgProcTupleGetOid(tuple),
							proc->prorettype,
							list_make1(tle->expr),
							InvalidOid,
							InvalidOid,
							COERCE_EXPLICIT_CALL);
	}
	else
	{
		elog(DEBUG2, "no such device function: %s",
			 funcname_signature_string("hll_hash", 1, NIL, &type_oid));
	}
	ReleaseSysCache(tuple);
	return func;
}

/*
 * __update_aggfunc_clause_cost
 */
static void
__update_aggfunc_clause_cost(PlannerInfo *root,
							 Aggref *aggref_alt,
							 Form_pg_aggregate agg_form,
							 const aggfunc_catalog_t *aggfn_cat,
							 AggClauseCosts *final_costs)
{
#if PG_VERSION_NUM < 140000
	get_agg_clause_costs(root,
						 (Node *)aggref_alt,
						 AGGSPLIT_SIMPLE,
						 final_costs);
#else
	/*
	 * MEMO: PG14 revised get_agg_clause_costs() to calculate
	 * the cost of aggregate functions that are tracked on
	 * the PlannerInfo; thus, not alternative once replaced
	 * by GpuPreAgg. So, we put simplified logic to calculate
	 * per-tuple cost of the alternative functions.
	 */
	if (OidIsValid(agg_form->aggtransfn))
		add_function_cost(root, agg_form->aggtransfn, NULL,
						  &final_costs->transCost);
	if (OidIsValid(agg_form->aggfinalfn))
		add_function_cost(root, agg_form->aggfinalfn, NULL,
						  &final_costs->finalCost);
	if (aggfn_cat->partfn_extra_sz)
		final_costs->transitionSpace += aggfn_cat->partfn_extra_sz();
#endif
}

/*
 * make_alternative_aggref
 *
 * It makes an alternative final aggregate function towards the supplied
 * Aggref, and append its arguments on the target_partial/target_device.
 */
static Node *
make_alternative_aggref(PlannerInfo *root,
						Aggref *aggref,
						PathTarget *target_partial,
						PathTarget *target_input,
						RelOptInfo *input_rel,
						AggClauseCosts *final_clause_costs)
{
	const aggfunc_catalog_t *aggfn_cat;
	Aggref	   *aggref_new;
	List	   *altfunc_args = NIL;
	Expr	   *expr_host;
	Oid			namespace_oid;
	Oid			func_oid;
	const char *func_name;
	oidvector  *func_argtypes;
	HeapTuple	tuple;
	int			i;
	Form_pg_proc proc_form;
	Form_pg_aggregate agg_form;

	if (aggref->aggorder != NIL || aggref->aggdistinct != NIL)
	{
		elog(DEBUG2, "Aggregate with ORDER BY/DISTINCT is not supported: %s",
			 nodeToString(aggref));
		return NULL;
	}
	if (AGGKIND_IS_ORDERED_SET(aggref->aggkind))
	{
		elog(DEBUG2, "ORDERED SET Aggregation is not supported: %s",
			 nodeToString(aggref));
		return NULL;
	}

	/*
	 * Lookup properties of aggregate function
	 */
	aggfn_cat = aggfunc_lookup_by_oid(aggref->aggfnoid);
	if (!aggfn_cat)
	{
		elog(DEBUG2, "Aggregate function is not device executable: %s",
			 format_procedure(aggref->aggfnoid));
		return NULL;
	}
	/* sanity checks */
	Assert(aggref->aggkind == AGGKIND_NORMAL &&
		   !aggref->aggvariadic &&
		   list_length(aggref->args) <= 2);

	/*
	 * construct arguments list of the partial aggregation
	 */
	for (i=0; i < aggfn_cat->partfn_nargs; i++)
	{
		cl_int		action = aggfn_cat->partfn_argexprs[i];
		cl_int		argtype = aggfn_cat->partfn_argtypes[i];
		FuncExpr   *pfunc;
		Node	   *temp;

		switch (action)
		{
			case ALTFUNC_EXPR_NROWS:    /* NROWS(X) */
				pfunc = make_altfunc_nrows_expr(aggref);
				break;
			case ALTFUNC_EXPR_PMIN:     /* PMIN(X) */
				pfunc = make_altfunc_minmax_expr(aggref, "pmin", argtype);
				break;
			case ALTFUNC_EXPR_PMAX:     /* PMAX(X) */
				pfunc = make_altfunc_minmax_expr(aggref, "pmax", argtype);
				break;
			case ALTFUNC_EXPR_PSUM:     /* PSUM(X) */
				pfunc = make_altfunc_psum_expr(aggref, "psum", argtype);
				break;
			case ALTFUNC_EXPR_PSUM_X2:  /* PSUM_X2(X) = PSUM(X^2) */
				pfunc = make_altfunc_psum_expr(aggref, "psum_x2", argtype);
				break;
			case ALTFUNC_EXPR_PCOV_X:   /* PCOV_X(X,Y) */
				pfunc = make_altfunc_pcov_xy(aggref, "pcov_x");
				break;
			case ALTFUNC_EXPR_PCOV_Y:   /* PCOV_Y(X,Y) */
				pfunc = make_altfunc_pcov_xy(aggref, "pcov_y");
				break;
			case ALTFUNC_EXPR_PCOV_X2:  /* PCOV_X2(X,Y) */
				pfunc = make_altfunc_pcov_xy(aggref, "pcov_x2");
				break;
			case ALTFUNC_EXPR_PCOV_Y2:  /* PCOV_Y2(X,Y) */
				pfunc = make_altfunc_pcov_xy(aggref, "pcov_y2");
				break;
			case ALTFUNC_EXPR_PCOV_XY:  /* PCOV_XY(X,Y) */
				pfunc = make_altfunc_pcov_xy(aggref, "pcov_xy");
				break;
			case ALTFUNC_EXPR_HLL_HASH:	/* HLL_HASH(X) */
				pfunc = make_altfunc_hll_hash(aggref);
				break;
			default:
				elog(ERROR, "unknown alternative function code: %d", action);
				break;
		}
		/* actually supported? */
		if (!pfunc)
			return NULL;
		
		/* device executable? */
		if (pfunc->args)
		{
			temp = replace_expression_by_outerref((Node *)pfunc->args,
												  target_input);
			/*
			 * MEMO: Expressions are replaced to Var-node that references
			 * one of target_input, and these Var-nodes have INDEX_VAR.
			 * So, it is obvious Var-nodes references adeauate input relation,
			 * thus no need to provide relids to pgstrom_device_expression().
			 */
			if (!pgstrom_device_expression(root, NULL, (Expr *)temp))
				return NULL;
		}
		/* append to the argument list */
		altfunc_args = lappend(altfunc_args, (Expr *)pfunc);
	}

	/*
	 * Lookup an alternative function that generates partial state
	 * of the final aggregate function, or varref if internal state
	 * of aggregation is as-is.
	 */
	if (strcmp(aggfn_cat->partfn_name, "varref") == 0)
	{
		Assert(list_length(altfunc_args) == 1);
		expr_host = linitial(altfunc_args);
	}
	else
	{
		Assert(list_length(altfunc_args) == aggfn_cat->partfn_nargs);
		if (strncmp(aggfn_cat->partfn_name, "c:", 2) == 0)
			namespace_oid = PG_CATALOG_NAMESPACE;
		else if (strncmp(aggfn_cat->partfn_name, "s:", 2) == 0)
			namespace_oid = get_namespace_oid("pgstrom", false);
		else
			elog(ERROR, "Bug? incorrect alternative function catalog");

		func_name = aggfn_cat->partfn_name + 2;
		func_argtypes = buildoidvector(aggfn_cat->partfn_argtypes,
									   aggfn_cat->partfn_nargs);
		tuple = SearchSysCache3(PROCNAMEARGSNSP,
								PointerGetDatum(func_name),
								PointerGetDatum(func_argtypes),
								ObjectIdGetDatum(namespace_oid));
		if (!HeapTupleIsValid(tuple))
			elog(ERROR, "cache lookup failed for function %s",
				 funcname_signature_string(func_name,
										   aggfn_cat->partfn_nargs,
										   NIL,
										   aggfn_cat->partfn_argtypes));
		proc_form = (Form_pg_proc) GETSTRUCT(tuple);
		expr_host = (Expr *)makeFuncExpr(PgProcTupleGetOid(tuple),
										 proc_form->prorettype,
										 altfunc_args,
										 InvalidOid,
										 InvalidOid,
										 COERCE_EXPLICIT_CALL);
		ReleaseSysCache(tuple);
	}
	/* add expression if unique */
	add_new_column_to_pathtarget(target_partial, expr_host);

	/* construction of the final Aggref */
	if (strncmp(aggfn_cat->finalfn_name, "c:", 2) == 0)
		namespace_oid = PG_CATALOG_NAMESPACE;
	else if (strncmp(aggfn_cat->finalfn_name, "s:", 2) == 0)
		namespace_oid = get_namespace_oid("pgstrom", false);
	else
		elog(ERROR, "Bug? incorrect alternative function catalog");

	func_name = aggfn_cat->finalfn_name + 2;
	func_argtypes = buildoidvector(&aggfn_cat->finalfn_argtype, 1);
	func_oid = get_function_oid(func_name,
								func_argtypes,
								namespace_oid, false);
	/* sanity check */
	Assert(aggref->aggtype == get_func_rettype(func_oid));

	tuple = SearchSysCache1(AGGFNOID, ObjectIdGetDatum(func_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for pg_aggregate %u", func_oid);
	agg_form = (Form_pg_aggregate) GETSTRUCT(tuple);

	aggref_new = makeNode(Aggref);
	aggref_new->aggfnoid		= func_oid;
	aggref_new->aggtype			= aggref->aggtype;
	aggref_new->aggcollid		= aggref->aggcollid;
	aggref_new->inputcollid		= aggref->inputcollid;
	aggref_new->aggtranstype	= agg_form->aggtranstype;
	aggref_new->aggargtypes		= list_make1_oid(exprType((Node *)expr_host));
	aggref_new->aggdirectargs	= NIL;
	aggref_new->args			= list_make1(makeTargetEntry(expr_host,
															 1,
															 NULL,
															 false));
	aggref_new->aggorder		= NIL;	/* see sanity check */
	aggref_new->aggdistinct		= NIL;	/* see sanity check */
	aggref_new->aggfilter		= NULL;	/* moved to GpuPreAgg */
	aggref_new->aggstar			= false;
	aggref_new->aggvariadic		= false;
	aggref_new->aggkind			= AGGKIND_NORMAL;	/* see sanity check */
	aggref_new->agglevelsup		= 0;
	aggref_new->aggsplit		= AGGSPLIT_SIMPLE;
	aggref_new->location		= aggref->location;

	/* update the final aggregation costs */
	__update_aggfunc_clause_cost(root, aggref_new, agg_form, aggfn_cat,
								 final_clause_costs);
	ReleaseSysCache(tuple);

	return (Node *)aggref_new;
}

typedef struct
{
	bool		device_executable;
	PlannerInfo *root;
	PathTarget *target_upper;
	PathTarget *target_partial;
	PathTarget *target_input;
	RelOptInfo *input_rel;
	Bitmapset  *__pfunc_bitmap__;
	List	   *groupby_keys;
	AggClauseCosts final_clause_costs;
} gpupreagg_build_path_target_context;

static Node *
replace_expression_by_altfunc(Node *node,
							  gpupreagg_build_path_target_context *con)
{
	PathTarget *target_input = con->target_input;
	ListCell   *lc;

	if (!node)
		return NULL;
	if (IsA(node, Aggref))
	{
		Node   *aggfn = make_alternative_aggref(con->root,
												(Aggref *)node,
												con->target_partial,
												con->target_input,
												con->input_rel,
												&con->final_clause_costs);
		if (!aggfn)
			con->device_executable = false;
		return aggfn;
	}

	foreach (lc, target_input->exprs)
	{
		Expr   *expr_in = lfirst(lc);

		if (equal(node, expr_in))
		{
			add_new_column_to_pathtarget(con->target_partial,
										 copyObject(expr_in));
			return copyObject(node);
		}
	}
	if (IsA(node, Var) || IsA(node, PlaceHolderVar))
		elog(ERROR, "Bug? referenced variable is neither grouping-key nor its dependent key: %s",
			 nodeToString(node));
	return expression_tree_mutator(node, replace_expression_by_altfunc, con);
}

/*
 * gpupreagg_build_path_target
 *
 *
 *
 */
static bool
gpupreagg_build_path_target(PlannerInfo *root,			/* in */
							PathTarget *target_upper,	/* in */
							PathTarget *target_final,	/* out */
							PathTarget *target_partial,	/* out */
							Path       *input_path,     /* in */
							Node **p_havingQual,		/* out */
							bool *p_can_pullup_outerscan, /* out */
							AggClauseCosts *p_final_clause_costs) /* out */
{
	gpupreagg_build_path_target_context con;
	Query	   *parse = root->parse;
	PathTarget *target_input = input_path->pathtarget;
	RelOptInfo *input_rel = input_path->parent;
	Node	   *havingQual = NULL;
	ListCell   *lc;
	cl_int		i;

	memset(&con, 0, sizeof(con));
	con.device_executable = true;
	con.root			= root;
	con.target_upper	= target_upper;
	con.target_partial	= target_partial;
	con.target_input	= target_input;
	con.input_rel       = input_rel;
	con.groupby_keys    = NIL;

	/*
	 * NOTE: Not to inject unnecessary projection on the sub-path node,
	 * target_device shall be initialized according to the target_input
	 * once, but its sortgrouprefs are not set.
	 */
	i = 0;
	foreach (lc, target_upper->exprs)
	{
		Expr   *expr = lfirst(lc);
		Index	sortgroupref = get_pathtarget_sortgroupref(target_upper, i);
	
		if (sortgroupref && parse->groupClause &&
			get_sortgroupref_clause_noerr(sortgroupref,
										  parse->groupClause) != NULL)
		{
			devtype_info   *dtype;
			Oid				coll_oid;

			/*
			 * Type of the grouping-key must have device equality-function
			 */
			dtype = pgstrom_devtype_lookup(exprType((Node *)expr));
			if (!dtype || !dtype->hash_func)
			{
				elog(DEBUG2, "GROUP BY contains unsupported type (%s): %s",
					 format_type_be(exprType((Node *)expr)),
					 nodeToString((Node *)expr));
				return false;
			}
			coll_oid = exprCollation((Node *)expr);
			if (!pgstrom_devfunc_lookup_type_equal(dtype, coll_oid))
			{
				elog(DEBUG2, "GROUP BY contains unsupported type (%s): %s",
					 format_type_be(exprType((Node *)expr)),
					 nodeToString((Node *)expr));
				return false;
			}

			/*
			 * If expression cannot execute on device, unable to pull up
			 * outer scan node. This expression must be calculated on the
			 * host-side.
			 */
			if (!pgstrom_device_expression(root, input_rel, expr))
				*p_can_pullup_outerscan = false;
			/* add grouping-keys */
			add_column_to_pathtarget(target_partial, expr, sortgroupref);
			add_column_to_pathtarget(target_final, expr, sortgroupref);
		}
		else
		{
			Expr   *temp;

			temp = (Expr *)replace_expression_by_altfunc((Node *)expr, &con);
			if (!con.device_executable)
			{
				elog(DEBUG2, "alt-aggregation is not device executable: %s",
					 nodeToString((Node *)expr));
				return false;
			}
			if (exprType((Node *)expr) != exprType((Node *)temp))
				elog(ERROR, "Bug? GpuPreAgg catalog is not consistent: %s",
					 nodeToString(expr));
			add_column_to_pathtarget(target_final, temp, 0);
		}
		i++;
	}

	/*
	 * If there's a HAVING clause, we'll need the Vars/Aggrefs it uses, too.
	 */
	if (parse->havingQual)
	{
		havingQual = replace_expression_by_altfunc(parse->havingQual, &con);
		if (!con.device_executable)
		{
			elog(DEBUG2, "HAVING clause is not device executable: %s",
				 nodeToString((Node *)parse->havingQual));
			return false;
		}
	}
	*p_havingQual = havingQual;
	memcpy(p_final_clause_costs, &con.final_clause_costs,
		   sizeof(AggClauseCosts));

	set_pathtarget_cost_width(root, target_final);
	set_pathtarget_cost_width(root, target_partial);

	return true;
}

/*
 * build_custom_scan_tlist
 */
static List *
build_custom_scan_tlist(PlannerInfo *root,
						PathTarget *target_device,
						Index outer_scanrelid,
						List *outer_tlist)
{
	List	   *results = NIL;
	ListCell   *lc;
	int			i, j;

	/*
	 * TLE list for the grouping-key and partial aggregation; that shall
	 * be returned from the GPU kernel.
	 * (a.k.a template of the kds_final)
	 */
	i = 0;
	foreach (lc, target_device->exprs)
	{
		Node	   *node = lfirst(lc);
		TargetEntry *tle;

		tle = makeTargetEntry((Expr *)node,
							  i + 1,
							  NULL,
							  false);
		if (target_device->sortgrouprefs &&
			target_device->sortgrouprefs[i])
			tle->ressortgroupref = target_device->sortgrouprefs[i];

		results = lappend(results, tle);
		i++;
	}

	/*
	 * Junk TLE entries for setrefs.c and EXPLAIN output
	 */
	if (outer_scanrelid != 0)
	{
		RangeTblEntry  *rte = root->simple_rte_array[outer_scanrelid];
		Relation		rel;
		TupleDesc		tupdesc;

		Assert(rte->rtekind == RTE_RELATION);
		rel = table_open(rte->relid, NoLock);
		tupdesc = RelationGetDescr(rel);
		for (j=0; j < tupdesc->natts; j++)
		{
			Form_pg_attribute attr = tupleDescAttr(tupdesc, j);
			Var	   *varnode;
			char   *resname;

			if (attr->attisdropped)
				continue;
			/* add junk entry if not yet */
			foreach(lc, results)
			{
				TargetEntry *tle = lfirst(lc);

				if (!IsA(tle->expr, Var))
					continue;
				varnode = (Var *)tle->expr;
				if (varnode->varno == outer_scanrelid &&
					varnode->varattno == attr->attnum)
				{
					Assert(varnode->vartype == attr->atttypid &&
						   varnode->vartypmod == attr->atttypmod &&
						   varnode->varcollid == attr->attcollation);
					break;
				}
			}

			if (!lc)
			{
				varnode = makeVar(outer_scanrelid,
								  attr->attnum,
								  attr->atttypid,
								  attr->atttypmod,
								  attr->attcollation, 0);
				resname = pstrdup(NameStr(attr->attname));
				results = lappend(results,
								  makeTargetEntry((Expr *)varnode,
												  list_length(results) + 1,
												  resname,
												  true));
			}
		}
		table_close(rel, NoLock);
	}
	else
	{
		foreach (lc, outer_tlist)
		{
			TargetEntry	   *tle = lfirst(lc);
			ListCell	   *cell;

			foreach (cell, results)
			{
				TargetEntry	   *__tle = lfirst(cell);

				if (equal(tle->expr, __tle->expr))
					break;
			}
			if (!cell)
			{
				char	   *resname = NULL;

				if (tle->resname)
					resname = pstrdup(tle->resname);

				results = lappend(results,
								  makeTargetEntry(copyObject(tle->expr),
												  list_length(results) + 1,
												  resname,
												  true));
			}
		}
	}
	return results;
}

/*
 * PlanGpuPreAggPath
 *
 * Entrypoint to create CustomScan node
 */
static Plan *
PlanGpuPreAggPath(PlannerInfo *root,
				  RelOptInfo *rel,
				  CustomPath *best_path,
				  List *tlist,
				  List *clauses,
				  List *custom_plans)
{
	CustomScan	   *cscan = makeNode(CustomScan);
	GpuPreAggInfo  *gpa_info;
	Index			outer_scanrelid = 0;
	List		   *outer_refs = NIL;
	Plan		   *outer_plan = NULL;
	List		   *outer_tlist = NIL;

	Assert(list_length(best_path->custom_private) == 1);
	gpa_info = linitial(best_path->custom_private);

	Assert(list_length(custom_plans) <= 1);
	if (custom_plans == NIL)
		outer_scanrelid = gpa_info->outer_scanrelid;
	else
	{
		outer_plan = linitial(custom_plans);
		outer_tlist = outer_plan->targetlist;
	}

	/* pick up referenced columns (for columnar-optimization) */
	if (outer_scanrelid)
	{
		RelOptInfo *baserel = root->simple_rel_array[outer_scanrelid];
		Bitmapset  *referenced = NULL;
		int			i, j, k;

		for (i=baserel->min_attr, j=0; i <= baserel->max_attr; i++, j++)
		{
			if (i < 0 || baserel->attr_needed[j] == NULL)
				continue;
			k = i - FirstLowInvalidHeapAttributeNumber;
			referenced = bms_add_member(referenced, k);
		}
		if (gpa_info->outer_quals)
			pull_varattnos((Node *)gpa_info->outer_quals,
						   outer_scanrelid, &referenced);
		
		for (k = bms_next_member(referenced, -1);
			 k >= 0;
			 k = bms_next_member(referenced, k))
		{
			i = k + FirstLowInvalidHeapAttributeNumber;
			outer_refs = lappend_int(outer_refs, i);
		}
	}
	gpa_info->outer_refs = outer_refs;

	/* setup CustomScan node */
	cscan->scan.plan.targetlist = tlist;
	cscan->scan.plan.qual = NIL;
	outerPlan(cscan) = outer_plan;
	cscan->scan.scanrelid = gpa_info->outer_scanrelid;
	cscan->flags = best_path->flags;
	cscan->methods = &gpupreagg_scan_methods;
	cscan->custom_scan_tlist = build_custom_scan_tlist(root,
													   best_path->path.pathtarget,
													   outer_scanrelid,
													   outer_tlist);
	/*
	 * construction of the GPU kernel code
	 */
	gpupreagg_codegen(root, best_path->path.parent, cscan, outer_tlist, gpa_info);

	form_gpupreagg_info(cscan, gpa_info);

	return &cscan->scan.plan;
}

/*
 * pgstrom_path_is_gpupreagg
 */
bool
pgstrom_path_is_gpupreagg(const Path *pathnode)
{
	if (IsA(pathnode, CustomPath) &&
		pathnode->pathtype == T_CustomScan &&
		((CustomPath *) pathnode)->methods == &gpupreagg_path_methods)
		return true;
	return false;
}

/*
 * pgstrom_plan_is_gpupreagg
 */
bool
pgstrom_plan_is_gpupreagg(const Plan *plan)
{
	if (IsA(plan, CustomScan) &&
		((CustomScan *) plan)->methods == &gpupreagg_scan_methods)
		return true;
	return false;
}

/*
 * pgstrom_planstate_is_gpupreagg
 */
bool
pgstrom_planstate_is_gpupreagg(const PlanState *ps)
{
	if (IsA(ps, CustomScanState) &&
		((CustomScanState *) ps)->methods == &gpupreagg_exec_methods)
		return true;
	return false;
}

/*
 * pgstrom_copy_gpupreagg_path
 */
Path *
pgstrom_copy_gpupreagg_path(const Path *pathnode)
{
	Assert(pgstrom_path_is_gpupreagg(pathnode));
	return pmemdup(pathnode, sizeof(CustomPath));
}

/*
 * make_tlist_device_projection
 *
 * It pulls a set of referenced resource numbers according to the supplied
 * outer_scanrelid/outer_tlist.
 */
typedef struct
{
	Bitmapset  *outer_refs_any;
	Bitmapset  *outer_refs_expr;
	bool		in_expression;
	Index		outer_scanrelid;
	List	   *outer_tlist;
} make_tlist_device_projection_context;

static Node *
__make_tlist_device_projection(Node *node, void *__con)
{
	make_tlist_device_projection_context *con = __con;
	bool		in_expression_saved = con->in_expression;
	int			k;
	Node	   *newnode;

	if (!node)
		return NULL;
	if (con->outer_scanrelid > 0)
	{
		Assert(con->outer_tlist == NIL);
		if (IsA(node, Var))
		{
			Var	   *varnode = (Var *) node;

			if (varnode->varno != con->outer_scanrelid)
				elog(ERROR, "Bug? varnode references unknown relid: %s",
					 nodeToString(varnode));
			k = varnode->varattno - FirstLowInvalidHeapAttributeNumber;
			con->outer_refs_any = bms_add_member(con->outer_refs_any, k);
			if (con->in_expression)
				con->outer_refs_expr = bms_add_member(con->outer_refs_expr, k);
			Assert(varnode->varlevelsup == 0);
			return (Node *)copyObject(varnode);
		}
	}
	else
	{
		ListCell	   *lc;

		foreach (lc, con->outer_tlist)
		{
			TargetEntry    *tle = lfirst(lc);

			if (equal(node, tle->expr))
			{
				k = tle->resno - FirstLowInvalidHeapAttributeNumber;
				con->outer_refs_any = bms_add_member(con->outer_refs_any, k);
				if (con->in_expression)
					con->outer_refs_expr = bms_add_member(con->outer_refs_expr, k);
				return (Node *)makeVar(OUTER_VAR,
									   tle->resno,
									   exprType((Node *)tle->expr),
									   exprTypmod((Node *)tle->expr),
									   exprCollation((Node *)tle->expr),
									   0);
			}
		}
		if (IsA(node, Var))
			elog(ERROR, "Bug? varnode (%s) references unknown outer entry: %s",
				 nodeToString(node),
				 nodeToString(con->outer_tlist));
	}
	con->in_expression = true;
	newnode = expression_tree_mutator(node, __make_tlist_device_projection, con);
	con->in_expression = in_expression_saved;

	return newnode;
}

static List *
make_tlist_device_projection(List *custom_scan_tlist,
							 Index outer_scanrelid,
							 List *outer_tlist,
							 Bitmapset **p_outer_refs_any,
							 Bitmapset **p_outer_refs_expr)
{
	make_tlist_device_projection_context con;
	List	   *tlist_part = NIL;
	ListCell   *lc;

	memset(&con, 0, sizeof(con));
	con.outer_scanrelid = outer_scanrelid;
	con.outer_tlist = outer_tlist;

	foreach (lc, custom_scan_tlist)
	{
		TargetEntry *tle = lfirst(lc);
		TargetEntry *tmp;
		Node	   *node;

		if (tle->resjunk)
			continue;
		con.in_expression = false;
		node = __make_tlist_device_projection((Node *)tle->expr, &con);

		tmp = flatCopyTargetEntry(tle);
		tmp->expr  = (Expr *)node;
		tmp->resno = list_length(tlist_part) + 1;
		tlist_part = lappend(tlist_part, tmp);
	}
	*p_outer_refs_any = con.outer_refs_any;
	*p_outer_refs_expr = con.outer_refs_expr;

	return tlist_part;
}

static Node *
__revert_tlist_device_projection(Node *node, void *datum)
{
	List	   *outer_tlist = (List *)datum;

	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		Var	   *varnode = (Var *)node;
		TargetEntry *tle;

		Assert(varnode->varno == OUTER_VAR &&
			   varnode->varattno > 0 &&
			   varnode->varattno <= list_length(outer_tlist));
		tle = list_nth(outer_tlist, varnode->varattno - 1);
		Assert(IsA(tle, TargetEntry));
		return (Node *)copyObject(tle->expr);
	}
	return expression_tree_mutator(node, __revert_tlist_device_projection, datum);
}

static List *
revert_tlist_device_projection(List *tlist_dev,
							   List *outer_tlist)
{
	if (outer_tlist == NIL)
		return copyObject(tlist_dev);
	return (List *)__revert_tlist_device_projection((Node *)tlist_dev, outer_tlist);
}

/*
 * gpupreagg_codegen_projection - code generator for
 *
 * DEVICE_FUNCTION(void)
 * gpupreagg_projection_row(kern_context *kcxt,
 *                          kern_data_store *kds_src,
 *                          HeapTupleHeaderData *htup,
 *                          Datum *dst_values,
 *                          cl_char *dst_isnull);
 * and
 *
 * DEVICE_FUNCTION(void)
 * gpupreagg_projection_slot(kern_context *kcxt,
 *                           Datum *src_values,
 *                           cl_bool *src_isnull,
 *                           Datum *dst_values,
 *                           cl_bool *dst_values);
 * and
 * DEVICE_FUNCTION(void)
 * gpupreagg_projection_arrow(kern_context *kcxt,
 *                            kern_data_store *kds_src,
 *                            cl_uint src_index,
 *                            Datum *dst_values,
 *                            cl_char *dst_isnull);
 * and
 * DEVICE_FUNCTION(void)
 * gpupreagg_projection_column(kern_context *kcxt,
 *                             kern_data_store *kds,
 *                             kern_data_extra *extra,
 *                             cl_uint rowid,
 *                             cl_char *dst_dclass,
 *                             Datum   *dst_values);
 */
static Expr *
codegen_projection_partial_funcion(FuncExpr *f, codegen_context *context)
{
	HeapTuple		tuple;
	Form_pg_proc	proc_form;
	const char	   *proc_name;
	devtype_info   *dtype;
	devfunc_info   *dfunc;
	Expr		   *expr;

	Assert(IsA(f, FuncExpr));
	tuple = SearchSysCache1(PROCOID, ObjectIdGetDatum(f->funcid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for function %u", f->funcid);
	proc_form = (Form_pg_proc) GETSTRUCT(tuple);
	proc_name = NameStr(proc_form->proname);
	if (proc_form->pronamespace != get_namespace_oid("pgstrom", false))
		elog(ERROR, "Bug? unexpected partial aggregate function: %s",
			 format_procedure(f->funcid));

	if (strcmp(proc_name, "nrows") == 0)
	{
		Assert(list_length(f->args) <= 1);
		expr = (Expr *)makeConst(INT8OID,
								 -1,
								 InvalidOid,
								 sizeof(int64),
								 1,
								 false,
								 FLOAT8PASSBYVAL);
		if (f->args)
			expr = make_expr_conditional(expr, linitial(f->args), true);
	}
	else if (strcmp(proc_name, "pmin") == 0 ||
			 strcmp(proc_name, "pmax") == 0)
	{
		Assert(list_length(f->args) == 1);
		expr = linitial(f->args);
		dtype = pgstrom_devtype_lookup_and_track(exprType((Node *)expr),
												 context);
		if (!dtype)
			elog(ERROR, "device type lookup failed: %s",
				 format_type_be(exprType((Node *)expr)));
	}
	else if (strcmp(proc_name, "psum") == 0 ||
			 strcmp(proc_name, "psum_x2") == 0)
	{
		Assert(list_length(f->args) == 1);
		expr = linitial(f->args);
		dtype = pgstrom_devtype_lookup_and_track(exprType((Node *)expr),
												 context);
		if (!dtype)
			elog(ERROR, "device type lookup failed: %s",
				 format_type_be(exprType((Node *)expr)));
		if (strcmp(proc_name, "psum_x2") == 0)
		{
			Assert(dtype->type_oid == FLOAT8OID);
			expr = (Expr *)makeFuncExpr(F_FLOAT8MUL,
										FLOAT8OID,
										list_make2(copyObject(expr),
												   copyObject(expr)),
										InvalidOid,
										InvalidOid,
										COERCE_EXPLICIT_CALL);
		}
	}
	else if (strcmp(proc_name, "pcov_x")  == 0 ||
			 strcmp(proc_name, "pcov_y")  == 0 ||
			 strcmp(proc_name, "pcov_x2") == 0 ||
			 strcmp(proc_name, "pcov_y2") == 0 ||
			 strcmp(proc_name, "pcov_xy") == 0)
	{
		Expr   *filter;
		Expr   *x_value;
		Expr   *y_value;

		Assert(list_length(f->args) == 3);
		filter = linitial(f->args);
		x_value = lsecond(f->args);
		y_value = lthird(f->args);

		if (strcmp(proc_name, "pcov_x") == 0)
			expr = x_value;
		else if (strcmp(proc_name, "pcov_y") == 0)
			expr = y_value;
		else if (strcmp(proc_name, "pcov_x2") == 0)
			expr = (Expr *)makeFuncExpr(F_FLOAT8MUL,
										FLOAT8OID,
										list_make2(x_value,
												   x_value),
										InvalidOid,
										InvalidOid,
										COERCE_EXPLICIT_CALL);
		else if (strcmp(proc_name, "pcov_y2") == 0)
			expr = (Expr *)makeFuncExpr(F_FLOAT8MUL,
										FLOAT8OID,
										list_make2(y_value,
												   y_value),
										InvalidOid,
										InvalidOid,
										COERCE_EXPLICIT_CALL);
		else if (strcmp(proc_name, "pcov_xy") == 0)
			expr = (Expr *)makeFuncExpr(F_FLOAT8MUL,
										FLOAT8OID,
										list_make2(x_value,
												   y_value),
										InvalidOid,
										InvalidOid,
										COERCE_EXPLICIT_CALL);
		else
			elog(ERROR, "Bug? unexpected code path");

		Assert(exprType((Node *)filter) == BOOLOID);
		expr = make_expr_conditional(expr, filter, true);
	}
	else if (strcmp(proc_name, "hll_sketch_new") == 0)
	{
		FuncExpr   *hfunc;
		char	   *hfunc_name;
		Oid			hfunc_namespace;

		Assert(list_length(f->args) == 1);
		hfunc = linitial(f->args);
		Assert(IsA(hfunc, FuncExpr));
		hfunc_name = get_func_name(hfunc->funcid);
		hfunc_namespace = get_func_namespace(hfunc->funcid);
		if (strcmp(hfunc_name, "hll_hash") != 0 ||
			hfunc_namespace != get_namespace_oid("pgstrom", false))
			elog(ERROR, "Bug? hll_sketch_new() is invoked with %s",
				 format_procedure(hfunc->funcid));
		pfree(hfunc_name);

		dfunc = pgstrom_devfunc_lookup(hfunc->funcid,
									   hfunc->funcresulttype,
									   hfunc->args,
									   hfunc->inputcollid);
		if (!dfunc)
			elog(ERROR, "device function lookup failed: %s",
				 format_procedure(hfunc->funcid));
		pgstrom_devfunc_track(context, dfunc);

		expr = (Expr *)hfunc;
	}
	else
	{
		elog(ERROR, "Bug? unexpected partial aggregate function: %s",
			 format_procedure(f->funcid));
	}
	ReleaseSysCache(tuple);

	return expr;
}

static void
gpupreagg_codegen_projection(StringInfo kern,
							 codegen_context *context,
							 CustomScan *cscan,
							 List *outer_tlist,
							 List **p_tlist_part,
							 List **p_tlist_prep,
							 Bitmapset **p_pfunc_bitmap)
{
	PlannerInfo	   *root = context->root;
	Bitmapset	   *outer_refs_any = NULL;
	Bitmapset	   *outer_refs_expr = NULL;
	List		   *tlist_part = NIL;
	List		   *tlist_prep = NIL;
	Bitmapset	   *pfunc_bitmap = NULL;
	StringInfoData	decl;
	StringInfoData	tbody;		/* Row/Block */
	StringInfoData	sbody;		/* Slot */
	StringInfoData	abody;		/* Arrow */
	StringInfoData	cbody;		/* Column */
	StringInfoData	temp;
	Relation		outer_rel = NULL;
	TupleDesc		outer_desc = NULL;
	List		   *type_oid_list = NIL;
	ListCell	   *lc;
	int				i, k, nattrs;
	int				outer_refno_max = -1;

	/*
	 * Extract tlist for device projection (a.k.a template of kds_slot)
	 */
	tlist_part = make_tlist_device_projection(cscan->scan.plan.targetlist,
											  cscan->scan.scanrelid,
											  outer_tlist,
											  &outer_refs_any,
											  &outer_refs_expr);
	initStringInfo(&decl);
	initStringInfo(&tbody);
	initStringInfo(&sbody);
	initStringInfo(&abody);
	initStringInfo(&cbody);
	initStringInfo(&temp);

	appendStringInfoString(
		&decl,
		"  void        *addr    __attribute__((unused));\n");

	/* open relation if GpuPreAgg looks at physical relation */
	if (cscan->scan.scanrelid == 0)
	{
		nattrs = list_length(outer_tlist);
	}
	else
	{
		RangeTblEntry  *rte;

		Assert(outer_tlist == NIL);
		rte = root->simple_rte_array[cscan->scan.scanrelid];
		outer_rel = table_open(rte->relid, NoLock);
		outer_desc = RelationGetDescr(outer_rel);
		nattrs = outer_desc->natts;
	}

	/* extract the supplied tuple and load variables */
	if (bms_is_empty(outer_refs_any))
		goto setup_expressions;

	for (i=0; i > FirstLowInvalidHeapAttributeNumber; i--)
	{
		k = i - FirstLowInvalidHeapAttributeNumber;
		if (bms_is_member(k,
						  outer_refs_any))
			elog(ERROR, "Bug? system column or whole-row is referenced");
	}

	resetStringInfo(&temp);
	for (i=1; i <= nattrs; i++)
	{
		devtype_info   *dtype;
		bool	referenced = false;

		k = i - FirstLowInvalidHeapAttributeNumber;
		if (!bms_is_member(k, outer_refs_any))
			continue;

		/* data type of the outer relation input stream */
		if (cscan->scan.scanrelid == 0)
		{
			TargetEntry *tle = list_nth(outer_tlist, i-1);
			Oid		type_oid = exprType((Node *)tle->expr);

			dtype = pgstrom_devtype_lookup_and_track(type_oid, context);
			if (!dtype)
				elog(ERROR, "device type lookup failed: %s",
					 format_type_be(type_oid));
		}
		else
		{
			Form_pg_attribute attr = tupleDescAttr(outer_desc, i-1);

			dtype = pgstrom_devtype_lookup_and_track(attr->atttypid, context);
			if (!dtype)
				elog(ERROR, "device type lookup failed: %s",
					 format_type_be(attr->atttypid));
		}

		foreach (lc, tlist_part)
		{
			TargetEntry *tle = lfirst(lc);
			Var		   *varnode;

			Assert(!tle->resjunk);
			if (!IsA(tle->expr, Var))
				continue;
			varnode = (Var *) tle->expr;
			if ((varnode->varno != cscan->scan.scanrelid &&
				 varnode->varno != OUTER_VAR) ||
				(varnode->varattno < 1 ||
				 varnode->varattno > nattrs))
				elog(ERROR, "Bug? unexpected varnode: %s", nodeToString(varnode));
			if (varnode->varattno != i)
				continue;

			/* row */
			if (!referenced)
			{
				appendStringInfo(
					&temp,
					"  case %d:\n", i - 1);
				outer_refno_max = i;
			}

			appendStringInfo(
				&temp,
				"    pg_datum_ref(kcxt, temp.%s_v, addr);\n"
				"    pg_datum_store(kcxt, temp.%s_v,\n"
				"                   dst_dclass[%d],\n"
				"                   dst_values[%d]);\n",
				dtype->type_name,
				dtype->type_name,
				tle->resno - 1,
				tle->resno - 1);

			/* slot */
			appendStringInfo(
				&sbody,
				"  dst_dclass[%d] = src_dclass[%d];\n"
				"  dst_values[%d] = src_values[%d];\n",
				tle->resno-1, i-1,
				tle->resno-1, i-1);

			/* arrow */
			appendStringInfo(
				&abody,
				"  pg_datum_ref_arrow(kcxt,temp.%s_v,kds_src,%d,src_index);\n"
				"  pg_datum_store(kcxt, temp.%s_v,\n"
				"                 dst_dclass[%d],\n"
				"                 dst_values[%d]);\n",
				dtype->type_name, i-1,
				dtype->type_name,
				tle->resno-1,
				tle->resno-1);
			/* column */
			appendStringInfo(
				&cbody,
				"  addr = kern_get_datum_column(kds,extra,%d,rowid);\n"
				"  if (!addr)\n"
				"    dst_dclass[%d] = DATUM_CLASS__NULL;\n"
				"  else\n"
				"  {\n"
				"    dst_dclass[%d] = DATUM_CLASS__NORMAL;\n"
				"    dst_values[%d] = %s(addr);\n"
				"  }\n",
				i-1, tle->resno-1, tle->resno-1, tle->resno-1,
				dtype->type_byval
				? (dtype->type_length == 1 ? "READ_INT8_PTR"  :
				   dtype->type_length == 2 ? "READ_INT16_PTR" :
				   dtype->type_length == 4 ? "READ_INT32_PTR" :
				   dtype->type_length == 8 ? "READ_INT64_PTR" : "NO_SUCH_TYPLEN")
				: "PointerGetDatum");
			referenced = true;
		}

		/*
		 * KVAR_x must be set up if variables are referenced by
		 * expressions.
		 */
		if (bms_is_member(k, outer_refs_expr))
		{
			appendStringInfo(
				&decl,
				"  pg_%s_t KVAR_%u;\n",
				dtype->type_name, i);
			/* row */
			if (!referenced)
			{
				appendStringInfo(
					&temp,
					"  case %d:\n", i - 1);
				outer_refno_max = i;
			}
			appendStringInfo(
				&temp,
				"    pg_datum_ref(kcxt, KVAR_%u, addr);\n", i);
			/* slot */
			appendStringInfo(
				&sbody,
				"  pg_datum_ref_slot(kcxt,KVAR_%u,\n"
				"                    src_dclass[%d],\n"
				"                    src_values[%d]);\n",
				i, i-1, i-1);
			/* arrow */
			if (referenced)
			{
				appendStringInfo(
					&abody,
					"  KVAR_%u = temp.%s_v;\n",
					i, dtype->type_name);
			}
			else
			{
				appendStringInfo(
					&abody,
					"  pg_datum_ref_arrow(kcxt,KVAR_%u,kds_src,%d,src_index);\n",
					i, i-1);
			}
			/* column */
			if (!referenced)
				appendStringInfo(
					&cbody,
					"  addr = kern_get_datum_column(kds,extra,%d,rowid);\n",
					i-1);
			appendStringInfo(
				&cbody,
				"  pg_datum_ref(kcxt, KVAR_%u, addr);\n", i);
			referenced = true;
		}
		context->extra_bufsz += MAXALIGN(dtype->extra_sz);
		type_oid_list = list_append_unique_oid(type_oid_list, dtype->type_oid);
		if (referenced)
			appendStringInfoString(
				&temp,
				"    break;\n");
	}

	if (temp.len > 0)
	{
		appendStringInfo(
			&tbody,
			"  EXTRACT_HEAP_TUPLE_BEGIN(kds_src,htup,%d);\n"
			"  switch (__colidx)\n"
			"  {\n"
			"%s"
			"  default:\n"
			"    break;\n"
			"  }\n"
			"  EXTRACT_HEAP_TUPLE_END();\n",
			outer_refno_max,
			temp.data);
	}

	/*
	 * Execute expression and store the value on dst_values/dst_isnull
	 */
setup_expressions:
	resetStringInfo(&temp);
	foreach (lc, tlist_part)
	{
		TargetEntry	   *tle = lfirst(lc);
		TargetEntry	   *tmp;
		Expr		   *expr;
		Oid				type_oid;
		devtype_info   *dtype;
		const char	   *label;

		Assert(!tle->resjunk);
		if (IsA(tle->expr, Var))
		{
			/* should be already loaded */
			expr = tle->expr;
			label = "grouping-key";
		}
		else
		{
			if (is_altfunc_expression((Node *)tle->expr, NULL))
			{
				FuncExpr   *f = (FuncExpr *) tle->expr;

				expr = codegen_projection_partial_funcion(f, context);
				pfunc_bitmap = bms_add_member(pfunc_bitmap,
											  list_length(tlist_prep));
				label = "aggfunc-arg";
			}
			else if (tle->ressortgroupref)
			{
				expr = tle->expr;
				label = "grouping-key";
			}
			else
				elog(ERROR, "Bug? unexpected expression: %s",
					 nodeToString(tle->expr));

			type_oid = exprType((Node *)expr);
			dtype = pgstrom_devtype_lookup_and_track(type_oid, context);
			if (!dtype)
				elog(ERROR, "device type lookup failed: %s",
					 format_type_be(type_oid));
			appendStringInfo(
				&temp,
				"\n"
				"  /* initial attribute %d (%s) */\n"
				"  temp.%s_v = %s;\n"
				"  if (temp.%s_v.isnull)\n"
				"    dst_dclass[%d] = DATUM_CLASS__NULL;\n"
				"  else\n"
				"    pg_datum_store(kcxt, temp.%s_v,\n"
				"                   dst_dclass[%d],\n"
				"                   dst_values[%d]);\n",
				tle->resno, label,
				dtype->type_name,
				pgstrom_codegen_expression((Node *)expr, context),
				dtype->type_name,
				tle->resno-1,
				dtype->type_name,
				tle->resno-1,
				tle->resno-1);
			context->extra_bufsz += MAXALIGN(dtype->extra_sz);
			type_oid_list = list_append_unique_oid(type_oid_list, type_oid);
		}
		tmp = flatCopyTargetEntry(tle);
		tmp->expr  = expr;
		tlist_prep = lappend(tlist_prep, tmp);
	}
	Assert(list_length(tlist_prep) == list_length(tlist_part));
	appendStringInfoString(&tbody, temp.data);
	appendStringInfoString(&sbody, temp.data);
	appendStringInfoString(&abody, temp.data);
	appendStringInfoString(&cbody, temp.data);

	*p_tlist_part = revert_tlist_device_projection(tlist_part, outer_tlist);
	*p_tlist_prep = revert_tlist_device_projection(tlist_prep, outer_tlist);
	*p_pfunc_bitmap = pfunc_bitmap;

	/* const/params and temporary variable */
	pgstrom_union_type_declarations(&decl, "temp", type_oid_list);

	/* writeout kernel functions */
	appendStringInfo(
		kern,
		"DEVICE_FUNCTION(void)\n"
		"gpupreagg_projection_row(kern_context *kcxt,\n"
		"                         kern_data_store *kds_src,\n"
		"                         HeapTupleHeaderData *htup,\n"
		"                         cl_char *dst_dclass,\n"
		"                         Datum   *dst_values)\n"
		"{\n"
		"%s\n"
		"%s"
		"}\n\n"
		"#ifdef GPUPREAGG_COMBINED_JOIN\n"
		"DEVICE_FUNCTION(void)\n"
		"gpupreagg_projection_slot(kern_context *kcxt,\n"
		"                          cl_char *src_dclass,\n"
		"                          Datum   *src_values,\n"
		"                          cl_char *dst_dclass,\n"
		"                          Datum   *dst_values)\n"
		"{\n"
		"%s\n"
		"%s"
		"}\n"
		"#endif /* GPUPREAGG_COMBINED_JOIN */\n\n"
		"DEVICE_FUNCTION(void)\n"
		"gpupreagg_projection_arrow(kern_context *kcxt,\n"
		"                           kern_data_store *kds_src,\n"
		"                           cl_uint src_index,\n"
		"                           cl_char *dst_dclass,\n"
		"                           Datum   *dst_values)\n"
		"{\n"
		"%s\n"
		"%s"
		"}\n\n"
		"DEVICE_FUNCTION(void)\n"
		"gpupreagg_projection_column(kern_context *kcxt,\n"
		"                            kern_data_store *kds,\n"
		"                            kern_data_extra *extra,\n"
		"                            cl_uint rowid,\n"
		"                            cl_char *dst_dclass,\n"
		"                            Datum   *dst_values)\n"
		"{\n"
		"%s\n"
		"%s"
		"}\n\n",
		decl.data, tbody.data,
		decl.data, sbody.data,
		decl.data, abody.data,
		decl.data, cbody.data);

	if (outer_rel)
		table_close(outer_rel, NoLock);

	pfree(decl.data);
	pfree(tbody.data);
	pfree(sbody.data);
	pfree(abody.data);
	pfree(temp.data);
}

/*
 * gpupreagg_codegen_hashvalue - code generator for
 *
 * DEVICE_FUNCTION(cl_uint)
 * gpupreagg_hashvalue(kern_context *kcxt,
 *                     cl_uint *crc32_table,
 *                     cl_uint hash_value,
 *                     cl_bool *slot_isnull,
 *                     Datum *slot_values);
 */
static void
gpupreagg_codegen_hashvalue(StringInfo kern,
							codegen_context *context,
							List *tlist_prep)
{
	StringInfoData	decl;
	StringInfoData	body;
	List		   *type_oid_list = NIL;
	ListCell	   *lc;
	bool			is_first_key = true;

	initStringInfo(&decl);
	initStringInfo(&body);

	appendStringInfoString(
		&decl,
		"  cl_uint      hash = 0xffffffffU;\n");

	foreach (lc, tlist_prep)
	{
		TargetEntry	   *tle = lfirst(lc);
		Oid				type_oid;
		devtype_info   *dtype;

		if (tle->resjunk || !tle->ressortgroupref)
			continue;

		type_oid = exprType((Node *)tle->expr);
		dtype = pgstrom_devtype_lookup_and_track(type_oid, context);
		if (!dtype || !OidIsValid(dtype->type_eqfunc))
			elog(ERROR, "Bug? type (%s) is not supported",
				 format_type_be(type_oid));
		/* load variable */
		appendStringInfo(
			&body,
			"  pg_datum_ref_slot(kcxt,temp.%s_v,\n"
			"                    slot_dclass[%d],\n"
			"                    slot_values[%d]);\n",
			dtype->type_name,
			tle->resno - 1,
			tle->resno - 1);
		/*
		 * update hash value
		 *
		 * NOTE: In case when GROUP BY has multiple keys and identical values
		 * may appear on some of them, we should not merge hash values without
		 * randomization, because (X xor Y) xor Y == X.
		 * We put a simple randomization using magic number (0x9e370001) here,
		 * so it enables to generate different hash value, even if key1 and
		 * key2 has same value.
		 */
		if (!is_first_key)
		{
			appendStringInfo(
				&body,
				"  hash = ((cl_ulong)hash * 0x9e370001UL) >> 32;\n");
		}
		appendStringInfo(
			&body,
			"  hash ^= pg_comp_hash(kcxt, temp.%s_v);\n",
			dtype->type_name);
		type_oid_list = list_append_unique_oid(type_oid_list,
											   dtype->type_oid);
		is_first_key = false;
	}
	pgstrom_union_type_declarations(&decl, "temp", type_oid_list);

	/* no constants should appear */
	appendStringInfo(
		kern,
		"DEVICE_FUNCTION(cl_uint)\n"
		"gpupreagg_hashvalue(kern_context *kcxt,\n"
		"                    cl_char *slot_dclass,\n"
		"                    Datum   *slot_values)\n"
		"{\n"
		"%s\n%s"
		"  return hash;\n"
		"}\n\n", decl.data, body.data);

	pfree(decl.data);
	pfree(body.data);
}

/*
 * gpupreagg_codegen_keymatch - code generator for
 *
 *
 * DEVICE_FUNCTION(cl_bool)
 * gpupreagg_keymatch(kern_context *kcxt,
 *                    kern_data_store *x_kds, size_t x_index,
 *                    kern_data_store *y_kds, size_t y_index);
 */
static void
gpupreagg_codegen_keymatch(StringInfo kern,
						   codegen_context *context,
						   List *tlist_prep)
{
	StringInfoData	decl;
	StringInfoData	body;
	List		   *type_oid_list = NIL;
	ListCell	   *lc;

	initStringInfo(&decl);
	initStringInfo(&body);

	appendStringInfoString(
		&decl,
		"  cl_char     *x_dclass = KERN_DATA_STORE_DCLASS(x_kds, x_index);\n"
		"  cl_char     *y_dclass = KERN_DATA_STORE_DCLASS(y_kds, y_index);\n"
		"  Datum       *x_values = KERN_DATA_STORE_VALUES(x_kds, x_index);\n"
		"  Datum       *y_values = KERN_DATA_STORE_VALUES(y_kds, y_index);\n");

	foreach (lc, tlist_prep)
	{
		TargetEntry	   *tle = lfirst(lc);
		Oid				type_oid;
		Oid				coll_oid;
		devtype_info   *dtype;
		devfunc_info   *dfunc;
		devtype_info   *darg1;
		devtype_info   *darg2;
		char		   *cast_darg1 = NULL;
		char		   *cast_darg2 = NULL;

		if (tle->resjunk || !tle->ressortgroupref)
			continue;

		/* find the function to compare this data-type */
		type_oid = exprType((Node *)tle->expr);
		coll_oid = exprCollation((Node *)tle->expr);
		dtype = pgstrom_devtype_lookup_and_track(type_oid, context);
		if (!dtype)
			elog(ERROR, "Bug? type (%s) is not supported at GPU",
				 format_type_be(type_oid));
		dfunc = pgstrom_devfunc_lookup_type_equal(dtype, coll_oid);
		if (!dfunc)
			elog(ERROR, "Bug? type (%s) has no device equality function",
				 format_type_be(type_oid));
		pgstrom_devfunc_track(context, dfunc);
		darg1 = linitial(dfunc->func_args);
		darg2 = lsecond(dfunc->func_args);
		if (dtype->type_oid != darg1->type_oid)
		{
			if (!pgstrom_devtype_can_relabel(dtype->type_oid,
											 darg1->type_oid))
				elog(ERROR, "Bug? no binary compatible cast for %s -> %s",
					 format_type_be(dtype->type_oid),
					 format_type_be(darg1->type_oid));
			cast_darg1 = psprintf("to_%s", darg1->type_name);
		}
		if (dtype->type_oid != darg2->type_oid)
		{
			if (!pgstrom_devtype_can_relabel(dtype->type_oid,
											 darg2->type_oid))
				elog(ERROR, "Bug? no binary compatible cast for %s -> %s",
					 format_type_be(dtype->type_oid),
					 format_type_be(darg2->type_oid));
			cast_darg2 = psprintf("to_%s", darg2->type_name);
		}

		/*
		 * Load the key values, then compare
		 *
		 * Please pay attention that key comparison function may take
		 * arguments in different type, but binary compatible.
		 * Union data structure temp_x/temp_y implicitly convert binary
		 * compatible types, so we don't inject PG_RELABEL operator here.
		 */
		appendStringInfo(
			&body,
			"  pg_datum_ref_slot(kcxt, x_temp.%s_v,\n"
			"                    x_dclass[%d], x_values[%d]);\n"
			"  pg_datum_ref_slot(kcxt, y_temp.%s_v,\n"
			"                    y_dclass[%d], y_values[%d]);\n"
			"  if (!x_temp.%s_v.isnull && !y_temp.%s_v.isnull)\n"
			"  {\n"
			"    if (!EVAL(pgfn_%s(kcxt, %s(x_temp.%s_v), %s(y_temp.%s_v))))\n"
			"      return false;\n"
			"  }\n"
			"  else if ((x_temp.%s_v.isnull && !y_temp.%s_v.isnull) ||\n"
			"           (!x_temp.%s_v.isnull && y_temp.%s_v.isnull))\n"
			"    return false;\n"
			"\n",
			dtype->type_name,
			tle->resno-1, tle->resno-1,
			dtype->type_name,
            tle->resno-1, tle->resno-1,
			dtype->type_name, dtype->type_name,
			dfunc->func_devname,
			cast_darg1 ? cast_darg1 : "", dtype->type_name,
			cast_darg2 ? cast_darg2 : "", dtype->type_name,
			dtype->type_name, dtype->type_name,
			dtype->type_name, dtype->type_name);

		type_oid_list = list_append_unique_oid(type_oid_list,
											   dtype->type_oid);
		if (cast_darg1)
			pfree(cast_darg1);
		if (cast_darg2)
			pfree(cast_darg2);
	}
	/* declaration of temporary variable */
	pgstrom_union_type_declarations(&decl, "x_temp", type_oid_list);
	pgstrom_union_type_declarations(&decl, "y_temp", type_oid_list);

	appendStringInfo(
		kern,
		"DEVICE_FUNCTION(cl_bool)\n"
		"gpupreagg_keymatch(kern_context *kcxt,\n"
		"                   kern_data_store *x_kds, size_t x_index,\n"
		"                   kern_data_store *y_kds, size_t y_index)\n"
		"{\n"
		"%s\n%s"
		"  return true;\n"
		"}\n\n", decl.data, body.data);
	pfree(decl.data);
	pfree(body.data);
}

/*
 * gpupreagg_codegen_common_calc
 *
 * common portion of the gpupreagg_xxxx_calc() kernels
 */
static const char *
gpupreagg_codegen_common_calc(TargetEntry *tle,
							  codegen_context *context,
							  const char *aggcalc_mode)
{
	FuncExpr	   *f = (FuncExpr *)tle->expr;
	char		   *func_name;
	devtype_info   *dtype;
	const char	   *aggcalc_ops;
	const char	   *aggcalc_type;
	static char		sbuffer[128];

	/* expression should be one of partial functions */
	if (!IsA(f, FuncExpr))
		elog(ERROR, "Bug? not a partial function expression: %s",
			 nodeToString(f));
	func_name = get_func_name(f->funcid);
	if (strcmp(func_name, "pmin") == 0)
		aggcalc_ops = "min";
	else if (strcmp(func_name, "pmax") == 0)
		aggcalc_ops = "max";
	else if (strcmp(func_name, "nrows") == 0 ||
			 strcmp(func_name, "psum") == 0 ||
			 strcmp(func_name, "psum_x2") == 0 ||
			 strcmp(func_name, "pcov_x") == 0 ||
			 strcmp(func_name, "pcov_y") == 0 ||
			 strcmp(func_name, "pcov_x2") == 0 ||
			 strcmp(func_name, "pcov_y2") == 0 ||
			 strcmp(func_name, "pcov_xy") == 0)
		aggcalc_ops = "add";
	else if (strcmp(func_name, "hll_sketch_new") == 0)
	{
		pfree(func_name);
		/* HLL registers are always bytea */
		snprintf(sbuffer, sizeof(sbuffer),
				 "aggcalc_%s_hll_sketch",
				 aggcalc_mode);
		return sbuffer;
	}
	else
		elog(ERROR, "Bug? unexpected partial function expression: %s",
			 nodeToString(f));
	pfree(func_name);

	dtype = pgstrom_devtype_lookup_and_track(f->funcresulttype, context);
	if (!dtype)
		elog(ERROR, "failed on device type lookup: %s",
			 format_type_be(f->funcresulttype));

	switch (dtype->type_oid)
	{
		case INT2OID:
			aggcalc_type = "short";
			break;
		case INT4OID:
		case DATEOID:
			aggcalc_type = "int";
			break;
		case INT8OID:
		case CASHOID:
		case TIMEOID:
		case TIMESTAMPOID:
		case TIMESTAMPTZOID:
			aggcalc_type = "long";
			break;
		case FLOAT4OID:
			aggcalc_type = "float";
			break;
		case FLOAT8OID:
			aggcalc_type = "double";
			break;
		default:
			elog(ERROR, "Bug? %s is not expected to use for GpuPreAgg",
				 format_type_be(dtype->type_oid));
	}
	snprintf(sbuffer, sizeof(sbuffer),
			 "aggcalc_%s_%s_%s",
			 aggcalc_mode,
			 aggcalc_ops,
			 aggcalc_type);
	return sbuffer;
}

/*
 * code generator for initialization of a slot
 */
static void
gpupreagg_codegen_init_slot(StringInfo kern,
							codegen_context *context,
							List *tlist_part,
							int *p_num_accum_values,
							int *p_accum_extra_bufsz)
{
	StringInfoData fbuf;
	StringInfoData lbuf;
	ListCell   *lc;
	int			count1 = 0;
	int			count2 = 0;
	int			extra_total = 0;

	initStringInfo(&fbuf);
	initStringInfo(&lbuf);

	foreach (lc, tlist_part)
	{
		TargetEntry	   *tle = lfirst(lc);
		int				extra_sz = 0;
		const char	   *label;

		if (tle->resjunk)
			continue;
		if (!is_altfunc_expression((Node *)tle->expr, &extra_sz))
		{
			/* grouping-keys */
			appendStringInfo(
				&fbuf,
				"  aggcalc_init_null(&dst_dclass[%d], &dst_values[%d]);\n",
				count1 + count2,
				count1 + count2);
			count2++;
		}
		else if (extra_sz == 0)
		{
			label = gpupreagg_codegen_common_calc(tle, context, "init");
			appendStringInfo(
				&fbuf,
				"  %s(&dst_dclass[%d], &dst_values[%d]);\n",
				label,
				count1 + count2,
				count1 + count2);
			appendStringInfo(
				&lbuf,
				"  %s(&dst_dclass[%d], &dst_values[%d]);\n",
				label,
				count1,
				count1);
			count1++;
		}
		else
		{
			label = gpupreagg_codegen_common_calc(tle, context, "init");
			appendStringInfo(
				&fbuf,
				"  %s(&dst_dclass[%d], &dst_values[%d], dst_extras);\n"
				"  dst_extras += %u;\n",
				label,
				count1 + count2,
				count1 + count2,
				extra_sz);
			appendStringInfo(
				&lbuf,
				"  %s(&dst_dclass[%d], &dst_values[%d], dst_extras);\n"
				"  dst_extras += %u;\n",
				label,
				count1,
				count1,
				extra_sz);
			extra_total += extra_sz;
			count1++;
		}
	}
	appendStringInfo(
		kern,
		"DEVICE_FUNCTION(void)\n"
		"gpupreagg_init_local_slot(cl_char  *dst_dclass,\n"
		"                          Datum    *dst_values,\n"
		"                          char     *dst_extras)\n"
		"{\n%s}\n\n"
		"DEVICE_FUNCTION(void)\n"
		"gpupreagg_init_final_slot(cl_char  *dst_dclass,\n"
		"                          Datum    *dst_values,\n"
		"                          char     *dst_extras)\n"
		"{\n%s}\n\n",
		lbuf.data,
		fbuf.data);
	pfree(lbuf.data);
	pfree(fbuf.data);

	/* number of accumulate values */
	*p_num_accum_values = count1;
	/* size of extra area consumed by the accumulate values */
	*p_accum_extra_bufsz = extra_total;
}

/*
 * code generator for accum-values merge
 */
static void
gpupreagg_codegen_accum_merge(StringInfo kern,
							  codegen_context *context,
							  List *tlist_part)
{
	StringInfoData sbuf;	/* _shuffle */
	StringInfoData nbuf;	/* _normal */
	StringInfoData mbuf;	/* _merge */
	StringInfoData ubuf;	/* _update */
	ListCell   *lc;
	int			count = 0;

	initStringInfo(&sbuf);
	initStringInfo(&nbuf);
	initStringInfo(&mbuf);
	initStringInfo(&ubuf);

	foreach (lc, tlist_part)
	{
		TargetEntry	   *tle = lfirst(lc);
		const char	   *label;

		/* only partial aggregate function's arguments */
		if (tle->resjunk || !is_altfunc_expression((Node *)tle->expr, NULL))
			continue;
		label = gpupreagg_codegen_common_calc(tle, context, "shuffle");
		appendStringInfo(
			&sbuf,
			"  index = priv_attmap[%d];\n"
			"  %s(&priv_dclass[index], &priv_values[index], lane_id);\n",
			count, label);

		label = gpupreagg_codegen_common_calc(tle, context, "normal");
		appendStringInfo(
			&nbuf,
			"  dst_index = dst_attmap[%d];\n"
			"  src_index = src_attmap[%d];\n"
			"  %s(&dst_dclass[dst_index], &dst_values[dst_index], src_dclass[src_index], src_values[src_index]);\n",
			count, count, label);

		label = gpupreagg_codegen_common_calc(tle, context, "merge");
		appendStringInfo(
			&mbuf,
			"  dst_index = dst_attmap[%d];\n"
			"  src_index = src_attmap[%d];\n"
			"  %s(&dst_dclass[dst_index], &dst_values[dst_index], src_dclass[src_index], src_values[src_index]);\n",
			count, count, label);

		label = gpupreagg_codegen_common_calc(tle, context, "update");
		appendStringInfo(
			&ubuf,
			"  dst_index = dst_attmap[%d];\n"
			"  src_index = src_attmap[%d];\n"
			"  %s(&dst_dclass[dst_index], &dst_values[dst_index], src_dclass[src_index], src_values[src_index]);\n",
			count, count, label);
		count++;
    }
	
	appendStringInfo(
		kern,
		"DEVICE_FUNCTION(void)\n"
		"gpupreagg_merge_shuffle(cl_char  *priv_dclass,\n"
		"                        Datum    *priv_values,\n"
		"                        cl_short *priv_attmap,\n"
		"                        int       lane_id)\n"
		"{\n"
		"  int index;\n\n"
		"%s"
		"}\n\n"
		"DEVICE_FUNCTION(void)\n"
		"gpupreagg_update_normal(cl_char  *dst_dclass,\n"
		"                        Datum    *dst_values,\n"
		"                        cl_short *dst_attmap,\n"
		"                        cl_char  *src_dclass,\n"
		"                        Datum    *src_values,\n"
		"                        cl_short *src_attmap)\n"
		"{\n"
		"  int dst_index;\n"
		"  int src_index;\n\n"
		"%s"
		"}\n\n"
		"DEVICE_FUNCTION(void)\n"
		"gpupreagg_merge_atomic(cl_char  *dst_dclass,\n"
		"                       Datum    *dst_values,\n"
		"                       cl_short *dst_attmap,\n"
		"                       cl_char  *src_dclass,\n"
		"                       Datum    *src_values,\n"
		"                       cl_short *src_attmap)\n"
		"{\n"
		"  int dst_index;\n"
		"  int src_index;\n\n"
		"%s"
		"}\n\n"
		"DEVICE_FUNCTION(void)\n"
		"gpupreagg_update_atomic(cl_char  *dst_dclass,\n"
		"                        Datum    *dst_values,\n"
		"                        cl_short *dst_attmap,\n"
		"                        cl_char  *src_dclass,\n"
		"                        Datum    *src_values,\n"
		"                        cl_short *src_attmap)\n"
		"{\n"
		"  int dst_index;\n"
		"  int src_index;\n\n"
		"%s"
		"}\n\n",
		sbuf.data,
		nbuf.data,
		mbuf.data,
		ubuf.data);
	
	pfree(sbuf.data);
	pfree(nbuf.data);
	pfree(mbuf.data);
	pfree(ubuf.data);
}

/*
 * gpupreagg_codegen_variables
 */
static void
gpupreagg_codegen_variables(StringInfo kern,
							List *tlist_dev,
							Bitmapset *pfunc_bitmap)
{
	ListCell *lc;
	int		count;

	appendStringInfoString(
		kern,
		"__device__ cl_short GPUPREAGG_ACCUM_MAP_LOCAL[]\n"
		"  = {");
	count = 0;
	foreach (lc, tlist_dev)
	{
		TargetEntry	   *tle = lfirst(lc);

		if (!tle->resjunk && is_altfunc_expression((Node *)tle->expr, NULL))
		{
			appendStringInfo(kern, " %d,", count);
			count++;
		}
	}
	appendStringInfoString(
		kern,
		" -1 };\n"
		"__device__ cl_short GPUPREAGG_ACCUM_MAP_GLOBAL[]\n"
		"  = {");
	count = 0;
	foreach (lc, tlist_dev)
	{
		TargetEntry	   *tle = lfirst(lc);

		if (!tle->resjunk && is_altfunc_expression((Node *)tle->expr, NULL))
		{
			appendStringInfo(kern, " %d,", tle->resno - 1);
			count++;
		}
	}
	appendStringInfoString(
		kern,
		" -1 };\n"
		"__device__ cl_bool GPUPREAGG_ATTR_IS_ACCUM_VALUES[]\n"
		"  = {");
	count = 0;
	foreach (lc, tlist_dev)
	{
		appendStringInfo(kern, " %s, ",
						 bms_is_member(count, pfunc_bitmap) ? "true" : "false");
		count++;
	}
	appendStringInfoString(
		kern,
		" -1 };\n\n");
}

/*
 * gpupreagg_codegen - entrypoint of code-generator for GpuPreAgg
 */
static void
gpupreagg_codegen(PlannerInfo *root,
				  RelOptInfo *baserel,
				  CustomScan *cscan,
				  List *outer_tlist,
				  GpuPreAggInfo *gpa_info)
{
	codegen_context	context;
	StringInfoData	vars;
	StringInfoData	body;
	Bitmapset	   *pfunc_bitmap = NULL;
	size_t			extra_bufsz = 0;
	int				nfields;

	pgstrom_init_codegen_context(&context, root, baserel);
	initStringInfo(&vars);
	initStringInfo(&body);

	/* gpuscan_quals_eval */
	codegen_gpuscan_quals(&body, &context, "gpupreagg",
						  cscan->scan.scanrelid,
						  gpa_info->outer_quals);
	/* gpupreagg_projection_xxxx */
	gpupreagg_codegen_projection(&body,
								 &context,
								 cscan,
								 outer_tlist,
								 &gpa_info->tlist_part,
								 &gpa_info->tlist_prep,
								 &pfunc_bitmap);
	nfields = list_length(gpa_info->tlist_prep);
	extra_bufsz = (context.extra_bufsz +
				   MAXALIGN(sizeof(cl_char) * nfields) +	/* tup_values */
				   MAXALIGN(sizeof(Datum)   * nfields) +	/* tup_isnull */
				   MAXALIGN(sizeof(cl_int)  * nfields));	/* tup_extra */
	/* device variables */
	gpupreagg_codegen_variables(&vars, gpa_info->tlist_part, pfunc_bitmap);
	/* gpupreagg_hashvalue */
	context.extra_bufsz = 0;
	gpupreagg_codegen_hashvalue(&body, &context,
								gpa_info->tlist_prep);
	extra_bufsz = Max(extra_bufsz, context.extra_bufsz);

	/* gpupreagg_keymatch */
	context.extra_bufsz = 0;
	gpupreagg_codegen_keymatch(&body, &context,
							   gpa_info->tlist_prep);
	extra_bufsz = Max(extra_bufsz, context.extra_bufsz);

	/* gpupreagg_init_xxxx_slot */
	gpupreagg_codegen_init_slot(&body, &context,
								gpa_info->tlist_part,
								&gpa_info->num_accum_values,
								&gpa_info->accum_extra_bufsz);
	/* gpupreagg_merge_* */
	gpupreagg_codegen_accum_merge(&body, &context, gpa_info->tlist_part);

	/* merge above kernel functions */
	appendStringInfo(&context.decl, "\n%s\n%s", vars.data, body.data);

	/* store the result */
	gpa_info->kern_source = context.decl.data;
	gpa_info->extra_flags = context.extra_flags | DEVKERNEL_NEEDS_GPUPREAGG;
	gpa_info->extra_bufsz = Max(context.extra_bufsz, extra_bufsz);
	gpa_info->used_params = context.used_params;

	pfree(vars.data);
	pfree(body.data);
}

/*
 * fixup_gpupreagg_outer_quals
 *
 * Var nodes in @outer_quals were transformed to INDEX_VAR + resno form
 * through the planner stage, however, executor assumes @outer_quals shall
 * be executed towards the raw-tuples fetched from the outer relation.
 * So, we need to adjust its varno/varattno to reference the original
 * column on the raw-tuple.
 */
static Node *
fixup_gpupreagg_outer_quals(Node *node, List *tlist_dev)
{
	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		TargetEntry	   *tle;
		Var			   *varnode = (Var *) node;

		if (varnode->varno != INDEX_VAR ||
			varnode->varattno <= 0 ||
			varnode->varattno > list_length(tlist_dev))
			elog(ERROR, "Bug? unexpected Var-node in outer-quals: %s",
				 nodeToString(varnode));
		tle = list_nth(tlist_dev, varnode->varattno - 1);
		if (!IsA(tle->expr, Var))
			elog(ERROR,
				 "Bug? Var-node of outer quals references an expression: %s",
				 nodeToString(varnode));
		return (Node *) copyObject(tle->expr);
	}
	return expression_tree_mutator(node,
								   fixup_gpupreagg_outer_quals,
								   tlist_dev);
}

/*
 * gpupreagg_post_planner
 */
void
gpupreagg_post_planner(PlannedStmt *pstmt, CustomScan *cscan)
{
	GpuPreAggInfo  *gpa_info = deform_gpupreagg_info(cscan);

	gpa_info->outer_quals = (List *)
		fixup_gpupreagg_outer_quals((Node *)gpa_info->outer_quals,
									cscan->custom_scan_tlist);

	form_gpupreagg_info(cscan, gpa_info);
}

/*
 * assign_gpupreagg_session_info
 */
void
assign_gpupreagg_session_info(StringInfo buf, GpuTaskState *gts)
{
	GpuPreAggState *gpas = (GpuPreAggState *) gts;

	Assert(pgstrom_plan_is_gpupreagg(gpas->gts.css.ss.ps.plan));

	/*
	 * struct __preagg_accum_item is a local buffer to save a cumulative sum
	 * for accumulation values.
	 */
	appendStringInfo(buf, "#define __GPUPREAGG_NUM_ACCUM_VALUES %u\n",
					 gpas->num_accum_values);
	appendStringInfo(buf, "#define __GPUPREAGG_ACCUM_EXTRA_BUFSZ %u\n",
					 gpas->accum_extra_bufsz);
	appendStringInfo(buf, "#define __GPUPREAGG_LOCAL_HASH_NROOMS %u\n",
					 gpas->local_hash_nrooms);
	appendStringInfo(buf, "#define __GPUPREAGG_HLL_REGISTER_BITS %u\n",
					 pgstrom_hll_register_bits);

	/*
	 * definition of GPUPREAGG_COMBINED_JOIN disables a dummy definition
	 * of gpupreagg_projection_slot() in cuda_gpujoin.h, and switch to
	 * use the auto-generated one for initial projection of GpuPreAgg.
	 */
	if (gpas->combined_gpujoin)
		appendStringInfo(buf, "#define GPUPREAGG_COMBINED_JOIN 1\n");
}

/*
 * build_cpu_fallback_tlist
 */
static Node *
__build_cpu_fallback_tlist_recurse(Node *node, List *outer_tlist)
{
	ListCell   *lc;

	if (!node)
		return NULL;
	foreach (lc, outer_tlist)
	{
		TargetEntry *tle = lfirst(lc);

		if (equal(node, tle->expr))
		{
			return (Node *)makeVar(INDEX_VAR,
								   tle->resno,
								   exprType(node),
								   exprTypmod(node),
								   exprCollation(node), 0);
		}
	}
	return expression_tree_mutator(node, __build_cpu_fallback_tlist_recurse,
								   (void *)outer_tlist);
}

static List *
build_cpu_fallback_tlist(List *tlist_part, CustomScan *cscan)
{
	Plan	   *outer_plan = outerPlan(cscan);

	if (cscan->custom_scan_tlist != NIL)
		tlist_part = (List *)fixup_varnode_to_origin((Node *)tlist_part,
													 cscan->custom_scan_tlist);
	if (outer_plan)
	{
		List   *outer_tlist = outer_plan->targetlist;

		if ((pgstrom_plan_is_gpujoin(outer_plan) ||
			 pgstrom_plan_is_gpuscan(outer_plan)) &&
			((CustomScan *)outer_plan)->custom_scan_tlist != NIL)
		{
			outer_tlist = (List *)
				fixup_varnode_to_origin((Node *)outer_tlist,
										((CustomScan *)outer_plan)->custom_scan_tlist);
		}
		tlist_part = (List *)
			__build_cpu_fallback_tlist_recurse((Node *)tlist_part, outer_tlist);
	}
	return tlist_part;
}

/*
 * CreateGpuPreAggScanState - constructor of GpuPreAggState
 */
static Node *
CreateGpuPreAggScanState(CustomScan *cscan)
{
	/*
	 * NOTE: Per-query memory context should not be used for GpuPreAggState,
	 * because some of fields may be referenced by the worker threads which
	 * shall no be terminated on error (by the resource-owner cleanup handler
	 * at least), so these memory structure must be kept as long as the
	 * worker threads can live. Likely, CurTransactionContext is a best choice.
	 */
	GpuPreAggState *gpas = MemoryContextAllocZero(CurTransactionContext,
												  sizeof(GpuPreAggState));
	/* Set tag and executor callbacks */
	NodeSetTag(gpas, T_CustomScanState);
	gpas->gts.css.flags = cscan->flags;
	gpas->gts.css.methods = &gpupreagg_exec_methods;

	return (Node *) gpas;
}

/*
 * ExecInitGpuPreAgg
 */
static void
ExecInitGpuPreAgg(CustomScanState *node, EState *estate, int eflags)
{
	Relation		scan_rel = node->ss.ss_currentRelation;
	ExprContext	   *econtext = node->ss.ps.ps_ExprContext;
	GpuPreAggState *gpas = (GpuPreAggState *) node;
	CustomScan	   *cscan = (CustomScan *) node->ss.ps.plan;
	GpuPreAggInfo  *gpa_info = deform_gpupreagg_info(cscan);
	TupleDesc		part_tupdesc;
	TupleDesc		prep_tupdesc;
	TupleDesc		outer_tupdesc;
	List		   *tlist_fallback;
	StringInfoData	kern_define;
	ProgramId		program_id;
	size_t			length;
	bool			explain_only = ((eflags & EXEC_FLAG_EXPLAIN_ONLY) != 0);

	Assert(scan_rel ? outerPlan(node) == NULL : outerPlan(cscan) != NULL);
	/* activate a GpuContext for CUDA kernel execution */
	gpas->gts.gcontext = AllocGpuContext(gpa_info->optimal_gpus, false, false);

	/* setup common GpuTaskState fields */
	pgstromInitGpuTaskState(&gpas->gts,
							gpas->gts.gcontext,
							GpuTaskKind_GpuPreAgg,
							gpa_info->outer_quals,
							gpa_info->outer_refs,
							gpa_info->used_params,
							gpa_info->optimal_gpus,
							gpa_info->outer_nrows_per_block,
							eflags);
	gpas->gts.cb_next_task       = gpupreagg_next_task;
	gpas->gts.cb_terminator_task = gpupreagg_terminator_task;
	gpas->gts.cb_next_tuple      = gpupreagg_next_tuple;
	gpas->gts.cb_process_task    = gpupreagg_process_task;
	gpas->gts.cb_release_task    = gpupreagg_release_task;
	gpas->num_group_keys		= gpa_info->num_group_keys;
	gpas->num_accum_values		= gpa_info->num_accum_values;
	gpas->accum_extra_bufsz		= gpa_info->accum_extra_bufsz;
	/*
	 * NOTE: groupby reduction tries to use 45kB of shared memory per SM
	 * for the local hash area. Number of the local hash items depends on
	 * the memory consumption for each row. It can be zero, if unit size
	 * of the shared memory consumption is too large.
	 */
	gpas->local_hash_nrooms
		= ((45 * 1024 - sizeof(preagg_local_hashtable))
		   / (sizeof(preagg_hash_item) +
			  sizeof(cl_char) * gpas->num_accum_values +
			  sizeof(Datum)   * gpas->num_accum_values +
			  gpas->accum_extra_bufsz));

	/* initialization of the outer relation */
	if (outerPlan(cscan))
	{
		Plan	   *outer_plan = outerPlan(cscan);
		PlanState  *outer_ps;

		Assert(!scan_rel);
		Assert(!gpa_info->outer_quals );
		outer_ps = ExecInitNode(outer_plan, estate, eflags);
		if (enable_pullup_outer_join &&
			pgstrom_planstate_is_gpujoin(outer_ps) &&
			!outer_ps->ps_ProjInfo)
		{
			gpas->combined_gpujoin = true;
		}
		outerPlanState(gpas) = outer_ps;
		/* GpuPreAgg don't need re-initialization of projection info */
		outer_tupdesc = planStateResultTupleDesc(outer_ps);
		/* should not have any usage of BRIN-index */
		Assert(!OidIsValid(gpa_info->index_oid));
	}
	else
	{
		Assert(scan_rel != NULL);
		gpas->outer_quals = ExecInitQual(gpa_info->outer_quals,
										 &gpas->gts.css.ss.ps);
		outer_tupdesc = RelationGetDescr(scan_rel);
		pgstromExecInitBrinIndexMap(&gpas->gts,
									gpa_info->index_oid,
									gpa_info->index_conds,
									gpa_info->index_quals);
	}

	/* Setup TupleTableSlot */
	part_tupdesc = ExecCleanTypeFromTL(gpa_info->tlist_part);
	gpas->part_slot = MakeSingleTupleTableSlot(part_tupdesc,
											   &TTSOpsVirtual);
	prep_tupdesc = ExecCleanTypeFromTL(gpa_info->tlist_prep);
	gpas->prep_slot = MakeSingleTupleTableSlot(prep_tupdesc,
											   &TTSOpsVirtual);
	gpas->outer_slot = MakeSingleTupleTableSlot(outer_tupdesc,
												&TTSOpsHeapTuple);
	tlist_fallback = build_cpu_fallback_tlist(gpa_info->tlist_part, cscan);
	gpas->fallback_proj = ExecBuildProjectionInfo(tlist_fallback,
												  econtext,
												  gpas->part_slot,
												  &gpas->gts.css.ss.ps,
												  outer_tupdesc);
	ExecInitScanTupleSlot(estate,
						  &gpas->gts.css.ss,
						  part_tupdesc,
						  &TTSOpsVirtual);
	ExecAssignScanProjectionInfoWithVarno(&gpas->gts.css.ss, INDEX_VAR);

	/* Template of kds_slot */
	length = KDS_calculateHeadSize(prep_tupdesc);
	gpas->kds_slot_head = MemoryContextAllocZero(CurTransactionContext,
												 length);
	init_kernel_data_store(gpas->kds_slot_head,
						   prep_tupdesc,
						   INT_MAX,		/* to be set individually */
						   KDS_FORMAT_SLOT,
						   INT_MAX);	/* to be set individually */

	/* Save the plan-time estimations */
	gpas->plan_nrows_per_chunk =
		(gpa_info->plan_nchunks > 0
		 ? gpa_info->outer_nrows / gpa_info->plan_nchunks
		 : gpa_info->outer_nrows);
    gpas->plan_nrows_in		= gpa_info->outer_nrows;
	gpas->plan_ngroups		= gpa_info->plan_ngroups;

	/* Get CUDA program and async build if any */
	if (gpas->combined_gpujoin)
	{
		program_id = GpuJoinCreateCombinedProgram(outerPlanState(gpas),
												  &gpas->gts,
												  gpa_info->extra_flags,
												  gpa_info->extra_bufsz,
												  gpa_info->kern_source,
												  explain_only);
	}
	else
	{
		initStringInfo(&kern_define);
		pgstrom_build_session_info(&kern_define,
								   &gpas->gts,
								   gpa_info->extra_flags);
		program_id = pgstrom_create_cuda_program(gpas->gts.gcontext,
												 gpa_info->extra_flags,
												 gpa_info->extra_bufsz,
												 gpa_info->kern_source,
												 kern_define.data,
												 false,
												 explain_only);
		pfree(kern_define.data);
	}
	gpas->gts.program_id = program_id;
}

/*
 * ExecReCheckGpuPreAgg
 */
static bool
ExecReCheckGpuPreAgg(CustomScanState *node, TupleTableSlot *slot)
{
	/*
	 * GpuPreAgg shall be never located under the LockRows, so we don't
	 * expect that we need to have valid EPQ recheck here.
	 */
	return true;
}

/*
 * ExecGpuPreAgg
 */
static TupleTableSlot *
ExecGpuPreAgg(CustomScanState *node)
{
	GpuPreAggState *gpas = (GpuPreAggState *) node;

	ActivateGpuContext(gpas->gts.gcontext);
	if (!gpas->gpa_sstate)
		createGpuPreAggSharedState(gpas, NULL, NULL);
	return ExecScan(&node->ss,
					(ExecScanAccessMtd) pgstromExecGpuTaskState,
					(ExecScanRecheckMtd) ExecReCheckGpuPreAgg);
}

/*
 * ExecEndGpuPreAgg
 */
static void
ExecEndGpuPreAgg(CustomScanState *node)
{
	GpuPreAggState *gpas = (GpuPreAggState *) node;
	GpuTaskRuntimeStat *gt_rtstat = (GpuTaskRuntimeStat *) gpas->gpa_rtstat;
	GpuContext	   *gcontext = gpas->gts.gcontext;
	CUresult		rc;

	/* wait for completion of any asynchronous GpuTask */
	if (gpas->ev_init_fhash)
	{
		GPUCONTEXT_PUSH(gcontext);
		rc = cuEventRecord(gpas->ev_init_fhash,
						   CU_STREAM_PER_THREAD);
		GPUCONTEXT_POP(gcontext);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventRecord: %s", errorText(rc));
	}
    SynchronizeGpuContext(gpas->gts.gcontext);
	/* close index related stuff if any */
	pgstromExecEndBrinIndexMap(&gpas->gts);
	/* clean up subtree, if any */
	if (outerPlanState(node))
		ExecEndNode(outerPlanState(node));

	/* release final buffer / hashslot */
	if (gpas->pds_final)
		PDS_release(gpas->pds_final);
	if (gpas->m_fhash)
		gpuMemFree(gcontext, gpas->m_fhash);

	/* release any other resources */
	if (gpas->part_slot)
		ExecDropSingleTupleTableSlot(gpas->part_slot);
	if (gpas->prep_slot)
		ExecDropSingleTupleTableSlot(gpas->prep_slot);
	if (gpas->outer_slot)
		ExecDropSingleTupleTableSlot(gpas->outer_slot);
	releaseGpuPreAggSharedState(gpas);
	pgstromReleaseGpuTaskState(&gpas->gts, gt_rtstat);
}

/*
 * ExecReScanGpuPreAgg
 */
static void
ExecReScanGpuPreAgg(CustomScanState *node)
{
	GpuPreAggState	   *gpas = (GpuPreAggState *) node;

	/* wait for completion of any asynchronous GpuTask */
	SynchronizeGpuContext(gpas->gts.gcontext);
	/* also rescan subtree, if any */
	if (outerPlanState(node))
		ExecEndNode(outerPlanState(node));
	/* reset shared state */
	resetGpuPreAggSharedState(gpas);
	/* common rescan handling */
	pgstromRescanGpuTaskState(&gpas->gts);
	/* reset other stuff */
	gpas->terminator_done = false;
}

/*
 * ExecGpuPreAggEstimateDSM
 */
static Size
ExecGpuPreAggEstimateDSM(CustomScanState *node, ParallelContext *pcxt)
{
	return (MAXALIGN(sizeof(GpuPreAggSharedState)) +
			pgstromSizeOfBrinIndexMap((GpuTaskState *) node) +
			pgstromEstimateDSMGpuTaskState((GpuTaskState *)node, pcxt));
}

/*
 * ExecGpuPreAggInitDSM
 */
static void
ExecGpuPreAggInitDSM(CustomScanState *node,
					 ParallelContext *pcxt,
					 void *coordinate)
{
	GpuPreAggState *gpas = (GpuPreAggState *) node;
	size_t		len;

	/* save ParallelContext */
	gpas->gts.pcxt = pcxt;
	on_dsm_detach(pcxt->seg,
				  SynchronizeGpuContextOnDSMDetach,
				  PointerGetDatum(gpas->gts.gcontext));
	/* allocation of shared state */
	len = createGpuPreAggSharedState(gpas, pcxt, coordinate);
	coordinate = (char *)coordinate + len;
	if (gpas->gts.outer_index_state)
	{
		gpas->gts.outer_index_map = (Bitmapset *)coordinate;
		gpas->gts.outer_index_map->nwords = -1;	/* uninitialized */
		coordinate = ((char *)coordinate +
					  pgstromSizeOfBrinIndexMap(&gpas->gts));
	}
	pgstromInitDSMGpuTaskState(&gpas->gts, pcxt, coordinate);
	pg_memory_barrier();
}

/*
 * ExecGpuPreAggInitWorker
 */
static void
ExecGpuPreAggInitWorker(CustomScanState *node,
						shm_toc *toc,
						void *coordinate)
{
	GpuPreAggState		   *gpas = (GpuPreAggState *) node;
	GpuPreAggSharedState   *gpa_sstate = coordinate;

	pg_memory_barrier();
	gpas->gpa_sstate = gpa_sstate;
	gpas->gpa_rtstat = &gpa_sstate->gpa_rtstat;

	on_dsm_detach(dsm_find_mapping(gpa_sstate->ss_handle),
				  SynchronizeGpuContextOnDSMDetach,
				  PointerGetDatum(gpas->gts.gcontext));
	coordinate = (char *)coordinate + gpa_sstate->ss_length;
	if (gpas->gts.outer_index_state)
	{
		gpas->gts.outer_index_map = (Bitmapset *)coordinate;
		coordinate = ((char *)coordinate +
					  pgstromSizeOfBrinIndexMap(&gpas->gts));
	}
	pgstromInitWorkerGpuTaskState(&gpas->gts, coordinate);
}

/*
 * ExecGpuPreAggReInitializeDSM
 */
static void
ExecGpuPreAggReInitializeDSM(CustomScanState *node,
							 ParallelContext *pcxt, void *coordinate)
{
	pgstromReInitializeDSMGpuTaskState((GpuTaskState *) node);
}

/*
 * ExecShutdownGpuPreAgg
 */
static void
ExecShutdownGpuPreAgg(CustomScanState *node)
{
	GpuPreAggState	   *gpas = (GpuPreAggState *) node;
	GpuPreAggRuntimeStat *gpa_rtstat_old = gpas->gpa_rtstat;
	GpuPreAggRuntimeStat *gpa_rtstat_new;

	/*
	 * If this GpuPreAgg node is located under the inner side of
	 * another GpuJoin, it should not be called under the background
	 * worker context, however, ExecShutdown walks down the node.
	 */
	if (!gpa_rtstat_old)
		return;
	/* parallel worker put runtime-stat on ExecEnd handler */
	if (IsParallelWorker())
		mergeGpuTaskRuntimeStatParallelWorker(&gpas->gts, &gpa_rtstat_old->c);
	else
	{
		EState	   *estate = gpas->gts.css.ss.ps.state;

		gpa_rtstat_new = MemoryContextAlloc(estate->es_query_cxt,
											sizeof(GpuPreAggRuntimeStat));
		memcpy(gpa_rtstat_new,
			   gpa_rtstat_old,
			   sizeof(GpuPreAggRuntimeStat));
		gpas->gpa_rtstat = gpa_rtstat_new;
	}
	pgstromShutdownDSMGpuTaskState(&gpas->gts);
}

/*
 * ExplainGpuPreAgg
 */
static void
ExplainGpuPreAgg(CustomScanState *node, List *ancestors, ExplainState *es)
{
	GpuPreAggState		   *gpas = (GpuPreAggState *) node;
	GpuPreAggRuntimeStat   *gpa_rtstat = gpas->gpa_rtstat;
	CustomScan			   *cscan = (CustomScan *) node->ss.ps.plan;
	GpuPreAggInfo		   *gpa_info = deform_gpupreagg_info(cscan);
	List				   *dcontext;
	List				   *group_keys = NIL;
	ListCell			   *lc;
	char				   *temp;
	char					buf[200];

	if (gpa_rtstat)
		mergeGpuTaskRuntimeStat(&gpas->gts, &gpa_rtstat->c);

	/* Set up deparsing context */
	dcontext = set_deparse_context_planstate(es->deparse_cxt,
                                            (Node *)&gpas->gts.css.ss.ps,
                                            ancestors);
	/* extract grouping keys */
	foreach (lc, cscan->custom_scan_tlist)
	{
		TargetEntry	   *tle = lfirst(lc);

		if (tle->ressortgroupref)
			group_keys = lappend(group_keys, tle->expr);
	}

	if (es->verbose)
	{
		List	   *__tlist_part = NIL;
		List	   *__tlist_prep = NIL;

		foreach (lc, gpa_info->tlist_part)
		{
			TargetEntry	   *tle = lfirst(lc);

			__tlist_part = lappend(__tlist_part, tle->expr);
		}
		temp = deparse_expression((Node *)__tlist_part,
								  dcontext, false, false);
		ExplainPropertyText("GPU Output", temp, es);

		foreach (lc, gpa_info->tlist_prep)
		{
			TargetEntry	   *tle = lfirst(lc);

			__tlist_prep = lappend(__tlist_prep, tle->expr);
        }
		temp = deparse_expression((Node *)__tlist_prep,
								  dcontext, false, false);
		ExplainPropertyText("GPU Setup", temp, es);
	}

	if (gpas->num_group_keys == 0)
	{
		Assert(group_keys == NIL);
		ExplainPropertyText("Reduction", "NoGroup", es);
	}
	else
	{
		Assert(group_keys != 0);
		if (gpas->local_hash_nrooms == 0)
			ExplainPropertyText("Reduction", "GroupBy (Global Only)", es);
		else
		{
			snprintf(buf, sizeof(buf),
					 "GroupBy (Global+Local [nrooms: %u])",
					 gpas->local_hash_nrooms);
			ExplainPropertyText("Reduction", buf, es);
		}
		temp = deparse_expression((Node *)group_keys, dcontext,
								  es->verbose, false);
		ExplainPropertyText("Group keys", temp, es);
	}

	pgstromExplainOuterScan(&gpas->gts,
							dcontext, ancestors, es,
							gpa_info->outer_quals,
							gpa_info->outer_startup_cost,
							gpa_info->outer_total_cost,
							gpa_info->outer_nrows,
							gpa_info->outer_width);
	/* combined GpuJoin + GpuPreAgg? */
	if (gpas->combined_gpujoin)
		ExplainPropertyText("Combined GpuJoin", "enabled", es);
	else if (es->format != EXPLAIN_FORMAT_TEXT)
		ExplainPropertyText("Combined GpuJoin", "disabled", es);
	/* other common fields */
	pgstromExplainGpuTaskState(&gpas->gts, es, dcontext);
	/* other run-time statistics, if any */
	if (gpa_rtstat)
	{
		uint64		fallback_count
			= pg_atomic_read_u64(&gpa_rtstat->c.fallback_count);

		if (fallback_count > 0)
			ExplainPropertyInteger("Num of CPU fallback rows",
								   NULL, fallback_count, es);
	}
}

/*
 * createGpuPreAggSharedState
 */
static size_t
createGpuPreAggSharedState(GpuPreAggState *gpas,
						   ParallelContext *pcxt,
						   void *dsm_addr)
{
	EState	   *estate = gpas->gts.css.ss.ps.state;
	GpuPreAggSharedState *gpa_sstate;
	GpuPreAggRuntimeStat *gpa_rtstat;
	size_t		ss_length = MAXALIGN(sizeof(GpuPreAggSharedState));

	if (dsm_addr)
		gpa_sstate = dsm_addr;
	else
		gpa_sstate = MemoryContextAlloc(estate->es_query_cxt, ss_length);
	memset(gpa_sstate, 0, ss_length);
	gpa_sstate->ss_handle = (pcxt ? dsm_segment_handle(pcxt->seg) : UINT_MAX);
	gpa_sstate->ss_length = ss_length;

	gpa_rtstat = &gpa_sstate->gpa_rtstat;
	SpinLockInit(&gpa_rtstat->c.lock);

	gpas->gpa_sstate = gpa_sstate;
	gpas->gpa_rtstat = gpa_rtstat;

	return ss_length;
}

/*
 * releaseGpuPreAggSharedState
 */
static void
releaseGpuPreAggSharedState(GpuPreAggState *gpas)
{
	/* to be moved to shutdown handler?? */
}

/*
 * resetGpuPreAggSharedState
 */
static void
resetGpuPreAggSharedState(GpuPreAggState *gpas)
{
	/* nothing to do */
}

/*
 * gpupreagg_alloc_final_buffer
 */
static void
gpupreagg_alloc_final_buffer(GpuPreAggState *gpas)
{
	GpuContext	   *gcontext = gpas->gts.gcontext;
	TupleTableSlot *part_slot = gpas->part_slot;
	TupleDesc		part_tupdesc = part_slot->tts_tupleDescriptor;
	pgstrom_data_store *pds_final;
	size_t			f_length;
	size_t			f_hash_nslots = 0;
	size_t			f_hash_length = 0;
	CUdeviceptr		m_fhash = 0UL;
	CUresult		rc;

	if (gpas->pds_final)
		return;

	/* final buffer allocation */
	if (gpas->num_group_keys == 0)
		f_length = 0x00ffe000UL;		/* almost 16MB managed */
	else
		f_length = 0x3ffffe000UL;		/* almost 16GB managed */

	pds_final = PDS_create_slot(gcontext,
								part_tupdesc,
								f_length);
	/* final hash-slot allocation */
	if (gpas->num_group_keys > 0)
	{
		if (gpas->plan_ngroups < 400000)
			f_hash_nslots = 4 * gpas->plan_ngroups;
		else if (gpas->plan_ngroups < 1200000)
			f_hash_nslots = 3 * gpas->plan_ngroups;
		else if (gpas->plan_ngroups < 4000000)
			f_hash_nslots = 2 * gpas->plan_ngroups;
		else if (gpas->plan_ngroups < 10000000)
			f_hash_nslots = (double)gpas->plan_ngroups * 1.25;
		else
			f_hash_nslots = gpas->plan_ngroups;

		f_hash_length = 0xffffe000UL;	/* almost 4GB managed */
		/*
		 * The final hash-slot allocation. It initially use the leading
		 * offsetof(kern_global_hashslot, slots[f_hash_nslots]) bytes
		 * of the managed device memory, thus, not entire memory chunk
		 * is used physically at that time.
		 * Once @f_hash_nslots becomes unsufficient, GPU kernel expand
		 * the hash-slot on the demand.
		 */
		rc = gpuMemAllocManaged(gcontext,
								&m_fhash,
								f_hash_length,
								CU_MEM_ATTACH_GLOBAL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on gpuMemAllocManaged: %s", errorText(rc));
	}
	gpas->pds_final		= pds_final;
	gpas->m_fhash		= m_fhash;
	gpas->ev_init_fhash	= NULL;
	gpas->f_hash_nslots	= f_hash_nslots;
	gpas->f_hash_length = f_hash_length;
}

/*
 * gpupreagg_create_task - constructor of GpuPreAggTask
 */
static GpuTask *
gpupreagg_create_task(GpuPreAggState *gpas,
					  pgstrom_data_store *pds_src,
					  CUdeviceptr m_kmrels,
					  int outer_depth)
{
	GpuContext	   *gcontext = gpas->gts.gcontext;
	GpuPreAggTask  *gpreagg;
	bool			with_nvme_strom = false;
	cl_uint			nrows_per_block = 0;
	kern_data_store *kds_slot = gpas->kds_slot_head;
	size_t			unitsz;
	size_t			kds_slot_nrooms;
	size_t			kds_slot_length;
	CUdeviceptr		m_deviceptr;
	CUresult		rc;
	cl_int			sm_count;
	Size			head_sz;
	Size			suspend_sz = 0;
	Size			kgjoin_len = 0;

	/* allocation of the final-buffer on demand */
	if (!gpas->pds_final)
		gpupreagg_alloc_final_buffer(gpas);

	/* rough estimation of the result buffer */
	if (pds_src)
	{
		cl_uint		__nrooms = pds_src->kds.nitems;

		if (pds_src->kds.format == KDS_FORMAT_BLOCK)
		{
			struct NVMEScanState *nvme_sstate;

			nvme_sstate = gpas->combined_gpujoin
				? ((GpuTaskState *)outerPlanState(gpas))->nvme_sstate
				: gpas->gts.nvme_sstate;

			Assert(nvme_sstate != NULL);
			Assert(pds_src->filedesc.rawfd >= 0 || pds_src->nblocks_uncached == 0);
			with_nvme_strom = (pds_src->nblocks_uncached > 0);
			nrows_per_block = nvme_sstate->nrows_per_block;
			/*
			 * It is arguable whether 150% of nrows_per_block * nitems; which
			 * means number of blocks in KDS_FORMAT_BLOCK, is adeque
			 * estimation, or not.
			 * Suspend/Resume like GpuJoin may make sense for the future
			 * improvement.
			 */
			__nrooms = 1.5 * (double)(__nrooms * nrows_per_block);
		}
		else if (pds_src->kds.format == KDS_FORMAT_ARROW &&
				 pds_src->iovec != NULL)
		{
			with_nvme_strom = true;
		}
		/* Extra buffer for suspend resume */
		sm_count = devAttrs[gcontext->cuda_dindex].MULTIPROCESSOR_COUNT;
		suspend_sz = STROMALIGN(sizeof(gpuscanSuspendContext) *
								GPUKERNEL_MAX_SM_MULTIPLICITY * sm_count);

		unitsz = MAXALIGN((sizeof(Datum) + sizeof(char)) * kds_slot->ncols);
		kds_slot_length = (KERN_DATA_STORE_HEAD_LENGTH(kds_slot) +
						   unitsz * __nrooms);
		kds_slot_length = Max(kds_slot_length, 16<<20);
	}
	else
	{
		/* combined RIGHT OUTER JOIN or terminator task */
		unitsz = MAXALIGN((sizeof(Datum) + sizeof(char)) * kds_slot->ncols);
		kds_slot_length = pgstrom_chunk_size();
	}
	kds_slot_nrooms = (kds_slot_length -
					   KERN_DATA_STORE_HEAD_LENGTH(kds_slot)) / unitsz;

	/* allocation of GpuPreAggTask */
	head_sz = STROMALIGN(offsetof(GpuPreAggTask, kern.data));
	if (gpas->combined_gpujoin)
	{
		GpuTaskState   *outer_gts = (GpuTaskState *) outerPlanState(gpas);
		kgjoin_len = GpuJoinSetupTask(NULL, outer_gts, pds_src);
	}

	rc = gpuMemAllocManaged(gcontext,
							&m_deviceptr,
							head_sz + suspend_sz + kgjoin_len,
							CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on gpuMemAllocManaged: %s", errorText(rc));
	gpreagg = (GpuPreAggTask *)m_deviceptr;
	memset(gpreagg, 0, head_sz + suspend_sz);

	pgstromInitGpuTask(&gpas->gts, &gpreagg->task);
	gpreagg->with_nvme_strom = with_nvme_strom;
	gpreagg->pds_src = pds_src;
	gpreagg->kds_slot = NULL;
	gpreagg->kds_slot_nrooms = kds_slot_nrooms;
	gpreagg->kds_slot_length = kds_slot_length;
	if (gpas->combined_gpujoin)
	{
		GpuTaskState   *outer_gts = (GpuTaskState *) outerPlanState(gpas);

		pgstromSetupKernParambuf(outer_gts);
		gpreagg->kgjoin = (kern_gpujoin *)((char *)gpreagg + head_sz + suspend_sz);
		GpuJoinSetupTask(gpreagg->kgjoin, outer_gts, pds_src);
		gpreagg->m_kmrels = m_kmrels;
		gpreagg->m_kparams = outer_gts->kern_params;
		gpreagg->outer_depth = outer_depth;
	}
	else
	{
		Assert(m_kmrels == 0UL);
	}
	/* if any grouping keys, determine the reduction policy later */
	gpreagg->kern.num_group_keys = gpas->num_group_keys;
	gpreagg->kern.suspend_size = suspend_sz;

	return &gpreagg->task;
}

/*
 * gpupreagg_next_task
 *
 * callback to construct a new GpuPreAggTask task object based on
 * the input data stream that is scanned.
 */
static GpuTask *
gpupreagg_next_task(GpuTaskState *gts)
{
	GpuPreAggState *gpas = (GpuPreAggState *) gts;
	GpuContext	   *gcontext = gpas->gts.gcontext;
	GpuTask		   *gtask = NULL;
	pgstrom_data_store *pds = NULL;
	CUdeviceptr		m_kmrels = 0UL;

	if (gpas->combined_gpujoin)
	{
		GpuTaskState   *outer_gts = (GpuTaskState *) outerPlanState(gpas);

		if (!GpuJoinInnerPreload(outer_gts, &m_kmrels))
			return NULL;	/* an empty results */
		pds = GpuJoinExecOuterScanChunk(outer_gts);
	}
	else if (gpas->gts.css.ss.ss_currentRelation)
	{
		if (gpas->gts.af_state)
			pds = ExecScanChunkArrowFdw(&gpas->gts);
		else if (gpas->gts.gc_state)
			pds = ExecScanChunkGpuCache(&gpas->gts);
		else
			pds = pgstromExecScanChunk(&gpas->gts);
	}
	else
	{
		PlanState	   *outer_ps = outerPlanState(gpas);
		TupleDesc		tupdesc = ExecGetResultType(outer_ps);
		TupleTableSlot *slot;

		while (true)
		{
			if (gpas->gts.scan_overflow)
			{
				if (gpas->gts.scan_overflow == (void *)(~0UL))
					break;
				slot = gpas->gts.scan_overflow;
				gpas->gts.scan_overflow = NULL;
			}
			else
			{
				slot = ExecProcNode(outer_ps);
				if (TupIsNull(slot))
				{
					gpas->gts.scan_overflow = (void *)(~0UL);
					break;
				}
			}

			/* create a new data-store on demand */
			if (!pds)
			{
				pds = PDS_create_row(gcontext,
									 tupdesc,
									 pgstrom_chunk_size());
			}

			if (!PDS_insert_tuple(pds, slot))
			{
				gpas->gts.scan_overflow = slot;
				break;
			}
		}
	}
	if (pds)
		gtask = gpupreagg_create_task(gpas, pds, m_kmrels, -1);
	return gtask;
}

/*
 * gpupreagg_terminator_task
 */
static GpuTask *
gpupreagg_terminator_task(GpuTaskState *gts, cl_bool *task_is_ready)
{
	GpuPreAggState *gpas = (GpuPreAggState *) gts;

	if (gpas->terminator_done)
		return NULL;

	/*
	 * Do we need to kick RIGHT OUTER JOIN + GpuPreAgg task if combined mode.
	 */
	if (gpas->combined_gpujoin)
	{
		GpuTaskState   *outer_gts = (GpuTaskState *)outerPlanState(gpas);
		CUdeviceptr		m_kmrels = 0UL;
		cl_int			outer_depth;

		/* Has RIGHT/FULL OUTER JOIN? */
		outer_depth = gpujoinNextRightOuterJoinIfAny(outer_gts);
		if (outer_depth > 0 &&
			GpuJoinInnerPreload(outer_gts, &m_kmrels))
			return gpupreagg_create_task(gpas, NULL, m_kmrels, outer_depth);
	}
	/* setup a terminator task */
	gpas->terminator_done = true;
	*task_is_ready = true;
	return gpupreagg_create_task(gpas, NULL, 0UL, -1);
}

/*
 * gpupreagg_next_tuple_fallback
 */
static TupleTableSlot *
gpupreagg_next_tuple_fallback(GpuPreAggState *gpas, GpuPreAggTask *gpreagg)
{
	GpuPreAggRuntimeStat *gpa_rtstat = gpas->gpa_rtstat;
	ExprContext	   *econtext = gpas->gts.css.ss.ps.ps_ExprContext;

	for (;;)
	{
		if (gpas->combined_gpujoin)
		{
			TupleTableSlot *slot;
			GpuTaskState   *outer_gts
				= (GpuTaskState *)outerPlanState(gpas);

			slot = gpujoinNextTupleFallbackUpper(outer_gts,
												 gpreagg->kgjoin,
												 gpreagg->pds_src,
												 Max(gpreagg->outer_depth, 0));
			if (TupIsNull(slot))
				return NULL;
			econtext->ecxt_scantuple = slot;
		}
		else
		{
			/* fetch a tuple from the data-store */
			ExecClearTuple(gpas->outer_slot);
			if (!gpreagg->pds_src ||
				!PDS_fetch_tuple(gpas->outer_slot,
								 gpreagg->pds_src,
								 &gpas->gts))
				return NULL;
			econtext->ecxt_scantuple = gpas->outer_slot;
		}
		pg_atomic_add_fetch_u64(&gpa_rtstat->c.source_nitems, 1L);

		/* filter out the tuple, if any outer quals */
		if (!ExecQual(gpas->outer_quals, econtext))
		{
			pg_atomic_add_fetch_u64(&gpa_rtstat->c.nitems_filtered, 1L);
			continue;
		}
		pg_atomic_add_fetch_u64(&gpa_rtstat->c.fallback_count, 1);
		/* makes a projection from the outer-scan to the pseudo-tlist */
		if (!gpas->fallback_proj)
			return econtext->ecxt_scantuple;
		return ExecProject(gpas->fallback_proj);
	}
	return NULL;
}

/*
 * gpupreagg_next_tuple
 */
static TupleTableSlot *
gpupreagg_next_tuple(GpuTaskState *gts)
{
	GpuPreAggState	   *gpas = (GpuPreAggState *) gts;
	GpuPreAggTask	   *gpreagg = (GpuPreAggTask *) gpas->gts.curr_task;
	pgstrom_data_store *pds_final = gpas->pds_final;
	TupleTableSlot	   *slot = NULL;

	if (gpreagg->task.cpu_fallback)
	{
		slot = gpupreagg_next_tuple_fallback(gpas, gpreagg);
	}
	else if (gpas->gts.curr_index < pds_final->kds.nitems)
	{
		slot = gpas->part_slot;
		ExecClearTuple(slot);
		PDS_fetch_tuple(slot, pds_final, &gpas->gts);
	}
	return slot;
}

/*
 * gpupreagg_init_final_hash
 */
static void
gpupreagg_init_final_hash(GpuPreAggTask *gpreagg,
						  CUmodule cuda_module)
{
	GpuPreAggState *gpas = (GpuPreAggState *)gpreagg->task.gts;
	CUfunction	kern_init_fhash;
	CUevent		ev_init_fhash;
	CUresult	rc;
	cl_int		grid_sz;
	cl_int		block_sz;
	void	   *kern_args[3];

	/*
	 * NoGroup reduction does not have final-hash buffer, thus
	 * no need to initialize this.
	 */
	if (gpas->m_fhash == 0UL)
		return;

	pthreadMutexLock(&gpas->f_mutex);
	STROM_TRY();
	{
		if (!gpas->ev_init_fhash)
		{
			rc = cuModuleGetFunction(&kern_init_fhash,
									 cuda_module,
									 "gpupreagg_init_final_hash");
			if (rc != CUDA_SUCCESS)
				werror("failed on cuModuleGetFunction: %s", errorText(rc));

			rc = cuEventCreate(&ev_init_fhash,
							   CU_EVENT_BLOCKING_SYNC);
			if (rc != CUDA_SUCCESS)
				werror("failed on cuEventCreate: %s", errorText(rc));

			rc = gpuOptimalBlockSize(&grid_sz,
									 &block_sz,
									 kern_init_fhash,
									 CU_DEVICE_PER_THREAD,
									 0, 0);
			if (rc != CUDA_SUCCESS)
				werror("failed on gpuOptimalBlockSize: %s", errorText(rc));
			grid_sz = Min(grid_sz, (gpas->f_hash_nslots +
									block_sz - 1) / block_sz);
			kern_args[0] = &gpas->m_fhash;
			kern_args[1] = &gpas->f_hash_nslots;
			kern_args[2] = &gpas->f_hash_length;
			rc = cuLaunchKernel(kern_init_fhash,
								grid_sz, 1, 1,
								block_sz, 1, 1,
								0,
								CU_STREAM_PER_THREAD,
								kern_args,
								NULL);
			if (rc != CUDA_SUCCESS)
				werror("failed on cuLaunchKernel: %s", errorText(rc));

			rc = cuEventRecord(ev_init_fhash,
							   CU_STREAM_PER_THREAD);
			if (rc != CUDA_SUCCESS)
				werror("failed on cuEventRecord: %s", errorText(rc));

			gpas->ev_init_fhash = ev_init_fhash;

			rc = cuStreamSynchronize(CU_STREAM_PER_THREAD);
			if (rc != CUDA_SUCCESS)
				werror("failed on cuStreamSynchronize: %s", errorText(rc));
		}
	}
	STROM_CATCH();
	{
		pthreadMutexUnlock(&gpas->f_mutex);
		STROM_RE_THROW();
	}
	STROM_END_TRY();
	pthreadMutexUnlock(&gpas->f_mutex);
	/* Point of synchronization */
	rc = cuStreamWaitEvent(CU_STREAM_PER_THREAD,
						   gpas->ev_init_fhash,
						   0);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuStreamWaitEvent: %s", errorText(rc));
}

/*
 * gpupreaggUpdateRunTimeStat
 */
static void
gpupreaggUpdateRunTimeStat(GpuTaskState *gts, kern_gpupreagg *kgpreagg)
{
	GpuPreAggState *gpas = (GpuPreAggState *) gts;
	GpuPreAggRuntimeStat *gpa_rtstat = gpas->gpa_rtstat;

	pg_atomic_add_fetch_u64(&gpa_rtstat->c.source_nitems,
							(int64)kgpreagg->nitems_real);
	pg_atomic_add_fetch_u64(&gpa_rtstat->c.nitems_filtered,
							(int64)kgpreagg->nitems_filtered);
	//TODO: other statistics
}

/*
 * gpupreagg_throw_partial_result
 */
static void
gpupreagg_throw_partial_result(GpuPreAggTask *gpreagg,
							   kern_data_store *kds_slot)
{
	GpuContext	   *gcontext = GpuWorkerCurrentContext;
	GpuTaskState   *gts = gpreagg->task.gts;
	GpuPreAggTask  *gresp;		/* responder task */
	CUresult		rc;

	/* async prefetch kds_slot; which should be on the device memory */
	rc = cuMemPrefetchAsync((CUdeviceptr) kds_slot,
							gpreagg->kds_slot_length,
							CU_DEVICE_CPU,
							CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuMemPrefetchAsync: %s", errorText(rc));

	/* setup responder task with supplied @kds_slot */
	rc = gpuMemAllocManaged(gcontext,
							(CUdeviceptr *)&gresp,
							offsetof(GpuPreAggTask, kern.data),
							CU_MEM_ATTACH_GLOBAL);
	if (rc != CUDA_SUCCESS)
		werror("failed on gpuMemAllocManaged: %s", errorText(rc));
	memset(gresp, 0, offsetof(GpuPreAggTask, kern.data));
	gresp->task.task_kind	= gpreagg->task.task_kind;
	gresp->task.program_id	= gpreagg->task.program_id;
	gresp->task.cpu_fallback= true;
	gresp->task.gts			= gts;
	gresp->pds_src			= PDS_retain(gpreagg->pds_src);
	gresp->kds_slot			= kds_slot;
	gresp->kds_slot_nrooms	= gpreagg->kds_slot_nrooms;
	gresp->kds_slot_length	= gpreagg->kds_slot_length;

	/* Back GpuTask to GTS */
	pthreadMutexLock(&gcontext->worker_mutex);
	dlist_push_tail(&gts->ready_tasks,
					&gresp->task.chain);
	gts->num_ready_tasks++;
	pthreadMutexUnlock(&gcontext->worker_mutex);

	SetLatch(MyLatch);
}

/*
 * gpupreagg_process_reduction_task
 *
 * main logic to kick GpuPreAgg kernel function.
 */
static int
gpupreagg_process_reduction_task(GpuPreAggTask *gpreagg,
								 CUmodule cuda_module)
{
	GpuPreAggState *gpas = (GpuPreAggState *) gpreagg->task.gts;
	GpuContext	   *gcontext = gpas->gts.gcontext;
	pgstrom_data_store *pds_final = gpas->pds_final;
	pgstrom_data_store *pds_src = gpreagg->pds_src;
	const char	   *kfunc_setup;
	CUfunction		kern_setup;
	CUfunction		kern_reduction;
	CUdeviceptr		m_gpreagg = (CUdeviceptr)&gpreagg->kern;
	CUdeviceptr		m_nullptr = 0UL;
	CUdeviceptr		m_kds_src = 0UL;
	CUdeviceptr		m_kds_extra = 0UL;
	CUdeviceptr		m_kds_slot = 0UL;
	CUdeviceptr		m_kds_final = (CUdeviceptr)&pds_final->kds;
	CUdeviceptr		m_fhash = gpas->m_fhash;
	bool			m_kds_src_release = false;
	cl_int			grid_sz;
	cl_int			block_sz;
	void		   *last_suspend = NULL;
	void		   *kern_args[6];
	void		   *temp;
	CUresult		rc;
	int				retval = 1;

	/*
	 * Ensure the final buffer & hashslot are ready to use
	 */
	gpupreagg_init_final_hash(gpreagg, cuda_module);

	/*
	 * Lookup kernel functions
	 */
	switch (pds_src->kds.format)
	{
		case KDS_FORMAT_ROW:
			kfunc_setup = "kern_gpupreagg_setup_row";
			break;
		case KDS_FORMAT_BLOCK:
			kfunc_setup = "kern_gpupreagg_setup_block";
			break;
		case KDS_FORMAT_ARROW:
			kfunc_setup = "kern_gpupreagg_setup_arrow";
			break;
		case KDS_FORMAT_COLUMN:
			kfunc_setup = "kern_gpupreagg_setup_column";
			break;
		default:
			werror("GpuPreAgg: unknown PDS format: %d", pds_src->kds.format);
	}
	rc = cuModuleGetFunction(&kern_setup,
							 cuda_module,
							 kfunc_setup);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuModuleGetFunction: %s", errorText(rc));

	rc = cuModuleGetFunction(&kern_reduction,
							 cuda_module,
							 gpreagg->kern.num_group_keys == 0
							 ? "kern_gpupreagg_nogroup_reduction"
							 : "kern_gpupreagg_groupby_reduction");
	if (rc != CUDA_SUCCESS)
		werror("failed on cuModuleGetFunction: %s", errorText(rc));

	/*
	 * Device memory allocation for short term
	 */

	/* kds_src */
	if (gpreagg->with_nvme_strom)
	{
		size_t	required = GPUMEMALIGN(pds_src->kds.length);

		Assert(pds_src->kds.format == KDS_FORMAT_BLOCK ||
			   pds_src->kds.format == KDS_FORMAT_ARROW);
		rc = gpuMemAllocIOMap(gcontext,
							  &m_kds_src,
							  required);
		if (rc == CUDA_SUCCESS)
			m_kds_src_release = true;
		else if (rc == CUDA_ERROR_OUT_OF_MEMORY)
		{
			gpreagg->with_nvme_strom = false;
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
				pds_src = PDS_fillup_arrow(gpreagg->pds_src);
				PDS_release(gpreagg->pds_src);
				gpreagg->pds_src = pds_src;
				Assert(!pds_src->iovec);
			}
		}
		else
			werror("failed on gpuMemAllocIOMap: %s", errorText(rc));
	}
	else if (pds_src->kds.format == KDS_FORMAT_COLUMN)
	{
		m_kds_src = pds_src->m_kds_main;
		m_kds_extra = pds_src->m_kds_extra;
	}
	else
	{
		m_kds_src = (CUdeviceptr)&pds_src->kds;
	}

	/* kds_slot */
	rc = gpuMemAllocManaged(gcontext,
							&m_kds_slot,
							gpreagg->kds_slot_length,
							CU_MEM_ATTACH_GLOBAL);
	if (rc == CUDA_ERROR_OUT_OF_MEMORY)
		goto out_of_resource;
	else if (rc != CUDA_SUCCESS)
		werror("failed on gpuMemAllocManaged: %s", errorText(rc));

	memcpy((void *)m_kds_slot, gpas->kds_slot_head,
		   KERN_DATA_STORE_HEAD_LENGTH(gpas->kds_slot_head));
	((kern_data_store *)m_kds_slot)->length = gpreagg->kds_slot_length;
	((kern_data_store *)m_kds_slot)->nrooms = gpreagg->kds_slot_nrooms;

	/*
	 * OK, kick a series of GpuPreAgg invocations
	 */

	/* source data to be reduced */
	if (gpreagg->with_nvme_strom)
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
	else if (pds_src->kds.format != KDS_FORMAT_COLUMN)
	{
		rc = cuMemPrefetchAsync(m_kds_src,
								pds_src->kds.length,
								CU_DEVICE_PER_THREAD,
								CU_STREAM_PER_THREAD);
		if (rc != CUDA_SUCCESS)
			werror("failed on cuMemPrefetchAsync: %s", errorText(rc));
	}

	/*
	 * Launch:
	 * gpupreagg_setup_XXXX(kern_gpupreagg *kgpreagg,
	 *                      kern_data_store *kds_src,
	 *                      kern_data_store *kds_slot)
	 */
	rc = gpuOptimalBlockSize(&grid_sz,
							 &block_sz,
							 kern_setup,
							 CU_DEVICE_PER_THREAD,
							 0, sizeof(int));
	if (rc != CUDA_SUCCESS)
		werror("failed on gpuOptimalBlockSize: %s", errorText(rc));
	gpreagg->kern.grid_sz = grid_sz;
	gpreagg->kern.block_sz = block_sz;
resume_kernel:
	/* make kds_slot empty */
	((kern_data_store *)m_kds_slot)->nitems = 0;
	((kern_data_store *)m_kds_slot)->usage = 0;

	kern_args[0] = &m_gpreagg;
	kern_args[1] = &gpas->gts.kern_params;
	kern_args[2] = &m_kds_src;
	kern_args[3] = &m_kds_extra;
	kern_args[4] = &m_kds_slot;
	rc = cuLaunchKernel(kern_setup,
						gpreagg->kern.grid_sz, 1, 1,
						gpreagg->kern.block_sz, 1, 1,
						sizeof(cl_int) * 1024,	/* for StairlikeSum */
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuLaunchKernel: %s", errorText(rc));

	/*
	 * Launch:
	 * KERNEL_FUNCTION(void)
	 * gpupreagg_XXXX_reduction(kern_gpupreagg *kgpreagg,
	 *                          kern_parambuf *kparams,
	 *                          kern_errorbuf *kgjoin_errorbuf,
	 *                          kern_data_store *kds_slot,
	 *                          kern_data_store *kds_final,
	 *                          kern_global_hashslot *f_hash)
	 */
	rc = gpuOptimalBlockSize(&grid_sz,
							 &block_sz,
							 kern_reduction,
							 CU_DEVICE_PER_THREAD,
							 0, sizeof(int));
	if (rc != CUDA_SUCCESS)
		werror("failed on gpuOptimalBlockSize: %s", errorText(rc));
	kern_args[0] = &m_gpreagg;
	kern_args[1] = &gpas->gts.kern_params;
	kern_args[2] = &m_nullptr;
	kern_args[3] = &m_kds_slot;
	kern_args[4] = &m_kds_final;
	kern_args[5] = &m_fhash;
	rc = cuLaunchKernel(kern_reduction,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						0,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuLaunchKernel: %s", errorText(rc));

	rc = cuEventRecord(CU_EVENT_PER_THREAD, CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuEventRecord: %s", errorText(rc));

	/* Point of synchronization */
	rc = cuEventSynchronize(CU_EVENT_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuEventSynchronize: %s", errorText(rc));

	/*
	 * XXX - Even though we speculatively allocate large virtual device
	 * memory, GPU kernel may raise NoDataSpace error.
	 * In this case, we need working device memory.
	 * For pds_final, all we need to do is pre-allocation of very large
	 * virtual device memory with 64bit offset support.
	 * In case of final hash table, we have to re-organize hash table
	 * again, because hash table must be initialized to (0,-1), thus, it
	 * leads page-fault and physical memory allocation. We cannot pre-allocate
	 * too large memory.
	 */

	/*
     * Clear the error code if CPU fallback case.
     * Elsewhere, update run-time statistics.
     */
	memcpy(&gpreagg->task.kerror,
		   &gpreagg->kern.kerror, sizeof(kern_errorbuf));
	if (gpreagg->task.kerror.errcode == ERRCODE_STROM_SUCCESS)
	{
		if (gpreagg->kern.suspend_count > 0)
		{
			CHECK_WORKER_TERMINATION();
			gpupreagg_reset_kernel_task(&gpreagg->kern, true);

			Assert(gpreagg->kern.suspend_size > 0);
			if (!last_suspend)
				last_suspend = alloca(gpreagg->kern.suspend_size);
			temp = KERN_GPUPREAGG_SUSPEND_CONTEXT(&gpreagg->kern, 0);
			memcpy(last_suspend, temp, gpreagg->kern.suspend_size);
			goto resume_kernel;
		}
		gpupreaggUpdateRunTimeStat(gpreagg->task.gts, &gpreagg->kern);
		retval = -1;
	}
	else if (pgstrom_cpu_fallback_enabled &&
			 (gpreagg->task.kerror.errcode & ERRCODE_FLAGS_CPU_FALLBACK) != 0 &&
			 !gpreagg->kern.final_buffer_modified)
	{
		/*
		 * As long as final buffer is not modified by the reduction process
		 * yet, we can help this GpuTask by CPU fallback.
		 * If CpuReCheck is reported by gpupreagg_setup_xxxx(), kds_slot is
		 * not built yet. So, CPU fallback routine has to refer the kds_src.
		 * Elsewhere, we can reuse kds_slot built by the kernel.
		 */
		memset(&gpreagg->task.kerror, 0, sizeof(kern_errorbuf));
		gpreagg->task.cpu_fallback = true;

		if (!gpreagg->kern.setup_slot_done)
		{
			/*
			 * gpupreagg_setup_xxxx reported CpuReCheck error.
			 * So, here is no partial results, and CPU fallback routine
			 * can generate alternative rows from pds_src.
			 */
			if (pds_src->kds.format == KDS_FORMAT_BLOCK &&
				pds_src->nblocks_uncached > 0)
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
				gpreagg->pds_src = PDS_writeback_arrow(pds_src, m_kds_src);
			}
			else if (pds_src->kds.format == KDS_FORMAT_COLUMN)
			{
				//TODO: memcopy D->H
			}
			/* restore the point where suspended most recently */
			gpreagg->kern.resume_context = (last_suspend != NULL);
			if (last_suspend)
			{
				temp = KERN_GPUPREAGG_SUSPEND_CONTEXT(&gpreagg->kern, 0);
				memcpy(temp, last_suspend, gpreagg->kern.suspend_size);
			}
		}
		else
		{
			/*
			 * gpupreagg_setup_xxxx successfully setup kds_slot, however,
			 * reduction kernel reported CpuReCheck error, fortunatelly,
			 * prior to any modification of the kds_final buffer.
			 * So, CPU fallback routine can use the kds_slot, as-is.
			 */
			rc = cuMemPrefetchAsync(m_kds_slot,
									gpreagg->kds_slot_length,
									CU_DEVICE_CPU,
									CU_STREAM_PER_THREAD);
			if (rc != CUDA_SUCCESS)
				werror("failed on cuMemPrefetchAsync: %s", errorText(rc));
			gpreagg->kds_slot = (kern_data_store *) m_kds_slot;
			m_kds_slot = 0UL;
		}
		retval = 0;
	}
	else
	{
		/* raise an error */
		gpreagg->task.kerror.errcode &= ~ERRCODE_FLAGS_CPU_FALLBACK;
		retval = 0;
	}
out_of_resource:
	if (m_kds_src_release)
		gpuMemFree(gcontext, m_kds_src);
	if (m_kds_slot != 0UL)
		gpuMemFree(gcontext, m_kds_slot);
	return retval;
}

/*
 * gpupreagg_process_combined_task
 *
 * It runs a combined GpuJoin+GpuPreAgg task.
 */
static int
gpupreagg_process_combined_task(GpuPreAggTask *gpreagg, CUmodule cuda_module)
{
	GpuPreAggState *gpas = (GpuPreAggState *) gpreagg->task.gts;
	GpuContext	   *gcontext = gpas->gts.gcontext;
	pgstrom_data_store *pds_final = gpas->pds_final;
	pgstrom_data_store *pds_src = gpreagg->pds_src;
	kern_gpujoin   *kgjoin = gpreagg->kgjoin;
	CUfunction		kern_gpujoin_main;
	CUfunction		kern_gpupreagg_reduction;
	CUdeviceptr		m_gpreagg = (CUdeviceptr)&gpreagg->kern;
	CUdeviceptr		m_kgjoin = (CUdeviceptr)kgjoin;
	CUdeviceptr		m_kparams = gpreagg->m_kparams;
	CUdeviceptr		m_kmrels = gpreagg->m_kmrels;
	CUdeviceptr		m_kds_src = 0UL;
	CUdeviceptr		m_kds_extra = 0UL;
	CUdeviceptr		m_kds_slot = 0UL;
	CUdeviceptr		m_kds_final = (CUdeviceptr)&pds_final->kds;
	CUdeviceptr		m_fhash = gpas->m_fhash;
	CUresult		rc;
	bool			m_kds_src_release = false;
	cl_int			grid_sz;
	cl_int			block_sz;
	void		   *kern_args[10];
	void		   *last_suspend = NULL;
	void		   *temp;
	int				retval = 1;

	/*
	 * Ensure the final buffer & hashslot are ready to use
	 */
	gpupreagg_init_final_hash(gpreagg, cuda_module);

	/*
	 * Lookup kernel functions
	 *
	 * XXX - needs to kick RIGHT OUTER JOIN
	 */
	rc = cuModuleGetFunction(&kern_gpujoin_main,
							 cuda_module,
							 pds_src != NULL
							 ? "kern_gpujoin_main"
							 : "kern_gpujoin_right_outer");
	if (rc != CUDA_SUCCESS)
		werror("failed on cuModuleGetFunction: %s", errorText(rc));

	rc = cuModuleGetFunction(&kern_gpupreagg_reduction,
							 cuda_module,
							 gpreagg->kern.num_group_keys == 0
							 ? "kern_gpupreagg_nogroup_reduction"
							 : "kern_gpupreagg_groupby_reduction");
	if (rc != CUDA_SUCCESS)
		werror("failed on cuModuleGetFunction: %s", errorText(rc));

	/* allocation of kds_src */
	if (!pds_src)
		m_kds_src = 0UL;
	else if (gpreagg->with_nvme_strom)
	{
		size_t	required = GPUMEMALIGN(pds_src->kds.length);

		Assert(pds_src->kds.format == KDS_FORMAT_BLOCK ||
			   pds_src->kds.format == KDS_FORMAT_ARROW);
		rc = gpuMemAllocIOMap(gcontext,
							  &m_kds_src,
							  required);
		if (rc == CUDA_SUCCESS)
			m_kds_src_release = true;
		else if (rc == CUDA_ERROR_OUT_OF_MEMORY)
		{
			gpreagg->with_nvme_strom = false;
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
				pds_src = PDS_fillup_arrow(gpreagg->pds_src);
				PDS_release(gpreagg->pds_src);
				gpreagg->pds_src = pds_src;
				Assert(!pds_src->iovec);
			}
		}
		else
			werror("failed on gpuMemAllocIOMap: %s", errorText(rc));
	}
	else if (pds_src->kds.format == KDS_FORMAT_COLUMN)
	{
		m_kds_src = pds_src->m_kds_main;
		m_kds_extra = pds_src->m_kds_extra;
	}
	else
	{
		m_kds_src = (CUdeviceptr)&pds_src->kds;
	}

	/*
	 * allocation of kds_slot
	 */
	rc = gpuMemAllocManaged(gcontext,
							&m_kds_slot,
							gpreagg->kds_slot_length,
							CU_MEM_ATTACH_GLOBAL);
	if (rc == CUDA_ERROR_OUT_OF_MEMORY)
		goto out_of_resource;
	else if (rc != CUDA_SUCCESS)
		werror("failed on gpuMemAllocManaged: %s", errorText(rc));

	memcpy((kern_data_store *)m_kds_slot, gpas->kds_slot_head,
		   KERN_DATA_STORE_HEAD_LENGTH(gpas->kds_slot_head));
	((kern_data_store *)m_kds_slot)->length = gpreagg->kds_slot_length;
	((kern_data_store *)m_kds_slot)->nrooms = gpreagg->kds_slot_nrooms;

	/*
	 * OK, kick a series of GpuPreAgg invocations
	 */
	if (pds_src)
	{
		if (gpreagg->with_nvme_strom)
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
				werror("failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		}
		else if (pds_src->kds.format != KDS_FORMAT_COLUMN)
		{
			rc = cuMemPrefetchAsync(m_kds_src,
									pds_src->kds.length,
									CU_DEVICE_PER_THREAD,
									CU_STREAM_PER_THREAD);
			if (rc != CUDA_SUCCESS)
				werror("failed on cuMemPrefetchAsync: %s", errorText(rc));
		}
	}
resume_kernel:
	/* make kds_slot empty again */
	((kern_data_store *)m_kds_slot)->nitems = 0;
	((kern_data_store *)m_kds_slot)->usage = 0;

	/*
	 * Launch:
	 * KERNEL_FUNCTION(void)
	 * gpujoin_main(kern_gpujoin *kgjoin,
	 *              kern_parambuf *kparams,
	 *              kern_multirels *kmrels,
	 *              kern_data_store *kds_src,
	 *              kern_data_extra *kds_extra,
	 *              kern_data_store *kds_slot,
	 *              kern_parambuf *kparams_gpreagg)
	 * OR
	 *
	 * KERNEL_FUNCTION(void)
	 * gpujoin_right_outer(kern_gpujoin *kgjoin,
	 *                     kern_parambuf *kparams,
	 *                     kern_multirels *kmrels,
	 *                     cl_int outer_depth,
	 *                     kern_data_store *kds_dst,
	 *                     kern_parambuf *kparams_gpreagg)
	 */
	rc = gpuOptimalBlockSize(&grid_sz,
							 &block_sz,
							 kern_gpujoin_main,
							 CU_DEVICE_PER_THREAD,
							 0, sizeof(int));
	if (rc != CUDA_SUCCESS)
		werror("failed on gpuOptimalBlockSize: %s", errorText(rc));
	gpreagg->kern.grid_sz = grid_sz;
	gpreagg->kern.block_sz = block_sz;

	kern_args[0] = &m_kgjoin;
	kern_args[1] = &m_kparams;
	kern_args[2] = &m_kmrels;
	if (pds_src != NULL)
	{
		kern_args[3] = &m_kds_src;
		kern_args[4] = &m_kds_extra;
		kern_args[5] = &m_kds_slot;
		kern_args[6] = &gpas->gts.kern_params;
	}
	else
	{
		kern_args[3] = &gpreagg->outer_depth;
		kern_args[4] = &m_kds_slot;
		kern_args[5] = &gpas->gts.kern_params;
	}
	rc = cuLaunchKernel(kern_gpujoin_main,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						sizeof(cl_int) * block_sz,
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuLaunchKernel: %s", errorText(rc));

	/*
	 * Launch:
	 * KERNEL_FUNCTION(void)
	 * gpupreagg_XXXX_reduction(kern_gpupreagg *kgpreagg,
	 *                          kern_parambuf *kparams,
	 *                          kern_errorbuf *kgjoin_errorbuf,
	 *                          kern_data_store *kds_slot,
	 *                          kern_data_store *kds_final,
	 *                          kern_global_hashslot *f_hash)
	 */
	rc = gpuOptimalBlockSize(&grid_sz,
							 &block_sz,
							 kern_gpupreagg_reduction,
							 CU_DEVICE_PER_THREAD,
							 0, sizeof(int));
	if (rc != CUDA_SUCCESS)
		werror("failed on gpuOptimalBlockSize: %s", errorText(rc));

	kern_args[0] = &m_gpreagg;
	kern_args[1] = &gpas->gts.kern_params;
	kern_args[2] = &m_kgjoin;
	kern_args[3] = &m_kds_slot;
	kern_args[4] = &m_kds_final;
	kern_args[5] = &m_fhash;
	rc = cuLaunchKernel(kern_gpupreagg_reduction,
						grid_sz, 1, 1,
						block_sz, 1, 1,
						sizeof(cl_int) * block_sz,	/* for StairlikeSum */
						CU_STREAM_PER_THREAD,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuLaunchKernel: %s", errorText(rc));

	rc = cuEventRecord(CU_EVENT_PER_THREAD, CU_STREAM_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuEventRecord: %s", errorText(rc));

	/* Point of synchronization */
	rc = cuEventSynchronize(CU_EVENT_PER_THREAD);
	if (rc != CUDA_SUCCESS)
		werror("failed on cuEventSynchronize: %s", errorText(rc));

	if (kgjoin->kerror.errcode != ERRCODE_STROM_SUCCESS)
	{
		if (pgstrom_cpu_fallback_enabled &&
			(kgjoin->kerror.errcode & ERRCODE_FLAGS_CPU_FALLBACK) != 0)
		{
			/*
			 * CPU fallback without partial results - If GpuJoin reported
			 * CpuReCheck error, obviously, it didn't touch the final buffer.
			 * So, CPU fallback routine will generate results based on the
			 * source buffer.
			 */
			memset(&kgjoin->kerror, 0, sizeof(kern_errorbuf));
			gpreagg->task.cpu_fallback = true;

			if (pds_src)
			{
				if (pds_src->kds.format == KDS_FORMAT_BLOCK &&
					pds_src->nblocks_uncached > 0)
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
					gpreagg->pds_src = PDS_writeback_arrow(pds_src, m_kds_src);
				}
				else if (pds_src->kds.format == KDS_FORMAT_COLUMN)
				{
					//TODO: cuMemCopyDtoH
				}
			}
			/* restore the suspend context if any */
			kgjoin->resume_context = (last_suspend != NULL);
			if (last_suspend)
			{
				memcpy(KERN_GPUJOIN_SUSPEND_CONTEXT(kgjoin, 0),
					   last_suspend,
					   kgjoin->suspend_size);
			}
		}
		memcpy(&gpreagg->task.kerror,
			   &kgjoin->kerror, sizeof(kern_errorbuf));
		retval = 0;
	}
	else if (gpreagg->kern.kerror.errcode != ERRCODE_STROM_SUCCESS)
	{
		if (pgstrom_cpu_fallback_enabled &&
			(gpreagg->kern.kerror.errcode & ERRCODE_FLAGS_CPU_FALLBACK) != 0 &&
			!gpreagg->kern.final_buffer_modified)
		{
			/*
			 * CPU fallback by GpuPreAgg kernel
			 *
			 * If GpuPreAgg reported CpuReCheck error, it means kds_slot
			 * is successfully built, however, CPU fallback is required
			 * during the reduction process prior to modification of the
			 * final buffer.
			 */
			memset(&gpreagg->kern.kerror, 0, sizeof(kern_errorbuf));

			Assert(gpreagg->kern.suspend_count == 0);
			if (kgjoin->suspend_count > 0)
			{
				kern_data_store	   *kds_slot
					= (kern_data_store *) m_kds_slot;

				CHECK_WORKER_TERMINATION();
				m_kds_slot = (CUdeviceptr) KDS_clone(gcontext, kds_slot);
				gpupreagg_throw_partial_result(gpreagg, kds_slot);

				/* save the suspend status at this point, then resume */
				gpujoin_reset_kernel_task(kgjoin, true);
				gpupreagg_reset_kernel_task(&gpreagg->kern, false);

				if (!last_suspend)
					last_suspend = alloca(kgjoin->suspend_size);
				temp = KERN_GPUJOIN_SUSPEND_CONTEXT(kgjoin, 0);
				memcpy(last_suspend, temp, kgjoin->suspend_size);
				goto resume_kernel;
			}
			else
			{
				rc = cuMemPrefetchAsync(m_kds_slot,
										gpreagg->kds_slot_length,
										CU_DEVICE_CPU,
										CU_STREAM_PER_THREAD);
				if (rc != CUDA_SUCCESS)
					werror("failed on cuMemPrefetchAsync: %s", errorText(rc));
				gpreagg->task.cpu_fallback = true;
				gpreagg->kds_slot = (kern_data_store *) m_kds_slot;
				m_kds_slot = 0UL;
				retval = 0;
			}
		}
		memcpy(&gpreagg->task.kerror,
			   &gpreagg->kern.kerror, sizeof(kern_errorbuf));
	}
	else
	{
		GpuTaskState   *gjs = (GpuTaskState *)
			outerPlanState(gpreagg->task.gts);
		/*
		 * GpuPreAgg-side has no code path that can cause suspend/resume.
		 * Only GpuJoin-side can break GPU kernel execution.
		 */
		Assert(gpreagg->kern.suspend_count == 0);
		if (kgjoin->suspend_count > 0)
		{
			CHECK_WORKER_TERMINATION();
			gpujoin_reset_kernel_task(kgjoin, true);
			gpupreagg_reset_kernel_task(&gpreagg->kern, false);
			if (!last_suspend)
				last_suspend = alloca(kgjoin->suspend_size);
			memcpy(last_suspend,
				   KERN_GPUJOIN_SUSPEND_CONTEXT(kgjoin, 0),
				   kgjoin->suspend_size);
			goto resume_kernel;
		}
		gpujoinUpdateRunTimeStat(gjs, gpreagg->kgjoin);
        gpupreaggUpdateRunTimeStat(gpreagg->task.gts, &gpreagg->kern);
		retval = -1;
	}
out_of_resource:
	if (m_kds_src_release)
		gpuMemFree(gcontext, m_kds_src);
	if (m_kds_slot)
		gpuMemFree(gcontext, m_kds_slot);
	return retval;
}

/*
 * gpupreagg_process_task
 */
static int
gpupreagg_process_task(GpuTask *gtask, CUmodule cuda_module)
{
	GpuPreAggTask  *gpreagg = (GpuPreAggTask *) gtask;
	pgstrom_data_store *pds_src = gpreagg->pds_src;
	volatile bool	gcache_mapped = false;
	int				retval;
	CUresult		rc;

	STROM_TRY();
	{
		/*
		 * NOTE: In case of combined RIGHT/FULL OUTER JOIN, the last GpuTask
		 * shall not have a valid pds_src; Its GPU kernel generates all-NULL
		 * rows according to the outer-join map.
		 * So, we cannot expect GpuPreAgg always have a valid pds_src also.
		 */
		if (pds_src && pds_src->kds.format == KDS_FORMAT_COLUMN)
		{
			rc = gpuCacheMapDeviceMemory(GpuWorkerCurrentContext, pds_src);
			if (rc != CUDA_SUCCESS)
				werror("failed on gpuCacheMapDeviceMemory: %s", errorText(rc));
			gcache_mapped = true;
		}
		if (!gpreagg->kgjoin)
			retval = gpupreagg_process_reduction_task(gpreagg, cuda_module);
		else
			retval = gpupreagg_process_combined_task(gpreagg, cuda_module);
	}
	STROM_CATCH();
	{
		if (gcache_mapped)
			gpuCacheUnmapDeviceMemory(GpuWorkerCurrentContext, pds_src);
		STROM_RE_THROW();
	}
	STROM_END_TRY();
	if (gcache_mapped)
		gpuCacheUnmapDeviceMemory(GpuWorkerCurrentContext, pds_src);
	return retval;
}

/*
 * gpupreagg_release_task
 */
static void
gpupreagg_release_task(GpuTask *gtask)
{
	GpuPreAggTask  *gpreagg = (GpuPreAggTask *)gtask;
	GpuContext	   *gcontext = gtask->gts->gcontext;

	if (gpreagg->pds_src)
		PDS_release(gpreagg->pds_src);
	if (gpreagg->kds_slot)
		gpuMemFree(gcontext, (CUdeviceptr)gpreagg->kds_slot);
	gpuMemFree(gcontext, (CUdeviceptr)gpreagg);
}

/*
 * entrypoint of GpuPreAgg
 */
void
pgstrom_init_gpupreagg(void)
{
	/* enable_gpupreagg parameter */
	DefineCustomBoolVariable("pg_strom.enable_gpupreagg",
							 "Enables the use of GPU preprocessed aggregate",
							 NULL,
							 &enable_gpupreagg,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* pg_strom.pullup_outer_join */
	DefineCustomBoolVariable("pg_strom.pullup_outer_join",
							 "Enables to pull up GpuJoin under GpuPreAgg",
							 NULL,
							 &enable_pullup_outer_join,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* pg_strom.enable_partitionwise_gpupreagg */
	DefineCustomBoolVariable("pg_strom.enable_partitionwise_gpupreagg",
							 "(EXPERIMENTAL) Enables partition wise GpuPreAgg",
							 NULL,
							 &enable_partitionwise_gpupreagg,
							 true,
							 PGC_USERSET,
                             GUC_NOT_IN_SAMPLE,
                             NULL, NULL, NULL);
	/* pg_strom.enable_numeric_aggfuncs */
	DefineCustomBoolVariable("pg_strom.enable_numeric_aggfuncs",
							 "Enables aggregate functions on numeric type",
							 NULL,
							 &enable_numeric_aggfuncs,
							 true,
							 PGC_USERSET,
							 GUC_NO_SHOW_ALL | GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* pg_strom.hll_registers_bits */
	DefineCustomIntVariable("pg_strom.hll_registers_bits",
							"Accuracy of HyperLogLog COUNT(distinct ...) estimation",
							NULL,
							&pgstrom_hll_register_bits,
							9,
							4,
							15,
							PGC_USERSET,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);
	/* pg_strom.gpupreagg_reduction_threshold */
	DefineCustomRealVariable("pg_strom.gpupreagg_reduction_threshold",
							 "Minimus reduction ratio to use GpuPreAgg",
							 NULL,
							 &gpupreagg_reduction_threshold,
							 20.0,
							 1.0,
							 DBL_MAX,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* initialization of path method table */
	memset(&gpupreagg_path_methods, 0, sizeof(CustomPathMethods));
	gpupreagg_path_methods.CustomName          = "GpuPreAgg";
	gpupreagg_path_methods.PlanCustomPath      = PlanGpuPreAggPath;

	/* initialization of plan method table */
	memset(&gpupreagg_scan_methods, 0, sizeof(CustomScanMethods));
	gpupreagg_scan_methods.CustomName          = "GpuPreAgg";
	gpupreagg_scan_methods.CreateCustomScanState
		= CreateGpuPreAggScanState;
	RegisterCustomScanMethods(&gpupreagg_scan_methods);

	/* initialization of exec method table */
	memset(&gpupreagg_exec_methods, 0, sizeof(CustomExecMethods));
	gpupreagg_exec_methods.CustomName          = "GpuPreAgg";
   	gpupreagg_exec_methods.BeginCustomScan     = ExecInitGpuPreAgg;
	gpupreagg_exec_methods.ExecCustomScan      = ExecGpuPreAgg;
	gpupreagg_exec_methods.EndCustomScan       = ExecEndGpuPreAgg;
	gpupreagg_exec_methods.ReScanCustomScan    = ExecReScanGpuPreAgg;
	gpupreagg_exec_methods.EstimateDSMCustomScan = ExecGpuPreAggEstimateDSM;
    gpupreagg_exec_methods.InitializeDSMCustomScan = ExecGpuPreAggInitDSM;
    gpupreagg_exec_methods.InitializeWorkerCustomScan = ExecGpuPreAggInitWorker;
	gpupreagg_exec_methods.ReInitializeDSMCustomScan = ExecGpuPreAggReInitializeDSM;
	gpupreagg_exec_methods.ShutdownCustomScan  = ExecShutdownGpuPreAgg;
	gpupreagg_exec_methods.ExplainCustomScan   = ExplainGpuPreAgg;
	/* hook registration */
	create_upper_paths_next = create_upper_paths_hook;
	create_upper_paths_hook = gpupreagg_add_grouping_paths;
}
