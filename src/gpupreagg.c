/*
 * gpupreagg.c
 *
 * Aggregate Pre-processing with GPU acceleration
 * ----
 * Copyright 2011-2016 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2016 (C) The PG-Strom Development Team
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
#include "access/xact.h"
#include "catalog/pg_cast.h"
#include "catalog/pg_namespace.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_type.h"
#include "executor/nodeAgg.h"
#include "executor/nodeCustom.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "parser/parse_func.h"
#include "optimizer/clauses.h"
#include "optimizer/cost.h"
#include "optimizer/planner.h"
#include "optimizer/var.h"
#include "parser/parsetree.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/pg_crc.h"
#include "utils/rel.h"
#include "utils/ruleutils.h"
#include "utils/syscache.h"
#include <math.h>
#include "pg_strom.h"
#include "cuda_common.h"
#include "cuda_numeric.h"
#include "cuda_gpupreagg.h"

static CustomScanMethods		gpupreagg_scan_methods;
static CustomExecMethods		gpupreagg_exec_methods;
static bool						enable_gpupreagg;
static bool						debug_force_gpupreagg;

#if 0
/* list of reduction mode */
#define GPUPREAGG_NOGROUP_REDUCTION		1
#define GPUPREAGG_LOCAL_REDUCTION		2
#define GPUPREAGG_GLOBAL_REDUCTION		3
#define GPUPREAGG_FINAL_REDUCTION		4
#define GPUPREAGG_ONLY_TERMINATION		99	/* used to urgent termination */
#endif

typedef struct
{
	List		   *tlist_dev;		/* requirement of device projection */
	int				numCols;		/* number of grouping columns */
	AttrNumber	   *grpColIdx;		/* their indexes in the target list */
	double			num_groups;		/* estimated number of groups */
	cl_int			num_chunks;		/* estimated number of chunks */
	Size			varlena_unitsz;	/* estimated unit size of varlena */
	cl_int			safety_limit;	/* reasonable limit for reduction */
	cl_int			key_dist_salt;	/* salt, if more distribution needed */
	double			outer_nitems;	/* number of expected outer input */
	List		   *outer_quals;	/* device executable quals of outer-scan */
	const char	   *kern_source;
	int				extra_flags;
	List		   *used_params;	/* referenced Const/Param */
} GpuPreAggInfo;

static inline void
form_gpupreagg_info(CustomScan *cscan, GpuPreAggInfo *gpa_info)
{
	List	   *privs = NIL;
	List	   *exprs = NIL;
	List	   *temp;
	int			i;

	exprs = lappend(exprs, gpa_info->tlist_dev);
	/* numCols and grpColIdx */
	temp = NIL;
	for (i = 0; i < gpa_info->numCols; i++)
		temp = lappend_int(temp, gpa_info->grpColIdx[i]);
	privs = lappend(privs, temp);
	privs = lappend(privs, makeInteger(double_as_long(gpa_info->num_groups)));
	privs = lappend(privs, makeInteger(gpa_info->num_chunks));
	privs = lappend(privs, makeInteger(gpa_info->varlena_unitsz));
	privs = lappend(privs, makeInteger(gpa_info->safety_limit));
	privs = lappend(privs, makeInteger(gpa_info->key_dist_salt));
	privs = lappend(privs,
					makeInteger(double_as_long(gpa_info->outer_nitems)));
	exprs = lappend(exprs, gpa_info->outer_quals);
	privs = lappend(privs, makeString(pstrdup(gpa_info->kern_source)));
	privs = lappend(privs, makeInteger(gpa_info->extra_flags));
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
	int			i = 0;
	List	   *temp;
	ListCell   *cell;

	gpa_info->tlist_dev = list_nth(exprs, eindex++);
	/* numCols and grpColIdx */
	temp = list_nth(privs, pindex++);
	gpa_info->numCols = list_length(temp);
	gpa_info->grpColIdx = palloc0(sizeof(AttrNumber) * gpa_info->numCols);
	foreach (cell, temp)
		gpa_info->grpColIdx[i++] = lfirst_int(cell);

	gpa_info->num_groups =
		long_as_double(intVal(list_nth(privs, pindex++)));
	gpa_info->num_chunks = intVal(list_nth(privs, pindex++));
	gpa_info->varlena_unitsz = intVal(list_nth(privs, pindex++));
	gpa_info->safety_limit = intVal(list_nth(privs, pindex++));
	gpa_info->key_dist_salt = intVal(list_nth(privs, pindex++));
	gpa_info->outer_nitems =
		long_as_double(intVal(list_nth(privs, pindex++)));
	gpa_info->outer_quals = list_nth(exprs, eindex++);
	gpa_info->kern_source = strVal(list_nth(privs, pindex++));
	gpa_info->extra_flags = intVal(list_nth(privs, pindex++));
	gpa_info->used_params = list_nth(exprs, eindex++);

	return gpa_info;
}

typedef struct
{
	GpuTaskState	gts;
	ProjectionInfo *bulk_proj;
	TupleTableSlot *bulk_slot;
	cl_int			safety_limit;
	cl_int			key_dist_salt;
	cl_int			reduction_mode;
	List		   *outer_quals;
	TupleTableSlot *outer_overflow;
	pgstrom_data_store *outer_pds;

	/*
	 * segment is a set of chunks to be aggregated
	 */
	struct gpupreagg_segment *curr_segment; 

	/*
	 * Run-time statistics; recorded per segment
	 */
	cl_uint			stat_num_segments;	/* # of total segments */
	double			stat_num_groups;	/* # of groups in plan/exec avg */
	double			stat_num_chunks;	/* # of chunks in plan/exec avg */
	double			stat_src_nitems;	/* # of source rows in plan/exec avg */
	double			stat_varlena_unitsz;/* unitsz of varlena buffer in plan */
} GpuPreAggState;

/*
 * gpupreagg_segment
 *
 * It groups a bunch of chunks that shares the final result buffer.
 * The finalizer task has to synchronize completion of other kernels.
 *
 * NOTE: CUDA event mechanism ensures all the concurrent kernel shall be
 * done prior to the device-to-host DMA onto final result buffer, however,
 * here is no guarantee completion hook of the finalizer is called on the
 * tail. So, we have to treat gpupreagg_segment carefully. The finalizer
 * task is responsible to back pds_final buffer to the state machine,
 * however, a series of buffer release shall be executed by the last task
 * who references this segment.
 */
typedef struct gpupreagg_segment
{
	GpuPreAggState *gpas;				/* reference to GpuPreAggState */
	pgstrom_data_store *pds_final;		/* final pds/kds buffer on host */
	CUdeviceptr		m_hashslot_final;	/* final reduction hash table */
	CUdeviceptr		m_kds_final;		/* final kds buffer on device */
	size_t			f_hashsize;			/* size of final reduction hashtable */
	cl_int			num_chunks;			/* # of chunks in this segment */
	cl_int			idx_chunks;			/* index of the chunk array */
	cl_int			refcnt;				/* referenced by pgstrom_gpupreagg */
	cl_int			cuda_index;			/* device to be used */
	cl_bool			has_terminator;		/* true, if any terminator task */
	cl_bool			needs_fallback;		/* true, if CPU fallback needed */
	pgstrom_data_store **pds_src;		/* reference to source PDSs */
	CUevent			ev_final_loaded;	/* event of final-buffer load */
	CUevent		   *ev_kern_main;		/* event of gpupreagg_main end */
	/* run-time statistics */
	size_t			allocated_nrooms;	/* nrooms of the kds_final */
	size_t			allocated_varlena;	/* length of the varlena buffer */
	size_t			total_ntasks;		/* # of tasks already processed */
	size_t			total_nitems;		/* # of items already consumed */
	size_t			total_ngroups;		/* # of groups already generated */
	size_t			total_varlena;		/* total usage of varlena buffer */
	size_t			delta_ngroups;		/* delta of the last task */
} gpupreagg_segment;

/*
 * pgstrom_gpupreagg
 *
 * Host side representation of kern_gpupreagg. It can perform as a message
 * object of PG-Strom, has key of OpenCL device program, a source row/column
 * store and a destination kern_data_store.
 */
typedef struct
{
	GpuTask			task;
	gpupreagg_segment *segment;		/* reference to the preagg segment */
	cl_int			segment_id;		/* my index within segment */
	bool			is_terminator;	/* If true, collector of final result */
	double			num_groups;		/* estimated number of groups */
	CUfunction		kern_main;
	CUfunction		kern_fixvar;
	CUdeviceptr		m_gpreagg;
	CUdeviceptr		m_kds_row;		/* source row-buffer */
	CUdeviceptr		m_kds_slot;		/* internal slot-buffer */
	CUdeviceptr		m_ghash;		/* global hash slot */
	CUevent			ev_dma_send_start;
	CUevent			ev_dma_send_stop;
	CUevent			ev_kern_fixvar;
	CUevent			ev_dma_recv_start;
	CUevent			ev_dma_recv_stop;
	pgstrom_data_store *pds_in;		/* source data-store */
	kern_data_store	   *kds_head;	/* header of intermediation data store */
	kern_resultbuf *kresults;
	kern_gpupreagg	kern;
} pgstrom_gpupreagg;

/* declaration of static functions */
static bool		gpupreagg_task_process(GpuTask *gtask);
static bool		gpupreagg_task_complete(GpuTask *gtask);
static void		gpupreagg_task_release(GpuTask *gtask);
static GpuTask *gpupreagg_next_chunk(GpuTaskState *gts);
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

/*
 * XXX - GpuPreAgg with Numeric arguments are problematic because
 * it is implemented with normal function call and iteration of
 * cmpxchg. Thus, larger reduction ratio (usually works better)
 * will increase atomic contension. So, at this moment we turned
 * off GpuPreAgg + Numeric
 */
#define GPUPREAGG_SUPPORT_NUMERIC			1

/*
 * List of supported aggregate functions
 */
typedef struct {
	/* aggregate function can be preprocessed */
	const char *aggfn_name;
	int			aggfn_nargs;
	Oid			aggfn_argtypes[4];
	/* alternative function to generate same result.
	 * prefix indicates the schema that stores the alternative functions
	 * c: pg_catalog ... the system default
	 * s: pgstrom    ... PG-Strom's special ones
	 */
	const char *altfn_name;
	int			altfn_nargs;
	Oid			altfn_argtypes[8];
	int			altfn_argexprs[8];
	int			extra_flags;
	int			safety_limit;
} aggfunc_catalog_t;
static aggfunc_catalog_t  aggfunc_catalog[] = {
	/* AVG(X) = EX_AVG(NROWS(), PSUM(X)) */
	{ "avg",    1, {INT2OID},
	  "s:avg",  2, {INT4OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "avg",    1, {INT4OID},
	  "s:avg",  2, {INT4OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "avg",    1, {INT8OID},
	  "s:avg_int8",  2, {INT4OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "avg",    1, {FLOAT4OID},
	  "s:avg",  2, {INT4OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "avg",    1, {FLOAT8OID},
	  "s:avg",  2, {INT4OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "avg",	1, {NUMERICOID},
	  "s:avg_numeric",	2, {INT4OID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  DEVKERNEL_NEEDS_NUMERIC, 100
	},
#endif
	/* COUNT(*) = SUM(NROWS(*|X)) */
	{ "count", 0, {},
	  "s:count", 1, {INT4OID},
	  {ALTFUNC_EXPR_NROWS}, 0, INT_MAX
	},
	{ "count", 1, {ANYOID},
	  "s:count", 1, {INT4OID},
	  {ALTFUNC_EXPR_NROWS}, 0, INT_MAX
	},
	/* MAX(X) = MAX(PMAX(X)) */
	{ "max", 1, {INT2OID},
	  "c:max", 1, {INT2OID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max", 1, {INT4OID},
	  "c:max", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max", 1, {INT8OID},
	  "c:max", 1, {INT8OID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max", 1, {FLOAT4OID},
	  "c:max", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max", 1, {FLOAT8OID},
	  "c:max", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "max", 1, {NUMERICOID},
	  "c:max", 1, {NUMERICOID},
	  {ALTFUNC_EXPR_PMAX}, DEVKERNEL_NEEDS_NUMERIC, INT_MAX
	},
#endif
	{ "max", 1, {CASHOID},
	  "c:max", 1, {CASHOID},
	  {ALTFUNC_EXPR_PMAX}, DEVKERNEL_NEEDS_MONEY, INT_MAX
	},
	{ "max", 1, {DATEOID},
	  "c:max", 1, {DATEOID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max", 1, {TIMEOID},
	  "c:max", 1, {TIMEOID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max", 1, {TIMESTAMPOID},
	  "c:max", 1, {TIMESTAMPOID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max", 1, {TIMESTAMPTZOID},
	  "c:max", 1, {TIMESTAMPTZOID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},

	/* MIX(X) = MIN(PMIN(X)) */
	{ "min", 1, {INT2OID},
	  "c:min", 1, {INT2OID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min", 1, {INT4OID},
	  "c:min", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min", 1, {INT8OID},
	  "c:min", 1, {INT8OID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min", 1, {FLOAT4OID},
	  "c:min", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min", 1, {FLOAT8OID},
	  "c:min", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "min", 1, {NUMERICOID},
	  "c:min", 1, {NUMERICOID},
	  {ALTFUNC_EXPR_PMIN}, DEVKERNEL_NEEDS_NUMERIC, INT_MAX
	},
#endif
	{ "min", 1, {CASHOID},
	  "c:max", 1, {CASHOID},
	  {ALTFUNC_EXPR_PMAX}, DEVKERNEL_NEEDS_MONEY, INT_MAX
	},
	{ "min", 1, {DATEOID},
	  "c:min", 1, {DATEOID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min", 1, {TIMEOID},
	  "c:min", 1, {TIMEOID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min", 1, {TIMESTAMPOID},
	  "c:min", 1, {TIMESTAMPOID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min", 1, {TIMESTAMPTZOID},
	  "c:min", 1, {TIMESTAMPTZOID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},

	/* SUM(X) = SUM(PSUM(X)) */
	{ "sum", 1, {INT2OID},
	  "s:sum", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "sum", 1, {INT4OID},
	  "s:sum", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "sum", 1, {INT8OID},
	  "c:sum", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "sum", 1, {FLOAT4OID},
	  "c:sum", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "sum", 1, {FLOAT8OID},
	  "c:sum", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "sum", 1, {NUMERICOID},
	  "c:sum", 1, {NUMERICOID},
	  {ALTFUNC_EXPR_PSUM}, DEVKERNEL_NEEDS_NUMERIC, 100
	},
#endif
	{ "sum", 1, {CASHOID},
	  "c:sum", 1, {CASHOID},
	  {ALTFUNC_EXPR_PSUM}, DEVKERNEL_NEEDS_MONEY, INT_MAX
	},
	/* STDDEV(X) = EX_STDDEV(NROWS(),PSUM(X),PSUM(X*X)) */
	{ "stddev", 1, {FLOAT4OID},
	  "s:stddev", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "stddev", 1, {FLOAT8OID},
	  "s:stddev", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "stddev", 1, {NUMERICOID},
	  "s:stddev", 3, {INT4OID, NUMERICOID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, DEVKERNEL_NEEDS_NUMERIC, 32
	},
#endif
	{ "stddev_pop", 1, {FLOAT4OID},
	  "s:stddev_pop", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "stddev_pop", 1, {FLOAT8OID},
	  "s:stddev_pop", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "stddev_pop", 1, {NUMERICOID},
	  "s:stddev_pop", 3, {INT4OID, NUMERICOID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2}, DEVKERNEL_NEEDS_NUMERIC, 32
	},
#endif
	{ "stddev_samp", 1, {FLOAT4OID},
	  "s:stddev_samp", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "stddev_samp", 1, {FLOAT8OID},
	  "s:stddev_samp", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "stddev_samp", 1, {NUMERICOID},
	  "s:stddev_samp", 3, {INT4OID, NUMERICOID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, DEVKERNEL_NEEDS_NUMERIC, 32
	},
#endif
	/* VARIANCE(X) = PGSTROM.VARIANCE(NROWS(), PSUM(X),PSUM(X^2)) */
	{ "variance", 1, {FLOAT4OID},
	  "s:variance", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "variance", 1, {FLOAT8OID},
	  "s:variance", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "variance", 1, {NUMERICOID},
	  "s:variance", 3, {INT4OID, NUMERICOID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2}, DEVKERNEL_NEEDS_NUMERIC, 32
	},
#endif
	{ "var_pop", 1, {FLOAT4OID},
	  "s:var_pop", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "var_pop", 1, {FLOAT8OID},
	  "s:var_pop", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "var_pop", 1, {NUMERICOID},
	  "s:var_pop", 3, {INT4OID, NUMERICOID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2}, DEVKERNEL_NEEDS_NUMERIC, 32
	},
#endif
	{ "var_samp", 1, {FLOAT4OID},
	  "s:var_samp", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "var_samp", 1, {FLOAT8OID},
	  "s:var_samp", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "var_samp", 1, {NUMERICOID},
	  "s:var_samp", 3, {INT4OID, NUMERICOID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2}, DEVKERNEL_NEEDS_NUMERIC, 32
	},
#endif
	/*
	 * CORR(X,Y) = PGSTROM.CORR(NROWS(X,Y),
	 *                          PCOV_X(X,Y),  PCOV_Y(X,Y)
	 *                          PCOV_X2(X,Y), PCOV_Y2(X,Y),
	 *                          PCOV_XY(X,Y))
	 */
	{ "corr", 2, {FLOAT8OID, FLOAT8OID},
	  "s:corr", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
	},
	{ "covar_pop", 2, {FLOAT8OID, FLOAT8OID},
	  "s:covar_pop", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
	},
	{ "covar_samp", 2, {FLOAT8OID, FLOAT8OID},
	  "s:covar_samp", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
	},
	/*
	 * Aggregation to support least squares method
	 *
	 * That takes PSUM_X, PSUM_Y, PSUM_X2, PSUM_Y2, PSUM_XY according
	 * to the function
	 */
	{ "regr_avgx", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_avgx", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PCOV_X,
       ALTFUNC_EXPR_PCOV_X2,
       ALTFUNC_EXPR_PCOV_Y,
       ALTFUNC_EXPR_PCOV_Y2,
       ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
	},
	{ "regr_avgy", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_avgy", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
	},
	{ "regr_count", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_count", 1, {INT4OID}, {ALTFUNC_EXPR_NROWS}, 0
	},
	{ "regr_intercept", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_intercept", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
	},
	{ "regr_r2", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_r2", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
	},
	{ "regr_slope", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_slope", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
	},
	{ "regr_sxx", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_sxx", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
	},
	{ "regr_sxy", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_sxy", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
	},
	{ "regr_syy", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_syy", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
	},
};

static const aggfunc_catalog_t *
aggfunc_lookup_by_oid(Oid aggfnoid)
{
	Form_pg_proc	proform;
	HeapTuple		htup;
	int				i;

	htup = SearchSysCache1(PROCOID, ObjectIdGetDatum(aggfnoid));
	if (!HeapTupleIsValid(htup))
		elog(ERROR, "cache lookup failed for function %u", aggfnoid);
	proform = (Form_pg_proc) GETSTRUCT(htup);

	for (i=0; i < lengthof(aggfunc_catalog); i++)
	{
		aggfunc_catalog_t  *catalog = &aggfunc_catalog[i];

		if (strcmp(catalog->aggfn_name, NameStr(proform->proname)) == 0 &&
			catalog->aggfn_nargs == proform->pronargs &&
			memcmp(catalog->aggfn_argtypes,
				   proform->proargtypes.values,
				   sizeof(Oid) * catalog->aggfn_nargs) == 0)
		{
			ReleaseSysCache(htup);
			return catalog;
		}
	}
	ReleaseSysCache(htup);
	return NULL;
}

/*
 * cost_gpupreagg
 *
 * cost estimation of Aggregate if GpuPreAgg is injected
 */
#define LOG2(x)		(log(x) / 0.693147180559945)

static void
cost_gpupreagg(const Agg *agg, const Sort *sort, const Plan *outer_plan,
			   AggStrategy new_agg_strategy,
			   List *gpupreagg_tlist,
			   AggClauseCosts *agg_clause_costs,
			   Plan *p_newcost_agg,
			   Plan *p_newcost_sort,
			   Plan *p_newcost_gpreagg,
			   double num_groups,
			   cl_int *p_num_chunks,
			   cl_int safety_limit)
{
	Cost		startup_cost;
	Cost		run_cost;
	QualCost	qual_cost;
	int			gpagg_width;
	double		outer_rows;
	double		nrows_per_chunk;
	double		num_chunks;
	double		segment_width;
	double		num_segments;
	double		gpagg_nrows;
	double		gpu_cpu_ratio;
	cl_uint		ncols;
	Size		htup_size;
	ListCell   *lc;
	Path		dummy;

	Assert(outer_plan != NULL);

	/*
	 * Fixed cost come from outer relation
	 */
	startup_cost = outer_plan->startup_cost;
	run_cost = outer_plan->total_cost - startup_cost;
	outer_rows = outer_plan->plan_rows;

	/*
	 * Fixed cost to setup/launch GPU kernel
	 */
	startup_cost += pgstrom_gpu_setup_cost;

	/*
	 * Estimate number of chunks and nrows per chunk
	 */
	ncols = list_length(outer_plan->targetlist);
	htup_size = MAXALIGN(offsetof(HeapTupleHeaderData,
								  t_bits[BITMAPLEN(ncols)]));
	foreach (lc, outer_plan->targetlist)
	{
		TargetEntry	   *tle = lfirst(lc);

		/*
		 * For more correctness, we'd like to reference table's
		 * statistics if underlying outer-plan is Scan plan on
		 * regular relation.
		 */
		htup_size += get_typavgwidth(exprType((Node *) tle->expr),
									 exprTypmod((Node *) tle->expr));
	}
	nrows_per_chunk = (pgstrom_chunk_size() -
					   STROMALIGN(offsetof(kern_data_store,
										   colmeta[ncols])))
		/ (MAXALIGN(htup_size) + sizeof(cl_uint));
	nrows_per_chunk = Min(nrows_per_chunk, outer_rows);
	num_chunks = Max(outer_rows / nrows_per_chunk, 1.0);
	segment_width = Max(num_groups / nrows_per_chunk, 1.0);
	num_segments = Max(num_chunks / (segment_width * safety_limit), 1.0);

	/* write back */
	*p_num_chunks = num_chunks;

	/*
	 * Then, how much rows does GpuPreAgg will produce?
	 */
	gpagg_nrows = num_segments * num_groups;

	/*
	 * Cost estimation of internal Hash operations on GPU
	 */
	gpu_cpu_ratio = pgstrom_gpu_operator_cost / cpu_operator_cost;
	memset(&qual_cost, 0, sizeof(QualCost));
	gpagg_width = 0;
	foreach (lc, gpupreagg_tlist)
	{
		TargetEntry	   *tle = lfirst(lc);
		QualCost		cost;

		/* no code uses PlannerInfo here. NULL may be OK */
		cost_qual_eval_node(&cost, (Node *) tle->expr, NULL);
		qual_cost.startup += cost.startup;
		qual_cost.per_tuple += cost.per_tuple;

		gpagg_width += get_typavgwidth(exprType((Node *) tle->expr),
									  exprTypmod((Node *) tle->expr));
	}
	startup_cost += qual_cost.startup * gpu_cpu_ratio;

	qual_cost.per_tuple += cpu_operator_cost * Max(agg->numCols, 1);

	/*
	 * TODO: run_cost of GPU execution depends on
	 * - policy: nogroup, local+global, global
	 * - probability of hash colision
	 */
	run_cost += qual_cost.per_tuple * gpu_cpu_ratio *
		nrows_per_chunk * num_chunks;

	/*
	 * Cost to communicate upper node
	 */
	run_cost += cpu_tuple_cost * gpagg_nrows;


	/*
	 * set cost values on GpuPreAgg
	 */
	p_newcost_gpreagg->startup_cost = startup_cost;
	p_newcost_gpreagg->total_cost = startup_cost + run_cost;
	p_newcost_gpreagg->plan_rows = gpagg_nrows;
	p_newcost_gpreagg->plan_width = gpagg_width;

	/*
	 * Update estimated sorting cost, if any.
	 */
	if (sort != NULL)
	{
		cost_sort(&dummy,
				  NULL,		/* PlannerInfo is not referenced! */
				  NIL,		/* NIL is acceptable */
				  p_newcost_gpreagg->total_cost,
				  p_newcost_gpreagg->plan_rows,
				  p_newcost_gpreagg->plan_width,
				  0.0,
				  work_mem,
				  -1.0);
		p_newcost_sort->startup_cost = dummy.startup_cost;
		p_newcost_sort->total_cost = dummy.total_cost;
		p_newcost_sort->plan_rows = gpagg_nrows;
		p_newcost_sort->plan_width = gpagg_width;
		/*
		 * increase of startup_cost/run_cost according to the Sort
		 * to be injected between Agg and GpuPreAgg.
		 */
		startup_cost = dummy.startup_cost;
		run_cost     = dummy.total_cost - dummy.startup_cost;
	}

	/*
	 * Update estimated aggregate cost.
	 * Calculation logic is cost_agg() as built-in code doing.
	 */
	cost_agg(&dummy,
			 NULL,		/* PlannerInfo is not referenced! */
			 new_agg_strategy,
			 agg_clause_costs,
			 agg->numCols,
			 (double) agg->numGroups,
			 startup_cost,
			 startup_cost + run_cost,
			 gpagg_nrows);
	p_newcost_agg->startup_cost = dummy.startup_cost;
	p_newcost_agg->total_cost   = dummy.total_cost;
	p_newcost_agg->plan_rows    = agg->plan.plan_rows;
	p_newcost_agg->plan_width   = agg->plan.plan_width;

	/* cost for HAVING clause, if any */
	if (agg->plan.qual)
	{
		cost_qual_eval(&qual_cost, agg->plan.qual, NULL);
		p_newcost_agg->startup_cost += qual_cost.startup;
		p_newcost_agg->total_cost += qual_cost.startup +
			qual_cost.per_tuple * gpagg_nrows;
	}
	/* cost for tlist */
	add_tlist_costs_to_plan(NULL, p_newcost_agg, agg->plan.targetlist);
}

/*
 * expr_fixup_varno - create a copy of expression node, but varno of Var
 * shall be fixed up, If required, it also applies sanity checks for
 * the source varno.
 */
typedef struct
{
	Index	src_varno;	/* if zero, all the source varno is accepted */
	Index	dst_varno;
} expr_fixup_varno_context;

static Node *
expr_fixup_varno_mutator(Node *node, expr_fixup_varno_context *context)
{
	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		Var	   *varnode = (Var *) node;
		Var	   *newnode;

		if (context->src_varno > 0 && context->src_varno != varnode->varno)
			elog(ERROR, "Bug? varno %d is not expected one (%d) : %s",
				 varnode->varno, context->src_varno, nodeToString(varnode));
		newnode = copyObject(varnode);
		newnode->varno = context->dst_varno;

		return (Node *) newnode;
	}
	return expression_tree_mutator(node, expr_fixup_varno_mutator, context);
}

static inline void *
expr_fixup_varno(void *from, Index src_varno, Index dst_varno)
{
	expr_fixup_varno_context context;

	context.src_varno = src_varno;
	context.dst_varno = dst_varno;

	return expr_fixup_varno_mutator((Node *) from, &context);
}

/*
 * makeZeroConst - create zero constant
 */
static Const *
makeZeroConst(Oid consttype, int32 consttypmod, Oid constcollid)
{
	int16		typlen;
	bool		typbyval;
	Datum		zero_datum;

	get_typlenbyval(consttype, &typlen, &typbyval);
	switch (consttype)
	{
		case INT4OID:
			zero_datum = Int32GetDatum(0);
			break;
		case INT8OID:
			zero_datum = Int64GetDatum(0);
			break;
		case FLOAT4OID:
			zero_datum = Float4GetDatum(0.0);
			break;
		case FLOAT8OID:
			zero_datum = Float8GetDatum(0.0);
			break;
		default:
			elog(ERROR, "type (%u) is not expected", consttype);
			break;
	}
	return makeConst(consttype,
					 consttypmod,
					 constcollid,
					 (int) typlen,
					 zero_datum,
					 false,
					 typbyval);
}

/*
 * functions to make expression node of alternative aggregate/functions
 *
 * make_expr_typecast() - makes type case to the destination type
 * make_expr_conditional() - makes the supplied expression conditional
 *   using CASE WHEN ... THEN ... ELSE ... END clause.
 * make_altfunc_expr() - makes alternative function expression
 * make_altfunc_nrows_expr() - makes expression node of number or rows.
 * make_altfunc_expr_pcov() - makes expression node of covariances.
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

static Expr *
make_expr_conditional(Expr *expr, Expr *filter, Expr *defresult)
{
	CaseWhen   *case_when;
	CaseExpr   *case_expr;

	/*
	 * NOTE: Var->vano shall be replaced to INDEX_VAR on the following
	 * make_altfunc_expr(), so we keep the expression as-is, at this
	 * moment.
	 */
	Assert(exprType((Node *) filter) == BOOLOID);
	if (defresult)
		defresult = expr_fixup_varno(defresult, OUTER_VAR, INDEX_VAR);
	else
	{
		defresult = (Expr *) makeNullConst(exprType((Node *) expr),
										   exprTypmod((Node *) expr),
										   exprCollation((Node *) expr));
	}
	Assert(exprType((Node *) expr) == exprType((Node *) defresult));

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

static Expr *
make_altfunc_expr(const char *func_name, List *args)
{
	Oid			namespace_oid = get_namespace_oid("pgstrom", false);
	Oid			typebuf[8];
	oidvector  *func_argtypes;
	HeapTuple	tuple;
	Form_pg_proc proc_form;
	Expr	   *expr;
	ListCell   *cell;
	int			i = 0;

	/* set up oidvector */
	foreach (cell, args)
		typebuf[i++] = exprType((Node *) lfirst(cell));
	func_argtypes = buildoidvector(typebuf, i);

	/* find an alternative aggregate function */
	tuple = SearchSysCache3(PROCNAMEARGSNSP,
							PointerGetDatum(func_name),
							PointerGetDatum(func_argtypes),
							ObjectIdGetDatum(namespace_oid));
	if (!HeapTupleIsValid(tuple))
		return NULL;
	proc_form = (Form_pg_proc) GETSTRUCT(tuple);
	expr = (Expr *) makeFuncExpr(HeapTupleGetOid(tuple),
								 proc_form->prorettype,
								 expr_fixup_varno(args, OUTER_VAR, INDEX_VAR),
								 InvalidOid,
								 InvalidOid,
								 COERCE_EXPLICIT_CALL);
	ReleaseSysCache(tuple);
	return expr;
}

static Expr *
make_altfunc_nrows_expr(Aggref *aggref)
{
	List	   *nrows_args = NIL;
	ListCell   *cell;

	/* NOTE: make_altfunc_expr() translates OUTER_VAR to INDEX_VAR,
	 * so we don't need to translate the expression nodes at this
	 * moment.
	 */
	if (aggref->aggfilter)
		nrows_args = lappend(nrows_args, copyObject(aggref->aggfilter));

	foreach (cell, aggref->args)
	{
		TargetEntry *tle = lfirst(cell);
		NullTest	*ntest = makeNode(NullTest);

		Assert(IsA(tle, TargetEntry));
		ntest->arg = copyObject(tle->expr);
		ntest->nulltesttype = IS_NOT_NULL;
		ntest->argisrow = false;

		nrows_args = lappend(nrows_args, ntest);
	}
	return make_altfunc_expr("nrows", nrows_args);
}

/*
 * make_altfunc_pcov_expr - constructs an expression node for partial
 * covariance aggregates.
 */
static Expr *
make_altfunc_pcov_expr(Aggref *aggref, const char *func_name)
{
	Expr		*filter;
	TargetEntry *tle_1 = linitial(aggref->args);
	TargetEntry *tle_2 = lsecond(aggref->args);

	Assert(IsA(tle_1, TargetEntry) && IsA(tle_2, TargetEntry));
	if (!aggref->aggfilter)
		filter = (Expr *) makeBoolConst(true, false);
	else
		filter = expr_fixup_varno(aggref->aggfilter, OUTER_VAR, INDEX_VAR);
	return make_altfunc_expr(func_name,
							 list_make3(filter, tle_1->expr, tle_2->expr));
}

/*
 * make_gpupreagg_refnode
 *
 * It tries to construct an alternative Aggref node that references
 * partially aggregated results on the target-list of GpuPreAgg node.
 */
static Aggref *
make_gpupreagg_refnode(Aggref *aggref, List **prep_tlist,
					   int *extra_flags, int *safety_limit)
{
	const aggfunc_catalog_t *aggfn_cat;
	const char *altfn_name;
	oidvector  *altfn_argtypes;
	Aggref	   *altnode;
	ListCell   *cell;
	Oid			namespace_oid;
	HeapTuple	tuple;
	Form_pg_proc proc_form;
	int			i;

	/*
	 * Right now, aggregate functions with DISTINCT or ORDER BY are not
	 * supported by GpuPreAgg
	 */
	if (aggref->aggorder || aggref->aggdistinct)
		return NULL;

	/* Only aggregated functions listed on the catalog above is supported. */
	aggfn_cat = aggfunc_lookup_by_oid(aggref->aggfnoid);
	if (!aggfn_cat)
		return NULL;

	/* MEMO: Right now, functions below are not supported, so should not
	 * be on the aggfunc_catalog.
	 * - ordered-set aggregate function
	 * - aggregate function that takes VARIADIC argument
	 * - length of arguments are less than 2.
	 */
	Assert(!aggref->aggdirectargs &&
		   !aggref->aggvariadic &&
		   list_length(aggref->args) <= 2);
	/* update extra flags */
	*extra_flags |= aggfn_cat->extra_flags;
	/* update safety limit */
	*safety_limit = Min(*safety_limit, aggfn_cat->safety_limit);

	/*
	 * Expression node that is executed in the device kernel has to be
	 * supported by codegen.c
	 */
	foreach (cell, aggref->args)
	{
		TargetEntry *tle = lfirst(cell);
		if (!pgstrom_device_expression(tle->expr))
			return NULL;
	}
	if (!pgstrom_device_expression(aggref->aggfilter))
		return NULL;

	/*
	 * pulls the definition of alternative aggregate functions from
	 * the catalog. we expect these are installed in "pgstrom" schema.
	 */
	if (strncmp(aggfn_cat->altfn_name, "c:", 2) == 0)
		namespace_oid = PG_CATALOG_NAMESPACE;
	else if (strncmp(aggfn_cat->altfn_name, "s:", 2) == 0)
	{
		namespace_oid = get_namespace_oid("pgstrom", true);
		if (!OidIsValid(namespace_oid))
		{
			ereport(NOTICE,
					(errcode(ERRCODE_UNDEFINED_SCHEMA),
					 errmsg("schema \"pgstrom\" was not found"),
					 errhint("Try to run: CREATE EXTENSION pg_strom")));
			return NULL;
		}
	}
	else
		elog(ERROR, "Bug? unexpected namespace of alternative aggregate");

	altfn_name = aggfn_cat->altfn_name + 2;
	altfn_argtypes = buildoidvector(aggfn_cat->altfn_argtypes,
									aggfn_cat->altfn_nargs);
	tuple = SearchSysCache3(PROCNAMEARGSNSP,
							PointerGetDatum(altfn_name),
							PointerGetDatum(altfn_argtypes),
							ObjectIdGetDatum(namespace_oid));
	if (!HeapTupleIsValid(tuple))
	{
		ereport(NOTICE,
				(errcode(ERRCODE_UNDEFINED_FUNCTION),
				 errmsg("no alternative aggregate function \"%s\" exists",
						funcname_signature_string(altfn_name,
												  aggfn_cat->altfn_nargs,
												  NIL,
												  aggfn_cat->altfn_argtypes)),
				 errhint("Try to run: CREATE EXTENSION pg_strom")));
		return NULL;
	}
	proc_form = (Form_pg_proc) GETSTRUCT(tuple);

	/* sanity checks */
	if (proc_form->prorettype != aggref->aggtype)
		elog(ERROR, "bug? alternative function has different result type");

	/*
	 * construct an Aggref node that represent alternative aggregate
	 * function with preprocessed arguments.
	 */
	altnode = makeNode(Aggref);
	altnode->aggfnoid      = HeapTupleGetOid(tuple);
	altnode->aggtype       = aggref->aggtype;
	altnode->aggcollid     = aggref->aggcollid;
	altnode->inputcollid   = aggref->inputcollid;
	altnode->aggdirectargs = NIL;	/* see the checks above */
	altnode->args          = NIL;	/* to be set below */
	altnode->aggorder      = aggref->aggorder;
	altnode->aggdistinct   = aggref->aggdistinct;
	altnode->aggfilter     = NULL;	/* moved to GpuPreAgg */
	altnode->aggstar       = false;	/* all the alt-agg takes arguments */
	altnode->aggvariadic   = false;	/* see the checks above */
	altnode->aggkind       = aggref->aggkind;
	altnode->agglevelsup   = aggref->agglevelsup;
	altnode->location      = aggref->location;

	ReleaseSysCache(tuple);

	/*
	 * construct arguments of alternative aggregate function. It references
	 * an entry of prep_tlist, so we put expression node on the tlist on
	 * demand.
	 */
	for (i=0; i < aggfn_cat->altfn_nargs; i++)
	{
		int			code = aggfn_cat->altfn_argexprs[i];
		Oid			argtype_oid = aggfn_cat->altfn_argtypes[i];
		TargetEntry *tle;
		Expr	   *expr;
		Var		   *varref;
		AttrNumber	resno;

		switch (code)
		{
			case ALTFUNC_EXPR_NROWS:
				expr = make_altfunc_nrows_expr(aggref);
				break;
			case ALTFUNC_EXPR_PMIN:
				tle = linitial(aggref->args);
				Assert(IsA(tle, TargetEntry));
				expr = tle->expr;
				if (aggref->aggfilter)
					expr = make_expr_conditional(expr, aggref->aggfilter,
												 NULL);
				expr = make_altfunc_expr("pmin", list_make1(expr));
				break;
			case ALTFUNC_EXPR_PMAX:
				tle = linitial(aggref->args);
				Assert(IsA(tle, TargetEntry));
				expr = tle->expr;
				if (aggref->aggfilter)
					expr = make_expr_conditional(expr, aggref->aggfilter,
												 NULL);
				expr = make_altfunc_expr("pmax", list_make1(expr));
				break;
			case ALTFUNC_EXPR_PSUM:
				tle = linitial(aggref->args);
				Assert(IsA(tle, TargetEntry));
				expr = tle->expr;
				if (exprType((Node *) expr) != argtype_oid)
					expr = make_expr_typecast(expr, argtype_oid);
				if (aggref->aggfilter)
				{
					Expr   *defresult
						= (Expr *) makeZeroConst(exprType((Node *) expr),
												 exprTypmod((Node *) expr),
												 exprCollation((Node *) expr));
					expr = make_expr_conditional(expr,
												 aggref->aggfilter,
												 defresult);
				}
				expr = make_altfunc_expr("psum", list_make1(expr));
				break;
			case ALTFUNC_EXPR_PSUM_X2:
				tle = linitial(aggref->args);
				Assert(IsA(tle, TargetEntry));
				expr = tle->expr;
				if (exprType((Node *) expr) != argtype_oid)
					expr = make_expr_typecast(expr, argtype_oid);
				if (aggref->aggfilter)
				{
					Expr   *defresult
						= (Expr *) makeZeroConst(exprType((Node *) expr),
												 exprTypmod((Node *) expr),
												 exprCollation((Node *) expr));
					expr = make_expr_conditional(expr,
												 aggref->aggfilter,
												 defresult);
				}
				expr = make_altfunc_expr("psum_x2", list_make1(expr));
				break;
			case ALTFUNC_EXPR_PCOV_X:
				expr = make_altfunc_pcov_expr(aggref, "pcov_x");
				break;
			case ALTFUNC_EXPR_PCOV_Y:
				expr = make_altfunc_pcov_expr(aggref, "pcov_y");
				break;
			case ALTFUNC_EXPR_PCOV_X2:
				expr = make_altfunc_pcov_expr(aggref, "pcov_x2");
				break;
			case ALTFUNC_EXPR_PCOV_Y2:
				expr = make_altfunc_pcov_expr(aggref, "pcov_y2");
				break;
			case ALTFUNC_EXPR_PCOV_XY:
				expr = make_altfunc_pcov_expr(aggref, "pcov_xy");
				break;
			default:
				elog(ERROR, "Bug? unexpected ALTFUNC_EXPR_* label");
		}
		/* does aggregate function contained unsupported expression? */
		if (!expr)
			return NULL;

		/* check return type of the alternative functions */
		if (argtype_oid != exprType((Node *) expr))
		{
			elog(NOTICE, "Bug? result type is \"%s\", but \"%s\" is expected",
				 format_type_be(exprType((Node *) expr)),
				 format_type_be(argtype_oid));
			return NULL;
		}

		/* add this expression node on the prep_tlist */
		foreach (cell, *prep_tlist)
		{
			tle = lfirst(cell);

			if (equal(tle->expr, expr))
			{
				resno = tle->resno;
				break;
			}
		}
		if (!cell)
		{
			tle = makeTargetEntry(expr,
								  list_length(*prep_tlist) + 1,
								  NULL,
								  false);
			*prep_tlist = lappend(*prep_tlist, tle);
			resno = tle->resno;
		}
		/*
		 * alternative aggregate function shall reference this resource.
		 */
		varref = makeVar(OUTER_VAR,
						 resno,
						 exprType((Node *) expr),
						 exprTypmod((Node *) expr),
						 exprCollation((Node *) expr),
						 (Index) 0);
		tle = makeTargetEntry((Expr *) varref,
							  list_length(altnode->args) + 1,
							  NULL,
							  false);
		altnode->args = lappend(altnode->args, tle);
	}
	return altnode;
}

typedef struct
{
	Agg		   *agg;
	List	   *tlist_gpa;
	Bitmapset  *attr_refs;
	AttrNumber *attr_maps;
	int			extra_flags;
	int			safety_limit;
	bool		not_available;
	const char *not_available_reason;
} gpupreagg_rewrite_context;

static Node *
gpupreagg_rewrite_mutator(Node *node, gpupreagg_rewrite_context *context)
{
	if (!node)
		return NULL;

	if (IsA(node, Aggref))
	{
		Aggref	   *orgagg = (Aggref *) node;
		Aggref	   *altagg = NULL;

		altagg = make_gpupreagg_refnode(orgagg,
										&context->tlist_gpa,
										&context->extra_flags,
										&context->safety_limit);
		if (!altagg)
		{
			context->not_available = true;
			context->not_available_reason =
				"because no alternative functions are supported";
		}
		return (Node *) altagg;
	}
	else if (IsA(node, Var))
	{
		Var		   *varnode = (Var *) node;
		Var		   *newnode;
		AttrNumber	varattno = context->attr_maps[varnode->varattno - 1];

		if (varattno > 0)
		{
			newnode = copyObject(varnode);
			newnode->varattno = varattno;

			return (Node *) newnode;
		}
		context->not_available = true;
		context->not_available_reason =
			"because var-node referenced non-grouping key";
		return NULL;
	}
	return expression_tree_mutator(node, gpupreagg_rewrite_mutator,
								   (void *)context);
}

static bool
gpupreagg_rewrite_expr(Agg *agg,
					   List **p_agg_tlist,
					   List **p_agg_quals,
					   List **p_tlist_gpa,
					   AttrNumber **p_attr_maps,
					   Bitmapset **p_attr_refs,
					   int	*p_extra_flags,
					   Size *p_varlena_unitsz,
					   cl_int *p_safety_limit)
{
	gpupreagg_rewrite_context context;
	Plan	   *outer_plan = outerPlan(agg);
	List	   *tlist_gpa = NIL;
	List	   *agg_tlist = NIL;
	List	   *agg_quals = NIL;
	AttrNumber *attr_maps;
	Bitmapset  *attr_refs = NULL;
	Bitmapset  *grouping_keys = NULL;
	Size		varlena_unitsz = 0;
	ListCell   *cell;
	Size		final_length;
	Size		final_nslots;
	int			i, ncols;

	/* In case of sort-aggregate, it has an underlying Sort node on top
	 * of the scan node. GpuPreAgg shall be injected under the Sort node
	 * to reduce burden of CPU sorting.
	 */
	if (IsA(outer_plan, Sort))
		outer_plan = outerPlan(outer_plan);

	/*
	 * Picks up all the grouping keys, includes the ones used to GROUPING
	 * SET clause, then add them to tlist_gpa.
	 */
	for (i=0; i < agg->numCols; i++)
		grouping_keys = bms_add_member(grouping_keys, agg->grpColIdx[i]);
	foreach (cell, agg->chain)
	{
		Agg	   *subagg = (Agg *)lfirst(cell);

		Assert(subagg->plan.targetlist == NIL && subagg->plan.qual == NIL);

		for (i=0; i < subagg->numCols; i++)
			grouping_keys = bms_add_member(grouping_keys,
										   subagg->grpColIdx[i]);
	}

	attr_maps = palloc0(sizeof(AttrNumber) *
						list_length(outer_plan->targetlist));
	foreach (cell, outer_plan->targetlist)
	{
		TargetEntry	   *tle = lfirst(cell);
		TargetEntry	   *tle_new;
		devtype_info   *dtype;
		Expr		   *varnode;
		Oid				type_oid;
		int32			type_mod;
		Oid				type_coll;
		char		   *resname;

		/* not a grouping key */
		if (!bms_is_member(tle->resno, grouping_keys))
			continue;

		type_oid = exprType((Node *) tle->expr);
		type_mod = exprTypmod((Node *) tle->expr);
		type_coll = exprCollation((Node *) tle->expr);
		resname = (tle->resname ? pstrdup(tle->resname) : NULL);

		/* Grouping key must be a supported data type */
		dtype = pgstrom_devtype_lookup(type_oid);
		if (!dtype || !OidIsValid(dtype->type_eqfunc))
		{
			elog(DEBUG1, "Unabled to apply GpuPreAgg "
				 "due to unsupported data type as grouping key: %s",
				 format_type_be(type_oid));
			return false;
		}
		/* grouping key type must have equality function on device */
		if (!pgstrom_devfunc_lookup(dtype->type_eqfunc, type_coll))
		{
			elog(DEBUG1, "Unabled to apply GpuPreAgg "
				 "due to unsupported equality function in grouping key: %s",
				 format_procedure(dtype->type_eqfunc));
			return false;
		}

		/* check whether types need special treatment */
		if (!dtype->type_byval)
		{
			/*
			 * We also need to estimate average size of varlene or indirect
			 * grouping keys to make copies of varlena datum on extra area
			 * of the final result buffer.
			 */
			varlena_unitsz += MAXALIGN(get_typavgwidth(type_oid, type_mod));
		}
		varnode = (Expr *) makeVar(INDEX_VAR,
								   tle->resno,
								   type_oid,
								   type_mod,
								   type_coll,
								   0);
		tle_new = makeTargetEntry(varnode,
								  list_length(tlist_gpa) + 1,
								  resname,
								  tle->resjunk);
		tlist_gpa = lappend(tlist_gpa, tle_new);
		attr_refs = bms_add_member(attr_refs, tle->resno -
								   FirstLowInvalidHeapAttributeNumber);
		attr_maps[tle->resno - 1] = tle_new->resno;
	}

	/*
	 * Estimation of the required final result buffer size.
	 * At least, it has to be smaller than allocatable length in GPU RAM.
	 * Elsewhere, we have no choice to run GpuPreAgg towards this node.
	 */
	ncols = list_length(tlist_gpa);
	final_nslots = (Size)(2.5 * agg->plan.plan_rows *
						  pgstrom_chunk_size_margin);
	final_length = STROMALIGN(offsetof(kern_data_store,
									   colmeta[ncols])) +
		STROMALIGN(LONGALIGN((sizeof(Datum) +
							  sizeof(char)) * ncols) * final_nslots) +
		STROMALIGN((Size)((double) varlena_unitsz *
						  agg->plan.plan_rows *
						  pgstrom_chunk_size_margin)) +
		STROMALIGN(sizeof(kern_gpupreagg) * final_nslots);
	if (final_length > gpuMemMaxAllocSize() / 2)
	{
		elog(DEBUG1, "GpuPreAgg: expected final result buffer too large");
		return false;
	}

	/*
	 * On the next, replace aggregate functions in tlist of Agg node
	 * according to the aggfunc_catalog[] definition.
	 */
	memset(&context, 0, sizeof(gpupreagg_rewrite_context));
	context.agg = agg;
	context.tlist_gpa = tlist_gpa;
	context.attr_refs = attr_refs;
	context.attr_maps = attr_maps;
	context.extra_flags = 0;
	context.safety_limit = INT_MAX;

	/*
	 * Construction of the modified target-list to be assigned 
	 *
	 * New Agg node shall have alternative aggregate function that takes
	 * partially aggregated result in GpuPreAgg.
	 */
	foreach (cell, agg->plan.targetlist)
	{
		TargetEntry	   *oldtle = lfirst(cell);
		TargetEntry	   *newtle = flatCopyTargetEntry(oldtle);

		newtle->expr = (Expr *)gpupreagg_rewrite_mutator((Node *)oldtle->expr,
														 &context);
		if (context.not_available)
		{
			elog(DEBUG1, "Unable to apply GpuPreAgg because %s: %s",
				 context.not_available_reason,
				 nodeToString(oldtle->expr));
			return false;
		}
		agg_tlist = lappend(agg_tlist, newtle);
	}

	/*
	 * Adjustment of varattno in the HAVING clause of newagg
	 */
	foreach (cell, agg->plan.qual)
	{
		Expr	   *old_expr = lfirst(cell);
		Expr	   *new_expr;

		new_expr = (Expr *)gpupreagg_rewrite_mutator((Node *)old_expr,
													 &context);
		if (context.not_available)
		{
			elog(DEBUG1, "Unable to apply GpuPreAgg because %s: %s",
                 context.not_available_reason,
                 nodeToString(old_expr));
			return false;
		}
		agg_quals = lappend(agg_quals, new_expr);
	}
	*p_agg_tlist = agg_tlist;
	*p_agg_quals = agg_quals;
	*p_tlist_gpa = context.tlist_gpa;
	*p_attr_maps = attr_maps;
	*p_attr_refs = context.attr_refs;
	*p_extra_flags = context.extra_flags;
	*p_varlena_unitsz = varlena_unitsz;
	*p_safety_limit = context.safety_limit;

	return true;
}

/*
 * gpupreagg_codegen_qual_eval - code generator of kernel gpupreagg_qual_eval()
 * that check qualifier on individual tuples prior to the job of preagg
 *
 * static bool
 * gpupreagg_qual_eval(kern_context *kcxt,
 *                     kern_parambuf *kparams,
 *                     kern_data_store *kds,
 *                     size_t kds_index);
 */
static char *
gpupreagg_codegen_qual_eval(GpuPreAggInfo *gpa_info,
							codegen_context *context)
{
	StringInfoData	str;

	/* init context */
	context->param_refs = NULL;
	context->used_vars = NIL;

	initStringInfo(&str);
	appendStringInfo(
        &str,
        "STATIC_FUNCTION(bool)\n"
        "gpupreagg_qual_eval(kern_context *kcxt,\n"
        "                    kern_data_store *kds,\n"
        "                    size_t kds_index)\n"
        "{\n");

	if (gpa_info->outer_quals != NIL)
	{
		/*
		 * Generate code for qual evaluation
		 * Note that outer_quals was pulled up from outer plan, thus,
		 * its varnodes are arranged to reference fields on outer
		 * relation. So, it does not need to pay attention for projection,
		 * however, needs to be careful to deal with...
		 */
		Node   *outer_quals = (Node *)gpa_info->outer_quals;
		char   *expr_code = pgstrom_codegen_expression(outer_quals,
													   context);
		Assert(expr_code != NULL);

		pgstrom_codegen_param_declarations(&str, context);
		pgstrom_codegen_var_declarations(&str, context);
		appendStringInfo(
			&str,
			"\n"
			"  return EVAL(%s);\n",
			expr_code);
	}
	else
	{
		appendStringInfo(&str, "  return true;\n");
	}
	appendStringInfo(&str, "}\n");

	return str.data;
}

/*
 *  
 *
 * static cl_uint
 * gpupreagg_hashvalue(kern_context *kcxt,
 *                     cl_uint *crc32_table,
 *                     cl_uint hash_value,
 *                     kern_data_store *kds,
 *                     size_t kds_index)
 */
static char *
gpupreagg_codegen_hashvalue(GpuPreAggInfo *gpa_info,
							List *tlist_gpa,
							codegen_context *context)
{
	StringInfoData	str;
	StringInfoData	decl;
	StringInfoData	body;
	int				i;

	initStringInfo(&str);
	initStringInfo(&decl);
    initStringInfo(&body);
	context->param_refs = NULL;

	appendStringInfo(&decl,
					 "STATIC_FUNCTION(cl_uint)\n"
					 "gpupreagg_hashvalue(kern_context *kcxt,\n"
					 "                    cl_uint *crc32_table,\n"
					 "                    cl_uint hash_value,\n"
					 "                    kern_data_store *kds,\n"
					 "                    size_t kds_index)\n"
					 "{\n");

	for (i=0; i < gpa_info->numCols; i++)
	{
		TargetEntry	   *tle;
		AttrNumber		resno = gpa_info->grpColIdx[i];
		devtype_info   *dtype;
		Var			   *var;

		tle = get_tle_by_resno(tlist_gpa, resno);
		var = (Var *) tle->expr;
		if (!IsA(var, Var) || var->varno != INDEX_VAR)
			elog(ERROR, "Bug? A simple Var node is expected for group key: %s",
				 nodeToString(var));

		/* find a datatype for comparison */
		dtype = pgstrom_devtype_lookup_and_track(var->vartype, context);
		if (!OidIsValid(dtype->type_cmpfunc))
			elog(ERROR, "Bug? type (%u) has no comparison function",
				 var->vartype);

		/* variable declarations */
		appendStringInfo(
			&decl,
			"  pg_%s_t keyval_%u = pg_%s_vref(kds,kcxt,%u,kds_index);\n",
			dtype->type_name, resno,
			dtype->type_name, resno - 1);

		/* crc32 computing */
		appendStringInfo(
			&body,
			"  hash_value = pg_%s_comp_crc32(crc32_table,\n"
			"                                hash_value, keyval_%u);\n",
			dtype->type_name, resno);
	}
	/* no constants should be appear */
	Assert(bms_is_empty(context->param_refs));

	appendStringInfo(&decl,
					 "%s\n"
					 "  return hash_value;\n"
					 "}\n", body.data);
	pfree(body.data);

	return decl.data;
}

/*
 * gpupreagg_codegen_keycomp - code generator of kernel gpupreagg_keymatch();
 * that compares two records indexed by x_index and y_index in kern_data_store,
 * then returns -1 if X < Y, 0 if X = Y or 1 if X > Y.
 *
 * static cl_bool
 * gpupreagg_keymatch(kern_context *kcxt,
 *                    kern_data_store *kds_x, size_t x_index,
 *                    kern_data_store *kds_y, size_t y_index)
 */
static char *
gpupreagg_codegen_keycomp(GpuPreAggInfo *gpa_info,
						  List *tlist_gpa,
						  codegen_context *context)
{
	StringInfoData	str;
	StringInfoData	decl;
	StringInfoData	body;
	int				i;

	initStringInfo(&str);
	initStringInfo(&decl);
    initStringInfo(&body);
	context->param_refs = NULL;

	for (i=0; i < gpa_info->numCols; i++)
	{
		TargetEntry	   *tle;
		AttrNumber		resno = gpa_info->grpColIdx[i];
		Var			   *var;
		devtype_info   *dtype;
		devfunc_info   *dfunc;

		tle = get_tle_by_resno(tlist_gpa, resno);
		var = (Var *) tle->expr;
		if (!IsA(var, Var) || var->varno != INDEX_VAR)
			elog(ERROR, "Bug? A simple Var node is expected for group key: %s",
				 nodeToString(var));

		/* find a function to compare this data-type */
		/* find a datatype for comparison */
		dtype = pgstrom_devtype_lookup_and_track(var->vartype, context);
		if (!OidIsValid(dtype->type_eqfunc))
			elog(ERROR, "Bug? type (%u) has no equality function",
				 var->vartype);
		dfunc = pgstrom_devfunc_lookup_and_track(dtype->type_eqfunc,
												 var->varcollid,
												 context);
		/* variable declarations */
		appendStringInfo(&decl,
						 "  pg_%s_t xkeyval_%u;\n"
						 "  pg_%s_t ykeyval_%u;\n",
						 dtype->type_name, resno,
						 dtype->type_name, resno);
		/*
		 * values comparison
		 *
		 * XXX - note that NUMERIC data type is already transformed to
		 * the internal format on projection timing, so not a good idea
		 * to use pg_numeric_vref() here, because it assumes Numeric in
		 * varlena format.
		 */
		appendStringInfo(
			&body,
			"  xkeyval_%u = pg_%s_vref(x_kds,kcxt,%u,x_index);\n"
			"  ykeyval_%u = pg_%s_vref(y_kds,kcxt,%u,y_index);\n"
			"  if (!xkeyval_%u.isnull && !ykeyval_%u.isnull)\n"
			"  {\n"
			"    if (!EVAL(pgfn_%s(kcxt, xkeyval_%u, ykeyval_%u)))\n"
			"      return false;\n"
			"  }\n"
			"  else if ((xkeyval_%u.isnull  && !ykeyval_%u.isnull) ||\n"
			"           (!xkeyval_%u.isnull &&  ykeyval_%u.isnull))\n"
			"      return false;\n",
			resno, dtype->type_name, resno - 1,
			resno, dtype->type_name, resno - 1,
			resno, resno,
			dfunc->func_devname, resno, resno,
			resno, resno,
			resno, resno);
	}
	/* add parameters, if referenced */
	if (!bms_is_empty(context->param_refs))
	{
		pgstrom_codegen_param_declarations(&decl, context);
		bms_free(context->param_refs);
	}

	/* make a whole key-compare function */
	appendStringInfo(
		&str,
		"STATIC_FUNCTION(cl_bool)\n"
		"gpupreagg_keymatch(kern_context *kcxt,\n"
		"                   kern_data_store *x_kds, size_t x_index,\n"
		"                   kern_data_store *y_kds, size_t y_index)\n"
		"{\n"
		"%s"	/* variable/params declarations */
		"%s"
		"  return true;\n"
		"}\n",
		decl.data,
		body.data);
	pfree(decl.data);
	pfree(body.data);

	return str.data;
}

/*
 * gpupreagg_codegen_local_calc - code generator of gpupreagg_local_calc()
 * kernel function that implements an operation to make partial aggregation
 * on the local memory.
 * The supplied accum is operated by newval, according to resno.
 *
 * gpupreagg_local_calc(kern_context *kcxt,
 *                      cl_int attnum,
 *                      pagg_datum *accum,
 *                      pagg_datum *newval);
 */
static inline const char *
aggcalc_method_of_typeoid(Oid type_oid)
{
	switch (type_oid)
	{
		case INT2OID:
			return "SHORT";
		case INT4OID:
		case DATEOID:
			return "INT";
		case INT8OID:
		case CASHOID:
		case TIMEOID:
		case TIMESTAMPOID:
		case TIMESTAMPTZOID:
			return "LONG";
		case FLOAT4OID:
			return "FLOAT";
		case FLOAT8OID:
			return "DOUBLE";
		case NUMERICOID:
			return "NUMERIC";
	}
	elog(ERROR, "unexpected partial aggregate data-type");
}

static char *
gpupreagg_codegen_aggcalc(GpuPreAggInfo *gpa_info,
						  List *tlist_gpa,
						  cl_int mode,
						  codegen_context *context)
{
	Oid				namespace_oid = get_namespace_oid("pgstrom", false);
	StringInfoData	body;
	const char	   *aggcalc_class;
	const char	   *aggcalc_args;
	ListCell	   *cell;

	initStringInfo(&body);
	switch(mode) {
	case GPUPREAGG_LOCAL_REDUCTION:
		appendStringInfo(
			&body,
			"STATIC_FUNCTION(void)\n"
			"gpupreagg_local_calc(kern_context *kcxt,\n"
			"                     cl_int attnum,\n"
			"                     pagg_datum *accum,\n"
			"                     pagg_datum *newval)\n"
			"{\n");
		aggcalc_class = "LOCAL";
        aggcalc_args = "kcxt,accum,newval";
		break;
	case GPUPREAGG_GLOBAL_REDUCTION:
		appendStringInfo(
			&body,
			"STATIC_FUNCTION(void)\n"
			"gpupreagg_global_calc(kern_context *kcxt,\n"
			"                      cl_int attnum,\n"
			"                      kern_data_store *accum_kds,\n"
			"                      size_t accum_index,\n"
			"                      kern_data_store *newval_kds,\n"
			"                      size_t newval_index)\n"
			"{\n"
			"  char    *accum_isnull	__attribute__((unused))\n"
			"   = KERN_DATA_STORE_ISNULL(accum_kds,accum_index) + attnum;\n"
			"  Datum   *accum_value		__attribute__((unused))\n"
			"   = KERN_DATA_STORE_VALUES(accum_kds,accum_index) + attnum;\n"
			"  char     new_isnull		__attribute__((unused))\n"
			"   = KERN_DATA_STORE_ISNULL(newval_kds,newval_index)[attnum];\n"
			"  Datum    new_value		__attribute__((unused))\n"
			"   = KERN_DATA_STORE_VALUES(newval_kds,newval_index)[attnum];\n"
			"\n"
			"  assert(accum_kds->format == KDS_FORMAT_SLOT);\n"
			"  assert(newval_kds->format == KDS_FORMAT_SLOT);\n"
			"\n");
		aggcalc_class = "GLOBAL";
		aggcalc_args = "kcxt,accum_isnull,accum_value,new_isnull,new_value";
		break;
	case GPUPREAGG_NOGROUP_REDUCTION:
		appendStringInfo(
			&body,
			"STATIC_FUNCTION(void)\n"
			"gpupreagg_nogroup_calc(kern_context *kcxt,\n"
			"                       cl_int attnum,\n"
			"                       pagg_datum *accum,\n"
			"                       pagg_datum *newval)\n"
			"{\n");
		aggcalc_class = "NOGROUP";
        aggcalc_args = "kcxt,accum,newval";
		break;
	default:
		elog(ERROR, "Invalid GpuPreAgg calc mode (%u)", mode);
		break;
	}

	appendStringInfo(
		&body,
		"  switch (attnum)\n"
		"  {\n"
		);
	/* NOTE: The targetList of GpuPreAgg are either Const (as NULL),
	 * Var node (as grouping key), or FuncExpr (as partial aggregate
	 * calculation).
	 */
	foreach (cell, tlist_gpa)
	{
		TargetEntry	   *tle = lfirst(cell);
		FuncExpr	   *func;
		Oid				type_oid;
		const char	   *func_name;

		if (!IsA(tle->expr, FuncExpr))
		{
			Assert(IsA(tle->expr, Const) || IsA(tle->expr, Var));
			continue;
		}
		func = (FuncExpr *) tle->expr;
		
		if (namespace_oid != get_func_namespace(func->funcid))
		{
			elog(NOTICE, "Bug? function not in pgstrom schema");
			continue;
		}
		func_name = get_func_name(func->funcid);

		if (strcmp(func_name, "nrows") == 0)
		{
			/* nrows() is always int4 */
			appendStringInfo(&body,
							 "  case %d:\n"
							 "    AGGCALC_%s_PADD_%s(%s);\n"
							 "    break;\n",
							 tle->resno - 1,
							 aggcalc_class,
							 aggcalc_method_of_typeoid(INT4OID),
							 aggcalc_args);
		}
		else if (strcmp(func_name, "pmax") == 0)
		{
			Assert(list_length(func->args) == 1);
			type_oid = exprType(linitial(func->args));
			appendStringInfo(&body,
							 "  case %d:\n"
							 "    AGGCALC_%s_PMAX_%s(%s);\n"
							 "    break;\n",
							 tle->resno - 1,
							 aggcalc_class,
							 aggcalc_method_of_typeoid(type_oid),
							 aggcalc_args);
		}
		else if (strcmp(func_name, "pmin") == 0)
		{
			Assert(list_length(func->args) == 1);
			type_oid = exprType(linitial(func->args));
			appendStringInfo(&body,
							 "  case %d:\n"
							 "    AGGCALC_%s_PMIN_%s(%s);\n"
							 "    break;\n",
							 tle->resno - 1,
							 aggcalc_class,
							 aggcalc_method_of_typeoid(type_oid),
							 aggcalc_args);
		}
		else if (strcmp(func_name, "psum") == 0    ||
				 strcmp(func_name, "psum_x2") == 0)
		{
			/* it should never be NULL */
			Assert(list_length(func->args) == 1);
			type_oid = exprType(linitial(func->args));
			appendStringInfo(&body,
							 "  case %d:\n"
							 "    AGGCALC_%s_PADD_%s(%s);\n"
							 "    break;\n",
							 tle->resno - 1,
							 aggcalc_class,
							 aggcalc_method_of_typeoid(type_oid),
							 aggcalc_args);
		}
		else if (strcmp(func_name, "pcov_x") == 0  ||
				 strcmp(func_name, "pcov_y") == 0  ||
				 strcmp(func_name, "pcov_x2") == 0 ||
				 strcmp(func_name, "pcov_y2") == 0 ||
				 strcmp(func_name, "pcov_xy") == 0)
		{
			/* covariance takes only float8 datatype */
			appendStringInfo(&body,
							 "  case %d:\n"
							 "    AGGCALC_%s_PADD_%s(%s);\n"
							 "    break;\n",
							 tle->resno - 1,
							 aggcalc_class,
							 aggcalc_method_of_typeoid(FLOAT8OID),
							 aggcalc_args);
		}
		else
		{
			elog(NOTICE, "Bug? unexpected function: %s", func_name);
		}
	}
	appendStringInfo(
		&body,
		"  default:\n"
		"    break;\n"
		"  }\n"
		"}\n");
	return body.data;
}

/*
 * gpupreagg_codegen_projection_nrows - put initial value of nrows()
 */
static void
gpupreagg_codegen_projection_nrows(StringInfo body, cl_int dst_index,
								   FuncExpr *func, codegen_context *context)
{
	devtype_info   *dtype;
	ListCell	   *cell;

	dtype = pgstrom_devtype_lookup_and_track(INT4OID, context);
	if (!dtype)
		elog(ERROR, "cache lookup failed for device type: %s",
			 format_type_be(INT4OID));

	if (list_length(func->args) > 0)
	{
		appendStringInfoString(body, "  temp.bool_v = (");
		foreach (cell, func->args)
		{
			if (cell != list_head(func->args))
				appendStringInfoString(
					body,
					" &&\n"
					"                 ");
			appendStringInfoString(
				body,
				pgstrom_codegen_expression(lfirst(cell), context));
		}
		appendStringInfo(
			body,
			");\n"
			"  dst_isnull[%d] = temp.bool_v.isnull;\n"
			"  dst_values[%d] = pg_int4_to_datum(EVAL(temp.bool_v) ? 1 : 0);\n",
			dst_index - 1,
			dst_index - 1);
	}
	else
		appendStringInfo(
			body,
			"  dst_isnull[%d] = false;\n"
			"  dst_values[%d] = 1;\n",
			dst_index - 1,
			dst_index - 1);
}

/*
 * gpupreagg_codegen_projection_misc - put initial value of MIN, MAX or SUM
 */
static void
gpupreagg_codegen_projection_misc(StringInfo body, cl_int dst_index,
								  FuncExpr *func, const char *func_name,
								  codegen_context *context)
{
	/* Store the original value as-is. If clause is conditional and
	 * false, NULL shall be set. Even if NULL, value fields MUST have
	 * reasonable initial value to the later atomic operation.
	 * In case of PMIN(), NULL takes possible maximum number.
	 */
	Node		   *clause = linitial(func->args);
	Oid				type_oid = exprType(clause);
	devtype_info   *dtype;
	const char	   *max_const;
	const char	   *min_const;
	const char	   *zero_const;

	switch (type_oid)
	{
		case INT2OID:
			/* NOTE: Only 32/64bit width are supported by atomic operation,
			 * thus, we deal with 16bit datum using 32bit field internally.
			 */
			clause = (Node *)
				makeFuncExpr(F_I2TOI4,
							 INT4OID,
							 list_make1(clause),
							 InvalidOid,
							 InvalidOid,
							 COERCE_IMPLICIT_CAST);
			type_oid = INT4OID;
			/* no break here */

		case INT4OID:
			max_const = "INT_MAX";
			min_const = "INT_MIN";
			zero_const = "0";
			break;

		case INT8OID:
            max_const = "LONG_MAX";
            min_const = "LONG_MIN";
            zero_const = "0";
			break;

		case FLOAT4OID:
			max_const = "FLT_MAX";
			min_const = "-FLT_MAX";
			zero_const = "0.0";
			break;

		case FLOAT8OID:
			max_const = "DBL_MAX";
			min_const = "-DBL_MAX";
			zero_const = "0.0";
			break;

		case NUMERICOID:
			max_const = "PG_NUMERIC_MAX";
			min_const = "PG_NUMERIC_MIN";
			zero_const = "PG_NUMERIC_ZERO";
			break;

		case CASHOID:
            max_const = "LONG_MAX";
            min_const = "LONG_MIN";
            zero_const = "0";
			break;

		case DATEOID:
			max_const = "INT_MAX";
			min_const = "INT_MIN";
			zero_const = "0";
			break;

		case TIMEOID:
			max_const = "LONG_MAX";
			min_const = "LONG_MIN";
			zero_const = "0";
			break;

		case TIMESTAMPOID:
			max_const = "LONG_MAX";
			min_const = "LONG_MIN";
			zero_const = "0";
			break;

		case TIMESTAMPTZOID:
			max_const = "LONG_MAX";
			min_const = "LONG_MIN";
			zero_const = "0";
			break;

		default:
			elog(ERROR, "Bug? cache lookup failed for device type: %s",
				 format_type_be(type_oid));
	}
	dtype = pgstrom_devtype_lookup_and_track(type_oid, context);
	if (!dtype)
		elog(ERROR, "device type lookup failed: %u", type_oid);

	appendStringInfo(
		body,
		"  temp.%s_v = %s;\n"
		"  dst_isnull[%d] = temp.%s_v.isnull;\n"
		"  if (!temp.%s_v.isnull)\n"
		"    dst_values[%d] = pg_%s_to_datum(temp.%s_v.value);\n"
		"  else\n"
		"    dst_values[%d] = %s;\n",
		dtype->type_name,
		pgstrom_codegen_expression(clause, context),
		dst_index - 1,
		dtype->type_name,
		dtype->type_name,
		dst_index - 1,
		dtype->type_name,
		dtype->type_name,
		dst_index - 1,
		strcmp(func_name, "pmin") == 0 ? max_const :
		strcmp(func_name, "pmax") == 0 ? min_const :
		strcmp(func_name, "psum") == 0 ? zero_const : "__invalid__");
}

/*
 * gpupreagg_codegen_projection_psum_x2 - Put initial value of psum_x2
 */
static void
gpupreagg_codegen_projection_psum_x2(StringInfo body, cl_int dst_index,
									 FuncExpr *func,
									 codegen_context *context)
{
	Node		   *clause = linitial(func->args);
	devtype_info   *dtype;
	devfunc_info   *dfunc;
	const char	   *zero_label;

	if (exprType(clause) == FLOAT8OID)
	{
		dtype = pgstrom_devtype_lookup_and_track(FLOAT8OID, context);
		if (!dtype)
			elog(ERROR, "cache lookup failed for device type: %s",
				 format_type_be(FLOAT8OID));
		dfunc = pgstrom_devfunc_lookup_and_track(F_FLOAT8MUL,
												 InvalidOid,
												 context);
		if (!dtype)
			elog(ERROR, "cache lookup failed for device function: %u",
				 F_FLOAT8MUL);
		zero_label = "0.0";
	}
	else if (exprType(clause) == NUMERICOID)
	{
		dtype = pgstrom_devtype_lookup_and_track(NUMERICOID, context);
		if (!dtype)
			elog(ERROR, "cache lookup failed for device type: %s",
				 format_type_be(NUMERICOID));
		dfunc = pgstrom_devfunc_lookup_and_track(F_NUMERIC_MUL,
												 InvalidOid,
												 context);
		if (!dtype)
			elog(ERROR, "device function lookup failed: %u", F_NUMERIC_MUL);
		zero_label = "PG_NUMERIC_ZERO";
	}
	else
		elog(ERROR, "Bug? psum_x2 expects either float8 or numeric");

	appendStringInfo(
		body,
		"  temp.%s_v = %s;\n"
		"  temp.%s_v = pgfn_%s(kcxt, temp.%s_v, temp.%s_v);\n"
		"  dst_isnull[%d] = temp.%s_v.isnull;\n"
		"  if (temp.%s_v.isnull)\n"
		"    dst_values[%d] = %s;\n"
		"  else\n"
		"    dst_values[%d] = pg_%s_to_datum(temp.%s_v.value);\n",
		dtype->type_name, pgstrom_codegen_expression(clause, context),
		dtype->type_name, dfunc->func_devname,
		dtype->type_name, dtype->type_name,
		dst_index - 1, dtype->type_name,
		dtype->type_name,
		dst_index - 1, zero_label,
		dst_index - 1, dtype->type_name, dtype->type_name);
}

/*
 * gpupreagg_codegen_projection_corr - Put initial value of pcov_XX
 */
static void
gpupreagg_codegen_projection_corr(StringInfo body, cl_int dst_index,
								  FuncExpr *func, const char *func_name,
								  codegen_context *context)
{
	devfunc_info *dfunc;
	Node	   *filter = linitial(func->args);
	Node	   *x_clause = lsecond(func->args);
	Node	   *y_clause = lthird(func->args);

	if (IsA(filter, Const))
	{
		Const  *cons = (Const *) filter;
		if (cons->consttype == BOOLOID &&
			!cons->constisnull &&
			DatumGetBool(cons->constvalue))
			filter = NULL;		/* no filter, actually */
	}

	appendStringInfo(
		body,
		"  temp_float8x = %s;\n"
		"  temp_float8y = %s;\n",
		pgstrom_codegen_expression(x_clause, context),
		pgstrom_codegen_expression(y_clause, context));
	appendStringInfo(
		body,
		"  if (temp_float8x.isnull || temp_float8y.isnull");
	if (filter)
		appendStringInfo(
			body,
			" ||\n"
			"      !EVAL(%s)",
			pgstrom_codegen_expression(filter, context));

	appendStringInfoString(
		body,
		")\n"
		"  {\n"
		"    temp.float8_v.isnull = true;\n"
		"    temp.float8_v.value = 0.0;\n"
		"  }\n"
		"  else\n");

	/* initial value according to the function */
	if (strcmp(func_name, "pcov_x") == 0)
	{
		appendStringInfoString(
			body,
			"    temp.float8_v = temp_float8x;\n");
	}
	else if (strcmp(func_name, "pcov_y") == 0)
	{
		appendStringInfoString(
			body,
			"    temp.float8_v = temp_float8y;\n");
	}
	else if (strcmp(func_name, "pcov_x2") == 0)
	{
		dfunc = pgstrom_devfunc_lookup_and_track(F_FLOAT8MUL,
												 InvalidOid,
												 context);
		appendStringInfo(
			body,
			"    temp.float8_v = pgfn_%s(kcxt,\n"
			"                            temp_float8x,\n"
			"                            temp_float8x);\n",
			dfunc->func_devname);
	}
	else if (strcmp(func_name, "pcov_y2") == 0)
	{
		dfunc = pgstrom_devfunc_lookup_and_track(F_FLOAT8MUL,
												 InvalidOid,
												 context);
		appendStringInfo(
			body,
			"    temp.float8_v = pgfn_%s(kcxt,\n"
			"                            temp_float8y,\n"
			"                            temp_float8y);\n",
			dfunc->func_devname);
	}
	else if (strcmp(func_name, "pcov_xy") == 0)
	{
		dfunc = pgstrom_devfunc_lookup_and_track(F_FLOAT8MUL,
												 InvalidOid,
												 context);
		appendStringInfo(
			body,
			"    temp.float8_v = pgfn_%s(kcxt,\n"
			"                            temp_float8x,\n"
			"                            temp_float8y);\n",
			dfunc->func_devname);
	}
	else
		elog(ERROR, "unexpected partial covariance function: %s", func_name);

	if (!pgstrom_devtype_lookup_and_track(FLOAT8OID, context))
		elog(ERROR, "cache lookup failed for device function: %u",
			 FLOAT8OID);

	appendStringInfo(
		body,
		"  dst_isnull[%d] = temp.float8_v.isnull;\n"
		"  if (temp.float8_v.isnull)\n"
		"    dst_values[%d] = 0.0;\n"
		"  else\n"
		"    dst_values[%d] = pg_float8_to_datum(temp.float8_v.value);\n",
		dst_index - 1,
		dst_index - 1,
		dst_index - 1);
}

/*
 * gpupreagg_codegen_projection
 *
 * Constructs the kernel projection function declared as:
 * STATIC_FUNCTION(void)
 * gpupreagg_projection(kern_context *kcxt,
 *                      kern_data_store *kds_src, // in, row-format
 *                      kern_tupitem *tupitem,    // in
 *                      kern_data_store *kds_dst  // out, slot_format
 *                      Datum *dst_values,        // out
 *                      cl_char *dst_isnull)      // out
 *
 *
 *
 */
static char *
gpupreagg_codegen_projection(GpuPreAggInfo *gpa_info,
							 List *tlist_gpa,
							 List *tlist_dev,
							 Index outer_scanrelid,
							 List *range_tables,
							 codegen_context *context)
{
	Oid				namespace_oid = get_namespace_oid("pgstrom", false);
	ListCell	   *lc;
	int				i, j, k;
	AttrNumber	   *varremaps = NULL;
	Bitmapset	   *varattnos = NULL;
	Bitmapset	   *kvars_decl = NULL;
	Bitmapset	   *ovars_decl = NULL;
	cl_bool			outer_compatible = true;
	Const		   *kparam_0;
	cl_char		   *gpagg_atts;
	Size			vl_length;
	struct varlena *vl_datum;
	StringInfoData	decl;
	StringInfoData	body;
	StringInfoData	temp;

	initStringInfo(&decl);
	initStringInfo(&body);
	initStringInfo(&temp);
	context->param_refs = NULL;

	appendStringInfoString(
		&decl,
		"STATIC_FUNCTION(void)\n"
		"gpupreagg_projection(kern_context *kcxt,\n"
		"                     kern_data_store *kds_src,\n"
		"                     kern_tupitem *tupitem,\n"
		"                     kern_data_store *kds_dst,\n"
		"                     Datum *dst_values,\n"
		"                     cl_char *dst_isnull)\n"
		"{\n");
	codegen_tempvar_declaration(&decl, "temp");
	appendStringInfoString(
		&decl,
		"  pg_float8_t        temp_float8x __attribute__ ((unused));\n"
		"  pg_float8_t        temp_float8y __attribute__ ((unused));\n"
		"  char              *addr         __attribute__ ((unused));\n");

	/*
	 * Step.1 - List up variables of outer-scan/relation to be referenced
	 * by aggregate function or grouping keys.
	 *
	 */
	varremaps = palloc0(sizeof(AttrNumber) * list_length(tlist_gpa));
	pull_varattnos((Node *) tlist_gpa, INDEX_VAR, &varattnos);
	foreach (lc, tlist_gpa)
	{
		TargetEntry *tle = lfirst(lc);

		if (IsA(tle->expr, Var))
		{
			Var	   *var = (Var *) tle->expr;

			Assert(var->varno == INDEX_VAR &&
				   var->varattno > InvalidAttrNumber &&
				   var->varattno <= list_length(tlist_dev));
			varremaps[tle->resno - 1] = var->varattno;
		}
		else
		{
			pull_varattnos((Node *) tle->expr, INDEX_VAR, &kvars_decl);
		}
	}

	if (outer_scanrelid > 0)
	{
		AttrNumber *varremaps_base = palloc0(sizeof(AttrNumber) *
											 list_length(tlist_gpa));
		foreach (lc, tlist_dev)
		{
			TargetEntry *tle = lfirst(lc);

			k = tle->resno - FirstLowInvalidHeapAttributeNumber;
			if (!bms_is_member(k, varattnos))
				continue;		/* ignore unreferenced column */

			if (IsA(tle->expr, Var))
			{
				Var	   *var = (Var *) tle->expr;

				if (tle->resno != var->varattno)
					outer_compatible = false;
				else
					Assert(exprType((Node *)tle->expr) == var->vartype);


				for (i=0; i < list_length(tlist_gpa); i++)
				{
					if (varremaps[i] == tle->resno)
						varremaps_base[i] = var->varattno;
				}
				if (bms_is_member(k, kvars_decl))
					ovars_decl = bms_add_member(ovars_decl, var->varattno -
										FirstLowInvalidHeapAttributeNumber);
			}
			else
			{
				outer_compatible = false;
				kvars_decl = bms_add_member(kvars_decl, k);
				pull_varattnos((Node *) tle->expr,
							   outer_scanrelid,
							   &ovars_decl);
			}
		}
		varremaps = varremaps_base;
	}

	/*
	 * Step.2 - Add declaration of the KVAR variables
	 */
	foreach (lc, tlist_dev)
	{
		TargetEntry	   *tle = lfirst(lc);
		devtype_info   *dtype;

		k = tle->resno - FirstLowInvalidHeapAttributeNumber;
		if (bms_is_member(k, kvars_decl))
		{
			dtype = pgstrom_devtype_lookup(exprType((Node *) tle->expr));
			if (!dtype)
				elog(ERROR, "cache lookup failed for device type: %s",
					 format_type_be(exprType((Node *) tle->expr)));
			appendStringInfo(
				&decl,
				"  pg_%s_t KVAR_%u;\n",
				dtype->type_name, tle->resno);
		}
	}

	/*
	 * Step.3 - Extract heap-tuple of the outer relation (may be base
	 * relation)
	 */
	appendStringInfo(
		&body,
		"  EXTRACT_HEAP_TUPLE_BEGIN(addr, kds_src, &tupitem->htup);\n");

	if (outer_scanrelid == 0 || outer_compatible)
	{
		foreach (lc, tlist_dev)
		{
			TargetEntry *tle = lfirst(lc);
			Var		   *var = (Var *) tle->expr;
			int16		typlen;
			bool		typbyval;
			bool		referenced = false;

			/* should be just a reference to the outer attribute */
			Assert(IsA(var, Var));
			get_typlenbyval(var->vartype, &typlen, &typbyval);
			/* direct move if this variable is referenced by varremaps */
			for (j=0; j < list_length(tlist_gpa); j++)
			{
				if (varremaps[j] != tle->resno)
					continue;

				if (var->vartype == NUMERICOID)
				{
					appendStringInfo(
						&temp,
						"  temp.numeric_v = pg_numeric_datum_ref(kcxt,addr,\n"
						"                                        false);\n"
						"  dst_isnull[%d] = temp.numeric_v.isnull;\n"
						"  dst_values[%d] = (Datum)temp.numeric_v.value;\n",
						j, j);
				}
				else if (!typbyval)
				{
					appendStringInfo(
						&temp,
						"  dst_isnull[%d] = (addr != NULL ? false : true);\n"
						"  dst_values[%d] = PointerGetDatum(addr);\n",
						j, j);
				}
				else
				{
					appendStringInfo(
						&temp,
						"  dst_isnull[%d] = (addr != NULL ? false : true);\n"
						"  if (addr)\n"
						"    dst_values[%d] = *((%s *) addr);\n",
						j, j,
						(typlen == sizeof(cl_long)  ? "cl_long" :
						 typlen == sizeof(cl_int)   ? "cl_int" :
						 typlen == sizeof(cl_short) ? "cl_short"
						 							: "cl_char"));
				}
				referenced = true;
			}

			/* Construct KVAR_%u if this variable is referenced by varattnos */
			k = tle->resno - FirstLowInvalidHeapAttributeNumber;
			if (bms_is_member(k, kvars_decl))
			{
				devtype_info   *dtype = pgstrom_devtype_lookup(var->vartype);
				appendStringInfo(
					&temp,
					"  KVAR_%u = pg_%s_datum_ref(kcxt,addr,false);\n",
					tle->resno,
					dtype->type_name);
				referenced = true;
			}
			/* we have to walk on until this column at least */
			if (referenced)
			{
				appendStringInfoString(&body, temp.data);
				resetStringInfo(&temp);
			}
			appendStringInfoString(
				&temp,
				"  EXTRACT_HEAP_TUPLE_NEXT(addr);\n");
		}
		appendStringInfoString(
			&body,
			"  EXTRACT_HEAP_TUPLE_END();\n\n");
	}
	else
	{
		RangeTblEntry  *rte = rt_fetch(outer_scanrelid, range_tables);
		Relation		outer_baserel = heap_open(rte->relid, NoLock);
		TupleDesc		tupdesc = RelationGetDescr(outer_baserel);
		const char	   *var_label_saved;

		for (i=0; i < tupdesc->natts; i++)
		{
			Form_pg_attribute attr = tupdesc->attrs[i];
			cl_bool		referenced = false;

			for (j=0; j < list_length(tlist_gpa); j++)
			{
				if (varremaps[j] != attr->attnum)
					continue;
				/* NUMERIC should have an internal representation */
				if (attr->atttypid == NUMERICOID)
				{
					appendStringInfo(
						&temp,
						"  temp.numeric_v = pg_numeric_datum_ref(kcxt,addr,\n"
						"                                        false);\n"
						"  dst_isnull[%d] = temp_numeric.isnull;\n"
						"  dst_values[%d] = (Datum)temp_numeric.value;\n",
						j, j);
				}
				else if (!attr->attbyval)
				{
					appendStringInfo(
						&temp,
						"  dst_isnull[%d] = (addr != NULL ? false : true);\n"
						"  dst_values[%d] = PointerGetDatum(addr);\n",
						j, j);
				}
				else
				{
					appendStringInfo(
						&temp,
						"  dst_isnull[%d] = (addr != NULL ? false : true);\n"
						"  if (addr)\n"
						"    dst_values[%d] = *((%s *) addr);\n",
						j, j,
						(attr->attlen == sizeof(cl_long)  ? "cl_long" :
						 attr->attlen == sizeof(cl_int)   ? "cl_int" :
						 attr->attlen == sizeof(cl_short) ? "cl_short"
						 								  : "cl_char"));
				}
				referenced = true;
			}
			/*
			 * Construct OVAR_%u
			 */
			k = attr->attnum - FirstLowInvalidHeapAttributeNumber;
			if (bms_is_member(k, ovars_decl))
			{
				devtype_info   *dtype = pgstrom_devtype_lookup(attr->atttypid);

				if (!dtype)
					elog(ERROR, "cache lookup failed for device type: %s",
						 format_type_be(attr->atttypid));

				appendStringInfo(
					&decl,
					"  pg_%s_t OVAR_%u;\n",
					dtype->type_name,
					attr->attnum);

				appendStringInfo(
					&temp,
					"  OVAR_%u = pg_%s_datum_ref(kcxt,addr,false);\n",
					attr->attnum,
					dtype->type_name);
				referenced = true;
			}
			/* we have to walk on until this column at least */
			if (referenced)
			{
				appendStringInfoString(&body, temp.data);
				resetStringInfo(&temp);
			}
			appendStringInfoString(
				&temp,
				"  EXTRACT_HEAP_TUPLE_NEXT(addr);\n");
		}
		appendStringInfoString(
			&body,
			"  EXTRACT_HEAP_TUPLE_END();\n\n");

		/*
		 * Construct KVAR_%u
		 */
		var_label_saved = context->var_label;
		context->var_label = "OVAR";
		foreach (lc, tlist_dev)
		{
			TargetEntry *tle = lfirst(lc);

			k = tle->resno - FirstLowInvalidHeapAttributeNumber;
			if (bms_is_member(k, varattnos) && varremaps[tle->resno - 1] == 0)
			{
				devtype_info   *dtype;

				dtype = pgstrom_devtype_lookup(exprType((Node *) tle->expr));
				if (!dtype)
					elog(ERROR, "cache lookup failed for device type: %s",
						 format_type_be(exprType((Node *) tle->expr)));

				appendStringInfo(
					&body,
					"  KVAR_%u = %s;\n",
					tle->resno,
					pgstrom_codegen_expression((Node *)tle->expr, context));
			}
		}
		context->var_label = var_label_saved;
		heap_close(outer_baserel, NoLock);
	}

	/*
	 * construction of kparam_0 - that is an array of cl_char, to inform
	 * kernel which fields are grouping-key, or aggregate function or not.
	 */
	kparam_0 = (Const *) linitial(context->used_params);
	vl_length = VARHDRSZ + sizeof(cl_char) * list_length(tlist_gpa);
	vl_datum = palloc0(vl_length);
	SET_VARSIZE(vl_datum, vl_length);
	kparam_0->constvalue = PointerGetDatum(vl_datum);
	kparam_0->constisnull = false;
	gpagg_atts = (cl_char *)VARDATA(vl_datum);

	foreach (lc, tlist_gpa)
	{
		TargetEntry *tle = lfirst(lc);

		/* Also track usage of this field */
		if (varremaps[tle->resno - 1] != 0)
		{
			gpagg_atts[tle->resno - 1] = GPUPREAGG_FIELD_IS_GROUPKEY;
			continue;
		}
		gpagg_atts[tle->resno - 1] = GPUPREAGG_FIELD_IS_AGGFUNC;

		appendStringInfoChar(&body, '\n');
		/* we should have KVAR_xx */
		if (IsA(tle->expr, Var))
		{
			Var			   *var = (Var *) tle->expr;
			devtype_info   *dtype;

			Assert(var->varno == INDEX_VAR);
			Assert(var->varattno > 0);

			dtype = pgstrom_devtype_lookup_and_track(var->vartype, context);
			if (!dtype)
				elog(ERROR, "cache lookup failed for device type: %s",
					 format_type_be(var->vartype));
			/*
			 * NOTE: Only numeric or fixed-length built-in data type can
			 * appear here. Other types shall be moved directly on the
			 * earlier stage, so we don't need to care about.
			 */
			Assert(dtype->type_oid == NUMERICOID || dtype->type_byval);
			appendStringInfo(
				&body,
				"  dst_isnull[%d] = KVAR_%u.isnull;\n"
				"  dst_values[%d] = pg_%s_to_datum(KVAR_%u.value);\n",
				tle->resno - 1,
				tle->resno,
				tle->resno - 1,
				dtype->type_name,
				tle->resno);
		}
		else if (IsA(tle->expr, FuncExpr))
		{
			FuncExpr   *func = (FuncExpr *) tle->expr;
			const char *func_name;

			if (namespace_oid != get_func_namespace(func->funcid))
				elog(ERROR, "Bug? unexpected FuncExpr: %s",
					 nodeToString(func));

			func_name = get_func_name(func->funcid);
			if (strcmp(func_name, "nrows") == 0)
				gpupreagg_codegen_projection_nrows(&body, tle->resno,
												   func, context);
			else if (strcmp(func_name, "pmax") == 0 ||
					 strcmp(func_name, "pmin") == 0 ||
					 strcmp(func_name, "psum") == 0)
				gpupreagg_codegen_projection_misc(&body, tle->resno,
												  func, func_name, context);
			else if (strcmp(func_name, "psum_x2") == 0)
				gpupreagg_codegen_projection_psum_x2(&body, tle->resno,
													 func, context);
			else if (strcmp(func_name, "pcov_x") == 0 ||
					 strcmp(func_name, "pcov_y") == 0 ||
					 strcmp(func_name, "pcov_x2") == 0 ||
					 strcmp(func_name, "pcov_y2") == 0 ||
					 strcmp(func_name, "pcov_xy") == 0)
				gpupreagg_codegen_projection_corr(&body, tle->resno,
												  func, func_name, context);
			else
				elog(ERROR, "Bug? unexpected partial aggregate function: %s",
					 func_name);
		}
		else
			elog(ERROR, "bug? unexpected node in tlist_gpa: %s",
				 nodeToString(tle->expr));
	}
	appendStringInfoString(&body, "}\n");

	pgstrom_codegen_param_declarations(&decl, context);
	appendStringInfo(&decl, "\n%s\n", body.data);
	pfree(body.data);
	pfree(temp.data);

	return decl.data;
}

static char *
gpupreagg_codegen(GpuPreAggInfo *gpa_info,
				  List *tlist_gpa,
				  List *tlist_dev,
				  Index outer_scanrelid,
				  List *range_tables,
				  codegen_context *context)
{
	StringInfoData	str;
	const char	   *fn_qualeval;
	const char	   *fn_hashvalue;
	const char	   *fn_keycomp;
	const char	   *fn_local_calc;
	const char	   *fn_global_calc;
	const char	   *fn_nogroup_calc;
	const char	   *fn_projection;

	/*
	 * System constants of GpuPreAgg:
	 * KPARAM_0 is an array of cl_char to inform which field is grouping
	 * keys, or target of (partial) aggregate function.
	 */
	context->used_params = list_make1(makeNullConst(BYTEAOID, -1, InvalidOid));
	pgstrom_devtype_lookup_and_track(BYTEAOID, context);

	/* generate a qual evaluation function */
	fn_qualeval = gpupreagg_codegen_qual_eval(gpa_info, context);
	/* generate gpupreagg_hashvalue function */
	fn_hashvalue = gpupreagg_codegen_hashvalue(gpa_info, tlist_gpa, context);
	/* generate a key comparison function */
	fn_keycomp = gpupreagg_codegen_keycomp(gpa_info, tlist_gpa, context);
	/* generate a gpupreagg_local_calc function */
	fn_local_calc = gpupreagg_codegen_aggcalc(gpa_info, tlist_gpa,
											  GPUPREAGG_LOCAL_REDUCTION,
											  context);
	/* generate a gpupreagg_global_calc function */
	fn_global_calc = gpupreagg_codegen_aggcalc(gpa_info, tlist_gpa,
											   GPUPREAGG_GLOBAL_REDUCTION,
											   context);
	/* generate a gpupreagg_global_calc function */
	fn_nogroup_calc = gpupreagg_codegen_aggcalc(gpa_info, tlist_gpa,
												GPUPREAGG_NOGROUP_REDUCTION,
												context);
	/* generate an initial data loading function */
	fn_projection = gpupreagg_codegen_projection(gpa_info,
												 tlist_gpa,
												 tlist_dev,
												 outer_scanrelid,
												 range_tables,
												 context);
	/* OK, add type/function declarations */
	initStringInfo(&str);
	/* function declarations */
	pgstrom_codegen_func_declarations(&str, context);
	/* special expression declarations */
	pgstrom_codegen_expr_declarations(&str, context);

	appendStringInfo(&str,
					 "%s\n"		/* gpupreagg_qual_eval() */
					 "%s\n"		/* gpupreagg_hashvalue() */
					 "%s\n"		/* gpupreagg_keycomp() */
					 "%s\n"		/* gpupreagg_local_calc() */
					 "%s\n"		/* gpupreagg_global_calc() */
					 "%s\n"		/* gpupreagg_nogroup_calc() */
					 "%s\n",	/* gpupreagg_projection() */
					 fn_qualeval,
					 fn_hashvalue,
					 fn_keycomp,
					 fn_local_calc,
					 fn_global_calc,
					 fn_nogroup_calc,
					 fn_projection);
	return str.data;
}

/*
 * indexvar_fixup_by_tlist - replaces the varnode in plan->targetlist that
 * references the custom_scan_tlist by the expression of the target-entry
 * referenced.
 */
static Node *
indexvar_fixup_by_tlist(Node *node, List *tlist_dev)
{
	if (!node)
		return NULL;
	if (IsA(node, Var))
	{
		Var			   *var = (Var *) node;
		TargetEntry	   *tle;

		Assert(var->varno == INDEX_VAR);
		Assert(var->varattno >= 1 &&
			   var->varattno <= list_length(tlist_dev));
		tle = list_nth(tlist_dev, var->varattno - 1);

		return copyObject(tle->expr);
	}
	return expression_tree_mutator(node, indexvar_fixup_by_tlist, tlist_dev);
}

/*
 * pgstrom_try_insert_gpupreagg
 *
 * Entrypoint of the gpupreagg. It checks whether the supplied Aggregate node
 * is consists of all supported expressions.
 */
void
pgstrom_try_insert_gpupreagg(PlannedStmt *pstmt, Agg *agg)
{
	CustomScan	   *cscan;
	GpuPreAggInfo	gpa_info;
	Sort		   *sort_node;
	Plan		   *outer_node;
	Index			outer_scanrelid = 0;
	List		   *tlist_gpa = NIL;
	List		   *agg_quals = NIL;
	List		   *agg_tlist = NIL;
	AttrNumber	   *attr_maps = NULL;
	Bitmapset	   *attr_refs = NULL;
	List		   *outer_tlist = NIL;
	List		   *outer_quals = NIL;
	List		   *tlist_dev = NIL;
	double			outer_nitems;
	Size			varlena_unitsz;
	cl_int			safety_limit;
	cl_int			key_dist_salt;
	double			num_groups;
	cl_int			num_chunks;
	ListCell	   *lc;
	int				i;
	AggStrategy		new_agg_strategy;
	AggClauseCosts	agg_clause_costs;
	Plan			newcost_agg;
	Plan			newcost_sort;
	Plan			newcost_gpreagg;
	int				extra_flags;
	codegen_context context;

	/* nothing to do, if feature is turned off */
	if (!pgstrom_enabled || !enable_gpupreagg)
		return;

	/* Try to construct target-list of both Agg and GpuPreAgg node.
	 * If unavailable to construct, it indicates this aggregation
	 * does not support partial aggregation.
	 */
	if (!gpupreagg_rewrite_expr(agg,
								&agg_tlist,
								&agg_quals,
								&tlist_gpa,
								&attr_maps,
								&attr_refs,
								&extra_flags,
								&varlena_unitsz,
								&safety_limit))
		return;
	/* main portion is always needed! */
	extra_flags |= DEVKERNEL_NEEDS_DYNPARA | DEVKERNEL_NEEDS_GPUPREAGG;

	/*
	 * cost estimation of aggregate clauses
	 */
	memset(&agg_clause_costs, 0, sizeof(AggClauseCosts));
	count_agg_clauses(NULL, (Node *) agg_tlist, &agg_clause_costs);
	count_agg_clauses(NULL, (Node *) agg_quals, &agg_clause_costs);

	/* be compiler quiet */
	memset(&newcost_agg,     0, sizeof(Plan));
	memset(&newcost_sort,    0, sizeof(Plan));
	memset(&newcost_gpreagg, 0, sizeof(Plan));

	if (agg->aggstrategy != AGG_SORTED)
	{
		/*
		 * If this aggregate strategy is either plain or hash, we don't
		 * need to pay attention something complicated.
		 * Just inject GpuPreAgg under the Agg node, because Agg does not
		 * expect order of input stream.
		 */
		sort_node = NULL;
		outer_node = outerPlan(agg);
		new_agg_strategy = agg->aggstrategy;
	}
	else if (IsA(outerPlan(agg), Sort))
	{
		/*
		 * If this aggregation expects the input stream is already sorted
		 * and Sort node is connected below, it makes sense to reduce
		 * number of rows to be processed by Sort, not only Agg.
		 * So, we try to inject GpuPreAgg prior to the Sort node.
		 */
		sort_node = (Sort *)outerPlan(agg);
		outer_node = outerPlan(sort_node);
		new_agg_strategy = agg->aggstrategy;
	}
	else
	{
		Size	hashentrysize;

		/*
		 * Elsewhere, outer-plan of Agg is not Sort even though its strategy
		 * is AGG_SORTED (it is likely index aware scan to ensure order of
		 * input stream). In this case, we try to replace this Agg by
		 * alternative Agg with AGG_HASHED strategy that takes underlying
		 * GpuPreAgg node.
		 */
		sort_node = NULL;
		outer_node = outerPlan(agg);
		new_agg_strategy = AGG_HASHED;

		/*
		 * NOTE: all the supported aggregate functions are available to
		 * aggregate values with both of hashed and sorted basis.
		 * So, we switch the strategy of Agg without checks.
		 * This assumption may change in the future version, but not now.
		 * All we need to check is over-consumption of local memory.
		 * If estimated amount of local memory usage is larger than
		 * work_mem, it is a case we should give up.
		 * (See the logic in choose_hashed_grouping)
		 */
		hashentrysize = (MAXALIGN(sizeof(MinimalTupleData)) +
						 MAXALIGN(agg->plan.plan_width) +
						 agg_clause_costs.transitionSpace +
						 hash_agg_entry_size(agg_clause_costs.numAggs));
		if (hashentrysize * agg->plan.plan_rows > work_mem * 1024L)
			return;
	}

	/*
	 * Estimation of the "effective" number of groups.
	 *
	 * NOTE: If number of groups are too small, it leads too much atomic
	 * contention. So, we add a small salt to distribute grouping keys
	 * to reasonable level. Of course, it shall be adjusted in run-time.
	 * So, it is just a baseline parameter.
	 */
	num_groups = Max(agg->plan.plan_rows, 1.0);
	if (num_groups < (gpuMaxThreadsPerBlock() / 4))
	{
		key_dist_salt = (gpuMaxThreadsPerBlock() / 4) / (cl_uint) num_groups;
		num_groups *= (double) key_dist_salt;
	}
	else
		key_dist_salt = 1;

	/*
	 * Estimate the cost if GpuPreAgg would be injected, and determine
	 * which plan is cheaper, unless pg_strom.debug_force_gpupreagg is
	 * not turned on.
	 */
	cost_gpupreagg(agg, sort_node, outer_node,
				   new_agg_strategy,
				   tlist_gpa,
				   &agg_clause_costs,
				   &newcost_agg,
				   &newcost_sort,
				   &newcost_gpreagg,
				   num_groups,
				   &num_chunks,
				   safety_limit);
	elog(DEBUG1,
		 "GpuPreAgg (cost=%.2f..%.2f) has%sadvantage to Agg(cost=%.2f...%.2f)",
		 newcost_agg.startup_cost,
		 newcost_agg.total_cost,
		 agg->plan.total_cost <= newcost_agg.total_cost ? " no " : " ",
		 agg->plan.startup_cost,
		 agg->plan.total_cost);

	if (!debug_force_gpupreagg &&
		agg->plan.total_cost <= newcost_agg.total_cost)
		return;

	/* OK, let's construct GpuPreAgg node here */

	/* Pulls-up outer node if it is a simple SeqScan or GpuScan */
	if (!pgstrom_pullup_outer_scan(outer_node, true, &outer_quals))
	{
		outer_tlist		= outer_node->targetlist;
		outer_nitems	= outer_node->plan_rows;
	}
	else
	{
		outer_scanrelid	= ((Scan *) outer_node)->scanrelid;
		outer_tlist		= outer_node->targetlist;
		outer_nitems	= outer_node->plan_rows;
		outer_node		= NULL;
	}

	/*
	 * Construction of tlist_dev - All GpuPreAgg shall do is just reference
	 * of the underlying outer plan, if any. If it performs relation scan
	 * by itself, GpuPreAgg node also does projection by itself.
	 *
	 * If GpuPreAgg node has no outer node, ExecInitCustomScan assumes
	 * custom_scan_tlist is the definition of input relation, so we have
	 * to assign tlist_dev (that is equivalent to the input) on the
	 * custom_scan_tlist.
	 */
	foreach (lc, outer_tlist)
	{
		TargetEntry	   *tle = lfirst(lc);
		TargetEntry	   *tle_new;
		Expr		   *expr;

		if (outer_node != NULL)
			expr = (Expr *) makeVar(OUTER_VAR,
									tle->resno,
									exprType((Node *) tle->expr),
									exprTypmod((Node *) tle->expr),
									exprCollation((Node *) tle->expr),
									0);
		else
			expr = copyObject(tle->expr);

		tle_new = makeTargetEntry(expr,
								  list_length(tlist_dev) + 1,
								  tle->resname,
								  false);
		tlist_dev = lappend(tlist_dev, tle_new);
	}

	/*
	 * OK, let's construct GpuPreAgg node then inject it.
	 */
	cscan = makeNode(CustomScan);
	cscan->scan.plan.startup_cost = newcost_gpreagg.startup_cost;
	cscan->scan.plan.total_cost   = newcost_gpreagg.total_cost;
	cscan->scan.plan.plan_rows    = newcost_gpreagg.plan_rows;
	cscan->scan.plan.plan_width   = newcost_gpreagg.plan_width;
	cscan->scan.plan.targetlist   = tlist_gpa;
	cscan->scan.plan.qual         = NIL;
	cscan->scan.plan.lefttree     = outer_node;
	cscan->scan.scanrelid         = outer_scanrelid;
	cscan->flags                  = 0;
	cscan->custom_scan_tlist      = (outer_node ? tlist_dev : NIL);
	cscan->custom_relids          = NULL;
	cscan->methods                = &gpupreagg_scan_methods;

	/* also set up private information */
	memset(&gpa_info, 0, sizeof(GpuPreAggInfo));
	gpa_info.tlist_dev      = tlist_dev;
	gpa_info.numCols        = agg->numCols;
	gpa_info.grpColIdx      = palloc0(sizeof(AttrNumber) * agg->numCols);
	for (i=0; i < agg->numCols; i++)
		gpa_info.grpColIdx[i] = attr_maps[agg->grpColIdx[i] - 1];
	gpa_info.num_groups     = num_groups;
	gpa_info.num_chunks     = num_chunks;
	gpa_info.varlena_unitsz = varlena_unitsz;
	gpa_info.safety_limit   = safety_limit;
	gpa_info.key_dist_salt  = key_dist_salt;
	gpa_info.outer_quals	= outer_quals;
	gpa_info.outer_nitems	= outer_nitems;

	/*
	 * construction of the kernel code according to the target-list
	 * and qualifiers (pulled-up from outer plan).
	 */
	pgstrom_init_codegen_context(&context);
	gpa_info.kern_source = gpupreagg_codegen(&gpa_info,
											 tlist_gpa,
											 tlist_dev,
											 outer_scanrelid,
											 pstmt->rtable,
											 &context);
	gpa_info.extra_flags = extra_flags | context.extra_flags;
	gpa_info.used_params = context.used_params;

	/*
	 * NOTE: In case when GpuPreAgg pull up outer base relation, CPU fallback
	 * routine wants to execute projection from the base relation in one step.
	 * So, we rewrite the target-list to represent base_rel -> tlist_dev ->
	 * tlist_gpa.
	 */
	if (outer_scanrelid > 0)
	{
		cscan->scan.plan.targetlist = (List *)
			indexvar_fixup_by_tlist((Node *)cscan->scan.plan.targetlist,
									tlist_dev);
	}
	form_gpupreagg_info(cscan, &gpa_info);

	/*
	 * OK, let's inject GpuPreAgg and update the cost values.
	 */
	if (!sort_node)
		outerPlan(agg) = &cscan->scan.plan;
	else
	{
		List   *tlist_sort = NIL;

		foreach (lc, tlist_gpa)
		{
			TargetEntry	   *tle = lfirst(lc);
			Var			   *var_sort;

			var_sort = makeVar(OUTER_VAR,
							   tle->resno,
							   exprType((Node *) tle->expr),
							   exprTypmod((Node *) tle->expr),
							   exprCollation((Node *) tle->expr),
							   0);
			tlist_sort = lappend(tlist_sort,
								 makeTargetEntry((Expr *) var_sort,
												 list_length(tlist_sort) + 1,
												 tle->resname,
												 tle->resjunk));
		}
		sort_node->plan.startup_cost = newcost_sort.startup_cost;
		sort_node->plan.total_cost   = newcost_sort.total_cost;
		sort_node->plan.plan_rows    = newcost_sort.plan_rows;
		sort_node->plan.plan_width   = newcost_sort.plan_width;
		sort_node->plan.targetlist   = tlist_sort;
		for (i=0; i < sort_node->numCols; i++)
			sort_node->sortColIdx[i] = attr_maps[sort_node->sortColIdx[i] - 1];
		outerPlan(sort_node) = &cscan->scan.plan;
	}
#ifdef NOT_USED
	/*
	 * We don't adjust top-leve Agg-node because it mislead later
	 * decision to inject GpuSort, or not.
	 */
	agg->plan.startup_cost = newcost_agg.startup_cost;
	agg->plan.total_cost   = newcost_agg.total_cost;
	agg->plan.plan_rows    = newcost_agg.plan_rows;
	agg->plan.plan_width   = newcost_agg.plan_width;
#endif
	agg->plan.targetlist = agg_tlist;
	agg->plan.qual = agg_quals;
	agg->aggstrategy = new_agg_strategy;
	for (i=0; i < agg->numCols; i++)
		agg->grpColIdx[i] = attr_maps[agg->grpColIdx[i] - 1];
	foreach (lc, agg->chain)
	{
		Agg	   *subagg = lfirst(lc);

		for (i=0; i < subagg->numCols; i++)
			subagg->grpColIdx[i] = attr_maps[subagg->grpColIdx[i] - 1];
	}
}

bool
pgstrom_plan_is_gpupreagg(const Plan *plan)
{
	if (IsA(plan, CustomScan) &&
		((CustomScan *) plan)->methods == &gpupreagg_scan_methods)
		return true;
	return false;
}

static Node *
gpupreagg_create_scan_state(CustomScan *cscan)
{
	GpuPreAggState *gpas = palloc0(sizeof(GpuPreAggState));

	/* Set tag and executor callbacks */
	NodeSetTag(gpas, T_CustomScanState);
	gpas->gts.css.flags = cscan->flags;
	gpas->gts.css.methods = &gpupreagg_exec_methods;

	return (Node *) gpas;
}

static void
gpupreagg_begin(CustomScanState *node, EState *estate, int eflags)
{
	GpuContext	   *gcontext = NULL;
	GpuPreAggState *gpas = (GpuPreAggState *) node;
	PlanState	   *ps = &node->ss.ps;
	CustomScan	   *cscan = (CustomScan *) ps->plan;
	GpuPreAggInfo  *gpa_info = deform_gpupreagg_info(cscan);
	TupleDesc		tupdesc;
	double			outer_nitems;

	/* activate GpuContext for device execution */
	if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
		gcontext = pgstrom_get_gpucontext();
	/* common GpuTaskState setup */
	pgstrom_init_gputaskstate(gcontext, &gpas->gts, estate);
	gpas->gts.cb_task_process = gpupreagg_task_process;
	gpas->gts.cb_task_complete = gpupreagg_task_complete;
	gpas->gts.cb_task_release = gpupreagg_task_release;
	gpas->gts.cb_next_chunk = gpupreagg_next_chunk;
	gpas->gts.cb_next_tuple = gpupreagg_next_tuple;

	/*
	 * initialize own data source
	 */
	if (outerPlan(cscan))
	{
		PlanState  *outer_ps;

		Assert(cscan->scan.scanrelid == 0);
		outer_ps = ExecInitNode(outerPlan(cscan), estate, eflags);
		if (pgstrom_bulk_exec_supported(outer_ps))
		{
			((GpuTaskState *) outer_ps)->be_row_format = true;
			gpas->gts.outer_bulk_exec = true;
		}
		outerPlanState(gpas) = outer_ps;

		/* re-initialization of scan-descriptor and projection-info */
		tupdesc = ExecCleanTypeFromTL(cscan->custom_scan_tlist, false);
		ExecAssignScanType(&gpas->gts.css.ss, tupdesc);
		ExecAssignScanProjectionInfoWithVarno(&gpas->gts.css.ss, INDEX_VAR);
	}
	else
	{
		Assert(gpas->gts.css.ss.ss_currentRelation != NULL);
	}
	outer_nitems = gpa_info->outer_nitems;

	/*
	 * Setting up kernel program if it will run actually.
	 *
	 * NOTE: GpuPreAgg always takes kparam_0 for GPUPREAGG_FIELD_IS_* array.
	 */
	Assert(list_length(gpa_info->used_params) >= 1);
	pgstrom_assign_cuda_program(&gpas->gts,
								gpa_info->used_params,
								gpa_info->kern_source,
								gpa_info->extra_flags);
	if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
        pgstrom_load_cuda_program(&gpas->gts, true);

	/*
	 * init misc stuff
	 */
	gpas->safety_limit = gpa_info->safety_limit;
	gpas->key_dist_salt = gpa_info->key_dist_salt;

	if (gpa_info->numCols == 0)
		gpas->reduction_mode = GPUPREAGG_NOGROUP_REDUCTION;
	else if (gpa_info->num_groups < (gpuMaxThreadsPerBlock() / 4))
		gpas->reduction_mode = GPUPREAGG_LOCAL_REDUCTION;
	else if (gpa_info->num_groups < (outer_nitems / gpa_info->num_chunks) / 4)
		gpas->reduction_mode = GPUPREAGG_GLOBAL_REDUCTION;
	else
		gpas->reduction_mode = GPUPREAGG_FINAL_REDUCTION;

	gpas->outer_quals = (List *)
		ExecInitExpr((Expr *) gpa_info->outer_quals, ps);
	gpas->outer_overflow = NULL;
	gpas->outer_pds = NULL;
	gpas->curr_segment = NULL;

	/*
	 * init run-time statistics
	 *
	 * NOTE: It is initialized to the plan estimated values, then first
	 * segment overwrites them entirely, and second or later segment
	 * also update according to the weighted average manner.
	 */
	gpas->stat_num_segments = 0;
	gpas->stat_num_groups = gpa_info->num_groups;
	gpas->stat_num_chunks = gpa_info->num_chunks;
	gpas->stat_src_nitems = outer_nitems;
	gpas->stat_varlena_unitsz = gpa_info->varlena_unitsz;
	/* init perfmon */
	pgstrom_init_perfmon(&gpas->gts);
}

/*
 * gpupreagg_check_segment_capacity
 *
 * It checks capacity of the supplied segment based on plan estimation and
 * run-time statistics. If this segment may be capable to run one more chunk
 * at least, returns true. Elsewhere, it returns false, then caller has to
 * detach this segment as soon as possible.
 *
 * The delta of ngroups tells us how many groups will be newly produced by
 * the next chunks. Also, here may be some pending or running tasks to be
 * executed prior to the next chunk. So, we have to estimate amount of
 * resource consumption by the pending/running tasks, not only the next task.
 * If expexted resource consumption exceeds a dangerous level, we need to
 * switch the segment to the new one.
 */
static bool
gpupreagg_check_segment_capacity(GpuPreAggState *gpas,
								 gpupreagg_segment *segment)
{
	GpuTaskState   *gts = &gpas->gts;
	size_t			extra_ngroups;	/* expected ngroups consumption */
	size_t			extra_varlena;	/* expected varlena consumption */
	cl_uint			num_tasks;

	/* fetch the latest task state */
	SpinLockAcquire(&gts->lock);
	num_tasks = gts->num_running_tasks + gts->num_pending_tasks;
	SpinLockRelease(&gts->lock);

	if (segment->total_ngroups < gpas->stat_num_groups / 3 ||
		segment->total_ntasks < 4)
	{
		double	nrows_per_chunk = ((double) gpas->stat_src_nitems /
								   (double) gpas->stat_num_chunks);
		double	aggregate_ratio = ((double) gpas->stat_num_groups /
								   (double) gpas->stat_src_nitems);
		size_t	varlena_unitsz = ceil(gpas->stat_varlena_unitsz);

		extra_ngroups = (nrows_per_chunk *
						 aggregate_ratio * (double)(num_tasks + 1));
		extra_varlena = MAXALIGN(varlena_unitsz) * extra_ngroups;
	}
	else
	{
		size_t		delta_ngroups = Max(segment->delta_ngroups,
										segment->total_ngroups /
										segment->total_ntasks) + 1;
		size_t		delta_varlena = MAXALIGN(segment->total_varlena /
											 segment->total_ngroups);
		extra_ngroups = delta_ngroups * (num_tasks + 1);
		extra_varlena = delta_varlena * (num_tasks + 1);
	}
	/* check available noom of the kds_final */
	extra_ngroups = (double)(segment->total_ngroups +
							 extra_ngroups) * pgstrom_chunk_size_margin;
	if (extra_ngroups > segment->allocated_nrooms)
	{
		elog(DEBUG1,
			 "expected ngroups usage is larger than allocation: %zu of %zu",
			 (Size)extra_ngroups,
			 (Size)segment->allocated_nrooms);
		return false;
	}

	/* check available space of the varlena buffer */
	extra_varlena = (double)(segment->total_varlena +
							 extra_varlena) * pgstrom_chunk_size_margin;
	if (extra_varlena > segment->allocated_varlena)
	{
		elog(DEBUG1,
			 "expected varlena usage is larger than allocation: %zu of %zu",
			 (Size)extra_varlena,
			 (Size)segment->allocated_varlena);
		return false;
	}
	/* OK, we still have rooms for final reduction */
	return true;
}

/*
 * gpupreagg_create_segment
 *
 * It makes a segment and final result buffer.
 */
static gpupreagg_segment *
gpupreagg_create_segment(GpuPreAggState *gpas)
{
	GpuContext	   *gcontext = gpas->gts.gcontext;
	TupleDesc		tupdesc
		= gpas->gts.css.ss.ps.ps_ResultTupleSlot->tts_tupleDescriptor;
	Size			num_chunks;
	Size			f_nrooms;
	Size			varlena_length;
	Size			total_length;
	Size			offset;
	Size			required;
	double			reduction_ratio;
	cl_int			cuda_index;
	pgstrom_data_store *pds_final;
	gpupreagg_segment  *segment;

	/*
	 * (50% + plan/exec avg) x configured margin is scale of the final
	 * reduction buffer.
	 *
	 * NOTE: We ensure the final buffer has at least 2039 rooms to store
	 * the reduction results, because planner often estimate Ngroups
	 * too smaller than actual and it eventually leads destructive
	 * performance loss. Also, saving the KB scale memory does not affect
	 * entire resource consumption.
	 *
	 * NOTE: We also guarantee at least 1/2 of chunk size for minimum
	 * allocation size of the final result buffer.
	 */
	f_nrooms = (Size)(1.5 * gpas->stat_num_groups *
					  pgstrom_chunk_size_margin);
	/* minimum available nrooms? */
	f_nrooms = Max(f_nrooms, 2039);
	varlena_length = (Size)(gpas->stat_varlena_unitsz * (double) f_nrooms);

	/* minimum available buffer size? */
	total_length = (STROMALIGN(offsetof(kern_data_store,
										colmeta[tupdesc->natts])) +
					LONGALIGN((sizeof(Datum) +
							   sizeof(char)) * tupdesc->natts) * f_nrooms +
					varlena_length);
	if (total_length < pgstrom_chunk_size() / 2)
	{
		f_nrooms = (pgstrom_chunk_size() / 2 -
					STROMALIGN(offsetof(kern_data_store,
										colmeta[tupdesc->natts]))) /
			(LONGALIGN((sizeof(Datum) +
						sizeof(char)) * tupdesc->natts) +
			 gpas->stat_varlena_unitsz);
		varlena_length = (Size)(gpas->stat_varlena_unitsz * (double) f_nrooms);
	}

	/*
	 * FIXME: At this moment, we have a hard limit (2GB) for temporary
	 * consumption by pds_src[] chunks. It shall be configurable and
	 * controled under GpuContext.
	 */
	num_chunks = 0x80000000UL / pgstrom_chunk_size();
	reduction_ratio = Max(gpas->stat_src_nitems /
						  gpas->stat_num_chunks, 1.0) /* nrows per chunk */
		/ (gpas->stat_num_groups * gpas->key_dist_salt);
	num_chunks = Min(num_chunks, reduction_ratio * gpas->safety_limit);
	num_chunks = Max(num_chunks, 1);	/* at least one chunk */

	/*
	 * Any tasks that share same segment has to be kicked on the same
	 * GPU device. At this moment, we don't support multiple device
	 * mode to process GpuPreAgg. It's a TODO.
	 */
	cuda_index = gcontext->next_context++ % gcontext->num_context;

	/* pds_final buffer */
	pds_final = PDS_create_slot(gcontext,
								tupdesc,
								f_nrooms,
								varlena_length,
								true);
	/* gpupreagg_segment itself */
	offset = STROMALIGN(sizeof(gpupreagg_segment));
	required = offset +
		STROMALIGN(sizeof(pgstrom_data_store *) * num_chunks) +
		STROMALIGN(sizeof(CUevent) * num_chunks);
	segment = MemoryContextAllocZero(gcontext->memcxt, required);
	segment->gpas = gpas;
	segment->pds_final = pds_final;		/* refcnt==1 */
	segment->m_kds_final = 0UL;
	segment->m_hashslot_final = 0UL;

	/*
	 * FIXME: f_hashsize is nroom of the pds_final. It is not a reasonable
	 * estimation, thus needs to be revised.
	 */
	segment->f_hashsize = f_nrooms;
	segment->num_chunks = num_chunks;
	segment->idx_chunks = 0;
	segment->refcnt = 1;
	segment->cuda_index = cuda_index;
	segment->has_terminator = false;
	segment->needs_fallback = false;
	segment->pds_src = (pgstrom_data_store **)((char *)segment + offset);
	offset += STROMALIGN(sizeof(pgstrom_data_store *) * num_chunks);
	segment->ev_kern_main = (CUevent *)((char *)segment + offset);

	/* run-time statistics */
	segment->allocated_nrooms = f_nrooms;
	segment->allocated_varlena = varlena_length;
	segment->total_ntasks = 0;
	segment->total_nitems = 0;
	segment->total_ngroups = 0;
	segment->total_varlena = 0;

	return segment;
}

/*
 * gpupreagg_cleanup_segment - release relevant CUDA resources
 */
static void
gpupreagg_cleanup_segment(gpupreagg_segment *segment)
{
	GpuContext *gcontext = segment->gpas->gts.gcontext;
	CUresult	rc;
	cl_int		i;

	if (segment->m_hashslot_final != 0UL)
	{
		__gpuMemFree(gcontext, segment->cuda_index,
					 segment->m_hashslot_final);
		segment->m_hashslot_final = 0UL;
		segment->m_kds_final = 0UL;
	}

	if (segment->ev_final_loaded)
	{
		rc = cuEventDestroy(segment->ev_final_loaded);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuEventDestroy: %s", errorText(rc));
		segment->ev_final_loaded = NULL;
	}

	for (i=0; i < segment->num_chunks; i++)
	{
		if (segment->ev_kern_main[i])
		{
			rc = cuEventDestroy(segment->ev_kern_main[i]);
			if (rc != CUDA_SUCCESS)
				elog(WARNING, "failed on cuEventDestroy: %s", errorText(rc));
			segment->ev_kern_main[i] = NULL;
		}
	}
}

/*
 * gpupreagg_get_segment
 */
static gpupreagg_segment *
gpupreagg_get_segment(gpupreagg_segment *segment)
{
	Assert(segment->refcnt > 0);
	segment->refcnt++;
	return segment;
}

/*
 * gpupreagg_put_segment
 */
static void
gpupreagg_put_segment(gpupreagg_segment *segment)
{
	int			i;
	double		n;

	Assert(segment->refcnt > 0);

	if (--segment->refcnt == 0)
	{
		GpuPreAggState *gpas = segment->gpas;

		/* update statistics */
		if (!segment->needs_fallback)
		{
			/*
			 * Unless GpuPreAggState does not have very restrictive (but
			 * nobody knows) outer_quals, GpuPreAgg operations shall have
			 * at least one results.
			 */
			Assert(gpas->outer_quals != NIL || segment->total_ngroups > 0);
			n = ++gpas->stat_num_segments;
			gpas->stat_num_groups =
				((double)gpas->stat_num_groups * (n-1) +
				 (double)segment->total_ngroups) / n;
			gpas->stat_num_chunks =
				((double)gpas->stat_num_chunks * (n-1) +
				 (double)segment->total_ntasks) / n;
			gpas->stat_src_nitems =
				((double)gpas->stat_src_nitems * (n-1) +
				 (double)segment->total_nitems) / n;
			if (segment->total_ngroups > 0)
			{
				gpas->stat_varlena_unitsz =
					((double)gpas->stat_varlena_unitsz * (n-1) +
					 (double)segment->total_varlena /
					 (double)segment->total_ngroups) / n;
			}
		}
		/* unless error path or fallback, it shall be released already */
		gpupreagg_cleanup_segment(segment);

		if (segment->pds_final)
		{
			PDS_release(segment->pds_final);
			segment->pds_final = NULL;
		}

		for (i=0; i < segment->num_chunks; i++)
		{
			if (segment->pds_src[i] != NULL)
			{
				PDS_release(segment->pds_src[i]);
				segment->pds_src[i] = NULL;
			}
		}
		pfree(segment);
	}
}

static bool
gpupreagg_setup_segment(pgstrom_gpupreagg *gpreagg, bool perfmon_enabled)
{
	gpupreagg_segment  *segment = gpreagg->segment;
	pgstrom_data_store *pds_final = segment->pds_final;
	size_t				length;
	size_t				grid_size;
	size_t				block_size;
	CUfunction			kern_final_prep;
	CUdeviceptr			m_hashslot_final;
	CUdeviceptr			m_kds_final;
	CUevent				ev_final_loaded;
	CUresult			rc;
	cl_int				i;
	void			   *kern_args[4];

	if (!segment->ev_final_loaded)
	{
		/* Lookup gpupreagg_final_preparation */
		rc = cuModuleGetFunction(&kern_final_prep,
								 gpreagg->task.cuda_module,
								 "gpupreagg_final_preparation");
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

		/* Device memory allocation for kds_final and final hashslot */
		length = (GPUMEMALIGN(offsetof(kern_global_hashslot,
									   hash_slot[segment->f_hashsize])) +
				  GPUMEMALIGN(KERN_DATA_STORE_LENGTH(pds_final->kds)));
		m_hashslot_final = gpuMemAlloc(&gpreagg->task, length);
		if (!m_hashslot_final)
			return false;
		m_kds_final = m_hashslot_final +
			GPUMEMALIGN(offsetof(kern_global_hashslot,
								 hash_slot[segment->f_hashsize]));

		/* Create an event object to synchronize setup of this segment */
		rc = cuEventCreate(&ev_final_loaded, CU_EVENT_DISABLE_TIMING);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

		/* Create event object to synchronize every kernel execution end */
		for (i=0; i < segment->num_chunks; i++)
		{
			Assert(segment->ev_kern_main[i] == NULL);
			rc = cuEventCreate(&segment->ev_kern_main[i],
							   perfmon_enabled ? 0 : CU_EVENT_DISABLE_TIMING);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));
		}

		/* enqueue DMA send request */
		rc = cuMemcpyHtoDAsync(m_kds_final, pds_final->kds,
							   KERN_DATA_STORE_HEAD_LENGTH(pds_final->kds),
							   gpreagg->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));

		/* Launch:
		 * KERNEL_FUNCTION(void)
		 * gpupreagg_final_preparation(size_t hash_size,
		 *                             kern_global_hashslot *f_hashslot)
		 */
		optimal_workgroup_size(&grid_size,
							   &block_size,
							   kern_final_prep,
							   gpreagg->task.cuda_device,
							   segment->f_hashsize,
							   0, sizeof(kern_errorbuf));
		kern_args[0] = &segment->f_hashsize;
		kern_args[1] = &m_hashslot_final;
		rc = cuLaunchKernel(kern_final_prep,
							grid_size, 1, 1,
							block_size, 1, 1,
							sizeof(kern_errorbuf) * block_size,
							gpreagg->task.cuda_stream,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));

		/* DMA send synchronization of final buffer */
		rc = cuEventRecord(ev_final_loaded, gpreagg->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuStreamWaitEvent: %s", errorText(rc));

		segment->m_hashslot_final = m_hashslot_final;
		segment->m_kds_final = m_kds_final;
		segment->ev_final_loaded = ev_final_loaded;
	}
	else
	{
		/* DMA Send synchronization, kicked by other task */
		ev_final_loaded = segment->ev_final_loaded;
		rc = cuStreamWaitEvent(gpreagg->task.cuda_stream, ev_final_loaded, 0);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuStreamWaitEvent: %s", errorText(rc));
	}
	return true;
}

static pgstrom_gpupreagg *
gpupreagg_task_create(GpuPreAggState *gpas,
					  pgstrom_data_store *pds_in,
					  gpupreagg_segment *segment,
					  bool is_terminator)
{
	GpuContext		   *gcontext = gpas->gts.gcontext;
	pgstrom_gpupreagg  *gpreagg;
	TupleDesc			tupdesc;
	size_t				nitems = pds_in->kds->nitems;
	Size				length;

	/* allocation of pgtrom_gpupreagg */
	tupdesc = gpas->gts.css.ss.ps.ps_ResultTupleSlot->tts_tupleDescriptor;
	length = (STROMALIGN(offsetof(pgstrom_gpupreagg, kern.kparams) +
						 gpas->gts.kern_params->length) +
			  STROMALIGN(offsetof(kern_resultbuf, results[0])) +
			  STROMALIGN(offsetof(kern_data_store,
								  colmeta[tupdesc->natts])));
	gpreagg = MemoryContextAllocZero(gcontext->memcxt, length);

	/* initialize GpuTask object */
	pgstrom_init_gputask(&gpas->gts, &gpreagg->task);
	/* NOTE: gpreagg task has to perform on the GPU device
	 * where the segment is located on. */
	gpreagg->task.cuda_index = segment->cuda_index;
	gpreagg->segment = segment;	/* caller already acquired */
	gpreagg->is_terminator = is_terminator;

	/*
	 * FIXME: If num_groups is larger than expectation, we may need to
	 * change the reduction policy on run-time
	 */
	gpreagg->num_groups = gpas->stat_num_groups;
	gpreagg->pds_in = pds_in;

	/* also initialize kern_gpupreagg portion */
	gpreagg->kern.reduction_mode = gpas->reduction_mode;
	memset(&gpreagg->kern.kerror, 0, sizeof(kern_errorbuf));
	gpreagg->kern.key_dist_salt = gpas->key_dist_salt;
	gpreagg->kern.hash_size = nitems;
	memcpy(gpreagg->kern.pg_crc32_table,
		   pg_crc32_table,
		   sizeof(uint32) * 256);
	/* kern_parambuf */
	memcpy(KERN_GPUPREAGG_PARAMBUF(&gpreagg->kern),
		   gpas->gts.kern_params,
		   gpas->gts.kern_params->length);
	/* offset of kern_resultbuf */
	gpreagg->kern.kresults_1_offset
		= STROMALIGN(offsetof(kern_gpupreagg, kparams) +
					 gpas->gts.kern_params->length);
	gpreagg->kern.kresults_2_offset
		= STROMALIGN(gpreagg->kern.kresults_1_offset +
					 offsetof(kern_resultbuf, results[nitems]));
	/* kds_head - template of intermediation buffer */
	gpreagg->kds_head = (kern_data_store *)
		((char *)KERN_GPUPREAGG_PARAMBUF(&gpreagg->kern) +
		 KERN_GPUPREAGG_PARAMBUF_LENGTH(&gpreagg->kern));
	length = (STROMALIGN(offsetof(kern_data_store,
								  colmeta[tupdesc->natts])) +
			  STROMALIGN(LONGALIGN((sizeof(Datum) + sizeof(char)) *
								   tupdesc->natts) * nitems));
	init_kernel_data_store(gpreagg->kds_head,
						   tupdesc,
						   length,
						   KDS_FORMAT_SLOT,
						   nitems,
						   true);
	return gpreagg;
}

static GpuTask *
gpupreagg_next_chunk(GpuTaskState *gts)
{
	GpuContext		   *gcontext = gts->gcontext;
	GpuPreAggState	   *gpas = (GpuPreAggState *) gts;
	PlanState		   *subnode = outerPlanState(gpas);
	pgstrom_data_store *pds = NULL;
	pgstrom_gpupreagg  *gpreagg;
	gpupreagg_segment  *segment;
	cl_uint				segment_id;
	bool				is_terminator = false;
	struct timeval		tv1, tv2;

	if (gpas->gts.scan_done)
		return NULL;

	PERFMON_BEGIN(&gts->pfm, &tv1);
	if (gpas->gts.css.ss.ss_currentRelation)
	{
		/* Load a bunch of records at once on the first time */
		if (!gpas->outer_pds)
			gpas->outer_pds = pgstrom_exec_scan_chunk(&gpas->gts,
													  pgstrom_chunk_size());
		/* Picks up the cached one to detect the final chunk */
		pds = gpas->outer_pds;
		if (!pds)
			pgstrom_deactivate_gputaskstate(&gpas->gts);
		else
			gpas->outer_pds = pgstrom_exec_scan_chunk(&gpas->gts,
													  pgstrom_chunk_size());
		/* Any more chunk expected? */
		if (!gpas->outer_pds)
			is_terminator = true;
	}
	else if (!gpas->gts.outer_bulk_exec)
	{
		/* Scan the outer relation using row-by-row mode */
		TupleDesc		tupdesc
			= subnode->ps_ResultTupleSlot->tts_tupleDescriptor;

		while (true)
		{
			TupleTableSlot *slot;

			if (HeapTupleIsValid(gpas->outer_overflow))
			{
				slot = gpas->outer_overflow;
				gpas->outer_overflow = NULL;
			}
			else
			{
				slot = ExecProcNode(subnode);
				if (TupIsNull(slot))
				{
					pgstrom_deactivate_gputaskstate(&gpas->gts);
					break;
				}
			}

			if (!pds)
				pds = PDS_create_row(gcontext,
									 tupdesc,
									 pgstrom_chunk_size());
			/* insert a tuple to the data-store */
			if (!PDS_insert_tuple(pds, slot))
			{
				gpas->outer_overflow = slot;
				break;
			}
		}
		/* Any more tuples expected? */
		if (!gpas->outer_overflow)
			is_terminator = true;
	}
	else
	{
		/* Load a bunch of records at once on the first time */
		if (!gpas->outer_pds)
			gpas->outer_pds = BulkExecProcNode((GpuTaskState *)subnode,
											   pgstrom_chunk_size());
		/* Picks up the cached one to detect the final chunk */
		pds = gpas->outer_pds;
		if (!pds)
			pgstrom_deactivate_gputaskstate(&gpas->gts);
		else
			gpas->outer_pds = BulkExecProcNode((GpuTaskState *)subnode,
											   pgstrom_chunk_size());
		/* Any more chunk expected? */
		if (!gpas->outer_pds)
			is_terminator = true;
	}
	PERFMON_END(&gts->pfm, time_outer_load, &tv1, &tv2);

	if (!pds)
		return NULL;	/* no more tuples to read */

	/*
	 * Create or acquire a segment that has final result buffer of this
	 * GpuPreAgg task.
	 */
retry_segment:
	if (!gpas->curr_segment)
		gpas->curr_segment = gpupreagg_create_segment(gpas);
	segment = gpupreagg_get_segment(gpas->curr_segment);
	/*
	 * Once a segment gets terminator task, it also means the segment is
	 * not capable to add new tasks/chunks any more. In the more urgent
	 * case, CUDA's callback set 'needs_fallback' to inform the final
	 * reduction buffer has no space to process groups.
	 * So, we have to switch to new segment immediately.
	 */
	pg_memory_barrier();	/* CUDA callback may set needs_fallback */
	if (segment->has_terminator || segment->needs_fallback)
	{
		gpupreagg_put_segment(segment);
		/* GpuPreAggState also unreference this segment */
		gpupreagg_put_segment(gpas->curr_segment);
		gpas->curr_segment = NULL;
		goto retry_segment;
	}

	/*
	 * OK, assign this PDS on the segment
	 */
	Assert(segment->idx_chunks < segment->num_chunks);
	segment_id = segment->idx_chunks++;
	segment->pds_src[segment_id] = PDS_retain(pds);
	if (segment->idx_chunks == segment->num_chunks)
		is_terminator = true;
	gpreagg = gpupreagg_task_create(gpas, pds, segment, is_terminator);
	gpreagg->segment_id = segment_id;

	if (is_terminator)
	{
		Assert(!segment->has_terminator);
		segment->has_terminator = true;

		gpupreagg_put_segment(gpas->curr_segment);
		gpas->curr_segment = NULL;
	}
	return &gpreagg->task;
}

/*
 * gpupreagg_next_tuple_fallback - a fallback routine if GPU returned
 * StromError_CpuReCheck, to suggest the backend to handle request
 * by itself. A fallback process looks like construction of special
 * partial aggregations that consist of individual rows; so here is
 * no performance benefit once it happen.
 */
static TupleTableSlot *
gpupreagg_next_tuple_fallback(GpuPreAggState *gpas, gpupreagg_segment *segment)
{
	TupleTableSlot	   *slot = gpas->gts.css.ss.ss_ScanTupleSlot;
	ExprContext		   *econtext = gpas->gts.css.ss.ps.ps_ExprContext;
	pgstrom_data_store *pds;
	cl_uint				row_index;
	HeapTupleData		tuple;

retry:
	if (segment->idx_chunks == 0)
		return NULL;
	pds = segment->pds_src[segment->idx_chunks - 1];
	Assert(pds != NULL);

	row_index = gpas->gts.curr_index++;

	/*
     * Fetch a tuple from the data-store
     */
	ExecClearTuple(slot);
	if (!pgstrom_fetch_data_store(slot, pds, row_index, &tuple))
	{
		PDS_release(pds);
		segment->pds_src[segment->idx_chunks - 1] = NULL;
		segment->idx_chunks--;
		gpas->gts.curr_index = 0;
		goto retry;
	}
	econtext->ecxt_scantuple = slot;

	/*
	 * Filter out the tuple, if any outer_quals
	 */
	if (gpas->outer_quals != NULL &&
		!ExecQual(gpas->outer_quals, econtext, false))
		goto retry;

	/*
	 * Projection from scan-tuple to result-tuple, if any
	 */
	if (gpas->gts.css.ss.ps.ps_ProjInfo != NULL)
	{
		ExprDoneCond	is_done;

		slot = ExecProject(gpas->gts.css.ss.ps.ps_ProjInfo, &is_done);
		if (is_done == ExprEndResult)
			goto retry;
	}
	return slot;
}

static TupleTableSlot *
gpupreagg_next_tuple(GpuTaskState *gts)
{
	GpuPreAggState	   *gpas = (GpuPreAggState *) gts;
	pgstrom_gpupreagg  *gpreagg = (pgstrom_gpupreagg *) gpas->gts.curr_task;
	gpupreagg_segment  *segment = gpreagg->segment;
	pgstrom_data_store *pds_final = segment->pds_final;
	kern_data_store	   *kds_final = pds_final->kds;
	TupleTableSlot	   *slot = NULL;
	HeapTupleData		tuple;
	struct timeval		tv1, tv2;

	Assert(gpreagg->is_terminator);

	PERFMON_BEGIN(&gts->pfm, &tv1);
	if (segment->needs_fallback)
		slot = gpupreagg_next_tuple_fallback(gpas, segment);
	else if (gpas->gts.curr_index < kds_final->nitems)
	{
		size_t		index = gpas->gts.curr_index++;

		slot = gts->css.ss.ps.ps_ResultTupleSlot;
		ExecClearTuple(slot);
		if (!pgstrom_fetch_data_store(slot, pds_final, index, &tuple))
		{
			elog(NOTICE, "Bug? empty slot was specified by kern_resultbuf");
			slot = NULL;
		}
	}
	PERFMON_END(&gts->pfm, time_materialize, &tv1, &tv2);
	return slot;
}

static TupleTableSlot *
gpupreagg_exec(CustomScanState *node)
{
	return pgstrom_exec_gputask((GpuTaskState *) node);
}

static void
gpupreagg_end(CustomScanState *node)
{
	GpuPreAggState	   *gpas = (GpuPreAggState *) node;

	/* Release current segment if any */
	if (gpas->curr_segment)
	{
		gpupreagg_put_segment(gpas->curr_segment);
		gpas->curr_segment = NULL;
	}
	/* Clean up subtree, if any */
	if (outerPlanState(node))
		ExecEndNode(outerPlanState(node));
	/* Cleanup and relase any concurrent tasks */
	pgstrom_release_gputaskstate(&gpas->gts);
}

static void
gpupreagg_rescan(CustomScanState *node)
{
	GpuPreAggState	   *gpas = (GpuPreAggState *) node;

	/* inform this GpuTaskState will produce more rows, prior to cleanup */
	pgstrom_activate_gputaskstate(&gpas->gts);
	/* Cleanup and relase any concurrent tasks */
	pgstrom_cleanup_gputaskstate(&gpas->gts);
	/* Rewind the subtree */
	if (gpas->gts.css.ss.ss_currentRelation)
		pgstrom_rewind_scan_chunk(&gpas->gts);
	else
		ExecReScan(outerPlanState(node));
}

static void
gpupreagg_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
	GpuPreAggState *gpas = (GpuPreAggState *) node;
	CustomScan	   *cscan = (CustomScan *) node->ss.ps.plan;
	GpuPreAggInfo  *gpa_info = deform_gpupreagg_info(cscan);
	List		   *context;
	List		   *dev_proj = NIL;
	ListCell	   *lc;
	const char	   *policy;
	char			temp[2048];

	if (gpas->reduction_mode == GPUPREAGG_NOGROUP_REDUCTION)
		policy = "NoGroup";
	else if (gpas->reduction_mode == GPUPREAGG_LOCAL_REDUCTION)
		policy = "Local + Global";
	else if (gpas->reduction_mode == GPUPREAGG_GLOBAL_REDUCTION)
		policy = "Global";
	else if (gpas->reduction_mode == GPUPREAGG_FINAL_REDUCTION)
		policy = "Only Final";
	else
		policy = "Unknown";
	ExplainPropertyText("Reduction", policy, es);

	/* Set up deparsing context */
	context = set_deparse_context_planstate(es->deparse_cxt,
                                            (Node *)&gpas->gts.css.ss.ps,
                                            ancestors);
	/* Show device projection */
	foreach (lc, gpa_info->tlist_dev)
		dev_proj = lappend(dev_proj, ((TargetEntry *) lfirst(lc))->expr);
	pgstrom_explain_expression(dev_proj, "GPU Projection",
							   &gpas->gts.css.ss.ps, context,
							   ancestors, es, false, false);
	/* statistics for outer scan, if it was pulled-up */
	pgstrom_explain_outer_bulkexec(&gpas->gts, context, ancestors, es);
	/* Show device filter */
	pgstrom_explain_expression(gpa_info->outer_quals, "GPU Filter",
							   &gpas->gts.css.ss.ps, context,
							   ancestors, es, false, true);
	// TODO: Add number of rows filtered by the device side

	if (es->verbose)
	{
		snprintf(temp, sizeof(temp),
				 "SafetyLimit: %u, KeyDistSalt: %u",
				 gpas->safety_limit,
				 gpas->key_dist_salt);
		ExplainPropertyText("Logic Parameter", temp, es);
	}
	pgstrom_explain_gputaskstate(&gpas->gts, es);
}

/*
 * gpupreagg_cleanup_cuda_resources
 *
 * handler to release all the cuda resource, but gpreagg is still retained
 */
static void
gpupreagg_cleanup_cuda_resources(pgstrom_gpupreagg *gpreagg)
{
	if (gpreagg->m_gpreagg)
		gpuMemFree(&gpreagg->task, gpreagg->m_gpreagg);

	CUDA_EVENT_DESTROY(gpreagg, ev_dma_send_start);
	CUDA_EVENT_DESTROY(gpreagg, ev_dma_send_stop);
	CUDA_EVENT_DESTROY(gpreagg, ev_kern_fixvar);
	CUDA_EVENT_DESTROY(gpreagg, ev_dma_recv_start);
	CUDA_EVENT_DESTROY(gpreagg, ev_dma_recv_stop);

	/* clear the pointers */
	gpreagg->kern_main = NULL;
	gpreagg->kern_fixvar = NULL;
	gpreagg->m_gpreagg = 0UL;
	gpreagg->m_kds_row = 0UL;
	gpreagg->m_kds_slot = 0UL;
	gpreagg->m_ghash = 0UL;
}

static void
gpupreagg_task_release(GpuTask *gtask)
{
	pgstrom_gpupreagg  *gpreagg = (pgstrom_gpupreagg *) gtask;

	/* cleanup cuda resources, if any */
	gpupreagg_cleanup_cuda_resources(gpreagg);

	if (gpreagg->pds_in)
		PDS_release(gpreagg->pds_in);
	if (gpreagg->segment)
		gpupreagg_put_segment(gpreagg->segment);
	pfree(gpreagg);
}

/*
 * gpupreagg_task_complete
 */
static bool
gpupreagg_task_complete(GpuTask *gtask)
{
	pgstrom_gpupreagg  *gpreagg = (pgstrom_gpupreagg *) gtask;
	gpupreagg_segment  *segment = gpreagg->segment;
	GpuPreAggState	   *gpas = (GpuPreAggState *) gtask->gts;
	pgstrom_perfmon	   *pfm = &gpas->gts.pfm;
	cl_uint				nitems_in = gpreagg->pds_in->kds->nitems;

	if (pfm->enabled)
	{
		CUevent			ev_kern_main;

		pfm->num_tasks++;

		CUDA_EVENT_ELAPSED(gpreagg, time_dma_send,
						   gpreagg->ev_dma_send_start,
						   gpreagg->ev_dma_send_stop,
						   skip);
		ev_kern_main = segment->ev_kern_main[gpreagg->segment_id];
		CUDA_EVENT_ELAPSED(gpreagg, gpreagg.tv_kern_main,
						   gpreagg->ev_dma_send_stop,
						   ev_kern_main,
						   skip);
		if (gpreagg->is_terminator)
		{
			pgstrom_data_store *pds_final = segment->pds_final;

			if (pds_final->kds->has_notbyval)
			{
				CUDA_EVENT_ELAPSED(gpreagg, gpreagg.tv_kern_fixvar,
								   gpreagg->ev_kern_fixvar,
								   gpreagg->ev_dma_recv_start,
								   skip);
			}
		}
		CUDA_EVENT_ELAPSED(gpreagg, time_dma_recv,
						   gpreagg->ev_dma_recv_start,
						   gpreagg->ev_dma_recv_stop,
						   skip);

		if (gpreagg->kern.pfm.num_kern_prep > 0)
		{
			pfm->gpreagg.num_kern_prep += gpreagg->kern.pfm.num_kern_prep;
			pfm->gpreagg.tv_kern_prep += gpreagg->kern.pfm.tv_kern_prep;
		}
		if (gpreagg->kern.pfm.num_kern_nogrp > 0)
		{
			pfm->gpreagg.num_kern_nogrp += gpreagg->kern.pfm.num_kern_nogrp;
			pfm->gpreagg.tv_kern_nogrp += gpreagg->kern.pfm.tv_kern_nogrp;
		}
		if (gpreagg->kern.pfm.num_kern_lagg > 0)
		{
			pfm->gpreagg.num_kern_lagg += gpreagg->kern.pfm.num_kern_lagg;
			pfm->gpreagg.tv_kern_lagg += gpreagg->kern.pfm.tv_kern_lagg;
		}
		if (gpreagg->kern.pfm.num_kern_gagg > 0)
		{
			pfm->gpreagg.num_kern_gagg += gpreagg->kern.pfm.num_kern_gagg;
			pfm->gpreagg.tv_kern_gagg += gpreagg->kern.pfm.tv_kern_gagg;
		}
		if (gpreagg->kern.pfm.num_kern_fagg > 0)
		{
			pfm->gpreagg.num_kern_fagg += gpreagg->kern.pfm.num_kern_fagg;
			pfm->gpreagg.tv_kern_fagg += gpreagg->kern.pfm.tv_kern_fagg;
		}
	}
skip:
	/* OK, CUDA resource of this task is no longer referenced */
	gpupreagg_cleanup_cuda_resources(gpreagg);

	/*
	 * Collection of run-time statistics
	 */
	elog(DEBUG1,
		 "chunk: %d, nitems: %u, ngroups: %u, vl_usage: %u, "
		 "gconflicts: %u, fconflicts: %u",
		 gpreagg->segment_id,
		 nitems_in,
		 gpreagg->kern.num_groups,
		 gpreagg->kern.varlena_usage,
		 gpreagg->kern.ghash_conflicts,
		 gpreagg->kern.fhash_conflicts);
	segment->total_ntasks++;
	segment->total_nitems += nitems_in;
	segment->total_ngroups += gpreagg->kern.num_groups;
	segment->total_varlena += gpreagg->kern.varlena_usage;
	segment->delta_ngroups = gpreagg->kern.num_groups;

	if (!gpupreagg_check_segment_capacity(gpas, segment))
	{
		/*
		 * NOTE: If and when above logic expects the segment will be filled
		 * up in the near future, best strategy is to terminate the segment
		 * as soon as possible, to avoid CPU fallback that throws away all
		 * the previous works.
		 *
		 * If backend attached no terminator task yet, this GpuPreAgg task
		 * will become the terminator - which shall synchronize completion
		 * of the other concurrent tasks in the same segment, launch the
		 * gpupreagg_fixup_varlena kernel, then receive the contents of
		 * the final-kds.
		 * Once 'reduction_mode' is changed to GPUPREAGG_ONLY_TERMINATION,
		 * it does not run any reduction job, but only termination.
		 *
		 * (BUG#0219) 
		 * We have to re-enqueue the urgent terminator task at end of the
		 * pending_list, to ensure gpupreagg_fixup_varlena() kernel shall
		 * be launched after completion of all the reduction tasks because
		 * it rewrite device pointer by equivalent host pointer - it leads
		 * unexpected kernel crash.
		 * cuEventCreate() creates an event object, however, at this point,
		 * cuStreamWaitEvent() does not block the stream because this event
		 * object records nothing thus it is considered not to block others.
		 * So, host code has to ensure the terminator task shall be processed
		 * later than others.
		 * Once a segment->has_terminator is set, no other tasks shall not
		 * be added, so what we have to do is use dlist_push_tail to
		 * re-enqueue the pending list.
		 */
		if (!segment->has_terminator)
		{
			gpupreagg_segment  *segment = gpreagg->segment;

			elog(NOTICE, "GpuPreAgg urgent termination: segment %p (ngroups %zu of %zu, extra %s of %s, ntasks %d or %d) by gpupreagg (id=%d)",
				 segment,
				 segment->total_ngroups, segment->allocated_nrooms,
				 format_bytesz(segment->total_varlena),
				 format_bytesz(segment->allocated_varlena),
				 (int)segment->total_ntasks, segment->idx_chunks,
				 gpreagg->segment_id);
			Assert(!gpreagg->is_terminator);
			gpreagg->is_terminator = true;
			gpreagg->kern.reduction_mode = GPUPREAGG_ONLY_TERMINATION;
			/* clear the statistics */
			gpreagg->kern.num_groups = 0;
			gpreagg->kern.varlena_usage = 0;
			gpreagg->kern.ghash_conflicts = 0;
			gpreagg->kern.fhash_conflicts = 0;
			/* ok, this segment get a terminator */
			segment->has_terminator = true;

			/* let's enqueue the task again */
			SpinLockAcquire(&gpas->gts.lock);
			dlist_push_tail(&gpas->gts.pending_tasks, &gpreagg->task.chain);
			gpas->gts.num_pending_tasks++;
			SpinLockRelease(&gpas->gts.lock);

			return false;
		}
	}

	/*
	 * NOTE: We have to ensure that a segment has terminator task, even if
	 * not attached yet. Also we have to pay attention that no additional
	 * task shall be added to the segment once 'needs_fallback' gets set
	 * (it might be set by CUDA callback).
	 * So, this segment may have no terminator task even though it has
	 * responsible to generate result.
	 */
	if (segment->needs_fallback)
	{
		if (pgstrom_cpu_fallback_enabled &&
			(gpreagg->task.kerror.errcode == StromError_CpuReCheck ||
			 gpreagg->task.kerror.errcode == StromError_DataStoreNoSpace))
			memset(&gpreagg->task.kerror, 0, sizeof(kern_errorbuf));

		/*
		 * Someone has to be segment terminator even if it is not assigned
		 * yet. Once needs_fallback is set, no task shall be added any more.
		 * So, this task will perform as segment terminator.
		 */
		if (!segment->has_terminator)
		{
			gpreagg->is_terminator = true;
			segment->has_terminator = true;
		}
	}
	else if (gpreagg->is_terminator)
	{
		/*
		 * Completion of the terminator task without 'needs_fallback' means
		 * no other GPU kernels are in-progress. So, we can release relevant
		 * CUDA resource immediately.
		 *
		 * TODO: We might be able to release CUDA resource sooner even if
		 * 'needs_fallback' scenario. However, at this moment, we have no
		 * mechanism to track concurrent tasks that may be launched
		 * asynchronously. So, we move on the safety side.
		 */
		gpupreagg_cleanup_segment(segment);
	}

	/*
	 * Only terminator task shall be returned to the main logic.
	 * Elsewhere, task shall be no longer referenced thus we can release
	 * relevant buffer immediately as if nothing were returned.
	 */
	if (!gpreagg->is_terminator &&
		gpreagg->task.kerror.errcode == StromError_Success)
	{
		/* detach from the task tracking list */
		SpinLockAcquire(&gpas->gts.lock);
		dlist_delete(&gpreagg->task.tracker);
		memset(&gpreagg->task.tracker, 0, sizeof(dlist_node));
		SpinLockRelease(&gpas->gts.lock);
		/* then release the task immediately */
		gpupreagg_task_release(&gpreagg->task);
		return false;
	}
	return true;
}

/*
 * gpupreagg_task_respond
 */
static void
gpupreagg_task_respond(CUstream stream, CUresult status, void *private)
{
	pgstrom_gpupreagg  *gpreagg = (pgstrom_gpupreagg *) private;
	gpupreagg_segment  *segment = gpreagg->segment;
	GpuTaskState	   *gts = gpreagg->task.gts;

	/* See comments in pgstrom_respond_gpuscan() */
	if (status == CUDA_ERROR_INVALID_CONTEXT || !IsTransactionState())
		return;

	if (status == CUDA_SUCCESS)
		gpreagg->task.kerror = gpreagg->kern.kerror;
	else
	{
		gpreagg->task.kerror.errcode = status;
		gpreagg->task.kerror.kernel = StromKernel_CudaRuntime;
		gpreagg->task.kerror.lineno = 0;
	}

	/*
	 * Set fallback flag if GPU kernel required CPU fallback to process
	 * this segment. Also, it means no more tasks can be added any more,
	 * so we don't want to wait for invocation of complete callback above.
	 *
	 * NOTE: We may have performance advantage if segment was retried
	 * with larger final reduction buffer. But not yet.
	 */
	if (gpreagg->task.kerror.errcode == StromError_CpuReCheck ||
		gpreagg->task.kerror.errcode == StromError_DataStoreNoSpace)
		segment->needs_fallback = true;

	/*
	 * Remove the GpuTask from the running_tasks list, and attach it
	 * on the completed_tasks list again. Note that this routine may
	 * be called by CUDA runtime, prior to attachment of GpuTask on
	 * the running_tasks by cuda_control.c.
	 */
	SpinLockAcquire(&gts->lock);
	if (gpreagg->task.chain.prev && gpreagg->task.chain.next)
	{
		dlist_delete(&gpreagg->task.chain);
		gts->num_running_tasks--;
	}
	if (gpreagg->task.kerror.errcode == StromError_Success)
		dlist_push_tail(&gts->completed_tasks, &gpreagg->task.chain);
	else
		dlist_push_head(&gts->completed_tasks, &gpreagg->task.chain);
	gts->num_completed_tasks++;
	SpinLockRelease(&gts->lock);

	SetLatch(&MyProc->procLatch);
}

/*
 * gpupreagg_task_process
 */
static bool
__gpupreagg_task_process(pgstrom_gpupreagg *gpreagg)
{
	pgstrom_data_store *pds_in = gpreagg->pds_in;
	kern_data_store	   *kds_head = gpreagg->kds_head;
	gpupreagg_segment  *segment = gpreagg->segment;
	pgstrom_perfmon	   *pfm = &gpreagg->task.gts->pfm;
	size_t				offset;
	size_t				length;
	CUevent				ev_kern_main;
	CUresult			rc;
	cl_int				i;
	void			   *kern_args[10];

	/*
	 * Emergency bail out if previous gpupreagg task that references 
	 * same segment already failed thus CPU failback is needed.
	 * It is entirely nonsense to run remaining tasks in GPU kernel.
	 *
	 * NOTE: We don't need to synchronize completion of other tasks
	 * on the CPU fallback scenario, because we have no chance to add
	 * new tasks any more and its kds_final shall be never referenced.
	 */
	pg_memory_barrier();	/* CUDA callback may set needs_fallback */
	if (segment->needs_fallback)
	{
		GpuTaskState   *gts = gpreagg->task.gts;

		SpinLockAcquire(&gts->lock);
		Assert(!gpreagg->task.chain.prev && !gpreagg->task.chain.next);
		dlist_push_tail(&gts->completed_tasks, &gpreagg->task.chain);
		SpinLockRelease(&gts->lock);
		return true;
	}

	/*
	 * Lookup kernel functions
	 */
	rc = cuModuleGetFunction(&gpreagg->kern_main,
							 gpreagg->task.cuda_module,
							 "gpupreagg_main");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	rc = cuModuleGetFunction(&gpreagg->kern_fixvar,
							 gpreagg->task.cuda_module,
							 "gpupreagg_fixup_varlena");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	/*
	 * Allocation of own device memory
	 */
	if (gpreagg->kern.reduction_mode != GPUPREAGG_ONLY_TERMINATION)
	{
		length = (GPUMEMALIGN(KERN_GPUPREAGG_LENGTH(&gpreagg->kern)) +
				  GPUMEMALIGN(KERN_DATA_STORE_LENGTH(pds_in->kds)) +
				  GPUMEMALIGN(KERN_DATA_STORE_LENGTH(kds_head)) +
				  GPUMEMALIGN(offsetof(kern_global_hashslot,
									   hash_slot[gpreagg->kern.hash_size])));

		gpreagg->m_gpreagg = gpuMemAlloc(&gpreagg->task, length);
		if (!gpreagg->m_gpreagg)
			goto out_of_resource;
		gpreagg->m_kds_row = gpreagg->m_gpreagg +
			GPUMEMALIGN(KERN_GPUPREAGG_LENGTH(&gpreagg->kern));
		gpreagg->m_kds_slot = gpreagg->m_kds_row +
			GPUMEMALIGN(KERN_DATA_STORE_LENGTH(pds_in->kds));
		gpreagg->m_ghash = gpreagg->m_kds_slot +
			GPUMEMALIGN(KERN_DATA_STORE_LENGTH(kds_head));
	}
	else
	{
		length = GPUMEMALIGN(offsetof(kern_gpupreagg, kparams) +
							 KERN_GPUPREAGG_PARAMBUF_LENGTH(&gpreagg->kern));
		gpreagg->m_gpreagg = gpuMemAlloc(&gpreagg->task, length);
		if (!gpreagg->m_gpreagg)
			goto out_of_resource;
	}

	/*
	 * Allocation and setup final result buffer of this segment, or
	 * synchronize initialization by other task
	 */
	if (!gpupreagg_setup_segment(gpreagg, pfm->enabled))
		goto out_of_resource;

	/*
	 * Creation of event objects, if any
	 */
	CUDA_EVENT_CREATE(gpreagg, ev_dma_send_start);
	CUDA_EVENT_CREATE(gpreagg, ev_dma_send_stop);
	CUDA_EVENT_CREATE(gpreagg, ev_kern_fixvar);
	CUDA_EVENT_CREATE(gpreagg, ev_dma_recv_start);
	CUDA_EVENT_CREATE(gpreagg, ev_dma_recv_stop);

	/*
	 * OK, enqueue a series of commands
	 */
	CUDA_EVENT_RECORD(gpreagg, ev_dma_send_start);

	offset = KERN_GPUPREAGG_DMASEND_OFFSET(&gpreagg->kern);
	length = KERN_GPUPREAGG_DMASEND_LENGTH(&gpreagg->kern);
	rc = cuMemcpyHtoDAsync(gpreagg->m_gpreagg,
						   (char *)&gpreagg->kern + offset,
						   length,
						   gpreagg->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	pfm->bytes_dma_send += length;
	pfm->num_dma_send++;

	if (gpreagg->kern.reduction_mode != GPUPREAGG_ONLY_TERMINATION)
	{
		/* source data to be reduced */
		length = KERN_DATA_STORE_LENGTH(pds_in->kds);
		rc = cuMemcpyHtoDAsync(gpreagg->m_kds_row,
							   pds_in->kds,
							   length,
							   gpreagg->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		pfm->bytes_dma_send += length;
		pfm->num_dma_send++;

		/* header of the internal kds-slot buffer */
		length = KERN_DATA_STORE_HEAD_LENGTH(kds_head);
		rc = cuMemcpyHtoDAsync(gpreagg->m_kds_slot,
							   kds_head,
							   length,
							   gpreagg->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		pfm->bytes_dma_send += length;
		pfm->num_dma_send++;
	}
	CUDA_EVENT_RECORD(gpreagg, ev_dma_send_stop);

	/* Launch:
	 * KERNEL_FUNCTION(void)
	 * gpupreagg_main(kern_gpupreagg *kgpreagg,
	 *                kern_data_store *kds_row,
	 *                kern_data_store *kds_slot,
	 *                kern_global_hashslot *g_hash,
	 *                kern_data_store *kds_final,
	 *                kern_global_hashslot *f_hash)
	 */
	if (gpreagg->kern.reduction_mode != GPUPREAGG_ONLY_TERMINATION)
	{
		kern_args[0] = &gpreagg->m_gpreagg;
		kern_args[1] = &gpreagg->m_kds_row;
		kern_args[2] = &gpreagg->m_kds_slot;
		kern_args[3] = &gpreagg->m_ghash;
		kern_args[4] = &segment->m_kds_final;
		kern_args[5] = &segment->m_hashslot_final;

		rc = cuLaunchKernel(gpreagg->kern_main,
							1, 1, 1,
							1, 1, 1,
							sizeof(kern_errorbuf),
							gpreagg->task.cuda_stream,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
		pfm->gpreagg.num_kern_main++;
	}
	/*
	 * Record normal kernel execution end event
	 */
	ev_kern_main = segment->ev_kern_main[gpreagg->segment_id];
	rc = cuEventRecord(ev_kern_main, gpreagg->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuEventRecord: %s", errorText(rc));

	/*
	 * Final cleanup by the terminator task
	 */
	if (gpreagg->is_terminator)
	{
		pgstrom_data_store *pds_final = segment->pds_final;
		cl_uint		final_nrooms = pds_final->kds->nrooms;

		/*
		 * Synchronization of any other concurrent tasks
		 */
		for (i=0; i < segment->idx_chunks; i++)
		{
			rc = cuStreamWaitEvent(gpreagg->task.cuda_stream,
								   segment->ev_kern_main[i], 0);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuStreamWaitEvent: %s", errorText(rc));
		}

		/*
		 * Fixup varlena values, if needed
		 */
		if (pds_final->kds->has_notbyval)
		{
			size_t		grid_size;
			size_t		block_size;

			CUDA_EVENT_RECORD(gpreagg, ev_kern_fixvar);
			/* Launch:
			 * KERNEL_FUNCTION(void)
			 * gpupreagg_fixup_varlena(kern_gpupreagg *kgpreagg,
			 *                         kern_data_store *kds_final)
			 */
			optimal_workgroup_size(&grid_size,
								   &block_size,
								   gpreagg->kern_fixvar,
								   gpreagg->task.cuda_device,
								   final_nrooms,
								   0, sizeof(kern_errorbuf));
			kern_args[0] = &gpreagg->m_gpreagg;
			kern_args[1] = &segment->m_kds_final;

			rc = cuLaunchKernel(gpreagg->kern_fixvar,
								grid_size, 1, 1,
								block_size, 1, 1,
								sizeof(kern_errorbuf) * block_size,
								gpreagg->task.cuda_stream,
								kern_args,
								NULL);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
			pfm->gpreagg.num_kern_fixvar++;
		}
	}

	/*
	 * DMA Recv of individual kern_gpupreagg
	 */
	CUDA_EVENT_RECORD(gpreagg, ev_dma_recv_start);

	offset = KERN_GPUPREAGG_DMARECV_OFFSET(&gpreagg->kern);
	length = KERN_GPUPREAGG_DMARECV_LENGTH(&gpreagg->kern);
	rc = cuMemcpyDtoHAsync(&gpreagg->kern,
						   gpreagg->m_gpreagg + offset,
                           length,
                           gpreagg->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyDtoHAsync: %s", errorText(rc));
    pfm->bytes_dma_recv += length;
    pfm->num_dma_recv++;

	/*
	 * DMA Recv of final result buffer
	 */
	if (gpreagg->is_terminator)
	{
		pgstrom_data_store *pds_final = segment->pds_final;

		/* recv of kds_final */
		length = pds_final->kds->length;
		rc = cuMemcpyDtoHAsync(pds_final->kds,
							   segment->m_kds_final,
							   length,
							   gpreagg->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		pfm->bytes_dma_recv += length;
        pfm->num_dma_recv++;
	}
	CUDA_EVENT_RECORD(gpreagg, ev_dma_recv_stop);

	/*
	 * Register callback
	 */
	rc = cuStreamAddCallback(gpreagg->task.cuda_stream,
							 gpupreagg_task_respond,
							 gpreagg, 0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuStreamAddCallback: %s", errorText(rc));

	return true;

out_of_resource:
	gpupreagg_cleanup_cuda_resources(gpreagg);
	return false;
}

static bool
gpupreagg_task_process(GpuTask *gtask)
{
	pgstrom_gpupreagg *gpreagg = (pgstrom_gpupreagg *) gtask;
	bool		status;
	CUresult	rc;

	/* Switch CUDA Context */
	rc = cuCtxPushCurrent(gpreagg->task.cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));

	PG_TRY();
	{
		status = __gpupreagg_task_process(gpreagg);
	}
	PG_CATCH();
	{
		gpupreagg_cleanup_cuda_resources(gpreagg);
		rc = cuCtxPopCurrent(NULL);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));
		PG_RE_THROW();
	}
	PG_END_TRY();

	rc = cuCtxPopCurrent(NULL);
	if (rc != CUDA_SUCCESS)
		elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));

	return status;
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
	/* pg_strom.debug_force_gpupreagg */
	DefineCustomBoolVariable("pg_strom.debug_force_gpupreagg",
							 "Force GpuPreAgg regardless of the cost (debug)",
							 NULL,
							 &debug_force_gpupreagg,
							 false,
							 PGC_USERSET,
                             GUC_NOT_IN_SAMPLE,
                             NULL, NULL, NULL);

	/* initialization of plan method table */
	memset(&gpupreagg_scan_methods, 0, sizeof(CustomScanMethods));
	gpupreagg_scan_methods.CustomName          = "GpuPreAgg";
	gpupreagg_scan_methods.CreateCustomScanState
		= gpupreagg_create_scan_state;

	/* initialization of exec method table */
	memset(&gpupreagg_exec_methods, 0, sizeof(CustomExecMethods));
	gpupreagg_exec_methods.CustomName          = "GpuPreAgg";
   	gpupreagg_exec_methods.BeginCustomScan     = gpupreagg_begin;
	gpupreagg_exec_methods.ExecCustomScan      = gpupreagg_exec;
	gpupreagg_exec_methods.EndCustomScan       = gpupreagg_end;
	gpupreagg_exec_methods.ReScanCustomScan    = gpupreagg_rescan;
	gpupreagg_exec_methods.ExplainCustomScan   = gpupreagg_explain;
}
