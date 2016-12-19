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
#include "catalog/pg_aggregate.h"
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
#include "optimizer/pathnode.h"
#include "optimizer/planner.h"
#include "optimizer/tlist.h"
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

static create_upper_paths_hook_type create_upper_paths_next;
static CustomPathMethods		gpupreagg_path_methods;
static CustomScanMethods		gpupreagg_scan_methods;
static CustomExecMethods		gpupreagg_exec_methods;
static bool						enable_gpupreagg;

typedef struct
{
	cl_int			num_group_keys;	/* number of grouping keys */
	double			plan_ngroups;	/* planned number of groups */
	cl_int			plan_nchunks;	/* planned number of chunks */
	cl_int			plan_extra_sz;	/* planned size of extra-sz per tuple */

	double			outer_nrows;	/* number of estimated outer nrows */
	Index			outer_scanrelid;/* RTI, if outer path pulled up */
	List		   *outer_quals;	/* device executable quals of outer-scan */
	char		   *kern_source;
	int				extra_flags;
	List		   *used_params;	/* referenced Const/Param */
} GpuPreAggInfo;

static inline void
form_gpupreagg_info(CustomScan *cscan, GpuPreAggInfo *gpa_info)
{
	List	   *privs = NIL;
	List	   *exprs = NIL;

	privs = lappend(privs, makeInteger(gpa_info->num_group_keys));
	privs = lappend(privs,
					makeInteger(double_as_long(gpa_info->plan_ngroups)));
	privs = lappend(privs, makeInteger(gpa_info->plan_nchunks));
	privs = lappend(privs, makeInteger(gpa_info->plan_extra_sz));
	privs = lappend(privs, makeInteger(double_as_long(gpa_info->outer_nrows)));
	privs = lappend(privs, makeInteger(gpa_info->outer_scanrelid));
	exprs = lappend(exprs, gpa_info->outer_quals);
	privs = lappend(privs, makeString(gpa_info->kern_source));
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

	gpa_info->num_group_keys = intVal(list_nth(privs, pindex++));
	gpa_info->plan_ngroups = long_as_double(intVal(list_nth(privs, pindex++)));
	gpa_info->plan_nchunks = intVal(list_nth(privs, pindex++));
	gpa_info->plan_extra_sz = intVal(list_nth(privs, pindex++));
	gpa_info->outer_nrows = long_as_double(intVal(list_nth(privs, pindex++)));
	gpa_info->outer_scanrelid = intVal(list_nth(privs, pindex++));
	gpa_info->outer_quals = list_nth(exprs, eindex++);
	gpa_info->kern_source = strVal(list_nth(privs, pindex++));
	gpa_info->extra_flags = intVal(list_nth(privs, pindex++));
	gpa_info->used_params = list_nth(exprs, eindex++);

	return gpa_info;
}

/*
 * GpuPreAggSharedState - run-time state to be shared by both of backend
 * and GPU server process. To be allocated on the shared memory.
 */
typedef struct
{
	pg_atomic_uint32	refcnt;
	slock_t				lock;
	pgstrom_data_store *pds_final;
	CUdeviceptr			m_kds_final;/* final kernel data store (slot) */
	CUdeviceptr			m_fhash;	/* final global hash slot */
	CUevent				ev_kds_final;/* sync object for kds_final buffer */
	cl_uint				f_ncols;	/* @nroms of kds_final (constant) */
	cl_uint				f_key_dist_salt; /* current key_dist_salt setting */
	cl_uint				f_nrooms;	/* @nrooms of the current kds_final */
	cl_uint				f_nitems;	/* latest nitems of kds_final on device */
	cl_uint				f_extra_sz;	/* latest usage of kds_final on device */

	/* overall statistics */
	cl_uint				n_tasks_nogrp;	/* num of nogroup reduction tasks */
	cl_uint				n_tasks_local;	/* num of local reduction tasks */
	cl_uint				n_tasks_global;	/* num of global reduction tasks */
	cl_uint				n_tasks_final;	/* num of final reduction tasks */
	cl_uint				plan_nrows_per_chunk; /* planned nrows/chunk */
	size_t				plan_nrows_in;	/* num of outer nrows planned */
	size_t				exec_nrows_in;	/* num of outer nrows actually */
	size_t				plan_ngroups;	/* num of groups planned */
	size_t				exec_ngroups;	/* num of groups actually */
	size_t				plan_extra_sz;	/* size of varlena planned */
	size_t				exec_extra_sz;	/* size of varlena actually */
} GpuPreAggSharedState;

typedef struct
{
	GpuTaskState_v2	gts;
	GpuPreAggSharedState *gpa_sstate;

	cl_int			num_group_keys;
	TupleTableSlot *pseudo_slot;

	List		   *outer_quals;	/* List of ExprState */
	ProjectionInfo *outer_proj;		/* outer tlist -> custom_scan_tlist */
	pgstrom_data_store *outer_pds;
} GpuPreAggState;

/*
 * GpuPreAggTask
 *
 * Host side representation of kern_gpupreagg. It can perform as a message
 * object of PG-Strom, has key of OpenCL device program, a source row/column
 * store and a destination kern_data_store.
 */
typedef struct
{
	GpuTask_v2			task;
	GpuPreAggSharedState *gpa_sstate;
	bool				with_nvme_strom;/* true, if NVMe-Strom */
	bool				is_last_task;	/* true, if last task */
	bool				is_retry;		/* true, if task is retried */

	/* CUDA resources */
	CUdeviceptr			m_gpreagg;		/* kern_gpupreagg */
	CUdeviceptr			m_kds_in;		/* input row/block buffer */
	CUdeviceptr			m_kds_slot;		/* working (global) slot buffer */
	CUdeviceptr			m_ghash;		/* global hash slot */
	CUdeviceptr			m_kds_final;	/* final slot buffer (shared) */
	CUdeviceptr			m_fhash;		/* final hash slot (shared) */
	CUevent				ev_kds_final;
	CUevent				ev_dma_send_start;
	CUevent				ev_dma_send_stop;
	CUevent				ev_kern_fixvar;
	CUevent				ev_dma_recv_start;
	CUevent				ev_dma_recv_stop;

	/* performance counters */
	cl_uint				num_dma_send;
	cl_uint				num_dma_recv;
	Size				bytes_dma_send;
	Size				bytes_dma_recv;
	cl_float			tv_dma_send;
	cl_float			tv_dma_recv;
	cl_uint				num_kern_main;
	cl_uint				num_kern_prep;
	cl_uint				num_kern_nogrp;
	cl_uint				num_kern_lagg;
	cl_uint				num_kern_gagg;
	cl_uint				num_kern_fagg;
	cl_uint				num_kern_fixvar;
	cl_float			tv_kern_main;
	cl_float			tv_kern_prep;
	cl_float			tv_kern_nogrp;
	cl_float			tv_kern_lagg;
	cl_float			tv_kern_gagg;
	cl_float			tv_kern_fagg;
	cl_float			tv_kern_fixvar;

	/* DMA buffers */
	pgstrom_data_store *pds_in;		/* input row/block buffer */
	kern_data_store	   *kds_head;	/* head of working buffer */
	pgstrom_data_store *pds_final;	/* final data store, if any. It shall be
									 * attached on the server side. */
	kern_gpupreagg		kern;
} GpuPreAggTask;

/* declaration of static functions */
static void		gpupreagg_build_path_target(PlannerInfo *root,
											PathTarget *target_orig,
											PathTarget *target_upper,
											PathTarget *target_host,
											PathTarget *target_dev);
static char	   *gpupreagg_codegen(codegen_context *context,
								  PlannerInfo *root,
								  CustomScan *cscan,
								  List *tlist_dev,
								  List *tlist_dev_action,
								  List *outer_tlist,
								  List *outer_quals);
static GpuPreAggSharedState *
create_gpupreagg_shared_state(GpuPreAggState *gpas, GpuPreAggInfo *gpa_info,
							  TupleDesc pseudo_tupdesc);
static GpuPreAggSharedState *
get_gpupreagg_shared_state(GpuPreAggSharedState *gpa_sstate);
static void
put_gpupreagg_shared_state(GpuPreAggSharedState *gpa_sstate);

static GpuTask_v2 *gpupreagg_next_task(GpuTaskState_v2 *gts);
static void gpupreagg_ready_task(GpuTaskState_v2 *gts, GpuTask_v2 *gtask);
static void gpupreagg_switch_task(GpuTaskState_v2 *gts, GpuTask_v2 *gtask);
static TupleTableSlot *gpupreagg_next_tuple(GpuTaskState_v2 *gts);

static void gpupreagg_push_terminator_task(GpuPreAggTask *gpreagg_old);

/*
 * Arguments of alternative functions.
 */
#define ALTFUNC_GROUPING_KEY		 50	/* GROUPING KEY */
#define ALTFUNC_CONST_VALUE			 51	/* constant values */
//#define ALTFUNC_CONST_NULL			 52	/* NULL constant value */
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

#ifndef INT8ARRAYOID
#define INT8ARRAYOID		1016	/* see pg_type.h */
#endif
#ifndef FLOAT8ARRAYOID
#define FLOAT8ARRAYOID		1022	/* see pg_type.h */
#endif
#ifndef NUMERICARRAYOID
#define NUMERICARRAYOID		1231	/* see pg_type.h */
#endif

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
	const char *uppfn_name;
	Oid			uppfn_argtype;
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
	  "s:favg",     INT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "avg",    1, {INT4OID},
	  "s:favg",     INT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "avg",    1, {INT8OID},
	  "s:favg",     INT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "avg",    1, {FLOAT4OID},
	  "s:favg",     FLOAT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "avg",    1, {FLOAT8OID},
	  "s:favg",     FLOAT8ARRAYOID,
	  "s:pavg", 2, {INT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "avg",	1, {NUMERICOID},
	  "s:favg",     NUMERICARRAYOID,
	  "s:pavg", 2, {INT8OID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, DEVKERNEL_NEEDS_NUMERIC, 100
	},
#endif
	/* COUNT(*) = SUM(NROWS(*|X)) */
	{ "count",  0, {},
	  "c:sum",      INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_NROWS}, 0, INT_MAX
	},
	{ "count",  1, {ANYOID},
	  "c:sum",      INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_NROWS}, 0, INT_MAX
	},
	/* MAX(X) = MAX(PMAX(X)) */
	{ "max",    1, {INT2OID},
	  "c:max",      INT2OID,
	  "varref", 1, {INT2OID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max",    1, {INT4OID},
	  "c:max",      INT4OID,
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max",    1, {INT8OID},
	  "c:max",      INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max",    1, {FLOAT4OID},
	  "c:max",      FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max",    1, {FLOAT8OID},
	  "c:max",      FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "max",    1, {NUMERICOID},
	  "c:max",      NUMERICOID,
	  "varref", 1, {NUMERICOID},
	  {ALTFUNC_EXPR_PMAX}, DEVKERNEL_NEEDS_NUMERIC, INT_MAX
	},
#endif
	{ "max",    1, {CASHOID},
	  "c:max",      CASHOID,
	  "varref", 1, {CASHOID},
	  {ALTFUNC_EXPR_PMAX}, DEVKERNEL_NEEDS_MONEY, INT_MAX
	},
	{ "max",    1, {DATEOID},
	  "c:max",      DATEOID,
	  "varref", 1, {DATEOID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max",    1, {TIMEOID},
	  "c:max",      TIMEOID,
	  "varref", 1, {TIMEOID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max",    1, {TIMESTAMPOID},
	  "c:max",      TIMESTAMPOID,
	  "varref", 1, {TIMESTAMPOID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max",    1, {TIMESTAMPTZOID},
	  "c:max",      TIMESTAMPTZOID,
	  "varref", 1, {TIMESTAMPTZOID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},

	/* MIX(X) = MIN(PMIN(X)) */
	{ "min",    1, {INT2OID},
	  "c:min",      INT2OID,
	  "varref", 1, {INT2OID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min",    1, {INT4OID},
	  "c:min",      INT4OID,
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min",    1, {INT8OID},
	  "c:min",      INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min",    1, {FLOAT4OID},
	  "c:min",      FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min",    1, {FLOAT8OID},
	  "c:min",      FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "min",    1, {NUMERICOID},
	  "c:min",      NUMERICOID,
	  "varref", 1, {NUMERICOID},
	  {ALTFUNC_EXPR_PMIN}, DEVKERNEL_NEEDS_NUMERIC, INT_MAX
	},
#endif
	{ "min",    1, {CASHOID},
	  "c:min",      CASHOID,
	  "varref", 1, {CASHOID},
	  {ALTFUNC_EXPR_PMAX}, DEVKERNEL_NEEDS_MONEY, INT_MAX
	},
	{ "min",    1, {DATEOID},
	  "c:min",      DATEOID,
	  "varref", 1, {DATEOID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min",    1, {TIMEOID},
	  "c:min",      TIMEOID,
	  "varref", 1, {TIMEOID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min",    1, {TIMESTAMPOID},
	  "c:min",      TIMESTAMPOID,
	  "varref", 1, {TIMESTAMPOID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min",    1, {TIMESTAMPTZOID},
	  "c:min",      TIMESTAMPTZOID,
	  "varref", 1, {TIMESTAMPTZOID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},

	/* SUM(X) = SUM(PSUM(X)) */
	{ "sum",    1, {INT2OID},
	  "s:sum",      INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "sum",    1, {INT4OID},
	  "s:sum",      INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "sum",    1, {INT8OID},
	  "c:sum",      INT8OID,
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "sum",    1, {FLOAT4OID},
	  "c:sum",      FLOAT4OID,
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "sum",    1, {FLOAT8OID},
	  "c:sum",      FLOAT8OID,
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "sum",    1, {NUMERICOID},
	  "c:sum",      NUMERICOID,
	  "varref", 1, {NUMERICOID},
	  {ALTFUNC_EXPR_PSUM}, DEVKERNEL_NEEDS_NUMERIC, 100
	},
#endif
	{ "sum",    1, {CASHOID},
	  "c:sum",      CASHOID,
	  "varref", 1, {CASHOID},
	  {ALTFUNC_EXPR_PSUM}, DEVKERNEL_NEEDS_MONEY, INT_MAX
	},
	/* STDDEV(X) = EX_STDDEV(NROWS(),PSUM(X),PSUM(X*X)) */
	{ "stddev",      1, {FLOAT4OID},
	  "s:stddev",        FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "stddev",      1, {FLOAT8OID},
	  "s:stddev",        FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "stddev_pop",  1, {FLOAT4OID},
	  "s:stddev_pop",    FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "stddev_pop",  1, {FLOAT8OID},
	  "s:stddev_pop",    FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "stddev_samp", 1, {FLOAT4OID},
	  "s:stddev_samp",   FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "stddev_samp", 1, {FLOAT8OID},
	  "s:stddev_samp",   FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	/* VARIANCE(X) = PGSTROM.VARIANCE(NROWS(), PSUM(X),PSUM(X^2)) */
	{ "variance",    1, {FLOAT4OID},
	  "s:variance",      FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "variance",    1, {FLOAT8OID},
	  "s:variance",      FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "var_pop",     1, {FLOAT4OID},
	  "s:var_pop",       FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "var_pop",     1, {FLOAT8OID},
	  "s:var_pop",       FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "var_samp",    1, {FLOAT4OID},
	  "s:var_samp",      FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "var_samp",    1, {FLOAT8OID},
	  "s:var_samp",      FLOAT8ARRAYOID,
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
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
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
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
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
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
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
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
       ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
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
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
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
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
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
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
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
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
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
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
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
	   ALTFUNC_EXPR_PCOV_XY}, 0, SHRT_MAX
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
 * gpupreagg_device_executable
 *
 * checks whether the aggregate function/grouping clause are executable
 * on the device side.
 */
static bool
gpupreagg_device_executable(PlannerInfo *root, PathTarget *target)
{
	devtype_info   *dtype;
	devfunc_info   *dfunc;
	int				resno = 1;
	ListCell	   *lc;
	ListCell	   *cell;

	foreach (lc, target->exprs)
	{
		Expr   *expr = lfirst(lc);

		if (IsA(expr, Aggref))
		{
			Aggref	   *aggref = (Aggref *) expr;
			const aggfunc_catalog_t *aggfn_cat;

			if (target->sortgrouprefs[resno - 1] > 0)
			{
				elog(WARNING, "Bug? Aggregation is referenced by GROUP BY");
				return false;
			}

			/*
			 * Aggregate function must be supported by GpuPreAgg
			 */
			aggfn_cat = aggfunc_lookup_by_oid(aggref->aggfnoid);
			if (!aggfn_cat)
			{
				elog(DEBUG2, "Aggref is not supported: %s",
					 nodeToString(aggref));
				return false;
			}

			/*
			 * If arguments of aggregate function are expression, it must be
			 * constructable on the device side.
			 */
			foreach (cell, aggref->args)
			{
				TargetEntry	   *tle = lfirst(cell);

				Assert(IsA(tle, TargetEntry));
				if (!IsA(tle->expr, Var) &&
					!IsA(tle->expr, PlaceHolderVar) &&
					!IsA(tle->expr, Const) &&
					!IsA(tle->expr, Param) &&
					!pgstrom_device_expression(tle->expr))
				{
					elog(DEBUG2, "Expression is not device executable: %s",
						 nodeToString(tle->expr));
					return false;
				}
			}
		}
		else
		{
			/*
			 * Data type of grouping-key must support equality function
			 * for hash-based algorithm.
			 */
			dtype = pgstrom_devtype_lookup(exprType((Node *)expr));
			if (!dtype)
			{
				elog(DEBUG2, "device type %s is not supported",
					 format_type_be(exprType((Node *)expr)));
				return false;
			}
			dfunc = pgstrom_devfunc_lookup(dtype->type_eqfunc, InvalidOid);
			if (!dfunc)
			{
				elog(DEBUG2, "device function %s is not supported",
					 format_procedure(dtype->type_eqfunc));
				return false;
			}

			/*
			 * If input is not a simple Var reference, expression must be
			 * constructable on the device side.
			 */
			if (!IsA(expr, Var) &&
				!IsA(expr, PlaceHolderVar) &&
				!IsA(expr, Const) &&
				!IsA(expr, Param) &&
				!pgstrom_device_expression(expr))
			{
				elog(DEBUG2, "Expression is not device executable: %s",
					 nodeToString(expr));
				return false;
			}
		}
		resno++;
	}
	return true;
}

/*
 * cost_gpupreagg
 *
 * cost estimation for GpuPreAgg node
 */
static bool
cost_gpupreagg(CustomPath *cpath,
			   GpuPreAggInfo *gpa_info,
			   PlannerInfo *root,
			   PathTarget *target,
			   Path *input_path,
			   double num_groups,
			   AggClauseCosts *agg_costs)
{
	double		input_ntuples = input_path->rows;
	Cost		startup_cost = input_path->total_cost;
	Cost		run_cost = 0.0;
	QualCost	qual_cost;
	int			num_group_keys = 0;
	Size		extra_sz = 0;
	Size		kds_length;
	double		gpu_cpu_ratio;
	cl_uint		ncols;
	cl_uint		nrooms;
	cl_int		index;
	cl_int		key_dist_salt;
	ListCell   *lc;

	/* Fixed cost to setup/launch GPU kernel */
	startup_cost += pgstrom_gpu_setup_cost;

	/*
	 * Estimation of the result buffer. It must fit to the target GPU device
	 * memory size.
	 */
	index = 0;
	foreach (lc, target->exprs)
	{
		Expr	   *expr = lfirst(lc);
		Oid			type_oid = exprType((Node *)expr);
		int32		type_mod = exprTypmod((Node *)expr);
		int16		typlen;
		bool		typbyval;

		/* extra buffer */
		if (type_oid == NUMERICOID)
			extra_sz += 32;
		else
		{
			get_typlenbyval(type_oid, &typlen, &typbyval);
			if (!typbyval)
				extra_sz += get_typavgwidth(type_oid, type_mod);
		}
		/* cound number of grouping keys */
		if (target->sortgrouprefs[index] > 0)
			num_group_keys++;
	}
	if (num_group_keys == 0)
		num_groups = 1.0;	/* AGG_PLAIN */
	/*
	 * NOTE: In case when the number of groups are too small, it leads too
	 * many atomic contention on the device. So, we add a small salt to
	 * distribute grouping keys than the actual number of keys.
	 * It shall be adjusted on run-time, so configuration below is just
	 * a baseline parameter.
	 */
	if (num_groups < (devBaselineMaxThreadsPerBlock / 5))
	{
		key_dist_salt = (devBaselineMaxThreadsPerBlock / (5 * num_groups));
		key_dist_salt = Max(key_dist_salt, 1);
	}
	else
		key_dist_salt = 1;

	ncols = list_length(target->exprs);
	nrooms = (cl_uint)(2.5 * num_groups * (double)key_dist_salt);
	kds_length = (STROMALIGN(offsetof(kern_data_store, colmeta[ncols])) +
				  STROMALIGN((sizeof(Datum) +
							  sizeof(bool)) * ncols) * nrooms +
				  STROMALIGN(extra_sz) * nrooms);
	if (kds_length > gpuMemMaxAllocSize())
		return false;	/* expected buffer size is too large */

	/* Cost estimation to setup initial values */
	gpu_cpu_ratio = pgstrom_gpu_operator_cost / cpu_operator_cost;
	startup_cost += (qual_cost.startup +
					 qual_cost.per_tuple * input_ntuples) * gpu_cpu_ratio;
	/* Cost estimation for grouping */
	startup_cost += pgstrom_gpu_operator_cost * num_group_keys * input_ntuples;
	/* Cost estimation for aggregate function */
	startup_cost += (agg_costs->transCost.startup +
					 agg_costs->transCost.per_tuple *
					 gpu_cpu_ratio * input_ntuples);
	/* Cost estimation to fetch results */
	run_cost += cpu_tuple_cost * num_groups;

	cpath->path.rows			= num_groups * (double)key_dist_salt;
	cpath->path.startup_cost	= startup_cost;
	cpath->path.total_cost		= startup_cost + run_cost;

	gpa_info->num_group_keys    = num_group_keys;
	gpa_info->plan_ngroups		= num_groups;
	gpa_info->plan_nchunks		= estimate_num_chunks(input_path);
	gpa_info->plan_extra_sz		= extra_sz;
	gpa_info->outer_nrows		= input_ntuples;

	return true;
}

/*
 * make_partial_grouping_target
 *
 * see optimizer/plan/planner.c
 */
static PathTarget *
make_partial_grouping_target(PlannerInfo *root, PathTarget *grouping_target)
{
	Query	   *parse = root->parse;
	PathTarget *partial_target;
	List	   *non_group_cols;
	List	   *non_group_exprs;
	int			i;
	ListCell   *lc;

	partial_target = create_empty_pathtarget();
	non_group_cols = NIL;

	i = 0;
	foreach(lc, grouping_target->exprs)
	{
		Expr	   *expr = (Expr *) lfirst(lc);
		Index		sgref = get_pathtarget_sortgroupref(grouping_target, i);

		if (sgref && parse->groupClause &&
			get_sortgroupref_clause_noerr(sgref, parse->groupClause) != NULL)
		{
			/*
			 * It's a grouping column, so add it to the partial_target as-is.
			 * (This allows the upper agg step to repeat the grouping calcs.)
			 */
			add_column_to_pathtarget(partial_target, expr, sgref);
		}
		else
		{
			/*
			 * Non-grouping column, so just remember the expression for later
			 * call to pull_var_clause.
			 */
			non_group_cols = lappend(non_group_cols, expr);
		}
		i++;
	}

	/*
	 * If there's a HAVING clause, we'll need the Vars/Aggrefs it uses, too.
	 */
	if (parse->havingQual)
		non_group_cols = lappend(non_group_cols, parse->havingQual);

	/*
	 * Pull out all the Vars, PlaceHolderVars, and Aggrefs mentioned in
	 * non-group cols (plus HAVING), and add them to the partial_target if not
	 * already present.  (An expression used directly as a GROUP BY item will
	 * be present already.)  Note this includes Vars used in resjunk items, so
	 * we are covering the needs of ORDER BY and window specifications.
	 */
	non_group_exprs = pull_var_clause((Node *) non_group_cols,
									  PVC_INCLUDE_AGGREGATES |
									  PVC_RECURSE_WINDOWFUNCS |
									  PVC_INCLUDE_PLACEHOLDERS);

	add_new_columns_to_pathtarget(partial_target, non_group_exprs);

	/*
	 * Adjust Aggrefs to put them in partial mode.  At this point all Aggrefs
	 * are at the top level of the target list, so we can just scan the list
	 * rather than recursing through the expression trees.
	 */
	foreach(lc, partial_target->exprs)
	{
		Aggref	   *aggref = (Aggref *) lfirst(lc);
		Aggref	   *newaggref;

		if (IsA(aggref, Aggref))
		{
			/*
			 * We shouldn't need to copy the substructure of the Aggref node,
			 * but flat-copy the node itself to avoid damaging other trees.
			 */
			newaggref = makeNode(Aggref);
			memcpy(newaggref, aggref, sizeof(Aggref));

			/* For now, assume serialization is required */
			mark_partial_aggref(newaggref, AGGSPLIT_INITIAL_SERIAL);

			lfirst(lc) = newaggref;
		}
	}

	/* clean up cruft */
	list_free(non_group_exprs);
	list_free(non_group_cols);

	/* XXX this causes some redundant cost calculation ... */
	return set_pathtarget_cost_width(root, partial_target);
}

/*
 * estimate_hashagg_tablesize
 *
 * See optimizer/plan/planner.c
 */
static Size
estimate_hashagg_tablesize(Path *path, const AggClauseCosts *agg_costs,
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

/*
 * gpupreagg_construct_path
 *
 * constructor of the GpuPreAgg path node
 */
static CustomPath *
gpupreagg_construct_path(PlannerInfo *root,
						 PathTarget *target_host,
						 PathTarget *target_dev,
						 RelOptInfo *group_rel,
						 Path *input_path,
						 double num_groups)
{
	CustomPath	   *cpath = makeNode(CustomPath);
	GpuPreAggInfo  *gpa_info = palloc0(sizeof(GpuPreAggInfo));
	List		   *custom_paths = NIL;
	PathTarget	   *partial_target;
	AggClauseCosts	agg_partial_costs;

	/* obviously, not suitable for GpuPreAgg */
	if (num_groups < 1.0 || num_groups > (double)INT_MAX)
		return false;

#if 0
	/* PathTarget of the partial stage */
	partial_target = make_partial_grouping_target(root, target);
	get_agg_clause_costs(root, (Node *) partial_target->exprs,
						 AGGSPLIT_INITIAL_SERIAL,
						 &agg_partial_costs);
#endif

	/* cost estimation */
	if (!cost_gpupreagg(cpath, gpa_info,
						root, target, input_path,
						num_groups, &agg_partial_costs))
	{
		pfree(cpath);
		return NULL;
	}

	/*
	 * Try to pull up input_path if it is enough simple scan.
	 */
	if (!pgstrom_pullup_outer_scan(input_path,
								   &gpa_info->outer_scanrelid,
								   &gpa_info->outer_quals))
		custom_paths = list_make1(input_path);

	/* Setup CustomPath */
	cpath->path.pathtype = T_CustomScan;
	cpath->path.parent = group_rel;
	cpath->path.pathtarget = partial_target;
	cpath->path.param_info = NULL;
	cpath->path.parallel_aware = false;
	cpath->path.parallel_safe = (group_rel->consider_parallel &&
								 input_path->parallel_safe);
	cpath->path.parallel_workers = input_path->parallel_workers;
	cpath->path.pathkeys = NIL;
	cpath->custom_paths = custom_paths;
	cpath->custom_private = list_make1(gpa_info);
	cpath->methods = &gpupreagg_path_methods;

	return cpath;
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
							 RelOptInfo *group_rel)
{
	Query		   *parse = root->parse;
	PathTarget	   *target = root->upper_targets[UPPERREL_GROUP_AGG];
	PathTarget	   *target_upper;
	PathTarget	   *target_host;
	PathTarget	   *target_dev;
	CustomPath	   *cpath;
	Path		   *input_path;
	Path		   *final_path;
	Path		   *sort_path;
//	Path		   *gather_path;
	double			num_groups;
	bool			can_sort;
	bool			can_hash;
	AggClauseCosts	agg_final_costs;

	if (create_upper_paths_next)
		(*create_upper_paths_next)(root, stage, input_rel, group_rel);

	if (stage != UPPERREL_GROUP_AGG)
		return;

	if (!pgstrom_enabled || !enable_gpupreagg)
		return;

	if (get_namespace_oid("pgstrom", true) == InvalidOid)
	{
		ereport(WARNING,
				(errcode(ERRCODE_UNDEFINED_SCHEMA),
				 errmsg("schema \"pgstrom\" was not found"),
				 errhint("Run: CREATE EXTENSION pg_strom")));
		return;
	}

	if (!gpupreagg_device_executable(root, target))
		return;

	/* number of estimated groups */
	if (!parse->groupClause)
		num_groups = 1.0;
	else
	{
		Path	   *pathnode = linitial(group_rel->pathlist);

		num_groups = pathnode->rows;
	}

	/* get cost of aggregations */
	memset(&agg_final_costs, 0, sizeof(AggClauseCosts));
	if (parse->hasAggs)
	{
		get_agg_clause_costs(root, (Node *)root->processed_tlist,
							 AGGSPLIT_SIMPLE, &agg_final_costs);
		get_agg_clause_costs(root, parse->havingQual,
							 AGGSPLIT_SIMPLE, &agg_final_costs);
	}

	/* GpuPreAgg does not support ordered aggregation */
	if (agg_final_costs.numOrderedAggs > 0)
		return;

	/*
	 * construction of PathTarget to be attached on
	 *
	 * @target_upper: PathTarget for final Agg node (includes Aggref)
	 * @target_host:  PathTarget for host-side of CustomScan
	 * @target_dev:   PathTarget of device results or CPU fallback
	 */
	target_upper = create_empty_pathtarget();
	target_host = create_empty_pathtarget();
	target_dev = create_empty_pathtarget();
	gpupreagg_build_path_target(root, target,
								target_upper,
								target_host,
								target_dev);

	/*
	 * construction of GpuPreAgg pathnode on top of the cheapest total
	 * cost pathnode (partial aggregation)
	 */
	input_path = input_rel->cheapest_total_path;
	cpath = gpupreagg_construct_path(root, target, group_rel,
									 input_path, num_groups);
	if (!cpath)
		return;

	/* strategy of the final aggregation */
	can_sort = grouping_is_sortable(parse->groupClause);
	can_hash = (parse->groupClause != NIL &&
				parse->groupingSets == NIL &&
				agg_final_costs.numOrderedAggs == 0 &&
				grouping_is_hashable(parse->groupClause));

	/* make a final grouping path (nogroup) */
	if (!parse->groupClause)
	{
		final_path = (Path *)create_agg_path(root,
											 group_rel,
											 &cpath->path,
											 target_upper,
											 AGG_PLAIN,
											 AGGSPLIT_SIMPLE,
											 parse->groupClause,
											 (List *) parse->havingQual,
											 &agg_final_costs,
											 num_groups);
		add_path(group_rel, final_path);
		// TODO: make a parallel grouping path (nogroup) */
	}
	else
	{
		/* make a final grouping path (sort) */
		if (can_sort)
		{
			sort_path = (Path *)
				create_sort_path(root,
								 group_rel,
								 &cpath->path,
								 root->group_pathkeys,
								 -1.0);
			if (parse->groupingSets)
			{
				List	   *rollup_lists = NIL;
				List	   *rollup_groupclauses = NIL;
				bool		found = false;
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
						rollup_groupclauses = pathnode->rollup_groupclauses;
						rollup_lists = pathnode->rollup_lists;
						found = true;
						break;
					}
				}
				if (!found)
					return;		/* give up */
				final_path = (Path *)
					create_groupingsets_path(root,
											 group_rel,
											 sort_path,
											 target_upper,
											 (List *)parse->havingQual,
											 rollup_lists,
											 rollup_groupclauses,
											 &agg_final_costs,
											 num_groups);
			}
			else if (parse->hasAggs)
				final_path = (Path *)
					create_agg_path(root,
									group_rel,
									sort_path,
									target_upper,
									AGG_SORTED,
									AGGSPLIT_SIMPLE,
									parse->groupClause,
									(List *)parse->havingQual,
									&agg_final_costs,
									num_groups);
			else if (parse->groupClause)
				final_path = (Path *)
					create_group_path(root,
									  group_rel,
									  sort_path,
									  target_upper,
									  parse->groupClause,
									  (List *)parse->havingQual,
									  num_groups);
			else
				elog(ERROR, "Bug? unexpected AGG/GROUP BY requirement");

			add_path(group_rel, final_path);

			// TODO: make a parallel grouping path (sort) */
		}

		/* make a final grouping path (hash) */
		if (can_hash)
		{
			Size	hashaggtablesize
				= estimate_hashagg_tablesize(&cpath->path,
											 &agg_final_costs,
											 num_groups);
			if (hashaggtablesize < work_mem * 1024L)
			{
				final_path = (Path *)
					create_agg_path(root,
									group_rel,
									&cpath->path,
									target_upper,
									AGG_HASHED,
									AGGSPLIT_SIMPLE,
									parse->groupClause,
									(List *) parse->havingQual,
									&agg_final_costs,
									num_groups);
				add_path(group_rel, final_path);
			}
			/* TODO: make a parallel grouping path (hash+gather) */
		}
	}	
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

	Assert(exprType((Node *) filter) == BOOLOID);
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
 * make_altfunc_simple_expr
 */
static Expr *
make_altfunc_simple_expr(const char *func_name, List *func_args)
{
	Oid			namespace_oid = get_namespace_oid("pgstrom", false);
	Oid			typebuf[8];
	oidvector  *func_argtypes;
	HeapTuple	tuple;
	Form_pg_proc proc_form;
	Expr	   *expr;
	ListCell   *lc;
	int			i = 0;

	/* set up oidvector */
	foreach (lc, func_args)
		typebuf[i++] = exprType((Node *) lfirst(lc));
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
								 func_args,
								 InvalidOid,
								 InvalidOid,
								 COERCE_EXPLICIT_CALL);
	ReleaseSysCache(tuple);

	return expr;
}

/*
 * make_altfunc_nrows_expr - constructor of the partial number of rows
 */
static Expr *
make_altfunc_nrows_expr(Aggref *aggref)
{
	List	   *nrows_args = NIL;
	ListCell   *lc;

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
		Assert(exprType(aggref->aggfilter) == BOOLOID);
		nrows_args = lappend(nrows_args, copyObject(aggref->aggfilter));
	}
	return make_altfunc_simple_expr("nrows", nrows_args);
}

/*
 * make_altfunc_minmax_expr
 */
static Expr *
make_altfunc_minmax_expr(Aggref *aggref, const char *func_name)
{
	Oid			namespace_oid = get_namespace_oid("pgstrom", false);
	Oid			anyelement_typeoid = ANYELEMENTOID;
	oidvector  *func_argtypes;
	HeapTuple	tuple;
	Form_pg_proc proc_form;
	Expr	   *expr;
	ListCell   *lc;
	int			i = 0;

	Assert(list_length(aggref->args) == 1);
    tle = linitial(aggref->args);
    Assert(IsA(tle, TargetEntry));
    expr = tle->expr;
    if (aggref->aggfilter)
	{
		Assert(exprType(aggref->aggfilter) == BOOLOID);
        expr = make_expr_conditional(expr, aggref->aggfilter, false);
	}
	/* find the partial min/max function */
	func_argtypes = buildoidvector(&anyelement_typeoid, 1);
	tuple = SearchSysCache3(PROCNAMEARGSNSP,
							PointerGetDatum(func_name),
							PointerGetDatum(func_argtypes),
							ObjectIdGetDatum(namespace_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for function: %s",
			 funcname_signature_string(func_name,
									   1, NIL, &anyelement_typeoid));
	proc_form = (Form_pg_proc) GETSTRUCT(tuple);
	expr = (Expr *) makeFuncExpr(HeapTupleGetOid(tuple),
								 proc_form->prorettype,
								 list_make1(expr),	/* argument */
								 InvalidOid,
                                 InvalidOid,
                                 COERCE_EXPLICIT_CALL);
	ReleaseSysCache(tuple);

	return expr;
}

/*
 * make_altfunc_psum - constructor of a SUM/SUM_X2 reference
 */
static Expr *
make_altfunc_psum_expr(Aggref *aggref, const char *func_name, Oid psum_typeoid)
{


	Assert(list_length(aggref->args) == 1);
	tle = linitial(aggref->args);
	Assert(IsA(tle, TargetEntry));

	/* cast to psum_typeoid, if mismatch */
	expr = make_expr_typecast(tle->expr, psum_typeoid);

	/* make conditional if aggref has no filter */
	if (aggref->aggfilter)
	{
		Assert(exprType(aggref->aggfilter) == BOOLOID);
		expr = make_expr_conditional(expr, aggref->aggfilter, true);
	}

	/* find the partial min/max function */
	func_argtypes = buildoidvector(&anyelement_typeoid, 1);
	tuple = SearchSysCache3(PROCNAMEARGSNSP,
							PointerGetDatum(func_name),
							PointerGetDatum(func_argtypes),
							ObjectIdGetDatum(namespace_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "cache lookup failed for function: %s",
			 funcname_signature_string(func_name,
									   1, NIL, &anyelement_typeoid));
	proc_form = (Form_pg_proc) GETSTRUCT(tuple);
	expr = (Expr *) makeFuncExpr(HeapTupleGetOid(tuple),
								 proc_form->prorettype,
								 list_make1(expr),	/* argument */
								 InvalidOid,
                                 InvalidOid,
                                 COERCE_EXPLICIT_CALL);
	ReleaseSysCache(tuple);

	return expr;
}


/*
 * make_altfunc_psum_x2 - constructor of a simple (variable)^2 reference
 */
static Expr *
make_altfunc_psum_x2(Aggref *aggref)
{
	TargetEntry	   *tle;
	FuncExpr	   *func_expr;
	Oid				type_oid;
	Oid				func_oid;

	Assert(list_length(aggref->args) == 1);
	tle = linitial(aggref->args);
	Assert(IsA(tle, TargetEntry));

	type_oid = exprType((Node *) tle->expr);
	if (type_oid == FLOAT4OID)
		func_oid = F_FLOAT4MUL;
	else if (type_oid == FLOAT8OID)
		func_oid = F_FLOAT8MUL;
	else if (type_oid == NUMERICOID)
		func_oid = F_NUMERIC_MUL;
	else
		elog(ERROR, "Bug? unexpected expression data type");

	func_expr = makeFuncExpr(func_oid,
							 type_oid,
							 list_make2(copyObject(tle->expr),
										copyObject(tle->expr)),
							 InvalidOid,
							 InvalidOid,
							 COERCE_EXPLICIT_CALL);
	if (!aggref->aggfilter)
		return (Expr *)func_expr;

	return make_expr_conditional((Expr *)func_expr, aggref->aggfilter, false);
}

/*
 * make_altfunc_pcov_xy - constructor of a co-variance arguments
 */
static Expr *
make_altfunc_pcov_xy(Aggref *aggref, int action)
{
	TargetEntry	   *tle_x;
	TargetEntry	   *tle_y;
	NullTest	   *nulltest_x;
	NullTest	   *nulltest_y;
	List		   *arg_checks = NIL;
	Expr		   *expr;

	Assert(list_length(aggref->args) == 2);
	tle_x = linitial(aggref->args);
	tle_y = lsecond(aggref->args);
	if (exprType((Node *)tle_x->expr) != FLOAT8OID ||
		exprType((Node *)tle_y->expr) != FLOAT8OID)
		elog(ERROR, "Bug? unexpected argument type for co-variance");

	if (aggref->aggfilter)
		arg_checks = lappend(arg_checks, aggref->aggfilter);
	/* nulltest for X-argument */
	nulltest_x = makeNode(NullTest);
	nulltest_x->arg = copyObject(tle_x->expr);
	nulltest_x->nulltesttype = IS_NOT_NULL;
	nulltest_x->argisrow = false;
	nulltest_x->location = aggref->location;
	arg_checks = lappend(arg_checks, nulltest_x);

	/* nulltest for Y-argument */
	nulltest_y = makeNode(NullTest);
	nulltest_y->arg = copyObject(tle_y->expr);
	nulltest_y->nulltesttype = IS_NOT_NULL;
	nulltest_y->argisrow = false;
	nulltest_y->location = aggref->location;
	arg_checks = lappend(arg_checks, nulltest_y);

	switch (action)
	{
		case ALTFUNC_EXPR_PCOV_X:	/* PCOV_X(X,Y) */
			expr = tle_x->expr;
			break;
		case ALTFUNC_EXPR_PCOV_Y:	/* PCOV_Y(X,Y) */
			expr = tle_y->expr;
			break;
		case ALTFUNC_EXPR_PCOV_X2:	/* PCOV_X2(X,Y) */
			expr = (Expr *)makeFuncExpr(F_FLOAT8MUL,
										FLOAT8OID,
										list_make2(tle_x->expr,
												   tle_x->expr),
										InvalidOid,
										InvalidOid,
										COERCE_EXPLICIT_CALL);
			break;
		case ALTFUNC_EXPR_PCOV_Y2:	/* PCOV_Y2(X,Y) */
			expr = (Expr *)makeFuncExpr(F_FLOAT8MUL,
										FLOAT8OID,
										list_make2(tle_y->expr,
												   tle_y->expr),
										InvalidOid,
										InvalidOid,
										COERCE_EXPLICIT_CALL);
			break;
		case ALTFUNC_EXPR_PCOV_XY:	/* PCOV_XY(X,Y) */
			expr = (Expr *)makeFuncExpr(F_FLOAT8MUL,
										FLOAT8OID,
										list_make2(tle_x->expr,
												   tle_y->expr),
										InvalidOid,
										InvalidOid,
										COERCE_EXPLICIT_CALL);
			break;
		default:
			elog(ERROR, "Bug? unexpected action type for co-variance ");
	}
	return make_expr_conditional(expr, make_andclause(arg_checks), false);
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
 * gpupreagg_build_path_target
 *
 *
 *
 */
static void
gpupreagg_build_path_target(PlannerInfo *root,
							PathTarget *target_orig,
							PathTarget *target_upper,
							PathTarget *target_host,
							PathTarget *target_dev)
{
	List	   *tlist_upper = NIL;
	List	   *tlist_host = NIL;
	List	   *tlist_dev = NIL;
	ListCell   *lc;
	cl_int		i;

	foreach (lc, target_orig->exprs)
	{
		TargetEntry	   *tle = lfirst(lc);
		TargetEntry	   *tle_host;
		TargetEntry	   *tle_upper;
		List		   *altfn_args = NIL;

		if (IsA(tle->expr, Aggref))
		{
			Aggref		   *aggref = (Aggref *)tle->expr;
			Aggref		   *aggref_new;
			Oid				namespace_oid;
			FuncExpr	   *func_expr;
			const char	   *func_name;
			oidvector	   *func_argtypes;
			HeapTuple		tuple;
			HeapTuple		agg_tuple;
			Form_pg_aggregate agg_form;
			Form_pg_proc	proc_form;
			Oid				aggfn_oid;
			const aggfunc_catalog_t *aggfn_cat;

			Assert(target_orig->sortgrouprefs == NULL ||
				   target_orig->sortgrouprefs[tle->resno - 1] == 0);
			aggfn_cat = aggfunc_lookup_by_oid(aggref->aggfnoid);
			if (!aggfn_cat)
				elog(ERROR, "lookup failed on aggregate function: %u",
					 aggref->aggfnoid);
			/*
			 * construction of the initial partial aggregation
			 */
			for (i=0; i < aggfn_cat->altfn_nargs; i++)
			{
				cl_int		action = aggfn_cat->altfn_argexprs[i];
				cl_int		argtype = aggfn_cat->altfn_argtypes[i];
				Expr	   *expr;

				switch (action)
				{
					case ALTFUNC_EXPR_NROWS:    /* NROWS(X) */
						expr = make_altfunc_nrows_expr(aggref);
						break;
					case ALTFUNC_EXPR_PMIN:     /* PMIN(X) */
						expr = make_altfunc_minmax_expr(aggref, "pmin");
						break;
					case ALTFUNC_EXPR_PMAX:     /* PMAX(X) */
						expr = make_altfunc_minmax_expr(aggref, "pmax");
						break;
					case ALTFUNC_EXPR_PSUM:     /* PSUM(X) */
						expr = make_altfunc_psum_expr(aggref, "psum", argtype);
						break;
					case ALTFUNC_EXPR_PSUM_X2:  /* PSUM_X2(X) = PSUM(X^2) */
						expr = make_altfunc_psum_expr(aggref, "psum_x2",
													  argtype);
						break;
					case ALTFUNC_EXPR_PCOV_X:   /* PCOV_X(X,Y) */
					case ALTFUNC_EXPR_PCOV_Y:   /* PCOV_Y(X,Y) */
					case ALTFUNC_EXPR_PCOV_X2:  /* PCOV_X2(X,Y) */
					case ALTFUNC_EXPR_PCOV_Y2:  /* PCOV_Y2(X,Y) */
					case ALTFUNC_EXPR_PCOV_XY:  /* PCOV_XY(X,Y) */
						expr = make_altfunc_pcov_xy(aggref, action);
						break;
					default:
						elog(ERROR, "unknown alternative function code: %d",
							 action);
						break;
				}
				/* force type cast if mismatch */
				expr = make_expr_typecast(expr, argtype);

				if (!tlist_member((Node *)expr, tlist_dev))
				{
					TargetEntry	   *tle_new
						= makeTargetEntry(copyObject(expr),
										  list_length(tlist_dev) + 1,
										  NULL,
										  false);
					tlist_dev = lappend(tlist_dev, tle_new);
				}
				altfn_args = lappend(altfn_args, expr);
			}

			/*
			 * Lookup an alternative function that generates partial state
			 * of the final aggregate function, or varref if internal state
			 * of aggregation is as-is.
			 */
			if (strcmp(aggfn_cat->altfn_name, "varref") == 0)
			{
				Assert(list_length(altfn_args) == 1);
				tle_host = makeTargetEntry(copyObject(linitial(altfn_args)),
										   tle->resno,
										   tle->resname,
										   tle->resjunk);
			}
			else
			{
				if (strncmp(aggfn_cat->altfn_name, "c:", 2) == 0)
					namespace_oid = PG_CATALOG_NAMESPACE;
				else if (strncmp(aggfn_cat->altfn_name, "s:", 2) == 0)
					namespace_oid = get_namespace_oid("pgstrom", false);
				else
					elog(ERROR, "Bug? incorrect alternative function catalog");

				func_name = aggfn_cat->altfn_name + 2;
				func_argtypes = buildoidvector(aggfn_cat->altfn_argtypes,
											   aggfn_cat->altfn_nargs);
				tuple = SearchSysCache3(PROCNAMEARGSNSP,
										PointerGetDatum(func_name),
										PointerGetDatum(func_argtypes),
										ObjectIdGetDatum(namespace_oid));
				if (!HeapTupleIsValid(tuple))
					elog(ERROR, "cache lookup failed for function %s",
						 funcname_signature_string(func_name,
												   aggfn_cat->altfn_nargs,
												   NIL,
												   aggfn_cat->altfn_argtypes));
				proc_form = (Form_pg_proc) GETSTRUCT(tuple);
				func_expr = makeFuncExpr(HeapTupleGetOid(tuple),
										 proc_form->prorettype,
										 altfn_args,
										 InvalidOid,
										 InvalidOid,
										 COERCE_EXPLICIT_CALL);
				tle_host = makeTargetEntry((Expr *)func_expr,
										   tle->resno,
										   tle->resname,
										   tle->resjunk);
				ReleaseSysCache(tuple);
			}
			tlist_host = lappend(tlist_host, tle_host);

			/*
			 * setting up Aggref node of the upper node
			 */
			if (strncmp(aggfn_cat->altfn_name, "c:", 2) == 0)
				namespace_oid = PG_CATALOG_NAMESPACE;
			else if (strncmp(aggfn_cat->altfn_name, "s:", 2) == 0)
				namespace_oid = get_namespace_oid("pgstrom", false);
			else
				elog(ERROR, "Bug? incorrect alternative function catalog");

			Assert(aggfn_cat->uppfn_argtype == exprType((Node *)tle->expr));

			func_name = aggfn_cat->uppfn_name;
			func_argtypes = buildoidvector(&aggfn_cat->uppfn_argtype, 1);
			tuple = SearchSysCache3(PROCNAMEARGSNSP,
									PointerGetDatum(func_name),
									PointerGetDatum(func_argtypes),
									ObjectIdGetDatum(namespace_oid));
			if (!HeapTupleIsValid(tuple))
				elog(ERROR, "cache lookup failed for function \"%s\"",
					 funcname_signature_string(func_name, 1, NIL,
											   &aggfn_cat->uppfn_argtype));
			aggfn_oid = HeapTupleGetOid(tuple);
			proc_form = (Form_pg_proc) GETSTRUCT(tuple);
			if (!proc_form->proisagg)
				elog(ERROR, "Bug? %s is not an aggregate function",
					 format_procedure(aggfn_oid));

			agg_tuple = SearchSysCache1(AGGFNOID, ObjectIdGetDatum(aggfn_oid));
			if (!HeapTupleIsValid(tuple))
				elog(ERROR, "cache lookup failed for aggregate function %s",
					 format_procedure(aggfn_oid));
			agg_form = (Form_pg_aggregate) GETSTRUCT(agg_tuple);

			aggref_new = makeNode(Aggref);
			aggref_new->aggfnoid = HeapTupleGetOid(tuple);
			aggref_new->aggtype = proc_form->prorettype;
			aggref_new->aggcollid = aggref->aggcollid;
			aggref_new->inputcollid = InvalidOid;
			aggref_new->aggtranstype = agg_form->aggtranstype;
			aggref_new->aggargtypes = list_make1_oid(aggfn_cat->uppfn_argtype);
			aggref_new->aggdirectargs = NIL;
			aggref_new->args = list_make1(copyObject(tle_host->expr));
			aggref_new->aggorder = NIL;
			aggref_new->aggdistinct = NIL;
			aggref_new->aggfilter = NULL;
			aggref_new->aggstar = false;
			aggref_new->aggvariadic = false;
			aggref_new->aggkind = AGGKIND_NORMAL;
			aggref_new->agglevelsup = 0;
			aggref_new->aggsplit = AGGSPLIT_SIMPLE;
			aggref_new->location = -1;

			tle_upper = makeTargetEntry((Expr *)aggref_new,
										tle->resno,
										tle->resname,
										tle->resjunk);
			tlist_upper = lappend(tlist_upper, tle_upper);

			ReleaseSysCache(agg_tuple);
			ReleaseSysCache(tuple);
		}
		else
		{
			if (!tlist_member((Node *)tle->expr, tlist_dev))
			{
				TargetEntry    *tle_new
					= makeTargetEntry(copyObject(tle->expr),
									  list_length(tlist_dev) + 1,
									  NULL,
									  false);
				tlist_dev = lappend(tlist_dev, tle_new);
			}
			tle_host = makeTargetEntry(copyObject(tle->expr),
									   tle->resno,
									   tle->resname,
									   tle->resjunk);
			tlist_host = lappend(tlist_host, tle_host);
			tlist_upper = lappend(tlist_upper, tle_host);
		}
	}
	/* setup results */
	Assert(list_length(target_orig->exprs) == list_length(tlist_upper));
	target_upper->exprs = tlist_upper;
	target_upper->sortgrouprefs = target_orig->sortgrouprefs;
	set_pathtarget_cost_width(root, target_upper);
	target_host->exprs  = tlist_host;
	set_pathtarget_cost_width(root, target_host);
	target_dev->exprs   = tlist_dev;
	set_pathtarget_cost_width(root, target_dev);
}

#if 0
/*
 * build_custom_scan_tlist
 *
 * constructor for the custom_scan_tlist of CustomScan node. It is equivalent
 * to the initial values of reduction steps.
 */
static void
build_custom_scan_tlist(PathTarget *target,
						List *tlist_orig,
						List **p_tlist_host,
						List **p_tlist_dev,
						List **p_tlist_dev_action)
{
	List	   *tlist_host = NIL;
	List	   *tlist_dev = NIL;
	List	   *tlist_dev_action = NIL;
	ListCell   *lc;
	cl_int		index = 0;

	foreach (lc, tlist_orig)
	{
		TargetEntry	   *tle = lfirst(lc);

		if (IsA(tle->expr, Aggref))
		{
			Aggref	   *aggref = (Aggref *)tle->expr;
			List	   *altfn_args = NIL;
			cl_int		j;
			const aggfunc_catalog_t *aggfn_cat;

			Assert(target->sortgrouprefs == NULL ||
				   target->sortgrouprefs[index] == 0);
			aggfn_cat = aggfunc_lookup_by_oid(aggref->aggfnoid);
			if (!aggfn_cat)
				elog(ERROR, "lookup failed on aggregate function: %u",
					 aggref->aggfnoid);

			/*
			 * construction of the initial partial aggregation
			 */
			for (j=0; j < aggfn_cat->altfn_nargs; j++)
			{
				ListCell	   *cell1;
				ListCell	   *cell2;
				cl_int			action = aggfn_cat->altfn_argexprs[j];
				Oid				argtype = aggfn_cat->altfn_argtypes[j];
				Expr		   *expr;
				TargetEntry	   *temp;
				cl_int			temp_action;

				switch (action)
				{
					case ALTFUNC_EXPR_NROWS:	/* NROWS(X) */
						expr = make_altfunc_nrows_expr(aggref);
						break;
					case ALTFUNC_EXPR_PMIN:		/* PMIN(X) */
					case ALTFUNC_EXPR_PMAX:		/* PMAX(X) */
						expr = make_altfunc_simple_expr(aggref, false);
						break;
					case ALTFUNC_EXPR_PSUM:		/* PSUM(X) */
						expr = make_altfunc_simple_expr(aggref, true);
						break;
					case ALTFUNC_EXPR_PSUM_X2:	/* PSUM_X2(X) = PSUM(X^2) */
						expr = make_altfunc_psum_x2(aggref);
						break;
					case ALTFUNC_EXPR_PCOV_X:	/* PCOV_X(X,Y) */
					case ALTFUNC_EXPR_PCOV_Y:	/* PCOV_Y(X,Y) */
					case ALTFUNC_EXPR_PCOV_X2:	/* PCOV_X2(X,Y) */
					case ALTFUNC_EXPR_PCOV_Y2:	/* PCOV_Y2(X,Y) */
					case ALTFUNC_EXPR_PCOV_XY:	/* PCOV_XY(X,Y) */
						expr = make_altfunc_pcov_xy(aggref, action);
						break;
					default:
						elog(ERROR, "unknown alternative function code: %d",
							 action);
						break;
				}
				/* force type cast if mismatch */
				expr = make_expr_typecast(expr, argtype);

				/*
				 * lookup same entity on the tlist_dev, then append it
				 * if not found. Resno is tracked to construct FuncExpr.
				 */
				forboth (cell1, tlist_dev,
						 cell2, tlist_dev_action)
				{
					temp = lfirst(cell1);
					temp_action = lfirst_int(cell2);

					if (temp_action == action &&
						equal(expr, temp->expr))
						goto found_tlist_dev_entry;
				}
				temp = makeTargetEntry(expr,
									   list_length(tlist_dev) + 1,
									   NULL,
									   false);
				tlist_dev = lappend(tlist_dev, temp);
				tlist_dev_action = lappend_int(tlist_dev_action, action);

			found_tlist_dev_entry:
				altfn_args = lappend(altfn_args, expr);
			}

			/*
			 * Lookup an alternative function that generates partial state
			 * of the final aggregate function, or varref if internal state
			 * of aggregation is as-is.
			 */
			if (strcmp(aggfn_cat->altfn_name, "varref") == 0)
			{
				Assert(list_length(altfn_args) == 1);

				tlist_host = lappend(tlist_host,
									 makeTargetEntry(linitial(altfn_args),
													 tle->resno,
													 tle->resname,
													 tle->resjunk));
			}
			else
			{
				Oid				namespace_oid;
				FuncExpr	   *altfn_expr;
				const char	   *altfn_name;
				oidvector	   *altfn_argtypes;
				HeapTuple		tuple;
				Form_pg_proc	altfn_form;

				if (strncmp(aggfn_cat->altfn_name, "c:", 2) == 0)
					namespace_oid = PG_CATALOG_NAMESPACE;
				else if (strncmp(aggfn_cat->altfn_name, "s:", 2) == 0)
					namespace_oid = get_namespace_oid("pgstrom", false);
				else
					elog(ERROR, "Bug? incorrect alternative function catalog");

				altfn_name = aggfn_cat->altfn_name + 2;
				altfn_argtypes = buildoidvector(aggfn_cat->altfn_argtypes,
												aggfn_cat->altfn_nargs);
				tuple = SearchSysCache3(PROCNAMEARGSNSP,
										PointerGetDatum(altfn_name),
										PointerGetDatum(altfn_argtypes),
										ObjectIdGetDatum(namespace_oid));
				if (!HeapTupleIsValid(tuple))
					ereport(ERROR,
							(errcode(ERRCODE_UNDEFINED_SCHEMA),
							 errmsg("no alternative function \"%s\" not found",
									funcname_signature_string(
										altfn_name,
										aggfn_cat->altfn_nargs,
										NIL,
										aggfn_cat->altfn_argtypes)),
							 errhint("Run: CREATE EXTENSION pg_strom")));
				altfn_form = (Form_pg_proc) GETSTRUCT(tuple);

				altfn_expr = makeNode(FuncExpr);
				altfn_expr->funcid = HeapTupleGetOid(tuple);
				altfn_expr->funcresulttype = altfn_form->prorettype;
				altfn_expr->funcretset = altfn_form->proretset;
				altfn_expr->funcvariadic = OidIsValid(altfn_form->provariadic);
				altfn_expr->funcformat = COERCE_EXPLICIT_CALL;
				altfn_expr->funccollid = aggref->aggcollid;
				altfn_expr->inputcollid = aggref->inputcollid;
				altfn_expr->args = altfn_args;
				altfn_expr->location = aggref->location;

				ReleaseSysCache(tuple);

				tlist_host = lappend(tlist_host,
									 makeTargetEntry((Expr *)altfn_expr,
													 tle->resno,
													 tle->resname,
													 tle->resjunk));
			}
		}
		else
		{
			tlist_dev = lappend(tlist_dev, copyObject(tle));
			tlist_dev_action = lappend_int(tlist_dev_action,
										   target->sortgrouprefs == NULL ||
										   target->sortgrouprefs[index] > 0
										   ? ALTFUNC_GROUPING_KEY
										   : ALTFUNC_CONST_VALUE);
			tlist_host = lappend(tlist_host, copyObject(tle));
		}
		index++;
	}
	/* return the results */
	*p_tlist_host = tlist_host;
	*p_tlist_dev = tlist_dev;
	*p_tlist_dev_action = tlist_dev_action;
}
#endif

/*
 * PlanGpuPreAggPath
 *
 * Entrypoint to create CustomScan node
 */
static Plan *
PlanGpuPreAggPath(PlannerInfo *root,
				  RelOptInfo *rel,
				  struct CustomPath *best_path,
				  List *tlist,
				  List *clauses,
				  List *custom_plans)
{
	CustomScan	   *cscan = makeNode(CustomScan);
	GpuPreAggInfo  *gpa_info;
	Plan		   *outer_plan = NULL;
	List		   *outer_tlist = NIL;
	List		   *tlist_host;
	List		   *tlist_dev;
	List		   *tlist_dev_action;
	char		   *kern_source;
	codegen_context	context;

	Assert(list_length(custom_plans) <= 1);
	Assert(list_length(best_path->custom_private) == 1);
	if (custom_plans != NIL)
	{
		outer_plan = linitial(custom_plans);
		outer_tlist = outer_plan->targetlist;
	}
	gpa_info = linitial(best_path->custom_private);

	/*
	 * construction of the alternative targetlist.
	 * @tlist_host: tlist of partial aggregation status
	 * @tlist_dev:  tlist of initial state on device reduction.
	 * @tlist_dev_action: one of ALTFUNC_* for each tlist_dev
	 */
	build_custom_scan_tlist(best_path->path.pathtarget,
							tlist,
							&tlist_host,
							&tlist_dev,
							&tlist_dev_action);

	cscan->scan.plan.targetlist = tlist_host;
	cscan->scan.plan.qual = NIL;
	outerPlan(cscan) = outer_plan;
	cscan->scan.scanrelid = gpa_info->outer_scanrelid;
	cscan->flags = best_path->flags;
	cscan->custom_scan_tlist = tlist_dev;
	cscan->methods = &gpupreagg_scan_methods;

	/*
	 * construction of the GPU kernel code
	 */
	pgstrom_init_codegen_context(&context);
	context.extra_flags |= (DEVKERNEL_NEEDS_DYNPARA |
							DEVKERNEL_NEEDS_GPUPREAGG);
	kern_source = gpupreagg_codegen(&context,
									root,
									cscan,
									tlist_dev,
									tlist_dev_action,
									outer_tlist,
									gpa_info->outer_quals);
//	elog(INFO, "source:\n%s", kern_source);

	gpa_info->kern_source = kern_source;
	gpa_info->extra_flags = context.extra_flags;
	gpa_info->used_params = context.used_params;

//	elog(INFO, "tlist_orig => %s", nodeToString(tlist));
//	elog(INFO, "tlist_dev => %s", nodeToString(tlist_dev));
//	elog(INFO, "tlist_dev_action => %s", nodeToString(tlist_dev_action));
//	elog(INFO, "used_params => %s", nodeToString(gpa_info->used_params));

	form_gpupreagg_info(cscan, gpa_info);

	return &cscan->scan.plan;
}

/*
 * pgstrom_plan_is_gpupreagg - returns true if GpuPreAgg
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
 * make_tlist_device_projection
 *
 * It pulls a set of referenced resource numbers according to the supplied
 * outer_scanrelid/outer_tlist.
 */
typedef struct
{
	Bitmapset  *outer_refs;
	Index		outer_scanrelid;
	List	   *outer_tlist;
} make_tlist_device_projection_context;

static Node *
__make_tlist_device_projection(Node *node, void *__con)
{
	make_tlist_device_projection_context *con = __con;
	int		k;

	if (!node)
		return false;
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
			con->outer_refs = bms_add_member(con->outer_refs, k);

			Assert(varnode->varlevelsup == 0);
			return (Node *)makeVar(INDEX_VAR,
								   varnode->varattno,
								   varnode->vartype,
								   varnode->vartypmod,
								   varnode->varcollid,
								   varnode->varlevelsup);
		}
	}
	else
	{
		ListCell	   *lc;

		foreach (lc, con->outer_tlist)
		{
			TargetEntry    *tle = lfirst(lc);
			Var			   *varnode;

			if (equal(node, tle->expr))
			{
				k = tle->resno - FirstLowInvalidHeapAttributeNumber;
				con->outer_refs = bms_add_member(con->outer_refs, k);

				varnode = makeVar(INDEX_VAR,
								  tle->resno,
								  exprType((Node *)tle->expr),
								  exprTypmod((Node *)tle->expr),
								  exprCollation((Node *)tle->expr),
								  0);
				return (Node *)varnode;
			}
		}

		if (IsA(node, Var))
			elog(ERROR, "Bug? varnode (%s) references unknown outer entry: %s",
				 nodeToString(node),
				 nodeToString(con->outer_tlist));
	}
	return expression_tree_mutator(node, __make_tlist_device_projection, con);
}

static List *
make_tlist_device_projection(List *tlist_dev,
							 Index outer_scanrelid,
							 List *outer_tlist,
							 Bitmapset **p_outer_refs)
{
	make_tlist_device_projection_context con;
	List	   *tlist_alt;

	memset(&con, 0, sizeof(con));
	con.outer_scanrelid = outer_scanrelid;
	con.outer_tlist = outer_tlist;

	tlist_alt = (List *)
		__make_tlist_device_projection((Node *)tlist_dev, &con);
	*p_outer_refs = con.outer_refs;

	return tlist_alt;
}

/*
 * gpupreagg_codegen_projection - code generator for
 *
 * STATIC_FUNCTION(void)
 * gpupreagg_projection(kern_context *kcxt,
 *                      kern_data_store *kds_src,
 *                      kern_tupitem *tupitem,
 *                      kern_data_store *kds_dst,
 *                      Datum *dst_values,
 *                      cl_char *dst_isnull);
 */
static void
gpupreagg_codegen_projection(StringInfo kern,
							 codegen_context *context,
							 PlannerInfo *root,
							 List *tlist_dev,
							 List *tlist_dev_action,
							 Index outer_scanrelid,
							 List *outer_tlist)
{
	StringInfoData	decl;
	StringInfoData	body;
	StringInfoData	temp;
	Relation		outer_rel = NULL;
	TupleDesc		outer_desc = NULL;
	Bitmapset	   *outer_refs = NULL;
	List		   *tlist_alt;
	ListCell	   *lc1;
	ListCell	   *lc2;
	int				i, k, nattrs;

	initStringInfo(&decl);
	initStringInfo(&body);
	initStringInfo(&temp);
	context->param_refs = NULL;

	appendStringInfoString(
		&decl,
		"STATIC_FUNCTION(void)\n"
		"gpupreagg_projection(kern_context *kcxt,\n"
		"                     kern_data_store *kds_src,\n"
		"                     HeapTupleHeaderData *htup,\n"
		"                     kern_data_store *kds_dst,\n"
		"                     Datum *dst_values,\n"
		"                     cl_char *dst_isnull)\n"
		"{\n"
		"  void        *addr    __attribute__((unused));\n"
		"  pg_anytype_t temp    __attribute__((unused));\n");

	/* open relation if GpuPreAgg looks at physical relation */
	if (outer_tlist == NIL)
	{
		RangeTblEntry  *rte;

		Assert(outer_scanrelid > 0 &&
			   outer_scanrelid < root->simple_rel_array_size);
		rte = root->simple_rte_array[outer_scanrelid];
		outer_rel = heap_open(rte->relid, NoLock);
		outer_desc = RelationGetDescr(outer_rel);
		nattrs = outer_desc->natts;
	}
	else
	{
		Assert(outer_scanrelid == 0);
		nattrs = list_length(outer_tlist);
	}

	/* pick up columns which are referenced on the initial projection */
	tlist_alt = make_tlist_device_projection(tlist_dev,
											 outer_scanrelid,
											 outer_tlist,
											 &outer_refs);
	Assert(list_length(tlist_alt) == list_length(tlist_dev));

	/* extract the supplied tuple and load variables */
	if (!bms_is_empty(outer_refs))
	{
		for (i=0; i > FirstLowInvalidHeapAttributeNumber; i--)
		{
			k = i - FirstLowInvalidHeapAttributeNumber;
			if (bms_is_member(k, outer_refs))
				elog(ERROR, "Bug? system column or whole-row is referenced");
		}

		appendStringInfoString(
			&body,
			"\n"
			"  /* extract the given htup and load variables */\n"
			"  EXTRACT_HEAP_TUPLE_BEGIN(addr, kds_src, htup);\n");
		for (i=1; i <= nattrs; i++)
		{
			k = i - FirstLowInvalidHeapAttributeNumber;
			if (bms_is_member(k, outer_refs))
			{
				devtype_info   *dtype;

				/* data type of the outer relation input stream */
				if (outer_tlist == NIL)
				{
					Form_pg_attribute	attr = outer_desc->attrs[i-1];
					
					dtype = pgstrom_devtype_lookup_and_track(attr->atttypid,
															 context);
					if (!dtype)
						elog(ERROR, "device type lookup failed: %s",
							 format_type_be(attr->atttypid));
				}
				else
				{
					TargetEntry	   *tle = list_nth(outer_tlist, i-1);
					Oid				type_oid = exprType((Node *)tle->expr);

					dtype = pgstrom_devtype_lookup_and_track(type_oid,
															 context);
					if (!dtype)
						elog(ERROR, "device type lookup failed: %s",
							 format_type_be(type_oid));
				}

				appendStringInfo(
					&decl,
					"  pg_%s_t KVAR_%u;\n",
					dtype->type_name, i);
				appendStringInfo(
					&temp,
					"  KVAR_%u = pg_%s_datum_ref(kcxt,addr,false);\n",
					i, dtype->type_name);
				/*
				 * MEMO: kds_src is either ROW or BLOCK format, so these KDS
				 * shall never have 'internal' format of NUMERIC data types.
				 */
				appendStringInfoString(&body, temp.data);
				resetStringInfo(&temp);
			}
			appendStringInfoString(
				&temp,
				"  EXTRACT_HEAP_TUPLE_NEXT(addr);\n");
		}
		appendStringInfoString(
			&body,
			"  EXTRACT_HEAP_TUPLE_END();\n");
	}

	/*
	 * Execute expression and store the value on dst_values/dst_isnull
	 */
	forboth (lc1, tlist_alt,
			 lc2, tlist_dev_action)
	{
		TargetEntry	   *tle = lfirst(lc1);
		Expr		   *expr = tle->expr;
		Oid				type_oid = exprType((Node *)expr);
		int				action = lfirst_int(lc2);
		devtype_info   *dtype;
		char		   *kvar_label;

		dtype = pgstrom_devtype_lookup_and_track(type_oid, context);
		if (!dtype)
			elog(ERROR, "device type lookup failed: %s",
				 format_type_be(type_oid));
		appendStringInfo(
			&body,
			"\n"
			"  /* initial attribute %d (%s) */\n",
			tle->resno,
			(action == ALTFUNC_GROUPING_KEY ? "group-key" :
			 action <  ALTFUNC_EXPR_NROWS   ? "const-value" : "aggfn-arg"));

		if (IsA(expr, Var))
		{
			Var	   *varnode = (Var *)expr;

			Assert(varnode->varno == INDEX_VAR);
			kvar_label = psprintf("KVAR_%u", varnode->varattno);
		}
		else
		{
			kvar_label = psprintf("temp.%s_v", dtype->type_name);
			appendStringInfo(
				&body,
				"  %s = %s;\n",
				kvar_label,
				pgstrom_codegen_expression((Node *)expr, context));
		}

		appendStringInfo(
			&body,
			"  dst_isnull[%d] = %s.isnull;\n"
			"  if (!%s.isnull)\n"
			"    dst_values[%d] = pg_%s_to_datum(%s.value);\n",
			tle->resno - 1, kvar_label,
			kvar_label,
			tle->resno - 1, dtype->type_name, kvar_label);
		/*
		 * dst_value must be also initialized to an proper initial value,
		 * even if dst_isnull would be NULL, because atomic operation
		 * expects dst_value has a particular initial value.
		 */
		if (action >= ALTFUNC_EXPR_NROWS)
		{
			const char	   *null_const_value;

			switch (action)
			{
				case ALTFUNC_EXPR_PMIN:
					null_const_value = dtype->min_const;
					break;
				case ALTFUNC_EXPR_PMAX:
					null_const_value = dtype->max_const;
					break;
				default:
					null_const_value = dtype->zero_const;
					break;
			}

			if (!null_const_value)
				elog(ERROR, "Bug? unable to use type %s in GpuPreAgg",
					 format_type_be(dtype->type_oid));

			appendStringInfo(
				&body,
				"  else\n"
				"    dst_values[%d] = pg_%s_to_datum(%s);\n",
				tle->resno - 1, dtype->type_name, null_const_value);
		}
	}
	/* const/params */
	pgstrom_codegen_param_declarations(&decl, context);
	appendStringInfo(
		&decl,
		"%s"
		"}\n\n", body.data);

	if (outer_rel)
		heap_close(outer_rel, NoLock);

	appendStringInfoString(kern, decl.data);
	pfree(decl.data);
	pfree(body.data);
}

/*
 * gpupreagg_codegen_hashvalue - code generator for
 *
 * STATIC_FUNCTION(cl_uint)
 * gpupreagg_hashvalue(kern_context *kcxt,
 *                     cl_uint *crc32_table,
 *                     cl_uint hash_value,
 *                     kern_data_store *kds,
 *                     size_t kds_index);
 */
static void
gpupreagg_codegen_hashvalue(StringInfo kern,
							codegen_context *context,
							List *tlist_dev,
							List *tlist_dev_action)
{
	StringInfoData	decl;
	StringInfoData	body;
	ListCell	   *lc1;
	ListCell	   *lc2;

	initStringInfo(&decl);
    initStringInfo(&body);
	context->param_refs = NULL;

	appendStringInfo(
		&decl,
		"STATIC_FUNCTION(cl_uint)\n"
		"gpupreagg_hashvalue(kern_context *kcxt,\n"
		"                    cl_uint *crc32_table,\n"
		"                    cl_uint hash_value,\n"
		"                    kern_data_store *kds,\n"
		"                    size_t kds_index)\n"
		"{\n");

	forboth (lc1, tlist_dev,
			 lc2, tlist_dev_action)
	{
		TargetEntry	   *tle = lfirst(lc1);
		int				action = lfirst_int(lc2);
		Oid				type_oid;
		devtype_info   *dtype;

		if (action != ALTFUNC_GROUPING_KEY)
			continue;

		type_oid = exprType((Node *)tle->expr);
		dtype = pgstrom_devtype_lookup_and_track(type_oid, context);
		if (!dtype || !OidIsValid(dtype->type_cmpfunc))
			elog(ERROR, "Bug? type (%s) is not supported",
				 format_type_be(type_oid));
		/* variable declarations */
		appendStringInfo(
			&decl,
			"  pg_%s_t keyval_%u = pg_%s_vref(kds,kcxt,%u,kds_index);\n",
			dtype->type_name, tle->resno,
			dtype->type_name, tle->resno - 1);
		/* compute crc32 value */
		appendStringInfo(
			&body,
			"  hash_value = pg_%s_comp_crc32(crc32_table, hash_value, keyval_%u);\n",
			dtype->type_name, tle->resno);
	}
	/* no constants should appear */
	Assert(bms_is_empty(context->param_refs));

	appendStringInfo(kern,
					 "%s\n"
					 "%s\n"
					 "  return hash_value;\n"
					 "}\n\n",
					 decl.data,
					 body.data);
	pfree(decl.data);
	pfree(body.data);
}

/*
 * gpupreagg_codegen_keymatch - code generator for
 *
 *
 * STATIC_FUNCTION(cl_bool)
 * gpupreagg_keymatch(kern_context *kcxt,
 *                    kern_data_store *x_kds, size_t x_index,
 *                    kern_data_store *y_kds, size_t y_index);
 */
static void
gpupreagg_codegen_keymatch(StringInfo kern,
						   codegen_context *context,
						   List *tlist_dev,
						   List *tlist_dev_action)
{
	StringInfoData	decl;
	StringInfoData	body;
	ListCell	   *lc1;
	ListCell	   *lc2;

	initStringInfo(&decl);
	initStringInfo(&body);
	context->param_refs = NULL;

	appendStringInfoString(
		kern,
		"STATIC_FUNCTION(cl_bool)\n"
		"gpupreagg_keymatch(kern_context *kcxt,\n"
		"                   kern_data_store *x_kds, size_t x_index,\n"
		"                   kern_data_store *y_kds, size_t y_index)\n"
		"{\n"
		"  pg_anytype_t temp_x  __attribute__((unused));\n"
		"  pg_anytype_t temp_y  __attribute__((unused));\n"
		"\n");

	forboth (lc1, tlist_dev,
			 lc2, tlist_dev_action)
	{
		TargetEntry	   *tle = lfirst(lc1);
		int				action = lfirst_int(lc2);
		Oid				type_oid;
		Oid				coll_oid;
		devtype_info   *dtype;
		devfunc_info   *dfunc;

		if (action != ALTFUNC_GROUPING_KEY)
			continue;

		/* find the function to compare this data-type */
		type_oid = exprType((Node *)tle->expr);
		coll_oid = exprCollation((Node *)tle->expr);
		dtype = pgstrom_devtype_lookup_and_track(type_oid, context);
		if (!dtype || !OidIsValid(dtype->type_eqfunc))
			elog(ERROR, "Bug? type (%s) has no device comparison function",
				 format_type_be(type_oid));

		dfunc = pgstrom_devfunc_lookup_and_track(dtype->type_eqfunc,
												 coll_oid,
												 context);
		if (!dfunc)
			elog(ERROR, "Bug? device function (%u) was not found",
				 dtype->type_eqfunc);

		/* load the key values, and compare */
		appendStringInfo(
			kern,
			"  temp_x.%s_v = pg_%s_vref(x_kds,kcxt,%u,x_index);\n"
			"  temp_y.%s_v = pg_%s_vref(y_kds,kcxt,%u,y_index);\n"
			"  if (!temp_x.%s_v.isnull && !temp_y.%s_v.isnull)\n"
			"  {\n"
			"    if (!EVAL(pgfn_%s(kcxt, temp_x.%s_v, temp_y.%s_v)))\n"
			"      return false;\n"
			"  }\n"
			"  else if ((temp_x.%s_v.isnull && !temp_y.%s_v.isnull) ||\n"
			"           (!temp_x.%s_v.isnull && temp_y.%s_v.isnull))\n"
			"      return false;\n"
			"\n",
			dtype->type_name, dtype->type_name, tle->resno - 1,
			dtype->type_name, dtype->type_name, tle->resno - 1,
			dtype->type_name, dtype->type_name,
			dfunc->func_devname, dtype->type_name, dtype->type_name,
			dtype->type_name, dtype->type_name,
			dtype->type_name, dtype->type_name);
	}
	/* no constant values should be referenced */
	Assert(bms_is_empty(context->param_refs));

	appendStringInfoString(
		kern,
		"  return true;\n"
		"}\n\n");
}

/*
 * gpupreagg_codegen_common_calc
 *
 * common portion of the gpupreagg_xxxx_calc() kernels
 */
static void
gpupreagg_codegen_common_calc(StringInfo kern,
							  codegen_context *context,
							  List *tlist_dev,
							  List *tlist_dev_action,
							  const char *aggcalc_class,
							  const char *aggcalc_args)
{
	ListCell   *lc1;
	ListCell   *lc2;

	appendStringInfoString(
		kern,
		"  switch (attnum)\n"
		"  {\n");

	forboth (lc1, tlist_dev,
			 lc2, tlist_dev_action)
	{
		TargetEntry	   *tle = lfirst(lc1);
		int				action = lfirst_int(lc2);
		Oid				type_oid = exprType((Node *)tle->expr);
		devtype_info   *dtype;
		const char	   *aggcalc_ops;
		const char	   *aggcalc_type;

		/* not aggregate-function's argument */
		if (action < ALTFUNC_EXPR_NROWS)
			continue;

		dtype = pgstrom_devtype_lookup_and_track(type_oid, context);
		if (!dtype)
			elog(ERROR, "failed on device type lookup: %s",
				 format_type_be(type_oid));

		switch (dtype->type_oid)
		{
			case INT2OID:
				aggcalc_type = "SHORT";
				break;
			case INT4OID:
			case DATEOID:
				aggcalc_type = "INT";
				break;
			case INT8OID:
			case CASHOID:
			case TIMEOID:
			case TIMESTAMPOID:
			case TIMESTAMPTZOID:
				aggcalc_type = "LONG";
				break;
			case FLOAT4OID:
				aggcalc_type = "FLOAT";
				break;
			case FLOAT8OID:
				aggcalc_type = "DOUBLE";
				break;
			case NUMERICOID:
				aggcalc_type = "NUMERIC";
				break;
			default:
				elog(ERROR, "Bug? %s is not expected to use for GpuPreAgg",
					 format_type_be(dtype->type_oid));
		}

		if (action == ALTFUNC_EXPR_PMIN)
			aggcalc_ops = "PMIN";
		else if (action == ALTFUNC_EXPR_PMAX)
			aggcalc_ops = "PMAX";
		else
			aggcalc_ops = "PADD";

		appendStringInfo(
			kern,
			"  case %d:\n"
			"    AGGCALC_%s_%s_%s(%s);\n"
			"    break;\n",
			tle->resno - 1,
			aggcalc_class,
			aggcalc_ops,
			aggcalc_type,
			aggcalc_args);
	}
	appendStringInfoString(
		kern,
		"  default:\n"
		"    break;\n"
		"  }\n");
}

/*
 * gpupreagg_codegen_local_calc - code generator for
 *
 * STATIC_FUNCTION(void)
 * gpupreagg_local_calc(kern_context *kcxt,
 *                      cl_int attnum,
 *                      pagg_datum *accum,
 *                      pagg_datum *newval);
 */
static void
gpupreagg_codegen_local_calc(StringInfo kern,
							 codegen_context *context,
							 List *tlist_dev,
							 List *tlist_dev_action)
{
	appendStringInfoString(
		kern,
		"STATIC_FUNCTION(void)\n"
		"gpupreagg_local_calc(kern_context *kcxt,\n"
		"                     cl_int attnum,\n"
		"                     pagg_datum *accum,\n"
		"                     pagg_datum *newval)\n"
		"{\n");
	gpupreagg_codegen_common_calc(kern,
								  context,
								  tlist_dev,
								  tlist_dev_action,
								  "LOCAL",
								  "kcxt,accum,newval");
	appendStringInfoString(
		kern,
		"}\n\n");
}

/*
 * gpupreagg_codegen_global_calc - code generator for
 *
 * STATIC_FUNCTION(void)
 * gpupreagg_global_calc(kern_context *kcxt,
 *                       cl_int attnum,
 *                       kern_data_store *accum_kds,  size_t accum_index,
 *                       kern_data_store *newval_kds, size_t newval_index);
 */
static void
gpupreagg_codegen_global_calc(StringInfo kern,
							  codegen_context *context,
							  List *tlist_dev,
							  List *tlist_dev_action)
{
	appendStringInfoString(
		kern,
		"STATIC_FUNCTION(void)\n"
		"gpupreagg_global_calc(kern_context *kcxt,\n"
		"                      cl_int attnum,\n"
		"                      kern_data_store *accum_kds,\n"
		"                      size_t accum_index,\n"
		"                      kern_data_store *newval_kds,\n"
		"                      size_t newval_index)\n"
		"{\n"
		"  char    *accum_isnull    __attribute__((unused))\n"
		"   = KERN_DATA_STORE_ISNULL(accum_kds,accum_index) + attnum;\n"
		"  Datum   *accum_value     __attribute__((unused))\n"
		"   = KERN_DATA_STORE_VALUES(accum_kds,accum_index) + attnum;\n"
		"  char     new_isnull      __attribute__((unused))\n"
		"   = KERN_DATA_STORE_ISNULL(newval_kds,newval_index)[attnum];\n"
		"  Datum    new_value       __attribute__((unused))\n"
		"   = KERN_DATA_STORE_VALUES(newval_kds,newval_index)[attnum];\n"
		"\n"
		"  assert(accum_kds->format == KDS_FORMAT_SLOT);\n"
		"  assert(newval_kds->format == KDS_FORMAT_SLOT);\n"
		"\n");
	gpupreagg_codegen_common_calc(kern,
								  context,
								  tlist_dev,
								  tlist_dev_action,
								  "GLOBAL",
						"kcxt,accum_isnull,accum_value,new_isnull,new_value");
	appendStringInfoString(
		kern,
		"}\n\n");
}

/*
 * gpupreagg_codegen_nogroup_calc - code generator for
 *
 * STATIC_FUNCTION(void)
 * gpupreagg_nogroup_calc(kern_context *kcxt,
 *                        cl_int attnum,
 *                        pagg_datum *accum,
 *                        pagg_datum *newval);
 */
static void
gpupreagg_codegen_nogroup_calc(StringInfo kern,
							   codegen_context *context,
							   List *tlist_dev,
							   List *tlist_dev_action)
{
	appendStringInfoString(
        kern,
		"STATIC_FUNCTION(void)\n"
		"gpupreagg_nogroup_calc(kern_context *kcxt,\n"
		"                       cl_int attnum,\n"
		"                       pagg_datum *accum,\n"
		"                       pagg_datum *newval)\n"
		"{\n");
	gpupreagg_codegen_common_calc(kern,
								  context,
                                  tlist_dev,
                                  tlist_dev_action,
								  "NOGROUP",
								  "kcxt,accum,newval");
	appendStringInfoString(
        kern,
		"}\n\n");
}

/*
 * gpupreagg_codegen - entrypoint of code-generator for GpuPreAgg
 */
static char *
gpupreagg_codegen(codegen_context *context,
				  PlannerInfo *root,
				  CustomScan *cscan,
				  List *tlist_dev,
				  List *tlist_dev_action,
				  List *outer_tlist,
				  List *outer_quals)
{
	StringInfoData	kern;
	StringInfoData	body;
	Size			length;
	bytea		   *kparam_0;
	ListCell	   *lc;
	int				i = 0;

	initStringInfo(&kern);
	initStringInfo(&body);
	/*
	 * System constants of GpuPreAgg:
	 * KPARAM_0 is an array of cl_char to inform which field is grouping
	 * keys, or target of (partial) aggregate function.
	 */
	length = sizeof(cl_char) * list_length(tlist_dev_action);
	kparam_0 = palloc0(length + VARHDRSZ);
	SET_VARSIZE(kparam_0, length + VARHDRSZ);
	foreach (lc, tlist_dev_action)
	{
		int		action = lfirst_int(lc);

		((cl_char *)VARDATA(kparam_0))[i++] = (action == ALTFUNC_GROUPING_KEY);
	}
	context->used_params = list_make1(makeConst(BYTEAOID,
												-1,
												InvalidOid,
												-1,
												PointerGetDatum(kparam_0),
												false,
												false));
	pgstrom_devtype_lookup_and_track(BYTEAOID, context);

	/* gpuscan_quals_eval (optional) */
	if (cscan->scan.scanrelid > 0)
	{
		codegen_gpuscan_quals(&kern, context,
							  cscan->scan.scanrelid,
							  outer_quals);
		context->extra_flags |= DEVKERNEL_NEEDS_GPUSCAN;
	}
	/* gpupreagg_projection */
	gpupreagg_codegen_projection(&kern, context, root,
								 tlist_dev, tlist_dev_action,
								 cscan->scan.scanrelid, outer_tlist);
	/* gpupreagg_hashvalue */
	gpupreagg_codegen_hashvalue(&kern, context,
								tlist_dev, tlist_dev_action);
	/* gpupreagg_keymatch */
	gpupreagg_codegen_keymatch(&kern, context,
							   tlist_dev, tlist_dev_action);
	/* gpupreagg_local_calc */
	gpupreagg_codegen_local_calc(&kern, context,
								 tlist_dev, tlist_dev_action);
	/* gpupreagg_global_calc */
	gpupreagg_codegen_global_calc(&kern, context,
								  tlist_dev, tlist_dev_action);
	/* gpupreagg_nogroup_calc */
	gpupreagg_codegen_nogroup_calc(&kern, context,
								   tlist_dev, tlist_dev_action);
	/* function declarations */
	pgstrom_codegen_func_declarations(&kern, context);
	/* special expression declarations */
	pgstrom_codegen_expr_declarations(&kern, context);
	/* merge above kernel functions */
	appendStringInfoString(&kern, body.data);
	pfree(body.data);

	return kern.data;
}

/*
 * assign_gpupreagg_session_info
 */
void
assign_gpupreagg_session_info(StringInfo buf, GpuTaskState_v2 *gts)
{
	CustomScan	   *cscan = (CustomScan *)gts->css.ss.ps.plan;

	Assert(pgstrom_plan_is_gpupreagg(&cscan->scan.plan));
	/*
	 * Put GPUPREAGG_PULLUP_OUTER_SCAN if GpuPreAgg pulled up outer scan
	 * node regardless of the outer-quals (because KDS may be BLOCK format,
	 * and only gpuscan_exec_quals_block() can extract it).
	 */
	if (cscan->scan.scanrelid > 0)
		appendStringInfo(buf, "#define GPUPREAGG_PULLUP_OUTER_SCAN 1\n");
}

/*
 * CreateGpuPreAggScanState - constructor of GpuPreAggState
 */
static Node *
CreateGpuPreAggScanState(CustomScan *cscan)
{
	GpuPreAggState *gpas = palloc0(sizeof(GpuPreAggState));

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
	GpuContext_v2  *gcontext = NULL;
	GpuPreAggState *gpas = (GpuPreAggState *) node;
	CustomScan	   *cscan = (CustomScan *) node->ss.ps.plan;
	GpuPreAggInfo  *gpa_info = deform_gpupreagg_info(cscan);
	List		   *pseudo_tlist;
	TupleDesc		pseudo_tupdesc;
	TupleDesc		outer_tupdesc;
	char		   *kern_define;
	ProgramId		program_id;
	bool			has_oid;
	bool			with_connection = ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0);

	Assert(gpa_info->outer_scanrelid == cscan->scan.scanrelid);
	Assert(scan_rel ? outerPlan(node) == NULL : outerPlan(node) != NULL);
	/* activate a GpuContext for CUDA kernel execution */
	gcontext = AllocGpuContext(with_connection);

	/* setup common GpuTaskState fields */
	pgstromInitGpuTaskState(&gpas->gts,
							gcontext,
							GpuTaskKind_GpuPreAgg,
							gpa_info->used_params,
							estate);
	gpas->gts.cb_next_task   = gpupreagg_next_task;
	gpas->gts.cb_ready_task  = gpupreagg_ready_task;
	gpas->gts.cb_switch_task = gpupreagg_switch_task;
	gpas->gts.cb_next_tuple  = gpupreagg_next_tuple;

	gpas->num_group_keys     = gpa_info->num_group_keys;

	/* initialization of the outer relation */
	if (outerPlan(cscan))
	{
		PlanState  *outer_ps;

		Assert(scan_rel == NULL);
		Assert(gpa_info->outer_quals == NIL);
		outer_ps = ExecInitNode(outerPlan(cscan), estate, eflags);
		if (pgstrom_bulk_exec_supported(outer_ps))
		{
			((GpuTaskState_v2 *) outer_ps)->row_format = true;
			gpas->gts.outer_bulk_exec = true;
		}
		outerPlanState(gpas) = outer_ps;
		/* GpuPreAgg don't need re-initialization of projection info */
		outer_tupdesc = outer_ps->ps_ResultTupleSlot->tts_tupleDescriptor;
    }
    else
    {
		Assert(scan_rel != NULL);
		gpas->outer_quals = (List *)
			ExecInitExpr((Expr *)gpa_info->outer_quals, &gpas->gts.css.ss.ps);
		outer_tupdesc = RelationGetDescr(scan_rel);
	}

	/*
	 * Initialization the stuff for CPU fallback.
	 *
	 * Projection from the outer-relation to the custom_scan_tlist is a job
	 * of CPU fallback. It is equivalent to the initial device projection.
	 */
	pseudo_tlist = (List *)
		ExecInitExpr((Expr *)cscan->custom_scan_tlist, &gpas->gts.css.ss.ps);
	if (!ExecContextForcesOids(&gpas->gts.css.ss.ps, &has_oid))
		has_oid = false;
	pseudo_tupdesc = ExecTypeFromTL(cscan->custom_scan_tlist, has_oid);
	gpas->pseudo_slot = MakeSingleTupleTableSlot(pseudo_tupdesc);
	gpas->outer_proj = ExecBuildProjectionInfo(pseudo_tlist,
											   econtext,
											   gpas->pseudo_slot,
											   outer_tupdesc);
	gpas->outer_pds = NULL;

	/* Create a shared state object */
	gpas->gpa_sstate = create_gpupreagg_shared_state(gpas, gpa_info,
													 pseudo_tupdesc);

	/* Get CUDA program and async build if any */
	kern_define = pgstrom_build_session_info(gpa_info->extra_flags,
											 &gpas->gts);
	program_id = pgstrom_create_cuda_program(gcontext,
											 gpa_info->extra_flags,
											 gpa_info->kern_source,
											 kern_define,
											 with_connection);
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
	GpuPreAggState	   *gpas = (GpuPreAggState *) node;

	/* release the shared status */
	put_gpupreagg_shared_state(gpas->gpa_sstate);
	/* clean up subtree, if any */
	if (outerPlanState(node))
		ExecEndNode(outerPlanState(node));
	/* release any other resources */
	pgstromReleaseGpuTaskState(&gpas->gts);
}

/*
 * ExecReScanGpuPreAgg
 */
static void
ExecReScanGpuPreAgg(CustomScanState *node)
{
	GpuPreAggState	   *gpas = (GpuPreAggState *) node;

	/* common rescan handling */
	pgstromRescanGpuTaskState(&gpas->gts);
	/* rewind the position to read */
	gpuscanRewindScanChunk(&gpas->gts);
}

static void
ExplainGpuPreAgg(CustomScanState *node, List *ancestors, ExplainState *es)
{
	GpuPreAggState		   *gpas = (GpuPreAggState *) node;
	CustomScan			   *cscan = (CustomScan *) node->ss.ps.plan;
	GpuPreAggSharedState   *gpa_sstate = gpas->gpa_sstate;
	GpuPreAggInfo		   *gpa_info = deform_gpupreagg_info(cscan);
	List				   *dcontext;
	List				   *gpu_proj = NIL;
	ListCell			   *lc;
	const char			   *policy;
	char				   *exprstr;
	cl_uint					n_tasks;

	SpinLockAcquire(&gpa_sstate->lock);
	n_tasks = (gpa_sstate->n_tasks_nogrp +
			   gpa_sstate->n_tasks_local +
			   gpa_sstate->n_tasks_global +
			   gpa_sstate->n_tasks_final);
	if (n_tasks == 0)
	{
		cl_uint		local_threshold = devBaselineMaxThreadsPerBlock / 4;
		cl_uint		global_threshold = gpa_sstate->plan_nrows_per_chunk / 4;

		if (gpas->num_group_keys == 0)
			policy = "NoGroup";
		else if (gpa_sstate->plan_ngroups < local_threshold)
			policy = "Local";
		else if (gpa_sstate->plan_ngroups < global_threshold)
			policy = "Global";
		else
			policy = "Final";
	}
	else
	{
		bool	with_percentage = false;
		char	temp[2048];
		int		ofs = 0;

		if ((gpa_sstate->n_tasks_nogrp > 0 ? 1 : 0) +
			(gpa_sstate->n_tasks_local > 0 ? 1 : 0) +
			(gpa_sstate->n_tasks_global > 0 ? 1 : 0) +
			(gpa_sstate->n_tasks_final > 0 ? 1 : 0) > 1)
			with_percentage = true;

		if (gpa_sstate->n_tasks_nogrp > 0)
		{
			ofs += snprintf(temp + ofs, sizeof(temp) - ofs,
							"%sNoGroup", ofs > 0 ? ", " : "");
			if (with_percentage)
				ofs += snprintf(temp + ofs, sizeof(temp) - ofs, " (%.1f%%)",
								(double)(100 * gpa_sstate->n_tasks_nogrp) /
								(double)(n_tasks));
		}
		if (gpa_sstate->n_tasks_local > 0)
		{
			ofs += snprintf(temp + ofs, sizeof(temp) - ofs,
							"%sLocal", ofs > 0 ? ", " : "");
			if (with_percentage)
				ofs += snprintf(temp + ofs, sizeof(temp) - ofs, " (%.1f%%)",
								(double)(100 * gpa_sstate->n_tasks_local) /
								(double)(n_tasks));
		}
		if (gpa_sstate->n_tasks_global > 0)
		{
			ofs += snprintf(temp + ofs, sizeof(temp) - ofs,
							"%sGlobal", ofs > 0 ? ", " : "");
			if (with_percentage)
				ofs += snprintf(temp + ofs, sizeof(temp) - ofs, " (%.1f%%)",
								(double)(100 * gpa_sstate->n_tasks_global) /
								(double)(n_tasks));
		}
		if (gpa_sstate->n_tasks_final > 0)
		{
			ofs += snprintf(temp + ofs, sizeof(temp) - ofs,
							"%sFinal", ofs > 0 ? ", " : "");
			if (with_percentage)
				ofs += snprintf(temp + ofs, sizeof(temp) - ofs, " (%.1f%%)",
								(double)(100 * gpa_sstate->n_tasks_global) /
								(double)(n_tasks));
		}
		policy = temp;
	}
	SpinLockRelease(&gpa_sstate->lock);
	ExplainPropertyText("Reduction", policy, es);

	/* Set up deparsing context */
	dcontext = set_deparse_context_planstate(es->deparse_cxt,
                                            (Node *)&gpas->gts.css.ss.ps,
                                            ancestors);
	/* Show device projection */
	foreach (lc, cscan->custom_scan_tlist)
		gpu_proj = lappend(gpu_proj, ((TargetEntry *) lfirst(lc))->expr);
	if (gpu_proj != NIL)
	{
		exprstr = deparse_expression((Node *)gpu_proj, dcontext,
									 es->verbose, false);
		ExplainPropertyText("GPU Projection", exprstr, es);
	}

	/* statistics for outer scan, if it was pulled-up */
	//pgstrom_explain_outer_bulkexec(&gpas->gts, context, ancestors, es);

	/* outer scan filter if any */
	if (gpa_info->outer_quals != NIL)
	{
		Node	   *outer_quals
			= (Node *) make_ands_explicit(gpa_info->outer_quals);
		exprstr = deparse_expression(outer_quals, dcontext,
									 es->verbose, false);
		ExplainPropertyText("GPU Filter", exprstr, es);
	}
	/* other common fields */
	pgstromExplainGpuTaskState(&gpas->gts, es);
}

/*
 * create_gpupreagg_shared_state
 */
static GpuPreAggSharedState *
create_gpupreagg_shared_state(GpuPreAggState *gpas, GpuPreAggInfo *gpa_info,
							  TupleDesc pseudo_tupdesc)
{
	GpuContext_v2  *gcontext = gpas->gts.gcontext;
	GpuPreAggSharedState   *gpa_sstate;

	Assert(pseudo_tupdesc->natts > 0);
	gpa_sstate = dmaBufferAlloc(gcontext, sizeof(GpuPreAggSharedState));
	memset(gpa_sstate, 0, sizeof(GpuPreAggSharedState));
	pg_atomic_init_u32(&gpa_sstate->refcnt, 1);
	SpinLockInit(&gpa_sstate->lock);
	gpa_sstate->pds_final = NULL;		/* creation on demand */
	gpa_sstate->m_fhash = 0UL;			/* creation on demand */
	gpa_sstate->m_kds_final = 0UL;		/* creation on demand */
	gpa_sstate->ev_kds_final = NULL;	/* creation on demand */
	gpa_sstate->f_ncols = pseudo_tupdesc->natts;
	gpa_sstate->f_key_dist_salt = 1;	/* assign on demand */
	gpa_sstate->f_nrooms = 0;			/* assign on demand */
	gpa_sstate->f_nitems = 0;			/* runtime statistics */
	gpa_sstate->f_extra_sz = 0;			/* runtime statistics */
	gpa_sstate->plan_nrows_per_chunk =
		(gpa_info->plan_nchunks > 0
		 ? gpa_sstate->plan_nrows_in / (double)gpa_info->plan_nchunks
		 : gpa_sstate->plan_nrows_in);
	gpa_sstate->plan_nrows_in = gpa_info->outer_nrows;
	gpa_sstate->plan_ngroups = gpa_info->plan_ngroups;
	gpa_sstate->plan_extra_sz = gpa_info->plan_extra_sz;

	return gpa_sstate;
}

/*
 * get_gpupreagg_shared_state
 */
static GpuPreAggSharedState *
get_gpupreagg_shared_state(GpuPreAggSharedState *gpa_sstate)
{
	int32		refcnt_old	__attribute__((unused));

	refcnt_old = (int32)pg_atomic_fetch_add_u32(&gpa_sstate->refcnt, 1);
	Assert(refcnt_old > 0);

	return gpa_sstate;
}

/*
 * put_gpupreagg_shared_state
 */
static void
put_gpupreagg_shared_state(GpuPreAggSharedState *gpa_sstate)
{
	int32		refcnt_new;

	refcnt_new = (int32)pg_atomic_sub_fetch_u32(&gpa_sstate->refcnt, 1);
	Assert(refcnt_new >= 0);
	if (refcnt_new == 0)
	{
		Assert(!gpa_sstate->pds_final);
		Assert(gpa_sstate->m_fhash == 0UL);
		Assert(gpa_sstate->m_kds_final);
		dmaBufferFree(gpa_sstate);
	}
}


















/*
 * gpupreagg_create_task - constructor of GpuPreAggTask
 */
static GpuTask_v2 *
gpupreagg_create_task(GpuPreAggState *gpas,
					  pgstrom_data_store *pds_in,
					  int file_desc,
					  bool is_last_task)
{
	GpuContext_v2  *gcontext = gpas->gts.gcontext;
	GpuPreAggTask  *gpreagg;
	TupleDesc		tupdesc;
	bool			with_nvme_strom = false;
	cl_uint			nrows_per_block = 0;
	cl_uint			nitems_real = pds_in->kds.nitems;
	Size			head_sz;
	Size			kds_len;

	/* adjust parameters if block format */
	if (pds_in->kds.format == KDS_FORMAT_BLOCK)
	{
		Assert(gpas->gts.nvme_sstate != NULL);
		with_nvme_strom = (pds_in->nblocks_uncached > 0);
		nrows_per_block = gpas->gts.nvme_sstate->nrows_per_block;
		nitems_real = pds_in->kds.nitems * nrows_per_block;
	}

	/* allocation of GpuPreAggTask */
	tupdesc = gpas->pseudo_slot->tts_tupleDescriptor;
	head_sz = STROMALIGN(offsetof(GpuPreAggTask, kern.kparams) +
						 gpas->gts.kern_params->length);
	kds_len = STROMALIGN(offsetof(kern_data_store,
								  colmeta[tupdesc->natts]));
	gpreagg = dmaBufferAlloc(gcontext, head_sz + kds_len);
	memset(gpreagg, 0, offsetof(GpuPreAggTask, kern.kparams));

	pgstromInitGpuTask(&gpas->gts, &gpreagg->task);
	gpreagg->gpa_sstate = get_gpupreagg_shared_state(gpas->gpa_sstate);
	gpreagg->with_nvme_strom = with_nvme_strom;
	gpreagg->is_last_task = is_last_task;
	gpreagg->is_retry = false;
	gpreagg->pds_in = pds_in;
	gpreagg->kds_head = (kern_data_store *)((char *)gpreagg + head_sz);
	gpreagg->pds_final = NULL;	/* to be attached later */

	/* if any grouping keys, determine the reduction policy later */
	gpreagg->kern.reduction_mode = (gpas->num_group_keys == 0
									? GPUPREAGG_NOGROUP_REDUCTION
									: GPUPREAGG_INVALID_REDUCTION);
	gpreagg->kern.nitems_real = nitems_real;
//	gpreagg->kern.key_dist_salt = gpas->key_dist_salt;
	gpreagg->kern.hash_size = nitems_real;
	memcpy(gpreagg->kern.pg_crc32_table,
		   pg_crc32_table,
		   sizeof(uint32) * 256);
	/* kern_parambuf */
	memcpy(KERN_GPUPREAGG_PARAMBUF(&gpreagg->kern),
		   gpas->gts.kern_params,
		   gpas->gts.kern_params->length);
	/* offset of kern_resultbuf-1 */
	gpreagg->kern.kresults_1_offset
		= STROMALIGN(offsetof(kern_gpupreagg, kparams) +
					 gpas->gts.kern_params->length);
	/* offset of kern_resultbuf-2 */
	gpreagg->kern.kresults_2_offset
		= STROMALIGN(gpreagg->kern.kresults_1_offset +
					 offsetof(kern_resultbuf, results[nitems_real]));

	/* kds_head for the working global buffer */
	kds_len += STROMALIGN(LONGALIGN((sizeof(Datum) + sizeof(char)) *
									tupdesc->natts) * nitems_real);
	init_kernel_data_store(gpreagg->kds_head,
						   tupdesc,
						   kds_len,
						   KDS_FORMAT_SLOT,
						   nitems_real,
						   true);
	return &gpreagg->task;
}

/*
 * gpupreagg_next_task
 *
 * callback to construct a new GpuPreAggTask task object based on
 * the input data stream that is scanned.
 */
static GpuTask_v2 *
gpupreagg_next_task(GpuTaskState_v2 *gts)
{
	GpuPreAggState	   *gpas = (GpuPreAggState *) gts;
	pgstrom_data_store *pds = NULL;
	int					filedesc = -1;
	bool				is_last_task = false;
	struct timeval		tv1, tv2;

	PFMON_BEGIN(&gts->pfm, &tv1);
	if (gpas->gts.css.ss.ss_currentRelation)
	{
		if (!gpas->outer_pds)
			gpas->outer_pds = gpuscanExecScanChunk(&gpas->gts, &filedesc);
		pds = gpas->outer_pds;
		if (pds)
			gpas->outer_pds = gpuscanExecScanChunk(&gpas->gts, &filedesc);
		else
			gpas->outer_pds = NULL;
		/* any more chunks expected? */
		if (!gpas->outer_pds)
			is_last_task = true;
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
				slot = gpas->gts.scan_overflow;
				gpas->gts.scan_overflow = NULL;
			}
			else
			{
				slot = ExecProcNode(outer_ps);
				if (TupIsNull(slot))
				{
					gpas->gts.scan_done = true;
					break;
				}

				/* create a new data-store on demand */
				if (!pds)
				{
					pds = PDS_create_row(gpas->gts.gcontext,
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
		if (!gpas->gts.scan_overflow)
			is_last_task = true;
	}
	PFMON_END(&gpas->gts.pfm, time_outer_load, &tv1, &tv2);

	return gpupreagg_create_task(gpas, pds, filedesc, is_last_task);
}


static void
gpupreagg_ready_task(GpuTaskState_v2 *gts, GpuTask_v2 *gtask)
{
	// needs a feature to drop task?
	// or, complete returns -1 to discard the task
	
}

static void
gpupreagg_switch_task(GpuTaskState_v2 *gts, GpuTask_v2 *gtask)
{}

/*
 * gpupreagg_next_tuple_fallback
 */
static TupleTableSlot *
gpupreagg_next_tuple_fallback(GpuPreAggState *gpas, GpuPreAggTask *gpreagg)
{
	TupleTableSlot	   *slot = gpas->pseudo_slot;
	ExprContext		   *econtext = gpas->gts.css.ss.ps.ps_ExprContext;
	pgstrom_data_store *pds_in = gpreagg->pds_in;

retry:
	/* fetch a tuple from the data-store */
	ExecClearTuple(slot);
	if (!PDS_fetch_tuple(slot, pds_in, &gpas->gts))
		return NULL;

	/* filter out the tuple, if any outer quals */
	if (gpas->outer_quals != NIL)
	{
		econtext->ecxt_scantuple = slot;
		if (!ExecQual(gpas->outer_quals, econtext, false))
			goto retry;
	}

	/* ok, makes projection from outer-scan to pseudo-tlist */
	if (gpas->outer_proj)
	{
		ExprDoneCond	is_done;

		slot = ExecProject(gpas->outer_proj, &is_done);
		if (is_done == ExprEndResult)
			goto retry;	/* really right? */
	}
	return slot;
}

/*
 * gpupreagg_next_tuple
 */
static TupleTableSlot *
gpupreagg_next_tuple(GpuTaskState_v2 *gts)
{
	GpuPreAggState	   *gpas = (GpuPreAggState *) gts;
	GpuPreAggTask	   *gpreagg = (GpuPreAggTask *) gpas->gts.curr_task;
	pgstrom_data_store *pds_final = gpreagg->pds_final;
	TupleTableSlot	   *slot = NULL;
	struct timeval		tv1, tv2;

	PFMON_BEGIN(&gts->pfm, &tv1);
	if (gpreagg->task.cpu_fallback)
		slot = gpupreagg_next_tuple_fallback(gpas, gpreagg);
	else if (gpas->gts.curr_index < pds_final->kds.nitems)
	{
		slot = gpas->pseudo_slot;
		ExecClearTuple(slot);
		PDS_fetch_tuple(slot, pds_final, &gpas->gts);
	}
	PFMON_END(&gts->pfm, time_materialize, &tv1, &tv2);

	return slot;
}

/*
 * adjust_final_buffer_size
 *
 * It calculates @nrooms/@extra_sz of the pds_final buffer to be allocated,
 * according to the run-time statistics or plan estimation if no statistics.
 *
 * NOTE: This function shall be called under the @gpa_sstate->lock
 */
static void
adjust_final_buffer_size(GpuPreAggSharedState *gpa_sstate,
						 size_t *p_key_dist_salt,
						 size_t *p_nrooms,
						 size_t *p_extra_sz,
						 size_t *p_hashsize)
{
	size_t		curr_ngroups;
	size_t		curr_nrows_in;
	size_t		head_sz;
	size_t		unit_sz;
	size_t		length;
	size_t		f_key_dist_salt;
	size_t		f_nrooms;
	size_t		f_extra_sz;
	double		alpha;

	/*
	 * If we have no run-time statistics, all we can do is relying on
	 * the plan time estimation.
	 * Elsewhere, we assume number of groups grows up according to:
	 *   (ngroups) = A * ln(nrows_in)
	 * We can determine "A" by the statistics, 
	 */
	if (gpa_sstate->exec_nrows_in < 1000)
		curr_ngroups = gpa_sstate->plan_ngroups;
	else
	{
		alpha = ((double)gpa_sstate->exec_ngroups
		   / log((double)gpa_sstate->exec_nrows_in));

		if (gpa_sstate->exec_nrows_in < gpa_sstate->plan_nrows_in / 2)
			curr_nrows_in = gpa_sstate->plan_nrows_in;
		else
			curr_nrows_in = 2 * gpa_sstate->exec_nrows_in;

		curr_ngroups = (size_t)(alpha * log((double)curr_nrows_in));
	}

	/* determine the unit size of extra buffer */
	if (gpa_sstate->exec_ngroups < 100)
		f_extra_sz = gpa_sstate->plan_extra_sz;
	else
	{
		f_extra_sz = (gpa_sstate->exec_extra_sz +
					  gpa_sstate->exec_ngroups - 1) / gpa_sstate->exec_ngroups;
		f_extra_sz = Max(f_extra_sz, gpa_sstate->plan_extra_sz);
	}

	/* update key_dist_salt */
	if (curr_ngroups < (devBaselineMaxThreadsPerBlock / 5))
	{
		f_key_dist_salt = devBaselineMaxThreadsPerBlock / (5 * curr_ngroups);
		f_key_dist_salt = Max(f_key_dist_salt, 1);
	}
	else
		f_key_dist_salt = 1;

	/* f_nrooms will have 250% of the nrooms for the estimated ngroups */
	f_nrooms = (double)(curr_ngroups * f_key_dist_salt) * 2.5 + 200.0;
	head_sz = KDS_CALCULATE_HEAD_LENGTH(gpa_sstate->f_ncols);
	unit_sz = (STROMALIGN((sizeof(Datum) +
						   sizeof(char)) * gpa_sstate->f_ncols) +
			   STROMALIGN(f_extra_sz));
	length = head_sz + unit_sz * f_nrooms;

	/*
	 * Expand nrooms if estimated length of the kds_final is small,
	 * because planner may estimate the number groups smaller than actual.
	 */
	if (length < pgstrom_chunk_size() / 2)
		f_nrooms = (pgstrom_chunk_size() - head_sz) / unit_sz;
	else if (length < pgstrom_chunk_size())
		f_nrooms = (2 * pgstrom_chunk_size() - head_sz) / unit_sz;
	else if (length < 3 * pgstrom_chunk_size())
		f_nrooms = (3 * pgstrom_chunk_size() - head_sz) / unit_sz;

	*p_key_dist_salt = f_key_dist_salt;
	*p_nrooms = f_nrooms;
	*p_extra_sz = f_extra_sz;
	*p_hashsize = 2 * f_nrooms;
}

/*
 * gpupreagg_alloc_final_buffer
 *
 * It allocates the @pds_final buffer on demand.
 *
 * NOTE: This function shall be called under the @gpa_sstate->lock
 */
static bool
gpupreagg_alloc_final_buffer(GpuPreAggTask *gpreagg,
							 CUmodule cuda_module,
							 CUstream cuda_stream)
{
	GpuPreAggSharedState   *gpa_sstate = gpreagg->gpa_sstate;
	kern_data_store		   *kds_head = gpreagg->kds_head;
	pgstrom_data_store	   *pds_final = NULL;
	CUdeviceptr				m_kds_final = 0UL;
	CUdeviceptr				m_fhash = 0UL;
	CUfunction				kern_final_prep = NULL;
	CUevent					ev_kds_final = NULL;
	void				   *kern_args[2];
	bool					retval = true;
	CUresult				rc;

	PG_TRY();
	{
		size_t		f_key_dist_salt;
		size_t		f_nrooms;
		size_t		f_extra_sz;
		size_t		f_hashsize;
		size_t		required;
		size_t		grid_size;
		size_t		block_size;

		adjust_final_buffer_size(gpa_sstate,
								 &f_key_dist_salt,
								 &f_nrooms,
								 &f_extra_sz,
								 &f_hashsize);
		pds_final = PDS_duplicate_slot(gpreagg->task.gcontext,
									   kds_head,
									   f_nrooms,
									   f_extra_sz);
		/* allocation of device memory */
		required = (GPUMEMALIGN(pds_final->kds.length) +
					GPUMEMALIGN(offsetof(kern_global_hashslot,
										 hash_slot[f_hashsize])));
		rc = gpuMemAlloc_v2(gpreagg->task.gcontext,
							&m_kds_final,
							required);
		if (rc == CUDA_ERROR_OUT_OF_MEMORY)
		{
			/* cleanup pds_final, and quick bailout */
			PDS_release(pds_final);
			retval = false;
			goto bailout;
		}
		else if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on gpuMemAlloc: %s", errorText(rc));
		m_fhash = m_kds_final + GPUMEMALIGN(pds_final->kds.length);

		/* creation of event object to synchronize kds_final load */
		rc = cuEventCreate(&ev_kds_final, CU_EVENT_DISABLE_TIMING);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

		/* DMA send of kds_final head */
		rc = cuMemcpyHtoDAsync(m_kds_final, &pds_final->kds,
							   KERN_DATA_STORE_HEAD_LENGTH(&pds_final->kds),
							   gpreagg->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));

		/* Launch:
		 * KERNEL_FUNCTION(void)
		 * gpupreagg_final_preparation(size_t hash_size,
		 *                             kern_global_hashslot *f_hashslot)
		 */
		rc = cuModuleGetFunction(&kern_final_prep,
								 cuda_module,
								 "gpupreagg_final_preparation");
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

		optimal_workgroup_size(&grid_size,
							   &block_size,
							   kern_final_prep,
							   gpuserv_cuda_device,
							   f_hashsize,
							   0, sizeof(kern_errorbuf));
		kern_args[0] = &f_hashsize;
		kern_args[1] = &m_fhash;
		rc = cuLaunchKernel(kern_final_prep,
							grid_size, 1, 1,
							block_size, 1, 1,
							sizeof(kern_errorbuf) * block_size,
							cuda_stream,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));

		/* Synchronization for setup of pds_final buffer */
		rc = cuEventRecord(ev_kds_final, cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuStreamWaitEvent: %s", errorText(rc));

		gpa_sstate->pds_final = pds_final;
		gpa_sstate->m_kds_final = m_kds_final;
		gpa_sstate->m_fhash = m_fhash;
		gpa_sstate->ev_kds_final = ev_kds_final;
		gpa_sstate->f_key_dist_salt = f_key_dist_salt;
		gpa_sstate->f_nrooms = pds_final->kds.nrooms;
		gpa_sstate->f_nitems = 0;
		gpa_sstate->f_extra_sz = 0;
	bailout:
		;
	}
	PG_CATCH();
	{
		rc = cuStreamSynchronize(cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(FATAL, "failed on cuStreamSynchronize: %s", errorText(rc));

		if (ev_kds_final != NULL)
		{
			rc = cuEventDestroy(ev_kds_final);
			if (rc != CUDA_SUCCESS)
				elog(FATAL, "failed on cuEventDestroy: %s", errorText(rc));
		}

		if (m_kds_final != 0UL)
		{
			rc = gpuMemFree_v2(gpreagg->task.gcontext, m_kds_final);
			if (rc != CUDA_SUCCESS)
				elog(FATAL, "failed on gpuMemFree: %s", errorText(rc));
		}

		if (pds_final != NULL)
			PDS_release(pds_final);
		PG_RE_THROW();
	}
	PG_END_TRY();

	return retval;
}

/*
 * gpupreagg_get_strategy
 *
 * It determines the strategy to run GpuPreAgg kernel according to the run-
 * time statistics.
 * Number of groups is the most important decision. If estimated number of
 * group is larger than the maximum block size, local reduction makes no
 * sense. If too small, final reduction without local/global reduction will
 * lead massive amount of atomic contention.
 * In addition, this function switches the @pds_final buffer if remaining
 * space is not sufficient to hold the groups appear.
 *
 * NOTE: This function shall be called under the @gpa_sstate->lock
 */
static bool
gpupreagg_get_strategy(GpuPreAggTask *gpreagg,
					   CUmodule cuda_module,
					   CUstream cuda_stream)
{
	GpuPreAggSharedState   *gpa_sstate = gpreagg->gpa_sstate;
	pgstrom_data_store	   *pds_in = gpreagg->pds_in;
	bool					retval = true;
	CUresult				rc;

	Assert(pds_in->kds.format == KDS_FORMAT_ROW ||
		   pds_in->kds.format == KDS_FORMAT_BLOCK);
	SpinLockAcquire(&gpa_sstate->lock);
	PG_TRY();
	{
		/* decision for the reduction mode */
		if (gpreagg->kern.reduction_mode == GPUPREAGG_INVALID_REDUCTION)
		{
			cl_double	plan_ngroups = (double)gpa_sstate->plan_ngroups;
			cl_double	exec_ngroups = (double)gpa_sstate->exec_ngroups;
			cl_double	real_ngroups;
			cl_double	exec_ratio;
			cl_double	num_tasks;

			num_tasks = (cl_double)(gpa_sstate->n_tasks_nogrp +
									gpa_sstate->n_tasks_local +
									gpa_sstate->n_tasks_global +
									gpa_sstate->n_tasks_final);
			exec_ratio = Min(num_tasks, 30.0) / 30.0;
			real_ngroups = (plan_ngroups * (1.0 - exec_ratio) +
							exec_ngroups * exec_ratio);
			if (real_ngroups < devBaselineMaxThreadsPerBlock / 4)
				gpreagg->kern.reduction_mode = GPUPREAGG_LOCAL_REDUCTION;
			else if (real_ngroups < gpreagg->kern.nitems_real / 4)
				gpreagg->kern.reduction_mode = GPUPREAGG_GLOBAL_REDUCTION;
			else
				gpreagg->kern.reduction_mode = GPUPREAGG_FINAL_REDUCTION;
		}
		else
		{
			Assert(gpreagg->kern.reduction_mode==GPUPREAGG_NOGROUP_REDUCTION);
		}
		/* attach pds_final and relevant CUDA resources */
		if (!gpa_sstate->pds_final)
		{
			retval = gpupreagg_alloc_final_buffer(gpreagg,
												  cuda_module,
												  cuda_stream);
		}
		else
		{
			rc = cuStreamWaitEvent(cuda_stream, gpa_sstate->ev_kds_final, 0);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuStreamWaitEvent: %s", errorText(rc));
		}
		gpreagg->pds_final    = PDS_retain(gpa_sstate->pds_final);
		gpreagg->pds_final->ntasks_running++;
		gpreagg->m_fhash      = gpa_sstate->m_fhash;
		gpreagg->m_kds_final  = gpa_sstate->m_kds_final;
		gpreagg->ev_kds_final = gpa_sstate->ev_kds_final;
		gpreagg->kern.key_dist_salt = gpa_sstate->f_key_dist_salt;
	}
	PG_CATCH();
	{
		SpinLockRelease(&gpa_sstate->lock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	SpinLockRelease(&gpa_sstate->lock);

	return retval;
}

/*
 * gpupreagg_put_strategy
 */
static bool
gpupreagg_put_strategy(GpuPreAggTask *gpreagg)
{
	GpuPreAggSharedState   *gpa_sstate = gpreagg->gpa_sstate;
	pgstrom_data_store	   *pds_final = gpreagg->pds_final;
	bool		retval = false;

	SpinLockAcquire(&gpa_sstate->lock);
	if (--pds_final->ntasks_running == 0 && pds_final->is_dereferenced)
		retval = true;	/* task is responsible to terminate the pds_final */
	SpinLockRelease(&gpa_sstate->lock);

	if (!retval)
	{
		PDS_release(pds_final);
		gpreagg->ev_kds_final = NULL;
		gpreagg->m_kds_final = 0UL;
		gpreagg->m_fhash = 0UL;
	}
	return retval;
}

/*
 * gpupreagg_cleanup_cuda_resources
 */
static void
gpupreagg_cleanup_cuda_resources(GpuPreAggTask *gpreagg, bool cleanup_any)
{
	CUresult	rc;

	PFMON_EVENT_DESTROY(gpreagg, ev_dma_send_start);
	PFMON_EVENT_DESTROY(gpreagg, ev_dma_send_stop);
	PFMON_EVENT_DESTROY(gpreagg, ev_dma_recv_start);
	PFMON_EVENT_DESTROY(gpreagg, ev_dma_recv_stop);

	rc = gpuMemFree_v2(gpreagg->task.gcontext, gpreagg->m_kds_in);
	if (rc != CUDA_SUCCESS)
		elog(FATAL, "failed on gpuMemFree: %s", errorText(rc));

	if (gpreagg->with_nvme_strom &&
		gpreagg->m_kds_in != 0UL)
	{
		rc = gpuMemFreeIOMap(gpreagg->task.gcontext, gpreagg->m_kds_in);
		if (rc != CUDA_SUCCESS)
			elog(FATAL, "failed on gpuMemFreeIOMap: %s", errorText(rc));
	}
	/* ensure pointers are NULL */
	gpreagg->m_gpreagg = 0UL;
	gpreagg->m_kds_in = 0UL;
	gpreagg->m_kds_slot = 0UL;
	gpreagg->m_ghash = 0UL;
	if (cleanup_any)
	{
		PFMON_EVENT_DESTROY(gpreagg, ev_kds_final);
		rc = gpuMemFree_v2(gpreagg->task.gcontext, gpreagg->m_kds_final);
		if (rc != CUDA_SUCCESS)
			elog(FATAL, "failed on gpuMemFree: %s", errorText(rc));
	}
	gpreagg->ev_kds_final = NULL;
	gpreagg->m_kds_final = 0UL;
	gpreagg->m_fhash = 0UL;
}

/*
 * gpupreagg_respond_task - callback handler on CUDA context
 */
static void
gpupreagg_respond_task(CUstream stream, CUresult status, void *private)
{
	GpuPreAggTask  *gpreagg = private;
	bool			is_urgent = false;

	if (status == CUDA_SUCCESS)
	{
		gpreagg->task.kerror = gpreagg->kern.kerror;
		if (gpreagg->task.kerror.errcode == StromError_Success)
		{
			GpuPreAggSharedState *gpa_sstate = gpreagg->gpa_sstate;

			SpinLockAcquire(&gpa_sstate->lock);
			gpa_sstate->f_nitems += gpreagg->kern.num_groups;
			gpa_sstate->f_extra_sz += gpreagg->kern.varlena_usage;
			gpa_sstate->exec_nrows_in += gpreagg->kern.nitems_real;
			gpa_sstate->exec_ngroups = Max(gpa_sstate->exec_ngroups,
										   gpa_sstate->f_nitems);
			gpa_sstate->exec_extra_sz = Max(gpa_sstate->exec_extra_sz,
											gpa_sstate->f_extra_sz);
			SpinLockRelease(&gpa_sstate->lock);
		}
		else
			is_urgent = true;	/* something error */
	}
	else
	{
		/*
		 * CUDA Run-time error - not recoverable
		 */
		gpreagg->task.kerror.errcode = status;
		gpreagg->task.kerror.kernel = StromKernel_CudaRuntime;
		gpreagg->task.kerror.lineno = 0;
		is_urgent = true;
	}
	gpuservCompleteGpuTask(&gpreagg->task, is_urgent);
}

/*
 * gpupreagg_process_reduction_task
 *
 * main logic to kick GpuPreAgg kernel function.
 */
static int
gpupreagg_process_reduction_task(GpuPreAggTask *gpreagg,
								 CUmodule cuda_module,
								 CUstream cuda_stream)
{
	GpuPreAggSharedState *gpa_sstate = gpreagg->gpa_sstate;
	pgstrom_data_store *pds_in = gpreagg->pds_in;
	CUfunction	kern_main;
	CUdeviceptr	devptr;
	Size		length;
	void	   *kern_args[6];
	CUresult	rc;

	/*
	 * Get GpuPreAgg execution strategy
	 */
	if (!gpupreagg_get_strategy(gpreagg, cuda_module, cuda_stream))
		return 1;	/* retry later */

	/*
	 * Lookup kernel functions
	 */
	rc = cuModuleGetFunction(&kern_main,
							 cuda_module,
							 "gpupreagg_main");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

	/*
	 * Allocation of own device memory
	 */
	length = (GPUMEMALIGN(KERN_GPUPREAGG_LENGTH(&gpreagg->kern)) +
			  GPUMEMALIGN(gpreagg->kds_head->length) +
			  GPUMEMALIGN(offsetof(kern_global_hashslot,
								   hash_slot[gpreagg->kern.hash_size])));
	if (gpreagg->with_nvme_strom)
	{
		rc = gpuMemAllocIOMap(gpreagg->task.gcontext,
							  &gpreagg->m_kds_in,
							  GPUMEMALIGN(pds_in->kds.length));
		if (rc == CUDA_ERROR_OUT_OF_MEMORY)
		{
			PDS_fillup_blocks(pds_in, gpreagg->task.peer_fdesc);
			gpreagg->m_kds_in = 0UL;
			gpreagg->with_nvme_strom = false;
			length += GPUMEMALIGN(pds_in->kds.length);
		}
		else if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on gpuMemAllocIOMap: %s", errorText(rc));
	}
	else
		length += GPUMEMALIGN(pds_in->kds.length);

	rc = gpuMemAlloc_v2(gpreagg->task.gcontext, &devptr, length);
	if (rc == CUDA_ERROR_OUT_OF_MEMORY)
		goto out_of_resource;
	else if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on gpuMemAlloc: %s", errorText(rc));

	gpreagg->m_gpreagg = devptr;
	devptr += GPUMEMALIGN(KERN_GPUPREAGG_LENGTH(&gpreagg->kern));
	if (gpreagg->with_nvme_strom)
		Assert(gpreagg->m_kds_in != 0UL);
	else
	{
		gpreagg->m_kds_in = devptr;
		devptr += GPUMEMALIGN(pds_in->kds.length);
	}
	gpreagg->m_kds_slot = devptr;
	devptr += GPUMEMALIGN(gpreagg->kds_head->length);
	gpreagg->m_ghash = devptr;
	devptr += GPUMEMALIGN(offsetof(kern_global_hashslot,
								   hash_slot[gpreagg->kern.hash_size]));
	Assert(devptr == gpreagg->m_gpreagg + length);
	Assert(gpreagg->m_kds_final != 0UL && gpreagg->m_fhash != 0UL);

	/*
	 * Creation of event objects, if any
	 */
	PFMON_EVENT_CREATE(gpreagg, ev_dma_send_start);
	PFMON_EVENT_CREATE(gpreagg, ev_dma_send_stop);
	PFMON_EVENT_CREATE(gpreagg, ev_dma_recv_start);
	PFMON_EVENT_CREATE(gpreagg, ev_dma_recv_stop);

	/*
	 * Count number of reduction kernel for each
	 */
	SpinLockAcquire(&gpa_sstate->lock);
	if (gpreagg->kern.reduction_mode == GPUPREAGG_NOGROUP_REDUCTION)
		gpa_sstate->n_tasks_nogrp++;
	else if (gpreagg->kern.reduction_mode == GPUPREAGG_LOCAL_REDUCTION)
		gpa_sstate->n_tasks_local++;
	else if (gpreagg->kern.reduction_mode == GPUPREAGG_GLOBAL_REDUCTION)
		gpa_sstate->n_tasks_global++;
	else if (gpreagg->kern.reduction_mode == GPUPREAGG_FINAL_REDUCTION)
		gpa_sstate->n_tasks_final++;
	else
	{
		SpinLockRelease(&gpa_sstate->lock);
		elog(ERROR, "Bug? unexpected reduction mode: %d",
			 gpreagg->kern.reduction_mode);
	}
	SpinLockRelease(&gpa_sstate->lock);

	/*
	 * OK, kick gpupreagg_main kernel function
	 */
	PFMON_EVENT_RECORD(gpreagg, ev_dma_send_start, cuda_stream);

	/* kern_gpupreagg */
	length = KERN_GPUPREAGG_DMASEND_LENGTH(&gpreagg->kern);
	rc = cuMemcpyHtoDAsync(gpreagg->m_gpreagg,
						   &gpreagg->kern,
						   length,
						   cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	gpreagg->bytes_dma_send += length;
	gpreagg->num_dma_send++;

	/* source data to be reduced */
	if (!gpreagg->with_nvme_strom)
	{
		length = pds_in->kds.length;
		rc = cuMemcpyHtoDAsync(gpreagg->m_kds_in,
							   &pds_in->kds,
							   length,
							   cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		gpreagg->bytes_dma_send += length;
		gpreagg->num_dma_send++;
	}
	else
	{
		Assert(pds_in->kds.format == KDS_FORMAT_BLOCK);
		gpuMemCopyFromSSDAsync(&gpreagg->task,
							   gpreagg->m_kds_in,
							   pds_in,
							   cuda_stream);
		gpuMemCopyFromSSDWait(&gpreagg->task,
							  cuda_stream);
	}

	/* header of the internal kds-slot buffer */
	length = KERN_DATA_STORE_HEAD_LENGTH(gpreagg->kds_head);
	rc = cuMemcpyHtoDAsync(gpreagg->m_kds_slot,
						   gpreagg->kds_head,
						   length,
						   cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	gpreagg->bytes_dma_send += length;
	gpreagg->num_dma_send++;

	PFMON_EVENT_RECORD(gpreagg, ev_dma_send_stop, cuda_stream);

	/* Launch:
	 * KERNEL_FUNCTION(void)
	 * gpupreagg_main(kern_gpupreagg *kgpreagg,
	 *                kern_data_store *kds_row,
	 *                kern_data_store *kds_slot,
	 *                kern_global_hashslot *g_hash,
	 *                kern_data_store *kds_final,
	 *                kern_global_hashslot *f_hash)
	 */
	kern_args[0] = &gpreagg->m_gpreagg;
	kern_args[1] = &gpreagg->m_kds_in;
	kern_args[2] = &gpreagg->m_kds_slot;
	kern_args[3] = &gpreagg->m_ghash;
	kern_args[4] = &gpreagg->m_kds_final;
	kern_args[5] = &gpreagg->m_fhash;

	rc = cuLaunchKernel(kern_main,
						1, 1, 1,
						1, 1, 1,
						sizeof(kern_errorbuf),
						gpreagg->task.cuda_stream,
						kern_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
	gpreagg->num_kern_main++;

	/*
	 * DMA Recv of individual kern_gpupreagg
	 *
	 * NOTE: DMA recv of the final buffer is job of the terminator task.
	 */
	PFMON_EVENT_RECORD(gpreagg, ev_dma_recv_start, cuda_stream);

	length = KERN_GPUPREAGG_DMARECV_LENGTH(&gpreagg->kern);
	rc = cuMemcpyDtoHAsync(&gpreagg->kern,
						   gpreagg->m_gpreagg,
						   length,
						   cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyDtoHAsync: %s", errorText(rc));
	gpreagg->bytes_dma_recv += length;
	gpreagg->num_dma_recv++;

	PFMON_EVENT_RECORD(gpreagg, ev_dma_recv_stop, cuda_stream);

	/*
	 * Callback registration
	 */
	rc = cuStreamAddCallback(cuda_stream,
							 gpupreagg_respond_task,
							 gpreagg, 0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuStreamAddCallback: %s", errorText(rc));
	return 0;

out_of_resource:
	/*
	 * If task got OUT_OF_RESOURCE during setup but it is also responsible
	 * to the pds_final, we have to kick another termination task because
	 * this task cannot execute as is.
	 */
	if (gpupreagg_put_strategy(gpreagg))
		gpupreagg_push_terminator_task(gpreagg);
	/*
	 * @is_retry == true means, this task was executed before but raised
	 * DataStoreNoSpace error once. Its own device memory contains data
	 * items both of already merged and not merged yet, thus, we cannot
	 * release device memory at this timing.
	 */
	if (!gpreagg->is_retry)
		gpupreagg_cleanup_cuda_resources(gpreagg, false);
	return 1;	/* retry later */
}

/*
 * gpupreagg_process_termination_task
 */
static int
gpupreagg_process_termination_task(GpuPreAggTask *gpreagg,
								   CUmodule cuda_module,
								   CUstream cuda_stream)
{
	GpuPreAggSharedState *gpa_sstate __attribute__((unused));
	pgstrom_data_store *pds_final = gpreagg->pds_final;
	CUfunction		kern_fixvar;
	CUresult		rc;
	Size			length;

	PFMON_EVENT_CREATE(gpreagg, ev_kern_fixvar);
	PFMON_EVENT_CREATE(gpreagg, ev_dma_recv_start);
	PFMON_EVENT_CREATE(gpreagg, ev_dma_recv_stop);

#ifdef USE_ASSERT_CHECKING
	/* @pds_final shall be already dereferenced and nobody is running on */
	gpa_sstate = gpreagg->gpa_sstate;
	SpinLockAcquire(&gpa_sstate->lock);
	Assert(pds_final->ntasks_running == 0 &&
		   pds_final->is_dereferenced);
	SpinLockRelease(&gpa_sstate->lock);
#endif

	/*
	 * Fixup varlena and numeric variables, if needed.
	 */
	if (pds_final->kds.has_notbyval)
	{
		size_t		grid_size;
		size_t		block_size;
		void	   *kern_args[2];

		/* kernel to fixup varlena/numeric */
		rc = cuModuleGetFunction(&kern_fixvar,
								 cuda_module,
								 "gpupreagg_fixup_varlena");
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuModuleGetFunction: %s", errorText(rc));

		/* allocation of the kern_gpupreagg */
		length = GPUMEMALIGN(KERN_GPUPREAGG_LENGTH(&gpreagg->kern));
		rc = gpuMemAlloc_v2(gpreagg->task.gcontext,
							&gpreagg->m_gpreagg,
							length);
		if (rc == CUDA_ERROR_OUT_OF_MEMORY)
			goto out_of_resource;
		else if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on gpuMemAlloc: %s", errorText(rc));

		/* Launch:
		 * KERNEL_FUNCTION(void)
		 * gpupreagg_fixup_varlena(kern_gpupreagg *kgpreagg,
		 *                         kern_data_store *kds_final)
		 *
		 * TODO: we can reduce # of threads to the latest number of groups
		 *       for more optimization.
		 */
		PFMON_EVENT_RECORD(gpreagg, ev_kern_fixvar, cuda_stream);

		optimal_workgroup_size(&grid_size,
							   &block_size,
							   kern_fixvar,
							   gpuserv_cuda_device,
							   pds_final->kds.nrooms,
							   0, sizeof(kern_errorbuf));
		kern_args[0] = &gpreagg->m_gpreagg;
		kern_args[1] = &gpreagg->m_kds_final;

		rc = cuLaunchKernel(kern_fixvar,
							grid_size, 1, 1,
							block_size, 1, 1,
							sizeof(kern_errorbuf) * block_size,
							cuda_stream,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
		gpreagg->num_kern_fixvar++;

		/*
		 * DMA Recv of individual kern_gpupreagg
		 */
		PFMON_EVENT_RECORD(gpreagg, ev_dma_recv_start, cuda_stream);

		length = KERN_GPUPREAGG_DMARECV_LENGTH(&gpreagg->kern);
		rc = cuMemcpyDtoHAsync(&gpreagg->kern,
							   gpreagg->m_gpreagg,
							   length,
							   cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyDtoHAsync: %s", errorText(rc));
		gpreagg->bytes_dma_recv += length;
		gpreagg->num_dma_recv++;
	}
	else
	{
		PFMON_EVENT_RECORD(gpreagg, ev_kern_fixvar, cuda_stream);
		PFMON_EVENT_RECORD(gpreagg, ev_dma_recv_start, cuda_stream);
	}

	/*
	 * DMA Recv of the final result buffer
	 */
	length = pds_final->kds.length;
	rc = cuMemcpyDtoHAsync(&pds_final->kds,
						   gpreagg->m_kds_final,
						   length,
						   cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	gpreagg->bytes_dma_recv += length;
	gpreagg->num_dma_recv++;

	PFMON_EVENT_RECORD(gpreagg, ev_dma_recv_stop, cuda_stream);

	/*
	 * Register the callback
	 */
	rc = cuStreamAddCallback(cuda_stream,
							 gpupreagg_respond_task,
							 gpreagg, 0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuStreamAddCallback: %s", errorText(rc));

	return 0;

out_of_resource:
	/* device memory of pds_final must be kept! */
	gpupreagg_cleanup_cuda_resources(gpreagg, false);
	return 1;	/* retry later */
}

/*
 * gpupreagg_process_task
 */
int
gpupreagg_process_task(GpuTask_v2 *gtask,
					   CUmodule cuda_module,
					   CUstream cuda_stream)
{
	GpuPreAggTask  *gpreagg = (GpuPreAggTask *) gtask;
	int				retval;

	if (gpreagg->kern.reduction_mode != GPUPREAGG_ONLY_TERMINATION)
	{
		PG_TRY();
		{
			retval = gpupreagg_process_reduction_task(gpreagg,
													  cuda_module,
													  cuda_stream);
		}
		PG_CATCH();
		{
			bool	release_any = gpupreagg_put_strategy(gpreagg);

			gpupreagg_cleanup_cuda_resources(gpreagg, release_any);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	else
	{
		PG_TRY();
		{
			retval = gpupreagg_process_termination_task(gpreagg,
														cuda_module,
														cuda_stream);
		}
		PG_CATCH();
		{
#ifdef USE_ASSERT_CHECKING
			/* pds_final should be already dereferenced */
			GpuPreAggSharedState   *gpa_sstate = gpreagg->gpa_sstate;
			pgstrom_data_store	   *pds_final  = gpreagg->pds_final;
			SpinLockAcquire(&gpa_sstate->lock);
			Assert(pds_final->ntasks_running == 0 &&
				   pds_final->is_dereferenced);
			SpinLockRelease(&gpa_sstate->lock);
#endif
			gpupreagg_cleanup_cuda_resources(gpreagg, true);
			PG_RE_THROW();
		}
		PG_END_TRY();
	}
	return retval;
}

/*
 * gpupreagg_push_terminator_task
 *
 * It pushes an urgent terminator task, if and when a terminator task got
 * NoDataSpace error on updates of the pds_final. The terminator task still
 * has rows not-reduced-yet, thus, a clone task has to handle its termination
 * job instead. We assume this function is called under the GPU server context.
 */
static void
gpupreagg_push_terminator_task(GpuPreAggTask *gpreagg_old)
{
	GpuContext_v2  *gcontext = gpreagg_old->task.gcontext;
	GpuPreAggTask  *gpreagg_new;
	Size			required;

	Assert(IsGpuServerProcess());
	required = STROMALIGN(offsetof(GpuPreAggTask, kern.kparams) +
						  gpreagg_old->kern.kparams.length);
	gpreagg_new = dmaBufferAlloc(gcontext, required);
	memset(gpreagg_new, 0, required);
	/* GpuTask fields */
	gpreagg_new->task.task_kind   = gpreagg_old->task.task_kind;
	gpreagg_new->task.program_id  = gpreagg_old->task.program_id;
	gpreagg_new->task.gts         = gpreagg_old->task.gts;
	gpreagg_new->task.revision    = gpreagg_old->task.revision;
	gpreagg_new->task.perfmon     = gpreagg_old->task.perfmon;
	gpreagg_new->task.file_desc   = -1;
	gpreagg_new->task.gcontext    = NULL;	/* to be set later */
	gpreagg_new->task.cuda_stream = NULL;	/* to be set later */
	gpreagg_new->task.peer_fdesc  = -1;
	gpreagg_new->task.dma_task_id = 0UL;

	/* GpuPreAggTask fields */
	gpreagg_new->pds_in           = NULL;
	gpreagg_new->kds_head         = NULL;	/* shall not be used */
	gpreagg_new->pds_final        = gpreagg_old->pds_final;
	gpreagg_new->m_kds_final      = gpreagg_old->m_kds_final;
	gpreagg_new->m_ghash          = gpreagg_old->m_ghash;

	gpreagg_old->pds_final        = NULL;
	gpreagg_old->m_kds_final      = 0UL;
	gpreagg_old->m_ghash          = 0UL;
	/* kern_gpupreagg fields */
	gpreagg_new->kern.reduction_mode = GPUPREAGG_ONLY_TERMINATION;
	memcpy(&gpreagg_new->kern.kparams,
		   &gpreagg_old->kern.kparams,
		   gpreagg_old->kern.kparams.length);

	gpuservPushGpuTask(gcontext, &gpreagg_new->task);
}

/*
 * gpupreagg_complete_task
 */
int
gpupreagg_complete_task(GpuTask_v2 *gtask)
{
	GpuPreAggTask		   *gpreagg = (GpuPreAggTask *) gtask;
	GpuPreAggSharedState   *gpa_sstate = gpreagg->gpa_sstate;
	int						retval;

	/*
	 * If this task is responsible to termination, pds_final should be
	 * already dereferenced, and this task is responsible to release
	 * any CUDA resources.
	 */
	if (gpreagg->kern.reduction_mode == GPUPREAGG_ONLY_TERMINATION)
	{
		pgstrom_data_store *pds_final = gpreagg->pds_final;

#ifdef USE_ASSERT_CHECKING
		/*
		 * Task with GPUPREAGG_ONLY_TERMINATION should be kicked on the
		 * pds_final buffer which is already dereferenced.
		 */
		SpinLockAcquire(&gpa_sstate->lock);
		Assert(pds_final->ntasks_running == 0 && pds_final->is_dereferenced);
		SpinLockRelease(&gpa_sstate->lock);
#endif
		/* cleanup any cuda resources */
		gpupreagg_cleanup_cuda_resources(gpreagg, true);

		/*
		 * NOTE: We have no way to recover NUMERIC allocation on fixvar.
		 * It may be preferable to do in the CPU side on demand.
		 * kds->has_numeric gives a hint...
		 */
		return 0;
	}

	if (!gpupreagg_put_strategy(gpreagg))
	{
		/*
		 * This code path means the gpreagg task is not responsible to
		 * terminate the pds_final buffer.
		 */
		gpupreagg_cleanup_cuda_resources(gpreagg, false);
		if (gpreagg->task.kerror.errcode == StromError_Success)
		{
			/* simply release this task, no need to return to the backend */
			retval = -1;
		}
		else if (gpreagg->task.kerror.errcode == StromError_DataStoreNoSpace)
		{
			/* retry GPU kernel again with fresh pds_final */
			retval = 1;
		}
		else
		{
			/*
			 * unless task didn't touch the pds_final buffer, we can help
			 * the error by CPU fallbacks.
			 */
			if (gpreagg->task.kerror.errcode == StromError_CpuReCheck &&
				!gpreagg->kern.progress_final)
			{
				memset(&gpreagg->task.kerror, 0, sizeof(kern_errorbuf));
				gpreagg->task.cpu_fallback = true;
			}
			/* return to the backend for error or fallback */
			retval = 0;
		}
		return retval;
	}

	/*
	 * Elsewhere, this task is responsible to terminate the pds_final buffer.
	 * If recoverable error, we try to recover the errors as long as possible
	 * we can.
	 */
	if (gpreagg->task.kerror.errcode == StromError_Success)
	{
		/* Execute this task again for termination. */
		gpreagg->kern.reduction_mode = GPUPREAGG_ONLY_TERMINATION;
		retval = 1;
	}
	else if (gpreagg->task.kerror.errcode == StromError_CpuReCheck)
	{
		if (!gpreagg->kern.progress_final)
		{
			/*
			 * recoverable CpuReCheck error - launch another terminator task,
			 * then return this task to the backend for CPU fallback.
			 */
			gpupreagg_push_terminator_task(gpreagg);

			memset(&gpreagg->task.kerror, 0, sizeof(kern_errorbuf));
			gpreagg->task.cpu_fallback = true;
			retval = 0;
		}
		else
		{
			/* no need to terminate pds_final, just cleans CUDA resources */
			gpupreagg_cleanup_cuda_resources(gpreagg, true);
		}
		retval = 0;
	}
	else if (gpreagg->task.kerror.errcode == StromError_DataStoreNoSpace)
	{
		/*
		 * MEMO: StromError_DataStoreNoSpace may happen on the two typical
		 * scenarios below:
		 * 1) Lack of @nrooms of kds_slot/ghash because we cannot determine
		 *    exact number of tuples in the pds_in (if KDS_FORMAT_BLOCK).
		 * 2) Lack of remaining tuple slot or extra varlena buffer of
		 *    @pds_final, if our expected number of groups were far from
		 *    the exact one.
		 */

		/* terminate the old pds_final anyway */
		gpupreagg_push_terminator_task(gpreagg);

		if (!gpreagg->kern.progress_final)
		{
			kern_data_store *kds_head = gpreagg->kds_head;
			cl_uint		nitems_real = gpreagg->kern.nitems_real;
			Size		kds_length;

			gpreagg->kern.hash_size
				= Max(gpreagg->kern.hash_size, nitems_real);
			gpreagg->kern.kresults_2_offset
				= STROMALIGN(gpreagg->kern.kresults_1_offset +
							 offsetof(kern_resultbuf, results[nitems_real]));
			kds_length = (STROMALIGN(offsetof(kern_data_store,
											  colmeta[kds_head->ncols])) +
						  STROMALIGN(LONGALIGN(sizeof(Datum) + sizeof(char)) *
									 kds_head->ncols) * nitems_real);
			kds_head->length = kds_length;
			kds_head->nrooms = nitems_real;

			/* Release task's own CUDA resources to restart it */
			gpupreagg_cleanup_cuda_resources(gpreagg, false);
			/* Reset reduction strategy, if not NOGROUP_REDUCTION */
			if (gpreagg->kern.reduction_mode != GPUPREAGG_NOGROUP_REDUCTION)
				gpreagg->kern.reduction_mode = GPUPREAGG_INVALID_REDUCTION;
		}
		else
		{
			/*
			 * We already run nogroup/local/global reduction, and part of
			 * the final reduction. So, next reduction strategy shall be
			 * GPUPREAGG_FINAL_REDUCTION based on the current device memory
			 * contents.
			 * So, we must not release task own CUDA resources here.
			 */
			gpreagg->kern.reduction_mode = GPUPREAGG_FINAL_REDUCTION;
		}
		retval = 1;
	}
	else
	{
		/*
		 * Cleanup any CUDA resources, then return to the backend to raise
		 * an error.
		 */
		gpupreagg_cleanup_cuda_resources(gpreagg, true);
		retval = 0;
	}
	return retval;
}

/*
 * gpupreagg_release_task
 */
void
gpupreagg_release_task(GpuTask_v2 *gtask)
{
	GpuPreAggTask  *gpreagg = (GpuPreAggTask *)gtask;

	if (gpreagg->pds_in)
		PDS_release(gpreagg->pds_in);
	if (gpreagg->pds_final)
		PDS_release(gpreagg->pds_final);
	dmaBufferFree(gpreagg);
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

	/* initialization of path method table */
	memset(&gpupreagg_path_methods, 0, sizeof(CustomPathMethods));
	gpupreagg_path_methods.CustomName          = "GpuPreAgg";
	gpupreagg_path_methods.PlanCustomPath      = PlanGpuPreAggPath;

	/* initialization of plan method table */
	memset(&gpupreagg_scan_methods, 0, sizeof(CustomScanMethods));
	gpupreagg_scan_methods.CustomName          = "GpuPreAgg";
	gpupreagg_scan_methods.CreateCustomScanState
		= CreateGpuPreAggScanState;

	/* initialization of exec method table */
	memset(&gpupreagg_exec_methods, 0, sizeof(CustomExecMethods));
	gpupreagg_exec_methods.CustomName          = "GpuPreAgg";
   	gpupreagg_exec_methods.BeginCustomScan     = ExecInitGpuPreAgg;
	gpupreagg_exec_methods.ExecCustomScan      = ExecGpuPreAgg;
	gpupreagg_exec_methods.EndCustomScan       = ExecEndGpuPreAgg;
	gpupreagg_exec_methods.ReScanCustomScan    = ExecReScanGpuPreAgg;
#if 0
	gpupreagg_exec_methods.EstimateDSMCustomScan = ExecGpuPreAggEstimateDSM;
    gpupreagg_exec_methods.InitializeDSMCustomScan = ExecGpuPreAggInitDSM;
    gpupreagg_exec_methods.InitializeWorkerCustomScan = ExecGpuPreAggInitWorker;
#endif
	gpupreagg_exec_methods.ExplainCustomScan   = ExplainGpuPreAgg;
	/* hook registration */
	create_upper_paths_next = create_upper_paths_hook;
	create_upper_paths_hook = gpupreagg_add_grouping_paths;
}

