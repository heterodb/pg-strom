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
	cl_int			num_grp_keys;	/* number of grouping keys */
	double			plan_ngroups;	/* planned number of groups */
	cl_int			plan_nchunks;	/* planned number of chunks */
	cl_int			plan_extra_sz;	/* planned size of extra-sz per tuple */
	cl_int			key_dist_salt;	/* salt, if more distribution needed */

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

	privs = lappend(privs, makeInteger(gpa_info->num_grp_keys));
	privs = lappend(privs, makeInteger(double_as_long(gpa_info->plan_ngroups)));
	privs = lappend(privs, makeInteger(gpa_info->plan_nchunks));
	privs = lappend(privs, makeInteger(gpa_info->plan_extra_sz));
	privs = lappend(privs, makeInteger(gpa_info->key_dist_salt));
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

	gpa_info->num_grp_keys = intVal(privs, pindex++);
	gpa_info->plan_ngroups = long_as_double(intVal(list_nth(privs, pindex++)));
	gpa_info->plan_nchunks = intVal(list_nth(privs, pindex++));
	gpa_info->plan_extra_sz = intVal(list_nth(privs, pindex++));
	gpa_info->key_dist_salt = intVal(list_nth(privs, pindex++));
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
	CUdeviceptr			m_fhash;	/* final global hash slot */
	CUdeviceptr			m_kds_final;/* final kernel data store (slot) */
	cl_uint				f_ncols;	/* @ncols of kds_final (constant) */
	cl_uint				f_nrooms;	/* @nrooms of kds_final (constant) */
	cl_uint				f_nitems;	/* latest nitems of kds_final on device */
	cl_uint				f_usage;	/* latest usage of kds_final on device */

	/* overall statistics */
	cl_uint				n_tasks;
	cl_uint				n_nogroup_tasks;
	cl_uint				n_local_tasks;
	cl_uint				n_global_tasks;
	cl_uint				n_final_tasks;
	cl_uint				plan_ngroups;		/* num of groups planned */
	cl_uint				exec_ngroups;		/* num of groups actually */
} GpuPreAggSharedState;

typedef struct
{
	GpuTaskState_v2	gts;
	GpuPreAggSharedState *gpa_sstate;

	cl_double		plan_outer_nrows;
	cl_double		plan_ngroups;
	cl_int			plan_nchunks;
	cl_int			plan_extra_sz;
	cl_int			key_dist_salt;
	TupleTableSlot *pseudo_slot;

	List		   *outer_quals;	/* List of ExprState */
	ProjectionInfo *outer_proj;		/* outer tlist -> custom_scan_tlist */
	TupleTableSlot *outer_overflow;
	pgstrom_data_store *outer_pds;
} GpuPreAggState;

/*
 * pgstrom_gpupreagg
 *
 * Host side representation of kern_gpupreagg. It can perform as a message
 * object of PG-Strom, has key of OpenCL device program, a source row/column
 * store and a destination kern_data_store.
 */
typedef struct
{
	GpuTask_v2			task;
	GpuPreAggSharedState *gpa_sstate;
	bool				is_terminator;	/* true, if final gpupreagg task */
	bool				is_retry;		/* true, if task is retried */
	cl_uint				nrows_per_block;/* copy from NVMEScanState, if any */

	/* CUDA resources */
	CUfunction			kern_main;
	CUfunction			kern_fixvar;
	CUdeviceptr			m_gpupreagg;
	CUdeviceptr			m_kds_in;		/* input row/block buffer */
	CUdeviceptr			m_kds_slot;		/* working (global) slot buffer */
	CUdeviceptr			m_ghash;		/* global hash slot */
	CUevent				ev_dma_send_start;
	CUevent				ev_dma_send_stop;
	CUevent				ev_kern_fixvar;
	CUevent				ev_dma_recv_start;
	CUevent				ev_dma_recv_stop;

	/* DMA buffers */
	pgstrom_data_store *pds_in;		/* input row/block buffer */
	pgstrom_data_store *pds_final;	/* final data store (!!shared!!) */
	kern_gpupreagg		kern;
} GpuPreAggTask;

/* declaration of static functions */
static char	   *gpupreagg_codegen(codegen_context *context,
								  PlannerInfo *root,
								  CustomScan *cscan,
								  List *tlist_dev,
								  List *tlist_dev_action,
								  List *outer_tlist,
								  List *outer_quals);
static GpuPreAggSharedState *
create_gpupreagg_shared_state(GpuPreAggState *gpa);
static GpuPreAggSharedState *
get_gpupreagg_shared_state(GpuPreAggSharedState *gpa_sstate);
static void
put_gpupreagg_shared_state(GpuPreAggSharedState *gpa_sstate);
static pgstrom_data_store *
renew_gpupreagg_shared_state(GpuPreAggSharedState *gpa_sstate);

static GpuTask_v2 *gpupreagg_next_task(GpuTaskState_v2 *gts);
static void gpupreagg_ready_task(GpuTaskState_v2 *gts, GpuTask_v2 *gtask);
static void gpupreagg_switch_task(GpuTaskState_v2 *gts, GpuTask_v2 *gtask);
static TupleTableSlot *gpupreagg_next_tuple(GpuTaskState_v2 *gts);

static bool		gpupreagg_task_process(GpuTask_v2 *gtask);
static bool		gpupreagg_task_complete(GpuTask_v2 *gtask);
static void		gpupreagg_task_release(GpuTask_v2 *gtask);

/*
 * Arguments of alternative functions.
 */
#define ALTFUNC_GROUPING_KEY		 50	/* GROUPING KEY */
#define ALTFUNC_CONST_VALUE			 51	/* other constant values */
#define ALTFUNC_CONST_NULL			 52	/* NULL constant value */
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
	  "s:pavg_int4", 2, {INT8OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "avg",    1, {INT4OID},
	  "s:pavg_int4", 2, {INT8OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "avg",    1, {INT8OID},
	  "s:pavg_int8",  3, {INTERNALOID, INT8OID, INT8OID},
	  {ALTFUNC_CONST_NULL, ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "avg",    1, {FLOAT4OID},
	  "s:pavg_fp8", 2, {INT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "avg",    1, {FLOAT8OID},
	  "s:pavg_fp8", 2, {INT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "avg",	1, {NUMERICOID},
	  "s:pavg_numeric",	3, {INTERNALOID, INT8OID, NUMERICOID},
	  {ALTFUNC_CONST_NULL, ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM},
	  DEVKERNEL_NEEDS_NUMERIC, 100
	},
#endif
	/* COUNT(*) = SUM(NROWS(*|X)) */
	{ "count", 0, {},
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_NROWS}, 0, INT_MAX
	},
	{ "count", 1, {ANYOID},
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_NROWS}, 0, INT_MAX
	},
	/* MAX(X) = MAX(PMAX(X)) */
	{ "max", 1, {INT2OID},
	  "varref", 1, {INT2OID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max", 1, {INT4OID},
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max", 1, {INT8OID},
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max", 1, {FLOAT4OID},
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max", 1, {FLOAT8OID},
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "max", 1, {NUMERICOID},
	  "varref", 1, {NUMERICOID},
	  {ALTFUNC_EXPR_PMAX}, DEVKERNEL_NEEDS_NUMERIC, INT_MAX
	},
#endif
	{ "max", 1, {CASHOID},
	  "varref", 1, {CASHOID},
	  {ALTFUNC_EXPR_PMAX}, DEVKERNEL_NEEDS_MONEY, INT_MAX
	},
	{ "max", 1, {DATEOID},
	  "varref", 1, {DATEOID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max", 1, {TIMEOID},
	  "varref", 1, {TIMEOID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max", 1, {TIMESTAMPOID},
	  "varref", 1, {TIMESTAMPOID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},
	{ "max", 1, {TIMESTAMPTZOID},
	  "varref", 1, {TIMESTAMPTZOID},
	  {ALTFUNC_EXPR_PMAX}, 0, INT_MAX
	},

	/* MIX(X) = MIN(PMIN(X)) */
	{ "min", 1, {INT2OID},
	  "varref", 1, {INT2OID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min", 1, {INT4OID},
	  "varref", 1, {INT4OID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min", 1, {INT8OID},
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min", 1, {FLOAT4OID},
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min", 1, {FLOAT8OID},
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "min", 1, {NUMERICOID},
	  "varref", 1, {NUMERICOID},
	  {ALTFUNC_EXPR_PMIN}, DEVKERNEL_NEEDS_NUMERIC, INT_MAX
	},
#endif
	{ "min", 1, {CASHOID},
	  "varref", 1, {CASHOID},
	  {ALTFUNC_EXPR_PMAX}, DEVKERNEL_NEEDS_MONEY, INT_MAX
	},
	{ "min", 1, {DATEOID},
	  "varref", 1, {DATEOID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min", 1, {TIMEOID},
	  "varref", 1, {TIMEOID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min", 1, {TIMESTAMPOID},
	  "varref", 1, {TIMESTAMPOID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},
	{ "min", 1, {TIMESTAMPTZOID},
	  "varref", 1, {TIMESTAMPTZOID},
	  {ALTFUNC_EXPR_PMIN}, 0, INT_MAX
	},

	/* SUM(X) = SUM(PSUM(X)) */
	{ "sum", 1, {INT2OID},
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "sum", 1, {INT4OID},
	  "varref", 1, {INT8OID},
	  {ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "sum", 1, {INT8OID},
	  "s:psum", 2, {INTERNALOID,INT8OID},
	  {ALTFUNC_CONST_NULL,ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "sum", 1, {FLOAT4OID},
	  "varref", 1, {FLOAT4OID},
	  {ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
	{ "sum", 1, {FLOAT8OID},
	  "varref", 1, {FLOAT8OID},
	  {ALTFUNC_EXPR_PSUM}, 0, INT_MAX
	},
#ifdef GPUPREAGG_SUPPORT_NUMERIC
	{ "sum", 1, {NUMERICOID},
	  "s:psum", 2, {INTERNALOID,NUMERICOID},
	  {ALTFUNC_CONST_NULL,ALTFUNC_EXPR_PSUM}, DEVKERNEL_NEEDS_NUMERIC, 100
	},
#endif
	{ "sum", 1, {CASHOID},
	  "varref", 1, {CASHOID},
	  {ALTFUNC_EXPR_PSUM}, DEVKERNEL_NEEDS_MONEY, INT_MAX
	},
	/* STDDEV(X) = EX_STDDEV(NROWS(),PSUM(X),PSUM(X*X)) */
	{ "stddev", 1, {FLOAT4OID},
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "stddev", 1, {FLOAT8OID},
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
#ifdef NOT_USED
	/* X^2 of numeric is risky to overflow errors */
	{ "stddev", 1, {NUMERICOID},
	  "s:pvariance", 3, {INT8OID, NUMERICOID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, DEVKERNEL_NEEDS_NUMERIC, 32
	},
#endif
	{ "stddev_pop", 1, {FLOAT4OID},
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "stddev_pop", 1, {FLOAT8OID},
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
#ifdef NOT_USED
	/* X^2 of numeric is risky to overflow errors */
	{ "stddev_pop", 1, {NUMERICOID},
	  "s:pvariance", 3, {INT8OID, NUMERICOID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2}, DEVKERNEL_NEEDS_NUMERIC, 32
	},
#endif
	{ "stddev_samp", 1, {FLOAT4OID},
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "stddev_samp", 1, {FLOAT8OID},
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
#ifdef NOT_USED
	/* X^2 of numeric is risky to overflow errors */
	{ "stddev_samp", 1, {NUMERICOID},
	  "s:pvariance", 3, {INT8OID, NUMERICOID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, DEVKERNEL_NEEDS_NUMERIC, 32
	},
#endif
	/* VARIANCE(X) = PGSTROM.VARIANCE(NROWS(), PSUM(X),PSUM(X^2)) */
	{ "variance", 1, {FLOAT4OID},
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "variance", 1, {FLOAT8OID},
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
#ifdef NOT_USED
	/* X^2 of numeric is risky to overflow errors */
	{ "variance", 1, {NUMERICOID},
	  "s:pvariance", 3, {INT8OID, NUMERICOID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2}, DEVKERNEL_NEEDS_NUMERIC, 32
	},
#endif
	{ "var_pop", 1, {FLOAT4OID},
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "var_pop", 1, {FLOAT8OID},
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
#ifdef NOT_USED
	/* X^2 of numeric is risky to overflow errors */
	{ "var_pop", 1, {NUMERICOID},
	  "s:pvariance", 3, {INT8OID, NUMERICOID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2}, DEVKERNEL_NEEDS_NUMERIC, 32
	},
#endif
	{ "var_samp", 1, {FLOAT4OID},
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
	{ "var_samp", 1, {FLOAT8OID},
	  "s:pvariance", 3, {INT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0, SHRT_MAX
	},
#ifdef NOT_USED
	/* X^2 of numeric is risky to overflow errors */
	{ "var_samp", 1, {NUMERICOID},
	  "s:pvariance", 3, {INT8OID, NUMERICOID, NUMERICOID},
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
	  "varref", 1, {INT8OID}, {ALTFUNC_EXPR_NROWS}, 0
	},
	{ "regr_intercept", 2, {FLOAT8OID, FLOAT8OID},
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
	int			num_grp_keys = 0;
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
			num_grp_keys++;
	}
	if (num_grp_keys == 0)
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
				  STROMALIGN((sizeof(Datum) + sizeof(bool)) * ncols) * nrooms +
				  STROMALIGN(extra_sz) * nrooms);
	if (kds_length > gpuMemMaxAllocSize())
		return false;	/* expected buffer size is too large */

	/* Cost estimation to setup initial values */
	gpu_cpu_ratio = pgstrom_gpu_operator_cost / cpu_operator_cost;
	startup_cost += (qual_cost.startup +
					 qual_cost.per_tuple * input_ntuples) * gpu_cpu_ratio;
	/* Cost estimation for grouping */
	startup_cost += pgstrom_gpu_operator_cost * num_grp_keys * input_ntuples;
	/* Cost estimation for aggregate function */
	startup_cost += (agg_costs->transCost.startup +
					 agg_costs->transCost.per_tuple *
					 gpu_cpu_ratio * input_ntuples);
	/* Cost estimation to fetch results */
	run_cost += cpu_tuple_cost * num_groups;

	cpath->path.rows			= num_groups * (double)key_dist_salt;
	cpath->path.startup_cost	= startup_cost;
	cpath->path.total_cost		= startup_cost + run_cost;

	gpa_info->plan_ngroups		= num_groups;
	gpa_info->plan_nchunks		= estimate_num_chunks(input_path);
	gpa_info->plan_extra_sz		= extra_sz;
	gpa_info->key_dist_salt		= key_dist_salt;
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
						 PathTarget *target,
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

	/* PathTarget of the partial stage */
	partial_target = make_partial_grouping_target(root, target);
	get_agg_clause_costs(root, (Node *) partial_target->exprs,
						 AGGSPLIT_INITIAL_SERIAL,
						 &agg_partial_costs);

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

	if (!pgstrom_enabled ||
		!enable_gpupreagg ||
		!gpupreagg_device_executable(root, target))
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
											 target,
											 AGG_PLAIN,
											 AGGSPLIT_FINAL_DESERIAL,
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
											 target,
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
									target,
									AGG_SORTED,
									AGGSPLIT_FINAL_DESERIAL,
									parse->groupClause,
									(List *)parse->havingQual,
									&agg_final_costs,
									num_groups);
			else if (parse->groupClause)
				final_path = (Path *)
					create_group_path(root,
									  group_rel,
									  sort_path,
									  target,
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
									target,
									AGG_HASHED,
									AGGSPLIT_FINAL_DESERIAL,
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
 * make_altfunc_nrows_expr - constructor of dummy NULL for 'internal' type
 */
static Expr *
make_altfunc_null_const(Aggref *aggref)
{
	return (Expr *)makeNullConst(INTERNALOID, -1, InvalidOid);
}

/*
 * make_altfunc_nrows_expr - constructor of the partial number of rows
 */
static Expr *
make_altfunc_nrows_expr(Aggref *aggref)
{
	Expr	   *nrows_expr;
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
		nrows_args = lappend(nrows_args, copyObject(aggref->aggfilter));

	nrows_expr = (Expr *) makeConst(INT8OID,
									-1,
									InvalidOid,
									sizeof(int64),
									(Datum) 1,
									false,
									true);
	if (nrows_args == NIL)
		return nrows_expr;

	return make_expr_conditional(nrows_expr,
								 list_length(nrows_args) <= 1
								 ? linitial(nrows_args)
								 : make_andclause(nrows_args),
								 true);
}

/*
 * make_altfunc_pmin_expr - constructor of a simple variable reference
 */
static Expr *
make_altfunc_simple_expr(Aggref *aggref, bool zero_if_unmatched)
{
	TargetEntry	   *tle;
	Expr   *expr;

	Assert(list_length(aggref->args) == 1);
	tle = linitial(aggref->args);
	Assert(IsA(tle, TargetEntry));
	expr = tle->expr;
	if (aggref->aggfilter)
		expr = make_expr_conditional(expr, aggref->aggfilter,
									 zero_if_unmatched);
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
					case ALTFUNC_CONST_NULL:
						/*
						 * NOTE: PostgreSQL does not allows to define
						 * functions that return 'internal' data type
						 * unless it has an 'internal' arguments.
						 * So, some of alternative functions need to have
						 * a dummay argument to avoid the restriction.
						 * It is ignored in the device code, thus, we don't
						 * need to add this entry on the tlist_dev.
						 */
						expr = make_altfunc_null_const(aggref);
						goto found_tlist_dev_entry;	/* skip to add tlist_dev */

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
				{
					namespace_oid = get_namespace_oid("pgstrom", true);
					if (!OidIsValid(namespace_oid))
						ereport(ERROR,
								(errcode(ERRCODE_UNDEFINED_SCHEMA),
								 errmsg("schema \"pgstrom\" was not found"),
								 errhint("Run: CREATE EXTENSION pg_strom")));
				}
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
	elog(INFO, "source:\n%s", kern_source);

	gpa_info->kern_source = kern_source;
	gpa_info->extra_flags = context.extra_flags;
	gpa_info->used_params = context.used_params;


	elog(INFO, "tlist_orig => %s", nodeToString(tlist));
	elog(INFO, "tlist_dev => %s", nodeToString(tlist_dev));
	elog(INFO, "tlist_dev_action => %s", nodeToString(tlist_dev_action));

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
	GpuContext_v2  *gcontext = NULL;
	GpuPreAggState *gpas = (GpuPreAggState *) node;
	CustomScan	   *cscan = (CustomScan *) node->ss.ps.plan;
	GpuPreAggInfo  *gpa_info = deform_gpupreagg_info(cscan);
	List		   *pseudo_tlist;
	TupleDesc		pseudo_tupdesc;
	TupleTableSlot *pseudo_slot;
	bool			has_oid;
	bool			with_connection = ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0);

	Assert(gpa_info->outer_scanrelid == cscan->scan.scanrelid);
	Assert(scan_rel ? outerPlan(node) == NULL : outerPlan(node) != NULL);
	/* activate a GpuContext for CUDA kernel execution */
	gcontext = AllocGpuContext(with_connection);

	/* setup common GpuTaskState fields */
	pgstromInitGpuTaskState(&gpa->gts,
							gcontext,
							GpuTaskKind_GpuPreAgg,
							gpa_info->used_params,
							estate);
	gpas->gts.cb_next_task   = gpupreagg_next_task;
	gpas->gts.cb_ready_task  = gpupreagg_ready_task;
	gpas->gts.cb_switch_task = gpupreagg_switch_task;
	gpas->gts.cb_next_tuple  = gpupreagg_next_tuple;

	gpas->plan_ngroups       = gpa_info->plan_ngroups;
	gpas->plan_nchunks       = gpa_info->plan_nchunks;
	gpas->plan_extra_sz      = gpa_info->plan_extra_sz;
	gpas->key_dist_salt      = gpa_info->key_dist_salt;
	gpas->plan_outer_nrows   = gpa_info->outer_nrows;

	/* initialization of the outer relation */
	if (outerPlan(cscan))
	{
		PlanState  *outer_ps;

		Assert(scan_rel == NULL);
		Assert(gpa_info->outer_quals == NIL);
		outer_ps = ExecInitNode(outerPlan(cscan), estate, eflags);
		if (pgstrom_bulk_exec_supported(outer_ps))
		{
			((GpuTaskState *) outer_ps)->be_row_format = true;
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
			ExecInitExpr((Expr *)gpa_info->outer_quals, &css->css.ss.ps);
		outer_tupdesc = RelationGetDescr(scan_rel);
	}

	/*
	 * Initialization the stuff for CPU fallback.
	 *
	 * Projection from the outer-relation to the custom_scan_tlist is a job
	 * of CPU fallback. It is equivalent to the initial device projection.
	 */
	pseudo_tlist = (List *)
		ExecInitExpr((Expr *)cscan->custom_scan_tlist, &gpas->css.ss.ps);
	if (!ExecContextForcesOids(&gpas->css.ss.ps, &has_oid))
		has_oid = false;
	pseudo_tupdesc = ExecTypeFromTL(cscan->custom_scan_tlist, has_oid);
	gpas->pseudo_slot = MakeSingleTupleTableSlot(pseudo_tupdesc);
	gpas->outer_proj = ExecBuildProjectionInfo(pseudo_tlist,
											   gpas->css.ss.ps.ps_ExprContext,
											   gpas->pseudo_slot,
											   outer_tupdesc);
	gpas->outer_overflow = NULL;
	gpas->outer_pds = NULL;

	/* Create a shared state object */
	gpas->gpa_sstate = create_gpupreagg_shared_state(gpas, pseudo_tupdesc);

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
	pgstromReleaseGpuTaskState(&gss->gts);
}

/*
 * ExecReScanGpuPreAgg
 */
static void
ExecReScanGpuPreAgg(CustomScanState *node)
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



/*
 * create_gpupreagg_shared_state
 */
static GpuPreAggSharedState *
create_gpupreagg_shared_state(GpuPreAggState *gpas, TupleDesc tupdesc)
{
	GpuContext_v2  *gcontext = gpas->gcontext;
	GpuPreAggSharedState   *gpa_sstate;
	cl_uint			nrooms;
	Size			head_sz;
	Size			unit_sz;
	Size			length;

	Assert(tupdesc->natts > 0);
	/* expected number of groups + safety margin */
	nrooms = (cl_uint)(gpas->plan_ngroups * 2.5 + 200.0);
	head_sz = (STROMALIGN(offsetof(kern_data_store,
								   colmeta[tupdesc->natts])));
	unit_sz = (STROMALIGN(LONGALIGN((sizeof(Datum) + sizeof(char)))) +
			   STROMALIGN(gpas->plan_extra_sz));
	length = head_sz + unit_sz * nrooms;

	/*
	 * Expand nrooms if length of kds_final is expected small, because
	 * planner tends to estimate # of groups smaller than actual.
	 */
	if (length < pgstrom_chunk_size() / 2)
		nrooms = (pgstrom_chunk_size() - head_sz) / unit_sz;
	else if (length < pgstrom_chunk_size())
		nrooms = (2 * pgstrom_chunk_size() - head_sz) / unit_sz;
	else if (length < 3 * pgstrom_chunk_size())
		nrooms = (3 * pgstrom_chunk_size() - head_sz) / unit_sz;

	gpa_sstate = dmaBufferAlloc(gcontext, sizeof(GpuPreAggSharedState));
	memset(gpa_sstate, 0, sizeof(GpuPreAggSharedState));
	pg_atomic_init_u32(&gpa_sstate->refcnt, 1);
	SpinLockInit(&gpa_sstate->lock);
	gpa_sstate->pds_final = PDS_create_slot(gcontext,
											tupdesc,
											nrooms,
											gpas->plan_extra_sz,
											true);
	gpa_sstate->m_fhash = 0UL;
	gpa_sstate->m_kds_final = 0UL;
	gpa_sstate->f_ncols = tupdesc->natts;
	gpa_sstate->f_nrooms = nrooms;
	gpa_sstate->f_nitems = 0;
	gpa_sstate->f_usage = 0

	return gpa_sstate;
}

/*
 * get_gpupreagg_shared_state
 */
static GpuPreAggSharedState *
get_gpupreagg_shared_state(GpuPreAggSharedState *gpa_sstate)
{
	uint32		refcnt_old	__attribute__((unused));

	refcnt_old = pg_atomic_fetch_add_u32(&gpa_sstate->refcnt);
	Assert(refcnt_old > 0);

	return gpa_sstate;
}

/*
 * put_gpupreagg_shared_state
 */
static void
put_gpupreagg_shared_state(GpuPreAggSharedState *gpa_sstate)
{
	if (pg_atomic_sub_fetch_u32() == 0)
	{
		Assert(!gpa_sstate->pds_final);
		Assert(gpa_sstate->m_fhash == 0UL);
		Assert(gpa_sstate->m_kds_final);
		dmaBufferFree(gpa_sstate);
	}
}

/*
 * renew_gpupreagg_shared_state
 */
static pgstrom_data_store *
renew_gpupreagg_shared_state(GpuPreAggSharedState *gpa_sstate,
							 GpuContext_v2 *gcontext)
{

	/* renew a pds_final according to the exec_ngroups */

	return NULL;
}

















/*
 * gpupreagg_create_task - constructor of GpuPreAggTask
 */
static GpuTask_v2 *
gpupreagg_create_task(GpuPreAggState *gpas,
					  pgstrom_data_store *pds,
					  int file_desc,
					  bool is_terminator)
{
	GpuContext_v2  *gcontext = gpas->gts.gcontext;
	GpuPreAggTask  *gpreagg;
	TupleDesc		tupdesc;
	cl_uint			nitems = pds->kds.nitems;
	Size			length;

	/* allocation of GpuPreAggTask */
	tupdesc = gpas->pseudo_slot->tts_tupleDescriptor;
	length = STROMALIGN(offsetof(GpuPreAggTask, kern.kparams) +
						gpas->gts.kern_params->length);
	gpreagg = dmaBufferAlloc(gcontext, length);
	memset(gpreagg, 0, length);

	pgstromInitGpuTask(&gpas->gts, &gpreagg->task);
	gpreagg->gpa_sstate = GetGpuPreAggSharedState(gpas->gpa_sstate);
	gpreagg->is_terminator = is_terminator;
	if (gpas->gts.nvme_sstate)
		gpreagg->nrows_per_block = gpas->gts.nvme_sstate->nrows_per_block;

	gpreagg->pds_in = pds_in;
	gpreagg->pds_final = NULL;	/* to be attached later */

	/* if any grouping keys, determine the reduction policy later */
	gpreagg->kern.reduction_mode = (gpas->num_group_keys == 0
									? GPUPREAGG_NOGROUP_REDUCTION
									: GPUPREAGG_INVALID_REDUCTION);
	gpreagg->kern.key_dist_salt = gpas->key_dist_salt;
	gpreagg->kern.hash_size = nitems;
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
					 offsetof(kern_resultbuf, results[nitems]));

	return &gpreagg->task;



#if HOGEHOGE
	// reduction mode is assigned on the task creation
	// any reduction are eventually moved to final_kds
	if (gpa_info->num_grp_keys == 0)
		gpas->reduction_mode	= GPUPREAGG_NOGROUP_REDUCTION;
	else if (gpa_info->plan_ngroups < devBaselineMaxThreadsPerBlock / 4)
		gpas->reduction_mode	= GPUPREAGG_LOCAL_REDUCTION;
	else if (gpa_info->plan_ngroups < (gpa_info->outer_nrows /
									   gpa_info->plan_nchunks) / 4)
		gpas->reduction_mode	= GPUPREAGG_GLOBAL_REDUCTION;
	else
		gpas->reduction_mode	= GPUPREAGG_FINAL_REDUCTION;
#endif



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
	pgstrom_data_store *pds = NULL;
	int					filedesc = -1;
	struct timeval		tv1, tv2;

	PFMON_BEGIN(&gts->pfm, &tv1);
	if (gjs->gts.css.ss.ss_currentRelation)
	{
		/* scan and load the outer relation by itself */
		pds = gpuscanExecScanChunk(gts, &filedesc);
		if (!pds)
			;
	}
	else
	{
		PlanState	   *outer_ps = outerPlanState(gjs);
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
					gjs->outer_scan_done = true;
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
	}
	PFMON_END(&gpas->gts.pfm, time_outer_load, &tv1, &tv2);

	return gpupreagg_create_task(gjs, pds, filedesc);
}


static void
gpupreagg_ready_task(GpuTaskState_v2 *gts, GpuTask_v2 *gtask)
{
	
}

static void
gpupreagg_switch_task(GpuTaskState_v2 *gts, GpuTask_v2 *gtask)
{}

static TupleTableSlot *
gpupreagg_next_tuple(GpuTaskState_v2 *gts)
{}



/*
 * gpupreagg_setup_strategy
 *
 * It determines the strategy to run GpuPreAgg kernel according to the run-
 * time statistics.
 * Number of groups is the most important decision. If estimated number of
 * group is larger than the maximum block size, local reduction makes no
 * sense. If too small, final reduction without local/global reduction will
 * lead massive amount of atomic contention.
 * In addition, this function switches the @pds_final buffer if remaining
 * space is not sufficient to hold the groups appear.
 */
static void
gpupreagg_setup_strategy(GpuPreAggTask *gpreagg)
{
	GpuPreAggSharedState   *gpa_sstate = gpreagg->gpa_sstate;
	pgstrom_data_store	   *pds_in = gpreagg->pds_in;
	pgstrom_data_store	   *pds_final;

	Assert(pds_in->kds.format == KDS_FORMAT_ROW ||
		   pds_in->kds.format == KDS_FORMAT_BLOCK);
	SpinLockAcquire(&gpa_sstate->lock);

	/* decision for the reduction mode */
	if (gpreagg->kern.reduction_mode != GPUPREAGG_INVALID_REDUCTION)
	{
		cl_double	real_ngroups;
		cl_double	exec_ratio;
		cl_uint		nitems;

		exec_ratio = (double)Min(gpa_sstate->n_tasks, 30) / 30.0;
		real_ngroups = ((double)gpa_sstate->plan_ngroups * (1.0 - exec_ratio) +
						(double)gpa_sstate->exec_ngroups * exec_ratio);
		nitems = (pds_in->kds.format == KDS_FORMAT_BLOCK
				  ? pds_in->kds.nitems * gpreagg->nrows_per_block
				  : pds_in->kds.nitems);
		if (real_ngroups < devBaselineMaxThreadsPerBlock / 4)
			gpreagg->kern.reduction_mode = GPUPREAGG_LOCAL_REDUCTION;
		else if (real_ngroups < nitems / 4)
			gpreagg->kern.reduction_mode = GPUPREAGG_GLOBAL_REDUCTION;
		else
			gpreagg->kern.reduction_mode = GPUPREAGG_FINAL_REDUCTION;
	}

	/*
	 * attach pds_final to the task
	 *
	 * TODO: switch pds_final is less amount of slots are remained
	 */
	pds_final = PDS_retain(gpa_sstate->pds_final);
	/* also mark this pds_final is in use by kernel function */
	pg_atomic_fetch_add_u32(&pds_final->ntasks_running, 1);
	SpinLockRelease(&gpa_sstate->lock);

	Assert(gpreagg->pds_final == NULL);
	gpreagg->pds_final = pds_final;
}

/*
 * gpupreagg_respond_task - callback handler on CUDA context
 */
static void
gpupreagg_respond_task(CUstream stream, CUresult status, void *private)
{
	GpuPreAggTask  *gpreagg = private;



}

/*
 * gpupreagg_process_task
 *
 * main logic to kick GpuPreAgg kernel function.
 */
static int
__gpupreagg_process_task(GpuPreAggTask *gpreagg,
						 CUmodule cuda_module,
						 CUstream cuda_stream)
{



}

int
gpupreagg_process_task(GpuTask_v2 *gtask,
					   CUmodule cuda_module,
					   CUstream cuda_stream)
{
	GpuPreAggTask  *gpreagg = (GpuPreAggTask *) gtask;
	int				retval;

	gpupreagg_setup_strategy(gpreagg);
	PG_TRY();
	{
		retval = __gpupreagg_process_task(gpreagg,
										  cuda_module,
										  cuda_stream);
	}
	PG_CATCH();
	{
		// detach pds_final on errors


		PG_RE_THROW();
	}
	PG_END_TRY();

	return retval;
}

/*
 * gpupreagg_complete_task
 */
int
gpupreagg_complete_task(GpuTask_v2 *gtask)
{}

/*
 * gpupreagg_release_task
 */
void
gpupreagg_release_task(GpuTask_v2 *gtask)
{}














#ifdef HOGEHOGEHOGE
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



static void
gpupreagg_ready_task(GpuTaskState_v2 *gts, GpuTask_v2 *gtask)
{
	// release task unless it has final_pds

}


static void
gpupreagg_switch_task(GpuTaskState_v2 *gts, GpuTask_v2 *gtask)
{

}

static TupleTableSlot *
gpupreagg_next_tuple(GpuTaskState_v2 *gts)
{
	return NULL;
}

static GpuTask_v2 *
gpupreagg_next_task(GpuTaskState_v2 *gts)
{
	return NULL;
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


static void
ExplainGpuPreAgg(CustomScanState *node, List *ancestors, ExplainState *es)
{
	GpuPreAggState *gpas = (GpuPreAggState *) node;
	CustomScan	   *cscan = (CustomScan *) node->ss.ps.plan;
	GpuPreAggInfo  *gpa_info = deform_gpupreagg_info(cscan);
	List		   *dcontext;
	List		   *dev_proj = NIL;
	ListCell	   *lc;
	const char	   *policy;
	char		   *exprstr;
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
	dcontext = set_deparse_context_planstate(es->deparse_cxt,
                                            (Node *)&gpas->gts.css.ss.ps,
                                            ancestors);
	/* Show device projection */
	foreach (lc, gpa_info->tlist_dev)
		dev_proj = lappend(dev_proj, ((TargetEntry *) lfirst(lc))->expr);
	if (dev_proj != NIL)
	{
		exprstr = deparse_expression((Node *)dev_proj, dcontext,
									 es->verbose, false);
		ExplainPropertyText("GPU Projection", exprstr, es);
	}
	/* statistics for outer scan, if it was pulled-up */
	pgstrom_explain_outer_bulkexec(&gpas->gts, context, ancestors, es);
	/* Show device filter */
	if (gpa_info->outer_quals != NIL)
	{
		Node	   *outer_quals
			= (Node *) make_ands_explicit(gpa_info->outer_quals);
		exprstr = deparse_expression(outer_quals, dcontext,
									 es->verbose, false);
		ExplainPropertyText("GPU Filter", exprstr, es);
	}
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
#endif //HOGEHOGEHOGE

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
	gpupreagg_exec_methods.ExplainCustomScan   = ExplainGpuPreAgg;
#endif
	/* hook registration */
	create_upper_paths_next = create_upper_paths_hook;
	create_upper_paths_hook = gpupreagg_add_grouping_paths;
}
