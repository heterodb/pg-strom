/*
 * gpupreagg.c
 *
 * Aggregate Pre-processing with GPU acceleration
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
#include "access/nbtree.h"
#include "access/sysattr.h"
#include "catalog/namespace.h"
#include "catalog/pg_aggregate.h"
#include "catalog/pg_cast.h"
#include "catalog/pg_namespace.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_type.h"
#include "executor/nodeAgg.h"
#include "miscadmin.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "parser/parse_func.h"
#include "optimizer/clauses.h"
#include "optimizer/cost.h"
#include "optimizer/var.h"
#include "parser/parsetree.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/numeric.h"
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

typedef enum
{
	AGGCALC_LOCAL_REDUCTION,
	AGGCALC_GLOBAL_REDUCTION,
	AGGCALC_NOGROUP_REDUCTION
} AggCalcMode_t;

typedef struct
{
	int				numCols;		/* number of grouping columns */
	AttrNumber	   *grpColIdx;		/* their indexes in the target list */
	bool			outer_bulkload;
	double			num_groups;		/* estimated number of groups */
	List		   *outer_quals;	/* device quals pulled-up */
	const char	   *kern_source;
	int				extra_flags;
	List		   *used_params;	/* referenced Const/Param */
	Bitmapset	   *outer_attrefs;	/* bitmap of referenced outer attributes */
	Bitmapset	   *tlist_attrefs;	/* bitmap of referenced tlist attributes */
	bool			has_numeric;	/* if true, result contains numeric val */
	bool			has_varlena;	/* if true, result contains varlena val */
} GpuPreAggInfo;

static inline void
form_gpupreagg_info(CustomScan *cscan, GpuPreAggInfo *gpa_info)
{
	List	   *privs = NIL;
	List	   *exprs = NIL;
	List	   *temp;
	Bitmapset  *tempset;
	int			i;
	union {
		long	ival;
		double	fval;
	} datum;

	/* numCols and grpColIdx */
	temp = NIL;
	for (i = 0; i < gpa_info->numCols; i++)
		temp = lappend_int(temp, gpa_info->grpColIdx[i]);

	privs = lappend(privs, temp);
	privs = lappend(privs, makeInteger(gpa_info->outer_bulkload));
	datum.fval = gpa_info->num_groups;
	privs = lappend(privs, makeInteger(datum.ival));
	exprs = lappend(exprs, gpa_info->outer_quals);
	privs = lappend(privs, makeString(pstrdup(gpa_info->kern_source)));
	privs = lappend(privs, makeInteger(gpa_info->extra_flags));
	exprs = lappend(exprs, gpa_info->used_params);
	/* outer_attrefs */
	temp = NIL;
	tempset = bms_copy(gpa_info->outer_attrefs);
	while ((i = bms_first_member(tempset)) >= 0)
		temp = lappend_int(temp, i);
	privs = lappend(privs, temp);
	bms_free(tempset);
	/* tlist_attrefs */
	temp = NIL;
	tempset = bms_copy(gpa_info->tlist_attrefs);
	while ((i = bms_first_member(tempset)) >= 0)
		temp = lappend_int(temp, i);
	privs = lappend(privs, temp);
	bms_free(tempset);
	privs = lappend(privs, makeInteger(gpa_info->has_numeric));
	privs = lappend(privs, makeInteger(gpa_info->has_varlena));

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
	Bitmapset  *tempset;
	List	   *temp;
	ListCell   *cell;
	union {
		long	ival;
		double	fval;
	} datum;

	/* numCols and grpColIdx */
	temp = list_nth(privs, pindex++);
	gpa_info->numCols = list_length(temp);
	gpa_info->grpColIdx = palloc0(sizeof(AttrNumber) * gpa_info->numCols);
	foreach (cell, temp)
		gpa_info->grpColIdx[i++] = lfirst_int(cell);

	gpa_info->outer_bulkload = intVal(list_nth(privs, pindex++));
	datum.ival = intVal(list_nth(privs, pindex++));
	gpa_info->num_groups = datum.fval;
	gpa_info->outer_quals = list_nth(exprs, eindex++);
	gpa_info->kern_source = strVal(list_nth(privs, pindex++));
	gpa_info->extra_flags = intVal(list_nth(privs, pindex++));
	gpa_info->used_params = list_nth(exprs, eindex++);
	/* outer_attrefs */
	tempset = NULL;
	temp = list_nth(privs, pindex++);
	foreach (cell, temp)
		tempset = bms_add_member(tempset, lfirst_int(cell));
	gpa_info->outer_attrefs = tempset;

	/* tlist_attrefs */
	tempset = NULL;
	temp = list_nth(privs, pindex++);
	foreach (cell, temp)
		tempset = bms_add_member(tempset, lfirst_int(cell));
	gpa_info->tlist_attrefs = tempset;

	gpa_info->has_numeric = intVal(list_nth(privs, pindex++));
	gpa_info->has_varlena = intVal(list_nth(privs, pindex++));

	return gpa_info;
}






typedef struct
{
	GpuTaskState	gts;
	ProjectionInfo *bulk_proj;
	TupleTableSlot *bulk_slot;
	double			num_groups;		/* estimated number of groups */
	double			ntups_per_page;	/* average number of tuples per page */
	List		   *outer_quals;
	//bool			outer_done;
	//bool			outer_bulkload;
	TupleTableSlot *outer_overflow;

	kern_parambuf  *kparams;
	bool			needs_grouping;
	bool			local_reduction;
	bool			has_numeric;
	bool			has_varlena;

	cl_uint			num_rechecks;
} GpuPreAggState;

/* Host side representation of kern_gpupreagg. It can perform as a message
 * object of PG-Strom, has key of OpenCL device program, a source row/column
 * store and a destination kern_data_store.
 */
typedef struct
{
	GpuTask			task;
	bool			needs_fallback;	/* true, if StromError_CpuReCheck */
	bool			needs_grouping;	/* true, if it takes GROUP BY clause */
	bool			local_reduction;/* true, if it needs local reduction */
	bool			has_varlena;	/* true, if it has varlena grouping keys */
	double			num_groups;		/* estimated number of groups */
	CUfunction		kern_prep;
	void		   *kern_prep_args[4];
	CUfunction		kern_lagg;
	void		   *kern_lagg_args[5];
	CUfunction		kern_gagg;
	void		   *kern_gagg_args[4];
	CUfunction		kern_nogrp;
	void		   *kern_nogrp_args[4];
	CUfunction		kern_fixvar;
	void		   *kern_fixvar_args[3];
	CUdeviceptr		m_gpreagg;
	CUdeviceptr		m_kds_in;		/* kds_in : input stream */
	CUdeviceptr		m_kds_src;		/* kds_src : slot form of kds_in */
	CUdeviceptr		m_kds_dst;		/* kds_dst : final aggregation result */
	CUdeviceptr		m_ghash;		/* global hash slot */
	CUevent			ev_dma_send_start;
	CUevent			ev_dma_send_stop;
	CUevent			ev_kern_prep_end;
	CUevent			ev_kern_lagg_end;
	CUevent			ev_dma_recv_start;
	CUevent			ev_dma_recv_stop;
	pgstrom_data_store *pds_in;		/* source data-store */
	pgstrom_data_store *pds_dst;	/* result data-store */
	kern_resultbuf *kresults;
	kern_gpupreagg	kern;
} pgstrom_gpupreagg;

/* declaration of static functions */
static bool		gpupreagg_task_process(GpuTask *gtask);
static bool		gpupreagg_task_complete(GpuTask *gtask);
static void		gpupreagg_task_fallback(GpuTask *gtask);
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
	int			altfn_flags;
} aggfunc_catalog_t;
static aggfunc_catalog_t  aggfunc_catalog[] = {
	/* AVG(X) = EX_AVG(NROWS(), PSUM(X)) */
	{ "avg",    1, {INT2OID},
	  "s:avg",  2, {INT4OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0
	},
	{ "avg",    1, {INT4OID},
	  "s:avg",  2, {INT4OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0
	},
	{ "avg",    1, {INT8OID},
	  "s:avg_int8",  2, {INT4OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0
	},
	{ "avg",    1, {FLOAT4OID},
	  "s:avg",  2, {INT4OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0
	},
	{ "avg",    1, {FLOAT8OID},
	  "s:avg",  2, {INT4OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, 0
	},
	{ "avg",	1, {NUMERICOID},
	  "s:avg_numeric",	2, {INT4OID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}, DEVFUNC_NEEDS_NUMERIC
	},
	/* COUNT(*) = SUM(NROWS(*|X)) */
	{ "count", 0, {},
	  "s:count", 1, {INT4OID},
	  {ALTFUNC_EXPR_NROWS}, 0},
	{ "count", 1, {ANYOID},
	  "s:count", 1, {INT4OID},
	  {ALTFUNC_EXPR_NROWS}, 0},
	/* MAX(X) = MAX(PMAX(X)) */
	{ "max", 1, {INT2OID},   "c:max", 1, {INT2OID},   {ALTFUNC_EXPR_PMAX}, 0},
	{ "max", 1, {INT4OID},   "c:max", 1, {INT4OID},   {ALTFUNC_EXPR_PMAX}, 0},
	{ "max", 1, {INT8OID},   "c:max", 1, {INT8OID},   {ALTFUNC_EXPR_PMAX}, 0},
	{ "max", 1, {FLOAT4OID}, "c:max", 1, {FLOAT4OID}, {ALTFUNC_EXPR_PMAX}, 0},
	{ "max", 1, {FLOAT8OID}, "c:max", 1, {FLOAT8OID}, {ALTFUNC_EXPR_PMAX}, 0},
	{ "max", 1, {NUMERICOID},"c:max", 1, {NUMERICOID},
	  {ALTFUNC_EXPR_PMAX}, DEVFUNC_NEEDS_NUMERIC},
	/* MIX(X) = MIN(PMIN(X)) */
	{ "min", 1, {INT2OID},   "c:min", 1, {INT2OID},   {ALTFUNC_EXPR_PMIN}, 0},
	{ "min", 1, {INT4OID},   "c:min", 1, {INT4OID},   {ALTFUNC_EXPR_PMIN}, 0},
	{ "min", 1, {INT8OID},   "c:min", 1, {INT8OID},   {ALTFUNC_EXPR_PMIN}, 0},
	{ "min", 1, {FLOAT4OID}, "c:min", 1, {FLOAT4OID}, {ALTFUNC_EXPR_PMIN}, 0},
	{ "min", 1, {FLOAT8OID}, "c:min", 1, {FLOAT8OID}, {ALTFUNC_EXPR_PMIN}, 0},
	{ "min", 1, {NUMERICOID},"c:min", 1, {NUMERICOID},
	  {ALTFUNC_EXPR_PMIN}, DEVFUNC_NEEDS_NUMERIC},
	/* SUM(X) = SUM(PSUM(X)) */
	{ "sum", 1, {INT2OID},   "s:sum", 1, {INT8OID},   {ALTFUNC_EXPR_PSUM}, 0},
	{ "sum", 1, {INT4OID},   "s:sum", 1, {INT8OID},   {ALTFUNC_EXPR_PSUM}, 0},
	{ "sum", 1, {INT8OID},   "c:sum", 1, {INT8OID},   {ALTFUNC_EXPR_PSUM}, 0},
	{ "sum", 1, {FLOAT4OID}, "c:sum", 1, {FLOAT4OID}, {ALTFUNC_EXPR_PSUM}, 0},
	{ "sum", 1, {FLOAT8OID}, "c:sum", 1, {FLOAT8OID}, {ALTFUNC_EXPR_PSUM}, 0},
	{ "sum", 1, {NUMERICOID},"c:sum", 1, {NUMERICOID},
	  {ALTFUNC_EXPR_PSUM}, DEVFUNC_NEEDS_NUMERIC},
	/* STDDEV(X) = EX_STDDEV(NROWS(),PSUM(X),PSUM(X*X)) */
	{ "stddev", 1, {FLOAT4OID},
	  "s:stddev", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0
	},
	{ "stddev", 1, {FLOAT8OID},
	  "s:stddev", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0
	},
	{ "stddev", 1, {NUMERICOID},
	  "s:stddev", 3, {INT4OID, NUMERICOID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, DEVFUNC_NEEDS_NUMERIC
	},
	{ "stddev_pop", 1, {FLOAT4OID},
	  "s:stddev_pop", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0
	},
	{ "stddev_pop", 1, {FLOAT8OID},
	  "s:stddev_pop", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0
	},
	{ "stddev_pop", 1, {NUMERICOID},
	  "s:stddev_pop", 3, {INT4OID, NUMERICOID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2}, DEVFUNC_NEEDS_NUMERIC
	},
	{ "stddev_samp", 1, {FLOAT4OID},
	  "s:stddev_samp", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0
	},
	{ "stddev_samp", 1, {FLOAT8OID},
	  "s:stddev_samp", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0
	},
	{ "stddev_samp", 1, {NUMERICOID},
	  "s:stddev_samp", 3, {INT4OID, NUMERICOID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, DEVFUNC_NEEDS_NUMERIC
	},
	/* VARIANCE(X) = PGSTROM.VARIANCE(NROWS(), PSUM(X),PSUM(X^2)) */
	{ "variance", 1, {FLOAT4OID},
	  "s:variance", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0
	},
	{ "variance", 1, {FLOAT8OID},
	  "s:variance", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0
	},
	{ "variance", 1, {NUMERICOID},
	  "s:variance", 3, {INT4OID, NUMERICOID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2}, DEVFUNC_NEEDS_NUMERIC
	},
	{ "var_pop", 1, {FLOAT4OID},
	  "s:var_pop", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0
	},
	{ "var_pop", 1, {FLOAT8OID},
	  "s:var_pop", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0
	},
	{ "var_pop", 1, {NUMERICOID},
	  "s:var_pop", 3, {INT4OID, NUMERICOID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2}, DEVFUNC_NEEDS_NUMERIC
	},
	{ "var_samp", 1, {FLOAT4OID},
	  "s:var_samp", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0
	},
	{ "var_samp", 1, {FLOAT8OID},
	  "s:var_samp", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}, 0
	},
	{ "var_samp", 1, {NUMERICOID},
	  "s:var_samp", 3, {INT4OID, NUMERICOID, NUMERICOID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_PSUM,
       ALTFUNC_EXPR_PSUM_X2}, DEVFUNC_NEEDS_NUMERIC
	},
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
	   ALTFUNC_EXPR_PCOV_XY}, 0},
	{ "covar_pop", 2, {FLOAT8OID, FLOAT8OID},
	  "s:covar_pop", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0},
	{ "covar_samp", 2, {FLOAT8OID, FLOAT8OID},
	  "s:covar_samp", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0},
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
       ALTFUNC_EXPR_PCOV_XY}, 0
	},
	{ "regr_avgy", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_avgy", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0
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
	   ALTFUNC_EXPR_PCOV_XY}, 0
	},
	{ "regr_r2", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_r2", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0
	},
	{ "regr_slope", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_slope", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0
	},
	{ "regr_sxx", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_sxx", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0
	},
	{ "regr_sxy", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_sxy", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0
	},
	{ "regr_syy", 2, {FLOAT8OID, FLOAT8OID},
	  "s:regr_syy", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}, 0
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
			   Plan *p_newcost_gpreagg)
{
	Cost		startup_cost;
	Cost		run_cost;
	Cost		comparison_cost;
	QualCost	pagg_cost;
	int			pagg_width;
	int			outer_width;
	double		outer_rows;
	double		rows_per_chunk;
	double		num_chunks;
	double		num_groups = Max(agg->plan.plan_rows, 1.0);
	ListCell   *cell;
	Path		dummy;

	Assert(outer_plan != NULL);
	/*
	 * GpuPreAgg internally takes partial sort and aggregation
	 * on GPU devices. It is a factor of additional calculation,
	 * but reduce number of rows to be processed on the later
	 * stage.
	 * Items to be considered is:
	 * - cost for sorting by GPU
	 * - cost for aggregation by GPU
	 * - number of rows being reduced.
	 */
	startup_cost = outer_plan->startup_cost;
	run_cost = outer_plan->total_cost - startup_cost;
	outer_rows = outer_plan->plan_rows;
	outer_width = outer_plan->plan_width;

	/*
	 * fixed cost to launch GPU feature
	 */
	startup_cost += pgstrom_gpu_setup_cost;

	/*
	 * cost estimation of internal sorting by GPU.
	 */
	rows_per_chunk =
		((double)((pgstrom_chunk_size()) / BLCKSZ)) *
		((double)(BLCKSZ - MAXALIGN(SizeOfPageHeaderData))) /
        ((double)(sizeof(ItemIdData) +
				  MAXALIGN(sizeof(HeapTupleHeaderData) +
						   outer_width)));
	num_chunks = outer_rows / rows_per_chunk;
	if (num_chunks < 1.0)
		num_chunks = 1.0;

	comparison_cost = 2.0 * pgstrom_gpu_operator_cost;
	startup_cost += (comparison_cost *
					 LOG2(rows_per_chunk * rows_per_chunk) *
					 num_chunks);
	run_cost += pgstrom_gpu_operator_cost * outer_rows;

	/*
	 * cost estimation of partial aggregate by GPU
	 */
	memset(&pagg_cost, 0, sizeof(QualCost));
	pagg_width = 0;
	foreach (cell, gpupreagg_tlist)
	{
		TargetEntry	   *tle = lfirst(cell);
		QualCost		cost;

		/* no code uses PlannerInfo here. NULL may be OK */
		cost_qual_eval_node(&cost, (Node *) tle->expr, NULL);
		pagg_cost.startup += cost.startup;
		pagg_cost.per_tuple += cost.per_tuple;

		pagg_width += get_typavgwidth(exprType((Node *) tle->expr),
									  exprTypmod((Node *) tle->expr));
	}
	startup_cost += pagg_cost.startup;
    run_cost += (pagg_cost.per_tuple *
				 pgstrom_gpu_operator_cost /
				 cpu_operator_cost *
				 LOG2(rows_per_chunk) *
				 num_chunks);
	/*
	 * set cost values on GpuPreAgg
	 */
	p_newcost_gpreagg->startup_cost = startup_cost;
	p_newcost_gpreagg->total_cost = startup_cost + run_cost;
	p_newcost_gpreagg->plan_rows = num_groups * num_chunks;
	p_newcost_gpreagg->plan_width = pagg_width;

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
		p_newcost_sort->plan_rows = p_newcost_gpreagg->plan_rows;
		p_newcost_sort->plan_width = p_newcost_gpreagg->plan_width;
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
			 p_newcost_gpreagg->plan_rows);
	p_newcost_agg->startup_cost = dummy.startup_cost;
	p_newcost_agg->total_cost   = dummy.total_cost;
	p_newcost_agg->plan_rows    = agg->plan.plan_rows;
	p_newcost_agg->plan_width   = agg->plan.plan_width;
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
	case_when->expr = expr_fixup_varno(filter, OUTER_VAR, INDEX_VAR);
	case_when->result = expr_fixup_varno(expr, OUTER_VAR, INDEX_VAR);
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
make_gpupreagg_refnode(Aggref *aggref, List **prep_tlist, int *extra_flags)
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
	*extra_flags |= aggfn_cat->altfn_flags;

	/*
	 * Expression node that is executed in the device kernel has to be
	 * supported by codegen.c
	 */
	foreach (cell, aggref->args)
	{
		TargetEntry *tle = lfirst(cell);
		if (!pgstrom_codegen_available_expression(tle->expr))
			return NULL;
	}
	if (!pgstrom_codegen_available_expression(aggref->aggfilter))
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
	List	   *pre_tlist;
	Bitmapset  *attr_refs;
	int			extra_flags;
	bool		gpupreagg_invalid;
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
										&context->pre_tlist,
										&context->extra_flags);
		if (!altagg)
			context->gpupreagg_invalid = true;
		return (Node *) altagg;
	}
	else if (IsA(node, Var))
	{
		Agg	   *agg = context->agg;
		Var	   *varnode = (Var *) node;
		int		i, x;

		if (varnode->varno != OUTER_VAR)
			elog(ERROR, "Bug? varnode references did not outer relation");
		for (i=0; i < agg->numCols; i++)
		{
			if (varnode->varattno == agg->grpColIdx[i])
			{
				x = varnode->varattno - FirstLowInvalidHeapAttributeNumber;
				context->attr_refs = bms_add_member(context->attr_refs, x);

				return copyObject(varnode);
			}
		}
		context->gpupreagg_invalid = true;
		return NULL;
	}
	return expression_tree_mutator(node, gpupreagg_rewrite_mutator,
								   (void *)context);
}

static bool
gpupreagg_rewrite_expr(Agg *agg,
					   List **p_agg_tlist,
					   List **p_agg_quals,
					   List **p_pre_tlist,
					   Bitmapset **p_attr_refs,
					   int	*p_extra_flags,
					   bool *p_has_numeric,
					   bool *p_has_varlena)
{
	gpupreagg_rewrite_context context;
	Plan	   *outer_plan = outerPlan(agg);
	List	   *pre_tlist = NIL;
	List	   *agg_tlist = NIL;
	List	   *agg_quals = NIL;
	Bitmapset  *attr_refs = NULL;
	bool		has_numeric = false;
	bool		has_varlena = false;
	ListCell   *cell;
	int			i;

	/* In case of sort-aggregate, it has an underlying Sort node on top
	 * of the scan node. GpuPreAgg shall be injected under the Sort node
	 * to reduce burden of CPU sorting.
	 */
	if (IsA(outer_plan, Sort))
		outer_plan = outerPlan(outer_plan);

	/* Head of target-list keeps original order not to adjust expression 
	 * nodes in the Agg (and Sort if exists) node, but replaced to NULL
	 * except for group-by key because all the non-key variables have to
	 * be partial calculation result.
	 */
	i = 0;
	foreach (cell, outer_plan->targetlist)
	{
		TargetEntry	*tle = lfirst(cell);
		TargetEntry *tle_new;
		Oid			type_oid = exprType((Node *) tle->expr);
		int32		type_mod = exprTypmod((Node *) tle->expr);
		Oid			type_coll = exprCollation((Node *) tle->expr);
		char	   *resname = (tle->resname ? pstrdup(tle->resname) : NULL);

		Assert(IsA(tle, TargetEntry));
		for (i=0; i < agg->numCols; i++)
		{
			devtype_info   *dtype;
			Expr		   *varnode;

			if (tle->resno != agg->grpColIdx[i])
				continue;

			/* grouping key must be a supported data type */
			dtype = pgstrom_devtype_lookup(type_oid);
			if (!dtype)
				return false;
			/* grouping key type must have equality function on device */
			if (!OidIsValid(dtype->type_eqfunc) ||
				!pgstrom_devfunc_lookup(dtype->type_eqfunc, type_coll))
				return false;

			/* check types that needs special treatment */
			if (type_oid == NUMERICOID)
				has_numeric = true;
			else if ((dtype->type_flags & DEVTYPE_IS_VARLENA) != 0)
				has_varlena = true;

			varnode = (Expr *) makeVar(INDEX_VAR,
									   tle->resno,
									   type_oid,
									   type_mod,
									   type_coll,
									   0);
			tle_new = makeTargetEntry(varnode,
									  list_length(pre_tlist) + 1,
									  resname,
									  tle->resjunk);
			pre_tlist = lappend(pre_tlist, tle_new);
			attr_refs = bms_add_member(attr_refs, tle->resno -
									   FirstLowInvalidHeapAttributeNumber);
			break;
		}
		/* if not a grouping key, NULL is set instead */
		if (i == agg->numCols)
		{
			Const  *cnst = makeNullConst(type_oid, type_mod, type_coll);

			tle_new = makeTargetEntry((Expr *) cnst,
									  list_length(pre_tlist) + 1,
									  resname,
									  tle->resjunk);
			pre_tlist = lappend(pre_tlist, tle_new);
		}
	}

	/* On the next, replace aggregate functions in tlist of Agg node
	 * according to the aggfunc_catalog[] definition.
	 */
	memset(&context, 0, sizeof(gpupreagg_rewrite_context));
	context.agg = agg;
	context.pre_tlist = pre_tlist;
	context.attr_refs = attr_refs;
	context.extra_flags = *p_extra_flags;

	foreach (cell, agg->plan.targetlist)
	{
		TargetEntry	   *oldtle = lfirst(cell);
		TargetEntry	   *newtle = flatCopyTargetEntry(oldtle);
		Oid				type_oid;

		newtle->expr = (Expr *)gpupreagg_rewrite_mutator((Node *)oldtle->expr,
														 &context);
		if (context.gpupreagg_invalid)
			return false;
		type_oid = exprType((Node *)newtle->expr);
		if (type_oid == NUMERICOID)
			has_numeric = true;
		else if (get_typlen(type_oid) < 0)
			has_varlena = true;
		agg_tlist = lappend(agg_tlist, newtle);
	}

	foreach (cell, agg->plan.qual)
	{
		Expr	   *old_expr = lfirst(cell);
		Expr	   *new_expr;
		Oid			type_oid;

		new_expr = (Expr *)gpupreagg_rewrite_mutator((Node *)old_expr,
													 &context);
		if (context.gpupreagg_invalid)
			return false;
		type_oid = exprType((Node *)new_expr);
		if (type_oid == NUMERICOID)
			has_numeric = true;
		else if (get_typlen(type_oid) < 0)
			has_varlena = true;
		agg_quals = lappend(agg_quals, new_expr);
	}
	*p_pre_tlist = context.pre_tlist;
	*p_agg_tlist = agg_tlist;
	*p_agg_quals = agg_quals;
	*p_attr_refs = context.attr_refs;
	*p_extra_flags = context.extra_flags;
	*p_has_numeric = has_numeric;
	*p_has_varlena = has_varlena;
	return true;
}

/*
 * gpupreagg_codegen_qual_eval - code generator of kernel gpupreagg_qual_eval()
 * that check qualifier on individual tuples prior to the job of preagg
 *
 * static bool
 * gpupreagg_qual_eval(__private cl_int *errcode,
 *                     __global kern_parambuf *kparams,
 *                     __global kern_data_store *kds,
 *                     __global kern_data_store *ktoast,
 *                     size_t kds_index);
 */
static char *
gpupreagg_codegen_qual_eval(CustomScan *cscan, GpuPreAggInfo *gpa_info,
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
        "gpupreagg_qual_eval(cl_int *errcode,\n"
        "                    kern_parambuf *kparams,\n"
        "                    kern_data_store *kds,\n"
        "                    kern_data_store *ktoast,\n"
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
		char   *expr_code
			= pgstrom_codegen_expression((Node *)gpa_info->outer_quals,
										 context);
		Assert(expr_code != NULL);

		appendStringInfo(
			&str,
			"%s%s\n"
			"  return EVAL(%s);\n",
			pgstrom_codegen_param_declarations(context),
			pgstrom_codegen_var_declarations(context),
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
 * gpupreagg_hashvalue(__private cl_int *errcode,
 *                     __local cl_uint *crc32_table,
 *                     __global kern_data_store *kds,
 *                     __global kern_data_store *ktoast,
 *                     size_t kds_index);
 */
static char *
gpupreagg_codegen_hashvalue(CustomScan *cscan, GpuPreAggInfo *gpa_info,
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
					 "gpupreagg_hashvalue(cl_int *errcode,\n"
					 "                    cl_uint *crc32_table,\n"
					 "                    kern_data_store *kds,\n"
					 "                    kern_data_store *ktoast,\n"
					 "                    size_t kds_index)\n"
					 "{\n");
	appendStringInfo(&body,
					 "  cl_uint hashval;\n"
					 "\n"
					 "  INIT_CRC32C(hashval);\n");

	for (i=0; i < gpa_info->numCols; i++)
	{
		TargetEntry	   *tle;
		AttrNumber		resno = gpa_info->grpColIdx[i];
		devtype_info   *dtype;
		Var			   *var;

		tle = get_tle_by_resno(cscan->scan.plan.targetlist, resno);
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
		appendStringInfo(&decl,
						 "  pg_%s_t keyval_%u"
						 " = pg_%s_vref(kds,ktoast,errcode,%u,kds_index);\n",
						 dtype->type_name, resno,
						 dtype->type_name, resno - 1);

		/* crc32 computing */
		appendStringInfo(
			&body,
			"  hashval = pg_%s_comp_crc32(crc32_table, hashval, keyval_%u);\n",
			dtype->type_name, resno);
	}
	/* no constants should be appear */
	Assert(bms_is_empty(context->param_refs));

	appendStringInfo(&body,
					 "  FIN_CRC32C(hashval);\n");
	appendStringInfo(&decl,
					 "%s\n"
					 "  return hashval;\n"
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
 * gpupreagg_keymatch(__private cl_int *errcode,
 *                    __global kern_data_store *kds,
 *                    __global kern_data_store *ktoast,
 *                    size_t x_index,
 *                    size_t y_index);
 */
static char *
gpupreagg_codegen_keycomp(CustomScan *cscan, GpuPreAggInfo *gpa_info,
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

		tle = get_tle_by_resno(cscan->scan.plan.targetlist, resno);
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
			"  xkeyval_%u = pg_%s_vref(kds,ktoast,errcode,%u,x_index);\n"
			"  ykeyval_%u = pg_%s_vref(kds,ktoast,errcode,%u,y_index);\n"
			"  if (!xkeyval_%u.isnull && !ykeyval_%u.isnull)\n"
			"  {\n"
			"    if (!EVAL(pgfn_%s(errcode, xkeyval_%u, ykeyval_%u)))\n"
			"      return false;\n"
			"  }\n"
			"  else if ((xkeyval_%u.isnull  && !ykeyval_%u.isnull) ||"
			"           (!xkeyval_%u.isnull &&  ykeyval_%u.isnull))\n"
			"      return false;\n",
			resno, dtype->type_name, resno - 1,
			resno, dtype->type_name, resno - 1,
			resno, resno,
			dfunc->func_alias, resno, resno,
			resno, resno,
			resno, resno);
	}
	/* add parameters, if referenced */
	if (context->param_refs)
	{
		char	   *params_decl
			= pgstrom_codegen_param_declarations(context);

		appendStringInfo(&decl, "%s", params_decl);
		pfree(params_decl);
		bms_free(context->param_refs);
	}

	/* make a whole key-compare function */
	appendStringInfo(&str,
					 "STATIC_FUNCTION(cl_bool)\n"
					 "gpupreagg_keymatch(int *errcode,\n"
					 "                   kern_data_store *kds,\n"
					 "                   kern_data_store *ktoast,\n"
					 "                   size_t x_index,\n"
					 "                   size_t y_index)\n"
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
 * gpupreagg_local_calc(__private cl_int *errcode,
 *                      cl_int attnum,
 *                      __local pagg_datum *accum,
 *                      __local pagg_datum *newval);
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
		case TIMEOID:
		case TIMESTAMPOID:
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
gpupreagg_codegen_aggcalc(CustomScan *cscan, GpuPreAggInfo *gpa_info,
						  AggCalcMode_t mode, codegen_context *context)
{
	Oid				namespace_oid = get_namespace_oid("pgstrom", false);
	StringInfoData	body;
	const char	   *aggcalc_class;
	const char	   *aggcalc_args;
	ListCell	   *cell;

	initStringInfo(&body);
	switch(mode) {
	case AGGCALC_LOCAL_REDUCTION:
		appendStringInfo(
			&body,
			"STATIC_FUNCTION(void)\n"
			"gpupreagg_local_calc(cl_int *errcode,\n"
			"                     cl_int attnum,\n"
			"                     pagg_datum *accum,\n"
			"                     pagg_datum *newval)\n"
			"{\n");
		aggcalc_class = "LOCAL";
        aggcalc_args = "errcode,accum,newval";
		break;
	case AGGCALC_GLOBAL_REDUCTION:
		appendStringInfo(
			&body,
			"STATIC_FUNCTION(void)\n"
			"gpupreagg_global_calc(cl_int *errcode,\n"
			"                      cl_int attnum,\n"
			"                      kern_data_store *kds,\n"
			"                      kern_data_store *ktoast,\n"
			"                      size_t accum_index,\n"
			"                      size_t newval_index)\n"
			"{\n"
			"  char    *accum_isnull;\n"
			"  Datum   *accum_value;\n"
			"  char     new_isnull;\n"
			"  Datum    new_value;\n"
			"\n"
			"  if (kds->format != KDS_FORMAT_SLOT)\n"
			"  {\n"
			"    STROM_SET_ERROR(errcode,StromError_SanityCheckViolation);\n"
			"    return;\n"
			"  }\n"
			"  accum_isnull = KERN_DATA_STORE_ISNULL(kds,accum_index) + attnum;\n"
			"  accum_value = KERN_DATA_STORE_VALUES(kds,accum_index) + attnum;\n"
			"  new_isnull = *(KERN_DATA_STORE_ISNULL(kds,newval_index) + attnum);\n"
			"  new_value = *(KERN_DATA_STORE_VALUES(kds,newval_index) + attnum);\n"
			"\n");
		aggcalc_class = "GLOBAL";
		aggcalc_args = "errcode,accum_isnull,accum_value,new_isnull,new_value";
		break;
	case AGGCALC_NOGROUP_REDUCTION:
		appendStringInfo(
			&body,
			"STATIC_FUNCTION(void)\n"
			"gpupreagg_nogroup_calc(cl_int *errcode,\n"
			"                       cl_int attnum,\n"
			"                       pagg_datum *accum,\n"
			"                       pagg_datum *newval)\n"
			"{\n");
		aggcalc_class = "NOGROUP";
        aggcalc_args = "errcode,accum,newval";
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
	foreach (cell, cscan->scan.plan.targetlist)
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
 * static void
 * gpupreagg_projection(__private cl_int *errcode,
 *                      __global kern_data_store *kds_in,
 *                      __global kern_data_store *kds_src,
 *                      size_t kds_index);
 */
typedef struct
{
	codegen_context *context;
	const char	   *kds_label;
	const char	   *ktoast_label;
	const char	   *rowidx_label;
	bool			use_temp_int2;
	bool			use_temp_int4;
	bool			use_temp_int8;
	bool			use_temp_float4;
	bool			use_temp_float8x;
	bool			use_temp_float8y;
	bool			use_temp_numeric;
	TargetEntry	   *tle;
} codegen_projection_context;

static void
gpupreagg_codegen_projection_nrows(StringInfo body, FuncExpr *func,
								   codegen_projection_context *pc)
{
	devtype_info   *dtype;
	ListCell	   *cell;

	dtype = pgstrom_devtype_lookup_and_track(INT4OID, pc->context);
	if (!dtype)
		elog(ERROR, "device type lookup failed: %u", INT4OID);

	pc->use_temp_int4 = true;
	if (list_length(func->args) > 0)
	{
		appendStringInfo(body, "  if (");
		foreach (cell, func->args)
		{
			if (cell != list_head(func->args))
				appendStringInfo(body,
								 " &&\n"
								 "      ");
			appendStringInfo(body, "EVAL(%s)",
							 pgstrom_codegen_expression(lfirst(cell),
														pc->context));
		}
		appendStringInfo(body,
						 ")\n"
						 "  {\n"
						 "    temp_int4.isnull = false;\n"
						 "    temp_int4.value = 1;\n"
						 "  }\n"
						 "  else\n"
						 "  {\n"
						 "    temp_int4.isnull = true;\n"
						 "    temp_int4.value = 0;\n"
						 "  }\n");
	}
	else
		appendStringInfo(body,
						 "  temp_int4.isnull = false;\n"
						 "  temp_int4.value = 1;\n");
	appendStringInfo(body,
					 "  pg_%s_vstore(%s,%s,errcode,%u,%s,temp_int4);\n",
					 dtype->type_name,
					 pc->kds_label,
					 pc->ktoast_label,
					 pc->tle->resno - 1,
					 pc->rowidx_label);
}

static void
gpupreagg_codegen_projection_misc(StringInfo body, FuncExpr *func,
								  const char *func_name,
								  codegen_projection_context *pc)
{
	/* Store the original value as-is. If clause is conditional and
	 * false, NULL shall be set. Even if NULL, value fields MUST have
	 * reasonable initial value to the later atomic operation.
	 * In case of PMIN(), NULL takes possible maximum number.
	 */
	Node		   *clause = linitial(func->args);
	Oid				type_oid = exprType(clause);
	devtype_info   *dtype;
	const char	   *temp_val;
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
		case DATEOID:
			pc->use_temp_int4 = true;
			temp_val = "temp_int4";
			max_const = "INT_MAX";
			min_const = "INT_MIN";
			zero_const = "0";
			break;

		case INT8OID:
		case TIMEOID:
		case TIMESTAMPOID:
			pc->use_temp_int8 = true;
            temp_val = "temp_int8";
            max_const = "LONG_MAX";
            min_const = "LONG_MIN";
            zero_const = "0";
			break;

		case FLOAT4OID:
			pc->use_temp_float4 = true;
			temp_val = "temp_float4";
			max_const = "FLT_MAX";
			min_const = "-FLT_MAX";
			zero_const = "0.0";
			break;

		case FLOAT8OID:
			pc->use_temp_float8x = true;
			temp_val = "temp_float8x";
			max_const = "DBL_MAX";
			min_const = "-DBL_MAX";
			zero_const = "0.0";
			break;

		case NUMERICOID:
			pc->use_temp_numeric = true;
			temp_val = "temp_numeric";
			max_const = "PG_NUMERIC_MAX";
			min_const = "PG_NUMERIC_MIN";
			zero_const = "PG_NUMERIC_ZERO";
			break;

		default:
			elog(ERROR, "Bug? device type %s is not expected",
				 format_type_be(type_oid));
	}
	dtype = pgstrom_devtype_lookup_and_track(type_oid, pc->context);
	if (!dtype)
		elog(ERROR, "device type lookup failed: %u", type_oid);

	appendStringInfo(body,
					 "  %s = %s;\n"
					 "  if (%s.isnull)\n"
					 "    %s.value = ",
					 temp_val,
					 pgstrom_codegen_expression(clause, pc->context),
					 temp_val,
					 temp_val);
	if (strcmp(func_name, "pmin") == 0)
		appendStringInfo(body, "%s;\n", max_const);
	else if (strcmp(func_name, "pmax") == 0)
		appendStringInfo(body, "%s;\n", min_const);
	else if (strcmp(func_name, "psum") == 0)
		appendStringInfo(body, "%s;\n", zero_const);
	else
		elog(ERROR, "unexpected partial aggregate function: %s", func_name);

	appendStringInfo(body,
					 "  pg_%s_vstore(%s,%s,errcode,%u,%s,%s);\n",
					 dtype->type_name,
					 pc->kds_label,
					 pc->ktoast_label,
					 pc->tle->resno - 1,
					 pc->rowidx_label,
					 temp_val);
}

static void
gpupreagg_codegen_projection_psum_x2(StringInfo body, FuncExpr *func,
									 codegen_projection_context *pc)
{
	Node		   *clause = linitial(func->args);
	devtype_info   *dtype;
	devfunc_info   *dfunc;
	const char	   *temp_label;
	const char	   *zero_label;

	if (exprType(clause) == FLOAT8OID)
	{
		pc->use_temp_float8x = true;
		dtype = pgstrom_devtype_lookup_and_track(FLOAT8OID, pc->context);
		if (!dtype)
			elog(ERROR, "device type lookup failed: %u", FLOAT8OID);
		dfunc = pgstrom_devfunc_lookup_and_track(F_FLOAT8MUL,
												 InvalidOid,
												 pc->context);
		if (!dtype)
			elog(ERROR, "device function lookup failed: %u", F_FLOAT8MUL);
		temp_label = "temp_float8x";
		zero_label = "0.0";
	}
	else if (exprType(clause) == NUMERICOID)
	{
		pc->use_temp_numeric = true;
		dtype = pgstrom_devtype_lookup_and_track(NUMERICOID, pc->context);
		if (!dtype)
			elog(ERROR, "device type lookup failed: %u", NUMERICOID);
		dfunc = pgstrom_devfunc_lookup_and_track(F_NUMERIC_MUL,
												 InvalidOid,
												 pc->context);
		if (!dtype)
			elog(ERROR, "device function lookup failed: %u", F_NUMERIC_MUL);
		temp_label = "temp_numeric";
		zero_label = "PG_NUMERIC_ZERO";
	}
	else
		elog(ERROR, "Bug? psum_x2 expect float8 or numeric");

	appendStringInfo(
		body,
		"  %s = %s;\n"
		"  if (%s.isnull)\n"
		"    %s.value = %s;\n"
		"  else\n"
		"    %s = pgfn_%s(errcode, %s, %s);\n"
		"  pg_%s_vstore(%s,%s,errcode,%u,%s,%s);\n",
		temp_label,
        pgstrom_codegen_expression(clause, pc->context),
		temp_label,
		temp_label,
		zero_label,
		temp_label,
		dfunc->func_alias,
		temp_label,
		temp_label,
		dtype->type_name,
		pc->kds_label,
        pc->ktoast_label,
        pc->tle->resno - 1,
        pc->rowidx_label,
		temp_label);
}

static void
gpupreagg_codegen_projection_corr(StringInfo body, FuncExpr *func,
								  const char *func_name,
								  codegen_projection_context *pc)
{
	devfunc_info   *dfunc;
	devtype_info   *dtype;
	Node		   *filter = linitial(func->args);
	Node		   *x_clause = lsecond(func->args);
	Node		   *y_clause = lthird(func->args);

	pc->use_temp_float8x = true;
	pc->use_temp_float8y = true;

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
		pgstrom_codegen_expression(x_clause, pc->context),
		pgstrom_codegen_expression(y_clause, pc->context));
	appendStringInfo(
		body,
		"  if (temp_float8x.isnull ||\n"
		"      temp_float8y.isnull");
	if (filter)
		appendStringInfo(
			body,
			" ||\n"
			"      !EVAL(%s)",
			pgstrom_codegen_expression(filter, pc->context));

	appendStringInfo(
		body,
		")\n"
		"  {\n"
		"    temp_float8x.isnull = true;\n"
		"    temp_float8x.value = 0.0;\n"
		"  }\n");

	/* initial value according to the function */
	if (strcmp(func_name, "pcov_y") == 0)
	{
		appendStringInfo(
			body,
			"  else\n"
			"    temp_float8x = temp_float8y;\n");
	}
	else if (strcmp(func_name, "pcov_x2") == 0)
	{
		dfunc = pgstrom_devfunc_lookup_and_track(F_FLOAT8MUL,
												 InvalidOid,
												 pc->context);
		appendStringInfo(
			body,
			"  else\n"
			"    temp_float8x = pgfn_%s(errcode,\n"
			"                           temp_float8x,\n"
			"                           temp_float8x);\n",
			dfunc->func_name);
	}
	else if (strcmp(func_name, "pcov_y2") == 0)
	{
		dfunc = pgstrom_devfunc_lookup_and_track(F_FLOAT8MUL,
												 InvalidOid,
												 pc->context);
		appendStringInfo(
			body,
			"  else\n"
			"    temp_float8x = pgfn_%s(errcode,\n"
			"                           temp_float8y,\n"
			"                           temp_float8y);\n",
			dfunc->func_alias);
	}
	else if (strcmp(func_name, "pcov_xy") == 0)
	{
		dfunc = pgstrom_devfunc_lookup_and_track(F_FLOAT8MUL,
												 InvalidOid,
												 pc->context);
		appendStringInfo(
			body,
			"  else\n"
			"    temp_float8x = pgfn_%s(errcode,\n"
			"                           temp_float8x,\n"
			"                           temp_float8y);\n",
			dfunc->func_alias);
	}
	else if (strcmp(func_name, "pcov_x") != 0)
		elog(ERROR, "unexpected partial covariance function: %s",
			 func_name);

	dtype = pgstrom_devtype_lookup_and_track(FLOAT8OID, pc->context);
	appendStringInfo(
		body,
		"  if (temp_float8x.isnull)\n"
		"    temp_float8x.value = 0.0;\n"
		"  pg_%s_vstore(%s,%s,errcode,%u,%s,temp_float8x);\n",
		dtype->type_name,
		pc->kds_label,
		pc->ktoast_label,
		pc->tle->resno - 1,
		pc->rowidx_label);
}

static char *
gpupreagg_codegen_projection(CustomScan *cscan, GpuPreAggInfo *gpa_info,
							 codegen_context *context)
{
	Oid				namespace_oid = get_namespace_oid("pgstrom", false);
	List		   *targetlist = cscan->scan.plan.targetlist;
	StringInfoData	str;
	StringInfoData	decl1;
	StringInfoData	decl2;
    StringInfoData	body;
	ListCell	   *cell;
	Bitmapset	   *attr_refs = NULL;
	devtype_info   *dtype;
	Plan		   *outer_plan;
	struct varlena *vl_datum;
	Const		   *kparam_0;
	cl_char		   *gpagg_atts;
	Size			length;
	codegen_projection_context pc;

	/* init projection context */
	memset(&pc, 0, sizeof(codegen_projection_context));
	pc.kds_label = "kds_src";
	pc.ktoast_label = "kds_in";
	pc.rowidx_label = "rowidx_out";
	pc.context = context;


	initStringInfo(&str);
	initStringInfo(&decl1);
	initStringInfo(&decl2);
	initStringInfo(&body);
	context->param_refs = NULL;

	/*
	 * construction of kparam_0 - that is an array of cl_char, to inform
	 * kernel which fields are grouping-key, or aggregate function or not.
	 */
	kparam_0 = (Const *) linitial(context->used_params);
	length = VARHDRSZ + sizeof(cl_char) * list_length(targetlist);
	vl_datum = palloc0(length);
	SET_VARSIZE(vl_datum, length);
	kparam_0->constvalue = PointerGetDatum(vl_datum);
	kparam_0->constisnull = false;
	gpagg_atts = (cl_char *)VARDATA(vl_datum);

	foreach (cell, targetlist)
	{
		pc.tle = lfirst(cell);

		if (IsA(pc.tle->expr, Var))
		{
			Var	   *var = (Var *) pc.tle->expr;

			Assert(var->varno == INDEX_VAR);
			Assert(var->varattno > 0);
			attr_refs = bms_add_member(attr_refs, var->varattno -
									   FirstLowInvalidHeapAttributeNumber);
			dtype = pgstrom_devtype_lookup_and_track(var->vartype, context);
			appendStringInfo(
				&body,
				"  /* projection for resource %u */\n"
				"  pg_%s_vstore(%s,%s,errcode,%u,%s,KVAR_%u);\n",
				pc.tle->resno - 1,
				dtype->type_name,
				pc.kds_label,
				pc.ktoast_label,
				pc.tle->resno - 1,
				pc.rowidx_label,
				var->varattno);
			/* track usage of this field */
			gpagg_atts[pc.tle->resno - 1] = GPUPREAGG_FIELD_IS_GROUPKEY;
		}
		else if (IsA(pc.tle->expr, Const))
		{
			/*
			 * Assignmnet of NULL value
			 *
			 * NOTE: we assume constant never appears on both of grouping-
			 * keys and aggregated function (as literal), so we assume
			 * target-entry with Const always represents junk fields.
			 */
			Assert(((Const *) pc.tle->expr)->constisnull);
			appendStringInfo(
				&body,
				"  /* projection for resource %u */\n"
				"  pg_common_vstore(%s,%s,errcode,%u,%s,true);\n",
				pc.tle->resno - 1,
				pc.kds_label,
				pc.ktoast_label,
				pc.tle->resno - 1,
				pc.rowidx_label);
		}
		else if (IsA(pc.tle->expr, FuncExpr))
		{
			FuncExpr   *func = (FuncExpr *) pc.tle->expr;
			const char *func_name;

			appendStringInfo(&body,
							 "  /* projection for resource %u */\n",
							 pc.tle->resno - 1);
			if (namespace_oid != get_func_namespace(func->funcid))
				elog(ERROR, "Bug? unexpected FuncExpr: %s",
					 nodeToString(func));

			pull_varattnos((Node *)func, INDEX_VAR, &attr_refs);

			func_name = get_func_name(func->funcid);
			if (strcmp(func_name, "nrows") == 0)
				gpupreagg_codegen_projection_nrows(&body, func, &pc);
			else if (strcmp(func_name, "pmax") == 0 ||
					 strcmp(func_name, "pmin") == 0 ||
					 strcmp(func_name, "psum") == 0)
				gpupreagg_codegen_projection_misc(&body, func, func_name, &pc);
			else if (strcmp(func_name, "psum_x2") == 0)
				gpupreagg_codegen_projection_psum_x2(&body, func, &pc);
			else if (strcmp(func_name, "pcov_x") == 0 ||
					 strcmp(func_name, "pcov_y") == 0 ||
					 strcmp(func_name, "pcov_x2") == 0 ||
					 strcmp(func_name, "pcov_y2") == 0 ||
					 strcmp(func_name, "pcov_xy") == 0)
				gpupreagg_codegen_projection_corr(&body, func, func_name, &pc);
			else
				elog(ERROR, "Bug? unexpected partial aggregate function: %s",
					 func_name);
			/* track usage of this field */
			gpagg_atts[pc.tle->resno - 1] = GPUPREAGG_FIELD_IS_AGGFUNC;
		}
		else
			elog(ERROR, "bug? unexpected node type: %s",
				 nodeToString(pc.tle->expr));
	}

	/*
	 * Declaration of variables
	 */
	outer_plan = outerPlan(cscan);
	if (gpa_info->outer_bulkload)
	{
		const char *saved_kds_label = context->kds_label;
		const char *saved_kds_index_label = context->kds_index_label;
		char	   *temp;

		context->kds_label = "kds_in";
		context->kds_index_label = "rowidx_in";

		temp = pgstrom_codegen_bulk_var_declarations(context,
													 outer_plan,
													 attr_refs);
		appendStringInfo(&decl1, "%s", temp);
		pfree(temp);

		context->kds_label = saved_kds_label;
		context->kds_index_label = saved_kds_index_label;
	}
	else
	{
		foreach (cell, outer_plan->targetlist)
		{
			TargetEntry	*tle = lfirst(cell);
			int		x = tle->resno - FirstLowInvalidHeapAttributeNumber;
			Oid		type_oid;

			if (!bms_is_member(x, attr_refs))
				continue;
			type_oid = exprType((Node *) tle->expr);
			dtype = pgstrom_devtype_lookup_and_track(type_oid, context);
			appendStringInfo(
				&decl1,
				"  pg_%s_t KVAR_%u"
				" = pg_%s_vref(kds_in,ktoast,errcode,%u,rowidx_in);\n",
				dtype->type_name,
				tle->resno,
				dtype->type_name,
				tle->resno - 1);
		}
	}

	/* declaration of parameter reference */
	if (context->param_refs)
	{
		char	   *params_decl
			= pgstrom_codegen_param_declarations(context);

		appendStringInfo(&decl2, "%s", params_decl);
		pfree(params_decl);
		bms_free(context->param_refs);
	}

	/* declaration of other temp variables */
	if (pc.use_temp_int2)
		appendStringInfo(&decl1, "  pg_int2_t temp_int2;\n");
	if (pc.use_temp_int4)
		appendStringInfo(&decl1, "  pg_int4_t temp_int4;\n");
	if (pc.use_temp_int8)
		appendStringInfo(&decl1, "  pg_int8_t temp_int8;\n");
	if (pc.use_temp_float4)
		appendStringInfo(&decl1, "  pg_float4_t temp_float4;\n");
	if (pc.use_temp_float8x)
		appendStringInfo(&decl1, "  pg_float8_t temp_float8x;\n");
	if (pc.use_temp_float8y)
		appendStringInfo(&decl1, "  pg_float8_t temp_float8y;\n");
	if (pc.use_temp_numeric)
		appendStringInfo(&decl1, "  pg_numeric_t temp_numeric;\n");

	appendStringInfo(
		&str,
		"STATIC_FUNCTION(void)\n"
		"gpupreagg_projection(cl_int *errcode,\n"
		"                     kern_parambuf *kparams,\n"
		"                     kern_data_store *kds_in,\n"
		"                     kern_data_store *kds_src,\n"
		"                     kern_data_store *ktoast,\n"
		"                     size_t rowidx_in, size_t rowidx_out)\n"
		"{\n"
		"%s"
		"%s"
		"\n"
		"%s"
		"}\n",
		decl2.data,
		decl1.data,
		body.data);

	return str.data;
}

static char *
gpupreagg_codegen(CustomScan *cscan, GpuPreAggInfo *gpa_info,
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
	context->type_defs = list_make1(pgstrom_devtype_lookup(BYTEAOID));

	/* generate a qual evaluation function */
	fn_qualeval = gpupreagg_codegen_qual_eval(cscan, gpa_info, context);
	/* generate gpupreagg_hashvalue function */
	fn_hashvalue = gpupreagg_codegen_hashvalue(cscan, gpa_info, context);
	/* generate a key comparison function */
	fn_keycomp = gpupreagg_codegen_keycomp(cscan, gpa_info, context);
	/* generate a gpupreagg_local_calc function */
	fn_local_calc = gpupreagg_codegen_aggcalc(cscan, gpa_info,
											  AGGCALC_LOCAL_REDUCTION,
											  context);
	/* generate a gpupreagg_global_calc function */
	fn_global_calc = gpupreagg_codegen_aggcalc(cscan, gpa_info,
											   AGGCALC_GLOBAL_REDUCTION,
											   context);
	/* generate a gpupreagg_global_calc function */
	fn_nogroup_calc = gpupreagg_codegen_aggcalc(cscan, gpa_info,
												AGGCALC_NOGROUP_REDUCTION,
												context);
	/* generate an initial data loading function */
	fn_projection = gpupreagg_codegen_projection(cscan, gpa_info, context);

	/* OK, add type/function declarations */
	initStringInfo(&str);
	appendStringInfo(&str,
					 "%s\n"		/* function declarations */
					 "%s\n"		/* gpupreagg_qual_eval() */
					 "%s\n"		/* gpupreagg_hashvalue() */
					 "%s\n"		/* gpupreagg_keycomp() */
					 "%s\n"		/* gpupreagg_local_calc() */
					 "%s\n"		/* gpupreagg_global_calc() */
					 "%s\n"		/* gpupreagg_nogroup_calc() */
					 "%s\n",	/* gpupreagg_projection() */
					 pgstrom_codegen_func_declarations(context),
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
	Plan		   *alter_node;
	List		   *pre_tlist = NIL;
	List		   *agg_quals = NIL;
	List		   *agg_tlist = NIL;
	Bitmapset	   *attr_refs = NULL;
	List		   *outer_quals = NIL;
	bool			has_numeric = false;
	bool			has_varlena = false;
	ListCell	   *cell;
	AggStrategy		new_agg_strategy;
	AggClauseCosts	agg_clause_costs;
	Plan			newcost_agg;
	Plan			newcost_sort;
	Plan			newcost_gpreagg;
	int				extra_flags = DEVKERNEL_NEEDS_GPUPREAGG;
	bool			outer_bulkload = false;
	codegen_context context;

	/* nothing to do, if feature is turned off */
	if (!pgstrom_enabled() || !enable_gpupreagg)
		return;

	/* Try to construct target-list of both Agg and GpuPreAgg node.
	 * If unavailable to construct, it indicates this aggregation
	 * does not support partial aggregation.
	 */
	if (!gpupreagg_rewrite_expr(agg,
								&agg_tlist,
								&agg_quals,
								&pre_tlist,
								&attr_refs,
								&extra_flags,
								&has_numeric,
								&has_varlena))
		return;

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
		alter_node = pgstrom_try_replace_plannode(outer_node,
												  pstmt->rtable,
												  &outer_quals);
		if (alter_node)
			outer_node = alter_node;
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
		alter_node = pgstrom_try_replace_plannode(outer_node,
												  pstmt->rtable,
												  &outer_quals);
		if (alter_node)
			outer_node = alter_node;
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
		alter_node = pgstrom_try_replace_plannode(outer_node,
												  pstmt->rtable,
												  &outer_quals);
		if (alter_node)
			outer_node = alter_node;
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

	/* check availability of outer bulkload */
	if (IsA(outer_node, CustomScan))
	{
		int		custom_flags = ((CustomScan *) outer_node)->flags;

		if ((custom_flags & CUSTOMPATH_SUPPORT_BULKLOAD) != 0)
			outer_bulkload = true;
	}

	/*
	 * Estimate the cost if GpuPreAgg would be injected, and determine
	 * which plan is cheaper, unless pg_strom.debug_force_gpupreagg is
	 * not turned on.
	 */
	cost_gpupreagg(agg, sort_node, outer_node,
				   new_agg_strategy,
				   pre_tlist,
				   &agg_clause_costs,
				   &newcost_agg,
				   &newcost_sort,
				   &newcost_gpreagg);
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

	/*
	 * OK, let's construct GpuPreAgg node then inject it.
	 */
	cscan = makeNode(CustomScan);
	cscan->scan.plan.startup_cost = newcost_gpreagg.startup_cost;
	cscan->scan.plan.total_cost   = newcost_gpreagg.total_cost;
	cscan->scan.plan.plan_rows    = newcost_gpreagg.plan_rows;
	cscan->scan.plan.plan_width   = newcost_gpreagg.plan_width;
	cscan->scan.plan.targetlist   = pre_tlist;
	cscan->scan.plan.qual         = NIL;
	cscan->scan.scanrelid         = 0;
	cscan->flags                  = 0;
	cscan->custom_ps_tlist        = NIL;
	cscan->custom_relids          = NULL;
	cscan->methods                = &gpupreagg_scan_methods;
	foreach (cell, outer_node->targetlist)
	{
		TargetEntry *tle = lfirst(cell);
		TargetEntry	*ps_tle;
		Var		   *varnode;

		varnode = makeVar(OUTER_VAR,
						  tle->resno,
						  exprType((Node *) tle->expr),
						  exprTypmod((Node *) tle->expr),
						  exprCollation((Node *) tle->expr),
						  0);
		ps_tle = makeTargetEntry((Expr *) varnode,
								 list_length(cscan->custom_ps_tlist) + 1,
								 tle->resname ? pstrdup(tle->resname) : NULL,
								 false);
		cscan->custom_ps_tlist = lappend(cscan->custom_ps_tlist, ps_tle);
	}
	if (IsA(outer_node, CustomScan))
		((CustomScan *) outer_node)->flags |= CUSTOMPATH_PREFERE_ROW_FORMAT;
	outerPlan(cscan)		= outer_node;

	/* also set up private information */
	memset(&gpa_info, 0, sizeof(GpuPreAggInfo));
	gpa_info.numCols        = agg->numCols;
	gpa_info.grpColIdx      = pmemcpy(agg->grpColIdx,
									  sizeof(AttrNumber) * agg->numCols);
	gpa_info.outer_quals    = outer_quals;
	gpa_info.outer_bulkload = outer_bulkload;
	gpa_info.num_groups     = Max(agg->plan.plan_rows, 1.0);
	gpa_info.outer_quals    = outer_quals;

	/*
	 * construction of the kernel code according to the target-list
	 * and qualifiers (pulled-up from outer plan).
	 */
	pgstrom_init_codegen_context(&context);
	gpa_info.kern_source = gpupreagg_codegen(cscan, &gpa_info, &context);
	gpa_info.extra_flags = extra_flags | context.extra_flags;
	gpa_info.used_params = context.used_params;
	pull_varattnos((Node *)context.used_vars,
				   INDEX_VAR,
				   &gpa_info.outer_attrefs);
	foreach (cell, cscan->scan.plan.targetlist)
	{
		TargetEntry	   *tle = lfirst(cell);

		if (IsA(tle->expr, Const))
			continue;
		gpa_info.tlist_attrefs =
			bms_add_member(gpa_info.tlist_attrefs, tle->resno -
						   FirstLowInvalidHeapAttributeNumber);
	}
	gpa_info.has_numeric = has_numeric;
	gpa_info.has_varlena = has_varlena;
	form_gpupreagg_info(cscan, &gpa_info);

	/*
	 * OK, let's inject GpuPreAgg and update the cost values.
	 */
	if (!sort_node)
		outerPlan(agg) = &cscan->scan.plan;
	else
	{
		sort_node->plan.startup_cost = newcost_sort.startup_cost;
		sort_node->plan.total_cost   = newcost_sort.total_cost;
		sort_node->plan.plan_rows    = newcost_sort.plan_rows;
		sort_node->plan.plan_width   = newcost_sort.plan_width;
		sort_node->plan.targetlist   = expr_fixup_varno(pre_tlist,
														INDEX_VAR,
														OUTER_VAR);
		outerPlan(sort_node) = &cscan->scan.plan;
	}
	agg->plan.startup_cost = newcost_agg.startup_cost;
	agg->plan.total_cost   = newcost_agg.total_cost;
	agg->plan.plan_rows    = newcost_agg.plan_rows;
	agg->plan.plan_width   = newcost_agg.plan_width;
	agg->plan.targetlist = agg_tlist;
	agg->plan.qual = agg_quals;
	agg->aggstrategy = new_agg_strategy;
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
	GpuContext	   *gcontext = pgstrom_get_gpucontext();
	GpuPreAggState *gpas = MemoryContextAllocZero(gcontext->memcxt,
												  sizeof(GpuPreAggState));
	/* Set tag and executor callbacks */
	NodeSetTag(gpas, T_CustomScanState);
	gpas->gts.css.flags = cscan->flags;
	gpas->gts.css.methods = &gpupreagg_exec_methods;
	/* GpuTaskState setup */
	pgstrom_init_gputaskstate(gcontext, &gpas->gts);
	gpas->gts.cb_task_process = gpupreagg_task_process;
	gpas->gts.cb_task_complete = gpupreagg_task_complete;
	gpas->gts.cb_task_fallback = gpupreagg_task_fallback;
	gpas->gts.cb_task_release = gpupreagg_task_release;
	gpas->gts.cb_next_chunk = gpupreagg_next_chunk;
	gpas->gts.cb_next_tuple = gpupreagg_next_tuple;
	gpas->gts.cb_cleanup = NULL;

	return (Node *) gpas;
}

static void
gpupreagg_begin(CustomScanState *node, EState *estate, int eflags)
{
	GpuPreAggState *gpas = (GpuPreAggState *) node;
	PlanState	   *ps = &node->ss.ps;
	CustomScan	   *cscan = (CustomScan *) ps->plan;
	GpuPreAggInfo  *gpa_info = deform_gpupreagg_info(cscan);
	int				outer_width;
	Const		   *kparam_0;

	/*
	 * initialize own child expression
	 */
	gpas->outer_quals = (List *)
		ExecInitExpr((Expr *) gpa_info->outer_quals, ps);

	/*
	 * initialize child node
	 */
	outerPlanState(gpas) = ExecInitNode(outerPlan(cscan), estate, eflags);
	gpas->gts.scan_bulk =
		(!pgstrom_debug_bulkload_enabled ? false : gpa_info->outer_bulkload);
	//gpas->outer_done = false;
	gpas->outer_overflow = NULL;

	outer_width = outerPlanState(gpas)->plan->plan_width;
	gpas->num_groups = gpa_info->num_groups;
	gpas->ntups_per_page =
		((double)(BLCKSZ - MAXALIGN(SizeOfPageHeaderData))) /
		((double)(sizeof(ItemIdData) +
				  sizeof(HeapTupleHeaderData) + outer_width));

	/*
	 * initialize result tuple type and projection info
	 */
	if (gpas->gts.scan_bulk)
	{
		CustomScanState *ocss = (CustomScanState *) outerPlanState(gpas);

		Assert(IsA(ocss, CustomScanState));
		gpas->bulk_proj = ocss->ss.ps.ps_ProjInfo;
		gpas->bulk_slot = ocss->ss.ss_ScanTupleSlot;
	}

	/*
	 * construction of kern_parambuf template; including system param of
	 * GPUPREAGG_FIELD_IS_* array.
	 * NOTE: we don't modify gpreagg->used_params here, so no need to
	 * make a copy.
	 */
	Assert(list_length(gpa_info->used_params) >= 1);
	kparam_0 = (Const *) linitial(gpa_info->used_params);
	Assert(IsA(kparam_0, Const) &&
		   kparam_0->consttype == BYTEAOID &&
		   !kparam_0->constisnull);
	gpas->kparams = pgstrom_create_kern_parambuf(gpa_info->used_params,
												 ps->ps_ExprContext);
	/*
	 * Setting up kernel program and message queue
	 */
	gpas->gts.kern_source = gpa_info->kern_source;
	gpas->gts.extra_flags = gpa_info->extra_flags;
	if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
        pgstrom_preload_cuda_program(&gpas->gts);

	/*
	 * init misc stuff
	 */
	gpas->needs_grouping = (gpa_info->numCols > 0 ? true : false);
	gpas->local_reduction = true;	/* tentative */
	gpas->has_numeric = gpa_info->has_numeric;
	gpas->has_varlena = gpa_info->has_varlena;
	gpas->num_rechecks = 0;
}

static pgstrom_gpupreagg *
gpupreagg_task_create(GpuPreAggState *gpas, pgstrom_data_store *pds_in)
{
	GpuContext		   *gcontext = gpas->gts.gcontext;
	pgstrom_gpupreagg  *gpreagg;
	kern_parambuf	   *kparams;
	kern_resultbuf	   *kresults;
	TupleDesc			tupdesc;
	size_t				nitems = pds_in->kds->nitems;
	Size				required;

	/* allocation of pgtrom_gpupreagg */
	required = (STROMALIGN(offsetof(pgstrom_gpupreagg, kern.kparams) +
						   gpas->kparams->length) +
				STROMALIGN(offsetof(kern_resultbuf, results[nitems])));
	gpreagg = MemoryContextAllocZero(gcontext->memcxt, required);

	/* initialize GpuTask object */
	pgstrom_init_gputask(&gpas->gts, &gpreagg->task);
	gpreagg->needs_fallback = false;
	gpreagg->needs_grouping = gpas->needs_grouping;
	gpreagg->local_reduction = gpas->local_reduction;
	gpreagg->has_varlena = gpas->has_varlena;
	gpreagg->num_groups = gpas->num_groups;
	gpreagg->pds_in = pds_in;

	/* also initialize kern_gpupreagg portion */
	gpreagg->kern.hash_size = nitems;
	memcpy(gpreagg->kern.pg_crc32_table,
		   pg_crc32c_table,
		   sizeof(uint32) * 256);
	/* kern_parambuf */
	kparams = KERN_GPUPREAGG_PARAMBUF(&gpreagg->kern);
	memcpy(kparams, gpas->kparams, gpas->kparams->length);
	/* kern_resultbuf */
	kresults = gpreagg->kresults = KERN_GPUPREAGG_RESULTBUF(&gpreagg->kern);
	kresults->nrels = 1;
	kresults->nrooms = nitems;
	kresults->nitems = 0;
	kresults->errcode = StromError_Success;
	kresults->has_rechecks = false;
	kresults->all_visible = true;

	/*
	 * allocation of the result buffer
	 *
	 * note: GpuPreAgg does not have bulk-load output so, no need to have
	 *       relevant toast buffer.
	 */
	tupdesc = gpas->gts.css.ss.ps.ps_ResultTupleSlot->tts_tupleDescriptor;
	gpreagg->pds_dst = pgstrom_create_data_store_slot(gcontext,
													  tupdesc,
													  nitems,
													  gpas->has_numeric,
													  NULL);
	return gpreagg;
}

static GpuTask *
gpupreagg_next_chunk(GpuTaskState *gts)
{
	GpuContext		   *gcontext = gts->gcontext;
	GpuPreAggState	   *gpas = (GpuPreAggState *) gts;
	PlanState		   *subnode = outerPlanState(gpas);
	pgstrom_data_store *pds = NULL;
	struct timeval		tv1, tv2;

	if (gpas->gts.scan_done)
		return NULL;

	PERFMON_BEGIN(&gts->pfm_accum, &tv1);

	if (!gpas->gts.scan_bulk)
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
					gpas->gts.scan_done = true;
					break;
				}
			}

			if (!pds)
				pds = pgstrom_create_data_store_row(gcontext,
													tupdesc,
													pgstrom_chunk_size(),
													false);
			/* insert a tuple to the data-store */
			if (!pgstrom_data_store_insert_tuple(pds, slot))
			{
				gpas->outer_overflow = slot;
				break;
			}
		}
	}
	else
	{
		/* Load a bunch of records at once */
		pds = BulkExecProcNode(subnode);
		if (!pds)
			gpas->gts.scan_done = true;
	}
	PERFMON_END(&gts->pfm_accum, time_outer_load, &tv1, &tv2);

	if (!pds)
		return NULL;	/* no more tuples to read */

	return (GpuTask *) gpupreagg_task_create(gpas, pds);
}

/*
 * gpupreagg_next_tuple_fallback - a fallback routine if GPU returned
 * StromError_CpuReCheck, to suggest the backend to handle request
 * by itself. A fallback process looks like construction of special
 * partial aggregations that consist of individual rows; so here is
 * no performance benefit once it happen.
 */
static TupleTableSlot *
gpupreagg_next_tuple_fallback(GpuPreAggState *gpas, pgstrom_gpupreagg *gpreagg)
{
	pgstrom_data_store *pds = gpreagg->pds_in;
	TupleTableSlot	   *slot;
	size_t				nitems = pds->kds->nitems;
	cl_uint				row_index;
	HeapTupleData		tuple;

	/* bulk-load uses individual slot; then may have a projection */
	if (!gpas->gts.scan_bulk)
		slot = gpas->gts.css.ss.ss_ScanTupleSlot;
	else
		slot = gpas->bulk_slot;

retry:
	if (gpas->gts.curr_index >= nitems)
		return NULL;
	row_index = gpas->gts.curr_index++;

	/*
	 * Fetch a tuple from the data-store
	 */
	if (!pgstrom_fetch_data_store(slot, pds, row_index, &tuple))
	{
		if (gpas->gts.scan_bulk)
		{
			ExprContext	   *econtext;
			ExprDoneCond	is_done;

			/*
			 * check qualifier being pulled up from the outer scan, if any.
			 * Outer_quals assumes fetched tuple is stored on the
			 * ecxt_scantuple (because it came from relation-scan), we need
			 * to adjust it.
			 */
			if (gpas->outer_quals != NIL)
			{
				econtext = gpas->gts.css.ss.ps.ps_ExprContext;
				econtext->ecxt_scantuple = slot;
				if (!ExecQual(gpas->outer_quals, econtext, false))
					goto retry;
			}

			/*
			 * In case of bulk-load mode, it may take additional projection
			 * because slot has a record type of underlying scan node, thus
			 * we need to translate this record into the form we expected.
			 * If bulk_proj is valid, it implies our expected input record is
			 * incompatible from the record type of underlying scan.
			 */
			if (gpas->bulk_proj)
			{
				econtext = gpas->bulk_proj->pi_exprContext;
				econtext->ecxt_scantuple = slot;
				slot = ExecProject(gpas->bulk_proj, &is_done);
				if (is_done == ExprEndResult)
					goto retry;
			}
		}
	}
	return slot;
}

static TupleTableSlot *
gpupreagg_next_tuple(GpuTaskState *gts)
{
	GpuPreAggState	   *gpas = (GpuPreAggState *) gts;
	pgstrom_gpupreagg  *gpreagg = (pgstrom_gpupreagg *) gpas->gts.curr_task;
	kern_resultbuf	   *kresults = gpreagg->kresults;
	pgstrom_data_store *pds_dst = gpreagg->pds_dst;
	TupleTableSlot	   *slot = NULL;
	HeapTupleData		tuple;
	struct timeval		tv1, tv2;

	Assert(kresults == KERN_GPUPREAGG_RESULTBUF(&gpreagg->kern));

	PERFMON_BEGIN(&gts->pfm_accum, &tv1);
	if (gpreagg->needs_fallback)
		slot = gpupreagg_next_tuple_fallback(gpas, gpreagg);
	else if (gpas->gts.curr_index < kresults->nitems)
	{
		size_t		index = kresults->results[gpas->gts.curr_index++];

		slot = gts->css.ss.ps.ps_ResultTupleSlot;
		if (!pgstrom_fetch_data_store(slot, pds_dst, index, &tuple))
		{
			elog(NOTICE, "Bug? empty slot was specified by kern_resultbuf");
			slot = NULL;
		}
		else if (gpas->has_numeric)
		{
			TupleDesc	tupdesc = slot->tts_tupleDescriptor;
			int			i;

			/*
			 * We have to fixup numeric values from the in-kernel format
			 * to PostgreSQL's internal format.
			 */
			slot_getallattrs(slot);
			for (i=0; i < tupdesc->natts; i++)
			{
				Form_pg_attribute	attr = tupdesc->attrs[i];

				if (attr->atttypid != NUMERICOID || slot->tts_isnull[i])
					continue;	/* no need to fixup */

				slot->tts_values[i] =
					pgstrom_fixup_kernel_numeric(slot->tts_values[i]);
			}
			/* Now we expect GpuPreAgg takes KDS_FORMAT_TUPSLOT for result
			 * buffer, it should not have tts_tuple to be fixed up too.
			 */
			Assert(!slot->tts_tuple);
		}
	}
	PERFMON_END(&gts->pfm_accum, time_materialize, &tv1, &tv2);

	return slot;
}

static TupleTableSlot *
gpupreagg_exec(CustomScanState *node)
{
	return ExecScan(&node->ss,
					(ExecScanAccessMtd) pgstrom_exec_gputask,
					(ExecScanRecheckMtd) pgstrom_recheck_gputask);
}

static void
gpupreagg_end(CustomScanState *node)
{
	GpuPreAggState	   *gpas = (GpuPreAggState *) node;

	/* Debug message if needed */
	if (gpas->num_rechecks > 0)
		elog(NOTICE, "GpuPreAgg: %u chunks were re-checked by CPU",
			 gpas->num_rechecks);

	/* Cleanup and relase any concurrent tasks */
	pgstrom_release_gputaskstate(&gpas->gts);
	/* Clean up subtree */
	ExecEndNode(outerPlanState(node));
}

static void
gpupreagg_rescan(CustomScanState *node)
{
	GpuPreAggState	   *gpas = (GpuPreAggState *) node;

	/* Cleanup and relase any concurrent tasks */
	pgstrom_cleanup_gputaskstate(&gpas->gts);
	/* Rewind the subtree */
	gpas->gts.scan_done = false;
	ExecReScan(outerPlanState(node));
}

static void
gpupreagg_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
	GpuPreAggState *gpas = (GpuPreAggState *) node;
	GpuPreAggInfo  *gpa_info
		= deform_gpupreagg_info((CustomScan *) node->ss.ps.plan);
	const char	   *policy;

	ExplainPropertyText("Bulkload", gpas->gts.scan_bulk ? "On" : "Off", es);
	if (!gpas->needs_grouping)
		policy = "NoGroup";
	else if (gpas->local_reduction)
		policy = "Local + Global";
	else
		policy = "Global";
	ExplainPropertyText("Reduction", policy, es);

	if (gpa_info->outer_quals != NIL)
	{
		show_scan_qual(gpa_info->outer_quals,
					   "Device Filter", &gpas->gts.css.ss.ps, ancestors, es);
		show_instrumentation_count("Rows Removed by Device Fileter",
                                   2, &gpas->gts.css.ss.ps, es);
	}
	pgstrom_explain_custom_flags(&gpas->gts.css, es);
	pgstrom_explain_kernel_source(&gpas->gts, es);
	if (es->analyze && gpas->gts.pfm_accum.enabled)
		pgstrom_explain_perfmon(&gpas->gts.pfm_accum, es);
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
	if (gpreagg->m_kds_in)
		gpuMemFree(&gpreagg->task, gpreagg->m_kds_in);
	if (gpreagg->m_kds_src)
		gpuMemFree(&gpreagg->task, gpreagg->m_kds_src);
	if (gpreagg->m_kds_dst)
		gpuMemFree(&gpreagg->task, gpreagg->m_kds_dst);

	CUDA_EVENT_DESTROY(gpreagg, ev_dma_send_start);
	CUDA_EVENT_DESTROY(gpreagg, ev_dma_send_stop);
	CUDA_EVENT_DESTROY(gpreagg, ev_kern_prep_end);
	CUDA_EVENT_DESTROY(gpreagg, ev_kern_lagg_end);
	CUDA_EVENT_DESTROY(gpreagg, ev_dma_recv_start);
	CUDA_EVENT_DESTROY(gpreagg, ev_dma_recv_stop);

	/* clear the pointers */
	gpreagg->kern_prep = NULL;
	memset(gpreagg->kern_prep_args, 0, sizeof(gpreagg->kern_prep_args));
	gpreagg->kern_lagg = NULL;
	memset(gpreagg->kern_lagg_args, 0, sizeof(gpreagg->kern_lagg_args));
	gpreagg->kern_gagg = NULL;
	memset(gpreagg->kern_gagg_args, 0, sizeof(gpreagg->kern_gagg_args));
	gpreagg->kern_nogrp = NULL;
	memset(gpreagg->kern_nogrp_args, 0, sizeof(gpreagg->kern_nogrp_args));
	gpreagg->kern_fixvar = NULL;
	memset(gpreagg->kern_fixvar_args, 0, sizeof(gpreagg->kern_fixvar_args));
	gpreagg->m_gpreagg = 0UL;
	gpreagg->m_kds_in = 0UL;
	gpreagg->m_kds_src = 0UL;
	gpreagg->m_kds_dst = 0UL;
	gpreagg->m_ghash = 0UL;
}

static void
gpupreagg_task_release(GpuTask *gtask)
{
	pgstrom_gpupreagg  *gpreagg = (pgstrom_gpupreagg *) gtask;

	fprintf(stderr, "gpupreagg_task_release called %p\n", gpreagg);

	/* cleanup cuda resources, if any */
	gpupreagg_cleanup_cuda_resources(gpreagg);

	if (gpreagg->pds_in)
		pgstrom_release_data_store(gpreagg->pds_in);
	if (gpreagg->pds_dst)
		pgstrom_release_data_store(gpreagg->pds_dst);
	pfree(gpreagg);
}

/*
 *
 */
static void
gpupreagg_task_fallback(GpuTask *gtask)
{
	pgstrom_gpupreagg  *gpreagg = (pgstrom_gpupreagg *) gtask;

	gpreagg->needs_fallback = true;
}

/*
 * gpupreagg_task_complete
 */
static bool
gpupreagg_task_complete(GpuTask *gtask)
{
	pgstrom_gpupreagg  *gpreagg = (pgstrom_gpupreagg *) gtask;

	if (gpreagg->task.pfm.enabled)
	{
		CUDA_EVENT_ELAPSED(gpreagg, time_dma_send,
						   ev_dma_send_start,
						   ev_dma_send_stop);
		CUDA_EVENT_ELAPSED(gpreagg, time_kern_prep,
						   ev_dma_send_stop,
						   ev_kern_prep_end);
		if (gpreagg->needs_grouping)
		{
			if (gpreagg->local_reduction)
			{
				CUDA_EVENT_ELAPSED(gpreagg, time_kern_lagg,
								   ev_kern_prep_end,
								   ev_kern_lagg_end);
				CUDA_EVENT_ELAPSED(gpreagg, time_kern_gagg,
								   ev_kern_lagg_end,
								   ev_dma_recv_start);
			}
			else
			{
				CUDA_EVENT_ELAPSED(gpreagg, time_kern_gagg,
                                   ev_kern_prep_end,
								   ev_dma_recv_start);
			}
		}
		else
		{
			CUDA_EVENT_ELAPSED(gpreagg, time_kern_nogrp,
							   ev_kern_prep_end,
							   ev_dma_recv_start);
		}
		CUDA_EVENT_ELAPSED(gpreagg, time_dma_recv,
                           ev_dma_recv_start,
                           ev_dma_recv_stop);
	}
	gpupreagg_cleanup_cuda_resources(gpreagg);	
	return false;
}

/*
 * gpupreagg_task_respond
 */
static void
gpupreagg_task_respond(CUstream stream, CUresult status, void *private)
{
	pgstrom_gpupreagg  *gpreagg = (pgstrom_gpupreagg *) private;
	kern_resultbuf	   *kresults = KERN_GPUPREAGG_RESULTBUF(&gpreagg->kern);
	GpuTaskState	   *gts = gpreagg->task.gts;

	SpinLockAcquire(&gts->lock);
	if (status != CUDA_SUCCESS)
		gpreagg->task.errcode = status;
	else
		gpreagg->task.errcode = kresults->errcode;

	/* remove from the running_tasks list */
	dlist_delete(&gpreagg->task.chain);
	gts->num_running_tasks--;
	/* then, attach it on the completed_tasks list */
	if (gpreagg->task.errcode == StromError_Success)
		dlist_push_tail(&gts->completed_tasks, &gpreagg->task.chain);
	else
		dlist_push_head(&gts->completed_tasks, &gpreagg->task.chain);
	SpinLockRelease(&gts->lock);

	SetLatch(&MyProc->procLatch);
}

/*
 * gpupreagg_task_process
 */
static bool
__gpupreagg_task_process(pgstrom_gpupreagg *gpreagg)
{
	kern_resultbuf	   *kresults = gpreagg->kresults;
	pgstrom_data_store *pds_in = gpreagg->pds_in;
	pgstrom_data_store *pds_dst = gpreagg->pds_dst;
	size_t				nitems = pds_in->kds->nitems;
	size_t				offset;
	size_t				length;
	size_t				grid_size;
	size_t				block_size;
	CUresult			rc;

	/*
	 * Kernel function lookup
	 */
	rc = cuModuleGetFunction(&gpreagg->kern_prep,
							 gpreagg->task.cuda_module,
							 "gpupreagg_preparation");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction : %s", errorText(rc));
	if (gpreagg->needs_grouping)
	{
		if (gpreagg->local_reduction)
		{
			rc = cuModuleGetFunction(&gpreagg->kern_lagg,
									 gpreagg->task.cuda_module,
									 "gpupreagg_local_reduction");
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuModuleGetFunction : %s",
					 errorText(rc));
		}
		rc = cuModuleGetFunction(&gpreagg->kern_gagg,
								 gpreagg->task.cuda_module,
								 "gpupreagg_global_reduction");
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuModuleGetFunction : %s", errorText(rc));
	}
	if (gpreagg->has_varlena)
	{
		rc = cuModuleGetFunction(&gpreagg->kern_fixvar,
								 gpreagg->task.cuda_module,
								 "gpupreagg_fixup_varlena");
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuModuleGetFunction : %s", errorText(rc));
	}

	/*
	 * Allocation of device memory
	 */
	length = KERN_GPUPREAGG_LENGTH(&gpreagg->kern, nitems);
	gpreagg->m_gpreagg = gpuMemAlloc(&gpreagg->task, length);
	if (!gpreagg->m_gpreagg)
		goto out_of_resource;

	length = KERN_DATA_STORE_LENGTH(pds_in->kds);
	gpreagg->m_kds_in = gpuMemAlloc(&gpreagg->task, length);
	if (!gpreagg->m_kds_in)
		goto out_of_resource;

	length = KERN_DATA_STORE_LENGTH(pds_dst->kds);
	gpreagg->m_kds_src = gpuMemAlloc(&gpreagg->task, length);
	if (!gpreagg->m_kds_src)
		goto out_of_resource;
	gpreagg->m_kds_dst = gpuMemAlloc(&gpreagg->task, length);
	if (!gpreagg->m_kds_dst)
		goto out_of_resource;

	length = STROMALIGN(gpreagg->kern.hash_size * sizeof(pagg_hashslot));
	gpreagg->m_ghash = gpuMemAlloc(&gpreagg->task, length);
	if (!gpreagg->m_ghash)
		goto out_of_resource;


	/*
	 * Creation of event objects, if any
	 */
	if (gpreagg->task.pfm.enabled)
	{
		rc = cuEventCreate(&gpreagg->ev_dma_send_start, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));
		rc = cuEventCreate(&gpreagg->ev_dma_send_stop, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));
		rc = cuEventCreate(&gpreagg->ev_kern_prep_end, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));
		rc = cuEventCreate(&gpreagg->ev_kern_lagg_end, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));
		rc = cuEventCreate(&gpreagg->ev_dma_recv_start, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));
		rc = cuEventCreate(&gpreagg->ev_dma_recv_stop, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));
	}

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
	gpreagg->task.pfm.bytes_dma_send += length;
	gpreagg->task.pfm.num_dma_send++;

	length = KERN_DATA_STORE_LENGTH(pds_in->kds);
	rc = cuMemcpyHtoDAsync(gpreagg->m_kds_in,
						   pds_in->kds,
						   length,
						   gpreagg->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	gpreagg->task.pfm.bytes_dma_send += length;
	gpreagg->task.pfm.num_dma_send++;

	length = KERN_DATA_STORE_HEAD_LENGTH(pds_dst->kds);
	rc = cuMemcpyHtoDAsync(gpreagg->m_kds_src,
						   pds_dst->kds,
						   length,
						   gpreagg->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	gpreagg->task.pfm.bytes_dma_send += length;
	gpreagg->task.pfm.num_dma_send++;

	rc = cuMemcpyHtoDAsync(gpreagg->m_kds_dst,
						   pds_dst->kds,
						   length,
						   gpreagg->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	gpreagg->task.pfm.bytes_dma_send += length;
	gpreagg->task.pfm.num_dma_send++;

	CUDA_EVENT_RECORD(gpreagg, ev_dma_send_stop);

	/*
	 * Launch kernel functions
	 */

	/* KERNEL_FUNCTION(void)
	 * gpupreagg_preparation(kern_gpupreagg *kgpreagg,
	 *                       kern_data_store *kds_in,
	 *                       kern_data_store *kds_src,
	 *                       pagg_hashslot *g_hashslot)
	 */
	pgstrom_compute_workgroup_size(&grid_size,
								   &block_size,
								   gpreagg->kern_prep,
								   gpreagg->task.cuda_device,
								   false,
								   nitems,
								   sizeof(cl_uint));

	gpreagg->kern_prep_args[0] = &gpreagg->m_gpreagg;
	gpreagg->kern_prep_args[1] = &gpreagg->m_kds_in;
	gpreagg->kern_prep_args[2] = &gpreagg->m_kds_src;
	gpreagg->kern_prep_args[3] = &gpreagg->m_ghash;

	rc = cuLaunchKernel(gpreagg->kern_prep,
						grid_size, 1, 1,
                        block_size, 1, 1,
						sizeof(cl_uint) * block_size,
						gpreagg->task.cuda_stream,
						gpreagg->kern_prep_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
	gpreagg->task.pfm.num_kern_prep++;
	CUDA_EVENT_RECORD(gpreagg, ev_kern_prep_end);

	if (gpreagg->needs_grouping)
	{
		if (gpreagg->local_reduction)
		{
			/* KERNEL_FUNCTION(void)
			 * gpupreagg_local_reduction(kern_gpupreagg *kgpreagg,
			 *                           kern_data_store *kds_src,
			 *                           kern_data_store *kds_dst,
			 *                           kern_data_store *ktoast)
			 */
			pgstrom_compute_workgroup_size(&grid_size,
										   &block_size,
										   gpreagg->kern_lagg,
										   gpreagg->task.cuda_device,
										   true,
										   nitems,
										   Max(sizeof(pagg_hashslot),
											   sizeof(pagg_datum)));
			gpreagg->kern_lagg_args[0] = &gpreagg->m_gpreagg;
			gpreagg->kern_lagg_args[1] = &gpreagg->m_kds_src;
			gpreagg->kern_lagg_args[2] = &gpreagg->m_kds_dst;
			gpreagg->kern_lagg_args[3] = &gpreagg->m_kds_in;

			rc = cuLaunchKernel(gpreagg->kern_lagg,
								grid_size, 1, 1,
								block_size, 1, 1,
								Max(sizeof(pagg_hashslot),
									sizeof(pagg_datum)) * block_size,
								gpreagg->task.cuda_stream,
								gpreagg->kern_lagg_args,
								NULL);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
			gpreagg->task.pfm.num_kern_lagg++;
			CUDA_EVENT_RECORD(gpreagg, ev_kern_lagg_end);
		}

		/* KERNEL_FUNCTION(void)
		 * gpupreagg_global_reduction(kern_gpupreagg *kgpreagg,
		 *                            kern_data_store *kds_dst,
		 *                            kern_data_store *ktoast,
		 *                            pagg_hashslot *g_hashslot)
		 */
		pgstrom_compute_workgroup_size(&grid_size,
									   &block_size,
									   gpreagg->kern_gagg,
									   gpreagg->task.cuda_device,
									   false,
									   nitems,
									   sizeof(cl_uint));
		gpreagg->kern_gagg_args[0] = &gpreagg->m_gpreagg;
		gpreagg->kern_gagg_args[1] = &gpreagg->m_kds_dst;
		gpreagg->kern_gagg_args[2] = &gpreagg->m_kds_in;
		gpreagg->kern_gagg_args[3] = &gpreagg->m_ghash;

		rc = cuLaunchKernel(gpreagg->kern_gagg,
							grid_size, 1, 1,
							block_size, 1, 1,
							sizeof(cl_uint) * block_size,
							gpreagg->task.cuda_stream,
							gpreagg->kern_gagg_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
		gpreagg->task.pfm.num_kern_gagg++;
	}
	else
	{
		/* KERNEL_FUNCTION(void)
		 * gpupreagg_nogroup_reduction(kern_gpupreagg *kgpreagg,
		 *                             kern_data_store *kds_src,
		 *                             kern_data_store *kds_dst,
		 *                             kern_data_store *ktoast)
		 */
		pgstrom_compute_workgroup_size(&grid_size,
									   &block_size,
									   gpreagg->kern_nogrp,
									   gpreagg->task.cuda_device,
									   true,
									   nitems,
									   Max(sizeof(pagg_datum),
										   sizeof(cl_uint)));
		gpreagg->kern_nogrp_args[0] = &gpreagg->m_gpreagg;
		gpreagg->kern_nogrp_args[1] = &gpreagg->m_kds_src;
		gpreagg->kern_nogrp_args[2] = &gpreagg->m_kds_dst;
		gpreagg->kern_nogrp_args[3] = &gpreagg->m_kds_in;

		/* 1st path: data reduction (kds_src => kds_dst) */
		rc = cuLaunchKernel(gpreagg->kern_nogrp,
							grid_size, 1, 1,
							block_size, 1, 1,
							Max(sizeof(pagg_datum),
								sizeof(cl_uint)) * block_size,
							gpreagg->task.cuda_stream,
							gpreagg->kern_nogrp_args,
							NULL);
		if (rc != CUDA_SUCCESS)
            elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
		gpreagg->task.pfm.num_kern_nogrp++;

		/* 2nd path: data reduction (kds_dst => kds_src) */
		gpreagg->kern_nogrp_args[1] = &gpreagg->m_kds_dst;
		gpreagg->kern_nogrp_args[2] = &gpreagg->m_kds_src;
		rc = cuLaunchKernel(gpreagg->kern_nogrp,
							grid_size, 1, 1,
							block_size, 1, 1,
							Max(sizeof(pagg_datum),
								sizeof(cl_uint)) * block_size,
							gpreagg->task.cuda_stream,
							gpreagg->kern_nogrp_args,
							NULL);
		if (rc != CUDA_SUCCESS)
            elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
		gpreagg->task.pfm.num_kern_nogrp++;

		/* 3rd path: data reduction (kds_src => kds_dst) */
		gpreagg->kern_nogrp_args[1] = &gpreagg->m_kds_src;
		gpreagg->kern_nogrp_args[2] = &gpreagg->m_kds_dst;
		rc = cuLaunchKernel(gpreagg->kern_nogrp,
							grid_size, 1, 1,
							block_size, 1, 1,
							Max(sizeof(pagg_datum),
								sizeof(cl_uint)) * block_size,
							gpreagg->task.cuda_stream,
							gpreagg->kern_nogrp_args,
							NULL);
		if (rc != CUDA_SUCCESS)
            elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
		gpreagg->task.pfm.num_kern_nogrp++;
	}

	if (gpreagg->has_varlena)
	{
		/* KERNEL_FUNCTION(void)
		 * gpupreagg_fixup_varlena(kern_gpupreagg *kgpreagg,
		 *                         kern_data_store *kds_dst,
		 *                         kern_data_store *ktoast)
		 */
		pgstrom_compute_workgroup_size(&grid_size,
									   &block_size,
									   gpreagg->kern_fixvar,
									   gpreagg->task.cuda_device,
									   false,
									   nitems,
									   sizeof(cl_uint));
		gpreagg->kern_fixvar_args[0] = &gpreagg->m_gpreagg;
		gpreagg->kern_fixvar_args[1] = &gpreagg->m_kds_dst;
		gpreagg->kern_fixvar_args[2] = &gpreagg->m_kds_in;

		rc = cuLaunchKernel(gpreagg->kern_fixvar,
							grid_size, 1, 1,
							block_size, 1, 1,
							sizeof(cl_uint) * block_size,
							gpreagg->task.cuda_stream,
							gpreagg->kern_fixvar_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
	}

	/*
	 * commands for DMA recv
	 */
	CUDA_EVENT_RECORD(gpreagg, ev_dma_recv_start);

	offset = KERN_GPUPREAGG_DMARECV_OFFSET(&gpreagg->kern);
	length = KERN_GPUPREAGG_DMARECV_LENGTH(&gpreagg->kern, nitems);
	rc = cuMemcpyDtoHAsync(kresults,
						   gpreagg->m_gpreagg + offset,
                           length,
                           gpreagg->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuMemcpyDtoHAsync: %s", errorText(rc));
    gpreagg->task.pfm.bytes_dma_recv += length;
    gpreagg->task.pfm.num_dma_recv++;

	length = KERN_DATA_STORE_HEAD_LENGTH(pds_dst->kds);
	rc = cuMemcpyDtoHAsync(pds_dst->kds,
						   gpreagg->m_kds_dst,
						   length,
						   gpreagg->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	gpreagg->task.pfm.bytes_dma_send += length;
	gpreagg->task.pfm.num_dma_send++;

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
	DefineCustomBoolVariable("enable_gpupreagg",
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
