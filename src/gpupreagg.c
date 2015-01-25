/*
 * gpupreagg.c
 *
 * Aggregate Pre-processing with GPU acceleration
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
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
#include "opencl_numeric.h"
#include "opencl_gpupreagg.h"

static CustomScanMethods		gpupreagg_scan_methods;
static CustomExecMethods		gpupreagg_exec_methods;
static bool						enable_gpupreagg;
static bool						debug_force_gpupreagg;

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
	CustomScanState	css;
	ProjectionInfo *bulk_proj;
	TupleTableSlot *bulk_slot;
	double			num_groups;		/* estimated number of groups */
	double			ntups_per_page;	/* average number of tuples per page */
	List		   *outer_quals;
	bool			outer_done;
	bool			outer_bulkload;
	TupleTableSlot *outer_overflow;

	pgstrom_queue  *mqueue;
	const char	   *kern_source;
	Datum			dprog_key;
	kern_parambuf  *kparams;
	bool			local_reduction;
	bool			has_numeric;
	bool			has_varlena;

	pgstrom_gpupreagg  *curr_chunk;
	cl_uint			curr_index;
	bool			curr_recheck;
	cl_uint			num_rechecks;
	cl_uint			num_running;
	dlist_head		ready_chunks;

	pgstrom_perfmon	pfm;		/* performance counter */
} GpuPreAggState;

/* declaration of static functions */
static void clserv_process_gpupreagg(pgstrom_message *message);

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
	  "c:sum", 1, {INT4OID},
	  {ALTFUNC_EXPR_NROWS}, 0},
	{ "count", 1, {ANYOID},
	  "c:sum", 1, {INT4OID},
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
		((double)((pgstrom_chunk_size << 20) / BLCKSZ)) *
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

			dtype = pgstrom_devtype_lookup(type_oid);
			/* grouping key must be a supported data type */
			if (!dtype)
				return false;
			/* data type of the grouping key must have comparison function */
			if (!OidIsValid(dtype->type_cmpfunc) ||
				!pgstrom_devfunc_lookup(dtype->type_cmpfunc,
										InvalidOid))
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
        "static bool\n"
        "gpupreagg_qual_eval(__private cl_int *errcode,\n"
        "                    __global kern_parambuf *kparams,\n"
        "                    __global kern_data_store *kds,\n"
        "                    __global kern_data_store *ktoast,\n"
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
					 "static cl_uint\n"
					 "gpupreagg_hashvalue(__private cl_int *errcode,\n"
					 "                    __local cl_uint *crc32_table,\n"
					 "                    __global kern_data_store *kds,\n"
					 "                    __global kern_data_store *ktoast,\n"
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
 * gpupreagg_codegen_keycomp - code generator of kernel gpupreagg_keycomp();
 * that compares two records indexed by x_index and y_index in kern_data_store,
 * then returns -1 if X < Y, 0 if X = Y or 1 if X > Y.
 *
 * static cl_int
 * gpupreagg_keycomp(__private cl_int *errcode,
 *                   __global kern_data_store *kds,
 *                   __global kern_data_store *ktoast,
 *                   size_t x_index,
 *                   size_t y_index);
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
		if (!OidIsValid(dtype->type_cmpfunc))
			elog(ERROR, "Bug? type (%u) has no comparison function",
				 var->vartype);
		dfunc = pgstrom_devfunc_lookup_and_track(dtype->type_cmpfunc,
												 InvalidOid,
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
			"    comp = pgfn_%s(errcode, xkeyval_%u, ykeyval_%u);\n"
			"    if (!comp.isnull && comp.value != 0)\n"
			"      return comp.value;\n"
			"  }\n"
			"  else if (xkeyval_%u.isnull  && !ykeyval_%u.isnull)\n"
			"    return -1;\n"
			"  else if (!xkeyval_%u.isnull &&  ykeyval_%u.isnull)\n"
			"    return 1;\n",
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
					 "static cl_int\n"
					 "gpupreagg_keycomp(__private int *errcode,\n"
					 "                  __global kern_data_store *kds,\n"
					 "                  __global kern_data_store *ktoast,\n"
					 "                  size_t x_index,\n"
					 "                  size_t y_index)\n"
					 "{\n"
					 "%s"	/* variable/params declarations */
					 "%s"	/* definition of pg_int4_t comp */
					 "%s"
					 "  return 0;\n"
					 "}\n",
					 decl.data,
					 gpa_info->numCols > 0 ? "  pg_int4_t comp;\n" : "",
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
						  bool is_global_aggcalc, codegen_context *context)
{
	Oid				namespace_oid = get_namespace_oid("pgstrom", false);
	StringInfoData	body;
	const char	   *aggcalc_class;
	const char	   *aggcalc_args;
	ListCell	   *cell;

	initStringInfo(&body);
	if (!is_global_aggcalc)
	{
		appendStringInfo(
			&body,
			"static void\n"
			"gpupreagg_local_calc(__private cl_int *errcode,\n"
			"                     cl_int attnum,\n"
			"                     __local pagg_datum *accum,\n"
			"                     __local pagg_datum *newval)\n"
			"{\n");
		aggcalc_class = "LOCAL";
        aggcalc_args = "errcode,accum,newval";
	}
	else
	{
		appendStringInfo(
			&body,
			"static void\n"
			"gpupreagg_global_calc(__private cl_int *errcode,\n"
			"                      cl_int attnum,\n"
			"                      __global kern_data_store *kds,\n"
			"                      __global kern_data_store *ktoast,\n"
			"                      size_t accum_index,\n"
			"                      size_t newval_index)\n"
			"{\n"
			"  __global char  *accum_isnull;\n"
			"  __global Datum *accum_value;\n"
			"  char            new_isnull;\n"
			"  Datum           new_value;\n"
			"\n"
			"  if (kds->format != KDS_FORMAT_TUPSLOT)\n"
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
			min_const = "-DBL_MIN";
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
		"  pg_%s_vstore(%s,%s,errcode,%u,%s,\n"
		"               pgfn_%s(errcode, %s, %s));\n",
		temp_label,
		pgstrom_codegen_expression(clause, pc->context),
		temp_label,
		temp_label,
		zero_label,
		dtype->type_name,
		pc->kds_label,
		pc->ktoast_label,
		pc->tle->resno - 1,
		pc->rowidx_label,
		dfunc->func_alias,
		temp_label,
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
		"static void\n"
		"gpupreagg_projection(__private cl_int *errcode,\n"
		"            __global kern_parambuf *kparams,\n"
		"            __global kern_data_store *kds_in,\n"
		"            __global kern_data_store *kds_src,\n"
		"            __global void *ktoast,\n"
		"            size_t rowidx_in, size_t rowidx_out)\n"
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
	fn_local_calc = gpupreagg_codegen_aggcalc(cscan, gpa_info, false, context);
	/* generate a gpupreagg_global_calc function */
	fn_global_calc = gpupreagg_codegen_aggcalc(cscan, gpa_info, true, context);
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
					 "%s\n",	/* gpupreagg_projection() */
					 pgstrom_codegen_func_declarations(context),
					 fn_qualeval,
					 fn_hashvalue,
					 fn_keycomp,
					 fn_local_calc,
					 fn_global_calc,
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
		alter_node = gpuscan_try_replace_relscan(outer_node,
												 pstmt->rtable,
												 attr_refs,
												 &outer_quals);
		if (alter_node)
		{
			outer_node = alter_node;
			outer_bulkload = true;
		}
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
		alter_node = gpuscan_try_replace_relscan(outer_node,
												 pstmt->rtable,
												 attr_refs,
												 &outer_quals);
		if (alter_node)
		{
			outer_node = alter_node;
			outer_bulkload = true;
		}
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
		alter_node = gpuscan_try_replace_relscan(outer_node,
												 pstmt->rtable,
												 attr_refs,
												 &outer_quals);
		if (alter_node)
		{
			outer_node = alter_node;
			outer_bulkload = true;
		}
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
								 tle->resjunk);
		cscan->custom_ps_tlist = lappend(cscan->custom_ps_tlist, ps_tle);
	}
	outerPlan(cscan)              = outer_node;

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
	gpa_info.extra_flags = extra_flags | context.extra_flags |
		(!devprog_enable_optimize ? DEVKERNEL_DISABLE_OPTIMIZE : 0);
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
	GpuPreAggState *gpas = palloc0(sizeof(GpuPreAggState));

	NodeSetTag(gpas, T_CustomScanState);
	gpas->css.methods = &gpupreagg_exec_methods;

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
	gpas->outer_bulkload =
		(!pgstrom_debug_bulkload_enabled ? false : gpa_info->outer_bulkload);
	gpas->outer_done = false;
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
	if (gpas->outer_bulkload)
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
	gpas->kern_source = gpa_info->kern_source;
	gpas->dprog_key = pgstrom_get_devprog_key(gpa_info->kern_source,
											  gpa_info->extra_flags);
	pgstrom_track_object((StromObject *)gpas->dprog_key, 0);
	gpas->mqueue = pgstrom_create_queue();
	pgstrom_track_object(&gpas->mqueue->sobj, 0);

	/*
	 * init misc stuff
	 */
	gpas->local_reduction = true;	/* tentative */
	gpas->has_numeric = gpa_info->has_numeric;
	gpas->has_varlena = gpa_info->has_varlena;
	gpas->curr_chunk = NULL;
	gpas->curr_index = 0;
	gpas->curr_recheck = false;
	gpas->num_rechecks = 0;
	dlist_init(&gpas->ready_chunks);

	/*
	 * Is perfmon needed?
	 */
	gpas->pfm.enabled = pgstrom_perfmon_enabled;
}

static void
pgstrom_release_gpupreagg(pgstrom_message *message)
{
	pgstrom_gpupreagg *gpupreagg = (pgstrom_gpupreagg *) message;

	/* unlink message queue and device program */
	pgstrom_put_queue(gpupreagg->msg.respq);
	pgstrom_put_devprog_key(gpupreagg->dprog_key);

	/* unlink source data-store */
	pgstrom_put_data_store(gpupreagg->pds);

	/* unlink result data-store */
	if (gpupreagg->pds_dest)
		pgstrom_put_data_store(gpupreagg->pds_dest);

	pgstrom_shmem_free(gpupreagg);
}

static pgstrom_gpupreagg *
pgstrom_create_gpupreagg(GpuPreAggState *gpas, pgstrom_bulkslot *bulk)
{
	pgstrom_gpupreagg  *gpupreagg;
	kern_parambuf	   *kparams;
	kern_row_map	   *krowmap;
	pgstrom_data_store *pds = bulk->pds;
	kern_data_store	   *kds = pds->kds;
	pgstrom_data_store *pds_dest;
	TupleDesc			tupdesc;
	cl_int				nvalids = bulk->nvalids;
	cl_uint				nitems = kds->nitems;
	Size				required;

	/*
	 * Allocation of pgtrom_gpupreagg message object
	 *
	 * NOTE: kern_row_map is also used to buffer of kds_index.
	 * So, even if we have no input row-map, at least nitems's
	 * slot needed to be allocated.
	 */
	required = STROMALIGN(offsetof(pgstrom_gpupreagg,
								   kern.kparams) +
						  gpas->kparams->length);
	if (nvalids < 0)
		required += STROMALIGN(offsetof(kern_row_map, rindex[nitems]));
	else
		required += STROMALIGN(offsetof(kern_row_map, rindex[nvalids]));
	gpupreagg = pgstrom_shmem_alloc(required);
	if (!gpupreagg)
		elog(ERROR, "out of shared memory");

	/* initialize the common message field */
	memset(gpupreagg, 0, required);
	gpupreagg->msg.sobj.stag = StromTag_GpuPreAgg;
	SpinLockInit(&gpupreagg->msg.lock);
	gpupreagg->msg.refcnt = 1;
    gpupreagg->msg.respq = pgstrom_get_queue(gpas->mqueue);
    gpupreagg->msg.cb_process = clserv_process_gpupreagg;
    gpupreagg->msg.cb_release = pgstrom_release_gpupreagg;
    gpupreagg->msg.pfm.enabled = gpas->pfm.enabled;
	/* other fields also */
	gpupreagg->dprog_key = pgstrom_retain_devprog_key(gpas->dprog_key);
	gpupreagg->local_reduction = gpas->local_reduction;
	gpupreagg->has_varlena = gpas->has_varlena;
	gpupreagg->num_groups = gpas->num_groups;
	gpupreagg->pds = pds;
	/*
	 * Once a row/column data-store connected to the pgstrom_gpupreagg
	 * structure, it becomes pgstrom_release_gpupreagg()'s role to
	 * unlink this data-store. So, we don't need to track individual
	 * data-store no longer.
	 */
	pgstrom_untrack_object(&pds->sobj);
	pgstrom_track_object(&gpupreagg->msg.sobj, 0);

	/*
	 * Also initialize kern_gpupreagg portion
	 */
	gpupreagg->kern.status = StromError_Success;
	gpupreagg->kern.hash_size = (nvalids < 0 ? nitems : nvalids);
	memcpy(gpupreagg->kern.pg_crc32_table,
		   pg_crc32c_table,
		   sizeof(uint32) * 256);
	/* kern_parambuf */
	kparams = KERN_GPUPREAGG_PARAMBUF(&gpupreagg->kern);
	memcpy(kparams, gpas->kparams, gpas->kparams->length);
	/* kern_row_map */
	krowmap = KERN_GPUPREAGG_KROWMAP(&gpupreagg->kern);
	if (nvalids < 0)
		krowmap->nvalids = -1;
	else
	{
		krowmap->nvalids = nvalids;
		memcpy(krowmap->rindex, bulk->rindex, sizeof(cl_uint) * nvalids);
	}

	/*
	 * Allocation of the result data-store
	 */
	tupdesc = gpas->css.ss.ps.ps_ResultTupleSlot->tts_tupleDescriptor;
	pds_dest = pgstrom_create_data_store_tupslot(tupdesc,
												 (nvalids < 0
												  ? nitems
												  : nvalids),
												 gpas->has_numeric);
	if (!pds_dest)
		elog(ERROR, "out of shared memory");
	gpupreagg->pds_dest = pds_dest;

	return gpupreagg;
}

static pgstrom_gpupreagg *
gpupreagg_load_next_outer(GpuPreAggState *gpas)
{
	PlanState		   *subnode = outerPlanState(gpas);
	pgstrom_gpupreagg  *gpupreagg = NULL;
	pgstrom_data_store *pds = NULL;
	pgstrom_bulkslot	bulkdata;
	pgstrom_bulkslot   *bulk = NULL;
	struct timeval		tv1, tv2;

	if (gpas->outer_done)
		return NULL;

	if (gpas->pfm.enabled)
		gettimeofday(&tv1, NULL);

	if (!gpas->outer_bulkload)
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
					gpas->outer_done = true;
					break;
				}
			}

			if (!pds)
			{
				Size	chunk_size = pgstrom_chunk_size << 20;
				pds = pgstrom_create_data_store_row_flat(tupdesc, chunk_size);
				pgstrom_track_object(&pds->sobj, 0);
			}
			/* insert tuple to the data-store */
			if (!pgstrom_data_store_insert_tuple(pds, slot))
			{
				gpas->outer_overflow = slot;
				break;
			}
		}

		if (pds)
		{
			memset(&bulkdata, 0, sizeof(pgstrom_bulkslot));
			bulkdata.pds = pds;
			bulkdata.nvalids = -1;	/* all valid */
			bulk = &bulkdata;
		}
	}
	else
	{
		/* Load a bunch of records at once */
		bulk = (pgstrom_bulkslot *) BulkExecProcNode(subnode);
        if (!bulk)
			gpas->outer_done = true;
	}
	if (gpas->pfm.enabled)
	{
		gettimeofday(&tv2, NULL);
		gpas->pfm.time_outer_load += timeval_diff(&tv1, &tv2);
	}
	if (bulk)
	{
		/* Older style no longer supported */
		if (bulk->nvalids >= 0)
			elog(ERROR, "Bulk-load with rowmap no longer supported");
		gpupreagg = pgstrom_create_gpupreagg(gpas, bulk);
	}

	return gpupreagg;
}

/*
 * gpupreagg_next_tuple_fallback - a fallback routine if GPU returned
 * StromError_CpuReCheck, to suggest the backend to handle request
 * by itself. A fallback process looks like construction of special
 * partial aggregations that consist of individual rows; so here is
 * no performance benefit once it happen.
 */
static TupleTableSlot *
gpupreagg_next_tuple_fallback(GpuPreAggState *gpas)
{
	pgstrom_gpupreagg  *gpreagg = gpas->curr_chunk;
	pgstrom_data_store *pds = gpreagg->pds;
	kern_data_store	   *kds = pds->kds;
	TupleTableSlot	   *slot_in;
	TupleTableSlot	   *slot_out = NULL;
	cl_uint				row_index;
	HeapTupleData		tuple;

	/* bulk-load uses individual slot; then may have a projection */
	if (!gpas->outer_bulkload)
		slot_in = gpas->css.ss.ss_ScanTupleSlot;
	else
		slot_in = gpas->bulk_slot;

retry:
	if (gpas->curr_index >= kds->nitems)
		return NULL;
	row_index = gpas->curr_index++;

	/*
	 * Fetch a tuple from the data-store
	 */
	if (pgstrom_fetch_data_store(slot_in, pds, row_index, &tuple))
	{
		ProjectionInfo *projection = gpas->css.ss.ps.ps_ProjInfo;
		ExprContext	   *econtext = gpas->css.ss.ps.ps_ExprContext;
		ExprDoneCond	is_done;

		/* reset per-tuple memory context */
		ResetExprContext(econtext);

		/*
		 * check qualifier being pulled up from the outer scan, if any.
		 * outer_quals assumes fetched tuple is in ecxt_scantuple (because
		 * it came from relation-scan), we need to adjust it.
		 */
		if (gpas->outer_quals != NIL)
		{
			econtext->ecxt_scantuple = slot_in;
			if (!ExecQual(gpas->outer_quals, econtext, false))
				goto retry;
		}

		/*
		 * In case of bulk-loading mode, it may take additional projection
		 * because slot_in has a record type of underlying scan node, thus
		 * we need to translate this record into the form we expected.
		 * If bulk_proj is valid, it implies our expected input record is
		 * incompatible from the record type of underlying scan.
		 */
		if (gpas->outer_bulkload)
		{
			if (gpas->bulk_proj)
			{
				ExprContext	*bulk_econtext = gpas->bulk_proj->pi_exprContext;

				bulk_econtext->ecxt_scantuple = slot_in;
				slot_in = ExecProject(gpas->bulk_proj, &is_done);
				if (is_done == ExprEndResult)
				{
					slot_out = NULL;
					goto retry;
				}
			}
		}
		/* put result tuple */
		if (!projection)
		{
			slot_out = gpas->css.ss.ps.ps_ResultTupleSlot;
			ExecCopySlot(slot_out, slot_in);
		}
		else
		{
			econtext->ecxt_scantuple = slot_in;
			slot_out = ExecProject(projection, &is_done);
			if (is_done == ExprEndResult)
			{
				slot_out = NULL;
				goto retry;
			}
			gpas->css.ss.ps.ps_TupFromTlist = (is_done == ExprMultipleResult);
		}
	}
	return slot_out;
}

static TupleTableSlot *
gpupreagg_next_tuple(GpuPreAggState *gpas)
{
	pgstrom_gpupreagg  *gpreagg = gpas->curr_chunk;
	kern_row_map       *krowmap = KERN_GPUPREAGG_KROWMAP(&gpreagg->kern);
	pgstrom_data_store *pds_dest = gpreagg->pds_dest;
	TupleTableSlot	   *slot = NULL;
	HeapTupleData		tuple;
	struct timeval		tv1, tv2;

	if (gpas->pfm.enabled)
		gettimeofday(&tv1, NULL);

	if (gpas->curr_recheck)
		slot = gpupreagg_next_tuple_fallback(gpas);
	else if (gpas->curr_index < krowmap->nvalids)
	{
		size_t		row_index = krowmap->rindex[gpas->curr_index++];

		slot = gpas->css.ss.ps.ps_ResultTupleSlot;
		if (!pgstrom_fetch_data_store(slot, pds_dest, row_index, &tuple))
		{
			elog(NOTICE, "Bug? empty slot was specified by kern_row_map");
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

	if (gpas->pfm.enabled)
	{
		gettimeofday(&tv2, NULL);
		gpas->pfm.time_materialize += timeval_diff(&tv1, &tv2);
	}
	return slot;
}

static TupleTableSlot *
gpupreagg_exec(CustomScanState *node)
{
	GpuPreAggState	   *gpas = (GpuPreAggState *) node;
	TupleTableSlot	   *slot = NULL;
	pgstrom_gpupreagg  *gpreagg;

	while (!gpas->curr_chunk || !(slot = gpupreagg_next_tuple(gpas)))
	{
		pgstrom_message	   *msg;
		dlist_node		   *dnode;

		/* release current gpupreagg chunk being already fetched */
		if (gpas->curr_chunk)
		{
			msg = &gpas->curr_chunk->msg;
			if (msg->pfm.enabled)
				pgstrom_perfmon_add(&gpas->pfm, &msg->pfm);
			Assert(msg->refcnt == 1);
			pgstrom_untrack_object(&msg->sobj);
			pgstrom_put_message(msg);
			gpas->curr_chunk = NULL;
			gpas->curr_index = 0;
			gpas->curr_recheck = false;
		}

		/*
		 * Keep number of asynchronous partial aggregate request a particular
		 * level unless it does not exceed pgstrom_max_async_chunks and any
		 * new response is not replied during the loading.
		 */
		while (!gpas->outer_done &&
			   gpas->num_running <= pgstrom_max_async_chunks)
		{
			gpreagg = gpupreagg_load_next_outer(gpas);
			if (!gpreagg)
				break;	/* outer scan reached to end of the relation */

			if (!pgstrom_enqueue_message(&gpreagg->msg))
			{
				pgstrom_put_message(&gpreagg->msg);
				elog(ERROR, "failed to enqueue pgstrom_gpuhashjoin message");
			}
            gpas->num_running++;

			msg = pgstrom_try_dequeue_message(gpas->mqueue);
			if (msg)
			{
				gpas->num_running--;
				dlist_push_tail(&gpas->ready_chunks, &msg->chain);
				break;
			}
		}

		/*
		 * wait for server's response if no available chunks were replied
		 */
		if (dlist_is_empty(&gpas->ready_chunks))
		{
			/* OK, no more request should be fetched */
			if (gpas->num_running == 0)
				break;
			msg = pgstrom_dequeue_message(gpas->mqueue);
			if (!msg)
				elog(ERROR, "message queue wait timeout");
			gpas->num_running--;
			dlist_push_tail(&gpas->ready_chunks, &msg->chain);
		}

		/*
		 * picks up next available chunks, if any
		 */
		Assert(!dlist_is_empty(&gpas->ready_chunks));
		dnode = dlist_pop_head_node(&gpas->ready_chunks);
		gpreagg = dlist_container(pgstrom_gpupreagg, msg.chain, dnode);

		/*
		 * Raise an error, if significan error was reported
		 */
		if (gpreagg->msg.errcode == StromError_Success)
			gpas->curr_recheck = false;
		else if (gpreagg->msg.errcode == StromError_CpuReCheck)
		{
			gpas->curr_recheck = true;	/* fallback by CPU */
			gpas->num_rechecks++;
		}
		else if (gpreagg->msg.errcode == CL_BUILD_PROGRAM_FAILURE)
		{
			const char *buildlog
				= pgstrom_get_devprog_errmsg(gpas->dprog_key);

			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("PG-Strom: OpenCL execution error (%s)\n%s",
							pgstrom_strerror(gpreagg->msg.errcode),
							gpas->kern_source),
					 errdetail("%s", buildlog)));
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("PG-Strom: OpenCL execution error (%s)",
							pgstrom_strerror(gpreagg->msg.errcode))));
		}
		gpas->curr_chunk = gpreagg;
		gpas->curr_index = 0;
	}
	return slot;
}

static void
gpupreagg_end(CustomScanState *node)
{
	GpuPreAggState	   *gpas = (GpuPreAggState *) node;
	pgstrom_message	   *msg;

	/* Debug message if needed */
	if (gpas->num_rechecks > 0)
		elog(NOTICE, "GpuPreAgg: %u chunks were re-checked by CPU",
			 gpas->num_rechecks);

	/* Clean up strom objects */
	if (gpas->curr_chunk)
	{
		msg = &gpas->curr_chunk->msg;
		if (msg->pfm.enabled)
			pgstrom_perfmon_add(&gpas->pfm, &msg->pfm);
		pgstrom_untrack_object(&msg->sobj);
		pgstrom_put_message(msg);
	}

	while (gpas->num_running > 0)
	{
		msg = pgstrom_dequeue_message(gpas->mqueue);
		if (!msg)
			elog(ERROR, "message queue wait timeout");
		pgstrom_untrack_object(&msg->sobj);
        pgstrom_put_message(msg);
		gpas->num_running--;
	}

	pgstrom_untrack_object((StromObject *)gpas->dprog_key);
	pgstrom_put_devprog_key(gpas->dprog_key);
	pgstrom_untrack_object(&gpas->mqueue->sobj);
	pgstrom_close_queue(gpas->mqueue);

	/* Clean up subtree */
	ExecEndNode(outerPlanState(node));
}

static void
gpupreagg_rescan(CustomScanState *node)
{
	GpuPreAggState	   *gpas = (GpuPreAggState *) node;
	pgstrom_message	   *msg;

	/* Clean up strom objects */
	if (gpas->curr_chunk)
	{
		msg = &gpas->curr_chunk->msg;
		if (msg->pfm.enabled)
			pgstrom_perfmon_add(&gpas->pfm, &msg->pfm);
		pgstrom_untrack_object(&msg->sobj);
		pgstrom_put_message(msg);
		gpas->curr_chunk = NULL;
		gpas->curr_index = 0;
		gpas->curr_recheck = false;
	}

	while (gpas->num_running > 0)
	{
		msg = pgstrom_dequeue_message(gpas->mqueue);
		if (!msg)
			elog(ERROR, "message queue wait timeout");
		pgstrom_untrack_object(&msg->sobj);
		pgstrom_put_message(msg);
		gpas->num_running--;
	}

	/* Rewind the subtree */
	gpas->outer_done = false;
	ExecReScan(outerPlanState(node));
}

static void
gpupreagg_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
	GpuPreAggState *gpas = (GpuPreAggState *) node;
	GpuPreAggInfo  *gpa_info
		= deform_gpupreagg_info((CustomScan *) node->ss.ps.plan);

	ExplainPropertyText("Bulkload",
						gpas->outer_bulkload ? "On" : "Off", es);
	show_device_kernel(gpas->dprog_key, es);
	if (gpa_info->outer_quals != NIL)
	{
		show_scan_qual(gpa_info->outer_quals,
					   "Device Filter", &gpas->css.ss.ps, ancestors, es);
		show_instrumentation_count("Rows Removed by Device Fileter",
                                   2, &gpas->css.ss.ps, es);
	}
	if (es->analyze && gpas->pfm.enabled)
		pgstrom_perfmon_explain(&gpas->pfm, es);
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

/* ----------------------------------------------------------------
 *
 * NOTE: below is the code being run on OpenCL server context
 *
 * ---------------------------------------------------------------- */

typedef struct
{
	pgstrom_gpupreagg  *gpreagg;
	cl_command_queue	kcmdq;
	cl_int				dindex;
	cl_program			program;
	cl_kernel			kern_prep;
	cl_kernel			kern_init;	/* global hash init */
	cl_kernel			kern_lagg;	/* local reduction */
	cl_kernel			kern_gagg;	/* global reduction */
	cl_kernel			kern_fixvar;/* fixup varlena (if any) */
	cl_mem				m_gpreagg;
	cl_mem				m_kds_in;	/* kds of input relation */
	cl_mem				m_kds_src;	/* kds of aggregation source */
	cl_mem				m_kds_dst;	/* kds of aggregation results */
	cl_mem				m_ghash;	/* global hashslot */
	cl_uint				ev_kern_prep;	/* event index of kern_prep */
	cl_uint				ev_kern_lagg;	/* event index of kern_lagg */
	cl_uint				ev_kern_init;	/* event index of kern_init */
	cl_uint				ev_kern_gagg;	/* event index of kern_gagg */
	cl_uint				ev_kern_fixvar;	/* event index of kern_fixvar */
	cl_uint				ev_dma_recv;	/* event index of DMA recv */
	cl_uint				ev_limit;
	cl_uint				ev_index;
	cl_event			events[FLEXIBLE_ARRAY_MEMBER];
} clstate_gpupreagg;

static void
clserv_respond_gpupreagg(cl_event event, cl_int ev_status, void *private)
{
	clstate_gpupreagg  *clgpa = (clstate_gpupreagg *) private;
	pgstrom_gpupreagg  *gpreagg = clgpa->gpreagg;
	cl_int				i, rc;

	if (ev_status == CL_COMPLETE)
		gpreagg->msg.errcode = gpreagg->kern.status;
	else
	{
		clserv_log("unexpected CL_EVENT_COMMAND_EXECUTION_STATUS: %d",
				   ev_status);
		gpreagg->msg.errcode = StromError_OpenCLInternal;
    }

	/* collect performance statistics */
	if (gpreagg->msg.pfm.enabled)
	{
		cl_ulong	tv_start;
		cl_ulong	tv_end;
		cl_ulong	temp;

		/*
		 * Time of all the DMA send
		 */
		tv_start = ~0UL;
		tv_end = 0;
		for (i=0; i < clgpa->ev_kern_prep; i++)
		{
			rc = clGetEventProfilingInfo(clgpa->events[i],
										 CL_PROFILING_COMMAND_START,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			tv_start = Min(tv_start, temp);

			rc = clGetEventProfilingInfo(clgpa->events[i],
										 CL_PROFILING_COMMAND_END,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			tv_end = Max(tv_end, temp);
		}
		gpreagg->msg.pfm.time_dma_send += (tv_end - tv_start) / 1000;

		/*
		 * Prep kernel execution time (includes global hash table init)
		 */
		i = clgpa->ev_kern_prep;
		rc = clGetEventProfilingInfo(clgpa->events[i],
									 CL_PROFILING_COMMAND_START,
									 sizeof(cl_ulong),
									 &tv_start,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;
		rc = clGetEventProfilingInfo(clgpa->events[i],
									 CL_PROFILING_COMMAND_END,
									 sizeof(cl_ulong),
									 &tv_end,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;
		gpreagg->msg.pfm.time_kern_prep += (tv_end - tv_start) / 1000;

		/*
		 * Local reduction kernel execution time (if any)
		 */
		if (clgpa->kern_lagg)
		{
			i = clgpa->ev_kern_lagg;
			rc = clGetEventProfilingInfo(clgpa->events[i],
										 CL_PROFILING_COMMAND_START,
										 sizeof(cl_ulong),
										 &tv_start,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			rc = clGetEventProfilingInfo(clgpa->events[i],
										 CL_PROFILING_COMMAND_END,
										 sizeof(cl_ulong),
										 &tv_end,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			gpreagg->msg.pfm.time_kern_lagg += (tv_end - tv_start) / 1000;
		}

		/*
		 * Global reduction kernel execution time
		 * (incl. global hash table init / fixup varlena datum)
		 */
		if (clgpa->kern_init)
		{
			i = clgpa->ev_kern_init;
			rc = clGetEventProfilingInfo(clgpa->events[i],
										 CL_PROFILING_COMMAND_START,
										 sizeof(cl_ulong),
										 &tv_start,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			rc = clGetEventProfilingInfo(clgpa->events[i],
										 CL_PROFILING_COMMAND_END,
										 sizeof(cl_ulong),
										 &tv_end,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			gpreagg->msg.pfm.time_kern_prep += (tv_end - tv_start) / 1000;
		}

		i = clgpa->ev_kern_gagg;
		rc = clGetEventProfilingInfo(clgpa->events[i],
									 CL_PROFILING_COMMAND_START,
									 sizeof(cl_ulong),
									 &tv_start,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;
		rc = clGetEventProfilingInfo(clgpa->events[i],
									 CL_PROFILING_COMMAND_END,
									 sizeof(cl_ulong),
									 &tv_end,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;
		gpreagg->msg.pfm.time_kern_gagg += (tv_end - tv_start) / 1000;

		if (clgpa->kern_fixvar)
		{
			i = clgpa->ev_kern_fixvar;
			rc = clGetEventProfilingInfo(clgpa->events[i],
										 CL_PROFILING_COMMAND_START,
										 sizeof(cl_ulong),
										 &tv_start,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			rc = clGetEventProfilingInfo(clgpa->events[i],
										 CL_PROFILING_COMMAND_END,
										 sizeof(cl_ulong),
										 &tv_end,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			gpreagg->msg.pfm.time_kern_gagg += (tv_end - tv_start) / 1000;
		}

		/*
		 * DMA recv time - last two event should be DMA receive request
		 */
		tv_start = ~0UL;
		tv_end = 0;
		for (i = clgpa->ev_dma_recv; i < clgpa->ev_index; i++)
		{
			rc = clGetEventProfilingInfo(clgpa->events[i],
										 CL_PROFILING_COMMAND_START,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			tv_start = Min(tv_start, temp);

			rc = clGetEventProfilingInfo(clgpa->events[i],
										 CL_PROFILING_COMMAND_END,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			tv_end = Max(tv_end, temp);
		}
		gpreagg->msg.pfm.time_dma_recv += (tv_end - tv_start) / 1000;

	skip_perfmon:
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clGetEventProfilingInfo (%s)",
					   opencl_strerror(rc));
            gpreagg->msg.pfm.enabled = false;   /* turn off profiling */
		}
	}

	/*
	 * release opencl resources
	 */
	while (clgpa->ev_index > 0)
		clReleaseEvent(clgpa->events[--clgpa->ev_index]);	
	if (clgpa->m_gpreagg)
		clReleaseMemObject(clgpa->m_gpreagg);
	if (clgpa->m_kds_in)
		clReleaseMemObject(clgpa->m_kds_in);
	if (clgpa->m_kds_src)
		clReleaseMemObject(clgpa->m_kds_src);
	if (clgpa->m_kds_dst)
		clReleaseMemObject(clgpa->m_kds_dst);
	if (clgpa->m_ghash)
		clReleaseMemObject(clgpa->m_ghash);
	if (clgpa->kern_prep)
		clReleaseKernel(clgpa->kern_prep);
	if (clgpa->kern_lagg)
		clReleaseKernel(clgpa->kern_lagg);
	if (clgpa->kern_gagg)
		clReleaseKernel(clgpa->kern_gagg);
	if (clgpa->kern_fixvar)
		clReleaseKernel(clgpa->kern_fixvar);
	if (clgpa->program && clgpa->program != BAD_OPENCL_PROGRAM)
		clReleaseProgram(clgpa->program);
	free(clgpa);

	/* dump kds */
	// clserv_dump_kds(gpreagg->kds_dst);

	/* reply the result to backend side */
	pgstrom_reply_message(&gpreagg->msg);
}

static cl_int
clserv_launch_preagg_preparation(clstate_gpupreagg *clgpa, cl_uint nitems)
{
	cl_int		rc;
	size_t		gwork_sz;
	size_t		lwork_sz;

	/* __kernel void
	 * gpupreagg_preparation(__global kern_gpupreagg *kgpreagg,
	 *                       __global kern_data_store *kds_in,
	 *                       __global kern_data_store *kds_src,
	 *                       __local void *local_memory)
	 */
	clgpa->kern_prep = clCreateKernel(clgpa->program,
									  "gpupreagg_preparation",
									  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		return rc;
	}

	if (!clserv_compute_workgroup_size(&gwork_sz,
									   &lwork_sz,
									   clgpa->kern_prep,
									   clgpa->dindex,
									   true,
									   nitems,
									   sizeof(cl_uint)))
	{
		clserv_log("failed to compute optimal gwork_sz/lwork_sz");
		return StromError_OpenCLInternal;
	}

	rc = clSetKernelArg(clgpa->kern_prep,
						0,		/* __kern_gpupreagg *kgpreagg */
						sizeof(cl_mem),
						&clgpa->m_gpreagg);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_prep,
						1,		/* __global kern_data_store *kds_in */
						sizeof(cl_mem),
						&clgpa->m_kds_in);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_prep,
						2,		/* __global kern_data_store *kds_src */
						sizeof(cl_mem),
						&clgpa->m_kds_src);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_prep,
						3,
						sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	/*
	 * kick gpupreagg_preparation() after all the DMA data send
	 */
	rc = clEnqueueNDRangeKernel(clgpa->kcmdq,
                                clgpa->kern_prep,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								clgpa->ev_index,
								&clgpa->events[0],
								&clgpa->events[clgpa->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		return rc;
	}
	clgpa->ev_kern_prep = clgpa->ev_index++;
	clgpa->gpreagg->msg.pfm.num_kern_prep++;

	return CL_SUCCESS;
}

static cl_int
clserv_launch_init_hashslot(clstate_gpupreagg *clgpa, cl_uint nitems)
{
	size_t		gwork_sz;
	size_t		lwork_sz;
	cl_int		rc;

	/*
	 * __kernel void
	 * gpupreagg_init_global_hashslot(__global kern_gpupreagg *kgpreagg,
	 *                                __global pagg_hashslot *g_hashslot)
	 */
	clgpa->kern_init = clCreateKernel(clgpa->program,
									  "gpupreagg_init_global_hashslot",
									  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		return rc;
	}

	if (!clserv_compute_workgroup_size(&gwork_sz,
									   &lwork_sz,
									   clgpa->kern_init,
									   clgpa->dindex,
									   true,
									   nitems,
									   sizeof(cl_uint)))
	{
		clserv_log("failed to compute optimal gwork_sz/lwork_sz");
		return StromError_OpenCLInternal;
	}

	rc = clSetKernelArg(clgpa->kern_init,
						0,		/* kern_gpupreagg *kgpreagg */
						sizeof(cl_mem),
						&clgpa->m_gpreagg);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_init,
						1,		/* pagg_hashslot *g_hashslot */
						sizeof(cl_mem),
						&clgpa->m_ghash);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	/*
	 * Kick gpupreagg_init_global_hashslot next to the preparation.
	 */
	rc = clEnqueueNDRangeKernel(clgpa->kcmdq,
								clgpa->kern_init,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								1,
                                &clgpa->events[clgpa->ev_index - 1],
                                &clgpa->events[clgpa->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		return rc;
	}
	clgpa->ev_kern_init = clgpa->ev_index++;
	clgpa->gpreagg->msg.pfm.num_kern_gagg++;

	return CL_SUCCESS;
}

static cl_int
clserv_launch_local_reduction(clstate_gpupreagg *clgpa, cl_uint nitems)
{
	size_t		gwork_sz;
	size_t		lwork_sz;
	cl_int		rc;

	/*
	 * __kernel void
	 * gpupreagg_local_reduction(__global kern_gpupreagg *kgpreagg,
	 *                           __global kern_data_store *kds_src,
	 *                           __global kern_data_store *kds_dst,
	 *                           __global kern_data_store *ktoast,
	 *                           __global pagg_hashslot *g_hashslot,
	 *                           KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
	 */
	clgpa->kern_lagg = clCreateKernel(clgpa->program,
									  "gpupreagg_local_reduction",
									  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		return rc;
	}

	if (!clserv_compute_workgroup_size(&gwork_sz,
									   &lwork_sz,
									   clgpa->kern_lagg,
									   clgpa->dindex,
									   true,
									   nitems,
									   Max(sizeof(pagg_hashslot),
										   sizeof(pagg_datum))))
	{
		clserv_log("failed to compute optimal gwork_sz/lwork_sz");
		return StromError_OpenCLInternal;
	}

	rc = clSetKernelArg(clgpa->kern_lagg,
						0,		/* kern_gpupreagg *kgpreagg */
						sizeof(cl_mem),
						&clgpa->m_gpreagg);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_lagg,
						1,		/* kern_data_store *kds_src */
						sizeof(cl_mem),
						&clgpa->m_kds_src);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_lagg,
						2,		/* kern_data_store *kds_dst */
						sizeof(cl_mem),
						&clgpa->m_kds_dst);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_lagg,
						3,		/* kern_data_store *ktoast */
						sizeof(cl_mem),
						&clgpa->m_kds_in);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_lagg,
						4,		/* pagg_hashslot *g_hashslot */
						sizeof(cl_mem),
						&clgpa->m_ghash);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_lagg,
						5,		/* KERN_DYNAMIC_LOCAL_WORKMEM_ARG */
						Max(sizeof(pagg_hashslot),
							sizeof(pagg_datum)) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clEnqueueNDRangeKernel(clgpa->kcmdq,
								clgpa->kern_lagg,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								1,
								&clgpa->events[clgpa->ev_index - 1],
								&clgpa->events[clgpa->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		return rc;
	}
	clgpa->ev_kern_lagg = clgpa->ev_index++;
	clgpa->gpreagg->msg.pfm.num_kern_lagg++;

	return CL_SUCCESS;
}

static cl_int
clserv_launch_global_reduction(clstate_gpupreagg *clgpa, cl_uint nitems)
{
	size_t		gwork_sz;
	size_t		lwork_sz;
	cl_int		rc;

	/*
	 * __kernel void
	 * gpupreagg_global_reduction(__global kern_gpupreagg *kgpreagg,
	 *                            __global kern_data_store *kds_dst,
	 *                            __global kern_data_store *ktoast,
	 *                            __global pagg_hashslot *g_hashslot,
	 *                            KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
	 */
	clgpa->kern_gagg = clCreateKernel(clgpa->program,
									  "gpupreagg_global_reduction",
									  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		return rc;
	}

	if (!clserv_compute_workgroup_size(&gwork_sz,
									   &lwork_sz,
									   clgpa->kern_gagg,
									   clgpa->dindex,
									   true,
									   nitems,
									   sizeof(cl_uint)))
	{
		clserv_log("failed to compute optimal gwork_sz/lwork_sz");
		return StromError_OpenCLInternal;
	}

	rc = clSetKernelArg(clgpa->kern_gagg,
						0,		/* kern_gpupreagg *kgpreagg */
						sizeof(cl_mem),
						&clgpa->m_gpreagg);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_gagg,
						1,		/* kern_data_store *kds_dst */
						sizeof(cl_mem),
						&clgpa->m_kds_dst);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_gagg,
						2,		/* kern_data_store *ktoast */
						sizeof(cl_mem),
						&clgpa->m_kds_in);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_gagg,
						3,		/* pagg_hashslot *g_hashslot */
						sizeof(cl_mem),
						&clgpa->m_ghash);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_gagg,
						4,		/* KERN_DYNAMIC_LOCAL_WORKMEM_ARG */
						sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clEnqueueNDRangeKernel(clgpa->kcmdq,
								clgpa->kern_gagg,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								1,
								&clgpa->events[clgpa->ev_index - 1],
								&clgpa->events[clgpa->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		return rc;
	}
	clgpa->ev_kern_gagg = clgpa->ev_index++;
	clgpa->gpreagg->msg.pfm.num_kern_gagg++;

	return CL_SUCCESS;
}

static cl_int
clserv_launch_fixup_varlena(clstate_gpupreagg *clgpa, cl_uint nitems)
{
	size_t		gwork_sz;
	size_t		lwork_sz;
	cl_int		rc;

	/*
	 * __kernel void
	 * gpupreagg_fixup_varlena(__global kern_gpupreagg *kgpreagg,
	 *                         __global kern_data_store *kds_dst,
	 *                         __global kern_data_store *ktoast,
	 *                         KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
	 */
	clgpa->kern_fixvar = clCreateKernel(clgpa->program,
										"gpupreagg_fixup_varlena",
										&rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		return rc;
	}

	if (!clserv_compute_workgroup_size(&gwork_sz,
									   &lwork_sz,
									   clgpa->kern_fixvar,
									   clgpa->dindex,
									   true,
									   nitems,
									   sizeof(cl_uint)))
	{
		clserv_log("failed to compute optimal gwork_sz/lwork_sz");
		return StromError_OpenCLInternal;
	}

	rc = clSetKernelArg(clgpa->kern_fixvar,
						0,		/* kern_gpupreagg *kgpreagg */
						sizeof(cl_mem),
						&clgpa->m_gpreagg);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_fixvar,
						1,		/* kern_data_store *kds_dst */
						sizeof(cl_mem),
						&clgpa->m_kds_dst);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_fixvar,
						2,		/* kern_data_store *ktoast */
						sizeof(cl_mem),
						&clgpa->m_kds_in);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_fixvar,
						3,		/* KERN_DYNAMIC_LOCAL_WORKMEM_ARG */
						sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clEnqueueNDRangeKernel(clgpa->kcmdq,
								clgpa->kern_fixvar,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								1,
								&clgpa->events[clgpa->ev_index - 1],
								&clgpa->events[clgpa->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		return rc;
	}
	clgpa->ev_kern_fixvar = clgpa->ev_index++;
	clgpa->gpreagg->msg.pfm.num_kern_gagg++;

	return CL_SUCCESS;
}

static void
clserv_process_gpupreagg(pgstrom_message *message)
{
	pgstrom_gpupreagg  *gpreagg = (pgstrom_gpupreagg *) message;
	pgstrom_data_store *pds = gpreagg->pds;
	kern_data_store	   *kds = pds->kds;
	pgstrom_data_store *pds_dest = gpreagg->pds_dest;
	kern_data_store	   *kds_dest = pds_dest->kds;
	clstate_gpupreagg  *clgpa;
	kern_row_map	   *krowmap;
	cl_uint				ev_limit;
	cl_uint				nitems = kds->nitems;
	cl_uint				nvalids;
	Size				offset;
	Size				length;
	cl_int				rc;

	Assert(StromTagIs(gpreagg, GpuPreAgg));
	Assert(kds->format == KDS_FORMAT_ROW ||
		   kds->format == KDS_FORMAT_ROW_FLAT);
	Assert(kds_dest->format == KDS_FORMAT_TUPSLOT);

	/*
	 * state object of gpupreagg
	 */
	ev_limit = 50000 + kds->nblocks;
	clgpa = calloc(1, offsetof(clstate_gpupreagg, events[ev_limit]));
	if (!clgpa)
	{
		rc = CL_OUT_OF_HOST_MEMORY;
		goto error;
	}
	clgpa->gpreagg = gpreagg;
	clgpa->ev_limit = ev_limit;

	/*
	 * First of all, it looks up a program object to be run on
	 * the supplied row-store. We may have three cases.
	 * 1) NULL; it means the required program is under asynchronous
	 *    build, and the message is kept on its internal structure
	 *    to be enqueued again. In this case, we have nothing to do
	 *    any more on the invocation.
	 * 2) BAD_OPENCL_PROGRAM; it means previous compile was failed
	 *    and unavailable to run this program anyway. So, we need
	 *    to reply StromError_ProgramCompile error to inform the
	 *    backend this program.
	 * 3) valid cl_program object; it is an ideal result. pre-compiled
	 *    program object was on the program cache, and cl_program
	 *    object is ready to use.
	 */
	clgpa->program = clserv_lookup_device_program(gpreagg->dprog_key,
												  &gpreagg->msg);
	if (!clgpa->program)
	{
		free(clgpa);
		return;	/* message is in waitq, being retried later */
	}
	if (clgpa->program == BAD_OPENCL_PROGRAM)
	{
		rc = CL_BUILD_PROGRAM_FAILURE;
		goto error;
	}

	/*
	 * choose a device to run
	 */
	clgpa->dindex = pgstrom_opencl_device_schedule(&gpreagg->msg);
	clgpa->kcmdq = opencl_cmdq[clgpa->dindex];

	/*
	 * construction of kernel buffer objects
	 *
	 * m_gpreagg  - control data of gpupreagg
	 * m_kds_in   - data store of input relation stream
	 * m_kds_src  - data store of partial aggregate source
	 * m_kds_dst  - data store of partial aggregate destination
	 * m_ghash    - global hash-slot
	 */
	krowmap = KERN_GPUPREAGG_KROWMAP(&gpreagg->kern);
	nvalids = (krowmap->nvalids < 0 ? nitems : krowmap->nvalids);

	/* allocation of m_gpreagg */
	length = KERN_GPUPREAGG_BUFFER_SIZE(&gpreagg->kern, nvalids);
	clgpa->m_gpreagg = clCreateBuffer(opencl_context,
									  CL_MEM_READ_WRITE,
									  length,
									  NULL,
									  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error;
	}

	/* allocation of kds_in */
	clgpa->m_kds_in = clCreateBuffer(opencl_context,
									 CL_MEM_READ_WRITE,
									 KERN_DATA_STORE_LENGTH(kds),
									 NULL,
									 &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error;
	}
	/* allocation of kds_src */
	clgpa->m_kds_src = clCreateBuffer(opencl_context,
									  CL_MEM_READ_WRITE,
									  KERN_DATA_STORE_LENGTH(kds_dest),
									  NULL,
									  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error;
	}
	/* allocation of kds_dst */
	clgpa->m_kds_dst = clCreateBuffer(opencl_context,
									  CL_MEM_READ_WRITE,
									  KERN_DATA_STORE_LENGTH(kds_dest),
									  NULL,
									  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error;
	}
	Assert(!pds->ktoast);
	/* allocation of g_hashslot */
	length = STROMALIGN(gpreagg->kern.hash_size * sizeof(pagg_hashslot));
	clgpa->m_ghash = clCreateBuffer(opencl_context,
									CL_MEM_READ_WRITE,
									length,
									NULL,
									&rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error;
	}

	/*
	 * Next, enqueuing DMA send requests, prior to kernel execution.
	 */
	offset = KERN_GPUPREAGG_DMASEND_OFFSET(&gpreagg->kern);
	length = KERN_GPUPREAGG_DMASEND_LENGTH(&gpreagg->kern);
	rc = clEnqueueWriteBuffer(clgpa->kcmdq,
							  clgpa->m_gpreagg,
							  CL_FALSE,
							  offset,
							  length,
							  &gpreagg->kern,
							  0,
							  NULL,
							  &clgpa->events[clgpa->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error;
	}
	clgpa->ev_index++;
	gpreagg->msg.pfm.bytes_dma_send += length;
	gpreagg->msg.pfm.num_dma_send++;

	/*
	 * Enqueue DMA send on the input data-store
	 */
	rc = clserv_dmasend_data_store(pds,
								   clgpa->kcmdq,
								   clgpa->m_kds_in,
								   NULL,
								   0,
								   NULL,
								   &clgpa->ev_index,
								   clgpa->events,
								   &gpreagg->msg.pfm);
	if (rc != CL_SUCCESS)
		goto error;

	/*
	 * Also, header portion of the result data-store
	 */
	length = offsetof(kern_data_store, colmeta[kds_dest->ncols]);
	rc = clEnqueueWriteBuffer(clgpa->kcmdq,
                              clgpa->m_kds_src,
							  CL_FALSE,
							  0,
							  length,
							  kds_dest,
							  0,
							  NULL,
							  &clgpa->events[clgpa->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error;
	}
	clgpa->ev_index++;
	gpreagg->msg.pfm.bytes_dma_send += length;
	gpreagg->msg.pfm.num_dma_send++;

	rc = clEnqueueWriteBuffer(clgpa->kcmdq,
							  clgpa->m_kds_dst,
							  CL_FALSE,
							  0,
							  length,
							  kds_dest,
							  0,
							  NULL,
							  &clgpa->events[clgpa->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error;
	}
	clgpa->ev_index++;
	gpreagg->msg.pfm.bytes_dma_send += length;
	gpreagg->msg.pfm.num_dma_send++;

	/*
	 * Kick the kernel functions.
	 *
	 * Fortunatelly, gpupreagg_preparation() is always kicked on the head
	 * of this call-chain, thus, this function is responsible to synchronize
	 * DMA transfer above. Rest of kernel function needs to synchronize the
	 * previous call on itself.
	 * The last call is always gpupreagg_reduction() also, so it can be the
	 * only blocker of DMA receive.
	 */

	/* kick, gpupreagg_preparation() */
	rc = clserv_launch_preagg_preparation(clgpa, nitems);
	if (rc != CL_SUCCESS)
		goto error;

	/*
	 * kick, gpupreagg_local_reduction, or gpupreagg_init_global_hashslot
	 * instead if no local reduction is expected.
	 */
	if (gpreagg->local_reduction)
	{
		rc = clserv_launch_local_reduction(clgpa, nitems);
		if (rc != CL_SUCCESS)
			goto error;
	}
	else
	{
		rc = clserv_launch_init_hashslot(clgpa, nitems);
		if (rc != CL_SUCCESS)
			goto error;
	}
	/* finally, kick gpupreagg_global_reduction */
	rc = clserv_launch_global_reduction(clgpa, nitems);
	if (rc != CL_SUCCESS)
		goto error;

	/* finally, fixup varlena datum if any */
	if (gpreagg->has_varlena)
	{
		rc = clserv_launch_fixup_varlena(clgpa, nitems);
		if (rc != CL_SUCCESS)
			goto error;
	}

	/* writing back the result buffer */
	length = KERN_DATA_STORE_LENGTH(kds_dest);
	rc = clEnqueueReadBuffer(clgpa->kcmdq,
							 clgpa->m_kds_dst,
							 CL_FALSE,
							 0,
							 length,
							 kds_dest,
							 1,
							 &clgpa->events[clgpa->ev_index - 1],
							 &clgpa->events[clgpa->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueReadBuffer: %s",
				   opencl_strerror(rc));
		goto error;
	}
	clgpa->ev_dma_recv = clgpa->ev_index++;
	gpreagg->msg.pfm.bytes_dma_recv += length;
	gpreagg->msg.pfm.num_dma_recv++;

	/* also, status and kern_row_map has to be written back */
	offset = KERN_GPUPREAGG_DMARECV_OFFSET(&gpreagg->kern);
	length = KERN_GPUPREAGG_DMARECV_LENGTH(&gpreagg->kern, nvalids);
	rc = clEnqueueReadBuffer(clgpa->kcmdq,
							 clgpa->m_gpreagg,
							 CL_FALSE,
							 offset,
							 length,
							 &gpreagg->kern,
							 1,
							 &clgpa->events[clgpa->ev_index - 1],
							 &clgpa->events[clgpa->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueReadBuffer: %s",
				   opencl_strerror(rc));
		goto error;
	}
	clgpa->ev_index++;
	gpreagg->msg.pfm.bytes_dma_recv += length;
	gpreagg->msg.pfm.num_dma_recv++;
	Assert(clgpa->ev_index < clgpa->ev_limit);

	/*
	 * Last, registers a callback to handle post gpupreagg process
	 */
	rc = clSetEventCallback(clgpa->events[clgpa->ev_index - 1],
							CL_COMPLETE,
							clserv_respond_gpupreagg,
							clgpa);
    if (rc != CL_SUCCESS)
    {
        clserv_log("failed on clSetEventCallback: %s", opencl_strerror(rc));
        goto error;
    }
    return;

error:
	if (clgpa)
	{
		if (clgpa->ev_index > 0)
		{
			clWaitForEvents(clgpa->ev_index, clgpa->events);
			while (clgpa->ev_index > 0)
				clReleaseEvent(clgpa->events[--clgpa->ev_index]);
		}
	
		if (clgpa->m_gpreagg)
			clReleaseMemObject(clgpa->m_gpreagg);
		if (clgpa->m_kds_in)
			clReleaseMemObject(clgpa->m_kds_in);
		if (clgpa->m_kds_src)
			clReleaseMemObject(clgpa->m_kds_src);
		if (clgpa->m_kds_dst)
			clReleaseMemObject(clgpa->m_kds_dst);
		if (clgpa->m_ghash)
			clReleaseMemObject(clgpa->m_ghash);
		if (clgpa->kern_prep)
			clReleaseKernel(clgpa->kern_prep);
		if (clgpa->kern_init)
			clReleaseKernel(clgpa->kern_init);
		if (clgpa->kern_lagg)
			clReleaseKernel(clgpa->kern_lagg);
		if (clgpa->kern_gagg)
			clReleaseKernel(clgpa->kern_gagg);
		if (clgpa->kern_fixvar)
			clReleaseKernel(clgpa->kern_fixvar);
		if (clgpa->program && clgpa->program != BAD_OPENCL_PROGRAM)
			clReleaseProgram(clgpa->program);
	}
	gpreagg->msg.errcode = rc;
	pgstrom_reply_message(&gpreagg->msg);
}

/* ----------------------------------------------------------------
 *
 * NOTE: below is the function to process enhanced aggregate operations
 *
 * ---------------------------------------------------------------- */

/* gpupreagg_partial_nrows - placeholder function that generate number
 * of rows being included in this partial group.
 */
Datum
gpupreagg_partial_nrows(PG_FUNCTION_ARGS)
{
	int		i;

	for (i=0; i < PG_NARGS(); i++)
	{
		if (PG_ARGISNULL(i) || !PG_GETARG_BOOL(i))
			PG_RETURN_INT32(0);
	}
	PG_RETURN_INT32(1);
}
PG_FUNCTION_INFO_V1(gpupreagg_partial_nrows);

/* gpupreagg_pseudo_expr - placeholder function that returns the supplied
 * variable as is (even if it is NULL). Used to MIX(), MAX() placeholder.
 */
Datum
gpupreagg_pseudo_expr(PG_FUNCTION_ARGS)
{
	PG_RETURN_DATUM(PG_GETARG_DATUM(0));
}
PG_FUNCTION_INFO_V1(gpupreagg_pseudo_expr);

/* gpupreagg_psum_* - placeholder function that generates partial sum
 * of the arguments. _x2 generates square value of the input
 */
Datum
gpupreagg_psum_int(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 1);
	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	PG_RETURN_INT64(PG_GETARG_INT64(0));
}
PG_FUNCTION_INFO_V1(gpupreagg_psum_int);

Datum
gpupreagg_psum_float4(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 1);
	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	PG_RETURN_FLOAT4(PG_GETARG_FLOAT4(0));
}
PG_FUNCTION_INFO_V1(gpupreagg_psum_float4);

Datum
gpupreagg_psum_float8(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 1);
	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(PG_GETARG_FLOAT8(0));
}
PG_FUNCTION_INFO_V1(gpupreagg_psum_float8);

Datum
gpupreagg_psum_x2_float(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 1);
	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(PG_GETARG_FLOAT8(0) * PG_GETARG_FLOAT8(0));
}
PG_FUNCTION_INFO_V1(gpupreagg_psum_x2_float);

Datum
gpupreagg_psum_numeric(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 1);
	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();
	PG_RETURN_NUMERIC(PG_GETARG_NUMERIC(0));
}
PG_FUNCTION_INFO_V1(gpupreagg_psum_numeric);

Datum
gpupreagg_psum_x2_numeric(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 1);
	if (PG_ARGISNULL(0))
		PG_RETURN_NULL();

	PG_RETURN_NUMERIC(DirectFunctionCall2(numeric_mul,
										  PG_GETARG_DATUM(0),
										  PG_GETARG_DATUM(0)));
}
PG_FUNCTION_INFO_V1(gpupreagg_psum_x2_numeric);

/* gpupreagg_corr_psum - placeholder function that generates partial sum
 * of the arguments. _x2 generates square value of the input
 */
Datum
gpupreagg_corr_psum_x(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 3);
	/* Aggregate Filter */
	if (PG_ARGISNULL(0) || !PG_GETARG_BOOL(0))
		PG_RETURN_NULL();
	/* NULL checks */
	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(PG_GETARG_FLOAT8(0));
}
PG_FUNCTION_INFO_V1(gpupreagg_corr_psum_x);

Datum
gpupreagg_corr_psum_y(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 3);
	/* Aggregate Filter */
	if (PG_ARGISNULL(0) || !PG_GETARG_BOOL(0))
		PG_RETURN_NULL();
	/* NULL checks */
	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(PG_GETARG_FLOAT8(1));
}
PG_FUNCTION_INFO_V1(gpupreagg_corr_psum_y);

Datum
gpupreagg_corr_psum_x2(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 3);
	/* Aggregate Filter */
	if (PG_ARGISNULL(0) || !PG_GETARG_BOOL(0))
		PG_RETURN_NULL();
	/* NULL checks */
	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(PG_GETARG_FLOAT8(0) * PG_GETARG_FLOAT8(0));
}
PG_FUNCTION_INFO_V1(gpupreagg_corr_psum_x2);

Datum
gpupreagg_corr_psum_y2(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 3);
	/* Aggregate Filter */
	if (PG_ARGISNULL(0) || !PG_GETARG_BOOL(0))
		PG_RETURN_NULL();
	/* NULL checks */
	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(PG_GETARG_FLOAT8(1) * PG_GETARG_FLOAT8(1));
}
PG_FUNCTION_INFO_V1(gpupreagg_corr_psum_y2);

Datum
gpupreagg_corr_psum_xy(PG_FUNCTION_ARGS)
{
	Assert(PG_NARGS() == 3);
	/* Aggregate Filter */
	if (PG_ARGISNULL(0) || !PG_GETARG_BOOL(0))
		PG_RETURN_NULL();
	/* NULL checks */
	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		PG_RETURN_NULL();
	PG_RETURN_FLOAT8(PG_GETARG_FLOAT8(0) * PG_GETARG_FLOAT8(1));
}
PG_FUNCTION_INFO_V1(gpupreagg_corr_psum_xy);

/*
 * ex_avg() - an enhanced average calculation that takes two arguments;
 * number of rows in this group and partial sum of the value.
 * Then, it eventually generate mathmatically compatible average value.
 */
static int64 *
check_int64_array(ArrayType *transarray, int n)
{
	if (ARR_NDIM(transarray) != 1 ||
		ARR_DIMS(transarray)[0] != n ||
		ARR_HASNULL(transarray) ||
		ARR_ELEMTYPE(transarray) != INT8OID)
		elog(ERROR, "Two elements int8 array is expected");
	return (int64 *) ARR_DATA_PTR(transarray);
}

Datum
pgstrom_avg_int8_accum(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	int32		nrows = PG_GETARG_INT32(1);
	int64		psumX = PG_GETARG_INT64(2);
	int64	   *transvalues;
	int64		newN;
	int64		newSumX;

	transvalues = check_int64_array(transarray, 2);
	newN = transvalues[0] + nrows;
	newSumX = transvalues[1] + psumX;

	if (AggCheckCallContext(fcinfo, NULL))
	{
		transvalues[0] = newN;
		transvalues[1] = newSumX;

		PG_RETURN_ARRAYTYPE_P(transarray);
	}
	else
	{
		Datum		transdatums[2];
		ArrayType  *result;

		transdatums[0] = Int64GetDatumFast(newN);
		transdatums[1] = Int64GetDatumFast(newSumX);

		result = construct_array(transdatums, 2,
								 INT8OID,
								 sizeof(int64), FLOAT8PASSBYVAL, 'd');
		PG_RETURN_ARRAYTYPE_P(result);
	}
}
PG_FUNCTION_INFO_V1(pgstrom_avg_int8_accum);

Datum
pgstrom_sum_int8_accum(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	int64		psumX = PG_GETARG_INT64(1);
	int64	   *transvalues;
	int64		newSumX;

	transvalues = check_int64_array(transarray, 2);
	newSumX = transvalues[1] + psumX;

	if (AggCheckCallContext(fcinfo, NULL))
	{
		transvalues[0] = 0;	/* dummy */
		transvalues[1] = newSumX;

		PG_RETURN_ARRAYTYPE_P(transarray);
	}
	else
	{
		Datum		transdatums[2];
		ArrayType  *result;

		transdatums[0] = Int64GetDatumFast(0);	/* dummy */
		transdatums[1] = Int64GetDatumFast(newSumX);

		result = construct_array(transdatums, 2,
								 INT8OID,
								 sizeof(int64), FLOAT8PASSBYVAL, 'd');
		PG_RETURN_ARRAYTYPE_P(result);
	}
}
PG_FUNCTION_INFO_V1(pgstrom_sum_int8_accum);

/*
 * The built-in final sum() function that accept int8 generates numeric
 * value, but it does not fit the specification of original int2/int4.
 * So, we put our original implementation that accepet nrows(int4) and
 * partial sum (int8) then generate total sum in int8 form.
 */
Datum
pgstrom_sum_int8_final(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	int64      *transvalues;

	transvalues = check_int64_array(transarray, 2);

	PG_RETURN_INT64(transvalues[1]);
}
PG_FUNCTION_INFO_V1(pgstrom_sum_int8_final);

/*
 * numeric_agg_state - self version of aggregation internal state; that
 * can keep N, sum(X) and sum(X*X) in numeric data-type.
 */
typedef struct
{
	int64	N;
	Datum	sumX;
	Datum	sumX2;
} numeric_agg_state;

Datum
pgstrom_int8_avg_accum(PG_FUNCTION_ARGS)
{
	int32			nrows = PG_GETARG_INT32(1);
	Datum			addNum;
	MemoryContext	aggcxt;
	MemoryContext	oldcxt;
	numeric_agg_state *state;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	if (nrows < 0 || PG_ARGISNULL(1))
		elog(ERROR, "Bug? negative or NULL nrows was given");

	/* make a state object and update it */
	oldcxt = MemoryContextSwitchTo(aggcxt);
	state = PG_ARGISNULL(0) ? NULL : (numeric_agg_state *)PG_GETARG_POINTER(0);
	if (!state)
	{
		state = palloc0(sizeof(numeric_agg_state));
		state->N = 0;
		state->sumX = DirectFunctionCall3(numeric_in,
										  CStringGetDatum("0"),
										  ObjectIdGetDatum(0),
										  Int32GetDatum(-1));
	}
	if (!PG_ARGISNULL(2))
	{
		state->N += nrows;
		addNum = DirectFunctionCall1(int8_numeric, PG_GETARG_DATUM(2));
		state->sumX = DirectFunctionCall2(numeric_add, state->sumX, addNum);
	}
	MemoryContextSwitchTo(oldcxt);

	PG_RETURN_POINTER(state);
}
PG_FUNCTION_INFO_V1(pgstrom_int8_avg_accum);

Datum
pgstrom_numeric_avg_accum(PG_FUNCTION_ARGS)
{
	int32			nrows = PG_GETARG_INT32(1);
	MemoryContext	aggcxt;
	MemoryContext	oldcxt;
	numeric_agg_state *state;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");
	if (nrows < 0 || PG_ARGISNULL(1))
		elog(ERROR, "Bug? negative or NULL nrows was given");

	/* make a state object and update it */
	oldcxt = MemoryContextSwitchTo(aggcxt);
	state = PG_ARGISNULL(0) ? NULL : (numeric_agg_state *)PG_GETARG_POINTER(0);
	if (!state)
	{
		state = palloc0(sizeof(numeric_agg_state));
		state->N = 0;
		state->sumX = DirectFunctionCall3(numeric_in,
										  CStringGetDatum("0"),
										  ObjectIdGetDatum(0),
										  Int32GetDatum(-1));
	}
	if (!PG_ARGISNULL(2))
	{
		state->N += nrows;
		state->sumX = DirectFunctionCall2(numeric_add,
										  state->sumX,
										  PG_GETARG_DATUM(2));
	}
	MemoryContextSwitchTo(oldcxt);

	PG_RETURN_POINTER(state);
}
PG_FUNCTION_INFO_V1(pgstrom_numeric_avg_accum);

Datum
pgstrom_numeric_avg_final(PG_FUNCTION_ARGS)
{
	numeric_agg_state *state;
	Datum		vN;
	Datum		result;

	state = PG_ARGISNULL(0) ? NULL : (numeric_agg_state *)PG_GETARG_POINTER(0);

	/* If there were no non-null inputs, return NULL */
	if (state == NULL || state->N == 0)
		PG_RETURN_NULL();
	/* If any NaN value is accumlated, return NaN */
	if (numeric_is_nan(DatumGetNumeric(state->sumX)))
		PG_RETURN_NUMERIC(state->sumX);

	vN = DirectFunctionCall1(int8_numeric, Int64GetDatum(state->N));
	result = DirectFunctionCall2(numeric_div, state->sumX, vN);

	PG_RETURN_NUMERIC(result);
}
PG_FUNCTION_INFO_V1(pgstrom_numeric_avg_final);

/* logic copied from utils/adt/float.c */
static inline float8 *
check_float8_array(ArrayType *transarray, int nitems)
{
	/*
	 * We expect the input to be an N-element float array; verify that. We
	 * don't need to use deconstruct_array() since the array data is just
	 * going to look like a C array of N float8 values.
	 */
	if (ARR_NDIM(transarray) != 1 ||
		ARR_DIMS(transarray)[0] != nitems ||
		ARR_HASNULL(transarray) ||
		ARR_ELEMTYPE(transarray) != FLOAT8OID)
		elog(ERROR, "%d-elements float8 array is expected", nitems);
	return (float8 *) ARR_DATA_PTR(transarray);
}

/* logic copied from utils/adt/float.c */
static inline void
check_float8_valid(float8 value, bool inf_is_valid, bool zero_is_valid)
{
	if (isinf(value) && !inf_is_valid)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("value out of range: overflow")));
	if (value == 0.0 && !zero_is_valid)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("value out of range: underflow")));
}

Datum
pgstrom_sum_float8_accum(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	int32		nrows = PG_GETARG_INT32(1);
	float8		psumX = PG_GETARG_FLOAT8(2);
	float8	   *transvalues;
	float8		newN;
	float8		newSumX;

	transvalues = check_float8_array(transarray, 3);
	newN = transvalues[0] + (float8) nrows;
	newSumX = transvalues[1] + psumX;
	check_float8_valid(newSumX, isinf(transvalues[1]) || isinf(psumX), true);

	if (AggCheckCallContext(fcinfo, NULL))
	{
		transvalues[0] = newN;
		transvalues[1] = newSumX;
		transvalues[2] = 0.0;	/* dummy */

		PG_RETURN_ARRAYTYPE_P(transarray);
	}
	else
	{
		Datum		transdatums[3];
		ArrayType  *result;

		transdatums[0] = Float8GetDatumFast(newN);
		transdatums[1] = Float8GetDatumFast(newSumX);
		transdatums[2] = Float8GetDatumFast(0.0);

		result = construct_array(transdatums, 3,
								 FLOAT8OID,
								 sizeof(float8), FLOAT8PASSBYVAL, 'd');
		PG_RETURN_ARRAYTYPE_P(result);
	}
}
PG_FUNCTION_INFO_V1(pgstrom_sum_float8_accum);

/*
 * variance and stddev - mathmatical compatible result can be lead using
 * nrows, psum(X) and psum(X*X). So, we track these variables.
 */
Datum
pgstrom_variance_float8_accum(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	int32		nrows = PG_GETARG_INT32(1);
	float8		psumX = PG_GETARG_FLOAT8(2);
	float8		psumX2 = PG_GETARG_FLOAT8(3);
	float8	   *transvalues;
	float8		newN;
	float8		newSumX;
	float8		newSumX2;

	transvalues = check_float8_array(transarray, 3);
	newN = transvalues[0] + (float8) nrows;
	newSumX = transvalues[1] + psumX;
	check_float8_valid(newSumX, isinf(transvalues[1]) || isinf(psumX), true);
	newSumX2 = transvalues[2] + psumX2;
	check_float8_valid(newSumX2, isinf(transvalues[2]) || isinf(psumX2), true);

	if (AggCheckCallContext(fcinfo, NULL))
	{
		transvalues[0] = newN;
		transvalues[1] = newSumX;
		transvalues[2] = newSumX2;

		PG_RETURN_ARRAYTYPE_P(transarray);
	}
	else
	{
		Datum		transdatums[3];
		ArrayType  *result;

		transdatums[0] = Float8GetDatumFast(newN);
		transdatums[1] = Float8GetDatumFast(newSumX);
		transdatums[2] = Float8GetDatumFast(newSumX2);

		result = construct_array(transdatums, 3,
								 FLOAT8OID,
								 sizeof(float8), FLOAT8PASSBYVAL, 'd');
		PG_RETURN_ARRAYTYPE_P(result);
	}
}
PG_FUNCTION_INFO_V1(pgstrom_variance_float8_accum);

Datum
pgstrom_numeric_var_accum(PG_FUNCTION_ARGS)
{
	int32			nrows = PG_GETARG_INT32(1);
	MemoryContext	aggcxt;
	MemoryContext	oldcxt;
	numeric_agg_state *state;

	if (!AggCheckCallContext(fcinfo, &aggcxt))
		elog(ERROR, "aggregate function called in non-aggregate context");

	if (nrows < 0 || PG_ARGISNULL(1))
		elog(ERROR, "Bug? negative or NULL nrows was given");

	/* make a state object and update it */
	oldcxt = MemoryContextSwitchTo(aggcxt);
	state = PG_ARGISNULL(0) ? NULL : (numeric_agg_state *)PG_GETARG_POINTER(0);
	if (!state)
	{
		state = palloc0(sizeof(numeric_agg_state));
		state->N = 0;
		state->sumX = DirectFunctionCall3(numeric_in,
										  CStringGetDatum("0"),
										  ObjectIdGetDatum(0),
										  Int32GetDatum(-1));
		state->sumX2 = DirectFunctionCall3(numeric_in,
										   CStringGetDatum("0"),
										   ObjectIdGetDatum(0),
										   Int32GetDatum(-1));
	}
	if (!PG_ARGISNULL(2) && !PG_ARGISNULL(3))
	{
		state->N += nrows;
		state->sumX = DirectFunctionCall2(numeric_add,
										  state->sumX,
										  PG_GETARG_DATUM(2));
		state->sumX2 = DirectFunctionCall2(numeric_add,
										   state->sumX2,
										   PG_GETARG_DATUM(3));
	}
	MemoryContextSwitchTo(oldcxt);

	PG_RETURN_POINTER(state);
}
PG_FUNCTION_INFO_V1(pgstrom_numeric_var_accum);

static Numeric
pgstrom_numeric_stddev_internal(numeric_agg_state *state,
								bool variance, bool sample)
{
	Datum	vZero;
	Datum	vN;
	Datum	vN2;
	Datum	vSumX;
	Datum	vSumX2;
	Datum	result;

	if (state == NULL)
		return NULL;
	/* NaN checks */
	if (numeric_is_nan(DatumGetNumeric(state->sumX)))
		return DatumGetNumeric(state->sumX);
	if (numeric_is_nan(DatumGetNumeric(state->sumX2)))
		return DatumGetNumeric(state->sumX2);

	/*
	 * Sample stddev and variance are undefined when N <= 1; population stddev
	 * is undefined when N == 0. Return NULL in either case.
	 */
	if (sample ? state->N <= 1 : state->N <= 0)
		return NULL;

	/* const_zero = (Numeric)0 */
	vZero  = DirectFunctionCall3(numeric_in,
								 CStringGetDatum("0"),
								 ObjectIdGetDatum(0),
								 Int32GetDatum(-1));
	/* vN = (Numeric)N */
	vN = DirectFunctionCall1(int8_numeric, Int64GetDatum(state->N));
	/* vsumX = sumX * sumX */
	vSumX = DirectFunctionCall2(numeric_mul, state->sumX, state->sumX);
	/* vsumX2 = N * sumX2 */
	vSumX2 = DirectFunctionCall2(numeric_mul, state->sumX2, vN);
	/* N * sumX2 - sumX * sumX */
	vSumX2 = DirectFunctionCall2(numeric_sub, vSumX2, vSumX);

	/* Watch out for roundoff error producing a negative numerator */
	if (DirectFunctionCall2(numeric_cmp, vSumX2, vZero) <= 0)
		return DatumGetNumeric(vZero);

	if (!sample)
		vN2 = DirectFunctionCall2(numeric_mul, vN, vN);	/* N * N */
	else
	{
		Datum	vOne;
		Datum	vNminus;

		vOne = DirectFunctionCall3(numeric_in,
								   CStringGetDatum("1"),
								   ObjectIdGetDatum(0),
								   Int32GetDatum(-1));
		vNminus = DirectFunctionCall2(numeric_sub, vN, vOne);
		vN2 = DirectFunctionCall2(numeric_mul, vN, vNminus); /* N * (N - 1) */
	}
	/* variance */
	result = DirectFunctionCall2(numeric_div, vSumX2, vN2);
	/* stddev? */
	if (!variance)
		result = DirectFunctionCall1(numeric_sqrt, result);

	return DatumGetNumeric(result);
}

Datum
pgstrom_numeric_var_samp(PG_FUNCTION_ARGS)
{
	numeric_agg_state *state;
	Numeric		result;

	state = PG_ARGISNULL(0) ? NULL : (numeric_agg_state *)PG_GETARG_POINTER(0);

	result = pgstrom_numeric_stddev_internal(state, true, true);
	if (!result)
		PG_RETURN_NULL();

	PG_RETURN_NUMERIC(result);
}
PG_FUNCTION_INFO_V1(pgstrom_numeric_var_samp);

Datum
pgstrom_numeric_stddev_samp(PG_FUNCTION_ARGS)
{
	numeric_agg_state *state;
	Numeric		result;

	state = PG_ARGISNULL(0) ? NULL : (numeric_agg_state *)PG_GETARG_POINTER(0);

	result = pgstrom_numeric_stddev_internal(state, false, true);
	if (!result)
		PG_RETURN_NULL();

	PG_RETURN_NUMERIC(result);
}
PG_FUNCTION_INFO_V1(pgstrom_numeric_stddev_samp);

Datum
pgstrom_numeric_var_pop(PG_FUNCTION_ARGS)
{
	numeric_agg_state *state;
	Numeric		result;

	state = PG_ARGISNULL(0) ? NULL : (numeric_agg_state *)PG_GETARG_POINTER(0);

	result = pgstrom_numeric_stddev_internal(state, true, false);
	if (!result)
		PG_RETURN_NULL();

	PG_RETURN_NUMERIC(result);
}
PG_FUNCTION_INFO_V1(pgstrom_numeric_var_pop);

Datum
pgstrom_numeric_stddev_pop(PG_FUNCTION_ARGS)
{
	numeric_agg_state *state;
	Numeric		result;

	state = PG_ARGISNULL(0) ? NULL : (numeric_agg_state *)PG_GETARG_POINTER(0);

	result = pgstrom_numeric_stddev_internal(state, false, false);
	if (!result)
		PG_RETURN_NULL();

	PG_RETURN_NUMERIC(result);
}
PG_FUNCTION_INFO_V1(pgstrom_numeric_stddev_pop);

/*
 * covariance - mathmatical compatible result can be lead using
 * nrows, psum(X), psum(X*X), psum(Y), psum(Y*Y), psum(X*Y)
 */
Datum
pgstrom_covariance_float8_accum(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	int32		nrows  = PG_GETARG_INT32(1);
	float8		psumX  = PG_GETARG_FLOAT8(2);
	float8		psumX2 = PG_GETARG_FLOAT8(3);
	float8		psumY  = PG_GETARG_FLOAT8(4);
	float8		psumY2 = PG_GETARG_FLOAT8(5);
	float8		psumXY = PG_GETARG_FLOAT8(6);
	float8	   *transvalues;
	float8		newN;
	float8		newSumX;
	float8		newSumX2;
	float8		newSumY;
	float8		newSumY2;
	float8		newSumXY;

	transvalues = check_float8_array(transarray, 6);
	newN = transvalues[0] + (float8) nrows;
	newSumX = transvalues[1] + psumX;
	check_float8_valid(newSumX, isinf(transvalues[1]) || isinf(psumX), true);
	newSumX2 = transvalues[2] + psumX2;
	check_float8_valid(newSumX2, isinf(transvalues[2]) || isinf(psumX2), true);
	newSumY = transvalues[3] + psumY;
	check_float8_valid(newSumX, isinf(transvalues[3]) || isinf(psumY), true);
	newSumY2 = transvalues[4] + psumY2;
	check_float8_valid(newSumY2, isinf(transvalues[4]) || isinf(psumY2), true);
	newSumXY = transvalues[5] + psumXY;
	check_float8_valid(newSumXY, isinf(transvalues[5]) || isinf(psumXY), true);

	if (AggCheckCallContext(fcinfo, NULL))
	{
		transvalues[0] = newN;
		transvalues[1] = newSumX;
		transvalues[2] = newSumX2;
		transvalues[3] = newSumY;
		transvalues[4] = newSumY2;
		transvalues[5] = newSumXY;

		PG_RETURN_ARRAYTYPE_P(transarray);
	}
	else
	{
		Datum		transdatums[6];
		ArrayType  *result;

		transdatums[0] = Float8GetDatumFast(newN);
		transdatums[1] = Float8GetDatumFast(newSumX);
		transdatums[2] = Float8GetDatumFast(newSumX2);
		transdatums[3] = Float8GetDatumFast(newSumY);
		transdatums[4] = Float8GetDatumFast(newSumY2);
		transdatums[5] = Float8GetDatumFast(newSumXY);

		result = construct_array(transdatums, 6,
								 FLOAT8OID,
								 sizeof(float8), FLOAT8PASSBYVAL, 'd');
		PG_RETURN_ARRAYTYPE_P(result);
	}
}
PG_FUNCTION_INFO_V1(pgstrom_covariance_float8_accum);
