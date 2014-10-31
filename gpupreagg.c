/*
 * gpupreagg.c
 *
 * Aggregate Pre-processing with GPU acceleration
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
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
#include "utils/syscache.h"
#include <math.h>
#include "pg_strom.h"
#include "opencl_gpupreagg.h"

static CustomPlanMethods		gpupreagg_plan_methods;
static bool						enable_gpupreagg;

typedef struct
{
	CustomPlan		cplan;
	int				numCols;		/* number of grouping columns */
	AttrNumber	   *grpColIdx;		/* their indexes in the target list */
	bool			outer_bulkload;
	double			num_groups;		/* estimated number of groups */
	const char	   *kern_source;
	int				extra_flags;
	List		   *used_params;	/* referenced Const/Param */
	Bitmapset	   *outer_attrefs;	/* bitmap of referenced outer attributes */
	Bitmapset	   *tlist_attrefs;	/* bitmap of referenced tlist attributes */
} GpuPreAggPlan;

typedef struct
{
	CustomPlanState	cps;
	TupleDesc		scan_desc;
	TupleTableSlot *scan_slot;
	ProjectionInfo *bulk_proj;
	TupleTableSlot *bulk_slot;
	double			num_groups;		/* estimated number of groups */
	double			ntups_per_page;	/* average number of tuples per page */
	bool			outer_done;
	bool			outer_bulkload;
	TupleTableSlot *outer_overflow;

	pgstrom_queue  *mqueue;
	Datum			dprog_key;
	kern_parambuf  *kparams;
	bool			needs_grouping;

	pgstrom_gpupreagg  *curr_chunk;
	cl_uint			curr_index;
	bool			curr_recheck;
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
} aggfunc_catalog_t;
static aggfunc_catalog_t  aggfunc_catalog[] = {
	/* AVG(X) = EX_AVG(NROWS(), PSUM(X)) */
	{ "avg",    1, {INT2OID},
	  "s:avg",  2, {INT4OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}
	},
	{ "avg",    1, {INT4OID},
	  "s:avg",  2, {INT4OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}
	},
	{ "avg",    1, {INT8OID},
	  "s:avg_numeric",  2, {INT4OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}
	},
	{ "avg",    1, {FLOAT4OID},
	  "s:avg",  2, {INT4OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}
	},
	{ "avg",    1, {FLOAT8OID},
	  "s:avg",  2, {INT4OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_PSUM}
	},
	/* COUNT(*) = SUM(NROWS(*|X)) */
	{ "count", 0, {},       "c:sum", 1, {INT4OID}, {ALTFUNC_EXPR_NROWS}},
	{ "count", 1, {ANYOID}, "c:sum", 1, {INT4OID}, {ALTFUNC_EXPR_NROWS}},
	/* MAX(X) = MAX(PMAX(X)) */
	{ "max", 1, {INT2OID},   "c:max", 1, {INT2OID},   {ALTFUNC_EXPR_PMAX}},
	{ "max", 1, {INT4OID},   "c:max", 1, {INT4OID},   {ALTFUNC_EXPR_PMAX}},
	{ "max", 1, {INT8OID},   "c:max", 1, {INT8OID},   {ALTFUNC_EXPR_PMAX}},
	{ "max", 1, {FLOAT4OID}, "c:max", 1, {FLOAT4OID}, {ALTFUNC_EXPR_PMAX}},
	{ "max", 1, {FLOAT8OID}, "c:max", 1, {FLOAT8OID}, {ALTFUNC_EXPR_PMAX}},
	/* MIX(X) = MIN(PMIN(X)) */
	{ "min", 1, {INT2OID},   "c:min", 1, {INT2OID},   {ALTFUNC_EXPR_PMIN}},
	{ "min", 1, {INT4OID},   "c:min", 1, {INT4OID},   {ALTFUNC_EXPR_PMIN}},
	{ "min", 1, {INT8OID},   "c:min", 1, {INT8OID},   {ALTFUNC_EXPR_PMIN}},
	{ "min", 1, {FLOAT4OID}, "c:min", 1, {FLOAT4OID}, {ALTFUNC_EXPR_PMIN}},
	{ "min", 1, {FLOAT8OID}, "c:min", 1, {FLOAT8OID}, {ALTFUNC_EXPR_PMIN}},
	/* SUM(X) = SUM(PSUM(X)) */
	{ "sum", 1, {INT2OID},   "s:sum", 1, {INT8OID},   {ALTFUNC_EXPR_PSUM}},
	{ "sum", 1, {INT4OID},   "s:sum", 1, {INT8OID},   {ALTFUNC_EXPR_PSUM}},
	{ "sum", 1, {FLOAT4OID}, "c:sum", 1, {FLOAT4OID}, {ALTFUNC_EXPR_PSUM}},
	{ "sum", 1, {FLOAT8OID}, "c:sum", 1, {FLOAT8OID}, {ALTFUNC_EXPR_PSUM}},
	/* STDDEV(X) = EX_STDDEV(NROWS(),PSUM(X),PSUM(X*X)) */
	{ "stddev", 1, {FLOAT4OID},
	  "s:stddev", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}},
	{ "stddev", 1, {FLOAT8OID},
	  "s:stddev", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}},
	{ "stddev_pop", 1, {FLOAT4OID},
	  "s:stddev_pop", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}},
	{ "stddev_pop", 1, {FLOAT8OID},
	  "s:stddev_pop", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}},
	{ "stddev_samp", 1, {FLOAT4OID},
	  "s:stddev_samp", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}},
	{ "stddev_samp", 1, {FLOAT8OID},
	  "s:stddev_samp", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}},
	/* VARIANCE(X) = PGSTROM.VARIANCE(NROWS(), PSUM(X),PSUM(X^2)) */
	{ "variance", 1, {FLOAT4OID},
	  "s:variance", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}},
	{ "variance", 1, {FLOAT8OID},
	  "s:variance", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}},
	{ "var_pop", 1, {FLOAT4OID},
	  "s:var_pop", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}},
	{ "var_pop", 1, {FLOAT8OID},
	  "s:var_pop", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}},
	{ "var_samp", 1, {FLOAT4OID},
	  "s:var_samp", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}},
	{ "var_samp", 1, {FLOAT8OID},
	  "s:var_samp", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PSUM,
	   ALTFUNC_EXPR_PSUM_X2}},
	/* CORR(X,Y) = PGSTROM.CORR(NROWS(X,Y),
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
	   ALTFUNC_EXPR_PCOV_XY}},
	{ "covar_pop", 2, {FLOAT8OID, FLOAT8OID},
	  "s:covar_pop", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}},
	{ "covar_samp", 2, {FLOAT8OID, FLOAT8OID},
	  "s:covar_samp", 6,
	  {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_PCOV_X,
	   ALTFUNC_EXPR_PCOV_X2,
	   ALTFUNC_EXPR_PCOV_Y,
	   ALTFUNC_EXPR_PCOV_Y2,
	   ALTFUNC_EXPR_PCOV_XY}},
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
cost_gpupreagg(Agg *agg, GpuPreAggPlan *gpreagg,
			   List *agg_tlist, List *agg_quals,
			   Cost *p_total_cost, Cost *p_startup_cost,
			   Cost *p_total_sort, Cost *p_startup_sort)
{
	Plan	   *sort_plan = NULL;
	Plan	   *outer_plan;
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
	Path		dummy;
	AggClauseCosts agg_costs;
	ListCell   *cell;

	outer_plan = outerPlan(agg);
	if (IsA(outer_plan, Sort))
	{
		sort_plan = outer_plan;
		outer_plan = outerPlan(sort_plan);
	}

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
	 * fixed cost to kick GPU feature
	 */
	startup_cost += pgstrom_gpu_setup_cost;

	/*
	 * cost estimation of internal sorting by GPU.
	 */
	rows_per_chunk =
		((double)((pgstrom_chunk_size << 20) / BLCKSZ)) *
		((double)(BLCKSZ - MAXALIGN(SizeOfPageHeaderData))) /
        ((double)(sizeof(ItemIdData) +
                  sizeof(HeapTupleHeaderData) + outer_width));
	num_chunks = outer_rows / rows_per_chunk;
	if (num_chunks < 1.0)
		num_chunks = 1.0;

	comparison_cost = 2.0 * pgstrom_gpu_operator_cost;
	startup_cost += (comparison_cost *
					 rows_per_chunk *
					 LOG2(rows_per_chunk) *
					 num_chunks);
	run_cost += pgstrom_gpu_operator_cost * outer_rows;

	/*
	 * cost estimation of partial aggregate by GPU
	 */
	memset(&pagg_cost, 0, sizeof(QualCost));
	pagg_width = 0;
	foreach (cell, gpreagg->cplan.plan.targetlist)
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
	gpreagg->cplan.plan.startup_cost = startup_cost;
	gpreagg->cplan.plan.total_cost = startup_cost + run_cost;
	gpreagg->cplan.plan.plan_rows = num_groups * num_chunks;
	gpreagg->cplan.plan.plan_width = pagg_width;
	gpreagg->num_groups = num_groups;

	/*
	 * Update estimated sorting cost, if any.
	 * Calculation logic is cost_sort() as built-in code doing.
	 */
	if (agg->aggstrategy == AGG_SORTED)
	{
		cost_sort(&dummy,
				  NULL,		/* PlannerInfo is not referenced! */
				  NIL, 		/* NIL is acceptable */
				  gpreagg->cplan.plan.total_cost,
				  gpreagg->cplan.plan.plan_rows,
				  gpreagg->cplan.plan.plan_width,
				  0.0,
				  work_mem,
				  -1.0);
		*p_startup_sort = dummy.startup_cost;
		*p_total_sort   = dummy.total_cost;
		startup_cost    = dummy.startup_cost;
		run_cost        = dummy.total_cost - dummy.startup_cost;
	}
	else
	{
		/* to avoid compiler warning */
		*p_startup_sort = 0.0;
		*p_total_sort   = 0.0;
	}

	/*
	 * Update estimated aggregate cost.
	 * Calculation logic is cost_agg() as built-in code doing.
	 */
	memset(&agg_costs, 0, sizeof(AggClauseCosts));
	count_agg_clauses(NULL, (Node *) agg_tlist, &agg_costs);
	count_agg_clauses(NULL, (Node *) agg_quals, &agg_costs);
	cost_agg(&dummy,
			 NULL,		/* PlannerInfo is not referenced! */
			 agg->aggstrategy,
			 &agg_costs,
			 agg->numCols,
			 (double) agg->numGroups,
			 startup_cost,
			 startup_cost + run_cost,
			 gpreagg->cplan.plan.plan_rows);
	*p_startup_cost = dummy.startup_cost;
	*p_total_cost = dummy.total_cost;

	elog(INFO, "Agg cost = %.2f..%.2f nrows (%.0f => %.0f)",
		 dummy.startup_cost, dummy.total_cost,
		 outer_rows,
		 gpreagg->cplan.plan.plan_rows);
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
		defresult = copyObject(defresult);
	else
	{
		defresult = (Expr *) makeNullConst(exprType((Node *) expr),
										   exprTypmod((Node *) expr),
										   exprCollation((Node *) expr));
	}
	Assert(exprType((Node *) expr) == exprType((Node *) defresult));

	/* in case when the 'filter' is matched */
	case_when = makeNode(CaseWhen);
	case_when->expr = copyObject(filter);
	case_when->result = copyObject(expr);
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
								 args,
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
		filter = copyObject(aggref->aggfilter);
	return make_altfunc_expr(func_name, list_make3(filter, tle_1->expr, tle_2->expr));
}

/*
 * make_gpupreagg_refnode
 *
 * It tries to construct an alternative Aggref node that references
 * partially aggregated results on the target-list of GpuPreAgg node.
 */
static Aggref *
make_gpupreagg_refnode(Aggref *aggref, List **prep_tlist)
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

		altagg = make_gpupreagg_refnode(orgagg, &context->pre_tlist);
		if (!altagg)
			context->gpupreagg_invalid = true;
		return (Node *) altagg;
	}
	else if (IsA(node, Var))
	{
		Agg	   *agg = context->agg;
		Var	   *var = (Var *) node;
		int		i, x;

		if (var->varno != OUTER_VAR)
			elog(ERROR, "Bug? varnode references did not outer relation");
		for (i=0; i < agg->numCols; i++)
		{
			if (var->varattno == agg->grpColIdx[i])
			{
				x = var->varattno - FirstLowInvalidHeapAttributeNumber;
				context->attr_refs = bms_add_member(context->attr_refs, x);
				return copyObject(var);
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
					   Bitmapset **p_attr_refs)
{
	gpupreagg_rewrite_context context;
	Plan	   *outer_plan = outerPlan(agg);
	List	   *pre_tlist = NIL;
	List	   *agg_tlist = NIL;
	List	   *agg_quals = NIL;
	Bitmapset  *attr_refs = NULL;
	ListCell   *cell;
	int			i;

	/* In case of sort-aggregate, it has an underlying Sort node on top
	 * of the scan node. GpuPreAgg shall be injected under the Sort node
	 * to reduce burden of CPU sorting.
	 */
	if (agg->aggstrategy == AGG_SORTED)
	{
		Assert(IsA(outer_plan, Sort));
		outer_plan = outerPlan(outer_plan);
	}

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
			Expr		   *var;

			if (tle->resno != agg->grpColIdx[i])
				continue;

			dtype = pgstrom_devtype_lookup(type_oid);
			/* grouping key must be a supported data type */
			if (!dtype)
				return false;
			/* data type of the grouping key must have comparison function */
			if (!OidIsValid(dtype->type_cmpfunc) ||
				!pgstrom_devfunc_lookup(dtype->type_cmpfunc))
				return false;

			var = (Expr *) makeVar(OUTER_VAR,
								   tle->resno,
								   type_oid,
								   type_mod,
								   type_coll,
								   0);
			tle_new = makeTargetEntry(var,
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

	foreach (cell, agg->plan.targetlist)
	{
		TargetEntry	   *oldtle = lfirst(cell);
		TargetEntry	   *newtle = flatCopyTargetEntry(oldtle);

		newtle->expr = (Expr *)gpupreagg_rewrite_mutator((Node *)oldtle->expr,
														 &context);
		if (context.gpupreagg_invalid)
			return false;
		agg_tlist = lappend(agg_tlist, newtle);
	}

	foreach (cell, agg->plan.qual)
	{
		Expr	   *old_expr = lfirst(cell);
		Expr	   *new_expr;

		new_expr = (Expr *)gpupreagg_rewrite_mutator((Node *)old_expr,
													 &context);
		if (context.gpupreagg_invalid)
			return false;
		agg_quals = lappend(agg_quals, new_expr);
	}
	*p_pre_tlist = context.pre_tlist;
	*p_agg_tlist = agg_tlist;
	*p_agg_quals = agg_quals;
	*p_attr_refs = context.attr_refs;
	return true;
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
gpupreagg_codegen_keycomp(GpuPreAggPlan *gpreagg, codegen_context *context)
{
	StringInfoData	str;
	StringInfoData	decl;
	StringInfoData	body;
	Bitmapset  *param_refs_saved = context->param_refs;
	int			i;

	initStringInfo(&str);
	initStringInfo(&decl);
    initStringInfo(&body);
	context->param_refs = NULL;

	for (i=0; i < gpreagg->numCols; i++)
	{
		TargetEntry	   *tle;
		AttrNumber		resno = gpreagg->grpColIdx[i];
		Var			   *var;
		devtype_info   *dtype;
		devfunc_info   *dfunc;

		tle = get_tle_by_resno(gpreagg->cplan.plan.targetlist, resno);
		var = (Var *) tle->expr;
		if (!IsA(var, Var) || var->varno != OUTER_VAR)
			elog(ERROR, "Bug? A simple Var node is expected for group key: %s",
				 nodeToString(var));

		/* find a function to compare this data-type */
		/* find a datatype for comparison */
		dtype = pgstrom_devtype_lookup_and_track(var->vartype, context);
		if (!OidIsValid(dtype->type_cmpfunc))
			elog(ERROR, "Bug? type (%u) has no comparison function",
				 var->vartype);
		dfunc = pgstrom_devfunc_lookup_and_track(dtype->type_cmpfunc, context);

		/* variable declarations */
		appendStringInfo(&decl,
						 "  pg_%s_t xkeyval_%u;\n"
						 "  pg_%s_t ykeyval_%u;\n",
						 dtype->type_name, resno,
						 dtype->type_name, resno);
		/* comparison logic */
		appendStringInfo(
			&body,
			"  xkeyval_%u = pg_%s_vref(kds,ktoast,errcode,%u,x_index);\n"
			"  ykeyval_%u = pg_%s_vref(kds,ktoast,errcode,%u,y_index);\n"
			//"printf(\"gid=%%zu xnull=%%d ynull=%%d\\n\", get_global_id(0), xkeyval_1.isnull, ykeyval_1.isnull);\n"
			"  if (!xkeyval_%u.isnull && !ykeyval_%u.isnull)\n"
			"  {\n"
			//"printf(\"gid=%%zu x=%%d y=%%d\\n\", get_global_id(0), xkeyval_1.value, ykeyval_1.value);\n"
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
			dfunc->func_name, resno, resno,
			resno, resno,
			resno, resno);
	}
	/* add parameters, if referenced */
	if (context->param_refs)
	{
		char	   *params_decl
			= pgstrom_codegen_param_declarations(context,
												 context->param_refs);
		appendStringInfo(&decl, "%s", params_decl);
		pfree(params_decl);
		bms_free(context->param_refs);
	}
	context->param_refs = param_refs_saved;

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
					 "  pg_int4_t comp;\n"
					 "\n"
					 "%s"
					 "  return 0;\n"
					 "}\n",
					 decl.data,
					 body.data);
	pfree(decl.data);
	pfree(body.data);

	return str.data;
}

/*
 * gpupreagg_codegen_aggcalc - code generator of kernel gpupreagg_aggcalc();
 * that implements an operation to calculate partial aggregation.
 * The supplied accum is operated by newval, according to resno.
 *
 * static void
 * gpupreagg_aggcalc(__private cl_int *errcode,
 *                 cl_int resno,
 *                 __local pagg_datum *accum,
 *                 __local pagg_datum *newval);
 */
static inline const char *
typeoid_to_pagg_field_name(Oid type_oid)
{
	switch (type_oid)
	{
		case INT2OID:
			return "short_val";
		case INT4OID:
			return "int_val";
		case INT8OID:
			return "long_val";
		case FLOAT4OID:
			return "float_val";
		case FLOAT8OID:
			return "double_val";
	}
	elog(ERROR, "unexpected partial aggregate data-type");
}


static char *
gpupreagg_codegen_aggcalc(GpuPreAggPlan *gpreagg, codegen_context *context)
{
	Oid				namespace_oid = get_namespace_oid("pgstrom", false);
    StringInfoData	body;
	ListCell	   *cell;

    initStringInfo(&body);
	appendStringInfo(
		&body,
		"static void\n"
		"gpupreagg_aggcalc(__private cl_int *errcode,\n"
		"                  cl_int resno,\n"
		"                  __local pagg_datum *accum,\n"
		"                  __local pagg_datum *newval)\n"
		"{\n"
		"  switch (resno)\n"
		"  {\n"
		);

	/* NOTE: The targetList of GpuPreAgg are either Const (as NULL),
	 * Var node (as grouping key), or FuncExpr (as partial aggregate
	 * calculation).
	 */
	foreach (cell, gpreagg->cplan.plan.targetlist)
	{
		TargetEntry	   *tle = lfirst(cell);
		FuncExpr	   *func;
		Oid				type_oid;
		const char	   *func_name;
		const char	   *field_name;

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
			/* XXX - nrows() should not have NULL */
			field_name = typeoid_to_pagg_field_name(INT4OID);
			appendStringInfo(
				&body,
				"  case %d:\n"
				"    accum->%s += newval->%s;\n"
				"    break;\n",
				tle->resno - 1,
				field_name, field_name);
		}
		else if (strcmp(func_name, "pmax") == 0 ||
				 strcmp(func_name, "pmin") == 0)
		{
			Assert(list_length(func->args) == 1);
			type_oid = exprType(linitial(func->args));
			field_name = typeoid_to_pagg_field_name(type_oid);
			appendStringInfo(
				&body,
				"  case %d:\n"
				"    if (!newval->isnull)\n"
				"    {\n"
				"      if (accum->isnull)\n"
				"        accum->%s = newval->%s;\n"
				"      else\n"
				"        accum->%s = %s(accum->%s, newval->%s);\n"
				"      accum->isnull = false;\n"
				"    }\n"
				"    break;\n",
				tle->resno - 1,
				field_name, field_name,
				field_name, func_name + 1, field_name, field_name);
		}
		else if (strcmp(func_name, "psum") == 0    ||
				 strcmp(func_name, "psum_x2") == 0)
		{
			/* it should never be NULL */
			Assert(list_length(func->args) == 1);
			type_oid = exprType(linitial(func->args));
			field_name = typeoid_to_pagg_field_name(type_oid);
			appendStringInfo(
				&body,
				"  case %d:\n"
				"    if (!accum->isnull)\n"
				"    {\n"
				"      if (!newval->isnull)\n"
				"      {\n"
				"        if (CHECK_OVERFLOW_%s(accum->%s, newval->%s))\n"
				"          STROM_SET_ERROR(errcode, StromError_CpuReCheck);\n"
				"        accum->%s += newval->%s;\n"
				"      }\n"
				"    }\n"
				"    else if (!newval->isnull)\n"
				"    {\n"
				"      accum->%s = newval->%s;\n"
				"      accum->isnull = false;\n"
				"    }\n"
				"    break;\n",
				tle->resno - 1,
				(type_oid == FLOAT4OID ||
				 type_oid == FLOAT8OID) ? "FLOAT" : "INT",
				field_name, field_name,
				field_name, field_name,
				field_name, field_name);
		}
		else if (strcmp(func_name, "pcov_x") == 0  ||
				 strcmp(func_name, "pcov_y") == 0  ||
				 strcmp(func_name, "pcov_x2") == 0 ||
				 strcmp(func_name, "pcov_y2") == 0 ||
				 strcmp(func_name, "pcov_xy") == 0)
		{
			/* covariance takes only float8 datatype */
			/* it should never be NULL */
			field_name = typeoid_to_pagg_field_name(FLOAT8OID);
			appendStringInfo(
				&body,
				"  case %d:\n"
				"    if (!accum->isnull)\n"
				"    {\n"
				"      if (!newval->isnull)\n"
				"      {\n"
				"        if (CHECK_OVERFLOW_FLOAT(accum->%s, newval->%s))\n"
				"          STROM_SET_ERROR(errcode, StromError_CpuReCheck);\n"
				"        accum->%s += newval->%s;\n"
				"      }\n"
				"    }\n"
				"    else if (!newval->isnull)\n"
				"    {\n"
				"      accum->%s = newval->%s;\n"
				"      accum->isnull = false;\n"
				"    }\n"
				"    break;\n",
				tle->resno - 1,
				field_name, field_name,
				field_name, field_name,
				field_name, field_name);
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
static char *
gpupreagg_codegen_projection(GpuPreAggPlan *gpreagg, codegen_context *context)
{
	Oid				namespace_oid = get_namespace_oid("pgstrom", false);
	List		   *targetlist = gpreagg->cplan.plan.targetlist;
	StringInfoData	str;
	StringInfoData	decl1;
	StringInfoData	decl2;
    StringInfoData	body;
	const char	   *kds_label = "kds_src";
	const char	   *ktoast_label = "kds_in";
	const char	   *rowidx_label = "rowidx_out";
	Expr		   *clause;
	ListCell	   *cell;
	Bitmapset	   *attr_refs = NULL;
	Bitmapset	   *param_refs_saved = context->param_refs;
	devtype_info   *dtype;
	devfunc_info   *dfunc;
	Plan		   *outer_plan;
	bool			use_temp_int4 = false;
	bool			use_temp_int8 = false;
	bool			use_temp_float8x = false;
	bool			use_temp_float8y = false;
	struct varlena *vl_datum;
	Const		   *kparam_0;
	cl_char		   *gpagg_atts;
	Size			length;

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
		TargetEntry	   *tle = lfirst(cell);

		if (IsA(tle->expr, Var))
		{
			Var	   *var = (Var *) tle->expr;

			Assert(var->varno == OUTER_VAR);
			Assert(var->varattno > 0);
			attr_refs = bms_add_member(attr_refs, var->varattno -
									   FirstLowInvalidHeapAttributeNumber);
			dtype = pgstrom_devtype_lookup_and_track(var->vartype, context);
			appendStringInfo(
				&body,
				"  /* projection for resource %u */\n"
				"  pg_%s_vstore(%s,%s,errcode,%u,%s,KVAR_%u);\n",
				tle->resno - 1,
				dtype->type_name,
				kds_label,
				ktoast_label,
				tle->resno - 1,
				rowidx_label,
				var->varattno);
			/* track usage of this field */
			gpagg_atts[tle->resno - 1] = GPUPREAGG_FIELD_IS_GROUPKEY;
		}
		else if (IsA(tle->expr, Const))
		{
			/* assign NULL value */
			appendStringInfo(
				&body,
				"  /* projection for resource %u */\n"
				"  pg_common_vstore(%s,%s,errcode,%u,%s,true);\n",
				tle->resno - 1,
				kds_label,
				ktoast_label,
				tle->resno - 1,
				rowidx_label);
		}
		else if (IsA(tle->expr, FuncExpr))
		{
			FuncExpr   *func = (FuncExpr *) tle->expr;
			const char *func_name;

			appendStringInfo(&body,
							 "  /* projection for resource %u */\n",
							 tle->resno - 1);
			if (namespace_oid != get_func_namespace(func->funcid))
				elog(ERROR, "Bug? unexpected FuncExpr: %s",
					 nodeToString(func));

			pull_varattnos((Node *)func, OUTER_VAR, &attr_refs);

			func_name = get_func_name(func->funcid);
			if (strcmp(func_name, "nrows") == 0)
			{
				dtype = pgstrom_devtype_lookup_and_track(INT4OID, context);
				use_temp_int4 = true;
				appendStringInfo(&body, "  temp_int4.isnull = false;\n");
				if (list_length(func->args) > 0)
				{
					ListCell   *lc;

					appendStringInfo(&body, "  if (");
					foreach (lc, func->args)
					{
						if (lc != list_head(func->args))
							appendStringInfo(&body,
											 " &&\n"
											 "      ");
						appendStringInfo(&body, "EVAL(%s)",
										 pgstrom_codegen_expression(lfirst(lc),
																	context));
					}
					appendStringInfo(&body,
									 ")\n"
									 "    temp_int4.value = 1;\n"
									 "  else\n"
									 "    temp_int4.value = 0;\n");
				}
				else
					appendStringInfo(&body, "  temp_int4.value = 1;\n");
				appendStringInfo(
					&body,
					"  pg_%s_vstore(%s,%s,errcode,%u,%s,temp_int4);\n",
					dtype->type_name,
					kds_label,
					ktoast_label,
					tle->resno - 1,
					rowidx_label);
			}
			else if (strcmp(func_name, "pmax") == 0 ||
					 strcmp(func_name, "pmin") == 0 ||
					 strcmp(func_name, "psum") == 0)
			{
				/* Store the original value as-is. In case when clause is
				 * conditional and its result is false, NULL shall be set
				 * on "pmax" and "pmin", or 0 shall be set on "psum".
				 */
				Oid		type_oid;

				Assert(list_length(func->args) == 1);
				clause = linitial(func->args);
				type_oid = exprType((Node *)clause);
				dtype = pgstrom_devtype_lookup_and_track(type_oid, context);
				Assert(dtype != NULL);

				appendStringInfo(
					&body,
					"  pg_%s_vstore(%s,%s,errcode,%u,%s,%s);\n",
					dtype->type_name,
					kds_label,
					ktoast_label,
					tle->resno - 1,
					rowidx_label,
					pgstrom_codegen_expression((Node *)clause,
											   context));
			}
			else if (strcmp(func_name, "psum_x2") == 0)
			{
				Assert(exprType(linitial(func->args)) == FLOAT8OID);
				clause = linitial(func->args);
				use_temp_float8x = true;
				dfunc = pgstrom_devfunc_lookup_and_track(F_FLOAT8MUL, context);
				dtype = pgstrom_devtype_lookup_and_track(FLOAT8OID, context);

				appendStringInfo(&body,
								 "  temp_float8x = %s;\n",
								 pgstrom_codegen_expression((Node *)clause,
															context));
				appendStringInfo(
					&body,
					"  pg_%s_vstore(%s,%s,errcode,%u,%s,\n"
					"               pgfn_%s(errcode, temp_float8x,\n"
					"                                temp_float8x));\n",
					dtype->type_name,
					kds_label,
					ktoast_label,
					tle->resno - 1,
					rowidx_label,
					dfunc->func_name);
			}
			else if (strcmp(func_name, "pcov_x") == 0 ||
					 strcmp(func_name, "pcov_y") == 0 ||
					 strcmp(func_name, "pcov_x2") == 0 ||
					 strcmp(func_name, "pcov_y2") == 0 ||
					 strcmp(func_name, "pcov_xy") == 0)
			{
				Expr   *filter = linitial(func->args);
				Expr   *x_clause = lsecond(func->args);
				Expr   *y_clause = lthird(func->args);

				use_temp_float8x = use_temp_float8y = true;

				if (IsA(filter, Const))
				{
					Const  *cons = (Const *) filter;
					if (cons->consttype == BOOLOID &&
						!cons->constisnull &&
						DatumGetBool(cons->constvalue))
						filter = NULL;	/* no filter, actually */
				}
				appendStringInfo(
					&body,
					"  temp_float8x = %s;\n"
					"  temp_float8y = %s;\n",
					pgstrom_codegen_expression((Node *) x_clause, context),
					pgstrom_codegen_expression((Node *) y_clause, context));
				appendStringInfo(
					&body,
					"  if (temp_float8x.isnull ||\n"
					"      temp_float8y.isnull");
				if (filter)
					appendStringInfo(
						&body,
						" ||\n"
						"      !EVAL(%s)",
						pgstrom_codegen_expression((Node *) filter, context));
				appendStringInfo(
					&body,
					")\n"
					"  {\n"
					"    temp_float8x.isnull = true;\n"
					"    temp_float8x.value = 0.0;\n"
					"  }\n");
				/* initial value according to the function */
				if (strcmp(func_name, "pcov_y") == 0)
				{
					appendStringInfo(
						&body,
						"  else\n"
						"    temp_float8x = temp_float8y;\n");
				}
				else if (strcmp(func_name, "pcov_x2") == 0)
				{
					dfunc = pgstrom_devfunc_lookup_and_track(F_FLOAT8MUL,
															 context);
					appendStringInfo(
						&body,
						"  else\n"
						"    temp_float8x = pgfn_%s(errcode,\n"
						"                           temp_float8x,\n"
						"                           temp_float8x);\n",
						dfunc->func_name);
				}
				else if (strcmp(func_name, "pcov_y2") == 0)
				{
					dfunc = pgstrom_devfunc_lookup_and_track(F_FLOAT8MUL,
                                                             context);
					appendStringInfo(
						&body,
						"  else\n"
						"    temp_float8x = pgfn_%s(errcode,\n"
						"                           temp_float8y,\n"
						"                           temp_float8y);\n",
						dfunc->func_name);
				}
				else if (strcmp(func_name, "pcov_xy") == 0)
				{
					dfunc = pgstrom_devfunc_lookup_and_track(F_FLOAT8MUL,
                                                             context);
					appendStringInfo(
						&body,
						"  else\n"
						"    temp_float8x = pgfn_%s(errcode,\n"
						"                           temp_float8x,\n"
						"                           temp_float8y);\n",
						dfunc->func_name);
				}
				else if (strcmp(func_name, "pcov_x") != 0)
					elog(ERROR, "unexpected partial covariance function: %s",
						 func_name);

				dtype = pgstrom_devtype_lookup_and_track(FLOAT8OID, context);
				appendStringInfo(
					&body,
					"  pg_%s_vstore(%s,%s,errcode,%u,%s,temp_float8x);\n",
					dtype->type_name,
					kds_label,
					ktoast_label,
					tle->resno - 1,
					rowidx_label);
			}
			else 
				elog(ERROR, "Bug? unexpected partial aggregate function: %s",
					 func_name);
			/* track usage of this field */
			gpagg_atts[tle->resno - 1] = GPUPREAGG_FIELD_IS_AGGFUNC;
		}
		else
			elog(ERROR, "bug? unexpected node type: %s",
				 nodeToString(tle->expr));
	}

	/*
	 * Declaration of variables
	 */
	outer_plan = outerPlan(gpreagg);
	if (gpreagg->outer_bulkload)
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
			= pgstrom_codegen_param_declarations(context,
												 context->param_refs);
		appendStringInfo(&decl2, "%s", params_decl);
		pfree(params_decl);
		bms_free(context->param_refs);
	}
	context->param_refs = param_refs_saved;

	/* declaration of other temp variables */
	if (use_temp_int4)
		appendStringInfo(&decl1, "  pg_int4_t temp_int4;\n");
	if (use_temp_int8)
		appendStringInfo(&decl1, "  pg_int8_t temp_int8;\n");
	if (use_temp_float8x)
		appendStringInfo(&decl1, "  pg_float8_t temp_float8x;\n");
	if (use_temp_float8y)
		appendStringInfo(&decl1, "  pg_float8_t temp_float8y;\n");

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
gpupreagg_codegen(GpuPreAggPlan *gpreagg, codegen_context *context)
{
	StringInfoData	str;
	const char	   *fn_keycomp;
	const char	   *fn_aggcalc;
	const char	   *fn_projection;

	pgstrom_init_codegen_context(context);
	/*
	 * System constants of GpuPreAgg:
	 * KPARAM_0 is an array of cl_char to inform which field is grouping
	 * keys, or target of (partial) aggregate function.
	 */
	context->used_params = list_make1(makeNullConst(BYTEAOID, -1, InvalidOid));
	context->type_defs = list_make1(pgstrom_devtype_lookup(BYTEAOID));

	/* generate a key comparison function */
	fn_keycomp = gpupreagg_codegen_keycomp(gpreagg, context);
	/* generate a partial aggregate function */
	fn_aggcalc = gpupreagg_codegen_aggcalc(gpreagg, context);
	/* generate an initial data loading function */
	fn_projection = gpupreagg_codegen_projection(gpreagg, context);

	/* OK, add type/function declarations */
	initStringInfo(&str);
	appendStringInfo(&str,
					 "%s\n"		/* type declarations */
					 "%s\n"		/* function declarations */
					 "%s\n"		/* gpupreagg_keycomp() */
					 "%s\n"		/* gpupreagg_aggcalc() */
					 "%s\n",	/* gpupreagg_projection() */
					 pgstrom_codegen_type_declarations(context),
					 pgstrom_codegen_func_declarations(context),
					 fn_keycomp,
					 fn_aggcalc,
					 fn_projection);
	return str.data;
}

/*
 * gpupreagg_use_bulkload
 *
 * It checks availability of bulk-loading
 */
static bool
gpupreagg_use_bulkload(GpuPreAggPlan *gpreagg)
{
	Plan	   *outer_plan = outerPlan(gpreagg);
	List	   *tlist = gpreagg->cplan.plan.targetlist;
	Bitmapset  *attrefs = NULL;
	int			i, resno;

	/*
	 * Only GpuScan and GpuHashJoin support bulk-loading right now.
	 */
	if (!pgstrom_plan_is_gpuscan(outer_plan) &&
		!pgstrom_plan_is_gpuhashjoin(outer_plan))
		return false;

	pull_varattnos((Node *)tlist, OUTER_VAR, &attrefs);
	while ((i = bms_first_member(attrefs)) >= 0)
	{
		TargetEntry	   *tle;

		resno = i + FirstLowInvalidHeapAttributeNumber;
		if (resno  < 1)
			elog(ERROR, "Bug? system column should not be in GpuPreAgg");

		tle = get_tle_by_resno(outer_plan->targetlist, resno);
		if (!pgstrom_codegen_available_expression(tle->expr))
			return false;
	}
	return true;
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
	GpuPreAggPlan  *gpreagg;
	List		   *pre_tlist = NIL;
	List		   *agg_quals = NIL;
	List		   *agg_tlist = NIL;
	Bitmapset	   *attr_refs = NULL;
	ListCell	   *cell;
	Cost			startup_cost;
	Cost			total_cost;
	Cost			startup_sort;
	Cost			total_sort;
	const char	   *kern_source;
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
								&pre_tlist,
								&attr_refs))
		return;

	/*
	 * Try to construct a GpuPreAggPlan node.
	 * If aggregate node contains any unsupported expression, we give up
	 * to insert GpuPreAgg node.
	 */
	gpreagg = palloc0(sizeof(GpuPreAggPlan));
	NodeSetTag(gpreagg, T_CustomPlan);
	gpreagg->cplan.methods = &gpupreagg_plan_methods;
	gpreagg->cplan.plan.targetlist = pre_tlist;
	gpreagg->cplan.plan.qual = NIL;
	gpreagg->numCols = agg->numCols;
	gpreagg->grpColIdx = pmemcpy(agg->grpColIdx,
								 sizeof(AttrNumber) * agg->numCols);
	if (!IsA(outerPlan(agg), Sort))
		outerPlan(gpreagg) = gpuscan_try_replace_seqscan_plan(pstmt,
															  outerPlan(agg),
															  attr_refs);
	else
	{
		Sort   *sort_plan = (Sort *) outerPlan(agg);
		Plan   *outer_plan = outerPlan(sort_plan);

		outerPlan(gpreagg) = gpuscan_try_replace_seqscan_plan(pstmt,
															  outer_plan,
															  attr_refs);
	}
	/* XXX - gpupreagg_use_bulkload references outer plan */
	gpreagg->outer_bulkload = gpupreagg_use_bulkload(gpreagg);

	/*
	 * Cost estimation of Aggregate node if GpuPreAgg is injected.
	 * Then, compare the two cases. If partial aggregate does not
	 * help the overall aggregation, nothing to do. Elsewhere, we
	 * inject a GpuPreAgg node under the Agg (or Sort) node to
	 * reduce number of rows to be processed.
	 */
	cost_gpupreagg(agg, gpreagg,
				   agg_tlist, agg_quals,
				   &startup_cost, &total_cost,
				   &startup_sort, &total_sort);
#if 0
	if (agg->plan.total_cost < total_cost)
		return;
#endif
	/*
	 * construction of kernel code, according to the above query
	 * rewrites.
	 */
	kern_source = gpupreagg_codegen(gpreagg, &context);
	gpreagg->kern_source = kern_source;
	gpreagg->extra_flags = context.extra_flags |
		DEVKERNEL_NEEDS_GPUPREAGG |
		(!devprog_enable_optimize ? DEVKERNEL_DISABLE_OPTIMIZE : 0);
	gpreagg->used_params = context.used_params;
	pull_varattnos((Node *)context.used_vars,
				   OUTER_VAR,
				   &gpreagg->outer_attrefs);
	foreach (cell, gpreagg->cplan.plan.targetlist)
	{
		TargetEntry	   *tle = lfirst(cell);

		if (IsA(tle->expr, Const))
			continue;
		gpreagg->tlist_attrefs =
			bms_add_member(gpreagg->tlist_attrefs, tle->resno -
						   FirstLowInvalidHeapAttributeNumber);
	}

	/* OK, inject it */
	agg->plan.startup_cost = startup_cost;
	agg->plan.total_cost = total_cost;
	agg->plan.targetlist = agg_tlist;
	agg->plan.qual = agg_quals;

	if (!IsA(outerPlan(agg), Sort))
		outerPlan(agg) = &gpreagg->cplan.plan;
	else
	{
		Sort   *sort_plan = (Sort *) outerPlan(agg);

		Assert(IsA(sort_plan, Sort));
		sort_plan->plan.startup_cost = startup_sort;
		sort_plan->plan.total_cost = total_sort;
		sort_plan->plan.plan_rows = gpreagg->cplan.plan.plan_rows;
		sort_plan->plan.plan_width = gpreagg->cplan.plan.plan_width;
		sort_plan->plan.targetlist = copyObject(pre_tlist);
		outerPlan(sort_plan) = &gpreagg->cplan.plan;
	}
}

static CustomPlanState *
gpupreagg_begin(CustomPlan *node, EState *estate, int eflags)
{
	GpuPreAggPlan  *gpreagg = (GpuPreAggPlan *) node;
	GpuPreAggState *gpas;
	int				outer_width;
	Const		   *kparam_0;

	/*
	 * construct a state structure
	 */
	gpas = palloc0(sizeof(GpuPreAggState));
	NodeSetTag(gpas, T_CustomPlanState);
	gpas->cps.ps.plan = &node->plan;
	gpas->cps.ps.state = estate;
	gpas->cps.methods = &gpupreagg_plan_methods;

	/*
	 * create expression context
	 */
	ExecAssignExprContext(estate, &gpas->cps.ps);

	/*
	 * initialize child expression
	 */
	gpas->cps.ps.targetlist = (List *)
		ExecInitExpr((Expr *) node->plan.targetlist, &gpas->cps.ps);
	gpas->cps.ps.qual = NIL;	/* never has qualifier here */

	/*
	 * initialize child node
	 */
	outerPlanState(gpas) = ExecInitNode(outerPlan(gpreagg), estate, eflags);
	gpas->scan_desc = ExecGetResultType(outerPlanState(gpas));
	gpas->scan_slot = ExecAllocTableSlot(&estate->es_tupleTable);
	ExecSetSlotDescriptor(gpas->scan_slot, gpas->scan_desc);
	gpas->outer_bulkload = gpreagg->outer_bulkload;
	gpas->outer_done = false;
	gpas->outer_overflow = NULL;

	outer_width = outerPlanState(gpas)->plan->plan_width;
	gpas->num_groups = gpreagg->num_groups;
	gpas->ntups_per_page =
		((double)(BLCKSZ - MAXALIGN(SizeOfPageHeaderData))) /
		((double)(sizeof(ItemIdData) +
				  sizeof(HeapTupleHeaderData) + outer_width));

	/*
	 * initialize result tuple type and projection info
	 */
	ExecInitResultTupleSlot(estate, &gpas->cps.ps);
	ExecAssignResultTypeFromTL(&gpas->cps.ps);
	if (tlist_matches_tupdesc(&gpas->cps.ps,
							  gpas->cps.ps.plan->targetlist,
							  OUTER_VAR,
							  gpas->scan_desc))
		gpas->cps.ps.ps_ProjInfo = NULL;
	else
		gpas->cps.ps.ps_ProjInfo =
			ExecBuildProjectionInfo(gpas->cps.ps.targetlist,
									gpas->cps.ps.ps_ExprContext,
									gpas->cps.ps.ps_ResultTupleSlot,
									gpas->scan_desc);

	if (gpas->outer_bulkload)
	{
		if (pgstrom_plan_is_gpuscan(outerPlan(gpreagg)))
			pgstrom_gpuscan_setup_bulkslot(outerPlanState(gpas),
										   &gpas->bulk_proj,
										   &gpas->bulk_slot);
		else if (pgstrom_plan_is_gpuhashjoin(outerPlan(gpreagg)))
			pgstrom_gpuhashjoin_setup_bulkslot(outerPlanState(gpas),
											   &gpas->bulk_proj,
											   &gpas->bulk_slot);
		else
			elog(ERROR, "Bug? unexpected PlanState node");
	}

	/*
	 * construction of kern_parambuf template; including system param of
	 * GPUPREAGG_FIELD_IS_* array.
	 * NOTE: we don't modify gpreagg->used_params here, so no need to
	 * make a copy.
	 */
	Assert(list_length(gpreagg->used_params) >= 1);
	kparam_0 = (Const *) linitial(gpreagg->used_params);
	Assert(IsA(kparam_0, Const) &&
		   kparam_0->consttype == BYTEAOID &&
		   !kparam_0->constisnull);
	gpas->kparams = pgstrom_create_kern_parambuf(gpreagg->used_params,
												 gpas->cps.ps.ps_ExprContext);
	/*
	 * Setting up kernel program and message queue
	 */
	gpas->dprog_key = pgstrom_get_devprog_key(gpreagg->kern_source,
											  gpreagg->extra_flags);
	pgstrom_track_object((StromObject *)gpas->dprog_key, 0);
	gpas->mqueue = pgstrom_create_queue();
	pgstrom_track_object(&gpas->mqueue->sobj, 0);

	/*
	 * init misc stuff
	 */
	gpas->needs_grouping = (gpreagg->numCols > 0);
	gpas->curr_chunk = NULL;
	gpas->curr_index = 0;
	gpas->curr_recheck = false;
	dlist_init(&gpas->ready_chunks);

	/*
	 * Is perfmon needed?
	 */
	gpas->pfm.enabled = pgstrom_perfmon_enabled;

	return &gpas->cps;
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
	 */
	required = STROMALIGN(offsetof(pgstrom_gpupreagg,
								   kern.kparams) +
						  gpas->kparams->length);
	if (nvalids < 0)
		required += STROMALIGN(offsetof(kern_row_map, rindex[0]));
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
	gpupreagg->needs_grouping = gpas->needs_grouping;
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
	gpupreagg->kern.sortbuf_len =
		(1UL << get_next_log2(nvalids < 0 ? nitems : nvalids));
	/* refresh kparams */
	kparams = KERN_GPUPREAGG_PARAMBUF(&gpupreagg->kern);
	memcpy(kparams, gpas->kparams, gpas->kparams->length);

	/*
	 * Also, initialization of kern_row_map portion
	 */
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
	tupdesc = gpas->cps.ps.ps_ResultTupleSlot->tts_tupleDescriptor;
	pds_dest = pgstrom_create_data_store_tupslot(tupdesc, (nvalids < 0
														   ? nitems
														   : nvalids));
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
				pds = pgstrom_create_data_store_row(tupdesc,
													pgstrom_chunk_size << 20,
													gpas->ntups_per_page);
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
		bulk = (pgstrom_bulkslot *) MultiExecProcNode(subnode);
        if (!bulk)
			gpas->outer_done = true;
	}
	if (gpas->pfm.enabled)
	{
		gettimeofday(&tv2, NULL);
		gpas->pfm.time_outer_load += timeval_diff(&tv1, &tv2);
	}
	if (bulk)
		gpupreagg = pgstrom_create_gpupreagg(gpas, bulk);

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
	kern_row_map	   *krowmap = KERN_GPUPREAGG_KROWMAP(&gpreagg->kern);
	TupleTableSlot	   *slot = NULL;
	TupleTableSlot	   *slot_in;
	cl_uint				row_index;
	HeapTupleData		tuple;

	/* bulk-load uses individual slot; then may have a projection */
	if (!gpas->outer_bulkload)
		slot_in = gpas->scan_slot;
	else
		slot_in = gpas->bulk_slot;

retry:
	/*
	 * identify the kds_index to be fetched
	 */
	if (krowmap->nvalids < 0)
	{
		if (gpas->curr_index >= kds->nitems)
			return NULL;
		row_index = gpas->curr_index++;
	}
	else
	{
		if (gpas->curr_index >= krowmap->nvalids)
			return NULL;
		row_index = krowmap->rindex[gpas->curr_index++];
	}

	/*
	 * Fetch a tuple from the data-store
	 */
	if (pgstrom_fetch_data_store(slot_in, pds, row_index, &tuple))
	{
		ProjectionInfo *projection = gpas->cps.ps.ps_ProjInfo;
		ExprContext	   *econtext = gpas->cps.ps.ps_ExprContext;
		ExprDoneCond	is_done;

		/* reset per-tuple memory context */
		ResetExprContext(econtext);

		/* additional projection if bulk-load required */
		if (gpas->outer_bulkload)
		{
			if (gpas->bulk_proj)
			{
				econtext->ecxt_outertuple = slot_in;
				gpas->scan_slot = ExecProject(gpas->bulk_proj, &is_done);
				if (is_done == ExprEndResult)
				{
					slot = NULL;
					goto retry;
				}
				slot_in = gpas->scan_slot;
			}
		}
		/* put result tuple */
		if (!projection)
		{
			slot = gpas->cps.ps.ps_ResultTupleSlot;
			ExecCopySlot(slot, slot_in);
		}
		else
		{
			econtext->ecxt_outertuple = slot_in;
			slot = ExecProject(projection, &is_done);
			if (is_done == ExprEndResult)
			{
				slot = NULL;
				goto retry;
			}
			gpas->cps.ps.ps_TupFromTlist = (is_done == ExprMultipleResult);
		}
	}
	return slot;
}

static TupleTableSlot *
gpupreagg_next_tuple(GpuPreAggState *gpas)
{
	pgstrom_gpupreagg  *gpreagg = gpas->curr_chunk;
	pgstrom_data_store *pds_dest = gpreagg->pds_dest;
	TupleTableSlot	   *slot = NULL;
	HeapTupleData		tuple;
	struct timeval		tv1, tv2;

	if (gpas->pfm.enabled)
		gettimeofday(&tv1, NULL);

	if (gpas->curr_recheck)
		slot = gpupreagg_next_tuple_fallback(gpas);
	else
	{
		slot = gpas->cps.ps.ps_ResultTupleSlot;
		if (!pgstrom_fetch_data_store(slot, pds_dest,
									  gpas->curr_index++,
									  &tuple))
			slot = NULL;
	}

	if (gpas->pfm.enabled)
	{
		gettimeofday(&tv2, NULL);
		gpas->pfm.time_materialize += timeval_diff(&tv1, &tv2);
	}
	return slot;
}

static TupleTableSlot *
gpupreagg_exec(CustomPlanState *node)
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
		 * dequeue the running gpupreagg chunk being already processed.
		 */
		while ((msg = pgstrom_try_dequeue_message(gpas->mqueue)) != NULL)
		{
			Assert(gpas->num_running > 0);
			gpas->num_running--;
			dlist_push_tail(&gpas->ready_chunks, &msg->chain);
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
			gpas->curr_recheck = true;	/* fallback by CPU */
		else if (gpreagg->msg.errcode == CL_BUILD_PROGRAM_FAILURE)
		{
			const char *buildlog
				= pgstrom_get_devprog_errmsg(gpas->dprog_key);
			const char *kern_source
				= ((GpuPreAggPlan *)node->ps.plan)->kern_source;

			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("PG-Strom: OpenCL execution error (%s)\n%s",
							pgstrom_strerror(gpreagg->msg.errcode),
							kern_source),
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
gpupreagg_end(CustomPlanState *node)
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

	/* Clean out the tuple table */
	ExecClearTuple(node->ps.ps_ResultTupleSlot);
	ExecClearTuple(gpas->scan_slot);

	/* Free the exprcontext */
    ExecFreeExprContext(&node->ps);
}

static void
gpupreagg_rescan(CustomPlanState *node)
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
gpupreagg_explain(CustomPlanState *node, List *ancestors, ExplainState *es)
{
	GpuPreAggState *gpas = (GpuPreAggState *) node;

	ExplainPropertyText("Bulkload",
						gpas->outer_bulkload ? "On" : "Off", es);
	show_device_kernel(gpas->dprog_key, es);
	if (es->analyze && gpas->pfm.enabled)
		pgstrom_perfmon_explain(&gpas->pfm, es);
}

static Bitmapset *
gpupreagg_get_relids(CustomPlanState *node)
{
	/* nothing to do in GpuPreAgg */
	return NULL;
}

static void
gpupreagg_textout_plan(StringInfo str, const CustomPlan *node)
{
	GpuPreAggPlan  *plannode = (GpuPreAggPlan *) node;
	int				i;

	appendStringInfo(str, " :numCols %u", plannode->numCols);

	appendStringInfo(str, " :grpColIdx [");
	for (i=0; i < plannode->numCols; i++)
		appendStringInfo(str, " %u", plannode->grpColIdx[i]);
	appendStringInfo(str, "]");

	appendStringInfo(str, " :kern_source ");
	_outToken(str, plannode->kern_source);

	appendStringInfo(str, " :extra_flags %u", plannode->extra_flags);

	appendStringInfo(str, " :used_params %s",
					 nodeToString(plannode->used_params));

	appendStringInfo(str, " :outer_attrefs ");
	_outBitmapset(str, plannode->outer_attrefs);

	appendStringInfo(str, " :tlist_attrefs ");
	_outBitmapset(str, plannode->tlist_attrefs);
}

static CustomPlan *
gpupreagg_copy_plan(const CustomPlan *from)
{
	GpuPreAggPlan  *oldnode = (GpuPreAggPlan *) from;
	GpuPreAggPlan  *newnode;

	newnode = palloc0(sizeof(GpuPreAggPlan));
	CopyCustomPlanCommon((Node *) oldnode, (Node *) newnode);
	newnode->numCols       = oldnode->numCols;
	newnode->grpColIdx     = pmemcpy(oldnode->grpColIdx,
									 sizeof(AttrNumber) * oldnode->numCols);
	newnode->kern_source   = pstrdup(oldnode->kern_source);
	newnode->extra_flags   = oldnode->extra_flags;
	newnode->used_params   = copyObject(oldnode->used_params);
	newnode->outer_attrefs = bms_copy(oldnode->outer_attrefs);
	newnode->tlist_attrefs = bms_copy(oldnode->tlist_attrefs);

	return &newnode->cplan;
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

	/* initialization of plan method table */
	memset(&gpupreagg_plan_methods, 0, sizeof(CustomPlanMethods));
	gpupreagg_plan_methods.CustomName          = "GpuPreAgg";
	gpupreagg_plan_methods.BeginCustomPlan     = gpupreagg_begin;
	gpupreagg_plan_methods.ExecCustomPlan      = gpupreagg_exec;
	gpupreagg_plan_methods.EndCustomPlan       = gpupreagg_end;
	gpupreagg_plan_methods.ReScanCustomPlan    = gpupreagg_rescan;
	gpupreagg_plan_methods.ExplainCustomPlan   = gpupreagg_explain;
	gpupreagg_plan_methods.GetRelidsCustomPlan = gpupreagg_get_relids;
	gpupreagg_plan_methods.TextOutCustomPlan   = gpupreagg_textout_plan;
	gpupreagg_plan_methods.CopyCustomPlan      = gpupreagg_copy_plan;
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
	cl_kernel			kern_set_rindex;
	cl_kernel		   *kern_sort;
	cl_kernel			kern_pagg;
	cl_int				kern_sort_nums;	/* number of sorting kernel */
	cl_mem				m_gpreagg;
	cl_mem				m_kds_in;	/* kds of input relation */
	cl_mem				m_kds_src;	/* kds of aggregation source */
	cl_mem				m_kds_dst;	/* kds of aggregation results */
	//cl_mem				m_ktoast;
	cl_uint				ev_kern_prep;	/* event index of kern_prep */
	cl_uint				ev_kern_pagg;	/* event index of kern_pagg */
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
		 * Prep kernel execution time
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
		 * Sort kernel execution time
		 */
		tv_start = ~0UL;
		tv_end = 0;
		for (i=clgpa->ev_kern_prep + 1; i < clgpa->ev_kern_pagg; i++)
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
		gpreagg->msg.pfm.time_kern_sort += (tv_end - tv_start) / 1000;

		/*
		 * Main kernel execution time
		 */
		i = clgpa->ev_kern_pagg;
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
		gpreagg->msg.pfm.time_kern_exec += (tv_end - tv_start) / 1000;

		/*
		 * DMA recv time - last two event should be DMA receive request
		 */
		tv_start = ~0UL;
		tv_end = 0;
		for (i=2; i > 0; i--)
		{
			rc = clGetEventProfilingInfo(clgpa->events[clgpa->ev_index - i],
										 CL_PROFILING_COMMAND_START,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			tv_start = Min(tv_start, temp);

			rc = clGetEventProfilingInfo(clgpa->events[clgpa->ev_index - i],
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
	if (clgpa->kern_prep)
		clReleaseKernel(clgpa->kern_prep);
	if (clgpa->kern_set_rindex)
		clReleaseKernel(clgpa->kern_set_rindex);
	for (i=0; i < clgpa->kern_sort_nums; i++)
		clReleaseKernel(clgpa->kern_sort[i]);
	if (clgpa->kern_pagg)
		clReleaseKernel(clgpa->kern_pagg);
	if (clgpa->program && clgpa->program != BAD_OPENCL_PROGRAM)
		clReleaseProgram(clgpa->program);
	if (clgpa->kern_sort)
		free(clgpa->kern_sort);
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
	 *                       __global kern_data_store *ktoast,
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
clserv_launch_set_rindex(clstate_gpupreagg *clgpa, cl_uint nvalids)
{
	cl_int		rc;
	size_t		gwork_sz;
	size_t		lwork_sz;


	/* Return without dispatch the kernel function if no data in the chunk.
	 */
	if (nvalids == 0)
		return CL_SUCCESS;

	/* __kernel void
	 * gpupreagg_set_rindex(__global kern_gpupreagg *kgpreagg,
	 *                      __global kern_data_store *kds,
	 *                      __local void *local_memory)
	 */
	clgpa->kern_set_rindex = clCreateKernel(clgpa->program,
											"gpupreagg_set_rindex",
											&rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		return rc;
	}

	/* calculation of workgroup size with assumption of a device thread
	 * consums "sizeof(cl_int)" local memory per thread.
	 */
	if (!clserv_compute_workgroup_size(&gwork_sz, &lwork_sz,
									   clgpa->kern_set_rindex,
									   clgpa->dindex,
									   true,
									   nvalids,
									   sizeof(cl_uint)))
	{
		clserv_log("failed to compute optimal gwork_sz/lwork_sz");
		return StromError_OpenCLInternal;
	}

	rc = clSetKernelArg(clgpa->kern_set_rindex,
						0,		/* __global kern_gpupreagg *kgpreagg */
						sizeof(cl_mem),
						&clgpa->m_gpreagg);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_set_rindex,
						1,		/* __global kern_data_store *kds_src */
						sizeof(cl_mem),
						&clgpa->m_kds_src);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_set_rindex,
						2,		/* __local void *local_memory */
						sizeof(cl_int) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clEnqueueNDRangeKernel(clgpa->kcmdq,
								clgpa->kern_set_rindex,
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
	clgpa->ev_index++;
	clgpa->gpreagg->msg.pfm.num_kern_sort++;

	return CL_SUCCESS;
}

static cl_int
clserv_launch_bitonic_local(clstate_gpupreagg *clgpa,
							size_t gwork_sz, size_t lwork_sz)
{
	cl_kernel	kernel;
	cl_int		rc;

	/* __kernel void
	 * gpupreagg_bitonic_local(__global kern_gpupreagg *kgpreagg,
	 *                         __global kern_data_store *kds,
	 *                         __global kern_data_store *ktoast,
	 *                         __local void *local_memory)
	 */
	kernel = clCreateKernel(clgpa->program,
							"gpupreagg_bitonic_local",
							&rc);
	if (rc != CL_SUCCESS)
    {
        clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
        return rc;
    }
	clgpa->kern_sort[clgpa->kern_sort_nums++] = kernel;

	rc = clSetKernelArg(kernel,
						0,		/* __kern_gpupreagg *kgpreagg */
						sizeof(cl_mem),
						&clgpa->m_gpreagg);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						1,      /* __global kern_data_store *kds */
						sizeof(cl_mem),
						&clgpa->m_kds_src);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						2,		/* __global kern_data_store *ktoast */
						sizeof(cl_mem),
						&clgpa->m_kds_in);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						3,		/* __local void *local_memory */
						2 * sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}
	rc = clEnqueueNDRangeKernel(clgpa->kcmdq,
								kernel,
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
	clgpa->ev_index++;
	clgpa->gpreagg->msg.pfm.num_kern_sort++;

	return CL_SUCCESS;
}

static cl_int
clserv_launch_bitonic_step(clstate_gpupreagg *clgpa,
						   bool reversing, cl_uint unitsz, size_t work_sz)
{
	cl_kernel	kernel;
	cl_int		bitonic_unitsz;
	cl_int		rc;
	size_t		gwork_sz;
	size_t		lwork_sz;

	/*
	 * __kernel void
	 * gpupreagg_bitonic_step(__global kern_gpupreagg *kgpreagg,
	 *                        cl_int bitonic_unitsz,
	 *                        __global kern_data_store *kds,
	 *                        __global kern_data_store *ktoast,
	 *                        __local void *local_memory)
	 */
	kernel = clCreateKernel(clgpa->program,
							"gpupreagg_bitonic_step",
							&rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		return rc;
	}
	clgpa->kern_sort[clgpa->kern_sort_nums++] = kernel;

	if (!clserv_compute_workgroup_size(&gwork_sz, &lwork_sz,
									   clgpa->kern_pagg,
									   clgpa->dindex,
									   false,
									   work_sz,
									   sizeof(int)))
	{
		clserv_log("failed to compute optimal gwork_sz/lwork_sz");
		return StromError_OpenCLInternal;
	}

	rc = clSetKernelArg(kernel,
						0,		/* __kern_gpupreagg *kgpreagg */
						sizeof(cl_mem),
						&clgpa->m_gpreagg);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	/*
	 * NOTE: bitonic_unitsz informs kernel function the unit size of
	 * sorting block and its direction. Sign of the value indicates
	 * the direction, and absolute value indicates the sorting block
	 * size. For example, -5 means reversing direction (because of
	 * negative sign), and 32 (= 2^5) for sorting block size.
	 */
	bitonic_unitsz = (!reversing ? unitsz : -unitsz);
	rc = clSetKernelArg(kernel,
						1,	/* cl_int bitonic_unitsz */
						sizeof(cl_int),
						&bitonic_unitsz);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						2,      /* __global kern_data_store *kds */
						sizeof(cl_mem),
						&clgpa->m_kds_src);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						3,		/* __global kern_data_store *ktoast */
						sizeof(cl_mem),
						&clgpa->m_kds_in);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						4,		/* __local void *local_memory */
						sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clEnqueueNDRangeKernel(clgpa->kcmdq,
								kernel,
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
	clgpa->ev_index++;
	clgpa->gpreagg->msg.pfm.num_kern_sort++;

	return CL_SUCCESS;
}

static cl_int
clserv_launch_bitonic_merge(clstate_gpupreagg *clgpa,
							size_t gwork_sz, size_t lwork_sz)
{
	cl_kernel	kernel;
	cl_int		rc;

	/* __kernel void
	 * gpupreagg_bitonic_merge(__global kern_gpupreagg *kgpreagg,
	 *                         __global kern_data_store *kds,
	 *                         __global kern_data_store *ktoast,
	 *                         __local void *local_memory)
	 */
	kernel = clCreateKernel(clgpa->program,
							"gpupreagg_bitonic_merge",
							&rc);
	if (rc != CL_SUCCESS)
    {
        clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
        return rc;
    }
	clgpa->kern_sort[clgpa->kern_sort_nums++] = kernel;

	rc = clSetKernelArg(kernel,
						0,		/* __kern_gpupreagg *kgpreagg */
						sizeof(cl_mem),
						&clgpa->m_gpreagg);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						1,      /* __global kern_data_store *kds */
						sizeof(cl_mem),
						&clgpa->m_kds_src);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						2,		/* __global kern_data_store *ktoast */
						sizeof(cl_mem),
						&clgpa->m_kds_in);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						3,		/* __local void *local_memory */
						2 * sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}
	rc = clEnqueueNDRangeKernel(clgpa->kcmdq,
								kernel,
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
	clgpa->ev_index++;
	clgpa->gpreagg->msg.pfm.num_kern_sort++;

	return CL_SUCCESS;
}

static cl_int
clserv_launch_preagg_reduction(clstate_gpupreagg *clgpa, cl_uint nvalids)
{
	cl_int		rc;
	size_t		gwork_sz;
	size_t		lwork_sz;

	/* __kernel void
	 * gpupreagg_reduction(__global kern_gpupreagg *kgpreagg,
	 *                     __global kern_data_store *kds_src,
	 *                     __global kern_data_store *kds_dst,
	 *                     __global kern_data_store *ktoast,
	 *                     __local void *local_memory)
	 */
	clgpa->kern_pagg = clCreateKernel(clgpa->program,
									  "gpupreagg_reduction",
									  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		return rc;
	}

	/* calculation of workgroup size with assumption of a device thread
	 * consums "sizeof(pagg_datum) + sizeof(cl_uint)" local memory per
	 * thread, that is larger than usual cl_uint cases.
	 */
	if (!clserv_compute_workgroup_size(&gwork_sz, &lwork_sz,
									   clgpa->kern_pagg,
									   clgpa->dindex,
									   true,
									   nvalids,
									   sizeof(pagg_datum)))
	{
		clserv_log("failed to compute optimal gwork_sz/lwork_sz");
		return StromError_OpenCLInternal;
	}

	rc = clSetKernelArg(clgpa->kern_pagg,
						0,		/* __global kern_gpupreagg *kgpreagg */
						sizeof(cl_mem),
						&clgpa->m_gpreagg);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_pagg,
						1,		/* __global kern_data_store *kds_src */
						sizeof(cl_mem),
						&clgpa->m_kds_src);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_pagg,
						2,		/* __global kern_data_store *kds_dst */
						sizeof(cl_mem),
						&clgpa->m_kds_dst);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_pagg,
						3,		/* __global kern_data_store *ktoast */
						sizeof(cl_mem),
						&clgpa->m_kds_in);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgpa->kern_pagg,
						4,		/* __local void *local_memory */
						sizeof(pagg_datum) * lwork_sz + STROMALIGN_LEN,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clEnqueueNDRangeKernel(clgpa->kcmdq,
								clgpa->kern_pagg,
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
	clgpa->ev_kern_pagg = clgpa->ev_index++;
	clgpa->gpreagg->msg.pfm.num_kern_exec++;

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
	cl_uint				nitems = kds->nitems;
	cl_uint				nvalids;
	Size				offset;
	Size				length;
	size_t				gwork_sz = 0;
	size_t				lwork_sz = 0;
	size_t				gsort_sz;
	cl_int				i, rc;

	Assert(StromTagIs(gpreagg, GpuPreAgg));
	Assert(kds->format == KDS_FORMAT_ROW ||
		   kds->format == KDS_FORMAT_ROW_FLAT);
	Assert(kds_dest->format == KDS_FORMAT_TUPSLOT);

	/*
	 * state object of gpupreagg
	 */
	clgpa = calloc(1, offsetof(clstate_gpupreagg,
							   events[50000 + 10 * kds->nblocks]));
	if (!clgpa)
	{
		rc = CL_OUT_OF_HOST_MEMORY;
		goto error;
	}
	clgpa->gpreagg = gpreagg;

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
	 */
	krowmap = KERN_GPUPREAGG_KROWMAP(&gpreagg->kern);
	nvalids = (krowmap->nvalids < 0 ? nitems : krowmap->nvalids);

	/* allocation of m_gpreagg */
	length = KERN_GPUPREAGG_BUFFER_SIZE(&gpreagg->kern);
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
	 * calculation of gwork_sz/lwork_sz for bitonic sorting.
	 * it consume sizeof(cl_uint) for each workitem.
	 */
	if (!clserv_compute_workgroup_size(&gwork_sz, &lwork_sz, NULL,
									   clgpa->dindex, true,
									   nvalids, sizeof(cl_uint)))
	{
		clserv_log("failed to compute optimal gwork_sz/lwork_sz");
        rc = StromError_OpenCLInternal;
		goto error;
	}

	/*
	 * bitonic sort using,
	 *  gpupreagg_bitonic_step()
	 *  gpupreagg_bitonic_local()
	 *  gpupreagg_bitonic_merge()
	 */
	if (!gpreagg->needs_grouping)
	{

		rc = clserv_launch_set_rindex(clgpa, nvalids);
		if (rc != CL_SUCCESS)
			goto error;
	}
	else if (nvalids > 0)
	{
		const pgstrom_device_info *devinfo =
			pgstrom_get_device_info(clgpa->dindex);
		size_t max_lwork_sz    = devinfo->dev_max_work_item_sizes[0];
		size_t max_lmem_sz     = devinfo->dev_local_mem_size;
		size_t lmem_per_thread = 2 * sizeof(cl_int);
		size_t nhalf           = (nvalids + 1) / 2;

		size_t gwork_sz, lwork_sz, i, j, nsteps, launches;

		lwork_sz = Min(nhalf, max_lwork_sz);
		lwork_sz = Min(max_lwork_sz, max_lmem_sz/lmem_per_thread);
		lwork_sz = 1 << get_next_log2(lwork_sz);

		gwork_sz = ((nhalf + lwork_sz - 1) / lwork_sz) * lwork_sz;

		nsteps   = get_next_log2(nhalf / lwork_sz) + 1;
		launches = (nsteps + 1) * nsteps / 2 + nsteps + 1;

		clgpa->kern_sort = calloc(launches, sizeof(cl_kernel));
		if (clgpa->kern_sort == NULL) {
			goto error;
		}
		/* Adjust size of global sorting */
		gsort_sz = 2 * nhalf;
		//clserv_log("gsort_sz=%zu nvalids=%u num_groups=%f", gsort_sz, nvalids, gpreagg->num_groups);
		while (gsort_sz > 2 * lwork_sz)
		{
			/* FIXME: fixed-threshold is not preferable in all cases.
			 * if available, we need to pay attention on the balance
			 * between row reduction ratio and expected sorting cost.
			 */
			if (((double) gsort_sz / gpreagg->num_groups) < 20.0)
				break;
			gsort_sz /= 2;
		}

		/* Sort key in each local work group */
        rc = clserv_launch_bitonic_local(clgpa, gwork_sz, lwork_sz);
		if (rc != CL_SUCCESS)
			goto error;

		/* Sort key value between inter work group. */
		for(i=lwork_sz*2; i < gsort_sz; i*=2)
		{
			for(j=i; lwork_sz<j; j/=2)
			{
				cl_uint unitsz    = 2 * j;
				bool	reversing = (j == i) ? true : false;
				size_t	work_sz   = (((nvalids + unitsz - 1) / unitsz) 
									 * unitsz / 2);

				rc = clserv_launch_bitonic_step(clgpa, reversing, unitsz, 
												work_sz);
				if (rc != CL_SUCCESS)
					goto error;
			}
			rc = clserv_launch_bitonic_merge(clgpa, gwork_sz, lwork_sz);
			if (rc != CL_SUCCESS)
				goto error;
		}
	}

	/* kick, gpupreagg_reduction() */
	rc = clserv_launch_preagg_reduction(clgpa, nvalids);
	if (rc != CL_SUCCESS)
		goto error;

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
	clgpa->ev_index++;
	gpreagg->msg.pfm.bytes_dma_recv += length;
	gpreagg->msg.pfm.num_dma_recv++;

	/* also, this sizeof(cl_int) bytes for result status */
	offset = KERN_GPUPREAGG_DMARECV_OFFSET(&gpreagg->kern);
	length = KERN_GPUPREAGG_DMARECV_LENGTH(&gpreagg->kern);
	rc = clEnqueueReadBuffer(clgpa->kcmdq,
							 clgpa->m_gpreagg,
							 CL_FALSE,
							 offset,
							 length,
							 &gpreagg->kern.status,
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
		if (clgpa->kern_prep)
			clReleaseKernel(clgpa->kern_prep);
		if (clgpa->kern_set_rindex)
			clReleaseKernel(clgpa->kern_set_rindex);
		for (i=0; i < clgpa->kern_sort_nums; i++)
			clReleaseKernel(clgpa->kern_sort[i]);
		if (clgpa->kern_pagg)
			clReleaseKernel(clgpa->kern_pagg);
		if (clgpa->program && clgpa->program != BAD_OPENCL_PROGRAM)
			clReleaseProgram(clgpa->program);
		if (clgpa->kern_sort)
			free(clgpa->kern_sort);
		free(clgpa);
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
 * pgstrom_avg_numeric_accum - It keeps an internal state using
 * NumericAggState for int8, to prevent overflow. Unlike built-in
 * implementation, it also takes "nrows" arguments in addition to
 * the partial sum.
 */
/* copy from numeric.c */
typedef struct NumericAggState
{
	bool		calcSumX2;		/* if true, calculate sumX2 */
	MemoryContext agg_context;	/* context we're calculating in */
	int64		N;				/* count of processed numbers */
#if 0
	NumericVar	sumX;			/* sum of processed numbers */
	NumericVar	sumX2;			/* sum of squares of processed numbers */
	int			maxScale;		/* maximum scale seen so far */
	int64		maxScaleCount;	/* number of values seen with maximum scale */
	int64		NaNcount;		/* count of NaN values (not included in N!) */
#endif
} NumericAggState;

Datum
pgstrom_avg_numeric_accum(PG_FUNCTION_ARGS)
{
	int32	nrows = PG_GETARG_INT32(1);
	Datum	datum;
	NumericAggState *state;

	/* nrows should be a non-null and non-negative input */
	if (PG_ARGISNULL(1) || nrows < 0)
		elog(ERROR, "Bug? NULL or negative nrows was given");

	/* adjust argument */
	fcinfo->nargs      = 2;
	fcinfo->arg[1]     = fcinfo->arg[2];
	fcinfo->argnull[1] = fcinfo->argnull[2];

	datum = int8_avg_accum(fcinfo);

	state = (NumericAggState *) DatumGetPointer(datum);
	if (state && nrows > 0)
		state->N += nrows - 1;
	PG_RETURN_POINTER(state);
}
PG_FUNCTION_INFO_V1(pgstrom_avg_numeric_accum);

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
