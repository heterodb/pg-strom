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
#include "catalog/pg_aggregate.h"
#include "catalog/pg_type.h"
#include "catalog/pg_proc.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/syscache.h"
#include <math.h>
#include "pg_strom.h"

static CustomPlanMethods		gpupreagg_plan_methods;
static bool						enable_gpupreagg;

typedef struct
{
	CustomPlan		cplan;
	const char	   *kernel_source;
	int				extra_flags;

} GpuPreAggPlan;

/*
 * Arguments of alternative functions.
 * An alternative functions 
 *
 */
#define ALTFUNC_EXPR_NROWS				101	/* PSUM(1) */
#define ALTFUNC_EXPR_NROWS_NOTNULL		102	/* PSUM((X IS NOT NULL)::int) */
#define ALTFUNC_EXPR_MIN_ARG1			103	/* PMIN(X) */
#define ALTFUNC_EXPR_MAX_ARG1			104	/* PMAX(X) */
#define ALTFUNC_EXPR_SUM_ARG1			105	/* PSUM(X) */
#define ALTFUNC_EXPR_SUM_ARG2			106	/* PSUM(Y) */
#define ALTFUNC_EXPR_SUM_ARG1_MUL_ARG1	107	/* PSUM(X*X) */
#define ALTFUNC_EXPR_SUM_ARG2_MUL_ARG2	108	/* PSUM(Y*Y) */
#define ALTFUNC_EXPR_SUM_ARG1_MUL_ARG2	109	/* PSUM(X*Y) */




/*
 * List of supported aggregate functions
 */
typedef struct {
	/* aggregate function can be preprocessed */
	const char *aggfn_name;
	int			aggfn_nargs;
	Oid			aggfn_argtypes[4];
	/* alternative function to generate same result */
	const char *altfn_name;
	int			altfn_nargs;
	Oid			altfn_argtypes[8];
	int			altfn_argexprs[8];
} aggfunc_catalog_t;
static aggfunc_catalog_t  aggfunc_catalog[] = {
	/* AVG(X) = EX_AVG(NROWS(), PSUM(X)) */
	{ "avg",    1, {INT2OID},
	  "ex_avg", 2, {INT4OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_SUM_ARG1}
	},
	{ "avg",    1, {INT4OID},
	  "ex_avg", 2, {INT4OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_SUM_ARG1}
	},
	{ "avg",    1, {INT8OID},
	  "ex_avg", 2, {INT4OID, INT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_SUM_ARG1}
	},
	{ "avg",    1, {FLOAT4OID},
	  "ex_avg", 2, {INT4OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_SUM_ARG1}
	},
	{ "avg",    1, {FLOAT8OID},
	  "ex_avg", 2, {INT4OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS, ALTFUNC_EXPR_SUM_ARG1}
	},
	/* COUNT(*) = SUM(PSUM(1)) */
	{ "count", 0, {},       "sum", 1, {INT4OID}, {ALTFUNC_EXPR_NROWS}},
	/* COUNT(X) = SUM(PSUM((X IS NOT NULL)::int)) */
	{ "count", 1, {ANYOID}, "sum", 1, {INT4OID}, {ALTFUNC_EXPR_NROWS_NOTNULL}},
	/* MAX(X) = MAX(PMAX(X)) */
	{ "max", 1, {INT2OID},   "max", 1, {INT2OID},   {ALTFUNC_EXPR_MAX_ARG1}},
	{ "max", 1, {INT4OID},   "max", 1, {INT4OID},   {ALTFUNC_EXPR_MAX_ARG1}},
	{ "max", 1, {INT8OID},   "max", 1, {INT8OID},   {ALTFUNC_EXPR_MAX_ARG1}},
	{ "max", 1, {FLOAT4OID}, "max", 1, {FLOAT4OID}, {ALTFUNC_EXPR_MAX_ARG1}},
	{ "max", 1, {FLOAT8OID}, "max", 1, {FLOAT8OID}, {ALTFUNC_EXPR_MAX_ARG1}},
	/* MIX(X) = MIN(PMIN(X)) */
	{ "min", 1, {INT2OID},   "min", 1, {INT2OID},   {ALTFUNC_EXPR_MIN_ARG1}},
	{ "min", 1, {INT4OID},   "min", 1, {INT4OID},   {ALTFUNC_EXPR_MIN_ARG1}},
	{ "min", 1, {INT8OID},   "min", 1, {INT8OID},   {ALTFUNC_EXPR_MIN_ARG1}},
	{ "min", 1, {FLOAT4OID}, "min", 1, {FLOAT4OID}, {ALTFUNC_EXPR_MIN_ARG1}},
	{ "min", 1, {FLOAT8OID}, "min", 1, {FLOAT8OID}, {ALTFUNC_EXPR_MIN_ARG1}},
	/* SUM(X) = SUM(PSUM(X)) */
	{ "sum", 1, {INT2OID}, "ex_sum", 1, {INT8OID}, {ALTFUNC_EXPR_SUM_ARG1}},
	{ "sum", 1, {INT4OID}, "ex_sum", 1, {INT8OID}, {ALTFUNC_EXPR_SUM_ARG1}},
	{ "sum", 1, {FLOAT4OID}, "sum", 1, {FLOAT8OID}, {ALTFUNC_EXPR_SUM_ARG1}},
	{ "sum", 1, {FLOAT8OID}, "sum", 1, {FLOAT8OID}, {ALTFUNC_EXPR_SUM_ARG1}},
	/* STDDEV(X) = EX_STDDEV(NROWS(),PSUM(X),PSUM(X*X)) */
	{ "stddev", 1, {FLOAT4OID},
	  "ex_stddev", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_SUM_ARG1,
	   ALTFUNC_EXPR_SUM_ARG1_MUL_ARG1}},
	{ "stddev", 1, {FLOAT8OID},
	  "ex_stddev", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_SUM_ARG1,
	   ALTFUNC_EXPR_SUM_ARG1_MUL_ARG1}},
	{ "stddev_pop", 1, {FLOAT4OID},
	  "ex_stddev_pop", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_SUM_ARG1,
	   ALTFUNC_EXPR_SUM_ARG1_MUL_ARG1}},
	{ "stddev_pop", 1, {FLOAT8OID},
	  "ex_stddev_pop", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_SUM_ARG1,
	   ALTFUNC_EXPR_SUM_ARG1_MUL_ARG1}},
	{ "stddev_samp", 1, {FLOAT4OID},
	  "ex_stddev_samp", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_SUM_ARG1,
	   ALTFUNC_EXPR_SUM_ARG1_MUL_ARG1}},
	{ "stddev_samp", 1, {FLOAT8OID},
	  "ex_stddev_samp", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_SUM_ARG1,
	   ALTFUNC_EXPR_SUM_ARG1_MUL_ARG1}},
	/* VARIANCE(X) = EX_VARIANCE(NROWS(), PSUM(X),PSUM(X*X)) */
	{ "variance", 1, {FLOAT4OID},
	  "ex_variance", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_SUM_ARG1,
	   ALTFUNC_EXPR_SUM_ARG1_MUL_ARG1}},
	{ "variance", 1, {FLOAT8OID},
	  "ex_variance", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_SUM_ARG1,
	   ALTFUNC_EXPR_SUM_ARG1_MUL_ARG1}},
	{ "var_pop", 1, {FLOAT4OID},
	  "ex_var_pop", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_SUM_ARG1,
	   ALTFUNC_EXPR_SUM_ARG1_MUL_ARG1}},
	{ "var_samp", 1, {FLOAT8OID},
	  "ex_var_samp", 3, {INT4OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
	   ALTFUNC_EXPR_SUM_ARG1,
	   ALTFUNC_EXPR_SUM_ARG1_MUL_ARG1}},
	/* CORR(X,Y) = EX_CORR(NROWS(), PSUM(X), PSUM(Y), PSUM(X*Y)) */
	{ "corr", 2, {FLOAT8OID},
	  "ex_corr", 4, {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_SUM_ARG1,
       ALTFUNC_EXPR_SUM_ARG2,
	   ALTFUNC_EXPR_SUM_ARG1_MUL_ARG2}},
	{ "covar_pop", 2, {FLOAT8OID},
	  "ex_covar_pop", 4, {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_SUM_ARG1,
       ALTFUNC_EXPR_SUM_ARG2,
	   ALTFUNC_EXPR_SUM_ARG1_MUL_ARG2}},
	{ "covar_samp", 2, {FLOAT8OID},
	  "ex_covar_samp", 4, {INT4OID, FLOAT8OID, FLOAT8OID, FLOAT8OID},
	  {ALTFUNC_EXPR_NROWS,
       ALTFUNC_EXPR_SUM_ARG1,
       ALTFUNC_EXPR_SUM_ARG2,
	   ALTFUNC_EXPR_SUM_ARG1_MUL_ARG2}},
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
 * is_aggregate_supported - It checks whether the supplied Aggregate node
 * is consists of only supported features. If OK, we can move to the cost
 * estimation and query rewriting.
 */
static bool
is_aggregate_supported(Agg *agg)
{
	ListCell   *lc1;
	ListCell   *lc2;
	int			i;

	/*
	 * check targetlist - if aggregate function, it has to be described
	 * on the catalog. elsewhere, it has to be supported expression.
	 */
	foreach (lc1, agg->plan.targetlist)
	{
		TargetEntry *tle = lfirst(lc1);

		Assert(IsA(tle, TargetEntry));

		if (IsA(tle->expr, Aggref))
		{
			Aggref *aggref = (Aggref *) tle->expr;

			/* aggregate function has to be supported */
			if (!aggfunc_lookup_by_oid(aggref->aggfnoid))
				return false;
			/* also, its arguments have to be supported expression */
			foreach (lc2, aggref->args)
			{
				TargetEntry *argtle = lfirst(lc2);
				Assert(IsA(argtle, TargetEntry));

				if (!pgstrom_codegen_available_expression(argtle->expr))
					return false;
			}
		}
		/*
		 * NOTE: no need to check grouped variable reference on the target-
		 * list, because its calculation shall be done on the Agg node of
		 * host side. All we need to pay attention is sorting the group-
		 * keys and calculation of partial result of aggregate function.
		 */
	}

	/*
	 * check group-by operators - comparison function has be supported to
	 * run on the device side.
	 */
	for (i=0; i < agg->numCols; i++)
	{
		Oid		opfnoid = get_opcode(agg->grpOperators[i]);

		if (!OidIsValid(opfnoid))
			elog(ERROR, "cache lookup failed for operator: %u",
				 agg->grpOperators[i]);

		if (!pgstrom_devfunc_lookup(opfnoid))
			return false;
	}
	/* OK, move to the cost estimation */
	return true;
}

/*
 * cost_gpupreagg
 *
 * cost estimation of Aggregate if GpuPreAgg is injected
 */
#define LOG2(x)		(log(x) / 0.693147180559945)

static Cost
cost_gpupreagg(PlannedStmt *pstmt, Agg *agg)
{
	Plan	   *sort_plan = NULL;
	Plan	   *outer_plan;
	Cost		startup_cost;
	Cost		run_cost;
	Cost		comparison_cost;
	double		rows_per_chunk;
	double		num_chunks;
	

	if (agg->aggstrategy == AGG_SORTED)
	{
		sort_plan = outerPlan(agg);
		Assert(IsA(sort_plan, Sort));
		outer_plan = outerPlan(sort_plan);
	}
	else
	{
		outer_plan = outerPlan(agg);
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
	num_rows = outer_plan->plan_rows;

	/*
	 * fixed cost to kick GPU feature
	 */
	startup_cost += pgstrom_gpu_setup_cost;

	/*
	 * cost estimation of internal sorting by GPU.
	 */
	if ((double)NUM_ROWS_PER_COLSTORE >= num_rows)
	{
		rows_per_chunk = num_rows;
		num_chunks = 1.0;
	}
	else
	{
		rows_per_chunk = (double) NUM_ROWS_PER_COLSTORE;
		num_chunks = num_rows / (double) NUM_ROWS_PER_COLSTORE + 1.0;
	}
	comparison_cost = 2.0 * pgstrom_gpu_operator_cost;
	startup_cost += (comparison_cost *
					 rows_per_chunk *
					 LOG2(rows_per_chunk) *
					 num_chunks);
	run_cost += gpu_operator_cost * num_rows;

	/*
	 * cost estimation of partial aggregate by GPU
	 */




	nrows = outer_plan->plan_rows;








}



static Expr *
make_altfunc_expr_typecast(Expr *expr, Oid target_type_oid)
{
	Oid				source_type_oid = exprType(expr);
	HeapTuple		tuple;
	Form_pg_cast	cast_form;

	tuple = SearchSysCache2(CASTSOURCETARGET,
                            ObjectIdGetDatum(source_type_oid),
							ObjectIdGetDatum(target_type_oid));
	if (!HeapTupleIsValid(tuple))
		return NULL;
	cast_form = (Form_pg_cast) GETSTRUCT(tuple);
	Assert(cast_form->castsource == source_type_oid);
	Assert(cast_form->casttarget == target_type_oid);

	if (cast_form->castmethod == COERCION_METHOD_FUNCTION)
	{
		Assert(OidIsValid(cast_form->castfunc));
		expr = (Expr *) makeFuncExpr(cast_form->castfunc,
									 target_type_oid,
									 list_make1(expr),
									 InvalidOid,
									 InvalidOid,
									 COERCE_IMPLICIT_CAST);
	}
	else if (cast_form->castmethod == COERCION_METHOD_BINARY)
	{
		expr = (Expr *) makeRelabelType(expr,
										target_type_oid,
										-1,			/* typemod */
										InvalidOid,	/* collid */
										COERCE_IMPLICIT_CAST);
	}
	else
	{
		/* We don't support COERCION_METHOD_INOUT here */
		expr = NULL;
	}
	ReleaseSysCache(tuple);

	return expr;
}

static Expr *
make_altfunc_expr_filter(Expr *expr, Expr *filter, Expr *defresult)
{
	CaseWhen   *case_when;
	CaseExpr   *case_expr;

	Assert(exprType(filter) == BOOLOID);

	if (!defresult)
	{
		Oid		type_oid = exprType(expr);
		int32	type_mod = exprTypmod(expr);
		Oid		type_coll = exprCollation(expr);
		int16	type_len;
		bool	type_byval;

		get_typlenbyval(type_oid, &typlen, &typbyval);
		Assert(typbyval);	/* only inline variable */
		defresult = (Expr *) makeConst(type_oid,
									   type_mod,
									   type_coll,
									   type_len,
									   (Datum) 0,
									   false,	/* isnull */
									   type_byval);
	}
	Assert(exprType(expr) == exprType(defresult));

	/* in case when the 'filter' is matched */
	case_when = makeNode(CaseWhen);
	case_when->expr = filter;
	case_when->result = expr;
	case_when->location = -1;

	/* case body */
	case_expr = makeNode(CaseExpr);
	case_expr->casetype = exprType(expr);
	case_expr->arg = list_make1(case_when);
	case_expr->args = NIL;
	case_expr->defresult = defresult;
	case_expr->location = -1;

	return (Expr *) case_expr;
}

static Expr *
make_altfunc_pseudo_aggfunc(Expr *expr, const char *func_name)
{
	Oid				namespace_oid = get_namespace_oid("pgstrom", false);
	Oid				func_argtype = exprType(expr);
	oidvector	   *func_argtypes = buildoidvector(&func_argtype, 1);
	HeapTuple		tuple;
	Form_pg_proc	proc_form;

	Assert(strcmp(func_name, "psum32") == 0 ||
		   strcmp(func_name, "psum") == 0 ||
		   strcmp(func_name, "pmin") == 0 ||
		   strcmp(func_name, "pmax") == 0);

	tuple = SearchSysCache3(PROCNAMEARGSNSP,
							PointerGetDatum(func_name),
							PointerGetDatum(func_argtypes),
							ObjectIdGetDatum(namespace_oid));
	if (!HeapTupleIsValid(tuple))
		return NULL;
	proc_form = (Form_pg_proc) GETSTRUCT(tuple);
	if (proc_form->prorettype == func_argtype)
	{
		/* Right now, pseudo aggregate function takes fixed-length
		 * numeric variables only, so no need care about typmod,
		 * colleace and so on.
		 */
		expr = (Expr *) makeFuncExpr(HeapTupleGetOid(tuple),
									 func_argtype,
									 list_make1(expr),
									 InvalidOid,
									 InvalidOid,
									 COERCE_EXPLICIT_CALL);
	}
	else
	{
		elog(INFO, "Bug? pseudo aggregate has incompatible types");
		expr = NULL;
	}
	ReleaseSysCache(tuple);
	return expr;
}

static Expr *
make_altfunc_expr_nrows(Aggref *aggref, List **pre_tlist, Oid type_oid)
{
	Expr	   *expr;

	expr = (Expr *) makeConst(INT4OID,
							  -1,
							  InvalidOid,
							  sizeof(int32),
							  (Datum) 1,
							  false,	/* isnull */
							  true);	/* byval */
	if (aggref->aggfilter)
	{
		expr = make_altfunc_expr_filter(expr, aggref->aggfilter, defresult);
		if (!expr)
			return NULL;
	}
	/* make-up pgstrom.psum(X) */
	return make_altfunc_pseudo_aggfunc(expr, "psum32");
}

static Expr *
make_altfunc_expr_nrows_notnull(Aggref *aggref, List **pre_tlist,
								Oid type_oid,  Expr *arg)
{
	NullTest   *ntest;
	Expr	   *cond;
	Expr	   *expr;

	expr = (Expr *) makeConst(INT4OID,
							  -1,
							  InvalidOid,
							  sizeof(int32),
							  (Datum) 1,
							  false,	/* isnull */
							  true);	/* byval */
	/* nulltest */
	ntest = (NullTest *) makeNode(NullTest);
	ntest->arg = arg;
	ntest->nulltesttype = IS_NOT_NULL;
	ntest->argisrow = false;

	/* merge, if aggref->aggfilter exist */
	if (aggref->aggfilter)
		cond = make_andclause(list_make2(ntest, aggref->aggfilter));
	else
		cond = ntest;
	/* 1, if expression is not null. 0, elsewhere */
	expr = make_altfunc_expr_filter(expr, cond, NULL);
	if (!expr)
		return NULL;
	/* make-up pgstrom.psum(X) */
	return make_altfunc_pseudo_aggfunc(expr, "psum32");
}

static Expr *
make_altfunc_expr_min(Aggref *aggref, List **pre_tlist,
					  Oid type_oid, Expr *arg)
{
	if (exprType(arg) != type_oid)
		arg = make_altfunc_expr_typecast(arg, type_oid);
	if (aggref->aggfilter)
	{
		Expr   *defresult
			= (Expr *)makeNullConst(exprType(arg),
									exprTypmod(arg),
									exprCollation(arg));
		arg = make_altfunc_expr_filter(arg, aggref->aggfilter, defresult);
	}
	/* make-up pgstrom.pmin(X) */
	return make_altfunc_pseudo_aggfunc(arg, "pmin");
}

static Expr *
make_altfunc_expr_max(Aggref *aggref, List **pre_tlist,
					  Oid type_oid, Expr *arg)
{
	if (exprType(arg) != type_oid)
		arg = make_altfunc_expr_typecast(arg, type_oid);
	if (aggref->aggfilter)
	{
		Expr   *defresult
			= (Expr *)makeNullConst(exprType(arg),
									exprTypmod(arg),
									exprCollation(arg));
		arg = make_altfunc_expr_filter(arg, aggref->aggfilter, defresult);
	}
	/* make-up pgstrom.pmin(X) */
	return make_altfunc_pseudo_aggfunc(arg, "pmax");
}

static Expr *
make_altfunc_expr_sum(Aggref *aggref, List **pre_tlist,
					  Oid type_oid, Expr *arg)
{
	if (exprType(arg) != type_oid)
		arg = make_altfunc_expr_typecast(arg, type_oid);
	if (aggref->aggfilter)
		arg = make_altfunc_expr_filter(arg, aggref->aggfilter, NULL);
	/* make-up pgstrom.psum(X) */
	return make_altfunc_pseudo_aggfunc(arg, "psum");
}

static Expr *
make_altfunc_expr_mul(Aggref *aggref, List **pre_tlist,
					  Oid type_oid, Expr *arg1, Expr *arg2)
{
	Expr	   *expr;
	HeapTuple	tuple;
	Form_pg_operator opform;

	if (exprType(arg1) != type_oid)
		arg1 = make_altfunc_expr_typecast(arg1, type_oid);
	if (exprType(arg2) != type_oid)
        arg2 = make_altfunc_expr_typecast(arg2, type_oid);

	tuple = SearchSysCache4(OPERNAMENSP,
							CStringGetDatum("*"),
							ObjectIdGetDatum(type_oid),
							ObjectIdGetDatum(type_oid),
							ObjectIdGetDatum(PG_CATALOG_NAMESPACE));
	if (!HeapTupleIsValid(tuple))
		return NULL;
	opform = (Form_pg_operator) GETSTRUCT(tuple);

	oper = makeNode(OpExpr);
	oper->opno = HeapTupleGetOid(tuple);
	oper->opfuncid = opform->oprcode;
	oper->opresulttype = opform->oprresult;
	oper->opretset = get_func_retset(oper->opfuncid);
	oper->args = list_make2(arg1, arg2);
	oper->opcollid = InvalidOid;
	oper->inputcollid = InvalidOid;
	oper->location = -1;

	if (aggref->aggfilter)
		expr = make_altfunc_expr_filter((Expr *)oper, aggref->aggfilter, NULL);
	else
		expr = (Expr *) oper;
	ReleaseSysCache(tuple);

	/* make-up pgstrom.psum(X*Y) */
	return make_altfunc_pseudo_aggfunc(arg, "psum");
}









static Aggref *
make_gpupreagg_refnode(Aggref *aggref, List **pre_tlist)
{
	const aggfunc_catalog_t *aggfn_cat;
	oidvector  *altfn_argtypes;
	Aggref	   *altnode;

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
		if (!codegen_available_expression(tle->expr))
			return false;
	}
	if (!codegen_available_expression(aggref->aggfilter))
		return false;

	/*
	 * pulls the definition of alternative aggregate functions from
	 * the catalog. we expect these are installed in "pgstrom" schema.
	 */
	namespace_oid = get_namespace_oid("pgstrom", true);
	if (!OidIsValid(namespace_oid))
		return false;

	altfn_argtypes = buildoidvector(aggfn_cat->altfn_argtypes,
									aggfn_cat->altfn_nargs);
	tuple = SearchSysCache3(PROCNAMEARGSNSP,
							PointerGetDatum(aggfn_cat->altfn_name),
							PointerGetDatum(altfn_argtypes),
							ObjectIdGetDatum(namespace_oid));
	if (!HeapTupleIsValid(tuple))
	{
		ereport(NOTICE,
				(errcode(ERRCODE_UNDEFINED_FUNCTION),
				 errmsg("no alternative aggregate function \"%s\" exists",
						funcname_signature_string(aggfn_cat->altfn_name,
												  aggfn_cat->altfn_nargs,
												  NIL,
												  aggfn_cat->altfn_argtypes)),
				 errhint("Try to run: CREATE EXTENSION pg_strom")));
		return false;
	}
	proform = (Form_pg_proc) GETSTRUCT(tuple);

	/* sanity checks */
	if (proform->prorettype != aggref->aggtype)
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
	altnode->aggfilter     = NIL;	/* moved to GpuPreAgg */
	altnode->aggstar       = aggref->aggstar;
	altnode->aggvariadic   = false;	/* see the checks above */
	altnode->aggkind       = aggref->aggkind;
	altnode->agglevelsup   = aggref->agglevelsup;
	altnode->location      = aggref->location;

	/*
	 * construct arguments of alternative aggregate function. It references
	 * an entry of pre_tlist, so we put expression node on the tlist on
	 * demand.
	 */
	for (i=0; i < aggfn_cat->altfn_nargs; i++)
	{
		int		code = aggfn_cat->altfn_argexprs[i];
		Oid		type_oid = aggfn_cat->altfn_argtypes[i];
		Expr   *expr;

		switch (code)
		{
			case ALTFUNC_EXPR_NROWS:
				expr = make_altfunc_expr_nrows(aggref, pre_tlist, type_oid);
				break;
			case ALTFUNC_EXPR_NROWS_NOTNULL:
				expr = make_altfunc_expr_nrows_notnull(aggref, pre_tlist,
													   type_oid,
													   linitial(aggref->args));
				break;
			case ALTFUNC_EXPR_MIN_ARG1:
				expr = make_altfunc_expr_min(aggref, pre_tlist, type_oid,
											 linitial(aggref->args));
				break;
			case ALTFUNC_EXPR_MAX_ARG1:
				expr = make_altfunc_expr_max(aggref, pre_tlist, type_oid,
											 linitial(aggref->args));
				break;
			case ALTFUNC_EXPR_SUM_ARG1:
				expr = make_altfunc_expr_sum(aggref, pre_tlist, type_oid,
											 linitial(aggref->args));
				break;
			case ALTFUNC_EXPR_SUM_ARG2:
				expr = make_altfunc_expr_sum(aggref, pre_tlist, type_oid,
											 lsecond(aggref->args));
				break;
			case ALTFUNC_EXPR_SUM_ARG1_MUL_ARG1:
				expr = make_altfunc_expr_mul(aggref, pre_tlist, type_oid,
											 linitial(aggref->args),
											 linitial(aggref->args));
				break;
			case ALTFUNC_EXPR_SUM_ARG2_MUL_ARG2:
				expr = make_altfunc_expr_mul(aggref, pre_tlist, type_oid,
											 lsecond(aggref->args),
											 lsecond(aggref->args));
				break;
			case ALTFUNC_EXPR_SUM_ARG1_MUL_ARG2:
				expr = make_altfunc_expr_mul(aggref, pre_tlist, type_oid,
											 linitial(aggref->args),
											 lsecond(aggref->args));
				break;
			default:
				elog(ERROR, "Bug? unexpected ALTFUNC_EXPR_* label");
				return false;
		}







	}








	/*
	 * OK, let's construct an alternative Aggref node
	 */
	newnode = makeNode(Aggref);
	newnode->aggfnoid;
	newnode->aggtype = ;
	newnode->aggcollid;
	newnode->inputcollid;
	newnode->aggdirectargs = aggref->aggdirectargs;
	newnode->args = new_args;
	newnode->aggorder = aggref->aggorder;
	newnode->aggdistinct = aggref->aggdistinct;
	newnode->aggfilter = aggref->aggfilter;
	newnode->aggstar = aggstart;
	newnode->aggvariadic = aggref->aggvariadic;
	newnode->aggkind = aggref->aggkind;
	newnode->agglevelsup = aggref->agglevelsup;
	newnode->location = aggref->location;



	if (aggref->aggfilter)
		; // needs to filter clause in the kernel code





	newnode = copyObject(aggref);

	



	return newnode;
}

static bool
gpupreagg_make_tlist(Agg *agg, List **p_agg_tlist, List **p_pre_tlist)
{
	Plan	   *outer_plan = outerPlan(agg);
	List	   *pre_tlist = NIL;
	List	   *agg_tlist = NIL;
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
		Oid			type_oid = exprType((Node *) tle->expr);
		int32		type_mod = exprTypmod((Node *) tle->expr);
		Oid			type_coll = exprCollation((Node *) tle->expr);
		char	   *resname = (tle->resname ? pstrdup(tle->resname) : NULL);

		Assert(IsA(tle, TargetEntry));
		if (i < agg->numCols && tle->resno == agg->grpColIdx[i])
		{
			Expr   *var = (Expr *) makeVar(OUTER_VAR,
										   tle->resno,
										   type_oid,
										   type_mod,
										   type_coll,
										   0);
			*pre_tlist = lappend(*pre_tlist,
								 makeTargetEntry(var,
												 list_length(tlist) + 1,
												 resname,
												 tle->resjunk));
			i++;
		}
		else
		{
			Expr   *cnst;
			int16	type_len;
			bool	type_byval;

			get_typlenbyval(type_oid, &type_len, &type_byval);
			cnst = (Expr *) makeConst(type_oid,
									  type_mod,
									  type_coll,
									  type_len,
									  (Datum) 0,
									  true,
									  type_byval);
			*pre_tlist = lappend(*pre_tlist,
								 makeTargetEntry(cnst,
												 list_length(tlist) + 1,
												 resname,
												 tle->resjunk));
		}
	}

	/* On the next, replace aggregate functions in tlist of Agg node
	 * according to the aggfunc_catalog[] definition.
	 */
	foreach (cell, agg->plan.targetlist)
	{
		TargetEntry *tle = lfirst(cell);
		Aggref		*aggref_old;
		Aggref		*aggref_new;
		const aggfunc_catalog_t *aggfunc_cat;

		/* Except for Aggref, node shall be an expression node that
		 * contains references to group-by keys. No needs to replace.
		 */
		if (!IsA(tle->expr, Aggref))
		{
			Bitmapset  *tempset = NULL;
			int			col;

			pull_varattnos((Node *) tle->expr, OUTER_VAR, &tempset);
			while ((col = bms_first_member(tempset)) >= 0)
			{
				col += FirstLowInvalidHeapAttributeNumber;

				for (i=0; i < agg->numCols; i++)
				{
					if (col == agg->grpColIdx[i])
						break;
				}
				if (i == agg->numCols)
					elog(ERROR, "Bug? references to out of grouping key");
			}
			bms_free(tempset);
			*agg_tlist = lappend(*agg_tlist, copyObject(tle));
			continue;
		}
		/* existing aggregate function shall be replaced */
		aggref = make_gpupreagg_refnode((Aggref *) tle->expr, pre_tlist);
		if (!aggref)
			return false;

		tle = flatCopyTargetEntry(tle);
		tle->expr = (Expr *) aggref;
		*agg_tlist = lappend(*agg_tlist, tle);
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
	List		   *agg_tlist = NIL;

	/* nothing to do, if feature is turned off */
	if (!pgstrom_enabled || !enable_gpupreagg)
		return;

	/* Try to construct target-list of both Agg and GpuPreAgg node.
	 * If unavailable to construct, it indicates this aggregation
	 * does not support partial aggregation.
	 */
	if (!gpupreagg_make_tlist(agg, &agg_tlist, &pre_tlist))
		return;




	/*
	 * Try to construct a GpuPreAggPlan node.
	 * If aggregate node contains any unsupported expression, we give up
	 * to insert GpuPreAgg node.
	 */
	gpreagg = palloc0(sizeof(GpuPreAggPlan));
	NodeSetTag(gpreagg, T_CustomPlan);
	gpreagg->cplan.methods = &gpupreagg_plan_methods;
	gpreagg->cplan.plan.targetlist = gpupreagg_make_tlist(agg, &agg_tlist);
	if (!gpreagg->cplan.plan.targetlist)
		return;





	/* Expected Aggregate cost, if GpuPreAgg is injected */
	total_cost = cost_gpupreagg(pstmt, agg);







	elog(INFO, "OK, this aggregate is supported");

}

static CustomPlanState *
gpupreagg_begin(CustomPlan *node, EState *estate, int eflags)
{
	return NULL;
}

static TupleTableSlot *
gpupreagg_exec(CustomPlanState *node)
{
	return NULL;
}

static void
gpupreagg_end(CustomPlanState *node)
{}

static void
gpupreagg_rescan(CustomPlanState *node)
{}

static void
gpupreagg_explain(CustomPlanState *node, List *ancestors, ExplainState *es)
{}

static Bitmapset *
gpupreagg_get_relids(CustomPlanState *node)
{
	return NULL;
}

static Node *
gpupreagg_get_special_var(CustomPlanState *node,
						  Var *varnode,
						  PlanState **child_ps)
{
	return NULL;
}

static void
gpupreagg_textout_plan(StringInfo str, const CustomPlan *node)
{}

static CustomPlan *
gpupreagg_copy_plan(const CustomPlan *from)
{
	GpuPreAggPlan  *oldnode = (GpuPreAggPlan *) from;
	GpuPreAggPlan  *newnode;

	newnode = palloc0(sizeof(GpuPreAggPlan));
	CopyCustomPlanCommon((Node *) oldnode, (Node *) newnode);

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
	gpupreagg_plan_methods.GetSpecialCustomVar = gpupreagg_get_special_var;
	gpupreagg_plan_methods.TextOutCustomPlan   = gpupreagg_textout_plan;
	gpupreagg_plan_methods.CopyCustomPlan      = gpupreagg_copy_plan;
}




/* ----------------------------------------------------------------
 *
 * NOTE: below is the code being run on OpenCL server context
 *
 * ---------------------------------------------------------------- */







/* ----------------------------------------------------------------
 *
 * NOTE: below is the function to process enhanced aggregate operations
 *
 * ---------------------------------------------------------------- */

/*
 * gpupreagg_pseudo_aggregate - Dummy placeholder function of pmin, pmax
 * and psum for partial groups. Usually, GPU device produces these values,
 * so this function should not be called during execution actually.
 * However, once GPU gave up execution of the kernel (because of external
 * or compressed toast datum for example), host  code generate a value as
 * if these placeholder represents a partially aggregated data --- note
 * that nrows of a group with one row is always 1, and min, max and sum
 * of a group with one row is always supplied expression.
 */
Datum
gpupreagg_pseudo_aggregate(PG_FUNCTION_ARGS)
{
	PG_RETURN_DATUM(PG_GETARG_DATUM(0));
}
PG_FUNCTION_INFO_V1(gpupreagg_pseudo_aggregate);

/*
 * ex_avg() - an enhanced average calculation that takes two arguments;
 * number of rows in this group and partial sum of the value.
 * Then, it eventually generate mathmatically compatible average value.
 */
Datum
pgstrom_sum_int8_accum(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray;
	int32		nrows = PG_GETARG_INT32(1);
	int64		psum = PG_GETARG_INT64(2);
	int64	   *transdata;

	if (AggCheckCallContext(fcinfo, NULL))
		transarray = PG_GETARG_ARRAYTYPE_P(0);
	else
		transarray = PG_GETARG_ARRAYTYPE_P_COPY(0);

	if (ARR_NDIM(transarray) != 1 ||
		ARR_DIMS(transarray)[0] != 2 ||
		ARR_HASNULL(transarray) ||
		ARR_ELEMTYPE(transarray) != INT8OID)
		elog(ERROR, "Two elements int8 array is expected");

	transdata = (int64 *) ARR_DATA_PTR(transarray);
	transdata[0] += (int64) nrows;	/* # of rows */
	transdata[1] += (int64) psum;	/* partial sum */

	PG_RETURN_ARRAYTYPE_P(transarray);
}
PG_FUNCTION_INFO_V1(pgstrom_int8_accum);

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
	int64	   *transdata = (int64 *) ARR_DATA_PTR(transarray);

	if (ARR_NDIM(transarray) != 1 ||
		ARR_DIMS(transarray)[0] != 2 ||
		ARR_HASNULL(transarray) ||
		ARR_ELEMTYPE(transarray) != INT8OID)
		elog(ERROR, "Two elements int8 array is expected");

	transdata = (int64 *) ARR_DATA_PTR(transarray);

	PG_RETURN_INT64(transdata[1]);
}



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
	ArrayType  *transarray;
	int32		nrows = PG_GETARG_INT32(1);
	float8		psum = PG_GETARG_FLOAT8(2);
	float8	   *transdata;
	float8		newsum;

	if (AggCheckCallContext(fcinfo, NULL))
		transarray = PG_GETARG_ARRAYTYPE_P(0);
	else
		transarray = PG_GETARG_ARRAYTYPE_P_COPY(0);

	transdata = check_float8_array(transarray, 3);
	transdata[0] += (float8) nrows;
	newsum = transdata[1] + psum;
	check_float8_valid(newsum, isinf(transdata[1]) || isinf(psum), true);
	transdata[1] = newsum;

    PG_RETURN_ARRAYTYPE_P(transarray);	
}
PG_FUNCTION_INFO_V1(pgstrom_float8_accum);

/*
 * variance and stddev - mathmatical compatible result can be lead using
 * nrows, psum(X) and psum(X*X). So, we track these variables.
 */
Datum
pgstrom_variance_float8_accum(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray = PG_GETARG_ARRAYTYPE_P(0);
	int32		nrows  = PG_GETARG_INT32(1);
	float8		psumX  = PG_GETARG_FLOAT8(2);
	float8		psumX2 = PG_GETARG_FLOAT8(3);
	float8	   *transdata;
	float8		newsumX;
	float8		newsumX2;

	if (AggCheckCallContext(fcinfo, NULL))
		transarray = PG_GETARG_ARRAYTYPE_P(0);
	else
		transarray = PG_GETARG_ARRAYTYPE_P_COPY(0);

	transdata = check_float8_array(transarray, 3);
	transdata[0] += (float8) nrows;
	/* SUM(X) */
	newsumX = transdata[1] + psumX;
	check_float8_valid(newsumX, isinf(transdata[1]) || isinf(psumX), true);
	transdata[1] = newsumX;
	/* SUM(X*X) */
	newsumX2 = transdata[2] + psumX2;
	check_float8_valid(newsumX2, isinf(transdata[2]) || isinf(psumX2), true);
	transdata[2] = newsumX2;

	PG_RETURN_ARRAYTYPE_P(transarray);
}
PG_FUNCTION_INFO_V1(pgstrom_variance_float8_accum);

/*
 * covariance - mathmatical compatible result can be lead using
 * nrows, psum(X), psum(X*X), psum(Y), psum(Y*Y), psum(X*Y)
 */
Datum
pgstrom_covariance_float8_accum(PG_FUNCTION_ARGS)
{
	ArrayType  *transarray;
	int32		nrows  = PG_GETARG_INT32(1);
	float8		psumX  = PG_GETARG_FLOAT8(2);
	float8		psumX2 = PG_GETARG_FLOAT8(3);
	float8		psumY  = PG_GETARG_FLOAT8(4);
	float8		psumY2 = PG_GETARG_FLOAT8(5);
	float8		psumXY = PG_GETARG_FLOAT8(6);
	float8		newval;
	float8	   *transdata;

	if (AggCheckCallContext(fcinfo, NULL))
        transarray = PG_GETARG_ARRAYTYPE_P(0);
    else
        transarray = PG_GETARG_ARRAYTYPE_P_COPY(0);

	transdata = check_float8_array(transarray, 3);
    transdata[0] += (float8) nrows;

	/* SUM(X) */
	newval = transdata[1] + psumX;
	check_float8_valid(newval, isinf(transdata[1]) || isinf(psumX), true);
	transdata[1] = newval;

	/* SUM(X*X) */
	newval = transdata[2] + psumX2;
    check_float8_valid(newval, isinf(transdata[2]) || isinf(psumX2), true);
    transdata[2] = newval;

	/* SUM(Y) */
	newval = transdata[3] + psumY;
	check_float8_valid(newval, isinf(transdata[3]) || isinf(psumY), true);
	transdata[3] = newval;

	/* SUM(Y*Y) */
	newval = transdata[4] + psumY2;
	check_float8_valid(newval, isinf(transdata[4]) || isinf(psumY2), true);
	transdata[4] = newval;

	/* SUM(X*Y) */
	newval = transdata[4] + psumXY;
	check_float8_valid(newval, isinf(transdata[4]) || isinf(psumXY), true);
	transdata[4] = newval;

	PG_RETURN_ARRAYTYPE_P(transarray);
}
PG_FUNCTION_INFO_V1(pgstrom_covariance_float8_accum);
