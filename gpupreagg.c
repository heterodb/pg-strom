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

#include "pg_strom.h"

static CustomPlanMethods		gpupreagg_plan_methods;
static bool						enable_gpupreagg;

typedef struct
{
	CustomPlan		cplan;

} GpuPreAggPlan;

/*
 * Arguments of alternative functions.
 * An alternative functions 
 *
 */
#define ALTFUNC_EXPR_NROWS				100	/* PSUM(1) */
#define ALTFUNC_EXPR_NROWS_NOTNULL		101	/* PSUM((X IS NOT NULL)::int) */
#define ALTFUNC_EXPR_MIN_ARG1			102	/* PMIN(X) */
#define ALTFUNC_EXPR_MAX_ARG1			103	/* PMAX(X) */
#define ALTFUNC_EXPR_SUM_ARG1			104	/* PSUM(X) */
#define ALTFUNC_EXPR_SUM_ARG2			105	/* PSUM(Y) */
#define ALTFUNC_EXPR_SUM_ARG1_MUL_ARG1	106	/* PSUM(X*X) */
#define ALTFUNC_EXPR_SUM_ARG2_MUL_ARG2	107	/* PSUM(Y*Y) */
#define ALTFUNC_EXPR_SUM_ARG1_MUL_ARG2	108	/* PSUM(X*Y) */

static struct {
	/* aggregate function can be preprocessed */
	const char *aggfn_name;
	int			aggfn_nargs;
	Oid			aggfn_argtypes[4]
	/* alternative function to generate same result */
	const char *altfn_name;
	int			altfn_nargs;
	Oid			altfn_argtypes[8];
	int			altfn_argexprs[8];
} aggfunc_catalog[] = {
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








void
pgstrom_try_insert_gpupreagg(PlannedStmt *pstmt, Agg *agg)
{
	/* Our major pain on aggregation is data sorting on the tons of
	 * input records. So, GpuPreAgg tries to reduce number of input
	 * records using partial data reduction and tlist rewriting.
	 * Elsewhere, we don't think it has great advantages.
	 */
	if (agg->aggstrategy != AGG_SORTED)
		return;





}

static CustomPlanState *
gpupreagg_begin(CustomPlan *node, EState *estate, int eflags)
{}

static TupleTableSlot *
gpupreagg_exec(CustomPlanState *node)
{}

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
{}

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
	DefineCustomBoolVariable("enable_gpuhashjoin",
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
 * pgstrom_psum() - that produce a partial sum value being computed
 * in GPU device, for the global aggregate calculation in the later stage.
 * If GPU gave up record reduction, this function just returns the supplied
 * expression, then host code handles the group consists of only one row.
 */
Datum pgstrom_psum(PG_FUNCTION_ARGS)
{
	PG_RETURN_DATUM(PG_GETARG_DATUM(0));
}
PG_FUNCTION_INFO_V1(pgstrom_psum);

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
 * NOTE: The built-in int8_avg() can handle final translation with int64
 * array (that has compatible layout with Int8TransTypeData in the core).
 * So, no need to re-invent same stuff again.
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

    float8      newvalY = PG_GETARG_FLOAT8(1);
    float8      newvalX = PG_GETARG_FLOAT8(2);
    float8     *transvalues;
    float8      N,
		sumX,
		sumX2,
		sumY,
		sumY2,
		sumXY;


}
PG_FUNCTION_INFO_V1(pgstrom_covariance_float8_accum);
