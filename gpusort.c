/*
 * gpusort.c
 *
 * Sort acceleration by GPU processors
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
#include "nodes/nodeFuncs.h"
#include "optimizer/cost.h"
#include "parser/parsetree.h"
#include "utils/lsyscache.h"
#include "pg_strom.h"
#include "opencl_gpusort.h"
#include <math.h>

/*
 * get_compare_function
 *
 * find up a comparison-function for the given data type
 */
static Oid
get_compare_function(Oid opno, bool *is_reverse)
{
	/* also see get_sort_function_for_ordering_op() */
	Oid		opfamily;
	Oid		opcintype;
	Oid		sort_func;
	int16	strategy;

	/* Find the operator in pg_amop */
	if (!get_ordering_op_properties(opno,
									&opfamily,
									&opcintype,
									&strategy))
		return InvalidOid;

	/* Find a function that implement comparison */
	sort_func = get_opfamily_proc(opfamily,
								  opcintype,
								  opcintype,
								  BTORDER_PROC);
	if (!OidIsValid(sort_func))	/* should not happen */
		elog(ERROR, "missing support function %d(%u,%u) in opfamily %u",
			 BTORDER_PROC, opcintype, opcintype, opfamily);

	*is_reverse = (strategy == BTGreaterStrategyNumber);
	return sort_func;
}

/*
 * cost_gpusort
 *
 * cost estimation for GpuScan
 */
#define LOG2(x)		(log(x) / 0.693147180559945)

static void
cost_gpusort(Cost *p_startup_cost, Cost *p_total_cost,
			 Cost input_cost, double ntuples, int width)
{
	Cost	comparison_cost = 0.0;	/* XXX where come from? */
	Cost	startup_cost = input_cost;
	Cost	run_cost = 0;

	if (ntuples < 2.0)
		ntuples = 2.0;

	/* Include the default cost-per-comparison */
	comparison_cost += 2.0 * cpu_operator_cost;

	/* adjustment for GPU */
	comparison_cost /= 100.0;

	/*
	 * We'll use bitonic sort on GPU device
	 * N * Log2(N)
	 */
	startup_cost += comparison_cost * ntuples * LOG2(ntuples);

	/* XXX - needs to adjust data transfer cost if non-integrated GPUs */
	run_cost += cpu_operator_cost * ntuples;

	/* result */
	*p_startup_cost = startup_cost;
	*p_total_cost = startup_cost + run_cost;
}

static char *
gpusort_codegen_comparison(Sort *sort, List **used_params, List **used_vars)
{
	StringInfoData	str;
	StringInfoData	decl;
	StringInfoData	body;
	codegen_context	context;
	int				i;

	initStringInfo(&str);
	initStringInfo(&decl);
	initStringInfo(&body);

	memset(&context, 0, sizeof(codegen_context));
	for (i=0; i < sort->numCols; i++)
	{
		TargetEntry *tle = get_tle_by_resno(sort->plan.targetlist,
											sort->sortColIdx[i]);
		bool	nullfirst = sort->nullsFirst[i];
		Oid		sort_op = sort->sortOperators[i];
		Oid		sort_func;
		Oid		sort_type;
		bool	is_reverse;
		char   *temp;
		devtype_info   *dtype;
		devfunc_info   *dfunc;

		/* type for comparison */
		sort_type = exprType((Node *)tle->expr);
		dtype = pgstrom_devtype_lookup(sort_type);
		Assert(dtype != NULL);
		context.type_defs = list_append_unique_ptr(context.type_defs, dtype);

		/* function for comparison */
		sort_func = get_compare_function(sort_op, &is_reverse);
		dfunc = pgstrom_devfunc_lookup(sort_func);
		Assert(dfunc != NULL);

		appendStringInfo(&decl,
						 "  pg_%s_t keyval_%ua;\n"
						 "  pg_%s_t keyval_%ub;\n",
						 dtype->type_name, i+1,
						 dtype->type_name, i+1);

		context.row_index = "r1";
		temp = pgstrom_codegen_expression((Node *)tle->expr, &context);
		appendStringInfo(&body, "  keyval_%ub = %s;\n", i+1, temp);
		pfree(temp);

		context.row_index = "r2";
		temp = pgstrom_codegen_expression((Node *)tle->expr, &context);
		appendStringInfo(&body, "  keyval_%ub = %s;\n", i+1, temp);
		pfree(temp);

		appendStringInfo(
			&body,
			"  if (!keyval_%ua.isnull && !keyval_%ub.isnull)\n"
			"  {\n"
			"    compval = pgfn_%s(errcode, keyval_%ua, keyval_%ub);\n"
			"    if (comp.value != 0)\n"
			"      return %s;\n"
			"  }\n"
			"  else if (keyval_%ua.isnull && !keyval_%ub.isnull)\n"
			"    return %d;\n"
			"  else if (!keyval_%ua.isnull && keyval_%ub.isnull)\n"
			"    return %d;\n"
			"\n",
			i+1, i+1,
			dfunc->func_name, i+1, i+1,
			is_reverse ? "-comp.value" : "comp.value",
			i+1, i+1, nullfirst ? -1 :  1,
			i+1, i+1, nullfirst ?  1 : -1);
	}

	/* make a comparison function */
	appendStringInfo(
		&str,
		"%s\n"	/* type function declarations */
		"static cl_int\n"
		"gpusort_comp(__global kern_gpusort *kgsort,\n"
		"             __global kern_column_store *kcs,\n"
		"             __global kern_toastbuf *toast,\n"
		"             __private int *errcode, cl_uint r1, cl_uint r2)\n"
		"{\n"
		"%s\n"	/* key variable declarations */
		"\n"
		"%s"	/* comparison body */
		"  return 0;\n"
		"}\n",
		pgstrom_codegen_declarations(&context),
		decl.data,
		body.data);

	pfree(decl.data);
	pfree(body.data);

	*used_params = context.used_params;
	*used_vars   = context.used_vars;

	return str.data;
}

/*
 * pgstrom_create_gpusort
 *
 * suggest an alternative sort using GPU devices
 */
CustomPlan *
pgstrom_create_gpusort(Sort *sort, List *rtable)
{
	List	   *tlist = sort->plan.targetlist;
	Plan	   *subplan = sort->plan.lefttree;
	Cost		startup_cost;
	Cost		total_cost;
	List	   *used_params;
	List	   *used_vars;
	ListCell   *cell;
	char	   *kernel_source;
	int			i;

	Assert(sort->plan.qual == NIL);
	Assert(!sort->plan.righttree);

	for (i=0; i < sort->numCols; i++)
	{
		TargetEntry	*tle = get_tle_by_resno(tlist, sort->sortColIdx[i]);
		Oid		sort_op = sort->sortOperators[i];
		Oid		sort_func;
		bool	is_reverse;

		/* sort key has to be runnable on GPU device */
		if (!pgstrom_codegen_available_expression(tle->expr))
			return NULL;

		sort_func = get_compare_function(sort_op, &is_reverse);
		if (!OidIsValid(sort_func) ||
			!pgstrom_devfunc_lookup(sort_func))
			return NULL;
	}

	/* next, cost estimation by GPU sort */
	cost_gpusort(&startup_cost, &total_cost,
				 subplan->total_cost,
				 subplan->plan_rows,
				 subplan->plan_width);
	elog(INFO, "sort cost %f => %f", sort->plan.total_cost, total_cost);
	if (total_cost >= sort->plan.total_cost)
		return NULL;

	kernel_source = gpusort_codegen_comparison(sort, &used_params, &used_vars);
	elog(INFO, "comparison kernel\n%s", kernel_source);

	foreach(cell, used_vars)
	{
		Var	   *var = lfirst(cell);

		Assert(IsA(var, Var) && var->varno == OUTER_VAR);
	}

	return NULL;
}
