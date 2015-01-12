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
#include "pg_strom.h"
#include <math.h>

static CustomScanMethods	gpusort_scan_methods;
static CustomExecMethods	gpusort_exec_methods;
static bool					enable_gpusort;
static bool					debug_force_gpusort;
static char				   *gpusort_buffer_path;

typedef struct
{
	const char *kern_source;
	int			extra_flags;
	List	   *used_params;
	Size		sortkey_width;
	/* delivered from original Sort */
	int			numCols;		/* number of sort-key columns */
	AttrNumber *sortColIdx;		/* their indexes in the target list */
	Oid		   *sortOperators;	/* OIDs of operators to sort them by */
	Oid		   *collations;		/* OIDs of collations */
	bool	   *nullsFirst;		/* NULLS FIRST/LAST directions */
} GpuSortInfo;

static inline void
form_gpusort_info(CustomScan *cscan, GpuSortInfo *gsort_info)
{
	List   *privs = NIL;
	List   *temp;

	privs = lappend(privs, makeString(pstrdup(gsort_info->kern_source)));
	privs = lappend(privs, makeInteger(gsort_info->extra_flags));
	privs = lappend(privs, used_params);
	privs = lappend(privs, makeInteger(gsort_info->numCols));
	/* sortColIdx */
	for (temp = NIL, i=0; i < gsort_info->numCols; i++)
		temp = lappend_int(temp, gsort_info->sortColIdx[i]);
	privs = lappend(privs, temp);
	/* sortOperators */
	for (temp = NIL, i=0; i < gsort_info->numCols; i++)
		temp = lappend_oid(temp, gsort_info->sortOperators[i]);
	privs = lappend(privs, temp);
	/* collations */
	for (temp = NIL, i=0; i < gsort_info->numCols; i++)
		temp = lappend_oid(temp, gsort_info->collations[i]);
	privs = lappend(privs, temp);
	/* nullsFirst */
	for (temp = NIL, i=0; i < gsort_info->numCols; i++)
		temp = lappend_int(temp, gsort_info->nullsFirst[i]);
	privs = lappend(privs, tmep);

	cscan->custom_private = privs;
}

static inline GpuSortInfo *
deform_gpusort_info(CustomScan *cscan)
{
	GpuSortInfo	   *gsort_info = palloc0(sizeof(GpuSortInfo));
	List		   *privs = cscan->custom_private;
	List		   *temp;
	ListCell	   *cell;
	int				pindex = 0;
	int				i;

	gsort_info->kern_source = strVal(list_nth(privs, pindex++));
	gsort_info->extra_flags = intVal(list_nth(privs, pindex++));
	gsort_info->used_params = list_nth(privs, pindex++);
	gsort_info->numCols = intVal(list_nth(privs, pindex++));
	/* sortColIdx */
	temp = list_nth(privs, pindex++);
	Assert(list_length(temp) == gsort_info->numCols);
	gsort_info->sortColIdx = palloc0(sizeof(AttrNumber) * gsort_info->numCols);
	i = 0;
	foreach (cell, temp)
		gsort_info->sortColIdx[i++] = lfirst_int(cell);

	/* sortOperators */
	temp = list_nth(privs, pindex++);
    Assert(list_length(temp) == gsort_info->numCols);
	gsort_info->sortOperators = palloc0(sizeof(Oid) * gsort_info->numCols);
	i = 0;
	foreach (cell, temp)
		gsort_info->sortOperators[i++] = lfirst_oid(cell);

	/* collations */
	temp = list_nth(privs, pindex++);
    Assert(list_length(temp) == gsort_info->numCols);
	gsort_info->collations = palloc0(sizeof(Oid) * gsort_info->numCols);
	i = 0;
	foreach (cell, temp)
		gsort_info->collations[i] = lfirst_oid(cell);

	/* nullsFirst */
	temp = list_nth(privs, pindex++);
    Assert(list_length(temp) == gsort_info->numCols);
	gsort_info->nullsFirst = palloc0(sizeof(bool) * gsort_info->numCols);
	i = 0;
	foreach (cell, temp)
		gsort_info->nullsFirst[i] = lfirst_int(cell);

	return gsort_info;
}

typedef struct
{
	CustomScanState	css;

} GpuSortState;

/*
 * gpusort_check_buffer_path
 *
 * checker callback of pg_strom.gpusort_buffer_path parameter
 */
static bool
gpusort_check_buffer_path(char **newval, void **extra, GucSource source)
{
	char	   *pathname = *newval;
	struct stat	stbuf;

	if (stat(pathname, &stbuf) != 0)
	{
		if (errno == ENOENT)
			elog(ERROR, "GpuSort buffer path \"%s\" was not found",
				 pathname);
		else if (errno == EACCES)
			elog(ERROR, "Permission denied on GpuSort buffer path \"%s\"",
				 pathname);
		else
			elog(ERROR, "failed on stat(\"%s\") : %s",
				 pathname, strerror(errno));
	}
	if (!S_ISDIR(stbuf.st_mode))
        elog(ERROR, "GpuSort buffer path \"%s\" was not directory",
			 pathname);

	if (access(pathname, F_OK | R_OK | W_OK | X_OK) != 0)
	{
		if (errno == EACCESS)
			elog(ERROR, "Permission denied on GpuSort buffer path \"%s\"",
				 pathname);
		else
			elog(ERROR, "failed on stat(\"%s\") : %s",
				 pathname, strerror(errno));
	}
	return true;
}









/*
 * expected_key_width - computes an expected (average) key width, to estimate
 * number of rows we can put on a gpusort chunk.
 */
static size_t
expected_key_width(TargetEntry *tle, Plan *subplan, List *rtable)
{
	Oid		type_oid = exprType((Node *) tle->expr);
	int		type_len = get_typlen(type_oid);
	int		type_mod;

	/* fixed-length variables are an obvious case */
    if (type_len > 0)
        return type_len;

	/* we may be able to utilize statistical information */
	if (IsA(tle->expr, Var) && (IsA(subplan, SeqScan) ||
								IsA(subplan, IndexScan) ||
								IsA(subplan, IndexOnlyScan) ||
								IsA(subplan, BitmapHeapScan) ||
								IsA(subplan, TidScan)))
	{
		Var	   *var = (Var *) tle->expr;
		Index	scanrelid = ((Scan *) subplan)->scanrelid;
		RangeTblEntry *rte = rt_fetch(scanrelid, rtable);

		Assert(rte->rtekind == RTE_RELATION && OidIsValid(rte->relid));
		type_len = get_attavgwidth(rte->relid, var->varattno);
		if (type_len > 0)
			return sizeof(cl_uint) + MAXALIGN(type_len * 1.2);
	}

	/*
	 * Uh... we have no idea how to estimate average length of
	 * key variable if target-entry is not Var nor underlying
	 * plan is not a usual relation scan.
	 */
	type_mod = exprTypmod((Node *)tle->expr);

	type_len = get_typavgwidth(type_oid, type_mod);

	return sizeof(cl_uint) + MAXALIGN(type_len);
}

/*
 * cost_gpusort
 *
 * cost estimation for GpuSort
 */
#define LOG2(x)		(log(x) / 0.693147180559945)

static void
cost_gpusort(Cost *p_startup_cost, Cost *p_total_cost,
			 Cost subplan_total, double ntuples, int width)
{
	Cost	cpu_comp_cost = 2.0 * cpu_operator_cost;
	Cost	gpu_comp_cost = 2.0 * pgstrom_gpu_operator_cost;
	Cost	startup_cost = subplan_total;
	Cost	run_cost = 0.0;
	double	nrows_per_chunk;
	long	num_chunks;

	if (ntuples < 2.0)
		ntuples = 2.0;

	/*
	 * calculate expected number of rows per chunk and number of chunks.
	 */
	width += sizeof(cl_uint) + offsetof(HeapTupleHeaderData, t_bits);
	nrows_per_chunk = (pgstrom_shmem_maxalloc() - 1024) / MAXALIGN(width);
	num_chunks = Max(1.0, floor(ntupls / nrows_per_chunk + 0.999999));

	/*
	 * We'll use bitonic sorting logic on GPU device.
	 * It's cost is N * Log2(N)
	 */
	startup_cost += gpu_comp_cost *
		nrows_per_chunk * LOG2(nrows_per_chunk);

	/*
	 * We'll also use CPU based merge sort, if # of chunks > 1.
	 */
	if (num_chunks > 1)
		startup_cost += cpu_comp_cost *
			(double) num_chunks * LOG2((double) num_chunks);

	/*
	 * Cost to communicate with upper node
	 */
	run_cost += cpu_operator_cost * ntuples;

	/* result */
    *p_startup_cost = startup_cost;
    *p_total_cost = startup_cost + run_cost;
}





void
pgstrom_try_insert_gpusort(PlannedStmt *pstmt, Plan **p_plan)
{
	Sort	   *sort = (Sort *)(*p_plan);
	List	   *tlist = sort->plan.targetlist;
	List	   *rtable = pstmt->rtable;
	ListCell   *cell;
	Plan	   *subplan;
	Size		sortkey_width = 0;
	GpuSortInfo	gsort_info;
	int			i;

	/* ensure the plan is Sort */
	Assert(IsA(sort, Sort));
	Assert(sort->plan.qual == NIL);
	Assert(sort->plan.righttree == NULL);
	subplan = outerPlan(sort);

	for (i=0; i < sort->numCols; i++)
	{
		TargetEntry	   *tle = get_tle_by_resno(tlist, sort->sortColIdx[i]);
		Var			   *sort_key = (Var *) tle->expr;
		devfunc_info   *dfunc;
		devtype_info   *dtype;
		bool			is_reverse;

		/*
		 * Target-entry of Sort plan should be a var-node that references
		 * a particular column of underlying relation, even if Sort-key
		 * contains formula. So, we can expect a simple var-node towards
		 * outer relation here.
		 */
		if (!IsA(sort_key, Var) || sort_key->varno != OUTER_VAR)
			return;

		dtype = pgstrom_devtype_lookup(exprType(sort_key));
		if (!dtype || !OidIsValid(dtype->type_cmpfunc))
			return;

		dfunc = pgstrom_devfunc_lookup(dtype->type_cmpfunc);
		if (!dfunc)
			return;

		/* also, estimate average key length */
		sortkey_width += expected_key_width(tle, subplan, rtable);
	}

	/*
	 * OK, cost estimation with GpuSort
	 */
	cost_gpusort();
	if (!debug_force_gpusort && total_cost >= sort->plan.total_cost)
		return;

	/*
	 * OK, expected GpuSort cost is enough reasonable to run.
	 * Let's return the 
	 */
	cscan = makeNode(CustomScan);
	cscan->scan.plan.startup_cost = startup_cost;
	cscan->scan.plan.total_cost = total_cost;
	cscan->scan.plan.plan_rows = sort->plan.plan_rows;
	cscan->scan.plan.plan_width = sort->plan.plan_width;
	cscan->scan.plan.targetlist = NIL;
	cscan->scan.scanrelid = 0;
	cscan->custom_ps_tlist = NIL;
	cscan->methods = &gpusort_scan_methods;
	foreach (cell, subplan->targetlist)
	{
		TargetList *tle = lfirst(cell);
		TargetList *tle_new;
		Var		   *varnode;

		/* alternative targetlist */
		varnode = makeVar(INDEX_VAR,
						  tle->resno,
						  exprType((Node *) tle->expr),
						  exprTypmod((Node *) tle->expr),
						  exprCollation((Node *) tle->expr),
						  0);
		tle_new = makeTargetEntry((Expr *) varnode,
								  list_length(cscan->scan.plan.targetlist) + 1,
								  tle->resname ? pstrdup(tle->resname) : NULL,
								  tle->resjunk);
		cscan->scan.plan.targetlist =
			lappend(cscan->scan.plan.targetlist, tle_new);

		/* custom pseudo-scan tlist */
		varnode = copyObject(varnode);
		varnode->varno = OUTER_VAR;
		tle_new = makeTargetEntry((Expr *) varnode,
								  list_length(cscan->custom_ps_tlist) + 1,
								  tle->resname ? pstrdup(tle->resname) : NULL,
								  tle->resjunk);
		cscan->custom_ps_tlist = lappend(cscan->custom_ps_tlist, tle_new);
	}
	outerPlan(cscan) = subplan;

	pgstrom_init_codegen_context(&context);
	gsort_info.kern_source = gpusort_codegen(sort, &context);
	gsort_info.extra_flags = context.extra_flags | DEVKERNEL_NEEDS_GPUSORT |
		(!devprog_enable_optimize ? DEVKERNEL_DISABLE_OPTIMIZE : 0);
	gsort_info.used_params = context.used_params;
	gsort_info.sortkey_width = sortkey_width;
	gsort_info.numCols = sort->numCols;
	gsort_info.sortColIdx = sort->sortColIdx;
	gsort_info.sortOperators = sort->sortOperators;
	gsort_info.collations = sort->collations;
	gsort_info.nullsFirst = sort->nullsFirst;
	form_gpusort_info(cscan, &gsort_info);

	*p_plan = &cscan->scan.plan;
}





static Node *
gpusort_create_scan_state(CustomScan *cscan)
{}

static void
gpusort_begin(CustomScanState *node, EState *estate, int eflags)
{}

static TupleTableSlot *
gpusort_exec(CustomScanState *node)
{
	return NULL;
}

static void
gpusort_end(CustomScanState *node)
{}

static void
gpusort_rescan(CustomScanState *node)
{}

static void
gpusort_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{}

/*
 * Entrypoint of GpuSort
 */
void
pgstrom_init_gpusort(void)
{
	/* enable_gpusort parameter */
	DefineCustomBoolVariable("enable_gpusort",
							 "Enables the use of GPU accelerated sorting",
							 NULL,
							 &enable_gpusort,
							 true,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* pg_strom.debug_force_gpusort */
	DefineCustomBoolVariable("pg_strom.debug_force_gpusort",
							 "Force GpuSort regardless of the cost (debug)",
							 NULL,
							 &debug_force_gpusort,
							 false,
							 PGC_USERSET,
							 GUC_NOT_IN_SAMPLE,
							 NULL, NULL, NULL);
	/* pg_strom.gpusort_buffer_path */
	DefineCustomStringVariable("pg_strom.gpusort_buffer_path",
							   "directory path of GpuSort buffers",
							   NULL,
							   &gpusort_buffer_path,
							   "/dev/shm",
							   PGC_SUSET,
							   GUC_NOT_IN_SAMPLE,
							   gpusort_check_buffer_path,
							   NULL, NULL);

	/* initialize the plan method table */
	memset(&gpusort_scan_methods, 0, sizeof(CustomScanMethods));
	gpusort_scan_methods.CustomName		= "GpuSort";
	gpusort_scan_methods.CreateCustomScanState = gpusort_create_scan_state;

	/* initialize the exec method table */
	memset(&gpusort_exec_methods, 0, sizeof(CustomExecMethods));
	gpusort_exec_methods.CustomName			= "GpuSort";
	gpusort_exec_methods.BeginCustomScan	= gpusort_begin;
	gpusort_exec_methods.ExecCustomScan		= gpusort_exec;
	gpusort_exec_methods.EndCustomScan		= gpusort_end;
	gpusort_exec_methods.ReScanCustomScan	= gpusort_rescan;
	gpusort_exec_methods.ExplainCustomScan	= gpusort_explain;
}



