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
	long		num_chunks;
	Size		chunk_size;
	/* delivered from original Sort */
	int			numCols;		/* number of sort-key columns */
	AttrNumber *sortColIdx;		/* their indexes in the target list */
	Oid		   *sortOperators;	/* OIDs of operators to sort them by */
	Oid		   *collations;		/* OIDs of collations */
	bool	   *nullsFirst;		/* NULLS FIRST/LAST directions */
} GpuSortInfo;

static inline void
form_gpusort_info(CustomScan *cscan, GpuSortInfo *gs_info)
{
	List   *privs = NIL;
	List   *temp;

	privs = lappend(privs, makeString(pstrdup(gs_info->kern_source)));
	privs = lappend(privs, makeInteger(gs_info->extra_flags));
	privs = lappend(privs, gs_info->used_params);
	privs = lappend(privs, makeInteger(gs_info->num_chunks));
	privs = lappend(privs, makeInteger(gs_info->chunk_size));
	privs = lappend(privs, makeInteger(gs_info->numCols));
	/* sortColIdx */
	for (temp = NIL, i=0; i < gs_info->numCols; i++)
		temp = lappend_int(temp, gs_info->sortColIdx[i]);
	privs = lappend(privs, temp);
	/* sortOperators */
	for (temp = NIL, i=0; i < gs_info->numCols; i++)
		temp = lappend_oid(temp, gs_info->sortOperators[i]);
	privs = lappend(privs, temp);
	/* collations */
	for (temp = NIL, i=0; i < gs_info->numCols; i++)
		temp = lappend_oid(temp, gs_info->collations[i]);
	privs = lappend(privs, temp);
	/* nullsFirst */
	for (temp = NIL, i=0; i < gs_info->numCols; i++)
		temp = lappend_int(temp, gs_info->nullsFirst[i]);
	privs = lappend(privs, tmep);

	cscan->custom_private = privs;
}

static inline GpuSortInfo *
deform_gpusort_info(CustomScan *cscan)
{
	GpuSortInfo	   *gs_info = palloc0(sizeof(GpuSortInfo));
	List		   *privs = cscan->custom_private;
	List		   *temp;
	ListCell	   *cell;
	int				pindex = 0;
	int				i;

	gs_info->kern_source = strVal(list_nth(privs, pindex++));
	gs_info->extra_flags = intVal(list_nth(privs, pindex++));
	gs_info->used_params = list_nth(privs, pindex++);
	gs_info->num_chunks = intVal(list_nth(privs, pindex++));
	gs_info->chunk_size = intVal(list_nth(privs, pindex++));
	gs_info->numCols = intVal(list_nth(privs, pindex++));
	/* sortColIdx */
	temp = list_nth(privs, pindex++);
	Assert(list_length(temp) == gs_info->numCols);
	gs_info->sortColIdx = palloc0(sizeof(AttrNumber) * gs_info->numCols);
	i = 0;
	foreach (cell, temp)
		gs_info->sortColIdx[i++] = lfirst_int(cell);

	/* sortOperators */
	temp = list_nth(privs, pindex++);
    Assert(list_length(temp) == gs_info->numCols);
	gs_info->sortOperators = palloc0(sizeof(Oid) * gs_info->numCols);
	i = 0;
	foreach (cell, temp)
		gs_info->sortOperators[i++] = lfirst_oid(cell);

	/* collations */
	temp = list_nth(privs, pindex++);
    Assert(list_length(temp) == gs_info->numCols);
	gs_info->collations = palloc0(sizeof(Oid) * gs_info->numCols);
	i = 0;
	foreach (cell, temp)
		gs_info->collations[i] = lfirst_oid(cell);

	/* nullsFirst */
	temp = list_nth(privs, pindex++);
    Assert(list_length(temp) == gs_info->numCols);
	gs_info->nullsFirst = palloc0(sizeof(bool) * gs_info->numCols);
	i = 0;
	foreach (cell, temp)
		gs_info->nullsFirst[i] = lfirst_int(cell);

	return gs_info;
}

typedef struct
{
	CustomScanState	css;

	/* for GPU bitonic sorting */
	pgstrom_queue  *mqueue;
	Datum			dprog_key;
	const char	   *kern_source;
	kern_parambuf  *kparambuf;
	pgstrom_perfmon	pfm;

	/* for CPU merge sorting */
	int				num_chunks;		/* number of valid sorting chunks */
	int				limit_chunks;	/* length of sort_chunks array */
	pgstrom_data_store **sort_chunks;

	Size			chunk_size;		/* expected best size of sorting chunk */
    int				numCols;		/* number of sort-key columns */
    AttrNumber	   *sortColIdx;		/* their indexes in the target list */
    Oid			   *sortOperators;	/* OIDs of operators to sort them by */
	Oid			   *collations;		/* OIDs of collations */
	bool		   *nullsFirst;		/* NULLS FIRST/LAST directions */

	/* running status */
	cl_int			num_gpu_running;
	cl_int			num_cpu_running;
	pgstrom_cpusort *sorted_chunk;	/* sorted but no pair chunk yet */
	dlist_head		pending_chunks;	/* chunks waiting for cpu sort */
	dlist_head		running_chunks;	/* chunks in running by bgworker */

} GpuSortState;

/*
 * CpuSortInfo - State object for CPU sorting. It shall be allocated on the
 * private memory, then dynamic worker will be able to reference the copy
 * because of process fork(2). It means, we don't mention about Windows
 * platform at this moment. :-)
 */
typedef struct
{
	dlist_node		chain;
	BackgroundWorkerHandle *bgw_handle;	/* valid, if running */

	dsm_segment	   *ritems_dsm;
	dsm_segment	   *litems_dsm;
	dsm_segment	   *oitems_dsm;

	/* sorting chunks */
	uint			num_chunks;
	pgstrom_data_store **sort_chunks;
	/* sorting keys */
	TupleDesc		tupdesc;
	int				numCols;
	AttrNumber	   *sortColIdx;
	Oid			   *sortOperators;
	Oid			   *collations;
	bool		   *nullsFirst;
} pgstrom_cpusort;









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
 * cost_gpusort
 *
 * cost estimation for GpuSort
 */
#define LOG2(x)		(log(x) / 0.693147180559945)

static void
cost_gpusort(Cost *p_startup_cost, Cost *p_total_cost,
			 long *p_num_chunks, Size *p_chunk_size,
			 Plan *subplan)
{
	Cost	subplan_total = subplan->total_cost;
	double	ntuples = subplan->plan_rows;
	int		width = subplan->plan_width;
	int		nattrs = list_length(subplan->targetlist);
	Cost	cpu_comp_cost = 2.0 * cpu_operator_cost;
	Cost	gpu_comp_cost = 2.0 * pgstrom_gpu_operator_cost;
	Cost	startup_cost = subplan_total;
	Cost	run_cost = 0.0;
	double	nrows_per_chunk;
	long	num_chunks;
	Size	chunk_size;

	if (ntuples < 2.0)
		ntuples = 2.0;

	/*
	 * calculate expected number of rows per chunk and number of chunks.
	 */
	width = MAXALIGN(width + sizeof(cl_uint) +
					 offsetof(HeapTupleHeaderData, t_bits) +
					 BITMAPLEN(list_length(subplan->targetlist)));
	chunk_size = (Size)((double)width * ntuples * 1.15) + 1024;
	if (pgstrom_shmem_maxalloc() > chunk_size)
		num_chunks = 1;
	else
	{
		double		nrows_per_chunk =
			(double)(pgstrom_shmem_maxalloc() - 1024) / ((double)width * 1.25);
		chunk_size = (Size)((double)width * nrows_per_chunk * 1.25);
		num_chunks = Max(1.0, floor(ntuples / nrows_per_chunk + 0.9999));
	}

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
	*p_num_chunks = num_chunks;
	*p_chunk_size = chunk_size;
}


static char *
pgstrom_gpusort_codegen(Sort *sort, codegen_context *context)
{
	StringInfoData	decl;
	StringInfoData	body;
	StringInfoData	result;
	int		i;

	initStringInfo(&decl);
	initStringInfo(&body);
	initStringInfo(&result);

	for (i=0; i < sort->numCols; i++)
	{
		TargetEntry *tle;
		AttrNumber	colidx = sort->sortColIdx[i];
		Oid			sort_op = sort->sortOperators[i];
		Oid			sort_func;
		Oid			sort_type;
		Oid			opfamily;
		Oid			opcintype;
		int16		strategy;
		bool		null_first = sort->nullsFirst[i];
		devtype_info *dtype;
		devfunc_info *dfunc;

		/* Find the operator in pg_amop */
		if (!get_ordering_op_properties(sort_op,
										&opfamily,
										&sort_type,
										&strategy))
			elog(ERROR, "operator %u is not a valid ordering operator",
				 sort_op);
		is_reverse = (strategy == BTGreaterStrategyNumber);

		/* Find the sort support function */
		sort_func = get_opfamily_proc(opfamily, sort_type, sort_type,
									  BTORDER_PROC);
		if (!OidIsValid(sort_func))
			elog(ERROR, "missing support function %d(%u,%u) in opfamily %u",
				 BTORDER_PROC, opcintype, opcintype, opfamily);

		/* Sanity check of the data type */
		tle = get_tle_by_resno(sort->plan.targetlist, colidx);
		if (exprType(tle->expr) != sort_type)
			elog(ERROR, "Bug? type mismatch tlist:%u /catalog:%u",
				 exprType(tle->expr), sort_type);

		/* device type for comparison */
		sort_type = exprType((Node *) varnode);
		dtype = pgstrom_devtype_lookup_and_track(sort_type, context);
		if (!dtype)
			elog(ERROR, "device type %u lookup failed", sort_type);

		/* device function for comparison */
		sort_func = sort_type->type_cmpfunc;
		dfunc = pgstrom_devfunc_lookup_and_track(sort_func, context);
		if (!dfunc)
			elog(ERROR, "device function %u lookup failed", sort_func);

		/* reference to X-variable / Y-variable */
		appendStringInfo(
			&decl,
			"  pg_%s_t KVAR_X%d"
			" = pg_%s_vref(kds,ktoast,errcode,%d,x_index);\n"
			"  pg_%s_t KVAR_Y%d"
			" = pg_%s_vref(kds,ktoast,errcode,%d,y_index);\n",
			dtype->type_name, i+1,
			dtype->type_name, attno,
			dtype->type_name, i+1,
			dtype->type_name, attno);

		/* logic to compare */
		appendStringInfo(
			&body,
			"  /* sort key comparison on the resource %d */"
			"  if (!KVAR_X%d.isnull && !KVAR_Y%d.isnull)\n"
			"  {\n"
			"    comp = pgfn_%s(errcode, KVAR_X%d, KVAR_Y%d);\n"
			"    if (comp.value != 0)\n"
			"      return %s;\n"
			"  }\n"
			"  else if (KVAR_X%d.isnull && !KVAR_Y%d.isnull)\n"
			"    return %d;\n"
			"  else if (!KVAR_X%d.isnull && KVAR_Y%d.isnull)\n"
			"    return %d;\n",
			i+1,
			i+1, i+1,
			dfunc->func_name, i+1, i+1,
			is_reverse ? "-comp.value" : "comp.value",
			i+1, i+1, nullfirst ? -1 : 1,
			i+1, i+1, nullfirst ? 1 : -1);
	}
	/* make a comparison function */
	appendStringInfo(&result,
					 "%s\n"		/* function declaration */
					 "static cl_int\n"
					 "gpusort_keycomp(__private cl_int *errcode,\n"
					 "                __global kern_data_store *kds,\n"
					 "                __global kern_data_store *ktoast,\n"
					 "                size_t x_index,\n"
					 "                size_t y_index)\n"
					 "{\n"
					 "%s\n"		/* variables declaration */
					 "  pg_int4_t comp;\n"
					 "%s"		/* comparison body */
					 "  return 0;\n"
					 "}\n",
					 pgstrom_codegen_func_declarations(context),
					 decl.data,
					 body.data);
	pfree(decl.data);
	pfree(body.data);

	return result.data;
}

void
pgstrom_try_insert_gpusort(PlannedStmt *pstmt, Plan **p_plan)
{
	Sort	   *sort = (Sort *)(*p_plan);
	List	   *tlist = sort->plan.targetlist;
	List	   *rtable = pstmt->rtable;
	ListCell   *cell;
	Cost		startup_cost;
	Cost		total_cost;
	long		num_chunks;
	Size		chunk_size;
	Plan	   *subplan;
	GpuSortInfo	gs_info;
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
	}

	/*
	 * OK, cost estimation with GpuSort
	 */
	cost_gpusort(&startup_cost, &total_cost,
				 &num_chunks, &chunk_size,
				 subplan);
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
	gs_info.kern_source = gpusort_codegen(sort, &context);
	gs_info.extra_flags = context.extra_flags | DEVKERNEL_NEEDS_GPUSORT |
		(!devprog_enable_optimize ? DEVKERNEL_DISABLE_OPTIMIZE : 0);
	gs_info.used_params = context.used_params;
	gs_info.num_chunks = num_chunks;
	gs_info.chunk_size = chunk_size;
	gs_info.numCols = sort->numCols;
	gs_info.sortColIdx = sort->sortColIdx;
	gs_info.sortOperators = sort->sortOperators;
	gs_info.collations = sort->collations;
	gs_info.nullsFirst = sort->nullsFirst;
	form_gpusort_info(cscan, &gs_info);

	*p_plan = &cscan->scan.plan;
}

static Node *
gpusort_create_scan_state(CustomScan *cscan)
{
	GpuSortState   *gss = palloc0(sizeof(GpuSortState));

	NodeSetTag(gss, T_CustomScanState);
	gss->css.methods = &gpusort_exec_methods;

	return (Node *) gss;
}

static void
gpusort_begin(CustomScanState *node, EState *estate, int eflags)
{
	GpuSortState   *gss = (GpuSortState *) node;
	CustomScan	   *cscan = (CustomScan *)node->ss.ps.plan;
	GpuSortInfo	   *gs_info = deform_gpusort_info(cscan);
	PlanState	   *ps = node->ss.ps;
	

	/* for GPU bitonic sorting */
	gss->kparams = pgstrom_create_kern_parambuf(gs_info->used_params,
												ps->ps_ExprContext);
	Assert(gs_info->kern_source != NULL);
	gss->dprog_key = pgstrom_get_devprog_key(ghj_info->kernel_source,
											 ghj_info->extra_flags);
	gss->kern_source = gs_info->kern_source;
	pgstrom_track_object((StromObject *)gss->dprog_key, 0);

	gss->mqueue = pgstrom_create_queue();
	pgstrom_track_object(&gss->mqueue->sobj, 0);

	/* Is perfmon needed? */
	gss->pfm.enabled = pgstrom_perfmon_enabled;


	/**/
	gss->num_chunks = 0;
	gss->limit_chunks = 2 * gs_info->num_chunks;
	gss->sort_chunks = palloc0(sizeof(pgstrom_data_store *) *
							   gss->limit_chunks);
	gss->chunk_size = gs_info->chunk_size;
	gss->numCols = gs_info->numCols;
	gss->sortColIdx = gs_info->sortColIdx;
	gss->sortOperators = gs_info->sortOperators;
	gss->collations = gs_info->collations;
	gss->nullsFirst = gs_info->nullsFirst;

}

static void
pgstrom_release_gpusort(pgstrom_message *msg)
{
	pgstrom_gpusort	   *gpusort = (pgstrom_gpusort *) msg;

	/* unlink message queue and device program */
	pgstrom_put_queue(msg->respq);
	pgstrom_put_devprog_key(gpusort->dprog_key);

	pgstrom_put_data_store(gpusort->pds);

	pgstrom_shmem_free(gpusort);
}

static pgstrom_gpusort *
pgstrom_create_gpusort(GpuSortState *gss)
{
	pgstrom_data_store *pds = NULL;
	TupleTableSlot *slot;
	TupleDesc		tupdesc;
	int				chunk_id = -1;

	/*
	 * Load tuples from the underlying plan node
	 */
	for (;;)
	{
		if (gss->overflow_slot != NULL)
		{
			slot = gss->overflow_slot;
			gss->overflow_slot = NULL;
		}
		else
		{
			slot = ExecProcNode(outerPlanState(gss));
			if (TupIsNull(slot))
			{
				gss->scan_done = true;
				slot = NULL;
				break;
			}
		}
		Assert(!TupIsNull(slot));

		/* Makes a sorting chunk on the first tuple */
		if (!pds)
		{
			tupdesc = gss->css.ss.ps.ps_ResultTupleSlot->tts_tupleDescriptor;
			pds = pgstrom_create_data_store_row_fmap(tupdesc, gss->chunk_size);
			chunk_id = gss->num_chunks++;
			if (chunk_id >= gss->limit_chunks)
			{
				pgstrom_data_store *new_chunks
					= repalloc(gss->sort_chunks,
							   sizeof(pgstrom_data_store *) *
							   2 * gss->limit_chunks);
				pfree(ss->sort_chunks);
				gss->sort_chunks = new_chunks;
				gss->limit_chunks = 2 * gss->limit_chunks;
			}
			gss->sort_chunks[chunk_id] = pds;
			pgstrom_track_object(&pds->sobj, 0);
		}
		/* Then, insert this tuple to the data store */
		if (!pgstrom_data_store_insert_tuple(pds, slot))
		{
			gss->overflow_slot = slot;
			break;
		}
	}

	if (!pds)
		return NULL;

	/* Makea a pgstrom_gpusort object based on the data-store */
	nitems = pds->kds->nitems;
	length = (STROMALIGN(offsetof(pgstrom_gpusort, kern.kparams)) +
			  STROMALIGN(gss->kparams->length) +
			  STROMALIGN(offsetof(kern_resultbuf, results[2 * nitems])));
	gpusort = pgstrom_shmem_alloc(length);
	if (!gpusort)
		elog(ERROR, "out of shared memory");

	/* common field of pgstrom_gpusort */
	pgstrom_init_message(&gpusort->msg,
                         StromTag_GpuSort,
						 gss->mqueue,
						 clserv_process_gpusort,
						 pgstrom_release_gpusort,
						 gss->pfm.enabled);
	gpusort->dprog_key = pgstrom_retain_devprog_key(gss->dprog_key);
	gpusort->chunk_id = chunk_id;
	gpusort->pds = pgstrom_get_data_store(pds);
	memcpy(KERN_GPUSORT_PARAMBUF(&gpusort->kern),
		   gss->kparams,
		   gss->kparams->length);
	kresults = KERN_GPUSORT_RESULTBUF(&gpusort->kernel);
	memset(kresults, 0, sizeof(kern_resultbuf));
	kresults->nrels = 2;
	kresults->nrooms = nitems;
	pgstrom_track_object(&gpusort->sobj, 0);

	return gpusort;
}

static void
gpusort_launch_cpusort_worker(GpuSortState *gss, pgstrom_cpusort *cpusort)
{
	BackgroundWorker	worker;

	/* set up dynamic background worker */
	memset(&worker, 0, sizeof(BackgroundWorker));
	snprintf(worker.bgw_name, sizeof(worker.bgw_name),
			 "GpuSort worker-%u", gss->cpusort_seqno++);
	worker.bgw_flags = BGWORKER_SHMEM_ACCESS |
        BGWORKER_BACKEND_DATABASE_CONNECTION;
    worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
	worker.bgw_restart_time = BGW_NEVER_RESTART;
	worker.bgw_main = gpusort_cpusort_main;
	worker.bgw_main_arg = PointerGetDatum(cpusort);
	/* XXX - we may need to serialize when we support windows shmem */

	if (!RegisterDynamicBackgroundWorker(&worker, &cpusort->bgw_handle))
		dlist_push_tail(&gss->pending_chunks, &cpusort->chain);
	else
	{
		dlist_push_tail(&gss->running_chunks, &cpusort->chain);
		gss->num_cpu_running++;
	}
}

static void
gpusort_check_cpusort_worker(GpuSortState *gss)
{
	dlist_mutable_iter	iter;

	dlist_foreach_modify(iter, &gss->running_chunks)
	{
		pgstrom_cpusort	   *cpusort;
		BgwHandleStatus		status;

		cpusort = dlist_container(pgstrom_cpusort, chain, iter.cur);
		status = GetBackgroundWorkerPid(cpusort->bgw_handle, &bgw_pid);

		if (status == BGWH_STOPPED)
		{
			dlist_delete(&cpusort->chain);
			/* decrement num_cpu_running */

			/* check results status */

			/* call the gpusort_merge_cpu_chunks */

		}
		else if (status == BGWH_POSTMASTER_DIED)
		{
			/* emergency. kills all the backend and raise an error */
			hoge;
		}
	}

}

static void
gpusort_merge_cpu_chunks(GpuSortState *gss, pgstrom_cpusort *cpusort_1)
{
	pgstrom_cpusort	   *cpusort_2;
	kern_resultbuf	   *kresults_1;
	kern_resultbuf	   *kresults_2;
	kern_resultbuf	   *kresults_next;
	cl_uint				nitems;

	/* ritems and litems are no longer referenced */
	dsm_detach(cpusort_1->ritems_dsm);
	cpusort_1->ritems_dsm = NULL;
	dsm_detach(cpusort_1->litems_dsm);
	cpusort_1->litems_dsm = NULL;

	if (!gss->sorted_chunk)
	{
		gss->sorted_chunk = cpusort_1;
		return;
	}
	cpusort_2 = gss->sorted_chunk;
	Assert(cpusort_2->oitems_dsm != NULL &&
		   !cpusort_2->ritems_dsm &&
		   !cpusort_2->litems_dsm);
	kresults_1 = dsm_segment_address(cpusort_1->oitems_dsm);
	kresults_2 = dsm_segment_address(cpusort_2->oitems_dsm);
	Assert(kresults_1->nrels == 2 && kresults_2->nrels == 2);
	nitems = kresults_1->nitems + kresults_2->nitems;

	cpusort_2->ritems_dsm = cpusort_1->oitems_dsm;
	cpusort_2->litems_dsm = cpusort_2->oitems_dsm;
	length = STROMALIGN(offsetof(kern_resultbuf,
								 kresults_1->nrels * nitems));

	cpusort_2->oitems_dsm = dsm_create(length);
	kresults_next = dsm_segment_address(cpusort_2->oitems_dsm);
	memset(kresults_new, 0, sizeof(kern_resultbuf));
	kresults_next->nrels = 2;
	kresults_next->nrooms = nitems;

	/* OK, kick a dynamic worker process */
	pfree(cpusort_1);
	gss->sorted_chunk = NULL;

	hogehoge;
}

static void
gpusort_merge_gpu_chunks(GpuSortState *gss, pgstrom_gpusort *gpusort)
{
	pgstrom_cpusort	   *cpusort;
	EState			   *estate = gss->css.ss.ps.state;
	MemoryContext		memcxt = estate->es_query_cxt;
	kern_resultbuf	   *kresults_1;
	kern_resultbuf	   *kresults_2;
	Size				length;

	if (!gss->sorted_chunk)
	{
		cpusort = MemoryContextAllocZero(memcxt, sizeof(pgstrom_cpusort));
		cpusort->num_chunks = gss->num_chunks;
		cpusort->sort_chunks = gss->sort_chunks;
		cpusort->tupdesc = gss->css.ss.ss_ScanTupleSlot->tts_tupleDescriptor;
		cpusort->numCols = gss->numCols;
		cpusort->sortColIdx = gss->sortColIdx;
		cpusort->sortOperators = gss->sortOperators;
		cpusort->collations = gss->collations;
		cpusort->nullsFirst = gss->nullsFirst;

		kresults_1 = KERN_GPUSORT_RESULTBUF(&gpusort->kern);
		length = STROMALIGN(offsetof(kern_resultbuf,
									 kresults_1->nrels *
									 kresults_1->nitems));
		cpusort->oitems_dsm = dsm_create(length);
		cpusort->ritems_dsm = NULL;		/* to be set later */
		cpusort->litems_dsm = NULL;		/* to be set later */
		kresults_2 = dsm_segment_address(cpusort->oitems_dsm);
		memcpy(kresults_2, kresults_1, length);

		/* pds is still valid, but gpusort is no longer referenced */
		pgstrom_put_message(gpusort->msg);

		/* to be merged with next chunk */
		gss->sorted_chunk = cpusort;
		return;
	}
	cpusort = gss->sorted_chunk;
	Assert(cpusort->oitems_dsm != NULL &&
		   !cpusort->ritems_dsm &&
		   !cpusort->litems_dsm);
	kresults_1 = KERN_GPUSORT_RESULTBUF(&gpusort->kern);
	length = STROMALIGN(offsetof(kern_resultbuf,
								 kresults_1->nrels *
								 kresults_1->nitems));
	cpusort->ritems_dsm = oitems_dsm;
	cpusort->litems_dsm = dsm_create(length);
	kresults_2 = dsm_segment_address(cpusort->litems_dsm);
	memcpy(kresults_2, kresults_1, length);

	/* pds is still valid, but gpusort is no longer referenced */
	pgstrom_put_message(gpusort->msg);

	/* to be merged with next chunk */
	gss->sorted_chunk = NULL;

	/* try to kick background worker */
	hogehoge;
}

static void
gpusort_exec_sort(GpuSortState *gss)
{
	pgstrom_gpusort	   *gpusort;

	while (!gss->scan_done ||
		   gss->num_gpu_running > 0 ||
		   gss->num_cpu_running > 0)
	{
		bool	launch_pending = false;

		/*  */
		if (!gss->scan_done)
		{
			gpusort = pgstrom_create_gpusort(gss);
			if (gpusort)
			{
				if (!pgstrom_enqueue_message(&gpusort->msg))
					elog(ERROR, "Bug? OpenCL server seems to dead");
				gss->num_gpu_running++;
			}
		}

		/* fetch and merge if pair is found */
		while ((msg = pgstrom_try_dequeue_message(mqueue)) != NULL)
		{
			Assert(gss->num_gpu_running > 0);
			gss->num_gpu_running--;

			if (msg->pfm.enabled)
				pgstrom_perfmon_add(&gss->pfm, &msg->pfm);
			Assert(StromTagIs(msg, GpuSort));
			gpusort = (pgstrom_gpusort *) msg;
			gpusort_merge_chunks(gss, gpusort);
		}

		/* also checks result of CPU sorting */
		while (fetch a CPU sort result)
		{
			Assert(gss->num_cpu_running > 0);
			gss->num_cpu_running--;
			launch_pending = true;

			if (msg->pfm.enabled)
				pgstrom_perfmon_add(&gss->pfm, &msg->pfm);
			Assert(StromTagIs(msg, GpuSort));
			gpusort = (pgstrom_gpusort *) msg;
			gpusort_merge_chunks(gss, gpusort);
		}

		/***/
		if (gss->scan_done)
		{



		}
	}








}


static TupleTableSlot *
gpusort_exec(CustomScanState *node)
{
	GpuSortState   *gss = (GpuSortState *) node;

	if (!gss->sort_done)
		gpusort_exec_sort();




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



