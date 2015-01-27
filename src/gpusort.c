/*
 * gpusort.c
 *
 * GPU+CPU Hybrid Sorting
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
#include "access/xact.h"
#include "commands/dbcommands.h"
#include "nodes/nodeFuncs.h"
#include "nodes/makefuncs.h"
#include "optimizer/cost.h"
#include "parser/parsetree.h"
#include "postmaster/bgworker.h"
#include "storage/dsm.h"
#include "storage/latch.h"
#include "storage/proc.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/ruleutils.h"
#include "utils/snapmgr.h"
#include "pg_strom.h"
#include "opencl_gpusort.h"
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static CustomScanMethods	gpusort_scan_methods;
static CustomExecMethods	gpusort_exec_methods;
static bool					enable_gpusort;
static bool					debug_force_gpusort;
static int					gpusort_max_workers;

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
	int		i;

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
	privs = lappend(privs, temp);

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

/*
 * pgstrom_cpusort - information of CPU sorting. It shall be allocated on
 * the private memory, then dynamic worker will be able to reference the copy
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
	pgstrom_data_store **pds_chunks;
	/* sorting keys */
	char		   *dbname;
	TupleDesc		tupdesc;
	int				numCols;
	AttrNumber	   *sortColIdx;
	Oid			   *sortOperators;
	Oid			   *collations;
	bool		   *nullsFirst;
} pgstrom_cpusort;


//hoge
typedef struct
{
	CustomScanState	css;

	/* data store saved in temporary file */
	int				num_chunks;
	int				num_chunks_limit;
	pgstrom_data_store **pds_chunks;

	/* for GPU bitonic sorting */
	pgstrom_queue  *mqueue;
	Datum			dprog_key;
	const char	   *kern_source;
	kern_parambuf  *kparams;
	pgstrom_perfmon	pfm;

	/* copied from the plan node */
	Size			chunk_size;		/* expected best size of sorting chunk */
    int				numCols;		/* number of sort-key columns */
    AttrNumber	   *sortColIdx;		/* their indexes in the target list */
    Oid			   *sortOperators;	/* OIDs of operators to sort them by */
	Oid			   *collations;		/* OIDs of collations */
	bool		   *nullsFirst;		/* NULLS FIRST/LAST directions */

	/* running status */
	char		   *database_name;	/* name of the current database */
	pgstrom_cpusort *sorted_chunk;	/* sorted but no pair chunk yet */
	cl_int			num_gpu_running;
	cl_int			num_cpu_running;
	dlist_head		pending_cpu_chunks;	/* chunks waiting for cpu sort */
	dlist_head		running_cpu_chunks;	/* chunks in running by bgworker */
	bool			scan_done;		/* if true, no tuples to read any more */
	bool			sort_done;		/* if true, now ready to fetch records */
	cl_int			cpusort_seqno;	/* seqno of cpusort to launch */

	/* final result */
	dsm_segment	   *sorted_result;	/* final index of the sorted result */
	cl_long			sorted_index;	/* index of the final index on scan */
	HeapTupleData	tuple_buf;		/* temp buffer during scan */
	TupleTableSlot *overflow_slot;
} GpuSortState;

/*
 * static function declarations
 */
static void clserv_process_gpusort(pgstrom_message *msg);
static void gpusort_entrypoint_cpusort(Datum main_arg);

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
		nrows_per_chunk = 
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
		Var		   *varnode;
		AttrNumber	colidx = sort->sortColIdx[i];
		Oid			sort_op = sort->sortOperators[i];
		Oid			sort_collid = sort->collations[i];
		Oid			sort_func;
		Oid			sort_type;
		Oid			opfamily;
		int16		strategy;
		bool		null_first = sort->nullsFirst[i];
		bool		is_reverse;
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
				 BTORDER_PROC, sort_type, sort_type, opfamily);

		/* Sanity check of the expression */
		tle = get_tle_by_resno(sort->plan.targetlist, colidx);
		if (!tle || !IsA(tle->expr, Var))
			elog(ERROR, "Bug? resno %d not found on tlist or not varnode: %s",
				 colidx, nodeToString(tle->expr));
		varnode = (Var *) tle->expr;
		if (varnode->vartype != sort_type)
			elog(ERROR, "Bug? type mismatch \"%s\" is expected, but \"%s\"",
				 format_type_be(sort_type),
				 format_type_be(varnode->vartype));

		/* device type for comparison */
		dtype = pgstrom_devtype_lookup_and_track(sort_type, context);
		if (!dtype)
			elog(ERROR, "device type %u lookup failed", sort_type);

		/* device function for comparison */
		sort_func = dtype->type_cmpfunc;
		dfunc = pgstrom_devfunc_lookup_and_track(sort_func,
												 sort_collid,
												 context);
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
			dtype->type_name, colidx,
			dtype->type_name, i+1,
			dtype->type_name, colidx);

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
			i+1, i+1, null_first ? -1 : 1,
			i+1, i+1, null_first ? 1 : -1);
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
	ListCell   *cell;
	Cost		startup_cost;
	Cost		total_cost;
	long		num_chunks;
	Size		chunk_size;
	CustomScan *cscan;
	Plan	   *subplan;
	GpuSortInfo	gs_info;
	codegen_context context;
	int			i;

	/* ensure the plan is Sort */
	Assert(IsA(sort, Sort));
	Assert(sort->plan.qual == NIL);
	Assert(sort->plan.righttree == NULL);
	subplan = outerPlan(sort);

	for (i=0; i < sort->numCols; i++)
	{
		TargetEntry	   *tle = get_tle_by_resno(tlist, sort->sortColIdx[i]);
		Var			   *varnode = (Var *) tle->expr;
		devfunc_info   *dfunc;
		devtype_info   *dtype;

		/*
		 * Target-entry of Sort plan should be a var-node that references
		 * a particular column of underlying relation, even if Sort-key
		 * contains formula. So, we can expect a simple var-node towards
		 * outer relation here.
		 */
		if (!IsA(varnode, Var) || varnode->varno != OUTER_VAR)
			return;

		dtype = pgstrom_devtype_lookup(exprType((Node *) varnode));
		if (!dtype || !OidIsValid(dtype->type_cmpfunc))
			return;

		dfunc = pgstrom_devfunc_lookup(dtype->type_cmpfunc,
									   sort->collations[i]);
		if (!dfunc)
			return;
	}

	/*
	 * OK, cost estimation with GpuSort
	 */
	cost_gpusort(&startup_cost, &total_cost,
				 &num_chunks, &chunk_size,
				 subplan);

	elog(DEBUG1,
		 "GpuSort (cost=%.2f..%.2f) has%sadvantage to Sort (cost=%.2f..%.2f)",
		 startup_cost,
		 total_cost,
		 total_cost >= sort->plan.total_cost ? " no " : " ",
		 sort->plan.startup_cost,
		 sort->plan.total_cost);

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
		TargetEntry	   *tle = lfirst(cell);
		TargetEntry	   *tle_new;
		Var			   *varnode;

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
	gs_info.kern_source = pgstrom_gpusort_codegen(sort, &context);
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
	PlanState	   *ps = &node->ss.ps;

	/* initialize child exec node */
	outerPlanState(gss) = ExecInitNode(outerPlan(cscan), estate, eflags);

	/* for GPU bitonic sorting */
	gss->kparams = pgstrom_create_kern_parambuf(gs_info->used_params,
												ps->ps_ExprContext);
	Assert(gs_info->kern_source != NULL);
	gss->dprog_key = pgstrom_get_devprog_key(gs_info->kern_source,
											 gs_info->extra_flags);
	gss->kern_source = gs_info->kern_source;
	pgstrom_track_object((StromObject *)gss->dprog_key, 0);

	gss->mqueue = pgstrom_create_queue();
	pgstrom_track_object(&gss->mqueue->sobj, 0);

	/* Is perfmon needed? */
	gss->pfm.enabled = pgstrom_perfmon_enabled;

	/**/
	gss->num_chunks = 0;
	gss->num_chunks_limit = gs_info->num_chunks + 32;
	gss->pds_chunks = palloc0(sizeof(pgstrom_data_store *) *
							  gss->num_chunks_limit);
	gss->chunk_size = gs_info->chunk_size;

	gss->numCols = gs_info->numCols;
	gss->sortColIdx = gs_info->sortColIdx;
	gss->sortOperators = gs_info->sortOperators;
	gss->collations = gs_info->collations;
	gss->nullsFirst = gs_info->nullsFirst;

	/* running status */
	gss->database_name = get_database_name(MyDatabaseId);
	gss->sorted_chunk = NULL;
	gss->num_gpu_running = 0;
	gss->num_cpu_running = 0;
	dlist_init(&gss->pending_cpu_chunks);
	dlist_init(&gss->running_cpu_chunks);
	gss->scan_done = false;
	gss->sort_done = false;
	gss->cpusort_seqno = 0;

	/* final result */
	gss->sorted_result = NULL;
	gss->sorted_index = 0;
	gss->overflow_slot = NULL;
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
	pgstrom_gpusort	   *gpusort = NULL;
	kern_resultbuf	   *kresults;
	pgstrom_data_store *pds = NULL;
	TupleTableSlot	   *slot;
	TupleDesc			tupdesc;
	cl_uint				nitems;
	Size				length;
	cl_int				chunk_id = -1;
	struct timeval		tv1, tv2;

	if (gss->pfm.enabled)
		gettimeofday(&tv1, NULL);

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
			if (chunk_id >= gss->num_chunks_limit)
			{
				pgstrom_data_store **new_pds_chunks;

				/* expand array twice */
				gss->num_chunks_limit = 2 * gss->num_chunks_limit;
				new_pds_chunks = repalloc(gss->pds_chunks,
										  sizeof(pgstrom_data_store *) *
										  gss->num_chunks_limit);
				pfree(gss->pds_chunks);
				gss->pds_chunks = new_pds_chunks;
			}
			gss->pds_chunks[chunk_id] = pds;
			pgstrom_track_object(&pds->sobj, 0);
		}
		/* Then, insert this tuple to the data store */
		if (!pgstrom_data_store_insert_tuple(pds, slot))
		{
			gss->overflow_slot = slot;
			break;
		}
	}

	/* can we read any tuples? */
	if (!pds)
		goto out;

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
	kresults = KERN_GPUSORT_RESULTBUF(&gpusort->kern);
	memset(kresults, 0, sizeof(kern_resultbuf));
	kresults->nrels = 2;
	kresults->nrooms = nitems;
	kresults->nitems = nitems;	/* no records will lost */
	pgstrom_track_object(&gpusort->msg.sobj, 0);
out:
	if (gss->pfm.enabled)
	{
		gettimeofday(&tv2, NULL);
		gss->pfm.time_outer_load += timeval_diff(&tv1, &tv2);
	}
	return gpusort;
}

static pgstrom_cpusort *
gpusort_merge_cpu_chunks(GpuSortState *gss, pgstrom_cpusort *cpusort_1)
{
	pgstrom_cpusort	   *cpusort_2;
	kern_resultbuf	   *kresults_1;
	kern_resultbuf	   *kresults_2;
	kern_resultbuf	   *kresults_next;
	cl_uint				nitems;
	Size				length;

	/* ritems and litems are no longer referenced */
	dsm_detach(cpusort_1->ritems_dsm);
	cpusort_1->ritems_dsm = NULL;
	dsm_detach(cpusort_1->litems_dsm);
	cpusort_1->litems_dsm = NULL;
	/* also, bgw_handle is no longer referenced */
	pfree(cpusort_1->bgw_handle);
	cpusort_1->bgw_handle = NULL;

	if (!gss->sorted_chunk)
	{
		gss->sorted_chunk = cpusort_1;
		return NULL;
	}
	cpusort_2 = gss->sorted_chunk;
	Assert(cpusort_2->oitems_dsm != NULL
		   && !cpusort_2->ritems_dsm
		   && !cpusort_2->litems_dsm);
	kresults_1 = dsm_segment_address(cpusort_1->oitems_dsm);
	kresults_2 = dsm_segment_address(cpusort_2->oitems_dsm);
	Assert(kresults_1->nrels == 2 && kresults_2->nrels == 2);
	nitems = kresults_1->nitems + kresults_2->nitems;

	cpusort_2->ritems_dsm = cpusort_1->oitems_dsm;
	cpusort_2->litems_dsm = cpusort_2->oitems_dsm;
	length = STROMALIGN(offsetof(kern_resultbuf, results[2 * nitems]));

	cpusort_2->oitems_dsm = dsm_create(length);
	kresults_next = dsm_segment_address(cpusort_2->oitems_dsm);
	memset(kresults_next, 0, sizeof(kern_resultbuf));
	kresults_next->nrels = 2;
	kresults_next->nrooms = nitems;

	/* Release either of them */
	pfree(cpusort_1);
	gss->sorted_chunk = NULL;

	return cpusort_2;
}

static pgstrom_cpusort *
gpusort_merge_gpu_chunks(GpuSortState *gss, pgstrom_gpusort *gpusort)
{
	pgstrom_cpusort	   *cpusort;
	EState			   *estate = gss->css.ss.ps.state;
	MemoryContext		memcxt = estate->es_query_cxt;
	kern_resultbuf	   *kresults_gpu;
	kern_resultbuf	   *kresults_cpu;
	Size				length;

	if (!gss->sorted_chunk)
	{
		TupleTableSlot *slot = gss->css.ss.ss_ScanTupleSlot;

		cpusort = MemoryContextAllocZero(memcxt, sizeof(pgstrom_cpusort));
		cpusort->num_chunks = gss->num_chunks;
		cpusort->pds_chunks = gss->pds_chunks;
		cpusort->dbname = gss->database_name;
		cpusort->tupdesc = slot->tts_tupleDescriptor;
		cpusort->numCols = gss->numCols;
		cpusort->sortColIdx = gss->sortColIdx;
		cpusort->sortOperators = gss->sortOperators;
		cpusort->collations = gss->collations;
		cpusort->nullsFirst = gss->nullsFirst;

		kresults_gpu = KERN_GPUSORT_RESULTBUF(&gpusort->kern);
		length = STROMALIGN(offsetof(kern_resultbuf,
									 results[kresults_gpu->nrels *
											 kresults_gpu->nitems]));
		cpusort->oitems_dsm = dsm_create(length);
		cpusort->ritems_dsm = NULL;		/* to be set later */
		cpusort->litems_dsm = NULL;		/* to be set later */
		kresults_cpu = dsm_segment_address(cpusort->oitems_dsm);
		memcpy(kresults_cpu, kresults_gpu, length);

		/* pds is still valid, but gpusort is no longer referenced */
		pgstrom_untrack_object(&gpusort->msg.sobj);
		pgstrom_put_message(&gpusort->msg);

		/* to be merged with next chunk */
		gss->sorted_chunk = cpusort;
		return NULL;
	}
	cpusort = gss->sorted_chunk;
	Assert(cpusort->oitems_dsm != NULL &&
		   !cpusort->ritems_dsm &&
		   !cpusort->litems_dsm);
	kresults_gpu = KERN_GPUSORT_RESULTBUF(&gpusort->kern);
	length = STROMALIGN(offsetof(kern_resultbuf,
								 results[kresults_gpu->nrels *
										 kresults_gpu->nitems]));
	cpusort->ritems_dsm = cpusort->oitems_dsm;
	cpusort->litems_dsm = dsm_create(length);
	kresults_cpu = dsm_segment_address(cpusort->litems_dsm);
	memcpy(kresults_cpu, kresults_gpu, length);

	elog(INFO, "merge gpu + cpu chunks (%u + %u) -> %u",
		 kresults_gpu->nitems, kresults_cpu->nitems,
		 kresults_gpu->nitems + kresults_cpu->nitems);

	/* pds is still valid, but gpusort is no longer referenced */
	pgstrom_untrack_object(&gpusort->msg.sobj);
	pgstrom_put_message(&gpusort->msg);

	/* OK, ritems/litems are filled, let's kick the CPU merge sort */
	gss->sorted_chunk = NULL;
	return cpusort;
}

static void
gpusort_check_gpu_tasks(GpuSortState *gss)
{
	pgstrom_message	   *msg;
	pgstrom_gpusort	   *gpusort;
	pgstrom_cpusort	   *cpusort;
	kern_resultbuf	   *kresults;

	/*
	 * Check status of GPU co-routine
	 */
	while ((msg = pgstrom_try_dequeue_message(gss->mqueue)) != NULL)
	{
		Assert(gss->num_gpu_running > 0);
		gss->num_gpu_running--;

		if (msg->pfm.enabled)
			pgstrom_perfmon_add(&gss->pfm, &msg->pfm);
		Assert(StromTagIs(msg, GpuSort));
		gpusort = (pgstrom_gpusort *) msg;
		kresults = KERN_GPUSORT_RESULTBUF(&gpusort->kern);

		if (msg->errcode != StromError_Success)
		{
			if (msg->errcode == CL_BUILD_PROGRAM_FAILURE)
			{
				const char *buildlog
					= pgstrom_get_devprog_errmsg(gpusort->dprog_key);

				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("PG-Strom: OpenCL execution error (%s)\n%s",
								pgstrom_strerror(msg->errcode),
								gss->kern_source),
						 errdetail("%s", buildlog)));
			}
			else
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("PG-Strom: OpenCL execution error (%s)",
								pgstrom_strerror(msg->errcode))));
			}
		}
		/* add performance information */
		if (msg->pfm.enabled)
			pgstrom_perfmon_add(&gss->pfm, &msg->pfm);
		/* try to form it to cpusort, and merge */
		cpusort = gpusort_merge_gpu_chunks(gss, gpusort);
		if (cpusort)
			dlist_push_tail(&gss->pending_cpu_chunks, &cpusort->chain);
	}
}

static void
gpusort_check_cpu_tasks(GpuSortState *gss)
{
	pgstrom_cpusort	   *cpusort;
	dlist_mutable_iter	iter;

	/*
	 * Check status of CPU-sorting background worker
	 */
	dlist_foreach_modify(iter, &gss->running_cpu_chunks)
	{
		BgwHandleStatus		status;
		kern_resultbuf	   *kresults;
		pid_t				bgw_pid;

		cpusort = dlist_container(pgstrom_cpusort, chain, iter.cur);
		status = GetBackgroundWorkerPid(cpusort->bgw_handle, &bgw_pid);

		/* emergency. kills all the backend and raise an error */
		if (status == BGWH_POSTMASTER_DIED)
		{
			/* TODO: terminate any other running workers*/
			elog(ERROR, "Bug? Postmaster or BgWorker got dead, bogus status");
		}
		else if (status == BGWH_STOPPED)
		{
			dlist_delete(&cpusort->chain);
			gss->num_cpu_running--;

			/*
			 * check the result status in the dynamic worker
			 *
			 * note: we intend dynamic worker returns error code and
			 * message on the kern_resultbuf if error happen.
			 * we may need to make a common format later.
			 */
			kresults = dsm_segment_address(cpusort->oitems_dsm);
			if (kresults->errcode != ERRCODE_SUCCESSFUL_COMPLETION)
				ereport(ERROR,
						(errcode(kresults->errcode),
						 errmsg("GpuSort worker: %s",
								(char *)kresults->results)));

			cpusort = gpusort_merge_cpu_chunks(gss, cpusort);
			if (cpusort)
				dlist_push_tail(&gss->pending_cpu_chunks, &cpusort->chain);
		}
	}
}

static void
gpusort_kick_pending_tasks(GpuSortState *gss)
{
	BackgroundWorker	worker;
	pgstrom_cpusort	   *cpusort;
	dlist_node		   *dnode;

	while (!dlist_is_empty(&gss->pending_cpu_chunks) &&
		   gss->num_cpu_running < gpusort_max_workers)
	{
		dnode = dlist_pop_head_node(&gss->pending_cpu_chunks);
		cpusort = dlist_container(pgstrom_cpusort, chain, dnode);

		/*
		 * Set up dynamic background worker
		 */
		memset(&worker, 0, sizeof(BackgroundWorker));
		snprintf(worker.bgw_name, sizeof(worker.bgw_name),
				 "GpuSort worker-%u", gss->cpusort_seqno++);
		worker.bgw_flags = BGWORKER_SHMEM_ACCESS |
			BGWORKER_BACKEND_DATABASE_CONNECTION;
		worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
		worker.bgw_restart_time = BGW_NEVER_RESTART;
		worker.bgw_main = gpusort_entrypoint_cpusort;
		/* XXX - form/deform operation on DSM will be needed on windows */
		worker.bgw_main_arg = PointerGetDatum(cpusort);

		if (!RegisterDynamicBackgroundWorker(&worker, &cpusort->bgw_handle))
			dlist_push_head(&gss->pending_cpu_chunks, &cpusort->chain);
		else
		{
			dlist_push_tail(&gss->running_cpu_chunks, &cpusort->chain);
			gss->num_cpu_running++;
		}
	}
}

static void
gpusort_exec_sort(GpuSortState *gss)
{
	pgstrom_gpusort	   *gpusort;
	pgstrom_cpusort	   *cpusort;
	int					rc;

	while (!gss->scan_done ||
		   gss->num_gpu_running > 0 ||
		   gss->num_cpu_running > 0 ||
		   !dlist_is_empty(&gss->pending_cpu_chunks))
	{
		CHECK_FOR_INTERRUPTS();

		/* if not end of relation, make one more chunk to be sorted */
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
		/*
		 * Check status of the asynchronous tasks. If finished, task shall
		 * be chained to gss->sorted_chunk or moved to pending_chunks if
		 * it has merged with a buddy.
		 * Once all the chunks are moved to the pending_chunks, then we will
		 * kick background worker jobs unless it does not touch upper limit.
		 */
		gpusort_check_gpu_tasks(gss);
		gpusort_check_cpu_tasks(gss);
		gpusort_kick_pending_tasks(gss);

		/*
		 * Unless scan of underlying relation is not finished, we try to
		 * make progress the scan (that makes gpusort chunks on the file-
		 * mapped data-store).
		 * Once scan is done, we will synchronize the completion of
		 * concurrent tasks. Please note that we may need to wait for
		 * a slot of background worker process occupied by another backend
		 * is released, even if num_*_running is zero. In this case, we
		 * set relatively short timeout.
		 */
		if (gss->scan_done)
		{
			long	timeout;

			if (gss->num_gpu_running > 0 || gss->num_cpu_running > 0)
				timeout = 20 * 1000L;	/* 20sec */
			else
				timeout =       400L;	/* 0.4sec */

			rc = WaitLatch(&MyProc->procLatch,
						   WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
						   timeout);
			ResetLatch(&MyProc->procLatch);

			/* emergency bailout if postmaster has died */
            if (rc & WL_POSTMASTER_DEATH)
                elog(ERROR, "failed on WaitLatch due to Postmaster die");
		}
	}

	/*
	 * Once we got the sorting completed, just one chunk should be attached
	 * on the gss->sorted_chunk. If NULL, it means outer relation has no
	 * rows anywhere.
	 */
	cpusort = gss->sorted_chunk;
	if (!cpusort)
		gss->sorted_result = NULL;
	else
	{
		gss->sorted_result = cpusort->oitems_dsm;
		pfree(cpusort);
		gss->sorted_chunk = NULL;
	}
	gss->sorted_index = 0;
}


static TupleTableSlot *
gpusort_exec(CustomScanState *node)
{
	GpuSortState	   *gss = (GpuSortState *) node;
	TupleTableSlot	   *slot = node->ss.ps.ps_ResultTupleSlot;
	kern_resultbuf	   *kresults;
	pgstrom_data_store *pds;

	if (!gss->sort_done)
		gpusort_exec_sort(gss);

	/* Does outer relation has any rows to read? */
	if (!gss->sorted_result)
		return NULL;

	kresults = dsm_segment_address(gss->sorted_result);
	if (gss->sorted_index < kresults->nitems)
	{
		cl_long		index = 2 * gss->sorted_index;
		cl_int		chunk_id = kresults->results[index];
		cl_int		item_id = kresults->results[index + 1];

		if (chunk_id >= gss->num_chunks || !gss->pds_chunks[chunk_id])
			elog(ERROR, "Bug? data-store of GpuSort missing (chunk-id: %d)",
				 chunk_id);
		pds = gss->pds_chunks[chunk_id];

		if (pgstrom_fetch_data_store(slot, pds, item_id, &gss->tuple_buf))
		{
			gss->sorted_index++;
			return slot;
		}
	}
	return NULL;
}

static void
gpusort_end(CustomScanState *node)
{
	GpuSortState   *gss = (GpuSortState *) node;
	int				i;

	Assert(dlist_is_empty(&gss->pending_cpu_chunks));
	Assert(dlist_is_empty(&gss->running_cpu_chunks));
	Assert(!gss->sorted_chunk);

	if (gss->sorted_result)
		dsm_detach(gss->sorted_result);

	for (i=0; i < gss->num_chunks; i++)
	{
		pgstrom_data_store *pds = gss->pds_chunks[i];
		pgstrom_untrack_object(&pds->sobj);
		pgstrom_put_data_store(pds);
	}

	if (gss->dprog_key)
	{
		pgstrom_untrack_object((StromObject *)gss->dprog_key);
        pgstrom_put_devprog_key(gss->dprog_key);
	}

	if (gss->mqueue)
	{
		pgstrom_untrack_object(&gss->mqueue->sobj);
		pgstrom_close_queue(gss->mqueue);
	}
	/* Clean up subtree */
    ExecEndNode(outerPlanState(node));
}

static void
gpusort_rescan(CustomScanState *node)
{
	GpuSortState   *gss = (GpuSortState *) node;
	int				i;

	if (!gss->sort_done)
		return;

	/* must drop pointer to sort result tuple */
	ExecClearTuple(node->ss.ps.ps_ResultTupleSlot);

	/*
     * If subnode is to be rescanned then we forget previous sort results; we
     * have to re-read the subplan and re-sort.  Also must re-sort if the
     * bounded-sort parameters changed or we didn't select randomAccess.
     *
     * Otherwise we can just rewind and rescan the sorted output.
     */
	if (outerPlanState(gss)->chgParam != NULL)
	{
		gss->sort_done = false;
		/*
		 * Release the previous read
		 */
		Assert(dlist_is_empty(&gss->pending_cpu_chunks));
		Assert(dlist_is_empty(&gss->running_cpu_chunks));
		Assert(!gss->sorted_chunk);

		if (gss->sorted_result)
			dsm_detach(gss->sorted_result);

		for (i=0; i < gss->num_chunks; i++)
		{
			pgstrom_data_store *pds = gss->pds_chunks[i];
			pgstrom_untrack_object(&pds->sobj);
			pgstrom_put_data_store(pds);
		}

		/*
		 * if chgParam of subnode is not null then plan will be re-scanned
		 * by first ExecProcNode, so we don't need to call ExecReScan() here.
		 */
    }
    else
	{
		/* otherwise, just rewind the pointer */
		gss->sorted_index = 0;
	}
}

static void
gpusort_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
	GpuSortState   *gss = (GpuSortState *) node;
	CustomScan	   *cscan = (CustomScan *) node->ss.ps.plan;
	GpuSortInfo	   *gs_info = deform_gpusort_info(cscan);
	List		   *context;
	List		   *sort_keys = NIL;
	bool			use_prefix;
	int				i;

	/* shows sorting keys */
	context = deparse_context_for_planstate((Node *) node,
											ancestors,
											es->rtable,
											es->rtable_names);
	use_prefix = (list_length(es->rtable) > 1 || es->verbose);

	for (i=0; i < gs_info->numCols; i++)
	{
		AttrNumber		resno = gs_info->sortColIdx[i];
		TargetEntry	   *tle;
		char		   *exprstr;

		tle = get_tle_by_resno(cscan->scan.plan.targetlist, resno);
		if (!tle)
			elog(ERROR, "no tlist entry for key %d", resno);
		exprstr = deparse_expression((Node *) tle->expr, context,
                                     use_prefix, true);
		sort_keys = lappend(sort_keys, exprstr);
	}
	if (sort_keys != NIL)
		ExplainPropertyList("Sort Key", sort_keys, es);

	/* shows resource comsumption, if executed */
	if (es->analyze && gss->sort_done)
	{
		const char *sort_method;
		const char *sort_storage;
		char		sort_resource[128];
		Size		total_consumption;

		if (gss->num_chunks > 1)
			sort_method = "GPU:bitonic + CPU:merge";
		else
			sort_method = "GPU:bitonic";

		total_consumption = (Size)gss->num_chunks * gss->chunk_size;
		total_consumption += dsm_segment_map_length(gss->sorted_result);
		if (total_consumption >= (Size)(1UL << 43))
			snprintf(sort_resource, sizeof(sort_resource), "%.2fTb",
					 (double)total_consumption / (double)(1UL << 40));
		else if (total_consumption >= (Size)(1UL << 33))
			snprintf(sort_resource, sizeof(sort_resource), "%.2fGb",
					 (double)total_consumption / (double)(1UL << 30));
		else if (total_consumption >= (Size)(1UL << 23))
			snprintf(sort_resource, sizeof(sort_resource), "%zuMb",
					 total_consumption >> 20);
		else
			snprintf(sort_resource, sizeof(sort_resource), "%zuKb",
					 total_consumption >> 10);

		/* full on-memory storage might be an option according to the size */
		sort_storage = "Disk";

		if (es->format == EXPLAIN_FORMAT_TEXT)
		{
			appendStringInfoSpaces(es->str, es->indent * 2);
			appendStringInfo(es->str, "Sort Method: %s %s used: %s\n",
							 sort_method,
							 sort_storage,
							 sort_resource);
		}
		else
		{
			ExplainPropertyText("Sort Method", sort_method, es);
			ExplainPropertyLong("Sort Space Used", total_consumption, es);
			ExplainPropertyText("Sort Space Type", sort_storage, es);
		}
	}
	show_device_kernel(gss->dprog_key, es);
	if (es->analyze && gss->pfm.enabled)
		pgstrom_perfmon_explain(&gss->pfm, es);
}

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
	/* pg_strom.gpusort_max_workers */
	DefineCustomIntVariable("pg_strom.max_workers",
							"Maximum number of sorting workers for GpuSort",
							NULL,
							&gpusort_max_workers,
							Max(1, max_worker_processes / 2),
							1,
							max_worker_processes / 2,
							PGC_USERSET,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	/* initialize the plan method table */
	memset(&gpusort_scan_methods, 0, sizeof(CustomScanMethods));
	gpusort_scan_methods.CustomName			= "GpuSort";
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

/* ================================================================
 *
 * Routines for CPU sorting
 *
 * ================================================================
 */
static void
gpusort_exec_cpusort(pgstrom_cpusort *cpusort)
{
	SortSupportData	   *sort_keys;
	pgstrom_data_store *pds;
	kern_resultbuf	   *ritems = dsm_segment_address(cpusort->ritems_dsm);
	kern_resultbuf	   *litems = dsm_segment_address(cpusort->litems_dsm);
	kern_resultbuf	   *oitems = dsm_segment_address(cpusort->oitems_dsm);
	TupleTableSlot	   *rslot = MakeSingleTupleTableSlot(cpusort->tupdesc);
	TupleTableSlot	   *lslot = MakeSingleTupleTableSlot(cpusort->tupdesc);
	long				rindex = 0;
	long				lindex = 0;
	long				oindex = 0;
	int					rchunk_id;
	int					ritem_id;
	int					lchunk_id;
	int					litem_id;
	int					i;

	/*
	 * Set up sorting keys
	 */
	sort_keys = palloc0(sizeof(SortSupportData) * cpusort->numCols);
	for (i=0; i < cpusort->numCols; i++)
	{
		SortSupport	ssup = sort_keys + i;

		ssup->ssup_cxt = CurrentMemoryContext;
		ssup->ssup_collation = cpusort->collations[i];
		ssup->ssup_nulls_first = cpusort->nullsFirst[i];
		ssup->ssup_attno = cpusort->sortColIdx[i];
		PrepareSortSupportFromOrderingOp(cpusort->sortOperators[i], ssup);
	}

	/*
	 * Begin merge sorting
	 */
	while (lindex < litems->nitems && rindex < ritems->nitems)
	{
		HeapTupleData dummy;
		int		comp = 0;

		if (TupIsNull(lslot))
		{
			lchunk_id = litems->results[2 * lindex];
			litem_id = litems->results[2 * lindex + 1];
			Assert(lchunk_id < cpusort->num_chunks);
			pds = cpusort->pds_chunks[lchunk_id];
			if (!pgstrom_fetch_data_store(lslot, pds, litem_id, &dummy))
				elog(ERROR, "failed to fetch sorting tuple in (%d,%d)",
					 lchunk_id, litem_id);
			lindex++;
		}

		if (TupIsNull(rslot))
		{
			rchunk_id = ritems->results[2 * rindex];
			ritem_id = ritems->results[2 * rindex + 1];
			Assert(rchunk_id < cpusort->num_chunks);
			pds = cpusort->pds_chunks[rchunk_id];
			if (!pgstrom_fetch_data_store(rslot, pds, ritem_id, &dummy))
				elog(ERROR, "failed to fetch sorting tuple in (%d,%d)",
					 rchunk_id, ritem_id);
			rindex++;
		}

		for (i=0; i < cpusort->numCols; i++)
		{
			SortSupport ssup = sort_keys + i;
			AttrNumber	attno = ssup->ssup_attno;

			comp = ApplySortComparator(lslot->tts_values[attno],
									   lslot->tts_isnull[attno],
									   rslot->tts_values[attno],
									   rslot->tts_isnull[attno],
									   ssup);
			if (comp != 0)
				break;
		}
		if (comp <= 0)
		{
			oitems->results[2 * oindex] = lchunk_id;
			oitems->results[2 * oindex + 1] = litem_id;
			oindex++;
			ExecClearTuple(lslot);
		}

		if (comp >= 0)
		{
			oitems->results[2 * oindex] = rchunk_id;
			oitems->results[2 * oindex + 1] = ritem_id;
			oindex++;
			ExecClearTuple(rslot);
		}
	}
	while (lindex < litems->nitems)
	{
		oitems->results[2 * oindex] = litems->results[2 * lindex];
		oitems->results[2 * oindex + 1] = litems->results[2 * lindex + 1];
		oindex++;
		lindex++;
	}

	while (rindex < ritems->nitems)
	{
		oitems->results[2 * oindex] = ritems->results[2 * rindex];
		oitems->results[2 * oindex + 1] = ritems->results[2 * rindex + 1];
		oindex++;
		rindex++;
	}
	Assert(oindex == litems->nitems + ritems->nitems);
	oitems->nitems = oindex;
}

static void
gpusort_entrypoint_cpusort(Datum main_arg)
{
	pgstrom_cpusort *cpusort = (pgstrom_cpusort *) DatumGetPointer(main_arg);

	/* We're now ready to receive signals */
	BackgroundWorkerUnblockSignals();

	/* Connect to our database */
	BackgroundWorkerInitializeConnection(cpusort->dbname, NULL);

	/*
	 * XXX - Eventually, we should use parallel-context to share
	 * the transaction snapshot, initialize misc stuff and so on.
	 * But just at this moment, we create a new transaction state
	 * to simplifies the implementation.
	 */
	StartTransactionCommand();
	PushActiveSnapshot(GetTransactionSnapshot());

	CurrentMemoryContext = AllocSetContextCreate(TopMemoryContext,
												 "cpusort",
												 ALLOCSET_DEFAULT_MINSIZE,
												 ALLOCSET_DEFAULT_INITSIZE,
												 ALLOCSET_DEFAULT_MAXSIZE);
	/* handle cpu sorting */
	gpusort_exec_cpusort(cpusort);

	/* we should have no side-effect */
	CommitTransactionCommand();
}

/* ================================================================
 *
 * Routines for GPU sorting
 *
 * ================================================================
 */
typedef struct
{
	pgstrom_message *msg;
	kern_data_store	*kds;
	int				kds_fdesc;
	cl_command_queue kcmdq;
	cl_program		program;
	cl_int			dindex;
	cl_mem			m_gpusort;
	cl_mem			m_kds;
	cl_kernel		kern_prep;
	cl_kernel	   *kern_sort;
	cl_uint			kern_sort_nums;
	cl_uint			ev_kern_prep;
	cl_uint			ev_dma_recv;	/* event index of DMA recv */
	cl_uint			ev_index;
	cl_event		events[FLEXIBLE_ARRAY_MEMBER];
} clstate_gpusort;

static void
clserv_respond_gpusort(cl_event event, cl_int ev_status, void *private)
{
	clstate_gpusort	   *clgss = (clstate_gpusort *) private;
	pgstrom_gpusort	   *gpusort = (pgstrom_gpusort *)clgss->msg;
	kern_resultbuf	   *kresult = KERN_GPUSORT_RESULTBUF(&gpusort->kern);
	cl_int				i, rc;

	if (ev_status == CL_COMPLETE)
		gpusort->msg.errcode = kresult->errcode;
	else
	{
		clserv_log("unexpected CL_EVENT_COMMAND_EXECUTION_STATUS: %d",
				   ev_status);
		gpusort->msg.errcode = StromError_OpenCLInternal;
	}

	/* collect performance statistics */
	if (gpusort->msg.pfm.enabled)
	{
		cl_ulong	tv_start;
		cl_ulong	tv_end;
		cl_ulong	temp;

		/*
		 * Time of all the DMA send
		 */
		tv_start = ~0UL;
		tv_end = 0;
		for (i=0; i < clgss->ev_kern_prep; i++)
		{
			rc = clGetEventProfilingInfo(clgss->events[i],
										 CL_PROFILING_COMMAND_START,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			tv_start = Min(tv_start, temp);

			rc = clGetEventProfilingInfo(clgss->events[i],
										 CL_PROFILING_COMMAND_END,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			tv_end = Max(tv_end, temp);
		}
		gpusort->msg.pfm.time_dma_send += (tv_end - tv_start) / 1000;

		/*
		 * Prep kernel execution time
		 */
		i = clgss->ev_kern_prep;
		rc = clGetEventProfilingInfo(clgss->events[i],
									 CL_PROFILING_COMMAND_START,
									 sizeof(cl_ulong),
									 &tv_start,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;
		rc = clGetEventProfilingInfo(clgss->events[i],
									 CL_PROFILING_COMMAND_END,
									 sizeof(cl_ulong),
									 &tv_end,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;
		gpusort->msg.pfm.time_kern_prep += (tv_end - tv_start) / 1000;

		/*
		 * Sort kernel execution time
		 */
		tv_start = ~0UL;
		tv_end = 0;
		for (i=clgss->ev_kern_prep + 1; i < clgss->ev_dma_recv; i++)
		{
			rc = clGetEventProfilingInfo(clgss->events[i],
										 CL_PROFILING_COMMAND_START,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			tv_start = Min(tv_start, temp);

			rc = clGetEventProfilingInfo(clgss->events[i],
										 CL_PROFILING_COMMAND_END,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			tv_end = Max(tv_end, temp);
		}
		gpusort->msg.pfm.time_kern_sort += (tv_end - tv_start) / 1000;

		/*
		 * DMA recv time
		 */
		tv_start = ~0UL;
        tv_end = 0;
        for (i=clgss->ev_dma_recv; i < clgss->ev_index; i++)
		{
			rc = clGetEventProfilingInfo(clgss->events[clgss->ev_index - i],
										 CL_PROFILING_COMMAND_START,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			tv_start = Min(tv_start, temp);

			rc = clGetEventProfilingInfo(clgss->events[clgss->ev_index - i],
										 CL_PROFILING_COMMAND_END,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			tv_end = Max(tv_end, temp);
		}
		gpusort->msg.pfm.time_dma_recv += (tv_end - tv_start) / 1000;

	skip_perfmon:
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clGetEventProfilingInfo (%s)",
					   opencl_strerror(rc));
			gpusort->msg.pfm.enabled = false;	/* turn off profiling */
		}
	}

	/*
	 * release opencl resources
	 */
	while (clgss->ev_index > 0)
		clReleaseEvent(clgss->events[--clgss->ev_index]);
	if (clgss->m_gpusort)
		clReleaseMemObject(clgss->m_gpusort);
	if (clgss->m_kds)
		clReleaseMemObject(clgss->m_kds);
	if (clgss->kern_prep)
		clReleaseKernel(clgss->kern_prep);
	for (i=0; i < clgss->kern_sort_nums; i++)
		clReleaseKernel(clgss->kern_sort[i]);
	if (clgss->program && clgss->program != BAD_OPENCL_PROGRAM)
		clReleaseProgram(clgss->program);
	if (clgss->kds)
	{
		munmap(clgss->kds, clgss->kds->length);
		close(clgss->kds_fdesc);
	}
	if (clgss->kern_sort)
		free(clgss->kern_sort);
	free(clgss);

	/* reply the result to backend side */
	pgstrom_reply_message(&gpusort->msg);
}

static cl_int
compute_bitonic_workgroup_size(clstate_gpusort *clgss, size_t nhalf,
							   size_t *p_gwork_sz, size_t *p_lwork_sz)
{
	static struct {
		const char *kern_name;
		size_t		kern_lmem;
	} kern_calls[] = {
		{ "gpusort_bitonic_local", 2 * sizeof(cl_uint) },
		{ "gpusort_bitonic_merge",     sizeof(cl_uint) },
	};
	size_t		least_sz = nhalf;
	size_t		lwork_sz;
	size_t		gwork_sz;
	cl_kernel	kernel;
	cl_int		i, rc;

	for (i=0; i < lengthof(kern_calls); i++)
	{
		kernel = clCreateKernel(clgss->program,
								kern_calls[i].kern_name,
								&rc);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
			return rc;
		}

		if(!clserv_compute_workgroup_size(&gwork_sz,
										  &lwork_sz,
										  kernel,
										  clgss->dindex,
										  true,
										  nhalf,
										  kern_calls[i].kern_lmem))
		{
			clserv_log("failed on clserv_compute_workgroup_size");
			clReleaseKernel(kernel);
			return StromError_OpenCLInternal;
		}
		clReleaseKernel(kernel);

		least_sz = Min(least_sz, lwork_sz);
	}
	/*
	 * NOTE: Local workgroup size is the largest 2^N value less than
	 * or equal to the least one of expected kernels.
	 */
	lwork_sz = 1UL << (get_next_log2(least_sz + 1) - 1);
	gwork_sz = ((nhalf + lwork_sz - 1) / lwork_sz) * lwork_sz;

	*p_lwork_sz = lwork_sz;
	*p_gwork_sz = gwork_sz;

	return CL_SUCCESS;
}

/*
 * clserv_launch_gpusort_preparation
 *
 * launcher of:
 *   __kernel void
 *   gpusort_preparation(__global kern_gpusort *kgsort)
 */
static cl_int
clserv_launch_gpusort_preparation(clstate_gpusort *clgss, size_t nitems)
{
	pgstrom_gpusort *gpusort = (pgstrom_gpusort *) clgss->msg;
	size_t		gwork_sz;
	size_t		lwork_sz;
	cl_int		rc;

	clgss->kern_prep = clCreateKernel(clgss->program,
									  "gpusort_preparation",
									  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		return rc;
	}

	if (!clserv_compute_workgroup_size(&gwork_sz,
									   &lwork_sz,
									   clgss->kern_prep,
									   clgss->dindex,
									   true,
									   nitems,
									   sizeof(cl_uint)))
	{
		clserv_log("failed to compute optimal gwork_sz/lwork_sz");
		return StromError_OpenCLInternal;
	}

	rc = clSetKernelArg(clgss->kern_prep,
						0,		/* __global kern_gpusort *kgsort */
						sizeof(cl_mem),
						&clgss->m_gpusort);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgss->kern_prep,
						1,		/* cl_int chunk_id */
						sizeof(cl_int),
						&gpusort->chunk_id);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgss->kern_prep,
						2,		/* KERN_DYNAMIC_LOCAL_WORKMEM_ARG */
						sizeof(cl_int) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clEnqueueNDRangeKernel(clgss->kcmdq,
								clgss->kern_prep,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								clgss->ev_index,
								&clgss->events[0],
								&clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		return rc;
	}
	clgss->ev_kern_prep = clgss->ev_index++;
	clgss->msg->pfm.num_kern_prep++;

	return CL_SUCCESS;
}

/*
 * clserv_launch_bitonic_local
 *
 * launcher of:
 *   __kernel void
 *   gpusort_bitonic_local(__global kern_gpusort *kgsort,
 *                         __global kern_data_store *kds,
 *                         __global kern_data_store *ktoast,
 *                         KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
 */
static cl_int
clserv_launch_bitonic_local(clstate_gpusort *clgss,
							size_t gwork_sz, size_t lwork_sz)
{
	cl_kernel	kernel;
	cl_int		rc;

	kernel = clCreateKernel(clgss->program,
							"gpusort_bitonic_local",
						    &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		return rc;
	}
	clgss->kern_sort[clgss->kern_sort_nums++] = kernel;

	rc = clSetKernelArg(kernel,
						0,		/* __global kern_gpusort *kgsort */
						sizeof(cl_mem),
						&clgss->m_gpusort);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						1,		/* __global kern_data_store *kds */
						sizeof(cl_mem),
						&clgss->m_kds);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						2,		/* __global kern_data_store *ktoast */
						sizeof(cl_mem),
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						3,		/* KERN_DYNAMIC_LOCAL_WORKMEM_ARG */
						2 * sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clEnqueueNDRangeKernel(clgss->kcmdq,
								kernel,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								1,
								&clgss->events[clgss->ev_index - 1],
								&clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		return rc;
	}
    clgss->ev_index++;
    clgss->msg->pfm.num_kern_sort++;

	return CL_SUCCESS;
}

/*
 * clserv_launch_bitonic_step
 *
 * launcher of:
 *   __kernel void
 *   gpusort_bitonic_step(__global kern_gpusort *kgsort,
 *                        cl_int bitonic_unitsz,
 *                        __global kern_data_store *kds,
 *                        __global kern_data_store *ktoast,
 *                        KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
 */
static cl_int
clserv_launch_bitonic_step(clstate_gpusort *clgss, bool reversing,
						   size_t unitsz, size_t work_sz)
{
	cl_kernel	kernel;
	cl_int		bitonic_unitsz;
	cl_int		rc;
	size_t		gwork_sz;
	size_t		lwork_sz;

	kernel = clCreateKernel(clgss->program,
							"gpusort_bitonic_step",
							&rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		return rc;
	}
	clgss->kern_sort[clgss->kern_sort_nums++] = kernel;

	if (!clserv_compute_workgroup_size(&gwork_sz, &lwork_sz,
									   kernel,
									   clgss->dindex,
									   false,
									   work_sz,
									   sizeof(int)))
	{
		clserv_log("failed to compute optimal gwork_sz/lwork_sz");
		return StromError_OpenCLInternal;
	}

	rc = clSetKernelArg(kernel,
						0,		/* __global kern_gpusort *kgsort */
						sizeof(cl_mem),
						&clgss->m_gpusort);
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
						1,		/* cl_int bitonic_unitsz */
						sizeof(cl_int),
						&bitonic_unitsz);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						2,		/* __global kern_data_store *kds */
						sizeof(cl_mem),
						&clgss->m_kds);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						3,		/* __global kern_data_store *ktoast */
						sizeof(cl_mem),
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						4,		/* KERN_DYNAMIC_LOCAL_WORKMEM_ARG */
						sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clEnqueueNDRangeKernel(clgss->kcmdq,
								kernel,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								1,
								&clgss->events[clgss->ev_index - 1],
								&clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		return rc;
	}
	clgss->ev_index++;
	clgss->msg->pfm.num_kern_sort++;

	return CL_SUCCESS;
}

/*
 * clserv_launch_bitonic_merge
 *
 * launcher of:
 *   __kernel void
 *   gpusort_bitonic_merge(__global kern_gpusort *kgpusort,
 *                         __global kern_data_store *kds,
 *                         __global kern_data_store *ktoast,
 *                         KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
 */
static cl_int
clserv_launch_bitonic_merge(clstate_gpusort *clgss,
                            size_t gwork_sz, size_t lwork_sz)
{
	cl_kernel	kernel;
	cl_int		rc;

	kernel = clCreateKernel(clgss->program,
							"gpusort_bitonic_merge",
							&rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		return rc;
	}
	clgss->kern_sort[clgss->kern_sort_nums++] = kernel;

	rc = clSetKernelArg(kernel,
						0,		/* __global kern_gpusort *kgsort */
						sizeof(cl_mem),
						&clgss->m_gpusort);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						1,		/* __global kern_data_store *kds */
						sizeof(cl_mem),
						&clgss->m_kds);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						2,		/* __global kern_data_store *ktoast */
						sizeof(cl_mem),
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(kernel,
						3,		/* KERN_DYNAMIC_LOCAL_WORKMEM_ARG */
						2 * sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clEnqueueNDRangeKernel(clgss->kcmdq,
								kernel,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								1,
								&clgss->events[clgss->ev_index - 1],
								&clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		return rc;
	}
	clgss->ev_index++;
	clgss->msg->pfm.num_kern_sort++;

	return CL_SUCCESS;
}

static void
clserv_process_gpusort(pgstrom_message *msg)
{
	pgstrom_gpusort	   *gpusort = (pgstrom_gpusort *) msg;
	pgstrom_data_store *pds = gpusort->pds;
	clstate_gpusort	   *clgss;
	cl_uint				nitems;
	Size				length;
	Size				offset;
	size_t				nhalf;
	size_t				gwork_sz;
	size_t				lwork_sz;
	size_t				nsteps;
	size_t				launches;
	size_t				i, j;
	cl_int				rc;

	Assert(StromTagIs(gpusort, GpuSort));

	/*
	 * state object of gpusort
	 */
	clgss = calloc(1, offsetof(clstate_gpusort, events[1000]));
	if (!clgss)
	{
		rc = CL_OUT_OF_HOST_MEMORY;
		goto error;
	}
	clgss->msg = &gpusort->msg;

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
	clgss->program = clserv_lookup_device_program(gpusort->dprog_key,
												  &gpusort->msg);
	if (!clgss->program)
	{
		free(clgss);
		return;		/* message is in waitq, being retried later */
	}
	if (clgss->program == BAD_OPENCL_PROGRAM)
	{
		rc = CL_BUILD_PROGRAM_FAILURE;
		goto error;
	}

	/*
	 * Map kern_data_store based on KDS_
	 */
	clgss->kds = pgstrom_map_data_store_row_fmap(pds, &clgss->kds_fdesc);
	if (!clgss->kds)
		goto error;
	Assert(pds->kds_length == clgss->kds->length);
	nitems = clgss->kds->nitems;

	/*
	 * choose a device to run
	 */
	clgss->dindex = pgstrom_opencl_device_schedule(&gpusort->msg);
	clgss->kcmdq = opencl_cmdq[clgss->dindex];

	/*
	 * construction of kernel buffer objects
	 *
	 * m_gpusort - control data of gpusort
	 * m_kds     - data store of records to be sorted
	 */
	length = KERN_GPUSORT_LENGTH(&gpusort->kern);
	clgss->m_gpusort = clCreateBuffer(opencl_context,
									  CL_MEM_READ_WRITE,
									  length,
									  NULL,
									  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error;
	}

	length = KERN_DATA_STORE_LENGTH(clgss->kds);
	clgss->m_kds = clCreateBuffer(opencl_context,
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
	 * OK, Enqueue DMA send requests prior to kernel execution
	 */
	/* __global kern_gpusort *kgpusort */
	offset = KERN_GPUSORT_DMASEND_OFFSET(&gpusort->kern);
	length = KERN_GPUSORT_DMASEND_LENGTH(&gpusort->kern);
	rc = clEnqueueWriteBuffer(clgss->kcmdq,
							  clgss->m_gpusort,
							  CL_FALSE,
							  offset,
							  length,
							  &gpusort->kern,
							  0,
							  NULL,
							  &clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error;
	}
	clgss->ev_index++;
	gpusort->msg.pfm.bytes_dma_send += length;
	gpusort->msg.pfm.num_dma_send++;

	/* __global kern_data_store *kds */
	length = KERN_DATA_STORE_LENGTH(clgss->kds);
	rc = clEnqueueWriteBuffer(clgss->kcmdq,
							  clgss->m_kds,
							  CL_FALSE,
							  0,
							  length,
							  clgss->kds,
							  0,
							  NULL,
							  &clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error;
	}
	clgss->ev_index++;
    gpusort->msg.pfm.bytes_dma_send += length;
    gpusort->msg.pfm.num_dma_send++;

	/* kick, gpusort_preparation kernel function */
	rc = clserv_launch_gpusort_preparation(clgss, nitems);
	if (rc != CL_SUCCESS)
		goto error;

	/* kick, a series of gpusort_bitonic_*() functions */
	nhalf = (nitems + 1) / 2;

	rc = compute_bitonic_workgroup_size(clgss, nhalf, &gwork_sz, &lwork_sz);
	if (rc != CL_SUCCESS)
		goto error;

	nsteps = get_next_log2(nhalf / lwork_sz) + 1;
	launches = (nsteps + 1) * nsteps / 2 + nsteps + 1;

	clgss->kern_sort = calloc(launches, sizeof(cl_kernel));
	if (!clgss->kern_sort)
		goto error;

	/* Sorting in each work groups */
	rc = clserv_launch_bitonic_local(clgss, gwork_sz, lwork_sz);
	if (rc != CL_SUCCESS)
		goto error;
	/* Sorting inter workgroups */
	for (i = 2 * lwork_sz; i < gwork_sz; i *= 2)
	{
		for (j = i; j > lwork_sz; j /= 2)
		{
			cl_uint		unitsz = 2 * j;
			bool		reversing = (j == i) ? true : false;
			size_t		work_sz;

			work_sz = (((nitems + unitsz - 1) / unitsz) * unitsz / 2);
			rc = clserv_launch_bitonic_step(clgss, reversing, unitsz, work_sz);
			if (rc != CL_SUCCESS)
				goto error;
		}
		rc = clserv_launch_bitonic_merge(clgss, gwork_sz, lwork_sz);
		if (rc != CL_SUCCESS)
			goto error;
	}

	/*
	 * Write back result buffer to the host memory
	 */
	offset = KERN_GPUSORT_DMARECV_OFFSET(&gpusort->kern);
	length = KERN_GPUSORT_DMARECV_LENGTH(&gpusort->kern);
	rc =  clEnqueueReadBuffer(clgss->kcmdq,
							  clgss->m_gpusort,
							  CL_FALSE,
							  offset,
							  length,
							  (char *)(&gpusort->kern) + offset,
							  1,
							  &clgss->events[clgss->ev_index - 1],
							  &clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueReadBuffer: %s",
				   opencl_strerror(rc));
		goto error;
	}
	clgss->ev_dma_recv = clgss->ev_index++;
	gpusort->msg.pfm.bytes_dma_recv += length;
    gpusort->msg.pfm.num_dma_recv++;

	/*
	 * Last, registers a callback to handle post gpusort process
	 */
	rc = clSetEventCallback(clgss->events[clgss->ev_index - 1],
							CL_COMPLETE,
							clserv_respond_gpusort,
							clgss);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetEventCallback: %s", opencl_strerror(rc));
		goto error;
	}
	return;

error:
	if (clgss)
	{
		if (clgss->ev_index > 0)
		{
			clWaitForEvents(clgss->ev_index, clgss->events);
			while (clgss->ev_index > 0)
				clReleaseEvent(clgss->events[--clgss->ev_index]);
		}
		if (clgss->m_gpusort)
			clReleaseMemObject(clgss->m_gpusort);
		if (clgss->m_kds)
			clReleaseMemObject(clgss->m_kds);
		if (clgss->kern_prep)
			clReleaseKernel(clgss->kern_prep);
		for (i=0; i < clgss->kern_sort_nums; i++)
			clReleaseKernel(clgss->kern_sort[i]);
		if (clgss->program && clgss->program != BAD_OPENCL_PROGRAM)
			clReleaseProgram(clgss->program);
		if (clgss->kern_sort)
			free(clgss->kern_sort);
		if (clgss->kds)
		{
			munmap(clgss->kds, clgss->kds->length);
			close(clgss->kds_fdesc);
		}
	}
	gpusort->msg.errcode = rc;
	pgstrom_reply_message(&gpusort->msg);
}
