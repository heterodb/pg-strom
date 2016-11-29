/*
 * gpusort.c
 *
 * GPU+CPU Hybrid Sorting
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
#include "access/xact.h"
#include "catalog/pg_type.h"
#include "commands/dbcommands.h"
#include "executor/nodeCustom.h"
#include "nodes/nodeFuncs.h"
#include "nodes/makefuncs.h"
#include "optimizer/cost.h"
#include "parser/parsetree.h"
#include "postmaster/bgworker.h"
#include "storage/dsm.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/ruleutils.h"
#include "utils/snapmgr.h"
#include "pg_strom.h"
#include "cuda_gpusort.h"

typedef struct
{
	Cost		startup_cost;	/* cost we actually estimated */
	Cost		total_cost;		/* cost we actually estimated */
	const char *kern_source;
	int			extra_flags;
	List	   *used_params;
	int			num_segments;
	Size		segment_nrooms;
	Size		segment_extra;
	/* delivered from original Sort */
	int			numCols;		/* number of sort-key columns */
	AttrNumber *sortColIdx;		/* their indexes in the target list */
	Oid		   *sortOperators;	/* OIDs of operators to sort them by */
	Oid		   *collations;		/* OIDs of collations */
	bool	   *nullsFirst;		/* NULLS FIRST/LAST directions */
	bool		varlena_keys;	/* True, if here are varlena keys */
} GpuSortInfo;

static inline void
form_gpusort_info(CustomScan *cscan, GpuSortInfo *gs_info)
{
	List   *privs = NIL;
	List   *temp;
	int		i;

	privs = lappend(privs, makeInteger(double_as_long(gs_info->startup_cost)));
	privs = lappend(privs, makeInteger(double_as_long(gs_info->total_cost)));
	privs = lappend(privs, makeString(pstrdup(gs_info->kern_source)));
	privs = lappend(privs, makeInteger(gs_info->extra_flags));
	privs = lappend(privs, gs_info->used_params);
	privs = lappend(privs, makeInteger(gs_info->num_segments));
	privs = lappend(privs, makeInteger(gs_info->segment_nrooms));
	privs = lappend(privs, makeInteger(gs_info->segment_extra));
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
	/* varlena_keys */
	privs = lappend(privs, makeInteger(gs_info->varlena_keys));

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

	gs_info->startup_cost = long_as_double(intVal(list_nth(privs, pindex++)));
	gs_info->total_cost = long_as_double(intVal(list_nth(privs, pindex++)));
	gs_info->kern_source = strVal(list_nth(privs, pindex++));
	gs_info->extra_flags = intVal(list_nth(privs, pindex++));
	gs_info->used_params = list_nth(privs, pindex++);
	gs_info->num_segments = intVal(list_nth(privs, pindex++));
	gs_info->segment_nrooms = intVal(list_nth(privs, pindex++));
	gs_info->segment_extra = intVal(list_nth(privs, pindex++));
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
		gs_info->collations[i++] = lfirst_oid(cell);

	/* nullsFirst */
	temp = list_nth(privs, pindex++);
    Assert(list_length(temp) == gs_info->numCols);
	gs_info->nullsFirst = palloc0(sizeof(bool) * gs_info->numCols);
	i = 0;
	foreach (cell, temp)
		gs_info->nullsFirst[i++] = lfirst_int(cell);
	/* varlena_keys */
	gs_info->varlena_keys = intVal(list_nth(privs, pindex++));

	return gs_info;
}

typedef struct
{
	GpuTaskState	   *gts;
	cl_int				refcnt;				/* reference counter */
	cl_int				segid;				/* index on gss->seg_* */
	CUdeviceptr			m_kds_slot;			/* kds_slot */
	CUdeviceptr			m_kresults;			/* kresults */
	CUevent				ev_setup_segment;	/* event to sync setup_segment */
	CUevent			   *ev_kern_proj;		/* event to sync projection */
	cl_int				cuda_index;
	cl_uint				num_chunks;
	cl_uint				max_chunks;
	cl_uint				nitems_total;
	cl_bool				has_terminator;
	cl_bool				cpu_fallback;
	pgstrom_data_store *pds_slot;
	kern_resultbuf		kresults;
} gpusort_segment;

typedef struct
{
	GpuTask				task;
	pgstrom_data_store *pds_in;			/* source of data chunk */
	gpusort_segment	   *segment;		/* sorting segment */
	cl_bool				is_terminator;	/* true, if terminator proces */
	cl_uint				seg_ev_index;	/* index to ev_kern_proj */
	CUfunction			kern_proj;		/* gpusort_projection */
	CUfunction			kern_main;		/* gpusort_main */
	CUdeviceptr			m_gpusort;
	CUdeviceptr			m_kds_in;
	CUevent				ev_dma_send_start;
	CUevent				ev_dma_send_stop;
	CUevent				ev_dma_recv_start;
	CUevent				ev_dma_recv_stop;
	kern_gpusort		kern;
} pgstrom_gpusort;


typedef struct
{
	GpuTaskState	gts;

	/* sorting segments  */
	cl_uint			num_segments;
	cl_uint			num_segments_limit;
	pgstrom_data_store **seg_slots;		/* copy of kern_data_store (SLOT) */
	kern_resultbuf **seg_results;	/* copy of kern_resultbuf */
	cl_uint		   *seg_curpos;	/* current position to fetch */
	cl_uint		  **seg_lstree;	/* large-small tree */
	cl_uint			seg_lstree_depth;	/* depth of lstree */
	gpusort_segment	*curr_segment; /* the latest segment */
	Size			segment_nrooms;	/* planned best nrooms per segment */
	Size			segment_extra;	/* planned best extra length */
	cl_uint			segment_nchunks;/* expected number of chunks per seg */

	/* copied from the plan node */
    int				numCols;		/* number of sort-key columns */
    AttrNumber	   *sortColIdx;		/* their indexes in the target list */
    Oid			   *sortOperators;	/* OIDs of operators to sort them by */
	Oid			   *collations;		/* OIDs of collations */
	bool		   *nullsFirst;		/* NULLS FIRST/LAST directions */
	bool			varlena_keys;	/* True, if varlena sorting key exists */
	SortSupportData *ssup_keys;		/* XXX - used by fallback function */

	/* misc stuff */
	cl_uint		   *markpos_buf;
	TupleTableSlot *overflow_slot;
	pgstrom_data_store *overflow_pds;
} GpuSortState;

/*
 * declaration of static variables and functions
 */
static CustomScanMethods	gpusort_scan_methods;
static CustomExecMethods	gpusort_exec_methods;
static bool					enable_gpusort;
static bool					debug_force_gpusort;

static GpuTask *gpusort_next_chunk(GpuTaskState *gts);
static TupleTableSlot *gpusort_next_tuple(GpuTaskState *gts);
static bool gpusort_task_process(GpuTask *gtask);
static bool gpusort_task_complete(GpuTask *gtask);
static void gpusort_task_release(GpuTask *gtask);
static void gpusort_fallback_quicksort(GpuSortState *gss,
									   kern_resultbuf *kresults,
									   kern_data_store *kds_slot,
									   cl_int lbound, cl_int rbound);

/*
 * cost_gpusort
 *
 * cost estimation for GpuSort
 */
#define LOG2(x)		(log(x) / 0.693147180559945)

static void
cost_gpusort(PlannedStmt *pstmt, Sort *sort,
			 Cost *p_startup_cost, Cost *p_total_cost,
			 cl_uint *p_num_segments,
			 Size *p_segment_nrooms, Size *p_segment_extra)
{
	Plan	   *outer_plan = outerPlan(sort);
	double		ntuples = outer_plan->plan_rows;
	double		ntuples_per_chunk;
	int			plan_width = outer_plan->plan_width;
	int			nattrs = list_length(outer_plan->targetlist);
	int			extra_len;
	int			unitsz_fmt_slot;
	int			unitsz_fmt_row;
	cl_uint		num_segments;
	double		segment_nrooms;
	double		segment_extra;
	bool		only_inline_attrs;
	Cost		startup_cost;
	Cost		run_cost;
	Cost		sorting_cost;
	Cost		cpu_comp_cost = 2.0 * cpu_operator_cost;
	Cost		gpu_comp_cost = 2.0 * pgstrom_gpu_operator_cost;
	int			k;
	ListCell   *lc;

	if (ntuples < 2.0)
		ntuples = 2.0;

	/* Cost come from outer-plan and sub-plans */
	startup_cost = outer_plan->total_cost;
	run_cost = 0.0;
	foreach (lc, sort->plan.initPlan)
	{
		SubPlan	   *subplan = lfirst(lc);
		Plan	   *temp = list_nth(pstmt->subplans, subplan->plan_id - 1);

		startup_cost += temp->total_cost;
	}
	/* Fixed cost to setup/launch GPU kernel */
	startup_cost += pgstrom_gpu_setup_cost;

	/*
	 * Estimation of unit size of extra buffer consumption.
	 *
	 * Non-inline attributes (indirect or varlena type) need extra area of
	 * KDS_slot on gpusort_projection(). We will estimate an average length
	 * of per-tuple consumption. 
	 */
	extra_len = plan_width;
	only_inline_attrs = true;
	foreach (lc, outer_plan->targetlist)
	{
		TargetEntry	   *tle = lfirst(lc);
		Oid				type_oid = exprType((Node *) tle->expr);
		int16			type_len;
		bool			type_byval;
		char			type_align;

		get_typlenbyvalalign(type_oid, &type_len, &type_byval, &type_align);
		if (type_byval)
			extra_len -= type_len;
		else
		{
			/* adjust for alignment */
			extra_len += typealign_get_width(type_align);
			only_inline_attrs = false;
		}
	}
	if (only_inline_attrs)
		extra_len = 0;		/* mistake in width estimation */
	else if (extra_len < 32)
		extra_len = 32;		/* minimum guarantee if extra area is needed */

	unitsz_fmt_slot = (MAXALIGN(sizeof(Datum) * nattrs) +
					   MAXALIGN(sizeof(cl_char) * nattrs) +
					   MAXALIGN(extra_len));

	/* also, unitsz in row format for kds_in */
	unitsz_fmt_row = MAXALIGN(offsetof(kern_tupitem, htup) +
							  MAXALIGN(offsetof(HeapTupleHeaderData, t_bits) +
									   sizeof(Oid) +
									   BITMAPLEN(nattrs)) +
							  MAXALIGN(plan_width)) + sizeof(cl_uint);
	/* number of chunks we can pack in a KDS chunk */
	ntuples_per_chunk = (pgstrom_chunk_size() -
						 KDS_CALCULATE_HEAD_LENGTH(nattrs)) / unitsz_fmt_row;

	/*
	 * Estimate an optimal number of segments
	 */
	num_segments = ceil((double)(sizeof(cl_uint) + unitsz_fmt_slot) * ntuples /
						(double)(gpuMemMaxAllocSize() / 2 -
								 KDS_CALCULATE_HEAD_LENGTH(nattrs) -
								 offsetof(kern_resultbuf, results)));
	k = num_segments = Max(num_segments, 1);

	sorting_cost = DBL_MAX;
	for (;;)
	{
		double	ntuples_per_segment = ntuples / (double) k;
		double	nchunks_per_segment;
		Cost	cost_load_chunk = cpu_tuple_cost * ntuples_per_chunk;
		Cost	cost_load_segment;
		Cost	cost_gpu_sorting;
		Cost	cost_dma_recv;
		Cost	cost_others;
		Cost	tentative;

		/*
		 * data load to a particular segment is consists of multiple kernel
		 * call with individual data-store. KDS setup and kernel execution
		 * are handled in asynchronously, so our cost estimation assumes
		 * that more expensive work dominates the entire data loading,
		 * except for the last small fraction.
		 */
		nchunks_per_segment = ceil(ntuples_per_segment / ntuples_per_chunk);
		if (nchunks_per_segment < 1.0)
			nchunks_per_segment = 1.0;
		cost_load_segment =
			Max(cost_load_chunk, pgstrom_gpu_dma_cost) * nchunks_per_segment +
			Min(cost_load_chunk, pgstrom_gpu_dma_cost);
		segment_nrooms = ntuples_per_segment;
		segment_extra  = (double)extra_len * segment_nrooms;

		/*
		 * Our bitonic sorting logic taks O(N * Log2(N)) on GPU device
		 */
		cost_gpu_sorting = (gpu_comp_cost *
							ntuples_per_segment *
							LOG2(ntuples_per_segment));
		/*
		 * Cost to write back the sorted results
		 */
		cost_dma_recv = pgstrom_gpu_dma_cost * nchunks_per_segment;

		/*
		 * Our cost model assumes asynchronous executions; each fraction of
		 * tasks are executed, but usually these tasks needs different time
		 * to run. Thus, entire response time shall be dominated by the most
		 * expensive portion. In the diagram below, "GPU sorting" dominates
		 * the entire processing time.
		 *
		 *  +-----------+---------------+--------+
		 *  | data load |  GPU sorting  |DMA Recv|
		 *  +-----------+-----------+---+--------+------+--------+
		 *              | data load |...|  GPU sorting  |DMA Recv|
		 *              +-----------+---+-------+-------+--------+------+---
		 *                          | data load |.......|  GPU sorting  |DMA
		 *                          +-----------+-------+---------------+---
		 * The total cost in GPU portion shall be:
		 *   ((cost of most expensive job) * (# of segments) +
		 *    (cost of other portions))
		 *
		 * Also, CPU has to merge the individual sorted segments. Its cost
		 * depends on number of segments and number of tuples per segment.
		 * If we have k-segments, CPU has to compare (k-1) times to return
		 * a tuple.
		 */
		cost_others = (cost_load_segment + cost_gpu_sorting + cost_dma_recv) -
			Max3(cost_load_segment, cost_gpu_sorting, cost_dma_recv);

		tentative = Max3(cost_load_segment,
						 cost_gpu_sorting,
						 cost_dma_recv) * (double) k + cost_others;
		/* fine grained segmentation also makes CPU busy... */
		tentative += cpu_comp_cost * (k-1) * ntuples;

		if (tentative < sorting_cost)
		{
			num_segments = k;
			sorting_cost = tentative;

			/* try next segment size */
			k += (k < 5 ? 1 : (k < 40 ? k / 2 : k));
			continue;
		}
		break;
	}
	startup_cost += sorting_cost;
	run_cost += cpu_operator_cost * ntuples;

	/* result */
	*p_startup_cost = startup_cost;
	*p_total_cost = startup_cost + run_cost;
	*p_num_segments = num_segments;
	*p_segment_nrooms = (Size)(segment_nrooms * pgstrom_chunk_size_margin);
	*p_segment_extra = (Size)(segment_extra * pgstrom_chunk_size_margin);
}

static char *
pgstrom_gpusort_codegen(Sort *sort, codegen_context *context)
{
	StringInfoData	kern;
	StringInfoData	body;
	int				i;

	initStringInfo(&kern);
	initStringInfo(&body);

	/*
	 * STATIC_FUNCTION(cl_int)
	 * gpusort_keycomp(kern_context *kcxt,
	 *                 kern_data_store *kds_slot,
	 *                 size_t x_index,
	 *                 size_t y_index);
	 */
	appendStringInfo(
		&body,
		"STATIC_FUNCTION(cl_int)\n"
		"gpusort_keycomp(kern_context *kcxt,\n"
		"                kern_data_store *kds_slot,\n"
		"                size_t x_index,\n"
		"                size_t y_index)\n"
		"{\n"
		"  pg_anytype_t KVAR_X  __attribute__((unused));\n"
		"  pg_anytype_t KVAR_Y  __attribute__((unused));\n"
		"  pg_int4_t comp;\n\n"
		"  assert(kds_slot->format == KDS_FORMAT_SLOT);\n\n");

	for (i=0; i < sort->numCols; i++)
	{
		TargetEntry *tle;
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

		/*
		 * Get direction of the sorting
		 */
		if (!get_ordering_op_properties(sort_op,
										&opfamily,
										&sort_type,
										&strategy))
			elog(ERROR, "operator %u is not a valid ordering operator",
				 sort_op);
		is_reverse = (strategy == BTGreaterStrategyNumber);

		/*
		 * device type lookup for comparison
		 */
		tle = get_tle_by_resno(sort->plan.targetlist, colidx);
		if (!tle || !IsA(tle->expr, Var))
			elog(ERROR, "Bug? resno %d not found on tlist or not varnode: %s",
				 colidx, nodeToString(tle->expr));
		sort_type = exprType((Node *) tle->expr);

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

		/* Logic to compare */
		appendStringInfo(
			&body,
			"  /* sort key comparison on the resource %d */\n"
			"  KVAR_X.%s_v = pg_%s_vref(kds_slot,kcxt,%d,x_index);\n"
			"  KVAR_Y.%s_v = pg_%s_vref(kds_slot,kcxt,%d,y_index);\n"
			"  if (!KVAR_X.%s_v.isnull && !KVAR_Y.%s_v.isnull)\n"
			"  {\n"
			"    comp = pgfn_%s(kcxt, KVAR_X.%s_v, KVAR_Y.%s_v);\n"
			"    if (comp.value != 0)\n"
			"      return %s;\n"
			"  }\n"
			"  else if (KVAR_X.%s_v.isnull && !KVAR_Y.%s_v.isnull)\n"
			"    return %d;\n"
			"  else if (!KVAR_X.%s_v.isnull && KVAR_Y.%s_v.isnull)\n"
			"    return %d;\n"
			"\n",
			tle->resno,
			dtype->type_name, dtype->type_name, colidx-1,
			dtype->type_name, dtype->type_name, colidx-1,
			dtype->type_name, dtype->type_name,
			dfunc->func_devname, dtype->type_name, dtype->type_name,
			is_reverse ? "-comp.value" : "comp.value",
			dtype->type_name, dtype->type_name,
			null_first ? -1 : 1,
			dtype->type_name, dtype->type_name,
			null_first ? 1 : -1);
	}
	appendStringInfo(
		&body,
		"  return 0;\n"
		"}\n");

	/* functions declarations, if any */
	pgstrom_codegen_func_declarations(&kern, context);
	/* special expression declarations, if any */
	pgstrom_codegen_expr_declarations(&kern, context);

	if (kern.len > 0)
		appendStringInfo(&kern, "\n");
	appendStringInfoString(&kern, body.data);

	pfree(body.data);

	return kern.data;
}

void
pgstrom_try_insert_gpusort(PlannedStmt *pstmt, Plan **p_plan)
{
	Sort	   *sort = (Sort *)(*p_plan);
	List	   *tlist = sort->plan.targetlist;
	ListCell   *cell;
	Cost		startup_cost;
	Cost		total_cost;
	cl_uint		num_segments;
	Size		segment_nrooms;
	Size		segment_extra;
	CustomScan *cscan;
	Plan	   *subplan;
	GpuSortInfo	gs_info;
	codegen_context context;
	bool		varlena_keys = false;
	int			i;

	/* nothing to do, if feature is turned off */
	if (!pgstrom_enabled || !enable_gpusort)
	  return;

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

		/* Does key contain varlena data type? */
		if (dtype->type_length < 0)
			varlena_keys = true;
	}

	/*
	 * OK, cost estimation with GpuSort
	 */
	cost_gpusort(pstmt, sort,
				 &startup_cost, &total_cost,
				 &num_segments, &segment_nrooms, &segment_extra);

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
	 * OK, estimated GpuSort cost is enough reasonable to inject.
	 * Let's construct a new one.
	 */

	/*
	 * XXX - Here would be a location to put pgstrom_pullup_outer_scan().
	 * However, the upcoming v9.6 support upper path construction by
	 * extensions. We can add entirely graceful approach than previous
	 * pull-up outer "plan" implementation.
	 */

	/*
	 * OK, expected GpuSort cost is enough reasonable to run.
	 * Let's return the 
	 */
	cscan = makeNode(CustomScan);
	cscan->scan.plan.startup_cost = sort->plan.startup_cost;
	cscan->scan.plan.total_cost = sort->plan.total_cost;
	cscan->scan.plan.plan_rows = sort->plan.plan_rows;
	cscan->scan.plan.plan_width = sort->plan.plan_width;
	cscan->scan.plan.targetlist = NIL;
	cscan->scan.scanrelid       = 0;
	cscan->custom_scan_tlist    = NIL;
	cscan->custom_relids        = NULL;
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
								  list_length(cscan->custom_scan_tlist) + 1,
								  tle->resname ? pstrdup(tle->resname) : NULL,
								  false);
		cscan->custom_scan_tlist = lappend(cscan->custom_scan_tlist, tle_new);
	}
	outerPlan(cscan) = subplan;
	cscan->scan.plan.initPlan = sort->plan.initPlan;

	pgstrom_init_codegen_context(&context);
	gs_info.startup_cost = startup_cost;
	gs_info.total_cost = total_cost;
	gs_info.kern_source = pgstrom_gpusort_codegen(sort, &context);
	gs_info.extra_flags = context.extra_flags |
		DEVKERNEL_NEEDS_DYNPARA | DEVKERNEL_NEEDS_GPUSORT;
	gs_info.used_params = context.used_params;
	gs_info.num_segments = num_segments;
	gs_info.segment_nrooms = segment_nrooms;
	gs_info.segment_extra = segment_extra;
	gs_info.numCols = sort->numCols;
	gs_info.sortColIdx = sort->sortColIdx;
	gs_info.sortOperators = sort->sortOperators;
	gs_info.collations = sort->collations;
	gs_info.nullsFirst = sort->nullsFirst;
	gs_info.varlena_keys = varlena_keys;	// still used?
	form_gpusort_info(cscan, &gs_info);

	*p_plan = &cscan->scan.plan;
}

bool
pgstrom_plan_is_gpusort(const Plan *plan)
{
	if (IsA(plan, CustomScan) &&
		((CustomScan *) plan)->methods == &gpusort_scan_methods)
		return true;
	return false;
}

void
assign_gpusort_session_info(StringInfo buf, GpuTaskState *gts)
{
	TupleDesc	tupdesc = GTS_GET_RESULT_TUPDESC(gts);

	appendStringInfo(
		buf,
		"#define GPUSORT_DEVICE_PROJECTION_NFIELDS %u\n\n",
		tupdesc->natts);
}

static Node *
gpusort_create_scan_state(CustomScan *cscan)
{
	GpuSortState   *gss = (GpuSortState *) newNode(sizeof(GpuSortState),
												   T_CustomScanState);
	/* Set tag and executor callbacks */
	gss->gts.css.flags = cscan->flags;
	gss->gts.css.methods = &gpusort_exec_methods;

	return (Node *) gss;
}

static void
gpusort_begin(CustomScanState *node, EState *estate, int eflags)
{
	GpuContext		   *gcontext = NULL;
	GpuSortState	   *gss = (GpuSortState *) node;
	CustomScan		   *cscan = (CustomScan *) node->ss.ps.plan;
	GpuSortInfo		   *gs_info = deform_gpusort_info(cscan);
	PlanState		   *subplan_state;
	TupleDesc			tupdesc;
	cl_int				i;

	/* activate GpuContext for device execution */
	if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
		gcontext = pgstrom_get_gpucontext();
	/* common GpuTaskState setup */
	pgstrom_init_gputaskstate(gcontext, &gss->gts, estate);
	gss->gts.cb_task_process = gpusort_task_process;
	gss->gts.cb_task_complete = gpusort_task_complete;
	gss->gts.cb_task_release = gpusort_task_release;
	gss->gts.cb_next_chunk = gpusort_next_chunk;
	gss->gts.cb_next_tuple = gpusort_next_tuple;
	/* re-initialization of scan-descriptor and projection-info */
	tupdesc = ExecCleanTypeFromTL(cscan->custom_scan_tlist, false);
	ExecAssignScanType(&gss->gts.css.ss, tupdesc);
	ExecAssignScanProjectionInfoWithVarno(&gss->gts.css.ss, INDEX_VAR);

	/*
	 * Unlike built-in Sort node doing, our GpuSort "always" provide
	 * a materialized output, so it is unconditionally possible to run
	 * backward scan, random accesses and rewind the position.
	 */
	eflags &= ~(EXEC_FLAG_REWIND | EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK);

	gss->markpos_buf = NULL;	/* to be set later */

	/* initialize child exec node */
	subplan_state = ExecInitNode(outerPlan(cscan), estate, eflags);
	/* informs our preferred tuple format, if supported */
	if (pgstrom_bulk_exec_supported(subplan_state))
	{
		((GpuTaskState *) subplan_state)->be_row_format = true;
		gss->gts.outer_bulk_exec = true;
	}
	outerPlanState(gss) = subplan_state;

	/* for GPU bitonic sorting */
	pgstrom_assign_cuda_program(&gss->gts,
								gs_info->used_params,
								gs_info->kern_source,
								gs_info->extra_flags);
	if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
		pgstrom_load_cuda_program_legacy(&gss->gts, true);

	/* array for data-stores */
	gss->num_segments = 0;
	gss->num_segments_limit = gs_info->num_segments + 10;
	gss->seg_slots = palloc0(sizeof(pgstrom_data_store *) *
							 gss->num_segments_limit);
	gss->seg_results = palloc0(sizeof(kern_resultbuf *) *
							   gss->num_segments_limit);
	gss->seg_curpos = NULL;	/* to be set later */
	gss->seg_lstree = NULL;	/* to be set later */
	gss->segment_nrooms = gs_info->segment_nrooms;
	gss->segment_extra = gs_info->segment_extra;

	/* sorting keys */
	gss->numCols = gs_info->numCols;
	gss->sortColIdx = gs_info->sortColIdx;
	gss->sortOperators = gs_info->sortOperators;
	gss->collations = gs_info->collations;
	gss->nullsFirst = gs_info->nullsFirst;

	gss->ssup_keys = palloc0(sizeof(SortSupportData) * gss->numCols);
	for (i=0; i < gss->numCols; i++)
	{
		SortSupport		ssup = gss->ssup_keys + i;

		ssup->ssup_cxt = estate->es_query_cxt;
		ssup->ssup_collation = gss->collations[i];
		ssup->ssup_nulls_first = gss->nullsFirst[i];
		ssup->ssup_attno = gss->sortColIdx[i];
		PrepareSortSupportFromOrderingOp(gss->sortOperators[i], ssup);
	}

	/* init perfmon */
	pgstrom_init_perfmon(&gss->gts);
}

static TupleTableSlot *
gpusort_exec(CustomScanState *node)
{
	return pgstrom_exec_gputask((GpuTaskState *) node);
}

static void
gpusort_end(CustomScanState *node)
{
	GpuSortState	   *gss = (GpuSortState *) node;
	pgstrom_data_store *pds_slot;
	gpusort_segment	   *segment;
	cl_int				i;

	/* Clean up subtree */
	ExecEndNode(outerPlanState(node));

	for (i=0; i < gss->num_segments; i++)
	{
		pds_slot = gss->seg_slots[i];
		PDS_release(pds_slot);

		segment = (gpusort_segment *)((char *)gss->seg_results[i] -
									  offsetof(gpusort_segment, kresults));
		pfree(segment);
	}

	/*
	 * Cleanup and relase any concurrent tasks
	 */
	pgstrom_release_gputaskstate(&gss->gts);
}

static void
gpusort_rescan(CustomScanState *node)
{
	GpuSortState   *gss = (GpuSortState *) node;

	/* must drop pointer to sort result tuple */
	ExecClearTuple(node->ss.ps.ps_ResultTupleSlot);

	if (gss->seg_curpos)
	{
		cl_uint		i;

		pfree(gss->seg_curpos);
		gss->seg_curpos = NULL;

		pfree(gss->markpos_buf);
		gss->markpos_buf = NULL;

		for (i=0; i <= gss->seg_lstree_depth; i++)
			pfree(gss->seg_lstree[i]);
		pfree(gss->seg_lstree);
		gss->seg_lstree = NULL;
	}

	/*
	 * If subnode is to be rescanned then we forget previous sort results;
	 * we have to re-read the subplan and re-sort. Also must re-sort if the
	 * bounded-sort parameters changed or we didn't select randomAccess.
	 */
	if (outerPlanState(gss)->chgParam != NULL)
	{
		cl_uint		i;

		/* cleanup and release any concurrent tasks */
		pgstrom_cleanup_gputaskstate(&gss->gts);

		for (i=0; i < gss->num_segments; i++)
		{
			gpusort_segment	   *segment = (gpusort_segment *)
				((char *)gss->seg_results[i] - offsetof(gpusort_segment,
														kresults));
			Assert(gss->seg_slots[i] == segment->pds_slot);
			PDS_release(gss->seg_slots[i]);
			pfree(segment);
		}
		gss->curr_segment = NULL;
		gss->num_segments = 0;
	}
}

static void
gpusort_mark_pos(CustomScanState *node)
{
	GpuSortState   *gss = (GpuSortState *) node;

	if (gss->seg_curpos)
	{
		Assert(gss->markpos_buf != NULL);
		memcpy(gss->markpos_buf,
			   gss->seg_curpos,
			   sizeof(cl_uint) * gss->num_segments);
	}
}

static void
gpusort_restore_pos(CustomScanState *node)
{
	GpuSortState   *gss = (GpuSortState *) node;

	if (gss->seg_curpos)
	{
		Assert(gss->seg_curpos != NULL);
		memcpy(gss->seg_curpos,
			   gss->markpos_buf,
			   sizeof(cl_uint) * gss->num_segments);
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

	/* actual cost we estimated */
	if (es->verbose && es->costs)
	{
		if (es->format == EXPLAIN_FORMAT_TEXT)
		{
			char   *temp = psprintf("%.2f...%.2f",
									gs_info->startup_cost,
									gs_info->total_cost);
			ExplainPropertyText("Real cost", temp, es);
			pfree(temp);
		}
		else
		{
			ExplainPropertyFloat("Real startup cost",
								 gs_info->startup_cost, 3, es);
			ExplainPropertyFloat("Real total cost",
								 gs_info->total_cost, 3, es);
		}
	}

	/* shows sorting keys */
	context = set_deparse_context_planstate(es->deparse_cxt,
											(Node *) node,
											ancestors);
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

	/*
	 * shows resource consumption, if executed and have more than zero
	 * rows.
	 */
	if (es->analyze)
	{
		const char *sort_method;
		Size		total_consumption = 0UL;

		if (gss->num_segments > 1)
			sort_method = "GPU/Bitonic + CPU/Merge";
		else
			sort_method = "GPU/Bitonic";

		for (i=0; i < gss->num_segments; i++)
		{
			pgstrom_data_store *pds = gss->seg_slots[i];
			kern_resultbuf	   *kresults = gss->seg_results[i];

			total_consumption += GPUMEMALIGN(pds->kds_length) +
				GPUMEMALIGN(offsetof(gpusort_segment, kresults) +
							offsetof(kern_resultbuf, results) +
							sizeof(cl_uint) * kresults->nrooms);
		}

		if (es->format == EXPLAIN_FORMAT_TEXT)
		{
			appendStringInfoSpaces(es->str, es->indent * 2);
			appendStringInfo(es->str, "Sort Method: %s used: %s\n",
							 sort_method,
							 format_bytesz(total_consumption));
		}
		else
		{
			ExplainPropertyText("Sort Method", sort_method, es);
			ExplainPropertyLong("Sort Space Used", total_consumption, es);
		}
		/* number of segments */
		ExplainPropertyInteger("Number of segments", gss->num_segments, es);
	}
	pgstrom_explain_gputaskstate(&gss->gts, es);
}

/*
 * Create/Get/Put gpusort_segment
 */
static gpusort_segment *
gpusort_create_segment(GpuSortState *gss)
{
	GpuContext		   *gcontext = gss->gts.gcontext;
	cl_uint				seg_nrooms = gss->segment_nrooms;
	cl_uint				seg_nchunks = gss->segment_nchunks + 20;
	TupleDesc			tupdesc = GTS_GET_RESULT_TUPDESC(gss);
	gpusort_segment	   *segment;
	kern_resultbuf	   *kresults;

	segment = MemoryContextAlloc(gcontext->memcxt,
								 offsetof(gpusort_segment, kresults) +
								 STROMALIGN(offsetof(kern_resultbuf,
													 results[seg_nrooms])) +
								 sizeof(CUevent) * seg_nchunks);
	kresults = &segment->kresults;

	memset(segment, 0, sizeof(gpusort_segment));
	segment->refcnt = 1;
	segment->segid = -1;	/* caller shall set */
	segment->m_kds_slot = 0UL;
	segment->m_kresults = 0UL;
	segment->cuda_index = gcontext->next_context++ % gcontext->num_context;
	segment->num_chunks = 0;
	segment->max_chunks = seg_nchunks;
	segment->nitems_total = 0;
	segment->has_terminator = false;
	segment->ev_setup_segment = NULL;
	segment->ev_kern_proj = (CUevent *)
		((char *)kresults + STROMALIGN(offsetof(kern_resultbuf,
												results[seg_nrooms])));
	memset(segment->ev_kern_proj, 0, sizeof(CUevent) * seg_nchunks);

	segment->pds_slot = PDS_create_slot(gcontext,
										tupdesc,
										gss->segment_nrooms,
										gss->segment_extra,
										false);
	kresults->nrels = 1;
	kresults->nrooms = seg_nrooms;
	kresults->nitems = 0;

	return segment;
}

static gpusort_segment *
gpusort_get_segment(gpusort_segment *segment)
{
	Assert(segment->refcnt > 0);
	segment->refcnt++;
	return segment;
}

static void
gpusort_put_segment(gpusort_segment *segment)
{
	Assert(segment->refcnt > 0);
	if (--segment->refcnt == 0)
	{
		CUresult	rc;
		int			i;

		/* device memory is either already released or not acquired yet */
		Assert(segment->m_kds_slot == 0UL);
		Assert(segment->m_kresults == 0UL);

		/* release the data store */
		PDS_release(segment->pds_slot);

		/* event objects also */
		rc = cuEventDestroy(segment->ev_setup_segment);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuEventDestroy: %s", errorText(rc));
		for (i=0; i < segment->num_chunks; i++)
		{
			if (segment->ev_kern_proj[i])
			{
				rc = cuEventDestroy(segment->ev_kern_proj[i]);
				if (rc != CUDA_SUCCESS)
					elog(WARNING, "failed on cuEventDestroy: %s",
						 errorText(rc));
			}
		}
		/* release itself */
		pfree(segment);
	}
}

static GpuTask *
gpusort_create_task(GpuSortState *gss, pgstrom_data_store *pds_in,
					cl_uint valid_nitems, bool is_last_chunk,
					gpusort_segment *segment)
{
	GpuContext		   *gcontext = gss->gts.gcontext;
	pgstrom_gpusort	   *pgsort;

	/* construction of pgstrom_gpusort */
	pgsort = MemoryContextAllocZero(gcontext->memcxt,
									offsetof(pgstrom_gpusort, kern) +
									offsetof(kern_gpusort, kparams) +
									gss->gts.kern_params->length);
	pgstrom_init_gputask(&gss->gts, &pgsort->task);
	pgsort->pds_in = pds_in;
	pgsort->segment = NULL;			/* to be set below */
	pgsort->is_terminator = false;	/* to be set below */
	pgsort->seg_ev_index = 0;		/* to be set below */

	memcpy(&pgsort->kern.kparams,
		   gss->gts.kern_params,
		   gss->gts.kern_params->length);

	if (segment)
	{
		/*
		 * MEMO: Caller may give a certain segment to attach.
		 * It is available only when the supplied segment has enough space
		 * and its terminator task was not launched yet; = the terminator
		 * task is still in the pending list.
		 *
		 * The caller has to ensure the target segment is available for
		 * this share ride
		 */
		Assert(segment->has_terminator);
		Assert(segment->num_chunks < segment->max_chunks);
		Assert(segment->nitems_total +
			   valid_nitems <= segment->kresults.nrooms);
	}
	else
	{
		segment = gss->curr_segment;

		if (!segment || segment->has_terminator)
		{
			segment = gpusort_create_segment(gss);
			if (gss->num_segments == gss->num_segments_limit)
			{
				gss->num_segments_limit += Max(20, gss->num_segments_limit);
				gss->seg_slots = repalloc(gss->seg_slots,
										  sizeof(pgstrom_data_store *) *
										  gss->num_segments_limit);
				gss->seg_results = repalloc(gss->seg_results,
											sizeof(kern_resultbuf *) *
											gss->num_segments_limit);
			}
			segment->segid = gss->num_segments;
			gss->seg_slots[gss->num_segments] = segment->pds_slot;
			gss->seg_results[gss->num_segments] = &segment->kresults;
			gss->num_segments++;
			gss->curr_segment = segment;
		}
	}
	Assert(segment->num_chunks < segment->max_chunks);
	pgsort->seg_ev_index = segment->num_chunks++;
	Assert(valid_nitems <= pds_in->kds->nitems);
	segment->nitems_total += valid_nitems;

	/*
	 * This task shall perform as a terminator if no more tasks shall be
	 * added or no more space for data load obviously.
	 */
	if (is_last_chunk ||
		segment->num_chunks == segment->max_chunks ||
        segment->nitems_total > segment->kresults.nrooms)
	{
		pgsort->is_terminator = true;
		segment->has_terminator = true;
	}
	pgsort->segment = gpusort_get_segment(segment);
	pgsort->kern.segid = pgsort->segment->segid;
	pgsort->task.cuda_index = segment->cuda_index;	/* bind to the same GPU */
	return &pgsort->task;
}

static GpuTask *
gpusort_next_chunk(GpuTaskState *gts)
{
	GpuContext		   *gcontext = gts->gcontext;
	GpuSortState	   *gss = (GpuSortState *) gts;
	TupleDesc			tupdesc = GTS_GET_SCAN_TUPDESC(gts);
	pgstrom_data_store *pds = NULL;
	TupleTableSlot	   *slot;
	bool				is_last_chunk = false;
	struct timeval		tv1, tv2;

	/*
	 * Load tuples from the underlying plan node
	 */
	PERFMON_BEGIN(&gts->pfm, &tv1);
	if (!gss->gts.outer_bulk_exec)
	{
		while (!gss->gts.scan_done)
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
					is_last_chunk = true;
					pgstrom_deactivate_gputaskstate(gts);
					break;
				}
			}
			Assert(!TupIsNull(slot));

			if (!pds)
				pds = PDS_create_row(gcontext,
									 tupdesc,
									 pgstrom_chunk_size());

			if (!PDS_insert_tuple(pds, slot))
			{
				gss->overflow_slot = slot;
				break;
			}
		}
	}
	else
	{
		GpuTaskState   *subnode = (GpuTaskState *) outerPlanState(gss);

		/* Load a bunch of records at once on the first time */
		if (!gss->overflow_pds)
			gss->overflow_pds = BulkExecProcNode(subnode,
												 pgstrom_chunk_size());
		pds = gss->overflow_pds;
		if (pds)
		{
			gss->overflow_pds = BulkExecProcNode(subnode,
												 pgstrom_chunk_size());
			if (!gss->overflow_pds)
			{
				is_last_chunk = true;
				pgstrom_deactivate_gputaskstate(&gss->gts);
			}
		}
	}
	PERFMON_END(&gts->pfm, time_outer_load, &tv1, &tv2);
	if (!pds)
		return NULL;
	return gpusort_create_task(gss, pds, pds->kds->nitems,
							   is_last_chunk, NULL);
}

/*
 * gpusort_cpu_keycomp
 *
 * It compares two records according to the sorting keys. It is also used
 * by CPU fallback routine.
 */
static inline int
gpusort_cpu_keycomp(GpuSortState *gss,
                    Datum *x_values, bool *x_isnull,
                    Datum *y_values, bool *y_isnull)
{
	int		i, j, comp;

	for (i=0; i < gss->numCols; i++)
	{
		SortSupport		ssup = gss->ssup_keys + i;

		j = ssup->ssup_attno - 1;
		comp = ApplySortComparator(x_values[j],
								   x_isnull[j],
								   y_values[j],
								   y_isnull[j],
								   ssup);
		if (comp != 0)
			return comp;
	}
	return 0;
}

static inline void
gpusort_update_lstree(GpuSortState *gss, cl_uint depth, cl_uint index)
{
	cl_int		x_segid;
	cl_int		y_segid;
	cl_int		r_segid;

	Assert(depth < gss->seg_lstree_depth);
	Assert(index < (1U << depth));

	x_segid = gss->seg_lstree[depth+1][2 * index];
	y_segid = gss->seg_lstree[depth+1][2 * index + 1];
	if (x_segid < gss->num_segments && y_segid < gss->num_segments)
	{
		pgstrom_data_store *x_pds = gss->seg_slots[x_segid];
		pgstrom_data_store *y_pds = gss->seg_slots[y_segid];
		kern_resultbuf	   *x_kresults = gss->seg_results[x_segid];
		kern_resultbuf	   *y_kresults = gss->seg_results[y_segid];
		cl_uint				x_curpos = gss->seg_curpos[x_segid];
		cl_uint				y_curpos = gss->seg_curpos[y_segid];

		if (x_curpos < x_kresults->nitems &&
			y_curpos < y_kresults->nitems)
		{
			cl_uint		x_index = x_kresults->results[x_curpos];
			cl_uint		y_index = y_kresults->results[y_curpos];
			Datum	   *x_values = KERN_DATA_STORE_VALUES(x_pds->kds, x_index);
			bool	   *x_isnull = KERN_DATA_STORE_ISNULL(x_pds->kds, x_index);
			Datum	   *y_values = KERN_DATA_STORE_VALUES(y_pds->kds, y_index);
			bool	   *y_isnull = KERN_DATA_STORE_ISNULL(y_pds->kds, y_index);

			if (gpusort_cpu_keycomp(gss,
									x_values, x_isnull,
									y_values, y_isnull) < 0)
				r_segid = x_segid;
			else
				r_segid = y_segid;
		}
		else if (x_curpos < x_kresults->nitems)
			r_segid = x_segid;
		else if (y_curpos < y_kresults->nitems)
			r_segid = y_segid;
		else
			r_segid = INT_MAX;
	}
	else if (x_segid < gss->num_segments)
	{
		if (gss->seg_curpos[x_segid] < gss->seg_results[x_segid]->nitems)
			r_segid = x_segid;
		else
			r_segid = INT_MAX;
	}
	else if (y_segid < gss->num_segments)
	{
		if (gss->seg_curpos[y_segid] < gss->seg_results[y_segid]->nitems)
			r_segid = y_segid;
		else
			r_segid = INT_MAX;
	}
	else
		r_segid = INT_MAX;
	/* OK, r_segid is smaller */
	gss->seg_lstree[depth][index] = r_segid;
}

static TupleTableSlot *
gpusort_next_tuple(GpuTaskState *gts)
{
	GpuSortState	   *gss = (GpuSortState *) gts;
	pgstrom_gpusort	   *pgsort = (pgstrom_gpusort *) gss->gts.curr_task;
	TupleTableSlot	   *slot = gss->gts.css.ss.ps.ps_ResultTupleSlot;
	AttrNumber			natts = slot->tts_tupleDescriptor->natts;
	pgstrom_data_store *pds;
	kern_resultbuf	   *kresults;
	cl_uint				segid;
	cl_uint				curpos;
	cl_uint				index;
	Datum			   *values;
	bool			   *isnull;
	struct timeval		tv1, tv2, tv3;

	if (!pgsort)
		return NULL;

	ExecClearTuple(slot);

	PERFMON_BEGIN(&gss->gts.pfm, &tv1);
	if (!gss->seg_curpos)
	{
		/* construction of the initial lstree */
		EState		   *estate = gss->gts.css.ss.ps.state;
		MemoryContext	oldcxt;
		cl_int			depth;
		cl_int			i, j, k;

		oldcxt = MemoryContextSwitchTo(estate->es_query_cxt);
		gss->seg_curpos = palloc0(sizeof(cl_uint) * gss->num_segments);
		gss->markpos_buf = palloc0(sizeof(cl_uint) * gss->num_segments);

		depth = get_next_log2(gss->num_segments);
		gss->seg_lstree = palloc0(sizeof(cl_uint *) * (depth + 1));
		for (i=0; i <= depth; i++)
			gss->seg_lstree[i] = palloc0(sizeof(cl_uint) * (1 << i));
		gss->seg_lstree_depth = depth;
		MemoryContextSwitchTo(oldcxt);

		for (i=0, k = (1 << depth); i < k; i++)
			gss->seg_lstree[depth][i] = i;	/* last depth */
		for (i=gss->seg_lstree_depth-1; i >= 0; i--)
		{
			for (j=0, k=(1 << i); j < k; j++)
				gpusort_update_lstree(gss, i, j);
		}
	}
	else
	{
		/*
		 * increment the current position of the last segment and update
		 * the ls-tree for the next tuple.
		 */
		cl_int		last_segid = gss->seg_lstree[0][0];
		cl_int		i, j;

		if (last_segid == INT_MAX)
			return NULL;	/* no more rows to fetch */

		Assert(last_segid >= 0 && last_segid < gss->num_segments);
		gss->seg_curpos[last_segid]++;

		for (i=gss->seg_lstree_depth-1; i >= 0; i--)
		{
			cl_uint		shift = gss->seg_lstree_depth - i;

			j = last_segid >> shift;
			Assert(gss->seg_lstree[i][j] == last_segid);
			gpusort_update_lstree(gss, i, j);
		}
	}
	PERFMON_END(&gss->gts.pfm, gsort.tv_cpu_sort, &tv1, &tv2);

	/*
	 * Fetch the next tuple
	 */
	segid = gss->seg_lstree[0][0];
	if (segid < 0 || segid >= gss->num_segments)
		return NULL;	/* end of the scan */

	pds = gss->seg_slots[segid];
	kresults = gss->seg_results[segid];
	curpos = gss->seg_curpos[segid];
	index = kresults->results[curpos];
	values = KERN_DATA_STORE_VALUES(pds->kds, index);
	isnull = KERN_DATA_STORE_ISNULL(pds->kds, index);

	memcpy(slot->tts_values, values, sizeof(Datum) * natts);
	memcpy(slot->tts_isnull, isnull, sizeof(bool) * natts);
	ExecStoreVirtualTuple(slot);

	PERFMON_END(&gss->gts.pfm, time_materialize, &tv2, &tv3);

	return slot;
}





static void
gpusort_cleanup_cuda_resources(pgstrom_gpusort *pgsort)
{
	if (pgsort->m_gpusort)
		gpuMemFree(&pgsort->task, pgsort->m_gpusort);
	CUDA_EVENT_DESTROY(pgsort, ev_dma_send_start);
	CUDA_EVENT_DESTROY(pgsort, ev_dma_send_stop);
	CUDA_EVENT_DESTROY(pgsort, ev_dma_recv_start);
	CUDA_EVENT_DESTROY(pgsort, ev_dma_recv_stop);

	/* clear the pointers */
	pgsort->kern_proj = NULL;
	pgsort->kern_main = NULL;
	pgsort->m_gpusort = 0UL;
	pgsort->m_kds_in = 0UL;

	/* also release segment if terminator */
	if (pgsort->is_terminator)
	{
		gpusort_segment *segment = pgsort->segment;
		cl_uint		i;

		Assert(segment != NULL);
		if (segment->m_kds_slot)
			gpuMemFree(&pgsort->task, segment->m_kds_slot);

		CUDA_EVENT_DESTROY(segment, ev_setup_segment);
		for (i=0; i < segment->num_chunks; i++)
			CUDA_EVENT_DESTROY(segment, ev_kern_proj[i]);

		segment->m_kds_slot = 0UL;
		segment->m_kresults = 0UL;
	}
}

static void
gpusort_task_release(GpuTask *gtask)
{
	pgstrom_gpusort	   *pgsort = (pgstrom_gpusort *) gtask;

	gpusort_cleanup_cuda_resources(pgsort);

	if (pgsort->pds_in)
	{
		PDS_release(pgsort->pds_in);
		pgsort->pds_in = NULL;
	}

	if (pgsort->segment)
	{
		gpusort_put_segment(pgsort->segment);
		pgsort->segment = NULL;
	}
	pfree(gtask);
}

static bool
gpusort_task_complete(GpuTask *gtask)
{
	GpuSortState	   *gss = (GpuSortState *) gtask->gts;
	pgstrom_gpusort	   *pgsort = (pgstrom_gpusort *) gtask;
	gpusort_segment	   *segment = pgsort->segment;
	pgstrom_perfmon	   *pfm = &gss->gts.pfm;

	if (pfm->enabled)
	{
		pfm->num_tasks++;
		CUDA_EVENT_ELAPSED(pgsort, time_dma_send,
						   pgsort->ev_dma_send_start,
						   pgsort->ev_dma_send_stop,
						   skip);
		if (pgsort->pds_in)
		{
			CUDA_EVENT_ELAPSED(pgsort, gsort.tv_kern_proj,
							   pgsort->ev_dma_send_stop,
							   segment->ev_kern_proj[pgsort->seg_ev_index],
							   skip);
		}

		if (pgsort->is_terminator)
		{
			CUDA_EVENT_ELAPSED(pgsort, gsort.tv_kern_main,
							   segment->ev_kern_proj[pgsort->seg_ev_index],
							   pgsort->ev_dma_recv_start,
							   skip);
			/* kernels launched by dynamic parallel */
			pfm->gsort.num_kern_lsort += pgsort->kern.pfm.num_kern_lsort;
			pfm->gsort.num_kern_ssort += pgsort->kern.pfm.num_kern_ssort;
			pfm->gsort.num_kern_msort += pgsort->kern.pfm.num_kern_msort;
			pfm->gsort.num_kern_fixvar += pgsort->kern.pfm.num_kern_fixvar;
			pfm->gsort.tv_kern_lsort += pgsort->kern.pfm.tv_kern_lsort;
			pfm->gsort.tv_kern_ssort += pgsort->kern.pfm.tv_kern_ssort;
			pfm->gsort.tv_kern_msort += pgsort->kern.pfm.tv_kern_msort;
			pfm->gsort.tv_kern_fixvar += pgsort->kern.pfm.tv_kern_fixvar;
		}
		CUDA_EVENT_ELAPSED(pgsort, time_dma_recv,
						   pgsort->ev_dma_recv_start,
						   pgsort->ev_dma_recv_stop,
						   skip);
	}
skip:
	/*
     * Release device memory and event objects acquired by the task.
     * If task is terminator of the segment, it also releases relevant
     * resources of the segment.
     */
    gpusort_cleanup_cuda_resources(pgsort);

	/* Quick bailout to raise an error if no recoverable */
	if (pgsort->task.kerror.errcode != StromError_Success)
		return true;

	/*
	 * StromError_CpuReCheck informs gpusort_keycomp could not compare
	 * the key variables on GPU side. So, segment needs to be processed
	 * by CPU fallback routine, to construct kern_resultbuf.
	 */
	if (segment->cpu_fallback)
	{
		int		i;

		Assert(pgsort->is_terminator);
		Assert(pgstrom_cpu_fallback_enabled);
		memset(&segment->kresults.kerror, 0, sizeof(kern_errorbuf));
		for (i=0; i < segment->kresults.nitems; i++)
			segment->kresults.results[i] = i;

		gpusort_fallback_quicksort(gss,
								   &segment->kresults,
								   segment->pds_slot->kds,
								   0, segment->kresults.nitems - 1);
		segment->cpu_fallback = false;
	}

	/*
	 * StromError_DataStoreNoSpace implies this gpusort task could not
	 * move all the tuples on kds_in into the kds_slot of the segment
	 * due to lack of the buffer space.
	 * At least, we have to move the remaining tuples to the new segment.
	 * Also, we have to terminate the segment if no terminator task was
	 * not assigned yet.
	 *
	 * We need to pay attention whether further chunks will be generated
	 * any more, or not. If we may have any further stream of chunks
	 * (when !gss->gts.scan_done), it will become a terminator, so no
	 * need to have special handling on termination.
	 *
	 * If we already reached end of the scan, no further rows shall be
	 * supplied. It means the last GpuSort task is terminator, but it
	 * is not certain whether the last segment is really filled up by
	 * the input stream.
	 * So, in case when the last segment still has space and the terminator
	 * task is not launched yet, we can inject this chunk prior to the
	 * segment.
	 */
	if (pgsort->kern.kerror.errcode == StromError_DataStoreNoSpace)
	{
		pgstrom_data_store *pds_in = pgsort->pds_in;
		pgstrom_gpusort	   *pgsort_new;
		cl_uint				valid_nitems;

		/*
		 * No other task shall be attached on the segment that raised
		 * StromError_DataStoreNoSpace error
		 */
		if (gss->curr_segment == segment)
			gss->curr_segment = NULL;

		/* some rows are remained */
		Assert(pds_in->kds->nitems > pgsort->kern.n_loaded);
		valid_nitems = pds_in->kds->nitems - pgsort->kern.n_loaded;
		/* detach PDS from the old task */
		pgsort->pds_in = NULL;

		/*
		 * The retry task may be able to ride on a exising and terminated
		 * segment, if scan is already completed but terminator task is
		 * not launched yet and its segment has enough space.
		 */
		if (gss->gts.scan_done)
		{
			gpusort_segment	   *segment_temp;
			pgstrom_gpusort	   *pgsort_temp;
			dlist_iter			iter;

			SpinLockAcquire(&gss->gts.lock);
			dlist_reverse_foreach (iter, &gss->gts.pending_tasks)
			{
				pgsort_temp = dlist_container(pgstrom_gpusort,
											  task.chain, iter.cur);
				segment_temp = pgsort_temp->segment;
				Assert(segment_temp->has_terminator);

				if (pgsort_temp->is_terminator &&
					segment_temp->num_chunks < segment_temp->max_chunks &&
					(segment_temp->nitems_total +
					 valid_nitems) <= segment_temp->kresults.nrooms)
				{
					SpinLockRelease(&gss->gts.lock);

					/* OK, we can interrupt this segment */
					pgsort_new = (pgstrom_gpusort *)
						gpusort_create_task(gss, pds_in, valid_nitems,
											false, segment_temp);
					/* append it prior to the terminator */
					SpinLockAcquire(&gss->gts.lock);
					dlist_insert_before(&pgsort_temp->task.chain,
										&pgsort_new->task.chain);
					gss->gts.num_pending_tasks++;
					SpinLockRelease(&gss->gts.lock);
					goto shareride_done;
				}
			}
			SpinLockRelease(&gss->gts.lock);
		}

		/*
		 * Elsewhere, we have no chance to have a share ride with other
		 * existing segment, so we will make a new segment, then attach
		 * GpuTask on the tail.
		 */
		pgsort_new = (pgstrom_gpusort *)
			gpusort_create_task(gss, pds_in, valid_nitems,
								gss->gts.scan_done, NULL);
		SpinLockAcquire(&gss->gts.lock);
		dlist_push_tail(&gss->gts.pending_tasks, &pgsort_new->task.chain);
		gss->gts.num_pending_tasks++;
		SpinLockRelease(&gss->gts.lock);
	shareride_done:
		;
	}

	/*
	 * Only final GpuTask shall be backed to the main logic because all the
	 * valuable data is already moved to gpusort_segment, thus, no need to
	 * maintain the pgstrom_gpusort and pgstrom_data_store structure any
	 * more.
	 */
	if (gss->gts.scan_done)
	{
		SpinLockAcquire(&gss->gts.lock);
		if (dlist_is_empty(&gss->gts.running_tasks) &&
			dlist_is_empty(&gss->gts.pending_tasks) &&
			dlist_is_empty(&gss->gts.completed_tasks))
		{
			/* sanity checks */
			Assert(pgsort->is_terminator);
			SpinLockRelease(&gss->gts.lock);
			return true;	/* OK, this task is exactly the last one */
		}
		SpinLockRelease(&gss->gts.lock);
	}
	/*
	 * Other tasks shall be released immediately
	 */
	SpinLockAcquire(&gss->gts.lock);
	dlist_delete(&pgsort->task.tracker);
	memset(&pgsort->task.tracker, 0, sizeof(dlist_node));
	SpinLockRelease(&gss->gts.lock);
	gpusort_task_release(&pgsort->task);

	return false;
}

/*
 * gpusort_task_respond
 */
static void
gpusort_task_respond(CUstream stream, CUresult status, void *private)
{
	pgstrom_gpusort	   *pgsort = (pgstrom_gpusort *) private;
	gpusort_segment	   *segment = pgsort->segment;
	GpuTaskState	   *gts = pgsort->task.gts;

	/* See comments in pgstrom_respond_gpuscan() */
	if (status == CUDA_ERROR_INVALID_CONTEXT || !IsTransactionState())
		return;

	/*
	 * NOTE: The pgsort->kern.kerror informs status of the gpusort_projection.
	 * We can handle only StromError_DataStoreNoSpace error with secondary
	 * trial with new segment.
	 * The kresults->kerror informs status of the gpusort_main. We can handle
	 * only StromError_CpuReCheck error with CPU fallback operation.
	 * Elsewhere, we will raise an error status.
	 */
	if (status == CUDA_SUCCESS)
	{
		if (pgsort->kern.kerror.errcode == StromError_Success ||
			pgsort->kern.kerror.errcode == StromError_DataStoreNoSpace)
		{
			if (!pgsort->is_terminator ||
				segment->kresults.kerror.errcode == StromError_Success)
				memset(&pgsort->task.kerror, 0, sizeof(kern_errorbuf));
			else if (pgstrom_cpu_fallback_enabled &&
					 segment->kresults.kerror.errcode == StromError_CpuReCheck)
			{
				memset(&pgsort->task.kerror, 0, sizeof(kern_errorbuf));
				segment->cpu_fallback = true;
			}
			else
				pgsort->task.kerror = segment->kresults.kerror;
		}
		else
			pgsort->task.kerror = pgsort->kern.kerror;
	}
	else
	{
		pgsort->task.kerror.errcode = status;
		pgsort->task.kerror.kernel = StromKernel_CudaRuntime;
		pgsort->task.kerror.lineno = 0;
	}

	/*
	 * Remove the GpuTask from the running_tasks list, and attach it
	 * on the completed_tasks list again. Note that this routine may
	 * be called by CUDA runtime, prior to attachment of GpuTask on
	 * the running_tasks by cuda_control.c.
	 */
	SpinLockAcquire(&gts->lock);
	if (pgsort->task.chain.prev && pgsort->task.chain.next)
	{
		dlist_delete(&pgsort->task.chain);
		gts->num_running_tasks--;
	}
	if (pgsort->task.kerror.errcode == StromError_Success)
		dlist_push_tail(&gts->completed_tasks, &pgsort->task.chain);
	else
		dlist_push_head(&gts->completed_tasks, &pgsort->task.chain);
	gts->num_completed_tasks++;
	SpinLockRelease(&gts->lock);

	SetLatch(&MyProc->procLatch);
}

static bool
gpusort_setup_segment(GpuSortState *gss, pgstrom_gpusort *pgsort)
{
	gpusort_segment *segment = pgsort->segment;
	pgstrom_perfmon *pfm = &gss->gts.pfm;
	CUresult	rc;

	if (!segment->m_kresults)
	{
		kern_data_store	   *kds_slot = segment->pds_slot->kds;
		kern_resultbuf	   *kresults = &segment->kresults;
		CUdeviceptr			m_kds_slot;
		CUdeviceptr			m_kresults;
		CUevent				ev_setup_segment;
		Size				length;

		length = (GPUMEMALIGN(kds_slot->length) +
				  GPUMEMALIGN(offsetof(kern_resultbuf, results) +
							  sizeof(cl_uint) * kresults->nrooms));
		m_kds_slot = gpuMemAlloc(&pgsort->task, length);
		if (!m_kds_slot)
			return false;	/* retry to enqueue task */
		m_kresults = m_kds_slot + GPUMEMALIGN(kds_slot->length);

		rc = cuEventCreate(&ev_setup_segment, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));

		/*
		 * DMA Send
		 */
		rc = cuMemcpyHtoDAsync(m_kds_slot,
							   kds_slot,
							   kds_slot->length,
							   pgsort->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		pfm->bytes_dma_send += length;
		pfm->num_dma_send++;

		rc = cuMemcpyHtoDAsync(m_kresults,
							   kresults,
							   offsetof(kern_resultbuf, results),
							   pgsort->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		pfm->bytes_dma_send += length;
		pfm->num_dma_send++;

		rc = cuEventRecord(ev_setup_segment, pgsort->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventRecord: %s", errorText(rc));

		segment->m_kds_slot = m_kds_slot;
		segment->m_kresults = m_kresults;
		segment->ev_setup_segment = ev_setup_segment;
	}
	else
	{
		Assert(segment->ev_setup_segment != NULL);
		rc = cuStreamWaitEvent(pgsort->task.cuda_stream,
							   segment->ev_setup_segment,
							   0);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuStreamWaitEvent: %s", errorText(rc));
	}
	return true;
}




static bool
__gpusort_task_process(GpuSortState *gss, pgstrom_gpusort *pgsort)
{
	pgstrom_perfmon	   *pfm = &gss->gts.pfm;
	gpusort_segment	   *segment = pgsort->segment;
	pgstrom_data_store *pds_in = pgsort->pds_in;
	void			   *kern_args[6];
	Size				length;
	size_t				grid_size;
	size_t				block_size;
	CUevent				ev_kern_proj;
	CUresult			rc;


	/*
	 * GPU kernel function lookup
	 */
	rc = cuModuleGetFunction(&pgsort->kern_proj,
							 pgsort->task.cuda_module,
							 "gpusort_projection");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction : %s", errorText(rc));

	rc = cuModuleGetFunction(&pgsort->kern_main,
							 pgsort->task.cuda_module,
							 "gpusort_main");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction : %s", errorText(rc));

	/*
	 * Allocation of the device memory
	 */
	length = GPUMEMALIGN(KERN_GPUSORT_DMASEND_LENGTH(&pgsort->kern));
	if (pds_in)
		length += GPUMEMALIGN(pds_in->kds->length);
	pgsort->m_gpusort = gpuMemAlloc(&pgsort->task, length);
	if (!pgsort->m_gpusort)
		goto out_of_resource;
	if (pds_in)
		pgsort->m_kds_in = pgsort->m_gpusort +
			GPUMEMALIGN(KERN_GPUSORT_DMASEND_LENGTH(&pgsort->kern));
	else
		pgsort->m_kds_in = 0UL;

	/*
	 * creation of event objects, if any
	 */
	CUDA_EVENT_CREATE(pgsort, ev_dma_send_start);
	CUDA_EVENT_CREATE(pgsort, ev_dma_send_stop);
	CUDA_EVENT_CREATE(pgsort, ev_dma_recv_start);
	CUDA_EVENT_CREATE(pgsort, ev_dma_recv_stop);

	/*
	 * OK, enqueue a series of commands
	 */
	CUDA_EVENT_RECORD(pgsort, ev_dma_send_start);

	/* send or sync GpuSort segment buffer */
	if (!gpusort_setup_segment(gss, pgsort))
		goto out_of_resource;

	/* send kern_gpusort */
	length = KERN_GPUSORT_DMASEND_LENGTH(&pgsort->kern);
	rc = cuMemcpyHtoDAsync(pgsort->m_gpusort,
						   &pgsort->kern,
						   length,
						   pgsort->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	pfm->bytes_dma_send += length;
	pfm->num_dma_send++;

	if (!pds_in)
	{
		Assert(pgsort->is_terminator);
		CUDA_EVENT_RECORD(pgsort, ev_dma_send_stop);
	}
	else
	{
		kern_data_store	   *kds_in = pds_in->kds;

		rc = cuMemcpyHtoDAsync(pgsort->m_kds_in,
							   kds_in,
							   kds_in->length,
							   pgsort->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
		pfm->bytes_dma_send += pds_in->kds->length;
		pfm->num_dma_send++;

		CUDA_EVENT_RECORD(pgsort, ev_dma_send_stop);

		/*
		 * KERNEL_FUNCTION(void)
		 * gpusort_projection(kern_gpusort *kgpusort,
		 *                    kern_resultbuf *kresults,
		 *                    kern_data_store *kds_slot,
		 *                    kern_data_store *kds_in)
		 */
		optimal_workgroup_size(&grid_size,
							   &block_size,
							   pgsort->kern_proj,
							   pgsort->task.cuda_device,
							   pds_in->kds->nitems,
							   0, sizeof(cl_uint));
		kern_args[0] = &pgsort->m_gpusort;
		kern_args[1] = &segment->m_kresults;
		kern_args[2] = &segment->m_kds_slot;
		kern_args[3] = &pgsort->m_kds_in;

		rc = cuLaunchKernel(pgsort->kern_proj,
							grid_size, 1, 1,
							block_size, 1, 1,
							sizeof(cl_uint) * block_size,
							pgsort->task.cuda_stream,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
		elog(DEBUG2, "gpusort_projection grid=%zu block=%zu nitems=%zu",
			 grid_size, block_size, (size_t)pds_in->kds->nitems);
		pfm->gsort.num_kern_proj++;
	}

	/* inject an event to synchronize the projection kernel */
	rc = cuEventCreate(&ev_kern_proj, CU_EVENT_DEFAULT);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));
	segment->ev_kern_proj[pgsort->seg_ev_index] = ev_kern_proj;

	rc = cuEventRecord(ev_kern_proj, pgsort->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuEventRecord: %s", errorText(rc));

	if (pgsort->is_terminator)
	{
		cl_uint		i;

		/*
		 * Synchronization of other concurrent gpusort_projection kernels
		 */
		for (i=0; i < segment->num_chunks; i++)
		{
			rc = cuStreamWaitEvent(pgsort->task.cuda_stream,
								   segment->ev_kern_proj[i], 0);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuEventRecord: %s", errorText(rc));
		}

		/*
		 * KERNEL_FUNCTION(void)
		 * gpusort_main(kern_gpusort *kgpusort,
		 *              kern_resultbuf *kresults,
		 *              kern_data_store *kds_slot)
		 */
		kern_args[0] = &pgsort->m_gpusort;
		kern_args[1] = &segment->m_kresults;
		kern_args[2] = &segment->m_kds_slot;

		rc = cuLaunchKernel(pgsort->kern_main,
							1, 1, 1,
							1, 1, 1,
							0,
							pgsort->task.cuda_stream,
							kern_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
		elog(DEBUG2, "gpusort_main grid {1,1,1} block{1,1,1}");
		pfm->gsort.num_kern_main++;
	}

	/*
	 * DMA Recv
	 */
	CUDA_EVENT_RECORD(pgsort, ev_dma_recv_start);
	if (pds_in)
	{
		kern_data_store	   *kds_in = pds_in->kds;

		length = KERN_GPUSORT_DMARECV_LENGTH(&pgsort->kern);
		rc = cuMemcpyDtoHAsync(&pgsort->kern,
							   pgsort->m_gpusort,
							   length,
							   pgsort->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "cuMemcpyDtoHAsync: %s", errorText(rc));
		pfm->bytes_dma_recv += length;
		pfm->num_dma_recv++;

		length = KDS_CALCULATE_ROW_FRONTLEN(kds_in->ncols, kds_in->nitems);
		rc = cuMemcpyDtoHAsync(kds_in,
							   pgsort->m_kds_in,
							   length,
							   pgsort->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "cuMemcpyDtoHAsync: %s", errorText(rc));
		pfm->bytes_dma_recv += length;
		pfm->num_dma_recv++;
	}

	if (pgsort->is_terminator)
	{
		kern_data_store	   *kds_slot = segment->pds_slot->kds;
		kern_resultbuf	   *kresults = &segment->kresults;

		rc = cuMemcpyDtoHAsync(kds_slot,
							   segment->m_kds_slot,
							   kds_slot->length,
							   pgsort->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "cuMemcpyDtoHAsync: %s", errorText(rc));
		pfm->bytes_dma_recv += kds_slot->length;
		pfm->num_dma_recv++;

		length = (offsetof(kern_resultbuf, results) +
				  sizeof(cl_uint) * kresults->nrooms);
		rc = cuMemcpyDtoHAsync(kresults,
							   segment->m_kresults,
							   length,
							   pgsort->task.cuda_stream);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "cuMemcpyDtoHAsync: %s", errorText(rc));
		pfm->bytes_dma_recv += length;
		pfm->num_dma_recv++;
	}
	CUDA_EVENT_RECORD(pgsort, ev_dma_recv_stop);

	/*
	 * Register the callback
	 */
	rc = cuStreamAddCallback(pgsort->task.cuda_stream,
							 gpusort_task_respond,
							 pgsort, 0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuStreamAddCallback: %s", errorText(rc));

	return true;

out_of_resource:
	gpusort_cleanup_cuda_resources(pgsort);
	return false;
}

static bool
gpusort_task_process(GpuTask *gtask)
{
	GpuSortState	   *gss = (GpuSortState *) gtask->gts;
	pgstrom_gpusort	   *pgsort = (pgstrom_gpusort *) gtask;
	CUresult			rc;
	bool				status;

	/* switch CUDA context */
	rc = cuCtxPushCurrent(pgsort->task.cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));
	PG_TRY();
	{
		status = __gpusort_task_process(gss, pgsort);
	}
	PG_CATCH();
	{
		gpusort_cleanup_cuda_resources(pgsort);
		rc = cuCtxPopCurrent(NULL);
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* reset CUDA context */
	rc = cuCtxPopCurrent(NULL);
	if (rc != CUDA_SUCCESS)
		elog(WARNING, "failed on cuCtxPopCurrent: %s", errorText(rc));

	return status;
}



/*
 * gpusort_fallback_quicksort
 *
 * Fallback routine to a particular GpuSort Segment, if GPU device gave up
 * sorting on device side. We assume kds_slot is successfully set up by
 * gpusort_projection() stage. This fallback routine will construct the
 * kern_resultbuf array using CPU quicksort algorithm.
 */
static void
gpusort_fallback_quicksort(GpuSortState *gss,
						   kern_resultbuf *kresults,
						   kern_data_store *kds_slot,
						   cl_int l_bound, cl_int r_bound)
{
	if (l_bound <= r_bound)
	{
		cl_int	l_index = l_bound;
		cl_int	r_index = r_bound;
		cl_int	p_index = kresults->results[(l_index + r_index) / 2];
		Datum  *p_values = (Datum *) KERN_DATA_STORE_VALUES(kds_slot, p_index);
		bool   *p_isnull = (bool *) KERN_DATA_STORE_ISNULL(kds_slot, p_index);

		while (l_index <= r_index)
		{
			while (l_index <= r_bound)
			{
				cl_uint	x_index = kresults->results[l_index];
				Datum  *x_values = KERN_DATA_STORE_VALUES(kds_slot, x_index);
				bool   *x_isnull = KERN_DATA_STORE_ISNULL(kds_slot, x_index);

				if (gpusort_cpu_keycomp(gss,
										x_values, x_isnull,
										p_values, p_isnull) >= 0)
					break;
				l_index++;
			}
			while (r_index >= l_bound)
			{
				cl_uint	y_index = kresults->results[l_index];
				Datum  *y_values = KERN_DATA_STORE_VALUES(kds_slot, y_index);
				bool   *y_isnull = KERN_DATA_STORE_ISNULL(kds_slot, y_index);

				if (gpusort_cpu_keycomp(gss,
										y_values, y_isnull,
										p_values, p_isnull) <= 0)
					break;
				r_index--;
			}

			if (l_index <= r_index)
			{
				cl_int	l_prev = kresults->results[l_index];
				cl_int	r_prev = kresults->results[r_index];

				kresults->results[l_index] = r_prev;
				kresults->results[r_index] = l_prev;
				l_index++;
				r_index--;
			}
		}
		gpusort_fallback_quicksort(gss, kresults, kds_slot, l_bound, r_index);
		gpusort_fallback_quicksort(gss, kresults, kds_slot, l_index, r_bound);
	}
}

/*
 * Entrypoint of GpuSort
 */
void
pgstrom_init_gpusort(void)
{
	/* enable_gpusort parameter */
	DefineCustomBoolVariable("pg_strom.enable_gpusort",
							 "Enables the use of GPU accelerated sorting",
							 NULL,
							 &enable_gpusort,
							 false,
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
	gpusort_exec_methods.MarkPosCustomScan	= gpusort_mark_pos;
	gpusort_exec_methods.RestrPosCustomScan	= gpusort_restore_pos;
	gpusort_exec_methods.ExplainCustomScan	= gpusort_explain;
}
