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
#include "access/xact.h"
#include "catalog/pg_type.h"
#include "commands/dbcommands.h"
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
	long		num_chunks;
	Size		chunk_size;
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
	GpuTask				task;

	/* class of number of chunks - GPU can sort only mc_class == 0 */
	cl_uint				mc_class;

	/*
	 * Index of the sorted tuples
	 */
	dsm_segment		   *litems_dsm;
	dsm_segment		   *ritems_dsm;
	dsm_segment		   *oitems_dsm;
	bool				oitems_pinned;

	/* ------------------------------------- *
	 * Fields to run GPU sorting             *
	 * ------------------------------------- */
	cl_int				chunk_id;		/* id of the chunk */
	CUfunction			kern_prep;		/* gpusort_preparation */
	CUfunction			kern_bitonic_local;	/* gpusort_bitonic_local */
	CUfunction			kern_bitonic_step;	/* gpusort_bitonic_step */
	CUfunction			kern_bitonic_merge;	/* gpusort_bitonic_merge */
	CUfunction			kern_fixup;		/* gpusort_fixup_datastore */
	CUdeviceptr			m_gpusort;
	CUdeviceptr			m_kds;
	CUdeviceptr			m_ktoast;
	CUevent				ev_dma_send_start;
	CUevent				ev_dma_send_stop;
	CUevent				ev_dma_recv_start;
	CUevent				ev_dma_recv_stop;

	/* ------------------------------------- *
	 * Fields to run CPU sorting             *
	 * ------------------------------------- */
	BackgroundWorkerHandle *bgw_handle;
	/* connection info */
	PGPROC			   *backend_proc;
	char			   *database_name;
	/* data chunks to be referenced (BGW context only) */
	Bitmapset		   *chunk_id_map;
	bool				varlena_keys;
	cl_uint				num_chunks;
	kern_data_store	  **kern_toasts;
	kern_data_store	  **kern_chunks;
	/* definition of relation and sorting keys (BGW context only) */
	TupleDesc			tupdesc;
	int					numCols;
	AttrNumber		   *sortColIdx;
	Oid				   *sortOperators;
	Oid				   *collations;
	bool			   *nullsFirst;
} pgstrom_gpusort;

typedef struct
{
	Size			dsm_length;		/* total length of this structure */
	Size			kresults_ofs;	/* offset from data[] */
	/* connection information */	
	PGPROC		   *backend_proc;	/* PGPROC of the coordinator backend */
	Size			database_name;	/* offset from data[] */
	/* IPC stuff */
	volatile bool	bgw_done;	/* flag to inform BGW gets completed */
	struct timeval	tv_bgw_launch;
	struct timeval	tv_sort_start;
	struct timeval	tv_sort_end;
	dsm_handle		litems_dsmhnd;
	dsm_handle		ritems_dsmhnd;
	/* chunks to be sorted  */
	cl_uint			varlena_keys;
	cl_uint			num_chunks;
	Size			kern_chunks;	/* offset from data[] */
	/* definition of relation and sorting keys */
	Size			tupdesc;		/* offset from data[] */
	int				numCols;
	Size			sortColIdx;		/* offset from data[] */
	Size			sortOperators;	/* offset from data[] */
	Size			collations;		/* offset from data[] */
	Size			nullsFirst;		/* offset from data[] */
	char			data[FLEXIBLE_ARRAY_MEMBER];
} pgstrom_flat_gpusort;

#define GPUSORT_GET_KRESULTS(dsmseg)									\
	((kern_resultbuf *)													\
	 (((pgstrom_flat_gpusort *)dsm_segment_address(dsmseg))->data +		\
	  ((pgstrom_flat_gpusort *)dsm_segment_address(dsmseg))->kresults_ofs))

#define MAX_MERGECHUNKS_CLASS		10		/* 1024 chunks are enough large */

typedef struct
{
	GpuTaskState	gts;

	/*
	 * Data store saved in temporary file
	 */
	cl_uint			num_chunks;
	cl_uint			num_chunks_limit;
	pgstrom_data_store **pds_chunks;
	pgstrom_data_store **pds_toasts;
	Size			chunk_size;		/* expected best size for sorting chunk */
	Size			chunk_nrooms;	/* expected nrooms of sorting chunk */

	/* copied from the plan node */
    int				numCols;		/* number of sort-key columns */
    AttrNumber	   *sortColIdx;		/* their indexes in the target list */
    Oid			   *sortOperators;	/* OIDs of operators to sort them by */
	Oid			   *collations;		/* OIDs of collations */
	bool		   *nullsFirst;		/* NULLS FIRST/LAST directions */
	bool			varlena_keys;	/* True, if varlena sorting key exists */
	SortSupportData *ssup_keys;		/* XXX - used by fallback function */

	/* running status */
	char		   *database_name;	/* name of the current database */
	/* chunks already sorted but no pair yet */
	pgstrom_gpusort	*sorted_chunks[MAX_MERGECHUNKS_CLASS];
	bool			sort_done;		/* if true, now ready to fetch records */
	cl_int			cpusort_seqno;	/* seqno of cpusort to launch */

	/* random access capability */
	bool			randomAccess;
	cl_long			markpos_index;

	/* final result */
	HeapTupleData	tuple_buf;		/* temp buffer during scan */
	TupleTableSlot *overflow_slot;
} GpuSortState;

/*
 * declaration of static variables and functions
 */
static CustomScanMethods	gpusort_scan_methods;
static CustomExecMethods	gpusort_exec_methods;
static bool					enable_gpusort;
static bool					debug_force_gpusort;
static int					gpusort_max_workers;

static GpuTask *gpusort_next_chunk(GpuTaskState *gts);
static TupleTableSlot *gpusort_next_tuple(GpuTaskState *gts);
static bool gpusort_task_process(GpuTask *gtask);
static bool gpusort_task_complete(GpuTask *gtask);
static void gpusort_task_release(GpuTask *gtask);
static void gpusort_task_polling(GpuTaskState *gts);
static dsm_segment *form_pgstrom_flat_gpusort_base(GpuSortState *gss,
												   cl_int chunk_id);
static dsm_segment *form_pgstrom_flat_gpusort(GpuSortState *gss,
											  pgstrom_gpusort *l_gpusort,
											  pgstrom_gpusort *r_gpusort);
static pgstrom_gpusort *deform_pgstrom_flat_gpusort(dsm_segment *dsm_seg);
static void gpusort_fallback_quicksort(GpuSortState *gss,
									   pgstrom_gpusort *gpusort);
static void bgw_cpusort_entrypoint(Datum main_arg);

/*
 *
 *
 */
static inline Size
gpusort_devmem_requirement(Size kparams_len, cl_int nattrs, Size nitems,
						   Size chunk_size)
{
	Size	length;
	Size	total_length = kparams_len;

	/* for kern_gpusort + kern_resultbuf */
	length = STROMALIGN(kparams_len) +
		STROMALIGN(offsetof(kern_resultbuf, results[2 * nitems]));
	total_length += GPUMEMALIGN(length);

	/* for ktoast(row) */
	total_length += GPUMEMALIGN(chunk_size);

	/* for kds(slot) */
	length = (STROMALIGN(offsetof(kern_data_store, colmeta[nattrs])) +
			  (LONGALIGN(sizeof(bool) * nattrs) +
			   LONGALIGN(sizeof(Datum) * nattrs)) * nitems);
	total_length += GPUMEMALIGN(length);

	return total_length;
}

/*
 * cost_gpusort
 *
 * cost estimation for GpuSort
 */
#define LOG2(x)		(log(x) / 0.693147180559945)

static void
cost_gpusort(PlannedStmt *pstmt, Sort *sort,
			 Cost *p_startup_cost, Cost *p_total_cost,
			 long *p_num_chunks, Size *p_chunk_size)
{
	Plan	   *outer_plan = outerPlan(sort);
	double		ntuples = outer_plan->plan_rows;
	int			width = outer_plan->plan_width;
	int			nattrs = list_length(outer_plan->targetlist);
	Cost		startup_cost;
	Cost		run_cost;
	Cost		cpu_comp_cost = 2.0 * cpu_operator_cost;
	Cost		gpu_comp_cost = 2.0 * pgstrom_gpu_operator_cost;
	double		nrows_per_chunk;
	double		num_chunks;
	double		chunk_margin = 1.10;
	Size		chunk_size;
	Size		kparams_len;
	Size		total_length;
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

		startup_cost += temp->startup_cost;
		run_cost += temp->total_cost - temp->startup_cost;
	}
	/* Fixed cost to setup/launch GPU kernel */
	startup_cost += pgstrom_gpu_setup_cost;

	/*
	 * Estimate number of chunks and nrows per chunk
	 */
	width = MAXALIGN(offsetof(kern_tupitem, htup) +
					 MAXALIGN(offsetof(HeapTupleHeaderData, t_bits) +
							  sizeof(Oid) +		/* if HEAP_HASOID */
							  BITMAPLEN(nattrs)) +
					 MAXALIGN(width));
	chunk_size = (STROMALIGN(offsetof(kern_data_store, colmeta[nattrs])) +
				  (sizeof(cl_uint) + width) * ntuples * chunk_margin);

	/*
	 * Does it suitable for device memory limitation?
	 */
	kparams_len = 1024;
	total_length = gpusort_devmem_requirement(kparams_len,
											  nattrs,
											  (Size)ntuples,
											  chunk_size);
	if (gpuMemMaxAllocSize() > total_length)
	{
		if (chunk_size < pgstrom_chunk_size())
			chunk_size = pgstrom_chunk_size();
		nrows_per_chunk = ntuples;
		num_chunks = 1.0;
	}
	else
	{
		nrows_per_chunk =
			(double)(gpuMemMaxAllocSize()
					 /* for alignment */
					 - 3 * GPUMEMALIGN_LEN
					 /* for kern_gpusort */
					 - STROMALIGN(kparams_len)
					 - STROMALIGN(offsetof(kern_resultbuf, results[0]))
					 /* for kern_data_store (row) */
					 - STROMALIGN(offsetof(kern_data_store, colmeta[nattrs]))
					 /* for kern_data_store (slot) */
					 - STROMALIGN(offsetof(kern_data_store, colmeta[nattrs]))
				) / (double)(2 * sizeof(cl_uint) +	/* kern_resultbuf */
							 sizeof(cl_uint) + width +	/* kds(row) */
							 LONGALIGN(sizeof(bool) * nattrs) +
							 LONGALIGN(sizeof(Datum) * nattrs));
		if (chunk_margin > 1.0)
			nrows_per_chunk /= chunk_margin;

		chunk_size = STROMALIGN(offsetof(kern_data_store, colmeta[nattrs])) +
			(sizeof(cl_uint) + width) * nrows_per_chunk * chunk_margin;
		num_chunks = Max(1.0, floor(ntuples / nrows_per_chunk + 0.9999));
	}

	/*
	 * We'll use bitonic sorting logic on GPU device.
	 * It's cost is N * Log2(N)
	 */
	startup_cost += num_chunks *
		gpu_comp_cost * nrows_per_chunk * LOG2(nrows_per_chunk);

	/*
	 * We'll also use CPU based N-way merge sort, if # of chunks > 1.
	 * It is usually expensive, so we like to avoid it as long as we can.
	 */
	if (num_chunks > 1.0)
		startup_cost += cpu_comp_cost * nrows_per_chunk *
			num_chunks * (num_chunks - 1.0);

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

/*
 * make a projection function
 */
static void
gpusort_projection_addcase(StringInfo body,
						   AttrNumber resno,
						   Oid type_oid,
						   int type_len,
						   bool type_byval,
						   bool is_sortkey)
{
	if (body->len == 0)
		appendStringInfo(body, "  void *datum;\n");

	if (type_byval)
	{
		const char *type_cast;

		Assert(type_len > 0);

		if (type_len == sizeof(cl_uchar))
			type_cast = "cl_uchar";
		else if (type_len == sizeof(cl_ushort))
			type_cast = "cl_ushort";
		else if (type_len == sizeof(cl_uint))
			type_cast = "cl_uint";
		else if (type_len == sizeof(cl_ulong))
			type_cast = "cl_ulong";
		else
			elog(ERROR, "unexpected length of data type");

		appendStringInfo(
			body,
			"\n"
			"  /* %s %d projection */\n"
			"  datum = kern_get_datum_tuple(ktoast->colmeta,htup,%d);\n"
			"  if (!datum)\n"
			"    ts_isnull[%d] = true;\n"
			"  else\n"
			"  {\n"
			"    ts_isnull[%d] = false;\n"
			"    ts_values[%d] = *((%s *) datum);\n"
			"  }\n",
			is_sortkey ? "sortkey" : "attribute",
			resno,
			resno - 1,
			resno - 1,
			resno - 1,
			resno - 1,
			type_cast);
	}
	else if (type_oid == NUMERICOID && is_sortkey)
	{
		/*
		 * NUMERIC data type has internal data format. So, it needs to
		 * be transformed GPU accessable representation.
		 */
		appendStringInfo(
			body,
			"\n"
			"  /* sortkey %d projection */\n"
			"  datum = kern_get_datum_tuple(ktoast->colmeta,htup,%d);\n"
			"  if (!datum)\n"
			"    ts_isnull[%d] = true;\n"
			"  else\n"
			"  {\n"
			"    pg_numeric_t temp =\n"
			"      pg_numeric_from_varlena(kcxt, (varlena *) datum);\n"
			"    ts_isnull[%d] = temp.isnull;\n"
			"    ts_values[%d] = temp.value;\n"
			"  }\n",
			resno,
			resno - 1,
			resno - 1,
			resno - 1,
			resno - 1);
	}
	else
	{
		/*
		 * Elsewhere, variables are device accessible pointer; either
		 * of varlena datum or fixed-length variables with !type_byval.
		 */
		appendStringInfo(
			body,
			"\n"
			"  /* attribute %d projection */\n"
			"  datum = kern_get_datum_tuple(ktoast->colmeta,htup,%d);\n"
			"  if (!datum)\n"
			"    ts_isnull[%d] = true;\n"
			"  else\n"
			"  {\n"
			"    ts_isnull[%d] = false;\n"
			"    ts_values[%d] = PointerGetDatum(datum);\n"
			"  }\n",
			resno,
			resno - 1,
			resno - 1,
			resno - 1,
			resno - 1);
	}
}

static void
gpusort_fixupvar_addcase(StringInfo body,
						 AttrNumber resno,
						 Oid type_oid,
						 int type_len,
						 bool type_byval,
						 bool is_sortkey)
{
	if (type_byval)
		return;

	if (body->len == 0)
		appendStringInfo(body, "  void *datum;\n");

   	appendStringInfo(
		body,
		"\n"
		"  /* varlena %s %d fixup */\n"
		"  datum = kern_get_datum_tuple(ktoast->colmeta,htup,%d);\n"
		"  if (!datum)\n"
		"    ts_isnull[%d] = true;\n"
		"  else\n"
		"  {\n"
		"    ts_isnull[%d] = false;\n"
		"    ts_values[%d] = (Datum)((hostptr_t)datum -\n"
		"                            (hostptr_t)&ktoast->hostptr +\n"
		"                            ktoast->hostptr);\n"
		"  }\n",
		is_sortkey ? "sortkey" : "attribute",
		resno,
		resno - 1,
		resno - 1,
		resno - 1,
		resno - 1);
}

static char *
pgstrom_gpusort_codegen(Sort *sort, codegen_context *context)
{
	StringInfoData	pj_body;
	StringInfoData	fv_body;
	StringInfoData	kc_decl;
	StringInfoData	kc_body;
	StringInfoData	result;
	ListCell	   *cell;
	int				i;

	initStringInfo(&pj_body);
	initStringInfo(&fv_body);
	initStringInfo(&kc_decl);
	initStringInfo(&kc_body);
	initStringInfo(&result);

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

		/*
		 * reference to X-variable / Y-variable
		 *
		 * Because KDS has tuple-slot format, colidx should be index
		 * from the sortkyes array, not resno on host-side.
		 */
		appendStringInfo(
			&kc_decl,
			"  pg_%s_t KVAR_X%d;\n"
			"  pg_%s_t KVAR_Y%d;\n",
			dtype->type_name, i+1,
			dtype->type_name, i+1);

		/* logic to compare */
		appendStringInfo(
			&kc_body,
			"  /* sort key comparison on the resource %d */\n"
			"  KVAR_X%d = pg_%s_vref(kds,kcxt,%d,x_index);\n"
			"  KVAR_Y%d = pg_%s_vref(kds,kcxt,%d,y_index);\n"
			"  if (!KVAR_X%d.isnull && !KVAR_Y%d.isnull)\n"
			"  {\n"
			"    comp = pgfn_%s(kcxt, KVAR_X%d, KVAR_Y%d);\n"
			"    if (comp.value != 0)\n"
			"      return %s;\n"
			"  }\n"
			"  else if (KVAR_X%d.isnull && !KVAR_Y%d.isnull)\n"
			"    return %d;\n"
			"  else if (!KVAR_X%d.isnull && KVAR_Y%d.isnull)\n"
			"    return %d;\n",
			colidx,
			i+1, dtype->type_name, colidx-1,
			i+1, dtype->type_name, colidx-1,
			i+1, i+1,
			dfunc->func_devname, i+1, i+1,
			is_reverse ? "-comp.value" : "comp.value",
			i+1, i+1, null_first ? -1 : 1,
			i+1, i+1, null_first ? 1 : -1);
	}

	/*
	 * Make projection / fixup-variable code
	 */
	foreach (cell, sort->plan.targetlist)
	{
		TargetEntry	   *tle = lfirst(cell);
		Oid				type_oid = exprType((Node *) tle->expr);
		int				type_len = get_typlen(type_oid);
		bool			type_byval = get_typbyval(type_oid);
		bool			is_sortkey = false;

		for (i=0; i < sort->numCols; i++)
		{
			if (tle->resno == sort->sortColIdx[i])
			{
				is_sortkey = true;
				break;
			}
		}
		gpusort_projection_addcase(&pj_body, tle->resno, type_oid,
								   type_len, type_byval, is_sortkey);
		gpusort_fixupvar_addcase(&fv_body, tle->resno, type_oid,
								 type_len, type_byval, is_sortkey);
	}

	/* functions declarations */
	pgstrom_codegen_func_declarations(&result, context);

	/* make a projection function */
	appendStringInfo(
		&result,
		"STATIC_FUNCTION(void)\n"
		"gpusort_projection(kern_context *kcxt,\n"
		"                   Datum *ts_values,\n"
		"                   cl_char *ts_isnull,\n"
		"                   kern_data_store *ktoast,\n"
		"                   HeapTupleHeaderData *htup)\n"
		"{\n"
		"%s"
		"}\n\n",
		pj_body.data);
	/* make a fixup-var function */
	appendStringInfo(
        &result,
		"STATIC_FUNCTION(void)\n"
		"gpusort_fixup_variables(kern_context *kcxt,\n"
		"                        Datum *ts_values,\n"
		"                        cl_char *ts_isnull,\n"
		"                        kern_data_store *ktoast,\n"
		"                        HeapTupleHeaderData *htup)\n"
		"{\n"
		"%s"
		"}\n\n",
		fv_body.data);

	/* make a comparison function */
	appendStringInfo(
		&result,
		"STATIC_FUNCTION(cl_int)\n"
		"gpusort_keycomp(kern_context *kcxt,\n"
		"                kern_data_store *kds,\n"
		"                kern_data_store *ktoast,\n"
		"                size_t x_index,\n"
		"                size_t y_index)\n"
		"{\n"
		"%s"		/* variables declaration */
		"  pg_int4_t comp;\n"
		"\n"
		"%s"		/* comparison body */
		"  return 0;\n"
		"}\n",
		kc_decl.data,
		kc_body.data);

	pfree(pj_body.data);
	pfree(kc_decl.data);
	pfree(kc_body.data);

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
		if ((dtype->type_flags & DEVTYPE_IS_VARLENA) != 0)
			varlena_keys = true;
	}

	/*
	 * OK, cost estimation with GpuSort
	 */
	cost_gpusort(pstmt, sort,
				 &startup_cost, &total_cost,
				 &num_chunks, &chunk_size);

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
	/* informs our preference to fetch tuples */
	if (IsA(subplan, CustomScan))
		((CustomScan *) subplan)->flags |= CUSTOMPATH_PREFERE_ROW_FORMAT;
	outerPlan(cscan) = subplan;
	cscan->scan.plan.initPlan = sort->plan.initPlan;

	pgstrom_init_codegen_context(&context);
	gs_info.startup_cost = startup_cost;
	gs_info.total_cost = total_cost;
	gs_info.kern_source = pgstrom_gpusort_codegen(sort, &context);
	gs_info.extra_flags = context.extra_flags | DEVKERNEL_NEEDS_GPUSORT;
	gs_info.used_params = context.used_params;
	gs_info.num_chunks = num_chunks;
	gs_info.chunk_size = chunk_size;
	gs_info.numCols = sort->numCols;
	gs_info.sortColIdx = sort->sortColIdx;
	gs_info.sortOperators = sort->sortOperators;
	gs_info.collations = sort->collations;
	gs_info.nullsFirst = sort->nullsFirst;
	gs_info.varlena_keys = varlena_keys;
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

static Node *
gpusort_create_scan_state(CustomScan *cscan)
{
	GpuSortState   *gss = palloc0(sizeof(GpuSortState));

	/* Set tag and executor callbacks */
    NodeSetTag(gss, T_CustomScanState);
    gss->gts.css.flags = cscan->flags;
    gss->gts.css.methods = &gpusort_exec_methods;

	return (Node *) gss;
}

static void
gpusort_begin(CustomScanState *node, EState *estate, int eflags)
{
	GpuContext	   *gcontext = NULL;
	GpuSortState   *gss = (GpuSortState *) node;
	CustomScan	   *cscan = (CustomScan *) node->ss.ps.plan;
	GpuSortInfo	   *gs_info = deform_gpusort_info(cscan);
	TupleDesc		tupdesc;

	/* activate GpuContext for device execution */
	if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
		gcontext = pgstrom_get_gpucontext();
	/* common GpuTaskState setup */
	pgstrom_init_gputaskstate(gcontext, &gss->gts);
	gss->gts.cb_task_process = gpusort_task_process;
	gss->gts.cb_task_complete = gpusort_task_complete;
	gss->gts.cb_task_release = gpusort_task_release;
	gss->gts.cb_task_polling = gpusort_task_polling;
	gss->gts.cb_next_chunk = gpusort_next_chunk;
	gss->gts.cb_next_tuple = gpusort_next_tuple;
	/* re-initialization of scan-descriptor and projection-info */
	tupdesc = ExecCleanTypeFromTL(cscan->custom_scan_tlist, false);
	ExecAssignScanType(&gss->gts.css.ss, tupdesc);
	ExecAssignScanProjectionInfoWithVarno(&gss->gts.css.ss, INDEX_VAR);

	/* Like built-in Sort node doing, we shall provide random access
	 * capability to the sort output, including backward scan or
	 * mark/restore. We also prefer to materialize the sort output
	 * if we might be called on to rewind and replay it many times.
	 */
	gss->randomAccess = (eflags & (EXEC_FLAG_REWIND |
								   EXEC_FLAG_BACKWARD |
								   EXEC_FLAG_MARK)) != 0;
	eflags &= ~(EXEC_FLAG_REWIND | EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK);
	gss->markpos_index = -1;

	/* initialize child exec node */
	outerPlanState(gss) = ExecInitNode(outerPlan(cscan), estate, eflags);

	/* for GPU bitonic sorting */
	pgstrom_assign_cuda_program(&gss->gts,
								gs_info->used_params,
								gs_info->kern_source,
								gs_info->extra_flags);
	if ((eflags & EXEC_FLAG_EXPLAIN_ONLY) == 0)
		pgstrom_preload_cuda_program(&gss->gts);

	/* array for data-stores */
	gss->num_chunks = 0;
	gss->num_chunks_limit = gs_info->num_chunks + 10;
	gss->pds_chunks = palloc0(sizeof(pgstrom_data_store *) *
							  gss->num_chunks_limit);
	gss->pds_toasts = palloc0(sizeof(pgstrom_data_store *) *
							  gss->num_chunks_limit);
	gss->chunk_size = gs_info->chunk_size;
	gss->chunk_nrooms = (Size)(outerPlan(cscan)->plan_rows /
							   (double)gs_info->num_chunks);
	/* sorting keys */
	gss->numCols = gs_info->numCols;
	gss->sortColIdx = gs_info->sortColIdx;
	gss->sortOperators = gs_info->sortOperators;
	gss->collations = gs_info->collations;
	gss->nullsFirst = gs_info->nullsFirst;
	gss->varlena_keys = gs_info->varlena_keys;
	gss->ssup_keys = NULL;	/* to be initialized on demand */

	/* running status */
	gss->database_name = get_database_name(MyDatabaseId);
	memset(gss->sorted_chunks, 0, sizeof(gss->sorted_chunks));
	gss->sort_done = false;
	gss->cpusort_seqno = 0;
	gss->overflow_slot = NULL;
}

static TupleTableSlot *
gpusort_exec(CustomScanState *node)
{
	return pgstrom_exec_gputask((GpuTaskState *) node);
}

static void
gpusort_end(CustomScanState *node)
{
	GpuSortState   *gss = (GpuSortState *) node;

#ifdef USE_ASSERT_CHECKING
	{
		int				i;

		for (i=0; i < MAX_MERGECHUNKS_CLASS; i++)
			Assert(!gss->sorted_chunks[i]);
	}
#endif
	/*
	 * Cleanup and relase any concurrent tasks
	 * (including pgstrom_data_store)
	 */
	pgstrom_release_gputaskstate(&gss->gts);

	/* Clean up subtree */
    ExecEndNode(outerPlanState(node));
}

static void
gpusort_rescan(CustomScanState *node)
{
	GpuSortState   *gss = (GpuSortState *) node;

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
		Size	length = sizeof(pgstrom_data_store *) * gss->num_chunks_limit;

		/* cleanup and release any concurrent tasks */
		pgstrom_cleanup_gputaskstate(&gss->gts);
		gss->sort_done = false;
		memset(gss->sorted_chunks, 0, sizeof(gss->sorted_chunks));
		memset(gss->pds_chunks, 0, length);
		memset(gss->pds_toasts, 0, length);
		gss->num_chunks = 0;

		/*
		 * if chgParam of subnode is not null then plan will be re-scanned by
		 * first ExecProcNode.
		 */
    }
    else
	{
		/* otherwise, just rewind the pointer */
		gss->gts.curr_index = 0;
	}
}

static void
gpusort_mark_pos(CustomScanState *node)
{
	GpuSortState   *gss = (GpuSortState *) node;

	if (gss->sort_done)
	{
		gss->markpos_index = gss->gts.curr_index;
	}
}

static void
gpusort_restore_pos(CustomScanState *node)
{
	GpuSortState   *gss = (GpuSortState *) node;

	if (gss->sort_done)
	{
		Assert(gss->markpos_index >= 0);
		gss->gts.curr_index = gss->markpos_index;
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
	if (es->analyze && gss->sort_done)
	{
		const char *sort_method;
		const char *sort_storage;
		char		sort_resource[128];
		Size		total_consumption = 0UL;

		if (gss->num_chunks > 1)
			sort_method = "GPU/Bitonic + CPU/Merge";
		else
			sort_method = "GPU/Bitonic";

		for (i=0; i < gss->num_chunks; i++)
		{
			pgstrom_data_store *pds = gss->pds_chunks[i];
			pgstrom_data_store *ptoast = gss->pds_toasts[i];

			total_consumption += TYPEALIGN(BLCKSZ, pds->kds_length);
			total_consumption += TYPEALIGN(BLCKSZ, ptoast->kds_length);
		}
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
	pgstrom_explain_gputaskstate(&gss->gts, es);
}

static GpuTask *
gpusort_next_chunk(GpuTaskState *gts)
{
	GpuContext		   *gcontext = gts->gcontext;
	GpuSortState	   *gss = (GpuSortState *) gts;
	TupleDesc			tupdesc = GTS_GET_SCAN_TUPDESC(gts);
	pgstrom_gpusort	   *gpusort = NULL;
	pgstrom_data_store *pds = NULL;
	pgstrom_data_store *ptoast = NULL;
	TupleTableSlot	   *slot;
	dsm_segment		   *oitems_dsm;
	cl_uint				nitems;
	Size				length;
	cl_int				chunk_id = -1;
	struct timeval		tv1, tv2;

	PERFMON_BEGIN(&gts->pfm_accum, &tv1);

	/*
	 * Load tuples from the underlying plan node
	 */
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
				gss->gts.scan_done = true;
				slot = NULL;
				break;
			}
		}
		Assert(!TupIsNull(slot));

		/* Makes a sorting chunk on the first tuple */
		if (!ptoast)
		{
			ptoast = pgstrom_create_data_store_row(gcontext,
												   tupdesc,
												   gss->chunk_size,
												   true);
			ptoast->kds->nrooms = (cl_uint)gss->chunk_nrooms;
		}

		/* Insert this tuple to the data store */
		if (!pgstrom_data_store_insert_tuple(ptoast, slot))
		{
			gss->overflow_slot = slot;

			/* Even if insertion of the tuple would be unavailable,
			 * we try to expand the ptoast as long as the expected
			 * length of kds + ktoast is less than max allocatable
			 * device memory.
			 */
			length = gpusort_devmem_requirement(gss->gts.kern_params->length,
												tupdesc->natts,
												2 * ptoast->kds->nitems,
												2 * gss->chunk_size);
			if (length > gpuMemMaxAllocSize())
				break;
			/* OK, expand it and try again */
			gss->chunk_size += gss->chunk_size;
			pgstrom_expand_data_store(gcontext, ptoast, gss->chunk_size, 0);
			ptoast->kds->nrooms = 2 * ptoast->kds->nitems;
		}
	}
	/* Did we read any tuples? */
	if (!ptoast)
	{
		PERFMON_END(&gts->pfm_accum, time_outer_load, &tv1, &tv2);
		return NULL;
	}
	/* Expand the backend file of the data chunk */
	nitems = ptoast->kds->nitems;
	pds = pgstrom_create_data_store_slot(gcontext, tupdesc,
										 nitems, true, 0, ptoast);
	/* Save this chunk on the global array */
	chunk_id = gss->num_chunks++;
	if (chunk_id >= gss->num_chunks_limit)
	{
		cl_uint		new_limit = 2 * gss->num_chunks_limit;

		/* expand the array twice */
		gss->pds_chunks = repalloc(gss->pds_chunks,
								   sizeof(pgstrom_data_store *) * new_limit);
		gss->pds_toasts = repalloc(gss->pds_toasts,
								   sizeof(pgstrom_data_store *) * new_limit);
		gss->num_chunks_limit = new_limit;
	}
	gss->pds_chunks[chunk_id] = pds;
	gss->pds_toasts[chunk_id] = ptoast;

	/* Make a shared memory segment for kern_resultbuf */
	oitems_dsm = form_pgstrom_flat_gpusort_base(gss, chunk_id);

	/* Make a gpusort, based on pds_row, pds_slot and kresults */
	gpusort = MemoryContextAllocZero(gcontext->memcxt,
									 sizeof(pgstrom_gpusort));
	/* initialize GpuTask object */
	pgstrom_init_gputask(&gss->gts, &gpusort->task);
	gpusort->mc_class = 0;
	gpusort->litems_dsm = NULL;
	gpusort->ritems_dsm = NULL;
	gpusort->oitems_dsm = oitems_dsm;
	gpusort->chunk_id = chunk_id;
	gpusort->chunk_id_map = bms_make_singleton(chunk_id);
	PERFMON_END(&gts->pfm_accum, time_outer_load, &tv1, &tv2);

	return &gpusort->task;
}

static TupleTableSlot *
gpusort_next_tuple(GpuTaskState *gts)
{
	GpuSortState	   *gss = (GpuSortState *) gts;
	pgstrom_gpusort	   *gpusort = (pgstrom_gpusort *) gss->gts.curr_task;
	TupleTableSlot	   *slot = gss->gts.css.ss.ps.ps_ResultTupleSlot;
	kern_resultbuf	   *kresults;
	pgstrom_data_store *pds;

	/* Does outer relation has any rows to read? */
	if (!gpusort)
		return NULL;
	Assert(gss->sort_done);

	kresults = GPUSORT_GET_KRESULTS(gpusort->oitems_dsm);
	Assert(kresults->nrels == 2);	/* a pair of chunk_id and item_id */

	if (gss->gts.curr_index < kresults->nitems)
	{
		cl_long		index = 2 * gss->gts.curr_index++;
		cl_int		chunk_id = kresults->results[index];
		cl_int		item_id = kresults->results[index + 1];

		ExecClearTuple(slot);

		if (chunk_id >= gss->num_chunks || !gss->pds_chunks[chunk_id])
			elog(ERROR, "Bug? data-store of GpuSort missing (chunk-id: %d)",
				 chunk_id);
		if ((gss->gts.css.flags & CUSTOMPATH_PREFERE_ROW_FORMAT) == 0)
			pds = gss->pds_chunks[chunk_id];
		else
			pds = gss->pds_toasts[chunk_id];

		if (pgstrom_fetch_data_store(slot, pds, item_id, &gss->tuple_buf))
			return slot;
		elog(ERROR, "Bug? failed to fetch chunk_id=%d item_id=%d",
			 chunk_id, item_id);
	}
	return NULL;
}





static void
gpusort_cleanup_cuda_resources(pgstrom_gpusort *gpusort)
{
	if (gpusort->m_gpusort)
		gpuMemFree(&gpusort->task, gpusort->m_gpusort);
	if (gpusort->oitems_pinned)
	{
		CUresult		rc
			= cuMemHostUnregister(dsm_segment_address(gpusort->oitems_dsm));
		if (rc != CUDA_SUCCESS)
			elog(WARNING, "failed on cuMemHostUnregister: %s", errorText(rc));
	}
	CUDA_EVENT_DESTROY(gpusort, ev_dma_send_start);
	CUDA_EVENT_DESTROY(gpusort, ev_dma_send_stop);
	CUDA_EVENT_DESTROY(gpusort, ev_dma_recv_start);
	CUDA_EVENT_DESTROY(gpusort, ev_dma_recv_stop);

	/* clear the pointers */
	gpusort->kern_prep = NULL;
	gpusort->kern_bitonic_local = NULL;
	gpusort->kern_bitonic_step = NULL;
	gpusort->kern_bitonic_merge = NULL;
	gpusort->kern_fixup = NULL;
	gpusort->m_gpusort = 0UL;
	gpusort->m_kds = 0UL;
	gpusort->m_ktoast = 0UL;
	gpusort->oitems_pinned = false;
}

static void
gpusort_task_release(GpuTask *gtask)
{
	pgstrom_gpusort	   *gpusort = (pgstrom_gpusort *) gtask;

	if (gpusort->litems_dsm)
		dsm_detach(gpusort->litems_dsm);
	if (gpusort->ritems_dsm)
		dsm_detach(gpusort->ritems_dsm);
	if (gpusort->oitems_dsm)
		dsm_detach(gpusort->oitems_dsm);
	if (gpusort->bgw_handle)
		pfree(gpusort->bgw_handle);

	gpusort_cleanup_cuda_resources(gpusort);

	pfree(gtask);
}

/********/
static pgstrom_gpusort *
gpusort_merge_chunks(GpuSortState *gss, pgstrom_gpusort *gpusort_1)
{
	pgstrom_gpusort	   *gpusort_2;
	dsm_segment		   *dsm_seg;
	cl_uint				i, n, x, y;
	cl_uint				mc_class = gpusort_1->mc_class;

	/*
	 * In case of no buddy at this moment, we have to wait for the next
	 * chunk to be merged. Or, this chunk might be the final result.
	 */
	Assert(mc_class < MAX_MERGECHUNKS_CLASS);
	if (!gss->sorted_chunks[mc_class])
	{
		dlist_iter		iter;
		pgstrom_gpusort *temp;
		bool			has_candidate;

		/* once cpusort_1 is put as partially sorted chunk */
		gss->sorted_chunks[mc_class] = gpusort_1;

		/*
		 * Unless scan of outer relation does not completed, we try to
		 * keep merging chunks with same size.
		 */
		if (!gss->gts.scan_done)
			return NULL;

		/*
		 * Once we forget the given cpusort_1, and find out the smallest
		 * one that is now waiting for merging.
		 */
		for (i=0; i < MAX_MERGECHUNKS_CLASS; i++)
		{
			if (gss->sorted_chunks[i])
			{
				gpusort_1 = gss->sorted_chunks[i];
				Assert(gpusort_1->mc_class == i);
				break;
			}
		}
		Assert(i < MAX_MERGECHUNKS_CLASS);
		mc_class = gpusort_1->mc_class;

		/*
		 * If we have any running or pending chunks with same merge-chunk
		 * class, it may be a good candidate to merge.
		 */
		has_candidate = false;
		SpinLockAcquire(&gss->gts.lock);
		dlist_foreach (iter, &gss->gts.running_tasks)
		{
			temp = dlist_container(pgstrom_gpusort, task.chain, iter.cur);
			if (temp->mc_class <= mc_class)
			{
				has_candidate = true;
				goto out;
			}
		}
		dlist_foreach (iter, &gss->gts.pending_tasks)
		{
			temp = dlist_container(pgstrom_gpusort, task.chain, iter.cur);
			if (temp->mc_class == mc_class)
			{
				has_candidate = true;
				goto out;
			}
		}
	out:
		SpinLockRelease(&gss->gts.lock);

		/* wait until smaller chunk gets sorted, if any */
		if (has_candidate)
			return NULL;

		/*
		 * Elsewhere, picks up a pair of smallest two chunks that are
		 * already sorted, to merge them on next step.
		 */
		gpusort_2 = NULL;
		for (i=gpusort_1->mc_class + 1; i < MAX_MERGECHUNKS_CLASS; i++)
		{
			if (gss->sorted_chunks[i])
			{
				gpusort_2 = gss->sorted_chunks[i];
				Assert(gpusort_2->mc_class == i);
				break;
			}
		}
		/* no merginable pair found? */
		if (!gpusort_2)
			return NULL;
		/* OK, let's merge this two chunks */
		gss->sorted_chunks[gpusort_1->mc_class] = NULL;
		gss->sorted_chunks[gpusort_2->mc_class] = NULL;
	}
	else
	{
		gpusort_2 = gss->sorted_chunks[mc_class];
		gss->sorted_chunks[mc_class] = NULL;
	}
	Assert(gpusort_2->oitems_dsm != NULL
		   && !gpusort_2->ritems_dsm
		   && !gpusort_2->litems_dsm);
	dsm_seg = form_pgstrom_flat_gpusort(gss, gpusort_1, gpusort_2);
	gpusort_1->litems_dsm = gpusort_1->oitems_dsm;
	gpusort_1->ritems_dsm = gpusort_2->oitems_dsm;
	gpusort_1->oitems_dsm = dsm_seg;
	gpusort_1->bgw_handle = NULL;
	gpusort_2->oitems_dsm = NULL;	/* clear it to avoid double free */
	x = bms_num_members(gpusort_1->chunk_id_map);
	y = bms_num_members(gpusort_2->chunk_id_map);
	gpusort_1->chunk_id_map = bms_union(gpusort_1->chunk_id_map,
										gpusort_2->chunk_id_map);
	n = bms_num_members(gpusort_1->chunk_id_map);
	gpusort_1->mc_class = get_next_log2(n);
	gpusort_1->mc_class = Min(gpusort_1->mc_class, MAX_MERGECHUNKS_CLASS - 1);
	elog(DEBUG1, "CpuSort merge: %d + %d => %d (class: %d)", x, y, n,
		 gpusort_1->mc_class);
	Assert(gpusort_1->mc_class > 0);
	/* release either of them */
	SpinLockAcquire(&gss->gts.lock);
	dlist_delete(&gpusort_2->task.tracker);
	SpinLockRelease(&gss->gts.lock);
	gpusort_task_release(&gpusort_2->task);

	/* mark not to assign cuda_stream again */
	gpusort_1->task.no_cuda_setup = true;
	pgstrom_cleanup_gputask_cuda_resources(&gpusort_1->task);

	return gpusort_1;
}	

static bool
gpusort_task_complete(GpuTask *gtask)
{
	GpuSortState	   *gss = (GpuSortState *) gtask->gts;
	pgstrom_gpusort	   *gpusort = (pgstrom_gpusort *) gtask;
	pgstrom_gpusort	   *newsort;

	if (gpusort->mc_class == 0)
	{
		if (gpusort->task.pfm.enabled)
		{
			CUDA_EVENT_ELAPSED(gpusort, time_dma_send,
							   ev_dma_send_start,
							   ev_dma_send_stop);
			CUDA_EVENT_ELAPSED(gpusort, time_gpu_sort,
							   ev_dma_send_stop,
							   ev_dma_recv_start);
			CUDA_EVENT_ELAPSED(gpusort, time_dma_recv,
							   ev_dma_recv_start,
							   ev_dma_recv_stop);
			pgstrom_accum_perfmon(&gss->gts.pfm_accum, &gpusort->task.pfm);
		}
		gpusort_cleanup_cuda_resources(gpusort);

		if (gpusort->task.kerror.errcode == StromError_CpuReCheck)
		{
			gpusort_fallback_quicksort(gss, gpusort);
			gpusort->task.kerror.errcode = StromError_Success;
		}
	}
	else
	{
		pgstrom_flat_gpusort   *pfg = (pgstrom_flat_gpusort *)
			dsm_segment_address(gpusort->oitems_dsm);

		if (gpusort->task.pfm.enabled)
		{
			struct timeval	tv_curr;

			gettimeofday(&tv_curr, NULL);

			gss->gts.pfm_accum.num_cpu_sort++;
			PERFMON_END(&gss->gts.pfm_accum, time_bgw_sync,
						&pfg->tv_bgw_launch, &pfg->tv_sort_start);
			PERFMON_END(&gss->gts.pfm_accum, time_cpu_sort,
						&pfg->tv_sort_start, &pfg->tv_sort_end);
			PERFMON_END(&gss->gts.pfm_accum, time_bgw_sync,
						&pfg->tv_sort_end, &tv_curr);
		}
	}

	/* ritems and litems are no longer referenced */
	if (gpusort->litems_dsm)
	{
		dsm_detach(gpusort->litems_dsm);
		gpusort->litems_dsm = NULL;
	}
	if (gpusort->ritems_dsm)
	{
		dsm_detach(gpusort->ritems_dsm);
		gpusort->ritems_dsm = NULL;
	}

	/* also, bgw_handle is no longer referenced */
	if (gpusort->bgw_handle)
	{
		pfree(gpusort->bgw_handle);
		gpusort->bgw_handle = NULL;
	}

	/*
	 * Let's try to merge with a preliminary sorted chunk.
	 * If the supplied newer chunk can find a buddy, gpusort_merge_chunks()
	 * put both of results as a input stream of the returned gpusort.
	 * Otherwise, gpusort was kept in the gss->sorted_chunks[] array to wait
	 * for the upcoming merginable chunk, or it is the final result of
	 * bitonic sorting.
	 */
	newsort = gpusort_merge_chunks(gss, gpusort);
	if (newsort)
	{
		/*
		 * If supplied gpusort chould find a pair to be merged,
		 * we need to enqueue the gpusort object (that shall have
		 * two input stream) again.
		 */
		SpinLockAcquire(&gss->gts.lock);
		dlist_push_head(&gss->gts.pending_tasks, &newsort->task.chain);
		gss->gts.num_pending_tasks++;
		SpinLockRelease(&gss->gts.lock);
	}
	else
	{
		SpinLockAcquire(&gss->gts.lock);

		/*
		 * Even if supplied gpusort chunk is kept in the sorted_chunks[]
		 * array, it might be the last chunk that contains a single
		 * sorted result array.
		 */
		if (gss->gts.scan_done &&
			gss->gts.num_running_tasks == 0 &&
			gss->gts.num_pending_tasks == 0)
		{
			Assert(gss->sorted_chunks[gpusort->mc_class] == gpusort);
			Assert(gss->num_chunks == bms_num_members(gpusort->chunk_id_map));
			gss->sorted_chunks[gpusort->mc_class] = NULL;
			dlist_push_tail(&gss->gts.ready_tasks, &gpusort->task.chain);
			gss->gts.num_ready_tasks++;

			elog(DEBUG1, "sort done (%s)",
				 gss->gts.css.methods->CustomName);
			gss->sort_done = true;	/* congratulation! */
		}
		SpinLockRelease(&gss->gts.lock);
	}
	return false;
}

/*
 * gpusort_task_respond
 */
static void
gpusort_task_respond(CUstream stream, CUresult status, void *private)
{
	pgstrom_gpusort	   *gpusort = (pgstrom_gpusort *) private;
	kern_resultbuf	   *kresults = GPUSORT_GET_KRESULTS(gpusort->oitems_dsm);
	GpuTaskState	   *gts = gpusort->task.gts;

	/* See comments in pgstrom_respond_gpuscan() */
	if (status == CUDA_ERROR_INVALID_CONTEXT || !IsTransactionState())
		return;

	if (status == CUDA_SUCCESS)
		gpusort->task.kerror = kresults->kerror;
	else
	{
		gpusort->task.kerror.errcode = status;
		gpusort->task.kerror.kernel = StromKernel_CudaRuntime;
		gpusort->task.kerror.lineno = 0;
	}

	/*
	 * Remove the GpuTask from the running_tasks list, and attach it
	 * on the completed_tasks list again. Note that this routine may
	 * be called by CUDA runtime, prior to attachment of GpuTask on
	 * the running_tasks by cuda_control.c.
	 */
	SpinLockAcquire(&gts->lock);
	if (gpusort->task.chain.prev && gpusort->task.chain.next)
	{
		dlist_delete(&gpusort->task.chain);
		gts->num_running_tasks--;
	}
	if (gpusort->task.kerror.errcode == StromError_Success)
		dlist_push_tail(&gts->completed_tasks, &gpusort->task.chain);
	else
		dlist_push_head(&gts->completed_tasks, &gpusort->task.chain);
	gts->num_completed_tasks++;
	SpinLockRelease(&gts->lock);

	SetLatch(&MyProc->procLatch);
}

/*
 * launch_gpu_bitonic_kernels
 *
 * Helper function to launch a series of bitonic-sort kernels
 */
static void
launch_gpu_bitonic_kernels(GpuSortState *gss,
						   pgstrom_gpusort *gpusort,
						   size_t nitems)
{
	size_t		block_size = INT_MAX;
	size_t		grid_size;
	size_t		block_temp;
	size_t		grid_temp;
	size_t		nhalf;
	CUfunction	sort_kernels[3];
	size_t		shmem_unitsz[3];
	void	   *kernel_args[4];
	cl_int		bitonic_unitsz;
	CUresult	rc;
	int			i, j;

	/*
	 * Find the maximum available block size for kernel functions
	 */
	sort_kernels[0] = gpusort->kern_bitonic_local;
	sort_kernels[1] = gpusort->kern_bitonic_step;
	sort_kernels[2] = gpusort->kern_bitonic_merge;
	shmem_unitsz[0] = Max(2 * sizeof(cl_uint), sizeof(kern_errorbuf));
	shmem_unitsz[1] = Max(sizeof(cl_uint), sizeof(kern_errorbuf));
	shmem_unitsz[2] = Max(sizeof(cl_uint), sizeof(kern_errorbuf));

	for (i=0; i < lengthof(sort_kernels); i++)
	{
		pgstrom_compute_workgroup_size(&grid_temp,
									   &block_temp,
									   sort_kernels[i],
									   gpusort->task.cuda_device,
									   true,
									   (nitems + 1) / 2,
									   shmem_unitsz[i]);
		block_size = Min(block_size, block_temp);
	}
	/*
	 * NOTE: block_size has to be common, and the least 2^N value less
	 * than or equal to the least block_size in the kernels above.
	 */
	block_size = 1UL << (get_next_log2(block_size + 1) - 1);

	/*
	 * OK, launch the series of GpuSort kernels
	 */

	/* NOTE: nhalf is the least power of two value that is larger than or
	 * equal to half of the nitems. */
	nhalf = 1UL << (get_next_log2(nitems + 1) - 1);

	/* KERNEL_FUNCTION_MAXTHREADS(void)
	 * gpusort_bitonic_local(kern_gpusort *kgpusort,
	 *                       kern_data_store *kds,
	 *                       kern_data_store *ktoast)
	 */
	kernel_args[0] = &gpusort->m_gpusort;
	kernel_args[1] = &gpusort->m_kds;
	kernel_args[2] = &gpusort->m_ktoast;
	kernel_args[3] = &bitonic_unitsz;

	grid_size = ((nitems + 1) / 2 + block_size - 1) / block_size;
	rc = cuLaunchKernel(gpusort->kern_bitonic_local,
						grid_size, 1, 1,
						block_size, 1, 1,
						Max(2 * sizeof(cl_uint),
							sizeof(kern_errorbuf)) * block_size,
						gpusort->task.cuda_stream,
						kernel_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
	elog(DEBUG2, "gpusort_bitonic_local(gridSz=%zu,blockSz=%zu)",
		 grid_size, block_size);
	gpusort->task.pfm.num_gpu_sort++;

	/* Sorting inter blocks */
	for (i = block_size; i < nhalf; i *= 2)
	{
		for (j = 2 * i; j > block_size; j /= 2)
		{
			cl_uint		unitsz = 2 * j;
			size_t		work_size;

			/*
			 * KERNEL_FUNCTION_MAXTHREADS(void)
			 * gpusort_bitonic_step(kern_gpusort *kgpusort,
			 *                      kern_data_store *kds,
			 *                      kern_data_store *ktoast,
			 *                      cl_int bitonic_unitsz)
			 */
			/* 4th argument of the kernel */
			bitonic_unitsz = ((j == 2 * i) ? -unitsz : unitsz);

			work_size = (((nitems + unitsz - 1) / unitsz) * unitsz / 2);
			grid_size = (work_size + block_size - 1) / block_size;
			rc = cuLaunchKernel(gpusort->kern_bitonic_step,
								grid_size, 1, 1,
								block_size, 1, 1,
								Max(sizeof(cl_uint),
									sizeof(kern_errorbuf)) * block_size,
								gpusort->task.cuda_stream,
								kernel_args,
								NULL);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
			elog(DEBUG2,
				 "gpusort_bitonic_step(gridSz=%zu,blockSz=%zu,unitsz=%d)",
				 grid_size, block_size,bitonic_unitsz);
			gpusort->task.pfm.num_gpu_sort++;
		}

		/*
		 * KERNEL_FUNCTION_MAXTHREADS(void)
		 * gpusort_bitonic_merge(kern_gpusort *kgpusort,
		 *                       kern_data_store *kds,
		 *                       kern_data_store *ktoast)
		 */
		grid_size = ((nitems + 1) / 2 + block_size - 1) / block_size;
		rc = cuLaunchKernel(gpusort->kern_bitonic_merge,
							grid_size, 1, 1,
							block_size, 1, 1,
							Max(2 * sizeof(cl_uint),
								sizeof(kern_errorbuf)) * block_size,
							gpusort->task.cuda_stream,
							kernel_args,
							NULL);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
		elog(DEBUG2,
			 "gpusort_bitonic_merge(gridSz=%zu,blockSz=%zu)",
			 grid_size, block_size);
		gpusort->task.pfm.num_gpu_sort++;
	}
}

static bool
__gpusort_task_process(GpuSortState *gss, pgstrom_gpusort *gpusort)
{
	kern_resultbuf	   *kresults = GPUSORT_GET_KRESULTS(gpusort->oitems_dsm);
	pgstrom_data_store *pds;
	pgstrom_data_store *ptoast;
	void			   *kernel_args[8];
	Size				total_length;
	Size				length;
	Size				offset;
	size_t				nitems;
	size_t				grid_size;
	size_t				block_size;
	CUresult			rc;

	Assert(gpusort->chunk_id >= 0 && gpusort->chunk_id < gss->num_chunks);
	pds = gss->pds_chunks[gpusort->chunk_id];
	ptoast = gss->pds_toasts[gpusort->chunk_id];
	nitems = ptoast->kds->nitems;

	/*
	 * kernel function lookup
	 */
	rc = cuModuleGetFunction(&gpusort->kern_prep,
							 gpusort->task.cuda_module,
							 "gpusort_preparation");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction : %s", errorText(rc));

	rc = cuModuleGetFunction(&gpusort->kern_bitonic_local,
							 gpusort->task.cuda_module,
							 "gpusort_bitonic_local");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction : %s", errorText(rc));

	rc = cuModuleGetFunction(&gpusort->kern_bitonic_step,
							 gpusort->task.cuda_module,
							 "gpusort_bitonic_step");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction : %s", errorText(rc));

	rc = cuModuleGetFunction(&gpusort->kern_bitonic_merge,
							 gpusort->task.cuda_module,
							 "gpusort_bitonic_merge");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction : %s", errorText(rc));

	rc = cuModuleGetFunction(&gpusort->kern_fixup,
							 gpusort->task.cuda_module,
							 "gpusort_fixup_datastore");
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuModuleGetFunction : %s", errorText(rc));

	/*
	 * allocation of device memory
	 */
	length = (STROMALIGN(gss->gts.kern_params->length) +
			  STROMALIGN(offsetof(kern_resultbuf, results[2 * nitems])));

	total_length = (GPUMEMALIGN(length) +
					GPUMEMALIGN(KERN_DATA_STORE_LENGTH(pds->kds)) +
					GPUMEMALIGN(KERN_DATA_STORE_LENGTH(ptoast->kds)));
	gpusort->m_gpusort = gpuMemAlloc(&gpusort->task, total_length);
	if (!gpusort->m_gpusort)
		goto out_of_resource;

	gpusort->m_kds = gpusort->m_gpusort + GPUMEMALIGN(length);
	gpusort->m_ktoast = gpusort->m_kds +
		GPUMEMALIGN(KERN_DATA_STORE_LENGTH(pds->kds));

	/*
	 * creation of event objects, if any
	 */
	if (gpusort->task.pfm.enabled)
	{
		rc = cuEventCreate(&gpusort->ev_dma_send_start, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));
		rc = cuEventCreate(&gpusort->ev_dma_send_stop, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));
		rc = cuEventCreate(&gpusort->ev_dma_recv_start, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));
		rc = cuEventCreate(&gpusort->ev_dma_recv_stop, CU_EVENT_DEFAULT);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuEventCreate: %s", errorText(rc));
	}

	/*
	 * Registers oitems_dsm as host pinned memory
	 */
	if (!gpusort->oitems_pinned)
	{
		rc = cuMemHostRegister(dsm_segment_address(gpusort->oitems_dsm),
							   dsm_segment_map_length(gpusort->oitems_dsm),
							   CU_MEMHOSTREGISTER_PORTABLE);
		Assert(rc == CUDA_SUCCESS);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuMemHostRegister: %s", errorText(rc));
		gpusort->oitems_pinned = true;
	}

	/*
	 * OK, enqueue a series of commands
	 */
	CUDA_EVENT_RECORD(gpusort, ev_dma_send_start);

	rc = cuMemcpyHtoDAsync(gpusort->m_gpusort,
						   gss->gts.kern_params,
						   gss->gts.kern_params->length,
						   gpusort->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	gpusort->task.pfm.bytes_dma_send += gss->gts.kern_params->length;
	gpusort->task.pfm.num_dma_send++;

	offset = STROMALIGN(gss->gts.kern_params->length);
	length = offsetof(kern_resultbuf, results[0]);
	rc = cuMemcpyHtoDAsync(gpusort->m_gpusort + offset,
						   kresults,
						   length,
						   gpusort->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	gpusort->task.pfm.bytes_dma_send += length;
	gpusort->task.pfm.num_dma_send++;

	rc = cuMemcpyHtoDAsync(gpusort->m_kds,
						   pds->kds,
						   KERN_DATA_STORE_HEAD_LENGTH(pds->kds),
						   gpusort->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	gpusort->task.pfm.bytes_dma_send += KERN_DATA_STORE_HEAD_LENGTH(pds->kds);
	gpusort->task.pfm.num_dma_send++;

	rc = cuMemcpyHtoDAsync(gpusort->m_ktoast,
						   ptoast->kds,
						   KERN_DATA_STORE_LENGTH(ptoast->kds),
						   gpusort->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemcpyHtoDAsync: %s", errorText(rc));
	gpusort->task.pfm.bytes_dma_send += KERN_DATA_STORE_LENGTH(ptoast->kds);
	gpusort->task.pfm.num_dma_send++;

	CUDA_EVENT_RECORD(gpusort, ev_dma_send_stop);

	/*
	 * KERNEL_FUNCTION(void)
	 * gpusort_preparation(kern_gpusort *kgpusort,
	 *                     kern_data_store *kds,
	 *                     kern_data_store *ktoast,
	 *                     cl_int chunk_id)
	 */
	pgstrom_compute_workgroup_size(&grid_size,
								   &block_size,
								   gpusort->kern_prep,
								   gpusort->task.cuda_device,
								   false,
								   nitems,
								   sizeof(kern_errorbuf));
	kernel_args[0] = &gpusort->m_gpusort;
	kernel_args[1] = &gpusort->m_kds;
	kernel_args[2] = &gpusort->m_ktoast;
	kernel_args[3] = &gpusort->chunk_id;
	rc = cuLaunchKernel(gpusort->kern_prep,
						grid_size, 1, 1,
						block_size, 1, 1,
						sizeof(kern_errorbuf) * block_size,
						gpusort->task.cuda_stream,
						kernel_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
	elog(DEBUG2, "kern_prep grid_size=%zu block_size=%zu nitems=%zu",
		 grid_size, block_size, (size_t)nitems);
	gpusort->task.pfm.num_kern_prep++;

	/*
	 * Launch kernel functions of bitonic-sorting
	 */
	launch_gpu_bitonic_kernels(gss, gpusort, nitems);

	/*
	 * KERNEL_FUNCTION(void)
	 * gpusort_fixup_datastore(kern_gpusort *kgpusort,
	 *                         kern_data_store *kds,
	 *                         kern_data_store *ktoast)
	 */
	pgstrom_compute_workgroup_size(&grid_size,
                                   &block_size,
                                   gpusort->kern_fixup,
								   gpusort->task.cuda_device,
								   false,
								   nitems,
								   sizeof(kern_errorbuf));
	kernel_args[0] = &gpusort->m_gpusort;
	kernel_args[1] = &gpusort->m_kds;
	kernel_args[2] = &gpusort->m_ktoast;
	rc = cuLaunchKernel(gpusort->kern_fixup,
						grid_size, 1, 1,
						block_size, 1, 1,
						sizeof(kern_errorbuf) * block_size,
						gpusort->task.cuda_stream,
						kernel_args,
						NULL);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuLaunchKernel: %s", errorText(rc));
	gpusort->task.pfm.num_kern_prep++;

	/*
	 * DMA Recv
	 */
	CUDA_EVENT_RECORD(gpusort, ev_dma_recv_start);

	offset = STROMALIGN(gss->gts.kern_params->length);
	length = STROMALIGN(offsetof(kern_resultbuf, results[2 * nitems]));
	rc = cuMemcpyDtoHAsync(kresults,
						   gpusort->m_gpusort + offset,
						   length,
						   gpusort->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuMemcpyDtoHAsync: %s", errorText(rc));
	gpusort->task.pfm.bytes_dma_recv += length;
	gpusort->task.pfm.num_dma_recv++;

	length = KERN_DATA_STORE_LENGTH(pds->kds);
	rc = cuMemcpyDtoHAsync(pds->kds,
						   gpusort->m_kds,
						   length,
						   gpusort->task.cuda_stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuMemcpyDtoHAsync: %s", errorText(rc));
	gpusort->task.pfm.bytes_dma_recv += length;
	gpusort->task.pfm.num_dma_recv++;

	CUDA_EVENT_RECORD(gpusort, ev_dma_recv_stop);

	/*
	 * register callback
	 */
	rc = cuStreamAddCallback(gpusort->task.cuda_stream,
							 gpusort_task_respond,
							 gpusort, 0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuStreamAddCallback: %s", errorText(rc));

	return true;

out_of_resource:
	gpusort_cleanup_cuda_resources(gpusort);
	return false;
}

static bool
gpusort_task_process(GpuTask *gtask)
{
	GpuSortState	   *gss = (GpuSortState *) gtask->gts;
	pgstrom_gpusort	   *gpusort = (pgstrom_gpusort *) gtask;
	CUresult			rc;
	bool				status;

	/*
	 * Launch a background worker if CPU sort is required.
	 */
	if (gpusort->mc_class > 0)
	{
		BackgroundWorker	worker;
		dsm_handle			dsm_hnd;

		Assert(gpusort->task.no_cuda_setup);

		/* setup dynamic background worker */
		dsm_hnd = dsm_segment_handle(gpusort->oitems_dsm);

		memset(&worker, 0, sizeof(BackgroundWorker));
		snprintf(worker.bgw_name, sizeof(worker.bgw_name),
				 "GpuSort worker-%u", gss->cpusort_seqno++);
		worker.bgw_flags = BGWORKER_SHMEM_ACCESS |
			BGWORKER_BACKEND_DATABASE_CONNECTION;
		worker.bgw_start_time = BgWorkerStart_RecoveryFinished;
		worker.bgw_restart_time = BGW_NEVER_RESTART;
		worker.bgw_main = bgw_cpusort_entrypoint;
		worker.bgw_main_arg = PointerGetDatum(dsm_hnd);

		return RegisterDynamicBackgroundWorker(&worker, &gpusort->bgw_handle);
	}

	/*
	 * Switch CUDA context, then kick GPU sorting elsewhere.
	 */
	rc = cuCtxPushCurrent(gpusort->task.cuda_context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxPushCurrent: %s", errorText(rc));
	PG_TRY();
	{
		status = __gpusort_task_process(gss, gpusort);
	}
	PG_CATCH();
	{
		gpusort_cleanup_cuda_resources(gpusort);
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
 * NOTE: This callback is called under the gts->lock held.
 */
static void
gpusort_task_polling(GpuTaskState *gts)
{
	GpuSortState	   *gss = (GpuSortState *) gts;
	pgstrom_gpusort	   *gpusort;
	BgwHandleStatus		status;
	pid_t				bgw_pid;
	dlist_mutable_iter	iter;

	dlist_foreach_modify(iter, &gss->gts.running_tasks)
	{
		gpusort = dlist_container(pgstrom_gpusort, task.chain, iter.cur);

		/* A background worker task? */
		if (!gpusort->bgw_handle)
			continue;
		/* Already finished? */
		status = GetBackgroundWorkerPid(gpusort->bgw_handle, &bgw_pid);
		if (status == BGWH_STARTED || status == BGWH_STOPPED)
		{
			pgstrom_flat_gpusort   *pfg = (pgstrom_flat_gpusort *)
				dsm_segment_address(gpusort->oitems_dsm);

			/* not yet finished */
			if (!pfg->bgw_done)
				continue;

			/* detach from running_tasks, then attach to completed tasks */
			dlist_delete(&gpusort->task.chain);
			gss->gts.num_running_tasks--;

			dlist_push_tail(&gss->gts.completed_tasks, &gpusort->task.chain);
			gss->gts.num_completed_tasks++;
		}
	}
}

/*
 * form_pgstrom_flat_gpusort(_base)
 * deform_pgstrom_flat_gpusort
 *
 * form/deform pgstrom_gpusort structure to exchange data through the
 * dynamic shared memory segment.
 */
static dsm_segment *
form_pgstrom_flat_gpusort_base(GpuSortState *gss, cl_int chunk_id)
{
	dsm_segment		   *dsm_seg;
	kern_resultbuf	   *kresults;
	Size				dsm_length;
	pgstrom_data_store *ptoast;
	size_t				nitems;
	pgstrom_flat_gpusort *pfg;

	ptoast = gss->pds_toasts[chunk_id];
	nitems = ptoast->kds->nitems;
	dsm_length = (STROMALIGN(offsetof(pgstrom_flat_gpusort, data)) +
				  STROMALIGN(offsetof(kern_resultbuf, results[2 * nitems])));
	dsm_seg = dsm_create(dsm_length, 0);
	pfg = dsm_segment_address(dsm_seg);
	memset(pfg, 0, sizeof(pgstrom_flat_gpusort));
	pfg->dsm_length = dsm_length;
	pfg->kresults_ofs = 0;

	kresults = GPUSORT_GET_KRESULTS(dsm_seg);
	memset(kresults, 0, sizeof(kern_resultbuf));
	kresults->nrels = 2;
	kresults->nrooms = nitems;
	kresults->nitems = nitems;
	memset(&kresults->kerror, 0, sizeof(kern_errorbuf));

	return dsm_seg;
}

static dsm_segment *
form_pgstrom_flat_gpusort(GpuSortState *gss,
						  pgstrom_gpusort *l_gpusort,
						  pgstrom_gpusort *r_gpusort)
{
	StringInfoData	buf;
	pgstrom_flat_gpusort  pfg;
	pgstrom_flat_gpusort *pfg_buf;
	TupleDesc			tupdesc = GTS_GET_SCAN_TUPDESC(&gss->gts);
	kern_resultbuf	   *kresults;
	kern_resultbuf	   *l_kresults;
	kern_resultbuf	   *r_kresults;
	Bitmapset		   *chunk_id_map;
	size_t				length;
	size_t				nitems;
	size_t				kresults_ofs;
	cl_int				index;
	dsm_segment		   *dsm_seg;

	Assert(l_gpusort && r_gpusort);
	initStringInfo(&buf);
	memset(&pfg, 0, sizeof(pgstrom_flat_gpusort));

	/* connection info */
	pfg.backend_proc = MyProc;
	/* database name */
	pfg.database_name = buf.len;
	length = strlen(gss->database_name) + 1;
	enlargeStringInfo(&buf, MAXALIGN(length));
	strcpy(buf.data + buf.len, gss->database_name);
    buf.len += MAXALIGN(length);

	/* calculate total number of items on the output buffer */
	l_kresults = GPUSORT_GET_KRESULTS(l_gpusort->oitems_dsm);
	r_kresults = GPUSORT_GET_KRESULTS(r_gpusort->oitems_dsm);

	nitems = l_kresults->nitems + r_kresults->nitems;
	pfg.litems_dsmhnd = dsm_segment_handle(l_gpusort->oitems_dsm);
	pfg.ritems_dsmhnd = dsm_segment_handle(r_gpusort->oitems_dsm);

	/* existance of varlena sorting key */
	pfg.varlena_keys = gss->varlena_keys;
	/* length of kern_data_store array */
	pfg.num_chunks = gss->num_chunks;

	/*
	 * put filename of file-mapped kds and its length
	 */
	pfg.kern_chunks = buf.len;
	Assert(bms_is_empty(bms_intersect(l_gpusort->chunk_id_map,
									  r_gpusort->chunk_id_map)));
	chunk_id_map = bms_union(l_gpusort->chunk_id_map,
							 r_gpusort->chunk_id_map);
	while ((index = bms_first_member(chunk_id_map)) >= 0)
	{
		pgstrom_data_store *pds = gss->pds_chunks[index];
		char	   *kds_fname = pds->kds_fname;
		size_t		kds_len = pds->kds_length;
		size_t		kds_ofs = pds->kds_offset;

		Assert(strcmp(kds_fname, gss->pds_toasts[index]->kds_fname) == 0);
		Assert(kds_ofs >= gss->pds_toasts[index]->kds_length);
		Assert(gss->pds_toasts[index]->kds_offset == 0);

		appendBinaryStringInfo(&buf, (const char *)&index, sizeof(int));
		appendBinaryStringInfo(&buf, (const char *)&kds_len, sizeof(size_t));
		appendBinaryStringInfo(&buf, (const char *)&kds_ofs, sizeof(size_t));

		length = strlen(kds_fname) + 1;
		enlargeStringInfo(&buf, MAXALIGN(length));
		strcpy(buf.data + buf.len, kds_fname);
		buf.len += MAXALIGN(length);
	}
	bms_free(chunk_id_map);
	index = -1;		/* end of chunks marker */
	appendBinaryStringInfo(&buf, (const char *)&index, sizeof(int));

	/* tuple descriptor of sorting keys */
	pfg.tupdesc = buf.len;
	enlargeStringInfo(&buf, MAXALIGN(sizeof(*tupdesc)));
	memcpy(buf.data + buf.len, tupdesc, sizeof(*tupdesc));
	buf.len += MAXALIGN(sizeof(*tupdesc));
	for (index=0; index < tupdesc->natts; index++)
	{
		enlargeStringInfo(&buf, MAXALIGN(ATTRIBUTE_FIXED_PART_SIZE));
		memcpy(buf.data + buf.len,
			   tupdesc->attrs[index],
			   ATTRIBUTE_FIXED_PART_SIZE);
		buf.len += MAXALIGN(ATTRIBUTE_FIXED_PART_SIZE);
	}
	/* numCols */
	pfg.numCols = gss->numCols;
	/* sortColIdx */
	pfg.sortColIdx = buf.len;
	length = sizeof(AttrNumber) * gss->numCols;
	enlargeStringInfo(&buf, MAXALIGN(length));
	memcpy(buf.data + buf.len, gss->sortColIdx, length);
	buf.len += MAXALIGN(length);
	/* sortOperators */
	pfg.sortOperators = buf.len;
	length = sizeof(Oid) * gss->numCols;
	enlargeStringInfo(&buf, MAXALIGN(length));
    memcpy(buf.data + buf.len, gss->sortOperators, length);
	buf.len += MAXALIGN(length);
	/* collations */
	pfg.collations = buf.len;
	length = sizeof(Oid) * gss->numCols;
    enlargeStringInfo(&buf, MAXALIGN(length));
	memcpy(buf.data + buf.len, gss->collations, length);
	buf.len += MAXALIGN(length);
	/* nullsFirst */
	pfg.nullsFirst = buf.len;
	length = sizeof(bool) * gss->numCols;
	enlargeStringInfo(&buf, MAXALIGN(length));
    memcpy(buf.data + buf.len, gss->nullsFirst, length);
	buf.len += MAXALIGN(length);

	/* result buffer */
	kresults_ofs = buf.len;
	enlargeStringInfo(&buf, MAXALIGN(sizeof(kern_resultbuf)) + 80);
	kresults = (kern_resultbuf *)(buf.data + buf.len);
	memset(kresults, 0, sizeof(kern_resultbuf));
	kresults->nrels = 2;
	kresults->nrooms = nitems;
	kresults->nitems = nitems;
	kresults->kerror.errcode = ERRCODE_INTERNAL_ERROR;
	snprintf((char *)kresults->results, sizeof(cl_int) * 2 * nitems,
			 "An internal error on worker prior to DSM attachment");
	buf.len += MAXALIGN(sizeof(kern_resultbuf)) + 80;

	/* allocation and setup of DSM */
	length = STROMALIGN(offsetof(pgstrom_flat_gpusort, data)) +
		STROMALIGN(kresults_ofs) +
		STROMALIGN(offsetof(kern_resultbuf, results[2 * nitems]));
	pfg.dsm_length = length;
	pfg.kresults_ofs = kresults_ofs;

	dsm_seg = dsm_create(pfg.dsm_length, 0);
	pfg_buf = dsm_segment_address(dsm_seg);
	memcpy(pfg_buf, &pfg, sizeof(pgstrom_flat_gpusort));
	memcpy(pfg_buf->data, buf.data, buf.len);

	/* release temp buffer */
	pfree(buf.data);

	return dsm_seg;
}

static pgstrom_gpusort *
deform_pgstrom_flat_gpusort(dsm_segment *dsm_seg)
{
	pgstrom_flat_gpusort *pfg = dsm_segment_address(dsm_seg);
	pgstrom_gpusort	*gpusort = palloc0(sizeof(pgstrom_gpusort));
	char	   *pos = pfg->data;
	TupleDesc	tupdesc;
	int			index;

	/* connection information */
	gpusort->backend_proc = pfg->backend_proc;
	gpusort->database_name = pfg->data + pfg->database_name;

	/*
	 * input/output result buffer
	 * NOTE: this function is called only when background worker process.
	 * So, both of valid input segments are expected/
	 */
	gpusort->litems_dsm = dsm_attach(pfg->litems_dsmhnd);
	if (!gpusort->litems_dsm)
		elog(ERROR, "failed to attach dsm segment: %u", pfg->litems_dsmhnd);
	gpusort->ritems_dsm = dsm_attach(pfg->ritems_dsmhnd);
	if (!gpusort->ritems_dsm)
		elog(ERROR, "failed to attach dsm segment: %u", pfg->ritems_dsmhnd);
	gpusort->oitems_dsm = dsm_seg;	/* myself */

	/* chunks to be sorted */
	gpusort->varlena_keys = pfg->varlena_keys;
	gpusort->num_chunks = pfg->num_chunks;
	gpusort->kern_toasts = palloc0(sizeof(kern_data_store *) *
								   pfg->num_chunks);
	gpusort->kern_chunks = palloc0(sizeof(kern_data_store *) *
								   pfg->num_chunks);
	/* chunks to be mapped */
	pos = pfg->data + pfg->kern_chunks;
	while (true)
	{
		Size		kds_length;
		Size		kds_offset;
		FileName	kds_fname;

		index = *((int *) pos);
		pos += sizeof(int);
		if (index < 0)
			break;		/* end of chunks that are involved */
		if (index >= gpusort->num_chunks)
			elog(ERROR, "Bug? chunk-id is out of range %d for %d",
				 index, gpusort->num_chunks);

		kds_length = *((size_t *) pos);
		pos += sizeof(size_t);
		kds_offset = *((size_t *) pos);
		pos += sizeof(size_t);
		kds_fname = pos;
		pos += MAXALIGN(strlen(kds_fname) + 1);

		pgstrom_file_mmap_data_store(kds_fname, kds_offset, kds_length,
									 gpusort->kern_chunks + index,
									 gpusort->varlena_keys
									 ? gpusort->kern_toasts + index
									 : NULL);
	}
	/* tuple descriptor */
	tupdesc = (TupleDesc)(pfg->data + pfg->tupdesc);
	gpusort->tupdesc = palloc0(sizeof(*tupdesc));
	gpusort->tupdesc->natts = tupdesc->natts;
	gpusort->tupdesc->attrs =
		palloc0(ATTRIBUTE_FIXED_PART_SIZE * tupdesc->natts);
	gpusort->tupdesc->tdtypeid = tupdesc->tdtypeid;
	gpusort->tupdesc->tdtypmod = tupdesc->tdtypmod;
	gpusort->tupdesc->tdhasoid = tupdesc->tdhasoid;
	gpusort->tupdesc->tdrefcount = -1;	/* not counting */
	pos += MAXALIGN(sizeof(*tupdesc));

	for (index=0; index < tupdesc->natts; index++)
	{
		gpusort->tupdesc->attrs[index] = (Form_pg_attribute) pos;
		pos += MAXALIGN(ATTRIBUTE_FIXED_PART_SIZE);
	}
	/* sorting keys */
	gpusort->numCols = pfg->numCols;
	gpusort->sortColIdx = (AttrNumber *)(pfg->data + pfg->sortColIdx);
	gpusort->sortOperators = (Oid *)(pfg->data + pfg->sortOperators);
	gpusort->collations = (Oid *)(pfg->data + pfg->collations);
	gpusort->nullsFirst = (bool *)(pfg->data + pfg->nullsFirst);

	return gpusort;
}

/*
 * gpusort_fallback_gpu_chunks
 *
 * Fallback routine towards a particular GpuSort chunk, if GPU device
 * gave up sorting on device side.
 */
static inline int
__gpusort_fallback_compare(GpuSortState *gss,
						   kern_data_store *kds, cl_uint index,
						   Datum *p_values, bool *p_isnull)
{
	Datum	   *x_values = (Datum *) KERN_DATA_STORE_VALUES(kds, index);
	bool	   *x_isnull = (bool *) KERN_DATA_STORE_ISNULL(kds, index);
	int			i, j, comp;

	for (i=0; i < gss->numCols; i++)
	{
		SortSupport		ssup = gss->ssup_keys + i;

		j = ssup->ssup_attno - 1;
		comp = ApplySortComparator(x_values[j],
								   x_isnull[j],
								   p_values[j],
								   p_isnull[j],
								   ssup);
		if (comp != 0)
			return comp;
	}
	return 0;
}

static void
__gpusort_fallback_quicksort(GpuSortState *gss,
							 kern_resultbuf *kresults,
							 kern_data_store *kds,
							 cl_int l_bound, cl_int r_bound)
{
	if (l_bound <= r_bound)
	{
		cl_int		l_index = l_bound;
		cl_int		r_index = r_bound;
		cl_int		p_index = kresults->results[(l_index + r_index) | 0x0001];
		Datum	   *p_values = (Datum *) KERN_DATA_STORE_VALUES(kds, p_index);
		bool	   *p_isnull = (bool *) KERN_DATA_STORE_ISNULL(kds, p_index);

		while (l_index <= r_index)
		{
			while (l_index <= r_bound &&
				   __gpusort_fallback_compare(gss, kds,
											  kresults->results[2*l_index + 1],
											  p_values, p_isnull) < 0)
				l_index++;
			while (r_index >= l_bound &&
				   __gpusort_fallback_compare(gss, kds,
											  kresults->results[2*r_index + 1],
											  p_values, p_isnull) > 0)
				r_index--;

			if (l_index <= r_index)
			{
				cl_int	l_prev = kresults->results[2*l_index + 1];
				cl_int	r_prev = kresults->results[2*r_index + 1];

				Assert(kresults->results[2*l_index] ==
					   kresults->results[2*r_index]);
				kresults->results[2*l_index + 1] = r_prev;
				kresults->results[2*r_index + 1] = l_prev;
				l_index++;
				r_index--;
			}
		}
		__gpusort_fallback_quicksort(gss, kresults, kds, l_bound, r_index);
		__gpusort_fallback_quicksort(gss, kresults, kds, l_index, r_bound);
	}
}

static void
gpusort_fallback_quicksort(GpuSortState *gss, pgstrom_gpusort *gpusort)
{
	SortSupportData	   *ssup_keys = gss->ssup_keys;
	kern_resultbuf	   *kresults = GPUSORT_GET_KRESULTS(gpusort->oitems_dsm);
	pgstrom_data_store *pds = gss->pds_chunks[gpusort->chunk_id];
	pgstrom_data_store *ptoast = gss->pds_chunks[gpusort->chunk_id];
	kern_data_store	   *kds = pds->kds;
	kern_data_store	   *ktoast = ptoast->kds;
	size_t				i, nitems = ktoast->nitems;

	/* initialize SortSupportData, if first time */
	if (!ssup_keys)
	{
		EState	   *estate = gss->gts.css.ss.ps.state;
		Size		len = sizeof(SortSupportData) * gss->numCols;

		ssup_keys = MemoryContextAllocZero(estate->es_query_cxt, len);
		for (i=0; i < gss->numCols; i++)
		{
			SortSupport		ssup = ssup_keys + i;

			ssup->ssup_cxt = estate->es_query_cxt;
			ssup->ssup_collation = gss->collations[i];
			ssup->ssup_nulls_first = gss->nullsFirst[i];
			ssup->ssup_attno = gss->sortColIdx[i];
			PrepareSortSupportFromOrderingOp(gss->sortOperators[i], ssup);
		}
		gss->ssup_keys = ssup_keys;
	}
	/* preparation of rindex[] */
	Assert(kresults->nrels == 2);
	Assert(kresults->nrooms == nitems);

	/*
	 * NOTE: we assume gpusort_preparation() and gpusort_fixup_datastore()
	 * has no code path that can return CpuReCheck error, so we expect
	 * receive DMA will back already flatten data store in kds/ktoast.
	 */
	Assert(kds->nitems == nitems);
	for (i=0; i < nitems; i++)
	{
		kresults->results[2*i] = gpusort->chunk_id;
		kresults->results[2*i + 1] = i;
	}
	/* fallback execution with QuickSort */
	__gpusort_fallback_quicksort(gss, kresults, kds, 0, nitems - 1);

	/* restore error status */
	kresults->kerror.errcode = StromError_Success;
	kresults->nitems = kds->nitems;
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
							 false, /* not recommended now */
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
	gpusort_exec_methods.MarkPosCustomScan	= gpusort_mark_pos;
	gpusort_exec_methods.RestrPosCustomScan	= gpusort_restore_pos;
	gpusort_exec_methods.ExplainCustomScan	= gpusort_explain;
}

/* ================================================================
 *
 * Routines for CPU sorting
 *
 * ================================================================
 */
static void
bgw_cpusort_exec(pgstrom_gpusort *gpusort)
{
	SortSupportData	   *sort_keys;
	kern_data_store	   *kds;
	kern_data_store	   *ltoast = NULL;
	kern_data_store	   *rtoast = NULL;
	kern_resultbuf	   *litems = GPUSORT_GET_KRESULTS(gpusort->litems_dsm);
	kern_resultbuf	   *ritems = GPUSORT_GET_KRESULTS(gpusort->ritems_dsm);
	kern_resultbuf	   *oitems = GPUSORT_GET_KRESULTS(gpusort->oitems_dsm);
	Datum			   *rts_values = NULL;
	Datum			   *lts_values = NULL;
	cl_char			   *rts_isnull = NULL;
	cl_char			   *lts_isnull = NULL;
	TupleDesc			tupdesc = gpusort->tupdesc;
	long				rindex = 0;
	long				lindex = 0;
	long				oindex = 0;
	int					rchunk_id = 0;
	int					ritem_id = 0;
	int					lchunk_id = 0;
	int					litem_id = 0;
	int					i;

	/*
	 * Set up sorting keys
	 */
	sort_keys = palloc0(sizeof(SortSupportData) * gpusort->numCols);
	for (i=0; i < gpusort->numCols; i++)
	{
		SortSupport	ssup = sort_keys + i;

		ssup->ssup_cxt = CurrentMemoryContext;
		ssup->ssup_collation = gpusort->collations[i];
		ssup->ssup_nulls_first = gpusort->nullsFirst[i];
		ssup->ssup_attno = gpusort->sortColIdx[i];
		PrepareSortSupportFromOrderingOp(gpusort->sortOperators[i], ssup);
	}

	/*
	 * Begin merge sorting
	 */
	while (lindex < litems->nitems && rindex < ritems->nitems)
	{
		int		comp = 0;

		if (!lts_values)
		{
			lchunk_id = litems->results[2 * lindex];
			litem_id = litems->results[2 * lindex + 1];
			Assert(lchunk_id < gpusort->num_chunks);
			kds = gpusort->kern_chunks[lchunk_id];
			Assert(litem_id < kds->nitems);
			if (litem_id >= kds->nitems)
				elog(ERROR, "item-id %u is out of range in the chunk %u",
					 litem_id, lchunk_id);
			lts_values = KERN_DATA_STORE_VALUES(kds, litem_id);
			lts_isnull = KERN_DATA_STORE_ISNULL(kds, litem_id);
			ltoast = gpusort->kern_toasts[lchunk_id];
		}

		if (!rts_values)
		{
			rchunk_id = ritems->results[2 * rindex];
			ritem_id = ritems->results[2 * rindex + 1];
			Assert(rchunk_id < gpusort->num_chunks);
			kds = gpusort->kern_chunks[rchunk_id];
			if (ritem_id >= kds->nitems)
				elog(ERROR, "item-id %u is out of range in the chunk %u",
					 ritem_id, rchunk_id);
			rts_values = KERN_DATA_STORE_VALUES(kds, ritem_id);
			rts_isnull = KERN_DATA_STORE_ISNULL(kds, ritem_id);
			rtoast = gpusort->kern_toasts[rchunk_id];
		}

		for (i=0; i < gpusort->numCols; i++)
		{
			SortSupport ssup = sort_keys + i;
			AttrNumber	anum = ssup->ssup_attno - 1;
			Datum		r_value = rts_values[anum];
			bool		r_isnull = rts_isnull[anum];
			Datum		l_value = lts_values[anum];
			bool		l_isnull = lts_isnull[anum];
			Form_pg_attribute attr = tupdesc->attrs[anum];

			/* toast datum has to be fixed up */
			if (!attr->attbyval)
			{
				Assert(ltoast != NULL && rtoast != NULL);
				if (!l_isnull)
				{
					Assert(l_value > ltoast->hostptr);
					l_value = ((uintptr_t)l_value -
							   (uintptr_t)ltoast->hostptr) +
							  (uintptr_t)&ltoast->hostptr;
				}
				if (!r_isnull)
				{
					Assert(r_value > rtoast->hostptr);
					r_value = ((uintptr_t)r_value -
							   (uintptr_t)rtoast->hostptr) +
							  (uintptr_t)&rtoast->hostptr;
				}
			}
			comp = ApplySortComparator(r_value, r_isnull,
									   l_value, l_isnull,
									   ssup);
			if (comp != 0)
				break;
		}

		if (comp >= 0)
		{
			oitems->results[2 * oindex] = lchunk_id;
			oitems->results[2 * oindex + 1] = litem_id;
			oindex++;
			lindex++;
			lts_values = NULL;
			lts_isnull = NULL;
		}

		if (comp <= 0)
		{
			oitems->results[2 * oindex] = rchunk_id;
			oitems->results[2 * oindex + 1] = ritem_id;
			oindex++;
			rindex++;
			rts_values = NULL;
			rts_isnull = NULL;
		}
	}
	/* move remaining left chunk-id/item-id, if any */
	if (lindex < litems->nitems)
	{
		memcpy(oitems->results + 2 * oindex,
			   litems->results + 2 * lindex,
			   2 * sizeof(cl_int) * (litems->nitems - lindex));
		Assert(rindex == ritems->nitems);
	}
	/* move remaining right chunk-id/item-id, if any */
	if (rindex < ritems->nitems)
	{
		memcpy(oitems->results + 2 * oindex,
			   ritems->results + 2 * rindex,
			   2 * sizeof(cl_int) * (ritems->nitems - rindex));
		Assert(lindex == litems->nitems);
	}
	oitems->nitems = litems->nitems + ritems->nitems;
}

static void
bgw_cpusort_entrypoint(Datum main_arg)
{
	MemoryContext		bgw_mcxt;
	dsm_handle			dsm_hnd = (dsm_handle) main_arg;
	dsm_segment		   *dsm_seg;
	PGPROC			   *backend_proc;
	pgstrom_gpusort	   *gpusort;
	pgstrom_flat_gpusort *pfg;
	kern_resultbuf	   *kresults;

	/* We're now ready to receive signals */
	BackgroundWorkerUnblockSignals();
	/* Makes up resource owner and memory context */
	Assert(CurrentResourceOwner == NULL);
	CurrentResourceOwner = ResourceOwnerCreate(NULL, "CpuSort");
	bgw_mcxt = AllocSetContextCreate(TopMemoryContext,
										 "CpuSort",
										 ALLOCSET_DEFAULT_MINSIZE,
										 ALLOCSET_DEFAULT_INITSIZE,
										 ALLOCSET_DEFAULT_MAXSIZE);
	CurrentMemoryContext = bgw_mcxt;
	/* Deform caller's request */
	dsm_seg = dsm_attach(dsm_hnd);
	if (!dsm_seg)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("unable to map dynamic shared memory segment %u",
						(uint)dsm_hnd)));
	/*
	 * Initialization of error status. The kresults->errcode is initialized
	 * to an error code to deal with dsm_attach() got failed, so we have to
	 * clear the code first of all. Later ereport() shall be traped by
	 * PG_TRY() block, then we put appropriate error message here.
	 */
	pfg = dsm_segment_address(dsm_seg);
	kresults = GPUSORT_GET_KRESULTS(dsm_seg);
	kresults->kerror.errcode = StromError_Success;

	/* get reference to the backend process */
	backend_proc = pfg->backend_proc;

	PG_TRY();
	{
		/* deform CpuSort request to the usual format */
		gpusort = deform_pgstrom_flat_gpusort(dsm_seg);

		/* Connect to our database */
		BackgroundWorkerInitializeConnection(gpusort->database_name, NULL);

		/*
		 * XXX - Eventually, we should use parallel-context to share
		 * the transaction snapshot, initialize misc stuff and so on.
		 * But just at this moment, we create a new transaction state
		 * to simplifies the implementation.
		 */
		StartTransactionCommand();
		PushActiveSnapshot(GetTransactionSnapshot());

		gettimeofday(&pfg->tv_sort_start, NULL);

		/* handle CPU merge sorting */
		bgw_cpusort_exec(gpusort);

		gettimeofday(&pfg->tv_sort_end, NULL);

		/* we should have no side-effect */
		PopActiveSnapshot();
		CommitTransactionCommand();
	}
	PG_CATCH();
	{
		MemoryContext	ecxt = MemoryContextSwitchTo(bgw_mcxt);
		ErrorData	   *edata = CopyErrorData();
		Size			buflen;

		kresults->kerror.errcode = edata->sqlerrcode;
		buflen = sizeof(cl_int) * kresults->nrels * kresults->nrooms;
		snprintf((char *)kresults->results, buflen, "%s (%s, %s:%d)",
				 edata->message, edata->funcname,
				 edata->filename, edata->lineno);
		MemoryContextSwitchTo(ecxt);

		SetLatch(&backend_proc->procLatch);

		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Inform the corrdinator worker got finished */
	pfg->bgw_done = true;
	pg_memory_barrier();
	SetLatch(&backend_proc->procLatch);
}
