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
#include "catalog/pg_type.h"
#include "commands/dbcommands.h"
#include "nodes/nodeFuncs.h"
#include "nodes/makefuncs.h"
#include "optimizer/cost.h"
#include "parser/parsetree.h"
#include "postmaster/bgworker.h"
#include "storage/dsm.h"
#include "storage/latch.h"
#include "storage/proc.h"
#include "storage/procsignal.h"
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
	bool		varlena_keys;	/* True, if here are varlena keys */
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

/*
 * pgstrom_cpusort - information of CPU sorting. It shall be allocated on
 * the private memory, then dynamic worker will be able to reference the copy
 * because of process fork(2). It means, we don't mention about Windows
 * platform at this moment. :-)
 */
typedef struct
{
	dlist_node		chain;
	dsm_segment	   *litems_dsm;
	dsm_segment	   *ritems_dsm;
	dsm_segment	   *oitems_dsm;
	pgstrom_perfmon	pfm;
	union
	{
		struct
		{
			BackgroundWorkerHandle *bgw_handle;	/* valid, if running */
			cl_uint			mc_class;
			Bitmapset	   *chunk_ids;
		} h;
		struct
		{
			/* input/output result buffer */
			kern_resultbuf *l_kresults;
			kern_resultbuf *r_kresults;
			kern_resultbuf *o_kresults;

			/* database connection */
			char		   *database_name;
			/* sorting chunks */
			uint			varlena_keys;
			uint			max_chunk_id;
			kern_data_store **kern_toasts;
			kern_data_store	**kern_chunks;
			/* tuple descriptor */
			TupleDesc		tupdesc;
			/* sorting keys */
			int				numCols;
			AttrNumber	   *sortColIdx;
			Oid			   *sortOperators;
			Oid			   *collations;
			bool		   *nullsFirst;
		} w;
	};
} pgstrom_cpusort;

typedef struct
{
	Size		dsm_length;			/* total length of this structure */
	Size		kresults_offset;	/* offset from data[] */
	PGPROC	   *master_proc;		/* PGPROC reference on shared memory */
	dsm_handle	litems_dsmhnd;
	dsm_handle	ritems_dsmhnd;
	volatile bool cpusort_worker_done;	/* set prior to setLatch() by worker */
	volatile long time_cpu_sort;	/* performance info */
	/* connection information */
	Size		database_name;		/* offset from data[] */
	/* existence of varlena sorting keys */
	uint		varlena_keys;
	/* file mapped data store */
	uint		max_chunk_id;
	Size		kern_chunks;		/* offset from data[] */
	/* tuple descriptor */
	Size		tupdesc;			/* offset from data[] */
	/* sorting keys */
	int			numCols;
	Size		sortColIdx;			/* offset from data[] */
	Size		sortOperators;		/* offset from data[] */
	Size		collations;			/* offset from data[] */
	Size		nullsFirst;			/* offset from data[] */
	char		data[FLEXIBLE_ARRAY_MEMBER];
} pgstrom_flat_cpusort;

#define PFCSEG_GET_KRESULTS(dsmseg)										\
	((kern_resultbuf *)													\
	 (((pgstrom_flat_cpusort *)dsm_segment_address(dsmseg))->data +		\
	  ((pgstrom_flat_cpusort *)dsm_segment_address(dsmseg))->kresults_offset))

#define CSS_GET_SCAN_TUPDESC(css)				\
	(((ScanState *)(css))->ss_ScanTupleSlot->tts_tupleDescriptor)
#define CSS_GET_RESULT_TUPDESC(css)				\
	(((ScanState *)(css))->ps.ps_ResultTupleSlot->tts_tupleDescriptor)

#define MAX_MERGECHUNKS_CLASS		10		/* 1024 chunks are enough large */

typedef struct
{
	CustomScanState	css;

	/* data store saved in temporary file */
	uint			num_chunks;
	uint			num_chunks_limit;
	pgstrom_data_store **pds_chunks;
	pgstrom_data_store **pds_toasts;

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
	bool			varlena_keys;	/* True, if varlena sorting key exists */
	SortSupportData *ssup_keys;		/* executable comparison function */

	/* running status */
	char		   *database_name;	/* name of the current database */
	/* chunks already sorted but no pair yet */
	pgstrom_cpusort	*sorted_chunks[MAX_MERGECHUNKS_CLASS];
	cl_int			num_gpu_running;
	cl_int			num_cpu_running;
	cl_int			num_cpu_pending;
	dlist_head		pending_cpu_chunks;	/* chunks waiting for cpu sort */
	dlist_head		running_cpu_chunks;	/* chunks in running by bgworker */
	bool			scan_done;		/* if true, no tuples to read any more */
	bool			sort_done;		/* if true, now ready to fetch records */
	cl_int			cpusort_seqno;	/* seqno of cpusort to launch */

	/* random access capability */
	bool			randomAccess;
	cl_long			markpos_index;

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
	Size	max_chunk_size = pgstrom_shmem_maxalloc();

	if (ntuples < 2.0)
		ntuples = 2.0;
	/*
	 * Fixed cost to kick GPU kernel
	 */
	startup_cost += pgstrom_gpu_setup_cost;

	/*
	 * calculate expected number of rows per chunk and number of chunks.
	 */
	width = MAXALIGN(width + sizeof(cl_uint) +
					 offsetof(HeapTupleHeaderData, t_bits) +
					 BITMAPLEN(list_length(subplan->targetlist)));
	chunk_size = (Size)((double)width * ntuples * 1.10) + 1024;

	if (max_chunk_size > chunk_size)
	{
		nrows_per_chunk = ntuples;
		num_chunks = 1;
	}
	else
	{
		nrows_per_chunk =
			(double)(max_chunk_size - 1024) / ((double)width * 1.25);
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
			"    ts_values[%d] = *((__global %s *) datum);\n"
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
			"      pg_numeric_from_varlena(errcode, datum);\n"
			"    ts_isnull[%d] = temp.isnull;\n"
			"    ts_values[%d] = temp.value;\n"
			"  }\n",
			resno,
			resno - 1,
			resno - 1,
			resno - 1,
			resno - 1);
	}
	else if (is_sortkey)
	{
		/*
		 * variables with !type_byval but referenced as sortkey has to
		 * be fixed up as an offset from the ktoast.
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
			"    ts_isnull[%d] = false;\n"
			"    ts_values[%d] = (Datum)((__global char *)datum -\n"
			"                            (__global char *)&ktoast->hostptr);\n"
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
		 * Elsewhere, variables are host accessible pointer; either of
		 * varlena datum or fixed-length variables with !type_byval.
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
			"    ts_values[%d] = (Datum)\n"
			"      (((__global char *)datum -\n"
			"        (__global char *)&ktoast->hostptr) + ktoast->hostptr);\n"
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
	if (!is_sortkey)
		return;

	if (type_oid == NUMERICOID)
	{
		appendStringInfo(
			body,
			"\n"
			"  /* sortkey %d fixup */\n"
			"  datum = kern_get_datum_tuple(ktoast->colmeta,htup,%d);\n"
			"  if (!datum)\n"
			"    ts_isnull[%d] = true;\n"
			"  else\n"
			"  {\n"
			"    ts_isnull[%d] = false;\n"
			"    ts_values[%d] = (Datum)\n"
			"      (((__global char *)datum -\n"
			"        (__global char *)&ktoast->hostptr) + ktoast->hostptr);\n"
			"  }\n",
			resno,
			resno - 1,
			resno - 1,
			resno - 1,
			resno - 1);
	}
	else if (!type_byval)
	{
		appendStringInfo(
			body,
			"\n"
			"  /* sortkey %d fixup */\n"
			"  if (!ts_isnull[%d])\n"
			"    ts_values[%d] += (Datum)ktoast->hostptr;\n",
			resno,
			resno - 1,
			resno - 1);
	}
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
			"  KVAR_X%d = pg_%s_vref(kds,ktoast,errcode,%d,x_index);\n"
			"  KVAR_Y%d = pg_%s_vref(kds,ktoast,errcode,%d,y_index);\n"
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
			colidx,
			i+1, dtype->type_name, colidx-1,
			i+1, dtype->type_name, colidx-1,
			i+1, i+1,
			dfunc->func_alias, i+1, i+1,
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
	appendStringInfo(&result, "%s\n",
					 pgstrom_codegen_func_declarations(context));
	/* make a projection function */
	appendStringInfo(
		&result,
		"static void\n"
		"gpusort_projection(__private cl_int *errcode,\n"
		"                   __global Datum *ts_values,\n"
		"                   __global cl_char *ts_isnull,\n"
		"                   __global kern_data_store *ktoast,\n"
		"                   __global HeapTupleHeaderData *htup)\n"
		"{\n"
		"  __global void *datum;\n"
		"%s"
		"}\n\n",
		pj_body.data);
	/* make a fixup-var function */
	appendStringInfo(
        &result,
		"static void\n"
		"gpusort_fixup_variables(__private cl_int *errcode,\n"
		"                        __global Datum *ts_values,\n"
		"                        __global cl_char *ts_isnull,\n"
		"                        __global kern_data_store *ktoast,\n"
		"                        __global HeapTupleHeaderData *htup)\n"
		"{\n"
		"  __global void *datum;\n"
		"%s"
		"}\n\n",
		fv_body.data);

	/* make a comparison function */
	appendStringInfo(
		&result,
		"static cl_int\n"
		"gpusort_keycomp(__private cl_int *errcode,\n"
		"                __global kern_data_store *kds,\n"
		"                __global kern_data_store *ktoast,\n"
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
	if (!pgstrom_enabled() || !enable_gpusort)
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
	cscan->scan.scanrelid       = 0;
	cscan->custom_ps_tlist      = NIL;
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
								  list_length(cscan->custom_ps_tlist) + 1,
								  tle->resname ? pstrdup(tle->resname) : NULL,
								  false);
		cscan->custom_ps_tlist = lappend(cscan->custom_ps_tlist, tle_new);
	}
	/* informs our preference to fetch tuples */
	if (IsA(subplan, CustomScan))
		((CustomScan *) subplan)->flags |= CUSTOMPATH_PREFERE_ROW_FORMAT;
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
	gs_info.varlena_keys = varlena_keys;
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
	gss->pds_toasts = palloc0(sizeof(pgstrom_data_store *) *
							  gss->num_chunks_limit);
	gss->chunk_size = gs_info->chunk_size;

	gss->numCols = gs_info->numCols;
	gss->sortColIdx = gs_info->sortColIdx;
	gss->sortOperators = gs_info->sortOperators;
	gss->collations = gs_info->collations;
	gss->nullsFirst = gs_info->nullsFirst;
	gss->varlena_keys = gs_info->varlena_keys;

	/* running status */
	gss->database_name = get_database_name(MyDatabaseId);
	memset(gss->sorted_chunks, 0, sizeof(gss->sorted_chunks));
	gss->num_gpu_running = 0;
	gss->num_cpu_running = 0;
	gss->num_cpu_pending = 0;
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
	pgstrom_data_store *pds_row = NULL;
	pgstrom_data_store *pds_slot = NULL;
	TupleTableSlot	   *slot;
	TupleDesc			tupdesc = CSS_GET_SCAN_TUPDESC(gss);
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
		if (!pds_row)
		{
			/* NOTE: maximum number of items is also restricted with
			 * max available consumption by tuple-store.
			 */
			nitems = (gss->chunk_size -
					  STROMALIGN(offsetof(kern_data_store,
										  colmeta[tupdesc->natts])))
				/ LONGALIGN((sizeof(Datum) +
							 sizeof(cl_char)) * tupdesc->natts);

			pds_row = pgstrom_create_data_store_row_fmap(tupdesc,
														 gss->chunk_size);
			pds_row->kds->nrooms = nitems;
			pgstrom_track_object(&pds_row->sobj, 0);
		}
		/* Insert this tuple to the data store */
		if (!pgstrom_data_store_insert_tuple(pds_row, slot))
		{
			gss->overflow_slot = slot;
			break;
		}
	}
	/* Did we read any tuples? */
	if (!pds_row)
		goto out;
	/* Expand file-mapped pds for space of tuple-slot also */
	nitems = pds_row->kds->nitems;
	pds_slot = pgstrom_extend_data_store_tupslot(pds_row, tupdesc, nitems);
	Assert(pds_slot->ktoast == pds_row);
	pgstrom_track_object(&pds_slot->sobj, 0);

	/* save this chunk on the global array */
	chunk_id = gss->num_chunks++;
	if (chunk_id >= gss->num_chunks_limit)
	{
		uint		new_limit = 2 * gss->num_chunks_limit;

		/* expand the array twice */
		gss->pds_chunks = repalloc(gss->pds_chunks,
								   sizeof(pgstrom_data_store *) * new_limit);
		gss->pds_toasts = repalloc(gss->pds_toasts,
								   sizeof(pgstrom_data_store *) * new_limit);
		gss->num_chunks_limit = new_limit;
	}
	gss->pds_chunks[chunk_id] = pds_slot;
	gss->pds_toasts[chunk_id] = pds_row;

	/* Make a pgstrom_gpusort object based on the data-store */
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
	gpusort->pds = pgstrom_get_data_store(pds_slot);
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

static dsm_segment *
form_pgstrom_cpusort(GpuSortState *gss,
					 pgstrom_cpusort *l_cpusort,
					 pgstrom_cpusort *r_cpusort)
{
	TupleDesc		tupdesc = CSS_GET_SCAN_TUPDESC(gss);
	Bitmapset	   *chunk_ids;
	StringInfoData	buf;
	Size			dsm_length;
	Size			kresults_offset;
	Size			length;
	long			nitems;
	int				index;
	dsm_segment	   *dsm_seg;
	pgstrom_flat_cpusort  pfc;
	pgstrom_flat_cpusort *pfc_buf;
	kern_resultbuf *kresults;

	memset(&pfc, 0, sizeof(pgstrom_flat_cpusort));
	initStringInfo(&buf);

	/* compute total number of items */
	kresults = PFCSEG_GET_KRESULTS(l_cpusort->oitems_dsm);
	Assert(kresults->nrels == 2);
	nitems = kresults->nitems;

	kresults = PFCSEG_GET_KRESULTS(r_cpusort->oitems_dsm);
	Assert(kresults->nrels == 2);
	nitems += kresults->nitems;

	/* DSM handles of litems_dsm and ritems_dsm */
	pfc.litems_dsmhnd = dsm_segment_handle(l_cpusort->oitems_dsm);
	pfc.ritems_dsmhnd = dsm_segment_handle(r_cpusort->oitems_dsm);

	/* Index of this process */
	pfc.master_proc = MyProc;

	/* name of target database */
	pfc.database_name = buf.len;
	length = strlen(gss->database_name) + 1;
	enlargeStringInfo(&buf, MAXALIGN(length));
	strcpy(buf.data + buf.len, gss->database_name);
	buf.len += MAXALIGN(length);

	/* existance of varlena sorting key */
	pfc.varlena_keys = gss->varlena_keys;
	/* length of kern_data_store array */
	pfc.max_chunk_id = gss->num_chunks;

	/* put filename of file-mapped kds and its length; being involved */
	pfc.kern_chunks = buf.len;
	chunk_ids = bms_union(l_cpusort->h.chunk_ids, r_cpusort->h.chunk_ids);
	while ((index = bms_first_member(chunk_ids)) >= 0)
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
	bms_free(chunk_ids);
	index = -1;		/* end of chunks marker */
	appendBinaryStringInfo(&buf, (const char *)&index, sizeof(int));

	/* tuple descriptor of sorting keys */
	pfc.tupdesc = buf.len;
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
	pfc.numCols = gss->numCols;
	/* sortColIdx */
	pfc.sortColIdx = buf.len;
	length = sizeof(AttrNumber) * gss->numCols;
	enlargeStringInfo(&buf, MAXALIGN(length));
	memcpy(buf.data + buf.len, gss->sortColIdx, length);
	buf.len += MAXALIGN(length);
	/* sortOperators */
	pfc.sortOperators = buf.len;
	length = sizeof(Oid) * gss->numCols;
	enlargeStringInfo(&buf, MAXALIGN(length));
    memcpy(buf.data + buf.len, gss->sortOperators, length);
	buf.len += MAXALIGN(length);
	/* collations */
	pfc.collations = buf.len;
	length = sizeof(Oid) * gss->numCols;
    enlargeStringInfo(&buf, MAXALIGN(length));
	memcpy(buf.data + buf.len, gss->collations, length);
	buf.len += MAXALIGN(length);
	/* nullsFirst */
	pfc.nullsFirst = buf.len;
	length = sizeof(bool) * gss->numCols;
	enlargeStringInfo(&buf, MAXALIGN(length));
    memcpy(buf.data + buf.len, gss->nullsFirst, length);
	buf.len += MAXALIGN(length);

	/* result buffer */
	kresults_offset = buf.len;
	enlargeStringInfo(&buf, MAXALIGN(sizeof(kern_resultbuf)) + 80);
	kresults = (kern_resultbuf *)(buf.data + buf.len);
	memset(kresults, 0, sizeof(kern_resultbuf));
	kresults->nrels = 2;
	kresults->nrooms = nitems;
	kresults->nitems = nitems;
	kresults->errcode = ERRCODE_INTERNAL_ERROR;
	snprintf((char *)kresults->results, sizeof(cl_int) * 2 * nitems,
			 "An internal error on worker prior to DSM attachment");
	buf.len += MAXALIGN(sizeof(kern_resultbuf)) + 80;

	/* allocation and setup of DSM */
	dsm_length = MAXALIGN(offsetof(pgstrom_flat_cpusort, data)) +
		MAXALIGN(kresults_offset) +
		MAXALIGN(offsetof(kern_resultbuf, results[2 * nitems]));
	pfc.dsm_length = dsm_length;
	pfc.kresults_offset = kresults_offset;

	dsm_seg = dsm_create(dsm_length, 0);
	pfc_buf = dsm_segment_address(dsm_seg);
	memcpy(pfc_buf, &pfc, sizeof(pgstrom_flat_cpusort));
	memcpy(pfc_buf->data, buf.data, buf.len);

	/* release temp buffer */
	pfree(buf.data);

	return dsm_seg;
}

static pgstrom_cpusort *
deform_pgstrom_cpusort(dsm_segment *dsm_seg)
{
	pgstrom_cpusort *cpusort = palloc0(sizeof(pgstrom_cpusort));
	pgstrom_flat_cpusort *pfc = dsm_segment_address(dsm_seg);
	char	   *pos = pfc->data;
	TupleDesc	tupdesc;
	int			index;

	/* input/output result buffers */
	cpusort->litems_dsm = dsm_attach(pfc->litems_dsmhnd);
	if (!cpusort->litems_dsm)
		elog(ERROR, "failed to attach dsm segment: %u", pfc->litems_dsmhnd);
	cpusort->w.l_kresults = PFCSEG_GET_KRESULTS(cpusort->litems_dsm);

	cpusort->ritems_dsm = dsm_attach(pfc->ritems_dsmhnd);
    if (!cpusort->ritems_dsm)
        elog(ERROR, "failed to attach dsm segment: %u", pfc->ritems_dsmhnd);
	cpusort->w.r_kresults = PFCSEG_GET_KRESULTS(cpusort->ritems_dsm);

	cpusort->oitems_dsm = dsm_seg;	/* myself */
	cpusort->w.o_kresults = PFCSEG_GET_KRESULTS(cpusort->oitems_dsm);

	/* database connection */
	cpusort->w.database_name = pfc->data + pfc->database_name;

	/* existence of varlena sorting keys */
	cpusort->w.varlena_keys = pfc->varlena_keys;

	/* size of kern_data_store array */
	cpusort->w.max_chunk_id = pfc->max_chunk_id;
	cpusort->w.kern_chunks = palloc0(sizeof(kern_data_store *) *
									 cpusort->w.max_chunk_id);
	cpusort->w.kern_toasts = palloc0(sizeof(kern_data_store *) *
									 cpusort->w.max_chunk_id);
	/* chunks to be mapped */
	pos = pfc->data + pfc->kern_chunks;
	while (true)
	{
		Size		kds_length;
		Size		kds_offset;
		const char *kds_fname;
		int			kds_fdesc;
		char	   *map_addr;

		index = *((int *) pos);
		pos += sizeof(int);
		if (index < 0)
			break;		/* end of chunks that are involved */
		//Assert(index < cpusort->w.max_chunk_id);
		if (index >= cpusort->w.max_chunk_id)
			elog(ERROR, "Bug? chunk-id is out of range %d for %d",
				 index, cpusort->w.max_chunk_id);

		kds_length = *((size_t *) pos);
		pos += sizeof(size_t);
		kds_offset = *((size_t *) pos);
		pos += sizeof(size_t);
		kds_fname = pos;
		pos += MAXALIGN(strlen(kds_fname) + 1);

		kds_fdesc = open(kds_fname, O_RDWR, 0);
		if (kds_fdesc < 0)
			elog(ERROR, "failed to open \"%s\": %m", kds_fname);

		if (cpusort->w.varlena_keys)
		{
			Assert(kds_offset >= BLCKSZ);
			map_addr = mmap(NULL, kds_offset + kds_length,
							PROT_READ | PROT_WRITE,
							MAP_SHARED 
#ifdef MAP_POPULATE
			   				| MAP_POPULATE 
#endif
							,
							kds_fdesc, 0);
			if (map_addr == MAP_FAILED)
			{
				close(kds_fdesc);
				elog(ERROR, "failed to mmap \"%s\": %m", kds_fname);
			}
			cpusort->w.kern_toasts[index] = (kern_data_store *)map_addr;
			cpusort->w.kern_chunks[index] =
				(kern_data_store *)(map_addr + kds_offset);
		}
		else
		{
			map_addr = mmap(NULL, kds_length,
							PROT_READ | PROT_WRITE,
							MAP_SHARED 
#ifdef MAP_POPULATE
			   				| MAP_POPULATE 
#endif
							,
							kds_fdesc, kds_offset);
			if (map_addr == MAP_FAILED)
			{
				close(kds_fdesc);
				elog(ERROR, "failed to mmap \"%s\": %m", kds_fname);
			}
			cpusort->w.kern_toasts[index] = NULL;
            cpusort->w.kern_chunks[index] = (kern_data_store *)map_addr;
		}
		close(kds_fdesc);	/* close on unmap */
	}
	/* tuple descriptor */
	tupdesc = (TupleDesc)(pfc->data + pfc->tupdesc);
	cpusort->w.tupdesc = palloc0(sizeof(*tupdesc));
	cpusort->w.tupdesc->natts = tupdesc->natts;
	cpusort->w.tupdesc->attrs =
		palloc0(ATTRIBUTE_FIXED_PART_SIZE * tupdesc->natts);
	cpusort->w.tupdesc->tdtypeid = tupdesc->tdtypeid;
	cpusort->w.tupdesc->tdtypmod = tupdesc->tdtypmod;
	cpusort->w.tupdesc->tdhasoid = tupdesc->tdhasoid;
	cpusort->w.tupdesc->tdrefcount = -1;	/* not counting */
	pos += MAXALIGN(sizeof(*tupdesc));

	for (index=0; index < tupdesc->natts; index++)
	{
		cpusort->w.tupdesc->attrs[index] = (Form_pg_attribute) pos;
		pos += MAXALIGN(ATTRIBUTE_FIXED_PART_SIZE);
	}
	/* sorting keys */
	cpusort->w.numCols = pfc->numCols;
	cpusort->w.sortColIdx = (AttrNumber *)(pfc->data + pfc->sortColIdx);
	cpusort->w.sortOperators = (Oid *)(pfc->data + pfc->sortOperators);
	cpusort->w.collations = (Oid *)(pfc->data + pfc->collations);
	cpusort->w.nullsFirst = (bool *)(pfc->data + pfc->nullsFirst);

	return cpusort;
}

/*
 * gpusort_fallback_gpu_chunks
 *
 * Fallback routine towards a particular GpuSort chunk, if GPU device
 * gave up sorting on device side.
 */
static inline int
gpusort_fallback_compare(GpuSortState *gss,
						 kern_data_store *kds, cl_uint index,
						 Datum *p_values, bool *p_isnull)
{
	Datum	   *x_values = (Datum *) KERN_DATA_STORE_VALUES(kds, index);
	bool	   *x_isnull = (bool *) KERN_DATA_STORE_ISNULL(kds, index);
	int			i, comp;

	for (i=0; i < gss->numCols; i++)
	{
		SortSupport		ssup = gss->ssup_keys + i;

		comp = ApplySortComparator(x_values[i],
								   x_isnull[i],
								   p_values[i],
								   p_isnull[i],
								   ssup);
		if (comp != 0)
			return comp;
	}
	return 0;
}

static void
gpusort_fallback_quicksort(GpuSortState *gss,
						   kern_resultbuf *kresults,
						   kern_data_store *kds,
						   cl_uint l_index, cl_uint r_index)
{
	if (l_index < r_index)
	{
		cl_uint		i = l_index;
		cl_uint		j = r_index;
		cl_uint		p_index = (l_index + r_index) / 2;
		Datum	   *p_values = (Datum *) KERN_DATA_STORE_VALUES(kds, p_index);
		bool	   *p_isnull = (bool *) KERN_DATA_STORE_ISNULL(kds, p_index);
		int			temp;

		while (true)
		{
			while (gpusort_fallback_compare(gss, kds, i,
											p_values, p_isnull) < 0)
				i++;
			while (gpusort_fallback_compare(gss, kds, j,
											p_values, p_isnull) > 0)
				j--;
			if (i >= j)
				break;
			/* swap index */
			temp = kresults->results[2 * i + 1];
			kresults->results[2 * i + 1] = kresults->results[2 * j + 1];
			kresults->results[2 * j + 1] = temp;
			/* make index advanced */
			i++;
			j--;
		}
		gpusort_fallback_quicksort(gss, kresults, kds, l_index, i - 1);
		gpusort_fallback_quicksort(gss, kresults, kds, j + 1, r_index);
	}
}

static void
gpusort_fallback_gpu_chunks(GpuSortState *gss, pgstrom_gpusort *gpusort)
{
	SortSupportData	   *ssup_keys = gss->ssup_keys;
	kern_resultbuf	   *kresults = KERN_GPUSORT_RESULTBUF(&gpusort->kern);
	pgstrom_data_store *pds = gpusort->pds;
	kern_data_store	   *kds = pds->kds;
	kern_data_store	   *ktoast = pds->ktoast->kds;
	TupleTableSlot	   *slot;
	Datum			   *tts_values;
	bool			   *tts_isnull;
	cl_uint				i, j, nitems = kds->nitems;

	/* initialize SortSupportData, if first time */
	if (!ssup_keys)
	{
		EState	   *estate = gss->css.ss.ps.state;
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
	slot = gss->css.ss.ss_ScanTupleSlot;
	for (i=0; i < nitems; i++)
	{
		kresults->results[2 * i] = gpusort->chunk_id;
		kresults->results[2 * i + 1] = i;

		tts_values = (Datum *) KERN_DATA_STORE_VALUES(kds, i);
		tts_isnull = (bool *) KERN_DATA_STORE_ISNULL(kds, i);

		if (!kern_fetch_data_store(slot, ktoast, i, NULL))
			elog(ERROR, "failed to fetch a tuple from data-store");
		slot_getallattrs(slot);

		/* NOTE: varlena is store as host accessable pointer */
		for (j=0; j < gss->numCols; j++)
		{
			SortSupport		ssup = ssup_keys + i;

			tts_values[j] = slot->tts_values[ssup->ssup_attno - 1];
            tts_isnull[j] = slot->tts_isnull[ssup->ssup_attno - 1];
		}
	}
	/* fallback execution with QuickSort */
	gpusort_fallback_quicksort(gss, kresults, pds->kds, 0, kds->nitems - 1);

	/* varlena datum should be offset from ktoast, as if GPU doing */
	if (gss->varlena_keys)
	{
		TupleDesc	tupdesc
			= gss->css.ss.ss_ScanTupleSlot->tts_tupleDescriptor;

		for (i=0; i < nitems; i++)
		{
			tts_values = (Datum *) KERN_DATA_STORE_VALUES(kds, i);
			tts_isnull = (bool *) KERN_DATA_STORE_ISNULL(kds, i);

			for (j=0; j < gss->numCols; j++)
			{
				SortSupport		ssup = ssup_keys + i;
				Form_pg_attribute attr = tupdesc->attrs[ssup->ssup_attno - 1];

				if (!tts_isnull[j] && attr->attlen < 0)
				{
					Assert(tts_values[j] > (uintptr_t)&ktoast->hostptr);
					tts_values[j] -= (uintptr_t)&ktoast->hostptr;
				}
			}
		}
	}

	/* restore error status */
	kresults->errcode = StromError_Success;
	kresults->nitems = kds->nitems;
}

static pgstrom_cpusort *
gpusort_merge_cpu_chunks(GpuSortState *gss, pgstrom_cpusort *cpusort_1)
{
	pgstrom_cpusort	   *cpusort_2;
	dsm_segment		   *dsm_seg;
	cl_uint				mc_class = cpusort_1->h.mc_class;
	int					i, n, x, y;

	Assert(CurrentMemoryContext == gss->css.ss.ps.state->es_query_cxt);

	/* ritems and litems are no longer referenced */
	if (cpusort_1->ritems_dsm)
	{
		dsm_detach(cpusort_1->ritems_dsm);
		cpusort_1->ritems_dsm = NULL;
	}
	if (cpusort_1->litems_dsm)
	{
		dsm_detach(cpusort_1->litems_dsm);
		cpusort_1->litems_dsm = NULL;
	}
	/* also, bgw_handle is no longer referenced */
	if (cpusort_1->h.bgw_handle)
	{
		pfree(cpusort_1->h.bgw_handle);
		cpusort_1->h.bgw_handle = NULL;
	}

	/*
	 * In case of no buddy at this moment, we have to wait for the next
	 * chunk to be merged. Or, this chunk might be the final result.
	 */
	Assert(mc_class < MAX_MERGECHUNKS_CLASS);
	if (!gss->sorted_chunks[mc_class])
	{
		dlist_iter		iter;
		pgstrom_cpusort *temp;
		bool			has_smaller;

		/* once cpusort_1 is put as partially sorted chunk */
		gss->sorted_chunks[mc_class] = cpusort_1;

		/*
		 * Unless scan of outer relation does not completed, we try to
		 * keep merging chunks with same size.
		 */
		if (!gss->scan_done)
			return NULL;

		/*
		 * Once we forget the given cpusort_1, and find out the smallest
		 * one that is now waiting for merging.
		 */
		for (i=0; i < MAX_MERGECHUNKS_CLASS; i++)
		{
			if (gss->sorted_chunks[i])
			{
				cpusort_1 = gss->sorted_chunks[i];
				Assert(cpusort_1->h.mc_class == i);
				break;
			}
		}
		mc_class = cpusort_1->h.mc_class;

		/*
		 * If we have any running or pending chunks with same merge-chunk
		 * class, it may be a good candidate to merge.
		 */
		has_smaller = (gss->num_gpu_running > 0 ? true : false);
		dlist_foreach (iter, &gss->running_cpu_chunks)
		{
			temp = dlist_container(pgstrom_cpusort, chain, iter.cur);
			if (temp->h.mc_class == cpusort_1->h.mc_class)
				return NULL;
			else if (temp->h.mc_class < cpusort_1->h.mc_class)
				has_smaller = true;
		}
		dlist_foreach (iter, &gss->pending_cpu_chunks)
		{
			temp = dlist_container(pgstrom_cpusort, chain, iter.cur);
			if (temp->h.mc_class == cpusort_1->h.mc_class)
				return NULL;
			else if (temp->h.mc_class < cpusort_1->h.mc_class)
				has_smaller = true;
		}

		/* wait until smaller chunk gets sorted, if any */
		if (has_smaller)
			return NULL;

		/*
		 * Elsewhere, picks up a pair of smallest two chunks that are
		 * already sorted, to merge them on next step.
		 */
		cpusort_2 = NULL;
		for (i=cpusort_1->h.mc_class + 1; i < MAX_MERGECHUNKS_CLASS; i++)
		{
			if (gss->sorted_chunks[i])
			{
				cpusort_2 = gss->sorted_chunks[i];
				Assert(cpusort_2->h.mc_class == i);
				break;
			}
		}

		/* no merginable pair found? */
		if (!cpusort_2)
			return NULL;
		/* OK, let's merge this two chunks */
		gss->sorted_chunks[cpusort_1->h.mc_class] = NULL;
		gss->sorted_chunks[cpusort_2->h.mc_class] = NULL;
	}
	else
	{
		cpusort_2 = gss->sorted_chunks[mc_class];
		gss->sorted_chunks[mc_class] = NULL;
	}
	Assert(cpusort_2->oitems_dsm != NULL
		   && !cpusort_2->ritems_dsm
		   && !cpusort_2->litems_dsm);

	dsm_seg = form_pgstrom_cpusort(gss, cpusort_1, cpusort_2);
	cpusort_2->litems_dsm = cpusort_1->oitems_dsm;
	cpusort_2->ritems_dsm = cpusort_2->oitems_dsm;
	cpusort_2->oitems_dsm = dsm_seg;
	cpusort_2->h.bgw_handle = NULL;
	x = bms_num_members(cpusort_1->h.chunk_ids);
	y = bms_num_members(cpusort_2->h.chunk_ids);
	cpusort_2->h.chunk_ids = bms_union(cpusort_1->h.chunk_ids,
									   cpusort_2->h.chunk_ids);
	n = bms_num_members(cpusort_2->h.chunk_ids);
	cpusort_2->h.mc_class = get_next_log2(n);
	elog(DEBUG1, "CpuSort merge: %d + %d => %d (class: %d)", x, y, n,
		 cpusort_2->h.mc_class);
	/* release either of them */
	pfree(cpusort_1);

	return cpusort_2;
}

static pgstrom_cpusort *
gpusort_merge_gpu_chunks(GpuSortState *gss, pgstrom_gpusort *gpusort)
{
	pgstrom_cpusort	   *cpusort;
	pgstrom_flat_cpusort *pfc;
	EState			   *estate = gss->css.ss.ps.state;
	MemoryContext		oldcxt;
	kern_resultbuf	   *kresults_gpu;
	Size				kresults_len;
	Size				dsm_length;

	oldcxt = MemoryContextSwitchTo(estate->es_query_cxt);

	/*
	 * Transform the supplied gpusort to the simplified cpusort form
	 */
	cpusort = palloc0(sizeof(pgstrom_cpusort));
	kresults_gpu = KERN_GPUSORT_RESULTBUF(&gpusort->kern);
	kresults_len = MAXALIGN(offsetof(kern_resultbuf,
									 results[2 * kresults_gpu->nitems]));
	dsm_length = MAXALIGN(offsetof(pgstrom_flat_cpusort,
								   data[kresults_len]));
	cpusort->oitems_dsm = dsm_create(dsm_length, 0);
	cpusort->litems_dsm = NULL;
	cpusort->ritems_dsm = NULL;
	cpusort->h.bgw_handle = NULL;
	cpusort->h.mc_class = 0;
	cpusort->h.chunk_ids = bms_make_singleton(gpusort->chunk_id);

	/* make dummy pgstrom_flat_cpusort */
	pfc = dsm_segment_address(cpusort->oitems_dsm);
	memset(pfc, 0, sizeof(pgstrom_flat_cpusort));
	pfc->dsm_length = dsm_length;
	pfc->kresults_offset = 0;
	memcpy(pfc->data, kresults_gpu, kresults_len);

	/* pds is still valid, but gpusort is no longer referenced */
	pgstrom_untrack_object(&gpusort->msg.sobj);
	pgstrom_put_message(&gpusort->msg);

	/*
	 * Finally, merge this cpusort chunk with a chunk preliminary sorted.
	 */
	cpusort = gpusort_merge_cpu_chunks(gss, cpusort);

	MemoryContextSwitchTo(oldcxt);

	return cpusort;
}

static void
gpusort_check_gpu_tasks(GpuSortState *gss)
{
	pgstrom_message	   *msg;
	pgstrom_gpusort	   *gpusort;
	pgstrom_cpusort	   *cpusort;

	/*
	 * Check status of GPU co-routine
	 */
	while ((msg = pgstrom_try_dequeue_message(gss->mqueue)) != NULL)
	{
		Assert(gss->num_gpu_running > 0);
		gss->num_gpu_running--;

		Assert(StromTagIs(msg, GpuSort));
		gpusort = (pgstrom_gpusort *) msg;

		if (msg->errcode != StromError_Success)
		{
			if (msg->errcode == StromError_CpuReCheck)
			{
				/* GPU raised an error, run single chunk sort by CPU */
				gpusort_fallback_gpu_chunks(gss, gpusort);
			}
			else if (msg->errcode == CL_BUILD_PROGRAM_FAILURE)
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
		{
			dlist_push_tail(&gss->pending_cpu_chunks, &cpusort->chain);
			gss->num_cpu_pending++;
		}
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
		MemoryContext		oldcxt;

		cpusort = dlist_container(pgstrom_cpusort, chain, iter.cur);
		status = GetBackgroundWorkerPid(cpusort->h.bgw_handle, &bgw_pid);

		/* emergency. kills all the backend and raise an error */
		if (status == BGWH_STARTED ||
			status == BGWH_STOPPED)
		{
			pgstrom_flat_cpusort *pfc = (pgstrom_flat_cpusort *)
				dsm_segment_address(cpusort->oitems_dsm);

			/* not yet finished? */
			if (status == BGWH_STARTED && !pfc->cpusort_worker_done)
				continue;

			dlist_delete(&cpusort->chain);
			memset(&cpusort->chain, 0, sizeof(dlist_node));
			gss->num_cpu_running--;

			/*
			 * check the result status in the dynamic worker
			 *
			 * note: we intend dynamic worker returns error code and
			 * message on the kern_resultbuf if error happen.
			 * we may need to make a common format later.
			 */
			kresults = PFCSEG_GET_KRESULTS(cpusort->oitems_dsm);
			if (kresults->errcode != ERRCODE_SUCCESSFUL_COMPLETION)
				ereport(ERROR,
						(errcode(kresults->errcode),
						 errmsg("GpuSort worker: %s",
								(char *)kresults->results)));

			/* accumlate performance counter */
			if (cpusort->pfm.enabled)
			{
				struct timeval tv;

				gettimeofday(&tv, NULL);
				cpusort->pfm.time_cpu_sort_real
					+= timeval_diff(&cpusort->pfm.tv, &tv);
				cpusort->pfm.time_cpu_sort = pfc->time_cpu_sort;
				pgstrom_perfmon_add(&gss->pfm, &cpusort->pfm);
			}

			oldcxt = MemoryContextSwitchTo(gss->css.ss.ps.state->es_query_cxt);
			cpusort = gpusort_merge_cpu_chunks(gss, cpusort);
			if (cpusort)
			{
				dlist_push_tail(&gss->pending_cpu_chunks, &cpusort->chain);
				gss->num_cpu_pending++;
			}
			MemoryContextSwitchTo(oldcxt);
		}
		else if (status == BGWH_POSTMASTER_DIED)
		{
			/* TODO: kills another running workers */
			elog(ERROR, "Bug? Postmaster or BGWorker crashed, bogus status");
		}
	}
}

static void
gpusort_kick_pending_tasks(GpuSortState *gss)
{
	BackgroundWorker	worker;
	pgstrom_cpusort	   *cpusort;
	dlist_node		   *dnode;
	dsm_handle 			dsm_hnd;

	while (!dlist_is_empty(&gss->pending_cpu_chunks) &&
		   gss->num_cpu_running < gpusort_max_workers)
	{
		dnode = dlist_pop_head_node(&gss->pending_cpu_chunks);
		cpusort = dlist_container(pgstrom_cpusort, chain, dnode);
		dsm_hnd = dsm_segment_handle(cpusort->oitems_dsm);
		gss->num_cpu_pending--;

		/* init performance counter */
		if (gss->pfm.enabled)
		{
			memset(&cpusort->pfm, 0, sizeof(pgstrom_perfmon));
			cpusort->pfm.enabled = gss->pfm.enabled;
			gettimeofday(&cpusort->pfm.tv, NULL);
		}

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
		worker.bgw_main_arg = PointerGetDatum(dsm_hnd);

		if (!RegisterDynamicBackgroundWorker(&worker, &cpusort->h.bgw_handle))
		{
			dlist_push_head(&gss->pending_cpu_chunks, &cpusort->chain);
			gss->num_cpu_pending++;
		}
		else
		{
			dlist_push_tail(&gss->running_cpu_chunks, &cpusort->chain);
			gss->num_cpu_running++;
			gss->pfm.num_cpu_sort++;
		}
	}
}

static void
gpusort_exec_sort(GpuSortState *gss)
{
	pgstrom_gpusort	   *gpusort;
	pgstrom_cpusort	   *cpusort;
	int					i, rc;

	while (true)
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
		 * be chained to gss->sorted_chunks or moved to pending_chunks if
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
			struct timeval tv1, tv2;

			if (gss->num_gpu_running == 0 &&
				gss->num_cpu_running == 0 &&
				dlist_is_empty(&gss->pending_cpu_chunks))
				break;

			if (gss->num_gpu_running > 0 || gss->num_cpu_running > 0)
				timeout = 5 * 1000L;	/* 5sec */
			else
				timeout =       400L;	/* 0.4sec */

			if (gss->pfm.enabled)
				gettimeofday(&tv1, NULL);

			rc = WaitLatch(&MyProc->procLatch,
						   WL_LATCH_SET | WL_TIMEOUT | WL_POSTMASTER_DEATH,
						   timeout);
			ResetLatch(&MyProc->procLatch);

			if (gss->pfm.enabled)
			{
				gettimeofday(&tv2, NULL);
				gss->pfm.time_bgw_sync += timeval_diff(&tv1, &tv2);
			}
			/* emergency bailout if postmaster has died */
			if (rc & WL_POSTMASTER_DEATH)
				elog(ERROR, "failed on WaitLatch due to Postmaster die");
		}
	}
	/* OK, data was sorted */
	gss->sort_done = true;
	elog(DEBUG1, "Sort done");

	/*
	 * Once we got the sorting completed, just one chunk should be attached
	 * on the gss->sorted_chunk. If NULL, it means outer relation has no
	 * rows anywhere.
	 */
	for (cpusort=NULL, i=0; i < MAX_MERGECHUNKS_CLASS; i++)
	{
		if (gss->sorted_chunks[i])
		{
			if (cpusort)
				elog(ERROR, "Bug? multiple chunk fraction still remain");
			cpusort = gss->sorted_chunks[i];
			Assert(cpusort->h.mc_class == i);
		}
	}
	if (!cpusort)
		gss->sorted_result = NULL;
	else
	{
		gss->sorted_result = cpusort->oitems_dsm;
		gss->sorted_chunks[cpusort->h.mc_class] = NULL;
		pfree(cpusort);
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
	{
		bool	save_set_latch_on_sigusr1 = set_latch_on_sigusr1;

		PG_TRY();
		{
			gpusort_exec_sort(gss);
		}
		PG_CATCH();
		{
			set_latch_on_sigusr1 = save_set_latch_on_sigusr1;
			PG_RE_THROW();
		}
		PG_END_TRY();
		set_latch_on_sigusr1 = save_set_latch_on_sigusr1;
	}
	Assert(gss->num_chunks == 0 || gss->sorted_result != NULL);
	/* Does outer relation has any rows to read? */
	if (!gss->sorted_result)
		return NULL;

	kresults = PFCSEG_GET_KRESULTS(gss->sorted_result);

	if (gss->sorted_index < kresults->nitems)
	{
		cl_long		index = 2 * gss->sorted_index;
		cl_int		chunk_id = kresults->results[index];
		cl_int		item_id = kresults->results[index + 1];

		if (chunk_id >= gss->num_chunks || !gss->pds_chunks[chunk_id])
			elog(ERROR, "Bug? data-store of GpuSort missing (chunk-id: %d)",
				 chunk_id);
		if ((gss->css.flags & CUSTOMPATH_PREFERE_ROW_FORMAT) == 0)
			pds = gss->pds_chunks[chunk_id];
		else
			pds = gss->pds_toasts[chunk_id];

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
#ifdef USE_ASSERT_CHECKING
	for (i=0; i < MAX_MERGECHUNKS_CLASS; i++)
		Assert(!gss->sorted_chunks[i]);
#endif

	if (gss->sorted_result)
		dsm_detach(gss->sorted_result);

	for (i=0; i < gss->num_chunks; i++)
	{
		pgstrom_data_store *pds = gss->pds_chunks[i];
		pgstrom_data_store *ptoast = gss->pds_toasts[i];
		pgstrom_untrack_object(&pds->sobj);
		pgstrom_put_data_store(pds);
		pgstrom_untrack_object(&ptoast->sobj);
		pgstrom_put_data_store(ptoast);
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
#ifdef USE_ASSERT_CHECKING
		for (i=0; i < MAX_MERGECHUNKS_CLASS; i++)
			Assert(!gss->sorted_chunks[i]);
#endif

		if (gss->sorted_result)
		{
			dsm_detach(gss->sorted_result);
			gss->sorted_result = NULL;
		}

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
gpusort_mark_pos(CustomScanState *node)
{
	GpuSortState   *gss = (GpuSortState *) node;

	if (gss->sort_done)
	{
		gss->markpos_index = gss->sorted_index;
	}
}

static void
gpusort_restore_pos(CustomScanState *node)
{
	GpuSortState   *gss = (GpuSortState *) node;

	if (gss->sort_done)
	{
		Assert(gss->markpos_index >= 0);
		gss->sorted_index = gss->markpos_index;
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
	if (es->analyze && gss->sort_done && gss->sorted_result)
	{
		const char *sort_method;
		const char *sort_storage;
		char		sort_resource[128];
		Size		total_consumption;

		if (gss->num_chunks > 1)
			sort_method = "GPU/Bitonic + CPU/Merge";
		else
			sort_method = "GPU/Bitonic";

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
	show_custom_flags(&gss->css, es);
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
gpusort_exec_cpusort(pgstrom_cpusort *cpusort)
{
	SortSupportData	   *sort_keys;
	kern_data_store	   *kds;
	kern_data_store	   *ltoast = NULL;
	kern_data_store	   *rtoast = NULL;
	kern_resultbuf	   *litems = cpusort->w.l_kresults;
	kern_resultbuf	   *ritems = cpusort->w.r_kresults;
	kern_resultbuf	   *oitems = cpusort->w.o_kresults;
	Datum			   *rts_values = NULL;
	Datum			   *lts_values = NULL;
	cl_char			   *rts_isnull = NULL;
	cl_char			   *lts_isnull = NULL;
	TupleDesc			tupdesc = cpusort->w.tupdesc;
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
	sort_keys = palloc0(sizeof(SortSupportData) * cpusort->w.numCols);
	for (i=0; i < cpusort->w.numCols; i++)
	{
		SortSupport	ssup = sort_keys + i;

		ssup->ssup_cxt = CurrentMemoryContext;
		ssup->ssup_collation = cpusort->w.collations[i];
		ssup->ssup_nulls_first = cpusort->w.nullsFirst[i];
		ssup->ssup_attno = cpusort->w.sortColIdx[i];
		PrepareSortSupportFromOrderingOp(cpusort->w.sortOperators[i], ssup);
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
			Assert(lchunk_id < cpusort->w.max_chunk_id);
			kds = cpusort->w.kern_chunks[lchunk_id];
			Assert(litem_id < kds->nitems);
			if (litem_id >= kds->nitems)
				elog(ERROR, "item-id %u is out of range in the chunk %u",
					 litem_id, lchunk_id);
			lts_values = KERN_DATA_STORE_VALUES(kds, litem_id);
		    lts_isnull = KERN_DATA_STORE_ISNULL(kds, litem_id);
			ltoast = cpusort->w.kern_toasts[lchunk_id];
		}

		if (!rts_values)
		{
			rchunk_id = ritems->results[2 * rindex];
			ritem_id = ritems->results[2 * rindex + 1];
			Assert(rchunk_id < cpusort->w.max_chunk_id);
			kds = cpusort->w.kern_chunks[rchunk_id];
			if (ritem_id >= kds->nitems)
				elog(ERROR, "item-id %u is out of range in the chunk %u",
					 ritem_id, rchunk_id);
			rts_values = KERN_DATA_STORE_VALUES(kds, ritem_id);
			rts_isnull = KERN_DATA_STORE_ISNULL(kds, ritem_id);
			rtoast = cpusort->w.kern_toasts[rchunk_id];
		}

		for (i=0; i < cpusort->w.numCols; i++)
		{
			SortSupport ssup = sort_keys + i;
			AttrNumber	anum = ssup->ssup_attno - 1;
			Datum		r_value = rts_values[anum];
			bool		r_isnull = rts_isnull[anum];
			Datum		l_value = lts_values[anum];
			bool		l_isnull = lts_isnull[anum];
			Form_pg_attribute attr = tupdesc->attrs[anum];

			/* toast datum has to be fixed up */
			if (attr->attlen < 0)
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
gpusort_entrypoint_cpusort(Datum main_arg)
{
	MemoryContext		cpusort_mcxt;
	dsm_handle			dsm_hnd = (dsm_handle) main_arg;
	dsm_segment		   *dsm_seg;
	PGPROC			   *master;
	pgstrom_cpusort	   *cpusort;
	pgstrom_flat_cpusort *pfc;
	kern_resultbuf	   *kresults;
	struct timeval		tv1, tv2;

	/* We're now ready to receive signals */
	BackgroundWorkerUnblockSignals();
	/* Makes up resource owner and memory context */
	Assert(CurrentResourceOwner == NULL);
	CurrentResourceOwner = ResourceOwnerCreate(NULL, "CpuSort");
	cpusort_mcxt = AllocSetContextCreate(TopMemoryContext,
										 "CpuSort",
										 ALLOCSET_DEFAULT_MINSIZE,
										 ALLOCSET_DEFAULT_INITSIZE,
										 ALLOCSET_DEFAULT_MAXSIZE);
	CurrentMemoryContext = cpusort_mcxt;
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
	pfc = dsm_segment_address(dsm_seg);
	kresults = (kern_resultbuf *)(pfc->data + pfc->kresults_offset);
	kresults->errcode = 0;

	/* get reference to the master process */
	master = pfc->master_proc;

	PG_TRY();
	{
		/* deform CpuSort request to the usual format */
		cpusort = deform_pgstrom_cpusort(dsm_seg);

		/* Connect to our database */
		BackgroundWorkerInitializeConnection(cpusort->w.database_name, NULL);

		/*
		 * XXX - Eventually, we should use parallel-context to share
		 * the transaction snapshot, initialize misc stuff and so on.
		 * But just at this moment, we create a new transaction state
		 * to simplifies the implementation.
		 */
		StartTransactionCommand();
		PushActiveSnapshot(GetTransactionSnapshot());

		gettimeofday(&tv1, NULL);

		/* handle CPU merge sorting */
		gpusort_exec_cpusort(cpusort);

		gettimeofday(&tv2, NULL);

		/* we should have no side-effect */
		PopActiveSnapshot();
		CommitTransactionCommand();
	}
	PG_CATCH();
	{
		MemoryContext	ecxt = MemoryContextSwitchTo(cpusort_mcxt);
		ErrorData	   *edata = CopyErrorData();
		Size			buflen;

		kresults->errcode = edata->sqlerrcode;
		buflen = sizeof(cl_int) * kresults->nrels * kresults->nrooms;
		snprintf((char *)kresults->results, buflen, "%s (%s, %s:%d)",
				 edata->message, edata->funcname,
				 edata->filename, edata->lineno);
		MemoryContextSwitchTo(ecxt);

		SetLatch(&master->procLatch);

		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Inform the corrdinator worker got finished */
	pfc->time_cpu_sort = timeval_diff(&tv1, &tv2);
	pfc->cpusort_worker_done = true;
	pg_memory_barrier();
	SetLatch(&master->procLatch);
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
	kern_data_store	*ktoast;
	char		   *map_addr;
	size_t			map_length;
	cl_command_queue kcmdq;
	cl_program		program;
	cl_int			dindex;
	cl_mem			m_gpusort;
	cl_mem			m_kds;
	cl_mem			m_ktoast;
	cl_kernel		kern_prep;
	cl_kernel		kern_fixds;
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
		cl_ulong	last = 0;

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
		last = 0;
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

			if (last != 0)
				gpusort->msg.pfm.time_debug1 += (temp - last) / 1000;

			rc = clGetEventProfilingInfo(clgss->events[i],
										 CL_PROFILING_COMMAND_END,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			tv_end = Max(tv_end, temp);
			last = temp;
		}
		gpusort->msg.pfm.time_gpu_sort += (tv_end - tv_start) / 1000;

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
	if (clgss->m_ktoast)
		clReleaseMemObject(clgss->m_ktoast);
	if (clgss->kern_prep)
		clReleaseKernel(clgss->kern_prep);
	if (clgss->kern_fixds)
		clReleaseKernel(clgss->kern_fixds);
	for (i=0; i < clgss->kern_sort_nums; i++)
		clReleaseKernel(clgss->kern_sort[i]);
	if (clgss->program && clgss->program != BAD_OPENCL_PROGRAM)
		clReleaseProgram(clgss->program);
	if (clgss->map_addr)
		munmap(clgss->map_addr, clgss->map_length);
	if (clgss->kern_sort)
		free(clgss->kern_sort);
	free(clgss);

	/* reply the result to backend side */
	pgstrom_reply_message(&gpusort->msg);
}

static cl_int
compute_bitonic_workgroup_size(clstate_gpusort *clgss, size_t nitems,
							   size_t *p_gwork_sz, size_t *p_lwork_sz)
{
	static struct {
		const char *kern_name;
		size_t		kern_lmem;
	} kern_calls[] = {
		{ "gpusort_bitonic_local", 2 * sizeof(cl_uint) },
		{ "gpusort_bitonic_merge",     sizeof(cl_uint) },
	};
	size_t		least_sz = (nitems + 1) / 2;
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
										  (nitems + 1) / 2,
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
	gwork_sz = (((nitems + 1) / 2 + lwork_sz - 1) / lwork_sz) * lwork_sz;

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
						1,		/* __global kern_data_store *kds */
						sizeof(cl_mem),
						&clgss->m_kds);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgss->kern_prep,
						2,		/* __global kern_data_store *ktoast */
						sizeof(cl_mem),
						&clgss->m_ktoast);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgss->kern_prep,
						3,		/* cl_int chunk_id */
						sizeof(cl_int),
						&gpusort->chunk_id);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgss->kern_prep,
						4,		/* KERN_DYNAMIC_LOCAL_WORKMEM_ARG */
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
						&clgss->m_ktoast);
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
    clgss->msg->pfm.num_gpu_sort++;

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
						&clgss->m_ktoast);
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
	clgss->msg->pfm.num_gpu_sort++;

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
						&clgss->m_ktoast);
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
	clgss->msg->pfm.num_gpu_sort++;

	return CL_SUCCESS;
}

/*
 * clserv_launch_gpusort_fixds
 *
 * launcher of:
 * __kernel void
 * gpusort_fixup_datastore(__global kern_gpusort *kgpusort,
 *                         __global kern_data_store *kds,
 *                         __global kern_data_store *ktoast,
 *                         KERN_DYNAMIC_LOCAL_WORKMEM_ARG)
 */
static cl_int
clserv_launch_gpusort_fixds(clstate_gpusort *clgss, size_t nitems)
{
	size_t		gwork_sz;
	size_t		lwork_sz;
	cl_int		rc;

	clgss->kern_prep = clCreateKernel(clgss->program,
									  "gpusort_fixup_datastore",
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
						1,		/* __global kern_data_store *kds */
						sizeof(cl_mem),
						&clgss->m_kds);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgss->kern_prep,
						2,		/* __global kern_data_store *ktoast */
						sizeof(cl_mem),
						&clgss->m_ktoast);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	rc = clSetKernelArg(clgss->kern_prep,
						3,		/* KERN_DYNAMIC_LOCAL_WORKMEM_ARG */
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
	clgss->ev_index++;
	clgss->msg->pfm.num_kern_prep++;

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
	size_t				gwork_sz = 0;
	size_t				lwork_sz = 0;
	size_t				nsteps;
	size_t				launches;
	size_t				i, j;
	int					map_fdesc;
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
	map_fdesc = open(pds->kds_fname, O_RDWR, 0);
	if (map_fdesc < 0)
	{
		clserv_log("failed to open \"%s\" :%m", pds->kds_fname);
		goto error;
	}
	clgss->map_length = pds->kds_offset + pds->kds_length;
	clgss->map_addr = mmap(NULL, clgss->map_length,
						   PROT_READ | PROT_WRITE,
						   MAP_SHARED
#ifdef MAP_POPULATE
			   | MAP_POPULATE 
#endif
						    ,
						   map_fdesc, 0);
	if (clgss->map_addr == MAP_FAILED)
	{
		clserv_log("failed to mmap \"%s\" :%m", pds->kds_fname);
		close(map_fdesc);
		goto error;
	}
	close(map_fdesc);	/* close on unmap */

	clgss->ktoast = (kern_data_store *) clgss->map_addr;
	clgss->kds = (kern_data_store *)(clgss->map_addr + pds->kds_offset);
	Assert(pds->kds_offset >= clgss->ktoast->length);
	Assert(pds->kds_length == clgss->kds->length);
	nitems = clgss->ktoast->nitems;

	/*
	 * choose a device to run
	 */
	clgss->dindex = pgstrom_opencl_device_schedule(&gpusort->msg);
	clgss->kcmdq = opencl_cmdq[clgss->dindex];

	/*
	 * construction of kernel buffer objects
	 *
	 * m_gpusort - control data of gpusort
	 * m_kds     - data store of row-flat format
	 * m_ktoast  - data store of tup-slot format
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

	length = KERN_DATA_STORE_LENGTH(clgss->ktoast);
	clgss->m_ktoast = clCreateBuffer(opencl_context,
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

	/* __global kern_data_store *kds (header only) */
	length = KERN_DATA_STORE_HEAD_LENGTH(clgss->kds);
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

	/* __global kern_data_store *ktoast */
	length = KERN_DATA_STORE_LENGTH(clgss->ktoast);
	rc = clEnqueueWriteBuffer(clgss->kcmdq,
							  clgss->m_ktoast,
							  CL_FALSE,
							  0,
							  length,
							  clgss->ktoast,
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
	rc = compute_bitonic_workgroup_size(clgss, nitems, &gwork_sz, &lwork_sz);
	if (rc != CL_SUCCESS)
		goto error;

	/* NOTE: nhalf is the least power of two value that is larger than or
	 * equal to half of the nitems.
	 */
	nhalf = 1UL << (get_next_log2(nitems + 1) - 1);
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
	for (i = lwork_sz; i < nhalf; i *= 2)
	{
		for (j = 2 * i; j > lwork_sz; j /= 2)
		{
			cl_uint		unitsz = 2 * j;
			bool		reversing = (j == 2 * i) ? true : false;
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
	 * fixup special internal format (like, numeric), if needed
	 */
	rc = clserv_launch_gpusort_fixds(clgss, nitems);
	if (rc != CL_SUCCESS)
		goto error;

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

	length = KERN_DATA_STORE_LENGTH(clgss->kds);
	rc = clEnqueueReadBuffer(clgss->kcmdq,
							 clgss->m_kds,
							 CL_FALSE,
							 0,
							 length,
							 (char *)clgss->kds,
							 1,
							 &clgss->events[clgss->ev_index - 1],
							 &clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueReadBuffer: %s",
				   opencl_strerror(rc));
		goto error;
	}
	clgss->ev_index++;
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
		if (clgss->m_ktoast)
			clReleaseMemObject(clgss->m_ktoast);
		if (clgss->kern_prep)
			clReleaseKernel(clgss->kern_prep);
		if (clgss->kern_fixds)
			clReleaseKernel(clgss->kern_fixds);
		for (i=0; i < clgss->kern_sort_nums; i++)
			clReleaseKernel(clgss->kern_sort[i]);
		if (clgss->program && clgss->program != BAD_OPENCL_PROGRAM)
			clReleaseProgram(clgss->program);
		if (clgss->kern_sort)
			free(clgss->kern_sort);
		if (clgss->map_addr)
			munmap(clgss->map_addr, clgss->map_length);
	}
	gpusort->msg.errcode = rc;
	pgstrom_reply_message(&gpusort->msg);
}
