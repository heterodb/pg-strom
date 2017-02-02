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
#include "access/sysattr.h"
#include "catalog/pg_type.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/cost.h"
#include "optimizer/var.h"
#include "parser/parsetree.h"
#include "storage/ipc.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "pg_strom.h"
#include "opencl_gpusort.h"
#include <math.h>


typedef struct
{
	CustomPlan	cplan;
	const char *kern_source;	/* source of opencl kernel */
	int			extra_flags;	/* extra libraries to be included */
	List	   *used_params;	/* list of Const/Param being referenced */
	Bitmapset  *sortkey_resnums;/* resource numbers of sortkey */
	Size		sortkey_width;	/* width of sortkey */

	/* delivered from original Sort */
	int			numCols;		/* number of sort-key columns */
	AttrNumber *sortColIdx;		/* their indexes in the target list */
	Oid		   *sortOperators;	/* OIDs of operators to sort them by */
	Oid		   *collations;		/* OIDs of collations */
	bool	   *nullsFirst;		/* NULLS FIRST/LAST directions */
} GpuSortPlan;

typedef struct
{
	CustomPlanState	cps;
	TupleTableSlot *scan_slot;

	pgstrom_queue  *mqueue;
	Datum			dprog_key;
	kern_parambuf  *kparambuf;
	List		   *sortkey_resnums;/* list of resno of sortkeys */
	List		   *sortkey_toast;	/* list of resno of variable sortkeys */
	Size			sortkey_width;
	Size			gpusort_chunksz;
	cl_uint			nrows_per_chunk;
	bool			bulk_scan;
	bool			scan_done;
	bool			sort_done;
	/* running status */
	cl_int			num_running;	/* number of async running requests */
	dlist_head		sorted_chunks;

	/* row-/column- stores being read */
	cl_uint			rcs_nums;	/* number of row-/column-stores in use */
	cl_uint			rcs_slotsz;	/* size of row-/column-store slot */
	StromObject	  **rcs_slot;	/* slot of row-/column-stores */

	/* fallback by CPU sorting */
	SortSupport		cpusort_keys;

	pgstrom_perfmon	pfm;		/* sum of performance counter */
} GpuSortState;

/* static variables */
static int						pgstrom_gpusort_chunksize;
static CustomPlanMethods		gpusort_plan_methods;

/* static declarations */
static void clserv_process_gpusort(pgstrom_message *msg);

/*
 * get_gpusort_chunksize
 *
 * A chunk (including kern_parambuf, kern_column_store and kern_toastbuf)
 * has to be smaller than the least "max memory allocation size" of the
 * installed device. It gives a preferred size of gpusort chunk according
 * to the device properties and GUC configuration.
 */
static void
compute_gpusort_chunksize(Size width, double nrows,
						  List *sortkey_resnums,
						  bool has_toastbuf,
						  kern_parambuf *kparams,
						  Size *gpusort_chunksz,
						  Size *nrows_per_chunk)
{
	Size		device_restriction;
	Size		chunk_sz;
	Size		base_len;
	int			nkeys = list_length(sortkey_resnums) + 2;

	/*
	 * zone length is initialized to device's max memory allocation size
	 * on the starting up time, so a quarter of them is a safe expectation.
	 */
	device_restriction = (pgstrom_shmem_zone_length() / 2);
	device_restriction &= ~(SHMEM_BLOCKSZ - 1);

	/*
	 * length of metadata regardless of nrows
	 */
	base_len = (offsetof(pgstrom_gpusort_chunk, kern) +
				STROMALIGN(kparams->length) +
				STROMALIGN(offsetof(kern_column_store, colmeta[nkeys])) +
				STROMALIGN(sizeof(cl_int)) +
				STROMALIGN(offsetof(kern_toastbuf, coldir[0])));
	/*
	 * Try to estimate the chunk size being required in this context.
	 */
	width += 2 * sizeof(cl_uint) + sizeof(cl_ulong);

	/* if too large, nrows per chunk has to be reduced */
	while ((chunk_sz = (Size)
			(base_len + width * (1.08 * nrows))) >= device_restriction)
		nrows /= 2;
	chunk_sz = (1UL << get_next_log2(chunk_sz)) - SHMEM_ALLOC_COST;

	/*
	 * on the other hands, we can expand nrows as possible as we can,
	 * if we have no toast buffer.
	 */
	if (!has_toastbuf)
		nrows = (chunk_sz - base_len) / width;

	/* Put results */
	*gpusort_chunksz = chunk_sz;
	*nrows_per_chunk = TYPEALIGN(PGSTROM_WORKGROUP_UNITSZ, (Size)nrows);
}

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
 * expected_key_width
 *
 * It computes an expected (average) key width; to estimate number of rows
 * we can put of a gpusort chunk.
 */
static int
expected_key_width(TargetEntry *tle, Plan *subplan, List *rtable)
{
	Oid		type_oid = exprType((Node *)tle->expr);
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
								IsA(subplan, TidScan) ||
								IsA(subplan, ForeignScan)))
	{
		Var	   *var = (Var *)tle->expr;
		Index	scanrelid = ((Scan *)subplan)->scanrelid;
		RangeTblEntry *rte = rt_fetch(scanrelid, rtable);

		Assert(var->varno == OUTER_VAR);
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
gpusort_codegen_comparison(Sort *sort, codegen_context *context)
{
	StringInfoData	str;
	StringInfoData	decl;
	StringInfoData	body;
	int				i, j;

	initStringInfo(&str);
	initStringInfo(&decl);
	initStringInfo(&body);

	memset(context, 0, sizeof(codegen_context));

	/*
	 * System constants for GpuSort
	 * KPARAM_0 is an array of bool to show referenced (thus moved to
	 * sorting chunk) columns, to translate data from row-store.
	 * KPARAM_1 is a template of kcs_head for columnar bulk-loading,
	 * even though its cs_ofs has to be adjusted for each c-store.
	 */
	context->used_params = list_make2(makeConst(BYTEAOID,
												-1,
												InvalidOid,
												-1,
												PointerGetDatum(NULL),
												true,
												false),
									  makeConst(BYTEAOID,
												-1,
												InvalidOid,
												-1,
												PointerGetDatum(NULL),
												true,
												false));
	context->type_defs = list_make1(pgstrom_devtype_lookup(BYTEAOID));

	/* generate a comparison function */
	for (i=0; i < sort->numCols; i++)
	{
		TargetEntry *tle;
		AttrNumber	resno = sort->sortColIdx[i];
		bool		nullfirst = sort->nullsFirst[i];
		Oid			sort_op = sort->sortOperators[i];
		Oid			sort_func;
		Oid			sort_type;
		bool		is_reverse;
		AttrNumber	var_index;
		Var		   *tvar;
		devtype_info *dtype;
		devfunc_info *dfunc;

		tle = get_tle_by_resno(sort->plan.targetlist, resno);
		Assert(IsA(tle->expr, Var));
		tvar = (Var *)tle->expr;
		Assert(tvar->varno == OUTER_VAR);

		/* type for comparison */
		sort_type = exprType((Node *)tle->expr);
		dtype = pgstrom_devtype_lookup_and_track(sort_type, context);

		/* function for comparison */
		sort_func = get_compare_function(sort_op, &is_reverse);
		dfunc = pgstrom_devfunc_lookup_and_track(sort_func, context);

		appendStringInfo(&decl,
						 "  pg_%s_t keyval_x%u;\n"
						 "  pg_%s_t keyval_y%u;\n",
						 dtype->type_name, i+1,
						 dtype->type_name, i+1);

		/*
		 * find out variable-index on kernel column store
		 * (usually, this poor logic works sufficiently)
		 */
		var_index = 0;
		for (j=0; j < sort->numCols; j++)
		{
			if (sort->sortColIdx[j] < resno)
				var_index++;
		}

		/* reference to kcs-x */
		appendStringInfo(&body, "  keyval_x%u = ", i+1);
		if (dtype->type_flags & DEVTYPE_IS_VARLENA)
			appendStringInfo(
				&body,
				"pg_%s_vref(kcs_x,ktoast_x,errcode,%u,x_index);\n",
				dtype->type_name, var_index);
		else
			appendStringInfo(
				&body,
				"pg_%s_vref(kcs_x,errcode,%u,x_index);\n",
				dtype->type_name, var_index);

		/* reference to kcs-x */
		appendStringInfo(&body, "  keyval_y%u = ", i+1);
		if (dtype->type_flags & DEVTYPE_IS_VARLENA)
			appendStringInfo(
				&body,
				"pg_%s_vref(kcs_y,ktoast_y,errcode,%u,y_index);\n",
				dtype->type_name, var_index);
		else
			appendStringInfo(
				&body,
				"pg_%s_vref(kcs_y,errcode,%u,y_index);\n",
				dtype->type_name, var_index);

		appendStringInfo(
			&body,
			"  if (!keyval_x%u.isnull && !keyval_y%u.isnull)\n"
			"  {\n"
			"    comp = pgfn_%s(errcode, keyval_x%u, keyval_y%u);\n"
			"    if (comp.value != 0)\n"
			"      return %s;\n"
			"  }\n"
			"  else if (keyval_x%u.isnull && !keyval_y%u.isnull)\n"
			"    return %d;\n"
			"  else if (!keyval_x%u.isnull && keyval_y%u.isnull)\n"
			"    return %d;\n",
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
		"gpusort_comp(__private int *errcode,\n"
		"             __global kern_column_store *kcs_x,\n"
		"             __global kern_toastbuf *ktoast_x,\n"
		"             __private cl_int x_index,\n"
		"             __global kern_column_store *kcs_y,\n"
		"             __global kern_toastbuf *ktoast_y,\n"
		"             __private cl_int y_index)\n"
		"{\n"
		"%s"	/* key variable declarations */
		"  pg_int4_t comp;\n"
		"\n"
		"%s"	/* comparison body */
		"  return 0;\n"
		"}\n",
		pgstrom_codegen_declarations(context),
		decl.data,
		body.data);

	pfree(decl.data);
	pfree(body.data);

	return str.data;
}

/*
 * pgstrom_create_gpusort_plan
 *
 * suggest an alternative sort using GPU devices
 */
CustomPlan *
pgstrom_create_gpusort_plan(Sort *sort, List *rtable)
{
	GpuSortPlan *gsort;
	List	   *tlist = sort->plan.targetlist;
	Plan	   *subplan = outerPlan(sort);
	Size		sortkey_width = 0;
	Bitmapset  *sortkey_resnums = NULL;
	Cost		startup_cost;
	Cost		total_cost;
	codegen_context context;
	int			i, n;

	Assert(sort->plan.qual == NIL);
	Assert(!sort->plan.righttree);

	n = sort->numCols;
	for (i=0; i < n; i++)
	{
		TargetEntry	*tle = get_tle_by_resno(tlist, sort->sortColIdx[i]);
		Var	   *key_var = (Var *)tle->expr;
		Oid		sort_op = sort->sortOperators[i];
		Oid		sort_func;
		bool	is_reverse;

		/*
		 * Target-entry of Sort plan should be a var-node that references
		 * a particular column of underlying relation, even if Sort-key
		 * contains formula. So, we can expect a simple var-node towards
		 * outer relation here.
		 */
		if (!IsA(key_var, Var) || key_var->varno != OUTER_VAR)
			return NULL;

		sort_func = get_compare_function(sort_op, &is_reverse);
		if (!OidIsValid(sort_func) ||
			!pgstrom_devfunc_lookup(sort_func))
			return NULL;

		/* also, estimate average key length */
		sortkey_width += expected_key_width(tle, subplan, rtable);
		pull_varattnos((Node *)tle->expr,
					   OUTER_VAR,
					   &sortkey_resnums);
	}

	/* next, cost estimation by GPU sort */
	cost_gpusort(&startup_cost, &total_cost,
				 subplan->total_cost,
				 subplan->plan_rows,
				 subplan->plan_width);
	if (total_cost >= sort->plan.total_cost)
		return NULL;

	/*
	 * OK, expected GpuSort cost is enough reasonable to run
	 * Let's suggest grafter to replace the original Sort node
	 */
	gsort = palloc0(sizeof(GpuSortPlan));
	memcpy(&gsort->cplan.plan, &sort->plan, sizeof(Plan));
	gsort->cplan.plan.type = T_CustomPlan;
	gsort->cplan.plan.startup_cost = startup_cost;
	gsort->cplan.plan.total_cost = total_cost;
	gsort->cplan.methods = &gpusort_plan_methods;

	gsort->kern_source = gpusort_codegen_comparison(sort, &context);
	gsort->extra_flags = context.extra_flags | DEVKERNEL_NEEDS_GPUSORT;
	gsort->used_params = context.used_params;
	gsort->sortkey_resnums = sortkey_resnums;
	gsort->sortkey_width = sortkey_width;

	gsort->numCols = sort->numCols;
	gsort->sortColIdx = pmemcpy(sort->sortColIdx, sizeof(AttrNumber) * n);
	gsort->sortOperators = pmemcpy(sort->sortOperators, sizeof(Oid) * n);
	gsort->collations = pmemcpy(sort->collations, sizeof(Oid) * n);
	gsort->nullsFirst = pmemcpy(sort->nullsFirst, sizeof(bool) * n);

	return &gsort->cplan;
}

static void
gpusort_set_plan_ref(PlannerInfo *root,
					 CustomPlan *custom_plan,
					 int rtoffset)
{
	/* logic copied from set_dummy_tlist_references */
	Plan	   *plan = &custom_plan->plan;
	List	   *output_targetlist;
	ListCell   *l;

	output_targetlist = NIL;
	foreach(l, plan->targetlist)
	{
		TargetEntry *tle = (TargetEntry *) lfirst(l);
		Var		   *oldvar = (Var *) tle->expr;
		Var		   *newvar;

		newvar = makeVar(OUTER_VAR,
						 tle->resno,
						 exprType((Node *) oldvar),
						 exprTypmod((Node *) oldvar),
						 exprCollation((Node *) oldvar),
						 0);
		if (IsA(oldvar, Var))
		{
			newvar->varnoold = oldvar->varno + rtoffset;
			newvar->varoattno = oldvar->varattno;
		}
		else
		{
			newvar->varnoold = 0;		/* wasn't ever a plain Var */
			newvar->varoattno = 0;
		}

		tle = flatCopyTargetEntry(tle);
		tle->expr = (Expr *) newvar;
		output_targetlist = lappend(output_targetlist, tle);
	}
	plan->targetlist = output_targetlist;

	/* We don't touch plan->qual here */
}

static void
gpusort_finalize_plan(PlannerInfo *root,
					  CustomPlan *custom_plan,
					  Bitmapset **paramids,
					  Bitmapset **valid_params,
					  Bitmapset **scan_params)
{
	/* nothing to do */
}

/*
 * gpusort_construct_kcshead
 *
 * It sets up header portion of kern_column_store that can contain two
 * system columns on gpusort (growid and rindex).
 */
static bytea *
gpusort_construct_kcshead(TupleDesc tupdesc, cl_char *attrefs)
{
	kern_column_store *kcs_head;
	bytea	   *result;
	AttrNumber	i_col;

	/* make a kcs_head with 2 additional columns */
	result = kparam_make_kcs_head(tupdesc, attrefs, 2, 100);
	kcs_head = (kern_column_store *) VARDATA(result);

	/*
	 * The second last column is reserved by GpuSort - fixed-length
	 * long integer as identifier of unsorted tuples, not null.
	 */
	i_col = kcs_head->ncols - 2;
	kcs_head->colmeta[i_col].attnotnull = true;
	kcs_head->colmeta[i_col].attalign = sizeof(cl_long);
	kcs_head->colmeta[i_col].attlen = sizeof(cl_long);
	kcs_head->colmeta[i_col].cs_ofs = kcs_head->length;
	kcs_head->length += STROMALIGN(sizeof(cl_long) * kcs_head->nrooms);
	i_col++;

	/*
	 * Last column is reserved by GpuSort - fixed-length integer as
	 * index of sorted tuples, not null.
	 * Note that this field has to be aligned to 2^N length, to
	 * simplify kernel implementation.
	 */
	i_col = kcs_head->ncols - 1;
	kcs_head->colmeta[i_col].attnotnull = true;
	kcs_head->colmeta[i_col].attalign = sizeof(cl_int);
	kcs_head->colmeta[i_col].attlen = sizeof(cl_int);
	kcs_head->colmeta[i_col].cs_ofs = kcs_head->length;
	kcs_head->length += STROMALIGN(sizeof(cl_int) *
								   (1UL << get_next_log2(kcs_head->nrooms)));
	return result;
}

static void
pgstrom_release_gpusort_chunk(pgstrom_gpusort_chunk *gs_chunk)
{
	int		i;

	/* Unference the row- and column- stores being associated */
	for (i=0; i < gs_chunk->rcs_nums; i++)
		pgstrom_put_rcstore(gs_chunk->rcs_slot[i].rcstore);
	/*
	 * OK, let's free it. Note that row- and column-store are
	 * inidividually tracked by resource tracker, we don't need
	 * to care about them here.
	 */
	pgstrom_shmem_free(gs_chunk);
}

static pgstrom_gpusort_chunk *
pgstrom_create_gpusort_chunk(GpuSortState *gsortstate)
{
	pgstrom_gpusort_chunk *gs_chunk;
	Size			allocsz_chunk;
	kern_parambuf  *kparams;
	kern_column_store *kcs;
	kern_column_store *kcs_head;
	cl_int		   *kstatus;
	kern_toastbuf  *ktoast;

	/* allocation of a shared memory block */
	gs_chunk = pgstrom_shmem_alloc_alap(gsortstate->gpusort_chunksz,
										&allocsz_chunk);
	if (!gs_chunk)
		elog(ERROR, "out of shared memory");
	memset(gs_chunk, 0, sizeof(pgstrom_gpusort_chunk));
	gs_chunk->rcs_nums = 0;
	gs_chunk->rcs_head = -1;	/* to be set later */

	/* next, initialization of kern_gpusort */
	kparams = KERN_GPUSORT_PARAMBUF(&gs_chunk->kern);
	memcpy(kparams, gsortstate->kparambuf, gsortstate->kparambuf->length);
	Assert(kparams->length == STROMALIGN(kparams->length));
	kparam_refresh_kcs_head(kparams, 0, gsortstate->nrows_per_chunk);

	kcs = KERN_GPUSORT_CHUNK(&gs_chunk->kern);
	kcs_head = KPARAM_GET_KCS_HEAD(kparams);
	memcpy(kcs, kcs_head, offsetof(kern_column_store,
								   colmeta[kcs_head->ncols]));

	/* next, initialization of kernel execution status field */
	kstatus = KERN_GPUSORT_STATUS(&gs_chunk->kern);
	*kstatus = StromError_Success;

	/* last, initialization of toast buffer in flat-mode */
	ktoast = KERN_GPUSORT_TOASTBUF(&gs_chunk->kern);
	if (gsortstate->sortkey_toast == NIL)
		ktoast->length = offsetof(kern_toastbuf, coldir[0]);
	else
		ktoast->length =
			STROMALIGN_DOWN(allocsz_chunk - ((uintptr_t)ktoast -
											 (uintptr_t)gs_chunk));
	ktoast->usage = offsetof(kern_toastbuf, coldir[0]);

	/* OK, initialized */
	return gs_chunk;
}

/*
 * pgstrom_setup_cpusort
 *
 * We may have to run sorting on CPU, as fallback, if OpenCL is unavailable
 * to run GpuSort on device side.
 */
static void
pgstrom_setup_cpusort(GpuSortState *gsortstate)
{
	GpuSortPlan *gsort = (GpuSortPlan *)gsortstate->cps.ps.plan;
	int		i = 0;
	int		nkeys = list_length(gsortstate->sortkey_resnums);

	gsortstate->cpusort_keys = palloc0(nkeys * sizeof(SortSupportData));
	for (i=0; i < nkeys; i++)
	{
		SortSupport sortKey = &gsortstate->cpusort_keys[i];
		AttrNumber	resno = gsort->sortColIdx[i];

		Assert(OidIsValid(gsort->sortOperators[i]));

		sortKey->ssup_cxt = CurrentMemoryContext;	/* may needs own special context*/
		sortKey->ssup_collation = gsort->collations[i];
		sortKey->ssup_nulls_first = gsort->nullsFirst[i];
		sortKey->ssup_attno = resno;

		PrepareSortSupportFromOrderingOp(gsort->sortOperators[i], sortKey);
		i++;
	}
}

/*
 * pgstrom_compare_cpusort
 *
 * It runs CPU based comparison between two values on kern_column_store.
 */
static int
pgstrom_compare_cpusort(GpuSortState *gsortstate,
						kern_column_store *x_kcs,
						kern_toastbuf *x_toast,
						int xindex,
						kern_column_store *y_kcs,
						kern_toastbuf *y_toast,
						int yindex)
{
	TupleDesc	tupdesc = gsortstate->scan_slot->tts_tupleDescriptor;
	ListCell   *cell;
	int			i = 0;

	foreach (cell, gsortstate->sortkey_resnums)
	{
		SortSupport	sort_key = &gsortstate->cpusort_keys[i];
		AttrNumber	resno = lfirst_int(cell);
		Form_pg_attribute attr = tupdesc->attrs[resno - 1];
		const void *x_addr;
		const void *y_addr;
		Datum		x_value;
		Datum		y_value;
		bool		x_isnull;
		bool		y_isnull;
		cl_uint		offset;
		int			comp;

		/* fetch a X-value to be compared */
		x_addr = kern_get_datum(x_kcs, i, xindex);
		if (!x_addr)
		{
			x_isnull = true;
			x_value = 0;
		}
		else
		{
			x_isnull = false;
			if (attr->attlen > 0)
				x_value = fetch_att(x_addr, attr->attbyval, attr->attlen);
			else
			{
				offset = *((cl_uint *)x_addr);
				x_value = PointerGetDatum((char *)x_toast + offset);
			}
		}

		/* fetch a Y-value to be compared */
		y_addr = kern_get_datum(y_kcs, i, yindex);
		if (!y_addr)
		{
			y_isnull = true;
			y_value = 0;
		}
		else
		{
			y_isnull = false;
			if (attr->attlen > 0)
				y_value = fetch_att(y_addr, attr->attbyval, attr->attlen);
			else
			{
				offset = *((cl_uint *)y_addr);
				y_value = PointerGetDatum((char *)y_toast + offset);
			}
		}

		comp = ApplySortComparator(x_value, x_isnull,
								   y_value, y_isnull,
								   sort_key);
		if (comp != 0)
			return comp;
	}
	return 0;
}



/*
 * pgstrom_cpu_quicksort
 *
 * It runs CPU based quick-sorting on a particular chunk.
 * After that, its result index shall be sorted according to key-comparison
 * results.
 */
static void
recursive_cpu_quicksort(GpuSortState *gsortstate,
						kern_column_store *kcs,
						kern_toastbuf *ktoast,
						cl_int *rindex,
						int left,
						int right)
{
	if (left < right)
	{
		int		i = left;
		int		j = right;
		int		temp;
		int		pivot = (i + j) / 2;

		while (true)
		{
			while (pgstrom_compare_cpusort(gsortstate,
										   kcs, ktoast, rindex[i],
										   kcs, ktoast, rindex[pivot]) < 0)
				i++;
			while (pgstrom_compare_cpusort(gsortstate,
										   kcs, ktoast, rindex[j],
										   kcs, ktoast, rindex[pivot]) > 0)
				j--;
			if (i >= j)
				break;
			/* swap it */
			temp = rindex[i];
			rindex[i] = rindex[j];
			rindex[j] = temp;
			i++;
			j--;
		}
		recursive_cpu_quicksort(gsortstate, kcs, ktoast,
								rindex, left, i-1);
		recursive_cpu_quicksort(gsortstate, kcs, ktoast,
								rindex, j+1, right);
	}
}

static void
pgstrom_cpu_quicksort(GpuSortState *gsortstate, pgstrom_gpusort *gpusort)
{
	pgstrom_gpusort_chunk *gs_chunk;
	kern_column_store  *kcs;
	kern_toastbuf	   *ktoast;
	cl_int			   *rindex;

	gs_chunk = dlist_container(pgstrom_gpusort_chunk, chain,
							   dlist_head_node(&gpusort->gs_chunks));
	kcs = KERN_GPUSORT_CHUNK(&gs_chunk->kern);
	ktoast = KERN_GPUSORT_TOASTBUF(&gs_chunk->kern);
	rindex = KERN_GPUSORT_RESULT_INDEX(kcs);

	recursive_cpu_quicksort(gsortstate, kcs, ktoast, rindex,
							0, kcs->nrows-1);
	gpusort->is_sorted = true;
}

/*
 * pgstrom_release_gpusort
 *
 * Destructor of pgstrom_gpusort_multi object, to be called when refcnt
 * as message object reached to zero.
 *
 * NOTE: this routine may be called in the OpenCL server's context
 */
static void
pgstrom_release_gpusort(pgstrom_message *msg)
{
	pgstrom_gpusort		   *gpusort = (pgstrom_gpusort *) msg;
	pgstrom_gpusort_chunk  *gs_chunk;
	dlist_iter	iter;

	/* unlink message queue and device program (should always exists) */
	pgstrom_put_queue(msg->respq);
	pgstrom_put_devprog_key(gpusort->dprog_key);

	/*
	 * pgstrom_gpusort usually has multiple pgstrom_gpusort_chunk
	 * objects in its input and output list, as literal.
	 * So, let's unlink them.
	 */
	dlist_foreach(iter, &gpusort->gs_chunks)
	{
		gs_chunk = dlist_container(pgstrom_gpusort_chunk, chain, iter.cur);
		pgstrom_release_gpusort_chunk(gs_chunk);
	}
	pgstrom_shmem_free(gpusort);
}

/*
 * pgstrom_create_gpusort
 *
 * constructor of pgstrom_gpusort object.
 */
static pgstrom_gpusort *
pgstrom_create_gpusort(GpuSortState *gsortstate)
{
	pgstrom_gpusort	   *gpusort;

	gpusort = pgstrom_shmem_alloc(sizeof(pgstrom_gpusort));
	if (!gpusort)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("out of shared memory")));
	memset(&gpusort->chain, 0, sizeof(dlist_node));

	/* initialize the common message field */
	memset(gpusort, 0, sizeof(pgstrom_gpusort));
	gpusort->msg.sobj.stag = StromTag_GpuSort;
	SpinLockInit(&gpusort->msg.lock);
	gpusort->msg.refcnt = 1;
	gpusort->msg.respq = pgstrom_get_queue(gsortstate->mqueue);
	gpusort->msg.cb_process = clserv_process_gpusort;
	gpusort->msg.cb_release = pgstrom_release_gpusort;
	gpusort->msg.pfm.enabled = gsortstate->pfm.enabled;
	/* other fields also */
	gpusort->dprog_key = pgstrom_retain_devprog_key(gsortstate->dprog_key);
	gpusort->has_rindex = gsortstate->bulk_scan;
	gpusort->is_sorted = false;
	dlist_init(&gpusort->gs_chunks);

	return gpusort;
}

/*
 * gpusort_preload_subplan
 *
 * It loads row-store (or column-store if available) from the underlying
 * relation scan, and enqueue a gpusort-chunk with them into OpenCL
 * server process. If a chunk can be successfully enqueued, it returns
 * reference of this chunk. Elsewhere, NULL shall be returns; that implies
 * all the records in the underlying relation were already read.
 */
static pgstrom_gpusort *
gpusort_preload_subplan(GpuSortState *gsortstate, HeapTuple *overflow)
{
	pgstrom_gpusort	   *gpusort;
	pgstrom_gpusort_chunk *gs_chunk;
	PlanState		   *subnode = outerPlanState(gsortstate);
	TupleDesc			tupdesc = ExecGetResultType(subnode);
	EState			   *estate = gsortstate->cps.ps.state;
	ScanDirection		dir_saved = estate->es_direction;
	tcache_row_store   *trs;
	kern_column_store  *kcs;
	kern_toastbuf	   *ktoast;
	TupleTableSlot	   *slot;
	HeapTuple			tuple;
	Size				toast_usage;
	int					nrows;
	struct timeval		tv1, tv2;

	/* do we have any more tuple to read? */
	if (gsortstate->scan_done)
		return NULL;

	/* create a gpusort message object */
	gpusort = pgstrom_create_gpusort(gsortstate);
	pgstrom_track_object(&gpusort->msg.sobj, 0);

	/* create a gpusort chunk */
	gs_chunk = pgstrom_create_gpusort_chunk(gsortstate);
	gs_chunk->rcs_head = gsortstate->rcs_nums;
	kcs = KERN_GPUSORT_CHUNK(&gs_chunk->kern);
	ktoast = KERN_GPUSORT_TOASTBUF(&gs_chunk->kern);
	toast_usage = ktoast->usage;

	dlist_push_head(&gpusort->gs_chunks, &gs_chunk->chain);

	/* subplan should take forward scan */
	estate->es_direction = ForwardScanDirection;

	if (gpusort->msg.pfm.enabled)
		gettimeofday(&tv1, NULL);

	/* copy tuples to tcache_row_store */
	trs = NULL;
	for (nrows=0; nrows < kcs->nrooms; nrows++)
	{
		if (HeapTupleIsValid(*overflow))
		{
			tuple = *overflow;
			*overflow = NULL;
		}
		else
		{
			slot = ExecProcNode(subnode);
			if (TupIsNull(slot))
			{
				gsortstate->scan_done = true;
				break;
			}
			tuple = ExecFetchSlotTuple(slot);
		}

		/*
		 * check whether the ktoast capacity to store variable length
		 * values. If now, we once break to scan and this tuple shall
		 * be moved to the next gpusort chunk.
		 */
		if (gsortstate->sortkey_toast != NIL)
		{
			Size		toastsz = 0;
			ListCell   *lc;
			bool		isnull;
			Datum		value;

			foreach (lc, gsortstate->sortkey_toast)
			{
				value = slot_getattr(slot, lfirst_int(lc), &isnull);
				if (!isnull)
					toastsz += MAXALIGN(VARSIZE_ANY(value));
			}

			/*
			 * If we can expect variable length field overflows in
			 * the opencl kernel, we don't push tuples onto this chunk
			 * any more. Then, this tuple shall be put on the next
			 * chunk instead.
			 */
			if (toast_usage + toastsz > ktoast->length)
			{
				*overflow = tuple;
				Assert(nrows > 0);
				break;
			}
			toast_usage += toastsz;
		}

		/* OK, let's put it on the row-store */
		if (!trs || !tcache_row_store_insert_tuple(trs, tuple))
		{
			/* break it, if this gs_chunk cannot host no more rcstores */
			if (gs_chunk->rcs_nums == lengthof(gs_chunk->rcs_slot))
			{
				*overflow = tuple;
				break;
			}

			/*
			 * allocation of a new row-store, but not tracked because trs
			 * is always kept by a particular gpusort-chunk.
			 */
			trs = pgstrom_create_row_store(tupdesc);

			/* put it on the r/c-store array on the GpuSortState */
			if (gsortstate->rcs_nums == gsortstate->rcs_slotsz)
			{
				gsortstate->rcs_slotsz += gsortstate->rcs_slotsz;
				gsortstate->rcs_slot = repalloc(gsortstate->rcs_slot,
												sizeof(StromObject *) *
											    gsortstate->rcs_slotsz);
			}
			gsortstate->rcs_slot[gsortstate->rcs_nums++] = &trs->sobj;

			/* also, put it on the r/c-store array on the gpusort_chunk */
			gs_chunk->rcs_slot[gs_chunk->rcs_nums].rindex = NULL;
			gs_chunk->rcs_slot[gs_chunk->rcs_nums].nitems = -1;
			gs_chunk->rcs_slot[gs_chunk->rcs_nums].rcstore = &trs->sobj;
			gs_chunk->rcs_nums++;

			/* insertion towards new empty row-store should not be failed */
			if (!tcache_row_store_insert_tuple(trs, tuple))
				elog(ERROR, "failed to put a tuple on a new row-store");
		}
	}
	estate->es_direction = dir_saved;

	if (gpusort->msg.pfm.enabled)
	{
		gettimeofday(&tv2, NULL);
		gpusort->msg.pfm.time_to_load = timeval_diff(&tv1, &tv2);
	}

	/* no tuples were read in actually, so nothing to do */
	if (nrows == 0)
	{
		pgstrom_untrack_object(&gpusort->msg.sobj);
		pgstrom_put_message(&gpusort->msg);
		return NULL;
	}
	return gpusort;
}

/*
 * gpusort_preload_subplan_bulk
 *
 * Like gpusort_preload_subplan, it also load tuples from subplan, but it
 * uses bulk-scan mode.
 */
static pgstrom_gpusort *
gpusort_preload_subplan_bulk(GpuSortState *gsortstate,
							 pgstrom_bulk_slot **overflow)
{
	pgstrom_gpusort	   *gpusort;
	pgstrom_gpusort_chunk *gs_chunk;
	PlanState		   *subnode = outerPlanState(gsortstate);
	TupleDesc			tupdesc = ExecGetResultType(subnode);
	kern_column_store  *kcs;
	kern_toastbuf	   *ktoast;
	cl_uint				kcs_usage;
	Size				toast_usage;
	cl_int			   *rindex;
	ListCell		   *lc;
	struct timeval		tv1, tv2;

	/* do we have any more row/column store to read? */
	if (gsortstate->scan_done)
		return NULL;

	/* create a gpusort message object */
	gpusort = pgstrom_create_gpusort(gsortstate);
	pgstrom_track_object(&gpusort->msg.sobj, 0);

	/* create a gpusort chunk */
	gs_chunk = pgstrom_create_gpusort_chunk(gsortstate);
	gs_chunk->rcs_head = gsortstate->rcs_nums;
	kcs = KERN_GPUSORT_CHUNK(&gs_chunk->kern);
	kcs_usage = kcs->nrows;
	ktoast = KERN_GPUSORT_TOASTBUF(&gs_chunk->kern);
	toast_usage = ktoast->usage;
	rindex = KERN_GPUSORT_RESULT_INDEX(kcs);

    dlist_push_head(&gpusort->gs_chunks, &gs_chunk->chain);

	if (gpusort->msg.pfm.enabled)
		gettimeofday(&tv1, NULL);

	while (true)
	{
		pgstrom_bulk_slot *bulk;
		cl_uint		nitems;
		cl_uint		toastsz;

		/* load the next bulk */
		if (*overflow)
		{
			bulk = *overflow;
			*overflow = NULL;
		}
		else
		{
			bulk = (pgstrom_bulk_slot *)MultiExecProcNode(subnode);
			if (!bulk)
			{
				gsortstate->scan_done = true;
				break;
			}
		}
		nitems = bulk->nitems;
		toastsz = 0;

		/* pgstrom_bulk_slot performs as if T_String */
		Assert(IsA(bulk, String) && ((Value *)bulk)->val.str == NULL);

		if (StromTagIs(bulk->rc_store, TCacheRowStore))
		{
			tcache_row_store *trs = (tcache_row_store *) bulk->rc_store;

			Assert(nitems <= trs->kern.nrows);
			/* no capacity to put nitems more */
			if (kcs_usage + nitems >= kcs->nrooms)
			{
				*overflow = bulk;
				break;
			}

			/*
			 * If variable length sort-keys are in use, we need to ensure
			 * toast buffer has enough space to store.
			 */
			if (gsortstate->sortkey_toast != NIL)
			{
				rs_tuple   *rs_tup;
				Datum		value;
				bool		isnull;
				int			i, j;

				for (i=0; i < nitems; i++)
				{
					j = bulk->rindex[i];
					Assert(j < trs->kern.nrows);

					rs_tup = kern_rowstore_get_tuple(&trs->kern, j);
					if (!rs_tup)
						elog(ERROR, "bug? null record was fetched");

					foreach (lc, gsortstate->sortkey_toast)
					{
						value = heap_getattr(&rs_tup->htup,
											 lfirst_int(lc),
											 tupdesc,
											 &isnull);
						if (!isnull)
							toastsz += MAXALIGN(VARSIZE_ANY(value));
					}
				}
				if (toast_usage + toastsz >= ktoast->length)
				{
					*overflow = bulk;
					break;
				}
			}
			elog(INFO, "bulkload row (nitems=%u of %u, length=%u)", nitems, trs->kern.nrows, trs->kern.length);
		}
		else if (StromTagIs(bulk->rc_store, TCacheColumnStore))
		{
			tcache_column_store *tcs = (tcache_column_store *) bulk->rc_store;
			cl_uint		nitems = bulk->nitems;
			Size		toastsz = 0;

			/* no capacity to put nitems any more? */
			if (kcs_usage + nitems >= kcs->nrooms)
			{
				*overflow = bulk;
				break;
            }

			if (gsortstate->sortkey_toast != NIL)
			{
				cl_uint		cs_ofs;
				char	   *vl_ptr;
				int         i, j, k;

				for (i=0; i < nitems; i++)
				{
					j = bulk->rindex[i];
					Assert(j < tcs->nrows);

					foreach (lc, gsortstate->sortkey_toast)
					{
						k = lfirst_int(lc) - 1;

						if (att_isnull(j, tcs->cdata[k].isnull))
							continue;
						cs_ofs = ((cl_uint *)tcs->cdata[k].values)[j];
						Assert(cs_ofs > 0);

						vl_ptr = ((char *)tcs->cdata[k].toast + cs_ofs);
						toastsz += MAXALIGN(VARSIZE_ANY(vl_ptr));
					}
				}
                if (toast_usage + toastsz >= ktoast->length)
                {
                    *overflow = bulk;
                    break;
                }
            }
		}
		else
			elog(ERROR, "bug? neither row nor column store");

		/*
		 * NOTE: Fetched bulk-store may contain rows already filtered.
		 * It is ideal if we would remove them prior to DMA send, but
		 * it usually takes expensive memory operations.
		 * So, we have rindex that shows which rows are still available,
		 * to indicate the rows to be processed.
		 */
		memcpy(rindex + kcs_usage, bulk->rindex, sizeof(cl_int) * nitems);

		/* OK, let's put this row/column store on this chunk */
		if (gsortstate->rcs_nums == gsortstate->rcs_slotsz)
		{
			gsortstate->rcs_slotsz += gsortstate->rcs_slotsz;
			gsortstate->rcs_slot = repalloc(gsortstate->rcs_slot,
											sizeof(StromObject *) *
											gsortstate->rcs_slotsz);
		}
		gsortstate->rcs_slot[gsortstate->rcs_nums++]
			= pgstrom_get_rcstore(bulk->rc_store);

		/* also, put it on the r/c-store array on the gpusort */
		gs_chunk->rcs_slot[gs_chunk->rcs_nums].rindex = rindex + kcs_usage;
		gs_chunk->rcs_slot[gs_chunk->rcs_nums].nitems = nitems;
		gs_chunk->rcs_slot[gs_chunk->rcs_nums].rcstore
			= pgstrom_get_rcstore(bulk->rc_store);
		gs_chunk->rcs_nums++;

		kcs_usage += nitems;
		toast_usage += toastsz;

		/* OK, this bulk-store itself is no longer referenced */
		pgstrom_release_bulk_slot(bulk);
	}
	if (gpusort->msg.pfm.enabled)
	{
		gettimeofday(&tv2, NULL);
		gpusort->msg.pfm.time_to_load = timeval_diff(&tv1, &tv2);
	}
	return gpusort;
}

/*
 * gpusort_process_response
 *
 * It processes the supplied gpusort message. If we have a pending gpusort,
 * it tries to merge two sequential chunks into one. Elsewhere, it should
 * be pending until next sequential chunks will come.
 */
static void
gpusort_process_response(GpuSortState *gsortstate, pgstrom_gpusort *gpusort)
{
	pgstrom_gpusort_chunk  *x_chunk;
	pgstrom_gpusort_chunk  *y_chunk;
	kern_column_store	   *kcs_x;
	kern_column_store	   *kcs_y;
	kern_toastbuf		   *ktoast_x;
	kern_toastbuf		   *ktoast_y;
	cl_int				   *rindex_x;
	cl_int				   *rindex_y;
	dlist_iter				iter;

	Assert(dlist_length(&gpusort->gs_chunks) == 1);
	if (gpusort->msg.errcode != StromError_Success)
	{
		if (gpusort->msg.errcode == CL_BUILD_PROGRAM_FAILURE)
		{
			const char *buildlog
				= pgstrom_get_devprog_errmsg(gpusort->dprog_key);
			const char *kern_source
				= ((GpuSortPlan *)gsortstate->cps.ps.plan)->kern_source;

			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("PG-Strom: OpenCL execution error (%s)\n%s",
							pgstrom_strerror(gpusort->msg.errcode),
							kern_source),
					 errdetail("%s", buildlog)));
		}
		else if (gpusort->msg.errcode == StromError_DataStoreReCheck)
		{
			pgstrom_cpu_quicksort(gsortstate, gpusort);
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("PG-Strom: OpenCL execution error (%s)",
							pgstrom_strerror(gpusort->msg.errcode))));
		}
	}

	/*
	 * sanity check - a processed chunk or chunks should be sorted, even
	 * if it took a fallback by CPU sorting. Elsewhere, we cannot continue
	 * sorting any more.
	 */
	if (!gpusort->is_sorted)
		elog(ERROR, "Bug? processed gpusort chunk(s) are not sorted");

	x_chunk = dlist_container(pgstrom_gpusort_chunk, chain,
							  dlist_pop_head_node(&gpusort->gs_chunks));
	kcs_x = KERN_GPUSORT_CHUNK(&x_chunk->kern);
	ktoast_x = KERN_GPUSORT_TOASTBUF(&x_chunk->kern);
	rindex_x = KERN_GPUSORT_RESULT_INDEX(kcs_x);

	/*
	 * gpusort structure is not used any more
	 *
	 * TODO: resource tracker cannot handle orphan gpusort_chunk,
	 * if error happen. We need to consolidate it with gpusort
	 */
	pgstrom_untrack_object(&gpusort->msg.sobj);
	pgstrom_put_message(&gpusort->msg);

	dlist_foreach(iter, &gsortstate->sorted_chunks)
	{
		y_chunk = dlist_container(pgstrom_gpusort_chunk, chain, iter.cur);
		kcs_y = KERN_GPUSORT_CHUNK(&y_chunk->kern);
		ktoast_y = KERN_GPUSORT_TOASTBUF(&y_chunk->kern);
		rindex_y = KERN_GPUSORT_RESULT_INDEX(kcs_y);

		if (pgstrom_compare_cpusort(gsortstate,
									kcs_x, ktoast_x, rindex_x[0],
									kcs_y, ktoast_y, rindex_y[0]) < 0)
		{
			dlist_insert_before(&y_chunk->chain, &x_chunk->chain);
			return;
		}
	}
	dlist_push_tail(&gsortstate->sorted_chunks, &x_chunk->chain);
}

/*
 * gpusort_support_multi_exec
 *
 * It gives a hint whether the supplied plan-state support bulk-exec mode,
 * or not. If it is GpuSort provided by PG-Strom, it does not allow bulk-
 * exec mode anyway.
 */
bool
gpusort_support_multi_exec(const CustomPlanState *cps)
{
	return false;
}

static CustomPlanState *
gpusort_begin(CustomPlan *node, EState *estate, int eflags)
{
	GpuSortPlan	   *gsortplan = (GpuSortPlan *) node;
	GpuSortState   *gsortstate;
	TupleDesc		tupdesc;
	List		   *sortkey_resnums = NIL;
	List		   *sortkey_toast = NIL;
	Bitmapset	   *tempset;
	AttrNumber		anum;
	Size			gpusort_chunksz;
	Size			nrows_per_chunk;
	Const		   *kparam_0;
	Const		   *kparam_1;
	bytea		   *pdatum;
	List		   *used_params;

	/*
	 * create state structure
	 */
	gsortstate = palloc0(sizeof(GpuSortState));
	gsortstate->cps.methods = &gpusort_plan_methods;
	gsortstate->cps.ps.type = T_CustomPlanState;
	gsortstate->cps.ps.plan = &node->plan;
	gsortstate->cps.ps.state = estate;

    /*
     * Miscellaneous initialization
     *
     * Sort nodes don't initialize their ExprContexts because they never call
     * ExecQual or ExecProject.
	 * So, ExecInitExpr is not needed to call on neither targetlist and qual.
     */

	/*
	 * tuple table initialization
	 *
	 * sort nodes only return scan tuples from their sorted relation.
	 */
	ExecInitResultTupleSlot(estate, &gsortstate->cps.ps);
	gsortstate->scan_slot = ExecAllocTableSlot(&estate->es_tupleTable);

	/*
	 * initialize child nodes
	 *
	 * We shield the child node from the need to support REWIND, BACKWARD, or
	 * MARK/RESTORE.
	 */
	eflags &= ~(EXEC_FLAG_REWIND | EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK);

	outerPlanState(gsortstate) = ExecInitNode(outerPlan(node), estate, eflags);

	/*
	 * initialize tuple type.  no need to initialize projection info because
	 * this node doesn't do projections.
	 */
	ExecAssignResultTypeFromTL(&gsortstate->cps.ps);

	tupdesc = ExecGetResultType(outerPlanState(&gsortstate->cps.ps));
	ExecSetSlotDescriptor(gsortstate->scan_slot, tupdesc);

	gsortstate->cps.ps.ps_ProjInfo = NULL;

	/*
	 * OK, above PostgreSQL's executor stuff was correctly initialized.
	 * Let's start to setting up GpuSort stuff.
	 */

	/* Setting up device kernel */
	gsortstate->dprog_key = pgstrom_get_devprog_key(gsortplan->kern_source,
													gsortplan->extra_flags);
	pgstrom_track_object((StromObject *)gsortstate->dprog_key, 0);

	/* Also, message queue */
	gsortstate->mqueue = pgstrom_create_queue();
	pgstrom_track_object(&gsortstate->mqueue->sobj, 0);

	/* is outerplan support bulk-scan mode? */
	gsortstate->bulk_scan =
		pgstrom_plan_can_multi_exec(outerPlanState(gsortstate));

	/* setup sortkey_resnums and sortkey_toast list */
	tempset = bms_copy(gsortplan->sortkey_resnums);
	while ((anum = bms_first_member(tempset)) >= 0)
	{
		anum += FirstLowInvalidHeapAttributeNumber;
		Assert(anum > 0 && anum <= tupdesc->natts);

		sortkey_resnums = lappend_int(sortkey_resnums, anum);
		if (tupdesc->attrs[anum - 1]->attlen < 0)
			sortkey_toast = lappend_int(sortkey_toast, anum);
	}

	/*
	 * construction of kparam_0 according to the actual column reference
	 */
	Assert(list_length(gsortplan->used_params) >= 2);
	used_params = copyObject(gsortplan->used_params);

	kparam_0 = (Const *)linitial(used_params);
    Assert(IsA(kparam_0, Const) &&
           kparam_0->consttype == BYTEAOID &&
           kparam_0->constisnull);
	pdatum = kparam_make_attrefs_by_resnums(tupdesc, sortkey_resnums);
    kparam_0->constvalue = PointerGetDatum(pdatum);
	kparam_0->constisnull = false;

	kparam_1 = (Const *)lsecond(used_params);
	Assert(IsA(kparam_1, Const) &&
           kparam_1->consttype == BYTEAOID &&
           kparam_1->constisnull);
	pdatum = gpusort_construct_kcshead(tupdesc, (cl_char *)VARDATA(pdatum));
	kparam_1->constvalue = PointerGetDatum(pdatum);
	kparam_1->constisnull = false;

	gsortstate->kparambuf = pgstrom_create_kern_parambuf(used_params, NULL);

	/*
	 * Estimation of appropriate chunk size.
	 */
	compute_gpusort_chunksize(gsortplan->sortkey_width,
							  gsortplan->cplan.plan.plan_rows,
							  sortkey_resnums,
							  (bool)(sortkey_toast != NIL),
							  gsortstate->kparambuf,
							  &gpusort_chunksz,
							  &nrows_per_chunk);

	/*
	 * misc initialization of GpuSortState
	 */
	gsortstate->sortkey_resnums = sortkey_resnums;
	gsortstate->sortkey_toast = sortkey_toast;
	gsortstate->sortkey_width = gsortplan->sortkey_width;
	gsortstate->gpusort_chunksz = gpusort_chunksz;
	gsortstate->nrows_per_chunk = nrows_per_chunk;

	/* running status */
	gsortstate->num_running = 0;
	dlist_init(&gsortstate->sorted_chunks);

	/* allocate a certain amount of row-/column-store slot */
	gsortstate->rcs_nums = 0;
	gsortstate->rcs_slotsz = 256;
	gsortstate->rcs_slot = palloc0(sizeof(StromObject *) *
								   gsortstate->rcs_slotsz);

	/* setup fallback sorting by CPU */
	pgstrom_setup_cpusort(gsortstate);

	/* Is perfmon needed? */
	gsortstate->pfm.enabled = pgstrom_perfmon_enabled;

	return &gsortstate->cps;
}

















static TupleTableSlot *
gpusort_exec(CustomPlanState *node)
{
	GpuSortState	   *gsortstate = (GpuSortState *) node;
	pgstrom_queue	   *mqueue = gsortstate->mqueue;
	pgstrom_message	   *msg;
	pgstrom_gpusort	   *gpusort;
	pgstrom_gpusort_chunk *gs_chunk;
	kern_column_store  *kcs;
	TupleTableSlot	   *slot;

	if (!gsortstate->sort_done)
	{
		HeapTuple		overflow_tup = NULL;
		pgstrom_bulk_slot *overflow_rcs = NULL;

		while (!gsortstate->scan_done)
		{
			if (!gsortstate->bulk_scan)
				gpusort = gpusort_preload_subplan(gsortstate,
												  &overflow_tup);
			else
				gpusort = gpusort_preload_subplan_bulk(gsortstate,
													   &overflow_rcs);
			if (!gpusort)
			{
				Assert(gsortstate->scan_done);
				Assert(!overflow_tup && !overflow_rcs);
				break;
			}
			Assert(dlist_length(&gpusort->gs_chunks) == 1);

			/*
			 * Special case handling if a chunk preloaded contains only
			 * one rows. It is obviously sorted, and should have no more
			 * tuples to read.
			 */
			gs_chunk = dlist_container(pgstrom_gpusort_chunk, chain,
									   dlist_head_node(&gpusort->gs_chunks));
			kcs = KERN_GPUSORT_CHUNK(&gs_chunk->kern);
			if (kcs->nrows == 1)
			{
				Assert(gsortstate->scan_done);
				Assert(!overflow_tup && !overflow_rcs);
				gpusort->is_sorted = true;
				gpusort_process_response(gsortstate, gpusort);
				break;
			}

			if (!pgstrom_enqueue_message(&gpusort->msg))
				elog(ERROR, "Bug? OpenCL server seems to dead");
			gsortstate->num_running++;

			/*
			 * previous gpusort chunk might be completed during
			 * preloading, so dequeue it and consolidate them
			 * into pgstrom_gpusort_multi for inter-chunk merge
			 * sorting.
			 */
			while ((msg = pgstrom_try_dequeue_message(mqueue)) != NULL)
			{
				gsortstate->num_running--;

				if (msg->pfm.enabled)
					pgstrom_perfmon_add(&gsortstate->pfm, &msg->pfm);

				Assert(StromTagIs(msg, GpuSort));
				gpusort = (pgstrom_gpusort *) msg;
				gpusort_process_response(gsortstate, gpusort);
			}
		}

		/*
		 * Once scan of underlying relation got done, we iterate to merge
		 * two series of chunks into one sequence until all the chunks
		 * get merged into one.
		 */
		while (gsortstate->num_running > 0)
		{
			msg = pgstrom_dequeue_message(mqueue);
			if (!msg)
				elog(ERROR, "Bug? response of OpenCL server too late");
			gsortstate->num_running--;

			if (msg->pfm.enabled)
				pgstrom_perfmon_add(&gsortstate->pfm, &msg->pfm);

			Assert(StromTagIs(msg, GpuSort));
			gpusort = (pgstrom_gpusort *) msg;
			gpusort_process_response(gsortstate, gpusort);
		}
		gsortstate->sort_done = true;
	}
	/* OK, sorting done, fetch tuples according to the result */
	slot = gsortstate->cps.ps.ps_ResultTupleSlot;
	ExecClearTuple(slot);
	if (!dlist_is_empty(&gsortstate->sorted_chunks))
	{
		pgstrom_gpusort_chunk  *x_chunk;
		pgstrom_gpusort_chunk  *y_chunk;
		kern_column_store	   *kcs_x;
		kern_column_store	   *kcs_y;
		kern_toastbuf		   *ktoast_x;
		kern_toastbuf		   *ktoast_y;
		cl_int				   *rindex_x;
		cl_int				   *rindex_y;
		dlist_node			   *dnode;
		cl_ulong			   *growid;
		cl_uint					rcs_idx;
		cl_uint					row_idx;
		cl_int					i, j;
		StromObject			   *sobject;
		dlist_iter				iter;

		dnode = dlist_pop_head_node(&gsortstate->sorted_chunks);
		x_chunk = dlist_container(pgstrom_gpusort_chunk, chain, dnode);
		kcs_x = KERN_GPUSORT_CHUNK(&x_chunk->kern);
		ktoast_x = KERN_GPUSORT_TOASTBUF(&x_chunk->kern);
		rindex_x = KERN_GPUSORT_RESULT_INDEX(kcs_x);
		growid = KERN_GPUSORT_GLOBAL_ROWID(kcs_x);
		Assert(x_chunk->scan_pos < kcs_x->nrows);

		i = x_chunk->scan_pos++;
		Assert(rindex_x[i] >= 0 && rindex_x[i] < kcs_x->nrows);
		j = rindex_x[i];

		rcs_idx = ((growid[j] >> 32) & 0xffffffff);
		row_idx = (growid[j] & 0xffffffff);
		Assert(rcs_idx < gsortstate->rcs_nums);

		sobject = gsortstate->rcs_slot[rcs_idx];
		if (StromTagIs(sobject, TCacheRowStore))
		{
			tcache_row_store *trs = (tcache_row_store *) sobject;
			rs_tuple   *rs_tup;

			Assert(row_idx < trs->kern.nrows);
			rs_tup = kern_rowstore_get_tuple(&trs->kern, row_idx);

			ExecStoreTuple(&rs_tup->htup, slot, InvalidBuffer, false);
		}
		else if (StromTagIs(sobject, TCacheColumnStore))
		{
			tcache_column_store *tcs = (tcache_column_store *) sobject;
			TupleDesc	tupdesc = slot->tts_tupleDescriptor;

			slot = ExecStoreAllNullTuple(slot);
			Assert(row_idx < tcs->nrows);
			for (i=0; i < tupdesc->natts; i++)
			{
				Form_pg_attribute attr = tupdesc->attrs[i];

				if (!tcs->cdata[i].values)
					continue;
				/*
				 * NOTE: See bug #32. When we scan a underlying relation
				 * with not-null constratint using bulk-exec mode, its
				 * tcache_column_store does not have null-bitmap due to
				 * optimization. However, once a relation is scanned,
				 * TupleDesc of SubPlan lost information of this constraint.
				 * So, we consider all the column with no-nullmap are come
				 * from columns with not-null restrictions, thus, we have
				 * to care about tcs->cdata[].isnull also.
				 */
				if (!attr->attnotnull &&
					tcs->cdata[i].isnull &&
					att_isnull(row_idx, tcs->cdata[i].isnull))
					continue;

				if (attr->attlen > 0)
				{
					slot->tts_values[i] = fetch_att(tcs->cdata[i].values +
													attr->attlen * row_idx,
													attr->attbyval,
													attr->attlen);
				}
				else
				{
					cl_uint	   *cs_offset = (cl_uint *)tcs->cdata[i].values;

					Assert(cs_offset[row_idx] > 0);
					Assert(tcs->cdata[i].toast != NULL);
					slot->tts_values[i]
						= PointerGetDatum((char *)tcs->cdata[i].toast +
										  cs_offset[row_idx]);
				}
				slot->tts_isnull[i] = false;
			}
		}
		else
			elog(ERROR, "bug? neither row nor column store");

		if (x_chunk->scan_pos == kcs_x->nrows)
		{
			pgstrom_release_gpusort_chunk(x_chunk);
		}
		else
		{
			i = x_chunk->scan_pos;
			dlist_foreach(iter, &gsortstate->sorted_chunks)
			{
				y_chunk = dlist_container(pgstrom_gpusort_chunk,
										  chain,
										  iter.cur);
				kcs_y = KERN_GPUSORT_CHUNK(&y_chunk->kern);
				ktoast_y = KERN_GPUSORT_TOASTBUF(&y_chunk->kern);
				rindex_y = KERN_GPUSORT_RESULT_INDEX(kcs_y);
				j = y_chunk->scan_pos;

				if (pgstrom_compare_cpusort(gsortstate,
											kcs_x, ktoast_x, rindex_x[i],
											kcs_y, ktoast_y, rindex_y[j]) < 0)
				{
					dlist_insert_before(&y_chunk->chain, &x_chunk->chain);
					return slot;
				}
			}
			dlist_push_tail(&gsortstate->sorted_chunks, &x_chunk->chain);
		}
	}
	return slot;
}

static Node *
gpusort_exec_multi(CustomPlanState *node)
{
	elog(ERROR, "Not supported yet");
	return NULL;
}

static void
gpusort_end(CustomPlanState *node)
{
	GpuSortState	   *gsortstate = (GpuSortState *) node;
	dlist_mutable_iter	miter;

	/*
	 * Release PG-Strom shared objects
	 */
	pgstrom_untrack_object((StromObject *)gsortstate->dprog_key);
	pgstrom_put_devprog_key(gsortstate->dprog_key);

	pgstrom_untrack_object(&gsortstate->mqueue->sobj);
	pgstrom_close_queue(gsortstate->mqueue);

	Assert(gsortstate->num_running == 0);
	dlist_foreach_modify(miter, &gsortstate->sorted_chunks)
	{
		pgstrom_gpusort_chunk  *gs_chunk
			= dlist_container(pgstrom_gpusort_chunk, chain, miter.cur);

		dlist_delete(&gs_chunk->chain);
		pgstrom_release_gpusort_chunk(gs_chunk);
	}

	/*
     * clean out the tuple table
     */
    ExecClearTuple(gsortstate->scan_slot);
    ExecClearTuple(node->ps.ps_ResultTupleSlot);

	/*
     * shut down the subplan
     */
    ExecEndNode(outerPlanState(node));
}

static void
gpusort_rescan(CustomPlanState *node)
{
	GpuSortState	   *gsortstate = (GpuSortState *) node;
	dlist_mutable_iter	miter;

	/* If we haven't sorted yet, just return. */
	if (!gsortstate->sort_done)
		return;
	/* no asynchronous job should not be there */
	Assert(gsortstate->num_running == 0);

	/* must drop pointer to sort result tuple */
	ExecClearTuple(gsortstate->cps.ps.ps_ResultTupleSlot);

	/* right now, we just re-scan again */
	dlist_foreach_modify(miter, &gsortstate->sorted_chunks)
	{
		pgstrom_gpusort_chunk  *gs_chunk
			= dlist_container(pgstrom_gpusort_chunk, chain, miter.cur);

		dlist_delete(&gs_chunk->chain);
		pgstrom_release_gpusort_chunk(gs_chunk);
	}
	gsortstate->scan_done = false;
	gsortstate->sort_done = false;

	memset(gsortstate->rcs_slot, 0,
		   sizeof(StromObject *) * gsortstate->rcs_slotsz);
	gsortstate->rcs_nums = 0;

	/* also rescan underlying relation */
	ExecReScan(outerPlanState(&gsortstate->cps.ps));
}

static Bitmapset *
gpusort_get_relids(CustomPlanState *node)
{
	/*
	 * Backend recursively walks down the outerPlanState
	 */
	return NULL;
}

static void
gpusort_explain(CustomPlanState *node, List *ancestors, ExplainState *es)
{
	GpuSortState   *gsortstate = (GpuSortState *) node;
	GpuSortPlan	   *gsort = (GpuSortPlan *)gsortstate->cps.ps.plan;
	List		   *context;
	bool			useprefix;
	List		   *tlist = gsortstate->cps.ps.plan->targetlist;
	List		   *sort_keys = NIL;
	int				i;

	/* logic copied from show_sort_group_keys */
	context = deparse_context_for_planstate((Node *) gsortstate,
											ancestors,
											es->rtable,
											es->rtable_names);
	useprefix = (list_length(es->rtable) > 1 || es->verbose);

	for (i=0; i < gsort->numCols; i++)
	{
		AttrNumber	resno = gsort->sortColIdx[i];
		TargetEntry	*tle = get_tle_by_resno(tlist, resno);
		char		*exprstr;

		if (!tle)
			elog(ERROR, "no tlist entry for key %d", resno);

		/* Deparse the expression, showing any top-level cast */
		exprstr = deparse_expression((Node *) tle->expr, context,
									 useprefix, true);
		sort_keys = lappend(sort_keys, exprstr);
	}

	ExplainPropertyList("Sort keys", sort_keys, es);

	ExplainPropertyInteger("Sort keys width", gsortstate->sortkey_width, es);
	ExplainPropertyInteger("Rows per chunk", gsortstate->nrows_per_chunk, es);

	show_device_kernel(gsortstate->dprog_key, es);

	if (es->analyze && gsortstate->pfm.enabled)
		pgstrom_perfmon_explain(&gsortstate->pfm, es);
}

static void
gpusort_textout_plan(StringInfo str, const CustomPlan *node)
{
	const GpuSortPlan  *plannode = (const GpuSortPlan *) node;
	char	   *temp;
	int			i;

	appendStringInfo(str, " :kern_source");
	_outToken(str, plannode->kern_source);

	appendStringInfo(str, " :extra_flags %u",
					 plannode->extra_flags);

	temp = nodeToString(plannode->used_params);
	appendStringInfo(str, " :used_params %s", temp);
	pfree(temp);

	temp = nodeToString(plannode->sortkey_resnums);
	appendStringInfo(str, " :sortkey_resnums %s", temp);
	pfree(temp);

	appendStringInfo(str, " :sortkey_width %zu",
					 plannode->sortkey_width);

	appendStringInfo(str, " :numCols %d", plannode->numCols);

	appendStringInfoString(str, " :sortColIdx");
	for (i = 0; i < plannode->numCols; i++)
		appendStringInfo(str, " %d", plannode->sortColIdx[i]);

	appendStringInfoString(str, " :sortOperators");
	for (i = 0; i < plannode->numCols; i++)
		appendStringInfo(str, " %u", plannode->sortOperators[i]);

	appendStringInfoString(str, " :collations");
	for (i = 0; i < plannode->numCols; i++)
		appendStringInfo(str, " %u", plannode->collations[i]);

	appendStringInfoString(str, " :nullsFirst");
	for (i = 0; i < plannode->numCols; i++)
		appendStringInfo(str, " %s",
						 plannode->nullsFirst[i] ? "true" : "false");
}

static CustomPlan *
gpusort_copy_plan(const CustomPlan *from)
{
	const GpuSortPlan *oldnode = (const GpuSortPlan *)from;
	GpuSortPlan *newnode = palloc0(sizeof(GpuSortPlan));
	int		n = oldnode->numCols;

	CopyCustomPlanCommon((Node *)from, (Node *)newnode);
	newnode->kern_source = pstrdup(oldnode->kern_source);
	newnode->extra_flags = oldnode->extra_flags;
	newnode->used_params = copyObject(oldnode->used_params);
	newnode->sortkey_resnums = bms_copy(oldnode->sortkey_resnums);
	newnode->sortkey_width = oldnode->sortkey_width;

	newnode->numCols     = oldnode->numCols;
	newnode->sortColIdx = pmemcpy(oldnode->sortColIdx, sizeof(AttrNumber) * n);
	newnode->sortOperators = pmemcpy(oldnode->sortOperators, sizeof(Oid) * n);
	newnode->collations = pmemcpy(oldnode->collations, sizeof(Oid) * n);
	newnode->nullsFirst = pmemcpy(oldnode->nullsFirst, sizeof(bool) * n);

	return &newnode->cplan;
}

/*
 * initialization of GpuSort
 */
void
pgstrom_init_gpusort(void)
{
	/* configuration of gpusort-chunksize */
	DefineCustomIntVariable("pg_strom.gpusort_chunksize",
							"size of gpusort chunk in MB",
							NULL,
							&pgstrom_gpusort_chunksize,
							0,		/* auto adjustment */
							0,
							INT_MAX,
							PGC_SUSET,
							GUC_NOT_IN_SAMPLE,
							NULL, NULL, NULL);

	/* initialization of plan method table */
	gpusort_plan_methods.CustomName = "GpuSort";
	gpusort_plan_methods.SetCustomPlanRef = gpusort_set_plan_ref;
	gpusort_plan_methods.SupportBackwardScan = NULL;
	gpusort_plan_methods.FinalizeCustomPlan	= gpusort_finalize_plan;
	gpusort_plan_methods.BeginCustomPlan	= gpusort_begin;
	gpusort_plan_methods.ExecCustomPlan		= gpusort_exec;
	gpusort_plan_methods.MultiExecCustomPlan = gpusort_exec_multi;
	gpusort_plan_methods.EndCustomPlan		= gpusort_end;
	gpusort_plan_methods.ReScanCustomPlan	= gpusort_rescan;
	gpusort_plan_methods.ExplainCustomPlanTargetRel = NULL;
	gpusort_plan_methods.ExplainCustomPlan	= gpusort_explain;
	gpusort_plan_methods.GetRelidsCustomPlan= gpusort_get_relids;
	gpusort_plan_methods.GetSpecialCustomVar = NULL;
	gpusort_plan_methods.TextOutCustomPlan	= gpusort_textout_plan;
	gpusort_plan_methods.CopyCustomPlan		= gpusort_copy_plan;
}

/* ----------------------------------------------------------------
 *
 * NOTE: below is the code being run on OpenCL server context
 *
 * ---------------------------------------------------------------- */
typedef struct
{
	pgstrom_message	*msg;
	cl_program		program;
	cl_mem			m_chunk;
	cl_kernel	   *prep_kernel;
	cl_kernel	   *sort_kernel;
	cl_int			dindex;
	cl_int			prep_nums;
	cl_int			sort_nums;
	Size			bytes_dma_send;
	Size			bytes_dma_recv;
	cl_int			ev_index;
	cl_event		events[20];
} clstate_gpusort_single;

static void
clserv_respond_gpusort_single(cl_event event, cl_int ev_status, void *private)
{
	clstate_gpusort_single *clgss = private;
	pgstrom_gpusort		   *gpusort = (pgstrom_gpusort *)clgss->msg;
	pgstrom_gpusort_chunk  *gs_chunk;
	cl_int					i, rc;

	/* gpusort-single has only one input chunk; also be an output chunk */
	Assert(dlist_length(&gpusort->gs_chunks) == 1);
	gs_chunk = dlist_container(pgstrom_gpusort_chunk, chain,
							   dlist_head_node(&gpusort->gs_chunks));
	/* put an error code */
	if (ev_status != CL_COMPLETE)
	{
		clserv_log("unexpected CL_EVENT_COMMAND_EXECUTION_STATUS: %d",
				   ev_status);
		gpusort->msg.errcode = StromError_OpenCLInternal;
	}
	else
	{
		gpusort->msg.errcode = *KERN_GPUSORT_STATUS(&gs_chunk->kern);
		if (gpusort->msg.errcode == StromError_Success)
			gpusort->is_sorted = true;
	}

	/* collect performance statistics */
	if (gpusort->msg.pfm.enabled)
	{
		cl_ulong	tv1, tv2;

		for (i=0; i < clgss->ev_index; i++)
		{
			rc = clGetEventProfilingInfo(clgss->events[i],
                                         CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong),
                                         &tv1,
                                         NULL);
            if (rc != CL_SUCCESS)
				break;

            rc = clGetEventProfilingInfo(clgss->events[i],
                                         CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong),
                                         &tv2,
                                         NULL);
            if (rc != CL_SUCCESS)
				break;

			/* first two events are DMA send */
			if (i < 2)
				gpusort->msg.pfm.time_dma_send += (tv2 - tv1) / 1000;
			/* its kernel execution */
			else if (i == clgss->ev_index - 2)
				gpusort->msg.pfm.time_kern_exec += (tv2 - tv1) / 1000;
			/* its DMA recv */
			else if (i == clgss->ev_index - 1)
				gpusort->msg.pfm.time_dma_recv += (tv2 - tv1) / 1000;
			/* DMA send of row-/column-store */
			else if (i % 2 == 0)
				gpusort->msg.pfm.time_dma_send += (tv2 - tv1) / 1000;
			/* Elsewhere setting up the chunk */
			else
				gpusort->msg.pfm.time_kern_exec += (tv2 - tv1) / 1000;
		}
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clGetEventProfilingInfo (%s)",
					   opencl_strerror(rc));
			gpusort->msg.pfm.enabled = false;	/* turn off profiling */
		}
	}

	/* release opencl objects */
	while (clgss->ev_index > 0)
		clReleaseEvent(clgss->events[--clgss->ev_index]);
	for (i=0; i < clgss->prep_nums; i++)
		clReleaseKernel(clgss->prep_kernel[i]);
	for (i=0; i < clgss->sort_nums; i++)
		clReleaseKernel(clgss->sort_kernel[i]);
	clReleaseMemObject(clgss->m_chunk);

	free(clgss->prep_kernel);
	free(clgss->sort_kernel);
	free(clgss);

	/* respond to the backend side */
	pgstrom_reply_message(&gpusort->msg);
}

static void
clserv_respond_gpusort_kmem(cl_event event, cl_int ev_status, void *private)
{
	cl_int	rc;

	rc = clReleaseMemObject((cl_mem)private);
	if (rc != CL_SUCCESS)
		clserv_log("failed on clReleaseMemObject: %s", opencl_strerror(rc));
	//clserv_log("release buffer object: %p", private);
}

static void
clserv_respond_gpusort_shmem(cl_event event, cl_int ev_status, void *private)
{
	pgstrom_shmem_free(private);
	//clserv_log("release shmem: %p", private);
}

/*
 * opencl kernel invocation of:
 *
 * __kernel void
 * gpusort_setup_chunk_rs(cl_uint rcs_gindex,
 *                        __global kern_gpusort *kgpusort,
 *                        __global kern_row_store *krs,
 *                        cl_uint src_nitems,
 *                        __global cl_int *src_rindex,
 *                        __local void *local_workmem)
 */
static cl_kernel
clserv_launch_gpusort_setup_row(clstate_gpusort_single *clgss,
								tcache_row_store *trs,
								cl_uint rcs_gindex,
								cl_int src_nitems,
								cl_int *src_rindex)
{
	cl_command_queue kcmdq = opencl_cmdq[clgss->dindex];
	cl_kernel		prep_kernel;
	cl_mem			m_rstore;
	size_t			length;
	size_t			gwork_sz;
	size_t			lwork_sz;
	cl_int			rc;
	cl_int			n_blocker = 1;

	length = STROMALIGN(trs->kern.length);
	if (src_rindex)
		length += STROMALIGN(sizeof(cl_uint) * src_nitems);
	else
		src_nitems = -1;

	m_rstore = clCreateBuffer(opencl_context,
							  CL_MEM_READ_WRITE,
							  length,
							  NULL,
							  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error_0;
	}

	prep_kernel = clCreateKernel(clgss->program,
								 "gpusort_setup_chunk_rs",
								 &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		goto error_1;
	}

	/* set optimal global/local workgroup size */
	if (!clserv_compute_workgroup_size(&gwork_sz, &lwork_sz,
									   prep_kernel,
									   clgss->dindex,
									   !src_rindex
									   ? trs->kern.nrows
									   : src_nitems,
									   sizeof(cl_uint)))
		goto error_2;

	rc = clSetKernelArg(prep_kernel,
						0,	/* cl_uint rcs_gindex */
						sizeof(cl_uint),
						&rcs_gindex);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_2;
	}

	rc = clSetKernelArg(prep_kernel,
						1,	/* kern_gpusort *kgsort */
						sizeof(cl_mem),
						&clgss->m_chunk);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_2;
	}

	rc = clSetKernelArg(prep_kernel,
						2,	/* kern_row_store *krs */
						sizeof(cl_mem),
						&m_rstore);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_2;
	}

	rc = clSetKernelArg(prep_kernel,
						3,	/* cl_uint src_nitems */
						sizeof(cl_uint),
						&src_nitems);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_2;
	}

	rc = clSetKernelArg(prep_kernel,
						4,	/* local_workmem */
						sizeof(cl_uint) * (lwork_sz + 1),
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_2;
	}

	/*
	 * Send contents of row-store
	 */
	rc = clEnqueueWriteBuffer(kcmdq,
							  m_rstore,
							  CL_FALSE,
							  0,
							  trs->kern.length,
							  &trs->kern,
							  2,
							  &clgss->events[0],
							  &clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s",
				   opencl_strerror(rc));
		goto error_2;
	}
	clgss->ev_index++;
	clgss->bytes_dma_send += trs->kern.length;

	/*
	 * Send contents of rindex (if needed)
	 */
	if (src_rindex && src_nitems > 0)
	{
		rc = clEnqueueWriteBuffer(kcmdq,
								  m_rstore,
								  CL_FALSE,
								  STROMALIGN(trs->kern.length),
								  sizeof(cl_uint) * src_nitems,
								  src_rindex,
								  2,
								  &clgss->events[0],
								  &clgss->events[clgss->ev_index]);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clEnqueueWriteBuffer: %s",
					   opencl_strerror(rc));
			goto error_2;
		}
		clgss->ev_index++;
		clgss->bytes_dma_send += sizeof(cl_uint) * src_nitems;
		n_blocker++;
	}

	/*
	 * Enqueue a kernel execution on this row-store.
	 */
	rc = clEnqueueNDRangeKernel(kcmdq,
								prep_kernel,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								n_blocker,
								&clgss->events[clgss->ev_index - n_blocker],
								&clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		goto error_3;
	}
	clgss->ev_index++;

	/*
	 * in-kernel row-store can be released immediately, once
	 * its sortkeys are copied to gpusort-chunk
	 */
	rc = clSetEventCallback(clgss->events[clgss->ev_index - 1],
							CL_COMPLETE,
							clserv_respond_gpusort_kmem,
							m_rstore);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetEventCallback: %s", opencl_strerror(rc));
		goto error_3;
	}
	return prep_kernel;
						
error_3:
	clWaitForEvents(clgss->ev_index, clgss->events);
error_2:
	clReleaseKernel(prep_kernel);
error_1:
	clReleaseMemObject(m_rstore);
error_0:
	return NULL;
}

/*
 * opencl kernel invocation of:
 *
 * __kernel void
 * gpusort_setup_chunk_cs(cl_uint rcs_gindex,
 *                        __global kern_gpusort *kgpusort,
 *                        __global kern_row_store *krs,
 *                        cl_uint src_nitems,
 *                        __global cl_int *src_rindex,
 *                        __local void *local_workmem)
 */
static cl_kernel
clserv_launch_gpusort_setup_column(clstate_gpusort_single *clgss,
								   tcache_column_store *tcs,
								   cl_uint rcs_gindex,
								   cl_int src_nitems,
								   cl_int *src_rindex,
								   bytea *kparam_0,
								   bytea *kparam_1)
{
	cl_command_queue	kcmdq = opencl_cmdq[clgss->dindex];
	kern_column_store  *kcs_tmpl;
	kern_column_store  *kcs_head;
	kern_toastbuf	   *ktoast_head;
	cl_char			   *attrefs;
	cl_mem				m_cstore = NULL;
	cl_kernel			prep_kernel = NULL;
	size_t				gwork_sz;
	size_t				lwork_sz;
	Size				length;
	Size				offset;
	Size				kcs_offset;
	Size				ktoast_offset;
	cl_uint				rs_natts;
	cl_uint				i_col;
	cl_uint				ncols;
	cl_uint				kcs_nitems;
	cl_int				i, rc;
	cl_int				ev_index_base;

	/*
	 * First of all, we need to set up header portion of kern_column_store
	 * on the source side, according to the template stuff.
	 * kparam_0 informs us which columns are referenced, and kparam_1 gives
	 * us template of kcs_head but needs to adjust its colmeta.
	 */
	attrefs = (cl_char *)VARDATA(kparam_0);
	rs_natts = VARSIZE_ANY_EXHDR(kparam_0);
	Assert(rs_natts == tcs->ncols);
	kcs_tmpl = (kern_column_store *)VARDATA(kparam_1);
	ncols = kcs_tmpl->ncols;
	kcs_nitems = tcs->nrows;

	kcs_offset = STROMALIGN(offsetof(kern_column_store, colmeta[ncols]));
	length = kcs_offset + STROMALIGN(offsetof(kern_toastbuf, coldir[ncols]));
	/* allocation of DMA source */
	kcs_head = pgstrom_shmem_alloc(length);
	if (!kcs_head)
	{
		clserv_log("out of shared memory");
		return NULL;
	}
	memcpy(kcs_head, kcs_tmpl, kcs_offset);
	kcs_head->nrows = kcs_nitems;
	kcs_head->nrooms = kcs_nitems;

	ktoast_head = (kern_toastbuf *)((char *)kcs_head + kcs_offset);
	ktoast_head->length = TOASTBUF_MAGIC;
	ktoast_head->ncols = ncols;
	ktoast_offset =  STROMALIGN(offsetof(kern_toastbuf, coldir[ncols]));

	/* calculate and update offset */
	i_col = 0;
	for (i=0; i < tcs->ncols; i++)
	{
		kern_colmeta   *colmeta;

		if (!attrefs[i])
			continue;

		colmeta = &kcs_head->colmeta[i_col];
		//clserv_log("colmeta[%d] {attnotnull=%d, attalign=%d, attlen=%d, cs_ofs=%u}", i_col, colmeta->attnotnull, colmeta->attalign, colmeta->attlen, colmeta->cs_ofs);
		colmeta->cs_ofs = kcs_offset;
		if (!colmeta->attnotnull)
			kcs_offset += STROMALIGN((kcs_nitems +
									  BITS_PER_BYTE - 1) / BITS_PER_BYTE);
		kcs_offset += STROMALIGN((colmeta->attlen > 0
								  ? colmeta->attlen
								  : sizeof(cl_uint)) * kcs_nitems);

		if (kcs_head->colmeta[i_col].attlen > 0)
			ktoast_head->coldir[i_col] = (cl_uint)(-1);
		else
		{
			ktoast_head->coldir[i_col] = ktoast_offset;
			ktoast_offset += STROMALIGN(tcs->cdata[i].toast->tbuf_usage);
		}
		i_col++;

		if (attrefs[i] < 0)
			break;
	}
	/*
	 * growid and rindex of the source kcs; kern_column_to_column assumes
	 * source and destination column-store are symmetric, so we need to
	 * assign a certain amount of region even if it is not actually in-use.
	 */
	kcs_head->colmeta[i_col].cs_ofs = kcs_offset;
	ktoast_head->coldir[i_col] = (cl_uint)(-1);
	i_col++;
	//kcs_offset += STROMALIGN(sizeof(cl_long) * src_nitems);

	kcs_head->colmeta[i_col].cs_ofs = kcs_offset;
	ktoast_head->coldir[i_col] = (cl_uint)(-1);
	i_col++;
	if (src_rindex)
		kcs_offset += STROMALIGN(sizeof(cl_int) * kcs_nitems);
	else
		src_nitems = -1;
	kcs_head->length = kcs_offset;
	Assert(i_col == ncols);

	/*
	 * allocation of source kernel column store on device memory
	 */
	m_cstore = clCreateBuffer(opencl_context,
							  CL_MEM_READ_WRITE,
							  kcs_offset + ktoast_offset,
							  NULL,
							  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error_release;
	}

	prep_kernel = clCreateKernel(clgss->program,
								 "gpusort_setup_chunk_cs",
								 &rc);
	if (rc != CL_SUCCESS)
	{
		goto error_release;
	}

	/* set optimal global/local workgroup size */
    if (!clserv_compute_workgroup_size(&gwork_sz, &lwork_sz,
                                       prep_kernel,
                                       clgss->dindex,
                                       !src_rindex
                                       ? kcs_nitems
                                       : src_nitems,
                                       sizeof(cl_uint)))
		goto error_release;

	/* OK, setting up kernel arguments */
	rc = clSetKernelArg(prep_kernel,
						0,	/* cl_uint rcs_gindex */
						sizeof(cl_uint),
						&rcs_gindex);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_release;
	}

	rc = clSetKernelArg(prep_kernel,
						1,	/* kern_gpusort *kgsort */
						sizeof(cl_mem),
						&clgss->m_chunk);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_release;
	}

	rc = clSetKernelArg(prep_kernel,
						2,	/* kern_column_store *kcs */
						sizeof(cl_mem),
						&m_cstore);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_release;
	}

	rc = clSetKernelArg(prep_kernel,
						3,	/* cl_uint src_nitems */
						sizeof(cl_uint),
						&src_nitems);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_release;
	}

	rc = clSetKernelArg(prep_kernel,
						4,	/* local_workmem */
						sizeof(cl_uint) * (lwork_sz + 1),
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_release;
	}

	/*
	 * OK, begin to enqueue DMA request
	 */
	ev_index_base = clgss->ev_index;

	/* kcs_head - header portion of kern_column_store */
	length = offsetof(kern_column_store, colmeta[ncols]);
	rc = clEnqueueWriteBuffer(kcmdq,
							  m_cstore,
							  CL_FALSE,
							  0,
							  length,
							  kcs_head,
							  0,
							  NULL,
							  &clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s",
				   opencl_strerror(rc));
		goto error_release;
	}
	clgss->ev_index++;
    clgss->bytes_dma_send += length;

	/* ktoast_head - header portion of kern_toastbuf */
	length = offsetof(kern_toastbuf, coldir[ncols]);
	offset = kcs_head->length;
	rc = clEnqueueWriteBuffer(kcmdq,
							  m_cstore,
							  CL_FALSE,
							  offset,
							  length,
							  ktoast_head,
							  0,
							  NULL,
							  &clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s",
				   opencl_strerror(rc));
		goto error_sync;
	}
	clgss->ev_index++;
	clgss->bytes_dma_send += length;

	i_col = 0;
	for (i=0; i < rs_natts; i++)
	{
		kern_colmeta   *colmeta;

		if (!attrefs[i])
			continue;

		colmeta = &kcs_head->colmeta[i_col];
		/* DMA send - null-bitmap and values of column store */
		offset = colmeta->cs_ofs;
		if (!colmeta->attnotnull)
		{
			length = (kcs_nitems + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
			if (tcs->cdata[i].isnull)
			{
				rc = clEnqueueWriteBuffer(kcmdq,
										  m_cstore,
										  CL_FALSE,
										  offset,
										  length,
										  tcs->cdata[i].isnull,
										  0,
										  NULL,
										  &clgss->events[clgss->ev_index]);
				if (rc != CL_SUCCESS)
				{
					clserv_log("failed on clEnqueueWriteBuffer: %s",
							   opencl_strerror(rc));
					goto error_sync;
				}
			}
			else
			{
				cl_uint	nullmap = (cl_uint)(-1);

				rc = clEnqueueFillBuffer(kcmdq,
										 m_cstore,
										 &nullmap,
										 sizeof(cl_uint),
										 offset,
										 INTALIGN(length),
										 0,
										 NULL,
										 &clgss->events[clgss->ev_index]);
				if (rc != CL_SUCCESS)
				{
					clserv_log("failed on clEnqueueFillBuffer: %s",
							   opencl_strerror(rc));
					goto error_sync;
				}
			}
			clgss->ev_index++;
			clgss->bytes_dma_send += length;

			offset += STROMALIGN(length);
		}
		length = (colmeta->attlen > 0
				  ? colmeta->attlen
				  : sizeof(cl_uint)) * kcs_nitems;
		rc = clEnqueueWriteBuffer(kcmdq,
								  m_cstore,
								  CL_FALSE,
								  offset,
								  length,
								  tcs->cdata[i].values,
								  0,
								  NULL,
								  &clgss->events[clgss->ev_index]);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clEnqueueWriteBuffer: %s",
					   opencl_strerror(rc));
			goto error_sync;
		}
		clgss->ev_index++;
		clgss->bytes_dma_send += length;

		/* DMA send - toast buffer of column store, if exists */
		if (tcs->cdata[i].toast)
		{
			Assert(colmeta->attlen < 0);
			offset = kcs_head->length + ktoast_head->coldir[i_col];
			length = tcs->cdata[i].toast->tbuf_usage;
			rc = clEnqueueWriteBuffer(kcmdq,
									  m_cstore,
									  CL_FALSE,
									  offset,
									  length,
									  tcs->cdata[i].toast,
									  0,
									  NULL,
									  &clgss->events[clgss->ev_index]);
			if (rc != CL_SUCCESS)
			{
				clserv_log("failed on clEnqueueWriteBuffer: %s",
						   opencl_strerror(rc));
				goto error_sync;
			}
			clgss->ev_index++;
			clgss->bytes_dma_send += length;
		}

		/* is it last column being referenced? */
		if (attrefs[i] < 0)
			break;
		i_col++;
	}
	/* DMA send - rindex of the column store, if exists */
	if (src_rindex)
	{
		kern_colmeta   *colmeta = &kcs_head->colmeta[ncols - 1];

		offset = colmeta->cs_ofs;
		length = sizeof(cl_uint) * src_nitems;
		rc = clEnqueueWriteBuffer(kcmdq,
								  m_cstore,
								  CL_FALSE,
								  offset,
								  length,
								  src_rindex,
								  0,
								  NULL,
								  &clgss->events[clgss->ev_index]);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clEnqueueWriteBuffer: %s",
					   opencl_strerror(rc));
			goto error_sync;
		}
		clgss->ev_index++;
		clgss->bytes_dma_send += length;
	}

	/*
	 * Kick gpusort_setup_chunk_cs() call
	 */
	rc = clEnqueueNDRangeKernel(kcmdq,
								prep_kernel,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								clgss->ev_index - ev_index_base,
								&clgss->events[ev_index_base],
								&clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		goto error_sync;
    }
	clgss->ev_index++;

	/* set callback to release shmem */
	rc = clSetEventCallback(clgss->events[clgss->ev_index - 1],
                            CL_COMPLETE,
							clserv_respond_gpusort_shmem,
							kcs_head);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetEventCallback: %s", opencl_strerror(rc));
		goto error_sync;
	}

	/* set callback to release m_cstore */
	rc = clSetEventCallback(clgss->events[clgss->ev_index - 1],
							CL_COMPLETE,
							clserv_respond_gpusort_kmem,
							m_cstore);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetEventCallback: %s", opencl_strerror(rc));
		goto error_sync;
	}
	return prep_kernel;

error_sync:
	clWaitForEvents(clgss->ev_index, clgss->events);
error_release:
	if (prep_kernel)
		clReleaseKernel(prep_kernel);
	if (m_cstore)
		clReleaseMemObject(m_cstore);
	pgstrom_shmem_free(kcs_head);
	return NULL;
}

/*
 * opencl kernel invocation of:
 *
 * __kernel void
 * gpusort_single(cl_int bitonic_unitsz,
 *                __global kern_gpusort *kgsort,
 *                __local void *local_workbuf)
 */
static cl_kernel
clserv_launch_gpusort_bitonic_step(clstate_gpusort_single *clgss,
								   cl_uint nrows, bool reversing,
								   cl_uint unitsz, bool is_first)
{
	cl_command_queue kcmdq = opencl_cmdq[clgss->dindex];
	cl_kernel	sort_kernel;
	size_t		lwork_sz;
	size_t		gwork_sz;
	size_t		unitlen;
	size_t		n_threads;
	cl_int		bitonic_unitsz;
	cl_int		rc;

	sort_kernel = clCreateKernel(clgss->program,
								 "gpusort_single_step",
								 &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		goto error_0;
	}

	/* Set optimal global/local workgroup size */
	unitlen = (1 << unitsz);
	n_threads = ((nrows + unitlen - 1) & ~(unitlen - 1)) / 2;
	if (!clserv_compute_workgroup_size(&gwork_sz, &lwork_sz,
									   sort_kernel,
									   clgss->dindex,
									   n_threads,
									   sizeof(cl_uint)))
		goto error_1;

	//clserv_log("kernel call (nrows=%u, unitlen=%zu, lworksz=%zu, gworksz=%zu)", nrows, unitlen, lwork_sz, gwork_sz);

	/* pack reversing and unitsz into one argument */
	if (reversing)
		bitonic_unitsz = -unitsz;
	else
		bitonic_unitsz = unitsz;

	rc = clSetKernelArg(sort_kernel,
						0,	/* cl_int bitonic_unitsz */
						sizeof(cl_int),
						&bitonic_unitsz);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_1;
	}

	rc = clSetKernelArg(sort_kernel,
						1,	/* kern_gpusort *kgsort */
						sizeof(cl_mem),
						&clgss->m_chunk);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_1;
	}

	rc = clSetKernelArg(sort_kernel,
						2,	/* void *local_workbuf */
						sizeof(cl_int) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_1;
	}

	rc = clEnqueueNDRangeKernel(kcmdq,
								sort_kernel,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								is_first ? clgss->ev_index : 1,
								is_first
								? &clgss->events[0]
								: &clgss->events[clgss->ev_index - 1],
								&clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		goto error_1;
	}
	clgss->ev_index++;

	return sort_kernel;

error_1:
	clReleaseKernel(sort_kernel);
error_0:
	return NULL;
}

static cl_kernel
clserv_launch_gpusort_bitonic_marge(clstate_gpusort_single *clgss,
									cl_uint nrows, cl_uint unitsz,
									bool is_first)
{
	cl_command_queue kcmdq = opencl_cmdq[clgss->dindex];
	cl_kernel	sort_kernel;
	size_t		lwork_sz;
	size_t		gwork_sz;
	size_t		unitlen;
	cl_int		rc;

	sort_kernel = clCreateKernel(clgss->program,
								 "gpusort_single_marge",
								 &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		goto error_0;
	}

	/* Set optimal global/local workgroup size */
	unitlen  = (1 << unitsz);
	lwork_sz = unitlen / 2;
	gwork_sz = ((nrows + unitlen - 1) & ~(unitlen - 1)) / 2;

	// clserv_log("kernel call (nrows=%u, unitlen=%zu, lworksz=%zu, gworksz=%zu)", nrows, unitlen, lwork_sz, gwork_sz);

	rc = clSetKernelArg(sort_kernel,
						0,	/* kern_gpusort *kgsort */
						sizeof(cl_mem),
						&clgss->m_chunk);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_1;
	}

	rc = clSetKernelArg(sort_kernel,
						1,	/* void *local_workbuf */
						sizeof(cl_int) * lwork_sz * 2,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_1;
	}

	rc = clEnqueueNDRangeKernel(kcmdq,
								sort_kernel,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								is_first ? clgss->ev_index : 1,
								is_first
								? &clgss->events[0]
								: &clgss->events[clgss->ev_index - 1],
								&clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		goto error_1;
	}
	clgss->ev_index++;

	return sort_kernel;

error_1:
	clReleaseKernel(sort_kernel);
error_0:
	return NULL;
}

static cl_kernel
clserv_launch_gpusort_bitonic_sort(clstate_gpusort_single *clgss,
								   cl_uint nrows, cl_uint unitsz,
								   bool is_first)
{
	cl_command_queue kcmdq = opencl_cmdq[clgss->dindex];
	cl_kernel	sort_kernel;
	size_t		lwork_sz;
	size_t		gwork_sz;
	size_t		unitlen;
	cl_int		rc;

	sort_kernel = clCreateKernel(clgss->program,
								 "gpusort_single_sort",
								 &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		goto error_0;
	}

	/* Set optimal global/local workgroup size */
	unitlen  = (1 << unitsz);
	lwork_sz = unitlen / 2;
	gwork_sz = ((nrows + unitlen - 1) & ~(unitlen - 1)) / 2;

	// clserv_log("kernel call (nrows=%u, unitlen=%zu, lworksz=%zu, gworksz=%zu)", nrows, unitlen, lwork_sz, gwork_sz);

	rc = clSetKernelArg(sort_kernel,
						0,	/* kern_gpusort *kgsort */
						sizeof(cl_mem),
						&clgss->m_chunk);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_1;
	}

	rc = clSetKernelArg(sort_kernel,
						1,	/* void *local_workbuf */
						sizeof(cl_int) * lwork_sz * 2,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error_1;
	}

	rc = clEnqueueNDRangeKernel(kcmdq,
								sort_kernel,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								is_first ? clgss->ev_index : 1,
								is_first
								? &clgss->events[0]
								: &clgss->events[clgss->ev_index - 1],
								&clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		goto error_1;
	}
	clgss->ev_index++;

	return sort_kernel;

error_1:
	clReleaseKernel(sort_kernel);
error_0:
	return NULL;
}

static void
clserv_process_gpusort_single(pgstrom_gpusort *gpusort)
{
	pgstrom_gpusort_chunk  *gs_chunk;
	kern_parambuf		   *kparams;
	bytea				   *kparam_0;
	bytea				   *kparam_1;
	kern_column_store	   *kcs;
	clstate_gpusort_single *clgss;
	cl_command_queue		kcmdq;
	cl_int				   *rindex;
	cl_uint					dindex;
	cl_uint					nrows;
	cl_uint					prep_nums;
	cl_uint					sort_nums;
	cl_uint					sort_size;
	cl_uint					event_nums;
	cl_int					i, j, rc;
	size_t					length;
	size_t					offset;

	/* only pgstom_gpusort (sorting chunk) should be attached */
	gs_chunk = dlist_container(pgstrom_gpusort_chunk, chain,
							   dlist_head_node(&gpusort->gs_chunks));
	kparams = KERN_GPUSORT_PARAMBUF(&gs_chunk->kern);
	kcs = KERN_GPUSORT_CHUNK(&gs_chunk->kern);
	prep_nums = gs_chunk->rcs_nums;

	/* also, fetch kparam_0 for column->column translation */
	Assert(kparams->nparams >= 2 &&
		   kparams->poffset[0] > 0 &&
		   kparams->poffset[1] > 0);
	kparam_0 = (bytea *)((char *)kparams + kparams->poffset[0]);
	kparam_1 = (bytea *)((char *)kparams + kparams->poffset[1]);

	nrows = 0;
	for (i=0; i < prep_nums; i++)
	{
		if (gs_chunk->rcs_slot[i].rindex)
			nrows += gs_chunk->rcs_slot[i].nitems;
		else
		{
			StromObject *sobject = gs_chunk->rcs_slot[i].rcstore;

			if (StromTagIs(sobject, TCacheRowStore))
				nrows += ((tcache_row_store *) sobject)->kern.nrows;
			else if (StromTagIs(sobject, TCacheColumnStore))
				nrows += ((tcache_column_store *) sobject)->nrows;
			else
			{
				rc = StromError_BadRequestMessage;
				goto error;
			}
		}
	}
	sort_size = get_next_log2(nrows);
	sort_nums = (sort_size * (sort_size + 1)) / 2;

	/* state object of gpuscan (single chunk) */
	event_nums = 3 * prep_nums + sort_nums + 4 + 10000;
	clgss = calloc(1, offsetof(clstate_gpusort_single,
							   events[event_nums]));
	if (!clgss ||
		!(clgss->prep_kernel = calloc(prep_nums, sizeof(cl_kernel))) ||
		!(clgss->sort_kernel = calloc(sort_nums, sizeof(cl_kernel))))
	{
		rc = CL_OUT_OF_HOST_MEMORY;
		goto error_release;
	}
	clgss->msg = &gpusort->msg;
	clgss->sort_nums = sort_nums;	/* scale of sorting */
	clgss->prep_nums = prep_nums;	/* number of preparation call */

	/*
	 * Choose a device to execute this kernel
	 */
	dindex = pgstrom_opencl_device_schedule(&gpusort->msg);
	clgss->dindex = dindex;
	kcmdq = opencl_cmdq[dindex];

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
		free(clgss->prep_kernel);
		free(clgss->sort_kernel);
		free(clgss);
		return;		/* message is in waitq, retry it! */
	}
	if (clgss->program == BAD_OPENCL_PROGRAM)
	{
		rc = CL_BUILD_PROGRAM_FAILURE;
		goto error_release;
	}

	/*
	 * Preparation (1)
	 *
	 * allocation of in-kernel sorting chunk, and translation of
	 * management data field.
	 */
	length = (KERN_GPUSORT_PARAMBUF_LENGTH(&gs_chunk->kern) +
			  KERN_GPUSORT_CHUNK_LENGTH(&gs_chunk->kern) +
			  KERN_GPUSORT_STATUS_LENGTH(&gs_chunk->kern) +
			  KERN_GPUSORT_TOASTBUF_LENGTH(&gs_chunk->kern));
	clgss->m_chunk = clCreateBuffer(opencl_context,
									CL_MEM_READ_WRITE,
									length,
									NULL,
									&rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}

	/*
	 * Send header portion of kparam, kcs, status and toast buffer.
	 * Note that role of the kernel is to set up contents of kcs
	 * and toast buffer, so no need to translate the contents itself
	 */

	/* kparam + header of kern_column_store */
	length = (KERN_GPUSORT_PARAMBUF_LENGTH(&gs_chunk->kern) +
			  STROMALIGN(offsetof(kern_column_store, colmeta[kcs->ncols])));
	rc = clEnqueueWriteBuffer(kcmdq,
							  clgss->m_chunk,
							  CL_FALSE,
							  0,
							  length,
							  kparams,
							  0,
							  NULL,
							  &clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		Assert(false);
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}
	clgss->ev_index++;
	clgss->bytes_dma_send += length;

	/* rindex array */
	if (gpusort->has_rindex)
	{
		rindex = KERN_GPUSORT_RESULT_INDEX(kcs);
		length = sizeof(cl_uint) * kcs->nrows;
		offset = (uintptr_t)rindex - (uintptr_t)(&gs_chunk->kern);
		rc = clEnqueueWriteBuffer(kcmdq,
								  clgss->m_chunk,
								  CL_FALSE,
								  offset,
								  length,
								  rindex,
								  0,
								  NULL,
								  &clgss->events[clgss->ev_index]);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clEnqueueWriteBuffer: %s",
					   opencl_strerror(rc));
			goto error_sync;
		}
		clgss->ev_index++;
		clgss->bytes_dma_send += length;
	}

	/* kstatus + header portion of toastbuf */
	length = (KERN_GPUSORT_STATUS_LENGTH(&gs_chunk->kern) +
			  STROMALIGN(offsetof(kern_toastbuf, coldir[0])));
	offset = ((uintptr_t)KERN_GPUSORT_STATUS(&gs_chunk->kern) -
			  (uintptr_t)(&gs_chunk->kern));
	rc = clEnqueueWriteBuffer(kcmdq,
							  clgss->m_chunk,
							  CL_FALSE,
							  offset,
							  length,
							  KERN_GPUSORT_STATUS(&gs_chunk->kern),
							  0,
							  NULL,
							  &clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}
	clgss->ev_index++;
	clgss->bytes_dma_send += length;

	/*
	 * Preparation for each row-/column-store
	 */
	for (i=0; i < prep_nums; i++)
	{
		StromObject *sobject = gs_chunk->rcs_slot[i].rcstore;
		cl_uint		rcs_gindex = gs_chunk->rcs_head + i;
		cl_uint		src_nitems = gs_chunk->rcs_slot[i].nitems;
		cl_int	   *src_rindex = gs_chunk->rcs_slot[i].rindex;
		cl_kernel	prep_kernel;

		if (StromTagIs(sobject, TCacheRowStore))
		{
			tcache_row_store *trs = (tcache_row_store *) sobject;
			prep_kernel = clserv_launch_gpusort_setup_row(clgss, trs,
														  rcs_gindex,
														  src_nitems,
														  src_rindex);
			if (!prep_kernel)
				goto error_sync;
		}
		else if (StromTagIs(sobject, TCacheColumnStore))
		{
			tcache_column_store *tcs = (tcache_column_store *) sobject;
			prep_kernel = clserv_launch_gpusort_setup_column(clgss, tcs,
															 rcs_gindex,
															 src_nitems,
															 src_rindex,
															 kparam_0,
															 kparam_1);
			if (!prep_kernel)
				goto error_sync;
		}
		else
		{
			rc = StromError_BadRequestMessage;
			goto error_sync;
		}
		clgss->prep_kernel[i] = prep_kernel;
	}

	/*
	 * OK, preparation was done. Let's launch gpusort_single kernel
	 * to sort key values within a gpusort-chunk.
	 */
	if (sort_size > 0)
	{
		/*
		 * FIXME: optimal size estimation should be integrated to
		 * clserv_compute_workgroup_size(), but tentatively we
		 * have logic here...
		 */
		const pgstrom_device_info *devinfo =
			pgstrom_get_device_info(clgss->dindex);
		size_t local_worksz    = devinfo->dev_max_work_item_sizes[0];
		size_t local_memsz     = devinfo->dev_local_mem_size;
		size_t lmem_per_thread = 2 * sizeof(cl_int);
		size_t max_threads     = ((local_worksz < local_memsz/lmem_per_thread)
								  ? local_worksz
								  : (local_memsz/lmem_per_thread));
		int max_sort_size      = LOG2(max_threads) + 1;
		cl_int k               = 0;

		cl_kernel sort_kernel;
		int prt_size;

		prt_size = (sort_size < max_sort_size) ? sort_size : max_sort_size;
		sort_kernel = clserv_launch_gpusort_bitonic_sort(clgss, nrows,
														 prt_size, true);
		if (!sort_kernel)
			goto error_sync;
		clgss->sort_kernel[k++] = sort_kernel;

		for (i=max_sort_size+1; i <= sort_size; i++)
		{
			for (j=i; max_sort_size<j; j--)
			{
				bool reversing = j == i ? true : false;
				sort_kernel = clserv_launch_gpusort_bitonic_step(clgss, nrows,
																 reversing, j,
																 false);
				if (!sort_kernel)
					goto error_sync;
				clgss->sort_kernel[k++] = sort_kernel;
			}
			sort_kernel = clserv_launch_gpusort_bitonic_marge(clgss, nrows,
															  max_sort_size,
															  false);
			if (!sort_kernel)
				goto error_sync;
			Assert(k <= sort_nums);
			clgss->sort_kernel[k++] = sort_kernel;
		}
	}

	/*
	 * Write back the gpusort chunk being prepared
	 */
	length = (KERN_GPUSORT_PARAMBUF_LENGTH(&gs_chunk->kern) +
			  KERN_GPUSORT_CHUNK_LENGTH(&gs_chunk->kern) +
			  KERN_GPUSORT_STATUS_LENGTH(&gs_chunk->kern) +
			  KERN_GPUSORT_TOASTBUF_LENGTH(&gs_chunk->kern));
	rc = clEnqueueReadBuffer(kcmdq,
							 clgss->m_chunk,
							 CL_FALSE,
							 0,
							 length,
							 &gs_chunk->kern,
							 1,
							 &clgss->events[clgss->ev_index - 1],
							 &clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueReadBuffer: %s",
				   opencl_strerror(rc));
		goto error_sync;
	}
	clgss->ev_index++;
    clgss->bytes_dma_recv += length;

	/* update performance counter */
	if (gpusort->msg.pfm.enabled)
	{
		gpusort->msg.pfm.bytes_dma_send = clgss->bytes_dma_send;
		gpusort->msg.pfm.bytes_dma_recv = clgss->bytes_dma_recv;
		gpusort->msg.pfm.num_dma_send = 2 + prep_nums;
		gpusort->msg.pfm.num_dma_recv = 1;
	}

	/* registers a callback routine that replies the message */
	rc = clSetEventCallback(clgss->events[clgss->ev_index - 1],
							CL_COMPLETE,
							clserv_respond_gpusort_single,
							clgss);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetEventCallback: %s",
				   opencl_strerror(rc));
		goto error_sync;
	}
	return;

error_sync:
	/*
	 * Once some requests were enqueued, we need to synchronize its
	 * completion prior to release resources, to avoid unexpected result.
	 */
	if (clgss->ev_index > 0)
	{
		clWaitForEvents(clgss->ev_index, clgss->events);
		while (--clgss->ev_index >= 0)
			clReleaseEvent(clgss->events[clgss->ev_index]);
	}

	/* NOTE: clgss->m_rcstore[i] shall be released by callback */
	for (i=0; i < prep_nums; i++)
	{
		if (clgss->prep_kernel[i])
			clReleaseKernel(clgss->prep_kernel[i]);
	}
	for (i=0; i < sort_nums; i++)
	{
		if (clgss->sort_kernel[i])
			clReleaseKernel(clgss->sort_kernel[i]);
	}
	if (clgss->m_chunk)
		clReleaseMemObject(clgss->m_chunk);
	if (clgss->program)
		clReleaseProgram(clgss->program);

error_release:
	if (clgss)
	{
		if (clgss->sort_kernel)
			free(clgss->sort_kernel);
		if (clgss->prep_kernel)
			free(clgss->prep_kernel);
		free(clgss);
	}
error:
	gpusort->msg.errcode = rc;
    pgstrom_reply_message(&gpusort->msg);
}

static void
clserv_process_gpusort(pgstrom_message *msg)
{
	pgstrom_gpusort	*gpusort = (pgstrom_gpusort *) msg;

	clserv_log("message with %d chunks", dlist_length(&gpusort->gs_chunks));
	Assert(dlist_length(&gpusort->gs_chunks) == 1);
	clserv_process_gpusort_single(gpusort);
}







/*
 *
 * DEADCODE of SORT BASED GPUPREAGG
 *
 */
#if 0
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
									   kernel,
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

/*
 * bitonic_compute_workgroup_size
 *
 * Bitonic-sorting logic will compare and exchange items being located
 * on distance of 2^N unit size. Its kernel entrypoint depends on the
 * unit size that depends on maximum available local workgroup size
 * for the kernel code. Because it depends on kernel resource consumption,
 * we need to ask runtime the max available local workgroup size prior
 * to the kernel enqueue.
 */
static cl_int
bitonic_compute_workgroup_size(clstate_gpupreagg *clgpa,
							   cl_uint nhalf,
							   size_t *p_gwork_sz,
							   size_t *p_lwork_sz)
{
	static struct {
		const char *kern_name;
		size_t		kern_lmem;
	} kern_calls[] = {
		{ "gpupreagg_bitonic_local", 2 * sizeof(cl_uint) },
		{ "gpupreagg_bitonic_merge",     sizeof(cl_uint) },
	};
	size_t		least_sz = nhalf;
	size_t		lwork_sz;
	size_t		gwork_sz;
	cl_kernel	kernel;
	cl_int		i, rc;

	for (i=0; i < lengthof(kern_calls); i++)
	{
		kernel = clCreateKernel(clgpa->program,
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
										  clgpa->dindex,
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
	 * NOTE: Local workgroup size is the largest 2^N value less than or
	 * equal to the least one of expected kernels.
	 */
	lwork_sz = 1UL << (get_next_log2(least_sz + 1) - 1);
	gwork_sz = ((nhalf + lwork_sz - 1) / lwork_sz) * lwork_sz;

	*p_lwork_sz = lwork_sz;
	*p_gwork_sz = gwork_sz;

	return CL_SUCCESS;
}
#endif

