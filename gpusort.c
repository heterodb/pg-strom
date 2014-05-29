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
	bool			scan_done;
	bool			sort_done;
	/* running status */
	cl_int			curr_index;
	cl_int			num_running;	/* number of async running requests */
	pgstrom_gpusort	*pending_gpusort;	/* a gpusort waiting for merge */

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
static shmem_startup_hook_type	shmem_startup_hook_next;
static struct
{
	slock_t		slab_lock;
	dlist_head	gpusort_freelist;
} *gpusort_shm_values;

/* static declarations */
static void clserv_process_gpusort(pgstrom_message *msg);

/*
 * Misc functions
 */
static inline void *
pmemcpy(void *from, size_t sz)
{
	void   *dest = palloc(sz);

	return memcpy(dest, from, sz);
}

/* additional dlist stuff */
static int
dlist_length(dlist_head *head)
{
	dlist_iter	iter;
	int			count = 0;

	dlist_foreach(iter, head)
		count++;
	return count;
}

static inline void
dlist_move_tail(dlist_head *head, dlist_node *node)
{
	/* fast path if it's already at the head */
	if (head->head.next == node)
		return;
	dlist_delete(node);
    dlist_push_tail(head, node);

    dlist_check(head);
}

static inline void
dlist_move_all(dlist_head *dest, dlist_head *src)
{
	Assert(dlist_is_empty(dest));

	dest->head.next = dlist_head_node(src);
	dest->head.prev = dlist_tail_node(src);
	dlist_head_node(src)->prev = &dest->head;
	dlist_tail_node(src)->next = &dest->head;

	dlist_init(src);
}

/*
 * get_next_log2
 *
 * It returns N of the least 2^N value that is larger than or equal to
 * the supplied value.
 */
static inline int
get_next_log2(Size size)
{
	int		shift = 0;

	if (size == 0 || size == 1)
		return 0;
	size--;
#ifdef __GNUC__
	shift = sizeof(Size) * BITS_PER_BYTE - __builtin_clzl(size);
#else
#if SIZEOF_VOID_P == 8
	if ((size & 0xffffffff00000000UL) != 0)
	{
		size >>= 32;
		shift += 32;
	}
#endif
	if ((size & 0xffff0000UL) != 0)
	{
		size >>= 16;
		shift += 16;
	}
	if ((size & 0x0000ff00UL) != 0)
	{
		size >>= 8;
		shift += 8;
	}
	if ((size & 0x000000f0UL) != 0)
	{
		size >>= 4;
		shift += 4;
	}
	if ((size & 0x0000000cUL) != 0)
	{
		size >>= 2;
		shift += 2;
	}
	if ((size & 0x00000002UL) != 0)
	{
		size >>= 1;
		shift += 1;
	}
	if ((size & 0x00000001UL) != 0)
		shift += 1;
#endif	/* !__GNUC__ */
	return shift;
}

/*
 * get_gpusort_chunksize
 *
 * A chunk (including kern_parambuf, kern_column_store and kern_toastbuf)
 * has to be smaller than the least "max memory allocation size" of the
 * installed device. It gives a preferred size of gpusort chunk according
 * to the device properties and GUC configuration.
 */
static inline Size
get_gpusort_chunksize(Size width, double nrows)
{
	Size		device_restriction;
	Size		chunk_sz;

	/*
	 * zone length is initialized to device's max memory allocation size
	 * on the starting up time, so a quarter of them is a safe expectation.
	 */
	device_restriction = (pgstrom_shmem_zone_length() / 4);
	device_restriction &= ~(SHMEM_BLOCKSZ - 1);

	/*
	 * Try to estimate the chunk size being required in this context.
	 */
	chunk_sz = SHMEM_BLOCKSZ;	/* all the metadata portion */
	chunk_sz += (width + sizeof(cl_uint) + sizeof(cl_ulong)) * (1.10 * nrows);

	while (chunk_sz > device_restriction)
		chunk_sz >>= 1;

	return (1UL << get_next_log2(chunk_sz)) - SHMEM_ALLOC_COST;
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
			return sizeof(cl_uint) + MAXALIGN(type_len);
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
	 * KPARAM_1 is also an array of AttrNumber (cl_ushort) to show
	 * referenced columns in the underlying column store.
	 * Both of them are just placeholder here. Actual values shall be
	 * be set up later in executor stage.
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



static void
pgstrom_release_gpusort_chunk(pgstrom_gpusort_chunk *gs_chunk)
{
	int		i;

	/*
	 * Unference the row- and column- stores being associated with
	 */
	for (i=0; i < gs_chunk->rcs_nums; i++)
	{
		StromObject	   *sobject = gs_chunk->rcs_slot[i];

		if (StromTagIs(sobject, TCacheRowStore))
			tcache_put_row_store((tcache_row_store *) sobject);
		else if (StromTagIs(sobject, TCacheColumnStore))
			tcache_put_column_store((tcache_column_store *) sobject);
		else
			elog(ERROR, "unexpected strom object is in rcs_slot: %d",
				 (int)sobject->stag);
	}

	/*
	 * OK, let's free it. Note that row- and column-store are
	 * inidividually tracked by resource tracker, we don't need
	 * to care about them here.
	 */
	pgstrom_shmem_free(gs_chunk->rcs_slot);
	pgstrom_shmem_free(gs_chunk);
}

static pgstrom_gpusort_chunk *
pgstrom_create_gpusort_chunk(GpuSortState *gsortstate)
{
	pgstrom_gpusort_chunk *gs_chunk;
	PlanState	   *subps = outerPlanState(gsortstate);
	TupleDesc		tupdesc = ExecGetResultType(subps);
	Size			allocsz_chunk;
	Size			allocsz_slot;
	kern_parambuf  *kparam;
	kern_column_store *kcs;
	cl_int		   *kstatus;
	kern_toastbuf  *ktoast;
	ListCell	   *cell;
	Size			offset;
	cl_int			i_col = 0;

	/* allocation of a shared memory block */
	gs_chunk = pgstrom_shmem_alloc_alap(gsortstate->gpusort_chunksz,
										&allocsz_chunk);
	if (!gs_chunk)
		elog(ERROR, "out of shared memory");
	memset(gs_chunk, 0, sizeof(pgstrom_gpusort_chunk));

	/* also, allocate slot for row-/column-store */
	gs_chunk->rcs_slot = pgstrom_shmem_alloc_alap(0, &allocsz_slot);
	if (!gs_chunk->rcs_slot)
	{
		pgstrom_shmem_free(gs_chunk);
		elog(ERROR, "out of shared memory");
	}
	gs_chunk->rcs_slotsz = (allocsz_slot / sizeof(StromObject *));
	gs_chunk->rcs_nums = 0;
	gs_chunk->rcs_global_index = -1;	/* to be set later */

	/* next, initialization of kern_gpusort */
	kparam = KERN_GPUSORT_PARAMBUF(&gs_chunk->kern);
	memcpy(kparam, gsortstate->kparambuf, gsortstate->kparambuf->length);
	Assert(kparam->length == STROMALIGN(kparam->length));

	/* next, initialization of kern_column_store */
	kcs = KERN_GPUSORT_CHUNK(&gs_chunk->kern);
	kcs->ncols = list_length(gsortstate->sortkey_resnums) + 2;
	kcs->nrows = 0;
	kcs->nrooms = gsortstate->nrows_per_chunk;
	offset = STROMALIGN(offsetof(kern_column_store,
								 colmeta[kcs->ncols]));
	/* regular sortkeys */
	foreach(cell, gsortstate->sortkey_resnums)
	{
		Form_pg_attribute	attr;
		int		resno = lfirst_int(cell);

		Assert(resno > 0 && resno <= tupdesc->natts);
		attr = tupdesc->attrs[resno - 1];

		kcs->colmeta[i_col].attnotnull = attr->attnotnull;
		if (attr->attalign == 'c')
			kcs->colmeta[i_col].attalign = sizeof(cl_char);
		else if (attr->attalign == 's')
			kcs->colmeta[i_col].attalign = sizeof(cl_short);
		else if (attr->attalign == 'i')
			kcs->colmeta[i_col].attalign = sizeof(cl_int);
		else if (attr->attalign == 'd')
			kcs->colmeta[i_col].attalign = sizeof(cl_long);
		else
			elog(ERROR, "unexpected attalign '%c'", attr->attalign);
		kcs->colmeta[i_col].attlen = attr->attlen;
		kcs->colmeta[i_col].cs_ofs = offset;
		if (!attr->attnotnull)
			offset += STROMALIGN(kcs->nrooms / BITS_PER_BYTE);
		if (attr->attlen > 0)
			offset += STROMALIGN(attr->attlen * kcs->nrooms);
		else
		{
			offset += STROMALIGN(sizeof(cl_uint) * kcs->nrooms);
			Assert(list_member_int(gsortstate->sortkey_toast, resno));
		}
		i_col++;
	}

	/*
	 * The second last column is reserved by GpuSort - fixed-length integer
	 * as identifier of unsorted tuples, not null.
	 */
	kcs->colmeta[i_col].attnotnull = true;
	kcs->colmeta[i_col].attalign = sizeof(cl_long);
	kcs->colmeta[i_col].attlen = sizeof(cl_long);
	kcs->colmeta[i_col].cs_ofs = offset;
	offset += STROMALIGN(sizeof(cl_long) * kcs->nrooms);
	i_col++;

	/*
	 * Last column is reserved by GpuSort - fixed-length integer as
	 * index of sorted tuples, not null.
	 * Note that this field has to be aligned to 2^N length, to
	 * simplify kernel implementation.
	 */
	kcs->colmeta[i_col].attnotnull = true;
	kcs->colmeta[i_col].attalign = sizeof(cl_int);
	kcs->colmeta[i_col].attlen = sizeof(cl_int);
	kcs->colmeta[i_col].cs_ofs = offset;
	offset += STROMALIGN(sizeof(cl_int) * (1UL << get_next_log2(kcs->nrooms)));
    i_col++;
	Assert(i_col == kcs->ncols);

	kcs->length = offset;

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
 * pgstrom_sanitycheck_gpusort_chunk
 *
 * It checks whether the gpusort-chunk being replied from OpenCL server
 * is sanity, or not.
 */
static void
pgstrom_sanitycheck_gpusort_chunk(GpuSortState *gsortstate,
								  pgstrom_gpusort_chunk *gs_chunk,
								  bool is_sorted)
{
#if 0
	kern_column_store  *kcs = KERN_GPUSORT_CHUNK(&gs_chunk->kern);
	kern_toastbuf	   *ktoast = KERN_GPUSORT_TOASTBUF(&gs_chunk->kern);
	cl_int			   *kstatus = KERN_GPUSORT_STATUS(&gs_chunk->kern);
	cl_uint				ncols;
	cl_uint				nrows;
	cl_int				i, j;

	if (*kstatus != StromError_Success)
		elog(INFO, "chunk: status %d (%s)",
			 *kstatus, pgstrom_strerror(*kstatus));

	ncols = list_length(gsortstate->sortkey_resnums) + 2;
	if (ncols != kcs->ncols)
		elog(ERROR, "chunk corrupted: expected ncols=%u, but kcs->ncols=%u",
			 ncols, kcs->ncols);

	nrows = 0;
	for (i=0; i < gs_chunk->rcs_nums; i++)
	{
		StromObject	   *sobject = gs_chunk->rcs_slot[i];

		if (StromTagIs(sobject, TCacheRowStore))
		{
			tcache_row_store *trs = (tcache_row_store *)sobject;
			nrows += trs->kern.nrows;
		}
		else if (StromTagIs(sobject, TCacheColumnStore))
		{
			tcache_column_store *tcs = (tcache_column_store *)sobject;
			nrows += tcs->nrows;
			elog(INFO, "chunk: rcs_slot[%d] is column store", i);
		}
		else
			elog(ERROR, "chunk: rcs_slot[%d] corrupted (stag: %d)",
				 i, sobject->stag);
	}
	if (nrows != kcs->nrows)
		elog(ERROR, "chunk corrupted: expected nrows=%u, but kcs->nrow=%u",
			 nrows, kcs->nrows);

	for (i=0; i < nrows; i++)
	{
		StromObject	*sobject;
		void	   *temp;
		cl_ulong	growid;
		cl_uint		rcs_idx;
		cl_uint		row_idx;

		if (!is_sorted)
		{
			cl_uint		kcs_index;

			temp = kern_get_datum(kcs, ncols - 1, i);
			if (!temp)
				elog(ERROR, "chunk corrupted: kcs_index is null");
			kcs_index = *((cl_uint *)temp);
			if (kcs_index != i)
				elog(ERROR, "chunk corrupted: kcs_index should be %u, but %u",
					 i, kcs_index);
		}

		temp = kern_get_datum(kcs, ncols - 2, i);
		if (!temp)
			elog(ERROR, "chunk corrupted: growid is null");
		growid = *((cl_ulong *)temp);
		rcs_idx = growid >> 32;
		row_idx = growid & 0xffffffff;

		if (rcs_idx >= gsortstate->rcs_nums)
			elog(ERROR, "chunk corrupted: rcs_index (=%u) out of range (<%u)",
				 rcs_idx, gsortstate->rcs_nums);

		sobject = gs_chunk->rcs_slot[rcs_idx];
		if (StromTagIs(sobject, TCacheRowStore))
		{
			tcache_row_store *trs = (tcache_row_store *)sobject;
			rs_tuple   *rs_tup = kern_rowstore_get_tuple(&trs->kern, row_idx);
			HeapTuple	tup = &rs_tup->htup;
			TupleDesc	tupdesc = gsortstate->scan_slot->tts_tupleDescriptor;
			ListCell   *cell;
			int			i_col = 0;

			foreach (cell, gsortstate->sortkey_resnums)
			{
				AttrNumber	anum = lfirst_int(cell);
				Form_pg_attribute attr = tupdesc->attrs[anum - 1];
				Datum	rs_value;
				bool	rs_isnull;
				void   *cs_value;

				rs_value = heap_getattr(tup, anum, tupdesc, &rs_isnull);

				cs_value = kern_get_datum(kcs, i_col, i);

				if ((!rs_isnull && !cs_value) || (rs_isnull && cs_value))
					elog(ERROR, "[%d] null status corrupted (%s => %s)",
						 nrows,
						 !rs_isnull ? "false" : "true",
						 !cs_value ? "true" : "false");
				if (rs_isnull)
					continue;
				if (attr->attlen > 0 && attr->attbyval)
				{
					if (memcmp(&rs_value, cs_value, attr->attlen) != 0)
						elog(ERROR, "[%d] value corrupted (%08lx=>%08lx)",
							 i, rs_value, *((Datum *)cs_value));
				}
				else if (attr->attlen > 0 && !attr->attbyval)
				{
					if (memcmp(&rs_value, (void *)cs_value, attr->attlen) != 0)
						elog(ERROR, "[%d] value corrupted", i);
				}
				else
				{
					cl_uint	vl_ofs = *((cl_uint *)cs_value);
					void   *vl_cs = (char *)ktoast + vl_ofs;
					void   *vl_rs = DatumGetPointer(rs_value);
					size_t	vl_len;

					if (VARSIZE_ANY(vl_rs) != VARSIZE_ANY(vl_cs))
						elog(INFO, "[%d] toast length corrupted (%zu=>%zu)",
							 i, VARSIZE_ANY(vl_rs), VARSIZE_ANY(vl_cs));
					vl_len = VARSIZE_ANY(vl_rs);

					if (memcmp(vl_rs, vl_cs, vl_len) != 0)
					{
						StringInfoData	buf;

						initStringInfo(&buf);
						appendStringInfo(&buf, "'");
						for (j=0; j < vl_len; j++)
						{
							int	c = ((char *)vl_rs)[j];

							if (isprint(c))
								appendStringInfo(&buf, "%c", c);
							else
								appendStringInfo(&buf, "\\%02x", c);
						}
						appendStringInfo(&buf, "' => '");
						for (j=0; j < vl_len; j++)
						{
							int	c = ((char *)vl_cs)[j];
							if (isprint(c))
								appendStringInfo(&buf, "%c", c);
							else
								appendStringInfo(&buf, "\\%02x", c);
						}
						appendStringInfo(&buf, "'");
						elog(ERROR, "[%d] toast value corrupted (%s)",
							 i, buf.data);
					}
				}
			}
		}
		else if (StromTagIs(sobject, TCacheColumnStore))
		{
			elog(ERROR, "column-store as sorting source, not implemented");
		}
		else
			elog(ERROR, "unexpected strom object");
	}
	if (kcs->nrows != nrows)
		elog(INFO, "chunk: nrows = %u (expected: %u)", kcs->nrows, nrows);
#endif
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
 * gpstrom_run_cpusort_single
 *
 * It runs CPU based quick-sorting on a particular chunk.
 * After that, its result index shall be sorted according to key-comparison
 * results.
 */
static void
recursive_run_cpusort_single(GpuSortState *gsortstate,
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
		recursive_run_cpusort_single(gsortstate, kcs, ktoast,
									 rindex, left, i-1);
		recursive_run_cpusort_single(gsortstate, kcs, ktoast,
									 rindex, j+1, right);
	}
}

static void
gpstrom_run_cpusort_single(GpuSortState *gsortstate, pgstrom_gpusort *gpusort)
{
	pgstrom_gpusort_chunk *gs_chunk;
	kern_column_store  *kcs;
	kern_toastbuf	   *ktoast;
	cl_int			   *rindex;

	if (gpusort->is_sorted)
	{
		elog(INFO, "chunk is already sorted, so no need to run %s",
			 __FUNCTION__);
		return;
	}
	gs_chunk = dlist_container(pgstrom_gpusort_chunk, chain,
							   dlist_head_node(&gpusort->gs_chunks));
	kcs = KERN_GPUSORT_CHUNK(&gs_chunk->kern);
	ktoast = KERN_GPUSORT_TOASTBUF(&gs_chunk->kern);
	rindex = KERN_GPUSORT_RESULT_INDEX(kcs);

	recursive_run_cpusort_single(gsortstate, kcs, ktoast, rindex,
								 0, kcs->nrows - 1);
	gpusort->is_sorted = true;
}

/*
 * pgstrom_run_cpusort_multi
 *
 *
 *
 *
 *
 */





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

	/*
	 * pgstrom_gpusort_multi is a slab object. So, we don't use
	 * pgstrom_shmem_alloc/free interface for each object.
	 */
	SpinLockAcquire(&gpusort_shm_values->slab_lock);
	memset(gpusort, 0, sizeof(pgstrom_gpusort));
	dlist_push_tail(&gpusort_shm_values->gpusort_freelist, &gpusort->chain);
    SpinLockRelease(&gpusort_shm_values->slab_lock);
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
	dlist_node		   *dnode;

	SpinLockAcquire(&gpusort_shm_values->slab_lock);
	if (dlist_is_empty(&gpusort_shm_values->gpusort_freelist))
	{
		Size		allocated;
		uintptr_t	tailaddr;

		gpusort = pgstrom_shmem_alloc_alap(sizeof(pgstrom_gpusort),
										   &allocated);
		tailaddr = (uintptr_t)gpusort + allocated;
		while (((uintptr_t)gpusort + sizeof(pgstrom_gpusort)) <= tailaddr)
		{
			dlist_push_tail(&gpusort_shm_values->gpusort_freelist,
							&gpusort->chain);
			gpusort++;
		}
	}
	Assert(!dlist_is_empty(&gpusort_shm_values->gpusort_freelist));
	dnode = dlist_pop_head_node(&gpusort_shm_values->gpusort_freelist);
	gpusort = dlist_container(pgstrom_gpusort, chain, dnode);
	memset(&gpusort->chain, 0, sizeof(dlist_node));
	SpinLockRelease(&gpusort_shm_values->slab_lock);

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
	gpusort->is_sorted = false;
	gpusort->by_cpusort = false;
	dlist_init(&gpusort->gs_chunks);

	return gpusort;
}

/*
 * gpusort_preload_chunk
 *
 * It loads row-store (or column-store if available) from the underlying
 * relation scan, and enqueue a gpusort-chunk with them into OpenCL
 * server process. If a chunk can be successfully enqueued, it returns
 * reference of this chunk. Elsewhere, NULL shall be returns; that implies
 * all the records in the underlying relation were already read.
 */
static pgstrom_gpusort *
gpusort_preload_chunk(GpuSortState *gsortstate, HeapTuple *overflow)
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
	Size				toast_length;
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
	gs_chunk->rcs_global_index = gsortstate->rcs_nums;
	kcs = KERN_GPUSORT_CHUNK(&gs_chunk->kern);
	ktoast = KERN_GPUSORT_TOASTBUF(&gs_chunk->kern);
	toast_length = ktoast->length;
	toast_usage = ktoast->usage;

	dlist_push_head(&gpusort->gs_chunks, &gs_chunk->chain);

	/*
	 * TODO: If SubPlan support ExecMulti for data exchange,
	 * we use this special service.
	 */

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
			if (toast_usage + toastsz > toast_length)
			{
				*overflow = tuple;
				Assert(nrows > 0);
				break;
			}
		}

		/* OK, let's put it on the row-store */
		if (!trs || !tcache_row_store_insert_tuple(trs, tuple))
		{
			/*
			 * allocation of a new row-store, but not tracked because trs
			 * is always kept by a particular gpusort-chunk.
			 */
			trs = tcache_create_row_store(tupdesc);

			/* put it on the r/c-store array on the GpuSortState */
			if (gsortstate->rcs_nums == gsortstate->rcs_slotsz)
			{
				gsortstate->rcs_slot = repalloc(gsortstate->rcs_slot,
												sizeof(StromObject *) *
												2 * gsortstate->rcs_slotsz);
				gsortstate->rcs_slotsz += gsortstate->rcs_slotsz;
			}
			Assert(gsortstate->rcs_nums < gsortstate->rcs_slotsz);
			gsortstate->rcs_slot[gsortstate->rcs_nums++] = &trs->sobj;

			/* also, put it on the r/c-store array on the gpusort */
			if (gs_chunk->rcs_nums == gs_chunk->rcs_slotsz)
			{
				StromObject	  **new_slot;

				new_slot = pgstrom_shmem_realloc(gs_chunk->rcs_slot,
												 sizeof(StromObject *) *
												 2 * gs_chunk->rcs_slotsz);
				if (!new_slot)
					elog(ERROR, "out of shared memory");
				gs_chunk->rcs_slot = new_slot;
				gs_chunk->rcs_slotsz += gs_chunk->rcs_slotsz;
			}
			Assert(gs_chunk->rcs_nums < gs_chunk->rcs_slotsz);
			gs_chunk->rcs_slot[gs_chunk->rcs_nums++] = &trs->sobj;

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
 * gpusort_process_response
 *
 * It processes the supplied gpusort message. If we have a pending gpusort,
 * it tries to merge two sequential chunks into one. Elsewhere, it should
 * be pending until next sequential chunks will come.
 */
static void
gpusort_process_response(GpuSortState *gsortstate, pgstrom_gpusort *gpusort)
{
	/* gs_chunks of response message should not be an empty list */
	Assert(!dlist_is_empty(&gpusort->gs_chunks));
#if 0
	{
		dlist_iter		iter;

		dlist_foreach(iter, &gpusort->gs_chunks)
		{
			pgstrom_gpusort_chunk  *gs_chunk
				= dlist_container(pgstrom_gpusort_chunk, chain, iter.cur);
			kern_toastbuf  *ktoast = KERN_GPUSORT_TOASTBUF(&gs_chunk->kern);
			elog(INFO, "ktoast usage %u of %u", ktoast->usage, ktoast->length);
		}
	}
#endif

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
			if (dlist_length(&gpusort->gs_chunks) == 1)
				gpstrom_run_cpusort_single(gsortstate, gpusort);
			else
				elog(ERROR, "not implemented yet (multi-chunk cpusort)");
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

	if (gsortstate->pending_gpusort)
	{
		pgstrom_gpusort	   *pending = gsortstate->pending_gpusort;
		dlist_head		   *x_chunks = &pending->gs_chunks;
		dlist_head		   *y_chunks = &gpusort->gs_chunks;
		dlist_head			new_list;
		dlist_mutable_iter	miter;

		/*
		 * Preparation of merge-sorting; OpenCL kernel expects a series
		 * of preliminary sorted chunks are provided according to the
		 * order of head item of each chunk.
		 */
		dlist_init(&new_list);
		while (!dlist_is_empty(x_chunks) && !dlist_is_empty(y_chunks))
		{
			pgstrom_gpusort_chunk  *gs_xchunk;
			pgstrom_gpusort_chunk  *gs_ychunk;
			kern_column_store	   *kcs_x;
			kern_column_store	   *kcs_y;
			kern_toastbuf		   *ktoast_x;
			kern_toastbuf		   *ktoast_y;
			cl_int				   *rindex_x;
			cl_int				   *rindex_y;

			/*
			 * XXX - Probably, sanity checks are needs to avoid backend
			 * crash...
			 */
			gs_xchunk = dlist_container(pgstrom_gpusort_chunk, chain,
										dlist_head_node(x_chunks));
			kcs_x = KERN_GPUSORT_CHUNK(&gs_xchunk->kern);
			ktoast_x = KERN_GPUSORT_TOASTBUF(&gs_xchunk->kern);
			rindex_x = KERN_GPUSORT_RESULT_INDEX(kcs_x);

			gs_ychunk = dlist_container(pgstrom_gpusort_chunk, chain,
										dlist_head_node(y_chunks));
			kcs_y = KERN_GPUSORT_CHUNK(&gs_ychunk->kern);
			ktoast_y = KERN_GPUSORT_TOASTBUF(&gs_ychunk->kern);
			rindex_y = KERN_GPUSORT_RESULT_INDEX(kcs_y);

			if (pgstrom_compare_cpusort(gsortstate,
										kcs_x, ktoast_x, rindex_x[0],
										kcs_y, ktoast_y, rindex_y[0]) < 0)
				dlist_move_tail(&new_list, &gs_xchunk->chain);
			else
				dlist_move_tail(&new_list, &gs_ychunk->chain);
		}
		/* either of loop below shall run */
		dlist_foreach_modify(miter, x_chunks)
			dlist_move_tail(&new_list, miter.cur);
		dlist_foreach_modify(miter, y_chunks)
			dlist_move_tail(&new_list, miter.cur);

		/*
		 * OK, all the chunks were pre-merged, let's link it again
		 * on the pending side
		 */
		gsortstate->pending_gpusort = NULL;
		dlist_move_all(&pending->gs_chunks, &new_list);

		/* infection of cpusort necessity */
		if (gpusort->by_cpusort)
			pending->by_cpusort = true;
		/* TODO: must be checked if CPU recheck is needed or not */

		/* release either of gpusorting request */
		pgstrom_untrack_object(&gpusort->msg.sobj);
		pgstrom_put_message(&gpusort->msg);

		/* OK, this two input stream shall be merged into one */
		if (!pgstrom_enqueue_message(&pending->msg))
			elog(ERROR, "Bug? OpenCL server seems to dead");
		gsortstate->num_running++;
	}
	else
	{
		/*
		 * No pending gpusort right now, so this request needs to wait
		 * for the next response message. It shall be merged with the
		 * next processed chunk.
		 */
		gsortstate->pending_gpusort = gpusort;
	}
}






static CustomPlanState *
gpusort_begin(CustomPlan *node, EState *estate, int eflags)
{
	GpuSortPlan	   *gsortplan = (GpuSortPlan *) node;
	GpuSortState   *gsortstate;
	TupleDesc		tupdesc;
	int				extra_flags;
	List		   *sortkey_resnums = NIL;
	List		   *sortkey_toast = NIL;
	Bitmapset	   *tempset;
	bytea		   *rs_attrefs;
	AttrNumber		anum;
	AttrNumber		anum_last;
	Const		   *kparam_0;
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
	extra_flags = gsortplan->extra_flags;
	if (pgstrom_kernel_debug)
		extra_flags |= DEVKERNEL_NEEDS_DEBUG;
	gsortstate->dprog_key = pgstrom_get_devprog_key(gsortplan->kern_source,
													extra_flags);
	pgstrom_track_object((StromObject *)gsortstate->dprog_key, 0);

	/* Also, message queue */
	gsortstate->mqueue = pgstrom_create_queue();
	pgstrom_track_object(&gsortstate->mqueue->sobj, 0);

	/* Construct param/const buffer including system constants */
	Assert(list_length(gsortplan->used_params) > 1);
	used_params = copyObject(gsortplan->used_params);

	rs_attrefs = palloc0(VARHDRSZ + sizeof(cl_char) * tupdesc->natts);
	SET_VARSIZE(rs_attrefs, VARHDRSZ + sizeof(cl_char) * tupdesc->natts);

	tempset = bms_copy(gsortplan->sortkey_resnums);
	anum_last = -1;
	while ((anum = bms_first_member(tempset)) >= 0)
	{
		anum += FirstLowInvalidHeapAttributeNumber;
		Assert(anum > 0 && anum <= tupdesc->natts);

		((cl_char *)VARDATA(rs_attrefs))[anum - 1] = 1;
		sortkey_resnums = lappend_int(sortkey_resnums, anum);
		if (tupdesc->attrs[anum - 1]->attlen < 0)
			sortkey_toast = lappend_int(sortkey_toast, anum);
		anum_last = anum - 1;
	}
	/* negative value is end of referenced columns marker */
	if (anum_last >= 0)
		((cl_char *)VARDATA(rs_attrefs))[anum_last] = -1;

	kparam_0 = (Const *)linitial(used_params);
	Assert(IsA(kparam_0, Const) &&
		   kparam_0->consttype == BYTEAOID &&
		   kparam_0->constisnull);
	kparam_0->constvalue = PointerGetDatum(rs_attrefs);
	kparam_0->constisnull = false;

	/*
	 * TODO: kparam_1 is needed to implement column-store exchange between
	 * plan nodes in case when several conditions are fine.
	 * - SubPlan is GpuScan
	 * - SubPlan has no projection
	 * - All the Vars required in the target-list of GpuSort are cached.
	 * 
	 * data exchange based on column-store is not supported right now,
	 * so kparam_1 == NULL is delivered to the kernel.
	 */
	gsortstate->kparambuf = pgstrom_create_kern_parambuf(used_params, NULL);
	gsortstate->sortkey_resnums = sortkey_resnums;
	gsortstate->sortkey_toast = sortkey_toast;
	gsortstate->sortkey_width = gsortplan->sortkey_width;
	gsortstate->gpusort_chunksz
		= get_gpusort_chunksize(gsortplan->sortkey_width,
								gsortplan->cplan.plan.plan_rows);
	gsortstate->nrows_per_chunk =
		(gsortstate->gpusort_chunksz -
		 STROMALIGN(gsortstate->kparambuf->length) -
		 STROMALIGN(offsetof(kern_column_store,
							 colmeta[gsortplan->numCols + 2])) -
		 STROMALIGN(sizeof(cl_int)) -
		 STROMALIGN(offsetof(kern_toastbuf, coldir[0])))
		/ (sizeof(cl_ulong) + gsortstate->sortkey_width + 2 * sizeof(cl_uint));
	gsortstate->nrows_per_chunk &= ~32UL;

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
	dlist_node		   *dnode;
	TupleTableSlot	   *slot;

	if (!gsortstate->sort_done)
	{
		HeapTuple	overflow = NULL;

		while (!gsortstate->scan_done)
		{
			/* preload a chunk onto pgstrom_gpusort */
			gpusort = gpusort_preload_chunk(gsortstate, &overflow);
			if (!gpusort)
			{
				Assert(gsortstate->scan_done);
				Assert(!overflow);
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
				Assert(!overflow);
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
	gpusort = gsortstate->pending_gpusort;
	slot = gsortstate->cps.ps.ps_ResultTupleSlot;
	ExecClearTuple(slot);

retry:
	if (!gpusort || dlist_is_empty(&gpusort->gs_chunks))
		return slot;

	dnode = dlist_head_node(&gpusort->gs_chunks);
	gs_chunk = dlist_container(pgstrom_gpusort_chunk, chain, dnode);
	kcs = KERN_GPUSORT_CHUNK(&gs_chunk->kern);

	/* for debugging */
	pgstrom_sanitycheck_gpusort_chunk(gsortstate, gs_chunk,
									  gpusort->is_sorted);

	if (gsortstate->curr_index < kcs->nrows)
	{
		cl_uint		   *kcs_index;
		cl_ulong	   *growid;
		cl_uint			rcs_idx;
		cl_uint			row_idx;
		cl_int			i, j;
		StromObject	   *sobject;

		i = gsortstate->curr_index;
		growid = (cl_ulong *)((char *)kcs +
							  kcs->colmeta[kcs->ncols - 2].cs_ofs);
		kcs_index = (cl_uint *)((char *)kcs +
								kcs->colmeta[kcs->ncols - 1].cs_ofs);
		Assert(kcs_index[i] >= 0 && kcs_index[i] < kcs->nrows);
		j = kcs_index[i];

		rcs_idx = ((growid[j] >> 32) & 0xffffffff);
		row_idx = (growid[j] & 0xffffffff);
		Assert(rcs_idx < gsortstate->rcs_nums);

		sobject = gsortstate->rcs_slot[rcs_idx];
		if (StromTagIs(sobject, TCacheRowStore))
		{
			tcache_row_store *trs = (tcache_row_store *)sobject;
			rs_tuple   *rs_tup;

			Assert(row_idx < trs->kern.nrows);
			rs_tup = kern_rowstore_get_tuple(&trs->kern, row_idx);

			ExecStoreTuple(&rs_tup->htup, slot, InvalidBuffer, false);
		}
		else if (StromTagIs(sobject, TCacheColumnStore))
		{
			/* to be implemented later */
			elog(ERROR, "to be implemented later");
		}
		else
			elog(ERROR, "unexpected strom object in rcs_slot");

		gsortstate->curr_index++;
		return slot;
	}
	/* moves to next chunk */
	dlist_delete(dnode);
	pgstrom_release_gpusort_chunk(gs_chunk);
	gsortstate->curr_index = 0;
	goto retry;
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
	pgstrom_gpusort	   *gpusort;

	/*
	 * Release PG-Strom shared objects
	 */
	pgstrom_untrack_object((StromObject *)gsortstate->dprog_key);
	pgstrom_put_devprog_key(gsortstate->dprog_key);

	pgstrom_untrack_object(&gsortstate->mqueue->sobj);
	pgstrom_close_queue(gsortstate->mqueue);

	Assert(gsortstate->num_running == 0);
	gpusort = gsortstate->pending_gpusort;
	if (gpusort)
	{
		pgstrom_untrack_object(&gpusort->msg.sobj);
		pgstrom_put_message(&gpusort->msg);
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
	pgstrom_gpusort	   *gpusort;

	/* If we haven't sorted yet, just return. */
	if (!gsortstate->sort_done)
		return;
	/* no asynchronous job should not be there */
	Assert(gsortstate->num_running == 0);

	/* must drop pointer to sort result tuple */
	ExecClearTuple(gsortstate->cps.ps.ps_ResultTupleSlot);

	/* right now, we just re-scan again */
	gpusort = gsortstate->pending_gpusort;
	if (gpusort)
	{
		pgstrom_untrack_object(&gpusort->msg.sobj);
		pgstrom_put_message(&gpusort->msg);
	}
	gsortstate->pending_gpusort = NULL;

	gsortstate->scan_done = false;
	gsortstate->sort_done = false;

	gsortstate->curr_index = 0;

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
	newnode->sortkey_resnums = copyObject(oldnode->sortkey_resnums);
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
static void
pgstrom_startup_gpusort(void)
{
	bool		found;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	gpusort_shm_values
		= ShmemInitStruct("gpusort_shm_values",
						  MAXALIGN(sizeof(*gpusort_shm_values)),
						  &found);
	Assert(!found);

	memset(gpusort_shm_values, 0, sizeof(*gpusort_shm_values));
	SpinLockInit(&gpusort_shm_values->slab_lock);
	dlist_init(&gpusort_shm_values->gpusort_freelist);
}

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

	/*
	 * pgstrom_gpusort_multi is small data structure.
	 * So, we apply slab on them
	 */
	RequestAddinShmemSpace(MAXALIGN(sizeof(*gpusort_shm_values)));
	shmem_startup_hook_next = shmem_startup_hook;
    shmem_startup_hook = pgstrom_startup_gpusort;
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
	cl_mem	kern_mem = private;
	cl_int	rc;

	rc = clReleaseMemObject(kern_mem);
	if (rc != CL_SUCCESS)
		clserv_log("failed on clReleaseMemObject: %s", opencl_strerror(rc));
}

/*
 * opencl kernel invocation of:
 *
 * __kernel void
 * gpusort_setup_chunk_rs(cl_uint rcs_global_index,
 *                        __global kern_gpusort *kgsort,
 *                        __global kern_row_store *krs,
 *                        __local void *local_workmem)
 */
static cl_kernel
clserv_launch_gpusort_setup_row(clstate_gpusort_single *clgss,
								tcache_row_store *trs,
								cl_uint rcs_global_index)
{
	cl_command_queue kcmdq = opencl_cmdq[clgss->dindex];
	cl_kernel		prep_kernel;
	cl_mem			m_rstore;
	size_t			gwork_sz;
	size_t			lwork_sz;
	cl_int			rc;

	m_rstore = clCreateBuffer(opencl_context,
							  CL_MEM_READ_WRITE,
							  trs->kern.length,
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
									   trs->kern.nrows,
									   sizeof(cl_uint)))
		goto error_2;

	rc = clSetKernelArg(prep_kernel,
						0,	/* cl_uint rcs_global_index */
						sizeof(cl_uint),
						&rcs_global_index);
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
						3,	/* local_workmem */
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
	 * Enqueue a kernel execution on this row-store.
	 */
	rc = clEnqueueNDRangeKernel(kcmdq,
								prep_kernel,
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
 * gpusort_setup_chunk_cs(cl_uint rcs_global_index,
 *                        __global kern_gpusort *kgsort,
 *                        __global kern_row_store *krs,
 *                        __local void *local_workmem)
 */
static cl_kernel
clserv_launch_gpusort_setup_column(clstate_gpusort_single *clgss,
								   tcache_column_store *trs,
								   cl_uint rcs_global_index)
{
	/* to be implemented later */
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
clserv_launch_gpusort_bitonic(clstate_gpusort_single *clgss,
							  cl_uint nrows, bool reversing, cl_uint unitsz,
							  bool is_first)
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
								 "gpusort_single",
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
						sizeof(cl_uint) * lwork_sz + sizeof(cl_uint),
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
	kern_parambuf		   *kparam;
	kern_column_store	   *kcs;
	clstate_gpusort_single *clgss;
	cl_command_queue		kcmdq;
	cl_uint					dindex;
	cl_uint					nrows;
	cl_uint					prep_nums;
	cl_uint					sort_nums;
	cl_uint					sort_size;
	cl_uint					event_nums;
	cl_int					i, j, k, rc;
	size_t					length;
	size_t					offset;

	/* only pgstom_gpusort (sorting chunk) should be attached */
	gs_chunk = dlist_container(pgstrom_gpusort_chunk, chain,
							   dlist_head_node(&gpusort->gs_chunks));
	kparam = KERN_GPUSORT_PARAMBUF(&gs_chunk->kern);
	kcs = KERN_GPUSORT_CHUNK(&gs_chunk->kern);
	prep_nums = gs_chunk->rcs_nums;

	nrows = 0;
	for (i=0; i < prep_nums; i++)
	{
		StromObject *sobject = gs_chunk->rcs_slot[i];
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
	sort_size = get_next_log2(nrows);
	sort_nums = (sort_size * (sort_size + 1)) / 2;

	/* state object of gpuscan (single chunk) */
	event_nums = 2 * prep_nums + sort_nums + 4;
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
							  kparam,
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
		StromObject *sobject = gs_chunk->rcs_slot[i];
		cl_uint		rcs_global_index = gs_chunk->rcs_global_index + i;
		cl_kernel	prep_kernel;

		if (StromTagIs(sobject, TCacheRowStore))
		{
			tcache_row_store   *trs = (tcache_row_store *) sobject;
			prep_kernel = clserv_launch_gpusort_setup_row(clgss, trs,
														  rcs_global_index);
			if (!prep_kernel)
				goto error_sync;
		}
		else if (StromTagIs(sobject, TCacheColumnStore))
		{
			tcache_column_store *tcs = (tcache_column_store *) sobject;
			prep_kernel = clserv_launch_gpusort_setup_column(clgss, tcs,
															 rcs_global_index);
			if (!prep_kernel)
				goto error_sync;
		}
		else
		{
			rc = StromError_BadRequestMessage;
			goto error_sync;
		}
	}

	/*
	 * OK, preparation was done. Let's launch gpusort_single kernel
	 * to sort key values within a gpusort-chunk.
	 */
	for (i=1, k=0; i <= sort_size; i++)
	{
		for (j=i; j > 0; j--)
		{
			cl_kernel	sort_kernel
				= clserv_launch_gpusort_bitonic(clgss, nrows,
												j == i ? true : false, j,
												k==0 ? true : false);
			if (!sort_kernel)
				goto error_sync;
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

/*
 * multi-chunks merge sorting
 *
 *
 *
 *
 *
 */
typedef struct
{
	pgstrom_message	*msg;
	cl_program		program;
	cl_kernel	   *kernels;
	cl_uint			i_kernel;
	cl_mem			buffers[6];
	Size			buffer_length;
	cl_int			dindex;
	Size			bytes_dma_send;
	Size			bytes_dma_recv;
	cl_int			ev_index;
	cl_event		events[20];
} clstate_gpusort_multi;

static void
clserv_respond_gpusort_multi(cl_event event, cl_int ev_status, void *private)
{
	clstate_gpusort_multi  *clgsm = private;
	pgstrom_gpusort		   *gpusort = (pgstrom_gpusort *)clgsm->msg;
	pgstrom_gpusort_chunk  *gs_chunk;
	cl_int		i;

	/* put an error code */
	if (ev_status != CL_COMPLETE)
	{
		clserv_log("unexpected CL_EVENT_COMMAND_EXECUTION_STATUS: %d",
				   ev_status);
		gpusort->msg.errcode = StromError_OpenCLInternal;
	}
	else
	{
		dlist_iter	iter;
		cl_int		errcode;

		dlist_foreach(iter, &gpusort->gs_chunks)
		{
			gs_chunk = dlist_container(pgstrom_gpusort_chunk, chain, iter.cur);

			errcode = *KERN_GPUSORT_STATUS(&gs_chunk->kern);
			if (errcode != StromError_Success &&
				StromErrorIsSignificant(errcode))
			{
				gpusort->msg.errcode = *KERN_GPUSORT_STATUS(&gs_chunk->kern);
				break;
			}
		}
		if (gpusort->msg.errcode == StromError_Success)
			gpusort->is_sorted = true;
	}

	/* collect performance statistics */
	if (gpusort->msg.pfm.enabled)
	{
		/* to be implemented here */

	}

	/* release opencl objects */
	while (clgsm->ev_index > 0)
		clReleaseEvent(clgsm->events[--clgsm->ev_index]);
	while (clgsm->i_kernel > 0)
		clReleaseKernel(clgsm->kernels[--clgsm->i_kernel]);
	for (i=0; i < lengthof(clgsm->buffers); i++)
		clReleaseMemObject(clgsm->buffers[i]);

	free(clgsm->kernels);
	free(clgsm);

	/* respond to the backend side */
	pgstrom_reply_message(&gpusort->msg);
}

/*
 * Launcher of gpusort_multi kernel function
 *
 * __kernel void
 * gpusort_multi(cl_int bitonic_unitsz,
 *               __global kern_gpusort *kgsort_x,
 *               __global kern_gpusort *kgsort_y,
 *               __global kern_gpusort *kgsort_z1,
 *               __global kern_gpusort *kgsort_z2,
 *               __local void *local_workbuf)
 */
static int
clserv_launch_gpusort_multi(clstate_gpusort_multi *clgsm,
							cl_uint i_buffer,
							cl_uint nrows,
							bool reversing,
							cl_uint unitsz,
							cl_uint kern_num_blockers,
							cl_event *kern_blockers)
{
	cl_command_queue kcmdq = opencl_cmdq[clgsm->dindex];
	cl_kernel	merge_kernel;
	size_t		lwork_sz;
	size_t		gwork_sz;
	cl_uint		n_threads;
	cl_int		unitlen;
	cl_int		bitonic_unitsz;
	cl_int		i, rc;

	/* create a kernel object */
	merge_kernel = clCreateKernel(clgsm->program,
								  "gpusort_multi",
								  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateKernel: %s", opencl_strerror(rc));
		return rc;
	}
	clgsm->kernels[clgsm->i_kernel++] = merge_kernel;

	/* Set optimal global/local workgroup size */
	unitlen = (1 << unitsz);
	n_threads = ((nrows + unitlen - 1) & ~(unitlen - 1)) / 2;
	if (!clserv_compute_workgroup_size(&gwork_sz, &lwork_sz,
									   merge_kernel,
									   clgsm->dindex,
									   n_threads,
									   sizeof(cl_uint)))
		return StromError_OpenCLInternal;

	/*
	 * 1st arg: cl_int bitonic_unitsz
	 */
	if (reversing)
		bitonic_unitsz = -unitsz;
	else
		bitonic_unitsz = unitsz;

	rc = clSetKernelArg(merge_kernel,
						0,
						sizeof(cl_int),
						&bitonic_unitsz);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	/*
	 * 2nd arg: __global kern_gpusort *kgsort_x
	 */
	i = i_buffer % lengthof(clgsm->buffers);
	rc = clSetKernelArg(merge_kernel,
						1,
						sizeof(cl_mem),
						&clgsm->buffers[i]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	/*
	 * 3rd arg: __global kern_gpusort *kgsort_y
	 */
	i = (i_buffer + 1) % lengthof(clgsm->buffers);
    rc = clSetKernelArg(merge_kernel,
						2,
						sizeof(cl_mem),
						&clgsm->buffers[i]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	/*
	 * 4th arg: __global kern_gpusort *kgsort_z1
	 */
	i = (i_buffer + 2) % lengthof(clgsm->buffers);
    rc = clSetKernelArg(merge_kernel,
						3,
						sizeof(cl_mem),
						&clgsm->buffers[i]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	/*
	 * 5th arg: __global kern_gpusort *kgsort_z1
	 */
	i = (i_buffer + 3) % lengthof(clgsm->buffers);
    rc = clSetKernelArg(merge_kernel,
						4,
						sizeof(cl_mem),
						&clgsm->buffers[i]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	/*
	 * 6th arg: __local void *local_workbuf
	 */
	rc = clSetKernelArg(merge_kernel,
						5,
						sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		return rc;
	}

	/*
	 * Enqueue kernel execution
	 */
	rc = clEnqueueNDRangeKernel(kcmdq,
								merge_kernel,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								kern_num_blockers,
								kern_blockers,
								&clgsm->events[clgsm->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
                   opencl_strerror(rc));
		return rc;
	}
	clgsm->ev_index++;
	return CL_SUCCESS;
}

static void
clserv_process_gpusort_multi(pgstrom_gpusort *gpusort)
{
	clstate_gpusort_multi  *clgsm = NULL;
	pgstrom_gpusort_chunk  *gs_chunk;
	pgstrom_gpusort_chunk  *prev_chunk;
	pgstrom_gpusort_chunk  *curr_chunk;
	kern_column_store	   *kcs;
	cl_command_queue		kcmdq;
	cl_event				ev_last_dma_send = NULL;
	cl_event				ev_last_kernel = NULL;
	cl_event				ev_prev_dma_recv = NULL;
	cl_event				ev_last_dma_recv = NULL;
	dlist_iter				iter;
	cl_uint					num_events;
	cl_uint					num_kernels;
	cl_uint					n_chunks;
	cl_uint					nrows;
	cl_uint					i_buffer;
	cl_uint					merge_size;
	size_t					buffer_length;
	cl_int					rc, i, j, k;

	/*
	 * allocation of clstate_gpusort_multi, according to number of
	 * kernels to be run
	 */
	n_chunks = dlist_length(&gpusort->gs_chunks);
	if (n_chunks < 2)
	{
		rc = StromError_BadRequestMessage;
		goto error_nowait;
	}

	/*
	 * Find out the length of kernel buffer object (to be same length for
	 * all the associated chunks) and the largest number of rows per chunk.
	 */
	nrows = 0;
	buffer_length = 0;
	dlist_foreach(iter, &gpusort->gs_chunks)
	{
		size_t	length;

		gs_chunk = dlist_container(pgstrom_gpusort_chunk, chain, iter.cur);
		kcs = KERN_GPUSORT_CHUNK(&gs_chunk->kern);

		length = KERN_GPUSORT_TOTAL_LENGTH(&gs_chunk->kern);
		Assert(buffer_length == 0 || buffer_length == length);
		buffer_length = length;

		nrows = Max(nrows, kcs->nrows);
	}
	merge_size = get_next_log2(nrows);
	num_kernels = (merge_size * (merge_size + 1)) / 2;

	/*
	 * state object of gpuscan (multi chunks)
	 */
	rc = CL_OUT_OF_HOST_MEMORY;
	num_events = 2 * n_chunks + num_kernels + 2;
	clgsm = calloc(1, offsetof(clstate_gpusort_multi,
							   events[num_events]));
	if (!clgsm)
		goto error_nowait;

	clgsm->kernels = calloc(num_kernels, sizeof(cl_kernel));
	if (!clgsm->kernels)
		goto error_nowait;

	clgsm->msg = &gpusort->msg;

	/*
	 * Choose a device to execute this kernel
	 */
	clgsm->dindex = pgstrom_opencl_device_schedule(&gpusort->msg);
	kcmdq = opencl_cmdq[clgsm->dindex];

	/*
	 * First of all, it looks up a program object to be run on the
	 * supplied gpusort-chunk. We may have three cases, however,
	 * it usually takes the last case because this program object
	 * should be already built.
	 * (1) NULL; program object is under asynchronous building.
	 * (2) BAD_OPENCL_PROGRAM; failed on program build.
	 * (3) valid cl_program object.
	 */
	clgsm->program = clserv_lookup_device_program(gpusort->dprog_key,
												  &gpusort->msg);
	if (!clgsm->program)
	{
		free(clgsm->kernels);
		free(clgsm);
		return;		/* message is in waitq, to be retried later! */
	}
	if (clgsm->program == BAD_OPENCL_PROGRAM)
	{
		rc = CL_BUILD_PROGRAM_FAILURE;
		goto error_nowait;
	}

	/*
	 * allocation of buffer object
	 *
	 * Multi-chunk merge sorting takes four chunks at once; two for input
	 * and another two for output. In addition to the four chunks, we
	 * allocate two more chunks for asynchronous DMA send/recv under the
	 * sort execution; one for DMA send to load next chunk, and another
	 * for DMA receive to write back sorted result.
	 */
	for (i=0; i < lengthof(clgsm->buffers); i++)
	{
		clgsm->buffers[i] = clCreateBuffer(opencl_context,
										   CL_MEM_READ_WRITE,
										   buffer_length,
										   NULL,
										   &rc);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
			goto error_nowait;
		}
	}

	/*
	 * Enqueuing the jobs
	 *
	 * GpuSort uses six chunks one after the other. When we take multi-
	 * chunks merge sorting, it tries to merge two chunks for each step.
	 * A step uses two chunks for read-buffer, two other chunks for
	 * write-buffer (one for smaller half, another for larger half) and
	 * another two are used for concurrent DMA transfer (one to load
	 * the data for next step, one to write back previous result).
	 *
	 * Let's assume here are fix boxes; named A, B, ..., F.
	 * Initially, data in the first and second chunks are loaded into
	 * A and B, then gpusort_multi() kernel function writes merge-
	 * sorting result into C and D. In concurrence with kernel execution,
	 * data of the third chunk shall be loaded to E.
	 * On the next step, the kernel deals with D and E as input buffers,
	 * F and A as output buffers. Second kernel execution is blocked
	 * until first kernel gets run out, DMA send to E gets done. Then,
	 * F (primary write buffer) shall be written back.
	 *
	 * In general, a kernel execution in Nth step has blockers below:
	 * - a kernel execution in (N-1)th step; that generates primary
	 *   read buffer.
	 * - a DMA send to the secondary read buffer
	 * - a DMA recv from the primary write buffer; being filled by
	 *   kernel execution in (n-2)th step.
	 * In case of N==1, load of the first chunk performs as like a kernel
	 * execution in 0th step, and load of the second chunk performs as
	 * DMA send to the secondary read buffer. Also, a DMA recv from the
	 * primary write buffer (to give over the write buffer to be used on
	 * the next) does not block kernel execution in case of N <= 2.
	 *
	 * Regarding of DMA, a DMA send to the secondary read buffer for
	 * kernel execution in Nth step has identical blocker set with kernel
	 * execution in (N-1)th step. Also, a DMA recv from the primary write
	 * buffer in Nth kernel execution is blocked by Nth kernel execution
	 * as literal.
	 */

	/*
	 * Preparation towards the main loop
	 *
	 * Enqueue the first and second chunks. It sets up the primary and
	 * secondary read buffer of 1st kernel execution. Its role is like
	 * as 0th kernel execution (to generate primary read buffer) and
	 * DMA send (to fill secondary read buffer), so its events objects
	 * are set on the variable according to its role.
	 */
	prev_chunk = dlist_container(pgstrom_gpusort_chunk, chain,
								 dlist_head_node(&gpusort->gs_chunks));
	rc = clEnqueueWriteBuffer(kcmdq,
							  clgsm->buffers[0],
							  CL_FALSE,
							  0,
							  buffer_length,
							  &prev_chunk->kern,
							  0,
							  NULL,
							  &clgsm->events[clgsm->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}
	ev_last_kernel = clgsm->events[clgsm->ev_index];
	clgsm->ev_index++;

	/*
	 * Also, this DMA fills the secondary read buffer
	 */
	curr_chunk = dlist_container(pgstrom_gpusort_chunk, chain,
								 dlist_next_node(&gpusort->gs_chunks,
												 &prev_chunk->chain));
	rc = clEnqueueWriteBuffer(kcmdq,
							  clgsm->buffers[1],
							  CL_FALSE,
							  0,
							  buffer_length,
							  &curr_chunk->kern,
							  0,
							  NULL,
							  &clgsm->events[clgsm->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}
	ev_last_dma_send = clgsm->events[clgsm->ev_index];
	clgsm->ev_index++;

	for (i_buffer=0; ;i_buffer+=3)
	{
		kern_column_store  *kcs_prev = KERN_GPUSORT_CHUNK(&prev_chunk->kern);
		kern_column_store  *kcs_curr = KERN_GPUSORT_CHUNK(&curr_chunk->kern);
		cl_uint				nrows = Max(kcs_prev->nrows, kcs_curr->nrows);
		cl_uint				num_blockers;
		cl_event			blockers[3];

		/*
		 * Kernel execution
		 *
		 * As mentioned above, a kernel execution shall be launched after
		 * the preparation of primary & secondary read buffer (generated
		 * by previous kernel execution and DMA send), and DMA send back
		 * of the primary write buffer to be used (if any previous kernel
		 * exists).
		 */
		nrows = Max(kcs_prev->nrows, kcs_curr->nrows);

		Assert(ev_last_kernel && ev_last_dma_send);
		num_blockers = 2;
		blockers[0] = ev_last_kernel;
		blockers[1] = ev_last_dma_send;
		if (ev_prev_dma_recv)
			blockers[num_blockers++] = ev_prev_dma_recv;

		for (i=1, k=0; i <= merge_size; i++)
		{
			for (j=i; j > 0; j--, k++)
			{
				rc = clserv_launch_gpusort_multi(
						clgsm,
						i_buffer,
						nrows,
						j == i ? true : false,
						j,
						k == 0 ? num_blockers : 1,
						k == 0 ? blockers : &clgsm->events[clgsm->ev_index - 1]
					);
				if (rc != CL_SUCCESS)
					goto error_sync;
			}
		}
		ev_last_kernel = clgsm->events[clgsm->ev_index - 1];

		/*
		 * Enqueue DMA receive request
		 *
		 * After the kernel execution, it moves contents of the primary
		 * write buffer into the previous chunk. Previous chunks shall
		 * be chained to the result list.
		 */
		i = (i_buffer + 2) % lengthof(clgsm->buffers);
		rc = clEnqueueReadBuffer(kcmdq,
								 clgsm->buffers[i],
								 CL_FALSE,
								 0,
								 buffer_length,
								 &prev_chunk->kern,
								 1,
								 &ev_last_kernel,
								 &clgsm->events[clgsm->ev_index]);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clEnqueueWriteBuffer: %s",
					   opencl_strerror(rc));
			goto error_sync;
		}
		ev_prev_dma_recv = ev_last_dma_recv;
		ev_last_dma_recv = clgsm->events[clgsm->ev_index];
		clgsm->ev_index++;

		/* any more chunks to be processed? */
		if (!dlist_has_next(&gpusort->gs_chunks,
							&curr_chunk->chain))
			break;

		prev_chunk = curr_chunk;
		curr_chunk = dlist_container(pgstrom_gpusort_chunk, chain,
									 dlist_next_node(&gpusort->gs_chunks,
													 &curr_chunk->chain));
		/*
		 * Concurrent DMA send request
		 *
		 * It tries to transfer the data to be merged into device DRAM
		 * during the current kernel execution.
		 */
		i = (i_buffer + 4) % lengthof(clgsm->buffers);
		rc = clEnqueueWriteBuffer(kcmdq,
								  clgsm->buffers[i],
								  CL_FALSE,
								  0,
								  buffer_length,
								  &curr_chunk->kern,
								  num_blockers,
								  blockers,
								  &clgsm->events[clgsm->ev_index]);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clEnqueueWriteBuffer: %s",
					   opencl_strerror(rc));
			goto error_sync;
		}
		ev_last_dma_send = clgsm->events[clgsm->ev_index];
		clgsm->ev_index++;
	}

	/*
	 * Enqueue DMA receive request
	 *
	 * Once all the chunks get processed, it also moves the contents
	 * of the secondary write buffer into the last chunk.
	 */
	i = (i_buffer + 3) % lengthof(clgsm->buffers);
	rc = clEnqueueReadBuffer(kcmdq,
							 clgsm->buffers[i],
							 CL_FALSE,
							 0,
							 buffer_length,
							 &curr_chunk->kern,
							 1,
							 &ev_last_kernel,
							 &clgsm->events[clgsm->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueReadBuffer: %s",
				   opencl_strerror(rc));
		goto error_sync;
	}
	clgsm->ev_index++;

	/*
	 * Callback function to return the series of merge-sorting result.
	 */
	rc = clSetEventCallback(clgsm->events[clgsm->ev_index - 1],
							CL_COMPLETE,
							clserv_respond_gpusort_multi,
							clgsm);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetEventCallback: %s", opencl_strerror(rc));
		goto error_sync;
	}
	return;

error_sync:
	/*
	 * Once some requests were enqueued, we need to synchronize its
	 * completion prior to release resources, to avoid unexpected result.
	 */
	if (clgsm->ev_index > 0)
	{
		clWaitForEvents(clgsm->ev_index, clgsm->events);
		while (clgsm->ev_index > 0)
			clReleaseEvent(clgsm->events[--clgsm->ev_index]);
		while (clgsm->i_kernel > 0)
			clReleaseKernel(clgsm->kernels[--clgsm->i_kernel]);
	}

error_nowait:
	if (clgsm)
	{
		for (i=0; i < lengthof(clgsm->buffers); i++)
		{
			if (clgsm->buffers[i])
				clReleaseMemObject(clgsm->buffers[i]);
		}
		if (clgsm->program)
			clReleaseProgram(clgsm->program);
		if (clgsm->kernels)
			free(clgsm->kernels);
	}
	gpusort->msg.errcode = rc;
	pgstrom_reply_message(&gpusort->msg);
}

static void
clserv_process_gpusort(pgstrom_message *msg)
{
	pgstrom_gpusort	*gpusort = (pgstrom_gpusort *) msg;

	clserv_log("message with %d chunks", dlist_length(&gpusort->gs_chunks));

	Assert(!dlist_is_empty(&gpusort->gs_chunks));
	if (dlist_length(&gpusort->gs_chunks) == 1)
		clserv_process_gpusort_single(gpusort);
	else
	    clserv_process_gpusort_multi(gpusort);
}
