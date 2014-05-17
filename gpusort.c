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
	pgstrom_gpusort_multi *pending_mgsort;	/* a chunk waiting for merge */

	/* row-/column- stores being read */
	cl_uint			rcs_nums;	/* number of row-/column-stores in use */
	cl_uint			rcs_slotsz;	/* size of row-/column-store slot */
	StromObject	  **rcs_slot;	/* slot of row-/column-stores */

	pgstrom_perfmon	pfm;		/* sum of performance counter */
} GpuSortState;

/* static variables */
static int						pgstrom_gpusort_chunksize;
static CustomPlanMethods		gpusort_plan_methods;
static shmem_startup_hook_type	shmem_startup_hook_next;
static struct
{
	slock_t		slab_lock;
	dlist_head	gsort_multi_freelist;
} *gpusort_shm_values;

/* static declarations */
static void clserv_process_gpusort_multi(pgstrom_message *msg);
static void clserv_process_gpusort(pgstrom_message *msg);

/*
 * Misc functions
 */
static inline void *
pmemcpy(void *from, size_t sz)
{
	void   *dest = palloc(sz);
	memcpy(from, dest, sz);
	return dest;
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

static void
dlist_move_all(dlist_head *dest, dlist_head *src)
{
	Assert(dlist_is_empty(dest));

	dest->head.next = src->head.next;
	src->head.next->prev = &dest->head;
	dest->head.prev = src->head.prev;
	src->head.prev->next = &dest->head;

	dlist_init(src);
}

/*
 * get_next_pow2
 *
 * It returns the least 2^N value larger than or equal to
 * the supplied value.
 */
static inline Size
get_next_pow2(Size size)
{
	int		shift = 0;

	if (size == 0)
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
	return (1UL << shift);
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
get_gpusort_chunksize(void)
{
	static Size	device_restriction = 0;
	Size		chunk_sz;

	if (!device_restriction)
	{
		int		i, n = pgstrom_get_device_nums();
		Size	curr_min = LONG_MAX;

		for (i=0; i < n; i++)
		{
			const pgstrom_device_info *dinfo = pgstrom_get_device_info(i);

			if (curr_min > dinfo->dev_max_mem_alloc_size)
				curr_min = dinfo->dev_max_mem_alloc_size;
		}
		curr_min &= ~(SHMEM_BLOCKSZ - 1);

		device_restriction = curr_min;
	}

	if (pgstrom_gpusort_chunksize == 0)
		chunk_sz = device_restriction / 4;
	else
		chunk_sz = Min(((Size)pgstrom_gpusort_chunksize) << 20,
					   device_restriction);
	return get_next_pow2(chunk_sz) - SHMEM_ALLOC_COST;
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
	{
		elog(INFO, "type_len = %d", type_len);
		return type_len;
	}

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
		{
			elog(INFO, "type_len(s) = %d", type_len);
			return sizeof(cl_uint) + MAXALIGN(type_len);
		}
	}
	/*
	 * Uh... we have no idea how to estimate average length of
	 * key variable if target-entry is not Var nor underlying
	 * plan is not a usual relation scan.
	 */
	type_mod = exprTypmod((Node *)tle->expr);

	type_len = get_typavgwidth(type_oid, type_mod);

	elog(INFO, "type_len(t) = %d", type_len);
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
	int				i;

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
		TargetEntry *tle = get_tle_by_resno(sort->plan.targetlist,
											sort->sortColIdx[i]);
		bool	nullfirst = sort->nullsFirst[i];
		Oid		sort_op = sort->sortOperators[i];
		Oid		sort_func;
		Oid		sort_type;
		bool	is_reverse;
		Var	   *tvar;
		char   *temp;
		devtype_info   *dtype;
		devfunc_info   *dfunc;

		Assert(IsA(tle->expr, Var));
		tvar = (Var *)tle->expr;
		Assert(tvar->varno == OUTER_VAR);

		/* type for comparison */
		sort_type = exprType((Node *)tle->expr);
		dtype = pgstrom_devtype_lookup(sort_type);
		Assert(dtype != NULL);
		context->type_defs = list_append_unique_ptr(context->type_defs, dtype);

		/* function for comparison */
		sort_func = get_compare_function(sort_op, &is_reverse);
		dfunc = pgstrom_devfunc_lookup(sort_func);
		Assert(dfunc != NULL);

		appendStringInfo(&decl,
						 "  pg_%s_t keyval_x%u;\n"
						 "  pg_%s_t keyval_y%u;\n",
						 dtype->type_name, i+1,
						 dtype->type_name, i+1);
		appendStringInfo(&decl,
						 "  pg_int4_t comp;\n");

		context->row_index = "x_index";
		temp = pgstrom_codegen_expression((Node *)tle->expr, context);
		appendStringInfo(&body, "  keyval_x%u = %s;\n", i+1, temp);
		pfree(temp);

		context->row_index = "y_index";
		temp = pgstrom_codegen_expression((Node *)tle->expr, context);
		appendStringInfo(&body, "  keyval_y%u = %s;\n", i+1, temp);
		pfree(temp);

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
		"gpusort_comp(__private int *errcode,\n"
		"             __global kern_column_store *kcs_x,\n"
		"             __global kern_toastbuf *ktoast_x,\n"
		"             __private cl_int x_index,\n"
		"             __global kern_column_store *kcs_y,\n"
		"             __global kern_toastbuf *ktoast_y,\n"
		"             __private cl_int y_index)\n"
		"{\n"
		"%s"	/* key variable declarations */
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
pgstrom_release_gpusort(pgstrom_message *msg)
{
	pgstrom_gpusort	   *gs_chunk = (pgstrom_gpusort *)msg;

	/*
	 * Unlink message queue and device program. These should exist
	 * unless this object is not linked to pgstrom_gpusort_multi.
	 */
	if (msg->respq)
		pgstrom_put_queue(msg->respq);
	if (gs_chunk->dprog_key != 0)
		pgstrom_put_devprog_key(gs_chunk->dprog_key);
	/*
	 * OK, let's free it. Note that row- and column-store are
	 * inidividually tracked by resource tracker, we don't need
	 * to care about them here.
	 */
	pgstrom_shmem_free(gs_chunk->rcs_slot);
	pgstrom_shmem_free(gs_chunk);
}

static pgstrom_gpusort *
pgstrom_create_gpusort(GpuSortState *gsortstate, bool is_executable)
{
	PlanState	   *subps = outerPlanState(gsortstate);
	TupleDesc		tupdesc = ExecGetResultType(subps);
	Size			allocsz_chunk;
	Size			allocsz_slot;
	pgstrom_gpusort *gs_chunk;
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
	memset(gs_chunk, 0, sizeof(pgstrom_gpusort));

	/* also, allocate slot for row-/column-store */
	gs_chunk->rcs_slot = pgstrom_shmem_alloc_alap(0, &allocsz_slot);
	if (!gs_chunk->rcs_slot)
	{
		pgstrom_shmem_free(gs_chunk);
		elog(ERROR, "out of shared memory");
	}
	gs_chunk->rcs_slotsz = (allocsz_slot / sizeof(StromObject *));
	gs_chunk->rcs_nums = 0;

	/* initialize fields in pgstrom_gpusort */
	gs_chunk->msg.sobj.stag = StromTag_GpuSort;
	SpinLockInit(&gs_chunk->msg.lock);
	gs_chunk->msg.refcnt = 1;
	gs_chunk->msg.errcode = StromError_Success;
	if (is_executable)
		gs_chunk->msg.respq = pgstrom_get_queue(gsortstate->mqueue);
	gs_chunk->msg.cb_process = clserv_process_gpusort;
	gs_chunk->msg.cb_release = pgstrom_release_gpusort;
	if (is_executable)
		gs_chunk->dprog_key
			= pgstrom_retain_devprog_key(gsortstate->dprog_key);

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
	/*
	 * First column is reserved by GpuSort - fixed-length integer as
	 * identifier of unsorted tuples, not null.
	 */
	kcs->colmeta[0].attnotnull = true;
	kcs->colmeta[0].attalign = sizeof(cl_long);
	kcs->colmeta[0].attlen = sizeof(cl_long);
	kcs->colmeta[0].cs_ofs = offset;
	offset += STROMALIGN(sizeof(cl_long) * kcs->nrooms);
	i_col++;

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
	 * Last column is reserved by GpuSort - fixed-length integer as
	 * index of sorted tuples, not null.
	 * Note that this field has to be aligned to 2^N length, to
	 * simplify kernel implementation.
	 */
	kcs->colmeta[i_col].attnotnull = true;
	kcs->colmeta[i_col].attalign = sizeof(cl_int);
	kcs->colmeta[i_col].attlen = sizeof(cl_int);
	kcs->colmeta[i_col].cs_ofs = offset;
	offset += STROMALIGN(sizeof(cl_int) * get_next_pow2(kcs->nrooms));
    i_col++;
	Assert(i_col == kcs->ncols);

	kcs->length = offset;

	/* next, initialization of kernel execution status field */
	kstatus = KERN_GPUSORT_STATUS(&gs_chunk->kern);
	*kstatus = StromError_Success;

	/* last, initialization of toast buffer in flat-mode */
	ktoast = KERN_GPUSORT_TOASTBUF(&gs_chunk->kern);
	if (gsortstate->sortkey_toast == NIL)
	{
		ktoast->length = 0;
		ktoast->usage = 0;
	}
	else
	{
		ktoast->length =
			STROMALIGN_DOWN(allocsz_chunk -
							((uintptr_t)ktoast -
							 (uintptr_t)(&gs_chunk->kern)));
		ktoast->usage = offsetof(kern_toastbuf, coldir[0]);
	}

	/* OK, initialized */
	return gs_chunk;
}

/*
 * gpusort_set_chunk_unexecutable
 *
 * Once a gpusort got preloaded and set up (by kernel), no need to be
 * executable any more, because pgstrom_gpusort is dealt with data
 * chunks of pgstrom_gpusort_multi. It releases response message-queue
 * and device program being acquired.
 */
static void
gpusort_set_chunk_unexecutable(pgstrom_gpusort *gs_chunk)
{
	pgstrom_untrack_object((StromObject *)&gs_chunk->msg.respq);
	pgstrom_put_queue(gs_chunk->msg.respq);
	gs_chunk->msg.respq = NULL;

	pgstrom_untrack_object((StromObject *)gs_chunk->dprog_key);
	pgstrom_put_devprog_key(gs_chunk->dprog_key);
	gs_chunk->dprog_key = 0;
}

/*
 * pgstrom_release_gpusort_multi
 *
 * Destructor of pgstrom_gpusort_multi object, to be called when refcnt
 * as message object reached to zero.
 *
 * NOTE: this routine may be called in the OpenCL server's context
 */
static void
pgstrom_release_gpusort_multi(pgstrom_message *msg)
{
	pgstrom_gpusort_multi *mgsort = (pgstrom_gpusort_multi *)msg;
	pgstrom_gpusort		  *bgsort;
	dlist_iter	iter;

	/* unlink message queue and device program (should always exists) */
	pgstrom_put_queue(msg->respq);
	pgstrom_put_devprog_key(mgsort->dprog_key);

	/*
	 * pgstrom_gpusort_multi usually has multiple pgstrom_gpusort
	 * objects in its input and output list, as literal.
	 * So, let's unlink them.
	 */
	dlist_foreach(iter, &mgsort->in_chunk1)
	{
		bgsort = dlist_container(pgstrom_gpusort, chain, iter.cur);
		pgstrom_put_message(&bgsort->msg);
	}

	dlist_foreach(iter, &mgsort->in_chunk2)
	{
		bgsort = dlist_container(pgstrom_gpusort, chain, iter.cur);
		pgstrom_put_message(&bgsort->msg);
	}

	dlist_foreach(iter, &mgsort->out_chunk)
	{
		bgsort = dlist_container(pgstrom_gpusort, chain, iter.cur);
		pgstrom_put_message(&bgsort->msg);
	}

	/*
	 * pgstrom_gpusort_multi is a slab object. So, we don't use
	 * pgstrom_shmem_alloc/free interface for each object.
	 */
	SpinLockAcquire(&gpusort_shm_values->slab_lock);
	memset(mgsort, 0, sizeof(pgstrom_gpusort_multi));
	dlist_push_tail(&gpusort_shm_values->gsort_multi_freelist,
					&mgsort->chain);
    SpinLockRelease(&gpusort_shm_values->slab_lock);	
}

/*
 * pgstrom_create_gpusort_multi
 *
 * constructor of pgstrom_gpusort_multi object.
 */
static pgstrom_gpusort_multi *
pgstrom_create_gpusort_multi(GpuSortState *gsortstate)
{
	pgstrom_gpusort_multi *mgsort;
	dlist_node	   *dnode;

	SpinLockAcquire(&gpusort_shm_values->slab_lock);
	if (dlist_is_empty(&gpusort_shm_values->gsort_multi_freelist))
	{
		Size		allocated;
		uintptr_t	tailaddr;

		mgsort = pgstrom_shmem_alloc_alap(sizeof(pgstrom_gpusort_multi),
										  &allocated);
		tailaddr = (uintptr_t)mgsort + allocated;
		while (((uintptr_t)mgsort +
				sizeof(pgstrom_gpusort_multi)) <= tailaddr)
		{
			dlist_push_tail(&gpusort_shm_values->gsort_multi_freelist,
							&mgsort->chain);
			mgsort++;
		}
	}
	Assert(!dlist_is_empty(&gpusort_shm_values->gsort_multi_freelist));
	dnode = dlist_pop_head_node(&gpusort_shm_values->gsort_multi_freelist);
	mgsort = dlist_container(pgstrom_gpusort_multi, chain, dnode);
	memset(&mgsort->chain, 0, sizeof(dlist_node));
	SpinLockRelease(&gpusort_shm_values->slab_lock);

	/* initialize the common message field */
	memset(mgsort, 0, sizeof(pgstrom_gpusort_multi));
	mgsort->msg.sobj.stag = StromTag_GpuSortMulti;
	SpinLockInit(&mgsort->msg.lock);
	mgsort->msg.refcnt = 1;
	mgsort->msg.respq = pgstrom_get_queue(gsortstate->mqueue);
	mgsort->msg.cb_process = clserv_process_gpusort_multi;
	mgsort->msg.cb_release = pgstrom_release_gpusort_multi;
	/* other fields also */
	mgsort->dprog_key = pgstrom_retain_devprog_key(gsortstate->dprog_key);
	dlist_init(&mgsort->in_chunk1);
	dlist_init(&mgsort->in_chunk2);
	dlist_init(&mgsort->out_chunk);
	dlist_init(&mgsort->work_chunk);

	return mgsort;
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
	pgstrom_gpusort	   *gs_chunk;
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

	/* do we have any more tuple to read? */
	if (gsortstate->scan_done)
		return NULL;

	/* create a gpusort chunk */
	gs_chunk = pgstrom_create_gpusort(gsortstate, true);
	pgstrom_track_object(&gs_chunk->msg.sobj, 0);

	kcs = KERN_GPUSORT_CHUNK(&gs_chunk->kern);
	ktoast = KERN_GPUSORT_TOASTBUF(&gs_chunk->kern);
	toast_length = ktoast->length;
	toast_usage = ktoast->usage;

	/*
	 * TODO: If SubPlan support ExecMulti for data exchange,
	 * we use this special service.
	 */


	/* subplan should take forward scan */
	estate->es_direction = ForwardScanDirection;

	/* copy tuples to tcache_row_store */
	trs = NULL;
	for (nrows=0; nrows < kcs->nrooms; nrows++)
	{
		slot = ExecProcNode(subnode);
		if (TupIsNull(slot))
		{
			gsortstate->scan_done = true;
			break;
		}
		tuple = ExecFetchSlotTuple(slot);

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
			/* allocate a new row-store, and track it */
			trs = tcache_create_row_store(tupdesc);
			pgstrom_track_object(&trs->sobj, 0);

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

	/* now tuples were read in actually, so nothing to do */
	if (nrows == 0)
	{
		pgstrom_untrack_object(&gs_chunk->msg.sobj);
		pgstrom_put_message(&gs_chunk->msg);
		return NULL;
	}
	return gs_chunk;
}

/*
 * gpusort_process_resp_single
 *
 * It processes the supplied gpusort chunk. If we already have pending
 * pgstrom_gpusort_multi object, it's a time to merge them into one.
 * Elsewhere, the replied object should be added as pending one.
 */
static void
gpusort_process_resp_single(GpuSortState *gsortstate,
							pgstrom_gpusort *gs_chunk)
{
	pgstrom_gpusort_multi *mgsort;

	if (gsortstate->pending_mgsort)
	{
		mgsort = gsortstate->pending_mgsort;
		gsortstate->pending_mgsort = NULL;

		Assert(!dlist_is_empty(&mgsort->in_chunk1) &&
			   dlist_is_empty(&mgsort->in_chunk2) &&
			   dlist_is_empty(&mgsort->out_chunk) &&
			   dlist_length(&mgsort->work_chunk) == 1);
		/*
		 * no need to have response queue and device program
		 * any more. so, release them now.
		 */
		gpusort_set_chunk_unexecutable(gs_chunk);
		pgstrom_untrack_object(&gs_chunk->msg.sobj);

		/* TODO: must be checked if CPU recheck is needed or not */

		/*
		 * OK, here is two stream of pre-sorted chunks. Let's merge
		 * them into a series of chunks.
		 */
		dlist_push_head(&mgsort->in_chunk2, &gs_chunk->chain);
		if (!pgstrom_enqueue_message(&mgsort->msg))
			elog(ERROR, "Bug? OpenCL server seems to dead");
		gsortstate->num_running++;
	}
	else
	{
		pgstrom_gpusort	*work_chunk;

		mgsort = pgstrom_create_gpusort_multi(gsortstate);
		pgstrom_track_object(&mgsort->msg.sobj, 0);

		/*
		 * no need to have response queue and device program
		 * any more. so, release them now.
		 */
		gpusort_set_chunk_unexecutable(gs_chunk);
		pgstrom_untrack_object(&gs_chunk->msg.sobj);

		dlist_push_head(&mgsort->in_chunk1, &gs_chunk->chain);

		/* also, a working chunk is needed */
		work_chunk = pgstrom_create_gpusort(gsortstate, false);
		dlist_push_head(&mgsort->work_chunk, &work_chunk->chain);

		/* OK, let's wait for another chunk's stream */
		gsortstate->pending_mgsort = mgsort;
	}
}

static void
gpusort_process_resp_multi(GpuSortState *gsortstate,
						   pgstrom_gpusort_multi *mgsort)
{
	if (gsortstate->pending_mgsort)
    {
		pgstrom_gpusort_multi *pending = gsortstate->pending_mgsort;

		Assert(!dlist_is_empty(&pending->in_chunk1) &&
			   dlist_is_empty(&pending->in_chunk2) &&
			   dlist_is_empty(&pending->out_chunk) &&
			   dlist_is_empty(&pending->work_chunk));
		Assert(dlist_is_empty(&mgsort->in_chunk1) &&
			   dlist_is_empty(&mgsort->in_chunk2) &&
			   !dlist_is_empty(&mgsort->out_chunk) &&
			   dlist_length(&mgsort->work_chunk) == 1);

		gsortstate->pending_mgsort = NULL;

		dlist_move_all(&pending->in_chunk2, &mgsort->out_chunk);
		dlist_move_all(&pending->work_chunk, &mgsort->work_chunk);

		pgstrom_untrack_object(&mgsort->msg.sobj);
		pgstrom_put_message(&mgsort->msg);

		/* TODO: must be checked if CPU recheck is needed or not */

		/* OK, this two input stream shall be merged into one */
        if (!pgstrom_enqueue_message(&pending->msg))
            elog(ERROR, "Bug? OpenCL server seems to dead");
        gsortstate->num_running++;
	}
	else
	{
		Assert(dlist_is_empty(&mgsort->in_chunk1) &&
			   dlist_is_empty(&mgsort->in_chunk2) &&
			   !dlist_is_empty(&mgsort->out_chunk) &&
			   dlist_length(&mgsort->work_chunk) == 1);
		dlist_move_all(&mgsort->in_chunk1, &mgsort->out_chunk);

		/* wait for another one */
		gsortstate->pending_mgsort = mgsort;
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
	while ((anum = bms_first_member(tempset)) >= 0)
	{
		anum += FirstLowInvalidHeapAttributeNumber;
		Assert(anum > 0 && anum <= tupdesc->natts);

		((bool *)VARDATA(rs_attrefs))[anum - 1] = true;
		sortkey_resnums = lappend_int(sortkey_resnums, anum);
		if (tupdesc->attrs[anum - 1]->attlen < 0)
			sortkey_toast = lappend_int(sortkey_toast, anum);
	}
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
	gsortstate->gpusort_chunksz = get_gpusort_chunksize();
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
	return &gsortstate->cps;
}

















static TupleTableSlot *
gpusort_exec(CustomPlanState *node)
{
	GpuSortState	   *gsortstate = (GpuSortState *) node;
	pgstrom_queue	   *mqueue = gsortstate->mqueue;
	pgstrom_message	   *msg;
	pgstrom_gpusort_multi *mgsort;
	pgstrom_gpusort	   *gs_chunk;
	kern_column_store  *kcs;
	dlist_node		   *dnode;
	TupleTableSlot	   *slot;

	if (!gsortstate->sort_done)
	{
		HeapTuple	overflow = NULL;

		while (!gsortstate->scan_done)
		{
			/* preload a chunk onto pgstrom_gpusort */
			gs_chunk = gpusort_preload_chunk(gsortstate, &overflow);
			if (!gs_chunk)
			{
				Assert(!gsortstate->scan_done);
				Assert(!overflow);
				break;
			}
			if (!pgstrom_enqueue_message(&gs_chunk->msg))
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

				if (StromTagIs(msg, GpuSort))
					gpusort_process_resp_single(gsortstate,
												(pgstrom_gpusort *)msg);
				else if (StromTagIs(msg, GpuSortMulti))
					gpusort_process_resp_multi(gsortstate,
											   (pgstrom_gpusort_multi *)msg);
				else
					elog(ERROR, "Bug? unexpected message dequeued: %d",
						 (int)msg->sobj.stag);
				
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

			if (StromTagIs(msg, GpuSort))
				gpusort_process_resp_single(gsortstate,
											(pgstrom_gpusort *)msg);
			else if (StromTagIs(msg, GpuSortMulti))
				gpusort_process_resp_multi(gsortstate,
										   (pgstrom_gpusort_multi *)msg);
			else
				elog(ERROR, "Bug? unexpected message dequeued: %d",
					 (int)msg->sobj.stag);
		}
		Assert(gsortstate->pending_mgsort != NULL);

		gsortstate->sort_done = true;
	}
	/* OK, sorting done, fetch tuples according to the result */
	mgsort = gsortstate->pending_mgsort;
	slot = gsortstate->cps.ps.ps_ResultTupleSlot;
	ExecClearTuple(slot);

retry:
	if (!mgsort || dlist_is_empty(&mgsort->in_chunk1))
		return slot;

	dnode = dlist_head_node(&mgsort->in_chunk1);
	gs_chunk = dlist_container(pgstrom_gpusort, chain, dnode);
	kcs = KERN_GPUSORT_CHUNK(&gs_chunk->kern);

	if (gsortstate->curr_index < kcs->nrows)
	{
		cl_uint		   *rindex;
		cl_ulong	   *tupids;
		cl_uint			tup_gid;
		cl_uint			tup_lid;
		cl_int			i = gsortstate->curr_index;
		StromObject	   *sobject;

		rindex = (cl_uint *)((char *)kcs +
							 kcs->colmeta[kcs->ncols - 1].cs_ofs);
		tupids = (cl_ulong *)((char *)kcs + kcs->colmeta[0].cs_ofs);
		Assert(rindex[i] >= 0 && rindex[i] < kcs->nrows);

		tup_gid = ((tupids[rindex[i]] >> 32) & 0xffffffff);
		tup_lid = (tupids[rindex[i]] & 0xffffffff);
		Assert(tup_gid < gsortstate->rcs_nums);

		sobject = gsortstate->rcs_slot[tup_gid];
		if (StromTagIs(sobject, TCacheRowStore))
		{
			tcache_row_store *trs = (tcache_row_store *)sobject;
			rs_tuple   *rs_tup;

			Assert(tup_lid < trs->kern.nrows);
			rs_tup = kern_rowstore_get_tuple(&trs->kern, tup_lid);

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
	pgstrom_put_message(&gs_chunk->msg);
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
	GpuSortState   *gsortstate = (GpuSortState *) node;

	/*
	 * Release PG-Strom shared objects
	 */
	pgstrom_untrack_object((StromObject *)gsortstate->dprog_key);
	pgstrom_untrack_object(&gsortstate->mqueue->sobj);

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
{}

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
	List		   *context;
	bool			useprefix;
	List		   *tlist = gsortstate->cps.ps.plan->targetlist;
	List		   *sort_keys = NIL;
	ListCell	   *cell;

	/* logic copied from show_sort_group_keys */
	context = deparse_context_for_planstate((Node *) gsortstate,
											ancestors,
											es->rtable,
											es->rtable_names);
	useprefix = (list_length(es->rtable) > 1 || es->verbose);

	foreach(cell, gsortstate->sortkey_resnums)
	{
		TargetEntry	*tle = get_tle_by_resno(tlist, lfirst_int(cell));
		char		*exprstr;

		if (!tle)
			elog(ERROR, "no tlist entry for key %d", lfirst_int(cell));

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
	dlist_init(&gpusort_shm_values->gsort_multi_freelist);
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
static void
clserv_respond_gpusort(cl_event event, cl_int ev_status, void *private)
{

}

static void
clserv_process_gpusort(pgstrom_message *msg)
{
	//pgstrom_gpusort		   *bgsort = (pgstrom_gpusort *) msg;
}


static void
clserv_respond_gpusort_multi(cl_event event, cl_int ev_status, void *private)
{

}

static void
clserv_process_gpusort_multi(pgstrom_message *msg)
{
	//pgstrom_gpusort_multi  *mgsort = (pgstrom_gpusort_multi *) msg;

}
