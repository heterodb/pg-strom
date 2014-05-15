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
	List	   *used_vars;		/* list of Var nodes being referenced */

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
	Bitmapset	   *dev_attnums;
	bool			sort_done;

	/* row-/column- stores being read */
	cl_uint			rcs_nums;	/* number of row-/column-stores in use */
	cl_uint			rcs_size;	/* size of row-/column-store slot */
	StromObject	  **rcs_slot;	/* slot of row-/column-stores */
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
	return shift;
}


static inline Size
get_gpusort_chunksize(void)
{
	static Size	device_restriction = 0;
	Size		chunk_sz;
	int			shift = 0;

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
pgstrom_create_gpusort_plan(Sort *sort)
{
	GpuSortPlan *gsort;
	List	   *tlist = sort->plan.targetlist;
	Plan	   *subplan = sort->plan.lefttree;
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
	gsort->extra_flags = context.extra_flags;
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
	pgstrom_gpusort	   *bgsort = (pgstrom_gpusort *)msg;

	/* unlink message queue and device program (should always exists) */
	pgstrom_put_queue(msg->respq);
	pgstrom_put_devprog_key(bgsort->dprog_key);

}

static pgstrom_gpusort *
pgstrom_create_gpusort(TupleDesc tupdesc, Bitmapset *dev_attnums)
{




static typedef struct
{
    pgstrom_message msg;    /* = StromTag_GpuSort */
    Datum           dprog_key;
    dlist_node      chain;  /* be linked to pgstrom_gpusort_multi */
    dlist_head      rcstore_list;
    cl_uint         rcstore_index;
    cl_uint         rcstore_nums;
    kern_gpusort    kern;
} pgstrom_gpusort;
static tcache_column_store *
gpusort_create




	return NULL;
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
pgstrom_create_gpusort_multi(pgstrom_queue *respq, Datum dprog_key)
{
	pgstrom_gpusort_multi  *mgsort;
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
	mgsort->msg.respq = pgstrom_get_queue(respq);
	mgsort->msg.cb_process = clserv_process_gpusort_multi;
	mgsort->msg.cb_release = pgstrom_release_gpusort_multi;
	/* other fields also */
	mgsort->dprog_key = pgstrom_retain_devprog_key(dprog_key);
	dlist_init(&mgsort->in_chunk1);
	dlist_init(&mgsort->in_chunk2);
	dlist_init(&mgsort->out_chunk);

	return mgsort;
}










static CustomPlanState *
gpusort_begin(CustomPlan *node, EState *estate, int eflags)
{
	GpuSortPlan	   *gsortplan = (GpuSortPlan *) node;
	GpuSortState   *gsortstate;
	TupleDesc		tupdesc;
	int				extra_flags;
	Bitmapset	   *dev_attnums;
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

	pull_varattnos((Node *)gsortplan->used_vars,
				   OUTER_VAR,
				   &dev_attnums);
	rs_attrefs = palloc0(VARHDRSZ + sizeof(cl_char) * tupdesc->natts);
	SET_VARSIZE(rs_attrefs, VARHDRSZ + sizeof(cl_char) * tupdesc->natts);
	for (anum=0; anum < tupdesc->natts; anum++)
	{
		Form_pg_attribute attr = tupdesc->attrs[anum];
		int		x = attr->attnum - FirstLowInvalidHeapAttributeNumber;

		if (bms_is_member(x, dev_attnums))
			((bool *)VARDATA(rs_attrefs))[anum] = true;
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
	gsortstate->dev_attnums = dev_attnums;

	/* allocate a certain amount of row-/column-store slot */
	gsortstate->rcs_nums = 0;
	gsortstate->rcs_size = 256;
	gsortstate->rcs_slot = palloc0(sizeof(StromObject *) *
								   gsortstate->rcs_size);
	return NULL;
}














static TupleTableSlot *
gpusort_exec(CustomPlanState *node)
{
	return NULL;
}

static Node *
gpusort_exec_multi(CustomPlanState *node)
{
	elog(ERROR, "Not supported yet");
	return NULL;
}

static void
gpusort_end(CustomPlanState *node)
{}

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

	temp = nodeToString(plannode->used_vars);
	appendStringInfo(str, " :used_vars %s", temp);
	pfree(temp);

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
	newnode->used_vars   = copyObject(oldnode->used_vars);
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
	pgstrom_gpusort		   *bgsort = (pgstrom_gpusort *) msg;
}


static void
clserv_respond_gpusort_multi(cl_event event, cl_int ev_status, void *private)
{

}

static void
clserv_process_gpusort_multi(pgstrom_message *msg)
{
	pgstrom_gpusort_multi  *mgsort = (pgstrom_gpusort_multi *) msg;

}
