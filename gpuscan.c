/*
 * gpuscan.c
 *
 * Sequential scan accelerated by GPU processors
 * ----
 * Copyright 2011-2014 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014 (C) The PG-Strom Development Team
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include "postgres.h"
#include "access/sysattr.h"
#include "catalog/pg_type.h"
#include "catalog/pg_namespace.h"
#include "commands/explain.h"
#include "miscadmin.h"
#include "nodes/bitmapset.h"
#include "nodes/execnodes.h"
#include "nodes/makefuncs.h"
#include "nodes/plannodes.h"
#include "nodes/relation.h"
#include "optimizer/cost.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/plancat.h"
#include "optimizer/planmain.h"
#include "optimizer/restrictinfo.h"
#include "optimizer/var.h"
#include "parser/parsetree.h"
#include "storage/bufmgr.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/spccache.h"
#include "utils/tqual.h"
#include "pg_strom.h"
#include "opencl_gpuscan.h"

static add_scan_path_hook_type	add_scan_path_next;
static CustomPathMethods		gpuscan_path_methods;
static CustomPlanMethods		gpuscan_plan_methods;

typedef struct {
	CustomPath	cpath;
	List	   *dev_quals;		/* RestrictInfo run on device */
	List	   *host_quals;		/* RestrictInfo run on host */
	Bitmapset  *dev_attnums;	/* attnums referenced in device */
	Bitmapset  *host_attnums;	/* attnums referenced in host */
} GpuScanPath;

typedef struct {
	CustomPlan	cplan;
	Index		scanrelid;		/* index of the range table */
	const char *kern_source;	/* source of opencl kernel */
	int			extra_flags;	/* extra libraries to be included */
	List	   *used_params;	/* list of Const/Param in use */
	List	   *used_vars;		/* list of Var in use */
	List	   *dev_clauses;	/* clauses to be run on device */
	Bitmapset  *dev_attnums;	/* attnums referenced in device */
	Bitmapset  *host_attnums;	/* attnums referenced in host */
} GpuScanPlan;

/*
 * Gpuscan has three strategy to scan a relation.
 * a) cache-only scan, if all the variables being referenced in target-list
 *    and scan-qualifiers are on the t-tree columnar cache.
 *    It is capable to return a columner-store, instead of individual rows,
 *    if upper plan node is also managed by PG-Strom.
 * b) hybrid-scan, if Var references by scan-qualifiers are on cache, but
 *    ones by target-list are not. It runs first screening by device, then
 *    fetch a tuple from the shared buffers.
 * c) heap-only scan, if all the variables in scan-qualifier are not on
 *    the cache, all we can do is read tuples from shared-buffer to the
 *    row-store, then picking it up.
 * In case of (a) and (b), gpuscan needs to be responsible to MVCC checks;
 * that is not done on the first evaluation timing.
 * In case of (c), it may construct a columnar cache entry that caches the
 * required columns.
 */
typedef struct {
	CustomPlanState		cps;
	Relation			scan_rel;
	HeapScanDesc		scan_desc;
	TupleTableSlot	   *scan_slot;
	List			   *dev_quals;
	tcache_head		   *tc_head;
	tcache_scandesc	   *tc_scan;
	bool				hybrid_scan;
	HeapTupleData		hybrid_htup;

	pgstrom_queue	   *mqueue;
	Datum				dprog_key;

	kern_parambuf	   *kparambuf;
	AttrNumber		   *cs_attidxs;
	int					cs_attnums;

	pgstrom_gpuscan	   *curr_chunk;
	uint32				curr_index;
	int					num_running;
	dlist_head			ready_chunks;

	pgstrom_perfmon		pfm;	/* sum of performance counter */
} GpuScanState;

/* static functions */
static void clserv_process_gpuscan_row(pgstrom_message *msg);
static void clserv_process_gpuscan_column(pgstrom_message *msg);
static void clserv_put_gpuscan(pgstrom_message *msg);

/*
 * cost_gpuscan
 *
 * cost estimation for GpuScan
 */
static void
cost_gpuscan(GpuScanPath *gpu_path, PlannerInfo *root,
			 RelOptInfo *baserel, ParamPathInfo *param_info,
			 List *dev_quals, List *host_quals)
{
	Path	   *path = &gpu_path->cpath.path;
	Cost		startup_cost = 0;
	Cost		run_cost = 0;
	double		spc_seq_page_cost;
	QualCost	dev_cost;
	QualCost	host_cost;
	Cost		gpu_per_tuple;
	Cost		cpu_per_tuple;
	Selectivity	dev_sel;

	/* Should only be applied to base relations */
	Assert(baserel->relid > 0);
	Assert(baserel->rtekind == RTE_RELATION);

	/* Mark the path with the correct row estimate */
	if (param_info)
		path->rows = param_info->ppi_rows;
	else
		path->rows = baserel->rows;

	/* fetch estimated page cost for tablespace containing table */
	get_tablespace_page_costs(baserel->reltablespace,
							  NULL,
							  &spc_seq_page_cost);
	/*
	 * disk costs
	 * XXX - needs to adjust after columner cache in case of bare heapscan,
	 * or partial heapscan if targetlist references out of cached columns
	 */
    run_cost += spc_seq_page_cost * baserel->pages;

	/* GPU costs */
	cost_qual_eval(&dev_cost, dev_quals, root);
	dev_sel = clauselist_selectivity(root, dev_quals, 0, JOIN_INNER, NULL);

	/*
	 * XXX - very rough estimation towards GPU startup and device calculation
	 *       to be adjusted according to device info
	 *
	 * TODO: startup cost takes NITEMS_PER_CHUNK * width to be carried, but
	 * only first chunk because data transfer is concurrently done, if NOT
	 * integrated GPU
	 * TODO: per_tuple calculation cost shall be divided by parallelism of
	 * average opencl spec.
	 */
	dev_cost.startup += 10000;
	dev_cost.per_tuple /= 100;

	/* CPU costs */
	cost_qual_eval(&host_cost, host_quals, root);
	if (param_info)
	{
		QualCost	param_cost;

		/* Include costs of pushed-down clauses */
		cost_qual_eval(&param_cost, param_info->ppi_clauses, root);
		host_cost.startup += param_cost.startup;
		host_cost.per_tuple += param_cost.per_tuple;
	}

	/* total path cost */
	startup_cost += dev_cost.startup + host_cost.startup;
	cpu_per_tuple = cpu_tuple_cost + host_cost.per_tuple;
	gpu_per_tuple = cpu_tuple_cost / 100 + dev_cost.per_tuple;
	run_cost += (gpu_per_tuple * baserel->tuples +
				 cpu_per_tuple * dev_sel * baserel->tuples);

    gpu_path->cpath.path.startup_cost = startup_cost;
    gpu_path->cpath.path.total_cost = startup_cost + run_cost;
}

static void
gpuscan_add_scan_path(PlannerInfo *root,
					  RelOptInfo *baserel,
					  RangeTblEntry *rte)
{
	GpuScanPath	   *pathnode;
	Relation		rel;
	List		   *dev_quals = NIL;
	List		   *host_quals = NIL;
	Bitmapset	   *dev_attnums = NULL;
	Bitmapset	   *host_attnums = NULL;
	ListCell	   *cell;
	codegen_context	context;

	/* call the secondary hook */
	if (add_scan_path_next)
		add_scan_path_next(root, baserel, rte);

	/* Is PG-Strom enabled? */
	if (!pgstrom_enabled)
		return;

	/* only base relation we can handle */
	if (baserel->rtekind != RTE_RELATION || baserel->relid == 0)
		return;

	/* system catalog is not supported */
	if (get_rel_namespace(rte->relid) == PG_CATALOG_NAMESPACE)
		return;

	/* also, relation has to have synchronizer trigger */
	rel = heap_open(rte->relid, NoLock);
	if (!pgstrom_relation_has_synchronizer(rel))
	{
		heap_close(rel, NoLock);
		elog(INFO, "no synchronizer!");
		return;
	}
	heap_close(rel, NoLock);

	/* check whether qualifier can run on GPU device */
	memset(&context, 0, sizeof(codegen_context));
	foreach (cell, baserel->baserestrictinfo)
	{
		RestrictInfo   *rinfo = lfirst(cell);

		if (pgstrom_codegen_available_expression(rinfo->clause))
		{
			pull_varattnos((Node *)rinfo->clause,
						   baserel->relid,
						   &dev_attnums);
			dev_quals = lappend(dev_quals, rinfo);
		}
		else
		{
			pull_varattnos((Node *)rinfo->clause,
						   baserel->relid,
						   &host_attnums);
			host_quals = lappend(host_quals, rinfo);
		}
	}
	/* also, picks up Var nodes in the target list */
	pull_varattnos((Node *)baserel->reltargetlist,
				   baserel->relid,
				   &host_attnums);
	/*
	 * FIXME: needs to pay attention for projection cost.
	 * It may make sense to use build_physical_tlist, if host_attnums
	 * are much wider than dev_attnums.
	 * Anyway, it needs investigation of the actual behavior.
	 */

	/* XXX - check whether columnar cache may be available */

	/*
	 * Construction of a custom-plan node.
	 */
	pathnode = palloc0(sizeof(GpuScanPath));
	pathnode->cpath.path.type = T_CustomPath;
	pathnode->cpath.path.pathtype = T_CustomPlan;
	pathnode->cpath.path.parent = baserel;
	pathnode->cpath.path.param_info
		= get_baserel_parampathinfo(root, baserel, baserel->lateral_relids);
	pathnode->cpath.path.pathkeys = NIL;	/* gpuscan has unsorted result */
	pathnode->cpath.methods = &gpuscan_path_methods;

	cost_gpuscan(pathnode, root, baserel,
				 pathnode->cpath.path.param_info,
				 dev_quals, host_quals);

	pathnode->dev_quals = dev_quals;
	pathnode->host_quals = host_quals;
	pathnode->dev_attnums = dev_attnums;
	pathnode->host_attnums = host_attnums;

	add_path(baserel, &pathnode->cpath.path);
}

/*
 * OpenCL code generation that can run on GPU/MIC device
 */
static char *
gpuscan_codegen_quals(PlannerInfo *root, List *dev_quals,
					  codegen_context *context)
{
	StringInfoData	str;
	char		   *expr_code;

	memset(context, 0, sizeof(codegen_context));
	if (dev_quals == NIL)
		return NULL;

	/*
	 * A dummy constant - KPARAM_0 is an array of bool to show referenced
	 * columns, in GpuScan.
	 * Just a placeholder here. Set it up later.
	 */
	context->used_params = list_make1(makeConst(BYTEAOID,
												-1,
												InvalidOid,
												-1,
												PointerGetDatum(NULL),
												true,
												false));
	context->type_defs = list_make1(pgstrom_devtype_lookup(BYTEAOID));

	/* OK, let's walk on the device expression tree */
	expr_code = pgstrom_codegen_expression((Node *)dev_quals, context);
	Assert(expr_code != NULL);

	initStringInfo(&str);

	/*
	 * Put declarations of device types, functions and macro definitions
	 */
	appendStringInfo(&str, "%s\n", pgstrom_codegen_declarations(context));

	/* qualifier definition with row-store */
	appendStringInfo(
		&str,
		"static pg_bool_t\n"
		"gpuscan_qual_eval(__private cl_int *errcode,\n"
		"                  __global kern_gpuscan *kgscan,\n"
		"                  __global kern_column_store *kcs,\n"
		"                  __global kern_toastbuf *toast,\n"
		"                  size_t kcs_index)\n"
		"{\n"
		"  __global kern_parambuf *kparams\n"
		"      = KERN_GPUSCAN_PARAMBUF(kgscan);\n"
		"\n"
		"  return %s;\n"
		"}\n", expr_code);
	return str.data;
}

static CustomPlan *
gpuscan_create_plan(PlannerInfo *root, CustomPath *best_path)
{
	RelOptInfo	   *rel = best_path->path.parent;
	GpuScanPath	   *gpath = (GpuScanPath *)best_path;
	GpuScanPlan	   *gscan;
	List		   *tlist;
	List		   *host_clauses;
	List		   *dev_clauses;
	char		   *kern_source;
	codegen_context	context;

	/*
	 * See the comments in create_scan_plan(). We may be able to omit
	 * projection of the table tuples, if possible.
	 */
	if (use_physical_tlist(root, rel))
	{
		tlist = build_physical_tlist(root, rel);
		if (tlist == NIL)
			tlist = build_path_tlist(root, &best_path->path);
	}
	else
		tlist = build_path_tlist(root, &best_path->path);

	/* it should be a base relation */
	Assert(rel->relid > 0);
	Assert(rel->rtekind == RTE_RELATION);

	/* Sort clauses into best execution order */
	host_clauses = order_qual_clauses(root, gpath->host_quals);
	dev_clauses = order_qual_clauses(root, gpath->dev_quals);

	/* Reduce RestrictInfo list to bare expressions; ignore pseudoconstants */
	host_clauses = extract_actual_clauses(host_clauses, false);
	dev_clauses = extract_actual_clauses(dev_clauses, false);

	/* Replace any outer-relation variables with nestloop params */
	if (best_path->path.param_info)
	{
		host_clauses = (List *)
			replace_nestloop_params(root, (Node *) host_clauses);
		dev_clauses = (List *)
			replace_nestloop_params(root, (Node *) dev_clauses);
	}

	/*
	 * Construct OpenCL kernel code - A kernel code contains two forms of
	 * entrypoints; for row-store and column-store. OpenCL intermediator
	 * invoked proper kernel function according to the class of data store.
	 * Once a kernel function for row-store is called, it translates the
	 * data format into column-store, then kicks jobs for row-evaluation.
	 * This design is optimized to process column-oriented data format on
	 * the relation cache.
	 */
	kern_source = gpuscan_codegen_quals(root, dev_clauses, &context);

	/*
	 * Construction of GpuScanPlan node; on top of CustomPlan node
	 */
	gscan = palloc0(sizeof(GpuScanPlan));
	gscan->cplan.plan.type = T_CustomPlan;
	gscan->cplan.plan.targetlist = tlist;
	gscan->cplan.plan.qual = host_clauses;
	gscan->cplan.plan.lefttree = NULL;
	gscan->cplan.plan.righttree = NULL;
	gscan->cplan.methods = &gpuscan_plan_methods;

	gscan->scanrelid = rel->relid;
	gscan->kern_source = kern_source;
	gscan->extra_flags = context.extra_flags | DEVKERNEL_NEEDS_GPUSCAN;
	gscan->used_params = context.used_params;
	gscan->used_vars = context.used_vars;
	gscan->dev_clauses = dev_clauses;
	gscan->dev_attnums = gpath->dev_attnums;
	gscan->host_attnums = gpath->host_attnums;

	return &gscan->cplan;
}

static void
gpuscan_textout_path(StringInfo str, Node *node)
{
	GpuScanPath	   *pathnode = (GpuScanPath *) node;
	char		   *temp;

	/* dev_quals */
	temp = nodeToString(pathnode->dev_quals);
	appendStringInfo(str, " :dev_quals %s", temp);
	pfree(temp);

	/* host_quals */
	temp = nodeToString(pathnode->host_quals);
	appendStringInfo(str, " :host_quals %s", temp);
	pfree(temp);

	/* dev_attnums */
	appendStringInfo(str, " :dev_attnums ");
	_outBitmapset(str, pathnode->dev_attnums);

	/* host_attnums */
	appendStringInfo(str, " :host_attnums ");
	_outBitmapset(str, pathnode->host_attnums);
}

static void
gpuscan_set_plan_ref(PlannerInfo *root,
					 CustomPlan *custom_plan,
					 int rtoffset)
{
	GpuScanPlan	   *gscan = (GpuScanPlan *)custom_plan;

	gscan->scanrelid += rtoffset;
	gscan->cplan.plan.targetlist = (List *)
		fix_scan_expr(root, (Node *)gscan->cplan.plan.targetlist, rtoffset);
	gscan->cplan.plan.qual = (List *)
		fix_scan_expr(root, (Node *)gscan->cplan.plan.qual, rtoffset);
	gscan->used_vars = (List *)
		fix_scan_expr(root, (Node *)gscan->used_vars, rtoffset);
	gscan->dev_clauses = (List *)
		fix_scan_expr(root, (Node *)gscan->dev_clauses, rtoffset);
}

static void
gpuscan_finalize_plan(PlannerInfo *root,
					  CustomPlan *custom_plan,
					  Bitmapset **paramids,
					  Bitmapset **valid_params,
					  Bitmapset **scan_params)
{
	*paramids = bms_add_members(*paramids, *scan_params);
}

static bool
gpuscan_begin_tcache_scan(GpuScanState *gss,
						  Bitmapset *host_attnums,
						  Bitmapset *dev_attnums)
{
	Relation		scan_rel = gss->scan_rel;
	Oid				scan_reloid = RelationGetRelid(scan_rel);
	tcache_head	   *tc_head = NULL;
	tcache_scandesc	*tc_scan = NULL;
	Bitmapset	   *tempset;
	bool			hybrid_scan = false;
	Datum			tr_private = 0;

	tempset = bms_union(host_attnums, dev_attnums);
	PG_TRY();
	{
		/*
		 * First of all, we try the plan-a; that uses cache-only scan
		 * if both of reference columns by host and by device are on
		 * the tcache.
		 * If found, it's perfect. We don't need to touch heap in
		 * this scan.
		 */
		tc_head = tcache_get_tchead(scan_reloid, tempset);
		if (!tc_head)
		{
			if (!bms_equal(dev_attnums, tempset))
			{
				/*
				 * As a second best, we try hybrid approach that uses tcache
				 * for computation in device, but host-side fetches tuples
				 * from the heap pages according to item-pointers.
				 */
				tc_head = tcache_get_tchead(scan_reloid, dev_attnums);
				if (tc_head)
					hybrid_scan = true;
			}
			/*
			 * In case when we have no suitable tcache anyway, let's
			 * create a new one and build up it.
			 */
			if (!tc_head)
				tc_head = tcache_try_create_tchead(scan_reloid, tempset);
		}
		/* OK, let's try to make tcache-scan handler */
		if (tc_head)
		{
			tc_scan = tcache_begin_scan(tc_head, scan_rel);
			if (!tc_scan)
			{
				/*
				 * Even if we could get tcache_head, we might fail to begin
				 * tcache-scan because of concurrent cache build. In this
				 * case, we take usual heap scan using row-store.
				 */
				tcache_put_tchead(tc_head);
				tc_head = NULL;
			}
			else if (tc_scan->heapscan)
			{
				/*
				 * In case when tcache_begin_scan returns a tcache_scandesc
				 * with a valid heapscan, it means this scan also performs
				 * as a tcache builder.
				 */
				tr_private = BoolGetDatum(true);
			}
		}
	}
	PG_CATCH();
	{
		if (tc_head)
			tcache_put_tchead(tc_head);
	}
	PG_END_TRY();

	if (!tc_head)
		return false;

	pgstrom_track_object(&tc_head->sobj, tr_private);
	gss->tc_head = tc_head;
	gss->tc_scan = tc_scan;
	gss->hybrid_scan = hybrid_scan;

	return true;
}

static  CustomPlanState *
gpuscan_begin(CustomPlan *node, EState *estate, int eflags)
{
	GpuScanPlan	   *gsplan = (GpuScanPlan *) node;
	Index			scanrelid = gsplan->scanrelid;
	GpuScanState   *gss;
	TupleDesc		tupdesc;
	bytea		   *attrefs;
	Const		   *kparam_0;
	int32			extra_flags;
	AttrNumber		anum;
	AttrNumber		anum_last;

	/* gpuscan should not have inner/outer plan now */
	Assert(outerPlan(node) == NULL);
    Assert(innerPlan(node) == NULL);

	/*
	 * create a state structure
	 */
	gss = palloc0(sizeof(GpuScanState));
	gss->cps.ps.type = T_CustomPlanState;
	gss->cps.ps.plan = (Plan *) node;
	gss->cps.ps.state = estate;
	gss->cps.methods = &gpuscan_plan_methods;

	/*
	 * create expression context
	 */
	ExecAssignExprContext(estate, &gss->cps.ps);

	/*
	 * initialize child expressions
	 */
	gss->cps.ps.targetlist = (List *)
		ExecInitExpr((Expr *) node->plan.targetlist, &gss->cps.ps);
	gss->cps.ps.qual = (List *)
		ExecInitExpr((Expr *) node->plan.qual, &gss->cps.ps);
	gss->dev_quals = (List *)
		ExecInitExpr((Expr *) gsplan->dev_clauses, &gss->cps.ps);

	/*
	 * tuple table initialization
	 */
	ExecInitResultTupleSlot(estate, &gss->cps.ps);
	gss->scan_slot = ExecAllocTableSlot(&estate->es_tupleTable);

	/*
	 * initialize scan relation
	 */
	gss->scan_rel = ExecOpenScanRelation(estate, scanrelid, eflags);
	tupdesc = RelationGetDescr(gss->scan_rel);
	ExecSetSlotDescriptor(gss->scan_slot, tupdesc);

	/*
	 * Initialize result tuple type and projection info.
	 */
	ExecAssignResultTypeFromTL(&gss->cps.ps);
	if (tlist_matches_tupdesc(&gss->cps.ps,
							  node->plan.targetlist,
							  scanrelid,
							  tupdesc))
		gss->cps.ps.ps_ProjInfo = NULL;
	else
		ExecAssignProjectionInfo(&gss->cps.ps, tupdesc);

	/*
	 * OK, initialization of common part is over.
	 * Let's have GPU stuff initialization
	 */
	if (!gpuscan_begin_tcache_scan(gss,
								   gsplan->host_attnums,
								   gsplan->dev_attnums))
	{
		/*
		 * Oops, someone concurrently build a tcache.
		 * We will move on the regular heap scan as fallback.
		 * One (small) benefit in this option is that MVCC
		 * check is already done in heap_fetch()
		 */
		gss->scan_desc = heap_beginscan(gss->scan_rel,
										estate->es_snapshot,
										0,
										NULL);
	}

	/*
	 * Setting up kernel program, if needed
	 */
	extra_flags = gsplan->extra_flags;
	if (pgstrom_kernel_debug)
		extra_flags |= DEVKERNEL_NEEDS_DEBUG;
	if (gsplan->kern_source)
	{
		gss->dprog_key = pgstrom_get_devprog_key(gsplan->kern_source,
												 extra_flags);
		pgstrom_track_object((StromObject *)gss->dprog_key, 0);

		/* also, message queue */
		gss->mqueue = pgstrom_create_queue();
		pgstrom_track_object(&gss->mqueue->sobj, 0);
	}

	/* construct an index of attributes being referenced in device */
	attrefs = palloc0(VARHDRSZ + sizeof(cl_bool) * tupdesc->natts);
	SET_VARSIZE(attrefs, VARHDRSZ + sizeof(bool) * tupdesc->natts);
	gss->cs_attidxs = palloc0(sizeof(AttrNumber) * tupdesc->natts);
	gss->cs_attnums = 0;
	anum_last = -1;
	for (anum=0; anum < tupdesc->natts; anum++)
	{
		Form_pg_attribute attr = tupdesc->attrs[anum];
		int		x = attr->attnum - FirstLowInvalidHeapAttributeNumber;

		if (bms_is_member(x, gsplan->dev_attnums))
		{
			gss->cs_attidxs[gss->cs_attnums++] = anum;
			((cl_char *)VARDATA(attrefs))[anum] = 1;
			anum_last = anum;
		}
	}
	/* negative value is end of referenced columns marker */
	if (anum_last >= 0)
		((cl_char *)VARDATA(attrefs))[anum_last] = -1;

	/*
	 * template of kern_parambuf.
	 * NOTE: KPARAM_0 is reserved to have referenced attributes
	 */
	if (gsplan->kern_source)
	{
		Assert(list_length(gsplan->used_params) > 0);
		kparam_0 = (Const *)linitial(gsplan->used_params);
		Assert(IsA(kparam_0, Const) &&
			   kparam_0->consttype == BYTEAOID &&
			   kparam_0->constisnull);
		kparam_0->constvalue = PointerGetDatum(attrefs);
		kparam_0->constisnull = false;
	}
	gss->kparambuf = pgstrom_create_kern_parambuf(gsplan->used_params,
												  gss->cps.ps.ps_ExprContext);
	/* rest of run-time parameters */
	gss->curr_chunk = NULL;
	gss->curr_index = 0;
	gss->num_running = 0;
	dlist_init(&gss->ready_chunks);

	/* Is perfmon needed? */
	gss->pfm.enabled = pgstrom_perfmon_enabled;

	return &gss->cps;
}

static void
pgstrom_setup_kern_colstore_head(GpuScanState *gss, kern_resultbuf *kresult,
								 bool with_row_store)
{
	kern_column_store *kcs_head = (kern_column_store *)kresult->results;
	kern_toastbuf *ktoast;
	TupleDesc	tupdesc = RelationGetDescr(gss->scan_rel);
	cl_uint		ncols = gss->cs_attnums;
	cl_uint		nrooms = kresult->nrooms;
	cl_uint	   *i_refcols;
	Size		offset;
	int			i, j, k;

	/*
	 * In case of row-store is processed, kern_row_to_column will increase
	 * the kcs_head->nrows in device. On the other hands, gpuscan_qual_cs()
	 * handles already preprocessed column-store, it has to be same value
	 * with kcs_head->nrooms (because we create a tightly fit kcs).
	 */
	kcs_head->ncols = ncols;
	kcs_head->nrows = (with_row_store ? 0 : nrooms);
	kcs_head->nrooms = nrooms;
	ktoast = (kern_toastbuf *)&kcs_head->colmeta[ncols];
	i_refcols = (cl_uint *)&ktoast->coldir[ncols];

	offset = STROMALIGN(offsetof(kern_column_store,
								 colmeta[gss->cs_attnums]));
	for (i=0, k=0; i < ncols; i++)
	{
		kern_colmeta   *kcmeta = &kcs_head->colmeta[i];
		Form_pg_attribute attr;

		j = gss->cs_attidxs[i];
		attr = tupdesc->attrs[j];

		memset(kcmeta, 0, sizeof(kern_colmeta));
		kcmeta->attnotnull = attr->attnotnull;
		if (attr->attalign == 'c')
			kcmeta->attalign = sizeof(cl_char);
		else if (attr->attalign == 's')
			kcmeta->attalign = sizeof(cl_short);
		else if (attr->attalign == 'i')
			kcmeta->attalign = sizeof(cl_int);
		else if (attr->attalign == 'd')
			kcmeta->attalign = sizeof(cl_long);
		else
			elog(ERROR, "unexpected attalign '%c'", attr->attalign);

		kcmeta->attlen = attr->attlen;
		kcmeta->cs_ofs = offset;

		if (!kcmeta->attnotnull)
			offset += STROMALIGN((nrooms + BITS_PER_BYTE-1) / BITS_PER_BYTE);
		offset += STROMALIGN(nrooms * (attr->attlen > 0
									   ? attr->attlen
									   : sizeof(cl_uint)));
		if (gss->tc_scan)
		{
			tcache_head	*tc_head = gss->tc_head;

			while (k < tc_head->ncols)
			{
				if (j == tc_head->i_cached[k])
				{
					i_refcols[i] = k;
					break;
				}
				k++;
			}
			if (k == tc_head->ncols)
				elog(ERROR, "Bug? uncached columns are referenced");
		}
	}
	kcs_head->length = offset;
}

static pgstrom_gpuscan *
pgstrom_load_gpuscan(GpuScanState *gss)
{
	pgstrom_gpuscan	   *gpuscan;
	tcache_row_store   *trs = NULL;
	tcache_column_store *tcs = NULL;
	kern_parambuf	   *kparam;
	kern_resultbuf	   *kresult;
	int			extra_flags;
	cl_uint		length;
	cl_uint		i, nrows;
	bool		kernel_debug;
	struct timeval tv1, tv2;

	/* no more records to read! */
	if (!gss->scan_desc && !gss->tc_scan)
		return NULL;

	/*
	 * check status of pg_strom.kernel_debug
	 */
	extra_flags = pgstrom_get_devprog_extra_flags(gss->dprog_key);
	if ((extra_flags & DEVKERNEL_NEEDS_DEBUG) != 0)
		kernel_debug = true;
	else
		kernel_debug = false;

	/*
	 * First of all, allocate a row- or column- store and fill it up.
	 */
	if (gss->pfm.enabled)
		gettimeofday(&tv1, NULL);

	PG_TRY();
	{
		StromObject	   *sobject;
		ScanDirection	direction;

		direction = gss->cps.ps.state->es_direction;
		Assert(!ScanDirectionIsNoMovement(direction));

		if (!gss->tc_scan)
		{
			TupleDesc	tupdesc = RelationGetDescr(gss->scan_rel);
			HeapTuple	tuple;

			trs = tcache_create_row_store(tupdesc);
			while (HeapTupleIsValid(tuple = heap_getnext(gss->scan_desc,
														 direction)))
			{
				if (!tcache_row_store_insert_tuple(trs, tuple))
				{
					/*
					 * In case when we have no room to put tuples on the
					 * row-store, we rewind the tuple (to be read on the
					 * next trial) and break the loop.
					 */
					heap_getnext(gss->scan_desc, -direction);
					break;
				}
			}

			if (HeapTupleIsValid(tuple))
				sobject = &trs->sobj;
			else
			{
				heap_endscan(gss->scan_desc);
				gss->scan_desc = NULL;
				if (trs->kern.nrows > 0)
					sobject = &trs->sobj;
				else
				{
					tcache_put_row_store(trs);
					sobject = NULL;
				}
			}
		}
		else
		{
			if (ScanDirectionIsForward(direction))
				sobject = tcache_scan_next(gss->tc_scan);
			else
				sobject = tcache_scan_prev(gss->tc_scan);
			if (!sobject)
			{
				tcache_scandesc	*tc_scan = gss->tc_scan;

				gss->pfm.time_tcache_build = tc_scan->time_tcache_build;
				tcache_end_scan(tc_scan);
				gss->tc_scan = NULL;
			}
		}
		if (gss->pfm.enabled)
			gettimeofday(&tv2, NULL);

		if (sobject)
		{
			/*
			 * OK, let's create a pgstrom_gpuscan structure according to
			 * the pgstrom_row_store being allocated above.
			 */
			if (StromTagIs(sobject, TCacheRowStore))
				nrows = ((tcache_row_store *)sobject)->kern.nrows;
			else
				nrows = ((tcache_column_store *)sobject)->nrows;

			length = (STROMALIGN(offsetof(pgstrom_gpuscan, kern.kparam)) +
					  STROMALIGN(gss->kparambuf->length) +
					  STROMALIGN(Max(offsetof(kern_resultbuf,
											  results[nrows]),
									 offsetof(kern_resultbuf,
											  results[0]) +
									 offsetof(kern_column_store,
											  colmeta[gss->cs_attnums]) +
									 offsetof(kern_toastbuf,
											  coldir[gss->cs_attnums]) +
									 sizeof(cl_int) * gss->cs_attnums)) +
					  (kernel_debug ? KERNEL_DEBUG_BUFSIZE : 0));
			gpuscan = pgstrom_shmem_alloc(length);
			if (!gpuscan)
				elog(ERROR, "out of shared memory");

			/* Fields of pgstrom_gpuscan */
			memset(gpuscan, 0, sizeof(pgstrom_gpuscan));
			gpuscan->msg.sobj.stag = StromTag_GpuScan;
			SpinLockInit(&gpuscan->msg.lock);
			gpuscan->msg.refcnt = 1;
			if (gss->dprog_key != 0)
			{
				gpuscan->msg.respq = pgstrom_get_queue(gss->mqueue);
				gpuscan->dprog_key
					= pgstrom_retain_devprog_key(gss->dprog_key);
			}
			gpuscan->msg.cb_process = (StromTagIs(sobject, TCacheRowStore)
									   ? clserv_process_gpuscan_row
									   : clserv_process_gpuscan_column);
			gpuscan->msg.cb_release = clserv_put_gpuscan;
			gpuscan->msg.pfm.enabled = gss->pfm.enabled;
			if (gpuscan->msg.pfm.enabled)
				gpuscan->msg.pfm.time_to_load += timeval_diff(&tv1, &tv2);
			gpuscan->rc_store = sobject;

			/* kern_parambuf */
			kparam = &gpuscan->kern.kparam;
			memcpy(kparam, gss->kparambuf, gss->kparambuf->length);
			Assert(gss->kparambuf->length
				   == STROMALIGN(gss->kparambuf->length));

			/* kern_resultbuf portion */
			kresult = KERN_GPUSCAN_RESULTBUF(&gpuscan->kern);
			kresult->nrooms = nrows;
			kresult->nitems = 0;
			kresult->debug_nums = 0;
			kresult->debug_usage = (kernel_debug ? 0 : KERN_DEBUG_UNAVAILABLE);
			kresult->errcode = 0;

			/*
			 * In case of no device program is valid, it expects all the
			 * filtering jobs are done on host side later, so we modify
			 * the result array as if all the records are visible according
			 * to opencl kernel.
			 * Elsewhere, result array of kern_resultbuf has another usage.
			 * It is never used until writeback-dma happen, so we can use
			 * this region as a source of the header portion of
			 * kern_column_store structure.
			 */
			if (gss->dprog_key == 0)
			{
				for (i=0; i < kresult->nrooms; i++)
					kresult->results[i] = i + 1;
				kresult->nitems = kresult->nrooms;
			}
			else
			{
				bool	with_row_store = StromTagIs(sobject, TCacheRowStore);
				pgstrom_setup_kern_colstore_head(gss, kresult, with_row_store);
			}
			Assert(pgstrom_shmem_sanitycheck(gpuscan));
		}
		else
			gpuscan = NULL;
	}
	PG_CATCH();
	{
		if (trs)
			tcache_put_row_store(trs);
		if (tcs)
			tcache_put_column_store(tcs);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* track local object */
	if (gpuscan)
		pgstrom_track_object(&gpuscan->msg.sobj, 0);

	return gpuscan;
}

static bool
gpuscan_next_tuple(GpuScanState *gss, TupleTableSlot *slot)
{
	pgstrom_gpuscan	*gpuscan = gss->curr_chunk;
	kern_resultbuf	*kresult;
	Snapshot	snapshot = gss->cps.ps.state->es_snapshot;
	cl_int		i_result;
	bool		do_recheck;

	if (!gpuscan)
		return false;

	kresult = KERN_GPUSCAN_RESULTBUF(&gpuscan->kern);
   	while (gss->curr_index < kresult->nitems)
	{
		StromObject	   *sobject = gpuscan->rc_store;

		Assert(StromTagIs(sobject, TCacheRowStore) ||
			   StromTagIs(sobject, TCacheColumnStore));

		i_result = kresult->results[gss->curr_index++];
		if (i_result> 0)
			do_recheck = false;
		else
		{
			i_result = -i_result;
			do_recheck = true;
		}
		Assert(i_result > 0 && i_result <= kresult->nrooms);

		if (StromTagIs(sobject, TCacheRowStore))
		{
			tcache_row_store *trs = (tcache_row_store *)sobject;
			rs_tuple   *rs_tup;

			rs_tup = kern_rowstore_get_tuple(&trs->kern, i_result - 1);
			if (!rs_tup)
				continue;
			if (!HeapTupleSatisfiesVisibility(&rs_tup->htup,
											  snapshot,
											  InvalidBuffer))
				continue;
			ExecStoreTuple(&rs_tup->htup, slot, InvalidBuffer, false);
		}
		else if (StromTagIs(sobject, TCacheColumnStore))
		{
			tcache_head			*tc_head = gss->tc_head;
			tcache_column_store *tcs = (tcache_column_store *)sobject;
			Form_pg_attribute	attr;
			int		i, j, k = i_result - 1;

			Assert(tc_head != NULL);

			if (!gss->hybrid_scan)
			{
				HeapTupleData		htup;

				/* A dummy HeapTuple to check visibility */
				htup.t_len = tcs->theads[k].t_hoff;
				htup.t_self = tcs->ctids[k];
				htup.t_tableOid = RelationGetRelid(gss->scan_rel);
				htup.t_data = &tcs->theads[k];

				if (!HeapTupleSatisfiesVisibility(&htup,
												  snapshot,
												  InvalidBuffer))
					continue;

				/* OK, put it on the result slot */
				slot = ExecStoreAllNullTuple(slot);
				for (i=0; i < tc_head->ncols; i++)
				{
					j = tc_head->i_cached[i];
					attr = tc_head->tupdesc->attrs[j];

					if (!attr->attnotnull &&
						att_isnull(k, tcs->cdata[i].isnull))
						continue;

					if (attr->attlen > 0)
					{
						slot->tts_values[j]
							= fetch_att(tcs->cdata[i].values +
										attr->attlen * k,
										attr->attbyval,
										attr->attlen);
					}
					else
					{
						cl_uint	   *cs_offset
							= (cl_uint *)tcs->cdata[i].values;

						Assert(cs_offset[k] > 0);
						Assert(tcs->cdata[i].toast != NULL);
						slot->tts_values[j]
							= PointerGetDatum((char *)tcs->cdata[i].toast +
											  cs_offset[k]);
					}
					slot->tts_isnull[j] = false;
				}
			}
			else
			{
				/*
				 * In case of hybrid-scan, we fetch heap buffer according
				 * to the item-pointer
				 */
				Buffer		buffer = InvalidBuffer;

				gss->hybrid_htup.t_self = tcs->ctids[k];
				if (!heap_fetch(gss->scan_rel,
								gss->cps.ps.state->es_snapshot,
								&gss->hybrid_htup,
								&buffer,
								false,
								NULL))
					continue;
				ExecStoreTuple(&gss->hybrid_htup, slot, buffer, false);

				/*
				 * At this point we have an extra pin on the buffer, because
				 * ExecStoreTuple incremented the pin count.
				 * Drop our local pin
				 */
				ReleaseBuffer(buffer);
			}
		}
		else
			elog(ERROR, "unexpected data store tag in gpuscan: %d",
				 (int)sobject->stag);

		if (do_recheck)
		{
			ExprContext *econtext = gss->cps.ps.ps_ExprContext;

			Assert(gss->dev_quals != NULL);
			econtext->ecxt_scantuple = gss->scan_slot;
			if (!ExecQual(gss->dev_quals, econtext, false))
				continue;
		}
		return true;
	}
	return false;
}

static TupleTableSlot *
gpuscan_fetch_tuple(CustomPlanState *node)
{
	GpuScanState   *gss = (GpuScanState *) node;
	TupleTableSlot *slot = gss->scan_slot;

	ExecClearTuple(slot);

	while (!gss->curr_chunk || !gpuscan_next_tuple(gss, slot))
	{
		pgstrom_message	   *msg;
		pgstrom_gpuscan	   *gpuscan;

		/*
		 * Release the current gpuscan chunk being already scanned
		 */
		if (gss->curr_chunk)
		{
			pgstrom_message	   *msg = &gss->curr_chunk->msg;

			if (msg->pfm.enabled)
				pgstrom_perfmon_add(&gss->pfm, &msg->pfm);
			Assert(msg->refcnt == 1);
			pgstrom_untrack_object(&msg->sobj);
			msg->cb_release(msg);
			gss->curr_chunk = NULL;
			gss->curr_index = 0;
		}

		/*
		 * In case when no device code will be executed, we have no
		 * message queue for asynchronous execution because of senseless.
		 * We handle the job synchronously, but may make sense if tcache
		 * reduces disk accesses.
		 */
		if (gss->dprog_key == 0)
		{
			gss->curr_chunk = pgstrom_load_gpuscan(gss);
			gss->curr_index = 0;
			if (gss->curr_chunk)
				continue;
			break;
		}
		Assert(gss->mqueue != NULL);

		/*
		 * Dequeue the current gpuscan chunks being already processed
		 */
		while ((msg = pgstrom_try_dequeue_message(gss->mqueue)) != NULL)
		{
			Assert(gss->num_running > 0);
			gss->num_running--;
			dlist_push_head(&gss->ready_chunks, &msg->chain);
		}

		/*
		 * Try to keep number of gpuscan chunks being asynchronously executed
		 * larger than minimum multiplicity, unless it does not exceed
		 * maximum one and OpenCL server does not return a new response.
		 */
		while ((gss->scan_desc || gss->tc_scan) &&
			   gss->num_running <= pgstrom_max_async_chunks)
		{
			pgstrom_gpuscan	*gpuscan = pgstrom_load_gpuscan(gss);

			if (!gpuscan)
				break;	/* scan reached end of the relation */

			if (!pgstrom_enqueue_message(&gpuscan->msg))
			{
				gpuscan->msg.cb_release(&gpuscan->msg);
				elog(ERROR, "failed to enqueue pgstrom_gpuscan message");
			}
			gss->num_running++;

			if (gss->num_running > pgstrom_min_async_chunks &&
				(msg = pgstrom_try_dequeue_message(gss->mqueue)) != NULL)
			{
				gss->num_running--;
				dlist_push_head(&gss->ready_chunks, &msg->chain);
				break;
			}
		}

		/*
		 * Wait for server's response if no available chunks were replied.
		 */
		if (dlist_is_empty(&gss->ready_chunks))
		{
			/* OK, no more chunks to be scanned */
			if (gss->num_running == 0)
				break;

			msg = pgstrom_dequeue_message(gss->mqueue);
			if (!msg)
				elog(ERROR, "message queue wait timeout");
			gss->num_running--;
			dlist_push_head(&gss->ready_chunks, &msg->chain);
		}

		/*
		 * Picks up next available chunks if any
		 */
		Assert(!dlist_is_empty(&gss->ready_chunks));
		gpuscan = dlist_container(pgstrom_gpuscan, msg.chain,
								dlist_pop_head_node(&gss->ready_chunks));
		Assert(StromTagIs(gpuscan, GpuScan));

		/*
         * Raise an error, if chunk-level error was reported.
         */
		if (gpuscan->msg.errcode != StromError_Success)
		{
			if (gpuscan->msg.errcode == CL_BUILD_PROGRAM_FAILURE)
			{
				const char *buildlog
					= pgstrom_get_devprog_errmsg(gpuscan->dprog_key);
				const char *kern_source
					= ((GpuScanPlan *)gss->cps.ps.plan)->kern_source;

				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("PG-Strom: OpenCL execution error (%s)\n%s",
								pgstrom_strerror(gpuscan->msg.errcode),
								kern_source),
						 errdetail("%s", buildlog)));
			}
			else
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("PG-Strom: OpenCL execution error (%s)",
								pgstrom_strerror(gpuscan->msg.errcode))));
			}
		}
		gss->curr_chunk = gpuscan;
		gss->curr_index = 0;
	}
	return slot;
}

static TupleTableSlot *
gpuscan_exec(CustomPlanState *node)
{
	/* overall logic were copied from ExecScan */
	ExprContext	   *econtext = node->ps.ps_ExprContext;
	List		   *qual = node->ps.qual;
	ProjectionInfo *projInfo = node->ps.ps_ProjInfo;
	ExprDoneCond	isDone;
	TupleTableSlot *resultSlot;

	/*
	 * If we have neither a qual to check nor a projection to do, just skip
	 * all the overhead and return the raw scan tuple.
	 */
	if (!qual && !projInfo)
	{
		ResetExprContext(econtext);
		return gpuscan_fetch_tuple(node);
	}

	/*
	 * Check to see if we're still projecting out tuples from a previous scan
	 * tuple (because there is a function-returning-set in the projection
	 * expressions).  If so, try to project another one.
	 */
	if (node->ps.ps_TupFromTlist)
	{
		Assert(projInfo);		/* can't get here if not projecting */
		resultSlot = ExecProject(projInfo, &isDone);
		if (isDone == ExprMultipleResult)
			return resultSlot;
		/* Done with that source tuple... */
		node->ps.ps_TupFromTlist = false;
	}

	/*
	 * Reset per-tuple memory context to free any expression evaluation
	 * storage allocated in the previous tuple cycle.  Note this can't happen
	 * until we're done projecting out tuples from a scan tuple.
	 */
	ResetExprContext(econtext);

	/*
	 * get a tuple from the access method.  Loop until we obtain a tuple that
	 * passes the qualification.
	 */
	for (;;)
	{
		TupleTableSlot *slot;

		CHECK_FOR_INTERRUPTS();

		slot = gpuscan_fetch_tuple(node);

		/*
		 * if the slot returned by the accessMtd contains NULL, then it means
		 * there is nothing more to scan so we just return an empty slot,
		 * being careful to use the projection result slot so it has correct
		 * tupleDesc.
		 */
		if (TupIsNull(slot))
		{
			if (projInfo)
				return ExecClearTuple(projInfo->pi_slot);
			else
				return slot;
		}

		/*
		 * place the current tuple into the expr context
		 */
		econtext->ecxt_scantuple = slot;

		/*
		 * check that the current tuple satisfies the qual-clause
		 *
		 * check for non-nil qual here to avoid a function call to ExecQual()
		 * when the qual is nil ... saves only a few cycles, but they add up
		 * ...
		 */
		if (!qual || ExecQual(qual, econtext, false))
		{
			/*
			 * Found a satisfactory scan tuple.
			 */
			if (projInfo)
			{
				/*
				 * Form a projection tuple, store it in the result tuple slot
				 * and return it --- unless we find we can project no tuples
				 * from this scan tuple, in which case continue scan.
				 */
				resultSlot = ExecProject(projInfo, &isDone);
				if (isDone != ExprEndResult)
				{
					node->ps.ps_TupFromTlist = (isDone == ExprMultipleResult);
					return resultSlot;
				}
			}
			else
			{
				/*
				 * Here, we aren't projecting, so just return scan tuple.
				 */
				return slot;
			}
		}
		else
			InstrCountFiltered1(node, 1);

		/*
		 * Tuple fails qual, so free per-tuple memory and try again.
		 */
		ResetExprContext(econtext);
	}
}
static Node *
gpuscan_exec_multi(CustomPlanState *node)
{
	elog(ERROR, "not implemented yet");
}

static void
gpuscan_end(CustomPlanState *node)
{
	GpuScanState   *gss = (GpuScanState *)node;

	if (gss->curr_chunk)
	{
		pgstrom_gpuscan *gpuscan = gss->curr_chunk;

		pgstrom_untrack_object(&gpuscan->msg.sobj);
		gpuscan->msg.cb_release(&gpuscan->msg);
		gss->curr_chunk = NULL;
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

	if (gss->tc_scan)
	{
		/*
		 * Usually, tc_scan should be already released during scan.
		 * If not, it implies this scan didn't reach to the relation end.
		 * tcache_end_scan() will release all the tc_node recursively,
		 * if tcache is not ready.
		 */
		gss->pfm.time_tcache_build = gss->tc_scan->time_tcache_build;
		tcache_end_scan(gss->tc_scan);
	}

	if (gss->tc_head)
	{
		tcache_head	*tc_head = gss->tc_head;

		pgstrom_untrack_object(&tc_head->sobj);

		tcache_put_tchead(tc_head);
	}

	/*
	 * Free the exprcontext
	 */
	ExecFreeExprContext(&gss->cps.ps);

	/*
	 * clean out the tuple table
	 */
	ExecClearTuple(gss->cps.ps.ps_ResultTupleSlot);
	ExecClearTuple(gss->scan_slot);

	/*
	 * close heap scan and relation
	 */
	if (gss->scan_desc)
		heap_endscan(gss->scan_desc);
	heap_close(gss->scan_rel, NoLock);
}

static void
gpuscan_rescan(CustomPlanState *node)
{
	elog(ERROR, "not implemented yet");
}

static void
gpuscan_explain_rel(CustomPlanState *node, ExplainState *es)
{
	GpuScanState   *gss = (GpuScanState *)node;
	GpuScanPlan	   *gsplan = (GpuScanPlan *)gss->cps.ps.plan;
	Relation		rel = gss->scan_rel;
	RangeTblEntry  *rte;
	char		   *refname;
	char		   *objectname;
	char		   *namespace = NULL;

	rte = rt_fetch(gsplan->scanrelid, es->rtable);
	refname = (char *) list_nth(es->rtable_names,
								gsplan->scanrelid - 1);
	if (refname == NULL)
		refname = rte->eref->aliasname;

	Assert(rte->rtekind == RTE_RELATION);
	objectname = RelationGetRelationName(rel);
	if (es->verbose)
		namespace = get_namespace_name(RelationGetNamespace(rel));

	if (es->format == EXPLAIN_FORMAT_TEXT)
	{
		appendStringInfoString(es->str, " on");
		if (namespace != NULL)
			appendStringInfo(es->str, " %s.%s",
							 quote_identifier(namespace),
							 quote_identifier(objectname));
		else
			appendStringInfo(es->str, " %s",
							 quote_identifier(objectname));
	}
	else
	{
		ExplainPropertyText("Relation Name", objectname, es);
		if (namespace != NULL)
			ExplainPropertyText("Schema", namespace, es);
		ExplainPropertyText("Alias", refname, es);
	}
}

static void
gpuscan_explain(CustomPlanState *node, List *ancestors, ExplainState *es)
{
	GpuScanState   *gss = (GpuScanState *) node;
	GpuScanPlan	   *gsplan = (GpuScanPlan *) gss->cps.ps.plan;
	Relation		rel = gss->scan_rel;
	Bitmapset	   *tempset;
	Oid				relid = RelationGetRelid(rel);
	AttrNumber		anum;
	bool			is_first;
	StringInfoData	str;

	initStringInfo(&str);
	tempset = bms_copy(gsplan->host_attnums);
	is_first = true;
	while ((anum = bms_first_member(tempset)) >= 0)
	{
		anum += FirstLowInvalidHeapAttributeNumber;

		appendStringInfo(&str, "%s%s",
						 is_first ? "" : ", ",
						 get_relid_attribute_name(relid, anum));
		is_first = false;
	}
	bms_free(tempset);
	ExplainPropertyText("Host References", str.data, es);

	resetStringInfo(&str);
	tempset = bms_copy(gsplan->dev_attnums);
	is_first = true;
	while ((anum = bms_first_member(tempset)) >= 0)
	{
		anum += FirstLowInvalidHeapAttributeNumber;

		appendStringInfo(&str, "%s%s",
						 is_first ? "" : ", ",
						 get_relid_attribute_name(relid, anum));
		is_first = false;
	}
	bms_free(tempset);
	ExplainPropertyText("Device References", str.data, es);

	if (gsplan->cplan.plan.qual != NIL)
	{
		show_scan_qual(gsplan->cplan.plan.qual,
					   "Filter", &gss->cps.ps, ancestors, es);
		show_instrumentation_count("Rows Removed by Filter",
								   1, &gss->cps.ps, es);
	}
	if (gsplan->dev_clauses != NIL)
	{
		show_scan_qual(gsplan->dev_clauses,
					   "Device Filter", &gss->cps.ps, ancestors, es);
		show_instrumentation_count("Rows Removed by Device Fileter",
								   2, &gss->cps.ps, es);
	}
	show_device_kernel(gss->dprog_key, es);

	if (es->analyze && gss->pfm.enabled)
		pgstrom_perfmon_explain(&gss->pfm, es);
}

static Bitmapset *
gpuscan_get_relids(CustomPlanState *node)
{
	GpuScanPlan	   *gsp = (GpuScanPlan *)node->ps.plan;

	return bms_make_singleton(gsp->scanrelid);
}

static void
gpuscan_textout_plan(StringInfo str, const CustomPlan *node)
{
	GpuScanPlan	   *plannode = (GpuScanPlan *)node;
	char		   *temp;

	appendStringInfo(str, " :scanrelid %u", plannode->scanrelid);

	appendStringInfo(str, " :kern_source ");
	_outToken(str, plannode->kern_source);

	appendStringInfo(str, " :extra_flags %u", plannode->scanrelid);

	temp = nodeToString(plannode->used_params);
	appendStringInfo(str, " :used_params %s", temp);
	pfree(temp);

	temp = nodeToString(plannode->used_vars);
	appendStringInfo(str, " :used_vars %s", temp);
	pfree(temp);

	temp = nodeToString(plannode->dev_clauses);
	appendStringInfo(str, " :dev_clauses %s", temp);
	pfree(temp);

	appendStringInfo(str, " :dev_attnums ");
	_outBitmapset(str, plannode->dev_attnums);

	appendStringInfo(str, " :host_attnums ");
	_outBitmapset(str, plannode->host_attnums);
}

static CustomPlan *
gpuscan_copy_plan(const CustomPlan *from)
{
	GpuScanPlan	   *oldnode = (GpuScanPlan *)from;
	GpuScanPlan	   *newnode = palloc(sizeof(GpuScanPlan));

	CopyCustomPlanCommon((Node *)from, (Node *)newnode);
	newnode->scanrelid = oldnode->scanrelid;
	newnode->used_params = copyObject(oldnode->used_params);
	newnode->used_vars = copyObject(oldnode->used_vars);
	newnode->extra_flags = oldnode->extra_flags;
	newnode->dev_clauses = oldnode->dev_clauses;
	newnode->dev_attnums = bms_copy(oldnode->dev_attnums);
	newnode->host_attnums = bms_copy(oldnode->host_attnums);

	return &newnode->cplan;
}

void
pgstrom_init_gpuscan(void)
{
	/* setup path methods */
	gpuscan_path_methods.CustomName			= "GpuScan";
	gpuscan_path_methods.CreateCustomPlan	= gpuscan_create_plan;
	gpuscan_path_methods.TextOutCustomPath	= gpuscan_textout_path;

	/* setup plan methods */
	gpuscan_plan_methods.CustomName			= "GpuScan";
	gpuscan_plan_methods.SetCustomPlanRef	= gpuscan_set_plan_ref;
	gpuscan_plan_methods.SupportBackwardScan= NULL;
	gpuscan_plan_methods.FinalizeCustomPlan	= gpuscan_finalize_plan;
	gpuscan_plan_methods.BeginCustomPlan	= gpuscan_begin;
	gpuscan_plan_methods.ExecCustomPlan		= gpuscan_exec;
	gpuscan_plan_methods.MultiExecCustomPlan= gpuscan_exec_multi;
	gpuscan_plan_methods.EndCustomPlan		= gpuscan_end;
	gpuscan_plan_methods.ReScanCustomPlan	= gpuscan_rescan;
	gpuscan_plan_methods.ExplainCustomPlanTargetRel	= gpuscan_explain_rel;
	gpuscan_plan_methods.ExplainCustomPlan	= gpuscan_explain;
	gpuscan_plan_methods.GetRelidsCustomPlan= gpuscan_get_relids;
	gpuscan_plan_methods.GetSpecialCustomVar= NULL;
	gpuscan_plan_methods.TextOutCustomPlan	= gpuscan_textout_plan;
	gpuscan_plan_methods.CopyCustomPlan		= gpuscan_copy_plan;

	/* hook registration */
	add_scan_path_next = add_scan_path_hook;
	add_scan_path_hook = gpuscan_add_scan_path;
}

/*
 * GpuScan message handler
 * ------------------------------------------------------------
 * Note that below routines are executed in the context of OpenCL
 * intermediation server, thus, usual PostgreSQL internal APIs are
 * not available, and need to pay attention that routines work in
 * another process's address space.
 */


typedef struct
{
	pgstrom_message	*msg;
	cl_program		program;
	cl_kernel		kernel;
	cl_mem			m_gpuscan;
	cl_mem			m_rstore;
	cl_mem			m_cstore;
	cl_int			ev_index;
	cl_event		events[10];
} clstate_gpuscan_row;

static void
clserv_respond_gpuscan_row(cl_event event, cl_int ev_status, void *private)
{
	clstate_gpuscan_row	*clgsr = private;
	pgstrom_gpuscan		*gpuscan = (pgstrom_gpuscan *)clgsr->msg;
	kern_resultbuf		*kresult = KERN_GPUSCAN_RESULTBUF(&gpuscan->kern);

	/* put error code */
	if (ev_status != CL_COMPLETE)
	{
		clserv_log("unexpected CL_EVENT_COMMAND_EXECUTION_STATUS: %d",
				   ev_status);
		gpuscan->msg.errcode = StromError_OpenCLInternal;
	}
	else
	{
		gpuscan->msg.errcode = kresult->errcode;
	}
	/* collect performance statistics */
	if (gpuscan->msg.pfm.enabled)
	{
		cl_ulong	dma_send_begin;
		cl_ulong	dma_send_end;
		cl_ulong	kern_exec_begin;
		cl_ulong	kern_exec_end;
		cl_ulong	dma_recv_begin;
		cl_ulong	dma_recv_end;
		cl_ulong	temp;
		cl_int		i, rc;

		for (i=0; i < 3; i++)
		{
			rc = clGetEventProfilingInfo(clgsr->events[i],
										 CL_PROFILING_COMMAND_START,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			if (i==0 || dma_send_begin > temp)
				dma_send_begin = temp;

			rc = clGetEventProfilingInfo(clgsr->events[i],
										 CL_PROFILING_COMMAND_END,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			if (i==0 || dma_send_end < temp)
				dma_send_end = temp;
		}
		rc = clGetEventProfilingInfo(clgsr->events[clgsr->ev_index - 2],
									 CL_PROFILING_COMMAND_START,
									 sizeof(cl_ulong),
									 &kern_exec_begin,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;

		rc = clGetEventProfilingInfo(clgsr->events[clgsr->ev_index - 2],
									 CL_PROFILING_COMMAND_END,
									 sizeof(cl_ulong),
									 &kern_exec_end,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;

		rc = clGetEventProfilingInfo(clgsr->events[clgsr->ev_index - 1],
									 CL_PROFILING_COMMAND_START,
									 sizeof(cl_ulong),
									 &dma_recv_begin,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;

		rc = clGetEventProfilingInfo(clgsr->events[clgsr->ev_index - 1],
									 CL_PROFILING_COMMAND_END,
									 sizeof(cl_ulong),
									 &dma_recv_end,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;

		gpuscan->msg.pfm.time_dma_send
			+= (dma_send_end - dma_send_begin) / 1000;
		gpuscan->msg.pfm.time_kern_exec
			+= (kern_exec_end - kern_exec_begin) / 1000;
		gpuscan->msg.pfm.time_dma_recv
			+= (dma_recv_end - dma_recv_begin) / 1000;

	skip_perfmon:
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clGetEventProfilingInfo (%s)",
					   opencl_strerror(rc));
			gpuscan->msg.pfm.enabled = false;	/* turn off profiling */
		}
	}
	/* dump debug messages */
	pgstrom_dump_kernel_debug(LOG, KERN_GPUSCAN_RESULTBUF(&gpuscan->kern));

	/* release opencl objects */
	while (clgsr->ev_index > 0)
		clReleaseEvent(clgsr->events[--clgsr->ev_index]);
	clReleaseMemObject(clgsr->m_cstore);
	clReleaseMemObject(clgsr->m_rstore);
	clReleaseMemObject(clgsr->m_gpuscan);
	clReleaseKernel(clgsr->kernel);
	clReleaseProgram(clgsr->program);
	free(clgsr);

	/* respond to the backend side */
	pgstrom_reply_message(&gpuscan->msg);
}

static void
clserv_process_gpuscan_row(pgstrom_message *msg)
{
	pgstrom_gpuscan	   *gpuscan = (pgstrom_gpuscan *)msg;
	tcache_row_store   *trs;
	clstate_gpuscan_row *clgss;
	kern_parambuf	   *kparams;
	kern_resultbuf	   *kresults;
	kern_row_store	   *krs;
	kern_column_store  *kcs_head;
	cl_command_queue	kcmdq;
	cl_uint				nrows;
	cl_uint				i;
	cl_int				rc;
	Size				bytes_dma_send = 0;
	Size				bytes_dma_recv = 0;
	size_t				length;
	size_t				gwork_sz;
	size_t				lwork_sz;

	/* only one store shall be attached! */
	Assert(StromTagIs(gpuscan->rc_store, TCacheRowStore));
	trs = (tcache_row_store *)gpuscan->rc_store;

	/* state object of gpuscan with row-store */
	clgss = malloc(sizeof(clstate_gpuscan_row));
	if (!clgss)
	{
		rc = CL_OUT_OF_HOST_MEMORY;
		goto error0;
	}
	memset(clgss, 0, sizeof(clstate_gpuscan_row));
	clgss->msg = &gpuscan->msg;
	kresults = KERN_GPUSCAN_RESULTBUF(&gpuscan->kern);
	krs = &trs->kern;
	kcs_head = (kern_column_store *)kresults->results;
	nrows = krs->nrows;

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
	clgss->program = clserv_lookup_device_program(gpuscan->dprog_key,
												  &gpuscan->msg);
	if (!clgss->program)
	{
		free(clgss);
		return;		/* message is in waitq, retry it! */
	}
	if (clgss->program == BAD_OPENCL_PROGRAM)
	{
		rc = CL_BUILD_PROGRAM_FAILURE;
		goto error1;
	}

	/*
	 * In this case, we use a kernel for row-store; that internally
	 * translate row-format into column-format, then evaluate the
	 * supplied qualifier.
	 */
	clgss->kernel = clCreateKernel(clgss->program,
								   "gpuscan_qual_rs",
								   &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error2;
	}

	/*
	 * Choose a device to execute this kernel
	 */
	i = pgstrom_opencl_device_schedule(&gpuscan->msg);
	kcmdq = opencl_cmdq[i];

	/* and, compute an optimal workgroup-size of this kernel */
	if (!clserv_compute_workgroup_size(&gwork_sz, &lwork_sz,
									   clgss->kernel, i, nrows,
									   sizeof(cl_uint)))
		goto error3;

	/* allocation of device memory for kern_gpuscan argument */
	clgss->m_gpuscan = clCreateBuffer(opencl_context,
									  CL_MEM_READ_WRITE,
									  KERN_GPUSCAN_LENGTH(&gpuscan->kern),
									  NULL,
									  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error3;
	}

	/* allocation of device memory for kern_row_store argument */
	clgss->m_rstore = clCreateBuffer(opencl_context,
									 CL_MEM_READ_WRITE,
									 krs->length,
									 NULL,
									 &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error4;
	}

	/* allocation of device memory for kern_column_store argument */
	clgss->m_cstore = clCreateBuffer(opencl_context,
									 CL_MEM_READ_WRITE,
									 kcs_head->length,
									 NULL,
									 &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error5;
	}

	/*
	 * OK, all the device memory and kernel objects acquired.
	 * Let's prepare kernel invocation.
	 *
	 * The kernel call is:
	 *   __kernel void
	 *   gpuscan_qual_rs(__global kern_gpuscan *gpuscan,
	 *                   __global kern_row_store *krs,
	 *                   __global kern_column_store *kcs,
	 *                   __local void *local_workmem)
	 */
	rc = clSetKernelArg(clgss->kernel,
						0,	/* kern_gpuscan * */
						sizeof(cl_mem),
						&clgss->m_gpuscan);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error6;
	}

	rc = clSetKernelArg(clgss->kernel,
						1,	/* kern_row_store */
						sizeof(cl_mem),
						&clgss->m_rstore);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error6;
	}

	rc = clSetKernelArg(clgss->kernel,
						2,	/* kern_column_store */
						sizeof(cl_mem),
						&clgss->m_cstore);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error6;
	}

	rc = clSetKernelArg(clgss->kernel,
						3,	/* local_workmem */
						2 * sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error6;
	}

	/*
	 * OK, enqueue DMA transfer, kernel execution x2, then DMA writeback.
	 *
	 * (1) gpuscan, row_store and header portion of column_store shall be
	 *     copied to the device memory from the kernel
	 * (2) kernel shall be launched
	 * (3) kern_result shall be written back
	 */
	kparams = &gpuscan->kern.kparam;
	length = kparams->length + offsetof(kern_resultbuf, results[0]);

	rc = clEnqueueWriteBuffer(kcmdq,
							  clgss->m_gpuscan,
							  CL_FALSE,
							  0,
							  length,
							  &gpuscan->kern,
							  0,
							  NULL,
							  &clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error6;
	}
	clgss->ev_index++;
	bytes_dma_send += length;

	rc = clEnqueueWriteBuffer(kcmdq,
							  clgss->m_rstore,
							  CL_FALSE,
							  0,
							  krs->length,
							  krs,
							  0,
							  NULL,
							  &clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}
	clgss->ev_index++;
	bytes_dma_send += krs->length;

	rc = clEnqueueWriteBuffer(kcmdq,
							  clgss->m_cstore,
							  CL_FALSE,
							  0,
							  offsetof(kern_column_store,
									   colmeta[kcs_head->ncols]),
							  kcs_head,
							  0,
							  NULL,
							  &clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}
	clgss->ev_index++;
	bytes_dma_send += offsetof(kern_column_store, colmeta[kcs_head->ncols]);

	/*
	 * Kick gpuscan_qual_rs() call
	 */
	rc = clEnqueueNDRangeKernel(kcmdq,
								clgss->kernel,
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
		goto error_sync;
	}
	clgss->ev_index++;

	/*
	 * Write back the result-buffer
	 */
	rc = clEnqueueReadBuffer(kcmdq,
							 clgss->m_gpuscan,
							 CL_FALSE,
							 ((Size)KERN_GPUSCAN_RESULTBUF(&gpuscan->kern) -
							  (Size)(&gpuscan->kern)),
							 KERN_GPUSCAN_DMA_RECVLEN(&gpuscan->kern),
							 KERN_GPUSCAN_RESULTBUF(&gpuscan->kern),
							 1,
							 &clgss->events[clgss->ev_index - 1],
							 &clgss->events[clgss->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueReadBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}
	clgss->ev_index++;
	bytes_dma_recv += ((Size)KERN_GPUSCAN_RESULTBUF(&gpuscan->kern) -
					   (Size)(&gpuscan->kern));
	/* update performance counter */
	if (gpuscan->msg.pfm.enabled)
	{
		gpuscan->msg.pfm.bytes_dma_send = bytes_dma_send;
		gpuscan->msg.pfm.bytes_dma_recv = bytes_dma_recv;
		gpuscan->msg.pfm.num_dma_send = 3;
		gpuscan->msg.pfm.num_dma_recv = 1;
	}

	/*
	 * Last, registers a callback routine that replies the message
	 * to the backend
	 */
	rc = clSetEventCallback(clgss->events[clgss->ev_index - 1],
							CL_COMPLETE,
							clserv_respond_gpuscan_row,
							clgss);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetEventCallback: %s", opencl_strerror(rc));
		goto error_sync;
	}
	return;

error_sync:
	/*
	 * Once DMA requests were enqueued, we need to synchronize completion
	 * of a series of jobs to avoid unexpected memory destruction if device
	 * write back calculation result onto the region already released.
	 */
	clWaitForEvents(clgss->ev_index, clgss->events);
error6:
	clReleaseMemObject(clgss->m_cstore);
error5:
	clReleaseMemObject(clgss->m_rstore);
error4:
	clReleaseMemObject(clgss->m_gpuscan);
error3:
	clReleaseKernel(clgss->kernel);
error2:
	clReleaseProgram(clgss->program);
error1:
	free(clgss);
error0:
	gpuscan->msg.errcode = rc;
	pgstrom_reply_message(&gpuscan->msg);
}









typedef struct
{
	pgstrom_message	*msg;
	cl_program		program;
	cl_kernel		kernel;
	cl_mem			m_gpuscan;
	cl_mem			m_cstore;
	cl_mem			m_toast;
	cl_int			ev_index;
	cl_event		events[20];
} clstate_gpuscan_column;

static void
clserv_respond_gpuscan_column(cl_event event, cl_int ev_status, void *private)
{
	clstate_gpuscan_column *clgsc = private;
	pgstrom_gpuscan		   *gpuscan = (pgstrom_gpuscan *)clgsc->msg;
	kern_resultbuf		   *kresult = KERN_GPUSCAN_RESULTBUF(&gpuscan->kern);
	cl_int	status, rc;

	/* put error code */
	if (ev_status != CL_COMPLETE)
	{
		clserv_log("unexpected CL_EVENT_COMMAND_EXECUTION_STATUS: %d",
				   ev_status);
		gpuscan->msg.errcode = StromError_OpenCLInternal;
	}
	else
	{
		gpuscan->msg.errcode = kresult->errcode;
	}

	/* collect performance statistics */
	if (gpuscan->msg.pfm.enabled)
	{
		cl_ulong	dma_send_begin;
		cl_ulong	dma_send_end;
		cl_ulong	kern_exec_begin;
		cl_ulong	kern_exec_end;
		cl_ulong	dma_recv_begin;
		cl_ulong	dma_recv_end;
		cl_ulong	temp;
		cl_int		i, n, rc;

		n = clgsc->ev_index - 2;
		for (i=0; i < n; i++)
		{
			rc = clGetEventProfilingInfo(clgsc->events[i],
										 CL_PROFILING_COMMAND_START,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			if (i==0 || dma_send_begin > temp)
				dma_send_begin = temp;

			rc = clGetEventProfilingInfo(clgsc->events[i],
										 CL_PROFILING_COMMAND_END,
										 sizeof(cl_ulong),
										 &temp,
										 NULL);
			if (rc != CL_SUCCESS)
				goto skip_perfmon;
			if (i==0 || dma_send_end < temp)
				dma_send_end = temp;
		}

		rc = clGetEventProfilingInfo(clgsc->events[clgsc->ev_index - 2],
									 CL_PROFILING_COMMAND_START,
									 sizeof(cl_ulong),
									 &kern_exec_begin,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;

		rc = clGetEventProfilingInfo(clgsc->events[clgsc->ev_index - 2],
									 CL_PROFILING_COMMAND_END,
									 sizeof(cl_ulong),
									 &kern_exec_end,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;

		rc = clGetEventProfilingInfo(clgsc->events[clgsc->ev_index - 1],
									 CL_PROFILING_COMMAND_START,
									 sizeof(cl_ulong),
									 &dma_recv_begin,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;

		rc = clGetEventProfilingInfo(clgsc->events[clgsc->ev_index - 1],
									 CL_PROFILING_COMMAND_END,
									 sizeof(cl_ulong),
									 &dma_recv_end,
									 NULL);
		if (rc != CL_SUCCESS)
			goto skip_perfmon;

		gpuscan->msg.pfm.time_dma_send
			+= (dma_send_end - dma_send_begin) / 1000;
		gpuscan->msg.pfm.time_kern_exec
			+= (kern_exec_end - kern_exec_begin) / 1000;
		gpuscan->msg.pfm.time_dma_recv
			+= (dma_recv_end - dma_recv_begin) / 1000;

	skip_perfmon:
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clGetEventProfilingInfo (%s)",
					   opencl_strerror(rc));
			gpuscan->msg.pfm.enabled = false;	/* turn off profiling */
		}
	}

	rc = clGetEventInfo(clgsc->events[clgsc->ev_index - 2],
						CL_EVENT_COMMAND_EXECUTION_STATUS,
						sizeof(cl_int),
						&status,
						NULL);
	Assert(rc == CL_SUCCESS);

	/* dump debug messages */
	pgstrom_dump_kernel_debug(LOG, KERN_GPUSCAN_RESULTBUF(&gpuscan->kern));

	/* release opencl objects */
	while (clgsc->ev_index > 0)
		clReleaseEvent(clgsc->events[--clgsc->ev_index]);
	clReleaseMemObject(clgsc->m_toast);
	clReleaseMemObject(clgsc->m_cstore);
	clReleaseMemObject(clgsc->m_gpuscan);
	clReleaseKernel(clgsc->kernel);
	clReleaseProgram(clgsc->program);
	free(clgsc);

	/* respond to the backend side */
	pgstrom_reply_message(&gpuscan->msg);
}

static void
clserv_process_gpuscan_column(pgstrom_message *msg)
{
	pgstrom_gpuscan	   *gpuscan = (pgstrom_gpuscan *)msg;
	tcache_column_store *tcs;
	clstate_gpuscan_column *clgsc;
	kern_parambuf	   *kparams;
	kern_resultbuf	   *kresults;
	kern_column_store  *kcs_head;
	kern_toastbuf	   *ktoast;
	cl_command_queue	kcmdq;
	cl_uint			   *i_refcols;
	cl_uint				ncols;
	cl_uint				nrooms;
	cl_uint				i, j;
	cl_int				rc;
	Size				bytes_dma_send = 0;
	Size				bytes_dma_recv = 0;
	Size				length;
	Size				offset;
	size_t				gwork_sz;
	size_t				lwork_sz;
	bool				with_toast;

	/* only column-store should be attached! */
	Assert(StromTagIs(gpuscan->rc_store, TCacheColumnStore));
	tcs = (tcache_column_store *)gpuscan->rc_store;

	/* state object of gpuscan with column-store */
	clgsc = malloc(sizeof(clstate_gpuscan_column));
	if (!clgsc)
	{
		rc = CL_OUT_OF_HOST_MEMORY;
		goto error0;
	}
	memset(clgsc, 0, sizeof(clstate_gpuscan_column));
	clgsc->msg = &gpuscan->msg;
	kresults = KERN_GPUSCAN_RESULTBUF(&gpuscan->kern);
	kcs_head = (kern_column_store *)kresults->results;
	ncols = kcs_head->ncols;
	nrooms = kcs_head->nrooms;
	ktoast = (kern_toastbuf *)&kcs_head->colmeta[ncols];
	i_refcols = (cl_uint *)&ktoast->coldir[ncols];

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
	clgsc->program = clserv_lookup_device_program(gpuscan->dprog_key,
												  &gpuscan->msg);
	if (!clgsc->program)
	{
		free(clgsc);
		return;		/* message is in waitq, retry it! */
	}
	if (clgsc->program == BAD_OPENCL_PROGRAM)
	{
		rc = CL_BUILD_PROGRAM_FAILURE;
		goto error1;
	}

	/*
	 * In this case, we use a kernel for column-store; that uses
	 * the supplied column-store as is, for evaluation of qualifier.
	 */
	clgsc->kernel = clCreateKernel(clgsc->program,
								   "gpuscan_qual_cs",
								   &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error2;
	}

	/*
	 * Choose a device to execute this kernel
	 */
	i = pgstrom_opencl_device_schedule(&gpuscan->msg);
	kcmdq = opencl_cmdq[i];

	/* and, compute an optimal workgroup-size of this kernel */
	if (!clserv_compute_workgroup_size(&gwork_sz, &lwork_sz,
									   clgsc->kernel, i, nrooms,
									   sizeof(cl_uint)))
		goto error3;

	/* allocation of device memory for kern_gpuscan argument */
	clgsc->m_gpuscan = clCreateBuffer(opencl_context,
									  CL_MEM_READ_WRITE,
									  KERN_GPUSCAN_LENGTH(&gpuscan->kern),
									  NULL,
									  &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error3;
	}

	/* allocation of device memory for kern_column_store argument */
	clgsc->m_cstore = clCreateBuffer(opencl_context,
									 CL_MEM_READ_WRITE,
									 kcs_head->length,
									 NULL,
									 &rc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
		goto error4;
	}

	/* allocation of device memory for kern_toastbuf argument */
	with_toast = false;
	length = STROMALIGN(offsetof(kern_toastbuf, coldir[ncols]));
	ktoast->length = TOASTBUF_MAGIC;
	ktoast->ncols = ncols;
	for (i=0; i < ncols; i++)
	{
		tcache_toastbuf *tc_toast;

		j = i_refcols[i];
		tc_toast = tcs->cdata[j].toast;
		if (tc_toast)
		{
			ktoast->coldir[i] = length;
			length += STROMALIGN(tc_toast->tbuf_usage);
			with_toast = true;
		}
		else
			ktoast->coldir[i] = (cl_uint)-1;
	}

	if (with_toast)
	{
		clgsc->m_toast = clCreateBuffer(opencl_context,
										CL_MEM_READ_WRITE,
										length,
										NULL,
										&rc);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clCreateBuffer: %s", opencl_strerror(rc));
			goto error5;
		}
	}
	else
	{
		clgsc->m_toast = NULL;
	}

    /*
     * OK, all the device memory and kernel objects acquired.
     * Let's prepare kernel invocation.
     *
     * The kernel call is:
	 *
	 *   __kernel void
	 *   gpuscan_qual_cs(__global kern_gpuscan *kgscan,
	 *                   __global kern_column_store *kcs,
	 *                   __global kern_toastbuf *toast,
	 *                   __local void *local_workmem)
	 */
	rc = clSetKernelArg(clgsc->kernel,
						0,	/* kern_gpuscan * */
						sizeof(cl_mem),
						&clgsc->m_gpuscan);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error6;
	}

	rc = clSetKernelArg(clgsc->kernel,
						1,	/* kern_column_store */
						sizeof(cl_mem),
						&clgsc->m_cstore);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error6;
	}

	rc = clSetKernelArg(clgsc->kernel,
						2,	/* kern_toastbuf */
						sizeof(cl_mem),
						&clgsc->m_toast);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error6;
	}

	rc = clSetKernelArg(clgsc->kernel,
						3,	/* local_workmem */
						2 * sizeof(cl_uint) * lwork_sz,
						NULL);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetKernelArg: %s", opencl_strerror(rc));
		goto error6;
	}

	/*
	 * OK, enqueue DMA transfer, kernel execution, then DMA writeback.
	 *
	 * (1) gpuscan, row_store and header portion of column_store shall be
	 *     copied to the device memory from the kernel
	 * (2) kernel shall be launched
	 * (3) kern_result shall be written back
	 */
	kparams = &gpuscan->kern.kparam;
	length = kparams->length + offsetof(kern_resultbuf, results[0]);

	/* kern_parambuf and kern_resultbuf  */
	rc = clEnqueueWriteBuffer(kcmdq,
							  clgsc->m_gpuscan,
							  CL_FALSE,
							  0,
							  length,
							  &gpuscan->kern,
							  0,
							  NULL,
							  &clgsc->events[clgsc->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error6;
	}
	clgsc->ev_index++;
	bytes_dma_send += length;

	/* header of kern_column_store */
	rc = clEnqueueWriteBuffer(kcmdq,
							  clgsc->m_cstore,
							  CL_FALSE,
							  0,
							  offsetof(kern_column_store,
									   colmeta[ncols]),
							  kcs_head,
							  0,
							  NULL,
							  &clgsc->events[clgsc->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueWriteBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}
	clgsc->ev_index++;
	bytes_dma_send += offsetof(kern_column_store, colmeta[ncols]);

	/* header of toast buffer */
	if (clgsc->m_toast)
	{
		rc = clEnqueueWriteBuffer(kcmdq,
								  clgsc->m_toast,
								  CL_FALSE,
								  0,
								  offsetof(kern_toastbuf,
										   coldir[ncols]),
								  ktoast,
								  0,
								  NULL,
								  &clgsc->events[clgsc->ev_index]);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clEnqueueWriteBuffer: %s",
					   opencl_strerror(rc));
			goto error_sync;
		}
		clgsc->ev_index++;
		bytes_dma_send += offsetof(kern_toastbuf, coldir[ncols]);
	}

	for (i=0; i < ncols; i++)
	{
		j = i_refcols[i];

		offset = kcs_head->colmeta[i].cs_ofs;
		if (!kcs_head->colmeta[i].attnotnull)
		{
			Assert(tcs->cdata[j].isnull != NULL);
			length = STROMALIGN((nrooms + BITS_PER_BYTE - 1) / BITS_PER_BYTE);
			rc = clEnqueueWriteBuffer(kcmdq,
									  clgsc->m_cstore,
									  CL_FALSE,
									  offset,
									  length,
									  tcs->cdata[j].isnull,
									  0,
									  NULL,
									  &clgsc->events[clgsc->ev_index]);
			if (rc != CL_SUCCESS)
			{
				clserv_log("failed on clEnqueueWriteBuffer: %s",
						   opencl_strerror(rc));
				goto error_sync;
			}
			clgsc->ev_index++;
			bytes_dma_send += length;

			offset += length;
		}
		length = (kcs_head->colmeta[i].attlen > 0
				  ? kcs_head->colmeta[i].attlen
				  : sizeof(cl_uint)) * nrooms;
		rc = clEnqueueWriteBuffer(kcmdq,
								  clgsc->m_cstore,
								  CL_FALSE,
								  offset,
								  length,
								  tcs->cdata[j].values,
								  0,
								  NULL,
								  &clgsc->events[clgsc->ev_index]);
		if (rc != CL_SUCCESS)
		{
			clserv_log("failed on clEnqueueWriteBuffer: %s",
					   opencl_strerror(rc));
			goto error_sync;
		}
		clgsc->ev_index++;
		bytes_dma_send += length;

		if (tcs->cdata[j].toast)
		{
			Assert(kcs_head->colmeta[i].attlen < 0);
			Assert(clgsc->m_toast);
			rc = clEnqueueWriteBuffer(kcmdq,
									  clgsc->m_toast,
									  CL_FALSE,
									  ktoast->coldir[i],
									  tcs->cdata[j].toast->tbuf_usage,
									  tcs->cdata[j].toast,
									  0,
									  NULL,
									  &clgsc->events[clgsc->ev_index]);
			if (rc != CL_SUCCESS)
			{
				clserv_log("failed on clEnqueueWriteBuffer: %s",
						   opencl_strerror(rc));
				goto error_sync;
			}
			clgsc->ev_index++;
			bytes_dma_send += tcs->cdata[j].toast->tbuf_usage;
		}
	}

	/*
	 * Kick gpuscan_qual_cs() call
	 */
	rc = clEnqueueNDRangeKernel(kcmdq,
								clgsc->kernel,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								clgsc->ev_index,
								&clgsc->events[0],
								&clgsc->events[clgsc->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueNDRangeKernel: %s",
				   opencl_strerror(rc));
		goto error_sync;
	}
	clgsc->ev_index++;

	/*
	 * Write back the result-buffer
	 */
	rc = clEnqueueReadBuffer(kcmdq,
							 clgsc->m_gpuscan,
							 CL_FALSE,
							 ((Size)KERN_GPUSCAN_RESULTBUF(&gpuscan->kern) -
							  (Size)(&gpuscan->kern)),
							 KERN_GPUSCAN_DMA_RECVLEN(&gpuscan->kern),
							 KERN_GPUSCAN_RESULTBUF(&gpuscan->kern),
							 1,
							 &clgsc->events[clgsc->ev_index - 1],
							 &clgsc->events[clgsc->ev_index]);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clEnqueueReadBuffer: %s", opencl_strerror(rc));
		goto error_sync;
	}
	clgsc->ev_index++;
	bytes_dma_recv += KERN_GPUSCAN_DMA_RECVLEN(&gpuscan->kern);

	/* update performance counter */
	if (gpuscan->msg.pfm.enabled)
	{
		gpuscan->msg.pfm.bytes_dma_send = bytes_dma_send;
		gpuscan->msg.pfm.bytes_dma_recv = bytes_dma_recv;
		gpuscan->msg.pfm.num_dma_send = clgsc->ev_index - 2;
		gpuscan->msg.pfm.num_dma_recv = 1;
	}

	/*
	 * Last, registers a callback routine that replies the message
	 * to the backend
	 */
	rc = clSetEventCallback(clgsc->events[clgsc->ev_index - 1],
							CL_COMPLETE,
							clserv_respond_gpuscan_column,
							clgsc);
	if (rc != CL_SUCCESS)
	{
		clserv_log("failed on clSetEventCallback: %s", opencl_strerror(rc));
		goto error_sync;
	}
	return;

error_sync:
	/*
	 * Once DMA requests were enqueued, we need to synchronize completion
	 * of a series of jobs to avoid unexpected memory destruction if device
	 * write back calculation result onto the region already released.
	 */
	clWaitForEvents(clgsc->ev_index, clgsc->events);
error6:
	clReleaseMemObject(clgsc->m_toast);
error5:
	clReleaseMemObject(clgsc->m_cstore);
error4:
	clReleaseMemObject(clgsc->m_gpuscan);
error3:
	clReleaseKernel(clgsc->kernel);
error2:
	clReleaseProgram(clgsc->program);
error1:
	free(clgsc);
error0:
	gpuscan->msg.errcode = rc;
	pgstrom_reply_message(&gpuscan->msg);
}

/*
 * clserv_put_gpuscan
 *
 * Callback handler when reference counter of pgstrom_gpuscan object
 * reached to zero, due to pgstrom_put_message.
 * It also unlinks associated device program and release row-store.
 * Also note that this routine can be called under the OpenCL server
 * context.
 */
static void
clserv_put_gpuscan(pgstrom_message *msg)
{
	pgstrom_gpuscan	   *gpuscan = (pgstrom_gpuscan *)msg;

	/* unlink message queue (if exist) */
	if (msg->respq)
		pgstrom_put_queue(msg->respq);

	/* unlink device program (if exist) */
	if (gpuscan->dprog_key != 0)
		pgstrom_put_devprog_key(gpuscan->dprog_key);

	/* release row-store */
	if (gpuscan->rc_store)
	{
		Assert(StromTagIs(gpuscan->rc_store, TCacheRowStore) ||
			   StromTagIs(gpuscan->rc_store, TCacheColumnStore));
		if (StromTagIs(gpuscan->rc_store, TCacheRowStore))
			tcache_put_row_store((tcache_row_store *)gpuscan->rc_store);
		else
			tcache_put_column_store((tcache_column_store *)gpuscan->rc_store);
	}
	pgstrom_shmem_free(gpuscan);
}
